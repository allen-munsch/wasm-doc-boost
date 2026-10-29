//! CUDA backend for the radon projection-variance sampler (native only).
//!
//! `libcuda.so.1` is dlopen'd at runtime, so a CPU-only build still imports
//! on machines with no NVIDIA driver. Parity with `features-core` is
//! bit-exact: the host computes the per-angle geometry with libm and the
//! final variance reduction; the device only runs the nearest-neighbour
//! sampling + in-order f64 row accumulation.

use std::ffi::{c_char, c_void, CString};
use std::sync::OnceLock;

use features_core::radon::{
    angle_geom, argmax, parabolic_refine, radon_pad, variance_from_row_sums, AngleGeom,
    FINE_HALF_WINDOW_DEG, FINE_STEP_DEG, RADON_BINS, RADON_DEG_PER_BIN,
};
use libloading::Library;

const PTX: &str = include_str!("../kernels/radon.ptx");

const CUDA_SUCCESS: i32 = 0;

// CUDA driver API opaque types.
type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;

#[derive(Clone, Copy)]
struct CudaApi {
    mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    launch: unsafe extern "C" fn(
        CUfunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    sync: unsafe extern "C" fn() -> CUresult,
}

impl CudaApi {
    unsafe fn mem_alloc(&self, p: *mut CUdeviceptr, n: usize) -> CUresult {
        (self.mem_alloc)(p, n)
    }
    unsafe fn mem_free(&self, d: CUdeviceptr) -> CUresult {
        (self.mem_free)(d)
    }
    unsafe fn memcpy_htod(&self, d: CUdeviceptr, s: *const c_void, n: usize) -> CUresult {
        (self.memcpy_htod)(d, s, n)
    }
    unsafe fn memcpy_dtoh(&self, d: *mut c_void, s: CUdeviceptr, n: usize) -> CUresult {
        (self.memcpy_dtoh)(d, s, n)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        f: CUfunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        sh: u32,
        stream: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult {
        (self.launch)(f, gx, gy, gz, bx, by, bz, sh, stream, params, extra)
    }
    unsafe fn sync(&self) -> CUresult {
        (self.sync)()
    }
}

struct Cuda {
    // Keeps the library mapped for the lifetime of the device.
    _lib: Library,
    _ctx: CUcontext,
    radon_rows: CUfunction,
    api: CudaApi,
}

// The device is created once and the raw handles are never mutated; CUDA
// contexts are usable from any thread that makes them current, and ours stays
// current on the creating thread.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

static CUDA: OnceLock<Option<Cuda>> = OnceLock::new();

/// Load a symbol and copy it out as an owned fn pointer immediately, so no
/// borrow of `lib` survives past this call.
fn load<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    unsafe { lib.get::<T>(name).ok().map(|s| *s) }
}

fn init() -> Option<&'static Cuda> {
    CUDA.get_or_init(|| unsafe {
        let lib = Library::new("libcuda.so.1").ok()?;
        let cu_init: unsafe extern "C" fn(u32) -> CUresult = load(&lib, b"cuInit\0")?;
        let cu_device_get: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult =
            load(&lib, b"cuDeviceGet\0")?;
        let cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult =
            load(&lib, b"cuCtxCreate_v2\0")?;
        let cu_module_load: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult =
            load(&lib, b"cuModuleLoadData\0")?;
        let cu_module_get_fn: unsafe extern "C" fn(
            *mut CUfunction,
            CUmodule,
            *const c_char,
        ) -> CUresult = load(&lib, b"cuModuleGetFunction\0")?;
        let mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult =
            load(&lib, b"cuMemAlloc_v2\0")?;
        let mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult =
            load(&lib, b"cuMemFree_v2\0")?;
        let memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult =
            load(&lib, b"cuMemcpyHtoD_v2\0")?;
        let memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult =
            load(&lib, b"cuMemcpyDtoH_v2\0")?;
        let launch: unsafe extern "C" fn(
            CUfunction,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            CUstream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult = load(&lib, b"cuLaunchKernel\0")?;
        let sync: unsafe extern "C" fn() -> CUresult = load(&lib, b"cuCtxSynchronize\0")?;

        if cu_init(0) != CUDA_SUCCESS {
            return None;
        }
        let mut dev: CUdevice = 0;
        if cu_device_get(&mut dev, 0) != CUDA_SUCCESS {
            return None;
        }
        let mut ctx: CUcontext = std::ptr::null_mut();
        if cu_ctx_create(&mut ctx, 0, dev) != CUDA_SUCCESS {
            return None;
        }
        let mut module: CUmodule = std::ptr::null_mut();
        let ptx = CString::new(PTX).ok()?;
        if cu_module_load(&mut module, ptx.as_ptr() as *const c_void) != CUDA_SUCCESS {
            return None;
        }
        let mut func: CUfunction = std::ptr::null_mut();
        let name = CString::new("radon_rows").ok()?;
        if cu_module_get_fn(&mut func, module, name.as_ptr()) != CUDA_SUCCESS {
            return None;
        }

        Some(Cuda {
            _lib: lib,
            _ctx: ctx,
            radon_rows: func,
            api: CudaApi {
                mem_alloc,
                mem_free,
                memcpy_htod,
                memcpy_dtoh,
                launch,
                sync,
            },
        })
    })
    .as_ref()
}

/// True when a CUDA device is reachable (used by the `auto` backend).
pub fn available() -> bool {
    init().is_some()
}

fn check(r: CUresult) -> Option<()> {
    if r == CUDA_SUCCESS {
        Some(())
    } else {
        None
    }
}

/// Run the sampling loop for a batch of angles on the device and return the
/// per-angle row sums (host still does the variance reduction).
fn row_sums_batch(padded: &[f64], size: usize, geoms: &[AngleGeom]) -> Option<Vec<f64>> {
    let cuda = init()?;
    let n = geoms.len();
    if n == 0 || size == 0 {
        return Some(Vec::new());
    }

    let mut d_angle = Vec::with_capacity(n * 8);
    let mut i_angle = Vec::with_capacity(n * 3);
    let mut row_off = Vec::with_capacity(n + 1);
    let mut total_rows = 0usize;
    let mut max_nrows = 0usize;
    for g in geoms {
        d_angle.extend_from_slice(&[
            g.cos_t, g.sin_t, g.cx, g.cy, g.last_x, g.last_y, g.step_x, g.step_y,
        ]);
        // Kernel reads gi[0]=nrows, gi[1]=ncols, gi[2]=sy.
        i_angle.extend_from_slice(&[g.nrows as i32, g.ncols as i32, g.sy as i32]);
        row_off.push(total_rows as i32);
        total_rows += g.nrows;
        max_nrows = max_nrows.max(g.nrows);
    }
    row_off.push(total_rows as i32);
    if total_rows == 0 {
        return Some(Vec::new());
    }

    unsafe {
        let mut d_gray: CUdeviceptr = 0;
        let mut d_dangle: CUdeviceptr = 0;
        let mut d_iangle: CUdeviceptr = 0;
        let mut d_rowsums: CUdeviceptr = 0;
        let mut d_rowoff: CUdeviceptr = 0;
        check(cuda.api.mem_alloc(&mut d_gray, size * size * 8))?;
        check(cuda.api.mem_alloc(&mut d_dangle, n * 8 * 8))?;
        check(cuda.api.mem_alloc(&mut d_iangle, n * 3 * 4))?;
        check(cuda.api.mem_alloc(&mut d_rowsums, total_rows * 8))?;
        check(cuda.api.mem_alloc(&mut d_rowoff, (n + 1) * 4))?;

        check(
            cuda.api
                .memcpy_htod(d_gray, padded.as_ptr() as *const c_void, size * size * 8),
        )?;
        check(
            cuda.api
                .memcpy_htod(d_dangle, d_angle.as_ptr() as *const c_void, n * 8 * 8),
        )?;
        check(
            cuda.api
                .memcpy_htod(d_iangle, i_angle.as_ptr() as *const c_void, n * 3 * 4),
        )?;
        check(
            cuda.api
                .memcpy_htod(d_rowoff, row_off.as_ptr() as *const c_void, (n + 1) * 4),
        )?;

        let grid_x = ((max_nrows as u32) + 255) / 256;
        let mut size_i = size as i32;
        let mut p: [*mut c_void; 6] = [
            &mut d_gray as *mut CUdeviceptr as *mut c_void,
            &mut size_i as *mut i32 as *mut c_void,
            &mut d_dangle as *mut CUdeviceptr as *mut c_void,
            &mut d_iangle as *mut CUdeviceptr as *mut c_void,
            &mut d_rowsums as *mut CUdeviceptr as *mut c_void,
            &mut d_rowoff as *mut CUdeviceptr as *mut c_void,
        ];
        check(cuda.api.launch(
            cuda.radon_rows,
            grid_x,
            n as u32,
            1,
            256,
            1,
            1,
            0,
            std::ptr::null_mut(),
            p.as_mut_ptr(),
            std::ptr::null_mut(),
        ))?;
        check(cuda.api.sync())?;

        let mut row_sums = vec![0.0f64; total_rows];
        check(cuda.api.memcpy_dtoh(
            row_sums.as_mut_ptr() as *mut c_void,
            d_rowsums,
            total_rows * 8,
        ))?;

        cuda.api.mem_free(d_gray);
        cuda.api.mem_free(d_dangle);
        cuda.api.mem_free(d_iangle);
        cuda.api.mem_free(d_rowsums);
        cuda.api.mem_free(d_rowoff);
        Some(row_sums)
    }
}

/// GPU radon curve, bit-identical to `features_core::radon::radon_curve`;
/// falls back to the CPU implementation if the device is unavailable.
pub fn radon_curve_cuda(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    if width.max(height) == 0 {
        return vec![0.0f64; RADON_BINS];
    }
    let (padded, size) = radon_pad(gray, width, height);
    let geoms: Vec<AngleGeom> = (0..RADON_BINS)
        .map(|i| angle_geom(size, i as f64 * RADON_DEG_PER_BIN, 1, 1))
        .collect();
    let rows = match row_sums_batch(&padded, size, &geoms) {
        Some(r) => r,
        None => return features_core::radon::radon_curve(gray, width, height),
    };
    let mut curve = Vec::with_capacity(RADON_BINS);
    let mut off = 0usize;
    for g in &geoms {
        curve.push(variance_from_row_sums(&rows[off..off + g.nrows], g.ncols));
        off += g.nrows;
    }
    curve
}

/// GPU fine axis, bit-identical to `features_core::radon::fine_axis`; falls
/// back to the CPU implementation if the device is unavailable.
pub fn fine_axis_cuda(gray: &[f64], width: usize, height: usize, coarse_axis_rad: f64) -> f64 {
    if width == 0 || height == 0 {
        return coarse_axis_rad;
    }
    let thr = features_core::ink::otsu_threshold(gray);
    let size = width.max(height);
    let mut mask = vec![0.0f64; size * size];
    let ox = (size - width) / 2;
    let oy = (size - height) / 2;
    let mut ink = 0usize;
    for y in 0..height {
        for x in 0..width {
            if gray[y * width + x] <= thr {
                mask[(y + oy) * size + (x + ox)] = 1.0;
                ink += 1;
            }
        }
    }
    if (ink as f64) < 0.004 * (size * size) as f64 {
        return coarse_axis_rad;
    }

    let coarse_deg = coarse_axis_rad.to_degrees() % 180.0;
    let n = ((2.0 * FINE_HALF_WINDOW_DEG / FINE_STEP_DEG + 0.5) as usize) + 1;
    let geoms: Vec<AngleGeom> = (0..n)
        .map(|i| {
            let deg = coarse_deg - FINE_HALF_WINDOW_DEG + i as f64 * FINE_STEP_DEG;
            angle_geom(size, deg, 2, 2)
        })
        .collect();
    let rows = match row_sums_batch(&mask, size, &geoms) {
        Some(r) => r,
        None => return features_core::radon::fine_axis(gray, width, height, coarse_axis_rad),
    };
    let mut curve = Vec::with_capacity(n);
    let mut off = 0usize;
    for g in &geoms {
        curve.push(variance_from_row_sums(&rows[off..off + g.nrows], g.ncols));
        off += g.nrows;
    }
    let (imax, _) = argmax(&curve);
    let refined_bin = parabolic_refine(&curve, imax);
    let mut refined_deg = (coarse_deg - FINE_HALF_WINDOW_DEG + refined_bin * FINE_STEP_DEG) % 180.0;
    if refined_deg < 0.0 {
        refined_deg += 180.0;
    }
    refined_deg.to_radians()
}
