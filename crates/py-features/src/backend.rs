//! Radon backend selection: CPU / CUDA / WebGPU with graceful degradation.
//!
//! `PY_FEATURES_BACKEND=cpu|cuda|webgpu|auto` (case-insensitive). `auto`
//! picks CUDA when a device is reachable, then CPU. WebGPU is the reserved
//! slot for the wasm-bridge work; on native builds it falls back to CPU until
//! that backend lands, so every choice still yields a working extractor.

use features_core::{FineAxisFn, RadonCurveFn};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Cpu,
    Cuda,
    Wgpu,
}

pub fn selected() -> Backend {
    let v = std::env::var("PY_FEATURES_BACKEND")
        .ok()
        .map(|s| s.to_ascii_lowercase());
    match v.as_deref() {
        Some("cpu") => Backend::Cpu,
        Some("cuda") => Backend::Cuda,
        Some("wgpu") | Some("webgpu") => Backend::Wgpu,
        _ => {
            #[cfg(feature = "gpu")]
            {
                if crate::cuda::available() {
                    return Backend::Cuda;
                }
            }
            Backend::Cpu
        }
    }
}

pub fn radon_pair(b: Backend) -> (RadonCurveFn, FineAxisFn) {
    match b {
        Backend::Cpu => (
            features_core::radon::radon_curve,
            features_core::radon::fine_axis,
        ),
        #[cfg(feature = "gpu")]
        Backend::Cuda => (crate::cuda::radon_curve_cuda, crate::cuda::fine_axis_cuda),
        #[cfg(not(feature = "gpu"))]
        Backend::Cuda => (
            features_core::radon::radon_curve,
            features_core::radon::fine_axis,
        ),
        Backend::Wgpu => (
            // WebGPU backend lands with the wasm-bridge work; native falls
            // back to CPU so the choice always yields a working backend.
            features_core::radon::radon_curve,
            features_core::radon::fine_axis,
        ),
    }
}

/// What will actually run once availability is accounted for (CUDA that
/// cannot init degrades to CPU, as does the not-yet-landed WebGPU backend).
pub fn effective() -> Backend {
    match selected() {
        Backend::Cuda => {
            #[cfg(feature = "gpu")]
            {
                if crate::cuda::available() {
                    return Backend::Cuda;
                }
            }
            Backend::Cpu
        }
        Backend::Wgpu => Backend::Cpu,
        b => b,
    }
}

pub fn name(b: Backend) -> &'static str {
    match b {
        Backend::Cpu => "cpu",
        Backend::Cuda => "cuda",
        Backend::Wgpu => "webgpu",
    }
}
