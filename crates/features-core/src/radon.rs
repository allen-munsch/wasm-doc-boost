//! Radon / projection-profile response for fine-axis orientation detection.
//!
//! For each angle in [0, 180) we rotate the grayscale image by -angle and
//! measure the variance of its horizontal projection (mean intensity per
//! row). Text lines run horizontally when that variance peaks, so
//! `argmax(curve)` estimates the text-line axis modulo 180 degrees. The
//! projection variance is exactly 180-periodic, so this signal carries the
//! axis (fine angle) but NOT the 0-vs-180 flip — the flip is the textline
//! ascender/descender head's job.
//!
//! no_std; nearest-neighbour sampling, no bilinear interpolation. Wired into
//! `extract_all` at the Phase 3 retrain step (kept out until then so the
//! existing 103-feature model contract stays loadable).

use alloc::vec;
use alloc::vec::Vec;

/// Number of angles sampled. 60 bins at 3 degrees over [0, 180).
pub const RADON_BINS: usize = 60;
/// Degrees per bin.
pub const RADON_DEG_PER_BIN: f64 = 3.0;

/// Projection-variance response curve over `RADON_BINS` angles in [0, 180).
///
/// `curve[i]` is the horizontal-projection variance of the image rotated by
/// `-i * 3` degrees. The peak bin is the text-line axis.
pub fn radon_curve(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    radon_curve_strided(gray, width, height, 1, 1)
}

/// Like `radon_curve`, but samples every `sx`-th output column and `sy`-th
/// output row. The projection variance is an aggregate over the canvas, so
/// mild strides cut the radon cost; x-only stride preserves the winner axis
/// while both-axes stride perturbs near-tie peaks (validated in
/// radon_downsample / winner_agree).
pub fn radon_curve_strided(
    gray: &[f64],
    width: usize,
    height: usize,
    sx: usize,
    sy: usize,
) -> Vec<f64> {
    if width.max(height) == 0 {
        return vec![0.0f64; RADON_BINS];
    }
    let (padded, size) = radon_pad(gray, width, height);
    let mut curve = Vec::with_capacity(RADON_BINS);
    for i in 0..RADON_BINS {
        let deg = i as f64 * RADON_DEG_PER_BIN;
        let g = angle_geom(size, deg, sx, sy);
        let rows = row_sums_at(&padded, size, &g);
        curve.push(variance_from_row_sums(&rows, g.ncols));
    }
    curve
}

/// Per-angle geometry constants for the projection-variance sampler.
///
/// `projection_variance_at` is decomposed into `angle_geom` + `row_sums_at`
/// + `variance_from_row_sums` so a GPU backend can run only the sampling loop
/// on the device while the host runs the same reduction, keeping the two
/// backends bit-identical.
#[derive(Clone, Copy)]
pub struct AngleGeom {
    pub cos_t: f64,
    pub sin_t: f64,
    pub cx: f64,
    pub cy: f64,
    pub last_x: f64,
    pub last_y: f64,
    pub step_x: f64,
    pub step_y: f64,
    pub ncols: usize,
    pub nrows: usize,
    pub sy: usize,
}

/// Geometry constants for one angle over a square `size`x`size` canvas,
/// sampled every `sx`-th output column and `sy`-th output row — identical
/// arithmetic to `projection_variance_at` (square callers only).
pub fn angle_geom(size: usize, deg: f64, sx: usize, sy: usize) -> AngleGeom {
    let sx = sx.max(1);
    let sy = sy.max(1);
    let rad = deg.to_radians();
    let sin_t = libm::sin(rad);
    let cos_t = libm::cos(rad);
    let cx = (size as f64 - 1.0) / 2.0;
    let cy = (size as f64 - 1.0) / 2.0;
    let last_x = size as f64 - 1.0;
    let last_y = size as f64 - 1.0;
    let step_x = cos_t * sx as f64;
    let step_y = sin_t * sx as f64;
    let nrows = (size + sy - 1) / sy;
    let ncols = (size + sx - 1) / sx;
    AngleGeom {
        cos_t,
        sin_t,
        cx,
        cy,
        last_x,
        last_y,
        step_x,
        step_y,
        ncols,
        nrows,
        sy,
    }
}

/// Row sums of the horizontal projection of a square `size`x`size` canvas
/// sampled at geometry `g` (nearest neighbour, in-order column accumulation —
/// the exact arithmetic of `projection_variance_at`).
pub fn row_sums_at(padded: &[f64], size: usize, g: &AngleGeom) -> Vec<f64> {
    let mut row_sums = vec![0.0f64; g.nrows];
    for (ry, slot) in row_sums.iter_mut().enumerate() {
        let y = ry * g.sy;
        let yc = y as f64 - g.cy;
        let mut xs = g.cx - g.cx * g.cos_t - yc * g.sin_t;
        let mut ys = g.cy - g.cx * g.sin_t + yc * g.cos_t;
        let mut row_sum = 0.0;
        for _ in 0..g.ncols {
            let xi = (xs + 0.5).clamp(0.0, g.last_x) as usize;
            let yi = (ys + 0.5).clamp(0.0, g.last_y) as usize;
            row_sum += padded[yi * size + xi];
            xs += g.step_x;
            ys += g.step_y;
        }
        *slot = row_sum;
    }
    row_sums
}

/// Variance of the horizontal projection given its per-row sums: the final
/// reduction of `projection_variance_at`, shared by the CPU and GPU paths.
pub fn variance_from_row_sums(row_sums: &[f64], ncols: usize) -> f64 {
    if row_sums.is_empty() || ncols == 0 {
        return 0.0;
    }
    let nrows = row_sums.len();
    let n = (nrows * ncols) as f64;
    let total: f64 = row_sums.iter().sum();
    let mean = total / n;
    let mut var = 0.0;
    for &row_sum in row_sums.iter() {
        let row_mean = row_sum / ncols as f64;
        let d = row_mean - mean;
        var += d * d;
    }
    var / nrows as f64
}

/// Square-pad with the border median (the document background), returning the
/// padded `size`x`size` buffer and `size`. Shared by the CPU curve and the
/// GPU backend so both sample the identical canvas.
pub fn radon_pad(gray: &[f64], width: usize, height: usize) -> (Vec<f64>, usize) {
    let size = width.max(height);
    if size == 0 {
        return (Vec::new(), 0);
    }
    let fill = border_median(gray, width, height);
    let mut padded = vec![fill; size * size];
    let ox = (size - width) / 2;
    let oy = (size - height) / 2;
    for y in 0..height {
        for x in 0..width {
            padded[(y + oy) * size + (x + ox)] = gray[y * width + x];
        }
    }
    (padded, size)
}

/// Median of the outer-frame pixels — the document's own background, used as
/// the square-padding fill so the padding does not inject edge structure.
fn border_median(gray: &[f64], width: usize, height: usize) -> f64 {
    if width == 0 || height == 0 {
        return 128.0;
    }
    let t = (width.min(height) / 40).max(1);
    let mut samples = Vec::new();
    for y in 0..t.min(height) {
        for x in 0..width {
            samples.push(gray[y * width + x]);
        }
    }
    for y in (height - t.min(height))..height {
        for x in 0..width {
            samples.push(gray[y * width + x]);
        }
    }
    for y in 0..height {
        for x in 0..t.min(width) {
            samples.push(gray[y * width + x]);
        }
        for x in (width - t.min(width))..width {
            samples.push(gray[y * width + x]);
        }
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Variance of the horizontal projection of a square image rotated by `-deg`,
/// sampling every `sx`-th output column and `sy`-th output row (1 = full).
// Legacy single-angle sampler, kept as the parity-test oracle: the test
// `helpers_match_projection_variance_at_bitwise` asserts the decomposed
// helpers reproduce it bit-for-bit.
#[allow(dead_code)]
fn projection_variance_at(
    gray: &[f64],
    width: usize,
    height: usize,
    deg: f64,
    sx: usize,
    sy: usize,
) -> f64 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let sx = sx.max(1);
    let sy = sy.max(1);
    let rad = deg.to_radians();
    let sin_t = libm::sin(rad);
    let cos_t = libm::cos(rad);
    let cx = (width as f64 - 1.0) / 2.0;
    let cy = (height as f64 - 1.0) / 2.0;
    let nrows = (height + sy - 1) / sy;
    let ncols = (width + sx - 1) / sx;
    let n = (nrows * ncols) as f64;

    let mut row_sums = vec![0.0f64; nrows];
    let mut total = 0.0;
    let last_x = width as f64 - 1.0;
    let last_y = height as f64 - 1.0;
    let step_x = cos_t * sx as f64;
    let step_y = sin_t * sx as f64;

    for (ry, slot) in row_sums.iter_mut().enumerate() {
        let y = ry * sy;
        let yc = y as f64 - cy;
        // Output point (x, y) back-rotated into the source: rotating the
        // image by -deg samples the source at rotate(+deg) of the output
        // point. Nearest neighbour, clamped to the border. xs/ys are linear
        // in x, so stride by incrementing step_x/step_y per sampled column.
        let mut xs = cx - cx * cos_t - yc * sin_t;
        let mut ys = cy - cx * sin_t + yc * cos_t;
        let mut row_sum = 0.0;
        for _ in 0..ncols {
            // Nearest neighbour: shift by 0.5 then clamp-then-cast, the
            // core-only pattern used across features-core (f64::round and
            // f64::floor are std-gated on this toolchain).
            let xi = (xs + 0.5).clamp(0.0, last_x) as usize;
            let yi = (ys + 0.5).clamp(0.0, last_y) as usize;
            row_sum += gray[yi * width + xi];
            xs += step_x;
            ys += step_y;
        }
        *slot = row_sum;
        total += row_sum;
    }

    let mean = total / n;
    let mut var = 0.0;
    for &row_sum in row_sums.iter() {
        let row_mean = row_sum / ncols as f64;
        let d = row_mean - mean;
        var += d * d;
    }
    var / nrows as f64
}

/// Index and value of the maximum of a slice.
pub fn argmax(curve: &[f64]) -> (usize, f64) {
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in curve.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    (best, best_val)
}

/// Parabolic refinement of the peak at bin `i`, returned in bins.
///
/// Fits a parabola through curve[i-1], curve[i], curve[i+1] and returns the
/// vertex position (within 0.5 bins of `i`). The curve is 180-periodic, so
/// the neighbours wrap: bin 0 refines against the last bin and bin 1, bin 59
/// against bin 58 and bin 0. Refusing to wrap here (returning the unrefined
/// bin) left a ~1.5-degree quantization error for axes near the 0/180 fold.
pub fn parabolic_refine(curve: &[f64], i: usize) -> f64 {
    if curve.len() < 3 {
        return i as f64;
    }
    let n = curve.len();
    let y0 = curve[(i + n - 1) % n];
    let y1 = curve[i];
    let y2 = curve[(i + 1) % n];
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-12 {
        return i as f64;
    }
    let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
    i as f64 + delta
}

/// Top-k local maxima of the 180-periodic curve, parabolically refined,
/// returned as axis angles in [0, pi) radians, highest curve value first.
///
/// When the text axis and a perpendicular structure (table rules, page
/// edges) produce a near-tie, the global argmax can land on the wrong peak.
/// Returning the top local maxima lets the caller score each candidate by
/// text-line quality and pick the one that actually deskews text lines.
pub fn radon_top_axes(curve: &[f64], k: usize) -> Vec<f64> {
    let n = curve.len();
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    for i in 0..n {
        let prev = curve[(i + n - 1) % n];
        let next = curve[(i + 1) % n];
        let v = curve[i];
        if v >= prev && v >= next {
            maxima.push((i, v));
        }
    }
    maxima.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut axes = Vec::with_capacity(k);
    for &(bin, _) in maxima.iter().take(k) {
        let refined = parabolic_refine(curve, bin);
        let mut deg = refined * RADON_DEG_PER_BIN;
        while deg < 0.0 {
            deg += 180.0;
        }
        while deg >= 180.0 {
            deg -= 180.0;
        }
        axes.push(deg.to_radians());
    }
    axes
}

/// Curve summary for the trust head: `(peakiness, peak_margin, concentration)`
/// computed from the curve's global argmax. Peakiness and peak_margin measure
/// how dominant the strongest peak is (low = near-tie, e.g. a perpendicular
/// structure competing with the text axis); concentration is the circular
/// (double-angle) resultant length of the curve treated as a distribution over
/// [0, pi), in [0, 1].
pub fn radon_summary(curve: &[f64]) -> (f64, f64, f64) {
    let (imax, peak) = argmax(curve);
    let mean = curve.iter().sum::<f64>() / curve.len() as f64;
    let peakiness = if mean > 1e-9 { peak / mean } else { 0.0 };

    // Strongest peak outside the argmax's immediate neighbourhood.
    let mut second = 0.0f64;
    for (i, &v) in curve.iter().enumerate() {
        if (i as isize - imax as isize).abs() > 1 && v > second {
            second = v;
        }
    }
    let peak_margin = peak - second;

    // Circular concentration over [0, pi): double the angle, weighted by the
    // curve values, then the resultant length / total weight.
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut total_w = 0.0;
    for (i, &v) in curve.iter().enumerate() {
        let ang = 2.0 * (i as f64 * RADON_DEG_PER_BIN).to_radians();
        sum_sin += v * libm::sin(ang);
        sum_cos += v * libm::cos(ang);
        total_w += v;
    }
    let concentration = if total_w > 1e-9 {
        let r = libm::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / total_w;
        r.clamp(0.0, 1.0)
    } else {
        0.0
    };

    (peakiness, peak_margin, concentration)
}

/// Summary features for the GBDT: the deterministic axis estimate plus how
/// much we should trust it.
///
/// Returns `[best_axis_rad, peakiness, peak_margin, concentration]`:
/// - best_axis_rad: parabolically refined argmax in [0, pi).
/// - peakiness: peak / mean of the curve (high = clear dominant axis).
/// - peak_margin: peak minus the strongest non-adjacent peak.
/// - concentration: circular concentration of the curve treated as a
///   distribution over [0, pi) (double-angle), in [0, 1].
///
/// Fine-stage step and half-window for the multistage deskew (S2). The
/// coarse 60-bin radon resolves the axis to ~3 deg; a deskew step wants
/// ~0.1 deg, so re-measure the projection variance of an Otsu ink mask at
/// these fine steps around the coarse axis (21 evaluations).
pub const FINE_STEP_DEG: f64 = 0.15;
pub const FINE_HALF_WINDOW_DEG: f64 = 1.5;

/// Fine axis refinement (S2): Otsu ink mask + projection-variance search.
///
/// Builds a binary ink mask (below-Otsu = ink, matching `deskew.py`) and
/// re-measures the horizontal-projection variance of that mask at
/// `FINE_STEP_DEG` over `coarse_axis ± FINE_HALF_WINDOW_DEG`, then
/// parabolic-refines the peak. Using the ink mask (not grayscale) ignores
/// background gradients and shadows. Returns the refined axis in radians
/// (mod pi).
pub fn fine_axis(gray: &[f64], width: usize, height: usize, coarse_axis_rad: f64) -> f64 {
    if width == 0 || height == 0 {
        return coarse_axis_rad;
    }
    let thr = crate::ink::otsu_threshold(gray);
    // Square-pad with background (0.0) so small rotations stay in-canvas.
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
    // ~blank page: no text lines to align, so don't refine on noise
    // (deskew.py abstains below 1% foreground).
    if (ink as f64) < 0.004 * (size * size) as f64 {
        return coarse_axis_rad;
    }

    let coarse_deg = coarse_axis_rad.to_degrees() % 180.0;
    let n = ((2.0 * FINE_HALF_WINDOW_DEG / FINE_STEP_DEG + 0.5) as usize) + 1;
    let mut curve = Vec::with_capacity(n);
    for i in 0..n {
        let deg = coarse_deg - FINE_HALF_WINDOW_DEG + i as f64 * FINE_STEP_DEG;
        let g = angle_geom(size, deg, 2, 2);
        let rows = row_sums_at(&mask, size, &g);
        curve.push(variance_from_row_sums(&rows, g.ncols));
    }
    let (imax, _) = argmax(&curve);
    let refined_bin = parabolic_refine(&curve, imax);
    let mut refined_deg = (coarse_deg - FINE_HALF_WINDOW_DEG + refined_bin * FINE_STEP_DEG) % 180.0;
    if refined_deg < 0.0 {
        refined_deg += 180.0;
    }
    refined_deg.to_radians()
}

/// This is the single-peak (argmax-only) axis used by the standalone/bench
/// entry points; `rotation_features` prefers the multi-candidate selection
/// that scores the top local maxima by text-line quality.
pub fn radon_features(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let curve = radon_curve(gray, width, height);
    let (imax, _) = argmax(&curve);
    let refined = parabolic_refine(&curve, imax);
    let mut best_axis_deg = refined * RADON_DEG_PER_BIN;
    while best_axis_deg < 0.0 {
        best_axis_deg += 180.0;
    }
    while best_axis_deg >= 180.0 {
        best_axis_deg -= 180.0;
    }
    let best_axis_rad = best_axis_deg.to_radians();
    let (peakiness, peak_margin, concentration) = radon_summary(&curve);

    alloc::vec![best_axis_rad, peakiness, peak_margin, concentration]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stripes whose lines make angle `deg` (CCW from horizontal): level sets
    /// of -x*sin(theta) + y*cos(theta).
    fn striped(deg: f64, w: usize, h: usize, period: f64) -> Vec<f64> {
        let theta = deg.to_radians();
        let (sin_t, cos_t) = (libm::sin(theta), libm::cos(theta));
        let mut gray = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let phase = -(x as f64) * sin_t + (y as f64) * cos_t;
                let v = if (phase / period).rem_euclid(2.0) < 1.0 {
                    0.0
                } else {
                    255.0
                };
                gray[y * w + x] = v;
            }
        }
        gray
    }

    #[test]
    fn test_radon_argmax_horizontal_stripes() {
        let gray = striped(0.0, 128, 128, 8.0);
        let curve = radon_curve(&gray, 128, 128);
        let (imax, _) = argmax(&curve);
        // Horizontal stripes: peak at 0 deg (bin 0); bin 59 (177 deg) is the
        // wrapped neighbour and may tie, so accept either end.
        assert!(imax == 0 || imax == RADON_BINS - 1, "imax={imax}");
    }

    #[test]
    fn test_radon_argmax_17_deg() {
        let gray = striped(17.0, 128, 128, 8.0);
        let curve = radon_curve(&gray, 128, 128);
        let (imax, _) = argmax(&curve);
        let refined = parabolic_refine(&curve, imax);
        let best_axis_deg = (refined * RADON_DEG_PER_BIN).rem_euclid(180.0);
        assert!(
            (best_axis_deg - 17.0).abs() < 1.5,
            "best_axis_deg={best_axis_deg} imax={imax}"
        );
    }

    #[test]
    fn test_fine_axis_refines() {
        for &deg in &[4.7, 17.3, 42.8, 89.1] {
            let gray = striped(deg, 256, 256, 10.0);
            let curve = radon_curve(&gray, 256, 256);
            let (imax, _) = argmax(&curve);
            let coarse_rad = (parabolic_refine(&curve, imax) * RADON_DEG_PER_BIN).to_radians();
            let fine = fine_axis(&gray, 256, 256, coarse_rad)
                .to_degrees()
                .rem_euclid(180.0);
            assert!((fine - deg).abs() < 0.35, "deg={deg} fine={fine}");
        }
    }

    #[test]
    fn test_radon_180_invariant() {
        // Rotating the SAME image by 180 deg must leave the curve (nearly)
        // unchanged: a 180 deg rotation maps each line onto itself and just
        // permutes the rows, so the axis signal is flip-blind up to the small
        // border-clamp artifact of the crop+clamp sampling (~1%). That
        // blindness is the contract that forces the flip decision onto the
        // textline head.
        let a = striped(17.0, 96, 96, 8.0);
        let mut b = vec![0.0f64; 96 * 96];
        for y in 0..96 {
            for x in 0..96 {
                b[y * 96 + x] = a[(95 - y) * 96 + (95 - x)];
            }
        }
        let ca = radon_curve(&a, 96, 96);
        let cb = radon_curve(&b, 96, 96);
        for i in 0..RADON_BINS {
            assert!(
                (ca[i] - cb[i]).abs() < 0.03 * (1.0 + ca[i].abs()),
                "bin {i}: {} vs {}",
                ca[i],
                cb[i]
            );
        }
    }

    #[test]
    fn test_radon_features_shape_and_peakiness() {
        let gray = striped(17.0, 96, 96, 8.0);
        let f = radon_features(&gray, 96, 96);
        assert_eq!(f.len(), 4);
        assert!(f[0] >= 0.0 && f[0] < core::f64::consts::PI, "axis={}", f[0]);
        // A crisp stripe pattern has a dominant axis: peakiness well above 1.
        assert!(f[1] > 1.5, "peakiness={}", f[1]);
    }

    #[test]
    fn test_radon_blank_image() {
        let gray = vec![128.0f64; 64 * 64];
        let curve = radon_curve(&gray, 64, 64);
        for v in &curve {
            assert!(v.abs() < 1e-9, "blank curve value {v}");
        }
        let f = radon_features(&gray, 64, 64);
        assert!(f[1].abs() < 1e-12 && f[3].abs() < 1e-12);
    }

    #[test]
    fn test_parabolic_refine_wraps_at_fold() {
        // A peak at bin 0 with curve[59] > curve[1] means the true peak sits
        // on the negative side of the 0/180 fold (near 179 deg); refinement
        // must move toward bin 59, not return the unrefined bin 0.
        let mut curve = vec![10.0f64; RADON_BINS];
        curve[59] = 90.0;
        curve[0] = 100.0;
        curve[1] = 80.0;
        let refined = parabolic_refine(&curve, 0);
        assert!(refined < 0.0, "refined={refined} should wrap toward bin 59");
    }

    #[test]
    fn test_radon_top_axes_returns_refined_peaks() {
        let gray = striped(17.0, 128, 128, 8.0);
        let curve = radon_curve(&gray, 128, 128);
        let axes = radon_top_axes(&curve, 4);
        assert!(!axes.is_empty());
        let top_deg = axes[0].to_degrees().rem_euclid(180.0);
        assert!((top_deg - 17.0).abs() < 1.5, "top_deg={top_deg}");
        for ax in &axes {
            assert!(*ax >= 0.0 && *ax < core::f64::consts::PI, "axis={ax}");
        }
    }

    #[test]
    fn helpers_match_projection_variance_at_bitwise() {
        let size = 64usize;
        let mut gray = vec![0.0f64; size * size];
        for y in 0..size {
            for x in 0..size {
                gray[y * size + x] = ((x * 7 + y * 13) % 256) as f64;
            }
        }
        for deg in [0.0, 3.0, 17.3, 45.0, 89.7, 123.456, 179.9] {
            for (sx, sy) in [(1usize, 1usize), (2, 2), (3, 1)] {
                let a = projection_variance_at(&gray, size, size, deg, sx, sy);
                let g = angle_geom(size, deg, sx, sy);
                let rows = row_sums_at(&gray, size, &g);
                let b = variance_from_row_sums(&rows, g.ncols);
                assert_eq!(a.to_bits(), b.to_bits(), "deg={deg} sx={sx} sy={sy}");
            }
        }
    }
}
