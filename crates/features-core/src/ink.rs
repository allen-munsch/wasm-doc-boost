use alloc::vec::Vec;
use libm::{atan2, cos, sin};

/// Ink mask and run-length features for distinguishing text documents
/// from photographs and detecting page rotation.
///
/// Returns 9 features:
///   0. ink_coverage — fraction of pixels below Otsu threshold
///   1. ink_mean — mean grayscale of ink pixels
///   2. ink_run_h_mean — mean horizontal run length
///   3. ink_run_h_std — std of horizontal run lengths
///   4. ink_run_v_mean — mean vertical run length
///   5. ink_run_v_std — std of vertical run lengths
///   6. ink_run_h_entropy — entropy of horizontal run-length distribution
///   7. ink_run_v_entropy — entropy of vertical run-length distribution
///   8. docstrum_angle — dominant angle of inter-row ink-pixel nearest-neighbor links
pub fn ink_features(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let threshold = otsu_threshold(gray);
    let n = width * height;

    let mut ink_count = 0usize;
    let mut ink_sum = 0.0;

    let mut h_runs: Vec<usize> = Vec::new();
    let mut v_runs: Vec<usize> = Vec::new();

    // Horizontal runs
    for y in 0..height {
        let row_start = y * width;
        let mut run_len = 0usize;
        for x in 0..width {
            if gray[row_start + x] <= threshold {
                run_len += 1;
                ink_count += 1;
                ink_sum += gray[row_start + x];
            } else if run_len > 0 {
                h_runs.push(run_len);
                run_len = 0;
            }
        }
        if run_len > 0 {
            h_runs.push(run_len);
        }
    }

    // Vertical runs
    for x in 0..width {
        let mut run_len = 0usize;
        for y in 0..height {
            if gray[y * width + x] <= threshold {
                run_len += 1;
            } else if run_len > 0 {
                v_runs.push(run_len);
                run_len = 0;
            }
        }
        if run_len > 0 {
            v_runs.push(run_len);
        }
    }

    let coverage = ink_count as f64 / n as f64;
    let ink_mean = if ink_count > 0 {
        ink_sum / ink_count as f64
    } else {
        255.0
    };

    let (h_mean, h_std, h_entropy) = run_stats(&h_runs);
    let (v_mean, v_std, v_entropy) = run_stats(&v_runs);

    let docstrum = docstrum_angle(gray, width, height, threshold);

    alloc::vec![
        coverage, ink_mean, h_mean, h_std, v_mean, v_std, h_entropy, v_entropy, docstrum,
    ]
}

/// Docstrum-inspired rotation angle from inter-row nearest-neighbor links.
///
/// For each ink pixel, finds the nearest ink pixel in the row below and
/// records the angle. Returns the circular mean angle (in radians, [-π/2, π/2]).
///
/// Horizontal text: angles cluster near 0. Rotated text: angles cluster
/// near the rotation angle. Photos: angles are nearly uniform.
fn docstrum_angle(gray: &[f64], width: usize, height: usize, threshold: f64) -> f64 {
    let max_search = 20isize;
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut count = 0u32;

    for y in 0..height - 1 {
        let row_start = y * width;
        let next_row_start = (y + 1) * width;
        for x in 0..width {
            if gray[row_start + x] > threshold {
                continue;
            }
            let mut best_dx: Option<isize> = None;
            let mut best_dist = max_search;
            for dx in -max_search..=max_search {
                let nx = x as isize + dx;
                if nx < 0 || nx >= width as isize {
                    continue;
                }
                if gray[next_row_start + nx as usize] <= threshold {
                    let dist = dx.abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_dx = Some(dx);
                    }
                }
            }
            if let Some(dx) = best_dx {
                let angle = atan2(1.0, dx as f64);
                sum_sin += sin(2.0 * angle);
                sum_cos += cos(2.0 * angle);
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    // Return half the circular mean (we doubled the angle for variance)
    0.5 * atan2(sum_sin / count as f64, sum_cos / count as f64)
}

/// Unfolded docstrum: raw circular mean of inter-row nearest-ink link angles
/// plus its concentration.
///
/// Unlike `docstrum_angle` (inside `ink_features`, double-angle folded to
/// [-pi/2, pi/2]), this keeps the raw link direction in (0, pi): links always
/// point down (dy=+1), so the mean is pi/2 for horizontal text and shifts with
/// tilt. The concentration is the resultant length / count in [0, 1] — the
/// "pi-ambiguity" indicator: low concentration means the link angles are
/// scattered (photo or noisy page) and the axis vote should not be trusted.
///
/// Returns [mean_angle_rad, concentration]. Kept out of `extract_all` until
/// the Phase 3 retrain.
pub fn docstrum_axis(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let threshold = otsu_threshold(gray);
    let max_search = 20isize;
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut count = 0u32;

    for y in 0..height.saturating_sub(1) {
        let row_start = y * width;
        let next_row_start = (y + 1) * width;
        for x in 0..width {
            if gray[row_start + x] > threshold {
                continue;
            }
            let mut best_dx: Option<isize> = None;
            let mut best_dist = max_search;
            for dx in -max_search..=max_search {
                let nx = x as isize + dx;
                if nx < 0 || nx >= width as isize {
                    continue;
                }
                if gray[next_row_start + nx as usize] <= threshold {
                    let dist = dx.abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_dx = Some(dx);
                    }
                }
            }
            if let Some(dx) = best_dx {
                let angle = atan2(1.0, dx as f64);
                sum_sin += sin(angle);
                sum_cos += cos(angle);
                count += 1;
            }
        }
    }

    if count == 0 {
        return alloc::vec![0.0, 0.0];
    }
    let mean = atan2(sum_sin / count as f64, sum_cos / count as f64);
    let r = libm::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / count as f64;
    alloc::vec![mean, r.clamp(0.0, 1.0)]
}

fn run_stats(runs: &[usize]) -> (f64, f64, f64) {
    let n = runs.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mean = runs.iter().sum::<usize>() as f64 / n as f64;
    let variance = runs
        .iter()
        .map(|&r| {
            let d = r as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = libm::sqrt(variance);

    // Entropy: 16-bin histogram of run lengths 1..16+
    let n_bins = 16;
    let mut bins = [0u32; 16];
    for &r in runs {
        let bin = (r.min(n_bins) - 1).min(n_bins - 1);
        bins[bin] += 1;
    }
    let total = n as f64;
    let mut entropy = 0.0;
    for &c in &bins {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * libm::log(p);
        }
    }

    (mean, std, entropy)
}

pub fn otsu_threshold(values: &[f64]) -> f64 {
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max_val - min_val) < 1e-12 {
        return 128.0;
    }

    let n_bins = 64;
    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut hist = [0u32; 64];
    for &v in values {
        let bin = ((v - min_val) / bin_width) as usize;
        let bin = bin.min(n_bins - 1);
        hist[bin] += 1;
    }

    let total = values.len() as f64;
    let mut sum = 0.0;
    for (i, &h) in hist.iter().enumerate().take(n_bins) {
        sum += (min_val + (i as f64 + 0.5) * bin_width) * h as f64;
    }

    let mut best_thresh = min_val;
    let mut best_variance = 0.0;
    let mut w_b = 0.0;
    let mut sum_b = 0.0;
    for (i, &h) in hist.iter().enumerate().take(n_bins) {
        w_b += h as f64;
        if w_b < 1.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f < 1.0 {
            break;
        }
        sum_b += (min_val + (i as f64 + 0.5) * bin_width) * h as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best_variance {
            best_variance = between;
            best_thresh = min_val + (i as f64 + 1.0) * bin_width;
        }
    }

    best_thresh
}

/// Sauvola local threshold binarization: ink = pixel <= local mean-adjusted
/// threshold. A single global Otsu threshold fails on photographs with uneven
/// lighting (one half lit, one half shadowed); Sauvola adapts per pixel.
///
/// T = m * (1 + k * (s / R - 1)) over an odd window whose side is
/// min(w,h)/24 clamped to [7, 41]; integral images keep it O(n). k = 0.2,
/// R = 128 (8-bit gray dynamic range).
pub fn sauvola_binarize(gray: &[f64], width: usize, height: usize) -> Vec<bool> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let mut win = (width.min(height) / 24) | 1;
    if win < 7 {
        win = 7;
    } else if win > 41 {
        win = 41;
    }
    let half = win / 2;

    let sw = width + 1;
    let mut integral = alloc::vec![0.0f64; sw * (height + 1)];
    let mut integral_sq = alloc::vec![0.0f64; sw * (height + 1)];
    for y in 0..height {
        let row = y * width;
        let mut row_sum = 0.0f64;
        let mut row_sq = 0.0f64;
        for x in 0..width {
            let v = gray[row + x];
            row_sum += v;
            row_sq += v * v;
            let above = y * sw + (x + 1);
            integral[(y + 1) * sw + (x + 1)] = integral[above] + row_sum;
            integral_sq[(y + 1) * sw + (x + 1)] = integral_sq[above] + row_sq;
        }
    }

    let k = 0.2;
    let r = 128.0;
    let mut mask = alloc::vec![false; width * height];
    for y in 0..height {
        let y0 = y.saturating_sub(half);
        let y1 = (y + half + 1).min(height);
        let row = y * width;
        for x in 0..width {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half + 1).min(width);
            let sum = integral[y1 * sw + x1] - integral[y0 * sw + x1] - integral[y1 * sw + x0]
                + integral[y0 * sw + x0];
            let sq =
                integral_sq[y1 * sw + x1] - integral_sq[y0 * sw + x1] - integral_sq[y1 * sw + x0]
                    + integral_sq[y0 * sw + x0];
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            let mean = sum / n;
            let var = (sq / n) - mean * mean;
            let std = if var > 0.0 { libm::sqrt(var) } else { 0.0 };
            let thresh = mean * (1.0 + k * (std / r - 1.0));
            mask[row + x] = gray[row + x] <= thresh;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn solid_white() -> Vec<f64> {
        vec![255.0; 100]
    }

    fn solid_black() -> Vec<f64> {
        vec![0.0; 100]
    }

    fn checkerboard_8x8() -> Vec<f64> {
        let w = 8;
        let h = 8;
        let mut gray = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 2 == 0 {
                    gray.push(0.0);
                } else {
                    gray.push(255.0);
                }
            }
        }
        gray
    }

    #[test]
    fn ink_coverage_solid_white() {
        let f = ink_features(&solid_white(), 10, 10);
        assert!(f[0] < 0.01, "coverage={}", f[0]);
    }

    #[test]
    fn ink_coverage_solid_black() {
        let f = ink_features(&solid_black(), 10, 10);
        assert!(f[0] > 0.99, "coverage={}", f[0]);
    }

    #[test]
    fn ink_coverage_checkerboard() {
        let f = ink_features(&checkerboard_8x8(), 8, 8);
        assert!((f[0] - 0.5).abs() < 0.05, "coverage={}", f[0]);
    }

    #[test]
    fn ink_runs_checkerboard() {
        let f = ink_features(&checkerboard_8x8(), 8, 8);
        assert!(f[2] > 0.0, "h_mean={}", f[2]);
        assert!(f[2] < 2.0, "h_mean={}", f[2]);
    }

    #[test]
    fn feature_count() {
        let f = ink_features(&checkerboard_8x8(), 8, 8);
        assert_eq!(f.len(), 9);
    }

    #[test]
    fn docstrum_angle_horizontal_text() {
        // Simulated horizontal text: black rows with white gaps
        let w = 20;
        let h = 12;
        let mut gray = vec![255.0; w * h];
        for y in 0..h {
            if y % 3 == 0 {
                continue; // white gap
            }
            for x in 2..18 {
                gray[y * w + x] = 0.0;
            }
        }
        let f = ink_features(&gray, w, h);
        // Vertical nearest-neighbor links for horizontal text → angle ~±π/2
        assert!(
            (f[8].abs() - core::f64::consts::FRAC_PI_2).abs() < 0.3,
            "docstrum_angle={}",
            f[8]
        );
    }

    #[test]
    fn docstrum_axis_horizontal_text() {
        // Same simulated horizontal text: ink rows with white gaps. Unfolded
        // links point straight down -> mean angle ~pi/2, high concentration.
        let w = 20;
        let h = 12;
        let mut gray = vec![255.0; w * h];
        for y in 0..h {
            if y % 3 == 0 {
                continue;
            }
            for x in 2..18 {
                gray[y * w + x] = 0.0;
            }
        }
        let f = docstrum_axis(&gray, w, h);
        assert!(
            (f[0] - core::f64::consts::FRAC_PI_2).abs() < 0.4,
            "mean={}",
            f[0]
        );
        assert!(f[1] > 0.5, "concentration={}", f[1]);
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn sauvola_finds_text_on_uneven_background() {
        // Left half dark background (~80), right half bright (~200); text bars
        // are full black in both halves. A global Otsu would split by the two
        // background modes and lose one half; Sauvola keeps the ink everywhere.
        let w = 80usize;
        let h = 40usize;
        let mut gray = alloc::vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                gray[y * w + x] = if x < 40 { 80.0 } else { 200.0 };
            }
        }
        for y in 10..14 {
            for x in 0..w {
                gray[y * w + x] = 0.0;
            }
        }
        for y in 26..30 {
            for x in 0..w {
                gray[y * w + x] = 0.0;
            }
        }
        let mask = sauvola_binarize(&gray, w, h);
        let ink = |x: usize, y: usize| mask[y * w + x];
        // text ink in both halves
        for x in [5usize, 75usize] {
            assert!(ink(x, 12), "text ink at x={} y=12", x);
            assert!(ink(x, 28), "text ink at x={} y=28", x);
        }
        // background stays paper in both halves, away from text edges
        for x in [5usize, 75usize] {
            assert!(!ink(x, 2), "bg paper at x={} y=2", x);
            assert!(!ink(x, 36), "bg paper at x={} y=36", x);
        }
    }
}
