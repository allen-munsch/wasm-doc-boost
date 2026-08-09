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
        coverage,
        ink_mean,
        h_mean,
        h_std,
        v_mean,
        v_std,
        h_entropy,
        v_entropy,
        docstrum,
    ]
}

/// Docstrum-inspired rotation angle from inter-row nearest-neighbor links.
///
/// For each ink pixel, finds the nearest ink pixel in the row below and
/// records the angle. Returns the circular mean angle (in radians, [-π/2, π/2]).
///
/// Horizontal text: angles cluster near 0. Rotated text: angles cluster
/// near the rotation angle. Photos: angles are nearly uniform.
fn docstrum_angle(
    gray: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
) -> f64 {
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

fn otsu_threshold(values: &[f64]) -> f64 {
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
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
    for i in 0..n_bins {
        sum += (min_val + (i as f64 + 0.5) * bin_width) * hist[i] as f64;
    }

    let mut best_thresh = min_val;
    let mut best_variance = 0.0;
    let mut w_b = 0.0;
    let mut sum_b = 0.0;
    for i in 0..n_bins {
        w_b += hist[i] as f64;
        if w_b < 1.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f < 1.0 {
            break;
        }
        sum_b += (min_val + (i as f64 + 0.5) * bin_width) * hist[i] as f64;
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
}
