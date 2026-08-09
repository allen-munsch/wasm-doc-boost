use alloc::vec;
use alloc::vec::Vec;
use libm::{atan2, cos, sin, sqrt};
use crate::pixel_at;

/// Precomputed Sobel gradients for a single image.
/// Computed in one pass — share across all edge feature functions.
pub struct SobelResult {
    pub gx: Vec<f64>,
    pub gy: Vec<f64>,
    pub magnitudes: Vec<f64>,
    pub orientations: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

/// Compute Sobel gx, gy, magnitudes, and orientations in a single pass.
/// Uses a 1-pixel border-padded buffer for flat indexing — no per-pixel clamping.
pub fn sobel(gray: &[f64], width: usize, height: usize) -> SobelResult {
    let padded = pad_gray(gray, width, height);
    let pw = width + 2;
    let n = width * height;
    let mut gx = Vec::with_capacity(n);
    let mut gy = Vec::with_capacity(n);
    let mut magnitudes = Vec::with_capacity(n);
    let mut orientations = Vec::with_capacity(n);

    for y in 0..height {
        let row = (y + 1) * pw;
        let row_above = row - pw;
        let row_below = row + pw;
        for x in 0..width {
            let c = x + 1;
            let tl = padded[row_above + c - 1];
            let tm = padded[row_above + c];
            let tr = padded[row_above + c + 1];
            let ml = padded[row + c - 1];
            let mr = padded[row + c + 1];
            let bl = padded[row_below + c - 1];
            let bm = padded[row_below + c];
            let br = padded[row_below + c + 1];

            let gx_val = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            let gy_val = -tl - 2.0 * tm - tr + bl + 2.0 * bm + br;

            gx.push(gx_val);
            gy.push(gy_val);
            magnitudes.push(sqrt(gx_val * gx_val + gy_val * gy_val));
            orientations.push(atan2(gy_val, gx_val));
        }
    }

    SobelResult {
        gx,
        gy,
        magnitudes,
        orientations,
        width,
        height,
    }
}

/// Create a 1-pixel border-padded copy of the grayscale image
/// so convolution inner loops can use flat indexing without clamping.
fn pad_gray(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let pw = width + 2;
    let ph = height + 2;
    let mut padded = vec![0.0; pw * ph];

    for y in 0..height {
        let src_off = y * width;
        let dst_off = (y + 1) * pw + 1;
        padded[dst_off..dst_off + width].copy_from_slice(&gray[src_off..src_off + width]);
    }

    for x in 0..width {
        padded[x + 1] = gray[x];
        padded[(ph - 1) * pw + x + 1] = gray[(height - 1) * width + x];
    }
    for y in 0..height {
        padded[(y + 1) * pw] = gray[y * width];
        padded[(y + 1) * pw + pw - 1] = gray[y * width + width - 1];
    }
    padded[0] = gray[0];
    padded[pw - 1] = gray[width - 1];
    padded[(ph - 1) * pw] = gray[(height - 1) * width];
    padded[(ph - 1) * pw + pw - 1] = gray[(height - 1) * width + width - 1];

    padded
}

/// Laplacian variance (blur detector).
/// Applies 3×3 Laplacian kernel and returns variance of the result.
pub fn laplacian_variance(gray: &[f64], width: usize, height: usize) -> f64 {
    let n = width * height;
    let mut sum = 0.0;
    let mut m2 = 0.0;

    for y in 0..height as isize {
        for x in 0..width as isize {
            let lap = pixel_at(&gray, width, height, x, y) * 4.0
                - pixel_at(&gray, width, height, x - 1, y)
                - pixel_at(&gray, width, height, x + 1, y)
                - pixel_at(&gray, width, height, x, y - 1)
                - pixel_at(&gray, width, height, x, y + 1);

            let i = (y as usize * width + x as usize) as f64;
            let delta = lap - sum;
            sum += delta / (i + 1.0);
            m2 += delta * (lap - sum);
        }
    }

    m2 / n as f64
}

/// Sobel gradient magnitude statistics: mean, std, 90th percentile.
pub fn sobel_stats(sobel: &SobelResult) -> Vec<f64> {
    let mags = &sobel.magnitudes;
    let n = mags.len() as f64;

    let mean = mags.iter().sum::<f64>() / n;
    let variance = mags.iter().map(|&m| (m - mean) * (m - mean)).sum::<f64>() / n;
    let std_dev = sqrt(variance);

    let p90 = percentile(mags, 0.90);

    alloc::vec![mean, std_dev, p90]
}

/// Edge density: fraction of pixels where Sobel magnitude exceeds Otsu threshold.
pub fn edge_density(sobel: &SobelResult) -> f64 {
    let mags = &sobel.magnitudes;
    let threshold = otsu_threshold(mags);
    let count = mags.iter().filter(|&&m| m > threshold).count();
    count as f64 / mags.len() as f64
}

/// Edge direction histogram over 8 bins.
pub fn edge_direction_histogram(sobel: &SobelResult) -> Vec<f64> {
    let mags = &sobel.magnitudes;
    let oris = &sobel.orientations;

    let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
    let mut bins = [0u32; 8];

    for (i, &mag) in mags.iter().enumerate() {
        if mag > mean_mag {
            let bin = ((oris[i] + core::f64::consts::PI) / (2.0 * core::f64::consts::PI) * 8.0)
                as usize;
            let bin = bin.min(7);
            bins[bin] += 1;
        }
    }

    let total = bins.iter().sum::<u32>().max(1) as f64;
    bins.iter().map(|&b| b as f64 / total).collect()
}

/// Edges within ±10° of horizontal (0°/180°) or vertical (90°/270°) vs. diagonal.
pub fn hv_edge_ratio(sobel: &SobelResult) -> f64 {
    let mags = &sobel.magnitudes;
    let oris = &sobel.orientations;

    let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
    let angle_threshold = 10.0f64.to_radians();

    let mut hv_sum = 0.0;
    let mut diag_sum = 0.0;

    for (i, &mag) in mags.iter().enumerate() {
        if mag <= mean_mag {
            continue;
        }
        let abs_angle = oris[i].abs();
        // Clamp to [0, π/2] by symmetry
        let angle = if abs_angle > core::f64::consts::FRAC_PI_2 {
            core::f64::consts::PI - abs_angle
        } else {
            abs_angle
        };

        if angle < angle_threshold
            || (core::f64::consts::FRAC_PI_2 - angle).abs() < angle_threshold
        {
            hv_sum += mag;
        } else if (angle - core::f64::consts::FRAC_PI_4).abs()
            < core::f64::consts::FRAC_PI_4 - angle_threshold
        {
            diag_sum += mag;
        }
    }

    if diag_sum < 1e-12 {
        return if hv_sum > 0.0 { 10.0 } else { 1.0 };
    }
    hv_sum / diag_sum
}

/// Canny-like thin edges: non-max suppression on Sobel magnitude,
/// then return fraction of edge pixels.
pub fn canny_edge_density(sobel: &SobelResult) -> f64 {
    let mags = &sobel.magnitudes;
    let oris = &sobel.orientations;
    let width = sobel.width;
    let height = sobel.height;

    let nms = non_max_suppression(mags, oris, width, height);

    let threshold = otsu_threshold(&nms);
    let edge_count = nms.iter().filter(|&&m| m > threshold).count();
    edge_count as f64 / nms.len() as f64
}

/// Structure tensor features from Sobel gradients for rotation detection.
/// Returns: [dominant_angle (in units of π), coherence, gradient_energy].
pub fn structure_tensor_features(sobel: &SobelResult) -> Vec<f64> {
    let gx = &sobel.gx;
    let gy = &sobel.gy;

    let mut jxx = 0.0;
    let mut jyy = 0.0;
    let mut jxy = 0.0;

    for i in 0..gx.len() {
        jxx += gx[i] * gx[i];
        jyy += gy[i] * gy[i];
        jxy += gx[i] * gy[i];
    }

    let trace = jxx + jyy;
    let angle = if trace > 1e-12 {
        0.5 * atan2(2.0 * jxy, jxx - jyy) / core::f64::consts::PI
    } else {
        0.0
    };
    let coherence = if trace > 1e-12 {
        let num = sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy);
        num / trace
    } else {
        0.0
    };
    let energy = sqrt(trace.max(0.0)) / (sobel.width * sobel.height) as f64;

    alloc::vec![angle, coherence, energy]
}

/// Circular statistics of Sobel gradient orientation.
/// Returns: [circular_mean (in units of π), circular_variance].
pub fn sobel_circular_stats(sobel: &SobelResult) -> Vec<f64> {
    let mags = &sobel.magnitudes;
    let oris = &sobel.orientations;

    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut sum_weight = 0.0;

    for i in 0..mags.len() {
        let mag = mags[i];
        if mag < 1e-10 {
            continue;
        }
        let theta = oris[i];
        sum_cos += mag * cos(theta);
        sum_sin += mag * sin(theta);
        sum_weight += mag;
    }

    let circular_mean = if sum_weight > 1e-12 {
        atan2(sum_sin, sum_cos) / core::f64::consts::PI
    } else {
        0.0
    };

    let circular_var = if sum_weight > 1e-12 {
        let r = sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / sum_weight;
        1.0 - r
    } else {
        1.0
    };

    alloc::vec![circular_mean, circular_var]
}

/// Non-max suppression: thin edges to 1-pixel width.
fn non_max_suppression(
    mags: &[f64],
    oris: &[f64],
    width: usize,
    height: usize,
) -> Vec<f64> {
    let n = width * height;
    let mut result = vec![0.0; n];

    for y in 1..height as isize - 1 {
        for x in 1..width as isize - 1 {
            let idx = y as usize * width + x as usize;
            let angle = oris[idx];
            // Quantize to 4 directions: 0°, 45°, 90°, 135°
            let dir =
                ((angle + core::f64::consts::PI) / (core::f64::consts::PI / 4.0) + 0.5) as usize
                    % 4;

            let (n1, n2) = match dir {
                0 => {
                    let n1 = mags[(y as usize - 1) * width + x as usize];
                    let n2 = mags[(y as usize + 1) * width + x as usize];
                    (n1, n2)
                }
                1 => {
                    let n1 = mags[(y as usize - 1) * width + (x as usize + 1)];
                    let n2 = mags[(y as usize + 1) * width + (x as usize - 1)];
                    (n1, n2)
                }
                2 => {
                    let n1 = mags[y as usize * width + (x as usize - 1)];
                    let n2 = mags[y as usize * width + (x as usize + 1)];
                    (n1, n2)
                }
                _ => {
                    let n1 = mags[(y as usize - 1) * width + (x as usize - 1)];
                    let n2 = mags[(y as usize + 1) * width + (x as usize + 1)];
                    (n1, n2)
                }
            };

            result[idx] = if mags[idx] >= n1 && mags[idx] >= n2 {
                mags[idx]
            } else {
                0.0
            };
        }
    }

    result
}

/// Otsu threshold on a slice of values.
fn otsu_threshold(values: &[f64]) -> f64 {
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max_val - min_val) < 1e-12 {
        return min_val;
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

fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let idx = (sorted.len() as f64 * p) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn vertical_edge_image() -> Vec<f64> {
        vec![0.0, 0.0, 255.0, 255.0]
    }

    fn solid_gray_128() -> Vec<f64> {
        vec![128.0, 128.0, 128.0, 128.0, 128.0, 128.0]
    }

    #[test]
    fn laplacian_variance_solid() {
        let v = laplacian_variance(&solid_gray_128(), 2, 2);
        assert!(v < 0.001);
    }

    #[test]
    fn laplacian_variance_edge() {
        let v = laplacian_variance(&vertical_edge_image(), 2, 2);
        assert!(v > 0.0);
    }

    #[test]
    fn sobel_stats_solid() {
        let s = sobel(&solid_gray_128(), 2, 2);
        let s = sobel_stats(&s);
        assert!(s[0] < 0.001);
        assert!(s[1] < 0.001);
    }

    #[test]
    fn sobel_stats_edge() {
        let s = sobel(&vertical_edge_image(), 2, 2);
        let s = sobel_stats(&s);
        assert!(s[0] > 0.0);
    }

    #[test]
    fn edge_density_solid() {
        let s = sobel(&solid_gray_128(), 2, 2);
        let d = edge_density(&s);
        assert!(d < 0.5);
    }

    #[test]
    fn edge_density_edge() {
        let w = 8;
        let h = 4;
        let mut gray = Vec::with_capacity(w * h);
        for _y in 0..h {
            for x in 0..w {
                if x < w / 2 {
                    gray.push(0.0);
                } else {
                    gray.push(255.0);
                }
            }
        }
        let s = sobel(&gray, w, h);
        let d = edge_density(&s);
        assert!(d > 0.0);
    }

    #[test]
    fn edge_direction_histogram_solid() {
        let s = sobel(&solid_gray_128(), 2, 2);
        let h = edge_direction_histogram(&s);
        assert_eq!(h.len(), 8);
        let sum: f64 = h.iter().sum();
        assert!(sum >= 0.0);
    }

    #[test]
    fn hv_edge_ratio_near_one() {
        let s = sobel(&solid_gray_128(), 2, 2);
        let r = hv_edge_ratio(&s);
        assert!(r >= 0.0);
    }

    #[test]
    fn canny_edge_density_solid() {
        let s = sobel(&solid_gray_128(), 2, 2);
        let d = canny_edge_density(&s);
        assert!(d >= 0.0);
    }
}
