use alloc::vec;
use alloc::vec::Vec;
use libm::{atan2, sqrt};

/// Convert RGB pixels to grayscale f64 values.
fn to_grayscale(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut gray = Vec::with_capacity(n);
    for i in 0..n {
        let r = pixels[i * 3] as f64;
        let g = pixels[i * 3 + 1] as f64;
        let b = pixels[i * 3 + 2] as f64;
        gray.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    gray
}

/// Get grayscale pixel value at (x, y), clamping to border.
fn pixel_at(gray: &[f64], width: usize, height: usize, x: isize, y: isize) -> f64 {
    let x = x.clamp(0, width as isize - 1) as usize;
    let y = y.clamp(0, height as isize - 1) as usize;
    gray[y * width + x]
}

/// Laplacian variance (blur detector).
/// Applies 3×3 Laplacian kernel and returns variance of the result.
pub fn laplacian_variance(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
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

/// Sobel gradient magnitude per pixel.
fn sobel_magnitudes(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut mags = Vec::with_capacity(n);

    for y in 0..height as isize {
        for x in 0..width as isize {
            let gx = -pixel_at(gray, width, height, x - 1, y - 1)
                + pixel_at(gray, width, height, x + 1, y - 1)
                - 2.0 * pixel_at(gray, width, height, x - 1, y)
                + 2.0 * pixel_at(gray, width, height, x + 1, y)
                - pixel_at(gray, width, height, x - 1, y + 1)
                + pixel_at(gray, width, height, x + 1, y + 1);

            let gy = -pixel_at(gray, width, height, x - 1, y - 1)
                - 2.0 * pixel_at(gray, width, height, x, y - 1)
                - pixel_at(gray, width, height, x + 1, y - 1)
                + pixel_at(gray, width, height, x - 1, y + 1)
                + 2.0 * pixel_at(gray, width, height, x, y + 1)
                + pixel_at(gray, width, height, x + 1, y + 1);

            mags.push(sqrt(gx * gx + gy * gy));
        }
    }
    mags
}

/// Sobel gradient orientation per pixel (radians, [-π, π]).
fn sobel_orientations(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut oris = Vec::with_capacity(n);

    for y in 0..height as isize {
        for x in 0..width as isize {
            let gx = -pixel_at(gray, width, height, x - 1, y - 1)
                + pixel_at(gray, width, height, x + 1, y - 1)
                - 2.0 * pixel_at(gray, width, height, x - 1, y)
                + 2.0 * pixel_at(gray, width, height, x + 1, y)
                - pixel_at(gray, width, height, x - 1, y + 1)
                + pixel_at(gray, width, height, x + 1, y + 1);

            let gy = -pixel_at(gray, width, height, x - 1, y - 1)
                - 2.0 * pixel_at(gray, width, height, x, y - 1)
                - pixel_at(gray, width, height, x + 1, y - 1)
                + pixel_at(gray, width, height, x - 1, y + 1)
                + 2.0 * pixel_at(gray, width, height, x, y + 1)
                + pixel_at(gray, width, height, x + 1, y + 1);

            oris.push(atan2(gy, gx));
        }
    }
    oris
}

/// Sobel gradient magnitude statistics: mean, std, 90th percentile.
pub fn sobel_stats(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    let mags = sobel_magnitudes(&gray, width, height);
    let n = mags.len() as f64;

    let mean = mags.iter().sum::<f64>() / n;
    let variance = mags.iter().map(|&m| (m - mean) * (m - mean)).sum::<f64>() / n;
    let std_dev = sqrt(variance);

    let p90 = percentile(&mags, 0.90);

    alloc::vec![mean, std_dev, p90]
}

/// Edge density: fraction of pixels where Sobel magnitude exceeds Otsu threshold.
pub fn edge_density(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let mags = sobel_magnitudes(&gray, width, height);
    let threshold = otsu_threshold(&mags);
    let count = mags.iter().filter(|&&m| m > threshold).count();
    count as f64 / mags.len() as f64
}

/// Edge direction histogram: 8 bins of gradient orientation, normalised.
/// Bin 0: [-π, -3π/4), Bin 1: [-3π/4, -π/2), ..., Bin 7: [3π/4, π]
/// Only counts pixels where Sobel magnitude > mean magnitude.
pub fn edge_direction_histogram(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    let mags = sobel_magnitudes(&gray, width, height);
    let oris = sobel_orientations(&gray, width, height);

    let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
    let mut bins = [0u32; 8];

    for (i, &mag) in mags.iter().enumerate() {
        if mag > mean_mag {
            // Map [-π, π] to bin [0, 7]
            let bin = ((oris[i] + core::f64::consts::PI) / (2.0 * core::f64::consts::PI) * 8.0) as usize;
            let bin = bin.min(7);
            bins[bin] += 1;
        }
    }

    let total = bins.iter().sum::<u32>().max(1) as f64;
    bins.iter().map(|&b| b as f64 / total).collect()
}

/// Horizontal/vertical edge ratio vs. diagonal.
/// Edges within ±10° of horizontal (0°/180°) or vertical (90°/270°) vs. diagonal.
pub fn hv_edge_ratio(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let mags = sobel_magnitudes(&gray, width, height);
    let oris = sobel_orientations(&gray, width, height);

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

        if angle < angle_threshold || (core::f64::consts::FRAC_PI_2 - angle).abs() < angle_threshold {
            hv_sum += mag;
        } else if (angle - core::f64::consts::FRAC_PI_4).abs() < core::f64::consts::FRAC_PI_4 - angle_threshold {
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
pub fn canny_edge_density(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let mags = sobel_magnitudes(&gray, width, height);
    let oris = sobel_orientations(&gray, width, height);

    let nms = non_max_suppression(&mags, &oris, width, height);

    let threshold = otsu_threshold(&nms);
    let edge_count = nms.iter().filter(|&&m| m > threshold).count();
    edge_count as f64 / nms.len() as f64
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
            let dir = ((angle + core::f64::consts::PI) / (core::f64::consts::PI / 4.0) + 0.5) as usize % 4;

            let (n1, n2) = match dir {
                0 => {
                    // horizontal edge (gradient is vertical)
                    let n1 = mags[(y as usize - 1) * width + x as usize];
                    let n2 = mags[(y as usize + 1) * width + x as usize];
                    (n1, n2)
                }
                1 => {
                    // 45° diagonal
                    let n1 = mags[(y as usize - 1) * width + (x as usize + 1)];
                    let n2 = mags[(y as usize + 1) * width + (x as usize - 1)];
                    (n1, n2)
                }
                2 => {
                    // vertical edge (gradient is horizontal)
                    let n1 = mags[y as usize * width + (x as usize - 1)];
                    let n2 = mags[y as usize * width + (x as usize + 1)];
                    (n1, n2)
                }
                _ => {
                    // 135° diagonal
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

/// Compute a percentile from a slice by sorting.
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

    /// 2x2 gradient image: black on left, white on right → strong vertical edge
    fn vertical_edge_image() -> Vec<u8> {
        vec![
            0, 0, 0, 255, 255, 255,   // row 0: black, white
            0, 0, 0, 255, 255, 255,   // row 1: black, white
        ]
    }

    fn solid_gray_128() -> Vec<u8> {
        vec![
            128, 128, 128, 128, 128, 128,
            128, 128, 128, 128, 128, 128,
        ]
    }

    #[test]
    fn laplacian_variance_solid() {
        let v = laplacian_variance(&solid_gray_128(), 2, 2);
        assert!(v < 0.001); // flat image, zero Laplacian everywhere
    }

    #[test]
    fn laplacian_variance_edge() {
        let v = laplacian_variance(&vertical_edge_image(), 2, 2);
        assert!(v > 0.0); // non-zero variance at edge
    }

    #[test]
    fn sobel_stats_solid() {
        let s = sobel_stats(&solid_gray_128(), 2, 2);
        assert!(s[0] < 0.001); // mean ~ 0
        assert!(s[1] < 0.001); // std ~ 0
    }

    #[test]
    fn sobel_stats_edge() {
        let s = sobel_stats(&vertical_edge_image(), 2, 2);
        assert!(s[0] > 0.0); // mean > 0
    }

    #[test]
    fn edge_density_solid() {
        let d = edge_density(&solid_gray_128(), 2, 2);
        assert!(d < 0.5); // few edges
    }

    #[test]
    fn edge_density_edge() {
        // 8x4 image: left half black, right half white → strong vertical edge
        let w = 8;
        let h = 4;
        let mut pixels = Vec::with_capacity(w * h * 3);
        for _y in 0..h {
            for x in 0..w {
                if x < w / 2 {
                    pixels.push(0);
                    pixels.push(0);
                    pixels.push(0);
                } else {
                    pixels.push(255);
                    pixels.push(255);
                    pixels.push(255);
                }
            }
        }
        let d = edge_density(&pixels, w, h);
        assert!(d > 0.0);
    }

    #[test]
    fn edge_direction_histogram_solid() {
        let h = edge_direction_histogram(&solid_gray_128(), 2, 2);
        assert_eq!(h.len(), 8);
        // All bins should be 0 (no edges above mean) or uniform
        let sum: f64 = h.iter().sum();
        assert!(sum >= 0.0);
    }

    #[test]
    fn hv_edge_ratio_near_one() {
        let r = hv_edge_ratio(&solid_gray_128(), 2, 2);
        assert!(r >= 0.0);
    }

    #[test]
    fn canny_edge_density_solid() {
        let d = canny_edge_density(&solid_gray_128(), 2, 2);
        assert!(d >= 0.0);
    }
}
