use alloc::vec;
use alloc::vec::Vec;
use libm::sqrt;

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

/// Get grayscale pixel at (x, y), clamping to border.
fn pixel_at(gray: &[f64], width: usize, height: usize, x: isize, y: isize) -> f64 {
    let x = x.clamp(0, width as isize - 1) as usize;
    let y = y.clamp(0, height as isize - 1) as usize;
    gray[y * width + x]
}

// ---------------------------------------------------------------------------
// 2d-a: High-pass residual variance (original − median-filtered 5×5)
// ---------------------------------------------------------------------------

/// High-pass residual: subtract 5×5 median-filtered image, return variance.
pub fn high_pass_residual_variance(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let filtered = median_filter_5x5(&gray, width, height);
    let n = gray.len() as f64;

    let mean_residual: f64 = gray
        .iter()
        .zip(filtered.iter())
        .map(|(&g, &f)| g - f)
        .sum::<f64>()
        / n;

    let variance: f64 = gray
        .iter()
        .zip(filtered.iter())
        .map(|(&g, &f)| {
            let r = g - f - mean_residual;
            r * r
        })
        .sum::<f64>()
        / n;

    variance
}

/// 5×5 median filter on grayscale image (border pixels use clamped values).
fn median_filter_5x5(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut result = vec![0.0; n];
    let mut window = [0.0f64; 25];

    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut k = 0;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    window[k] = pixel_at(gray, width, height, x + dx, y + dy);
                    k += 1;
                }
            }
            // Sort window and take median
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            result[y as usize * width + x as usize] = window[12];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// 2d-b: JPEG blockiness (8×8 block boundary vs. interior differences)
// ---------------------------------------------------------------------------

/// JPEG blockiness: average difference across 8×8 block boundaries
/// vs. interior differences for horizontal boundaries.
fn blockiness_h(gray: &[f64], width: usize, height: usize) -> f64 {
    let mut boundary_diff = 0.0;
    let mut boundary_count = 0u32;
    let mut interior_diff = 0.0;
    let mut interior_count = 0u32;

    for by in 0..(height / 8) {
        for bx in 0..(width / 8) {
            let start_y = by * 8;
            let start_x = bx * 8;
            // Interior: rows 0..7, compare adjacent rows
            for row in 0..7 {
                let y = start_y + row;
                for x in start_x..start_x + 8 {
                    if x < width && y + 1 < height {
                        interior_diff +=
                            (gray[y * width + x] - gray[(y + 1) * width + x])
                                .abs();
                        interior_count += 1;
                    }
                }
            }
            // Bottom boundary: row 7 vs row 8 (next block's top)
            if by + 1 < height / 8 {
                let y = start_y + 7;
                for x in start_x..start_x + 8 {
                    if x < width && y + 1 < height {
                        boundary_diff +=
                            (gray[y * width + x] - gray[(y + 1) * width + x])
                                .abs();
                        boundary_count += 1;
                    }
                }
            }
        }
    }

    if interior_count == 0 || boundary_count == 0 {
        return 0.0;
    }
    let interior_avg = interior_diff / interior_count as f64;
    let boundary_avg = boundary_diff / boundary_count as f64;
    if interior_avg < 1e-12 {
        0.0
    } else {
        (boundary_avg - interior_avg).max(0.0) / interior_avg
    }
}

/// JPEG blockiness: vertical direction.
fn blockiness_v(gray: &[f64], width: usize, height: usize) -> f64 {
    let mut boundary_diff = 0.0;
    let mut boundary_count = 0u32;
    let mut interior_diff = 0.0;
    let mut interior_count = 0u32;

    for by in 0..(height / 8) {
        for bx in 0..(width / 8) {
            let start_y = by * 8;
            let start_x = bx * 8;
            // Interior: columns 0..7, compare adjacent cols
            for col in 0..7 {
                let x = start_x + col;
                for y in start_y..start_y + 8 {
                    if x + 1 < width && y < height {
                        interior_diff +=
                            (gray[y * width + x] - gray[y * width + (x + 1)])
                                .abs();
                        interior_count += 1;
                    }
                }
            }
            // Right boundary: col 7 vs col 8
            if bx + 1 < width / 8 {
                let x = start_x + 7;
                for y in start_y..start_y + 8 {
                    if x + 1 < width && y < height {
                        boundary_diff +=
                            (gray[y * width + x] - gray[y * width + (x + 1)])
                                .abs();
                        boundary_count += 1;
                    }
                }
            }
        }
    }

    if interior_count == 0 || boundary_count == 0 {
        return 0.0;
    }
    let interior_avg = interior_diff / interior_count as f64;
    let boundary_avg = boundary_diff / boundary_count as f64;
    if interior_avg < 1e-12 {
        0.0
    } else {
        (boundary_avg - interior_avg).max(0.0) / interior_avg
    }
}

/// JPEG blockiness features: horizontal blockiness, vertical blockiness.
pub fn jpeg_blockiness(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    if width < 16 || height < 16 {
        return alloc::vec![0.0, 0.0];
    }
    alloc::vec![blockiness_h(&gray, width, height), blockiness_v(&gray, width, height)]
}

// ---------------------------------------------------------------------------
// 2d-c: Gradient signal-to-noise ratio
// ---------------------------------------------------------------------------

/// Gradient SNR: ratio of mean Sobel magnitude to residual variance.
/// Higher SNR → sharper image (documents, digital graphics).
/// Lower SNR → noisy/natural images.
pub fn gradient_snr(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let residual_var = high_pass_residual_variance(pixels, width, height);

    // Compute mean Sobel magnitude manually to avoid dependency
    let mut sum_mag = 0.0;
    let mut count = 0u32;
    for y in 1..height as isize - 1 {
        for x in 1..width as isize - 1 {
            let gx = pixel_at(&gray, width, height, x + 1, y)
                - pixel_at(&gray, width, height, x - 1, y);
            let gy = pixel_at(&gray, width, height, x, y + 1)
                - pixel_at(&gray, width, height, x, y - 1);
            sum_mag += sqrt(gx * gx + gy * gy);
            count += 1;
        }
    }

    let mean_mag = if count > 0 {
        sum_mag / count as f64
    } else {
        0.0
    };

    if residual_var < 1e-12 {
        return if mean_mag > 0.0 { 100.0 } else { 0.0 };
    }
    mean_mag / residual_var
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn solid_128(w: usize, h: usize) -> Vec<u8> {
        let n = w * h;
        let mut p = Vec::with_capacity(n * 3);
        for _ in 0..n {
            p.push(128);
            p.push(128);
            p.push(128);
        }
        p
    }

    #[test]
    fn high_pass_residual_variance_solid() {
        let v = high_pass_residual_variance(&solid_128(8, 8), 8, 8);
        // Solid image → median filter preserves it → residual ≈ 0
        assert!(v < 0.001);
    }

    #[test]
    fn jpeg_blockiness_solid() {
        let b = jpeg_blockiness(&solid_128(16, 16), 16, 16);
        assert_eq!(b.len(), 2);
        // Solid image: no blockiness
        assert!(b[0] < 0.001);
        assert!(b[1] < 0.001);
    }

    #[test]
    fn gradient_snr_solid() {
        let snr = gradient_snr(&solid_128(8, 8), 8, 8);
        // Solid image: near-zero gradients and near-zero residual
        assert!(snr >= 0.0);
    }
}
