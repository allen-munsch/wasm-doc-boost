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

/// Get pixel at (x, y), clamping to border.
fn pixel_at(gray: &[f64], width: usize, height: usize, x: isize, y: isize) -> f64 {
    let x = x.clamp(0, width as isize - 1) as usize;
    let y = y.clamp(0, height as isize - 1) as usize;
    gray[y * width + x]
}

// ---------------------------------------------------------------------------
// 2e: Shadow & illumination features
// ---------------------------------------------------------------------------

/// Compute the illumination field via morphological closing
/// (dilation followed by erosion) with a disk of given radius.
fn morphological_closing(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let dilated = morph_dilate(gray, width, height, radius);
    morph_erode(&dilated, width, height, radius)
}

/// Greyscale dilation with disk structuring element.
fn morph_dilate(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = width * height;
    let mut result = vec![0.0; n];
    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut max_val = f64::NEG_INFINITY;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    // Check if within disk
                    if dx * dx + dy * dy <= (radius as isize) * (radius as isize) {
                        let v = pixel_at(gray, width, height, x + dx, y + dy);
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            }
            result[y as usize * width + x as usize] = max_val;
        }
    }
    result
}

/// Greyscale erosion with disk structuring element.
fn morph_erode(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = width * height;
    let mut result = vec![0.0; n];
    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut min_val = f64::INFINITY;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    if dx * dx + dy * dy <= (radius as isize) * (radius as isize) {
                        let v = pixel_at(gray, width, height, x + dx, y + dy);
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
            }
            result[y as usize * width + x as usize] = min_val;
        }
    }
    result
}

/// Shadow & illumination features.
/// Returns:
/// - dark_region_ratio: fraction of pixels where intensity < 0.7 × illumination
/// - illum_variance: variance of the illumination field
/// - shadow_depth_mean: mean of negative deviations (illum − original)
/// - shadow_depth_std: std of negative deviations
/// - shadow_edge_magnitude: mean Sobel magnitude on illumination field
pub fn shadow_features(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    let radius = if width.min(height) >= 21 { 10 } else { (width.min(height) as f64 * 0.1) as usize };
    let radius = radius.max(1);
    let illum = morphological_closing(&gray, width, height, radius);
    let n = gray.len() as f64;

    // Dark region ratio
    let mut dark_count = 0u32;
    for i in 0..gray.len() {
        if gray[i] < 0.7 * illum[i] {
            dark_count += 1;
        }
    }
    let dark_ratio = dark_count as f64 / n;

    // Illumination variance
    let illum_mean = illum.iter().sum::<f64>() / n;
    let illum_var = illum
        .iter()
        .map(|&v| (v - illum_mean) * (v - illum_mean))
        .sum::<f64>()
        / n;

    // Shadow depth: negative deviations only (where original < illumination)
    let mut neg_deviations = Vec::new();
    for i in 0..gray.len() {
        let dev = illum[i] - gray[i];
        if dev > 0.0 {
            neg_deviations.push(dev);
        }
    }
    let shadow_mean = if neg_deviations.is_empty() {
        0.0
    } else {
        neg_deviations.iter().sum::<f64>() / neg_deviations.len() as f64
    };
    let shadow_std = if neg_deviations.len() < 2 {
        0.0
    } else {
        let m = shadow_mean;
        let var = neg_deviations
            .iter()
            .map(|&d| (d - m) * (d - m))
            .sum::<f64>()
            / neg_deviations.len() as f64;
        sqrt(var)
    };

    // Shadow edge detector: Sobel on illumination field
    let mut edge_sum = 0.0;
    let mut edge_count = 0u32;
    for y in 1..height as isize - 1 {
        for x in 1..width as isize - 1 {
            let gx = pixel_at(&illum, width, height, x + 1, y)
                - pixel_at(&illum, width, height, x - 1, y);
            let gy = pixel_at(&illum, width, height, x, y + 1)
                - pixel_at(&illum, width, height, x, y - 1);
            edge_sum += sqrt(gx * gx + gy * gy);
            edge_count += 1;
        }
    }
    let shadow_edge = if edge_count > 0 {
        edge_sum / edge_count as f64
    } else {
        0.0
    };

    alloc::vec![
        dark_ratio,
        illum_var,
        shadow_mean,
        shadow_std,
        shadow_edge,
    ]
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
    fn shadow_features_solid() {
        let f = shadow_features(&solid_128(40, 40), 40, 40);
        assert_eq!(f.len(), 5);
        // Solid image: no shadows
        assert!(f[0] < 0.01); // dark_ratio ~ 0
        assert!(f[1] < 0.01); // illum_variance ~ 0
        assert!(f[2] < 0.01); // shadow_mean ~ 0
        assert!(f[4] < 0.01); // shadow_edge ~ 0
    }

    #[test]
    fn shadow_features_dark_region() {
        // 80x80 image: center 40x40 dark (0), rest bright (255)
        // The dark patch is larger than the r=10 structuring element → survives closing
        let w = 80;
        let h = 80;
        let mut pixels = vec![];
        for y in 0..h {
            for x in 0..w {
                if x >= 20 && x < 60 && y >= 20 && y < 60 {
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
        let f = shadow_features(&pixels, w, h);
        // Dark patch should survive morphological closing
        assert!(f[0] > 0.0); // dark_ratio
        assert!(f[1] > 0.0); // illum_variance
    }
}
