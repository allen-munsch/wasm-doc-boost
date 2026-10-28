use crate::pixel_at;
use alloc::vec;
use alloc::vec::Vec;
use libm::sqrt;

// ---------------------------------------------------------------------------
// 2e: Shadow & illumination features
// ---------------------------------------------------------------------------

/// Compute the illumination field via morphological closing
/// (dilation followed by erosion) with a disk of given radius.
fn morphological_closing(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let dilated = morph_dilate(gray, width, height, radius);
    morph_erode(&dilated, width, height, radius)
}

/// Greyscale dilation with a square structuring element (side 2r+1),
/// computed separably: horizontal running max then vertical running max.
/// A square SE is a separable max filter, so this is O(N·r) instead of the
/// old disk's O(N·r²) (r=10 disk ≈ 314 px/out px → ~8× slower). The closing
/// only needs the slow illumination field, so square-vs-disk is immaterial.
fn morph_dilate(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = width * height;
    let mut h = vec![0.0; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(width);
            let mut m = f64::NEG_INFINITY;
            for xi in lo..hi {
                let v = gray[row + xi];
                if v > m {
                    m = v;
                }
            }
            h[row + x] = m;
        }
    }
    let mut result = vec![0.0; n];
    for x in 0..width {
        for y in 0..height {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(height);
            let mut m = f64::NEG_INFINITY;
            for yi in lo..hi {
                let v = h[yi * width + x];
                if v > m {
                    m = v;
                }
            }
            result[y * width + x] = m;
        }
    }
    result
}

/// Greyscale erosion with a square structuring element (side 2r+1),
/// computed separably: horizontal running min then vertical running min.
fn morph_erode(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = width * height;
    let mut h = vec![0.0; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(width);
            let mut m = f64::INFINITY;
            for xi in lo..hi {
                let v = gray[row + xi];
                if v < m {
                    m = v;
                }
            }
            h[row + x] = m;
        }
    }
    let mut result = vec![0.0; n];
    for x in 0..width {
        for y in 0..height {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(height);
            let mut m = f64::INFINITY;
            for yi in lo..hi {
                let v = h[yi * width + x];
                if v < m {
                    m = v;
                }
            }
            result[y * width + x] = m;
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
pub fn shadow_features(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let radius = if width.min(height) >= 21 {
        10
    } else {
        (width.min(height) as f64 * 0.1) as usize
    };
    let radius = radius.max(1);
    let illum = morphological_closing(gray, width, height, radius);
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

    alloc::vec![dark_ratio, illum_var, shadow_mean, shadow_std, shadow_edge,]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn solid_128(w: usize, h: usize) -> Vec<f64> {
        vec![128.0; w * h]
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
        // 80x80 image: center 40x40 dark (0), rest bright (255).
        // The dark patch is larger than the r=10 structuring element, so it
        // survives closing and shows up as illumination VARIANCE (f[1]), not
        // as a dark_ratio band (that needs a feature smaller than the SE).
        let w = 80;
        let h = 80;
        let mut pixels = vec![];
        for y in 0..h {
            for x in 0..w {
                if (20..60).contains(&x) && (20..60).contains(&y) {
                    pixels.push(0.0);
                } else {
                    pixels.push(255.0);
                }
            }
        }
        let f = shadow_features(&pixels, w, h);
        assert!(f[1] > 0.0); // illum_variance: large shadow captured
    }

    #[test]
    fn shadow_features_dark_ratio_small_feature() {
        // 80x80 bright image with an 8x8 dark spot, SMALLER than the r=10 SE.
        // Closing fills the spot (illumination stays bright over it) while the
        // pixel stays dark, so dark_ratio (f[0]) must be positive.
        let w = 80;
        let h = 80;
        let mut pixels = vec![255.0; w * h];
        for y in 36..44 {
            for x in 36..44 {
                pixels[y * w + x] = 0.0;
            }
        }
        let f = shadow_features(&pixels, w, h);
        assert!(f[0] > 0.0); // dark_ratio
    }
}
