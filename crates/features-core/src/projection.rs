use alloc::vec;
use alloc::vec::Vec;

/// Grayscale projection features for rotation detection.
///
/// Documents with horizontal text lines have high row variance (alternating
/// text/blank rows) and moderate column variance. Rotating 90° swaps this
/// pattern. Row/column entropy captures the structure of the projection
/// distribution.
///
/// Returns [row_proj_var, col_proj_var, row_col_ratio, row_entropy, col_entropy].
pub fn projection_features(gray: &[f64], width: usize, height: usize) -> Vec<f64> {

    // Row projections: mean intensity per row
    let mut row_proj = Vec::with_capacity(height);
    for y in 0..height {
        let row_start = y * width;
        let row_sum: f64 = gray[row_start..row_start + width].iter().sum();
        row_proj.push(row_sum / width as f64);
    }

    // Column projections: mean intensity per column
    let mut col_proj = Vec::with_capacity(width);
    for x in 0..width {
        let mut col_sum = 0.0;
        for y in 0..height {
            col_sum += gray[y * width + x];
        }
        col_proj.push(col_sum / height as f64);
    }

    // Variance of projections
    let row_mean = row_proj.iter().sum::<f64>() / height as f64;
    let row_var = row_proj.iter()
        .map(|&v| (v - row_mean) * (v - row_mean))
        .sum::<f64>()
        / height as f64;

    let col_mean = col_proj.iter().sum::<f64>() / width as f64;
    let col_var = col_proj.iter()
        .map(|&v| (v - col_mean) * (v - col_mean))
        .sum::<f64>()
        / width as f64;

    // Ratio: row_var / col_var, clamped to [1e-3, 1e3]
    let ratio = if col_var > 1e-10 {
        (row_var / col_var).clamp(1e-3, 1e3)
    } else {
        1e3
    };

    // Entropy of row projections (16-bin histogram)
    let row_entropy = projection_entropy(&row_proj, 16);
    let col_entropy = projection_entropy(&col_proj, 16);

    alloc::vec![
        row_var,
        col_var,
        ratio,
        row_entropy,
        col_entropy,
    ]
}

/// Entropy of a 1D projection using `n_bins` equal-width bins.
fn projection_entropy(values: &[f64], n_bins: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut bins = vec![0u32; n_bins];
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range < 1e-10 {
        return 0.0;
    }

    for &v in values {
        let idx = ((v - min_val) / range * n_bins as f64) as usize;
        let idx = idx.min(n_bins - 1);
        bins[idx] += 1;
    }

    let n = values.len() as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * libm::log(p);
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_gray_128(w: usize, h: usize) -> Vec<f64> {
        vec![128.0; w * h]
    }

    #[test]
    fn test_projection_solid_image() {
        let pixels = solid_gray_128(32, 32);
        let f = projection_features(&pixels, 32, 32);
        // Solid image: all rows/cols identical → near-zero variance
        assert!(f[0] < 0.001, "row_var={}", f[0]);
        assert!(f[1] < 0.001, "col_var={}", f[1]);
        assert!(f[3] < 0.001, "row_entropy={}", f[3]);
        assert!(f[4] < 0.001, "col_entropy={}", f[4]);
    }

    #[test]
    fn test_projection_with_structure() {
        // Image with horizontal stripes: alternating dark/light rows
        let w = 32;
        let h = 32;
        let mut gray = vec![0.0f64; w * h];
        for y in 0..h {
            let val = if y % 8 < 4 { 32.0f64 } else { 224.0f64 };
            for x in 0..w {
                gray[y * w + x] = val;
            }
        }
        let f = projection_features(&gray, w, h);
        // Row variance should be high (alternating stripes)
        assert!(f[0] > 100.0, "row_var={}", f[0]);
        // Column variance should be low (uniform per column)
        assert!(f[1] < 1.0, "col_var={}", f[1]);
        // Ratio should be > 1
        assert!(f[2] > 1.0, "ratio={}", f[2]);
    }

    #[test]
    fn test_projection_five_features() {
        let pixels = solid_gray_128(16, 16);
        let f = projection_features(&pixels, 16, 16);
        assert_eq!(f.len(), 5);
    }
}
