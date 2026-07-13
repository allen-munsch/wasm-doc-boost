use alloc::vec::Vec;
use libm::{log2, sqrt};

/// Compute per-channel statistics for an RGB image.
///
/// `pixels` is a flat slice of `[R, G, B, R, G, B, ...]`.
/// Returns: [R_mean, R_std, R_skew, R_kurt, G_mean, G_std, G_skew, G_kurt, B_mean, B_std, B_skew, B_kurt]
pub fn per_channel_stats(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut sums = [0.0f64; 3];
    let mut m2 = [0.0f64; 3];
    let mut m3 = [0.0f64; 3];
    let mut m4 = [0.0f64; 3];

    for i in 0..n {
        for c in 0..3 {
            let x = pixels[i * 3 + c] as f64;
            let delta = x - sums[c];
            let delta_n = delta / (i + 1) as f64;
            let delta_n2 = delta_n * delta_n;
            let term1 = delta * delta_n * i as f64;

            sums[c] += delta_n;
            m4[c] += term1 * delta_n2 * (i as f64 * i as f64 - 3.0 * i as f64 + 3.0)
                + 6.0 * delta_n2 * m2[c]
                - 4.0 * delta_n * m3[c];
            m3[c] += term1 * delta_n * (i as f64 - 2.0) - 3.0 * delta_n * m2[c];
            m2[c] += delta * (x - sums[c]);
        }
    }

    let mut features = Vec::with_capacity(12);
    for c in 0..3 {
        let nf = n as f64;
        let mean = sums[c];
        let variance = m2[c] / nf;
        let std_dev = sqrt(variance);

        let skew = if variance > 1e-12 {
            (m3[c] / nf) / (variance * std_dev)
        } else {
            0.0
        };

        let kurt = if variance > 1e-12 {
            (m4[c] / nf) / (variance * variance) - 3.0
        } else {
            0.0
        };

        features.push(mean);
        features.push(std_dev);
        features.push(skew);
        features.push(kurt);
    }

    features
}

/// Convert RGB pixels to grayscale using luminance weights.
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

/// Grayscale statistics: mean, median (approximated by histogram), std, entropy.
pub fn grayscale_stats(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    let n = gray.len() as f64;

    let mean = gray.iter().sum::<f64>() / n;

    let variance = gray.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std_dev = sqrt(variance);

    let median = histogram_median(&gray);

    let entropy = histogram_entropy(&gray);

    alloc::vec![mean, median, std_dev, entropy]
}

/// Approximate median from a 256-bin histogram.
fn histogram_median(values: &[f64]) -> f64 {
    let mut hist = [0u32; 256];
    for &v in values {
        let idx = (v.clamp(0.0, 255.0)) as usize;
        let idx = idx.min(255);
        hist[idx] += 1;
    }
    let half = values.len() as u32 / 2;
    let mut cumulative = 0u32;
    for (i, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative >= half {
            return i as f64;
        }
    }
    128.0
}

/// Entropy of an 8-bit grayscale image using a 256-bin histogram.
fn histogram_entropy(values: &[f64]) -> f64 {
    let mut hist = [0u32; 256];
    for &v in values {
        let idx = (v.clamp(0.0, 255.0)) as usize;
        let idx = idx.min(255);
        hist[idx] += 1;
    }
    let n = values.len() as f64;
    let mut entropy = 0.0;
    for &count in hist.iter() {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * log2(p);
        }
    }
    entropy
}

/// Colorfulness metric (Hasler & Süsstrunk):
/// `σ(rg) + σ(yb)` where `rg = R - G` and `yb = 0.5*(R+G) - B`
pub fn colorfulness(pixels: &[u8], width: usize, height: usize) -> f64 {
    let n = width * height;
    let nf = n as f64;

    let mut sum_rg = 0.0;
    let mut sum_yb = 0.0;
    let mut m2_rg = 0.0;
    let mut m2_yb = 0.0;

    for i in 0..n {
        let r = pixels[i * 3] as f64;
        let g = pixels[i * 3 + 1] as f64;
        let b = pixels[i * 3 + 2] as f64;

        let rg = r - g;
        let yb = 0.5 * (r + g) - b;

        let delta_rg = rg - sum_rg;
        sum_rg += delta_rg / (i + 1) as f64;
        m2_rg += delta_rg * (rg - sum_rg);

        let delta_yb = yb - sum_yb;
        sum_yb += delta_yb / (i + 1) as f64;
        m2_yb += delta_yb * (yb - sum_yb);
    }

    let std_rg = sqrt(m2_rg / nf);
    let std_yb = sqrt(m2_yb / nf);
    std_rg + std_yb
}

/// Integer-based RGB to HSV saturation.
/// Returns (mean_saturation, variance_saturation).
pub fn saturation_stats(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut sum = 0.0;
    let mut m2 = 0.0;

    for i in 0..n {
        let r = pixels[i * 3] as f64;
        let g = pixels[i * 3 + 1] as f64;
        let b = pixels[i * 3 + 2] as f64;

        let max_val = r.max(g).max(b);
        let min_val = r.min(g).min(b);
        let delta = max_val - min_val;

        let s = if max_val > 0.0 {
            delta / max_val
        } else {
            0.0
        };

        let delta_s = s - sum;
        sum += delta_s / (i + 1) as f64;
        m2 += delta_s * (s - sum);
    }

    let mean = sum;
    let variance = m2 / n as f64;
    alloc::vec![mean, variance]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Create a 2x2 solid red image: [255,0,0, 255,0,0, 255,0,0, 255,0,0]
    fn solid_red() -> Vec<u8> {
        vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]
    }

    fn solid_gray() -> Vec<u8> {
        vec![128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    }

    #[test]
    fn per_channel_stats_solid_red() {
        let stats = per_channel_stats(&solid_red(), 2, 2);
        // R: mean=255, std=0, skew=0, kurt=0
        assert!((stats[0] - 255.0).abs() < 0.001);
        assert!(stats[1] < 0.001);
        // G: mean=0, std=0
        assert!(stats[4] < 0.001);
        assert!(stats[5] < 0.001);
        // B: mean=0, std=0
        assert!(stats[8] < 0.001);
        assert!(stats[9] < 0.001);
    }

    #[test]
    fn grayscale_stats_solid_gray() {
        let stats = grayscale_stats(&solid_gray(), 2, 2);
        assert!((stats[0] - 128.0).abs() < 1.0);
        assert!(stats[2] < 0.001); // std ~ 0
        assert!(stats[3] < 0.001); // entropy ~ 0 for uniform
    }

    #[test]
    fn colorfulness_solid_gray() {
        let c = colorfulness(&solid_gray(), 2, 2);
        assert!(c < 0.001);
    }

    #[test]
    fn colorfulness_solid_red() {
        let c = colorfulness(&solid_red(), 2, 2);
        // All pixels identical → zero variance → colorfulness = 0
        assert!(c < 0.001);
    }

    #[test]
    fn colorfulness_varied() {
        // 2x2 image with different colors
        let pixels = vec![
            255, 0, 0,   // red
            0, 255, 0,   // green
            0, 0, 255,   // blue
            128, 128, 128, // gray
        ];
        let c = colorfulness(&pixels, 2, 2);
        assert!(c > 1.0);
    }

    #[test]
    fn saturation_stats_solid_gray() {
        let s = saturation_stats(&solid_gray(), 2, 2);
        assert!(s[0] < 0.001); // saturation ~ 0 for gray
    }

    #[test]
    fn saturation_stats_solid_red() {
        let s = saturation_stats(&solid_red(), 2, 2);
        assert!(s[0] > 0.99); // saturation ~ 1 for pure red
    }
}
