use alloc::vec;
use alloc::vec::Vec;
use libm::{cos, log2, sqrt};

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
// 2c-a: Block-wise 8×8 DCT energy
// ---------------------------------------------------------------------------

/// Low-frequency energy ratio from block-wise 8×8 DCT.
/// Divides image into 8×8 blocks, applies DCT-II, computes ratio of
/// low-freq energy (first 3 zig-zag coeffs) to total, averaged over blocks.
pub fn dct_low_freq_ratio(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    if blocks_x == 0 || blocks_y == 0 {
        return 0.0;
    }

    let mut sum_ratio = 0.0;
    let mut count = 0u32;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block = [0.0f64; 64];
            for j in 0..8 {
                for i in 0..8 {
                    block[j * 8 + i] =
                        gray[(by * 8 + j) * width + (bx * 8 + i)];
                }
            }
            let dct = dct_8x8(&block, 0);
            let low = dct[0].abs() + dct[1].abs() + dct[2].abs();
            let total: f64 = dct.iter().map(|&c| c.abs()).sum();
            if total > 1e-12 {
                sum_ratio += low / total;
            }
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum_ratio / count as f64
    }
}

/// 8×8 DCT-II on `block` (row-major). `block` is used as scratch space for
/// the 1D DCT, then the 8×8 output replaces it.
fn dct_8x8(block: &[f64; 64], _stride: usize) -> [f64; 64] {
    let mut tmp = [0.0f64; 64];
    let mut out = [0.0f64; 64];

    // DCT on rows
    for j in 0..8 {
        for k in 0..8 {
            let mut sum = 0.0;
            for n in 0..8 {
                sum += block[j * 8 + n]
                    * cos(core::f64::consts::PI / 8.0 * (n as f64 + 0.5) * k as f64);
            }
            tmp[j * 8 + k] = sum;
        }
    }

    // DCT on columns
    for i in 0..8 {
        for k in 0..8 {
            let mut sum = 0.0;
            for n in 0..8 {
                sum += tmp[n * 8 + i]
                    * cos(core::f64::consts::PI / 8.0 * (n as f64 + 0.5) * k as f64);
            }
            out[k * 8 + i] = sum;
        }
    }

    // Normalize: c(0) = 1/sqrt(2), c(k>0) = 1, and factor 2/N overall → 2/8 = 1/4 per dimension
    for j in 0..8 {
        for i in 0..8 {
            let cu = if j == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if i == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            out[j * 8 + i] *= 0.25 * cu * cv;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// 2c-b: Rotation-invariant uniform LBP histogram
// ---------------------------------------------------------------------------

/// Rotation-invariant uniform LBP histogram (10 bins).
/// Bin 0-8: uniform patterns (0-8 transitions), Bin 9: non-uniform.
pub fn lbp_histogram(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    if width < 3 || height < 3 {
        return vec![0.0; 10];
    }

    let mut bins = [0u32; 10];
    let mut total = 0u32;

    for y in 1..height as isize - 1 {
        for x in 1..width as isize - 1 {
            let center = pixel_at(&gray, width, height, x, y);
            let neighbors = [
                pixel_at(&gray, width, height, x, y - 1),
                pixel_at(&gray, width, height, x + 1, y - 1),
                pixel_at(&gray, width, height, x + 1, y),
                pixel_at(&gray, width, height, x + 1, y + 1),
                pixel_at(&gray, width, height, x, y + 1),
                pixel_at(&gray, width, height, x - 1, y + 1),
                pixel_at(&gray, width, height, x - 1, y),
                pixel_at(&gray, width, height, x - 1, y - 1),
            ];

            let mut pattern: u8 = 0;
            for (i, &n) in neighbors.iter().enumerate() {
                if n >= center {
                    pattern |= 1 << i;
                }
            }

            let ri_pattern = ri_lbp(pattern);
            let transitions = count_transitions(ri_pattern);
            if transitions <= 2 {
                let ones = ri_pattern.count_ones() as usize;
                bins[ones] += 1;
            } else {
                bins[9] += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        return vec![0.0; 10];
    }
    bins.iter()
        .map(|&b| b as f64 / total as f64)
        .collect()
}

/// Find the rotation that gives the minimum 8-bit value (rotation-invariant).
fn ri_lbp(pattern: u8) -> u8 {
    let mut min_val = pattern;
    let mut p = pattern;
    for _ in 1..8 {
        p = p.rotate_right(1);
        if p < min_val {
            min_val = p;
        }
    }
    min_val
}

/// Count 0→1 or 1→0 transitions in circular 8-bit pattern.
fn count_transitions(pattern: u8) -> u32 {
    let mut count = 0;
    for i in 0..8 {
        let bit1 = (pattern >> i) & 1;
        let bit2 = (pattern >> ((i + 1) % 8)) & 1;
        if bit1 != bit2 {
            count += 1;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// 2c-c: GLCM on 64×64 thumbnail
// ---------------------------------------------------------------------------

/// Downscale grayscale image to fit within `max_dim` along long edge.
fn downscale_gray(gray: &[f64], width: usize, height: usize, max_dim: usize) -> Vec<f64> {
    let scale = if width >= height {
        max_dim as f64 / width as f64
    } else {
        max_dim as f64 / height as f64
    };

    let new_w = (width as f64 * scale) as usize;
    let new_h = (height as f64 * scale) as usize;
    let new_w = new_w.max(1);
    let new_h = new_h.max(1);

    let mut result = Vec::with_capacity(new_w * new_h);
    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x as f64 / scale) as usize;
            let src_y = (y as f64 / scale) as usize;
            let src_x = src_x.min(width - 1);
            let src_y = src_y.min(height - 1);
            result.push(gray[src_y * width + src_x]);
        }
    }
    result
}

/// Quantize f64 values into `levels` gray levels (0..levels-1).
fn quantize(values: &[f64], levels: usize) -> Vec<usize> {
    let min_val = values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-12 {
        return vec![0; values.len()];
    }

    values
        .iter()
        .map(|&v| {
            let q = ((v - min_val) / range * (levels - 1) as f64) as usize;
            q.min(levels - 1)
        })
        .collect()
}

/// GLCM features: contrast, correlation, energy, homogeneity.
/// Computed on a downscaled thumbnail (max 64px), 5-pixel offset, 4 directions, averaged.
pub fn glcm_features(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    let thumb = downscale_gray(&gray, width, height, 64);
    // Derive tw from actual thumb dimensions to avoid f64 rounding mismatch
    let scale = 64.0 / width.max(height) as f64;
    let tw = (width as f64 * scale) as usize;
    let tw = tw.max(1);
    let th = thumb.len() / tw;
    let th = th.max(1);
    // guard against any residual mismatch
    let q = quantize(&thumb[..tw * th], 16);

    let offsets: [(isize, isize); 4] = [(5, 0), (0, 5), (5, 5), (-5, 5)];

    let mut contrast_sum = 0.0;
    let mut corr_sum = 0.0;
    let mut energy_sum = 0.0;
    let mut homo_sum = 0.0;
    let mut dirs_with_data = 0u32;

    for &(dx, dy) in &offsets {
        let mut glcm = vec![0u32; 16 * 16];
        let mut count = 0u32;

        for y in 0..th as isize {
            for x in 0..tw as isize {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || nx >= tw as isize || ny < 0 || ny >= th as isize {
                    continue;
                }
                let i = q[y as usize * tw + x as usize];
                let j = q[ny as usize * tw + nx as usize];
                glcm[i * 16 + j] += 1;
                count += 1;
            }
        }

        if count < 50 {
            continue;
        }
        dirs_with_data += 1;

        let total = count as f64;
        let mut p = vec![0.0f64; 16 * 16];
        for k in 0..256 {
            p[k] = glcm[k] as f64 / total;
        }

        // Marginal probabilities
        let mut px = [0.0f64; 16];
        let mut py = [0.0f64; 16];
        for i in 0..16 {
            for j in 0..16 {
                px[i] += p[i * 16 + j];
                py[j] += p[i * 16 + j];
            }
        }

        // Means and std devs
        let mut mu_x = 0.0;
        let mut mu_y = 0.0;
        for k in 0..16 {
            mu_x += k as f64 * px[k];
            mu_y += k as f64 * py[k];
        }

        let mut sx = 0.0;
        let mut sy = 0.0;
        for k in 0..16 {
            sx += (k as f64 - mu_x) * (k as f64 - mu_x) * px[k];
            sy += (k as f64 - mu_y) * (k as f64 - mu_y) * py[k];
        }

        // Contrast
        let mut contrast = 0.0;
        for i in 0..16 {
            for j in 0..16 {
                let diff = (i as f64 - j as f64) * (i as f64 - j as f64);
                contrast += diff * p[i * 16 + j];
            }
        }
        contrast_sum += contrast;

        // Correlation
        let mut corr = 0.0;
        if sx > 1e-12 && sy > 1e-12 {
            for i in 0..16 {
                for j in 0..16 {
                    corr += (i as f64 - mu_x) * (j as f64 - mu_y) * p[i * 16 + j];
                }
            }
            corr /= sqrt(sx * sy);
        }
        corr_sum += corr;

        // Energy
        let mut energy = 0.0;
        for k in 0..256 {
            energy += p[k] * p[k];
        }
        energy_sum += energy;

        // Homogeneity
        let mut homo = 0.0;
        for i in 0..16 {
            for j in 0..16 {
                homo += p[i * 16 + j] / (1.0 + (i as f64 - j as f64).abs());
            }
        }
        homo_sum += homo;
    }

    let d = dirs_with_data.max(1) as f64;
    alloc::vec![
        contrast_sum / d,
        corr_sum / d,
        energy_sum / d,
        homo_sum / d,
    ]
}

// ---------------------------------------------------------------------------
// 2c-d: Fractal dimension (box-counting on edge map)
// ---------------------------------------------------------------------------

/// Approximate fractal dimension via box-counting on edge map.
pub fn fractal_dimension(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);

    // Simple edge map: threshold gradient magnitude
    let mut edge_map = vec![false; width * height];
    for y in 1..height as isize - 1 {
        for x in 1..width as isize - 1 {
            let gx = pixel_at(&gray, width, height, x + 1, y)
                - pixel_at(&gray, width, height, x - 1, y);
            let gy = pixel_at(&gray, width, height, x, y + 1)
                - pixel_at(&gray, width, height, x, y - 1);
            let mag = sqrt(gx * gx + gy * gy);
            edge_map[y as usize * width + x as usize] = mag > 10.0;
        }
    }

    // Box counting at multiple scales
    let scales = [2, 4, 8, 16, 32, 64];
    let mut log_scales = Vec::new();
    let mut log_counts = Vec::new();

    for &box_size in &scales {
        if box_size > width || box_size > height {
            continue;
        }
        let bx = width / box_size;
        let by = height / box_size;
        let mut box_count = 0u32;

        for by_i in 0..by {
            for bx_i in 0..bx {
                let mut has_edge = false;
                for dy in 0..box_size {
                    for dx in 0..box_size {
                        let idx =
                            (by_i * box_size + dy) * width + (bx_i * box_size + dx);
                        if edge_map[idx] {
                            has_edge = true;
                            break;
                        }
                    }
                    if has_edge {
                        break;
                    }
                }
                if has_edge {
                    box_count += 1;
                }
            }
        }

        if box_count > 0 {
            log_scales.push(log2(1.0 / box_size as f64));
            log_counts.push(log2(box_count as f64));
        }
    }

    if log_scales.len() < 2 {
        return 0.0;
    }

    // Linear regression: slope = fractal dimension
    let n = log_scales.len() as f64;
    let sum_x: f64 = log_scales.iter().sum();
    let sum_y: f64 = log_counts.iter().sum();
    let sum_xy: f64 = log_scales
        .iter()
        .zip(log_counts.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_xx: f64 = log_scales.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn solid_128() -> Vec<u8> {
        let mut p = Vec::with_capacity(8 * 8 * 3);
        for _ in 0..64 {
            p.push(128);
            p.push(128);
            p.push(128);
        }
        p
    }

    #[test]
    fn dct_low_freq_solid() {
        let r = dct_low_freq_ratio(&solid_128(), 8, 8);
        // Solid image: DC component dominates → high low-freq ratio
        assert!(r > 0.8);
    }

    #[test]
    fn lbp_histogram_solid() {
        let h = lbp_histogram(&solid_128(), 8, 8);
        assert_eq!(h.len(), 10);
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // normalised
        // Solid image: all neighbors >= center → all bits = 1 → pattern 0xFF
        // RI of 0xFF = 0xFF, 8 ones, 0 transitions → bin 8
        assert!(h[8] > 0.9);
    }

    #[test]
    fn glcm_features_solid() {
        let f = glcm_features(&solid_128(), 8, 8);
        assert_eq!(f.len(), 4);
        // Solid image: zero contrast, max homogeneity, energy=1
        assert!(f[0] < 0.001); // contrast ~ 0
        assert!(f[3] > 0.99);  // homogeneity ~ 1
    }

    #[test]
    fn fractal_dimension_solid() {
        let fd = fractal_dimension(&solid_128(), 8, 8);
        // Flat image has few edges → FD near 0 or small
        assert!(fd >= 0.0);
        assert!(fd < 3.0);
    }

    #[test]
    fn fractal_dimension_noise() {
        // Noisy image should have higher FD
        let mut noisy = Vec::with_capacity(16 * 16 * 3);
        for i in 0..256 {
            let v = ((i * 37 + 13) % 256) as u8;
            noisy.push(v);
            noisy.push(v);
            noisy.push(v);
        }
        let fd = fractal_dimension(&noisy, 16, 16);
        assert!(fd > 0.0);
    }
}
