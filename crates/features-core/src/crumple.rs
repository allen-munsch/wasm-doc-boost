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
// 2f: Crumple detection features
// ---------------------------------------------------------------------------

/// LBP variance: compute LBP for each pixel, weight bins by local variance.
/// Returns LBP histogram bin values weighted by local variance.
pub fn lbp_variance(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    if width < 3 || height < 3 {
        return alloc::vec![0.0; 10];
    }

    let mut weighted_bins = [0.0f64; 10];
    let mut total_weight = 0.0;

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

            // Local variance in 3x3 neighborhood
            let mut local_sum = center;
            for &n in &neighbors {
                local_sum += n;
            }
            let local_mean = local_sum / 9.0;
            let mut local_var = center - local_mean;
            local_var = local_var * local_var;
            for &n in &neighbors {
                let d = n - local_mean;
                local_var += d * d;
            }
            local_var /= 9.0;
            let weight = sqrt(local_var);

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
                weighted_bins[ones] += weight;
            } else {
                weighted_bins[9] += weight;
            }
            total_weight += weight;
        }
    }

    if total_weight < 1e-10 {
        return alloc::vec![0.0; 10];
    }
    weighted_bins
        .iter()
        .map(|&b| b / total_weight)
        .collect()
}

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

/// Edge density standard deviation across 8×8 grid cells.
pub fn edge_density_std(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let cell_w = (width / 8).max(1);
    let cell_h = (height / 8).max(1);

    let mut densities = Vec::new();

    for cy in 0..8 {
        for cx in 0..8 {
            let start_x = cx * cell_w;
            let start_y = cy * cell_h;
            let end_x = ((cx + 1) * cell_w).min(width);
            let end_y = ((cy + 1) * cell_h).min(height);

            let mut edge_count = 0u32;
            let mut total = 0u32;

            for y in start_y..end_y {
                for x in start_x..end_x {
                    if x > 0 && x < width - 1 && y > 0 && y < height - 1 {
                        let gx = pixel_at(&gray, width, height, x as isize + 1, y as isize)
                            - pixel_at(&gray, width, height, x as isize - 1, y as isize);
                        let gy = pixel_at(&gray, width, height, x as isize, y as isize + 1)
                            - pixel_at(&gray, width, height, x as isize, y as isize - 1);
                        let mag = sqrt(gx * gx + gy * gy);
                        if mag > 20.0 {
                            edge_count += 1;
                        }
                    }
                    total += 1;
                }
            }

            if total > 0 {
                densities.push(edge_count as f64 / total as f64);
            }
        }
    }

    if densities.len() < 2 {
        return 0.0;
    }

    let mean: f64 = densities.iter().sum::<f64>() / densities.len() as f64;
    let var: f64 = densities
        .iter()
        .map(|&d| (d - mean) * (d - mean))
        .sum::<f64>()
        / densities.len() as f64;
    sqrt(var)
}

/// Texture anisotropy: ratio of GLCM contrast in vertical vs. horizontal direction.
pub fn texture_anisotropy(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);

    // Quantize to 16 levels
    let min_val = gray.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = gray.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-12 {
        return 1.0;
    }

    let q: Vec<usize> = gray
        .iter()
        .map(|&v| ((v - min_val) / range * 15.0) as usize)
        .collect();

    let contrast_h = glcm_contrast(&q, width, height, 5, 0);
    let contrast_v = glcm_contrast(&q, width, height, 0, 5);

    if contrast_h < 1e-12 {
        return if contrast_v > 0.0 { 10.0 } else { 1.0 };
    }
    contrast_v / contrast_h
}

fn glcm_contrast(
    q: &[usize],
    width: usize,
    height: usize,
    dx: isize,
    dy: isize,
) -> f64 {
    let mut glcm = [0u32; 16 * 16];
    let mut count = 0u32;

    for y in 0..height as isize {
        for x in 0..width as isize {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || nx >= width as isize || ny < 0 || ny >= height as isize {
                continue;
            }
            let i = q[y as usize * width + x as usize];
            let j = q[ny as usize * width + nx as usize];
            glcm[i * 16 + j] += 1;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let total = count as f64;
    let mut contrast = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let p = glcm[i * 16 + j] as f64 / total;
            let diff = (i as f64 - j as f64) * (i as f64 - j as f64);
            contrast += diff * p;
        }
    }
    contrast
}

/// Peak local entropy: divide image into small tiles, compute entropy per tile,
/// return max/min ratio.
pub fn peak_local_entropy(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let tile_size = (width.min(height) / 8).max(4);

    let tiles_x = width / tile_size;
    let tiles_y = height / tile_size;
    if tiles_x < 2 || tiles_y < 2 {
        return 1.0;
    }

    let mut entropies = Vec::new();

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut hist = [0u32; 256];
            let mut count = 0u32;

            for dy in 0..tile_size {
                for dx in 0..tile_size {
                    let x = tx * tile_size + dx;
                    let y = ty * tile_size + dy;
                    if x < width && y < height {
                        let idx = (gray[y * width + x].clamp(0.0, 255.0)) as usize;
                        hist[idx.min(255)] += 1;
                        count += 1;
                    }
                }
            }

            let n = count as f64;
            let mut entropy = 0.0;
            for &c in &hist {
                if c > 0 {
                    let p = c as f64 / n;
                    entropy -= p * libm::log2(p);
                }
            }
            entropies.push(entropy);
        }
    }

    let min_e = entropies
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .max(0.001);
    let max_e = entropies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    max_e / min_e
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
    fn lbp_variance_solid() {
        let v = lbp_variance(&solid_128(16, 16), 16, 16);
        assert_eq!(v.len(), 10);
        let sum: f64 = v.iter().sum();
        // Solid image: all weights zero → sum = 0
        assert!(sum < 0.01, "sum={sum}, v={v:?}");
    }

    #[test]
    fn edge_density_std_solid() {
        let s = edge_density_std(&solid_128(16, 16), 16, 16);
        // Solid image → uniform edge density → std ≈ 0
        assert!(s < 0.01);
    }

    #[test]
    fn texture_anisotropy_solid() {
        let a = texture_anisotropy(&solid_128(16, 16), 16, 16);
        // Solid image → near-isotropic → ratio ≈ 1
        assert!((a - 1.0).abs() < 0.01);
    }

    #[test]
    fn peak_local_entropy_solid() {
        let r = peak_local_entropy(&solid_128(32, 32), 32, 32);
        // Solid image: all tiles have zero entropy → max = 0, min = 0.001, ratio = 0
        assert!(r < 0.1);
    }
}
