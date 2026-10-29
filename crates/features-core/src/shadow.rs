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
/// Monotonic queue of indices over a sliding window, as a fixed-capacity
/// ring buffer. `head == tail` means empty, so it holds `cap - 1` elements.
struct MonoQueue {
    idx: Vec<usize>,
    head: usize,
    tail: usize,
}

impl MonoQueue {
    fn new(cap: usize) -> Self {
        MonoQueue {
            idx: vec![0usize; cap],
            head: 0,
            tail: 0,
        }
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }
    #[inline]
    fn back(&self) -> usize {
        self.idx[(self.tail + self.idx.len() - 1) % self.idx.len()]
    }
    #[inline]
    fn front(&self) -> usize {
        self.idx[self.head]
    }
    #[inline]
    fn push_back(&mut self, v: usize) {
        self.idx[self.tail] = v;
        self.tail = (self.tail + 1) % self.idx.len();
    }
    #[inline]
    fn pop_back(&mut self) {
        self.tail = (self.tail + self.idx.len() - 1) % self.idx.len();
    }
    #[inline]
    fn pop_front(&mut self) {
        self.head = (self.head + 1) % self.idx.len();
    }
}

/// 1-D sliding-window running maximum (`want_max`) or minimum over
/// `[i - radius, i + radius]` (clamped to the slice ends), written to `out`.
/// O(len) via a monotonic queue instead of the O(len * radius) naive scan.
/// Bit-identical results to the naive per-pixel window scan.
fn sliding_extreme_1d(row: &[f64], out: &mut [f64], radius: usize, want_max: bool) {
    let width = row.len();
    debug_assert_eq!(out.len(), width);
    if width == 0 {
        return;
    }
    let mut q = MonoQueue::new(2 * radius + 4);

    // Seed the first window [0, radius].
    let seed_end = (radius + 1).min(width);
    for idx in 0..seed_end {
        while !q.is_empty() {
            let b = q.back();
            let drops = if want_max {
                row[b] <= row[idx]
            } else {
                row[b] >= row[idx]
            };
            if drops {
                q.pop_back();
            } else {
                break;
            }
        }
        q.push_back(idx);
    }

    for i in 0..width {
        let lo = i.saturating_sub(radius);
        while !q.is_empty() && q.front() < lo {
            q.pop_front();
        }
        out[i] = row[q.front()];
        let add = i + radius + 1;
        if add < width {
            while !q.is_empty() {
                let b = q.back();
                let drops = if want_max {
                    row[b] <= row[add]
                } else {
                    row[b] >= row[add]
                };
                if drops {
                    q.pop_back();
                } else {
                    break;
                }
            }
            q.push_back(add);
        }
    }
}

fn morph_dilate(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = width * height;
    let mut h = vec![0.0; n];
    let mut tmp = vec![0.0; width];
    for y in 0..height {
        let row = y * width;
        sliding_extreme_1d(&gray[row..row + width], &mut tmp, radius, true);
        h[row..row + width].copy_from_slice(&tmp);
    }
    let mut result = vec![0.0; n];
    let mut col_in = vec![0.0; height];
    let mut col_out = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            col_in[y] = h[y * width + x];
        }
        sliding_extreme_1d(&col_in, &mut col_out, radius, true);
        for y in 0..height {
            result[y * width + x] = col_out[y];
        }
    }
    result
}

/// Greyscale erosion with a square structuring element (side 2r+1),
/// computed separably: horizontal running min then vertical running min.
fn morph_erode(gray: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let n = width * height;
    let mut h = vec![0.0; n];
    let mut tmp = vec![0.0; width];
    for y in 0..height {
        let row = y * width;
        sliding_extreme_1d(&gray[row..row + width], &mut tmp, radius, false);
        h[row..row + width].copy_from_slice(&tmp);
    }
    let mut result = vec![0.0; n];
    let mut col_in = vec![0.0; height];
    let mut col_out = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            col_in[y] = h[y * width + x];
        }
        sliding_extreme_1d(&col_in, &mut col_out, radius, false);
        for y in 0..height {
            result[y * width + x] = col_out[y];
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
