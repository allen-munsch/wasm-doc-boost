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
// 2g: Document vs. natural image features
// ---------------------------------------------------------------------------

/// Otsu threshold on grayscale values [0,255].
fn otsu_threshold(gray: &[f64]) -> f64 {
    let mut hist = [0u32; 256];
    for &v in gray {
        let idx = (v.clamp(0.0, 255.0)) as usize;
        hist[idx.min(255)] += 1;
    }

    let total = gray.len() as f64;
    let mut sum_all = 0.0;
    for i in 0..256 {
        sum_all += i as f64 * hist[i] as f64;
    }

    let mut best_thresh = 128.0;
    let mut best_variance = 0.0;
    let mut w_b = 0.0;
    let mut sum_b = 0.0;

    for i in 0..256 {
        w_b += hist[i] as f64;
        if w_b < 1.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f < 1.0 {
            break;
        }
        sum_b += i as f64 * hist[i] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best_variance {
            best_variance = between;
            best_thresh = i as f64;
        }
    }
    best_thresh
}

/// Connected component analysis on binary image.
/// Returns Vec of (pixel_count, x_min, x_max, y_min, y_max) for each component.
fn connected_components(
    binary: &[bool],
    width: usize,
    height: usize,
) -> Vec<(u32, usize, usize, usize, usize)> {
    let n = width * height;
    let mut labels = vec![-1i32; n];
    let mut components: Vec<(u32, usize, usize, usize, usize)> = Vec::new();

    let mut next_label = 0i32;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !binary[idx] || labels[idx] != -1 {
                continue;
            }

            // Flood fill
            let mut stack = vec![(x, y)];
            labels[idx] = next_label;
            let mut count = 0u32;
            let mut x_min = x;
            let mut x_max = x;
            let mut y_min = y;
            let mut y_max = y;

            while let Some((cx, cy)) = stack.pop() {
                count += 1;
                x_min = x_min.min(cx);
                x_max = x_max.max(cx);
                y_min = y_min.min(cy);
                y_max = y_max.max(cy);

                for &(dx, dy) in &[(0isize, 1isize),(0,-1),(1,0),(-1,0)] {
                    let nx = cx as isize + dx;
                    let ny = cy as isize + dy;
                    if nx >= 0
                        && nx < width as isize
                        && ny >= 0
                        && ny < height as isize
                    {
                        let nidx = ny as usize * width + nx as usize;
                        if binary[nidx] && labels[nidx] == -1 {
                            labels[nidx] = next_label;
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }
            }

            components.push((count, x_min, x_max, y_min, y_max));
            next_label += 1;
        }
    }

    components
}

/// Document vs. natural image features.
/// Returns:
/// - component_count: number of connected components after Otsu binarization
/// - median_aspect_ratio: median aspect ratio (width/height) of components
/// - median_stroke_width: median stroke width (area / perimeter approx)
/// - pct_small_aspect: % of small, high-aspect-ratio, near-h/v components
/// - dominant_line_count: simplified Hough line count
/// - saturation_edge_ratio: ratio of saturation in low-edge vs high-edge regions
pub fn document_features(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = to_grayscale(pixels, width, height);
    let threshold = otsu_threshold(&gray);

    // Binarize: foreground = dark pixels (text typically dark on light)
    let binary: Vec<bool> = gray.iter().map(|&v| v <= threshold).collect();

    let comps = connected_components(&binary, width, height);

    let comp_count = comps.len() as f64;

    // Median aspect ratio
    let mut aspect_ratios: Vec<f64> = Vec::new();
    let mut stroke_widths: Vec<f64> = Vec::new();
    let mut small_aspect_count = 0u32;
    let mut valid_count = 0u32;

    for &(count, x_min, x_max, y_min, y_max) in &comps {
        if count < 3 {
            continue;
        }
        valid_count += 1;
        let w = (x_max - x_min + 1) as f64;
        let h = (y_max - y_min + 1) as f64;
        let aspect = if h > 0.0 { w / h } else { 1.0 };
        aspect_ratios.push(aspect);

        // Stroke width approx: area / perimeter
        // Perimeter ≈ 2*(w+h) or use actual edge pixel count
        let perimeter = (2.0 * (w + h)).max(1.0);
        stroke_widths.push(count as f64 / perimeter);

        // Small, high-aspect-ratio, near-horizontal/vertical
        if count < 100 && (aspect > 3.0 || aspect < 0.33) {
            small_aspect_count += 1;
        }
    }

    let median_aspect = median(&aspect_ratios);
    let median_stroke = median(&stroke_widths);
    let pct_small_aspect = if valid_count > 0 {
        small_aspect_count as f64 / valid_count as f64
    } else {
        0.0
    };

    // Simplified Hough: count dominant lines via accumulator peaks
    let hough_lines = simplified_hough(&binary, width, height);

    // Saturation in low-edge vs. high-edge regions
    let sat_ratio = saturation_edge_ratio(pixels, width, height);

    alloc::vec![
        comp_count,
        median_aspect,
        median_stroke,
        pct_small_aspect,
        hough_lines,
        sat_ratio,
    ]
}

/// Median of a slice.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 && mid > 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Simplified Hough transform on a 64×64 accumulator.
/// Returns dominant line count (peaks above threshold).
fn simplified_hough(binary: &[bool], width: usize, height: usize) -> f64 {
    let acc_w = 64usize;
    let acc_h = 64usize;
    let mut accumulator = vec![0u32; acc_w * acc_h];
    let diag = sqrt((width * width + height * height) as f64);
    let rho_max = diag;
    let d_rho = 2.0 * rho_max / acc_h as f64;
    let d_theta = core::f64::consts::PI / acc_w as f64;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !binary[idx] {
                continue;
            }
            for t_idx in 0..acc_w {
                let theta = t_idx as f64 * d_theta;
                let rho = x as f64 * libm::cos(theta) + y as f64 * libm::sin(theta);
                let r_idx = ((rho + rho_max) / d_rho) as usize;
                if r_idx < acc_h {
                    accumulator[r_idx * acc_w + t_idx] += 1;
                }
            }
        }
    }

    // Count peaks: cells with value > 80% of max
    let max_val = *accumulator.iter().max().unwrap_or(&0);
    if max_val < 3 {
        return 0.0;
    }
    let threshold = (max_val as f64 * 0.5) as u32;
    let mut peak_count = 0u32;

    // Non-max suppression: count local maxima above threshold
    for r in 1..acc_h - 1 {
        for c in 1..acc_w - 1 {
            let v = accumulator[r * acc_w + c];
            if v < threshold {
                continue;
            }
            let is_peak = v >= accumulator[(r - 1) * acc_w + c]
                && v >= accumulator[(r + 1) * acc_w + c]
                && v >= accumulator[r * acc_w + (c - 1)]
                && v >= accumulator[r * acc_w + (c + 1)]
                && v >= accumulator[(r - 1) * acc_w + (c - 1)]
                && v >= accumulator[(r - 1) * acc_w + (c + 1)]
                && v >= accumulator[(r + 1) * acc_w + (c - 1)]
                && v >= accumulator[(r + 1) * acc_w + (c + 1)];
            if is_peak {
                peak_count += 1;
            }
        }
    }

    peak_count as f64
}

/// Ratio of mean saturation in low-edge vs. high-edge regions.
fn saturation_edge_ratio(pixels: &[u8], width: usize, height: usize) -> f64 {
    let gray = to_grayscale(pixels, width, height);
    let n = width * height;

    // Compute edge magnitude per pixel (simple gradient)
    let mut edge_mags = vec![0.0f64; n];
    for y in 1..height as isize - 1 {
        for x in 1..width as isize - 1 {
            let gx = pixel_at(&gray, width, height, x + 1, y)
                - pixel_at(&gray, width, height, x - 1, y);
            let gy = pixel_at(&gray, width, height, x, y + 1)
                - pixel_at(&gray, width, height, x, y - 1);
            edge_mags[y as usize * width + x as usize] = sqrt(gx * gx + gy * gy);
        }
    }

    let edge_threshold = edge_mags
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        * 0.3;

    let mut low_edge_sat_sum = 0.0;
    let mut low_edge_count = 0u32;
    let mut high_edge_sat_sum = 0.0;
    let mut high_edge_count = 0u32;

    for i in 0..n {
        let r = pixels[i * 3] as f64;
        let g = pixels[i * 3 + 1] as f64;
        let b = pixels[i * 3 + 2] as f64;
        let max_val = r.max(g).max(b);
        let min_val = r.min(g).min(b);
        let sat = if max_val > 0.0 {
            (max_val - min_val) / max_val
        } else {
            0.0
        };

        if edge_mags[i] < edge_threshold {
            low_edge_sat_sum += sat;
            low_edge_count += 1;
        } else {
            high_edge_sat_sum += sat;
            high_edge_count += 1;
        }
    }

    let low_avg = if low_edge_count > 0 {
        low_edge_sat_sum / low_edge_count as f64
    } else {
        0.0
    };
    let high_avg = if high_edge_count > 0 {
        high_edge_sat_sum / high_edge_count as f64
    } else {
        0.0
    };

    if high_avg < 1e-12 {
        return 1.0;
    }
    low_avg / high_avg
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
    fn document_features_solid() {
        let f = document_features(&solid_128(32, 32), 32, 32);
        assert_eq!(f.len(), 6);
        // Solid gray: no foreground after Otsu
        assert!(f[0] < 10.0); // few components
    }

    #[test]
    fn document_features_text_like() {
        // 32x32 image: white background with a few dark "text" blobs
        let w = 32;
        let h = 32;
        let mut pixels = vec![];
        for y in 0..h {
            for x in 0..w {
                // Dark horizontal strokes at y=8 and y=16
                let is_text = (y == 8 && x >= 4 && x <= 12)
                    || (y == 8 && x >= 18 && x <= 26)
                    || (y == 16 && x >= 6 && x <= 14)
                    || (y == 16 && x >= 20 && x <= 28);
                if is_text {
                    pixels.push(0);
                    pixels.push(0);
                    pixels.push(0);
                } else {
                    pixels.push(240);
                    pixels.push(240);
                    pixels.push(240);
                }
            }
        }
        let f = document_features(&pixels, w, h);
        // Should detect connected components (text-like)
        assert!(f[0] > 0.0); // component count > 0
        assert!(f[3] > 0.0); // % small high-aspect-ratio components > 0
    }
}
