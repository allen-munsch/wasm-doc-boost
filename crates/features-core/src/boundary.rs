//! Page-boundary axis vote: minimum-area rotated rectangle of the page mask.
//!
//! A receipt or invoice photographed against a contrasting background has a
//! visible page boundary. The long edge of the minimum-area rotated rectangle
//! of the page mask carries the page's principal axis (mod 180). For a
//! landscape receipt that is the text-line axis; for a portrait invoice it is
//! perpendicular to it — the GBDT fuses this vote with radon, which already
//! distinguishes 0 from 90 degrees. On full-bleed crops the page mask is empty
//! and `boundary_visible` is 0, so the model learns to ignore the vote.
//!
//! no_std. Kept out of `extract_all` until the Phase 3 retrain
//! (103-feature contract).

use crate::document::connected_components;
use alloc::vec::Vec;

/// Page-boundary orientation features.
///
/// Returns 4 features:
///   0. long_edge_rad — long edge of the min-area rect, in [0, pi).
///   1. area_frac     — page mask area / image area.
///   2. aspect_ratio  — long/short side of the min-area rect (>= 1).
///   3. boundary_visible — 1.0 when a page component was found, else 0.0.
pub fn boundary_features(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    if width == 0 || height == 0 {
        return alloc::vec![0.0; 4];
    }
    let (br, bg, bb) = median_border_color(pixels, width, height);
    let mask = page_mask(pixels, width, height, (br, bg, bb));

    // Largest connected component of the page mask = the page.
    let comps = connected_components(&mask, width, height);
    let largest = comps.iter().max_by_key(|c| c.0);
    let Some(&(count, x0, x1, y0, y1)) = largest else {
        return alloc::vec![0.0, 0.0, 1.0, 0.0];
    };

    let area_frac = count as f64 / (width * height) as f64;
    let boundary_visible = if area_frac >= 0.01 { 1.0 } else { 0.0 };
    if boundary_visible == 0.0 {
        return alloc::vec![0.0, area_frac, 1.0, 0.0];
    }

    // Collect page points inside the component bbox on a stride-2 grid.
    let mut points = Vec::new();
    for y in (y0..=y1).step_by(2) {
        for x in (x0..=x1).step_by(2) {
            if mask[y * width + x] {
                points.push((x, y));
            }
        }
    }

    let hull = convex_hull(&points);
    if hull.len() < 4 {
        return alloc::vec![0.0, area_frac, 1.0, 0.0];
    }
    let (long_edge_rad, _, w, h) = min_area_rect(&hull);
    let aspect = if w.min(h) > 1e-6 {
        (w.max(h) / w.min(h)).clamp(1.0, 100.0)
    } else {
        1.0
    };

    alloc::vec![long_edge_rad, area_frac, aspect, boundary_visible]
}

/// Median color of the outer frame (the scanner bed / desk background).
fn median_border_color(pixels: &[u8], width: usize, height: usize) -> (u8, u8, u8) {
    let t = (width.min(height) / 40).max(1);
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
    let mut push = |x: usize, y: usize| {
        let i = (y * width + x) * 3;
        rs.push(pixels[i]);
        gs.push(pixels[i + 1]);
        bs.push(pixels[i + 2]);
    };
    for y in 0..t.min(height) {
        for x in 0..width {
            push(x, y);
        }
    }
    for y in (height - t.min(height))..height {
        for x in 0..width {
            push(x, y);
        }
    }
    for y in 0..height {
        for x in 0..t.min(width) {
            push(x, y);
        }
        for x in (width - t.min(width))..width {
            push(x, y);
        }
    }
    (median_u8(&rs), median_u8(&gs), median_u8(&bs))
}

/// Page mask: true where the pixel differs from the border background.
fn page_mask(pixels: &[u8], width: usize, height: usize, bg: (u8, u8, u8)) -> Vec<bool> {
    let mut mask = alloc::vec![false; width * height];
    for i in 0..width * height {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;
        let dr = r - bg.0 as i32;
        let dg = g - bg.1 as i32;
        let db = b - bg.2 as i32;
        let dmax = dr.abs().max(dg.abs()).max(db.abs());
        if dmax > 25 {
            mask[i] = true;
        }
    }
    mask
}

fn median_u8(values: &[u8]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let mut hist = [0u32; 256];
    for &v in values {
        hist[v as usize] += 1;
    }
    let half = values.len().div_ceil(2);
    let mut acc = 0u32;
    for (i, &c) in hist.iter().enumerate() {
        acc += c;
        if acc as usize >= half {
            return i as u8;
        }
    }
    0
}

/// Andrew monotone-chain convex hull, returned counter-clockwise.
fn convex_hull(points: &[(usize, usize)]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    }
    let mut pts: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Minimum-area rotated rectangle by rotating calipers.
///
/// Returns (long_edge_angle_rad, area, width, height) where width/height are
/// the min-area rect's side lengths and the long-edge angle is in [0, pi).
fn min_area_rect(hull: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let n = hull.len();
    let mut best_area = f64::INFINITY;
    let mut best_angle = 0.0f64;
    let mut best_w = 0.0f64;
    let mut best_h = 0.0f64;
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        if dx.abs() < 1e-12 && dy.abs() < 1e-12 {
            continue;
        }
        let angle = libm::atan2(dy, dx);
        let cos_t = libm::cos(angle);
        let sin_t = libm::sin(angle);
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &(x, y) in hull {
            let rx = x * cos_t + y * sin_t;
            let ry = -x * sin_t + y * cos_t;
            min_x = min_x.min(rx);
            max_x = max_x.max(rx);
            min_y = min_y.min(ry);
            max_y = max_y.max(ry);
        }
        let w = max_x - min_x;
        let h = max_y - min_y;
        let area = w * h;
        if area < best_area {
            best_area = area;
            best_w = w;
            best_h = h;
            best_angle = angle;
        }
    }
    // If the bounding box is taller than wide, the long edge is the side
    // perpendicular to the hull edge we aligned with.
    let long_angle = if best_h > best_w {
        best_angle + core::f64::consts::FRAC_PI_2
    } else {
        best_angle
    };
    let mut long_angle = long_angle;
    while long_angle < 0.0 {
        long_angle += core::f64::consts::PI;
    }
    while long_angle >= core::f64::consts::PI {
        long_angle -= core::f64::consts::PI;
    }
    (long_angle, best_area, best_w, best_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dark scanner bed, a white landscape page (200x100) centred in a 300x200
    /// frame, rotated 15 deg CCW by the caller's construction.
    fn rotated_page(deg: f64) -> (Vec<u8>, usize, usize, f64) {
        let w = 300usize;
        let h = 200usize;
        let mut px = alloc::vec![0u8; w * h * 3]; // black bed
        let rad = deg.to_radians();
        let cos_t = libm::cos(rad);
        let sin_t = libm::sin(rad);
        let cx = w as f64 / 2.0;
        let cy = h as f64 / 2.0;
        for y in 0..h {
            for x in 0..w {
                let xc = x as f64 - cx;
                let yc = y as f64 - cy;
                let xs = xc * cos_t + yc * sin_t;
                let ys = -xc * sin_t + yc * cos_t;
                // page local coords within 100x50 half-extent
                if xs.abs() < 100.0 && ys.abs() < 50.0 {
                    let i = (y * w + x) * 3;
                    px[i] = 255;
                    px[i + 1] = 255;
                    px[i + 2] = 255;
                }
            }
        }
        (px, w, h, deg)
    }

    #[test]
    fn boundary_axis_matches_page_rotation() {
        let (px, w, h, deg) = rotated_page(15.0);
        let f = boundary_features(&px, w, h);
        let axis_deg = f[0].to_degrees().rem_euclid(180.0);
        let err = (axis_deg - deg).abs().min((axis_deg - deg - 180.0).abs());
        assert!(err < 2.0, "axis_deg={axis_deg} for deg={deg}");
        assert!(f[3] > 0.5, "boundary_visible={}", f[3]);
        assert!(f[2] > 1.5, "aspect={}", f[2]);
    }

    #[test]
    fn full_bleed_returns_invisible() {
        // Uniform white image: border color == page color, no boundary.
        let w = 64usize;
        let h = 64usize;
        let px = alloc::vec![255u8; w * h * 3];
        let f = boundary_features(&px, w, h);
        assert_eq!(f[3], 0.0, "boundary_visible={}", f[3]);
    }

    #[test]
    fn convex_hull_of_square() {
        let pts = [(0usize, 0usize), (10, 0), (10, 10), (0, 10), (5, 5)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        let (angle, area, w, h) = min_area_rect(&hull);
        assert!((area - 100.0).abs() < 1e-9, "area={area}");
        let _ = (angle, w, h);
    }
}
