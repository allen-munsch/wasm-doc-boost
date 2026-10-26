//! Text-line morphology for the 0-vs-180 flip head.
//!
//! The axis head (radon) estimates the text-line angle modulo 180 degrees:
//! it cannot tell an upright page from an upside-down one. Latin text can:
//! ascenders (b d f h k l t, and acute accents) rise above the x-height band
//! while descenders (g j p q y, cedilla) hang below the baseline. Upright text
//! has ink mass above the x-height band; flipped text has ink mass above it
//! too, but that mass is the former descenders. So the sign of
//! (asc - desc) disambiguates the two flips.
//!
//! We deskew the ink mask by the axis estimate, find text lines via the
//! horizontal projection, estimate each line's x-height band from the mode of
//! its column ink heights, then accumulate ink above/below that band.
//!
//! no_std; nearest-neighbour deskew (same sampling as radon.rs). Kept out of
//! `extract_all` until the Phase 3 retrain (103-feature contract).
//!
//! Reference: Bloomberg et al. 1995 (Leptonica pixOrientDetect); van
//! Beusekom/Shafait/Breuel 2010 (geometric text-line model).

use crate::ink::otsu_threshold;
use crate::radon::radon_features;
use alloc::vec::Vec;

/// Flip features with the axis computed internally (standalone / tests / bench).
pub fn textline_flip_features(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let axis_rad = radon_features(gray, width, height)[0];
    textline_flip_features_with_axis(gray, width, height, axis_rad)
}

/// Flip features given the text-line axis (radians, mod pi) from the axis head.
///
/// Returns 6 features:
///   0. flip_score    — signed ascender/descender evidence in [-1, 1];
///                      positive = upright, negative = flipped.
///   1. vote_frac     — fraction of text lines whose asc > desc.
///   2. n_lines       — number of valid text lines (coverage signal).
///   3. asc_mass_frac — ascender ink / total line ink.
///   4. desc_mass_frac— descender ink / total line ink.
///   5. body_mass_frac— x-height body ink / total line ink (low when the page
///                      is all caps/digits, i.e. the flip is unreliable).
/// Per-line ink statistics for one deskew axis.
struct LineStats {
    asc_total: usize,
    desc_total: usize,
    body_total: usize,
    n_valid: usize,
    n_upright: usize,
}

/// Deskew the ink mask by `axis_rad` and accumulate per-line ascender/
/// descender/body ink. Shared by the flip-feature extractor and the
/// line-quality scorer used for multi-candidate axis selection.
fn analyze_lines(gray: &[f64], width: usize, height: usize, axis_rad: f64) -> LineStats {
    let mut stats = LineStats {
        asc_total: 0,
        desc_total: 0,
        body_total: 0,
        n_valid: 0,
        n_upright: 0,
    };
    if width == 0 || height == 0 {
        return stats;
    }
    let threshold = otsu_threshold(gray);
    let mask: Vec<bool> = gray.iter().map(|&v| v <= threshold).collect();
    // radon's axis is the text-line angle mod 180; the text line angle is
    // (180 - axis_deg) mod 180, so deskewing by axis_deg - 180 brings text
    // lines to 0 deg WITHOUT a 180 flip (fold-free away from the 0/180 wrap:
    // for every applied angle A the deskew cancels A exactly, so flip_score
    // is positive for A<180 and negative for A>=180). Near the 0/180 wrap the
    // radon refinement lands at 0.2 vs 179.8 for the same horizontal text;
    // snap that fold region to 0 deg so the near-horizontal case never gets
    // a spurious ~180 deg flip.
    let axis_deg = axis_rad.to_degrees() % 180.0;
    let deskew_deg = if axis_deg < 1.5 || axis_deg > 178.5 {
        0.0
    } else {
        axis_deg - 180.0
    };
    let (deskewed, size) = rotate_mask_square(&mask, width, height, deskew_deg);
    let bands = find_line_bands(&deskewed, size, size);

    for &(y0, y1) in &bands {
        let (asc, desc, body) = line_asc_desc(&deskewed, size, y0, y1);
        if body == 0 {
            continue;
        }
        stats.asc_total += asc;
        stats.desc_total += desc;
        stats.body_total += body;
        stats.n_valid += 1;
        if asc > desc {
            stats.n_upright += 1;
        }
    }
    stats
}

pub fn textline_flip_features_with_axis(
    gray: &[f64],
    width: usize,
    height: usize,
    axis_rad: f64,
) -> Vec<f64> {
    if width == 0 || height == 0 {
        return alloc::vec![0.0; 6];
    }
    let s = analyze_lines(gray, width, height, axis_rad);
    let flip_score = if s.asc_total + s.desc_total > 0 {
        (s.asc_total as f64 - s.desc_total as f64) / (s.asc_total + s.desc_total) as f64
    } else {
        0.0
    };
    let vote_frac = if s.n_valid > 0 {
        s.n_upright as f64 / s.n_valid as f64
    } else {
        0.5
    };
    let ink_total = s.asc_total + s.desc_total + s.body_total;
    let (asc_frac, desc_frac, body_frac) = if ink_total > 0 {
        (
            s.asc_total as f64 / ink_total as f64,
            s.desc_total as f64 / ink_total as f64,
            s.body_total as f64 / ink_total as f64,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    alloc::vec![
        flip_score,
        vote_frac,
        s.n_valid as f64,
        asc_frac,
        desc_frac,
        body_frac,
    ]
}

/// Line-quality score for an axis candidate: `(n_valid_lines, total_line_ink)`.
///
/// A correct axis deskews text lines horizontally and yields many tall bands
/// with substantial ink; a wrong (perpendicular) axis yields few fragmented
/// bands. Used by `rotation_features` to pick among the radon curve's local
/// maxima when the projection response is ambiguous (a near-tie between the
/// text axis and a perpendicular structure such as table rules).
pub fn textline_line_quality(
    gray: &[f64],
    width: usize,
    height: usize,
    axis_rad: f64,
) -> (f64, f64) {
    if width == 0 || height == 0 {
        return (0.0, 0.0);
    }
    let s = analyze_lines(gray, width, height, axis_rad);
    let ink = (s.asc_total + s.desc_total + s.body_total) as f64;
    (s.n_valid as f64, ink)
}

/// Pad the mask to a max(w,h) square (paper = false) and rotate by `+deg`
/// (same direction as PIL's rotate(+deg), nearest neighbour, border-clamped).
/// Returns the square mask and its side length. Padding keeps every rotation
/// inside the sampling canvas — the same reason radon.rs pads — so a landscape
/// page deskewed by 90 degrees is not cropped to a strip before the flip
/// measurement.
fn rotate_mask_square(mask: &[bool], width: usize, height: usize, deg: f64) -> (Vec<bool>, usize) {
    let size = width.max(height);
    if size == 0 {
        return (Vec::new(), 0);
    }
    let ox = (size - width) / 2;
    let oy = (size - height) / 2;
    let mut padded = alloc::vec![false; size * size];
    for y in 0..height {
        for x in 0..width {
            padded[(y + oy) * size + (x + ox)] = mask[y * width + x];
        }
    }

    let rad = deg.to_radians();
    let sin_t = libm::sin(rad);
    let cos_t = libm::cos(rad);
    let c = (size as f64 - 1.0) / 2.0;
    let mut out = alloc::vec![false; size * size];
    for y in 0..size {
        let yc = y as f64 - c;
        for x in 0..size {
            let xc = x as f64 - c;
            let xs = xc * cos_t - yc * sin_t + c;
            let ys = xc * sin_t + yc * cos_t + c;
            let xi = (xs + 0.5).clamp(0.0, size as f64 - 1.0) as usize;
            let yi = (ys + 0.5).clamp(0.0, size as f64 - 1.0) as usize;
            out[y * size + x] = padded[yi * size + xi];
        }
    }
    (out, size)
}

/// Text-line bands: maximal runs of rows that contain at least one ink pixel.
/// Inter-line spacing is truly blank, so a single threshold of 1 separates
/// lines; short specks are dropped later by the band-height filter.
fn find_line_bands(mask: &[bool], width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut row_ink = alloc::vec![0u32; height];
    for y in 0..height {
        let mut c = 0u32;
        for x in 0..width {
            if mask[y * width + x] {
                c += 1;
            }
        }
        row_ink[y] = c;
    }
    let mut bands = Vec::new();
    let mut y = 0usize;
    while y < height {
        if row_ink[y] > 0 {
            let y0 = y;
            let mut y1 = y + 1;
            while y1 < height && row_ink[y1] > 0 {
                y1 += 1;
            }
            bands.push((y0, y1));
            y = y1;
        } else {
            y += 1;
        }
    }
    bands
}

/// Ink mass above and below a text line's x-height band, plus the body mass.
///
/// x-height = mode (2px bins) of per-column ink heights. Columns whose height
/// sits inside [3/4, 5/4] of that mode define the body: their median top is
/// the x-height top, their median bottom is the baseline.
fn line_asc_desc(mask: &[bool], width: usize, y0: usize, y1: usize) -> (usize, usize, usize) {
    let band_h = y1 - y0;
    if band_h < 4 {
        return (0, 0, 0);
    }
    let mut tops = Vec::with_capacity(width);
    let mut bottoms = Vec::with_capacity(width);
    for x in 0..width {
        let mut top = usize::MAX;
        let mut bottom = usize::MAX;
        for y in y0..y1 {
            if mask[y * width + x] {
                if top == usize::MAX {
                    top = y;
                }
                bottom = y;
            }
        }
        if top != usize::MAX {
            tops.push(top);
            bottoms.push(bottom);
        }
    }
    let n_cols = tops.len();
    if n_cols < 4 {
        return (0, 0, 0);
    }

    let mut heights = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        heights.push(bottoms[i] - tops[i] + 1);
    }
    let x_height = binned_mode(&heights, 2);
    if x_height < 3 {
        return (0, 0, 0);
    }

    let lo = x_height * 3 / 4;
    let hi = x_height * 5 / 4;
    let mut body_tops = Vec::with_capacity(n_cols);
    let mut body_bottoms = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        let hgt = bottoms[i] - tops[i] + 1;
        if hgt >= lo && hgt <= hi {
            body_tops.push(tops[i]);
            body_bottoms.push(bottoms[i]);
        }
    }
    let (body_top, body_bottom) = if body_tops.is_empty() {
        // All-caps or uniform-height line: no x-height band. Median top and
        // bottom are the body, so asc/desc come out ~0 and the line is neutral.
        (median_usize(&tops), median_usize(&bottoms))
    } else {
        (median_usize(&body_tops), median_usize(&body_bottoms))
    };

    let mut asc = 0usize;
    let mut desc = 0usize;
    let mut body = 0usize;
    for x in 0..width {
        for y in y0..y1 {
            if mask[y * width + x] {
                if y < body_top {
                    asc += 1;
                } else if y > body_bottom {
                    desc += 1;
                } else {
                    body += 1;
                }
            }
        }
    }
    (asc, desc, body)
}

/// Mode of `values` quantized into `bin_width`-sized bins, returned as the
/// bin centre. Tolerates the 1-2px height noise of real text columns.
fn binned_mode(values: &[usize], bin_width: usize) -> usize {
    if values.is_empty() {
        return 0;
    }
    let max = values.iter().cloned().max().unwrap_or(0);
    let n_bins = max / bin_width + 1;
    let mut counts = alloc::vec![0u32; n_bins];
    for &v in values {
        counts[v / bin_width] += 1;
    }
    let mut best_bin = 0usize;
    let mut best_count = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        if c > best_count {
            best_count = c;
            best_bin = i;
        }
    }
    best_bin * bin_width + bin_width / 2
}

/// Median of a usize slice via a count histogram (values are small: pixel rows).
fn median_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let min = values.iter().cloned().min().unwrap_or(0);
    let max = values.iter().cloned().max().unwrap_or(0);
    let mut counts = alloc::vec![0u32; max - min + 1];
    for &v in values {
        counts[v - min] += 1;
    }
    let half = (values.len() + 1) / 2;
    let mut acc = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        acc += c;
        if acc as usize >= half {
            return min + i;
        }
    }
    min
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One synthetic text line: x-height body (60 cols), ascenders above
    /// (20 cols), descenders below (10 cols), and a word gap (10 cols).
    /// 0 = ink, 255 = paper.
    fn synthetic_line() -> (Vec<f64>, usize, usize) {
        let w = 100usize;
        let h = 50usize;
        let mut gray = alloc::vec![255.0f64; w * h];
        for x in 0..100usize {
            if x >= 90 {
                continue; // word gap
            }
            let (y0, y1) = if x < 60 {
                (20usize, 30usize) // x-height
            } else if x < 80 {
                (10usize, 30usize) // ascender (b/d/f/h/k/l/t)
            } else {
                (20usize, 40usize) // descender (g/j/p/q/y)
            };
            for y in y0..y1 {
                gray[y * w + x] = 0.0;
            }
        }
        (gray, w, h)
    }

    fn flip180(gray: &[f64], w: usize, h: usize) -> Vec<f64> {
        let mut out = alloc::vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = gray[(h - 1 - y) * w + (w - 1 - x)];
            }
        }
        out
    }

    #[test]
    fn flip_score_upright_vs_flipped() {
        let (gray, w, h) = synthetic_line();
        let f_up = textline_flip_features_with_axis(&gray, w, h, 0.0);
        let flipped = flip180(&gray, w, h);
        let f_down = textline_flip_features_with_axis(&flipped, w, h, 0.0);
        // Upright: ascenders (20 cols) outnumber descenders (10 cols).
        assert!(f_up[0] > 0.15, "upright flip_score={}", f_up[0]);
        assert!(f_down[0] < -0.15, "flipped flip_score={}", f_down[0]);
        // The two are near-negatives of each other (the signal is the flip).
        assert!(
            (f_up[0] + f_down[0]).abs() < 0.15,
            "asymmetry: {} vs {}",
            f_up[0],
            f_down[0]
        );
        assert!(f_up[1] > 0.5, "vote_frac={}", f_up[1]);
        assert_eq!(f_up[2], 1.0, "n_lines={}", f_up[2]);
        assert_eq!(f_up.len(), 6);
    }

    #[test]
    fn all_caps_line_is_neutral() {
        // One band where every column has the same height: no asc/desc.
        let w = 60usize;
        let h = 30usize;
        let mut gray = alloc::vec![255.0f64; w * h];
        for x in 0..w {
            for y in 8..26 {
                gray[y * w + x] = 0.0;
            }
        }
        let f = textline_flip_features_with_axis(&gray, w, h, 0.0);
        assert!(f[0].abs() < 0.05, "all-caps flip_score={}", f[0]);
        assert!(f[5] > 0.9, "body_mass_frac={}", f[5]);
    }

    #[test]
    fn blank_image_returns_neutral() {
        let w = 32usize;
        let h = 32usize;
        let gray = alloc::vec![255.0f64; w * h];
        let f = textline_flip_features(&gray, w, h);
        assert_eq!(f.len(), 6);
        assert!(f[0].abs() < 1e-12, "flip_score={}", f[0]);
        assert_eq!(f[2], 0.0);
    }
}
