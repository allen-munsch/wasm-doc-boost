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
//! no_std; bilinear deskew with Sauvola local binarization (uneven-lighting tolerant). Kept out of
//! `extract_all` until the Phase 3 retrain (103-feature contract).
//!
//! Reference: Bloomberg et al. 1995 (Leptonica pixOrientDetect); van
//! Beusekom/Shafait/Breuel 2010 (geometric text-line model).

use crate::ink::sauvola_binarize;
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
///
/// - `flip_score` — signed ascender/descender evidence in [-1, 1]; positive = upright, negative = flipped.
/// - `vote_frac` — fraction of text lines whose asc > desc.
/// - `n_lines` — number of valid text lines (coverage signal).
/// - `asc_mass_frac` — ascender ink / total line ink.
/// - `desc_mass_frac` — descender ink / total line ink.
/// - `body_mass_frac` — x-height body ink / total line ink (low when the page is all caps/digits, i.e. the flip is unreliable).
///
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
/// Deskewed grayscale + ink mask + line bands + per-line stats from ONE
/// deskew pass. Gradient and character-weighting flip features share this
/// with the asc/desc extractor so the image is rotated once, not twice.
struct DeskewedLines {
    gray: Vec<f64>,
    mask: Vec<bool>,
    size: usize,
    bands: Vec<(usize, usize)>,
    stats: LineStats,
}

/// Bilinear rotate of grayscale to a max(w,h) square, paper = 255.
/// Bilinear (rather than nearest-neighbour) keeps 1-2px strokes from
/// aliasing away at 512px long-edge export resolution.
fn rotate_gray_square(gray: &[f64], width: usize, height: usize, deg: f64) -> (Vec<f64>, usize) {
    let size = width.max(height);
    if size == 0 {
        return (Vec::new(), 0);
    }
    let ox = (size - width) / 2;
    let oy = (size - height) / 2;
    let mut padded = alloc::vec![255.0f64; size * size];
    for y in 0..height {
        for x in 0..width {
            padded[(y + oy) * size + (x + ox)] = gray[y * width + x];
        }
    }

    let rad = deg.to_radians();
    let sin_t = libm::sin(rad);
    let cos_t = libm::cos(rad);
    let c = (size as f64 - 1.0) / 2.0;
    let last = (size - 1) as f64;
    let mut out = alloc::vec![255.0f64; size * size];
    for y in 0..size {
        let yc = y as f64 - c;
        // xs/ys are linear in x (slope cos_t/sin_t): increment per column
        // rather than rebuilding the affine map per pixel. xsc/ysc are
        // clamped to [0, last] before the cast, so `as usize` truncates
        // toward zero == floor — no libm::floor per pixel.
        let mut xs = c - c * cos_t - yc * sin_t;
        let mut ys = c - c * sin_t + yc * cos_t;
        for x in 0..size {
            let xsc = xs.clamp(0.0, last);
            let ysc = ys.clamp(0.0, last);
            let x0 = xsc as usize;
            let y0 = ysc as usize;
            let fx = xsc - x0 as f64;
            let fy = ysc - y0 as f64;
            let x1 = (x0 + 1).min(size - 1);
            let y1 = (y0 + 1).min(size - 1);
            let v00 = padded[y0 * size + x0];
            let v10 = padded[y0 * size + x1];
            let v01 = padded[y1 * size + x0];
            let v11 = padded[y1 * size + x1];
            out[y * size + x] = (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11;
            xs += cos_t;
            ys += sin_t;
        }
    }
    (out, size)
}

/// Threshold, deskew (one pass), find line bands, and accumulate asc/desc.
fn deskew_analyze(gray: &[f64], width: usize, height: usize, axis_rad: f64) -> DeskewedLines {
    let mut stats = LineStats {
        asc_total: 0,
        desc_total: 0,
        body_total: 0,
        n_valid: 0,
        n_upright: 0,
    };
    if width == 0 || height == 0 {
        return DeskewedLines {
            gray: Vec::new(),
            mask: Vec::new(),
            size: 0,
            bands: Vec::new(),
            stats,
        };
    }
    // radon's axis is the text-line angle mod 180; the text line angle is
    // (180 - axis_deg) mod 180, so deskewing by axis_deg - 180 brings text
    // lines to 0 deg WITHOUT a 180 flip (fold-free away from the 0/180 wrap:
    // for every applied angle A the deskew cancels A exactly, so flip_score
    // is positive for A<180 and negative for A>=180). Near the 0/180 wrap the
    // radon refinement lands at 0.2 vs 179.8 for the same horizontal text;
    // C1: keep one frame across the fold. `signed_tilt_deg` gives the small
    // signed residual inside the band, so we deskew by -tau instead of
    // snapping to 0 and leaving up to 1.5 deg of tilt in the evidence.
    let axis_deg = axis_rad.to_degrees() % 180.0;
    let tau = crate::signed_tilt_deg(axis_deg);
    let deskew_deg = if tau.abs() < crate::FOLD_BAND_DEG {
        -tau
    } else {
        axis_deg - 180.0
    };
    let (gray_sq, size) = rotate_gray_square(gray, width, height, deskew_deg);
    let mask: Vec<bool> = sauvola_binarize(&gray_sq, size, size);
    let bands = find_line_bands(&mask, size, size);

    for &(y0, y1) in &bands {
        let (asc, desc, body) = line_asc_desc(&mask, size, y0, y1);
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
    DeskewedLines {
        gray: gray_sq,
        mask,
        size,
        bands,
        stats,
    }
}

/// Per-line asc/desc stats for one deskew axis (line-quality scorer uses this).
fn analyze_lines(gray: &[f64], width: usize, height: usize, axis_rad: f64) -> LineStats {
    deskew_analyze(gray, width, height, axis_rad).stats
}

pub fn textline_flip_features_with_axis(
    gray: &[f64],
    width: usize,
    height: usize,
    axis_rad: f64,
) -> Vec<f64> {
    if width == 0 || height == 0 {
        return alloc::vec![0.0; 14];
    }
    let d = deskew_analyze(gray, width, height, axis_rad);
    flip_features_from_lines(&d)
}

/// 14 flip features from an ALREADY deskewed line analysis (no re-deskew).
/// Shared by the single-axis path above and the multi-axis selector below so
/// the winner is deskewed exactly once.
fn flip_features_from_lines(d: &DeskewedLines) -> Vec<f64> {
    let s = &d.stats;
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

    let mut feats = alloc::vec![
        flip_score,
        vote_frac,
        s.n_valid as f64,
        asc_frac,
        desc_frac,
        body_frac,
    ];
    let grad = gradient_flip_features(&d.gray, &d.mask, d.size, &d.bands);
    let chars = char_flip_features(&d.mask, d.size, &d.bands);
    feats.extend_from_slice(&grad);
    feats.extend_from_slice(&chars);
    feats
}

/// Best-axis flip analysis: winner axis, 14 flip features, and the per-glyph
/// classifier crops from the SAME winner deskew (no extra rotation pass).
pub struct BestAxisFlip {
    pub axis: f64,
    pub features: Vec<f64>,
    pub glyph_crops: Vec<f32>,
    pub glyph_count: usize,
}

/// Select the best text-line axis among `axes` by (n_valid, total ink), then
/// return the winner's flip features and glyph crops from a single deskew of
/// the winner. This replaces the old rotation_features loop (4x
/// textline_line_quality) + a 5th deskew with exactly one deskew per candidate
/// and no extra pass.
pub fn textline_flip_best_axis(
    gray: &[f64],
    width: usize,
    height: usize,
    axes: &[f64],
) -> BestAxisFlip {
    let mut best_axis = axes[0];
    let mut best_lines: Option<DeskewedLines> = None;
    let mut best_quality = (0.0f64, 0.0f64);
    for ax in axes {
        let d = deskew_analyze(gray, width, height, *ax);
        let n_lines = d.stats.n_valid as f64;
        let ink = (d.stats.asc_total + d.stats.desc_total + d.stats.body_total) as f64;
        let better = best_lines.is_none()
            || n_lines > best_quality.0
            || (n_lines == best_quality.0 && ink > best_quality.1);
        if better {
            best_axis = *ax;
            best_quality = (n_lines, ink);
            best_lines = Some(d);
        }
    }
    let best = best_lines.expect("axes non-empty");
    let features = flip_features_from_lines(&best);
    let glyph_crops = crate::glyph::glyph_crops_from_mask(&best.mask, best.size, &best.bands);
    let glyph_count = glyph_crops.len() / crate::glyph::GLYPH_DIM;
    BestAxisFlip {
        axis: best_axis,
        features,
        glyph_crops,
        glyph_count,
    }
}

/// Nearest-neighbour rotate of grayscale to a max(w,h) square, paper = 255.
/// Cheaper than bilinear; used only for axis-candidate SCORING, where we need
/// line-band structure, not 1-2px stroke fidelity.
fn rotate_gray_square_nn(gray: &[f64], width: usize, height: usize, deg: f64) -> (Vec<f64>, usize) {
    let size = width.max(height);
    if size == 0 {
        return (Vec::new(), 0);
    }
    let ox = (size - width) / 2;
    let oy = (size - height) / 2;
    let mut padded = alloc::vec![255.0f64; size * size];
    for y in 0..height {
        for x in 0..width {
            padded[(y + oy) * size + (x + ox)] = gray[y * width + x];
        }
    }
    let rad = deg.to_radians();
    let sin_t = libm::sin(rad);
    let cos_t = libm::cos(rad);
    let c = (size as f64 - 1.0) / 2.0;
    let last = (size - 1) as f64;
    let mut out = alloc::vec![255.0f64; size * size];
    for y in 0..size {
        let yc = y as f64 - c;
        let mut xs = c - c * cos_t - yc * sin_t;
        let mut ys = c - c * sin_t + yc * cos_t;
        for x in 0..size {
            let xi = (xs + 0.5).clamp(0.0, last) as usize;
            let yi = (ys + 0.5).clamp(0.0, last) as usize;
            out[y * size + x] = padded[yi * size + xi];
            xs += cos_t;
            ys += sin_t;
        }
    }
    (out, size)
}

/// Otsu threshold on a grayscale slice (256-bin histogram).
fn otsu256(gray: &[f64]) -> f64 {
    let mut hist = [0u32; 256];
    for &v in gray {
        let idx = (v.clamp(0.0, 255.0)) as usize;
        hist[idx.min(255)] += 1;
    }
    let total = gray.len() as f64;
    let mut sum_all = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        sum_all += i as f64 * h as f64;
    }
    let mut best_t = 128.0f64;
    let mut best_var = -1.0f64;
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        w_b += h as f64;
        if w_b < 1.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f < 1.0 {
            break;
        }
        sum_b += i as f64 * h as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best_var {
            best_var = between;
            best_t = i as f64;
        }
    }
    best_t
}

/// Cheap candidate-axis line quality: NN-rotate + Otsu threshold + row-run
/// band count. Only the ORDERING among radon's top-4 axes matters, so the
/// cheaper binarization and nearest-neighbour sampling are fine.
pub fn cheap_line_quality(gray: &[f64], width: usize, height: usize, axis_rad: f64) -> (f64, f64) {
    let axis_deg = axis_rad.to_degrees() % 180.0;
    let tau = crate::signed_tilt_deg(axis_deg);
    let deskew_deg = if tau.abs() < crate::FOLD_BAND_DEG {
        -tau
    } else {
        axis_deg - 180.0
    };
    let (rot, size) = rotate_gray_square_nn(gray, width, height, deskew_deg);
    let thr = otsu256(&rot);
    let mut n = 0.0f64;
    let mut ink = 0.0f64;
    let mut y = 0usize;
    while y < size {
        let mut row_ink = 0usize;
        for x in 0..size {
            if rot[y * size + x] <= thr {
                row_ink += 1;
            }
        }
        if row_ink == 0 {
            y += 1;
            continue;
        }
        let y0 = y;
        let mut y1 = y + 1;
        let mut band_ink = row_ink;
        while y1 < size {
            let mut ri = 0usize;
            for x in 0..size {
                if rot[y1 * size + x] <= thr {
                    ri += 1;
                }
            }
            if ri == 0 {
                break;
            }
            band_ink += ri;
            y1 += 1;
        }
        ink += band_ink as f64;
        if y1 - y0 >= 2 {
            n += 1.0;
        }
        y = y1;
    }
    (n, ink)
}

/// Flip analysis at a SPECIFIC axis (S3): one deskew at the refined fine-axis
/// value, then the 14 flip features + per-glyph crops from that single deskew.
///
/// The fine stage S2 refines the coarse radon axis to ~0.1 deg; running the
/// flip evidence on that straightened mask (instead of the 3-deg coarse axis)
/// keeps ascender/descender shape sharp so the glyph/line-vote gate works at
/// any tilt, not just the cardinals.
pub fn textline_flip_at_axis(
    gray: &[f64],
    width: usize,
    height: usize,
    axis_rad: f64,
) -> BestAxisFlip {
    if width == 0 || height == 0 {
        return BestAxisFlip {
            axis: axis_rad,
            features: alloc::vec![0.0; 14],
            glyph_crops: Vec::new(),
            glyph_count: 0,
        };
    }
    let d = deskew_analyze(gray, width, height, axis_rad);
    let features = flip_features_from_lines(&d);
    let glyph_crops = crate::glyph::glyph_crops_from_mask(&d.mask, d.size, &d.bands);
    let glyph_count = glyph_crops.len() / crate::glyph::GLYPH_DIM;
    BestAxisFlip {
        axis: axis_rad,
        features,
        glyph_crops,
        glyph_count,
    }
}

/// Best-axis selection with CHEAP candidate scoring (NN + Otsu at the caller's
/// chosen low resolution) and the full bilinear+Sauvola flip analysis only on
/// the winner. `core_gray` is the low-res image for scoring; `gray` is the
/// higher-res image for the winner's flip features and glyph crops.
pub fn textline_flip_best_axis_cheap(
    core_gray: &[f64],
    cw: usize,
    ch: usize,
    gray: &[f64],
    width: usize,
    height: usize,
    axes: &[f64],
) -> BestAxisFlip {
    let mut best_axis = axes[0];
    let mut best_q = (0.0f64, 0.0f64);
    for ax in axes {
        let (n, ink) = cheap_line_quality(core_gray, cw, ch, *ax);
        if n > best_q.0 || (n == best_q.0 && ink > best_q.1) {
            best_axis = *ax;
            best_q = (n, ink);
        }
    }
    let d = deskew_analyze(gray, width, height, best_axis);
    let features = flip_features_from_lines(&d);
    let glyph_crops = crate::glyph::glyph_crops_from_mask(&d.mask, d.size, &d.bands);
    let glyph_count = glyph_crops.len() / crate::glyph::GLYPH_DIM;
    BestAxisFlip {
        axis: best_axis,
        features,
        glyph_crops,
        glyph_count,
    }
}

/// Tranche A: signed-vertical-gradient flip evidence on the deskewed image,
/// restricted to text-line bands. Flip-antisymmetric under 180 deg:
/// signed_vert_grad (mean signed Sobel-y), baseline_edge_weight (|gy| mass
/// bottom vs top half), gy_skew (skewness of signed gy), and
/// top_bottom_edge_sharpness (baselines are sharper/flatter than cap tops).
fn gradient_flip_features(
    gray: &[f64],
    mask: &[bool],
    size: usize,
    bands: &[(usize, usize)],
) -> [f64; 4] {
    if size == 0 || bands.is_empty() {
        return [0.0; 4];
    }
    let sobel = crate::edges::sobel(gray, size, size);

    let mut sum_gy = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut bottom_abs = 0.0f64;
    let mut top_abs = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0.0f64;
    for &(y0, y1) in bands {
        let mid = (y0 + y1 - 1) as f64 / 2.0; // mid row excluded: antisymmetric under 180 flip
        for y in y0..y1 {
            let row = y * size;
            for x in 0..size {
                let gy = sobel.gy[row + x];
                let a = gy.abs();
                sum_gy += gy;
                sum_abs += a;
                sum2 += gy * gy;
                n += 1.0;
                if (y as f64) < mid {
                    top_abs += a;
                } else if (y as f64) > mid {
                    bottom_abs += a;
                }
            }
        }
    }

    let signed_vert_grad = if sum_abs > 1e-9 {
        sum_gy / sum_abs
    } else {
        0.0
    };
    let baseline_edge_weight = if bottom_abs + top_abs > 1e-9 {
        (bottom_abs - top_abs) / (bottom_abs + top_abs)
    } else {
        0.0
    };

    let gy_skew = if n >= 8.0 && sum_abs > 1e-9 {
        let mu = sum_gy / n;
        let var = sum2 / n - mu * mu;
        if var > 1e-12 {
            let sigma = libm::sqrt(var);
            let mut sum3 = 0.0f64;
            for &(y0, y1) in bands {
                for y in y0..y1 {
                    let row = y * size;
                    for x in 0..size {
                        let d = sobel.gy[row + x] - mu;
                        sum3 += d * d * d;
                    }
                }
            }
            sum3 / (n * sigma * sigma * sigma)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut top_sum = 0.0f64;
    let mut top_n = 0usize;
    let mut bot_sum = 0.0f64;
    let mut bot_n = 0usize;
    for &(y0, y1) in bands {
        for y in y0..y1 {
            let row = y * size;
            for x in 0..size {
                if !mask[row + x] {
                    continue;
                }
                let above_bg = y == 0 || !mask[(y - 1) * size + x];
                let below_bg = y + 1 >= size || !mask[(y + 1) * size + x];
                let a = sobel.gy[row + x].abs();
                if above_bg && !below_bg {
                    top_sum += a;
                    top_n += 1;
                } else if below_bg && !above_bg {
                    bot_sum += a;
                    bot_n += 1;
                }
            }
        }
    }
    let top_bottom_edge_sharpness = if top_n > 0 && bot_n > 0 {
        let mean_top = top_sum / top_n as f64;
        let mean_bot = bot_sum / bot_n as f64;
        let overall = (top_sum + bot_sum) / (top_n + bot_n) as f64;
        if overall > 1e-9 {
            (mean_top - mean_bot) / overall
        } else {
            0.0
        }
    } else {
        0.0
    };

    [
        signed_vert_grad,
        baseline_edge_weight,
        gy_skew,
        top_bottom_edge_sharpness,
    ]
}

/// Iterate glyph bounding boxes (column runs within line bands) from a
/// deskewed ink mask. Shared by `char_flip_features` (tranche B) and the
/// per-glyph 0-vs-180 classifier (`glyph` module): one slicing rule, one
/// source of truth. A run is a maximal span of columns with >=1 ink pixel
/// inside a band; runs with fewer than 6 ink pixels are dropped as specks.
pub(crate) fn for_each_glyph_bbox(
    mask: &[bool],
    size: usize,
    bands: &[(usize, usize)],
    mut f: impl FnMut(usize, usize, usize, usize),
) {
    for &(y0, y1) in bands {
        let band_h = y1 - y0;
        if band_h < 4 {
            continue;
        }
        let mut col_ink = alloc::vec![0u32; size];
        for y in y0..y1 {
            let row = y * size;
            for x in 0..size {
                if mask[row + x] {
                    col_ink[x] += 1;
                }
            }
        }
        let mut x = 0usize;
        while x < size {
            if col_ink[x] == 0 {
                x += 1;
                continue;
            }
            let x0 = x;
            let mut x1 = x + 1;
            while x1 < size && col_ink[x1] > 0 {
                x1 += 1;
            }
            let mut ink = 0usize;
            for xx in x0..x1 {
                ink += col_ink[xx] as usize;
            }
            if ink >= 6 {
                f(x0, x1, y0, y1);
            }
            x = x1;
        }
    }
}

/// Tranche B: per-character vertical-mass asymmetry. Slices each line band
/// into character-like runs at projection gaps and measures where each slice's
/// ink sits (top-heavy vs bottom-heavy) — catches all-caps/digit lines whose
/// x-height asc/desc is ~0.
fn char_flip_features(mask: &[bool], size: usize, bands: &[(usize, usize)]) -> [f64; 4] {
    let mut sum_offset = 0.0f64;
    let mut sum_abs_offset = 0.0f64;
    let mut n_slices = 0usize;
    let mut n_top_heavy = 0usize;
    let mut top_ink = 0usize;
    let mut bottom_ink = 0usize;

    // Band-level top/bottom ink (mid row excluded: antisymmetric under 180).
    for &(y0, y1) in bands {
        let band_h = (y1 - y0) as f64;
        if band_h < 4.0 {
            continue;
        }
        let band_mid = (y0 + y1 - 1) as f64 / 2.0;
        for y in y0..y1 {
            let row = y * size;
            for x in 0..size {
                if mask[row + x] {
                    if (y as f64) < band_mid {
                        top_ink += 1;
                    } else if (y as f64) > band_mid {
                        bottom_ink += 1;
                    }
                }
            }
        }
    }

    // Per-character vertical-mass asymmetry via the shared glyph slicer.
    for_each_glyph_bbox(mask, size, bands, |x0, x1, y0, y1| {
        let band_h = (y1 - y0) as f64;
        let band_mid = (y0 + y1 - 1) as f64 / 2.0;
        let mut slice_ink = 0usize;
        let mut slice_y = 0usize;
        for xx in x0..x1 {
            for y in y0..y1 {
                if mask[y * size + xx] {
                    slice_ink += 1;
                    slice_y += y;
                }
            }
        }
        if slice_ink >= 6 {
            let centroid = slice_y as f64 / slice_ink as f64;
            let offset = (centroid - band_mid) / band_h;
            sum_offset += offset;
            sum_abs_offset += offset.abs();
            n_slices += 1;
            if offset < 0.0 {
                n_top_heavy += 1;
            }
        }
    });

    let char_centroid_offset = if n_slices > 0 {
        sum_offset / n_slices as f64
    } else {
        0.0
    };
    let char_topheavy_frac = if n_slices > 0 {
        n_top_heavy as f64 / n_slices as f64
    } else {
        0.5
    };
    let char_asym_strength = if n_slices > 0 {
        sum_abs_offset / n_slices as f64
    } else {
        0.0
    };
    let body_vert_bias = if top_ink + bottom_ink > 0 {
        (bottom_ink as f64 - top_ink as f64) / (top_ink + bottom_ink) as f64
    } else {
        0.0
    };

    [
        char_centroid_offset,
        char_topheavy_frac,
        char_asym_strength,
        body_vert_bias,
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

// rotate_mask_square removed: dead code (no callers). Its square-pad + NN
// sampling concern is handled by rotate_gray_square, now bilinear.

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
    let half = values.len().div_ceil(2);
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
        assert_eq!(f_up.len(), 14);
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
    fn bilinear_rotate_90_exact() {
        // 2x2 with distinct values; +90 deg maps integer centres exactly, so
        // bilinear must reproduce the four values in their rotated places.
        let gray = alloc::vec![10.0, 20.0, 30.0, 40.0];
        let (out, size) = rotate_gray_square(&gray, 2, 2, 90.0);
        assert_eq!(size, 2);
        let expect = [20.0, 40.0, 10.0, 30.0];
        for (i, &e) in expect.iter().enumerate() {
            assert!(
                (out[i] - e).abs() < 1e-9,
                "out[{}]={} expect={}",
                i,
                out[i],
                e
            );
        }
    }

    #[test]
    fn blank_image_returns_neutral() {
        let w = 32usize;
        let h = 32usize;
        let gray = alloc::vec![255.0f64; w * h];
        let f = textline_flip_features(&gray, w, h);
        assert_eq!(f.len(), 14);
        assert!(f[0].abs() < 1e-12, "flip_score={}", f[0]);
        assert_eq!(f[2], 0.0);
    }
}
