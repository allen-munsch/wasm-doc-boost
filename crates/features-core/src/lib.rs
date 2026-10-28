#![no_std]

extern crate alloc;

use alloc::vec::Vec;

/// Convert RGB pixels to grayscale f64 values (BT.601 luminance).
/// This is THE single grayscale conversion — all modules share it.
pub fn to_grayscale(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
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

/// Get grayscale pixel value at (x, y), clamping to border.
#[inline]
pub fn pixel_at(gray: &[f64], width: usize, _height: usize, x: isize, y: isize) -> f64 {
    let x = x.clamp(0, width as isize - 1) as usize;
    let y = y.clamp(0, _height as isize - 1) as usize;
    gray[y * width + x]
}

pub mod boundary;
pub mod color;
pub mod crumple;
pub mod document;
pub mod edges;
pub mod glyph;
pub mod ink;
pub mod noise;
pub mod projection;
pub mod radon;
pub mod shadow;
pub mod textline;
pub mod texture;

/// Noise + shadow features: the 9 core indices 55..=63 (noise
/// high_pass_residual_variance, blockiness_h, blockiness_v, gradient_snr;
/// shadow dark_ratio, illum_var, shadow_depth_mean, shadow_depth_std,
/// shadow_edge). Kept as one function so the hybrid-resolution path can
/// recompute them at full resolution — k=4 averaging destroys high-frequency
/// noise and shadow structure (blockiness collapses to 0, crumpled/shadow F1
/// drops below target).
pub fn extract_noise_shadow(gray: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut features = Vec::with_capacity(9);
    let residual_var = crate::noise::high_pass_residual_variance(gray, width, height);
    features.push(residual_var);
    features.extend(crate::noise::jpeg_blockiness(gray, width, height));
    features.push(crate::noise::gradient_snr_given_residual(
        gray, width, height, residual_var,
    ));
    features.extend(crate::shadow::shadow_features(gray, width, height));
    features
}

/// Index of the first noise feature in the core 100-feature vector.
/// color 19 + edges 20 + texture 16 = 55, then noise 4 (55..58) + shadow 5
/// (59..63). Asserted by `noise_shadow_index_contract` below.
pub const NOISE_SHADOW_START: usize = 55;

/// Extract the 100 pre-rotation pixel heuristics from raw RGB bytes.
///
/// Shared by `extract_all` (backward-compatible 103 = 100 + 3 zero-pad) and
/// `extract_all` (116 = 100 + 16 rotation signals for the per-degree
/// rework).
pub fn extract_core(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = crate::to_grayscale(pixels, width, height);
    let mut features = Vec::new();

    features.extend(crate::color::per_channel_stats(pixels, width, height));
    features.extend(crate::color::grayscale_stats(pixels, width, height));
    features.push(crate::color::colorfulness(pixels, width, height));
    features.extend(crate::color::saturation_stats(pixels, width, height));

    let sobel = crate::edges::sobel(&gray, width, height);
    features.push(crate::edges::laplacian_variance(&gray, width, height));
    features.extend(crate::edges::sobel_stats(&sobel));
    features.push(crate::edges::edge_density(&sobel));
    features.extend(crate::edges::edge_direction_histogram(&sobel));
    features.push(crate::edges::hv_edge_ratio(&sobel));
    features.push(crate::edges::canny_edge_density(&sobel));
    features.extend(crate::edges::structure_tensor_features(&sobel));
    features.extend(crate::edges::sobel_circular_stats(&sobel));

    features.push(crate::texture::dct_low_freq_ratio(&gray, width, height));
    features.extend(crate::texture::lbp_histogram(&gray, width, height));
    features.extend(crate::texture::glcm_features(&gray, width, height));
    features.push(crate::texture::fractal_dimension(&gray, width, height));

    features.extend(crate::extract_noise_shadow(&gray, width, height));

    features.extend(crate::crumple::lbp_variance(&gray, width, height));
    features.push(crate::crumple::edge_density_std(&gray, width, height));
    features.push(crate::crumple::texture_anisotropy(&gray, width, height));
    features.push(crate::crumple::peak_local_entropy(&gray, width, height));

    features.extend(crate::document::document_features_gray(
        &gray, pixels, width, height,
    ));
    features.extend(crate::projection::projection_features(&gray, width, height));
    features.extend(crate::ink::ink_features(&gray, width, height));

    features
}

/// The 24 rotation signals (16 Phase 1 + 8 Phase 1b flip features).
///
/// Radon is computed once; the deterministic axis drives the textline deskew,
/// so the axis and flip heads share a single Radon pass. The 8 new flip
/// features are APPENDED after the Phase 1 block, so indices 0..116 stay
/// byte-for-byte backward compatible with the 116-feature model. Order:
/// radon [axis_rad, peakiness, peak_margin, concentration] (4), textline flip
/// [flip_score, vote_frac, n_lines, asc_frac, desc_frac, body_frac] (6),
/// boundary [long_edge_rad, area_frac, aspect, boundary_visible] (4),
/// unfolded docstrum [mean_angle_rad, concentration] (2), tranche A gradient
/// [signed_vert_grad, baseline_edge_weight, gy_skew, top_bottom_edge_sharpness]
/// (4), tranche B character [char_centroid_offset, char_topheavy_frac,
/// char_asym_strength, body_vert_bias] (4).
///
/// The axis is chosen by multi-candidate selection: the top-4 local maxima of
/// the Radon curve are each scored by text-line quality (n_lines, then total
/// line ink), and the best wins. This fixes the near-tie failure where a
/// perpendicular structure (table rules, page edge) out-peaks the text axis
/// and the bare argmax lands 90 degrees off.
/// Rotation extraction result: the 24 rotation features plus the winner axis
/// and the per-glyph classifier crops from the same single winner deskew.
pub struct RotationExtract {
    pub features: Vec<f64>,
    pub axis_rad: f64,
    pub glyph_crops: Vec<f32>,
    pub glyph_count: usize,
}

pub fn rotation_features(gray: &[f64], pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    rotation_extract(gray, pixels, width, height).features
}

pub fn rotation_extract(
    gray: &[f64],
    pixels: &[u8],
    width: usize,
    height: usize,
) -> RotationExtract {
    let curve = crate::radon::radon_curve(gray, width, height);
    let axes = crate::radon::radon_top_axes(&curve, 4);
    // S0/S1: pick the coarse winner by text-line quality.
    let coarse = crate::textline::textline_flip_best_axis(gray, width, height, &axes);
    // S2: refine the coarse axis on an Otsu ink mask (fine projection search).
    let refined = crate::radon::fine_axis(gray, width, height, coarse.axis);
    // S3: flip features + glyph crops from ONE deskew at the refined axis.
    let best = crate::textline::textline_flip_at_axis(gray, width, height, refined);
    let (peakiness, peak_margin, concentration) = crate::radon::radon_summary(&curve);
    let mut features = Vec::new();
    features.push(best.axis);
    features.push(peakiness);
    features.push(peak_margin);
    features.push(concentration);
    features.extend_from_slice(&best.features[0..6]);
    features.extend(crate::boundary::boundary_features(pixels, width, height));
    features.extend(crate::ink::docstrum_axis(gray, width, height));
    features.extend_from_slice(&best.features[6..14]);
    RotationExtract {
        features,
        axis_rad: best.axis,
        glyph_crops: best.glyph_crops,
        glyph_count: best.glyph_count,
    }
}

/// Extract the canonical 124-feature vector: 100 core heuristics + 24 rotation signals.
///
/// Canonical for the per-degree model (Phase 2 export + Phase 4 WASM).
pub fn extract_all(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = crate::to_grayscale(pixels, width, height);
    let mut features = extract_core(pixels, width, height);
    features.extend(rotation_features(&gray, pixels, width, height));
    features
}

/// Extract the rework feature vector plus the per-glyph flip evidence from
/// the same winner deskew: `(features, glyph_sum_logodds, glyph_count)`.
/// Positive glyph_sum_logodds = flipped, negative = upright.
pub fn extract_all_with_glyph(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f64>, f64, f64) {
    let gray = crate::to_grayscale(pixels, width, height);
    let mut features = extract_core(pixels, width, height);
    let rot = rotation_extract(&gray, pixels, width, height);
    features.extend(rot.features);
    let (sum_logodds, count) = crate::glyph::glyph_logodds_from_crops(&rot.glyph_crops);
    (features, sum_logodds, count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rgb(w: usize, h: usize) -> Vec<u8> {
        let mut px = alloc::vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                px[i] = (x * 7 % 256) as u8;
                px[i + 1] = (y * 13 % 256) as u8;
                px[i + 2] = ((x + y) * 5 % 256) as u8;
            }
        }
        px
    }

    #[test]
    fn feature_count_contract() {
        let px = sample_rgb(64, 64);
        assert_eq!(extract_all(&px, 64, 64).len(), 124);
        assert_eq!(
            rotation_features(&to_grayscale(&px, 64, 64), &px, 64, 64).len(),
            24
        );
    }

    #[test]
    fn noise_shadow_index_contract() {
        // extract_core must place extract_noise_shadow exactly at
        // NOISE_SHADOW_START..NOISE_SHADOW_START+9, so the hybrid path can
        // splice full-res noise/shadow into the k=4 core vector.
        let px = sample_rgb(128, 128);
        let gray = to_grayscale(&px, 128, 128);
        let core = extract_core(&px, 128, 128);
        let ns = extract_noise_shadow(&gray, 128, 128);
        assert_eq!(ns.len(), 9);
        assert_eq!(
            &core[NOISE_SHADOW_START..NOISE_SHADOW_START + ns.len()],
            &ns[..]
        );
    }

    /// 12 horizontal "text" lines (6px tall, gray, 80px wide, 4px gaps),
    /// returned as (gray, rgb).
    fn text_lines(w: usize, h: usize) -> (Vec<f64>, Vec<u8>) {
        let mut gray = alloc::vec![255.0f64; w * h];
        for &y0 in &[8usize, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118] {
            for y in y0..(y0 + 6).min(h) {
                for x in 24..104 {
                    gray[y * w + x] = 80.0;
                }
            }
        }
        let mut px = alloc::vec![0u8; w * h * 3];
        for i in 0..w * h {
            let v = gray[i].clamp(0.0, 255.0) as u8;
            px[i * 3] = v;
            px[i * 3 + 1] = v;
            px[i * 3 + 2] = v;
        }
        (gray, px)
    }

    #[test]
    fn textline_line_quality_prefers_text_axis() {
        // The discriminator behind multi-candidate selection: at 0 deg the
        // text lines are distinct bands; at 90 deg they merge into one column
        // mass. Line quality must be strictly higher at the text axis.
        let (gray, _) = text_lines(128, 128);
        let (n0, ink0) = crate::textline::textline_line_quality(&gray, 128, 128, 0.0);
        let (n90, _) =
            crate::textline::textline_line_quality(&gray, 128, 128, core::f64::consts::FRAC_PI_2);
        assert!(n0 > n90, "n_lines at 0 deg ({n0}) must beat 90 deg ({n90})");
        assert!(ink0 > 0.0);
    }

    #[test]
    fn rotation_features_keeps_horizontal_text_axis() {
        // Regression guard: with a clear horizontal text axis, the
        // multi-candidate selection must keep it (no 90-deg flip).
        let (gray, px) = text_lines(128, 128);
        let feats = rotation_features(&gray, &px, 128, 128);
        let axis_deg = feats[0].to_degrees().rem_euclid(180.0);
        assert!(
            axis_deg < 5.0 || axis_deg > 175.0,
            "axis should be ~0 deg, got {axis_deg}"
        );
    }
}
