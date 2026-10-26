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
pub mod ink;
pub mod noise;
pub mod projection;
pub mod radon;
pub mod shadow;
pub mod textline;
pub mod texture;

/// Extract the 100 pre-rotation pixel heuristics from raw RGB bytes.
///
/// Shared by `extract_all` (backward-compatible 103 = 100 + 3 zero-pad) and
/// `extract_all_v2` (116 = 100 + 16 rotation signals for the per-degree
/// rework).
fn extract_core(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
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

    features.push(crate::noise::high_pass_residual_variance(
        &gray, width, height,
    ));
    features.extend(crate::noise::jpeg_blockiness(&gray, width, height));
    features.push(crate::noise::gradient_snr(&gray, width, height));

    features.extend(crate::shadow::shadow_features(&gray, width, height));

    features.extend(crate::crumple::lbp_variance(&gray, width, height));
    features.push(crate::crumple::edge_density_std(&gray, width, height));
    features.push(crate::crumple::texture_anisotropy(&gray, width, height));
    features.push(crate::crumple::peak_local_entropy(&gray, width, height));

    features.extend(crate::document::document_features(pixels, width, height));
    features.extend(crate::projection::projection_features(&gray, width, height));
    features.extend(crate::ink::ink_features(&gray, width, height));

    features
}

/// The 16 Phase 1 rotation signals (per-degree rework).
///
/// Radon is computed once; the deterministic axis drives the textline deskew,
/// so the axis and flip heads share a single Radon pass. Order:
/// radon [axis_rad, peakiness, peak_margin, concentration] (4), textline flip
/// [flip_score, vote_frac, n_lines, asc_frac, desc_frac, body_frac] (6),
/// boundary [long_edge_rad, area_frac, aspect, boundary_visible] (4),
/// unfolded docstrum [mean_angle_rad, concentration] (2).
///
/// The axis is chosen by multi-candidate selection: the top-4 local maxima of
/// the Radon curve are each scored by text-line quality (n_lines, then total
/// line ink), and the best wins. This fixes the near-tie failure where a
/// perpendicular structure (table rules, page edge) out-peaks the text axis
/// and the bare argmax lands 90 degrees off.
pub fn rotation_features(gray: &[f64], pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let mut features = Vec::new();
    let curve = crate::radon::radon_curve(gray, width, height);
    let axes = crate::radon::radon_top_axes(&curve, 4);
    let mut best_axis = axes[0];
    let mut best_quality = (0.0f64, 0.0f64);
    for (k, ax) in axes.iter().enumerate() {
        let (n_lines, ink) = crate::textline::textline_line_quality(gray, width, height, *ax);
        if k == 0 || n_lines > best_quality.0 || (n_lines == best_quality.0 && ink > best_quality.1)
        {
            best_axis = *ax;
            best_quality = (n_lines, ink);
        }
    }
    let (peakiness, peak_margin, concentration) = crate::radon::radon_summary(&curve);
    features.push(best_axis);
    features.push(peakiness);
    features.push(peak_margin);
    features.push(concentration);
    features.extend(crate::textline::textline_flip_features_with_axis(
        gray, width, height, best_axis,
    ));
    features.extend(crate::boundary::boundary_features(pixels, width, height));
    features.extend(crate::ink::docstrum_axis(gray, width, height));
    features
}

/// Extract all 103 pixel heuristics from raw RGB bytes.
///
/// Backward-compatible with the pre-rework model: 100 real features + 3
/// zero-padded to match the dimensionality of `data/features_v3.npz`. The
/// per-degree rework switches `py-features`/`wasm-bridge` to `extract_all_v2`.
///
/// This is the single canonical feature vector for the CURRENT model — both
/// `py-features` (training export) and `wasm-bridge` (inference) delegate to
/// this function.
pub fn extract_all(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let mut features = extract_core(pixels, width, height);

    // Pad to 103 features to match data/features_v3.npz dimensionality.
    // The 3 padding features (indices 100-102) are set to 0.0 — trees that
    // split on them will always take the left branch, which is harmless.
    features.extend([0.0_f64; 3]);

    features
}

/// Extract the rework feature vector: 100 heuristics + 16 rotation signals.
///
/// Canonical for the per-degree model (Phase 2 export + Phase 4 WASM).
pub fn extract_all_v2(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let gray = crate::to_grayscale(pixels, width, height);
    let mut features = extract_core(pixels, width, height);
    features.extend(rotation_features(&gray, pixels, width, height));
    features
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
        assert_eq!(extract_all(&px, 64, 64).len(), 103);
        assert_eq!(extract_all_v2(&px, 64, 64).len(), 116);
        assert_eq!(
            rotation_features(&to_grayscale(&px, 64, 64), &px, 64, 64).len(),
            16
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
