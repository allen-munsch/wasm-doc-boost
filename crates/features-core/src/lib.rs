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

pub mod color;
pub mod crumple;
pub mod document;
pub mod edges;
pub mod ink;
pub mod noise;
pub mod projection;
pub mod shadow;
pub mod texture;

/// Extract all 103 pixel heuristics from raw RGB bytes.
///
/// Returns a Vec of 103 f64 features (100 real + 3 zero-padded to match the
/// dimensionality of `data/features_v3.npz`).
///
/// This is the single canonical feature vector — both `py-features` (training
/// export) and `wasm-bridge` (inference) delegate to this function.
pub fn extract_all(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
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

    features.push(crate::noise::high_pass_residual_variance(&gray, width, height));
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

    // Pad to 103 features to match data/features_v3.npz dimensionality.
    // The 3 padding features (indices 100-102) are set to 0.0 — trees that
    // split on them will always take the left branch, which is harmless.
    features.extend([0.0_f64; 3]);

    features
}
