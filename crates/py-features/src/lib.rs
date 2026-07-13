use pyo3::prelude::*;

/// Extract all features from raw RGB pixel bytes.
///
/// Args:
///     pixels: Raw RGB bytes (length must be width * height * 3)
///     width: Image width in pixels
///     height: Image height in pixels
///
/// Returns:
///     List of ~78 float features
#[pyfunction]
fn extract_all(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Vec<f64>> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected, width, height, pixels.len()
        )));
    }

    let mut features = Vec::new();

    features.extend(features_core::color::per_channel_stats(&pixels, width, height));
    features.extend(features_core::color::grayscale_stats(&pixels, width, height));
    features.push(features_core::color::colorfulness(&pixels, width, height));
    features.extend(features_core::color::saturation_stats(&pixels, width, height));

    features.push(features_core::edges::laplacian_variance(&pixels, width, height));
    features.extend(features_core::edges::sobel_stats(&pixels, width, height));
    features.push(features_core::edges::edge_density(&pixels, width, height));
    features.extend(features_core::edges::edge_direction_histogram(&pixels, width, height));
    features.push(features_core::edges::hv_edge_ratio(&pixels, width, height));
    features.push(features_core::edges::canny_edge_density(&pixels, width, height));

    features.push(features_core::texture::dct_low_freq_ratio(&pixels, width, height));
    features.extend(features_core::texture::lbp_histogram(&pixels, width, height));
    features.extend(features_core::texture::glcm_features(&pixels, width, height));
    features.push(features_core::texture::fractal_dimension(&pixels, width, height));

    features.push(features_core::noise::high_pass_residual_variance(&pixels, width, height));
    features.extend(features_core::noise::jpeg_blockiness(&pixels, width, height));
    features.push(features_core::noise::gradient_snr(&pixels, width, height));

    features.extend(features_core::shadow::shadow_features(&pixels, width, height));

    features.extend(features_core::crumple::lbp_variance(&pixels, width, height));
    features.push(features_core::crumple::edge_density_std(&pixels, width, height));
    features.push(features_core::crumple::texture_anisotropy(&pixels, width, height));
    features.push(features_core::crumple::peak_local_entropy(&pixels, width, height));

    features.extend(features_core::document::document_features(&pixels, width, height));

    Ok(features)
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_all, m)?)?;
    Ok(())
}
