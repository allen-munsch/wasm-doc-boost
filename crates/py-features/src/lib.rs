use pyo3::prelude::*;

/// Extract all 103 pixel heuristics from raw RGB bytes (delegates to features-core).
///
/// Args:
///     pixels: Raw RGB bytes (length must be width * height * 3)
///     width: Image width in pixels
///     height: Image height in pixels
///
/// Returns:
///     List of 103 float features (100 real + 3 zero-padded)
#[pyfunction]
fn extract_all(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Vec<f64>> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected, width, height, pixels.len()
        )));
    }

    Ok(features_core::extract_all(&pixels, width, height))
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_all, m)?)?;
    Ok(())
}
