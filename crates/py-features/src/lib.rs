use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pdf_inspector_core::{PdfType, types::ItemType};

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

/// Classify a PDF from raw bytes.
///
/// Returns a dict with keys: pdf_type (str), page_count (int),
/// pages_needing_ocr (list[int]), confidence (float).
#[pyfunction]
fn classify_pdf(data: Vec<u8>) -> PyResult<PyObject> {
    let result = pdf_inspector_core::classify_pdf_mem(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        let pdf_type = match result.pdf_type {
            PdfType::TextBased => "TextBased",
            PdfType::Scanned => "Scanned",
            PdfType::ImageBased => "ImageBased",
            PdfType::Mixed => "Mixed",
        };
        dict.set_item("pdf_type", pdf_type)?;
        dict.set_item("page_count", result.page_count)?;
        dict.set_item("pages_needing_ocr", result.pages_needing_ocr)?;
        dict.set_item("confidence", result.confidence as f64)?;
        Ok(dict.into())
    })
}

/// Extract text items with positions from a PDF memory buffer.
///
/// Returns a list of dicts, each with keys: text, x, y, width, height,
/// font, font_size, page, is_bold, is_italic, is_underline, is_strikeout,
/// item_type.
#[pyfunction]
fn extract_text_with_positions(data: Vec<u8>) -> PyResult<PyObject> {
    let items = pdf_inspector_core::extract_text_with_positions_mem(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let list = PyList::empty(py);
        for item in items {
            let dict = PyDict::new(py);
            let item_type = match item.item_type {
                ItemType::Text => "text",
                ItemType::Image => "image",
                ItemType::Link(_) => "link",
                ItemType::FormField => "formField",
            };
            dict.set_item("text", item.text)?;
            dict.set_item("x", item.x)?;
            dict.set_item("y", item.y)?;
            dict.set_item("width", item.width)?;
            dict.set_item("height", item.height)?;
            dict.set_item("font", item.font)?;
            dict.set_item("font_size", item.font_size)?;
            dict.set_item("page", item.page)?;
            dict.set_item("is_bold", item.is_bold)?;
            dict.set_item("is_italic", item.is_italic)?;
            dict.set_item("is_underline", item.is_underline)?;
            dict.set_item("is_strikeout", item.is_strikeout)?;
            dict.set_item("item_type", item_type)?;
            list.append(dict)?;
        }
        Ok(list.into())
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_all, m)?)?;
    m.add_function(wrap_pyfunction!(classify_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(extract_text_with_positions, m)?)?;
    Ok(())
}
