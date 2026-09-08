//! # py_features — Rust-powered feature extraction + PDF inspection for Python
//!
//! **Install (prebuilt wheel, CPython 3.8+):**
//!
//! - Linux x86_64: `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
//! - Linux aarch64: `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
//! - macOS Intel (x86_64): `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-macosx_10_12_x86_64.whl`
//! - macOS Apple Silicon (arm64): `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-macosx_11_0_arm64.whl`
//!
//! **From source (requires Rust):** `cd crates/py-features && pip install maturin && maturin develop --release`
//!
//! ## Python API
//!
//! ```python
//! import py_features
//!
//! # ── Image features ──
//! from PIL import Image
//! import numpy as np
//!
//! img = Image.open("document.jpg").convert("RGB")
//! pixels = np.array(img, dtype=np.uint8).tobytes()
//! feats = py_features.extract_all(pixels, img.width, img.height)
//! # -> list[float]  (length 103: 100 real + 3 zero-padded)
//!
//! # ── PDF classification ──
//! pdf_bytes = open("document.pdf", "rb").read()
//! result = py_features.classify_pdf(pdf_bytes)
//! # -> {'pdf_type': 'TextBased', 'page_count': 7,
//! #     'pages_needing_ocr': [], 'confidence': 1.0}
//!
//! # ── PDF text extraction ──
//! items = py_features.extract_text_with_positions(pdf_bytes)
//! # -> [{'text': 'Hello', 'x': 72.0, 'y': 720.0, 'width': 45.0,
//! #      'height': 12.0, 'font': 'F1', 'font_size': 10.0, 'page': 1,
//! #      'is_bold': False, 'is_italic': False, 'is_underline': False,
//! #      'is_strikeout': False, 'item_type': 'text'}, ...]
//! ```
//!
//! ---

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pdf_inspector_core::{PdfType, types::ItemType};

// ── extract_all ──────────────────────────────────────────────────────────

/// Extract 103 pixel heuristics from raw RGB bytes.
///
/// ```python
/// from PIL import Image
/// import numpy as np
///
/// img = Image.open("page.jpg").convert("RGB")
/// pixels = np.array(img, dtype=np.uint8).tobytes()
/// feats = py_features.extract_all(pixels, img.width, img.height)
/// assert len(feats) == 103  # 100 real + 3 zero-padded
/// ```
///
/// * pixels: `bytes` — RGB bytes (length must equal width × height × 3)
/// * width: `int`
/// * height: `int`
/// * Returns: `list[float]` (length 103)
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

// ── classify_pdf ─────────────────────────────────────────────────────────

/// Classify a PDF from raw bytes.
///
/// ```python
/// pdf_bytes = open("document.pdf", "rb").read()
/// result = py_features.classify_pdf(pdf_bytes)
/// # -> {'pdf_type': 'TextBased', 'page_count': 7,
/// #     'pages_needing_ocr': [], 'confidence': 1.0}
/// ```
///
/// * data: `bytes` — raw PDF file contents
/// * Returns: `dict` with keys:
///     - `pdf_type`: `str` — one of `"TextBased"`, `"Scanned"`, `"ImageBased"`, `"Mixed"`
///     - `page_count`: `int`
///     - `pages_needing_ocr`: `list[int]` — 1-indexed page numbers
///     - `confidence`: `float`
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

// ── extract_text_with_positions ───────────────────────────────────────────

/// Extract text items with positions from a PDF memory buffer.
///
/// ```python
/// pdf_bytes = open("document.pdf", "rb").read()
/// items = py_features.extract_text_with_positions(pdf_bytes)
/// # -> [{'text': 'Hello', 'x': 72.0, 'y': 720.0, 'width': 45.0,
/// #      'height': 12.0, 'font': 'F1', 'font_size': 10.0, 'page': 1,
/// #      'is_bold': False, 'is_italic': False, 'is_underline': False,
/// #      'is_strikeout': False, 'item_type': 'text'}, ...]
/// ```
///
/// * data: `bytes` — raw PDF file contents
/// * Returns: `list[dict]` — each dict has:
///     - `text`: `str`
///     - `x`, `y`, `width`, `height`: `float`
///     - `font`: `str` — font name
///     - `font_size`: `float`
///     - `page`: `int` — 1-indexed page number
///     - `is_bold`, `is_italic`, `is_underline`, `is_strikeout`: `bool`
///     - `item_type`: `str` — one of `"text"`, `"image"`, `"link"`, `"formField"`
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

// ── is_release_build ─────────────────────────────────────────────────────

/// Check if this Rust binary was built in optimized release mode.
/// Returns true if release, false if debug (compiled with debug assertions).
#[pyfunction]
fn is_release_build() -> bool {
    cfg!(not(debug_assertions))
}

/// `py_features` — 4 functions, zero config.
///
/// ```python
/// import py_features
///
/// feats = py_features.extract_all(raw_rgb_bytes, width, height)
/// result = py_features.classify_pdf(pdf_bytes)
/// items = py_features.extract_text_with_positions(pdf_bytes)
/// is_rel = py_features.is_release_build()
/// ```
#[pymodule]
fn py_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_all, m)?)?;
    m.add_function(wrap_pyfunction!(classify_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(extract_text_with_positions, m)?)?;
    m.add_function(wrap_pyfunction!(is_release_build, m)?)?;
    Ok(())
}
