//! # py_features — Rust-powered feature extraction + PDF inspection for Python
//!
//! **Install (prebuilt wheel, CPython 3.8+):**
//!
//! - Linux x86_64: `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.2.0/py_features-0.2.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
//! - Linux aarch64: `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.2.0/py_features-0.2.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
//! - macOS Intel (x86_64): `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.2.0/py_features-0.2.0-cp38-abi3-macosx_10_12_x86_64.whl`
//! - macOS Apple Silicon (arm64): `pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.2.0/py_features-0.2.0-cp38-abi3-macosx_11_0_arm64.whl`
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

mod backend;
#[cfg(feature = "gpu")]
mod cuda;
mod engines;

use std::sync::Mutex;

use gbdt_core::{combine_rotation_calibrated, GlyphEvidence, ModelManifest, DEFAULT_TAU, LABELS};
use pdf_inspector_core::{types::ItemType, PdfType};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

static MODEL: Mutex<Option<ModelManifest>> = Mutex::new(None);

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
/// Decode an image from encoded bytes and resize to `max_edge` EXACTLY the
/// way wasm-bridge's `resize_max_edge` does (image crate decode, Triangle
/// filter, round-to-nearest dims). The training export must use this so the
/// features it computes match the wasm inference path pixel-for-pixel.
///
/// Returns `(rgb_bytes, width, height)`.
#[pyfunction]
fn load_rgb_resized(encoded: Vec<u8>, max_edge: u32) -> PyResult<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(&encoded)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("decode: {e}")))?;
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    let img = if longest <= max_edge {
        img
    } else {
        let ratio = max_edge as f64 / longest as f64;
        let new_w = (w as f64 * ratio).round() as u32;
        let new_h = (h as f64 * ratio).round() as u32;
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
    };
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    Ok((rgb.into_raw(), width as usize, height as usize))
}

#[pyfunction]
fn extract_all(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Vec<f64>> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }

    let (radon_fn, fine_fn) = backend::radon_pair(backend::selected());
    Ok(features_core::extract_all_with_radon(
        &pixels, width, height, radon_fn, fine_fn,
    ))
}

/// Which radon backend `extract_all`/`predict` will use in this process.
/// `PY_FEATURES_BACKEND=cpu|cuda|webgpu|auto` (auto = cuda when reachable,
/// else cpu; webgpu is reserved for the wasm-bridge work and falls back to
/// cpu on native builds until it lands).
#[pyfunction]
fn backend_info() -> String {
    backend::name(backend::effective()).to_string()
}

/// Extract per-glyph 20x20 ink-density crops from the best-axis deskew.
/// Training input for scripts/train_glyph_classifier.py: call on an upright
/// image for label 0 and on its 180-rotated RGB for label 1.
///
/// * pixels: `bytes` — RGB bytes (length must equal width × height × 3)
/// * Returns: `list[list[float]]` — one 400-float row-major crop per glyph.
#[pyfunction]
fn extract_glyph_crops(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Vec<Vec<f32>>> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }
    let gray = features_core::to_grayscale(&pixels, width, height);
    let rot = features_core::rotation_extract(&gray, &pixels, width, height);
    let mut out = Vec::with_capacity(rot.glyph_count);
    for c in rot
        .glyph_crops
        .chunks_exact(features_core::glyph::GLYPH_DIM)
    {
        out.push(c.to_vec());
    }
    Ok(out)
}

/// Extract page-level per-glyph flip evidence: `(sum_logodds, glyph_count)`.
/// Positive sum_logodds = flipped, negative = upright. Diagnostic twin of the
/// channel that combine_rotation uses when enough glyphs voted.
#[pyfunction]
fn extract_glyph_logodds(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<(f64, f64)> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }
    let gray = features_core::to_grayscale(&pixels, width, height);
    let rot = features_core::rotation_extract(&gray, &pixels, width, height);
    let (sum_logodds, count) = features_core::glyph::glyph_logodds_from_crops(&rot.glyph_crops);
    Ok((sum_logodds, count as f64))
}

/// Debug helper for the rotation rework: radon curve + textline line-quality
/// at the top-4 local maxima of the curve.
///
/// Returns 60 radon-curve bins, then for each of the top-4 local maxima
/// (highest first, parabolic-refined): [axis_rad, n_lines, total_line_ink].
/// Used to validate the multi-candidate axis selection before wiring it into
/// `extract_all`.
#[pyfunction]
fn rotation_diagnostic(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Vec<f64>> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }
    let gray = features_core::to_grayscale(&pixels, width, height);
    let curve = features_core::radon::radon_curve(&gray, width, height);
    let mut out = curve.clone();

    // Top-4 local maxima of the 180-periodic curve.
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    for i in 0..curve.len() {
        let prev = curve[(i + curve.len() - 1) % curve.len()];
        let next = curve[(i + 1) % curve.len()];
        let v = curve[i];
        if v >= prev && v >= next {
            maxima.push((i, v));
        }
    }
    maxima.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(bin, _) in maxima.iter().take(4) {
        let refined = features_core::radon::parabolic_refine(&curve, bin);
        let axis_rad = ((refined * features_core::radon::RADON_DEG_PER_BIN) % 180.0).to_radians();
        let tl = features_core::textline::textline_flip_features_with_axis(
            &gray, width, height, axis_rad,
        );
        out.push(axis_rad);
        out.push(tl[2]); // n_lines
        out.push(tl[3] + tl[4] + tl[5]); // asc + desc + body mass fractions
    }
    Ok(out)
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
fn classify_pdf(data: Vec<u8>) -> PyResult<Py<PyAny>> {
    let result = pdf_inspector_core::classify_pdf_mem(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Python::attach(|py| {
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
        Ok(dict.unbind().into_any())
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
fn extract_text_with_positions(data: Vec<u8>) -> PyResult<Py<PyAny>> {
    let items = pdf_inspector_core::extract_text_with_positions_mem(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Python::attach(|py| {
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
        Ok(list.unbind().into_any())
    })
}

// ── scan_pii ─────────────────────────────────────────────────────────────

/// Scan text for PII/PCI using deterministic rules.
///
/// ```python
/// hits = py_features.scan_pii("Card 4111 1111 1111 1111 exp 08/27 cvv 123")
/// # -> [{'kind': 'pan', 'text': '4111 1111 1111 1111', 'start': 5, 'end': 24}, ...]
/// ```
///
/// * text: `str` — the text to scan
/// * Returns: `list[dict]` — each dict has:
///     - `kind`: `str` — one of `"pan"`, `"cvv"`, `"expiry"`, `"ssn"`,
///       `"email"`, `"phone"`, `"routing"`, `"account"`, `"dob"`,
///       `"address"`, `"name"`, `"zip"`, `"track"`, `"pin"`, `"pin_block"`
///     - `text`: `str` — the matched text
///     - `start`: `int` — byte offset where the match starts
///     - `end`: `int` — byte offset where the match ends (exclusive)
#[pyfunction]
fn scan_pii(text: &str) -> PyResult<Py<PyAny>> {
    use pii_core::HitKind;

    let hits = pii_core::scan(text);

    Python::attach(|py| {
        let list = PyList::empty(py);
        for hit in hits {
            let dict = PyDict::new(py);
            let kind = match hit.kind {
                HitKind::Pan => "pan",
                HitKind::Cvv => "cvv",
                HitKind::Expiry => "expiry",
                HitKind::Ssn => "ssn",
                HitKind::Email => "email",
                HitKind::Phone => "phone",
                HitKind::RoutingNumber => "routing",
                HitKind::BankAccount => "account",
                HitKind::Dob => "dob",
                HitKind::Address => "address",
                HitKind::Name => "name",
                HitKind::Zip => "zip",
                HitKind::TrackData => "track",
                HitKind::Pin => "pin",
                HitKind::PinBlock => "pin_block",
            };
            dict.set_item("kind", kind)?;
            dict.set_item("text", hit.text)?;
            dict.set_item("start", hit.start as u64)?;
            dict.set_item("end", hit.end as u64)?;
            list.append(dict)?;
        }
        Ok(list.unbind().into_any())
    })
}

// ── is_release_build ─────────────────────────────────────────────────────

/// Check if this Rust binary was built in optimized release mode.
/// Returns true if release, false if debug (compiled with debug assertions).
#[pyfunction]
fn is_release_build() -> bool {
    cfg!(not(debug_assertions))
}

// ── v3 GBDT model loading + prediction ───────────────────────────────────

/// Load the v3 model manifest `{labels, tau, version, trees, trees_per_label}` —
/// the exact same JSON the frontend passes to `load_model`.
#[pyfunction]
fn load_model(json: &str) -> PyResult<()> {
    let manifest = ModelManifest::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if manifest.labels.len() != LABELS.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "manifest must have {} labels, got {}: {:?}",
            LABELS.len(),
            manifest.labels.len(),
            manifest.labels
        )));
    }
    *MODEL.lock().unwrap() = Some(manifest);
    Ok(())
}

/// Score the 7 heads on raw RGB pixels and combine the deterministic radon
/// Resize raw RGB pixels to a `max_edge` long edge with the same Triangle
/// filter `classify_file` uses, so `predict` and the browser twin see
/// identical inputs. The GBDT + isotonic trust calibration are trained on
/// 512-long-edge features; native-resolution input is out-of-distribution
/// and miscalibrates `rotation_confidence` (90°/270° pages read as upright
/// at ~0.94 confidence). Idempotent for inputs already <= max_edge.
fn resize_rgb_max_edge(
    pixels: &[u8],
    width: usize,
    height: usize,
    max_edge: u32,
) -> (Vec<u8>, usize, usize) {
    let longest = width.max(height) as u32;
    if longest <= max_edge {
        return (pixels.to_vec(), width, height);
    }
    let img = image::RgbImage::from_raw(width as u32, height as u32, pixels.to_vec())
        .expect("valid rgb buffer");
    let ratio = max_edge as f64 / longest as f64;
    let new_w = (width as f64 * ratio).round().max(1.0) as u32;
    let new_h = (height as f64 * ratio).round().max(1.0) as u32;
    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);
    let (w, h) = resized.dimensions();
    (resized.into_raw(), w as usize, h as usize)
}

/// axis with the flip/trust heads. Backend twin of the frontend's
/// `classify_file`: same features-core extraction, same gbdt-core forest.
///
/// * pixels: `bytes` — RGB bytes (length must equal width × height × 3)
/// * width: `int`
/// * height: `int`
/// * Returns: `dict` with the 7 label probabilities plus `rotation_deg`,
///   `rotation_confidence`, `needs_review`.
#[pyfunction]
fn predict(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Py<PyAny>> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}x3), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }

    // Twin parity with classify_file: resize to the 512-long-edge input
    // the model was trained on. Native-resolution input otherwise produces
    // out-of-distribution features and a miscalibrated rotation_confidence.
    let (pixels, width, height) = resize_rgb_max_edge(&pixels, width, height, 512);

    let manifest = {
        let guard = MODEL.lock().unwrap();
        guard.as_ref().cloned().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("model not loaded — call load_model first")
        })?
    };

    let (radon_fn, fine_fn) = backend::radon_pair(backend::selected());
    let (features, glyph_logodds, glyph_count) =
        features_core::extract_all_with_glyph_with_radon(&pixels, width, height, radon_fn, fine_fn);
    let preds = manifest.model.predict(&features);
    let tau = if manifest.tau.is_finite() && manifest.tau > 0.0 {
        manifest.tau
    } else {
        DEFAULT_TAU
    };
    let d = combine_rotation_calibrated(
        &features,
        &preds,
        tau,
        manifest.confidence_calibration.as_deref(),
        Some(GlyphEvidence {
            sum_logodds: glyph_logodds,
            count: glyph_count,
        }),
    );

    Python::attach(|py| {
        let dict = PyDict::new(py);
        for (label, &pred) in manifest.labels.iter().zip(preds.iter()) {
            dict.set_item(label.as_str(), pred)?;
        }
        dict.set_item("rotation_deg", d.rotation_deg)?;
        dict.set_item("rotation_confidence", d.confidence)?;
        dict.set_item("needs_review", d.needs_review)?;
        dict.set_item("quarter", d.quarter)?;
        dict.set_item("skew_deg", d.skew_deg)?;
        dict.set_item("axis_ok", d.axis_ok)?;
        dict.set_item("flip_ok", d.flip_ok)?;
        dict.set_item("flip_vote_count", d.flip_vote_count)?;
        dict.set_item("flip_vote_agreement", d.flip_vote_agreement)?;
        dict.set_item("axis_vote_count", d.axis_vote_count)?;
        dict.set_item("axis_agreement", d.axis_agreement)?;
        dict.set_item("glyph_count", d.glyph_count)?;
        dict.set_item("glyph_logodds", d.glyph_logodds)?;
        // #6: binarized trait decisions at shipped thresholds + the ±0.08
        // near-boundary band, matching classify_file's trait_labels output.
        let trait_labels = PyDict::new(py);
        let trait_uncertain = PyDict::new(py);
        for i in 0..5 {
            let p = preds[i];
            let t = manifest
                .label_thresholds
                .as_ref()
                .and_then(|v| v.get(i).copied())
                .unwrap_or(0.5);
            trait_labels.set_item(manifest.labels[i].as_str(), p >= t)?;
            trait_uncertain.set_item(manifest.labels[i].as_str(), (p - t).abs() <= 0.08)?;
        }
        dict.set_item("trait_labels", trait_labels)?;
        dict.set_item("trait_uncertain", trait_uncertain)?;
        Ok(dict.unbind().into_any())
    })
}

/// Expose the raw combine step for diagnostics and parity checks.
#[pyfunction]
fn combine_rotation(features: Vec<f64>, preds: Vec<f64>, tau: f64) -> PyResult<Py<PyAny>> {
    let d = combine_rotation_calibrated(&features, &preds, tau, None, None);
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("rotation_deg", d.rotation_deg)?;
        dict.set_item("rotation_confidence", d.confidence)?;
        dict.set_item("needs_review", d.needs_review)?;
        dict.set_item("flip_vote_count", d.flip_vote_count)?;
        dict.set_item("flip_vote_agreement", d.flip_vote_agreement)?;
        dict.set_item("axis_vote_count", d.axis_vote_count)?;
        dict.set_item("axis_agreement", d.axis_agreement)?;
        Ok(dict.unbind().into_any())
    })
}

/// `py_features` — feature extraction, v3 GBDT prediction, PDF inspection,
/// PII scanning, and the ONNX engines (layout, OCR, span PII) behind optional
/// Python classes.
///
/// ```python
/// import py_features
///
/// feats = py_features.extract_all(raw_rgb_bytes, width, height)
/// py_features.load_model(open("model.json").read())
/// result = py_features.predict(raw_rgb_bytes, width, height)
/// pdf = py_features.classify_pdf(pdf_bytes)
/// hits = py_features.scan_pii(text)
/// ```
#[pymodule]
fn py_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_all, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    m.add_function(wrap_pyfunction!(extract_glyph_crops, m)?)?;
    m.add_function(wrap_pyfunction!(extract_glyph_logodds, m)?)?;
    m.add_function(wrap_pyfunction!(load_rgb_resized, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_diagnostic, m)?)?;
    m.add_function(wrap_pyfunction!(classify_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(extract_text_with_positions, m)?)?;
    m.add_function(wrap_pyfunction!(scan_pii, m)?)?;
    m.add_function(wrap_pyfunction!(is_release_build, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(combine_rotation, m)?)?;
    m.add_class::<engines::LayoutEngine>()?;
    m.add_class::<engines::OcrEngine>()?;
    m.add_class::<engines::PiiEngine>()?;
    Ok(())
}
