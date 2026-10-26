//! Python bindings for the Rust inference engines: layout detection, OCR, and
//! span-based PII extraction. Each engine is a class that loads its model once
//! and infers many times.

use image::{DynamicImage, RgbImage};
use layout_core::LayoutEngine as LayoutEngineCore;
use ocr_core::{OcrEngine as OcrEngineCore, OrderBy};
use pii_span_core::{SchemaTask, SpanConfig, SpanEngine as SpanEngineCore};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn decode_rgb(pixels: Vec<u8>, width: usize, height: usize) -> PyResult<DynamicImage> {
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(PyValueError::new_err(format!(
            "Expected {expected} bytes ({width}x{height}x3), got {}",
            pixels.len()
        )));
    }
    let img = RgbImage::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| PyValueError::new_err("invalid RGB buffer"))?;
    Ok(DynamicImage::ImageRgb8(img))
}

/// PP-DocLayoutV3 document layout detector.
///
/// ```python
/// import py_features
/// from PIL import Image
/// import numpy as np
///
/// engine = py_features.LayoutEngine("PP-DocLayoutV3.onnx")
/// img = Image.open("page.png").convert("RGB")
/// pixels = np.array(img, dtype=np.uint8).tobytes()
/// regions = engine.detect(pixels, img.width, img.height)
/// # -> [{'label': 'text', 'score': 0.95, 'x1': 10.0, 'y1': 20.0,
/// #      'x2': 300.0, 'y2': 60.0}, ...]
/// ```
#[pyclass]
pub struct LayoutEngine {
    inner: LayoutEngineCore,
}

#[pymethods]
impl LayoutEngine {
    /// * model_path: `str` — path to the PP-DocLayoutV3 ONNX file (its `.data`
    ///   sidecar must sit next to it).
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let inner = LayoutEngineCore::from_file(model_path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Detect layout regions in raw RGB pixels.
    ///
    /// * pixels: `bytes` — RGB bytes (length must equal width × height × 3)
    /// * width: `int`
    /// * height: `int`
    /// * Returns: `list[dict]` — each dict has `label`, `score`, `x1`, `y1`,
    ///   `x2`, `y2` (bounding box in original-image coordinates).
    fn detect(&mut self, pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Py<PyAny>> {
        let image = decode_rgb(pixels, width, height)?;
        let regions = self
            .inner
            .detect(&image)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::attach(|py| {
            let list = PyList::empty(py);
            for r in regions {
                let d = PyDict::new(py);
                d.set_item("label", r.label)?;
                d.set_item("score", r.score)?;
                d.set_item("x1", r.x1)?;
                d.set_item("y1", r.y1)?;
                d.set_item("x2", r.x2)?;
                d.set_item("y2", r.y2)?;
                list.append(d)?;
            }
            Ok(list.unbind().into_any())
        })
    }
}

/// PP-OCR detection + recognition engine.
///
/// ```python
/// import py_features
/// from PIL import Image
/// import numpy as np
///
/// det = open("ch_PP-OCRv4_det_infer.onnx", "rb").read()
/// rec = open("ch_PP-OCRv4_rec_infer.onnx", "rb").read()
/// keys = open("ppocr_keys_v1.txt", "rb").read()
/// engine = py_features.OcrEngine(det, rec, keys)
///
/// img = Image.open("page.png").convert("RGB")
/// pixels = np.array(img, dtype=np.uint8).tobytes()
/// blocks = engine.recognize(pixels, img.width, img.height)
/// # -> [{'text': 'Hello', 'confidence': 0.98, 'x': 10.0, 'y': 20.0,
/// #      'width': 200.0, 'height': 40.0}, ...]
/// ```
#[pyclass]
pub struct OcrEngine {
    inner: OcrEngineCore,
}

#[pymethods]
impl OcrEngine {
    /// * det_model: `bytes` — detection ONNX model
    /// * rec_model: `bytes` — recognition ONNX model
    /// * keys: `bytes` — UTF-8 character dictionary (one char per line)
    #[new]
    fn new(det_model: Vec<u8>, rec_model: Vec<u8>, keys: Vec<u8>) -> PyResult<Self> {
        let inner = OcrEngineCore::new(&det_model, &rec_model, &keys)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Detect text regions in raw RGB pixels (detection only, no recognition).
    ///
    /// * Returns: `list[dict]` — each dict has `confidence` and `bbox`
    ///   (a list of four `[x, y]` corners).
    fn detect(&self, pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Py<PyAny>> {
        let image = decode_rgb(pixels, width, height)?;
        let regions = self
            .inner
            .detect_text_regions(&image)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::attach(|py| {
            let list = PyList::empty(py);
            for r in regions {
                let d = PyDict::new(py);
                d.set_item("confidence", r.confidence)?;
                let bbox = PyList::empty(py);
                for p in r.bbox {
                    let pt = PyList::empty(py);
                    pt.append(p[0])?;
                    pt.append(p[1])?;
                    bbox.append(pt)?;
                }
                d.set_item("bbox", bbox)?;
                list.append(d)?;
            }
            Ok(list.unbind().into_any())
        })
    }

    /// Detect and recognize text in raw RGB pixels.
    ///
    /// * pixels: `bytes` — RGB bytes (length must equal width × height × 3)
    /// * width: `int`
    /// * height: `int`
    /// * Returns: `list[dict]` — each dict has `text`, `confidence`, `x`, `y`,
    ///   `width`, `height` (sorted top-to-bottom, left-to-right).
    fn recognize(&self, pixels: Vec<u8>, width: usize, height: usize) -> PyResult<Py<PyAny>> {
        let image = decode_rgb(pixels, width, height)?;
        let blocks = self
            .inner
            .recognize_all(&image, OrderBy::Horizontal)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::attach(|py| {
            let list = PyList::empty(py);
            for b in blocks {
                let d = PyDict::new(py);
                d.set_item("text", b.text)?;
                d.set_item("confidence", b.confidence)?;
                d.set_item("x", b.x)?;
                d.set_item("y", b.y)?;
                d.set_item("width", b.width)?;
                d.set_item("height", b.height)?;
                list.append(d)?;
            }
            Ok(list.unbind().into_any())
        })
    }
}

/// GLiNER2 span-based PII/entity extractor.
///
/// ```python
/// import py_features
///
/// engine = py_features.PiiEngine("gliner2-pii-onnx/")
/// spans = engine.extract(
///     "Card 4111 1111 1111 1111, DL A123-456-789",
///     ["payment_card", "driver_license"],
/// )
/// # -> [{'label': 'payment_card', 'text': '4111 1111 1111 1111',
/// #      'start': 5, 'end': 24, 'score': 0.98}, ...]
/// ```
#[pyclass(unsendable)]
pub struct PiiEngine {
    inner: SpanEngineCore,
}

#[pymethods]
impl PiiEngine {
    /// * models_dir: `str` — directory holding the 8 GLiNER2 ONNX fragments and
    ///   `tokenizer.json`.
    #[new]
    fn new(models_dir: &str) -> PyResult<Self> {
        let config = SpanConfig::new(models_dir);
        let inner =
            SpanEngineCore::new(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Extract entity spans from text.
    ///
    /// * text: `str` — text to scan
    /// * labels: `list[str]` — entity labels to look for (e.g. `"payment_card"`,
    ///   `"driver_license"`, `"passport"`)
    /// * Returns: `list[dict]` — each dict has `label`, `text`, `score`,
    ///   `start`, `end` (byte offsets in the original text).
    fn extract(&mut self, text: &str, labels: Vec<String>) -> PyResult<Py<PyAny>> {
        let tasks = vec![SchemaTask::Entities(labels)];
        let output = self
            .inner
            .extract(text, &tasks)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::attach(|py| {
            let list = PyList::empty(py);
            for e in output.entities {
                let d = PyDict::new(py);
                d.set_item("label", e.label)?;
                d.set_item("text", e.text)?;
                d.set_item("score", e.score)?;
                d.set_item("start", e.char_start as u64)?;
                d.set_item("end", e.char_end as u64)?;
                list.append(d)?;
            }
            Ok(list.unbind().into_any())
        })
    }
}
