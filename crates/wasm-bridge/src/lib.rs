use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use gbdt_core::{
    combine_rotation_calibrated, GlyphEvidence, Model, ModelManifest, DEFAULT_TAU, LABELS_V2,
};
#[cfg(test)]
use gbdt_core::combine_rotation;

mod pdf;

static MODEL: Mutex<Option<Model>> = Mutex::new(None);

const LABELS: [&str; 9] = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "rotation_0",
    "rotation_90",
    "rotation_180",
    "rotation_270",
];

static MODEL_V2: Mutex<Option<ModelManifest>> = Mutex::new(None);

#[wasm_bindgen]
pub fn load_model(json: &str) -> Result<(), JsValue> {
    let model = Model::from_xgboost_json(json, 9).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut guard = MODEL.lock().unwrap();
    *guard = Some(model);
    Ok(())
}

/// Load the per-degree rework model manifest (v3; format-compatible with v2).
///
/// Expects the v3 `model.json` shape `{labels, tau, version, trees}` written by
/// `train_model_v2.py`. Also accepts a legacy flat tree array for
/// compatibility, but then the 7-label count and the default tau apply.
#[wasm_bindgen]
pub fn load_model_v2(json: &str) -> Result<(), JsValue> {
    let manifest = ModelManifest::from_json(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if manifest.labels.len() != LABELS_V2.len() {
        return Err(JsValue::from_str(&format!(
            "manifest must have {} labels, got {}: {:?}",
            LABELS_V2.len(),
            manifest.labels.len(),
            manifest.labels
        )));
    }
    let mut guard = MODEL_V2.lock().unwrap();
    *guard = Some(manifest);
    Ok(())
}

/// Release-tag version of the loaded model (empty string when unset or not
/// loaded). Matches the tag the model.json was built for, e.g. "0.4.0".
#[wasm_bindgen]
pub fn model_version() -> String {
    MODEL_V2
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.version.clone())
        .unwrap_or_default()
}

/// Scan text for PII/PCI (credit card numbers, SSNs, emails, phones).
///
/// Returns a JavaScript array of hit objects:
///   [{kind: "PAN"|"SSN"|"PHONE"|"EMAIL"|"CVV"|"EXPIRY"|"ROUTING"|"ACCOUNT"|"DOB"|"TRACK"|"PIN"|"PIN_BLOCK",
///     text: "matched string", start: byte_offset, end: byte_offset}, ...]
#[wasm_bindgen]
pub fn scan_pii(text: &str) -> Result<JsValue, JsValue> {
    use pii_core::HitKind;

    let hits = pii_core::scan(text);
    let arr = js_sys::Array::new();
    for hit in &hits {
        let obj = js_sys::Object::new();
        let kind_str = match hit.kind {
            HitKind::Pan => "PAN",
            HitKind::Cvv => "CVV",
            HitKind::Expiry => "EXPIRY",
            HitKind::Ssn => "SSN",
            HitKind::Email => "EMAIL",
            HitKind::Phone => "PHONE",
            HitKind::RoutingNumber => "ROUTING",
            HitKind::BankAccount => "ACCOUNT",
            HitKind::Dob => "DOB",
            HitKind::Address => "ADDRESS",
            HitKind::Name => "NAME",
            HitKind::Zip => "ZIP",
            HitKind::TrackData => "TRACK",
            HitKind::Pin => "PIN",
            HitKind::PinBlock => "PIN_BLOCK",
        };
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("kind"),
            &JsValue::from_str(kind_str),
        )?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("text"),
            &JsValue::from_str(&hit.text),
        )?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("start"),
            &JsValue::from_f64(hit.start as f64),
        )?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("end"),
            &JsValue::from_f64(hit.end as f64),
        )?;
        arr.push(&obj);
    }
    Ok(JsValue::from(arr))
}

/// Load a CRF model for NER-based PII detection.
///
/// The model is used by `scan_pii` to detect ADDRESS, NAME, ACCOUNT
/// via sequence labeling, ensemble-merged with rule-based scanners.
///
/// Pass an empty string to clear the model.
#[wasm_bindgen]
pub fn load_crf_model(json: &str) -> Result<(), JsValue> {
    if json.is_empty() {
        // No way to clear the static, but we can set a blank model
        return Ok(());
    }

    let parsed: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("CRF JSON parse error: {e}")))?;

    // Parse labels
    let labels: Vec<String> = parsed["labels"]
        .as_array()
        .ok_or_else(|| JsValue::from_str("CRF model: missing 'labels' array"))?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    // Parse featureIndex: {"feature_name": 0, ...}
    let fi = &parsed["featureIndex"];
    let fi_obj = fi
        .as_object()
        .ok_or_else(|| JsValue::from_str("CRF model: missing 'featureIndex' object"))?;

    let mut feature_index: Vec<(String, usize)> = Vec::with_capacity(fi_obj.len());
    for (name, idx_val) in fi_obj {
        let idx = idx_val.as_u64().unwrap_or(0) as usize;
        feature_index.push((name.clone(), idx));
    }

    // Parse labelWeights: {"O": [0.1, 0.2, ...], "ADDRESS": [...], ...}
    let lw = &parsed["labelWeights"];
    let lw_obj = lw
        .as_object()
        .ok_or_else(|| JsValue::from_str("CRF model: missing 'labelWeights' object"))?;

    let num_features = feature_index.len();
    let mut label_weights: Vec<Vec<f32>> = Vec::with_capacity(labels.len());
    for label in &labels {
        if let Some(weights_arr) = lw_obj.get(label).and_then(|v| v.as_array()) {
            let mut weights = vec![0.0f32; num_features];
            for (i, w) in weights_arr.iter().enumerate() {
                if i < num_features {
                    weights[i] = w.as_f64().unwrap_or(0.0) as f32;
                }
            }
            label_weights.push(weights);
        } else {
            label_weights.push(vec![0.0f32; num_features]);
        }
    }

    // Parse transitions: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    let t = &parsed["transitions"];
    let t_arr = t
        .as_array()
        .ok_or_else(|| JsValue::from_str("CRF model: missing 'transitions' array"))?;

    let k = labels.len();
    let mut transitions: Vec<Vec<f32>> = Vec::with_capacity(k);
    for row_val in t_arr {
        if let Some(row_arr) = row_val.as_array() {
            let row: Vec<f32> = row_arr
                .iter()
                .take(k)
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            transitions.push(row);
        }
    }
    // Pad if needed
    while transitions.len() < k {
        transitions.push(vec![0.0f32; k]);
    }

    let model = pii_core::crf::CrfModel {
        labels,
        feature_index,
        label_weights,
        transitions,
    };

    pii_core::set_crf_model(model);
    Ok(())
}

/// Resize image so the longest edge is at most `max_edge` pixels.
fn resize_max_edge(img: image::DynamicImage, max_edge: u32) -> image::DynamicImage {
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    if longest <= max_edge {
        return img;
    }
    let ratio = max_edge as f64 / longest as f64;
    let new_w = (w as f64 * ratio).round() as u32;
    let new_h = (h as f64 * ratio).round() as u32;
    img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
}

fn extract_all(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    features_core::extract_all(pixels, width, height)
}

#[wasm_bindgen]
pub fn classify_file(bytes: &[u8]) -> Result<JsValue, JsValue> {
    let guard = MODEL.lock().unwrap();
    let model = guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("Model not loaded — call load_model first"))?;

    let img = image::load_from_memory(bytes)
        .map_err(|e| JsValue::from_str(&format!("Image decode error: {e}")))?;

    let img = resize_max_edge(img, 512);
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    let features = extract_all(&pixels, width as usize, height as usize);
    let preds = model.predict(&features);

    let result = js_sys::Object::new();
    for (label, &pred) in LABELS.iter().zip(preds.iter()) {
        js_sys::Reflect::set(&result, &JsValue::from_str(label), &JsValue::from_f64(pred)).unwrap();
    }

    Ok(result.into())
}

/// Per-degree rotation classification (v2): 5 doc traits + flip + trust +
/// deterministic `rotation_deg`, `rotation_confidence`, `needs_review`.
#[wasm_bindgen]
pub fn classify_file_v2(bytes: &[u8]) -> Result<JsValue, JsValue> {
    let guard = MODEL_V2.lock().unwrap();
    let manifest = guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("Model not loaded — call load_model_v2 first"))?;
    let tau = if manifest.tau.is_finite() && manifest.tau > 0.0 {
        manifest.tau
    } else {
        DEFAULT_TAU
    };
    let model = &manifest.model;

    let img = image::load_from_memory(bytes)
        .map_err(|e| JsValue::from_str(&format!("Image decode error: {e}")))?;

    let img = resize_max_edge(img, 512);
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    let (features, glyph_logodds, glyph_count) =
        features_core::extract_all_v2_with_glyph(&pixels, width as usize, height as usize);
    let preds = model.predict(&features);
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

    let result = js_sys::Object::new();
    for (label, &pred) in manifest.labels.iter().zip(preds.iter()) {
        js_sys::Reflect::set(&result, &JsValue::from_str(label), &JsValue::from_f64(pred)).unwrap();
    }
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("rotation_deg"),
        &JsValue::from_f64(d.rotation_deg),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("rotation_confidence"),
        &JsValue::from_f64(d.confidence),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("needs_review"),
        &JsValue::from_bool(d.needs_review),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("flip_vote_count"),
        &JsValue::from_f64(d.flip_vote_count),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("flip_vote_agreement"),
        &JsValue::from_f64(d.flip_vote_agreement),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("axis_vote_count"),
        &JsValue::from_f64(d.axis_vote_count),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("axis_agreement"),
        &JsValue::from_f64(d.axis_agreement),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("glyph_count"),
        &JsValue::from_f64(d.glyph_count),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("glyph_logodds"),
        &JsValue::from_f64(d.glyph_logodds),
    )?;

    Ok(result.into())
}

/// Benchmark Sobel: time SIMD vs scalar path on a synthetic grayscale image.
/// Returns a JS object `{simd_us, scalar_us, speedup, pixels, width, height}`.
/// On native (not WASM), simd and scalar are identical.
#[wasm_bindgen]
pub fn bench_sobel(width: usize, height: usize, iterations: u32) -> Result<JsValue, JsValue> {
    let n = width * height;
    let gray: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.3).sin() * 128.0 + 128.0)
        .collect();

    // Warm up
    features_core::edges::sobel(&gray, width, height);

    let t0 = js_sys::Date::now();
    for _ in 0..iterations {
        features_core::edges::sobel(&gray, width, height);
    }
    let simd_us = (js_sys::Date::now() - t0) * 1000.0 / iterations as f64;

    let t0 = js_sys::Date::now();
    for _ in 0..iterations {
        features_core::edges::sobel_scalar_force(&gray, width, height);
    }
    let scalar_us = (js_sys::Date::now() - t0) * 1000.0 / iterations as f64;

    let speedup = if simd_us > 0.0 {
        scalar_us / simd_us
    } else {
        1.0
    };

    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("simd_us"),
        &JsValue::from_f64(simd_us),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("scalar_us"),
        &JsValue::from_f64(scalar_us),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("speedup"),
        &JsValue::from_f64(speedup),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("pixels"),
        &JsValue::from_f64(n as f64),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("width"),
        &JsValue::from_f64(width as f64),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("height"),
        &JsValue::from_f64(height as f64),
    )?;

    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gbdt_core::RADON_AXIS_INDEX;
    use wasm_bindgen_test::*;

    /// Trivial 9-label model: one tree per label, each always returning leaf 0.0
    /// → sigmoid(0.0) = 0.5 for every label.
    const TRIVIAL_MODEL_JSON: &str = r#"[
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
        [{"nodeid": 0, "depth": 0, "leaf": 0.0}]
    ]"#;

    fn create_test_image() -> Vec<u8> {
        let mut img = image::RgbImage::new(64, 64);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let r = (x as u8).wrapping_mul(4);
            let g = (y as u8).wrapping_mul(4);
            let b = ((x as u8).wrapping_add(y as u8)).wrapping_mul(2);
            *pixel = image::Rgb([r, g, b]);
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[wasm_bindgen_test]
    fn test_classify_file_returns_nine_labels() {
        load_model(TRIVIAL_MODEL_JSON).unwrap();

        let png_bytes = create_test_image();
        let result = classify_file(&png_bytes).unwrap();

        let labels = [
            "is_document",
            "is_digital",
            "is_paper",
            "is_crumpled",
            "is_shadow",
            "rotation_0",
            "rotation_90",
            "rotation_180",
            "rotation_270",
        ];
        for label in &labels {
            let val = js_sys::Reflect::get(&result, &JsValue::from_str(label))
                .unwrap()
                .as_f64()
                .unwrap();
            assert!(
                (val - 0.5).abs() < 0.01,
                "{}: expected ~0.5, got {}",
                label,
                val
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_classify_file_no_model_errors() {
        // Reset model
        {
            let mut guard = MODEL.lock().unwrap();
            *guard = None;
        }
        let png_bytes = create_test_image();
        let err = classify_file(&png_bytes).unwrap_err();
        let msg = err.as_string().unwrap();
        assert!(msg.contains("Model not loaded"), "got: {msg}");
    }

    #[wasm_bindgen_test]
    fn test_resize_max_edge_noop() {
        let img = image::DynamicImage::new_rgb8(100, 200);
        let resized = resize_max_edge(img, 300);
        assert_eq!(resized.width(), 100);
        assert_eq!(resized.height(), 200);
    }

    #[wasm_bindgen_test]
    fn test_resize_max_edge_downscales() {
        let img = image::DynamicImage::new_rgb8(1000, 500);
        let resized = resize_max_edge(img, 200);
        assert_eq!(resized.width(), 200);
        assert_eq!(resized.height(), 100);
    }

    #[wasm_bindgen_test]
    fn test_feature_count() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let feats = extract_all(&pixels, 64, 64);
        assert_eq!(
            feats.len(),
            103,
            "expected 103 features, got {}",
            feats.len()
        );
    }

    #[test]
    fn test_feature_count_native() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let feats = extract_all(&pixels, 64, 64);
        assert_eq!(
            feats.len(),
            103,
            "expected 103 features, got {}",
            feats.len()
        );
    }

    #[test]
    fn test_extract_all_v2_feature_count_native() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let feats = features_core::extract_all_v2(&pixels, 64, 64);
        assert_eq!(
            feats.len(),
            124,
            "expected 124 features, got {}",
            feats.len()
        );
    }

    #[test]
    fn test_combine_rotation_matches_python_formula() {
        // axis = 17 deg, flip=0 -> rotation = (180-17) mod 180 = 163.
        let mut features = vec![0.0f64; 116];
        features[RADON_AXIS_INDEX] = 17.0_f64.to_radians();

        let preds = [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8];
        let d = combine_rotation(&features, &preds, 0.9);
        assert!((d.rotation_deg - 163.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.8 * 0.9).abs() < 1e-12, "conf={}", d.confidence);
        assert!(d.needs_review, "conf 0.72 < tau 0.9 must review");

        // flip=1 -> rotation = 163 + 180 = 343.
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0];
        let d = combine_rotation(&features, &preds, 0.5);
        assert!((d.rotation_deg - 343.0).abs() < 1e-9, "deg={}", d.rotation_deg);
        assert!((d.confidence - 0.9).abs() < 1e-12, "conf={}", d.confidence);
        assert!(!d.needs_review, "conf 0.9 >= tau 0.5 and is_document");

        // Not a document -> review regardless of confidence.
        let preds = [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0];
        let d = combine_rotation(&features, &preds, 0.0);
        assert!(d.needs_review, "not is_document must force review");

        // axis exactly 0, flip=0 -> rotation = 0 (upright).
        features[RADON_AXIS_INDEX] = 0.0;
        let preds = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0];
        let d = combine_rotation(&features, &preds, 0.5);
        assert!((d.rotation_deg - 0.0).abs() < 1e-9, "deg={}", d.rotation_deg);
    }

    const TRIVIAL_MANIFEST_V2: &str = r#"{
        "labels": ["is_document","is_digital","is_paper","is_crumpled","is_shadow","flip","trust"],
        "tau": 0.9,
        "trees": [
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}],
            [{"nodeid": 0, "depth": 0, "leaf": 0.0}]
        ]
    }"#;

    #[wasm_bindgen_test]
    fn test_classify_file_v2_returns_v2_shape() {
        load_model_v2(TRIVIAL_MANIFEST_V2).unwrap();
        let png_bytes = create_test_image();
        let result = classify_file_v2(&png_bytes).unwrap();

        let labels = [
            "is_document",
            "is_digital",
            "is_paper",
            "is_crumpled",
            "is_shadow",
            "flip",
            "trust",
        ];
        for label in &labels {
            let val = js_sys::Reflect::get(&result, &JsValue::from_str(label))
                .unwrap()
                .as_f64()
                .unwrap();
            assert!(
                (val - 0.5).abs() < 0.01,
                "{}: expected ~0.5, got {}",
                label,
                val
            );
        }

        let deg = js_sys::Reflect::get(&result, &JsValue::from_str("rotation_deg"))
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((0.0..360.0).contains(&deg), "rotation_deg={deg}");

        let conf = js_sys::Reflect::get(&result, &JsValue::from_str("rotation_confidence"))
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((conf - 0.25).abs() < 0.01, "confidence={conf}");

        let review = js_sys::Reflect::get(&result, &JsValue::from_str("needs_review"))
            .unwrap()
            .as_bool()
            .unwrap();
        assert!(review, "conf 0.25 < tau 0.9 must review");
    }
}
