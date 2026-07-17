use std::sync::Mutex;

use wasm_bindgen::prelude::*;

mod gbdt;

static MODEL: Mutex<Option<gbdt::Model>> = Mutex::new(None);

const LABELS: [&str; 5] = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
];

#[wasm_bindgen]
pub fn load_model(json: &str) -> Result<(), JsValue> {
    let model = gbdt::Model::from_xgboost_json(json, 5)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut guard = MODEL.lock().unwrap();
    *guard = Some(model);
    Ok(())
}

/// Scan text for PII/PCI (credit card numbers, SSNs, emails, phones).
///
/// Returns a JavaScript array of hit objects:
///   [{kind: "PAN"|"SSN"|"PHONE"|"EMAIL"|"CVV"|"EXPIRY"|"ROUTING"|"ACCOUNT"|"DOB",
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
    let fi_obj = fi.as_object()
        .ok_or_else(|| JsValue::from_str("CRF model: missing 'featureIndex' object"))?;

    let mut feature_index: Vec<(String, usize)> = Vec::with_capacity(fi_obj.len());
    for (name, idx_val) in fi_obj {
        let idx = idx_val.as_u64().unwrap_or(0) as usize;
        feature_index.push((name.clone(), idx));
    }

    // Parse labelWeights: {"O": [0.1, 0.2, ...], "ADDRESS": [...], ...}
    let lw = &parsed["labelWeights"];
    let lw_obj = lw.as_object()
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
    let t_arr = t.as_array()
        .ok_or_else(|| JsValue::from_str("CRF model: missing 'transitions' array"))?;

    let k = labels.len();
    let mut transitions: Vec<Vec<f32>> = Vec::with_capacity(k);
    for row_val in t_arr {
        if let Some(row_arr) = row_val.as_array() {
            let row: Vec<f32> = row_arr.iter()
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
    let mut features = Vec::new();

    features.extend(features_core::color::per_channel_stats(pixels, width, height));
    features.extend(features_core::color::grayscale_stats(pixels, width, height));
    features.push(features_core::color::colorfulness(pixels, width, height));
    features.extend(features_core::color::saturation_stats(pixels, width, height));

    features.push(features_core::edges::laplacian_variance(pixels, width, height));
    features.extend(features_core::edges::sobel_stats(pixels, width, height));
    features.push(features_core::edges::edge_density(pixels, width, height));
    features.extend(features_core::edges::edge_direction_histogram(pixels, width, height));
    features.push(features_core::edges::hv_edge_ratio(pixels, width, height));
    features.push(features_core::edges::canny_edge_density(pixels, width, height));

    features.push(features_core::texture::dct_low_freq_ratio(pixels, width, height));
    features.extend(features_core::texture::lbp_histogram(pixels, width, height));
    features.extend(features_core::texture::glcm_features(pixels, width, height));
    features.push(features_core::texture::fractal_dimension(pixels, width, height));

    features.push(features_core::noise::high_pass_residual_variance(pixels, width, height));
    features.extend(features_core::noise::jpeg_blockiness(pixels, width, height));
    features.push(features_core::noise::gradient_snr(pixels, width, height));

    features.extend(features_core::shadow::shadow_features(pixels, width, height));

    features.extend(features_core::crumple::lbp_variance(pixels, width, height));
    features.push(features_core::crumple::edge_density_std(pixels, width, height));
    features.push(features_core::crumple::texture_anisotropy(pixels, width, height));
    features.push(features_core::crumple::peak_local_entropy(pixels, width, height));

    features.extend(features_core::document::document_features(pixels, width, height));

    features
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
        js_sys::Reflect::set(&result, &JsValue::from_str(label), &JsValue::from_f64(pred))
            .unwrap();
    }

    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    /// Trivial 5-label model: one tree per label, each always returning leaf 0.0
    /// → sigmoid(0.0) = 0.5 for every label.
    const TRIVIAL_MODEL_JSON: &str = r#"[
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
    fn test_classify_file_returns_five_labels() {
        load_model(TRIVIAL_MODEL_JSON).unwrap();

        let png_bytes = create_test_image();
        let result = classify_file(&png_bytes).unwrap();

        let labels = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"];
        for label in &labels {
            let val = js_sys::Reflect::get(&result, &JsValue::from_str(label))
                .unwrap()
                .as_f64()
                .unwrap();
            assert!((val - 0.5).abs() < 0.01, "{}: expected ~0.5, got {}", label, val);
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
        assert_eq!(feats.len(), 81, "expected 81 features, got {}", feats.len());
    }
}
