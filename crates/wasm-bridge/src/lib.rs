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
        assert_eq!(feats.len(), 78, "expected 78 features, got {}", feats.len());
    }
}
