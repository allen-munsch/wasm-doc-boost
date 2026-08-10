use pdf_inspector_core::PdfType;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn pdf_type_name(t: PdfType) -> &'static str {
    match t {
        PdfType::TextBased => "TextBased",
        PdfType::Scanned => "Scanned",
        PdfType::ImageBased => "ImageBased",
        PdfType::Mixed => "Mixed",
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PdfClassifyResult {
    pdf_type: String,
    page_count: u32,
    pages_needing_ocr: Vec<u32>,
    confidence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TextItemResult {
    text: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    font: String,
    font_size: f32,
    page: u32,
    is_bold: bool,
    is_italic: bool,
    is_underline: bool,
    is_strikeout: bool,
    item_type: String,
}

impl From<pdf_inspector_core::TextItem> for TextItemResult {
    fn from(i: pdf_inspector_core::TextItem) -> Self {
        let item_type = match i.item_type {
            pdf_inspector_core::types::ItemType::Text => "text",
            pdf_inspector_core::types::ItemType::Image => "image",
            pdf_inspector_core::types::ItemType::Link(_) => "link",
            pdf_inspector_core::types::ItemType::FormField => "formField",
        };
        Self {
            text: i.text,
            x: i.x,
            y: i.y,
            width: i.width,
            height: i.height,
            font: i.font,
            font_size: i.font_size,
            page: i.page,
            is_bold: i.is_bold,
            is_italic: i.is_italic,
            is_underline: i.is_underline,
            is_strikeout: i.is_strikeout,
            item_type: item_type.to_string(),
        }
    }
}

fn js_error(context: &str, error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&format!("{context}: {error}")).into()
}

/// Classify a PDF from bytes. Returns pdfType, pageCount, pagesNeedingOcr, confidence.
#[wasm_bindgen(js_name = classifyPdf)]
pub fn classify_pdf(data: &[u8]) -> Result<JsValue, JsValue> {
    let result =
        pdf_inspector_core::classify_pdf_mem(data).map_err(|e| js_error("classify PDF", e))?;
    let value = serde_wasm_bindgen::to_value(&PdfClassifyResult {
        pdf_type: pdf_type_name(result.pdf_type).to_string(),
        page_count: result.page_count,
        pages_needing_ocr: result.pages_needing_ocr,
        confidence: result.confidence as f64,
    })
    .map_err(|e| js_error("serialize result", e))?;
    Ok(value)
}

/// Extract text items with positions from a PDF memory buffer.
#[wasm_bindgen(js_name = extractTextWithPositions)]
pub fn extract_text_with_positions(data: &[u8]) -> Result<JsValue, JsValue> {
    let items =
        pdf_inspector_core::extract_text_with_positions_mem(data)
            .map_err(|e| js_error("extract text", e))?;
    let results: Vec<TextItemResult> = items.into_iter().map(Into::into).collect();
    let value = serde_wasm_bindgen::to_value(&results)
        .map_err(|e| js_error("serialize result", e))?;
    Ok(value)
}
