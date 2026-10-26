//! PP-DocLayoutV3 document layout detection.
//!
//! Ported from the DETR-style ONNX export (`PPDocLayoutV3ForObjectDetection`)
//! published as `Bei0001/PP-DocLayoutV3-ONNX` / `phungpx/PP-DocLayoutV3-ONNX`.
//! Input `pixel_values` (1, 3, 800, 800) scaled to [0, 1]; outputs `logits`
//! (1, 300, 25), `pred_boxes` (1, 300, 4) in cxcywh, `order_logits`
//! (1, 300, 300) and `out_masks` (1, 300, 200, 200). Returns a label, score and
//! bounding box per detected region, sorted by reading order, in
//! original-image coordinates.

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::DynamicImage;
use ort::session::Session;
use ort::value::Tensor;

const INPUT_SIZE: usize = 800;
const NUM_QUERIES: usize = 300;
const NUM_CLASSES: usize = 25;
const CONF_THRES: f32 = 0.5;

/// Ordered class labels for the 25-way head (id2label of the export).
const LABELS: &[&str] = &[
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "header",
    "image",
    "formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "text",
    "vision_footnote",
];

/// A detected layout region with its semantic label.
#[derive(Debug, Clone)]
pub struct LayoutRegion {
    pub label: String,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// PP-DocLayoutV3 layout detector.
pub struct LayoutEngine {
    session: Session,
}

impl LayoutEngine {
    /// Loads the layout model from an ONNX file on disk.
    ///
    /// The model uses ONNX external data (`PP-DocLayoutV3.onnx.data`), so it
    /// must be loaded from a path rather than from memory.
    pub fn from_file(path: &str) -> Result<Self> {
        let _ = ort::init().with_name("layout-core").commit();

        let mut builder = Session::builder()?;
        let session = builder
            .commit_from_file(path)
            .with_context(|| format!("loading layout model {path}"))?;

        Ok(Self { session })
    }

    /// The ordered class labels the model emits.
    pub fn labels(&self) -> &[&str] {
        LABELS
    }

    /// Detects layout regions in an image.
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<LayoutRegion>> {
        let ori_w = image.width() as f32;
        let ori_h = image.height() as f32;

        // Fixed 800x800 resize, scale to [0, 1] in RGB order, then HWC -> CHW
        // with a batch dimension (mean/std are identity for this export).
        let resized = image
            .resize_exact(INPUT_SIZE as u32, INPUT_SIZE as u32, FilterType::CatmullRom)
            .to_rgb8();
        let mut img = Vec::with_capacity(3 * INPUT_SIZE * INPUT_SIZE);
        for c in 0..3 {
            img.extend(resized.pixels().map(|p| p[c] as f32 / 255.0));
        }

        let input =
            Tensor::from_array((vec![1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64], img))?;

        let outputs = self
            .session
            .run(vec![("pixel_values", input.into_dyn())])
            .context("layout inference failed")?;

        let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (_, pred_boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>()?;
        let (_, order_logits) = outputs["order_logits"].try_extract_tensor::<f32>()?;

        Ok(postprocess(logits, pred_boxes, order_logits, ori_w, ori_h))
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Reading-order rank per query, ported from the image processor's
/// `_get_order_seqs` (pointer-logit votes, stable argsort).
fn get_order_seq(order_logits: &[f32]) -> Vec<usize> {
    let mut votes = vec![0.0f32; NUM_QUERIES];
    for i in 0..NUM_QUERIES {
        let mut v = 0.0;
        for j in (i + 1)..NUM_QUERIES {
            v += sigmoid(order_logits[i * NUM_QUERIES + j]);
        }
        for j in 0..i {
            v += 1.0 - sigmoid(order_logits[j * NUM_QUERIES + i]);
        }
        votes[i] = v;
    }

    let mut pointers: Vec<usize> = (0..NUM_QUERIES).collect();
    pointers.sort_by(|&a, &b| {
        votes[a]
            .partial_cmp(&votes[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut order_seq = vec![0usize; NUM_QUERIES];
    for (rank, &p) in pointers.iter().enumerate() {
        order_seq[p] = rank;
    }
    order_seq
}

fn postprocess(
    logits: &[f32],
    pred_boxes: &[f32],
    order_logits: &[f32],
    ori_w: f32,
    ori_h: f32,
) -> Vec<LayoutRegion> {
    let order_seq = get_order_seq(order_logits);

    // cxcywh (normalised) -> xyxy in original-image coordinates.
    let mut boxes = [[0.0f32; 4]; NUM_QUERIES];
    for q in 0..NUM_QUERIES {
        let cx = pred_boxes[q * 4];
        let cy = pred_boxes[q * 4 + 1];
        let w = pred_boxes[q * 4 + 2];
        let h = pred_boxes[q * 4 + 3];
        boxes[q] = [
            (cx - 0.5 * w) * ori_w,
            (cy - 0.5 * h) * ori_h,
            (cx + 0.5 * w) * ori_w,
            (cy + 0.5 * h) * ori_h,
        ];
    }

    // Flatten the (query, class) score grid, take the top NUM_QUERIES scores.
    let mut scored: Vec<(f32, usize)> = (0..NUM_QUERIES * NUM_CLASSES)
        .map(|i| (sigmoid(logits[i]), i))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(NUM_QUERIES);

    let mut dets: Vec<(usize, usize, f32, [f32; 4])> = Vec::new();
    for (score, flat_idx) in scored {
        if score < CONF_THRES {
            continue;
        }
        let label = flat_idx % NUM_CLASSES;
        let query = flat_idx / NUM_CLASSES;
        dets.push((order_seq[query], label, score, boxes[query]));
    }

    // Sort by reading order.
    dets.sort_by_key(|d| d.0);

    dets.into_iter()
        .map(|(_, label, score, b)| LayoutRegion {
            label: LABELS
                .get(label)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("cls_{label}")),
            score,
            x1: b[0].clamp(0.0, ori_w),
            y1: b[1].clamp(0.0, ori_h),
            x2: b[2].clamp(0.0, ori_w),
            y2: b[3].clamp(0.0, ori_h),
        })
        .collect()
}
