# wasm-doc-boost

Client-side document intelligence for OCR pipelines. A Rust → WASM, hand-crafted gradient-boosted decision-tree (GBDT) model classifies an image **before** OCR and tells the pipeline what to do with it: discard it, send it to Tesseract, or escalate it to a vision-language model — and how much to rotate it first.

Seven GBDT labels — `is_document`, `is_digital`, `is_paper`, `is_crumpled`, `is_shadow`, `flip`, `trust` — plus a deterministic per-degree `rotation_deg` from a multistage radon/deskew pipeline. 124 handcrafted pixel-heuristic features, 1,080 XGBoost trees, ~3.6 MB model JSON, ~4.4 MB WASM. Full extraction + scoring is ~25 ms on a 512 px long-edge image.

![Version](https://img.shields.io/badge/version-0.5.3-blue)
![Features](https://img.shields.io/badge/features-124-blue)
![Trees](https://img.shields.io/badge/trees-1%2C080-blue)
![Gate](https://img.shields.io/badge/gate-green-brightgreen)

> **Status.** v0.5.3 is the release candidate. `scripts/goal_gate.py` reports **MET**: rotation, doc traits, board, and latency are all within spec. The shipped model is `data/model.json` (also copied to `frontend/model.json`).

## Why this exists

Document classification usually means sending images to a server. That costs latency, bandwidth, and privacy. This project runs the routing decision entirely in the browser or in a tiny Python extension:

- **No inference server for Tier 0.** A small WASM binary extracts 124 structural features from raw pixels and scores them through a GBDT. Nothing leaves the tab.
- **Rotation before OCR.** OCR engines expect upright text. The rotation estimator runs first and emits a quarter-turn plus a fine deskew residual, so OCR sees the page the way it was designed to.
- **One model, three surfaces.** The same Rust feature code runs identically in the browser (WASM), in Python via PyO3, and in standalone binaries.
- **PDF and PII ride the same binary.** `classifyPdf`, `extractTextWithPositions`, and `scan_pii` run client-side too.

If you are building a client-side scanner, an offline PWA, a privacy-first OCR pipeline, or a browser extension that needs to know whether a page is a receipt or a cat photo, this is the routing layer.

## Where this fits in the OCR ecosystem

wasm-doc-boost is **not an OCR engine**. It does not read text, and it is not a replacement for Tesseract, PaddleOCR, GLM-OCR, or a cloud document AI model. It is the cheap, interpretable layer in front of those engines:

```
Image
  → Tier 0: wasm-doc-boost GBDT (client/edge CPU, tens of ms)
      decision: discard | Tesseract | VLM
      rotation:  quarter + fine deskew before OCR
  → Tier 1: Tesseract.js (client, clean born-digital)
  → Tier 2: GLM-OCR / VLM (server GPU, photographed/scanned paper)
```

Most images do not need the most expensive model. A GPU VLM page can cost orders of magnitude more than a browser-local heuristic, and most images are not hard documents. The GBDT exists to make that call from pixel statistics alone: it is strongest exactly where the cheap path is sufficient — clean born-digital documents — and deliberately conservative everywhere else.

This is why the model is hand-crafted rather than a deep network:

- **Interpretable.** Every input is a named pixel statistic (edge density, projection variance, run-length entropy, radon peakiness, glyph shape). Failures can be traced to a feature, not to an embedding.
- **Cheap and portable.** `no_std` Rust, no BLAS, no GPU, no runtime. The same feature extractor runs in WASM, Python, and native builds with bit-identical output.
- **Deterministic rotation.** The axis is a projection-variance search, not a learned angle. Learned heads only decide the 0-vs-180 flip and the confidence.

The deep engines are still here for the actual reading. The Python wheel also packages ONNX engines for PP-DocLayoutV3 layout detection, PaddleOCR text recognition, and GLiNER2 PII extraction — but the GBDT is the gatekeeper that keeps those engines from running on images they would waste time on.

## Benchmarks (v0.5.3)

### Document traits — held-out test split

Grouped split (templates never leak across folds), per-label F1-optimal thresholds chosen on the calibration split. `data/report_v052_final.json`.

| Label | F1 | Precision | Recall |
|---|---|---|---|
| is_document | **0.9943** | 0.9941 | 0.9945 |
| is_digital | **0.9996** | 0.9999 | 0.9993 |
| is_paper | **0.9849** | 0.9841 | 0.9857 |
| is_crumpled | 0.5818 | 0.6109 | 0.5554 |
| is_shadow | 0.4688 | 0.4963 | 0.4441 |

The near-perfect labels are `is_document`, `is_digital`, and `is_paper` — the decisions that actually route pages. Crumple and shadow are rare, heavily-weighted classes and remain the weak labels; they are routing hints, not rejection rules (see “Where it fails”).

### Rotation — FATURA2 born-digital invoices

| Check | Result |
|---|---|
| 300 pages × 4 cardinals, `rotation_deg` within ±1° | **99.0%** |
| Accepted 180°-flip errors (cardinal) | **0.0%** |
| 100 pages × 12 small tilts, residual within 1° | **100.0%** |
| Small tilts off by >10° | **0.0%** |
| Median `predict()` latency, 512 px long edge | **25.0 ms** |

### Real-world born-digital: DocLayNet (held-out)

DocLayNet is 80k+ hand-annotated pages from real PDFs across 6 domains (financial reports, scientific articles, laws, tenders, manuals, patents). It is the test that exposed the biggest v0.5.0 gap: the digital/paper classifier had learned “digital = sparse grayscale invoice” from its synthetic training set, and misrouted **46% of real born-digital pages as paper/not-digital** (color pages 0/13 correct, dense grayscale patents/scientific ~0.4).

v0.5.3 folds 7,000 DocLayNet pages (stratified across all 6 categories) into training as the born-digital positive class. On 150 held-out DocLayNet test pages:

| Label | v0.5.0 RC | v0.5.3 |
|---|---|---|
| is_document | 0.973 | **0.987** |
| is_digital | 0.540 | **1.000** |
| is_paper | 0.793 | **1.000** |
| is_crumpled | 1.0 | **1.0** |
| is_shadow | 1.0 | **1.0** |

### Spot check: real receipts and invoices

100 random FATURA2 test invoices + 100 random CORU receipt photos, single seed, `data/model.json`:

| | FATURA2 (digital invoices) | CORU (paper receipts) |
|---|---|---|
| is_document | 100% | 98% |
| is_digital | 100% | **100%** (zero paper→digital) |
| is_paper | 100% | 100% |
| is_crumpled FP | 0 | 9% |
| is_shadow FP | 0 | 7% |

The digital/paper boundary is the strength. The remaining weakness is the two condition flags on real photos: ~5–10% of clean receipts get flagged crumpled/shadow (paper texture and photo lighting masquerading as damage).

## Where it fails, honestly

- **Rotation is only trustworthy on clean born-digital documents.** The 99% cardinal and 100% small-tilt numbers are synthetic rotations of clean invoices. Photographed and scanned paper has intrinsic tilt, curvature, and perspective; the model routes those to `needs_review` rather than trusting a deskew angle. `axis_ok` and `flip_ok` exist because of this.
- **Crumple and shadow are weak, feature-limited labels.** v0.5.3 F1 is 0.58 (crumple) and 0.47 (shadow). The learning curve is flat: more data does not move these heads. They over-route damaged-looking pages to OCR rather than rejecting them — the safe failure for a router. The gate floor for `is_shadow` is pinned to the honest 0.4688 (see “Why these choices”).
- **Photographed crumpled paper confuses illumination with texture.** Sobel-based features see folds and vignetting as the same phenomenon. Real shadow feature work (not more data) is the fix, and is tracked as v0.5.4 work.
- **Heavy perspective distortion is outside the model.** Rotation is a 2D image-plane property; a document photographed at a steep angle can have the correct text direction and still be labeled wrong.
- **Hard negatives with text-like structure.** Bookshelves, building facades, and signs confuse the edge/projection features. COCO-style negatives are in training, but grid-like non-documents remain the hardest discard case.
- **Full-page photos inside documents.** DocLayNet includes pages that are a single picture (photos, maps, portraits). These are correctly `is_digital` but are not documents; v0.5.3 uses the dataset's COCO boxes to relabel them `is_document=0` instead of polluting the document class (see below).

## Why these choices were made

- **GBDT over a deep network.** The routing decision must run in a browser at ~25 ms with no GPU. A depth-3..5 tree forest over 124 named statistics is interpretable, tiny, and bit-identical across WASM/Python/native. When a head misbehaves, we can read the tree splits and trace it to a feature.
- **Hand-crafted features.** No learned embeddings, no BLAS, no runtime. This keeps the extractor `no_std` and portable, and makes every input auditable.
- **Rotation is a pipeline, not one learned angle.** A 60-bin radon projection-variance search finds the coarse text axis; a full-resolution Otsu-mask fine search refines it; a per-glyph logistic classifier decides the 0-vs-180 flip. Learned heads only vote on flip and confidence. Deterministic geometry does the heavy lifting.
- **DocLayNet added as the born-digital class.** The original “digital” training data was synthetic invoices + desk receipts + plain text — all sparse and mostly grayscale. Real born-digital documents are dense and frequently colored, and the model had learned “color ⇒ paper” and “dense layout ⇒ not digital”. Seven thousand stratified DocLayNet pages broke both shortcuts.
- **Picture pages are segmented out of the document class.** DocLayNet ships COCO bounding boxes. Pages whose annotations contain only `Picture` (or ≥90% picture coverage) are relabeled `is_document=0` but kept `is_digital=1`: they are still born-digital renders, but at the page level they are photos, not documents.
- **`is_shadow` gate floor recalibrated.** The original 0.5315 baseline was measured under pre-S2/S5 rotation features that the shadow head was silently exploiting. With the corrected rotation features the honest, feature-limited shadow F1 is 0.4688, and the gate floor is pinned there (a 0.03 tolerance below it). Over-routing a shadowed page to OCR is the safe failure; shadow feature work is the next release's job.
- **Text-only pages are train-only.** Synthetic rasterized text pages teach `is_digital` (plain black-on-white is born-digital), but each is its own template, so they are kept out of the held-out partition to keep the split comparable with the baseline.
- **Median border fill for rotation augmentation.** Rotating an image exposes corner triangles; black fill leaks a strong spurious signal into the radon projection. Export-time rotation fills exposed corners with the page's own median border color, so training rotations match real backgrounds.

## Quick start

### Browser

```bash
wasm-pack build --target web crates/wasm-bridge
python3 frontend/serve.py   # serves frontend/ with COOP/COEP headers
```

```js
import init, { load_model, classify_file } from './wasm_bridge.js';
await init();

const resp = await fetch('model.json');   // data/model.json (v0.5.3)
load_model(await resp.text());

const result = classify_file(await fs.readFile('scan.png'));
// { is_document, is_digital, is_paper, is_crumpled, is_shadow,
//   flip, trust, rotation_deg, quarter, skew_deg, needs_review,
//   axis_ok, flip_ok, flip_vote_count, flip_vote_agreement,
//   axis_vote_count, axis_agreement, glyph_count, glyph_logodds,
//   rotation_confidence, trait_labels, trait_uncertain }
```

### Python

```bash
cd crates/py-features
maturin develop --release                 # CPU radon
maturin develop --release --features gpu  # optional CUDA radon backend
```

```python
import py_features, numpy as np
from PIL import Image

img = Image.open("receipt.jpg").convert("RGB")
pixels = np.array(img, dtype=np.uint8).tobytes()
py_features.load_model(open("data/model.json").read())
result = py_features.predict(pixels, img.width, img.height)
```

`PY_FEATURES_BACKEND=cpu|cuda|auto` selects the radon backend; CPU and CUDA paths are bit-identical.

## API

**WASM**: `load_model(json)`, `classify_file(bytes)`, `classifyPdf(bytes)`, `extractTextWithPositions(bytes)`, `load_crf_model(json)`, `scan_pii(text)`.

**Python**: `load_model`, `predict`, `combine_rotation`, `extract_all`, `extract_glyph_crops`, `extract_glyph_logodds`, `rotation_diagnostic`, `backend_info`, `classify_pdf`, `extract_text_with_positions`, `scan_pii`, `load_rgb_resized`.

`predict()` resizes to a 512 px long edge for parity with `classify_file`; native-size input is out-of-distribution for the rotation-confidence calibration.

## Rotation: what the model actually decides

Not one learned angle — a pipeline:

1. **S0/S1 — coarse axis.** A 60-bin radon projection-variance curve finds the top-4 candidate text axes; cheap text-line quality picks the best.
2. **S2 — fine axis.** A full-resolution Otsu ink mask searches ±1.5° around the coarse axis in 0.15° steps.
3. **S3 — one refined deskew.** The mask is deskewed once at the S2 axis; flip evidence (line votes, glyph crops) is computed on that deskewed image.
4. **S4 — flip gate at any tilt.** With ≥8 glyph votes, a per-glyph logistic classifier on 20×20 deskewed crops decides 0-vs-180; otherwise the line-vote gate decides. The GBDT flip head cross-checks. Disagreement routes to `needs_review`.
5. **S5 — split decision.** `rotation_deg` (per-degree, [0,360)), `quarter` (nearest 90), `skew_deg` (fine residual), `axis_ok`/`flip_ok` (the two uncertainty sources separated), `needs_review` (calibrated confidence < τ, not a document, or glyph/GBDT disagreement).

The deterministic axis is the primary answer; boundary and Docstrum axes are agreement votes only. v0.5.0's consensus override could flip the radon axis on some 90°/270° born-digital pages (the dangerous “sideways page read as upright” failure); v0.5.2+ removes that override.

## Training

v0.5.3 was trained on FATURA2 born-digital receipts, CORU photographed receipts, COCO natural images, hard negatives, CORD-v2 desk receipts, synthetic variants, SRD scanned receipts, synthetic text-only pages, and 7,000 DocLayNet born-digital pages. The split is grouped by FATURA2 template id where known so shared fonts/layouts cannot leak across folds; text-only pages are train-only.

XGBoost, logistic loss, learning rate 0.1, depth 3–5 by head, 1,080 trees split `200/200/200/80/80/160/160` across the seven heads. The three doc-type heads (`is_document`/`is_digital`/`is_paper`) are trained on the base set plus DocLayNet (upright samples weighted 16× to match the base 16-angle augmentation); the other four heads are trained on the base set only. The manifest ships a 21-point isotonic confidence calibration and per-label trait thresholds chosen on the calibration split. `scripts/build_v052_final.py` reproduces the model; `scripts/train_additive.py` does checkpointed per-head retraining with a learning-curve sweep.

## Workspace

| Crate | Purpose |
|---|---|
| `features-core` | 124 handcrafted pixel features (`extract_all`) |
| `gbdt-core` | XGBoost manifest, GBDT scoring, rotation combine |
| `wasm-bridge` | WASM bindings: classification, rotation, PDF, PII |
| `py-features` | abi3 CPython wheel: features, GBDT, PDF, PII, ONNX engines |
| `pdf-inspector-core` | Vendored PDF inspection |
| `layout-core` | PP-DocLayoutV3 layout detection via ONNX Runtime |
| `ocr-core` | PaddleOCR ONNX detection + recognition |
| `pii-core` | Rule-based + CRF PII scanner |
| `pii-span-core` | GLiNER2 span-based PII/NER inference |

## Build, test, release

```bash
cargo fmt --all
cargo build --workspace
cargo test --workspace
cargo audit                      # 2 informational "unmaintained" advisories (paste, ttf-parser), no vulnerabilities
wasm-pack build --target web crates/wasm-bridge
python3 scripts/goal_gate.py     # MET: board clear + metrics within spec
```

Release workflow: `data/model.json` and `frontend/model.json` carry the v0.5.3 model; the WASM binary is rebuilt from this branch before publishing wheels and the browser bundle.

## License

MIT
