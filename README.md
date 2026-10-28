# wasm-doc-boost

Client-side document classification via Rust → WASM + GBDT inference. Seven GBDT labels — `is_document`, `is_digital`, `is_paper`, `is_crumpled`, `is_shadow`, `flip`, `trust` — plus a deterministic per-degree `rotation_deg` (radon axis + fold-snap + glyph/vote flip gate). 124 handcrafted pixel-heuristic features, 1,080 XGBoost trees, 4.5 MB WASM (2.0 MB gzip), 2.7 MB model JSON (317 KB gzip). In-browser privacy, sub-millisecond model scoring.

![WASM size](https://img.shields.io/badge/wasm-4.5%20MB%20%7C%202.0%20MB%20gzip-blue)
![Features](https://img.shields.io/badge/features-124-blue)
![Trees](https://img.shields.io/badge/trees-1%2C080-blue)
![Version](https://img.shields.io/badge/version-0.4.0-green)
![Rotation](https://img.shields.io/badge/rotation-1200%2F1200%20cardinal-brightgreen)

## Why This Exists

Document classification usually requires sending images to a server. That means latency, bandwidth, and a privacy trade-off: someone else's hardware sees every document you classify.

wasm-doc-boost runs entirely in the browser. A 4.5 MB WASM binary extracts 124 structural features from pixel data and scores them through a gradient-boosted decision tree ensemble — no GPU, no network, no inference server. The model JSON is 2.7 MB (317 KB gzip), so the whole classifier loads from a CDN in a couple of seconds on a typical connection, and the GBDT scoring itself takes well under a millisecond per image (end-to-end extraction + scoring is single-digit milliseconds at 512 px — see Performance).

If you are building a client-side document scanner, an offline-capable PWA, a privacy-first OCR pipeline, or a browser extension that needs to know whether the current page contains a receipt versus a photo of a cat, this is the library you want.

## Quick Start

```bash
# Build the WASM package
wasm-pack build --target web crates/wasm-bridge
```

Serve the `frontend/` directory with COOP/COEP headers (required for SharedArrayBuffer — the OCR pipeline uses threads):

```bash
python3 frontend/serve.py
```

```js
import init, { load_model_v2, classify_file_v2, model_version } from './wasm_bridge.js';

await init();

// Load the v3 manifest (XGBoost dump JSON: 7 heads + per-degree rotation)
const resp = await fetch('model.json');
load_model_v2(await resp.text());
if (model_version() !== '0.4.0') throw new Error('expected model 0.4.0');

// Classify an image from a Uint8Array (PNG or JPEG)
const fileBytes = await fs.readFile('scan.png');  // or file input, drop event, etc.
const result = classify_file_v2(fileBytes);
// {
//   is_document: 0.97, is_digital: 0.92, is_paper: 0.03,
//   is_crumpled: 0.01, is_shadow: 0.12,
//   flip: 0.02, trust: 0.95,
//   rotation_deg: 0.01, rotation_confidence: 0.94, needs_review: false
// }
```

### Python

```bash
# Prebuilt wheel (CPython 3.8+), no Rust toolchain required.
# Linux x86_64:
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.4.0/py_features-0.4.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# Linux aarch64:
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.4.0/py_features-0.4.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# macOS Intel (x86_64):
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.4.0/py_features-0.4.0-cp38-abi3-macosx_10_12_x86_64.whl
# macOS Apple Silicon (arm64):
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.4.0/py_features-0.4.0-cp38-abi3-macosx_11_0_arm64.whl

# Or build from source:
cd crates/py-features && maturin develop --release
```

```python
import py_features
import numpy as np
from PIL import Image

img = Image.open("receipt.jpg").convert("RGB")
pixels = np.array(img, dtype=np.uint8).tobytes()
features = py_features.extract_all_v2(pixels, img.width, img.height)
# features is a list[float] of 124 values
# (py_features.extract_all is the legacy 103-feature vector)

# Same model + combine as the browser: load the v3 manifest and predict.
py_features.load_model_v2(open("model.json").read())
result = py_features.predict_v2(pixels, img.width, img.height)
# result = {is_document, is_digital, is_paper, is_crumpled, is_shadow,
#           flip, trust, rotation_deg, rotation_confidence, needs_review,
#           flip_vote_count, flip_vote_agreement, axis_vote_count,
#           axis_agreement, glyph_count, glyph_logodds}
```

### API

- `load_model_v2(json)` — parses the v3 manifest `{labels, tau, version, trees, trees_per_label, confidence_calibration}`; 7-label multi-output GBDT (5 document traits + flip + trust)
- `classify_file_v2(bytes)` — decodes PNG/JPEG, resizes to max 512 px long edge, extracts 124 features, runs the 7 heads, then combines the deterministic radon axis with the per-glyph/vote flip gate into `rotation_deg` ∈ [0, 360), `rotation_confidence`, and `needs_review`
- `model_version()` — the loaded manifest's `version` (a release-tag string such as `"0.4.0"`; empty for an unversioned manifest)
- `load_model(json)` / `classify_file(bytes)` — legacy 9-label API (`rotation_0/90/180/270`); kept for the old wheel, not used by the frontend
- `scan_pii(text)` — rule-based + CRF PII scanner (credit cards, SSNs, emails, phones, addresses, names)

The same functions exist in `py_features` (`load_model_v2`, `model_version`, `predict_v2`, `combine_rotation_v2`), so a backend can score pixel-identical features and get the same predictions as the browser.

## Cost-Controlled OCR Ladder

Using a single OCR pipeline for every image is wasteful. An LLM-grade OCR model running on a GPU might cost 100-1000× more per page than a browser-local heuristic, and most images don't need it. wasm-doc-boost implements a three-tier ladder:

```
Image → [Tier 0: WASM GBDT] → decision
         ├─ is_document ≤ 0.3?  → discard (cat photo, meme, screenshot)
         ├─ is_digital ≥ 0.9?   → [Tier 1: Tesseract.js] → PII scan → done
         └─ is_paper ≥ 0.5?     → [Tier 2: GLM-OCR GPU]  → PII scan → done
                                 ↘ is_crumpled ≥ 0.5 OR is_shadow ≥ 0.5?
                                    → [Tier 2: GLM-OCR] (Tesseract struggles on these)
```

**Tier 0 — WASM GBDT classifier** (sub-millisecond scoring, 4.5 MB WASM + 2.7 MB model download, costs nothing after load). Runs in the browser's main thread. Determines *what kind of image this is* before any OCR engine spins up. The 7 labels plus `rotation_deg` answer every routing question: Is this a document at all? Digital or paper? Rotated? Crumpled/shadowed (i.e., will Tesseract fail)? `rotation_deg` also tells the OCR tier which way to orient the image.

**Tier 1 — Tesseract.js** (~500 ms–2 s in-browser, free, no network). Offline-capable, privacy-preserving, good enough for clean digital receipts and screenshots. The frontend automatically uses Tesseract for digital documents and falls through to Tier 2 for paper, crumpled, or shadowed images.

**Tier 2 — GLM-OCR via GPU backend** (~1–3 s per page, requires container with NVIDIA GPU). High-quality multimodal OCR for photographed paper documents. Only invoked when the GBDT classifier says it's needed — paper receipts shot on a kitchen counter, crumpled invoices, shadowed documents. The backend is included in this repo (FastAPI + zai-org/GLM-OCR) and runs via `podman-compose`.

The ladder is embedded in the frontend's `DocBoost.analyze()` function: GBDT runs first, rotation is corrected, Tesseract and GLM-OCR fire in parallel (GLM-OCR is skippable if unavailable), then results are merged and scanned for PII.

### Key Insight: Rotation Before OCR

OCR engines expect upright text. The GBDT rotation detector runs *before* any OCR — the image is physically rotated before being sent to Tesseract or GLM-OCR. This is ~100× cheaper than running OCR on a rotated image and trying to correct the output text.

### Cost Model

| Tier | Engine               | Cost per 1M pages | Latency     | Best for                |
|------|---------------------|-------------------|-------------|------------------------|
| 0    | WASM GBDT           | $0 (client)       | < 1 ms      | Routing decision        |
| 1    | Tesseract.js        | $0 (client)       | 500 ms–2 s  | Digital receipts        |
| 2    | GLM-OCR GPU         | $100–$500 (GPU)   | 1–3 s       | Paper, crumple, shadow  |

A naive pipeline that sends every image to a GPU OCR endpoint pays for 1,000,000 GPU inferences. With the ladder, only paper documents (~11% of the training set) and crumpled/shadowed images (~5%) reach Tier 2 — an 85% cost reduction from routing alone.

## GLM-OCR Backend

The repo includes a containerized GLM-OCR backend that serves as the Tier 2 OCR engine. It wraps [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) in a FastAPI server with region detection and per-region OCR.

### Setup

```bash
# Build the container (one-time, ~15 min)
podman build -f backend/Containerfile -t localhost/wasm-doc-boost-backend:latest .

# Start both services
podman-compose up -d
```

The backend service:
- Exposes `POST /ocr` — accepts `{image: <base64>, mode: "regions"|"full", prompt: "..."}`, returns text regions with bounding boxes
- Exposes `GET /health` — health check, used by the frontend to discover availability
- Watches `./backend/` for live code changes (`--reload` flag)
- Mounts HuggingFace model cache from `/home/jm/.cache/huggingface` (first run downloads ~2 GB)

The frontend's `createDocBoost()` constructor health-checks the GLM backend on init. If unreachable, GLM-OCR is gracefully disabled and Tesseract.js handles all OCR.

### API

```http
POST /ocr
Content-Type: application/json

{
  "image": "<base64-encoded JPEG/PNG>",
  "prompt": "Text Recognition:",
  "mode": "regions"
}
```

Response (regions mode):
```json
{
  "regions": [
    {"text": "INVOICE #1234", "bbox": [120, 45, 380, 82]},
    {"text": "Date: 2025-03-15", "bbox": [118, 90, 340, 120]}
  ],
  "_meta": {"engine": "GLM-OCR (cuda)", "latency_ms": 1423}
}
```

Requirements: NVIDIA GPU with CUDA, Podman (or Docker with nvidia-container-toolkit), ~4 GB VRAM for the 0.9B parameter model.

### Experimental: In-Browser GLM-OCR via ONNX/WebGPU (Backlogged)

An earlier attempt ran GLM-OCR entirely client-side using ONNX Runtime Web + WebGPU. The ONNX models (2.7 GB vendored, both fp16 and q4f16 variants) are still in `frontend/vendor/glm-ocr-onnx/` along with test harnesses (`test_glm_ocr.mjs`, `test_glm_node.mjs`, `test_onnx_node.mjs`), but this path is on hold for three reasons:

- **Operator support**: the q4f16 models (~624 MB) use `GatherBlockQuantized` ops that require ONNX Runtime's full JSEP bundle (398 KB) and WebGPU shader extensions — the slim `ort.webgpu.bundle.min.mjs` (113 KB) lacks them.
- **Memory**: the fp16 models are 2.1 GB total (decoder 1.1 GB, vision encoder 829 MB, embeddings 175 MB). Loading into a browser tab hits OOM even with `--max-old-space-size=8192` and Playwright's `--enable-unsafe-webgpu` flags. The q4f16 models fit at 624 MB but crash on `GatherBlockQuantized` shader initialization.
- **INT8 quantization** (brad-agi/glm-ocr-onnx-webgpu) uses `ConvInteger` ops, which aren't supported in the browser build of ONNX Runtime.

If ONNX Runtime Web adds JSEP kernels for `GatherBlockQuantized` on WebGPU, or if a float16 WebGPU-compatible export with smaller memory footprint becomes available, this path could replace the GPU backend entirely — bringing Tier 2 OCR client-side. The repo has everything else wired up (serve.py serves `.onnx` files with proper MIME types and no-cache headers, the importmap is ready, and the test harnesses validate the full cascade).

## Model Performance (manifest v6, 124 features, 1,080 trees)

Training: 305,600 samples (19,100 images × 16 rotations: the 4 cardinals + 6 seeded-random flip pairs) across 6 sources — FATURA2 digital receipts (52%), COCO natural images (26%), hard negatives (10%), CORD-v2 desk receipts (5%), synthetic variants (5%), SRD scanned receipts (1%). Every non-cardinal sample has its exact 180° flip counterpart at the same axis, so the flip head sees matched upright/upside-down pairs. Split 80/10/10 train/cal/test grouped by FATURA2 template id (50 templates, recovered by `scripts/cluster_fatura_templates2.py`) where known, else by base image — whole templates stay together so shared fonts/layouts cannot leak across folds. XGBoost lr=0.1, depth 3–5 by head, 1,080 trees split unevenly across the seven heads (200/200/200/80/80/160/160). `data/model_v6.json` carries `version: "0.4.0"` — the release tag the model ships in, not a format counter; the WASM `model_version()` export returns it. The manifest also carries a 21-point isotonic `confidence_calibration` curve fitted on the cal split.

**Headline results, measured on the shipped WASM build:**

- **Per-degree rotation on clean born-digital documents**: 0/1,000 upright invoices flipped, 99.9% within 1° (the one miss is flagged for review), and 1,200/1,200 rotated-copy cardinals recovered exactly.
- **Perfect document/digital separation**: `is_document` AUC 1.0000, `is_digital` AUC 1.0000.
- **Sub-millisecond GBDT scoring** from a 4.5 MB WASM + 2.7 MB model download (2.0 MB + 317 KB gzip).

### Document Classification (5 trait heads)

Held-out test split (33,744 samples); 5 document-trait heads of the 7 trained on `data/features_v6.npz`:

| Label        | AUC    | F1     | Precision | Recall |
|-------------|--------|--------|-----------|--------|
| is_document | 1.0000 | 0.9976 | 0.9984    | 0.9967 |
| is_digital  | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| is_paper    | 0.9999 | 0.9830 | 0.9771    | 0.9890 |
| is_crumpled | 0.9892 | 0.5558 | 0.3982    | 0.9195 |
| is_shadow   | 0.9890 | 0.5315 | 0.3797    | 0.8856 |

`is_document` and `is_digital` are near-perfect — structural properties visible in pixel statistics. `is_paper` is strong at 0.98 F1. Crumple and shadow are rare classes with extreme imbalance: the model catches most cases (recall 0.92 / 0.89) but produces false positives (precision 0.40 / 0.38). This is the right trade-off for a routing pre-filter — you would rather over-route to OCR than lose a crumpled receipt.

### Rotation (per-degree: deterministic axis + GBDT flip/trust)

Rotation is no longer four one-hot labels. It is a per-degree estimate in [0, 360):

1. **Axis (deterministic radon)** — a 60-bin projection response curve at 3° resolution, parabolic-refined. The top-4 local maxima are each scored by text-line quality (n_lines, then total line ink) and the best wins, fixing near-ties where a perpendicular structure (table rule, page edge) out-peaks the text axis.
2. **Fold-snap** — the axis is mod-180 blind: for exactly-horizontal text the argmax lands randomly on bin 0 (~0.01°) or bin 59 (~179.99°). `combine_rotation` snaps `<1° or >179°` to 0 before combining. (Before the fix, this one bug flipped 55% of upright invoices.)
3. **Flip gate (glyph + line-vote)** — at a cardinal axis the deterministic textline evidence is primary. When ≥8 glyphs voted, a per-glyph logistic classifier on 20×20 deskewed glyph crops (L2 logistic + page-level calibration on `(sum_logodds, glyph_count)`) decides 0-vs-180 — this is the only channel that works on all-caps/digit invoices where ascender/descender ink mass is ~0 and vote_frac is a coin flip. With fewer glyphs it falls back to the line-vote gate (≥5 lines, |vote_frac − 0.5| > 0.05). At non-cardinal axes the GBDT flip head decides.
4. **Trust head (GBDT) + isotonic calibration** — `raw_confidence = trust × flip_conf`, then the manifest's 21-point isotonic curve maps it to an honest P(within 1°). `needs_review` fires when confidence < tau (0.9), not a document, or the glyph gate and GBDT flip head disagree.

**Measured on the real WASM, gold upright FATURA2 (1,000 born-digital invoices, gold = 0°):**

| Metric | Value |
|---|---|
| is_document | 1000 / 1000 |
| rotation_deg within 1° | 99.9% (999 / 1000) |
| 180°-flip errors | 0% (0 / 1000) |
| needs_review | 0.1% (1 / 1000 — the single miss, flagged) |
| mean confidence | 1.000 |

**Measured on the real WASM, rotated copies (300 held-out fatura2 test images × 4 cardinals = 1,200):**

| Metric | Value |
|---|---|
| axis (mod 180) exact | 1200 / 1200 |
| rotation_deg exact recovery | 1200 / 1200 |
| 0° / 90° / 180° / 270° exact | 300/300 each |

The 0 flip errors on born-digital invoices come from the deterministic flip gate, not a better flip head. At a cardinal axis the line-vote gate (or the per-glyph shape gate when ≥8 glyphs voted) overrides the GBDT flip head; the GBDT head alone is still ~76% accurate and is used off-cardinal. The per-glyph gate is the fix for the v0.3.0 gap: it works on all-caps/digit invoices where ascender/descender ink mass is ~0, so vote_frac was a coin flip there. When the glyph gate and the GBDT flip head disagree, the answer routes to `needs_review` rather than trusting the weaker per-glyph signal alone.

Photographed and scanned documents still route through the GBDT flip head off-cardinal and must be treated as needing review — PLAN.md Path B (quantized vision encoder) is the escalation for those.

### What changed in v0.4.0

- **Retrained on a template-grouped split.** FATURA2's 50 shared templates previously leaked across train/cal/test when splitting by base image and inflated held-out numbers. `data/fatura2_template_ids.json` keeps whole templates together, so the split is honest.
- **Per-glyph flip classifier.** A 20×20 logistic glyph classifier (weights baked into `crates/features-core/src/glyph.rs`) plus a page-level calibration on `(sum_logodds, glyph_count)` fixes 180° flips on all-caps/digit invoices where line-vote evidence is a coin flip. The gold upright set stays at 0 flip errors, now from real evidence rather than the v0.3.0 "all three signals agree" body-fraction guard, which is removed.
- **Honest confidence.** A 21-point isotonic calibration curve ships in the manifest; `rotation_confidence` is now P(within 1°), not the raw `trust × flip_conf` product.
- **Richer diagnostics.** `predict_v2` / `classify_file_v2` now return 16 keys — the 7 label probabilities plus `rotation_deg`, `rotation_confidence`, `needs_review`, `flip_vote_count`, `flip_vote_agreement`, `axis_vote_count`, `axis_agreement`, `glyph_count`, `glyph_logodds`.
- **Version bump.** Workspace crates, the frontend package, and the Python wheel are now 0.4.0.

### What changed in v0.3.0

- **Born-digital 180° flips eliminated.** The deterministic flip guard in `combine_rotation` takes 180°-flip errors on the 1,000-image gold set from **65 (v0.2.0) → 0 (v0.3.0)** — without retraining: the same v4 manifest, the same 116 features, a stricter evidence gate on the digital path.
- **Cleaner WASM surface.** The `debug_features_v2` export (a development aid, not a public API) is removed from the shipped build.
- **Version bump.** Workspace crates, the frontend package, and the Python wheel are now 0.3.0.

### What changed vs the old 9-label model

The four one-hot rotation labels (`rotation_0/90/180/270`, ~0.74 F1) are gone, replaced by the deterministic radon axis + `flip`/`trust` heads — sub-degree accuracy on clean digital documents instead of a 4-way guess. The feature vector grew from 103 to 124 (adds radon response summary, textline ascender/descender, boundary minAreaRect, docstrum axis, and 8 tranche-A/tranche-B flip features).

## Performance

The Sobel gradient computation — the most expensive single pass in the feature pipeline — was optimized with WASM SIMD128 intrinsics (`f64x2`). The SIMD path processes 2 pixels per inner-loop iteration instead of 1.

| Image Size  | Scalar (µs) | SIMD (µs) | Speedup | SIMD Throughput |
|-------------|-------------|-----------|---------|-----------------|
| 256×256     | 1,830       | 1,684     | 1.09×   | 38.9 Mpix/s     |
| 512×512     | 7,345       | 6,915     | 1.06×   | 37.9 Mpix/s     |
| 1024×768    | 22,670      | 21,110    | 1.07×   | 37.3 Mpix/s     |
| 1920×1080   | 59,267      | 54,433    | 1.09×   | 38.1 Mpix/s     |

Measured on Node.js v24.12.0 (V8 13.6) with the same synthetic sine-wave image for both paths. The SIMD speedup is modest (1.06–1.09×) because `sqrt` and `atan2` — which dominate the Sobel inner loop — remain scalar. The gradient arithmetic (loads, adds, muls) benefits fully from 2-wide SIMD, but the overall win is diluted by the transcendental operations and the shared `pad_gray()` allocation.

The scalar path is used for native (py-features via PyO3) and serves as the fallback when WASM SIMD128 is unavailable. Both paths produce bit-identical results — verified by comparing `gx`, `gy`, `magnitudes`, and `orientations` element-by-element across a 256×256 test image.

Combined with Sobel-pass sharing (one `SobelResult` instead of seven independent Sobel calls), the end-to-end feature extraction speedup over the previous release is approximately 3–4× on WASM and 2–3× on native.

## Where It Fails (And Why)

- **180° flips on photographed/scanned documents** — the flip head (ascender/descender heuristics) is the weak point there. On upright born-digital invoices the glyph/vote gate eliminates all 1,000/1,000 flip errors, but that gate only applies at a cardinal axis. Photographed and scanned documents still route through the raw flip head (held-out ~76% accuracy) off-cardinal, and `needs_review` does not catch all of them because the head is confidently wrong. Path B (vision encoder) is the planned fix.
- **Photographed crumpled paper** — illumination gradients look like shadows to the Sobel-based features. The edge-density and Otsu-threshold features can't distinguish a fold from vignetting. Solution: more training data, not more features.
- **Heavy perspective distortion** — the model sees rotation as a 2D image-plane property. If the document is photographed at a 45° angle, the rotation label may be wrong even though the "true" text direction is correct. Perspective correction should happen upstream.
- **Crumple precision** — 30% precision means most images flagged as crumpled are actually flat. This is acceptable for a pre-filter that prompts re-scanning but not for automated rejection.
- **Hard negatives** — COCO images with grid-like or text-like structure (bookshelves, building facades, signs) confuse the edge-direction and projection features. The model sees rows of "text" that are actually bricks.

## Feature Taxonomy

Every feature is computed from raw pixel data — no deep learning, no learned embeddings, no external dependencies beyond `libm` for `no_std` math. This is deliberate: the same Rust code runs identically in the browser (WASM), in Python via PyO3, and in standalone binaries. The canonical vector is `extract_all_v2`: 124 features = 100 core heuristics + 24 rotation signals.

### Color (5 features)

Per-channel mean/std/skewness, grayscale statistics, colorfulness (RGB dispersion), and saturation stats. Separates digital graphics (high saturation, narrow histograms) from photographs (low saturation, wide histograms).

### Edges (20 features, single Sobel pass)

Laplacian variance (blur detector), Sobel magnitude statistics (mean, std, 90th percentile), edge density after Otsu thresholding, 8-bin edge direction histogram, horizontal/vertical edge ratio, Canny-like thin-edge density, structure tensor features (dominant angle, coherence, gradient energy), and Sobel circular statistics. All edge features share a single SobelResult computed in one pass over the grayscale image.

### Texture (4 features)

DCT low-frequency energy ratio (frequency-domain texture density), LBP histogram (local binary patterns, rotation-invariant), GLCM contrast/correlation/energy/homogeneity (gray-level co-occurrence), and fractal dimension (box-counting). Distinguishes text (repeating high-frequency patterns) from smooth photo gradients.

### Noise (3 features)

High-pass residual variance (sensor noise estimation), JPEG blockiness (8×8 grid artifact detection), and gradient SNR. Separates clean digital graphics from compressed/denoised photographs.

### Shadow (6 features)

Illumination gradient direction and magnitude, shadow coverage via Otsu thresholding on low-pass filtered image, and regional contrast features. Detects the vignetting and uneven illumination common in photographs of paper documents.

### Crumple (10 features)

LBP variance (creases create local texture), edge-density standard deviation across grid cells, texture anisotropy, and peak local entropy. Identifies the irregular deformation patterns of crumpled paper.

### Document Structure (6 features)

Aspect ratio, text-line spacing via horizontal projection FFT, margin ratios, ink coverage fraction, and connected-component statistics. Operates on the binarized image after Otsu thresholding.

### Projection (5 features)

Grayscale row/column projection variance, variance ratio, and row/column entropy. Clean text documents show high row variance (alternating text/blank rows) and low column variance. Rotation swaps this pattern — the ratio is a strong rotation signal.

### Ink Mask & Run-Length (9 features)

Ink coverage (fraction of pixels below Otsu threshold), mean ink intensity, horizontal and vertical run-length statistics (mean, std, entropy), and a Docstrum-inspired nearest-neighbor rotation angle. Run-length entropy is the key differentiator: text has a characteristic bimodal run-length distribution (character strokes ~3-5 px, word gaps ~10-20 px), while photos have nearly uniform run lengths. The Docstrum angle links each ink pixel to its nearest neighbor in the row below and computes the circular mean of those angles — horizontal text lines produce angles near ±π/2, rotated text shifts them.

### Rotation Signals (24 features) — the per-degree rework

Radon response curve summary (best axis, peakiness, peak margin, concentration — the deterministic axis lives at feature index 100), textline ascender/descender mass + line-vote fraction (the flip head's evidence), boundary minimum-area-rectangle axis vote (visible-boundary complement to the radon axis), an unfolded Docstrum axis, and 8 tranche-A/tranche-B flip features (signed vertical gradient, baseline edge weight, gradient skew, top/bottom edge sharpness, character centroid offset, top-heavy fraction, asymmetry strength, body vertical bias). The taxonomy above is a representative selection of the 124, not an exhaustive list.

## Build

```bash
wasm-pack build --target web crates/wasm-bridge     # browser
wasm-pack build --target nodejs crates/wasm-bridge  # Node.js

# With SIMD auto-vectorization (LLVM autovectorizes the Sobel inner loop)
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web crates/wasm-bridge
```

## Test

```bash
cargo test -p features-core                         # Rust unit tests (58)
cargo test -p wasm-bridge                           # WASM integration tests (10)
```

## License

MIT
