# wasm-doc-boost

Client-side document classification via Rust → WASM + GBDT inference. Seven GBDT labels — `is_document`, `is_digital`, `is_paper`, `is_crumpled`, `is_shadow`, `flip`, `trust` — plus a deterministic per-degree `rotation_deg` (radon axis + fold-snap + flip head). 116 handcrafted pixel-heuristic features, 1,080 XGBoost trees, 4.5 MB WASM (2.0 MB gzip), 2.3 MB model JSON (291 KB gzip). In-browser privacy, sub-millisecond model scoring.

![WASM size](https://img.shields.io/badge/wasm-4.5%20MB%20%7C%202.0%20MB%20gzip-blue)
![Features](https://img.shields.io/badge/features-116-blue)
![Trees](https://img.shields.io/badge/trees-1%2C080-blue)
![Version](https://img.shields.io/badge/version-0.3.0-green)
![Rotation](https://img.shields.io/badge/rotation-0.335%C2%B0%20axis%20MAE-brightgreen)

## Why This Exists

Document classification usually requires sending images to a server. That means latency, bandwidth, and a privacy trade-off: someone else's hardware sees every document you classify.

wasm-doc-boost runs entirely in the browser. A 4.5 MB WASM binary extracts 116 structural features from pixel data and scores them through a gradient-boosted decision tree ensemble — no GPU, no network, no inference server. The model JSON is 2.3 MB (291 KB gzip), so the whole classifier loads from a CDN in a couple of seconds on a typical connection, and the GBDT scoring itself takes well under a millisecond per image (end-to-end extraction + scoring is single-digit milliseconds at 512 px — see Performance).

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

// Load the v4 model manifest (XGBoost dump JSON: 7 heads + per-degree rotation)
const resp = await fetch('model.json');
load_model_v2(await resp.text());
if (model_version() !== 4) throw new Error('expected model v4');

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
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.3.0/py_features-0.3.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# Linux aarch64:
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.3.0/py_features-0.3.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# macOS Intel (x86_64):
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.3.0/py_features-0.3.0-cp38-abi3-macosx_10_12_x86_64.whl
# macOS Apple Silicon (arm64):
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.3.0/py_features-0.3.0-cp38-abi3-macosx_11_0_arm64.whl

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
# features is a list[float] of 116 values
# (py_features.extract_all is the legacy 103-feature vector)

# Same model + combine as the browser: load the v4 manifest and predict.
py_features.load_model_v2(open("model.json").read())
result = py_features.predict_v2(pixels, img.width, img.height)
# result = {is_document, is_digital, is_paper, is_crumpled, is_shadow,
#           flip, trust, rotation_deg, rotation_confidence, needs_review}
```

### API

- `load_model_v2(json)` — parses the v4 manifest `{labels, tau, version, trees, trees_per_label}`; 7-label multi-output GBDT (5 document traits + flip + trust)
- `classify_file_v2(bytes)` — decodes PNG/JPEG, resizes to max 512 px long edge, extracts 116 features, runs the 7 heads, then combines the deterministic radon axis with the flip head into `rotation_deg` ∈ [0, 360), `rotation_confidence`, and `needs_review`
- `model_version()` — the loaded manifest's `version` (4 for the current model; 0 for an unversioned manifest)
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

**Tier 0 — WASM GBDT classifier** (sub-millisecond scoring, 4.5 MB WASM + 2.3 MB model download, costs nothing after load). Runs in the browser's main thread. Determines *what kind of image this is* before any OCR engine spins up. The 7 labels plus `rotation_deg` answer every routing question: Is this a document at all? Digital or paper? Rotated? Crumpled/shadowed (i.e., will Tesseract fail)? `rotation_deg` also tells the OCR tier which way to orient the image.

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

## Model Performance (manifest v4, 116 features, 1,080 trees)

Training: 152,800 samples (19,100 images × 8 random rotations) across 6 sources — FATURA2 digital receipts (52%), COCO natural images (26%), hard negatives (10%), CORD-v2 desk receipts (5%), synthetic variants (5%), SRD scanned receipts (1%). Rotations are continuous random degrees, so the axis and flip heads see the full circle rather than four cardinals. Split 80/10/10 train/cal/test grouped by base image — no data leak from rotation augmentation. XGBoost lr=0.1, depth 3–5 by head, 1,080 trees split unevenly across the seven heads (200/200/200/80/80/160/160). `data/model_v4.json` carries `version: 4`; the WASM `model_version()` export checks it.

**Headline results, measured on the shipped WASM build:**

- **Per-degree rotation on clean born-digital documents: 100% within 1°** — 0/1,000 upright invoices mis-oriented, radon-axis circular MAE 0.335° (sub-degree).
- **Perfect document/digital separation**: `is_document` AUC 0.9999, `is_digital` AUC 1.0000.
- **Sub-millisecond GBDT scoring** from a 4.5 MB WASM + 2.3 MB model download (2.0 MB + 291 KB gzip).

### Document Classification (5 trait heads)

Held-out test split, `data/report_v5.json`:

| Label        | AUC    | F1     | Precision | Recall |
|-------------|--------|--------|-----------|--------|
| is_document | 0.9999 | 0.9974 | 0.9967    | 0.9980 |
| is_digital  | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| is_paper    | 0.9998 | 0.9705 | 0.9518    | 0.9900 |
| is_crumpled | 0.9840 | 0.4450 | 0.2987    | 0.8719 |
| is_shadow   | 0.9913 | 0.5820 | 0.4354    | 0.8776 |

`is_document` and `is_digital` are near-perfect — structural properties visible in pixel statistics. `is_paper` is strong at 0.97 F1. Crumple and shadow are rare classes with extreme imbalance: the model catches most cases (recall 0.87 / 0.88) but produces false positives (precision 0.30 / 0.44). This is the right trade-off for a routing pre-filter — you would rather over-route to OCR than lose a crumpled receipt.

### Rotation (per-degree: deterministic axis + GBDT flip/trust)

Rotation is no longer four one-hot labels. It is a per-degree estimate in [0, 360):

1. **Axis (deterministic radon)** — a 60-bin projection response curve at 3° resolution, parabolic-refined. On clean born-digital invoices the axis is sub-degree: benchmark circular MAE 0.335° (3 fatura2 docs × 18 angles).
2. **Fold-snap** — the axis is mod-180 blind: for exactly-horizontal text the argmax lands randomly on bin 0 (~0.01°) or bin 59 (~179.99°). `combine_rotation` snaps `<1° or >179°` to 0 before combining. (Before the fix, this one bug flipped 55% of upright invoices.)
3. **Flip head (GBDT)** — ascender/descender textline evidence predicts 0 vs 180.
4. **Trust head (GBDT)** — predicts P(axis within 1°); `rotation_confidence = trust × max(flip, 1−flip)`; `needs_review` when confidence < tau (0.9) or not a document.

**Rotation by document type:**

| Document type | Rotation result |
|---|---|
| Born-digital invoices (gold 0°) | **Solved**: 100% within 1°, 0/1,000 180° flips, axis MAE 0.335° |
| Photographed / scanned documents | flip head held-out ~76% → routed to `needs_review` |

**Measured on the real WASM, gold upright FATURA2 (1,000 born-digital invoices, gold = 0°):**

| Metric | Value |
|---|---|
| is_document | 1000 / 1000 |
| axis (mod 180) within 1° | 100% |
| rotation_deg within 1° | 100% |
| 180°-flip errors | 0% (0 / 1000) |
| needs_review | 0% |
| mean confidence | 1.000 |

The 0 flips come from a deterministic guard in `combine_rotation`, not from a better flip head. Before the fold-snap, 55% of upright invoices were flipped; after the snap the mod-180 axis was correct on every image but the GBDT flip head still guessed 180° on 6.5% of them (confidently wrong — mean confidence 0.976, only 3.6% flagged for review). The v4 guard closes that gap for born-digital documents: at a cardinal axis with ≥5 text lines, a digital document is only marked 180° when *all three* flip signals agree — ink body fraction above the x-height zone, a minority of text lines voting upright, and a strongly negative flip score; otherwise it stays 0°.

Held-out flip head: ~76% accuracy, flip AUC 0.87, trust AUC 0.99. The head itself is unchanged; the guard overrides it on the born-digital path where the flip evidence is weakest. `rotation_deg` is now production-grade (0/1000 flip errors) for clean born-digital documents at cardinal axes. Photographed and scanned documents still route through the raw flip head and must be treated as needing review — PLAN.md Path B (quantized vision encoder) is the escalation for those.

### What changed in v0.3.0

- **Born-digital 180° flips eliminated.** The deterministic flip guard in `combine_rotation` takes 180°-flip errors on the 1,000-image gold set from **65 (v0.2.0) → 0 (v0.3.0)** — without retraining: the same v4 manifest, the same 116 features, a stricter evidence gate on the digital path.
- **Cleaner WASM surface.** The `debug_features_v2` export (a development aid, not a public API) is removed from the shipped build.
- **Version bump.** Workspace crates, the frontend package, and the Python wheel are now 0.3.0.

### What changed vs the old 9-label model

The four one-hot rotation labels (`rotation_0/90/180/270`, ~0.74 F1) are gone, replaced by the deterministic radon axis + `flip`/`trust` heads — sub-degree accuracy on clean digital documents instead of a 4-way guess. The feature vector grew from 103 to 116 (adds radon response summary, textline ascender/descender, boundary minAreaRect, docstrum axis).

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

- **180° flips on photographed/scanned documents** — the flip head (ascender/descender heuristics) is the weak point there. On upright born-digital invoices the v4 guard in `combine_rotation` eliminated all 1,000/1,000 flip errors, but that guard only applies to digital documents at a cardinal axis with enough text lines. Photographed and scanned documents still route through the raw flip head (held-out ~76% accuracy), and `needs_review` does not catch all of them because the head is confidently wrong. Path B (vision encoder) is the planned fix.
- **Photographed crumpled paper** — illumination gradients look like shadows to the Sobel-based features. The edge-density and Otsu-threshold features can't distinguish a fold from vignetting. Solution: more training data, not more features.
- **Heavy perspective distortion** — the model sees rotation as a 2D image-plane property. If the document is photographed at a 45° angle, the rotation label may be wrong even though the "true" text direction is correct. Perspective correction should happen upstream.
- **Crumple precision** — 30% precision means most images flagged as crumpled are actually flat. This is acceptable for a pre-filter that prompts re-scanning but not for automated rejection.
- **Hard negatives** — COCO images with grid-like or text-like structure (bookshelves, building facades, signs) confuse the edge-direction and projection features. The model sees rows of "text" that are actually bricks.

## Feature Taxonomy

Every feature is computed from raw pixel data — no deep learning, no learned embeddings, no external dependencies beyond `libm` for `no_std` math. This is deliberate: the same Rust code runs identically in the browser (WASM), in Python via PyO3, and in standalone binaries. The canonical vector is `extract_all_v2`: 116 features = 100 core heuristics + 16 rotation signals.

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

### Rotation Signals (16 features) — the per-degree rework

Radon response curve summary (best axis, peakiness, peak margin, concentration — the deterministic axis lives at feature index 100), textline ascender/descender mass + line-vote fraction (the flip head's evidence), boundary minimum-area-rectangle axis vote (visible-boundary complement to the radon axis), and an unfolded Docstrum axis. The taxonomy above is a representative selection of the 116, not an exhaustive list.

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
