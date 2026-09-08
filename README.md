# wasm-doc-boost

Client-side document classification via Rust → WASM + GBDT inference. Nine binary labels: `is_document`, `is_digital`, `is_paper`, `is_crumpled`, `is_shadow`, `rotation_0`, `rotation_90`, `rotation_180`, `rotation_270`. 103 handcrafted pixel-heuristic features, 1,160 XGBoost trees, 645 KB WASM (254 KB gzip). In-browser privacy, sub-millisecond inference.

![WASM size](https://img.shields.io/badge/wasm-645%20KB%20%7C%20254%20KB%20gzip-blue)
![Features](https://img.shields.io/badge/features-103-blue)
![Trees](https://img.shields.io/badge/trees-1%2C160-blue)

## Why This Exists

Document classification usually requires sending images to a server. That means latency, bandwidth, and a privacy trade-off: someone else's hardware sees every document you classify.

wasm-doc-boost runs entirely in the browser. A 645 KB WASM binary extracts 103 structural features from pixel data and scores them through a gradient-boosted decision tree ensemble — no GPU, no network, no inference server. The model is small enough to load from a CDN in under 200 ms on a typical connection, and inference takes under a millisecond per image.

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
import init, { load_model, classify_file } from './wasm_bridge.js';

await init();

// Load the model (XGBoost dump_model JSON format, 9-label)
const resp = await fetch('model.json');
load_model(await resp.text());

// Classify an image from a Uint8Array (PNG or JPEG)
const fileBytes = await fs.readFile('scan.png');  // or file input, drop event, etc.
const result = classify_file(fileBytes);
// {
//   is_document: 0.97, is_digital: 0.92, is_paper: 0.03,
//   is_crumpled: 0.01, is_shadow: 0.12,
//   rotation_0: 0.94, rotation_90: 0.02, rotation_180: 0.01, rotation_270: 0.03
// }
```

### Python

```bash
# Prebuilt wheel (CPython 3.8+), no Rust toolchain required.
# Linux x86_64:
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# Linux aarch64:
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# macOS Intel (x86_64):
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-macosx_10_12_x86_64.whl
# macOS Apple Silicon (arm64):
pip install https://github.com/allen-munsch/wasm-doc-boost/releases/download/v0.1.0/py_features-0.1.0-cp38-abi3-macosx_11_0_arm64.whl

# Or build from source:
cd crates/py-features && maturin develop --release
```

```python
import py_features
import numpy as np
from PIL import Image

img = Image.open("receipt.jpg").convert("RGB")
pixels = np.array(img, dtype=np.uint8).tobytes()
features = py_features.extract_all(pixels, img.width, img.height)
# features is a list[float] of 103 values — feed to any classifier
```

### API

- `load_model(json)` — parses XGBoost `dump_model()` JSON; 9-label multi-output GBDT
- `classify_file(bytes)` — decodes PNG/JPEG, resizes to max 512 px long edge, extracts 103 features, runs inference, returns nine probability scores
- `scan_pii(text)` — rule-based + CRF PII scanner (credit cards, SSNs, emails, phones, addresses, names)

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

**Tier 0 — WASM GBDT classifier** (~0.5 ms, 645 KB download, costs nothing after load). Runs in the browser's main thread. Determines *what kind of image this is* before any OCR engine spins up. The 9 labels answer every routing question: Is this a document at all? Digital or paper? Rotated? Crumpled/shadowed (i.e., will Tesseract fail)? The rotation label also tells the OCR tier which way to orient the image.

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

## Model Performance (v4, 103 features, 2026-08-09)

Training: 74,944 samples (18,736 images × 4 rotations) across 6 sources — FATURA2 (digital receipts), CORD-v2 (receipts on desk), SRD (scanned receipts), COCO (natural images), RVL-CDIP (document pages), and synthetic variants. 80/20 GroupShuffleSplit grouped by base image — no data leak from rotation augmentation. XGBoost depth=5, lr=0.1, 1,160 trees total.

### Document Classification

| Label        | AUC    | F1     | Precision | Recall |
|-------------|--------|--------|-----------|--------|
| is_document | 1.0000 | 0.997  | 0.997     | 0.998  |
| is_digital  | 1.0000 | 1.000  | 1.000     | 1.000  |
| is_paper    | 0.9999 | 0.973  | 0.958     | 0.990  |
| is_crumpled | 0.9893 | 0.629  | 0.486     | 0.892  |
| is_shadow   | 0.9969 | 0.706  | 0.567     | 0.936  |

`is_document` and `is_digital` are near-perfect — these are structural properties visible in pixel statistics. `is_paper` is strong at 0.973 F1. Crumple and shadow are rare classes (2.4% positive each) with extreme imbalance: the model catches most cases (recall 0.89 / 0.94) but produces false positives (precision 0.49 / 0.57). This is the right trade-off for a document-scanning pipeline — you would rather re-scan a false positive than lose a crumpled receipt.

### Rotation Detection

| Label        | AUC    | F1     | Precision | Recall |
|-------------|--------|--------|-----------|--------|
| rotation_0  | 0.9476 | 0.734  | 0.637     | 0.866  |
| rotation_90 | 0.9484 | 0.750  | 0.678     | 0.840  |
| rotation_180| 0.9483 | 0.741  | 0.648     | 0.864  |
| rotation_270| 0.9484 | 0.741  | 0.652     | 0.858  |

All four rotations are balanced (25% each), so F1 is the metric. Rotation at ~0.74 F1 is ~3× better than random (25%) but not production-grade across all image types. The story changes when you break it down by source.

### Rotation Detection by Image Source

| Source     | Type              | Rotation AUC | Rotation F1 |
|-----------|-------------------|-------------|-------------|
| fatura2   | Digital receipts  | 1.000       | 0.997       |
| cord-v2   | Receipts on desk  | 0.917       | 0.717       |
| srd       | Scanned receipts  | 0.894       | 0.688       |
| variants  | Synthetic variants| 0.863       | 0.632       |
| coco      | Natural images    | 0.764       | 0.537       |
| negatives | Hard negatives    | 0.626       | 0.400       |

For clean digital receipts, rotation detection is solved (99.7% F1). For photographed and scanned documents, the model is ~2× to 2.7× better than random. The gap comes from illumination variance, perspective distortion, and weak text-line structure in non-digital images.

### What Changed from v3 to v4

v4 adds 9 ink-mask features: ink coverage, mean intensity, horizontal/vertical run-length statistics (mean, std, entropy), and a Docstrum-inspired nearest-neighbor rotation angle. The Sobel gradient passes were also refactored — all edge features now share a single Sobel computation, and the inner loop uses a pre-padded buffer with flat indexing rather than per-pixel border clamping. This is faster (1 Sobel pass instead of 7) and enables LLVM auto-vectorization when built with `RUSTFLAGS="-C target-feature=+simd128"`.

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

Combined with Sobel-pass sharing (one `SobelResult` instead of seven independent Sobel calls), the end-to-end feature extraction speedup over v3 is approximately 3–4× on WASM and 2–3× on native.

## Where It Fails (And Why)

- **Photographed crumpled paper** — illumination gradients look like shadows to the Sobel-based features. The edge-density and Otsu-threshold features can't distinguish a fold from vignetting. Solution: more training data, not more features.
- **Heavy perspective distortion** — the model sees rotation as a 2D image-plane property. If the document is photographed at a 45° angle, the rotation label may be wrong even though the "true" text direction is correct. Perspective correction should happen upstream.
- **Crumple precision** — 49% precision means half of the images flagged as crumpled are actually flat. This is acceptable for a pre-filter that prompts re-scanning but not for automated rejection.
- **Hard negatives** — COCO images with grid-like or text-like structure (bookshelves, building facades, signs) confuse the edge-direction and projection features. The model sees rows of "text" that are actually bricks.

## Feature Taxonomy

Every feature is computed from raw pixel data — no deep learning, no learned embeddings, no external dependencies beyond `libm` for `no_std` math. This is deliberate: the feature extractor compiles to a 28 KB WASM module (before the image decoder), and the same Rust code runs identically in the browser, in Python via PyO3, and in standalone binaries.

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

### Ink Mask & Run-Length (9 features) — new in v4

Ink coverage (fraction of pixels below Otsu threshold), mean ink intensity, horizontal and vertical run-length statistics (mean, std, entropy), and a Docstrum-inspired nearest-neighbor rotation angle. Run-length entropy is the key differentiator: text has a characteristic bimodal run-length distribution (character strokes ~3-5 px, word gaps ~10-20 px), while photos have nearly uniform run lengths. The Docstrum angle links each ink pixel to its nearest neighbor in the row below and computes the circular mean of those angles — horizontal text lines produce angles near ±π/2, rotated text shifts them.

## Build

```bash
wasm-pack build --target web crates/wasm-bridge     # browser
wasm-pack build --target nodejs crates/wasm-bridge  # Node.js

# With SIMD auto-vectorization (LLVM autovectorizes the Sobel inner loop)
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web crates/wasm-bridge
```

## Test

```bash
cargo test -p features-core                         # Rust unit tests (41)
cargo test -p wasm-bridge                           # WASM integration tests (4)
```

## License

MIT
