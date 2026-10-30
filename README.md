# wasm-doc-boost

Client-side document triage for OCR. A Rust→WASM gradient-boosted model looks at an image and says: **discard it, OCR it, or send it to a VLM — and how much to rotate it first.** No server, nothing leaves the page.

![Version](https://img.shields.io/badge/version-0.5.5-blue) ![Trees](https://img.shields.io/badge/trees-1%2C080-blue) ![Features](https://img.shields.io/badge/features-124-blue)

```
image
  → Tier 0  wasm-doc-boost (browser, ~25 ms)
       decide:  discard | Tesseract | VLM
       rotate:  quarter-turn + fine skew
  → Tier 1  Tesseract.js  (clean born-digital)
  → Tier 2  GLM-OCR / VLM (photographed paper)
```

It emits seven labels — `is_document`, `is_digital`, `is_paper`, `is_crumpled`, `is_shadow`, `flip`, `trust` — plus `rotation_deg`, `quarter`, `skew_deg`, and `needs_review`. 124 handcrafted pixel features, 1,080 trees, ~4 MB WASM.

## Quick start

### Browser

```bash
wasm-pack build --target web crates/wasm-bridge
python3 frontend/serve.py
```

```js
import init, { load_model, classify_file } from './wasm_bridge.js';
await init();
load_model(await (await fetch('model.json')).text());
const r = classify_file(await file.arrayBuffer());
// r.is_document, r.is_digital, r.is_paper, r.rotation_deg,
// r.skew_deg, r.needs_review, r.trait_labels, ...
```

### Python

```bash
cd crates/py-features && maturin develop --release
```

```python
import py_features
py_features.load_model(open('model.json').read())
r = py_features.predict(pixels_rgb, width, height)  # dict, same keys as WASM
```

`PY_FEATURES_BACKEND=cpu|cuda|auto` picks the radon backend; CPU and CUDA are bit-identical.

## Benchmarks (v0.5.5)

| what | result |
|---|---|
| rotation: 600 clean pages × 4 cardinals, within 1° | **100%** |
| accepted 180° flip errors | **0.0%** |
| small tilts (−3°…+3°) wrong half-turn | **0–0.5%** |
| review rate on clean pages | **0%** |
| `is_digital` / `is_paper` (held-out DocLayNet, 150 pages) | **1.00 / 1.00** |
| median `predict()` latency, 512 px | **25 ms** |

Doc-trait F1 (held-out): `is_document` 0.994 · `is_digital` 0.9996 · `is_paper` 0.985 · `is_crumpled` 0.58 · `is_shadow` 0.47.

## Honest limits

- **Rotation is trusted only on clean born-digital pages.** Photographed/curled paper routes to `needs_review` rather than risking a wrong deskew.
- **`is_crumpled` / `is_shadow` are weak, feature-limited.** They over-route to OCR (the safe failure), not reject.
- **Perspective distortion and hard negatives** (bookshelves, facades) remain the hardest cases.

## API

**WASM:** `load_model(json)` · `classify_file(bytes)` · `classifyPdf(bytes)` · `extractTextWithPositions(bytes)` · `load_crf_model(json)` · `scan_pii(text)`.

**Python:** `load_model` · `predict` · `extract_all` · `combine_rotation` · `classify_pdf` · `scan_pii` · `backend_info`.

## Build & test

```bash
cargo fmt --all && cargo build --workspace && cargo test --workspace
python3 scripts/goal_gate.py   # MET
```

## License

MIT
