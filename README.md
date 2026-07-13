# wasm-doc-boost

Client-side document classification via Rust → WASM + GBDT inference. Five binary labels: `is_document`, `is_digital`, `is_paper`, `is_crumpled`, `is_shadow`. In-browser privacy, microsecond inference.

![WASM size](https://img.shields.io/badge/wasm-538%20KB%20%7C%20216%20KB%20gzip-blue)

## API

```js
import init, { load_model, classify_file } from 'wasm-bridge';

await init();

// Load a trained XGBoost model (dump_model JSON format)
load_model(JSON.stringify(modelJson));

// Classify an image from a Uint8Array (PNG or JPEG)
const fileBytes = await fs.readFile('scan.png');
const result = classify_file(fileBytes);
// { is_document: 0.97, is_digital: 0.92, is_paper: 0.03, is_crumpled: 0.01, is_shadow: 0.12 }
```

- `load_model(json)` — parses XGBoost `dump_model()` JSON; 5-label multi-output GBDT
- `classify_file(bytes)` — decodes PNG/JPEG, resizes to max 512 px long edge, extracts 78 features, runs inference, returns an object with five probability scores

## Build

```bash
wasm-pack build --target web crates/wasm-bridge     # browser
wasm-pack build --target nodejs crates/wasm-bridge  # Node.js
```

## Test

```bash
cargo test -p features-core                         # Rust unit tests (32)
wasm-pack test --node crates/wasm-bridge            # WASM tests (5)
node --test crates/wasm-bridge/tests/node_integration.mjs  # Node integration (3)
```
