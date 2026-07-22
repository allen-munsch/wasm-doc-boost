# DocBoost — Unified Document Analysis Layer

Zero-build vanilla JS abstraction over three engines: WASM GBDT classification, Tesseract.js OCR, and GLM-OCR backend. Each engine output is Zod-validated, normalized into a common `Region` shape, deduplicated by IoU, and enriched with provenance.

## Quick Start

```html
<!-- In your HTML -->
<script type="importmap">
{ "imports": { "zod": "https://esm.sh/zod@3.24.2" } }
</script>
<script src="vendor/tesseract/tesseract.min.js"></script>
<script type="module">
import { createDocBoost } from './src/docboost.js';
import { regions, redactions, redactRegion, classification } from './src/enrich.js';

const db = await createDocBoost({
  modelPath: 'model.json',
  tessWorkerPath: 'vendor/tesseract/worker.min.js',
  tessCorePath: 'vendor/tesseract',
  glmBackend: 'http://localhost:8765',  // omit to skip GLM-OCR
});

// Per-document analysis
const doc = await db.analyze(imageFile);

// Render scores
const cls = classification(doc);
for (const [label, score] of Object.entries(cls)) {
  console.log(`${label}: ${Math.round(score * 100)}%`);
}

// Render text regions with bounding boxes
for (const r of regions(doc)) {
  drawBox(r.bbox, r.source, r.confidence);
  console.log(`[${r.source}] ${r.text} (${r.confidence}%)`);
}
</script>
```

## EnrichedDocument API

`createDocBoost(...).analyze(file)` returns a plain object (the "doc"). All access goes through lens functions; all mutations return a new doc.

### Lenses (read-only)

```js
import { regions, redactions, classification, provenance, fields, image, docId, region } from './src/enrich.js';

regions(doc)         // Region[] — every text region from all engines, merged
redactions(doc)      // Region[] — only redacted regions
classification(doc)  // { is_document, is_digital, is_paper, is_crumpled, is_shadow }
provenance(doc)      // { engine, tesseract, glmOcr, analysisMs, mergeStats }
fields(doc)          // FieldAnnotation[] — user-added metadata
image(doc)           // { width, height, mime }
region(doc, id)      // Region | null — O(1) lookup by ID
```

### Curried Mutations

Each mutation is data-last (curried), takes a doc, returns a new doc. The original is never modified.

```js
import { addFieldAnnotation, removeFieldAnnotation, redactRegion, unredactRegion } from './src/enrich.js';

// Annotate a region with a field type
doc = addFieldAnnotation('tess-b0-p0-l0-w0', 'invoice_number', 0.9)(doc);

// Remove the annotation
doc = removeFieldAnnotation('ann-id-123')(doc);

// Redact one or more regions
doc = redactRegion('tess-b0-p0-l0-w1')(doc);

// Un-redact
doc = unredactRegion('tess-b0-p0-l0-w1')(doc);
```

Chaining with a `pipe` helper:

```js
const pipe = (val, ...fns) => fns.reduce((v, f) => f(v), val);

doc = pipe(doc,
  addFieldAnnotation('tess-b0-p0-l0-w0', 'vendor_name'),
  redactRegion('tess-b0-p0-l2-w3'),
  redactRegion('glm-5'),
);
```

### Serialization

```js
import { toJSON, fromJSON } from './src/enrich.js';

// Save to localStorage or POST to a server
const plain = toJSON(doc);
localStorage.setItem('last-doc', JSON.stringify(plain));

// Restore
const rebuilt = fromJSON(JSON.parse(localStorage.getItem('last-doc')));
```

## The Region Shape

Every engine output normalizes to this flat structure:

```js
{
  id: "tess-b0-p0-l1-w3",          // stable ID from source hierarchy
  text: "Invoice",
  bbox: { x0: 12, y0: 12, x1: 70, y1: 28 },
  confidence: 95,                   // 0-100
  source: "merged",                 // "tesseract" | "glm-ocr" | "merged"
  sourceDetail: {
    blockType: "PT_FLOWING_TEXT",   // Tesseract PolyBlockType (null for GLM)
    level: "word",
    parentId: "tess-b0-p0-l1",      // links to parent line
    regionIndex: null,              // GLM region index (null for Tesseract)
  },
  mergeDecision: {                  // null unless source === "merged"
    won: "tesseract",
    reason: "tesseract (95.0) over glm-ocr (75.0), IoU=87%",
    loserId: "glm-0",
    loserConfidence: 75,
    iou: 0.87,
  },
}
```

## Provenance

```js
const prov = provenance(doc);
// {
//   engine: "wasm-doc-boost 0.1.0",
//   tesseract: { version: "5.4.0", oem: "LSTM_ONLY", psm: "AUTO" },
//   glmOcr: { backend: "GLM-OCR (cuda)" },
//   analysisMs: 1523,
//   mergeStats: { tesseractRegions: 42, glmRegions: 15, mergedPairs: 8, finalRegions: 49 },
// }
```

`mergeStats.mergedPairs` counts how many overlapping Tesseract/GLM region pairs were resolved. `finalRegions - (tesseractRegions + glmRegions - mergedPairs)` is the number of novel GLM regions (no Tesseract overlap).

## Progress Callback

```js
const db = await createDocBoost(config, (stage, status, message) => {
  console.log(`${stage}: ${status} — ${message}`);
  // stage: "wasm" | "tesseract" | "glm" | "classify" | "ocr" | "pii"
  // status: "running" | "ready" | "ok" | "err"
});
```

## Standalone Modules

You don't need `createDocBoost` to use the normalization and merge:

```js
import { normalizeTesseract } from './src/normalize.js';
import { normalizeGlmOcr } from './src/normalize.js';
import { mergeRegions } from './src/merge.js';
import { buildEnrichedDocument } from './src/enrich.js';

const tessRegions = normalizeTesseract(tesseractResult);  // RecognizeResult → Region[]
const glmRegions  = normalizeGlmOcr(glmResult);           // GLM JSON → Region[]
const { regions: merged, stats } = mergeRegions(tessRegions, glmRegions);

const doc = buildEnrichedDocument({
  image: { width: 800, height: 600, mime: 'image/png' },
  classification: { is_document: 0.95, is_digital: 0.1, is_paper: 0.8, is_crumpled: 0.05, is_shadow: 0.02 },
  regions: merged,
  pii: [{ kind: 'PAN', text: '4111111111111111', start: 0, end: 16 }],
  tesseract: tesseractResult,
  glmOcr: glmResult,
  analysisMs: 1200,
  mergeStats: stats,
});
```

## Extracting PII from Regions

```js
import { regions } from './src/enrich.js';

// PII hits are document-level (from combined text), not per-region.
// To find which regions contain a specific PII hit:
const hitsInRegion = (doc, regionId) => {
  const r = region(doc, regionId);
  if (!r) return [];
  return doc.pii.filter(h => r.text.includes(h.text));
};

for (const r of regions(doc)) {
  const hits = hitsInRegion(doc, r.id);
  if (hits.length > 0) {
    console.log(`Region ${r.id} contains: ${hits.map(h => h.kind).join(', ')}`);
  }
}
```

## Image Input Formats

`analyze()` accepts `File`, `Blob`, or `HTMLImageElement`:

```js
// From <input type="file">
const doc = await db.analyze(fileInput.files[0]);

// From drag-and-drop
dropZone.ondrop = async (e) => {
  const doc = await db.analyze(e.dataTransfer.files[0]);
};

// From an existing <img> element
const doc = await db.analyze(document.querySelector('#preview-image'));
```

## Error Handling

If an engine fails during init, it's non-fatal — the others still work. `doc.classification`, `provenance(doc).tesseract`, and `provenance(doc).glmOcr` will be `null` for engines that failed. If classification fails during `analyze()`, `classification(doc)` is `null` but OCR results are still present.
