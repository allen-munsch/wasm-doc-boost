#!/usr/bin/env node
// eval_cordv2.mjs — CRF PII scanner evaluation on CORD-v2 OCR output
// Usage: node scripts/eval_cordv2.mjs [--sample N]

import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SAMPLE = parseInt(process.argv[process.argv.indexOf('--sample') + 1] || '0', 10) || 0;

// ── Load OCR output ──────────────────────────────────
const ocr = JSON.parse(readFileSync(resolve(ROOT, 'data/cordv2_ocr.json'), 'utf-8'));
let entries = ocr;
if (SAMPLE > 0) entries = entries.slice(0, SAMPLE);
console.log(`Evaluating ${entries.length} CORD-v2 receipts`);

// ── Load WASM + init ────────────────────────────────
const wasmBytes = readFileSync(resolve(ROOT, 'frontend/wasm_bridge_bg.wasm'));
const { initSync, scan_pii, load_crf_model } = await import(
  resolve(ROOT, 'frontend/wasm_bridge.js')
);
initSync(wasmBytes);

// ── Load CRF model ──────────────────────────────────
const crfJson = readFileSync(resolve(ROOT, 'data/crf_model.json'), 'utf-8');
load_crf_model(crfJson);
console.log('CRF model loaded');

// ── Per-type hit counters ───────────────────────────
const ALL_PII = ['PHONE','EMAIL','ADDRESS','NAME','ZIP','ACCOUNT'];
const hitCounts = {};
for (const k of ALL_PII) hitCounts[k] = 0;
let receiptsWithHits = 0;
let totalHits = 0;

const start = performance.now();

for (let si = 0; si < entries.length; si++) {
  const entry = entries[si];
  const text = entry.ocrText || '';

  let hits;
  try { hits = scan_pii(text); } catch { hits = []; }

  totalHits += hits.length;
  if (hits.length > 0) receiptsWithHits++;

  for (const h of hits) {
    if (ALL_PII.includes(h.kind)) hitCounts[h.kind]++;
  }
}

const elapsed = ((performance.now() - start) / 1000).toFixed(1);

// ── Report ──────────────────────────────────────────
const report = {
  dataset: 'CORD-v2',
  totalReceipts: entries.length,
  receiptsWithHits,
  totalHits,
  elapsed: `${elapsed}s`,
  perType: {},
};

console.log(`\n=== CORD-v2 CRF Evaluation ===`);
console.log(`Receipts: ${entries.length}`);
console.log(`With PII hits: ${receiptsWithHits} (${(receiptsWithHits / entries.length * 100).toFixed(1)}%)`);
console.log(`Total hits: ${totalHits}`);
console.log(`Elapsed: ${elapsed}s\n`);

for (const k of ALL_PII) {
  const c = hitCounts[k] || 0;
  console.log(`  ${k}: ${c} hits (${(c / entries.length).toFixed(2)} per receipt)`);
  report.perType[k] = { hits: c, perReceipt: +(c / entries.length).toFixed(4) };
}

// ── Sample output for manual review ──────────────────
console.log(`\n=== Sample OCR text with PII hits ===`);
let shown = 0;
for (let si = 0; si < Math.min(entries.length, 100); si++) {
  if (shown >= 5) break;
  const entry = entries[si];
  let hits;
  try { hits = scan_pii(entry.ocrText || ''); } catch { hits = []; }
  if (hits.length > 0) {
    shown++;
    console.log(`\n--- ${entry.filename} ---`);
    console.log(`OCR text: ${entry.ocrText.substring(0, 300)}`);
    console.log(`Hits: ${hits.map(h => `${h.kind}="${h.text}"`).join(', ')}`);
  }
}

const outPath = resolve(ROOT, 'data', 'cordv2_report.json');
writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`\nReport: ${outPath}`);
