#!/usr/bin/env node
// eval_crf.mjs — CRF + context.rs ensemble evaluation on FATURA2
// Usage: node scripts/eval_crf.mjs [--sample N]

import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SAMPLE = parseInt(process.argv[process.argv.indexOf('--sample') + 1] || '0', 10) || 0;

// ── Load ground truth ───────────────────────────────
const gt = JSON.parse(readFileSync(resolve(ROOT, 'data/fatura2_pii_gt.json'), 'utf-8'));
let samples = gt.samples;
if (SAMPLE > 0) samples = samples.slice(0, SAMPLE);
console.log(`Evaluating ${samples.length} of ${gt.totalInvoices} FATURA2 invoices`);

// ── Load WASM + init ────────────────────────────────
const wasmBytes = readFileSync(resolve(ROOT, 'frontend/wasm_bridge_bg.wasm'));
const { initSync, scan_pii, load_crf_model } = await import(
  resolve(ROOT, 'frontend/wasm_bridge.js')
);
initSync(new WebAssembly.Module(wasmBytes));

// ── Load CRF model ──────────────────────────────────
const crfJson = readFileSync(resolve(ROOT, 'data/crf_model.json'), 'utf-8');
load_crf_model(crfJson);
console.log('CRF model loaded');

// ── Eval helper regex ───────────────────────────────
const EMAIL_RE = /^[\w.-]+@[\w.-]+\.\w+$/;
const PHONE_RE = /^\+\(\d{3}\)\d{3}-\d{4}$/;
const ZIP_RE = /^\d{5}(-\d{4})?$/;
const ACCT_RE = /^\d{6,12}$/;

// Ground truth types per region (used for per-type scoring)
const ALL_PII = ['PHONE','EMAIL','ADDRESS','NAME','ZIP','ACCOUNT'];

// ── Per-type confusion ──────────────────────────────
const confusion = {};
for (const k of ALL_PII) confusion[k] = { tp: 0, fp: 0, fn: 0 };

let totalHits = 0;
let crfHits = 0;  // hits that came from CRF (i.e., ADDRESS, NAME, ACCOUNT)
const start = performance.now();

for (let si = 0; si < samples.length; si++) {
  const s = samples[si];
  const tokens = s.tokens;
  const tags = s.tags;
  const fullText = tokens.join(' ');

  // Build char offsets
  const offsets = [];
  let pos = 0;
  for (const t of tokens) {
    offsets.push({ start: pos, end: pos + t.length });
    pos += t.length + 1;
  }

  // ── Run ensemble scan (context.rs + CRF) ────────────
  let hits;
  try { hits = scan_pii(fullText); } catch { hits = []; }
  if (!Array.isArray(hits)) hits = [];
  totalHits += hits.length;

  // Count CRF-sourced hits (ADDRESS, NAME, ACCOUNT)
  crfHits += hits.filter(h => ['ADDRESS','NAME','ACCOUNT'].includes(h.kind)).length;

  // ── Extract GT regions by tag ──────────────────────
  const regions = [];
  let rs = -1, rt = 0;
  for (let i = 0; i < tags.length; i++) {
    if (tags[i] > 0) {
      if (rs === -1) { rs = i; rt = tags[i]; }
      else if (tags[i] !== rt) {
        regions.push({ start: rs, end: i - 1, tag: rt });
        rs = i; rt = tags[i];
      }
    } else if (rs !== -1) {
      regions.push({ start: rs, end: i - 1, tag: rt });
      rs = -1;
    }
  }
  if (rs !== -1) regions.push({ start: rs, end: tags.length - 1, tag: rt });

  // ── Match GT regions vs PII hits ──────────────────
  for (const r of regions) {
    const rTokens = tokens.slice(r.start, r.end + 1);
    const rStartChar = offsets[r.start].start;
    const rEndChar = offsets[r.end].end;
    const rHits = hits.filter(h => h.start < rEndChar && h.end > rStartChar);

    if (r.tag === 5) {
      // Tag 5: seller address block → PHONE, EMAIL, ADDRESS, NAME, ZIP
      const gtTypes = new Set(['ADDRESS','NAME']);
      for (const t of rTokens) {
        if (EMAIL_RE.test(t)) gtTypes.add('EMAIL');
        else if (PHONE_RE.test(t)) gtTypes.add('PHONE');
        else if (ZIP_RE.test(t)) gtTypes.add('ZIP');
      }
      for (const kind of gtTypes) {
        confusion[kind][rHits.some(h => h.kind === kind) ? 'tp' : 'fn']++;
      }
      for (const h of rHits) {
        if (!gtTypes.has(h.kind)) {
          confusion[h.kind] = confusion[h.kind] || { tp:0, fp:0, fn:0 };
          confusion[h.kind].fp++;
        }
      }
    } else if (r.tag === 13) {
      // Tag 13: payment details → ACCOUNT
      const gtTypes = new Set();
      for (const t of rTokens) {
        if (ACCT_RE.test(t)) gtTypes.add('ACCOUNT');
      }
      for (const kind of gtTypes) {
        confusion[kind][rHits.some(h => h.kind === kind) ? 'tp' : 'fn']++;
      }
      for (const h of rHits) {
        if (!gtTypes.has(h.kind) && h.kind !== 'ADDRESS' && h.kind !== 'NAME') {
          confusion[h.kind] = confusion[h.kind] || { tp:0, fp:0, fn:0 };
          confusion[h.kind].fp++;
        }
      }
    }
  }

  // Hits outside any region → FP
  for (const h of hits) {
    const inAny = regions.some(r => {
      const rcStart = offsets[r.start].start;
      const rcEnd = offsets[r.end].end;
      return h.start < rcEnd && h.end > rcStart;
    });
    if (!inAny) {
      confusion[h.kind] = confusion[h.kind] || { tp:0, fp:0, fn:0 };
      confusion[h.kind].fp++;
    }
  }

  if ((si + 1) % 1000 === 0) {
    const el = ((performance.now() - start) / 1000).toFixed(1);
    console.log(`  ${si+1}/${samples.length} (${el}s, ${totalHits} hits, ${crfHits} crf)`);
  }
}

const elapsed = ((performance.now() - start) / 1000).toFixed(1);
console.log(`Done. ${totalHits} hits (${crfHits} from CRF) in ${elapsed}s\n`);

// ── Compute metrics ─────────────────────────────────
const fScore = (p, r, beta) => {
  const b2 = beta * beta;
  return (p + r > 0) ? (1 + b2) * p * r / (b2 * p + r) : 0;
};

console.log('=== CRF + Context Ensemble Evaluation ===');
console.log(`Invoices: ${samples.length}\n`);

let allTp = 0, allFp = 0, allFn = 0;
for (const kind of ALL_PII) {
  const c = confusion[kind] || { tp: 0, fp: 0, fn: 0 };
  const { tp, fp, fn } = c;
  allTp += tp; allFp += fp; allFn += fn;
  const p = tp + fp > 0 ? tp / (tp + fp) : 0;
  const r = tp + fn > 0 ? tp / (tp + fn) : 0;
  console.log(
    `  ${kind.padEnd(10)} P=${p.toFixed(4)}  R=${r.toFixed(4)}  ` +
    `F1=${fScore(p,r,1).toFixed(4)}  F5=${fScore(p,r,5).toFixed(4)}  ` +
    `(TP=${tp} FP=${fp} FN=${fn})`
  );
}

const mp = allTp + allFp > 0 ? allTp / (allTp + allFp) : 0;
const mr = allTp + allFn > 0 ? allTp / (allTp + allFn) : 0;
console.log(
  `  ${'MACRO'.padEnd(10)} P=${mp.toFixed(4)}  R=${mr.toFixed(4)}  ` +
  `F1=${fScore(mp,mr,1).toFixed(4)}  F5=${fScore(mp,mr,5).toFixed(4)}`
);

// Save
const report = {
  invoices: samples.length,
  totalHits,
  crfHits,
  elapsed,
  macro: { precision: +mp.toFixed(4), recall: +mr.toFixed(4), f1: +fScore(mp,mr,1).toFixed(4), f5: +fScore(mp,mr,5).toFixed(4) },
  perType: {},
};
for (const kind of ALL_PII) {
  const c = confusion[kind] || { tp: 0, fp: 0, fn: 0 };
  const p = c.tp + c.fp > 0 ? c.tp / (c.tp + c.fp) : 0;
  const r = c.tp + c.fn > 0 ? c.tp / (c.tp + c.fn) : 0;
  report.perType[kind] = {
    precision: +p.toFixed(4), recall: +r.toFixed(4),
    f1: +fScore(p,r,1).toFixed(4), f5: +fScore(p,r,5).toFixed(4),
    tp: c.tp, fp: c.fp, fn: c.fn,
  };
}
writeFileSync(resolve(ROOT, 'data/crf_report.json'), JSON.stringify(report, null, 2));
console.log(`\nReport: data/crf_report.json`);
