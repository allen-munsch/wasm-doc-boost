#!/usr/bin/env node
// eval_pii.mjs — PII scanner evaluation against FATURA2 ground truth
// Usage: node scripts/eval_pii.mjs [--sample N]

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
console.log(`Loaded ${gt.totalInvoices} invoices, evaluating ${samples.length}`);
console.log(`GT: ${gt.tag5Regions} tag-5 regions (addr), ${gt.tag13Regions} tag-13 regions (payment)`);
console.log(`GT type counts: ${JSON.stringify(gt.typeCounts)}`);
console.log(`Reference: ${gt.panPatternCount} PAN-like patterns (no GT)`);

// ── Load WASM ───────────────────────────────────────
const wasmBytes = readFileSync(resolve(ROOT, 'frontend/wasm_bridge_bg.wasm'));
const { initSync, scan_pii } = await import(resolve(ROOT, 'frontend/wasm_bridge.js'));
initSync(new WebAssembly.Module(wasmBytes));

// ── Regex for GT classification ─────────────────────
const EMAIL_RE = /^[\w.-]+@[\w.-]+\.\w+$/;
const PHONE_RE = /^\+\(\d{3}\)\d{3}-\d{4}$/;
const ZIP_RE = /^\d{5}(-\d{4})?$/;
const ACCT_RE = /^\d{6,12}$/;

// ── Per-type confusion ──────────────────────────────
const PII_KINDS = ['PHONE','EMAIL','ADDRESS','NAME','ZIP','ACCOUNT',
                    'PAN','SSN','EIN','CVV','EXPIRY','ROUTING','DOB'];
const confusion = {};
for (const k of PII_KINDS) confusion[k] = { tp: 0, fp: 0, fn: 0 };

let totalHits = 0;
let invoicesWithHits = 0;
let start = performance.now();

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

  // Run scanner
  let hits;
  try { hits = scan_pii(fullText); } catch { hits = []; }
  if (!Array.isArray(hits)) hits = [];
  if (hits.length > 0) invoicesWithHits++;
  totalHits += hits.length;

  // ── Extract contiguous regions by tag ──────────────
  const regions = []; // {startIdx, endIdx, tag}
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
      // FP: hits in this region of wrong type
      for (const h of rHits) {
        if (!gtTypes.has(h.kind)) {
          confusion[h.kind] = confusion[h.kind] || { tp:0, fp:0, fn:0 };
          confusion[h.kind].fp++;
        }
      }
    } else if (r.tag === 13) {
      // Tag 13: payment details → ACCOUNT, EMAIL (some invoices put email here)
      const gtTypes = new Set();
      for (const t of rTokens) {
        if (ACCT_RE.test(t)) gtTypes.add('ACCOUNT');
        else if (EMAIL_RE.test(t)) gtTypes.add('EMAIL');
        else if (PHONE_RE.test(t)) gtTypes.add('PHONE');
        else if (ZIP_RE.test(t)) gtTypes.add('ZIP');
      }
      for (const kind of gtTypes) {
        confusion[kind][rHits.some(h => h.kind === kind) ? 'tp' : 'fn']++;
      }
      // FP: hits of wrong type in this region
      for (const h of rHits) {
        if (!gtTypes.has(h.kind) && h.kind !== 'ADDRESS' && h.kind !== 'NAME') {
          confusion[h.kind] = confusion[h.kind] || { tp:0, fp:0, fn:0 };
          confusion[h.kind].fp++;
        }
      }
    }
  }

  // Hits entirely outside any region → FP
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
    console.log(`  ${si+1}/${samples.length} (${el}s, ${totalHits} hits)`);
  }
}

const elapsed = ((performance.now() - start) / 1000).toFixed(1);
console.log(`Done. ${totalHits} hits in ${invoicesWithHits} invoices, ${elapsed}s\n`);

// ── Compute metrics ─────────────────────────────────
const fScore = (p, r, beta) => {
  const b2 = beta * beta;
  return (p + r > 0) ? (1 + b2) * p * r / (b2 * p + r) : 0;
};

const evaluatedKinds = ['PHONE','EMAIL','ADDRESS','NAME','ZIP','ACCOUNT'];
const observedKinds = ['PAN','SSN','EIN','CVV','EXPIRY','ROUTING','DOB'];

console.log('=== PII Scanner Evaluation (FATURA2) ===');
console.log(`Invoices: ${samples.length}\n`);

const report = { invoices: samples.length, perType: {} };
let allTp = 0, allFp = 0, allFn = 0;

console.log('--- With Ground Truth ---');
for (const kind of evaluatedKinds) {
  const c = confusion[kind] || { tp:0, fp:0, fn:0 };
  const { tp, fp, fn } = c;
  allTp += tp; allFp += fp; allFn += fn;

  const p = tp + fp > 0 ? tp / (tp + fp) : 0;
  const r = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = fScore(p, r, 1);
  const f5 = fScore(p, r, 5);

  report.perType[kind] = { precision:+p.toFixed(4), recall:+r.toFixed(4), f1:+f1.toFixed(4), f5:+f5.toFixed(4), tp, fp, fn };
  console.log(`  ${kind.padEnd(10)} P=${p.toFixed(4)}  R=${r.toFixed(4)}  F1=${f1.toFixed(4)}  F5=${f5.toFixed(4)}  (TP=${tp} FP=${fp} FN=${fn})`);
}

// Macro avg (only evaluated kinds)
const mp = allTp + allFp > 0 ? allTp / (allTp + allFp) : 0;
const mr = allTp + allFn > 0 ? allTp / (allTp + allFn) : 0;
report.macro = { precision:+mp.toFixed(4), recall:+mr.toFixed(4), f1:+fScore(mp,mr,1).toFixed(4), f5:+fScore(mp,mr,5).toFixed(4) };
console.log(`  ${'macro_avg'.padEnd(10)} P=${mp.toFixed(4)}  R=${mr.toFixed(4)}  F1=${fScore(mp,mr,1).toFixed(4)}  F5=${fScore(mp,mr,5).toFixed(4)}`);

// Observed but no GT
console.log('\n--- No Ground Truth (FP counts only) ---');
for (const kind of observedKinds) {
  const c = confusion[kind] || { tp:0, fp:0, fn:0 };
  if (c.fp > 0) {
    console.log(`  ${kind.padEnd(10)} FP=${c.fp} (no GT available — FATURA2 has no ${kind}s)`);
  }
}

writeFileSync(resolve(ROOT, 'data/pii_report.json'), JSON.stringify(report, null, 2));
console.log(`\nReport: data/pii_report.json`);
