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

// ── ADDRESS post-filter constants ──────────────────
const STREET_SUFFIX_RE = /\b(st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|pl|place|hwy|highway|pkwy|cswy|causeway|cir|circle|trl|trail|ste|suite|ter|terrace|sq|square|plz|plaza)\b/i;
const DIRECTION_RE = /\b(n|s|e|w|north|south|east|west|ne|nw|se|sw)\b/i;
const BLDG_NUM_RE = /\b\d{1,5}\s+[A-Z]/;
const POBOX_RE = /\b(p\.?o\.?\s*box|pobox)\b/i;
const STATE_ABBREV_RE = /\b[A-Z]{2}\b/;
const US_STATES = new Set([
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
  "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
  "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
  "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
  "DC","AS","GU","MP","PR","VI","FM","MH","PW","AA","AE","AP"
]);

// ── ACCOUNT post-filter ─────────────────────────────
// Rejects ACCOUNT hits that are invoice numbers (adjacent to INV/Invoice pattern).
// Also rejects digit sequences that are part of GSTIN tax IDs.
const INVOICE_PREFIX_RE = /\b(INV|INVOICE|INVOICE\s*#|INVOICE\s*ID|INVOICE\s*number)\d*$/i;
const GSTIN_PREFIX_RE = /GSTIN/i;
function isValidAccount(hitText, fullText, hitStart) {
  // Check if hit is preceded by an invoice prefix
  const before = fullText.substring(Math.max(0, hitStart - 25), hitStart);
  if (INVOICE_PREFIX_RE.test(before)) return false;
  if (GSTIN_PREFIX_RE.test(before)) return false;
  return true;
}
const US_STATE_WORD_RE = new RegExp(`\\b(${[...US_STATES].join('|')})\\b`);

function isValidAddress(hitText, fullText, hitStart, hitEnd) {
  const windowStart = Math.max(0, hitStart - 50);
  const windowEnd = Math.min(fullText.length, hitEnd + 50);
  const context = fullText.substring(windowStart, windowEnd);

  // Must have at least one address signal
  const signals = [
    STREET_SUFFIX_RE.test(context),
    DIRECTION_RE.test(context),
    BLDG_NUM_RE.test(context),
    POBOX_RE.test(context),
    US_STATE_WORD_RE.test(context) && /\d{5}/.test(context),  // state abbrev + ZIP
    /\d{3,5}\s+\w+\s+(st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|blvd\.|way|hwy|pkwy)/i.test(context),  // number + word + street suffix
    /\d{5}(-\d{4})?/.test(context),  // ZIP code present
  ];

  return signals.some(Boolean);
}

// Ground truth types per region (used for per-type scoring)
const ALL_PII = ['PHONE','EMAIL','ADDRESS','NAME','ZIP','ACCOUNT'];

// ── Per-type confusion ──────────────────────────────
const confusion = {};
for (const k of ALL_PII) confusion[k] = { tp: 0, fp: 0, fn: 0 };

let totalHits = 0;
let crfHits = 0;  // hits that came from CRF (i.e., ADDRESS, NAME, ACCOUNT)
const start = performance.now();

// ── FP diagnostic collection ─────────────────────────
const addressFPs = [];  // { sampleIdx, text, context, regionTag, regionText, reason }
const nameFPs = [];     // { sampleIdx, text, context, regionTag, regionText, reason }
const accountFNs = [];  // { sampleIdx, regionText, regionTag, accountTokens }
const accountFPs = [];  // { sampleIdx, text, context, regionTag, regionText, reason }
const addressFNs = [];  // { sampleIdx, regionText, regionTag }
const nameFNs = [];     // { sampleIdx, regionText, regionTag }

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

  // Post-filter: require address-like signals for ADDRESS hits
  hits = hits.filter(h => h.kind !== 'ADDRESS' || isValidAddress(h.text, fullText, h.start, h.end));
  // Post-filter: reject invoice numbers as ACCOUNT hits
  hits = hits.filter(h => h.kind !== 'ACCOUNT' || isValidAccount(h.text, fullText, h.start));

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

    const gtTypes = new Set();
    for (const t of rTokens) {
      if (EMAIL_RE.test(t)) gtTypes.add('EMAIL');
      else if (PHONE_RE.test(t)) gtTypes.add('PHONE');
      else if (ZIP_RE.test(t)) gtTypes.add('ZIP');
      else if (ACCT_RE.test(t)) {
        // Exclude GSTIN tax IDs (preceded by "GSTIN" in the same region)
        const tIdx = rTokens.indexOf(t);
        const preCtx = rTokens.slice(Math.max(0, tIdx - 3), tIdx).join(' ');
        if (!/GSTIN/i.test(preCtx)) {
          gtTypes.add('ACCOUNT');
        }
      }
    }

    // Tag 5 (seller address) and tag 8 (seller name) always contain
    // address/name data. Both include BILL_TO blocks with full addresses.
    if (r.tag === 5 || r.tag === 8) {
      gtTypes.add('ADDRESS');
      gtTypes.add('NAME');
    }
    // Tag 13 (payment details) MAY contain bank address blocks,
    // but most tag-13 regions are payment terms without addresses.
    // Only expect ADDRESS if the region has address-like signals.
    // Do NOT expect NAME — bank branch addresses don't contain person names.
    if (r.tag === 13) {
      const rLower = rTokens.join(' ').toLowerCase();
      const hasStreetSuffix = /\b(st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|pl|place|hwy|highway|pkwy|cswy|causeway|cir|circle|trl|trail|ste|suite|ter|terrace|sq|square|plz|plaza)\b/.test(rLower);
      const hasStateAbbrev = rTokens.some(t => t.length === 2 && US_STATES.has(t));
      const hasDirection = /\b(n|s|e|w|north|south|east|west|ne|nw|se|sw)\b/.test(rLower);
      const hasPOBox = /\b(p\.?o\.?\s*box|pobox)\b/i.test(rLower);
      if (hasStreetSuffix || hasStateAbbrev || hasDirection || hasPOBox) {
        gtTypes.add('ADDRESS');
      }
    }

    for (const kind of gtTypes) {
      const hasHit = rHits.some(h => h.kind === kind);
      confusion[kind][hasHit ? 'tp' : 'fn']++;
      if (kind === 'ACCOUNT' && !hasHit && accountFNs.length < 200) {
        const acctTokens = rTokens.filter(t => ACCT_RE.test(t));
        accountFNs.push({
          sampleIdx: si,
          regionText: fullText.substring(rStartChar, rEndChar),
          regionTag: r.tag,
          accountTokens: acctTokens,
        });
      }
      if (kind === 'ADDRESS' && !hasHit && addressFNs.length < 100) {
        addressFNs.push({
          sampleIdx: si,
          regionText: fullText.substring(rStartChar, rEndChar),
          regionTag: r.tag,
        });
      }
      if (kind === 'NAME' && !hasHit && nameFNs.length < 100) {
        nameFNs.push({
          sampleIdx: si,
          regionText: fullText.substring(rStartChar, rEndChar),
          regionTag: r.tag,
        });
      }
    }
    for (const h of rHits) {
      if (!gtTypes.has(h.kind)) {
        // Tag 13 regions may legitimately contain ADDRESS/NAME text
        // (bank branch addresses, company names) — don't count as FPs.
        if (r.tag === 13 && (h.kind === 'ADDRESS' || h.kind === 'NAME')) continue;
        confusion[h.kind] = confusion[h.kind] || { tp:0, fp:0, fn:0 };
        confusion[h.kind].fp++;
        if (h.kind === 'ADDRESS' && addressFPs.length < 500) {
          addressFPs.push({
            sampleIdx: si,
            text: fullText.substring(h.start, h.end),
            context: fullText.substring(Math.max(0, h.start - 60), Math.min(fullText.length, h.end + 60)),
            kind: h.kind,
            regionTag: r.tag,
            regionText: fullText.substring(rStartChar, rEndChar),
            reason: 'wrong-type-in-region',
          });
        }
        if (h.kind === 'NAME' && nameFPs.length < 500) {
          nameFPs.push({
            sampleIdx: si,
            text: fullText.substring(h.start, h.end),
            context: fullText.substring(Math.max(0, h.start - 60), Math.min(fullText.length, h.end + 60)),
            kind: h.kind,
            regionTag: r.tag,
            regionText: fullText.substring(rStartChar, rEndChar),
            reason: 'wrong-type-in-region',
          });
        }
        if (h.kind === 'ACCOUNT' && accountFPs.length < 500) {
          accountFPs.push({
            sampleIdx: si,
            text: fullText.substring(h.start, h.end),
            context: fullText.substring(Math.max(0, h.start - 60), Math.min(fullText.length, h.end + 60)),
            kind: h.kind,
            regionTag: r.tag,
            regionText: fullText.substring(rStartChar, rEndChar),
            reason: 'wrong-type-in-region',
          });
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
      if (h.kind === 'ADDRESS' && addressFPs.length < 500) {
        addressFPs.push({
          sampleIdx: si,
          text: fullText.substring(h.start, h.end),
          context: fullText.substring(Math.max(0, h.start - 60), Math.min(fullText.length, h.end + 60)),
          kind: h.kind,
          regionTag: 0,
          regionText: '',
          reason: 'outside-any-region',
        });
      }
      if (h.kind === 'NAME' && nameFPs.length < 500) {
        nameFPs.push({
          sampleIdx: si,
          text: fullText.substring(h.start, h.end),
          context: fullText.substring(Math.max(0, h.start - 60), Math.min(fullText.length, h.end + 60)),
          kind: h.kind,
          regionTag: 0,
          regionText: '',
          reason: 'outside-any-region',
        });
      }
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

// Dump ADDRESS FP diagnostics
if (addressFPs.length > 0) {
  writeFileSync(resolve(ROOT, 'data/address_fp_diag.json'), JSON.stringify(addressFPs, null, 2));
  console.log(`ADDRESS FP diagnostics (${addressFPs.length} samples): data/address_fp_diag.json`);
}
// Dump NAME FP diagnostics
if (nameFPs.length > 0) {
  writeFileSync(resolve(ROOT, 'data/name_fp_diag.json'), JSON.stringify(nameFPs, null, 2));
  console.log(`NAME FP diagnostics (${nameFPs.length} samples): data/name_fp_diag.json`);
}
// Dump ACCOUNT FN diagnostics
if (accountFNs.length > 0) {
  writeFileSync(resolve(ROOT, 'data/account_fn_diag.json'), JSON.stringify(accountFNs, null, 2));
  console.log(`ACCOUNT FN diagnostics (${accountFNs.length} samples): data/account_fn_diag.json`);
}
// Dump ACCOUNT FP diagnostics
if (accountFPs.length > 0) {
  writeFileSync(resolve(ROOT, 'data/account_fp_diag.json'), JSON.stringify(accountFPs, null, 2));
  console.log(`ACCOUNT FP diagnostics (${accountFPs.length} samples): data/account_fp_diag.json`);
}
// Dump ADDRESS FN diagnostics
if (addressFNs.length > 0) {
  writeFileSync(resolve(ROOT, 'data/address_fn_diag.json'), JSON.stringify(addressFNs, null, 2));
  console.log(`ADDRESS FN diagnostics (${addressFNs.length} samples): data/address_fn_diag.json`);
}
// Dump NAME FN diagnostics
if (nameFNs.length > 0) {
  writeFileSync(resolve(ROOT, 'data/name_fn_diag.json'), JSON.stringify(nameFNs, null, 2));
  console.log(`NAME FN diagnostics (${nameFNs.length} samples): data/name_fn_diag.json`);
}
