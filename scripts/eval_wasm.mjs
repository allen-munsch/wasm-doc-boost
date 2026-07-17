#!/usr/bin/env node
// eval_wasm.mjs — End-to-end WASM classification evaluation (parallel)
// Usage: node scripts/eval_wasm.mjs [--sample N] [--workers N] [--labels data/labels_small.csv]

import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Worker } from 'worker_threads';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const args = process.argv.slice(2);
const SAMPLE = (() => { const i = args.indexOf('--sample'); return i >= 0 ? parseInt(args[i+1],10) : 0; })();
const WORKERS = (() => { const i = args.indexOf('--workers'); return i >= 0 ? parseInt(args[i+1],10) : 8; })();
const LABELS_CSV = (() => { const i = args.indexOf('--labels'); return i >= 0 ? args[i+1] : 'data/labels.csv'; })();

const LABEL_NAMES = ['is_document','is_digital','is_paper','is_crumpled','is_shadow'];

// ── Parse labels CSV ────────────────────────────────
const labelsRaw = readFileSync(resolve(ROOT, LABELS_CSV), 'utf-8').trim().split('\n');
if (labelsRaw[0].startsWith('filename')) labelsRaw.shift();

const samples = labelsRaw.map((line, idx) => {
  const parts = line.split(',');
  return { idx, filename: parts[0], labels: LABEL_NAMES.map((_,i) => parseInt(parts[i+1],10)) };
});

let evalSet = samples;
if (SAMPLE > 0 && SAMPLE < evalSet.length) {
  const byLabel = {};
  for (const s of evalSet) {
    const key = s.labels.join(',');
    (byLabel[key] ??= []).push(s);
  }
  evalSet = [];
  for (const group of Object.values(byLabel)) {
    evalSet.push(...group.slice(0, Math.max(1, Math.ceil(SAMPLE * group.length / samples.length))));
  }
  evalSet = evalSet.slice(0, SAMPLE);
}

console.log(`Evaluating ${evalSet.length} images on ${WORKERS} workers (from ${samples.length} total)`);

// ── Divide into chunks ──────────────────────────────
const chunkSize = Math.ceil(evalSet.length / WORKERS);
const chunks = [];
for (let i = 0; i < evalSet.length; i += chunkSize) {
  chunks.push(evalSet.slice(i, i + chunkSize));
}

// ── Spawn workers ───────────────────────────────────
const workerPath = resolve(__dirname, 'eval_worker.mjs');
const start = performance.now();

const workerPromises = chunks.map((chunk) => {
  return new Promise((resolve, reject) => {
    const w = new Worker(workerPath, {
      workerData: { chunk, root: ROOT },
      env: {},
    });
    w.on('message', resolve);
    w.on('error', reject);
    w.on('exit', (code) => { if (code !== 0) reject(new Error(`Worker exit ${code}`)); });
  });
});

const results = await Promise.all(workerPromises);
const elapsed = ((performance.now() - start) / 1000).toFixed(1);

// ── Aggregate confusion matrices ────────────────────
const confusion = LABEL_NAMES.map(() => ({ tp: 0, fp: 0, tn: 0, fn: 0 }));
let totalClassified = 0;
let totalErrors = 0;

for (const r of results) {
  totalClassified += r.classified;
  totalErrors += r.errors;
  for (let i = 0; i < LABEL_NAMES.length; i++) {
    confusion[i].tp += r.confusion[i].tp;
    confusion[i].fp += r.confusion[i].fp;
    confusion[i].tn += r.confusion[i].tn;
    confusion[i].fn += r.confusion[i].fn;
  }
}

console.log(`Done. ${totalClassified} classified, ${totalErrors} errors in ${elapsed}s`);

// ── Compute metrics ─────────────────────────────────
const metrics = {};
for (let i = 0; i < LABEL_NAMES.length; i++) {
  const { tp, fp, tn, fn } = confusion[i];
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  metrics[LABEL_NAMES[i]] = {
    precision: +precision.toFixed(4), recall: +recall.toFixed(4), f1: +f1.toFixed(4),
    accuracy: +((tp + tn) / (tp + fp + tn + fn)).toFixed(4),
    support: tp + fn,
    tp, fp, tn, fn,
  };
}

const macro = { precision: 0, recall: 0, f1: 0 };
for (const name of LABEL_NAMES) {
  macro.precision += metrics[name].precision;
  macro.recall += metrics[name].recall;
  macro.f1 += metrics[name].f1;
}
macro.precision = +(macro.precision / 5).toFixed(4);
macro.recall = +(macro.recall / 5).toFixed(4);
macro.f1 = +(macro.f1 / 5).toFixed(4);
metrics.macro_avg = macro;

// ── Print ───────────────────────────────────────────
console.log('\n=== WASM Classification Evaluation ===');
console.log(`Images: ${evalSet.length}, Errors: ${totalErrors}, Time: ${elapsed}s\n`);
for (const name of [...LABEL_NAMES, 'macro_avg']) {
  const m = metrics[name];
  console.log(`  ${name.padEnd(14)} P=${m.precision.toFixed(4)}  R=${m.recall.toFixed(4)}  F1=${m.f1.toFixed(4)}`);
}

const reportPath = resolve(ROOT, 'data/wasm_report.json');
writeFileSync(reportPath, JSON.stringify(metrics, null, 2) + '\n');
console.log(`\nReport: ${reportPath}`);
