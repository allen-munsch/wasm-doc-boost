// eval_rotation_gold.mjs — run the shipped v2 rotation model on fatura2
// gold-labeled (born-digital, upright) invoices and report rotation numbers.
// Usage: node scripts/eval_rotation_gold.mjs [N] [startIdx]
//   N defaults to 1000 (fatura2_test_0 .. _999); startIdx defaults to 0.
// Gold truth: every fatura2 invoice is stored upright -> rotation_deg = 0.

import { readFileSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(process.cwd());
const N = parseInt(process.argv[2] || '1000', 10);
const START = parseInt(process.argv[3] || '0', 10);

const wasmBytes = readFileSync(resolve(ROOT, 'frontend/wasm_bridge_bg.wasm'));
const { initSync, classify_file_v2, load_model_v2, model_version } = await import(
  resolve(ROOT, 'frontend/wasm_bridge.js')
);
initSync(new WebAssembly.Module(wasmBytes));
load_model_v2(readFileSync(resolve(ROOT, 'data/model_v4.json'), 'utf-8'));
if (model_version() !== 4) throw new Error(`expected model v4, got ${model_version()}`);

function circularErr(a, b, period = 360) {
  const d = Math.abs(a - b);
  return Math.min(d, period - d);
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const frac = (a) => a.filter(Boolean).length / a.length;

const errs = [], axisErrs = [], confs = [];
let n = 0, nDoc = 0, notDoc = 0, review = 0, flipErr = 0;
const dist = {};

for (let i = START; i < START + N; i++) {
  const fn = `fatura2_test_${i}.jpg`;
  const bytes = readFileSync(resolve(ROOT, 'data/images/fatura2', fn));
  const r = classify_file_v2(bytes);
  n++;
  const isDoc = r.is_document >= 0.5;
  if (isDoc) nDoc++; else notDoc++;
  const deg = r.rotation_deg;
  const e = circularErr(deg, 0);
  const ae = circularErr(deg % 180, 0, 180);
  errs.push(e);
  axisErrs.push(ae);
  confs.push(r.rotation_confidence);
  if (r.needs_review) review++;
  if (e > 90) flipErr++;
  const b = Math.round(deg);
  dist[b] = (dist[b] || 0) + 1;
  if (n % 100 === 0) process.stderr.write(`progress ${n}/${N}\n`);
}

errs.sort((a, b) => a - b);
const p = (q) => errs[Math.floor((errs.length - 1) * q)];

console.log(JSON.stringify({
  n,
  is_document: nDoc,
  not_document: notDoc,
  mean_strict_err_deg: +mean(errs).toFixed(3),
  median_strict_err_deg: +p(0.5).toFixed(3),
  p90_strict_err_deg: +p(0.9).toFixed(3),
  p99_strict_err_deg: +p(0.99).toFixed(3),
  pm1_accuracy: +frac(errs.map((e) => e <= 1)).toFixed(4),
  pm2_accuracy: +frac(errs.map((e) => e <= 2)).toFixed(4),
  pm5_accuracy: +frac(errs.map((e) => e <= 5)).toFixed(4),
  axis_mean_err_deg: +mean(axisErrs).toFixed(3),
  axis_pm1_accuracy: +frac(axisErrs.map((e) => e <= 1)).toFixed(4),
  flip_error_rate_strict: +(flipErr / n).toFixed(4),
  needs_review_rate: +(review / n).toFixed(4),
  mean_confidence: +mean(confs).toFixed(3),
  rotation_deg_distribution: Object.keys(dist).sort((a, b) => a - b).map((k) => [k, dist[k]]),
}, null, 1));
