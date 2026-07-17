#!/usr/bin/env node
// ocr_fatura2.mjs — Batch Tesseract.js OCR on FATURA2 images with word-level bbox
// Usage: node scripts/ocr_fatura2.mjs [--workers N] [--sample N]

import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Worker } from 'worker_threads';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const IMAGE_DIR = resolve(ROOT, 'data', 'images', 'fatura2');

const args = process.argv.slice(2);
const WORKERS = parseInt(args[args.indexOf('--workers') + 1] || '4', 10) || 4;
const SAMPLE = parseInt(args[args.indexOf('--sample') + 1] || '0', 10) || 10000;

// ── Build image list ────────────────────────────────
const images = [];
for (let i = 0; i < Math.min(SAMPLE, 8600); i++) {
  images.push({ filename: `fatura2_train_${i}.jpg`, idx: i });
}
for (let i = 0; i < Math.min(SAMPLE - 8600, 1400); i++) {
  if (images.length >= SAMPLE) break;
  images.push({ filename: `fatura2_test_${i}.jpg`, idx: 8600 + i });
}

console.log(`Processing ${images.length} images with ${WORKERS} workers`);

// ── Split into chunks ───────────────────────────────
const chunkSize = Math.ceil(images.length / WORKERS);
const chunks = [];
for (let i = 0; i < images.length; i += chunkSize) {
  chunks.push(images.slice(i, i + chunkSize));
}

// ── Spawn workers ───────────────────────────────────
const workerPath = resolve(__dirname, 'ocr_worker.mjs');
const start = performance.now();
let completed = 0;

const workerPromises = chunks.map((chunk) =>
  new Promise((resolve, reject) => {
    const w = new Worker(workerPath, { workerData: { chunk } });
    const results = [];
    w.on('message', (msg) => {
      if (msg.type === 'result') results.push(...msg.data);
      else if (msg.type === 'progress') {
        completed += 5;
        const el = ((performance.now() - start) / 1000).toFixed(0);
        console.log(`  ${Math.min(completed, images.length)}/${images.length} (${el}s)`);
      }
    });
    w.on('error', reject);
    w.on('exit', (code) => {
      if (code === 0) resolve(results);
      else reject(new Error(`Worker exit ${code}`));
    });
  }),
);

const allResults = (await Promise.all(workerPromises)).flat();
const elapsed = ((performance.now() - start) / 1000).toFixed(0);
console.log(`Done. ${allResults.length} results in ${elapsed}s`);

// ── Save ────────────────────────────────────────────
const outPath = resolve(ROOT, 'data', 'fatura2_ocr.json');
writeFileSync(outPath, JSON.stringify(allResults));
console.log(`Saved to ${outPath} (${(JSON.stringify(allResults).length / 1e6).toFixed(1)} MB)`);
