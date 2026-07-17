// eval_worker.mjs — Worker thread for eval_wasm.mjs
// Loads WASM + model, classifies its chunk of images, returns confusion matrices.

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { workerData, parentPort } from 'worker_threads';

const { chunk, root } = workerData;
const LABEL_NAMES = ['is_document','is_digital','is_paper','is_crumpled','is_shadow'];
const IMAGES_DIR = resolve(root, 'data/images');
const MODEL_JSON = resolve(root, 'data/model.json');

// Load WASM
const wasmBytes = readFileSync(resolve(root, 'frontend/wasm_bridge_bg.wasm'));
const { initSync, classify_file, load_model } = await import(
  resolve(root, 'frontend/wasm_bridge.js')
);
initSync(new WebAssembly.Module(wasmBytes));

// Load model
load_model(readFileSync(MODEL_JSON, 'utf-8'));

const confusion = LABEL_NAMES.map(() => ({ tp: 0, fp: 0, tn: 0, fn: 0 }));
let classified = 0;
let errors = 0;

for (const sample of chunk) {
  const imgPath = resolve(IMAGES_DIR, sample.filename);
  let bytes;
  try {
    bytes = readFileSync(imgPath);
  } catch {
    errors++;
    for (let i = 0; i < 5; i++) {
      if (sample.labels[i] === 0) confusion[i].tn++;
      else confusion[i].fn++;
    }
    continue;
  }

  let result;
  try {
    result = classify_file(bytes);
  } catch {
    errors++;
    for (let i = 0; i < 5; i++) {
      if (sample.labels[i] === 0) confusion[i].tn++;
      else confusion[i].fn++;
    }
    continue;
  }

  for (let i = 0; i < 5; i++) {
    const pred = result[LABEL_NAMES[i]] >= 0.5 ? 1 : 0;
    const actual = sample.labels[i];
    if (pred === 1 && actual === 1) confusion[i].tp++;
    else if (pred === 1 && actual === 0) confusion[i].fp++;
    else if (pred === 0 && actual === 0) confusion[i].tn++;
    else confusion[i].fn++;
  }
  classified++;
}

parentPort.postMessage({ classified, errors, confusion });
