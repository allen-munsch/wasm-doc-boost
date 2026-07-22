// ocr_worker.mjs — Tesseract.js worker for batch OCR

import { parentPort, workerData } from 'worker_threads';
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import Tesseract from 'tesseract.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const { chunk, subdir } = workerData;

let worker;
async function getWorker() {
  if (!worker) {
    worker = await Tesseract.createWorker('eng', 1, { logger: () => {} });
  }
  return worker;
}

async function ocrImage(filename) {
  const imgPath = resolve(ROOT, 'data', 'images', subdir, filename);
  let imgBuf;
  try {
    imgBuf = readFileSync(imgPath);
  } catch {
    return null;
  }

  const w = await getWorker();
  const { data } = await w.recognize(imgBuf, {}, { text: true, tsv: true });

  // Parse TSV: level 5 = word
  const words = [];
  const tsvLines = (data.tsv || '').split('\n');
  for (const line of tsvLines) {
    if (!line.trim()) continue;
    const cols = line.split('\t');
    if (cols[0] !== '5') continue;
    const conf = parseFloat(cols[10]) || 0;
    const text = cols[11] || '';
    if (!text.trim()) continue;
    words.push({
      text,
      bbox: [
        parseInt(cols[6], 10),  // x0
        parseInt(cols[7], 10),  // y0
        parseInt(cols[8], 10),  // x1
        parseInt(cols[9], 10),  // y1
      ],
      confidence: conf,
    });
  }

  return { filename, ocrText: data.text || '', words, confidence: data.confidence };
}

// Process chunk
const results = [];
let progress = 0;

for (const img of chunk) {
  const result = await ocrImage(img.filename);
  if (result) results.push(result);
  progress++;
  if (progress % 5 === 0) {
    parentPort.postMessage({ type: 'progress' });
    progress = 0;
  }
}
if (progress > 0) {
  parentPort.postMessage({ type: 'progress' });
}

parentPort.postMessage({ type: 'result', data: results });

if (worker) await worker.terminate();
