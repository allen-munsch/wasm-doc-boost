#!/usr/bin/env node
// eval_pdf.mjs — End-to-end PDF classification and text extraction smoke test
// Usage: node scripts/eval_pdf.mjs

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import init, { classifyPdf, extractTextWithPositions } from '../frontend/wasm_bridge.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

async function loadWasm() {
  const wasmPath = resolve(ROOT, 'frontend/wasm_bridge_bg.wasm');
  const wasmBytes = readFileSync(wasmPath);
  await init(wasmBytes);
}

const TESTS = [
  {
    file: 'tests/fixtures/firecrawl_docs_tagged.pdf',
    describe: 'text-based tagged PDF',
    expectType: 'TextBased',
    expectMinPages: 1,
    expectMinItems: 10,
  },
  {
    file: 'tests/fixtures/scan_with_native_header_text.pdf',
    describe: 'mixed scanned PDF with native header',
    expectType: null, // can be ImageBased or Mixed — key is pages_needing_ocr
    expectMinPages: 1,
    expectHasOcrPage: true,
  },
  {
    file: 'tests/fixtures/broken_startxref_pointer.pdf',
    describe: 'PDF with broken startxref pointer (repair codepath)',
    expectType: null, // only test that it doesn't throw
    expectMinPages: 1,
  },
];

async function main() {
  console.log('Loading WASM...');
  await loadWasm();
  console.log('WASM loaded.\n');

  let passed = 0;
  let failed = 0;

  for (const { file, describe, expectType, expectMinPages, expectMinItems } of TESTS) {
    const fullPath = resolve(ROOT, file);
    console.log(`Testing: ${describe}`);
    console.log(`  File: ${file}`);

    try {
      const buf = readFileSync(fullPath);
      const classifyResult = classifyPdf(buf);
      console.log(`  Classify: type=${classifyResult.pdfType}, pages=${classifyResult.pageCount}, ` +
        `needingOcr=${classifyResult.pagesNeedingOcr.length}, confidence=${classifyResult.confidence.toFixed(3)}`);

      let ok = true;
      if (expectType && classifyResult.pdfType !== expectType) {
        console.log(`  FAIL: expected type ${expectType}, got ${classifyResult.pdfType}`);
        ok = false;
      }
      if (expectMinPages && classifyResult.pageCount < expectMinPages) {
        console.log(`  FAIL: expected >=${expectMinPages} pages, got ${classifyResult.pageCount}`);
        ok = false;
      }

      if (expectMinItems !== undefined) {
        const items = extractTextWithPositions(buf);
        console.log(`  Extract: ${items.length} text items`);
        if (items.length < expectMinItems) {
          console.log(`  FAIL: expected >=${expectMinItems} items, got ${items.length}`);
          ok = false;
        }
      }

      if (ok) {
        console.log('  PASS\n');
        passed++;
      } else {
        failed++;
      }
    } catch (err) {
      console.log(`  FAIL: ${err.message}\n`);
      failed++;
    }
  }

  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => { console.error(err); process.exit(1); });
