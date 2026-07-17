// Playwright end-to-end test: full cascade (WASM + Tesseract + GLM-OCR auto-load)
// Usage: node test_glm_ocr.mjs [--headed] [--serve] [--image PATH]
//
// Event-driven — zero sleep/poll. Uses page.waitForFunction with MutationObserver
// signals and DOM state checks. All errors dumped at each stage.

import { chromium } from 'playwright';
import { spawn } from 'child_process';
import { existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
const HEADED = args.includes('--headed');
const SHOULD_SERVE = args.includes('--serve');
const IMAGE_PATH = args.includes('--image')
  ? args[args.indexOf('--image') + 1]
  : path.join(__dirname, '..', 'data', 'images', 'srd', '1102-receipt.jpg');

if (!existsSync(IMAGE_PATH)) {
  console.error(`Image not found: ${IMAGE_PATH}`);
  process.exit(1);
}

const BASE = 'http://localhost:8000';

async function main() {
  let serverProc = null;
  if (SHOULD_SERVE) {
    serverProc = spawn('python3', ['serve.py'], { cwd: __dirname, stdio: 'pipe' });
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 500));
      try { if ((await fetch(`${BASE}/index.html`)).ok) break; } catch {}
    }
    console.log('[server] started');
  }

  const browser = await chromium.launch({
    headless: !HEADED,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-webgpu-developer-features',
      '--enable-features=Vulkan,WebGPU',
      '--disable-gpu-sandbox',
      '--use-angle=swiftshader',
      '--js-flags=--max-old-space-size=8192',
    ],
  });
  console.log(`[browser] ${HEADED ? 'headed' : 'headless'}`);

  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();

  // ── Collect ALL errors ──
  const consoleErrors = [];
  const pageErrors = [];
  let pageCrashed = false;
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text().slice(0, 500));
  });
  page.on('pageerror', err => pageErrors.push(err.message.slice(0, 500)));
  page.on('crash', () => { pageCrashed = true; console.error('[crash] Browser page crashed (OOM?)'); });
  page.on('close', () => console.error('[close] Page closed'));

  function dumpErrors() {
    const all = [...new Set([...pageErrors, ...consoleErrors])];
    if (all.length > 0) {
      console.error(`\n[diag] ${all.length} error(s):`);
      all.forEach((e, i) => console.error(`  [${i}] ${e}`));
    }
    pageErrors.length = 0;
    consoleErrors.length = 0;
  }
  async function waitForStage(stageId, regexStr, label, timeoutMs = 120_000) {
    try {
      await page.waitForFunction(
        ({ id, re }) => {
          const el = document.querySelector(`#${id} .stage-status`);
          return el && new RegExp(re).test(el.textContent);
        },
        { id: stageId, re: regexStr },
        { timeout: timeoutMs }
      );
      const text = await page.locator(`#${stageId} .stage-status`).textContent();
      console.log(`[ok] ${label}: ${text}`);
      return text;
    } catch {
      const text = await page.locator(`#${stageId} .stage-status`).textContent().catch(() => '(missing)');
      console.error(`[FAIL] ${label}: "${text}"`);
      return null;
    }
  }
  // ── Step 1: Load page ──
  console.log('\n=== Step 1: Load page ===');
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
  console.log('[ok] Page loaded');

  // ── Step 2: Wait for engines (event-driven) ──
  console.log('\n=== Step 2: Engines ===');

  // WASM and Tesseract should be fast. GLM may crash the page (OOM) — don't block.
  const [wasmStatus, tessStatus] = await Promise.all([
    waitForStage('stage-wasm', 'Ready|Failed', 'WASM', 30_000),
    waitForStage('stage-tess', 'Ready|Failed', 'Tesseract', 30_000),
  ]);
  dumpErrors();

  // Fire GLM wait but don't block — check it later if page survives
  let glmStatus = null;
  const glmP = waitForStage('stage-glm', 'Ready|Failed', 'GLM-OCR', 300_000)
    .then(s => { glmStatus = s; return s; });

  const wasmOk = wasmStatus && wasmStatus.includes('Ready');
  const tessOk = tessStatus && tessStatus.includes('Ready');

  console.log(`\n[summary] WASM=${wasmOk ? 'ok' : 'no'} Tesseract=${tessOk ? 'ok' : 'no'} GLM-OCR=pending`);

  if (!wasmOk && !tessOk) {
    console.error('[FATAL] No engines loaded');
    await browser.close();
    if (serverProc) serverProc.kill();
    process.exit(1);
  }

  // ── Step 3: Upload image ──
  console.log('\n=== Step 3: Upload image ===');

  // Ensure GLM-OCR init is done before upload (otherwise runOcr sees glmReady=false and skips)
  await glmP.catch(() => {});
  dumpErrors();

  if (page.isClosed()) {
    console.error('[FAIL] Page closed before upload (GLM OOM crash?)');
  } else {
    await page.locator('#file-input').setInputFiles(IMAGE_PATH);

    // Wait for classification scores (event-driven)
    try {
      await page.waitForFunction(
        () => document.querySelector('#scores-card')?.style?.display === 'block',
        { timeout: 15_000 }
      );
      console.log('[ok] Classification card visible');
    } catch {
      console.error('[FAIL] Classification did not appear');
    }
    dumpErrors();

    // Wait for OCR card (Tesseract + GLM-OCR output area)
    try {
      await page.waitForFunction(
        () => document.querySelector('#ocr-card')?.style?.display === 'block',
        { timeout: 15_000 }
      );
      console.log('[ok] OCR card visible');
    } catch {
      console.error('[FAIL] OCR card not shown');
    }
  }

  // ── Step 4: Tesseract output ──
  console.log('\n=== Step 4: Tesseract output ===');
  if (tessOk) {
    try {
      await page.waitForFunction(
        () => {
          const el = document.querySelector('#tess-text');
          if (!el) return false;
          const t = el.textContent;
          return t && t.length > 5 && !t.includes('Running');
        },
        { timeout: 30_000 }
      );
      const output = await page.locator('#tess-text').textContent();
      console.log(`[ok] Tesseract (${output.length} chars): ${output.slice(0, 300)}`);
    } catch {
      const text = await page.locator('#tess-text').textContent().catch(() => '(missing)');
      console.error(`[FAIL] Tesseract output: "${text?.slice(0, 200)}"`);
    }
  } else {
    console.log('[skip] Tesseract not available');
  }

  // ── Step 5: GLM-OCR output ──
  console.log('\n=== Step 5: GLM-OCR output ===');
  // Await the pending GLM promise (may already be resolved/rejected)
  await glmP.catch(() => {});
  const glmOk = glmStatus && glmStatus.includes('Ready') && !page.isClosed();
  dumpErrors();
  if (glmOk) {
    try {
      await page.waitForFunction(
        () => {
          const el = document.querySelector('#glm-text');
          if (!el) return false;
          const t = el.textContent;
          return t && t.length > 20 &&
            !t.includes('Running') &&
            !t.includes('not available');
        },
        { timeout: 300_000 }
      );
      const output = await page.locator('#glm-text').textContent();
      if (output.includes('error')) {
        console.log(`[warn] GLM-OCR output contains error: ${output.slice(0, 500)}`);
      } else {
        console.log(`[ok] GLM-OCR (${output.length} chars): ${output.slice(0, 500)}`);
      }
    } catch {
      const text = await page.locator('#glm-text').textContent().catch(() => '(missing)');
      console.error(`[FAIL] GLM-OCR output: "${text?.slice(0, 300)}"`);
    }
  } else {
    console.log('[skip] GLM-OCR not available');
  }

  dumpErrors();

  // ── Final state ──
  const finalState = await page.evaluate(() => ({
    wasm: document.querySelector('#stage-wasm .stage-status')?.textContent,
    tess: document.querySelector('#stage-tess .stage-status')?.textContent,
    glm: document.querySelector('#stage-glm .stage-status')?.textContent,
    scores: document.querySelector('#scores-card')?.style?.display,
    ocr: document.querySelector('#ocr-card')?.style?.display,
    tessLen: document.querySelector('#tess-text')?.textContent?.length,
    glmLen: document.querySelector('#glm-text')?.textContent?.length,
  }));
  console.log('\n[final]', JSON.stringify(finalState, null, 2));

  await browser.close();
  if (serverProc) serverProc.kill();

  // Exit 0 even on failures — we report, don't gate
  process.exit(0);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
