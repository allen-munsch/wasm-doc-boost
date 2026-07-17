// Quick Node.js test: load GLM-OCR ONNX model directly via onnxruntime-web
// Usage: node --max-old-space-size=8192 test_onnx_node.mjs
// Skips browser/Playwright/transformers.js — just tests ORT WASM + ONNX files.

import * as ort from 'onnxruntime-web';
import { readFileSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ONNX_DIR = path.join(__dirname, 'vendor', 'glm-ocr-onnx', 'onnx');

async function main() {
  console.log('=== ORT Node.js ONNX Test ===');
  console.log(`ort version: ${ort.env.version || 'unknown'}`);

  // Step 1: Check files exist
  const files = [
    'decoder_model_merged_q4f16.onnx',
    'decoder_model_merged_fp16.onnx',
    'embed_tokens_q4f16.onnx',
    'embed_tokens_fp16.onnx',
    'vision_encoder_q4f16.onnx',
    'vision_encoder_fp16.onnx',
  ];
  for (const f of files) {
    const fp = path.join(ONNX_DIR, f);
    const exists = existsSync(fp);
    const size = exists ? readFileSync(fp).length : 0;
    const mb = (size / 1024 / 1024).toFixed(0);
    console.log(`  ${exists ? '✓' : '✗'} ${f} (${mb} MB)`);
  }

  // Step 2: Try loading the smallest model (embed_tokens_q4f16.onnx, 51MB)
  const modelPath = path.join(ONNX_DIR, 'embed_tokens_q4f16.onnx');
  console.log(`\n--- Loading: ${path.basename(modelPath)} ---`);
  const t0 = Date.now();

  try {
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
      logSeverityLevel: 1,
      enableMemPattern: false,
    });
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`Loaded in ${dt}s`);

    // Print session info
    console.log(`Inputs: ${session.inputNames.join(', ')}`);
    console.log(`Outputs: ${session.outputNames.join(', ')}`);

    // Step 3: Try a dummy inference with zeros
    const inputName = session.inputNames[0];
    const inputShape = [1, 1];  // minimal shape — real input is [1, seq_len]
    const inputData = new BigInt64Array([0n]);
    const feed = {};
    feed[inputName] = new ort.Tensor('int64', inputData, inputShape);

    console.log(`\n--- Dummy inference on ${inputName} ---`);
    const results = await session.run(feed);
    const outputName = session.outputNames[0];
    const output = results[outputName];
    console.log(`Output: ${outputName} shape=[${output.dims}] type=${output.type}`);
    console.log(`Result sample: ${Array.from(output.data).slice(0, 5)}`);

    console.log('\n✓ ONNX model loads and runs in Node.js/WASM!');
  } catch (e) {
    console.error(`Failed: ${e.message}`);
    console.error(e.stack?.split('\n').slice(0, 5).join('\n'));
    process.exit(1);
  }
}

main().catch(e => {
  console.error('Fatal:', e);
  process.exit(1);
});
