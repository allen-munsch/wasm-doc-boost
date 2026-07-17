// Debug GLM-OCR model loading in Node.js using full transformers.js source
// Usage: node --max-old-space-size=8192 test_glm_node.mjs
//
// Node.js package uses 'cpu' (native ORT), not 'wasm' (browser-only).
// Browser uses 'wasm' or 'webgpu'.

import { env, AutoProcessor, GlmOcrForConditionalGeneration } from '@huggingface/transformers';
import { existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

env.localModelPath = __dirname;
env.allowLocalModels = true;
env.allowRemoteModels = false;

const MODEL = 'onnx-community/GLM-OCR-ONNX';

async function main() {
  console.log('=== Transformers.js (source) Node.js GLM-OCR Test ===');
  console.log(`localModelPath: ${env.localModelPath}`);

  const expectedDir = path.join(__dirname, 'vendor', 'glm-ocr-onnx');
  for (const f of ['config.json', 'tokenizer.json', 'onnx/embed_tokens_q4f16.onnx']) {
    const fp = path.join(expectedDir, f);
    console.log(`  ${existsSync(fp) ? '✓' : '✗'} vendor/glm-ocr-onnx/${f}`);
  }

  // Load processor
  console.log('\n--- Loading processor ---');
  const t0 = Date.now();
  const processor = await AutoProcessor.from_pretrained(MODEL, {
    local_files_only: true,
    progress_callback: (info) => {
      if (info.status === 'progress' || info.status === 'download')
        console.log(`  [${info.status}] ${info.file || ''}`);
    },
  });
  console.log(`✓ Processor loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  // Node.js ORT uses 'cpu' provider, not 'wasm'
  // Try fp16 first (standard ops, merged files, no external data needed)
  for (const [device, dtype, label] of [
    ['cpu', 'fp16', 'cpu/fp16 (~2.1GB, standard ops)'],
    ['cpu', 'q4f16', 'cpu/q4f16 (~624MB, GatherBlockQuantized)'],
  ]) {
    console.log(`\n--- Loading model (${label}) ---`);
    const t1 = Date.now();
    try {
      const model = await GlmOcrForConditionalGeneration.from_pretrained(MODEL, {
        device,
        dtype,
        local_files_only: true,
        session_options: { enableMemoryPattern: false, logSeverityLevel: 2 },
        progress_callback: (info) => {
          if (info.status === 'progress' && info.progress > 10)
            console.log(`  load: ${info.file || 'model'} ${info.progress?.toFixed(0) || ''}%`);
        },
      });
      console.log(`✓ Model (${device}/${dtype}) loaded in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
      console.log('\n✓✓✓ GLM-OCR works in Node.js!');
      return;
    } catch (e) {
      console.error(`✗ ${device}/${dtype} failed: ${e.message.split('\n')[0]}`);
    }
  }
}

main().catch(e => {
  console.error(`FAIL: ${e.message}`);
  if (e.stack) for (const l of e.stack.split('\n').slice(0, 6)) console.error(l);
  process.exit(1);
});
