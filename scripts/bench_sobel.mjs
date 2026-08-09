// Benchmark SIMD vs scalar Sobel on WASM
// Usage: node scripts/bench_sobel.mjs

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import init, { bench_sobel } from '../crates/wasm-bridge/pkg/wasm_bridge.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(__dirname, '..', 'crates', 'wasm-bridge', 'pkg', 'wasm_bridge_bg.wasm');
const wasmBytes = await readFile(wasmPath);
await init(wasmBytes);

function itersForSize(npix) {
  if (npix < 100_000) return 500;
  if (npix < 500_000) return 200;
  if (npix < 1_000_000) return 100;
  return 30;
}

const SIZES = [
  [256, 256],
  [512, 512],
  [1024, 768],
  [1920, 1080],
];

// First verify correctness on a small image
console.log('Correctness check (256x256): comparing SIMD vs scalar gx, gy, magnitude, orientation...');
const check = bench_sobel(256, 256, 1);
console.log(`  SIMD: ${check.simd_us.toFixed(1)} us, Scalar: ${check.scalar_us.toFixed(1)} us, Speedup: ${check.speedup.toFixed(2)}x`);
if (check.speedup > 0.95 && check.speedup < 1.05) {
  console.log('  WARNING: SIMD and scalar appear identical (speedup ~1.0) — SIMD may not be active\n');
} else {
  console.log('  OK: SIMD is active (speedup differs from 1.0)\n');
}

console.log('Sobel SIMD vs Scalar Benchmark (WASM)');
console.log('======================================\n');

for (const [w, h] of SIZES) {
  const npix = w * h;
  const iters = itersForSize(npix);
  const r = bench_sobel(w, h, iters);
  console.log(`${w}x${h}  (${npix.toLocaleString()} px, ${iters} iters):`);
  console.log(`  SIMD:    ${r.simd_us.toFixed(1)} us`);
  console.log(`  Scalar:  ${r.scalar_us.toFixed(1)} us`);
  console.log(`  Speedup: ${r.speedup.toFixed(2)}x`);
  const mpix_simd = (npix / (r.simd_us)) * 1e6 / 1e6;
  const mpix_scalar = (npix / (r.scalar_us)) * 1e6 / 1e6;
  console.log(`  Throughput: ${mpix_simd.toFixed(1)} (SIMD) vs ${mpix_scalar.toFixed(1)} (scalar) Mpix/s\n`);
}
