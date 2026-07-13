#!/usr/bin/env node
// Node.js integration test — validates the full classify pipeline.
// Run: node --test tests/node_integration.mjs
// Requires: wasm-pack build --target nodejs (pkg/ must exist)

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { deflateSync } from 'node:zlib';

const require = createRequire(import.meta.url);
const wasmBridge = require('../pkg/wasm_bridge.js');

const TRIVIAL_MODEL = JSON.stringify([
    [{ nodeid: 0, depth: 0, leaf: 0.0 }],
    [{ nodeid: 0, depth: 0, leaf: 0.0 }],
    [{ nodeid: 0, depth: 0, leaf: 0.0 }],
    [{ nodeid: 0, depth: 0, leaf: 0.0 }],
    [{ nodeid: 0, depth: 0, leaf: 0.0 }],
]);

function crc32(buf) {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) {
            c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        }
        table[n] = c;
    }
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) {
        c = table[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    }
    return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const typeB = Buffer.from(type, 'ascii');
    const crcInput = Buffer.concat([typeB, data]);
    const crcVal = Buffer.alloc(4);
    crcVal.writeUInt32BE(crc32(crcInput), 0);
    return Buffer.concat([len, typeB, data, crcVal]);
}

function makePng(w, h) {
    // Build IDAT: raw RGB rows, each preceded by filter byte 0 (None)
    const raw = Buffer.alloc(h * (1 + w * 3));
    for (let y = 0; y < h; y++) {
        const rowOff = y * (1 + w * 3);
        raw[rowOff] = 0; // filter: None
        for (let x = 0; x < w; x++) {
            const off = rowOff + 1 + x * 3;
            raw[off] = (x * 4) & 0xff;
            raw[off + 1] = (y * 4) & 0xff;
            raw[off + 2] = ((x + y) * 2) & 0xff;
        }
    }

    const idat = deflateSync(raw);

    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(w, 0);
    ihdr.writeUInt32BE(h, 4);
    ihdr[8] = 8;  // bit depth
    ihdr[9] = 2;  // color type: RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace

    const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    return Buffer.concat([
        signature,
        pngChunk('IHDR', ihdr),
        pngChunk('IDAT', idat),
        pngChunk('IEND', Buffer.alloc(0)),
    ]);
}

describe('wasm-bridge Node.js integration', () => {
    it('load_model succeeds', () => {
        assert.doesNotThrow(() => wasmBridge.load_model(TRIVIAL_MODEL));
    });

    it('classify_file returns 5 labels with sigmoid(0) ≈ 0.5', () => {
        const png = makePng(64, 64);
        const result = wasmBridge.classify_file(png);

        const expectedLabels = ['is_document', 'is_digital', 'is_paper', 'is_crumpled', 'is_shadow'];
        for (const label of expectedLabels) {
            const val = result[label];
            assert.ok(typeof val === 'number', `${label}: expected number, got ${typeof val}`);
            assert.ok(Math.abs(val - 0.5) < 0.01, `${label}: expected ~0.5, got ${val}`);
        }
    });

    it('classify_file rejects invalid input', () => {
        assert.throws(
            () => wasmBridge.classify_file(Buffer.from([0, 1, 2, 3])),
            /Image decode error/,
        );
    });
});
