#!/usr/bin/env node
// Serve wasm-doc-boost frontend with COOP/COEP headers for SharedArrayBuffer.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const PORT = parseInt(process.argv[2], 10) || 8000;
const DIR = path.dirname(new URL(import.meta.url).pathname);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.data': 'application/octet-stream',
};

const NO_CACHE = new Set(['.onnx', '.json', '.wasm', '.mjs']);

http.createServer((req, res) => {
  const urlPath = req.url.split('?')[0];
  const filePath = path.join(DIR, urlPath === '/' ? '/index.html' : urlPath);

  // Security: prevent directory traversal
  if (!filePath.startsWith(DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const ext = path.extname(filePath);
  const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'cross-origin',
    'Content-Type': MIME[ext] || 'application/octet-stream',
  };

  if (NO_CACHE.has(ext)) {
    headers['Cache-Control'] = 'no-store, must-revalidate';
    headers['Pragma'] = 'no-cache';
  }

  // Stream the file directly — Node handles concurrent streams natively
  const stream = fs.createReadStream(filePath);
  stream.on('error', () => {
    res.writeHead(404);
    res.end('Not found');
  });
  stream.on('open', () => {
    res.writeHead(200, headers);
  });
  stream.pipe(res);
  // Don't crash on broken client connection
  res.on('error', () => {});
  stream.on('error', (e) => {
    if (e.code !== 'ENOENT') console.error(`Error streaming ${filePath}: ${e.message}`);
  });
}).listen(PORT, '0.0.0.0', () => {
  console.error(`Serving ${DIR} on http://localhost:${PORT}`);
});
