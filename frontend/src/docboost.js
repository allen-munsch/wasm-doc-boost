// ── DocBoost: one-shot document analysis pipeline ─────────────────────
//
// Usage:
//   import { createDocBoost } from './src/docboost.js';
//   const db = await createDocBoost({ modelPath: 'model.json', ... });
//   const doc = await db.analyze(imageFile);
//   for (const r of regions(doc)) drawBox(r.bbox);
//
// Init is once. analyze is per-document.

import init, { load_model, classify_file, scan_pii } from '../wasm_bridge.js';
import { ClassifySchema, PiiHitSchema } from './schemas.js';
import { normalizeTesseract, normalizeGlmOcr } from './normalize.js';
import { mergeRegions } from './merge.js';
import { buildEnrichedDocument } from './enrich.js';

// ═══════════════════════════════════════════════════════════════════════
// Factory
// ═══════════════════════════════════════════════════════════════════════

/**
 * Create a DocBoost instance (init WASM, Tesseract, health-check GLM).
 *
 * @param {object} config
 * @param {string} config.modelPath        — path to model.json
 * @param {string} config.tessWorkerPath   — e.g. 'vendor/tesseract/worker.min.js'
 * @param {string} config.tessCorePath     — e.g. 'vendor/tesseract'
 * @param {string} [config.tessLang='eng'] — Tesseract language
 * @param {string} [config.glmBackend]     — e.g. 'http://localhost:8765'
 * @param {function} [onProgress]          — (stage, status, msg) => void
 * @returns {Promise<{config, analyze}>}
 */
export async function createDocBoost(config, onProgress) {
    var report = onProgress || function () {};

    // ── Init WASM ──
    report('wasm', 'running', 'Loading WASM...');
    try {
        await init();
        report('wasm', 'ready', 'WASM loaded');
    } catch (e) {
        report('wasm', 'err', 'WASM init: ' + (e.message || String(e)));
        throw e;
    }

    // ── Load GBDT model ──
    report('wasm', 'running', 'Loading model...');
    try {
        var resp = await fetch(config.modelPath);
        if (!resp.ok) throw new Error('HTTP ' + resp.status + ' fetching model.json');
        load_model(await resp.text());
        report('wasm', 'ok', 'Model loaded');
    } catch (e) {
        report('wasm', 'err', 'Model load: ' + (e.message || String(e)));
        throw e;
    }

    // ── Init Tesseract ──
    var tessWorker = null;
    report('tesseract', 'running', 'Loading Tesseract...');

    try {
        var Tess = globalThis.Tesseract;
        if (!Tess) throw new Error('Tesseract global not found — is tesseract.min.js loaded?');

        tessWorker = await Tess.createWorker(config.tessLang || 'eng', 1, {
            workerPath: config.tessWorkerPath,
            corePath: config.tessCorePath,
        });
        report('tesseract', 'ok', 'Tesseract ready');
    } catch (e) {
        report('tesseract', 'err', 'Tesseract init: ' + (e.message || String(e)));
        // Non-fatal — OCR can still work via GLM
    }

    // ── Health-check GLM-OCR ──
    var glmAvailable = false;
    if (config.glmBackend) {
        report('glm', 'running', 'Checking GLM-OCR...');
        try {
            var hResp = await fetch(config.glmBackend + '/health', {
                signal: AbortSignal.timeout(3000),
            });
            if (hResp.ok) {
                glmAvailable = true;
                report('glm', 'ok', 'Backend ready');
            } else {
                report('glm', 'err', 'Health check: HTTP ' + hResp.status);
            }
        } catch (e) {
            report('glm', 'err', 'Backend unreachable');
        }
    }

    // ── Return the API ──
    return {
        config: config,
        analyze: analyze,
    };

    // ═══════════════════════════════════════════════════════════════
    // analyze — closure captures tessWorker, glmAvailable, config
    // ═══════════════════════════════════════════════════════════════

    /**
     * Analyze an image file/blob/element — classify, OCR, merge, scan PII.
     *
     * @param {File|Blob|HTMLImageElement} imageOrFile
     * @returns {Promise<object>} EnrichedDocument data object
     */
    async function analyze(imageOrFile) {
        var t0 = performance.now();
        var bytes, imgEl;

        // Convert input to Uint8Array for WASM + get dimensions
        if (imageOrFile instanceof File || imageOrFile instanceof Blob) {
            bytes = new Uint8Array(await imageOrFile.arrayBuffer());
            imgEl = await blobToImage(imageOrFile);
        } else if (imageOrFile instanceof HTMLImageElement) {
            bytes = await imageElementToBytes(imageOrFile);
            imgEl = imageOrFile;
        } else {
            throw new Error('analyze expects File, Blob, or HTMLImageElement');
        }

        var imageW = imgEl.naturalWidth;
        var imageH = imgEl.naturalHeight;

        // ── Step 1: Classification (sync, fast) ──
        var classification = null;
        try {
            classification = ClassifySchema.parse(classify_file(bytes));
        } catch (e) {
            report('classify', 'err', e.message || String(e));
        }

        // ── Step 2: OCR in parallel ──
        var tessResult = null;
        var glmResult = null;

        var parallel = [];
        parallel.push(runTess(imgEl));
        if (glmAvailable) {
            parallel.push(runGlm(bytes, imgEl));
        }

        var settled = await Promise.allSettled(parallel);
        var idx = 0;

        if (tessWorker) {
            tessResult = settled[idx].status === 'fulfilled' ? settled[idx].value : null;
            idx++;
        }

        if (glmAvailable) {
            glmResult = settled[idx].status === 'fulfilled' ? settled[idx].value : null;
        }

        // ── Step 3: Normalize → Region[] ──
        var tessRegions = tessResult ? normalizeTesseract(tessResult) : [];
        var glmRegions = glmResult ? normalizeGlmOcr(glmResult) : [];

        // ── Step 4: Merge ──
        var merged = mergeRegions(tessRegions, glmRegions);

        // ── Step 5: PII scan on combined text ──
        var combinedText = merged.regions.map(function (r) { return r.text; }).join('\n');
        var pii = [];
        try {
            var rawPii = scan_pii(combinedText);
            if (rawPii && rawPii.length) {
                pii = PiiHitSchema.array().parse(rawPii);
            }
        } catch (e) {
            report('pii', 'err', e.message || String(e));
        }

        // ── Step 6: Build enriched document ──
        return buildEnrichedDocument({
            image: {
                width: imageW,
                height: imageH,
                mime: imageOrFile.type || (imageOrFile instanceof HTMLImageElement ? 'image/png' : 'application/octet-stream'),
            },
            classification: classification,
            regions: merged.regions,
            pii: pii,
            tesseract: tessResult,
            glmOcr: glmResult,
            analysisMs: Math.round(performance.now() - t0),
            mergeStats: merged.stats,
        });
    }

    // ── OCR runners ───────────────────────────────────────────────

    async function runTess(imageEl) {
        if (!tessWorker) return null;
        return tessWorker.recognize(imageEl, {}, {
            blocks: true,
            hocr: true,
            tsv: true,
            text: true,
        });
    }

    async function runGlm(bytes, imgEl) {
        var b64;
        try {
            b64 = bytesToBase64(bytes);
        } catch (_) {
            // Fallback: canvas-based encoding
            b64 = imageElToBase64(imgEl);
        }

        var resp = await fetch(config.glmBackend + '/ocr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: b64 }),
        });
        if (!resp.ok) throw new Error('GLM-OCR HTTP ' + resp.status + ': ' + await resp.text());
        return resp.json();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

function blobToImage(blob) {
    return new Promise(function (resolve, reject) {
        var img = new Image();
        img.onload = function () { return resolve(img); };
        img.onerror = function () { return reject(new Error('Failed to load image from blob')); };
        img.src = URL.createObjectURL(blob);
    });
}

async function imageElementToBytes(img) {
    var canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    var blob = await new Promise(function (resolve) {
        return canvas.toBlob(resolve, 'image/png');
    });
    return new Uint8Array(await blob.arrayBuffer());
}

function bytesToBase64(bytes) {
    var binary = '';
    for (var i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

function imageElToBase64(img) {
    var canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
}
