// ── DocBoost: one-shot document analysis pipeline ─────────────────────
//
// Usage:
//   import { createDocBoost } from './src/docboost.js';
//   const db = await createDocBoost({ modelPath: 'model.json', ... });
//   const doc = await db.analyze(imageFile);
//   for (const r of regions(doc)) drawBox(r.bbox);
//
// Init is once. analyze is per-document.

import init, { load_model, classify_file, classifyPdf, extractTextWithPositions, scan_pii } from '../wasm_bridge.js';
import { ClassifySchema, PdfClassifySchema, PdfTextItemSchema, PiiHitSchema } from './schemas.js';
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
        analyzePdf: analyzePdf,
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

        // ── Step 2: OCR in parallel, normalize, merge ──
        var { regions: mergedRegions, tessResult, glmResult, mergeStats } = await runOcrPipeline(imgEl, bytes);

        // ── Step 5: PII scan on combined text ──
        var combinedText = mergedRegions.map(function (r) { return r.text; }).join('\n');
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
            regions: mergedRegions,
            pii: pii,
            tesseract: tessResult,
            glmOcr: glmResult,
            analysisMs: Math.round(performance.now() - t0),
            mergeStats: mergeStats,
        });
    }

    // ═══════════════════════════════════════════════════════════════
    // analyzePdf — mixed PDF: native text for text pages, OCR for scans
    // ═══════════════════════════════════════════════════════════════

    /**
     * Analyze a PDF file — classify, split text/scanned pages, OCR scans, merge.
     *
     * @param {File|Blob} pdfFile
     * @returns {Promise<object>} EnrichedDocument data object
     */
    async function analyzePdf(pdfFile) {
        var t0 = performance.now();
        var bytes = new Uint8Array(await pdfFile.arrayBuffer());

        // ── Step 1: Classify PDF type ──
        var pdfClassify = null;
        try {
            pdfClassify = PdfClassifySchema.parse(classifyPdf(bytes));
            report('classify', 'ok', 'PDF: ' + pdfClassify.pdfType + ', ' + pdfClassify.pageCount + ' pages');
        } catch (e) {
            report('classify', 'err', e.message || String(e));
            throw e;
        }

        // ── Step 2: Extract all native text items ──
        var textItems = [];
        try {
            var rawItems = extractTextWithPositions(bytes);
            textItems = PdfTextItemSchema.array().parse(rawItems);
            report('ocr', 'ok', 'Extracted ' + textItems.length + ' native text items');
        } catch (e) {
            report('ocr', 'err', e.message || String(e));
        }

        var needingOcr = new Set(pdfClassify.pagesNeedingOcr || []);
        var allRegions = [];
        var tessResult = null;
        var glmResult = null;
        var mergeStats = { tesseractRegions: 0, glmRegions: 0, mergedPairs: 0, finalRegions: 0 };

        // ── Step 3: Native regions from text-based pages (not in pagesNeedingOcr) ──
        var nativeCount = 0;
        var nativeItemsByPage = {};
        for (var i = 0; i < textItems.length; i++) {
            var item = textItems[i];
            if (needingOcr.has(item.page)) continue;
            if (!nativeItemsByPage[item.page]) nativeItemsByPage[item.page] = [];
            nativeItemsByPage[item.page].push(item);
            nativeCount++;
        }
        report('ocr', 'ok', 'Native text from ' + Object.keys(nativeItemsByPage).length + ' text-based page(s), ' + nativeCount + ' items');

        for (var pg in nativeItemsByPage) {
            var pgItems = nativeItemsByPage[pg];
            for (var j = 0; j < pgItems.length; j++) {
                var ni = pgItems[j];
                allRegions.push({
                    id: 'pdf-' + ni.page + '-' + j,
                    text: ni.text,
                    bbox: { x0: ni.x, y0: ni.y, x1: ni.x + ni.width, y1: ni.y + ni.height },
                    confidence: 100,
                    source: 'pdf',
                    sourceDetail: {
                        blockType: ni.itemType,
                        level: null,
                        parentId: null,
                        regionIndex: j,
                    },
                    mergeDecision: null,
                });
            }
        }

        // ── Step 4: OCR-needed pages → render + run image pipeline ──
        var ocrPages = Array.from(needingOcr).sort(function (a, b) { return a - b; });
        var ocrRegionCount = 0;

        for (var k = 0; k < ocrPages.length; k++) {
            var pageIdx = ocrPages[k]; // 0-indexed from WASM
            var pdfPageNum = pageIdx + 1; // 1-indexed for pdf.js

            report('ocr', 'running', 'Rendering page ' + pdfPageNum + ' for OCR...');

            var rendered;
            try {
                rendered = await renderPdfPage(bytes, pdfPageNum, 2.0);
            } catch (e) {
                report('ocr', 'err', 'Failed to render page ' + pdfPageNum + ': ' + (e.message || String(e)));
                continue;
            }

            var pageBytes;
            try {
                pageBytes = await canvasToBytes(rendered.canvas);
            } catch (e) {
                report('ocr', 'err', 'Failed to convert page ' + pdfPageNum + ' canvas: ' + (e.message || String(e)));
                continue;
            }

            var ocr = await runOcrPipeline(rendered.canvas, pageBytes);
            if (!tessResult) tessResult = ocr.tessResult;
            if (!glmResult) glmResult = ocr.glmResult;

            // Convert OCR bboxes from canvas pixel coords (top-left) to PDF user space (bottom-left)
            // PDF user space origin: bottom-left, Y increases upward
            // Canvas origin: top-left, Y increases downward
            // pdf.js viewport maps PDF → canvas by flipping Y internally
            var scale = 2.0;
            var pdfH = rendered.originalHeight;

            for (var m = 0; m < ocr.regions.length; m++) {
                var r = ocr.regions[m];
                var canvasX0 = r.bbox.x0;
                var canvasY0 = r.bbox.y0; // top of bbox in canvas
                var canvasX1 = r.bbox.x1;
                var canvasY1 = r.bbox.y1; // bottom of bbox in canvas

                allRegions.push({
                    id: 'pdf-ocr-' + pageIdx + '-' + (ocrRegionCount++),
                    text: r.text,
                    bbox: {
                        x0: canvasX0 / scale,
                        y0: pdfH - canvasY1 / scale,
                        x1: canvasX1 / scale,
                        y1: pdfH - canvasY0 / scale,
                    },
                    confidence: r.confidence,
                    source: 'pdf-ocr',
                    sourceDetail: r.sourceDetail || {},
                    mergeDecision: null,
                });
            }

            report('ocr', 'ok', 'Page ' + pdfPageNum + ': ' + ocr.regions.length + ' OCR regions');
        }

        // ── Step 5: Sort all regions by page ──
        allRegions.sort(function (a, b) {
            var pa = parseInt(a.id.replace('pdf-ocr-', '').replace('pdf-', '').split('-')[0]);
            var pb = parseInt(b.id.replace('pdf-ocr-', '').replace('pdf-', '').split('-')[0]);
            if (pa !== pb) return pa - pb;
            return a.bbox.y1 - b.bbox.y1;
        });

        // ── Step 6: PII scan on combined text ──
        var combinedText = allRegions.map(function (r) { return r.text; }).join('\n');
        var pii = [];
        try {
            var rawPii = scan_pii(combinedText);
            if (rawPii && rawPii.length) {
                pii = PiiHitSchema.array().parse(rawPii);
            }
        } catch (e) {
            report('pii', 'err', e.message || String(e));
        }

        // ── Step 7: Build enriched document ──
        return buildEnrichedDocument({
            image: null,
            pdf: pdfClassify,
            classification: null,
            regions: allRegions,
            pii: pii,
            tesseract: tessResult,
            glmOcr: glmResult,
            analysisMs: Math.round(performance.now() - t0),
            mergeStats: mergeStats,
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

    // ── Shared OCR pipeline (used by analyze + analyzePdf) ─────────

    /**
     * Run Tesseract + GLM-OCR in parallel, normalize, merge.
     * @param {HTMLImageElement|HTMLCanvasElement} imageSource
     * @param {Uint8Array} [optBytes] — raw bytes (required for GLM path)
     * @returns {Promise<{regions:object[], tessResult:object|null, glmResult:object|null, mergeStats:object}>}
     */
    async function runOcrPipeline(imageSource, optBytes) {
        var tessResult = null;
        var glmResult = null;

        var parallel = [];
        parallel.push(runTess(imageSource));
        if (glmAvailable && optBytes) {
            parallel.push(runGlm(optBytes, imageSource));
        }

        var settled = await Promise.allSettled(parallel);
        var idx = 0;

        if (tessWorker) {
            tessResult = settled[idx].status === 'fulfilled' ? settled[idx].value : null;
            idx++;
        }

        if (glmAvailable && optBytes) {
            glmResult = settled[idx].status === 'fulfilled' ? settled[idx].value : null;
        }

        var tessRegions = tessResult ? normalizeTesseract(tessResult) : [];
        var glmRegions = glmResult ? normalizeGlmOcr(glmResult) : [];

        var merged = mergeRegions(tessRegions, glmRegions);

        return {
            regions: merged.regions,
            tessResult: tessResult,
            glmResult: glmResult,
            mergeStats: merged.stats,
        };
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

// ── PDF page renderer (pdf.js, lazy-imported) ─────────────────────────

/**
 * Render a single PDF page to a canvas via pdf.js.
 * @param {Uint8Array} pdfBytes
 * @param {number} pageNum — 1-indexed page number
 * @param {number} [scale=2.0] — render scale (higher = better OCR quality)
 * @returns {Promise<{canvas:HTMLCanvasElement, width:number, height:number, originalWidth:number, originalHeight:number}>}
 */
async function renderPdfPage(pdfBytes, pageNum, scale) {
    scale = scale || 2.0;
    var { getDocument } = await import('pdfjs-dist/build/pdf.mjs');
    var pdf = await getDocument({ data: pdfBytes, disableWorker: true }).promise;
    var page = await pdf.getPage(pageNum);
    var viewport = page.getViewport({ scale: scale });
    var origView = page.getViewport({ scale: 1.0 });
    var canvas = document.createElement('canvas');
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    var ctx = canvas.getContext('2d');
    await page.render({ canvasContext: ctx, viewport: viewport }).promise;
    return {
        canvas: canvas,
        width: viewport.width,
        height: viewport.height,
        originalWidth: origView.width,
        originalHeight: origView.height,
    };
}

/** Convert a canvas to Uint8Array (PNG). */
async function canvasToBytes(canvas) {
    var blob = await new Promise(function (resolve) {
        return canvas.toBlob(resolve, 'image/png');
    });
    return new Uint8Array(await blob.arrayBuffer());
}
