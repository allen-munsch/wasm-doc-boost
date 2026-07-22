// ── Normalize backend-specific outputs into unified Region[] ──────────
//
// normalizeTesseract:  Tesseract.js v6 RecognizeResult → Region[]
// normalizeGlmOcr:     GLM-OCR backend /ocr response → Region[]
//
// Both produce the same Region shape — they can be merged downstream.

import { RecognizeResultSchema, GlmOcrResultSchema, RegionSchema } from './schemas.js';

// ── Tesseract normalization ────────────────────────────────────────────

const VALID_BLOCK_TYPES = new Set([
    'PT_UNKNOWN', 'PT_FLOWING_TEXT', 'PT_HEADING_TEXT', 'PT_PULLOUT_TEXT',
    'PT_EQUATION', 'PT_INLINE_EQUATION', 'PT_TABLE', 'PT_VERTICAL_TEXT',
    'PT_CAPTION_TEXT', 'PT_FLOWING_IMAGE', 'PT_HEADING_IMAGE',
    'PT_PULLOUT_IMAGE', 'PT_HORZ_LINE', 'PT_VERT_LINE', 'PT_NOISE', 'PT_COUNT',
]);

/**
 * Flatten Tesseract's block→paragraph→line→word tree into a flat Region[].
 *
 * Each word-level region gets:
 *   - id: "tess-b{bi}-p{pi}-l{li}-w{wi}"
 *   - parentId: "tess-b{bi}-p{pi}-l{li}" (its line)
 *   - source: 'tesseract'
 *   - sourceDetail.level: 'word', blockType, parentId, regionIndex: null
 *
 * Only words are emitted (the most useful granularity for bbox rendering
 * and redaction). Non-text blocks (images, lines, noise) are skipped.
 */
export function normalizeTesseract(rawResult) {
    const parsed = RecognizeResultSchema.parse(rawResult);
    const page = parsed.data;

    if (!page.blocks) return [];

    const regions = [];

    for (let bi = 0; bi < page.blocks.length; bi++) {
        const block = page.blocks[bi];
        const blockType = VALID_BLOCK_TYPES.has(block.blocktype)
            ? block.blocktype
            : 'PT_UNKNOWN';

        // Skip non-text blocks entirely
        if (!block.paragraphs || block.paragraphs.length === 0) continue;

        for (let pi = 0; pi < block.paragraphs.length; pi++) {
            const para = block.paragraphs[pi];

            if (!para.lines || para.lines.length === 0) continue;

            for (let li = 0; li < para.lines.length; li++) {
                const line = para.lines[li];

                if (!line.words || line.words.length === 0) continue;

                const lineId = `tess-b${bi}-p${pi}-l${li}`;

                for (let wi = 0; wi < line.words.length; wi++) {
                    const word = line.words[wi];

                    // Skip empty words
                    if (!word.text || word.text.trim().length === 0) continue;

                    const region = {
                        id: `${lineId}-w${wi}`,
                        text: word.text,
                        bbox: {
                            x0: word.bbox.x0,
                            y0: word.bbox.y0,
                            x1: word.bbox.x1,
                            y1: word.bbox.y1,
                        },
                        confidence: clamp(word.confidence, 0, 100),
                        source: 'tesseract',
                        sourceDetail: {
                            blockType,
                            level: 'word',
                            parentId: lineId,
                            regionIndex: null,
                        },
                        mergeDecision: null,
                    };

                    regions.push(RegionSchema.parse(region));
                }
            }
        }
    }

    return regions;
}

// ── GLM-OCR normalization ──────────────────────────────────────────────

/**
 * GLM-OCR backend default per-region confidence.
 *
 * The backend does not return confidence scores per crop.
 * We assign a moderate default so that, during merge, Tesseract
 * regions with known confidence can override when they overlap.
 */
const GLM_DEFAULT_CONFIDENCE = 75.0;

/**
 * Convert GLM-OCR backend /ocr response into unified Region[].
 *
 * GLM bbox format: [x1, y1, x2, y2] (top-left, bottom-right tuple).
 * Converted to {x0, y0, x1, y1} for unified BboxSchema.
 *
 * Each region gets:
 *   - id: "glm-{i}"
 *   - source: 'glm-ocr'
 *   - sourceDetail.regionIndex: i
 */
export function normalizeGlmOcr(rawResult) {
    const parsed = GlmOcrResultSchema.parse(rawResult);

    if (!parsed.regions || parsed.regions.length === 0) return [];

    return parsed.regions.map((r, i) => {
        const [x1, y1, x2, y2] = r.bbox;

        return RegionSchema.parse({
            id: `glm-${i}`,
            text: r.text,
            bbox: { x0: x1, y0: y1, x1: x2, y1: y2 },
            confidence: GLM_DEFAULT_CONFIDENCE,
            source: 'glm-ocr',
            sourceDetail: {
                blockType: null,
                level: null,
                parentId: null,
                regionIndex: i,
            },
            mergeDecision: null,
        });
    });
}

// ── Helpers ────────────────────────────────────────────────────────────

function clamp(v, lo, hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
