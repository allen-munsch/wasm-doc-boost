// ── Four-layer Zod schemas for unified document analysis ──────────────
//
// Layer 0: Raw backend outputs (Tesseract, GLM-OCR, WASM classification, PII)
// Layer 1: Unified Region (engine-agnostic document token)
// Layer 2: FieldAnnotation (user-added metadata on regions)
// Layer 3: EnrichedDocumentData (the full analysis result)
//
// Import via import map: import { z } from 'zod';

import { z } from 'zod';

// ═══════════════════════════════════════════════════════════════════════
// Primitives
// ═══════════════════════════════════════════════════════════════════════

export const BboxSchema = z.object({
    x0: z.number(),
    y0: z.number(),
    x1: z.number(),
    y1: z.number(),
});

export const BaselineSchema = BboxSchema.extend({
    has_baseline: z.boolean(),
});

export const RowAttributesSchema = z.object({
    ascenders: z.number(),
    descenders: z.number(),
    row_height: z.number(),
});

export const RectangleSchema = z.object({
    left: z.number(),
    top: z.number(),
    width: z.number(),
    height: z.number(),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 0 — Tesseract.js v6 full type tree
// ═══════════════════════════════════════════════════════════════════════

const ChoiceSchema = z.object({
    text: z.string(),
    confidence: z.number(),
});

export const SymbolSchema = z.object({
    text: z.string(),
    confidence: z.number(),
    bbox: BboxSchema,
    baseline: BaselineSchema,
    choices: z.array(ChoiceSchema),
    is_superscript: z.boolean(),
    is_subscript: z.boolean(),
    is_dropcap: z.boolean(),
});

export const WordSchema = z.lazy(() =>
    z.object({
        text: z.string(),
        confidence: z.number(),
        bbox: BboxSchema,
        baseline: BaselineSchema,
        symbols: z.array(SymbolSchema),
        choices: z.array(ChoiceSchema),
        font_name: z.string(),
        language: z.string(),
        direction: z.string(),
        is_numeric: z.boolean().nullable(),
        in_dictionary: z.boolean().nullable(),
    })
);

export const LineSchema = z.lazy(() =>
    z.object({
        text: z.string(),
        confidence: z.number(),
        bbox: BboxSchema,
        baseline: BaselineSchema,
        words: z.array(WordSchema),
        rowAttributes: RowAttributesSchema,
    })
);

export const ParagraphSchema = z.lazy(() =>
    z.object({
        text: z.string(),
        confidence: z.number(),
        bbox: BboxSchema,
        baseline: BaselineSchema,
        lines: z.array(LineSchema),
        is_ltr: z.boolean(),
    })
);

export const BlockSchema = z.lazy(() =>
    z.object({
        text: z.string(),
        confidence: z.number(),
        bbox: BboxSchema,
        baseline: BaselineSchema,
        paragraphs: z.array(ParagraphSchema),
        blocktype: z.string(),
        page: z.object({}).passthrough(), // cyclic Page ref — accept any object
    })
);

export const PageSchema = z.object({
    blocks: z.array(BlockSchema).nullable(),
    text: z.string(),
    confidence: z.number(),
    oem: z.string(),
    osd: z.string(),
    psm: z.string(),
    version: z.string(),
    hocr: z.string().nullable(),
    tsv: z.string().nullable(),
    box: z.string().nullable(),
    unlv: z.string().nullable(),
    sd: z.string().nullable(),
    imageColor: z.string().nullable(),
    imageGrey: z.string().nullable(),
    imageBinary: z.string().nullable(),
    rotateRadians: z.number().nullable(),
    pdf: z.array(z.number()).nullable(),
    debug: z.string().nullable(),
});

export const RecognizeResultSchema = z.object({
    jobId: z.string(),
    data: PageSchema,
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 0 — GLM-OCR backend response (regions mode)
// ═══════════════════════════════════════════════════════════════════════

// GLM bbox is [x1, y1, x2, y2] — tuple, different from Tesseract's {x0,y0,x1,y1}
const GlmBboxTuple = z.tuple([z.number(), z.number(), z.number(), z.number()]);

const GlmRegionSchema = z.object({
    text: z.string(),
    bbox: GlmBboxTuple,
});

export const GlmOcrResultSchema = z.object({
    regions: z.array(GlmRegionSchema),
    _meta: z.object({}).passthrough().optional(),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 0 — WASM classification output
// ═══════════════════════════════════════════════════════════════════════

export const ClassifySchema = z.object({
    is_document: z.number().min(0).max(1),
    is_digital: z.number().min(0).max(1),
    is_paper: z.number().min(0).max(1),
    is_crumpled: z.number().min(0).max(1),
    is_shadow: z.number().min(0).max(1),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 0 — PII hit (from wasm-bridge scan_pii)
// ═══════════════════════════════════════════════════════════════════════

export const PiiHitSchema = z.object({
    kind: z.enum([
        'PAN', 'SSN', 'PHONE', 'EMAIL', 'CVV', 'EXPIRY',
        'ROUTING', 'ACCOUNT', 'DOB', 'ADDRESS', 'NAME', 'ZIP',
    ]),
    text: z.string(),
    start: z.number().int().nonnegative(),
    end: z.number().int().nonnegative(),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 1 — Unified Region (the core abstraction)
// ═══════════════════════════════════════════════════════════════════════

export const MergeDecisionSchema = z.object({
    won: z.enum(['tesseract', 'glm-ocr']),
    reason: z.string(),
    loserId: z.string(),
    loserConfidence: z.number(),
    iou: z.number(),
});

export const RegionSchema = z.object({
    // Stable ID derived from source hierarchy path (e.g. "tess-b0-p1-l2-w3")
    // or source index (e.g. "glm-5")
    id: z.string(),

    text: z.string(),
    bbox: BboxSchema,
    confidence: z.number().min(0).max(100),

    source: z.enum(['tesseract', 'glm-ocr', 'merged']),

    sourceDetail: z.object({
        // Tesseract-specific (null for GLM)
        blockType: z.string().nullable(),
        level: z.enum(['word']).nullable(),
        parentId: z.string().nullable(),

        // GLM-specific (null for Tesseract)
        regionIndex: z.number().int().nullable(),
    }),

    // Only set when source === 'merged'
    mergeDecision: MergeDecisionSchema.nullable(),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — FieldAnnotation (user-added metadata)
// ═══════════════════════════════════════════════════════════════════════

export const FieldAnnotationSchema = z.object({
    id: z.string(),
    regionId: z.string(),
    fieldType: z.string(),
    confidence: z.number().min(0).max(1),
    manual: z.boolean(),
    timestamp: z.string(),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — Provenance
// ═══════════════════════════════════════════════════════════════════════

export const ProvenanceSchema = z.object({
    engine: z.string(),
    tesseract: z
        .object({
            version: z.string(),
            oem: z.string(),
            psm: z.string(),
        })
        .nullable(),
    glmOcr: z
        .object({
            backend: z.string(),
        })
        .nullable(),
    analysisMs: z.number().int().nonnegative(),
    mergeStats: z.object({
        tesseractRegions: z.number().int().nonnegative(),
        glmRegions: z.number().int().nonnegative(),
        mergedPairs: z.number().int().nonnegative(),
        finalRegions: z.number().int().nonnegative(),
    }),
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 3 — EnrichedDocumentData (the full analysis result)
// ═══════════════════════════════════════════════════════════════════════

export const ImageInfoSchema = z.object({
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    mime: z.string(),
});

export const EnrichedDocumentDataSchema = z.object({
    id: z.string(),
    timestamp: z.string(),
    image: ImageInfoSchema,
    classification: ClassifySchema.nullable(),
    regions: z.array(RegionSchema),
    pii: z.array(PiiHitSchema),
    fields: z.array(FieldAnnotationSchema),
    redactionIds: z.array(z.string()),
    provenance: ProvenanceSchema,
});
