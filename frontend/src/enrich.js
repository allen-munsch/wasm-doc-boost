// ── EnrichedDocument: factory, lenses, and curried mutators ──────────
//
// createEnrichedDocument({image, classification, regions, pii, provenance})
//   → plain data object (immutable surface, hidden _regionIndex)
//
// Lenses (pure accessors):
//   regions(doc) → Region[]
//   redactions(doc) → Region[]
//   classification(doc), provenance(doc), fields(doc), region(doc, id)
//
// Curried mutations (data-last, return new doc):
//   addFieldAnnotation(regionId, fieldType, conf)(doc)
//   redactRegion(regionId)(doc)
//   unredactRegion(regionId)(doc)
//
// Serialization:
//   toJSON(doc) → plain object (strips _regionIndex)
//   fromJSON(json) → doc (rebuilds _regionIndex)

import { EnrichedDocumentDataSchema } from './schemas.js';

// ═══════════════════════════════════════════════════════════════════════
// Factory
// ═══════════════════════════════════════════════════════════════════════

export function createEnrichedDocument(params) {
    var data = EnrichedDocumentDataSchema.parse({
        id: crypto.randomUUID ? crypto.randomUUID() : fallbackUUID(),
        timestamp: new Date().toISOString(),
        image: params.image,
        classification: params.classification || null,
        regions: params.regions || [],
        pii: params.pii || [],
        fields: [],
        redactionIds: [],
        provenance: params.provenance,
    });

    // Hidden index for O(1) region lookups — stripped by toJSON()
    data._regionIndex = buildIndex(data.regions);

    return data;
}

// ═══════════════════════════════════════════════════════════════════════
// Lenses — pure accessors
// ═══════════════════════════════════════════════════════════════════════

export function regions(doc) { return doc.regions; }
export function classification(doc) { return doc.classification; }
export function provenance(doc) { return doc.provenance; }
export function fields(doc) { return doc.fields; }
export function image(doc) { return doc.image; }
export function docId(doc) { return doc.id; }

/** O(1) lookup by region id */
export function region(doc, id) {
    return doc._regionIndex.get(id) || null;
}

/** Filter regions to only redacted ones */
export function redactions(doc) {
    var ids = new Set(doc.redactionIds);
    return doc.regions.filter(function (r) { return ids.has(r.id); });
}

// ═══════════════════════════════════════════════════════════════════════
// Curried mutations — data-last for pipe/compose
// ═══════════════════════════════════════════════════════════════════════

/**
 * Add a user field annotation to a region.
 * Returns a new doc (does not mutate the input).
 *
 * Usage: doc = addFieldAnnotation('r3', 'invoice_number', 0.9)(doc);
 */
export function addFieldAnnotation(regionId, fieldType, confidence) {
    var conf = typeof confidence === 'number' ? confidence : 0.9;
    return function (doc) {
        if (!doc._regionIndex.has(regionId)) {
            throw new Error('Region ' + regionId + ' not found');
        }

        var ann = {
            id: crypto.randomUUID ? crypto.randomUUID() : fallbackUUID(),
            regionId: regionId,
            fieldType: fieldType,
            confidence: conf,
            manual: true,
            timestamp: new Date().toISOString(),
        };

        return replace(doc, { fields: doc.fields.concat([ann]) });
    };
}

/**
 * Remove a field annotation by id.
 * Usage: doc = removeFieldAnnotation('ann-123')(doc);
 */
export function removeFieldAnnotation(annotationId) {
    return function (doc) {
        return replace(doc, {
            fields: doc.fields.filter(function (f) { return f.id !== annotationId; }),
        });
    };
}

/**
 * Mark a region as redacted.
 * Idempotent — no-op if already redacted.
 * Usage: doc = redactRegion('r3')(doc);
 */
export function redactRegion(regionId) {
    return function (doc) {
        if (!doc._regionIndex.has(regionId)) {
            throw new Error('Region ' + regionId + ' not found');
        }
        if (doc.redactionIds.indexOf(regionId) !== -1) return doc;

        return replace(doc, { redactionIds: doc.redactionIds.concat([regionId]) });
    };
}

/**
 * Un-redact a region.
 * Usage: doc = unredactRegion('r3')(doc);
 */
export function unredactRegion(regionId) {
    return function (doc) {
        return replace(doc, {
            redactionIds: doc.redactionIds.filter(function (id) { return id !== regionId; }),
        });
    };
}

// ═══════════════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════════════

/**
 * Serialize to plain JSON-safe object.
 * Strips _regionIndex (internal-only, rebuilt by fromJSON).
 */
export function toJSON(doc) {
    var keys = Object.keys(doc);
    var out = {};
    for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        if (k === '_regionIndex') continue;
        out[k] = doc[k];
    }
    return EnrichedDocumentDataSchema.parse(out);
}

/**
 * Deserialize from JSON (rebuilds _regionIndex).
 */
export function fromJSON(json) {
    var data = EnrichedDocumentDataSchema.parse(json);
    data._regionIndex = buildIndex(data.regions);
    return data;
}

// ═══════════════════════════════════════════════════════════════════════
// Convenience: full build from normalized inputs
// ═══════════════════════════════════════════════════════════════════════

/**
 * Build an EnrichedDocument from the raw outputs of all pipelines.
 * Handles merge internally via the passed-in regions (should be
 * already merged by the caller or by docboost's analyze flow).
 */
export function buildEnrichedDocument(params) {
    return createEnrichedDocument({
        image: params.image,
        classification: params.classification,
        regions: params.regions || [],
        pii: params.pii || [],
        provenance: {
            engine: 'wasm-doc-boost 0.1.0',
            tesseract: params.tesseract
                ? {
                      version: params.tesseract.data.version,
                      oem: params.tesseract.data.oem,
                      psm: params.tesseract.data.psm,
                  }
                : null,
            glmOcr: params.glmOcr
                ? { backend: params.glmOcr._meta?.engine || 'GLM-OCR' }
                : null,
            analysisMs: params.analysisMs || 0,
            mergeStats: params.mergeStats || {
                tesseractRegions: 0,
                glmRegions: 0,
                mergedPairs: 0,
                finalRegions: 0,
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

function buildIndex(regionList) {
    var m = new Map();
    for (var i = 0; i < regionList.length; i++) {
        m.set(regionList[i].id, regionList[i]);
    }
    return m;
}

/**
 * Replace fields on a doc, re-validating the result.
 * Preserves _regionIndex (it never changes after creation).
 */
function replace(doc, overrides) {
    var keys = Object.keys(overrides);
    var next = {};

    // Copy all existing keys
    var docKeys = Object.keys(doc);
    for (var i = 0; i < docKeys.length; i++) {
        var k = docKeys[i];
        next[k] = doc[k];
    }

    // Apply overrides
    for (var j = 0; j < keys.length; j++) {
        var ok = keys[j];
        next[ok] = overrides[ok];
    }

    // Re-validate (also strips any stray keys)
    var validated = EnrichedDocumentDataSchema.parse(next);
    validated._regionIndex = doc._regionIndex;
    return validated;
}

function fallbackUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0;
        var v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
