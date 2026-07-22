// ── Merge overlapping regions from multiple OCR engines ───────────────
//
// mergeRegions(tessRegions, glmRegions): Region[] → { regions, stats }
//
// Strategy: Tesseract is the trusted baseline. GLM regions are compared
// against Tesseract regions by IoU. Overlapping pairs (IoU > 0.5) are
// resolved to the higher-confidence candidate. Non-overlapping GLM
// regions are appended as novel detections.
//
// Merge decisions are recorded in the winner's mergeDecision field for
// full provenance audit.

import { RegionSchema } from './schemas.js';

const IOU_THRESHOLD = 0.5;

/**
 * Merge Tesseract and GLM-OCR regions into a single deduplicated list.
 */
export function mergeRegions(tessRegions, glmRegions) {
    const stats = {
        tesseractRegions: tessRegions.length,
        glmRegions: glmRegions.length,
        mergedPairs: 0,
        finalRegions: 0,
    };

    // Start with all Tesseract regions (trusted baseline).
    const regions = tessRegions.map(function (r) { return { ...r }; });

    // Track which Tesseract region indices have been replaced by GLM winners.
    const replacedIndices = new Set();

    for (var gi = 0; gi < glmRegions.length; gi++) {
        var glm = glmRegions[gi];

        // Find the best-overlapping Tesseract region
        var bestIdx = -1;
        var bestIou = 0;

        for (var ti = 0; ti < tessRegions.length; ti++) {
            var iouVal = iou(glm.bbox, tessRegions[ti].bbox);
            if (iouVal > bestIou) {
                bestIou = iouVal;
                bestIdx = ti;
            }
        }

        if (bestIou >= IOU_THRESHOLD && bestIdx >= 0) {
            var tess = tessRegions[bestIdx];

            if (!replacedIndices.has(bestIdx)) {
                stats.mergedPairs++;
            }

            if (glm.confidence > tess.confidence) {
                // GLM wins — replace the Tesseract region
                var decision = {
                    won: 'glm-ocr',
                    reason: makeReason('glm-ocr', glm.confidence, tess.confidence, bestIou),
                    loserId: tess.id,
                    loserConfidence: tess.confidence,
                    iou: bestIou,
                };

                regions[bestIdx] = RegionSchema.parse({
                    id: glm.id,
                    text: glm.text,
                    bbox: glm.bbox,
                    confidence: glm.confidence,
                    source: 'merged',
                    sourceDetail: glm.sourceDetail,
                    mergeDecision: decision,
                });

                replacedIndices.add(bestIdx);
            } else {
                // Tesseract wins — enrich it with merge decision
                if (!regions[bestIdx].mergeDecision) {
                    var tDecision = {
                        won: 'tesseract',
                        reason: makeReason('tesseract', tess.confidence, glm.confidence, bestIou),
                        loserId: glm.id,
                        loserConfidence: glm.confidence,
                        iou: bestIou,
                    };

                    regions[bestIdx] = RegionSchema.parse({
                        id: tess.id,
                        text: tess.text,
                        bbox: tess.bbox,
                        confidence: tess.confidence,
                        source: 'merged',
                        sourceDetail: tess.sourceDetail,
                        mergeDecision: tDecision,
                    });
                }
            }
        } else {
            // No overlap — GLM region is novel
            regions.push({ ...glm });
        }
    }

    stats.finalRegions = regions.length;

    return { regions: regions, stats: stats };
}

// ── Helpers ────────────────────────────────────────────────────────────

function iou(a, b) {
    var ix0 = Math.max(a.x0, b.x0);
    var iy0 = Math.max(a.y0, b.y0);
    var ix1 = Math.min(a.x1, b.x1);
    var iy1 = Math.min(a.y1, b.y1);

    if (ix0 >= ix1 || iy0 >= iy1) return 0;

    var inter = (ix1 - ix0) * (iy1 - iy0);
    var areaA = (a.x1 - a.x0) * (a.y1 - a.y0);
    var areaB = (b.x1 - b.x0) * (b.y1 - b.y0);

    return inter / (areaA + areaB - inter);
}

function makeReason(winner, winnerConf, loserConf, iouVal) {
    var iouPct = Math.round(iouVal * 100);
    var wPct = winnerConf.toFixed(1);
    var lPct = loserConf.toFixed(1);
    return winner + ' (' + wPct + ') over ' + (loserConf === winnerConf ? 'equal' : lPct) + ', IoU=' + iouPct + '%';
}
