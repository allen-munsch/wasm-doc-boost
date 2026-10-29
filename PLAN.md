# wasm-doc-boost — Per-Degree Rotation Rework: Plan

Status: drafting (accumulated from live investigation of the repo + literature).
Evidence: crates/features-core/src/*.rs, crates/wasm-bridge/src/gbdt.rs,
scripts/train_model.py, scripts/export_worker.py, frontend/src/docboost.js,
frontend/src/schemas.js, .dirge/skills/model-json-pipeline/SKILL.md.

## Goal

Replace the 4-way rotation gate (rotation_0/90/180/270) with a per-degree
estimator: `rotation_deg` in [0,360), plus `rotation_confidence` and a
`needs_review` flag. Document images only (gated on is_document; a COCO photo
has no "upright").

Targets (proposed, to validate in Phase 3):
- strict circular MAE: under ~3 deg on documents; +/-1 deg accuracy >= 0.95.
- axis-only (180-tolerant) accuracy: >= 0.999.
- accepted-flip rate (strict error > 90 deg that passes review): <= 0.1%, goal 0.
- digital receipts (FATURA2): +/-1 deg -> 0.999 goal.

## Architecture (REVISED after Phase 1 benchmark)

Three stages — matching the literature's division of labour (fine angle vs
flip) and docangle's pipeline:

1. AXIS (fine angle mod 180): DETERMINISTIC Radon/projection response curve
   (60 bins at 3 deg), argmax refined parabolically -> axis in [0,180).
   Phase 1 benchmark: 0.335 deg circular MAE on real invoices — no GBDT
   head needed. (A 180-way softmax was considered and DROPPED: too many
   trees for WASM; the deterministic radon already beats the 3-deg target.)
2. FLIP (0 vs 180): GBDT head on textline ascender/descender ratio +
   line-vote fraction + body-mass fraction -> P(flip).
3. TRUST: GBDT head predicting P(radon axis error <= 1 deg) from radon
   peakiness/margin/concentration, n_lines, body_mass_frac, boundary_visible
   and the 5 doc traits -> rotation_confidence.

COMBINE (deterministic): rotation_deg = (axis + 180*flip) mod 360;
rotation_confidence = trust * flip-calibration; needs_review when
confidence < tau (tau chosen from data) or not is_document.

Total GBDT labels: 5 doc traits + flip + trust = 7 (~900 trees at the
current ~129/label budget). The 5 document-trait heads are unchanged.

## The flip head on a curve (operating point from data)

Do NOT hardcode the flip as a binary gate. The flip head emits a continuous
flip_score in [-1,1] (signed ascender/descender evidence) and a calibrated
P(flip); the axis head emits its softmax mass and a deterministic tilt
residual. Combine into one confidence curve.

Calibration data already exists: features_v3.npz holds 4 rotations
(0/90/180/270) of every base image (74,944 rows = N x 4). The 0 vs 180 pair
of the same base image is exactly the flip problem, so the flip curve can be
calibrated TODAY, before Phase 2's K-angle data lands (that data is for the
per-degree axis head, not the flip).

Procedure (scripted, artifact -> report.json):
1. Split by base filename: train / cal / test (GroupShuffleSplit, as in
   train_model.py, so an image's 0 and 180 rotations never straddle the split).
2. Train both heads on train. On cal, compute flip_score + axis softmax mass
   + tilt residual per doc; combine into a single confidence curve.
3. Calibrate: isotonic/Platt map raw flip_score -> honest P(flip wrong).
4. Sweep tau densely over the curve. For each tau record: accepted-flip rate
   = P(flip wrong | confidence >= tau), review rate = P(confidence < tau),
   attempt coverage = P(confidence >= tau), and axis-MAE on attempted docs.
5. Pick tau = largest threshold with accepted-flip <= 0.1% (goal 0) while
   review rate stays a few %; if the curve never reaches 0.1% at useful
   coverage, that is the data saying the GBDT flip head is not enough ->
   trigger Path B (quantized vision encoder). The curve decides before we ship.
6. Repeat per source (receipts vs scans) to decide global vs per-source tau.
Pin tau as a constant in model.json v2 manifest, ship the sweep table with it
for auditability. Reference mechanism: docangle's needs_review at conf<0.90.

## Signals to add (features-core only; py-features and wasm-bridge delegate)

- projection.rs: radon_response() — 60-bin @3deg response curve + best_axis
  (parabolic) + peakiness + flip_residual (curve[t] - curve[t+180]).
- textline.rs (new): morphological scale-space (Bloomberg/Leptonica style)
  -> baseline/x-height per line -> ascender/descender mass ratio (mean +
  line-vote fraction); blob opening direction (Aradhye 2005); run-transition
  asymmetry (Lu 2006).
- ink.rs: unfold docstrum — raw circular mean (no double-angle fold) + a
  pi-ambiguity feature.
- boundary.rs: minAreaRect axis vote — fit the minimum-area rotated rectangle
  to the is_document mask/contour (AADD/ArbDR's inference axis-align step);
  long-edge angle is a second axis estimate. ms-fast, near-perfect on bounded
  receipts/invoices, useless on full-bleed crops -> complement of
  Radon/Fourier, so fuse as a vote only when the boundary is visible.
- Phase 1 benchmark: Radon vs 2-D Fourier tilt vs Hough vs minAreaRect
  boundary for the axis; pick by synthetic-stripe test + real-doc MAE, keep
  minAreaRect as a fused vote.

## Phases

Phase 0 — DONE: leak fix (export_worker.py) — median_border_color() fills the
  regions exposed by rotation with the document's own border background
  instead of black; verified 17-deg corners went (0,0,0) -> (255,255,255)
  and 90-deg cardinal rotation is byte-identical. jitter + per-degree labels
  moved to Phase 2 (they belong with the K-angle data generation).
Phase 1 — signals (features-core): radon DONE (crates/features-core/src/radon.rs,
  radon_curve 60-bin @3deg + parabolic refine + radon_features summary;
  synthetic tests incl. stripes@17deg and 180-flip-blindness), textline flip
  extractor DONE (textline.rs: deskew by axis -> line bands -> x-height from
  binned mode of column ink heights -> asc/desc mass; 6 features incl. signed
  flip_score and line-vote fraction; tests upright vs flipped and all-caps
  neutrality), boundary minAreaRect DONE (boundary.rs: median border bg ->
  page mask -> largest component -> monotone-chain hull -> rotating calipers;
  4 features incl. boundary_visible; tests 15deg synthetic page and
  full-bleed), benchmark estimators added to radon.rs (fourier_tilt 32x32
  direct DFT polar histogram; hough_axis 96x96 ink votes peak-cell). All 56
  unit tests pass; kept OUT of extract_all until retrain (103-feature
  contract). Phase 1 benchmark in crates/features-core/tests/orientation_bench.rs
  (real fatura2 fixtures, data/bench_pgms). KEY PITFALL: fixture rotation
  MUST fill with the median border color (Phase 0 fix) — plain PIL rotate
  leaves black corner triangles that dominate the projection variance and
  give ~40deg radon MAE; with white fill radon is sub-degree on invoices.
  BENCHMARK DECISION (3 fatura2 docs x 18 angles, white-filled): radon
  circular MAE 0.335 deg — wins decisively; minAreaRect 36.9 deg (visible on
  36/54, white-born-digital invoices have no boundary); Fourier tilt 38.1 deg
  and Hough 66.3 deg were also tried and DROPPED from radon.rs. Ground truth
  is (180 - a) mod 180 because PIL CCW rotate = -a in image coords. NEXT:
  Phase 2 K-angle data (K random angles/base, background-filled), then Phase 3.
Phase 2 — data: K=8 angles/base = 4 cardinal (exact 0/180 + 90/270 flip
  pairs) + 4 seeded-random in [0,360); export_worker.py --v2 + train_model_v2.py
  written; py-features extract_all_v2 (116 feats) + ocr-core ort
  default-features=false fix (openssl-sys was pulling native-tls via ort's
  default download-binaries). KEY FINDINGS from tiny-export validation:
  (a) radon MUST pad to a max(w,h) square (border-fill) before the curve — a
  landscape canvas cannot hold its 90-deg rotation (crop-to-strip) and a
  centre crop drops portrait text lines; after the pad fix fatura2 is
  sub-degree at ALL angles incl. 90/270 (err 0.07-0.45). (b) photo receipts
  (cord-v2 ~25%, srd) have border/vertical dominance -> radon 90-deg off;
  trust=0 there (correct: needs_review). (c) photo-receipt base images have
  INTRINSIC tilt, so rotation_deg vs the augmentation angle is ill-defined
  for them — the per-degree target is well-posed for born-digital docs
  (fatura2) and the trust head must gate the rest.
Phase 2b — axis fix (DONE, from Phase 3 run's failure analysis): the v2
  training run showed fatura2 axis only 92.3% within 1 deg — the aggregate
  was worse (59%) because cord-v2/srd/variants (photo receipts / crumpled /
  shadow) are off-by-90 57-65% (border/vertical dominance, intrinsic tilt)
  and were never meant to auto-rotate. Root cause of the fatura2 residual:
  ~9% of base docs have a Radon NEAR-TIE (peak_margin 5.4 vs 64.2 for good
  docs) between the text axis and a perpendicular structure (table rules);
  the bare argmax lands 90 deg off, and neither boundary long-edge nor
  docstrum rescues it (boundary ~50% coin-flip, docstrum MAE 42 deg). FIX
  (deterministic, zero new trees):
  1. `parabolic_refine` now WRAPS at the 0/180 fold (was returning the
     unrefined bin for i==0/59 — a ~1.5 deg quantization error).
  2. `radon_top_axes(curve, k)` returns the top-k parabolically-refined
     local maxima.
  3. `rotation_features` picks the axis by scoring each candidate with
     `textline_line_quality` (n_lines, then total line ink) instead of the
     bare argmax: at the right axis text lines are separate bands, at the
     wrong axis they merge into one column mass.
  VALIDATED on held-out test split (real Rust extract_all_v2, re-rotated
  fixtures): fatura2 off-90 (459) 100% within 1 deg (MAE 0.24), good (400)
  99.8% within 1 deg (MAE 0.22, no regression), scattered (170) 80.6%
  within 1 deg / 95.3% within 2 deg (MAE 1.5). Projected fatura2 axis:
  ~99.5% within 1 deg, ~0.25 deg MAE (was 92.3% / 5.9 deg). cord-v2/srd/
  variants stay bad (correct — route to review via the trust head).
  58 features-core unit tests pass incl. new wrap/top-axes/line-quality
  guards. NEXT: subset re-export + retrain to confirm the flip head
  improves on the now-clean axis, then the full 19K-base re-export.

Phase 3 — train: 5 trait heads + flip + trust (7 labels, ~900 trees);
  train_model_v2.py with GroupShuffleSplit by base filename + circular-MAE
  report + tau sweep (accepted-flip <= 0.1%, review <= 20%). TODO after
  smoke test: full K-angle export (~19K base x 8), train, calibrate tau.
Phase 4 — Rust/WASM: gbdt.rs manifest + softmax group + confidence +
  needs_review; LABELS update; rotation reproduction test (retires the
  SKILL pitfall). DONE except real-model reproduction test: gbdt.rs
  ModelManifest {labels,tau,trees} + from_json; lib.rs load_model_v2 +
  classify_file_v2 + combine_rotation (formula parity unit-tested);
  train_model_v2.py writes the manifest with pinned tau + JSON-safe report.
Phase 5 — frontend: ClassifySchema (rotation_deg, rotation_confidence,
  needs_review); docboost.js rotates by -rotation_deg before OCR only when
  confidence >= tau. DONE (schemas.js + docboost.js + eval_worker.mjs v2
  switch; not runnable in this shell — verified in Phase 6).
Phase 6 — release: wasm-pack build, eval_wasm/eval_pdf/bench_sobel, README
  metrics (circular MAE, +/-1 deg, tree count, sizes), tag + release.

## Research references

- van Beusekom, Shafait, Breuel 2010 (IJDAR): geometric text-line model;
  99% UW-I / 100% OCRopus.
- Bloomberg et al. 1995 (Leptonica): morphological asc/desc ratio;
  pixOrientDetect is the reference implementation.
- Le et al. 1994; Caprari 1999; Aradhye 2005 (opening direction);
  Lu et al. 2006 (run transitions); Avila & Lins (x-height + baseline).
- docangle (KazimMammadli, 2026, MIT code / research-use weights): Fourier
  tilt + two quadrant voters + 360-deg fallback + needs_review at conf<0.90;
  0.8% wrong-quadrant on any angle, accepted flips ~0, 35-45ms CPU, 36MB ONNX.
  Borrow the operating-point mechanism, not the models.
- Goswami et al. 2025/2026 "Seeing Straight" (arXiv 2511.04161, v2
  2026-03): 4-class rotation head on the Phi-3.5-Vision *vision encoder*
  (not the full VLM), dynamic image cropping; 96% ORB-En / 92% ORB-Indic.
  Vision-encoder scale — out of our WASM GBDT budget, but validates Path B
  (quantized vision-encoder quadrant head) and crop-then-classify.
- AADD / ArbDR (Wang et al. 2026 AAAI / TMM, chaoyunwang): dewarping, but its
  inference stage does "axis alignment preprocessing" — rotate the doc to its
  principal axes (min-area rotated rectangle from the doc region) before the
  heavy model. Transfers: (1) axis-align-first validates our stage split;
  (2) minAreaRect on is_document is a fast, borrowable axis signal.
- Awesome-Document-Image-Rectification (fh2019ustc): read in full — it is a
  deep-dewarping list (DocUNet..BookNet), no orientation work, nothing CPU/
  WASM-cheap; keep only as the future upstream for perspective rectification.

## Open decisions

1. Axis signal: Radon curve vs Fourier tilt vs both (Phase 1 benchmark).
2. Flip signal: asc/desc vs opening direction vs both (Phase 3 ablation).
3. tau operating point: global vs per-source (receipts stricter).
4. Tree/size budget: axis 180-way n_rounds 12-16; measure gzip <= ~1.5MB.
5. Escalation Path B: quantized MobileNetV3 quadrant head in WASM if the
   GBDT flip head stalls below 0.95.

## Conventions

- rotation_deg = degrees clockwise from upright, [0,360); correction =
  -rotation_deg. (docangle uses CCW [-180,180]; we keep ours, pinned by a
  unit test.)
- flip = strict error > 90 deg; axis-only = 180-tolerant error.
- no_std gotcha (features-core): f64::round/floor/ceil/trunc are std-gated on
  this toolchain (rustc 1.99.0 source-tarball build). Use the codebase's
  clamp-then-`as usize` pattern for float->int conversion. Verified by
  `cargo build -p features-core` (plain lib target), which the `cargo test`
  target alone does NOT exercise — test it explicitly after any new module.

---

## v0.4.0 user-testing feedback (next direction, recorded 2025-11-25)

Queued after the sub-10ms pass, which shipped in v0.5.0 as accuracy-first (30ms budget, 27.9ms measured). Item 2 below is fixed in v0.5.0 (f61/f62 masked from is_digital); item 1 (rename) remains open.

1. **Rename the `_v2` naming mess.** `extract_all_v2`, `extract_all_v2_hybrid`,
   `extract_all_v2_with_glyph`, `extract_all_v2_scaled`, `process_one_v2`,
   `predict_v2`, `load_model_v2`, `train_model_v2.py`, `combine_rotation_v2`,
   `LABELS_V2` etc. are the per-degree rework's working names. Pick clean,
   intent-revealing names once the rework is the only path.

2. **is_digital head is over-fit to shadow depth.** 3 pages rotated 80 deg
   scored low; nearly the whole score gap between a low and a high page comes
   from f62 = shadow_depth_std (and f61 = shadow_depth_mean), the head's
   most-used feature, with splits clustering on it. shadow_depth_* measures
   how dark ink is against the local background — confounded with digital vs
   paper, not intrinsic to it. Fix: drop f61/f62 from the is_digital head OR
   normalize them by ink contrast.

3. **Add fade + overlay augmentation to training.** So the trait heads see
   illumination/overlay variation and stop keying on absolute ink darkness.

4. **Soften near-boundary scores.** is_digital outputs 0.47 and 0.16 currently
   read as confident rejection; they should be uncertain near 0.5. Add
   calibration/softening for the trait heads, not just the rotation confidence.

Cross-check: all 10 fatura pages turned correctly and all 6 upright left alone.
