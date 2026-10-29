# wasm-doc-boost v0.4.0 vs v0.5.3: source-level comparison and the exact changes needed

Follow-up to `wasm-doc-boost-0.5.3-evaluation.md`. This note reads the Rust source of both tags and ties each measured
difference to the code that causes it. **Measured** = reproduced with the scripts in `docs/wasm-doc-boost-0.3.0-feedback/`.
**Inferred** = consistent with the source and the data, not isolated by an experiment. Line numbers are v0.5.3 unless stated.

## 1. The short version

- v0.5.3 implemented the multistage design (S2 fine axis, S3 single deskew, S4 flip gate at any tilt, S5 split outputs) and it
  works where it works: `axis_ok` pages have the axis within 1 deg on 531/531 tilted pages, 5x faster, much better `is_digital`.
- It is worse than v0.4.0 in three separate places, and each has its own cause in the source:
  1. a **fold-band inconsistency** (two different snap constants for the same decision, plus no deskew inside the band);
  2. a **review veto driven by a flip head that got worse** on rotated pages;
  3. **weaker glyph evidence on sideways pages**, where the mask is actually rotated.
- All three are small, local fixes (section 5). None needs a retrain except the flip head (change 3).

## 2. Architecture, side by side

| stage | v0.4.0 | v0.5.3 |
|---|---|---|
| coarse axis | `radon_curve` on the 512 px image, 60 bins, parabola, top-4 | `radon_curve` on a **k=4 (128 px)** copy, top-4 (`lib.rs` `rotation_extract_hybrid_with_radon`) |
| candidate pick | `textline_flip_best_axis`, line quality at 512 px | `cheap_line_quality` at **k=2 (256 px)** |
| fine axis | none (3 deg bins + parabola) | **S2 `fine_axis`** (`radon.rs:407`): Otsu ink mask, projection variance, 0.15 deg steps over +-1.5 deg, parabola. A port of `deskew.py` |
| mask for the evidence | `rotate_mask_square`: nearest-neighbour rotation of an Otsu mask | `rotate_gray_square` (`textline.rs:68`): **bilinear rotation of grayscale, then Sauvola** |
| deskew angle for the evidence | `axis - 180`, but **0 when the axis is within 1.5 deg of the fold** | same, 1.5 deg (`textline.rs:147`, and again at `:364`) |
| flip gate scope | **cardinal axis only** (`gbdt-core:217`); GBDT flip head elsewhere | **any axis** (S4) |
| fold-snap in `combine_rotation` | `< 1.0 or > 179.0` (`gbdt-core:203`) | `< 0.75 or > 179.25` (`gbdt-core:235`) |
| glyph vs GBDT-head check | `needs_review` if they disagree (`:241`) | **same line** (`:271`) |
| outputs | `rotation_deg`, confidence, `needs_review`, glyph/vote counts | + `quarter`, `skew_deg`, `axis_ok`, `flip_ok`, `trait_labels` |
| inference features | 512 px, full-resolution everything | k=4 core, rotation parts at 512 |

## 3. Where v0.4.0 is still better, and why

### 3.1 The half-turn near the 0/180 fold (measured, causes inferred from source and confirmed by two experiments)

The radon axis is only defined mod 180, so the code carries a convention: `rotation_deg = (180 - axis) mod 180 + 180 * flip`, where
`flip` is the wrap bit (`rotation_deg >= 180`). The flip evidence must be computed in the **same frame** as that formula. Away from
the fold it is: `deskew_analyze` rotates the mask by `axis - 180`, so an upright page tilted -3 deg (axis 3) is rotated by -177,
appears upside-down in that frame, and the evidence says `flip = 1`, giving `rotation_deg = 177 + 180 = 357`. Correct.

Inside the snap band the frame changes. `deskew_analyze` (`textline.rs:147`) uses **no rotation** when the axis is within 1.5 deg of
the fold, so the evidence is in the *content* frame (upright = 0). `combine_rotation` (`gbdt-core:235`) only snaps within 0.75 deg.
For an axis in [0.75, 1.5) (tilt -0.75 to -1.5 deg) the evidence is content-frame and the formula is wrap-frame, so an upright page
gets `rotation_deg = 179` instead of 359. Two things make it worse:

- inside the band the mask is not deskewed, so up to 1.5 deg of tilt is left in the image the glyph classifier looks at;
- `fine_axis` returns its result mod 180, so a refinement that crosses 0 comes back as 179.x, which is exactly the input that the
  comment above `textline.rs:147` says the snap exists to handle.

Why v0.4.0 did not show it: the **same mismatch existed** (textline 1.5, combine 1.0, so [1.0, 1.5)), but the flip gate ran only at a
cardinal axis. After the snap, an axis of 1.2 is not cardinal, so the GBDT flip head (trained with the wrap convention) decided.
S4 removed that restriction and exposed the latent inconsistency; moving the combine snap from 1.0 to 0.25 and then 0.75 changed the
size of the gap but never closed it.

Evidence (measured, v0.5.3, 20 FATURA pages x upright/upside-down):

| measured refined axis | pages | half-turn wrong |
|---|---|---|
| [0.75, 1.5) deg | 114 | **65.8%** |
| [1.5, 3.5) deg | 162 | 5.6% |
| [177, 178.5) deg | 146 | 7.5% |
| [178.5, 179.25) deg | 120 | 13.3% |

For upright pages tilted -1 deg, 12 of 20 return `rotation_deg ~ 179` with a **negative** glyph log-odds (the evidence correctly says
"upright") and `flip_ok = True` for several of them: the glyph gate is right and the formula wraps it wrongly. Straightening those
pages first (so the axis is ~0 and the mask is clean) takes the band from **48.0% wrong to 7.0%**, and `needs_review` from 60% to 31%.
Flipping the bit alone does not fix it (66% -> 34%), which is the second effect: the evidence there is degraded by the uncorrected tilt.

### 3.2 Review rate: 2.2% -> 15.8% on clean cardinal pages (measured, cause of the head's drop inferred)

Of the 95 `needs_review` pages among 600 clean cardinal pages, **79 involve `glyph_gbdt_disagrees`** (`gbdt-core:271`), and for 70 of them
the confidence was fine. The line is identical in v0.4.0 (`:241`). What changed is the head it compares against:

| GBDT flip head right, same 600 pages | 0 | 90 | 180 | 270 |
|---|---|---|---|---|
| v0.4.0 | 146/150 | 149 | 147 | 149 |
| v0.5.3 | 148/150 | 139 | **127** | **121** |

The head was retrained (README: more sources, `train_additive.py`) and is now right only 81 to 93% of the time on rotated born-digital
pages, so a veto that was almost never triggered now fires on a quarter of them. The veto is wrong far more often than the signal it
overrides: of 79 disagreements, when the glyph evidence is strong (|sum log-odds| >= 20) the glyph gate is right **49 of 52** times and
the head 3 of 52; between 10 and 20 it is a coin flip (8 vs 6); below 10 the head is better (9 vs 4, on 13 pages).

### 3.3 Sideways pages: weaker evidence (measured; interpolation cause inferred)

Glyph log-odds, median |sum|: 0 deg 45.4 and 180 deg 44.3 in **both** versions (identical). At 90 deg: 38.7 (v0.4.0) vs 33.9; at 270 deg:
38.6 vs 30.6. Exact recovery at 90/270 is 94% and 93% vs 99%. The only difference between the pages that match and the pages that
do not is that 90/270 pages need a real rotation of the mask (`axis - 180 = -90`), whereas 0/180 pages hit the snap and are not rotated.
v0.4.0 rotated an Otsu mask with nearest-neighbour sampling, which is exact for multiples of 90; v0.5.3 rotates grayscale bilinearly
(`rotate_gray_square`, accumulating `xs += cos_t`) and then binarises with Sauvola, which blurs strokes even at exactly 90 deg.

### 3.4 Arbitrary angles: more misses (measured), all flagged

8.3% of random-angle pages have an axis error over 5 deg (v0.4.0: ~1%). 19 of the 20 have `axis_ok = false` and all 20 `needs_review`, so
this is a recall loss, not silent error. Source: candidate axes come from the **128 px** radon curve (top-4), so a true axis that is not
among the top four coarse peaks is never re-examined by the k=2 / S2 stages.

## 4. Where v0.5.3 is better, and why

| gain | source of the gain | measured |
|---|---|---|
| fine axis precision | S2 `fine_axis` (0.15 deg steps on an Otsu mask), i.e. `deskew.py`'s method inside the core | median axis error 0.08 deg vs 0.31; benchmark off-cardinal worst 1.1 deg vs 1.9 |
| `axis_ok` is trustworthy | `axis_ok = trust_p >= tau` (`gbdt-core:315`) with the trust head | 531/531 `axis_ok` tilted pages within 1 deg; the 29 others are wrong (median 11.7 deg) |
| speed | k=4 core features, coarse radon at 128 px | 21 ms vs 110 ms per page |
| `is_digital` | `HEAD_DROPPED_FEATURES = {"is_digital": [61, 62]}` in `train_model.py` (the shadow-depth features), DocLayNet and text-page training data | 80-degree FATURA pages 0.80 to 0.99; 25-line text page 1.0 (was 0.005) |
| calibration on non-digital pages | calibrated confidence + glyph calibration `GLYPH_CAL_B0..B2` (`gbdt-core:191`) | SRD wrong with conf >= 0.5: 3/333 vs 54/308 |
| wrong axis is flagged | `axis_ok`, `needs_review` | arbitrary-angle misses 19/20 flagged |
| API | `quarter`, `skew_deg`, `axis_ok`, `flip_ok`, `trait_labels` | `skew_deg` sign equals `deskew.py`'s applied angle |

## 5. Exact changes

Ordered by value. Each lists the file, the change, the evidence it is needed, and how to verify.

**C1. One fold convention, no skipped deskew** (`features-core/src/textline.rs:147` and `:364`, `gbdt-core/src/lib.rs:235`).
Put the decision in one place and stop using two thresholds:

```rust
// features-core (re-exported to gbdt-core)
pub const FOLD_BAND_DEG: f64 = 1.5;
/// Signed text-line tilt in (-90, 90] from the radon axis (mod 180).
pub fn signed_tilt_deg(axis_deg: f64) -> f64 {
    let alpha = (180.0 - axis_deg).rem_euclid(180.0);
    if alpha > 90.0 { alpha - 180.0 } else { alpha }
}

// textline.rs deskew_analyze (and cheap_line_quality): rotate by the EXACT residual inside the band
let tau = signed_tilt_deg(axis_deg);
let deskew_deg = if tau.abs() < FOLD_BAND_DEG { -tau } else { axis_deg - 180.0 };

// gbdt-core combine_rotation_calibrated: evidence is in the content frame inside the band
let tau = signed_tilt_deg(axis_deg);
let rotation_deg = if tau.abs() < FOLD_BAND_DEG {
    (tau + 180.0 * flip).rem_euclid(360.0)
} else {
    (180.0 - axis_deg).rem_euclid(180.0) + 180.0 * flip
};
```

The two branches agree at the band edge (an upright page at tau -1.6 gives 358.4 in the wrap branch, at tau -1.4 gives 358.6 in the
content branch). `fine_axis` should also return a signed value, or at least not wrap mod 180 before the caller has taken `tau`.
Delete the 0.75 / 1.0 constants and the stale comment above `textline.rs:147` ("snap that fold region to 0 deg").
*Evidence:* band 66% wrong; straightened-first 7%. *Verify:* `fold_sweep_053.py` and `fold_band_rootcause_053.py` (every cell under 5%),
then `tilt_eval_053.py` (0-vs-180 wrong under 5% overall, accepted errors flagged).

**C2. Do not let a weak head veto a strong glyph signal** (`gbdt-core/src/lib.rs:271` and `:314`).

```rust
const GLYPH_STRONG_LOGODDS: f64 = 30.0; // 20 clears more reviews but admits 3 wrong pages per 600
let glyph_strong = matches!(glyph, Some(ev) if ev.sum_logodds.abs() >= GLYPH_STRONG_LOGODDS);
let glyph_gbdt_disagrees = glyph_gate_active && flip != gbdt_flip && !glyph_strong;
```

*Evidence (exact, 600 clean cardinal pages; 95 reviews before):*

| threshold on abs(sum log-odds) | reviews after | cleared | glyph right | wrong accepted |
|---|---|---|---|---|
| 10 | 31 | 64 | 56 | 8 |
| 15 | 39 | 56 | 51 | 5 |
| 20 | 43 | 52 | 49 | 3 |
| **30** | **56** | 39 | 39 | **0** |

30 is the conservative choice (review 15.8% -> 9.3%, no new accepted error on this set); 20 reaches 7.2% at a cost of 3 pages (0.5%).
Either way keep a lower-confidence path rather than dropping the check, e.g. scale `flip_conf` by 0.8 on disagreement.
*Verify:* `veto_analysis_053.py`, `c2_exact_053.py`.

**C3. Retrain or re-weight the flip head on rotated born-digital copies** (`scripts/train_model.py`, `train_additive.py`).
The head fell from 97 to 99% to 81 to 93% per rotation. Make per-rotation head accuracy a gate in `goal_gate.py` (target >= 97% at each of 0/90/180/270
on rotated copies of an unseen born-digital set). Until then C2 is what stops it hurting.

**C4. Lossless 90-degree path** (`textline.rs:68 rotate_gray_square`). When `deskew_deg` is a multiple of 90, transpose/flip the buffer instead
of resampling. *Verify:* median |glyph log-odds| at 90/270 returns to >= 38 (v0.4.0: 38.7 / 38.6; v0.5.3: 33.9 / 30.6) and exact recovery at 90/270 returns to >= 98%.

**C5. Candidate recall** (`lib.rs rotation_extract_hybrid_with_radon`). Take the coarse curve at k=2 (256 px) or keep the top-6 maxima instead of top-4,
so the true axis is more often among the candidates. *Verify:* `arbitrary_angle_misses_053.py` (target at most 4 of 240, from 20).

**C6. Release hygiene.** Publish wheels, `model.json` and the wasm for v0.5.3 (the tag has no release, so no hashes can be pinned); keep
`model_version()` in the Python module (removed in 0.5.3) and keep the `_v2` names as deprecated aliases for one release.

**C7. Add to `goal_gate.py`:** (a) the +-3 deg fold sweep at 0.25 deg steps; (b) per-rotation head accuracy; (c) review rate on clean rotated
copies under 5%; (d) glyph evidence parity on sideways pages; (e) one born-digital set that is not in the training mix (our FATURA gives 96.3% exact
vs the README's 99.0% on FATURA2, so some of the headline is in-distribution).

*Estimate (not run):* C1 + C2 should remove the fold-band errors and cut the clean-page review rate to 7 to 9%, leaving cardinal exact recovery near
96% until C3/C4 land.

## 6. What we did not establish

- That the flip head's drop comes from the retraining (it is the likeliest reading of the README and the script names, not an ablation).
- That bilinear/Sauvola is the cause of the weaker sideways evidence (the 0/180 vs 90/270 split points to the rotation step; no A/B swap was run).
- Anything about the wasm bundle, DocLayNet or CORU (not available here), or real scans beyond SRD/CORD.
- Effects of C1 to C5 on the full model: the 7%/31% figures use `deskew.py` to straighten before calling the existing model, not the proposed Rust change.
