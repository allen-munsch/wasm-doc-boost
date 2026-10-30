# wasm-doc-boost v0.5.4: evaluation against v0.4.0 and v0.5.3

**Verdict: not ready to replace v0.4.0.** v0.5.4 fixes the defect we reported in 0.5.3 (the half-turn near the 0/180 fold) and is a real
release with pinnable assets, but the fix introduces a regression on perfectly straight pages, which is Dove's main workload. A one-line
follow-up (a small dead band) recovers it in our test build. v0.4.0 stays the QA candidate.

Tested the **published release assets** (aarch64 wheel, `model.json`, wasm), the same inputs and scripts as the 0.4.0, 0.5.0 and 0.5.3 reports
(`docs/wasm-doc-boost-0.3.0-feedback/`). The tag was re-checked before writing: same commit (0a5b17d), same asset sizes and publish time.

## 1. What 0.5.4 is

- **A logic release; the model is unchanged.** `model.json` is byte-identical to v0.5.3's, and its manifest still says `"version": "0.5.3"` inside a
  v0.5.4 release, so the manifest cannot be used to confirm the matching wheel.
- The commit implements our C1 (fold band: shared `signed_tilt_deg` / `FOLD_BAND_DEG`, deskew by `-tau` inside the band, `rotation_deg = (tilt + 180*flip) mod 360`),
  C2 (skip the glyph-vs-head veto at |sum log-odds| >= 30) and C6 (`model_version()` and the `_v2` names restored; release assets published).
- Not done: C3 (retrain the flip head), C4 (lossless 90 deg path), C5 (candidate recall). Nothing else in the Rust changed.
- Pins now possible: aarch64 wheel `2041ac18a6cd241f18fce9173f49a4b4eaa582ac27940c4567c5ef3762f07ace`, x86_64 wheel
  `8e33e30e7e9bcb75887b2c150953376cf553afb28cd27d0099b4bcb73cd7023e`, `model.json` `69bb03f12bdd3ad367008b1c2738957c74d1a0aab277aa22c2fa7e38685b368f`.

## 2. Scorecard

`0.5.4 + dead band` is a **test build we made** (our patch in section 4), not a release.

| | v0.4.0 | v0.5.3 | **v0.5.4 release** | 0.5.4 + dead band (ours) |
|---|---|---|---|---|
| cardinal rotations, 600 pages: exact | **99.5%** | 96.3% | **93.3%** | 96.3% |
| &nbsp;&nbsp;0 / 90 / 180 / 270 | 100 / 99 / 99 / 99 | 99 / 94 / 99 / 93 | **92** / 94 / 95 / 93 | 99 / 94 / 99 / 93 |
| upright pages dove would turn (`confidence >= 0.5`), of 150 | 0 | 1 | **12** | 1 |
| `needs_review` on those clean pages | 2.2% | 15.8% | 14.7% | 10.2% |
| median abs glyph log-odds on untouched 0 deg pages | 45.4 | 45.4 | **34.1** | 44.7 |
| tilted pages (560): half-turn wrong | 10% (single stage) | 16.2% | **8.0%** | 9.6% |
| tilted pages: wrong among accepted (not review) | - | 6.5% | **1.8%** | 1.9% |
| tilted pages: off by > 10 deg | - | 111 | 67 | 76 |
| worst fold-sweep cell (wrong of 40) | - | **27** (at -1.0 deg) | 5 | 13 (at -0.5 deg) |
| `axis_ok` pages, axis within 1 deg (worst) | - | 531/531 (1.28) | 530/530 (0.68) | - |
| arbitrary angles: axis within 5 deg | 99% | 92% | 92% (same 20 misses, all flagged) | 92% |
| plain-text pages `is_digital` (1/3/10/25 lines) | 1/1/1/0.005 | 1/1/1/1 | 1/1/1/1 | - |
| SRD+CORD `is_digital >= 0.5` (190 pages) | 0 | 0 | 0 | - |
| SRD/CORD wrong answers with confidence >= 0.5 | 54/308 | 3/333 | 2/333 (SRD), 3/282 (CORD) | - |
| benchmark FATURA quarter-turn correct (16) | 16/16 | 16/16 | 16/16 | - |
| benchmark FATURA flagged for review | - | 4/16 | 2/16 | - |
| `predict` per page, 512 px | 110 ms | 21 ms | 26 ms | - |

The `is_digital` and routing rows are carried over from 0.5.3: the model file is identical and those features do not depend on the rotation code.

## 3. The fix works, and costs something

**The fold defect is gone.** Wrong half-turn near horizontal text, of 40 pages per cell:

| tilt (deg) | -3 | -2 | -1.5 | -1 | -0.75 | -0.5 | -0.25 | +0.25 | +0.5 | +0.75 | +1 | +1.5 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5.3 | 3 | 3 | 13 | **27** | **23** | 15 | 2 | 1 | 1 | 2 | 3 | 8 | 3 | 5 |
| 0.5.4 | 3 | 3 | 2 | 3 | 5 | 4 | 2 | 4 | 4 | 5 | 5 | 3 | 3 | 5 |

It is now the same flat 2 to 5 as the vertical-text neighbourhood. On tilted pages the half-turn error halves (16.2% -> 8.0%) and the accepted-wrong rate falls from
6.5% to 1.8%; every tilt bin is 7.5 to 10% (0.5.3: 25 to 29% for small tilts).

**The cost: straight pages got worse.** In 0.5.3, a page measured 0.3 to 0.5 deg off (the fine axis drifts that much on a true-0 page) fell inside the 1.5 deg snap
and was not rotated at all. In 0.5.4 it is deskewed by `-tau` with a bilinear rotation. The glyph classifier is very sensitive to that: 19 pages that were right in 0.5.3 are
wrong in 0.5.4 (1 improved), all at 0 and 180 deg. Example: `Template25_Instance64` goes from glyph log-odds **-49.9 to +13.5** (the sign flips) after a 0.53 deg resample.
The damage is broader than the flips: median glyph evidence on untouched 0 deg pages drops from 45.4 to 34.1, which is why C2 clears fewer reviews than predicted
(15.8% -> 14.7%, not 9.3%): the pages that carry a strong glyph signal in 0.5.3 carry a weaker one now.

## 4. A dead band recovers it (our test build)

We rebuilt v0.5.4 from source with one change in `deskew_analyze` and `cheap_line_quality` (`textline.rs`): do not rotate the mask when `|tau| < 0.6 deg`; keep the
content-frame formula in the band, so the fold fix is untouched:

```rust
pub const DESKEW_DEADBAND_DEG: f64 = 0.6;   // features-core/src/lib.rs
let deskew_deg = if tau.abs() < crate::DESKEW_DEADBAND_DEG { 0.0 }
                 else if tau.abs() < crate::FOLD_BAND_DEG { -tau }
                 else { axis_deg - 180.0 };
```

Result: cardinal exact 93.3% -> 96.3% (back to 0.5.3), upright pages turned 12 -> 1, median glyph evidence at 0 deg 34.1 -> 44.7, review 14.7% -> 10.2%,
and with the safe gate (`axis_ok` and `flip_ok`) 89.5% right, **0.0% wrong**, 10.5% to the VLM. The price is on tilted pages: half-turn wrong 8.0% -> 9.6% (15.6% in the
0.5 to 1 deg bin) and a new spike at -0.5 deg in the fold sweep (13 of 40). So 0.6 is too wide; the right value is between 0 and 0.6, and the real fix is a glyph classifier that
is not fragile to a sub-degree resample (train with small-rotation and blur augmentation, or extract crops with a shear/line-wise correction instead of a full bilinear rotation).

## 5. Gates, 600 cardinal pages

| gate | v0.4.0 | v0.5.3 | v0.5.4 release | + dead band |
|---|---|---|---|---|
| always apply: right / wrong | 99.5 / 0.5 | 96.3 / 3.7 | 93.3 / 6.7 | 96.3 / 3.7 |
| `confidence >= 0.5` (dove today): right / wrong / VLM | 99.5 / 0.5 / 0 | 95.7 / 3.5 / 0.8 | 92.7 / 6.5 / 0.8 | 95.7 / 3.5 / 0.8 |
| not `needs_review`: right / wrong / VLM | 97.8 / 0.0 / 2.2 | 84.0 / 0.2 / 15.8 | 84.5 / 0.8 / 14.7 | 89.7 / 0.2 / 10.2 |
| `axis_ok` and `flip_ok`: right / wrong / VLM | - | 83.7 / 0.0 / 16.3 | 84.3 / 0.7 / 15.0 | 89.5 / 0.0 / 10.5 |

## 6. Still open (unchanged from 0.5.3)

- **C3** the flip head: right on 81 to 93% of rotated pages (v0.4.0: 97 to 99%), still the source of most reviews.
- **C4** sideways pages: glyph evidence at 90/270 deg is 33.9 / 30.6 (v0.4.0: 38.7 / 38.6); exact recovery 94% / 93% vs 99%.
- **C5** candidate recall: 20 of 240 arbitrary-angle pages have the axis off by > 5 deg (all flagged).
- Manifest `version` should equal the release tag.

## 7. What would make 0.5.x the QA candidate

1. Straight pages no worse than 0.5.3 (cardinal exact >= 96%, upright pages turned <= 1 of 150) while keeping the fold fix (fold sweep <= 5 of 40 in every cell): a tuned dead band, or better a resample-robust glyph classifier.
2. Review rate on clean rotated pages under 5% (needs C3 and a less fragile glyph path), with the safe gate at 0% wrong.
3. Cardinal exact >= 98% on a born-digital set outside the training mix (our FATURA: 99.5% for v0.4.0).

The Dove-side wiring would now be small: `_MODEL_SERIES = "0.5"`, pin the SHA-256s above, and gate on `axis_ok` and `flip_ok`.

## 8. Not verified

- DocLayNet and CORU claims (datasets not on this machine); the wasm bundle (only the Python wheel was tested).
- The x86_64 wheel (this machine is aarch64; hash recorded only).
- The dead-band build is ours and tested on FATURA only; 0.6 is not an optimum.
- Synthetic rotations and tilts of FATURA invoices; SRD/CORD are unrecoverable by any version and are routed by `needs_review`.
