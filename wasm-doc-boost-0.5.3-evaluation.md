# wasm-doc-boost v0.5.3 vs v0.4.0: can it replace v0.4.0?

**Verdict: not yet. Keep v0.4.0 for QA. v0.5.3 is the first 0.5.x that is a serious candidate, with three blockers (section 4).**

Built from the `v0.5.3` tag (`maturin build --release`, aarch64). The tag has **no GitHub release and no assets** (the API returns 404
for it; the newest release is v0.5.0), so there are no wheel hashes to pin yet. Same inputs, scripts and methods as the 0.4.0 and 0.5.0
reports; scripts are in `docs/wasm-doc-boost-0.3.0-feedback/` (`run_with_053.py` runs the unchanged v0.4.0 scripts against the new
API names). v0.4.0 numbers are from the earlier runs on the same data.

## 1. Scorecard

| | v0.4.0 | v0.5.3 |
|---|---|---|
| cardinal rotations, 600 pages (150 FATURA x 4), exact | **99.5%** | 96.3% |
| &nbsp;&nbsp;of which 0 / 90 / 180 / 270 | 100 / 99 / 99 / 99% | 99 / 94 / 99 / 93% |
| axis (mod 180) right on cardinal pages | 600/600 | 598/600 |
| `needs_review` rate on those clean pages | 2.2% | 15.8% |
| upright pages dove would turn (`confidence >= 0.5`) | 0 | 1 |
| arbitrary angles (240 pages): axis median error | 0.31 deg | **0.08 deg** |
| arbitrary angles: axis within 5 deg | 99% | 92% (all 20 misses are flagged) |
| benchmark FATURA, quarter-turn correct (16 pages) | 16/16 | 16/16 |
| benchmark FATURA, worst residual of the model's own fine angle | 1.9 deg | **1.1 deg** |
| FATURA pages `is_digital` / routed clean (16) | 13 / 13 | **16 / 16** |
| SRD+CORD routed to the VLM lane (32) | 32/32 | 32/32 |
| SRD/CORD pages scored `is_digital >= 0.5` (190 pages: SRD 100 + CORD 90) | 0 | 0 |
| plain-text page `is_digital` (1 / 3 / 10 / 25 lines) | 1.0 / 1.0 / 1.0 / 0.005 | **1.0 / 1.0 / 1.0 / 1.0** |
| SRD wrong answers with confidence >= 0.5 | 54 of 308 | **3 of 333** |
| `predict` per page at 512 px | 110 ms | **21 ms** |
| tilted pages (560): 0-vs-180 wrong | 10% (single stage) | **16%** |

Better: speed (5x), `is_digital` on text-heavy and 80-degree pages, calibration on non-digital pages, axis precision, and the model now says
when its axis is unreliable. Worse: the half-turn on slightly tilted pages, the cardinal exact rate, and how often it asks for review.

## 2. The model's fine angle can now replace `deskew.py` (with a fallback)

560 tilted pages (20 FATURA x 14 tilts of +-0.5..12 deg x upright/upside-down), compared with `deskew.py`:

| | within 0.5 deg | within 1 deg | within 2 deg | worst |
|---|---|---|---|---|
| model axis, pages with `axis_ok` = true (531, 94.8%) | 95% | **100%** | 100% | 1.28 deg |
| `deskew.py` on the same pages | 100% | 100% | 100% | 1.00 deg |
| model axis, pages with `axis_ok` = false (29) | - | 7% | - | median 11.7 deg |
| `deskew.py` on those 29 | - | 100% | - | - |
| hybrid: `skew_deg` if `axis_ok` else `deskew.py` | 95% | **100%** | 100% | 1.28 deg |

`axis_ok` is a trustworthy flag: when it is true the axis is within 1 deg on every page; when false the model is wrong and `deskew.py`
is still right. So for born-digital pages dove could drop the separate deskew step and keep `deskew.py` only as the fallback (or send
those pages to the VLM). The `skew_deg` sign matches `deskew.py`'s applied angle (quarter 90, skew -4.86 for a page rotated 94.86).

## 3. The defect: the half-turn near the 0/180 fold

The README says 100 pages x 12 small tilts are 100% within 1 deg with none off by more than 10 deg. We do not reproduce that.
On the 560 tilted pages, 16% have the wrong 0-vs-180 decision and 111 are off by more than 10 deg; among pages the model **accepts**
(not `needs_review`), 6.5% are wrong, and for |tilt| 0.5 to 3 deg about 13 to 15% of the pages with `axis_ok` and `flip_ok` are wrong.

Fine sweep, 20 pages x upright/upside-down = 40 per cell, wrong half-turn near horizontal text:

| tilt (deg) | -3 | -2 | -1.5 | -1 | -0.75 | -0.5 | -0.25 | +0.25 | +0.5 | +0.75 | +1 | +1.5 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wrong of 40 | 3 | 3 | 13 | **27** | **23** | 15 | 2 | 1 | 1 | 2 | 3 | 8 | 3 | 5 |
| `needs_review` of 40 | 12 | 15 | 20 | 24 | 24 | 22 | 8 | 3 | 6 | 14 | 11 | 25 | 12 | 12 |

The untouched page (tilt 0) is 0 wrong of 40. The error is concentrated on **negative tilts of about -0.5 to -1.5 deg** (up to 68% wrong at
-1.0), while the mirror positive tilts are mostly fine. Around 90/270 (vertical text) there is no such band (1 to 5 wrong of 40 everywhere).
Examples: an upright page at -0.5 deg returns `rotation_deg = 180.0` with `skew_deg = 0`; at -1.5 deg it returns 178.64 instead of 358.5.
This looks like a wrap-around problem at the 0/180 fold (the fold-snap was reduced to 0.25 deg in 0.5.2), not weak evidence: the same
glyph/vote gate is right at 0 and at +1. Real scans are skewed by a degree or two in both directions, so this is the common case.

## 4. Blockers, and what would change the verdict

1. **The half-turn defect above.** Target: wrong half-turn under 5% for every tilt in -3..+3 deg (use `fold_sweep_053.py`), and accepted errors flagged.
2. **No release assets for v0.5.3.** Dove pins wheels by SHA-256 in `pr-gate.yml` and the `Containerfile`; we cannot do that for a bare tag.
3. **The review rate on clean pages is 15.8% (v0.4.0: 2.2%).** With the safe gate (`axis_ok` and `flip_ok`), 0 of 600 cardinal pages are wrong
   but 16.3% go to the VLM. With dove's current gate (`confidence >= 0.5`), 3.5% of cardinal pages are turned wrongly (v0.4.0: 0.5%).

| gate on 600 cardinal pages | applied right | applied wrong | left/VLM |
|---|---|---|---|
| v0.4.0, always | 99.5% | 0.5% | 0% |
| v0.4.0, not `needs_review` | 97.8% | 0.0% | 2.2% |
| v0.5.3, `confidence >= 0.5` (dove today) | 95.7% | 3.5% | 0.8% |
| v0.5.3, `confidence >= 0.9` | 93.7% | 2.2% | 4.2% |
| v0.5.3, not `needs_review` | 84.0% | 0.2% | 15.8% |
| v0.5.3, `axis_ok` and `flip_ok` | 83.7% | 0.0% | 16.3% |

## 5. Integration changes dove would need (small, all on our side)

- Python names changed: `load_model_v2` -> `load_model`, `predict_v2` -> `predict`, `extract_all_v2` -> `extract_all`.
- `model_version()` was removed from the Python module; dove's loader used it to check wheel against `model.json`.
- `_MODEL_SERIES` -> `"0.5"`; new keys: `quarter`, `skew_deg`, `axis_ok`, `flip_ok`, `trait_labels`, `trait_uncertain`.
- Dove's orient gate should use `axis_ok`/`flip_ok`, not `rotation_confidence` alone.

## 6. Not verified here

- Upstream's DocLayNet and CORU claims: neither dataset is on this machine (the CORU photos were removed from `data/`).
- The wasm bundle (only the Python wheel was tested), and crumple/shadow quality.
- Synthetic rotations of FATURA invoices only; SRD/CORD pages cannot be recovered by any version and are routed by `needs_review`.
- v0.5.3 was built locally (manylinux_2_35), not from CI artifacts.
