# NEXT FIX — rotation fold-snap (root cause found, fix VERIFIED numerically, not yet applied)

Paste this into a fresh session to pick up exactly where we stopped.

## The bug (measured on shipped model)

Gold set: `data/images/fatura2` — 1000 upright born-digital A4 invoices,
gold `rotation_deg = 0`. Shipped model = `frontend/wasm_bridge_bg.wasm` +
`data/model_v2.json` (tau=0.9), run via `scripts/eval_rotation_gold.mjs`:

- `is_document` 1000/1000 OK
- axis head 100% within 1 deg (mean 0.135) OK  — the AXIS is fine
- `rotation_deg`: 544 @180 + 6 @179  vs  386 @0 + 64 @360  -> **55% flip error**
- `needs_review` fired only 3.6% (so it is confidently wrong), mean conf 0.976

On the full cached features (`data/features_v5.npz`, fatura2, all 8 rotations,
TRUE flip labels): OLD combine is 90.6% <=1deg, 8.6% flipped; a=0 only 66%.

## Root cause

`RADON_AXIS_INDEX` (feature[100]) is deterministic and mod-180 blind. For
exactly-horizontal text the radon argmax lands randomly on bin 0 (~0.01 deg)
or bin 59 (~179.99 deg). `combine_rotation` maps it with
`rotation_deg = (180 - axis) % 180 + 180*flip`, so:

- axis ~0.01   -> rotation 179.99  (180 deg WRONG)
- axis ~179.99 -> rotation 0.01    (correct)

That bimodality is the 45/55 split. The flip head is NOT the culprit: it
predicts flip ~0.01-0.05 on upright input. It dodged the bug because
`crates/features-core/src/textline.rs` (lines ~79-83) ALREADY snaps the same
fold region to 0 in its deskew path:

    let deskew_deg = if axis_deg < 1.5 || axis_deg > 178.5 { 0.0 } else { axis_deg - 180.0 };

Only the final combine uses the RAW feature[100] without that snap.

## The fix (pure combine change — NO retrain)

`crates/wasm-bridge/src/lib.rs`, `combine_rotation` (~line 269). After
`axis_deg = axis_rad.to_degrees().rem_euclid(180.0)`, canonicalize the fold
region, mirroring textline.rs's convention:

```rust
let axis_rad = features[RADON_AXIS_INDEX];
let mut axis_deg = axis_rad.to_degrees().rem_euclid(180.0);
if axis_deg < 1.0 || axis_deg > 179.0 {
    axis_deg = 0.0; // fold-snap: ~0 and ~179.99 are the SAME horizontal text
}
```

Mirror the identical snap in `scripts/analyze_sweep.py` (axis_deg at ~line 37,
rot_pred at ~line 53) so eval and runtime agree. No other code computes the
rotation_deg combine.

## Verification already done (numeric, no rebuild needed)

Simulated on `data/features_v5.npz` fatura2 with TRUE flip labels
(`/tmp/snapcheck.py` logic, eps swept):

    OLD              <=1deg 90.6%  flip(>90) 8.6%   a=0 66.0%  a=180 66.1%
    SNAP(eps=1.0)    <=1deg 98.8%  flip(>90) 0.41%  a=0 100%   a=180 99.5%
    SNAP(eps=0.5)    <=1deg 98.9%  flip(>90) 0.34%  a=0 99.94% a=180 99.4%

a=90 and a=270 stay 99.9% under the snap (the earlier SIGNED-tilt idea broke
them to 58%; do NOT use signed tilt — it contradicts the flip-label semantics
`flip = (a>=180) XOR source_down`). Residual ~0.5% at a=180 is the flip head's
own error, not the fold; trust/needs_review should cover it. eps=1.0 gives
a=0 exactly 100%; pick 1.0 (or 1.5 to match textline.rs).

## Next steps (in order)

1. Apply the snap in `crates/wasm-bridge/src/lib.rs` + `scripts/analyze_sweep.py`.
2. Rebuild and copy the wasm:
   - `wasm-pack build --target web crates/wasm-bridge`
   - `cp crates/wasm-bridge/pkg/wasm_bridge.js crates/wasm-bridge/pkg/wasm_bridge.d.ts crates/wasm-bridge/pkg/wasm_bridge_bg.wasm crates/wasm-bridge/pkg/wasm_bridge_bg.wasm.d.ts frontend/`
3. Re-run the gold eval: `node scripts/eval_rotation_gold.mjs`
   Expect on fatura2_test_0..999: rotation_deg distribution ~all at 0 (+/-1),
   `pm1_accuracy` ~0.99+, `flip_error_rate_strict` ~0, needs_review small.
4. Re-run `scripts/analyze_sweep.py --sweep <sweep.npz>` (axis-only now;
   the GBDT flip/trust heads are evaluated through the WASM gold eval) to
   confirm the axis stays sane. The old `--models <models.joblib>` path was
   removed along with the joblib trainer/analyzer machinery.
5. Also re-check the earlier standing regression tests:
   `cargo test -p features-core` and `cargo test -p wasm-bridge` (the
   combine_rotation unit test at wasm-bridge/src/lib.rs ~line 469 expects
   `axis=17, flip=0 -> 163`; 17 deg is outside the snap band so it still holds).

## Environment gotchas (this box)

- node: `/home/jm/.local/share/mise/installs/node/24.22.1/bin/node`
  (plain `node` is NOT on PATH)
- python + numpy: prefix with
  `LD_LIBRARY_PATH=/nix/store/ji36842higmp6b06hbbpmzbd59g1c07c-zlib-1.3.2/lib:/nix/store/dqj57l1prad38vjrad4bs9d977i4q5zm-gcc-15.3.0-lib/lib PYTHONPATH=.venv-p0/lib/python3.14/site-packages python3`
  (system python is 3.14.8; the .venv symlinks are broken -> mise 3.14.6 gone)

## Key files

- `crates/wasm-bridge/src/lib.rs` — `combine_rotation` (the bug), RADON_AXIS_INDEX=100
- `crates/features-core/src/textline.rs:79-83` — the existing snap that proves the convention
- `crates/features-core/src/radon.rs` — deterministic radon, mod-180, bin 0/59 tie
- `scripts/analyze_sweep.py:37,53` — eval mirror of the combine
- `scripts/eval_rotation_gold.mjs` — gold upright eval (already written)
- `data/features_v5.npz` — cached features for fast numeric checks
- `PLAN.md` — overall plan; escalation Path B (vision encoder) is only for the
  flip head if it stalls below 0.95 AFTER this fold fix is shipped
