#!/usr/bin/env python3
"""Analyze a per-degree rotation sweep: where does the deterministic radon
axis break down, angle by angle, image by image (mod-180, with the fold-snap
that matches wasm-bridge's combine_rotation).

The GBDT flip/trust heads are evaluated through the WASM now
(scripts/eval_rotation_gold.mjs); there is no python-side model loading here.
"""

import argparse

import numpy as np

RADON_AXIS_INDEX = 100


def circular_err(a, b, period=360.0):
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--base", default=None, help="only analyze this base filename")
    parser.add_argument(
        "--top", type=int, default=8, help="per image, print worst N angles by axis err"
    )
    args = parser.parse_args()

    d = np.load(args.sweep, allow_pickle=True)
    F, A, fnames = d["features"], d["angles"].astype(float), np.asarray(d["filenames"])
    base = np.array([f.rsplit("_rot", 1)[0] for f in fnames])
    axis_deg = np.degrees(F[:, RADON_AXIS_INDEX]) % 180.0
    axis_deg[(axis_deg < 1.0) | (axis_deg > 179.0)] = (
        0.0  # fold-snap (matches wasm combine_rotation)
    )
    true_axis = (180.0 - A) % 180.0
    axis_err = circular_err(axis_deg, true_axis, period=180.0)

    print(f"Sweep: {len(F)} samples, {len(np.unique(base))} images")
    print("\n=== AXIS ONLY (deterministic) ===")
    print(
        f"  axis <=1deg: {np.mean(axis_err <= 1.0) * 100:.2f}%  "
        f"<=2deg: {np.mean(axis_err <= 2.0) * 100:.2f}%  "
        f"mean err {axis_err.mean():.3f}  max {axis_err.max():.2f}"
    )

    # Per-image worst angles
    print("\n=== PER-IMAGE WORST (axis err) ===")
    for b in np.unique(base):
        if args.base and b != args.base:
            continue
        m = base == b
        err = axis_err[m]
        order = np.argsort(err)[::-1][: args.top]
        am = A[m]
        parts = [f"{am[i]:3.0f}deg:{err[i]:5.2f}" for i in order]
        print(
            f"  {b}: mean {err.mean():.2f} max {err.max():.2f}  "
            f"worst {', '.join(parts)}"
        )

    # Angle histogram
    print("\n=== MEAN ERR BY ABSOLUTE ANGLE (axis) ===")
    for a in range(0, 360, 10):
        m = (A >= a) & (A < a + 10)
        if m.any():
            print(
                f"  {a:3d}-{a + 9:3d}: n={m.sum():3d} axis_mean={axis_err[m].mean():.3f} "
                f"axis_max={axis_err[m].max():.2f}"
            )


if __name__ == "__main__":
    main()
