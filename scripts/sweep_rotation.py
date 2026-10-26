#!/usr/bin/env python3
"""Per-degree rotation sweep: rotate N images through 360 one-degree steps,
extract v2 features (deterministic radon axis + GBDT inputs), save for offline
analysis. Mirrors export_worker.py's preprocessing exactly so the features
match the training distribution.

Usage:
    python scripts/sweep_rotation.py --images data/images \
        --filenames /tmp/sweep_fnames.txt --output /tmp/sweep10.npz --workers 8
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import py_features
from PIL import Image

MAX_LONG_EDGE = 512
BORDER_FILL_WIDTH = 4
V2_FEAT_COUNT = 116
RADON_AXIS_INDEX = 100


def median_border_color(rgb, border=BORDER_FILL_WIDTH):
    a = np.asarray(rgb)
    top = a[:border].reshape(-1, 3)
    bottom = a[-border:].reshape(-1, 3)
    left = a[border:-border, :border].reshape(-1, 3)
    right = a[border:-border, -border:].reshape(-1, 3)
    border_px = np.concatenate([top, bottom, left, right], axis=0)
    return tuple(int(v) for v in np.median(border_px, axis=0))


def sweep_one(args):
    fname, images_dir = args
    path = os.path.join(images_dir, fname)
    try:
        # Decode + resize EXACTLY as wasm-bridge (image crate, Triangle),
        # so the sweep reflects the inference path pixel-for-pixel.
        with open(path, "rb") as fh:
            raw = fh.read()
        pixels, rw, rh = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(rh, rw, 3)
        img = Image.fromarray(rgb, "RGB")
        fill = median_border_color(img)
    except (OSError, ValueError, RuntimeError) as e:
        return fname, None, str(e)

    feats_list, angle_list = [], []
    for angle in range(360):
        if angle == 0:
            rotated = img
        else:
            rotated = img.rotate(
                angle, expand=True, resample=Image.BICUBIC, fillcolor=fill
            )
        rw, rh = rotated.size
        pixels = np.array(rotated, dtype=np.uint8).tobytes()
        feats = py_features.extract_all_v2(pixels, rw, rh)
        assert len(feats) == V2_FEAT_COUNT, f"{fname}: {len(feats)} feats"
        feats_list.append(feats)
        angle_list.append(angle)
    return (
        fname,
        (np.array(feats_list, dtype=np.float64), np.array(angle_list, dtype=np.int32)),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--filenames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.filenames) as f:
        filenames = [ln.strip() for ln in f if ln.strip()]
    print(f"Sweeping {len(filenames)} images x 360 angles", flush=True)

    tasks = [(fn, args.images) for fn in filenames]
    all_feats, all_angles, all_fnames = [], [], []
    with Pool(args.workers) as pool:
        for fname, result, err in pool.imap_unordered(sweep_one, tasks):
            if err:
                print(f"ERROR {fname}: {err}", flush=True)
                continue
            feats, angles = result
            all_feats.append(feats)
            all_angles.append(angles)
            all_fnames.extend(f"{fname}_rot{a}" for a in angles)
            print(f"  done {fname} ({len(angles)} angles)", flush=True)

    F = np.vstack(all_feats)
    A = np.concatenate(all_angles)
    np.savez_compressed(
        args.output, features=F, angles=A, filenames=np.array(all_fnames)
    )
    print(
        f"Wrote {args.output}: {F.shape}, axis span "
        f"{np.degrees(F[:, RADON_AXIS_INDEX]).min():.1f}.."
        f"{np.degrees(F[:, RADON_AXIS_INDEX]).max():.1f} deg",
        flush=True,
    )


if __name__ == "__main__":
    main()
