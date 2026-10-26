#!/usr/bin/env python3
"""Worker subprocess for feature extraction. Reads image list from stdin, writes .npz to --output.

Cardinal mode (default, pre-rework): 4 rotations (0/90/180/270), 9 labels,
103 features (extract_all).

--v2 (per-degree rework): per base image K=8 angles = the 4 cardinal
rotations + 4 seeded-random angles in [0,360); 116 features (extract_all_v2);
7 labels [is_document, is_digital, is_paper, is_crumpled, is_shadow, flip,
trust] plus a rotation_deg metadata array. flip = 1 when the applied PIL CCW
rotation is >= 180 (after deskew by the radon axis the text is upside down);
trust = 1 when the deterministic radon axis is within 1 deg (circular) of the
true axis (180 - angle) mod 180.
"""

import argparse
import csv
import hashlib
import math
import os
import random
import sys

import numpy as np
import py_features
from PIL import Image

LABEL_NAMES = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "rotation_0",
    "rotation_90",
    "rotation_180",
    "rotation_270",
]
V2_LABEL_NAMES = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "flip",
    "trust",
]
CSV_LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
ROTATION_ANGLES = [0, 90, 180, 270]
V2_CARDINAL_ANGLES = [0, 90, 180, 270]
V2_K_RANDOM = 4
MAX_LONG_EDGE = 512
BORDER_FILL_WIDTH = 4
V2_FEAT_COUNT = 116
V2_RADON_AXIS_INDEX = 100  # first rotation feature (radon best_axis_rad)
V2_FLIP_SCORE_INDEX = 104  # textline flip_score (asc-desc ink, + = upright)
TRUST_TOLERANCE_DEG = 1.0


def median_border_color(rgb, border=BORDER_FILL_WIDTH):
    """Median color of the image's border pixels — the document's own background.

    Used as the rotate() fillcolor so the regions exposed by rotation blend
    into the background instead of leaking black (0,0,0) corners into the
    training data.
    """
    a = np.asarray(rgb)
    top = a[:border].reshape(-1, 3)
    bottom = a[-border:].reshape(-1, 3)
    left = a[border:-border, :border].reshape(-1, 3)
    right = a[border:-border, -border:].reshape(-1, 3)
    border_px = np.concatenate([top, bottom, left, right], axis=0)
    return tuple(int(v) for v in np.median(border_px, axis=0))


def v2_angles(fname):
    """K=8 angles per base image: 4 cardinal (exact 0/180 and 90/270 flip
    pairs) + 4 seeded-random angles for continuous axis coverage."""
    seed = hashlib.sha256(fname.encode()).digest()
    rng = random.Random(seed)
    angles = list(V2_CARDINAL_ANGLES) + [
        rng.uniform(0.0, 360.0) for _ in range(V2_K_RANDOM)
    ]
    return angles


def process_one(fname, label_vec, images_dir):
    img_path = os.path.join(images_dir, fname)
    if not os.path.exists(img_path):
        return None
    try:
        # Decode + resize EXACTLY as wasm-bridge (image crate, Triangle),
        # so training features match the inference path pixel-for-pixel.
        with open(img_path, "rb") as fh:
            raw = fh.read()
        pixels, rw, rh = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(rh, rw, 3)
        img = Image.fromarray(rgb, "RGB")
        fill = median_border_color(img)

        feats_list, labels_list, fnames_list = [], [], []
        for rot_idx, angle in enumerate(ROTATION_ANGLES):
            if angle == 0:
                rotated = img
            else:
                rotated = img.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=fill)
            rgb = rotated
            rw, rh = rgb.size
            pixels = np.array(rgb, dtype=np.uint8).tobytes()
            feats = py_features.extract_all(pixels, rw, rh)
            rotation_labels = [0, 0, 0, 0]
            rotation_labels[rot_idx] = 1
            full_labels = label_vec + rotation_labels
            feats_list.append(feats)
            labels_list.append(full_labels)
            fnames_list.append(f"{fname}_rot{angle}")
        return (feats_list, labels_list, fnames_list)
    except Exception:
        return None


def process_one_v2(fname, label_vec, images_dir):
    img_path = os.path.join(images_dir, fname)
    if not os.path.exists(img_path):
        return None
    try:
        # Decode + resize EXACTLY as wasm-bridge (image crate, Triangle),
        # so training features match the inference path pixel-for-pixel.
        with open(img_path, "rb") as fh:
            raw = fh.read()
        pixels, rw, rh = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(rh, rw, 3)
        img = Image.fromarray(rgb, "RGB")
        fill = median_border_color(img)

        feats_list, labels_list, fnames_list, angles_list = [], [], [], []
        for angle in v2_angles(fname):
            if angle == 0:
                rotated = img
            else:
                rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=fill)
            rw, rh = rotated.size
            pixels = np.array(rotated, dtype=np.uint8).tobytes()
            feats = py_features.extract_all_v2(pixels, rw, rh)
            assert len(feats) == V2_FEAT_COUNT, f"expected {V2_FEAT_COUNT} features, got {len(feats)}"

            # Radon is deterministic and mod-180 blind; PIL CCW rotate by
            # `angle` puts the text-line axis at (180 - angle) mod 180.
            axis_deg = math.degrees(feats[V2_RADON_AXIS_INDEX]) % 180.0
            true_axis = (180.0 - angle) % 180.0
            err = min(abs(axis_deg - true_axis), 180.0 - abs(axis_deg - true_axis))
            trust = 1 if err <= TRUST_TOLERANCE_DEG else 0

            feats_list.append(feats)
            labels_list.append(label_vec + [0, trust])  # flip patched below
            fnames_list.append(f"{fname}_rot{angle:.1f}")
            angles_list.append(angle)

        # flip = "text is upside down after deskew", NOT "angle >= 180".
        # Some fatura2 sources are stored 180-deg flipped, so the applied
        # angle alone does not say whether the text is upside down. The
        # source orientation oracle is the flip_score (asc-desc ink) of the
        # unrotated source, cross-checked between the 0 and 90 rotations
        # (both deskew to upright, so their flip_scores must agree).
        fs0 = next(f[V2_FLIP_SCORE_INDEX] for f, a in zip(feats_list, angles_list) if a == 0.0)
        fs90 = next(f[V2_FLIP_SCORE_INDEX] for f, a in zip(feats_list, angles_list) if a == 90.0)
        src_up = (fs0 + fs90) > 0.0
        for k, angle in enumerate(angles_list):
            needs_flip = (angle >= 180.0) != (not src_up)
            labels_list[k][5] = 1 if needs_flip else 0
        return (feats_list, labels_list, fnames_list, angles_list)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--v2", action="store_true", help="per-degree rework export")
    args = parser.parse_args()

    labels = {}
    with open(args.labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = [int(row[name]) for name in CSV_LABEL_NAMES]

    filenames = [line.strip() for line in sys.stdin if line.strip()]

    feature_list, label_list, filename_list, angle_list = [], [], [], []
    errors = 0
    for i, fname in enumerate(filenames):
        if fname not in labels:
            errors += 1
            continue
        if args.v2:
            result = process_one_v2(fname, labels[fname], args.images)
            if result is None:
                errors += 1
            else:
                feats, lbls, fnames, angles = result
                feature_list.extend(feats)
                label_list.extend(lbls)
                filename_list.extend(fnames)
                angle_list.extend(angles)
        else:
            result = process_one(fname, labels[fname], args.images)
            if result is None:
                errors += 1
            else:
                feats, lbls, fnames = result
                feature_list.extend(feats)
                label_list.extend(lbls)
                filename_list.extend(fnames)
        if (i + 1) % 500 == 0:
            print(f"  [{os.getpid()}] {i + 1}/{len(filenames)}", file=sys.stderr, flush=True)

    if not feature_list:
        print(f"Worker {os.getpid()}: no features extracted ({errors} errors)", file=sys.stderr)
        return

    features = np.array(feature_list, dtype=np.float64)
    labels_arr = np.array(label_list, dtype=np.int8)

    if args.v2:
        np.savez_compressed(
            args.output,
            features=features,
            labels=labels_arr,
            filenames=np.array(filename_list),
            rotation_deg=np.array(angle_list, dtype=np.float64),
            label_names=np.array(V2_LABEL_NAMES),
        )
    else:
        np.savez_compressed(
            args.output,
            features=features,
            labels=labels_arr,
            filenames=np.array(filename_list),
        )
    print(
        f"Worker {os.getpid()}: wrote {args.output} ({features.shape[0]} samples, {errors} errors)",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
