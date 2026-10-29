#!/usr/bin/env python3
"""Worker subprocess for feature extraction. Reads image list from stdin, writes .npz to --output.

Cardinal mode (default, pre-rework): 4 rotations (0/90/180/270), 9 labels,
103 features (extract_all).

Per base image, K=16 angles = the 4 cardinal
rotations + 6 seeded-random theta in (0,180), each exported at theta and
theta+180 (flip-contrast pairs); 124 features (extract_all);
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

import gpu_rotate

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
LABEL_NAMES = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "flip",
    "trust",
]
CSV_LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
CARDINAL_ANGLES = [0, 90, 180, 270]
K_FLIP_PAIRS = 6  # random theta in (0,180), each exported at theta and theta+180
MAX_LONG_EDGE = 512
BORDER_FILL_WIDTH = 4
FEAT_COUNT = 124
RADON_AXIS_INDEX = 100  # first rotation feature (radon best_axis_rad)
FLIP_SCORE_INDEX = 104  # textline flip_score (asc-desc ink, + = upright)
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


def augment_source(img, fname):
    """Deterministic mild degradation for robustness training (v0.4.0 feedback).

    Seeded by the base filename so every angle of one image shares the same
    augmentation (the theta/theta+180 flip-contrast pairs stay pure). Fade
    pushes ink toward the paper background; overlay adds faint scan noise.
    Rotation and doc-trait labels are unchanged.
    """
    rng = random.Random(hashlib.sha256(fname.encode()).digest())
    a = np.asarray(img, dtype=np.float32)
    fill = np.array(median_border_color(img), dtype=np.float32)

    if rng.random() < 0.5:  # fade toward paper
        t = rng.uniform(0.35, 0.65)
        a = a + (fill - a) * t

    if rng.random() < 0.5:  # overlay faint scan noise
        a = a + rng.gauss(0.0, 6.0)

    return Image.fromarray(np.clip(a, 0.0, 255.0).astype(np.uint8), "RGB")


def export_angles(fname):
    """K=16 angles per base image: 4 cardinals + 6 seeded-random theta in
    (0,180), each exported at theta and theta+180 so every non-cardinal
    sample has its exact flip counterpart at the same axis."""
    seed = hashlib.sha256(fname.encode()).digest()
    rng = random.Random(seed)
    angles = list(CARDINAL_ANGLES)
    for _ in range(K_FLIP_PAIRS):
        theta = rng.uniform(0.0, 180.0)
        angles.append(theta)
        angles.append(theta + 180.0)
    return angles


def process_one(fname, label_vec, images_dir, gpu=False):
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
        img = augment_source(img, fname)
        fill = median_border_color(img)

        feats_list, labels_list, fnames_list, angles_list = [], [], [], []
        for angle in export_angles(fname):
            if angle == 0:
                rotated_rgb = rgb
            elif gpu:
                rotated_rgb = gpu_rotate.rotate_rgb(rgb, angle, fill)
            else:
                rotated = img.rotate(
                    angle, expand=True, resample=Image.BICUBIC, fillcolor=fill
                )
                rotated_rgb = np.array(rotated, dtype=np.uint8)
            h2, w2 = rotated_rgb.shape[:2]
            pixels = np.ascontiguousarray(rotated_rgb).tobytes()
            feats = py_features.extract_all(pixels, w2, h2)
            assert len(feats) == FEAT_COUNT, (
                f"expected {FEAT_COUNT} features, got {len(feats)}"
            )

            # Radon is deterministic and mod-180 blind; PIL CCW rotate by
            # `angle` puts the text-line axis at (180 - angle) mod 180.
            axis_deg = math.degrees(feats[RADON_AXIS_INDEX]) % 180.0
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
        fs0 = next(
            f[FLIP_SCORE_INDEX] for f, a in zip(feats_list, angles_list) if a == 0.0
        )
        fs90 = next(
            f[FLIP_SCORE_INDEX] for f, a in zip(feats_list, angles_list) if a == 90.0
        )
        # Born-digital sources (is_digital == 1) are upright by construction;
        # the asc/desc oracle is a coin flip on all-caps/digit invoices
        # (flip_score ~ 0), so trust the source property, not the ink oracle.
        is_digital = label_vec[1] if len(label_vec) > 1 else 0
        src_up = (is_digital == 1) or ((fs0 + fs90) > 0.0)
        for k, angle in enumerate(angles_list):
            needs_flip = (angle >= 180.0) != (not src_up)
            labels_list[k][5] = 1 if needs_flip else 0
        return (feats_list, labels_list, fnames_list, angles_list)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"    skipping {fname}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--gpu", action="store_true", help="rotate on CUDA via torch (radon runs on the py_features backend; auto=cuda)")
    args = parser.parse_args()

    print(
        f"[{os.getpid()}] radon backend: {py_features.backend_info()}",
        file=sys.stderr,
        flush=True,
    )

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
        result = process_one(fname, labels[fname], args.images, gpu=args.gpu)
        if result is None:
            errors += 1
        else:
            feats, lbls, fnames, angles = result
            feature_list.extend(feats)
            label_list.extend(lbls)
            filename_list.extend(fnames)
            angle_list.extend(angles)
        if (i + 1) % 500 == 0:
            print(
                f"  [{os.getpid()}] {i + 1}/{len(filenames)}",
                file=sys.stderr,
                flush=True,
            )

    if not feature_list:
        print(
            f"Worker {os.getpid()}: no features extracted ({errors} errors)",
            file=sys.stderr,
        )
        return

    features = np.array(feature_list, dtype=np.float64)
    labels_arr = np.array(label_list, dtype=np.int8)

    np.savez_compressed(
        args.output,
        features=features,
        labels=labels_arr,
        filenames=np.array(filename_list),
        rotation_deg=np.array(angle_list, dtype=np.float64),
        label_names=np.array(LABEL_NAMES),
    )
    print(
        f"Worker {os.getpid()}: wrote {args.output} ({features.shape[0]} samples, {errors} errors)",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
