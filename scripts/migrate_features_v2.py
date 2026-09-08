#!/usr/bin/env python3
"""
Migrate existing features.npz (81-feature, 5-label) to v2 (84-feature, 9-label)
with rotation augmentation (0°, 90°, 180°, 270°).

Note: Feature count has since increased to 94 (10 new structure-tensor, circular,
and projection features). Use export_features.py for fresh extraction.
Historical migration from 81→84 features.

Strategy:
  - Keep existing 81 features, re-extract just the 3 new document features
    (tall_fraction, v_run_mean, v_run_std) per image.
  - For each image, generate 3 rotated copies and extract all 84 features.
  - Output: 4× samples per original, 84 features, 9 labels.

Usage:
    python scripts/migrate_features_v2.py \
        --input data/features.npz \
        --images data/images \
        --output data/features_v2.npz \
        [--sample 100]

Set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 if using Python > 3.13.
"""

import argparse
import os
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
ROTATION_ANGLES = [0, 90, 180, 270]
MAX_LONG_EDGE = 512


def resize_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= MAX_LONG_EDGE:
        return img
    scale = MAX_LONG_EDGE / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def extract_features(img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB")
    w, h = rgb.size
    pixels = np.array(rgb, dtype=np.uint8).tobytes()
    feats = py_features.extract_all(pixels, w, h)
    return np.array(feats, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Migrate features to v2 with rotation labels")
    parser.add_argument("--input", required=True, help="Existing features.npz (v1)")
    parser.add_argument("--images", required=True, help="Image directory (e.g. data/images)")
    parser.add_argument("--output", required=True, help="Output .npz path (v2)")
    parser.add_argument("--sample", type=int, default=0, help="Limit to N images (0=all)")
    parser.add_argument(
        "--chunk", type=str, default="", help="Slice as 'start:end' (0-indexed, end exclusive)"
    )
    args = parser.parse_args()

    data = np.load(args.input)
    old_features = data["features"]
    old_labels = data["labels"]
    old_filenames = data["filenames"]

    total = len(old_features)
    chunk_start, chunk_end = 0, total
    if args.chunk:
        parts = args.chunk.split(":")
        chunk_start = int(parts[0])
        chunk_end = int(parts[1]) if len(parts) > 1 else total
        chunk_end = min(chunk_end, total)

    n_samples = chunk_end - chunk_start
    if args.sample > 0:
        n_samples = min(args.sample, n_samples)

    print(f"Source: {len(old_features)} samples, {old_features.shape[1]} features")
    print(f"Processing: {n_samples} images × 4 rotations = {n_samples * 4} output samples")

    new_features = []
    new_labels = []
    new_filenames = []
    errors = 0

    for i in range(chunk_start, chunk_start + n_samples):
        fname = str(old_filenames[i])
        img_path = os.path.join(args.images, fname)
        if not os.path.exists(img_path):
            errors += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = resize_image(img)
        except Exception:
            errors += 1
            continue

        old_label_vec = old_labels[i].tolist()

        for rot_idx, angle in enumerate(ROTATION_ANGLES):
            if angle == 0:
                rotated = img
            else:
                rotated = img.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)

            try:
                feats = extract_features(rotated)
            except Exception:
                errors += 1
                continue

            rotation_labels = [0, 0, 0, 0]
            rotation_labels[rot_idx] = 1
            full_labels = old_label_vec + rotation_labels

            new_features.append(feats)
            new_labels.append(full_labels)
            new_filenames.append(f"{fname}_rot{angle}")

        if (i - chunk_start + 1) % 100 == 0:
            print(
                f"  {i - chunk_start + 1}/{n_samples} images processed ({errors} errors)",
                flush=True,
            )

    if not new_features:
        print("No samples produced. Exiting.")
        sys.exit(1)

    features = np.stack(new_features, axis=0)
    labels = np.array(new_labels, dtype=np.int8)

    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        filenames=np.array(new_filenames),
        label_names=np.array(LABEL_NAMES),
    )

    file_size = os.path.getsize(args.output)
    print(
        f"\nDone: {features.shape[0]} samples, {features.shape[1]} features, "
        f"{labels.shape[1]} labels ({errors} errors)"
    )
    print(f"Wrote {args.output} ({file_size / 1024:.1f} KB)")

    # Label distribution
    print("\nLabel distribution:")
    for j, name in enumerate(LABEL_NAMES):
        pos = labels[:, j].sum()
        print(f"  {name}: {int(pos)} positive ({pos / len(labels) * 100:.1f}%)")


if __name__ == "__main__":
    main()
