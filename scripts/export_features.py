#!/usr/bin/env python3
"""
Export pixel features from a labelled image dataset.

Usage:
    python scripts/export_features.py \\
        --images /path/to/images/ \\
        --labels labels.csv \\
        --output features.npz

Labels CSV format:
    filename,is_document,is_digital,is_paper,is_crumpled,is_shadow
    img_001.jpg,1,0,1,0,0
    img_002.png,1,1,0,0,0
    ...

Output format:
    .npz file containing:
      - features: float64 array of shape (N, 78)
      - labels: int8 array of shape (N, 5)
      - filenames: str array of shape (N,)
      - label_names: ['is_document', 'is_digital', 'is_paper', 'is_crumpled', 'is_shadow']

Set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 if using Python > 3.13.
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

import py_features

LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
MAX_LONG_EDGE = 512


def load_labels(path: str) -> dict[str, list[int]]:
    """Parse labels CSV into {filename: [label_values]}."""
    labels: dict[str, list[int]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            labels[fname] = [int(row[name]) for name in LABEL_NAMES]
    return labels


def resize_image(img: Image.Image) -> Image.Image:
    """Resize so the long edge is at most MAX_LONG_EDGE pixels."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= MAX_LONG_EDGE:
        return img
    scale = MAX_LONG_EDGE / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def extract_features(img: Image.Image) -> np.ndarray:
    """Extract 78 features from a PIL RGB image."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    pixels = np.array(rgb, dtype=np.uint8).tobytes()
    feats = py_features.extract_all(pixels, w, h)
    return np.array(feats, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Export pixel features from labelled images")
    parser.add_argument("--images", required=True, help="Directory containing image files")
    parser.add_argument("--labels", required=True, help="CSV file with labels")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labelled images")

    feature_list = []
    label_list = []
    filename_list = []
    missing = 0
    errors = 0

    for i, (fname, label_vec) in enumerate(labels.items()):
        img_path = os.path.join(args.images, fname)
        if not os.path.exists(img_path):
            missing += 1
            continue

        try:
            img = Image.open(img_path)
            img = resize_image(img)
            feats = extract_features(img)
            feature_list.append(feats)
            label_list.append(label_vec)
            filename_list.append(fname)
        except Exception as e:
            errors += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(labels)} processed ({missing} missing, {errors} errors)",
                  flush=True)

    if not feature_list:
        print("No images processed. Exiting.")
        sys.exit(1)

    features = np.stack(feature_list, axis=0)
    labels_arr = np.array(label_list, dtype=np.int8)

    print(f"Processed {len(feature_list)} images ({missing} missing, {errors} errors)")
    print(f"Features shape: {features.shape} (dtype={features.dtype})")
    print(f"Labels shape: {labels_arr.shape} (dtype={labels_arr.dtype})")

    np.savez_compressed(
        args.output,
        features=features,
        labels=labels_arr,
        filenames=np.array(filename_list),
        label_names=np.array(LABEL_NAMES),
    )
    file_size = os.path.getsize(args.output)
    print(f"Wrote {args.output} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
