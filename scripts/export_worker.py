#!/usr/bin/env python3
"""Worker subprocess for feature extraction. Reads image list from stdin, writes .npz to --output."""

import argparse
import csv
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
CSV_LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
ROTATION_ANGLES = [0, 90, 180, 270]
MAX_LONG_EDGE = 512


def process_one(fname, label_vec, images_dir):
    img_path = os.path.join(images_dir, fname)
    if not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path)
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        feats_list, labels_list, fnames_list = [], [], []
        for rot_idx, angle in enumerate(ROTATION_ANGLES):
            if angle == 0:
                rotated = img
            else:
                rotated = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
            rgb = rotated.convert("RGB")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-csv", required=True)
    args = parser.parse_args()

    # Load all labels
    labels = {}
    with open(args.labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = [int(row[name]) for name in CSV_LABEL_NAMES]

    # Read filenames from stdin (one per line)
    filenames = [line.strip() for line in sys.stdin if line.strip()]

    feature_list, label_list, filename_list = [], [], []
    errors = 0
    for i, fname in enumerate(filenames):
        if fname not in labels:
            errors += 1
            continue
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
