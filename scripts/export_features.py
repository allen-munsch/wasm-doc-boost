#!/usr/bin/env python3
"""
Export pixel features from a labelled image dataset.

Usage:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 python scripts/export_features.py \\
        --images data/images --labels data/labels.csv --output data/features_v3.npz

Each image is augmented with 90, 180, and 270 degree rotations, producing 4
samples per original image. Rotation labels are mutually exclusive.

Output: .npz with features (N*4, 103), labels (N*4, 9), filenames (N*4,).
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

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


def load_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = [int(row[name]) for name in CSV_LABEL_NAMES]
    return labels


def _process_image(args):
    """Process one image: resize, rotate 4 ways, extract 103 features.

    Module-level function for ProcessPoolExecutor pickling.
    Returns (feats_4x103, labels_4x9, fnames_4) or None on error.
    """
    fname, label_vec, images_dir = args
    img_path = os.path.join(images_dir, fname)
    if not os.path.exists(img_path):
        return None

    try:
        img = Image.open(img_path)
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        feats_list, labels_list, fnames_list = [], [], []
        for rot_idx, angle in enumerate(ROTATION_ANGLES):
            rotated = img if angle == 0 else img.rotate(-angle, expand=True, resample=Image.BICUBIC)
            rgb = rotated.convert("RGB")
            rw, rh = rgb.size
            pixels = np.array(rgb, dtype=np.uint8).tobytes()
            feats = py_features.extract_all(pixels, rw, rh)
            rotation_labels = [0, 0, 0, 0]
            rotation_labels[rot_idx] = 1
            feats_list.append(feats)
            labels_list.append(label_vec + rotation_labels)
            fnames_list.append(f"{fname}_rot{angle}")
        return (feats_list, labels_list, fnames_list)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} images", flush=True)

    tasks = [(fname, labels[fname], args.images) for fname in labels]
    n_workers = args.workers if args.workers > 0 else mp.cpu_count()
    n_workers = min(n_workers, len(tasks))
    print(f"Using {n_workers} workers", flush=True)

    feature_list, label_list, filename_list = [], [], []
    errors, processed = 0, 0

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for result in executor.map(_process_image, tasks, chunksize=20):
            processed += 1
            if result is None:
                errors += 1
            else:
                feats, lbls, fnames = result
                feature_list.extend(feats)
                label_list.extend(lbls)
                filename_list.extend(fnames)
            if processed % 500 == 0:
                print(f"  {processed}/{len(tasks)} ({errors} errors)", flush=True)

    if not feature_list:
        print("No features extracted. Exiting.", flush=True)
        sys.exit(1)

    features = np.array(feature_list, dtype=np.float64)
    labels_arr = np.array(label_list, dtype=np.int8)
    print(f"Done: {features.shape[0]} samples, {errors} errors", flush=True)

    np.savez_compressed(
        args.output,
        features=features,
        labels=labels_arr,
        filenames=np.array(filename_list),
        label_names=np.array(LABEL_NAMES),
    )
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
