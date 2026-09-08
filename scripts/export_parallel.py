#!/usr/bin/env python3
"""
Launch parallel feature export via subprocess workers.

Splits the image list into chunks, runs one worker subprocess per chunk,
then merges the resulting .npz files.

Usage:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 python scripts/export_parallel.py \\
        --images data/images --labels data/labels.csv --output data/features_v3.npz
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

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


def chunk_list(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    # Load filenames from CSV
    filenames = []
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f):
            filenames.append(row["filename"])

    n_workers = args.workers if args.workers > 0 else os.cpu_count()
    n_workers = min(n_workers, len(filenames))
    print(f"Loaded {len(filenames)} images, splitting across {n_workers} workers")

    chunks = chunk_list(filenames, n_workers)

    tmpdir = tempfile.mkdtemp(prefix="export_")
    chunk_files = []
    procs = []

    t0 = time.time()
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(tmpdir, f"chunk_{i:04d}.npz")
        chunk_files.append(chunk_file)

        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "scripts/export_worker.py",
                "--images",
                args.images,
                "--labels-csv",
                args.labels,
                "--output",
                chunk_file,
            ],
            stdin=subprocess.PIPE,
            text=True,
        )
        proc.stdin.write("\n".join(chunk) + "\n")
        proc.stdin.close()
        procs.append((i, proc))

    print(f"Launched {n_workers} workers, waiting...")

    for i, proc in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"Worker {i}/{n_workers}: {status}")

    t1 = time.time()
    print(f"All workers done in {t1 - t0:.1f}s")

    # Merge chunks
    print("Merging chunks...")
    all_features = []
    all_labels = []
    all_filenames = []

    for chunk_file in chunk_files:
        if not os.path.exists(chunk_file):
            print(f"  WARNING: {chunk_file} missing, skipping")
            continue
        data = np.load(chunk_file, allow_pickle=True)
        all_features.append(data["features"])
        all_labels.append(data["labels"])
        all_filenames.append(data["filenames"])

    features = np.concatenate(all_features, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    filenames_arr = np.concatenate(all_filenames, axis=0)

    np.savez_compressed(
        args.output,
        features=features,
        labels=labels_arr,
        filenames=filenames_arr,
        label_names=np.array(LABEL_NAMES),
    )
    file_size = os.path.getsize(args.output)
    print(
        f"Written {args.output}: {features.shape} features, {labels_arr.shape} labels ({file_size / 1024:.0f} KB)"
    )

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
