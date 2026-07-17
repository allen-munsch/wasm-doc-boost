#!/usr/bin/env python3
"""
Parallel feature extraction via subprocess chunks.

Splits labels.csv into N chunks, runs export_features.py on each chunk
in parallel (subprocess, not multiprocessing.Pool — PyO3 hangs with Pool),
then merges the .npz files.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import numpy as np

LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]


def split_labels(labels_path: str, n_workers: int, tmpdir: str) -> list[str]:
    """Split labels.csv into n_workers chunk files, return their paths."""
    rows = []
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    chunk_size = (len(rows) + n_workers - 1) // n_workers
    chunks = []
    for i in range(n_workers):
        chunk_rows = rows[i * chunk_size:(i + 1) * chunk_size]
        if not chunk_rows:
            break
        chunk_path = os.path.join(tmpdir, f"labels_chunk_{i:03d}.csv")
        with open(chunk_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename"] + LABEL_NAMES)
            for r in chunk_rows:
                writer.writerow([r["filename"]] + [r[n] for n in LABEL_NAMES])
        chunks.append(chunk_path)
        print(f"  Chunk {i}: {len(chunk_rows)} images → {chunk_path}")
    return chunks


def run_worker(chunk_path: str, images_dir: str, output_path: str) -> subprocess.Popen:
    """Launch export_features.py as a subprocess."""
    env = os.environ.copy()
    env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
    return subprocess.Popen(
        [sys.executable, "scripts/export_features.py",
         "--images", images_dir,
         "--labels", chunk_path,
         "--output", output_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def merge_npz(outputs: list[str], final_path: str):
    """Merge multiple .npz files into one."""
    all_features = []
    all_labels = []
    all_filenames = []
    total = 0
    for path in outputs:
        data = np.load(path)
        all_features.append(data["features"])
        all_labels.append(data["labels"])
        all_filenames.append(data["filenames"])
        total += len(data["features"])
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    filenames = np.concatenate(all_filenames, axis=0)
    np.savez_compressed(
        final_path,
        features=features, labels=labels,
        filenames=filenames,
        label_names=np.array(LABEL_NAMES),
    )
    print(f"Merged {total} samples → {final_path} ({os.path.getsize(final_path) / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = split_labels(args.labels, args.workers, tmpdir)
        print(f"\nLaunching {len(chunks)} workers...")
        procs = []
        outputs = []
        for i, chunk_path in enumerate(chunks):
            out_path = os.path.join(tmpdir, f"features_chunk_{i:03d}.npz")
            outputs.append(out_path)
            procs.append(run_worker(chunk_path, args.images, out_path))

        print(f"Running {len(procs)} workers in parallel...")
        for i, proc in enumerate(procs):
            stdout, _ = proc.communicate()
            if proc.returncode != 0:
                print(f"Worker {i} FAILED (rc={proc.returncode}):")
                print(stdout[-500:])
            else:
                lines = stdout.strip().split("\n")
                last = lines[-1] if lines else "(no output)"
                first = lines[0] if lines else ""
                print(f"  Worker {i}: {first} ... {last}")

        missing = [p for p in outputs if not os.path.exists(p)]
        if missing:
            print(f"ERROR: {len(missing)} output files missing")
            sys.exit(1)

        merge_npz(outputs, args.output)


if __name__ == "__main__":
    main()
