#!/usr/bin/env python3
"""Parallel per-degree feature extraction, GPU-first.

Splits labels.csv into N chunks, runs export_worker.py on each chunk as a
subprocess (PyO3 hangs with Pool), streams each worker's progress live, then
merges the .npz files. Radon runs on the py_features backend (auto -> CUDA
when the .so was built with --features gpu and a device is reachable).
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

LABEL_NAMES = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "flip",
    "trust",
]


def stream_worker(i, proc):
    for line in proc.stdout:
        print(f"[w{i}] {line.rstrip()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    try:
        import py_features

        backend = py_features.backend_info()
        print(f"radon backend: {backend}", flush=True)
        if backend == "cpu":
            print("WARNING: CPU radon (py_features built without --features gpu)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"radon backend probe failed: {e}", flush=True)

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    chunk_size = (len(rows) + args.workers - 1) // args.workers
    with tempfile.TemporaryDirectory() as tmpdir:
        procs = []
        outputs = []
        threads = []
        for i in range(args.workers):
            chunk = rows[i * chunk_size : (i + 1) * chunk_size]
            if not chunk:
                break
            flist_path = os.path.join(tmpdir, f"files_{i:03d}.txt")
            with open(flist_path, "w") as f:
                for r in chunk:
                    f.write(r["filename"] + "\n")
            out_path = os.path.join(tmpdir, f"chunk_{i:03d}.npz")
            outputs.append(out_path)
            env = os.environ.copy()
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/export_worker.py",
                    "--images",
                    args.images,
                    "--labels-csv",
                    args.labels,
                    "--output",
                    out_path,
                ],
                stdin=open(flist_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            procs.append((i, proc))
            t = threading.Thread(target=stream_worker, args=(i, proc), daemon=True)
            t.start()
            threads.append(t)
            print(f"launched worker {i} ({len(chunk)} images)", flush=True)

        for t in threads:
            t.join()
        for i, proc in procs:
            rc = proc.wait()
            if rc != 0:
                print(f"ERROR: worker {i} rc={rc}", flush=True)
                sys.exit(1)

        missing = [o for o in outputs if not os.path.exists(o)]
        if missing:
            print(f"ERROR: {len(missing)} output files missing", flush=True)
            sys.exit(1)

        feats, lbls, fns, degs = [], [], [], []
        total = 0
        for o in outputs:
            d = np.load(o)
            feats.append(d["features"])
            lbls.append(d["labels"])
            fns.append(d["filenames"])
            degs.append(d["rotation_deg"])
            total += len(d["features"])
        np.savez_compressed(
            args.output,
            features=np.concatenate(feats, axis=0),
            labels=np.concatenate(lbls, axis=0),
            filenames=np.concatenate(fns, axis=0),
            rotation_deg=np.concatenate(degs, axis=0),
            label_names=np.array(LABEL_NAMES),
        )
        size_mb = os.path.getsize(args.output) / 1e6
        print(f"merged {total} samples -> {args.output} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
