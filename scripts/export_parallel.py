#!/usr/bin/env python3
"""Parallel v2 (per-degree) feature extraction.

Splits labels.csv into N chunks, runs export_worker.py --v2 on each chunk
as a subprocess (PyO3 hangs with Pool), then merges the .npz files.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    chunk_size = (len(rows) + args.workers - 1) // args.workers
    with tempfile.TemporaryDirectory() as tmpdir:
        procs = []
        outputs = []
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
                    "--v2",
                ],
                stdin=open(flist_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            procs.append((i, proc))
            print(f"launched worker {i} ({len(chunk)} images)", flush=True)

        for i, proc in procs:
            out_txt, _ = proc.communicate()
            tail = out_txt.strip()[-200:] if out_txt.strip() else "(no output)"
            print(f"worker {i} rc={proc.returncode}: {tail}", flush=True)

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
