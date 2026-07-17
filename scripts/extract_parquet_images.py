#!/usr/bin/env python3
"""
Extract images from CORD-v2 and PubLayNet HuggingFace parquet files.

Uses Python 3.14 with pyarrow (these datasets store images as bytes in parquet).
Run with: mise exec python@3.14.6 -- python3 scripts/extract_parquet_images.py

Outputs flat image directories ready for feature extraction.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


def extract_cordv2(data_dir: str, out_dir: str, max_images: int | None = None) -> int:
    """Extract PNG/JPEG images from CORD-v2 parquet files."""
    import pandas as pd

    parquet_dir = os.path.join(data_dir, "data")
    pq_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not pq_files:
        print("  No parquet files found. Download first: hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir data/cord-v2")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for pq_path in pq_files:
        print(f"  Reading {pq_path.name}...")
        df = pd.read_parquet(pq_path)
        for idx, row in df.iterrows():
            img_data = row["image"]
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
            else:
                img_bytes = img_data

            if img_bytes:
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    fname = f"cordv2_{pq_path.stem}_{idx}.jpg"
                    img.save(os.path.join(out_dir, fname), quality=95)
                    count += 1
                except Exception as e:
                    print(f"    Skipping row {idx}: {e}")

            if max_images and count >= max_images:
                break
        if max_images and count >= max_images:
            break

    print(f"  Extracted {count} images to {out_dir}")
    return count


def extract_publaynet(data_dir: str, out_dir: str, max_images: int | None = None) -> int:
    """Extract images from PubLayNet HF parquet files."""
    import pandas as pd

    parquet_dir = os.path.join(data_dir, "data")
    pq_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not pq_files:
        print("  No parquet files found. Download first via hf download jordanparker6/publaynet --repo-type dataset --local-dir data/publaynet")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for pq_path in pq_files:
        print(f"  Reading {pq_path.name}...")
        df = pd.read_parquet(pq_path)
        for idx, row in df.iterrows():
            img_data = row.get("image")
            if img_data is None:
                continue
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
            else:
                img_bytes = img_data

            if img_bytes:
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    fname = f"publaynet_{pq_path.stem}_{idx}.jpg"
                    img.save(os.path.join(out_dir, fname), quality=95)
                    count += 1
                except Exception:
                    pass

            if max_images and count >= max_images:
                break
        if max_images and count >= max_images:
            break

    print(f"  Extracted {count} images to {out_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract images from HF parquet datasets")
    parser.add_argument("--dataset", choices=["cord-v2", "publaynet", "all"],
                        default="all", help="Which dataset to extract")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--out-dir", default="data/images", help="Output image directory")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images per dataset")
    args = parser.parse_args()

    datasets = ["cord-v2", "publaynet"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        print(f"\n── {ds} ──")
        ds_data_dir = os.path.join(args.data_dir, ds)
        ds_out_dir = os.path.join(args.out_dir, ds)
        if ds == "cord-v2":
            extract_cordv2(ds_data_dir, ds_out_dir, args.max_images)
        elif ds == "publaynet":
            extract_publaynet(ds_data_dir, ds_out_dir, args.max_images)


if __name__ == "__main__":
    main()
