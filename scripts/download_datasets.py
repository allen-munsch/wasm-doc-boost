#!/usr/bin/env python3
"""
Download and prepare 7 training datasets for wasm-doc-boost.

  1. RVL-CDIP      (gdown) — 400k scanned documents
  2. Tobacco800    (gdown) — 1,290 scanned docs
  3. COCO train2017 (wget) — natural images (negatives)
  4. PubLayNet     (wget) — digital-born document layouts
  5. SmartDoc QA   (wget) — mobile-captured docs w/ shadows
  6. CORU          (hf)   — receipt images (is_document, is_paper)
  7. CORD-v2       (hf)   — receipt images (is_document, is_paper)

Outputs:
  data/images/          -- all images
  data/labels.csv       -- manifest for export_features.py
"""

import argparse
import csv
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

SEED = 42
LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]

# ── Dataset URLs exactly as provided ──────────────────────────────────────

DATASETS = {
    "rvl-cdip": {
        "url": "https://drive.google.com/uc?id=1QPjXJ8W0HvT3pBJKaGxkPQmDY6LZrVfI",
        "tool": "gdown",
        "archive": "rvl-cdip.zip",
        "extract_dir": "rvl-cdip",
        "label": (1, 0, 1, 0, 0),
        "max_images": 5000,
    },
    "tobacco800": {
        "url": "https://drive.google.com/uc?id=1iHmz0CkC42U5kgVJx5Qj5sV1j4M6_0uE",
        "tool": "gdown",
        "archive": "Tobacco800.zip",
        "extract_dir": "tobacco800",
        "label": (1, 0, 1, 0, 0),
        "max_images": None,
    },
    "coco": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "tool": "wget",
        "archive": "val2017.zip",
        "extract_dir": "coco",
        "label": (0, 0, 0, 0, 0),
        "max_images": 5000,
    },
    "publaynet": {
        "url": "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/PubLayNet.zip",
        "tool": "wget",
        "archive": "PubLayNet.zip",
        "extract_dir": "publaynet",
        "label": (1, 1, 0, 0, 0),
        "max_images": 5000,
    },
    "smartdoc": {
        "url": "http://liris.cnrs.fr/smartdoc/15/dataset/SmartDoc-QA-2015.zip",
        "tool": "wget",
        "archive": "SmartDoc-QA-2015.zip",
        "extract_dir": "smartdoc",
        "label": (1, 0, 1, 0, 1),
        "max_images": None,
    },
    "coru": {
        "url": "https://huggingface.co/datasets/abdoelsayed/CORU",
        "tool": "hf-zip",
        "archive": None,  # downloads individual zips
        "extract_dir": "coru",
        "label": (1, 0, 1, 0, 0),
        "max_images": 5000,
        "hf_zips": [
            "Receipt/train.zip",
            "Receipt/val.zip",
            "OCR/train.zip",
        ],
    },
    "cord-v2": {
        "url": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
        "tool": "hf-parquet",
        "archive": None,
        "extract_dir": "cord-v2",
        "label": (1, 0, 1, 0, 0),
        "max_images": None,
    },
}


# ── Downloaders ───────────────────────────────────────────────────────────

def download_gdown(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  gdown {url} → {dest}")
    subprocess.run(["gdown", "-O", dest, url], check=True)


def download_wget(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  wget {url} → {dest}")
    subprocess.run(
        ["wget", "--no-check-certificate", "-c", "-O", dest, url],
        check=True,
    )


def download_url_raw(url: str, dest: str) -> None:
    """Straight urllib download (no wget/gdown dependency)."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} → {dest}")
    req = Request(url, headers={"User-Agent": "wasm-doc-boost/1.0"})
    with urlopen(req) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)


# ── Extractors ────────────────────────────────────────────────────────────

def extract_zip(path: str, dest_dir: str) -> None:
    done_marker = os.path.join(dest_dir, ".extracted")
    if os.path.exists(done_marker):
        print(f"  Already extracted: {dest_dir}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  Extracting {path} → {dest_dir}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest_dir)
    Path(done_marker).touch()


def extract_tar(path: str, dest_dir: str) -> None:
    done_marker = os.path.join(dest_dir, ".extracted")
    if os.path.exists(done_marker):
        print(f"  Already extracted: {dest_dir}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  Extracting {path} → {dest_dir}")
    with tarfile.open(path) as tar:
        tar.extractall(dest_dir, filter="fully_trusted")
    Path(done_marker).touch()


# ── Helpers ───────────────────────────────────────────────────────────────

def find_images(base_dir: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    results = []
    for root, _dirs, files in os.walk(base_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                results.append(os.path.relpath(os.path.join(root, f), base_dir))
    return results


def copy_images(src_dir: str, out_dir: str, ds_name: str, max_images: int | None,
                label: tuple) -> list:
    """Find images under src_dir, optionally subset, copy to out_dir, return CSV rows."""
    os.makedirs(out_dir, exist_ok=True)
    img_paths = find_images(src_dir)
    if not img_paths:
        print(f"  WARNING: no images found in {src_dir}")
        return []

    random.seed(SEED)
    random.shuffle(img_paths)
    if max_images and len(img_paths) > max_images:
        img_paths = img_paths[:max_images]

    print(f"  Processing {len(img_paths)} images")
    rows = []
    for rel_path in img_paths:
        src = os.path.join(src_dir, rel_path)
        fname = f"{ds_name}_{os.path.basename(rel_path)}"
        dest = os.path.join(out_dir, fname)
        if not os.path.exists(dest):
            try:
                img = Image.open(src).convert("RGB")
                img.save(dest)
            except Exception:
                continue
        rows.append([fname, *label])
    print(f"  Wrote {len(rows)} images to {out_dir}")
    return rows


# ── HF dataset handlers ───────────────────────────────────────────────────

def download_hf_zip(dataset_id: str, hf_path: str, raw_dir: str) -> str:
    """Download a zip file from a HuggingFace dataset repo."""
    fname = hf_path.replace("/", "_")
    dest = os.path.join(raw_dir, fname)
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{hf_path}?download=true"
    download_url_raw(url, dest)
    return dest


def process_coru(raw_dir: str, out_dir: str, cfg: dict) -> list:
    """CORU: download receipt zip files from HF, extract, collect images."""
    os.makedirs(raw_dir, exist_ok=True)
    extract_root = os.path.join(raw_dir, cfg["extract_dir"])
    done_marker = os.path.join(extract_root, ".extracted")
    if os.path.exists(done_marker):
        print(f"  Already extracted CORU: {extract_root}")
        return copy_images(extract_root, out_dir, "coru", cfg["max_images"], cfg["label"])

    os.makedirs(extract_root, exist_ok=True)
    dataset_id = "abdoelsayed/CORU"
    for hf_path in cfg["hf_zips"]:
        zip_dest = download_hf_zip(dataset_id, hf_path, raw_dir)
        extract_zip(zip_dest, extract_root)
    Path(done_marker).touch()

    return copy_images(extract_root, out_dir, "coru", cfg["max_images"], cfg["label"])


def process_cordv2(raw_dir: str, out_dir: str, cfg: dict) -> list:
    """CORD-v2: download parquet files from HF, extract images via pillow."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for CORD-v2. Install: pip install pandas pyarrow")

    extract_root = os.path.join(raw_dir, cfg["extract_dir"])
    done_marker = os.path.join(extract_root, ".extracted")
    if os.path.exists(done_marker):
        print(f"  Already processed CORD-v2: {extract_root}")
        return copy_images(extract_root, out_dir, "cord-v2", cfg["max_images"], cfg["label"])

    os.makedirs(extract_root, exist_ok=True)
    base_url = "https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/parquet/default"

    # Get parquet file list
    for split in ("train",):
        req = Request(f"{base_url}/{split}", headers={"User-Agent": "wasm-doc-boost/1.0"})
        with urlopen(req) as resp:
            parquet_urls = json.loads(resp.read().decode())

        for pq_url in parquet_urls:
            df = pd.read_parquet(pq_url)
            for idx, row in df.iterrows():
                img_data = row["image"]
                if isinstance(img_data, dict):
                    img_bytes = img_data.get("bytes")
                    if img_bytes:
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img.save(os.path.join(extract_root, f"cord-v2_{split}_{idx}.png"))

    Path(done_marker).touch()
    return copy_images(extract_root, out_dir, "cord-v2", cfg["max_images"], cfg["label"])


# ── Per-dataset processor ─────────────────────────────────────────────────

def process_dataset(ds_name: str, cfg: dict, raw_dir: str, out_dir: str) -> list:
    print(f"\n── {ds_name} ──")

    tool = cfg["tool"]

    if tool == "hf-zip":
        return process_coru(raw_dir, out_dir, cfg)
    if tool == "hf-parquet":
        return process_cordv2(raw_dir, out_dir, cfg)

    # Standard download + extract workflow
    archive_path = os.path.join(raw_dir, cfg["archive"])

    if tool == "gdown":
        download_gdown(cfg["url"], archive_path)
    else:
        download_wget(cfg["url"], archive_path)

    extract_dir = os.path.join(raw_dir, cfg["extract_dir"])
    ext = os.path.splitext(cfg["archive"])[1].lower()
    if ext == ".zip":
        extract_zip(archive_path, extract_dir)
    elif ext in (".tar", ".gz"):
        extract_tar(archive_path, extract_dir)
    else:
        extract_zip(archive_path, extract_dir)

    return copy_images(extract_dir, out_dir, ds_name, cfg["max_images"], cfg["label"])


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--skip", action="append", default=[],
                        help="Datasets to skip")
    parser.add_argument("--only", action="append", default=None,
                        help="Only process these datasets")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Don't abort on first download failure")
    args = parser.parse_args()

    data_dir = args.data_dir
    raw_dir = os.path.join(data_dir, "raw")
    out_dir = os.path.join(data_dir, "images")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    to_process = [d for d in DATASETS if d not in args.skip]
    if args.only:
        to_process = args.only

    print(f"Datasets to process: {', '.join(to_process)}")
    print(f"Data dir: {os.path.abspath(data_dir)}\n")

    all_rows = []
    failed = []
    for ds_name in to_process:
        try:
            rows = process_dataset(ds_name, DATASETS[ds_name], raw_dir, out_dir)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(ds_name)
            if not args.continue_on_error:
                print(f"\nAborting after {ds_name} failure. Use --continue-on-error to skip.")
                raise

    if failed:
        print(f"\nDatasets with download failures: {', '.join(failed)}")

    # Shuffle and write CSV
    random.seed(SEED)
    random.shuffle(all_rows)

    csv_path = os.path.join(data_dir, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)
        writer.writerows(all_rows)

    print(f"\n=== Done ===")
    print(f"Total images: {len(all_rows)}")
    if all_rows:
        arr = np.array([r[1:] for r in all_rows], dtype=np.int8)
        for i, name in enumerate(LABEL_NAMES):
            print(f"  {name}: {arr[:, i].sum()} positive, {len(arr) - arr[:, i].sum()} negative")
    print(f"Labels: {csv_path}")
    print(f"\nNext: python scripts/export_features.py --images {out_dir}/ "
          f"--labels {csv_path} --output {data_dir}/features.npz")


if __name__ == "__main__":
    main()
