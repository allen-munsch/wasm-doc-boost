#!/usr/bin/env python3
"""
Build labels.csv from all image dirs and generate missing training data.

Labels:
  FATURA2   → is_document=1, is_digital=1, is_paper=0  (synthetic invoices)
  CORD-v2   → is_document=1, is_paper=1                 (photographed receipts)
  SRD       → is_document=1, is_paper=1                 (scanned receipts)

Generates:
  - 2000 synthetic negative images (solid colors, noise, gradients)
  - 500 crumpled variants from CORD-v2/SRD (is_crumpled=1)
  - 500 shadow variants from CORD-v2/SRD (is_shadow=1)
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

SEED = 42
LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]

# Dir → (is_document, is_digital, is_paper, is_crumpled, is_shadow)
DATASET_LABELS = {
    "fatura2": (1, 1, 0, 0, 0),
    "cord-v2": (1, 0, 1, 0, 0),
    "srd": (1, 0, 1, 0, 0),
    "coco": (0, 0, 0, 0, 0),  # natural images → all labels negative
}


def generate_negatives(out_dir: str, n: int) -> list:
    """Solid colors, gradients, noise — all is_document=0."""
    os.makedirs(out_dir, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    sizes = [(640, 480), (800, 600), (512, 512), (400, 300), (300, 400),
             (600, 800), (480, 640), (256, 256), (1024, 768), (768, 1024)]
    rows = []
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        style = random.choice(["solid", "gradient", "noise"])
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        if style == "solid":
            color = np.random.randint(0, 256, 3)
            arr[:] = color.reshape(1, 1, 3)
        elif style == "gradient":
            yy = np.linspace(0, 1, h).reshape(-1, 1, 1)
            c1 = np.random.randint(0, 256, 3).reshape(1, 1, 3)
            c2 = np.random.randint(0, 256, 3).reshape(1, 1, 3)
            arr = (c1 * (1 - yy) + c2 * yy).astype(np.uint8)
        # else: noise (already random)
        fname = f"negative_{i:06d}.png"
        Image.fromarray(arr).save(os.path.join(out_dir, fname))
        rows.append([f"negatives/{fname}", 0, 0, 0, 0, 0])
    print(f"  Generated {n} negative images")
    return rows


def augment_shadow(img: Image.Image) -> Image.Image:
    w, h = img.size
    arr = np.array(img.convert("RGB"), dtype=np.float64)
    shadow_type = random.choice(["corner", "edge", "vignette"])
    strength = random.uniform(0.3, 0.7)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    if shadow_type == "corner":
        mask = (xx + yy) / 2.0
    elif shadow_type == "edge":
        mask = np.minimum(xx, 1.0 - xx)
    else:
        mask = 1.0 - np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / np.sqrt(2)
        mask = np.clip(mask, 0, 1)
    mask_3 = np.stack([mask, mask, mask], axis=-1) * strength
    arr = np.clip(arr * (1.0 - mask_3), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_crumple(img: Image.Image) -> Image.Image:
    """
    Draw thick crease lines with gaussian blur falloff for visibility at 512px.

    Strategy: draw thick lines on a mask at full resolution, then gaussian-blur
    the mask and multiply against the image. This produces crease-like darkening
    that survives downscaling to 512px feature extraction.
    """
    w, h = img.size
    arr = np.array(img.convert("RGB"), dtype=np.float64)

    # Build a crease mask: 0 = no crease, 1 = full darkening
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    n_creases = random.randint(5, 12)
    for _ in range(n_creases):
        # Crease goes edge-to-edge or edge-to-interior
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            x1, y1 = random.randint(0, w), 0
        elif edge == "bottom":
            x1, y1 = random.randint(0, w), h - 1
        elif edge == "left":
            x1, y1 = 0, random.randint(0, h)
        else:
            x1, y1 = w - 1, random.randint(0, h)

        # End point: middle 60% of image
        x2 = random.randint(int(w * 0.2), int(w * 0.8))
        y2 = random.randint(int(h * 0.2), int(h * 0.8))

        thickness = random.randint(15, 35)
        # Use intermediate gray (128) — will scale intensity after blur
        draw.line([(x1, y1), (x2, y2)], fill=128, width=thickness)

        # Sometimes add a second segment (crease network)
        if random.random() < 0.5:
            x3 = random.randint(0, w)
            y3 = random.randint(0, h)
            draw.line([(x2, y2), (x3, y3)], fill=128,
                      width=max(8, thickness - random.randint(5, 20)))

    # Gaussian blur for smooth falloff (σ=8 for thick lines)
    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.GaussianBlur(radius=8.0))

    # Convert mask to float and scale intensity
    mask_arr = np.array(mask, dtype=np.float64) / 255.0
    darken = random.uniform(0.4, 0.75)
    mask_arr *= darken

    # Apply to all channels
    arr *= (1.0 - np.stack([mask_arr, mask_arr, mask_arr], axis=-1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_variants(images_dir: str, out_dir: str, ds_name: str,
                      n_shadow: int, n_crumple: int) -> list:
    """Generate shadow and crumple variants from existing receipts."""
    os.makedirs(out_dir, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg"}
    src_files = [f for f in sorted(os.listdir(images_dir))
                 if os.path.splitext(f)[1].lower() in exts]
    if not src_files:
        return []
    random.seed(SEED)
    random.shuffle(src_files)
    rows = []
    for kind, count, fn_aug in [
        ("shadow", n_shadow, augment_shadow),
        ("crumple", n_crumple, augment_crumple),
    ]:
        for i, fname in enumerate(src_files[:count]):
            src = os.path.join(images_dir, fname)
            dest_fname = f"{ds_name}_{kind}_{fname}"
            dest = os.path.join(out_dir, dest_fname)
            if not os.path.exists(dest):
                try:
                    img = Image.open(src).convert("RGB")
                    img = fn_aug(img)
                    img.save(dest)
                except Exception:
                    continue
            # is_document=1, is_paper=1, plus shadow/crumple flag
            label = [1, 0, 1, 0, 0]
            if kind == "shadow":
                label[4] = 1
            else:
                label[3] = 1
            rows.append([f"variants/{dest_fname}"] + label)
    print(f"  Generated {len(rows)} variants from {ds_name}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build labels.csv for wasm-doc-boost")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--n-negatives", type=int, default=2000)
    parser.add_argument("--n-shadow", type=int, default=500)
    parser.add_argument("--n-crumple", type=int, default=500)
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    variants_dir = os.path.join(images_dir, "variants")
    negatives_dir = os.path.join(images_dir, "negatives")
    os.makedirs(images_dir, exist_ok=True)

    all_rows = []
    random.seed(SEED)

    # Standard datasets
    for ds_name, label in DATASET_LABELS.items():
        ds_dir = os.path.join(images_dir, ds_name)
        if not os.path.isdir(ds_dir):
            print(f"  SKIP {ds_name}: directory not found")
            continue
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted(f for f in os.listdir(ds_dir)
                       if os.path.splitext(f)[1].lower() in exts)
        for fname in files:
            all_rows.append([f"{ds_name}/{fname}", *label])
        print(f"  {ds_name}: {len(files)} images → {label}")

    # Generate negatives
    neg_rows = generate_negatives(negatives_dir, args.n_negatives)
    all_rows.extend(neg_rows)

    # Generate shadow/crumple variants from CORD-v2
    for ds_name in ["cord-v2", "srd"]:
        ds_dir = os.path.join(images_dir, ds_name)
        if os.path.isdir(ds_dir):
            var_rows = generate_variants(ds_dir, variants_dir, ds_name,
                                         args.n_shadow // 2, args.n_crumple // 2)
            all_rows.extend(var_rows)

    # Shuffle and write
    random.shuffle(all_rows)
    csv_path = os.path.join(data_dir, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)
        writer.writerows(all_rows)

    print(f"\n=== labels.csv ===")
    print(f"Total: {len(all_rows)} rows")
    arr = np.array([r[1:] for r in all_rows], dtype=np.int8)
    for i, name in enumerate(LABEL_NAMES):
        pos = arr[:, i].sum()
        neg = len(arr) - pos
        print(f"  {name}: {pos} positive, {neg} negative ({pos/len(arr)*100:.1f}%)")
    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
