#!/usr/bin/env python3
"""
Generate synthetic labeled images for end-to-end pipeline testing.

Produces images that exercise all 5 binary labels with separable visual
characteristics, plus a labels CSV for export_features.py.

Categories:
  - digital: flat white/light bg, crisp text-like rows, solid shapes
  - paper: yellowed bg with noise, text-like rows, slight rotation
  - paper_crumpled: paper + displacement noise and crease lines
  - paper_shadow: paper + dark gradient regions
  - natural: photos of grass/sky/scene (gradient fills, noise textures)

Usage:
    python scripts/generate_synthetic_dataset.py --outdir data/synthetic/

Output:
    data/synthetic/images/    -- PNG images
    data/synthetic/labels.csv -- labels manifest
"""

import argparse
import csv
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED = 42
LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
SIZES = [(256, 256), (320, 240), (400, 300), (512, 384), (480, 360)]


def text_like_pattern(draw, w, h, color, n_rows=12, n_cols=30):
    """Draw horizontal rows of short line segments to mimic text."""
    row_h = h / (n_rows + 1)
    seg_w = w / (n_cols + 1)
    for row_i in range(1, n_rows + 1):
        y = int(row_i * row_h)
        x = int(seg_w * 0.5)
        while x < w - 10:
            length = random.randint(int(seg_w * 0.3), int(seg_w * 1.2))
            draw.line([(x, y), (x + length, y)], fill=color, width=random.randint(1, 3))
            x += length + random.randint(int(seg_w * 0.1), int(seg_w * 0.6))


def generate_digital(outdir, n=60):
    """Digital-born documents: crisp text on flat background."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for i in range(n):
        w, h = random.choice(SIZES)
        bg_color = tuple(np.random.randint(230, 256, 3).tolist())
        img = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        # Text-like content
        text_color = tuple(np.random.randint(0, 60, 3).tolist())
        text_like_pattern(draw, w, h, text_color, n_rows=random.randint(8, 18))

        # Occasional colored box or line
        if random.random() < 0.3:
            box_color = tuple(np.random.randint(100, 200, 3).tolist())
            x0, y0 = random.randint(10, w // 3), random.randint(10, h // 3)
            x1, y1 = random.randint(2 * w // 3, w - 10), random.randint(2 * h // 3, h - 10)
            draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)

        fname = f"digital_{i:04d}.png"
        img.save(os.path.join(outdir, fname))
        rows.append([fname, 1, 1, 0, 0, 0])
    return rows


def generate_paper(outdir, n=60):
    """Scanned paper: warm/yellowed bg, noise, slight artifacts."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for i in range(n):
        w, h = random.choice(SIZES)
        # Warm paper color
        base = np.random.randint(235, 252, 3).astype(np.float64)
        base = base * np.array([1.0, 0.95, 0.88])  # warm shift
        base = np.clip(base, 0, 255).astype(np.uint8)
        arr = np.full((h, w, 3), base, dtype=np.uint8)

        # Add paper texture noise
        noise = np.random.randint(-8, 9, (h, w, 3)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Text content
        text_color = tuple(np.random.randint(10, 70, 3).tolist())
        text_like_pattern(draw, w, h, text_color, n_rows=random.randint(6, 16))

        # Slight rotation via skew (optional)
        if random.random() < 0.2:
            img = img.rotate(random.uniform(-2, 2), expand=False, fillcolor=tuple(base.tolist()))

        fname = f"paper_{i:04d}.png"
        img.save(os.path.join(outdir, fname))
        rows.append([fname, 1, 0, 1, 0, 0])
    return rows


def generate_crumpled(outdir, n=30):
    """Crumpled paper: paper base + displacement + crease shadows."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for i in range(n):
        w, h = random.choice(SIZES)

        # Generate paper image first
        base = np.random.randint(230, 250, 3).astype(np.float64)
        base = base * np.array([1.0, 0.95, 0.88])
        base = np.clip(base, 0, 255).astype(np.uint8)
        arr = np.full((h, w, 3), base, dtype=np.uint8)
        noise = np.random.randint(-6, 7, (h, w, 3)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add pseudo-random crease lines (dark lines at various angles)
        n_creases = random.randint(3, 8)
        for _ in range(n_creases):
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            angle = random.uniform(0, np.pi)
            length = random.randint(min(w, h) // 3, min(w, h) // 2)
            darken = random.uniform(0.3, 0.7)
            for t in np.linspace(-length, length, max(w, h)):
                px = int(cx + t * np.cos(angle))
                py = int(cy + t * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    dist = abs(t) / length
                    fade = np.exp(-dist * 3) * darken
                    arr[py, px] = np.clip(arr[py, px] * (1 - fade), 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Text content
        text_color = tuple(np.random.randint(10, 70, 3).tolist())
        text_like_pattern(draw, w, h, text_color, n_rows=random.randint(4, 10))

        fname = f"crumpled_{i:04d}.png"
        img.save(os.path.join(outdir, fname))
        rows.append([fname, 1, 0, 1, 1, 0])
    return rows


def generate_shadow(outdir, n=30):
    """Shadow documents: paper with dark illumination gradients."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for i in range(n):
        w, h = random.choice(SIZES)

        # Paper base
        base = np.random.randint(235, 252, 3).astype(np.float64)
        base = base * np.array([1.0, 0.95, 0.88])
        base = np.clip(base, 0, 255).astype(np.uint8)
        arr = np.full((h, w, 3), base, dtype=np.uint8)

        # Add shadow gradient
        shadow_type = random.choice(["corner", "edge", "vignette"])
        shadow_strength = random.uniform(0.4, 0.8)
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")

        if shadow_type == "corner":
            mask = (xx + yy) / 2.0
        elif shadow_type == "edge":
            mask = np.minimum(xx, 1.0 - xx)
        else:  # vignette
            cx, cy = 0.5, 0.5
            mask = 1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(2)
            mask = np.clip(mask, 0, 1)

        mask = mask * shadow_strength
        mask_3 = np.stack([mask, mask, mask], axis=-1)
        arr = np.clip(arr.astype(np.float64) * (1.0 - mask_3), 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        text_color = tuple(np.random.randint(10, 70, 3).tolist())
        text_like_pattern(draw, w, h, text_color, n_rows=random.randint(6, 14))

        fname = f"shadow_{i:04d}.png"
        img.save(os.path.join(outdir, fname))
        rows.append([fname, 1, 0, 1, 0, 1])
    return rows


def generate_natural(outdir, n=80):
    """Natural images: gradients, noise textures, no text patterns."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for i in range(n):
        w, h = random.choice(SIZES)

        style = random.choice(["sky", "grass", "texture", "abstract"])
        if style == "sky":
            # Horizontal gradient blue to white
            top = np.random.randint(30, 120, 3)
            bot = np.random.randint(180, 255, 3)
            arr = np.linspace(top, bot, h, dtype=np.float64).reshape(h, 1, 3)
            arr = np.tile(arr, (1, w, 1))
        elif style == "grass":
            # Green gradient with noise
            arr = np.random.randint(0, 100, (h, w, 3)).astype(np.float64)
            arr[:, :, 1] += 80  # green channel
            arr[:, :, 0] *= 0.5
            arr[:, :, 2] *= 0.5
        elif style == "texture":
            # High-frequency noise pattern
            freq = random.uniform(0.05, 0.2)
            xx = np.linspace(0, freq * w, w)
            yy = np.linspace(0, freq * h, h)
            grid_x, grid_y = np.meshgrid(xx, yy)
            arr = (np.sin(grid_x) * np.cos(grid_y) * 127 + 128).astype(np.float64)
            arr = np.stack([arr, arr * 0.8, arr * 1.2], axis=-1)
        else:  # abstract
            arr = np.random.randint(0, 255, (h, w, 3)).astype(np.float64)
            arr = np.clip(arr, 0, 255)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        fname = f"natural_{i:04d}.png"
        img.save(os.path.join(outdir, fname))
        rows.append([fname, 0, 0, 0, 0, 0])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic labeled dataset")
    parser.add_argument("--outdir", default="data/synthetic", help="Output directory")
    args = parser.parse_args()

    outdir = args.outdir
    imgdir = os.path.join(outdir, "images")
    csvpath = os.path.join(outdir, "labels.csv")

    random.seed(SEED)
    np.random.seed(SEED)

    all_rows = []
    all_rows += generate_digital(imgdir, n=60)
    all_rows += generate_paper(imgdir, n=60)
    all_rows += generate_crumpled(imgdir, n=30)
    all_rows += generate_shadow(imgdir, n=30)
    all_rows += generate_natural(imgdir, n=80)

    random.shuffle(all_rows)

    with open(csvpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)
        writer.writerows(all_rows)

    print(f"Generated {len(all_rows)} images in {imgdir}/")
    print(f"Labels: {csvpath}")
    print("Class distribution:")
    arr = np.array([r[1:] for r in all_rows], dtype=np.int8)
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {arr[:, i].sum()} positive, {len(arr) - arr[:, i].sum()} negative")


if __name__ == "__main__":
    main()
