#!/usr/bin/env python3
"""Prototype: per-glyph 0-vs-180 classifier, summed as page-level log-odds.

Operator's idea, tested before touching Rust: invoices are heavy on all-caps
and lining digits, whose ascender/descender mass is ~0, so flip_score and
vote_frac (asc/desc ink) are coin flips there. A small learned 0-vs-180
classifier on individual glyph crops only has to be modestly above chance;
summing log-odds over the page's glyphs makes the page-level error shrink as
the glyph count grows.

Pure-Python mirror of the Rust pipeline (textline.rs): gray -> Sauvola local
binarize (NOT global Otsu) -> line bands via row projection -> glyph runs via
column projection inside each band -> 20x20 normalized glyph crop. Images are
split (not glyphs) so the classifier must generalize across templates/fonts.

Usage:
    python scripts/prototype_glyph_classifier.py --n 300
"""

import argparse
import glob
import random

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

GLYPH = 20
MAX_EDGE = 512


def sauvola_binarize(gray: np.ndarray, w: int = 25, k: float = 0.2, r: float = 128.0):
    """Sauvola local threshold; ink=True (dark). Mirrors textline.rs."""
    g = gray.astype(np.float64)
    mean = ndimage.uniform_filter(g, size=w)
    sq = ndimage.uniform_filter(g * g, size=w)
    var = np.maximum(sq - mean * mean, 0.0)
    std = np.sqrt(var)
    thr = mean * (1.0 + k * (std / r - 1.0))
    return g < thr


def line_bands(mask: np.ndarray):
    rowink = mask.sum(axis=1)
    bands = []
    y = 0
    h = mask.shape[0]
    while y < h:
        while y < h and rowink[y] < 2:
            y += 1
        if y >= h:
            break
        y0 = y
        while y < h and rowink[y] >= 2:
            y += 1
        y1 = y
        if y1 - y0 >= 6:
            bands.append((y0, y1))
    return bands


def crop_glyph(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    sub = mask[y0:y1, x0:x1].astype(np.float32)
    h, w = sub.shape
    s = max(h, w)
    if s < 1:
        return None
    out = np.zeros((s, s), np.float32)
    oy = (s - h) // 2
    ox = (s - w) // 2
    out[oy : oy + h, ox : ox + w] = sub
    img = Image.fromarray((out * 255.0).astype(np.uint8))
    img = img.resize((GLYPH, GLYPH), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32).ravel()


def extract_glyphs(mask: np.ndarray):
    """Return list of flattened 20x20 glyph crops (mirrors char_flip_features slicing)."""
    glyphs = []
    for y0, y1 in line_bands(mask):
        band = mask[y0:y1]
        colink = band.sum(axis=0)
        x = 0
        w = mask.shape[1]
        while x < w:
            while x < w and colink[x] == 0:
                x += 1
            if x >= w:
                break
            x0 = x
            while x < w and colink[x] > 0:
                x += 1
            x1 = x
            ink = band[:, x0:x1].sum()
            if ink >= 6:
                g = crop_glyph(mask, x0, x1, y0, y1)
                if g is not None:
                    glyphs.append(g)
            x = x1
    return glyphs


def load_gray(path: str):
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = MAX_EDGE / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def page_logodds(model, glyphs):
    if not glyphs:
        return 0.0, 0
    X = np.stack(glyphs)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)).sum()), len(glyphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--images", default="data/images/fatura2")
    ap.add_argument("--pattern", default="fatura2_train_*.jpg")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.images}/{args.pattern}"))
    random.Random(0).shuffle(paths)
    paths = paths[: args.n]
    n_train = int(len(paths) * 0.6)
    train_paths, test_paths = paths[:n_train], paths[n_train:]
    print(f"{len(train_paths)} train images, {len(test_paths)} test images")

    def build(paths, label, out_X, out_y):
        for p in paths:
            gray = load_gray(p)
            mask = sauvola_binarize(gray)
            if label == 1:
                mask = mask[::-1, ::-1]  # exact 180 flip
            for g in extract_glyphs(mask):
                out_X.append(g)
                out_y.append(label)

    Xtr, ytr, Xte, yte = [], [], [], []
    build(train_paths, 0, Xtr, ytr)
    build(train_paths, 1, Xtr, ytr)
    build(test_paths, 0, Xte, yte)
    build(test_paths, 1, Xte, yte)

    Xtr = np.asarray(Xtr, np.float32)
    ytr = np.asarray(ytr)
    Xte = np.asarray(Xte, np.float32)
    yte = np.asarray(yte)
    print(f"glyphs: {len(Xtr)} train, {len(Xte)} test")

    model = LogisticRegression(C=1.0, max_iter=1000, n_jobs=1)
    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"per-glyph: accuracy={acc * 100:.2f}%  AUC={auc:.4f}  (chance 50%)")

    # page-level: sum log-odds over each test image's glyphs, both orientations
    correct_u = 0
    correct_f = 0
    n_u = n_f = 0
    glyph_counts = []
    for p in test_paths:
        gray = load_gray(p)
        mask = sauvola_binarize(gray)
        lo_u, n_u_g = page_logodds(model, extract_glyphs(mask))
        lo_f, n_f_g = page_logodds(model, extract_glyphs(mask[::-1, ::-1]))
        if n_u_g == 0:
            continue
        glyph_counts.append(n_u_g)
        n_u += 1
        n_f += 1
        correct_u += lo_u < 0   # P(flipped) < P(upright) => log-odds negative
        correct_f += lo_f > 0   # flipped page => log-odds positive
    print(f"page-level log-odds: upright {correct_u}/{n_u} correct, "
          f"flipped {correct_f}/{n_f} correct")
    if n_u:
        print(f"page-level accuracy: {(correct_u + correct_f) / (n_u + n_f) * 100:.1f}% "
              f"(mean {np.mean(glyph_counts):.0f} glyphs/page)")


if __name__ == "__main__":
    main()
