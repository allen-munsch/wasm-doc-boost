#!/usr/bin/env python3
"""Train the per-glyph 0-vs-180 flip classifier + page-level calibration.

Operator's idea, now against the Rust pipeline itself: invoices are heavy on
all-caps and lining digits, whose ascender/descender ink mass is ~0, so
flip_score and vote_frac are coin flips there. A small logistic classifier on
20x20 glyph crops only has to be modestly above chance per glyph; summing
log-odds over the page's glyphs shrinks the page-level error with glyph count.

Crops come from py_features.extract_glyph_crops — the SAME Rust deskew,
Sauvola binarization, line-band and glyph-run slicing that inference uses, so
no Python mirror can drift from the Rust crop definition.

Split is template-grounded (data/fatura2_template_ids.json): whole templates
stay together in train/test so shared fonts/layouts cannot inflate held-out
numbers. Outputs the weights to bake into crates/features-core/src/glyph.rs.

Usage:
    python scripts/train_glyph_classifier.py --n-train 4000 --n-test 400 \
        --cache data/glyph_crops_cache.npz --out data/glyph_weights.json
"""

import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import py_features

MAX_LONG_EDGE = 512
TEMPLATE_IDS_PATH = "data/fatura2_template_ids.json"


def base_filename(fn: str) -> str:
    return fn.rsplit("_rot", 1)[0]


def template_groups(filenames):
    tmap = None
    if os.path.exists(TEMPLATE_IDS_PATH):
        with open(TEMPLATE_IDS_PATH) as f:
            tmap = json.load(f)
    groups = []
    for f in filenames:
        key = os.path.basename(f)
        if tmap and key in tmap:
            groups.append(f"tpl{tmap[key]:02d}")
        else:
            groups.append(key)
    return np.asarray(groups)


def load_rgb(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    pixels, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)
    return rgb


def extract_crops(rgb):
    h, w = rgb.shape[:2]
    crops = py_features.extract_glyph_crops(rgb.tobytes(), w, h)
    if not crops:
        return np.zeros((0, 400), dtype=np.float32)
    return np.asarray(crops, dtype=np.float32)


def page_logodds(crops, coef, intercept):
    if crops.shape[0] == 0:
        return 0.0, 0
    z = crops.astype(np.float64) @ coef + intercept
    return float(z.sum()), int(crops.shape[0])


def _extract_one(p):
    rgb = load_rgb(p)
    up = extract_crops(rgb)
    fl = extract_crops(rgb[::-1, ::-1])  # exact 180 flip
    return p, up, fl


def load_all(images, jobs=1):
    """Return {path: (upright_crops, flipped_crops)} as (N,400) float32 arrays."""
    out = {}
    if jobs <= 1:
        for i, p in enumerate(images):
            out[p] = _extract_one(p)[1:]
            if (i + 1) % 50 == 0:
                print(f"    extracted {i + 1}/{len(images)}", flush=True)
        return out
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        for i, (p, up, fl) in enumerate(pool.imap_unordered(_extract_one, images)):
            out[p] = (up, fl)
            if (i + 1) % 50 == 0:
                print(f"    extracted {i + 1}/{len(images)}", flush=True)
    return out


def build_glyph_dataset(image_crops, paths, cap_glyphs):
    Xs, ys = [], []
    for p in paths:
        up, fl = image_crops[p]
        Xs.append(up)
        ys.append(np.zeros(up.shape[0], dtype=np.int8))
        Xs.append(fl)
        ys.append(np.ones(fl.shape[0], dtype=np.int8))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if cap_glyphs and len(X) > cap_glyphs:
        idx = np.random.RandomState(0).choice(len(X), cap_glyphs, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images/fatura2")
    ap.add_argument("--pattern", default="fatura2_train_*.jpg")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--cap-glyphs", type=int, default=120000)
    ap.add_argument("--cache", default="data/glyph_crops_cache.npz")
    ap.add_argument("--out", default="data/glyph_weights.json")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.images}/{args.pattern}"))
    random.Random(0).shuffle(paths)
    n_total = args.n_train + args.n_test
    paths = paths[:n_total]
    print(f"{len(paths)} images ({args.n_train} train + {args.n_test} test budget)")

    groups = template_groups(paths)
    test_frac = args.n_test / n_total
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
    tr_idx, te_idx = next(gss.split(paths, groups=groups))
    train_paths = [paths[i] for i in tr_idx]
    test_paths = [paths[i] for i in te_idx]
    print(
        f"train {len(train_paths)} images ({len(set(groups[tr_idx]))} tpl), "
        f"test {len(test_paths)} images ({len(set(groups[te_idx]))} tpl)"
    )

    all_paths = train_paths + test_paths
    if os.path.exists(args.cache):
        cache = np.load(args.cache)
        image_crops = {p: (cache[f"{i}_up"], cache[f"{i}_fl"]) for i, p in enumerate(all_paths)}
        print(f"loaded crops cache ({args.cache})")
    else:
        image_crops = load_all(all_paths, jobs=args.jobs)
        arr = {}
        for i, p in enumerate(all_paths):
            arr[f"{i}_up"] = image_crops[p][0]
            arr[f"{i}_fl"] = image_crops[p][1]
        np.savez_compressed(args.cache, **arr)
        print(f"wrote crops cache ({args.cache})")

    # Per-glyph logistic on train glyphs.
    Xtr, ytr = build_glyph_dataset(image_crops, train_paths, args.cap_glyphs)
    Xte, yte = build_glyph_dataset(image_crops, test_paths, args.cap_glyphs)
    print(f"glyphs: {len(Xtr)} train, {len(Xte)} test")

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=1)
    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"per-glyph: accuracy={acc * 100:.2f}%  AUC={auc:.4f}  (chance 50%)")

    # Page-level evaluation with the per-glyph model.
    coef = model.coef_[0].astype(np.float64)
    intercept = float(model.intercept_[0])

    def page_results(paths):
        rows = []
        for p in paths:
            up, fl = image_crops[p]
            lo_u, n_u = page_logodds(up, coef, intercept)
            lo_f, n_f = page_logodds(fl, coef, intercept)
            rows.append((lo_u, n_u, 0))
            rows.append((lo_f, n_f, 1))
        return rows

    test_rows = page_results(test_paths)
    correct_u = sum(1 for lo, n, lab in test_rows if lab == 0 and lo < 0)
    correct_f = sum(1 for lo, n, lab in test_rows if lab == 1 and lo > 0)
    n_u = sum(1 for lo, n, lab in test_rows if lab == 0)
    n_f = sum(1 for lo, n, lab in test_rows if lab == 1)
    print(f"page-level log-odds (uncalibrated): upright {correct_u}/{n_u}, "
          f"flipped {correct_f}/{n_f}")

    # Page-level calibration via 5-fold CV over TRAIN images: per-fold per-glyph
    # model fit on the other folds, so per-page scores are out-of-fold.
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=5)
    cv_rows = []
    for fold, (fit_idx, val_idx) in enumerate(gkf.split(train_paths, groups=groups[tr_idx])):
        fit_paths = [train_paths[i] for i in fit_idx]
        val_paths = [train_paths[i] for i in val_idx]
        Xf, yf = build_glyph_dataset(image_crops, fit_paths, args.cap_glyphs)
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=1)
        m.fit(Xf, yf)
        cf = m.coef_[0].astype(np.float64)
        ic = float(m.intercept_[0])
        for p in val_paths:
            up, fl = image_crops[p]
            lo_u, n_u = page_logodds(up, cf, ic)
            lo_f, n_f = page_logodds(fl, cf, ic)
            cv_rows.append((lo_u, n_u, 0))
            cv_rows.append((lo_f, n_f, 1))
        print(f"    fold {fold} done", flush=True)

    cv = np.asarray(cv_rows)
    L = cv[:, 0].astype(np.float64)
    N = cv[:, 1].astype(np.float64)
    ylab = cv[:, 2].astype(np.float64)
    cal = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    cal.fit(np.column_stack([L, N]), ylab)
    b0 = float(cal.intercept_[0])
    b1 = float(cal.coef_[0][0])
    b2 = float(cal.coef_[0][1])
    print(f"page calibration: P(flipped)=sigmoid({b0:.6f} + {b1:.6f}*L + {b2:.6f}*N)")

    # Calibrated page decision on the held-out test images.
    def calibrated(lo, n):
        z = b0 + b1 * lo + b2 * n
        return 1.0 / (1.0 + np.exp(-z))

    correct = 0
    total = 0
    for lo, n, lab in test_rows:
        p = calibrated(lo, n)
        pred = 1 if p >= 0.5 else 0
        correct += pred == lab
        total += 1
    print(f"page-level calibrated accuracy: {correct}/{total} = {correct / total * 100:.1f}%")

    # Dump weights.
    out = {
        "glyph_intercept": intercept,
        "glyph_coefs": model.coef_[0].tolist(),
        "cal_b0": b0,
        "cal_b1": b1,
        "cal_b2": b2,
        "per_glyph_auc": auc,
        "per_glyph_accuracy": acc,
        "page_accuracy": correct / total,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    # Rust consts snippet for pasting into glyph.rs / gbdt-core.
    coefs_rs = ", ".join(f"{v:.9}" for v in model.coef_[0])
    print("\n// glyph.rs")
    print(f"pub const GLYPH_INTERCEPT: f32 = {intercept:.9}f32;")
    print(f"pub const GLYPH_COEFS: [f32; GLYPH_DIM] = [{coefs_rs}];")
    print("\n// gbdt-core/src/lib.rs")
    print(f"pub const GLYPH_CAL_B0: f64 = {b0:.9};")
    print(f"pub const GLYPH_CAL_B1: f64 = {b1:.9};")
    print(f"pub const GLYPH_CAL_B2: f64 = {b2:.9};")


if __name__ == "__main__":
    main()
