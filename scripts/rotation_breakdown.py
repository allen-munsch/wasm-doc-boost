#!/usr/bin/env python3
"""Per-dataset rotation breakdown for the SHIPPED Rust inference.

Runs py_features.predict (the exact backend twin of the frontend's
classify_file: same features-core extraction, same gbdt-core forest, same
glyph gate + confidence calibration) over the held-out test split of a v2
features npz, reconstructing each rotated sample from its base image exactly
as export_worker.py did. Numbers here match the shipped WASM bit-for-bit.

The Python-only mirror in train_model.combine_rotation_py does NOT have
the glyph gate (glyph crops aren't stored in the npz), so its rotation
numbers differ from shipped inference; use this script for reporting.

Usage:
    python scripts/rotation_breakdown.py \
        --features data/features_v6.npz --model data/model_v6.json \
        --images data/images --out /tmp/test_preds_v6_rust.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import py_features

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_model as t

MAX_LONG_EDGE = 512
BORDER_FILL_WIDTH = 4


def median_border_color(rgb, border=BORDER_FILL_WIDTH):
    a = np.asarray(rgb)
    top = a[:border].reshape(-1, 3)
    bottom = a[-border:].reshape(-1, 3)
    left = a[border:-border, :border].reshape(-1, 3)
    right = a[border:-border, -border:].reshape(-1, 3)
    border_px = np.concatenate([top, bottom, left, right], axis=0)
    return tuple(int(v) for v in np.median(border_px, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_v6.npz")
    ap.add_argument("--model", default="data/model_v6.json")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="/tmp/test_preds_v6_rust.npz")
    args = ap.parse_args()

    X, y, filenames, rotation_deg = t.load_data(args.features)
    _, _, test_idx = t.split_groups(X, y, filenames, rotation_deg)
    yt = y[test_idx]
    fnst = filenames[test_idx]
    rott = rotation_deg[test_idx]
    print(f"test samples: {len(test_idx)}", flush=True)

    py_features.load_model(open(args.model).read())

    base = np.array([str(f).rsplit("_rot", 1)[0] for f in fnst])
    src_down = {}
    for b, flip in zip(base, yt[:, 5].astype(bool)):
        if b not in src_down:
            src_down[b] = flip
    src_down = np.array([src_down[b] for b in base])
    true_rot = (rott + 180.0 * src_down) % 360.0
    is_doc = yt[:, 0].astype(bool)

    rot_pred = np.empty(len(fnst))
    conf = np.empty(len(fnst))
    review = np.empty(len(fnst), dtype=bool)
    probs = np.empty((len(fnst), len(t.LABEL_NAMES)))

    t0 = time.time()
    n = len(fnst)
    for i in range(n):
        img_path = os.path.join(args.images, base[i])
        with open(img_path, "rb") as fh:
            raw = fh.read()
        pixels, rw, rh = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(rh, rw, 3)
        img = Image.fromarray(rgb, "RGB")
        angle = float(rott[i])
        if angle != 0.0:
            fill = median_border_color(img)
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=fill)
        rw, rh = img.size
        px = np.asarray(img, dtype=np.uint8).tobytes()
        r = py_features.predict(px, rw, rh)
        rot_pred[i] = r["rotation_deg"]
        conf[i] = r["rotation_confidence"]
        review[i] = bool(r["needs_review"])
        for j, name in enumerate(t.LABEL_NAMES):
            probs[i, j] = r[name]
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{n} ({el:.0f}s, {el / (i + 1) * n - el:.0f}s left)", flush=True)

    err = t.circular_err(rot_pred, true_rot, 360.0)
    flip_pred = (rot_pred >= 180.0).astype(bool)
    flip_gt = (true_rot >= 180.0).astype(bool)
    dss = np.array([str(f).split("/")[0] for f in fnst])

    print("\n=== Per-dataset rotation (test, docs only) ===")
    print("dataset     n_doc  pm1_acc  flip_acc  mae_deg  review%  mean_conf")
    for d in sorted(set(dss)):
        m = (dss == d) & is_doc
        if m.sum() == 0:
            continue
        print(f"{d:10s} {m.sum():7d} {(err[m] <= 1.0).mean():8.4f} "
              f"{(flip_pred[m] == flip_gt[m]).mean():9.4f} {err[m].mean():7.2f} "
              f"{review[m].mean() * 100:7.1f} {conf[m].mean():9.3f}")
    m = is_doc
    print(f"{'ALL docs':10s} {m.sum():7d} {(err[m] <= 1.0).mean():8.4f} "
          f"{(flip_pred[m] == flip_gt[m]).mean():9.4f} {err[m].mean():7.2f} "
          f"{review[m].mean() * 100:7.1f} {conf[m].mean():9.3f}")

    print("\n=== Trait heads (test) ===")
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    for j, name in enumerate(t.LABEL_NAMES[:5]):
        p = probs[:, j]
        yv = yt[:, j]
        yp = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yv, yp, average="binary", zero_division=0)
        print(f"  {name}: AUC={roc_auc_score(yv, p):.4f} P={pr:.4f} R={rc:.4f} F1={f1:.4f}")

    np.savez(args.out, rot_pred=rot_pred, true_rot=true_rot, err=err, is_doc=is_doc,
             filenames=fnst, flip_gt=flip_gt, flip_pred=flip_pred, conf=conf,
             review=review, probs=probs, dss=dss)
    print(f"\nSaved per-sample test predictions -> {args.out}")


if __name__ == "__main__":
    main()
