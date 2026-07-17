#!/usr/bin/env python3
"""End-to-end GBDT classifier evaluation: py-features extraction + model.json inference.

Usage: python scripts/eval_python.py [--sample N] [--workers N] [--labels data/labels_small.csv]
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from py_features import extract_all

ROOT = Path(__file__).resolve().parent.parent
LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]

# ── Global state (set once per worker process via initializer) ──

_model_trees = None
_trees_per_label = None


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _init_worker(model_path: str, tpl: int):
    global _model_trees, _trees_per_label
    with open(model_path) as f:
        raw = json.load(f)
    _model_trees = [tree[0] for tree in raw]
    _trees_per_label = tpl


def _predict_one(trees, tree_idx: int, features: list[float]) -> float:
    node = trees[tree_idx]
    while "leaf" not in node:
        feat_idx = int(node["split"][1:])
        if features[feat_idx] < node["split_condition"]:
            node = node["children"][0]
        else:
            node = node["children"][1]
    return node["leaf"]


def _classify_chunk(chunk: list[tuple[str, list[int]]]) -> dict:
    """Process a chunk of (filename, labels) tuples. Returns per-label confusion + error count."""
    confusion = [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for _ in LABEL_NAMES]
    errors = 0
    images_dir = ROOT / "data" / "images"

    for filename, actual_labels in chunk:
        img_path = images_dir / filename
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            errors += 1
            for j in range(len(LABEL_NAMES)):
                if actual_labels[j] == 0:
                    confusion[j]["tn"] += 1
                else:
                    confusion[j]["fn"] += 1
            continue

        pixels = np.array(img).tobytes()
        w, h = img.size

        try:
            features = extract_all(pixels, w, h)
        except Exception:
            errors += 1
            for j in range(len(LABEL_NAMES)):
                if actual_labels[j] == 0:
                    confusion[j]["tn"] += 1
                else:
                    confusion[j]["fn"] += 1
            continue

        # Sum logits per label, apply sigmoid
        preds = []
        for label_idx in range(len(LABEL_NAMES)):
            tpl = _trees_per_label or 0
            start = label_idx * tpl
            end = start + tpl
            logit = sum(_predict_one(_model_trees, i, features) for i in range(start, end))
            preds.append(sigmoid(logit))

        for j in range(len(LABEL_NAMES)):
            pred = 1 if preds[j] >= 0.5 else 0
            actual = actual_labels[j]
            if pred == 1 and actual == 1:
                confusion[j]["tp"] += 1
            elif pred == 1 and actual == 0:
                confusion[j]["fp"] += 1
            elif pred == 0 and actual == 0:
                confusion[j]["tn"] += 1
            else:
                confusion[j]["fn"] += 1

    return {"confusion": confusion, "errors": errors, "classified": len(chunk)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--labels", default="data/labels_small.csv")
    args = parser.parse_args()

    labels_path = ROOT / args.labels
    model_path = ROOT / "data" / "model.json"

    # Load model once just to get tree count
    with open(model_path) as f:
        raw = json.load(f)
    total_trees = len(raw)
    assert total_trees % len(LABEL_NAMES) == 0
    trees_per_label = total_trees // len(LABEL_NAMES)
    print(f"Model: {total_trees} trees ({trees_per_label} per label), {args.workers} workers")

    # Parse labels
    with open(labels_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines[0].startswith("filename"):
        lines = lines[1:]

    samples = []
    for line in lines:
        parts = line.split(",")
        samples.append((parts[0], [int(p) for p in parts[1:]]))

    eval_set = samples
    if args.sample > 0 and args.sample < len(samples):
        by_label = defaultdict(list)
        for s in samples:
            by_label[tuple(s[1])].append(s)
        eval_set = []
        for group in by_label.values():
            take = max(1, int(args.sample * len(group) / len(samples)))
            eval_set.extend(group[:take])
        eval_set = eval_set[: args.sample]

    # Split into chunks (one per worker)
    chunk_size = max(1, len(eval_set) // args.workers)
    chunks = [eval_set[i : i + chunk_size] for i in range(0, len(eval_set), chunk_size)]
    print(f"Evaluating {len(eval_set)} images in {len(chunks)} chunks")

    # ── Run in parallel ──
    t0 = time.monotonic()
    confusion = [{k: 0 for k in ("tp", "fp", "tn", "fn")} for _ in LABEL_NAMES]
    total_errors = 0
    total_classified = 0
    completed = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(model_path), trees_per_label),
    ) as pool:
        for result in pool.map(_classify_chunk, chunks):
            completed += 1
            total_classified += result["classified"]
            total_errors += result["errors"]
            for j in range(len(LABEL_NAMES)):
                for k in ("tp", "fp", "tn", "fn"):
                    confusion[j][k] += result["confusion"][j][k]
            elapsed = time.monotonic() - t0
            rate = total_classified / elapsed if elapsed > 0 else 0
            print(f"  chunk {completed}/{len(chunks)} — {total_classified} done ({elapsed:.1f}s, {rate:.0f} img/s)")

    elapsed = time.monotonic() - t0
    print(f"Done. {total_classified} classified, {total_errors} errors in {elapsed:.1f}s ({total_classified/elapsed:.0f} img/s)")

    # Metrics
    print(f"\n=== Python GBDT Classification Evaluation ===")
    print(f"Images: {len(eval_set)}, Errors: {total_errors}, Time: {elapsed:.1f}s\n")

    report = {}
    macro_p, macro_r, macro_f1 = 0, 0, 0
    for j, name in enumerate(LABEL_NAMES):
        c = confusion[j]
        support = c["tp"] + c["fn"]
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        recall = c["tp"] / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (c["tp"] + c["tn"]) / support if support > 0 else 1.0
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "support": support,
            **c,
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        print(f"  {name:<14} P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  (n={support})")

    macro_p /= len(LABEL_NAMES)
    macro_r /= len(LABEL_NAMES)
    macro_f1 /= len(LABEL_NAMES)
    report["macro_avg"] = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
    }
    print(f"  {'macro_avg':<14} P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")

    out_path = ROOT / "data" / "python_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
