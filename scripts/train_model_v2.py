#!/usr/bin/env python3
"""Train the per-degree rotation rework model (v2).

Input:  features npz from `export_parallel.py --v2` (116 features, 7 labels,
        plus rotation_deg metadata).
Output: model.json (v3 manifest `{labels, tau, version, trees, trees_per_label}`,
        7 heads: 5 doc traits + flip + trust), report.json.

Rotation is NOT a GBDT head: the axis is the deterministic radon feature
(index 100). The GBDT only predicts flip (P(flip)) and trust
(P(radon axis within 1 deg)). Combined in wasm-bridge as
    axis_deg = degrees(radon_axis) mod 180; snapped to 0 when <1 or >179
    rotation_deg = (180 - axis_deg) mod 180 + 180 * (flip >= 0.5)
    confidence  = trust * max(flip, 1 - flip)
    needs_review = confidence < tau (tau calibrated here) or not is_document.

Usage:
    python scripts/train_model_v2.py --features data/features_v5.npz \
        --output data/model_v3.json --report data/report_v3.json
"""

import argparse
import json
import os
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

LABEL_NAMES = [
    "is_document",
    "is_digital",
    "is_paper",
    "is_crumpled",
    "is_shadow",
    "flip",
    "trust",
]
RADON_AXIS_INDEX = 100
TRUST_TOLERANCE_DEG = 1.0

BASE_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
}
LABEL_PARAMS: dict[str, dict[str, Any]] = {
    "is_crumpled": {"max_depth": 3, "n_estimators": 80, "reg_alpha": 0.5,
                     "reg_lambda": 5.0, "min_child_weight": 3, "subsample": 0.7,
                     "colsample_bytree": 0.7},
    "is_shadow": {"max_depth": 3, "n_estimators": 80, "reg_alpha": 0.5,
                   "reg_lambda": 5.0, "min_child_weight": 3, "subsample": 0.7,
                   "colsample_bytree": 0.7},
    "flip": {"max_depth": 4, "n_estimators": 160},
    "trust": {"max_depth": 4, "n_estimators": 160},
}


def _params_for(name: str) -> dict[str, Any]:
    p = dict(BASE_PARAMS)
    p.update(LABEL_PARAMS.get(name, {}))
    return p


def base_filename(fn: str) -> str:
    return fn.rsplit("_rot", 1)[0]


def load_data(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["features"]
    y = data["labels"]
    filenames = np.asarray(data["filenames"])
    rotation_deg = np.asarray(data["rotation_deg"], dtype=np.float64)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features, {y.shape[1]} labels")
    for i, name in enumerate(LABEL_NAMES):
        pos = y[:, i].sum()
        print(f"  {name}: {pos} positive ({pos / len(y) * 100:.1f}%)")
    return X, y, filenames, rotation_deg


def split_groups(X, y, filenames, rotation_deg):
    groups = np.array([base_filename(f) for f in filenames])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, rest_idx = next(gss1.split(X, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
    cal_idx, test_idx = next(gss2.split(rest_idx, groups=groups[rest_idx]))
    cal_idx, test_idx = rest_idx[cal_idx], rest_idx[test_idx]
    print(f"Train: {len(train_idx)}, Cal: {len(cal_idx)}, Test: {len(test_idx)} (by base image)")
    return train_idx, cal_idx, test_idx


def train_models(X_train, y_train):
    models = []
    for i, name in enumerate(LABEL_NAMES):
        pos = y_train[:, i].sum()
        neg = len(y_train) - pos
        scale_pos_weight = min(neg / max(pos, 1), 20.0) if pos > 0 else 1.0
        params = _params_for(name)
        model = xgb.XGBClassifier(
            tree_method="hist",
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train[:, i])
        models.append(model)
        print(f"  {name}: {model.n_estimators} trees, pos_weight={scale_pos_weight:.1f}")
    return models


def predict(models, X):
    return [m.predict_proba(X)[:, 1] for m in models]


def circular_err(a, b, period=360.0):
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def source_down_by_base(filenames, y):
    """Per-sample bool: is the base image stored 180-deg flipped?

    With the corrected flip label (flip = text upside down after deskew),
    the flip label at angle 0 IS the source orientation: 1 = source stored
    upside down. Used to turn the applied angle back into the true
    content rotation (applied_angle + 180 * source_down) mod 360.
    """
    base = np.array([f.rsplit("_rot", 1)[0] for f in filenames])
    src_down = {}
    for b, flip in zip(base, y[:, 5].astype(bool)):
        if b not in src_down:
            src_down[b] = flip
    return np.array([src_down[b] for b in base])


def rotation_metrics(X, y, rotation_deg, P, tag, filenames=None):
    axis_deg = np.degrees(X[:, RADON_AXIS_INDEX]) % 180.0
    flip_p = P[5]
    trust_p = P[6]
    flip_pred = (flip_p >= 0.5).astype(int)
    # axis is the radon line direction mod 180 = -applied_angle mod 180,
    # so the applied angle is (180 - axis) mod 180 plus 180 when flipped.
    rot_pred = ((180.0 - axis_deg) % 180.0) + 180.0 * flip_pred
    # True content rotation: applied angle + 180 if the source is stored
    # upside down (the corrected flip label carries that orientation).
    src_down = (source_down_by_base(filenames, y) if filenames is not None
                else np.zeros(len(y), dtype=bool))
    true_rot = (rotation_deg + 180.0 * src_down) % 360.0
    err = circular_err(rot_pred, true_rot)

    true_axis = (180.0 - rotation_deg) % 180.0
    axis_err = circular_err(axis_deg, true_axis, period=180.0)

    is_doc = y[:, 0].astype(bool)
    out = {
        "strict_mae_deg_all": float(np.mean(err)),
        "strict_mae_deg_docs": float(np.mean(err[is_doc])),
        "pm1_accuracy_all": float(np.mean(err <= 1.0)),
        "pm1_accuracy_docs": float(np.mean(err[is_doc] <= 1.0)),
        "axis_only_accuracy_1deg": float(np.mean(axis_err <= 1.0)),
        "accepted_flip_rate_strict": float(np.mean(err > 90.0)),
        "flip_accuracy": float(np.mean(flip_pred == y[:, 5])),
        "trust_auc": float(roc_auc_score(y[:, 6], trust_p)),
        "flip_auc": float(roc_auc_score(y[:, 5], flip_p)),
    }
    print(f"\n=== Rotation metrics [{tag}] ===")
    for k, v in out.items():
        print(f"  {k}: {v:.4f}")
    return out


def trait_metrics(y, P, tag):
    metrics = {}
    print(f"\n=== Trait metrics [{tag}] ===")
    for i, name in enumerate(LABEL_NAMES[:5]):
        yp = (P[i] >= 0.5).astype(int)
        auc = roc_auc_score(y[:, i], P[i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[:, i], yp, average="binary", zero_division=0)
        metrics[name] = {"auc": float(auc), "precision": float(precision),
                         "recall": float(recall), "f1": float(f1)}
        print(f"  {name}: AUC={auc:.4f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    return metrics


def calibrate_tau(X_cal, y_cal, rotation_cal, P_cal, filenames_cal=None):
    """Sweep tau over combined confidence = trust * max(flip, 1-flip)."""
    axis_deg = np.degrees(X_cal[:, RADON_AXIS_INDEX]) % 180.0
    flip_p = P_cal[5]
    trust_p = P_cal[6]
    flip_pred = (flip_p >= 0.5).astype(int)
    rot_pred = ((180.0 - axis_deg) % 180.0) + 180.0 * flip_pred
    src_down = (source_down_by_base(filenames_cal, y_cal) if filenames_cal is not None
                else np.zeros(len(y_cal), dtype=bool))
    true_rot = (rotation_cal + 180.0 * src_down) % 360.0
    err = circular_err(rot_pred, true_rot)
    conf = trust_p * np.maximum(flip_p, 1.0 - flip_p)

    rows = []
    print("\n=== Tau sweep (cal) ===")
    for tau in np.arange(0.0, 1.001, 0.05):
        tau_f = float(tau)
        attempted = conf >= tau
        review_rate = float(1.0 - attempted.mean())
        if attempted.sum() == 0:
            rows.append({"tau": tau_f, "review_rate": review_rate, "accepted_flip_rate": 0.0,
                         "mae_attempted": float("inf"), "coverage": 0.0})
            continue
        accepted_flip = float(np.mean(err[attempted] > 90.0))
        mae = float(np.mean(err[attempted]))
        rows.append({"tau": tau_f, "review_rate": review_rate,
                     "accepted_flip_rate": accepted_flip, "mae_attempted": mae,
                     "coverage": float(attempted.mean())})
        print(f"  tau={tau:.2f} review={review_rate*100:5.1f}% "
              f"accepted_flip={accepted_flip*100:.3f}% mae={mae:.3f}")

    # largest tau with accepted-flip <= 0.1% (goal 0) and review rate <= 20%
    chosen = None
    for r in rows:
        if r["accepted_flip_rate"] <= 0.001 and r["review_rate"] <= 0.20:
            chosen = r
    return rows, chosen


def export_merged_model(models, output_path, tau):
    all_trees = []
    trees_per_label = []
    for model in models:
        tree_dump = model.get_booster().get_dump(dump_format="json")
        all_trees.extend([json.loads(t)] for t in tree_dump)
        trees_per_label.append(len(tree_dump))
    # v3 manifest: labels + pinned operating point + tree forest + exact
    # per-label tree counts. The seven heads use uneven n_estimators
    # (200/200/200/80/80/160/160), so wasm-bridge must NOT split `trees`
    # evenly; trees_per_label tells it the exact boundaries. `version: 3`
    # is reserved for the fold-snap combine in wasm-bridge (model_version());
    # that combine change is verified numerically but not yet applied.
    manifest = {
        "labels": LABEL_NAMES,
        "tau": tau,
        "version": 3,
        "trees": all_trees,
        "trees_per_label": trees_per_label,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nExported {len(all_trees)} trees -> {output_path} ({size_kb:.0f} KB), "
          f"tau={tau}, trees_per_label={trees_per_label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="data/model_v3.json")
    parser.add_argument("--report", default="data/report_v3.json")
    args = parser.parse_args()

    X, y, filenames, rotation_deg = load_data(args.features)
    train_idx, cal_idx, test_idx = split_groups(X, y, filenames, rotation_deg)

    print("\n=== Training ===")
    models = train_models(X[train_idx], y[train_idx])

    P_train = predict(models, X[train_idx])
    P_cal = predict(models, X[cal_idx])
    P_test = predict(models, X[test_idx])

    report = {
        "train": trait_metrics(y[train_idx], P_train, "train"),
        "cal": trait_metrics(y[cal_idx], P_cal, "cal"),
        "test": trait_metrics(y[test_idx], P_test, "test"),
        "rotation_train": rotation_metrics(X[train_idx], y[train_idx], rotation_deg[train_idx], P_train, "train", filenames[train_idx]),
        "rotation_cal": rotation_metrics(X[cal_idx], y[cal_idx], rotation_deg[cal_idx], P_cal, "cal", filenames[cal_idx]),
        "rotation_test": rotation_metrics(X[test_idx], y[test_idx], rotation_deg[test_idx], P_test, "test", filenames[test_idx]),
    }

    rows, chosen = calibrate_tau(X[cal_idx], y[cal_idx], rotation_deg[cal_idx], P_cal, filenames[cal_idx])
    report["tau_sweep"] = rows
    report["chosen_tau"] = chosen
    if chosen is not None:
        print(f"\nChosen tau = {chosen['tau']:.2f} "
              f"(accepted_flip={chosen['accepted_flip_rate']*100:.3f}%, "
              f"review={chosen['review_rate']*100:.1f}%)")
        pinned_tau = float(chosen["tau"])
    else:
        print("\nNo tau met the accepted-flip <= 0.1% + review <= 20% bar "
              "(Path B trigger: quantized vision encoder)")
        pinned_tau = 0.90
        print(f"Pinning conservative tau={pinned_tau} so the WASM manifest stays loadable")
    report["pinned_tau"] = pinned_tau

    export_merged_model(models, args.output, pinned_tau)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {args.report}")


if __name__ == "__main__":
    main()
