#!/usr/bin/env python3
"""
Train a multi-label XGBoost GBDT classifier on extracted pixel features.

Input:  features.npz from export_features.py
Output: model.json (XGBoost JSON tree dump, readable by wasm-bridge)

Usage:
    python scripts/train_model.py \
        --features data/features.npz \
        --output data/model.json

Strategy:
    - Stratified 80/20 train/test split per label
    - XGBClassifier per label (binary:logistic) via MultiOutputClassifier-like approach
    - Export merged model JSON
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
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

# Per-label overrides for rare classes (prevents memorization)
LABEL_PARAMS: dict[str, dict[str, Any]] = {
    "is_crumpled": {
        "max_depth": 3,
        "n_estimators": 80,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "min_child_weight": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    },
    "is_shadow": {
        "max_depth": 3,
        "n_estimators": 80,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "min_child_weight": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    },
}


def _params_for(name: str) -> dict[str, Any]:
    """Merge base params with per-label overrides."""
    p = dict(BASE_PARAMS)
    p.update(LABEL_PARAMS.get(name, {}))
    return p


def load_data(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features + labels from NPZ."""
    data = np.load(npz_path)
    X = data["features"]
    y = data["labels"]
    filenames = data["filenames"]
    print(f"Loaded {len(X)} samples, {X.shape[1]} features, {y.shape[1]} labels")
    for i, name in enumerate(LABEL_NAMES):
        pos = y[:, i].sum()
        print(f"  {name}: {pos} positive ({pos / len(y) * 100:.1f}%)")
    return X, y, filenames


def split_data(X: np.ndarray, y: np.ndarray):
    """80/20 stratified split, trying to maintain label distribution."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")
    return X_tr, X_te, y_tr, y_te


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> list[xgb.XGBClassifier]:
    """Train one XGBClassifier per label."""
    models = []
    for i, name in enumerate(LABEL_NAMES):
        pos = y_train[:, i].sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1) if pos > 0 else 1.0
        scale_pos_weight = min(scale_pos_weight, 20.0)  # cap for extreme imbalance

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
        print(f"  {name}: {model.n_estimators} trees, "
              f"pos_weight={scale_pos_weight:.1f}")
    return models


def evaluate(models: list[xgb.XGBClassifier], X_test: np.ndarray,
             y_test: np.ndarray) -> dict[str, Any]:
    """Compute per-label AUC, precision, recall."""
    metrics = {}
    print("\n=== Evaluation ===")
    for i, name in enumerate(LABEL_NAMES):
        y_pred_proba = models[i].predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test[:, i], y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test[:, i], y_pred, average="binary", zero_division=0
        )
        metrics[name] = {"auc": auc, "precision": precision,
                         "recall": recall, "f1": f1}
        pos = y_test[:, i].sum()
        print(f"  {name}: AUC={auc:.4f}, P={precision:.3f}, R={recall:.3f}, "
              f"F1={f1:.3f} (pos={int(pos)})")
    return metrics


def kfold_cv(X: np.ndarray, y: np.ndarray, k: int = 5) -> dict[str, Any]:
    """Stratified k-fold cross-validation with train vs val comparison.

    Stratifies on label-combination keys so rare classes (crumpled, shadow)
    appear in every fold proportionally.
    """
    # Composite strata: unique label combination → class index
    label_keys = y.dot(1 << np.arange(y.shape[1])).astype(int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Per-label accumulators: train_metrics and val_metrics per fold
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, label_keys)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        print(f"\n--- Fold {fold_idx + 1}/{k}: train={len(X_tr)}, val={len(X_val)} ---")
        models = train_models(X_tr, y_tr)

        train_metrics = _eval_split(models, X_tr, y_tr, "Train")
        val_metrics = _eval_split(models, X_val, y_val, "Val")

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(X_tr),
            "val_size": len(X_val),
            "train": train_metrics,
            "val": val_metrics,
        })

    # Aggregate across folds: mean ± std per label per metric
    def _aggregate(key: str) -> dict:
        out = {}
        for name in LABEL_NAMES:
            vals = [f[key][name]["auc"] for f in fold_results]
            out[name] = {
                "auc_mean": float(np.mean(vals)),
                "auc_std": float(np.std(vals, ddof=1)),
                "auc_min": float(np.min(vals)),
                "auc_max": float(np.max(vals)),
            }
            # F5 for binary classification: (1+25)*P*R / (25*P + R)
            for metric in ["precision", "recall", "f1"]:
                vals = [f[key][name][metric] for f in fold_results]
                out[name][f"{metric}_mean"] = float(np.mean(vals))
                out[name][f"{metric}_std"] = float(np.std(vals, ddof=1))

            # F5
            f5_vals = []
            for f in fold_results:
                p = f[key][name]["precision"]
                r = f[key][name]["recall"]
                f5 = (26 * p * r) / (25 * p + r) if (p + r) > 0 else 0.0
                f5_vals.append(f5)
            out[name]["f5_mean"] = float(np.mean(f5_vals))
            out[name]["f5_std"] = float(np.std(f5_vals, ddof=1))

        return out

    train_agg = _aggregate("train")
    val_agg = _aggregate("val")

    # Overfitting gap: val - train delta per label
    print("\n=== K-Fold Summary (mean ± std) ===")
    print(f"{'Label':<14} {'Split':<6} {'AUC':<22} {'F5':<22} {'Recall':<22} {'Precision':<22}")
    print("-" * 108)
    for name in LABEL_NAMES:
        t = train_agg[name]
        v = val_agg[name]
        auc_gap = v["auc_mean"] - t["auc_mean"]
        f5_gap = v["f5_mean"] - t["f5_mean"]
        rec_gap = v["recall_mean"] - t["recall_mean"]
        print(f"{name:<14} {'Train':<6} "
              f"{t['auc_mean']:.4f}±{t['auc_std']:.4f}     "
              f"{t['f5_mean']:.4f}±{t['f5_std']:.4f}     "
              f"{t['recall_mean']:.4f}±{t['recall_std']:.4f}   "
              f"{t['precision_mean']:.4f}±{t['precision_std']:.4f}")
        print(f"{'':<14} {'Val':<6} "
              f"{v['auc_mean']:.4f}±{v['auc_std']:.4f}     "
              f"{v['f5_mean']:.4f}±{v['f5_std']:.4f}     "
              f"{v['recall_mean']:.4f}±{v['recall_std']:.4f}   "
              f"{v['precision_mean']:.4f}±{v['precision_std']:.4f}")
        gap_marker = " <-- OVERFIT?" if (rec_gap < -0.02 or f5_gap < -0.02) else ""
        print(f"{'':<14} {'Gap':<6} "
              f"{auc_gap:+.4f}             "
              f"{f5_gap:+.4f}             "
              f"{rec_gap:+.4f}           "
              f"{v['precision_mean'] - t['precision_mean']:+.4f}{gap_marker}")

    return {
        "k": k,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "folds": fold_results,
        "train_aggregate": train_agg,
        "val_aggregate": val_agg,
    }


def _eval_split(models: list[xgb.XGBClassifier], X: np.ndarray,
                 y: np.ndarray, tag: str) -> dict[str, Any]:
    """Evaluate models on a split, print summary."""
    metrics = {}
    for i, name in enumerate(LABEL_NAMES):
        y_pred_proba = models[i].predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y[:, i], y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[:, i], y_pred, average="binary", zero_division=0
        )
        metrics[name] = {"auc": auc, "precision": precision,
                         "recall": recall, "f1": f1}
    return metrics


def export_merged_model(models: list[xgb.XGBClassifier], output_path: str) -> None:
    """
    Export all per-label trees as a flat JSON array: [[tree_nodes], [tree_nodes], ...]

    Trees are ordered by label: first all is_document trees, then is_digital, etc.
    gbdt.rs::from_xgboost_json splits by total_trees / num_labels.
    """
    all_trees = []
    for model in models:
        booster = model.get_booster()
        tree_dump = booster.get_dump(dump_format="json")
        # Each tree is a single root dict; gbdt.rs expects each tree as [root_node]
        all_trees.extend([json.loads(t)] for t in tree_dump)

    with open(output_path, "w") as f:
        json.dump(all_trees, f)

    size_kb = os.path.getsize(output_path) / 1024
    total_trees = len(all_trees)
    print(f"\nExported {total_trees} trees → {output_path} ({size_kb:.0f} KB)")


def compute_feature_importance(models: list[xgb.XGBClassifier]) -> np.ndarray:
    """Average feature importance across all labels."""
    importances = np.zeros(len(LABEL_NAMES), dtype=np.float64)
    for model in models:
        importances += model.feature_importances_
    return importances / len(models)


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-label XGBoost for wasm-doc-boost")
    parser.add_argument("--features", required=True, help="features.npz path")
    parser.add_argument("--output", default="data/model.json",
                        help="Output JSON model path")
    parser.add_argument("--report", default="data/report.json",
                        help="Evaluation report path")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Run k-fold CV (e.g., --kfold 5); skips model export")
    args = parser.parse_args()

    X, y, filenames = load_data(args.features)

    if args.kfold > 0:
        print(f"\n=== {args.kfold}-Fold Cross-Validation ===")
        cv_report = kfold_cv(X, y, k=args.kfold)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(cv_report, f, indent=2)
            print(f"\nCV report written: {args.report}")
        return

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n=== Training ===")
    models = train_models(X_train, y_train)

    report = evaluate(models, X_test, y_test)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {args.report}")

    export_merged_model(models, args.output)
    print("\nDone. Next: wasm-bridge loads model.json for inference.")


if __name__ == "__main__":
    main()
