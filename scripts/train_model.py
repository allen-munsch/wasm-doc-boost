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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

LABEL_NAMES = ["is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow"]
TREE_PARAMS = {
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

        model = xgb.XGBClassifier(
            tree_method="hist",
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            max_depth=TREE_PARAMS["max_depth"],
            learning_rate=TREE_PARAMS["learning_rate"],
            n_estimators=TREE_PARAMS["n_estimators"],
            subsample=TREE_PARAMS["subsample"],
            colsample_bytree=TREE_PARAMS["colsample_bytree"],
            min_child_weight=TREE_PARAMS["min_child_weight"],
            reg_alpha=TREE_PARAMS["reg_alpha"],
            reg_lambda=TREE_PARAMS["reg_lambda"],
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
    args = parser.parse_args()

    X, y, filenames = load_data(args.features)
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
