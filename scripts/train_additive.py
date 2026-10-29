#!/usr/bin/env python3
"""Additive, checkpointed retrain of the 7 GBDT heads, with a learning-curve
sweep to find the data cutoff.

Reuses train_model.py's split (textonly train-only fix), per-head params,
threshold selection, calibration, and manifest writer. Adds:

  * Phase A -- learning curve: for each doc-trait head, train on an increasing
    fraction of TRAIN GROUPS (templates stay together) and record test F1, so
    we can read off the data cutoff where F1 stops improving.
  * Phase B -- additive final train: train each head once, checkpoint its trees
    to data/checkpoints/head-{i}-{name}.json as soon as it finishes, then merge
    into the model manifest. Resume skips heads whose checkpoint exists.

Usage:
    python scripts/train_additive.py --features data/features_v9.npz \\
        --output data/model_v9b.json --report data/report_v9b.json \\
        --device cuda --curve-fractions 0.05 0.1 0.2 0.35 0.5 0.7 1.0
"""
import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_model as t


def train_one_head(X, y, name, device="cpu"):
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = min(neg / max(pos, 1), 20.0) if pos > 0 else 1.0
    params = t._params_for(name)
    model = xgb.XGBClassifier(
        device=device,
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
    model.fit(t._head_X(X, name), y)
    return model


def head_trees(model):
    return [json.loads(d) for d in model.get_booster().get_dump(dump_format="json")]


def subset_train_groups(train_idx, groups, frac, seed=0):
    """Pick a fraction of TRAIN GROUPS (templates kept whole)."""
    g = groups[train_idx]
    ug = np.unique(g)
    rng = np.random.RandomState(seed)
    if frac >= 1.0:
        return train_idx
    chosen = rng.choice(len(ug), max(1, int(len(ug) * frac)), replace=False)
    keep = np.isin(g, ug[chosen])
    return train_idx[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", default="data/model_v9b.json")
    ap.add_argument("--report", default="data/report_v9b.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    ap.add_argument("--curve", default="data/learning_curve.json")
    ap.add_argument("--curve-fractions", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    ap.add_argument("--final-fraction", type=float, default=1.0)
    ap.add_argument("--skip-curve", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    X, y, filenames, rotation_deg = t.load_data(args.features)
    train_idx, cal_idx, test_idx = t.split_groups(X, y, filenames, rotation_deg)
    groups = t._template_groups(filenames)

    # ---- Phase A: learning curve ----
    curve = {}
    if not args.skip_curve:
        from sklearn.metrics import f1_score
        print("\n=== Phase A: learning curve (train-group fraction -> test F1) ===")
        for i, name in enumerate(t.LABEL_NAMES[:5]):
            pts = []
            for frac in args.curve_fractions:
                sub = subset_train_groups(train_idx, groups, frac)
                m = train_one_head(X[sub], y[sub][:, i], name, args.device)
                p = m.predict_proba(t._head_X(X[test_idx], name))[:, 1]
                f1 = f1_score(y[test_idx][:, i], (p >= 0.5).astype(int))
                pts.append({"fraction": frac, "train_groups": len(np.unique(groups[sub])),
                            "train_n": len(sub), "test_f1": float(f1)})
                print(f"  {name}: frac={frac:.2f} groups={len(np.unique(groups[sub]))} "
                      f"n={len(sub)} test_f1={f1:.4f}", flush=True)
            curve[name] = pts
        with open(args.curve, "w") as f:
            json.dump(curve, f, indent=2)
        print(f"wrote {args.curve}")

    # ---- Phase B: additive final train with per-head checkpoints ----
    print(f"\n=== Phase B: final train at fraction {args.final_fraction} ===")
    sub = subset_train_groups(train_idx, groups, args.final_fraction)
    Xtr, ytr = X[sub], y[sub]
    models = []
    for i, name in enumerate(t.LABEL_NAMES):
        cp = os.path.join(args.checkpoint_dir, f"head-{i}-{name}.json")
        if os.path.exists(cp):
            trees = json.load(open(cp))
            booster = xgb.Booster()
            booster.load_model(bytearray("\n".join(
                json.dumps(tr) if isinstance(tr, dict) else tr for tr in trees
            ), "utf-8"))
            # wrap in a minimal object exposing predict_proba + get_booster
            class _W:
                def __init__(self, b): self._b = b
                def predict_proba(self, Xm):
                    d = xgb.DMatrix(Xm)
                    p = self._b.predict(d)
                    return np.column_stack([1 - p, p])
                def get_booster(self): return self._b
            models.append(_W(booster))
            print(f"  {name}: RESUMED from {cp}", flush=True)
            continue
        m = train_one_head(Xtr, ytr[:, i], name, args.device)
        models.append(m)
        json.dump(head_trees(m), open(cp, "w"))
        print(f"  {name}: trained + checkpointed -> {cp}", flush=True)

    # ---- evaluate + thresholds + calibration (reuse trainer) ----
    P_train = t.predict(models, X[sub])
    P_cal = t.predict(models, X[cal_idx])
    P_test = t.predict(models, X[test_idx])
    trait_thresholds = t.select_trait_thresholds(y[cal_idx], P_cal)
    report = {
        "train": t.trait_metrics(y[sub], P_train, "train", trait_thresholds),
        "cal": t.trait_metrics(y[cal_idx], P_cal, "cal", trait_thresholds),
        "test": t.trait_metrics(y[test_idx], P_test, "test", trait_thresholds),
        "rotation_train": t.rotation_metrics(X[sub], y[sub], rotation_deg[sub], P_train,
                                              "train", filenames[sub]),
        "rotation_cal": t.rotation_metrics(X[cal_idx], y[cal_idx], rotation_deg[cal_idx],
                                           P_cal, "cal", filenames[cal_idx]),
        "rotation_test": t.rotation_metrics(X[test_idx], y[test_idx], rotation_deg[test_idx],
                                            P_test, "test", filenames[test_idx]),
    }
    rows, chosen, calibration = t.calibrate_confidence(
        X[cal_idx], y[cal_idx], rotation_deg[cal_idx], P_cal, filenames[cal_idx]
    )
    report["tau_sweep"] = rows
    report["chosen_tau"] = chosen
    report["confidence_calibration"] = calibration
    report["pinned_tau"] = float(chosen["tau"]) if chosen is not None else 0.9
    t.export_merged_model(models, args.output, report["pinned_tau"],
                          calibration, trait_thresholds)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output} + {args.report}")


if __name__ == "__main__":
    main()
