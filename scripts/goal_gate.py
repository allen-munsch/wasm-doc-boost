#!/usr/bin/env python3
"""Goal gate for wasm-doc-boost v0.5.2.

Exits 0 iff:
  1. The wasm-doc-boost taskboard has zero active tasks (open/in_progress/blocked).
  2. data/metrics_v052.json exists and meets the v0.5.2 spec thresholds.

v0.5.2 is the multistage deskew (S2 fine axis + S3 refined-deskew flip evidence
+ S4 flip gate at any tilt). Rotation is therefore judged on born-digital
FATURA2 by CARDINAL recovery and by SMALL-TILT recovery (|t| in 0.5-15 deg),
not just the upright gold set ("always report 0" scores 100% there).
"""
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.expanduser("~/.nodecode/racklr-board.db")
METRICS = os.path.join(ROOT, "data", "metrics_v052.json")

BASELINE_DOC_TRAITS = {
    "is_document_f1": 0.9976,
    "is_digital_f1": 1.0000,
    "is_paper_f1": 0.9830,
    "is_crumpled_f1": 0.5558,
    "is_shadow_f1": 0.5315,
}
DOC_TRAIT_TOL = {
    "is_document_f1": 0.005,
    "is_digital_f1": 0.02,
    "is_paper_f1": 0.005,
    "is_crumpled_f1": 0.03,
    "is_shadow_f1": 0.03,
}

ROTATION_CHECKS = [
    ("fatura2_cardinal_pm1_acc", ">=", 0.95),
    ("fatura2_tilt_within1", ">=", 0.95),
    ("fatura2_tilt_gt10_rate", "<=", 0.02),
    ("fatura2_accepted_flip_rate", "<=", 0.001),
]


def board_active_count():
    con = sqlite3.connect(DB)
    try:
        return con.execute(
            "select count(*) from issues where project='wasm-doc-boost' "
            "and status in ('open','in_progress','blocked')"
        ).fetchone()[0]
    finally:
        con.close()


def main():
    problems = []
    n = board_active_count()
    if n != 0:
        problems.append(f"taskboard: {n} active wasm-doc-boost task(s)")
    if not os.path.exists(METRICS):
        problems.append("metrics: data/metrics_v052.json missing")
    else:
        with open(METRICS) as f:
            m = json.load(f)
        rot = m.get("rotation", {})
        for key, op, thresh in ROTATION_CHECKS:
            v = rot.get(key)
            if v is None:
                problems.append(f"metrics: rotation.{key} missing")
            elif op == ">=" and v < thresh:
                problems.append(f"metrics: rotation.{key}={v:.4f} < {thresh}")
            elif op == "<=" and v > thresh:
                problems.append(f"metrics: rotation.{key}={v:.4f} > {thresh}")
        dt = m.get("doc_traits", {})
        for key, base in BASELINE_DOC_TRAITS.items():
            v = dt.get(key)
            if v is None:
                problems.append(f"metrics: doc_traits.{key} missing")
            elif v < base - DOC_TRAIT_TOL[key]:
                problems.append(f"metrics: doc_traits.{key}={v:.4f} < {base}-{DOC_TRAIT_TOL[key]}")
        ms = m.get("predict_ms_per_image")
        if ms is None:
            problems.append("metrics: predict_ms_per_image missing")
        elif ms > 200.0:
            problems.append(f"metrics: predict_ms_per_image={ms:.2f} > 200.0")

    if problems:
        print("NOT MET: " + "; ".join(problems))
        return 1
    print("MET: board clear + v0.5.2 metrics within spec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
