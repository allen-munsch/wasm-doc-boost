#!/usr/bin/env python3
"""Hstack features_v6.npz with the incremental CORU export into features_v7.npz."""
import sys
import numpy as np

V6 = "data/features_v6.npz"
CORU = "data/features_coru_v2.npz"
OUT = "data/features_v7.npz"

d6 = np.load(V6, allow_pickle=True)
dc = np.load(CORU, allow_pickle=True)
print(f"v6: {d6['features'].shape}, coru: {dc['features'].shape}")
assert d6["features"].shape[1] == dc["features"].shape[1], "feature width mismatch"
assert d6["labels"].shape[1] == dc["labels"].shape[1], "label width mismatch"

features = np.concatenate([d6["features"], dc["features"]], axis=0)
labels = np.concatenate([d6["labels"], dc["labels"]], axis=0)
filenames = np.concatenate([d6["filenames"], dc["filenames"]], axis=0)
rotation_deg = np.concatenate([d6["rotation_deg"], dc["rotation_deg"]], axis=0)

np.savez_compressed(
    OUT,
    features=features,
    labels=labels,
    filenames=filenames,
    rotation_deg=rotation_deg,
    label_names=np.array([
        "is_document", "is_digital", "is_paper", "is_crumpled", "is_shadow",
        "flip", "trust",
    ]),
)
print(f"wrote {OUT}: {features.shape}, base imgs ~ {len(set(f.rsplit('_rot',1)[0] for f in filenames))}")
