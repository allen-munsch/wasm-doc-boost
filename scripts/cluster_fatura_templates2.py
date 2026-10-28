#!/usr/bin/env python3
"""Recover FATURA2 template ids via KMeans on a coarse 2D text-occupancy grid.
Robust to variable table length (only cell occupancy matters). Saves
filename -> template_id, and reports train/test template overlap."""
import pyarrow.parquet as pq
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import json

COLS, ROWS = 30, 42  # 595x841 -> ~20px cells

def occupancy(bboxes):
    grid = np.zeros((ROWS, COLS), dtype=np.float32)
    for b in bboxes:
        x1, y1, x2, y2 = b
        c1, c2 = int(x1 * COLS / 595), min(COLS - 1, int((x2 - 1) * COLS / 595))
        r1, r2 = int(y1 * ROWS / 841), min(ROWS - 1, int((y2 - 1) * ROWS / 841))
        for r in range(max(0, r1), min(ROWS, r2 + 1)):
            for c in range(max(0, c1), min(COLS, c2 + 1)):
                grid[r, c] = 1.0
    return grid.ravel()

rows = []
for split, path in (('train', 'data/fatura2/data/train-00000-of-00001.parquet'),
                    ('test', 'data/fatura2/data/test-00000-of-00001.parquet')):
    t = pq.read_table(path)
    bboxes = t['bboxes'].to_pylist()
    for i, bl in enumerate(bboxes):
        rows.append((split, i, occupancy(bl)))

X = np.array([r[2] for r in rows], dtype=np.float32)
print('X shape', X.shape)

km = KMeans(n_clusters=50, n_init=10, random_state=0, max_iter=300).fit(X)
labels = km.labels_
print('cluster sizes:', sorted(Counter(labels).values()))

# validation: exact (x,y,y2)-signature group members should share a cluster
GRID = 4
sig_members = defaultdict(list)
for split, path in (('train', 'data/fatura2/data/train-00000-of-00001.parquet'),
                    ('test', 'data/fatura2/data/test-00000-of-00001.parquet')):
    t = pq.read_table(path)
    for i, bl in enumerate(t['bboxes'].to_pylist()):
        s = frozenset((round(b[0] / GRID), round(b[1] / GRID), round(b[3] / GRID)) for b in bl)
        sig_members[s].append((split, i))

# map (split,i) -> cluster label
row_lab = {(split, i): lab for (split, i, _), lab in zip(rows, labels)}
agree = 0
tot = 0
for s, mem in sig_members.items():
    labs = {row_lab[m] for m in mem}
    if len(labs) == 1:
        agree += len(mem)
    tot += len(mem)
print(f'exact-signature members sharing one cluster: {agree}/{tot} = {agree/tot:.3f}')

# save mapping
fn_map = {f'fatura2_{split}_{i}.jpg': int(lab) for (split, i, _), lab in zip(rows, labels)}
json.dump(fn_map, open('data/fatura2_template_ids.json', 'w'))
print('saved data/fatura2_template_ids.json')

# train/test template overlap (operator: both splits share all 50 templates)
tr = Counter(lab for (split, i, _), lab in zip(rows, labels) if split == 'train')
te = Counter(lab for (split, i, _), lab in zip(rows, labels) if split == 'test')
shared = len(set(tr) & set(te))
print('templates present in both train and test:', shared, '/', len(set(labels)))
