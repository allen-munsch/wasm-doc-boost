#!/usr/bin/env python3
"""Recover FATURA2 template identity by clustering pages on their bbox layout
signature: same template => same field positions; only word widths vary with
content. Signature = frozenset of (x1, y1, y2) rounded to a 4px grid, width
ignored. Saves data/fatura2_template_ids.json mapping filename -> template_id."""
import pyarrow.parquet as pq
from collections import Counter, defaultdict
import json

GRID = 4
splits = {}
order = []
for split, path, n in (('train', 'data/fatura2/data/train-00000-of-00001.parquet', 8600),
                       ('test', 'data/fatura2/data/test-00000-of-00001.parquet', 1400)):
    t = pq.read_table(path)
    bboxes = t['bboxes'].to_pylist()
    assert len(bboxes) == n, (split, len(bboxes), n)
    splits[split] = bboxes
    order.append((split, n))

sig = {}
rows = []
for split, bboxes in splits.items():
    for i, bl in enumerate(bboxes):
        # bbox [x1,y1,x2,y2]; keep (x1,y1,y2) on a GRID, drop width
        s = frozenset((round(b[0] / GRID), round(b[1] / GRID), round(b[3] / GRID)) for b in bl)
        sig[(split, i)] = s
        rows.append((split, i, s))

# union signatures that are >= 0.9 jaccard-similar into template clusters
sigs = list({id(s) for _, _, s in rows})  # distinct signature objects
print('distinct (x,y,y2) signatures:', len(sigs))

# greedy template assignment: a template = one canonical signature + near-matches
siglist = list(set(sigs))
parent = {s: s for s in siglist}

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

# exact-match grouping first
by_sig = defaultdict(list)
for split, i, s in rows:
    by_sig[s].append((split, i))

# count templates as exact signature groups; report the top group sizes
sizes = sorted((len(v) for v in by_sig.values()), reverse=True)
print('exact-signature groups:', len(by_sig))
print('largest 12 group sizes:', sizes[:12])
print('smallest 5 group sizes:', sizes[-5:])

# How many groups if we merge at jaccard >= 0.9? (union-find, coarse)
keys = list(by_sig.keys())
par = {k: k for k in keys}
def find(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        par[rb] = ra

# compare only keys with the same length (cheap heuristic)
bylen = defaultdict(list)
for k in keys:
    bylen[len(k)].append(k)
merged_pairs = 0
for ln, ks in bylen.items():
    for a_i in range(len(ks)):
        for b_i in range(a_i + 1, len(ks)):
            if jaccard(ks[a_i], ks[b_i]) >= 0.9:
                union(ks[a_i], ks[b_i])
                merged_pairs += 1
print('pairs merged at jaccard>=0.9:', merged_pairs)
roots = {find(k) for k in keys}
print('template clusters after 0.9 merge:', len(roots))

# assign each row a template id (root signature index) and save
root_ids = {r: idx for idx, r in enumerate(sorted(roots, key=lambda s: -len(by_sig.get(s, []))))}
fn_map = {}
for split, i, s in rows:
    tid = root_ids[find(s)]
    fn_map[f'fatura2_{split}_{i}.jpg'] = tid

out = 'data/fatura2_template_ids.json'
json.dump(fn_map, open(out, 'w'))
print('saved', out, 'with', len(fn_map), 'images,', len(roots), 'template ids')
# per-template train/test overlap summary
cnt = Counter(tid for (split, i), s in rows for tid in [root_ids[find(s)]])
print('template id counts (first 10):', sorted(cnt.items())[:10])
