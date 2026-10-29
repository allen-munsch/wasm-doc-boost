#!/usr/bin/env python3
"""Probe: are the 8 new flip features (116-123) redundant with flip_score (104)
and vote_frac (105)? Extract 124 feats on a sample of fatura2 at angle 0 and 180,
report Pearson correlations and partial correlation with the flip label."""
import numpy as np
import py_features
from pathlib import Path
import sys

MAX_EDGE = 512
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
imgs = sorted(Path('data/images/fatura2').glob('fatura2_train_*.jpg'))
import random
random.Random(0).shuffle(imgs)
imgs = imgs[:N]

idx = [104, 105, 116, 117, 118, 119, 120, 121, 122, 123]
names = ['flip_score', 'vote_frac', 'signed_vert_grad', 'baseline_edge_weight', 'gy_skew',
         'top_bottom_edge', 'char_centroid_offset', 'char_topheavy', 'char_asym_strength', 'body_vert_bias']

rows = []
labels = []
for p in imgs:
    px, w, h = py_features.load_rgb_resized(p.read_bytes(), MAX_EDGE)
    a = np.frombuffer(px, np.uint8).reshape(h, w, 3)
    for rot, flip in ((0, 1), (180, 0)):
        arr = a if rot == 0 else a[::-1, ::-1, :]
        f = py_features.extract_all(arr.tobytes(), w, h)
        rows.append(f)
        labels.append(flip)

F = np.array(rows)  # (2N, 124)
L = np.array(labels, dtype=float)
print('N samples:', F.shape[0])
print('mean per-feature (104,105,116-123):')
print([f'{F[:, i].mean():.4f}' for i in idx])
print()

# full correlation matrix
sel = np.column_stack([F[:, i] for i in idx] + [L])
C = np.corrcoef(sel, rowvar=False)
cols = names + ['flip_label']
print(f'{"":22s}' + ''.join(f'{c:>13s}' for c in cols))
for r, cname in enumerate(cols):
    print(f'{cname:22s}' + ''.join(f'{C[r, c]:13.3f}' for c in range(len(cols))))
print()

# partial correlation with flip label, controlling for 104+105
def partial_corr(x, y, Z):
    Z1 = np.column_stack([np.ones(len(y)), Z])
    b_x = np.linalg.lstsq(Z1, x, rcond=None)[0]
    b_y = np.linalg.lstsq(Z1, y, rcond=None)[0]
    rx = x - Z1 @ b_x
    ry = y - Z1 @ b_y
    return np.corrcoef(rx, ry)[0, 1]

print('Per new feature (116-123): corr with flip_score / vote_frac / flip_label / partial(control 104,105)')
for i, nm in zip(idx[2:], names[2:]):
    c104 = np.corrcoef(F[:, i], F[:, 104])[0, 1]
    c105 = np.corrcoef(F[:, i], F[:, 105])[0, 1]
    cL = np.corrcoef(F[:, i], L)[0, 1]
    pL = partial_corr(F[:, i], L, F[:, [104, 105]])
    print(f'{nm:24s}  vs104={c104:+.3f}  vs105={c105:+.3f}  vsFlip={cL:+.3f}  partial={pL:+.3f}')
