#!/usr/bin/env python3
"""Train the tiny GBDT flip-decision combiner: 6 raw flip signals -> P(flipped).

Signals: [glyph_logodds, glyph_count, vote_frac, flip_score, flip_p, trust_p]
(matches combine_rotation's available evidence). A depth-3, ~32-tree GBDT so it
scores through the SAME tree infra as the 7 heads.

Outputs data/flip_combiner_trees.json: xgboost JSON tree dumps + feature names.
"""
import argparse, json, os, random
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from PIL import Image
import py_features
import sys
sys.path.insert(0, 'scripts')
import train_model as t

MAX_LONG_EDGE = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-flip', type=int, default=2500)
    ap.add_argument('--trees', type=int, default=32)
    ap.add_argument('--out', default='data/flip_combiner_trees.json')
    args = ap.parse_args()

    py_features.load_model(open('data/model.json').read())
    X, y, fn, rot = t.load_data('data/features_v9.npz')
    train_idx, cal_idx, test_idx = t.split_groups(X, y, fn, rot)

    def train_head(col, name):
        pos = y[train_idx, col].sum(); neg = len(train_idx) - pos
        p = t._params_for(name)
        m = xgb.XGBClassifier(device='cuda', tree_method='hist', objective='binary:logistic',
            scale_pos_weight=min(neg / max(pos, 1), 20.0), max_depth=p['max_depth'],
            learning_rate=p['learning_rate'], n_estimators=p['n_estimators'],
            subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
            min_child_weight=p['min_child_weight'], reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
            eval_metric='auc', random_state=42, n_jobs=-1)
        m.fit(X[train_idx], y[train_idx, col]); return m

    print('training flip + trust heads for their signals...', flush=True)
    flip_head = train_head(5, 'flip'); trust_head = train_head(6, 'trust')

    m0 = (np.abs(rot[train_idx] - 0) < 0.5) & (y[train_idx, 5] == 0)
    m180 = (np.abs(rot[train_idx] - 180) < 0.5) & (y[train_idx, 5] == 1)
    idx = train_idx[m0][:args.n_per_flip].tolist() + train_idx[m180][:args.n_per_flip].tolist()
    random.Random(0).shuffle(idx)

    rows = []
    for i in idx:
        base = fn[i].rsplit('_rot', 1)[0]
        p = f'data/images/{base}'
        if not os.path.exists(p):
            continue
        raw = open(p, 'rb').read(); px, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(px, np.uint8).reshape(h, w, 3)
        if abs(rot[i] - 180) < 0.5:
            im = Image.fromarray(rgb).rotate(180, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
            a = np.asarray(im, np.uint8); px2, w2, h2 = a.tobytes(), im.size[0], im.size[1]
        else:
            px2, w2, h2 = px, w, h
        gll, gcnt = py_features.extract_glyph_logodds(px2, w2, h2)
        feat = py_features.extract_all(px2, w2, h2)
        flip_p = float(flip_head.predict_proba([feat])[0][1])
        trust_p = float(trust_head.predict_proba([feat])[0][1])
        rows.append((gll, gcnt, feat[105], feat[104], flip_p, trust_p, y[i, 5]))

    R = np.array(rows)
    S, Y = R[:, :6], R[:, 6]
    perm = np.random.RandomState(42).permutation(len(R)); S, Y = S[perm], Y[perm]
    n = int(0.8 * len(R))

    c = xgb.XGBClassifier(device='cuda', tree_method='hist', objective='binary:logistic',
        max_depth=3, learning_rate=0.1, n_estimators=args.trees, subsample=0.8, colsample_bytree=1.0,
        min_child_weight=3, reg_alpha=0.5, reg_lambda=1.0, eval_metric='auc', random_state=42)
    c.fit(S[:n], Y[:n])
    p = c.predict_proba(S[n:])[:, 1]
    print(f'combiner: acc={accuracy_score(Y[n:], (p >= .5).astype(int)):.4f} '
          f'auc={roc_auc_score(Y[n:], p):.4f} trees={len(c.get_booster().get_dump())}', flush=True)
    json.dump({'trees': c.get_booster().get_dump(dump_format='json'),
               'features': ['glyph_logodds', 'glyph_count', 'vote_frac', 'flip_score', 'flip_p', 'trust_p']},
              open(args.out, 'w'))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
