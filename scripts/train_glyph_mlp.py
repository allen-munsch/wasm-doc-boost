#!/usr/bin/env python3
"""Train a small 2-layer MLP (400 -> 64 -> 1) glyph-orientation classifier on
BALANCED sources: fatura2 all-caps (reduced), DocLayNet mixed-case text,
synthetic text-only, CORD-v2 digits. Early-stopping stops when the validation
curve flattens, so we do not pay for unproductive epochs.

Outputs data/glyph_mlp_weights.json: w1 (64x400 row-major), b1 (64), w2 (64),
b2 (1), plus the train/val curve.
"""
import argparse, glob, json, os, random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import py_features

MAX_LONG_EDGE = 512


def load_rgb(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    px, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
    return np.frombuffer(px, np.uint8).reshape(h, w, 3)


def extract_crops(rgb):
    h, w = rgb.shape[:2]
    c = py_features.extract_glyph_crops(rgb.tobytes(), w, h)
    return np.asarray(c, dtype=np.float32) if c else np.zeros((0, 400), np.float32)


def tilt_blur_augment(X, y, angles=(0.5, 1.0), sigmas=(0.5, 1.0)):
    from scipy.ndimage import gaussian_filter, rotate as ndi_rotate
    grids = X.reshape(-1, 20, 20)
    Xs, ys = [X], [y]
    for s in sigmas:
        bl = np.stack([gaussian_filter(g, s, mode='constant', cval=0.0) for g in grids])
        Xs.append(bl.reshape(-1, 400)); ys.append(y)
    for a in angles:
        for sg in (+1.0, -1.0):
            r = np.stack([ndi_rotate(g, sg * a, reshape=False, order=1, mode='constant', cval=0.0) for g in grids])
            Xs.append(r.reshape(-1, 400)); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def source_crops(paths, cap):
    up, fl = [], []
    n = 0
    for p in paths:
        rgb = load_rgb(p)
        u = extract_crops(rgb)
        f = extract_crops(rgb[::-1, ::-1])
        up.append(u); fl.append(f)
        n += len(u) + len(f)
        if n >= 2 * cap:  # stop once we have enough per orientation
            break
    up = np.concatenate(up) if up else np.zeros((0, 400), np.float32)
    fl = np.concatenate(fl) if fl else np.zeros((0, 400), np.float32)
    up = up[:cap]; fl = fl[:cap]
    return up, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fatura-cap', type=int, default=20000)
    ap.add_argument('--doclaynet-cap', type=int, default=15000)
    ap.add_argument('--textonly-cap', type=int, default=7500)
    ap.add_argument('--cord-cap', type=int, default=7500)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--out', default='data/glyph_mlp_weights.json')
    args = ap.parse_args()

    pic = json.load(open('data/doclaynet_picture.json'))
    fatura = sorted(glob.glob('data/images/fatura2/fatura2_train_*.jpg'))
    random.Random(0).shuffle(fatura)
    dl = [f'data/images/doclaynet/{b}' for b in sorted(os.listdir('data/images/doclaynet'))
          if b.endswith('.png') and pic.get(b, {}).get('is_document', 1) == 1]
    textonly = sorted(glob.glob('data/images/textonly/*.png'))
    cord = sorted(glob.glob('data/images/cord-v2/*.jpg'))

    X, y = [], []
    for paths, cap, name in [(fatura, args.fatura_cap, 'fatura2'),
                             (dl, args.doclaynet_cap, 'doclaynet'),
                             (textonly, args.textonly_cap, 'textonly'),
                             (cord, args.cord_cap, 'cord-v2')]:
        up, fl = source_crops(paths, cap)
        X.append(up); y.append(np.zeros(len(up), np.int8))
        X.append(fl); y.append(np.ones(len(fl), np.int8))
        print(f'{name}: {len(up)} upright + {len(fl)} flipped', flush=True)
    X = np.concatenate(X); y = np.concatenate(y)
    print(f'total base crops: {len(X)}', flush=True)

    # train/test split (per-image is hard here; simple stratified on crops)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    Xtr, ytr = tilt_blur_augment(Xtr, ytr)
    print(f'augmented train: {len(Xtr)}', flush=True)

    model = MLPClassifier(hidden_layer_sizes=(args.hidden,), activation='tanh',
                          solver='adam', alpha=1e-3, max_iter=200, batch_size=256,
                          early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
                          random_state=42, verbose=True)
    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f'MLP: test accuracy={acc*100:.2f}% AUC={auc:.4f}', flush=True)

    w1 = model.coefs_[0].astype(np.float32)   # (400, hidden)
    b1 = model.intercepts_[0].astype(np.float32)
    w2 = model.coefs_[1].astype(np.float32)   # (hidden, 1)
    b2 = model.intercepts_[1].astype(np.float32)
    # row-major [hidden][400] for the no_std scorer
    w1_rm = w1.T.ravel().tolist()
    out = {
        'hidden': args.hidden,
        'w1': w1_rm, 'b1': b1.ravel().tolist(),
        'w2': w2.ravel().tolist(), 'b2': b2.ravel().tolist(),
        'test_accuracy': acc, 'test_auc': auc,
        'curve': {'train_loss': list(model.loss_curve_),
                  'val_score': list(model.validation_scores_),
                  'n_iter': int(model.n_iter_)},
    }
    json.dump(out, open(args.out, 'w'))
    print(f'wrote {args.out} (hidden={args.hidden}, n_iter={model.n_iter_})', flush=True)


if __name__ == '__main__':
    main()
