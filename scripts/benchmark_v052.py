#!/usr/bin/env python3
"""v0.5.2 benchmark: cardinal recovery + small-tilt recovery + plain-text
check + latency, then write data/metrics_v052.json (goal-gate format).

Run AFTER training:
  PYTHONPATH=/tmp/smoke python3 scripts/benchmark_v052.py \
      --model data/model.json --report data/report.json
"""
import argparse, json, os, time

import numpy as np
from PIL import Image
import py_features

MAX_LONG_EDGE = 512
TILTS = [-12, -8, -5, -3, -1.5, -1, -0.5, 0.5, 1, 1.5, 3, 5, 8, 12]


def cerr(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='data/model.json')
    ap.add_argument('--report', default='data/report.json')
    ap.add_argument('--cardinal-n', type=int, default=300)
    ap.add_argument('--tilt-n', type=int, default=100)
    ap.add_argument('--out', default='data/metrics_v052.json')
    args = ap.parse_args()

    py_features.load_model(open(args.model).read())

    # ---- latency ----
    raw = open('data/images/fatura2/fatura2_test_0.jpg', 'rb').read()
    px512, w512, h512 = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        py_features.predict(px512, w512, h512)
        ts.append((time.perf_counter() - t0) * 1000.0)
    predict_ms = float(np.median(ts))
    print(f'predict_ms median: {predict_ms:.1f}')

    # ---- cardinal recovery ----
    card = {'n': 0, 'pm1': 0, 'acc_flip': 0, 'review': 0}
    for i in range(args.cardinal_n):
        raw = open(f'data/images/fatura2/fatura2_test_{i}.jpg', 'rb').read()
        px, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(px, np.uint8).reshape(h, w, 3)
        base = Image.fromarray(rgb, 'RGB')
        for angle in (0, 90, 180, 270):
            im = base if angle == 0 else base.rotate(angle, expand=True, resample=Image.BICUBIC)
            pxr = np.asarray(im, np.uint8).tobytes()
            r = py_features.predict(pxr, im.size[0], im.size[1])
            err = cerr(r['rotation_deg'], angle)
            card['n'] += 1
            card['pm1'] += err <= 1.0
            if r['needs_review']:
                card['review'] += 1
            elif err > 90.0:
                card['acc_flip'] += 1
    cardinal_pm1 = card['pm1'] / card['n']
    accepted_flip = card['acc_flip'] / (card['n'] - card['review'])
    print(f'cardinal: n={card["n"]} pm1={cardinal_pm1:.4f} review={card["review"]/card["n"]:.3f} accepted_flip={accepted_flip:.5f}')

    # ---- small tilt ----
    tilt = {'n': 0, 'w1': 0, 'gt10': 0}
    for i in range(args.tilt_n):
        raw = open(f'data/images/fatura2/fatura2_test_{i}.jpg', 'rb').read()
        px, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(px, np.uint8).reshape(h, w, 3)
        base = Image.fromarray(rgb, 'RGB')
        for t in TILTS:
            if abs(t) <= 0.5:
                continue
            im = base.rotate(t, expand=True, resample=Image.BICUBIC)
            pxr = np.asarray(im, np.uint8).tobytes()
            r = py_features.predict(pxr, im.size[0], im.size[1])
            residual = abs(r['skew_deg'] + t)
            tilt['n'] += 1
            tilt['w1'] += residual <= 1.0
            tilt['gt10'] += residual > 10.0
    tilt_within1 = tilt['w1'] / tilt['n']
    tilt_gt10 = tilt['gt10'] / tilt['n']
    print(f'tilt: n={tilt["n"]} within1={tilt_within1:.4f} gt10_rate={tilt_gt10:.4f}')

    # ---- plain-text (textonly negatives must be digital) ----
    dig = []
    for i in range(100):
        raw = open(f'data/images/textonly/text_{i:05d}.png', 'rb').read()
        px, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        r = py_features.predict(px, w, h)
        dig.append(r['is_digital'])
    print(f'plain-text is_digital: mean={np.mean(dig):.3f} min={np.min(dig):.3f}')

    # ---- doc traits from train report ----
    report = json.load(open(args.report))
    doc_traits = {f'{name}_f1': report['test'][name]['f1'] for name in
                  ['is_document', 'is_digital', 'is_paper', 'is_crumpled', 'is_shadow']}

    metrics = {
        'rotation': {
            'fatura2_cardinal_pm1_acc': cardinal_pm1,
            'fatura2_tilt_within1': tilt_within1,
            'fatura2_tilt_gt10_rate': tilt_gt10,
            'fatura2_accepted_flip_rate': accepted_flip,
        },
        'doc_traits': doc_traits,
        'predict_ms_per_image': predict_ms,
    }
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
