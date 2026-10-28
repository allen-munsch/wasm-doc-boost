#!/usr/bin/env python3
"""Fine-tilt recovery benchmark for the v0.5.2 multistage deskew.

Rotates fatura2 test pages by small tilts and measures the model's residual
(skew_deg vs the applied tilt), binned like dove's deskew_small_angles.py:
|t| in (0.5,1], (1,3], (3,8], (8,15] degrees, plus the 0.5-deg dead band.

Usage:
  PYTHONPATH=/tmp/smoke python3 scripts/tilt_benchmark.py \
      --n 100 --model data/model.json --images data/images
"""
import argparse, os, time

import numpy as np
from PIL import Image
import py_features

TILTS = [-12, -8, -5, -3, -1.5, -1, -0.5, 0.5, 1, 1.5, 3, 5, 8, 12]
BINS = [('0.5-1', 0.5, 1.0), ('1-3', 1.0, 3.0), ('3-8', 3.0, 8.0), ('8-15', 8.0, 15.0)]
MAX_LONG_EDGE = 512


def cerr(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--model', default='data/model.json')
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--start', type=int, default=0)
    args = ap.parse_args()

    py_features.load_model(open(args.model).read())

    stats = {name: {'n': 0, 'w05': 0, 'w1': 0, 'w2': 0, 'gt10': 0} for name, _, _ in BINS}
    stats['dead'] = {'n': 0, 'w05': 0, 'w1': 0, 'w2': 0, 'gt10': 0}
    t0 = time.time()
    n = 0
    for i in range(args.start, args.start + args.n):
        img_path = os.path.join(args.images, 'fatura2', f'fatura2_test_{i}.jpg')
        if not os.path.exists(img_path):
            continue
        raw = open(img_path, 'rb').read()
        pixels, w, h = py_features.load_rgb_resized(raw, MAX_LONG_EDGE)
        rgb = np.frombuffer(pixels, np.uint8).reshape(h, w, 3)
        base = Image.fromarray(rgb, 'RGB')
        for tilt in TILTS:
            im = base.rotate(tilt, expand=True, resample=Image.BICUBIC)
            px = np.asarray(im, np.uint8).tobytes()
            r = py_features.predict(px, im.size[0], im.size[1])
            # skew_deg is the CW correction, so for a CCW tilt it is -tilt.
            residual = abs(r['skew_deg'] + tilt)
            gt10 = residual > 10.0
            t = abs(tilt)
            key = 'dead' if t <= 0.5 else next((k for k, lo, hi in BINS if lo < t <= hi), None)
            if key is None:
                continue
            s = stats[key]
            s['n'] += 1
            s['w05'] += residual <= 0.5
            s['w1'] += residual <= 1.0
            s['w2'] += residual <= 2.0
            s['gt10'] += gt10
            n += 1
        if (i - args.start + 1) % 20 == 0:
            el = time.time() - t0
            print(f'  {i - args.start + 1}/{args.n} ({el:.0f}s)', flush=True)

    print(f'\n=== Tilt recovery (n={n} page-tilts, {args.n} pages x {len(TILTS)} tilts) ===')
    print('bin       n     <=0.5    <=1      <=2      >10deg')
    for key in ['dead'] + [k for k, _, _ in BINS]:
        s = stats[key]
        if s['n'] == 0:
            continue
        print(f'{key:8s} {s["n"]:5d}  {s["w05"]/s["n"]:7.3f} {s["w1"]/s["n"]:7.3f} {s["w2"]/s["n"]:7.3f} {s["gt10"]:7d}')


if __name__ == '__main__':
    main()
