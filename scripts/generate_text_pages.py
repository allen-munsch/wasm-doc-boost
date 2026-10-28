#!/usr/bin/env python3
"""Generate real rendered-text-only pages as clean digital negatives.

Dove's v0.5.0 report: rasterised text-only PDFs (fpdf2 Helvetica) now score
is_paper 0.76-0.99 because the CORU photographed-receipt additions pulled the
is_digital boundary. These pages re-anchor "clean black-on-white rendered
text -> digital" so such pages route to Tesseract, not the VLM.

Renders PNGs at page scale with DejaVuSans, then the export pipeline resizes
to a 512 long edge. Labels: is_document=1, is_digital=1, paper/crumpled/shadow=0.

Usage: python scripts/generate_text_pages.py --n 1500 --out data/images/textonly
Appends rows to data/labels.csv (or writes --labels).
"""
import argparse, csv, os, random
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    '/nix/store/cis3qxbpsfh0ppj31iyrzpyixfjgw9my-dejavu-fonts-minimal-2.37/share/fonts/truetype/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

WORDS = ('the quick brown fox jumps over the lazy dog invoice receipt total amount '
         'payment customer order item quantity price date reference account statement '
         'summary balance due description vendor address').split()


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit('no DejaVuSans.ttf found; set FONT in the script')


def make_page(rng, page_w=1240, page_h=1754):
    img = Image.new('RGB', (page_w, page_h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font_path = find_font()
    n_lines = rng.choice([1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 35, 45])
    body_size = rng.randint(16, 30)
    line_gap = rng.uniform(1.35, 1.8)
    left = rng.randint(90, 160)
    top = rng.randint(80, 140)
    ink = rng.randint(0, 40)

    has_title = rng.random() < 0.5
    if has_title:
        title_size = body_size + rng.randint(6, 14)
        title_font = ImageFont.truetype(font_path, title_size)
        title = ' '.join(rng.choice(WORDS) for _ in range(rng.randint(3, 8))).title()
        d.text((left, top), title, fill=ink, font=title_font)
        top += int(title_size * line_gap) + rng.randint(8, 24)

    body_font = ImageFont.truetype(font_path, body_size)
    for _ in range(n_lines):
        n_words = rng.randint(4, 14)
        line = ' '.join(rng.choice(WORDS) for _ in range(n_words))
        if rng.random() < 0.15:
            line += '  ' + str(rng.randint(0, 999))
        d.text((left, top), line, fill=ink, font=body_font)
        top += int(body_size * line_gap)
        if top > page_h - 80:
            break
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--out', default='data/images/textonly')
    ap.add_argument('--labels', default='data/labels.csv')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        img = make_page(rng)
        fn = f'text_{i:05d}.png'
        img.save(os.path.join(args.out, fn))
        rows.append((f'textonly/{fn}', 1, 1, 0, 0, 0))

    header = ['filename', 'is_document', 'is_digital', 'is_paper', 'is_crumpled', 'is_shadow']
    with open(args.labels, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f'wrote {args.n} pages to {args.out}, appended to {args.labels}')


if __name__ == '__main__':
    main()
