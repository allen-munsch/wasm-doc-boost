#!/usr/bin/env python3
"""Train a CRF for PII detection on document OCR text.

Data sources:
  1. Clean FATURA2 tokenized text (from fatura2_pii_gt.json):
     - GT NER tags directly from dataset metadata
     - 10,000 invoices, 399K PII-tagged tokens
  2. Tesseract OCR output (from fatura2_ocr.json, optional):
     - OCR text + word bboxes, aligned to GT tags via fuzzy matching
     - Training on OCR noise makes CRF robust to recognition errors
  3. Synthetic variants (generated in this script):
     - Character-level noise (OCR-like substitutions)
     - Token dropout (missing words)
     - Line break variations

FATURA2 NER tag → CRF label mapping:
  Tag 5 (SELLER_ADDRESS) → ADDRESS (with sub-token detection for PHONE/EMAIL/ZIP)
  Tag 8 (SELLER_NAME)    → NAME
  Tag 10 (BUYER)         → NAME
  Tag 13 (PAYMENT_DETAILS) → ACCOUNT
  Tag 0 (O)              → O (background)
  Other                   → O (structural tags like DATE, TOTAL, etc.)

Output: data/crf_model.json
  {
    "labels": ["O","ADDRESS","NAME","PHONE","EMAIL","ZIP","ACCOUNT"],
    "labelWeights": [...K arrays of feature_weight vectors...],
    "transitions": [K[K] matrix],
    "featureTemplate": {...template→index mapping...},
    "config": {...training params...}
  }

Usage: python scripts/train_crf.py [--ocr data/fatura2_ocr.json]
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# ── Label mapping ────────────────────────────────────

# FATURA2 NER tag IDs → CRF label (simplified PII categories)
# The paper confirms: 0=O, 1=TABLE, 2=LOGO, 3=DATE, 4=NUMBER,
# 5=SELLER_ADDRESS, 6=TOTAL, 7=TITLE, 8=SELLER_NAME, 9=SUB_TOTAL,
# 10=BUYER, 11=DUE_DATE, 12=NOTE, 13=PAYMENT_DETAILS
TAG_TO_LABEL = {
    0: "O",
    1: "O",  # TABLE → background
    2: "O",  # LOGO → background
    3: "O",  # DATE → background
    4: "O",  # NUMBER → background
    5: "ADDRESS",  # SELLER_ADDRESS
    6: "O",  # TOTAL → background
    7: "O",  # TITLE → background
    8: "NAME",  # SELLER_NAME
    9: "O",  # SUB_TOTAL → background
    10: "NAME",  # BUYER
    11: "O",  # DUE_DATE → background
    12: "O",  # NOTE → background
    13: "ACCOUNT",  # PAYMENT_DETAILS
}

# Sub-token detection within ADDRESS spans
# Tokens within ADDRESS that match specific patterns get fine-grained labels
EMAIL_RE = re.compile(r"^[\w.-]+@[\w.-]+\.\w+$")
PHONE_RE = re.compile(r"^\+\(\d{3}\)\d{3}-\d{4}$")
ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")

CRF_LABELS = ["O", "ADDRESS", "NAME", "PHONE", "EMAIL", "ZIP", "ACCOUNT"]


def refine_labels(tokens: list[str], ner_tags: list[int]) -> list[str]:
    """Map FATURA2 NER tags to CRF labels.

    FATURA2 tag semantics:
    - 5 (SELLER_ADDRESS): the full address block including name, street, city, phone, email, zip
    - 8 (SELLER_NAME), 10 (BUYER): the label KEYWORD only (\"Buyer\", \"Seller:\") — NOT the name value
    - 13 (PAYMENT_DETAILS): bank account numbers, payment terms

    Strategy: within tag-5 spans, the first alphabetic token(s) before
    a street number are NAME; the rest is ADDRESS.  Tag 8/10 are O
    (structural keywords).
    """
    labels = ["O"] * len(tokens)

    # First pass: identify tag-5 spans
    spans = []  # (start, end)
    in_span = False
    start = 0
    for i, tag in enumerate(ner_tags):
        if tag == 5 and not in_span:
            start = i
            in_span = True
        elif tag != 5 and in_span:
            spans.append((start, i - 1))
            in_span = False
    if in_span:
        spans.append((start, len(tokens) - 1))

    # Process each tag-5 span
    STREET_SUFFIXES = {"street", "st", "st.", "avenue", "ave", "ave.", "road", "rd",
                       "rd.", "drive", "dr", "dr.", "lane", "ln", "ln.", "court", "ct",
                       "ct.", "boulevard", "blvd", "blvd.", "way", "place", "pl", "pl.",
                       "causeway", "cswy", "highway", "hwy", "parkway", "pkwy",
                       "circle", "cir", "trail", "trl", "suite", "ste", "apt", "unit",
                       "po", "p.o.", "box"}

    for span_start, span_end in spans:
        # Find the split point: first street-number-like token or street suffix
        split_at = span_end + 1
        for i in range(span_start, span_end + 1):
            t = tokens[i].lower().rstrip(",.")
            # Street number: starts with digit, 1-5 chars
            if t[0].isdigit() and 1 <= len(t) <= 5:
                split_at = i
                break
            # Street suffix
            if t in STREET_SUFFIXES:
                # The name is everything before the street number + street name
                # Name typically ends at the street suffix
                split_at = i + 1
                break

        # Everything before split_at is NAME, after is ADDRESS
        for i in range(span_start, min(split_at, span_end + 1)):
            t = tokens[i]
            if t.isalpha() and len(t) > 1:
                labels[i] = "NAME"
            else:
                labels[i] = "ADDRESS"

        for i in range(split_at, span_end + 1):
            t = tokens[i]
            if EMAIL_RE.match(t):
                labels[i] = "EMAIL"
            elif PHONE_RE.match(t):
                labels[i] = "PHONE"
            elif ZIP_RE.match(t):
                labels[i] = "ZIP"
            else:
                labels[i] = "ADDRESS"

    # Tag-13: payment details → ACCOUNT for numeric tokens
    for i, tag in enumerate(ner_tags):
        if tag == 13:
            t = tokens[i]
            if t.isdigit() and 4 <= len(t) <= 20:
                labels[i] = "ACCOUNT"
            elif EMAIL_RE.match(t):
                labels[i] = "EMAIL"

    return labels


# ── Feature extraction ───────────────────────────────

def token_shape(token: str) -> str:
    """Character shape: uppercase→A, lowercase→a, digit→0, other→-"""
    result = []
    for c in token:
        if c.isupper():
            result.append("A")
        elif c.islower():
            result.append("a")
        elif c.isdigit():
            result.append("0")
        else:
            result.append("-")
    return "".join(result)


def extract_features(tokens: list[str], i: int, bboxes: list[list] | None = None) -> dict[str, Any]:
    """Extract CRF features for token at position i."""
    t = tokens[i]
    feats = {}

    # Current token features
    feats["w"] = t.lower() if len(t) <= 20 else "__LONG__"
    feats["w_shape"] = token_shape(t)[:10]
    feats["w_len"] = str(min(len(t), 20))
    feats["w_upper"] = "1" if t.isupper() else "0"
    feats["w_title"] = "1" if t.istitle() else "0"
    feats["w_digit"] = "1" if t.isdigit() else "0"
    feats["w_alpha"] = "1" if t.isalpha() else "0"
    feats["w_has_digit"] = "1" if any(c.isdigit() for c in t) else "0"
    feats["w_has_at"] = "1" if "@" in t else "0"
    feats["w_has_hyphen"] = "1" if "-" in t else "0"
    feats["w_prefix2"] = t[:2].lower() if len(t) >= 2 else t.lower()
    feats["w_suffix2"] = t[-2:].lower() if len(t) >= 2 else t.lower()
    feats["w_prefix3"] = t[:3].lower() if len(t) >= 3 else t.lower()
    feats["w_suffix3"] = t[-3:].lower() if len(t) >= 3 else t.lower()

    # Position features
    feats["pos_bias"] = "1" if i == 0 else ("1" if i == len(tokens) - 1 else "0")

    # Neighboring tokens (within window of 2)
    for offset in [-2, -1, 1, 2]:
        j = i + offset
        if 0 <= j < len(tokens):
            nt = tokens[j].lower()
            feats[f"w{offset:+d}"] = nt if len(nt) <= 20 else "__LONG__"
            feats[f"w{offset:+d}_shape"] = token_shape(tokens[j])[:8]
        else:
            feats[f"w{offset:+d}"] = "__BOS__" if j < 0 else "__EOS__"

    # Bbox layout features (if available)
    if bboxes and i < len(bboxes):
        b = bboxes[i]
        w = b[2] - b[0]
        h = b[3] - b[1]
        feats["bbox_w"] = f"{w // 50 * 50}"
        feats["bbox_h"] = f"{h // 10 * 10}"
        feats["bbox_x"] = f"{b[0] // 100 * 100}"
        feats["bbox_y"] = f"{b[1] // 50 * 50}"

    return feats


def tokens_to_features(tokens: list[str], bboxes: list[list] | None = None) -> list[dict]:
    return [extract_features(tokens, i, bboxes) for i in range(len(tokens))]


# ── Data loading ─────────────────────────────────────

def load_clean_gt(gt_path: str, max_samples: int = 0) -> list[tuple[list[str], list[str]]]:
    """Load clean FATURA2 tokenized GT: returns [(tokens, crf_labels), ...]."""
    with open(gt_path) as f:
        data = json.load(f)

    sequences = []
    samples = data["samples"]
    if max_samples > 0:
        samples = samples[:max_samples]

    for s in samples:
        tokens = s["tokens"]
        labels = refine_labels(tokens, s["tags"])
        # Skip sequences with only O labels
        if all(l == "O" for l in labels):
            continue
        sequences.append((tokens, labels))

    return sequences


def align_ocr_to_gt(ocr_words: list[dict], gt_tokens: list[str], gt_labels: list[str]) -> tuple[list[str], list[str], list[list]]:
    """
    Fuzzy-align OCR word output to ground truth tokens.
    Returns (ocr_tokens, aligned_labels, bboxes).
    Uses simple Levenshtein-like greedy alignment.
    """
    ocr_texts = [w["text"] for w in ocr_words]
    ocr_bboxes = [w["bbox"] for w in ocr_words]
    ocr_confidences = [w["confidence"] for w in ocr_words]

    # Build clean text from GT tokens
    gt_text = " ".join(gt_tokens)

    # Simple approach: align by word position and fuzzy match
    # For each GT token, find the closest OCR word by Levenshtein distance
    aligned_labels = []
    aligned_bboxes = []
    ocr_idx = 0

    for gt_i, gt_token in enumerate(gt_tokens):
        # Search window: look ahead up to 5 words
        best_dist = 999
        best_idx = ocr_idx
        for j in range(ocr_idx, min(ocr_idx + 5, len(ocr_texts))):
            dist = _levenshtein(gt_token.lower(), ocr_texts[j].lower())
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_dist <= max(3, len(gt_token) // 2):  # Allow some OCR errors
            aligned_labels.append(gt_labels[gt_i])
            aligned_bboxes.append(ocr_bboxes[best_idx])
            ocr_idx = best_idx + 1
        # else: OCR missed this token, skip it

    return ocr_texts, aligned_labels, aligned_bboxes


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance (fast path for short strings)."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[-1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def load_ocr_data(ocr_path: str, gt_path: str, max_samples: int = 0) -> list[tuple[list[str], list[str], list[list]]]:
    """Load OCR data and align to GT. Returns [(tokens, labels, bboxes), ...]."""
    with open(ocr_path) as f:
        ocr_data = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Map OCR filenames to GT sample indices
    # OCR filename: fatura2_train_N.jpg → GT sample index N
    sequences = []

    for ocr_entry in ocr_data[:max_samples] if max_samples > 0 else ocr_data:
        # Extract parquet row index from filename
        fn = ocr_entry.get("filename", "")
        match = re.search(r"fatura2_(train|test)_(\d+)", fn)
        if not match:
            continue
        split, idx = match.groups()
        idx = int(idx)
        if split == "test":
            idx += 8600  # test samples come after train

        if idx >= len(gt_data["samples"]):
            continue

        gt_sample = gt_data["samples"][idx]
        gt_tokens = gt_sample["tokens"]
        gt_labels = refine_labels(gt_tokens, gt_sample["tags"])

        ocr_words = ocr_entry.get("words", [])
        if not ocr_words:
            continue

        ocr_tokens, aligned_labels, aligned_bboxes = align_ocr_to_gt(ocr_words, gt_tokens, gt_labels)

        if not ocr_tokens or not any(l != "O" for l in aligned_labels):
            continue

        sequences.append((ocr_tokens, aligned_labels, aligned_bboxes))

    return sequences


# ── Synthetic augmentation ───────────────────────────

OCR_MISTAKES = {
    "I": "l", "l": "I", "O": "0", "0": "O",
    "S": "5", "5": "S", "B": "8", "8": "B",
    "1": "l", "rn": "m", "m": "rn",
}


def generate_synthetic(sequences: list[tuple[list[str], list[str]]], n_variants: int = 2) -> list[tuple[list[str], list[str]]]:
    """Generate synthetic variants with OCR-like noise."""
    synthetic = []
    rng = random.Random(SEED)

    for tokens, labels in sequences:
        for _ in range(n_variants):
            variant_tokens = []
            variant_labels = []

            for token, label in zip(tokens, labels):
                # 10% chance of token dropout (missing from OCR output)
                if rng.random() < 0.10:
                    continue

                # 15% chance of character-level OCR error
                if rng.random() < 0.15 and len(token) >= 3:
                    noisy = list(token)
                    pos = rng.randint(0, len(token) - 1)
                    orig_char = noisy[pos]
                    # Apply OCR mistake mapping
                    if orig_char in OCR_MISTAKES:
                        noisy[pos] = OCR_MISTAKES[orig_char]
                    elif orig_char.isalpha():
                        noisy[pos] = rng.choice("aeiouAEIOU809")
                    variant_tokens.append("".join(noisy))
                    variant_labels.append(label)
                else:
                    variant_tokens.append(token)
                    variant_labels.append(label)

            # 5% chance of line break: insert random newline splitting
            if rng.random() < 0.05 and len(variant_tokens) > 10:
                split_at = rng.randint(3, len(variant_tokens) - 3)
                variant_tokens.insert(split_at, "\n")
                variant_labels.insert(split_at, "O")

            if len(variant_tokens) >= 2 and any(l != "O" for l in variant_labels):
                synthetic.append((variant_tokens, variant_labels))

    return synthetic


# ── Main training ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="data/fatura2_pii_gt.json")
    parser.add_argument("--ocr", default="")
    parser.add_argument("--samples", type=int, default=0, help="Max clean GT samples")
    parser.add_argument("--ocr-samples", type=int, default=0, help="Max OCR samples")
    parser.add_argument("--synthetic", type=int, default=2, help="Variants per clean sequence")
    parser.add_argument("--c1", type=float, default=1.0, help="L1 regularization")
    parser.add_argument("--c2", type=float, default=1.0, help="L2 regularization")
    parser.add_argument("--max-iter", type=int, default=200, help="L-BFGS iterations")
    parser.add_argument("--output", default="data/crf_model.json")
    args = parser.parse_args()

    gt_path = ROOT / args.gt

    # ── Load clean GT ─────────────────────────────────
    print(f"Loading clean GT from {gt_path}...")
    clean_sequences = load_clean_gt(str(gt_path), args.samples)
    label_count = Counter()
    for _, labels in clean_sequences:
        label_count.update(labels)
    print(f"  {len(clean_sequences)} clean sequences")
    for lbl in CRF_LABELS:
        print(f"    {lbl}: {label_count[lbl]:>8,}")

    # ── Load OCR data ─────────────────────────────────
    ocr_sequences_bbox = []
    if args.ocr:
        ocr_path = ROOT / args.ocr
        if ocr_path.exists():
            print(f"\nLoading OCR data from {ocr_path}...")
            ocr_sequences_bbox = load_ocr_data(str(ocr_path), str(gt_path), args.ocr_samples)
            print(f"  {len(ocr_sequences_bbox)} OCR-aligned sequences")

    # ── Generate synthetics ───────────────────────────
    print(f"\nGenerating {args.synthetic} synthetic variants per clean sequence...")
    synthetic_sequences = generate_synthetic(clean_sequences, args.synthetic)
    print(f"  {len(synthetic_sequences)} synthetic sequences")

    # ── Combine all data ──────────────────────────────
    all_sequences = []
    # Clean GT
    for tokens, labels in clean_sequences:
        all_sequences.append((tokens, labels, None))  # no bbox
    # OCR-aligned
    for tokens, labels, bboxes in ocr_sequences_bbox:
        all_sequences.append((tokens, labels, bboxes))
    # Synthetics
    for tokens, labels in synthetic_sequences:
        all_sequences.append((tokens, labels, None))

    print(f"\nTotal training sequences: {len(all_sequences)}")
    total_tokens = sum(len(t[0]) for t in all_sequences)
    print(f"Total tokens: {total_tokens:,}")

    # ── Convert to CRF format ─────────────────────────
    print("\nExtracting features...")
    X = []
    y = []
    for tokens, labels, bboxes in all_sequences:
        feats = tokens_to_features(tokens, bboxes)
        X.append(feats)
        y.append(labels)

    # ── Train/val split ───────────────────────────────
    rng = random.Random(SEED)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    split = int(len(X) * 0.85)
    train_idx = set(indices[:split])
    val_idx = set(indices[split:])

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    print(f"Train: {len(X_train)} sequences, Val: {len(X_val)} sequences")

    # ── Train CRF ─────────────────────────────────────
    print(f"\nTraining CRF (c1={args.c1}, c2={args.c2}, max_iter={args.max_iter})...")
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter,
        all_possible_transitions=True,
        verbose=False,
    )

    crf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────
    y_pred = crf.predict(X_val)
    labels_present = sorted(set(l for seq in y_val for l in seq))

    print("\n=== CRF Validation Metrics ===")
    print(f"Sequences: {len(X_val)}, Labels: {labels_present}\n")

    for label in labels_present:
        tp = sum(
            1 for seq_true, seq_pred in zip(y_val, y_pred)
            for t, p in zip(seq_true, seq_pred)
            if t == label and p == label
        )
        fp = sum(
            1 for seq_true, seq_pred in zip(y_val, y_pred)
            for t, p in zip(seq_true, seq_pred)
            if t != label and p == label
        )
        fn = sum(
            1 for seq_true, seq_pred in zip(y_val, y_pred)
            for t, p in zip(seq_true, seq_pred)
            if t == label and p != label
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {label:<12} P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  (TP={tp} FP={fp} FN={fn})")

    # ── Serialize: dump to temp file, parse manually ──
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        dump_file = tmp.name

    crf.tagger_.dump(dump_file)
    with open(dump_file) as f:
        raw = f.read()
    os.unlink(dump_file)

    # Parse ATTRIBUTES section
    attr_start = raw.find("ATTRIBUTES = {") + len("ATTRIBUTES = {")
    attr_end = raw.find("}\n\nSTATE_FEATURES")
    feat_index = {}
    for line in raw[attr_start:attr_end].strip().split("\n"):
        m = re.match(r"(\d+):\s+(.+)", line.strip())
        if m:
            feat_index[m.group(2)] = int(m.group(1))

    print(f"  Parsed {len(feat_index)} features")

    # Parse STATE_FEATURES
    sf_start = raw.find("STATE_FEATURES = {") + len("STATE_FEATURES = {")
    sf_end = raw.find("}\n\nTRANSITION_FEATURES")
    sf_text = raw[sf_start:sf_end]

    num_feats = len(feat_index)
    label_weights = {lbl: [0.0] * num_feats for lbl in CRF_LABELS}
    for line in sf_text.strip().split("\n"):
        m = re.match(r"\(\d+\)\s+(.+?)\s+-->\s+(\S+):\s+([-\d.e+]+)", line.strip())
        if m:
            feat_name = m.group(1)
            label = m.group(2)
            weight = float(m.group(3))
            if feat_name in feat_index and label in label_weights:
                label_weights[label][feat_index[feat_name]] = weight

    # Parse TRANSITION_FEATURES
    tf_start = raw.find("TRANSITION_FEATURES = {") + len("TRANSITION_FEATURES = {")
    tf_text = raw[tf_start:]
    tf_end = tf_text.find("}\n")
    tf_text = tf_text[:tf_end] if tf_end >= 0 else tf_text

    transitions = [[0.0] * len(CRF_LABELS) for _ in range(len(CRF_LABELS))]
    for line in tf_text.strip().split("\n"):
        m = re.match(r"\(\d+\)\s+(\S+)\s+-->\s+(\S+):\s+([-\d.e+]+)", line.strip())
        if m:
            l_from, l_to = m.group(1), m.group(2)
            weight = float(m.group(3))
            if l_from in CRF_LABELS and l_to in CRF_LABELS:
                i = CRF_LABELS.index(l_from)
                j = CRF_LABELS.index(l_to)
                transitions[i][j] = weight

    model = {
        "labels": CRF_LABELS,
        "featureIndex": feat_index,
        "labelWeights": label_weights,
        "transitions": transitions,
        "config": {
            "c1": args.c1,
            "c2": args.c2,
            "maxIter": args.max_iter,
            "numSequences": len(X_train),
            "numFeatures": num_feats,
        },
    }

    out_path = ROOT / args.output
    with open(out_path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"\nModel size: {size_kb:.1f} KB")

    import gzip
    with open(out_path, "rb") as f:
        compressed = gzip.compress(f.read())
    print(f"Gzipped size: {len(compressed) / 1024:.1f} KB")
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
