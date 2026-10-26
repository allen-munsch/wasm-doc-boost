#!/usr/bin/env python3
"""Distill the fine-tuned Laya PII gate into a small encoder for WASM deployment.

Teacher: a Laya checkpoint (421M ModernBERT-large DecisionModel) that emits a
binary ``noul`` probability -- P(text contains PII/PCI). Its softened
probability distribution is the distillation target.

Student: any small HF encoder (default ``sentence-transformers/all-MiniLM-L6-v2``,
22M params) with a mean-pooled 2-class linear head. The student is the artifact
that eventually exports to ONNX / onnxruntime-web.

Two phases keep the 4 GB GPU from ever holding both models at once:

    --phase label   run the FROZEN teacher over the dataset, cache its soft logits
    --phase train   load the cache, train the student with KL + hard cross-entropy
    --phase both    label then train (smoke tests / small datasets only)

Run with the vendored source on the path:

    PYTHONPATH=$PWD/laya python3 scripts/distill_laya_pii.py \
        --teacher-dir /path/to/laya_pii_checkpoint \
        --phase both --data synthetic --max-samples 32 --epochs 1 --batch 8 \
        --output-dir /tmp/laya_student_smoke
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QTYPE_NOUL = 2

QUESTION = {
    "type": "noul",
    "instructions": (
        "Does this text contain payment card data (full card number, CVV, "
        "full magnetic-stripe track data, PIN, PIN block) or government and "
        "identity numbers (driver's license, government ID, passport number)?"
    ),
    "criteria": {},
}

# Reuse the synthetic / ai4privacy data builders from the training script.
from train_laya_pii import make_synthetic_pair, target_type_of


class StudentClassifier(nn.Module):
    """Small encoder + mean pool + 2-class head."""

    def __init__(self, encoder, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.head(pooled)


def iter_pairs(args):
    """Yield (text, is_pii) pairs.

    Synthetic: template sentences with real-format PAN/CVV/track/PIN/DL/etc.
    and their blanked counterparts. ai4privacy: rows whose ``privacy_mask``
    contains a target type (positive) plus the same text blanked (negative).
    """
    if args.data == "ai4privacy":
        from datasets import load_dataset

        ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
        n = 0
        for row in ds:
            spans = row.get("privacy_mask") or []
            text = row.get("source_text") or ""
            if not isinstance(spans, list) or not text:
                continue
            hits = []
            for sp in spans:
                label = sp.get("label", "") if isinstance(sp, dict) else ""
                if isinstance(label, str) and target_type_of(label) is not None:
                    try:
                        s, e = int(sp["start"]), int(sp["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0 <= s < e <= len(text):
                        hits.append((s, e))
            if not hits:
                continue
            chars = list(text)
            for s, e in hits:
                for i in range(s, e):
                    chars[i] = "X"
            yield text, True
            yield "".join(chars), False
            n += 2
            if args.max_samples and n >= args.max_samples:
                break
        return

    for _ in range(args.max_samples or 128):
        pos, neg = make_synthetic_pair()
        yield pos, True
        yield neg, False


def load_teacher(teacher_dir, device):
    import json

    from laya.common import build_model
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(teacher_dir, "tokenizer"))
    with open(os.path.join(teacher_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(teacher_dir, "encoder"))
    from safetensors.torch import load_file

    model.load_state_dict(
        load_file(os.path.join(teacher_dir, "model.safetensors")), strict=True
    )
    model.to(device).eval()
    return tok, cfg, model


def teacher_batch(teacher, tok_teacher, cfg, texts, device):
    """Build Laya-format sequences and run the teacher, returning [B, 2] logits."""
    from laya.common import build_sequence

    q = {"t": "noul", "ins": QUESTION["instructions"], "crit": {}}
    ids, markers = [], []
    for text in texts:
        seq, mk = build_sequence(
            tok_teacher, text, q, cfg["max_len"], cfg["head_max_len"]
        )
        ids.append(seq)
        markers.append(mk)

    n, L = len(ids), max(len(s) for s in ids)
    kmax = max(len(m) for m in markers)
    input_ids = torch.full((n, L), tok_teacher.pad_token_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, s in enumerate(ids):
        input_ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        k = len(markers[i])
        mpos[i, :k] = torch.tensor(markers[i])
        mmask[i, :k] = True
    qtype = torch.full((n,), QTYPE_NOUL, dtype=torch.long)

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")),
    ):
        logits, _ = teacher(
            input_ids.to(device),
            att.to(device),
            mpos.to(device),
            mmask.to(device),
            qtype.to(device),
        )
    return logits.float().cpu()


def phase_label(args, device):
    from transformers import AutoTokenizer

    tok_teacher, cfg, teacher = load_teacher(args.teacher_dir, device)
    tok_student = AutoTokenizer.from_pretrained(args.student_id)

    rows = []  # (text, label, teacher_logits)
    texts, labels = [], []
    t0 = time.time()
    print(f"[label] building pairs ({args.data})")
    for text, is_pii in iter_pairs(args):
        texts.append(text)
        labels.append(1 if is_pii else 0)
        if len(texts) == args.batch:
            tl = teacher_batch(teacher, tok_teacher, cfg, texts, device)
            for t, l, z in zip(texts, labels, tl):
                rows.append((t, l, z))
            texts, labels = [], []
    if texts:
        tl = teacher_batch(teacher, tok_teacher, cfg, texts, device)
        for t, l, z in zip(texts, labels, tl):
            rows.append((t, l, z))

    print(
        f"[label] {len(rows)} pairs labeled in {time.time() - t0:.1f}s; tokenizing student side"
    )
    encs = tok_student(
        [r[0] for r in rows],
        padding=True,
        truncation=True,
        max_length=args.student_max_len,
        return_tensors="pt",
    )
    cache = {
        "input_ids": encs["input_ids"],
        "attention_mask": encs["attention_mask"],
        "label": torch.tensor([r[1] for r in rows], dtype=torch.long),
        "teacher_logits": torch.stack([r[2] for r in rows]),
    }
    torch.save(cache, args.teacher_cache)
    print(f"[label] cached {len(rows)} samples to {args.teacher_cache}")


def phase_train(args, device):
    from transformers import AutoModel, AutoTokenizer

    cache = torch.load(args.teacher_cache, weights_only=False)
    n = cache["label"].size(0)
    print(f"[train] loaded {n} cached samples")

    tok_student = AutoTokenizer.from_pretrained(args.student_id)
    encoder = AutoModel.from_pretrained(args.student_id)
    student = StudentClassifier(encoder, encoder.config.hidden_size).to(device)
    student.train()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_updates = (n // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_updates), eta_min=1e-6
    )

    T = args.temperature
    alpha = args.alpha  # weight on KL; (1-alpha) on hard CE

    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(idx)
        epoch_loss, steps = 0.0, 0
        for b in range(0, n, args.batch):
            sel = idx[b : b + args.batch]
            ids = cache["input_ids"][sel].to(device)
            att = cache["attention_mask"][sel].to(device)
            label = cache["label"][sel].to(device)
            t_logits = cache["teacher_logits"][sel].to(device)

            s_logits = student(ids, att)
            p_t = F.softmax(t_logits / T, -1)
            log_p_s = F.log_softmax(s_logits / T, -1)
            kl = (
                (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_p_s)).sum(-1).mean()
                * T
                * T
            )
            ce = F.cross_entropy(s_logits, label)
            loss = alpha * kl + (1 - alpha) * ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()

            epoch_loss += loss.item()
            steps += 1
            if steps % 20 == 0:
                print(
                    f"  epoch {epoch + 1}/{args.epochs} step {steps} loss {loss.item():.4f} "
                    f"kl {kl.item():.4f} ce {ce.item():.4f}"
                )
        print(
            f"=== epoch {epoch + 1}/{args.epochs} avg loss {epoch_loss / max(1, steps):.4f} ==="
        )

    # save student in standard HF format
    os.makedirs(args.output_dir, exist_ok=True)
    student.encoder.save_pretrained(args.output_dir)
    torch.save(student.head.state_dict(), os.path.join(args.output_dir, "head.pt"))
    tok_student.save_pretrained(args.output_dir)
    print(f"[train] student saved to {args.output_dir}")

    # quick held-out eval on synthetic pairs
    student.eval()
    texts, labels = [], []
    for _ in range(64):
        pos, neg = make_synthetic_pair()
        texts.append(pos)
        labels.append(1)
        texts.append(neg)
        labels.append(0)
    correct = 0
    with torch.no_grad():
        for b in range(0, len(texts), 16):
            enc = tok_student(
                texts[b : b + 16],
                padding=True,
                truncation=True,
                max_length=args.student_max_len,
                return_tensors="pt",
            )
            logits = student(
                enc["input_ids"].to(device), enc["attention_mask"].to(device)
            )
            correct += (
                (logits.argmax(-1).cpu() == torch.tensor(labels[b : b + 16]))
                .sum()
                .item()
            )
    print(
        f"[eval] held-out accuracy {correct}/{len(texts)} = {correct / len(texts):.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher-dir", required=True, help="path to a saved Laya checkpoint"
    )
    ap.add_argument("--student-id", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--data", choices=["synthetic", "ai4privacy"], default="synthetic")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="cap on pairs (label phase); None = whole dataset",
    )
    ap.add_argument("--phase", choices=["label", "train", "both"], default="both")
    ap.add_argument("--teacher-cache", default="/tmp/laya_teacher_labels.pt")
    ap.add_argument("--output-dir", default="/tmp/laya_student")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument(
        "--alpha", type=float, default=0.7, help="weight on KL; (1-alpha) on hard CE"
    )
    ap.add_argument("--student-max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[setup] device={device}")

    if args.phase in ("label", "both"):
        phase_label(args, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.phase in ("train", "both"):
        phase_train(args, device)


if __name__ == "__main__":
    main()
