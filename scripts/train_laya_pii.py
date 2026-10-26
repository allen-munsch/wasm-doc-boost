#!/usr/bin/env python3
"""Fine-tune a Laya checkpoint into a binary PII/PCI triage gate.

Trains a single ``noul`` decision -- "does this text contain card / track / PIN /
government-identity data?" -- over text extracted from documents. The gate is the
System-1 "should we run the expensive span extractor + redactor on this page"
decision, not a redactor itself (Laya emits a probability, not character spans).

Two data sources:

* ``--data synthetic`` (default) -- deterministic templates with real-format PAN /
  CVV / track / PIN / DL / passport / gov-ID values embedded in sentences, paired
  with the same sentence redacted. Used for smoke tests and to validate the loop.

* ``--data ai4privacy`` -- binarizes ``ai4privacy/pii-masking-300k``. Positive =
  ``source_text`` rows whose ``privacy_mask`` contains a target type; negative =
  the same text with ALL sensitive spans blanked (so it is genuinely clean, not
  still carrying name/address/email/phone PII).

* ``--data hybrid`` -- the recommended mix: real ai4privacy identity positives +
  synthetic Luhn-valid PCI (PAN/CVV/track/PIN/PIN-block, including bare values),
  with negatives from clean prose, masked/redacted twins, near-miss numbers, and
  fully-blanked ai4privacy text.

Training is supervised cross-entropy over the binary ``noul`` decision,
single-GPU, with LoRA adapters on the frozen ModernBERT encoder and the decision
head trained fully. After training the adapters are merged and a standard Laya
checkpoint is written so ``laya.load(output_dir)`` works unchanged.

Run with the vendored source on the path:

    PYTHONPATH=$PWD/laya python3 scripts/train_laya_pii.py --data synthetic \
        --max-samples 256 --epochs 1 --micro-batch 2
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

QTYPE_NOUL = 2  # laya.common.QTYPES["noul"]

QUESTION = {
    "type": "noul",
    "instructions": (
        "Does this text contain payment card data (full card number, CVV, "
        "full magnetic-stripe track data, PIN, PIN block) or government and "
        "identity numbers (driver's license, government ID, passport number)?"
    ),
    "criteria": {},
}

TARGET_ALIASES = {
    "PAN": {
        "CREDIT_CARD",
        "CREDITCARD",
        "CARD_NUMBER",
        "CARDNUMBER",
        "PAN",
        "BANK_CARD",
        "BANKCARD",
        "CARD",
        "DEBIT_CARD",
        "DEBITCARD",
    },
    "CVV": {
        "CVV",
        "CVC",
        "CVV2",
        "CVC2",
        "SECURITY_CODE",
        "SECURITYCODE",
        "CARD_SECURITY_CODE",
    },
    "DL": {
        "DRIVER_LICENSE",
        "DRIVERLICENSE",
        "DRIVERS_LICENSE",
        "DRIVER_LICENCE",
        "DL_NUMBER",
        "DLNUMBER",
        "DRIVING_LICENSE",
        "LICENSE_NUMBER",
        "DRIVING_LICENCE",
    },
    "GOV_ID": {
        "SSN",
        "SOCIAL_SECURITY",
        "SOCIALSECURITY",
        "GOVERNMENT_ID",
        "GOVERNMENTID",
        "NATIONAL_ID",
        "NATIONALID",
        "TAX_ID",
        "TAXID",
        "NID",
        "GOV_ID",
        "GOVT_ID",
        "IDCARD",
        "ID_CARD",
        "IDENTITY_CARD",
    },
    "PASSPORT": {"PASSPORT", "PASSPORT_NUMBER", "PASSPORTNUMBER"},
    "TRACK": {
        "TRACK",
        "TRACK_DATA",
        "TRACKDATA",
        "MAGSTRIPE",
        "MAG_STRIPE",
        "MAGNETIC_STRIPE",
        "TRACK1",
        "TRACK2",
        "TRACK_1",
        "TRACK_2",
    },
    "PIN": {
        "PIN",
        "PIN_CODE",
        "PINCODE",
        "PIN_NUMBER",
        "PINNUMBER",
        "ATM_PIN",
        "ATMPIN",
    },
    "PIN_BLOCK": {"PIN_BLOCK", "PINBLOCK", "PIN_BLOCK_DATA"},
}


def _norm_label(label: str) -> str:
    return "".join(ch for ch in label.upper() if ch.isalnum())


def target_type_of(label: str) -> str | None:
    """Map a dataset entity label to one of the 8 target types, or None."""
    key = _norm_label(label)
    for target, aliases in TARGET_ALIASES.items():
        if key in aliases:
            return target
    return None


# ── synthetic data ──────────────────────────────────────────────────────────


def luhn_checksum(number: str) -> int:
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10


def make_pan(prefix: str = "4", length: int = 16) -> str:
    body = prefix + "".join(
        random.choice("0123456789") for _ in range(length - 1 - len(prefix))
    )
    total = 0
    rev_body = body[::-1]
    for i, ch in enumerate(rev_body):
        d = int(ch)
        if (i + 1) % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return body + str((10 - total % 10) % 10)


def _rand_digits(n: int) -> str:
    return "".join(random.choice("0123456789") for _ in range(n))


TEMPLATES = [
    (
        "Customer card {pan} exp {exp} cvv {cvv} was charged $99.00.",
        "Customer card XXXX XXXX XXXX XXXX exp XX/XX cvv XXX was charged $99.00.",
    ),
    (
        "Order paid with Visa {pan}, security code {cvv}, expires {exp}.",
        "Order paid with Visa XXXX XXXX XXXX XXXX, security code XXX, expires XX/XX.",
    ),
    (
        "Track 2 data: ;{pan}={exp}101{cvv}0000000?",
        "Track 2 data: ;XXXX=XX101XXX0000000?",
    ),
    ("The card PIN is {pin}.", "The card PIN is XXXX."),
    (
        "Driver's license number {dl} on file.",
        "Driver's license number XX-XXX-XXXX on file.",
    ),
    ("Passport number {passport} for travel.", "Passport number XXXXXXXX for travel."),
    (
        "Government ID {gov} for the applicant.",
        "Government ID XXX-XX-XXXX for the applicant.",
    ),
    (
        "Card {pan} with PIN {pin} and CVV {cvv}.",
        "Card XXXX XXXX XXXX XXXX with PIN XXXX and CVV XXX.",
    ),
    (
        "Track 1 data: %B{pan}^DOE/JOHN^{exp}10100000000000000000?",
        "Track 1 data: %BXXXX^DOE/JOHN^XXXX10100000000000000000?",
    ),
    (
        "PIN block: {pin_block} (ISO 9564 format 0).",
        "PIN block: XXXXXXXXXXXXXXXX (ISO 9564 format 0).",
    ),
]


def make_synthetic_pair() -> tuple[str, str]:
    tmpl_pos, tmpl_neg = random.choice(TEMPLATES)
    values = {
        "pan": make_pan(),
        "exp": f"{random.randint(1, 12):02d}/{random.randint(24, 30)}",
        "cvv": _rand_digits(3),
        "pin": _rand_digits(4),
        "dl": f"{_rand_digits(3)}-{_rand_digits(3)}-{_rand_digits(4)}",
        "passport": _rand_digits(8),
        "gov": f"{_rand_digits(3)}-{_rand_digits(2)}-{_rand_digits(4)}",
        "pin_block": "".join(random.choice("0123456789ABCDEF") for _ in range(16)),
    }
    return tmpl_pos.format(**values), tmpl_neg.format(**values)


# ── training item construction ──────────────────────────────────────────────


def build_training_item(state: str, is_pii: bool, tok, cfg: dict) -> dict | None:
    from laya.common import build_sequence, render_options

    target = [0.0, 1.0] if is_pii else [1.0, 0.0]
    q = {"t": "noul", "ins": QUESTION["instructions"], "crit": {}}
    seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != len(render_options(q)):
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPE_NOUL,
        "target": target,
        "label": 1 if is_pii else 0,
    }


def collate_train_batch(items: list[dict], pad_id: int) -> dict:
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel) -> float:
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


# ── data loaders ────────────────────────────────────────────────────────────


def load_synthetic_items(n: int, tok, cfg: dict) -> list[dict]:
    items = []
    for _ in range(n):
        pos, neg = make_synthetic_pair()
        it_pos = build_training_item(pos, True, tok, cfg)
        it_neg = build_training_item(neg, False, tok, cfg)
        if it_pos:
            items.append(it_pos)
        if it_neg:
            items.append(it_neg)
    return items


def load_ai4privacy_items(limit: int, tok, cfg: dict) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
    items = []
    seen_types = {}
    for row in ds:
        spans = row.get("privacy_mask") or []
        text = row.get("source_text") or ""
        if not isinstance(spans, list) or not text:
            continue
        target_hits = []
        all_hits = []
        for sp in spans:
            label = sp.get("label", "") if isinstance(sp, dict) else ""
            if not isinstance(label, str):
                continue
            try:
                s, e = int(sp["start"]), int(sp["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= s < e <= len(text)):
                continue
            all_hits.append((s, e))
            tt = target_type_of(label)
            if tt is not None:
                target_hits.append((s, e))
                seen_types[tt] = seen_types.get(tt, 0) + 1
        if not target_hits:
            continue
        # negative: blank ALL sensitive spans (not just target types) so it is
        # genuinely clean rather than still carrying name/address/email/phone PII
        chars = list(text)
        for s, e in all_hits:
            for i in range(s, e):
                chars[i] = " "
        negative = " ".join("".join(chars).split())
        it_pos = build_training_item(text, True, tok, cfg)
        it_neg = build_training_item(negative, False, tok, cfg)
        if it_pos:
            items.append(it_pos)
        if it_neg:
            items.append(it_neg)
        if limit and len(items) >= limit:
            break
    print(
        f"[data] ai4privacy: {len(items)} items; target-type span counts: {seen_types}"
    )
    return items


def make_non_luhn_pan(prefix: str = "4", length: int = 16) -> str:
    """A card-shaped number that FAILS Luhn (an ordinary reference/id, not a card)."""
    pan = make_pan(prefix, length)
    return pan[:-1] + str((int(pan[-1]) + 1) % 10)


CLEAN_TEMPLATES = [
    "The quarterly earnings call is scheduled for Thursday at 10:00 AM.",
    "Invoice #48392 for office supplies totals $1,240.75 and is due in 30 days.",
    "Meeting notes: review the Q3 roadmap, assign owners, and set deadlines.",
    "Please find attached the signed contract and the updated statement of work.",
    "The shipping department reported a 12% increase in on-time deliveries.",
    "Project status: the migration is on schedule and under budget.",
    "Team lunch is at noon in the conference room; bring your laptop.",
    "The server maintenance window is set for Saturday between 2:00 and 4:00 AM.",
    "All employees must complete the annual security training by the end of the month.",
    "The building access code changes on the first of every quarter.",
]


def make_hard_negative(rng: random.Random) -> str:
    """A clean or near-miss negative with no LIVE PII.

    Mixes masked/redacted twins (``XXXX``), clean prose, and numbers that fail
    validation (non-Luhn card shapes, short digit codes, hex digests) so the
    gate cannot just pattern-match digits or the literal 'XXXX' blanking.
    """
    kind = rng.random()
    if kind < 0.35:
        # masked twin of a synthetic PII sentence: structurally card-like but
        # no live values, so it must be classified negative
        return make_synthetic_pair()[1]
    if kind < 0.60:
        return rng.choice(CLEAN_TEMPLATES)
    if kind < 0.80:
        pan = make_non_luhn_pan()
        return rng.choice(
            [
                f"Order reference {pan} will ship within two business days.",
                f"Your ticket number {pan} has been confirmed.",
                f"Lot number {pan} is currently out of stock.",
                f"Reference code {pan} for your records.",
            ]
        )
    if kind < 0.88:
        d = _rand_digits(rng.choice([4, 5, 6]))
        return rng.choice(
            [
                f"The invoice total comes to ${d}.00 including tax.",
                f"Please dial extension {d} for billing support.",
                f"Room {d} is reserved under your company name.",
                f"Your confirmation code is {d}.",
            ]
        )
    hex_digest = "".join(rng.choice("0123456789ABCDEF") for _ in range(16))
    return rng.choice(
        [
            f"Checksum {hex_digest} verified for the archive.",
            "Legacy data: ;4111111111111112=25121010000000000000?",
        ]
    )


def build_hybrid_texts(
    limit: int, seed: int = 42, synth_frac: float = 0.5
) -> tuple[list[str], list[str]]:
    """Return (positives, negatives) raw strings, 50/50 balanced.

    ``synth_frac`` of positives are synthetic Luhn-valid PCI (PAN/CVV/track/PIN/
    PIN block) so the 5 card types the dataset lacks are always covered; the rest
    are real ai4privacy text containing DL/GovID/Passport spans. The same fraction
    of negatives are hard near-miss / clean prose; the rest are real ai4privacy
    text with ALL sensitive spans blanked (clean document structure).
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    n_synth_pos = int(limit * synth_frac)
    n_real_pos = limit - n_synth_pos
    n_hard_neg = int(limit * synth_frac)
    n_real_neg = limit - n_hard_neg

    pos_texts: list[str] = []
    neg_texts: list[str] = []

    ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
    for row in ds:
        if len(pos_texts) >= n_real_pos and len(neg_texts) >= n_real_neg:
            break
        spans = row.get("privacy_mask") or []
        text = row.get("source_text") or ""
        if not isinstance(spans, list) or not text:
            continue
        target_hits: list[tuple[int, int]] = []
        other_hits: list[tuple[int, int]] = []
        for sp in spans:
            label = sp.get("label", "") if isinstance(sp, dict) else ""
            if not isinstance(label, str):
                continue
            try:
                s, e = int(sp["start"]), int(sp["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= s < e <= len(text)):
                continue
            (target_hits if target_type_of(label) is not None else other_hits).append(
                (s, e)
            )

        if target_hits:
            if len(pos_texts) < n_real_pos:
                pos_texts.append(text)
            if len(neg_texts) < n_real_neg:
                chars = list(text)
                for s, e in target_hits + other_hits:
                    for i in range(s, e):
                        chars[i] = " "
                clean = " ".join("".join(chars).split())
                if len(clean) > 20:
                    neg_texts.append(clean)
        elif other_hits and len(neg_texts) < n_real_neg:
            chars = list(text)
            for s, e in other_hits:
                for i in range(s, e):
                    chars[i] = " "
            clean = " ".join("".join(chars).split())
            if len(clean) > 20:
                neg_texts.append(clean)

    # synthetic PCI positives cover the 5 card types the dataset lacks
    while len(pos_texts) < limit:
        pos_texts.append(make_synthetic_pair()[0])

    while len(neg_texts) < limit:
        neg_texts.append(make_hard_negative(rng))

    return pos_texts[:limit], neg_texts[:limit]


def load_hybrid_items(limit: int, tok, cfg: dict) -> list[dict]:
    pos_texts, neg_texts = build_hybrid_texts(limit)
    items: list[dict] = []
    for t in pos_texts:
        it = build_training_item(t, True, tok, cfg)
        if it:
            items.append(it)
    for t in neg_texts:
        it = build_training_item(t, False, tok, cfg)
        if it:
            items.append(it)
    print(
        f"[data] hybrid: {len(pos_texts)} positives + {len(neg_texts)} negatives "
        f"-> {len(items)} items"
    )
    return items


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument(
        "--data", choices=["synthetic", "ai4privacy", "hybrid"], default="synthetic"
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="cap on positive examples (items = 2x this)",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--output-dir", default=None, help="if set, write a merged Laya checkpoint here"
    )
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from transformers import AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[setup] device={device}")

    model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    # page-of-OCR sized sequences for the smoke run; raise for the full run
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = min(256, args.max_len // 2)
    cfg["gradient_checkpointing"] = True

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    from safetensors.torch import load_file

    model.load_state_dict(
        load_file(os.path.join(model_dir, "model.safetensors")), strict=True
    )

    # LoRA over the frozen ModernBERT encoder linears; head trains fully.
    # Wrap only the encoder submodule: peft must not wrap DecisionModel itself,
    # whose non-standard forward signature confuses peft's inputs_embeds path.
    from peft import LoraConfig, get_peft_model

    model.encoder = get_peft_model(
        model.encoder,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["Wqkv", "Wo", "Wi"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] LoRA trainable params: {trainable:,}")

    try:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except (ValueError, AttributeError) as e:
        print(f"[setup] gradient checkpointing skipped: {e}")

    # data
    t0 = time.time()
    if args.data == "ai4privacy":
        items = load_ai4privacy_items(args.max_samples * 2, tok, cfg)
    elif args.data == "hybrid":
        items = load_hybrid_items(args.max_samples, tok, cfg)
    else:
        items = load_synthetic_items(args.max_samples, tok, cfg)
    random.shuffle(items)
    print(f"[data] {len(items)} training items in {time.time() - t0:.1f}s")

    if not items:
        sys.exit("[data] no items built — aborting")

    # optimizer split: LoRA adapters vs head
    lora_params = [
        p for n, p in model.named_parameters() if p.requires_grad and "lora" in n
    ]
    head_params = [
        p for n, p in model.named_parameters() if p.requires_grad and "lora" not in n
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 2.5e-5},
            {"params": head_params, "lr": 1.0e-4},
        ],
        weight_decay=0.01,
    )

    EPOCHS = args.epochs
    MICRO_BATCH = args.micro_batch
    GRAD_ACCUM = args.grad_accum

    total_updates = (len(items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    model.to(device)
    model.train()
    pad_id = tok.pad_token_id

    print(
        f"[train] {EPOCHS} epochs, micro_batch={MICRO_BATCH}, grad_accum={GRAD_ACCUM}, "
        f"updates={total_updates}"
    )

    for epoch in range(EPOCHS):
        random.shuffle(items)
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        epoch_loss = 0.0
        n_batches = 0

        for b_idx in range(0, len(items), MICRO_BATCH):
            chunk = items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, pad_id)
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            mpos = batch["marker_pos"].to(device)
            mmask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                logits, _ = model(ids, att, mpos, mmask, qtype)

            logits = logits.float()
            loss = (
                torch.nn.functional.cross_entropy(logits, batch["label"].to(device))
                / GRAD_ACCUM
            )

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % 20 == 0:
                print(
                    f"  epoch {epoch + 1}/{EPOCHS} step {n_batches} loss "
                    f"{loss.item() * GRAD_ACCUM:.4f}"
                )

        print(
            f"=== epoch {epoch + 1}/{EPOCHS} done in {time.time() - t0:.1f}s | "
            f"avg loss {epoch_loss / max(1, n_batches):.4f} ==="
        )

    # held-out eval across three buckets: live PII, clean prose/near-miss, masked forms
    model.eval()
    rng_eval = random.Random(args.seed ^ 0x5EED)
    pos_texts = [make_synthetic_pair()[0] for _ in range(32)]
    clean_neg = [
        rng_eval.choice(CLEAN_TEMPLATES)
        if i % 2 == 0
        else f"Order reference {make_non_luhn_pan()} will ship within two business days."
        for i in range(32)
    ]
    masked_neg = [make_synthetic_pair()[1] for _ in range(32)]

    def score_bucket(texts, label):
        items = [
            it for it in (build_training_item(t, label, tok, cfg) for t in texts) if it
        ]
        correct = 0
        with torch.no_grad():
            for b_idx in range(0, len(items), 16):
                batch = collate_train_batch(items[b_idx : b_idx + 16], pad_id)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=(device.type == "cuda")
                ):
                    logits, _ = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["marker_pos"].to(device),
                        batch["marker_mask"].to(device),
                        batch["qtype"].to(device),
                    )
                preds = logits.float().argmax(-1).cpu()
                correct += (preds == batch["label"]).sum().item()
        return correct, len(items)

    pc, pn = score_bucket(pos_texts, True)
    cc, cn = score_bucket(clean_neg, False)
    mc, mn = score_bucket(masked_neg, False)
    print(f"[eval] live-PII {pc}/{pn} | clean {cc}/{cn} | masked {mc}/{mn}")

    if args.no_save or not args.output_dir:
        print("[save] skipped (--no-save or no --output-dir)")
        return

    # fit noul temperature, merge LoRA, write a standard Laya checkpoint
    del optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    calib_preds = []
    calib = items[:: max(1, len(items) // 400)][:400]
    with torch.no_grad():
        for b_idx in range(0, len(calib), 16):
            chunk = calib[b_idx : b_idx + 16]
            batch = collate_train_batch(chunk, pad_id)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                l_sub, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            l_np = l_sub.float().cpu().numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))
    fitted_temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
        if sel:
            fitted_temps[qt] = fit_one_temp(sel)
    print(
        f"[save] fitted temperatures (choice,score,noul): {[round(x, 3) for x in fitted_temps]}"
    )

    model.encoder = model.encoder.merge_and_unload()
    merged = model
    os.makedirs(args.output_dir, exist_ok=True)
    from safetensors.torch import save_file

    sd = {k: v.half().contiguous().cpu() for k, v in merged.state_dict().items()}
    save_file(sd, os.path.join(args.output_dir, "model.safetensors"))
    merged.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-pii-gate"
    cfg["temperature"] = fitted_temps
    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[save] checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    main()
