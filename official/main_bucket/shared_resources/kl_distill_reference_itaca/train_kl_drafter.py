#!/usr/bin/env python3
"""Reference KL-distillation training loop for the Gemma4 MTP drafter.

Init from `Tonykip/gemma4-e4b-mtp-drafter-ft@ft-v1-epoch_000` (or any
checkpoint with the same `Gemma4AssistantForCausalLM` architecture +
`gemma4_assistant` model_type). One epoch of KL-divergence loss against
the int4 target's top-k softmax over a propose-call trace corpus.

This is a reference, not a benchmark-tuned recipe — it emphasizes
correctness and readability over speed. Drop-in suitable for a single H100;
~1M records × 1 epoch finishes in a few hours.

Trace record schema (one JSON per line):

    {
        "prefix_token_ids": [int, ...],     # context fed to the drafter
        "target_topk_ids":  [int, ...],     # k token IDs the target gives mass to (k = 2048)
        "target_topk_probs": [float, ...],  # softmax probabilities, sum -> 1.0
        "target_argmax_id": int,            # for hybrid loss + sanity
        "draft_position":   int,            # 0..K-1 within the propose call
    }

The drafter is wrapped to expose logits over the **same vocab subset**
indexed by `target_topk_ids` per record (the loss is computed on the
drafter's full-vocab logits gathered at those indices, then softmaxed).

Usage:

    python train_kl_drafter.py \
        --init Tonykip/gemma4-e4b-mtp-drafter-ft \
        --init-revision ft-v1-epoch_000 \
        --corpus ./corpus/train.jsonl \
        --epochs 1 --batch-size 64 --lr 2e-4 \
        --alpha 0.0 --temperature 1.0 \
        --out ./drafter-ft-kl-epoch_001/

`--alpha` is the hybrid-loss weight on argmax CE: `L = α·CE + (1-α)·KL`.
`--alpha 0.0` is pure KL (DeepSeek-V3 recipe). `--alpha 0.3-0.5` is the
LK-hybrid kenyan-duma cited in their paxenos reply.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TraceRecord:
    prefix: torch.Tensor          # [seq_len] int64
    topk_ids: torch.Tensor        # [k] int64
    topk_probs: torch.Tensor      # [k] float32
    argmax_id: int


class TraceJSONL(IterableDataset):
    """Streams JSONL records lazily — corpora are >1M records, don't fit in RAM."""

    def __init__(self, path: str, max_prefix_len: int = 4096):
        self.path = path
        self.max_prefix_len = max_prefix_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        nw = worker_info.num_workers if worker_info else 1
        with open(self.path) as fh:
            for i, line in enumerate(fh):
                if i % nw != wid:
                    continue
                rec = json.loads(line)
                prefix = rec["prefix_token_ids"]
                if len(prefix) > self.max_prefix_len:
                    prefix = prefix[-self.max_prefix_len:]
                yield TraceRecord(
                    prefix=torch.tensor(prefix, dtype=torch.long),
                    topk_ids=torch.tensor(rec["target_topk_ids"], dtype=torch.long),
                    topk_probs=torch.tensor(rec["target_topk_probs"], dtype=torch.float32),
                    argmax_id=int(rec["target_argmax_id"]),
                )


def collate(batch: list[TraceRecord], pad_id: int):
    """Right-pad prefixes to the batch max so we can run a single forward."""
    seq_max = max(r.prefix.size(0) for r in batch)
    k = batch[0].topk_ids.size(0)
    B = len(batch)
    prefix = torch.full((B, seq_max), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(B, seq_max, dtype=torch.bool)
    last_pos = torch.zeros(B, dtype=torch.long)
    topk_ids = torch.zeros(B, k, dtype=torch.long)
    topk_probs = torch.zeros(B, k, dtype=torch.float32)
    argmax_id = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(batch):
        L = r.prefix.size(0)
        prefix[i, :L] = r.prefix
        attn_mask[i, :L] = True
        last_pos[i] = L - 1
        topk_ids[i] = r.topk_ids
        topk_probs[i] = r.topk_probs
        argmax_id[i] = r.argmax_id
    return prefix, attn_mask, last_pos, topk_ids, topk_probs, argmax_id


def kl_loss(drafter_logits: torch.Tensor, topk_ids: torch.Tensor, topk_probs: torch.Tensor,
            temperature: float = 1.0) -> torch.Tensor:
    """KL(target_topk_probs || drafter_softmax_at_topk_ids), per-batch mean."""
    gathered = drafter_logits.gather(1, topk_ids)   # [B, k]
    log_q = F.log_softmax(gathered / temperature, dim=-1)
    p = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    log_p = (p.clamp(min=1e-12)).log()
    return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


def ce_loss(drafter_logits: torch.Tensor, argmax_id: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(drafter_logits, argmax_id)


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    tok = AutoTokenizer.from_pretrained(args.init, revision=args.init_revision, trust_remote_code=True)
    pad_id = tok.pad_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(
        args.init,
        revision=args.init_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.train()

    ds = TraceJSONL(args.corpus, max_prefix_len=args.max_prefix_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=lambda b: collate(b, pad_id),
        pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    step = 0
    running = 0.0
    for epoch in range(args.epochs):
        for prefix, attn_mask, last_pos, topk_ids, topk_probs, argmax_id in loader:
            prefix = prefix.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            last_pos = last_pos.to(device, non_blocking=True)
            topk_ids = topk_ids.to(device, non_blocking=True)
            topk_probs = topk_probs.to(device, non_blocking=True)
            argmax_id = argmax_id.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(input_ids=prefix, attention_mask=attn_mask, use_cache=False)
                # Per-batch select the logits at the last (real) position of each row.
                B = prefix.size(0)
                last_logits = out.logits[torch.arange(B), last_pos]   # [B, V]

                loss_kl = kl_loss(last_logits, topk_ids, topk_probs, temperature=args.temperature)
                loss_ce = ce_loss(last_logits, argmax_id) if args.alpha > 0 else torch.tensor(0.0, device=device)
                loss = args.alpha * loss_ce + (1.0 - args.alpha) * loss_kl

            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            running = 0.99 * running + 0.01 * float(loss) if step else float(loss)
            step += 1
            if step % args.log_every == 0:
                print(f"[train] epoch={epoch} step={step:>7} loss={float(loss):.4f} ema={running:.4f} "
                      f"kl={float(loss_kl):.4f} ce={float(loss_ce):.4f}")

        ckpt = os.path.join(args.out, f"epoch_{epoch:03d}")
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
        print(f"[train] saved {ckpt}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="HF repo id or local path")
    ap.add_argument("--init-revision", default=None)
    ap.add_argument("--corpus", required=True, help="JSONL trace stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Hybrid loss weight on argmax CE (0.0 = pure KL, DeepSeek-V3 recipe)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-prefix-len", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
