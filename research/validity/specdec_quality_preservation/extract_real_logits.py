#!/usr/bin/env python
"""Extract real first-token target SAMPLING distributions for the #497
reasoning/STEM prompts from the deployed int4 target model, under the deployed
generation_config.json sampling params (do_sample, T=1.0, top_k=64, top_p=0.95).

The target sampling distribution p = top_p(top_k(softmax(logits / T))) is exactly
the distribution the standard rejection sampler must preserve. Saves a
{'p': [N, vocab], 'meta': [...]} blob for specdec_dist_preserve.py --logits.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForImageTextToText

MODEL = "/tmp/osoi5-v0-baked"
TEMPERATURE = 1.0
TOP_K = 64
TOP_P = 0.95


def gen_config_sampling_dist(logits: torch.Tensor) -> torch.Tensor:
    """Replicate HF/vLLM temperature->top_k->top_p filtering, return p over vocab."""
    logits = logits.float() / TEMPERATURE
    vocab = logits.numel()
    # top_k: keep top-K by logit
    if TOP_K and TOP_K < vocab:
        kth = torch.topk(logits, TOP_K).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    # top_p (nucleus): sort desc, keep smallest prefix with cumsum >= TOP_P
    sp, si = torch.sort(probs, descending=True)
    csum = torch.cumsum(sp, dim=-1)
    keep = csum < TOP_P
    keep[0] = True  # always keep the top token
    # also keep the first token that crosses the threshold (shift)
    keep[1:] = keep[1:] | (csum[:-1] < TOP_P)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[si[keep]] = True
    p = torch.where(mask, probs, torch.zeros_like(probs))
    p = p / p.sum()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="/workspace/senpai/target/research/validity/private_attention_flip_bound/shifted_reasoning_stem.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="real_reasoning_p.pt")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.split)][: args.n]
    print(f"loading {MODEL} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    print("loaded", flush=True)

    P, meta = [], []
    for r in recs:
        ids = torch.tensor([r["context_token_ids"]], device="cuda:0")
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1, :]
        p = gen_config_sampling_dist(logits).cpu()
        support = int((p > 0).sum())
        P.append(p)
        meta.append({"id": r["id"], "source": r["source"], "format": r["format"],
                     "domain": r["domain"], "support": support, "p_max": float(p.max())})
    P = torch.stack(P)  # [N, vocab]
    torch.save({"p": P, "meta": meta,
                "gen_config": {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P,
                               "do_sample": True, "model": MODEL}}, args.out)
    sup = [m["support"] for m in meta]
    pmax = [m["p_max"] for m in meta]
    print(f"[wrote] {args.out}  N={len(P)} vocab={P.shape[1]}")
    print(f"support: min={min(sup)} median={sorted(sup)[len(sup)//2]} max={max(sup)}")
    print(f"p_max:   min={min(pmax):.3f} median={sorted(pmax)[len(pmax)//2]:.3f} max={max(pmax):.3f}")


if __name__ == "__main__":
    main()
