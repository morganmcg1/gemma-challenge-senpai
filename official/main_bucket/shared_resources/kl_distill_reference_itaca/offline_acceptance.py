#!/usr/bin/env python3
"""Offline acceptance simulator for greedy spec-decode at K=7.

Why this exists: a drafter that beats kduma1 on KL training loss but
doesn't beat it on accepted-tokens/step is a drafter whose offline gain
will be eaten by the verifier's 5%-Δ TPS noise band (see
`shared_resources/tps_repro_gap_itaca/`). This script runs both drafters
on a held-out trace shard and reports accepted-tokens/step at depth 1..K
without launching vLLM.

The greedy spec-decode acceptance rule for one draft chain of length K:
  the chain accepts position i iff drafter_argmax[i] == target_argmax[i]
  AND positions 0..i-1 also accepted. (Compounding.) The accepted count
  per propose call is `bonus + 1` where `bonus` is the number of
  consecutive matches from position 0.

Trace-shard schema is identical to `train_kl_drafter.py`'s but enriched
with the next K-1 target argmaxes per call (i.e. the trace must be
captured along the actual greedy decode trajectory):

    {
        "prefix_token_ids": [int, ...],
        "target_argmaxes": [int, int, ..., int],   # length K
        ...                                         # other train fields
    }

Usage:

    python offline_acceptance.py \
        --drafter-baseline Tonykip/gemma4-e4b-mtp-drafter-ft \
        --baseline-revision ft-v1-epoch_000 \
        --drafter-candidate ./drafter-ft-kl-epoch_001/ \
        --traces ./corpus/heldout.jsonl \
        --K 7

Returns 0 only if candidate beats baseline by >= GATE accepted-tokens/step.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GATE = 0.05  # accepted-tokens/step uplift required to pass


def load_drafter(path: str, revision: str | None, device: str):
    tok = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    return tok, model


@torch.no_grad()
def step_draft(model, prefix: torch.Tensor) -> int:
    """Run one drafter forward; return argmax over the last position."""
    out = model(input_ids=prefix.unsqueeze(0), use_cache=False)
    return int(out.logits[0, -1].argmax())


@torch.no_grad()
def run(traces_path: str, drafter, tok, K: int, device: str, label: str):
    pad = tok.pad_token_id or 0
    accepted_total = 0
    calls = 0
    by_depth = defaultdict(int)
    with open(traces_path) as fh:
        for line in fh:
            rec = json.loads(line)
            prefix = torch.tensor(rec["prefix_token_ids"], dtype=torch.long, device=device)
            targets = rec["target_argmaxes"][:K]
            calls += 1
            ctx = prefix.clone()
            bonus = 0
            for i in range(K):
                draft_id = step_draft(drafter, ctx)
                if draft_id == targets[i]:
                    bonus += 1
                    by_depth[i] += 1
                    ctx = torch.cat([ctx, torch.tensor([draft_id], device=device)])
                else:
                    break
            accepted_total += bonus + 1   # +1 for the target's correction or argmax
    mean = accepted_total / calls if calls else 0.0
    by_depth_pct = {d: by_depth[d] / calls if calls else 0.0 for d in range(K)}
    print(f"[{label}] calls={calls} accepted/step={mean:.4f}")
    for d in range(K):
        print(f"  depth-{d+1} acc-rate: {by_depth_pct[d]*100:.2f}%")
    return mean, by_depth_pct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-baseline", required=True)
    ap.add_argument("--baseline-revision", default=None)
    ap.add_argument("--drafter-candidate", required=True)
    ap.add_argument("--candidate-revision", default=None)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_tok, base = load_drafter(args.drafter_baseline, args.baseline_revision, device)
    cand_tok, cand = load_drafter(args.drafter_candidate, args.candidate_revision, device)

    base_mean, _ = run(args.traces, base, base_tok, args.K, device, "baseline")
    cand_mean, _ = run(args.traces, cand, cand_tok, args.K, device, "candidate")

    delta = cand_mean - base_mean
    print(f"\nΔ accepted-tokens/step = {delta:+.4f}  (gate: {args.gate:+.4f})")
    if delta >= args.gate:
        print("PASS — candidate clears the gate; bench is justified.")
        return 0
    print("FAIL — candidate within training noise of baseline; do not spend bench-quota.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
