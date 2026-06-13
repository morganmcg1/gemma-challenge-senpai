#!/usr/bin/env python
"""Offline teacher-forced PPL on the pruned checkpoint (no serving).

Loads the pruned checkpoint with transformers, runs the lm_head, scatters the
kept-row logits back to full vocab (-inf elsewhere) -- the same transform serve.py
applies inside vLLM -- and computes token-level PPL over the 128 ground-truth
records. This verifies the pruned checkpoint loads and that every scored GT target
token gets a finite logprob BEFORE spending a GPU serve/benchmark cycle.

Requires torch + the pruned checkpoint -> GPU window. Pure-CPU systems can run a
tiny --max-records subset on CPU but it is slow; intended for the A10G window.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT_FILE = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
DEFAULT_MODEL = "/workspace/gemma_build/lmhead12k_empirical"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gt-file", default=str(GT_FILE))
    ap.add_argument("--max-records", type=int, default=0, help="0 = all 128")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    kept = json.loads((Path(args.model) / "kept_ids.json").read_text())["kept_ids"]
    full_vocab = 262144
    kept_t = torch.tensor(kept, dtype=torch.long, device=args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    records = _read_jsonl(Path(args.gt_file))
    if args.max_records:
        records = records[: args.max_records]

    total_nll = 0.0
    total_tokens = 0
    finite_violations = 0
    for rec in records:
        ctx = rec["context_token_ids"]
        tgt = rec["target_token_ids"]
        ids = torch.tensor([ctx + tgt], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(ids)
            hidden = out.logits  # (1, T, kept_size) for a pruned-head model
        # Scatter kept-row logits -> full vocab (matches serve.py transform).
        T = hidden.shape[1]
        full = torch.full((T, full_vocab), float("-inf"),
                          dtype=hidden.dtype, device=args.device)
        full.scatter_(1, kept_t.unsqueeze(0).expand(T, -1), hidden[0])
        logp = torch.log_softmax(full.float(), dim=-1)
        # Score the target region: predict token t+1 from position t.
        start = len(ctx)
        for i in range(start, len(ctx) + len(tgt)):
            if i == 0:
                continue
            gold = (ctx + tgt)[i]
            lp = logp[i - 1, gold].item()
            if not math.isfinite(lp):
                finite_violations += 1
                continue
            total_nll += -lp
            total_tokens += 1

    ppl = math.exp(total_nll / max(1, total_tokens))
    print(json.dumps({
        "model": args.model,
        "records": len(records),
        "scored_tokens": total_tokens,
        "finite_violations": finite_violations,
        "ppl_offline": ppl,
        "ppl_gate": 2.42,
        "passes_gate": ppl <= 2.42 and finite_violations == 0,
    }, indent=2))


if __name__ == "__main__":
    main()
