#!/usr/bin/env python
"""Plain greedy autoregressive reference decode for the int4 QAT checkpoint.

Produces a `decode_outputs.jsonl` whose `completion_token_ids` come from a
transparent, one-token-at-a-time argmax loop in HF transformers over the SAME
int4 weights the vLLM endpoint serves. This is the "plain greedy AR decode of
the same submitted checkpoint" the challenge validity rule compares the
endpoint's greedy decode against.

It reads the candidate `decode_outputs.jsonl` (from the harness
`decode_outputs.py` run against the vLLM endpoint) and reuses each record's
`prompt_token_ids`, so the two files are prompt-aligned by `id`. EOS is never
treated as terminal (it can be emitted as a normal token), matching the
harness's `ignore_eos: true`. Output is written in the harness record shape so
`check_greedy_identity.py` can compare them directly.

Run this only after the vLLM server is stopped (both want the whole A10G).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def sha256_tokens(tokens: list[int]) -> str:
    body = ",".join(str(t) for t in tokens)
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    p.add_argument("--candidate", required=True, help="endpoint decode_outputs.jsonl (source of prompts)")
    p.add_argument("--output-file", required=True, help="reference decode_outputs.jsonl to write")
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="cap number of prompts (0 = all in candidate)")
    return p.parse_args()


@torch.inference_mode()
def greedy_ar(model, prompt_token_ids: list[int], output_len: int, device: str) -> list[int]:
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
    generated = [next_id]
    cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    for _ in range(output_len - 1):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated


def main() -> int:
    args = parse_args()
    device = "cuda:0"

    records = []
    with open(args.candidate) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"loaded {len(records)} prompt records from {args.candidate}", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s; running greedy AR ...", flush=True)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            prompt_token_ids = rec["prompt_token_ids"]
            ts = time.time()
            comp = greedy_ar(model, prompt_token_ids, args.output_len, device)
            row = {
                "id": rec["id"],
                "index": i,
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": comp,
                "completion_token_sha256": sha256_tokens(comp),
                "num_completion_tokens": len(comp),
            }
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            print(f"  [{i+1}/{len(records)}] id={rec['id']} tokens={len(comp)} ({time.time()-ts:.1f}s)", flush=True)

    print(f"reference written to {out_path} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
