#!/usr/bin/env python
"""Measure chat-templated token lengths of the official decode prompt set.

Replicates decode_outputs.read_sharegpt_prompts (shuffle seed 1, take N) and
encode_prompt (apply_chat_template add_generation_prompt=True), then reports the
length distribution. If max <= MAX_NUM_BATCHED_TOKENS, the prefill chunk cap can
never split a decode prompt, so serve.py's greedy decode is chunk-identical to a
default-config plain vLLM decode (the official greedy-identity reference).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json")
    ap.add_argument("--tokenizer", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    data = json.loads(Path(args.dataset).read_text())
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        first = conv[0]
        if not isinstance(first, dict):
            continue
        p = first.get("value")
        if not isinstance(p, str) or not p:
            continue
        records.append({"id": str(item.get("id", index)), "prompt_text": p})
    rng = random.Random(args.seed)
    rng.shuffle(records)
    sample = records[: args.num_prompts]

    def tmpl_len(p: str) -> int:
        enc = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        if hasattr(enc, "input_ids"):
            enc = enc.input_ids
        if isinstance(enc, dict):
            enc = enc.get("input_ids", enc)
        if hasattr(enc, "tolist"):
            enc = enc.tolist()
        if isinstance(enc, list) and enc and isinstance(enc[0], list):
            enc = enc[0]
        return len(enc)

    lens = sorted(tmpl_len(r["prompt_text"]) for r in sample)
    n = len(lens)
    over512 = sum(1 for L in lens if L > 512)
    print(json.dumps({
        "num_prompts": n,
        "min": lens[0],
        "p50": lens[n // 2],
        "p90": lens[int(n * 0.9)],
        "p99": lens[min(n - 1, int(n * 0.99))],
        "max": lens[-1],
        "num_over_512": over512,
        "num_over_2048": sum(1 for L in lens if L > 2048),
    }, indent=2))


if __name__ == "__main__":
    main()
