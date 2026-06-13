#!/usr/bin/env python
"""Exploratory single-stream output-TPS probe against a local vLLM endpoint.

Mirrors the official harness decode settings (max_concurrency=1, output_len=512,
temperature=0, ignore_eos, integer-token prompts via the chat template) so the
local number is comparable in shape to summary.json:output_tps. This is an
A10G-here exploratory measurement only, NOT an a10g-small leaderboard number.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


def read_sharegpt_prompts(path: Path, num_prompts: int, seed: int):
    data = json.loads(path.read_text())
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[0], dict):
            continue
        prompt = conv[0].get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        records.append({"id": str(item.get("id", index)), "prompt_text": prompt})
    random.Random(seed).shuffle(records)
    return records[:num_prompts]


def post(url, payload, timeout_s):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--summary-file", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = read_sharegpt_prompts(Path(args.dataset_path), args.num_prompts + args.warmup, args.seed)

    def encode(text):
        # transformers 5.9 returns a BatchEncoding from tokenize=True; unwrap to a
        # flat list[int] the same way the harness decode_outputs.py normalizes it.
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        if hasattr(enc, "input_ids"):
            enc = enc.input_ids
        if hasattr(enc, "tolist"):
            enc = enc.tolist()
        if isinstance(enc, dict):
            enc = enc.get("input_ids", enc)
        if len(enc) == 1 and isinstance(enc[0], (list, tuple)):
            enc = enc[0]
        return [int(t) for t in enc]

    per_req_tps = []
    total_out_tokens = 0
    total_gen_time = 0.0
    completed = 0
    for i, rec in enumerate(records):
        ptoks = encode(rec["prompt_text"])
        payload = {
            "model": args.model,
            "prompt": ptoks,
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "stream": False,
            "add_special_tokens": False,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        t0 = time.time()
        resp = post(f"{args.base_url}/v1/completions", payload, timeout_s=300)
        dt = time.time() - t0
        ch = resp["choices"][0]
        tids = ch.get("token_ids") or []
        n_out = ch.get("usage", {}).get("completion_tokens") if isinstance(ch.get("usage"), dict) else None
        if n_out is None:
            usage = resp.get("usage", {})
            n_out = usage.get("completion_tokens", len(tids))
        is_warmup = i < args.warmup
        tag = "warmup" if is_warmup else "measure"
        tps = n_out / dt if dt > 0 else 0.0
        print(f"  [{i+1}/{len(records)}] {tag} out_tokens={n_out} dt={dt:.2f}s tps={tps:.2f}", flush=True)
        if not is_warmup:
            per_req_tps.append(tps)
            total_out_tokens += n_out
            total_gen_time += dt
            completed += 1

    summary = {
        "exploratory": True,
        "note": "A10G-here single-stream local probe; NOT an a10g-small leaderboard number",
        "num_prompts_measured": completed,
        "output_len": args.output_len,
        "aggregate_output_tps": (total_out_tokens / total_gen_time) if total_gen_time else 0.0,
        "median_per_request_tps": statistics.median(per_req_tps) if per_req_tps else 0.0,
        "mean_per_request_tps": statistics.mean(per_req_tps) if per_req_tps else 0.0,
        "total_output_tokens": total_out_tokens,
        "total_generation_time_s": total_gen_time,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.summary_file:
        Path(args.summary_file).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
