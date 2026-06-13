#!/usr/bin/env python
"""Local single-stream (concurrency=1) output-TPS probe for the served endpoint.

Proxy for the official sglang.bench_serving run (MAX_CONCURRENCY=1, OUTPUT_LEN=512,
sharegpt prompts, 4 warmups, seed 1). This is a LOCAL A10G exploratory number, not
the official HF-Jobs score. Reports output_tps = sum(completion_tokens)/sum(latency)
over the measured requests, matching sglang's output_throughput at concurrency 1.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def completion_len(resp: dict, prompt_ids: list[int]) -> int:
    ch = resp["choices"][0]
    for v in (ch.get("token_ids"), ch.get("output_token_ids")):
        if isinstance(v, list) and all(isinstance(t, int) for t in v):
            if len(v) >= len(prompt_ids) and v[:len(prompt_ids)] == prompt_ids:
                return len(v) - len(prompt_ids)
            return len(v)
    usage = resp.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])
    raise ValueError("cannot determine completion length")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--tokenizer", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--dataset", default="official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--measure", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="research/_probe/tps_probe_g128_head128.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.loads(Path(args.dataset).read_text())
    prompts = []
    for it in data:
        conv = it.get("conversations")
        if isinstance(conv, list) and conv and isinstance(conv[0], dict):
            v = conv[0].get("value")
            if isinstance(v, str) and v:
                prompts.append(v)
    random.Random(args.seed).shuffle(prompts)

    def encode(p: str) -> list[int]:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if isinstance(ids, dict):
            ids = ids.get("input_ids", ids)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]

    def one(prompt_ids: list[int]) -> tuple[float, int]:
        payload = {
            "model": args.model, "prompt": prompt_ids, "max_tokens": args.output_len,
            "temperature": 0.0, "stream": False, "add_special_tokens": False,
            "ignore_eos": True, "return_token_ids": True,
        }
        t0 = time.perf_counter()
        resp = post(f"{args.base_url}/v1/completions", payload)
        dt = time.perf_counter() - t0
        return dt, completion_len(resp, prompt_ids)

    pool = [encode(p) for p in prompts]
    print(f"[warmup] {args.warmup} requests ...", flush=True)
    for i in range(args.warmup):
        one(pool[i % len(pool)])

    total_dt = 0.0
    total_out = 0
    per_req = []
    print(f"[measure] {args.measure} requests (concurrency=1, output_len={args.output_len}) ...", flush=True)
    for i in range(args.measure):
        dt, n = one(pool[(args.warmup + i) % len(pool)])
        total_dt += dt
        total_out += n
        per_req.append(n / dt if dt else 0.0)
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{args.measure} (req_tps={n/dt:.1f}, running_out_tps={total_out/total_dt:.1f})", flush=True)

    summary = {
        "output_tps": total_out / total_dt,
        "mean_per_request_tps": sum(per_req) / len(per_req),
        "total_completion_tokens": total_out,
        "total_decode_wall_s": total_dt,
        "measured_requests": args.measure,
        "output_len": args.output_len,
        "concurrency": 1,
        "note": "LOCAL A10G single-stream proxy for sglang.bench_serving output_throughput; not official",
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("[tps] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
