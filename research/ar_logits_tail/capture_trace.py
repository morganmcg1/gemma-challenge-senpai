#!/usr/bin/env python
"""Capture a torch-profiler trace of a steady AR decode window (#604).

Assumes the server was started with VLLM_TORCH_PROFILER_DIR set. Warms up, then
brackets a single greedy request with /start_profile and /stop_profile so the
trace contains one prefill + N steady decode steps on the int4_g128_lmhead AR
path. LOCAL profiling only.
"""
from __future__ import annotations
import argparse, json, time, urllib.request
from pathlib import Path


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--tokenizer", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--dataset", default="official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json")
    ap.add_argument("--decode-tokens", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.loads(Path(args.dataset).read_text())
    recs = []
    for item in data:
        if isinstance(item, dict):
            conv = item.get("conversations")
            if isinstance(conv, list) and len(conv) >= 2 and isinstance(conv[0], dict):
                v = conv[0].get("value")
                if isinstance(v, str) and v:
                    recs.append(v)
    random.Random(1).shuffle(recs)
    raw = tok.apply_chat_template([{"role": "user", "content": recs[0]}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(raw, "input_ids"):
        raw = raw.input_ids
    if isinstance(raw, dict):
        raw = raw.get("input_ids", raw)
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    ids = [int(t) for t in raw]

    def gen(n):
        return post(args.base_url + "/v1/completions", {
            "model": args.model, "prompt": ids, "max_tokens": n,
            "temperature": 0.0, "stream": False, "add_special_tokens": False,
            "ignore_eos": True, "min_tokens": 8})

    print("warmup...", flush=True)
    for _ in range(args.warmup):
        gen(64)
    print("start_profile", post(args.base_url + "/start_profile", {}), flush=True)
    time.sleep(0.2)
    t0 = time.perf_counter()
    gen(args.decode_tokens)
    dt = time.perf_counter() - t0
    print(f"profiled request: {args.decode_tokens} tokens in {dt*1000:.1f} ms "
          f"({args.decode_tokens/dt:.1f} tok/s incl prefill)", flush=True)
    print("stop_profile", post(args.base_url + "/stop_profile", {}), flush=True)
    time.sleep(3.0)  # let the trace flush
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
