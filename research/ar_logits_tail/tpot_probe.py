#!/usr/bin/env python
"""Decode-only steady-state probe (TPOT) for the AR logits->token tail card (#604).

Streams a single long greedy completion and times inter-token arrival. The
median inter-token gap (TPOT) is the decode-only steady-state step time, free of
prefill (that lands in TTFT, the first gap). Streaming SSE adds a tiny per-token
output-path cost -- which is itself part of the host-side tail this card prices,
so we report both the streaming TPOT and (separately) the non-streaming wall.

LOCAL profiling only. temp=0 greedy, ignore_eos, return_token_ids per the
canonical #319 predicate.
"""
from __future__ import annotations
import argparse, json, statistics, time, urllib.request
from pathlib import Path


def read_first_sharegpt_prompt(path: str, seed: int, idx: int) -> str:
    import random
    data = json.loads(Path(path).read_text())
    recs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if isinstance(conv, list) and len(conv) >= 2 and isinstance(conv[0], dict):
            v = conv[0].get("value")
            if isinstance(v, str) and v:
                recs.append(v)
    random.Random(seed).shuffle(recs)
    return recs[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--tokenizer", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--dataset", default="official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-idx", type=int, default=0)
    ap.add_argument("--out", default="research/ar_logits_tail/tpot_result.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ptext = read_first_sharegpt_prompt(args.dataset, seed=1, idx=args.prompt_idx)
    raw = tok.apply_chat_template([{"role": "user", "content": ptext}],
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

    def stream_once(max_tokens: int):
        payload = {"model": args.model, "prompt": ids, "max_tokens": max_tokens,
                   "temperature": 0.0, "stream": True, "add_special_tokens": False,
                   "ignore_eos": True, "min_tokens": 8}
        req = urllib.request.Request(args.base_url + "/v1/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        t_send = time.perf_counter()
        stamps = []
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                txt = obj.get("choices", [{}])[0].get("text", "")
                if txt != "":
                    stamps.append(time.perf_counter())
        return t_send, stamps

    # warmup
    for _ in range(args.warmup):
        stream_once(64)

    all_tpot_ms = []
    ttft_ms = []
    wall_tps_list = []
    for _ in range(args.reps):
        t_send, stamps = stream_once(args.output_len)
        if len(stamps) < 10:
            continue
        ttft_ms.append((stamps[0] - t_send) * 1000.0)
        gaps = [(stamps[i] - stamps[i - 1]) * 1000.0 for i in range(1, len(stamps))]
        # steady window: drop first 8 (warm decode/graph) and last 2
        steady = gaps[8:-2] if len(gaps) > 12 else gaps
        all_tpot_ms.extend(steady)
        wall = stamps[-1] - t_send
        wall_tps_list.append((len(stamps)) / wall)

    med_tpot = statistics.median(all_tpot_ms)
    mean_tpot = statistics.mean(all_tpot_ms)
    p10 = statistics.quantiles(all_tpot_ms, n=10)[0]
    res = {
        "n_tokens_timed": len(all_tpot_ms),
        "median_tpot_ms": med_tpot,
        "mean_tpot_ms": mean_tpot,
        "p10_tpot_ms": p10,
        "decode_only_steady_tps_median": 1000.0 / med_tpot,
        "decode_only_steady_tps_p10gap": 1000.0 / p10,
        "ttft_ms_mean": statistics.mean(ttft_ms) if ttft_ms else None,
        "stream_wall_tps_mean": statistics.mean(wall_tps_list) if wall_tps_list else None,
        "note": "streaming TPOT; gap excludes prefill (in TTFT). steady=drop first8/last2.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
