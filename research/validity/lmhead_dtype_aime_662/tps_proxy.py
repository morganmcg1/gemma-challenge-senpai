#!/usr/bin/env python
"""Single-stream decode-TPS proxy for the lm_head-dtype arms (PR #662).

The official speed metric is sglang.bench_serving at MAX_CONCURRENCY=1, where decode
is memory-bandwidth-bound on per-step weight reads. The `lm_head` (262144x2560) is
read once per decode step, so int4->int8->bf16 raises HBM read/step (~+0.33 / ~+1.0
GB-per-token) and lowers decode TPS. This is a LOCAL proxy (FlashAttention + native
sampler, not the official sglang number); only the *cross-arm ratio* is meaningful,
since every arm is served byte-identically except the head dtype.

Method: stream a greedy (temperature=0, ignore_eos) completion and timestamp each
emitted token. Drop a warmup prefix, take the steady-state inter-token latency
(median) -> decode_tps = 1/median. A few warmup requests prime CUDA-graph/caches;
several measured reps -> report median across reps + the per-step latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

PROMPT = ("Write a long detailed explanation of how a four-stroke internal "
          "combustion engine works, step by step, in full prose.")


def stream_decode(base_url: str, model: str, n_tokens: int, warmup_drop: int) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": n_tokens,
        "min_tokens": n_tokens,   # force exactly n_tokens (vLLM ext)
        "ignore_eos": True,       # don't stop early
        "temperature": 0.0,
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t_send = time.perf_counter()
    stamps: list[float] = []
    ttft = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            txt = obj.get("choices", [{}])[0].get("text", "")
            now = time.perf_counter()
            if txt:
                if ttft is None:
                    ttft = now - t_send
                stamps.append(now)
    # inter-token latencies between consecutive emitted tokens
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    steady = deltas[warmup_drop:] if len(deltas) > warmup_drop else deltas
    med = statistics.median(steady) if steady else float("nan")
    mean = statistics.fmean(steady) if steady else float("nan")
    return {
        "n_emitted": len(stamps),
        "ttft_s": ttft,
        "decode_tps_median": (1.0 / med) if med and med == med else None,
        "decode_tps_mean": (1.0 / mean) if mean and mean == mean else None,
        "inter_token_ms_median": med * 1e3 if med == med else None,
        "steady_steps": len(steady),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n-tokens", type=int, default=600)
    ap.add_argument("--warmup-drop", type=int, default=50,
                    help="drop the first N inter-token deltas (graph capture / ramp)")
    ap.add_argument("--warmup-reps", type=int, default=2)
    ap.add_argument("--measure-reps", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"[tps] arm={args.arm} warmup x{args.warmup_reps} ...", flush=True)
    for _ in range(args.warmup_reps):
        stream_decode(args.base_url, args.model, 128, 20)

    reps = []
    for i in range(args.measure_reps):
        m = stream_decode(args.base_url, args.model, args.n_tokens, args.warmup_drop)
        reps.append(m)
        print(f"[tps] rep{i}: decode_tps_median={m['decode_tps_median']:.2f} "
              f"itl_ms={m['inter_token_ms_median']:.3f} emitted={m['n_emitted']} "
              f"ttft={m['ttft_s']:.3f}", flush=True)

    tps_med = [r["decode_tps_median"] for r in reps if r["decode_tps_median"]]
    out = {
        "arm": args.arm,
        "n_tokens": args.n_tokens,
        "warmup_drop": args.warmup_drop,
        "measure_reps": args.measure_reps,
        "decode_tps_median_across_reps": statistics.median(tps_med) if tps_med else None,
        "decode_tps_max_across_reps": max(tps_med) if tps_med else None,
        "inter_token_ms_median_across_reps":
            statistics.median([r["inter_token_ms_median"] for r in reps]) if reps else None,
        "reps": reps,
        "analysis_only": True,
        "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[tps] arm={args.arm} decode_tps_median_across_reps="
          f"{out['decode_tps_median_across_reps']:.2f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
