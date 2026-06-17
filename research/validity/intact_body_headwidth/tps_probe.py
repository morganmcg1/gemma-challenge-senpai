#!/usr/bin/env python3
"""Single-stream decode-TPS probe for the intact-body head-width sweep (PR #547).

Sends K sequential (concurrency=1) chat completions, each forced to emit exactly
GEN decode tokens (ignore_eos + min_tokens=GEN, max_tokens=GEN), and reports the
steady-state single-stream tokens/sec. The first request is a warmup and dropped.

This is an UNOFFICIAL local TPS (official_tps=0 for this analysis PR); it exists
only to chart the speed axis of the head-width knob:
  * off (262144 full head)  -> full-head GEMV TPS baseline.
  * slice (K-row head)      -> genuine pruned-head GEMV TPS (mask mode would report
    full-head speed because it does not reduce lm_head FLOPs).

TPS is reported two ways: end-to-end (sum tokens / sum latency over the timed
requests) and median per-request (tokens / per-request latency). Both exclude the
warmup. Short prompt => latency is decode-dominated, so this approximates the
per-token decode throughput a single user sees.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

PROMPT = "Write a detailed explanation of how a four-stroke internal combustion engine works."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--gen", type=int, default=256, help="forced decode tokens per request")
    ap.add_argument("--k", type=int, default=12, help="timed requests (excludes 1 warmup)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lat = []
    toks = []
    n_req = args.k + 1  # +1 warmup
    for i in range(n_req):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.gen,
            "min_tokens": args.gen,
            "ignore_eos": True,
        }
        t0 = time.perf_counter()
        r = requests.post(f"{args.base_url}/chat/completions", json=body, timeout=600)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        d = r.json()
        ct = int(d.get("usage", {}).get("completion_tokens") or 0)
        if i == 0:
            print(f"[tps:{args.label}] warmup: {ct} tok in {dt:.3f}s "
                  f"({ct/dt:.1f} tok/s) [dropped]", flush=True)
            continue
        lat.append(dt)
        toks.append(ct)
        print(f"[tps:{args.label}] req {i}: {ct} tok in {dt:.3f}s ({ct/dt:.1f} tok/s)",
              flush=True)

    total_tok = sum(toks)
    total_lat = sum(lat)
    e2e_tps = total_tok / total_lat if total_lat else 0.0
    per_req = [t / l for t, l in zip(toks, lat) if l > 0]
    med_tps = statistics.median(per_req) if per_req else 0.0
    mean_tps = statistics.mean(per_req) if per_req else 0.0
    res = {
        "label": args.label,
        "gen_tokens_per_req": args.gen,
        "k_timed": len(lat),
        "completion_tokens_total": total_tok,
        "latency_total_s": total_lat,
        "tps_end_to_end": e2e_tps,
        "tps_median_per_req": med_tps,
        "tps_mean_per_req": mean_tps,
        "per_req_latency_s": lat,
        "per_req_tokens": toks,
        "official_tps": 0,
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(f"[tps:{args.label}] SINGLE-STREAM TPS: e2e={e2e_tps:.1f} "
          f"median={med_tps:.1f} mean={mean_tps:.1f} tok/s "
          f"(K={len(lat)} x {args.gen} tok) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
