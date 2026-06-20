#!/usr/bin/env python
"""Single-stream decode-TPS probe (PR #814) against an already-running server.

Uses harness.probe_tps -- the SAME single-stream metric as stark #798's int4head
256.74 reference (steady-state decode, prefill subtracted via the 1-tok/N-tok
trick). capture_decode/measure_arm.py instead reports sequential N-prompt wall
throughput (official-leaderboard-style, includes prefill); this script reports
the prefill-excluded steady-state number so g128 is directly comparable to the
256.74 anchor.

LOCAL ONLY -- talks to a live local server; launches nothing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness  # noqa: E402

INT4HEAD_SINGLE_STREAM_TPS = 256.74  # stark #798 probe_tps reference


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8021")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples: list[dict] = []
    for r in range(args.reps):
        res = harness.probe_tps(
            args.base_url, args.model, decode_tokens=args.decode_tokens
        )
        samples.append(res)
        print(
            f"  rep{r}: decode_tps_single_stream={res['decode_tps_single_stream']:.2f} "
            f"naive_tps={res['naive_tps']:.2f} ttft={res['ttft_s_approx']:.3f}s "
            f"n={res['decode_tokens']}",
            flush=True,
        )

    ss = [s["decode_tps_single_stream"] for s in samples if s["decode_tps_single_stream"] == s["decode_tps_single_stream"]]
    rec = {
        "arm": args.arm,
        "metric": "probe_tps_single_stream_decode",
        "reps": args.reps,
        "decode_tokens": args.decode_tokens,
        "decode_tps_single_stream": ss,
        "mean": statistics.fmean(ss) if ss else None,
        "std": statistics.pstdev(ss) if len(ss) > 1 else 0.0,
        "median": statistics.median(ss) if ss else None,
        "int4head_ref": INT4HEAD_SINGLE_STREAM_TPS,
    }
    if rec["mean"] is not None:
        rec["delta_pct_vs_int4head_ref"] = round(
            100.0 * (rec["mean"] - INT4HEAD_SINGLE_STREAM_TPS) / INT4HEAD_SINGLE_STREAM_TPS, 3
        )
    print("\n[probe] ===== RESULT =====", flush=True)
    print(
        f"  arm={rec['arm']} mean={rec['mean']:.2f} std={rec['std']:.2f} "
        f"median={rec['median']:.2f} (n={len(ss)} reps)",
        flush=True,
    )
    print(
        f"  vs int4head probe_tps ref {INT4HEAD_SINGLE_STREAM_TPS}: "
        f"delta={rec.get('delta_pct_vs_int4head_ref')}%",
        flush=True,
    )
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
