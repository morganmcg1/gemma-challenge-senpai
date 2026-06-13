#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Quantify how the #43 split-KV verify patch reshaped the int4 cost(M) curve
(PR #51, Task 1).  Loads matched baseline vs split-KV cost-model JSONs (same
M-sweep, same ctx) and tabulates, per M and per context length:

  * verify step time (ms) baseline vs split-KV and the delta / %;
  * attention fraction baseline vs split-KV (the split-KV redirect target);
  * the GEMM staircase cliffs (M=17/33/49) -- unchanged by an attention-only
    patch, so the argmax K* (pinned just below a cliff) does not move.

Prints a per-ctx table and a compact summary (operating-point M=45 delta, cliff
magnitudes).  CPU-only; reads the JSONs produced by spec_cost_model.py.
"""
from __future__ import annotations

import argparse
import json


def load(path):
    d = json.load(open(path))
    return d.get("cost_model", {}), d.get("config", {})


def step_by_m(node):
    return {int(m): float(v) for m, v in node["latency_ms_by_M"].items()}


def attn_pct(node):
    a = node.get("attention_pct_step_by_M") or {}
    return {int(m): float(v) for m, v in a.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="baseline cost-model JSON(s)")
    ap.add_argument("--splitkv", nargs="+", required=True,
                    help="split-KV cost-model JSON(s)")
    ap.add_argument("--operating-M", type=int, default=45)
    args = ap.parse_args()

    base_cm = {}
    base_cfg = {}
    for p in args.baseline:
        cm, cfg = load(p)
        base_cm.update(cm)
        base_cfg = cfg
    sk_cm = {}
    sk_cfg = {}
    for p in args.splitkv:
        cm, cfg = load(p)
        sk_cm.update(cm)
        sk_cfg = cfg

    print(f"baseline splitkv_info: {base_cfg.get('splitkv_info')}")
    print(f"splitkv  splitkv_info: {json.dumps(sk_cfg.get('splitkv_info'))}")

    keys = [k for k in base_cm if k in sk_cm]
    summary = {}
    for key in sorted(keys):
        b = step_by_m(base_cm[key])
        s = step_by_m(sk_cm[key])
        ba = attn_pct(base_cm[key])
        sa = attn_pct(sk_cm[key])
        Ms = sorted(set(b) & set(s))
        print(f"\n===== {key} =====")
        print(f"{'M':>4} {'base ms':>8} {'splitkv':>8} {'Δms':>7} {'Δ%':>7} "
              f"{'attn%base':>9} {'attn%sk':>8} {'cliff':>6}")
        prev = None
        for M in Ms:
            d = s[M] - b[M]
            dp = 100 * d / b[M]
            cliff = ""
            if prev is not None and b[M] - b[prev] > 1.0:
                cliff = f"+{b[M]-b[prev]:.1f}"   # baseline GEMM cliff entering M
            print(f"{M:>4} {b[M]:>8.3f} {s[M]:>8.3f} {d:>+7.3f} {dp:>+6.1f}% "
                  f"{ba.get(M,0):>8.1f}% {sa.get(M,0):>7.1f}% {cliff:>6}")
            prev = M
        opM = args.operating_M
        if opM in b and opM in s:
            summary[key] = {
                "op_M": opM, "base_ms": b[opM], "splitkv_ms": s[opM],
                "delta_ms": s[opM] - b[opM], "delta_pct": 100 * (s[opM] - b[opM]) / b[opM],
                "attn_pct_base": ba.get(opM), "attn_pct_splitkv": sa.get(opM),
            }

    print("\n===== OPERATING-POINT (M=%d) SUMMARY =====" % args.operating_M)
    print(f"{'ctx':>16} {'base ms':>8} {'splitkv':>8} {'Δms':>7} {'Δ%':>7} "
          f"{'attn%b':>7} {'attn%s':>7}")
    for key, v in summary.items():
        print(f"{key:>16} {v['base_ms']:>8.3f} {v['splitkv_ms']:>8.3f} "
              f"{v['delta_ms']:>+7.3f} {v['delta_pct']:>+6.1f}% "
              f"{v['attn_pct_base'] or 0:>6.1f}% {v['attn_pct_splitkv'] or 0:>6.1f}%")


if __name__ == "__main__":
    main()
