#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Merge the PR #28 dense verify-latency curve with (a) the PR #33 Marlin
tile-boundary fine sweep and (b) the PR #33 tree-causal mask saving, producing a
cost-model JSON that `tree_acceptance_model.py --cost-model-json` can read.

Why a merge step (PR #33 Step 3):
  - The coarse PR #28 sweep has no M=33 or M=49 sample; `LatencyCurve` interpolates
    them LINEARLY across a Marlin tile step (M=32->40 and M=48->64). Because the
    GEMM jumps at the tile boundary (thread_m_blocks = ceil(M/16)), that linear
    interpolation UNDER-states the boundary M values. The tile-boundary sweep
    measures M=33 and M=49 DIRECTLY, so we fold those in.
  - The tree-causal mask only shrinks the (small) among-token core-attention block.
    We subtract the measured/ideal saving at M=25/33/49 to get the tree-masked
    verify latency the tree model should use at those exact tree shapes.

Saving lenses (from results_tree_mask.json):
  - none      : tree-masked = dense (control = corrected dense curve)
  - sdpa      : production-realistic (dense attention + topology mask) -> saving ~0
  - flopideal : unrealisable element-sparse ceiling -> the most optimistic saving
"""
from __future__ import annotations

import argparse
import json
import os


def _curve(d: dict, key: str) -> dict[int, float]:
    node = d["cost_model"][key]
    return {int(m): float(v) for m, v in node["latency_ms_by_M"].items()}


def _attn_pct(d: dict, key: str) -> dict[int, float]:
    node = d["cost_model"][key]
    apc = node.get("attention_pct_step_by_M", {})
    return {int(m): float(v) for m, v in apc.items()}


def _interp(table: dict[int, float], M: float) -> float:
    xs = sorted(table)
    if M <= xs[0]:
        return table[xs[0]]
    if M >= xs[-1]:
        return table[xs[-1]]
    lo = max(x for x in xs if x <= M)
    hi = min(x for x in xs if x >= M)
    if lo == hi:
        return table[lo]
    t = (M - lo) / (hi - lo)
    return table[lo] * (1 - t) + table[hi] * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msweep", default="research/spec_cost_model/results_msweep.json")
    ap.add_argument("--tile-boundary", default=None,
                    help="optional fine-sweep curve to fold measured boundary M in")
    ap.add_argument("--tree-mask",
                    default="research/spec_cost_model/results_tree_mask.json")
    ap.add_argument("--key", default="graph|ctx256")
    ap.add_argument("--saving", choices=["none", "sdpa", "flopideal"],
                    default="flopideal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = _curve(json.load(open(args.msweep)), args.key)
    base_apc = _attn_pct(json.load(open(args.msweep)), args.key)
    merged = dict(base)
    provenance = {m: "msweep" for m in base}

    # (a) fold in directly-measured tile-boundary points (prefer fresh measurement)
    if args.tile_boundary and os.path.exists(args.tile_boundary):
        tb = _curve(json.load(open(args.tile_boundary)), args.key)
        shared = sorted(set(base) & set(tb))
        offs = {m: base[m] - tb[m] for m in shared}
        print(f"[merge] tile-boundary continuity at shared M={shared}:")
        for m in shared:
            print(f"[merge]   M={m:3d}  msweep={base[m]:7.3f}  tile={tb[m]:7.3f}  "
                  f"Δ(msweep-tile)={offs[m]:+.3f} ms")
        if shared:
            mean_off = sum(offs.values()) / len(offs)
            print(f"[merge]   mean offset {mean_off:+.3f} ms (run-to-run/thermal); "
                  f"using measured tile points as-is")
        for m, v in tb.items():
            merged[m] = v
            provenance[m] = "tile-boundary(measured)"

    # (b) apply the tree-causal mask saving at the tree M points
    tmd = json.load(open(args.tree_mask))
    tree_pts = {}
    for r in tmd["rows"]:
        M = int(r["M"])
        dense_at_M = _interp(merged, M)  # measured/merged dense at this M
        if args.saving == "none":
            save = 0.0
        elif args.saving == "sdpa":
            save = max(0.0, float(r["delta_sdpa_ms"]))
        else:  # flopideal
            save = float(r["delta_flopideal_vllmcal_ms"])
        tree_v = dense_at_M - save
        merged[M] = tree_v
        provenance[M] = f"tree-mask({args.saving}, dense={dense_at_M:.3f}-{save:.3f})"
        tree_pts[M] = {"dense_merged": dense_at_M, "saving_ms": save,
                       "tree_masked": tree_v}
        print(f"[merge] tree M={M:2d}: dense(merged)={dense_at_M:7.3f}  "
              f"-{save:.4f} ({args.saving})  -> tree-masked={tree_v:7.3f} ms")

    # attention-pct carried (interpolated) for completeness; tree model ignores it
    apc_out = {str(m): (base_apc.get(m) if m in base_apc else
                        round(_interp(base_apc, m), 4) if base_apc else None)
               for m in sorted(merged)}
    out = {
        "config": {
            "source_msweep": args.msweep, "source_tile_boundary": args.tile_boundary,
            "source_tree_mask": args.tree_mask, "key": args.key,
            "saving_lens": args.saving, "tree_points": tree_pts,
            "provenance": {str(m): provenance[m] for m in sorted(merged)},
        },
        "cost_model": {
            args.key: {
                "latency_ms_by_M": {str(m): merged[m] for m in sorted(merged)},
                "attention_pct_step_by_M": apc_out,
            }
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[merge] wrote {args.out}  ({len(merged)} M points, saving={args.saving})")


if __name__ == "__main__":
    main()
