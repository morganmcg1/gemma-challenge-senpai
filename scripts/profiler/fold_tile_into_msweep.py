#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Fold the PR #33 directly-measured Marlin tile-boundary fine sweep into the
canonical PR #28 dense curve (results_msweep.json) IN PLACE (PR #37 Step 4).

WHY: the coarse PR #28 sweep has no M=17/33/49 sample, so `LatencyCurve`
interpolates the verify step LINEARLY across each Marlin tile cliff
(thread_m_blocks = ceil(M/16) jumps at M=17/33/49). The worst case is M=49: the
coarse curve interpolates 48->64 ACROSS the 4th-tile cliff and reports ~15.45 ms
where the boundary is really 18.13 ms (-14.8%). Any #26/#28 consumer reading
results_msweep.json WITHOUT the `--cost-model-json` tile override therefore
UNDER-states large-K tree verify cost and over-rewards deep K (this is the source
of the run aid45far K*=15-at-the-cap artifact; PR #37 Step 5).

WHAT: for the graph|ctx256 key, overlay the tile-boundary measured rows (M=16..52)
onto the msweep rows (tile wins on the 5 shared M; the run-to-run delta there is
<=0.054 ms, sub-thermal), keeping msweep's small-M (<16) and M=64 tail, then
re-run build_cost_model so EVERY derived field (latency, marginal, ideal/realistic
TPS, K*, knee, attention crossings) is recomputed consistently. The eager and
ctx512 keys are untouched (the tile sweep only measured graph|ctx256) and keep a
flag noting they still need their own boundary measurement.

A pre-fold copy is written to results_msweep_prefold.json for provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from spec_cost_model import build_cost_model

FOLD_KEY = ("graph", 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msweep", default="research/spec_cost_model/results_msweep.json")
    ap.add_argument("--tile", default="research/spec_cost_model/results_tile_boundary.json")
    ap.add_argument("--prefold-out",
                    default="research/spec_cost_model/results_msweep_prefold.json")
    args = ap.parse_args()

    msd = json.load(open(args.msweep))
    tbd = json.load(open(args.tile))
    accept_models = {k: (v[0], v[1]) for k, v in msd["config"]["accept_models"].items()}

    # ---- merge rows: tile-boundary overrides msweep at (graph,256) shared M ----
    ms_rows = msd["rows"]
    tb_rows = [r for r in tbd["rows"] if (r["mode"], r["ctx"]) == FOLD_KEY]
    tb_M = {r["M"] for r in tb_rows}
    kept_ms, dropped = [], []
    for r in ms_rows:
        if (r["mode"], r["ctx"]) == FOLD_KEY and r["M"] in tb_M:
            dropped.append(r["M"])  # superseded by direct tile measurement
            continue
        kept_ms.append(r)
    merged_rows = kept_ms + tb_rows

    # ---- recompute the entire cost model from the merged rows ----
    cost_model = build_cost_model(merged_rows, accept_models)

    # ---- continuity report at the shared M (provenance) ----
    base = {int(k): float(v) for k, v in
            msd["cost_model"]["graph|ctx256"]["latency_ms_by_M"].items()}
    new = {int(k): float(v) for k, v in
           cost_model["graph|ctx256"]["latency_ms_by_M"].items()}
    shared = sorted(set(base) & tb_M)
    print("[fold] graph|ctx256 continuity at tile/msweep shared M:")
    for m in shared:
        print(f"[fold]   M={m:3d}  prefold={base[m]:7.3f}  folded={new[m]:7.3f}  "
              f"Δ={new[m] - base[m]:+.3f} ms")
    print(f"[fold] folded M points ({len(new)}): {sorted(new)}")
    print(f"[fold] msweep graph|ctx256 M superseded by tile measurement: "
          f"{sorted(dropped)}")
    # large-K cliff fix headline
    for m in (33, 49):
        print(f"[fold]   M={m}: prefold(interp)≈"
              f"{_interp(base, m):7.3f}  folded(measured)={new[m]:7.3f}  "
              f"(+{new[m] - _interp(base, m):.3f} ms recovered)")

    # ---- write pre-fold provenance copy, then fold in place ----
    shutil.copy2(args.msweep, args.prefold_out)
    print(f"[fold] pre-fold copy -> {args.prefold_out}")

    msd["rows"] = merged_rows
    msd["cost_model"] = cost_model
    msd["config"]["tile_boundary_folded"] = {
        "source": args.tile, "key": "graph|ctx256",
        "superseded_msweep_M": sorted(dropped),
        "added_tile_M": sorted(tb_M - set(base)),
        "note": "graph|ctx256 carries the PR#33 directly-measured Marlin tile "
                "cliffs (M=17/33/49); eager|* and *|ctx512 keys are NOT "
                "tile-corrected (tile sweep measured graph|ctx256 only).",
        "prefold_copy": args.prefold_out,
    }
    json.dump(msd, open(args.msweep, "w"), indent=2)
    print(f"[fold] wrote folded canonical curve in place -> {args.msweep}")


def _interp(tab, M):
    xs = sorted(tab)
    if M <= xs[0]:
        return tab[xs[0]]
    if M >= xs[-1]:
        return tab[xs[-1]]
    lo = max(x for x in xs if x <= M)
    hi = min(x for x in xs if x >= M)
    if lo == hi:
        return tab[lo]
    t = (M - lo) / (hi - lo)
    return tab[lo] * (1 - t) + tab[hi] * t


if __name__ == "__main__":
    main()
