#!/usr/bin/env python
"""PR #728 — augment the sweep report with the base-anchored official-equiv bracket.

The harness logs official_equiv = wall_tps * TAU_LO(1.0352), the banked local->official
scalar from the fa2sw LINEAR path (project_local_official_tps_transfer). That scalar is
CONSERVATIVE here: it carries the BI=1 determinism tax (the AR base reads 106.02 local
under BI=1 vs the known 126.378 official / 127.08 no-BI) into the projection, i.e. it
projects "official IF shipped with BI=1 still on" — a config you would never ship.

The base-anchored ratio calibrates to the SAME checkpoint's known official anchor:
    ratio_anchored = 126.378 / ar_base_local_BI1
and projects each spec config by its measured speedup over the AR base (the speedup is a
same-hardware ratio, so the hardware/clock factor cancels; assumes the spec speedup is
~BI-invariant since BI taxes the AR and the verify matmuls proportionally).

Bracket per spec config:  [ wall_tps*1.0352 (floor, ship-with-BI),  wall_tps*ratio_anchored (estimate, ship-no-BI) ].
Both are PROJECTIONS — analysis_only=1, official_tps=0, no HF Job fired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "runs" / "sweep" / "report.json"
ANCHOR_TPS = 126.378
TAU_LO = 1.0352


def main() -> int:
    r = json.loads(REPORT.read_text())
    ar_local = r["ar_reference"]["wall_tps_local"]
    ratio_anchored = ANCHOR_TPS / ar_local
    r["transfer_model"] = {
        "ar_base_local_bi1": ar_local,
        "anchor_official": ANCHOR_TPS,
        "tau_lo_flat": TAU_LO,
        "ratio_anchored": ratio_anchored,
        "bi_tax_vs_noBI_127_08": ar_local / 127.083,
        "note": ("official_equiv_flat = wall_tps*1.0352 is the CONSERVATIVE floor (carries the "
                 "BI=1 tax). official_equiv_anchored = wall_tps*(126.378/ar_base_local) is the "
                 "base-anchored estimate (ship without BI). Both are projections; no HF job fired."),
    }
    for x in r["results"]:
        wt = x.get("wall_tps_local")
        if isinstance(wt, (int, float)):
            x["official_equiv_flat_tau_lo"] = wt * TAU_LO          # == existing official_equiv_tps
            x["official_equiv_anchored"] = wt * ratio_anchored
            x["official_equiv_bracket"] = [round(wt * TAU_LO, 2), round(wt * ratio_anchored, 2)]
    for key in ("fastest_self_consistent", "fastest_any"):
        x = r.get(key)
        if x and isinstance(x.get("wall_tps_local"), (int, float)):
            wt = x["wall_tps_local"]
            x["official_equiv_flat_tau_lo"] = wt * TAU_LO
            x["official_equiv_anchored"] = wt * ratio_anchored
            x["official_equiv_bracket"] = [round(wt * TAU_LO, 2), round(wt * ratio_anchored, 2)]
    REPORT.write_text(json.dumps(r, indent=2, default=str))
    print(f"[aug] ratio_anchored = {ANCHOR_TPS}/{ar_local:.4f} = {ratio_anchored:.5f}")
    for x in r["results"]:
        print(f"[aug] K={x['k']}: wall={x['wall_tps_local']:.2f} -> bracket "
              f"[{x['official_equiv_flat_tau_lo']:.2f} floor, {x['official_equiv_anchored']:.2f} anchored] "
              f"self_consistent={x['self_consistent_tau03']}")

    # ---- resume the wandb run to log the base-anchored fields (best-effort) ----
    try:
        import os
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb
        run = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                         id="wju9brji", resume="allow")
        sm = {"transfer/ratio_anchored": ratio_anchored,
              "transfer/ar_base_local_bi1": ar_local,
              "transfer/bi_tax_vs_noBI": ar_local / 127.083}
        for x in r["results"]:
            if isinstance(x.get("wall_tps_local"), (int, float)):
                sm[f"k{x['k']}/official_equiv_anchored"] = x["official_equiv_anchored"]
        fsc = r.get("fastest_self_consistent")
        if fsc:
            sm["fastest_self_consistent_official_equiv_anchored"] = fsc["official_equiv_anchored"]
        run.summary.update(sm)
        wandb.finish()
        print("[aug] wandb run wju9brji updated with base-anchored projections")
    except Exception as exc:  # noqa: BLE001
        print(f"[aug] wandb resume skipped ({type(exc).__name__}: {exc}); report.json still augmented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
