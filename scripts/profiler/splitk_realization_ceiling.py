#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""SplitK REALIZATION-CEILING roofline (PR #117): can SplitK PHYSICALLY reach
denken #109's 14.34% conservative corner, or is the 540-margin tau/tree-gated?

THE GAP THIS PRICES
-------------------
denken #109 set the ship corner at SplitK >= 14.34% and bolted the SplitK
*gap ceiling* at SPLITK_CEILING = 1/HBM_util - 1 = 1/0.771 - 1 = 29.7% -- the s
at which the verify-GEMM achieved bandwidth closes all the way to **100% of the
A10G 600 GB/s DATASHEET peak**. ubel #108's BUILD sizes SplitK at central 8.5%
(CI 5-12%). NOBODY has asked whether the +29.8% achieved-vs-roofline gap denken
#68 MEASURED is PHYSICALLY convertible to wall-time, or whether the practical
HBM wall / occupancy distribution caps SplitK far below the corner. This module
roofline the realizable ceiling and hands ubel the number to stop tuning at.

UNITS (load-bearing -- pinned against the #105/#109 model)
---------------------------------------------------------
SplitK `s` is a BANDWIDTH-HEADROOM fraction: vg -> vg/(1+s) (#105 compose). s is
the fractional INCREASE in achieved aggregate verify-GEMM bandwidth. #109's
corner 14.34%, ubel's 8.5%/12%, and the gap ceiling 29.7% are ALL in these s
units. So is our `splitk_realization_ceiling_pct`. (Check: s=29.7% <-> BW 462 ->
600 GB/s <-> 22.9% verify-GEMM wall-time reduction = the PR's "1 - 0.771".)

THE PHYSICS (researcher pass + #68 MEASURED data; arxiv:2402.00025 is the
directly-analogous W4A16-SplitK reference)
----------------------------------------------------------------------------
SplitK partitions the K reduction across more CTAs. On a BW-bound GEMM its ONLY
lever is OCCUPANCY: at M=8 a small-N GEMM emits few output tiles (N/tile_N CTAs,
1 M-tile) and starves the 80 SMs; SplitK x g fills them, raising memory-level
parallelism and achieved BW. THREE real ceilings bound the gain:

  1. HBM PRACTICAL roofline. The datasheet 600 GB/s is NOT reachable: GDDR6
     sustained tops out ~80-88% (refresh/ECC/page-conflict; GPU-STREAM ~80%).
     #68 MEASURED the wall in-situ: gate_up (N=20480) is already CTA-saturated
     (160 CTAs >> 80 SMs) yet tops out at 79.2% datasheet = 475 GB/s. That 79.2%
     is the DRAM-efficiency wall SplitK drives toward -- full occupancy = exactly
     gate_up's state. The aggregate is already at 77.1%. So the realizable BW
     headroom is ~79-vs-77 = a few pp, NOT the 23pp to datasheet.
  2. The dominant GEMM is ALREADY at the wall. gate_up is 54% of verify time and
     CTA-saturated -> SplitK gives it ~0 BW (only overhead). Only the
     occupancy-limited laggards (down + attention, small N) have headroom, and
     only up to the same ~79% wall.
  3. SplitK reduction OVERHEAD. g partial [M,N] sums must be reduced (extra
     writes+read, separate launch). Byte-tiny at M=8 (out << weight) but
     launch/L2-dominated -> trims the small gross gain; our net s is an UPPER
     bound (we model only the byte overhead).

COMPUTE FLOOR never binds: AI=28 is 3.8x below the ridge (107); even at 100% BW
the compute util only reaches ~26% (=20.2%/0.771). So the binding constraint is
the HBM PRACTICAL roofline, NOT the compute floor and NOT the datasheet 29.7%.

OUTPUTS
  PRIMARY  splitk_realization_ceiling_pct  -- max realizable s (BW-headroom),
           band over the wall assumption, with the binding constraint NAMED.
  TEST     splitk_headroom_to_corner = ceiling - 14.34%  (>=0 => corner reachable
           on SplitK alone).
  VERDICT  GREEN / AMBER / RED + fleet hand-off (ubel #108 target %, lawine #116
           tau co-requirement, denken #109 ship gate).

LOCAL, CPU-ONLY, ANALYTIC. Loads denken #68 verify_gemm_roofline.json (MEASURED)
+ reuses denken #105/#109 compose model + lawine #99 multiplier. No GPU, no vLLM,
no HF Job, no submission, no kernel build (ubel #108 owns the BUILD; we roofline
the CEILING that bounds it). Greedy identity untouched by construction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- reuse denken #105 compose model + #109 ship corner (lawine #99 multiplier) ----
tf = _load("tree_free_500_ceiling", os.path.join(_HERE, "tree_free_500_ceiling.py"))
sr = _load("tree_free_ship_readiness", os.path.join(_HERE, "tree_free_ship_readiness.py"))

# ---- physical constants (A10G; #68 config block) ----
A10G_HBM_GBS = 600.0          # datasheet peak (#68 config)
SM_COUNT = 80                 # A10G (GA102) streaming multiprocessors
OPERATING_M = 8               # deployed verify width (K+1, num_speculative_tokens=7)

# ---- wall scenarios: the achievable sustained HBM fraction of datasheet ----
# measured  = gate_up's CTA-saturated 79.2% (the in-situ DRAM-efficiency wall;
#             SplitK's terminal state = full occupancy = exactly this).
# practical = 88% datasheet, the optimistic GDDR6 sustained ceiling (literature),
#             granted to the occupancy-limited laggards only (gate_up frozen).
# datasheet = 100% = #109's SPLITK_CEILING assumption (PHYSICALLY UNREACHABLE,
#             shown only to mark where the 29.7% gap ceiling comes from).
WALL_SCENARIOS = {"measured": None, "practical": 0.88, "datasheet": 1.00}

# corner ladder (computed live from the #109 ship model; tau drives it):
CORNER_TAUS = [0.96, 0.97, 0.98, 0.99, 1.00]
UBEL_BAND = {"low": 0.05, "central": 0.085, "high": 0.12}   # ubel #108 sizing


def _corner_splitk(tau: float) -> float:
    """min SplitK s (conservative corner) to clear 500 at a given tau."""
    p = dict(sr.ship_point("conservative", "local"))
    p["tau"] = tau
    return sr.splitk_threshold_for(p, sr.TARGET)


def load_m8_gemms(roofline_json: str):
    """Per-(role,shape) MEASURED M=8 GEMMs from denken #68's roofline."""
    with open(roofline_json) as f:
        rf = json.load(f)
    gemms = []
    for r in rf["rows"]:
        if r["M"] != OPERATING_M:
            continue
        n_out = r["out"]
        ctas = math.ceil(n_out / TILE_N)            # M-tile=1 at M=8 (<16)
        gemms.append({
            "role": r["role"], "in": r["in"], "out": n_out, "count": r["count"],
            "t_us": r["t_us"], "gbytes_s": r["gbytes_s"],
            "pct_hbm": r["pct_hbm_peak"], "w_bytes": r["w_bytes"],
            "out_bytes": r["out_bytes"], "total_bytes": r["total_bytes"],
            "n_tiles_ctas": ctas,
            "cta_saturated": ctas >= SM_COUNT,      # already fills the GPU -> no SplitK BW
        })
    agg = rf["aggregate_by_M"][str(OPERATING_M)]
    return gemms, agg, rf["config"]


def splitk_factor(ctas: int) -> int:
    """K-split g to reach SM saturation (capped). g=1 if already saturated."""
    if ctas >= SM_COUNT:
        return 1
    return min(MAX_SPLIT, max(1, math.ceil(SM_COUNT / ctas)))


def ceiling_for_wall(gemms, wall_frac_practical):
    """Recompute aggregate verify-GEMM time when SplitK lifts every
    occupancy-limited GEMM to the DRAM-efficiency wall, freezing the
    CTA-saturated gate_up and the L2-artifact rows. Returns gross+net s."""
    # the measured wall = the best CTA-saturated HBM-real util (gate_up).
    sat = [g for g in gemms if g["cta_saturated"]]
    measured_wall_pct = max((g["pct_hbm"] for g in sat), default=79.0)
    wall_pct = measured_wall_pct if wall_frac_practical is None else wall_frac_practical * 100.0

    t_old = sum(g["t_us"] * g["count"] for g in gemms)
    t_new_gross = 0.0
    t_new_net = 0.0
    per = []
    for g in gemms:
        t0 = g["t_us"]
        liftable = (not g["cta_saturated"]) and (g["pct_hbm"] < wall_pct)
        if liftable:
            # BW lifts current -> wall: time scales by (cur_pct / wall_pct).
            t_lift = t0 * (g["pct_hbm"] / wall_pct)
            g_split = splitk_factor(g["n_tiles_ctas"])
            # reduction overhead: (2g-1) extra [M,N] partial-sum bytes vs 1 write,
            # streamed at the wall BW. Byte-tiny (out << weight); launch/L2 extra
            # is UNMODELLED -> net s is an upper bound.
            ovh_bytes = (2 * g_split - 1) * g["out_bytes"]
            ovh_frac = ovh_bytes / g["total_bytes"]
            t_net = t_lift * (1.0 + ovh_frac)
        else:
            t_lift = t0          # frozen (saturated gate_up, or L2-artifact >wall)
            t_net = t0
            g_split = 1
        t_new_gross += t_lift * g["count"]
        t_new_net += t_net * g["count"]
        per.append({**{k: g[k] for k in ("role", "in", "out", "count", "pct_hbm",
                                         "n_tiles_ctas", "cta_saturated")},
                    "liftable": liftable, "split_factor": g_split,
                    "t_us_old": t0, "t_us_lifted": t_lift, "t_us_net": t_net})
    s_gross = t_old / t_new_gross - 1.0
    s_net = t_old / t_new_net - 1.0
    # aggregate util after the lift (net):
    util_old = sum(g["gbytes_s"] * g["t_us"] * g["count"] for g in gemms) / t_old / A10G_HBM_GBS
    util_new = util_old * (t_old / t_new_net)
    return {
        "wall_pct_datasheet": wall_pct,
        "measured_wall_pct": measured_wall_pct,
        "t_us_old": t_old, "t_us_new_gross": t_new_gross, "t_us_new_net": t_new_net,
        "s_gross": s_gross, "s_net": s_net,
        "agg_util_old_pct": util_old * 100.0, "agg_util_new_pct": util_new * 100.0,
        "compute_util_old_pct": None,  # filled by caller
        "per_gemm": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roofline", default="research/spec_cost_model/verify_gemm_roofline.json")
    ap.add_argument("--out", default="research/spec_cost_model/splitk_realization_ceiling_results.json")
    ap.add_argument("--tile-n", type=int, default=128, help="Marlin N-tile (CTA count = N/tile_n)")
    ap.add_argument("--max-split", type=int, default=8, help="cap on the K-split factor")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="denken/splitk-realization-ceiling")
    ap.add_argument("--wandb-group", default="splitk-realization-ceiling")
    args = ap.parse_args()

    global TILE_N, MAX_SPLIT
    TILE_N, MAX_SPLIT = args.tile_n, args.max_split

    gemms, agg, cfg = load_m8_gemms(args.roofline)

    # compute-floor check: at M=8 the verify-GEMM uses agg_pct_compute_peak of the
    # fp16 tensor peak; saturating BW raises it inversely with the wall-time cut.
    compute_util_m8 = agg["agg_pct_compute_peak"]   # 20.15%
    hbm_util_m8 = agg["agg_pct_hbm_peak"]           # 77.06%

    # ---- the three wall scenarios ----
    scen = {}
    for name, wf in WALL_SCENARIOS.items():
        r = ceiling_for_wall(gemms, wf)
        # compute util at this scenario's BW = current * (t_old/t_new)
        r["compute_util_old_pct"] = compute_util_m8
        r["compute_util_new_pct"] = compute_util_m8 * (r["t_us_old"] / r["t_us_new_net"])
        r["compute_floor_binds"] = r["compute_util_new_pct"] >= 100.0
        scen[name] = r

    # primary ceiling = the MEASURED-wall GROSS s: SplitK perfectly recovers
    # occupancy, lifting the occupancy-limited laggards to gate_up's MEASURED
    # 79.2% DRAM-efficiency wall (gate_up frozen, CTA-saturated). Overhead-free
    # upper bound under the empirical wall -> the headline doesn't hinge on the
    # overhead model. Band high = the PRACTICAL-88% GROSS (laggards granted the
    # optimistic GDDR6 sustained ceiling, gate_up still frozen).
    s_primary = scen["measured"]["s_gross"]
    s_primary_net = scen["measured"]["s_net"]     # after byte-overhead (launch/L2 unmodelled)
    s_band_high = scen["practical"]["s_gross"]
    s_datasheet = scen["datasheet"]["s_gross"]    # laggards->100%, gate_up STILL frozen

    ceiling_pct = s_primary * 100.0
    ceiling_net_pct = s_primary_net * 100.0
    band_high_pct = s_band_high * 100.0

    # ---- corner ladder (live from #109 ship model) ----
    corner = {f"tau_{t:.2f}": _corner_splitk(t) * 100.0 for t in CORNER_TAUS}
    corner_096 = corner["tau_0.96"]      # 14.34% -- the PR primary corner
    corner_099 = corner["tau_0.99"]      # 7.57%  -- lawine #116 mechanism floor (AMBER floor)
    corner_100 = corner["tau_1.00"]      # 5.49%  -- corner at PERFECT tau
    gap_ceiling_pct = tf.SPLITK_CEILING * 100.0   # 29.7%

    headroom_to_corner = ceiling_pct - corner_096          # TEST metric (<0 => unreachable)
    headroom_to_corner_high = band_high_pct - corner_096

    # ---- TPS cross-check via #105 compose (corner levers, tau band) + field ----
    def corner_tps(s_frac, tau):
        p = dict(sr.ship_point("conservative", "local")); p["tau"] = tau
        return sr.official_ship(s_frac, p)
    tps_at_ceiling = {f"tau_{t:.2f}": corner_tps(s_primary, t) for t in (0.96, 0.99, 1.0)}
    tps_at_band_high = {f"tau_{t:.2f}": corner_tps(s_band_high, t) for t in (0.96, 0.99, 1.0)}
    # field realized SplitK gains ~+0.6-1.7% TPS over 481.53 -> implied s:
    # TPS uplift u -> verify_reduction = (1-1/(1+u))/0.53 -> s = vr/(1-vr).
    def tps_uplift_to_s(u):
        step_red = 1.0 - 1.0 / (1.0 + u)
        vr = step_red / tf.BUDGET["verify_gemm"]
        return vr / (1.0 - vr)
    field_s_low = tps_uplift_to_s(0.006) * 100.0
    field_s_high = tps_uplift_to_s(0.017) * 100.0

    # ---- binding constraint ----
    # compute floor binds? (never, at M=8). practical-wall binds (we are ~at it).
    if scen["measured"]["compute_floor_binds"]:
        binding = "compute-floor"
    else:
        binding = "HBM-practical-roofline"
    binding_detail = (
        f"HBM PRACTICAL roofline: the verify-GEMM is already at {hbm_util_m8:.1f}% of "
        f"datasheet and the in-situ DRAM-efficiency wall (gate_up, CTA-saturated, "
        f"measured) is {scen['measured']['measured_wall_pct']:.1f}% -> only "
        f"{ceiling_pct:.1f}% BW-headroom realizable. The dominant gate_up GEMM "
        f"(54% of verify time) is already AT the wall and CTA-saturated -> SplitK "
        f"adds ~0 there. Compute floor does NOT bind (compute util rises only "
        f"{compute_util_m8:.1f}% -> {scen['datasheet']['compute_util_new_pct']:.1f}% even at "
        f"100% BW). The 29.7% gap ceiling assumes UNREACHABLE 100%-datasheet BW.")

    # ---- verdict ----
    # GREEN: ceiling >= corner_096 with margin AND ubel-high (12%) below it.
    # AMBER: ceiling in [corner_099 (7.57%), corner_096 (14.34%)).
    # RED:   ceiling < ubel realistic central 8.5%  OR  compute floor binds.
    ubel_central = UBEL_BAND["central"] * 100.0
    ubel_high = UBEL_BAND["high"] * 100.0
    if scen["measured"]["compute_floor_binds"]:
        verdict = "RED"
    elif ceiling_pct >= corner_096 and ubel_high < ceiling_pct:
        verdict = "GREEN"
    elif corner_099 <= ceiling_pct < corner_096:
        verdict = "AMBER"
    elif ceiling_pct < ubel_central:
        verdict = "RED"
    else:
        verdict = "AMBER"   # in [ubel_central, corner_099)

    if verdict == "RED":
        verdict_label = (
            f"540-margin (and the 500 corner) is genuinely tau/tree-gated. The "
            f"physical SplitK realization ceiling is ~{ceiling_pct:.1f}% (band to "
            f"~{band_high_pct:.1f}%), BELOW ubel #108's realistic central {ubel_central:.1f}% "
            f"and far below #109's {corner_096:.1f}% corner -- even below the {corner_100:.1f}% "
            f"corner at PERFECT tau=1.0. SplitK CANNOT close the corner alone. Tell ubel "
            f"to stop tuning SplitK past ~{band_high_pct:.0f}%; confirm the lawine #116 "
            f"tau-anchor / land #71 tree as the only paths to the corner.")
    elif verdict == "AMBER":
        verdict_label = (
            f"SplitK + tau are CO-REQUIRED. The ceiling ~{ceiling_pct:.1f}% (band "
            f"~{band_high_pct:.1f}%) clears the corner only down to the mechanism-floor "
            f"tau=0.99 ({corner_099:.1f}%); at the tau=0.96 floor the corner ({corner_096:.1f}%) "
            f"is unreachable. Bank the ceiling; lawine #116's tau verdict decides the ship.")
    else:
        verdict_label = (
            f"SplitK alone reaches the corner: ceiling {ceiling_pct:.1f}% >= {corner_096:.1f}% "
            f"and ubel's {ubel_high:.1f}% sits below it -> push SplitK toward the ceiling.")

    gate = {
        "primary_metric_name": "splitk_realization_ceiling_pct",
        "splitk_realization_ceiling_pct": ceiling_pct,
        "splitk_realization_ceiling_net_pct": ceiling_net_pct,
        "splitk_realization_ceiling_band_high_pct": band_high_pct,
        "binding_constraint": binding,
        "binding_constraint_detail": binding_detail,
        "test_metric_name": "splitk_headroom_to_corner",
        "splitk_headroom_to_corner": headroom_to_corner,
        "splitk_headroom_to_corner_band_high": headroom_to_corner_high,
        "corner_reachable_on_splitk_alone": headroom_to_corner >= 0.0,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "ubel_target_pct": round(band_high_pct, 1),   # stop-tuning target for ubel #108
    }

    out = {
        "gate": gate,
        "units_note": ("SplitK s = fractional INCREASE in achieved aggregate verify-GEMM "
                       "bandwidth (vg->vg/(1+s)); ALL of corner 14.34%, ubel 8.5/12%, gap "
                       "ceiling 29.7% and this ceiling are in these s units. s=29.7% <-> "
                       "BW 462->600 GB/s <-> 22.9% verify-GEMM wall-time cut."),
        "wall_scenarios": scen,
        "ceiling_summary": {
            "measured_wall_net_pct": ceiling_pct,
            "practical88_wall_net_pct": band_high_pct,
            "datasheet100_net_pct": s_datasheet * 100.0,
            "gap_ceiling_109_pct": gap_ceiling_pct,
            "note": ("primary = measured-wall (SplitK lifts occupancy-limited laggards to "
                     "gate_up's MEASURED 79.2% DRAM wall, gate_up frozen). band-high = laggards "
                     "granted 88% GDDR6 practical, gate_up still frozen. datasheet100 ~ #109's "
                     "29.7% gap ceiling, shown UNREACHABLE."),
        },
        "corner_ladder": {
            "definition": "min SplitK s (conservative corner, #109 ship model) to clear 500 at each tau",
            "by_tau_pct": corner,
            "corner_tau096_primary_pct": corner_096,
            "corner_tau099_mechanism_floor_pct": corner_099,
            "corner_tau100_perfect_pct": corner_100,
            "gap_ceiling_pct": gap_ceiling_pct,
            "ubel_band_pct": {k: v * 100.0 for k, v in UBEL_BAND.items()},
        },
        "compute_floor_check": {
            "compute_util_m8_pct": compute_util_m8,
            "hbm_util_m8_pct": hbm_util_m8,
            "compute_util_at_100pct_bw_pct": scen["datasheet"]["compute_util_new_pct"],
            "binds_before_corner": scen["datasheet"]["compute_floor_binds"],
            "note": ("AI=28 is 3.8x below the ridge (107); even at 100% BW the compute util "
                     "reaches only ~26% -> compute floor NEVER binds; the cap is HBM."),
        },
        "tps_crosscheck": {
            "corner_tps_at_ceiling": tps_at_ceiling,
            "corner_tps_at_band_high": tps_at_band_high,
            "field_realized_tps_uplift_pct": [0.6, 1.7],
            "field_implied_s_pct": [field_s_low, field_s_high],
            "note": (f"public SplitK-class submissions realize ~+0.6-1.7% TPS over 481.53 -> "
                     f"implied s ~{field_s_low:.1f}-{field_s_high:.1f}% -- straddles our "
                     f"measured-wall ceiling, an independent corroboration that SplitK is "
                     f"already near its physical wall."),
        },
        "config": {
            "roofline_source": args.roofline, "tile_n": TILE_N, "max_split": MAX_SPLIT,
            "sm_count": SM_COUNT, "a10g_hbm_gbs": A10G_HBM_GBS, "operating_M": OPERATING_M,
            "model_device": cfg.get("device"), "frontier_tps": cfg.get("frontier_tps"),
        },
        "provenance": ("extends denken #68 verify_gemm_roofline.json (MEASURED Marlin W4A16) + "
                       "denken #105/#109 compose ship model + lawine #99 multiplier; "
                       "researcher pass arxiv:2402.00025 (W4A16 SplitK), arxiv:2301.03598 "
                       "(Stream-K CTA quantization), GPU-STREAM (~80% practical BW)."),
        "method": ("LOCAL CPU-only analytic roofline; no GPU/vLLM/HF Job/submission/kernel "
                   "build. Greedy identity untouched (SplitK 0-flip kanna #87). ubel #108 owns "
                   "the BUILD; this bounds the CEILING."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: (None if o == float("inf") else o))

    # ------------------------------- console -------------------------------
    print("=" * 92)
    print("SPLITK REALIZATION-CEILING ROOFLINE (PR #117) -- can SplitK reach the 14.34% corner?")
    print("=" * 92)
    print(f"\nverify-GEMM M=8 (MEASURED #68): {hbm_util_m8:.1f}% HBM / {compute_util_m8:.1f}% compute "
          f"| datasheet {A10G_HBM_GBS:.0f} GB/s | SMs {SM_COUNT} | tile_n {TILE_N}")
    print(f"\n[per-GEMM occupancy at M=8]  (CTA = out/{TILE_N}; saturated >= {SM_COUNT} SMs)")
    print(f"  {'role':28s} {'out':>6s} {'CTAs':>5s} {'sat?':>5s} {'%HBM':>6s} {'split':>6s} {'time%':>6s}")
    t_old = scen["measured"]["t_us_old"]
    for g, p in zip(gemms, scen["measured"]["per_gemm"]):
        tshare = g["t_us"] * g["count"] / t_old * 100.0
        print(f"  {g['role'][:28]:28s} {g['out']:6d} {p['n_tiles_ctas']:5d} "
              f"{('YES' if p['cta_saturated'] else 'no'):>5s} {g['pct_hbm']:5.1f}% "
              f"{p['split_factor']:5d}x {tshare:5.1f}%")
    print(f"\n[ceiling by wall scenario]  (s = realizable BW-headroom)")
    print(f"  {'scenario':12s} {'wall%ds':>8s} {'agg util':>14s} {'s_gross':>8s} {'s_net':>8s} {'compute@BW':>11s}")
    for name in ("measured", "practical", "datasheet"):
        r = scen[name]
        print(f"  {name:12s} {r['wall_pct_datasheet']:7.1f}% "
              f"{r['agg_util_old_pct']:5.1f}->{r['agg_util_new_pct']:5.1f}% "
              f"{r['s_gross']*100:7.2f}% {r['s_net']*100:7.2f}% {r['compute_util_new_pct']:9.1f}%")
    print(f"\n[corner ladder] (#109 ship model: min SplitK to clear 500 at each tau)")
    for t in CORNER_TAUS:
        tag = ("  <- PRIMARY corner (tau floor)" if t == 0.96 else
               "  <- lawine #116 mechanism floor (AMBER floor)" if t == 0.99 else
               "  <- corner at PERFECT tau" if t == 1.0 else "")
        print(f"     tau={t:.2f} -> corner {corner[f'tau_{t:.2f}']:6.2f}%{tag}")
    print(f"     gap ceiling (#109, BW->100% datasheet, UNREACHABLE) = {gap_ceiling_pct:.1f}%")
    print(f"     ubel #108 band: {UBEL_BAND['low']*100:.0f}/{UBEL_BAND['central']*100:.1f}/{UBEL_BAND['high']*100:.0f}%")
    print(f"\n[PRIMARY] splitk_realization_ceiling_pct = {ceiling_pct:.2f}%  "
          f"(net-after-overhead {ceiling_net_pct:.2f}%; band to {band_high_pct:.2f}% at 88%-GDDR6 wall)")
    print(f"[TEST]    splitk_headroom_to_corner = {headroom_to_corner:+.2f}%  "
          f"(band {headroom_to_corner_high:+.2f}%)  -> corner "
          f"{'REACHABLE' if headroom_to_corner >= 0 else 'UNREACHABLE'} on SplitK alone")
    print(f"[binding] {binding}")
    print(f"\n[TPS cross-check] field SplitK-class realized +0.6-1.7% TPS -> implied s "
          f"{field_s_low:.1f}-{field_s_high:.1f}% (straddles our {ceiling_pct:.1f}% ceiling)")
    print(f"  corner TPS at ceiling s={ceiling_pct:.1f}%: tau0.96 {tps_at_ceiling['tau_0.96']:.1f} | "
          f"tau0.99 {tps_at_ceiling['tau_0.99']:.1f} | tau1.0 {tps_at_ceiling['tau_1.00']:.1f}  (target 500)")
    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "splitk-realization-ceiling",
                                 "method": "cpu-analytic-roofline-extends-68-109",
                                 "tile_n": TILE_N, "max_split": MAX_SPLIT, "sm_count": SM_COUNT,
                                 "a10g_hbm_gbs": A10G_HBM_GBS, "operating_M": OPERATING_M,
                                 "hbm_util_m8_pct": hbm_util_m8, "compute_util_m8_pct": compute_util_m8})
        s = wandb.summary
        s["splitk_realization_ceiling_pct"] = ceiling_pct
        s["splitk_realization_ceiling_net_pct"] = ceiling_net_pct
        s["splitk_realization_ceiling_band_high_pct"] = band_high_pct
        s["splitk_headroom_to_corner"] = headroom_to_corner
        s["splitk_headroom_to_corner_band_high"] = headroom_to_corner_high
        s["corner_reachable_on_splitk_alone"] = bool(headroom_to_corner >= 0)
        s["binding_constraint"] = binding
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label
        s["corner_tau096_pct"] = corner_096
        s["corner_tau099_pct"] = corner_099
        s["corner_tau100_pct"] = corner_100
        s["gap_ceiling_109_pct"] = gap_ceiling_pct
        s["ubel_central_pct"] = ubel_central
        s["ubel_high_pct"] = ubel_high
        s["s_measured_wall_pct"] = scen["measured"]["s_net"] * 100.0
        s["s_practical88_pct"] = scen["practical"]["s_net"] * 100.0
        s["s_datasheet100_pct"] = scen["datasheet"]["s_net"] * 100.0
        s["compute_util_at_100pct_bw_pct"] = scen["datasheet"]["compute_util_new_pct"]
        s["compute_floor_binds"] = bool(scen["measured"]["compute_floor_binds"])
        s["field_implied_s_low_pct"] = field_s_low
        s["field_implied_s_high_pct"] = field_s_high
        s["ubel_target_pct"] = round(band_high_pct, 1)

        wt = wandb.Table(columns=["scenario", "wall_pct_datasheet", "agg_util_new_pct",
                                  "s_gross_pct", "s_net_pct", "compute_util_at_bw_pct"])
        for name in ("measured", "practical", "datasheet"):
            r = scen[name]
            wt.add_data(name, r["wall_pct_datasheet"], r["agg_util_new_pct"],
                        r["s_gross"] * 100.0, r["s_net"] * 100.0, r["compute_util_new_pct"])
        wandb.log({"wall_scenarios": wt})

        ct = wandb.Table(columns=["tau", "corner_splitk_pct"])
        for t in CORNER_TAUS:
            ct.add_data(t, corner[f"tau_{t:.2f}"])
        wandb.log({"corner_ladder": ct})

        gt = wandb.Table(columns=["role", "out", "ctas", "cta_saturated", "pct_hbm",
                                  "split_factor", "liftable"])
        for g, p in zip(gemms, scen["measured"]["per_gemm"]):
            gt.add_data(g["role"], g["out"], p["n_tiles_ctas"], bool(p["cta_saturated"]),
                        g["pct_hbm"], p["split_factor"], bool(p["liftable"]))
        wandb.log({"per_gemm_occupancy": gt})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
