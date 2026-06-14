#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-free 500-path ceiling gate (PR #105): the maximum official-TPS reachable
from the BUILD-COMPLETE lever stack with NO tree, and the SplitK threshold at
which that tree-free stack clears 500.

WHY THIS GATE EXISTS
--------------------
fern #100 proved the *tree* alone clears 500 (cons 518 / cent 563), but that
result assumes E[T]=5.207 and the tree is BUILD-BLOCKED / re-measure-pending:
the as-built tree gives tok/step=2.10 (denken #101), and the star-attention
verify op is not CUDA-graph-safe under FULL capture (public board, chiku-inu
2026-06-14: tok/step collapses to 1.098 + illegal memory under graphs; only
2.097 under enforce-eager). We do not control when the build team unblocks it.

This gate prices the COMPLEMENT of fern #102 (which asks "how good must the tree
be?"). It asks: do we even NEED the tree? i.e. what is the maximum official TPS
from the levers that are build-COMPLETE today -- SplitK #84 + LK #95 +
double-quant #104, with NO tree (E[T] stays at the linear 3.844) -- and at what
SplitK% does that tree-free stack clear 500? The verdict converts "we hope the
tree lands" into "here is whether 500 is reachable even if it never does, and
exactly what SplitK must deliver."

THE MODEL (fern #100 absolute-time slice composition, tree-EXCLUSIVE branch)
---------------------------------------------------------------------------
official_TPS = K_cal * (E[T] / step_time) * tau,   K_cal = 481.53 / 3.844

The deployed M=8 linear-MTP step = 1.0, decomposed into ABSOLUTE slices
(CURRENT_RESEARCH_STATE.md): verify-GEMM 0.53 (int4 W4A16 Marlin, BW-bound,
77.1% HBM util at M=8 -> a 22.9% bandwidth gap = the SplitK headroom ceiling),
drafter 0.07, attention 0.08, other 0.32 (of which only 2.17% is reclaimable
GPU-idle, denken #97; the rest is GPU-busy small-kernel tail).

Tree-free lever placement:
  * LK #95            E[T] NUMERATOR -- multiplies E[T] by lk_mult (+1.0..2.4%,
                      central near floor 1.010 per #80 single-layer capacity;
                      PREDICTION channel, needs a head probe to REALIZE, so it
                      is projected-not-banked).
  * SplitK #84        verify-GEMM DENOMINATOR via BANDWIDTH UTILISATION: closes
                      the 77.1%->100% HBM gap. vg -> vg / (1 + s). It reads the
                      SAME bytes faster. The SWEPT parameter (ubel #84 sized
                      +5..12%; hard ceiling = full gap close = +29.7%).
  * double-quant #104 verify-GEMM DENOMINATOR via BYTE COUNT: INT8 scale-of-
                      scales (+ FP16 sparse exceptions) removes scale bytes.
                      vg -> vg * (1 - f_dq). It reads FEWER bytes.

HONEST NETTING of SplitK x double-quant (the PR's "don't double-count the same
bytes"): they act on ORTHOGONAL factors of the same slice -- SplitK on achieved
bandwidth (utilisation), double-quant on the byte count -- because verify-GEMM
time = bytes / bandwidth. SplitK removes NO bytes; double-quant changes NO
utilisation. So they COMPOUND multiplicatively with no double-count:
      vg = 0.53 * (1 - f_dq) / (1 + s).
(A double-count would only arise if BOTH reduced bytes; SplitK does not. The
roofline #68 classifies SplitK as a utilisation lever, #104 as a byte lever.)

CARRIED INPUT BANDS (same scenario machinery as fern #100):
  * fp32-star-attn haircut (#98): on the tree-free path there is NO M=32
    star-attention (it is a tree-only attention pattern); the deployed M=8
    split-KV attention already runs fp32 partials (#98 Interp A), so the haircut
    is structurally ~0. Carried as the M=8 conservative bound 0.0102% (= 1.0e-4
    step units) for the conservative corner; central/optimistic 0. Negligible.
  * persist #97: CLOSED at 2.17% reclaimable idle (realizable ~1.76%), recommend
    CLOSE (< 3% build bar). On the tree-free path there is NO tree to hide the
    idle (no a_tree_hide dilution), so persist would reclaim its full ceiling.
    Carried UPSIDE-ONLY: 0 conservative, 0 central (CLOSED, not build-worth),
    0.0217 optimistic. NOTE: this supersedes #100's stale R_IDLE high=0.13 with
    denken #97's MERGED measurement -- the one place this gate deviates from
    #100's literal band, because #97 closed it.
  * tau #99 (local->official): [0.96, 1.00], central 1.00 (deployed ratio 1.0599
    folded into K_cal). The conservative corner's dominant driver.

GATE (Step 3):
  GREEN / tree is INSURANCE   tree-free clears 500 at a SplitK% ubel can
                              plausibly hit (inside the 23% gap, with LK+dq help)
                              -> the #101 build defect does NOT block 500.
  AMBER / tree strongly pref. clears 500 only at an aggressive SplitK% near the
                              top of the 23% gap, no margin -> name the cheapest
                              extra margin lever.
  RED   / tree CRITICAL-PATH  cannot clear 500 at any plausible SplitK% -> the
                              tree MUST be unblocked before quota.

PRIMARY metric  tree_free_max_official_tps  (central, with band)
TEST    metric  splitk_threshold_for_500

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no
served-file change. Greedy identity untouched by construction (all three levers
are greedy-lossless: SplitK 0-flip #87, double-quant bit-exact-or-sparse, LK
prediction-only). Composition consumes committed advisor-branch state only.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# ----------------------------------------------------------------------------
# Frontier anchors (committed). Leaderboard fa2sw-precache-splitkv-linear-mtp-k7.
# ----------------------------------------------------------------------------
FRONTIER_OFFICIAL = 481.53          # official best (lawine #52; private-verified VALID)
FRONTIER_LOCAL_WALL = 454.338       # K=7 locked local wall_tps (lawine #90)
E_T_LINEAR = 3.844                  # deployed MTP K=7 E[T] (#76/#90) -- tree-free stays here
K_CAL = FRONTIER_OFFICIAL / E_T_LINEAR                                  # 125.268
TARGET_OFFICIAL = 500.0             # theykk human directive

# Decode budget (M=8 linear; committed CURRENT_RESEARCH_STATE.md).
BUDGET = {"verify_gemm": 0.53, "drafter": 0.07, "attention": 0.08, "other": 0.32}

# SplitK headroom: verify-GEMM runs at 77.1% HBM util (#68 roofline). Fully
# closing the 22.9% gap (util 77.1% -> 100%) is the hard ceiling on s.
HBM_UTIL_M8 = 0.771
SPLITK_CEILING = 1.0 / HBM_UTIL_M8 - 1.0                                # 0.2971
SPLITK_UBEL = {"low": 0.05, "central": 0.085, "high": 0.12}            # ubel #84 sizing

# ---- lever bands (low, central, high) -- see module docstring for provenance --
# LK #95 E[T] multiplier on the linear chain (central near floor #80).
LK_MULT = {"low": 1.010, "central": 1.010, "high": 1.024}
# double-quant #104 ISOLATED TPS gain band; g=128 realistic central +0.5%, g=32
# upside +1.1% (re-quant, gated on re-validation). Converted to a verify-GEMM
# byte-reduction f_dq below.
DQ_TPS = {"low": 0.004, "central": 0.005, "high": 0.011}
# fp32-star-attn #98 ABS step add on the tree-free M=8 path (~0; #98 M=8 cons).
FP32_M8 = {"low": 0.0, "central": 0.0, "high": 0.000102}
# persist #97 reclaimable idle (UPSIDE-ONLY; #97 MERGED ceiling 2.17%, recommend
# CLOSE). No tree on this path -> reclaims full ceiling at the optimistic corner.
PERSIST_RECLAIM = {"low": 0.0, "central": 0.0, "high": 0.0217}
# local->official transfer tau (lawine #99).
TAU = {"low": 0.96, "central": 1.00, "high": 1.00}


def dq_tps_to_fdq(g: float) -> float:
    """Convert an ISOLATED double-quant TPS gain g (linear, s=0, no other lever)
    to a verify-GEMM byte-reduction fraction f_dq. Isolated: official scales as
    1/step, step = 1 - 0.53*f_dq, so 1+g = 1/(1 - 0.53*f_dq)."""
    return (g / (1.0 + g)) / BUDGET["verify_gemm"]


def compose(splitk_s: float, p: dict) -> dict:
    """Compose tree-free official-TPS at SplitK speedup `splitk_s` under point p.

    p holds lk_mult, f_dq (double-quant byte reduction), fp32_m8 (ABS add),
    persist_reclaim (ABS subtract), tau."""
    e_t = E_T_LINEAR * p["lk_mult"]                       # LK numerator
    # verify-GEMM: SplitK (bandwidth) x double-quant (bytes), orthogonal -> multiply.
    vg = BUDGET["verify_gemm"] * (1.0 - p["f_dq"]) / (1.0 + splitk_s)
    attn = BUDGET["attention"] + p["fp32_m8"]
    # residual = drafter + busy "other"; the reclaimable idle sits inside it and
    # is removed only when persist reclaims it (persist_reclaim>0).
    residual = (1.0 - BUDGET["verify_gemm"] - BUDGET["attention"]) - p["persist_reclaim"]
    step = vg + attn + residual
    official = K_CAL * (e_t / step) * p["tau"]
    return {"E_T": e_t, "step_time": step, "official_tps": official,
            "verify_gemm_slice": vg, "attn_slice": attn}


def point(scenario: str) -> dict:
    """conservative MINIMISES tree-free TPS; optimistic MAXIMISES it."""
    if scenario == "central":
        return {"lk_mult": LK_MULT["central"], "f_dq": dq_tps_to_fdq(DQ_TPS["central"]),
                "fp32_m8": FP32_M8["central"], "persist_reclaim": PERSIST_RECLAIM["central"],
                "tau": TAU["central"]}
    if scenario == "conservative":     # weak gains, heavy haircuts
        return {"lk_mult": LK_MULT["low"], "f_dq": dq_tps_to_fdq(DQ_TPS["low"]),
                "fp32_m8": FP32_M8["high"], "persist_reclaim": PERSIST_RECLAIM["low"],
                "tau": TAU["low"]}
    if scenario == "optimistic":       # strong gains, no haircuts
        return {"lk_mult": LK_MULT["high"], "f_dq": dq_tps_to_fdq(DQ_TPS["high"]),
                "fp32_m8": FP32_M8["low"], "persist_reclaim": PERSIST_RECLAIM["high"],
                "tau": TAU["high"]}
    raise ValueError(scenario)


def splitk_threshold_for_500(p: dict) -> float | None:
    """Invert compose() for the minimum SplitK speedup s that clears 500 under p.
    Returns None if even s=0 already clears 500 (s<=0), or float('inf') if the
    SplitK ceiling cannot reach 500."""
    e_t = E_T_LINEAR * p["lk_mult"]
    step_needed = K_CAL * e_t * p["tau"] / TARGET_OFFICIAL
    attn = BUDGET["attention"] + p["fp32_m8"]
    residual = (1.0 - BUDGET["verify_gemm"] - BUDGET["attention"]) - p["persist_reclaim"]
    vg_needed = step_needed - attn - residual           # required verify-GEMM slice
    if vg_needed <= 0:
        return float("inf")                              # impossible: vg can't be <=0
    vg_full = BUDGET["verify_gemm"] * (1.0 - p["f_dq"])  # vg at s=0
    one_plus_s = vg_full / vg_needed
    s = one_plus_s - 1.0
    if s <= 0:
        return 0.0                                       # clears already at s=0
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/tree_free_500_ceiling_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="denken/tree-free-500-ceiling")
    ap.add_argument("--wandb-group", default="tree-free-500-ceiling")
    args = ap.parse_args()

    pc, pcons, popt = point("central"), point("conservative"), point("optimistic")

    # ---------- Step 1: TPS-vs-SplitK% curve for the tree-free stack ----------
    s_grid = sorted(set(np.round(np.concatenate([
        np.linspace(0.0, 0.12, 13),                      # ubel's nominal window, fine
        np.linspace(0.13, SPLITK_CEILING, 10),           # out to the gap ceiling
        [SPLITK_UBEL["low"], SPLITK_UBEL["central"], SPLITK_UBEL["high"], SPLITK_CEILING],
    ]), 4).tolist()))
    curve = []
    for s in s_grid:
        row = {
            "splitk_pct": s,
            "conservative_tps": compose(s, pcons)["official_tps"],
            "central_tps": compose(s, pc)["official_tps"],
            "optimistic_tps": compose(s, popt)["official_tps"],
        }
        row["clears_500_central"] = row["central_tps"] >= TARGET_OFFICIAL
        row["clears_500_conservative"] = row["conservative_tps"] >= TARGET_OFFICIAL
        curve.append(row)

    # ---------- Step 1: tree_free_max_official_tps (primary) ----------
    # Realistic max = ubel's nominal-high SplitK (s=0.12); ceiling = full gap close.
    s_ubel_high = SPLITK_UBEL["high"]
    max_at_ubel_high = {sc: compose(s_ubel_high, point(sc))["official_tps"]
                        for sc in ("conservative", "central", "optimistic")}
    max_at_ceiling = {sc: compose(SPLITK_CEILING, point(sc))["official_tps"]
                      for sc in ("conservative", "central", "optimistic")}

    # ---------- Step 2: splitk_threshold_for_500 (test) ----------
    thr = {sc: splitk_threshold_for_500(point(sc))
           for sc in ("conservative", "central", "optimistic")}
    # tau-sensitivity of the threshold: tau is the conservative corner's main
    # driver. Hold LK+dq at CENTRAL, sweep tau over [0.96, 1.00].
    tau_sens = []
    for tau in [0.96, 0.97, 0.98, 0.99, 1.00]:
        p = dict(pc); p["tau"] = tau
        tau_sens.append({"tau": tau, "splitk_threshold": splitk_threshold_for_500(p)})
    # LK-off sensitivity: does central still clear at a reachable s if LK delivers 0?
    p_lkoff = dict(pc); p_lkoff["lk_mult"] = 1.0
    thr_central_lkoff = splitk_threshold_for_500(p_lkoff)
    # double-quant-off sensitivity.
    p_dqoff = dict(pc); p_dqoff["f_dq"] = 0.0
    thr_central_dqoff = splitk_threshold_for_500(p_dqoff)

    # ---------- Step 3: gate ----------
    def inside_ubel(s):  # within ubel's nominal [5%,12%] delivery
        return s is not None and s != float("inf") and SPLITK_UBEL["low"] <= s <= SPLITK_UBEL["high"]

    def below_ubel(s):   # clears even below ubel's floor (very comfortable)
        return s is not None and s != float("inf") and s < SPLITK_UBEL["low"]

    def inside_gap(s):   # reachable at all (within the 23% bandwidth-gap ceiling)
        return s is not None and s != float("inf") and s <= SPLITK_CEILING

    thr_c, thr_cons = thr["central"], thr["conservative"]
    # RED iff even the SplitK CEILING cannot clear 500 at the CENTRAL corner.
    red = not inside_gap(thr_c)
    # GREEN iff the CENTRAL threshold is comfortably reachable (<= ubel high) AND
    # the conservative threshold is at least inside the bandwidth-gap ceiling.
    green_central = (below_ubel(thr_c) or inside_ubel(thr_c)) and inside_gap(thr_cons)
    # Strict #100-style conservative gate: conservative threshold inside ubel range.
    green_conservative = inside_ubel(thr_cons) or below_ubel(thr_cons)

    if red:
        verdict = "RED"
        verdict_label = "tree CRITICAL-PATH (tree-free cannot clear 500 at any plausible SplitK)"
    elif green_conservative:
        verdict = "GREEN"
        verdict_label = "tree is INSURANCE (tree-free clears 500 inside ubel's SplitK range even at the conservative corner)"
    elif green_central:
        verdict = "GREEN*"
        verdict_label = ("tree is INSURANCE at central/expected inputs (SplitK threshold below ubel's floor); "
                         "conservative-corner caveat -> tree strongly preferred only if tau collapses to 0.96")
    else:
        verdict = "AMBER"
        verdict_label = "tree strongly preferred (tree-free clears 500 only at an aggressive SplitK near the gap ceiling)"

    # cheapest margin lever for the conservative corner: which single input, moved
    # to its central value, drops the conservative threshold the most?
    margin_probe = {}
    for lever, override in (("tau->1.00", {"tau": 1.00}),
                            ("LK->1.024(high)", {"lk_mult": 1.024}),
                            ("dq->+1.1%(high)", {"f_dq": dq_tps_to_fdq(0.011)}),
                            ("persist->2.17%", {"persist_reclaim": 0.0217})):
        p = dict(pcons); p.update(override)
        margin_probe[lever] = splitk_threshold_for_500(p)

    gate = {
        "primary_metric_name": "tree_free_max_official_tps",
        "tree_free_max_official_tps": {
            "definition": "central tree-free TPS at ubel #84 nominal-high SplitK (s=0.12)",
            "central": max_at_ubel_high["central"],
            "conservative": max_at_ubel_high["conservative"],
            "optimistic": max_at_ubel_high["optimistic"],
            "band": [max_at_ubel_high["conservative"], max_at_ubel_high["optimistic"]],
        },
        "tree_free_ceiling_official_tps": {
            "definition": "central tree-free TPS at the SplitK bandwidth-gap ceiling (s=%.4f, full 77.1%%->100%% close)" % SPLITK_CEILING,
            "central": max_at_ceiling["central"],
            "band": [max_at_ceiling["conservative"], max_at_ceiling["optimistic"]],
        },
        "test_metric_name": "splitk_threshold_for_500",
        "splitk_threshold_for_500": {
            "central": thr_c, "conservative": thr_cons, "optimistic": thr["optimistic"],
            "central_if_LK_delivers_0": thr_central_lkoff,
            "central_if_doublequant_0": thr_central_dqoff,
        },
        "splitk_reference": {
            "ubel_nominal_range": [SPLITK_UBEL["low"], SPLITK_UBEL["high"]],
            "bandwidth_gap_ceiling": SPLITK_CEILING,
            "hbm_util_m8": HBM_UTIL_M8,
            "central_threshold_as_frac_of_gap": (thr_c / SPLITK_CEILING) if inside_gap(thr_c) else None,
            "conservative_threshold_as_frac_of_gap": (thr_cons / SPLITK_CEILING) if inside_gap(thr_cons) else None,
        },
        "tau_sensitivity": tau_sens,
        "conservative_margin_levers": margin_probe,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "rule": ("GREEN=tree-free clears 500 inside ubel's SplitK range / "
                 "GREEN*=clears below ubel floor at central but conservative needs >ubel-high (tau-driven) / "
                 "AMBER=clears only near gap ceiling / RED=cannot clear within the gap"),
    }

    out = {
        "gate": gate,
        "step1_tps_vs_splitk_curve": curve,
        "model": {
            "formula": "official_TPS = K_cal * (E[T]/step) * tau ; tree-free, E[T]=3.844*lk_mult",
            "verify_gemm_netting": "vg = 0.53*(1-f_dq)/(1+s); SplitK=bandwidth(util), double-quant=bytes -> orthogonal, multiply, no double-count",
            "K_cal": K_CAL, "E_T_linear": E_T_LINEAR, "budget": BUDGET,
            "frontier_official": FRONTIER_OFFICIAL, "target_official": TARGET_OFFICIAL,
        },
        "band_inputs": {
            "lk_mult": LK_MULT, "doublequant_isolated_tps": DQ_TPS,
            "doublequant_f_dq": {k: dq_tps_to_fdq(v) for k, v in DQ_TPS.items()},
            "fp32_m8_abs": FP32_M8, "persist_reclaim_abs": PERSIST_RECLAIM, "tau": TAU,
            "splitk_ubel": SPLITK_UBEL, "splitk_ceiling": SPLITK_CEILING,
            "persist_note": "supersedes #100 R_IDLE high=0.13 with denken #97 MERGED 2.17% ceiling, upside-only",
            "fp32_note": "tree-free M=8 path has no star-attention; deployed split-KV already fp32 -> ~0",
            "lk_note": "LK prediction channel UNREALIZED (#95 AMBER, needs a head probe); applied as projected central 1.010",
        },
        "method": ("CPU-only analytic absolute-time slice composition (fern #100 model, "
                   "tree-EXCLUSIVE branch) over the committed decode budget + lever bands; "
                   "no GPU, no served-file change. Greedy identity untouched by construction."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    # ----------------------------- console -----------------------------
    def fmt(s):
        return "clears@s=0" if s == 0.0 else (">ceiling" if s == float("inf") else f"{s*100:.2f}%")

    print("=" * 84)
    print("TREE-FREE 500-PATH CEILING GATE (PR #105) -- build-complete stack, NO tree")
    print("=" * 84)
    print(f"\nfrontier {FRONTIER_OFFICIAL} official | target {TARGET_OFFICIAL} | E[T]=3.844 (linear, NO tree)")
    print(f"SplitK: ubel nominal {SPLITK_UBEL['low']*100:.0f}-{SPLITK_UBEL['high']*100:.0f}% | "
          f"bandwidth-gap ceiling {SPLITK_CEILING*100:.1f}% (close 77.1%->100% HBM)")
    print("\n[STEP 1] tree-free TPS vs SplitK% (conservative .. central .. optimistic):")
    print(f"  {'splitk%':>8s} {'cons':>8s} {'centr':>8s} {'opt':>8s}  clears500?")
    for r in curve:
        if r["splitk_pct"] in (0.0, 0.05, 0.085, 0.12, round(SPLITK_CEILING, 4)) or \
           abs(r["splitk_pct"] - round(thr_c, 4)) < 0.011:
            flag = "YES(cons)" if r["clears_500_conservative"] else (
                "yes(cent)" if r["clears_500_central"] else "no")
            print(f"  {r['splitk_pct']*100:7.2f}% {r['conservative_tps']:8.1f} "
                  f"{r['central_tps']:8.1f} {r['optimistic_tps']:8.1f}  {flag}")
    print(f"\n  tree_free_max_official_tps @ ubel-high s=12% (PRIMARY): "
          f"{max_at_ubel_high['central']:.1f} "
          f"[{max_at_ubel_high['conservative']:.1f}, {max_at_ubel_high['optimistic']:.1f}]")
    print(f"  tree_free_ceiling @ gap-close s={SPLITK_CEILING*100:.1f}%: "
          f"{max_at_ceiling['central']:.1f} "
          f"[{max_at_ceiling['conservative']:.1f}, {max_at_ceiling['optimistic']:.1f}]")
    print(f"\n[STEP 2] splitk_threshold_for_500 (TEST):")
    print(f"  central:      {fmt(thr_c)}  (= {(thr_c/SPLITK_CEILING*100):.0f}% of the gap)" if inside_gap(thr_c) else f"  central:      {fmt(thr_c)}")
    print(f"  conservative: {fmt(thr_cons)}" + (f"  (= {(thr_cons/SPLITK_CEILING*100):.0f}% of the gap)" if inside_gap(thr_cons) else ""))
    print(f"  optimistic:   {fmt(thr['optimistic'])}")
    print(f"  central if LK delivers 0:        {fmt(thr_central_lkoff)}")
    print(f"  central if double-quant delivers 0: {fmt(thr_central_dqoff)}")
    print(f"\n  tau-sensitivity of the threshold (LK+dq central):")
    for t in tau_sens:
        print(f"     tau={t['tau']:.2f} -> splitk_threshold {fmt(t['splitk_threshold'])}")
    print(f"\n  cheapest conservative-corner margin lever (threshold after each):")
    for lev, s in margin_probe.items():
        print(f"     {lev:18s} -> {fmt(s)}")
    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         job_type="analysis", config={
                             "gate": "tree-free-500-ceiling",
                             "method": "cpu-analytic-abs-time-slice-tree-exclusive",
                             "frontier_official": FRONTIER_OFFICIAL,
                             "target_official": TARGET_OFFICIAL,
                             "E_T_linear": E_T_LINEAR, "K_cal": K_CAL, "budget": BUDGET,
                             "splitk_ubel": SPLITK_UBEL, "splitk_ceiling": SPLITK_CEILING})

        def jnum(x):  # inf -> a large sentinel for W&B numeric panels
            return 9.99 if x == float("inf") else (x if x is not None else -1.0)

        s = wandb.summary
        s["tree_free_max_official_tps"] = max_at_ubel_high["central"]
        s["tree_free_max_official_tps_conservative"] = max_at_ubel_high["conservative"]
        s["tree_free_max_official_tps_optimistic"] = max_at_ubel_high["optimistic"]
        s["tree_free_ceiling_official_tps"] = max_at_ceiling["central"]
        s["splitk_threshold_for_500_central"] = jnum(thr_c)
        s["splitk_threshold_for_500_conservative"] = jnum(thr_cons)
        s["splitk_threshold_for_500_optimistic"] = jnum(thr["optimistic"])
        s["splitk_threshold_central_if_LK_0"] = jnum(thr_central_lkoff)
        s["splitk_bandwidth_gap_ceiling"] = SPLITK_CEILING
        s["ubel_nominal_high"] = SPLITK_UBEL["high"]
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label
        # curve table
        ct = wandb.Table(columns=["splitk_pct", "conservative_tps", "central_tps",
                                  "optimistic_tps", "clears_500_central",
                                  "clears_500_conservative"])
        for r in curve:
            ct.add_data(r["splitk_pct"], r["conservative_tps"], r["central_tps"],
                        r["optimistic_tps"], r["clears_500_central"],
                        r["clears_500_conservative"])
            wandb.log({"curve/splitk_pct": r["splitk_pct"],
                       "curve/central_tps": r["central_tps"],
                       "curve/conservative_tps": r["conservative_tps"],
                       "curve/optimistic_tps": r["optimistic_tps"],
                       "curve/target": TARGET_OFFICIAL})
        wandb.log({"tps_vs_splitk": ct})
        tt = wandb.Table(columns=["tau", "splitk_threshold_for_500"])
        for t in tau_sens:
            tt.add_data(t["tau"], jnum(t["splitk_threshold"]))
        wandb.log({"tau_sensitivity": tt})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
