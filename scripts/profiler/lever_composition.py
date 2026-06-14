#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Lever-composition economics gate (PR #100): the COMPOSED official-TPS
landscape over the in-flight 500-path levers, and the minimal lever ORDERING
that clears 500 with the most margin -- accounting for which pairs compound vs
anti-compound.

WHY THIS GATE EXISTS
--------------------
Every in-flight lever on the 500-path is now sized or being sized, but they are
priced *in isolation*. Some compound on the E[T] numerator; some attack the same
wall-time-per-step denominator; and at least one ANTI-compounds with the tree.
This gate folds the E[T]-numerator levers and the wall-time-denominator levers
into ONE official-TPS projection so the team sequences the 500-path optimally
instead of pricing levers in isolation.

THE MODEL (absolute-time slice composition)
-------------------------------------------
official_TPS = K_cal * (E[T] / step_time) * tau

where the decode step is decomposed into ABSOLUTE time slices (normalised so the
deployed M=8 linear-MTP step = 1.0), each lever modifies the slice it acts on,
and tau is the local->official transfer (lawine #99). Working in absolute-time
space -- not relative multipliers -- is what makes the composition correct:
  * E[T]-NUMERATOR levers (tree #71, LK #95) multiply E[T];
  * WALL-TIME-DENOMINATOR levers (SplitK #84, persistent-kernel #97) subtract an
    ABSOLUTE saving from their slice. Two denominator levers on DIFFERENT slices
    add their absolute savings (the final step is order-independent); the
    apparent "ordering matters" is only an artefact of attributing RELATIVE gains
    sequentially.

K_cal is calibrated so the deployed frontier reproduces exactly:
  official = 481.53 at (E[T]=3.844, step=1.0, tau=1)  ->  K_cal = 481.53/3.844.
At tau=1 with only the tree on, TPS = 481.53*(1+net_tree), i.e. the committed
denken #85 / fern #92 projection (568 central, band 558-581).

DECODE BUDGET (committed, M=8 linear -- research/CURRENT_RESEARCH_STATE.md):
  verify-GEMM ~53% (int4 W4A16 Marlin, weight-BW-bound, FLAT for M<=32, hard tile
  cliff M=33), drafter ~7%, attention <8% (conc=1 floor), ~32% other/overhead
  (host-device scheduling, Python round-trips -- largely un-mined).

LEVER CLASSIFICATION (Step 1) -- WHERE each acts on official-TPS=f(E[T], t_step):
  tree #71      E[T] NUMERATOR (+ M-widen denominator). NET committed effect:
                E[T] 3.844->5.207, net +18.2% (denken #83/#85; fern #92 band
                558-581 official). verify-GEMM stays 0.53 ABS (FLAT M<=32, #85);
                attention amortises 1.06x (#85). Carries the fp32-star-attn
                haircut (wirbel #98) on the denominator.
  SplitK #84    WALL-TIME DENOMINATOR. +5-12% verify-GEMM speedup on the 0.53 ABS
                GEMM slice; same absolute saving in linear AND tree (GEMM flat).
  persist #97   WALL-TIME DENOMINATOR. Reclaims the GPU-idle slice of the ~32%
                "other" bucket (+8-15% IFF idle; denken #65 says decode 99.41%
                GPU-bound and #94 says the A10G bus serialises -> prior is LOW).
                ANTI-COMPOUNDS with the tree: the tree's longer M=32 compute
                hides part of the reclaimable idle AND amortises the fixed
                per-step overhead over more tokens -> persistent-kernel reclaims
                less once the tree lands.
  LK #95        E[T] NUMERATOR. +1.0-2.4% E[T] (fern #95 MERGED; prediction
                channel only, re-rank channel CLOSED). Compounds multiplicatively
                with the tree on accept_length, BUT carries a partial-overlap
                caveat: LK promotes rank-2->rank-1 mass the tree already harvests
                root-to-leaf (#88) -> its marginal value SHRINKS on the tree.

INPUTS CARRIED AS BANDS (pending gates; sized from the PR body + committed
advisor-branch state, NOT by inspecting other students' unmerged branches):
  fp32-star-attn haircut (wirbel #98): ABS add to the step, [0, 0.04], central
        0.01 ("likely ~free at the conc=1 BW-bound attn floor"; tail-only-fp32
        on the 0.537% near-tie tail is the fallback).
  GPU-idle reclaim       (denken #97): R_idle in [0, 0.13] linear (0=RED/busy,
        0.13=+15% GREEN), central LOW 0.03 given the #65/#94 GPU-bound prior;
        tree hides a_tree_hide in [0, 0.5] of it (central 0.30).
  local->official mult   (lawine #99): tau in [0.96, 1.00], central 1.00 (the
        deployed config's measured ratio 481.53/454.338=1.0599 is folded into
        K_cal; tau stress-tests whether that ratio holds for the new configs).

GATE (Step 3):
  GREEN / tree-sufficient   tree alone clears 500 with margin even at the
                            CONSERVATIVE corner (fp32 haircut #98 + conservative
                            tau #99) -> compounding levers are INSURANCE.
  AMBER  / one-lever-needed tree clears 500 only stacked with ONE lever -> name
                            the cheapest sufficient lever + the ordering.
  RED    / stack-insufficient tree + all in-flight levers straddle 500 under
                            conservative bands -> escalate; a new lever CLASS is
                            needed; size the residual gap.

PRIMARY metric  composed_official_tps  (full stack, with band)
TEST    metric  min_levers_to_clear_500 + the anti-compounding map

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no
served-file change. A projection model computes nothing served -> greedy
identity untouched by construction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

# ----------------------------------------------------------------------------
# Frontier anchors (committed). Leaderboard fa2sw-precache-splitkv-linear-mtp-k7.
# ----------------------------------------------------------------------------
FRONTIER_OFFICIAL = 481.53           # official best (lawine #52; private-verified VALID)
FRONTIER_LOCAL_WALL = 454.338        # K=7 locked local wall_tps (lawine #90)
E_T_LINEAR = 3.844                   # deployed MTP K=7 E[T] (#76/#90)
E_T_TREE = 5.207                     # realized tree E[T] (fern #92 5.20824; #91 5.20695)
DEPLOYED_LOCAL_TO_OFFICIAL = FRONTIER_OFFICIAL / FRONTIER_LOCAL_WALL   # 1.0599

# Calibration: official = K_CAL * E[T]/step * tau, fixed so linear reproduces 481.53.
K_CAL = FRONTIER_OFFICIAL / E_T_LINEAR                                  # 125.268

# Decode budget (M=8 linear; committed CURRENT_RESEARCH_STATE.md line 10).
BUDGET = {"verify_gemm": 0.53, "drafter": 0.07, "attention": 0.08, "other": 0.32}
ATTN_TREE_AMORTIZE = 1.06            # denken #85: M=32 attention = 1.06x M=8 (split-KV)

TARGET_OFFICIAL = 500.0              # theykk human directive: "target is 500tps"

# ---- lever bands (low, central, high) -- see module docstring for provenance --
# tree NET local gain spanning fern #92's 558-581 official envelope at tau=1.
NET_TREE = {"low": 558.0 / FRONTIER_OFFICIAL - 1.0,   # 0.1588 -> 558
            "central": 568.0 / FRONTIER_OFFICIAL - 1.0,  # 0.1796 -> 568
            "high": 581.0 / FRONTIER_OFFICIAL - 1.0}   # 0.2065 -> 581
# SplitK #84 verify-GEMM speedup s (acts on the 0.53 ABS GEMM slice).
SPLITK_S = {"low": 0.05, "central": 0.085, "high": 0.12}
# LK #95 E[T] multiplier; STANDALONE on the linear chain.
LK_MULT = {"low": 1.010, "central": 1.010, "high": 1.024}  # central nearer floor (#80)
# LK on the TREE: partial-overlap haircut (tree already harvests promoted rank-2).
LK_MULT_TREE = {"low": 1.005, "central": 1.008, "high": 1.024}
# fp32-star-attn haircut (wirbel #98): ABS add to the step (linear-step units).
FP32_HAIRCUT = {"low": 0.0, "central": 0.01, "high": 0.04}
# persistent-kernel #97 reclaimable GPU-idle slice R_idle (linear-step units).
R_IDLE = {"low": 0.0, "central": 0.03, "high": 0.13}
# fraction of R_idle the tree's longer compute HIDES (anti-compound, denken #97).
A_TREE_HIDE = {"low": 0.0, "central": 0.30, "high": 0.50}
# local->official transfer tau (lawine #99).
TAU = {"low": 0.96, "central": 1.00, "high": 1.00}

LEVERS = ("tree", "splitk", "persist", "lk")


# ----------------------------------------------------------------------------
# Core composition: build the decode step in ABSOLUTE time, return official TPS.
# ----------------------------------------------------------------------------
def compose(levers: set[str], p: dict) -> dict:
    """Compose official-TPS for a set of levers under parameter point `p`.

    `p` holds the scalar value of every band input (net_tree, splitk_s, lk_mult,
    lk_mult_tree, fp32_haircut, r_idle, a_tree_hide, tau). Returns the resolved
    step decomposition + official TPS so the caller can audit every slice.
    """
    tree = "tree" in levers
    splitk = "splitk" in levers
    persist = "persist" in levers
    lk = "lk" in levers

    # ---- E[T] numerator ----
    e_t = E_T_TREE if tree else E_T_LINEAR
    if lk:
        e_t *= p["lk_mult_tree"] if tree else p["lk_mult"]

    # ---- wall-time denominator: build the step from ABS slices ----
    vg = BUDGET["verify_gemm"]                       # 0.53; FLAT under tree (#85)
    if splitk:
        vg = vg / (1.0 + p["splitk_s"])              # ABS saving 0.53*s/(1+s)

    if tree:
        # Tree NET commitment pins the pre-haircut step (denken #85 / fern #92):
        #   step_base = (E_T_tree/E_T_lin)/(1+net_tree)  -> TPS=481.53*(1+net) at tau=1.
        step_base = (E_T_TREE / E_T_LINEAR) / (1.0 + p["net_tree"])
        attn = BUDGET["attention"] * ATTN_TREE_AMORTIZE + p["fp32_haircut"]
        # idle remaining for persistent-kernel AFTER the tree hides part of it.
        idle = p["r_idle"] * (1.0 - p["a_tree_hide"])
        # residual (drafter + busy overhead) absorbs the net-band variation; the
        # fp32 haircut adds to BOTH attn and the total, so it nets out of residual.
        residual = (step_base + p["fp32_haircut"]) - BUDGET["verify_gemm"] - attn - idle
        if persist:
            idle = 0.0                               # reclaim the remaining idle
        step = vg + attn + idle + residual
    else:
        # Linear M=8 regime (no star-attn tree -> no fp32 haircut).
        attn = BUDGET["attention"]
        idle = p["r_idle"]
        residual = 1.0 - BUDGET["verify_gemm"] - attn - idle  # = drafter+other_busy
        if persist:
            idle = 0.0
        # vg may already be SplitK-reduced; rebuild from the (possibly reduced) vg.
        step = vg + attn + idle + residual

    official = K_CAL * (e_t / step) * p["tau"]
    return {"levers": sorted(levers), "E_T": e_t, "step_time": step,
            "official_tps": official, "verify_gemm_slice": vg, "attn_slice": attn}


def point(scenario: str, overrides: dict | None = None) -> dict:
    """Resolve a full parameter point. scenario in {central, conservative,
    optimistic}; conservative MINIMISES composed TPS, optimistic MAXIMISES it."""
    if scenario == "central":
        p = {"net_tree": NET_TREE["central"], "splitk_s": SPLITK_S["central"],
             "lk_mult": LK_MULT["central"], "lk_mult_tree": LK_MULT_TREE["central"],
             "fp32_haircut": FP32_HAIRCUT["central"], "r_idle": R_IDLE["central"],
             "a_tree_hide": A_TREE_HIDE["central"], "tau": TAU["central"]}
    elif scenario == "conservative":   # worst case: weak gains, heavy haircuts
        p = {"net_tree": NET_TREE["low"], "splitk_s": SPLITK_S["low"],
             "lk_mult": LK_MULT["low"], "lk_mult_tree": LK_MULT_TREE["low"],
             "fp32_haircut": FP32_HAIRCUT["high"], "r_idle": R_IDLE["low"],
             "a_tree_hide": A_TREE_HIDE["high"], "tau": TAU["low"]}
    elif scenario == "optimistic":     # best case: strong gains, no haircuts
        p = {"net_tree": NET_TREE["high"], "splitk_s": SPLITK_S["high"],
             "lk_mult": LK_MULT["high"], "lk_mult_tree": LK_MULT_TREE["high"],
             "fp32_haircut": FP32_HAIRCUT["low"], "r_idle": R_IDLE["high"],
             "a_tree_hide": A_TREE_HIDE["low"], "tau": TAU["high"]}
    else:
        raise ValueError(scenario)
    if overrides:
        p.update(overrides)
    return p


def classify_pair(base_levers: set[str], a: str, b: str, p: dict) -> dict:
    """On top of `base_levers`, classify how marginal gains of a and b combine:
    additive / multiplicative(compound) / anti-compound(fight) / super-compound.

    Compares the JOINT marginal gain to the sum and the product of the individual
    marginal gains. Anti-compound = joint materially BELOW the additive sum."""
    t0 = compose(base_levers, p)["official_tps"]
    ta = compose(base_levers | {a}, p)["official_tps"]
    tb = compose(base_levers | {b}, p)["official_tps"]
    tab = compose(base_levers | {a, b}, p)["official_tps"]
    ga, gb, gab = ta / t0 - 1, tb / t0 - 1, tab / t0 - 1
    add = ga + gb
    mult = (1 + ga) * (1 + gb) - 1
    # ratio of joint to the additive null; <1 fights, ~1 additive, >mult super.
    ratio_add = gab / add if abs(add) > 1e-9 else float("nan")
    if abs(add) < 1e-9:
        kind = "neutral (one lever ~0 on this base)"
    elif gab < add * 0.97:
        kind = "ANTI-COMPOUND (fight)"
    elif gab > mult * 1.02:
        kind = "super-compound"
    elif gab >= mult * 0.98:
        kind = "multiplicative (compound)"
    else:
        kind = "additive"
    return {"base": sorted(base_levers), "pair": [a, b],
            "marginal_gain_a_pct": ga * 100, "marginal_gain_b_pct": gb * 100,
            "joint_gain_pct": gab * 100, "additive_null_pct": add * 100,
            "multiplicative_null_pct": mult * 100, "joint_over_additive": ratio_add,
            "kind": kind}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/lever_composition_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/lever-composition-economics")
    ap.add_argument("--wandb-group", default="lever-composition-economics")
    args = ap.parse_args()

    pc = point("central")
    pcons = point("conservative")
    popt = point("optimistic")

    # ---------- Step 2: compose the official-TPS landscape over subsets ----------
    # Every subset that matters for the 500-path sequencing decision.
    subsets = [
        set(), {"tree"}, {"splitk"}, {"persist"}, {"lk"},
        {"tree", "splitk"}, {"tree", "persist"}, {"tree", "lk"},
        {"tree", "splitk", "persist"}, {"tree", "splitk", "lk"},
        {"tree", "persist", "lk"}, {"tree", "splitk", "persist", "lk"},
    ]
    landscape = []
    for s in subsets:
        row = {
            "levers": sorted(s) if s else ["(frontier)"],
            "n_levers": len(s),
            "central": compose(s, pc),
            "conservative": compose(s, pcons),
            "optimistic": compose(s, popt),
        }
        row["central_tps"] = row["central"]["official_tps"]
        row["conservative_tps"] = row["conservative"]["official_tps"]
        row["optimistic_tps"] = row["optimistic"]["official_tps"]
        row["clears_500_central"] = row["central_tps"] >= TARGET_OFFICIAL
        row["clears_500_conservative"] = row["conservative_tps"] >= TARGET_OFFICIAL
        row["margin_conservative"] = row["conservative_tps"] - TARGET_OFFICIAL
        landscape.append(row)

    # ---------- Step 3: minimal lever set to clear 500 (CONSERVATIVE gate) ----------
    # Honest stress test: the minimal subset whose CONSERVATIVE-corner TPS >= 500.
    def min_set_clearing(scenario_key: str) -> dict:
        best = None
        for s in subsets:
            if not s:
                continue
            row = next(r for r in landscape if set(
                r["levers"]) == s or (r["levers"] == ["(frontier)"] and not s))
            if row[f"{scenario_key}_tps"] >= TARGET_OFFICIAL:
                if best is None or len(s) < len(best["set"]):
                    best = {"set": s, "tps": row[f"{scenario_key}_tps"]}
                elif len(s) == len(best["set"]) and row[f"{scenario_key}_tps"] > best["tps"]:
                    best = {"set": s, "tps": row[f"{scenario_key}_tps"]}
        return best or {"set": None, "tps": None}

    min_cons = min_set_clearing("conservative")
    min_cent = min_set_clearing("central")
    tree_alone = next(r for r in landscape if r["levers"] == ["tree"])
    full_stack = next(r for r in landscape if r["n_levers"] == 4)

    # ---------- anti-compounding map (pairwise, on the tree base) ----------
    pairs = [("splitk", "persist"), ("splitk", "lk"), ("persist", "lk")]
    anti_map = [classify_pair({"tree"}, a, b, pc) for a, b in pairs]
    # tree vs each denominator/numerator lever: marginal on tree vs standalone.
    tree_interactions = []
    for lev in ("splitk", "persist", "lk"):
        standalone = compose({lev}, pc)["official_tps"] / compose(set(), pc)["official_tps"] - 1
        on_tree = compose({"tree", lev}, pc)["official_tps"] / tree_alone["central_tps"] - 1
        tree_interactions.append({
            "lever": lev,
            "standalone_gain_pct": standalone * 100,
            "marginal_on_tree_pct": on_tree * 100,
            "tree_dilution_ratio": (on_tree / standalone) if abs(standalone) > 1e-9 else float("nan"),
            "kind": ("ANTI-COMPOUND (tree shrinks its value)"
                     if on_tree < standalone * 0.97 else
                     "compounds (≈multiplicative)" if on_tree >= standalone * 0.97
                     else "neutral"),
        })

    # ---------- verdict ----------
    if tree_alone["clears_500_conservative"]:
        verdict = "GREEN"
        verdict_label = "tree-sufficient (compounding levers are INSURANCE)"
    elif min_cons["set"] is not None and len(min_cons["set"]) == 2:
        verdict = "AMBER"
        verdict_label = "one-lever-needed"
    elif full_stack["clears_500_conservative"]:
        verdict = "AMBER"
        verdict_label = "stack-needed"
    else:
        verdict = "RED"
        verdict_label = "stack-insufficient (new lever CLASS needed)"

    cheapest_after_tree = None
    if verdict == "AMBER" and min_cons["set"] and len(min_cons["set"]) == 2:
        extra = (min_cons["set"] - {"tree"})
        cheapest_after_tree = sorted(extra)[0] if extra else None

    gate = {
        "primary_metric_name": "composed_official_tps",
        "composed_official_tps_full_stack": {
            "central": full_stack["central_tps"],
            "conservative": full_stack["conservative_tps"],
            "optimistic": full_stack["optimistic_tps"],
            "band": [full_stack["conservative_tps"], full_stack["optimistic_tps"]],
        },
        "test_metric_name": "min_levers_to_clear_500",
        "min_levers_to_clear_500_conservative": {
            "set": sorted(min_cons["set"]) if min_cons["set"] else None,
            "n": len(min_cons["set"]) if min_cons["set"] else None,
            "tps": min_cons["tps"]},
        "min_levers_to_clear_500_central": {
            "set": sorted(min_cent["set"]) if min_cent["set"] else None,
            "n": len(min_cent["set"]) if min_cent["set"] else None,
            "tps": min_cent["tps"]},
        "tree_alone_official": {
            "central": tree_alone["central_tps"],
            "conservative": tree_alone["conservative_tps"],
            "optimistic": tree_alone["optimistic_tps"],
            "clears_500_conservative": tree_alone["clears_500_conservative"]},
        "cheapest_sufficient_lever_after_tree": cheapest_after_tree,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "rule": ("GREEN=tree clears 500 at conservative corner / "
                 "AMBER=needs 1 lever or full stack / RED=stack straddles 500"),
    }

    out = {
        "gate": gate,
        "step1_lever_classification": {
            "tree": {"axis": "E[T]-numerator (+M-widen denominator)",
                     "effect": "E[T] 3.844->5.207; net +18.2% (band 558-581 official)",
                     "carries": "fp32-star-attn haircut (wirbel #98) on denominator"},
            "splitk": {"axis": "wall-time denominator",
                       "effect": "+5-12% verify-GEMM speedup on the 0.53 ABS GEMM slice; FLAT M<=32 so applies on the tree's widened GEMM too"},
            "persist": {"axis": "wall-time denominator",
                        "effect": "reclaims GPU-idle slice of the ~32% other bucket (+8-15% IFF idle)",
                        "anti_compounds_with": "tree (longer M=32 compute hides idle + amortises fixed overhead)"},
            "lk": {"axis": "E[T]-numerator",
                   "effect": "+1.0-2.4% E[T] (prediction channel; re-rank CLOSED)",
                   "caveat": "partial overlap with tree rank-2 harvest -> marginal value shrinks on the tree"},
        },
        "step2_landscape": landscape,
        "step3_anti_compound_map": anti_map,
        "step3_tree_interactions": tree_interactions,
        "frontier_anchors": {
            "official": FRONTIER_OFFICIAL, "local_wall": FRONTIER_LOCAL_WALL,
            "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE,
            "deployed_local_to_official": DEPLOYED_LOCAL_TO_OFFICIAL,
            "K_cal": K_CAL, "target_official": TARGET_OFFICIAL},
        "band_inputs": {
            "net_tree": NET_TREE, "splitk_s": SPLITK_S, "lk_mult": LK_MULT,
            "lk_mult_tree": LK_MULT_TREE, "fp32_haircut_abs": FP32_HAIRCUT,
            "r_idle_abs": R_IDLE, "a_tree_hide": A_TREE_HIDE, "tau": TAU},
        "method": ("CPU-only analytic absolute-time slice composition off the "
                   "committed decode budget + lever bands; no GPU, no served-file "
                   "change. Pending gates (#97/#98/#99) carried as bands."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(
            o, np.generic) else o)

    # ----------------------------- console -----------------------------
    print("=" * 80)
    print("LEVER-COMPOSITION ECONOMICS GATE (PR #100) -- composed official-TPS landscape")
    print("=" * 80)
    print(f"\nfrontier {FRONTIER_OFFICIAL} official / {FRONTIER_LOCAL_WALL} local "
          f"(ratio {DEPLOYED_LOCAL_TO_OFFICIAL:.4f}) | target {TARGET_OFFICIAL} | "
          f"E[T] {E_T_LINEAR}->{E_T_TREE}")
    print("\n[STEP 2] composed official-TPS landscape (conservative .. central .. optimistic):")
    print(f"  {'levers':38s} {'cons':>8s} {'centr':>8s} {'opt':>8s}  clears500?")
    for r in sorted(landscape, key=lambda x: x["central_tps"]):
        name = "+".join(r["levers"]) if r["levers"] != ["(frontier)"] else "(frontier)"
        flag = "YES(cons)" if r["clears_500_conservative"] else (
            "yes(cent)" if r["clears_500_central"] else "no")
        print(f"  {name:38s} {r['conservative_tps']:8.1f} {r['central_tps']:8.1f} "
              f"{r['optimistic_tps']:8.1f}  {flag}")
    print(f"\n[STEP 3] tree ALONE: cons {tree_alone['conservative_tps']:.1f} / "
          f"cent {tree_alone['central_tps']:.1f} / opt {tree_alone['optimistic_tps']:.1f} "
          f"-> clears500@conservative={tree_alone['clears_500_conservative']}")
    print(f"   min levers to clear 500 (CONSERVATIVE): "
          f"{sorted(min_cons['set']) if min_cons['set'] else None} "
          f"(tps {min_cons['tps']:.1f})" if min_cons["set"] else
          "   min levers to clear 500 (CONSERVATIVE): NONE clears")
    print(f"   full stack: cons {full_stack['conservative_tps']:.1f} / "
          f"cent {full_stack['central_tps']:.1f} / opt {full_stack['optimistic_tps']:.1f}")
    print("\n   anti-compound map (pairwise marginal, on the TREE base):")
    for m in anti_map:
        print(f"     {m['pair'][0]:8s} x {m['pair'][1]:8s}: joint {m['joint_gain_pct']:+.2f}% "
              f"vs additive {m['additive_null_pct']:+.2f}% -> {m['kind']}")
    print("\n   tree x lever interactions (standalone vs marginal-on-tree):")
    for m in tree_interactions:
        print(f"     tree x {m['lever']:8s}: standalone {m['standalone_gain_pct']:+.2f}% -> "
              f"on-tree {m['marginal_on_tree_pct']:+.2f}% "
              f"(dilution {m['tree_dilution_ratio']:.2f}) -> {m['kind']}")
    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"   composed_official_tps (full stack) = {full_stack['central_tps']:.1f} "
          f"[{full_stack['conservative_tps']:.1f}, {full_stack['optimistic_tps']:.1f}]")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         job_type="analysis", config={
                             "gate": "lever-composition-economics",
                             "method": "cpu-analytic-abs-time-slice-composition",
                             "frontier_official": FRONTIER_OFFICIAL,
                             "target_official": TARGET_OFFICIAL,
                             "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE,
                             "K_cal": K_CAL, "budget": BUDGET})
        s = wandb.summary
        s["composed_official_tps"] = full_stack["central_tps"]
        s["composed_official_tps_conservative"] = full_stack["conservative_tps"]
        s["composed_official_tps_optimistic"] = full_stack["optimistic_tps"]
        s["tree_alone_official_central"] = tree_alone["central_tps"]
        s["tree_alone_official_conservative"] = tree_alone["conservative_tps"]
        s["tree_alone_clears_500_conservative"] = tree_alone["clears_500_conservative"]
        s["min_levers_to_clear_500_conservative"] = (
            len(min_cons["set"]) if min_cons["set"] else -1)
        s["min_levers_to_clear_500_central"] = (
            len(min_cent["set"]) if min_cent["set"] else -1)
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label
        s["cheapest_sufficient_lever_after_tree"] = cheapest_after_tree or "none-needed"
        # landscape table
        lt = wandb.Table(columns=["levers", "n_levers", "conservative_tps",
                                  "central_tps", "optimistic_tps",
                                  "clears_500_conservative", "E_T_central",
                                  "step_time_central"])
        for r in landscape:
            lt.add_data("+".join(r["levers"]), r["n_levers"], r["conservative_tps"],
                        r["central_tps"], r["optimistic_tps"],
                        r["clears_500_conservative"], r["central"]["E_T"],
                        r["central"]["step_time"])
        wandb.log({"composed_landscape": lt})
        # anti-compound table
        at = wandb.Table(columns=["pair", "marginal_a_pct", "marginal_b_pct",
                                  "joint_pct", "additive_null_pct",
                                  "multiplicative_null_pct", "kind"])
        for m in anti_map:
            at.add_data(f"{m['pair'][0]}x{m['pair'][1]}", m["marginal_gain_a_pct"],
                        m["marginal_gain_b_pct"], m["joint_gain_pct"],
                        m["additive_null_pct"], m["multiplicative_null_pct"], m["kind"])
        wandb.log({"anti_compound_map": at})
        it = wandb.Table(columns=["lever", "standalone_gain_pct",
                                  "marginal_on_tree_pct", "tree_dilution_ratio", "kind"])
        for m in tree_interactions:
            it.add_data(m["lever"], m["standalone_gain_pct"], m["marginal_on_tree_pct"],
                        m["tree_dilution_ratio"], m["kind"])
        wandb.log({"tree_interactions": it})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
