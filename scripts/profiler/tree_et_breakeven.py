#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree E[T] break-even / margin-of-safety gate (PR #102) -- the INVERSE of the
PR #100 lever-composition model.

WHY THIS GATE EXISTS
--------------------
PR #100 (MERGED, GREEN) proved the tree clears 500 official-TPS with margin --
but ONLY conditional on the realized tree accept_length E[T] reaching the
analytical 5.207. byteshark's first real tree build (`tree-v2-merge-eager-v1`,
CURRENT_RESEARCH_STATE.md L44) delivers tok/step = 2.097 -- accept-length
COLLAPSED, far below the linear chain's 3.844, nowhere near 5.207. denken #101
is diagnosing WHERE the 2.10-vs-5.207 gap comes from and what is RECOVERABLE.
This gate answers the complement -- what is REQUIRED:

    Given official_TPS = 500, what is the MINIMUM realized tree accept_length
    E[T]* -- alone, and with each compounding lever stacked -- and where do
    byteshark's 2.097 and denken #101's recoverable band fall relative to it?

The intersection of "what denken says we can recover" and "what this gate says
we need" is the real go/no-go for the #1 lever.

THE INVERSION (why it is a clean linear rescaling)
--------------------------------------------------
The #100 model is
        official_TPS = K_cal * (E[T] / step_time) * tau
which is LINEAR in the accept_length numerator E[T]. The tree's step_time is set
by the M=32 TOPOLOGY -- verify-GEMM width (FLAT M<=32), attention amortization,
drafter, host/overhead, the fp32 star-attn haircut, and any denominator levers
(SplitK, persistent-kernel). It is NOT a function of how many drafted tokens get
ACCEPTED. accept_length is a NUMERATOR-only quality property of the drafter's
guesses. So we may hold step_time at its topology value and solve official=500:

    E[T]*_raw = 500 * step_time / (K_cal * lk_factor * tau)
              = E_T_TREE * 500 / official_TPS_stack(@E[T]=E_T_TREE)

The second form is an exact rescaling: #100 already evaluated official_TPS at
E[T]=5.207 for every lever stack, so the break-even raw tree accept_length is
just 5.207 * 500 / (that stack's official_TPS@5.207). `lk_factor` is folded in
because #100's numerator for the LK stacks is E_T_TREE*lk_mult_tree, so dividing
by that stack's official already returns the RAW (pre-LK-boost) accept_length.

This SEPARATES the two quantities #100's `net_tree` band bundled together: the
M=32 denominator widening (a step_time fact) vs the accept-length numerator gain
(the free variable here). It is the physically honest decomposition -- and it is
faithful to #100 by construction (it reuses #100's `compose`/`point` verbatim).

WHAT WE PLACE ON THE E[T] AXIS
------------------------------
  byteshark as-built  2.097  (where the build is now; accept-hist mean ~2.10)
  linear-chain        3.844  (the linear MTP K=7 accept_length)
  beat-linear-OFFICIAL ~4.45 (the tree accept_length that TIES linear's 481.53
                              official -- higher than 3.844 because the tree
                              step is ~1.16x heavier; beating linear's
                              accept_length is NOT enough)
  analytical          5.207  (the de-risked ceiling, fern #92 / denken #85)
  denken #101 band    PARAMETERIZED until it lands (a free variable E_rec)

GATE (on breakeven_ET_tree_alone)
---------------------------------
  GREEN / robust   breakeven_ET_tree_alone <= ~4.0 -- the tree clears 500 even
                   if the build only recovers to ~linear-chain accept_length;
                   compounding levers give comfortable margin.
  AMBER            breakeven in ~4.0-5.0 -- the tree must recover MOST of the way
                   to 5.207, likely needs a compounding lever; name the cheapest
                   stack that pulls E[T]* to a reachable level.
  RED / fragile    breakeven near 5.207 with no lever able to lower it -- the
                   500-path has no margin; any build shortfall sinks it.

PRIMARY metric  breakeven_ET_tree_alone
TEST    metric  ET_recovery_needed_from_2p10 (full lever-stack ladder)

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no
served-file change. A projection model computes nothing served -> greedy
identity untouched by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Reuse the PR #100 model VERBATIM so the inversion is guaranteed consistent
# with the forward composition it inverts.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever_composition import (  # noqa: E402
    E_T_LINEAR,
    E_T_TREE,
    FRONTIER_OFFICIAL,
    K_CAL,
    LK_MULT_TREE,
    TARGET_OFFICIAL,
    compose,
    point,
)

# Empirical / reference accept_length anchors on the E[T] axis.
BYTESHARK_ASBUILT_ET = 2.097          # tree-v2-merge-eager-v1 tok/step (CRS L44)
LINEAR_CHAIN_ET = E_T_LINEAR          # 3.844 -- the linear MTP K=7 accept_length
ANALYTICAL_ET = E_T_TREE              # 5.207 -- the de-risked ceiling
# byteshark accept histogram (CRS L44): index = #accepted tokens, value = #steps.
BYTESHARK_ACCEPT_HIST = [0, 5761, 5061, 1765, 854, 355, 214, 126, 200]

# The four cumulative lever stacks the PR names (the primary ladder), plus the
# persist-inclusive upside rows for completeness (persist ~= 0 at the
# conservative corner per denken #97: only 2.17% reclaimable idle).
STACKS_PRIMARY = [
    ("tree", {"tree"}),
    ("tree+splitk", {"tree", "splitk"}),
    ("tree+lk", {"tree", "lk"}),
    ("tree+lk+splitk", {"tree", "lk", "splitk"}),
]
STACKS_UPSIDE = [
    ("tree+persist", {"tree", "persist"}),
    ("tree+splitk+persist", {"tree", "splitk", "persist"}),
    ("tree+lk+persist", {"tree", "lk", "persist"}),
    ("tree+lk+splitk+persist", {"tree", "lk", "splitk", "persist"}),
]
CORNERS = ("conservative", "central", "optimistic")


def breakeven_raw_et(levers: set[str], p: dict, target: float = TARGET_OFFICIAL) -> dict:
    """Invert #100 for the RAW tree accept_length E[T]* that makes this lever
    stack hit `target` official-TPS, holding step_time at its topology value.

    Returns the break-even raw E[T]*, the topology step_time, the LK numerator
    factor, and -- as a cross-check -- the #100 forward official_TPS at E[T]=5.207
    that the rescaling form divides into.
    """
    c = compose(levers, p)
    step_time = c["step_time"]
    official_at_tree_et = c["official_tps"]               # #100 forward value @5.207
    lk_factor = p["lk_mult_tree"] if "lk" in levers else 1.0
    # Direct inversion of official = K_cal * (rawET*lk_factor) / step_time * tau.
    raw_et_star = target * step_time / (K_CAL * lk_factor * p["tau"])
    # Exact-rescaling cross-check: rawET* = E_T_TREE * target / official@E_T_TREE.
    raw_et_star_rescale = E_T_TREE * target / official_at_tree_et
    # The tree accept_length that merely TIES the linear chain's official TPS
    # (the honest "worth building" floor -- higher than 3.844 because the tree
    # step is heavier). Same inversion with target = FRONTIER_OFFICIAL.
    beat_linear_et = FRONTIER_OFFICIAL * step_time / (K_CAL * lk_factor * p["tau"])
    return {
        "levers": sorted(levers),
        "step_time": step_time,
        "lk_factor": lk_factor,
        "official_at_tree_et_5p207": official_at_tree_et,
        "breakeven_raw_et": raw_et_star,
        "breakeven_raw_et_rescale_check": raw_et_star_rescale,
        "rescale_abs_err": abs(raw_et_star - raw_et_star_rescale),
        "beat_linear_raw_et": beat_linear_et,
    }


def build_ladder(stacks: list[tuple[str, set[str]]]) -> list[dict]:
    """Break-even ladder: for every stack, the raw E[T]* at each corner plus the
    recovery needed from byteshark's 2.097 and the margin vs the 5.207 ceiling."""
    pts = {c: point(c) for c in CORNERS}
    ladder = []
    for name, levers in stacks:
        row = {"stack": name, "levers": sorted(levers), "n_levers": len(levers)}
        for c in CORNERS:
            be = breakeven_raw_et(levers, pts[c])
            row[c] = {
                "breakeven_raw_et": be["breakeven_raw_et"],
                "step_time": be["step_time"],
                "lk_factor": be["lk_factor"],
                "official_at_5p207": be["official_at_tree_et_5p207"],
                "beat_linear_raw_et": be["beat_linear_raw_et"],
                # recovery the build must claw back from the as-built 2.097.
                "et_recovery_needed_from_2p10": be["breakeven_raw_et"] - BYTESHARK_ASBUILT_ET,
                # margin of safety IF the build fully recovers to the 5.207 ceiling.
                "margin_vs_analytical_ceiling": ANALYTICAL_ET - be["breakeven_raw_et"],
                "rescale_abs_err": be["rescale_abs_err"],
            }
        ladder.append(row)
    return ladder


def margin_curve(stacks: list[tuple[str, set[str]]], corner: str,
                 e_rec_grid: list[float]) -> list[dict]:
    """Margin of safety = recoverable_E[T] - breakeven_E[T]* as a function of a
    PARAMETERIZED recoverable accept_length E_rec (denken #101 lands the real
    band; until then sweep it). For each E_rec, mark which stacks clear 500."""
    p = point(corner)
    be_by_stack = {name: breakeven_raw_et(levers, p)["breakeven_raw_et"]
                   for name, levers in stacks}
    curve = []
    for e_rec in e_rec_grid:
        clears = {name: (e_rec >= be) for name, be in be_by_stack.items()}
        margins = {name: (e_rec - be) for name, be in be_by_stack.items()}
        curve.append({
            "recoverable_et": e_rec,
            "clears_500": clears,
            "margin_of_safety": margins,
            "cheapest_clearing_stack": next(
                (name for name, _ in stacks if clears[name]), None),
            "any_clears": any(clears.values()),
        })
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/tree_et_breakeven_results.json")
    ap.add_argument("--target", type=float, default=TARGET_OFFICIAL)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/tree-et-breakeven")
    ap.add_argument("--wandb-group", default="tree-et-breakeven")
    args = ap.parse_args()

    all_stacks = STACKS_PRIMARY + STACKS_UPSIDE
    ladder_primary = build_ladder(STACKS_PRIMARY)
    ladder_upside = build_ladder(STACKS_UPSIDE)

    # ---- the headline: tree-alone break-even (PRIMARY metric) ----
    tree_alone = next(r for r in ladder_primary if r["stack"] == "tree")
    breakeven_ET_tree_alone = {
        "central": tree_alone["central"]["breakeven_raw_et"],
        "conservative": tree_alone["conservative"]["breakeven_raw_et"],
        "optimistic": tree_alone["optimistic"]["breakeven_raw_et"],
        "band": [tree_alone["optimistic"]["breakeven_raw_et"],
                 tree_alone["conservative"]["breakeven_raw_et"]],
    }
    # the lowest (easiest) break-even reachable by stacking levers, per corner.
    lowest_breakeven = {}
    for c in CORNERS:
        best = min(all_stacks, key=lambda nl: breakeven_raw_et(nl[1], point(c))["breakeven_raw_et"])
        lowest_breakeven[c] = {
            "stack": best[0],
            "breakeven_raw_et": breakeven_raw_et(best[1], point(c))["breakeven_raw_et"],
        }

    # ---- TEST metric: ET_recovery_needed_from_2p10 across the stack ladder ----
    et_recovery_ladder = {
        r["stack"]: {c: r[c]["et_recovery_needed_from_2p10"] for c in CORNERS}
        for r in ladder_primary + ladder_upside
    }

    # ---- margin-of-safety vs a parameterized recoverable E[T] (denken #101) ----
    e_rec_grid = [2.097, 2.5, 3.0, 3.5, 3.844, 4.0, 4.34, 4.45, 4.62, 5.0, 5.207]
    margin_central = margin_curve(all_stacks, "central", e_rec_grid)
    margin_conservative = margin_curve(all_stacks, "conservative", e_rec_grid)
    # the critical recoverable threshold: the smallest E_rec at which ANY stack
    # clears 500 (central) and at which tree-alone clears 500 (central).
    crit_any_central = next(
        (m["recoverable_et"] for m in margin_central if m["any_clears"]), None)
    crit_tree_alone_central = breakeven_ET_tree_alone["central"]

    # ---- the as-built and reference anchors, scored ----
    # where each anchor sits vs the tree-alone central break-even.
    def score_anchor(et: float) -> dict:
        return {
            "accept_length": et,
            "official_tps_tree_alone_central": K_CAL * et / tree_alone["central"]["step_time"] * 1.0,
            "clears_500_tree_alone_central": et >= breakeven_ET_tree_alone["central"],
            "clears_500_full_stack_central": et >= lowest_breakeven["central"]["breakeven_raw_et"],
        }
    anchors = {
        "byteshark_asbuilt_2p097": score_anchor(BYTESHARK_ASBUILT_ET),
        "linear_chain_3p844": score_anchor(LINEAR_CHAIN_ET),
        "beat_linear_official_floor": {
            "accept_length": tree_alone["central"]["beat_linear_raw_et"],
            "note": ("tree accept_length that TIES linear's 481.53 official; "
                     "the honest 'worth building' floor -- ABOVE 3.844 because "
                     "the tree step is heavier"),
        },
        "analytical_5p207": score_anchor(ANALYTICAL_ET),
    }

    # ---- gate on breakeven_ET_tree_alone ----
    be_c = breakeven_ET_tree_alone["central"]
    be_cons = breakeven_ET_tree_alone["conservative"]
    lowest_central = lowest_breakeven["central"]["breakeven_raw_et"]
    levers_pull_below_4 = lowest_central <= 4.0
    if be_c <= 4.0:
        verdict = "GREEN"
        verdict_label = ("tree-robust: break-even <= linear-chain+a-bit; clears "
                         "500 even on a partially-recovered build")
    elif be_c <= 5.0:
        verdict = "AMBER"
        verdict_label = (
            "must-recover-most-of-the-way: break-even in 4.0-5.0; "
            + (f"cheapest stack '{lowest_breakeven['central']['stack']}' pulls "
               f"E[T]* to {lowest_central:.2f}"
               if levers_pull_below_4 else
               f"even the full stack only reaches {lowest_central:.2f} (>4.0) -- "
               "levers shave little; recovery to ~5.2 is the real dependency"))
    else:
        verdict = "RED"
        verdict_label = ("fragile: break-even near the 5.207 ceiling with no "
                         "lever able to lower it materially -- no margin")

    gate = {
        "primary_metric_name": "breakeven_ET_tree_alone",
        "breakeven_ET_tree_alone": breakeven_ET_tree_alone,
        "test_metric_name": "ET_recovery_needed_from_2p10",
        "ET_recovery_needed_from_2p10_tree_alone": {
            c: tree_alone[c]["et_recovery_needed_from_2p10"] for c in CORNERS},
        "ET_recovery_needed_from_2p10_ladder": et_recovery_ladder,
        "lowest_breakeven_by_corner": lowest_breakeven,
        "levers_pull_breakeven_below_4_central": levers_pull_below_4,
        "critical_recoverable_et_central_any_stack": crit_any_central,
        "critical_recoverable_et_central_tree_alone": crit_tree_alone_central,
        "margin_of_safety_if_full_recovery_to_5p207": {
            "tree_alone_central": tree_alone["central"]["margin_vs_analytical_ceiling"],
            "tree_alone_conservative": tree_alone["conservative"]["margin_vs_analytical_ceiling"],
            "full_stack_central": ANALYTICAL_ET - lowest_central,
        },
        "verdict": verdict,
        "verdict_label": verdict_label,
        "rule": ("GREEN=breakeven_ET_tree_alone<=4.0 / AMBER=4.0-5.0 / "
                 "RED=near 5.207 with no lever able to lower it"),
    }

    out = {
        "gate": gate,
        "inversion_model": {
            "formula": "E[T]*_raw = target * step_time / (K_cal * lk_factor * tau) "
                       "= E_T_TREE * target / official_TPS@E_T_TREE",
            "why_linear": ("official_TPS is linear in the accept_length numerator; "
                           "the tree step_time is a TOPOLOGY (M=32) fact, independent "
                           "of how many drafted tokens are accepted"),
            "faithful_to_100": ("reuses lever_composition.compose/point verbatim; "
                                "rescale cross-check error reported per row"),
            "K_cal": K_CAL, "E_T_tree": E_T_TREE, "E_T_linear": E_T_LINEAR,
            "frontier_official": FRONTIER_OFFICIAL, "target_official": args.target,
        },
        "axis_anchors": anchors,
        "byteshark_accept_hist": {
            "hist": BYTESHARK_ACCEPT_HIST,
            "mean_accept_length": (sum(i * n for i, n in enumerate(BYTESHARK_ACCEPT_HIST))
                                   / sum(BYTESHARK_ACCEPT_HIST)),
            "full_accept_frac": BYTESHARK_ACCEPT_HIST[-1] / sum(BYTESHARK_ACCEPT_HIST),
        },
        "breakeven_ladder_primary": ladder_primary,
        "breakeven_ladder_upside": ladder_upside,
        "margin_of_safety_curve_central": margin_central,
        "margin_of_safety_curve_conservative": margin_conservative,
        "lk_mult_tree_band": LK_MULT_TREE,
        "method": ("CPU-only analytic inversion of PR #100's absolute-time slice "
                   "model; holds step_time at its M=32 topology value and solves "
                   "official_TPS=target for the raw tree accept_length. denken #101's "
                   "recoverable-E[T] band parameterized until it lands."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(
            o, np.generic) else o)

    # ----------------------------- console -----------------------------
    print("=" * 84)
    print("TREE E[T] BREAK-EVEN / MARGIN-OF-SAFETY GATE (PR #102) -- inverse of #100")
    print("=" * 84)
    print(f"\nmodel: E[T]* = {args.target:.0f} * step_time / (K_cal*lk*tau), "
          f"K_cal={K_CAL:.3f} | invert official_TPS(E[T])={args.target:.0f} for E[T]*")
    print(f"axis anchors: byteshark as-built 2.097 | linear 3.844 | "
          f"beat-linear-official {tree_alone['central']['beat_linear_raw_et']:.2f} | "
          f"analytical 5.207")

    print("\n[STEP 1] break-even RAW tree accept_length E[T]* (cons .. central .. opt):")
    print(f"  {'stack':24s} {'cons':>7s} {'centr':>7s} {'opt':>7s}   "
          f"{'recov@cons':>10s} {'recov@cent':>10s}")
    for r in ladder_primary:
        print(f"  {r['stack']:24s} {r['conservative']['breakeven_raw_et']:7.3f} "
              f"{r['central']['breakeven_raw_et']:7.3f} "
              f"{r['optimistic']['breakeven_raw_et']:7.3f}   "
              f"{r['conservative']['et_recovery_needed_from_2p10']:10.3f} "
              f"{r['central']['et_recovery_needed_from_2p10']:10.3f}")
    print("  --- upside (persist; ~=0 reclaim at conservative per denken #97) ---")
    for r in ladder_upside:
        print(f"  {r['stack']:24s} {r['conservative']['breakeven_raw_et']:7.3f} "
              f"{r['central']['breakeven_raw_et']:7.3f} "
              f"{r['optimistic']['breakeven_raw_et']:7.3f}   "
              f"{r['conservative']['et_recovery_needed_from_2p10']:10.3f} "
              f"{r['central']['et_recovery_needed_from_2p10']:10.3f}")

    max_err = max(r[c]["rescale_abs_err"] for r in ladder_primary + ladder_upside
                  for c in CORNERS)
    print(f"\n  [consistency] max |direct - rescale| break-even error = {max_err:.2e} "
          f"(must be ~0 -> faithful to #100)")

    print("\n[STEP 2] anchors on the E[T] axis (official @ tree-alone central step):")
    for k, a in anchors.items():
        if "official_tps_tree_alone_central" in a:
            print(f"  {k:28s} E[T]={a['accept_length']:.3f} -> "
                  f"{a['official_tps_tree_alone_central']:6.1f} official | "
                  f"clears500(tree-alone)={a['clears_500_tree_alone_central']} "
                  f"clears500(full-stack)={a['clears_500_full_stack_central']}")
        else:
            print(f"  {k:28s} E[T]={a['accept_length']:.3f}  ({a['note']})")

    print(f"\n[STEP 3] GATE on breakeven_ET_tree_alone:")
    print(f"  breakeven_ET_tree_alone = {be_c:.3f} central "
          f"[{breakeven_ET_tree_alone['optimistic']:.3f} opt, {be_cons:.3f} cons]")
    print(f"  lowest break-even via stacking (central) = {lowest_central:.3f} "
          f"({lowest_breakeven['central']['stack']})  -> pulls below 4.0? {levers_pull_below_4}")
    print(f"  recovery needed from 2.097 (tree-alone central) = "
          f"{tree_alone['central']['et_recovery_needed_from_2p10']:.3f} accept_length")
    print(f"  margin if build fully recovers to 5.207 (tree-alone central) = "
          f"{tree_alone['central']['margin_vs_analytical_ceiling']:+.3f}  "
          f"(conservative {tree_alone['conservative']['margin_vs_analytical_ceiling']:+.3f})")
    print(f"  critical recoverable E[T] (central): tree-alone needs >= "
          f"{crit_tree_alone_central:.3f}; any stack needs >= {crit_any_central}")
    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         job_type="analysis", config={
                             "gate": "tree-et-breakeven",
                             "method": "cpu-analytic-inverse-of-pr100",
                             "frontier_official": FRONTIER_OFFICIAL,
                             "target_official": args.target,
                             "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE,
                             "K_cal": K_CAL,
                             "byteshark_asbuilt_et": BYTESHARK_ASBUILT_ET})
        s = wandb.summary
        s["breakeven_ET_tree_alone"] = be_c
        s["breakeven_ET_tree_alone_conservative"] = be_cons
        s["breakeven_ET_tree_alone_optimistic"] = breakeven_ET_tree_alone["optimistic"]
        s["lowest_breakeven_central"] = lowest_central
        s["lowest_breakeven_central_stack"] = lowest_breakeven["central"]["stack"]
        s["levers_pull_breakeven_below_4_central"] = levers_pull_below_4
        s["ET_recovery_needed_from_2p10_tree_alone_central"] = (
            tree_alone["central"]["et_recovery_needed_from_2p10"])
        s["ET_recovery_needed_from_2p10_full_stack_central"] = (
            lowest_central - BYTESHARK_ASBUILT_ET)
        s["margin_if_full_recovery_tree_alone_central"] = (
            tree_alone["central"]["margin_vs_analytical_ceiling"])
        s["margin_if_full_recovery_tree_alone_conservative"] = (
            tree_alone["conservative"]["margin_vs_analytical_ceiling"])
        s["critical_recoverable_et_central_tree_alone"] = crit_tree_alone_central
        s["beat_linear_official_floor_et"] = tree_alone["central"]["beat_linear_raw_et"]
        s["byteshark_asbuilt_et"] = BYTESHARK_ASBUILT_ET
        s["rescale_max_abs_err"] = max_err
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label

        # break-even ladder table
        lt = wandb.Table(columns=["stack", "n_levers", "breakeven_cons",
                                  "breakeven_central", "breakeven_opt",
                                  "recovery_from_2p10_central",
                                  "margin_vs_5p207_central", "step_time_central"])
        for r in ladder_primary + ladder_upside:
            lt.add_data(r["stack"], r["n_levers"],
                        r["conservative"]["breakeven_raw_et"],
                        r["central"]["breakeven_raw_et"],
                        r["optimistic"]["breakeven_raw_et"],
                        r["central"]["et_recovery_needed_from_2p10"],
                        r["central"]["margin_vs_analytical_ceiling"],
                        r["central"]["step_time"])
        wandb.log({"breakeven_ladder": lt})

        # margin-of-safety vs recoverable E[T] (central) -- the load-bearing curve
        mt = wandb.Table(columns=["recoverable_et", "margin_tree_alone",
                                  "margin_full_stack", "any_stack_clears_500",
                                  "cheapest_clearing_stack"])
        for m in margin_central:
            mt.add_data(m["recoverable_et"], m["margin_of_safety"]["tree"],
                        m["margin_of_safety"]["tree+lk+splitk+persist"],
                        m["any_clears"], m["cheapest_clearing_stack"] or "none")
        wandb.log({"margin_of_safety_curve": mt})

        # anchors table
        at = wandb.Table(columns=["anchor", "accept_length",
                                  "official_tree_alone_central",
                                  "clears_500_tree_alone", "clears_500_full_stack"])
        for k, a in anchors.items():
            if "official_tps_tree_alone_central" in a:
                at.add_data(k, a["accept_length"],
                            a["official_tps_tree_alone_central"],
                            a["clears_500_tree_alone_central"],
                            a["clears_500_full_stack_central"])
        wandb.log({"axis_anchors": at})

        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
