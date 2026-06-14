#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-vs-tree-free CROSSOVER + build-milestone ladder gate (PR #106) -- the
CAPSTONE synthesis of fern #100 (forward composition), fern #102 (inverse
break-even), denken #101 (recoverable band) and denken #105 (tree-free ceiling).

WHY THIS GATE EXISTS
--------------------
fern #100 (MERGED, GREEN) priced the tree's composed official-TPS landscape;
fern #102 (MERGED, AMBER) inverted it for the MINIMUM realized tree accept_length
E[T]* that clears 500 (break-even 4.624 alone). Together they fully characterize
the tree path IN ISOLATION: it clears 500 iff realized E[T] >= 4.624, and it is a
REGRESSION below 4.45 (ties linear) -- and the as-built build delivers only 2.097
(denken #101: a FIXABLE build defect, recoverable band [3.844, 5.207]).

But the tree no longer stands alone. denken #105 is pricing the build-COMPLETE
tree-FREE stack (SplitK #84 + LK #95 + double-quant #104, NO tree). The decision
the fleet now needs is the CROSSOVER:

    At what realized tree-E[T] does the tree path OVERTAKE the best tree-free
    lever stack -- and is continuing the (build-blocked, AMBER) tree build worth
    it given how much TPS the tree-free path already delivers?

Below the crossover the tree-free path wins and the build is not worth
continuing; above it the tree is the win. This gate computes the crossover as a
function of denken #105's ceiling, lays the build-milestone ladder official(E[T])
with the load-bearing ship-gates marked, and converts "the tree is AMBER and
build-blocked" into "here is exactly when it is worth continuing vs when to
pivot to the tree-free path."

THE MODEL (faithful to #100 and #102 by construction)
-----------------------------------------------------
The #100 forward model is official_TPS = K_cal * (E[T] / step_time) * tau, LINEAR
in the accept_length numerator E[T]. For the tree the step_time is an M=32
TOPOLOGY fact -- independent of how many drafted tokens are accepted. So:

  tree_official(E[T]) = K_cal * (E[T] * lk_factor) / step_time_tree * tau
                      = compose(levers, p).official * (E[T] / E_T_TREE)     [Step 1]

i.e. an exact linear rescaling of #100's value at E[T]=5.207. The CROSSOVER with a
tree-free ceiling C is the E[T] that makes tree_official(E[T]) = C:

  E[T]_x(C) = E_T_TREE * C / compose(levers, p).official                    [Step 2]
            = C * step_time / (K_cal * lk_factor * tau)

which is EXACTLY fern #102's break-even inversion with target=C instead of 500.
We REUSE `tree_et_breakeven.breakeven_raw_et(levers, p, target=C)` verbatim, so
the crossover is faithful to #102 (and through it to #100). At C=500 the crossover
collapses to #102's break-even 4.624 -- the synthesis is consistent by
construction.

WHAT WE PUT ON THE AXES
-----------------------
  E[T] axis gates (PR-named): 2.097 (as-built regression ~227) | 3.844 (denken
    #101 structural floor, still a TPS loss ~416) | 4.45 (ties linear 481.53,
    the abort line) | 4.624 (clears 500 alone, fern #102) | 5.207 (analytical
    ceiling, ~563).
  tree-free ceiling C: PARAMETERIZED (denken #105 lands it; sweep 470..565 until
    then). A rough INDEPENDENT bracket from fern's OWN #100 linear-chain levers
    (splitk+lk +/- double-quant; NOT denken's branch) marks the likely region.
  denken #101 recoverable band [3.844, 5.207]: placed against the crossover --
    does a PLAUSIBLY-RECOVERED tree clear it?

PRIMARY metric  tree_vs_treefree_crossover_ET  (E[T]_x as a function of C)
TEST    metric  build_milestone_ladder         (official(E[T]) with gates marked)

GATE (Step 3, central headline; parameterized on denken #105's ceiling C)
-------------------------------------------------------------------------
  GREEN / tree clearly worth it   C < 500 (tree-free CANNOT hit 500 alone) AND
        crossover comfortably < 4.624 -> the tree is on the critical path for 500
        and beats tree-free at a reachable E[T] -> keep the build the #1 priority.
  AMBER / conditional             C >= 500 (denken #105 GREEN) but the tree
        overtakes only if recovery pushes past the crossover (4.624..5.0) -> 500
        is reachable tree-free; the tree is UPSIDE worth continuing only if the
        build recovers most of the way -> name the recovery threshold.
  RED / deprioritize the tree     crossover >= 5.0 (tree barely beats tree-free
        even near its own ceiling) -> the build-blocked tree is not worth the
        build risk; pivot to the tree-free levers + escalate for a fresh
        accept-length lever class. Deep-RED if crossover > 5.207 (the tree can
        NEVER beat tree-free, even at its analytical ceiling).

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

# Reuse the PR #100 forward model + PR #102 inverse verbatim so the crossover is
# guaranteed consistent with the two gates it synthesizes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever_composition import (  # noqa: E402
    E_T_LINEAR,
    E_T_TREE,
    FRONTIER_OFFICIAL,
    K_CAL,
    TARGET_OFFICIAL,
    compose,
    point,
)
from tree_et_breakeven import (  # noqa: E402
    ANALYTICAL_ET,
    BYTESHARK_ASBUILT_ET,
    LINEAR_CHAIN_ET,
    breakeven_raw_et,
)

CORNERS = ("conservative", "central", "optimistic")

# denken #101 recoverable accept_length band [floor, ceiling] (re-measure-pending).
REC_BAND = (E_T_LINEAR, E_T_TREE)   # [3.844, 5.207]

# The two tree paths we cross against the tree-free ceiling:
#   PRIMARY  = tree ALONE (matches the PR's official(E[T]) ladder + the gate
#              anchors 4.45/4.624; the honest "is the bare tree worth the entire
#              tree-free stack" bar).
#   COMPANION= tree + splitk + lk (the build-realistic marginal-tree decision: if
#              the fleet keeps the tree it also builds the compounding levers, so
#              the tree gets their help -> a lower, more tree-favorable crossover).
TREE_PATHS = [("tree_alone", {"tree"}),
              ("tree_full_stack", {"tree", "splitk", "lk"})]

# denken #105 tree-free ceiling: PARAMETERIZED sweep until it lands. Covers the
# PR's 480..540 plus the deep-RED boundary (the tree's own central ceiling ~563).
TREEFREE_CEILING_SWEEP = [470, 480, 490, 500, 507, 510, 511, 520,
                          530, 540, 550, 560, 563, 565]


# ----------------------------------------------------------------------------
# tree official-TPS at an arbitrary realized accept_length (exact #100 rescale).
# ----------------------------------------------------------------------------
def tree_official_at_et(et: float, levers: set[str], p: dict) -> float:
    """official-TPS of a tree lever-stack at a GIVEN raw tree accept_length `et`,
    holding step_time at its M=32 topology value. Exact linear rescaling of #100's
    compose value at E[T]=5.207 -> faithful by construction (lk folded in)."""
    return compose(levers, p)["official_tps"] * (et / E_T_TREE)


def crossover_et(ceiling: float, levers: set[str], p: dict) -> float:
    """The realized tree accept_length E[T]_x at which this tree stack's official
    TPS equals the tree-free `ceiling`. Identical to fern #102's break-even with
    target=ceiling (REUSED verbatim) -> the crossover IS the generalized #102."""
    return breakeven_raw_et(levers, p, target=ceiling)["breakeven_raw_et"]


# ----------------------------------------------------------------------------
# Rough INDEPENDENT tree-free bracket from fern's OWN merged #100 linear-chain
# levers (NOT denken's unmerged branch). Marks the likely region pending #105.
# ----------------------------------------------------------------------------
# wirbel #104 double-quant verify-GEMM scales band (+0.4..1.1%); central ~0.75%.
DOUBLE_QUANT_PCT = {"conservative": 0.004, "central": 0.0075, "optimistic": 0.011}


def treefree_bracket_own_model() -> dict:
    """tree-FREE official-TPS (no tree) from #100's splitk+lk on the linear chain,
    +/- double-quant #104. A SANITY bracket for denken #105's authoritative number
    (which also folds in double-quant properly). persist ~0 at the conservative
    corner (denken #97: only 2.17% reclaimable idle) -> excluded from the floor."""
    base = {c: compose({"splitk", "lk"}, point(c))["official_tps"] for c in CORNERS}
    with_dq = {c: base[c] * (1.0 + DOUBLE_QUANT_PCT[c]) for c in CORNERS}
    return {"splitk_lk": base, "splitk_lk_double_quant": with_dq,
            "note": ("fern's OWN #100 linear-chain levers; rough bracket only -- "
                     "denken #105 is the authoritative tree-free ceiling lane")}


# ----------------------------------------------------------------------------
# STEP 1 -- the build-milestone ladder official(E[T]).
# ----------------------------------------------------------------------------
def build_milestone_ladder() -> dict:
    """official_TPS as a function of realized tree E[T] over 2.097..5.207 at the
    three corners (tree ALONE -- the PR's primary curve), with the load-bearing
    ship-gates marked. Returns a machine-readable ladder the build team can use to
    set intermediate ship-gates."""
    pts = {c: point(c) for c in CORNERS}

    # dense curve for plotting / interpolation
    grid = sorted(set(np.round(np.arange(BYTESHARK_ASBUILT_ET, E_T_TREE + 1e-9, 0.1), 4))
                  | {BYTESHARK_ASBUILT_ET, E_T_LINEAR, E_T_TREE})
    curve = []
    for et in grid:
        curve.append({"E_T": float(et),
                      **{f"{c}_official": tree_official_at_et(et, {"tree"}, pts[c])
                         for c in CORNERS}})

    # FIXED-E[T] gates (official varies by corner).
    fixed_gates = [
        ("asbuilt", BYTESHARK_ASBUILT_ET,
         "byteshark as-built tok/step -- accept-collapse REGRESSION"),
        ("linear_floor", E_T_LINEAR,
         "denken #101 structural floor (linear-chain accept_length) -- still a TPS LOSS"),
        ("analytical_ceiling", E_T_TREE,
         "analytical ceiling (de-risked, fern #92)"),
    ]
    # THRESHOLD-E[T] gates (E[T] varies by corner; official is fixed by definition).
    #   beat_linear  = E[T] at which tree official ties the linear frontier 481.53.
    #   breakeven500 = E[T] at which tree official hits the 500 target (fern #102).
    ladder_gates = {}
    for key, et, desc in fixed_gates:
        ladder_gates[key] = {
            "kind": "fixed_E_T", "E_T": et, "desc": desc,
            "official": {c: tree_official_at_et(et, {"tree"}, pts[c]) for c in CORNERS},
        }
    ladder_gates["beat_linear"] = {
        "kind": "threshold_official", "target_official": FRONTIER_OFFICIAL,
        "desc": "ties linear 481.53 official -- the ABORT line (below it the tree is a regression)",
        "E_T": {c: crossover_et(FRONTIER_OFFICIAL, {"tree"}, pts[c]) for c in CORNERS},
    }
    ladder_gates["breakeven_500"] = {
        "kind": "threshold_official", "target_official": TARGET_OFFICIAL,
        "desc": "clears the 500 target ALONE (fern #102 break-even)",
        "E_T": {c: crossover_et(TARGET_OFFICIAL, {"tree"}, pts[c]) for c in CORNERS},
    }

    # ordered ship-gate ladder (central E[T], ascending) for the build team.
    ship_gates = [
        {"order": 1, "gate": "beat-linear", "E_T_central": ladder_gates["beat_linear"]["E_T"]["central"],
         "rule": "first ship-gate: realized E[T] >= ~4.45 or the tree is a NET REGRESSION vs linear 481.53"},
        {"order": 2, "gate": "clear-500", "E_T_central": ladder_gates["breakeven_500"]["E_T"]["central"],
         "rule": "target ship-gate: realized E[T] >= ~4.62 clears 500 with the bare tree (fern #102)"},
        {"order": 3, "gate": "analytical-ceiling", "E_T_central": E_T_TREE,
         "rule": "stretch: realized E[T] -> 5.207 gives ~563 central (the de-risked ceiling)"},
    ]
    return {"curve_tree_alone": curve, "gates": ladder_gates, "ship_gates": ship_gates}


# ----------------------------------------------------------------------------
# STEP 2 -- the tree-vs-tree-free crossover as a function of denken #105's ceiling.
# ----------------------------------------------------------------------------
def crossover_curve() -> dict:
    """E[T]_x(C) for the PRIMARY (tree-alone) and COMPANION (tree+splitk+lk) paths,
    swept over the parameterized tree-free ceiling C, at the three corners. Places
    denken #101's recoverable band [3.844, 5.207] against each crossover."""
    pts = {c: point(c) for c in CORNERS}
    be500 = {name: crossover_et(TARGET_OFFICIAL, lv, pts["central"])
             for name, lv in TREE_PATHS}

    rows = []
    for C in TREEFREE_CEILING_SWEEP:
        row = {"treefree_ceiling": float(C),
               "treefree_clears_500": C >= TARGET_OFFICIAL}
        for name, lv in TREE_PATHS:
            xc = {c: crossover_et(C, lv, pts[c]) for c in CORNERS}
            row[name] = {
                "crossover_E_T": xc,
                # is the crossover reachable inside denken #101's band [3.844,5.207]?
                "reachable_within_rec_band_central": xc["central"] <= REC_BAND[1],
                "above_500_breakeven_central": xc["central"] > be500[name],
                # the tree's official at its band ceiling minus the tree-free C
                # (>0 => a fully-recovered tree beats this tree-free ceiling).
                "margin_at_band_ceiling_central":
                    tree_official_at_et(REC_BAND[1], lv, pts["central"]) - C,
            }
        rows.append(row)

    # the slope d(crossover)/dC (central, tree-alone) + the three threshold Cs.
    slope_central = (crossover_et(600.0, {"tree"}, pts["central"])
                     - crossover_et(500.0, {"tree"}, pts["central"])) / 100.0
    off_tree_central = compose({"tree"}, pts["central"])["official_tps"]  # @5.207

    def ceiling_for_crossover(target_xc: float) -> float:
        # invert E[T]_x(C) = E_T_TREE*C/off  ->  C = target_xc*off/E_T_TREE
        return target_xc * off_tree_central / E_T_TREE

    thresholds = {
        "C_treefree_just_clears_500": TARGET_OFFICIAL,
        "C_crossover_equals_5p0_central": ceiling_for_crossover(5.0),
        "C_crossover_equals_ceiling_5p207_central": ceiling_for_crossover(E_T_TREE),
        "interpretation": {
            "below_500": "tree-free CANNOT hit 500 -> tree on the critical path (GREEN region)",
            "500_to_C5p0": "tree-free clears 500; tree is upside, overtakes only if E[T]>=crossover in [4.624,5.0] (AMBER region)",
            "C5p0_to_C5p207": "crossover in [5.0,5.207] -> tree barely beats tree-free even near its ceiling (RED region)",
            "above_C5p207": "crossover > 5.207 -> tree can NEVER beat tree-free, even at its analytical ceiling (deep-RED)",
        },
    }

    # corner-MATCHED robustness: if denken #105's C is itself banded and we compare
    # corner-for-corner (the same optimism lifts BOTH the tree and tree-free), the
    # crossover is far more stable than the exogenous-scalar sweep suggests.
    own = treefree_bracket_own_model()["splitk_lk_double_quant"]
    corner_matched = {c: crossover_et(own[c], {"tree"}, pts[c]) for c in CORNERS}

    return {"rows": rows, "be500_by_path": be500,
            "slope_dcrossover_dC_central": slope_central,
            "thresholds_central": thresholds,
            "corner_matched_crossover_tree_alone": corner_matched}


# ----------------------------------------------------------------------------
# STEP 3 -- the gate (parameterized on denken #105's ceiling C).
# ----------------------------------------------------------------------------
def gate_for_ceiling(C: float, pts: dict) -> dict:
    """The continue-vs-pivot verdict at a given tree-free ceiling C (central)."""
    xc = crossover_et(C, {"tree"}, pts["central"])
    xc_stack = crossover_et(C, {"tree", "splitk", "lk"}, pts["central"])
    be500 = crossover_et(TARGET_OFFICIAL, {"tree"}, pts["central"])
    if C < TARGET_OFFICIAL:
        verdict, label = "GREEN", (
            f"tree on the CRITICAL PATH for 500 (tree-free central {C:.0f} < 500); "
            f"crossover {xc:.3f} < {be500:.3f} -> keep the tree build the #1 priority")
    elif xc < 5.0:
        verdict, label = "AMBER", (
            f"tree-free clears 500; the tree is UPSIDE -- it overtakes only if the "
            f"build recovers E[T] >= {xc:.3f} (alone) / {xc_stack:.3f} (with splitk+lk), "
            f"i.e. {(xc - REC_BAND[0]) / (REC_BAND[1] - REC_BAND[0]) * 100:.0f}% of the "
            f"way up denken #101's band -> continue only if recovery clears that")
    elif xc <= REC_BAND[1]:
        verdict, label = "RED", (
            f"crossover {xc:.3f} >= 5.0: the tree barely beats tree-free even near "
            f"its own ceiling -> deprioritize the build-blocked tree, pivot to the "
            f"tree-free levers")
    else:
        verdict, label = "RED", (
            f"crossover {xc:.3f} > 5.207: the tree can NEVER beat tree-free, even at "
            f"its analytical ceiling -> pivot to tree-free + escalate for a fresh "
            f"accept-length lever class")
    return {"treefree_ceiling": C, "crossover_tree_alone": xc,
            "crossover_tree_full_stack": xc_stack, "verdict": verdict,
            "verdict_label": label}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/tree_vs_treefree_crossover_results.json")
    ap.add_argument("--treefree-ceiling", type=float, default=None,
                    help="consume denken #105's landed ceiling for the HEADLINE verdict")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/tree-vs-treefree-crossover")
    ap.add_argument("--wandb-group", default="tree-vs-treefree-crossover")
    args = ap.parse_args()

    pts = {c: point(c) for c in CORNERS}

    ladder = build_milestone_ladder()
    crossover = crossover_curve()
    own_bracket = treefree_bracket_own_model()

    # ---- headline verdict ----
    # If denken #105 has landed (--treefree-ceiling), consume it; else use fern's
    # own central bracket as the LIKELY ceiling and flag the verdict pending #105.
    if args.treefree_ceiling is not None:
        likely_C = args.treefree_ceiling
        source = "denken #105 (consumed)"
    else:
        likely_C = own_bracket["splitk_lk_double_quant"]["central"]
        source = "fern's own #100 bracket (PENDING denken #105)"
    headline = gate_for_ceiling(likely_C, pts)
    headline["ceiling_source"] = source

    # gate across the whole sweep (the decision surface).
    gate_surface = [gate_for_ceiling(float(C), pts) for C in TREEFREE_CEILING_SWEEP]

    # ---- PRIMARY metric: tree_vs_treefree_crossover_ET (as a function of C) ----
    primary = {
        "name": "tree_vs_treefree_crossover_ET",
        "definition": "realized tree E[T] at which tree_official(E[T]) = tree_free_ceiling",
        "at_C_500_tree_alone_central": crossover["be500_by_path"]["tree_alone"],
        "at_likely_ceiling_tree_alone_central": headline["crossover_tree_alone"],
        "likely_ceiling": likely_C,
        "slope_dE_T_per_TPS_central": crossover["slope_dcrossover_dC_central"],
        "thresholds": crossover["thresholds_central"],
        "corner_matched_robust_band": crossover["corner_matched_crossover_tree_alone"],
    }

    out = {
        "gate": {
            "primary_metric_name": "tree_vs_treefree_crossover_ET",
            "test_metric_name": "build_milestone_ladder",
            "headline_verdict": headline["verdict"],
            "headline_verdict_label": headline["verdict_label"],
            "headline_ceiling": likely_C,
            "headline_ceiling_source": source,
            "primary_metric": primary,
            "rule": ("GREEN=C<500 (tree critical path) / AMBER=C>=500 & crossover<5.0 "
                     "(tree upside, name recovery threshold) / RED=crossover>=5.0 "
                     "(tree barely beats tree-free; pivot)"),
        },
        "step1_build_milestone_ladder": ladder,
        "step2_crossover": crossover,
        "step3_gate_surface": gate_surface,
        "treefree_bracket_own_model": own_bracket,
        "anchors": {
            "frontier_official": FRONTIER_OFFICIAL, "target_official": TARGET_OFFICIAL,
            "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE, "K_cal": K_CAL,
            "byteshark_asbuilt_et": BYTESHARK_ASBUILT_ET, "rec_band": list(REC_BAND),
            "fern_102_breakeven_500_central": crossover["be500_by_path"]["tree_alone"],
        },
        "method": ("CPU-only analytic synthesis of fern #100 (forward) + fern #102 "
                   "(inverse, REUSED verbatim) + denken #101 (recoverable band) + "
                   "denken #105 (tree-free ceiling, PARAMETERIZED). The crossover is "
                   "#102's break-even generalized from target=500 to target=C; at "
                   "C=500 it collapses to 4.624 by construction. No GPU, no served "
                   "change -> greedy identity untouched."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(
            o, np.generic) else o)

    # ----------------------------- console -----------------------------
    print("=" * 88)
    print("TREE-vs-TREE-FREE CROSSOVER + BUILD-MILESTONE LADDER (PR #106) -- capstone synthesis")
    print("=" * 88)
    print(f"\nmodel: tree_official(E[T]) = K_cal*(E[T]/step)*tau (LINEAR in E[T]); "
          f"crossover E[T]_x(C) = #102 break-even @ target=C  (K_cal={K_CAL:.3f})")

    print("\n[STEP 1] build-milestone ladder official(E[T]) -- tree ALONE (cons / central / opt):")
    for key in ("asbuilt", "linear_floor", "beat_linear", "breakeven_500", "analytical_ceiling"):
        g = ladder["gates"][key]
        if g["kind"] == "fixed_E_T":
            o = g["official"]
            print(f"  E[T]={g['E_T']:.3f}  {key:20s} -> {o['conservative']:6.1f} / "
                  f"{o['central']:6.1f} / {o['optimistic']:6.1f}   ({g['desc']})")
        else:
            e = g["E_T"]
            print(f"  official={g['target_official']:.2f}  {key:20s} -> E[T]= "
                  f"{e['conservative']:.3f} / {e['central']:.3f} / {e['optimistic']:.3f}   ({g['desc']})")
    print("  ship-gates (build team): "
          + " | ".join(f"{s['gate']} E[T]>={s['E_T_central']:.2f}" for s in ladder["ship_gates"]))

    print("\n[STEP 2] crossover E[T]_x(C) vs tree-free ceiling C  (tree-alone central):")
    print(f"  {'C':>6s}  {'xover(alone)':>12s}  {'xover(+sk+lk)':>13s}  treefree>=500?  reach<=5.207?")
    for r in crossover["rows"]:
        a = r["tree_alone"]; s = r["tree_full_stack"]
        print(f"  {r['treefree_ceiling']:6.1f}  {a['crossover_E_T']['central']:12.3f}  "
              f"{s['crossover_E_T']['central']:13.3f}  "
              f"{str(r['treefree_clears_500']):>13s}  {a['reachable_within_rec_band_central']}")
    th = crossover["thresholds_central"]
    print(f"\n  thresholds (central): tree-free needs the tree below C={th['C_treefree_just_clears_500']:.0f} | "
          f"crossover=5.0 at C={th['C_crossover_equals_5p0_central']:.1f} | "
          f"tree-NEVER-wins above C={th['C_crossover_equals_ceiling_5p207_central']:.1f}")
    cm = crossover["corner_matched_crossover_tree_alone"]
    print(f"  corner-MATCHED crossover (robust; same optimism lifts both): "
          f"{cm['conservative']:.3f} / {cm['central']:.3f} / {cm['optimistic']:.3f}")

    print("\n[fern's own rough tree-free bracket -- PENDING denken #105]")
    b = own_bracket["splitk_lk"]; bd = own_bracket["splitk_lk_double_quant"]
    print(f"  splitk+lk:            cons {b['conservative']:.1f} / central {b['central']:.1f} / opt {b['optimistic']:.1f}")
    print(f"  +double-quant #104:   cons {bd['conservative']:.1f} / central {bd['central']:.1f} / opt {bd['optimistic']:.1f}")

    print(f"\n[STEP 3 / VERDICT] headline ceiling C={likely_C:.1f}  ({source})")
    print(f"  {headline['verdict']} -- {headline['verdict_label']}")
    print("\n  gate surface across the sweep:")
    for g in gate_surface:
        print(f"    C={g['treefree_ceiling']:6.1f} -> {g['verdict']:5s}  "
              f"(crossover alone {g['crossover_tree_alone']:.3f})")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         job_type="analysis", config={
                             "gate": "tree-vs-treefree-crossover",
                             "method": "cpu-analytic-synthesis-100-102-101-105",
                             "frontier_official": FRONTIER_OFFICIAL,
                             "target_official": TARGET_OFFICIAL,
                             "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE,
                             "K_cal": K_CAL, "rec_band": list(REC_BAND),
                             "headline_ceiling": likely_C,
                             "headline_ceiling_source": source})
        s = wandb.summary
        s["tree_vs_treefree_crossover_ET"] = primary["at_likely_ceiling_tree_alone_central"]
        s["tree_vs_treefree_crossover_ET_at_C500"] = primary["at_C_500_tree_alone_central"]
        s["crossover_slope_dE_T_per_TPS_central"] = primary["slope_dE_T_per_TPS_central"]
        s["C_treefree_just_clears_500"] = th["C_treefree_just_clears_500"]
        s["C_crossover_equals_5p0_central"] = th["C_crossover_equals_5p0_central"]
        s["C_tree_never_wins_above_central"] = th["C_crossover_equals_ceiling_5p207_central"]
        s["corner_matched_crossover_central"] = cm["central"]
        s["treefree_ceiling_own_bracket_central"] = bd["central"]
        s["headline_verdict"] = headline["verdict"]
        s["headline_verdict_label"] = headline["verdict_label"]
        # break-even-500 cross-check (must equal fern #102's 4.624).
        s["crossover_at_C500_equals_fern102_breakeven"] = primary["at_C_500_tree_alone_central"]

        # Step 1 ladder table
        lt = wandb.Table(columns=["E_T", "official_conservative", "official_central",
                                  "official_optimistic"])
        for c in ladder["curve_tree_alone"]:
            lt.add_data(c["E_T"], c["conservative_official"], c["central_official"],
                        c["optimistic_official"])
        wandb.log({"build_milestone_ladder": lt})

        # Step 2 crossover table
        ct = wandb.Table(columns=["treefree_ceiling", "crossover_tree_alone_central",
                                  "crossover_tree_full_stack_central",
                                  "treefree_clears_500", "reachable_within_band",
                                  "verdict"])
        for r, g in zip(crossover["rows"], gate_surface):
            ct.add_data(r["treefree_ceiling"],
                        r["tree_alone"]["crossover_E_T"]["central"],
                        r["tree_full_stack"]["crossover_E_T"]["central"],
                        r["treefree_clears_500"],
                        r["tree_alone"]["reachable_within_rec_band_central"],
                        g["verdict"])
        wandb.log({"crossover_vs_ceiling": ct})

        # gate anchors table (the named milestone gates)
        gt = wandb.Table(columns=["gate", "kind", "E_T_or_target",
                                  "official_or_E_T_central", "desc"])
        for key in ("asbuilt", "linear_floor", "beat_linear", "breakeven_500",
                    "analytical_ceiling"):
            g = ladder["gates"][key]
            if g["kind"] == "fixed_E_T":
                gt.add_data(key, g["kind"], g["E_T"], g["official"]["central"], g["desc"])
            else:
                gt.add_data(key, g["kind"], g["target_official"], g["E_T"]["central"], g["desc"])
        wandb.log({"milestone_gates": gt})

        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
