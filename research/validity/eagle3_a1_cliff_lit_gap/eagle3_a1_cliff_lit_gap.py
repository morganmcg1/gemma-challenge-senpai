#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #342 -- a_1-cliff lit-gap: price the a_1-lift that resolves #294's lit straddle.

WHAT THIS ANSWERS
-----------------
kanna #294 (j0ss47bv, MERGED) found the honest twist in the EAGLE-3 Phase-1 gate:
the published EAGLE-3 lift TRANSPLANTED onto our deployed a_1=0.7293 cliff projects
`eagle3_lit_projected_et = 4.6901 < 4.9029` (eagle3_lit_clears_target=False) BUT
BRACKETS the target (4.6901 hold-a_1 < 4.9029 < ~5.5 direct chain). #294 flagged that
the ENTIRE lit shortfall is driven by holding a_1 = 0.7293 fixed -- and that pricing
the exact a_1-lift needed turns the straddle into a second measurable sub-trigger.

This card prices that second sub-trigger. It builds the lit projection AS A FUNCTION
of a_1 (hold the EAGLE-3 flat deep accept a_lit on positions 2..7, sweep a_1 upward
from the 0.7293 cliff), solves for the a_1* at which the lit-anchored projection
reaches 4.9029, and compares a_1* to EAGLE-3's achievable position-1 acceptance on
DENSE un-quantized models -- the upper bound on what a real EAGLE-3 retrain can deliver.

THE TWO SUB-TRIGGERS the human's #319 decision now has
-----------------------------------------------------
  (i)  CHEAP -- the a_2 >= 0.8342 Phase-1 proxy (#294). Measurable in a 2h frozen-backbone
       adaptation; 1:1 local->official (lawine #288 tau_acc=1.0).
  (ii) HARD -- the a_1-lift a_1 >= a_1* (THIS card). The base-model cliff-break that
       determines whether the lit projection clears AT ALL. #289 found a_1 is a base-model
       predictability property (built_raise_requires_nonlinear_drafter=True), so a_1 is
       likely shifted only by a full retrain, not a 2h adaptation.

THE CONSTRUCTION (identical to #294's lit_reality_check, generalized over a_1)
-----------------------------------------------------------------------------
  a_lit  = fit_flat_accept(L_chain)      # flat deep accept reproducing EAGLE-3 chain length
  lit_projected_et(a_1) = E[T]([a_1, a_lit, a_lit, ..., a_lit])
                        = 1 + a_1 * sum_{i=0..K-1} a_lit^i      (LINEAR + monotone in a_1)
  a_1* solves lit_projected_et(a_1*) = 4.9029                   (unique monotone root)

CPU analytic over BANKED W&B numbers (kanna #294 j0ss47bv a_lit + lit anchor + a_star;
kanna #289 fi34s269 a_k profile; wirbel #290 ub3kpsso target 4.9029) + the published
EAGLE-3 per-position acceptance anchor (literature import, cited). NO new GPU, NO model
forward, NO Phase-1 run, NO served-file change, NO HF Job, NO submission. NOT open2,
NOT a build, NOT a launch. BASELINE stays 481.53; this leg adds 0 TPS.
a1_cliff_card_is_cpu_analytic = True.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../target

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED).
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
OFFICIAL = 481.53                  # PR #52 official frontier TPS (2x9fm2zx); this leg adds 0
CEILING_LAMBDA1 = 520.95           # lambda=1 step-side ceiling (headline value)
K_CAL = 125.268                    # kanna #217/#269: official = K_cal * E[T]  (== 481.53/3.844)
E_T_ANCHOR = 3.844                 # kanna #217 deployed linear served E[T] (vgovdrjc)
K_SPEC = 7                         # num_speculative_tokens (linear MTP depth K=7)
E_T_MAX = float(K_SPEC + 1)        # 8.0 -- full-acceptance ceiling
LINEAR_CAP = 3.8445                # denken #119: LINEAR drafter E[T] cap at perfect capacity
TARGET_BANKED = 4.9029             # wirbel #290 ub3kpsso: step-banked BUILT-raise target (HEADLINE)
PRIVATE_VERIFIED = 460.85          # private-verified reference (PR baseline)

# ---- kanna #289 fi34s269: deployed per-position conditional acceptance (full precision) ----
A_BASE = [
    0.7292532942898975,   # a_1 -- THE CLIFF (the swept variable here; a base-model property)
    0.759556697719242,    # a_2 -- the #294 Phase-1 a_2 GO-gate handle
    0.7929794882639035,   # a_3
    0.8228,               # a_4
    0.8348727920920435,   # a_5
    0.8357919254658385,   # a_6
    0.8464932652113331,   # a_7
]
A1_DEPLOYED = A_BASE[0]            # 0.7292532942898975 -- the deployed a_1 cliff (#289)
E_T_DECOMP_289 = 3.851185944363104           # #289 E[T] reproduced by the survival product

# ---- kanna #294 j0ss47bv: lit straddle anchors (BANKED -- the numbers we RESOLVE here) ----
A_STAR_J2 = 0.9088740395297805    # #294: uniform j>=2 accept that clears 4.9029 (HOLDING a_1)
LIT_PROJ_HOLDA1_CENTRAL = 4.690056968034645   # #294: lit projected E[T] on our base (hold a_1) -- 4.6901
LIT_PROJ_HOLDA1_LOW = 4.1515952532910525      # #294: low-anchor lit projected E[T] (hold a_1)
EAGLE3_LIT_ACCEPT_CENTRAL = 0.8893195559667377  # #294: flat deep accept a_lit @ chain-length 5.5
EAGLE3_LIT_ACCEPT_LOW = 0.8330104751560863      # #294: flat deep accept a_lit @ chain-length 4.6

# ---- lawine #288 i1e5054m: local->official transfer (a_2 proxy is 1:1) ----
TAU_ACC = 1.0                      # local accept == official accept (clock-invariant probability)
PHASE1_A2_GO = 0.8342153686245113  # #294: the cheap a_2 Phase-1 GO threshold (f=0.5)

# --------------------------------------------------------------------------- #
# EAGLE-3 LITERATURE ANCHOR (import, cite). Chain-equivalent accepted length -> flat
# deep accept a_lit (the SAME map #294 used; re-derived + asserted == banked).
#
# Provenance (literature pass; see PR body "Research" section + #294 EAGLE3_LIT_NOTE):
#   EAGLE-3 (Li et al. 2025, arXiv:2503.01840 Table 1): average accepted length tau~6.2
#   (tree, T=0, LLaMA-3.1-8B). Our lane is a LINEAR depth-7 CHAIN; EAGLE-1 (arXiv:2401.15077
#   Table 5) tree-over-chain premium 0.68 -> chain-equivalent ~5.5 central / ~4.6 low.
#   EAGLE-3 Fig 7 reports a NEAR-FLAT per-depth acceptance, so the flat deep accept a_lit
#   reproducing the chain length doubles as EAGLE-3's per-position acceptance estimate.
# --------------------------------------------------------------------------- #
EAGLE3_LIT_L_CHAIN_CENTRAL = 5.5   # chain-equivalent accepted length (central)
EAGLE3_LIT_L_CHAIN_LOW = 4.6       # chain-equivalent accepted length (conservative)

# EAGLE-3 ACHIEVABLE DENSE-MODEL position-1 acceptance BAND (the FEASIBILITY CEILING, step 3).
# Literature pass (PR body "Research"): EAGLE-3 0-alpha ~0.91 central, range 0.89-0.93. No paper
# prints a verbatim 0-alpha digit (the Fig-7 y-axis is graphic-embedded, not text-extractable);
# the band is INFERRED from (a) tau=6.2 tree, depth-8 (arXiv:2503.01840 Table 1) back-calculated to
# a flat chain-equivalent alpha ~0.89-0.91 after a tree->chain discount, and (b) EAGLE-3 being
# "significantly higher" than EAGLE-1 at 0-alpha (arXiv:2503.01840 sec5 / Fig 7) over EAGLE-1's
# PUBLISHED 0-alpha 0.74-0.85 (arXiv:2401.15077 sec5.1) -> floor >= 0.89. EAGLE-3's near-flat
# profile means position-1 ~ deep accept (first-order), so #294's chain-equivalent flat accept
# a_lit=0.8893 doubles as the CONSERVATIVE (published-floor) dense ceiling.
EAGLE3_DENSE_A1_CONSERVATIVE = EAGLE3_LIT_ACCEPT_CENTRAL   # 0.8893 -- chain-equiv flat accept ~ lit floor 0.89
EAGLE3_DENSE_A1_CENTRAL = 0.91                             # EAGLE-3 0-alpha central estimate (inferred)
EAGLE3_DENSE_A1_OPTIMISTIC = 0.93                          # EAGLE-3 0-alpha upper range (inferred)
EAGLE3_LIT_NOTE = (
    "EAGLE-3 (arXiv:2503.01840 Table 1) accepted-length tau~6.2 (tree, T=0, LLaMA-3.1-8B); "
    "chain-equivalent ~5.5 central / ~4.6 low after the EAGLE-1 (arXiv:2401.15077 Table 5) "
    "tree-over-chain premium 0.68 -> flat deep accept a_lit (Fig-7 near-flat profile). For the "
    "feasibility ceiling, EAGLE-3's dense-model position-1 acceptance (0-alpha) is ~0.91 central "
    "(range 0.89-0.93): no verbatim digit is published (Fig-7 y-axis is graphic-embedded), so the "
    "band is inferred from the tau=6.2 tree->chain back-calc (~0.89-0.91) and 'significantly "
    "higher than EAGLE-1' (arXiv:2503.01840 sec5/Fig7) over EAGLE-1's published 0-alpha 0.74-0.85 "
    "(arXiv:2401.15077 sec5.1). The near-flat profile (pos-1 ~ deep accept) makes a_lit=0.8893 the "
    "CONSERVATIVE published-floor ceiling. int4 + reasoning/STEM OOD is unpriced extra downside (#294)."
)

OUT_PATH = HERE / "eagle3_a1_cliff_lit_gap_results.json"
A1_SWEEP_GRID = [round(0.70 + 0.02 * i, 4) for i in range(16)]   # 0.70..1.00 for monotone/plot


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv) -- IDENTICAL to #294
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-14, max_it: int = 400) -> float:
    """Robust bracketed root find; raises if [lo,hi] does not bracket a root."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(f"bisect: no sign change on [{lo},{hi}] -> {flo},{fhi}")
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def et_of_profile(a: list[float]) -> float:
    """E[T] = 1 + sum_{k=1..K} prod_{j<=k} a_j (survival sum of committed length, #289)."""
    s, prod = 0.0, 1.0
    for ak in a:
        prod *= ak
        s += prod
    return 1.0 + s


def tps_from_et(et: float) -> float:
    """official = K_cal * E[T] (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


def deep_uniform_profile(a1: float, a_deep: float, k: int = K_SPEC) -> list[float]:
    """[a1, a_deep, a_deep, ...] -- vary a_1, set j>=2 to a common deep accept (#294)."""
    return [a1] + [a_deep] * (k - 1)


def fit_flat_accept(length: float) -> float:
    """Common conditional acceptance a s.t. a FLAT depth-K chain reproduces accepted-length:
    1 + sum_{k=1..K} a^k = length (EAGLE-3's near-flat Fig-7 profile). IDENTICAL to #294."""
    length = min(max(length, 1.0 + 1e-9), E_T_MAX - 1e-9)
    return bisect(lambda a: et_of_profile([a] * K_SPEC) - length, 1e-9, 1.0 - 1e-12)


def deep_survival_sum(a_lit: float, k: int = K_SPEC) -> float:
    """S = sum_{i=0..K-1} a_lit^i  (lit_projected_et(a1) = 1 + a1 * S; closed form)."""
    return sum(a_lit ** i for i in range(k))


# --------------------------------------------------------------------------- #
# THE lit projection as a function of a_1 (hold a_lit on positions 2..7)
# --------------------------------------------------------------------------- #
def lit_projected_et(a1: float, a_lit: float) -> float:
    """EAGLE-3 lit projection on our base with a_1 swept and positions 2..7 = a_lit.
    == #294's deep_uniform_profile transplant, generalized to vary a_1."""
    return et_of_profile(deep_uniform_profile(a1, a_lit))


def solve_a1_star(a_lit: float, target: float = TARGET_BANKED) -> float:
    """Solve lit_projected_et(a1*, a_lit) = target for a1*. E[T] is monotone increasing in
    a_1 (linear: 1 + a1*S, S>0) -> unique root. Monotone bisection on a1 in (0, 1)."""
    return bisect(lambda a1: lit_projected_et(a1, a_lit) - target, 1e-9, 1.0 - 1e-12)


# --------------------------------------------------------------------------- #
# Step 1 -- define + validate the a_1-swept lit projection (reproduce #294 at a_1=cliff)
# --------------------------------------------------------------------------- #
def step1_define_and_validate() -> dict[str, Any]:
    # Re-derive a_lit from the EAGLE-3 chain length EXACTLY as #294 did, then assert it
    # matches the banked #294 a_lit (construction-identity to the merged card).
    a_lit_c = fit_flat_accept(EAGLE3_LIT_L_CHAIN_CENTRAL)
    a_lit_lo = fit_flat_accept(EAGLE3_LIT_L_CHAIN_LOW)
    et_hold_c = lit_projected_et(A1_DEPLOYED, a_lit_c)
    et_hold_lo = lit_projected_et(A1_DEPLOYED, a_lit_lo)
    sweep = [{"a1": a1,
              "lit_et_central": lit_projected_et(a1, a_lit_c),
              "lit_et_low": lit_projected_et(a1, a_lit_lo),
              "lit_tps_central": tps_from_et(lit_projected_et(a1, a_lit_c))}
             for a1 in A1_SWEEP_GRID]
    monotone_c = all(sweep[i]["lit_et_central"] < sweep[i + 1]["lit_et_central"] + 1e-15
                     for i in range(len(sweep) - 1))
    monotone_lo = all(sweep[i]["lit_et_low"] < sweep[i + 1]["lit_et_low"] + 1e-15
                      for i in range(len(sweep) - 1))
    return {
        "a_lit_central_rederived": a_lit_c,
        "a_lit_low_rederived": a_lit_lo,
        "a_lit_central_banked_294": EAGLE3_LIT_ACCEPT_CENTRAL,
        "a_lit_low_banked_294": EAGLE3_LIT_ACCEPT_LOW,
        "a_lit_central_matches_294": abs(a_lit_c - EAGLE3_LIT_ACCEPT_CENTRAL) <= 1e-9,
        "a_lit_low_matches_294": abs(a_lit_lo - EAGLE3_LIT_ACCEPT_LOW) <= 1e-9,
        "lit_projected_et_holda1_central": et_hold_c,
        "lit_projected_et_holda1_low": et_hold_lo,
        "reproduces_294_central": abs(et_hold_c - LIT_PROJ_HOLDA1_CENTRAL) <= 1e-9,
        "reproduces_294_low": abs(et_hold_lo - LIT_PROJ_HOLDA1_LOW) <= 1e-9,
        "deep_survival_sum_central": deep_survival_sum(a_lit_c),
        "deep_survival_sum_low": deep_survival_sum(a_lit_lo),
        "a1_sweep": sweep,
        "lit_et_monotone_in_a1_central": bool(monotone_c),
        "lit_et_monotone_in_a1_low": bool(monotone_lo),
    }


# --------------------------------------------------------------------------- #
# Step 2 -- solve for a_1* (PRIMARY solve): the a_1 the lit projection needs to clear
# --------------------------------------------------------------------------- #
def step2_solve_a1_star(s1: dict[str, Any]) -> dict[str, Any]:
    a_lit_c = s1["a_lit_central_rederived"]
    a_lit_lo = s1["a_lit_low_rederived"]
    a1_star_c = solve_a1_star(a_lit_c)
    a1_star_lo = solve_a1_star(a_lit_lo)
    # closed-form cross-check: a1* = (target - 1) / S
    a1_star_c_closed = (TARGET_BANKED - 1.0) / deep_survival_sum(a_lit_c)
    a1_star_lo_closed = (TARGET_BANKED - 1.0) / deep_survival_sum(a_lit_lo)
    return {
        "a1_star_central": a1_star_c,                     # TEST metric (central anchor)
        "a1_star_low": a1_star_lo,
        "a1_star_band": [a1_star_c, a1_star_lo],
        "a1_star_central_closed_form": a1_star_c_closed,
        "a1_star_low_closed_form": a1_star_lo_closed,
        "a1_star_central_closed_matches": abs(a1_star_c - a1_star_c_closed) <= 1e-9,
        "a1_star_low_closed_matches": abs(a1_star_lo - a1_star_lo_closed) <= 1e-9,
        "et_at_a1_star_central": lit_projected_et(a1_star_c, a_lit_c),
        "et_at_a1_star_low": lit_projected_et(a1_star_lo, a_lit_lo),
        "a1_lift_required_central": a1_star_c - A1_DEPLOYED,
        "a1_lift_required_low": a1_star_lo - A1_DEPLOYED,
        # for reference: how far a_1 sits below the j>=2 uniform target a_star
        "a1_star_central_vs_a_star_j2": a1_star_c - A_STAR_J2,
        "a1_star_low_vs_a_star_j2": a1_star_lo - A_STAR_J2,
    }


# --------------------------------------------------------------------------- #
# Step 3 -- feasibility vs EAGLE-3's achievable dense-model position-1 acceptance
# --------------------------------------------------------------------------- #
def step3_feasibility(s2: dict[str, Any]) -> dict[str, Any]:
    a1_star_c = s2["a1_star_central"]
    a1_star_lo = s2["a1_star_low"]
    dense_cons = EAGLE3_DENSE_A1_CONSERVATIVE     # 0.8893 -- chain-equiv flat accept (lit floor ~0.89)
    dense_cen = EAGLE3_DENSE_A1_CENTRAL           # 0.91 -- EAGLE-3 0-alpha central (inferred)
    dense_opt = EAGLE3_DENSE_A1_OPTIMISTIC        # 0.93 -- EAGLE-3 0-alpha upper range (inferred)
    # feasible iff a1_star <= achievable dense a_1 (does EAGLE-3 reach the needed position-1?)
    feasible_c_cons = a1_star_c <= dense_cons
    feasible_c_cen = a1_star_c <= dense_cen
    feasible_lo_cons = a1_star_lo <= dense_cons
    feasible_lo_cen = a1_star_lo <= dense_cen
    feasible_lo_opt = a1_star_lo <= dense_opt
    # int4 haircut flag: if even the OPTIMISTIC dense ceiling is below a1_star, the int4
    # quantized base is a harder ask than any dense EAGLE-3 number supports (unpriced #294 risk).
    int4_haircut_binds_central = a1_star_c > dense_opt
    int4_haircut_binds_low = a1_star_lo > dense_opt
    return {
        "eagle3_dense_a1_conservative": dense_cons,
        "eagle3_dense_a1_central": dense_cen,
        "eagle3_dense_a1_optimistic": dense_opt,
        # headline feasibility = central a1_star vs CONSERVATIVE (published-floor) ceiling -- the
        # honest, non-over-claiming bar. (Also feasible vs central+optimistic -- robust.)
        "a1_lift_is_feasible": bool(feasible_c_cons),           # TEST bool (central a1*, conservative ceiling)
        "a1_lift_is_feasible_central_vs_central": bool(feasible_c_cen),
        "a1_lift_is_feasible_low_conservative": bool(feasible_lo_cons),
        "a1_lift_is_feasible_low_central": bool(feasible_lo_cen),
        "a1_lift_is_feasible_low_optimistic": bool(feasible_lo_opt),
        "feasibility_margin_central_conservative": dense_cons - a1_star_c,   # >0 -> feasible
        "feasibility_margin_central_central": dense_cen - a1_star_c,
        "feasibility_margin_low_conservative": dense_cons - a1_star_lo,
        "feasibility_margin_low_central": dense_cen - a1_star_lo,
        "feasibility_margin_low_optimistic": dense_opt - a1_star_lo,
        "int4_haircut_binds_central": bool(int4_haircut_binds_central),
        "int4_haircut_binds_low": bool(int4_haircut_binds_low),
        "feasibility_interpretation": (
            "central anchor: a1* = %.4f needs an a_1-lift of +%.4f from the 0.7293 cliff; that sits "
            "%.4f BELOW EAGLE-3's conservative published-floor dense ceiling %.4f and well below the "
            "central 0-alpha %.4f -> FEASIBLE (robust). low anchor: a1* = %.4f needs +%.4f, which "
            "%s the conservative floor %.4f but IS feasible vs EAGLE-3's central 0-alpha %.4f "
            "(margin %+.4f) -- so even the downside leg clears under the central published estimate, "
            "with the int4-quantization haircut the residual unpriced risk (#294)."
            % (a1_star_c, a1_star_c - A1_DEPLOYED, dense_cons - a1_star_c, dense_cons, dense_cen,
               a1_star_lo, a1_star_lo - A1_DEPLOYED,
               "EXCEEDS" if a1_star_lo > dense_cons else "clears", dense_cons, dense_cen,
               dense_cen - a1_star_lo)),
    }


# --------------------------------------------------------------------------- #
# Step 4 -- tie to the #294 a_2 gate: the TWO sub-triggers
# --------------------------------------------------------------------------- #
def step4_two_subtriggers(s2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    # a_1 is a base-model predictability property (#289 built_raise_requires_nonlinear_drafter)
    # => a 2h frozen-backbone adaptation moves a_2 (deep positions) but NOT a_1.
    a1_moved_by_phase1 = False    # #289: a_1 cliff is a base-model property; only full retrain shifts it
    return {
        "subtrigger_cheap_a2_phase1_go": PHASE1_A2_GO,        # (i) #294 cheap a_2 proxy (2h)
        "subtrigger_cheap_a2_local_to_official": TAU_ACC,     # 1:1 transfer (lawine #288)
        "subtrigger_hard_a1_star_central": s2["a1_star_central"],  # (ii) hard a_1 cliff-break
        "subtrigger_hard_a1_star_low": s2["a1_star_low"],
        "a1_moved_by_2h_phase1": a1_moved_by_phase1,
        "a1_requires_full_retrain": True,
        "phase1_measures_a2_not_a1": True,
        "subtrigger_note": (
            "the human's #319 decision now has TWO sub-triggers: (i) the CHEAP a_2 >= %.4f "
            "Phase-1 proxy (#294, 2h, 1:1 local->official) and (ii) the HARD a_1 >= %.4f "
            "cliff-break (this card, central anchor). A 2h frozen-backbone Phase-1 run moves "
            "a_2 (deep-position adaptation) but NOT a_1 -- #289 found a_1 is a base-model "
            "predictability property (built_raise_requires_nonlinear_drafter=True), so the a_1 "
            "cliff-break is a FULL-retrain property the cheap Phase-1 cannot de-risk. The Phase-1 "
            "a_2 gate is necessary-but-not-sufficient: it screens the deep lift while the a_1 "
            "cliff-break remains the binding feasibility unknown only the full build resolves."
            % (PHASE1_A2_GO, s2["a1_star_central"])),
    }


# --------------------------------------------------------------------------- #
# Step 5 -- decision framing: does the straddle resolve UP or DOWN?
# --------------------------------------------------------------------------- #
def step5_decision(s2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    # Straddle resolves UP iff the central a_1-lift is feasible vs the conservative dense
    # ceiling (small, reachable lift). DOWN iff a1_star exceeds the achievable dense a_1.
    resolves_up = s3["a1_lift_is_feasible"]
    verdict = "UP" if resolves_up else "DOWN"
    return {
        "straddle_resolves": verdict,
        "straddle_resolves_up": bool(resolves_up),
        "build_justified_on_demand_axis": bool(resolves_up),
        "decision_note": (
            "the a_1-cliff lit-gap resolves the #294 straddle %s: the central lit anchor needs "
            "a_1 >= %.4f (lift +%.4f), which is FEASIBLE vs EAGLE-3's dense ceiling %.4f, so the "
            "DEMAND-side feasibility clears on the central anchor (build justified on this axis). "
            "The LOW anchor needs a_1 >= %.4f (lift +%.4f) -- at/above the dense ceiling -- so the "
            "downside leg stays RISKIER (int4 haircut unpriced). Cross-ref: fern #335 binding axis "
            "= SUPPLY; lawine #336/#339 demand REACHABLE-MARGINAL on the coverage axis; this card "
            "is the DEMAND-side a_1 ACCEPTANCE crux, complementary to the supply floor."
            % (verdict, s2["a1_star_central"], s2["a1_lift_required_central"],
               s3["eagle3_dense_a1_conservative"], s2["a1_star_low"],
               s2["a1_lift_required_low"])),
    }


# --------------------------------------------------------------------------- #
# Step 6 -- greedy-safety note (SPEED axis; greedy identity preserved by construction)
# --------------------------------------------------------------------------- #
def step6_greedy_safety() -> dict[str, Any]:
    return {
        "a1_cliff_card_is_cpu_analytic": True,
        "no_gpu": True,
        "no_served_change": True,
        "a1_is_speed_axis": True,
        "greedy_identity_preserved_by_construction": True,
        "note": (
            "a_1 / per-position acceptance is the SPEED axis; greedy identity is preserved BY "
            "CONSTRUCTION (EAGLE-3 emission = verify argmax -- the drafter only PROPOSES, the "
            "verifier's argmax token is always emitted). This card is CPU-analytic over banked "
            "W&B numbers: no GPU, no model forward, no served-file change, no HF Job, no "
            "submission. BASELINE stays 481.53."),
    }


# --------------------------------------------------------------------------- #
# Step 7 -- self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(s1: dict[str, Any], s2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    a_lit_c = s1["a_lit_central_rederived"]
    a1_star_c = s2["a1_star_central"]
    a1_star_lo = s2["a1_star_low"]
    checks: dict[str, bool] = {}

    # (a) lit_projected_et(0.7293) reproduces #294's hold-a_1 number 4.6901 (central + low).
    checks["a_reproduces_294_holda1"] = (
        abs(s1["lit_projected_et_holda1_central"] - LIT_PROJ_HOLDA1_CENTRAL) <= 1e-9
        and abs(s1["lit_projected_et_holda1_low"] - LIT_PROJ_HOLDA1_LOW) <= 1e-9)

    # (b) lit_projected_et monotone increasing in a_1 (unique root) -- central + low.
    checks["b_lit_et_monotone_in_a1"] = (
        bool(s1["lit_et_monotone_in_a1_central"]) and bool(s1["lit_et_monotone_in_a1_low"]))

    # (c) a1_star_central solves lit_projected_et = 4.9029 within tol.
    checks["c_a1_star_central_solves_target"] = abs(s2["et_at_a1_star_central"] - TARGET_BANKED) <= 1e-9

    # (d) a1_star_low > a1_star_central (lower deep-accept needs higher a_1 to compensate).
    checks["d_a1_star_low_gt_central"] = a1_star_lo > a1_star_c

    # (e) a1_lift_required = a1_star_central - 0.7293 > 0 (the cliff must lift).
    checks["e_a1_lift_required_positive"] = s2["a1_lift_required_central"] > 0.0

    # (f) at a_1 = a_star_j2 = 0.9089 (uniform full j>=2 target), E[T] >= 4.9029 (over-clears).
    checks["f_a1_at_a_star_j2_overclears"] = lit_projected_et(A_STAR_J2, a_lit_c) >= TARGET_BANKED

    # (g) constants imported EXACT (a1_deployed, target_et, lit_proj_holda1).
    checks["g_constants_imported_exact"] = (
        abs(A1_DEPLOYED - 0.7292532942898975) <= 1e-15
        and abs(round(A1_DEPLOYED, 4) - 0.7293) <= 1e-12
        and abs(TARGET_BANKED - 4.9029) <= 1e-12
        and abs(LIT_PROJ_HOLDA1_CENTRAL - 4.690056968034645) <= 1e-12
        and abs(A_STAR_J2 - 0.9088740395297805) <= 1e-12)

    # (h) NaN-clean over the a_1 sweep.
    flat = ([r["lit_et_central"] for r in s1["a1_sweep"]]
            + [r["lit_et_low"] for r in s1["a1_sweep"]]
            + [a1_star_c, a1_star_lo, s2["a1_lift_required_central"],
               s3["feasibility_margin_central_conservative"]])
    checks["h_nan_clean_sweep"] = all(math.isfinite(float(x)) for x in flat)

    # (i) a1_star in (0.7293, 1.0) -- valid acceptance above the cliff (central + low).
    checks["i_a1_star_above_cliff_in_unit"] = (
        A1_DEPLOYED < a1_star_c < 1.0 and A1_DEPLOYED < a1_star_lo < 1.0)

    # bonus: re-derived a_lit matches banked #294 (construction-identity).
    checks["j_a_lit_matches_294"] = (
        bool(s1["a_lit_central_matches_294"]) and bool(s1["a_lit_low_matches_294"]))
    # bonus: closed-form a1* == bisection a1* (root-find sanity).
    checks["k_a1_star_closed_form_matches"] = (
        bool(s2["a1_star_central_closed_matches"]) and bool(s2["a1_star_low_closed_matches"]))
    # bonus: deployed base reproduces #289's E[T] decomposition.
    checks["l_base_reproduces_289_decomp"] = abs(et_of_profile(A_BASE) - E_T_DECOMP_289) <= 1e-9

    gate = bool(all(checks.values()))
    return {"checks": checks, "a1_cliff_lit_gap_self_test_passes": gate}


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def build_report() -> dict[str, Any]:
    s1 = step1_define_and_validate()
    s2 = step2_solve_a1_star(s1)
    s3 = step3_feasibility(s2)
    s4 = step4_two_subtriggers(s2, s3)
    s5 = step5_decision(s2, s3)
    s6 = step6_greedy_safety()
    st = self_test(s1, s2, s3)

    handoff = (
        "the EAGLE-3 lit projection clears 4.9029 on our base only if a_1 lifts from 0.7293 to "
        "a1_star=%.4f (central) / %.4f (low), an a_1-lift of +%.4f (central) / +%.4f (low) that "
        "%s feasible vs EAGLE-3's published dense a_1 (~%.3f conservative floor / ~%.2f central "
        "0-alpha) -- so the straddle resolves %s, and the #319 decision now has two sub-triggers: "
        "the cheap a_2>=0.8342 Phase-1 proxy (#294) AND the hard a_1>=%.4f cliff-break (this card), "
        "with a_1 likely a base-model property only the full retrain shifts (#289)." % (
            s2["a1_star_central"], s2["a1_star_low"],
            s2["a1_lift_required_central"], s2["a1_lift_required_low"],
            "IS" if s3["a1_lift_is_feasible"] else "is NOT",
            s3["eagle3_dense_a1_conservative"], s3["eagle3_dense_a1_central"],
            s5["straddle_resolves"], s2["a1_star_central"]))

    return {
        "pr": 342,
        "leg": "a_1-cliff lit-gap: price the a_1-lift that resolves #294's lit straddle",
        "a1_cliff_lit_gap_analysis_only": True,
        "imported": {
            "official": OFFICIAL, "ceiling_lambda1": CEILING_LAMBDA1, "K_cal": K_CAL,
            "E_T_anchor": E_T_ANCHOR, "K_spec": K_SPEC, "E_T_max": E_T_MAX,
            "linear_cap_denken119": LINEAR_CAP, "target_banked_wirbel290": TARGET_BANKED,
            "private_verified": PRIVATE_VERIFIED,
            "a_base_kanna289": A_BASE, "a1_deployed_kanna289": A1_DEPLOYED,
            "E_T_decomp_289": E_T_DECOMP_289,
            "a_star_j2_kanna294": A_STAR_J2,
            "lit_proj_holda1_central_kanna294": LIT_PROJ_HOLDA1_CENTRAL,
            "lit_proj_holda1_low_kanna294": LIT_PROJ_HOLDA1_LOW,
            "eagle3_lit_accept_central_kanna294": EAGLE3_LIT_ACCEPT_CENTRAL,
            "eagle3_lit_accept_low_kanna294": EAGLE3_LIT_ACCEPT_LOW,
            "tau_acc_lawine288": TAU_ACC, "phase1_a2_go_kanna294": PHASE1_A2_GO,
            "eagle3_lit_L_chain_central": EAGLE3_LIT_L_CHAIN_CENTRAL,
            "eagle3_lit_L_chain_low": EAGLE3_LIT_L_CHAIN_LOW,
            "eagle3_dense_a1_conservative": EAGLE3_DENSE_A1_CONSERVATIVE,
            "eagle3_dense_a1_central": EAGLE3_DENSE_A1_CENTRAL,
            "eagle3_dense_a1_optimistic": EAGLE3_DENSE_A1_OPTIMISTIC,
            "eagle3_lit_note": EAGLE3_LIT_NOTE,
            "wandb_sources": {
                "kanna_294": "j0ss47bv", "kanna_289": "fi34s269", "wirbel_290": "ub3kpsso",
                "lawine_288": "i1e5054m", "fern_335": "5pos499e", "lawine_336": "krroookz",
            },
        },
        "step1_define_validate": s1,
        "step2_solve_a1_star": s2,
        "step3_feasibility": s3,
        "step4_two_subtriggers": s4,
        "step5_decision": s5,
        "step6_greedy_safety": s6,
        "self_test": st["checks"],
        # ---- headline metrics ----
        "a1_cliff_lit_gap_self_test_passes": st["a1_cliff_lit_gap_self_test_passes"],  # PRIMARY
        "a1_star_for_lit_clears": s2["a1_star_central"],                               # TEST (central)
        "a1_star_low": s2["a1_star_low"],
        "a1_lift_required_central": s2["a1_lift_required_central"],
        "a1_lift_is_feasible": s3["a1_lift_is_feasible"],
        "straddle_resolves": s5["straddle_resolves"],
        "a1_cliff_card_is_cpu_analytic": s6["a1_cliff_card_is_cpu_analytic"],
        "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN audit
# --------------------------------------------------------------------------- #
def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


# --------------------------------------------------------------------------- #
# W&B logging (mirrors kanna #294; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, report: dict[str, Any]) -> str | None:
    if getattr(args, "no_wandb", False):
        return None
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub/namespace wandb with no .init")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[a1-cliff-lit-gap] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return None

    s1 = report["step1_define_validate"]
    s2 = report["step2_solve_a1_star"]
    s3 = report["step3_feasibility"]
    s4 = report["step4_two_subtriggers"]
    s5 = report["step5_decision"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="kanna", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-a1-cliff-lit-gap", "a1-lift", "lit-straddle-resolve", "demand-feasibility",
                  "second-subtrigger", "greedy-identity-safety", "bank-the-analysis", "pr-342"],
            config={
                "official": OFFICIAL, "K_cal": K_CAL, "linear_cap": LINEAR_CAP,
                "target_banked": TARGET_BANKED, "a1_deployed": A1_DEPLOYED,
                "a_star_j2": A_STAR_J2, "K_spec": K_SPEC,
                "eagle3_lit_L_chain_central": EAGLE3_LIT_L_CHAIN_CENTRAL,
                "eagle3_lit_L_chain_low": EAGLE3_LIT_L_CHAIN_LOW,
                "eagle3_dense_a1_conservative": EAGLE3_DENSE_A1_CONSERVATIVE,
                "eagle3_dense_a1_optimistic": EAGLE3_DENSE_A1_OPTIMISTIC,
                "imports": "kanna#294(j0ss47bv a_lit/lit-straddle/a_star) x kanna#289(fi34s269 a_k) "
                           "x wirbel#290(ub3kpsso target=4.9029) x lawine#288(i1e5054m tau_acc=1.0)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[a1-cliff-lit-gap] wandb init failed (analysis unaffected): {exc}", flush=True)
        return None
    if run is None:
        print("[a1-cliff-lit-gap] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "a1_cliff_lit_gap_self_test_passes": int(bool(report["a1_cliff_lit_gap_self_test_passes"])),
        "a1_star_for_lit_clears": report["a1_star_for_lit_clears"],
        "a1_star_low": report["a1_star_low"],
        "a1_lift_required_central": report["a1_lift_required_central"],
        "a1_lift_required_low": s2["a1_lift_required_low"],
        "a1_lift_is_feasible": int(bool(report["a1_lift_is_feasible"])),
        "a1_lift_is_feasible_central_vs_central": int(bool(s3["a1_lift_is_feasible_central_vs_central"])),
        "a1_lift_is_feasible_low_conservative": int(bool(s3["a1_lift_is_feasible_low_conservative"])),
        "a1_lift_is_feasible_low_central": int(bool(s3["a1_lift_is_feasible_low_central"])),
        "a1_lift_is_feasible_low_optimistic": int(bool(s3["a1_lift_is_feasible_low_optimistic"])),
        "straddle_resolves_up": int(bool(report["step5_decision"]["straddle_resolves_up"])),
        "lit_projected_et_holda1_central": s1["lit_projected_et_holda1_central"],
        "lit_projected_et_holda1_low": s1["lit_projected_et_holda1_low"],
        "reproduces_294_central": int(bool(s1["reproduces_294_central"])),
        "reproduces_294_low": int(bool(s1["reproduces_294_low"])),
        "a_lit_central_rederived": s1["a_lit_central_rederived"],
        "a_lit_low_rederived": s1["a_lit_low_rederived"],
        "a_lit_central_matches_294": int(bool(s1["a_lit_central_matches_294"])),
        "et_at_a1_star_central": s2["et_at_a1_star_central"],
        "et_at_a1_star_low": s2["et_at_a1_star_low"],
        "eagle3_dense_a1_conservative": s3["eagle3_dense_a1_conservative"],
        "eagle3_dense_a1_central": s3["eagle3_dense_a1_central"],
        "eagle3_dense_a1_optimistic": s3["eagle3_dense_a1_optimistic"],
        "feasibility_margin_central_conservative": s3["feasibility_margin_central_conservative"],
        "feasibility_margin_central_central": s3["feasibility_margin_central_central"],
        "feasibility_margin_low_conservative": s3["feasibility_margin_low_conservative"],
        "feasibility_margin_low_central": s3["feasibility_margin_low_central"],
        "feasibility_margin_low_optimistic": s3["feasibility_margin_low_optimistic"],
        "int4_haircut_binds_central": int(bool(s3["int4_haircut_binds_central"])),
        "int4_haircut_binds_low": int(bool(s3["int4_haircut_binds_low"])),
        "subtrigger_cheap_a2_phase1_go": s4["subtrigger_cheap_a2_phase1_go"],
        "subtrigger_hard_a1_star_central": s4["subtrigger_hard_a1_star_central"],
        "a1_moved_by_2h_phase1": int(bool(s4["a1_moved_by_2h_phase1"])),
        "a1_requires_full_retrain": int(bool(s4["a1_requires_full_retrain"])),
        "a1_cliff_card_is_cpu_analytic": int(bool(report["a1_cliff_card_is_cpu_analytic"])),
        "nan_clean": int(bool(report["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in report["self_test"].items()},
        **{f"lit_et_central_a1_{str(r['a1']).replace('.', 'p')}": r["lit_et_central"]
           for r in s1["a1_sweep"]},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_a1_cliff_lit_gap_result", artifact_type="validity",
                          data=report)
        rid = getattr(run, "id", None)
        finish_wandb(run)
        print(f"[a1-cliff-lit-gap] wandb logged {len(summary)} summary keys (run {rid})", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[a1-cliff-lit-gap] wandb write failed (analysis unaffected): {exc}", flush=True)
        return None


def _print_human(report: dict[str, Any]) -> None:
    s1 = report["step1_define_validate"]
    s2 = report["step2_solve_a1_star"]
    s3 = report["step3_feasibility"]
    s5 = report["step5_decision"]
    print("\n" + "=" * 100, flush=True)
    print(" a_1-CLIFF LIT-GAP: price the a_1-lift that resolves #294's lit straddle (PR #342)",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  validate hold-a_1 lit E[T] (central)  : {s1['lit_projected_et_holda1_central']:.6f}  "
          f"(reproduces #294 4.6901 = {s1['reproduces_294_central']})", flush=True)
    print(f"  a_lit central / low (re-derived)      : {s1['a_lit_central_rederived']:.4f} / "
          f"{s1['a_lit_low_rederived']:.4f}  (matches #294 = {s1['a_lit_central_matches_294']})",
          flush=True)
    print(f"  a1_star CENTRAL (TEST) / LOW          : {s2['a1_star_central']:.4f} / "
          f"{s2['a1_star_low']:.4f}   (deployed cliff {A1_DEPLOYED:.4f})", flush=True)
    print(f"  a1_lift_required central / low        : +{s2['a1_lift_required_central']:.4f} / "
          f"+{s2['a1_lift_required_low']:.4f}", flush=True)
    print(f"  EAGLE-3 dense a_1 ceiling cons / opt  : {s3['eagle3_dense_a1_conservative']:.4f} / "
          f"{s3['eagle3_dense_a1_optimistic']:.4f}", flush=True)
    print(f"  a1_lift_is_feasible (TEST, central)   : {s3['a1_lift_is_feasible']}  "
          f"(margin {s3['feasibility_margin_central_conservative']:+.4f} vs conservative ceiling)",
          flush=True)
    print(f"  straddle_resolves                     : {s5['straddle_resolves']}", flush=True)
    print(f"  PRIMARY self_test                     : {report['a1_cliff_lit_gap_self_test_passes']}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  HAND-OFF: {report['handoff']}\n", flush=True)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the a_1-cliff lit-gap analytic + PRIMARY self-test over banked numbers.")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-a1-cliff-lit-gap")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="kanna/eagle3-a1-cliff-lit-gap")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    report = build_report()
    nan_paths = _assert_nan_clean(report)
    report["nan_clean"] = not nan_paths
    report["a1_cliff_lit_gap_self_test_passes"] = bool(
        report["a1_cliff_lit_gap_self_test_passes"] and report["nan_clean"])
    if nan_paths:
        print(f"[a1-cliff-lit-gap] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_mem_mib"] = round(peak_kib / 1024.0, 3)
    report["created_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    wid = _maybe_log_wandb(args, report)
    report["wandb_run_id"] = wid

    HERE.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=float))

    _print_human(report)
    print(f"[a1-cliff-lit-gap] wrote {OUT_PATH}", flush=True)
    print(f"[a1-cliff-lit-gap] PRIMARY a1_cliff_lit_gap_self_test_passes = "
          f"{report['a1_cliff_lit_gap_self_test_passes']}", flush=True)
    print(f"[a1-cliff-lit-gap] TEST a1_star_for_lit_clears = "
          f"{report['a1_star_for_lit_clears']:.6f}", flush=True)
    print(f"[a1-cliff-lit-gap] a1_lift_is_feasible = {report['a1_lift_is_feasible']}", flush=True)
    print(f"[a1-cliff-lit-gap] straddle_resolves = {report['straddle_resolves']}", flush=True)
    print(f"[a1-cliff-lit-gap] wandb run = {wid}", flush=True)

    if args.self_test and not report["a1_cliff_lit_gap_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[a1-cliff-lit-gap] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
