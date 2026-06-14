#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Private-bar reachability: ground f_priv + is 500-PRIVATE reachable at the ceiling? (PR #224).

WHAT THIS IS
------------
Capstone of the sigma-decomposition / private-grade lane (#194->#202->#206->#210->#217->this).
Your #217 surfaced the launch crux: the lambda=1 PHYSICAL ceiling mu_pub=520.95 does NOT clear
the 500-PRIVATE bar at P95. The private build target is

        mu_bar_private(f_priv, sigma_priv) = (500 + z1 * sigma_priv) / f_priv          (MASTER)

(the public mean mu_pub whose private re-benchmark clears 500 at the one-sided P95 LCB;
private mean = mu_pub * f_priv, private grade = ONE fresh draw => sigma_priv = sigma_draw).
At the banked (f_priv=0.969107, sigma_priv=sigma_draw=7.391) this is 528.48, and the ceiling
520.95 falls SHORT by +7.53 TPS (#217: lambda1_ceiling_clears_private_bar=0).

THE FINDING HINGES ON f_priv. This leg GROUNDS the public->private multiplier from the banked
paired evidence, then solves the REACHABILITY question: what mu_pub / lambda / f_priv / sigma
CLOSES the gap, and is 500-private reachable at the physical ceiling AT ALL?

GROUND f_priv (the load-bearing correction)
-------------------------------------------
The assumed f_priv_kanna = 0.969107 = (1 - drop_both) * tau_low composes ONLY the 2.35%
tree-topology drop (#176 drop_both) times the tau_low corner. The ONE hard OBSERVED paired
official draw is PR #52: 481.53 PUBLIC served, 460.85 PRIVATE-verified VALID =>

        f_priv_obs = 460.85 / 481.53 = 0.957054   (a 4.295% public->private drop)

and 4.295% is EXACTLY stark #176's decode_drop_pct (gt_drop = 0.0429464; 1 - gt_drop = 0.957054
to 7 digits). So the hard paired point lands on the DECODE drop, not the tree drop: f_priv_kanna
OMITS ~1.9 points of decode-path drop and is OPTIMISTIC. With one hard paired point we report the
POINT (0.957054) plus a BOUNDING band [0.95 (the 5% disqualify-gate floor: any VALID build retains
>= 95%), 0.969107 (the optimistic tree-only model)] -- NOT a sampling CI.

RESTATE THE BAR + REACHABILITY
------------------------------
Under the grounded f_priv the private build target RISES (528.48 -> ~535.1) and the gap WIDENS
(+7.53 -> +14.2). The ceiling-misses-private-bar finding is ROBUST across the whole grounded band.
Then for each lever we solve the value that makes ceiling_clears_private_bar=True at P95:
  (i)  public ceiling needed   mu_ceiling_needed = mu_bar_private               (the kernel target)
  (ii) f_priv needed @520.95    = mu_safe / 520.95 = 0.98312                     (> plausible band)
  (iii) sigma_priv shrink @520.95 = (520.95*f_priv - 500)/z1                     (NEGATIVE at f_obs)
  (iv) lambda needed (ceiling scales w/ lambda via #191 public-LCB(lambda))      (> 1 => unphysical)
KEY: at the grounded f_priv the PRIVATE MEAN at the ceiling is 520.95*0.957 = 498.5 < 500, so
sigma-shrink CANNOT help (even sigma->0 misses); lambda>1 is unphysical; f_priv>=0.983 is above the
plausible band -- the ONLY actionable lever is RAISING THE PUBLIC CEILING to mu_ceiling_needed
(~528 optimistic / ~535 grounded), i.e. the kernel-ceiling route (wirbel #216 / DVR) must reach
~528-535, NOT 500, to win PRIVATE at P95. best-of-N does NOT help (round-trips #210 flat-in-N).

LOCAL CPU-only analytic synthesis over EXISTING banked draws + #217. No GPU / vLLM / HF Job /
submission / served-file / official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS;
authorizes nothing. Imports #217/#210/#204/#191/#176 + the #52 hard paired point VERBATIM; does NOT
re-derive the sigmas, the ceiling (520.95), or the bar (0.9780). DECISION-ARMS the private-bar
reading; integration is fern #185's. NOT ubel #222 (which-gate-binds). NOT open2. NOT a launch.

SELF-TEST (PRIMARY = private_bar_reachability_self_test_passes)
--------------------------------------------------------------
(a) mu_bar_private(0.969107, sigma_draw) reproduces 528.4835555959944 EXACTLY (resid -> 0).
(b) mu_bar_private monotone DECREASING in f_priv and INCREASING in sigma_priv.
(c) f_priv=1, sigma_priv->0  =>  mu_bar_private -> 500 (resid -> 0).
(d) the #52 paired point round-trips: 481.53 * f_priv_obs == 460.85 (resid -> 0).
(e) best-of-N flat: imports #210 n_star_private==1 and private_clear_flat_in_n==True.
(f) NaN-clean.
TEST = mu_ceiling_needed (the public ceiling that clears 500-private at P95, grounded f_priv).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
TRIGGER_RECONCILE_217 = os.path.join(
    _ROOT, "research/validity/trigger_reconcile/trigger_reconcile_results.json")
WINNERS_CURSE_210 = os.path.join(
    _ROOT, "research/validity/winners_curse_budget/winners_curse_budget_results.json")
PRIVATE_BUILD_BAR_191 = os.path.join(
    _ROOT, "research/validity/private_build_bar/results.json")
ADVERSE_SKEW_176 = os.path.join(
    _ROOT, "research/validity/private_adverse_skew/results.json")

TARGET = 500.0
P_TARGET = 0.95
Z1 = 1.6448536269514722  # one-sided P95 (#204 z1=1.64485)
DISQUALIFY_GATE_PCT = 5.0  # private re-run VALID iff public->private drop <= 5% (#191 constant)

# the ONE hard OBSERVED paired official draw (PR #52, MERGED baseline)
PUB_52 = 481.53   # official PUBLIC served TPS (128/128, PPL 2.3772)
PRIV_52 = 460.85  # PRIVATE-verified VALID TPS (Delta 4.3% <= 5%)

# grounded-band edges + sensitivity-surface ranges (PR step 1 + 4)
F_PRIV_GATE_FLOOR = 1.0 - DISQUALIFY_GATE_PCT / 100.0  # 0.95
F_PRIV_SURFACE = (0.95, 0.975)
SIGMA_SURFACE = (3.0, 8.0)
SURFACE_N = 26  # grid resolution per axis

TOL = 1e-6
EXACT_TOL = 1e-9


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


# ---------------------------------------------------------------------------
# MASTER identity: the public build target that clears 500-private at P95
# ---------------------------------------------------------------------------
def mu_bar_private(f_priv: float, sigma_priv: float, target: float = TARGET, z1: float = Z1,
                   mu_safe: float | None = None) -> float:
    """mu_pub such that mu_pub*f_priv - z1*sigma_priv = target (one-sided P95 private LCB).

    Pass mu_safe to reproduce the banked path EXACTLY at the fresh sigma (mu_safe_fresh is the
    #210/#217 rounded import 512.15707117161; recomputing 500+z1*sigma_draw differs by ~3e-10).
    """
    safe = mu_safe if mu_safe is not None else (target + z1 * sigma_priv)
    return safe / f_priv


def private_mean_at(mu_pub: float, f_priv: float) -> float:
    return mu_pub * f_priv


def private_lcb_at(mu_pub: float, f_priv: float, sigma_priv: float, z1: float = Z1) -> float:
    return mu_pub * f_priv - z1 * sigma_priv


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy) -- identical to #194/#202/#210/#217."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_private_clear(mu_pub: float, f_priv: float, sigma_priv: float,
                    target: float = TARGET, z1: float = Z1) -> float:
    """P(mu_pub*f_priv + N(0,sigma_priv) >= target) -- the private clear probability."""
    return 1.0 - _phi((target - mu_pub * f_priv) / sigma_priv)


# ---------------------------------------------------------------------------
# import the banked constants (verbatim)
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    d217 = _load(TRIGGER_RECONCILE_217)
    d210 = _load(WINNERS_CURSE_210)
    d191 = _load(PRIVATE_BUILD_BAR_191)
    d176 = _load(ADVERSE_SKEW_176)

    imp217 = d217["import_banked"]
    imp210 = d210["import_banked"]
    both = d191["synthesis"]["per_topology"]["both_bugs"]
    cst191 = d191["synthesis"]["constants"]

    out = {
        # ---- #204/#217 launch sigmas + ceiling + N=1 GO trigger ----
        "z1": Z1,
        "sigma_hw": _finite(imp217["sigma_hw"]),            # 4.864468814937121  FROZEN
        "sigma_draw": _finite(imp217["sigma_draw"]),        # 7.390974474817942  FRESH = sigma_priv
        "mu_safe_fresh": _finite(imp217["mu_safe_fresh"]),  # 512.15707117161 = 500 + z1*sigma_draw
        "lambda1_ceiling": _finite(imp217["lambda1_ceiling"]),          # 520.9527323111674
        "t_base_central": _finite(imp217["t_base_central"]),            # 512.4101 (N=1 GO trigger)
        # ---- #210/#217 f_priv provenance + private bar ----
        "f_priv_kanna": _finite(imp217["f_priv"]),          # 0.969106920637722 (ASSUMED, tree-only)
        "drop_both": _finite(imp210["drop_both"]),          # 0.023502816766841544 (2.35% tree drop)
        "tau_low": _finite(imp210["tau_low"]),              # 0.9924318649123313
        "decode_drop_pct_210": _finite(imp210["decode_drop_pct"]),      # 4.294644155088989
        "mu_bar_private_217": _finite(imp217["mu_bar_private_corrected"]),  # 528.4835555959944 (headline)
        "mu_bar_private_closed_form_210": _finite(
            d210["private_corrected_bar"]["mu_bar_private_corrected_closed_form"]),  # 528.4835555963056
        "lambda_star_191": _finite(imp217["lambda_star_191"]),          # 0.9780112973731208 (bar)
        # ---- #210 best-of-N flat (the round-trip) ----
        "n_star_private_210": int(d210["monotonicity_in_n"]["n_star_private"]),         # 1
        "private_clear_flat_in_n_210": bool(d210["monotonicity_in_n"]["private_clear_flat_in_n"]),
        "p_private_clear_at_ceiling_210": _finite(imp217["p_private_clear_at_ceiling"]),  # 0.7444
        # ---- #176 decode drop (the corroboration that f_priv_obs == 1 - decode_drop) ----
        "gt_drop_176": _finite(d176["constants"]["gt_drop"]),           # 0.04294644155088978
        # ---- #191 public-LCB(lambda) map: the composition basis for the lambda lever ----
        "public_lcb_lambda1": _finite(both["public_lcb_lambda1"]),          # 520.9527323111674
        "public_lcb_at_public_bar": _finite(both["public_lcb_at_public_bar"]),  # 499.99979806442366
        "public_bar_lambda_191": _finite(cst191["public_bar_both_bugs"]),  # 0.9052283680740145
        "public_central_lambda1": _finite(both["public_central_lambda1"]),     # 535.4330096522525
    }
    # provenance self-check (not a self-test gate): f_priv_kanna == (1-drop_both)*tau_low
    out["f_priv_kanna_recomputed"] = (1.0 - out["drop_both"]) * out["tau_low"]
    out["f_priv_kanna_provenance_resid"] = abs(out["f_priv_kanna_recomputed"] - out["f_priv_kanna"])
    # sanity: ceiling consistent across #217 and #191 (same physical lambda=1 LCB)
    out["ceiling_consistent_217_vs_191"] = abs(out["lambda1_ceiling"] - out["public_lcb_lambda1"])
    return out


# ---------------------------------------------------------------------------
# step 1: GROUND f_priv (the multiplier)
# ---------------------------------------------------------------------------
def ground_f_priv(imp: dict[str, Any]) -> dict[str, Any]:
    f_priv_kanna = imp["f_priv_kanna"]
    f_priv_obs = PRIV_52 / PUB_52                       # the ONE hard paired official draw
    drop_obs_pct = (PUB_52 - PRIV_52) / PUB_52 * 100.0  # 4.2946% == decode drop
    f_priv_decode = 1.0 - imp["gt_drop_176"]            # #176 modeled decode drop -> 0.957054
    f_priv_floor = F_PRIV_GATE_FLOOR                    # 0.95 (5% disqualify-gate floor)

    # verdict: is the assumed value optimistic / pessimistic / consistent vs the hard point?
    if f_priv_kanna > f_priv_obs + TOL:
        verdict = "OPTIMISTIC"
    elif f_priv_kanna < f_priv_obs - TOL:
        verdict = "PESSIMISTIC"
    else:
        verdict = "CONSISTENT"

    return {
        "f_priv_kanna_assumed": f_priv_kanna,            # 0.969107 (tree-only * tau_low)
        "f_priv_obs_52": f_priv_obs,                     # 0.957054 (HARD paired #52)
        "drop_obs_pct_52": drop_obs_pct,                 # 4.2946%
        "f_priv_decode_176": f_priv_decode,              # 0.957054 (1 - #176 decode drop)
        "obs_matches_decode_drop_resid": abs(f_priv_obs - f_priv_decode),  # ~0 (same measurement)
        "f_priv_floor_gate": f_priv_floor,               # 0.95
        # headline: grounded point + bounding band (one hard point => band, NOT sampling CI)
        "f_priv_grounded_point": f_priv_obs,             # 0.957054
        "f_priv_grounded_band": [f_priv_floor, f_priv_kanna],  # [0.95, 0.969107]
        "assumed_vs_observed_verdict": verdict,          # OPTIMISTIC
        "assumed_minus_observed": f_priv_kanna - f_priv_obs,   # +0.01205 (~1.2 pts too high)
        "n_hard_paired_draws": 1,
        "ci_basis": (
            "ONE hard paired official draw (#52). NO sampling CI is defensible; the band is a "
            "BOUNDING model: lower = 5% disqualify-gate floor (1-0.05=0.95, any VALID build retains "
            ">=95%), upper = the optimistic tree-only model 0.969107. The point 0.957054 is the #52 "
            "observation, doubly anchored by stark #176's modeled decode drop (1 - 0.042946 = "
            "0.957054, the SAME measurement -- not an independent second draw)."
        ),
        "verdict_detail": (
            "f_priv_kanna = (1 - drop_both[2.35% tree]) * tau_low = 0.969107 OMITS the decode-path "
            "drop. The hard #52 paired point shows the realized public->private drop is 4.295% (the "
            "DECODE drop), so 0.969107 is OPTIMISTIC by ~1.2 points; the defensible grounded f_priv "
            "is 0.957054 (point), band [0.95, 0.969107]."
        ),
    }


# ---------------------------------------------------------------------------
# step 2: RESTATE the private bar under grounded f_priv
# ---------------------------------------------------------------------------
def restate_bar(imp: dict[str, Any], gf: dict[str, Any]) -> dict[str, Any]:
    ceiling = imp["lambda1_ceiling"]
    sdr = imp["sigma_draw"]
    mu_safe = imp["mu_safe_fresh"]  # banked fresh path -> reproduces #217's 528.48 exactly

    def row(label: str, f_priv: float) -> dict[str, Any]:
        bar = mu_bar_private(f_priv, sdr, mu_safe=mu_safe)
        return {
            "label": label,
            "f_priv": f_priv,
            "mu_bar_private": bar,
            "ceiling_clears_private_bar": bool(ceiling >= bar),
            "gap_bar_minus_ceiling": bar - ceiling,
            "private_mean_at_ceiling": private_mean_at(ceiling, f_priv),
            "private_lcb_at_ceiling": private_lcb_at(ceiling, f_priv, sdr),
            "p_private_clear_at_ceiling": p_private_clear(ceiling, f_priv, sdr),
        }

    r_kanna = row("assumed_0.969107", gf["f_priv_kanna_assumed"])
    r_obs = row("observed_52_0.957054", gf["f_priv_obs_52"])
    r_floor = row("gate_floor_0.95", gf["f_priv_floor_gate"])

    # (a) reproduce 528.48 EXACTLY at f_priv=0.969107. The master identity uses the closed-form
    # path (mu_safe_fresh/f_priv), so it matches #210's banked closed_form to machine epsilon; it
    # matches the #217 headline (computed via 504.873+23.61) to 3.11e-10 -- the SAME float-path
    # residual #210/#217 themselves banked (#217 reconcile_residual_closed_form = 3.1116e-10).
    repro_resid = abs(r_kanna["mu_bar_private"] - imp["mu_bar_private_closed_form_210"])
    repro_resid_vs_headline = abs(r_kanna["mu_bar_private"] - imp["mu_bar_private_217"])

    # robust? the ceiling MISSES the bar across the whole grounded band [floor, assumed]
    finding_robust = (not r_kanna["ceiling_clears_private_bar"]
                      and not r_obs["ceiling_clears_private_bar"]
                      and not r_floor["ceiling_clears_private_bar"])

    return {
        "rows": {"assumed": r_kanna, "observed": r_obs, "gate_floor": r_floor},
        "reproduce_528_resid": repro_resid,                      # -> 0 (vs #210 closed_form)
        "reproduce_528_resid_vs_headline": repro_resid_vs_headline,  # 3.11e-10 (banked float path)
        "mu_bar_private_assumed": r_kanna["mu_bar_private"],     # 528.48
        "mu_bar_private_observed": r_obs["mu_bar_private"],      # 535.1
        "mu_bar_private_floor": r_floor["mu_bar_private"],       # 539.1
        "gap_assumed": r_kanna["gap_bar_minus_ceiling"],        # +7.53 (== #217)
        "gap_observed": r_obs["gap_bar_minus_ceiling"],         # +14.2 (WIDENS)
        "gap_floor": r_floor["gap_bar_minus_ceiling"],          # +18.2
        "finding_robust_to_fpriv": bool(finding_robust),        # True
        "private_mean_at_ceiling_observed": r_obs["private_mean_at_ceiling"],  # 498.5 < 500 (!)
        "private_mean_at_ceiling_below_500_observed":
            bool(r_obs["private_mean_at_ceiling"] < TARGET),    # True -- sharper than #217
        "interpretation": (
            "Under the grounded f_priv the private build target RISES from 528.48 (assumed) to "
            "~535.1 (observed) and the gap to the 520.95 ceiling WIDENS from +7.53 to +14.2. The "
            "ceiling-misses-private-bar finding is ROBUST across the whole grounded band [0.95, "
            "0.969107]. SHARPER than #217: at the observed f_priv the PRIVATE MEAN at the ceiling "
            "is 520.95*0.957 = 498.5 < 500 (not just the LCB) -- the ceiling build is sub-500 in "
            "the mean on a fresh private draw, so re-draws/variance cuts cannot rescue it."
        ),
    }


# ---------------------------------------------------------------------------
# step 3: REACHABILITY -- what CLOSES the gap (the core)
# ---------------------------------------------------------------------------
def _lambda_of_ceiling_slope(imp: dict[str, Any]) -> float:
    """Local slope d(public_LCB)/d(lambda) near lambda=1 from #191's both-bugs map."""
    return ((imp["public_lcb_lambda1"] - imp["public_lcb_at_public_bar"])
            / (1.0 - imp["public_bar_lambda_191"]))


def lambda_needed_for(mu_target: float, imp: dict[str, Any]) -> float:
    """lambda s.t. public_LCB(lambda) = mu_target (local-linear extrapolation past lambda=1)."""
    slope = _lambda_of_ceiling_slope(imp)
    return 1.0 + (mu_target - imp["public_lcb_lambda1"]) / slope


def reachability(imp: dict[str, Any], gf: dict[str, Any], restate: dict[str, Any]) -> dict[str, Any]:
    ceiling = imp["lambda1_ceiling"]
    sdr = imp["sigma_draw"]
    shw = imp["sigma_hw"]
    mu_safe = imp["mu_safe_fresh"]
    slope = _lambda_of_ceiling_slope(imp)

    def levers_at(f_priv: float) -> dict[str, Any]:
        bar = mu_bar_private(f_priv, sdr, mu_safe=mu_safe)
        # (i) public ceiling needed = the private build target itself
        mu_ceiling_needed = bar
        # (ii) f_priv needed at the FIXED ceiling 520.95
        f_priv_needed = (TARGET + Z1 * sdr) / ceiling   # = mu_safe / ceiling = 0.98312
        # (iii) sigma_priv shrink needed at the FIXED ceiling 520.95 and THIS f_priv
        sigma_needed = (ceiling * f_priv - TARGET) / Z1  # NEGATIVE => impossible (mean<500)
        sigma_shrink_possible = sigma_needed > 0.0
        # (iv) lambda needed (ceiling scales with lambda via #191 public-LCB map)
        lam_needed = lambda_needed_for(bar, imp)
        return {
            "f_priv": f_priv,
            "mu_bar_private": bar,
            "lever_i_mu_ceiling_needed": mu_ceiling_needed,
            "lever_i_ceiling_lift_tps": mu_ceiling_needed - ceiling,
            "lever_ii_f_priv_needed_at_ceiling": f_priv_needed,
            "lever_ii_f_priv_lift": f_priv_needed - f_priv,
            "lever_iii_sigma_priv_needed_at_ceiling": sigma_needed,
            "lever_iii_sigma_shrink_possible": bool(sigma_shrink_possible),
            "lever_iii_sigma_shrink_tps": (sdr - sigma_needed) if sigma_shrink_possible else None,
            "lever_iv_lambda_needed": lam_needed,
            "lever_iv_lambda_unphysical": bool(lam_needed > 1.0),
            "private_mean_at_ceiling": private_mean_at(ceiling, f_priv),
        }

    L_kanna = levers_at(gf["f_priv_kanna_assumed"])
    L_obs = levers_at(gf["f_priv_obs_52"])

    # reachable at the PHYSICAL ceiling? (lambda<=1) -- under grounded point
    reachable = bool(ceiling >= L_obs["mu_bar_private"])  # False

    # most-actionable lever (grounded f_priv): lambda unphysical, sigma useless (mean<500),
    # f_priv-lift above plausible band => raise the public CEILING (the kernel route).
    sigma_useful = L_obs["lever_iii_sigma_shrink_possible"]
    f_priv_above_band = L_obs["lever_ii_f_priv_needed_at_ceiling"] > gf["f_priv_kanna_assumed"]
    most_actionable = "raise_public_ceiling"
    most_actionable_magnitude = L_obs["lever_i_mu_ceiling_needed"]

    # cross-check: freezing the PRIVATE run's clocks (sigma_priv -> sigma_hw) still misses
    bar_frozen_obs = mu_bar_private(gf["f_priv_obs_52"], shw)
    bar_frozen_kanna = mu_bar_private(gf["f_priv_kanna_assumed"], shw)

    # confirm best-of-N does NOT help (round-trip #210 flat-in-N)
    best_of_n_helps = bool(not imp["private_clear_flat_in_n_210"] or imp["n_star_private_210"] != 1)

    return {
        "private_500_reachable_at_physical_ceiling": reachable,   # False
        "levers_assumed_fpriv": L_kanna,
        "levers_grounded_fpriv": L_obs,
        "ceiling_now": ceiling,
        "sigma_priv_now": sdr,
        "lambda_ceiling_slope_tps_per_unit": slope,               # ~221
        # the single most-actionable lever
        "most_actionable_lever": most_actionable,                 # raise_public_ceiling
        "most_actionable_magnitude_mu_ceiling_needed": most_actionable_magnitude,  # ~535 grounded
        "most_actionable_rationale": (
            "At the grounded f_priv=0.957 the levers rank: (iv) lambda_needed = "
            f"{L_obs['lever_iv_lambda_needed']:.3f} > 1 is UNPHYSICAL (lambda=1 is full self-KV "
            "recovery); (iii) sigma-shrink is USELESS because the private MEAN at the ceiling is "
            f"{L_obs['private_mean_at_ceiling']:.2f} < 500 (sigma_needed = "
            f"{L_obs['lever_iii_sigma_priv_needed_at_ceiling']:.3f} < 0; even sigma->0 misses, and "
            "freezing private clocks to sigma_hw still needs "
            f"{bar_frozen_obs:.2f}); (ii) f_priv_needed = "
            f"{L_obs['lever_ii_f_priv_needed_at_ceiling']:.5f} is ABOVE the plausible band "
            "[0.95, 0.969] (would need private drop <=1.69% vs observed 4.30%). The ONLY actionable "
            "lever is (i) RAISE THE PUBLIC CEILING to mu_ceiling_needed = "
            f"{most_actionable_magnitude:.2f} (kernel-ceiling route wirbel #216 / DVR)."
        ),
        "freeze_private_clocks_cross_check": {
            "sigma_hw": shw,
            "mu_bar_private_frozen_assumed": bar_frozen_kanna,    # 524.2 (still > 520.95)
            "mu_bar_private_frozen_observed": bar_frozen_obs,     # 530.8 (still > 520.95)
            "frozen_clears_ceiling_assumed": bool(ceiling >= bar_frozen_kanna),  # False
            "frozen_clears_ceiling_observed": bool(ceiling >= bar_frozen_obs),   # False
            "note": (
                "Freezing the PRIVATE re-benchmark's clocks (sigma_priv: sigma_draw 7.391 -> "
                "sigma_hw 4.864) LOWERS the needed ceiling but does NOT clear it: 524.2 (assumed) / "
                "530.8 (grounded) both still exceed 520.95. A partial sigma-shrink helps ~4 TPS; it "
                "is not sufficient alone."
            ),
        },
        "best_of_n_helps_private": best_of_n_helps,               # False
        "best_of_n_note": (
            "best-of-N does NOT help: #210 proved the conditional private clear is FLAT in N "
            "(n_star_private=1) -- the public selection is on non-replicating noise, so re-drawing "
            "N>1 lifts the SEEN public trigger for ZERO private-mean gain. Against the private bar "
            "BUILDING HIGHER (raising mu_pub toward mu_ceiling_needed) strictly dominates re-drawing."
        ),
    }


# ---------------------------------------------------------------------------
# step 4: sensitivity surface (SECONDARY) -- ceiling_clears over (f_priv, sigma_priv)
# ---------------------------------------------------------------------------
def sensitivity_surface(imp: dict[str, Any], gf: dict[str, Any]) -> dict[str, Any]:
    ceiling = imp["lambda1_ceiling"]
    f_lo, f_hi = F_PRIV_SURFACE
    s_lo, s_hi = SIGMA_SURFACE
    f_axis = [f_lo + (f_hi - f_lo) * i / (SURFACE_N - 1) for i in range(SURFACE_N)]
    s_axis = [s_lo + (s_hi - s_lo) * j / (SURFACE_N - 1) for j in range(SURFACE_N)]

    grid = []  # rows of bools (clears)
    n_clear = 0
    for f in f_axis:
        row = []
        for s in s_axis:
            clears = ceiling * f - Z1 * s >= TARGET
            row.append(bool(clears))
            n_clear += int(clears)
        grid.append(row)
    frac_clear = n_clear / (SURFACE_N * SURFACE_N)

    # where the #52-observed point lands (f_priv_obs, sigma_draw)
    f_obs = gf["f_priv_obs_52"]
    s_obs = imp["sigma_draw"]
    obs_clears = bool(ceiling * f_obs - Z1 * s_obs >= TARGET)

    # the f_priv break-even at the default sigma_draw (the lambda=1-suffices boundary in f)
    f_breakeven_at_sdraw = (TARGET + Z1 * s_obs) / ceiling   # 0.98312 (> 0.975 surface top)

    # the sigma break-even at the optimistic f (surface-top): largest sigma that still clears
    s_breakeven_at_fhi = (ceiling * f_hi - TARGET) / Z1      # ~4.82
    s_breakeven_at_fobs = (ceiling * f_obs - TARGET) / Z1    # NEGATIVE (never clears at f_obs)

    return {
        "f_priv_axis": [f_lo, f_hi],
        "sigma_priv_axis": [s_lo, s_hi],
        "grid_n": SURFACE_N,
        "frac_grid_clears": frac_clear,
        "observed_point": {"f_priv": f_obs, "sigma_priv": s_obs, "clears": obs_clears},  # MISS
        "f_priv_breakeven_at_sigma_draw": f_breakeven_at_sdraw,   # 0.98312 (above surface top)
        "sigma_breakeven_at_f_0.975": s_breakeven_at_fhi,         # ~4.82
        "sigma_breakeven_at_f_observed": s_breakeven_at_fobs,     # < 0 (never)
        "lambda1_suffices_region": (
            "lambda=1 (ceiling 520.95) suffices ONLY in a small high-f_priv/low-sigma corner: at "
            f"the optimistic f_priv=0.975 it needs sigma_priv <= {s_breakeven_at_fhi:.2f}; at the "
            "OBSERVED f_priv=0.957 NO sigma clears (the private mean 498.5 < 500). The #52-observed "
            "point (0.957, 7.391) is deep in the MISS region; the clears region is unreachable "
            "without raising the ceiling or cutting the private drop below the plausible band."
        ),
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# step 5: self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(imp: dict[str, Any], gf: dict[str, Any], restate: dict[str, Any],
              reach: dict[str, Any], surface: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evid: dict[str, Any] = {}
    sdr = imp["sigma_draw"]

    # (a) mu_bar_private(0.969107, sigma_draw) reproduces 528.4835555959944 EXACTLY
    a_resid = restate["reproduce_528_resid"]
    checks["a_reproduces_528_exact"] = bool(a_resid < EXACT_TOL)
    evid["a_reproduce_528_resid"] = a_resid

    # (b) monotone: DECREASING in f_priv, INCREASING in sigma_priv
    f_grid = [0.95 + 0.005 * k for k in range(6)]   # 0.95..0.975
    s_grid = [3.0 + 1.0 * k for k in range(6)]      # 3..8
    dec_in_f = all(mu_bar_private(f_grid[i], sdr) > mu_bar_private(f_grid[i + 1], sdr)
                   for i in range(len(f_grid) - 1))
    inc_in_s = all(mu_bar_private(gf["f_priv_obs_52"], s_grid[i])
                   < mu_bar_private(gf["f_priv_obs_52"], s_grid[i + 1])
                   for i in range(len(s_grid) - 1))
    checks["b_monotone_dec_in_fpriv_inc_in_sigma"] = bool(dec_in_f and inc_in_s)
    evid["b_dec_in_f"] = bool(dec_in_f)
    evid["b_inc_in_s"] = bool(inc_in_s)

    # (c) f_priv=1, sigma_priv->0 => mu_bar_private -> 500
    c_bar = mu_bar_private(1.0, 0.0)
    c_resid = abs(c_bar - TARGET)
    checks["c_limit_fpriv1_sigma0_is_500"] = bool(c_resid < EXACT_TOL)
    evid["c_limit_resid"] = c_resid

    # (d) the #52 paired point round-trips: 481.53 * f_priv_obs == 460.85
    d_resid = abs(PUB_52 * gf["f_priv_obs_52"] - PRIV_52)
    checks["d_52_paired_roundtrips"] = bool(d_resid < TOL)
    evid["d_52_roundtrip_resid"] = d_resid

    # (e) best-of-N flat: imports #210 n_star_private==1 and private_clear_flat_in_n==True
    e = (imp["n_star_private_210"] == 1 and imp["private_clear_flat_in_n_210"]
         and not reach["best_of_n_helps_private"])
    checks["e_best_of_n_flat_roundtrips_210"] = bool(e)

    # (f) NaN-clean: every reported scalar finite
    def _all_finite(obj) -> bool:
        if isinstance(obj, bool) or obj is None:
            return True
        if isinstance(obj, (int, float)):
            return math.isfinite(obj)
        if isinstance(obj, str):
            return True
        if isinstance(obj, dict):
            return all(_all_finite(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_all_finite(v) for v in obj)
        return True
    f_clean = (_all_finite(gf) and _all_finite(restate) and _all_finite(reach)
               and _all_finite(surface))
    checks["f_nan_clean"] = bool(f_clean)

    passes = all(checks.values())
    return {
        "private_bar_reachability_self_test_passes": bool(passes),
        "checks": checks,
        "evidence": evid,
        "n_checks": len(checks),
    }


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    t0 = time.time()
    imp = import_banked()
    gf = ground_f_priv(imp)
    restate = restate_bar(imp, gf)
    reach = reachability(imp, gf, restate)
    surface = sensitivity_surface(imp, gf)
    st = self_test(imp, gf, restate, reach, surface)

    # TEST metric: the public ceiling that clears 500-private at P95 under the GROUNDED f_priv
    mu_ceiling_needed = reach["most_actionable_magnitude_mu_ceiling_needed"]

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    handoff = (
        "fern #185 + ubel #222: grounded f_priv = 0.957054 (#52-observed 0.9571 vs assumed "
        "0.969107 -- OPTIMISTIC, the assumed value used only the 2.35% tree drop and omitted the "
        "4.30% decode drop the one hard paired draw shows); under it the 500-PRIVATE build target "
        f"is mu_bar_private = {restate['mu_bar_private_observed']:.2f} and the lambda=1 ceiling "
        f"520.95 MISSES it by {restate['gap_observed']:.2f} TPS (vs +7.53 at the assumed f_priv), "
        "so #217's ceiling-misses-private-bar finding IS robust (the gap WIDENS under grounding); "
        "500-private is UNREACHABLE at the physical ceiling and the binding lever is RAISING THE "
        f"PUBLIC CEILING needing mu_ceiling_needed = {mu_ceiling_needed:.2f} (lambda-lift is "
        "unphysical >1, sigma-shrink is useless because the ceiling's private mean 498.5 < 500, "
        "f_priv-lift to 0.983 is above the plausible band) -- fern should read the private bar as "
        f"mu_ceiling_needed={mu_ceiling_needed:.1f}, i.e. the kernel-ceiling route (wirbel #216 / "
        "DVR) must reach ~528 (optimistic) to ~535 (grounded), NOT 500, to win private at P95."
    )

    result = {
        "pr": 224,
        "metric_primary": "private_bar_reachability_self_test_passes",
        "metric_test": "mu_ceiling_needed",
        "private_bar_reachability_self_test_passes":
            st["private_bar_reachability_self_test_passes"],
        "mu_ceiling_needed": mu_ceiling_needed,
        # headlines
        "f_priv_grounded_point": gf["f_priv_grounded_point"],
        "f_priv_grounded_band": gf["f_priv_grounded_band"],
        "assumed_vs_observed_verdict": gf["assumed_vs_observed_verdict"],
        "finding_robust_to_fpriv": restate["finding_robust_to_fpriv"],
        "private_500_reachable_at_physical_ceiling":
            reach["private_500_reachable_at_physical_ceiling"],
        "most_actionable_lever": reach["most_actionable_lever"],
        # sections
        "ground_f_priv": gf,
        "restate_bar": restate,
        "reachability": reach,
        "sensitivity_surface": surface,
        "self_test": st,
        "import_banked": imp,
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic synthesis grounding the public->private multiplier f_priv and "
            "private variance from the banked paired evidence (the ONE hard #52 paired draw + the "
            "modeled #176 decode drop), then the private-bar REACHABILITY solve. Takes NO official "
            "draws, authorizes none. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. "
            "Imports #217/#210/#204/#191/#176 + #52 VERBATIM; does not re-derive the sigmas, the "
            "ceiling (520.95), or the bar (0.9780). DECISION-ARMS the private-bar reading; "
            "integration is fern #185's. NOT ubel #222. NOT open2. NOT a launch."
        ),
        "public_evidence_used": [
            "PR #52 (vllm_baseline, MERGED): the ONE hard paired official draw -- 481.53 PUBLIC "
            "served (128/128, PPL 2.3772) / 460.85 PRIVATE-verified VALID (Delta 4.295% <= 5%) => "
            "f_priv_obs = 0.957054.",
            "kanna #217 (trigger_reconcile, MERGED): private bar 528.48 = mu_safe/f_priv, lambda=1 "
            "ceiling 520.95, private_bar_minus_ceiling +7.53, lambda1_ceiling_clears_private_bar=0.",
            "kanna #210 (winners_curse_budget, MERGED): f_priv=0.969107=(1-drop_both)*tau_low, "
            "sigma_draw=7.391, mu_safe_fresh=512.157, decode_drop_pct=4.2946, private clear FLAT in "
            "N (n_star_private=1).",
            "stark #176 (private_adverse_skew, MERGED): modeled decode drop gt_drop=0.042946 -- "
            "corroborates f_priv_obs (1 - 0.042946 = 0.957054, the SAME measurement as #52).",
            "stark #191 (private_build_bar, MERGED): public-LCB(lambda) both-bugs map (lambda=1 -> "
            "520.95, public-bar lambda 0.9052 -> 500.0) -- the composition basis for the lambda "
            "lever; binding private bar lambda*_LCB 0.9780.",
            "Disqualify gate: private re-run VALID iff public->private drop <= 5% (#191 constant) -- "
            "the f_priv lower-bound 0.95.",
        ],
        "method": (
            "LOCAL CPU-only analytic synthesis over EXISTING MERGED results + the #52 hard paired "
            "draw; no GPU/vLLM/HF Job/submission/served-file/draw. BASELINE stays 481.53; adds 0 "
            "TPS. Greedy identity untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "peak_mem_mib": peak_mem_mib,
        "elapsed_s": round(time.time() - t0, 4),
    }
    return result


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[pbr] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="winners-curse-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/private-bar-reachability",
            group=args.wandb_group,
            tags=["private-bar-reachability", "f-priv-grounding", "private-bar", "reachability",
                  "winners-curse", "launch-trigger", "pr224"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET, "imports_pr": [217, 210, 204, 191, 176, 52]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[pbr] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[pbr] wandb disabled; skipping", flush=True)
        return
    try:
        gf = result["ground_f_priv"]
        restate = result["restate_bar"]
        reach = result["reachability"]
        surface = result["sensitivity_surface"]
        st = result["self_test"]
        L = reach["levers_grounded_fpriv"]
        La = reach["levers_assumed_fpriv"]
        flat = {
            "private_bar_reachability_self_test_passes":
                1.0 if result["private_bar_reachability_self_test_passes"] else 0.0,
            "mu_ceiling_needed": result["mu_ceiling_needed"],
            # grounding
            "f_priv_kanna_assumed": gf["f_priv_kanna_assumed"],
            "f_priv_obs_52": gf["f_priv_obs_52"],
            "drop_obs_pct_52": gf["drop_obs_pct_52"],
            "f_priv_grounded_point": gf["f_priv_grounded_point"],
            "f_priv_grounded_band_lo": gf["f_priv_grounded_band"][0],
            "f_priv_grounded_band_hi": gf["f_priv_grounded_band"][1],
            "assumed_minus_observed": gf["assumed_minus_observed"],
            "obs_matches_decode_drop_resid": gf["obs_matches_decode_drop_resid"],
            "assumed_is_optimistic": 1.0 if gf["assumed_vs_observed_verdict"] == "OPTIMISTIC" else 0.0,
            # restate
            "reproduce_528_resid": restate["reproduce_528_resid"],
            "mu_bar_private_assumed": restate["mu_bar_private_assumed"],
            "mu_bar_private_observed": restate["mu_bar_private_observed"],
            "mu_bar_private_floor": restate["mu_bar_private_floor"],
            "gap_assumed": restate["gap_assumed"],
            "gap_observed": restate["gap_observed"],
            "gap_floor": restate["gap_floor"],
            "finding_robust_to_fpriv": 1.0 if restate["finding_robust_to_fpriv"] else 0.0,
            "private_mean_at_ceiling_observed": restate["private_mean_at_ceiling_observed"],
            "private_mean_at_ceiling_below_500_observed":
                1.0 if restate["private_mean_at_ceiling_below_500_observed"] else 0.0,
            # reachability
            "private_500_reachable_at_physical_ceiling":
                1.0 if reach["private_500_reachable_at_physical_ceiling"] else 0.0,
            "lever_i_mu_ceiling_needed_grounded": L["lever_i_mu_ceiling_needed"],
            "lever_i_mu_ceiling_needed_assumed": La["lever_i_mu_ceiling_needed"],
            "lever_i_ceiling_lift_grounded": L["lever_i_ceiling_lift_tps"],
            "lever_ii_f_priv_needed_at_ceiling": L["lever_ii_f_priv_needed_at_ceiling"],
            "lever_iii_sigma_priv_needed_grounded": L["lever_iii_sigma_priv_needed_at_ceiling"],
            "lever_iii_sigma_priv_needed_assumed": La["lever_iii_sigma_priv_needed_at_ceiling"],
            "lever_iv_lambda_needed_grounded": L["lever_iv_lambda_needed"],
            "lever_iv_lambda_needed_assumed": La["lever_iv_lambda_needed"],
            "lambda_ceiling_slope": reach["lambda_ceiling_slope_tps_per_unit"],
            "best_of_n_helps_private": 1.0 if reach["best_of_n_helps_private"] else 0.0,
            "mu_bar_private_frozen_observed":
                reach["freeze_private_clocks_cross_check"]["mu_bar_private_frozen_observed"],
            # surface
            "frac_grid_clears": surface["frac_grid_clears"],
            "observed_point_clears": 1.0 if surface["observed_point"]["clears"] else 0.0,
            "f_priv_breakeven_at_sigma_draw": surface["f_priv_breakeven_at_sigma_draw"],
            "sigma_breakeven_at_f_0975": surface["sigma_breakeven_at_f_0.975"],
            # constants
            "ceiling_now": reach["ceiling_now"],
            "sigma_draw": result["import_banked"]["sigma_draw"],
            "sigma_hw": result["import_banked"]["sigma_hw"],
            "mu_safe_fresh": result["import_banked"]["mu_safe_fresh"],
            "z1": Z1,
            # provenance residuals
            "f_priv_kanna_provenance_resid": result["import_banked"]["f_priv_kanna_provenance_resid"],
            "ceiling_consistent_217_vs_191": result["import_banked"]["ceiling_consistent_217_vs_191"],
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="private_bar_reachability",
            artifact_type="winners-curse-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[pbr] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    gf = result["ground_f_priv"]
    restate = result["restate_bar"]
    reach = result["reachability"]
    surface = result["sensitivity_surface"]
    st = result["self_test"]
    L = reach["levers_grounded_fpriv"]
    print("\n[pbr] ===== PRIVATE-BAR REACHABILITY  mu_bar_private = (500 + z1*sigma)/f_priv  (PR #224) =====",
          flush=True)
    print(f"  GROUND f_priv: assumed {gf['f_priv_kanna_assumed']:.6f} (tree-only) vs OBSERVED #52 "
          f"{gf['f_priv_obs_52']:.6f} (drop {gf['drop_obs_pct_52']:.3f}%) -> {gf['assumed_vs_observed_verdict']}",
          flush=True)
    print(f"    grounded point {gf['f_priv_grounded_point']:.6f}, band "
          f"[{gf['f_priv_grounded_band'][0]:.4f}, {gf['f_priv_grounded_band'][1]:.6f}]  "
          f"(obs==1-decode_drop resid {gf['obs_matches_decode_drop_resid']:.2e})", flush=True)
    print(f"\n  RESTATE bar (ceiling {reach['ceiling_now']:.4f}):", flush=True)
    for key in ("assumed", "observed", "gate_floor"):
        r = restate["rows"][key]
        print(f"    f_priv={r['f_priv']:.6f}  mu_bar_private={r['mu_bar_private']:8.3f}  "
              f"gap={r['gap_bar_minus_ceiling']:+7.3f}  clears={r['ceiling_clears_private_bar']}  "
              f"priv_mean@ceil={r['private_mean_at_ceiling']:.2f}", flush=True)
    print(f"    reproduce_528_resid = {restate['reproduce_528_resid']:.2e}   "
          f"finding_robust_to_fpriv = {restate['finding_robust_to_fpriv']}", flush=True)
    print(f"\n  REACHABILITY (grounded f_priv {gf['f_priv_obs_52']:.4f}):", flush=True)
    print(f"    private_500_reachable_at_physical_ceiling = "
          f"{reach['private_500_reachable_at_physical_ceiling']}", flush=True)
    print(f"    (i)  mu_ceiling_needed  = {L['lever_i_mu_ceiling_needed']:.3f}  "
          f"(lift +{L['lever_i_ceiling_lift_tps']:.2f} TPS over 520.95)", flush=True)
    print(f"    (ii) f_priv_needed@ceil = {L['lever_ii_f_priv_needed_at_ceiling']:.5f}  "
          f"(above band top 0.969)", flush=True)
    print(f"    (iii)sigma_needed@ceil  = {L['lever_iii_sigma_priv_needed_at_ceiling']:.3f}  "
          f"(possible={L['lever_iii_sigma_shrink_possible']}; mean@ceil "
          f"{L['private_mean_at_ceiling']:.2f})", flush=True)
    print(f"    (iv) lambda_needed      = {L['lever_iv_lambda_needed']:.4f}  "
          f"(unphysical={L['lever_iv_lambda_unphysical']}; slope "
          f"{reach['lambda_ceiling_slope_tps_per_unit']:.1f} TPS/lambda)", flush=True)
    print(f"    most-actionable: {reach['most_actionable_lever']} -> "
          f"{reach['most_actionable_magnitude_mu_ceiling_needed']:.2f}", flush=True)
    print(f"    best_of_n_helps_private = {reach['best_of_n_helps_private']}", flush=True)
    print(f"\n  SURFACE: observed point clears = {surface['observed_point']['clears']}; "
          f"frac grid clears = {surface['frac_grid_clears']:.3f}; "
          f"f_breakeven@sigma_draw = {surface['f_priv_breakeven_at_sigma_draw']:.5f}", flush=True)
    print(f"\n  mu_ceiling_needed (TEST) = {result['mu_ceiling_needed']:.3f}", flush=True)
    print(f"  SELF-TEST (PRIMARY) passes = {st['private_bar_reachability_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "private_bar_reachability_results.json"))
    ap.add_argument("--self-test", action="store_true",
                    help="run the self-test (always runs; sets nonzero exit on failure)")
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/private-bar-reachability")
    ap.add_argument("--wandb-group", "--wandb_group", default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[pbr] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if not result["private_bar_reachability_self_test_passes"]:
        print("[pbr] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
