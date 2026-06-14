#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Launch-trigger reconcile: 512.4 (N=1) vs 528.5 (best-of-N) as ONE T(N) (PR #217).

WHAT THIS IS
------------
Capstone of the sigma-decomposition / redraw lane (#194->#202->#206->#210). Two
banked launch triggers appear to disagree:

  * ubel #204/#207 (MERGED): the public GO trigger is T_base = 512.41 central /
    514.63 worst-case (mu_pub at which the one-sided P95 LCB on the combined
    launch sigma clears 500); the lambda=1 ceiling 520.95 CLEARS it (+8.54 / +6.32).
  * kanna #210 (MERGED): against the binding 500 PRIVATE bar the public build must
    reach mu_bar_private_corrected = 528.48, and a build at mu=512.2 clears the
    private bar only 0.312 of the time.

This leg expresses BOTH in one frame, the best-of-N public GO trigger:

        T(N) = T_base + sigma_sel * E[Z_(N:N)]                       (step 1)

where sigma_sel * E[Z_(N:N)] is #210's winner's-curse order-statistic inflation
(FROZEN sigma_sel = sigma_hw = 4.864; FRESH sigma_sel = sigma_draw = 7.391;
E[Z_(N:N)] the expected max of N standard normals). At N=1 the tax is 0 (no curse)
so T(1) == #204's 512.41 / 514.63 EXACTLY; best-of-N RAISES the seen trigger.

PREMISE CORRECTION (honest flag to the advisor -- read this)
------------------------------------------------------------
The PR body states `delta_mu_winners_curse = 23.61 = sigma_sel * E[Z_(5:5)]` and
asks to show `528.48 = T(5) = T_base + 23.61`. The banked #210 artifact does NOT
support that identity; two distinct quantities were conflated:

  (i)  the order-statistic winner's-curse tax sigma_sel * E[Z_(5:5)] is SMALL:
       4.864 * 1.16296 = 5.657 TPS (frozen) / 7.391 * 1.16296 = 8.595 (fresh).
       This is #210's `winners_curse_tps_n5_{frozen,fresh}` -- reproduced here EXACTLY.
  (ii) `delta_mu_winners_curse = 23.61` is a COMPOSITE measured against the #202
       FROZEN-PUBLIC best-of-5 bar 504.873 (NOT #204's 512.41 GO trigger):
       23.61 = 7.28 [public best-of-N discount that EVAPORATES privately]
             + 16.33 [private-drop gross-up f_priv].  (#210 tax_decomposition.)

So the true round-trip is  528.48 = 504.873 + 23.61  (resid 0.0), equivalently
528.48 = mu_safe_fresh(512.157) / f_priv(0.969107). The PR's literal
T_base + 23.61 = 512.41 + 23.61 = 536.02 and T(5)_frozen = 518.07 are BOTH != 528.48.
The 512.41-vs-528.48 gap (16.07 TPS) is therefore the N-INDEPENDENT public->private
gross-up, NOT the winner's curse -- the curse is a separate, smaller, genuinely
N-dependent effect that vanishes at N=1. We report the CORRECT decomposition and
flag the PR identity as not-holding (pr_premise_t5_equals_528_holds = False), rather
than forcing a false equality.

THE N* POLICY + HARM VERDICT (step 3)
-------------------------------------
Cross the lambda=1 ceiling 520.95 against T(N):
  * n_max_clearable_at_lambda1 = largest N with T(N) <= 520.95 (per regime / rho).
    cf 15, wf 6, cF 4, wF 3 -- binding (most conservative) = 3 (worst-case fresh).
  * best_of_n_is_harmful = True: T(N) STRICTLY rises in N (raises the seen bar) while
    #210 proved the conditional private clear is FLAT in N (n_star_private=1, the
    selection is on non-replicating noise) -- so any N>1 lifts the bar for ZERO
    private-mean gain.
  * n_star_launch = 1: lowest trigger (512.41), cleared by the ceiling, no curse.

THE PRIVATE-BAR-vs-CEILING FINDING (the load-bearing correction)
---------------------------------------------------------------
Pinning N*=1 dissolves the (small) winner's curse but NOT the public-vs-private
tension. The PRIVATE-corrected build target 528.48 is N-INDEPENDENT and EXCEEDS the
physical lambda=1 ceiling 520.95 by +7.53 TPS: at the ceiling the private clear is
only 0.744 (P95 LCB 492.70 < 500). So #207's public-axis GREEN (ceiling clears the
N=1 GO trigger) and #210's private-axis RED (ceiling does NOT reach the 528.48
private build target) are BOTH right -- they are different bars, reconciled by axis
(public-confidence vs private-grade), not by N alone.

LOCAL CPU-only analytic synthesis over EXISTING MERGED results. No GPU / vLLM / HF
Job / submission / served-file / official draw. BASELINE stays 481.53; greedy/PPL
untouched; adds 0 TPS; authorizes nothing. Imports #204/#207/#210 VERBATIM; does NOT
re-derive the bar (0.9780), the sigmas, or the ceiling (520.95). Orthogonal to Issue
#192 (greedy gate). NOT open2. NOT a launch.

SELF-TEST (PRIMARY = trigger_reconcile_self_test_passes)
--------------------------------------------------------
(a) T(1) central=512.4101, worst=514.6346 EXACT (E[Z_(1:1)]=0, round-trips #204).
(b) the order-statistic tax sigma_sel*E[Z_(5:5)] round-trips #210's
    winners_curse_tps_n5 (frozen 5.657 / fresh 8.595) to tol; AND the #210 private
    round-trip 504.873 + 23.61 == 528.48 and mu_safe/f_priv == 528.48 to tol.
(c) T(N) strictly increasing in N (all four series).
(d) n_max_clearable_at_lambda1 >= 1 (N=1 always clearable) and finite, every series.
(e) FRESH tax >= FROZEN tax at every N (sigma_draw > sigma_hw).
(f) NaN-clean.
(g) E[Z_(N:N)] reproduces #210's tabulated e_max_order_stats to tol.
TEST = n_star_launch (int, expect 1).
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
UNIT_REBASE_204 = os.path.join(
    _ROOT, "research/validity/launch_sigma_unit_rebase/launch_sigma_unit_rebase_results.json")
RECONCILE_207 = os.path.join(
    _ROOT, "research/validity/launch_sigma_175_reconcile/launch_sigma_175_reconcile_results.json")
WINNERS_CURSE_210 = os.path.join(
    _ROOT, "research/validity/winners_curse_budget/winners_curse_budget_results.json")

TARGET = 500.0
P_TARGET = 0.95
Z1 = 1.6448536269514722  # one-sided P95 (#204 z1=1.64485)
N_SET = [1, 2, 3, 5, 10]  # PR step 1 trigger table
N_MAX_SEARCH = 200        # ceiling-crossing search bound
TOL = 1e-6
EXACT_TOL = 1e-9

# ---------------------------------------------------------------------------
# numerics (identical to #194/#202/#210 -- erf normal CDF + composite Simpson)
# ---------------------------------------------------------------------------
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy) -- identical to #194/#202/#210."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _simpson(f, lo: float, hi: float, n_grid: int) -> float:
    """Deterministic composite-Simpson integral of f on [lo, hi] (n_grid odd)."""
    h = (hi - lo) / (n_grid - 1)
    acc = 0.0
    for i in range(n_grid):
        x = lo + i * h
        w = 1.0 if i in (0, n_grid - 1) else (4.0 if i % 2 == 1 else 2.0)
        acc += w * f(x)
    return acc * h / 3.0


def e_max_order_stat(n: int, lo: float = -12.0, hi: float = 14.0, n_grid: int = 200001) -> float:
    """E[Z_(N:N)] = N * int z phi(z) Phi(z)^(N-1) dz (David & Nagaraja 2003).

    Identical integrator/grid to #210 so the order-statistic tax reproduces its
    winners_curse_tps_n5_{frozen,fresh} to machine tol.
    """
    if n <= 1:
        return 0.0
    return n * _simpson(lambda z: z * _norm_pdf(z, 0.0, 1.0) * _phi(z) ** (n - 1), lo, hi, n_grid)


# ---------------------------------------------------------------------------
# import the banked constants (verbatim)
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    d204 = _load(UNIT_REBASE_204)
    d207 = _load(RECONCILE_207)
    d210 = _load(WINNERS_CURSE_210)

    ct = d204["clean_trigger"]
    imp = d210["import_banked"]
    pcb = d210["private_corrected_bar"]
    wc = d210["winners_curse"]
    mono = d210["monotonicity_in_n"]

    # private clear AT the lambda=1 ceiling (mu_pub=520.95) from #210's gap_table
    p_priv_at_ceiling = None
    for r in d210["gap_table"]["rows"]:
        if abs(r["mu_pub"] - 520.95) < 1e-9 and r["n"] == 1:
            p_priv_at_ceiling = r["p_private_clear_given_trigger_frozen"]
            break

    out = {
        # ---- #204 launch_sigma_unit_rebase (the N=1 GO trigger, T_base) ----
        "t_base_central": _finite(ct["mu_clears_500_clean_central"]),          # 512.4101
        "t_base_worstcase": _finite(ct["mu_clears_500_clean_worstcase"]),      # 514.6346
        "sigma_combined_central": _finite(ct["combined_sigma_launch_clean_central"]),    # 7.5448
        "sigma_combined_worstcase": _finite(ct["combined_sigma_launch_clean_worstcase"]),  # 8.8972
        "lambda1_ceiling": _finite(ct["lambda1_ceiling_mu"]),                  # 520.9527
        "central_headroom_204": _finite(ct["central_headroom_below_ceiling_tps"]),   # 8.5426
        "worstcase_headroom_204": _finite(ct["worstcase_headroom_below_ceiling_tps"]),  # 6.3181
        # ---- #207 launch_sigma_175_reconcile (robust-YES survives) ----
        "robust_yes_survives_207": bool(d207["verdict"]["robust_yes_survives"]),
        "h_out_207": _finite(d207["readings"]["A_hout_launch_correct"]["acceptance"]["out_route_halfwidth"]),
        "trigger_central_207": _finite(d207["trigger_central_hout"]),          # 512.4101 (anchor == #204)
        # ---- #210 winners_curse_budget (the winner's-curse + private bar) ----
        "sigma_hw": _finite(imp["sigma_hw_tps"]),          # 4.8645  FROZEN sigma_sel
        "sigma_draw": _finite(imp["sigma_draw_tps"]),      # 7.3910  FRESH sigma_sel
        "mu_safe_fresh": _finite(imp["mu_safe_fresh_tps"]),       # 512.157
        "mu_bar_frozen_public_202": _finite(imp["mu_bar_frozen_p95"]),  # 504.873
        "f_priv": _finite(imp["f_priv"]),                  # 0.969107
        "lambda_star_191": _finite(imp["lambda_star_191"]),  # 0.9780 (bar)
        "mu_bar_private_corrected": _finite(pcb["mu_bar_private_corrected"]),   # 528.4836
        "delta_mu_winners_curse_composite": _finite(pcb["delta_mu_winners_curse"]),  # 23.610 COMPOSITE
        "discount_evaporates_tps": _finite(pcb["tax_decomposition"]["public_bestofN_discount_evaporates_tps"]),  # 7.284
        "private_drop_grossup_tps": _finite(pcb["tax_decomposition"]["private_drop_grossup_tps"]),  # 16.326
        "p_private_clear_at_mu512p2": _finite(pcb["p_private_clear_at_mu512p2_n1"]),  # 0.31198
        "winners_curse_tps_n5_frozen": _finite(wc["winners_curse_tps_n5_frozen"]),  # 5.657 ORDER-STAT
        "winners_curse_tps_n5_fresh": _finite(wc["winners_curse_tps_n5_fresh"]),    # 8.595 ORDER-STAT
        "n_star_private_210": int(mono["n_star_private"]),         # 1
        "private_clear_flat_in_n_210": bool(mono["private_clear_flat_in_n"]),  # True
        "e_max_order_stats_210": {int(k): _finite(v) for k, v in d210["e_max_order_stats"].items()},
        "p_private_clear_at_ceiling": _finite(p_priv_at_ceiling),  # 0.7444 at mu=520.95
    }
    return out


# ---------------------------------------------------------------------------
# step 1: the unified best-of-N public GO trigger  T(N)
# ---------------------------------------------------------------------------
def trigger_T(n: int, t_base: float, sigma_sel: float, e_max: dict[int, float]) -> float:
    return t_base + sigma_sel * e_max[n]


def build_trigger_table(imp: dict[str, Any], e_max: dict[int, float]) -> dict[str, Any]:
    tbc, tbw = imp["t_base_central"], imp["t_base_worstcase"]
    shw, sdr = imp["sigma_hw"], imp["sigma_draw"]
    rows = []
    for n in N_SET:
        e = e_max[n]
        rows.append({
            "n": n,
            "e_max_order_stat": e,
            "tax_frozen": shw * e,
            "tax_fresh": sdr * e,
            "T_central_frozen": trigger_T(n, tbc, shw, e_max),
            "T_central_fresh": trigger_T(n, tbc, sdr, e_max),
            "T_worstcase_frozen": trigger_T(n, tbw, shw, e_max),
            "T_worstcase_fresh": trigger_T(n, tbw, sdr, e_max),
        })
    return {
        "rows": rows,
        "t_base_central": tbc,
        "t_base_worstcase": tbw,
        "law": "T(N) = T_base + sigma_sel * E[Z_(N:N)]; sigma_sel = sigma_hw (FROZEN) "
               "or sigma_draw (FRESH); T_base = 500 + z1*sigma_combined (#204). "
               "T(1) = T_base (E[Z_(1:1)] = 0, no winner's curse).",
        "t1_central": rows[0]["T_central_frozen"],
        "t1_worstcase": rows[0]["T_worstcase_frozen"],
    }


# ---------------------------------------------------------------------------
# step 2: reconcile with #210's 528.48 (the CORRECT decomposition + the
#          PR-literal diagnostic showing the conflation)
# ---------------------------------------------------------------------------
def reconcile_528(imp: dict[str, Any], e_max: dict[int, float]) -> dict[str, Any]:
    priv = imp["mu_bar_private_corrected"]
    frozen202 = imp["mu_bar_frozen_public_202"]
    dmu = imp["delta_mu_winners_curse_composite"]
    safe = imp["mu_safe_fresh"]
    fpriv = imp["f_priv"]
    tbc = imp["t_base_central"]
    shw, sdr = imp["sigma_hw"], imp["sigma_draw"]

    # CORRECT round-trips
    reconcile_residual_528 = abs(frozen202 + dmu - priv)
    closed_form = safe / fpriv
    reconcile_residual_closed_form = abs(closed_form - priv)

    # the order-statistic tax really IS small (round-trips #210)
    tax_n5_frozen = shw * e_max[5]
    tax_n5_fresh = sdr * e_max[5]

    # PR-literal claims (both fail -- documents the conflation)
    pr_literal_t5_frozen_central = trigger_T(5, tbc, shw, e_max)   # 518.07
    pr_literal_tbase_plus_composite = tbc + dmu                    # 536.02
    pr_premise_t5_equals_528_holds = abs(pr_literal_t5_frozen_central - priv) < TOL

    # the 512.4-vs-528.48 gap is the N-INDEPENDENT public->private gross-up
    gap_512_to_528 = priv - tbc
    mu512_short_of_private = priv - 512.2  # 16.28 -- PR's "16 TPS too low"

    return {
        "mu_bar_private_corrected": priv,
        "reconcile_residual_528": reconcile_residual_528,
        "reconcile_identity": "528.48 = mu_bar_frozen_public_202 (504.873) + "
                              "delta_mu_winners_curse (23.610)  [resid ~ 0]",
        "reconcile_residual_closed_form": reconcile_residual_closed_form,
        "reconcile_closed_form": "528.48 = mu_safe_fresh (512.157) / f_priv (0.969107)",
        "tax_decomposition": {
            "public_bestofN_discount_evaporates_tps": imp["discount_evaporates_tps"],  # 7.284
            "private_drop_grossup_tps": imp["private_drop_grossup_tps"],                # 16.326
            "sum": imp["discount_evaporates_tps"] + imp["private_drop_grossup_tps"],
        },
        "order_stat_tax_n5_frozen": tax_n5_frozen,           # 5.657 (== #210)
        "order_stat_tax_n5_fresh": tax_n5_fresh,             # 8.595 (== #210)
        "order_stat_tax_roundtrips_210_frozen": abs(tax_n5_frozen - imp["winners_curse_tps_n5_frozen"]),
        "order_stat_tax_roundtrips_210_fresh": abs(tax_n5_fresh - imp["winners_curse_tps_n5_fresh"]),
        # --- premise correction (honest flag) ---
        "pr_premise_t5_equals_528_holds": pr_premise_t5_equals_528_holds,  # False
        "pr_literal_t5_frozen_central": pr_literal_t5_frozen_central,      # 518.07
        "pr_literal_t5_minus_528": pr_literal_t5_frozen_central - priv,    # -10.42
        "pr_literal_tbase_plus_composite": pr_literal_tbase_plus_composite,  # 536.02
        "pr_literal_tbase_plus_composite_minus_528": pr_literal_tbase_plus_composite - priv,  # +7.54
        "premise_correction": (
            "PR identity 528.48 = T_base + 23.61 (with 23.61 labelled sigma_sel*E[Z_(5:5)]) "
            "does NOT hold. (i) sigma_sel*E[Z_(5:5)] is 5.657 (frozen) / 8.595 (fresh), NOT "
            "23.61. (ii) 23.61 = delta_mu_winners_curse is a COMPOSITE relative to the #202 "
            "FROZEN-PUBLIC bar 504.873 (not #204's 512.41): 7.28 evaporating-discount + 16.33 "
            "private-grossup. True identity: 528.48 = 504.873 + 23.61 = mu_safe/f_priv. The "
            "512.41-vs-528.48 gap (16.07) is the N-INDEPENDENT public->private gross-up, not "
            "the winner's curse (which is separate, smaller, and the only N-dependent piece)."
        ),
        # --- the mu=512.2 private-clear explanation ---
        "p_private_clear_at_mu512p2": imp["p_private_clear_at_mu512p2"],  # 0.31198
        "mu512_short_of_private_target_tps": mu512_short_of_private,      # 16.28
        "mu512_private_031_explained": (
            "#210's 0.312 private clear at mu_pub=512.2 is because 512.2 sits 16.28 TPS BELOW "
            "the N-independent private build target 528.48 (private grade = one fresh draw, "
            "clear FLAT in N): a build at 512.2 has private mean 512.2*f_priv=496.4 < 500. It is "
            "NOT a failure of the N=1 PUBLIC GO trigger (512.41) -- that trigger answers a "
            "DIFFERENT question (95%-confidence the PUBLIC mean clears 500), not the private bar."
        ),
        "gap_512_to_528_is_public_to_private_grossup": gap_512_to_528,    # 16.07
    }


# ---------------------------------------------------------------------------
# step 3: the N* policy + harm verdict (cross the lambda=1 ceiling with T(N))
# ---------------------------------------------------------------------------
def n_star_policy(imp: dict[str, Any], e_max_fn) -> dict[str, Any]:
    ceiling = imp["lambda1_ceiling"]
    tbc, tbw = imp["t_base_central"], imp["t_base_worstcase"]
    shw, sdr = imp["sigma_hw"], imp["sigma_draw"]

    series = {
        "central_frozen": (tbc, shw),
        "central_fresh": (tbc, sdr),
        "worstcase_frozen": (tbw, shw),
        "worstcase_fresh": (tbw, sdr),
    }
    n_max = {}
    for name, (tb, ss) in series.items():
        last_ok = 0
        for n in range(1, N_MAX_SEARCH + 1):
            if tb + ss * e_max_fn(n) <= ceiling + 1e-12:
                last_ok = n
            else:
                break
        n_max[name] = last_ok
    n_max_binding = min(n_max.values())  # most conservative regime/rho

    # harm: T(N) strictly rises in N AND private clear flat in N (#210)
    trigger_monotone_up = all(
        e_max_fn(b) > e_max_fn(a) for a, b in zip(N_SET[:-1], N_SET[1:])
    )  # E[Z] strictly increases => every series (positive sigma_sel) strictly increases
    private_flat_in_n = imp["private_clear_flat_in_n_210"] and imp["n_star_private_210"] == 1
    best_of_n_is_harmful = bool(trigger_monotone_up and private_flat_in_n)

    return {
        "lambda1_ceiling": ceiling,
        "n_max_clearable_at_lambda1": n_max,
        "n_max_clearable_at_lambda1_binding": n_max_binding,
        "n_max_clearable_central": n_max["central_frozen"],   # reported pair (PR step 3)
        "n_max_clearable_worstcase": n_max["worstcase_frozen"],
        "best_of_n_is_harmful": best_of_n_is_harmful,
        "best_of_n_harm_rationale": (
            "T(N) STRICTLY increases in N (winner's-curse order-statistic tax sigma_sel*E[Z] "
            "moves only the SEEN public max), while #210 proved the conditional private clear is "
            "FLAT in N (n_star_private=1; the selection is on non-replicating noise). So N>1 lifts "
            "the bar you must hit (toward/past the 520.95 ceiling) for ZERO private-mean gain."
        ),
        "trigger_monotone_up": trigger_monotone_up,
        "private_clear_flat_in_n": private_flat_in_n,
        "n_star_launch": 1,
        "n_star_rationale": (
            "N*=1: T(1)=512.41 is the lowest trigger, the lambda=1 ceiling 520.95 clears it "
            "(+8.54 central / +6.32 worst), there is no winner's curse (E[Z_(1:1)]=0), and "
            "best-of-N adds zero private clear. Any N>1 is dominated."
        ),
    }


# ---------------------------------------------------------------------------
# the load-bearing finding: the lambda=1 ceiling does NOT clear the PRIVATE bar
# ---------------------------------------------------------------------------
def private_bar_vs_ceiling(imp: dict[str, Any]) -> dict[str, Any]:
    ceiling = imp["lambda1_ceiling"]
    priv = imp["mu_bar_private_corrected"]
    fpriv = imp["f_priv"]
    sdr = imp["sigma_draw"]
    clears = ceiling >= priv
    priv_mean_at_ceiling = ceiling * fpriv
    priv_p95_lcb_at_ceiling = priv_mean_at_ceiling - Z1 * sdr
    return {
        "lambda1_ceiling_clears_private_bar": bool(clears),  # False
        "private_bar_minus_ceiling": priv - ceiling,         # +7.53
        "private_mean_at_ceiling": priv_mean_at_ceiling,     # 504.86
        "private_p95_lcb_at_ceiling": priv_p95_lcb_at_ceiling,  # 492.70 < 500
        "p_private_clear_at_ceiling": imp["p_private_clear_at_ceiling"],  # 0.7444 < 0.95
        "interpretation": (
            "On the PUBLIC-confidence axis (#207) the lambda=1 ceiling 520.95 CLEARS the N=1 GO "
            "trigger 512.41 (GREEN). On the PRIVATE-bar axis (#210) the SAME ceiling does NOT "
            "reach the N-independent private build target 528.48 (gap +7.53; private clear only "
            "0.744 < 0.95 at the ceiling). The #207-vs-#210 tension is reconciled by AXIS "
            "(public-confidence vs private-grade), not by the N-policy alone: pinning N*=1 "
            "removes the (small) winner's curse but the dominant public->private gross-up is "
            "N-INDEPENDENT and exceeds the physical ceiling. The launch fires GO at N=1 on the "
            "public reading yet does NOT clear the private bar at P95 even at full self-KV recovery."
        ),
    }


# ---------------------------------------------------------------------------
# step 4: self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(imp, e_max, table, recon, policy, e_max_fn) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evid: dict[str, Any] = {}

    # (a) T(1) reproduces #204 exactly (both regimes; tax 0)
    t1cf = table["rows"][0]["T_central_frozen"]
    t1cF = table["rows"][0]["T_central_fresh"]
    t1wf = table["rows"][0]["T_worstcase_frozen"]
    t1wF = table["rows"][0]["T_worstcase_fresh"]
    a_central = abs(t1cf - imp["t_base_central"]) < EXACT_TOL and abs(t1cF - imp["t_base_central"]) < EXACT_TOL
    a_worst = abs(t1wf - imp["t_base_worstcase"]) < EXACT_TOL and abs(t1wF - imp["t_base_worstcase"]) < EXACT_TOL
    checks["a_t1_reproduces_204_exact"] = bool(a_central and a_worst)
    evid["a_t1_central_err"] = abs(t1cf - imp["t_base_central"])
    evid["a_t1_worstcase_err"] = abs(t1wf - imp["t_base_worstcase"])

    # (b) order-stat tax round-trips #210 + #210 private round-trips
    b_tax = (recon["order_stat_tax_roundtrips_210_frozen"] < TOL
             and recon["order_stat_tax_roundtrips_210_fresh"] < TOL)
    b_priv = (recon["reconcile_residual_528"] < TOL
              and recon["reconcile_residual_closed_form"] < TOL)
    checks["b_tax_and_private_roundtrip_210"] = bool(b_tax and b_priv)
    evid["b_order_stat_tax_err_frozen"] = recon["order_stat_tax_roundtrips_210_frozen"]
    evid["b_order_stat_tax_err_fresh"] = recon["order_stat_tax_roundtrips_210_fresh"]
    evid["b_reconcile_residual_528"] = recon["reconcile_residual_528"]
    evid["b_reconcile_residual_closed_form"] = recon["reconcile_residual_closed_form"]

    # (c) T(N) strictly increasing in N (all four series)
    def _strict_up(key: str) -> bool:
        vals = [r[key] for r in table["rows"]]
        return all(b > a for a, b in zip(vals[:-1], vals[1:]))
    c = all(_strict_up(k) for k in
            ("T_central_frozen", "T_central_fresh", "T_worstcase_frozen", "T_worstcase_fresh"))
    checks["c_trigger_strictly_increasing_in_n"] = bool(c)

    # (d) n_max >= 1 (N=1 always clearable) and finite, every series
    d = all(1 <= v < N_MAX_SEARCH for v in policy["n_max_clearable_at_lambda1"].values())
    checks["d_n_max_ge_1_and_finite"] = bool(d)
    evid["d_n_max"] = policy["n_max_clearable_at_lambda1"]

    # (e) FRESH tax >= FROZEN tax at every N (sigma_draw > sigma_hw)
    e = all(r["tax_fresh"] >= r["tax_frozen"] - 1e-15 for r in table["rows"])
    checks["e_fresh_tax_ge_frozen_tax"] = bool(e)

    # (f) NaN-clean: every reported scalar finite
    def _all_finite(obj) -> bool:
        if isinstance(obj, bool):
            return True
        if isinstance(obj, (int, float)):
            return math.isfinite(obj)
        if isinstance(obj, dict):
            return all(_all_finite(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_all_finite(v) for v in obj)
        return True
    f = _all_finite(table) and _all_finite(recon) and _all_finite(policy)
    checks["f_nan_clean"] = bool(f)

    # (g) E[Z_(N:N)] reproduces #210's tabulated e_max_order_stats
    g_err = 0.0
    for n, ref in imp["e_max_order_stats_210"].items():
        if n in e_max:
            g_err = max(g_err, abs(e_max[n] - ref))
    checks["g_e_max_reproduces_210"] = bool(g_err < TOL)
    evid["g_e_max_max_abs_err_vs_210"] = g_err

    passes = all(checks.values())
    return {
        "trigger_reconcile_self_test_passes": bool(passes),
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
    e_max = {n: e_max_order_stat(n) for n in sorted(set(N_SET + list(imp["e_max_order_stats_210"].keys())))}

    table = build_trigger_table(imp, e_max)
    recon = reconcile_528(imp, e_max)
    policy = n_star_policy(imp, e_max_order_stat)
    pbc = private_bar_vs_ceiling(imp)
    st = self_test(imp, e_max, table, recon, policy, e_max_order_stat)

    # anchor: T_base round-trips 500 + z1*sigma_combined (convention check)
    anchor_central_err = abs((TARGET + Z1 * imp["sigma_combined_central"]) - imp["t_base_central"])
    anchor_worstcase_err = abs((TARGET + Z1 * imp["sigma_combined_worstcase"]) - imp["t_base_worstcase"])

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    result = {
        "pr": 217,
        "metric_primary": "trigger_reconcile_self_test_passes",
        "metric_test": "n_star_launch",
        "trigger_reconcile_self_test_passes": st["trigger_reconcile_self_test_passes"],
        "n_star_launch": policy["n_star_launch"],
        # the unified trigger
        "trigger_vs_N": table,
        # reconciliation with 528.48 (+ premise correction)
        "reconcile_528": recon,
        # N* policy + harm verdict
        "n_star_policy": policy,
        # the load-bearing finding
        "private_bar_vs_ceiling": pbc,
        # self-test
        "self_test": st,
        # imported constants (provenance)
        "import_banked": imp,
        "e_max_order_stats": {int(n): e_max[n] for n in e_max},
        "anchors": {
            "anchor_t_base_central_err": anchor_central_err,
            "anchor_t_base_worstcase_err": anchor_worstcase_err,
            "z1": Z1,
            "robust_yes_survives_207": imp["robust_yes_survives_207"],
        },
        "handoff": (
            "fern #185: the launch GO trigger is T(N) = 512.41 + sigma_sel*E[Z_(N:N)] (central; "
            "514.63 worst-case), reproducing #204's 512.41 at N=1 (tax 0) and #210's small "
            "order-statistic winner's curse (5.66 TPS frozen / 8.60 fresh at N=5); the lambda=1 "
            "ceiling 520.95 clears the PUBLIC trigger T(N) for N <= n_max (central 15 / worst 6 "
            "frozen; central 4 / worst 3 fresh -- bind N<=3), so fire GO at the N=1 trigger "
            "512.41/514.63 with the pinned policy N*=1 (best-of-N is HARMFUL: it raises the seen "
            "trigger for ZERO private-mean gain, #210 private-clear flat in N). CRUCIAL: the "
            "PRIVATE-corrected build target 528.48 is N-INDEPENDENT (= mu_safe/f_priv) and EXCEEDS "
            "the ceiling 520.95 by +7.53 TPS (private clear only 0.744 at the ceiling), so the "
            "#207-vs-#210 tension resolves by AXIS (public-confidence vs private-grade), not by "
            "the N-policy alone -- the N=1 GO fires on the public reading but the private bar is "
            "NOT cleared at P95 even at full self-KV recovery. PR's literal 528.48 = T_base+23.61 "
            "does NOT hold (23.61 is a composite vs the #202 frozen-public bar 504.873, not the "
            "order-statistic tax 5.66) -- see reconcile_528.premise_correction."
        ),
        "scope": (
            "LOCAL CPU-only analytic reconciliation of two banked launch triggers (#204/#207 GO "
            "trigger 512.41/514.63 and #210 private bar 528.48) as one function of N. Takes NO "
            "official draws, authorizes none. BASELINE stays 481.53; adds 0 TPS; greedy/PPL "
            "untouched. Imports #204/#207/#210 VERBATIM; does not re-derive the bar (0.9780), the "
            "sigmas, or the ceiling (520.95). Orthogonal to Issue #192 (greedy gate). NOT open2. "
            "NOT a launch."
        ),
        "public_evidence_used": [
            "ubel #204 (launch_sigma_unit_rebase, MERGED): clean GO trigger 512.41 central / "
            "514.63 worst-case, combined sigma 7.5448/8.8972, lambda=1 ceiling 520.95, z1=1.64485.",
            "ubel #207 (launch_sigma_175_reconcile, MERGED): launch-correct h_out reading 5.178 "
            "confirms the 512.41/514.63 trigger; robust_yes_survives=True.",
            "kanna #210 (winners_curse_budget, MERGED): order-statistic tax sigma_sel*E[Z_(N:N)] "
            "(5.657 frozen / 8.595 fresh at N=5), private-corrected build target 528.48 "
            "(=mu_safe/f_priv), conditional private clear FLAT in N (n_star_private=1), 0.312 at "
            "mu=512.2, tax_decomposition 7.28+16.33 vs the #202 frozen-public bar 504.873.",
            "Winner's curse / optimizer's curse: Capen-Clapp-Campbell (1971); Smith & Winkler "
            "(2006) Mgmt Sci 52(3) Prop. 1; order statistics E[Z_(N:N)] David & Nagaraja (2003).",
            "stark #191 binding private bar lambda*_LCB 0.9780 -- the launch is graded privately.",
        ],
        "method": (
            "LOCAL CPU-only analytic synthesis over EXISTING MERGED results; no GPU/vLLM/HF Job/"
            "submission/served-file/draw. BASELINE stays 481.53; adds 0 TPS. Greedy identity "
            "untouched. NOT a launch."
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
        print(f"[tr] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="winners-curse-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/trigger-reconcile",
            group=args.wandb_group,
            tags=["trigger-reconcile", "launch-trigger", "winners-curse", "best-of-n",
                  "order-statistics", "private-bar", "n-policy", "pr217"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "p_target": P_TARGET, "imports_pr": [204, 207, 210, 191]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[tr] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[tr] wandb disabled; skipping", flush=True)
        return
    try:
        imp = result["import_banked"]
        recon = result["reconcile_528"]
        policy = result["n_star_policy"]
        pbc = result["private_bar_vs_ceiling"]
        st = result["self_test"]
        flat = {
            "trigger_reconcile_self_test_passes":
                1.0 if result["trigger_reconcile_self_test_passes"] else 0.0,
            "n_star_launch": float(result["n_star_launch"]),
            # T(N) table (all four series)
            "t1_central": result["trigger_vs_N"]["t1_central"],
            "t1_worstcase": result["trigger_vs_N"]["t1_worstcase"],
            # reconciliation
            "mu_bar_private_corrected": recon["mu_bar_private_corrected"],
            "reconcile_residual_528": recon["reconcile_residual_528"],
            "reconcile_residual_closed_form": recon["reconcile_residual_closed_form"],
            "order_stat_tax_n5_frozen": recon["order_stat_tax_n5_frozen"],
            "order_stat_tax_n5_fresh": recon["order_stat_tax_n5_fresh"],
            "delta_mu_winners_curse_composite": imp["delta_mu_winners_curse_composite"],
            "discount_evaporates_tps": recon["tax_decomposition"]["public_bestofN_discount_evaporates_tps"],
            "private_drop_grossup_tps": recon["tax_decomposition"]["private_drop_grossup_tps"],
            "gap_512_to_528_grossup": recon["gap_512_to_528_is_public_to_private_grossup"],
            "pr_premise_t5_equals_528_holds": 1.0 if recon["pr_premise_t5_equals_528_holds"] else 0.0,
            "pr_literal_t5_frozen_central": recon["pr_literal_t5_frozen_central"],
            "pr_literal_t5_minus_528": recon["pr_literal_t5_minus_528"],
            "pr_literal_tbase_plus_composite": recon["pr_literal_tbase_plus_composite"],
            "p_private_clear_at_mu512p2": recon["p_private_clear_at_mu512p2"],
            "mu512_short_of_private_target_tps": recon["mu512_short_of_private_target_tps"],
            # N* policy
            "n_max_clearable_central_frozen": float(policy["n_max_clearable_at_lambda1"]["central_frozen"]),
            "n_max_clearable_central_fresh": float(policy["n_max_clearable_at_lambda1"]["central_fresh"]),
            "n_max_clearable_worstcase_frozen": float(policy["n_max_clearable_at_lambda1"]["worstcase_frozen"]),
            "n_max_clearable_worstcase_fresh": float(policy["n_max_clearable_at_lambda1"]["worstcase_fresh"]),
            "n_max_clearable_binding": float(policy["n_max_clearable_at_lambda1_binding"]),
            "best_of_n_is_harmful": 1.0 if policy["best_of_n_is_harmful"] else 0.0,
            # private-bar-vs-ceiling finding
            "lambda1_ceiling_clears_private_bar":
                1.0 if pbc["lambda1_ceiling_clears_private_bar"] else 0.0,
            "private_bar_minus_ceiling": pbc["private_bar_minus_ceiling"],
            "private_p95_lcb_at_ceiling": pbc["private_p95_lcb_at_ceiling"],
            "p_private_clear_at_ceiling": pbc["p_private_clear_at_ceiling"],
            # constants
            "lambda1_ceiling": imp["lambda1_ceiling"],
            "t_base_central": imp["t_base_central"],
            "t_base_worstcase": imp["t_base_worstcase"],
            "sigma_hw": imp["sigma_hw"],
            "sigma_draw": imp["sigma_draw"],
            "f_priv": imp["f_priv"],
            "lambda_star_191": imp["lambda_star_191"],
            "g_e_max_max_abs_err_vs_210": st["evidence"]["g_e_max_max_abs_err_vs_210"],
            "anchor_t_base_central_err": result["anchors"]["anchor_t_base_central_err"],
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
        }
        # per-N trigger curve
        for r in result["trigger_vs_N"]["rows"]:
            n = r["n"]
            flat[f"e_max_order_stat_n{n}"] = r["e_max_order_stat"]
            flat[f"T_central_frozen_n{n}"] = r["T_central_frozen"]
            flat[f"T_central_fresh_n{n}"] = r["T_central_fresh"]
            flat[f"T_worstcase_frozen_n{n}"] = r["T_worstcase_frozen"]
            flat[f"T_worstcase_fresh_n{n}"] = r["T_worstcase_fresh"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="trigger_reconcile", artifact_type="winners-curse-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[tr] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    imp = result["import_banked"]
    table = result["trigger_vs_N"]
    recon = result["reconcile_528"]
    policy = result["n_star_policy"]
    pbc = result["private_bar_vs_ceiling"]
    st = result["self_test"]
    print("\n[tr] ===== LAUNCH-TRIGGER RECONCILE  T(N) = T_base + sigma_sel*E[Z_(N:N)]  (PR #217) =====",
          flush=True)
    print(f"  T_base = {imp['t_base_central']:.4f} central / {imp['t_base_worstcase']:.4f} worst  "
          f"(#204);  lambda=1 ceiling = {imp['lambda1_ceiling']:.4f}", flush=True)
    print(f"  sigma_sel: FROZEN sigma_hw={imp['sigma_hw']:.4f}  FRESH sigma_draw={imp['sigma_draw']:.4f}",
          flush=True)
    print("\n  T(N)  [central frozen / central fresh / worst frozen / worst fresh]:", flush=True)
    for r in table["rows"]:
        print(f"    N={r['n']:>2}  E[Z]={r['e_max_order_stat']:.4f}   "
              f"{r['T_central_frozen']:8.3f} / {r['T_central_fresh']:8.3f} / "
              f"{r['T_worstcase_frozen']:8.3f} / {r['T_worstcase_fresh']:8.3f}", flush=True)
    print(f"\n  RECONCILE 528.48:  504.873 + 23.610 = {imp['mu_bar_frozen_public_202'] + imp['delta_mu_winners_curse_composite']:.4f}  "
          f"(resid {recon['reconcile_residual_528']:.2e});  mu_safe/f_priv resid {recon['reconcile_residual_closed_form']:.2e}",
          flush=True)
    print(f"  order-stat tax @N=5: frozen {recon['order_stat_tax_n5_frozen']:.4f} / "
          f"fresh {recon['order_stat_tax_n5_fresh']:.4f}  (== #210)", flush=True)
    print(f"  PR PREMISE 528.48 = T(5):  HOLDS = {recon['pr_premise_t5_equals_528_holds']}  "
          f"(T(5)_frozen={recon['pr_literal_t5_frozen_central']:.3f} -> {recon['pr_literal_t5_minus_528']:+.3f}; "
          f"T_base+23.61={recon['pr_literal_tbase_plus_composite']:.3f} -> "
          f"{recon['pr_literal_tbase_plus_composite_minus_528']:+.3f})", flush=True)
    print(f"\n  n_max_clearable_at_lambda1: {policy['n_max_clearable_at_lambda1']}  "
          f"(binding {policy['n_max_clearable_at_lambda1_binding']})", flush=True)
    print(f"  best_of_n_is_harmful = {policy['best_of_n_is_harmful']}   n_star_launch = {policy['n_star_launch']}",
          flush=True)
    print(f"\n  PRIVATE BAR vs CEILING:  lambda=1 ceiling clears private bar = "
          f"{pbc['lambda1_ceiling_clears_private_bar']}  (528.48 - 520.95 = "
          f"{pbc['private_bar_minus_ceiling']:+.3f}; private clear @ceiling {pbc['p_private_clear_at_ceiling']:.4f}; "
          f"P95 LCB {pbc['private_p95_lcb_at_ceiling']:.3f} < 500)", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['trigger_reconcile_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "trigger_reconcile_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/trigger-reconcile")
    ap.add_argument("--wandb-group", "--wandb_group", default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[tr] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if not result["trigger_reconcile_self_test_passes"]:
        print("[tr] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
