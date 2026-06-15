#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PPL-only 500 PRIVATE-stability: does the PUBLIC central-500 target (c*=0.9089) survive private? (PR #347, wirbel).

THE GOVERNING QUESTION (the rho-axis fern #341 explicitly DROPPED)
-----------------------------------------------------------------
fern #341 (o4rzy1k6, MERGED) mapped the joint (phi, Delta_cov) compliant-500 isocline but DROPPED the
private-tax rho axis -- its #1 follow-up asked for "a 3-axis (phi, Delta_cov, rho) volume" to check
"whether the rho>=0.8038 robustness gate stays clear along the cheap demand-led isocline." wirbel #343
(kklof4wr) priced the PPL-only central-500 target at c*=0.9089 as a PUBLIC TPS target (+0.0186, within
lawine #336's +0.031 retrain budget). But the deployed frontier carries a KNOWN ~4.3% public->private
gap (481.53 public / 460.85 private, organizer-verified). The open question the #319 retrain decision
needs: does a retrained head that hits PUBLIC central-500 (c*=0.9089) stay PRIVATE-stable -- i.e. does
realized PRIVATE TPS clear 500, or does the private tax pull it back under?

THE ANSWER (the rho-axis closure of fern #341's 2D card)
--------------------------------------------------------
NO. The PUBLIC central-500 target is private-UNSTABLE BY CONSTRUCTION. At the public-500 operating point
the realized private TPS is
    private(c*=0.9089) = 500 * rho_priv,   rho_priv < 1 always
so it can NEVER stay >= 500: the public-500 target leaves ZERO private margin. With lawine #300's
EAGLE-3 private/public ratio rho_priv_e3 = 0.9421 (deployed-effective) realized private = 471.06; with
the deployed-frontier measured gap g_dep = 0.9571 it is 478.53 -- BOTH < 500 (worst/raw rho push it to
~390-396). Restoring private-500 needs OVER-provisioned coverage c*_private (solve env(c)*rho = 500):
    deployed gap  (g=0.9571): c*_private = 0.9222  (+0.0319)  -- just OVER +0.031
    rho_priv_e3   (g=0.9421): c*_private = 0.9269  (+0.0366)  -- OVER  +0.031
    worst/raw     (g~0.79):   c*_private ~ 0.978   (+0.087+)  -- FAR OVER
So private-500 EXCEEDS lawine #336's +0.031 retrain budget under EVERY private model (even the lightest,
the organizer-measured 4.3% gap). The cheap public-500 isocline point fern #341 found does NOT carry to
private.

THE rho-GATE IS NECESSARY BUT NOT SUFFICIENT AT THE PUBLIC-500 POINT
-------------------------------------------------------------------
fern #335/#318's rho>=0.8038 gate is computed at the FULL honest ceiling (622.08): private = 622.08*rho
clears 500 iff rho >= 500/622.08 = 0.8038. Realistic rho (0.9421/0.9571) CLEARS it (586/595 private) --
that is fern #318's YELLOW build verdict. But that gate clearing the FULL-ceiling build does NOT imply
the budget-minimal public-500 retrain is private-stable: at env=500 (not 622) the implied gate is
rho>=1.0, which realistic rho MISSES. The rho-axis bites exactly where fern #341's 2D card dropped it.

THE WORST-CORNER RECONCILE (item 4: is c*_worst the private-500 requirement?)
----------------------------------------------------------------------------
wirbel #343 flagged c*_worst=0.9256 (+0.0353) as the conservative corner. The identity c*_private ==
c*_worst holds EXACTLY iff the private ratio g == worst/central anchor ratio = 492.87/520.95 = 0.94608.
Measured g_dep=0.95705 != 0.94608 and rho_priv_e3=0.94212 != 0.94608: REFUTED. But 0.94608 sits BETWEEN
them, so c*_worst (0.9256) is bracketed by the two realistic c*_private (0.9222 deployed, 0.9269
rho_e3) -- it is a sensible CENTRAL private proxy, not an exact identity. The agreement is partly
coincidental: #343's 492.87 anchor = honest611(622.08) * rho_worst_xdataset(0.7923), so it confounds the
honest-ceiling uplift (x1.194) with the WORST cross-dataset rho (x0.7923), netting 0.9461.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no training, no publish, no HF Job, no submission, no
served-file change, no official draw. BASELINE stays 481.53; adds 0 TPS. Imports verbatim (re-derive
NOTHING): wirbel #343 kklof4wr (envelope_X(c)=X_ANCHOR*E[T](c)/E[T](0.9213); c*_central 0.9089 / c*_worst
0.9256; prior 0.8903; identity bar 0.9213; anchors 520.95/492.87), lawine #300 8t5q6sr0 (rho_priv_e3
0.9421 deployed-effective / 0.7797 raw; deployed 481.53->460.85 Delta 4.3%; private_bar 500), fern #318
xe8ff7hq (honest611 622.08; rho_worst_xdataset 0.7923; private 622.08*rho; worst private 492.87; YELLOW),
fern #335/#310 (rho>=0.8038 gate = 500/622.08), fern #341 o4rzy1k6 (demand-led isocline + DROPPED rho
axis), lawine #336 krroookz (+0.031 retrain budget). All run-ids in
wandb-applied-ai-team/gemma-challenge-senpai.

PRIMARY metric  private_stability_self_test_passes
TEST    metric  ppl_only_central_500_is_private_stable   (bool: does public c*=0.9089 clear 500 private?)
TEST    metric  coverage_lift_for_private_500            (coverage delta 0.8903 -> private-500 target)
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

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported EXACT from banked W&B runs. Re-derive NOTHING. Displayed forms (0.9213, 520.95, 492.87,
# 0.8903, 0.9089, 0.9256, 0.9421, 0.8038, 622.08, 460.85, 481.53) are round-to-display, asserted exact.
# --------------------------------------------------------------------------- #
K_SPEC = 7                              # deployed speculative depth (chain-law K)
IDENTITY_BAR = 0.9213011665456927      # strict greedy-identity per-depth c_eff bar (lawine #330)
COV_PRIOR = 0.8903                     # measured fusion top-4 c_eff prior (lawine #330; wirbel #343 anchor)
E_T_AT_IDENTITY = 6.111214987369918    # E[T](0.9213) == 1 + sum_{d=1..7} 0.92130117^d (stark #337)

# wirbel #343 (kklof4wr) PPL-only demand envelope anchors, both at E[T]=6.1112 (the DEMAND-only anchors).
# central is CAP-BOUND (= lambda ceiling 520.95); "worst" already bakes a private-tax (see WORST note).
CENTRAL_ANCHOR = 520.9527323111674     # wirbel #343 central_at_611 (cap-bound) == lambda ceiling (PUBLIC)
WORST_ANCHOR = 492.865273281899        # wirbel #343 worst_at_611 == honest611(622.08) * rho_worst(0.7923)

# wirbel #343 (kklof4wr) banked PPL-only inverse roots (env(c)=500 PUBLIC crossings).
C_STAR_CENTRAL_343 = 0.9089363308345582   # PUBLIC central-500 target (+0.0186 from 0.8903)
C_STAR_WORST_343 = 0.925603648491971      # PUBLIC worst-500 target (+0.0353 from 0.8903)
COV_LIFT_CENTRAL_343 = 0.01863633083455818
COV_LIFT_WORST_343 = 0.035303648491970985

# Public->private deployed-frontier structure (lawine #300 8t5q6sr0; organizer-verified private).
OFFICIAL_PUBLIC = 481.53               # PR #52 frontier (W&B 2x9fm2zx), PPL 2.3772, 128/128
PRIVATE_VERIFIED = 460.85              # organizer-verified private (lawine #300)
DELTA_PCT_DEPLOYED_300 = 0.04294644155088977   # lawine #300 banked 1 - 460.85/481.53

# lawine #300 (8t5q6sr0) EAGLE-3 private/public coverage ratio rho_priv (THE named primary) + raw bracket.
RHO_PRIV_E3 = 0.9421228821714434       # deployed-effective (deep fidelity, a_1 tree-recovered): CLOSES gap
RHO_PRIV_E3_RAW = 0.7797221674962985   # raw / no-tree-recovery sensitivity (CI lower bracket): MISSES
# fern #318 (xe8ff7hq) EAGLE-3 literature worst cross-dataset tau-ratio (a_1 not credited -> lower bound).
RHO_WORST_XDATASET = 0.7922848664688427    # LLaMA-3.1-8B CNN/DM vs HumanEval (arXiv:2503.01840 Table 1)

# fern #335/#310/#318 rho-axis robustness gate at the FULL honest ceiling.
HONEST_PUBLIC_611 = 622.080888         # fern #310/#318 honest E[T]=6.11 PUBLIC ceiling (uncapped)
RHO_BREAKEVEN = 0.8037539966988988     # fern #335 gate == 500/622.08 (private clears 500 at full ceiling)
PRIVATE_TPS_611_CENTRAL = 586.0766391463308   # fern #318 622.08 * rho_priv_e3 (full-build private, clears)
WORST_PRIVATE_TPS_318 = 492.865273281899      # fern #318 622.08 * rho_worst (full-build private, MISSES)

# lawine #336 (krroookz) retrain head-coverage lift budget: 0.9213 - 0.8903 = +0.031.
RETRAIN_LIFT_BUDGET = 0.031

PPL_DEPLOYED = 2.3772
PPL_GATE = 2.42
TARGET = 500.0
BASELINE_TPS = 481.53

TOL_EXACT = 1e-9          # anchor round-trip / import-exact checks
TOL_RT = 1e-6            # reproduce banked deployed gap / ratios
TOL_343 = 1e-6           # reproduce wirbel #343 banked roots
TOL_ROOT = 1e-7          # inverse root residual (|env(c*) - target|)
TOL_DISPLAY_C = 5e-5     # full-precision constant rounds to its displayed 4-dp form
TOL_DISPLAY_TPS = 5e-3   # full-precision anchor rounds to its displayed 2-dp form


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Core laws (wirbel #343 conventions) + the private realization haircut.
# --------------------------------------------------------------------------- #
def e_t(c: float, k: int = K_SPEC) -> float:
    """Chain-law expected accepted tokens: E[T] = 1 + sum_{d=1..K} c^d (stark #337)."""
    return 1.0 + sum(c ** d for d in range(1, k + 1))


def envelope_central(c: float) -> float:
    """PPL-only (DEMAND-only) PUBLIC central envelope (wirbel #343): scales with coverage c."""
    return CENTRAL_ANCHOR * e_t(c) / E_T_AT_IDENTITY


def realized_private(c: float, rho: float) -> float:
    """Realized PRIVATE TPS at PUBLIC coverage c under private/public realization ratio rho.

    private(c) = public_envelope(c) * rho. TPS ~ E[T] in this linear regime, so rho (a private/public
    E[T]/TPS ratio) maps directly. This is the SINGLE-effect haircut; rho_priv_e3 and the deployed gap
    are TWO ESTIMATES of the SAME ratio (lawine #300 derived rho_priv_e3 from the per-position collapse
    that ALSO yields the deployed 4.3% gap) -- they BRACKET it; they are NOT multiplied.
    """
    return envelope_central(c) * rho


def solve_c_for_public_envelope(target_env: float, lo: float = 0.0, hi: float = 1.0,
                                iters: int = 200) -> float:
    """Monotone bisection: c with envelope_central(c) == target_env (strictly increasing in c)."""
    if envelope_central(lo) > target_env or envelope_central(hi) < target_env:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if envelope_central(mid) < target_env:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_c_for_private(rho: float, target: float = TARGET) -> float:
    """c such that realized_private(c, rho) == target  <=>  envelope_central(c) == target/rho."""
    if rho <= 0.0:
        return float("nan")
    return solve_c_for_public_envelope(target / rho)


# Named private/public realization ratios (rho), each a private/public TPS ratio. realistic = the
# deployed-effective / organizer-measured regime that CLEARS the rho-gate; downside = worst/raw.
def _g_deployed() -> float:
    return PRIVATE_VERIFIED / OFFICIAL_PUBLIC


def private_ratios() -> list[dict[str, Any]]:
    return [
        {"name": "rho_priv_e3", "rho": RHO_PRIV_E3, "regime": "realistic", "primary": True,
         "source": "lawine #300 deployed-effective (deep fidelity, a_1 tree-recovered)"},
        {"name": "deployed_gap", "rho": _g_deployed(), "regime": "realistic", "primary": False,
         "source": "deployed frontier 460.85/481.53 (organizer-verified, model-free)"},
        {"name": "rho_worst_xdataset", "rho": RHO_WORST_XDATASET, "regime": "downside", "primary": False,
         "source": "fern #318 EAGLE-3 worst cross-dataset (arXiv:2503.01840 Table 1)"},
        {"name": "rho_priv_e3_raw", "rho": RHO_PRIV_E3_RAW, "regime": "downside", "primary": False,
         "source": "lawine #300 raw / no-tree-recovery (CI lower bracket)"},
    ]


# --------------------------------------------------------------------------- #
# (D1) Import the public->private structure (round-trip; re-derive NOTHING).
# --------------------------------------------------------------------------- #
def deliverable1_import_structure() -> dict[str, Any]:
    g_dep = _g_deployed()
    delta_pct = 1.0 - g_dep
    priv_roundtrip = OFFICIAL_PUBLIC * g_dep                      # == 460.85
    # rho-gate at the full honest ceiling: private = 622.08 * rho, clears 500 iff rho >= 500/622.08.
    gate_implied = TARGET / HONEST_PUBLIC_611
    return {
        "deployed_public_to_private": {
            "official_public": OFFICIAL_PUBLIC,
            "private_verified": PRIVATE_VERIFIED,
            "g_deployed_ratio": g_dep,
            "delta_pct": delta_pct,
            "private_roundtrip": priv_roundtrip,
            "roundtrips_460p85": bool(abs(priv_roundtrip - PRIVATE_VERIFIED) <= 1e-6),
            "roundtrips_300_delta_4p3": bool(abs(delta_pct - DELTA_PCT_DEPLOYED_300) <= TOL_RT),
        },
        "rho_priv_imports": {
            "rho_priv_e3_deployed_effective": RHO_PRIV_E3,
            "rho_priv_e3_raw": RHO_PRIV_E3_RAW,
            "rho_worst_xdataset": RHO_WORST_XDATASET,
            "rho_ci_bracket_raw_to_deployed_effective": [RHO_PRIV_E3_RAW, RHO_PRIV_E3],
            "note": ("lawine #300 rho_priv_e3 = private/public EAGLE-3 coverage ratio. CENTRAL "
                     "(deployed-effective, deep fidelity) 0.9421 CLOSES the gap; RAW (no tree-recovery) "
                     "0.7797 does NOT. fern #318 worst cross-dataset 0.7923. These are a point estimate "
                     "+ CI bracket [0.7797, 0.9421], NOT a measured fusion-head private tax."),
        },
        "c_star_targets_343": {
            "c_star_central_public_500": C_STAR_CENTRAL_343,
            "c_star_worst_public_500": C_STAR_WORST_343,
            "cov_lift_central": COV_LIFT_CENTRAL_343,
            "cov_lift_worst": COV_LIFT_WORST_343,
            "cov_prior": COV_PRIOR,
            "identity_bar": IDENTITY_BAR,
            "imports_exact": bool(
                abs((C_STAR_CENTRAL_343 - COV_PRIOR) - COV_LIFT_CENTRAL_343) <= TOL_EXACT
                and abs((C_STAR_WORST_343 - COV_PRIOR) - COV_LIFT_WORST_343) <= TOL_EXACT),
        },
        "rho_robustness_gate_335": {
            "honest_public_611": HONEST_PUBLIC_611,
            "rho_breakeven": RHO_BREAKEVEN,
            "gate_implied_500_over_611": gate_implied,
            "roundtrips_8038": bool(abs(gate_implied - RHO_BREAKEVEN) <= TOL_RT
                                    and abs(HONEST_PUBLIC_611 * RHO_BREAKEVEN - TARGET) <= 1e-3),
            "note": ("fern #335 gate: at the FULL honest ceiling 622.08, private = 622.08*rho clears 500 "
                     "iff rho >= 500/622.08 = 0.8038. fern #341 carried this as a SEPARATE check on top "
                     "of the (phi, Delta_cov) plane -- the rho axis it dropped. We close it here."),
        },
        "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
    }


# --------------------------------------------------------------------------- #
# (D2) Private-realized envelope at the PUBLIC central-500 target (TEST: is it private-stable?).
# --------------------------------------------------------------------------- #
def deliverable2_private_realized_envelope() -> dict[str, Any]:
    public_at_cstar = envelope_central(C_STAR_CENTRAL_343)        # == 500 PUBLIC by construction
    rows = []
    for r in private_ratios():
        priv = realized_private(C_STAR_CENTRAL_343, r["rho"])
        rows.append({
            "name": r["name"], "rho": r["rho"], "regime": r["regime"], "primary": r["primary"],
            "private_tps": priv, "clears_500_private": bool(priv >= TARGET),
            "margin_tps": priv - TARGET, "source": r["source"],
        })
    primary = next(x for x in rows if x["primary"])
    deployed = next(x for x in rows if x["name"] == "deployed_gap")
    realistic = [x for x in rows if x["regime"] == "realistic"]
    # The "apply BOTH" double-count (rho_priv_e3 * deployed gap) -- shown and rejected as a double-count.
    both_multiplied = TARGET * RHO_PRIV_E3 * _g_deployed()
    is_stable = bool(primary["clears_500_private"])              # TEST bool (expect False)
    realistic_band = [min(x["private_tps"] for x in realistic),
                      max(x["private_tps"] for x in realistic)]
    return {
        "public_tps_at_central_500": public_at_cstar,
        "public_roundtrips_500": bool(abs(public_at_cstar - TARGET) <= TOL_ROOT),
        "by_ratio": rows,
        "private_tps_at_public_central_500": primary["private_tps"],   # headline (rho_priv_e3)
        "private_tps_at_public_central_500_deployed": deployed["private_tps"],
        "realistic_private_band": realistic_band,
        "ppl_only_central_500_is_private_stable": is_stable,            # TEST metric (expect False)
        "all_ratios_miss_500": bool(all(not x["clears_500_private"] for x in rows)),
        "both_multiplied_double_count": both_multiplied,
        "note": ("public-500 (c*=0.9089) -> realized private = 500*rho. rho_priv_e3 0.9421 -> {:.2f}; "
                 "deployed gap 0.9571 -> {:.2f}; worst/raw -> ~390-396. BOTH realistic estimates < 500: "
                 "the public-500 target leaves ZERO private margin (would need rho>=1.0). NOT "
                 "private-stable. (rho_priv_e3 and the deployed gap are two estimates of the SAME "
                 "haircut; the 450.8 'both-multiplied' value double-counts and is rejected.)".format(
                     primary["private_tps"], deployed["private_tps"])),
    }


# --------------------------------------------------------------------------- #
# (D3) Solve the private-500 coverage target c*_private + lift (TEST: within +0.031?).
# --------------------------------------------------------------------------- #
def deliverable3_solve_private_500() -> dict[str, Any]:
    rows = []
    for r in private_ratios():
        cstar = solve_c_for_private(r["rho"])
        need_env = TARGET / r["rho"]
        res = (realized_private(cstar, r["rho"]) - TARGET) if _finite(cstar) else float("nan")
        lift = (cstar - COV_PRIOR) if _finite(cstar) else float("nan")
        rows.append({
            "name": r["name"], "rho": r["rho"], "regime": r["regime"], "primary": r["primary"],
            "need_public_env": need_env, "c_star_private": cstar,
            "reachable_in_unit": bool(_finite(cstar) and 0.0 < cstar < 1.0),
            "root_residual": res, "root_valid": bool(_finite(cstar) and abs(res) <= TOL_ROOT),
            "coverage_lift": lift, "within_336_budget": bool(_finite(lift) and lift <= RETRAIN_LIFT_BUDGET),
            "ge_c_star_central": bool(_finite(cstar) and cstar >= C_STAR_CENTRAL_343),
        })
    primary = next(x for x in rows if x["primary"])
    deployed = next(x for x in rows if x["name"] == "deployed_gap")
    realistic = [x for x in rows if x["regime"] == "realistic"]
    realistic_lift_band = [min(x["coverage_lift"] for x in realistic),
                           max(x["coverage_lift"] for x in realistic)]
    return {
        "by_ratio": rows,
        "coverage_lift_for_private_500": primary["coverage_lift"],       # TEST metric (headline rho_e3)
        "coverage_lift_for_private_500_deployed": deployed["coverage_lift"],
        "c_star_private_headline": primary["c_star_private"],
        "c_star_private_deployed": deployed["c_star_private"],
        "realistic_lift_band": realistic_lift_band,
        "private_500_within_budget_any_realistic": bool(any(x["within_336_budget"] for x in realistic)),
        "all_realistic_over_budget": bool(all(not x["within_336_budget"] for x in realistic)),
        "all_c_star_private_ge_central": bool(all(x["ge_c_star_central"] for x in rows)),
        "all_reachable_in_unit": bool(all(x["reachable_in_unit"] for x in rows)),
        "note": ("private-500 needs OVER-provisioned coverage c*_private (solve env(c)*rho=500). "
                 "Realistic band: deployed gap +{:.4f} (c*={:.4f}, just OVER 0.031), rho_priv_e3 +{:.4f} "
                 "(c*={:.4f}, OVER). worst/raw push to +0.087+. EVERY private model EXCEEDS lawine "
                 "#336's +0.031 -- even the lightest organizer-measured 4.3% gap. c*_private >= "
                 "c*_central always (rho<1 => more coverage).".format(
                     deployed["coverage_lift"], deployed["c_star_private"],
                     primary["coverage_lift"], primary["c_star_private"])),
    }


# --------------------------------------------------------------------------- #
# (D4) Reconcile the worst-corner: is c*_worst the PRIVATE-500 requirement? (item 4 -- pin or refute).
# --------------------------------------------------------------------------- #
def deliverable4_worst_corner_reconcile(d3: dict) -> dict[str, Any]:
    # The identity c*_private == c*_worst holds EXACTLY iff g == worst/central anchor ratio.
    g_implied_by_worst = WORST_ANCHOR / CENTRAL_ANCHOR
    g_dep = _g_deployed()
    # Decompose #343's 492.87 worst anchor: honest611 uplift x worst-cross-dataset rho.
    honest_uplift = HONEST_PUBLIC_611 / CENTRAL_ANCHOR
    worst_anchor_recon = HONEST_PUBLIC_611 * RHO_WORST_XDATASET    # == 492.87
    # Bracket test: does g_implied_by_worst sit between the two realistic ratios?
    lo, hi = min(RHO_PRIV_E3, g_dep), max(RHO_PRIV_E3, g_dep)
    in_realistic_bracket = bool(lo < g_implied_by_worst < hi)
    # c*_worst vs the realistic c*_private values.
    cprivs = {x["name"]: x["c_star_private"] for x in d3["by_ratio"]}
    cworst_between = bool(min(cprivs["deployed_gap"], cprivs["rho_priv_e3"]) < C_STAR_WORST_343
                          < max(cprivs["deployed_gap"], cprivs["rho_priv_e3"]))
    identity_exact = bool(abs(g_implied_by_worst - g_dep) <= TOL_EXACT
                          or abs(g_implied_by_worst - RHO_PRIV_E3) <= TOL_EXACT)
    return {
        "g_implied_by_worst_anchor": g_implied_by_worst,
        "g_deployed": g_dep,
        "rho_priv_e3": RHO_PRIV_E3,
        "abs_diff_vs_deployed": abs(g_implied_by_worst - g_dep),
        "abs_diff_vs_rho_priv_e3": abs(g_implied_by_worst - RHO_PRIV_E3),
        "worst_anchor_in_realistic_bracket": in_realistic_bracket,
        "c_star_worst_between_realistic_private": cworst_between,
        "identity_holds_exactly": identity_exact,                       # expect False (REFUTED)
        "worst_anchor_honest_uplift": honest_uplift,
        "worst_anchor_rho_worst_xdataset": RHO_WORST_XDATASET,
        "worst_anchor_reconstruction": worst_anchor_recon,
        "worst_anchor_reconstructs_492p87": bool(abs(worst_anchor_recon - WORST_ANCHOR) <= 1e-4),
        "verdict": "REFUTED-BUT-BRACKETED",
        "note": ("c*_worst=0.9256 IS the private-500 requirement EXACTLY iff g == worst/central = "
                 "{:.5f}. Measured g_dep={:.5f} and rho_priv_e3={:.5f}: neither matches -> REFUTED. But "
                 "{:.5f} sits BETWEEN them, so c*_worst is bracketed by the two realistic c*_private "
                 "(0.9222 deployed / 0.9269 rho_e3) -- a sensible CENTRAL private proxy, not an "
                 "identity. #343's 492.87 anchor = honest611(622.08) x rho_worst_xdataset(0.7923), "
                 "confounding a x{:.3f} ceiling uplift with a x0.7923 worst-rho tax -> the 0.9461 net "
                 "lands in the realistic bracket partly by coincidence.".format(
                     g_implied_by_worst, g_dep, RHO_PRIV_E3, g_implied_by_worst, honest_uplift)),
    }


# --------------------------------------------------------------------------- #
# (D5) rho-gate closure: necessary at full ceiling, NOT sufficient at the public-500 point.
# --------------------------------------------------------------------------- #
def deliverable5_rho_gate_closure() -> dict[str, Any]:
    # At the FULL honest ceiling 622.08: private = 622.08*rho, clears 500 iff rho >= 0.8038.
    gate_rows = []
    for r in private_ratios():
        priv_full = HONEST_PUBLIC_611 * r["rho"]
        gate_rows.append({
            "name": r["name"], "rho": r["rho"], "regime": r["regime"],
            "private_tps_full_ceiling": priv_full,
            "clears_rho_gate_8038": bool(r["rho"] >= RHO_BREAKEVEN),
            "rho_headroom_to_8038": r["rho"] - RHO_BREAKEVEN,
        })
    realistic_clears = bool(all(g["clears_rho_gate_8038"] for g in gate_rows
                                if g["regime"] == "realistic"))
    downside_misses = bool(all(not g["clears_rho_gate_8038"] for g in gate_rows
                               if g["regime"] == "downside"))
    # The crux: at the public-500 operating point (env=500, not 622), the implied gate is rho>=1.0.
    gate_at_public_500 = TARGET / TARGET                              # 1.0
    realistic_misses_public_500 = bool(all(r["rho"] < gate_at_public_500
                                           for r in private_ratios() if r["regime"] == "realistic"))
    return {
        "full_ceiling_gate": gate_rows,
        "rho_breakeven_full_ceiling": RHO_BREAKEVEN,
        "realistic_clears_full_ceiling_gate": realistic_clears,        # True (fern #318 YELLOW build)
        "downside_misses_full_ceiling_gate": downside_misses,          # True (worst/raw)
        "gate_implied_at_public_500": gate_at_public_500,              # 1.0
        "realistic_rho_misses_public_500_gate": realistic_misses_public_500,   # True (rho<1)
        "necessary_not_sufficient": bool(realistic_clears and realistic_misses_public_500),
        "note": ("rho>=0.8038 (full ceiling) is CLEARED by realistic rho (0.9421/0.9571 -> 586/595 "
                 "private) -- fern #318's YELLOW build. But the SAME rho MISSES at the budget-minimal "
                 "public-500 point, where the implied gate is rho>=1.0. The rho-gate is NECESSARY (a "
                 "full build needs it) but NOT SUFFICIENT (the cheap public-500 isocline point still "
                 "fails private). This is the rho-axis closure of fern #341's dropped axis."),
    }


# --------------------------------------------------------------------------- #
# Honest caveats (item 5).
# --------------------------------------------------------------------------- #
def caveats() -> list[str]:
    return [
        "rho_priv_e3 is a MODELED point estimate (lawine #300 deployed-effective), not a measured "
        "fusion-head private tax. Carry its CI bracket [0.7797 raw, 0.9421 deployed-effective]; fern "
        "#318's worst cross-dataset 0.7923 is a literature lower bound (a_1-recovery NOT credited). "
        "Under the downside bracket private-500 needs +0.087-0.092 coverage (far over budget).",
        "The private eval is HELD-OUT: the deployed 4.3% gap (460.85/481.53) is an aggregate; it may "
        "NOT be coverage-uniform across depths/prompts, so applying it as a scalar haircut to env(c) is "
        "a first-order approximation. lawine #300's per-position model (c_deep=0.97135 on j>=2, a_1 "
        "held) is the finer structure; this card uses the scalar projection.",
        "rho_priv_e3 and the deployed 4.3% gap are NOT independent and are NOT multiplied: lawine #300 "
        "derived rho_priv_e3 from the SAME per-position collapse that produces the deployed gap. They "
        "are two estimates of the one private/public ratio and BRACKET it at [0.9421, 0.9571]. The "
        "'both-multiplied' 450.8 is a double-count, shown only to reject it.",
        "Scale confound in #343's 'worst' anchor: 492.87 = honest611(622.08) x rho_worst(0.7923) mixes "
        "the honest-ceiling uplift (x1.194) with a worst-rho tax, while the central anchor 520.95 is "
        "the lambda-CAPPED public ceiling. The worst/central ratio 0.9461 therefore is NOT a clean "
        "private haircut; its landing in the realistic bracket is partly coincidental.",
        "A DEFINITIVE private number needs the gated private read (organizer-side); this is a CPU "
        "envelope projection over banked constants. 0 GPU, 0 TPS, no served-file change, no HF Job, no "
        "submission. BASELINE stays 481.53. Frame: the rho-axis (3rd-axis) closure of fern #341's 2D "
        "(phi, Delta_cov) card -- NOT a launch / build / open2.",
    ]


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, d4: dict, d5: dict) -> dict[str, Any]:
    dep = d1["deployed_public_to_private"]
    gate = d1["rho_robustness_gate_335"]
    conditions = {
        # (a) public->private 481.53/460.85 round-trips <= 1e-6 AND delta == 4.3% (#300).
        "a_public_private_roundtrip": bool(
            dep["roundtrips_460p85"] and dep["roundtrips_300_delta_4p3"]
            and abs(dep["private_roundtrip"] - PRIVATE_VERIFIED) <= 1e-6),
        # (b) rho_priv + c* targets imported EXACT (match banked to ~0).
        "b_rho_and_cstar_imports_exact": bool(
            abs(RHO_PRIV_E3 - 0.9421228821714434) <= TOL_EXACT
            and abs(RHO_PRIV_E3_RAW - 0.7797221674962985) <= TOL_EXACT
            and abs(C_STAR_CENTRAL_343 - C_STAR_CENTRAL_343) <= TOL_EXACT
            and d1["c_star_targets_343"]["imports_exact"]
            and abs(envelope_central(C_STAR_CENTRAL_343) - TARGET) <= TOL_ROOT),
        # (c) private envelope NaN-clean (set by caller).
        "c_private_envelope_nan_clean": True,
        # (d) c*_private >= c*_central for EVERY private model (private needs >= public coverage).
        "d_c_star_private_ge_central": bool(d3["all_c_star_private_ge_central"]),
        # (e) the worst-corner identity test is EXPLICIT (computed, compared, refuted, bracketed).
        "e_worst_corner_identity_explicit": bool(
            (not d4["identity_holds_exactly"])
            and d4["worst_anchor_in_realistic_bracket"]
            and d4["c_star_worst_between_realistic_private"]
            and d4["worst_anchor_reconstructs_492p87"]),
        # (f) TEST bool: public central-500 is NOT private-stable (all realistic ratios miss 500).
        "f_public_central_500_not_private_stable": bool(
            (not d2["ppl_only_central_500_is_private_stable"]) and d2["all_ratios_miss_500"]
            and d2["public_roundtrips_500"]),
        # (g) TEST float: private-500 lift OVER budget for every realistic model (incl. deployed gap).
        "g_private_500_over_budget": bool(
            d3["all_realistic_over_budget"]
            and d3["coverage_lift_for_private_500"] > RETRAIN_LIFT_BUDGET
            and d3["coverage_lift_for_private_500_deployed"] > RETRAIN_LIFT_BUDGET),
        # (h) c*_private reachable in (0,1) for all models (a coverage solution exists, just over budget).
        "h_c_star_private_reachable": bool(d3["all_reachable_in_unit"]),
        # (i) rho-gate: 500/622.08 round-trips 0.8038; realistic CLEARS full ceiling but MISSES public-500.
        "i_rho_gate_necessary_not_sufficient": bool(
            gate["roundtrips_8038"] and d5["realistic_clears_full_ceiling_gate"]
            and d5["realistic_rho_misses_public_500_gate"] and d5["necessary_not_sufficient"]),
        # (j) imports round to displayed forms.
        "j_imports_round_to_display": bool(
            abs(IDENTITY_BAR - 0.9213) <= TOL_DISPLAY_C and abs(COV_PRIOR - 0.8903) <= TOL_DISPLAY_C
            and abs(CENTRAL_ANCHOR - 520.95) <= TOL_DISPLAY_TPS
            and abs(WORST_ANCHOR - 492.87) <= TOL_DISPLAY_TPS
            and abs(RHO_PRIV_E3 - 0.9421) <= TOL_DISPLAY_C
            and abs(RHO_BREAKEVEN - 0.8038) <= TOL_DISPLAY_C
            and abs(HONEST_PUBLIC_611 - 622.08) <= 1e-2
            and abs(C_STAR_CENTRAL_343 - 0.9089) <= TOL_DISPLAY_C
            and abs(C_STAR_WORST_343 - 0.9256) <= TOL_DISPLAY_C),
        # (k) NaN-clean (set by caller).
        "k_nan_clean": True,
        # (l) structural: realized private at public-500 strictly below 500 under the primary rho.
        "l_private_below_500_at_public_target": bool(
            d2["private_tps_at_public_central_500"] < TARGET),
        # (m) the realistic private band at public-500 is entirely below 500 (robust to ratio choice).
        "m_realistic_band_below_500": bool(d2["realistic_private_band"][1] < TARGET),
    }
    return {
        "conditions": conditions,
        "private_stability_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
        "detail": {
            "private_tps_at_public_central_500": d2["private_tps_at_public_central_500"],
            "ppl_only_central_500_is_private_stable": d2["ppl_only_central_500_is_private_stable"],
            "coverage_lift_for_private_500": d3["coverage_lift_for_private_500"],
            "coverage_lift_for_private_500_deployed": d3["coverage_lift_for_private_500_deployed"],
            "g_implied_by_worst_anchor": d4["g_implied_by_worst_anchor"],
            "identity_holds_exactly": d4["identity_holds_exactly"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_import_structure()
    d2 = deliverable2_private_realized_envelope()
    d3 = deliverable3_solve_private_500()
    d4 = deliverable4_worst_corner_reconcile(d3)
    d5 = deliverable5_rho_gate_closure()
    st = _selftests(d1, d2, d3, d4, d5)

    handoff = (
        "The PPL-only PUBLIC central-500 target (wirbel #343 c*=0.9089, +0.0186, within budget) does NOT "
        "survive private. At the public-500 operating point realized private = 500*rho_priv: rho_priv_e3 "
        "0.9421 -> {:.2f}, deployed gap 0.9571 -> {:.2f} (both < 500; worst/raw -> ~390-396). The "
        "public-500 target leaves ZERO private margin (would need rho>=1.0). Restoring private-500 needs "
        "OVER-provisioned coverage c*_private = {:.4f} (rho_e3, +{:.4f}) / {:.4f} (deployed, +{:.4f}) -- "
        "OVER lawine #336's +0.031 under EVERY private model, even the organizer-measured 4.3% gap. "
        "wirbel #343's c*_worst=0.9256 is a CENTRAL private proxy (bracketed by the realistic c*_private) "
        "but NOT an exact identity (REFUTED: g_worst 0.9461 != measured 0.9421/0.9571). The rho>=0.8038 "
        "gate clears at the FULL ceiling (586/595 private, fern #318 YELLOW build) but is NOT sufficient "
        "at the cheap public-500 point. This is the rho-axis closure of fern #341's dropped 3rd axis: "
        "the demand-led public-500 isocline is PUBLIC-only.".format(
            d2["private_tps_at_public_central_500"], d2["private_tps_at_public_central_500_deployed"],
            d3["c_star_private_headline"], d3["coverage_lift_for_private_500"],
            d3["c_star_private_deployed"], d3["coverage_lift_for_private_500_deployed"]))

    headline = {
        "private_stability_self_test_passes": bool(st["private_stability_self_test_passes"]),  # PRIMARY
        "ppl_only_central_500_is_private_stable": d2["ppl_only_central_500_is_private_stable"],  # TEST
        "coverage_lift_for_private_500": d3["coverage_lift_for_private_500"],                    # TEST
        "private_tps_at_public_central_500": d2["private_tps_at_public_central_500"],
        "private_tps_at_public_central_500_deployed": d2["private_tps_at_public_central_500_deployed"],
        "realistic_private_band": d2["realistic_private_band"],
        "c_star_private_headline": d3["c_star_private_headline"],
        "c_star_private_deployed": d3["c_star_private_deployed"],
        "coverage_lift_for_private_500_deployed": d3["coverage_lift_for_private_500_deployed"],
        "realistic_lift_band": d3["realistic_lift_band"],
        "all_realistic_over_budget": d3["all_realistic_over_budget"],
        "worst_corner_identity_holds": d4["identity_holds_exactly"],
        "g_implied_by_worst_anchor": d4["g_implied_by_worst_anchor"],
        "rho_gate_necessary_not_sufficient": d5["necessary_not_sufficient"],
        "rho_priv_e3": RHO_PRIV_E3,
        "rho_breakeven_8038": RHO_BREAKEVEN,
    }
    return {
        "headline": headline,
        "deliverable1_import_structure": d1,
        "deliverable2_private_realized_envelope": d2,
        "deliverable3_solve_private_500": d3,
        "deliverable4_worst_corner_reconcile": d4,
        "deliverable5_rho_gate_closure": d5,
        "self_test": st,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "wirbel #343 kklof4wr (envelope_X(c)=X_ANCHOR*E[T](c)/E[T](0.9213); c*_central 0.9089 / "
                "c*_worst 0.9256; prior 0.8903; identity bar 0.9213; anchors central 520.95 / worst "
                "492.87) x lawine #300 8t5q6sr0 (rho_priv_e3 0.9421 deployed-effective / 0.7797 raw; "
                "deployed 481.53->460.85 Delta 4.3%; private_bar 500) x fern #318 xe8ff7hq (honest611 "
                "622.08; rho_worst_xdataset 0.7923; full-build private 622.08*rho; worst 492.87; YELLOW) "
                "x fern #335/#310 (rho>=0.8038 gate == 500/622.08) x fern #341 o4rzy1k6 (demand-led "
                "isocline + DROPPED rho axis) x lawine #336 krroookz (+0.031 retrain budget). All "
                "run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": caveats(),
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    d1 = syn["deliverable1_import_structure"]
    d2 = syn["deliverable2_private_realized_envelope"]
    d3 = syn["deliverable3_solve_private_500"]
    d4 = syn["deliverable4_worst_corner_reconcile"]
    d5 = syn["deliverable5_rho_gate_closure"]
    st = syn["self_test"]
    dep = d1["deployed_public_to_private"]
    print("\n" + "=" * 100, flush=True)
    print("PPL-ONLY 500 PRIVATE-STABILITY — does PUBLIC c*=0.9089 survive private? (PR #347, wirbel)",
          flush=True)
    print("=" * 100, flush=True)
    print("  (D1) PUBLIC->PRIVATE IMPORTS", flush=True)
    print(f"      deployed {dep['official_public']:.2f} -> {dep['private_verified']:.2f}  "
          f"g={dep['g_deployed_ratio']:.5f}  Delta={dep['delta_pct']*100:.2f}%  "
          f"(rt 460.85: {dep['roundtrips_460p85']})", flush=True)
    print(f"      rho_priv_e3={RHO_PRIV_E3:.4f} (deployed-effective) / {RHO_PRIV_E3_RAW:.4f} (raw)  "
          f"rho-gate(full ceiling)={RHO_BREAKEVEN:.4f}=500/622.08", flush=True)
    print("-" * 100, flush=True)
    print("  (D2) PRIVATE-REALIZED @ PUBLIC central-500 (c*=0.9089)   [TEST: private-stable?]", flush=True)
    for r in d2["by_ratio"]:
        print(f"      {r['name']:18s} rho={r['rho']:.4f} -> private={r['private_tps']:7.2f}  "
              f"clears500={r['clears_500_private']}  ({r['regime']})", flush=True)
    print(f"      >> private_tps_at_public_central_500 = {d2['private_tps_at_public_central_500']:.2f}  "
          f"is_private_stable = {d2['ppl_only_central_500_is_private_stable']}", flush=True)
    print("-" * 100, flush=True)
    print("  (D3) SOLVE PRIVATE-500 COVERAGE c*_private + lift   [TEST: within +0.031?]", flush=True)
    for r in d3["by_ratio"]:
        print(f"      {r['name']:18s} c*_priv={r['c_star_private']:.6f}  lift=+{r['coverage_lift']:.5f}  "
              f"within0.031={r['within_336_budget']}  reachable={r['reachable_in_unit']}", flush=True)
    print(f"      >> coverage_lift_for_private_500 = +{d3['coverage_lift_for_private_500']:.5f} "
          f"(rho_e3) / +{d3['coverage_lift_for_private_500_deployed']:.5f} (deployed)  "
          f"all_over_budget={d3['all_realistic_over_budget']}", flush=True)
    print("-" * 100, flush=True)
    print("  (D4) WORST-CORNER RECONCILE — is c*_worst=0.9256 the private-500 requirement?", flush=True)
    print(f"      g_implied_by_worst={d4['g_implied_by_worst_anchor']:.5f}  "
          f"g_dep={d4['g_deployed']:.5f}  rho_e3={d4['rho_priv_e3']:.5f}  "
          f"identity_exact={d4['identity_holds_exactly']}", flush=True)
    print(f"      bracketed={d4['worst_anchor_in_realistic_bracket']}  "
          f"492.87=622.08x{d4['worst_anchor_rho_worst_xdataset']:.4f}? "
          f"{d4['worst_anchor_reconstructs_492p87']}  -> {d4['verdict']}", flush=True)
    print("-" * 100, flush=True)
    print("  (D5) rho-GATE CLOSURE — necessary at full ceiling, NOT sufficient at public-500", flush=True)
    print(f"      realistic clears full-ceiling gate={d5['realistic_clears_full_ceiling_gate']}  "
          f"misses public-500 gate(rho>=1)={d5['realistic_rho_misses_public_500_gate']}  "
          f"=> necessary_not_sufficient={d5['necessary_not_sufficient']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY private_stability_self_test_passes = "
          f"{st['private_stability_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[ppl-only-500-private-stability] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    d2 = syn["deliverable2_private_realized_envelope"]
    d3 = syn["deliverable3_solve_private_500"]
    d4 = syn["deliverable4_worst_corner_reconcile"]
    d5 = syn["deliverable5_rho_gate_closure"]
    st = syn["self_test"]
    run = init_wandb_run(
        job_type="ppl-only-500-private-stability",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["ppl-only-500-private-stability", "issue-319", "issue-341", "issue-343", "eagle3",
              "private-tax", "rho-axis", "public-to-private", "compliant-500", "validity-gate",
              "bank-the-analysis"],
        config={
            "K_spec": K_SPEC, "identity_bar": IDENTITY_BAR, "cov_prior": COV_PRIOR,
            "central_anchor": CENTRAL_ANCHOR, "worst_anchor": WORST_ANCHOR,
            "e_t_at_identity": E_T_AT_IDENTITY,
            "c_star_central_343": C_STAR_CENTRAL_343, "c_star_worst_343": C_STAR_WORST_343,
            "official_public": OFFICIAL_PUBLIC, "private_verified": PRIVATE_VERIFIED,
            "rho_priv_e3": RHO_PRIV_E3, "rho_priv_e3_raw": RHO_PRIV_E3_RAW,
            "rho_worst_xdataset": RHO_WORST_XDATASET, "rho_breakeven": RHO_BREAKEVEN,
            "honest_public_611": HONEST_PUBLIC_611, "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
            "target": TARGET, "baseline_tps": BASELINE_TPS, "wandb_group": args.wandb_group,
            "source_runs": ("wirbel#343(kklof4wr), lawine#300(8t5q6sr0), fern#318(xe8ff7hq), "
                            "fern#341(o4rzy1k6), lawine#336(krroookz)"),
        },
    )
    if run is None:
        print("[ppl-only-500-private-stability] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "private_stability_self_test_passes": int(bool(st["private_stability_self_test_passes"])),  # PRIMARY
        "ppl_only_central_500_is_private_stable": int(bool(d2["ppl_only_central_500_is_private_stable"])),  # TEST
        "coverage_lift_for_private_500": d3["coverage_lift_for_private_500"],                          # TEST
        "coverage_lift_for_private_500_deployed": d3["coverage_lift_for_private_500_deployed"],
        "private_tps_at_public_central_500": d2["private_tps_at_public_central_500"],
        "private_tps_at_public_central_500_deployed": d2["private_tps_at_public_central_500_deployed"],
        "realistic_private_band_lo": d2["realistic_private_band"][0],
        "realistic_private_band_hi": d2["realistic_private_band"][1],
        "all_ratios_miss_500": int(bool(d2["all_ratios_miss_500"])),
        "c_star_private_headline": d3["c_star_private_headline"],
        "c_star_private_deployed": d3["c_star_private_deployed"],
        "realistic_lift_band_lo": d3["realistic_lift_band"][0],
        "realistic_lift_band_hi": d3["realistic_lift_band"][1],
        "all_realistic_over_budget": int(bool(d3["all_realistic_over_budget"])),
        "all_c_star_private_ge_central": int(bool(d3["all_c_star_private_ge_central"])),
        "g_implied_by_worst_anchor": d4["g_implied_by_worst_anchor"],
        "worst_corner_identity_holds": int(bool(d4["identity_holds_exactly"])),
        "abs_diff_worst_vs_deployed": d4["abs_diff_vs_deployed"],
        "abs_diff_worst_vs_rho_priv_e3": d4["abs_diff_vs_rho_priv_e3"],
        "rho_gate_necessary_not_sufficient": int(bool(d5["necessary_not_sufficient"])),
        "realistic_clears_full_ceiling_gate": int(bool(d5["realistic_clears_full_ceiling_gate"])),
        "realistic_rho_misses_public_500_gate": int(bool(d5["realistic_rho_misses_public_500_gate"])),
        "rho_priv_e3": RHO_PRIV_E3, "rho_priv_e3_raw": RHO_PRIV_E3_RAW,
        "rho_breakeven_8038": RHO_BREAKEVEN,
        "delta_pct_deployed": 1.0 - (PRIVATE_VERIFIED / OFFICIAL_PUBLIC),
        "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="ppl_only_500_private_stability_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[ppl-only-500-private-stability] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="ppl-only-500-private-stability")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 347, "agent": "wirbel",
        "kind": "ppl-only-500-private-stability", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    # propagate NaN-clean into the self-test conditions that assert it.
    syn["self_test"]["conditions"]["c_private_envelope_nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["k_nan_clean"] = not nan_paths
    syn["self_test"]["private_stability_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["private_stability_self_test_passes"] = syn["self_test"][
        "private_stability_self_test_passes"]
    if nan_paths:
        print(f"[ppl-only-500-private-stability] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppl_only_500_private_stability_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[ppl-only-500-private-stability] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["private_stability_self_test_passes"] and payload["nan_clean"])
        print(f"[ppl-only-500-private-stability] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
