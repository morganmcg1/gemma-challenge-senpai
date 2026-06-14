#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Ceiling-gap topology headroom (PR #244, wirbel) — CPU-only analytic bank-the-analysis.

THE QUESTION (the open edge of my #235 two-ceiling reconcile; the human's Issue #211)
------------------------------------------------------------------------------------
My #235 (MERGED) established that the OPERATIVE compliant λ=1 ceiling for the lane-a custom
batch-invariant int4 verify kernel is the int4-spec 520.95 (E[T]=5.0662), NOT the optimistic
reach-DP 536.66 (E[T]=5.21888) — the determinism-only kernel inherits the shipped topology +
coverage and does NOT deliver the rank-1-coverage 0.7304 over-count baked into the reach-DP edge.
stark #226's worst-case private build bar is 528.48. So at the operative ceiling the
compliant-PRIVATE-500 lane is INFEASIBLE (528.48 > 520.95). #235 flagged the reach-DP upper edge
as "a land #71 TOPOLOGY question, not a kernel question". This leg ANSWERS that question.

CRUX — is there ANY reachable verify-tree TOPOLOGY change (branch width beyond the deployed
max-branch-3, depth, or rank-coverage ρ) that lifts the OPERATIVE int4-spec ceiling 520.95 ABOVE
the 528.48 private bar, given the MEASURED declining rank-coverage ladder (ρ₂=0.4165 ≫ ρ₃=0.2655 >
ρ₄=0.1908, my #79/#83) — or is the compliant-private lane TOPOLOGY-DEAD, reopenable only by a
COVERAGE recovery (the λ→1 / batch-invariance lever, NOT topology)?

THE MECHANISM (all imported / banked; NOTHING re-derived)
---------------------------------------------------------
The deployed verify tree IS my #83 ρ-optimal max-branch-3, depth-9, M=32 array. #83 already ran
the FULL Sequoia/DP topology optimisation under the measured ladder and banked, for every reachable
(width, depth): the topology E[T] `F_tree` AND the drafter-aware TPS `gain`. I import that banked
sweep verbatim (`research/spec_cost_model/rho_optimal_topology_results.json`, run 6tghbnjn) and
reproduce its cost model:

    gemm_ratio(M)        = GEMM68(M)/GEMM68(8)                          (#68 verify-GEMM roofline)
    cost_mult(M, depth)  = gemm_ratio(M) + g_d·(depth − 7)/7           g_d=0.168 (#69/#77 drafter)
    gain(F, M, depth)    = (F / F_linear8) / cost_mult(M, depth) − 1   F_linear8=3.84445 (#76 anchor)

The drafter runs `depth` SEQUENTIAL weight-re-reading passes, so a deeper spine raises F_tree
(raw E[T]) but pays g_d·(depth−7)/7 more decode-step cost. Under this model the deployed depth-9 is
the TPS-MAXIMAL tree; deeper trees have HIGHER E[T] but LOWER TPS (the "depth-18 +32% artifact"
#83 already identified and regularised). Width-4 adds EXACTLY ZERO E[T] over width-3 and the DP
never places a rank-5 child even when allowed (`uses_rank5plus=False`).

BRIDGE TO THE #235 CEILING SCALE. The operative ceiling 520.95 and the bar 528.48 live on the #235
launch composition  μ = K_cal·(E[T]/step)·τ. I map any candidate topology onto that scale by the
ratio of #83 drafter-aware gains (calibration-independent; the realized-coverage over-count is
uniform across shapes so it cancels in the ratio), anchored so the deployed tree maps to 520.95:

    tps_operative(cand) = INT4_SPEC_CEILING · (1 + gain_83[cand]) / (1 + gain_83[deployed])

THE DELIVERABLE
---------------
(1) delta_ET_needed: E_T_needed = E_T(520.95)·(528.48/520.95); the E[T]/TPS lift to clear the bar.
(2) sweep reachable topology levers (width, depth, ρ); topology_lift_max vs delta_ET_needed;
    headline `lane_reopenable` (bool).
(3) verdict table: topology candidate × (realized E[T], operative-ceiling TPS, clears-528.48?,
    coverage cost) — incl. the m_only-cost ARTIFACT rows that FALSELY clear if drafter cost ignored.
(4) PRIMARY self-test (5 conditions a–e) → `ceiling_gap_topology_headroom_self_test_passes`
    (PRIMARY) and `topology_lift_needed` (TEST, E[T] units).
(5) one-sentence hand-off (fern's card + land #71 build packet + #211 ROI).

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. Imports my #235 ceilings/bar
(reconcile), #83 6tghbnjn topology sweep + cost model, #79 measured ρ ladder, stark #226 private
bar, kanna #217 composition. Re-derives nothing. NOT a launch. NOT open2.

PRIMARY metric  ceiling_gap_topology_headroom_self_test_passes
TEST    metric  topology_lift_needed  (E[T] the topology must add to clear 528.48; 0 reachable)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Import the #199/#235 composition machinery (K_cal, step, et_via_reachdp) for the
# ceiling provenance round-trip, and load the #83 banked topology sweep.
# --------------------------------------------------------------------------- #
def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("compliant_spec_et", "research/validity/compliant_spec_et/compliant_spec_et.py")
K_CAL = C.K_CAL                     # 125.26795005202914 (#148/#169, kanna #217)
STEP = C.STEP                       # 1.2182 (#168 deployed M=32 step, normalized)
TAU_CENTRAL = 1.0                   # served-fraction central corner (#181)

RHO83_PATH = REPO_ROOT / "research/spec_cost_model/rho_optimal_topology_results.json"

# --------------------------------------------------------------------------- #
# Banked constants from my #235 two-ceiling reconcile (provenance: run ids in
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, not re-derived.
# --------------------------------------------------------------------------- #
REACH_DP_CENTRAL = 536.6590426143789   # #199 wdyqnx3g  reach-DP ceiling (full-coverage, OPTIMISTIC)
REACH_DP_ET = 5.21887717841078         # #199 wdyqnx3g  reach-DP E[T] (rank-1 0.7304 ladder)
INT4_SPEC_CEILING = 520.9527323111674  # #204/#220 pqjnybbf  int4-spec λ=1 ceiling (OPERATIVE, #235)
INT4_SPEC_ET = 5.0661371742562835      # #220 pqjnybbf  int4_anchor_et1 (= ceiling·step/K_cal, τ=1)
RANK1_COVERAGE_TOP1 = 0.7304444056147708   # #199/#227  q_compliant depth-1 top-1 match

# stark #226 (tzcc5xuq) — the private build bar the operative ceiling is tested against.
PRIVATE_BAR_WORSTCASE = 528.4835555959945   # private_bar_worstcase (NLS worst realizable blend)

TARGET = 500.0                  # official PUBLIC clear bar
BASELINE_TPS = 481.53           # PR #52 official (this leg adds 0 TPS)

# realized-coverage over-count fraction (#235): the shipped stack realizes this fraction of the
# optimistic full rank-1 coverage E[T]. Uniform across tree shapes (it is a per-step argmax
# property, not a topology property) — the defensible uniform discount used in the bridge.
REALIZED_FRAC = INT4_SPEC_ET / REACH_DP_ET   # 0.970735

TOL_ROUNDTRIP = 1e-6            # composition inversion / banked round-trip
TOL_PROV = 1e-9                # reach-DP reproduction of the banked ceiling E[T]
TOL_GAIN = 5e-5               # #83 cost-model reproduction of banked drafter-aware gains
GAP_BAR_LITERAL = 7.53        # PR arithmetic: 528.48 − 520.95 (the bar-gap, NOT the 15.71 over-count)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def official_tps(et: float, tau: float = TAU_CENTRAL, step: float = STEP) -> float:
    return K_CAL * (et / step) * tau


def et_from_tps(tps: float, tau: float = TAU_CENTRAL, step: float = STEP) -> float:
    """Invert the launch composition: E[T](1) = μ·step / (K_cal·τ)."""
    return tps * step / (K_CAL * tau)


# --------------------------------------------------------------------------- #
# Load the #83 (run 6tghbnjn) banked topology sweep + cost-model inputs.
# --------------------------------------------------------------------------- #
def load_rho83() -> dict[str, Any]:
    d = json.loads(RHO83_PATH.read_text(encoding="utf-8"))
    m32 = d["per_budget"]["32"]
    return {
        "F_linear8": d["anchor_F_linear8"],                         # 3.84445 (#76 anchor)
        "gemm_cost_mult": d["inputs"]["gemm_cost_mult"],            # GEMM68(M)/GEMM68(8) table
        "g_drafter": d["config"]["g_drafter"],                     # 0.168 (#69/#77)
        "base_drafter_depth": d["config"]["base_drafter_depth"],   # 7 (deployed K=7 chain)
        "rho_cond_measured": d["inputs"]["rho_cond_measured"],     # [0.4165, 0.2655, 0.1908]
        "deployed_F_tree": m32["optimal"]["F_tree"],               # 5.206954 (depth-9 mb-3)
        "deployed_depth": m32["optimal"]["depth"],                 # 9
        "deployed_max_branch": m32["optimal"]["max_branch"],       # 3
        "deployed_gain_drafter_aware": d["verdict"]["measured_rho_optimal_M32_gain_pct"],  # 0.18170
        "deployed_gain_m_only": d["g_drafter_sensitivity"]["gd0.0"]["M32_opt_gain"],       # 0.23336
        "width_branch_factor": d["width_branch_factor"],           # M32_branch2/3/4 F values
        "beyond_width4_M32": d["beyond_width4"]["M32"],            # uses_rank5plus etc.
        "depth_sweep": m32["depth_sweep"],                        # [{depth, F_tree, gain_*}]
    }


def gemm_ratio(R83: dict[str, Any], M: int) -> float:
    return float(R83["gemm_cost_mult"][str(M)])


def cost_mult(R83: dict[str, Any], M: int, depth: int) -> float:
    """#83 drafter-aware decode-step multiplier (reproduces the banked cost_mult exactly)."""
    g_d = R83["g_drafter"]
    base = R83["base_drafter_depth"]
    return gemm_ratio(R83, M) + g_d * (depth - base) / base


def gain_drafter_aware(R83: dict[str, Any], F_tree: float, M: int, depth: int) -> float:
    return (F_tree / R83["F_linear8"]) / cost_mult(R83, M, depth) - 1.0


def gain_m_only(R83: dict[str, Any], F_tree: float, M: int) -> float:
    """PR-literal #68 M-only cost (g_d=0): the cost model that PRODUCES the depth artifact."""
    return (F_tree / R83["F_linear8"]) / gemm_ratio(R83, M) - 1.0


# --------------------------------------------------------------------------- #
# Bridge: map a candidate topology's #83 gain onto the #235 operative-ceiling TPS scale.
# Anchored so the deployed tree maps to exactly INT4_SPEC_CEILING (520.95).
# --------------------------------------------------------------------------- #
def tps_operative(R83: dict[str, Any], gain_cand: float, *, m_only: bool = False) -> float:
    g_dep = R83["deployed_gain_m_only"] if m_only else R83["deployed_gain_drafter_aware"]
    return INT4_SPEC_CEILING * (1.0 + gain_cand) / (1.0 + g_dep)


def realized_et(R83: dict[str, Any], F_tree: float) -> float:
    """Realized (operative-coverage) E[T] for a candidate tree: scale the deployed reach-DP E[T]
    by the #83 F_tree ratio, then apply the uniform realized-coverage over-count discount."""
    full_cov_et = REACH_DP_ET * (F_tree / R83["deployed_F_tree"])
    return REALIZED_FRAC * full_cov_et


# --------------------------------------------------------------------------- #
# (1) delta_ET_needed — the E[T]/TPS lift required to clear the private bar.
# --------------------------------------------------------------------------- #
def needed_lift() -> dict[str, Any]:
    et_operative = INT4_SPEC_ET
    # PR formula: E_T_needed = E_T(520.95) · (528.48/520.95). Algebraically == et_from_tps(528.48).
    et_needed = et_operative * (PRIVATE_BAR_WORSTCASE / INT4_SPEC_CEILING)
    et_needed_alt = et_from_tps(PRIVATE_BAR_WORSTCASE)         # cross-check (identical)
    delta_et = et_needed - et_operative
    delta_tps = PRIVATE_BAR_WORSTCASE - INT4_SPEC_CEILING      # 7.53 (the bar-gap)
    # coverage over-count headroom (operative→reach-DP) for transparent contrast with the PR's
    # mislabelled "15.71 TPS gap" (which is the operative→reach-DP coverage over-count, NOT the bar).
    coverage_overcount_tps = REACH_DP_CENTRAL - INT4_SPEC_CEILING   # 15.71
    return {
        "et_operative": et_operative,
        "et_needed": et_needed,
        "et_needed_via_et_from_tps": et_needed_alt,
        "et_needed_resid": abs(et_needed - et_needed_alt),
        "delta_ET_needed": delta_et,                           # TEST basis (E[T] units)
        "delta_ET_needed_tps": delta_tps,                      # 7.53
        "delta_ET_needed_pct": 100.0 * delta_et / et_operative,
        "coverage_overcount_tps_operative_to_reachdp": coverage_overcount_tps,   # 15.71
        "note_pr_label": (
            "PR prose labels the 520.95→528.48 step a '15.71 TPS gap'; the arithmetic gives "
            "528.48 − 520.95 = 7.53. The 15.71 is the operative→reach-DP COVERAGE over-count "
            "(536.66 − 520.95), of which only 7.53 is needed to reach the bar. delta_ET_needed "
            "corresponds to the 7.53 bar-gap; both are reported."),
    }


# --------------------------------------------------------------------------- #
# (2)+(3) Topology sweep + verdict table.
# --------------------------------------------------------------------------- #
def _candidate(R83, label, F_tree, depth, M, lever, coverage_cost) -> dict[str, Any]:
    gda = gain_drafter_aware(R83, F_tree, M, depth)
    gmo = gain_m_only(R83, F_tree, M)
    tps = tps_operative(R83, gda)
    tps_mo = tps_operative(R83, gmo, m_only=True)
    ret = realized_et(R83, F_tree)
    return {
        "label": label,
        "lever": lever,
        "F_tree": F_tree,
        "depth": depth,
        "M": M,
        "max_branch": R83["deployed_max_branch"],
        "realized_et": ret,
        "gain_drafter_aware": gda,
        "gain_m_only": gmo,
        "tps_operative": tps,                                  # honest (drafter-aware) TPS
        "tps_operative_m_only_artifact": tps_mo,              # if drafter cost IGNORED
        "tps_lift_vs_operative": tps - INT4_SPEC_CEILING,
        "clears_private_bar": bool(tps >= PRIVATE_BAR_WORSTCASE),
        "clears_private_bar_m_only_artifact": bool(tps_mo >= PRIVATE_BAR_WORSTCASE),
        "coverage_cost": coverage_cost,
    }


def topology_sweep(R83: dict[str, Any]) -> dict[str, Any]:
    wbf = R83["width_branch_factor"]
    F_mb2 = wbf["M32_branch2"]["F"]
    F_mb3 = wbf["M32_branch3"]["F"]          # == deployed
    F_mb4 = wbf["M32_branch4"]["F"]
    dep = {d["depth"]: d for d in R83["depth_sweep"]}        # depth -> sweep row

    cands: list[dict[str, Any]] = []
    # WIDTH lever (all depth-9, the deployed/TPS-optimal depth):
    cands.append(_candidate(R83, "width2_mb2_d9", F_mb2, 9, 32, "width",
                            "drops a rank-2 branch; loses +0.0057 E[T] vs deployed"))
    cands.append(_candidate(R83, "width3_mb3_d9_DEPLOYED", F_mb3, 9, 32, "width",
                            "deployed ρ-optimal; ρ₂=0.4165 rank-2 + spine"))
    cands.append(_candidate(R83, "width4_mb4_d9", F_mb4, 9, 32, "width",
                            "rank-4 child marginal ~0.022 loses node budget; +0.0 E[T]"))
    cands.append(_candidate(R83, "width6_rank5plus_d9", F_mb3, 9, 32, "width",
                            "DP never places rank-5/6 (uses_rank5plus=False); +0.0 E[T]"))
    # DEPTH lever (DP-optimal tree at each depth cap; raises F_tree but pays drafter cost):
    for dlabel, dval in (("depth7_mb3", 7), ("depth12_mb3", 12), ("depth18_mb3", 18)):
        row = dep[dval]
        note = ("drafter runs %d sequential passes; cost g_d·(%d−7)/7" % (dval, dval)
                if dval != 9 else "deployed")
        cands.append(_candidate(R83, dlabel, row["F_tree"], dval, 32, "depth", note))

    # provenance: recomputed gains must match the banked depth-sweep gains.
    gain_prov_resid = []
    for row in R83["depth_sweep"]:
        gda = gain_drafter_aware(R83, row["F_tree"], 32, row["depth"])
        gmo = gain_m_only(R83, row["F_tree"], 32)
        gain_prov_resid.append(abs(gda - row["gain_drafter_aware"]))
        gain_prov_resid.append(abs(gmo - row["gain_m_only"]))
    max_gain_resid = max(gain_prov_resid)

    # the deployed tree is the TPS-max over ALL candidates (width + depth):
    best = max(cands, key=lambda c: c["tps_operative"])
    topology_lift_max_tps = best["tps_operative"] - INT4_SPEC_CEILING
    topology_lift_max_et = realized_et(R83, best["F_tree"]) - INT4_SPEC_ET \
        if best["depth"] == R83["deployed_depth"] else None  # only E[T]-comparable at same depth

    # width verdict (exact-zero E[T] from width-4; DP never uses rank-5):
    width_lift_et = F_mb4 - F_mb3
    width3_over_width2 = F_mb3 - F_mb2
    uses_rank5 = bool(R83["beyond_width4_M32"]["uses_rank5plus"])

    # depth verdict: raw F_tree monotone-increasing in depth, but drafter-aware TPS peaks at depth-9.
    depths_sorted = sorted(dep)
    f_monotone_up = all(dep[depths_sorted[i + 1]]["F_tree"] >= dep[depths_sorted[i]]["F_tree"]
                        for i in range(len(depths_sorted) - 1))
    # coverage diminishing-returns (PR self-test c): the marginal E[T] per extra depth STRICTLY
    # DECREASES (F_tree is concave / sub-linear in depth) — the measured declining ρ ladder means
    # each deeper rung rescues fewer divergences.
    f_marginals = [dep[depths_sorted[i + 1]]["F_tree"] - dep[depths_sorted[i]]["F_tree"]
                   for i in range(len(depths_sorted) - 1)]
    f_tree_concave_in_depth = all(f_marginals[i + 1] <= f_marginals[i] + 1e-12
                                  for i in range(len(f_marginals) - 1))
    tps_at_depth = {d: tps_operative(R83, gain_drafter_aware(R83, dep[d]["F_tree"], 32, d))
                    for d in depths_sorted}
    depth_tps_argmax = max(tps_at_depth, key=tps_at_depth.get)
    depth_tps_peaks_at_deployed = bool(depth_tps_argmax == R83["deployed_depth"])

    return {
        "candidates": cands,
        "best_candidate_label": best["label"],
        "best_candidate_tps": best["tps_operative"],
        "topology_lift_max_tps": topology_lift_max_tps,       # ~0 (deployed is the TPS max)
        "topology_lift_max_et": topology_lift_max_et,
        "width_lift_et_mb4_minus_mb3": width_lift_et,         # 0.0 exact
        "width3_over_width2_et": width3_over_width2,          # +0.0057 (the binding optimum)
        "uses_rank5plus": uses_rank5,                         # False
        "F_tree_monotone_increasing_in_depth": bool(f_monotone_up),
        "F_tree_concave_in_depth": bool(f_tree_concave_in_depth),   # sub-linear E[T] in depth
        "f_tree_marginals_by_depth": f_marginals,
        "depth_tps_argmax": depth_tps_argmax,                 # 9
        "depth_tps_peaks_at_deployed_depth": depth_tps_peaks_at_deployed,
        "tps_at_depth": tps_at_depth,
        "gain_model_provenance_max_resid": max_gain_resid,    # #83 cost-model reproduction
    }


# --------------------------------------------------------------------------- #
# (2b) Coverage alternative — the ONLY reopener, and it is NOT topology.
# --------------------------------------------------------------------------- #
def coverage_alternative() -> dict[str, Any]:
    """To clear the bar at the DEPLOYED topology, the realized-coverage over-count fraction must
    recover from REALIZED_FRAC to realized_frac_needed = REALIZED_FRAC·(528.48/520.95). This is the
    #199/#178/#193 self-KV / batch-invariance λ→1 lever (drafter/kernel determinism), NOT topology."""
    rf_needed = REALIZED_FRAC * (PRIVATE_BAR_WORSTCASE / INT4_SPEC_CEILING)
    return {
        "realized_frac_current": REALIZED_FRAC,
        "realized_frac_needed_to_clear_bar": rf_needed,
        "coverage_recovery_needed_pp": 100.0 * (rf_needed - REALIZED_FRAC),
        "reachable_by_coverage": bool(REALIZED_FRAC < rf_needed <= 1.0),   # strictly above, ≤ full
        "lever_is_topology": False,
        "note": (
            "the gap is COVERAGE, not topology: lifting the realized rank-1 over-count fraction from "
            f"{REALIZED_FRAC:.4f} to {rf_needed:.4f} (+{100.0*(rf_needed-REALIZED_FRAC):.2f}pp) at "
            "the DEPLOYED depth-9 ρ-optimal tree clears 528.48. That recovery is the λ→1 / "
            "batch-invariance determinism lever (my #199/#213, #178/#193 self-KV), draft/kernel "
            "side — NOT a verify-tree topology change. It is reachable in principle (≤ full "
            "rank-1 coverage 0.7304), so the lane reopens ONLY along the coverage axis."),
    }


# --------------------------------------------------------------------------- #
# (4) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(R83, lift, sweep, cov, prov_resid) -> dict[str, Any]:
    # (a) ceiling round-trip + #83 cost-model reproduction.
    rt_int4 = abs(official_tps(INT4_SPEC_ET) - INT4_SPEC_CEILING)
    rt_reach = abs(official_tps(REACH_DP_ET) - REACH_DP_CENTRAL)
    dep_gain = gain_drafter_aware(R83, R83["deployed_F_tree"], 32, R83["deployed_depth"])
    dep_gain_resid = abs(dep_gain - R83["deployed_gain_drafter_aware"])
    cond_a = bool(rt_int4 <= TOL_ROUNDTRIP and rt_reach <= TOL_ROUNDTRIP
                  and prov_resid <= TOL_PROV
                  and dep_gain_resid <= TOL_GAIN
                  and sweep["gain_model_provenance_max_resid"] <= TOL_GAIN)

    # (b) delta_ET_needed consistency: PR formula == et_from_tps(bar); maps to the 7.53 bar-gap.
    cond_b = bool(lift["et_needed_resid"] <= TOL_PROV
                  and abs(official_tps(lift["et_needed"]) - PRIVATE_BAR_WORSTCASE) <= TOL_ROUNDTRIP
                  and abs(lift["delta_ET_needed_tps"] - GAP_BAR_LITERAL) <= 0.01
                  and lift["delta_ET_needed"] > 0.0)

    # (c) WIDTH exhausted: width-4 adds exactly 0 E[T]; width-3 IS the binding optimum; rank-5 never
    # placed ⇒ no width lift.
    cond_c = bool(sweep["width_lift_et_mb4_minus_mb3"] == 0.0
                  and sweep["width3_over_width2_et"] > 0.0
                  and sweep["uses_rank5plus"] is False)

    # (d) COVERAGE DIMINISHING-RETURNS + DEPTH artifact (PR self-test c): raw E[T] (F_tree) is
    # monotone-increasing AND CONCAVE (sub-linear) in depth — the measured declining ρ ladder makes
    # each deeper rung rescue fewer divergences; yet drafter-aware TPS PEAKS at the deployed depth-9
    # (deeper raises E[T] but lowers TPS); AND the deepest tree FALSELY clears the bar under m_only
    # cost while the honest drafter-aware TPS does NOT.
    d18 = next(c for c in sweep["candidates"] if c["label"] == "depth18_mb3")
    cond_d = bool(sweep["F_tree_monotone_increasing_in_depth"]
                  and sweep["F_tree_concave_in_depth"]
                  and sweep["depth_tps_peaks_at_deployed_depth"]
                  and d18["clears_private_bar_m_only_artifact"] is True
                  and d18["clears_private_bar"] is False)

    # (e) VERDICT closure: topology_lift_max (width+depth, drafter-aware) ≤ 0 < delta_ET_needed_tps
    # ⇒ lane NOT reopenable by topology; AND the coverage alternative is real & reachable
    # (realized_frac_needed strictly above current, ≤ full coverage) ⇒ the gap is coverage.
    no_topo_candidate_clears = all(not c["clears_private_bar"] for c in sweep["candidates"])
    cond_e = bool(sweep["topology_lift_max_tps"] <= 1e-9
                  and lift["delta_ET_needed_tps"] > 0.0
                  and no_topo_candidate_clears
                  and cov["reachable_by_coverage"] is True
                  and cov["lever_is_topology"] is False)

    cond_f = True  # NaN-clean, set by caller.

    conditions = {
        "a_ceiling_roundtrip_and_rho83_cost_model_reproduced": cond_a,
        "b_delta_ET_needed_consistent_and_equals_bar_gap_7p53": cond_b,
        "c_width_exhausted_mb4_zero_and_no_rank5": cond_c,
        "d_coverage_diminishing_Etconcave_TPSdown_m_only_false_clear": cond_d,
        "e_no_topology_clears_bar_reopener_is_coverage": cond_e,
        "f_nan_clean": cond_f,
    }
    return {
        "conditions": conditions,
        "ceiling_gap_topology_headroom_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "roundtrip_int4_tps_err": rt_int4, "roundtrip_reach_tps_err": rt_reach,
            "deployed_gain_resid": dep_gain_resid,
            "reach_dp_provenance_resid": prov_resid,
            "no_topo_candidate_clears_bar": no_topo_candidate_clears,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    R83 = load_rho83()
    # reach-DP provenance: reproduce the banked 5.21888 from the deployed-tree ceiling spine.
    prov_resid = _reach_provenance()

    lift = needed_lift()
    sweep = topology_sweep(R83)
    cov = coverage_alternative()
    st = _selftests(R83, lift, sweep, cov, prov_resid)

    lane_reopenable = bool(sweep["topology_lift_max_tps"] >= lift["delta_ET_needed_tps"])
    topology_lift_needed = lift["delta_ET_needed"]            # TEST metric (E[T] units)

    headline = {
        "ceiling_gap_topology_headroom_self_test_passes":
            bool(st["ceiling_gap_topology_headroom_self_test_passes"]),         # PRIMARY
        "topology_lift_needed": topology_lift_needed,                          # TEST (E[T])
        "topology_lift_needed_tps": lift["delta_ET_needed_tps"],
        "topology_lift_needed_pct": lift["delta_ET_needed_pct"],
        "topology_lift_max_tps": sweep["topology_lift_max_tps"],
        "lane_reopenable": lane_reopenable,
        "operative_ceiling_tps": INT4_SPEC_CEILING,
        "private_bar_worstcase": PRIVATE_BAR_WORSTCASE,
        "best_topology_candidate": sweep["best_candidate_label"],
        "best_topology_candidate_tps": sweep["best_candidate_tps"],
        "width_lift_et_mb4_minus_mb3": sweep["width_lift_et_mb4_minus_mb3"],
        "depth_tps_peaks_at_deployed_depth": sweep["depth_tps_peaks_at_deployed_depth"],
        "reopener_is_coverage_not_topology": bool(not cov["lever_is_topology"]
                                                  and cov["reachable_by_coverage"]),
        "realized_frac_current": cov["realized_frac_current"],
        "realized_frac_needed_to_clear_bar": cov["realized_frac_needed_to_clear_bar"],
        "coverage_recovery_needed_pp": cov["coverage_recovery_needed_pp"],
        "both_ceilings_clear_public_500": bool(INT4_SPEC_CEILING > TARGET
                                               and REACH_DP_CENTRAL > TARGET),
    }

    verdict = _verdict(lane_reopenable)
    handoff = _handoff(lift, sweep, cov, lane_reopenable)

    return {
        "headline": headline,
        "needed_lift": lift,
        "topology_sweep": sweep,
        "coverage_alternative": cov,
        "self_test": st,
        "composition": {
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL, "target_official": TARGET,
            "baseline_tps": BASELINE_TPS,
            "reach_dp_central_tps": REACH_DP_CENTRAL, "reach_dp_et": REACH_DP_ET,
            "int4_spec_ceiling_tps": INT4_SPEC_CEILING, "int4_spec_et": INT4_SPEC_ET,
            "private_bar_worstcase": PRIVATE_BAR_WORSTCASE,
            "realized_frac": REALIZED_FRAC, "rank1_coverage_top1": RANK1_COVERAGE_TOP1,
            "rho_cond_measured": R83["rho_cond_measured"],
            "g_drafter": R83["g_drafter"], "base_drafter_depth": R83["base_drafter_depth"],
            "F_linear8": R83["F_linear8"],
            "deployed_F_tree": R83["deployed_F_tree"], "deployed_depth": R83["deployed_depth"],
            "deployed_max_branch": R83["deployed_max_branch"],
            "deployed_gain_drafter_aware": R83["deployed_gain_drafter_aware"],
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "wirbel#235 (two-ceiling reconcile: operative int4-spec 520.95 / reach-DP 536.66 / "
                "realized_frac 0.9707) x #83 6tghbnjn (ρ-optimal topology sweep: deployed depth-9 "
                "mb-3 F_tree 5.207, gain 0.1817, width/depth sweep, cost model g_d=0.168) x #79 "
                "(measured ρ ladder 0.4165/0.2655/0.1908) x #199 wdyqnx3g (reach-DP E[T] 5.21888) x "
                "#204/#220 pqjnybbf (int4-spec ceiling) x stark#226 tzcc5xuq (private bar 528.48) x "
                "kanna#217 vgovdrjc (K_cal 125.268, step 1.2182). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": ("compliant_spec_et.et_via_reachdp (#175/#184) for the reach-DP ceiling "
                          "provenance; #83 banked topology sweep imported verbatim (not re-run)."),
        },
    }


def _reach_provenance() -> float:
    """Reproduce the banked reach-DP E[T] 5.21888 from the #199 deployed-tree ceiling spine."""
    p = REPO_ROOT / "research/validity/compliant_spec_et/compliant_spec_et_results.json"
    syn = json.loads(p.read_text(encoding="utf-8"))["synthesis"]
    spine = list(syn["brackets"]["both_bugs"]["ceiling_spine"])
    et = C.et_via_reachdp(spine)["et_pmf_mean"]
    return abs(et - REACH_DP_ET)


def _verdict(lane_reopenable: bool) -> str:
    if not lane_reopenable:
        return "COMPLIANT-PRIVATE-500-TOPOLOGY-DEAD-REOPENABLE-ONLY-BY-COVERAGE"
    return "COMPLIANT-PRIVATE-500-TOPOLOGY-REOPENABLE"


def _handoff(lift, sweep, cov, lane_reopenable) -> str:
    return (
        f"the operative int4-spec ceiling 520.95 needs +{lift['delta_ET_needed_tps']:.2f} TPS "
        f"(ΔE[T]={lift['delta_ET_needed']:.4f}, +{lift['delta_ET_needed_pct']:.2f}%) to clear stark "
        f"#226's 528.48 private bar, but EVERY reachable verify-tree topology lever delivers "
        f"≤0 (topology_lift_max={sweep['topology_lift_max_tps']:+.2f} TPS): the deployed depth-9 "
        f"max-branch-3 tree is simultaneously WIDTH-optimal (width-4 adds exactly 0 E[T], DP never "
        f"places rank-5) and DEPTH-optimal at the TPS objective (deeper trees raise E[T] but the "
        f"drafter's depth sequential passes, g_d=0.168, cut TPS — the depth-18 'lift' clears the bar "
        f"ONLY if you ignore drafter cost), so the compliant-PRIVATE-500 lane is TOPOLOGY-DEAD and "
        f"reopens ONLY by recovering realized rank-1 coverage from {cov['realized_frac_current']:.4f} "
        f"to {cov['realized_frac_needed_to_clear_bar']:.4f} "
        f"(+{cov['coverage_recovery_needed_pp']:.2f}pp, the λ→1/batch-invariance lever, NOT topology) "
        f"— hand fern's card + land #71 + the #211 build packet: do NOT spend node budget chasing a "
        f"wider/deeper verify tree; the only private-500 reopener is coverage/determinism. NOT a "
        f"launch. NOT open2."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #235; never fatal).
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
    h, lift, sweep = syn["headline"], syn["needed_lift"], syn["topology_sweep"]
    cov, st = syn["coverage_alternative"], syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("CEILING-GAP TOPOLOGY HEADROOM (PR #244, wirbel) — can topology lift int4-spec 520.95 "
          "> 528.48?", flush=True)
    print("=" * 100, flush=True)
    print(f"  (1) NEEDED LIFT  operative E[T]={lift['et_operative']:.4f} → needed "
          f"E[T]={lift['et_needed']:.4f}  (×528.48/520.95)", flush=True)
    print(f"      delta_ET_needed = {lift['delta_ET_needed']:.4f} E[T] = "
          f"+{lift['delta_ET_needed_tps']:.2f} TPS (+{lift['delta_ET_needed_pct']:.2f}%)  "
          f"[bar-gap 528.48−520.95; the PR's '15.71' is the "
          f"{lift['coverage_overcount_tps_operative_to_reachdp']:.2f} coverage over-count]", flush=True)
    print("-" * 100, flush=True)
    print("  (2/3) TOPOLOGY SWEEP   lever   F_tree   depth  realE[T]  oper-TPS  clears528?  "
          "[m_only-artifact]", flush=True)
    for c in sweep["candidates"]:
        print(f"      {c['label']:<24} {c['lever']:<6} {c['F_tree']:7.4f}  {c['depth']:>4}  "
              f"{c['realized_et']:7.4f}  {c['tps_operative']:8.3f}   "
              f"{str(c['clears_private_bar']):>5}     [{c['tps_operative_m_only_artifact']:7.2f} "
              f"{str(c['clears_private_bar_m_only_artifact']):>5}]", flush=True)
    print(f"      topology_lift_max = {sweep['topology_lift_max_tps']:+.3f} TPS  "
          f"(best={sweep['best_candidate_label']} @ {sweep['best_candidate_tps']:.2f})  "
          f"width-4−width-3 E[T]={sweep['width_lift_et_mb4_minus_mb3']:+.4f}  "
          f"rank5-used={sweep['uses_rank5plus']}", flush=True)
    print(f"      depth: F_tree↑ & CONCAVE in depth (sub-linear E[T])="
          f"{sweep['F_tree_concave_in_depth']}, but drafter-aware TPS peaks at "
          f"depth-{sweep['depth_tps_argmax']} (deeper = higher E[T], LOWER TPS)", flush=True)
    print("-" * 100, flush=True)
    print(f"  (2b) COVERAGE reopener (NOT topology): realized_frac "
          f"{cov['realized_frac_current']:.4f} → {cov['realized_frac_needed_to_clear_bar']:.4f} "
          f"(+{cov['coverage_recovery_needed_pp']:.2f}pp); reachable={cov['reachable_by_coverage']}",
          flush=True)
    print(f"      HEADLINE lane_reopenable (by topology) = {h['lane_reopenable']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) PRIMARY ceiling_gap_topology_headroom_self_test_passes = "
          f"{st['ceiling_gap_topology_headroom_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"      TEST topology_lift_needed = {h['topology_lift_needed']:.5f} E[T] "
          f"(+{h['topology_lift_needed_tps']:.2f} TPS)  delivered by topology = 0", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[ceiling-gap-topology] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, lift, sweep = syn["headline"], syn["needed_lift"], syn["topology_sweep"]
    cov, st = syn["coverage_alternative"], syn["self_test"]
    run = init_wandb_run(
        job_type="ceiling-gap-topology-headroom",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["ceiling-gap-topology-headroom", "issue-211", "validity-gate",
              "compliant-lane-feasibility", "topology-headroom", "private-bar",
              "winners-curse-redraw-budget", "bank-the-analysis"],
        config={
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL, "target_official": TARGET,
            "reach_dp_central_tps": REACH_DP_CENTRAL, "reach_dp_et": REACH_DP_ET,
            "int4_spec_ceiling_tps": INT4_SPEC_CEILING, "int4_spec_et": INT4_SPEC_ET,
            "private_bar_worstcase": PRIVATE_BAR_WORSTCASE, "realized_frac": REALIZED_FRAC,
            "g_drafter": syn["composition"]["g_drafter"],
            "base_drafter_depth": syn["composition"]["base_drafter_depth"],
            "deployed_F_tree": syn["composition"]["deployed_F_tree"],
            "deployed_depth": syn["composition"]["deployed_depth"],
            "deployed_max_branch": syn["composition"]["deployed_max_branch"],
            "wandb_group": args.wandb_group, "baseline_tps": BASELINE_TPS,
            "source_runs": "wirbel#235, #83 6tghbnjn, #199 wdyqnx3g, #220 pqjnybbf, "
                           "stark#226 tzcc5xuq, kanna#217 vgovdrjc",
        },
    )
    if run is None:
        print("[ceiling-gap-topology] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "ceiling_gap_topology_headroom_self_test_passes":
            int(bool(st["ceiling_gap_topology_headroom_self_test_passes"])),       # PRIMARY
        "topology_lift_needed": h["topology_lift_needed"],                         # TEST
        "topology_lift_needed_tps": h["topology_lift_needed_tps"],
        "topology_lift_needed_pct": h["topology_lift_needed_pct"],
        "topology_lift_max_tps": h["topology_lift_max_tps"],
        "lane_reopenable": int(bool(h["lane_reopenable"])),
        "operative_ceiling_tps": INT4_SPEC_CEILING,
        "private_bar_worstcase": PRIVATE_BAR_WORSTCASE,
        "best_topology_candidate_tps": h["best_topology_candidate_tps"],
        "width_lift_et_mb4_minus_mb3": h["width_lift_et_mb4_minus_mb3"],
        "uses_rank5plus": int(bool(sweep["uses_rank5plus"])),
        "depth_tps_peaks_at_deployed_depth": int(bool(h["depth_tps_peaks_at_deployed_depth"])),
        "reopener_is_coverage_not_topology": int(bool(h["reopener_is_coverage_not_topology"])),
        "realized_frac_current": h["realized_frac_current"],
        "realized_frac_needed_to_clear_bar": h["realized_frac_needed_to_clear_bar"],
        "coverage_recovery_needed_pp": h["coverage_recovery_needed_pp"],
        "both_ceilings_clear_public_500": int(bool(h["both_ceilings_clear_public_500"])),
        "delta_ET_needed_tps": lift["delta_ET_needed_tps"],
        "coverage_overcount_tps_operative_to_reachdp":
            lift["coverage_overcount_tps_operative_to_reachdp"],
        "gain_model_provenance_max_resid": sweep["gain_model_provenance_max_resid"],
        "reach_dp_provenance_resid": st["detail"]["reach_dp_provenance_resid"],
        "verdict_topology_dead":
            int(syn["verdict"] == "COMPLIANT-PRIVATE-500-TOPOLOGY-DEAD-REOPENABLE-ONLY-BY-COVERAGE"),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-candidate operative TPS as logged scalars.
    for c in sweep["candidates"]:
        tag = c["label"].replace(".", "p")
        summary[f"cand_tps_{tag}"] = c["tps_operative"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="ceiling_gap_topology_headroom_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[ceiling-gap-topology] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="winners-curse-redraw-budget")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 244, "agent": "wirbel",
        "kind": "ceiling-gap-topology-headroom", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["ceiling_gap_topology_headroom_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["ceiling_gap_topology_headroom_self_test_passes"] = syn["self_test"][
        "ceiling_gap_topology_headroom_self_test_passes"]
    if nan_paths:
        print(f"[ceiling-gap-topology] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[ceiling-gap-topology] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["ceiling_gap_topology_headroom_self_test_passes"]
              and payload["nan_clean"])
        print(f"[ceiling-gap-topology] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
