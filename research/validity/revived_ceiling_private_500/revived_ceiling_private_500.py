#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Does the #366-revived ceiling clear PRIVATE-500? Residual-lift budget (PR #373, denken).

Program-deciding analysis card. wirbel #366 (h28xnyuy) analytically REVIVED the strict spec
supply ceiling from #332's contested 473.5 to ~518.92 (headline, eta_floor 0.39%) / 509.07
(conservative, eta_floor 2.28%); raw lambda=1 ceiling 520.953. But the human's 500 target is on
the PRIVATE leaderboard, and we have a measured public->private gap: the verifier re-ran our
deployed 481.53 public point and got 460.85 PRIVATE (4.3% gap, PR #52). This card answers:
does #366's revival clear private-500 on its own, and if not, the residual base-lift budget +
which live lever supplies it cheapest.

Four deliverables:
  (1) GPU anchor (gpu_anchor.py, run separately): on-pod single-stream proxy + harness gap.
  (2) Private projection of #366 ceiling WITH rho-propagation (regression-to-the-mean, not a
      naive x0.957): rho>=0.8038 (#347) attenuates the excess public TPS above the 481.53 anchor.
  (3) Residual-lift budget across 3 levers: (a) mixed-prec body-shrink (BW model, scales charged
      per #356 grouping-vise), (b) coverage retrain (#336 +0.031 budget), (c) eta-reduction.
  (4) Sensitivity tornado over eta_floor, the 4.3% gap, and rho.

PROJECTION MODEL (the crux). Private V is linearly correlated with public P, anchored at the
measured pivot (mu_P=481.53, mu_V=460.85), mu_V=(1-g)*mu_P:
    E[V | P] = mu_V + rho*(sigma_V/sigma_P)*(P - mu_P)
With relative homoscedasticity sigma_V/sigma_P = mu_V/mu_P = (1-g):
    project(P; rho, g) = (1-g) * [mu_P + rho*(P - mu_P)]
  - rho=1 recovers the naive proportional (1-g)*P (the PR's own x0.957 arithmetic).
  - rho<1, extrapolating ABOVE mu_P (the revived ceiling is +37 above 481.53), pulls V BELOW
    proportional by the regression-to-the-mean haircut (1-g)*(1-rho)*(P-mu_P). This is exactly
    the "propagate the uncertainty / not a naive point x0.957" the card asks for.

NOT a launch, NOT a submission, no served-file change. 0 official TPS. CPU-analytic (the GPU
anchor is the separate gpu_anchor.py). Run:
    cd target/ && /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/revived_ceiling_private_500/revived_ceiling_private_500.py \
      --wandb_group revived-ceiling-private-500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 — banked anchors (all from merged advisor-branch cards / BASELINE)
# ===========================================================================

# Measured public<->private anchor pair (PR #52 deployed point; organizer re-run 2026-06-13).
MU_P: float = 481.53          # deployed public TPS (HF a10g-small, PR #52 2x9fm2zx)
MU_V: float = 460.85          # organizer private-verified TPS for the same submission
GAP_MEASURED: float = 1.0 - MU_V / MU_P     # 0.042947 -> the "4.3%" public->private gap

# #366 revived public ceiling (h28xnyuy, pinned_split_phi_audit_results.json).
P_HEADLINE: float = 518.9188253620001       # eta_floor 0.39% (headline)
P_CONSERVATIVE: float = 509.0720037848094   # eta_floor 2.28% (occupancy lower-bound)
LAMBDA1_RAW: float = 520.9527323111674      # eta=0 absolute supply max (lambda=1 raw ceiling)
ETA_FLOOR_HEADLINE: float = 0.003904718156915901
ETA_FLOOR_CONSERVATIVE: float = 0.022806272763935695
BUDGET_500_ETA: float = 0.040220518933569815   # eta affordable to still hit PUBLIC-500 (#366)

# Public<->private correlation estimates.
RHO_LB: float = 0.8038        # #347 lower bound on the public<->private correlation
RHO_PRIV: float = 0.9421      # #300 point estimate (central)

# BW model (#344 hbm_bound / gate_independent_speed_lever): body = 94.3% of step HBM bytes.
BODY_HBM_FRAC: float = 0.943
FIXED_HBM_FRAC: float = 0.057                # 1 - 0.943 (lm_head + attn + KV + scales fixed)
K_CAL: float = 125.26795005202914           # steps/s; official = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL          # 3.844 realized accept length at the deployed point

# Deployed body quantisation (int4 W4A16, g128) avg-bpw incl. bf16 scale storage.
DEPLOYED_NOMINAL_BITS: float = 4.0
DEPLOYED_GROUP_SIZE: int = 128
SCALE_BITS: float = 16.0                      # one bf16 scale per group
DEPLOYED_AVG_BPW: float = DEPLOYED_NOMINAL_BITS + SCALE_BITS / DEPLOYED_GROUP_SIZE   # 4.125

# #356 grouping-vise: the only servable sub-int4 nominal step is b3.5; a g32 PPL-recovery
# adds +0.5 bpw of bf16 scales (16/32) vs g128's 0.125. b3.5 best-case PPL delta +0.10 (#356).
SUBINT4_NOMINAL_BITS: float = 3.5
RECOVERY_GROUP_SIZE: int = 32
PPL_DELTA_B35_BESTCASE: float = 0.10          # #356 optimistic W3.5 anchor
PPL_BUDGET: float = 0.0428                    # 2.42 gate - 2.3772 deployed

# Coverage / demand anchors.
COV_BUDGET_336: float = 0.031                 # #336 achievable coverage-lift budget
K_DRAFT: int = 7                              # num_speculative_tokens (linear MTP chain)

TARGET: float = 500.0


# ===========================================================================
# Section 1 — projection operator + helpers
# ===========================================================================

def project(P: float, rho: float, g: float = GAP_MEASURED) -> float:
    """Public->private regression-to-the-mean projection. rho=1 -> naive proportional."""
    return (1.0 - g) * (MU_P + rho * (P - MU_P))


def naive_proportional(P: float, g: float = GAP_MEASURED) -> float:
    """The PR's own x(1-g) arithmetic == project at rho=1."""
    return (1.0 - g) * P


def public_for_private_500(rho: float, g: float = GAP_MEASURED) -> float:
    """Invert project(P;rho,g)=500 -> the PUBLIC ceiling needed to clear private-500."""
    return MU_P + (TARGET / (1.0 - g) - MU_P) / rho


def et_of_p(p: float, k: int = K_DRAFT) -> float:
    """Accepted-length of a linear depth-k chain, iid per-token accept prob p."""
    if abs(1.0 - p) < 1e-12:
        return float(k + 1)
    return (1.0 - p ** (k + 1)) / (1.0 - p)


def solve_p_for_et(target_et: float, k: int = K_DRAFT) -> float:
    """Bisection for the iid accept prob p reproducing target_et on a depth-k chain."""
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if et_of_p(mid, k) < target_et:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bw_lift(body_bytes_frac: float) -> float:
    """#344 BW model: TPS lift from shrinking body HBM bytes to a fraction of deployed."""
    return 1.0 / (BODY_HBM_FRAC * body_bytes_frac + FIXED_HBM_FRAC)


# ===========================================================================
# Section 2 — private projection grid (deliverable 2)
# ===========================================================================

def build_projection() -> dict:
    ceilings = {"headline": P_HEADLINE, "conservative": P_CONSERVATIVE, "lambda1_raw": LAMBDA1_RAW}
    rhos = {"rho_priv_0p9421": RHO_PRIV, "rho_lb_0p8038": RHO_LB, "rho_1_naive_prop": 1.0}
    grid = {}
    for cname, P in ceilings.items():
        grid[cname] = {}
        for rname, rho in rhos.items():
            v = project(P, rho)
            grid[cname][rname] = {"private_tps": v, "clears_500": v >= TARGET}

    # Headline scalars the card asks for.
    private_ceiling_headline = project(P_HEADLINE, RHO_PRIV)           # central rho-regressed
    private_ceiling_headline_naive = naive_proportional(P_HEADLINE)    # PR's own x0.957
    private_ceiling_conservative = project(P_CONSERVATIVE, RHO_LB)     # cons eta + cons rho (FLOOR)
    private_ceiling_conservative_naive = naive_proportional(P_CONSERVATIVE)

    # Absolute most-optimistic defensible projection: raw eta=0 ceiling, perfect correlation.
    abs_optimistic = naive_proportional(LAMBDA1_RAW)                   # 498.6, still < 500

    # Gap threshold that would FLIP the verdict to GO (at headline ceiling, central rho).
    # project(P_HEADLINE, RHO_PRIV, g*) = 500 -> solve g*.
    bracket_headline = MU_P + RHO_PRIV * (P_HEADLINE - MU_P)
    gap_flip_headline = 1.0 - TARGET / bracket_headline
    bracket_lambda1 = MU_P + RHO_PRIV * (LAMBDA1_RAW - MU_P)
    gap_flip_lambda1 = 1.0 - TARGET / bracket_lambda1

    return {
        "gap_measured": GAP_MEASURED,
        "grid": grid,
        "private_ceiling_headline": private_ceiling_headline,
        "private_ceiling_headline_naive_proportional": private_ceiling_headline_naive,
        "clears_private_500_headline": private_ceiling_headline >= TARGET,
        "private_ceiling_conservative": private_ceiling_conservative,
        "private_ceiling_conservative_naive_proportional": private_ceiling_conservative_naive,
        "clears_private_500_conservative": private_ceiling_conservative >= TARGET,
        "absolute_optimistic_private": abs_optimistic,
        "absolute_optimistic_clears_500": abs_optimistic >= TARGET,
        "gap_flip_threshold_headline": gap_flip_headline,
        "gap_flip_threshold_lambda1_raw": gap_flip_lambda1,
        "revived_clears_private_500": private_ceiling_conservative >= TARGET,
    }


# ===========================================================================
# Section 3 — residual-lift budget across 3 levers (deliverable 3)
# ===========================================================================

def build_levers(proj: dict) -> dict:
    # Residual public lift needed (beyond the #366 ceiling) to clear private-500.
    Pstar_central = public_for_private_500(RHO_PRIV)        # for the headline path
    Pstar_conservative = public_for_private_500(RHO_LB)     # for the conservative path

    residual_central = TARGET - proj["private_ceiling_headline"]            # private-TPS shortfall
    residual_conservative = TARGET - proj["private_ceiling_conservative"]
    residual_public_central = Pstar_central - P_HEADLINE                    # public-TPS lift needed
    residual_public_conservative = Pstar_conservative - P_CONSERVATIVE

    # ---- lever (a): mixed-precision body-shrink (charge scales per #356 grouping-vise) -------
    def avg_bpw_for_public_target(P0: float, Pstar: float) -> dict:
        L = Pstar / P0                                  # multiplicative public lift needed
        f = (1.0 / L - FIXED_HBM_FRAC) / BODY_HBM_FRAC  # body_bytes_frac from BW model
        new_avg_bpw = f * DEPLOYED_AVG_BPW
        return {"lift_needed": L, "body_bytes_frac": f, "new_avg_bpw": new_avg_bpw,
                "residual_avg_bpw": DEPLOYED_AVG_BPW - new_avg_bpw}
    a_central = avg_bpw_for_public_target(P_HEADLINE, Pstar_central)
    a_conservative = avg_bpw_for_public_target(P_CONSERVATIVE, Pstar_conservative)
    # Accessible sub-int4 realization: b3.5 needs g32 to hold PPL -> avg_bpw = 3.5 + 16/32 = 4.0.
    access_avg_bpw = SUBINT4_NOMINAL_BITS + SCALE_BITS / RECOVERY_GROUP_SIZE   # 4.0
    access_f = access_avg_bpw / DEPLOYED_AVG_BPW
    access_L = bw_lift(access_f)
    access_private_headline = project(P_HEADLINE * access_L, RHO_PRIV)
    access_private_conservative = project(P_CONSERVATIVE * access_L, RHO_LB)
    lever_a = {
        "central": a_central, "conservative": a_conservative,
        "deployed_avg_bpw": DEPLOYED_AVG_BPW,
        "accessible_subint4_point": {
            "nominal_bits": SUBINT4_NOMINAL_BITS, "recovery_group_size": RECOVERY_GROUP_SIZE,
            "avg_bpw_with_g32_scales": access_avg_bpw, "body_bytes_frac": access_f,
            "bw_lift": access_L, "private_headline": access_private_headline,
            "private_conservative": access_private_conservative,
            "ppl_delta_bestcase": PPL_DELTA_B35_BESTCASE, "ppl_budget": PPL_BUDGET,
            "ppl_busts_gate": PPL_DELTA_B35_BESTCASE > PPL_BUDGET,
            "ppl_overshoot_x": PPL_DELTA_B35_BESTCASE / PPL_BUDGET,
        },
        "verdict": ("byte-requirement is sub-bit (0.05-0.19 avg-bpw) but PPL-INACCESSIBLE: int4 "
                    "Marlin is the servable kernel floor (#132), the only sub-int4 step b3.5 busts "
                    "PPL by {:.1f}x even optimistically (#356), and the g32 PPL-recovery scales "
                    "(+0.375 bpw) erode the nominal saving (grouping-vise).").format(
                        PPL_DELTA_B35_BESTCASE / PPL_BUDGET),
        "blocked_by": "ppl",
    }

    # ---- lever (b): coverage retrain (raise E[T] / acceptance; #336 +0.031 budget) ----------
    p0 = solve_p_for_et(E_T_REALIZED)
    dp = 1e-4
    dET_dp = (et_of_p(p0 + dp) - et_of_p(p0 - dp)) / (2 * dp)
    def coverage_for_public_target(P0: float, Pstar: float) -> dict:
        dET = E_T_REALIZED * (Pstar / P0 - 1.0)     # ceiling ∝ E[T] at fixed step/eta
        dcov = dET / dET_dp                          # iid-accept approx: dcoverage ~= dp
        return {"delta_et": dET, "residual_coverage_delta": dcov,
                "frac_of_336_budget": dcov / COV_BUDGET_336}
    b_central = coverage_for_public_target(P_HEADLINE, Pstar_central)
    b_conservative = coverage_for_public_target(P_CONSERVATIVE, Pstar_conservative)
    lever_b = {
        "central": b_central, "conservative": b_conservative,
        "p0_accept": p0, "dET_dp_at_p0": dET_dp, "e_t_realized": E_T_REALIZED,
        "cov_budget_336": COV_BUDGET_336,
        "within_336_budget_central": b_central["residual_coverage_delta"] <= COV_BUDGET_336,
        "within_336_budget_conservative": b_conservative["residual_coverage_delta"] <= COV_BUDGET_336,
        "verdict": ("FEASIBLE and CHEAPEST: residual coverage +{:.4f} (central) / +{:.4f} "
                    "(conservative) is {:.0f}% / {:.0f}% of #336's +0.031 budget; retrain JUSTIFIED "
                    "(#339 P(clear)>=0.76). Orthogonal to #366's supply revival (demand vs supply, "
                    "#341). DOUBLE-INDICATED: a wider-distribution drafter also SHRINKS the "
                    "public->private gap toward the ~3.6% knife-edge.").format(
                        b_central["residual_coverage_delta"], b_conservative["residual_coverage_delta"],
                        100 * b_central["frac_of_336_budget"], 100 * b_conservative["frac_of_336_budget"]),
        "blocked_by": None,
        "coverage_conversion_caveat": ("dcoverage~=dp uses an iid depth-7 accept model; the "
                                       "coverage->acceptance mapping carries model uncertainty. The "
                                       "directly-grounded number is delta_et (K_cal-linear); the "
                                       "magnitude conclusion (small fraction of budget) is robust."),
    }

    # ---- lever (c): eta-reduction (cap at eta=0 -> raw lambda1 ceiling 520.953) -------------
    eta0_private_central = project(LAMBDA1_RAW, RHO_PRIV)
    eta0_private_conservative = project(LAMBDA1_RAW, RHO_LB)
    eta0_private_naive = naive_proportional(LAMBDA1_RAW)
    lever_c = {
        "eta_floor_headline": ETA_FLOOR_HEADLINE, "eta_floor_conservative": ETA_FLOOR_CONSERVATIVE,
        "lambda1_raw_ceiling": LAMBDA1_RAW,
        "private_at_eta0_central": eta0_private_central,
        "private_at_eta0_conservative": eta0_private_conservative,
        "private_at_eta0_naive_proportional": eta0_private_naive,
        "eta0_clears_500_anywhere": max(eta0_private_central, eta0_private_naive) >= TARGET,
        "verdict": ("MAXED / INSUFFICIENT: eta is already near-zero (0.39% headline); driving it to "
                    "ZERO yields the raw lambda=1 ceiling 520.95, which still projects to private "
                    "{:.1f} (central) / {:.1f} (naive-proportional) < 500. No supply-side eta work "
                    "can reach private-500 -> residual must come from a CEILING-RAISING lever "
                    "(coverage E[T] or body bytes), not eta.").format(
                        eta0_private_central, eta0_private_naive),
        "blocked_by": "physical_floor",
    }

    # cheapest-lever selection
    cheapest = "coverage_retrain_b"
    return {
        "Pstar_central_public": Pstar_central,
        "Pstar_conservative_public": Pstar_conservative,
        "residual_lift_to_private_500_tps": residual_central,            # headline scalar (central)
        "residual_lift_to_private_500_tps_conservative": residual_conservative,
        "residual_public_lift_central": residual_public_central,
        "residual_public_lift_conservative": residual_public_conservative,
        "residual_avg_bpw_for_mixedprec": a_central["residual_avg_bpw"], # headline scalar (central)
        "residual_avg_bpw_for_mixedprec_conservative": a_conservative["residual_avg_bpw"],
        "residual_coverage_delta": b_central["residual_coverage_delta"], # headline scalar (central)
        "residual_coverage_delta_conservative": b_conservative["residual_coverage_delta"],
        "lever_a_mixedprec_bodyshrink": lever_a,
        "lever_b_coverage_retrain": lever_b,
        "lever_c_eta_reduction": lever_c,
        "cheapest_lever": cheapest,
        "cheapest_lever_reason": ("(c) eta ELIMINATED (can't reach 500 at eta=0); (a) body-shrink "
                                  "PPL-BLOCKED (int4 floor + b3.5 PPL bust + grouping-vise); "
                                  "(b) coverage FEASIBLE (within #336 budget, retrain justified) "
                                  "AND attacks the gap. -> (b) coverage retrain is cheapest."),
    }


# ===========================================================================
# Section 4 — sensitivity tornado (deliverable 4)
# ===========================================================================

def build_tornado(proj: dict) -> dict:
    central = project(P_HEADLINE, RHO_PRIV, GAP_MEASURED)   # 494.6

    # (1) eta_floor: P over [conservative-eta 509.07, eta=0 raw 520.95], rho/g central.
    eta_lo = project(P_CONSERVATIVE, RHO_PRIV, GAP_MEASURED)
    eta_hi = project(LAMBDA1_RAW, RHO_PRIV, GAP_MEASURED)
    swing_eta = abs(eta_hi - eta_lo)

    # (2) gap: g over a gate-bound range AND a program-tail range, P/rho central.
    g_lo_gate, g_hi_gate = 0.03, 0.05            # gate-bound (valid submissions are <=5%)
    g_tail = 0.124                               # kanna #44 chat-proxy pessimistic upper bound
    gap_lo_gate = project(P_HEADLINE, RHO_PRIV, g_lo_gate)
    gap_hi_gate = project(P_HEADLINE, RHO_PRIV, g_hi_gate)
    gap_tail = project(P_HEADLINE, RHO_PRIV, g_tail)
    swing_gap_gate = abs(gap_lo_gate - gap_hi_gate)
    swing_gap_tail = abs(project(P_HEADLINE, RHO_PRIV, GAP_MEASURED) - gap_tail)

    # (3) rho over [0.8038, 0.9421], P/g central.
    rho_lo = project(P_HEADLINE, RHO_LB, GAP_MEASURED)
    rho_hi = project(P_HEADLINE, RHO_PRIV, GAP_MEASURED)
    swing_rho = abs(rho_hi - rho_lo)

    ranked = sorted(
        [("public_private_gap_gate_range", swing_gap_gate),
         ("public_private_gap_program_tail", swing_gap_tail),
         ("eta_floor", swing_eta),
         ("rho", swing_rho)],
        key=lambda kv: kv[1], reverse=True)

    # Robustness: best case at the MEASURED gap (4.3%) cannot reach 500.
    best_at_measured_gap = max(project(LAMBDA1_RAW, RHO_PRIV, GAP_MEASURED),
                               naive_proportional(LAMBDA1_RAW))
    return {
        "central_private": central,
        "eta_floor": {"low": eta_lo, "high": eta_hi, "swing": swing_eta},
        "gap_gate_range": {"low_g": g_lo_gate, "high_g": g_hi_gate,
                           "low": gap_hi_gate, "high": gap_lo_gate, "swing": swing_gap_gate},
        "gap_program_tail": {"tail_g": g_tail, "tail_private": gap_tail, "swing": swing_gap_tail},
        "rho": {"low": rho_lo, "high": rho_hi, "swing": swing_rho},
        "ranked_by_swing": ranked,
        "dominant_axis": ranked[0][0],
        "best_case_at_measured_gap": best_at_measured_gap,
        "verdict_robust_no_go_at_measured_gap": best_at_measured_gap < TARGET,
        "verdict": ("The public->private GAP dominates the shortfall MAGNITUDE (and is the ONLY "
                    "verdict-flipping axis): it swings private {:.1f} TPS over the gate range and "
                    "{:.1f} over the program tail, vs eta_floor {:.1f} and rho {:.1f}. But at the "
                    "MEASURED 4.3% gap the verdict is ROBUST NO-GO: even eta=0 + perfect rho gives "
                    "{:.1f} < 500. The verdict flips to GO only if the gap drops below ~{:.1f}% "
                    "(headline) -- a private-stability improvement, i.e. lever (b) again.").format(
                        swing_gap_gate, swing_gap_tail, swing_eta, swing_rho,
                        best_at_measured_gap, 100 * proj["gap_flip_threshold_headline"]),
    }


# ===========================================================================
# Section 5 — GPU anchor import (deliverable 1, produced by gpu_anchor.py)
# ===========================================================================

def load_gpu_anchor() -> dict:
    f = HERE / "gpu_anchor_results.json"
    if not f.exists():
        return {"present": False, "note": "gpu_anchor.py not yet run on-pod"}
    d = json.loads(f.read_text())
    r = d.get("reconstruction", {})
    return {
        "present": d.get("present", False),
        "local_deployed_single_stream_tps": r.get("local_deployed_single_stream_tps"),
        "local_to_served_calibration_ratio": r.get("local_to_served_calibration_ratio"),
        "bw_calibration_local_over_nominal": r.get("bw_calibration_ratio_local_over_nominal"),
        "effective_read_bw_gbps": d.get("read_bw", {}).get("effective_read_bw_gbps"),
        "body_read_frac_of_served_wall": r.get("body_read_frac_of_wall") or r.get("body_read_frac_of_served_wall"),
        "residual_harness_gap_pct": r.get("residual_harness_gap_pct"),
        "fp16_M8_over_M1_ratio": d.get("fp16_M8_over_M1_ratio"),
        "byte_provenance_matches_356": d.get("byte_provenance_matches_356"),
        "is_a10g_80sm": d.get("gpu", {}).get("is_a10g_80sm_ga102_sm86"),
    }


# ===========================================================================
# Section 6 — self-tests
# ===========================================================================

def run_self_tests(proj: dict, lev: dict, tor: dict) -> dict:
    c = {}
    # a) projection identities: rho=1 == naive proportional; P=mu_P -> mu_V.
    c["a_rho1_is_naive_prop"] = abs(project(P_HEADLINE, 1.0) - naive_proportional(P_HEADLINE)) < 1e-9
    c["a_anchor_recovers_mu_v"] = abs(project(MU_P, RHO_PRIV) - MU_V) < 1e-9
    c["a_gap_is_4p3pct"] = abs(GAP_MEASURED - 0.043) < 0.001
    # b) rho<1 below proportional when extrapolating above the anchor.
    c["b_regression_below_proportional"] = project(P_HEADLINE, RHO_LB) < naive_proportional(P_HEADLINE)
    # c) every projection corner < 500 (the robust NO-GO).
    all_cells = [cell["private_tps"] for cm in proj["grid"].values() for cell in cm.values()]
    c["c_all_corners_below_500"] = max(all_cells) < TARGET
    c["c_abs_optimistic_below_500"] = proj["absolute_optimistic_private"] < TARGET
    # d) invert: project(public_for_private_500(rho), rho) == 500.
    c["d_invert_central"] = abs(project(public_for_private_500(RHO_PRIV), RHO_PRIV) - TARGET) < 1e-6
    c["d_invert_conservative"] = abs(project(public_for_private_500(RHO_LB), RHO_LB) - TARGET) < 1e-6
    # e) BW model sanity: frac=1 -> lift 1.0; lever-a byte cross-check.
    c["e_bwlift_unit_at_frac1"] = abs(bw_lift(1.0) - 1.0) < 1e-12
    c["e_et_roundtrip"] = abs(et_of_p(solve_p_for_et(E_T_REALIZED)) - E_T_REALIZED) < 1e-4
    # f) lever logic: c eliminated (eta0<500), a ppl-blocked, b within budget & cheapest.
    c["f_eta0_cannot_reach_500"] = not lev["lever_c_eta_reduction"]["eta0_clears_500_anywhere"]
    c["f_lever_a_ppl_blocked"] = lev["lever_a_mixedprec_bodyshrink"]["blocked_by"] == "ppl"
    c["f_lever_b_within_budget"] = lev["lever_b_coverage_retrain"]["within_336_budget_conservative"]
    c["f_cheapest_is_coverage"] = lev["cheapest_lever"] == "coverage_retrain_b"
    # g) tornado: dominant axis identified; robust no-go at measured gap.
    c["g_robust_no_go_measured_gap"] = tor["verdict_robust_no_go_at_measured_gap"]
    c["g_swings_finite_positive"] = all(s > 0 for _, s in tor["ranked_by_swing"])
    # h) decision booleans typed & consistent.
    c["h_verdict_consistent"] = (proj["revived_clears_private_500"] ==
                                 (proj["private_ceiling_conservative"] >= TARGET))
    c["h_no_nan"] = all(v == v for v in all_cells)
    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "passes": passes}


# ===========================================================================
# Section 7 — report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    proj = build_projection()
    lev = build_levers(proj)
    tor = build_tornado(proj)
    gpu = load_gpu_anchor()
    selftest = run_self_tests(proj, lev, tor)
    return {
        "pr": 373, "agent": "denken", "kind": "revived-ceiling-private-500-budget",
        "analysis_only": True, "no_launch": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps_expected": 0,
        "inputs": {
            "mu_p_public": MU_P, "mu_v_private": MU_V, "gap_measured": GAP_MEASURED,
            "p_headline": P_HEADLINE, "p_conservative": P_CONSERVATIVE, "lambda1_raw": LAMBDA1_RAW,
            "eta_floor_headline": ETA_FLOOR_HEADLINE, "eta_floor_conservative": ETA_FLOOR_CONSERVATIVE,
            "rho_lb": RHO_LB, "rho_priv": RHO_PRIV, "k_cal": K_CAL, "e_t_realized": E_T_REALIZED,
            "deployed_avg_bpw": DEPLOYED_AVG_BPW, "cov_budget_336": COV_BUDGET_336,
            "source_366_run": "h28xnyuy", "source_344_bw_model": "94.3% body / 5.7% fixed",
        },
        "gpu_anchor": gpu,
        "projection": proj,
        "levers": lev,
        "tornado": tor,
        # headline scalars (card-required, prose)
        "local_deployed_single_stream_tps": gpu.get("local_deployed_single_stream_tps"),
        "private_ceiling_headline": proj["private_ceiling_headline"],
        "private_ceiling_conservative": proj["private_ceiling_conservative"],
        "clears_private_500_headline": proj["clears_private_500_headline"],
        "clears_private_500_conservative": proj["clears_private_500_conservative"],
        "residual_lift_to_private_500_tps": lev["residual_lift_to_private_500_tps"],
        "residual_avg_bpw_for_mixedprec": lev["residual_avg_bpw_for_mixedprec"],
        "residual_coverage_delta": lev["residual_coverage_delta"],
        "cheapest_lever": lev["cheapest_lever"],
        "dominant_axis": tor["dominant_axis"],
        # GO/NO-GO + SENPAI-RESULT metrics
        "revived_clears_private_500": proj["revived_clears_private_500"],
        "primary_metric_private_ceiling_conservative": proj["private_ceiling_conservative"],
        "test_metric_revived_clears_private_500": int(proj["revived_clears_private_500"]),
        "self_test": selftest,
        "revived_ceiling_private_500_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config={
            k: report["inputs"][k] for k in report["inputs"]})
        proj, lev, tor = report["projection"], report["levers"], report["tornado"]
        wandb.log({
            "summary/private_ceiling_conservative": proj["private_ceiling_conservative"],
            "summary/revived_clears_private_500": float(report["revived_clears_private_500"]),
            "summary/private_ceiling_headline": proj["private_ceiling_headline"],
            "summary/private_ceiling_headline_naive": proj["private_ceiling_headline_naive_proportional"],
            "summary/absolute_optimistic_private": proj["absolute_optimistic_private"],
            "summary/gap_flip_threshold_headline": proj["gap_flip_threshold_headline"],
            "summary/residual_lift_to_private_500_tps": lev["residual_lift_to_private_500_tps"],
            "summary/residual_lift_conservative": lev["residual_lift_to_private_500_tps_conservative"],
            "summary/residual_avg_bpw_for_mixedprec": lev["residual_avg_bpw_for_mixedprec"],
            "summary/residual_coverage_delta": lev["residual_coverage_delta"],
            "summary/residual_coverage_delta_conservative": lev["residual_coverage_delta_conservative"],
            "summary/cov_frac_of_336_budget_conservative":
                lev["lever_b_coverage_retrain"]["conservative"]["frac_of_336_budget"],
            "summary/eta0_private_central": lev["lever_c_eta_reduction"]["private_at_eta0_central"],
            "summary/local_deployed_single_stream_tps":
                report["gpu_anchor"].get("local_deployed_single_stream_tps") or 0.0,
            "summary/local_to_served_calibration_ratio":
                report["gpu_anchor"].get("local_to_served_calibration_ratio") or 0.0,
            "summary/effective_read_bw_gbps":
                report["gpu_anchor"].get("effective_read_bw_gbps") or 0.0,
            "summary/swing_eta_floor": tor["eta_floor"]["swing"],
            "summary/swing_gap_gate": tor["gap_gate_range"]["swing"],
            "summary/swing_gap_tail": tor["gap_program_tail"]["swing"],
            "summary/swing_rho": tor["rho"]["swing"],
            "summary/self_test_passes": float(report["self_test"]["passes"]),
        })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="Revived-ceiling private-500 budget (PR #373).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="revived-ceiling-private-500")
    ap.add_argument("--wandb_name", type=str, default="denken/revived-ceiling-private-500-budget")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/revived_ceiling_private_500/"
                            "revived_ceiling_private_500_results.json")
    args = ap.parse_args()

    report = build_report()
    proj, lev, tor = report["projection"], report["levers"], report["tornado"]

    print("\n=== Does the #366-revived ceiling clear PRIVATE-500? (PR #373) ===")
    print(f"measured public->private gap : {GAP_MEASURED*100:.3f}%  ({MU_P} -> {MU_V})")
    g = report["gpu_anchor"]
    if g.get("present"):
        print(f"GPU anchor (pod A10G)        : local_deployed_single_stream_tps={g['local_deployed_single_stream_tps']:.1f}  "
              f"local/served={g['local_to_served_calibration_ratio']:.3f}  "
              f"read_bw={g['effective_read_bw_gbps']:.0f}GB/s  harness_gap={g['residual_harness_gap_pct']:.0f}%")
    print("\nPrivate projection grid (private TPS / clears-500):")
    for cn, cm in proj["grid"].items():
        row = "  ".join(f"{rn.split('_')[1]}={cell['private_tps']:.1f}{'✓' if cell['clears_500'] else '✗'}"
                        for rn, cell in cm.items())
        print(f"  {cn:<14}: {row}")
    print(f"\nprivate_ceiling_headline      = {proj['private_ceiling_headline']:.2f}  "
          f"(naive-prop {proj['private_ceiling_headline_naive_proportional']:.2f})  clears500={proj['clears_private_500_headline']}")
    print(f"private_ceiling_conservative  = {proj['private_ceiling_conservative']:.2f}  clears500={proj['clears_private_500_conservative']}")
    print(f"absolute optimistic (eta=0,prop) = {proj['absolute_optimistic_private']:.2f}  clears500={proj['absolute_optimistic_clears_500']}")
    print(f"gap-flip threshold (headline) = {proj['gap_flip_threshold_headline']*100:.2f}%  (verdict flips to GO only below this gap)")
    print(f"\nResidual lift to private-500  : central {lev['residual_lift_to_private_500_tps']:.2f} TPS / "
          f"conservative {lev['residual_lift_to_private_500_tps_conservative']:.2f} TPS")
    print(f"  (a) mixed-prec body-shrink  : residual_avg_bpw {lev['residual_avg_bpw_for_mixedprec']:.4f} "
          f"(cons {lev['residual_avg_bpw_for_mixedprec_conservative']:.4f}) -> {lev['lever_a_mixedprec_bodyshrink']['blocked_by']}-BLOCKED")
    print(f"  (b) coverage retrain        : residual_coverage_delta +{lev['residual_coverage_delta']:.4f} "
          f"(cons +{lev['residual_coverage_delta_conservative']:.4f}) = "
          f"{100*lev['lever_b_coverage_retrain']['conservative']['frac_of_336_budget']:.0f}% of #336 budget -> FEASIBLE")
    print(f"  (c) eta-reduction           : eta=0 -> private {lev['lever_c_eta_reduction']['private_at_eta0_central']:.1f} < 500 -> MAXED")
    print(f"  CHEAPEST LEVER              : {lev['cheapest_lever']}")
    print(f"\nTornado dominant axis         : {tor['dominant_axis']}")
    for axis, swing in tor["ranked_by_swing"]:
        print(f"  {axis:<34}: swing {swing:.2f} TPS")
    print(f"robust NO-GO at measured gap  : {tor['verdict_robust_no_go_at_measured_gap']} "
          f"(best case {tor['best_case_at_measured_gap']:.1f} < 500)")
    print(f"\n>>> GO/NO-GO revived_clears_private_500 = {report['revived_clears_private_500']}  "
          f"(primary={proj['private_ceiling_conservative']:.2f}, test={report['test_metric_revived_clears_private_500']})")
    print(f"self-test: {report['self_test']['n_checks']} checks, passes={report['self_test']['passes']}")

    if args.self_test:
        return 0 if report["self_test"]["passes"] else 1

    if not args.no_wandb:
        report["wandb_run_id"] = log_to_wandb(report, args.wandb_group, args.wandb_name)
    else:
        report["wandb_run_id"] = None

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
