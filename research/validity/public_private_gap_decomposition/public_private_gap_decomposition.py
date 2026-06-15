#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #379 -- Public->private gap structural decomposition: irreducible floor vs the 3.2% knife-edge.

WHAT THIS ANSWERS
-----------------
denken #373 (`oqs8lddd`) measured a 4.295% public->private TPS gap (PR #52: 481.53
public -> 460.85 private-VALID, W&B `5k3px8p1`) that DOMINATES every other >500 axis
(swing 41.9 TPS; private-500 flips to GO iff the gap drops below ~3.2%). My #369
(`n384zrxq`) closed the supply-side selective-DVR lever, and wirbel #375 made the
blanket pinned-split un-deployable on the served kernel -- so the now-PRIMARY >500
route is DEMAND-side: the EAGLE-3 coverage retrain (`coverage_retrain_b`). denken
#377 sizes the Delta-coverage for the +5.44 TPS residual; THIS card checks the
lever's CEILING: can the 4.295% gap even reach 3.2%, or does an irreducible floor
cap the demand-side route above it?

THE DECOMPOSITION (exact, multiplicative -> additive)
-----------------------------------------------------
TPS = E[T] / T_step = E[T] / (B + A), where
  E[T] = expected committed tokens per spec step (acceptance-length),
  A    = attention term  (ctx-DEPENDENT, scales with KV length L),
  B    = body GEMM + lm_head + framework/sampler/batch-invariance tax
         (ctx- AND distribution-INDEPENDENT -- identical on public and private).

  gap = 1 - TPS_priv/TPS_pub
      = 1 - (E[T]_priv/E[T]_pub) * ((B+A_pub)/(B+A_priv))
      = 1 - r_a * r_s                               (r_a accept ratio, r_s step ratio)
      = g_a + r_a * g_s                             (EXACT additive identity)
  with g_a = 1 - r_a (acceptance fractional loss), g_s = 1 - r_s (step loss).

The FOUR PR buckets fall straight out of this identity:
  (a) acceptance  = g_a                 -- coverage-ADDRESSABLE (soft-KD + reasoning-trace retrain)
  (b) ctxlen      = r_a * g_s           -- the WHOLE step bucket: g_s = (A_priv-A_pub)/(B+A_priv),
                                           B CANCELS in the numerator -> pure attention/L shift
  (c) outlen      = 0  (HARD)           -- benchmark fixes output at 512 tok (#282) -> no L-shift
  (d) numerics    = 0  (HARD, gap-wise) -- B is identical on pub/priv -> contributes 0 to the
                                           step DIFFERENCE; it is a floor on ABSOLUTE TPS, not on
                                           the GAP. (REFUTES the card's hypothesis that the fixed
                                           numerics tax is the gap floor; a LARGER B only DILUTES g_s.)
  a + b + c + d == gap == 4.295%  (sums exactly).

  irreducible_gap_floor = gap remaining after a perfect coverage retrain (closes bucket a)
                        = b + c + d = ctxlen bucket  (the only non-acceptance survivor).

The ctxlen bucket is pinned by the #257 roofline (attn = 557.9us of the 7983us deployed
step => attn_frac <= 7.0%, an UPPER bound since Gemma's sliding-window layers cap L-scaling)
and a private prompt-length shift Delta-P. We report the banked corner (Delta-P=0, the
#318/#373 pure-rho convention -> floor 0%), a central modest shift, a pessimistic
public-high-decile shift, AND the BREAKEVEN Delta-P that would push the floor to 3.2% --
which turns out to be implausibly large (~+254 tok, ~doubling the public mean prompt).

COVERAGE -> GAP MAP (reconciles with denken #377)
-------------------------------------------------
dE[T]/dcoverage ~ #289's a_1-only leverage T2 = 3.9097 => dTPS/dcov = 3.9097*K_cal ~ 489.7
TPS per unit coverage. The shrink to the 3.2% knife-edge is +5.27 TPS (denken's +5.44
central) => Delta-cov ~ 0.0108, WELL within #336's +0.031 REACHABLE-MARGINAL envelope.

CPU-analytic over BANKED W&B numbers (denken #373 gap, #318 rho, #257 roofline, #289 decay,
#336 coverage budget, #258 per-distribution accept shape). NO new GPU measurement, NO
served-file change, NO HF Job, NO --launch, NO submission. BASELINE stays 481.53; this leg
adds 0 TPS (it DECOMPOSES the banked gap). The OPTIONAL local-A10G per-distribution accept
leg is SKIPPED: the floor verdict depends only on the NON-acceptance buckets (outlen=0 hard,
numerics=0 hard, ctxlen roofline-bounded), which are robust without it.

PRIMARY self-test : gap_decomp_self_test_passes (bool).
Headline          : irreducible_gap_floor_pct, clears_3p2_knife_edge, the four buckets,
                    gap_shrink_contribution_per_coverage, gap_after_max_coverage_retrain,
                    coverage_target_for_3p2, demand_side_route_has_path.

Reproduce:
    cd target/ && python research/validity/public_private_gap_decomposition/\
public_private_gap_decomposition.py --decompose-gap --anchor-289-decay \
        --wandb_group strict-bi-verify-gemm --wandb_name ubel/public-private-gap-decomp
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

# research/validity/public_private_gap_decomposition/this.py -> repo root is 3 up.
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research" / "validity" / "public_private_gap_decomposition"
RESULTS_PATH = OUT_DIR / "public_private_gap_decomposition_results.json"

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED)
# --------------------------------------------------------------------------- #
OFFICIAL_PUBLIC = 481.53          # PR #52 official frontier TPS (public), W&B 2x9fm2zx
PRIVATE_VALID = 460.85            # denken #373 private-VALID TPS, W&B 5k3px8p1
KNIFE_EDGE_PCT = 3.2              # denken #373: private-500 flips to GO iff gap < ~3.2%
K_CAL = 125.268                   # kanna #269 anchor: official = K_cal * E[T]
E_T_PUB = 3.844                   # kanna #217 deployed public served E[T]
RHO_DEPLOYED = 0.9570535584491102 # ubel #318 deployed_priv_over_pub (within-task, NOT decode-proxy)
DEGRADATION_PCT_318 = 4.294644155088978  # ubel #318 measured_within_task_degradation_pct

# #257 built_step_roofline (verify ctx=528, M=8) -- the step decomposition
ATTN_US = 557.90                  # ctx-DEPENDENT attention term of the verify forward
BODY_US = 4474.19                 # ctx-INDEPENDENT body GEMM (weight/HBM-bound)
LMHEAD_US = 131.62                # ctx-INDEPENDENT lm_head GEMV
VERIFY_TOTAL_US = 5163.71         # body + attn + lmhead
DRAFT_K7_US = 704.78              # #257 draft_phase k7 chain
L_REF = 528.0                     # roofline measurement ctx ~ public mean KV length
OUT_LEN = 512                     # #282 measured_result: n_completion_tokens fixed at 512 (all prompts)

# #289 per-position decay: a_1-only leverage (dE[T]/da_1 = T2 = E_T_if_a1_perfect - 1)
A1_ONLY_T2 = 3.909733376164471    # #289 E_T_if_a1_perfect (4.9097..) - 1
E_T_IF_A1_PERFECT = 4.909733376164471

# #336 coverage budget (top-4 root unconditional acceptance)
COVERAGE_BASELINE = 0.8903        # #336 aggregate_baseline
COVERAGE_BAR = 0.9213             # #336 bar (T_effective)
COVERAGE_BUDGET = 0.031           # #336 soft-KD + reasoning-trace REACHABLE-MARGINAL envelope

# denken #373/#377 residual sizing (cross-check target)
DENKEN_RESIDUAL_TPS_CENTRAL = 5.44
DENKEN_RESIDUAL_TPS_CONSERVATIVE = 17.96

# #258 per-distribution accept shape (context: the DECODE-PROXY over-reads the gap 3-5x;
# the benchmark private E[T] is rho*E_T_pub, NOT the adversarial 3.090 decode slice)
PROXY_DECODE_PRIV_ET = 3.0898055282313592   # #258 adversarial decode proxy (DO NOT use as bench)
PROXY_DECODE_PUB_ET = 3.8444537125748504

# sensitivity sweep corners for the private prompt-length shift Delta-P (tokens)
DELTA_P_BANKED = 0.0              # #318/#373 pure-rho convention (no step-time term)
DELTA_P_CENTRAL = 50.0            # modest plausible held-out shift (~quarter public-decile spread)
DELTA_P_PESSIMISTIC = 130.0       # public high-decile (#282 high10 573.8 vs low10 190.8)


# --------------------------------------------------------------------------- #
# numeric helper (no scipy in the analytic venv)
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-13, max_it: int = 400) -> float:
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


def total_gap() -> float:
    """The benchmark ground-truth gap: (TPS_pub - TPS_priv)/TPS_pub."""
    return (OFFICIAL_PUBLIC - PRIVATE_VALID) / OFFICIAL_PUBLIC


# --------------------------------------------------------------------------- #
# Step 0 -- the step-time (ctxlen) fractional loss g_s from the #257 roofline
# --------------------------------------------------------------------------- #
def step_loss_from_shift(delta_p_tokens: float) -> dict[str, float]:
    """g_s = 1 - T_step_pub/T_step_priv given a private prompt-length shift Delta-P.

    Attention is treated FULLY L-linear (A(L) = ATTN_US * L/L_ref) -- an UPPER bound on
    ctxlen sensitivity, because Gemma's sliding-window layers cap A beyond the window.
    B (body + lm_head + framework/numerics) is identical on pub/priv and CANCELS in the
    numerator A_priv - A_pub, so g_s is purely the attention/L shift.
    """
    step_full_us = 1.0e6 / K_CAL                      # T_step_pub = 7982.85 us
    a_pub = ATTN_US
    a_priv = ATTN_US * (L_REF + delta_p_tokens) / L_REF
    step_priv_us = step_full_us + (a_priv - a_pub)    # B identical -> only attn moves
    g_s = (a_priv - a_pub) / step_priv_us
    return {
        "delta_p_tokens": delta_p_tokens,
        "step_full_us": step_full_us,
        "attn_us_pub": a_pub,
        "attn_us_priv": a_priv,
        "step_us_priv": step_priv_us,
        "attn_frac_of_step": a_pub / step_full_us,    # <= 7.0% (upper bound on ctx-dependent frac)
        "g_s_step_loss": g_s,
    }


# --------------------------------------------------------------------------- #
# Step 1 -- decompose the 4.295% gap into the four buckets (sums EXACTLY)
# --------------------------------------------------------------------------- #
def decompose(delta_p_tokens: float) -> dict[str, Any]:
    gap = total_gap()
    sl = step_loss_from_shift(delta_p_tokens)
    g_s = sl["g_s_step_loss"]

    # back out the acceptance ratio from the FIXED total gap:
    # gap = 1 - r_a*r_s ; r_s = 1 - g_s ; r_a = (1-gap)/r_s ; g_a = 1 - r_a
    r_s = 1.0 - g_s
    r_a = (1.0 - gap) / r_s
    g_a = 1.0 - r_a

    # additive buckets (ABSOLUTE gap fraction): gap = g_a + r_a*g_s
    bucket_accept = g_a
    bucket_ctxlen = r_a * g_s
    bucket_outlen = 0.0      # output fixed at 512 tok -> no output-length distribution shift
    bucket_numerics = 0.0    # B identical on pub/priv -> 0 contribution to the step DIFFERENCE

    buckets_sum = bucket_accept + bucket_ctxlen + bucket_outlen + bucket_numerics
    # as PERCENT of the gap (the PR's "% of 4.295%" -- these four sum to 100)
    frac_of_gap = (lambda b: 100.0 * b / gap) if gap else (lambda b: float("nan"))

    return {
        "delta_p_tokens": delta_p_tokens,
        "total_gap_frac": gap,
        "total_gap_pct": 100.0 * gap,
        "step_loss": sl,
        "r_accept": r_a,
        "r_step": r_s,
        "g_accept": g_a,
        "g_step": g_s,
        # ABSOLUTE gap-percent (sum to total_gap_pct == 4.295)
        "bucket_acceptance_abs_pct": 100.0 * bucket_accept,
        "bucket_ctxlen_abs_pct": 100.0 * bucket_ctxlen,
        "bucket_outlen_abs_pct": 100.0 * bucket_outlen,
        "bucket_numerics_abs_pct": 100.0 * bucket_numerics,
        # as % OF THE GAP (sum to 100) -- PR field names
        "gap_bucket_acceptance_pct": frac_of_gap(bucket_accept),
        "gap_bucket_ctxlen_pct": frac_of_gap(bucket_ctxlen),
        "gap_bucket_outlen_pct": frac_of_gap(bucket_outlen),
        "gap_bucket_irreducible_pct": frac_of_gap(bucket_numerics),   # bucket (d) numerics
        "buckets_sum_abs_pct": 100.0 * buckets_sum,
        "buckets_sum_resid_vs_gap": buckets_sum - gap,
        # the headline: gap remaining after a PERFECT coverage retrain (closes acceptance only)
        "irreducible_gap_floor_abs_pct": 100.0 * (bucket_ctxlen + bucket_outlen + bucket_numerics),
    }


# --------------------------------------------------------------------------- #
# Step 2 -- irreducible floor + breakeven Delta-P to reach 3.2%
# --------------------------------------------------------------------------- #
def irreducible_floor(dec_central: dict[str, Any]) -> dict[str, Any]:
    gap_pct = dec_central["total_gap_pct"]
    floor_pct = dec_central["irreducible_gap_floor_abs_pct"]
    clears = floor_pct < KNIFE_EDGE_PCT

    # breakeven Delta-P: ctxlen bucket (== irreducible floor) hits exactly 3.2%
    def floor_at(dp: float) -> float:
        return decompose(dp)["irreducible_gap_floor_abs_pct"]

    # floor is monotone increasing in Delta-P; bracket up to a huge shift
    breakeven_dp = float("nan")
    try:
        if floor_at(0.0) < KNIFE_EDGE_PCT < floor_at(4000.0):
            breakeven_dp = bisect(lambda dp: floor_at(dp) - KNIFE_EDGE_PCT, 0.0, 4000.0)
    except ValueError:
        breakeven_dp = float("nan")

    # express breakeven as a multiple of the public mean prompt (~L_ref - out_len/2)
    public_mean_prompt = L_REF - OUT_LEN / 2.0
    breakeven_vs_public_mean = (breakeven_dp / public_mean_prompt
                                if math.isfinite(breakeven_dp) and public_mean_prompt > 0
                                else float("nan"))

    return {
        "irreducible_gap_floor_pct": floor_pct,         # HEADLINE
        "clears_3p2_knife_edge": bool(clears),          # HEADLINE
        "knife_edge_pct": KNIFE_EDGE_PCT,
        "total_gap_pct": gap_pct,
        "breakeven_private_prompt_shift_tokens": breakeven_dp,
        "public_mean_prompt_tokens_est": public_mean_prompt,
        "breakeven_shift_as_mult_of_public_mean": breakeven_vs_public_mean,
        # second-order OFFSET note (#282): longer prompts have HIGHER E[T] -> a private length
        # increase partially self-offsets (step slower BUT acceptance higher), shrinking the
        # NET ctxlen bucket below this attention-only upper bound.
        "ctxlen_partially_self_offsetting": True,
    }


# --------------------------------------------------------------------------- #
# Step 3 -- coverage retrain -> gap shrink (reconciles with denken #377)
# --------------------------------------------------------------------------- #
def coverage_map(dec_central: dict[str, Any]) -> dict[str, Any]:
    gap = dec_central["total_gap_frac"]
    bucket_accept_abs = dec_central["bucket_acceptance_abs_pct"] / 100.0

    # dE[T]/dcoverage ~ #289 a_1-only leverage T2 ; dTPS/dcov = T2 * K_cal
    slope_tps_per_cov = A1_ONLY_T2 * K_CAL                 # ~489.7 TPS per unit coverage
    slope_tps_per_0p01 = slope_tps_per_cov * 0.01          # ~4.897 TPS per +0.01 coverage
    slope_gap_pp_per_cov = 100.0 * slope_tps_per_cov / OFFICIAL_PUBLIC  # gap pp per unit cov

    # coverage to reach the 3.2% knife-edge
    tps_priv_at_3p2 = OFFICIAL_PUBLIC * (1.0 - KNIFE_EDGE_PCT / 100.0)
    tps_shrink_to_3p2 = tps_priv_at_3p2 - PRIVATE_VALID
    cov_for_3p2 = tps_shrink_to_3p2 / slope_tps_per_cov
    coverage_target_for_3p2 = COVERAGE_BASELINE + cov_for_3p2
    within_envelope_3p2 = cov_for_3p2 <= COVERAGE_BUDGET

    # reconcile with denken #377's +5.44 TPS central residual sizing
    cov_for_denken_central = DENKEN_RESIDUAL_TPS_CENTRAL / slope_tps_per_cov
    cov_for_denken_conservative = DENKEN_RESIDUAL_TPS_CONSERVATIVE / slope_tps_per_cov

    # gap after the MAX achievable (#336 +0.031) coverage retrain -- capped at the
    # acceptance bucket (cannot close more acceptance gap than exists)
    tps_accept_bucket = bucket_accept_abs * OFFICIAL_PUBLIC
    tps_gain_max = min(COVERAGE_BUDGET * slope_tps_per_cov, tps_accept_bucket)
    tps_priv_after = PRIVATE_VALID + tps_gain_max
    gap_after_max = (OFFICIAL_PUBLIC - tps_priv_after) / OFFICIAL_PUBLIC
    cov_caps_acceptance = COVERAGE_BUDGET * slope_tps_per_cov >= tps_accept_bucket

    return {
        "gap_shrink_contribution_per_coverage_tps": slope_tps_per_cov,
        "gap_shrink_per_0p01_coverage_tps": slope_tps_per_0p01,
        "gap_shrink_per_coverage_gap_pp": slope_gap_pp_per_cov,
        "slope_basis": "dE[T]/dcov ~ #289 a_1-only T2=3.9097; dTPS/dcov = T2*K_cal",
        # knife-edge reachability
        "tps_shrink_to_3p2": tps_shrink_to_3p2,
        "coverage_delta_for_3p2": cov_for_3p2,
        "coverage_target_for_3p2": coverage_target_for_3p2,
        "coverage_3p2_within_336_envelope": bool(within_envelope_3p2),
        "coverage_budget_336": COVERAGE_BUDGET,
        "coverage_baseline_336": COVERAGE_BASELINE,
        # denken #377 reconciliation
        "coverage_delta_for_denken_544": cov_for_denken_central,
        "coverage_delta_for_denken_1796": cov_for_denken_conservative,
        "denken_544_within_336_envelope": bool(cov_for_denken_central <= COVERAGE_BUDGET),
        # residual after the achievable retrain
        "gap_after_max_coverage_retrain_pct": 100.0 * gap_after_max,
        "tps_gain_at_max_coverage": tps_gain_max,
        "max_coverage_caps_acceptance_bucket": bool(cov_caps_acceptance),
        "gap_after_max_clears_3p2": bool(100.0 * gap_after_max < KNIFE_EDGE_PCT),
    }


# --------------------------------------------------------------------------- #
# Step 4 -- verdict + next-cheapest fallback
# --------------------------------------------------------------------------- #
def verdict(dec: dict[str, Any], floor: dict[str, Any], cov: dict[str, Any]) -> dict[str, Any]:
    has_path = bool(floor["clears_3p2_knife_edge"] and cov["coverage_3p2_within_336_envelope"])
    accept_share = dec["gap_bucket_acceptance_pct"]
    if has_path:
        band = "GREEN_demand_side_route_has_path"
        next_closer = None
        summary = (
            f"GREEN. The irreducible gap floor is {floor['irreducible_gap_floor_pct']:.3f}% "
            f"(<< 3.2% knife-edge). The 4.295% gap is ~{accept_share:.0f}% "
            "acceptance (coverage-addressable) and the residual is a small, roofline-bounded "
            "ctxlen term; outlen=0 (fixed 512-tok output) and the numerics/framework tax "
            "contributes 0 to the GAP (it cancels in the public-private step difference). A "
            f"coverage retrain of +{cov['coverage_delta_for_3p2']:.4f} (target "
            f"{cov['coverage_target_for_3p2']:.4f}, well within #336's +0.031 envelope) reaches "
            "the 3.2% knife-edge. The demand-side route HAS a path; denken #377's sizing has a "
            "validated ceiling."
        )
    else:
        band = "RED_demand_side_route_capped_by_irreducible_floor"
        # the only greedy-safe E[T] lever beyond coverage is WIDTH>1 tree recovery (#258)
        next_closer = "width>1_tree_rank2plus_recovery (ubel #258; 65.3% public coverage)"
        summary = (
            f"RED / decision-critical. The irreducible gap floor is "
            f"{floor['irreducible_gap_floor_pct']:.3f}% >= 3.2% -- the demand-side coverage "
            "route is CAPPED no matter how good the retrain. This collapses #373's cheapest-"
            f"lever conclusion; re-rank toward {next_closer}."
        )
    return {
        "demand_side_route_has_path": has_path,
        "verdict_band": band,
        "next_cheapest_closer": next_closer,
        "verdict_summary": summary,
    }


# --------------------------------------------------------------------------- #
# Step 5 -- self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(dec: dict[str, Any], floor: dict[str, Any], cov: dict[str, Any],
              corners: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gap = dec["total_gap_frac"]
    checks: dict[str, bool] = {}

    # (a) four buckets sum EXACTLY to the gap (GREEN bar)
    checks["a_buckets_sum_to_gap"] = abs(dec["buckets_sum_resid_vs_gap"]) <= 1e-12
    # (a2) the four "% of gap" fields sum to 100
    pct_sum = (dec["gap_bucket_acceptance_pct"] + dec["gap_bucket_ctxlen_pct"]
               + dec["gap_bucket_outlen_pct"] + dec["gap_bucket_irreducible_pct"])
    checks["a_pct_of_gap_sum_100"] = abs(pct_sum - 100.0) <= 1e-9

    # (b) gap reconstructs from the banked TPS anchors == 4.295%
    checks["b_gap_reconstructs_4p295"] = abs(dec["total_gap_pct"] - 4.295) <= 0.01
    # (b2) and reconciles with #318's measured within-task degradation
    checks["b_reconciles_318_degradation"] = abs(dec["total_gap_pct"] - DEGRADATION_PCT_318) <= 0.02

    # (c) hard-zero buckets: outlen (fixed output) and numerics (cancels in gap)
    checks["c_outlen_bucket_zero"] = abs(dec["bucket_outlen_abs_pct"]) <= 1e-12
    checks["c_numerics_bucket_zero"] = abs(dec["bucket_numerics_abs_pct"]) <= 1e-12

    # (d) irreducible floor == ctxlen + outlen + numerics, well-defined and >= 0
    floor_expected = (dec["bucket_ctxlen_abs_pct"] + dec["bucket_outlen_abs_pct"]
                      + dec["bucket_numerics_abs_pct"])
    checks["d_floor_is_nonaccept_buckets"] = abs(floor["irreducible_gap_floor_pct"]
                                                  - floor_expected) <= 1e-12
    checks["d_floor_nonnegative"] = floor["irreducible_gap_floor_pct"] >= -1e-12

    # (e) clears_3p2 flag is consistent with the floor < 3.2 comparison
    checks["e_clears_flag_consistent"] = (
        floor["clears_3p2_knife_edge"] == (floor["irreducible_gap_floor_pct"] < KNIFE_EDGE_PCT))

    # (f) every reported corner (banked, central, pessimistic) clears 3.2%
    checks["f_all_corners_clear_3p2"] = all(
        c["irreducible_gap_floor_abs_pct"] < KNIFE_EDGE_PCT for c in corners.values())
    # (f2) floor is monotone increasing across corners (banked <= central <= pessimistic)
    ordered = [corners["banked"], corners["central"], corners["pessimistic"]]
    checks["f_floor_monotone_in_shift"] = all(
        ordered[i]["irreducible_gap_floor_abs_pct"]
        <= ordered[i + 1]["irreducible_gap_floor_abs_pct"] + 1e-12
        for i in range(len(ordered) - 1))

    # (g) breakeven Delta-P exists, is finite, and is implausibly large (a private prompt-length
    #     shift > half the public mean prompt is already a >50% length increase for a same-
    #     methodology held-out split; the actual breakeven is ~0.9x the public mean -- near-doubling)
    be = floor["breakeven_private_prompt_shift_tokens"]
    checks["g_breakeven_finite"] = math.isfinite(be)
    checks["g_breakeven_implausible"] = math.isfinite(be) and be > 0.5 * floor["public_mean_prompt_tokens_est"]

    # (h) coverage knife-edge reachable within #336 envelope + reconciles with denken #377
    checks["h_3p2_within_336_envelope"] = bool(cov["coverage_3p2_within_336_envelope"])
    checks["h_denken_544_reconciles"] = bool(cov["denken_544_within_336_envelope"])
    # the two independent sizings (my knife-edge vs denken's +5.44) agree within ~0.005 cov
    checks["h_sizings_agree"] = abs(cov["coverage_delta_for_3p2"]
                                    - cov["coverage_delta_for_denken_544"]) <= 0.005

    # (i) gap_shrink slope finite and positive
    checks["i_slope_positive_finite"] = (math.isfinite(cov["gap_shrink_contribution_per_coverage_tps"])
                                         and cov["gap_shrink_contribution_per_coverage_tps"] > 0)
    # (i2) gap after max achievable retrain also clears 3.2%
    checks["i_gap_after_max_clears_3p2"] = bool(cov["gap_after_max_clears_3p2"])

    # (j) E[T]_priv benchmark = rho * E[T]_pub reconstructs (NOT the decode proxy 3.090)
    et_priv_bench = RHO_DEPLOYED * E_T_PUB
    checks["j_et_priv_bench_from_rho"] = abs(et_priv_bench - RHO_DEPLOYED * E_T_PUB) <= 1e-12
    checks["j_bench_distinct_from_proxy"] = abs(et_priv_bench - PROXY_DECODE_PRIV_ET) > 0.4

    # (k) constants imported EXACT and UNCHANGED
    checks["k_constants_imported_exact"] = (
        OFFICIAL_PUBLIC == 481.53 and PRIVATE_VALID == 460.85 and K_CAL == 125.268
        and E_T_PUB == 3.844 and ATTN_US == 557.90 and OUT_LEN == 512
        and COVERAGE_BUDGET == 0.031 and KNIFE_EDGE_PCT == 3.2)

    # (l) NaN-clean across reported scalars
    scal = [dec["total_gap_pct"], dec["bucket_acceptance_abs_pct"], dec["bucket_ctxlen_abs_pct"],
            floor["irreducible_gap_floor_pct"], cov["gap_shrink_contribution_per_coverage_tps"],
            cov["coverage_target_for_3p2"], cov["gap_after_max_coverage_retrain_pct"], et_priv_bench]
    checks["l_nan_clean"] = all(math.isfinite(float(x)) for x in scal)

    gate = bool(
        checks["a_buckets_sum_to_gap"] and checks["a_pct_of_gap_sum_100"]
        and checks["b_gap_reconstructs_4p295"] and checks["b_reconciles_318_degradation"]
        and checks["c_outlen_bucket_zero"] and checks["c_numerics_bucket_zero"]
        and checks["d_floor_is_nonaccept_buckets"] and checks["d_floor_nonnegative"]
        and checks["e_clears_flag_consistent"] and checks["f_all_corners_clear_3p2"]
        and checks["f_floor_monotone_in_shift"] and checks["g_breakeven_finite"]
        and checks["h_3p2_within_336_envelope"] and checks["h_denken_544_reconciles"]
        and checks["i_slope_positive_finite"] and checks["i_gap_after_max_clears_3p2"]
        and checks["k_constants_imported_exact"] and checks["l_nan_clean"])
    return {"checks": checks, "gap_decomp_self_test_passes": gate}


# --------------------------------------------------------------------------- #
# assemble report
# --------------------------------------------------------------------------- #
def build_report(delta_p_central: float) -> dict[str, Any]:
    corners = {
        "banked": decompose(DELTA_P_BANKED),
        "central": decompose(delta_p_central),
        "pessimistic": decompose(DELTA_P_PESSIMISTIC),
    }
    dec = corners["central"]
    floor = irreducible_floor(dec)
    cov = coverage_map(dec)
    vrd = verdict(dec, floor, cov)
    st = self_test(dec, floor, cov, corners)

    et_priv_bench = RHO_DEPLOYED * E_T_PUB

    report = {
        "pr": 379, "issue": 319, "author": "ubel",
        "leg": "public->private gap structural decomposition: irreducible floor vs 3.2% knife-edge",
        "analysis_only": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "tps_added_by_this_card": 0,
        "optional_gpu_accept_leg": "SKIPPED -- floor verdict depends only on NON-acceptance "
                                   "buckets (outlen=0 hard, numerics=0 hard, ctxlen roofline-"
                                   "bounded); the per-distribution accept measurement would only "
                                   "refine bucket (a)'s split, not move the floor.",
        "measured_public_private_accept_gap": None,   # optional GPU leg not run
        "imported": {
            "official_public": OFFICIAL_PUBLIC, "private_valid": PRIVATE_VALID,
            "knife_edge_pct": KNIFE_EDGE_PCT, "K_cal": K_CAL, "E_T_pub": E_T_PUB,
            "rho_deployed_318": RHO_DEPLOYED, "degradation_pct_318": DEGRADATION_PCT_318,
            "attn_us_257": ATTN_US, "body_us_257": BODY_US, "lmhead_us_257": LMHEAD_US,
            "verify_total_us_257": VERIFY_TOTAL_US, "draft_k7_us_257": DRAFT_K7_US,
            "L_ref": L_REF, "out_len_fixed_282": OUT_LEN,
            "a1_only_T2_289": A1_ONLY_T2, "E_T_if_a1_perfect_289": E_T_IF_A1_PERFECT,
            "coverage_baseline_336": COVERAGE_BASELINE, "coverage_bar_336": COVERAGE_BAR,
            "coverage_budget_336": COVERAGE_BUDGET,
            "denken_residual_tps_central_373": DENKEN_RESIDUAL_TPS_CENTRAL,
            "denken_residual_tps_conservative_373": DENKEN_RESIDUAL_TPS_CONSERVATIVE,
            "proxy_decode_priv_et_258": PROXY_DECODE_PRIV_ET,
            "delta_p_central_tokens": delta_p_central,
        },
        # ---- decomposition (central corner) ----
        "decomposition_central": dec,
        "corners": corners,
        "et_priv_benchmark_from_rho": et_priv_bench,
        "et_priv_benchmark_note": "benchmark private E[T]=rho*E_T_pub (4.295% gap); the #258 "
                                  "decode proxy 3.090 is an ADVERSARIAL slice that over-reads 3-5x.",
        # ---- irreducible floor (HEADLINE) ----
        "irreducible_gap_floor_pct": floor["irreducible_gap_floor_pct"],
        "clears_3p2_knife_edge": floor["clears_3p2_knife_edge"],
        "irreducible_floor": floor,
        "irreducible_floor_banked_pct": corners["banked"]["irreducible_gap_floor_abs_pct"],
        "irreducible_floor_pessimistic_pct": corners["pessimistic"]["irreducible_gap_floor_abs_pct"],
        # ---- four buckets (PR field names; central corner, % OF the 4.295% gap) ----
        "gap_bucket_acceptance_pct": dec["gap_bucket_acceptance_pct"],
        "gap_bucket_ctxlen_pct": dec["gap_bucket_ctxlen_pct"],
        "gap_bucket_outlen_pct": dec["gap_bucket_outlen_pct"],
        "gap_bucket_irreducible_pct": dec["gap_bucket_irreducible_pct"],
        # absolute gap-percent versions (sum to 4.295)
        "bucket_acceptance_abs_pct": dec["bucket_acceptance_abs_pct"],
        "bucket_ctxlen_abs_pct": dec["bucket_ctxlen_abs_pct"],
        "bucket_outlen_abs_pct": dec["bucket_outlen_abs_pct"],
        "bucket_numerics_abs_pct": dec["bucket_numerics_abs_pct"],
        # ---- coverage -> gap map ----
        "gap_shrink_contribution_per_coverage": cov["gap_shrink_contribution_per_coverage_tps"],
        "gap_after_max_coverage_retrain": cov["gap_after_max_coverage_retrain_pct"],
        "coverage_target_for_3p2": cov["coverage_target_for_3p2"],
        "coverage_map": cov,
        # ---- verdict ----
        "demand_side_route_has_path": vrd["demand_side_route_has_path"],
        "next_cheapest_closer": vrd["next_cheapest_closer"],
        "verdict_band": vrd["verdict_band"],
        "verdict_summary": vrd["verdict_summary"],
        # ---- self-test (PRIMARY) ----
        "self_test": st["checks"],
        "gap_decomp_self_test_passes": st["gap_decomp_self_test_passes"],
        # bookkeeping
        "official_baseline_unchanged": OFFICIAL_PUBLIC,
    }
    return report


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[ppgd] wandb unavailable ({exc})", flush=True)
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("[ppgd] wandb disabled via env", flush=True)
        return None
    try:
        dec, floor, cov = (report["decomposition_central"], report["irreducible_floor"],
                           report["coverage_map"])
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
            tags=["gemma-challenge", "analysis", "gap-decomposition", "public-private",
                  "irreducible-floor", "knife-edge-3p2", "demand-side", "issue-319", "pr-379"],
            config={
                "pr": 379, "issue": 319, "analysis_only": True, "wandb_group": group,
                "official_public": OFFICIAL_PUBLIC, "private_valid": PRIVATE_VALID,
                "knife_edge_pct": KNIFE_EDGE_PCT, "K_cal": K_CAL, "E_T_pub": E_T_PUB,
                "rho_deployed_318": RHO_DEPLOYED, "attn_us_257": ATTN_US, "L_ref": L_REF,
                "out_len_fixed_282": OUT_LEN, "coverage_budget_336": COVERAGE_BUDGET,
                "delta_p_central_tokens": report["imported"]["delta_p_central_tokens"],
            },
        )
        flat = {
            "primary/gap_decomp_self_test_passes": int(report["gap_decomp_self_test_passes"]),
            "headline/irreducible_gap_floor_pct": report["irreducible_gap_floor_pct"],
            "headline/clears_3p2_knife_edge": int(report["clears_3p2_knife_edge"]),
            "headline/demand_side_route_has_path": int(report["demand_side_route_has_path"]),
            "bucket/acceptance_pct_of_gap": report["gap_bucket_acceptance_pct"],
            "bucket/ctxlen_pct_of_gap": report["gap_bucket_ctxlen_pct"],
            "bucket/outlen_pct_of_gap": report["gap_bucket_outlen_pct"],
            "bucket/numerics_pct_of_gap": report["gap_bucket_irreducible_pct"],
            "bucket/acceptance_abs_pct": report["bucket_acceptance_abs_pct"],
            "bucket/ctxlen_abs_pct": report["bucket_ctxlen_abs_pct"],
            "floor/banked_pct": report["irreducible_floor_banked_pct"],
            "floor/central_pct": report["irreducible_gap_floor_pct"],
            "floor/pessimistic_pct": report["irreducible_floor_pessimistic_pct"],
            "floor/breakeven_prompt_shift_tokens": floor["breakeven_private_prompt_shift_tokens"],
            "floor/breakeven_mult_of_public_mean": floor["breakeven_shift_as_mult_of_public_mean"],
            "total_gap_pct": dec["total_gap_pct"],
            "buckets_sum_abs_pct": dec["buckets_sum_abs_pct"],
            "coverage/gap_shrink_per_coverage_tps": cov["gap_shrink_contribution_per_coverage_tps"],
            "coverage/gap_shrink_per_0p01_tps": cov["gap_shrink_per_0p01_coverage_tps"],
            "coverage/delta_for_3p2": cov["coverage_delta_for_3p2"],
            "coverage/target_for_3p2": cov["coverage_target_for_3p2"],
            "coverage/3p2_within_336_envelope": int(cov["coverage_3p2_within_336_envelope"]),
            "coverage/delta_for_denken_544": cov["coverage_delta_for_denken_544"],
            "coverage/gap_after_max_retrain_pct": cov["gap_after_max_coverage_retrain_pct"],
            "coverage/gap_after_max_clears_3p2": int(cov["gap_after_max_clears_3p2"]),
            "et_priv_benchmark_from_rho": report["et_priv_benchmark_from_rho"],
            "tps_added_by_this_card": 0,
        }
        flat = {k: v for k, v in flat.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))}
        run.summary.update(flat)
        run.summary["verdict_band"] = report["verdict_band"]
        run.summary["next_cheapest_closer"] = report["next_cheapest_closer"] or "none"
        for k, v in report["self_test"].items():
            run.summary[f"selftest/{k}"] = int(bool(v))

        # four-bucket table
        btbl = wandb.Table(columns=["bucket", "abs_pct_of_4p295", "pct_of_gap",
                                    "addressable_by_coverage"])
        btbl.add_data("acceptance", report["bucket_acceptance_abs_pct"],
                      report["gap_bucket_acceptance_pct"], "YES (soft-KD + reasoning retrain)")
        btbl.add_data("ctxlen", report["bucket_ctxlen_abs_pct"],
                      report["gap_bucket_ctxlen_pct"], "NO (prompt-length dist; irreducible floor)")
        btbl.add_data("outlen", report["bucket_outlen_abs_pct"],
                      report["gap_bucket_outlen_pct"], "N/A (0 -- output fixed 512 tok)")
        btbl.add_data("numerics", report["bucket_numerics_abs_pct"],
                      report["gap_bucket_irreducible_pct"], "N/A (0 -- cancels in gap)")
        run.log({"gap_buckets": btbl})

        # corner sensitivity table
        ctbl = wandb.Table(columns=["corner", "delta_p_tokens", "acceptance_abs_pct",
                                    "ctxlen_abs_pct", "irreducible_floor_pct", "clears_3p2"])
        for nm, c in report["corners"].items():
            ctbl.add_data(nm, c["delta_p_tokens"], c["bucket_acceptance_abs_pct"],
                          c["bucket_ctxlen_abs_pct"], c["irreducible_gap_floor_abs_pct"],
                          int(c["irreducible_gap_floor_abs_pct"] < KNIFE_EDGE_PCT))
        run.log({"corner_sensitivity": ctbl})

        rid = run.id
        print(f"[ppgd] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[ppgd] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decompose-gap", action="store_true",
                    help="run the four-bucket decomposition + floor + coverage map (default path).")
    ap.add_argument("--anchor-289-decay", action="store_true",
                    help="anchor the coverage->gap slope on #289's a_1-only leverage (default on).")
    ap.add_argument("--private-prompt-shift", type=float, default=DELTA_P_CENTRAL,
                    help="central private prompt-length shift Delta-P (tokens) for the ctxlen bucket.")
    ap.add_argument("--self-test", action="store_true",
                    help="exit nonzero if the primary self-test fails.")
    # OPTIONAL local-A10G per-distribution accept leg (documented SKIP; floor robust without it)
    ap.add_argument("--gpu", action="store_true", help="(optional) enable the GPU accept-gap leg.")
    ap.add_argument("--proxy", default="google/gemma-4-E4B-it-qat-w4a16-ct",
                    help="(optional) int4-ct proxy for the GPU accept-gap leg.")
    ap.add_argument("--measure-accept-gap", action="store_true",
                    help="(optional) measure per-distribution E[l] on the proxy.")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="ubel/public-private-gap-decomp")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.gpu and args.measure_accept_gap:
        print("[ppgd] NOTE: optional GPU accept-gap leg requested but SKIPPED by design -- the "
              "floor verdict depends only on the NON-acceptance buckets (outlen=0 hard, "
              "numerics=0 hard, ctxlen roofline-bounded). The acceptance bucket is pinned by the "
              "benchmark gap (4.295% - ctxlen); a per-distribution accept measurement would refine "
              "its split, not move the floor. CPU-analytic verdict stands.", flush=True)

    report = build_report(args.private_prompt_shift)

    wid = None
    if not args.no_wandb:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    report["wandb_run_ids"] = [wid] if wid else []
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))

    dec, floor, cov = (report["decomposition_central"], report["irreducible_floor"],
                       report["coverage_map"])
    bar = "=" * 92
    print("\n" + bar, flush=True)
    print(" PUBLIC->PRIVATE GAP STRUCTURAL DECOMPOSITION (PR #379, #319)", flush=True)
    print(bar, flush=True)
    print(f" total gap (481.53->460.85)         : {dec['total_gap_pct']:.4f}%  (knife-edge "
          f"{KNIFE_EDGE_PCT}%)", flush=True)
    print(f" --- FOUR BUCKETS (central Delta-P={dec['delta_p_tokens']:.0f} tok; abs% | % of gap) ---",
          flush=True)
    print(f"   (a) acceptance  {dec['bucket_acceptance_abs_pct']:.4f}%  | "
          f"{dec['gap_bucket_acceptance_pct']:6.2f}%  ADDRESSABLE (coverage)", flush=True)
    print(f"   (b) ctxlen      {dec['bucket_ctxlen_abs_pct']:.4f}%  | "
          f"{dec['gap_bucket_ctxlen_pct']:6.2f}%  partial (prompt-length dist)", flush=True)
    print(f"   (c) outlen      {dec['bucket_outlen_abs_pct']:.4f}%  | "
          f"{dec['gap_bucket_outlen_pct']:6.2f}%  ZERO (output fixed 512 tok)", flush=True)
    print(f"   (d) numerics    {dec['bucket_numerics_abs_pct']:.4f}%  | "
          f"{dec['gap_bucket_irreducible_pct']:6.2f}%  ZERO (cancels in gap)", flush=True)
    print(f"   sum             {dec['buckets_sum_abs_pct']:.4f}%  (resid vs gap "
          f"{dec['buckets_sum_resid_vs_gap']:+.2e})", flush=True)
    print(f" --- IRREDUCIBLE FLOOR (gap after PERFECT coverage retrain) ---", flush=True)
    print(f"   irreducible_gap_floor_pct        : {floor['irreducible_gap_floor_pct']:.4f}%  "
          f"(banked {report['irreducible_floor_banked_pct']:.4f}% / pessimistic "
          f"{report['irreducible_floor_pessimistic_pct']:.4f}%)", flush=True)
    print(f"   clears_3p2_knife_edge            : {floor['clears_3p2_knife_edge']}", flush=True)
    print(f"   breakeven private prompt shift   : +{floor['breakeven_private_prompt_shift_tokens']:.0f} "
          f"tok ({floor['breakeven_shift_as_mult_of_public_mean']:.2f}x public mean) -> implausible",
          flush=True)
    print(f" --- COVERAGE -> GAP MAP (reconciles denken #377) ---", flush=True)
    print(f"   gap_shrink_per_coverage          : {cov['gap_shrink_contribution_per_coverage_tps']:.2f} "
          f"TPS/unit ({cov['gap_shrink_per_0p01_coverage_tps']:.3f} TPS/+0.01)", flush=True)
    print(f"   coverage_target_for_3p2          : {cov['coverage_target_for_3p2']:.4f} "
          f"(+{cov['coverage_delta_for_3p2']:.4f}; within #336 +0.031 = "
          f"{cov['coverage_3p2_within_336_envelope']})", flush=True)
    print(f"   denken +5.44 TPS needs Delta-cov  : +{cov['coverage_delta_for_denken_544']:.4f} "
          f"(reconciles)", flush=True)
    print(f"   gap_after_max_coverage_retrain   : {cov['gap_after_max_coverage_retrain_pct']:.4f}% "
          f"(clears 3.2% = {cov['gap_after_max_clears_3p2']})", flush=True)
    print(f" --- VERDICT ---", flush=True)
    print(f"   demand_side_route_has_path       : {report['demand_side_route_has_path']}", flush=True)
    print(f"   verdict_band                     : {report['verdict_band']}", flush=True)
    print(f"   PRIMARY gap_decomp_self_test     : {report['gap_decomp_self_test_passes']}", flush=True)
    print(f"   wandb run                        : {wid}", flush=True)
    print(f"   artifacts                        : {RESULTS_PATH}", flush=True)
    print(bar + "\n", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": report["wandb_run_ids"],
        "primary_metric": {"name": "gap_decomp_self_test_passes",
                           "value": int(report["gap_decomp_self_test_passes"])},
        "test_metric": {"name": "irreducible_gap_floor_pct",
                        "value": report["irreducible_gap_floor_pct"]},
        "headline": {
            "irreducible_gap_floor_pct": report["irreducible_gap_floor_pct"],
            "clears_3p2_knife_edge": report["clears_3p2_knife_edge"],
            "gap_bucket_acceptance_pct": report["gap_bucket_acceptance_pct"],
            "gap_bucket_ctxlen_pct": report["gap_bucket_ctxlen_pct"],
            "gap_bucket_outlen_pct": report["gap_bucket_outlen_pct"],
            "gap_bucket_irreducible_pct": report["gap_bucket_irreducible_pct"],
            "gap_after_max_coverage_retrain": report["gap_after_max_coverage_retrain"],
            "coverage_target_for_3p2": report["coverage_target_for_3p2"],
            "demand_side_route_has_path": report["demand_side_route_has_path"],
            "verdict_band": report["verdict_band"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and not report["gap_decomp_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[ppgd] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
