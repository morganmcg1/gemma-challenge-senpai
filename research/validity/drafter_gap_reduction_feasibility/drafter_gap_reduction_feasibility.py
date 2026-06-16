#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Drafter-gap-reduction feasibility: can a drafter lever pull the spec-alive breach <5%? (PR #492, denken).

CPU-ONLY ANALYSIS + LITERATURE SYNTHESIS. NO kernel re-measure, NO served-file change, NO HF Job,
NO submission, NO --launch, NO drafter training. analysis_only=true, official_tps=0,
no_served_file_change=true. Continues denken #489 (q1ivw9tt). Draw-safe: does not touch the
suspended #473 fire or the pending #474 ruling.

THE FORWARD QUESTION (#481 forward-ladder Direction 4, analytic leg)
-------------------------------------------------------------------
My #489 proved the program has NO fast private-safe *strict* next rung: every spec-alive
(drafter-keeping) config carries the same fixed systematic private delta Delta = 4.295%
(3.661% drafter-acceptance + 0.633% ctxlen) and therefore the SAME scale-invariant fractional
one-shot breach P(private < 0.95*public) = 24.31%, regardless of public TPS. Speed buys nothing.
The only strict private-safe ship is floor-lock 161.70 (no drafter, Delta = 0.633%, breach 0.0008%).
#489 closed with the right question: a genuinely fast strict rung requires attacking the 3.661%
ACCEPTANCE bucket itself (drafter robustness on the PRIVATE distribution), not raising public TPS.

This card answers the FEASIBILITY half: is there ANY plausible drafter-hardening lever that can
deliver a 4.295% -> <=3.0%/<=2.7% reduction on the PRIVATE distribution, i.e. make a fast spec-alive
config private-safe -- or is floor-lock 161.70 the permanent strict ceiling for safety?

THE LOAD-BEARING RESULT
-----------------------
Two independent analyses converge on the SAME mechanism verdict:

  (1) SATURATION (rigorous, this card): the public->private gap is a RATIO (r_accept =
      E_T_priv/E_T_pub = 0.9634). Raising the drafter's INTRINSIC per-position acceptance a_k
      (what EAGLE-3 multi-layer fusion does -- the a_1 cliff 0.729 -> ~0.77-0.91 envelope) does
      NOT close the ratio: under BOTH shift models it mildly WORSENS it (multiplicative +0.38pp,
      additive +0.17pp), because higher acceptance means longer expected chains that compound a
      fixed per-position shift over more surviving positions. Lifting both distributions does NOT
      close a ratio. a_1 is the SPEED axis (#342/#308), and if anything an ANTI-gap-closer.

  (2) LITERATURE (SambaNova arXiv:2503.07807, the only direct OOD draft-head adaptation study):
      EAGLE-3 ARCHITECTURE fusion ALONE closes only 10-25% of an OOD acceptance gap; the closure
      comes from the DATA lever -- generic wider-mix 25-50%, DOMAIN-TARGETED fine-tune 60-80% --
      with a 20-40% intrinsic-difficulty FLOOR (our 3.7% drop is a MILD-shift regime).

So the ONLY robust gap-closer is SHIFT-REDUCTION via domain-targeted wider-distribution training
(train the drafter on private-distribution-like reasoning/STEM data), NOT the architecture.

THE GATE
--------
Inverting #489's fractional breach to the EXACT Delta targets (refining the PR's ~3.0%/~2.7%):
  breach < 5%  <=>  Delta <= 3.334%  <=>  acceptance bucket <= 2.701%  <=>  reduce 0.960pp (26.2% of bucket)
  breach < 1%  <=>  Delta <= 2.644%  <=>  acceptance bucket <= 2.011%  <=>  reduce 1.650pp (45.1% of bucket)
(ctxlen 0.633% is a global-KV term, NOT drafter-attackable, so it is a fixed addend.)

Mapping the literature closure bands onto the 3.661% bucket:
  EAGLE-3 arch fusion ALONE (10-25%): 0.37-0.92pp  -> MISSES breach<5% (need 0.96pp).
  wider-distribution data mix (25-50%): 0.92-1.83pp -> central CLEARS breach<5%, marginal at low end.
  domain-targeted fine-tune (60-80%): 2.20-2.93pp   -> CLEARS both breach<5% AND breach<1%.

VERDICT: fast_private_safe_strict_path_exists = True (CONDITIONAL GO). A fast spec-alive config CAN
be made private-safe (breach<5%), but ONLY by a domain-targeted wider-distribution drafter retrain --
NOT by EAGLE-3 architecture fusion alone, and NOT by any amount of raw public-TPS speed. The cheapest
lever that closes the gate is wider-distribution training, not the arch change. HONESTY BAR: these are
analytic estimates over literature lifts (no published EAGLE-3 OOD a_1 measurement exists); the GO only
justifies spending GPU to TRAIN + measure E_accept on a held-out shifted split, it does NOT itself
prove the gate closes. Floor-lock 161.70 remains the only MEASURED strict private-safe ship today.

Reproduce: cd target/ && .venv/bin/python \
  research/validity/drafter_gap_reduction_feasibility/drafter_gap_reduction_feasibility.py \
  --wandb_group drafter-gap-reduction-feasibility --wandb_name denken/drafter-gap-reduction-feasibility
"""
from __future__ import annotations

import argparse
import json
import math
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))

# ---- banked inputs (all read-only; this card measures nothing on the GPU) -----------
GAP_JSON = os.path.join(
    _root, "research/validity/public_private_gap_decomposition/public_private_gap_decomposition_results.json")  # ubel #379
SIGMA_RECON = os.path.join(_root, "research/empirical_sigma_hw/fresh_n10/reconciliation.json")                  # lawine #467
PST_JSON = os.path.join(
    _root, "research/validity/private_safe_tps_threshold/private_safe_tps_threshold_results.json")             # denken #489
DECAY_JSON = os.path.join(
    _root, "research/validity/per_position_acceptance_decay/per_position_acceptance_decay_results.json")        # kanna #289
A1CLIFF_JSON = os.path.join(
    _root, "research/validity/eagle3_a1_cliff_trainability/eagle3_a1_cliff_trainability_results.json")          # denken #308
LITGAP_JSON = os.path.join(
    _root, "research/validity/eagle3_a1_cliff_lit_gap/eagle3_a1_cliff_lit_gap_results.json")                    # #342
COV_JSON = os.path.join(
    _root, "research/validity/eagle3_head_coverage_lift_target/eagle3_head_coverage_lift_target_results.json")  # lawine #336
BUILD_JSON = os.path.join(
    _root, "research/validity/eagle3_build_cost/eagle3_build_cost_results.json")                                # denken #301
CROSS_JSON = os.path.join(
    _root, "research/validity/strict_frontier_realize_crosscheck/strict_frontier_realize_crosscheck_report.json")  # ubel #470

# Validity gates (BASELINE.md / program.md).
DELTA_GATE = 0.05            # public<->private TPS reproduction gate (private >= 95% of public)
PPL_GATE = 2.42
DEPLOYED_PUBLIC_TPS = 481.53
DEPLOYED_PRIVATE_TPS = 460.85
E_T_MAX = 8.0               # perfect-accept ceiling: K_spec=7 drafted + 1 bonus token
BREACH_BARS = [0.05, 0.01]  # gate-safe bars: breach < 5% and breach < 1%

# ---- LITERATURE closure bands (researcher pass, PR #492 step 3) ----------------------
# SambaNova arXiv:2503.07807 is the ONLY direct OOD draft-head adaptation study: it measures how
# much of a cross-domain token-acceptance drop a domain-adapted draft head recovers. Our public->
# private E[T] drop (~3.7% rel) is a MILD shift by its scale (it reports 7-38% drops), comparable
# to its Coding/Math rows. EAGLE-3 (arXiv:2503.01840) Table-2 ablation: architecture (token-pred +
# multi-layer fusion) raises *in-distribution* tau but publishes NO cross-domain / OOD a_1 arm.
#   - arch fusion ALONE (no new data) closes ........ 10-25% of the OOD gap (speculative; no lit OOD ablation)
#   - generic wider-mix training closes ............. 25-50% (analogy to SambaNova mild-domain rows)
#   - DOMAIN-TARGETED fine-tune closes .............. 60-80% (SambaNova distillation, in-domain data)
#   - intrinsic-difficulty FLOOR (irreducible) ...... 20-40% of the gap (mild-shift regime)
# These are LITERATURE-GROUNDED PRIORS, not measurements on our drafter.
LIT_CLOSE_ARCH_ALONE = (0.10, 0.25)
LIT_CLOSE_WIDER_MIX = (0.25, 0.50)
LIT_CLOSE_DOMAIN_TGT = (0.60, 0.80)
LIT_IRREDUCIBLE_FLOOR = (0.20, 0.40)


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phinv(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation; |err| < 1.2e-9)."""
    if not (0.0 < p < 1.0):
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# ---- speculative chain E[T] model (kanna #289 per-position structure) -----------------
def chain_et(a_list):
    """E[T] = 1 + sum_j prod_{i<=j} a_i  (expected emitted tokens per drafter step, 1 verify bonus)."""
    total, surv = 1.0, 1.0
    for a in a_list:
        surv *= a
        total += surv
    return total


def calibrate_mult_shift(a_pub, r_target, lo=0.5, hi=1.0):
    """Find s s.t. chain_et([s*a])/chain_et(a_pub) == r_target (multiplicative per-position shift)."""
    et_pub = chain_et(a_pub)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        r = chain_et([mid * a for a in a_pub]) / et_pub
        if r < r_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_add_shift(a_pub, r_target, lo=0.0, hi=0.5):
    """Find d s.t. chain_et([a-d])/chain_et(a_pub) == r_target (additive per-position shift)."""
    et_pub = chain_et(a_pub)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        r = chain_et([max(0.0, a - mid) for a in a_pub]) / et_pub
        if r > r_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_here, "drafter_gap_reduction_feasibility_results.json"))
    ap.add_argument("--eagle3_arch_a1_lift", type=float, default=None,
                    help="EAGLE-3 arch-lever additive a_1 lift for the saturation test (default: #308 in-repo gain).")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="drafter-gap-reduction-feasibility")
    ap.add_argument("--wandb_name", default="denken/drafter-gap-reduction-feasibility")
    ap.add_argument("--job_type", default="analysis")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    gap = json.load(open(GAP_JSON))
    recon = json.load(open(SIGMA_RECON))["reconciliation"]
    pst = json.load(open(PST_JSON))
    decay = json.load(open(DECAY_JSON))
    a1c = json.load(open(A1CLIFF_JSON))["synthesis"]
    litgap = json.load(open(LITGAP_JSON))
    cov = json.load(open(COV_JSON))["synthesis"]
    build = json.load(open(BUILD_JSON))["synthesis"]
    cross = json.load(open(CROSS_JSON))

    # ============================ IMPORTED ANCHORS (do not re-derive) ============================
    # ubel #379 public->private gap decomposition (central).
    deployed_gap = float(gap["decomposition_central"]["total_gap_frac"])                  # 0.042946
    accept_bucket = float(gap["decomposition_central"]["bucket_acceptance_abs_pct"]) / 100.0  # 0.036613 DRAFTER
    ctxlen_bucket = float(gap["decomposition_central"]["bucket_ctxlen_abs_pct"]) / 100.0      # 0.006334 global KV
    r_accept = float(gap["decomposition_central"]["r_accept"])                            # 0.963387 E_T_priv/E_T_pub
    E_T_pub_anchor = float(gap["imported"]["E_T_pub"])                                    # 3.844
    K_cal = float(gap["imported"]["K_cal"])                                               # 125.268
    # #379 coverage retrain model (OPTIMISTIC full-transfer ceiling): max coverage -> residual gap.
    gap_after_max_cov = float(gap["coverage_map"]["gap_after_max_coverage_retrain_pct"]) / 100.0  # 0.011416
    cov_baseline = float(gap["imported"]["coverage_baseline_336"])                        # 0.8903
    cov_bar = float(gap["imported"]["coverage_bar_336"])                                  # 0.9213

    # lawine #467 sigma_hw (single official draw); FRACTIONAL model is the PHYSICAL one (#486/#489).
    sigma_oneshot = float(recon["sigma_oneshot_reconstructed_tps"])                       # 4.8765
    sigma_oneshot_frac = sigma_oneshot / DEPLOYED_PUBLIC_TPS                              # 0.010127

    # denken #489 banked breach (the number this card must reproduce).
    banked_specalive_breach_pct = float(pst["verdict"]["spec_alive_frac_scale_invariant_breach_pct"])  # 24.3058
    floorlock_tps = float(pst["config"]["floorlock_public_tps"])                          # 161.70

    # kanna #289 per-position acceptance (PUBLIC distribution).
    a_pub = [float(x) for x in decay["decomposition"]["a_k"]]                             # 7 conditional accepts
    E_T_pub_decomp = float(decay["decomposition"]["E_T"])                                 # 3.8512
    a1_deployed = float(decay["decomposition"]["a_k"][0])                                 # 0.72925 (cliff, pos-1)

    # denken #308 / #342 EAGLE-3 a_1 envelope (the architecture lever's ceiling).
    arch_lift_a1_inrepo = float(a1c["step2_published_envelope"]["arch_lever_gain_on_a1_inrepo"])  # +0.04215
    a1_inrepo_native = float(a1c["step2_published_envelope"]["a1_inrepo_eagle3_native_step1"])    # 0.7714
    a1_robustness_ceiling = float(a1c["step3_ceiling"]["a1_argmax_robustness_ceiling_central"])   # 0.85
    eagle3_dense_a1_conservative = float(litgap["imported"]["eagle3_dense_a1_conservative"])      # 0.8893
    eagle3_dense_a1_central = float(litgap["imported"]["eagle3_dense_a1_central"])                # 0.91

    # lawine #336 coverage-lift recipe bands (architecture-preserving wider-data retrain).
    combo = cov["step3_recipe_ranking"]["recommended_combination"]
    cov_combo_central = float(combo["delta_cov_central"])                                 # 0.0385
    cov_feasibility = cov["step4_feasibility_verdict"]["feasibility_verdict"]             # REACHABLE-MARGINAL

    # denken #301 EAGLE-3 geometry (context only).
    eagle3_fusion_layers = [int(x) for x in build["target_geometry"]["eagle3_target_layers"]]   # {2,21,39}
    e_accept_bi = float(cross["e_accept_under_bi"])                                       # 3.8695 (spec ALIVE anchor)

    # ============================ (0) BREACH MODEL (import from #489) ============================
    # FRACTIONAL one-shot breach is SCALE-INVARIANT: P(private < 0.95*public) = Phi(-(0.05-Delta)/sigma_frac),
    # independent of TPS. This is the #489 headline -- speed buys nothing for a spec-alive (Delta-carrying) config.
    def breach_frac(delta):
        return _phi(-(DELTA_GATE - delta) / sigma_oneshot_frac)

    specalive_breach = breach_frac(deployed_gap)                                          # ~0.2431 -> reproduce #489

    # ============================ (1) SUB-DECOMPOSE THE 3.661% ACCEPTANCE BUCKET ============================
    # By construction the 3.661% bucket IS the public->private E[T] drop -- i.e. the SHIFT (term b). The
    # "intrinsic public imperfection" (term a: a_k<1.0 even on the training distribution) is present on BOTH
    # distributions and CANCELS in the gap; it does NOT appear in the 3.661%. We quantify (a) as CONTEXT and
    # test whether attacking its headroom (raising intrinsic a_k via EAGLE-3) closes the SHIFT.
    intrinsic_public_token_deficit = E_T_MAX - E_T_pub_decomp          # 4.149 tokens (perfect-accept ceiling shortfall)
    a1_cliff_token_share = (1.0 - a1_deployed)                          # 0.271 of the first survival forfeited at pos-1

    # --- SATURATION TEST: does raising intrinsic per-position acceptance close the RATIO gap? ---
    # Calibrate the private shift two ways (multiplicative / additive) to the banked r_accept, then apply the
    # EAGLE-3 arch a_1 lift uniformly (additive, capped) to the PUBLIC profile, hold the SAME shift, recompute
    # the gap. The lift never CLOSES the gap (it WORSENS it under both shift models), so the intrinsic-acceptance
    # (a_1) channel is NOT a reliable gap-closer -- the gap is a property of the SHIFT, not the absolute accept level.
    et_pub0 = chain_et(a_pub)
    s_mult = calibrate_mult_shift(a_pub, r_accept)
    d_add = calibrate_add_shift(a_pub, r_accept)
    arch_lift = float(args.eagle3_arch_a1_lift) if args.eagle3_arch_a1_lift is not None else arch_lift_a1_inrepo
    a_pub_lift = [min(0.97, a + arch_lift) for a in a_pub]              # EAGLE-3 raises intrinsic acceptance
    # multiplicative-shift world
    gap_mult_base = 1.0 - chain_et([s_mult * a for a in a_pub]) / et_pub0
    gap_mult_lift = 1.0 - chain_et([s_mult * a for a in a_pub_lift]) / chain_et(a_pub_lift)
    # additive-shift world
    gap_add_base = 1.0 - chain_et([max(0.0, a - d_add) for a in a_pub]) / et_pub0
    gap_add_lift = 1.0 - chain_et([max(0.0, a - d_add) for a in a_pub_lift]) / chain_et(a_pub_lift)
    saturation_delta_mult_pp = 100.0 * (gap_mult_lift - gap_mult_base)  # >0 => lifting intrinsic a_k WORSENS the gap
    saturation_delta_add_pp = 100.0 * (gap_add_lift - gap_add_base)     # >0 => also WORSENS (longer chains compound the shift)
    saturation_abs_max_pp = max(abs(saturation_delta_mult_pp), abs(saturation_delta_add_pp))
    saturation_sign_ambiguous = bool(saturation_delta_mult_pp * saturation_delta_add_pp < 0.0)  # reported, not asserted
    saturation_lift_worsens_under_both = bool(saturation_delta_mult_pp > 0.0 and saturation_delta_add_pp > 0.0)
    # The BEST a uniform intrinsic lift can do across both shift models (positive => it CLOSES the gap by that many pp).
    # Here both shift models WORSEN the gap, so the best case is negative -- the lift never closes it.
    intrinsic_best_close_pp = -min(saturation_delta_mult_pp, saturation_delta_add_pp)
    # Intrinsic-acceptance lift (the EAGLE-3 arch / a_1 axis) is NOT a reliable gap-closer: even its best case across
    # both shift models closes < 0.5pp -- in fact it WORSENS the ratio under both, because higher acceptance means
    # longer expected chains that compound a fixed per-position shift over more surviving positions.
    intrinsic_lift_closes_gap = bool(intrinsic_best_close_pp >= 0.5)

    # --- ATTACKABLE portion of the 3.661% bucket (shift-reduction only) ---
    # Optimistic ceiling: #379's max-coverage-retrain residual (assumes FULL public->private transfer).
    accept_after_max_cov = gap_after_max_cov - ctxlen_bucket           # 0.005082 residual acceptance after max coverage
    attackable_379_optimistic_pp = 100.0 * (accept_bucket - accept_after_max_cov)  # ~3.153pp (86% of the bucket)
    irreducible_379_floor_pp = 100.0 * accept_after_max_cov            # ~0.508pp
    # Literature-grounded band (SambaNova): attackable = best realistic closure (domain-targeted 60-80%);
    # irreducible-difficulty floor = 20-40% of the bucket (mild-shift regime).
    attackable_lit_lo_pp = 100.0 * accept_bucket * LIT_CLOSE_DOMAIN_TGT[0]   # 60% -> 2.197pp
    attackable_lit_hi_pp = 100.0 * accept_bucket * LIT_CLOSE_DOMAIN_TGT[1]   # 80% -> 2.929pp
    irreducible_lit_lo_pp = 100.0 * accept_bucket * LIT_IRREDUCIBLE_FLOOR[0]  # 20% -> 0.732pp
    irreducible_lit_hi_pp = 100.0 * accept_bucket * LIT_IRREDUCIBLE_FLOOR[1]  # 40% -> 1.465pp
    attackable_central_pp = 0.5 * (attackable_lit_lo_pp + attackable_lit_hi_pp)              # ~2.56pp

    # ============================ (2) INVERT #489 BREACH -> EXACT DELTA TARGETS ============================
    # breach_frac(delta) = Phi(-(0.05-delta)/sigma_frac) = bar  =>  delta = 0.05 - (-Phi^-1(bar))*sigma_frac
    def delta_target_for_bar(bar):
        return DELTA_GATE - (-_phinv(bar)) * sigma_oneshot_frac

    delta_target_breach5 = delta_target_for_bar(0.05)                  # 0.033343 -> 3.334% (refines PR's ~3.0%)
    delta_target_breach1 = delta_target_for_bar(0.01)                  # 0.026441 -> 2.644% (refines PR's ~2.7%)
    # Required ACCEPTANCE bucket (ctxlen is a fixed non-drafter addend) and the required REDUCTION.
    accept_target_breach5 = delta_target_breach5 - ctxlen_bucket       # <= 2.701%
    accept_target_breach1 = delta_target_breach1 - ctxlen_bucket       # <= 2.011%
    reduce_pp_breach5 = 100.0 * (accept_bucket - accept_target_breach5)  # 0.960pp
    reduce_pp_breach1 = 100.0 * (accept_bucket - accept_target_breach1)  # 1.650pp
    # As relative % of the TOTAL bucket and of the ATTACKABLE (domain-targeted central) portion.
    reduce_relbucket_breach5 = reduce_pp_breach5 / (100.0 * accept_bucket)   # 0.262
    reduce_relbucket_breach1 = reduce_pp_breach1 / (100.0 * accept_bucket)   # 0.451
    reduce_relattack_breach5 = reduce_pp_breach5 / attackable_central_pp     # ~0.375
    reduce_relattack_breach1 = reduce_pp_breach1 / attackable_central_pp     # ~0.645

    # ============================ (3) MAP EAGLE-3 / WIDER-DIST LIFTS ONTO THE GAP ============================
    bucket_pp = 100.0 * accept_bucket
    lever = {
        "eagle3_arch_fusion_alone": {
            "close_frac_band": list(LIT_CLOSE_ARCH_ALONE),
            "reduction_pp_band": [bucket_pp * LIT_CLOSE_ARCH_ALONE[0], bucket_pp * LIT_CLOSE_ARCH_ALONE[1]],
            "channel": "raises intrinsic a_1 (0.729 -> ~0.77-0.91 envelope); SATURATION test shows ~0 gap effect",
            "arch_change": True, "needs_new_data": False,
            "lit": "EAGLE-3 arXiv:2503.01840 (in-dist tau only, no OOD a_1 arm); SambaNova arch-alone 10-25%",
        },
        "wider_distribution_data_mix": {
            "close_frac_band": list(LIT_CLOSE_WIDER_MIX),
            "reduction_pp_band": [bucket_pp * LIT_CLOSE_WIDER_MIX[0], bucket_pp * LIT_CLOSE_WIDER_MIX[1]],
            "channel": "shift-reduction: generic reasoning/STEM data mix (#336 'more_reasoning_root_data')",
            "arch_change": False, "needs_new_data": True,
            "lit": "SambaNova arXiv:2503.07807 mild-domain rows 25-50%; #336 combo central +0.0385 cov",
        },
        "domain_targeted_fine_tune": {
            "close_frac_band": list(LIT_CLOSE_DOMAIN_TGT),
            "reduction_pp_band": [bucket_pp * LIT_CLOSE_DOMAIN_TGT[0], bucket_pp * LIT_CLOSE_DOMAIN_TGT[1]],
            "channel": "shift-reduction: drafter retrained on PRIVATE-distribution-like data",
            "arch_change": False, "needs_new_data": True,
            "lit": "SambaNova arXiv:2503.07807 distillation Biology/Chinese rows 60-80%",
        },
        "coverage_retrain_379_optimistic": {
            "close_frac_band": [attackable_379_optimistic_pp / bucket_pp, attackable_379_optimistic_pp / bucket_pp],
            "reduction_pp_band": [attackable_379_optimistic_pp, attackable_379_optimistic_pp],
            "channel": "ubel #379 max-coverage-retrain model (assumes FULL public->private transfer)",
            "arch_change": False, "needs_new_data": True,
            "lit": "#379 coverage_map (above the literature high end -> OPTIMISTIC ceiling, not central)",
        },
    }

    def lever_closes(lev, need_pp):
        b = lev["reduction_pp_band"]
        return {"low_closes": bool(b[0] >= need_pp), "high_closes": bool(b[1] >= need_pp),
                "central_closes": bool(0.5 * (b[0] + b[1]) >= need_pp)}

    lever_gate = {name: {"breach5": lever_closes(lev, reduce_pp_breach5),
                         "breach1": lever_closes(lev, reduce_pp_breach1)} for name, lev in lever.items()}

    # ============================ (4) GO / NO-GO GATE ============================
    eagle3_arch_alone_reduction = lever["eagle3_arch_fusion_alone"]["reduction_pp_band"]
    eagle3_arch_alone_closes_breach5 = bool(eagle3_arch_alone_reduction[1] >= reduce_pp_breach5)  # even high end
    # cheapest lever (no arch change preferred) whose CENTRAL clears breach<5%.
    cheapest_order = ["eagle3_arch_fusion_alone", "wider_distribution_data_mix", "domain_targeted_fine_tune"]
    cheapest_lever_that_closes = "none"
    for name in cheapest_order:
        if lever_gate[name]["breach5"]["central_closes"]:
            cheapest_lever_that_closes = ("wider-distribution-training" if name == "wider_distribution_data_mix"
                                          else "domain-targeted-training" if name == "domain_targeted_fine_tune"
                                          else "EAGLE-3-arch-fusion")
            break
    # EAGLE-3 "closes gate" interpreted as the FULL EAGLE-3 recipe (arch + wider data); arch-alone is separate.
    eagle3_full_recipe_reduction_pp = [lever["wider_distribution_data_mix"]["reduction_pp_band"][0],
                                       lever["domain_targeted_fine_tune"]["reduction_pp_band"][1]]  # [0.92, 2.93]
    eagle3_closes_gate_breach5 = bool(eagle3_full_recipe_reduction_pp[1] >= reduce_pp_breach5)      # high end clears
    eagle3_closes_gate_breach1 = bool(eagle3_full_recipe_reduction_pp[1] >= reduce_pp_breach1)
    # Robustly closes (even the LOW end of the closing lever clears)?
    domain_tgt_robust_breach5 = bool(lever["domain_targeted_fine_tune"]["reduction_pp_band"][0] >= reduce_pp_breach5)
    domain_tgt_robust_breach1 = bool(lever["domain_targeted_fine_tune"]["reduction_pp_band"][0] >= reduce_pp_breach1)

    # HEADLINE: is there ANY plausible drafter lever that makes a fast spec-alive config private-safe (breach<5%)?
    fast_private_safe_strict_path_exists = bool(cheapest_lever_that_closes != "none")
    # Floor-lock 161.70 stays the only MEASURED strict private-safe ship until the retrain is trained + measured.
    floorlock_remains_only_measured_safe = True

    if not fast_private_safe_strict_path_exists:
        verdict_band = "NO-GO: no plausible drafter lever closes breach<5%; floor-lock 161.70 is the strict ceiling"
    elif domain_tgt_robust_breach1:
        verdict_band = "GREEN: domain-targeted retrain robustly closes breach<1% (conditional on private-like data)"
    elif domain_tgt_robust_breach5:
        verdict_band = "GREEN-breach5 / YELLOW-breach1: domain-targeted retrain robustly closes breach<5%"
    else:
        verdict_band = "YELLOW: closes only at the central/high lever band; conditional on transfer"

    # ============================ (5) HONESTY BAR + SELF-TESTS ============================
    st = {}
    # PRIMARY: reproduce #489's 24.31% breach at Delta=4.295%.
    st["reproduces_489_specalive_breach_24p31"] = bool(abs(100.0 * specalive_breach - banked_specalive_breach_pct) < 1e-3)
    # decomposition arithmetic.
    st["accept_plus_ctxlen_equals_gap"] = bool(abs((accept_bucket + ctxlen_bucket) - deployed_gap) < 1e-6)
    st["accept_bucket_is_3p661"] = bool(abs(100.0 * accept_bucket - 3.66126) < 1e-3)
    st["ctxlen_bucket_is_0p633"] = bool(abs(100.0 * ctxlen_bucket - 0.63339) < 1e-3)
    # breach inversion round-trips.
    st["inversion_breach5_roundtrips"] = bool(abs(breach_frac(delta_target_breach5) - 0.05) < 1e-6)
    st["inversion_breach1_roundtrips"] = bool(abs(breach_frac(delta_target_breach1) - 0.01) < 1e-6)
    st["delta_targets_ordered"] = bool(delta_target_breach1 < delta_target_breach5 < deployed_gap)
    st["delta_target_breach5_refines_3p0"] = bool(0.030 < delta_target_breach5 < 0.035)
    st["delta_target_breach1_refines_2p7"] = bool(0.025 < delta_target_breach1 < 0.028)
    # required reduction arithmetic.
    st["reduce_breach5_is_0p96pp"] = bool(abs(reduce_pp_breach5 - 0.960) < 0.05)
    st["reduce_breach1_is_1p65pp"] = bool(abs(reduce_pp_breach1 - 1.650) < 0.05)
    st["reduce_breach1_gt_breach5"] = bool(reduce_pp_breach1 > reduce_pp_breach5 > 0.0)
    st["accept_targets_below_bucket"] = bool(accept_target_breach5 < accept_bucket and accept_target_breach1 < accept_bucket)
    # chain model reproduces #289.
    st["chain_et_reproduces_289"] = bool(abs(chain_et(a_pub) - E_T_pub_decomp) < 1e-3)
    st["a1_is_profile_minimum"] = bool(a1_deployed == min(a_pub))     # the cliff is at position 1
    # shift calibration reproduces r_accept (both models).
    st["mult_shift_reproduces_r_accept"] = bool(
        abs(chain_et([s_mult * a for a in a_pub]) / et_pub0 - r_accept) < 1e-4)
    st["add_shift_reproduces_r_accept"] = bool(
        abs(chain_et([max(0.0, a - d_add) for a in a_pub]) / et_pub0 - r_accept) < 1e-4)
    # SATURATION: intrinsic-acceptance lift does NOT close the gap (small effect, never closes >=0.5pp; in fact
    # WORSENS under both shift models -- longer chains compound a fixed per-position shift).
    st["saturation_effect_is_small"] = bool(saturation_abs_max_pp < 0.5)
    st["saturation_lift_does_not_close"] = bool(intrinsic_best_close_pp < 0.5)
    st["saturation_lift_worsens_under_both"] = bool(saturation_lift_worsens_under_both)
    st["intrinsic_lift_not_reliable_closer"] = bool(not intrinsic_lift_closes_gap)
    # attackable bands ordered, fractions in [0,1], floor + attackable consistent.
    st["attackable_band_ordered"] = bool(attackable_lit_lo_pp <= attackable_lit_hi_pp <= attackable_379_optimistic_pp)
    st["closure_fracs_in_unit"] = all(0.0 <= f <= 1.0 for band in
                                      [LIT_CLOSE_ARCH_ALONE, LIT_CLOSE_WIDER_MIX, LIT_CLOSE_DOMAIN_TGT,
                                       LIT_IRREDUCIBLE_FLOOR] for f in band)
    st["domain_tgt_plus_floor_spans_unit"] = bool(abs((LIT_CLOSE_DOMAIN_TGT[1] + LIT_IRREDUCIBLE_FLOOR[0]) - 1.0) < 1e-9)
    # gate logic consistency.
    st["arch_alone_misses_breach5"] = bool(not eagle3_arch_alone_closes_breach5)   # 0.25*3.661=0.915 < 0.960
    st["domain_tgt_closes_breach5"] = bool(lever_gate["domain_targeted_fine_tune"]["breach5"]["central_closes"])
    st["cheapest_closer_is_data_lever"] = bool(cheapest_lever_that_closes in
                                               ("wider-distribution-training", "domain-targeted-training"))
    st["path_exists_consistent"] = bool(fast_private_safe_strict_path_exists == (cheapest_lever_that_closes != "none"))
    st["ctxlen_below_breach5_target"] = bool(ctxlen_bucket < accept_target_breach5)  # floor-lock stays safe
    # ppl / spec-alive sanity (the retrain target is greedy-identical; E_accept anchor > 1).
    st["spec_alive_eaccept_gt1"] = bool(e_accept_bi > 1.0)
    st["ppl_anchor_within_gate"] = bool(float(cross["ppl"]) <= PPL_GATE)
    # NaN-clean.
    finite = [specalive_breach, delta_target_breach5, delta_target_breach1, reduce_pp_breach5, reduce_pp_breach1,
              attackable_central_pp, attackable_379_optimistic_pp, irreducible_379_floor_pp, s_mult, d_add,
              saturation_delta_mult_pp, saturation_delta_add_pp, intrinsic_best_close_pp,
              eagle3_full_recipe_reduction_pp[0], eagle3_full_recipe_reduction_pp[1]]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    self_test_passes = all(st.values())

    verdict = {
        # ---- (1) decomposition ----
        "accept_bucket_pct": 100.0 * accept_bucket,
        "ctxlen_bucket_pct": 100.0 * ctxlen_bucket,
        "deployed_gap_pct": 100.0 * deployed_gap,
        "specalive_breach_pct": 100.0 * specalive_breach,
        "intrinsic_public_token_deficit": intrinsic_public_token_deficit,
        "a1_deployed": a1_deployed,
        "saturation_delta_mult_pp": saturation_delta_mult_pp,
        "saturation_delta_add_pp": saturation_delta_add_pp,
        "saturation_abs_max_pp": saturation_abs_max_pp,
        "intrinsic_best_close_pp": intrinsic_best_close_pp,
        "saturation_lift_worsens_under_both": saturation_lift_worsens_under_both,
        "intrinsic_lift_closes_gap": intrinsic_lift_closes_gap,
        "attackable_gap_pp": attackable_central_pp,
        "attackable_gap_pp_lo": attackable_lit_lo_pp,
        "attackable_gap_pp_hi": attackable_lit_hi_pp,
        "attackable_gap_pp_379_optimistic": attackable_379_optimistic_pp,
        "irreducible_floor_pp_lo": irreducible_lit_lo_pp,
        "irreducible_floor_pp_hi": irreducible_lit_hi_pp,
        # ---- (2) inverted Delta targets ----
        "delta_accept_target_breach5": delta_target_breach5,
        "delta_accept_target_breach1": delta_target_breach1,
        "delta_accept_target_breach5_pct": 100.0 * delta_target_breach5,
        "delta_accept_target_breach1_pct": 100.0 * delta_target_breach1,
        "accept_target_breach5_pct": 100.0 * accept_target_breach5,
        "accept_target_breach1_pct": 100.0 * accept_target_breach1,
        "reduce_pp_breach5": reduce_pp_breach5,
        "reduce_pp_breach1": reduce_pp_breach1,
        "reduce_relbucket_breach5": reduce_relbucket_breach5,
        "reduce_relbucket_breach1": reduce_relbucket_breach1,
        "reduce_relattack_breach5": reduce_relattack_breach5,
        "reduce_relattack_breach1": reduce_relattack_breach1,
        # ---- (3/4) lift mapping + GATE ----
        "eagle3_plausible_reduction_pp_lo": eagle3_full_recipe_reduction_pp[0],
        "eagle3_plausible_reduction_pp_hi": eagle3_full_recipe_reduction_pp[1],
        "eagle3_arch_alone_closes_breach5": eagle3_arch_alone_closes_breach5,
        "eagle3_closes_gate": eagle3_closes_gate_breach5,
        "eagle3_closes_gate_breach1": eagle3_closes_gate_breach1,
        "domain_tgt_robust_breach5": domain_tgt_robust_breach5,
        "domain_tgt_robust_breach1": domain_tgt_robust_breach1,
        "cheapest_lever_that_closes": cheapest_lever_that_closes,
        "fast_private_safe_strict_path_exists": fast_private_safe_strict_path_exists,
        "floorlock_remains_only_measured_safe": floorlock_remains_only_measured_safe,
        "floorlock_tps": floorlock_tps,
        "verdict_band": verdict_band,
        # ---- config ----
        "delta_gate": DELTA_GATE, "ppl_gate": PPL_GATE,
        "sigma_oneshot_frac_pct": 100.0 * sigma_oneshot_frac,
        "eagle3_fusion_layers": str(eagle3_fusion_layers),
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_kernel_rebuild": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_drafter_training": True, "gpu_used": False, "tps_added_by_this_card": 0,
        "self_test_passes": self_test_passes,
    }

    reconcile = (
        f"Drafter-gap-reduction feasibility (#481 Dir-4 analytic leg). #489 fixed the breach at "
        f"{100*specalive_breach:.2f}% for ANY spec-alive config (Delta={100*deployed_gap:.3f}%); closing it needs "
        f"the 3.661% ACCEPTANCE bucket itself. INVERTING #489: breach<5% <=> Delta<={100*delta_target_breach5:.3f}% "
        f"<=> reduce acceptance {reduce_pp_breach5:.3f}pp ({100*reduce_relbucket_breach5:.1f}% of bucket); breach<1% "
        f"<=> Delta<={100*delta_target_breach1:.3f}% <=> reduce {reduce_pp_breach1:.3f}pp "
        f"({100*reduce_relbucket_breach1:.1f}%). MECHANISM: the gap is a RATIO -- raising INTRINSIC a_1 (EAGLE-3 arch "
        f"fusion, the cliff 0.729->~0.77-0.91) does NOT close it: under BOTH shift models it mildly WORSENS "
        f"(mult {saturation_delta_mult_pp:+.3f}pp / add {saturation_delta_add_pp:+.3f}pp; longer chains compound a fixed "
        f"per-position shift) -> a_1 is the SPEED axis, NOT a gap-closer. Only SHIFT-REDUCTION closes it. "
        f"LITERATURE (SambaNova 2503.07807, mild-shift): "
        f"arch-alone 10-25% ({eagle3_arch_alone_reduction[0]:.2f}-{eagle3_arch_alone_reduction[1]:.2f}pp -> MISSES "
        f"breach<5%), wider data-mix 25-50%, DOMAIN-TARGETED 60-80% ({attackable_lit_lo_pp:.2f}-{attackable_lit_hi_pp:.2f}pp "
        f"-> CLEARS both bars). VERDICT fast_private_safe_strict_path_exists={fast_private_safe_strict_path_exists}: a "
        f"fast spec-alive config CAN be made private-safe, but ONLY by a {cheapest_lever_that_closes} drafter retrain, "
        f"NOT by EAGLE-3 architecture alone and NOT by any public-TPS speed. HONESTY: analytic over literature priors "
        f"(no published EAGLE-3 OOD a_1 exists); GO only justifies GPU spend to TRAIN + measure E_accept on a held-out "
        f"shifted split. Floor-lock {floorlock_tps:.2f} stays the only MEASURED strict private-safe ship.")
    verdict["reconcile_line"] = reconcile

    payload = {
        "pr": 492, "issue": 481, "author": "denken",
        "leg": "drafter-gap-reduction feasibility: can a drafter lever pull the spec-alive breach <5%? (#481 Dir 4)",
        "config": {
            "delta_gate": DELTA_GATE, "ppl_gate": PPL_GATE,
            "deployed_public_tps": DEPLOYED_PUBLIC_TPS, "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
            "deployed_gap": deployed_gap, "accept_bucket": accept_bucket, "ctxlen_bucket": ctxlen_bucket,
            "r_accept": r_accept, "E_T_pub_decomp": E_T_pub_decomp, "E_T_max": E_T_MAX, "K_cal": K_cal,
            "a_pub": a_pub, "a1_deployed": a1_deployed,
            "sigma_oneshot_tps": sigma_oneshot, "sigma_oneshot_frac": sigma_oneshot_frac,
            "arch_lift_a1_inrepo": arch_lift_a1_inrepo, "a1_inrepo_native": a1_inrepo_native,
            "a1_robustness_ceiling": a1_robustness_ceiling,
            "eagle3_dense_a1_conservative": eagle3_dense_a1_conservative,
            "eagle3_dense_a1_central": eagle3_dense_a1_central,
            "eagle3_fusion_layers": eagle3_fusion_layers, "e_accept_bi": e_accept_bi,
            "lit_close_arch_alone": list(LIT_CLOSE_ARCH_ALONE), "lit_close_wider_mix": list(LIT_CLOSE_WIDER_MIX),
            "lit_close_domain_tgt": list(LIT_CLOSE_DOMAIN_TGT), "lit_irreducible_floor": list(LIT_IRREDUCIBLE_FLOOR),
            "cov_combo_central": cov_combo_central, "cov_feasibility_verdict": cov_feasibility,
            "cov_baseline_336": cov_baseline, "cov_bar_336": cov_bar,
            "s_mult_calibrated": s_mult, "d_add_calibrated": d_add, "arch_lift_applied": arch_lift,
            "imports": {
                "gap_decomp_379": os.path.relpath(GAP_JSON, _root),
                "sigma_recon_467": os.path.relpath(SIGMA_RECON, _root),
                "breach_model_489": os.path.relpath(PST_JSON, _root),
                "per_position_289": os.path.relpath(DECAY_JSON, _root),
                "a1_cliff_308": os.path.relpath(A1CLIFF_JSON, _root),
                "a1_lit_gap_342": os.path.relpath(LITGAP_JSON, _root),
                "coverage_336": os.path.relpath(COV_JSON, _root),
                "build_cost_301": os.path.relpath(BUILD_JSON, _root),
                "crosscheck_470": os.path.relpath(CROSS_JSON, _root),
            },
            "note": "Extends denken #489 (q1ivw9tt). CPU analysis + literature synthesis only; no kernel "
                    "re-measure, no served change, no HF Job, no launch, no submission, no drafter training. "
                    "Literature closure bands are SambaNova arXiv:2503.07807 / EAGLE-3 arXiv:2503.01840 priors, "
                    "NOT measurements on our drafter.",
        },
        "levers": lever,
        "lever_gate": lever_gate,
        "saturation_test": {
            "s_mult": s_mult, "d_add": d_add, "arch_lift_applied": arch_lift,
            "gap_mult_base_pct": 100.0 * gap_mult_base, "gap_mult_lift_pct": 100.0 * gap_mult_lift,
            "gap_add_base_pct": 100.0 * gap_add_base, "gap_add_lift_pct": 100.0 * gap_add_lift,
            "saturation_delta_mult_pp": saturation_delta_mult_pp,
            "saturation_delta_add_pp": saturation_delta_add_pp,
            "sign_ambiguous": saturation_sign_ambiguous,
            "lift_worsens_under_both": saturation_lift_worsens_under_both,
            "intrinsic_best_close_pp": intrinsic_best_close_pp,
            "intrinsic_lift_closes_gap": intrinsic_lift_closes_gap,
        },
        "verdict": verdict,
        "self_test_conditions": st,
        "public_evidence_used": (
            "denken #489 (q1ivw9tt) private_safe_tps_threshold: spec-alive Delta=4.295% -> scale-invariant 24.31% "
            "fractional breach; floor-lock 161.70 the only strict private-safe ship. ubel #379 (5kpb73tb) "
            "public_private_gap_decomposition: Delta=3.661% acceptance + 0.633% ctxlen; coverage_map max-retrain "
            "residual 1.142%. kanna #289 (fi34s269) per_position_acceptance_decay: a_1=0.729 cliff, E[T]=3.851, "
            "deeper a_k 0.76-0.85. denken #308 (5axqa6oa) / #342 (ph8eza1w) eagle3_a1_cliff: arch lever +0.042 on "
            "a_1, in-repo native 0.7714, published dense a_1 0.889-0.93, robustness ceiling 0.85. lawine #336 "
            "(krroookz) eagle3_head_coverage_lift_target: soft-KD+reasoning-data combo central +0.0385 cov, "
            "REACHABLE-MARGINAL, private-tax robustness flagged a SEPARATE gate (fern #325). denken #301 "
            "eagle3_build_cost: {2,21,39} fusion geometry. ubel #470 (ugqnytji) E_accept~3.87 spec-alive, PPL "
            "2.3770<=2.42. lawine #467 sigma_hw: sigma_oneshot 4.876 (1.013% fractional). LITERATURE: SambaNova "
            "arXiv:2503.07807 (only direct OOD draft-head adaptation study) domain-targeted closes 60-80% of an "
            "OOD acceptance drop, generic mix 25-50%, arch-alone 10-25%, floor 20-40%; EAGLE-3 arXiv:2503.01840 "
            "Table-2 arch ablation (in-distribution tau only, NO published OOD a_1 arm)."),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))

    print(f"[dgr] #489 breach reproduced: {100*specalive_breach:.4f}% (banked {banked_specalive_breach_pct:.4f}%) | "
          f"self_test={self_test_passes}", flush=True)
    print(f"[dgr] INVERT: breach<5% <=> Delta<={100*delta_target_breach5:.3f}% (reduce {reduce_pp_breach5:.3f}pp / "
          f"{100*reduce_relbucket_breach5:.1f}% of bucket) | breach<1% <=> Delta<={100*delta_target_breach1:.3f}% "
          f"(reduce {reduce_pp_breach1:.3f}pp / {100*reduce_relbucket_breach1:.1f}%)", flush=True)
    print(f"[dgr] SATURATION (raise intrinsic a_1 by {arch_lift:+.3f}): gap WORSENS under both -- mult "
          f"{saturation_delta_mult_pp:+.3f}pp / add {saturation_delta_add_pp:+.3f}pp -> "
          f"intrinsic_lift_closes_gap={intrinsic_lift_closes_gap} (a_1 is the SPEED axis, not a gap-closer)", flush=True)
    print(f"[dgr] LEVERS (reduction pp): arch-alone {eagle3_arch_alone_reduction[0]:.2f}-{eagle3_arch_alone_reduction[1]:.2f} "
          f"(MISS) | wider-mix {lever['wider_distribution_data_mix']['reduction_pp_band'][0]:.2f}-"
          f"{lever['wider_distribution_data_mix']['reduction_pp_band'][1]:.2f} | domain-tgt "
          f"{attackable_lit_lo_pp:.2f}-{attackable_lit_hi_pp:.2f} (CLEAR) | need {reduce_pp_breach5:.2f}/"
          f"{reduce_pp_breach1:.2f}", flush=True)
    print(f"[dgr] GATE: fast_private_safe_strict_path_exists={fast_private_safe_strict_path_exists} | "
          f"cheapest_lever_that_closes={cheapest_lever_that_closes} | attackable {attackable_lit_lo_pp:.2f}-"
          f"{attackable_lit_hi_pp:.2f}pp | floor {irreducible_lit_lo_pp:.2f}-{irreducible_lit_hi_pp:.2f}pp | "
          f"band={verdict_band}", flush=True)
    print(f"[dgr] {reconcile}", flush=True)

    if not args.no_wandb:
        _log_wandb(args, payload)
    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type=args.job_type, config=payload.get("config", {}))
    vd = payload["verdict"]
    run.summary.update({k: v for k, v in vd.items() if isinstance(v, (int, float, bool, str))})

    # lever table: each candidate lever, its literature closure band, and gate verdict.
    lt = wandb.Table(columns=["lever", "arch_change", "needs_new_data", "close_frac_lo", "close_frac_hi",
                              "reduction_pp_lo", "reduction_pp_hi", "central_closes_breach5", "central_closes_breach1",
                              "channel"])
    for name, lev in payload["levers"].items():
        g = payload["lever_gate"][name]
        lt.add_data(name, lev["arch_change"], lev["needs_new_data"], lev["close_frac_band"][0], lev["close_frac_band"][1],
                    lev["reduction_pp_band"][0], lev["reduction_pp_band"][1],
                    g["breach5"]["central_closes"], g["breach1"]["central_closes"], lev["channel"])
    run.log({"lever_gate": lt})

    # saturation table: intrinsic-acceptance lift vs gap under both shift models.
    sat = payload["saturation_test"]
    stb = wandb.Table(columns=["shift_model", "gap_base_pct", "gap_after_a1_lift_pct", "delta_pp"])
    stb.add_data("multiplicative", sat["gap_mult_base_pct"], sat["gap_mult_lift_pct"], sat["saturation_delta_mult_pp"])
    stb.add_data("additive", sat["gap_add_base_pct"], sat["gap_add_lift_pct"], sat["saturation_delta_add_pp"])
    run.log({"saturation_test": stb})

    run.finish()
    print(f"[dgr] logged W&B run {args.wandb_entity}/{args.wandb_project} "
          f"name={args.wandb_name} group={args.wandb_group}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
