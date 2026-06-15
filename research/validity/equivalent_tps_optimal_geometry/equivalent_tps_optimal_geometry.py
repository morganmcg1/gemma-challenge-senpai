#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Equivalent-TPS-optimal speculator geometry (draft length K, verify width M=K+1) (PR #413, denken).

THE QUESTION (re-scope from the human, Issue #407): forget 500+; find the geometry that MAXIMIZES
  single-stream TPS subject to STRICT byte-exact greedy-token-equivalence. The deployed geometry is
  K_spec=7 / verify-width M=8 (linear chain: verify the 7 drafted + 1 bonus = 8 candidates) -- and it
  was tuned for RAW (non-equivalent) TPS. The equivalence-restoration tax is M-dependent (the M=8
  batched verify carries 3/882 reduction-order flips #381/#405; M=1 carries 0; near-ties grow with the
  verify batch width). So the EQUIVALENCE-optimal draft length K* might be SHORTER than 7: a shorter
  chain -> smaller M -> fewer near-tie flips -> cheaper to make byte-identical. Find K* and quantify
  equiv_tps(K*) - equiv_tps(7). Headline: is the deployed draft TOO LONG for the *equivalence* goal?

THE ANSWER (decision-critical, honest -- the hypothesis is LARGELY REFUTED):
  K* = 7 (== deployed) across the BULK of the deployment-consistent parameter space. The deployed
  draft is NOT too long for the equivalence objective. Two mechanisms kill the "shorter K is cheaper"
  intuition:
    1. The verify-M roofline N_nr(M)=(ceil(M/4)+1)*2 is FLAT across M in {5,6,7,8} (BLOCK_Q=4
       query-block quantization -> all four widths share ONE tile tier, N_nr=6). So trimming K=7->6
       (M=8->7) recovers EXACTLY ZERO step-time roofline tax. To drop a query-block tier you must
       reach M<=4 (K<=3), which collapses E[T] from 3.851 to <=2.722 -- far too much acceptance lost.
    2. The identity tax is an ABSOLUTE level set by the batched-verify width (2.6 TPS at M=8, #397),
       but the MARGINAL tax of the LAST draft step (M=7->8) is small: <=0.371 TPS (rows-linear near-
       tie model), <=0.690 (quadratic), 0 (query-block-reduction model). The advisor's premise
       conflated the absolute M=8 tax (2.6) with the marginal cost of the 7th draft step. Cutting
       K=7->6 only buys back that small MARGINAL tax, while it loses ~6.5 TPS of raw acceptance speed
       for any drafter cost in the interior of the raw-optimal band.
  => K* = 7 for t_d below ~0.0547 (the top ~3% edge of the raw-optimal drafter-cost band) and for the
     query-block-reduction near-tie model ACROSS THE WHOLE band. K* dips to 6 ONLY in a thin sliver at
     the expensive-drafter edge where K=7 is ALREADY a raw near-tie with K=6, and even there the gain
     is <1 TPS (+0.354 rows-linear / +0.690 quadratic). Base-invariant (481.53 fast vs 467.48 floor
     give the same crossover). equiv_tps(7) = 481.53 - 2.6 = 478.93 TPS, reconciling the #397 selective-
     recompute band [476,479] and sitting above the #393 blanket-strict floor 467.48.

WHY THE COST MODEL IS DEPLOYMENT-PINNED (the one soft knob, handled like beta in #409):
  The deployed drafter is a SEPARATE small model (the qat-assistant), drafting AUTOREGRESSIVELY width-1
  for K iterations (sitecustomize: K width-1 drafter forwards, then one split-KV target verify over
  M=K+1 rows). Its per-forward cost t_d is NOT the target verify cost (a full-target MTP-head reading
  would over-cost the draft and would make K=7 NOT raw-optimal -- refuted by deployment). So t_d is the
  genuinely-uncertain parameter. We PIN its plausible band by deployment-consistency: t_d in
  [t_d_lo, t_d_hi] is EXACTLY the range for which the deployed K=7 is the raw-TPS argmax. We sweep t_d
  across this band and report K* robustness, and independently sweep >=2 near-tie growth models for the
  identity tax (rows-linear, query-block-reduction, quadratic, cubic).

GREEDY IDENTITY (exact by construction at every K): the linear-chain spec verify emits the target's
  argmax token (drafter only PROPOSES; the verify accepts the longest target-argmax-matching prefix and
  emits the target argmax at the first mismatch), so the emitted token is the target greedy token at
  EVERY K -> PPL UNCHANGED 2.3772 <= 2.42 for all K. The equiv_tax is precisely the cost of making the
  M=K+1 BATCHED verify byte-identical to the M=1 sequential reference (removing the #381/#405 reduction-
  order flips). At M=1 (K=0) there is no batch -> 0 tax -> trivially identical. The tree dimension
  (M>K+1) is already closed negligible by denken #409 (+1.33 TPS, beta-fragile); this card is the
  LINEAR chain only.

WHAT THIS IS / IS NOT:
  Pure-CPU analytic card (stdlib math). 0 GPU, 0 official TPS, 0 HF Job, NO served-file change, NO
  submission, NO kernel build, analysis_only=True. Imports the BANKED #402 module
  (tree_verify_net_tps_go_nogo) byte-exactly for: the corrected strict base 467.475 (#393), the fast
  frontier MU_P=481.53 (#52), the step calibration K_CAL (#344), the #289 per-position acceptance
  ladder + E[T]=3.851, the verify-M roofline N_nr (#402/#332), the served attn step-fraction 9.5%
  (#378/#393), and the demand secant S (#387/#399). NOTHING from those anchors is re-derived. The ONLY
  new modelling is (a) E[T](K) swept over draft length from the ladder, (b) the per-cycle cost
  T_cycle(K) = K drafter forwards + 1 verify-M forward with t_d pinned by deployment-consistency, and
  (c) equiv_tax(M) = (M=8 anchor 2.6) * neartie_ratio(M) under several growth models. The M=8 tax 2.6
  is a ONE-LINE calibratable anchor (EQUIV_TAX_AT_M8) -- swap in stark #412's measured value trivially.

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python -m research.validity.equivalent_tps_optimal_geometry.\
equivalent_tps_optimal_geometry --self-test
    cd target/ && .venv/bin/python -m research.validity.equivalent_tps_optimal_geometry.\
equivalent_tps_optimal_geometry \
      --wandb_group equivalent-tps-optimal-geometry --wandb_name denken/equivalent-tps-optimal-geometry
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

# ---- import the BANKED #402 machinery byte-exactly (NOTHING re-derived) ----------------------------
from research.validity.tree_verify_net_tps_go_nogo import tree_verify_net_tps_go_nogo as t402

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors re-exported from #402 (and the upstream merged cards it banks)
# ===========================================================================
BASE_467: float = t402.BASE_467          # #393 (0q7ynumg) corrected realized BLANKET-strict decode TPS
GAP_32: float = t402.GAP_32              # 500 - BASE_467 (strict gap, banked #393)
MU_P: float = t402.MU_P                  # 481.53 deployed FAST (non-equivalent) frontier (#52, 2x9fm2zx)
K_CAL: float = t402.K_CAL                # 125.268 steps/s; public TPS = E[T] * K_cal (#344)
E_T_REALIZED: float = t402.E_T_REALIZED  # MU_P/K_CAL = 3.844 secant-consistent realized accept length
S_CENTRAL: float = t402.S_CENTRAL        # demand secant E[T]/cov (#387, 7.912609) -- imported for x-check
LADDER_289: list[float] = list(t402.LADDER_289)   # per-position conditional acceptance a_1..a_7 (#289)
E_ACCEPTED_289: float = t402.E_ACCEPTED_289       # 2.851185944363104 (#289)
E_T_289: float = t402.E_T_289                     # 3.851185944363104 = 1 + E[accepted] (#289 ladder)
F_ATTN_STEP: float = t402.F_ATTN_STEP    # M=8 verify attention = 9.5% of the served spec step (#378/#393)
M_DEPLOYED: int = t402.M_DEPLOYED        # 8 deployed verify rows = K_spec(7) + 1 bonus (linear chain)
BLOCK_Q: int = t402.BLOCK_Q              # 4 query rows per CTA (split-KV tiling; the tax quantum)
K_DEPLOYED: int = 7                      # deployed draft length (num_speculative_tokens=7, manifest)

PPL_DEPLOYED: float = t402.PPL_DEPLOYED  # 2.3772
PPL_GATE: float = t402.PPL_GATE          # 2.42
TARGET: float = 500.0

# ---- the EQUIVALENCE-restoration tax anchor (the ONE-LINE calibratable knob; stark #412 supersedes) -
# #397 models the M=8 selective-recompute identity tax at ~2.6 TPS: 481.53 (fast) - 2.6 = 478.9, which
# lands in the strict-equivalent selective band [476,479] and above the #393 blanket floor 467.48.
# stark #412 is MEASURING this; to swap in its number, change ONLY this constant.
EQUIV_TAX_AT_M8: float = 2.6             # #397 modeled selective-recompute identity tax at M=8 (TPS)
EQUIV_TPS_SELECTIVE_LO: float = 476.0    # #397 selective-recompute strict-equivalent band lower
EQUIV_TPS_SELECTIVE_HI: float = 479.0    # #397 selective-recompute strict-equivalent band upper

TOL_PROV: float = 1e-6
K_MAX_SWEEP: int = 12                    # sweep K up to here to GUARANTEE no higher-K optimum is missed
K_GRID: list[int] = list(range(1, 11))   # PR asks for K in {1..10} reporting


# ===========================================================================
# Section 1 -- E[T](K): expected emitted tokens per cycle from the #289 ladder (linear chain)
# ===========================================================================
# Linear chain of length K: position p accepts w.p. a_p CONDITIONAL on p-1 accepted (the #289 ladder).
# E[accepted](K) = sum_{k=1..K} prod_{j<=k} a_j ; E[T](K) = 1 + E[accepted](K) (the +1 bonus token the
# target always emits). K<=7 uses the ladder directly; K>7 HOLDS the last rung a_7 (acceptance plateaus
# ~0.846 -- the conservative saturation assumption; the K>7 tail is never the equivalence optimum so it
# is not load-bearing, but reported for completeness and flagged).

def a_cond(k: int) -> float:
    """Per-position conditional acceptance a_k; hold-last (a_7) for k>7 (saturation, flagged)."""
    return LADDER_289[k - 1] if k <= len(LADDER_289) else LADDER_289[-1]


def expected_accepted(k: int) -> float:
    """E[accepted draft tokens] for a length-K linear chain = sum_k prod_{j<=k} a_j."""
    cum, acc = 1.0, 0.0
    for kk in range(1, k + 1):
        cum *= a_cond(kk)
        acc += cum
    return acc


def expected_tokens(k: int) -> float:
    """E[T](K) = 1 + E[accepted](K) (emitted tokens per spec cycle)."""
    return 1.0 + expected_accepted(k)


def marginal_accepted(k: int) -> float:
    """The accepted-token gain from the K-th draft slot = prod_{j<=K} a_j (E[T](K)-E[T](K-1))."""
    cum = 1.0
    for kk in range(1, k + 1):
        cum *= a_cond(kk)
    return cum


# ===========================================================================
# Section 2 -- T_cycle(K): per-cycle wall time = K drafter forwards + 1 verify-M forward (M=K+1)
# ===========================================================================
# Deployed cycle (sitecustomize): K AUTOREGRESSIVE width-1 drafter forwards (separate small qat-assistant
# model, cost t_d each, K-independent per forward) + ONE split-KV target verify over M=K+1 rows. We
# normalize the deployed cycle T_cycle(7) = 1. Within it the verify-M=8 attention lane is F_ATTN_STEP =
# 9.5% (#378/#393), scaling with the #402 roofline N_nr(M); the verify weight-bound lanes (lm_head + body
# GEMMs, << compute crossover at these M) are flat in M. So:
#       T_cycle(K) = K*t_d + w_v + a_v * N_nr(K+1),     a_v*N_nr(8) = F_ATTN_STEP,  7*t_d + w_v = 1 - F_ATTN_STEP.
# The split between draft (7*t_d) and verify-weight-bound (w_v) is the genuinely-uncertain draft-vs-verify
# cost ratio. The deployed drafter is a separate SMALL model, NOT an on-target MTP head, so we do NOT cost
# it from the target roofline (that would over-cost it and break K=7 raw-optimality). Instead t_d is a free
# parameter pinned to the band where the deployed K=7 is the raw-TPS argmax (deployment-consistency).

A_V: float = F_ATTN_STEP / t402.n_nonreduction(M_DEPLOYED)   # verify attention coeff (T_cycle(7)=1 norm)
NONVERIFY_ATTN_BUDGET: float = 1.0 - F_ATTN_STEP             # 7*t_d + w_v share of the deployed cycle


def n_nr(m: int) -> int:
    """#402 verify roofline query-block x KV-head tile count N_nr(M)=(ceil(M/4)+1)*2 (byte-exact import)."""
    return int(t402.n_nonreduction(m))


def w_v_of_td(t_d: float) -> float:
    """Verify weight-bound share given drafter forward cost t_d (so 7*t_d + w_v = 1 - F_ATTN_STEP)."""
    return NONVERIFY_ATTN_BUDGET - K_DEPLOYED * t_d


def t_cycle(k: int, t_d: float) -> float:
    """Per-cycle wall time (deployed cycle normalized to 1): K drafter forwards + verify-M(K+1) forward."""
    return k * t_d + w_v_of_td(t_d) + A_V * n_nr(k + 1)


def tau_step(k: int, t_d: float) -> float:
    """The PR's tau_step(K,M=K+1): fractional per-cycle step-time change vs the deployed K=7 cycle."""
    return t_cycle(k, t_d) / t_cycle(K_DEPLOYED, t_d) - 1.0


def fast_tps(k: int, t_d: float, base_fast: float = MU_P) -> float:
    """NON-equivalent fast-path TPS at draft length K, anchored at the deployed frontier base_fast=481.53.
    fast_tps(K) = base * [E[T](K)/E[T](7)] / [T_cycle(K)/T_cycle(7)] = base*(E[T] ratio)/(1+tau_step)."""
    return base_fast * (expected_tokens(k) / E_T_289) / (t_cycle(k, t_d) / t_cycle(K_DEPLOYED, t_d))


def raw_argmax_k(t_d: float, kmax: int = K_MAX_SWEEP, base_fast: float = MU_P) -> int:
    """argmax_K fast_tps(K) (the RAW, non-equivalent optimum). Deployment chose this == 7."""
    return max(range(1, kmax + 1), key=lambda k: fast_tps(k, t_d, base_fast))


def deployment_consistent_td_band(kmax: int = K_MAX_SWEEP, base_fast: float = MU_P,
                                  lo: float = 1e-4, hi: float = 0.13, step: float = 5e-5) -> dict:
    """Scan t_d for the (contiguous) band where the deployed K=7 is the raw-TPS argmax. Returns
    [lo,hi,center]; center (maximin, furthest from both tie-edges) is the nominal drafter-cost estimate.
    The deployment settling on K=7 (not 6 or 8) is revealed-preference evidence that t_d is INTERIOR."""
    band: list[float] = []
    x = lo
    while x <= hi + 1e-12:
        if raw_argmax_k(x, kmax, base_fast) == K_DEPLOYED:
            band.append(x)
        x += step
    if not band:
        return {"found": False, "lo": None, "hi": None, "center": None, "width": 0.0}
    blo, bhi = band[0], band[-1]
    return {"found": True, "lo": blo, "hi": bhi, "center": 0.5 * (blo + bhi), "width": bhi - blo}


# ===========================================================================
# Section 3 -- equiv_tax(M): identity-restoration tax, calibrated to 2.6 TPS at M=8, swept over models
# ===========================================================================
# equiv_tax(M) = EQUIV_TAX_AT_M8 * neartie_ratio(M),  neartie_ratio(M) = neartie_frac(M)/neartie_frac(8).
# Each near-tie model VANISHES at M=1 (no batch -> no reduction-order divergence -> 0 flips). The #381/#405
# mechanism is reduction-order divergence across the M parallel verify rows; the most direct growth is in
# the number of batched rows (rows-linear), with quadratic/cubic as superlinear (pairwise/triple divergence)
# variants, and the query-block-reduction model (tied to N_nr, the split-KV tile count) as the alternative.

def _neartie_frac(m: int, model: str) -> float:
    """Unnormalized near-tie fraction (only ratios matter); all vanish at M=1."""
    if model == "rows_linear":
        return max(m - 1, 0)                       # ∝ batched verify rows (the direct #381/#405 mechanism)
    if model == "reductions_nr":
        return max(n_nr(m) - n_nr(1), 0)           # ∝ split-KV query-block reduction count (N_nr - N_nr(1))
    if model == "rows_quadratic":
        return max(m - 1, 0) ** 2                  # ∝ pairwise row divergence (superlinear)
    if model == "rows_cubic":
        return max(m - 1, 0) ** 3                  # ∝ triple-wise (steepest superlinear, pessimistic)
    raise ValueError(f"unknown neartie model {model!r}")


NEARTIE_MODELS: list[str] = ["rows_linear", "reductions_nr", "rows_quadratic", "rows_cubic"]
NOMINAL_NEARTIE_MODEL: str = "rows_linear"   # most direct row-batch mechanism; headline nominal


def equiv_tax(m: int, model: str = NOMINAL_NEARTIE_MODEL, tax_m8: float | None = None) -> float:
    """TPS to restore strict identity-1.0 at verify width M. Calibrated: equiv_tax(8)=tax_m8 (#397, 2.6);
    equiv_tax(1)=0 (M=1 no batch). Swap stark #412's measured M=8 tax via EQUIV_TAX_AT_M8 (one-line source
    edit) or the tax_m8 arg / --equiv-tax-m8 CLI flag; tax_m8=None reads the live global at CALL time."""
    if tax_m8 is None:
        tax_m8 = EQUIV_TAX_AT_M8
    denom = _neartie_frac(M_DEPLOYED, model)
    if denom == 0:
        return 0.0
    return tax_m8 * _neartie_frac(m, model) / denom


def marginal_equiv_tax_last_step(model: str = NOMINAL_NEARTIE_MODEL) -> float:
    """The MARGINAL identity tax of the LAST (7th) draft step: equiv_tax(M=8) - equiv_tax(M=7). This is
    what a K=7->6 trim buys back -- NOT the full 2.6 absolute tax. Small for every model (<=0.69)."""
    return equiv_tax(M_DEPLOYED, model) - equiv_tax(M_DEPLOYED - 1, model)


# ===========================================================================
# Section 4 -- equiv_tps(K) = fast_tps(K) - equiv_tax(K+1); the K-sweep, K*, and the gain vs deployed 7
# ===========================================================================

def equiv_tps(k: int, t_d: float, model: str = NOMINAL_NEARTIE_MODEL, base_fast: float = MU_P,
              tax_m8: float | None = None) -> float:
    """Strictly-equivalent (selective-recompute) TPS at draft length K (linear chain, M=K+1).
    tax_m8=None reads the live EQUIV_TAX_AT_M8 global at CALL time (so --equiv-tax-m8 propagates)."""
    if tax_m8 is None:
        tax_m8 = EQUIV_TAX_AT_M8
    return fast_tps(k, t_d, base_fast) - equiv_tax(k + 1, model, tax_m8)


def sweep_k(t_d: float, model: str = NOMINAL_NEARTIE_MODEL, base_fast: float = MU_P,
            tax_m8: float | None = None, kmax: int = K_MAX_SWEEP) -> dict:
    """Compute equiv_tps(K) over K=1..kmax, find K* = argmax, and the gain vs the deployed K=7.
    tax_m8=None reads the live EQUIV_TAX_AT_M8 global at CALL time (so --equiv-tax-m8 propagates)."""
    if tax_m8 is None:
        tax_m8 = EQUIV_TAX_AT_M8
    rows = []
    for k in range(1, kmax + 1):
        rows.append({
            "k": k, "m": k + 1,
            "e_t": expected_tokens(k), "e_accepted": expected_accepted(k),
            "marginal_accepted": marginal_accepted(k),
            "n_nr_verify": n_nr(k + 1), "tau_step": tau_step(k, t_d),
            "fast_tps": fast_tps(k, t_d, base_fast),
            "equiv_tax": equiv_tax(k + 1, model, tax_m8),
            "equiv_tps": equiv_tps(k, t_d, model, base_fast, tax_m8),
        })
    kstar = max(rows, key=lambda r: r["equiv_tps"])["k"]
    eq_kstar = equiv_tps(kstar, t_d, model, base_fast, tax_m8)
    eq_7 = equiv_tps(K_DEPLOYED, t_d, model, base_fast, tax_m8)
    return {
        "t_d": t_d, "model": model, "base_fast": base_fast, "tax_m8": tax_m8,
        "rows": rows, "k_star": kstar, "m_star": kstar + 1,
        "equiv_tps_at_kstar": eq_kstar, "equiv_tps_at_7": eq_7,
        "equiv_tps_gain_vs_deployed7": eq_kstar - eq_7,
        "kstar_below_7": bool(kstar < K_DEPLOYED),
        "raw_argmax_k": raw_argmax_k(t_d, kmax, base_fast),
    }


def robustness_grid(band: dict, base_fast: float = MU_P, tax_m8: float | None = None) -> dict:
    """K* across (t_d in {lo, center, hi}) x (every near-tie model). The headline robustness object.
    tax_m8=None reads the live EQUIV_TAX_AT_M8 global at CALL time (so --equiv-tax-m8 propagates)."""
    if tax_m8 is None:
        tax_m8 = EQUIV_TAX_AT_M8
    td_points = {"lo": band["lo"], "center": band["center"], "hi": band["hi"]}
    grid: dict[str, dict] = {}
    for tname, td in td_points.items():
        for model in NEARTIE_MODELS:
            s = sweep_k(td, model, base_fast, tax_m8)
            grid[f"{tname}__{model}"] = {
                "t_d": td, "td_point": tname, "model": model,
                "k_star": s["k_star"], "kstar_below_7": s["kstar_below_7"],
                "equiv_tps_at_kstar": s["equiv_tps_at_kstar"],
                "equiv_tps_gain_vs_deployed7": s["equiv_tps_gain_vs_deployed7"],
            }
    # at the NOMINAL operating point (center t_d), is K* identical across all near-tie models?
    center_kstars = {m: grid[f"center__{m}"]["k_star"] for m in NEARTIE_MODELS}
    kstar_robust_across_neartie = len(set(center_kstars.values())) == 1
    nominal_kstar = grid[f"center__{NOMINAL_NEARTIE_MODEL}"]["k_star"]
    # over the WHOLE grid, does K* ever drop below 7? where?
    below7_cells = {kk: v for kk, v in grid.items() if v["kstar_below_7"]}
    max_gain_anywhere = max((v["equiv_tps_gain_vs_deployed7"] for v in grid.values()), default=0.0)
    return {
        "grid": grid,
        "center_kstars_by_model": center_kstars,
        "kstar_robust_across_neartie_models": bool(kstar_robust_across_neartie),
        "nominal_kstar": nominal_kstar,
        "nominal_kstar_below_7": bool(nominal_kstar < K_DEPLOYED),
        "below7_cells": list(below7_cells.keys()),
        "kstar_ever_below_7_in_band": bool(len(below7_cells) > 0),
        "max_equiv_gain_anywhere_in_band": max_gain_anywhere,
        "reductions_nr_kstar_all_band": [grid[f"{tp}__reductions_nr"]["k_star"]
                                         for tp in ("lo", "center", "hi")],
    }


def kstar_flip_crossover_td(model: str, band: dict, base_fast: float = MU_P,
                            tax_m8: float | None = None, step: float = 5e-5) -> float | None:
    """The t_d at which K* first flips off 7 inside the band (None if it never does for this model).
    tax_m8=None reads the live EQUIV_TAX_AT_M8 global at CALL time (so --equiv-tax-m8 propagates)."""
    if tax_m8 is None:
        tax_m8 = EQUIV_TAX_AT_M8
    if not band["found"]:
        return None
    x = band["lo"]
    while x <= band["hi"] + 1e-12:
        if sweep_k(x, model, base_fast, tax_m8)["k_star"] != K_DEPLOYED:
            return x
        x += step
    return None


def min_k_to_drop_roofline_tier() -> dict:
    """The smallest M (and K=M-1) that drops a verify query-block tier below the deployed N_nr(8)=6, plus
    the E[T] sacrificed. Shows the ONLY way to recover roofline tax is K<=3 (far too much acceptance lost)."""
    n8 = n_nr(M_DEPLOYED)
    m_drop = next((m for m in range(M_DEPLOYED, 0, -1) if n_nr(m) < n8), 1)
    k_drop = m_drop - 1
    return {
        "n_nr_deployed_m8": n8, "first_lower_tier_M": m_drop, "first_lower_tier_K": k_drop,
        "n_nr_at_lower_tier": n_nr(m_drop),
        "e_t_at_deployed_7": expected_tokens(K_DEPLOYED),
        "e_t_at_lower_tier_K": expected_tokens(k_drop),
        "e_t_sacrificed": expected_tokens(K_DEPLOYED) - expected_tokens(k_drop),
        "n_nr_flat_m5_to_m8": [n_nr(m) for m in (5, 6, 7, 8)],
    }


# ===========================================================================
# Section 5 -- self-tests (>= 20 checks; PRIMARY gate)
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(band: dict, nominal: dict, robust: dict, tier: dict) -> dict:
    c: dict[str, bool] = {}

    # a) provenance: banked anchors imported byte-exactly from #402 / upstream merged cards.
    c["a_base467_is_393"] = abs(BASE_467 - 467.475218449957) < TOL_PROV
    c["a_mu_p_is_481p53"] = abs(MU_P - 481.53) < TOL_PROV
    c["a_e_t_realized_is_mu_over_kcal"] = abs(E_T_REALIZED - MU_P / K_CAL) < 1e-9
    c["a_ladder_len_7"] = len(LADDER_289) == 7
    c["a_ladder_monotone_increasing"] = all(LADDER_289[i] <= LADDER_289[i + 1] for i in range(6))
    c["a_secant_matches_387"] = abs(S_CENTRAL - 7.912609135742992) < 1e-9
    c["a_f_attn_step_is_378"] = abs(F_ATTN_STEP - 0.09506718019009251) < TOL_PROV
    c["a_block_q_is_4"] = BLOCK_Q == 4
    c["a_m_deployed_is_kdep_plus_1"] = M_DEPLOYED == K_DEPLOYED + 1

    # b) E[T](K) from the #289 ladder: reproduces #289 at K=7, monotone, saturating.
    c["b_et7_reproduces_289"] = abs(expected_tokens(7) - E_T_289) < 1e-6
    c["b_eaccepted7_reproduces_289"] = abs(expected_accepted(7) - E_ACCEPTED_289) < 1e-9
    c["b_et_monotone_in_k"] = all(expected_tokens(k) < expected_tokens(k + 1) for k in range(1, 10))
    c["b_marginal_accepted_decreasing"] = all(
        marginal_accepted(k) > marginal_accepted(k + 1) for k in range(1, 10))
    c["b_et1_is_1_plus_a1"] = abs(expected_tokens(1) - (1.0 + LADDER_289[0])) < 1e-12
    c["b_ladder_holdlast_beyond_7"] = abs(a_cond(8) - LADDER_289[-1]) < 1e-12

    # c) verify-M roofline import + the KEY flat-tier mechanism (BLOCK_Q=4 quantization).
    c["c_n_nr_m8_is_6"] = n_nr(8) == 6
    c["c_n_nr_m1_is_4"] = n_nr(1) == 4
    c["c_n_nr_flat_m5_to_m8"] = n_nr(5) == n_nr(6) == n_nr(7) == n_nr(8) == 6
    c["c_n_nr_jumps_at_m9"] = n_nr(9) == 8 and n_nr(9) > n_nr(8)
    c["c_first_lower_tier_is_m4_k3"] = tier["first_lower_tier_M"] == 4 and tier["first_lower_tier_K"] == 3

    # d) cost model + deployment-consistency: K=7 is the raw argmax for a non-empty t_d band.
    c["d_a_v_recovers_f_attn_step"] = abs(A_V * n_nr(8) - F_ATTN_STEP) < 1e-12
    c["d_t_cycle_7_normalized_to_1"] = abs(t_cycle(7, band["center"]) - 1.0) < 1e-9
    c["d_tau_step_7_is_zero"] = abs(tau_step(7, band["center"])) < 1e-12
    c["d_t_cycle_monotone_in_k"] = all(
        t_cycle(k, band["center"]) < t_cycle(k + 1, band["center"]) for k in range(1, 10))
    c["d_band_found_nonempty"] = band["found"] and band["width"] > 0.0
    c["d_raw_argmax_at_center_is_7"] = raw_argmax_k(band["center"]) == K_DEPLOYED
    c["d_w_v_nonneg_in_band"] = w_v_of_td(band["hi"]) >= 0.0

    # e) equiv_tax model: calibrated to 2.6 at M=8, 0 at M=1, monotone; SMALL marginal last-step tax.
    c["e_tax_m8_is_2p6_all_models"] = all(
        abs(equiv_tax(8, m) - EQUIV_TAX_AT_M8) < 1e-12 for m in NEARTIE_MODELS)
    c["e_tax_m1_is_zero_all_models"] = all(abs(equiv_tax(1, m)) < 1e-12 for m in NEARTIE_MODELS)
    c["e_tax_monotone_in_m_all_models"] = all(
        all(equiv_tax(m, mod) <= equiv_tax(m + 1, mod) + 1e-15 for m in range(1, 12))
        for mod in NEARTIE_MODELS)
    c["e_marginal_last_step_small"] = all(
        marginal_equiv_tax_last_step(m) <= 1.0 for m in NEARTIE_MODELS)
    c["e_reductions_nr_marginal_is_zero"] = abs(marginal_equiv_tax_last_step("reductions_nr")) < 1e-12
    c["e_tax_m8_swap_is_one_line"] = abs(equiv_tax(8, NOMINAL_NEARTIE_MODEL, tax_m8=9.9) - 9.9) < 1e-12

    # f) the DECISION: anchors, equiv_tps(7) reconciliation, K*=7 nominal, gain=0, K*<=7 always.
    c["f_fast_tps_7_is_mu_p"] = abs(fast_tps(7, band["center"]) - MU_P) < 1e-9
    c["f_equiv_tps_7_is_478p93"] = abs(nominal["equiv_tps_at_7"] - (MU_P - EQUIV_TAX_AT_M8)) < 1e-9
    c["f_equiv_tps_7_in_selective_band"] = EQUIV_TPS_SELECTIVE_LO <= nominal["equiv_tps_at_7"] <= EQUIV_TPS_SELECTIVE_HI
    c["f_equiv_tps_7_above_blanket_floor"] = nominal["equiv_tps_at_7"] > BASE_467
    c["f_nominal_kstar_is_7"] = nominal["k_star"] == K_DEPLOYED
    c["f_nominal_kstar_not_below_7"] = nominal["kstar_below_7"] is False
    c["f_nominal_gain_is_zero"] = abs(nominal["equiv_tps_gain_vs_deployed7"]) < 1e-9
    c["f_kstar_never_above_7"] = all(
        sweep_k(td, NOMINAL_NEARTIE_MODEL)["k_star"] <= K_DEPLOYED
        for td in (band["lo"], band["center"], band["hi"]))

    # g) robustness: K* robust across near-tie models at nominal; reductions_nr K*=7 across whole band.
    c["g_kstar_robust_across_neartie_at_center"] = robust["kstar_robust_across_neartie_models"]
    c["g_center_all_models_kstar_7"] = all(v == 7 for v in robust["center_kstars_by_model"].values())
    c["g_reductions_nr_kstar_7_whole_band"] = all(k == 7 for k in robust["reductions_nr_kstar_all_band"])
    c["g_max_gain_anywhere_under_1tps"] = robust["max_equiv_gain_anywhere_in_band"] < 1.0

    # h) the roofline-tier argument: recovering tax needs K<=3, which sacrifices >1 token of E[T].
    c["h_lower_tier_needs_k_le_3"] = tier["first_lower_tier_K"] <= 3
    c["h_lower_tier_sacrifices_et"] = tier["e_t_sacrificed"] > 1.0

    # i) base-invariance: K* at the high edge is the same anchored at MU_P (fast) vs BASE_467 (floor).
    s_hi_fast = sweep_k(band["hi"], NOMINAL_NEARTIE_MODEL, base_fast=MU_P)["k_star"]
    s_hi_floor = sweep_k(band["hi"], NOMINAL_NEARTIE_MODEL, base_fast=BASE_467)["k_star"]
    c["i_kstar_base_invariant_at_hi_edge"] = s_hi_fast == s_hi_floor

    # j) PPL / greedy identity: linear-chain verify emits target argmax at every K -> PPL unchanged.
    c["j_ppl_passes_gate"] = PPL_DEPLOYED <= PPL_GATE

    # k) numeric hygiene.
    flat = [band["center"], nominal["equiv_tps_at_kstar"], nominal["equiv_tps_at_7"],
            fast_tps(7, band["center"]), equiv_tax(8), robust["max_equiv_gain_anywhere_in_band"],
            tier["e_t_sacrificed"]]
    c["k_no_nan_inf"] = all(_finite(v) for v in flat)

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v),
            "passes": passes}


# ===========================================================================
# Section 6 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    band = deployment_consistent_td_band()
    nominal = sweep_k(band["center"], NOMINAL_NEARTIE_MODEL)
    robust = robustness_grid(band)
    tier = min_k_to_drop_roofline_tier()
    selftest = run_self_tests(band, nominal, robust, tier)

    flip_crossovers = {m: kstar_flip_crossover_td(m, band) for m in NEARTIE_MODELS}

    geometry_str = (
        "linear chain, draft length K, verify width M=K+1. Deployed K=7/M=8 (manifest "
        "num_speculative_tokens=7). Cycle = K autoregressive width-1 drafter forwards (separate small "
        "qat-assistant, cost t_d each) + 1 split-KV target verify over M=K+1 rows. T_cycle(7) normalized "
        "to 1; verify-M attention = F_ATTN_STEP(9.5%)*N_nr(M)/N_nr(8) (#402 roofline); t_d pinned by "
        "deployment-consistency to the band where K=7 is the raw-TPS argmax. equiv_tax(M)=2.6(#397)*"
        "neartie_ratio(M), calibratable via EQUIV_TAX_AT_M8 (stark #412 supersedes)."
    )
    return {
        "pr": 413, "agent": "denken", "kind": "equivalent-tps-optimal-geometry",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_fast_frontier_tps": MU_P, "baseline_fast_frontier_ppl": PPL_DEPLOYED,
        "blanket_strict_base_tps": BASE_467, "gap_to_500_tps": GAP_32,
        "inputs": {
            "base_467_393": BASE_467, "mu_p_fast_52": MU_P, "k_cal_344": K_CAL,
            "e_t_realized_396": E_T_REALIZED, "gap_32_393": GAP_32,
            "ladder_289": LADDER_289, "e_accepted_289": E_ACCEPTED_289, "e_t_289": E_T_289,
            "f_attn_step_378": F_ATTN_STEP, "block_q_332": BLOCK_Q, "m_deployed": M_DEPLOYED,
            "k_deployed": K_DEPLOYED, "s_central_387": S_CENTRAL,
            "equiv_tax_at_m8_397": EQUIV_TAX_AT_M8,
            "equiv_tps_selective_band_397": [EQUIV_TPS_SELECTIVE_LO, EQUIV_TPS_SELECTIVE_HI],
            "a_v_verify_attn_coeff": A_V, "nonverify_attn_budget": NONVERIFY_ATTN_BUDGET,
            "neartie_models": NEARTIE_MODELS, "nominal_neartie_model": NOMINAL_NEARTIE_MODEL,
            "target": TARGET, "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
            "k_grid": K_GRID, "k_max_sweep": K_MAX_SWEEP,
            "source_393_run": "0q7ynumg", "source_402_run": "8pcyhe2r", "source_409_run": "(tree neg)",
            "source_399_run": "ec7i3z5t", "source_387_run": "z8osvif8", "source_289_run": "fi34s269",
            "source_332_run": "y5cl0ena", "source_378_ref": "served attn step-frac 9.5%",
            "source_397_ref": "selective recompute ~2.6 TPS @ M=8",
            "source_412_ref": "stark measuring M=8 selective tax (supersedes EQUIV_TAX_AT_M8)",
            "source_405_381_ref": "3/882 reduction-order flips @ M=8; eps*=0.125=16 ULP",
            "source_407_ref": "human re-scope: maximize fastest strictly-equivalent TPS",
        },
        "geometry": geometry_str,
        "deployment_consistent_td_band": band,
        # ---- HEADLINE deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "k_star": nominal["k_star"],
        "m_star": nominal["m_star"],
        "equiv_tps_at_kstar": nominal["equiv_tps_at_kstar"],
        "equiv_tps_at_deployed7": nominal["equiv_tps_at_7"],
        "equiv_tps_gain_vs_deployed7": nominal["equiv_tps_gain_vs_deployed7"],
        "kstar_below_7": nominal["kstar_below_7"],
        "equiv_tax_at_m8_used": EQUIV_TAX_AT_M8,
        "neartie_frac_model": NOMINAL_NEARTIE_MODEL,
        "kstar_robust_across_neartie_models": robust["kstar_robust_across_neartie_models"],
        "kstar_ever_below_7_in_band": robust["kstar_ever_below_7_in_band"],
        "max_equiv_gain_anywhere_in_band": robust["max_equiv_gain_anywhere_in_band"],
        "marginal_equiv_tax_last_step_by_model": {m: marginal_equiv_tax_last_step(m) for m in NEARTIE_MODELS},
        "kstar_flip_crossover_td_by_model": flip_crossovers,
        "nominal_sweep": nominal,
        "robustness": robust,
        "roofline_tier": tier,
        "ppl_unchanged": PPL_DEPLOYED,
        "self_test": selftest,
        "equivalent_tps_optimal_geometry_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        band = report["deployment_consistent_td_band"]
        wandb.summary.update({
            "geometry": report["geometry"],
            "k_star": report["k_star"], "m_star": report["m_star"],
            "equiv_tps_at_kstar": report["equiv_tps_at_kstar"],
            "equiv_tps_at_deployed7": report["equiv_tps_at_deployed7"],
            "equiv_tps_gain_vs_deployed7": report["equiv_tps_gain_vs_deployed7"],
            "kstar_below_7": report["kstar_below_7"],
            "equiv_tax_at_m8_used": report["equiv_tax_at_m8_used"],
            "neartie_frac_model": report["neartie_frac_model"],
            "kstar_robust_across_neartie_models": report["kstar_robust_across_neartie_models"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "equivalent_tps_optimal_geometry_self_test_passes": report["equivalent_tps_optimal_geometry_self_test_passes"],
        })
        wandb.log({
            "summary/k_star": float(report["k_star"]),
            "summary/m_star": float(report["m_star"]),
            "summary/equiv_tps_at_kstar": report["equiv_tps_at_kstar"],
            "summary/equiv_tps_at_deployed7": report["equiv_tps_at_deployed7"],
            "summary/equiv_tps_gain_vs_deployed7": report["equiv_tps_gain_vs_deployed7"],
            "summary/kstar_below_7": float(report["kstar_below_7"]),
            "summary/equiv_tax_at_m8_used": report["equiv_tax_at_m8_used"],
            "summary/kstar_robust_across_neartie_models": float(report["kstar_robust_across_neartie_models"]),
            "summary/kstar_ever_below_7_in_band": float(report["kstar_ever_below_7_in_band"]),
            "summary/max_equiv_gain_anywhere_in_band": report["max_equiv_gain_anywhere_in_band"],
            "summary/td_band_lo": band["lo"], "summary/td_band_hi": band["hi"],
            "summary/td_band_center": band["center"], "summary/td_band_width": band["width"],
            "summary/blanket_strict_base_tps": BASE_467, "summary/fast_frontier_tps": MU_P,
            "summary/gap_to_500_tps": report["gap_to_500_tps"],
            "summary/e_t_deployed7": E_T_289,
            "summary/roofline_first_lower_tier_K": float(report["roofline_tier"]["first_lower_tier_K"]),
            "summary/roofline_e_t_sacrificed_to_drop_tier": report["roofline_tier"]["e_t_sacrificed"],
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # per-K curve at the nominal operating point (the E[T]/tau/fast/tax/equiv_tps shape).
        for r in report["nominal_sweep"]["rows"]:
            wandb.log({"k_sweep/k": float(r["k"]), "k_sweep/m": float(r["m"]),
                       "k_sweep/e_t": r["e_t"], "k_sweep/marginal_accepted": r["marginal_accepted"],
                       "k_sweep/tau_step": r["tau_step"], "k_sweep/fast_tps": r["fast_tps"],
                       "k_sweep/equiv_tax": r["equiv_tax"], "k_sweep/equiv_tps": r["equiv_tps"],
                       "k_sweep/n_nr_verify": float(r["n_nr_verify"])})
        # robustness grid (t_d x neartie model -> K*).
        for tag, cell in report["robustness"]["grid"].items():
            wandb.log({f"robust/{tag}/k_star": float(cell["k_star"]),
                       f"robust/{tag}/kstar_below_7": float(cell["kstar_below_7"]),
                       f"robust/{tag}/equiv_gain": cell["equiv_tps_gain_vs_deployed7"]})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    band = r["deployment_consistent_td_band"]
    tier = r["roofline_tier"]
    print("\n=== Equivalent-TPS-optimal speculator geometry (PR #413, denken) ===")
    print(f"fast frontier (#52) = {MU_P:.2f} TPS   blanket-strict floor (#393) = {BASE_467:.2f}   "
          f"selective M=8 tax (#397) = {EQUIV_TAX_AT_M8} -> equiv_tps(7) = {MU_P - EQUIV_TAX_AT_M8:.2f}")
    print(f"#289 ladder E[T](7) = {E_T_289:.4f}   verify roofline N_nr(M)=(ceil(M/4)+1)*2 "
          f"(BLOCK_Q={BLOCK_Q}); N_nr flat M5..8 = {tier['n_nr_flat_m5_to_m8']}")
    print(f"deployment-consistent drafter-cost band t_d: [{band['lo']:.4f}, {band['hi']:.4f}]  "
          f"center(nominal) = {band['center']:.4f}  (raw argmax K=7 over this band)")
    print("\n-- K-sweep at nominal t_d (model=%s) --" % r["neartie_frac_model"])
    print(f"  {'K':>2}{'M':>3}{'E[T]':>8}{'marg_acc':>10}{'N_nr':>6}{'tau_step':>10}"
          f"{'fast_tps':>10}{'eq_tax':>8}{'equiv_tps':>11}")
    for row in r["nominal_sweep"]["rows"]:
        if row["k"] > 10:
            continue
        mark = "  <- deployed" if row["k"] == 7 else ("  <- K*" if row["k"] == r["k_star"] else "")
        print(f"  {row['k']:>2}{row['m']:>3}{row['e_t']:>8.4f}{row['marginal_accepted']:>10.4f}"
              f"{row['n_nr_verify']:>6}{row['tau_step']:>10.4f}{row['fast_tps']:>10.2f}"
              f"{row['equiv_tax']:>8.3f}{row['equiv_tps']:>11.3f}{mark}")
    print("\n-- HEADLINE --")
    print(f"  K* = {r['k_star']}  (M* = {r['m_star']})   kstar_below_7 = {r['kstar_below_7']}")
    print(f"  equiv_tps(K*) = {r['equiv_tps_at_kstar']:.3f} TPS   equiv_tps(7) = {r['equiv_tps_at_deployed7']:.3f}")
    print(f"  equiv_tps_gain_vs_deployed7 = {r['equiv_tps_gain_vs_deployed7']:+.3f} TPS")
    print(f"  marginal identity tax of the LAST (7th) draft step (M=7->8), by near-tie model:")
    for m, v in r["marginal_equiv_tax_last_step_by_model"].items():
        print(f"     {m:<16} marginal_tax = {v:+.3f} TPS   (vs absolute M=8 tax {EQUIV_TAX_AT_M8})")
    print("\n-- ROBUSTNESS (K* across t_d band x near-tie model) --")
    print(f"  kstar_robust_across_neartie_models (at center) = {r['kstar_robust_across_neartie_models']}  "
          f"center K* by model = {r['robustness']['center_kstars_by_model']}")
    print(f"  K* ever below 7 in band = {r['kstar_ever_below_7_in_band']}  "
          f"(cells: {r['robustness']['below7_cells']})")
    print(f"  reductions_nr K* across [lo,center,hi] = {r['robustness']['reductions_nr_kstar_all_band']} "
          f"(flat N_nr -> always 7)")
    print(f"  K*-flip crossover t_d by model = "
          + ", ".join(f"{m}:{('%.5f'%v) if v else 'never'}" for m, v in r['kstar_flip_crossover_td_by_model'].items()))
    print(f"  max equiv gain ANYWHERE in band = {r['max_equiv_gain_anywhere_in_band']:+.3f} TPS (<1)")
    print("\n-- WHY K*=7 (roofline-tier argument) --")
    print(f"  N_nr(M=8 deployed) = {tier['n_nr_deployed_m8']}; first LOWER tier at M={tier['first_lower_tier_M']} "
          f"(K={tier['first_lower_tier_K']}), N_nr={tier['n_nr_at_lower_tier']}")
    print(f"  dropping a query-block tier sacrifices E[T] {tier['e_t_at_deployed_7']:.3f} -> "
          f"{tier['e_t_at_lower_tier_K']:.3f}  (lose {tier['e_t_sacrificed']:.3f} tokens/cycle)")
    print(f"\nPPL: linear-chain verify emits target argmax at every K -> PPL unchanged {PPL_DEPLOYED} <= {PPL_GATE}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"equivalent_tps_optimal_geometry_self_test_passes = {r['equivalent_tps_optimal_geometry_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Equivalent-TPS-optimal speculator geometry (PR #413).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #413 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--equiv-tax-m8", type=float, default=None,
                    help="override the M=8 selective-recompute identity tax (default 2.6, #397; #412 supersedes)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="equivalent-tps-optimal-geometry")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/equivalent-tps-optimal-geometry")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/equivalent_tps_optimal_geometry/equivalent_tps_optimal_geometry_results.json")
    args = ap.parse_args()

    global EQUIV_TAX_AT_M8
    if args.equiv_tax_m8 is not None:
        EQUIV_TAX_AT_M8 = float(args.equiv_tax_m8)

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/equivalent_tps_optimal_geometry/equivalent_tps_optimal_geometry_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\nequivalent_tps_optimal_geometry_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "k_star": int(report["k_star"]), "m_star": int(report["m_star"]),
        "equiv_tps_at_kstar": float(report["equiv_tps_at_kstar"]),
        "equiv_tps_gain_vs_deployed7": float(report["equiv_tps_gain_vs_deployed7"]),
        "kstar_below_7": bool(report["kstar_below_7"]),
        "equiv_tax_at_m8_used": float(report["equiv_tax_at_m8_used"]),
        "neartie_frac_model": report["neartie_frac_model"],
        "kstar_robust_across_neartie_models": bool(report["kstar_robust_across_neartie_models"]),
        "kstar_ever_below_7_in_band": bool(report["kstar_ever_below_7_in_band"]),
        "max_equiv_gain_anywhere_in_band": float(report["max_equiv_gain_anywhere_in_band"]),
        "equivalent_tps_optimal_geometry_self_test_passes": bool(report["equivalent_tps_optimal_geometry_self_test_passes"]),
        "primary_metric": {"name": "equiv_tps_at_kstar", "value": float(report["equiv_tps_at_kstar"])},
        "test_metric": {"name": "equivalent_tps_optimal_geometry_self_test_passes",
                        "value": float(report["equivalent_tps_optimal_geometry_self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
