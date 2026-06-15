#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict-500 composite reachability: can ANY composition of known levers clear 500 TPS?

Governing question
------------------
We have a strict-500 target (>=500 accepted tok/step at batch=1 on A10G).  The spec-decode
head (wirbel #354, custom-kernel) already reaches 481.53 TPS at int4 (W4A16).  This script
composes three orthogonal speedup levers to compute the optimistic composite ceiling and asks
whether strict_500 is reachable through known techniques.

The verdict is PARAMETERIZED on the measured sub-int4 bit-width b* (advisor HOLD, PR #357)
-----------------------------------------------------------------------------------------
The original capstone closed sub-int4 (L_quant) using literature PPL deltas (QuIP#/QTIP on
Llama-2 lineage) and held the supply cap fixed at the int4-derived 473.53.  The advisor flagged
two coupled errors (PR #357 review, 2026-06-15):

  1. The PPL gate on sub-int4 must be driven by a MEASURED Gemma-4-E4B PPL (denken #356's
     `ppl_at_best_sub_int4_bits`), NOT transplanted Llama-2 literature ("measure, don't
     guess", #319 11:27Z).  Gemma's GQA/shared-KV + MLP gating may scale very differently.
  2. The 473.53 supply cap is int4-derived and CANNOT be applied to a sub-int4 substrate.
     denken #356's `ceiling(b)` curve RISES as the body shrinks (advisor-relayed anchors:
     473.53 @ 4.0 bpw, 523 @ 3.5 bpw, 585 @ 3.0 bpw).  The moment sub-int4 is PPL-viable the
     substrate moves to denken's b=3 ceiling (585), not 473.53, and the composite re-opens
     above 500.

So both L_quant AND the supply cap are now functions of b*, and the verdict FLIPS on a single
measured number, `measured_ppl_at_best_sub_int4` at bit-width b*:

  measured PPL > 2.42  ->  sub-int4 excluded -> b=4 substrate, cap 473.53 -> NO-GO (gap 26.47)
  measured PPL <= 2.42 ->  sub-int4 LIVE at b* -> cap rises to ceiling(b*) -> >500 candidate

Until that measured input lands this capstone HOLDS: it emits BOTH branches and a non-terminal
`verdict_pending_measured_ppl` instead of stamping "provably out of reach".  It does NOT re-run
any GPU eval (denken owns the single measured-PPL gate; the eval runs once).

Second coupled gate: the verify-locus IDENTITY tax (advisor HOLD #2, PR #357 review 13:08Z)
---------------------------------------------------------------------------------------------
A strict-compliant config must pay the cost of byte-exact greedy identity at the verify locus.
stark #363 (a0oi2esq, MERGED) measured that the ATTENTION-locus identity tax is FREE: a
fixed-split-k / M-invariant attention GEMM restores bit-exact identity at all M in {2,4,8} and
the best K=8 is even faster than the deployed heuristic (eta_ratio 0.9167 < 1).  So the
verify-locus identity tax DECOMPOSES as `eta_total = eta_attn(~0) + eta_lmhead`, NOT the old
blanket 9.841%.  The lm_head locus is the ONLY open identity cost; stark #365 is measuring
`lmhead_bi_gemm_eta` directly.  The >500 identity budget is the slack the lambda ceiling (520.953,
PPL-only/no-supply-tax) can absorb and stay >=500:  ETA_BUDGET_500 = 1 - 500/LAMBDA_CEIL ~= 4.02%.
The old blanket 9.841% > 4.02% (could NOT fit), so the decomposition is exactly what could open
the door — IF the measured lm_head eta clears the budget:

  measured eta_lmhead <= 4.02% (with eta_attn~0)  ->  identity-compliant config fits the budget
  measured eta_lmhead >  4.02%                     ->  identity tax alone forecloses >500

Route A (supply-side) is now PENDING on BOTH measured inputs and clears only if BOTH gates pass:
route_a reachable  <=>  (sub-int4 PPL-viable -> supply cap rises >500) AND (eta_total <= 4.02%).
We do NOT read stark's branch; we consume only the eta numbers the advisor relayed into this PR.

DEMAND-SIDE route is the NOW-PRIMARY path (advisor HOLD #3, PR #357 review 16:02Z)
----------------------------------------------------------------------------------
The eta-axis (identity-tax reduction route to >500) DEFLATED on two grounds: #373 insufficient +
#375 un-deployable; and the supply-side mixed-precision path is PPL-blocked (lawine #372).  The
now-PRIMARY route to private-500 is DEMAND-SIDE: retrain the drafter (soft-KD reasoning) to raise
acceptance COVERAGE c, which both lifts E[T] (accepted tok/step -> TPS) AND shrinks the dominant
public->private TPS-drift gap (4.295pp).  denken #377 (merged 030uc5mk, independently verified)
SIZED the closer: recommended_retrain_target c* >= 0.9010 (Δcov robust 0.0107 = 35% of #336's
0.031 envelope; central 0.00565 = 18%).  Non-iid is 1.41x pricier than the deprecated #373 iid
model (cov->E[T] slope 11.12 -> 7.91 via the #289 a1=0.7293 first-token cliff + kappa=0.672 int4-ct
transfer).  We CONSUME denken #377's sized output (NOT by reading its branch); the demand route is
HELD pending two hardening inputs:
  - denken #380 : deliverability_survives_conservative / recipe_is_real (the p_deliver~1.0 prior
                  rests on #339's modeled N(0.0385,0.0074) and the kappa transfer).
  - ubel  #379 : coverage->gap split (irreducible vs coverage-addressable pp of the 4.295pp gap);
                 we PULL this split rather than re-derive #377's 0.3914pp/cov elasticity.

Composite verdict (OR over routes; demand-side primary)
-------------------------------------------------------
  private_500 reachable  <=>  (Route A supply-side clears)  OR  (Route B demand-side closes).
The composite HOLDS while any operative route is pending; it stamps True only when a route is
confirmed to close, and False only when BOTH routes are determined dead.

Levers
------
  L_kernel  : kernel-level GEMM / memory-BW improvement.
              On the spec substrate the custom Marlin W4A16 kernel is already incorporated
              into the 481.53 baseline (#354), so L_kernel=1.0x on that path.  FlashInfer is
              slower at batch=1 (#349), so no further kernel lever on the non-spec substrate.

  L_quant(b): sub-int4 quantization Amdahl gain, BW-bound at M=1.  Going int4->b bits shrinks
              the dominant body-GEMM weight-read traffic by b/4:
                L_quant(b) = 1 / (NON_BODY_FRAC + BODY_FRAC * b/4)
              b=4 -> 1.000x (int4, baseline)   b=3 -> 1.308x   b=2 -> 1.892x (int2 ceiling)
              GATED by the MEASURED Gemma PPL at b* (denken #356), NOT literature.

  L_step    : step-overhead shave via CUDA Graphs.  A10G (sm_86, Ampere) ceiling 3-5%
              (H100 measured 20.6%, arXiv 2605.30571v1 Table 3, scaled down for A10G).
              L_step = 1.05x (optimistic), 1.03x (conservative floor).  Fixed, fine (advisor).

Supply cap  : ceiling(b) -- method-independent batched-verify BW floor, a FUNCTION of body
              bits (denken #356 curve; #332's 473.5296 is the b=4 anchor).  As the body
              shrinks the per-step verify BW cost shrinks and the cap rises.

Composite at b*
---------------
  base_lifted(b) = BASELINE_TPS * L_quant(b)            # sub-int4 body lifts the spec base
  precap(b)      = base_lifted(b) * L_kernel * L_step
  tps_eff(b)     = min(precap(b), ceiling(b))           # denken cap binds in the live branch
  clears_500(b)  = tps_eff(b) >= 500

  b=4 (int4, PPL-excluded branch): 481.53*1.0*1.05 = 505.61 precap, ceiling 473.53 -> 473.53 < 500
  b=3 (PPL-viable branch):        481.53*1.308*1.05 = 661.3 precap, ceiling 585  -> 585    > 500

PRIMARY metric  strict_500_composite_reachability_self_test_passes
TEST    metrics tps_max_optimistic_nonspec, tps_max_optimistic_spec,
                strict_500_reachable_via_known_levers (composite OR over routes; None while pending),
                primary_route, eta_axis_deflated, binding_constraint, residual_gap_to_500,
                verdict_pending (composite), verdict_pending_measured_ppl, verdict_pending_identity_eta,
                ppl_flip_threshold, tps_eff_int4_branch, tps_eff_subint4_branch (at b*),
                eta_budget_500, eta_attn_stark363, lmhead_eta_flip_threshold,
                eta_total_verify_locus, identity_clears_500_budget (None while pending),
                demand_route_closes_private_500 (None while pending), demand_route_pending,
                recommended_retrain_target_c, delta_cov_robust(_budget_frac), within_336_budget,
                noniid_price_multiplier, gap_shrink_per_coverage, public_private_gap_pct,
                kappa_int4_ct_transfer, triple_tail_out_of_budget.
"""

from __future__ import annotations
import argparse
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
# Banked constants — all sourced from merged PRs / literature (see provenance).
# --------------------------------------------------------------------------- #

# Substrate baselines (TPS at strict greedy, batch=1, A10G)
TPS_NONSPEC: float = 165.44           # lawine #196: non-spec AR baseline
TPS_SPEC_OFFSHELF_BI: float = 357.32  # wirbel #326: off-shelf spec-decode BI
BASELINE_TPS: float = 481.53          # wirbel #354: custom-kernel-compliant spec baseline (int4)

# Supply cap at int4 — method-independent batched-verify BW floor (denken #332 y5cl0ena)
SUPPLY_CAP_INT4: float = 473.5295953446407   # strict ceiling from #332 (b=4 anchor)
SUPPLY_FLOOR_GEO: float = 0.09103155435261377  # geometric-phi supply floor fraction

# Lambda ceiling (PPL-only: E[T] infinite, no supply tax) from denken #332
LAMBDA_CEIL: float = 520.9527323111674

# --------------------------------------------------------------------------- #
# denken #356 ceiling(b) curve — the supply cap as a FUNCTION of body bit-width.
# Anchors relayed by the advisor into the PR #357 review (2026-06-15); the 4.0-bpw
# point is denken #332's 473.5296.  These are samples of denken's measured/derived
# curve; the published curve supersedes them on the terminal re-run.  We do NOT
# fetch denken's branch — we use only the values the advisor handed into this PR.
# --------------------------------------------------------------------------- #
CEILING_ANCHORS_BPW: dict[float, float] = {
    4.0: SUPPLY_CAP_INT4,  # 473.5296 (denken #332)
    3.5: 523.0,            # advisor-relayed (denken #356)
    3.0: 585.0,            # advisor-relayed (denken #356)
}
B_STAR_DEFAULT: float = 3.0  # advisor's canonical "best sub-int4" demonstration point

# PPL gate
PPL_GATE: float = 2.42
PPL_DEPLOYED: float = 2.3772
PPL_HEADROOM: float = PPL_GATE - PPL_DEPLOYED  # 0.0428 ~ 0.043

# Hardware / roofline constants (denken #344 waterfall, A10G sm_86)
BODY_FRAC: float = 0.943       # fraction of batch=1 step HBM traffic from body GEMM weights
NON_BODY_FRAC: float = 0.057   # 1 - BODY_FRAC
STEP_US: float = 1218.2        # step duration in microseconds (#344)

# Kernel-level lever
ETA_KERNEL_FLOOR: float = 0.0095    # #326 floor vs non-spec
ETA_KERNEL_OFFSHELF: float = 0.3141 # #326 off-shelf spec-decode gain vs non-spec
L_KERNEL_SPEC: float = 1.0          # already incorporated in BASELINE_TPS (#354)

# Step-shave lever (CUDA Graphs, A10G) — fixed (advisor: keep)
L_STEP_OPTIMISTIC: float = 1.05    # 5% overhead elimination ceiling (literature A10G)
L_STEP_FLOOR: float = 1.03         # conservative 3% floor

# Target
TARGET: float = 500.0

# --------------------------------------------------------------------------- #
# Verify-locus IDENTITY tax (advisor HOLD #2, PR #357 review 13:08Z).
# stark #363 (a0oi2esq, MERGED): attention-locus identity tax is FREE (eta~0); a
# fixed-split-k / M-invariant attention GEMM restores byte-exact greedy identity at
# all M in {2,4,8}, best K=8 even faster than the deployed heuristic (eta_ratio 0.9167).
# So verify-locus eta DECOMPOSES as eta_total = eta_attn(~0) + eta_lmhead, superseding
# the old blanket 9.841%.  The lm_head locus is the only open identity cost; stark #365
# measures lmhead_bi_gemm_eta directly (pending -> consumed as --lmhead-eta).
# --------------------------------------------------------------------------- #
ETA_VERIFY_BLANKET: float = 0.09841        # pre-decomposition blanket verify-locus identity tax
ETA_ATTN_STARK363: float = 0.0             # attention-locus identity tax (stark #363, FREE)
ETA_ATTN_RATIO_STARK363: float = 0.9167    # best-K=8 vs deployed-heuristic latency ratio (<1 -> faster)
# >500 identity budget: max verify-locus eta the lambda ceiling can absorb and stay >=500.
ETA_BUDGET_500: float = 1.0 - TARGET / LAMBDA_CEIL   # ~0.04022 (advisor's "4.02% >500 budget")

# --------------------------------------------------------------------------- #
# DEMAND-SIDE coverage closer (denken #377, advisor-relayed PR #357 review 16:02Z).
# The eta-axis (identity-tax reduction route to >500) DEFLATED on two grounds: #373
# insufficient + #375 un-deployable.  The now-PRIMARY route to private-500 is demand-side:
# retrain the drafter (soft-KD reasoning) to raise acceptance COVERAGE c, which both lifts
# E[T] (accepted tok/step -> TPS) AND shrinks the dominant public->private TPS-drift gap.
# denken #377 (merged 030uc5mk, independently verified) replaced the #373 iid coverage model
# with a non-iid, position-correlated, depth-decaying one anchored on #289's per-position
# profile (a1=0.7293 first-token cliff).  Its SIZED result for this integrator:
#   recommended_retrain_target c >= 0.9010, the costed residual closer for private-500.
# We CONSUME denken #377's sized output via the PR thread (NOT by reading denken's branch);
# the demand-side route is HELD pending two hardening inputs: denken #380 (deliverability)
# and ubel #379 (coverage->gap split).
# --------------------------------------------------------------------------- #
RECOMMENDED_RETRAIN_TARGET_C: float = 0.9010   # denken #377 demand-side closer for private-500
DELTA_COV_ROBUST: float = 0.0107               # robust Δcoverage to reach c* (USE THIS, non-iid)
DELTA_COV_CENTRAL: float = 0.00565             # central Δcoverage to reach c*
COV_TO_ET_SLOPE_NONIID: float = 7.91           # E[T] gained per unit coverage (non-iid, #377)
COV_TO_ET_SLOPE_IID: float = 11.12             # old #373 iid slope (deprecated)
NONIID_PRICE_MULTIPLIER: float = COV_TO_ET_SLOPE_IID / COV_TO_ET_SLOPE_NONIID  # ~1.406 (~1.41x pricier)
OLD_373_IID_DELTA_COV: float = 0.004           # DEPRECATED iid figure — do NOT use (1.41x too cheap)
GAP_SHRINK_PER_COVERAGE: float = 0.3914        # pp of public->private gap closed per Δcov unit (#377)
PUBLIC_PRIVATE_GAP_PCT: float = 4.295          # dominant public->private TPS-drift gap (pp)
BUDGET_336_ENVELOPE: float = 0.031             # #336 retrain coverage budget envelope (Δcov)
P_SOFTKD_REASONING_RETRAIN_DELIVERS: float = 1.0  # #377: ~certain a soft-KD retrain delivers c*
KAPPA_INT4_CT_TRANSFER: float = 0.672          # int4-ct transfer of the fp coverage gain (#377)
DELIVERABILITY_339_MEAN: float = 0.0385        # #339 modeled retrain Δcov ~ N(mean, std)
DELIVERABILITY_339_STD: float = 0.0074
PER_POSITION_A1_CLIFF_289: float = 0.7293      # #289 first-token acceptance cliff (non-iid anchor)
# Triple-tail OUT-OF-BUDGET corner (sensitivity band, NOT the central operating point): a
# low-probability simultaneous worst-case (conservative ceiling AND conservative rho AND worst c*)
# that also ignores the gap co-benefit.  denken #377 prices it at 136% of #336's envelope.
TRIPLE_TAIL: dict[str, float] = {
    "conservative_ceiling_tps": 509.07,
    "conservative_rho": 0.8038,
    "worst_c_star": 0.9256,
    "cost_frac_of_336_budget": 1.36,   # 136% -> out of budget in this corner only
}

# --------------------------------------------------------------------------- #
# Sub-int4 PPL LITERATURE PRIOR (Llama-2-7B wikitext-2 baseline ~5.47 PPL).
# NON-AUTHORITATIVE: these deltas are on non-Gemma checkpoints and are used here
# only as a forecast/prior to bracket expectations.  The authoritative gate is
# denken #356's MEASURED Gemma-4-E4B PPL at b* ("measure, don't guess", #319).
# Deltas are INT2-vs-INT4 additional degradation in PPL points (W2A16 or equiv).
# --------------------------------------------------------------------------- #
INT2_PPL_DELTAS: dict[str, dict[str, Any]] = {
    "QuIP#": {
        "delta_ppl_int2": 1.19,
        "arxiv": "2402.04396",
        "note": "best published int2 (incoherence + lattice codebook); int3 delta +0.32",
    },
    "AQLM": {
        "delta_ppl_int2": 1.47,
        "arxiv": "2401.06118",
        "note": "additive quantization LM; multi-codebook int2",
    },
    "QTIP": {
        "delta_ppl_int2": 1.70,
        "arxiv": "2406.11235",
        "note": "quantization with trellises, incoherence, and proxies; int3 delta +0.28",
    },
    "TesseraQ+AWQ": {
        "delta_ppl_int2": 1.35,
        "arxiv": "2410.19103",
        "note": "AWQ + Tessera weight compression; int2 W2A16",
    },
}
INT2_PPL_DELTA_BEST: float = min(v["delta_ppl_int2"] for v in INT2_PPL_DELTAS.values())
INT2_PPL_DELTA_WORST: float = max(v["delta_ppl_int2"] for v in INT2_PPL_DELTAS.values())
INT3_PPL_DELTA_BEST_LIT: float = 0.28  # QuIP#/QTIP int3 (Llama-2-7B wikitext-2)

# Tolerances for self-tests
TOL_EXACT: float = 1e-9
TOL_332: float = 1e-6
TOL_DISPLAY_TPS: float = 5e-3
TOL_PPL: float = 1e-6


# --------------------------------------------------------------------------- #
# Parameterized lever / cap functions of body bit-width b.
# --------------------------------------------------------------------------- #
def l_quant_of_b(b: float) -> float:
    """Amdahl BW-bound speedup of going int4->b bits at M=1.

    Body weight-read traffic scales as b/4; non-body traffic is unchanged.
    L_quant(4)=1.0, L_quant(3)=1.308, L_quant(2)=1.892 (= the old int2 ceiling).
    """
    return 1.0 / (NON_BODY_FRAC + BODY_FRAC * (b / 4.0))


def ceiling_of_b(b: float) -> dict[str, Any]:
    """denken #356 supply-cap ceiling at body bit-width b (piecewise-linear over anchors).

    Monotone increasing as b decreases.  Outside the relayed anchor span [3.0, 4.0]
    we extrapolate from the nearest segment and flag it; b>=4 clamps to the int4 cap.
    """
    anchors = sorted(CEILING_ANCHORS_BPW.items())  # ascending in b
    b_lo, b_hi = anchors[0][0], anchors[-1][0]
    extrapolated = False
    if b >= b_hi:
        # >= int4 bits: clamp to int4 cap (more bits never raises the cap)
        val = CEILING_ANCHORS_BPW[b_hi]
        extrapolated = b > b_hi
    elif b <= b_lo:
        # below the lowest relayed anchor: extrapolate using the lowest segment slope
        (x0, y0), (x1, y1) = anchors[0], anchors[1]
        slope = (y1 - y0) / (x1 - x0)
        val = y0 + slope * (b - x0)
        extrapolated = b < b_lo
    else:
        # bracket and interpolate
        val = None
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= b <= x1:
                t = (b - x0) / (x1 - x0)
                val = y0 + t * (y1 - y0)
                break
        assert val is not None
    return {"bits": b, "ceiling_tps": val, "extrapolated_outside_anchors": extrapolated}


def composite_at_b(b: float, l_step: float) -> dict[str, Any]:
    """Optimistic composite at body bit-width b: min(lever composite, denken ceiling(b))."""
    lq = l_quant_of_b(b)
    base_lifted = BASELINE_TPS * lq
    precap = base_lifted * L_KERNEL_SPEC * l_step
    cap_info = ceiling_of_b(b)
    cap = cap_info["ceiling_tps"]
    eff = min(precap, cap)
    cap_binds = cap <= precap
    return {
        "bits": b,
        "l_quant": lq,
        "base_lifted_tps": base_lifted,
        "l_step": l_step,
        "precap_tps": precap,
        "ceiling_tps": cap,
        "ceiling_extrapolated": cap_info["extrapolated_outside_anchors"],
        "tps_eff": eff,
        "cap_binds": cap_binds,
        "clears_500": eff >= TARGET,
        "margin_to_500": eff - TARGET,  # >0 clears, <0 short
        "binding_constraint": (
            f"supply_cap_ceiling_at_b={b:g}bpw" if cap_binds else f"lever_composite_at_b={b:g}bpw"
        ),
    }


# --------------------------------------------------------------------------- #
# Verify-locus identity tax: eta_total = eta_attn(~0, stark #363) + eta_lmhead (stark #365).
# --------------------------------------------------------------------------- #
def identity_locus_analysis(lmhead_eta: float | None) -> dict[str, Any]:
    """Decompose the strict-identity verify tax and test it against the 4.02% >500 budget.

    stark #363 measured the attention-locus tax as FREE (eta_attn~0); the lm_head locus is the
    only open identity cost.  The gate is whether eta_total = eta_attn + eta_lmhead fits the
    >500 budget ETA_BUDGET_500 = 1 - 500/LAMBDA_CEIL (~4.02%).  Pending until stark #365's
    measured lmhead_bi_gemm_eta lands (lmhead_eta is None).
    """
    eta_attn = ETA_ATTN_STARK363
    budget = ETA_BUDGET_500
    pending = lmhead_eta is None
    eta_total = None if pending else eta_attn + lmhead_eta
    clears = None if pending else (eta_total <= budget)
    lam_with_identity = None if pending else LAMBDA_CEIL * (1.0 - eta_total)
    return {
        "eta_attn_stark363": eta_attn,
        "eta_attn_ratio_stark363": ETA_ATTN_RATIO_STARK363,
        "eta_blanket_predecomp": ETA_VERIFY_BLANKET,
        "lmhead_eta_measured": lmhead_eta,   # stark #365 (pending -> None)
        "eta_total_verify_locus": eta_total,
        "eta_budget_500": budget,
        "eta_budget_500_derivation": "1 - TARGET/LAMBDA_CEIL",
        "lmhead_eta_flip_threshold": budget - eta_attn,   # measured lm_head eta at which the gate flips
        "identity_pending": pending,
        "identity_clears_500_budget": clears,
        "lambda_ceiling_with_identity_tax": lam_with_identity,
        "blanket_would_clear_budget": ETA_VERIFY_BLANKET <= budget,  # False: 9.841% > 4.02%
        "decomposition_note": (
            "stark #363 (a0oi2esq, MERGED): attention-locus identity tax FREE (eta~0, ratio 0.9167, "
            "best K=8, M-invariant fixed-split-k). Verify-locus eta = attn(~0) + lm_head; the blanket "
            "9.841% (> 4.02% budget) is superseded. stark #365 measures lmhead_bi_gemm_eta (pending)."
        ),
    }


# --------------------------------------------------------------------------- #
# DEMAND-SIDE route (denken #377): raise drafter acceptance coverage c to the sized
# closer c* >= 0.9010.  This is the NOW-PRIMARY route to private-500 (the eta-axis
# deflated: #373 insufficient + #375 un-deployable).  We consume denken #377's SIZED
# output; the route is HELD pending denken #380 (deliverability) + ubel #379 (gap split).
# We do NOT re-derive denken #377's coverage->E[T]->private-500 sizing or ubel's gap split.
# --------------------------------------------------------------------------- #
def demand_side_route_analysis(deliverability: str = "pending",
                               gap_addressable_pp: float | None = None) -> dict[str, Any]:
    """Integrate denken #377's demand-side coverage closer; pending #380 + #379.

    `deliverability` (denken #380 `deliverability_survives_conservative`): one of
        "pending" | "survives" | "fails".
    `gap_addressable_pp` (ubel #379 split): pp of the 4.295pp public->private gap that is
        coverage-ADDRESSABLE (vs irreducible).  None -> the gap split is still PENDING.

    The route CLOSES private-500 iff denken #377's in-budget closer is BOTH deliverable
    (denken #380 survives the conservative tail) AND backed by a positive coverage->gap
    channel (ubel #379 finds part of the gap coverage-addressable).  denken #377 already
    SIZED that reaching c* closes the residual; we do not re-derive that arithmetic.
    """
    # Budget check against #336's +0.031 envelope (the robust Δcov is the one to use).
    robust_budget_frac = DELTA_COV_ROBUST / BUDGET_336_ENVELOPE      # ~0.345 (35%)
    central_budget_frac = DELTA_COV_CENTRAL / BUDGET_336_ENVELOPE    # ~0.182 (18%)
    within_336_budget = robust_budget_frac < 1.0                     # True (and central too)

    # Implied baseline coverage (c* minus the Δcov needed); the robust path assumes a
    # lower starting coverage, so its implied baseline is below the central one.
    baseline_cov_robust = RECOMMENDED_RETRAIN_TARGET_C - DELTA_COV_ROBUST    # 0.8903
    baseline_cov_central = RECOMMENDED_RETRAIN_TARGET_C - DELTA_COV_CENTRAL  # 0.89535

    # Deliverability prior (central): #339 models delivered Δcov ~ N(0.0385, 0.0074); after
    # the int4-ct transfer kappa=0.672 the delivered Δcov is still well above the required
    # robust 0.0107 -> modeled p_deliver ~= 1.0.  denken #380 hardens this against the tail.
    delivered_after_kappa = DELIVERABILITY_339_MEAN * KAPPA_INT4_CT_TRANSFER  # ~0.02587
    deliverability_margin = delivered_after_kappa - DELTA_COV_ROBUST          # ~0.0152 (>0)
    modeled_p_deliver = P_SOFTKD_REASONING_RETRAIN_DELIVERS                   # ~1.0 (prior)

    # Triple-tail OUT-OF-BUDGET corner: a sensitivity band, NOT the central operating point.
    tt = dict(TRIPLE_TAIL)
    tt["out_of_budget"] = tt["cost_frac_of_336_budget"] > 1.0  # True (136%)
    tt["c_star_above_recommended"] = tt["worst_c_star"] > RECOMMENDED_RETRAIN_TARGET_C  # True
    tt["is_sensitivity_band_not_central"] = True

    # Gap split (ubel #379) — authoritative; we PULL it, never re-derive.
    gap_pending = gap_addressable_pp is None
    gap_irreducible_pp = None if gap_pending else (PUBLIC_PRIVATE_GAP_PCT - gap_addressable_pp)
    gap_channel_live = None if gap_pending else (gap_addressable_pp > 0.0)

    # Route verdict: pending until BOTH hardening inputs land.
    deliver_pending = deliverability == "pending"
    route_pending = deliver_pending or gap_pending
    deliver_survives = deliverability == "survives"
    if route_pending:
        route_closes: bool | None = None
        binding = "PENDING_denken380_deliverability_and_or_ubel379_gap_split"
    elif not deliver_survives:
        route_closes = False
        binding = "deliverability_fails_conservative_denken380"
    elif not gap_channel_live:
        route_closes = False
        binding = "gap_irreducible_no_coverage_channel_ubel379"
    else:
        route_closes = True
        binding = "demand_side_closer_in_budget_deliverable_gap_addressable"

    return {
        "route": "demand_side_coverage_denken377",
        "is_primary_route": True,
        "eta_axis_deflated": True,
        "eta_axis_deflation_grounds": "#373 insufficient + #375 un-deployable",
        # sized closer (denken #377, consumed via thread)
        "recommended_retrain_target_c": RECOMMENDED_RETRAIN_TARGET_C,
        "delta_cov_robust": DELTA_COV_ROBUST,
        "delta_cov_central": DELTA_COV_CENTRAL,
        "baseline_cov_robust": baseline_cov_robust,
        "baseline_cov_central": baseline_cov_central,
        # non-iid pricing
        "cov_to_et_slope_noniid": COV_TO_ET_SLOPE_NONIID,
        "cov_to_et_slope_iid": COV_TO_ET_SLOPE_IID,
        "noniid_price_multiplier": NONIID_PRICE_MULTIPLIER,
        "deprecated_iid_delta_cov_373": OLD_373_IID_DELTA_COV,
        "a1_first_token_cliff_289": PER_POSITION_A1_CLIFF_289,
        # budget
        "budget_336_envelope": BUDGET_336_ENVELOPE,
        "delta_cov_robust_budget_frac": robust_budget_frac,
        "delta_cov_central_budget_frac": central_budget_frac,
        "within_336_budget": within_336_budget,
        "triple_tail_corner": tt,
        # deliverability (pending denken #380)
        "deliverability_input_denken380": deliverability,
        "deliverability_pending": deliver_pending,
        "modeled_p_deliver": modeled_p_deliver,
        "kappa_int4_ct_transfer": KAPPA_INT4_CT_TRANSFER,
        "deliverability_339_mean": DELIVERABILITY_339_MEAN,
        "deliverability_339_std": DELIVERABILITY_339_STD,
        "delivered_after_kappa": delivered_after_kappa,
        "deliverability_margin": deliverability_margin,
        # gap channel (pending ubel #379)
        "public_private_gap_pct": PUBLIC_PRIVATE_GAP_PCT,
        "gap_shrink_per_coverage": GAP_SHRINK_PER_COVERAGE,
        "gap_coverage_addressable_pp_ubel379": gap_addressable_pp,
        "gap_irreducible_pp": gap_irreducible_pp,
        "gap_channel_live": gap_channel_live,
        "gap_split_pending": gap_pending,
        # verdict
        "route_pending": route_pending,
        "route_closes_private_500": route_closes,
        "binding_constraint": binding,
        "note": (
            "denken #377 (030uc5mk, merged, independently verified) SIZED c* >= 0.9010 as the "
            "costed demand-side residual closer for private-500 (Δcov robust 0.0107 = 35% of #336's "
            "0.031 envelope; central 0.00565 = 18%). Non-iid is 1.41x pricier than the deprecated "
            "#373 iid model (slope 11.12->7.91 via the a1=0.7293 cliff + kappa=0.672 int4-ct transfer). "
            "The route is the NOW-PRIMARY path (eta-axis deflated: #373 insufficient + #375 "
            "un-deployable) but HELD pending denken #380 (deliverability survives conservative) + "
            "ubel #379 (coverage->gap split). We pull ubel's split rather than re-deriving the "
            "0.3914pp/unit gap elasticity (#377's least-grounded input)."
        ),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Deliverable 1: lever analysis — what each lever can and cannot contribute.
# --------------------------------------------------------------------------- #
def deliverable1_lever_analysis(b_star: float) -> dict[str, Any]:
    """Characterise each lever; L_quant and the cap are now functions of b*."""
    lq_table = {f"b={b:g}": l_quant_of_b(b) for b in (4.0, 3.5, 3.0, 2.0)}
    ceil_table = {f"b={b:g}": ceiling_of_b(b)["ceiling_tps"] for b in (4.0, 3.5, 3.0, 2.0)}
    return {
        "l_kernel_spec": L_KERNEL_SPEC,
        "l_kernel_note": "custom Marlin W4A16 kernel already in BASELINE_TPS #354; L_kernel=1.0 on spec substrate",
        "l_quant_formula": "L_quant(b) = 1 / (NON_BODY_FRAC + BODY_FRAC * b/4)",
        "l_quant_table": lq_table,
        "l_quant_at_b_star": l_quant_of_b(b_star),
        "l_quant_int2_ceiling": l_quant_of_b(2.0),
        "l_quant_gated_by": "MEASURED Gemma PPL at b* (denken #356), NOT literature",
        "ceiling_formula": "denken #356 piecewise-linear over anchors {4.0:473.53, 3.5:523, 3.0:585}",
        "ceiling_table": ceil_table,
        "ceiling_at_b_star": ceiling_of_b(b_star)["ceiling_tps"],
        "ceiling_anchors_advisor_relayed": dict(sorted(CEILING_ANCHORS_BPW.items())),
        "l_step_optimistic": L_STEP_OPTIMISTIC,
        "l_step_floor": L_STEP_FLOOR,
        "l_step_source": "CUDA Graphs A10G ceiling 3-5%; H100 20.6% arXiv 2605.30571v1 Table 3",
        "flashinfer_excluded": True,
        "flashinfer_note": "FlashInfer batch-1 SDPA 36.05us/layer vs FlashInfer 48.20us/layer (#349); slower at batch=1",
    }


# --------------------------------------------------------------------------- #
# Deliverable 2: sub-int4 PPL FORECAST (literature prior) — NOT the verdict gate.
# --------------------------------------------------------------------------- #
def deliverable2_ppl_forecast() -> dict[str, Any]:
    """Literature prior for sub-int4 PPL on Llama-2 lineage.

    NON-AUTHORITATIVE.  The advisor flagged that closing L_quant on transplanted
    literature PPL violates "measure, don't guess" (#319 11:27Z).  We report these
    as a FORECAST only; the verdict consumes denken #356's MEASURED Gemma PPL.
    """
    forecast: list[dict[str, Any]] = []
    for method, info in INT2_PPL_DELTAS.items():
        delta = info["delta_ppl_int2"]
        ppl_result = PPL_DEPLOYED + delta
        forecast.append({
            "method": method,
            "arxiv": info["arxiv"],
            "delta_ppl_int2": delta,
            "ppl_result_if_gemma_matched_llama": ppl_result,
            "would_violate_if_transplanted": ppl_result > PPL_GATE,
            "headroom_ratio": delta / PPL_HEADROOM,
        })
    all_would_violate = all(f["would_violate_if_transplanted"] for f in forecast)
    best_entry = min(forecast, key=lambda x: x["delta_ppl_int2"])

    int3_ppl = PPL_DEPLOYED + INT3_PPL_DELTA_BEST_LIT
    return {
        "authoritative": False,
        "gate_source": "denken #356 MEASURED Gemma-4-E4B ppl_at_best_sub_int4_bits (pending)",
        "per_method_forecast": forecast,
        "literature_int2_all_would_violate_if_transplanted": all_would_violate,
        "best_int2_method": best_entry["method"],
        "best_int2_delta": best_entry["delta_ppl_int2"],
        "best_int2_headroom_ratio": best_entry["headroom_ratio"],
        "int3_delta_best_lit": INT3_PPL_DELTA_BEST_LIT,
        "int3_ppl_result_if_transplanted": int3_ppl,
        "int3_would_violate_if_transplanted": int3_ppl > PPL_GATE,
        "int3_overshoot_ratio_lit": INT3_PPL_DELTA_BEST_LIT / PPL_HEADROOM,
        "caveat": (
            "Llama-2-lineage deltas; Gemma-4-E4B (GQA/shared-KV, MLP gating) may scale "
            "differently. Used as a prior to bracket expectations, NOT to close L_quant. "
            "The headroom is only 0.043 PPL (~1.8% rel), so the literature prior LEANS toward "
            "violation, but the measured Gemma PPL at b* is what decides the verdict."
        ),
    }


# --------------------------------------------------------------------------- #
# Deliverable 3: composite TPS ceiling, parameterized on b*, both branches.
# --------------------------------------------------------------------------- #
def deliverable3_composite_tps(b_star: float) -> dict[str, Any]:
    """Composite at the int4 substrate (PPL-excluded branch) and at b* (PPL-viable branch)."""
    # Non-spec substrate (base = 165.44); L_kernel=1 (FlashInfer excluded #349); L_quant=1 here
    # (sub-int4 on the non-spec AR substrate is the same PPL story; the spec substrate dominates).
    tps_nonspec_optimistic = TPS_NONSPEC * 1.0 * 1.0 * L_STEP_OPTIMISTIC
    tps_nonspec_floor = TPS_NONSPEC * 1.0 * 1.0 * L_STEP_FLOOR

    # int4 substrate (b=4): the PPL-EXCLUDED branch — sub-int4 not viable -> stuck at int4 cap.
    int4_opt = composite_at_b(4.0, L_STEP_OPTIMISTIC)
    int4_floor = composite_at_b(4.0, L_STEP_FLOOR)

    # sub-int4 substrate (b=b*): the PPL-VIABLE branch — cap rises to denken ceiling(b*).
    sub_opt = composite_at_b(b_star, L_STEP_OPTIMISTIC)
    sub_floor = composite_at_b(b_star, L_STEP_FLOOR)

    # tps_max_optimistic_spec retains its original meaning: the int4 lever composite pre-cap.
    tps_max_optimistic_spec = int4_opt["precap_tps"]

    return {
        # non-spec
        "base_nonspec": TPS_NONSPEC,
        "tps_max_optimistic_nonspec": tps_nonspec_optimistic,
        "tps_max_floor_nonspec": tps_nonspec_floor,
        # spec int4 (PPL-excluded branch)
        "base_spec_int4": BASELINE_TPS,
        "tps_max_optimistic_spec": tps_max_optimistic_spec,  # 505.61 pre-cap
        "int4_branch": int4_opt,
        "int4_branch_floor": int4_floor,
        # sub-int4 (PPL-viable branch at b*)
        "b_star": b_star,
        "subint4_branch": sub_opt,
        "subint4_branch_floor": sub_floor,
        # off-shelf spec (#326) for the ladder
        "base_offshelf": TPS_SPEC_OFFSHELF_BI,
        "composite_formula": "tps_eff(b) = min(BASELINE*L_quant(b)*L_kernel*L_step, ceiling(b))",
        "note": (
            f"int4 branch (PPL-excluded): precap {int4_opt['precap_tps']:.2f}, cap "
            f"{int4_opt['ceiling_tps']:.4f} -> eff {int4_opt['tps_eff']:.4f} "
            f"({'CLEARS' if int4_opt['clears_500'] else 'SHORT'} 500). "
            f"sub-int4 branch (PPL-viable, b*={b_star:g}): precap {sub_opt['precap_tps']:.2f}, cap "
            f"{sub_opt['ceiling_tps']:.2f} -> eff {sub_opt['tps_eff']:.2f} "
            f"({'CLEARS' if sub_opt['clears_500'] else 'SHORT'} 500)."
        ),
    }


# --------------------------------------------------------------------------- #
# Deliverable 4: verdict as a function of the MEASURED PPL at b* (pending-aware).
# --------------------------------------------------------------------------- #
def verdict_given_ppl(measured_ppl: float | None, b_star: float,
                      lmhead_eta: float | None = None) -> dict[str, Any]:
    """Resolve the composite verdict from TWO measured gates; pending if EITHER is missing.

    Gate 1 (denken #356 measured Gemma PPL at b*): PPL-viable -> sub-int4 substrate -> supply cap
            rises to ceiling(b*) > 500.  PPL-excluded -> int4 substrate -> cap 473.53 < 500.
    Gate 2 (stark #365 measured lmhead_bi_gemm_eta): eta_total = eta_attn(~0, stark #363) +
            eta_lmhead must clear the 4.02% >500 budget, else the strict-identity verify tax alone
            forecloses >500.
    strict_500 reachable <=> BOTH gates pass.
    """
    int4 = composite_at_b(4.0, L_STEP_OPTIMISTIC)        # PPL-excluded substrate
    sub = composite_at_b(b_star, L_STEP_OPTIMISTIC)      # PPL-viable substrate at b*
    ident = identity_locus_analysis(lmhead_eta)

    pending_ppl = measured_ppl is None
    pending_identity = lmhead_eta is None
    pending = pending_ppl or pending_identity

    ppl_viable = None if pending_ppl else (measured_ppl <= PPL_GATE)
    identity_clears = ident["identity_clears_500_budget"]      # None if pending_identity
    eta_total = ident["eta_total_verify_locus"]
    lam_with_id = ident["lambda_ceiling_with_identity_tax"]

    branch_violate = {
        "label": "measured_ppl_gt_gate",
        "ppl_viable": False,
        "substrate_bits": 4.0,
        "tps_eff": int4["tps_eff"],
        "reachable": int4["clears_500"],          # False (473.53 < 500)
        "residual_gap_to_500": TARGET - int4["tps_eff"],
        "binding_constraint": "supply_cap_int4_473p53_ppl_excludes_sub_int4",
    }
    branch_viable = {                              # supply-side only (identity gate applied separately)
        "label": "measured_ppl_le_gate_supply_side",
        "ppl_viable": True,
        "substrate_bits": b_star,
        "tps_eff": sub["tps_eff"],
        "reachable": sub["clears_500"],           # supply-side True at b*=3 (585 > 500)
        "residual_gap_to_500": TARGET - sub["tps_eff"],   # negative = margin above 500
        "binding_constraint": (
            f"supply_cap_ceiling_at_b={b_star:g}bpw" if sub["cap_binds"]
            else f"lever_composite_at_b={b_star:g}bpw"
        ),
    }
    branch_identity_blocked = {                    # PPL viable, but identity tax overruns the budget
        "label": "ppl_viable_identity_tax_exceeds_budget",
        "reachable": False,
        "eta_total_verify_locus": eta_total,
        "lambda_ceiling_with_identity_tax": lam_with_id,
        "residual_gap_to_500": (None if lam_with_id is None else TARGET - lam_with_id),
        "binding_constraint": "lmhead_identity_tax_exceeds_4p02pct_budget",
    }

    # Combined verdict: PPL gate first (it governs the substrate), then the identity gate.
    if pending:
        reachable: bool | None = None
        binding = "PENDING_measured_inputs"
        residual = branch_violate["residual_gap_to_500"]
    elif not ppl_viable:
        reachable = False
        binding = branch_violate["binding_constraint"]
        residual = branch_violate["residual_gap_to_500"]
    elif not identity_clears:
        reachable = False
        binding = branch_identity_blocked["binding_constraint"]
        residual = branch_identity_blocked["residual_gap_to_500"]
    else:
        reachable = bool(sub["clears_500"])       # both gates clear -> supply-side margin governs
        binding = f"reachable_subint4_b{b_star:g}_and_identity_clear__cap_{sub['ceiling_tps']:.0f}"
        residual = branch_viable["residual_gap_to_500"]

    pending_inputs = [s for s, miss in (
        ("denken#356_ppl_at_b_star", pending_ppl),
        ("stark#365_lmhead_bi_gemm_eta", pending_identity),
    ) if miss]

    if pending:
        verdict_text = (
            f"PENDING measured input(s): {', '.join(pending_inputs)} — composite is a 2-gate fork "
            f"(PPL@b* x lm_head identity eta); both gates must clear for >500."
        )
    elif reachable:
        verdict_text = (
            f"strict_500 REACHABLE via known levers at b*={b_star:g}bpw: measured PPL {measured_ppl:.4f} "
            f"<= {PPL_GATE} -> sub-int4 LIVE -> cap rises to {sub['ceiling_tps']:.2f} (eff {sub['tps_eff']:.2f} "
            f"TPS, +{sub['tps_eff']-TARGET:.2f}) AND lm_head identity eta_total {eta_total:.4f} <= "
            f"{ETA_BUDGET_500:.4f} budget. Approval-gated a10g candidate (#319)."
        )
    elif not ppl_viable:
        verdict_text = (
            f"strict_500 NOT reachable: measured PPL {measured_ppl:.4f} > {PPL_GATE} -> sub-int4 excluded "
            f"-> L_quant=1.0 -> int4 supply cap {int4['tps_eff']:.4f} TPS binds, residual gap "
            f"{TARGET-int4['tps_eff']:.4f} TPS. Genuinely-new-method problem (~3x from 165.44 floor)."
        )
    else:  # ppl viable, identity blocked
        verdict_text = (
            f"strict_500 NOT reachable: PPL {measured_ppl:.4f} <= {PPL_GATE} (sub-int4 LIVE, supply OK) BUT "
            f"lm_head identity eta_total {eta_total:.4f} > {ETA_BUDGET_500:.4f} budget -> identity-taxed "
            f"lambda ceiling {lam_with_id:.2f} TPS < 500 (gap {TARGET-lam_with_id:.2f}). The strict-identity "
            f"verify tax alone forecloses >500."
        )

    return {
        "measured_ppl_at_b_star": measured_ppl,
        "lmhead_eta_measured": lmhead_eta,
        "b_star": b_star,
        "ppl_flip_threshold": PPL_GATE,
        "lmhead_eta_flip_threshold": ident["lmhead_eta_flip_threshold"],
        "eta_budget_500": ETA_BUDGET_500,
        "eta_total_verify_locus": eta_total,
        # pending flags
        "verdict_pending_measured_ppl": pending_ppl,        # legacy key (PPL-specific)
        "verdict_pending_identity_eta": pending_identity,
        "verdict_pending": pending,
        "pending_inputs": pending_inputs,
        # gates
        "ppl_viable": ppl_viable,
        "identity_clears_500_budget": identity_clears,
        "strict_500_reachable_via_known_levers": reachable,
        "binding_constraint": binding,
        "residual_gap_to_500": residual,
        # fork branches
        "branch_ppl_violates_gate": branch_violate,
        "branch_ppl_viable": branch_viable,
        "branch_ppl_viable_identity_blocked": branch_identity_blocked,
        "identity": ident,
        "flip_explanation": (
            f"Two coupled gates at b*={b_star:g}bpw (denken ceiling {sub['ceiling_tps']:.2f} >= 500): "
            f"(1) PPL gate — measured Gemma PPL <= {PPL_GATE} opens sub-int4 (supply cap {sub['ceiling_tps']:.2f}); "
            f"PPL > {PPL_GATE} -> int4 cap {int4['tps_eff']:.4f} NO-GO (gap {TARGET-int4['tps_eff']:.4f}). "
            f"(2) identity gate — eta_total = eta_attn({ETA_ATTN_STARK363:.3f}, stark #363 FREE) + "
            f"lm_head; clears iff <= {ETA_BUDGET_500:.4f} (flip at lm_head eta {ident['lmhead_eta_flip_threshold']:.4f}). "
            f"The blanket 9.841% would NOT have fit; the decomposition is what could open the door."
        ),
        "verdict_text": verdict_text,
    }


# --------------------------------------------------------------------------- #
# Composite verdict: OR over routes (supply-side sub-int4 + demand-side coverage).
# Demand-side (denken #377) is the NOW-PRIMARY route; the supply-side mixed-precision
# path is PPL-blocked (lawine #372) and the eta-axis deflated (#373/#375), so Route A
# leans dead.  private-500 reachable <=> (Route A clears) OR (Route B closes).
# --------------------------------------------------------------------------- #
def composite_verdict(measured_ppl: float | None, b_star: float,
                      lmhead_eta: float | None = None,
                      deliverability: str = "pending",
                      gap_addressable_pp: float | None = None) -> dict[str, Any]:
    """Combine Route A (supply-side PPL+identity) and Route B (demand-side coverage), OR-coupled."""
    route_a = verdict_given_ppl(measured_ppl, b_star, lmhead_eta)          # supply-side + identity
    route_b = demand_side_route_analysis(deliverability, gap_addressable_pp)  # demand-side (primary)

    a_reachable = route_a["strict_500_reachable_via_known_levers"]   # True / False / None(pending)
    b_closes = route_b["route_closes_private_500"]                   # True / False / None(pending)

    # OR-combine, pending-aware: a pending route can still flip the verdict to True, so we
    # only stamp False when BOTH routes are determined False; only stamp True when EITHER is True.
    known_true = (a_reachable is True) or (b_closes is True)
    any_pending = (a_reachable is None) or (b_closes is None)
    if known_true:
        reachable: bool | None = True
    elif any_pending:
        reachable = None
    else:
        reachable = False
    pending = reachable is None

    pending_inputs = list(route_a["pending_inputs"])  # denken#356 ppl, stark#365 eta (Route A)
    if route_b["deliverability_pending"]:
        pending_inputs.append("denken#380_deliverability_survives_conservative")
    if route_b["gap_split_pending"]:
        pending_inputs.append("ubel#379_coverage_to_gap_split")

    if reachable is True:
        via = ("demand_side_coverage_denken377" if b_closes is True
               else f"supply_side_subint4_b{b_star:g}")
        binding = f"private_500_REACHABLE_via_{via}"
    elif reachable is False:
        binding = "both_routes_dead__supply_ppl_blocked_lawine372__demand_fails_380_or_379"
    else:
        binding = f"PENDING__primary_route_demand_side__inputs={','.join(pending_inputs)}"

    if pending:
        verdict_text = (
            "private-500 reachability HELD — multi-route fork, demand-side (denken #377) is the "
            f"NOW-PRIMARY route (eta-axis deflated: {route_b['eta_axis_deflation_grounds']}). "
            f"Route B (demand-side coverage c* >= {RECOMMENDED_RETRAIN_TARGET_C}): "
            f"{'PENDING ' + route_b['binding_constraint'] if route_b['route_pending'] else route_b['binding_constraint']}. "
            f"Route A (supply-side sub-int4 + identity): "
            f"{'PENDING' if route_a['verdict_pending'] else route_a['binding_constraint']} "
            "(PPL-blocked per lawine #372 / eta-axis deflated -> leans dead). "
            f"Pending inputs: {pending_inputs}."
        )
    elif reachable:
        verdict_text = (
            f"private-500 REACHABLE: {binding}. Demand-side closer c* >= {RECOMMENDED_RETRAIN_TARGET_C} "
            f"({route_b['delta_cov_robust_budget_frac']*100:.0f}% of #336 budget) closes the residual; "
            "flag as approval-gated a10g candidate (#319)."
        )
    else:
        verdict_text = (
            "private-500 NOT reachable via known levers: supply-side PPL-blocked (lawine #372) + "
            "eta-axis deflated (#373/#375), and the demand-side coverage route failed its hardening "
            "(denken #380 deliverability or ubel #379 gap-split). Genuinely-new-method problem."
        )

    return {
        "primary_route": "demand_side_coverage_denken377",
        "eta_axis_deflated": True,
        "private_500_reachable_via_known_levers": reachable,
        "verdict_pending": pending,
        "pending_inputs": pending_inputs,
        "binding_constraint": binding,
        "verdict_text": verdict_text,
        "route_a_supply_side": {
            "reachable": a_reachable,
            "pending": route_a["verdict_pending"],
            "binding": route_a["binding_constraint"],
            "leans_dead_reason": "supply-side mixed-precision PPL-blocked (lawine #372); eta-axis deflated (#373/#375)",
        },
        "route_b_demand_side": {
            "closes_private_500": b_closes,
            "pending": route_b["route_pending"],
            "binding": route_b["binding_constraint"],
            "is_primary": True,
        },
        "route_a_detail": route_a,
        "route_b_detail": route_b,
    }


# --------------------------------------------------------------------------- #
# Deliverable 5: caveats.
# --------------------------------------------------------------------------- #
def deliverable5_caveats(b_star: float) -> dict[str, Any]:
    return {
        "caveats": [
            "COMPOSITE VERDICT IS MULTI-ROUTE and HELD. The NOW-PRIMARY route is demand-side "
            "(denken #377 coverage closer); the eta-axis deflated (#373 insufficient + #375 "
            "un-deployable) and the supply-side mixed-precision path is PPL-blocked (lawine #372). "
            "private-500 reachable <=> (Route A supply-side clears) OR (Route B demand-side closes). "
            "All cross-PR numbers are consumed via the PR thread, NOT by reading other branches.",
            "COUPLING ASSUMPTION TO CONFIRM (flagged for advisor): I model the routes as OR-combined "
            "(reachable if EITHER closes) with demand-side primary, and I treat denken #377's c*>=0.9010 "
            "as ALREADY-SIZED to close the private-500 residual (I do not re-derive coverage->E[T]->"
            "private-500). The demand route closes iff BOTH harden: denken #380 deliverability survives "
            "conservative AND ubel #379 finds a positive coverage-addressable gap. If instead the routes "
            "are AND-coupled, or the demand closer must be re-priced against a specific residual number, "
            "tell me and I will re-wire the combiner.",
            "GAP CHANNEL NOT RE-DERIVED: denken #377's 0.3914pp/coverage gap-elasticity is its "
            "least-grounded input; per the advisor I PULL ubel #379's coverage->gap split "
            "(--gap-coverage-addressable-pp) rather than computing a gap-closure number myself. Until it "
            "lands the gap channel (and thus the demand route) stays PENDING.",
            "DELIVERABILITY IS A PRIOR, NOT BANKED: p_deliver~1.0 rests on #339's modeled "
            "N(0.0385,0.0074) and the kappa=0.672 int4-ct transfer (delivered-after-kappa ~0.0259 still "
            "exceeds the required robust 0.0107). denken #380 must confirm recipe_is_real / "
            "deliverability_survives_conservative before the target is banked hard.",
            "The triple-tail corner (conservative ceiling 509.07 AND conservative rho 0.8038 AND worst "
            "c*=0.9256 = 136% of #336's budget) is a low-probability simultaneous worst-case that also "
            "ignores the gap co-benefit; it is a SENSITIVITY BAND, not the central operating point.",
            "ROUTE A still PENDING two measured inputs consumed via the PR thread: denken #356's "
            "Gemma-4-E4B ppl_at_best_sub_int4_bits at b*, AND stark #365's lmhead_bi_gemm_eta. We do NOT "
            "re-run either GPU eval (denken owns the single measured-PPL gate, stark owns the lm_head "
            "identity gate; each runs once). Route A leans dead (PPL-blocked + eta deflated).",
            "Identity gate: stark #363 (MERGED) measured the attention-locus identity tax as FREE "
            "(eta~0, ratio 0.9167, best K=8), so verify-locus eta = eta_attn(~0) + eta_lmhead. The "
            ">500 budget ETA_BUDGET_500 = 1 - 500/LAMBDA_CEIL ~= 4.02% is the slack the lambda ceiling "
            "(520.953, PPL-only/no-supply-tax) can absorb and stay >=500. We evaluate the identity gate "
            "against THAT ceiling as relayed; the precise coupling of eta to the sub-int4 supply "
            "substrate is stark/denken's to confirm. eta_attn=0 is an upper-bound-optimistic read of "
            "stark #363's ratio<1 result (best K=8 was actually faster than the deployed heuristic).",
            "ceiling(b) anchors {4.0:473.53, 3.5:523, 3.0:585} are advisor-relayed samples of "
            "denken #356's curve; the published curve supersedes them on the terminal re-run. "
            "Piecewise-linear interpolation between anchors; extrapolation below 3.0 bpw is flagged.",
            "L_quant(b) is a batch=1 BW-bound Amdahl model on BODY_FRAC=0.943 (#344). It assumes the "
            "body GEMM read traffic scales linearly with bits and the non-body fraction is fixed; "
            "real sub-int4 kernels carry dequant overhead that would lower the realized gain.",
            "Sub-int4 also needs a kernel: Marlin W4A16 (arXiv 2408.11743) is 4-bit only; a viable "
            "sub-int4 path requires a compatible low-bit kernel (GPTQ-style W3/W2 or codebook). The "
            "ceiling(b) curve presumes such a kernel exists at the stated overhead.",
            "L_step CUDA-Graphs ceiling 3-5% is an A10G literature estimate; actual benefit depends "
            "on graph capture overhead and the model call graph and may be lower than 3%.",
            "All numbers are for strict greedy token-identity (argmax bit-identical). Relaxing to "
            "approximate speculative decoding is a different research question outside this scope.",
            f"The PPL-viable branch is evaluated at b*={b_star:g}bpw; if denken's best viable bits "
            "differ, re-evaluate ceiling(b*) and L_quant(b*) at that bit-width.",
        ],
        "assumptions": [
            "BASELINE_TPS=481.53 is the current best strict-compliant serve point at int4 (#354).",
            "ceiling(4.0)=473.5295953446407 is method-independent (#332 geometric-phi supply floor).",
            "Arithmetic intensity at M=1 is 4.0, well below ridge point 208.3 (pure BW-bound).",
            "BODY_FRAC=0.943 reflects batch=1 HBM traffic decomposition (#344 waterfall).",
        ],
    }


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, d4: dict, d5: dict, d6: dict, d7: dict,
               b_star: float) -> dict[str, Any]:
    int4 = d3["int4_branch"]
    sub = d3["subint4_branch"]

    # a: L_quant(2.0) reproduces the old int2 Amdahl ceiling (~1.892x)
    a_lquant_int2_reproduces = abs(l_quant_of_b(2.0) - 1.0 / (NON_BODY_FRAC + BODY_FRAC / 2.0)) < TOL_EXACT

    # b: L_quant(4.0) == 1.0 (int4 baseline is the no-op point)
    b_lquant_int4_unit = abs(l_quant_of_b(4.0) - 1.0) < TOL_EXACT

    # c: L_quant monotone increasing as bits decrease
    c_lquant_monotone = l_quant_of_b(2.0) > l_quant_of_b(3.0) > l_quant_of_b(4.0)

    # d: ceiling(b) round-trips every advisor-relayed anchor exactly
    d_ceiling_roundtrips_anchors = all(
        abs(ceiling_of_b(b)["ceiling_tps"] - v) < TOL_DISPLAY_TPS
        for b, v in CEILING_ANCHORS_BPW.items()
    )

    # e: ceiling(b) monotone increasing as bits decrease (585 > 523 > 473.53)
    e_ceiling_monotone = (ceiling_of_b(3.0)["ceiling_tps"] > ceiling_of_b(3.5)["ceiling_tps"]
                          > ceiling_of_b(4.0)["ceiling_tps"])

    # f: b=4 anchor round-trips denken #332's supply cap value exactly
    f_supply_cap_roundtrips_332 = abs(ceiling_of_b(4.0)["ceiling_tps"] - SUPPLY_CAP_INT4) < TOL_332

    # g: int4 branch reproduces the original NO-GO (eff ~473.53, residual ~26.47)
    g_int4_branch_nogo = (abs(int4["tps_eff"] - SUPPLY_CAP_INT4) < TOL_332
                          and not int4["clears_500"]
                          and abs((TARGET - int4["tps_eff"]) - (TARGET - SUPPLY_CAP_INT4)) < TOL_332)

    # h: int4 lever composite clears 500 PRE-cap (505.61) — proves the cap is what binds
    h_int4_precap_clears_500 = int4["precap_tps"] >= TARGET

    # i: sub-int4 branch at b* clears 500 (cap rises to denken ceiling >= 500)
    i_subint4_clears_500 = sub["clears_500"] and sub["tps_eff"] >= TARGET

    # j: sub-int4 branch at b=3.5 also clears 500 (523 >= 500) — robustness across relayed anchors
    sub35 = composite_at_b(3.5, L_STEP_OPTIMISTIC)
    j_subint4_b35_clears_500 = sub35["clears_500"]

    # eta probes: one comfortably inside the budget (clears identity), one outside (blocks).
    eta_clear = ETA_BUDGET_500 - 0.005
    eta_block = ETA_BUDGET_500 + 0.005

    # k: with identity HELD clear, the verdict FLIPS exactly at the PPL gate
    v_violate = verdict_given_ppl(PPL_GATE + 0.10, b_star, eta_clear)
    v_viable = verdict_given_ppl(PPL_GATE - 0.10, b_star, eta_clear)
    k_verdict_flips_at_gate = (v_violate["strict_500_reachable_via_known_levers"] is False
                               and v_viable["strict_500_reachable_via_known_levers"] is True)

    # l: fully-pending mode (no measured inputs) yields pending=True, reachable=None, branches present
    v_pending = verdict_given_ppl(None, b_star)
    l_pending_mode = (v_pending["verdict_pending_measured_ppl"] is True
                      and v_pending["strict_500_reachable_via_known_levers"] is None
                      and v_pending["branch_ppl_violates_gate"]["reachable"] is False
                      and v_pending["branch_ppl_viable"]["reachable"] is True)

    # m: PPL-excluded branch caps L_quant at 1.0x, BELOW the unconstrained int2 ceiling (~1.892)
    m_ppl_caps_lquant = abs(int4["l_quant"] - 1.0) < TOL_EXACT and l_quant_of_b(2.0) > 1.5

    # n: ladder monotonicity, and the live composite tops the ladder
    ladder = [TPS_NONSPEC, TPS_SPEC_OFFSHELF_BI, BASELINE_TPS]
    n_ladder_monotone = (ladder == sorted(ladder)
                         and sub["tps_eff"] >= BASELINE_TPS
                         and int4["tps_eff"] >= TPS_SPEC_OFFSHELF_BI)

    # o: literature PPL prior is explicitly NON-authoritative (not the verdict gate)
    o_literature_non_authoritative = d2["authoritative"] is False

    # --- identity-locus gate (stark #363 attn-free + stark #365 lm_head eta, pending) --- #
    ident_clear = identity_locus_analysis(eta_clear)
    ident_block = identity_locus_analysis(eta_block)

    # q: the 4.02% >500 identity budget is exactly 1 - 500/LAMBDA_CEIL (~0.0402)
    q_eta_budget_derivation = (abs(ETA_BUDGET_500 - (1.0 - TARGET / LAMBDA_CEIL)) < TOL_EXACT
                               and abs(ETA_BUDGET_500 - 0.0402) < 5e-4)

    # r: attention-locus identity tax is FREE (stark #363: eta~0, ratio<1 => best-K=8 even faster)
    r_attn_free_stark363 = (abs(ETA_ATTN_STARK363) < TOL_EXACT and ETA_ATTN_RATIO_STARK363 < 1.0)

    # s: the OLD blanket 9.841% would NOT clear the budget -> the decomposition is what opens the door
    s_blanket_would_not_fit = (ETA_VERIFY_BLANKET > ETA_BUDGET_500
                               and d4["identity"]["blanket_would_clear_budget"] is False)

    # t: the identity gate flips EXACTLY at the budget (PPL held viable)
    ppl_ok = PPL_GATE - 0.10
    t_clear = verdict_given_ppl(ppl_ok, b_star, eta_clear)
    t_block = verdict_given_ppl(ppl_ok, b_star, eta_block)
    t_identity_flips_at_budget = (
        t_clear["strict_500_reachable_via_known_levers"] is True
        and t_block["strict_500_reachable_via_known_levers"] is False
        and t_block["binding_constraint"] == "lmhead_identity_tax_exceeds_4p02pct_budget")

    # u: BOTH gates required — only (PPL-viable AND identity-clears) reaches >500
    u_both_gates_required = (
        verdict_given_ppl(ppl_ok, b_star, eta_clear)["strict_500_reachable_via_known_levers"] is True
        and verdict_given_ppl(ppl_ok, b_star, eta_block)["strict_500_reachable_via_known_levers"] is False
        and verdict_given_ppl(PPL_GATE + 0.10, b_star, eta_clear)["strict_500_reachable_via_known_levers"] is False)

    # v: verdict is PENDING if EITHER measured input is missing
    v_ppl_only = verdict_given_ppl(ppl_ok, b_star, None)         # lm_head eta missing
    v_id_only = verdict_given_ppl(None, b_star, eta_clear)       # ppl missing
    v_pending_if_either_missing = (
        v_ppl_only["verdict_pending"] is True
        and v_ppl_only["verdict_pending_identity_eta"] is True
        and v_ppl_only["strict_500_reachable_via_known_levers"] is None
        and v_id_only["verdict_pending"] is True
        and v_id_only["verdict_pending_measured_ppl"] is True
        and v_id_only["strict_500_reachable_via_known_levers"] is None)

    # w: identity-blocked branch reports identity-taxed lambda < 500 and a positive residual gap
    w_block = verdict_given_ppl(ppl_ok, b_star, eta_block)["branch_ppl_viable_identity_blocked"]
    w_identity_block_gap_positive = (
        ident_block["identity_clears_500_budget"] is False
        and ident_clear["identity_clears_500_budget"] is True
        and w_block["lambda_ceiling_with_identity_tax"] < TARGET
        and w_block["residual_gap_to_500"] > 0.0)

    # --- demand-side route (denken #377 coverage closer; pending #380 + #379) --- #
    # x: budget fractions round-trip the advisor's 35% robust / 18% central of #336's envelope
    x_budget_fracs = (
        abs(d6["delta_cov_robust_budget_frac"] - DELTA_COV_ROBUST / BUDGET_336_ENVELOPE) < TOL_EXACT
        and abs(d6["delta_cov_central_budget_frac"] - DELTA_COV_CENTRAL / BUDGET_336_ENVELOPE) < TOL_EXACT
        and round(d6["delta_cov_robust_budget_frac"] * 100) == 35
        and round(d6["delta_cov_central_budget_frac"] * 100) == 18
        and d6["within_336_budget"] is True)

    # y: non-iid is 1.41x pricier — the price multiplier IS the iid/non-iid slope ratio (>1)
    y_noniid_price = (
        abs(NONIID_PRICE_MULTIPLIER - COV_TO_ET_SLOPE_IID / COV_TO_ET_SLOPE_NONIID) < TOL_EXACT
        and NONIID_PRICE_MULTIPLIER > 1.0
        and abs(NONIID_PRICE_MULTIPLIER - 1.41) < 0.01)

    # z: triple-tail is the only OUT-OF-budget corner (136%) and is a sensitivity band, c* above c
    z_triple_tail = (
        d6["triple_tail_corner"]["out_of_budget"] is True
        and d6["triple_tail_corner"]["cost_frac_of_336_budget"] > 1.0
        and d6["triple_tail_corner"]["is_sensitivity_band_not_central"] is True
        and d6["triple_tail_corner"]["worst_c_star"] > RECOMMENDED_RETRAIN_TARGET_C)

    # aa: demand route is PENDING if EITHER denken #380 or ubel #379 is missing
    r_deliver_only = demand_side_route_analysis("survives", None)      # gap split missing
    r_gap_only = demand_side_route_analysis("pending", 2.0)            # deliverability missing
    aa_route_pending_if_either_missing = (
        d6["route_pending"] is True and d6["route_closes_private_500"] is None
        and r_deliver_only["route_pending"] is True
        and r_deliver_only["route_closes_private_500"] is None
        and r_gap_only["route_pending"] is True
        and r_gap_only["route_closes_private_500"] is None)

    # ab/ac/ad: route resolves only when BOTH harden — closes iff deliverable AND gap addressable
    r_closes = demand_side_route_analysis("survives", 2.0)
    r_deliver_fails = demand_side_route_analysis("fails", 2.0)
    r_gap_irreducible = demand_side_route_analysis("survives", 0.0)
    ab_route_closes_when_both = r_closes["route_closes_private_500"] is True
    ac_route_dead_if_deliver_fails = (r_deliver_fails["route_closes_private_500"] is False
                                      and "deliverability_fails" in r_deliver_fails["binding_constraint"])
    ad_route_dead_if_gap_irreducible = (r_gap_irreducible["route_closes_private_500"] is False
                                        and "gap_irreducible" in r_gap_irreducible["binding_constraint"])

    # ae: implied baseline coverage round-trips (c* - Δcov); robust assumes a lower baseline
    ae_baseline_cov = (
        abs(d6["baseline_cov_robust"] - (RECOMMENDED_RETRAIN_TARGET_C - DELTA_COV_ROBUST)) < TOL_EXACT
        and abs(d6["baseline_cov_central"] - (RECOMMENDED_RETRAIN_TARGET_C - DELTA_COV_CENTRAL)) < TOL_EXACT
        and d6["baseline_cov_central"] > d6["baseline_cov_robust"])

    # af: kappa in (0,1) and the kappa-transferred delivered Δcov still clears the required robust Δcov
    af_kappa_deliver_margin = (
        0.0 < KAPPA_INT4_CT_TRANSFER < 1.0
        and abs(d6["delivered_after_kappa"] - DELIVERABILITY_339_MEAN * KAPPA_INT4_CT_TRANSFER) < TOL_EXACT
        and d6["deliverability_margin"] > 0.0)

    # ag: composite OR-combiner — fully pending->None; demand closes->True even if Route A dead;
    #     both determined dead->False
    c_pending = composite_verdict(None, b_star, None, "pending", None)
    c_demand_closes = composite_verdict(PPL_GATE + 0.10, b_star, eta_block, "survives", 2.0)  # A dead, B closes
    c_both_dead = composite_verdict(PPL_GATE + 0.10, b_star, eta_block, "fails", 2.0)
    ag_composite_or = (
        c_pending["private_500_reachable_via_known_levers"] is None
        and c_pending["verdict_pending"] is True
        and c_demand_closes["private_500_reachable_via_known_levers"] is True
        and c_both_dead["private_500_reachable_via_known_levers"] is False)

    # ah: demand-side is flagged the PRIMARY route and the eta-axis is marked deflated
    ah_demand_primary_eta_deflated = (
        d7["primary_route"] == "demand_side_coverage_denken377"
        and d7["eta_axis_deflated"] is True
        and d6["is_primary_route"] is True
        and d6["eta_axis_deflated"] is True)

    # p: NaN clean (placeholder — finalized in main() after _nan_paths check)
    p_nan_clean = True

    conditions = {
        "a_lquant_int2_reproduces": bool(a_lquant_int2_reproduces),
        "b_lquant_int4_unit": bool(b_lquant_int4_unit),
        "c_lquant_monotone_in_bits": bool(c_lquant_monotone),
        "d_ceiling_roundtrips_anchors": bool(d_ceiling_roundtrips_anchors),
        "e_ceiling_monotone_in_bits": bool(e_ceiling_monotone),
        "f_supply_cap_roundtrips_332": bool(f_supply_cap_roundtrips_332),
        "g_int4_branch_reproduces_nogo": bool(g_int4_branch_nogo),
        "h_int4_precap_clears_500": bool(h_int4_precap_clears_500),
        "i_subint4_branch_clears_500": bool(i_subint4_clears_500),
        "j_subint4_b35_clears_500": bool(j_subint4_b35_clears_500),
        "k_verdict_flips_at_ppl_gate": bool(k_verdict_flips_at_gate),
        "l_pending_mode_emits_both_branches": bool(l_pending_mode),
        "m_ppl_caps_lquant_below_unconstrained": bool(m_ppl_caps_lquant),
        "n_ladder_monotone_and_topped": bool(n_ladder_monotone),
        "o_literature_prior_non_authoritative": bool(o_literature_non_authoritative),
        "q_eta_budget_is_1_minus_500_over_lambda": bool(q_eta_budget_derivation),
        "r_attn_locus_free_stark363": bool(r_attn_free_stark363),
        "s_blanket_would_not_fit_budget": bool(s_blanket_would_not_fit),
        "t_identity_gate_flips_at_budget": bool(t_identity_flips_at_budget),
        "u_both_gates_required_for_500": bool(u_both_gates_required),
        "v_pending_if_either_input_missing": bool(v_pending_if_either_missing),
        "w_identity_block_gap_positive": bool(w_identity_block_gap_positive),
        "x_demand_budget_fracs_35_18": bool(x_budget_fracs),
        "y_noniid_price_is_slope_ratio": bool(y_noniid_price),
        "z_triple_tail_out_of_budget_sensitivity": bool(z_triple_tail),
        "aa_demand_route_pending_if_either_missing": bool(aa_route_pending_if_either_missing),
        "ab_demand_route_closes_when_both_harden": bool(ab_route_closes_when_both),
        "ac_demand_route_dead_if_deliver_fails": bool(ac_route_dead_if_deliver_fails),
        "ad_demand_route_dead_if_gap_irreducible": bool(ad_route_dead_if_gap_irreducible),
        "ae_demand_baseline_cov_roundtrip": bool(ae_baseline_cov),
        "af_kappa_transfer_deliver_margin": bool(af_kappa_deliver_margin),
        "ag_composite_or_route_combiner": bool(ag_composite_or),
        "ah_demand_primary_eta_deflated": bool(ah_demand_primary_eta_deflated),
        "p_nan_clean": bool(p_nan_clean),
    }
    return {
        "conditions": conditions,
        "strict_500_composite_reachability_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
    }


# --------------------------------------------------------------------------- #
# Synthesize
# --------------------------------------------------------------------------- #
def synthesize(measured_ppl: float | None, b_star: float,
               lmhead_eta: float | None = None,
               deliverability: str = "pending",
               gap_addressable_pp: float | None = None) -> dict[str, Any]:
    d1 = deliverable1_lever_analysis(b_star)
    d2 = deliverable2_ppl_forecast()
    d3 = deliverable3_composite_tps(b_star)
    d4 = verdict_given_ppl(measured_ppl, b_star, lmhead_eta)
    d_ident = identity_locus_analysis(lmhead_eta)
    d6 = demand_side_route_analysis(deliverability, gap_addressable_pp)   # Route B (now-primary)
    d7 = composite_verdict(measured_ppl, b_star, lmhead_eta, deliverability, gap_addressable_pp)
    d5 = deliverable5_caveats(b_star)
    st = _selftests(d1, d2, d3, d4, d5, d6, d7, b_star)

    headline = {
        # PRIMARY
        "strict_500_composite_reachability_self_test_passes": (
            st["strict_500_composite_reachability_self_test_passes"]),
        # TEST metrics — top-level verdict is now the composite OR over routes (demand-side primary)
        "tps_max_optimistic_nonspec": d3["tps_max_optimistic_nonspec"],
        "tps_max_optimistic_spec": d3["tps_max_optimistic_spec"],
        "strict_500_reachable_via_known_levers": d7["private_500_reachable_via_known_levers"],
        "primary_route": d7["primary_route"],
        "eta_axis_deflated": d7["eta_axis_deflated"],
        "binding_constraint": d7["binding_constraint"],
        "residual_gap_to_500": d4["residual_gap_to_500"],
        # demand-side route (denken #377, now-primary; pending denken #380 + ubel #379)
        "demand_route_closes_private_500": d6["route_closes_private_500"],
        "demand_route_pending": d6["route_pending"],
        "recommended_retrain_target_c": d6["recommended_retrain_target_c"],
        "delta_cov_robust": d6["delta_cov_robust"],
        "delta_cov_robust_budget_frac": d6["delta_cov_robust_budget_frac"],
        "within_336_budget": d6["within_336_budget"],
        "noniid_price_multiplier": d6["noniid_price_multiplier"],
        "gap_shrink_per_coverage": d6["gap_shrink_per_coverage"],
        "public_private_gap_pct": d6["public_private_gap_pct"],
        "kappa_int4_ct_transfer": d6["kappa_int4_ct_transfer"],
        # pending-aware extras (composite over routes; Route-A sub-gates kept for continuity)
        "verdict_pending": d7["verdict_pending"],          # composite (any operative route pending)
        "pending_inputs": d7["pending_inputs"],            # union: route A (ppl, eta) + route B (#380, #379)
        "route_a_supply_side_reachable": d4["strict_500_reachable_via_known_levers"],
        "verdict_pending_measured_ppl": d4["verdict_pending_measured_ppl"],
        "verdict_pending_identity_eta": d4["verdict_pending_identity_eta"],
        "verdict_pending_route_a": d4["verdict_pending"],
        "ppl_flip_threshold": d4["ppl_flip_threshold"],
        "b_star": b_star,
        "tps_eff_int4_branch": d3["int4_branch"]["tps_eff"],
        "tps_eff_subint4_branch": d3["subint4_branch"]["tps_eff"],
        "reachable_if_ppl_violates_gate": d4["branch_ppl_violates_gate"]["reachable"],
        "reachable_if_ppl_viable_at_b_star": d4["branch_ppl_viable"]["reachable"],
        # identity gate
        "eta_attn_stark363": d_ident["eta_attn_stark363"],
        "eta_blanket_predecomp": d_ident["eta_blanket_predecomp"],
        "eta_budget_500": d_ident["eta_budget_500"],
        "lmhead_eta_flip_threshold": d_ident["lmhead_eta_flip_threshold"],
        "eta_total_verify_locus": d_ident["eta_total_verify_locus"],
        "identity_clears_500_budget": d_ident["identity_clears_500_budget"],
    }

    if d7["verdict_pending"]:
        handoff = (
            f"COMPOSITE VERDICT HELD — multi-route fork, demand-side (denken #377) is the NOW-PRIMARY "
            f"route (eta-axis deflated: {d6['eta_axis_deflation_grounds']}). Pending inputs: "
            f"{d7['pending_inputs']}. "
            f"ROUTE B (demand-side coverage, PRIMARY): retrain drafter to c* >= "
            f"{RECOMMENDED_RETRAIN_TARGET_C} (Δcov robust {DELTA_COV_ROBUST} = "
            f"{d6['delta_cov_robust_budget_frac']*100:.0f}% of #336's {BUDGET_336_ENVELOPE} envelope; "
            f"non-iid {NONIID_PRICE_MULTIPLIER:.2f}x pricier than the deprecated #373 iid model). "
            f"{'CLOSES private-500' if d6['route_closes_private_500'] else 'HELD: ' + d6['binding_constraint']} "
            f"-- pending denken #380 (deliverability survives conservative; modeled p~1.0 via #339 "
            f"N({DELIVERABILITY_339_MEAN},{DELIVERABILITY_339_STD}) x kappa={KAPPA_INT4_CT_TRANSFER}) + "
            f"ubel #379 (coverage->gap split of the {PUBLIC_PRIVATE_GAP_PCT}pp public->private gap). "
            f"ROUTE A (supply-side sub-int4 + identity): "
            f"{'PENDING denken #356 PPL@b* + stark #365 lm_head eta' if d4['verdict_pending'] else d4['binding_constraint']} "
            f"-- leans dead (mixed-precision PPL-blocked per lawine #372; eta-axis deflated). "
            f"At b*={b_star:g}bpw: int4 cap {d3['int4_branch']['tps_eff']:.2f} NO-GO; sub-int4 cap "
            f"{d3['subint4_branch']['ceiling_tps']:.2f} (eff {d3['subint4_branch']['tps_eff']:.2f}) iff PPL+identity clear."
        )
    else:
        handoff = (
            f"COMPOSITE VERDICT RESOLVED: " + d7["verdict_text"]
        )

    return {
        "headline": headline,
        "deliverable1_lever_analysis": d1,
        "deliverable2_ppl_forecast": d2,
        "deliverable3_composite_tps": d3,
        "deliverable4_verdict": d4,
        "deliverable_identity_locus": d_ident,
        "deliverable6_demand_side_route": d6,
        "deliverable7_composite_verdict": d7,
        "deliverable5_caveats": d5,
        "self_test": st,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "lawine #196 (TPS_NONSPEC=165.44), wirbel #326 (TPS_SPEC_OFFSHELF_BI=357.32, "
                "ETA_KERNEL_OFFSHELF=0.3141), wirbel #354 (BASELINE_TPS=481.53 int4 custom Marlin W4A16), "
                "denken #332 y5cl0ena (ceiling(4.0)=473.5295953446407 method-independent batched-verify BW "
                "floor, LAMBDA_CEIL=520.9527323111674), denken #356 ceiling(b) curve anchors {3.0:585, 3.5:523} "
                "(advisor-relayed in PR #357 review), denken #344 waterfall (BODY_FRAC=0.943 STEP_US=1218.2), "
                "kasane #349 (FlashInfer batch-1 excluded: SDPA 36.05us vs FlashInfer 48.20us/layer). "
                "Identity-locus decomposition: stark #363 a0oi2esq (attention-locus identity tax FREE, eta~0, "
                "ratio 0.9167, best K=8, M-invariant fixed-split-k); stark #365 (lmhead_bi_gemm_eta MEASURED, "
                "pending) — verify-locus eta = attn(~0) + lm_head, supersedes blanket 9.841%; >500 budget "
                "ETA_BUDGET_500 = 1 - 500/520.9527 = 0.04022. "
                "Literature PRIOR (non-authoritative): QuIP# arXiv:2402.04396 (int2 +1.19 PPL); "
                "AQLM arXiv:2401.06118 (int2 +1.47 PPL); QTIP arXiv:2406.11235 (int2 +1.70 PPL); "
                "TesseraQ+AWQ arXiv:2410.19103 (int2 +1.35 PPL); Marlin arXiv:2408.11743 (W4A16 4-bit only); "
                "CUDA Graphs arXiv:2605.30571v1 Table 3 (H100 20.6% step overhead; A10G 3-5% ceiling). "
                "PPL gate: deployed 2.3772 gate 2.42 headroom 0.043; the AUTHORITATIVE sub-int4 PPL is "
                "denken #356's MEASURED Gemma-4-E4B value at b* (pending). "
                "DEMAND-SIDE (now-primary; advisor-relayed 16:02Z): denken #377 030uc5mk (merged, "
                "independently verified) recommended_retrain_target c*>=0.9010, Δcov robust 0.0107 / central "
                "0.00565, non-iid cov->E[T] slope 7.91 (vs iid 11.12, 1.41x pricier) anchored on #289 "
                "a1=0.7293 cliff, kappa=0.672 int4-ct transfer, gap_shrink 0.3914pp/cov of the 4.295pp "
                "public->private gap, within #336's +0.031 envelope (35% robust / 18% central), triple-tail "
                "out-of-budget corner (ceil 509.07, rho 0.8038, c* 0.9256 = 136%); deliverability prior "
                "#339 N(0.0385,0.0074). Eta-axis deflated (#373 insufficient + #375 un-deployable); "
                "supply-side mixed-precision PPL-blocked (lawine #372). PENDING hardening: denken #380 "
                "(deliverability_survives_conservative / recipe_is_real) + ubel #379 (coverage->gap split)."
            ),
            "caveats": d5["caveats"],
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
    d1 = syn["deliverable1_lever_analysis"]
    d2 = syn["deliverable2_ppl_forecast"]
    d3 = syn["deliverable3_composite_tps"]
    d4 = syn["deliverable4_verdict"]
    st = syn["self_test"]
    b_star = d3["b_star"]
    int4 = d3["int4_branch"]
    sub = d3["subint4_branch"]
    print("\n" + "=" * 98, flush=True)
    print("STRICT/PRIVATE-500 COMPOSITE REACHABILITY (#357, fern) — multi-route: supply-side (PPL+identity) "
          "OR demand-side coverage (denken #377, NOW-PRIMARY)", flush=True)
    print("=" * 98, flush=True)
    print("  (D1) LEVERS — L_quant and the supply cap are FUNCTIONS of body bits b", flush=True)
    print(f"      L_kernel (spec):  {d1['l_kernel_spec']:.3f}x  [custom Marlin W4A16 in baseline #354]",
          flush=True)
    print(f"      L_quant(b):       {d1['l_quant_formula']}", flush=True)
    print(f"        b=4 {l_quant_of_b(4.0):.4f}x  b=3 {l_quant_of_b(3.0):.4f}x  b=2 {l_quant_of_b(2.0):.4f}x"
          f"   (gated by MEASURED Gemma PPL at b*, not literature)", flush=True)
    print(f"      ceiling(b):       denken #356 anchors {d1['ceiling_anchors_advisor_relayed']}",
          flush=True)
    print(f"        b=4 {ceiling_of_b(4.0)['ceiling_tps']:.2f}  b=3.5 {ceiling_of_b(3.5)['ceiling_tps']:.2f}"
          f"  b=3 {ceiling_of_b(3.0)['ceiling_tps']:.2f}  (RISES as body shrinks)", flush=True)
    print(f"      L_step optimistic: {d1['l_step_optimistic']:.2f}x  (CUDA Graphs A10G 3-5%)", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) SUB-INT4 PPL — LITERATURE PRIOR, NON-AUTHORITATIVE (verdict uses MEASURED Gemma PPL)",
          flush=True)
    print(f"      gate source: {d2['gate_source']}", flush=True)
    print(f"      PPL gate={PPL_GATE:.3f} deployed={PPL_DEPLOYED:.4f} headroom={PPL_HEADROOM:.4f} "
          f"(~{PPL_HEADROOM/PPL_DEPLOYED*100:.1f}% rel)", flush=True)
    print(f"      literature int2 best (QuIP#): +{d2['best_int2_delta']:.2f} PPL -> would_violate_if_"
          f"transplanted={d2['per_method_forecast'][0]['would_violate_if_transplanted']} "
          f"(prior LEANS violate; not decisive)", flush=True)
    print("-" * 98, flush=True)
    di = syn["deliverable_identity_locus"]
    print("  (D2b) IDENTITY LOCUS — verify-locus eta = attn(~0, stark #363) + lm_head (stark #365)",
          flush=True)
    print(f"      eta_attn (stark #363):  {di['eta_attn_stark363']:.4f}  [FREE; ratio "
          f"{di['eta_attn_ratio_stark363']:.4f} <1 -> best K=8 faster than deployed heuristic]", flush=True)
    print(f"      blanket (pre-decomp):   {di['eta_blanket_predecomp']:.5f}  -> would_clear_budget="
          f"{di['blanket_would_clear_budget']}  (9.841% > budget; decomposition is what opens the door)",
          flush=True)
    print(f"      >500 budget:            {di['eta_budget_500']:.5f}  = 1 - 500/LAMBDA_CEIL  "
          f"(lm_head eta flip threshold {di['lmhead_eta_flip_threshold']:.5f})", flush=True)
    print(f"      lm_head eta (stark #365): {di['lmhead_eta_measured']}  -> identity_clears_500_budget="
          f"{di['identity_clears_500_budget']}  (None = pending)", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) COMPOSITE — two branches", flush=True)
    print(f"      Branch A  int4 (PPL-excluded): {int4['base_lifted_tps']:.2f} * {int4['l_step']:.2f} = "
          f"{int4['precap_tps']:.2f} precap | cap {int4['ceiling_tps']:.4f} -> eff {int4['tps_eff']:.4f} "
          f"-> {'CLEARS' if int4['clears_500'] else 'SHORT'} 500", flush=True)
    print(f"      Branch B  b*={b_star:g} (PPL-viable): {sub['base_lifted_tps']:.2f} * {sub['l_step']:.2f} = "
          f"{sub['precap_tps']:.2f} precap | cap {sub['ceiling_tps']:.2f} -> eff {sub['tps_eff']:.2f} "
          f"-> {'CLEARS' if sub['clears_500'] else 'SHORT'} 500 (margin {sub['margin_to_500']:+.2f})",
          flush=True)
    print(f"      Non-spec: {d3['base_nonspec']:.2f} * {L_STEP_OPTIMISTIC:.2f} = "
          f"{d3['tps_max_optimistic_nonspec']:.2f} TPS (<< 500)", flush=True)
    print("-" * 98, flush=True)
    print("  (D4) ROUTE A — supply-side sub-int4, two coupled gates (PPL@b* AND lm_head identity eta)",
          flush=True)
    print(f"      route_a_pending = {d4['verdict_pending']}  "
          f"(ppl={d4['verdict_pending_measured_ppl']}, identity_eta={d4['verdict_pending_identity_eta']})  "
          f"reachable={d4['strict_500_reachable_via_known_levers']}  (leans dead: PPL-blocked lawine #372)",
          flush=True)
    d6 = syn["deliverable6_demand_side_route"]
    d7 = syn["deliverable7_composite_verdict"]
    print("-" * 98, flush=True)
    print("  (D6) ROUTE B — DEMAND-SIDE coverage closer (denken #377) — NOW-PRIMARY route", flush=True)
    print(f"      eta-axis deflated: {d6['eta_axis_deflation_grounds']}", flush=True)
    print(f"      recommended c* >= {d6['recommended_retrain_target_c']}  "
          f"(Δcov robust {d6['delta_cov_robust']} = {d6['delta_cov_robust_budget_frac']*100:.0f}% of #336 "
          f"envelope {BUDGET_336_ENVELOPE}; central {d6['delta_cov_central']} = "
          f"{d6['delta_cov_central_budget_frac']*100:.0f}%)  within_budget={d6['within_336_budget']}",
          flush=True)
    print(f"      non-iid {d6['noniid_price_multiplier']:.2f}x pricier (slope {d6['cov_to_et_slope_iid']}->"
          f"{d6['cov_to_et_slope_noniid']}; deprecated #373 iid +{OLD_373_IID_DELTA_COV})  "
          f"kappa={d6['kappa_int4_ct_transfer']} -> delivered {d6['delivered_after_kappa']:.4f} "
          f"(margin {d6['deliverability_margin']:+.4f})", flush=True)
    print(f"      gap {d6['public_private_gap_pct']}pp, elasticity {d6['gap_shrink_per_coverage']}pp/cov  "
          f"-> ubel #379 split (addressable={d6['gap_coverage_addressable_pp_ubel379']})", flush=True)
    print(f"      triple-tail corner (sensitivity band): c*={d6['triple_tail_corner']['worst_c_star']}, "
          f"cost {d6['triple_tail_corner']['cost_frac_of_336_budget']*100:.0f}% of budget "
          f"(out_of_budget={d6['triple_tail_corner']['out_of_budget']})", flush=True)
    print(f"      deliverability(denken #380)={d6['deliverability_input_denken380']}  "
          f"route_pending={d6['route_pending']}  closes_private_500={d6['route_closes_private_500']}  "
          f"[{d6['binding_constraint']}]", flush=True)
    print("-" * 98, flush=True)
    print("  (D7) COMPOSITE VERDICT — OR over routes (demand-side primary)", flush=True)
    print(f"      verdict_pending = {d7['verdict_pending']}  missing={d7['pending_inputs']}", flush=True)
    print(f"      strict_500/private_500_reachable_via_known_levers = "
          f"{d7['private_500_reachable_via_known_levers']}  (None = pending)", flush=True)
    print(f"      >> {d7['verdict_text']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  PRIMARY strict_500_composite_reachability_self_test_passes = "
          f"{st['strict_500_composite_reachability_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98 + "\n", flush=True)


def _maybe_log_wandb(args: Any, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[strict-500-composite-reachability] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    d1 = syn["deliverable1_lever_analysis"]
    d2 = syn["deliverable2_ppl_forecast"]
    d3 = syn["deliverable3_composite_tps"]
    d4 = syn["deliverable4_verdict"]
    d6 = syn["deliverable6_demand_side_route"]
    d7 = syn["deliverable7_composite_verdict"]
    st = syn["self_test"]
    h = syn["headline"]
    int4 = d3["int4_branch"]
    sub = d3["subint4_branch"]
    b_star = d3["b_star"]

    run = init_wandb_run(
        job_type="strict-500-composite-reachability",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=[
            "strict-500-composite-reachability", "composite-levers", "reachability",
            "ppl-gate", "supply-cap", "ceiling-curve", "denken-356-ceiling", "parameterized-b-star",
            "verdict-pending", "measure-not-guess", "cuda-graphs", "sub-int4", "quantization",
            "amdahl", "marlin-w4a16", "validity-gate", "bank-the-analysis",
            "identity-gate", "lmhead-eta", "stark-363-attn-free", "stark-365-lmhead", "two-gate-fork",
            "demand-side", "coverage-closer", "denken-377", "private-500", "public-private-gap",
            "multi-route", "eta-axis-deflated", "denken-380-pending", "ubel-379-pending",
            "softkd-retrain", "non-iid-coverage",
        ],
        config={
            "baseline_tps_int4": BASELINE_TPS,
            "tps_nonspec": TPS_NONSPEC,
            "tps_spec_offshelf_bi": TPS_SPEC_OFFSHELF_BI,
            "supply_cap_int4_332": SUPPLY_CAP_INT4,
            "ceiling_anchors_bpw": {str(k): v for k, v in sorted(CEILING_ANCHORS_BPW.items())},
            "lambda_ceil": LAMBDA_CEIL,
            "ppl_gate": PPL_GATE,
            "ppl_deployed": PPL_DEPLOYED,
            "ppl_headroom": PPL_HEADROOM,
            "body_frac": BODY_FRAC,
            "non_body_frac": NON_BODY_FRAC,
            "step_us": STEP_US,
            "l_step_optimistic": L_STEP_OPTIMISTIC,
            "l_step_floor": L_STEP_FLOOR,
            "l_kernel_spec": L_KERNEL_SPEC,
            "b_star": b_star,
            "measured_ppl_at_b_star": args.measured_ppl,
            "lmhead_eta_measured": args.lmhead_eta,
            "eta_attn_stark363": ETA_ATTN_STARK363,
            "eta_attn_ratio_stark363": ETA_ATTN_RATIO_STARK363,
            "eta_blanket_predecomp": ETA_VERIFY_BLANKET,
            "eta_budget_500": ETA_BUDGET_500,
            "target": TARGET,
            # demand-side route (denken #377, now-primary)
            "demand_deliverability_input": args.deliverability,
            "gap_coverage_addressable_pp": args.gap_addressable_pp,
            "recommended_retrain_target_c": RECOMMENDED_RETRAIN_TARGET_C,
            "delta_cov_robust": DELTA_COV_ROBUST,
            "delta_cov_central": DELTA_COV_CENTRAL,
            "cov_to_et_slope_noniid": COV_TO_ET_SLOPE_NONIID,
            "cov_to_et_slope_iid": COV_TO_ET_SLOPE_IID,
            "noniid_price_multiplier": NONIID_PRICE_MULTIPLIER,
            "gap_shrink_per_coverage": GAP_SHRINK_PER_COVERAGE,
            "public_private_gap_pct": PUBLIC_PRIVATE_GAP_PCT,
            "budget_336_envelope": BUDGET_336_ENVELOPE,
            "kappa_int4_ct_transfer": KAPPA_INT4_CT_TRANSFER,
            "deliverability_339_mean": DELIVERABILITY_339_MEAN,
            "deliverability_339_std": DELIVERABILITY_339_STD,
            "a1_first_token_cliff_289": PER_POSITION_A1_CLIFF_289,
            "triple_tail_cost_frac_of_budget": TRIPLE_TAIL["cost_frac_of_336_budget"],
            "wandb_group": args.wandb_group,
            "source_runs": (
                "lawine#196, wirbel#326, wirbel#354, denken#332(y5cl0ena), "
                "denken#356(ceiling-curve), denken#344, kasane#349, stark#363(a0oi2esq), stark#365, "
                "denken#377(030uc5mk demand-side), denken#380(deliverability pending), "
                "ubel#379(gap-split pending), #336(budget), #289(per-position), #339(deliverability dist), "
                "lawine#372(supply PPL-blocked), #373/#375(eta-axis deflated)"
            ),
            "literature_prior_non_authoritative": (
                "QuIP# arXiv:2402.04396; AQLM arXiv:2401.06118; QTIP arXiv:2406.11235; "
                "TesseraQ+AWQ arXiv:2410.19103; Marlin arXiv:2408.11743; CUDA Graphs arXiv:2605.30571v1"
            ),
        },
    )
    if run is None:
        print("[strict-500-composite-reachability] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        # PRIMARY
        "strict_500_composite_reachability_self_test_passes": int(bool(
            st["strict_500_composite_reachability_self_test_passes"])),
        # TEST metrics
        "tps_max_optimistic_nonspec": h["tps_max_optimistic_nonspec"],
        "tps_max_optimistic_spec": h["tps_max_optimistic_spec"],
        "residual_gap_to_500": h["residual_gap_to_500"],
        # pending-aware verdict (composite over routes)
        "verdict_pending": int(bool(d7["verdict_pending"])),
        "verdict_pending_measured_ppl": int(bool(d4["verdict_pending_measured_ppl"])),
        "verdict_pending_identity_eta": int(bool(d4["verdict_pending_identity_eta"])),
        "verdict_pending_route_a": int(bool(d4["verdict_pending"])),
        "ppl_flip_threshold": d4["ppl_flip_threshold"],
        "b_star": b_star,
        # demand-side route (denken #377, now-primary; pending denken #380 + ubel #379)
        "eta_axis_deflated": int(bool(d7["eta_axis_deflated"])),
        "demand_route_pending": int(bool(d6["route_pending"])),
        "recommended_retrain_target_c": d6["recommended_retrain_target_c"],
        "delta_cov_robust": d6["delta_cov_robust"],
        "delta_cov_central": d6["delta_cov_central"],
        "delta_cov_robust_budget_frac": d6["delta_cov_robust_budget_frac"],
        "delta_cov_central_budget_frac": d6["delta_cov_central_budget_frac"],
        "within_336_budget": int(bool(d6["within_336_budget"])),
        "noniid_price_multiplier": d6["noniid_price_multiplier"],
        "gap_shrink_per_coverage": d6["gap_shrink_per_coverage"],
        "public_private_gap_pct": d6["public_private_gap_pct"],
        "kappa_int4_ct_transfer": d6["kappa_int4_ct_transfer"],
        "delivered_after_kappa": d6["delivered_after_kappa"],
        "deliverability_margin": d6["deliverability_margin"],
        "triple_tail_cost_frac_of_budget": d6["triple_tail_corner"]["cost_frac_of_336_budget"],
        "triple_tail_out_of_budget": int(bool(d6["triple_tail_corner"]["out_of_budget"])),
        "baseline_cov_robust": d6["baseline_cov_robust"],
        "baseline_cov_central": d6["baseline_cov_central"],
        "reachable_if_ppl_violates_gate": int(bool(d4["branch_ppl_violates_gate"]["reachable"])),
        "reachable_if_ppl_viable_at_b_star": int(bool(d4["branch_ppl_viable"]["reachable"])),
        # identity gate (stark #363 attn-free + stark #365 lm_head eta)
        "eta_attn_stark363": ETA_ATTN_STARK363,
        "eta_blanket_predecomp": ETA_VERIFY_BLANKET,
        "eta_budget_500": ETA_BUDGET_500,
        "lmhead_eta_flip_threshold": d4["lmhead_eta_flip_threshold"],
        # branch composites
        "tps_eff_int4_branch": int4["tps_eff"],
        "tps_eff_subint4_branch": sub["tps_eff"],
        "subint4_margin_to_500": sub["margin_to_500"],
        "int4_precap_tps": int4["precap_tps"],
        "subint4_precap_tps": sub["precap_tps"],
        # parameterized levers / cap
        "l_quant_at_b_star": d1["l_quant_at_b_star"],
        "l_quant_int2_ceiling": d1["l_quant_int2_ceiling"],
        "ceiling_at_b_star": d1["ceiling_at_b_star"],
        "ceiling_at_b35": ceiling_of_b(3.5)["ceiling_tps"],
        "l_step_optimistic": L_STEP_OPTIMISTIC,
        "l_step_floor": L_STEP_FLOOR,
        # literature prior (non-authoritative)
        "lit_int2_delta_best": d2["best_int2_delta"],
        "lit_int2_all_would_violate_if_transplanted": int(bool(
            d2["literature_int2_all_would_violate_if_transplanted"])),
        "lit_authoritative": int(bool(d2["authoritative"])),
        "ppl_headroom": PPL_HEADROOM,
        "baseline_tps_int4": BASELINE_TPS,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # resolved verdict only when the composite is decided (avoid a misleading 0/1 while pending)
    if d7["private_500_reachable_via_known_levers"] is not None:
        summary["strict_500_reachable_via_known_levers"] = int(bool(
            d7["private_500_reachable_via_known_levers"]))
    if d4["strict_500_reachable_via_known_levers"] is not None:
        summary["route_a_supply_side_reachable"] = int(bool(
            d4["strict_500_reachable_via_known_levers"]))
        summary["measured_ppl_at_b_star"] = d4["measured_ppl_at_b_star"]
    if d6["route_closes_private_500"] is not None:
        summary["demand_route_closes_private_500"] = int(bool(d6["route_closes_private_500"]))
    if d6["gap_coverage_addressable_pp_ubel379"] is not None:
        summary["gap_coverage_addressable_pp"] = d6["gap_coverage_addressable_pp_ubel379"]
        summary["gap_irreducible_pp"] = d6["gap_irreducible_pp"]
    if d4["identity_clears_500_budget"] is not None:
        summary["identity_clears_500_budget"] = int(bool(d4["identity_clears_500_budget"]))
        summary["eta_total_verify_locus"] = d4["eta_total_verify_locus"]
        summary["lmhead_eta_measured"] = d4["lmhead_eta_measured"]

    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="strict_500_composite_reachability_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[strict-500-composite-reachability] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--measured-ppl-at-best-sub-int4", "--measured-ppl",
                    dest="measured_ppl", type=float, default=None,
                    help="denken #356 MEASURED Gemma PPL at b* (omit -> verdict stays PENDING)")
    ap.add_argument("--best-sub-int4-bits", "--b-star", dest="b_star", type=float,
                    default=B_STAR_DEFAULT, help="sub-int4 bit-width b* for the PPL-viable branch")
    ap.add_argument("--lmhead-bi-gemm-eta", "--lmhead-eta", dest="lmhead_eta", type=float,
                    default=None,
                    help="stark #365 MEASURED lm_head BI-GEMM identity eta (omit -> identity gate PENDING)")
    ap.add_argument("--demand-deliverability", dest="deliverability",
                    choices=["pending", "survives", "fails"], default="pending",
                    help="denken #380 deliverability_survives_conservative for the demand-side coverage "
                         "closer (omit/pending -> demand route stays PENDING)")
    ap.add_argument("--gap-coverage-addressable-pp", dest="gap_addressable_pp", type=float, default=None,
                    help="ubel #379 pp of the 4.295pp public->private gap that is coverage-addressable "
                         "(omit -> gap split PENDING). PULL ubel's split; do not re-derive it.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="strict-frontier")
    args = ap.parse_args(argv)

    syn = synthesize(args.measured_ppl, args.b_star, args.lmhead_eta,
                     args.deliverability, args.gap_addressable_pp)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 357, "agent": "fern",
        "kind": "strict-500-composite-reachability", "analysis_only": True,
        "measured_ppl_at_b_star": args.measured_ppl, "b_star": args.b_star,
        "lmhead_eta_measured": args.lmhead_eta,
        "demand_deliverability": args.deliverability,
        "gap_coverage_addressable_pp": args.gap_addressable_pp,
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["p_nan_clean"] = not nan_paths
    syn["self_test"]["strict_500_composite_reachability_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["strict_500_composite_reachability_self_test_passes"] = syn["self_test"][
        "strict_500_composite_reachability_self_test_passes"]
    if nan_paths:
        print(f"[strict-500-composite-reachability] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strict_500_composite_reachability_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[strict-500-composite-reachability] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["strict_500_composite_reachability_self_test_passes"]
              and payload["nan_clean"])
        print(f"[strict-500-composite-reachability] self-test {'PASS' if ok else 'FAIL'}",
              flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
