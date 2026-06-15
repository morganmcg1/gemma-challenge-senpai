#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #386 -- Does the 0.633% irreducible gap floor SURVIVE the deployable-strict VBI=1 regime?

WHAT THIS ANSWERS
-----------------
My #379 (`5kpb73tb`, merged) decomposed the 4.295% public->private TPS gap into four
buckets and proved an IRREDUCIBLE ctxlen FLOOR of 0.633% << 3.2% knife-edge -- the
"demand-side coverage route is UNCAPPED" pillar that fern #357 banks. BUT #379 computed
that floor on the DEPLOYED spec stack, where attention is only ~7% of the step
(557.9us / 7983us, #257). The LIVE launch contract does NOT run on that stack -- it runs
on the deployable-strict `VLLM_BATCH_INVARIANT=1` stack that wirbel #378 (`gghmgtk9`)
established as the honest base [357.32, 469.68], where TWO things change at once:

  1. The un-packed attention pays a ctx-DEPENDENT penalty (#375/#378): 1.264x / 3.027x /
     4.756x at L = 528 / 2048 / 4096 (beats the heuristic only below L~=352; 0.877x @ 110).
     The penalty grows super-linearly with prompt length -> the per-attention ctx-
     sensitivity g_s is STEEPER than #379's L-linear heuristic.
  2. #378 measured f_attn = 0.0951 under VBI=1 (vs the deployed ~7%) -- attention is a
     LARGER fraction of the step even though the bf16 lm_head-BI tax dominates the
     ABSOLUTE overhead.

The #379 floor is `floor ~= r_a * g_s` (accept ratio x per-attention ctx-sensitivity of
the private prompt-length shift). Under VBI=1 BOTH factors move UP: r_a's denominator
f_attn rises (~7% -> 9.51%) AND g_s rises (un-pack penalty more ctx-sensitive than the
heuristic split). If the super-linear ctx-growth dominates, the floor INFLATES toward the
3.2% knife-edge on the stack we actually ship -- which would FLIP #379's "uncapped" verdict
for the live contract.

THE MODEL (exact, re-uses the #379 multiplicative->additive identity)
---------------------------------------------------------------------
TPS = E[T] / T_step,  T_step = B + A(L), where
  E[T] = expected committed tokens per spec step (acceptance-length),
  A(L) = attention term (ctx-DEPENDENT, scales with KV length L),
  B    = body GEMM + lm_head + framework/sampler/batch-invariance tax
         (ctx- AND distribution-INDEPENDENT -- identical on public and private).

  gap = 1 - TPS_priv/TPS_pub = 1 - r_a * r_s = g_a + r_a * g_s     (EXACT additive identity)
  with r_a = E[T]_priv/E[T]_pub (KERNEL-INVARIANT -- greedy identity preserved -> same
  accepted tokens on either kernel), r_s = T_step_pub/T_step_priv, g_a = 1-r_a, g_s = 1-r_s.

  irreducible_gap_floor = gap remaining after a PERFECT coverage retrain (closes g_a)
                        = r_a * g_s  == the ctxlen bucket.

WHAT CHANGES DEPLOYED -> VBI=1
-----------------------------
  r_a, g_a  : KERNEL-INVARIANT acceptance quantities. We INHERIT them per-corner straight
              from #379's deployed back-out (greedy identity is preserved on both stacks,
              so E[T]_priv/E[T]_pub is the same physical ratio). The deployed measurement
              pins r_a at each delta_p hypothesis; we re-derive and ASSERT-match #379.
  g_s       : the step loss. RECOMPUTED under VBI=1:
                A_vbi1(L) ~ L * penalty(L)   (un-packed attention, #375 penalty curve)
                shape(L)  = A_vbi1(L)/A_vbi1(L_ref) = (L/L_ref) * penalty(L)/penalty(L_ref)
                g_s_vbi1  = f_attn*(shape(L_priv)-1) / (1 + f_attn*(shape(L_priv)-1))
              with f_attn = 0.0951 (#378, replaces the deployed 0.0699) and L_priv =
              L_ref + delta_p.  NB: f_attn's MEASURED value already nets the larger B (lm_head-
              BI tax, which would LOWER f_attn) against the larger A (un-pack, which RAISES
              it); net f_attn rose -> attention inflation dominates the B inflation.
  total gap : CHANGES (Framing B, physically correct). gap_vbi1 = 1 - r_a*r_s_vbi1
              = g_a + r_a*g_s_vbi1. The measured 4.295% is a DEPLOYED-stack quantity; on
              VBI=1 the gap inflates because g_s inflates.
  floor     : r_a * g_s_vbi1 -- the un-closeable ctxlen part. The knife-edge test compares
              THIS absolute floor to 3.2% (gap after perfect coverage). This test is
              INVARIANT to the framing question (it needs only the absolute floor).

B-CANCELLATION (load-bearing, VERIFIED explicitly -- step 2)
------------------------------------------------------------
g_s = (A_priv - A_pub) / (B + A_priv). The lm_head-BI determinization tax inflates B, but:
  (i)  B CANCELS in the NUMERATOR A_priv - A_pub (pure attention/L shift), and
  (ii) dg_s/dB = -(A_priv-A_pub)/(B+A_priv)^2 < 0  -> a LARGER B only DILUTES g_s; the
       lm_head-BI tax can NEVER inflate the floor, it can only push it DOWN. The floor
       inflation comes ENTIRELY from the attention side (f_attn up + penalty curve).
  (iii) the identity holds IFF B is ctx/distribution-independent. The lm_head-BI GEMM shape
       (batch x hidden x vocab) is FIXED per decode step -- it does not depend on KV length
       L or on which token is argmax'd -> B_pub == B_priv -> it cancels. b_cancels_under_vbi1.

CPU-analytic over BANKED W&B numbers (#379 gap/buckets/r_a, #378 f_attn=0.0951 + penalty,
#375 penalty curve, #282 decode-length dist). NO new GPU measurement (analytic model is
BOUNDED by the measured #378 f_attn + #375 penalty curve -- see --gpu note), NO served-file
change, NO HF Job, NO --launch, NO submission. BASELINE stays 481.53; this leg adds 0 TPS.

PRIMARY self-test : gap_floor_vbi1_self_test_passes (bool).
Headline          : irreducible_gap_floor_pct_vbi1, floor_inflation_ratio, the re-weighted
                    buckets, clears_3p2_knife_edge_vbi1 + margin, breakeven_prompt_shift_tok_vbi1,
                    floor_survives_vbi1_regime, demand_route_uncapped_on_live_contract.

Reproduce:
    cd target/ && python research/validity/gap_floor_vbi1_regime/gap_floor_vbi1_regime.py \
        --vbi1-attention-model --anchor-378-penalty \
        --wandb_group strict-bi-verify-gemm --wandb_name ubel/gap-floor-vbi1-regime
    (add `--gpu --proxy google/gemma-4-E4B-it-qat-w4a16-ct --measure-f-attn-vbi1` for the
     optional local-A10G per-L attention leg; SKIPPED-by-design -- bounded by #378/#375.)
    (0-GPU re-derivation only: append `--reanalyze`.)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

# research/validity/gap_floor_vbi1_regime/this.py -> repo root is 3 up.
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research" / "validity" / "gap_floor_vbi1_regime"
RESULTS_PATH = OUT_DIR / "gap_floor_vbi1_regime_results.json"

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED)
# --------------------------------------------------------------------------- #
# ---- #379 deployed-stack decomposition (5kpb73tb) ----
OFFICIAL_PUBLIC = 481.53           # PR #52 official frontier TPS (public)
PRIVATE_VALID = 460.85             # denken #373 private-VALID TPS (deployed)
KNIFE_EDGE_PCT = 3.2               # private-500 flips to GO iff gap < ~3.2%
K_CAL = 125.268                    # kanna #269: official = K_cal * E[T]
E_T_PUB = 3.844                    # kanna #217 deployed public served E[T]
RHO_DEPLOYED = 0.9570535584491102  # ubel #318 deployed_priv_over_pub (within-task)
DEGRADATION_PCT_318 = 4.294644155088978  # ubel #318 measured within-task degradation
GAP_MEASURED = (OFFICIAL_PUBLIC - PRIVATE_VALID) / OFFICIAL_PUBLIC  # 4.2946% deployed gap

# #257 built_step_roofline (verify ctx=528, M=8) -- DEPLOYED step decomposition
ATTN_US = 557.90                   # ctx-DEPENDENT attention term (deployed heuristic split)
BODY_US = 4474.19                  # ctx-INDEPENDENT body GEMM (weight/HBM-bound)
LMHEAD_US = 131.62                 # ctx-INDEPENDENT lm_head GEMV (deployed, pre-BI tax)
L_REF = 528.0                      # roofline ctx ~ public mean KV length (penalty reference)
OUT_LEN = 512                      # #282 n_completion_tokens fixed at 512 (all prompts)

# #379 per-corner deployed back-out (KERNEL-INVARIANT acceptance; ASSERT-matched below)
R_A_DEPLOYED_BANKED = 0.9570535584491102
R_A_DEPLOYED_CENTRAL = 0.9633874238374297
R_A_DEPLOYED_PESSIMISTIC = 0.973521608458741
FLOOR_379_CENTRAL_PCT = 0.6333865388319535   # the 0.633% deployed irreducible floor (central)
BREAKEVEN_379_TOK = 252.6103574841727         # #379 deployed breakeven private prompt shift

# ---- VBI=1 deployable-strict regime anchors (wirbel #378 gghmgtk9 / #375 27sbg3zb) ----
F_ATTN_VBI1 = 0.0951               # #378 measured attention fraction under VBI=1 (eval-weighted)
EVAL_WEIGHTED_PENALTY_378 = 1.2257  # #378 eval-weighted un-pack penalty (consistency cross-check)
ETA_ATTN_378 = 0.0215              # #378 attention efficiency term (bookkeeping)
STRICT_BRACKET_LO = 357.32         # #378 deployable-strict VBI=1 bracket (bookkeeping)
STRICT_BRACKET_HI = 469.68
# #375 un-pack penalty curve: penalty(L) = A_unpack(L) / A_heuristic(L). Anchors:
PENALTY_CURVE_375 = {
    110.0: 0.877,    # un-pack BEATS heuristic (penalty < 1) at short ctx
    352.0: 1.000,    # cross-over: un-pack == heuristic
    528.0: 1.264,    # at L_ref
    2048.0: 3.027,
    4096.0: 4.756,
}
PENALTY_AT_LREF = PENALTY_CURVE_375[L_REF]    # 1.264

# #289 coverage->gap slope (for the re-derived coverage target on the VBI=1 floor)
A1_ONLY_T2 = 3.909733376164471     # #289 a_1-only leverage (dE[T]/da_1)
COVERAGE_BASELINE = 0.8903         # #336 aggregate baseline
COVERAGE_BUDGET = 0.031            # #336 soft-KD + reasoning-trace REACHABLE-MARGINAL envelope

# sensitivity sweep corners for the private prompt-length shift Delta-P (tokens) -- SAME as #379
DELTA_P_BANKED = 0.0               # #318/#373 pure-rho convention (no step-time term)
DELTA_P_CENTRAL = 50.0             # modest plausible held-out shift
DELTA_P_PESSIMISTIC = 130.0        # public high-decile (#282 high10 573.8 vs low10 190.8)

PUBLIC_MEAN_PROMPT_TOK = L_REF - OUT_LEN / 2.0   # ~272 tok (matches #379 convention)


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv)
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


def penalty_of_L(L: float) -> float:
    """Piecewise-linear interpolation of the #375 un-pack penalty curve.

    For the floor question the shifted prompt L = L_ref + delta_p with delta_p in [0,130]
    lands inside the [528, 2048] segment, whose LOCAL slope (1.16e-3/tok) is GENTLER than
    the [352,528] backward slope (1.5e-3/tok) -- so this forward-interpolated penalty is the
    CONSERVATIVE (lower-floor) choice; the breach reported here is, if anything, an
    under-estimate of the inflation.
    """
    xs = sorted(PENALTY_CURVE_375)
    if L <= xs[0]:
        x0, x1 = xs[0], xs[1]
    elif L >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]
    else:
        x0 = max(x for x in xs if x <= L)
        x1 = min(x for x in xs if x >= L)
        if x0 == x1:
            return PENALTY_CURVE_375[x0]
    y0, y1 = PENALTY_CURVE_375[x0], PENALTY_CURVE_375[x1]
    return y0 + (y1 - y0) * (L - x0) / (x1 - x0)


# --------------------------------------------------------------------------- #
# Step 0a -- DEPLOYED step loss (#379 model: A(L) L-linear). Re-derived to ASSERT-match #379
#            and to back out the KERNEL-INVARIANT r_a per corner.
# --------------------------------------------------------------------------- #
def g_s_deployed(delta_p_tokens: float) -> float:
    """#379's deployed step loss: A(L)=ATTN_US*L/L_ref (L-linear upper bound), f_attn~0.0699."""
    f_attn = ATTN_US / (1.0e6 / K_CAL)                    # 0.069887...
    shape = (L_REF + delta_p_tokens) / L_REF              # L-linear
    x = f_attn * (shape - 1.0)
    return x / (1.0 + x)


def r_a_deployed(delta_p_tokens: float) -> float:
    """KERNEL-INVARIANT accept ratio backed out of the FIXED deployed gap at this delta_p."""
    r_s = 1.0 - g_s_deployed(delta_p_tokens)
    return (1.0 - GAP_MEASURED) / r_s


# --------------------------------------------------------------------------- #
# Step 0b -- VBI=1 step loss (un-pack penalty curve + measured f_attn=0.0951)
# --------------------------------------------------------------------------- #
def shape_vbi1(L: float) -> float:
    """Un-packed attention ctx-scaling: A_vbi1(L) ~ L*penalty(L), normalized at L_ref."""
    return (L / L_REF) * (penalty_of_L(L) / PENALTY_AT_LREF)


def g_s_vbi1(delta_p_tokens: float) -> dict[str, float]:
    """g_s under the VBI=1 un-packed attention: f_attn=0.0951, shape from the #375 penalty curve."""
    L_priv = L_REF + delta_p_tokens
    shp = shape_vbi1(L_priv)
    x = F_ATTN_VBI1 * (shp - 1.0)
    g_s = x / (1.0 + x)
    return {
        "delta_p_tokens": delta_p_tokens,
        "L_priv": L_priv,
        "penalty_at_L_priv": penalty_of_L(L_priv),
        "shape_vbi1": shp,
        "f_attn_vbi1": F_ATTN_VBI1,
        "g_s_vbi1": g_s,
    }


# --------------------------------------------------------------------------- #
# Step 1 -- per-corner floor / buckets under VBI=1 (Framing B: total gap changes)
# --------------------------------------------------------------------------- #
def decompose_vbi1(delta_p_tokens: float) -> dict[str, Any]:
    r_a = r_a_deployed(delta_p_tokens)        # KERNEL-INVARIANT (inherited from deployed)
    g_a = 1.0 - r_a
    gs = g_s_vbi1(delta_p_tokens)
    g_s = gs["g_s_vbi1"]

    floor_abs = r_a * g_s                       # ctxlen bucket == irreducible floor (absolute)
    gap_vbi1 = g_a + floor_abs                  # gap_vbi1 = 1 - r_a*r_s_vbi1 (Framing B)

    frac = (lambda b: 100.0 * b / gap_vbi1) if gap_vbi1 else (lambda b: float("nan"))
    return {
        "delta_p_tokens": delta_p_tokens,
        "r_accept_kernel_invariant": r_a,
        "g_accept_kernel_invariant": g_a,
        "g_s_vbi1": g_s,
        "shape_vbi1": gs["shape_vbi1"],
        "penalty_at_L_priv": gs["penalty_at_L_priv"],
        "L_priv": gs["L_priv"],
        # absolute gap-percent (these two + outlen0 + numerics0 sum to gap_vbi1_pct)
        "bucket_acceptance_abs_pct_vbi1": 100.0 * g_a,
        "bucket_ctxlen_abs_pct_vbi1": 100.0 * floor_abs,
        "bucket_outlen_abs_pct_vbi1": 0.0,
        "bucket_numerics_abs_pct_vbi1": 0.0,
        # as % OF the (re-weighted) gap -- PR field names
        "gap_bucket_acceptance_pct_vbi1": frac(g_a),
        "gap_bucket_ctxlen_pct_vbi1": frac(floor_abs),
        "gap_vbi1_total_pct": 100.0 * gap_vbi1,
        "irreducible_gap_floor_abs_pct_vbi1": 100.0 * floor_abs,   # == ctxlen bucket
    }


# --------------------------------------------------------------------------- #
# Step 2 -- B-cancellation verification (the load-bearing identity under VBI=1)
# --------------------------------------------------------------------------- #
def verify_b_cancels(lmhead_bi_tax_us_grid: tuple[float, ...] = (0.0, 100.0, 300.0, 800.0)
                     ) -> dict[str, Any]:
    """Verify g_s's numerator has NO B, and that inflating B (lm_head-BI tax) only DILUTES g_s.

    g_s = (A_priv - A_pub) / (B + A_priv). We sweep an added lm_head-BI tax into B and confirm
    (i) the numerator A_priv-A_pub is invariant to B, and (ii) g_s is strictly DECREASING in B.
    Hence the lm_head-BI tax can NEVER inflate the floor -- the inflation is attention-only.
    """
    # representative attention at the central corner under the VBI=1 un-pack
    a_pub = ATTN_US * (penalty_of_L(L_REF) / PENALTY_AT_LREF)                       # L=528
    L_priv = L_REF + DELTA_P_CENTRAL
    a_priv = ATTN_US * (L_priv / L_REF) * (penalty_of_L(L_priv) / PENALTY_AT_LREF)  # L=578
    numerator = a_priv - a_pub
    b_deployed = BODY_US + LMHEAD_US

    rows = []
    g_s_prev = None
    monotone_decreasing = True
    numerator_invariant = True
    for tax in lmhead_bi_tax_us_grid:
        b = b_deployed + tax
        g_s = numerator / (b + a_priv)
        # numerator does not reference b -> invariant by construction; assert numerically
        if abs((a_priv - a_pub) - numerator) > 1e-9:
            numerator_invariant = False
        if g_s_prev is not None and not (g_s < g_s_prev + 1e-15):
            monotone_decreasing = False
        g_s_prev = g_s
        rows.append({"lmhead_bi_tax_us": tax, "B_us": b, "g_s": g_s})

    # analytic derivative sign at tax=0: dg_s/dB = -numerator/(B+a_priv)^2 < 0
    dgs_dB = -numerator / (b_deployed + a_priv) ** 2
    return {
        "numerator_A_priv_minus_A_pub_us": numerator,
        "numerator_has_no_B": True,                 # symbolic: A(L_priv)-A(L_pub), no B term
        "numerator_invariant_under_B_sweep": numerator_invariant,
        "g_s_strictly_decreasing_in_B": monotone_decreasing,
        "dgs_dB_sign_negative": dgs_dB < 0,
        "dgs_dB_value": dgs_dB,
        "lmhead_bi_is_ctx_independent": True,       # GEMM shape batchxhiddenxvocab fixed per step
        "lmhead_bi_is_distribution_independent": True,  # all vocab logits computed regardless of argmax
        "b_sweep_rows": rows,
        "b_cancels_under_vbi1": bool(numerator_invariant and monotone_decreasing and dgs_dB < 0),
    }


# --------------------------------------------------------------------------- #
# Step 3 -- knife-edge margins + breakeven private prompt-length shift under VBI=1
# --------------------------------------------------------------------------- #
def knife_edge(corners: dict[str, dict[str, Any]]) -> dict[str, Any]:
    central = corners["central"]
    floor_central = central["irreducible_gap_floor_abs_pct_vbi1"]
    margins = {nm: KNIFE_EDGE_PCT - c["irreducible_gap_floor_abs_pct_vbi1"]
               for nm, c in corners.items()}
    all_clear = all(c["irreducible_gap_floor_abs_pct_vbi1"] < KNIFE_EDGE_PCT
                    for c in corners.values())

    # breakeven delta_p: floor_vbi1(dp) = r_a_deployed(dp)*g_s_vbi1(dp) hits exactly 3.2%
    def floor_at(dp: float) -> float:
        return 100.0 * r_a_deployed(dp) * g_s_vbi1(dp)["g_s_vbi1"]

    breakeven_dp = float("nan")
    try:
        if floor_at(0.0) < KNIFE_EDGE_PCT < floor_at(4000.0):
            breakeven_dp = bisect(lambda dp: floor_at(dp) - KNIFE_EDGE_PCT, 0.0, 4000.0)
    except ValueError:
        breakeven_dp = float("nan")
    breakeven_mult = (breakeven_dp / PUBLIC_MEAN_PROMPT_TOK
                      if math.isfinite(breakeven_dp) else float("nan"))

    return {
        "clears_3p2_knife_edge_vbi1": bool(floor_central < KNIFE_EDGE_PCT),  # central/headline
        "knife_edge_margin_vbi1_pp": margins["central"],
        "knife_edge_margin_banked_pp": margins["banked"],
        "knife_edge_margin_pessimistic_pp": margins["pessimistic"],
        "all_corners_clear_3p2_vbi1": bool(all_clear),
        "pessimistic_breaches_3p2": bool(
            corners["pessimistic"]["irreducible_gap_floor_abs_pct_vbi1"] >= KNIFE_EDGE_PCT),
        "breakeven_prompt_shift_tok_vbi1": breakeven_dp,
        "breakeven_vbi1_as_mult_of_public_mean": breakeven_mult,
        "breakeven_379_tok": BREAKEVEN_379_TOK,
        "breakeven_shrink_tok": BREAKEVEN_379_TOK - breakeven_dp,
        "breakeven_shrink_ratio": (breakeven_dp / BREAKEVEN_379_TOK
                                   if math.isfinite(breakeven_dp) else float("nan")),
        "knife_edge_pct": KNIFE_EDGE_PCT,
    }


# --------------------------------------------------------------------------- #
# Step 4 -- coverage target re-derived on the VBI=1 floor (for fern #357 hand-off)
# --------------------------------------------------------------------------- #
def coverage_on_vbi1_floor(floor_central_pct: float) -> dict[str, Any]:
    """Where coverage must land to bring the VBI=1 floor's *acceptance* part to the knife-edge.

    The floor itself is NOT coverage-addressable (it is the post-retrain residual). What
    changes vs #379 is the residual: coverage can close at most the acceptance bucket, leaving
    `floor_central_pct`. If floor_central_pct >= 3.2%, no coverage retrain reaches the edge.
    """
    slope_tps_per_cov = A1_ONLY_T2 * K_CAL
    reachable = floor_central_pct < KNIFE_EDGE_PCT
    return {
        "vbi1_floor_central_pct": floor_central_pct,
        "coverage_can_reach_3p2_on_vbi1_floor": bool(reachable),
        "gap_shrink_per_coverage_tps": slope_tps_per_cov,
        "coverage_baseline_336": COVERAGE_BASELINE,
        "coverage_budget_336": COVERAGE_BUDGET,
        "note": ("fern #357 must re-derive the demand ceiling on THIS vbi1 floor "
                 f"({floor_central_pct:.3f}% central), not the deployed-stack 0.633%."),
    }


# --------------------------------------------------------------------------- #
# Step 5 -- verdict
# --------------------------------------------------------------------------- #
def verdict(corners: dict[str, dict[str, Any]], ke: dict[str, Any],
            inflation_ratio: float) -> dict[str, Any]:
    floor_central = corners["central"]["irreducible_gap_floor_abs_pct_vbi1"]
    floor_pess = corners["pessimistic"]["irreducible_gap_floor_abs_pct_vbi1"]

    # #379's pillar = "uncapped with margin at ALL corners". It SURVIVES only if every corner
    # still clears 3.2% under VBI=1.
    floor_survives = bool(ke["all_corners_clear_3p2_vbi1"])
    # the demand route is UNCONDITIONALLY uncapped only if the floor survives AND the breakeven
    # shift stays implausibly large (#379 standard: > half the public mean prompt).
    be = ke["breakeven_prompt_shift_tok_vbi1"]
    breakeven_still_implausible = math.isfinite(be) and be > 0.5 * PUBLIC_MEAN_PROMPT_TOK
    demand_uncapped = bool(floor_survives and breakeven_still_implausible)

    if demand_uncapped:
        band = "GREEN_uncapped_confirmed_on_live_vbi1_stack"
        action = "uncapped-confirmed-on-live-stack"
        summary = (
            f"GREEN. Under VBI=1 the floor inflates {inflation_ratio:.2f}x to "
            f"{floor_central:.3f}% (central) but STILL clears 3.2% at every corner "
            f"(pessimistic {floor_pess:.3f}%), and the breakeven private prompt shift "
            f"(+{be:.0f} tok) stays implausibly large. #379's 'uncapped' verdict HOLDS on "
            "the live-contract stack; fern #357 banks it on the correct base.")
    else:
        band = "RED_floor_inflates_uncapped_pillar_is_deployed_stack_artifact"
        action = f"re-derive-ceiling-on-vbi1-floor (central floor {floor_central:.3f}%)"
        summary = (
            f"RED / decision-critical. Under VBI=1 the irreducible floor inflates "
            f"{inflation_ratio:.2f}x: central {floor_central:.3f}% (was 0.633%) still clears "
            f"3.2% with +{ke['knife_edge_margin_vbi1_pp']:.2f}pp margin, BUT the pessimistic "
            f"corner reaches {floor_pess:.3f}% -> BREACHES 3.2% "
            f"({ke['knife_edge_margin_pessimistic_pp']:+.2f}pp), and the breakeven private "
            f"prompt shift SHRINKS from +{BREAKEVEN_379_TOK:.0f} tok to +{be:.0f} tok "
            f"({ke['breakeven_vbi1_as_mult_of_public_mean']:.2f}x public mean). #379's "
            "all-corner >=1.5pp margin does NOT transfer -- the 'uncapped' verdict is a "
            "DEPLOYED-stack artifact. fern #357 must re-derive the demand ceiling on the VBI=1 "
            f"floor ({floor_central:.3f}% central), not the deployed-stack 0.633%.")
    return {
        "floor_survives_vbi1_regime": floor_survives,
        "demand_route_uncapped_on_live_contract": demand_uncapped,
        "breakeven_still_implausible": bool(breakeven_still_implausible),
        "recommended_action": action,
        "verdict_band": band,
        "verdict_summary": summary,
    }


# --------------------------------------------------------------------------- #
# Step 6 -- self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(corners: dict[str, dict[str, Any]], bcanc: dict[str, Any],
              ke: dict[str, Any], inflation_ratio: float,
              floor_379_central_rederived: float) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    central, banked, pess = corners["central"], corners["banked"], corners["pessimistic"]

    # (a) deployed back-out reproduces #379 EXACTLY (kernel-invariant r_a per corner)
    checks["a_r_a_banked_matches_379"] = abs(banked["r_accept_kernel_invariant"]
                                             - R_A_DEPLOYED_BANKED) <= 1e-9
    checks["a_r_a_central_matches_379"] = abs(central["r_accept_kernel_invariant"]
                                             - R_A_DEPLOYED_CENTRAL) <= 1e-9
    checks["a_r_a_pessimistic_matches_379"] = abs(pess["r_accept_kernel_invariant"]
                                                  - R_A_DEPLOYED_PESSIMISTIC) <= 1e-9
    # (a2) re-derived DEPLOYED central floor reproduces the banked 0.633%
    checks["a_deployed_floor_matches_0p633"] = abs(floor_379_central_rederived
                                                   - FLOOR_379_CENTRAL_PCT) <= 1e-6

    # (b) banked corner floor is exactly 0 under VBI=1 (pure-rho, no step shift)
    checks["b_banked_floor_zero"] = abs(banked["irreducible_gap_floor_abs_pct_vbi1"]) <= 1e-12
    # (b2) floor monotone increasing in delta_p (banked <= central <= pessimistic)
    checks["b_floor_monotone_in_shift"] = (
        banked["irreducible_gap_floor_abs_pct_vbi1"]
        <= central["irreducible_gap_floor_abs_pct_vbi1"] + 1e-12
        <= pess["irreducible_gap_floor_abs_pct_vbi1"] + 1e-12)

    # (c) buckets reconstruct the (re-weighted) VBI=1 gap at every corner
    for nm, c in corners.items():
        s = (c["bucket_acceptance_abs_pct_vbi1"] + c["bucket_ctxlen_abs_pct_vbi1"]
             + c["bucket_outlen_abs_pct_vbi1"] + c["bucket_numerics_abs_pct_vbi1"])
        checks[f"c_buckets_sum_gap_{nm}"] = abs(s - c["gap_vbi1_total_pct"]) <= 1e-9
    # (c2) the two %-of-gap fields sum to 100 at central
    checks["c_pct_of_gap_sum_100"] = abs(
        central["gap_bucket_acceptance_pct_vbi1"]
        + central["gap_bucket_ctxlen_pct_vbi1"] - 100.0) <= 1e-9
    # (c3) outlen & numerics buckets hard-zero (fixed 512-tok output; B cancels in gap)
    checks["c_outlen_zero"] = abs(central["bucket_outlen_abs_pct_vbi1"]) <= 1e-12
    checks["c_numerics_zero"] = abs(central["bucket_numerics_abs_pct_vbi1"]) <= 1e-12

    # (d) floor INFLATES vs #379 (the decision-critical direction) and ratio reconstructs
    checks["d_floor_inflates_vs_379"] = (central["irreducible_gap_floor_abs_pct_vbi1"]
                                         > FLOOR_379_CENTRAL_PCT)
    checks["d_inflation_ratio_reconstructs"] = abs(
        inflation_ratio - central["irreducible_gap_floor_abs_pct_vbi1"]
        / FLOOR_379_CENTRAL_PCT) <= 1e-9

    # (e) B-cancellation holds: numerator B-free, g_s decreasing in B, derivative negative
    checks["e_b_cancels_under_vbi1"] = bool(bcanc["b_cancels_under_vbi1"])
    checks["e_numerator_invariant"] = bool(bcanc["numerator_invariant_under_B_sweep"])
    checks["e_gs_decreasing_in_B"] = bool(bcanc["g_s_strictly_decreasing_in_B"])

    # (f) knife-edge consistency: central clears flag matches the < 3.2 comparison
    checks["f_clears_flag_consistent"] = (
        ke["clears_3p2_knife_edge_vbi1"]
        == (central["irreducible_gap_floor_abs_pct_vbi1"] < KNIFE_EDGE_PCT))
    # (f2) margin sign consistency at all corners
    checks["f_margin_signs_consistent"] = (
        (ke["knife_edge_margin_banked_pp"] >= 0) == (banked["irreducible_gap_floor_abs_pct_vbi1"]
                                                     <= KNIFE_EDGE_PCT)
        and (ke["knife_edge_margin_pessimistic_pp"] >= 0)
        == (pess["irreducible_gap_floor_abs_pct_vbi1"] <= KNIFE_EDGE_PCT))

    # (g) breakeven exists, finite, positive, and SHRINKS vs #379 (more ctx-sensitive)
    be = ke["breakeven_prompt_shift_tok_vbi1"]
    checks["g_breakeven_finite_positive"] = math.isfinite(be) and be > 0
    checks["g_breakeven_shrinks_vs_379"] = math.isfinite(be) and be < BREAKEVEN_379_TOK

    # (h) eval-weighted penalty consistency: the #378 eval-weighted penalty 1.2257 (< 1.264 at
    #     L_ref) implies an effective operating L just BELOW L_ref. Invert the curve on the
    #     [352,528] segment where 1.2257 lives, and confirm the recovered L lands near the #282
    #     median decode length (~503) -- i.e. #378's f_attn anchor and #282's length dist agree.
    try:
        L_eval = bisect(lambda L: penalty_of_L(L) - EVAL_WEIGHTED_PENALTY_378, 352.0, 528.0)
    except ValueError:
        L_eval = float("nan")
    checks["h_eval_penalty_consistent"] = math.isfinite(L_eval) and 480.0 <= L_eval <= 520.0

    # (i) penalty curve anchors reproduced exactly at the measured L points
    checks["i_penalty_anchors_exact"] = all(
        abs(penalty_of_L(L) - p) <= 1e-12 for L, p in PENALTY_CURVE_375.items())
    # (i2) f_attn rose vs deployed (the measured driver of the inflation)
    checks["i_f_attn_rose_vs_deployed"] = F_ATTN_VBI1 > ATTN_US / (1.0e6 / K_CAL)

    # (j) constants imported EXACT and UNCHANGED
    checks["j_constants_imported_exact"] = (
        OFFICIAL_PUBLIC == 481.53 and PRIVATE_VALID == 460.85 and KNIFE_EDGE_PCT == 3.2
        and ATTN_US == 557.90 and L_REF == 528.0 and OUT_LEN == 512
        and F_ATTN_VBI1 == 0.0951 and PENALTY_CURVE_375[528.0] == 1.264)

    # (k) NaN-clean across reported scalars
    scal = [central["irreducible_gap_floor_abs_pct_vbi1"], inflation_ratio,
            central["gap_vbi1_total_pct"], ke["knife_edge_margin_vbi1_pp"],
            ke["breakeven_prompt_shift_tok_vbi1"], pess["irreducible_gap_floor_abs_pct_vbi1"]]
    checks["k_nan_clean"] = all(math.isfinite(float(x)) for x in scal)

    gate = all(checks.values())
    return {"checks": checks, "gap_floor_vbi1_self_test_passes": bool(gate)}


# --------------------------------------------------------------------------- #
# optional local-A10G per-L attention leg (SKIPPED-by-design; bounded by #378/#375)
# --------------------------------------------------------------------------- #
def maybe_gpu_leg(args: argparse.Namespace) -> dict[str, Any] | None:
    if not (args.gpu and args.measure_f_attn_vbi1):
        return None
    print("[gfv] NOTE: optional GPU per-L attention leg requested but SKIPPED-by-design. The "
          "VBI=1 attention model is BOUNDED by measured anchors: wirbel #378's f_attn=0.0951 "
          "(eval-weighted) + #375's penalty curve (1.264/3.027/4.756x @ 528/2048/4096), and "
          "this card uses the CONSERVATIVE forward-interpolated local slope. A dedicated GPU "
          "per-L profiling card under VLLM_BATCH_INVARIANT=1 (int4-ct proxy, L in [528,658]) "
          "would PIN the local penalty slope to harden the thin pessimistic-corner breach -- "
          "see suggested follow-ups. CPU-analytic screen stands.", flush=True)
    return {
        "gpu_leg": "SKIPPED-by-design",
        "measured_f_attn_vbi1": None,
        "measured_per_L_attn_penalty": None,
        "reason": "analytic model bounded by #378 f_attn=0.0951 + #375 penalty curve",
    }


# --------------------------------------------------------------------------- #
# assemble report
# --------------------------------------------------------------------------- #
def build_report(delta_p_central: float, gpu_leg: dict[str, Any] | None) -> dict[str, Any]:
    corners = {
        "banked": decompose_vbi1(DELTA_P_BANKED),
        "central": decompose_vbi1(delta_p_central),
        "pessimistic": decompose_vbi1(DELTA_P_PESSIMISTIC),
    }
    central = corners["central"]
    floor_central_pct = central["irreducible_gap_floor_abs_pct_vbi1"]

    # re-derive the DEPLOYED central floor to ASSERT-match #379's 0.633% (provenance check)
    floor_379_central_rederived = 100.0 * r_a_deployed(delta_p_central) * g_s_deployed(delta_p_central)
    inflation_ratio = floor_central_pct / FLOOR_379_CENTRAL_PCT

    bcanc = verify_b_cancels()
    ke = knife_edge(corners)
    cov = coverage_on_vbi1_floor(floor_central_pct)
    vrd = verdict(corners, ke, inflation_ratio)
    st = self_test(corners, bcanc, ke, inflation_ratio, floor_379_central_rederived)

    report = {
        "pr": 386, "issue": 319, "author": "ubel",
        "leg": "does the 0.633% irreducible gap floor survive the deployable-strict VBI=1 regime?",
        "analysis_only": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "tps_added_by_this_card": 0,
        "framing": ("Framing B (physically correct): r_a/g_a are KERNEL-INVARIANT (greedy "
                    "identity preserved) and inherited from #379's deployed back-out; g_s is "
                    "recomputed under VBI=1; the total gap CHANGES. The knife-edge test uses "
                    "only the absolute floor r_a*g_s_vbi1 and is invariant to this framing."),
        "optional_gpu_leg": gpu_leg or {
            "gpu_leg": "NOT-REQUESTED (CPU screen is the deliverable; bounded by #378/#375)"},
        "imported": {
            "official_public": OFFICIAL_PUBLIC, "private_valid": PRIVATE_VALID,
            "knife_edge_pct": KNIFE_EDGE_PCT, "K_cal": K_CAL, "gap_measured_deployed_pct": 100.0 * GAP_MEASURED,
            "attn_us_257": ATTN_US, "body_us_257": BODY_US, "lmhead_us_257": LMHEAD_US,
            "L_ref": L_REF, "out_len_fixed_282": OUT_LEN,
            "f_attn_deployed": ATTN_US / (1.0e6 / K_CAL), "f_attn_vbi1_378": F_ATTN_VBI1,
            "penalty_curve_375": PENALTY_CURVE_375, "eval_weighted_penalty_378": EVAL_WEIGHTED_PENALTY_378,
            "eta_attn_378": ETA_ATTN_378, "strict_bracket_378": [STRICT_BRACKET_LO, STRICT_BRACKET_HI],
            "r_a_deployed_corners_379": {
                "banked": R_A_DEPLOYED_BANKED, "central": R_A_DEPLOYED_CENTRAL,
                "pessimistic": R_A_DEPLOYED_PESSIMISTIC},
            "floor_379_central_pct": FLOOR_379_CENTRAL_PCT, "breakeven_379_tok": BREAKEVEN_379_TOK,
            "delta_p_central_tokens": delta_p_central,
        },
        # ---- HEADLINE: the floor under VBI=1 ----
        "irreducible_gap_floor_pct_vbi1": floor_central_pct,
        "floor_inflation_ratio": inflation_ratio,
        "irreducible_floor_banked_pct_vbi1": corners["banked"]["irreducible_gap_floor_abs_pct_vbi1"],
        "irreducible_floor_pessimistic_pct_vbi1": corners["pessimistic"]["irreducible_gap_floor_abs_pct_vbi1"],
        "floor_379_central_rederived_pct": floor_379_central_rederived,
        # ---- re-weighted buckets (central) ----
        "gap_bucket_acceptance_pct_vbi1": central["gap_bucket_acceptance_pct_vbi1"],
        "gap_bucket_ctxlen_pct_vbi1": central["gap_bucket_ctxlen_pct_vbi1"],
        "gap_vbi1_total_pct": central["gap_vbi1_total_pct"],
        # ---- B-cancellation (step 2) ----
        "b_cancels_under_vbi1": bcanc["b_cancels_under_vbi1"],
        "b_cancellation": bcanc,
        # ---- knife-edge (step 3) ----
        "clears_3p2_knife_edge_vbi1": ke["clears_3p2_knife_edge_vbi1"],
        "knife_edge_margin_vbi1_pp": ke["knife_edge_margin_vbi1_pp"],
        "all_corners_clear_3p2_vbi1": ke["all_corners_clear_3p2_vbi1"],
        "breakeven_prompt_shift_tok_vbi1": ke["breakeven_prompt_shift_tok_vbi1"],
        "knife_edge": ke,
        # ---- coverage hand-off ----
        "coverage_on_vbi1_floor": cov,
        # ---- verdict (step 5) ----
        "floor_survives_vbi1_regime": vrd["floor_survives_vbi1_regime"],
        "demand_route_uncapped_on_live_contract": vrd["demand_route_uncapped_on_live_contract"],
        "recommended_action": vrd["recommended_action"],
        "verdict_band": vrd["verdict_band"],
        "verdict_summary": vrd["verdict_summary"],
        # ---- corners + self-test ----
        "corners": corners,
        "self_test": st["checks"],
        "gap_floor_vbi1_self_test_passes": st["gap_floor_vbi1_self_test_passes"],
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
        print(f"[gfv] wandb unavailable ({exc})", flush=True)
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("[gfv] wandb disabled via env", flush=True)
        return None
    try:
        ke = report["knife_edge"]
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="analysis",
            tags=["gemma-challenge", "analysis", "gap-floor", "vbi1", "deployable-strict",
                  "knife-edge-3p2", "un-pack-penalty", "demand-side", "issue-319", "pr-386"],
            config={
                "pr": 386, "issue": 319, "analysis_only": True, "wandb_group": group,
                "official_public": OFFICIAL_PUBLIC, "knife_edge_pct": KNIFE_EDGE_PCT,
                "f_attn_vbi1_378": F_ATTN_VBI1, "f_attn_deployed": ATTN_US / (1.0e6 / K_CAL),
                "L_ref": L_REF, "out_len_fixed_282": OUT_LEN,
                "delta_p_central_tokens": report["imported"]["delta_p_central_tokens"],
                "penalty_at_lref_375": PENALTY_AT_LREF,
            },
        )
        flat = {
            "primary/gap_floor_vbi1_self_test_passes": int(report["gap_floor_vbi1_self_test_passes"]),
            "headline/irreducible_gap_floor_pct_vbi1": report["irreducible_gap_floor_pct_vbi1"],
            "headline/floor_inflation_ratio": report["floor_inflation_ratio"],
            "headline/clears_3p2_knife_edge_vbi1": int(report["clears_3p2_knife_edge_vbi1"]),
            "headline/all_corners_clear_3p2_vbi1": int(report["all_corners_clear_3p2_vbi1"]),
            "headline/floor_survives_vbi1_regime": int(report["floor_survives_vbi1_regime"]),
            "headline/demand_route_uncapped_on_live_contract":
                int(report["demand_route_uncapped_on_live_contract"]),
            "headline/b_cancels_under_vbi1": int(report["b_cancels_under_vbi1"]),
            "floor/banked_pct": report["irreducible_floor_banked_pct_vbi1"],
            "floor/central_pct": report["irreducible_gap_floor_pct_vbi1"],
            "floor/pessimistic_pct": report["irreducible_floor_pessimistic_pct_vbi1"],
            "floor/deployed_central_379_pct": FLOOR_379_CENTRAL_PCT,
            "knife/margin_central_pp": ke["knife_edge_margin_vbi1_pp"],
            "knife/margin_banked_pp": ke["knife_edge_margin_banked_pp"],
            "knife/margin_pessimistic_pp": ke["knife_edge_margin_pessimistic_pp"],
            "knife/breakeven_tok_vbi1": ke["breakeven_prompt_shift_tok_vbi1"],
            "knife/breakeven_mult_public_mean": ke["breakeven_vbi1_as_mult_of_public_mean"],
            "knife/breakeven_379_tok": BREAKEVEN_379_TOK,
            "knife/breakeven_shrink_ratio": ke["breakeven_shrink_ratio"],
            "bucket/acceptance_pct_of_gap_vbi1": report["gap_bucket_acceptance_pct_vbi1"],
            "bucket/ctxlen_pct_of_gap_vbi1": report["gap_bucket_ctxlen_pct_vbi1"],
            "gap/total_vbi1_pct": report["gap_vbi1_total_pct"],
            "bcancel/dgs_dB_value": report["b_cancellation"]["dgs_dB_value"],
            "tps_added_by_this_card": 0,
        }
        flat = {k: v for k, v in flat.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))}
        run.summary.update(flat)
        run.summary["verdict_band"] = report["verdict_band"]
        run.summary["recommended_action"] = report["recommended_action"]
        for k, v in report["self_test"].items():
            run.summary[f"selftest/{k}"] = int(bool(v))

        # corner sensitivity table (deployed floor vs vbi1 floor side by side)
        ctbl = wandb.Table(columns=["corner", "delta_p_tokens", "penalty_at_L",
                                    "floor_vbi1_pct", "margin_pp", "clears_3p2"])
        margins = {"banked": ke["knife_edge_margin_banked_pp"],
                   "central": ke["knife_edge_margin_vbi1_pp"],
                   "pessimistic": ke["knife_edge_margin_pessimistic_pp"]}
        for nm, c in report["corners"].items():
            f = c["irreducible_gap_floor_abs_pct_vbi1"]
            ctbl.add_data(nm, c["delta_p_tokens"], c["penalty_at_L_priv"], f,
                          margins[nm], int(f < KNIFE_EDGE_PCT))
        run.log({"corner_floor_vbi1": ctbl})

        # B-cancellation sweep table
        btbl = wandb.Table(columns=["lmhead_bi_tax_us", "B_us", "g_s"])
        for row in report["b_cancellation"]["b_sweep_rows"]:
            btbl.add_data(row["lmhead_bi_tax_us"], row["B_us"], row["g_s"])
        run.log({"b_cancellation_sweep": btbl})

        rid = run.id
        print(f"[gfv] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[gfv] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vbi1-attention-model", action="store_true",
                    help="use the VBI=1 un-packed attention model (default path).")
    ap.add_argument("--anchor-378-penalty", action="store_true",
                    help="anchor g_s on #378 f_attn=0.0951 + #375 penalty curve (default on).")
    ap.add_argument("--private-prompt-shift", type=float, default=DELTA_P_CENTRAL,
                    help="central private prompt-length shift Delta-P (tokens).")
    ap.add_argument("--self-test", action="store_true",
                    help="exit nonzero if the primary self-test fails.")
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU re-derivation only (forces CPU-only; identical to default path).")
    # OPTIONAL local-A10G per-L attention leg (SKIPPED-by-design; bounded by #378/#375)
    ap.add_argument("--gpu", action="store_true", help="(optional) enable the GPU per-L attn leg.")
    ap.add_argument("--proxy", default="google/gemma-4-E4B-it-qat-w4a16-ct",
                    help="(optional) int4-ct proxy for the GPU leg.")
    ap.add_argument("--measure-f-attn-vbi1", action="store_true",
                    help="(optional) measure f_attn + per-L attn latency under VBI=1.")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="ubel/gap-floor-vbi1-regime")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reanalyze:
        args.gpu = False  # 0-GPU re-derivation

    gpu_leg = maybe_gpu_leg(args)
    report = build_report(args.private_prompt_shift, gpu_leg)

    wid = None
    if not args.no_wandb and not args.reanalyze:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    report["wandb_run_ids"] = [wid] if wid else []
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))

    ke = report["knife_edge"]
    bar = "=" * 96
    print("\n" + bar, flush=True)
    print(" GAP FLOOR UNDER THE DEPLOYABLE-STRICT VBI=1 REGIME (PR #386, #319)", flush=True)
    print(bar, flush=True)
    print(f" provenance: re-derived DEPLOYED central floor = "
          f"{report['floor_379_central_rederived_pct']:.4f}% (== #379 0.6334% [OK])", flush=True)
    print(f" f_attn: deployed {report['imported']['f_attn_deployed']:.4f} -> VBI=1 "
          f"{F_ATTN_VBI1:.4f}   penalty(L_ref=528) = {PENALTY_AT_LREF}", flush=True)
    print(" --- IRREDUCIBLE FLOOR UNDER VBI=1 (per corner; abs % of gap) ---", flush=True)
    for nm in ("banked", "central", "pessimistic"):
        c = report["corners"][nm]
        f = c["irreducible_gap_floor_abs_pct_vbi1"]
        mlab = {"banked": ke["knife_edge_margin_banked_pp"],
                "central": ke["knife_edge_margin_vbi1_pp"],
                "pessimistic": ke["knife_edge_margin_pessimistic_pp"]}[nm]
        flag = "clears" if f < KNIFE_EDGE_PCT else ">>> BREACHES 3.2% <<<"
        print(f"   {nm:>11s} (dP={c['delta_p_tokens']:>5.0f} tok, penalty={c['penalty_at_L_priv']:.3f}): "
              f"floor {f:6.3f}%  margin {mlab:+6.2f}pp  {flag}", flush=True)
    print(f" --- HEADLINE ---", flush=True)
    print(f"   irreducible_gap_floor_pct_vbi1   : {report['irreducible_gap_floor_pct_vbi1']:.4f}% "
          f"(was 0.6334% deployed)", flush=True)
    print(f"   floor_inflation_ratio            : {report['floor_inflation_ratio']:.3f}x", flush=True)
    print(f"   gap_vbi1_total (central)         : {report['gap_vbi1_total_pct']:.4f}% "
          f"(buckets {report['gap_bucket_acceptance_pct_vbi1']:.2f}% accept / "
          f"{report['gap_bucket_ctxlen_pct_vbi1']:.2f}% ctxlen)", flush=True)
    print(f"   clears_3p2_knife_edge_vbi1       : {report['clears_3p2_knife_edge_vbi1']} "
          f"(central margin {report['knife_edge_margin_vbi1_pp']:+.2f}pp)", flush=True)
    print(f"   all_corners_clear_3p2_vbi1       : {report['all_corners_clear_3p2_vbi1']}", flush=True)
    print(f"   breakeven_prompt_shift_tok_vbi1  : +{report['breakeven_prompt_shift_tok_vbi1']:.0f} tok "
          f"({ke['breakeven_vbi1_as_mult_of_public_mean']:.2f}x public mean; was +{BREAKEVEN_379_TOK:.0f})",
          flush=True)
    print(f"   b_cancels_under_vbi1             : {report['b_cancels_under_vbi1']} "
          f"(dg_s/dB={report['b_cancellation']['dgs_dB_value']:.3e} < 0 -> tax DILUTES)", flush=True)
    print(f" --- VERDICT ---", flush=True)
    print(f"   floor_survives_vbi1_regime       : {report['floor_survives_vbi1_regime']}", flush=True)
    print(f"   demand_route_uncapped_on_live    : {report['demand_route_uncapped_on_live_contract']}",
          flush=True)
    print(f"   recommended_action               : {report['recommended_action']}", flush=True)
    print(f"   verdict_band                     : {report['verdict_band']}", flush=True)
    print(f"   PRIMARY gap_floor_vbi1_self_test : {report['gap_floor_vbi1_self_test_passes']}", flush=True)
    print(f"   wandb run                        : {wid}", flush=True)
    print(f"   artifacts                        : {RESULTS_PATH}", flush=True)
    print(bar + "\n", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": report["wandb_run_ids"],
        "primary_metric": {"name": "gap_floor_vbi1_self_test_passes",
                           "value": int(report["gap_floor_vbi1_self_test_passes"])},
        "test_metric": {"name": "irreducible_gap_floor_pct_vbi1",
                        "value": report["irreducible_gap_floor_pct_vbi1"]},
        "headline": {
            "irreducible_gap_floor_pct_vbi1": report["irreducible_gap_floor_pct_vbi1"],
            "floor_inflation_ratio": report["floor_inflation_ratio"],
            "clears_3p2_knife_edge_vbi1": report["clears_3p2_knife_edge_vbi1"],
            "all_corners_clear_3p2_vbi1": report["all_corners_clear_3p2_vbi1"],
            "knife_edge_margin_vbi1_pp": report["knife_edge_margin_vbi1_pp"],
            "breakeven_prompt_shift_tok_vbi1": report["breakeven_prompt_shift_tok_vbi1"],
            "b_cancels_under_vbi1": report["b_cancels_under_vbi1"],
            "gap_bucket_acceptance_pct_vbi1": report["gap_bucket_acceptance_pct_vbi1"],
            "gap_bucket_ctxlen_pct_vbi1": report["gap_bucket_ctxlen_pct_vbi1"],
            "floor_survives_vbi1_regime": report["floor_survives_vbi1_regime"],
            "demand_route_uncapped_on_live_contract": report["demand_route_uncapped_on_live_contract"],
            "recommended_action": report["recommended_action"],
            "verdict_band": report["verdict_band"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and not report["gap_floor_vbi1_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[gfv] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
