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

GO is a SUPPLY x DEMAND AND-PRODUCT; the SUPPLY lane is the critical path (PR #357, 16:45Z-18:12Z)
-------------------------------------------------------------------------------------------------
The verdict structure EVOLVED again.  It is NOT demand-alone and it is NOT (demand x identity):

  private_500_GO  <=>  (SUPPLY base enables 500)  AND  (DEMAND closer delivers the residual).

wirbel #378 (gghmgtk9) found the honest deployable-strict base is BELOW 500 TODAY, so the supply
base is now the BINDING GO axis (no longer a fixed 518.92 we can bank):
  - Served strict band TODAY is [357.32, 469.68] (off-shelf VBI=1 floor 469.68) -- all < 500.
  - The #375 attn rebuild buys only ~11 TPS (eta_attn=0.0215, NOT #326's whole-step 0.3141), so
    even the rebuilt honest base ~480.7 is still < 500.  wirbel #384 (4f32ks1e, RED, BANKED 18:12Z)
    REFUTED #378's ~93%/~150-TPS "bf16 lm_head-BI" attribution of that remainder: the deployed
    lm_head is already byte-exact int4-Marlin at decode (should_use_atomic_add_reduce(M=8)=False ->
    fixed fp32 reduce; eta_lmhead=0, FREE; f_lmhead=2.24%), so the dominant non-attention strict
    tax actually lives in the 37-layer int4-Marlin BODY (#376).  Corrected ledger = 2 kernel
    rebuilds (attn #375 + body #376), NOT 3; lm_head shares the body Marlin kernel (0 rebuilds).
  - The 518.92 eta-axis pin is DEFLATED on three grounds: #373 insufficient (498.58 < 500),
    #375 un-deployable-without-rebuild, #378 rebuild buys only ~11 TPS.  The 510.01/518.92 SHIPPABLE
    ceilings were bf16-lm_head-premised and are now refuted -> wirbel #390 reseated to re-roll them.
  - STRUCTURAL: private <= public, so if the public-strict base is < 500 the private base is too,
    and demand-alone is INSUFFICIENT.  denken #383 (t68af2yw, RED, BANKED 17:53Z) CONFIRMED this on
    the honest base: private serve point 450.2 -> residual +49.8 TPS = +0.0572 Δcov = 1.84x #336
    budget; a supply lift of +17.2 TPS (floor-joint; +23.8 E[T]-only) is required FIRST, and the
    attn rebuild ALONE does not close it.  So the supply leaf HOLDS pending a MEASURED supply lift
    >= +17.2 TPS from EITHER de-risker:
      wirbel #390 : corrected SHIPPABLE deployable-strict ceiling (re-rolls the refuted 510.01/518.92), OR
      lawine #388 : realized M=1 TPS of lawine #372's GREEN mixed-precision BODY allocation.
  - lawine #372 (mpzfw116, GREEN, BANKED 17:53Z): a SUPPLY lever is ALIVE — NOT uniform sub-int4
    (that died on PPL) but a sensitivity-weighted mixed-precision ALLOCATION (88.8% of body at 3-bit,
    3.2369 avg bpw, +0.17% PPL, gate 2.3812 <= 2.42) buying -21.5% body read (+132.72/+42.15 ANALYTIC
    TPS).  The realized kernel speed vs int4-Marlin at M=1 is UN-measured -> lawine #388 microbenches
    it.  The coverage pilot is OFF the critical path until the base clears ~487-493.

DEMAND residual leaf is RESOLVED central=GREEN / robust=pending-pilot (denken #380 + ubel #382)
----------------------------------------------------------------------------------------------
Retrain the drafter (soft-KD reasoning) to raise acceptance COVERAGE c, which both lifts E[T]
(accepted tok/step -> TPS) AND shrinks the public->private TPS-drift gap.  denken #377 (030uc5mk,
verified) SIZED the closer; denken #380 (00oijpwg, BANKED YELLOW) split deliverability two-tier and
ubel #382 (bn0v5rqr, BANKED GREEN) banked the slope as private-robust:
  - denken #380 : central c>=0.8959 (+0.00565) p_deliver=0.958>=0.90 -> GREEN-deliverable-now;
                  robust c>=0.9010 (+0.0107) p_deliver=0.811<0.90 -> pending ~25-A10G-GPU-hr pilot.
                  kappa_breakeven 0.1222, kappa_margin 0.549 (the kappa transfer axis is ROBUST).
  - ubel  #382 : the 489.8 TPS/unit slope IS private-robust -> 437.3 TPS/unit (flattening 0.893);
                 conservative bank target ~0.911 (66.6% of #336's budget vs 38.9% central); the a1
                 first-token collapse deepens 0.729 -> 0.598 under private OOD but the route survives.
The demand leaf DELIVERS now (central GREEN); it dies only if the gap is forced fully irreducible.

ubel #379 (5kpb73tb, GREEN) BANKED the gap CEILING-CHECK: the 4.295pp gap = 85.25% acceptance
(coverage-ADDRESSABLE) + 14.75% ctxlen (IRREDUCIBLE) + 0% outlen + 0% numerics; the fixed
numerics/identity tax CANCELS in the public->private step diff (floors absolute TPS, not the gap),
refuting "numerics is the irreducible floor".  The off-VBI irreducible floor is 0.633% central, BUT
ubel #386 (xxzujn7a, RED, BANKED 17:53Z) found it does NOT survive the live VBI=1 un-packed-attention
contract -- it INFLATES 2.07x -> 1.310% central.  The route is NOT dead (central 1.310% still clears
the 3.2% knife-edge by +1.89pp), but all_corners_clear_3p2_vbi1=False (the pessimistic corner breaches
at 3.5235%, -0.32pp) and the breakeven private prompt shift HALVES (+253 -> +119 tok), so prompt-shift
sensitivity is now a BINDING risk.  Re-derive the demand ceiling on the 1.310% LIVE floor, NOT 0.633%;
ubel #389 (GPU per-L attention) is reseated to PIN the thin -0.32pp breach.

Identity FOLDS INTO the supply leaf as a compliance prerequisite, two-branch (stark #376 + #381)
-----------------------------------------------------------------------------------------------
Strict greedy-token-identity is a HARD gate on any served config.  stark #376 (ipe3ofie, RED) found
on REAL weights that pinning attention (VLLM_BATCH_INVARIANT->num_splits=1) leaves e2e identity
0.992555 (~heuristic 0.992708); the residual ~0.73% flip is the int4-Marlin BODY GEMM (custom CUDA
op outside the aten dispatcher, env-unpatchable).  BUT the RED is GEOMETRY-SPECIFIC: Marlin is
BIT-EXACT at the 8-row decode-verify width and only M-variant at the 2048-row prefill-replication
width.  stark #381 resolves the served 8-row geometry: GREEN -> env-reachable@decode (1 rebuild:
#375 mha_varlen); RED -> Marlin-rebuild-gated (2 rebuilds).  Identity is REACHABLE in BOTH branches;
it is a COST line-item inside the supply leaf (1 vs 2 kernel rebuilds), NOT a top-level GO factor.

Composite verdict
-----------------
  private_500_GO  <=>  (supply_base_enables_500)  AND  (demand_closer_delivers_residual).
The composite HOLDS while the supply base is pending a MEASURED supply lift >= +17.2 TPS (wirbel #390
corrected SHIPPABLE ceiling OR lawine #388 realized mixed-precision body TPS) -- denken #383 has RESOLVED
the honest re-price (demand-alone insufficient) and wirbel #384 has RESOLVED the lm_head-BI lever as a
non-source (lm_head FREE).  It stamps True iff the supply base enables 500 AND the demand closer
delivers, and False iff either leaf is determined dead (demand gap forced irreducible, or the available
supply lift is < the +17.2 TPS required).  Refinements (NOT GO-gating): the robust coverage pilot (off
the critical path per #383), ubel #389 (pin the VBI=1 floor breach), stark #381 (identity cost branch).
Supply-side Route A (the sub-int4 eta-axis) is computed for continuity but EXCLUDED from the GO.

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
                strict_500_reachable_via_known_levers (AND-product GO: supply x demand; None while
                pending), go_formula, supply_base_enables_500 (None while pending),
                demand_closer_delivers, demand_alone_may_be_insufficient, primary_route,
                eta_axis_deflated, binding_constraint, residual_gap_to_500,
                verdict_pending (composite), verdict_pending_measured_ppl, verdict_pending_identity_eta,
                ppl_flip_threshold, tps_eff_int4_branch, tps_eff_subint4_branch (at b*),
                eta_budget_500, eta_attn_stark363, lmhead_eta_flip_threshold,
                eta_total_verify_locus, identity_clears_500_budget (None while pending),
                supply_base_today_tps, deficit_to_500_today_tps, honest_strict_base_floor_378,
                honest_strict_base_plus_attn_378, eta_axis_base_deflated_518, supply_pending,
                denken383_pending, supply_lift_measured_pending, wirbel384_lmhead_free,
                eta_lmhead_targeted_384, n_kernel_rebuilds_strict_500_384,
                wirbel390_shippable_ceiling_pending,
                demand_leaf_delivers, demand_central_green, demand_robust_modeled,
                demand_leaf_robust_pending, demand_conservative_target_382,
                slope_tps_per_coverage_private_382, slope_is_private_robust_382,
                recommended_retrain_target_c, delta_cov_robust(_budget_frac), within_336_budget,
                noniid_price_multiplier, gap_shrink_per_coverage, public_private_gap_pct,
                kappa_breakeven_380, kappa_margin_380, kappa_int4_ct_transfer, triple_tail_out_of_budget,
                gap_addressable_pp_ubel379, gap_irreducible_pp_central_ubel379, gap_channel_live,
                closer_not_capped_by_irreducible_floor, coverage_target_for_3p2_ubel379,
                slope_tps_per_coverage_ubel379, irreducible_floor_inflates_vbi_386,
                irreducible_floor_vbi1_central_pct_386, central_clears_3p2_vbi1_386,
                all_corners_clear_3p2_vbi1_386, prompt_shift_sensitivity_binding_risk_386,
                supply_lift_required_first_tps_383, demand_alone_insufficient_confirmed_383,
                lawine372_supply_lever_alive, lawine388_realized_tps_pending,
                identity_residual_flip_stark376, identity_cost_branch_pending,
                identity_rebuild_line_items.
GO-gating pending input (now SUPPLY-binding): a MEASURED supply lift >= +17.2 TPS from wirbel #390
                (corrected SHIPPABLE strict ceiling; 510.01/518.92 refuted) OR lawine #388 (realized
                TPS of lawine #372's GREEN mixed-precision body allocation).  denken #383 (demand-alone
                insufficient) and wirbel #384 (lm_head-BI lever refuted — lm_head FREE) are RESOLVED.
COST/REFINEMENT-only (NOT GO-gating): robust-coverage pilot (denken #380 tier-2, off critical path),
                ubel #389 (pin the VBI=1 floor breach), stark #381 (decode-width identity -> 1 vs 2
                rebuilds).
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
# ubel #379 (5kpb73tb, GREEN, advisor-relayed PR #357 review 16:41Z) — the demand-side
# closer's CEILING-CHECK.  The 4.295pp public->private gap DECOMPOSES into a coverage-
# ADDRESSABLE part (acceptance) and an IRREDUCIBLE part (ctxlen); the fixed numerics/identity
# tax CANCELS in the public->private STEP difference (it floors absolute TPS, not the gap),
# which REFUTES the "numerics tax is the irreducible floor" hypothesis.  Net: the demand-side
# closer c >= 0.9010 is NOT capped by an irreducible floor -> BANK it as the demand-side closer.
# (The gap split that was PENDING on ubel #379 is now RESOLVED GREEN; the slope it rests on is
# re-stress-tested by ubel #382 — see below.)
# --------------------------------------------------------------------------- #
GAP_ACCEPTANCE_FRAC_UBEL379: float = 0.8525   # coverage-ADDRESSABLE share of the 4.295pp gap
GAP_CTXLEN_FRAC_UBEL379: float = 0.1475       # IRREDUCIBLE (ctxlen) share
GAP_OUTLEN_FRAC_UBEL379: float = 0.0          # outlen share (fixed 512 -> 0)
GAP_NUMERICS_FRAC_UBEL379: float = 0.0        # numerics tax CANCELS in the public->private step diff
# Coverage-ADDRESSABLE pp of the 4.295pp gap (BANKED GREEN default when no override is supplied).
GAP_ADDRESSABLE_PP_UBEL379: float = PUBLIC_PRIVATE_GAP_PCT * GAP_ACCEPTANCE_FRAC_UBEL379  # ~3.6615pp
# Irreducible floor (central) and the corner band ubel #379 swept; every corner clears the 3.2%
# knife-edge with >= 1.5pp margin.  Central floor == ctxlen share of the gap (4.295 * 0.1475).
GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379: float = 0.633
GAP_IRREDUCIBLE_CORNERS_UBEL379: tuple[float, float, float] = (0.0, 0.633, 1.647)
KNIFE_EDGE_GAP_PCT: float = 3.2               # the 3.2% public->private knife-edge target
KNIFE_EDGE_MIN_MARGIN_PP: float = 1.5         # min margin to the knife-edge across all corners
# ubel #379 independent coverage target (reconciles denken #377 to within 0.0003).
BASELINE_COV_336: float = 0.8903              # #336 baseline coverage
COVERAGE_TARGET_FOR_3P2_UBEL379: float = 0.9011   # +0.0108 from 0.8903
DELTA_COV_UBEL379: float = COVERAGE_TARGET_FOR_3P2_UBEL379 - BASELINE_COV_336  # 0.0108
DENKEN377_DELTA_COV_RECONCILE: float = 0.0111     # denken #377 figure ubel #379 reconciles to <=0.0003
GAP_AFTER_MAX_COVERAGE_RETRAIN_PCT_UBEL379: float = 1.142  # residual gap at full #336 envelope (~3x headroom)
PRIVATE_PROMPT_SHIFT_BREAKEVEN_TOK: float = 253   # +253-tok (~93%) shift to breach 3.2% — implausible OOD
# ubel #379's slope (TPS gained per unit coverage) — DISTINCT from denken #377's pp/cov gap-elasticity.
# This is the number ubel #382 re-stress-tests for private OOD robustness (do NOT hard-bank yet).
SLOPE_TPS_PER_COVERAGE_UBEL379: float = 489.8

# --------------------------------------------------------------------------- #
# IDENTITY-AXIS reframe (stark #376 ipe3ofie RED + reseated stark #381, advisor-relayed 16:41Z).
# The verify-locus identity tax was first decomposed as eta_attn(~0, stark #363 MERGED) + eta_lmhead
# (stark #365).  stark #376 then measured on REAL weights that pinning the attention split
# (VLLM_BATCH_INVARIANT -> num_splits=1) leaves e2e M=8-vs-M=1 identity at 0.992555 (~= the deployed
# heuristic 0.992708): the residual ~0.73% flip is the int4-Marlin BODY GEMM, a custom CUDA op OUTSIDE
# the aten dispatcher that VLLM_BATCH_INVARIANT structurally CANNOT patch.  So the real-weight
# identity=1.0 factor is NOT env-reachable at the PREFILL-REPLICATION geometry; it would be a SECOND
# deployed-kernel rebuild line item (fixed-split-K int4 Marlin/Machete GEMM) alongside #375's
# mha_varlen attention rebuild.
#
# CAVEAT (advisor): the RED is GEOMETRY-SPECIFIC.  stark's size_m sweep found Marlin is BIT-EXACT at
# the decode-verify width (8 rows) and only M-variant at the prefill-replication width (2048 = 8*seq_len).
# The #376 RED is measured at prefill-replication (the only geometry tractable via vLLM's high-level
# prompt_logprobs API), NOT necessarily the DEPLOYED decode-verify geometry.  stark #381 (reseated)
# resolves identity at the literal 8-row served verify width:
#   #381 GREEN -> env-reachable@decode-width (only #375 mha_varlen attention rebuild needed)
#   #381 RED   -> Marlin-rebuild-gated (BOTH #375 attention + a fixed-split-K int4 Marlin GEMM rebuild)
# In BOTH branches byte-exact identity is REACHABLE on the served path; #381 only decides the
# deployment COST (1 vs 2 kernel rebuilds), not whether identity is reachable.
# --------------------------------------------------------------------------- #
IDENTITY_PINNED_E2E_STARK376: float = 0.992555     # pinned (num_splits=1) M=8-vs-M=1 e2e identity, real weights
IDENTITY_HEURISTIC_E2E_STARK376: float = 0.992708  # deployed-heuristic e2e identity (pin doesn't move it)
IDENTITY_RESIDUAL_FLIP_STARK376: float = 0.0073    # ~0.73% residual flip = int4-Marlin body GEMM
DECODE_VERIFY_WIDTH_ROWS: int = 8                  # served verify geometry (Marlin BIT-EXACT here)
PREFILL_REPLICATION_WIDTH_ROWS: int = 2048         # 8*seq_len (Marlin M-variant; the #376 RED geometry)

# --------------------------------------------------------------------------- #
# GO REFRAME (advisor-relayed 16:45Z/17:03Z/17:16Z): the composite GO is now an AND-product of a
# SUPPLY-side honest-base leaf and a DEMAND-side residual leaf — NOT the 16:41Z (demand x identity).
#   private_500_GO  <=>  (supply-side public-strict base reaches/enables 500)  x  (demand closer
#                        delivers the residual).
# Identity reachability (stark #376/#381) folds into the SUPPLY leaf as a strict-byte-exact COMPLIANCE
# prerequisite of the deployable-strict base; it is no longer a standalone GO multiplicand.
# --------------------------------------------------------------------------- #

# --- denken #380 (00oijpwg, merged, YELLOW): two-tier demand-side deliverability ----------------- #
# The single c*>=0.9010 closer SPLITS into a central tier (deliverable now) and a robust tier
# (pending a coverage-lift pilot).  The kappa transfer is ROBUST (not the weak link); the binding
# uncertainty is the DELIVERY distribution.  #339's optimistic N(0.0385,0.0074) is the top of the
# from-scratch-head literature; the DEFENSIBLE fine-tune delivery is N(0.016,0.006).
DEMAND_CLOSER_CENTRAL_C: float = 0.8959        # central tier target (Δcov +0.00565)
DEMAND_CLOSER_ROBUST_C: float = 0.9010         # robust tier target (Δcov +0.0107)
DELIVERABLE_FINETUNE_MEAN_380: float = 0.016   # DEFENSIBLE fine-tune delivered Δcov ~ N(mean, std)
DELIVERABLE_FINETUNE_STD_380: float = 0.006
P_DELIVER_CENTRAL_DEFENSIBLE_380: float = 0.958  # P(deliver central +0.00565) >= 0.90 -> GREEN
P_DELIVER_ROBUST_DEFENSIBLE_380: float = 0.811   # P(deliver robust +0.0107) < 0.90 -> pending pilot
P_DELIVER_THRESHOLD_380: float = 0.90            # delivery-probability bar to bank a tier
RECIPE_IS_REAL_380: bool = True                  # the +0.0107 recipe exists (soft-KD + reasoning trace)
DELIVERABILITY_SURVIVES_CONSERVATIVE_380: bool = False  # robust tier does NOT survive the conservative tail
KAPPA_BREAKEVEN_380: float = 0.1222   # kappa at which the route breaks (< worst program c* corner 0.354)
KAPPA_MARGIN_380: float = 0.549       # margin of kappa=0.672 over breakeven (kappa-axis ROBUST)
COVERAGE_LIFT_PILOT_GPU_HR: float = 25.0  # ~25 A10G-GPU-hr soft-KD + reasoning-trace pilot (#79 RANKPROBE_W=4
#                                            re-measure); separate future card, HUMAN-APPROVAL-GATED spend

# --- ubel #382 (bn0v5rqr, merged 17:11Z, GREEN): slope survives private OOD -------------------- #
# Does ubel #379's 489.8 TPS/unit coverage slope survive the #263 private rank-2+ collapse?  YES:
# the slope's leverage T2 = dE[T]/da1 is the DOWNSTREAM (k>=2) conditional-acceptance tail, blind to
# a1 — and the #263 collapse is concentrated at the FIRST token (a1 0.729->0.598) while the deep tail
# HOLDS/improves (survivor effect).  So the part of the slope private OOD hits hardest is exactly the
# part the slope doesn't see.  Bank the CONSERVATIVE private-anchored target ~0.911 (NOT bare public).
SLOPE_IS_PRIVATE_ROBUST_382: bool = True
DEMAND_ROUTE_SURVIVES_PRIVATE_OOB_382: bool = True
SLOPE_FLATTENING_RATIO_382: float = 0.893       # measured private slope retention (489.8 -> 437.3)
SLOPE_TPS_PER_COVERAGE_PRIVATE_382: float = SLOPE_TPS_PER_COVERAGE_UBEL379 * SLOPE_FLATTENING_RATIO_382  # 437.3
COVERAGE_TARGET_FOR_3P2_PRIVATE_382: float = 0.9024       # measured private-anchored coverage target
COVERAGE_TARGET_FOR_3P2_PRIVATE_CONSERVATIVE_382: float = 0.9109  # conservative stress (0.521 retention)
FLATTENING_BREAKEVEN_382: float = 0.347         # slope must lose 65% (~1.9x #263) to exit #336's budget
SLOPE_CONSERVATIVE_RETENTION_382: float = 0.521  # the conservative-stress slope-retention ratio
DEMAND_CONSERVATIVE_TARGET_382: float = 0.911   # BANK THIS (private-anchored), not the bare public 0.9011
DEMAND_BUDGET_FRAC_CONSERVATIVE_382: float = 0.666  # conservative sizing consumes 66.6% of #336's +0.031
DEMAND_BUDGET_FRAC_CENTRAL_382: float = 0.389       # central sizing consumes 38.9%
A1_PRIVATE_OOD_FIRST_TOKEN_382: tuple[float, float] = (0.729, 0.598)  # a1 first-token collapse (private OOD)

# --- wirbel #378 (gghmgtk9, merged): honest deployable-strict base — do NOT bank 518.92 -------- #
# The only STRICT-byte-exact served knob is VLLM_BATCH_INVARIANT=1 (whole-step batch-invariant
# determinism).  The deployable-strict served band TODAY is [357.32, 469.68] < 500.  The 518.92
# eta-axis pin needs a kernel rebuild that buys only ~11 TPS (eta_attn=0.0215, NOT #326's whole-step
# 0.3141).  #378 ORIGINALLY attributed the ~93%/~150-TPS non-attention deficit to a "bf16 lm_head-BI"
# tax — but wirbel #384 (276e04a, 18:12Z) REFUTED that by-elimination split: the deployed lm_head is
# already byte-exact int4-Marlin at decode (eta_lmhead=0, FREE), and the dominant non-attention strict
# overhead actually lives in the 37-layer int4-Marlin BODY (#376).  Bank the supply base at <=480.7-today.
DEPLOYABLE_STRICT_BAND_378: tuple[float, float] = (357.32, 469.68)  # served strict band TODAY (< 500)
HONEST_STRICT_BASE_FLOOR_378: float = 469.68    # off-the-shelf deployable-strict floor (VBI=1)
ATTN_REBUILD_TPS_GAIN_378: float = 11.0         # ~11 TPS bought by the #375 attention rebuild
HONEST_STRICT_BASE_PLUS_ATTN_378: float = 480.7  # floor + ~11-TPS attn rebuild (still < 500)
ETA_ATTN_378: float = 0.0215                    # attention-un-pack eta (NOT #326's whole-step 0.3141)
ETA_AXIS_BASE_DEFLATED_518: float = 518.92      # the deflated eta-axis pin — do NOT bank as the base
LMHEAD_BI_TAX_TPS_378_REFUTED: float = 150.0    # #378 by-elimination lm_head-BI figure — REFUTED by #384
ATTN_DEFICIT_UNTOUCHED_FRAC_378: float = 0.93   # pinning num_splits leaves ~93% of the strict deficit
ETA_AXIS_DEFLATION_GROUNDS_378: str = (
    "#373 insufficient (498.58<500) + #375 un-deployable-without-rebuild + #378 rebuild buys only "
    "~11 TPS (eta_attn=0.0215), not the headline whole-step eta")

# --- ubel #386 (xxzujn7a, merged b0de7eb 17:53Z): RESOLVED RED — the floor INFLATES under VBI=1 --- #
# ubel #379's IRREDUCIBLE 0.633% gap floor does NOT survive the VBI=1 un-packed-attention contract: it
# inflates 2.07x -> 1.310% central.  Central STILL clears the 3.2% knife-edge (+1.89pp) so the demand
# route is not dead, BUT all_corners_clear_3p2_vbi1=False (the pessimistic corner breaches at 3.5235%,
# -0.32pp) and the breakeven private prompt-shift HALVES (+253 -> +119 tok).  Re-derive the demand-route
# ceiling on the 1.310% central floor (NOT 0.633%) and treat private prompt-length-shift sensitivity as
# a BINDING risk.  The -0.32pp breach is thin/slope-interpolated; ubel reseated to a GPU per-L attention
# measurement (#389) to PIN it.  Banked default: irreducible_floor_survives_vbi="inflates".
UBEL386_FLOOR_SURVIVES_VBI: str = "inflates"        # banked resolution (was pending)
IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386: float = 1.310   # % — inflated central floor under VBI=1
FLOOR_INFLATION_MULT_386: float = 2.07              # 0.633% -> 1.310% inflation factor
ALL_CORNERS_CLEAR_3P2_VBI1_386: bool = False        # pessimistic corner now breaches
PESSIMISTIC_CORNER_VBI1_PCT_386: float = 3.5235     # the breaching pessimistic corner (> 3.2% knife-edge)
PESSIMISTIC_CORNER_MARGIN_PP_386: float = -0.32     # the corner's (negative) margin to 3.2%
BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386: float = 119.0  # halved breakeven prompt shift (was 253)
BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK: float = 253.0   # pre-VBI comfortable buffer (now gone)
CENTRAL_CLEARS_3P2_VBI1_386: bool = True            # central 1.310% still clears 3.2% (+1.89pp)
CENTRAL_MARGIN_TO_3P2_VBI1_PP_386: float = 1.89     # central margin to the knife-edge under VBI=1
UBEL389_PIN_BREACH_PENDING: bool = False            # ubel #389 LANDED (fqt33bj3): breach REFUTED on the measured slope

# --- denken #383 (t68af2yw, merged ce5f39a 17:53Z): RED — demand-alone INSUFFICIENT on honest base -- #
# Clean FLIP from #377, driven ENTIRELY by the base move 518.92 -> <=480.8 (reproduces_377_under_revival
# =True, round-trip exact: the math didn't change, the base did).  On the honest floor (469.68) the
# private serve point is 450.2 -> residual +49.8 TPS -> required Δcoverage +0.0572 = 1.84x the +0.031
# #336 budget.  So demand-side coverage retrain ALONE does NOT reach private-strict-500 at ANY coverage
# in budget.  A private-500 GO requires a SUPPLY lift of +17.2 TPS (floor-joint) to clear the public base
# from ~480.8 to ~487-493 FIRST, after which a #377-sized demand sliver finishes.  The ~25-GPU-hr coverage
# pilot is OFF the critical path until the base clears.  Banked default: demand_reaches_500_on_floor="no".
DENKEN383_REACHES_500_ON_FLOOR: str = "no"          # banked resolution (demand-alone insufficient)
SUPPLY_LIFT_REQUIRED_FIRST_TPS_383: float = 17.2    # +17.2 TPS floor-joint supply lift required first
SUPPLY_LIFT_REQUIRED_ET_ONLY_TPS_383: float = 23.8  # +23.8 TPS E[T]-only variant
PRIVATE_ON_FLOOR_383: float = 450.2                 # private serve point on the 469.68 honest floor
RESIDUAL_TO_500_ON_FLOOR_383: float = 49.8          # +49.8 TPS residual to 500 on the floor
REQUIRED_DCOV_383: float = 0.0572                   # required Δcoverage (demand-alone) on the honest base
REQUIRED_DCOV_BUDGET_MULT_383: float = 1.84         # 1.84x the +0.031 #336 budget (out of budget)
PRIVATE_CAP_CENTRAL_FLOOR_383: float = 483.3        # demand-max private cap, central, on the floor
PRIVATE_CAP_WORST_FLOOR_383: float = 470.3          # demand-max private cap, worst, on the floor
PRIVATE_CAP_CENTRAL_ATTN_383: float = 494.1         # demand-max private cap, central, +attn rebuild
PRIVATE_CAP_WORST_ATTN_383: float = 480.9           # demand-max private cap, worst, +attn rebuild
REPRODUCES_377_UNDER_REVIVAL_383: bool = True       # #377 round-trips exactly under the 518.92 base
ATTN_REBUILD_ALONE_CLOSES_SUPPLY_GAP_383: bool = False  # remainder is lm_head-BI, not attention
PILOT_ON_CRITICAL_PATH_383: bool = False            # coverage pilot off critical path until base clears
BASE_CLEARS_PILOT_RELEVANT_BAND_383: tuple[float, float] = (487.0, 493.0)  # base must reach here first

# --- lawine #372 (mpzfw116, merged fc8152d 17:53Z): GREEN — a SUPPLY lever is ALIVE --------------- #
# Mixed-precision SENSITIVITY-WEIGHTED bit ALLOCATION (NOT uniform — uniform-3bit died on PPL) achieves
# 3.2369 avg bpw with 88.8% of body params at 3-bit for only +0.17% PPL, gate 2.3812 <= 2.42, and a
# -21.5% body read.  This de-risks the supply lift denken #383 says is required first.  CAVEAT (the gate):
# the +132.72 / +42.15 TPS lifts are BW-bound ANALYTIC predictions; the REALIZED cb3 (RHT+VQ, QTIP/QuIP#
# -class) kernel speed vs int4-Marlin at M=1 on A10G is UN-measured — lawine #388 is microbenching it.
# (This SUPERSEDES the earlier "mixed-precision PPL-blocked" read, which was the uniform-3bit variant.)
LAWINE372_SUPPLY_LEVER_ALIVE: bool = True           # the supply mixed-precision lever is alive (GREEN)
MIXED_PRECISION_AVG_BPW_372: float = 3.2369         # sensitivity-weighted average body bit-width
BODY_3BIT_FRAC_372: float = 0.888                   # 88.8% of body params allocated to 3-bit
MIXED_PRECISION_PPL_DELTA_PCT_372: float = 0.17     # +0.17% PPL from the allocation
MIXED_PRECISION_GATE_PPL_372: float = 2.3812        # resulting PPL, <= 2.42 gate (passes)
BODY_READ_REDUCTION_372: float = 0.215              # -21.5% body read traffic
MIXED_PRECISION_ANALYTIC_LIFT_TPS_372: tuple[float, float] = (132.72, 42.15)  # BW-bound ANALYTIC lifts
UNIFORM_3BIT_DIED_ON_PPL_372: bool = True           # the uniform-3bit variant failed the gate
LAWINE388_REALIZED_TPS_PENDING: bool = False        # LANDED (g5lfdpgw): -21.5% read -> +33 honest / +38.34 TPS

# --- wirbel #384 (4f32ks1e, merged 276e04a 18:12Z): CORRECTION — lm_head is FREE, tax is in the BODY - #
# Refutes the 17:53Z "wirbel #384 = lm_head-BI ~93%/~150-TPS determinization tax" relay.  The deployed
# lm_head is ALREADY byte-exact strict at decode: it is UNTIED int4-Marlin (channel-wise W4A16, the same
# Marlin kernel family as the body #376), NOT a bf16 GEMM.  On A10G sm_86, should_use_atomic_add_reduce(
# M=8, n=16384)=False forces the fixed fp32 reduce -> byte-exact across M in {1,2,4,8}.  So there is NO
# lm_head determinization tax (eta_lmhead_targeted=0.0; lm_head is f_lmhead=2.24% of the step).  #378's
# ~93% was a BY-ELIMINATION artifact (lmhead_bi_share_of_vbi_overhead=0.3467 steelman / ~0% incremental):
# it attributed the whole non-attention VBI tax to the lm_head, but the tax lives in the 37-layer int4-
# Marlin BODY (#376).  CORRECTED supply ledger: n_distinct_kernel_rebuilds_for_strict_500 = 2 (attention
# #375 mha_varlen + body-Marlin #376 size_m), NOT 3 — lm_head shares the body Marlin kernel (0 rebuilds,
# ~0 eta).  Open: stark #381 (in flight) may DROP the count to 1 (does the body int4-Marlin ALSO go
# byte-exact at the 8-row decode width via the same atomic-add mechanism that freed the lm_head?).
WIRBEL384_LMHEAD_FREE: bool = True                   # deployed lm_head already byte-exact strict (int4-Marlin)
ETA_LMHEAD_TARGETED_384: float = 0.0                 # NO lm_head determinization tax (eta=0)
F_LMHEAD_384: float = 0.0224                         # lm_head is 2.24% of the step
LMHEAD_IS_INT4_MARLIN_NOT_BF16_384: bool = True      # untied int4-Marlin W4A16, NOT a bf16 GEMM
LMHEAD_BI_SHARE_OF_VBI_OVERHEAD_384: float = 0.3467  # steelman share (#378's ~93% by-elimination, ~0% incremental)
LMHEAD_BI_INCREMENTAL_SHARE_384: float = 0.0         # ~0% incremental — the tax is NOT in the lm_head
N_KERNEL_REBUILDS_STRICT_500_384: int = 2            # attn #375 + body-Marlin #376 (NOT 3); lm_head FREE
DOMINANT_NONATTN_STRICT_LOCUS_384: str = "int4-Marlin BODY (#376)"  # NOT bf16 lm_head (refuted)
BODY_MARLIN_DECODE_STRICT_PENDING_STARK381_384: bool = True  # #381: does body also free at 8-row decode -> rebuilds->1

# --- wirbel #390 (reseated 18:12Z): re-roll the corrected SHIPPABLE deployable-strict ceiling --------- #
# The 510.01 ("holds lm_head at bf16-deterministic") and 518.92 ceilings were computed under the
# now-REFUTED bf16-lm_head premise.  wirbel #390 GPU-measures the realized shippable strict decode TPS
# under the corrected model (lm_head free int4-Marlin) and reports the true gap-to-500.  It is the
# NOW-PRIMARY supply-resolution card: HOLD the composite GO until #390 lands the corrected realized
# shippable number (the supply lift can also come from lawine #388's realized body-allocation TPS).
WIRBEL390_SHIPPABLE_CEILING_PENDING: bool = False    # LANDED (5y64zbjz): corrected base 471.42 measured (e2e stark #381 separate)
SHIPPABLE_CEILING_REFUTED_BF16_PREMISE_390: tuple[float, float] = (510.01, 518.92)  # the bf16-lm_head pin (#390 reinstates 510.01, refutes 518.92)

# --- wirbel #390 (5y64zbjz, merged a6130d21, 19:01Z): LANDED — corrected SHIPPABLE strict base ---- #
# The reseated #390 GPU-measured the realized shippable strict decode TPS under the corrected
# (lm_head-free int4-Marlin) model AND pre-answered stark #381 at the per-GEMM level.  Key results:
#   - The body int4-Marlin is ALREADY byte-exact strict at the M=8 decode-verify width — all 8 GEMMs
#     (q/k/v/o/gate/up/down + lm_head), 3 seeds x 8 trials, real geometry g128, INCL the atomic-add-
#     eligible small-n k/v_proj (all A10G guards force the fixed fp32 reduce).  So the per-GEMM identity
#     question is GREEN => n_distinct_kernel_rebuilds_for_strict_500 = 1 (attention only; lm_head=0 #384,
#     body=0 here).  This is the `--stark381-decode-identity green` condition: drop the ledger 2 -> 1.
#     (stark #381 stays open as the independent e2e confirmation incl norms; the per-GEMM Q is answered.)
#   - Corrected shippable band = [471.20, 509.78], deployed-floor +113.88 over the #378 [357.32, 469.68]
#     bracket (that bracket was the PHANTOM bf16 body+lm_head over-determinization;
#     spread_is_lmhead_bf16_tax=False).  The 510.01 was the corrected shippable CEILING all along
#     (#378 mislabeled it "NOT shippable"); only the 518.92 stays refuted.
#   - Realized deployed strict = 471.42; gap_to_500 = 28.58; supply_alone_closes_500 = False (NO cb3
#     shrink).  This is the CORRECTED supply BASE — replaces the banked "469.68 floor + 11 attn = 480.7"
#     with 471.42 (rebuilds=1, attention eta_attn=0.02145 the SOLE strict tax).
WIRBEL390_LANDED: bool = True                        # corrected shippable strict base — MEASURED (5y64zbjz)
REALIZED_DEPLOYED_STRICT_390: float = 471.42         # corrected supply BASE (rebuilds=1; replaces 480.7)
SHIPPABLE_BAND_390: tuple[float, float] = (471.20, 509.78)  # corrected shippable strict band
GAP_TO_500_390: float = 28.58                        # bare gap_to_500 on the corrected base (500 - 471.42)
SUPPLY_ALONE_CLOSES_500_390: bool = False            # supply-alone (NO cb3 shrink) does NOT clear 500
ETA_ATTN_390: float = 0.02145                        # attention eta — the SOLE strict tax (rebuilds=1)
N_KERNEL_REBUILDS_STRICT_500_390: int = 1            # attn only (lm_head=0 #384, body=0 #390); ledger 2->1
BODY_MARLIN_DECODE_STRICT_GREEN_390: bool = True     # body int4-Marlin byte-exact @ M=8 (all 8 GEMMs, 3 seeds x 8 trials, g128)
STARK381_DECODE_IDENTITY_PER_GEMM_GREEN_390: bool = True  # the --stark381-decode-identity green condition (per-GEMM)
SPREAD_IS_LMHEAD_BF16_TAX_390: bool = False          # the [357.32,469.68] bracket was PHANTOM bf16 over-determinization
SHIPPABLE_CEILING_510_REINSTATED_390: float = 510.01  # corrected shippable CEILING all along (#378 mislabeled "NOT shippable")
SHIPPABLE_CEILING_518_STILL_REFUTED_390: float = 518.92  # the only number that stays refuted (bf16-lm_head premise)
DEPLOYED_FLOOR_LIFT_OVER_378_BAND_390: float = 113.88  # 471.20 - 357.32 (band-lower lift over the phantom #378 bracket)

# --- lawine #388 (g5lfdpgw, merged c8bcdd2, 19:01Z): LANDED — realized cb3 supply LIFT ------------- #
# The realized body-allocation supply lift that GO-gated the supply leaf.  The cb3 (RHT+VQ, QTIP/QuIP#
# -class) body shrink of lawine #372's GREEN sensitivity-weighted mixed-precision allocation realizes:
#   +38.34 TPS realistic (1.1234x) / +33 honest (draft-separated).  BOTH >= the +17.2 required (denken
#   #383, floor-joint) AND >= the +23.8 E[T]-only robust variant.  closes_383_supply_gap_floor = True.
# ★ THE CAVEAT THAT GATES DEPLOYABILITY: m1_is_bw_bound = False — Marlin at M=1 runs at HBM efficiency
#   only 0.256 (overhead/launch-bound at single-token), so this is the M=1 number and the byte-shrink
#   buys ~46% of its 1.2744x roofline.  The SERVED regime is M=8 (verify width); lawine #391 measures
#   whether the efficiency rises toward BW-bound there.
# ★ OPTIMISM FLAG: the +38.34 lumps the #378 draft fraction 0.1201 (un-shrunk MTP drafter) into the
#   shrinkable body — honest body-only is +33; USE +33 for --supply-lift-available-tps.
LAWINE388_LANDED: bool = True                        # realized cb3 supply lift — MEASURED (g5lfdpgw)
CB3_LIFT_HONEST_TPS_388: float = 33.0                # draft-separated body-only lift — USE THIS for supply-lift-available
CB3_LIFT_REALISTIC_TPS_388: float = 38.34            # realistic lift (1.1234x; lumps the #378 draft fraction)
CB3_LIFT_MULT_388: float = 1.1234                    # realistic body-shrink TPS multiplier
CB3_CLOSES_383_SUPPLY_GAP_FLOOR_388: bool = True     # both lifts >= the +17.2 required (and >= +23.8 E[T]-only)
CB3_M1_IS_BW_BOUND_388: bool = False                 # ★ Marlin M=1 is overhead/launch-bound (NOT BW-bound)
CB3_M1_HBM_EFF_388: float = 0.256                    # Marlin M=1 HBM efficiency (low -> overhead-bound)
CB3_M1_ROOFLINE_MULT_388: float = 1.2744             # the M=1 BW-bound roofline the byte-shrink targets (~46% realized)
CB3_REALIZED_FRAC_OF_ROOFLINE_388: float = 0.46      # fraction of the 1.2744x roofline realized at M=1
CB3_DRAFT_FRAC_LUMPED_388: float = 0.1201            # #378 un-shrunk MTP drafter fraction lumped into +38.34 (-> use +33)

# --- denken #387 (z8osvif8, merged c0e88d2, 19:01Z): LANDED — demand anchor MEASURED + MTP K=7 -------- #
# The demand-leaf coverage anchor is now MEASURED (not modeled), and it carries a premise correction.
#   measured_top4_coverage = 0.89027, coverage_anchor_gap = +0.000 -> denken #383 RED is ROBUST to the
#   measured anchor (required_delta_floor_measured = +0.0572, still 1.84x the #336 budget).  Drop the
#   "modeled 0.8903 baseline" hedge — it is grounded.
# ★ The deployed drafter is MTP K=7 (PR #52 method="mtp", num_speculative_tokens=7), NOT EAGLE-3 — any
#   EAGLE-3-tree framing on the demand leaf re-labels to the #289 MTP conditional-accept ladder.
DENKEN387_LANDED: bool = True                        # demand anchor MEASURED (z8osvif8)
MEASURED_TOP4_COVERAGE_387: float = 0.89027          # measured baseline top-4 coverage (grounds 0.8903)
COVERAGE_ANCHOR_GAP_387: float = 0.000               # measured vs modeled anchor gap (+0.000 -> #383 robust)
REQUIRED_DELTA_FLOOR_MEASURED_387: float = 0.0572    # required Δcov on the MEASURED anchor (still 1.84x budget)
DENKEN383_RED_ROBUST_TO_MEASURED_ANCHOR_387: bool = True  # #383 RED survives the measured anchor
DEPLOYED_DRAFTER_MTP_K_387: int = 7                  # deployed drafter is MTP K=7 (NOT EAGLE-3; PR #52)
DRAFTER_IS_MTP_NOT_EAGLE3_387: bool = True           # premise correction for the demand leaf
DEMAND_LADDER_LABEL_387: str = "#289 MTP K=7 conditional-accept ladder (NOT EAGLE-3 tree)"

# --- kanna #374 (djia6icp, merged e051b73, 19:01Z): LANDED — fusion lever CLOSED (Route-A excluded) -- #
# Triton fusion numerics are NOT byte-exact-pinnable, so the fusion lever is closed; capture/land #371
# is the sole identity-safe non-spec leg.  Doesn't touch the composite directly; confirms Route-A
# (sub-int4 UNIFORM eta-axis) stays excluded.
KANNA374_FUSION_LEVER_CLOSED: bool = True            # fusion lever closed (Triton numerics not byte-exact)
FUSION_BYTE_EXACT_PINNABLE_374: bool = False         # Triton fusion is NOT byte-exact-pinnable
CAPTURE_LAND_371_SOLE_IDENTITY_SAFE_NONSPEC_LEG_374: bool = True  # #371 is the only identity-safe non-spec leg
ROUTE_A_STAYS_EXCLUDED_374: bool = True              # confirms Route-A (sub-int4 uniform eta-axis) excluded

# --- cb3-lift DEPLOYABILITY (advisor 19:25Z): TWO of the three 19:01Z contingencies LANDED ----------- #
# supply_base_enables_500 resolves TRUE on paper at the REALISTIC tier (471.42 + honest cb3 +32.65 ->
# ~504; denken #392's combined route reaches 500).  The 19:01Z GO-flip pair has RESOLVED:
#   (a) lawine #391 (3udzpoq8, LANDED 19:11Z): the M=8 served-width contingency — REALISTIC tier HOLDS
#       (cb3 lift +38.02 at M=8 clears +17.2 and +23.75 robust), MEASURED-FLOOR straddles (+15.67 misses
#       the +23.75 robust).  int4-Marlin HBM-eff is FLAT across the verify width (M=8 0.2559 vs M=1
#       0.2578) -> the served MTP-K7 regime does NOT raise efficiency.  realistic-tier-GREEN.
#   (c) denken #392 (2evhfxi7, LANDED 19:12Z): the AUTHORITATIVE honest composed number on the 471.42
#       base — crediting cb3 only to f_verify_body=0.7624, M=1 honest +32.65 off-shelf / +42.91 floor
#       clears BOTH #383 targets; combined route 469.68 -> 512.60, residual demand +0.0117 d-cov (38% of
#       the #336 budget).  USE +32.65 for --supply-lift-available-tps (authoritative honest).
# So the SOLE remaining ANALYTIC gate is now (b) kanna #394 (PPL held-out).  ★ BUT a DEEPER deployability
# layer opened: the cb3 kernel is a SOURCE-BUILD = a FLAGGED served-file change (does NOT ship in vLLM
# 0.22).  Even with kanna #394 GREEN, the on-paper 500 needs a flagged kernel.  The advisor opened 5
# probes for a DEPLOYABLE-WITHOUT-FLAG path to the 28.58 gap (lawine #395 / denken #396 are the GO-flip
# pair; stark #397 / land #398 / ubel #399 sharpen).  NEW GO-flip gate = kanna #394 (analytic, sole) AND
# >= 1 of {lawine #395, denken #396} (deployable-without-flag).  Hold None until BOTH land.
CB3_INSAMPLE_PPL_MARGIN_372: float = 0.039           # #372's in-sample PPL margin kanna #394 stress-tests held-out
KANNA394_HELDOUT_PPL_PENDING: bool = False           # LANDED RED 20:02Z (d184kbey): +0.039 margin is winner's-curse (see 20:19Z block)

# --- lawine #391 (3udzpoq8, merged 19:11Z): LANDED — M=8 served-width contingency (realistic-GREEN) --- #
LAWINE391_LANDED: bool = True                        # M=8 served-width cb3-lift contingency MEASURED
LAWINE391_M8_HBM_EFF: float = 0.2559                 # int4-Marlin weight-read HBM-eff at M=8 (served verify width)
LAWINE391_M1_HBM_EFF: float = 0.2578                 # at M=1 (Δ-0.0020 vs M=8 -> FLAT across width)
LAWINE391_M4_HBM_EFF: float = 0.2590                 # at M=4
CB3_LIFT_M8_REALISTIC_391: float = 38.02             # cb3 lift at M=8, realistic tier (clears +17.2 & +23.75)
CB3_LIFT_M8_MEASURED_FLOOR_391: float = 15.67        # cb3 lift at M=8, measured-floor tier (MISSES +23.75 robust)
CB3_CLOSES_383_ROBUST_M8_391: bool = False           # measured-floor M=8 does NOT clear the +23.75 robust
SUPPLY_LIFT_REQUIRED_ROBUST_TPS_383: float = 23.75   # the robust (E[T]) supply-lift target the floor tier misses
CB3_M8_EFFICIENCY_FLAT_391: bool = True              # served MTP-K7 regime does NOT raise Marlin efficiency
CB3_LANE_REALISTIC_GREEN_FLOOR_YELLOW_391: bool = True  # realistic-tier-GREEN / measured-floor-YELLOW

# --- denken #392 (2evhfxi7, merged 19:12Z): LANDED — AUTHORITATIVE honest composed number on 471.42 --- #
DENKEN392_LANDED: bool = True                        # authoritative draft-separated cb3 composition MEASURED
CB3_LIFT_HONEST_DENKEN392: float = 32.65             # M=1 honest off-shelf lift (AUTHORITATIVE --supply-lift)
CB3_LIFT_FLOOR_DENKEN392: float = 42.91              # M=1 floor-tier lift (clears both #383 targets)
CB3_F_VERIFY_BODY_392: float = 0.7624                # the shrinkable verify-body fraction cb3 is credited to
CB3_388_OPTIMISM_TPS_392: float = 5.69               # #388's +38.3 was +5.69 (~15%) optimistic (lumped draft frac)
COMBINED_ROUTE_REACHES_392: float = 512.60           # combined supply+demand route TPS (469.68 -> 512.60)
COMBINED_ROUTE_RESIDUAL_DCOV_392: float = 0.0117     # residual demand sliver (38% of the +0.031 #336 budget)
COMBINED_ROUTE_REACHES_500_HONEST_392: bool = True   # the combined route reaches 500 on paper (honest)
ET_LADDER_MATCH_REALIZED_PCT_392: float = 0.18       # E[T] ladder matches deployed E_T_REALIZED 3.844 to 0.18%

# --- ubel #389 (fqt33bj3, merged 19:15Z): LANDED — demand leaf ROBUST on measured slope (#386 refuted) - #
UBEL389_LANDED: bool = True                          # measured per-L attention identity floor under VBI=1
UBEL389_MEASURED_FLOOR_VBI1_PCT: float = 0.5764      # measured greedy-identity gap floor under VBI=1 (< 0.633)
UBEL389_PESSIMISTIC_BREACHES_3P2_MEASURED: int = 0   # 0 corners breach 3.2% on the measured slope
UBEL389_ALL_CORNERS_CLEAR_3P2_MEASURED: bool = True  # all corners clear 3.2% (the #386 breach was an artifact)
UBEL389_MEASURED_SLOPE_RATIO_TO_386: float = 0.353   # measured local-penalty slope is 0.353x the #386 interpolation
UBEL389_386_BREACH_REFUTED: bool = True              # the #386 pessimistic-corner breach is REFUTED (artifact)

# --- stark #381 (9edps20u, merged 19:16Z): LANDED — e2e confirms #390 Arm A; rebuilds=1; DON'T flag Marlin #
STARK381_LANDED: bool = True                         # e2e decode-geometry identity confirmation
STARK381_BODY_MARLIN_BITEXACT_M8_E2E: bool = True    # body int4-Marlin bit-exact at the M=8 decode geometry (e2e)
STARK381_RESIDUAL_FLIPS: int = 1                     # sole residual: 1 flip ...
STARK381_RESIDUAL_TOKENS: int = 891                  # ... in 891 tokens
STARK381_KNIFE_EDGE_NAT: float = 0.125               # 0.125-nat knife-edge near-tie (in TRITON_ATTN #375, NOT Marlin)
STARK381_RESIDUAL_IS_KNIFE_EDGE_NEAR_TIE: bool = True  # residual_is_knife_edge_near_tie=True
STARK381_PINNED_REACHES_IDENTITY_1P0: bool = False   # pinned_reaches_identity_1p0=False (the 1-flip residual)
STARK381_RESIDUAL_LOCUS: str = "TRITON_ATTN (#375), NOT int4-Marlin"  # the eta_attn locus wirbel #393 measures
STARK381_DO_NOT_FLAG_MARLIN_REBUILD: bool = True     # rebuild ledger stays 1 (attention only); NOT a Marlin rebuild
STARK381_REBUILD_LEDGER: int = 1                     # confirmed rebuild ledger (attention only)

# --- land #385 (a30iri8i): TANGENTIAL — floor hardened, doesn't move the composite ------------------- #
LAND385_NONSPEC_STRICT_FLOOR_MOVES: bool = False     # 165.44 self-referential-strict (like 481.53); abs floor 91.43
LAND385_SELF_REF_STRICT_FLOOR: float = 165.44        # self-referential-strict floor (like the 481.53 baseline)
LAND385_ABS_BYTE_EXACT_NONSPEC_FLOOR: float = 91.43  # absolute-byte-exact non-spec floor

# --- 19:25Z DEPLOYABILITY-WITHOUT-FLAG probes: the cb3 kernel is a FLAGGED source-build --------------- #
# The cb3 (RHT+VQ QTIP/QuIP#-class) kernel is a SOURCE-BUILD = a flagged served-file change that does NOT
# ship in vLLM 0.22.  So even with kanna #394 GREEN (the cb3 PPL margin holds held-out), the on-paper 500
# needs a kernel behind a flag.  The advisor opened 5 probes for a DEPLOYABLE-WITHOUT-FLAG path to the
# 28.58 gap; fold verdicts in as they land.  GO-flip pair: lawine #395 OR denken #396 (>=1 green -> 500
# reachable with ZERO flagged changes, the cleanest GO).  stark #397 / land #398 / ubel #399 SHARPEN.
CB3_KERNEL_IS_FLAGGED_SOURCE_BUILD: bool = True      # cb3 kernel = source-build, does NOT ship in vLLM 0.22
LAWINE395_SHIPPING_KERNEL_PENDING: bool = True       # does a SHIPPING quant kernel deliver cb3-class read-shrink + identity?
DENKEN396_DEMAND_ALONE_500_PENDING: bool = False     # LANDED RED 20:03Z (yc5ji486): demand-ALONE busts even bare on 467.48 (see 20:19Z block)
STARK397_KNIFE_EDGE_RECOVERY_PENDING: bool = True    # knife-edge identity recovery cheaper than the 11-TPS FA_SLIDING=0?
LAND398_LMHEAD_READ_REDUCTION_PENDING: bool = True   # loadable identity-safe lm_head read-reduction?
UBEL399_DRAFTER_HEADROOM_PENDING: bool = False        # LANDED RED 20:08Z (ec7i3z5t): no cheap deployable demand lever (see 20:19Z block)
DEPLOYABLE_WITHOUT_FLAG_GO_FLIP_PROBES: tuple[str, str] = (  # >=1 green -> deployable-without-flag GO
    "lawine#395_shipping_kernel_cb3class_readshrink", "denken#396_demand_alone_500_in_budget")
DEPLOYABLE_WITHOUT_FLAG_SHARPEN_PROBES: tuple[str, str, str] = (
    "stark#397_knife_edge_identity_recovery", "land#398_lmhead_read_reduction",
    "ubel#399_drafter_acceptance_headroom")
# NEW terminal GO-flip gate (supersedes the 19:01Z #391+#394 pair): kanna #394 (analytic, sole) AND >= 1
# of {lawine #395, denken #396} (deployable-without-flag).  #391/#392 LANDED (no longer pending gates).
CB3_LIFT_DEPLOYABILITY_GATES: tuple[str, str] = (    # the SUPERSEDED 19:25Z GO-flip structure (see 20:19Z block)
    "kanna#394_heldout_ppl_analytic", "lawine#395_or_denken#396_deployable_without_flag")

# --- 20:19Z ★★ DECISION-CRITICAL re-pricing: the coupled cb3-PPL + demand cluster LANDED and moves ---- #
# BOTH legs of the combined route the wrong way.  The "clean zero-flag GO" path (demand-alone + the cheap
# demand sliver) the advisor flagged at 19:25Z is CLOSED; the supply leaf's +32.65 HEADLINE number is
# PPL-DEAD.  Re-base on the corrected 467.48 and treat the real supply/demand numbers as PENDING three new
# feeder cards.  The route is NOT closed — it is RE-PRICED toward "conservative-k cb3 + a NET-positive
# tree"; that is exactly what kanna #403 / ubel #401 / denken #402 will resolve.  HOLD None (sharpen, not flip).
#
# (0) wirbel #393 (0q7ynumg, merged 19:48Z): BASE CORRECTION.  The decode-specific attention strict tax is
#     3.01% (decode band [528,658] on the rising un-pack curve), LARGER than the #378 eval-weighted 2.15%
#     (+0.86pp).  Realized deployed strict moves 471.42 -> 467.48; ceiling 509.78 -> 505.29; gap-to-500
#     widens 28.58 -> 32.52.  Attention is the SOLE strict tax and irreducible rebuild-free (attn_eta_
#     reducible=False; FlashInfer-BI MEASURED-FALSE; pinned-K is a flagged rebuild).  Re-base on 467.48.
WIRBEL393_LANDED: bool = True                        # decode-specific attention strict tax MEASURED (0q7ynumg)
REALIZED_DEPLOYED_STRICT_393: float = 467.48         # CORRECTED supply BASE (supersedes #390's 471.42)
SHIPPABLE_CEILING_393: float = 505.29                # corrected shippable ceiling (was 509.78)
GAP_TO_500_393: float = 32.52                        # bare gap_to_500 on the corrected base (500 - 467.48)
DECODE_ATTN_STRICT_TAX_PCT_393: float = 3.01         # decode-specific attention strict tax (decode band [528,658])
EVAL_WEIGHTED_ATTN_STRICT_TAX_PCT_378: float = 2.15  # the #378 eval-weighted tax (SMALLER; the band was mislabeled)
DECODE_TAX_DELTA_PP_393: float = 0.86                # +0.86pp decode-vs-eval-weighted (3.01 - 2.15)
DECODE_BAND_393: tuple[int, int] = (528, 658)        # the decode band on the rising un-pack curve
ATTN_ETA_REDUCIBLE_393: bool = False                 # attention strict tax is irreducible rebuild-free
ATTN_SOLE_STRICT_TAX_393: bool = True                # attention is the SOLE strict tax (FlashInfer-BI MEASURED-FALSE)
BASE_390_SUPERSEDED_BY_393: float = 471.42           # the #390 base #393 corrects (kept for provenance)

# (a) kanna #394 (d184kbey, merged 20:02:35Z): the supply leaf's HEADLINE number is PPL-DEAD.  #372's
#     in-sample gate PPL reproduces exactly (2.3816) but its +0.039 margin is WINNER'S-CURSE — disjoint-
#     split selection (3 seeds) chases the 2.42 ceiling to k=243-246 and the held-out worst-seed (2.4223)
#     AND OOD ShareGPT (2.4270) both BREACH 2.42 -> cb3_supply_deployable=False AT THE HEADLINE LIFT.  The
#     +32.65 honest realistic-tier number is NOT PPL-deployable.  BUT k=232 itself still clears (~2.39
#     held-out) -> cb3 IS deployable at a more conservative k, at a smaller (UN-COSTED) lift.  Drop
#     --supply-lift-available-tps 32.65; the real PPL-safe supply number is PENDING kanna #403.
KANNA394_LANDED: bool = True                         # held-out PPL stress-test MEASURED (d184kbey)
KANNA394_INSAMPLE_PPL_REPRO_372: float = 2.3816      # #372's in-sample gate PPL reproduces exactly
KANNA394_MARGIN_IS_WINNERS_CURSE: bool = True        # the +0.039 in-sample margin is winner's-curse
KANNA394_SELECTED_K_RANGE: tuple[int, int] = (243, 246)  # disjoint-split selection chases the ceiling here
KANNA394_HELDOUT_WORST_SEED_PPL: float = 2.4223      # held-out worst-seed BREACHES 2.42
KANNA394_OOD_SHAREGPT_PPL: float = 2.4270            # OOD ShareGPT BREACHES 2.42
KANNA394_PPL_GATE: float = 2.42                      # the PPL ceiling both breach
CB3_SUPPLY_DEPLOYABLE_AT_HEADLINE_394: bool = False  # +32.65 headline lift is NOT PPL-deployable
CB3_HEADLINE_LIFT_PPL_DEAD_394: bool = True          # the +32.65 honest realistic-tier number is PPL-dead
KANNA394_CONSERVATIVE_K_232: int = 232               # k=232 still clears (~2.39 held-out)
KANNA394_CONSERVATIVE_K_HELDOUT_PPL: float = 2.39    # held-out PPL at the conservative k (clears 2.42)
CB3_DEPLOYABLE_AT_CONSERVATIVE_K_394: bool = True    # cb3 IS deployable at a conservative k (smaller, un-costed lift)
CB3_CONSERVATIVE_K_LIFT_PENDING_KANNA403: bool = True  # the real PPL-safe lift is pending kanna #403

# (b) denken #396 (yc5ji486, merged 20:03:51Z): demand-ALONE is INSUFFICIENT; the clean zero-flag GO path
#     is CLOSED.  On the bare 471.42 base required_dcov=+0.02946 (94.9% of the +0.031 budget) — fits bare,
#     but the minimum #389 private attn-identity floor (0.5764%) pushes it to 0.03244 > 0.031 ->
#     robust_under_389_slope=False.  And on the CORRECTED 467.48 base required_dcov rises to ~0.0338 (109%
#     of budget) -> demand-alone busts EVEN BARE.  The combined supply+demand route is the only robust plan.
DENKEN396_LANDED: bool = True                        # demand-alone-500 budget check MEASURED (yc5ji486)
DENKEN396_DEMAND_ALONE_500_GREEN: bool = False       # demand-alone does NOT reach 500 within budget
DENKEN396_REQUIRED_DCOV_BARE_471: float = 0.02946    # required Δcov on bare 471.42 (94.9% of budget — fits bare)
DENKEN396_REQUIRED_DCOV_BARE_471_BUDGET_FRAC: float = 0.949
DENKEN396_REQUIRED_DCOV_UNDER_389_FLOOR: float = 0.03244  # under the #389 floor -> > 0.031 (NOT robust)
DENKEN396_ROBUST_UNDER_389_SLOPE: bool = False       # the #389 floor pushes it over budget -> not robust
DENKEN396_REQUIRED_DCOV_ON_467: float = 0.0338       # required Δcov on the corrected 467.48 base
DENKEN396_REQUIRED_DCOV_ON_467_BUDGET_FRAC: float = 1.09  # 109% of the +0.031 budget -> busts EVEN BARE
DENKEN396_DEMAND_ALONE_BUSTS_EVEN_BARE_467: bool = True   # demand-alone busts even bare on the corrected base
DENKEN396_ZERO_FLAG_GO_PATH_CLOSED: bool = True      # the 19:25Z "demand-alone -> 500 zero-flag" branch is NEGATIVE
BUDGET_336_PLUS_031: float = 0.031                   # the #336 +0.031 coverage budget envelope

# (c) ubel #399 (ec7i3z5t, merged 20:08:20Z): there is NO cheap deployable demand lever; the d-cov needs a
#     RETRAIN or a TREE.  Every monotone draft-head lever (temperature, affine calibration) is a RANK-
#     INVARIANT no-op (MC max|Δcov|=0.00e+00; a rank-changing per-class-bias control fires at 0.59 -> the
#     zeros are physics, not a broken harness); frac_of_28p58_gap_covered=0%.  The demand d-cov (#392's
#     +0.0117 sliver and anything beyond) can ONLY come from a FORBIDDEN drafter retrain or a TREE-verify
#     kernel rebuild (the locked +0.1286 top-1->top-4 prize; coverage gap [0,0.1097] is an UPPER bound).
#     PPL is untouched throughout (spec-decode emits the target's greedy token) — the binding constraint is
#     DEPLOYABILITY, never the gate.
UBEL399_LANDED: bool = True                          # draft-head demand-lever sweep MEASURED (ec7i3z5t)
UBEL399_CHEAP_DEMAND_LEVER_EXISTS: bool = False      # no cheap deployable demand lever
UBEL399_MONOTONE_LEVERS_RANK_INVARIANT: bool = True  # temperature / affine calibration are rank-invariant no-ops
UBEL399_MC_MAX_DCOV: float = 0.0                     # MC max|Δcov| = 0.00e+00 (the zeros are physics)
UBEL399_RANK_CHANGING_CONTROL_FIRES: float = 0.59    # a rank-changing per-class-bias control fires at 0.59 (harness OK)
UBEL399_FRAC_OF_GAP_COVERED: float = 0.0             # frac_of_28p58_gap_covered = 0%
UBEL399_DCOV_ONLY_FROM_RETRAIN_OR_TREE: bool = True  # d-cov requires a forbidden retrain or a tree-verify rebuild
UBEL399_TREE_TOP1_TO_TOP4_PRIZE: float = 0.1286      # the locked top-1->top-4 coverage prize (tree-verify)
UBEL399_COVERAGE_GAP_UPPER_BOUND: tuple[float, float] = (0.0, 0.1097)  # coverage gap [0, 0.1097] is an UPPER bound
UBEL399_PPL_UNTOUCHED: bool = True                   # PPL untouched (spec-decode emits target greedy token)

# NEW feeder cards (20:19Z): the route is re-priced toward "conservative-k cb3 + a NET-positive tree".
# kanna #403 = the PPL-safe conservative-k supply re-cost (largest k with held-out worst-seed <= 2.41 ->
# re-costed lift).  ubel #401 = the locked top-8/16 tree coverage ceiling (sizes the +0.1286 prize).
# denken #402 = whether the tree NETs that d-cov after its verify-M step-time tax on 467.48.  The NEW
# terminal GO-flip gate = kanna #403 (PPL-safe supply) AND >= 1 of {ubel #401, denken #402} (tree net-supply).
KANNA403_PPL_SAFE_SUPPLY_PENDING: bool = True        # PPL-safe conservative-k supply re-cost (the real supply number)
KANNA403_HELDOUT_WORST_SEED_TARGET: float = 2.41     # largest k with held-out worst-seed <= 2.41 -> re-costed lift
UBEL401_TREE_COVERAGE_CEILING_PENDING: bool = True   # locked top-8/16 tree coverage ceiling (sizes the +0.1286 prize)
DENKEN402_TREE_NET_SUPPLY_PENDING: bool = True       # does the tree NET d-cov after its verify-M step-time tax on 467.48?
REPRICE_2019Z_GO_FLIP_GATES: tuple[str, str] = (     # the NEW (20:19Z) terminal GO-flip structure
    "kanna#403_ppl_safe_conservative_k_supply", "ubel#401_or_denken#402_tree_net_supply")
REPRICE_2019Z_TREE_NET_SUPPLY_PROBES: tuple[str, str] = (  # the demand-leg tree feeders (>=1 green)
    "ubel#401_tree_top8_16_coverage_ceiling", "denken#402_tree_nets_dcov_after_verifyM_tax")
REPRICE_2019Z_SUPERSEDES_1925Z_PAIR: bool = True     # kanna #394 RED + denken #396 RED closed the 19:25Z pair
TREE_IS_GENUINELY_NEW_LEVER_2019Z: bool = True       # tree/retrain work = the genuinely-new-lever requirement

# --------------------------------------------------------------------------- #
# 22:08Z ADVISOR RE-POINT (GitHub #357): convert the binary ">500 reachable?"
# rollup into ONE max-equivalent-TPS frontier rollup over the equivalence ladder.
# The #407 human directive (21:13Z) DROPPED 500 as the target; the live objective
# is now max TPS subject to STRICT byte-exact greedy-token-equivalence (served
# identity == 1.0).  The strict_500_reachable_via_known_levers=0 result (run
# 34tv4krw) is kept ONLY as a historical sub-result, NOT the capstone.
# Tag each node MEASURED vs MODELED; keep per-lever PPL/identity/deployability gates.
# --------------------------------------------------------------------------- #
EQUIV_FRONTIER_REPOINTED_2208Z: bool = True

# The deployed 481.53 fast path is NOT on the equivalence ladder: served identity
# 0.9966 (3/882 reduction-order flips under M=8 batched verify, stark #381/#405).
DEPLOYED_FASTPATH_TPS: float = 481.53
DEPLOYED_FASTPATH_SERVED_IDENTITY: float = 0.9966
DEPLOYED_FASTPATH_IS_EQUIVALENT: bool = False
DEPLOYED_FASTPATH_FLIPS: tuple[int, int] = (3, 882)   # reduction-order flips / total under M=8

# Ladder node 0 — FLOOR (MEASURED): blanket-strict batch-invariant attention every
# step, served identity 1.0 (wirbel #393, = REALIZED_DEPLOYED_STRICT_393).
EQUIV_FLOOR_TPS: float = REALIZED_DEPLOYED_STRICT_393   # 467.48 (MEASURED, identity 1.0)
EQUIV_FLOOR_MEASURED: bool = True
EQUIV_FLOOR_IDENTITY: float = 1.0

# Ladder node 1 — SELECTIVE HIGHER-PRECISION RECOMPUTE (#397, stark #412 MEASURING).
# Fast attention everywhere + recompute ONLY the ~23.6% near-tie <=eps-flagged steps
# at higher precision -> equivalence restored by construction (identity 1.0).  Tie is
# readable from the fast path (stark #405 tie_identifiable_from_fast_path=True).
# MODELED tax ~2.6 TPS off the 481.53 fast path -> ~478.93 within band [476,479].
SELECTIVE_RECOMPUTE_BAND_TPS: tuple[float, float] = (476.0, 479.0)
SELECTIVE_RECOMPUTE_MODELED_TPS: float = 478.93        # 481.53 - 2.6 (MODELED; stark #412 measuring)
SELECTIVE_RECOMPUTE_TAX_TPS: float = 2.6
SELECTIVE_RECOMPUTE_FLAGGED_FRAC: float = 0.236
SELECTIVE_RECOMPUTE_MEASURED: bool = False             # stark #412 building local research prototype on A10G
SELECTIVE_RECOMPUTE_TIE_IDENTIFIABLE_405: bool = True
SELECTIVE_RECOMPUTE_IDENTITY: float = 1.0              # by construction (recompute restores the flagged ties)

# Ladder node 2 — cb3 BODY-READ SHRINK supply (kanna #403, BANKED/merged 440fd484).
# RHT+VQ QTIP/QuIP#-class kernel: changes BYTES READ in the verify body, NOT tokens
# emitted -> equivalence-NEUTRAL.  Conservative k*=229 holds held-out worst-seed
# <=2.41.  +15.60 TPS, residual headroom 16.93, W&B iv9i2wks, self-test 30/30.
CB3_CONSERVATIVE_LIFT_TPS_403: float = 15.60
CB3_403_K_STAR: int = 229
CB3_403_PPL_SAFE: bool = True
CB3_403_RESIDUAL_HEADROOM_TPS: float = 16.93
CB3_403_EQUIVALENCE_NEUTRAL: bool = True
CB3_403_BANKED: bool = True
CB3_403_WANDB: str = "iv9i2wks"
# 22:26Z: cb3 onto the recompute point is CONFIRMED ADDITIVE (lawine #417), so the
# stack is BANKED as the modeled headline.  kanna #416 still PRICES the exact
# additivity_gap_tps but only TIGHTENS the [492.08, 494.08] bracket — it no longer
# GATES banking.  (Flag retained for the report's "exact gap pending" line.)
CB3_ADDITIVITY_PENDING_KANNA416: bool = True   # exact gap pending; additivity itself CONFIRMED (lawine #417)

# Ladder node 3 — FIXED-OVERHEAD-FLOOR reduction (wirbel #415 DECOMPOSING).
# Decomposes the 146.30us / 12.01% fixed floor measured in #408 (qc9bz8sv).  Treat
# as 0 TPS until wirbel #415 lands a measured reduction.
FIXED_OVERHEAD_FLOOR_US_408: float = 146.30
FIXED_OVERHEAD_FLOOR_FRAC_408: float = 0.1201
FLOOR_REDUCTION_PENDING_WIRBEL415: bool = True
FLOOR_REDUCTION_TPS_UNTIL_415: float = 0.0

# Deployability surface feeder: lawine #417 (PENDING) -> gates whether the modeled
# frontier points are actually deployable (kernel build / served-file change).
DEPLOYABILITY_SURFACE_PENDING_LAWINE417: bool = True

# Tree net-supply leg — CLOSED (denken #409, BANKED, W&B 3zr7i8ad, self-test 42/42).
# DP-optimal verify-tree widths w*=(3,2,2,1,1,1,1) at M=12 net only +1.33 TPS and are
# beta-fragile -> treat the tree leg as ~0 reliable equivalence-neutral supply.
# SUPERSEDES the 20:19Z ubel #401 / denken #402 tree feeders.
DENKEN409_TREE_NET_TPS: float = 1.33
DENKEN409_TREE_BETA_FRAGILE: bool = True
DENKEN409_TREE_RELIABLE_SUPPLY_TPS: float = 0.0
DENKEN409_DP_WIDTHS: tuple[int, ...] = (3, 2, 2, 1, 1, 1, 1)
DENKEN409_M: int = 12
DENKEN409_SUPERSEDES_401_402: bool = True
DENKEN409_BANKED: bool = True
DENKEN409_WANDB: str = "3zr7i8ad"

# --------------------------------------------------------------------------- #
# 22:26Z ADVISOR FORMAL RELAY (GitHub #357) — TWO MORE FEEDERS BANKED.
# Consume these nodes, DO NOT re-derive.  They move the MODELED equivalent
# frontier ABOVE the non-strict deployed 481.53 WITH the byte-identity guarantee
# that 481.53 lacks.  ("Your re-pointed ladder now reads ... the headline
# equivalent number, beats 481.53.")
# --------------------------------------------------------------------------- #
EQUIV_FRONTIER_FEEDERS_2226Z: bool = True

# lawine #417 BANKED/merged (commit 8c80b5d, W&B 2mv6ssw4, self-test 63/63):
#   (1) cb3 +15.60 is ADDITIVE onto the selective-recompute point — CONFIRMED.  The
#       cb3 stack is no longer "modeled-only / do-not-sum"; kanna #416 now only PRICES
#       the exact additivity_gap_tps (a TIGHTENING of the bracket, NOT a gate on banking).
#   (2) deploy surface PRICED: 7 served files, 41.8 GPU-min identity-verify (shared-e2e
#       SURVIVES the in-place edit, NOT the naive 81.6); whole-stack reversible, 1 binding
#       in-place line, human-gated -> DEPLOYABLE (banked-green).
#   (3) modeled fastest_equivalent_tps BRACKET [492.08, 494.08] = selective-recompute
#       (~476.48-478.48) + cb3 +15.60 -> the HEADLINE equivalent number; BEATS the
#       non-strict deployed 481.53 with the byte-identity guarantee 481.53 lacks.
LAWINE417_BANKED: bool = True
LAWINE417_WANDB: str = "2mv6ssw4"
LAWINE417_CB3_ADDITIVE_CONFIRMED: bool = True            # cb3 stack is additive -> BANK it (not pending-only)
LAWINE417_DEPLOY_SURFACE_FILES: int = 7
LAWINE417_DEPLOY_IDENTITY_VERIFY_GPU_MIN: float = 41.8   # shared-e2e survives the edit (NOT naive 81.6)
LAWINE417_DEPLOY_REVERSIBLE: bool = True
LAWINE417_DEPLOY_BINDING_INPLACE_LINES: int = 1
LAWINE417_DEPLOYABLE_GREEN: bool = True
FASTEST_EQUIVALENT_BRACKET_TPS: tuple[float, float] = (492.08, 494.08)   # lawine #417 banked HEADLINE
FASTEST_EQUIVALENT_BRACKET_BEATS_DEPLOYED: bool = True   # 492.08 (lower bound) > 481.53 deployed-nonequiv

# land #414 BANKED/merged (commit 09166a3, W&B bq7xkfcv, self-test 29/29):
#   operative equivalence is SELF-REFERENTIAL — the submission's OWN 16384-row truncated-
#   head greedy.  Deployed config + ANY in-keepset speculator pass it FOR FREE
#   (deployed_passes_self_referential=True, 0 cost).  lm_head truncation is therefore FREE
#   -> keep it OFF the equivalent-frontier cost ledger.  ABSOLUTE full-vocab equivalence is
#   STRONGER and NOT required: a 261,976-row head (#406 cert: 245,592 pruned rows globally
#   reachable) costs 54.07 TPS — a CONTINGENCY line only, never on the operative ladder.
LAND414_BANKED: bool = True
LAND414_WANDB: str = "bq7xkfcv"
LAND414_SELF_REFERENTIAL_GATE: bool = True
LAND414_DEPLOYED_PASSES_SELF_REFERENTIAL: bool = True
LAND414_LMHEAD_TRUNCATION_TPS_COST: float = 0.0           # FREE under the self-referential gate
LAND414_TRUEVOCAB_LMHEAD_TPS_COST_CONTINGENCY: float = 54.07   # absolute full-vocab head (NOT required)
LAND414_TRUEVOCAB_HEAD_ROWS: int = 261_976
LAND414_PRUNED_ROWS_GLOBALLY_REACHABLE_406: int = 245_592

# denken #413 (W&B se8mf9ax): selective-recompute modeled POINT 478.93 = 481.53 - 2.6 @ M=8.
# lawine #417's combined bracket uses the recompute bracket [476.48, 478.48] (its upper is
# +0.45 below the denken #413 point, so the naive additive point 494.53 sits just ABOVE the
# banked bracket upper 494.08 — the bracket is the conservative banked range).  denken #418
# is testing whether the tax is < 2.6 via per-position asymmetry (would RAISE the bracket).
DENKEN413_RECOMPUTE_POINT_TPS: float = 478.93
DENKEN413_WANDB: str = "se8mf9ax"
SELECTIVE_RECOMPUTE_BRACKET_417_TPS: tuple[float, float] = (476.48, 478.48)
DENKEN418_TESTING_TAX_LT_2P6: bool = True

# Remaining parameterized refinements (measured numbers drop straight in, modeled until then):
#   stark #412  -> MEASURED selective-recompute equivalent-TPS + identity 1.0 on A10G
#   kanna #416  -> exact cb3 additivity_gap_tps (TIGHTENS the [492.08, 494.08] bracket)
#   wirbel #415 -> MEASURED fixed-overhead-floor reduction TPS (146.30us/12.01% floor, #408)
#   denken #418 -> tax < 2.6 via per-position asymmetry (would RAISE the recompute point/bracket)
#   lawine #419 -> executable deploy GO/NO-GO + verify CI + feature-flag
#   land   #420 -> speculator equivalence by construction + in-keepset acceptance upside

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
            "9.841% (> 4.02% budget) is superseded. stark #365 measures lmhead_bi_gemm_eta (pending). "
            "NOTE: the eta-AS-SPEEDUP-lever interpretation is DEFLATED (advisor 16:41Z); identity is now "
            "treated as a strict-lock COMPLIANCE factor whose reachability/cost is governed by "
            "identity_reachability_analysis (stark #376 RED + reseated #381)."
        ),
    }


# --------------------------------------------------------------------------- #
# IDENTITY as a strict-lock COMPLIANCE factor (stark #376 RED + reseated #381, advisor 16:41Z).
# Strict greedy-token-identity is a HARD gate on ANY served config (any route, incl. the demand-side
# coverage route).  The question is no longer "is the lm_head eta within a 4.02% speedup budget"
# (that eta-as-lever read is DEFLATED) but: is byte-exact identity REACHABLE on the served decode-
# verify path, and at what DEPLOYMENT cost?  stark #363 made the attention locus FREE; stark #376
# found the residual ~0.73% flip on REAL weights is the int4-Marlin BODY GEMM (a custom CUDA op the
# VLLM_BATCH_INVARIANT env knob cannot patch) — but ONLY at the prefill-replication geometry (2048
# rows); Marlin is BIT-EXACT at the decode-verify width (8 rows).  stark #381 resolves the served
# 8-row geometry.  In BOTH outcomes identity is REACHABLE; #381 only sets the cost (1 vs 2 rebuilds).
# --------------------------------------------------------------------------- #
def identity_reachability_analysis(stark381_decode: str = "pending") -> dict[str, Any]:
    """Two-branch identity REACHABILITY on the served path (env-reachable@decode vs Marlin-rebuild-gated).

    `stark381_decode` (stark #381 decode-width direct e2e identity): one of
        "pending" | "green" (identity==1.0 at the 8-row served width) | "red" (still M-variant @8 rows).

    Returns identity_reachable=True in BOTH resolved branches (byte-exact compliance is achievable);
    the pending flag governs only the deployment COST branch (number of kernel rebuilds), NOT
    reachability.  GREEN -> env-reachable@decode (1 rebuild: #375 mha_varlen attention only).
    RED -> Marlin-rebuild-gated (2 rebuilds: #375 attention + a fixed-split-K int4 Marlin/Machete GEMM).
    """
    pending = stark381_decode == "pending"
    green = stark381_decode == "green"
    red = stark381_decode == "red"

    # Attention locus is FREE (stark #363); the open locus is the int4-Marlin body GEMM (stark #376).
    # Both #381 outcomes leave identity REACHABLE — the difference is the rebuild line-item count.
    if green:
        cost_branch = "env_reachable_at_decode_width"
        rebuild_items = ["#375_mha_varlen_attention_rebuild"]
        marlin_rebuild_needed = False
    elif red:
        cost_branch = "marlin_rebuild_gated"
        rebuild_items = ["#375_mha_varlen_attention_rebuild",
                         "fixed_split_k_int4_marlin_machete_gemm_rebuild"]
        marlin_rebuild_needed = True
    else:
        cost_branch = "pending_stark381_decode_width"
        rebuild_items = None
        marlin_rebuild_needed = None

    rebuild_line_items = None if rebuild_items is None else len(rebuild_items)

    return {
        # identity is REACHABLE on the served path in BOTH branches (env or rebuild) — TRUE even
        # while #381 is pending, because both resolved outcomes are reachable.
        "identity_reachable_env_or_rebuild": True,
        "cost_branch": cost_branch,
        "stark381_decode_input": stark381_decode,
        "cost_branch_pending": pending,
        "rebuild_line_items": rebuild_line_items,
        "rebuild_items": rebuild_items,
        "marlin_rebuild_needed": marlin_rebuild_needed,
        # stark #363 attention locus (banked FREE)
        "eta_attn_stark363": ETA_ATTN_STARK363,
        "attn_locus_free_stark363": True,
        # stark #376 RED measurement (banked) at the prefill-replication geometry
        "identity_pinned_e2e_stark376": IDENTITY_PINNED_E2E_STARK376,
        "identity_heuristic_e2e_stark376": IDENTITY_HEURISTIC_E2E_STARK376,
        "identity_residual_flip_stark376": IDENTITY_RESIDUAL_FLIP_STARK376,
        "pin_does_not_help_stark376": abs(
            IDENTITY_PINNED_E2E_STARK376 - IDENTITY_HEURISTIC_E2E_STARK376) < 1e-3,
        "residual_is_int4_marlin_body_gemm": True,
        "env_knob_cannot_patch_marlin": True,   # VLLM_BATCH_INVARIANT is aten-dispatcher-only
        # geometry caveat — the RED is geometry-specific
        "decode_verify_width_rows": DECODE_VERIFY_WIDTH_ROWS,
        "prefill_replication_width_rows": PREFILL_REPLICATION_WIDTH_ROWS,
        "marlin_bit_exact_at_decode_width": True,   # 8 rows -> bit-exact (stark size_m sweep)
        "marlin_m_variant_at_prefill_width": True,   # 2048 rows -> M-variant (#376 RED geometry)
        "red_is_geometry_specific": True,
        # wirbel #384 (4f32ks1e, 18:12Z): the deployed lm_head — itself an int4-Marlin GEMM (n=16384) —
        # is byte-exact at the 8-row decode width (should_use_atomic_add_reduce(M=8)=False), an existence
        # proof for the SAME atomic-add mechanism #381 is testing on the body.  Corrected strict ledger:
        # 2 rebuilds (attn #375 + body #376); lm_head shares the body kernel and is FREE (0 rebuilds).
        "lmhead_int4_marlin_free_at_decode_wirbel384": WIRBEL384_LMHEAD_FREE,
        "n_kernel_rebuilds_strict_500_wirbel384": N_KERNEL_REBUILDS_STRICT_500_384,
        "note": (
            "stark #376 (ipe3ofie, RED): on real weights, pinning the attention split "
            f"(num_splits=1) leaves e2e identity {IDENTITY_PINNED_E2E_STARK376} ~= heuristic "
            f"{IDENTITY_HEURISTIC_E2E_STARK376}; the residual ~{IDENTITY_RESIDUAL_FLIP_STARK376*100:.2f}% "
            "flip is the int4-Marlin body GEMM (custom CUDA op outside the aten dispatcher -> "
            "VLLM_BATCH_INVARIANT cannot patch it). BUT this is the PREFILL-REPLICATION geometry "
            f"({PREFILL_REPLICATION_WIDTH_ROWS} rows); Marlin is BIT-EXACT at the decode-verify width "
            f"({DECODE_VERIFY_WIDTH_ROWS} rows). wirbel #384 (18:12Z) is the existence proof: the deployed "
            "lm_head (an int4-Marlin GEMM, n=16384) is already byte-exact at the 8-row decode width via "
            "should_use_atomic_add_reduce(M=8)=False — the SAME atomic-add mechanism #381 tests on the body. "
            f"Corrected ledger = {N_KERNEL_REBUILDS_STRICT_500_384} rebuilds (attn #375 + body #376), lm_head "
            "FREE. stark #381 resolves the body geometry: GREEN -> body also free at decode -> 1 rebuild "
            "(#375 mha_varlen only); RED -> Marlin-rebuild-gated (2 rebuilds). Identity is REACHABLE in "
            "BOTH; #381 only sets the deployment cost."
        ),
    }


# --------------------------------------------------------------------------- #
# SUPPLY-side honest deployable-strict base (wirbel #378, advisor-relayed 17:03Z) — the NOW-BINDING
# GO leaf.  The demand closer transfers FROM a public-strict serve base; wirbel #378 measured that
# the honest deployable-strict base is <=480.7-today (floor 469.68 off-the-shelf + ~11-TPS attn
# rebuild), NOT the 518.92 eta-axis pin.  Since demand-side gap-closure can at best drive rho->1
# (private <= public), a public base < 500 may make private-500 UNREACHABLE by demand-side coverage
# retrain ALONE at any coverage — a structurally different verdict from #377's "+5.44 TPS, in budget"
# (premised on 518.92).  denken #383 (RED, BANKED 17:53Z) CONFIRMED demand-alone insufficient on the
# honest base -> a +17.2-TPS supply lift is required FIRST.  wirbel #384 (18:12Z) REFUTED the lm_head-BI
# determinization lever (lm_head already byte-exact int4-Marlin, eta=0, FREE; tax in body #376), so the
# supply base is HELD pending a MEASURED lift from wirbel #390 (corrected SHIPPABLE strict ceiling) OR
# lawine #388 (realized TPS of lawine #372's GREEN mixed-precision body allocation).  Identity
# reachability (stark #376/#381) folds in here as the strict-byte-exact
# COMPLIANCE prerequisite of this base (the VBI=1 strict knob + the #375 attn rebuild).
# --------------------------------------------------------------------------- #
def supply_side_base_analysis(supply_lift_available_tps: float | None = None,
                              demand_reaches_500_on_floor: str = "pending",
                              supply_lift_required_tps: float | None = None,
                              stark381_decode: str = "pending",
                              kanna403_ppl_safe_supply: str = "pending") -> dict[str, Any]:
    """Honest deployable-strict supply base (wirbel #378), the BINDING GO axis (advisor 17:03Z/17:53Z).

    denken #383 (merged 17:53Z) is the decisive honest re-price: demand-alone does NOT reach
    private-strict-500 on the honest base, so a SUPPLY lift of +17.2 TPS (floor-joint) is required
    FIRST.  wirbel #384 (18:12Z) then REFUTED the "lm_head-BI determinization" supply lever: the
    deployed lm_head is already byte-exact int4-Marlin (eta=0, FREE) and the dominant non-attention
    strict tax lives in the int4-Marlin BODY (#376).  The leaf is HELD until a measured supply number
    lands from EITHER de-risker:
      - wirbel #390 : corrected SHIPPABLE deployable-strict ceiling (re-rolls the refuted 510.01/518.92), OR
      - lawine #388 : realized TPS of lawine #372's GREEN mixed-precision BODY allocation (-21.5% body read).

    `supply_lift_available_tps`: the best MEASURED supply lift available from wirbel #390 OR lawine #388
        (TPS).  None -> both supply de-riskers pending.
    `demand_reaches_500_on_floor` (denken #383 `demand_route_reaches_500_on_deployable_floor`): one of
        "pending" | "yes" (demand closes 500 on the honest deployable floor) | "no" (a supply lift is
        required FIRST).  Banked default "no" (denken #383 RED).
    `supply_lift_required_tps` (denken #383 `supply_lift_required_first_tps`): the supply lift the base
        needs before the demand closer can reach 500.  Banked default +17.2 (floor-joint).
    `stark381_decode`: identity decode-width compliance branch (folds in as the strict prerequisite).

    Returns `supply_base_enables_500` True/False/None: whether the supply base (after any measured
    supply lift) reaches a point from which the demand closer can clear private-500.
    """
    floor = HONEST_STRICT_BASE_FLOOR_378            # 469.68 off-the-shelf (VBI=1) — #378 estimate (provenance)
    plus_attn = HONEST_STRICT_BASE_PLUS_ATTN_378    # 480.7 (floor + ~11-TPS attn) — #378 estimate (superseded)
    # wirbel #390 (5y64zbjz, 19:01Z) measured 471.42; wirbel #393 (0q7ynumg, 20:19Z) then CORRECTED it:
    # the DECODE-specific attention strict tax is 3.01% (decode band [528,658]), LARGER than the #378
    # eval-weighted 2.15% (+0.86pp), so the realized deployed strict moves 471.42 -> 467.48 (gap_to_500
    # widens 28.58 -> 32.52; ceiling 509.78 -> 505.29).  Attention is the SOLE strict tax and irreducible
    # rebuild-free (attn_eta_reducible=False; FlashInfer-BI MEASURED-FALSE; pinned-K is a flagged rebuild).
    base_today = REALIZED_DEPLOYED_STRICT_393        # 467.48 CORRECTED supply BASE (#393, supersedes #390's 471.42)
    band_lo, band_hi = DEPLOYABLE_STRICT_BAND_378
    clears_500_today = base_today >= TARGET          # False (467.48 < 500)
    deficit_to_500_today = TARGET - base_today       # 32.52 TPS short (matches #393 gap_to_500)

    # cb3 supply lift.  denken #392 (LANDED) sized the HEADLINE honest composed number +32.65 off-shelf
    # (crediting cb3 only to f_verify_body=0.7624).  ★ 20:19Z: kanna #394 (LANDED RED) made that HEADLINE
    # number PPL-DEAD — #372's +0.039 in-sample margin is winner's-curse; the held-out worst-seed (2.4223)
    # AND OOD ShareGPT (2.4270) BREACH the 2.42 gate at the selected k=243-246.  So the +32.65 is NOT
    # PPL-deployable; DO NOT carry it.  BUT k=232 still clears (~2.39 held-out) -> cb3 IS deployable at a
    # more conservative k, at a smaller (UN-COSTED) lift -> the REAL PPL-safe supply number is PENDING
    # kanna #403.  The supply leaf resolves on kanna #403, not the dead +32.65 headline.
    cb3_lift_honest = CB3_LIFT_HONEST_DENKEN392      # +32.65 HEADLINE — PPL-DEAD (kanna #394 RED); provenance only
    cb3_headline_lift_ppl_dead = CB3_HEADLINE_LIFT_PPL_DEAD_394    # True (do NOT carry +32.65)
    cb3_lift_honest_388 = CB3_LIFT_HONEST_TPS_388    # +33 (lawine #388 draft-separated; superseded by #392/#394)
    cb3_lift_realistic = CB3_LIFT_REALISTIC_TPS_388  # +38.34 (#388, lumps the un-shrunk draft fraction)
    lift_for_paper = supply_lift_available_tps if supply_lift_available_tps is not None else 0.0
    base_plus_cb3_honest = base_today + cb3_lift_honest        # 500.13 on the 467.48 base — but PPL-DEAD
    base_plus_cb3_realistic = base_today + cb3_lift_realistic  # 505.82 — also rests on the PPL-dead headline
    # The +32.65 numerically clears 500 on the corrected base, but it is PPL-DEAD -> NOT a deployable path.
    supply_alone_clears_500_with_cb3 = False                  # PPL-dead headline -> no supply-alone clear
    supply_alone_clears_500_with_headline_arith = base_plus_cb3_honest >= TARGET  # arithmetic only (PPL-dead)
    supply_plus_available_lift = base_today + lift_for_paper
    available_lift_clears_bare_gap = (supply_lift_available_tps is not None
                                      and supply_lift_available_tps >= deficit_to_500_today)

    # 518.92 eta-axis base is deflated on THREE grounds (#373/#375/#378).
    eta_axis_base_deflated = ETA_AXIS_BASE_DEFLATED_518

    # Structural finding (advisor 17:03Z): private <= public; a public-strict base < 500 may make
    # private-strict-500 unreachable by the demand-side coverage retrain ALONE at any coverage.
    # denken #383 (17:53Z) CONFIRMED this on the honest base: demand-alone needs +0.0572 Δcov = 1.84x
    # the #336 budget -> a supply lift of +17.2 TPS (floor-joint) is required FIRST.
    demand_alone_may_be_insufficient = base_today < TARGET   # True today (471.42 < 500)
    demand_alone_insufficient_confirmed = demand_reaches_500_on_floor == "no"  # denken #383 RED

    # Identity compliance (folds in): the deployable-strict base is "strict" only if byte-exact
    # identity is reachable on the served path; stark #381 sets the rebuild cost (1 vs 2).
    ident = identity_reachability_analysis(stark381_decode)

    denken383_pending = demand_reaches_500_on_floor == "pending"
    supply_lift_measured_pending = supply_lift_available_tps is None

    # Resolve the supply-side GO leaf.  denken #383's reach-on-floor verdict is decisive:
    #   yes -> demand closes 500 even on the honest deployable floor (no supply lift required first).
    #   no  -> a supply lift is required first; the measured lift (wirbel #390 corrected shippable
    #          ceiling OR lawine #388 realized body-allocation TPS) must cover the required lift.
    # 20:19Z kanna #403 gate: the +32.65 cb3 HEADLINE lift is PPL-DEAD (kanna #394 RED), so the supply
    # leaf no longer resolves on it.  The REAL PPL-safe supply number is the conservative-k re-cost
    # PENDING kanna #403 (largest k with held-out worst-seed <= 2.41).  Resolve the "no" branch on kanna
    # #403: pending -> HOLD None; red -> no PPL-safe lift clears -> False; green -> resolve on the re-costed
    # lift value (supply_lift_available_tps) vs the required lift.
    kanna403_pending = kanna403_ppl_safe_supply == "pending"
    if denken383_pending:
        supply_base_enables_500: bool | None = None
        binding = ("PENDING_denken383_honest_reprice"
                   + ("_and_supply_lift_pending_kanna403" if kanna403_pending
                      else "_supply_lift_kanna403_landed"))
    elif demand_reaches_500_on_floor == "yes":
        supply_base_enables_500 = True
        binding = "demand_route_reaches_500_on_deployable_floor_denken383"
    else:  # "no" (denken #383 RED) -> a supply lift is required first; kanna #403 (PPL-safe) decides it
        if kanna403_ppl_safe_supply == "red":
            supply_base_enables_500 = False
            binding = "kanna403_no_ppl_safe_conservative_k_lift_clears_private500_unreachable_demand_alone"
        elif kanna403_ppl_safe_supply == "green":
            if supply_lift_measured_pending or supply_lift_required_tps is None:
                supply_base_enables_500 = None
                binding = "PENDING_resolve_kanna403_ppl_safe_recosted_lift_value"
            else:
                lift_sufficient = supply_lift_available_tps >= supply_lift_required_tps
                supply_base_enables_500 = bool(lift_sufficient)
                binding = ("kanna403_ppl_safe_lift_ge_required_enables_500" if lift_sufficient
                           else "kanna403_ppl_safe_lift_insufficient_private500_unreachable_demand_alone")
        else:  # pending (default) -> HOLD: the +32.65 headline is PPL-dead, the real lift awaits kanna #403
            supply_base_enables_500 = None
            binding = ("PENDING_kanna403_ppl_safe_conservative_k_supply_recost"
                       "__headline_32p65_ppl_dead_kanna394")

    supply_pending = supply_base_enables_500 is None

    return {
        "source": "wirbel #378 (gghmgtk9, merged) honest deployable-strict base",
        "is_binding_go_leaf": True,
        # honest base (do NOT bank 518.92)
        "deployable_strict_band_today": [band_lo, band_hi],
        "honest_strict_base_floor": floor,
        "attn_rebuild_tps_gain": ATTN_REBUILD_TPS_GAIN_378,
        "honest_strict_base_plus_attn": plus_attn,
        "supply_base_today_tps": base_today,
        "supply_base_clears_500_today": clears_500_today,         # False
        "deficit_to_500_today_tps": deficit_to_500_today,
        "eta_axis_base_deflated_518": eta_axis_base_deflated,
        "eta_axis_deflation_grounds": ETA_AXIS_DEFLATION_GROUNDS_378,
        "eta_attn_378": ETA_ATTN_378,
        "lmhead_bi_tax_tps_378_refuted": LMHEAD_BI_TAX_TPS_378_REFUTED,   # #378 by-elimination — REFUTED by #384
        "attn_deficit_untouched_frac_378": ATTN_DEFICIT_UNTOUCHED_FRAC_378,
        "strict_knob": "VLLM_BATCH_INVARIANT=1 (whole-step batch-invariant determinism)",
        "dominant_strict_overhead": "int4-Marlin BODY (#376), NOT bf16 lm_head (refuted by wirbel #384; lm_head FREE)",
        # wirbel #384 (4f32ks1e, 18:12Z) CORRECTION — lm_head FREE, the tax is in the int4-Marlin body
        "wirbel384_lmhead_free": WIRBEL384_LMHEAD_FREE,
        "eta_lmhead_targeted_384": ETA_LMHEAD_TARGETED_384,
        "f_lmhead_384": F_LMHEAD_384,
        "lmhead_is_int4_marlin_not_bf16_384": LMHEAD_IS_INT4_MARLIN_NOT_BF16_384,
        "lmhead_bi_share_of_vbi_overhead_384": LMHEAD_BI_SHARE_OF_VBI_OVERHEAD_384,
        "lmhead_bi_incremental_share_384": LMHEAD_BI_INCREMENTAL_SHARE_384,
        "n_kernel_rebuilds_strict_500_384": N_KERNEL_REBUILDS_STRICT_500_384,
        "dominant_nonattn_strict_locus_384": DOMINANT_NONATTN_STRICT_LOCUS_384,
        "body_marlin_decode_strict_pending_stark381_384": BODY_MARLIN_DECODE_STRICT_PENDING_STARK381_384,
        # structural finding (denken #383 CONFIRMED demand-alone insufficient on the honest base)
        "private_le_public_rho_ceiling": True,
        "demand_alone_may_be_insufficient": demand_alone_may_be_insufficient,
        "demand_alone_insufficient_confirmed_383": demand_alone_insufficient_confirmed,
        # denken #383 (t68af2yw, RED) banked honest re-price detail
        "denken383_reaches_500_on_floor": demand_reaches_500_on_floor,
        "supply_lift_required_first_tps_383": (supply_lift_required_tps
                                               if supply_lift_required_tps is not None else None),
        "supply_lift_required_et_only_tps_383": SUPPLY_LIFT_REQUIRED_ET_ONLY_TPS_383,
        "private_on_floor_383": PRIVATE_ON_FLOOR_383,
        "residual_to_500_on_floor_383": RESIDUAL_TO_500_ON_FLOOR_383,
        "required_dcov_383": REQUIRED_DCOV_383,
        "required_dcov_budget_mult_383": REQUIRED_DCOV_BUDGET_MULT_383,
        "private_cap_central_floor_383": PRIVATE_CAP_CENTRAL_FLOOR_383,
        "private_cap_worst_floor_383": PRIVATE_CAP_WORST_FLOOR_383,
        "private_cap_central_attn_383": PRIVATE_CAP_CENTRAL_ATTN_383,
        "private_cap_worst_attn_383": PRIVATE_CAP_WORST_ATTN_383,
        "reproduces_377_under_revival_383": REPRODUCES_377_UNDER_REVIVAL_383,
        "attn_rebuild_alone_closes_supply_gap_383": ATTN_REBUILD_ALONE_CLOSES_SUPPLY_GAP_383,
        "pilot_on_critical_path_383": PILOT_ON_CRITICAL_PATH_383,
        "base_clears_pilot_relevant_band_383": list(BASE_CLEARS_PILOT_RELEVANT_BAND_383),
        # lawine #372 (mpzfw116, GREEN) supply lever — de-risks the required lift (pending lawine #388)
        "lawine372_supply_lever_alive": LAWINE372_SUPPLY_LEVER_ALIVE,
        "mixed_precision_avg_bpw_372": MIXED_PRECISION_AVG_BPW_372,
        "body_3bit_frac_372": BODY_3BIT_FRAC_372,
        "mixed_precision_ppl_delta_pct_372": MIXED_PRECISION_PPL_DELTA_PCT_372,
        "mixed_precision_gate_ppl_372": MIXED_PRECISION_GATE_PPL_372,
        "body_read_reduction_372": BODY_READ_REDUCTION_372,
        "mixed_precision_analytic_lift_tps_372": list(MIXED_PRECISION_ANALYTIC_LIFT_TPS_372),
        "uniform_3bit_died_on_ppl_372": UNIFORM_3BIT_DIED_ON_PPL_372,
        # lawine #388 (g5lfdpgw, LANDED 19:01Z) — realized cb3 body-allocation supply LIFT (M=1)
        "lawine388_landed": LAWINE388_LANDED,
        "lawine388_realized_tps_pending": not LAWINE388_LANDED,    # LANDED -> False
        "cb3_lift_honest_tps_388": CB3_LIFT_HONEST_TPS_388,         # +33 (#388 draft-separated; superseded by #392)
        "cb3_lift_realistic_tps_388": CB3_LIFT_REALISTIC_TPS_388,   # +38.34 (lumps un-shrunk draft fraction)
        "cb3_lift_mult_388": CB3_LIFT_MULT_388,
        "cb3_closes_383_supply_gap_floor_388": CB3_CLOSES_383_SUPPLY_GAP_FLOOR_388,  # both >= +17.2 and +23.8
        "cb3_m1_is_bw_bound_388": CB3_M1_IS_BW_BOUND_388,          # False (M=1 overhead-bound; served is M=8)
        "cb3_m1_hbm_eff_388": CB3_M1_HBM_EFF_388,
        "cb3_m1_roofline_mult_388": CB3_M1_ROOFLINE_MULT_388,
        "cb3_realized_frac_of_roofline_388": CB3_REALIZED_FRAC_OF_ROOFLINE_388,
        "cb3_draft_frac_lumped_388": CB3_DRAFT_FRAC_LUMPED_388,
        # lawine #391 (3udzpoq8, LANDED 19:11Z) — M=8 served-width contingency (realistic-GREEN / floor-YELLOW)
        "lawine391_landed": LAWINE391_LANDED,
        "lawine391_m8_hbm_eff": LAWINE391_M8_HBM_EFF,             # 0.2559 (FLAT vs M=1 0.2578)
        "lawine391_m1_hbm_eff": LAWINE391_M1_HBM_EFF,
        "cb3_lift_m8_realistic_391": CB3_LIFT_M8_REALISTIC_391,   # +38.02 (clears +17.2 & +23.75 robust)
        "cb3_lift_m8_measured_floor_391": CB3_LIFT_M8_MEASURED_FLOOR_391,  # +15.67 (MISSES +23.75 robust)
        "cb3_closes_383_robust_m8_391": CB3_CLOSES_383_ROBUST_M8_391,      # False (floor tier misses robust)
        "supply_lift_required_robust_tps_383": SUPPLY_LIFT_REQUIRED_ROBUST_TPS_383,  # 23.75
        "cb3_m8_efficiency_flat_391": CB3_M8_EFFICIENCY_FLAT_391,          # served regime does NOT raise eff
        "cb3_lane_realistic_green_floor_yellow_391": CB3_LANE_REALISTIC_GREEN_FLOOR_YELLOW_391,
        # denken #392 (2evhfxi7, LANDED 19:12Z) — AUTHORITATIVE honest composed number on 471.42 (USE +32.65)
        "denken392_landed": DENKEN392_LANDED,
        "cb3_lift_honest_denken392": CB3_LIFT_HONEST_DENKEN392,   # +32.65 AUTHORITATIVE (USE for --supply-lift)
        "cb3_lift_floor_denken392": CB3_LIFT_FLOOR_DENKEN392,     # +42.91 floor tier
        "cb3_f_verify_body_392": CB3_F_VERIFY_BODY_392,           # 0.7624 shrinkable verify-body fraction
        "cb3_388_optimism_tps_392": CB3_388_OPTIMISM_TPS_392,     # +5.69 (~15%) #388 optimism
        "combined_route_reaches_392": COMBINED_ROUTE_REACHES_392, # 512.60 (469.68 -> 512.60)
        "combined_route_residual_dcov_392": COMBINED_ROUTE_RESIDUAL_DCOV_392,  # +0.0117 (38% of #336 budget)
        "combined_route_reaches_500_honest_392": COMBINED_ROUTE_REACHES_500_HONEST_392,  # True on paper
        "et_ladder_match_realized_pct_392": ET_LADDER_MATCH_REALIZED_PCT_392,  # 0.18% match to E_T_REALIZED
        "cb3_lift_honest": cb3_lift_honest,                       # +32.65 HEADLINE — PPL-DEAD (kanna #394 RED)
        "cb3_headline_lift_ppl_dead_394": cb3_headline_lift_ppl_dead,  # True — do NOT carry +32.65
        "cb3_lift_honest_388_superseded": cb3_lift_honest_388,    # +33 (#388, superseded by #392/#394)
        "cb3_lift_realistic": cb3_lift_realistic,
        "base_plus_cb3_honest": base_plus_cb3_honest,             # 500.13 on 467.48 — arithmetic only (PPL-dead)
        "base_plus_cb3_realistic": base_plus_cb3_realistic,       # 505.82 — rests on the PPL-dead headline
        "supply_alone_clears_500_with_cb3": supply_alone_clears_500_with_cb3,  # False (PPL-dead headline)
        "supply_alone_clears_500_with_headline_arith": supply_alone_clears_500_with_headline_arith,  # arithmetic only
        "supply_plus_available_lift": supply_plus_available_lift,
        "available_lift_clears_bare_gap": available_lift_clears_bare_gap,      # lift >= 32.52 gap_to_500
        # wirbel #393 (0q7ynumg, LANDED 20:19Z) — BASE CORRECTION: decode attn strict tax 3.01% -> 467.48
        "wirbel393_landed": WIRBEL393_LANDED,
        "realized_deployed_strict_393": REALIZED_DEPLOYED_STRICT_393,   # 467.48 CORRECTED supply BASE (supersedes 471.42)
        "shippable_ceiling_393": SHIPPABLE_CEILING_393,                 # 505.29 (was 509.78)
        "gap_to_500_393": GAP_TO_500_393,                              # 32.52 bare gap_to_500 (widened from 28.58)
        "decode_attn_strict_tax_pct_393": DECODE_ATTN_STRICT_TAX_PCT_393,   # 3.01% (decode band [528,658])
        "eval_weighted_attn_strict_tax_pct_378": EVAL_WEIGHTED_ATTN_STRICT_TAX_PCT_378,  # 2.15% (smaller)
        "decode_tax_delta_pp_393": DECODE_TAX_DELTA_PP_393,           # +0.86pp decode-vs-eval-weighted
        "decode_band_393": list(DECODE_BAND_393),
        "attn_eta_reducible_393": ATTN_ETA_REDUCIBLE_393,             # False (irreducible rebuild-free)
        "attn_sole_strict_tax_393": ATTN_SOLE_STRICT_TAX_393,         # True (FlashInfer-BI MEASURED-FALSE)
        "base_390_superseded_by_393": BASE_390_SUPERSEDED_BY_393,     # 471.42 (provenance)
        # kanna #394 (d184kbey, LANDED RED 20:02Z) — the +32.65 HEADLINE supply lift is PPL-DEAD
        "kanna394_landed": KANNA394_LANDED,
        "kanna394_heldout_ppl_pending": KANNA394_HELDOUT_PPL_PENDING,  # False (LANDED)
        "kanna394_insample_ppl_repro_372": KANNA394_INSAMPLE_PPL_REPRO_372,   # 2.3816 reproduces exactly
        "kanna394_margin_is_winners_curse": KANNA394_MARGIN_IS_WINNERS_CURSE,
        "kanna394_selected_k_range": list(KANNA394_SELECTED_K_RANGE),         # k=243-246 chases the ceiling
        "kanna394_heldout_worst_seed_ppl": KANNA394_HELDOUT_WORST_SEED_PPL,   # 2.4223 breaches 2.42
        "kanna394_ood_sharegpt_ppl": KANNA394_OOD_SHAREGPT_PPL,              # 2.4270 breaches 2.42
        "cb3_supply_deployable_at_headline_394": CB3_SUPPLY_DEPLOYABLE_AT_HEADLINE_394,  # False
        "cb3_deployable_at_conservative_k_394": CB3_DEPLOYABLE_AT_CONSERVATIVE_K_394,    # True (k=232 ~2.39)
        "kanna394_conservative_k_232": KANNA394_CONSERVATIVE_K_232,
        "kanna394_conservative_k_heldout_ppl": KANNA394_CONSERVATIVE_K_HELDOUT_PPL,      # 2.39 (clears 2.42)
        # kanna #403 — the REAL PPL-safe conservative-k supply re-cost (PENDING; the operative supply gate)
        "kanna403_ppl_safe_supply_input": kanna403_ppl_safe_supply,
        "kanna403_ppl_safe_supply_pending": kanna403_pending,
        "kanna403_ppl_safe_supply_landed_constant_pending": KANNA403_PPL_SAFE_SUPPLY_PENDING,  # True (card open)
        "kanna403_heldout_worst_seed_target": KANNA403_HELDOUT_WORST_SEED_TARGET,  # <= 2.41
        "cb3_conservative_k_lift_pending_kanna403": CB3_CONSERVATIVE_K_LIFT_PENDING_KANNA403,
        # wirbel #390 (5y64zbjz, LANDED 19:01Z) — prior shippable base (superseded by #393's 467.48)
        "wirbel390_landed": WIRBEL390_LANDED,
        "wirbel390_shippable_ceiling_pending": not WIRBEL390_LANDED,  # LANDED -> False
        "realized_deployed_strict_390": REALIZED_DEPLOYED_STRICT_390,    # 471.42 (superseded by #393's 467.48)
        "shippable_band_390": list(SHIPPABLE_BAND_390),
        "gap_to_500_390": GAP_TO_500_390,                         # 28.58 (superseded by #393's 32.52)
        "supply_alone_closes_500_390": SUPPLY_ALONE_CLOSES_500_390,   # False (no cb3 shrink)
        "eta_attn_390": ETA_ATTN_390,                            # 0.02145 the SOLE strict tax (rebuilds=1)
        "n_kernel_rebuilds_strict_500_390": N_KERNEL_REBUILDS_STRICT_500_390,  # 1 (attn only; ledger 2->1)
        "body_marlin_decode_strict_green_390": BODY_MARLIN_DECODE_STRICT_GREEN_390,  # byte-exact @ M=8
        "stark381_decode_identity_per_gemm_green_390": STARK381_DECODE_IDENTITY_PER_GEMM_GREEN_390,
        "spread_is_lmhead_bf16_tax_390": SPREAD_IS_LMHEAD_BF16_TAX_390,   # False (phantom #378 bracket)
        "shippable_ceiling_510_reinstated_390": SHIPPABLE_CEILING_510_REINSTATED_390,   # 510.01 REINSTATED
        "shippable_ceiling_518_still_refuted_390": SHIPPABLE_CEILING_518_STILL_REFUTED_390,  # 518.92 refuted
        "deployed_floor_lift_over_378_band_390": DEPLOYED_FLOOR_LIFT_OVER_378_BAND_390,
        "shippable_ceiling_refuted_bf16_premise_390": [SHIPPABLE_CEILING_518_STILL_REFUTED_390],
        # denken #387 (z8osvif8, LANDED 19:01Z) — demand anchor MEASURED + MTP K=7 premise correction
        "denken387_landed": DENKEN387_LANDED,
        "measured_top4_coverage_387": MEASURED_TOP4_COVERAGE_387,
        "coverage_anchor_gap_387": COVERAGE_ANCHOR_GAP_387,
        "required_delta_floor_measured_387": REQUIRED_DELTA_FLOOR_MEASURED_387,
        "denken383_red_robust_to_measured_anchor_387": DENKEN383_RED_ROBUST_TO_MEASURED_ANCHOR_387,
        "deployed_drafter_mtp_k_387": DEPLOYED_DRAFTER_MTP_K_387,     # 7 (MTP K=7, NOT EAGLE-3)
        "drafter_is_mtp_not_eagle3_387": DRAFTER_IS_MTP_NOT_EAGLE3_387,
        "demand_ladder_label_387": DEMAND_LADDER_LABEL_387,
        # kanna #374 (djia6icp, LANDED 19:01Z) — fusion lever CLOSED (Route-A stays excluded)
        "kanna374_fusion_lever_closed": KANNA374_FUSION_LEVER_CLOSED,
        "fusion_byte_exact_pinnable_374": FUSION_BYTE_EXACT_PINNABLE_374,
        "capture_land_371_sole_identity_safe_nonspec_leg_374": CAPTURE_LAND_371_SOLE_IDENTITY_SAFE_NONSPEC_LEG_374,
        "route_a_stays_excluded_374": ROUTE_A_STAYS_EXCLUDED_374,
        # stark #381 (9edps20u, LANDED 19:16Z) — e2e confirms #390 Arm A; rebuilds=1; DON'T flag Marlin
        "stark381_landed": STARK381_LANDED,
        "stark381_body_marlin_bitexact_m8_e2e": STARK381_BODY_MARLIN_BITEXACT_M8_E2E,  # e2e bit-exact @ M=8
        "stark381_residual_flips": STARK381_RESIDUAL_FLIPS,         # 1 flip ...
        "stark381_residual_tokens": STARK381_RESIDUAL_TOKENS,       # ... in 891 tokens
        "stark381_knife_edge_nat": STARK381_KNIFE_EDGE_NAT,         # 0.125-nat near-tie (TRITON_ATTN, NOT Marlin)
        "stark381_residual_is_knife_edge_near_tie": STARK381_RESIDUAL_IS_KNIFE_EDGE_NEAR_TIE,
        "stark381_pinned_reaches_identity_1p0": STARK381_PINNED_REACHES_IDENTITY_1P0,  # False (1-flip residual)
        "stark381_residual_locus": STARK381_RESIDUAL_LOCUS,         # TRITON_ATTN (#375), NOT int4-Marlin
        "stark381_do_not_flag_marlin_rebuild": STARK381_DO_NOT_FLAG_MARLIN_REBUILD,    # ledger stays 1
        "stark381_rebuild_ledger": STARK381_REBUILD_LEDGER,         # 1 (attention only)
        # identity compliance folds in here (strict prerequisite of the deployable-strict base)
        "identity_reachable_env_or_rebuild": ident["identity_reachable_env_or_rebuild"],
        "identity_cost_branch": ident["cost_branch"],
        "identity_cost_branch_pending": ident["cost_branch_pending"],
        "identity_rebuild_line_items": ident["rebuild_line_items"],
        # supply-lift DATA has LANDED but the +32.65 HEADLINE is PPL-DEAD (kanna #394 RED); the binding GO
        # axis is now the kanna #403 PPL-safe conservative-k re-cost (the real supply number), NOT the
        # dead headline.  denken #396/ubel #399 closed the demand-alone + cheap-lever escape routes.
        "denken383_input": demand_reaches_500_on_floor,
        "denken383_pending": denken383_pending,
        "supply_lift_available_tps": supply_lift_available_tps,
        "supply_lift_measured_pending": supply_lift_measured_pending,   # this-run CLI resolution
        "supply_lift_data_landed": (LAWINE388_LANDED and WIRBEL390_LANDED),   # both de-riskers measured
        "supply_lift_headline_ppl_dead_394": CB3_HEADLINE_LIFT_PPL_DEAD_394,  # +32.65 is PPL-dead -> don't carry it
        "supply_lift_pending_kanna403": kanna403_pending,              # the real PPL-safe lift awaits kanna #403
        "supply_lift_required_tps": supply_lift_required_tps,
        # verdict
        "supply_pending": supply_pending,
        "supply_base_enables_500": supply_base_enables_500,
        "binding_constraint": binding,
        "note": (
            "wirbel #378 (gghmgtk9): the only STRICT-byte-exact served knob is VLLM_BATCH_INVARIANT=1 "
            "(whole-step batch-invariant determinism). wirbel #390 (5y64zbjz, 19:01Z) measured 471.42; "
            f"wirbel #393 (0q7ynumg, LANDED 20:19Z) CORRECTED it to {base_today:g}: the DECODE-specific "
            f"attention strict tax is {DECODE_ATTN_STRICT_TAX_PCT_393:g}% (decode band "
            f"{list(DECODE_BAND_393)}), {DECODE_TAX_DELTA_PP_393:g}pp LARGER than the #378 eval-weighted "
            f"{EVAL_WEIGHTED_ATTN_STRICT_TAX_PCT_378:g}% -> gap_to_500 widens to {deficit_to_500_today:g} "
            f"(ceiling {SHIPPABLE_CEILING_393:g}). Attention is the SOLE strict tax and irreducible "
            "rebuild-free (attn_eta_reducible=False; FlashInfer-BI MEASURED-FALSE; pinned-K is a flagged "
            "rebuild). The ledger stays 1 kernel rebuild (attention only): lm_head FREE int4-Marlin (#384) "
            "AND body int4-Marlin byte-exact @ M=8 (#390/#381). STRUCTURAL: private <= public, so a public "
            "base < 500 may make private-500 unreachable by the demand closer ALONE. denken #383 (RED) + "
            f"denken #387 (MEASURED anchor {MEASURED_TOP4_COVERAGE_387:g}) CONFIRMED a supply lift is "
            "required FIRST. lawine #388/#392 sized the cb3 body-allocation lift +"
            f"{cb3_lift_honest:g} honest HEADLINE, BUT kanna #394 (d184kbey, LANDED RED 20:02Z) made that "
            f"HEADLINE PPL-DEAD: #372's +{CB3_INSAMPLE_PPL_MARGIN_372:g} in-sample margin is winner's-curse "
            f"(k={KANNA394_SELECTED_K_RANGE[0]}-{KANNA394_SELECTED_K_RANGE[1]}); the held-out worst-seed "
            f"({KANNA394_HELDOUT_WORST_SEED_PPL:g}) AND OOD ShareGPT ({KANNA394_OOD_SHAREGPT_PPL:g}) BREACH "
            f"the {KANNA394_PPL_GATE:g} gate. So {base_today:g}+{cb3_lift_honest:g} is arithmetic-only, NOT "
            f"deployable. BUT k={KANNA394_CONSERVATIVE_K_232} still clears (~"
            f"{KANNA394_CONSERVATIVE_K_HELDOUT_PPL:g} held-out) -> cb3 IS deployable at a conservative k, at "
            "a smaller UN-COSTED lift. The REAL PPL-safe supply number is PENDING kanna #403 (largest k with "
            f"held-out worst-seed <= {KANNA403_HELDOUT_WORST_SEED_TARGET:g} -> re-costed lift). HOLD the "
            "supply leaf at 'TRUE only at a conservative-k lift, value pending kanna #403'. 518.92 deflated "
            "on three grounds: " + ETA_AXIS_DEFLATION_GROUNDS_378 + "."
        ),
    }


# --------------------------------------------------------------------------- #
# ubel #379 (GREEN) gap-decomposition CEILING-CHECK of the demand-side closer.
# The 4.295pp public->private gap splits into a coverage-ADDRESSABLE part (acceptance) and an
# IRREDUCIBLE part (ctxlen); the fixed numerics/identity tax CANCELS in the public->private STEP
# difference (it floors absolute TPS, not the gap).  Net: the closer c >= 0.9010 is NOT capped by
# an irreducible floor -> BANK it.  We CONSUME ubel #379's split (advisor-relayed), never re-derive.
# (ubel #386, RESOLVED RED 17:53Z: the 0.633% floor does NOT survive the VBI=1 un-packed-attention
# regime — it inflates 2.07x -> 1.310% central.  Central still clears 3.2% (+1.89pp) but the
# pessimistic corner breaches (3.5235%, -0.32pp) and the breakeven prompt shift halves +253 -> +119 tok.
# Re-derive the demand ceiling on the 1.310% live floor and treat prompt-shift sensitivity as binding.)
# --------------------------------------------------------------------------- #
def gap_decomposition_analysis(irreducible_floor_survives_vbi: str = UBEL386_FLOOR_SURVIVES_VBI
                               ) -> dict[str, Any]:
    """Integrate ubel #379's gap decomposition (BANKED GREEN); re-derive on ubel #386's VBI=1 floor."""
    fracs = {
        "acceptance_coverage_addressable": GAP_ACCEPTANCE_FRAC_UBEL379,
        "ctxlen_irreducible": GAP_CTXLEN_FRAC_UBEL379,
        "outlen": GAP_OUTLEN_FRAC_UBEL379,
        "numerics": GAP_NUMERICS_FRAC_UBEL379,
    }
    frac_sum = sum(fracs.values())
    addressable_pp = PUBLIC_PRIVATE_GAP_PCT * GAP_ACCEPTANCE_FRAC_UBEL379   # ~3.6615pp
    irreducible_pp_from_frac = PUBLIC_PRIVATE_GAP_PCT * GAP_CTXLEN_FRAC_UBEL379  # ~0.6335pp
    # Knife-edge margins: each irreducible corner must sit >= KNIFE_EDGE_MIN_MARGIN_PP below 3.2%.
    corner_margins = {c: KNIFE_EDGE_GAP_PCT - c for c in GAP_IRREDUCIBLE_CORNERS_UBEL379}
    all_corners_clear = all(m >= KNIFE_EDGE_MIN_MARGIN_PP for m in corner_margins.values())
    # Reconcile ubel #379's independent +0.0108 with denken #377's relayed figure.
    reconcile_delta = abs(DELTA_COV_UBEL379 - DENKEN377_DELTA_COV_RECONCILE)
    # ubel #386 (RESOLVED RED 17:53Z): the 0.633% off-VBI floor does NOT survive the VBI=1 un-packed-
    # attention regime — it inflates 2.07x -> 1.310% central.  The "uncapped on the live stack" claim
    # FAILS (uncapped_on_live_vbi_stack=False); re-derive the demand ceiling on the 1.310% live floor.
    floor_vbi_pending = irreducible_floor_survives_vbi == "pending"
    floor_survives_vbi = irreducible_floor_survives_vbi == "survives"
    floor_inflates_vbi = irreducible_floor_survives_vbi == "inflates"
    uncapped_on_live_stack: bool | None = (None if floor_vbi_pending else floor_survives_vbi)
    # Live-VBI central floor: 0.633% if it survives, 1.310% if it inflates, off-VBI 0.633% while pending.
    live_floor_central = (IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386 if floor_inflates_vbi
                          else GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379)
    # Under VBI=1 the central floor still clears the 3.2% knife-edge (+1.89pp) but the pessimistic
    # corner breaches (banked ubel #386); the comfortable +253-tok breakeven buffer halves to +119.
    central_clears_3p2_vbi1 = (live_floor_central < KNIFE_EDGE_GAP_PCT - KNIFE_EDGE_MIN_MARGIN_PP
                               if not floor_inflates_vbi else CENTRAL_CLEARS_3P2_VBI1_386)
    all_corners_clear_3p2_vbi1: bool | None = (
        None if floor_vbi_pending
        else (ALL_CORNERS_CLEAR_3P2_VBI1_386 if floor_inflates_vbi else all_corners_clear))
    breakeven_prompt_shift_vbi1_tok = (BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386 if floor_inflates_vbi
                                       else PRIVATE_PROMPT_SHIFT_BREAKEVEN_TOK)
    # ubel #389 (fqt33bj3, merged 19:15Z) — measured per-L attention identity floor under VBI=1 is
    # 0.5764% (< the 0.633% off-VBI floor, NOT the 1.310% #386 interpolation); 0 corners breach 3.2%;
    # the measured local-penalty slope is 0.353x the #386 interpolation -> the #386 pessimistic-corner
    # breach was a CONSERVATIVE-SLOPE ARTIFACT.  Bank the refutation: the OPERATIVE breach state drops
    # the #386 RED.  (The _386 fields stay as provenance of the relayed hypothesis.)
    breach_386_refuted = UBEL389_386_BREACH_REFUTED                          # True (banked)
    operative_floor_vbi1_central_pct = (UBEL389_MEASURED_FLOOR_VBI1_PCT if breach_386_refuted
                                        else live_floor_central)              # 0.5764% measured
    all_corners_clear_3p2_operative: bool | None = (
        UBEL389_ALL_CORNERS_CLEAR_3P2_MEASURED if breach_386_refuted else all_corners_clear_3p2_vbi1)
    prompt_shift_binding_risk_operative = floor_inflates_vbi and not breach_386_refuted  # False
    return {
        "source": "ubel #379 (5kpb73tb, GREEN, independently verified)",
        "public_private_gap_pct": PUBLIC_PRIVATE_GAP_PCT,
        "gap_fractions": fracs,
        "gap_fractions_sum": frac_sum,
        "gap_addressable_pp": addressable_pp,
        "gap_irreducible_pp_central": GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379,
        "gap_irreducible_pp_from_ctxlen_frac": irreducible_pp_from_frac,
        "gap_irreducible_corners": list(GAP_IRREDUCIBLE_CORNERS_UBEL379),
        "knife_edge_gap_pct": KNIFE_EDGE_GAP_PCT,
        "knife_edge_corner_margins": corner_margins,
        "all_corners_clear_knife_edge": all_corners_clear,
        "knife_edge_min_margin_pp": KNIFE_EDGE_MIN_MARGIN_PP,
        "private_prompt_shift_breakeven_tok": PRIVATE_PROMPT_SHIFT_BREAKEVEN_TOK,
        # numerics CANCELS — refutes "numerics tax is the irreducible floor"
        "numerics_tax_cancels_in_step_diff": True,
        "numerics_floors_absolute_tps_not_gap": True,
        "refutes_numerics_is_irreducible_floor": True,
        # coverage target + reconciliation
        "coverage_target_for_3p2": COVERAGE_TARGET_FOR_3P2_UBEL379,
        "baseline_cov_336": BASELINE_COV_336,
        "delta_cov_ubel379": DELTA_COV_UBEL379,
        "denken377_delta_cov_reconcile": DENKEN377_DELTA_COV_RECONCILE,
        "reconcile_delta_vs_denken377": reconcile_delta,
        "reconciles_within_0p0003": reconcile_delta <= 0.0003 + TOL_EXACT,
        # slope (pending ubel #382 robustness) + headroom
        "slope_tps_per_coverage_ubel379": SLOPE_TPS_PER_COVERAGE_UBEL379,
        "gap_after_max_coverage_retrain_pct": GAP_AFTER_MAX_COVERAGE_RETRAIN_PCT_UBEL379,
        # ceiling-check verdict: the demand closer is NOT capped by an irreducible floor
        "gap_channel_live": GAP_ACCEPTANCE_FRAC_UBEL379 > 0.0,
        "closer_not_capped_by_irreducible_floor": (
            GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379 < KNIFE_EDGE_GAP_PCT - KNIFE_EDGE_MIN_MARGIN_PP),
        # ubel #386 (RESOLVED RED): the floor INFLATES under VBI=1 -> re-derive on the 1.310% live floor
        "irreducible_floor_survives_vbi_input_ubel386": irreducible_floor_survives_vbi,
        "irreducible_floor_vbi_pending_ubel386": floor_vbi_pending,
        "irreducible_floor_inflates_vbi_386": floor_inflates_vbi,
        "uncapped_on_live_vbi_stack": uncapped_on_live_stack,   # False once #386 resolved "inflates"
        "irreducible_floor_vbi1_central_pct_386": live_floor_central,
        "floor_inflation_mult_386": FLOOR_INFLATION_MULT_386,
        "central_clears_3p2_vbi1_386": central_clears_3p2_vbi1,
        "central_margin_to_3p2_vbi1_pp_386": CENTRAL_MARGIN_TO_3P2_VBI1_PP_386,
        "all_corners_clear_3p2_vbi1_386": all_corners_clear_3p2_vbi1,   # False once "inflates"
        "pessimistic_corner_vbi1_pct_386": PESSIMISTIC_CORNER_VBI1_PCT_386,
        "pessimistic_corner_margin_pp_386": PESSIMISTIC_CORNER_MARGIN_PP_386,
        "breakeven_prompt_shift_vbi1_tok_386": breakeven_prompt_shift_vbi1_tok,
        "prompt_shift_sensitivity_binding_risk_386": floor_inflates_vbi,
        # ubel #389 (LANDED 19:15Z): the #386 pessimistic-corner breach is REFUTED on the measured slope
        "ubel389_landed": UBEL389_LANDED,
        "ubel389_pin_breach_pending": UBEL389_PIN_BREACH_PENDING,                 # now False (resolved)
        "ubel389_386_breach_refuted": breach_386_refuted,                         # True
        "ubel389_measured_floor_vbi1_pct": UBEL389_MEASURED_FLOOR_VBI1_PCT,       # 0.5764 (< 0.633)
        "ubel389_pessimistic_breaches_3p2_measured": UBEL389_PESSIMISTIC_BREACHES_3P2_MEASURED,  # 0
        "ubel389_all_corners_clear_3p2_measured": UBEL389_ALL_CORNERS_CLEAR_3P2_MEASURED,        # True
        "ubel389_measured_slope_ratio_to_386": UBEL389_MEASURED_SLOPE_RATIO_TO_386,  # 0.353x
        # OPERATIVE (post-#389) demand-floor state — the leaf is ROBUST on the measured slope
        "irreducible_floor_vbi1_central_pct_operative": operative_floor_vbi1_central_pct,  # 0.5764%
        "all_corners_clear_3p2_vbi1_operative": all_corners_clear_3p2_operative,            # True
        "prompt_shift_sensitivity_binding_risk_operative": prompt_shift_binding_risk_operative,  # False
        "note": (
            "ubel #379 (GREEN): the 4.295pp gap = 85.25% acceptance (coverage-ADDRESSABLE) + 14.75% "
            "ctxlen (IRREDUCIBLE) + 0% outlen + 0% numerics. The fixed numerics/identity tax CANCELS in "
            "the public->private step diff (floors absolute TPS, not the gap), REFUTING 'numerics is the "
            "irreducible floor'. coverage_target_for_3p2 = 0.9011 (+0.0108 from #336's 0.8903) reconciles "
            "denken #377 to within 0.0003; slope 489.8 TPS/unit (CONFIRMED private-robust by ubel #382 "
            "GREEN -> 437.3 private; conservative target ~0.911); gap_after_max_coverage_retrain 1.142%. "
            "BUT ubel #386 (RESOLVED RED, 17:53Z): the off-VBI 0.633% floor does NOT survive the VBI=1 "
            f"un-packed-attention contract — it inflates {FLOOR_INFLATION_MULT_386:g}x -> "
            f"{IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386:g}% central. Central STILL clears the 3.2% knife-edge "
            f"(+{CENTRAL_MARGIN_TO_3P2_VBI1_PP_386:g}pp) so the route is not dead, but all_corners_clear_"
            f"3p2_vbi1=False (the pessimistic corner breaches at {PESSIMISTIC_CORNER_VBI1_PCT_386:g}%, "
            f"{PESSIMISTIC_CORNER_MARGIN_PP_386:g}pp) and the breakeven private prompt shift HALVES "
            f"({BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK:g} -> {BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386:g} tok). "
            "BUT ubel #389 (LANDED 19:15Z) MEASURED the per-L attention identity floor under VBI=1 at "
            f"{UBEL389_MEASURED_FLOOR_VBI1_PCT:g}% (< the 0.633% off-VBI floor, NOT 1.310%); 0 corners "
            f"breach 3.2% and the measured local-penalty slope is only {UBEL389_MEASURED_SLOPE_RATIO_TO_386:g}x "
            "the #386 interpolation -> the #386 pessimistic-corner breach was a CONSERVATIVE-SLOPE ARTIFACT. "
            "OPERATIVE: drop the #386 RED; the demand leaf is ROBUST on the measured slope (all corners "
            "clear 3.2%, prompt-shift sensitivity NOT binding). The _386 fields remain as provenance."
        ),
    }


# --------------------------------------------------------------------------- #
# DEMAND-SIDE residual leaf (denken #377 sized + denken #380 two-tier deliverability + ubel #382
# slope private-robustness).  This leaf is now RESOLVED to `central=GREEN / robust=pending-pilot`:
#   - DELIVERABILITY (denken #380, BANKED): the single closer SPLITS into a central tier
#     (c >= 0.8959, Δcov +0.00565, p_deliver 0.958 >= 0.90 -> DELIVERABLE NOW) and a robust tier
#     (c >= 0.9010, Δcov +0.0107, p_deliver 0.811 < 0.90 -> pending a coverage-lift pilot).  The
#     kappa-axis is ROBUST (kappa_breakeven 0.1222 << worst c* corner); the binding uncertainty is
#     the DELIVERY distribution (defensible fine-tune N(0.016,0.006), not #339's optimistic tail).
#   - SLOPE (ubel #382, BANKED GREEN): the 489.8 TPS/unit slope IS private-robust -> 437.3 private
#     (flattening_ratio 0.893); bank the CONSERVATIVE private-anchored target ~0.911 (NOT bare 0.9011).
#   - GAP CEILING-CHECK (ubel #379, BANKED GREEN): the closer is NOT capped by the off-VBI floor, BUT
#     ubel #386 (RESOLVED RED) inflated it 2.07x -> 1.310% central under VBI=1 (central still clears).
# So the demand leaf DELIVERS the residual at CENTRAL confidence now; the robust tier is the only
# remaining demand-side pending (a human-approval-gated coverage-lift pilot).  We do NOT re-derive
# any sizing.  Note: this leaf transfers FROM the supply-side honest base (wirbel #378) — see
# supply_side_base_analysis; a public base < 500 can make the residual unreachable demand-alone.
# --------------------------------------------------------------------------- #
def demand_side_route_analysis(robust_pilot: str = "pending",
                               gap_addressable_pp: float | None = None,
                               irreducible_floor_survives_vbi: str = UBEL386_FLOOR_SURVIVES_VBI,
                               ubel401_tree_coverage_ceiling: str = "pending",
                               denken402_tree_net_supply: str = "pending") -> dict[str, Any]:
    """Demand-side residual leaf — 20:19Z RE-PRICED: the d-cov can ONLY come from the TREE (genuinely-new).

    Per denken #383 the demand closer alone does NOT reach private-500 on the honest base.  The 20:19Z
    cluster then closed both deployable escape routes:
      - denken #396 (LANDED RED): demand-ALONE busts EVEN BARE on the corrected 467.48 base (required_dcov
        ~0.0338 = 109% of the #336 +0.031 budget; on bare 471.42 the #389 floor already pushed it over).
      - ubel #399 (LANDED RED): there is NO cheap deployable demand lever — every monotone draft-head lever
        (temperature, affine calibration) is a RANK-INVARIANT no-op (MC max|Δcov|=0; frac_of_gap_covered=0%).
    So the demand d-cov (#392's +0.0117 sliver and anything beyond) can ONLY be SUPPLIED by the TREE (the
    locked +0.1286 top-1->top-4 prize) — a genuinely-new-lever.  The leaf now DELIVERS only if the tree
    NETs the d-cov, gated on ubel #401 (top-8/16 coverage ceiling) OR denken #402 (net after verify-M tax).
    The central-GREEN coverage-retrain sizing below is kept for PROVENANCE (it sized the target) but a
    drafter retrain is FORBIDDEN, so it is not a deployable delivery path.

    `robust_pilot`: the (forbidden-retrain) coverage-lift pilot — provenance only now; OFF the critical path.
    `gap_addressable_pp` (ubel #379, BANKED GREEN): None -> use ubel #379's banked value.
    `irreducible_floor_survives_vbi` (ubel #386, RESOLVED): banked default "inflates" (-> 1.310% floor).
    `ubel401_tree_coverage_ceiling`: tree top-8/16 coverage-ceiling probe — "pending"|"green"|"red".
    `denken402_tree_net_supply`: does the tree NET d-cov after its verify-M step-time tax on 467.48? same.
    """
    # Budget check against #336's +0.031 envelope.  ubel #382 prices the CONSERVATIVE private-anchored
    # sizing at 66.6% of the budget (central 38.9%); both fit.
    robust_budget_frac = DELTA_COV_ROBUST / BUDGET_336_ENVELOPE      # ~0.345 (35%) public-anchored
    central_budget_frac = DELTA_COV_CENTRAL / BUDGET_336_ENVELOPE    # ~0.182 (18%) public-anchored
    within_336_budget = robust_budget_frac < 1.0                     # True (and central too)

    # Implied baseline coverage (c* minus the Δcov needed).
    baseline_cov_robust = RECOMMENDED_RETRAIN_TARGET_C - DELTA_COV_ROBUST    # 0.8903
    baseline_cov_central = RECOMMENDED_RETRAIN_TARGET_C - DELTA_COV_CENTRAL  # 0.89535

    # --- denken #380 two-tier deliverability (BANKED YELLOW) --- #
    # CENTRAL tier: deliverable now (p_deliver 0.958 >= 0.90).  ROBUST tier: pending a pilot
    # (p_deliver 0.811 < 0.90).  The defensible fine-tune delivery is N(0.016, 0.006).
    deliver_central = P_DELIVER_CENTRAL_DEFENSIBLE_380 >= P_DELIVER_THRESHOLD_380   # True (GREEN)
    deliver_robust_modeled = P_DELIVER_ROBUST_DEFENSIBLE_380 >= P_DELIVER_THRESHOLD_380  # False (pending pilot)
    robust_pilot_pending = robust_pilot == "pending"
    if robust_pilot_pending:
        deliver_robust: bool | None = None      # robust tier unresolved until the pilot lands
    else:
        deliver_robust = robust_pilot == "delivers"
    # The deliverability of the DEPRECATED optimistic prior (kept for provenance/continuity only).
    delivered_after_kappa = DELIVERABILITY_339_MEAN * KAPPA_INT4_CT_TRANSFER  # ~0.02587 (#339 optimistic)
    deliverability_margin = delivered_after_kappa - DELTA_COV_ROBUST          # ~0.0152 (>0, optimistic)

    # Triple-tail OUT-OF-BUDGET corner: a sensitivity band, NOT the central operating point.
    tt = dict(TRIPLE_TAIL)
    tt["out_of_budget"] = tt["cost_frac_of_336_budget"] > 1.0  # True (136%)
    tt["c_star_above_recommended"] = tt["worst_c_star"] > RECOMMENDED_RETRAIN_TARGET_C  # True
    tt["is_sensitivity_band_not_central"] = True

    # Gap split BANKED by ubel #379 (GREEN): None -> use the banked coverage-addressable value.
    gap_split_banked = gap_addressable_pp is None
    gap_pp = GAP_ADDRESSABLE_PP_UBEL379 if gap_addressable_pp is None else gap_addressable_pp
    gap_irreducible_pp = PUBLIC_PRIVATE_GAP_PCT - gap_pp
    gap_channel_live = gap_pp > 0.0
    floor_vbi_pending = irreducible_floor_survives_vbi == "pending"   # ubel #386
    floor_inflates_vbi = irreducible_floor_survives_vbi == "inflates"  # ubel #386 RESOLVED RED
    # ubel #386: the live floor is 1.310% (inflated) vs 0.633% (off-VBI); central still clears 3.2%.
    live_floor_central_pct = (IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386 if floor_inflates_vbi
                              else GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379)
    prompt_shift_binding_risk = floor_inflates_vbi   # the +253-tok comfortable buffer halves to +119
    # ubel #389 (LANDED 19:15Z): the #386 breach is REFUTED on the MEASURED slope (0.5764% floor, 0
    # corners breach 3.2%, slope 0.353x the #386 interpolation).  Bank the OPERATIVE robust state.
    breach_386_refuted = UBEL389_386_BREACH_REFUTED                          # True (banked)
    operative_floor_vbi1_central_pct = (UBEL389_MEASURED_FLOOR_VBI1_PCT if breach_386_refuted
                                        else live_floor_central_pct)          # 0.5764% measured
    all_corners_clear_3p2_operative = (UBEL389_ALL_CORNERS_CLEAR_3P2_MEASURED if breach_386_refuted
                                       else (ALL_CORNERS_CLEAR_3P2_VBI1_386 if floor_inflates_vbi else True))
    prompt_shift_binding_risk_operative = prompt_shift_binding_risk and not breach_386_refuted  # False

    # --- ubel #382 slope (BANKED GREEN) --- #
    # The slope is CONFIRMED private-robust; bank the conservative private-anchored target ~0.911.
    slope_private_robust = SLOPE_IS_PRIVATE_ROBUST_382                # True (banked)
    slope_private_tps = SLOPE_TPS_PER_COVERAGE_PRIVATE_382            # 437.3 (489.8 * 0.893)

    # --- 20:19Z tree-net-supply gate (the ONLY deployable demand delivery path) --- #
    # denken #396 (RED) closed demand-alone; ubel #399 (RED) closed the cheap monotone levers.  So the
    # d-cov must be SUPPLIED by the TREE (the +0.1286 top-1->top-4 prize): the leaf DELIVERS iff the tree
    # NETs the d-cov (ubel #401 coverage ceiling OR denken #402 net-after-verify-M-tax green); it is DEAD
    # iff both tree probes red; PENDING otherwise.  The central-GREEN coverage-retrain sizing is provenance.
    tree_green = (ubel401_tree_coverage_ceiling == "green" or denken402_tree_net_supply == "green")
    tree_both_red = (ubel401_tree_coverage_ceiling == "red" and denken402_tree_net_supply == "red")
    tree_pending = not (tree_green or tree_both_red)
    demand_alone_busts_467 = DENKEN396_DEMAND_ALONE_BUSTS_EVEN_BARE_467   # True (109% of budget)
    no_cheap_demand_lever = not UBEL399_CHEAP_DEMAND_LEVER_EXISTS         # True (monotone levers no-op)

    # --- leaf verdict (20:19Z) --- #
    if not gap_channel_live:
        demand_leaf_delivers: bool | None = False
        binding = "gap_irreducible_no_coverage_channel_ubel379_override"
    elif tree_both_red:
        demand_leaf_delivers = False
        binding = "demand_dcov_unsuppliable__tree_nets_nothing_ubel401_denken402_both_red"
    elif tree_green:
        demand_leaf_delivers = True
        binding = "demand_leaf_delivers_via_tree_net_supply__ubel401_or_denken402_green"
    else:  # tree pending -> the demand residual has no deployable supplier yet
        demand_leaf_delivers = None
        binding = ("demand_dcov_needs_tree__demand_alone_busts_denken396__no_cheap_lever_ubel399__"
                   "PENDING_ubel401_or_denken402_tree_net_supply")

    # The robust coverage pilot is provenance-only now (a forbidden retrain); the operative pending axis
    # is the tree net-supply (ubel #401 / denken #402).
    leaf_robust_pending = robust_pilot_pending
    demand_leaf_pending_tree = tree_pending and gap_channel_live

    return {
        "route": "demand_side_coverage_denken377",
        "is_go_leaf": True,
        "eta_axis_deflated": True,
        "eta_axis_deflation_grounds": ETA_AXIS_DEFLATION_GROUNDS_378,
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
        # two-tier deliverability (denken #380, BANKED YELLOW)
        "demand_closer_central_c": DEMAND_CLOSER_CENTRAL_C,
        "demand_closer_robust_c": DEMAND_CLOSER_ROBUST_C,
        "deliver_central_green": deliver_central,                  # True (p_deliver 0.958 >= 0.90)
        "deliver_robust_modeled": deliver_robust_modeled,         # False (p_deliver 0.811 < 0.90)
        "deliver_robust_resolved": deliver_robust,                # None/True/False per pilot
        "robust_pilot_input": robust_pilot,
        "robust_pilot_pending": robust_pilot_pending,
        "p_deliver_central_defensible": P_DELIVER_CENTRAL_DEFENSIBLE_380,
        "p_deliver_robust_defensible": P_DELIVER_ROBUST_DEFENSIBLE_380,
        "p_deliver_threshold": P_DELIVER_THRESHOLD_380,
        "deliverable_finetune_mean_380": DELIVERABLE_FINETUNE_MEAN_380,
        "deliverable_finetune_std_380": DELIVERABLE_FINETUNE_STD_380,
        "recipe_is_real_380": RECIPE_IS_REAL_380,
        "deliverability_survives_conservative_380": DELIVERABILITY_SURVIVES_CONSERVATIVE_380,
        "kappa_breakeven_380": KAPPA_BREAKEVEN_380,
        "kappa_margin_380": KAPPA_MARGIN_380,
        "kappa_axis_robust": KAPPA_BREAKEVEN_380 < 0.354,          # below the worst program c* corner
        "coverage_lift_pilot_gpu_hr": COVERAGE_LIFT_PILOT_GPU_HR,
        # deprecated optimistic prior (kept for provenance/continuity)
        "kappa_int4_ct_transfer": KAPPA_INT4_CT_TRANSFER,
        "deliverability_339_mean": DELIVERABILITY_339_MEAN,
        "deliverability_339_std": DELIVERABILITY_339_STD,
        "delivered_after_kappa": delivered_after_kappa,
        "deliverability_margin": deliverability_margin,
        # gap channel (BANKED by ubel #379 GREEN; floor INFLATES to 1.310% under VBI=1 per ubel #386)
        "public_private_gap_pct": PUBLIC_PRIVATE_GAP_PCT,
        "gap_shrink_per_coverage": GAP_SHRINK_PER_COVERAGE,
        "gap_coverage_addressable_pp_ubel379": gap_pp,
        "gap_irreducible_pp": gap_irreducible_pp,
        "gap_channel_live": gap_channel_live,
        "gap_split_pending": False,                # ubel #379 has LANDED (GREEN)
        "gap_split_banked_ubel379": gap_split_banked,
        # ubel #386 (RESOLVED RED): floor inflates to 1.310% under VBI=1; prompt-shift now binding
        "irreducible_floor_vbi_pending_ubel386": floor_vbi_pending,
        "irreducible_floor_inflates_vbi_386": floor_inflates_vbi,
        "irreducible_floor_vbi1_central_pct_386": live_floor_central_pct,
        "central_clears_3p2_vbi1_386": CENTRAL_CLEARS_3P2_VBI1_386 if floor_inflates_vbi else True,
        "all_corners_clear_3p2_vbi1_386": (ALL_CORNERS_CLEAR_3P2_VBI1_386 if floor_inflates_vbi
                                           else (None if floor_vbi_pending else True)),
        "prompt_shift_sensitivity_binding_risk_386": prompt_shift_binding_risk,
        "breakeven_prompt_shift_vbi1_tok_386": (BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386 if floor_inflates_vbi
                                                else PRIVATE_PROMPT_SHIFT_BREAKEVEN_TOK),
        # ubel #389 (LANDED 19:15Z): the #386 pessimistic-corner breach is REFUTED on the measured slope
        "ubel389_landed": UBEL389_LANDED,
        "ubel389_pin_breach_pending": UBEL389_PIN_BREACH_PENDING,                 # now False (resolved)
        "ubel389_386_breach_refuted": breach_386_refuted,                         # True
        "ubel389_measured_floor_vbi1_pct": UBEL389_MEASURED_FLOOR_VBI1_PCT,       # 0.5764 (< 0.633)
        "ubel389_pessimistic_breaches_3p2_measured": UBEL389_PESSIMISTIC_BREACHES_3P2_MEASURED,  # 0
        "ubel389_all_corners_clear_3p2_measured": UBEL389_ALL_CORNERS_CLEAR_3P2_MEASURED,        # True
        "ubel389_measured_slope_ratio_to_386": UBEL389_MEASURED_SLOPE_RATIO_TO_386,  # 0.353x
        # OPERATIVE (post-#389) demand-floor state — the leaf is ROBUST on the measured slope
        "irreducible_floor_vbi1_central_pct_operative": operative_floor_vbi1_central_pct,  # 0.5764%
        "all_corners_clear_3p2_vbi1_operative": all_corners_clear_3p2_operative,            # True
        "prompt_shift_sensitivity_binding_risk_operative": prompt_shift_binding_risk_operative,  # False
        "coverage_target_for_3p2_ubel379": COVERAGE_TARGET_FOR_3P2_UBEL379,
        # slope private-OOD robustness (ubel #382, BANKED GREEN)
        "slope_tps_per_coverage_ubel379": SLOPE_TPS_PER_COVERAGE_UBEL379,
        "slope_is_private_robust": slope_private_robust,           # True (banked)
        "slope_tps_per_coverage_private_382": slope_private_tps,   # 437.3
        "slope_flattening_ratio_382": SLOPE_FLATTENING_RATIO_382,
        "flattening_breakeven_382": FLATTENING_BREAKEVEN_382,
        "coverage_target_for_3p2_private_382": COVERAGE_TARGET_FOR_3P2_PRIVATE_382,
        "coverage_target_for_3p2_private_conservative_382": COVERAGE_TARGET_FOR_3P2_PRIVATE_CONSERVATIVE_382,
        "demand_conservative_target_382": DEMAND_CONSERVATIVE_TARGET_382,   # 0.911 BANK THIS
        "demand_budget_frac_conservative_382": DEMAND_BUDGET_FRAC_CONSERVATIVE_382,  # 66.6%
        "demand_budget_frac_central_382": DEMAND_BUDGET_FRAC_CENTRAL_382,            # 38.9%
        # denken #396 (yc5ji486, LANDED RED 20:03Z) — demand-ALONE busts; the zero-flag GO path is CLOSED
        "denken396_landed": DENKEN396_LANDED,
        "denken396_demand_alone_500_green": DENKEN396_DEMAND_ALONE_500_GREEN,        # False
        "denken396_required_dcov_bare_471": DENKEN396_REQUIRED_DCOV_BARE_471,        # 0.02946 (94.9% — fits bare)
        "denken396_required_dcov_bare_471_budget_frac": DENKEN396_REQUIRED_DCOV_BARE_471_BUDGET_FRAC,
        "denken396_required_dcov_under_389_floor": DENKEN396_REQUIRED_DCOV_UNDER_389_FLOOR,  # 0.03244 > 0.031
        "denken396_robust_under_389_slope": DENKEN396_ROBUST_UNDER_389_SLOPE,        # False
        "denken396_required_dcov_on_467": DENKEN396_REQUIRED_DCOV_ON_467,            # 0.0338 (109%)
        "denken396_required_dcov_on_467_budget_frac": DENKEN396_REQUIRED_DCOV_ON_467_BUDGET_FRAC,
        "denken396_demand_alone_busts_even_bare_467": DENKEN396_DEMAND_ALONE_BUSTS_EVEN_BARE_467,  # True
        "denken396_zero_flag_go_path_closed": DENKEN396_ZERO_FLAG_GO_PATH_CLOSED,    # True
        "demand_alone_busts_467": demand_alone_busts_467,
        # ubel #399 (ec7i3z5t, LANDED RED 20:08Z) — NO cheap deployable demand lever; d-cov needs the tree
        "ubel399_landed": UBEL399_LANDED,
        "ubel399_cheap_demand_lever_exists": UBEL399_CHEAP_DEMAND_LEVER_EXISTS,      # False
        "ubel399_monotone_levers_rank_invariant": UBEL399_MONOTONE_LEVERS_RANK_INVARIANT,  # True (no-op)
        "ubel399_mc_max_dcov": UBEL399_MC_MAX_DCOV,                                  # 0.0
        "ubel399_rank_changing_control_fires": UBEL399_RANK_CHANGING_CONTROL_FIRES,  # 0.59 (harness OK)
        "ubel399_frac_of_gap_covered": UBEL399_FRAC_OF_GAP_COVERED,                  # 0%
        "ubel399_dcov_only_from_retrain_or_tree": UBEL399_DCOV_ONLY_FROM_RETRAIN_OR_TREE,  # True
        "ubel399_tree_top1_to_top4_prize": UBEL399_TREE_TOP1_TO_TOP4_PRIZE,          # +0.1286
        "ubel399_coverage_gap_upper_bound": list(UBEL399_COVERAGE_GAP_UPPER_BOUND),  # [0, 0.1097]
        "ubel399_ppl_untouched": UBEL399_PPL_UNTOUCHED,                              # True (binding=deployability)
        "no_cheap_demand_lever": no_cheap_demand_lever,
        # tree net-supply gate (20:19Z) — ubel #401 (coverage ceiling) OR denken #402 (net after verify-M tax)
        "ubel401_tree_coverage_ceiling_input": ubel401_tree_coverage_ceiling,
        "denken402_tree_net_supply_input": denken402_tree_net_supply,
        "ubel401_tree_coverage_ceiling_pending_constant": UBEL401_TREE_COVERAGE_CEILING_PENDING,
        "denken402_tree_net_supply_pending_constant": DENKEN402_TREE_NET_SUPPLY_PENDING,
        "tree_net_supply_green": tree_green,                        # >=1 of ubel #401 / denken #402 green
        "tree_net_supply_both_red": tree_both_red,                  # both red -> demand dead
        "tree_net_supply_pending": tree_pending,                    # held until >=1 lands
        "tree_is_genuinely_new_lever": TREE_IS_GENUINELY_NEW_LEVER_2019Z,  # True
        # verdict (20:19Z): the demand residual has no deployable supplier until the tree NETs it
        "demand_leaf_delivers": demand_leaf_delivers,             # None pending tree (denken #396/ubel #399 RED)
        "leaf_robust_pending": leaf_robust_pending,
        "demand_leaf_pending_tree": demand_leaf_pending_tree,     # held until ubel #401 / denken #402 lands
        "robust_pilot_off_critical_path_383": not PILOT_ON_CRITICAL_PATH_383,  # True (denken #383)
        "binding_constraint": binding,
        "note": (
            "denken #377 SIZED c* and denken #380 (BANKED YELLOW) split it into central (c >= 0.8959, "
            "+0.00565, p_deliver 0.958 >= 0.90 -> DELIVERABLE NOW) and robust (c >= 0.9010, +0.0107, "
            "p_deliver 0.811 < 0.90 -> pending a ~25 A10G-GPU-hr coverage-lift pilot). The kappa-axis is "
            "ROBUST (breakeven 0.1222 << worst c* corner 0.354); the binding uncertainty is the delivery "
            "distribution (defensible fine-tune N(0.016,0.006), not #339's optimistic tail). ubel #382 "
            "(GREEN) CONFIRMED the 489.8 slope is private-robust -> 437.3 private (flattening_ratio 0.893); "
            "bank the CONSERVATIVE private-anchored target ~0.911. BUT denken #383 (RED, 17:53Z): the "
            "demand closer ALONE does NOT reach private-500 on the honest base (residual +49.8 TPS = "
            "+0.0572 Δcov = 1.84x budget) -> this leaf delivers only the RESIDUAL after a supply lift, and "
            "the ~25-GPU-hr pilot is OFF the critical path until the base clears ~487-493. ubel #386 (RED, "
            "17:53Z): the irreducible floor inflates 2.07x -> 1.310% under VBI=1; central still clears 3.2% "
            "(+1.89pp) so the leaf is not dead, but all_corners_clear_3p2_vbi1=False and the breakeven "
            "prompt shift halves (+253 -> +119 tok) -> prompt-shift sensitivity was relayed as a BINDING "
            f"risk. BUT ubel #389 (LANDED 19:15Z) MEASURED the per-L attention identity floor under VBI=1 "
            f"at {UBEL389_MEASURED_FLOOR_VBI1_PCT:g}% (< 0.633% off-VBI, NOT 1.310%); 0 corners breach 3.2% "
            f"and the measured slope is {UBEL389_MEASURED_SLOPE_RATIO_TO_386:g}x the #386 interpolation -> "
            "the #386 breach was a CONSERVATIVE-SLOPE ARTIFACT (the irreducible-floor risk is OFF). ★ BUT "
            "the 20:19Z cluster RE-PRICED the demand leaf DOWN: denken #396 (LANDED RED 20:03Z) -> demand-"
            f"ALONE busts EVEN BARE on the corrected {REALIZED_DEPLOYED_STRICT_393:g} base (required_dcov ~"
            f"{DENKEN396_REQUIRED_DCOV_ON_467:g} = {DENKEN396_REQUIRED_DCOV_ON_467_BUDGET_FRAC*100:g}% of "
            f"the +{BUDGET_336_PLUS_031:g} budget; on bare 471.42 the #389 floor already pushed +"
            f"{DENKEN396_REQUIRED_DCOV_BARE_471:g} to {DENKEN396_REQUIRED_DCOV_UNDER_389_FLOOR:g} > 0.031), "
            "so the clean zero-flag GO path is CLOSED. ubel #399 (LANDED RED 20:08Z) -> there is NO cheap "
            "deployable demand lever: every monotone draft-head lever (temperature, affine) is a RANK-"
            f"INVARIANT no-op (MC max|Δcov|={UBEL399_MC_MAX_DCOV:g}, frac_of_gap_covered="
            f"{UBEL399_FRAC_OF_GAP_COVERED*100:g}%; a rank-changing control fires at "
            f"{UBEL399_RANK_CHANGING_CONTROL_FIRES:g} -> the zeros are physics). So the demand d-cov "
            f"(#392's +0.0117 sliver and beyond) can ONLY be SUPPLIED by the TREE (the locked +"
            f"{UBEL399_TREE_TOP1_TO_TOP4_PRIZE:g} top-1->top-4 prize; coverage gap "
            f"{list(UBEL399_COVERAGE_GAP_UPPER_BOUND)} is an UPPER bound) — a genuinely-new-lever. PPL is "
            "untouched throughout (spec-decode emits the target greedy token; binding=deployability, never "
            "the gate). The central-GREEN coverage-retrain sizing above is PROVENANCE (a drafter retrain is "
            "forbidden). OPERATIVE: the demand leaf DELIVERS only if the tree NETs the d-cov -> PENDING "
            "ubel #401 (top-8/16 coverage ceiling) OR denken #402 (net after the verify-M step-time tax on "
            f"{REALIZED_DEPLOYED_STRICT_393:g}). Necessary-but-insufficient on its own (denken #383); it "
            "transfers FROM the supply-side base — see supply_side_base_analysis, the binding GO axis."
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
# Terminal GO-flip gate (advisor 20:19Z REPRICE).  The 19:25Z cb3-deployability pair RESOLVED RED and is
# SUPERSEDED: kanna #394 (RED) made the +32.65 cb3 HEADLINE lift PPL-DEAD, denken #396 (RED) closed the
# demand-alone zero-flag GO, ubel #399 (RED) closed the cheap-monotone-lever escape.  The route did NOT
# close — it is RE-PRICED toward "conservative-k cb3 + a NET-positive tree", with the terminal GO now a
# clean AND-product of the two RE-PRICED leaves:
#   SUPPLY leg = kanna #403 — the PPL-SAFE conservative-k cb3 supply re-cost (largest k with held-out
#                worst-seed <= 2.41).  It resolves `supply_base_enables_500` inside supply_side_base_analysis.
#   DEMAND leg = ubel #401 (locked top-8/16 tree coverage ceiling) OR denken #402 (does the tree NET d-cov
#                after its verify-M step-time tax on 467.48?).  It resolves `demand_leaf_delivers` inside
#                demand_side_route_analysis.
# terminal_go GREEN iff kanna #403 (supply leg) AND >=1 of {ubel #401, denken #402} (demand leg) land
# green; RED iff either leg red; else PENDING (held None).  "Sharpen, don't flip."
# --------------------------------------------------------------------------- #
def terminal_go_gate_analysis(supply_base_enables_500: bool | None,
                              demand_leaf_delivers: bool | None,
                              kanna403_ppl_safe_supply: str = "pending",
                              ubel401_tree_coverage_ceiling: str = "pending",
                              denken402_tree_net_supply: str = "pending") -> dict[str, Any]:
    """20:19Z REPRICE terminal GO gate: GO = (supply leg) AND (demand leg), held None until both land.

    Takes the two ALREADY-COMPUTED re-priced leaf verdicts (so terminal_go is the leaves' None-aware
    AND-product BY CONSTRUCTION) plus the three feeder states (for the pending-gate list).  The SUPPLY
    leg is kanna #403 (PPL-safe conservative-k cb3 supply re-cost; the +32.65 HEADLINE is PPL-DEAD per
    kanna #394 RED); the DEMAND leg is the tree net-supply (ubel #401 coverage ceiling OR denken #402 net
    after the verify-M tax) — the ONLY deployable d-cov supplier left after denken #396 / ubel #399 RED.

    SUPERSEDED 19:25Z pair, banked as the re-pricing rationale (no longer runtime gates):
      - kanna #394 (LANDED RED): cb3's +32.65 honest HEADLINE lift is PPL-DEAD (winner's-curse selection
        to k=243-246; held-out 2.4223 + OOD ShareGPT 2.4270 breach the 2.42 gate).  cb3 stays deployable
        at a CONSERVATIVE k (~232 clears ~2.39) at a smaller UN-COSTED lift -> kanna #403.
      - denken #396 (LANDED RED): demand-ALONE busts even bare on 467.48 (0.0338 = 109% of the budget).
      - ubel #399 (LANDED RED): no cheap deployable demand lever (monotone levers rank-invariant no-ops).

    Each feeder is one of "pending" | "green" | "red".  Returns terminal_go in {green, red, pending} and
    go_reachable in {True, False, None}.
    """
    supply_leg = supply_base_enables_500    # None / True / False (kanna #403 resolves it in the supply leaf)
    demand_leg = demand_leaf_delivers       # None / True / False (ubel #401 / denken #402 in the demand leaf)

    # None-aware AND-product: RED if either leg red; GREEN iff both green; else PENDING (held).
    if supply_leg is False or demand_leg is False:
        terminal_go = "red"
        go_reachable: bool | None = False
    elif supply_leg is True and demand_leg is True:
        terminal_go = "green"
        go_reachable = True
    else:
        terminal_go = "pending"
        go_reachable = None

    kanna403_pending = kanna403_ppl_safe_supply == "pending"
    tree_green = (ubel401_tree_coverage_ceiling == "green" or denken402_tree_net_supply == "green")
    tree_both_red = (ubel401_tree_coverage_ceiling == "red" and denken402_tree_net_supply == "red")
    tree_pending = not (tree_green or tree_both_red)

    pending_gates: list[str] = []
    if kanna403_pending:
        pending_gates.append("kanna#403_ppl_safe_conservative_k_supply_recost")      # SUPPLY leg
    if tree_pending:
        pending_gates.append("ubel#401_or_denken#402_tree_net_supply")               # DEMAND leg

    supply_leg_state = ("green" if supply_leg is True else "red" if supply_leg is False else "pending")
    demand_leg_state = ("green" if demand_leg is True else "red" if demand_leg is False else "pending")

    return {
        "source": ("advisor 20:19Z REPRICE — terminal GO = (supply leg kanna #403 PPL-safe) AND "
                   "(demand leg ubel #401 OR denken #402 tree net-supply)"),
        "go_formula": ("terminal_go <=> kanna#403_ppl_safe_supply AND "
                       "(ubel#401 OR denken#402 tree_net_supply)"),
        # the two re-priced legs (already computed in the leaves)
        "supply_leg_enables_500": supply_leg,
        "demand_leg_delivers": demand_leg,
        "supply_leg_state": supply_leg_state,
        "demand_leg_state": demand_leg_state,
        # the three feeders (20:19Z)
        "kanna403_ppl_safe_supply": kanna403_ppl_safe_supply,
        "ubel401_tree_coverage_ceiling": ubel401_tree_coverage_ceiling,
        "denken402_tree_net_supply": denken402_tree_net_supply,
        "kanna403_pending": kanna403_pending,
        "tree_net_supply_green": tree_green,                # >=1 of ubel #401 / denken #402 green
        "tree_net_supply_both_red": tree_both_red,          # both red -> demand leg dead
        "tree_net_supply_pending": tree_pending,            # held until >=1 lands
        # verdict
        "terminal_go": terminal_go,                         # green / red / pending
        "go_reachable": go_reachable,                       # True / False / None (the AND-product)
        "terminal_go_confirmed": terminal_go == "green",    # TERMINAL-GO flip
        "terminal_go_blocked": terminal_go == "red",        # a leg landed red
        "terminal_go_pending": terminal_go == "pending",    # held None
        "terminal_go_pending_gates": pending_gates,         # kanna #403 + >=1 of ubel #401 / denken #402
        "reprice_go_flip_gates": list(REPRICE_2019Z_GO_FLIP_GATES),
        "tree_net_supply_probes": list(REPRICE_2019Z_TREE_NET_SUPPLY_PROBES),
        # SUPERSEDED 19:25Z pair (banked as the re-pricing rationale)
        "superseded_1925z_pair": REPRICE_2019Z_SUPERSEDES_1925Z_PAIR,
        "kanna394_headline_ppl_dead": CB3_HEADLINE_LIFT_PPL_DEAD_394,            # +32.65 PPL-dead
        "denken396_demand_alone_closed": DENKEN396_ZERO_FLAG_GO_PATH_CLOSED,     # demand-alone busts
        "ubel399_no_cheap_demand_lever": not UBEL399_CHEAP_DEMAND_LEVER_EXISTS,  # cheap-lever escape closed
        "cb3_deployable_at_conservative_k_394": CB3_DEPLOYABLE_AT_CONSERVATIVE_K_394,  # the re-price basis
        "tree_is_genuinely_new_lever": TREE_IS_GENUINELY_NEW_LEVER_2019Z,
        "cb3_insample_ppl_margin_372": CB3_INSAMPLE_PPL_MARGIN_372,
        "cb3_lift_honest_denken392": CB3_LIFT_HONEST_DENKEN392,    # +32.65 headline (PPL-dead, provenance)
        "note": (
            "20:19Z REPRICE: the 19:25Z cb3-deployability pair RESOLVED RED and is SUPERSEDED — kanna #394 "
            f"(RED) made the +{CB3_LIFT_HONEST_DENKEN392:g} cb3 HEADLINE lift PPL-DEAD (winner's-curse "
            f"selection to k={KANNA394_SELECTED_K_RANGE[0]}-{KANNA394_SELECTED_K_RANGE[1]}; held-out "
            f"{KANNA394_HELDOUT_WORST_SEED_PPL:g} + OOD {KANNA394_OOD_SHAREGPT_PPL:g} breach "
            f"{KANNA394_PPL_GATE:g}), denken #396 (RED) closed demand-alone (busts even bare on "
            f"{REALIZED_DEPLOYED_STRICT_393:g}), ubel #399 (RED) closed the cheap monotone levers. The "
            "route did NOT close — it is RE-PRICED toward 'conservative-k cb3 + a NET-positive tree'. The "
            "terminal GO is the None-aware AND-product of the two re-priced legs: SUPPLY = kanna #403 (the "
            f"PPL-SAFE conservative-k re-cost; cb3 stays deployable at k={KANNA394_CONSERVATIVE_K_232} ~"
            f"{KANNA394_CONSERVATIVE_K_HELDOUT_PPL:g} held-out, lift UN-COSTED) AND DEMAND = >=1 of ubel "
            f"#401 (top-8/16 tree coverage ceiling, sizing the +{UBEL399_TREE_TOP1_TO_TOP4_PRIZE:g} prize) "
            "/ denken #402 (does the tree NET d-cov after its verify-M step-time tax?). GREEN iff both legs "
            "green; RED iff either red; HELD None until both land. Sharpen, don't flip."
        ),
    }


# --------------------------------------------------------------------------- #
# Composite verdict (advisor 17:03Z-17:53Z): GO = (supply-side public-strict base ENABLES 500) x
# (demand closer DELIVERS the residual).  The SUPPLY leaf (wirbel #378 honest base <=480.7-today;
# denken #383 RED CONFIRMED demand-alone insufficient -> +17.2-TPS lift required first; pending a
# measured lift from wirbel #384 OR lawine #388) is the NOW-BINDING axis; the DEMAND leaf
# (denken #377 sized + denken #380 two-tier + ubel #382 slope-GREEN) is RESOLVED at central=GREEN /
# robust=pending-pilot.  Identity reachability folds into the SUPPLY leaf as the strict-byte-exact
# COMPLIANCE prerequisite (stark #376/#381).  Route A (sub-int4 eta-axis) is computed for continuity
# but DEFLATED and EXCLUDED from the GO.  Structural: private <= public, so a public base < 500 can
# make private-500 UNREACHABLE by the demand closer ALONE at any coverage.
# --------------------------------------------------------------------------- #
def composite_verdict(measured_ppl: float | None, b_star: float,
                      lmhead_eta: float | None = None, *,
                      supply_lift_available_tps: float | None = None,
                      demand_reaches_500_on_floor: str = "pending",
                      supply_lift_required_tps: float | None = None,
                      robust_pilot: str = "pending",
                      gap_addressable_pp: float | None = None,
                      stark381_decode: str = "pending",
                      irreducible_floor_survives_vbi: str = "pending",
                      kanna403_ppl_safe_supply: str = "pending",
                      ubel401_tree_coverage_ceiling: str = "pending",
                      denken402_tree_net_supply: str = "pending") -> dict[str, Any]:
    """AND-product GO (advisor 20:19Z REPRICE): (supply leg) AND (demand leg), held None until both land.

    The 19:25Z cb3-deployability pair RESOLVED RED and is SUPERSEDED: kanna #394 (RED) made the +32.65 cb3
    HEADLINE lift PPL-DEAD, denken #396 (RED) closed demand-alone, ubel #399 (RED) closed the cheap
    monotone levers.  The route did NOT close — it is RE-PRICED toward "conservative-k cb3 + a NET-positive
    tree".  The terminal GO is the None-aware AND-product of the two re-priced leaves:
      SUPPLY leg = kanna #403 (PPL-safe conservative-k cb3 supply re-cost) — resolves supply_base_enables_500.
      DEMAND leg = ubel #401 (tree coverage ceiling) OR denken #402 (tree net after verify-M tax) — resolves
                   demand_leaf_delivers.
    GREEN iff both legs green; RED iff either red; HELD None until both land.  Route A (sub-int4 eta-axis)
    is retained for continuity but DEFLATED/excluded from the GO.
    """
    route_a = verdict_given_ppl(measured_ppl, b_star, lmhead_eta)              # sub-int4 eta-axis (deflated)
    supply = supply_side_base_analysis(supply_lift_available_tps, demand_reaches_500_on_floor,
                                       supply_lift_required_tps, stark381_decode,
                                       kanna403_ppl_safe_supply)              # SUPPLY leaf (binding; kanna #403 leg)
    demand = demand_side_route_analysis(robust_pilot, gap_addressable_pp,
                                        irreducible_floor_survives_vbi,
                                        ubel401_tree_coverage_ceiling,
                                        denken402_tree_net_supply)            # DEMAND leaf (tree net-supply leg)
    ident_reach = identity_reachability_analysis(stark381_decode)             # identity compliance (folds into supply)

    a_reachable = route_a["strict_500_reachable_via_known_levers"]    # sub-int4 eta-axis standalone (deflated)
    supply_enables = supply["supply_base_enables_500"]               # True / False / None (kanna #403 leg)
    demand_delivers = demand["demand_leaf_delivers"]                 # True / False / None (tree net-supply leg)
    identity_reachable = ident_reach["identity_reachable_env_or_rebuild"]  # True (env or rebuild)

    gate = terminal_go_gate_analysis(supply_enables, demand_delivers, kanna403_ppl_safe_supply,
                                     ubel401_tree_coverage_ceiling,
                                     denken402_tree_net_supply)      # GO = the two re-priced legs' AND-product

    # AND-product GO, None-aware: the terminal gate ANDs the two RE-PRICED leaves (supply leg kanna #403,
    # demand leg ubel #401 / denken #402).  False iff either leg red; True iff both green; else None (held).
    reachable: bool | None = gate["go_reachable"]
    pending = reachable is None

    # GO-gating pending inputs (advisor 20:19Z): the terminal gate's two re-priced legs.  The SUPPLY leg is
    # kanna #403 (the PPL-safe conservative-k cb3 re-cost; the +32.65 headline is PPL-DEAD per kanna #394
    # RED).  The DEMAND leg is the tree net-supply (>=1 of ubel #401 / denken #402; denken #396 RED closed
    # demand-alone, ubel #399 RED closed the cheap monotone levers).  denken #383 (floor) surfaces only if
    # itself pending.  The robust demand pilot is now PROVENANCE-only (a forbidden retrain) and stark #381
    # e2e identity cost remain REFINEMENTS; ubel #389 LANDED (refuted the #386 breach -> no longer pending).
    pending_inputs: list[str] = []
    if supply["denken383_pending"]:
        pending_inputs.append("denken#383_demand_route_reaches_500_on_deployable_floor")
    pending_inputs.extend(gate["terminal_go_pending_gates"])   # kanna #403 (supply leg) + tree (demand leg)
    refinement_inputs: list[str] = []
    if demand["robust_pilot_pending"]:
        refinement_inputs.append("coverage_lift_pilot_robust_tier_provenance_only_forbidden_retrain_383")
    if demand["irreducible_floor_vbi_pending_ubel386"]:
        refinement_inputs.append("ubel#386_irreducible_floor_under_vbi1")
    identity_cost_pending = ident_reach["cost_branch_pending"]
    if identity_cost_pending:
        refinement_inputs.append("stark#381_decode_width_identity_cost_branch")

    cost_phrase = (
        f"identity reachable via {ident_reach['cost_branch']} "
        f"({ident_reach['rebuild_line_items']} rebuild line-item(s))"
        if not identity_cost_pending else
        "identity reachable (env-or-rebuild; stark #381 will fix the cost: env@decode=1 rebuild "
        "vs Marlin-gated=2 rebuilds)")
    demand_phrase = (
        "demand leg needs the TREE net-supply (denken #396 RED closed demand-alone; ubel #399 RED the "
        f"cheap monotone levers; sized closer ~{DEMAND_CONSERVATIVE_TARGET_382} is PROVENANCE — a drafter "
        "retrain is forbidden)")

    if reachable is True:
        binding = ("private_500_REACHABLE_supply_leg_kanna403_x_demand_leg_tree__"
                   f"{supply['binding_constraint']}")
    elif reachable is False:
        binding = (f"demand_leg_dead_tree_nets_nothing__{demand['binding_constraint']}"
                   if demand_delivers is False
                   else f"supply_leg_dead_no_ppl_safe_conservative_k_lift__{supply['binding_constraint']}")
    else:
        binding = f"PENDING_reprice__supply_leg_kanna403_AND_demand_leg_tree__gates={','.join(pending_inputs)}"

    if pending:
        # HELD on the 20:19Z REPRICE terminal GO-flip gate: kanna #403 (supply leg) AND >=1 of {ubel #401,
        # denken #402} (demand-leg tree net-supply).  The 19:25Z cb3-deployability pair is SUPERSEDED RED.
        verdict_text = (
            "private-500 reachability HELD ON THE 20:19Z REPRICE TERMINAL GO-FLIP GATE — GO = (supply leg "
            "kanna #403 PPL-safe) AND (demand leg ubel #401 OR denken #402 tree net-supply). The 19:25Z "
            f"cb3-deployability pair RESOLVED RED and is SUPERSEDED: kanna #394 (RED) made the +"
            f"{supply['cb3_lift_honest']:g} cb3 HEADLINE lift PPL-DEAD (held-out "
            f"{KANNA394_HELDOUT_WORST_SEED_PPL:g} + OOD {KANNA394_OOD_SHAREGPT_PPL:g} breach "
            f"{KANNA394_PPL_GATE:g}), denken #396 (RED) closed demand-alone (busts even bare on "
            f"{supply['supply_base_today_tps']:g}: required Δcov {DENKEN396_REQUIRED_DCOV_ON_467:g} = "
            f"{DENKEN396_REQUIRED_DCOV_ON_467_BUDGET_FRAC*100:g}% of budget), ubel #399 (RED) closed the "
            "cheap monotone levers. The route did NOT close — it is RE-PRICED toward 'conservative-k cb3 + "
            f"a NET-positive tree'. SUPPLY base {supply['supply_base_today_tps']:g} (wirbel #393 corrected; "
            f"gap_to_500 {supply['deficit_to_500_today_tps']:g}); the {demand_phrase}; {cost_phrase}. GO "
            "flips True iff kanna #403 (PPL-safe conservative-k supply, largest k with held-out worst-seed "
            f"<= {KANNA403_HELDOUT_WORST_SEED_TARGET:g}) AND >=1 of ubel #401 (top-8/16 tree coverage "
            "ceiling) / denken #402 (tree NETs d-cov after the verify-M step-time tax) land green. Route A "
            f"(sub-int4 eta-axis): DEFLATED/excluded. GO-gating pending: {pending_inputs}"
            f"{' (+refinements ' + str(refinement_inputs) + ')' if refinement_inputs else ''}."
        )
    elif reachable is True:
        verdict_text = (
            "private-500 REACHABLE via the 20:19Z re-priced route: SUPPLY leg kanna #403 (PPL-safe "
            "conservative-k cb3 supply re-cost) AND DEMAND leg tree net-supply (>=1 of ubel #401 / denken "
            f"#402) BOTH landed green ({supply['binding_constraint']}; {demand['binding_constraint']}). The "
            f"{demand_phrase}; {cost_phrase}. Flag as approval-gated a10g candidate (#319)."
        )
    elif demand_delivers is False:
        verdict_text = (
            "private-500 NOT reachable via known levers: the DEMAND leg is dead — the tree NETs nothing "
            f"(ubel #401 AND denken #402 both RED: {demand['binding_constraint']}), and denken #396 (RED) "
            "already closed demand-alone + ubel #399 (RED) the cheap monotone levers. The d-cov has no "
            "deployable supplier -> genuinely-new-method problem."
        )
    else:
        verdict_text = (
            "private-500 NOT reachable via known levers: the SUPPLY leg is dead — kanna #403 found NO "
            f"PPL-safe conservative-k cb3 lift that clears (the +{supply['cb3_lift_honest']:g} headline is "
            f"PPL-DEAD per kanna #394 RED), so supply stays at the {supply['supply_base_today_tps']:g} base "
            "(<500) and the demand closer CANNOT close private-500 alone (private <= public). A genuinely-"
            "new supply lift would be required first."
        )

    return {
        "primary_route": "supply_x_demand_and_product",
        "eta_axis_deflated": True,
        "go_formula": ("private_500_GO <=> (supply_base_enables_500) AND "
                       "(demand_closer_delivers_residual)"),
        "private_500_reachable_via_known_levers": reachable,
        "verdict_pending": pending,
        "pending_inputs": pending_inputs,                       # GO-gating: kanna #403 + tree (ubel #401 / denken #402)
        "refinement_inputs": refinement_inputs,                 # robust pilot (provenance) + ubel#386 + stark#381
        "cost_refinement_inputs": refinement_inputs,            # back-compat alias
        "binding_constraint": binding,
        "verdict_text": verdict_text,
        # the two GO factors: supply base enables 500  x  demand closer delivers residual
        "supply_base_enables_500": supply_enables,
        "demand_closer_delivers": demand_delivers,
        "demand_alone_may_be_insufficient": supply["demand_alone_may_be_insufficient"],
        # identity reachability (folds into the supply leaf as the strict compliance prerequisite)
        "identity_reachable_env_or_rebuild": identity_reachable,
        "identity_cost_branch": ident_reach["cost_branch"],
        "identity_cost_branch_pending": identity_cost_pending,
        "identity_rebuild_line_items": ident_reach["rebuild_line_items"],
        "identity_factor": ident_reach,
        # demand leaf two-tier
        "demand_central_green": demand["deliver_central_green"],
        "demand_robust_resolved": demand["deliver_robust_resolved"],
        "demand_robust_pending": demand["robust_pilot_pending"],
        "demand_conservative_target": demand["demand_conservative_target_382"],
        "demand_leaf": {
            "delivers": demand_delivers,
            "robust_pending": demand["leaf_robust_pending"],
            "pending_tree": demand["demand_leaf_pending_tree"],
            "binding": demand["binding_constraint"],
        },
        # supply leaf (binding GO axis; kanna #403 PPL-safe leg)
        "supply_leaf": {
            "enables_500": supply_enables,
            "pending": supply["supply_pending"],
            "binding": supply["binding_constraint"],
            "honest_base_today_tps": supply["supply_base_today_tps"],
            "clears_500_today": supply["supply_base_clears_500_today"],
            "supply_alone_clears_500_with_cb3": supply["supply_alone_clears_500_with_cb3"],  # False (headline PPL-dead)
            "base_plus_cb3_honest": supply["base_plus_cb3_honest"],
            "kanna403_pending": supply["kanna403_ppl_safe_supply_pending"],
        },
        # TERMINAL GO-flip gate (advisor 20:19Z REPRICE): kanna #403 (supply leg) AND >=1 of {ubel #401,
        # denken #402} (demand-leg tree net-supply).  Supersedes the 19:25Z cb3-deployability pair (RED).
        "terminal_go": gate["terminal_go"],                     # green / red / pending
        "terminal_go_confirmed": gate["terminal_go_confirmed"], # TERMINAL-GO flip (both legs green)
        "terminal_go_blocked": gate["terminal_go_blocked"],     # a leg landed red
        "terminal_go_pending": gate["terminal_go_pending"],     # held None
        "terminal_go_pending_gates": gate["terminal_go_pending_gates"],
        "reprice_go_flip_gates": gate["reprice_go_flip_gates"],
        "tree_net_supply_green": gate["tree_net_supply_green"],       # >=1 of ubel #401 / denken #402 green
        "tree_net_supply_both_red": gate["tree_net_supply_both_red"], # both red -> demand leg dead
        "tree_net_supply_pending": gate["tree_net_supply_pending"],   # held until >=1 lands
        "kanna403_ppl_safe_supply": gate["kanna403_ppl_safe_supply"],
        "ubel401_tree_coverage_ceiling": gate["ubel401_tree_coverage_ceiling"],
        "denken402_tree_net_supply": gate["denken402_tree_net_supply"],
        "superseded_1925z_pair": gate["superseded_1925z_pair"],
        "terminal_go_factor": gate,
        # sub-int4 eta-axis standalone — DEFLATED, excluded from the GO (kept for continuity)
        "route_a_supply_side": {
            "reachable": a_reachable,
            "pending": route_a["verdict_pending"],
            "binding": route_a["binding_constraint"],
            "in_go_product": False,
            "leans_dead_reason": "sub-int4 UNIFORM eta-axis standalone deflated (#373/#375/#378) + kanna "
                                 "#374 (fusion lever closed, Triton not byte-exact) -> un-deployable "
                                 "standalone path. NOTE: lawine #372's sensitivity-weighted mixed-precision "
                                 "ALLOCATION is GREEN/alive, but its +32.65 HEADLINE lift is PPL-DEAD (kanna "
                                 "#394 RED); the PPL-safe conservative-k re-cost is the SUPPLY-leg de-risker "
                                 "(kanna #403, pending), NOT a Route A revival.",
        },
        "route_a_detail": route_a,
        "supply_leaf_detail": supply,
        "demand_leaf_detail": demand,
    }


# --------------------------------------------------------------------------- #
# 22:08Z CAPSTONE: max-equivalent-TPS frontier rollup over the equivalence ladder.
# The live objective (#407 human, 21:13Z; advisor re-point 22:08Z) is max TPS subject
# to STRICT byte-exact greedy-token-equivalence (served identity == 1.0).  This rollup
# walks the equivalence ladder node-by-node, tags each MEASURED vs MODELED, carries the
# per-node identity/PPL/deployability gates, and returns the max equivalent TPS we can
# (a) stand behind MEASURED today and (b) project MODELED with the in-flight feeders.
# The deployed 481.53 fast path is OFF this ladder (served identity 0.9966, NOT 1.0).
# Ladder:  467.48 floor (MEASURED) -> selective-recompute (MODELED ~478.93) -> +cb3
# (equivalence-neutral, do NOT naive-sum) -> -fixed-overhead-floor reduction (0 today).
# --------------------------------------------------------------------------- #
def equivalent_tps_frontier_rollup(*,
                                   selective_recompute_measured_tps: float | None = None,
                                   cb3_additivity_gap_tps: float | None = None,
                                   floor_reduction_tps: float | None = None,
                                   deployability_surface: str | None = None) -> dict[str, Any]:
    """Roll up the equivalence ladder into a single max-equivalent-TPS frontier.

    The live objective (advisor 22:08Z re-point of the #407 21:13Z human directive):
    MAX TPS subject to STRICT byte-exact greedy-token-equivalence (served identity 1.0).
    22:26Z folds in the two banked feeders lawine #417 (cb3 ADDITIVE + deployable +
    bracket) and land #414 (lm_head FREE under the self-referential scorer).

    Feeder inputs (CLI-resolvable; default to the MODELED state with the 22:26Z banks in):
      selective_recompute_measured_tps  — stark #412 MEASURED equivalent TPS for the
          fast-attn-everywhere + selective-higher-precision-recompute config (#397).
          None -> the MODELED denken #413 point 478.93 (= 481.53 - 2.6).  identity 1.0.
      cb3_additivity_gap_tps  — kanna #416 MEASURED gap that TIGHTENS the cb3 stack (the
          read-shrink partly overlaps the recompute re-reads on the flagged steps).  cb3 is
          ALREADY additive-CONFIRMED (lawine #417), so None just leaves the headline as the
          banked bracket [492.08, 494.08]; the gap only narrows it.
      floor_reduction_tps  — wirbel #415 MEASURED reduction of the 146.30us/12.01% fixed
          overhead floor (#408).  None/0 -> no floor credit yet.
      deployability_surface — lawine #417 {green|red|pending}; None -> the BANKED green
          (22:26Z: 7 served files, 41.8 GPU-min identity-verify, reversible, human-gated).

    Returns the ladder (5 nodes, each tagged measured/modeled + gates) plus the headline
    frontier numbers: max_equivalent_tps_measured (stand-behind-today 467.48) and the
    modeled fastest-equivalent bracket [492.08, 494.08] (lawine #417) that BEATS 481.53.
    """
    if deployability_surface is None:
        deployability_surface = "green" if LAWINE417_DEPLOYABLE_GREEN else "pending"

    recompute_measured = selective_recompute_measured_tps is not None
    gap_measured = cb3_additivity_gap_tps is not None
    floor_measured = floor_reduction_tps is not None
    any_feeder_measured = recompute_measured or gap_measured or floor_measured

    # --- Node 0: FLOOR (MEASURED, identity 1.0) -----------------------------
    floor_tps = EQUIV_FLOOR_TPS                                   # 467.48 (wirbel #393)
    node0 = {
        "node": "floor_blanket_strict_attention",
        "tps": floor_tps,
        "status": "measured",
        "identity": EQUIV_FLOOR_IDENTITY,                         # 1.0
        "equivalence_neutral": True,
        "source": "wirbel #393 (0q7ynumg) — blanket-strict batch-invariant attention every step",
        "gate": "MEASURED served identity 1.0; the deployable strict floor.",
    }

    # --- Node 1: SELECTIVE HIGHER-PRECISION RECOMPUTE (#397) -----------------
    # fast attention everywhere + recompute the ~23.6% near-tie <=eps-flagged steps at
    # higher precision -> equivalence restored BY CONSTRUCTION (identity 1.0).
    if recompute_measured:
        recompute_tps = float(selective_recompute_measured_tps)
        recompute_status = "measured"
        recompute_src = "stark #412 (MEASURED local research prototype on A10G)"
    else:
        recompute_tps = SELECTIVE_RECOMPUTE_MODELED_TPS          # 478.93 = 481.53 - 2.6 (denken #413)
        recompute_status = "modeled"
        recompute_src = ("MODELED denken #413 point 478.93 (= 481.53 - 2.6 TPS tax on the ~23.6% flagged "
                         "steps; se8mf9ax); lawine #417 bracket [476.48,478.48]; stark #412 measuring")
    recompute_in_band = SELECTIVE_RECOMPUTE_BAND_TPS[0] <= recompute_tps <= SELECTIVE_RECOMPUTE_BAND_TPS[1]
    node1 = {
        "node": "selective_higher_precision_recompute_397",
        "tps": recompute_tps,
        "status": recompute_status,
        "identity": SELECTIVE_RECOMPUTE_IDENTITY,                 # 1.0 by construction
        "equivalence_neutral": True,
        "flagged_frac": SELECTIVE_RECOMPUTE_FLAGGED_FRAC,         # ~0.236
        "tie_identifiable_from_fast_path": SELECTIVE_RECOMPUTE_TIE_IDENTIFIABLE_405,
        "modeled_band_tps": list(SELECTIVE_RECOMPUTE_BAND_TPS),
        "in_modeled_band": recompute_in_band,
        "denken413_point_tps": DENKEN413_RECOMPUTE_POINT_TPS,    # 478.93
        "lawine417_bracket_tps": list(SELECTIVE_RECOMPUTE_BRACKET_417_TPS),   # [476.48, 478.48]
        "source": recompute_src,
        "gate": ("identity 1.0 BY CONSTRUCTION (recompute restores the flagged ties); tie set readable "
                 "from the fast path (stark #405). TPS pending stark #412; denken #418 testing tax<2.6."),
    }

    # --- Node 2: +cb3 BODY-READ SHRINK (kanna #403, ADDITIVE-confirmed lawine #417) --
    # cb3 changes BYTES READ in the verify body, NOT tokens emitted -> identity-neutral.
    # 22:26Z: lawine #417 CONFIRMED the cb3 stack is ADDITIVE onto the recompute point, so
    # the stack is BANKED (not "do-not-sum/pending").  kanna #416 will MEASURE the exact
    # additivity_gap to TIGHTEN the [492.08, 494.08] bracket — a refinement, not a gate.
    cb3_lift = CB3_CONSERVATIVE_LIFT_TPS_403                      # +15.60 (k*=229, PPL-safe)
    naive_stack_tps = recompute_tps + cb3_lift                   # additive (lawine #417) point estimate
    if gap_measured:
        gap: float | None = float(cb3_additivity_gap_tps)
        cb3_point_tps = naive_stack_tps - gap                    # kanna #416 tightens the exact overlap
        cb3_src = (f"kanna #403 +{cb3_lift:g} cb3 (iv9i2wks) ADDITIVE on the recompute point "
                   f"(lawine #417 confirmed), NET of kanna #416 measured additivity_gap {gap:g} TPS")
    else:
        gap = None
        cb3_point_tps = naive_stack_tps                          # additive-confirmed (lawine #417) -> banked
        cb3_src = (f"kanna #403 +{cb3_lift:g} cb3 (iv9i2wks) ADDITIVE on the recompute point — CONFIRMED "
                   "by lawine #417 (2mv6ssw4); kanna #416 will PRICE the exact additivity_gap (tightening)")
    # The stack is MEASURED-grade only when BOTH the recompute TPS (stark #412) AND the exact
    # additivity gap (kanna #416) are measured; with recompute-only it is additive-CONFIRMED but
    # the precise stacked value is still +-kanna #416's gap -> keep it MODELED (banked, not measured).
    cb3_status = "measured" if (recompute_measured and gap_measured) else "modeled"
    node2 = {
        "node": "cb3_body_read_shrink_403",
        "tps": cb3_point_tps,
        "naive_stack_tps": naive_stack_tps,
        "cb3_lift_tps": cb3_lift,
        "additivity_gap_tps": gap,
        "status": cb3_status,
        "identity": node1["identity"],                           # cb3 equivalence-neutral -> inherits 1.0
        "equivalence_neutral": CB3_403_EQUIVALENCE_NEUTRAL,      # True
        "ppl_safe": CB3_403_PPL_SAFE,                            # k*=229 holds held-out worst-seed <=2.41
        "k_star": CB3_403_K_STAR,                                # 229
        "banked": True,                                          # 22:26Z: additive CONFIRMED (lawine #417)
        "additive_confirmed_lawine417": LAWINE417_CB3_ADDITIVE_CONFIRMED,
        "exact_gap_pending_kanna416": gap is None,
        "lawine417_bracket_tps": list(FASTEST_EQUIVALENT_BRACKET_TPS),   # [492.08, 494.08] banked headline
        "source": cb3_src,
        "gate": ("cb3 is PPL-safe (k*=229, held-out worst-seed <=2.41) AND equivalence-neutral (body "
                 "bytes, not tokens). ADDITIVE confirmed (lawine #417) -> stack BANKED; kanna #416 tightens."),
    }

    # --- Node 3: -FIXED-OVERHEAD-FLOOR reduction (wirbel #415) ---------------
    reduction = floor_reduction_tps if floor_measured else FLOOR_REDUCTION_TPS_UNTIL_415  # 0.0
    node3_tps = cb3_point_tps + reduction
    node3_status = "measured" if (floor_measured and cb3_status == "measured") else "modeled"
    node3 = {
        "node": "fixed_overhead_floor_reduction_415",
        "tps": node3_tps,
        "floor_reduction_tps": reduction,
        "status": node3_status,
        "identity": node2["identity"],
        "equivalence_neutral": True,
        "fixed_floor_us": FIXED_OVERHEAD_FLOOR_US_408,           # 146.30
        "fixed_floor_frac": FIXED_OVERHEAD_FLOOR_FRAC_408,       # 0.1201
        "reduction_pending_wirbel415": not floor_measured,
        "source": (f"wirbel #415 decomposing the {FIXED_OVERHEAD_FLOOR_US_408:g}us/"
                   f"{FIXED_OVERHEAD_FLOOR_FRAC_408*100:g}% fixed floor (#408 qc9bz8sv); "
                   "0 TPS credit until a measured reduction lands"),
        "gate": "equivalence-neutral (host/launch overhead, not tokens). 0 until wirbel #415.",
    }

    # --- Node 4: lm_head TRUNCATION — FREE under the self-referential gate (land #414) --
    # The operative scorer is SELF-REFERENTIAL (the submission's OWN 16384-row truncated-head
    # greedy); the deployed config passes it FOR FREE -> lm_head truncation costs 0 TPS and is
    # kept OFF the operative ladder.  ABSOLUTE full-vocab equivalence (a 261,976-row head) is a
    # STRONGER, NOT-required notion costing 54.07 TPS — a CONTINGENCY line only.
    node4_tps = node3_tps + LAND414_LMHEAD_TRUNCATION_TPS_COST    # + 0.0 (free)
    node4 = {
        "node": "lmhead_truncation_free_self_referential_414",
        "tps": node4_tps,
        "lmhead_tps_cost": LAND414_LMHEAD_TRUNCATION_TPS_COST,    # 0.0 (free)
        "status": node3_status,                                   # inherits the stack's status (free add)
        "identity": node3["identity"],
        "equivalence_neutral": True,
        "self_referential_gate": LAND414_SELF_REFERENTIAL_GATE,
        "deployed_passes_self_referential": LAND414_DEPLOYED_PASSES_SELF_REFERENTIAL,
        "truevocab_lmhead_tps_cost_contingency": LAND414_TRUEVOCAB_LMHEAD_TPS_COST_CONTINGENCY,  # 54.07
        "truevocab_head_rows": LAND414_TRUEVOCAB_HEAD_ROWS,      # 261,976
        "absolute_fullvocab_required": False,
        "source": "land #414 (bq7xkfcv) — self-referential scorer; lm_head truncation FREE (0 cost)",
        "gate": ("lm_head truncation is FREE: the deployed config passes the submission's OWN truncated-head "
                 "greedy. Absolute full-vocab (261,976 rows) costs 54.07 TPS — a contingency, NOT required."),
    }

    ladder = [node0, node1, node2, node3, node4]

    # --- Headline frontier numbers ------------------------------------------
    # MEASURED frontier = the highest node whose TPS is MEASURED (stand-behind-today).
    measured_nodes = [n for n in ladder if n["status"] == "measured"]
    max_equivalent_tps_measured = max((n["tps"] for n in measured_nodes), default=floor_tps)

    # MODELED frontier (22:26Z): cb3 is ADDITIVE-confirmed (lawine #417) -> the cb3 stack is
    # BANKED as the modeled headline even before kanna #416 prices the exact gap.  In the
    # fully-modeled default the headline is the lawine #417 banked BRACKET [492.08, 494.08]
    # (which BEATS 481.53); once any feeder is MEASURED the bracket collapses to the realized
    # top-of-ladder point.  lm_head adds 0 (free); floor-reduction 0 until wirbel #415.
    modeled_top_point = node4_tps                                # top of ladder (lm_head free)
    if any_feeder_measured:
        modeled_bracket = (modeled_top_point, modeled_top_point)
        max_equivalent_tps_modeled = modeled_top_point
        modeled_is_bracket = False
    else:
        modeled_bracket = FASTEST_EQUIVALENT_BRACKET_TPS         # [492.08, 494.08] (lawine #417 banked)
        max_equivalent_tps_modeled = FASTEST_EQUIVALENT_BRACKET_TPS[0]   # 492.08 conservative lower bound
        modeled_is_bracket = True
    naive_stacked_ceiling_tps = naive_stack_tps                  # recompute + cb3 additive point (494.53 default)

    pending_feeders = [f for f, pend in [
        ("stark#412_selective_recompute_measured_tps", not recompute_measured),
        ("kanna#416_cb3_additivity_gap_tps_tightening", not gap_measured),
        ("wirbel#415_fixed_overhead_floor_reduction_tps", not floor_measured),
    ] if pend]
    banked_feeders = [
        "lawine#417_deployable_green+cb3_additive+bracket[492.08,494.08]",
        "land#414_lmhead_free_self_referential+54.07_truevocab_contingency",
        "denken#413_recompute_point_478.93",
        "denken#409_tree_leg_closed",
        "kanna#403_cb3_+15.60_k229_ppl_safe",
    ]

    return {
        "objective": "max_equivalent_tps__strict_byte_exact_greedy_token_identity_1p0",
        "source": ("advisor 22:08Z re-point (#357) of the #407 human directive (21:13Z): forget 500, "
                   "maximize TPS subject to strict byte-exact greedy-token-equivalence; 22:26Z banked "
                   "lawine #417 (cb3 additive + deployable + bracket) and land #414 (lm_head free)"),
        "deployed_fastpath_off_ladder": {
            "tps": DEPLOYED_FASTPATH_TPS,                        # 481.53
            "served_identity": DEPLOYED_FASTPATH_SERVED_IDENTITY,  # 0.9966 -> NOT equivalent
            "is_equivalent": DEPLOYED_FASTPATH_IS_EQUIVALENT,    # False
            "flips": list(DEPLOYED_FASTPATH_FLIPS),              # (3, 882) reduction-order flips under M=8
            "note": ("the deployed 481.53 is the NON-equivalent fast path (3/882 reduction-order flips "
                     "under M=8 batched verify, stark #381/#405); NOT a legal frontier point."),
        },
        "ladder": ladder,
        # headline frontier
        "max_equivalent_tps_measured": max_equivalent_tps_measured,    # 467.48 today (node 0, identity 1.0)
        "max_equivalent_tps_modeled": max_equivalent_tps_modeled,      # 492.08 (bracket lower; BEATS 481.53)
        "max_equivalent_tps_modeled_bracket": list(modeled_bracket),   # [492.08, 494.08] (lawine #417 banked)
        "max_equivalent_tps_modeled_point": modeled_top_point,         # 494.53 (denken #413 additive point)
        "modeled_is_bracket": modeled_is_bracket,
        "modeled_beats_deployed_nonequiv": max_equivalent_tps_modeled > DEPLOYED_FASTPATH_TPS,  # 492.08 > 481.53
        "naive_stacked_ceiling_tps": naive_stacked_ceiling_tps,        # 494.53 (recompute+cb3 additive point)
        "cb3_stack_banked_additive_417": True,                         # additive CONFIRMED (lawine #417)
        "frontier_identity": 1.0,                                      # every ladder node is served identity 1.0
        "equivalence_tax_vs_deployed_nonequiv_tps": DEPLOYED_FASTPATH_TPS - max_equivalent_tps_measured,
        # land #414 lm_head ledger (FREE under the self-referential gate)
        "lmhead_truncation_free_self_referential": True,
        "lmhead_tps_cost": LAND414_LMHEAD_TRUNCATION_TPS_COST,         # 0.0
        "truevocab_lmhead_tps_cost_contingency": LAND414_TRUEVOCAB_LMHEAD_TPS_COST_CONTINGENCY,  # 54.07 (not required)
        # feeders / gates
        "selective_recompute_measured": recompute_measured,
        "cb3_additivity_gap_resolved": gap_measured,
        "floor_reduction_resolved": floor_measured,
        "deployability_surface": deployability_surface,                # "green" banked (lawine #417)
        "deployability_banked_green": deployability_surface == "green",
        "pending_feeders": pending_feeders,
        "banked_feeders": banked_feeders,
        "all_feeders_resolved": not pending_feeders,
        # tree leg CLOSED (denken #409): ~0 reliable equivalence-neutral supply
        "tree_leg_closed": True,
        "tree_leg_note": (f"tree net-supply CLOSED (denken #409, 3zr7i8ad, 42/42): DP-optimal widths "
                          f"{DENKEN409_DP_WIDTHS} at M={DENKEN409_M} net only +{DENKEN409_TREE_NET_TPS:g} "
                          "TPS and are beta-fragile -> 0 reliable supply; supersedes ubel #401 / denken #402."),
    }


# --------------------------------------------------------------------------- #
# Deliverable 5: caveats.
# --------------------------------------------------------------------------- #
def deliverable5_caveats(b_star: float) -> dict[str, Any]:
    return {
        "caveats": [
            "GO IS A SUPPLY x DEMAND AND-PRODUCT, RE-PRICED 20:19Z AND HELD ON THE TERMINAL GO-FLIP GATE. "
            "private_500_GO <=> (SUPPLY base enables 500) AND (DEMAND closer delivers the residual). It is "
            "NO LONGER demand-alone and NO LONGER (demand x identity): wirbel #393 (LANDED 20:19Z) CORRECTED "
            "the deployable-strict base to 467.48 (< 500; rebuilds=1; the DECODE-specific attn strict tax "
            "3.01% is +0.86pp over #378's eval-weighted 2.15% -> gap_to_500 32.52, ceiling 505.29), so the "
            "SUPPLY base is the binding GO axis. The +32.65 cb3 HEADLINE lift is PPL-DEAD (kanna #394 RED: "
            "held-out 2.4223 + OOD 2.4270 breach 2.42), so 467.48+32.65 is ARITHMETIC-ONLY — NEITHER leaf "
            "clears on paper. The route is NOT closed: it is RE-PRICED toward 'conservative-k cb3 + a NET-"
            "positive tree'. The terminal GO now hinges on kanna #403 (PPL-safe conservative-k supply "
            "re-cost) AND >=1 of ubel #401 / denken #402 (tree net-supply). Identity FOLDS INTO the supply "
            "leaf as a compliance prerequisite (cost, not a top-level factor). The sub-int4 eta-axis "
            "(Route A) stays DEFLATED/EXCLUDED (kanna #374 closed the fusion lever too). All cross-PR "
            "numbers are consumed via the PR thread, NOT by reading other branches.",
            "SUPPLY LEAF IS THE BINDING GO AXIS; THE +32.65 HEADLINE LIFT IS PPL-DEAD, THE PPL-SAFE RE-COST "
            "IS PENDING. The #375 attn rebuild buys only ~11 TPS (eta_attn=0.0215, NOT #326's whole-step "
            "0.3141). wirbel #384 (RED) REFUTED #378's 'bf16 lm_head-BI ~150-TPS' by-elimination: the "
            "deployed lm_head is already byte-exact int4-Marlin at decode (eta_lmhead=0, FREE; f_lmhead="
            "2.24%). wirbel #390 (LANDED) GPU-measured the body int4-Marlin as ALSO byte-exact at the M=8 "
            "decode-verify width (all 8 GEMMs, 3 seeds x 8 trials, g128) -> the per-GEMM --stark381-decode-"
            "identity is GREEN and the ledger collapses to 1 kernel rebuild (attn #375 only; lm_head=0, "
            "body=0). wirbel #393 (LANDED 20:19Z) then CORRECTED the shippable strict base to 467.48 on the "
            "DECODE-specific 3.01% attn tax (510.01 REINSTATED as the ceiling premise; only 518.92 stays "
            "refuted). STRUCTURAL: private <= public, so a <500 public base makes demand-alone INSUFFICIENT. "
            "denken #383 (RED) + denken #387 (MEASURED anchor 0.89027, gap +0.000) CONFIRMED this. lawine "
            "#388/#392 SIZED the cb3 body-allocation lift +32.65 honest HEADLINE, but kanna #394 (RED) made "
            "that HEADLINE PPL-DEAD (winner's-curse selection to k=243-246); k=232 still clears ~2.39 "
            "held-out, so cb3 is deployable at a CONSERVATIVE k at a smaller UN-COSTED lift -> the real "
            "PPL-safe supply number is PENDING kanna #403. I do NOT re-run any GPU eval; I CONSUME "
            "#383/#384/#387/#388/#390/#393/#394 via the thread.",
            "SUPPLY LEVER IS ALIVE BUT ITS HEADLINE LIFT IS PPL-DEAD; THE 20:19Z RE-PRICE SUPERSEDES THE "
            "19:25Z DEPLOYABILITY PAIR. The de-risker is NOT uniform sub-int4 (that died on PPL) but a "
            "SENSITIVITY-WEIGHTED mixed-precision ALLOCATION (cb3): 88.8% of body params at 3-bit for a "
            "3.2369 avg bpw, -21.5% body read. lawine #391 (LANDED) confirmed the M=8 lift is preserved and "
            "denken #392 (LANDED) is the authoritative composed number, BUT kanna #394 (RED) made the +32.65 "
            "HEADLINE PPL-DEAD: #372's +0.039 PPL margin is IN-SAMPLE (winner's-curse), and the held-out "
            "worst-seed 2.4223 + OOD ShareGPT 2.4270 breach the 2.42 gate. The 20:19Z cluster also closed "
            "both deployable escape routes: denken #396 (RED) -> demand-ALONE busts EVEN BARE on 467.48 "
            "(required_dcov ~0.0338 = 109% of the +0.031 budget); ubel #399 (RED) -> NO cheap deployable "
            "demand lever (every monotone draft-head lever is a RANK-INVARIANT no-op, MC max|Δcov|=0). The "
            "route is NOT closed — it is RE-PRICED. Hold the TERMINAL GO None until kanna #403 (PPL-safe "
            "conservative-k SUPPLY re-cost) AND >=1 of ubel #401 (top-8/16 tree coverage ceiling) / denken "
            "#402 (tree NETs d-cov after the verify-M step-time tax) BOTH land green. The tree (+0.1286 "
            "locked top-1->top-4 prize) is a GENUINELY-NEW lever. Sharpen, don't flip.",
            "DEMAND LEAF IS NECESSARY-BUT-INSUFFICIENT (denken #383/#387): it DELIVERS the RESIDUAL at "
            "central confidence after a supply lift, but does NOT reach private-500 ALONE on the corrected "
            "base. The deployed drafter is MTP K=7 (PR #52 method=mtp, num_speculative_tokens=7), NOT "
            "EAGLE-3 -> the leaf is the #289 MTP K=7 conditional-accept ladder. denken #387 MEASURED the "
            "baseline top-4 coverage anchor at 0.89027 (gap +0.000 to the modeled 0.8903 -> drop the hedge; "
            "denken #383 RED is robust to the measured anchor, required Δcov +0.0572 still 1.84x budget). "
            "denken #377 SIZED the closer; denken #380 split deliverability TWO-TIER: central c>=0.8959 "
            "(+0.00565) p_deliver=0.958>=0.90 GREEN-deliverable-now, robust c>=0.9010 (+0.0107) "
            "p_deliver=0.811<0.90 pending a ~25-A10G-GPU-hr pilot (now OFF the critical path per #383). "
            "kappa_breakeven 0.1222, kappa_margin 0.549 (kappa transfer axis ROBUST). I treat the "
            "conservative target ~0.911 as ALREADY-SIZED to close the residual. The demand leaf dies ONLY "
            "if the gap is forced fully irreducible.",
            "SLOPE BANKED GREEN (ubel #382). ubel #379's 489.8 TPS/unit coverage slope IS private-robust "
            "-> 437.3 TPS/unit (flattening ratio 0.893); the conservative bank target ~0.911 is 66.6% of "
            "#336's +0.031 budget (vs 38.9% central). The a1 first-token collapse deepens 0.729 -> 0.598 "
            "under private OOD (#263 rank-2+), but the route SURVIVES (no longer pending #382). "
            "coverage_target_for_3p2=0.9011 reconciles denken #377's +0.0111 to within 0.0003; "
            "gap_after_max_coverage_retrain 1.142% (full #336 envelope -> ~3x headroom).",
            "GAP CEILING-CHECK BANKED (ubel #379 GREEN); FLOOR INFLATES UNDER VBI=1 (ubel #386 RED, BANKED "
            "17:53Z). The 4.295pp public->private gap = 85.25% acceptance (coverage-ADDRESSABLE) + 14.75% "
            "ctxlen (IRREDUCIBLE) + 0% outlen + 0% numerics. The fixed numerics/identity tax CANCELS in the "
            "public->private STEP difference (floors absolute TPS, not the gap), REFUTING 'numerics is the "
            "irreducible floor'. The off-VBI irreducible floor is 0.633% central, BUT ubel #386 found it "
            "does NOT survive the live VBI=1 un-packed-attention contract — it INFLATES 2.07x -> 1.310% "
            "central. The route is NOT dead (central 1.310% still clears the 3.2% knife-edge by +1.89pp), "
            "but all_corners_clear_3p2_vbi1=False (the pessimistic corner breaches at 3.5235%, -0.32pp) and "
            "the breakeven private prompt shift HALVES (+253 -> +119 tok), so prompt-length-shift "
            "sensitivity is now a BINDING risk. Re-derive the demand ceiling on the 1.310% LIVE floor, NOT "
            "0.633%; ubel #389 (GPU per-L attention measurement) is reseated to PIN the thin -0.32pp breach. "
            "I PULL ubel #379's split + ubel #386's inflation; I do NOT re-derive the 0.3914pp/cov elasticity.",
            "DELIVERABILITY central GREEN, robust modeled-only (denken #380). The central tier banks now "
            "(p_deliver=0.958 on #339's modeled N(0.0385,0.0074) + kappa=0.672 transfer; delivered-after-"
            "kappa ~0.0259 exceeds the central 0.00565). The robust tier (p_deliver=0.811) needs the "
            "~25-A10G-GPU-hr pilot before it banks hard. The triple-tail corner (ceiling 509.07 AND rho "
            "0.8038 AND worst c*=0.9256 = 136% of #336's budget) is a low-probability simultaneous "
            "worst-case that ignores the gap co-benefit; a SENSITIVITY BAND, not the operating point.",
            "IDENTITY IS A COMPLIANCE PREREQUISITE inside the SUPPLY leaf (strict greedy-token-identity "
            "HARD gate), NOT a speedup lever and NOT a top-level GO factor. stark #363 (MERGED) measured "
            "the attention locus as FREE (eta~0, ratio 0.9167, best K=8). stark #376 (RED) then found on "
            "REAL weights that pinning attention (VLLM_BATCH_INVARIANT->num_splits=1) leaves e2e identity "
            "0.992555 (~heuristic 0.992708); the residual ~0.73% flip is the int4-Marlin body GEMM, a "
            "custom CUDA op OUTSIDE the aten dispatcher that the env knob structurally CANNOT patch. BUT "
            "the RED is GEOMETRY-SPECIFIC: Marlin is BIT-EXACT at the 8-row decode-verify width and only "
            "M-variant at the 2048-row prefill-replication width (the only geometry tractable via vLLM's "
            "prompt_logprobs API). wirbel #390 (LANDED) RESOLVED the served 8-row geometry at the PER-GEMM "
            "level: the body int4-Marlin is byte-exact at M=8 across all 8 GEMMs (3 seeds x 8 trials, g128, "
            "incl the atomic-add-eligible small-n k/v_proj) -> the --stark381-decode-identity per-GEMM "
            "condition is GREEN -> env-reachable@decode (1 rebuild: #375 mha_varlen; lm_head=0, body=0). "
            "stark #381 stays open only as the INDEPENDENT e2e confirmation incl norms; the per-GEMM Q is "
            "answered. Identity is REACHABLE in both branches; the cost is now 1 kernel rebuild, folded "
            "into the supply leaf. The old eta-budget gate (ETA_BUDGET_500 ~= 4.02%) is retained in the "
            "deflated Route A only.",
            "ROUTE A (sub-int4 eta-axis, EXCLUDED from GO) is distinct from the supply BASE leaf. It still "
            "carries two measured inputs consumed via the PR thread for continuity: denken #356's "
            "Gemma-4-E4B ppl_at_best_sub_int4_bits at b*, AND stark #365's lmhead_bi_gemm_eta. We do NOT "
            "re-run either GPU eval. Route A is deflated/un-deployable as a standalone path (PPL-blocked + "
            "eta deflated), so it neither rescues nor blocks the GO.",
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
               d8: dict, d_supply: dict, d_ident_reach: dict, d_frontier: dict,
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

    # aa: 20:19Z RE-PRICE — the central-GREEN coverage-retrain sizing (denken #380 + ubel #382) is now
    #     PROVENANCE ONLY (a drafter retrain is FORBIDDEN) and the robust coverage-lift pilot is OFF the
    #     critical path (denken #383).  So deliver_central_green stays True (it SIZED the target) but the
    #     demand leaf is TREE-gated: demand_leaf_delivers is None pending ubel #401 / denken #402, and the
    #     robust-pilot input only resolves the provenance robust tier -- it does NOT flip leaf delivery.
    d6_pilot_ok = demand_side_route_analysis("delivers")
    d6_pilot_fail = demand_side_route_analysis("fails")
    aa_demand_central_green_robust_tiered = (
        d6["deliver_central_green"] is True              # central coverage sizing GREEN (provenance only)
        and d6["deliver_robust_modeled"] is False        # robust tier needs the (forbidden) pilot
        and d6["deliver_robust_resolved"] is None        # pending default -> unresolved
        and d6["leaf_robust_pending"] is True
        and d6["robust_pilot_off_critical_path_383"] is True   # pilot is provenance now (denken #383)
        and d6["demand_leaf_delivers"] is None           # TREE-gated -> NOT delivered on central-green alone
        and d6["demand_leaf_pending_tree"] is True
        # the robust pilot resolves the PROVENANCE robust tier but does NOT flip the tree-gated delivery
        and d6_pilot_ok["deliver_robust_resolved"] is True
        and d6_pilot_ok["demand_leaf_delivers"] is None
        and d6_pilot_fail["deliver_robust_resolved"] is False
        and d6_pilot_fail["demand_leaf_delivers"] is None)

    # ab: the demand leaf DIES only if the gap is forced irreducible (override the banked split to 0 ->
    #     no coverage channel). The banked ubel #379 GREEN split keeps the channel live by default.
    d6_gap_killed = demand_side_route_analysis("pending", 0.0)
    ab_demand_leaf_dead_if_gap_forced_irreducible = (
        d6_gap_killed["demand_leaf_delivers"] is False
        and d6_gap_killed["gap_channel_live"] is False
        and "gap_irreducible" in d6_gap_killed["binding_constraint"])

    # ac: the 489.8 slope is BANKED private-robust (ubel #382 GREEN -> 437.3 private, flattening 0.893);
    #     the CONSERVATIVE private-anchored target ~0.911 consumes more of #336's budget than central.
    ac_slope_banked_private_robust_382 = (
        d6["slope_is_private_robust"] is True
        and abs(d6["slope_tps_per_coverage_private_382"]
                - SLOPE_TPS_PER_COVERAGE_UBEL379 * SLOPE_FLATTENING_RATIO_382) < TOL_EXACT
        and abs(d6["demand_conservative_target_382"] - DEMAND_CONSERVATIVE_TARGET_382) < TOL_EXACT
        and d6["demand_budget_frac_conservative_382"] > d6["demand_budget_frac_central_382"])

    # ad: deliverability is TWO-TIER (denken #380): central p_deliver >= 0.90 (banked GREEN), robust
    #     p_deliver < 0.90 (pending pilot); the kappa-axis is ROBUST (breakeven 0.1222 << worst c* 0.354).
    ad_deliverability_two_tier_380 = (
        d6["p_deliver_central_defensible"] >= d6["p_deliver_threshold"]
        and d6["p_deliver_robust_defensible"] < d6["p_deliver_threshold"]
        and d6["demand_closer_central_c"] < d6["demand_closer_robust_c"]
        and d6["kappa_axis_robust"] is True
        and KAPPA_BREAKEVEN_380 < 0.354)

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

    # --- supply-side base leaf (wirbel #393 corrected base 467.48, the NOW-BINDING GO axis) --- #
    # ao: the corrected deployable-strict base is 467.48-today (wirbel #393; < 500; gap_to_500 32.52);
    #     demand-alone may be insufficient (private <= public) and denken #383 (RED) CONFIRMED it on the
    #     corrected base; the 518.92 eta-axis pin is deflated; this is the binding GO leaf.
    ao_supply_leaf_honest_base_below_500 = (
        abs(d_supply["supply_base_today_tps"] - REALIZED_DEPLOYED_STRICT_393) < TOL_EXACT   # 467.48 (wirbel #393)
        and d_supply["supply_base_today_tps"] < TARGET
        and abs(d_supply["deficit_to_500_today_tps"] - GAP_TO_500_393) < TOL_EXACT          # 32.52
        and abs(d_supply["honest_strict_base_floor"] - HONEST_STRICT_BASE_FLOOR_378) < TOL_EXACT
        and d_supply["supply_base_clears_500_today"] is False
        and d_supply["demand_alone_may_be_insufficient"] is True
        and d_supply["demand_alone_insufficient_confirmed_383"] is True   # banked denken #383 RED
        and abs(d_supply["eta_axis_base_deflated_518"] - ETA_AXIS_BASE_DEFLATED_518) < TOL_EXACT
        and d_supply["is_binding_go_leaf"] is True)

    # ap: the supply leaf RESOLVES on denken #383 (reach-on-floor) + kanna #403 (PPL-safe conservative-k
    #     supply re-cost; the +32.65 HEADLINE is PPL-DEAD per kanna #394 RED, so the "no" branch now defers
    #     to kanna #403, NOT the dead headline): denken383 pending -> None; "yes" -> True; "no" + kanna #403
    #     pending -> None (HELD; headline PPL-dead); "no" + kanna #403 red -> False; "no" + kanna #403 green
    #     + insufficient re-costed lift -> False; "no" + kanna #403 green + sufficient -> True.
    s_pending = supply_side_base_analysis()
    s_yes = supply_side_base_analysis(demand_reaches_500_on_floor="yes")
    s_no_k403_pending = supply_side_base_analysis(supply_lift_available_tps=25.0,
                                                  demand_reaches_500_on_floor="no",
                                                  supply_lift_required_tps=15.0)            # kanna #403 default pending
    s_no_k403_red = supply_side_base_analysis(demand_reaches_500_on_floor="no",
                                              kanna403_ppl_safe_supply="red")
    s_no_k403_green_insuff = supply_side_base_analysis(supply_lift_available_tps=10.0,
                                                       demand_reaches_500_on_floor="no",
                                                       supply_lift_required_tps=40.0,
                                                       kanna403_ppl_safe_supply="green")
    s_no_k403_green_suff = supply_side_base_analysis(supply_lift_available_tps=25.0,
                                                     demand_reaches_500_on_floor="no",
                                                     supply_lift_required_tps=15.0,
                                                     kanna403_ppl_safe_supply="green")
    ap_supply_leaf_resolves_on_inputs = (
        s_pending["supply_base_enables_500"] is None
        and s_pending["supply_pending"] is True
        and s_pending["denken383_pending"] is True
        and s_yes["supply_base_enables_500"] is True
        and s_no_k403_pending["supply_base_enables_500"] is None          # HELD: headline PPL-dead, awaits kanna #403
        and s_no_k403_pending["kanna403_ppl_safe_supply_pending"] is True
        and s_no_k403_red["supply_base_enables_500"] is False             # no PPL-safe conservative-k lift clears
        and s_no_k403_green_insuff["supply_base_enables_500"] is False    # re-costed lift insufficient
        and s_no_k403_green_suff["supply_base_enables_500"] is True)      # re-costed lift sufficient

    # --- composite supply x demand AND-product (the 20:19Z RE-PRICED GO) --- #
    # aq: GO = (supply leg kanna #403 PPL-safe) AND (demand leg ubel #401 OR denken #402 tree net-supply),
    #     None-aware.  Fully pending -> None; supply enables ("yes" floor) + demand delivers (tree green)
    #     -> True; supply cannot enable (kanna #403 red) -> False; demand leaf dead (gap channel forced
    #     dead) -> False.  The lift-RESOLVED supply path ("no" + kanna #403 green + sufficient re-costed
    #     lift) + tree green also flips True.  Route A (sub-int4 eta-axis) is EXCLUDED (in_go_product False).
    c_pending = composite_verdict(None, b_star, None)
    c_go = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="yes",
                             ubel401_tree_coverage_ceiling="green")
    c_nogo = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                               kanna403_ppl_safe_supply="red", ubel401_tree_coverage_ceiling="green")
    c_go_lift = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                                  supply_lift_required_tps=15.0, supply_lift_available_tps=25.0,
                                  kanna403_ppl_safe_supply="green", denken402_tree_net_supply="green")
    c_demand_dead = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="yes",
                                      gap_addressable_pp=0.0)
    aq_composite_supply_x_demand_product = (
        c_pending["private_500_reachable_via_known_levers"] is None
        and c_pending["verdict_pending"] is True
        and c_go["private_500_reachable_via_known_levers"] is True
        and c_go["supply_base_enables_500"] is True
        and c_go["demand_closer_delivers"] is True
        and c_nogo["private_500_reachable_via_known_levers"] is False
        and c_nogo["supply_base_enables_500"] is False
        and c_go_lift["private_500_reachable_via_known_levers"] is True
        and c_demand_dead["private_500_reachable_via_known_levers"] is False
        and c_demand_dead["demand_closer_delivers"] is False
        and c_go["route_a_supply_side"]["in_go_product"] is False)

    # ar: composite framing — primary_route is the supply x demand product; the GO formula is an AND of
    #     (supply_base_enables_500) and (demand closer delivers); eta-axis deflated; Route A excluded;
    #     the structural "demand-alone may be insufficient" (private <= public) flag is set.
    ar_composite_framing_supply_x_demand = (
        d7["primary_route"] == "supply_x_demand_and_product"
        and "AND" in d7["go_formula"]
        and "supply_base_enables_500" in d7["go_formula"]
        and d7["eta_axis_deflated"] is True
        and d7["route_a_supply_side"]["in_go_product"] is False
        and d7["demand_alone_may_be_insufficient"] is True
        and d6["is_go_leaf"] is True
        and d6["eta_axis_deflated"] is True)

    # as: ubel #386 (RESOLVED RED 17:53Z) — the off-VBI 0.633% floor does NOT survive the live VBI=1
    #     contract: it INFLATES 2.07x -> 1.310% central, so the "uncapped on the live stack" claim FAILS.
    #     The function resolves on the input (pending -> None; survives -> True; inflates -> False); the
    #     BANKED default ("inflates") is wired through d8 -> floor RESOLVED (not pending), uncapped False.
    d8_vbi_pending = gap_decomposition_analysis("pending")
    d8_vbi_survives = gap_decomposition_analysis("survives")
    d8_vbi_inflates = gap_decomposition_analysis("inflates")
    as_ubel386_floor_inflates_resolved = (
        d8_vbi_pending["uncapped_on_live_vbi_stack"] is None
        and d8_vbi_pending["irreducible_floor_vbi_pending_ubel386"] is True
        and d8_vbi_survives["uncapped_on_live_vbi_stack"] is True
        and d8_vbi_inflates["uncapped_on_live_vbi_stack"] is False
        # banked default ("inflates") is wired through the live d8:
        and d8["irreducible_floor_vbi_pending_ubel386"] is False
        and d8["irreducible_floor_inflates_vbi_386"] is True
        and d8["uncapped_on_live_vbi_stack"] is False
        and abs(d8["irreducible_floor_vbi1_central_pct_386"] - IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386) < TOL_EXACT
        and abs(d8["floor_inflation_mult_386"] - FLOOR_INFLATION_MULT_386) < TOL_EXACT)

    # at: ubel #389 (LANDED 19:15Z) REFUTES the ubel #386 pessimistic-corner breach. #386 INTERPOLATED an
    #     inflated 1.310% live floor whose pessimistic corner breaches the 3.2% knife-edge by -0.32pp
    #     (the binding-risk read). ubel #389 then MEASURED the per-L attention identity floor under VBI=1
    #     at 0.5764% (< the 0.633% central, far below #386's 1.310% interpolation): 0 corners breach 3.2%,
    #     and the measured local-penalty slope is only 0.353x the #386 interpolation -> the breach was a
    #     conservative-slope ARTIFACT. So the demand leaf is ROBUST on the measured slope (advisor 19:25Z:
    #     "drop the #386 RED"); the central tier still clears (+1.89pp) and delivers. The #386 figures are
    #     kept as the superseded interpolation; ubel #389 is the operative measurement (pin not pending).
    at_ubel389_refutes_386_breach_demand_robust = (
        # #386 INTERPOLATED state (the superseded record): central clears but the pessimistic corner breaches
        d8["central_clears_3p2_vbi1_386"] is True
        and abs(d8["central_margin_to_3p2_vbi1_pp_386"] - CENTRAL_MARGIN_TO_3P2_VBI1_PP_386) < TOL_EXACT
        and d8["all_corners_clear_3p2_vbi1_386"] is False
        and d8["pessimistic_corner_margin_pp_386"] < 0.0
        and abs(d8["breakeven_prompt_shift_vbi1_tok_386"] - BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386) < TOL_EXACT
        and d8["breakeven_prompt_shift_vbi1_tok_386"] < BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK
        # ubel #389 MEASURED state REFUTES the #386 breach (LANDED; pin no longer pending)
        and d8["ubel389_landed"] is True
        and d8["ubel389_pin_breach_pending"] is False
        and d8["ubel389_386_breach_refuted"] is True
        and abs(d8["ubel389_measured_floor_vbi1_pct"] - UBEL389_MEASURED_FLOOR_VBI1_PCT) < TOL_EXACT
        and d8["ubel389_measured_floor_vbi1_pct"] < GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379  # 0.5764 < 0.633
        and d8["ubel389_pessimistic_breaches_3p2_measured"] == 0
        and d8["ubel389_all_corners_clear_3p2_measured"] is True
        and abs(d8["ubel389_measured_slope_ratio_to_386"] - UBEL389_MEASURED_SLOPE_RATIO_TO_386) < TOL_EXACT
        and d8["ubel389_measured_slope_ratio_to_386"] < 1.0  # measured slope < #386 interp -> artifact
        # the demand leaf carries the record through; ubel #389 makes the irreducible-floor risk OFF
        # (robust on the measured slope) so the floor is NOT the binding constraint -- but under the
        # 20:19Z re-price the leaf is TREE-gated (denken #396 / ubel #399 RED), so it delivers None
        # PENDING the tree (ubel #401 / denken #402), NOT dead on the floor.
        and d6["ubel389_386_breach_refuted"] is True
        and d6["ubel389_all_corners_clear_3p2_measured"] is True
        and d6["central_clears_3p2_vbi1_386"] is True
        and d6["prompt_shift_sensitivity_binding_risk_operative"] is False  # floor-risk OFF (measured slope)
        and d6["demand_leaf_delivers"] is None                              # TREE-gated, NOT floor-dead
        and d6["demand_leaf_pending_tree"] is True)

    # au: denken #383 (RED 17:53Z) — demand-alone does NOT reach private-500 on the honest base; a supply
    #     lift of +17.2 TPS (floor-joint, < the +23.8 E[T]-only variant) is required FIRST; the #375 attn
    #     rebuild ALONE does not close the supply gap (remainder is the int4-Marlin body #376, per #384);
    #     #377 reproduces under the 518.92 revival; the ~25-GPU-hr pilot is OFF the critical path; OOB Δcov.
    au_denken383_supply_lift_required_first = (
        d_supply["demand_alone_insufficient_confirmed_383"] is True
        and d_supply["denken383_reaches_500_on_floor"] == "no"
        and abs(d_supply["supply_lift_required_first_tps_383"] - SUPPLY_LIFT_REQUIRED_FIRST_TPS_383) < TOL_EXACT
        and d_supply["supply_lift_required_et_only_tps_383"] > d_supply["supply_lift_required_first_tps_383"]
        and d_supply["attn_rebuild_alone_closes_supply_gap_383"] is False
        and d_supply["reproduces_377_under_revival_383"] is True
        and d_supply["pilot_on_critical_path_383"] is False
        and d6["robust_pilot_off_critical_path_383"] is True
        and d_supply["required_dcov_budget_mult_383"] > 1.0)

    # av: lawine #372 (GREEN 17:53Z) — a SUPPLY lever is ALIVE via sensitivity-weighted mixed-precision
    #     ALLOCATION (88.8% of body at 3-bit, 3.2369 avg bpw) that PASSES the PPL gate (2.3812 <= 2.42)
    #     for -21.5% body read; the UNIFORM-3bit variant died on PPL. lawine #388 has since LANDED the
    #     realized TPS of this allocation (no longer pending; banked facts asserted in ax).
    av_lawine372_supply_lever_alive = (
        d_supply["lawine372_supply_lever_alive"] is True
        and abs(d_supply["mixed_precision_avg_bpw_372"] - MIXED_PRECISION_AVG_BPW_372) < TOL_EXACT
        and abs(d_supply["body_3bit_frac_372"] - BODY_3BIT_FRAC_372) < TOL_EXACT
        and d_supply["mixed_precision_gate_ppl_372"] <= PPL_GATE
        and abs(d_supply["body_read_reduction_372"] - BODY_READ_REDUCTION_372) < TOL_EXACT
        and d_supply["uniform_3bit_died_on_ppl_372"] is True
        and d_supply["lawine388_realized_tps_pending"] is False    # LANDED (g5lfdpgw)
        and d_supply["lawine388_landed"] is True)

    # aw: wirbel #384 (RED, BANKED 18:12Z) — CORRECTION: the deployed lm_head is ALREADY byte-exact
    #     strict at decode (untied int4-Marlin, eta_lmhead=0, FREE), so the ~150-TPS "bf16 lm_head-BI"
    #     determinization tax was a by-elimination artifact (incremental share ~0). The dominant
    #     non-attention strict tax lives in the int4-Marlin BODY (#376). The #384-era rebuild ledger is 2
    #     (attn #375 + body #376), NOT 3 (lm_head shares the body kernel); wirbel #390 then collapses it
    #     to 1 (asserted in ax). The supply-lift DATA has since LANDED (wirbel #390 base + lawine #388
    #     lift), so the ceiling is no longer pending and the GO-flip axis is the cb3-deployability pair.
    aw_wirbel384_lmhead_free_supply_tax_in_body = (
        d_supply["wirbel384_lmhead_free"] is True
        and abs(d_supply["eta_lmhead_targeted_384"] - 0.0) < TOL_EXACT
        and d_supply["n_kernel_rebuilds_strict_500_384"] == 2     # #384-era count (wirbel #390 -> 1, see ax)
        and abs(d_supply["lmhead_bi_incremental_share_384"] - 0.0) < TOL_EXACT
        and d_supply["lmhead_is_int4_marlin_not_bf16_384"] is True
        and d_supply["dominant_nonattn_strict_locus_384"] == DOMINANT_NONATTN_STRICT_LOCUS_384
        # the supply-lift DATA has LANDED (wirbel #390 base + lawine #388 lift); ceiling no longer pending
        and d_supply["wirbel390_shippable_ceiling_pending"] is False
        and d_supply["supply_lift_data_landed"] is True
        # the identity-reachability ledger agrees: lm_head free at decode (#384-era count = 2)
        and d_ident_reach["lmhead_int4_marlin_free_at_decode_wirbel384"] is True
        and d_ident_reach["n_kernel_rebuilds_strict_500_wirbel384"] == 2)

    # ax: wirbel #393 (0q7ynumg, LANDED 20:19Z) — CORRECTED shippable strict BASE = 467.48 (supersedes
    #     wirbel #390's 471.42): the DECODE-specific attention strict tax is 3.01% (decode band [528,658]),
    #     +0.86pp larger than #378's eval-weighted 2.15% -> gap_to_500 widens to 32.52, ceiling 505.29.
    #     The rebuild ledger stays 1 (attention only: lm_head=0 #384 + body int4-Marlin byte-exact @ M=8
    #     #390 -> the per-GEMM --stark381-decode-identity GREEN condition); 510.01 REINSTATED, only 518.92
    #     refuted.  ★ 20:19Z: the +32.65 cb3 HEADLINE lift is PPL-DEAD (kanna #394 RED), so 467.48 + 32.65
    #     = 500.13 is ARITHMETIC-ONLY (supply_alone_clears_500_with_cb3 False); the real PPL-safe supply
    #     number awaits kanna #403, so s393 (a lift but kanna #403 default pending) -> supply HELD None.
    s393 = supply_side_base_analysis(supply_lift_available_tps=CB3_LIFT_HONEST_TPS_388,
                                     demand_reaches_500_on_floor="no",
                                     supply_lift_required_tps=SUPPLY_LIFT_REQUIRED_FIRST_TPS_383,
                                     stark381_decode="green")
    ax_wirbel393_corrected_base_467_headline_ppl_dead = (
        d_supply["wirbel390_landed"] is True
        and d_supply["wirbel393_landed"] is True
        and abs(d_supply["realized_deployed_strict_390"] - REALIZED_DEPLOYED_STRICT_390) < TOL_EXACT  # 471.42 provenance
        and abs(d_supply["realized_deployed_strict_393"] - REALIZED_DEPLOYED_STRICT_393) < TOL_EXACT  # 467.48 (supersedes)
        and abs(d_supply["supply_base_today_tps"] - REALIZED_DEPLOYED_STRICT_393) < TOL_EXACT          # base = 467.48
        and abs(d_supply["gap_to_500_393"] - GAP_TO_500_393) < TOL_EXACT                               # 32.52
        and abs(d_supply["deficit_to_500_today_tps"] - GAP_TO_500_393) < TOL_EXACT
        and abs(d_supply["shippable_ceiling_393"] - SHIPPABLE_CEILING_393) < TOL_EXACT                 # 505.29
        and abs(d_supply["decode_attn_strict_tax_pct_393"] - DECODE_ATTN_STRICT_TAX_PCT_393) < TOL_EXACT  # 3.01%
        and d_supply["supply_alone_closes_500_390"] is False
        and d_supply["n_kernel_rebuilds_strict_500_390"] == 1            # ledger 2 -> 1 (attention only)
        and d_supply["body_marlin_decode_strict_green_390"] is True
        and d_supply["stark381_decode_identity_per_gemm_green_390"] is True
        and d_supply["spread_is_lmhead_bf16_tax_390"] is False
        and abs(d_supply["shippable_ceiling_510_reinstated_390"]
                - SHIPPABLE_CEILING_510_REINSTATED_390) < TOL_EXACT
        and abs(d_supply["shippable_ceiling_518_still_refuted_390"]
                - SHIPPABLE_CEILING_518_STILL_REFUTED_390) < TOL_EXACT
        and SHIPPABLE_CEILING_518_STILL_REFUTED_390 in d_supply["shippable_ceiling_refuted_bf16_premise_390"]
        and SHIPPABLE_CEILING_510_REINSTATED_390 not in d_supply["shippable_ceiling_refuted_bf16_premise_390"]
        # the +32.65 cb3 HEADLINE (denken #392) is PPL-DEAD (kanna #394 RED): 467.48 + 32.65 = 500.13 is
        # ARITHMETIC-ONLY, NOT a deployable supply-alone clear -> the supply leaf does NOT rest on it.
        and d_supply["lawine388_landed"] is True
        and abs(d_supply["cb3_lift_honest"] - CB3_LIFT_HONEST_DENKEN392) < TOL_EXACT                   # +32.65 headline
        and d_supply["cb3_headline_lift_ppl_dead_394"] is True
        and abs(d_supply["base_plus_cb3_honest"]
                - (REALIZED_DEPLOYED_STRICT_393 + CB3_LIFT_HONEST_DENKEN392)) < TOL_EXACT              # 500.13 arithmetic
        and d_supply["supply_alone_clears_500_with_cb3"] is False        # PPL-dead headline -> no supply-alone clear
        and d_supply["cb3_deployable_at_conservative_k_394"] is True     # but deployable at a conservative k (kanna #403)
        and s393["supply_base_enables_500"] is None                      # HELD: real lift awaits kanna #403
        and s393["kanna403_ppl_safe_supply_pending"] is True)

    # ay: the TERMINAL GO is the 20:19Z RE-PRICED AND-product — (supply leg kanna #403 PPL-safe) AND
    #     (demand leg ubel #401 OR denken #402 tree net-supply).  The 19:25Z cb3-deployability pair is
    #     SUPERSEDED RED (kanna #394 made the +32.65 headline PPL-dead; denken #396 closed demand-alone;
    #     ubel #399 closed the cheap monotone levers).  GREEN iff both legs green; RED iff either leg red;
    #     HELD None until both land.  Defaults (both legs pending) -> held None with BOTH gates listed.
    c_t_pending = composite_verdict(None, b_star, None)
    c_t_green = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                                  supply_lift_required_tps=15.0, supply_lift_available_tps=25.0,
                                  kanna403_ppl_safe_supply="green", denken402_tree_net_supply="green")
    c_t_supply_red = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                                       kanna403_ppl_safe_supply="red", ubel401_tree_coverage_ceiling="green")
    c_t_demand_red = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="yes",
                                       ubel401_tree_coverage_ceiling="red", denken402_tree_net_supply="red")
    c_t_hold_tree = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                                      supply_lift_required_tps=15.0, supply_lift_available_tps=25.0,
                                      kanna403_ppl_safe_supply="green")            # supply green, tree pending
    c_t_hold_supply = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                                        ubel401_tree_coverage_ceiling="green")     # tree green, supply (kanna #403) pending
    ay_terminal_go_is_supply_leg_and_tree_leg = (
        # fully pending -> held None, both legs listed, the 19:25Z cb3-deployability pair superseded
        c_t_pending["private_500_reachable_via_known_levers"] is None
        and c_t_pending["terminal_go"] == "pending"
        and c_t_pending["terminal_go_pending"] is True
        and c_t_pending["superseded_1925z_pair"] is True
        and c_t_pending["terminal_go_factor"]["kanna394_headline_ppl_dead"] is True
        and c_t_pending["terminal_go_factor"]["denken396_demand_alone_closed"] is True
        and c_t_pending["terminal_go_factor"]["ubel399_no_cheap_demand_lever"] is True
        and c_t_pending["terminal_go_factor"]["tree_is_genuinely_new_lever"] is True
        and "kanna#403_ppl_safe_conservative_k_supply_recost" in c_t_pending["terminal_go_pending_gates"]
        and "ubel#401_or_denken#402_tree_net_supply" in c_t_pending["terminal_go_pending_gates"]
        # supply leg green (kanna #403 + sufficient re-costed lift) AND demand leg green (denken #402) -> GO True
        and c_t_green["private_500_reachable_via_known_levers"] is True
        and c_t_green["terminal_go"] == "green"
        and c_t_green["terminal_go_confirmed"] is True
        and c_t_green["supply_base_enables_500"] is True
        and c_t_green["demand_closer_delivers"] is True
        # supply leg red (kanna #403 no PPL-safe lift clears) -> GO False even with the tree green
        and c_t_supply_red["private_500_reachable_via_known_levers"] is False
        and c_t_supply_red["terminal_go"] == "red"
        and c_t_supply_red["terminal_go_blocked"] is True
        and c_t_supply_red["supply_base_enables_500"] is False
        # demand leg red (tree both red) -> GO False even with supply green
        and c_t_demand_red["private_500_reachable_via_known_levers"] is False
        and c_t_demand_red["terminal_go"] == "red"
        and c_t_demand_red["tree_net_supply_both_red"] is True
        and c_t_demand_red["demand_closer_delivers"] is False
        # supply resolved (kanna #403 green + lift) but tree pending -> HELD on the tree gate ONLY
        and c_t_hold_tree["private_500_reachable_via_known_levers"] is None
        and c_t_hold_tree["terminal_go"] == "pending"
        and c_t_hold_tree["terminal_go_pending_gates"] == ["ubel#401_or_denken#402_tree_net_supply"]
        # tree resolved (ubel #401 green) but supply (kanna #403) pending -> HELD on the kanna #403 gate ONLY
        and c_t_hold_supply["private_500_reachable_via_known_levers"] is None
        and c_t_hold_supply["terminal_go"] == "pending"
        and c_t_hold_supply["terminal_go_pending_gates"] == ["kanna#403_ppl_safe_conservative_k_supply_recost"])

    # az: denken #387 (z8osvif8, LANDED) — the demand coverage anchor is MEASURED (top-4 = 0.89027,
    #     anchor_gap +0.000 -> denken #383 RED ROBUST to the measured anchor; required Δcov +0.0572 still
    #     1.84x the #336 budget) and the deployed drafter is MTP K=7 (NOT EAGLE-3). kanna #374 (djia6icp,
    #     LANDED) CLOSED the fusion lever (Triton not byte-exact-pinnable; capture/land #371 the sole
    #     identity-safe non-spec leg) -> Route-A stays excluded from the GO product.
    az_denken387_measured_anchor_mtp_k7_kanna374_fusion_closed = (
        d_supply["denken387_landed"] is True
        and abs(d_supply["measured_top4_coverage_387"] - MEASURED_TOP4_COVERAGE_387) < TOL_EXACT
        and abs(d_supply["coverage_anchor_gap_387"] - COVERAGE_ANCHOR_GAP_387) < TOL_EXACT
        and abs(d_supply["required_delta_floor_measured_387"] - REQUIRED_DELTA_FLOOR_MEASURED_387) < TOL_EXACT
        and d_supply["denken383_red_robust_to_measured_anchor_387"] is True
        and d_supply["deployed_drafter_mtp_k_387"] == 7
        and d_supply["drafter_is_mtp_not_eagle3_387"] is True
        and d_supply["required_dcov_budget_mult_383"] > 1.0
        and d_supply["kanna374_fusion_lever_closed"] is True
        and d_supply["fusion_byte_exact_pinnable_374"] is False
        and d_supply["capture_land_371_sole_identity_safe_nonspec_leg_374"] is True
        and d_supply["route_a_stays_excluded_374"] is True
        and d7["route_a_supply_side"]["in_go_product"] is False)

    # --- ubel #379 (GREEN) gap-decomposition ceiling-check (BANKED) --- #
    # ai: the four gap shares (acceptance/ctxlen/outlen/numerics) sum to exactly 1.0
    ai_gap_fracs_sum_to_one = abs(d8["gap_fractions_sum"] - 1.0) < TOL_EXACT

    # aj: irreducible floor (central) == ctxlen-share x gap; addressable == acceptance-share x gap
    aj_gap_floor_matches_ctxlen = (
        abs(d8["gap_irreducible_pp_from_ctxlen_frac"]
            - PUBLIC_PRIVATE_GAP_PCT * GAP_CTXLEN_FRAC_UBEL379) < TOL_EXACT
        and abs(d8["gap_addressable_pp"]
                - PUBLIC_PRIVATE_GAP_PCT * GAP_ACCEPTANCE_FRAC_UBEL379) < TOL_EXACT
        and abs(d8["gap_irreducible_pp_central"] - d8["gap_irreducible_pp_from_ctxlen_frac"]) < 1e-3)

    # ak: every irreducible corner clears the 3.2% knife-edge by >= 1.5pp; closer NOT floor-capped
    ak_gap_corners_clear_knife_edge = (
        d8["all_corners_clear_knife_edge"] is True
        and all(m >= KNIFE_EDGE_MIN_MARGIN_PP for m in d8["knife_edge_corner_margins"].values())
        and d8["closer_not_capped_by_irreducible_floor"] is True
        and d8["gap_channel_live"] is True)

    # al: coverage target reconciles denken #377 to within 0.0003; numerics CANCELS (refutes the
    #     "numerics tax is the irreducible floor" hypothesis)
    al_gap_reconciles_numerics_cancels = (
        d8["reconciles_within_0p0003"] is True
        and abs(d8["delta_cov_ubel379"]
                - (COVERAGE_TARGET_FOR_3P2_UBEL379 - BASELINE_COV_336)) < TOL_EXACT
        and d8["numerics_tax_cancels_in_step_diff"] is True
        and d8["refutes_numerics_is_irreducible_floor"] is True)

    # --- identity REACHABILITY two-branch (stark #376 RED + stark #381 pending) --- #
    ident_green = identity_reachability_analysis("green")
    ident_red = identity_reachability_analysis("red")
    ident_pending = identity_reachability_analysis("pending")
    # am: identity is REACHABLE in BOTH #381 branches; deployment COST differs (1 vs 2 rebuilds);
    #     pending leaves reachability True but the cost branch pending (asserted on an EXPLICIT pending
    #     construction so this is robust to the headline run's --stark381-decode-identity green, which
    #     resolves the LIVE d_ident_reach to the env-reachable@decode 1-rebuild branch).
    am_identity_reachable_both_branches = (
        ident_green["identity_reachable_env_or_rebuild"] is True
        and ident_red["identity_reachable_env_or_rebuild"] is True
        and d_ident_reach["identity_reachable_env_or_rebuild"] is True
        and ident_green["rebuild_line_items"] == 1
        and ident_red["rebuild_line_items"] == 2
        and ident_green["cost_branch"] == "env_reachable_at_decode_width"
        and ident_red["cost_branch"] == "marlin_rebuild_gated"
        and ident_pending["identity_reachable_env_or_rebuild"] is True
        and ident_pending["rebuild_line_items"] is None
        and ident_pending["cost_branch_pending"] is True)

    # an: stark #376 residual flip is the int4-Marlin body GEMM (env-unpatchable) and the RED is
    #     geometry-specific (Marlin is bit-exact at the 8-row decode-verify width)
    an_identity_residual_marlin_decode_caveat = (
        d_ident_reach["residual_is_int4_marlin_body_gemm"] is True
        and d_ident_reach["env_knob_cannot_patch_marlin"] is True
        and d_ident_reach["pin_does_not_help_stark376"] is True
        and d_ident_reach["marlin_bit_exact_at_decode_width"] is True
        and d_ident_reach["red_is_geometry_specific"] is True
        and abs(d_ident_reach["identity_residual_flip_stark376"]
                - IDENTITY_RESIDUAL_FLIP_STARK376) < TOL_EXACT)

    # --- 22:08Z max-equivalent-TPS frontier rollup (#407/#357 re-point) --- #
    # Assert default-mode values on an EXPLICIT all-feeders-pending construction so the self-test validates
    # the machinery independent of the operator's CLI feeder flags (mirrors `ab`/`am` explicit builds).  The
    # operator's LIVE rollup (d_frontier) is cross-checked only for MODE-INVARIANT properties (monotone ladder,
    # served identity 1.0 at every node) which must hold under any feeder resolution.
    fr = equivalent_tps_frontier_rollup()            # canonical default-mode (all feeders pending) rollup
    fr_nodes = fr["ladder"]
    d_fr_nodes = d_frontier["ladder"]                # operator's LIVE rollup — invariant cross-checks only
    # ba: floor node is MEASURED 467.48 at identity 1.0; pending-mode measured frontier == floor
    ba_frontier_floor_measured = (
        abs(fr_nodes[0]["tps"] - EQUIV_FLOOR_TPS) < TOL_DISPLAY_TPS
        and fr_nodes[0]["status"] == "measured"
        and abs(fr_nodes[0]["identity"] - 1.0) < TOL_EXACT
        and abs(fr["max_equivalent_tps_measured"] - EQUIV_FLOOR_TPS) < TOL_DISPLAY_TPS)
    # bb: selective-recompute node is MODELED 478.93 (denken #413 point), in band [476,479], identity 1.0;
    #     carries the lawine #417 recompute bracket [476.48, 478.48]
    bb_frontier_recompute_modeled = (
        abs(fr_nodes[1]["tps"] - SELECTIVE_RECOMPUTE_MODELED_TPS) < TOL_DISPLAY_TPS
        and fr_nodes[1]["status"] == "modeled"
        and fr_nodes[1]["in_modeled_band"] is True
        and abs(fr_nodes[1]["identity"] - 1.0) < TOL_EXACT
        and abs(fr_nodes[1]["denken413_point_tps"] - DENKEN413_RECOMPUTE_POINT_TPS) < TOL_DISPLAY_TPS
        and tuple(fr_nodes[1]["lawine417_bracket_tps"]) == SELECTIVE_RECOMPUTE_BRACKET_417_TPS)
    # bc: measured frontier <= modeled frontier (467.48 <= 492.08 bracket-lower headline)
    bc_frontier_measured_le_modeled = (
        fr["max_equivalent_tps_measured"] <= fr["max_equivalent_tps_modeled"] + TOL_DISPLAY_TPS)
    # bd: ladder TPS is monotone non-decreasing (each lever adds >= 0) — in BOTH the canonical default rollup
    #     AND the operator's live rollup (mode-invariant)
    bd_frontier_ladder_monotone = (
        all(fr_nodes[i + 1]["tps"] >= fr_nodes[i]["tps"] - TOL_DISPLAY_TPS for i in range(len(fr_nodes) - 1))
        and all(d_fr_nodes[i + 1]["tps"] >= d_fr_nodes[i]["tps"] - TOL_DISPLAY_TPS
                for i in range(len(d_fr_nodes) - 1)))
    # be: cb3 stack is BANKED additive (lawine #417 confirmed); naive point == recompute + 15.60; exact gap
    #     still pending kanna #416 (a tightening of the [492.08, 494.08] bracket, not a gate)
    be_frontier_cb3_stack_banked_additive = (
        fr["cb3_stack_banked_additive_417"] is True
        and fr_nodes[2]["banked"] is True
        and fr_nodes[2]["additive_confirmed_lawine417"] is True
        and fr_nodes[2]["exact_gap_pending_kanna416"] is True
        and abs(fr["naive_stacked_ceiling_tps"]
                - (SELECTIVE_RECOMPUTE_MODELED_TPS + CB3_CONSERVATIVE_LIFT_TPS_403)) < TOL_DISPLAY_TPS
        and tuple(fr_nodes[2]["lawine417_bracket_tps"]) == FASTEST_EQUIVALENT_BRACKET_TPS)
    # bf: the deployed 481.53 fast path is OFF the ladder (served identity 0.9966 != 1.0, 3/882 flips)
    bf_frontier_deployed_off_ladder = (
        fr["deployed_fastpath_off_ladder"]["is_equivalent"] is False
        and abs(fr["deployed_fastpath_off_ladder"]["served_identity"]
                - DEPLOYED_FASTPATH_SERVED_IDENTITY) < TOL_EXACT
        and tuple(fr["deployed_fastpath_off_ladder"]["flips"]) == DEPLOYED_FASTPATH_FLIPS)
    # bg: equivalence tax we pay today = deployed-nonequiv 481.53 - measured-equiv floor 467.48
    bg_frontier_equivalence_tax = (
        abs(fr["equivalence_tax_vs_deployed_nonequiv_tps"]
            - (DEPLOYED_FASTPATH_TPS - EQUIV_FLOOR_TPS)) < TOL_DISPLAY_TPS)
    # bh: every ladder node is served identity 1.0 (strict equivalence holds along the whole ladder) — in
    #     BOTH the canonical default rollup AND the operator's live rollup (mode-invariant)
    bh_frontier_all_nodes_identity1 = (
        all(abs(n["identity"] - 1.0) < TOL_EXACT for n in fr_nodes)
        and abs(fr["frontier_identity"] - 1.0) < TOL_EXACT
        and all(abs(n["identity"] - 1.0) < TOL_EXACT for n in d_fr_nodes)
        and abs(d_frontier["frontier_identity"] - 1.0) < TOL_EXACT)
    # bi: tree net-supply leg is CLOSED (denken #409 ~0 reliable supply)
    bi_frontier_tree_leg_closed = (
        fr["tree_leg_closed"] is True
        and abs(DENKEN409_TREE_RELIABLE_SUPPLY_TPS) < TOL_EXACT)
    # bj: pending-mode lists the 3 remaining in-flight feeders (lawine #417 + land #414 BANKED 22:26Z)
    bj_frontier_pending_feeders = (
        len(fr["pending_feeders"]) == 3
        and fr["all_feeders_resolved"] is False
        and len(fr["banked_feeders"]) >= 5
        and any("lawine#417" in b for b in fr["banked_feeders"])
        and any("land#414" in b for b in fr["banked_feeders"]))
    # bk: a MEASURED selective-recompute promotes node 1 to measured + raises the measured frontier
    fr_rc = equivalent_tps_frontier_rollup(selective_recompute_measured_tps=477.0)
    bk_frontier_recompute_measured_promotes = (
        fr_rc["ladder"][1]["status"] == "measured"
        and abs(fr_rc["max_equivalent_tps_measured"] - 477.0) < TOL_DISPLAY_TPS)
    # bl: a kanna #416 additivity gap (with recompute measured) TIGHTENS node 2 NET of the gap
    fr_cb = equivalent_tps_frontier_rollup(selective_recompute_measured_tps=477.0, cb3_additivity_gap_tps=3.0)
    bl_frontier_cb3_additivity_banks = (
        fr_cb["ladder"][2]["banked"] is True
        and fr_cb["cb3_additivity_gap_resolved"] is True
        and abs(fr_cb["ladder"][2]["tps"] - (477.0 + CB3_CONSERVATIVE_LIFT_TPS_403 - 3.0)) < TOL_DISPLAY_TPS
        and abs(fr_cb["max_equivalent_tps_modeled"]
                - (477.0 + CB3_CONSERVATIVE_LIFT_TPS_403 - 3.0)) < TOL_DISPLAY_TPS)
    # bm: a measured fixed-floor reduction (wirbel #415) adds on top; the lm_head node (free) inherits it
    fr_fl = equivalent_tps_frontier_rollup(selective_recompute_measured_tps=477.0,
                                           cb3_additivity_gap_tps=3.0, floor_reduction_tps=2.0)
    _bm_expected_top = 477.0 + CB3_CONSERVATIVE_LIFT_TPS_403 - 3.0 + 2.0   # 491.60
    bm_frontier_floor_reduction_adds = (
        abs(fr_fl["ladder"][3]["tps"] - _bm_expected_top) < TOL_DISPLAY_TPS        # floor-reduction node
        and abs(fr_fl["ladder"][4]["tps"] - _bm_expected_top) < TOL_DISPLAY_TPS    # lm_head node (free, == #3)
        and abs(fr_fl["max_equivalent_tps_modeled"] - _bm_expected_top) < TOL_DISPLAY_TPS)
    # bn: the modeled HEADLINE in pending mode is the lawine #417 banked BRACKET [492.08, 494.08] (cb3
    #     additive); the conservative lower bound 492.08 BEATS the deployed-nonequiv 481.53; the naive
    #     additive point (494.53) is carried alongside
    bn_frontier_cb3_bracket_headline_beats_481 = (
        fr["modeled_is_bracket"] is True
        and tuple(fr["max_equivalent_tps_modeled_bracket"]) == FASTEST_EQUIVALENT_BRACKET_TPS
        and abs(fr["max_equivalent_tps_modeled"] - FASTEST_EQUIVALENT_BRACKET_TPS[0]) < TOL_DISPLAY_TPS
        and fr["modeled_beats_deployed_nonequiv"] is True
        and FASTEST_EQUIVALENT_BRACKET_TPS[0] > DEPLOYED_FASTPATH_TPS            # 492.08 > 481.53
        and abs(fr["max_equivalent_tps_modeled_point"]
                - (SELECTIVE_RECOMPUTE_MODELED_TPS + CB3_CONSERVATIVE_LIFT_TPS_403)) < TOL_DISPLAY_TPS)  # 494.53
    # bo: lm_head truncation is a FREE node (land #414, self-referential gate); 0 TPS cost; the 54.07-TPS
    #     absolute full-vocab head is a CONTINGENCY (not required); the free node == the floor-reduction node
    bo_frontier_lmhead_free_self_referential = (
        fr_nodes[4]["node"] == "lmhead_truncation_free_self_referential_414"
        and abs(fr_nodes[4]["lmhead_tps_cost"]) < TOL_EXACT                      # 0.0 (free)
        and fr_nodes[4]["self_referential_gate"] is True
        and fr_nodes[4]["deployed_passes_self_referential"] is True
        and fr_nodes[4]["absolute_fullvocab_required"] is False
        and abs(fr_nodes[4]["truevocab_lmhead_tps_cost_contingency"]
                - LAND414_TRUEVOCAB_LMHEAD_TPS_COST_CONTINGENCY) < TOL_DISPLAY_TPS  # 54.07
        and abs(fr_nodes[4]["tps"] - fr_nodes[3]["tps"]) < TOL_EXACT             # free add: == node 3
        and fr["lmhead_truncation_free_self_referential"] is True
        and abs(fr["truevocab_lmhead_tps_cost_contingency"]
                - LAND414_TRUEVOCAB_LMHEAD_TPS_COST_CONTINGENCY) < TOL_DISPLAY_TPS)
    # bp: deployability is BANKED green (lawine #417): 7 served files, 41.8 GPU-min identity-verify,
    #     reversible, 1 binding in-place line, human-gated
    bp_frontier_deployability_banked_green = (
        fr["deployability_surface"] == "green"
        and fr["deployability_banked_green"] is True
        and LAWINE417_DEPLOYABLE_GREEN is True
        and LAWINE417_DEPLOY_SURFACE_FILES == 7
        and LAWINE417_DEPLOY_BINDING_INPLACE_LINES == 1
        and LAWINE417_DEPLOY_REVERSIBLE is True
        and abs(LAWINE417_DEPLOY_IDENTITY_VERIFY_GPU_MIN - 41.8) < TOL_DISPLAY_TPS)

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
        "aa_demand_central_green_robust_tiered": bool(aa_demand_central_green_robust_tiered),
        "ab_demand_leaf_dead_if_gap_forced_irreducible": bool(ab_demand_leaf_dead_if_gap_forced_irreducible),
        "ac_slope_banked_private_robust_382": bool(ac_slope_banked_private_robust_382),
        "ad_deliverability_two_tier_380": bool(ad_deliverability_two_tier_380),
        "ae_demand_baseline_cov_roundtrip": bool(ae_baseline_cov),
        "af_kappa_transfer_deliver_margin": bool(af_kappa_deliver_margin),
        "ao_supply_leaf_honest_base_below_500": bool(ao_supply_leaf_honest_base_below_500),
        "ap_supply_leaf_resolves_on_inputs": bool(ap_supply_leaf_resolves_on_inputs),
        "aq_composite_supply_x_demand_product": bool(aq_composite_supply_x_demand_product),
        "ar_composite_framing_supply_x_demand": bool(ar_composite_framing_supply_x_demand),
        "as_ubel386_floor_inflates_resolved": bool(as_ubel386_floor_inflates_resolved),
        "at_ubel389_refutes_386_breach_demand_robust": bool(at_ubel389_refutes_386_breach_demand_robust),
        "au_denken383_supply_lift_required_first": bool(au_denken383_supply_lift_required_first),
        "av_lawine372_supply_lever_alive": bool(av_lawine372_supply_lever_alive),
        "aw_wirbel384_lmhead_free_supply_tax_in_body": bool(aw_wirbel384_lmhead_free_supply_tax_in_body),
        "ax_wirbel393_corrected_base_467_headline_ppl_dead": bool(ax_wirbel393_corrected_base_467_headline_ppl_dead),
        "ay_terminal_go_is_supply_leg_and_tree_leg": bool(ay_terminal_go_is_supply_leg_and_tree_leg),
        "az_denken387_measured_anchor_mtp_k7_kanna374_fusion_closed": bool(
            az_denken387_measured_anchor_mtp_k7_kanna374_fusion_closed),
        "ai_gap_fractions_sum_to_one": bool(ai_gap_fracs_sum_to_one),
        "aj_gap_irreducible_floor_matches_ctxlen": bool(aj_gap_floor_matches_ctxlen),
        "ak_gap_corners_clear_knife_edge": bool(ak_gap_corners_clear_knife_edge),
        "al_gap_reconciles_denken377_numerics_cancels": bool(al_gap_reconciles_numerics_cancels),
        "am_identity_reachable_both_branches_cost_differs": bool(am_identity_reachable_both_branches),
        "an_identity_residual_marlin_decode_width_caveat": bool(an_identity_residual_marlin_decode_caveat),
        # 22:08Z max-equivalent-TPS frontier rollup (#407/#357 re-point); 22:26Z banked lawine #417 + land #414
        "ba_frontier_floor_measured_467_identity1": bool(ba_frontier_floor_measured),
        "bb_frontier_recompute_modeled_478_in_band": bool(bb_frontier_recompute_modeled),
        "bc_frontier_measured_le_modeled": bool(bc_frontier_measured_le_modeled),
        "bd_frontier_ladder_monotone_nondecreasing": bool(bd_frontier_ladder_monotone),
        "be_frontier_cb3_stack_banked_additive_417": bool(be_frontier_cb3_stack_banked_additive),
        "bf_frontier_deployed_481_off_ladder_nonequiv": bool(bf_frontier_deployed_off_ladder),
        "bg_frontier_equivalence_tax_481_minus_467": bool(bg_frontier_equivalence_tax),
        "bh_frontier_all_nodes_served_identity1": bool(bh_frontier_all_nodes_identity1),
        "bi_frontier_tree_leg_closed_denken409": bool(bi_frontier_tree_leg_closed),
        "bj_frontier_pending_feeders_3_lawine417_land414_banked": bool(bj_frontier_pending_feeders),
        "bk_frontier_recompute_measured_promotes": bool(bk_frontier_recompute_measured_promotes),
        "bl_frontier_cb3_additivity_gap_tightens_net": bool(bl_frontier_cb3_additivity_banks),
        "bm_frontier_floor_reduction_adds_lmhead_free_inherits": bool(bm_frontier_floor_reduction_adds),
        "bn_frontier_cb3_bracket_headline_492_494_beats_481": bool(bn_frontier_cb3_bracket_headline_beats_481),
        "bo_frontier_lmhead_free_self_referential_414": bool(bo_frontier_lmhead_free_self_referential),
        "bp_frontier_deployability_banked_green_417": bool(bp_frontier_deployability_banked_green),
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
               lmhead_eta: float | None = None, *,
               supply_lift_available_tps: float | None = None,
               demand_reaches_500_on_floor: str = DENKEN383_REACHES_500_ON_FLOOR,  # banked "no" (#383)
               supply_lift_required_tps: float | None = SUPPLY_LIFT_REQUIRED_FIRST_TPS_383,  # +17.2
               robust_pilot: str = "pending",
               gap_addressable_pp: float | None = None,
               stark381_decode: str = "pending",
               irreducible_floor_survives_vbi: str = UBEL386_FLOOR_SURVIVES_VBI,  # banked "inflates" (#386)
               kanna403_ppl_safe_supply: str = "pending",      # SUPPLY leg (PPL-safe conservative-k re-cost)
               ubel401_tree_coverage_ceiling: str = "pending", # DEMAND leg (tree top-8/16 coverage ceiling)
               denken402_tree_net_supply: str = "pending",     # DEMAND leg (tree net after verify-M tax)
               # 22:08Z RE-POINT feeders — the max-equivalent-TPS frontier ladder
               selective_recompute_measured_tps: float | None = None,  # stark #412 measured equiv TPS (#397)
               cb3_additivity_gap_tps: float | None = None,     # kanna #416 measured (recompute+cb3) additivity gap (tightens)
               floor_reduction_tps: float | None = None,        # wirbel #415 measured fixed-floor reduction
               deployability_surface: str = "green"             # lawine #417 BANKED green 22:26Z {green|red|pending}
               ) -> dict[str, Any]:
    d1 = deliverable1_lever_analysis(b_star)
    d2 = deliverable2_ppl_forecast()
    d3 = deliverable3_composite_tps(b_star)
    d4 = verdict_given_ppl(measured_ppl, b_star, lmhead_eta)              # Route A supply-side (deflated)
    d_ident = identity_locus_analysis(lmhead_eta)                        # eta-locus lever (deflated read)
    d_ident_reach = identity_reachability_analysis(stark381_decode)      # identity REACHABILITY compliance
    d_supply = supply_side_base_analysis(supply_lift_available_tps, demand_reaches_500_on_floor,
                                         supply_lift_required_tps, stark381_decode,
                                         kanna403_ppl_safe_supply)        # SUPPLY leaf (binding; kanna #403 leg)
    d8 = gap_decomposition_analysis(irreducible_floor_survives_vbi)       # ubel #379 GREEN + ubel #386 pend
    d6 = demand_side_route_analysis(robust_pilot, gap_addressable_pp,     # DEMAND leaf (tree net-supply leg)
                                    irreducible_floor_survives_vbi,
                                    ubel401_tree_coverage_ceiling, denken402_tree_net_supply)
    d7 = composite_verdict(measured_ppl, b_star, lmhead_eta,
                           supply_lift_available_tps=supply_lift_available_tps,
                           demand_reaches_500_on_floor=demand_reaches_500_on_floor,
                           supply_lift_required_tps=supply_lift_required_tps,
                           robust_pilot=robust_pilot,
                           gap_addressable_pp=gap_addressable_pp,
                           stark381_decode=stark381_decode,
                           irreducible_floor_survives_vbi=irreducible_floor_survives_vbi,
                           kanna403_ppl_safe_supply=kanna403_ppl_safe_supply,
                           ubel401_tree_coverage_ceiling=ubel401_tree_coverage_ceiling,
                           denken402_tree_net_supply=denken402_tree_net_supply)
    d5 = deliverable5_caveats(b_star)
    # 22:08Z CAPSTONE: max-equivalent-TPS frontier rollup over the equivalence ladder.
    d_frontier = equivalent_tps_frontier_rollup(
        selective_recompute_measured_tps=selective_recompute_measured_tps,
        cb3_additivity_gap_tps=cb3_additivity_gap_tps,
        floor_reduction_tps=floor_reduction_tps,
        deployability_surface=deployability_surface)
    st = _selftests(d1, d2, d3, d4, d5, d6, d7, d8, d_supply, d_ident_reach, d_frontier, b_star)

    headline = {
        # ====================================================================
        # 22:08Z CAPSTONE METRIC — max equivalent TPS (strict byte-exact greedy-
        # token-identity, served identity == 1.0).  The #407 human directive
        # (21:13Z) dropped 500 as the target; advisor re-point (22:08Z) made the
        # frontier the live objective.  The >500 rollup is now historical_sub_result.
        # ====================================================================
        "max_equivalent_tps_measured": d_frontier["max_equivalent_tps_measured"],     # 467.48 (MEASURED, identity 1.0)
        "max_equivalent_tps_modeled": d_frontier["max_equivalent_tps_modeled"],       # 492.08 (MODELED bracket lower; BEATS 481.53)
        "max_equivalent_tps_modeled_bracket": d_frontier["max_equivalent_tps_modeled_bracket"],  # [492.08,494.08] lawine #417
        "max_equivalent_tps_modeled_point": d_frontier["max_equivalent_tps_modeled_point"],      # 494.53 (denken #413 additive point)
        "frontier_modeled_beats_deployed_nonequiv": d_frontier["modeled_beats_deployed_nonequiv"],  # True (492.08 > 481.53)
        "frontier_cb3_stack_banked_additive_417": d_frontier["cb3_stack_banked_additive_417"],   # True (lawine #417)
        "frontier_naive_stacked_ceiling_tps": d_frontier["naive_stacked_ceiling_tps"],  # 494.53 (recompute+cb3 additive point)
        "frontier_lmhead_truncation_free_self_referential": d_frontier["lmhead_truncation_free_self_referential"],  # True (land #414)
        "frontier_truevocab_lmhead_tps_cost_contingency": d_frontier["truevocab_lmhead_tps_cost_contingency"],  # 54.07 (not required)
        "frontier_identity": d_frontier["frontier_identity"],                         # 1.0 (every ladder node)
        "frontier_equivalence_tax_vs_deployed_tps": d_frontier["equivalence_tax_vs_deployed_nonequiv_tps"],  # 481.53-467.48
        "frontier_deployability_surface": d_frontier["deployability_surface"],        # "green" (lawine #417 banked)
        "frontier_pending_feeders": d_frontier["pending_feeders"],
        "frontier_all_feeders_resolved": d_frontier["all_feeders_resolved"],
        "deployed_fastpath_is_equivalent": d_frontier["deployed_fastpath_off_ladder"]["is_equivalent"],  # False (0.9966)
        "deployed_fastpath_served_identity": d_frontier["deployed_fastpath_off_ladder"]["served_identity"],  # 0.9966
        # PRIMARY analytic gate (0-GPU self-test integrity)
        "strict_500_composite_reachability_self_test_passes": (
            st["strict_500_composite_reachability_self_test_passes"]),
        # ====================================================================
        # HISTORICAL SUB-RESULT (NOT the capstone): the binary ">500 reachable?"
        # AND-PRODUCT GO rollup.  Kept for provenance per advisor 22:08Z; resolves
        # to strict_500_reachable_via_known_levers (run 34tv4krw=0 with the banked
        # feeders).  private_500_GO <=> (supply base enables 500) AND (demand delivers).
        # ====================================================================
        "tps_max_optimistic_nonspec": d3["tps_max_optimistic_nonspec"],
        "tps_max_optimistic_spec": d3["tps_max_optimistic_spec"],
        "strict_500_reachable_via_known_levers": d7["private_500_reachable_via_known_levers"],
        "go_formula": d7["go_formula"],
        "supply_base_enables_500": d7["supply_base_enables_500"],
        "demand_closer_delivers": d7["demand_closer_delivers"],
        "demand_alone_may_be_insufficient": d7["demand_alone_may_be_insufficient"],
        "identity_reachable_env_or_rebuild": d7["identity_reachable_env_or_rebuild"],
        "identity_cost_branch": d7["identity_cost_branch"],
        "identity_cost_branch_pending": d7["identity_cost_branch_pending"],
        "identity_rebuild_line_items": d7["identity_rebuild_line_items"],
        "refinement_inputs": d7["refinement_inputs"],
        "cost_refinement_inputs": d7["cost_refinement_inputs"],
        "primary_route": d7["primary_route"],
        "eta_axis_deflated": d7["eta_axis_deflated"],
        "binding_constraint": d7["binding_constraint"],
        "residual_gap_to_500": d4["residual_gap_to_500"],
        # supply-side base leaf (wirbel #393 corrected base 467.48, now-BINDING GO axis; kanna #403 leg pending)
        "supply_base_today_tps": d_supply["supply_base_today_tps"],          # 467.48 (wirbel #393)
        "supply_base_clears_500_today": d_supply["supply_base_clears_500_today"],
        "honest_strict_base_floor_378": d_supply["honest_strict_base_floor"],
        "honest_strict_base_plus_attn_378": d_supply["honest_strict_base_plus_attn"],
        "deficit_to_500_today_tps": d_supply["deficit_to_500_today_tps"],    # 32.52
        "supply_pending": d_supply["supply_pending"],
        "supply_binding_constraint": d_supply["binding_constraint"],
        "realized_deployed_strict_393": d_supply["realized_deployed_strict_393"],  # 467.48 (#393, supersedes 471.42)
        "shippable_ceiling_393": d_supply["shippable_ceiling_393"],          # 505.29
        "gap_to_500_393": d_supply["gap_to_500_393"],                        # 32.52
        "decode_attn_strict_tax_pct_393": d_supply["decode_attn_strict_tax_pct_393"],  # 3.01%
        "realized_deployed_strict_390": d_supply["realized_deployed_strict_390"],  # 471.42 (superseded by #393)
        "n_kernel_rebuilds_strict_500_390": d_supply["n_kernel_rebuilds_strict_500_390"],
        "eta_attn_390": d_supply["eta_attn_390"],
        "shippable_ceiling_510_reinstated_390": d_supply["shippable_ceiling_510_reinstated_390"],
        "shippable_ceiling_518_still_refuted_390": d_supply["shippable_ceiling_518_still_refuted_390"],
        "cb3_lift_honest": d_supply["cb3_lift_honest"],                      # +32.65 HEADLINE — PPL-DEAD (kanna #394)
        "cb3_headline_lift_ppl_dead_394": d_supply["cb3_headline_lift_ppl_dead_394"],  # True (do NOT carry +32.65)
        "base_plus_cb3_honest": d_supply["base_plus_cb3_honest"],           # 500.13 — arithmetic only (PPL-dead)
        "supply_alone_clears_500_with_cb3": d_supply["supply_alone_clears_500_with_cb3"],  # False (PPL-dead headline)
        "cb3_deployable_at_conservative_k_394": d_supply["cb3_deployable_at_conservative_k_394"],  # True (k=232 ~2.39)
        "measured_top4_coverage_387": d_supply["measured_top4_coverage_387"],
        "deployed_drafter_mtp_k_387": d_supply["deployed_drafter_mtp_k_387"],
        "drafter_is_mtp_not_eagle3_387": d_supply["drafter_is_mtp_not_eagle3_387"],
        # TERMINAL GO-flip gate (advisor 20:19Z REPRICE): kanna #403 (supply leg) AND >=1 of {ubel #401,
        # denken #402} (demand-leg tree net-supply); supersedes the 19:25Z cb3-deployability pair (RED)
        "terminal_go": d7["terminal_go"],                                   # green / red / pending
        "terminal_go_confirmed": d7["terminal_go_confirmed"],
        "terminal_go_blocked": d7["terminal_go_blocked"],
        "terminal_go_pending": d7["terminal_go_pending"],
        "terminal_go_pending_gates": d7["terminal_go_pending_gates"],
        "reprice_go_flip_gates": d7["reprice_go_flip_gates"],
        "kanna403_ppl_safe_supply": d7["kanna403_ppl_safe_supply"],
        "ubel401_tree_coverage_ceiling": d7["ubel401_tree_coverage_ceiling"],
        "denken402_tree_net_supply": d7["denken402_tree_net_supply"],
        "tree_net_supply_green": d7["tree_net_supply_green"],
        "tree_net_supply_both_red": d7["tree_net_supply_both_red"],
        "tree_net_supply_pending": d7["tree_net_supply_pending"],
        "superseded_1925z_pair": d7["superseded_1925z_pair"],
        # demand-side residual leaf (denken #377 sized; #380 two-tier; #382 slope GREEN; RESOLVED central)
        "demand_leaf_delivers": d6["demand_leaf_delivers"],
        "demand_leaf_robust_pending": d6["leaf_robust_pending"],
        "demand_central_green": d7["demand_central_green"],
        "demand_robust_resolved": d7["demand_robust_resolved"],
        "demand_robust_pending": d7["demand_robust_pending"],
        "demand_conservative_target": d7["demand_conservative_target"],
        "demand_closer_central_c": d6["demand_closer_central_c"],
        "demand_closer_robust_c": d6["demand_closer_robust_c"],
        "p_deliver_central_defensible_380": d6["p_deliver_central_defensible"],
        "p_deliver_robust_defensible_380": d6["p_deliver_robust_defensible"],
        "kappa_axis_robust_380": d6["kappa_axis_robust"],
        "recommended_retrain_target_c": d6["recommended_retrain_target_c"],
        "delta_cov_robust": d6["delta_cov_robust"],
        "delta_cov_robust_budget_frac": d6["delta_cov_robust_budget_frac"],
        "within_336_budget": d6["within_336_budget"],
        "noniid_price_multiplier": d6["noniid_price_multiplier"],
        "gap_shrink_per_coverage": d6["gap_shrink_per_coverage"],
        "public_private_gap_pct": d6["public_private_gap_pct"],
        "kappa_int4_ct_transfer": d6["kappa_int4_ct_transfer"],
        # ubel #379 (GREEN) gap-decomposition ceiling-check — BANKED
        "gap_addressable_pp_ubel379": d8["gap_addressable_pp"],
        "gap_irreducible_pp_central_ubel379": d8["gap_irreducible_pp_central"],
        "gap_channel_live": d8["gap_channel_live"],
        "closer_not_capped_by_irreducible_floor": d8["closer_not_capped_by_irreducible_floor"],
        "coverage_target_for_3p2_ubel379": d8["coverage_target_for_3p2"],
        "slope_tps_per_coverage_ubel379": d8["slope_tps_per_coverage_ubel379"],
        "gap_after_max_coverage_retrain_pct_ubel379": d8["gap_after_max_coverage_retrain_pct"],
        # ubel #382 slope private-OOD robustness — BANKED GREEN (489.8 -> 437.3 private)
        "slope_is_private_robust_382": d6["slope_is_private_robust"],
        "slope_tps_per_coverage_private_382": d6["slope_tps_per_coverage_private_382"],
        "slope_flattening_ratio_382": d6["slope_flattening_ratio_382"],
        # ubel #386 irreducible-floor-under-VBI=1 (RESOLVED RED: inflates 2.07x -> 1.310% central)
        "irreducible_floor_vbi_pending_ubel386": d8["irreducible_floor_vbi_pending_ubel386"],
        "irreducible_floor_inflates_vbi_386": d8["irreducible_floor_inflates_vbi_386"],
        "irreducible_floor_vbi1_central_pct_386": d8["irreducible_floor_vbi1_central_pct_386"],
        "all_corners_clear_3p2_vbi1_386": d8["all_corners_clear_3p2_vbi1_386"],
        "uncapped_on_live_vbi_stack": d8["uncapped_on_live_vbi_stack"],
        # pending-aware extras (GO-gating = SUPPLY-base hardeners only; Route A excluded from GO)
        "verdict_pending": d7["verdict_pending"],          # composite GO (supply base pending)
        "pending_inputs": d7["pending_inputs"],            # GO-gating: denken #383 + wirbel #384
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
            f"PRIVATE-500 GO HELD ON THE 20:19Z REPRICE TERMINAL GATE — the GO is an AND-PRODUCT: (SUPPLY "
            f"leg kanna #403 PPL-safe conservative-k cb3 re-cost) x (DEMAND leg >=1 of ubel #401 / denken "
            f"#402 tree net-supply). GO-gating pending inputs: {d7['pending_inputs']}"
            f"{' (+refinements ' + str(d7['refinement_inputs']) + ')' if d7['refinement_inputs'] else ''}. "
            f"SUPPLY BASE (wirbel #393 LANDED, the NOW-BINDING axis): the corrected deployable-strict base "
            f"is {d_supply['supply_base_today_tps']:g} (wirbel #393 re-priced 471.42 -> "
            f"{d_supply['supply_base_today_tps']:g} on the DECODE-specific "
            f"{d_supply['decode_attn_strict_tax_pct_393']:g}% attn strict tax; gap_to_500 "
            f"{d_supply['deficit_to_500_today_tps']:g}, ceiling {d_supply['shippable_ceiling_393']:g}, "
            f"rebuilds=1, attn the SOLE strict tax) < 500. ★ 20:19Z RE-PRICE: lawine #388/#392 sized the cb3 "
            f"body-allocation lift +{d_supply['cb3_lift_honest']:g} honest HEADLINE, BUT kanna #394 (LANDED "
            f"RED) made it PPL-DEAD — #372's in-sample margin is winner's-curse (held-out "
            f"{d_supply['kanna394_heldout_worst_seed_ppl']:g} + OOD {d_supply['kanna394_ood_sharegpt_ppl']:g} "
            f"breach {KANNA394_PPL_GATE:g}); so {d_supply['supply_base_today_tps']:g}+"
            f"{d_supply['cb3_lift_honest']:g} is arithmetic-only, NOT deployable. cb3 STAYS deployable at a "
            f"conservative k={d_supply['kanna394_conservative_k_232']} "
            f"(~{d_supply['kanna394_conservative_k_heldout_ppl']:g} held-out) at a smaller UN-COSTED lift -> "
            f"the SUPPLY leg is PENDING kanna #403 (largest k with held-out worst-seed <= "
            f"{KANNA403_HELDOUT_WORST_SEED_TARGET:g}). {d7['binding_constraint']}. "
            f"DEMAND RESIDUAL (20:19Z RE-PRICED DOWN to tree-only): denken #396 (LANDED RED) -> demand-ALONE "
            f"busts EVEN BARE on {d_supply['supply_base_today_tps']:g} (required Δcov "
            f"{d6['denken396_required_dcov_on_467']:g} = "
            f"{d6['denken396_required_dcov_on_467_budget_frac']*100:g}% of the +{BUDGET_336_PLUS_031:g} "
            f"budget); ubel #399 (LANDED RED) -> NO cheap deployable demand lever (every monotone draft-head "
            f"lever is a RANK-INVARIANT no-op, MC max|Δcov|={d6['ubel399_mc_max_dcov']:g}). So the d-cov can "
            f"ONLY be SUPPLIED by the TREE (the locked +{d6['ubel399_tree_top1_to_top4_prize']:g} top-1->"
            f"top-4 prize) -> the DEMAND leg is PENDING >=1 of ubel #401 (top-8/16 tree coverage ceiling) / "
            f"denken #402 (tree NETs d-cov after its verify-M step-time tax). The sized closer "
            f"~{DEMAND_CONSERVATIVE_TARGET_382} is PROVENANCE (a drafter retrain is forbidden). ubel #382 "
            f"(GREEN) BANKED the slope private-robust ({d8['slope_tps_per_coverage_ubel379']:g} -> "
            f"{d6['slope_tps_per_coverage_private_382']:.1f} private, flattening {SLOPE_FLATTENING_RATIO_382}); "
            f"ubel #379 (GREEN) BANKED the gap ceiling-check ({d8['gap_addressable_pp']:.3f}pp of the "
            f"{PUBLIC_PRIVATE_GAP_PCT}pp gap coverage-addressable); ubel #389 (LANDED) REFUTED the #386 breach "
            f"(measured floor {d6['ubel389_measured_floor_vbi1_pct']:g}% < 0.633%, 0 corners breach 3.2%). "
            f"IDENTITY (strict-lock compliance, folds into the supply leaf, REACHABLE in BOTH #381 branches): "
            f"{'cost branch PENDING stark #381 (env@decode=1 rebuild vs Marlin-gated=2 rebuilds)' if d7['identity_cost_branch_pending'] else d7['identity_cost_branch'] + ' (' + str(d7['identity_rebuild_line_items']) + ' rebuild line-item(s))'}"
            f" — #381 sets COST, not GO. Route A (sub-int4 eta-axis standalone): "
            f"{'PENDING denken #356 PPL@b* + stark #365 lm_head eta' if d4['verdict_pending'] else d4['binding_constraint']} "
            f"-- DEFLATED/un-deployable (PPL-blocked lawine #372; #373/#375), EXCLUDED from the GO."
        )
    else:
        handoff = (
            f"PRIVATE-500 GO RESOLVED: " + d7["verdict_text"]
        )

    # HISTORICAL SUB-RESULT: the binary >500 verdict, kept for provenance (run 34tv4krw=0).
    # The FROZEN historical record is run 34tv4krw = strict_500_reachable_via_known_levers=0 (RED): with
    # the banked feeders (kanna #403 PPL-safe supply +15.60 < the +17.2 required-first lift; tree leg CLOSED
    # per denken #409 ~0), the AND-product GO is RED.  The LIVE leg here resolves to whatever the legacy CLI
    # flags select (default = pending/None, since the >500 leg is no longer the thing being resolved).
    historical_sub_result = {
        "label": "strict_500_reachable_via_known_levers (binary >500 GO rollup — HISTORICAL, superseded 22:08Z)",
        "frozen_run_34tv4krw_strict_500_reachable_via_known_levers": 0,   # RED — the preserved historical record
        "frozen_run_id": "34tv4krw",
        "frozen_verdict": "red",
        "live_leg_strict_500_reachable_via_known_levers": d7["private_500_reachable_via_known_levers"],
        "live_leg_terminal_go": d7["terminal_go"],
        "live_leg_binding_constraint": d7["binding_constraint"],
        "strict_500_reachable_via_known_levers": d7["private_500_reachable_via_known_levers"],  # live (back-compat)
        "terminal_go": d7["terminal_go"],
        "binding_constraint": d7["binding_constraint"],
        "superseded_by": "max_equivalent_tps frontier (advisor 22:08Z re-point of #407 human directive)",
        "note": ("the human dropped 500 as the target in #407 (21:13Z); this binary rollup is retained ONLY "
                 "as a historical sub-result, NOT the capstone. FROZEN record: run 34tv4krw = 0 (RED). The "
                 "capstone is equivalent_tps_frontier (max_equivalent_tps)."),
    }
    return {
        "headline": headline,
        "equivalent_tps_frontier": d_frontier,           # 22:08Z CAPSTONE (max-equivalent-TPS frontier ladder)
        "historical_sub_result": historical_sub_result,  # binary >500 verdict (provenance only)
        "deliverable1_lever_analysis": d1,
        "deliverable2_ppl_forecast": d2,
        "deliverable3_composite_tps": d3,
        "deliverable4_verdict": d4,
        "deliverable_identity_locus": d_ident,
        "deliverable_identity_reachability": d_ident_reach,
        "deliverable_supply_side_base": d_supply,
        "deliverable8_gap_decomposition": d8,
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
                "DEMAND-SIDE residual leaf (advisor-relayed 16:02Z/16:41Z): denken #377 030uc5mk "
                "(merged, independently verified) recommended_retrain_target c*>=0.9010, Δcov robust 0.0107 / "
                "central 0.00565, non-iid cov->E[T] slope 7.91 (vs iid 11.12, 1.41x pricier) anchored on #289 "
                "a1=0.7293 cliff, kappa=0.672 int4-ct transfer, gap_shrink 0.3914pp/cov of the 4.295pp "
                "public->private gap, within #336's +0.031 envelope (35% robust / 18% central), triple-tail "
                "out-of-budget corner (ceil 509.07, rho 0.8038, c* 0.9256 = 136%); deliverability prior "
                "#339 N(0.0385,0.0074). GAP CEILING-CHECK BANKED: ubel #379 5kpb73tb (GREEN, independently "
                "verified) — 4.295pp gap = 85.25% acceptance (coverage-addressable) + 14.75% ctxlen "
                "(irreducible) + 0% outlen + 0% numerics; numerics/identity tax CANCELS in the public->private "
                "step diff (floors absolute TPS, not the gap); irreducible floor 0.633% central, corners "
                "{0.0,0.633,1.647} each clear the 3.2% knife-edge by >=1.5pp; coverage_target_for_3p2=0.9011 "
                "(+0.0108) reconciles #377 to <=0.0003; slope 489.8 TPS/unit; gap_after_max_coverage_retrain "
                "1.142% (~3x headroom). IDENTITY REACHABILITY two-branch: stark #376 ipe3ofie (RED) — on real "
                "weights, pinning attention (num_splits=1) leaves e2e identity 0.992555 (~heuristic 0.992708); "
                "residual ~0.73% flip is the int4-Marlin body GEMM (custom CUDA op outside the aten dispatcher, "
                "VLLM_BATCH_INVARIANT cannot patch it), but the RED is at the prefill-replication geometry "
                "(2048 rows); Marlin is BIT-EXACT at the 8-row decode-verify width. stark #381 (reseated) "
                "resolves the served geometry: GREEN -> env-reachable@decode (1 rebuild: #375 mha_varlen); "
                "RED -> Marlin-rebuild-gated (2 rebuilds). Identity is REACHABLE in BOTH; #381 sets COST, not "
                "GO; it folds into the SUPPLY leaf as the strict-byte-exact compliance prerequisite. "
                "GO REFRAME (advisor 16:45Z/17:03Z/17:16Z): GO = (supply-side public-strict base ENABLES 500) "
                "x (demand closer DELIVERS the residual). SUPPLY base (wirbel #378 gghmgtk9): honest "
                "deployable-strict base <=480.7-today (floor 469.68 off-shelf VBI=1 + ~11-TPS #375 attn "
                "rebuild), BOTH < 500; the 518.92 eta-axis pin DEFLATED on three grounds (#373 498.58<500 + "
                "#375 un-deployable + #378 rebuild buys only ~11 TPS, eta_attn 0.0215); dominant non-attention "
                "strict overhead is the int4-Marlin BODY (#376) — NOT bf16 lm_head (wirbel #384 refuted the "
                "~150-TPS by-elimination figure; lm_head is FREE). STRUCTURAL: private <= public, so a public base "
                "< 500 can make private-500 unreachable demand-alone. DEMAND residual RESOLVED central=GREEN / "
                "robust=pending-pilot: denken #380 00oijpwg (BANKED YELLOW) two-tier deliverability (central "
                "c>=0.8959 p_deliver 0.958>=0.90 GREEN; robust c>=0.9010 p_deliver 0.811<0.90 -> ~25 "
                "A10G-GPU-hr pilot; kappa-axis ROBUST breakeven 0.1222<<0.354; defensible fine-tune "
                "N(0.016,0.006)); ubel #382 bn0v5rqr (BANKED GREEN) slope survives private OOD (489.8->437.3, "
                "flattening 0.893, conservative target ~0.911 = 66.6% of #336 budget vs 38.9% central). "
                "CYCLE-53 MERGES (17:53Z): denken #383 t68af2yw (RED) CONFIRMED demand-alone insufficient on "
                "the honest base (private-on-floor 450.2 -> residual +49.8 TPS = 1.84x budget) -> +17.2-TPS "
                "supply lift required FIRST; ubel #386 xxzujn7a (RED) the off-VBI 0.633% gap floor INFLATES "
                "2.07x -> 1.310% central under VBI=1 (central still clears 3.2% +1.89pp, but all corners no "
                "longer clear -> prompt-shift sensitivity BINDING, breakeven 253->119 tok); lawine #372 "
                "mpzfw116 (GREEN) a SUPPLY lever is ALIVE — sensitivity-weighted mixed-precision allocation "
                "(88.8% body @3-bit, 3.2369 avg bpw, +0.17% PPL, gate 2.3812<=2.42, -21.5% body read; uniform "
                "3-bit died on PPL). CORRECTION (18:12Z): wirbel #384 4f32ks1e (RED) REFUTED the lm_head-BI "
                "supply lever — the deployed lm_head is already byte-exact int4-Marlin at decode "
                "(should_use_atomic_add_reduce(M=8,n=16384)=False -> fixed fp32 reduce; eta_lmhead=0, FREE; "
                "f_lmhead=0.0224); #378's ~93% was a by-elimination artifact, the dominant non-attention tax "
                "lives in the int4-Marlin BODY (#376); corrected ledger = 2 kernel rebuilds (attn #375 + body "
                "#376), NOT 3, lm_head free; the 510.01/518.92 SHIPPABLE ceilings (bf16-lm_head-premised) are "
                "refuted -> wirbel #390 reseated to re-roll the corrected realized shippable number. Route A "
                "(sub-int4 UNIFORM eta-axis standalone) stays deflated/excluded. PENDING GO (now SUPPLY-binding): "
                "a MEASURED supply lift >= +17.2 TPS from wirbel #390 (corrected SHIPPABLE strict ceiling) OR "
                "lawine #388 (realized TPS of #372's body allocation). REFINEMENTS (not GO-gating): robust "
                "coverage-lift pilot (off critical path) + ubel #389 (pin the VBI=1 floor breach) + stark #381 "
                "(decode-width identity cost: body free at decode -> 1 rebuild vs Marlin-gated 2 rebuilds)."
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
    print("MAX-EQUIVALENT-TPS FRONTIER (#357, fern) — CAPSTONE re-pointed 22:08Z (#407 dropped 500): maximize "
          "TPS subject to STRICT byte-exact greedy-token-equivalence (served identity == 1.0)", flush=True)
    print("=" * 98, flush=True)
    fr = syn["equivalent_tps_frontier"]
    hsr = syn["historical_sub_result"]
    _brk = fr["max_equivalent_tps_modeled_bracket"]
    print("  (CAPSTONE) MAX-EQUIVALENT-TPS FRONTIER — ladder of strictly-equivalent configs (identity 1.0)",
          flush=True)
    print(f"      max_equivalent_tps  MEASURED={fr['max_equivalent_tps_measured']:.2f} (identity 1.0)  "
          f"MODELED bracket=[{_brk[0]:.2f}, {_brk[1]:.2f}] (lawine #417; point {fr['max_equivalent_tps_modeled_point']:.2f})",
          flush=True)
    print(f"      -> MODELED frontier BEATS the non-strict deployed {fr['deployed_fastpath_off_ladder']['tps']:.2f} "
          f"(by +{_brk[0] - fr['deployed_fastpath_off_ladder']['tps']:.2f} at the conservative lower bound) "
          f"WITH the byte-identity 481.53 lacks  [beats={fr['modeled_beats_deployed_nonequiv']}]", flush=True)
    print(f"      deployed fast path {fr['deployed_fastpath_off_ladder']['tps']:.2f} is OFF the ladder: "
          f"served identity {fr['deployed_fastpath_off_ladder']['served_identity']:.4f} != 1.0 "
          f"({fr['deployed_fastpath_off_ladder']['flips'][0]}/{fr['deployed_fastpath_off_ladder']['flips'][1]} "
          f"flips); equivalence tax vs floor {fr['equivalence_tax_vs_deployed_nonequiv_tps']:.2f} TPS", flush=True)
    for n in fr["ladder"]:
        print(f"        - {n['node']:<46} {n['tps']:7.2f} TPS  [{n['status']:<8}] identity={n['identity']:.2f}",
              flush=True)
    print(f"      cb3 stack BANKED additive (lawine #417); lm_head truncation FREE (land #414, self-referential "
          f"gate); true-vocab head contingency {fr['truevocab_lmhead_tps_cost_contingency']:.2f} TPS (NOT required)",
          flush=True)
    print(f"      deployability: {fr['deployability_surface']} (lawine #417 BANKED); pending feeders: "
          f"{fr['pending_feeders']}  (all_resolved={fr['all_feeders_resolved']})", flush=True)
    print(f"      {fr['tree_leg_note']}", flush=True)
    print(f"      HISTORICAL sub-result (>500 binary GO, superseded): FROZEN run {hsr['frozen_run_id']} = "
          f"strict_500_reachable_via_known_levers={hsr['frozen_run_34tv4krw_strict_500_reachable_via_known_levers']} "
          f"(RED); live leg={hsr['live_leg_strict_500_reachable_via_known_levers']} "
          f"(terminal_go={hsr['live_leg_terminal_go']})", flush=True)
    print("-" * 98, flush=True)
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
    print("  (D4) ROUTE A — supply-side sub-int4 (PPL@b* AND lm_head eta) — DEFLATED, EXCLUDED from the GO",
          flush=True)
    print(f"      route_a_pending = {d4['verdict_pending']}  "
          f"(ppl={d4['verdict_pending_measured_ppl']}, identity_eta={d4['verdict_pending_identity_eta']})  "
          f"reachable={d4['strict_500_reachable_via_known_levers']}  (deflated/un-deployable standalone: "
          f"PPL-blocked lawine #372; eta-axis #373/#375)", flush=True)
    d6 = syn["deliverable6_demand_side_route"]
    d7 = syn["deliverable7_composite_verdict"]
    d8 = syn["deliverable8_gap_decomposition"]
    dir_ = syn["deliverable_identity_reachability"]
    print("-" * 98, flush=True)
    print("  (D6) DEMAND residual leaf — coverage closer (denken #377/#380/#382) — RESOLVED central=GREEN "
          "/ robust=pending-pilot", flush=True)
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
    print(f"      triple-tail corner (sensitivity band): c*={d6['triple_tail_corner']['worst_c_star']}, "
          f"cost {d6['triple_tail_corner']['cost_frac_of_336_budget']*100:.0f}% of budget "
          f"(out_of_budget={d6['triple_tail_corner']['out_of_budget']})", flush=True)
    print(f"      deliverability TWO-TIER (denken #380): central c>={d6['demand_closer_central_c']} "
          f"p_deliver={d6['p_deliver_central_defensible']}>=0.90 GREEN | robust c>={d6['demand_closer_robust_c']} "
          f"p_deliver={d6['p_deliver_robust_defensible']}<0.90 -> {'pending pilot' if d6['robust_pilot_pending'] else ('pilot delivers' if d6['deliver_robust_resolved'] else 'pilot fails')}  "
          f"(kappa-axis robust={d6['kappa_axis_robust']}, breakeven {d6['kappa_breakeven_380']})", flush=True)
    print(f"      slope (ubel #382 BANKED GREEN): {d6['slope_tps_per_coverage_ubel379']:g}->"
          f"{d6['slope_tps_per_coverage_private_382']:.1f} private (flattening {d6['slope_flattening_ratio_382']}); "
          f"conservative target {d6['demand_conservative_target_382']} = "
          f"{d6['demand_budget_frac_conservative_382']*100:.0f}% of #336 budget (central "
          f"{d6['demand_budget_frac_central_382']*100:.0f}%)", flush=True)
    print(f"      ubel #386 (RESOLVED RED): demand floor INFLATES to {d6['irreducible_floor_vbi1_central_pct_386']:g}% "
          f"under VBI=1 (central clears 3.2%={d6['central_clears_3p2_vbi1_386']}, all_corners_clear="
          f"{d6['all_corners_clear_3p2_vbi1_386']}); prompt-shift sensitivity BINDING="
          f"{d6['prompt_shift_sensitivity_binding_risk_386']} (breakeven {d6['breakeven_prompt_shift_vbi1_tok_386']:g} tok); "
          f"robust pilot off critical path={d6['robust_pilot_off_critical_path_383']} (denken #383)", flush=True)
    print(f"      demand_leaf_delivers={d6['demand_leaf_delivers']} (TREE-gated: ubel #401 / denken #402; "
          f"central-green coverage sizing is PROVENANCE)  pending_tree={d6['demand_leaf_pending_tree']}  "
          f"robust_pending={d6['leaf_robust_pending']}  [{d6['binding_constraint']}]", flush=True)
    d_sup = syn["deliverable_supply_side_base"]
    print("-" * 98, flush=True)
    print("  (D-SUPPLY) SUPPLY base leaf — wirbel #393 CORRECTED base 467.48 (the NOW-BINDING GO axis)", flush=True)
    print(f"      deployable-strict base TODAY {d_sup['realized_deployed_strict_393']:g} < 500 (wirbel #393: "
          f"DECODE attn strict tax {d_sup['decode_attn_strict_tax_pct_393']:g}% over band {d_sup['decode_band_393']}, "
          f"+{d_sup['decode_tax_delta_pp_393']:g}pp vs #378 eval-weighted "
          f"{d_sup['eval_weighted_attn_strict_tax_pct_378']:g}%; supersedes #390's "
          f"{d_sup['realized_deployed_strict_390']:g})", flush=True)
    print(f"      clears_500_today={d_sup['supply_base_clears_500_today']} (gap_to_500 "
          f"{d_sup['deficit_to_500_today_tps']:.2f}, ceiling {d_sup['shippable_ceiling_393']:g}, rebuilds="
          f"{d_sup['n_kernel_rebuilds_strict_500_390']} attn-only); 518.92 eta-axis pin DEFLATED; cb3 HEADLINE +"
          f"{d_sup['cb3_lift_honest']:g} -> {d_sup['base_plus_cb3_honest']:g} ARITHMETIC-ONLY (PPL-DEAD kanna #394: "
          f"supply_alone_clears_500_with_cb3={d_sup['supply_alone_clears_500_with_cb3']}); real PPL-safe lift "
          f"PENDING kanna #403 (held-out worst-seed <= {d_sup['kanna403_heldout_worst_seed_target']:g})", flush=True)
    print(f"      wirbel #384 (RED, BANKED 18:12Z): lm_head FREE={d_sup['wirbel384_lmhead_free']} "
          f"(int4-Marlin, eta_lmhead={d_sup['eta_lmhead_targeted_384']:g}, f_lmhead={d_sup['f_lmhead_384']:g}); "
          f"~150-TPS bf16 lm_head-BI was a by-elimination artifact -> rebuilds={d_sup['n_kernel_rebuilds_strict_500_384']} "
          f"(attn #375 + body #376), NOT 3; 510.01/518.92 shippable ceilings refuted -> wirbel #390 reseated "
          f"(pending={d_sup['wirbel390_shippable_ceiling_pending']})", flush=True)
    print(f"      STRUCTURAL private<=public -> demand_alone_may_be_insufficient="
          f"{d_sup['demand_alone_may_be_insufficient']}", flush=True)
    print(f"      denken #383 (RED, BANKED): demand-alone reaches 500 on floor={d_sup['denken383_reaches_500_on_floor']} "
          f"(confirmed insufficient={d_sup['demand_alone_insufficient_confirmed_383']}); private-on-floor "
          f"{d_sup['private_on_floor_383']:g} -> residual +{d_sup['residual_to_500_on_floor_383']:g} TPS = "
          f"req Δcov +{d_sup['required_dcov_383']:g} ({d_sup['required_dcov_budget_mult_383']:g}x #336 budget)",
          flush=True)
    print(f"      -> supply lift required FIRST +{d_sup['supply_lift_required_first_tps_383']:g} TPS (floor-joint; "
          f"+{d_sup['supply_lift_required_et_only_tps_383']:g} E[T]-only); attn rebuild alone closes gap="
          f"{d_sup['attn_rebuild_alone_closes_supply_gap_383']}; coverage pilot on critical path="
          f"{d_sup['pilot_on_critical_path_383']} (base must clear {d_sup['base_clears_pilot_relevant_band_383']})",
          flush=True)
    print(f"      lawine #372 (GREEN, BANKED) supply lever ALIVE={d_sup['lawine372_supply_lever_alive']}: "
          f"mixed-precision {d_sup['mixed_precision_avg_bpw_372']:g} avg bpw ({d_sup['body_3bit_frac_372']*100:.1f}% "
          f"body @3-bit) PPL {d_sup['mixed_precision_gate_ppl_372']:g}<={PPL_GATE:g} for "
          f"-{d_sup['body_read_reduction_372']*100:.1f}% body read (uniform-3bit died on PPL="
          f"{d_sup['uniform_3bit_died_on_ppl_372']}); realized M=1 TPS pending lawine #388="
          f"{d_sup['lawine388_realized_tps_pending']}", flush=True)
    print(f"      denken#383(reach-on-floor)={d_sup['denken383_input']} wirbel#390/lawine#388(lift_avail)="
          f"{d_sup['supply_lift_available_tps']} (req +{d_sup['supply_lift_required_first_tps_383']:g}) "
          f"-> supply_base_enables_500={d_sup['supply_base_enables_500']}  [{d_sup['binding_constraint']}]",
          flush=True)
    print("-" * 98, flush=True)
    print("  (D8) GAP CEILING-CHECK — ubel #379 (GREEN, BANKED): gap NOT capped by an irreducible floor",
          flush=True)
    print(f"      {d8['public_private_gap_pct']}pp gap = "
          f"{d8['gap_fractions']['acceptance_coverage_addressable']*100:.2f}% acceptance "
          f"(addressable {d8['gap_addressable_pp']:.3f}pp) + "
          f"{d8['gap_fractions']['ctxlen_irreducible']*100:.2f}% ctxlen (irreducible) + 0% outlen + "
          f"0% numerics", flush=True)
    print(f"      numerics tax CANCELS in step-diff (refutes 'numerics is the floor'={d8['refutes_numerics_is_irreducible_floor']}); "
          f"irreducible floor {d8['gap_irreducible_pp_central']:g}% central, corners "
          f"{d8['gap_irreducible_corners']} clear 3.2% knife-edge by >=1.5pp={d8['all_corners_clear_knife_edge']}",
          flush=True)
    print(f"      coverage_target_for_3p2={d8['coverage_target_for_3p2']} (+{d8['delta_cov_ubel379']:.4f}); "
          f"reconciles #377 to <=0.0003={d8['reconciles_within_0p0003']}; slope "
          f"{d8['slope_tps_per_coverage_ubel379']:g} TPS/unit (ubel #382 BANKED private-robust); "
          f"gap_after_max_retrain {d8['gap_after_max_coverage_retrain_pct']:g}% (~3x headroom)", flush=True)
    print(f"      ubel #386 (RESOLVED RED, BANKED): floor-survives-VBI={d8['irreducible_floor_survives_vbi_input_ubel386']} "
          f"-> uncapped_on_live_vbi_stack={d8['uncapped_on_live_vbi_stack']}; the 0.633% off-VBI floor INFLATES "
          f"{d8['floor_inflation_mult_386']:g}x -> {d8['irreducible_floor_vbi1_central_pct_386']:g}% central "
          f"under VBI=1", flush=True)
    print(f"      central still clears 3.2% (+{d8['central_margin_to_3p2_vbi1_pp_386']:g}pp)="
          f"{d8['central_clears_3p2_vbi1_386']} BUT all_corners_clear_3p2_vbi1={d8['all_corners_clear_3p2_vbi1_386']} "
          f"(pessimistic corner {d8['pessimistic_corner_vbi1_pct_386']:g}%, {d8['pessimistic_corner_margin_pp_386']:g}pp); "
          f"breakeven prompt shift HALVES {BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK:g}->{d8['breakeven_prompt_shift_vbi1_tok_386']:g} tok", flush=True)
    print(f"      -> re-derive demand ceiling on the {d8['irreducible_floor_vbi1_central_pct_386']:g}% live floor; "
          f"prompt-shift sensitivity now BINDING={d8['prompt_shift_sensitivity_binding_risk_386']}; ubel #389 to PIN "
          f"the -0.32pp breach (pending={d8['ubel389_pin_breach_pending']})", flush=True)
    print("-" * 98, flush=True)
    print("  (D2c) IDENTITY REACHABILITY — strict-lock compliance, REACHABLE in BOTH #381 branches",
          flush=True)
    print(f"      stark #376 (RED): pinned e2e {dir_['identity_pinned_e2e_stark376']} ~= heuristic "
          f"{dir_['identity_heuristic_e2e_stark376']} (pin doesn't help={dir_['pin_does_not_help_stark376']}); "
          f"residual {dir_['identity_residual_flip_stark376']*100:.2f}% = int4-Marlin body GEMM "
          f"(env-unpatchable={dir_['env_knob_cannot_patch_marlin']})", flush=True)
    print(f"      GEOMETRY caveat: Marlin BIT-EXACT @ {dir_['decode_verify_width_rows']}-row decode-verify "
          f"width; M-variant @ {dir_['prefill_replication_width_rows']}-row prefill-replication (the RED "
          f"geometry). RED is geometry-specific={dir_['red_is_geometry_specific']}", flush=True)
    print(f"      stark #381 (decode-width)={dir_['stark381_decode_input']} -> cost_branch="
          f"{dir_['cost_branch']} (rebuilds={dir_['rebuild_line_items']}); identity_REACHABLE="
          f"{dir_['identity_reachable_env_or_rebuild']} in BOTH branches (#381 sets COST, not GO)", flush=True)
    print("-" * 98, flush=True)
    print("  (D7) PRIVATE-500 GO [HISTORICAL SUB-RESULT, superseded 22:08Z] — AND-PRODUCT: (supply base "
          "ENABLES 500) x (demand closer DELIVERS residual)", flush=True)
    print(f"      go_formula: {d7['go_formula']}", flush=True)
    print(f"      supply_base_enables_500={d7['supply_base_enables_500']}  "
          f"demand_closer_delivers={d7['demand_closer_delivers']}  "
          f"demand_alone_may_be_insufficient={d7['demand_alone_may_be_insufficient']}", flush=True)
    print(f"      identity (folds into supply)={d7['identity_reachable_env_or_rebuild']} "
          f"({d7['identity_cost_branch']})  "
          f"route_a_in_go_product={d7['route_a_supply_side']['in_go_product']} (DEFLATED, excluded)",
          flush=True)
    print(f"      SUPPLY (wirbel #393 LANDED): base={d_sup['supply_base_today_tps']:g} "
          f"(rebuilds={d_sup['n_kernel_rebuilds_strict_500_390']}, gap_to_500={d_sup['deficit_to_500_today_tps']:g}) "
          f"+ cb3 HEADLINE +{d_sup['cb3_lift_honest']:g} -> {d_sup['base_plus_cb3_honest']:g} "
          f"ARITHMETIC-ONLY (PPL-DEAD kanna #394: supply_alone_clears_500_with_cb3="
          f"{d_sup['supply_alone_clears_500_with_cb3']}); real lift PENDING kanna #403", flush=True)
    print(f"      TERMINAL GO (20:19Z REPRICE) = {d7['terminal_go']}  "
          f"supply_leg(kanna #403)={d7['supply_base_enables_500']}  "
          f"demand_leg(tree ubel #401/denken #402)={d7['demand_closer_delivers']}  "
          f"pending_gates={d7['terminal_go_pending_gates']}  flip_gates={d7['reprice_go_flip_gates']}",
          flush=True)
    print(f"      verdict_pending = {d7['verdict_pending']}  GO-gating missing={d7['pending_inputs']}  "
          f"refinements={d7['refinement_inputs']}", flush=True)
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
    d8 = syn["deliverable8_gap_decomposition"]
    d_sup = syn["deliverable_supply_side_base"]
    dir_ = syn["deliverable_identity_reachability"]
    fr = syn["equivalent_tps_frontier"]              # 22:08Z CAPSTONE (max-equivalent-TPS frontier)
    hsr = syn["historical_sub_result"]               # binary >500 verdict (provenance only)
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
            "identity-gate", "lmhead-eta", "stark-363-attn-free", "stark-365-lmhead",
            "demand-side", "coverage-closer", "denken-377", "private-500", "public-private-gap",
            "eta-axis-deflated", "softkd-retrain", "non-iid-coverage",
            # advisor 16:45Z/17:03Z/17:16Z: supply x demand AND-product GO; supply base is now binding
            "supply-x-demand-go", "and-product-go", "go-gating", "supply-honest-base", "wirbel-378",
            "honest-strict-base", "private-le-public", "demand-alone-insufficient",
            "gap-decomposition", "ubel-379-banked", "gap-ceiling-check", "numerics-cancels", "knife-edge",
            "identity-reachability", "stark-376-red", "stark-381-cost-only", "marlin-body-gemm",
            "decode-verify-geometry", "denken-380-banked-two-tier", "ubel-382-slope-banked",
            "demand-central-green", "demand-robust-pending-pilot",
            # advisor 17:53Z cycle-53 merges: denken #383 RED + ubel #386 RED + lawine #372 GREEN
            "denken-383-red-banked", "demand-alone-insufficient-confirmed", "supply-lift-required-17p2",
            "ubel-386-floor-inflates-banked", "floor-inflates-vbi1", "prompt-shift-binding-risk",
            "ubel-389-pin-pending", "lawine-372-supply-lever-green", "mixed-precision-allocation",
            "route-a-deflated",
            # advisor 18:12Z correction: wirbel #384 RED (lm_head FREE int4-Marlin; tax in body #376)
            "wirbel-384-lmhead-free", "lmhead-int4-marlin-not-bf16", "lmhead-bi-by-elimination-refuted",
            "supply-tax-in-body-376",
            # advisor 19:01Z cycle-53 merges (sharpen, not flip): #390 base + #388 lift + #387 anchor + #374 fusion
            "wirbel-390-landed", "corrected-base-471p42", "rebuilds-1-not-2", "body-marlin-byte-exact-m8",
            "stark381-per-gemm-green", "shippable-ceiling-510-reinstated", "ceiling-518-refuted",
            "lawine-388-landed", "cb3-lift-33-honest", "cb3-supply-alone-504", "cb3-m1-not-bw-bound",
            "denken-387-measured-anchor", "drafter-is-mtp-k7", "not-eagle3", "kanna-374-fusion-closed",
            # the now-binding TERMINAL GO-flip gate: cb3-lift deployability pair
            "cb3-deployability-gate", "lawine-391-m8-efficiency-pending", "kanna-394-heldout-ppl-pending",
            "denken-392-composition-sharpen", "supply-enables-500-on-paper", "verdict-held-non-terminal",
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
            # supply-side honest-base leaf (wirbel #378, the now-BINDING GO axis) — pending wirbel #390 / lawine #388
            "supply_lift_available_tps_input_wirbel390_or_lawine388": args.supply_lift_available_tps,
            "demand_reaches_500_on_floor_input_denken383": args.demand_reaches_500_on_floor,
            "supply_lift_required_tps_input_denken383": args.supply_lift_required_tps,
            "honest_strict_base_floor_378": HONEST_STRICT_BASE_FLOOR_378,
            "honest_strict_base_plus_attn_378": HONEST_STRICT_BASE_PLUS_ATTN_378,
            "eta_axis_base_deflated_518": ETA_AXIS_BASE_DEFLATED_518,
            "lmhead_bi_tax_tps_378_refuted": LMHEAD_BI_TAX_TPS_378_REFUTED,
            "attn_rebuild_tps_gain_378": ATTN_REBUILD_TPS_GAIN_378,
            # wirbel #384 (4f32ks1e, RED, BANKED 18:12Z): lm_head FREE int4-Marlin; supply tax in body #376
            "wirbel384_lmhead_free": WIRBEL384_LMHEAD_FREE,
            "eta_lmhead_targeted_384": ETA_LMHEAD_TARGETED_384,
            "f_lmhead_384": F_LMHEAD_384,
            "lmhead_is_int4_marlin_not_bf16_384": LMHEAD_IS_INT4_MARLIN_NOT_BF16_384,
            "lmhead_bi_share_of_vbi_overhead_384": LMHEAD_BI_SHARE_OF_VBI_OVERHEAD_384,
            "lmhead_bi_incremental_share_384": LMHEAD_BI_INCREMENTAL_SHARE_384,
            "n_kernel_rebuilds_strict_500_384": N_KERNEL_REBUILDS_STRICT_500_384,
            "dominant_nonattn_strict_locus_384": DOMINANT_NONATTN_STRICT_LOCUS_384,
            "body_marlin_decode_strict_pending_stark381_384": BODY_MARLIN_DECODE_STRICT_PENDING_STARK381_384,
            # wirbel #390 (reseated 18:12Z): corrected SHIPPABLE strict ceiling (510.01/518.92 refuted)
            "wirbel390_shippable_ceiling_pending": WIRBEL390_SHIPPABLE_CEILING_PENDING,
            "shippable_ceiling_refuted_bf16_premise_390": list(SHIPPABLE_CEILING_REFUTED_BF16_PREMISE_390),
            # denken #383 (t68af2yw, RED, BANKED 17:53Z): demand-alone insufficient on the honest base
            "denken383_reaches_500_on_floor": DENKEN383_REACHES_500_ON_FLOOR,
            "supply_lift_required_first_tps_383": SUPPLY_LIFT_REQUIRED_FIRST_TPS_383,
            "supply_lift_required_et_only_tps_383": SUPPLY_LIFT_REQUIRED_ET_ONLY_TPS_383,
            "private_on_floor_383": PRIVATE_ON_FLOOR_383,
            "residual_to_500_on_floor_383": RESIDUAL_TO_500_ON_FLOOR_383,
            "required_dcov_383": REQUIRED_DCOV_383,
            "required_dcov_budget_mult_383": REQUIRED_DCOV_BUDGET_MULT_383,
            "attn_rebuild_alone_closes_supply_gap_383": ATTN_REBUILD_ALONE_CLOSES_SUPPLY_GAP_383,
            "pilot_on_critical_path_383": PILOT_ON_CRITICAL_PATH_383,
            "base_clears_pilot_relevant_band_383": list(BASE_CLEARS_PILOT_RELEVANT_BAND_383),
            # lawine #372 (mpzfw116, GREEN, BANKED 17:53Z): supply lever ALIVE (pending lawine #388 realized TPS)
            "lawine372_supply_lever_alive": LAWINE372_SUPPLY_LEVER_ALIVE,
            "mixed_precision_avg_bpw_372": MIXED_PRECISION_AVG_BPW_372,
            "body_3bit_frac_372": BODY_3BIT_FRAC_372,
            "mixed_precision_ppl_delta_pct_372": MIXED_PRECISION_PPL_DELTA_PCT_372,
            "mixed_precision_gate_ppl_372": MIXED_PRECISION_GATE_PPL_372,
            "body_read_reduction_372": BODY_READ_REDUCTION_372,
            "mixed_precision_analytic_lift_tps_372": list(MIXED_PRECISION_ANALYTIC_LIFT_TPS_372),
            "uniform_3bit_died_on_ppl_372": UNIFORM_3BIT_DIED_ON_PPL_372,
            "lawine388_realized_tps_pending": LAWINE388_REALIZED_TPS_PENDING,
            "cb3_insample_ppl_margin_372": CB3_INSAMPLE_PPL_MARGIN_372,
            # wirbel #390 (5y64zbjz, LANDED 19:01Z): corrected SHIPPABLE strict base (471.42, rebuilds=1)
            "wirbel390_landed": WIRBEL390_LANDED,
            "realized_deployed_strict_390": REALIZED_DEPLOYED_STRICT_390,
            "shippable_band_390": list(SHIPPABLE_BAND_390),
            "gap_to_500_390": GAP_TO_500_390,
            "supply_alone_closes_500_390": SUPPLY_ALONE_CLOSES_500_390,
            "eta_attn_390": ETA_ATTN_390,
            "n_kernel_rebuilds_strict_500_390": N_KERNEL_REBUILDS_STRICT_500_390,
            "body_marlin_decode_strict_green_390": BODY_MARLIN_DECODE_STRICT_GREEN_390,
            "stark381_decode_identity_per_gemm_green_390": STARK381_DECODE_IDENTITY_PER_GEMM_GREEN_390,
            "shippable_ceiling_510_reinstated_390": SHIPPABLE_CEILING_510_REINSTATED_390,
            "shippable_ceiling_518_still_refuted_390": SHIPPABLE_CEILING_518_STILL_REFUTED_390,
            "deployed_floor_lift_over_378_band_390": DEPLOYED_FLOOR_LIFT_OVER_378_BAND_390,
            # lawine #388 (g5lfdpgw, LANDED 19:01Z): realized cb3 body-allocation supply LIFT
            "lawine388_landed": LAWINE388_LANDED,
            "cb3_lift_honest_tps_388": CB3_LIFT_HONEST_TPS_388,
            "cb3_lift_realistic_tps_388": CB3_LIFT_REALISTIC_TPS_388,
            "cb3_lift_mult_388": CB3_LIFT_MULT_388,
            "cb3_closes_383_supply_gap_floor_388": CB3_CLOSES_383_SUPPLY_GAP_FLOOR_388,
            "cb3_m1_is_bw_bound_388": CB3_M1_IS_BW_BOUND_388,
            "cb3_m1_hbm_eff_388": CB3_M1_HBM_EFF_388,
            "cb3_m1_roofline_mult_388": CB3_M1_ROOFLINE_MULT_388,
            "cb3_realized_frac_of_roofline_388": CB3_REALIZED_FRAC_OF_ROOFLINE_388,
            "cb3_draft_frac_lumped_388": CB3_DRAFT_FRAC_LUMPED_388,
            # denken #387 (z8osvif8, LANDED 19:01Z): demand anchor MEASURED + MTP K=7 premise correction
            "denken387_landed": DENKEN387_LANDED,
            "measured_top4_coverage_387": MEASURED_TOP4_COVERAGE_387,
            "coverage_anchor_gap_387": COVERAGE_ANCHOR_GAP_387,
            "required_delta_floor_measured_387": REQUIRED_DELTA_FLOOR_MEASURED_387,
            "denken383_red_robust_to_measured_anchor_387": DENKEN383_RED_ROBUST_TO_MEASURED_ANCHOR_387,
            "deployed_drafter_mtp_k_387": DEPLOYED_DRAFTER_MTP_K_387,
            "drafter_is_mtp_not_eagle3_387": DRAFTER_IS_MTP_NOT_EAGLE3_387,
            "demand_ladder_label_387": DEMAND_LADDER_LABEL_387,
            # kanna #374 (djia6icp, LANDED 19:01Z): fusion lever CLOSED (Route-A stays excluded)
            "kanna374_fusion_lever_closed": KANNA374_FUSION_LEVER_CLOSED,
            "fusion_byte_exact_pinnable_374": FUSION_BYTE_EXACT_PINNABLE_374,
            "route_a_stays_excluded_374": ROUTE_A_STAYS_EXCLUDED_374,
            # TERMINAL GO-flip gate (advisor 20:19Z REPRICE): kanna #403 (PPL-safe conservative-k supply
            # re-cost) AND >=1 of {ubel #401, denken #402} (tree net-supply). Supersedes the 19:25Z pair
            # (kanna #394 RED made the +32.65 headline PPL-dead; denken #396/ubel #399 RED closed the escapes).
            "kanna403_ppl_safe_supply_input": args.kanna403_ppl_safe_supply,
            "ubel401_tree_coverage_ceiling_input": args.ubel401_tree_coverage_ceiling,
            "denken402_tree_net_supply_input": args.denken402_tree_net_supply,
            "lawine391_m8_realistic_green_landed": LAWINE391_LANDED,
            "denken392_authoritative_composed_landed": DENKEN392_LANDED,
            "reprice_2019z_go_flip_gates": list(REPRICE_2019Z_GO_FLIP_GATES),
            "reprice_2019z_tree_net_supply_probes": list(REPRICE_2019Z_TREE_NET_SUPPLY_PROBES),
            # demand-side residual leaf (denken #377 sized; #380 two-tier; #382 slope GREEN)
            "robust_coverage_pilot_input": args.robust_pilot,
            "irreducible_floor_survives_vbi_input_ubel386": args.irreducible_floor_survives_vbi,
            "gap_coverage_addressable_pp": args.gap_addressable_pp,
            "stark381_decode_identity_input": args.stark381_decode,
            "demand_closer_central_c_380": DEMAND_CLOSER_CENTRAL_C,
            "demand_closer_robust_c_380": DEMAND_CLOSER_ROBUST_C,
            "p_deliver_central_defensible_380": P_DELIVER_CENTRAL_DEFENSIBLE_380,
            "p_deliver_robust_defensible_380": P_DELIVER_ROBUST_DEFENSIBLE_380,
            "kappa_breakeven_380": KAPPA_BREAKEVEN_380,
            "kappa_margin_380": KAPPA_MARGIN_380,
            "deliverable_finetune_mean_380": DELIVERABLE_FINETUNE_MEAN_380,
            "deliverable_finetune_std_380": DELIVERABLE_FINETUNE_STD_380,
            "coverage_lift_pilot_gpu_hr": COVERAGE_LIFT_PILOT_GPU_HR,
            "slope_flattening_ratio_382": SLOPE_FLATTENING_RATIO_382,
            "slope_tps_per_coverage_private_382": SLOPE_TPS_PER_COVERAGE_PRIVATE_382,
            "demand_conservative_target_382": DEMAND_CONSERVATIVE_TARGET_382,
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
            # ubel #379 (GREEN, BANKED) gap-decomposition ceiling-check
            "gap_acceptance_frac_ubel379": GAP_ACCEPTANCE_FRAC_UBEL379,
            "gap_ctxlen_frac_ubel379": GAP_CTXLEN_FRAC_UBEL379,
            "gap_addressable_pp_ubel379": GAP_ADDRESSABLE_PP_UBEL379,
            "gap_irreducible_pp_central_ubel379": GAP_IRREDUCIBLE_PP_CENTRAL_UBEL379,
            "knife_edge_gap_pct": KNIFE_EDGE_GAP_PCT,
            "knife_edge_min_margin_pp": KNIFE_EDGE_MIN_MARGIN_PP,
            "coverage_target_for_3p2_ubel379": COVERAGE_TARGET_FOR_3P2_UBEL379,
            "baseline_cov_336": BASELINE_COV_336,
            "slope_tps_per_coverage_ubel379": SLOPE_TPS_PER_COVERAGE_UBEL379,
            "gap_after_max_coverage_retrain_pct_ubel379": GAP_AFTER_MAX_COVERAGE_RETRAIN_PCT_UBEL379,
            "private_prompt_shift_breakeven_tok": PRIVATE_PROMPT_SHIFT_BREAKEVEN_TOK,
            # ubel #386 (xxzujn7a, RED, BANKED 17:53Z): irreducible floor INFLATES under VBI=1
            "irreducible_floor_vbi1_central_386": IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386,
            "floor_inflation_mult_386": FLOOR_INFLATION_MULT_386,
            "all_corners_clear_3p2_vbi1_386": ALL_CORNERS_CLEAR_3P2_VBI1_386,
            "central_clears_3p2_vbi1_386": CENTRAL_CLEARS_3P2_VBI1_386,
            "central_margin_to_3p2_vbi1_pp_386": CENTRAL_MARGIN_TO_3P2_VBI1_PP_386,
            "pessimistic_corner_vbi1_pct_386": PESSIMISTIC_CORNER_VBI1_PCT_386,
            "pessimistic_corner_margin_pp_386": PESSIMISTIC_CORNER_MARGIN_PP_386,
            "breakeven_prompt_shift_vbi1_tok_386": BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386,
            "breakeven_prompt_shift_pre_vbi_tok": BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK,
            "ubel389_pin_breach_pending": UBEL389_PIN_BREACH_PENDING,
            # stark #376 (RED) + stark #381 (cost-only) identity reachability
            "identity_pinned_e2e_stark376": IDENTITY_PINNED_E2E_STARK376,
            "identity_heuristic_e2e_stark376": IDENTITY_HEURISTIC_E2E_STARK376,
            "identity_residual_flip_stark376": IDENTITY_RESIDUAL_FLIP_STARK376,
            "decode_verify_width_rows": DECODE_VERIFY_WIDTH_ROWS,
            "prefill_replication_width_rows": PREFILL_REPLICATION_WIDTH_ROWS,
            "go_formula": d7["go_formula"],
            "wandb_group": args.wandb_group,
            "source_runs": (
                "lawine#196, wirbel#326, wirbel#354, denken#332(y5cl0ena), "
                "denken#356(ceiling-curve), denken#344, kasane#349, stark#363(a0oi2esq), stark#365, "
                "denken#377(030uc5mk demand-side sized), denken#380(00oijpwg two-tier deliverability "
                "BANKED YELLOW), ubel#379(5kpb73tb gap-split BANKED GREEN), ubel#382(bn0v5rqr slope "
                "private-robust BANKED GREEN), wirbel#378(gghmgtk9 honest supply base), "
                "stark#376(ipe3ofie identity RED), stark#381(decode-width identity cost-only), "
                "#336(budget), #289(per-position), #339(deliverability dist), #373/#375(eta-axis deflated). "
                "CYCLE-53 MERGES (17:53Z): denken#383(t68af2yw RED, demand-alone insufficient on honest base "
                "-> +17.2-TPS supply lift required first), ubel#386(xxzujn7a RED, irreducible floor inflates "
                "2.07x -> 1.310% central under VBI=1), lawine#372(mpzfw116 GREEN, mixed-precision supply lever "
                "ALIVE -21.5% body read). CORRECTION (18:12Z): wirbel#384(4f32ks1e RED, lm_head FREE int4-Marlin "
                "eta=0 -> lm_head-BI supply lever refuted; supply tax in body #376). CYCLE-53 LANDED (19:01Z, "
                "sharpen-not-flip): wirbel#390(5y64zbjz LANDED, corrected base 471.42, rebuilds 2->1 [body "
                "int4-Marlin byte-exact @ M=8 -> per-GEMM stark#381 GREEN], 510.01 reinstated/only 518.92 "
                "refuted), lawine#388(g5lfdpgw LANDED, realized cb3 lift +33 honest/+38.34 realistic -> 504.42 "
                "clears 500 SUPPLY-ALONE on paper; m1_is_bw_bound=False), denken#387(z8osvif8 LANDED, MEASURED "
                "anchor 0.89027 -> #383 robust; drafter is MTP K=7 NOT EAGLE-3), kanna#374(djia6icp LANDED, "
                "fusion lever closed). HELD GO (now on cb3 DEPLOYABILITY, non-terminal): lawine#391(M=8 "
                "efficiency) + kanna#394(held-out PPL) GO-flip pair; denken#392(authoritative cb3 composition) "
                "sharpens. REFINEMENTS: robust coverage pilot (off critical path) + ubel#389(pin VBI=1 floor breach)"
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
        # PRIMARY (0-GPU self-test integrity gate)
        "strict_500_composite_reachability_self_test_passes": int(bool(
            st["strict_500_composite_reachability_self_test_passes"])),
        # ====== 22:08Z CAPSTONE: max-equivalent-TPS frontier (22:26Z banks lawine #417 + land #414) ======
        "max_equivalent_tps_measured": fr["max_equivalent_tps_measured"],          # 467.48 (MEASURED, identity 1.0)
        "max_equivalent_tps_modeled": fr["max_equivalent_tps_modeled"],            # 492.08 (MODELED bracket lower; BEATS 481.53)
        "max_equivalent_tps_modeled_bracket_lo": fr["max_equivalent_tps_modeled_bracket"][0],   # 492.08 (lawine #417)
        "max_equivalent_tps_modeled_bracket_hi": fr["max_equivalent_tps_modeled_bracket"][1],   # 494.08 (lawine #417)
        "max_equivalent_tps_modeled_point": fr["max_equivalent_tps_modeled_point"],  # 494.53 (denken #413 additive point)
        "frontier_modeled_beats_deployed_nonequiv": int(bool(fr["modeled_beats_deployed_nonequiv"])),  # 1 (492.08>481.53)
        "frontier_cb3_stack_banked_additive_417": int(bool(fr["cb3_stack_banked_additive_417"])),  # 1 (lawine #417)
        "frontier_naive_stacked_ceiling_tps": fr["naive_stacked_ceiling_tps"],     # 494.53 (recompute+cb3 additive point)
        "frontier_lmhead_truncation_free_self_referential": int(bool(fr["lmhead_truncation_free_self_referential"])),  # 1
        "frontier_truevocab_lmhead_tps_cost_contingency": fr["truevocab_lmhead_tps_cost_contingency"],  # 54.07 (not required)
        "frontier_identity": fr["frontier_identity"],                              # 1.0
        "frontier_equivalence_tax_vs_deployed_tps": fr["equivalence_tax_vs_deployed_nonequiv_tps"],  # 481.53-467.48
        "frontier_deployability_banked_green": int(bool(fr["deployability_banked_green"])),  # 1 (lawine #417)
        "frontier_n_pending_feeders": len(fr["pending_feeders"]),
        "frontier_n_banked_feeders": len(fr["banked_feeders"]),
        "frontier_all_feeders_resolved": int(bool(fr["all_feeders_resolved"])),
        "frontier_selective_recompute_measured": int(bool(fr["selective_recompute_measured"])),
        "frontier_cb3_additivity_gap_resolved": int(bool(fr["cb3_additivity_gap_resolved"])),
        "frontier_floor_reduction_resolved": int(bool(fr["floor_reduction_resolved"])),
        "frontier_tree_leg_closed": int(bool(fr["tree_leg_closed"])),
        "deployed_fastpath_tps": fr["deployed_fastpath_off_ladder"]["tps"],        # 481.53
        "deployed_fastpath_served_identity": fr["deployed_fastpath_off_ladder"]["served_identity"],  # 0.9966
        "deployed_fastpath_is_equivalent": int(bool(fr["deployed_fastpath_off_ladder"]["is_equivalent"])),  # 0
        # HISTORICAL sub-result (binary >500 GO, superseded 22:08Z)
        "historical_frozen_run_34tv4krw_strict_500_reachable": int(
            hsr["frozen_run_34tv4krw_strict_500_reachable_via_known_levers"]),   # 0 (RED) — the preserved record
        "historical_live_leg_strict_500_reachable": (
            -1 if hsr["live_leg_strict_500_reachable_via_known_levers"] is None
            else int(bool(hsr["live_leg_strict_500_reachable_via_known_levers"]))),
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
        # supply x demand AND-product GO factors (advisor 16:45Z/17:03Z/17:16Z)
        "identity_reachable_env_or_rebuild": int(bool(d7["identity_reachable_env_or_rebuild"])),
        "identity_cost_branch_pending": int(bool(d7["identity_cost_branch_pending"])),
        "route_a_in_go_product": int(bool(d7["route_a_supply_side"]["in_go_product"])),
        "demand_alone_may_be_insufficient": int(bool(d7["demand_alone_may_be_insufficient"])),
        "n_go_gating_pending_inputs": len(d7["pending_inputs"]),
        "n_refinement_inputs": len(d7["refinement_inputs"]),
        # supply-side honest-base leaf (wirbel #378, the now-BINDING GO axis; pending wirbel #390 / lawine #388)
        "supply_base_today_tps": d_sup["supply_base_today_tps"],
        "supply_base_clears_500_today": int(bool(d_sup["supply_base_clears_500_today"])),
        "deficit_to_500_today_tps": d_sup["deficit_to_500_today_tps"],
        "honest_strict_base_floor_378": d_sup["honest_strict_base_floor"],
        "honest_strict_base_plus_attn_378": d_sup["honest_strict_base_plus_attn"],
        "eta_axis_base_deflated_518": d_sup["eta_axis_base_deflated_518"],
        "supply_pending": int(bool(d_sup["supply_pending"])),
        "denken383_pending": int(bool(d_sup["denken383_pending"])),
        "supply_lift_measured_pending": int(bool(d_sup["supply_lift_measured_pending"])),
        "supply_lift_pending_kanna403": int(bool(d_sup["supply_lift_pending_kanna403"])),
        # wirbel #384 (RED, BANKED 18:12Z): lm_head FREE int4-Marlin; supply tax in body #376; rebuilds 2 not 3
        "wirbel384_lmhead_free": int(bool(d_sup["wirbel384_lmhead_free"])),
        "eta_lmhead_targeted_384": d_sup["eta_lmhead_targeted_384"],
        "f_lmhead_384": d_sup["f_lmhead_384"],
        "lmhead_bi_share_of_vbi_overhead_384": d_sup["lmhead_bi_share_of_vbi_overhead_384"],
        "n_kernel_rebuilds_strict_500_384": d_sup["n_kernel_rebuilds_strict_500_384"],
        "body_marlin_decode_strict_pending_stark381_384": int(bool(d_sup["body_marlin_decode_strict_pending_stark381_384"])),
        # wirbel #390 (LANDED 19:01Z): corrected SHIPPABLE strict base 471.42 (rebuilds=1; 510.01 reinstated)
        "wirbel390_landed": int(bool(d_sup["wirbel390_landed"])),
        "wirbel390_shippable_ceiling_pending": int(bool(d_sup["wirbel390_shippable_ceiling_pending"])),
        "realized_deployed_strict_390": d_sup["realized_deployed_strict_390"],
        "gap_to_500_390": d_sup["gap_to_500_390"],
        "eta_attn_390": d_sup["eta_attn_390"],
        "n_kernel_rebuilds_strict_500_390": d_sup["n_kernel_rebuilds_strict_500_390"],
        "body_marlin_decode_strict_green_390": int(bool(d_sup["body_marlin_decode_strict_green_390"])),
        "stark381_decode_identity_per_gemm_green_390": int(bool(d_sup["stark381_decode_identity_per_gemm_green_390"])),
        "shippable_ceiling_510_reinstated_390": d_sup["shippable_ceiling_510_reinstated_390"],
        "shippable_ceiling_518_still_refuted_390": d_sup["shippable_ceiling_518_still_refuted_390"],
        # lawine #388 (LANDED 19:01Z): realized cb3 supply lift (+33 honest / +38.34 realistic)
        "lawine388_landed": int(bool(d_sup["lawine388_landed"])),
        "cb3_lift_honest_tps_388": d_sup["cb3_lift_honest_tps_388"],
        "cb3_lift_realistic_tps_388": d_sup["cb3_lift_realistic_tps_388"],
        "base_plus_cb3_honest": d_sup["base_plus_cb3_honest"],
        "base_plus_cb3_realistic": d_sup["base_plus_cb3_realistic"],
        "supply_alone_clears_500_with_cb3": int(bool(d_sup["supply_alone_clears_500_with_cb3"])),
        "cb3_closes_383_supply_gap_floor_388": int(bool(d_sup["cb3_closes_383_supply_gap_floor_388"])),
        "cb3_m1_is_bw_bound_388": int(bool(d_sup["cb3_m1_is_bw_bound_388"])),
        "cb3_m1_hbm_eff_388": d_sup["cb3_m1_hbm_eff_388"],
        "supply_lift_data_landed": int(bool(d_sup["supply_lift_data_landed"])),
        # denken #387 (LANDED): demand anchor MEASURED 0.89027 + drafter is MTP K=7 (NOT EAGLE-3)
        "measured_top4_coverage_387": d_sup["measured_top4_coverage_387"],
        "coverage_anchor_gap_387": d_sup["coverage_anchor_gap_387"],
        "required_delta_floor_measured_387": d_sup["required_delta_floor_measured_387"],
        "deployed_drafter_mtp_k_387": d_sup["deployed_drafter_mtp_k_387"],
        "drafter_is_mtp_not_eagle3_387": int(bool(d_sup["drafter_is_mtp_not_eagle3_387"])),
        # kanna #374 (LANDED): fusion lever CLOSED (Route-A stays excluded)
        "kanna374_fusion_lever_closed": int(bool(d_sup["kanna374_fusion_lever_closed"])),
        "route_a_stays_excluded_374": int(bool(d_sup["route_a_stays_excluded_374"])),
        # TERMINAL GO-flip gate (advisor 20:19Z REPRICE): kanna #403 (supply leg) AND tree (ubel #401/denken #402)
        "terminal_go": d7["terminal_go"],                                       # green / red / pending
        "terminal_go_confirmed": int(bool(d7["terminal_go_confirmed"])),
        "terminal_go_blocked": int(bool(d7["terminal_go_blocked"])),
        "terminal_go_pending": int(bool(d7["terminal_go_pending"])),
        "n_terminal_go_pending_gates": len(d7["terminal_go_pending_gates"]),
        "tree_net_supply_green": int(bool(d7["tree_net_supply_green"])),
        "tree_net_supply_both_red": int(bool(d7["tree_net_supply_both_red"])),
        "tree_net_supply_pending": int(bool(d7["tree_net_supply_pending"])),
        "superseded_1925z_pair": int(bool(d7["superseded_1925z_pair"])),
        # denken #383 (RED, BANKED 17:53Z): demand-alone insufficient on the honest base
        "demand_alone_insufficient_confirmed_383": int(bool(d_sup["demand_alone_insufficient_confirmed_383"])),
        "supply_lift_required_first_tps_383": d_sup["supply_lift_required_first_tps_383"],
        "supply_lift_required_et_only_tps_383": d_sup["supply_lift_required_et_only_tps_383"],
        "private_on_floor_383": d_sup["private_on_floor_383"],
        "residual_to_500_on_floor_383": d_sup["residual_to_500_on_floor_383"],
        "required_dcov_383": d_sup["required_dcov_383"],
        "required_dcov_budget_mult_383": d_sup["required_dcov_budget_mult_383"],
        "attn_rebuild_alone_closes_supply_gap_383": int(bool(d_sup["attn_rebuild_alone_closes_supply_gap_383"])),
        "pilot_on_critical_path_383": int(bool(d_sup["pilot_on_critical_path_383"])),
        # lawine #372 (GREEN, BANKED 17:53Z): supply lever ALIVE (pending lawine #388 realized TPS)
        "lawine372_supply_lever_alive": int(bool(d_sup["lawine372_supply_lever_alive"])),
        "mixed_precision_avg_bpw_372": d_sup["mixed_precision_avg_bpw_372"],
        "body_3bit_frac_372": d_sup["body_3bit_frac_372"],
        "mixed_precision_gate_ppl_372": d_sup["mixed_precision_gate_ppl_372"],
        "body_read_reduction_372": d_sup["body_read_reduction_372"],
        "uniform_3bit_died_on_ppl_372": int(bool(d_sup["uniform_3bit_died_on_ppl_372"])),
        "lawine388_realized_tps_pending": int(bool(d_sup["lawine388_realized_tps_pending"])),
        # demand-side residual leaf (denken #377 sized; #380 two-tier BANKED; #382 slope GREEN)
        "eta_axis_deflated": int(bool(d7["eta_axis_deflated"])),
        "demand_leaf_delivers": int(bool(d6["demand_leaf_delivers"])),
        "demand_leaf_robust_pending": int(bool(d6["leaf_robust_pending"])),
        "demand_central_green": int(bool(d6["deliver_central_green"])),
        "demand_robust_modeled": int(bool(d6["deliver_robust_modeled"])),
        "demand_closer_central_c": d6["demand_closer_central_c"],
        "demand_closer_robust_c": d6["demand_closer_robust_c"],
        "p_deliver_central_defensible_380": d6["p_deliver_central_defensible"],
        "p_deliver_robust_defensible_380": d6["p_deliver_robust_defensible"],
        "kappa_breakeven_380": d6["kappa_breakeven_380"],
        "kappa_margin_380": d6["kappa_margin_380"],
        "kappa_axis_robust_380": int(bool(d6["kappa_axis_robust"])),
        "demand_conservative_target_382": d6["demand_conservative_target_382"],
        "demand_budget_frac_conservative_382": d6["demand_budget_frac_conservative_382"],
        "demand_budget_frac_central_382": d6["demand_budget_frac_central_382"],
        "slope_is_private_robust_382": int(bool(d6["slope_is_private_robust"])),
        "slope_tps_per_coverage_private_382": d6["slope_tps_per_coverage_private_382"],
        "slope_flattening_ratio_382": d6["slope_flattening_ratio_382"],
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
        "gap_channel_live_demand_leaf": int(bool(d6["gap_channel_live"])),
        # ubel #386 (RED, BANKED 17:53Z): irreducible floor INFLATES under VBI=1 (demand leaf carries it)
        "irreducible_floor_vbi_pending_ubel386": int(bool(d6["irreducible_floor_vbi_pending_ubel386"])),
        "irreducible_floor_inflates_vbi_386": int(bool(d6["irreducible_floor_inflates_vbi_386"])),
        "irreducible_floor_vbi1_central_pct_386": d6["irreducible_floor_vbi1_central_pct_386"],
        "floor_inflation_mult_386": d8["floor_inflation_mult_386"],
        "central_clears_3p2_vbi1_386": int(bool(d6["central_clears_3p2_vbi1_386"])),
        "central_margin_to_3p2_vbi1_pp_386": d8["central_margin_to_3p2_vbi1_pp_386"],
        "all_corners_clear_3p2_vbi1_386": int(bool(d8["all_corners_clear_3p2_vbi1_386"])),
        "pessimistic_corner_vbi1_pct_386": d8["pessimistic_corner_vbi1_pct_386"],
        "pessimistic_corner_margin_pp_386": d8["pessimistic_corner_margin_pp_386"],
        "breakeven_prompt_shift_vbi1_tok_386": d6["breakeven_prompt_shift_vbi1_tok_386"],
        "prompt_shift_sensitivity_binding_risk_386": int(bool(d6["prompt_shift_sensitivity_binding_risk_386"])),
        "uncapped_on_live_vbi_stack_386": int(bool(d8["uncapped_on_live_vbi_stack"])),
        "ubel389_pin_breach_pending": int(bool(d8["ubel389_pin_breach_pending"])),
        "robust_pilot_off_critical_path_383": int(bool(d6["robust_pilot_off_critical_path_383"])),
        # ubel #379 (GREEN, BANKED) gap-decomposition ceiling-check — always logged
        "gap_acceptance_frac_ubel379": d8["gap_fractions"]["acceptance_coverage_addressable"],
        "gap_ctxlen_frac_ubel379": d8["gap_fractions"]["ctxlen_irreducible"],
        "gap_fractions_sum": d8["gap_fractions_sum"],
        "gap_addressable_pp_ubel379": d8["gap_addressable_pp"],
        "gap_irreducible_pp_central_ubel379": d8["gap_irreducible_pp_central"],
        "gap_channel_live": int(bool(d8["gap_channel_live"])),
        "closer_not_capped_by_irreducible_floor": int(bool(d8["closer_not_capped_by_irreducible_floor"])),
        "all_corners_clear_knife_edge": int(bool(d8["all_corners_clear_knife_edge"])),
        "numerics_tax_cancels_in_step_diff": int(bool(d8["numerics_tax_cancels_in_step_diff"])),
        "coverage_target_for_3p2_ubel379": d8["coverage_target_for_3p2"],
        "delta_cov_ubel379": d8["delta_cov_ubel379"],
        "reconcile_delta_vs_denken377": d8["reconcile_delta_vs_denken377"],
        "reconciles_within_0p0003": int(bool(d8["reconciles_within_0p0003"])),
        "slope_tps_per_coverage_ubel379": d8["slope_tps_per_coverage_ubel379"],
        "gap_after_max_coverage_retrain_pct_ubel379": d8["gap_after_max_coverage_retrain_pct"],
        "private_prompt_shift_breakeven_tok": d8["private_prompt_shift_breakeven_tok"],
        # stark #376 (RED) + stark #381 (cost-only) identity reachability — always logged
        "identity_pinned_e2e_stark376": dir_["identity_pinned_e2e_stark376"],
        "identity_heuristic_e2e_stark376": dir_["identity_heuristic_e2e_stark376"],
        "identity_residual_flip_stark376": dir_["identity_residual_flip_stark376"],
        "pin_does_not_help_stark376": int(bool(dir_["pin_does_not_help_stark376"])),
        "residual_is_int4_marlin_body_gemm": int(bool(dir_["residual_is_int4_marlin_body_gemm"])),
        "marlin_bit_exact_at_decode_width": int(bool(dir_["marlin_bit_exact_at_decode_width"])),
        "red_is_geometry_specific": int(bool(dir_["red_is_geometry_specific"])),
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
    if d7["demand_closer_delivers"] is not None:
        summary["demand_closer_delivers"] = int(bool(d7["demand_closer_delivers"]))
    if d7["identity_rebuild_line_items"] is not None:
        summary["identity_rebuild_line_items"] = int(d7["identity_rebuild_line_items"])
    # supply leaf (the NOW-BINDING GO axis) resolves on a measured supply lift >= +17.2 TPS from
    # wirbel #390 (corrected SHIPPABLE strict ceiling) OR lawine #388 (realized TPS of #372's body allocation)
    if d_sup["supply_base_enables_500"] is not None:
        summary["supply_base_enables_500"] = int(bool(d_sup["supply_base_enables_500"]))
    # demand robust tier resolves only on the ~25-A10G-GPU-hr pilot
    if d6["deliver_robust_resolved"] is not None:
        summary["demand_robust_delivers"] = int(bool(d6["deliver_robust_resolved"]))
    if d4["strict_500_reachable_via_known_levers"] is not None:
        summary["route_a_supply_side_reachable"] = int(bool(
            d4["strict_500_reachable_via_known_levers"]))
        summary["measured_ppl_at_b_star"] = d4["measured_ppl_at_b_star"]
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
    ap.add_argument("--demand-reaches-500-on-deployable-floor", dest="demand_reaches_500_on_floor",
                    choices=["pending", "yes", "no"], default=DENKEN383_REACHES_500_ON_FLOOR,
                    help="denken #383 honest re-price (BANKED RED 17:53Z, default 'no'; denken #387 confirmed "
                         "on the MEASURED 0.89027 anchor): the demand closer does NOT reach 500 on the "
                         "corrected deployable-strict base (471.42 today, wirbel #390) -> a supply lift is "
                         "required first (see --supply-lift-required-tps). Override to 'pending' only to "
                         "model the pre-#383 unresolved state.")
    ap.add_argument("--supply-lift-required-tps", dest="supply_lift_required_tps", type=float,
                    default=SUPPLY_LIFT_REQUIRED_FIRST_TPS_383,
                    help="denken #383 (BANKED, default +17.2 floor-joint): TPS of supply lift the honest "
                         "base needs before demand can close (meaningful when "
                         "--demand-reaches-500-on-deployable-floor=no).")
    ap.add_argument("--supply-lift-available-tps", dest="supply_lift_available_tps", type=float,
                    default=None,
                    help="The PPL-SAFE conservative-k supply lift available to close the 32.52-TPS deployable-"
                         "strict deficit on the corrected 467.48 base (wirbel #393). The +32.65 cb3 HEADLINE "
                         "lift is PPL-DEAD (kanna #394 RED), so this value is consumed ONLY when "
                         "--kanna403-ppl-safe-supply=green (the conservative-k re-cost): green + (lift >= "
                         "required) -> supply leg TRUE. NOTE: even when the supply leg resolves TRUE, the "
                         "TERMINAL GO stays HELD on the DEMAND-leg tree net-supply (>=1 of "
                         "--ubel401-tree-coverage-ceiling / --denken402-tree-net-supply). Omit -> SUPPLY leg "
                         "value unresolved (PENDING under kanna #403).")
    ap.add_argument("--robust-coverage-pilot", dest="robust_pilot",
                    choices=["pending", "delivers", "fails"], default="pending",
                    help="denken #380 tier-2 ~25-A10G-GPU-hr coverage-lift pilot that hardens the ROBUST "
                         "demand tier (c>=0.9010). COST/REFINEMENT-only: the central tier is GREEN now, "
                         "so this does NOT gate the GO (omit/pending -> robust tier stays modeled-only).")
    ap.add_argument("--irreducible-floor-survives-vbi", dest="irreducible_floor_survives_vbi",
                    choices=["pending", "survives", "inflates"], default=UBEL386_FLOOR_SURVIVES_VBI,
                    help="ubel #386 REFINEMENT (BANKED RED 17:53Z, default 'inflates'): ubel #379's "
                         "0.633%% irreducible gap floor does NOT survive the VBI=1 un-packed-attention "
                         "regime -> it inflates 2.07x to 1.310%% central (central still clears 3.2%%, but "
                         "all corners no longer clear; prompt-shift sensitivity becomes binding). Override "
                         "to 'pending' only to model the pre-#386 unresolved state.")
    ap.add_argument("--gap-coverage-addressable-pp", dest="gap_addressable_pp", type=float, default=None,
                    help="ubel #379 pp of the 4.295pp public->private gap that is coverage-addressable. "
                         "BANKED GREEN (omit -> use ubel #379's banked ~3.6615pp). Override only to "
                         "stress-test; do not re-derive it.")
    ap.add_argument("--stark381-decode-identity", dest="stark381_decode",
                    choices=["pending", "green", "red"], default="pending",
                    help="stark #381 decode-width (8-row) e2e identity: green -> env-reachable@decode "
                         "(1 rebuild); red -> Marlin-rebuild-gated (2 rebuilds). COST-only: identity is "
                         "REACHABLE in both branches; this does NOT gate the GO. wirbel #390 (LANDED) "
                         "pre-answered the PER-GEMM M=8 question GREEN (body int4-Marlin byte-exact); pass "
                         "'green' to bank the 1-rebuild ledger (e2e stark #381 stays an independent confirm).")
    ap.add_argument("--kanna403-ppl-safe-supply", dest="kanna403_ppl_safe_supply",
                    choices=["pending", "green", "red"], default="pending",
                    help="kanna #403 — the SUPPLY leg of the 20:19Z RE-PRICED terminal GO: the PPL-SAFE "
                         "conservative-k cb3 supply re-cost. The +32.65 cb3 HEADLINE lift is PPL-DEAD (kanna "
                         "#394 RED: held-out worst-seed 2.4223 + OOD ShareGPT 2.4270 breach 2.42 at k=243-246); "
                         "k=232 still clears ~2.39 held-out, so cb3 is deployable at a conservative k at a "
                         "smaller UN-COSTED lift. green -> a PPL-safe conservative-k lift (largest k with "
                         "held-out worst-seed <= 2.41) clears the required supply lift (resolve its value via "
                         "--supply-lift-available-tps); red -> NO PPL-safe lift clears -> supply reverts to the "
                         "467.48 base (<500); pending -> SUPPLY leg HELD None.")
    ap.add_argument("--ubel401-tree-coverage-ceiling", dest="ubel401_tree_coverage_ceiling",
                    choices=["pending", "green", "red"], default="pending",
                    help="ubel #401 — a DEMAND-leg tree net-supply probe (20:19Z): does the locked top-8/16 "
                         "tree coverage ceiling NET the demand d-cov (sizing the +0.1286 top-1->top-4 prize)? "
                         "green -> the tree supplies the d-cov; red -> it does not. The DEMAND leg is green iff "
                         "this OR --denken402 is green (demand-alone is closed: denken #396/ubel #399 RED).")
    ap.add_argument("--denken402-tree-net-supply", dest="denken402_tree_net_supply",
                    choices=["pending", "green", "red"], default="pending",
                    help="denken #402 — the other DEMAND-leg tree net-supply probe (20:19Z): does the tree NET "
                         "d-cov AFTER its verify-M step-time tax on the 467.48 base? green -> net-positive tree "
                         "supply; red -> the verify-M tax eats the coverage gain. Pairs (OR) with --ubel401.")
    # 22:08Z RE-POINT — the max-equivalent-TPS frontier ladder feeders (#407/#357).
    ap.add_argument("--selective-recompute-measured-tps", dest="selective_recompute_measured_tps",
                    type=float, default=None,
                    help="stark #412 — MEASURED equivalent TPS for the selective-higher-precision-recompute "
                         "config (#397: fast attention everywhere + recompute the ~23.6%% near-tie <=eps-flagged "
                         "steps -> served identity 1.0 BY CONSTRUCTION). Omit -> use the MODELED 478.93 "
                         "(= 481.53 - 2.6 tax; within band [476,479]) and tag node 1 MODELED.")
    ap.add_argument("--cb3-additivity-gap-tps", dest="cb3_additivity_gap_tps", type=float, default=None,
                    help="kanna #416 — MEASURED additivity gap (TPS) that TIGHTENS the cb3 stack: cb3's body-read "
                         "shrink partly OVERLAPS the recompute re-reads on the flagged steps. cb3 is ALREADY "
                         "additive-CONFIRMED (lawine #417) and BANKED; omit -> headline stays the banked bracket "
                         "[492.08, 494.08], the gap only narrows it.")
    ap.add_argument("--floor-reduction-tps", dest="floor_reduction_tps", type=float, default=None,
                    help="wirbel #415 — MEASURED reduction (TPS) of the 146.30us / 12.01%% fixed-overhead floor "
                         "(#408 qc9bz8sv). Omit -> 0 TPS credit (floor reduction not yet landed).")
    ap.add_argument("--lawine417-deployability-surface", dest="deployability_surface",
                    choices=["pending", "green", "red"], default="green",
                    help="lawine #417 — deploy surface (7 served files, 41.8 GPU-min identity-verify, reversible, "
                         "1 binding in-place line, human-gated). BANKED green 22:26Z (default); override red/pending "
                         "for sensitivity.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="strict-frontier")
    args = ap.parse_args(argv)

    syn = synthesize(args.measured_ppl, args.b_star, args.lmhead_eta,
                     supply_lift_available_tps=args.supply_lift_available_tps,
                     demand_reaches_500_on_floor=args.demand_reaches_500_on_floor,
                     supply_lift_required_tps=args.supply_lift_required_tps,
                     robust_pilot=args.robust_pilot,
                     gap_addressable_pp=args.gap_addressable_pp,
                     stark381_decode=args.stark381_decode,
                     irreducible_floor_survives_vbi=args.irreducible_floor_survives_vbi,
                     kanna403_ppl_safe_supply=args.kanna403_ppl_safe_supply,
                     ubel401_tree_coverage_ceiling=args.ubel401_tree_coverage_ceiling,
                     denken402_tree_net_supply=args.denken402_tree_net_supply,
                     selective_recompute_measured_tps=args.selective_recompute_measured_tps,
                     cb3_additivity_gap_tps=args.cb3_additivity_gap_tps,
                     floor_reduction_tps=args.floor_reduction_tps,
                     deployability_surface=args.deployability_surface)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 357, "agent": "fern",
        "kind": "strict-500-composite-reachability", "analysis_only": True,
        "measured_ppl_at_b_star": args.measured_ppl, "b_star": args.b_star,
        "lmhead_eta_measured": args.lmhead_eta,
        "supply_lift_available_tps_wirbel390_or_lawine388": args.supply_lift_available_tps,
        "demand_reaches_500_on_deployable_floor_denken383": args.demand_reaches_500_on_floor,
        "supply_lift_required_tps_denken383": args.supply_lift_required_tps,
        "robust_coverage_pilot": args.robust_pilot,
        "irreducible_floor_survives_vbi_ubel386": args.irreducible_floor_survives_vbi,
        "gap_coverage_addressable_pp": args.gap_addressable_pp,
        "stark381_decode_identity": args.stark381_decode,
        "kanna403_ppl_safe_supply": args.kanna403_ppl_safe_supply,
        "ubel401_tree_coverage_ceiling": args.ubel401_tree_coverage_ceiling,
        "denken402_tree_net_supply": args.denken402_tree_net_supply,
        # 22:08Z RE-POINT frontier feeders
        "selective_recompute_measured_tps_stark412": args.selective_recompute_measured_tps,
        "cb3_additivity_gap_tps_kanna416": args.cb3_additivity_gap_tps,
        "floor_reduction_tps_wirbel415": args.floor_reduction_tps,
        "deployability_surface_lawine417": args.deployability_surface,
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
