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
UBEL389_PIN_BREACH_PENDING: bool = True             # ubel #389 GPU per-L attn to PIN the -0.32pp breach

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
LAWINE388_REALIZED_TPS_PENDING: bool = True         # #388 microbench: does -21.5% read -> positive TPS

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
WIRBEL390_SHIPPABLE_CEILING_PENDING: bool = True     # corrected shippable strict ceiling — pending GPU measure
SHIPPABLE_CEILING_REFUTED_BF16_PREMISE_390: tuple[float, float] = (510.01, 518.92)  # both refuted (bf16-lm_head)

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
                              stark381_decode: str = "pending") -> dict[str, Any]:
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
    floor = HONEST_STRICT_BASE_FLOOR_378            # 469.68 off-the-shelf (VBI=1)
    plus_attn = HONEST_STRICT_BASE_PLUS_ATTN_378    # 480.7 (floor + ~11-TPS #375 attn rebuild)
    base_today = plus_attn                           # bank honestly at <=480.7-today (wirbel #378)
    band_lo, band_hi = DEPLOYABLE_STRICT_BAND_378
    clears_500_today = base_today >= TARGET          # False (480.7 < 500)
    deficit_to_500_today = TARGET - base_today       # ~19.3 TPS short today

    # 518.92 eta-axis base is deflated on THREE grounds (#373/#375/#378).
    eta_axis_base_deflated = ETA_AXIS_BASE_DEFLATED_518

    # Structural finding (advisor 17:03Z): private <= public; a public-strict base < 500 may make
    # private-strict-500 unreachable by the demand-side coverage retrain ALONE at any coverage.
    # denken #383 (17:53Z) CONFIRMED this on the honest base: demand-alone needs +0.0572 Δcov = 1.84x
    # the #336 budget -> a supply lift of +17.2 TPS (floor-joint) is required FIRST.
    demand_alone_may_be_insufficient = base_today < TARGET   # True today (480.7 < 500)
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
    if denken383_pending:
        supply_base_enables_500: bool | None = None
        binding = ("PENDING_denken383_honest_reprice"
                   + ("_and_supply_lift_wirbel390_or_lawine388" if supply_lift_measured_pending
                      else "_supply_lift_landed"))
    elif demand_reaches_500_on_floor == "yes":
        supply_base_enables_500 = True
        binding = "demand_route_reaches_500_on_deployable_floor_denken383"
    else:  # "no" (denken #383 RED) -> a supply lift is required first; wirbel #390 OR lawine #388 decide
        if supply_lift_measured_pending or supply_lift_required_tps is None:
            supply_base_enables_500 = None
            binding = "PENDING_supply_lift_wirbel390_or_lawine388_ge_required17p2_denken383"
        else:
            lift_sufficient = supply_lift_available_tps >= supply_lift_required_tps
            supply_base_enables_500 = bool(lift_sufficient)
            binding = ("supply_lift_available_ge_required_enables_500" if lift_sufficient
                       else "supply_lift_insufficient_private500_unreachable_demand_alone")

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
        "lawine388_realized_tps_pending": LAWINE388_REALIZED_TPS_PENDING,
        # wirbel #390 (reseated 18:12Z) — corrected SHIPPABLE strict ceiling (510.01/518.92 refuted)
        "wirbel390_shippable_ceiling_pending": WIRBEL390_SHIPPABLE_CEILING_PENDING,
        "shippable_ceiling_refuted_bf16_premise_390": list(SHIPPABLE_CEILING_REFUTED_BF16_PREMISE_390),
        # identity compliance folds in here (strict prerequisite of the deployable-strict base)
        "identity_reachable_env_or_rebuild": ident["identity_reachable_env_or_rebuild"],
        "identity_cost_branch": ident["cost_branch"],
        "identity_cost_branch_pending": ident["cost_branch_pending"],
        "identity_rebuild_line_items": ident["rebuild_line_items"],
        # pending inputs (the now-binding GO axis): supply lift from wirbel #390 OR lawine #388
        "denken383_input": demand_reaches_500_on_floor,
        "denken383_pending": denken383_pending,
        "supply_lift_available_tps": supply_lift_available_tps,
        "supply_lift_measured_pending": supply_lift_measured_pending,
        "supply_lift_pending": (supply_lift_measured_pending
                                and WIRBEL390_SHIPPABLE_CEILING_PENDING
                                and LAWINE388_REALIZED_TPS_PENDING),
        "supply_lift_required_tps": supply_lift_required_tps,
        # verdict
        "supply_pending": supply_pending,
        "supply_base_enables_500": supply_base_enables_500,
        "binding_constraint": binding,
        "note": (
            "wirbel #378 (gghmgtk9): the only STRICT-byte-exact served knob is VLLM_BATCH_INVARIANT=1 "
            f"(whole-step batch-invariant determinism); the deployable-strict band TODAY is "
            f"[{band_lo}, {band_hi}] < 500. The 518.92 eta-axis pin needs a rebuild that buys only "
            f"~{ATTN_REBUILD_TPS_GAIN_378:g} TPS (eta_attn={ETA_ATTN_378}, NOT #326's whole-step 0.3141). "
            f"wirbel #384 (4f32ks1e, 18:12Z) REFUTED #378's ~{ATTN_DEFICIT_UNTOUCHED_FRAC_378*100:.0f}% "
            f"by-elimination 'bf16 lm_head-BI ~{LMHEAD_BI_TAX_TPS_378_REFUTED:g}-TPS' attribution: the "
            f"deployed lm_head is already byte-exact int4-Marlin at decode (eta_lmhead={ETA_LMHEAD_TARGETED_384:g}, "
            f"FREE; f_lmhead={F_LMHEAD_384:g}), so the dominant non-attention strict tax lives in the "
            f"{DOMINANT_NONATTN_STRICT_LOCUS_384}. Corrected ledger: {N_KERNEL_REBUILDS_STRICT_500_384} kernel "
            "rebuilds (attn #375 + body-Marlin #376), NOT 3; lm_head shares the body kernel (0 rebuilds). "
            f"Bank the supply base at <={plus_attn:g}-today. STRUCTURAL: "
            "private <= public, so a public base < 500 may make private-500 unreachable by demand-side "
            "coverage retrain ALONE at any coverage. denken #383 (t68af2yw, RED, 17:53Z) CONFIRMED this: "
            f"on the honest floor the private serve point is {PRIVATE_ON_FLOOR_383:g} -> residual "
            f"+{RESIDUAL_TO_500_ON_FLOOR_383:g} TPS -> required Δcov +{REQUIRED_DCOV_383:g} = "
            f"{REQUIRED_DCOV_BUDGET_MULT_383:g}x the #336 budget; a supply lift of "
            f"+{SUPPLY_LIFT_REQUIRED_FIRST_TPS_383:g} TPS (floor-joint) is required FIRST (the ~25-GPU-hr "
            "coverage pilot is OFF the critical path until the base clears ~487-493). HELD pending a "
            "measured supply number from wirbel #390 (corrected SHIPPABLE strict ceiling; 510.01/518.92 "
            "refuted) OR lawine #388 (realized TPS of lawine #372's GREEN -21.5%-body-read mixed-precision "
            "BODY allocation). 518.92 deflated on three grounds: " + ETA_AXIS_DEFLATION_GROUNDS_378 + "."
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
        "ubel389_pin_breach_pending": UBEL389_PIN_BREACH_PENDING,
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
            "Re-derive the demand ceiling on the 1.310% live floor and treat private prompt-length-shift "
            "sensitivity as a BINDING risk. ubel reseated to a GPU per-L attention measurement (#389) to "
            "PIN the thin -0.32pp breach."
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
                               irreducible_floor_survives_vbi: str = UBEL386_FLOOR_SURVIVES_VBI
                               ) -> dict[str, Any]:
    """Demand-side residual leaf — necessary-but-insufficient (denken #383); central GREEN under VBI=1.

    Per denken #383 (17:53Z) the demand closer alone does NOT reach private-500 on the honest base, so
    this leaf delivers only the RESIDUAL after a supply lift (see supply_side_base_analysis).  It still
    DELIVERS at central confidence: deliverability central GREEN (denken #380) + slope GREEN (ubel #382)
    + gap channel live (ubel #379).  ubel #386 (RESOLVED) inflates the irreducible floor to 1.310% under
    VBI=1; central still clears 3.2% (+1.89pp) so the leaf is not dead, but prompt-shift sensitivity is
    now a BINDING risk and the robust tier no longer comfortably clears every corner.

    `robust_pilot`: the coverage-lift pilot that would harden the ROBUST tier (c >= 0.9010): one of
        "pending" | "delivers" | "fails".  Per denken #383 this pilot is OFF the critical path (the
        supply base must clear ~487-493 first); it gates only the robust-tier upgrade, never GO.
    `gap_addressable_pp` (ubel #379, BANKED GREEN): pp of the 4.295pp gap that is coverage-addressable.
        None -> use ubel #379's banked value.
    `irreducible_floor_survives_vbi` (ubel #386, RESOLVED): banked default "inflates" (-> 1.310% floor).
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

    # --- ubel #382 slope (BANKED GREEN) --- #
    # The slope is CONFIRMED private-robust; bank the conservative private-anchored target ~0.911.
    slope_private_robust = SLOPE_IS_PRIVATE_ROBUST_382                # True (banked)
    slope_private_tps = SLOPE_TPS_PER_COVERAGE_PRIVATE_382            # 437.3 (489.8 * 0.893)

    # --- leaf verdict --- #
    # The leaf DELIVERS the residual at CENTRAL confidence now: central deliverability GREEN AND slope
    # GREEN AND gap-channel live.  (If the gap were forced irreducible via an override, the leaf dies.)
    if not gap_channel_live:
        demand_leaf_delivers: bool | None = False
        binding = "gap_irreducible_no_coverage_channel_ubel379_override"
    elif deliver_central and slope_private_robust:
        demand_leaf_delivers = True
        binding = ("demand_leaf_delivers_central_green__robust_"
                   + ("pending_pilot" if deliver_robust is None
                      else ("delivers" if deliver_robust else "fails")))
    else:
        demand_leaf_delivers = False
        binding = "demand_central_or_slope_failed"

    # The leaf is "pending" only in the sense that the robust tier is not yet hardened; the central
    # delivery is RESOLVED.  GO-gating uses the central delivery (the supply base is the binding axis).
    leaf_robust_pending = robust_pilot_pending

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
        "ubel389_pin_breach_pending": UBEL389_PIN_BREACH_PENDING,
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
        # verdict
        "demand_leaf_delivers": demand_leaf_delivers,             # True now (central GREEN under VBI=1)
        "leaf_robust_pending": leaf_robust_pending,
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
            "prompt shift halves (+253 -> +119 tok) -> prompt-shift sensitivity is a BINDING risk (ubel "
            "#389 reseated to PIN the -0.32pp breach). Net: the demand leaf DELIVERS the residual at "
            "CENTRAL confidence under VBI=1, necessary-but-insufficient on its own (denken #383); it "
            "transfers FROM the supply-side honest base (wirbel #378 <=480.7-today) — see "
            "supply_side_base_analysis, the binding GO axis."
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
                      irreducible_floor_survives_vbi: str = "pending") -> dict[str, Any]:
    """AND-product GO: (supply base enables 500) x (demand closer delivers residual).

    The SUPPLY leaf (wirbel #378 honest base; denken #383 RED -> +17.2-TPS lift required first, pending
    a measured lift from wirbel #384 OR lawine #388) is the binding axis; the DEMAND leaf
    (denken #377/#380/#382) delivers the residual at central confidence now.  Route A (sub-int4 eta-axis)
    is retained for continuity but DEFLATED/excluded from the GO.
    """
    route_a = verdict_given_ppl(measured_ppl, b_star, lmhead_eta)              # sub-int4 eta-axis (deflated)
    supply = supply_side_base_analysis(supply_lift_available_tps, demand_reaches_500_on_floor,
                                       supply_lift_required_tps, stark381_decode)  # SUPPLY leaf (binding)
    demand = demand_side_route_analysis(robust_pilot, gap_addressable_pp,
                                        irreducible_floor_survives_vbi)         # DEMAND leaf (resolved central)
    ident_reach = identity_reachability_analysis(stark381_decode)             # identity compliance (folds into supply)

    a_reachable = route_a["strict_500_reachable_via_known_levers"]    # sub-int4 eta-axis standalone (deflated)
    supply_enables = supply["supply_base_enables_500"]               # True / False / None(pending #383/#384)
    demand_delivers = demand["demand_leaf_delivers"]                 # True now (central GREEN)
    identity_reachable = ident_reach["identity_reachable_env_or_rebuild"]  # True (env or rebuild)

    # AND-product GO, pending-aware.  The demand leaf delivers the residual at central confidence
    # (True); the GO hinges on the SUPPLY base enabling 500.  Stamp False if the demand leaf is dead
    # OR the supply base cannot enable 500 (demand-alone insufficient); True iff both clear; else None.
    if demand_delivers is False:
        reachable: bool | None = False
    elif supply_enables is False:
        reachable = False
    elif supply_enables is True and demand_delivers is True:
        reachable = True
    else:
        reachable = None
    pending = reachable is None

    # GO-gating pending inputs (advisor 17:53Z + 18:12Z correction): denken #383 has RESOLVED
    # (demand-alone insufficient on the honest base) -> the GO now hinges on a MEASURED SUPPLY LIFT of
    # +17.2 TPS.  wirbel #384 (18:12Z) RESOLVED the lm_head-BI lever as a non-source (lm_head is already
    # byte-exact int4-Marlin, eta=0, FREE; the tax is in the body #376), so the supply lift must come
    # from wirbel #390 (corrected SHIPPABLE strict ceiling, re-rolling the refuted 510.01/518.92) OR
    # lawine #388 (realized TPS of lawine #372's GREEN mixed-precision BODY allocation).  The robust
    # demand pilot (OFF the critical path per #383), ubel #389 (pin the VBI=1 floor breach), and stark
    # #381 identity cost are REFINEMENTS (not GO-gating: the demand leaf already delivers central under
    # VBI=1, identity is reachable env-or-rebuild).
    pending_inputs: list[str] = []
    if supply["denken383_pending"]:
        pending_inputs.append("denken#383_demand_route_reaches_500_on_deployable_floor")
    if supply["supply_lift_measured_pending"]:
        pending_inputs.append("supply_lift_wirbel390_shippable_OR_lawine388_realized_ge_required17p2")
    refinement_inputs: list[str] = []
    if demand["robust_pilot_pending"]:
        refinement_inputs.append("coverage_lift_pilot_robust_tier_c0p9010_off_critical_path_383")
    if demand["irreducible_floor_vbi_pending_ubel386"]:
        refinement_inputs.append("ubel#386_irreducible_floor_under_vbi1")
    if demand.get("ubel389_pin_breach_pending"):
        refinement_inputs.append("ubel#389_pin_vbi1_floor_breach")
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
        f"demand leaf DELIVERS the residual at central confidence (c >= {DEMAND_CLOSER_CENTRAL_C} "
        f"GREEN; robust c >= {DEMAND_CLOSER_ROBUST_C} "
        + ("pending pilot" if demand["robust_pilot_pending"]
           else ("delivered" if demand["deliver_robust_resolved"] else "failed"))
        + f"; sized at conservative {DEMAND_CONSERVATIVE_TARGET_382})")

    if reachable is True:
        binding = ("private_500_REACHABLE_supply_base_enables_x_demand_delivers__"
                   f"{supply['binding_constraint']}")
    elif reachable is False:
        binding = (f"demand_leaf_dead__{demand['binding_constraint']}" if demand_delivers is False
                   else f"supply_base_cannot_enable_500_demand_alone_insufficient__{supply['binding_constraint']}")
    else:
        binding = f"PENDING__supply_base_x_demand_residual__inputs={','.join(pending_inputs)}"

    if pending:
        verdict_text = (
            "private-500 reachability HELD — GO = (supply-side public-strict base ENABLES 500) x "
            f"(demand closer DELIVERS the residual). denken #383 (RED) CONFIRMED demand-alone does NOT "
            f"reach 500 on the honest base (wirbel #378 <={supply['honest_strict_base_plus_attn']:g}-today "
            f"< 500; residual +{RESIDUAL_TO_500_ON_FLOOR_383:g} TPS = {REQUIRED_DCOV_BUDGET_MULT_383:g}x "
            f"budget), so the SUPPLY base is the BINDING axis: it needs a +{SUPPLY_LIFT_REQUIRED_FIRST_TPS_383:g}-TPS "
            f"lift FIRST, HELD pending a measured number from wirbel #390 (corrected SHIPPABLE strict "
            f"ceiling; 510.01/518.92 refuted) OR lawine #388 (realized TPS of lawine #372's GREEN "
            f"-21.5%-read mixed-precision BODY allocation); "
            f"{supply['binding_constraint']}. The {demand_phrase}; {cost_phrase}. wirbel #384 (RED) "
            f"refuted the lm_head-BI lever (lm_head FREE int4-Marlin, tax in body #376). ubel #386 (RED) inflated "
            f"the demand floor to 1.310% under VBI=1 (central still clears; prompt-shift now a binding "
            f"risk, ubel #389 to pin). Route A (sub-int4 eta-axis): DEFLATED/excluded. GO-gating pending: "
            f"{pending_inputs}"
            f"{' (+refinements ' + str(refinement_inputs) + ')' if refinement_inputs else ''}."
        )
    elif reachable:
        verdict_text = (
            f"private-500 REACHABLE: the supply-side public-strict base ENABLES 500 "
            f"({supply['binding_constraint']}) and the {demand_phrase}; {cost_phrase}. "
            f"Flag as approval-gated a10g candidate (#319)."
        )
    elif demand_delivers is False:
        verdict_text = (
            "private-500 NOT reachable via known levers: the demand leaf failed "
            f"({demand['binding_constraint']}). Genuinely-new-method problem."
        )
    else:
        verdict_text = (
            "private-500 NOT reachable via known levers: the supply-side public-strict base cannot "
            f"enable 500 even with the available supply lift (wirbel #384 / lawine #388) "
            f"({supply['binding_constraint']}), so the demand closer CANNOT close private-500 alone "
            "(private <= public). A genuinely-new supply lift would be required first."
        )

    return {
        "primary_route": "supply_x_demand_and_product",
        "eta_axis_deflated": True,
        "go_formula": ("private_500_GO <=> (supply_base_enables_500) AND "
                       "(demand_closer_delivers_residual)"),
        "private_500_reachable_via_known_levers": reachable,
        "verdict_pending": pending,
        "pending_inputs": pending_inputs,                       # GO-gating (supply-base hardeners)
        "refinement_inputs": refinement_inputs,                 # robust pilot + ubel#386 + stark#381 (not GO)
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
            "binding": demand["binding_constraint"],
        },
        # supply leaf (binding GO axis)
        "supply_leaf": {
            "enables_500": supply_enables,
            "pending": supply["supply_pending"],
            "binding": supply["binding_constraint"],
            "honest_base_today_tps": supply["supply_base_today_tps"],
            "clears_500_today": supply["supply_base_clears_500_today"],
        },
        # sub-int4 eta-axis standalone — DEFLATED, excluded from the GO (kept for continuity)
        "route_a_supply_side": {
            "reachable": a_reachable,
            "pending": route_a["verdict_pending"],
            "binding": route_a["binding_constraint"],
            "in_go_product": False,
            "leans_dead_reason": "sub-int4 UNIFORM eta-axis standalone deflated (#373/#375/#378) -> "
                                 "un-deployable standalone path. NOTE: lawine #372's sensitivity-weighted "
                                 "mixed-precision ALLOCATION is GREEN/alive, but as a SUPPLY-leaf de-risker "
                                 "(pending lawine #388 realized TPS), not a Route A revival.",
        },
        "route_a_detail": route_a,
        "supply_leaf_detail": supply,
        "demand_leaf_detail": demand,
    }


# --------------------------------------------------------------------------- #
# Deliverable 5: caveats.
# --------------------------------------------------------------------------- #
def deliverable5_caveats(b_star: float) -> dict[str, Any]:
    return {
        "caveats": [
            "GO IS A SUPPLY x DEMAND AND-PRODUCT, and HELD. private_500_GO <=> (SUPPLY base enables 500) "
            "AND (DEMAND closer delivers the residual). It is NO LONGER demand-alone and NO LONGER "
            "(demand x identity): wirbel #378 found the honest deployable-strict base is < 500 TODAY "
            "(served band [357.32, 469.68]; rebuilt ~480.7), so the SUPPLY base is now the binding GO axis. "
            "Identity FOLDS INTO the supply leaf as a compliance prerequisite (cost, not a top-level "
            "factor). The sub-int4 eta-axis (Route A) stays DEFLATED/EXCLUDED. All cross-PR numbers are "
            "consumed via the PR thread, NOT by reading other branches.",
            "SUPPLY LEAF IS THE BINDING GO AXIS, PENDING A MEASURED LIFT (wirbel #378). The #375 attn "
            "rebuild buys only ~11 TPS (eta_attn=0.0215, NOT #326's whole-step 0.3141), leaving ~93% of the "
            "deficit. wirbel #384 (RED, BANKED 18:12Z) REFUTED #378's 'bf16 lm_head-BI ~150-TPS' "
            "by-elimination attribution of that remainder: the deployed lm_head is already byte-exact "
            "int4-Marlin at decode (eta_lmhead=0, FREE; f_lmhead=2.24%), so the dominant non-attention "
            "strict tax actually lives in the 37-layer int4-Marlin BODY (#376); corrected ledger = 2 kernel "
            "rebuilds (attn #375 + body #376), NOT 3. The 518.92 eta-axis pin is DEFLATED on three grounds "
            "(#373 insufficient 498.58<500; #375 un-deployable-without-rebuild; #378 rebuild buys ~11 TPS), "
            "and the 510.01/518.92 SHIPPABLE ceilings are now refuted (computed under the bf16-lm_head "
            "premise) -> wirbel #390 reseated to re-roll the corrected realized shippable number. STRUCTURAL: "
            "private <= public, so a <500 public base makes demand-alone INSUFFICIENT. denken #383 (RED, "
            "BANKED 17:53Z) CONFIRMED this on the honest base: the private serve point is 450.2 -> residual "
            "+49.8 TPS = +0.0572 Δcov = 1.84x #336 budget; a supply lift of +17.2 TPS (floor-joint; +23.8 "
            "E[T]-only) is required FIRST, and the attn rebuild ALONE does not close it. So the supply leaf "
            "now resolves on a MEASURED supply lift >= +17.2 TPS from wirbel #390 (corrected SHIPPABLE strict "
            "ceiling) OR lawine #388 (realized TPS of lawine #372's GREEN mixed-precision BODY allocation); "
            "the ~25-GPU-hr coverage pilot is OFF the critical path until the base clears ~487-493. I do NOT "
            "re-run any GPU eval; I CONSUME #383/#384/#388/#390 via the thread.",
            "SUPPLY LEVER IS ALIVE (lawine #372 GREEN, BANKED 17:53Z) — pending realized TPS (lawine #388). "
            "The supply de-risker is NOT uniform sub-int4 (that died on PPL) but a SENSITIVITY-WEIGHTED "
            "mixed-precision ALLOCATION: 88.8% of body params at 3-bit for a 3.2369 avg bpw, +0.17% PPL "
            "(gate 2.3812 <= 2.42), buying -21.5% body read traffic and +132.72/+42.15 ANALYTIC TPS lift. "
            "The realized cb3-style kernel speed vs int4-Marlin at M=1 on A10G is UN-measured — lawine #388 "
            "is microbenching exactly that. Bank the lever as alive but HOLD the supply GO on #388's "
            "realized number (analytic lift is not a deployable TPS).",
            "DEMAND LEAF IS NECESSARY-BUT-INSUFFICIENT (denken #383): it DELIVERS the RESIDUAL at central "
            "confidence after a supply lift, but does NOT reach private-500 ALONE on the honest base. denken "
            "#377 SIZED the closer; denken #380 split deliverability TWO-TIER: central c>=0.8959 (+0.00565) "
            "p_deliver=0.958>=0.90 GREEN-deliverable-now, robust c>=0.9010 (+0.0107) p_deliver=0.811<0.90 "
            "pending a ~25-A10G-GPU-hr pilot (now OFF the critical path per #383). kappa_breakeven 0.1222, "
            "kappa_margin 0.549 (kappa transfer axis ROBUST). I treat the conservative target ~0.911 as "
            "ALREADY-SIZED to close the residual (I do not re-derive coverage->E[T]->private-500). The demand "
            "leaf dies ONLY if the gap is forced fully irreducible.",
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
            "prompt_logprobs API). stark #381 resolves the served 8-row geometry: GREEN -> env-reachable@"
            "decode (1 rebuild: #375 mha_varlen); RED -> Marlin-rebuild-gated (2 rebuilds). Identity is "
            "REACHABLE in BOTH branches; #381 sets the deployment COST (1 vs 2 kernel rebuilds), folded "
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
               d8: dict, d_supply: dict, d_ident_reach: dict, b_star: float) -> dict[str, Any]:
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

    # aa: the DEMAND leaf is RESOLVED central=GREEN / robust=pending-pilot (denken #380 + ubel #382).
    #     Central delivery HOLDS across every robust-pilot state (the robust tier is a refinement, not
    #     a GO gate); the robust tier resolves only with the coverage-lift pilot.
    d6_pilot_ok = demand_side_route_analysis("delivers")
    d6_pilot_fail = demand_side_route_analysis("fails")
    aa_demand_central_green_robust_tiered = (
        d6["demand_leaf_delivers"] is True
        and d6["deliver_central_green"] is True
        and d6["deliver_robust_modeled"] is False        # robust tier needs the pilot (p_deliver<0.90)
        and d6["deliver_robust_resolved"] is None        # pending default -> unresolved
        and d6["leaf_robust_pending"] is True
        and d6_pilot_ok["deliver_robust_resolved"] is True
        and d6_pilot_fail["deliver_robust_resolved"] is False
        and d6_pilot_fail["demand_leaf_delivers"] is True)  # central delivery holds regardless of pilot

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

    # --- supply-side honest-base leaf (wirbel #378, the NOW-BINDING GO axis) --- #
    # ao: the honest deployable-strict base is <=480.7-today (< 500); demand-alone may be insufficient
    #     (private <= public) and denken #383 (RED) CONFIRMED it on the honest base; the 518.92 eta-axis
    #     pin is deflated; this is the binding GO leaf.
    ao_supply_leaf_honest_base_below_500 = (
        abs(d_supply["supply_base_today_tps"] - HONEST_STRICT_BASE_PLUS_ATTN_378) < TOL_EXACT
        and d_supply["supply_base_today_tps"] < TARGET
        and abs(d_supply["honest_strict_base_floor"] - HONEST_STRICT_BASE_FLOOR_378) < TOL_EXACT
        and d_supply["supply_base_clears_500_today"] is False
        and d_supply["demand_alone_may_be_insufficient"] is True
        and d_supply["demand_alone_insufficient_confirmed_383"] is True   # banked denken #383 RED
        and abs(d_supply["eta_axis_base_deflated_518"] - ETA_AXIS_BASE_DEFLATED_518) < TOL_EXACT
        and d_supply["is_binding_go_leaf"] is True)

    # ap: the supply leaf RESOLVES on denken #383 (reach-on-floor) + wirbel #384 (available lift):
    #     pending -> None; "yes" -> True; "no" + insufficient lift -> False; "no" + sufficient -> True.
    s_pending = supply_side_base_analysis()
    s_yes = supply_side_base_analysis(demand_reaches_500_on_floor="yes")
    s_no_insuff = supply_side_base_analysis(supply_lift_available_tps=10.0,
                                            demand_reaches_500_on_floor="no",
                                            supply_lift_required_tps=40.0)
    s_no_suff = supply_side_base_analysis(supply_lift_available_tps=25.0,
                                          demand_reaches_500_on_floor="no",
                                          supply_lift_required_tps=15.0)
    ap_supply_leaf_resolves_on_inputs = (
        s_pending["supply_base_enables_500"] is None
        and s_pending["supply_pending"] is True
        and s_pending["denken383_pending"] is True
        and s_yes["supply_base_enables_500"] is True
        and s_no_insuff["supply_base_enables_500"] is False
        and s_no_suff["supply_base_enables_500"] is True)

    # --- composite supply x demand AND-product (the new GO) --- #
    # aq: fully pending -> None; supply enables ("yes") + demand central GREEN -> True; supply cannot
    #     enable ("no" + insufficient lift) -> False; demand leaf dead (gap forced) -> False. Route A
    #     (sub-int4 eta-axis) is EXCLUDED from the product (in_go_product False).
    c_pending = composite_verdict(None, b_star, None)
    c_go = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="yes")
    c_nogo = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                               supply_lift_required_tps=40.0, supply_lift_available_tps=10.0)
    c_go_lift = composite_verdict(None, b_star, None, demand_reaches_500_on_floor="no",
                                  supply_lift_required_tps=15.0, supply_lift_available_tps=25.0)
    c_demand_dead = composite_verdict(None, b_star, None,
                                      demand_reaches_500_on_floor="yes", gap_addressable_pp=0.0)
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

    # at: ubel #386 detail — under the inflated 1.310% live floor the CENTRAL still clears the 3.2%
    #     knife-edge (+1.89pp) so the demand leaf is NOT dead, but all corners no longer clear (the
    #     pessimistic corner breaches at -0.32pp), the breakeven private prompt shift HALVES (253 ->
    #     119 tok), prompt-shift sensitivity is a BINDING risk, and ubel #389 pins the breach (pending).
    at_ubel386_inflated_floor_binding_risk = (
        d8["central_clears_3p2_vbi1_386"] is True
        and abs(d8["central_margin_to_3p2_vbi1_pp_386"] - CENTRAL_MARGIN_TO_3P2_VBI1_PP_386) < TOL_EXACT
        and d8["all_corners_clear_3p2_vbi1_386"] is False
        and d8["pessimistic_corner_margin_pp_386"] < 0.0
        and abs(d8["breakeven_prompt_shift_vbi1_tok_386"] - BREAKEVEN_PROMPT_SHIFT_VBI1_TOK_386) < TOL_EXACT
        and d8["breakeven_prompt_shift_vbi1_tok_386"] < BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK
        and d8["prompt_shift_sensitivity_binding_risk_386"] is True
        and d8["ubel389_pin_breach_pending"] is True
        # the demand leaf carries the inflated floor through but still delivers central:
        and d6["irreducible_floor_inflates_vbi_386"] is True
        and abs(d6["irreducible_floor_vbi1_central_pct_386"] - IRREDUCIBLE_FLOOR_VBI1_CENTRAL_386) < TOL_EXACT
        and d6["central_clears_3p2_vbi1_386"] is True
        and d6["all_corners_clear_3p2_vbi1_386"] is False
        and d6["prompt_shift_sensitivity_binding_risk_386"] is True
        and d6["demand_leaf_delivers"] is True)

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
    #     for -21.5% body read; the UNIFORM-3bit variant died on PPL. Realized M=1 TPS is pending #388.
    av_lawine372_supply_lever_alive = (
        d_supply["lawine372_supply_lever_alive"] is True
        and abs(d_supply["mixed_precision_avg_bpw_372"] - MIXED_PRECISION_AVG_BPW_372) < TOL_EXACT
        and abs(d_supply["body_3bit_frac_372"] - BODY_3BIT_FRAC_372) < TOL_EXACT
        and d_supply["mixed_precision_gate_ppl_372"] <= PPL_GATE
        and abs(d_supply["body_read_reduction_372"] - BODY_READ_REDUCTION_372) < TOL_EXACT
        and d_supply["uniform_3bit_died_on_ppl_372"] is True
        and d_supply["lawine388_realized_tps_pending"] is True)

    # aw: wirbel #384 (RED, BANKED 18:12Z) — CORRECTION: the deployed lm_head is ALREADY byte-exact
    #     strict at decode (untied int4-Marlin, eta_lmhead=0, FREE), so the ~150-TPS "bf16 lm_head-BI"
    #     determinization tax was a by-elimination artifact (incremental share ~0). The dominant
    #     non-attention strict tax lives in the int4-Marlin BODY (#376); the rebuild ledger is 2 (attn
    #     #375 + body #376), NOT 3 (lm_head shares the body kernel). The 510.01/518.92 ceilings are
    #     REFUTED (bf16-lm_head premise); wirbel #390 is reseated to re-roll the shippable ceiling, and
    #     the binding supply-lift input now comes from wirbel #390 OR lawine #388 (default: pending).
    aw_wirbel384_lmhead_free_supply_tax_in_body = (
        d_supply["wirbel384_lmhead_free"] is True
        and abs(d_supply["eta_lmhead_targeted_384"] - 0.0) < TOL_EXACT
        and d_supply["n_kernel_rebuilds_strict_500_384"] == 2
        and abs(d_supply["lmhead_bi_incremental_share_384"] - 0.0) < TOL_EXACT
        and d_supply["lmhead_is_int4_marlin_not_bf16_384"] is True
        and d_supply["dominant_nonattn_strict_locus_384"] == DOMINANT_NONATTN_STRICT_LOCUS_384
        and d_supply["wirbel390_shippable_ceiling_pending"] is True
        and d_supply["supply_lift_measured_pending"] is True
        # the identity-reachability ledger agrees: lm_head free at decode -> 2 rebuilds, not 3
        and d_ident_reach["lmhead_int4_marlin_free_at_decode_wirbel384"] is True
        and d_ident_reach["n_kernel_rebuilds_strict_500_wirbel384"] == 2
        # the composite GO-gate carries the corrected wirbel #390 OR lawine #388 supply-lift input
        and ("supply_lift_wirbel390_shippable_OR_lawine388_realized_ge_required17p2"
             in d7["pending_inputs"]))

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
    # am: identity is REACHABLE in BOTH #381 branches; deployment COST differs (1 vs 2 rebuilds);
    #     pending leaves reachability True but the cost branch pending
    am_identity_reachable_both_branches = (
        ident_green["identity_reachable_env_or_rebuild"] is True
        and ident_red["identity_reachable_env_or_rebuild"] is True
        and d_ident_reach["identity_reachable_env_or_rebuild"] is True
        and ident_green["rebuild_line_items"] == 1
        and ident_red["rebuild_line_items"] == 2
        and ident_green["cost_branch"] == "env_reachable_at_decode_width"
        and ident_red["cost_branch"] == "marlin_rebuild_gated"
        and d_ident_reach["cost_branch_pending"] is True)

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
        "at_ubel386_inflated_floor_binding_risk": bool(at_ubel386_inflated_floor_binding_risk),
        "au_denken383_supply_lift_required_first": bool(au_denken383_supply_lift_required_first),
        "av_lawine372_supply_lever_alive": bool(av_lawine372_supply_lever_alive),
        "aw_wirbel384_lmhead_free_supply_tax_in_body": bool(aw_wirbel384_lmhead_free_supply_tax_in_body),
        "ai_gap_fractions_sum_to_one": bool(ai_gap_fracs_sum_to_one),
        "aj_gap_irreducible_floor_matches_ctxlen": bool(aj_gap_floor_matches_ctxlen),
        "ak_gap_corners_clear_knife_edge": bool(ak_gap_corners_clear_knife_edge),
        "al_gap_reconciles_denken377_numerics_cancels": bool(al_gap_reconciles_numerics_cancels),
        "am_identity_reachable_both_branches_cost_differs": bool(am_identity_reachable_both_branches),
        "an_identity_residual_marlin_decode_width_caveat": bool(an_identity_residual_marlin_decode_caveat),
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
               irreducible_floor_survives_vbi: str = UBEL386_FLOOR_SURVIVES_VBI  # banked "inflates" (#386)
               ) -> dict[str, Any]:
    d1 = deliverable1_lever_analysis(b_star)
    d2 = deliverable2_ppl_forecast()
    d3 = deliverable3_composite_tps(b_star)
    d4 = verdict_given_ppl(measured_ppl, b_star, lmhead_eta)              # Route A supply-side (deflated)
    d_ident = identity_locus_analysis(lmhead_eta)                        # eta-locus lever (deflated read)
    d_ident_reach = identity_reachability_analysis(stark381_decode)      # identity REACHABILITY compliance
    d_supply = supply_side_base_analysis(supply_lift_available_tps, demand_reaches_500_on_floor,
                                         supply_lift_required_tps, stark381_decode)  # SUPPLY leaf (binding)
    d8 = gap_decomposition_analysis(irreducible_floor_survives_vbi)       # ubel #379 GREEN + ubel #386 pend
    d6 = demand_side_route_analysis(robust_pilot, gap_addressable_pp,     # DEMAND leaf (resolved central)
                                    irreducible_floor_survives_vbi)
    d7 = composite_verdict(measured_ppl, b_star, lmhead_eta,
                           supply_lift_available_tps=supply_lift_available_tps,
                           demand_reaches_500_on_floor=demand_reaches_500_on_floor,
                           supply_lift_required_tps=supply_lift_required_tps,
                           robust_pilot=robust_pilot,
                           gap_addressable_pp=gap_addressable_pp,
                           stark381_decode=stark381_decode,
                           irreducible_floor_survives_vbi=irreducible_floor_survives_vbi)
    d5 = deliverable5_caveats(b_star)
    st = _selftests(d1, d2, d3, d4, d5, d6, d7, d8, d_supply, d_ident_reach, b_star)

    headline = {
        # PRIMARY
        "strict_500_composite_reachability_self_test_passes": (
            st["strict_500_composite_reachability_self_test_passes"]),
        # TEST metrics — top-level verdict is the AND-PRODUCT GO:
        #   private_500_GO <=> (supply base enables 500) AND (demand closer delivers residual)
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
        # supply-side honest-base leaf (wirbel #378, now-BINDING GO axis; denken #383 RED, pending supply lift)
        "supply_base_today_tps": d_supply["supply_base_today_tps"],
        "supply_base_clears_500_today": d_supply["supply_base_clears_500_today"],
        "honest_strict_base_floor_378": d_supply["honest_strict_base_floor"],
        "honest_strict_base_plus_attn_378": d_supply["honest_strict_base_plus_attn"],
        "deficit_to_500_today_tps": d_supply["deficit_to_500_today_tps"],
        "supply_pending": d_supply["supply_pending"],
        "supply_binding_constraint": d_supply["binding_constraint"],
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
            f"PRIVATE-500 GO HELD — the GO is an AND-PRODUCT: (supply-side public-strict base ENABLES "
            f"500) x (demand closer DELIVERS the residual). GO-gating pending inputs: {d7['pending_inputs']}"
            f"{' (+refinements ' + str(d7['refinement_inputs']) + ')' if d7['refinement_inputs'] else ''}. "
            f"SUPPLY BASE (wirbel #378, the NOW-BINDING axis): the honest deployable-strict base is "
            f"<={d_supply['honest_strict_base_plus_attn']:g}-today (floor {d_supply['honest_strict_base_floor']:g} "
            f"off-the-shelf VBI=1 + ~{ATTN_REBUILD_TPS_GAIN_378:g}-TPS #375 attn rebuild), BOTH < 500; the "
            f"518.92 eta-axis pin is DEFLATED ({d_supply['eta_axis_deflation_grounds']}). STRUCTURAL: private "
            f"<= public, so a public base < 500 makes private-500 unreachable by the demand closer ALONE at "
            f"any coverage; denken #383 (RED, BANKED) CONFIRMED it (private-on-floor "
            f"{d_supply['private_on_floor_383']:g} -> residual +{d_supply['residual_to_500_on_floor_383']:g} TPS "
            f"= {d_supply['required_dcov_budget_mult_383']:g}x budget) -> +{SUPPLY_LIFT_REQUIRED_FIRST_TPS_383:g}-TPS "
            f"supply lift required FIRST. wirbel #384 (RED, BANKED 18:12Z) REFUTED the lm_head-BI lever "
            f"(deployed lm_head already byte-exact int4-Marlin, eta=0/FREE; the tax is in the body #376; "
            f"corrected ledger = {N_KERNEL_REBUILDS_STRICT_500_384} rebuilds, NOT 3), so the supply lift is "
            f"HELD pending a MEASURED number from wirbel #390 (corrected SHIPPABLE strict ceiling; "
            f"510.01/518.92 refuted) OR lawine #388 (realized TPS of lawine #372's GREEN mixed-precision "
            f"BODY allocation); {d_supply['binding_constraint']}. "
            f"DEMAND RESIDUAL (denken #377 sized; RESOLVED central=GREEN / robust=pending-pilot): retrain the "
            f"drafter to the conservative private-anchored target ~{DEMAND_CONSERVATIVE_TARGET_382} "
            f"(central c >= {DEMAND_CLOSER_CENTRAL_C} p_deliver {P_DELIVER_CENTRAL_DEFENSIBLE_380} >= 0.90 "
            f"GREEN; robust c >= {DEMAND_CLOSER_ROBUST_C} p_deliver {P_DELIVER_ROBUST_DEFENSIBLE_380} < 0.90 "
            f"-> {'pending a ~' + str(int(COVERAGE_LIFT_PILOT_GPU_HR)) + ' A10G-GPU-hr pilot' if d6['robust_pilot_pending'] else ('pilot DELIVERS' if d6['deliver_robust_resolved'] else 'pilot FAILS')}). "
            f"ubel #382 (GREEN) BANKED the slope private-robust ({d8['slope_tps_per_coverage_ubel379']:g} -> "
            f"{d6['slope_tps_per_coverage_private_382']:.1f} private, flattening {SLOPE_FLATTENING_RATIO_382}); "
            f"ubel #379 (GREEN) BANKED the gap ceiling-check ({d8['gap_addressable_pp']:.3f}pp of the "
            f"{PUBLIC_PRIVATE_GAP_PCT}pp gap coverage-addressable); ubel #386 (RED, BANKED) found the "
            f"off-VBI {d8['gap_irreducible_pp_central']:g}% floor INFLATES {d8['floor_inflation_mult_386']:g}x -> "
            f"{d8['irreducible_floor_vbi1_central_pct_386']:g}% central under VBI=1 (central still clears 3.2% "
            f"+{d8['central_margin_to_3p2_vbi1_pp_386']:g}pp, but all corners no longer clear -> prompt-shift "
            f"sensitivity BINDING, breakeven {BREAKEVEN_PROMPT_SHIFT_PRE_VBI_TOK:g}->{d8['breakeven_prompt_shift_vbi1_tok_386']:g} tok; "
            f"ubel #389 to pin the -0.32pp breach). IDENTITY (strict-lock compliance, folds "
            f"into the supply leaf, REACHABLE in BOTH #381 branches): "
            f"{'cost branch PENDING stark #381 (env@decode=1 rebuild vs Marlin-gated=2 rebuilds)' if d7['identity_cost_branch_pending'] else d7['identity_cost_branch'] + ' (' + str(d7['identity_rebuild_line_items']) + ' rebuild line-item(s))'}"
            f" — #381 sets COST, not GO. Route A (sub-int4 eta-axis standalone): "
            f"{'PENDING denken #356 PPL@b* + stark #365 lm_head eta' if d4['verdict_pending'] else d4['binding_constraint']} "
            f"-- DEFLATED/un-deployable (PPL-blocked lawine #372; #373/#375), EXCLUDED from the GO."
        )
    else:
        handoff = (
            f"PRIVATE-500 GO RESOLVED: " + d7["verdict_text"]
        )

    return {
        "headline": headline,
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
    print("STRICT/PRIVATE-500 COMPOSITE REACHABILITY (#357, fern) — GO = (supply-side public-strict base "
          "ENABLES 500) x (demand closer DELIVERS residual); Route A eta-axis DEFLATED (excluded)", flush=True)
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
    print(f"      demand_leaf_delivers={d6['demand_leaf_delivers']} (central GREEN)  robust_pending="
          f"{d6['leaf_robust_pending']}  [{d6['binding_constraint']}]", flush=True)
    d_sup = syn["deliverable_supply_side_base"]
    print("-" * 98, flush=True)
    print("  (D-SUPPLY) SUPPLY honest-base leaf — wirbel #378 (the NOW-BINDING GO axis)", flush=True)
    print(f"      deployable-strict band TODAY {d_sup['deployable_strict_band_today']} < 500; honest base "
          f"<= {d_sup['honest_strict_base_plus_attn']:g} (floor {d_sup['honest_strict_base_floor']:g} off-shelf "
          f"VBI=1 + ~{d_sup['attn_rebuild_tps_gain']:g}-TPS #375 attn rebuild)", flush=True)
    print(f"      clears_500_today={d_sup['supply_base_clears_500_today']} (deficit "
          f"{d_sup['deficit_to_500_today_tps']:.1f} TPS); 518.92 eta-axis pin DEFLATED; dominant non-attn tax "
          f"= {d_sup['dominant_nonattn_strict_locus_384']}", flush=True)
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
    print("  (D7) PRIVATE-500 GO — AND-PRODUCT: (supply base ENABLES 500) x (demand closer DELIVERS residual)",
          flush=True)
    print(f"      go_formula: {d7['go_formula']}", flush=True)
    print(f"      supply_base_enables_500={d7['supply_base_enables_500']}  "
          f"demand_closer_delivers={d7['demand_closer_delivers']}  "
          f"demand_alone_may_be_insufficient={d7['demand_alone_may_be_insufficient']}", flush=True)
    print(f"      identity (folds into supply)={d7['identity_reachable_env_or_rebuild']} "
          f"({d7['identity_cost_branch']})  "
          f"route_a_in_go_product={d7['route_a_supply_side']['in_go_product']} (DEFLATED, excluded)",
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
            "lawine-388-realized-tps-pending", "route-a-deflated",
            # advisor 18:12Z correction: wirbel #384 RED (lm_head FREE int4-Marlin; tax in body #376)
            "wirbel-384-lmhead-free", "lmhead-int4-marlin-not-bf16", "lmhead-bi-by-elimination-refuted",
            "supply-tax-in-body-376", "rebuilds-2-not-3", "wirbel-390-shippable-ceiling-pending",
            "shippable-ceiling-510-518-refuted",
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
                "eta=0 -> lm_head-BI supply lever refuted; supply tax in body #376; rebuilds 2 not 3; 510.01/"
                "518.92 shippable ceilings refuted -> wirbel#390 reseated). PENDING GO (now SUPPLY-binding): "
                "supply_lift_available from wirbel#390(corrected SHIPPABLE strict ceiling) OR lawine#388(realized "
                "TPS of #372 body allocation). REFINEMENTS: robust coverage pilot (off critical path) + "
                "ubel#389(pin VBI=1 floor breach) + stark#381(identity cost: body free at decode -> rebuilds->1)"
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
        "supply_lift_pending": int(bool(d_sup["supply_lift_pending"])),
        # wirbel #384 (RED, BANKED 18:12Z): lm_head FREE int4-Marlin; supply tax in body #376; rebuilds 2 not 3
        "wirbel384_lmhead_free": int(bool(d_sup["wirbel384_lmhead_free"])),
        "eta_lmhead_targeted_384": d_sup["eta_lmhead_targeted_384"],
        "f_lmhead_384": d_sup["f_lmhead_384"],
        "lmhead_bi_share_of_vbi_overhead_384": d_sup["lmhead_bi_share_of_vbi_overhead_384"],
        "n_kernel_rebuilds_strict_500_384": d_sup["n_kernel_rebuilds_strict_500_384"],
        "body_marlin_decode_strict_pending_stark381_384": int(bool(d_sup["body_marlin_decode_strict_pending_stark381_384"])),
        # wirbel #390 (reseated 18:12Z): corrected SHIPPABLE strict ceiling (510.01/518.92 refuted)
        "wirbel390_shippable_ceiling_pending": int(bool(d_sup["wirbel390_shippable_ceiling_pending"])),
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
                    help="denken #383 honest re-price (BANKED RED 17:53Z, default 'no'): the demand "
                         "closer does NOT reach 500 on the deployable-strict floor (<=480.7 today) -> a "
                         "supply lift is required first (see --supply-lift-required-tps). Override to "
                         "'pending' only to model the pre-#383 unresolved state.")
    ap.add_argument("--supply-lift-required-tps", dest="supply_lift_required_tps", type=float,
                    default=SUPPLY_LIFT_REQUIRED_FIRST_TPS_383,
                    help="denken #383 (BANKED, default +17.2 floor-joint): TPS of supply lift the honest "
                         "base needs before demand can close (meaningful when "
                         "--demand-reaches-500-on-deployable-floor=no).")
    ap.add_argument("--supply-lift-available-tps", dest="supply_lift_available_tps", type=float,
                    default=None,
                    help="MEASURED supply lift available to close the <500 deployable-strict deficit, from "
                         "wirbel #390 (corrected SHIPPABLE strict ceiling; 510.01/518.92 refuted under #384's "
                         "lm_head-free correction) OR lawine #388 (realized TPS of lawine #372's mixed-precision "
                         "body allocation). The wirbel #384 lm_head-BI determinization lever is REFUTED (lm_head "
                         "already byte-exact int4-Marlin, eta=0). Omit -> both pending; SUPPLY leaf stays PENDING.")
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
                         "REACHABLE in both branches; this does NOT gate the GO.")
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
                     irreducible_floor_survives_vbi=args.irreducible_floor_survives_vbi)

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
