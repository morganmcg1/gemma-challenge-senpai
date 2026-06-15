#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #332 (denken) -- Analytic phi-floor: can a DETERMINISTIC SDPA recover the
forgone split-KV reduction parallelism (identity lane C -> B)?

THE DECISIVE QUESTION (refines #327's open curvature)
-----------------------------------------------------
denken #327 (`eagle3_bi_reduction_floor`) priced the batch-invariant bf16
lm_head+attn reduction FLOOR at 9.841% of step -> DEAD (C) vs the lambda=1 budget
7.332% (#213), the binding term being the bf16 SDPA @ 34.9% BW (9.451% alone).
But #327 charged the FULL forgone slack and flagged ONE dominant uncertainty: the
SDPA penalty-law CURVATURE -- "an ideal kernel recovering > 25.5% of the SDPA
split-KV parallelism on NON-REDUCTION axes would flip the verdict to alive-needs-
kernel (B)". #327 parameterised that recovery by a free coefficient phi (floor
scales as floor_full * phi) and reported only the break-even phi* = 0.745 without
estimating where the ACTUAL geometry lands.

This card RESOLVES phi from the measured M=8 SDPA decode launch geometry. It does
NOT build a kernel (human-approval-gated); it asks: given the real GQA / head /
KV-split tiling on the A10G, how much of that forgone split-KV parallelism does a
DETERMINISTIC (fixed-split, ordered-combine) SDPA actually have to give up? If the
non-reduction axes ALONE fill the machine, phi is small -> floor collapses toward
lm_head-only (0.39%) -> lane (B)/(A). If the KV-sequence split is REQUIRED to fill
the 80 SMs, phi is large -> floor stays above budget -> (C).

THE PARTITION (reduction vs non-reduction parallelism)
------------------------------------------------------
vLLM's `unified_attention` (the deployed splitkv_verify_patch / PR #39 path) tiles
the M=8 verify attention as:
  * REDUCTION axis (FORGONE for determinism): the KV-sequence split. The 3D
    split-KV (FlashDecoding) path partitions the KV/context axis into
    NUM_PAR_SOFTMAX_SEGMENTS=16 segments computed in parallel, then merges them
    with an online-softmax `reduce_segments`. A deterministic reduction must fix
    the split count and order the combine -> it forgoes this axis's occupancy lift.
  * NON-REDUCTION axes (FREE to a deterministic schedule): the M query rows and
    the q-heads, tiled into the 2D launch grid. The natural kernel packs
    BLOCK_Q = BLOCK_M // (q_heads/kv_heads) = 16//4 = 4 query rows per CTA, giving
    total_num_q_blocks = M//BLOCK_Q + num_seqs = 8//4 + 1 = 3, times num_kv_heads=2
    -> only N_nonreduction = 6 CTAs. head_dim=256 is the QK^T CONTRACTION axis (a
    reduction), NOT a free-parallel axis.

OCCUPANCY -> phi (the geometric estimate)
-----------------------------------------
The A10G has 80 SMs (GA102, 80 enabled; the deployed splitkv_verify_patch states
"A10G's 80 SMs"; repo-measured star/gate cards report sm_count=80). The natural
non-reduction grid fills only 6 / 80 SMs. The 3D split-KV adds the 16-way
reduction split -> 6*16 = 96 CTAs, OVER-subscribing the 80 SMs (this is exactly
why PR #39 measured the 3D M=1 path at 12us vs the 2D verify path at 53us, 4.14x).
So the machine is filled BY the reduction split; forgoing it leaves ~92.5% of the
SMs idle:
  phi_occupancy = 1 - min(1, N_nonreduction / SMs) = 1 - 6/80 = 0.925   (headline)
Corroborated by the MEASURED 2D-vs-3D ratio (PR #39): the deterministic
(non-reduction-only) 2D path costs 53us, the adaptive split-KV path 12us, so the
realized forgone-parallelism fraction is 1 - 12/53 = 0.7736. Both >> the #327
break-even phi* = 0.745 -> (C) stands for the NATURAL kernel.

THE (B) DOOR (un-packing ceiling)
---------------------------------
A hand-written kernel could UN-PACK the non-reduction work: BLOCK_Q=1 + per-head
tiles expose up to M * num_q_heads = 8 * 8 = 64 independent (query-row, q-head)
tiles -- all non-reduction. 64 > the break-even CTA count (1-phi*)*80 = 20.4, so
the (B) door geometrically EXISTS: a custom kernel that fills >= 20.4 SMs from
non-reduction tiles alone pays phi_unpack = 1 - 64/80 = 0.20 < 0.745 -> revives.
But that kernel is precisely the UNBUILT, human-approval-gated artifact. The
NATURAL launch geometry (6 CTAs) does not reach it -> (C) for the deployed kernel.

floor_sdpa(phi) = SDPA_share * pi(phi),  pi(phi) = (1 - BW_SDPA) * phi
floor_combined(phi) = floor_full * phi   (the #327 convention; round-trips phi*=0.745)

WHAT THIS CARD DOES (CPU analytic over banked numbers; 0 GPU, 0 TPS)
-------------------------------------------------------------------
1. Reconstruct the M=8 SDPA decode launch geometry (GQA, head_dim, KV-bytes,
   2D/3D grid) and VERIFY it reproduces denken #291 / kanna #280's measured SDPA
   row (us, BW 34.9%, exposed 505.4us) to <=1e-6.
2. Partition parallelism: reduction (KV-split, forgone) vs non-reduction (M x
   q-heads, free). Report N_nonreduction, N_full_3d, the un-pack ceiling.
3. Occupancy phi: non-reduction CTAs vs 80 SMs. geometric_phi_estimate.
4. Map phi -> floor -> verdict vs the 7.332% budget; the phi* = 0.745 round-trip;
   whether the geometric phi lands above (C) or below (B/A) it.
5. Self-test (>=20 checks): geometry reproduces #291's exposed-us to tol; phi in
   [0,1]; verdict monotone in phi; round-trips #327's 9.841% floor at phi=1;
   round-trips phi* = 0.745; NaN-clean.

HONEST SCOPE
------------
0 TPS. BASELINE 481.53 unchanged. NO new GPU measurement, NO model forward, NO
served-file change, NO HF Job, NO submission, NO build, NO launch. This RESOLVES
the geometric value of #327's free recovery coefficient phi from banked launch
geometry; it is NOT a buildability proof. It REFINES #327's lower bracket; wirbel
#326 measures the config-only UPPER ceiling (complementary -- NOT re-measured
here). The custom un-packing kernel is UNBUILT + human-approval-gated. NOT a
launch. NOT a build. NOT a served-file change.

PRIMARY metric  phi_floor_self_test_passes
TEST    metric  geometric_phi_estimate
                + identity_lane_verdict_at_geometric_phi  {C_dead, B_alive_needs_kernel, A_alive_free}
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
# Imported anchors -- DO NOT re-derive. Import EXACTLY, UNCHANGED, with source.
# --------------------------------------------------------------------------- #
# denken #291 (`verify_compute_hideability`) / kanna #280 (`verify_step_component_
# roofline`, roofline.json) -- the MEASURED M=8 SDPA row of the A10G verify step.
# exposed_us == us_at_m8 * (1 - bw_util) is the measured BW-gap (#291) = the
# forgone-parallelism penalty model imported from #327.
SDPA_US_M8 = 776.2124633789062
SDPA_ROOFLINE_US = 270.7729066666667
SDPA_ROOFLINE_US_GQA = 70.72426666666667
SDPA_BW_UTIL = 0.34883864849061247
SDPA_PCT_OF_VERIFY = 14.513725794080164          # % of the 5348.13us full verify
SDPA_EXPOSED_US = 505.43955671223955             # above-roofline exposed slack
KV_BYTES_DENKEN = 160038912.0                    # full (un-GQA'd) KV bytes, M=8
KV_BYTES_GQA = 40009728.0                         # GQA KV bytes (group 4)
SDPA_TOTAL_BYTES = 162463744.0                    # KV + Q/out activations
SDPA_AI_FLOP_PER_BYTE = 7.880597014925373

# lm_head row (#291) -- near-roofline; the small FIXED locus term.
LM_HEAD_US_M8 = 126.136474609375
LM_HEAD_BW_UTIL = 0.8344417980018903
LM_HEAD_PCT_OF_VERIFY = 2.358516890006141
LM_HEAD_EXPOSED_US = 20.88292794270834

# io_residual / norms (#291): measured above-roofline slack 0.0us.
NORMS_EXPOSED_US = 0.0
NORMS_PCT_OF_VERIFY = 0.28874034208321014

TOTAL_VERIFY_US_291 = 5348.1268310546875

# kanna #280 roofline.json config (the byte/roofline geometry source).
A10G_BW_GBPS = 600.0
RIDGE_AI = 208.33333333333334
DEPLOYED_NUM_LAYERS = 37
ROOFLINE_CTX = 528
# kanna #280's verdict phi bracket (the fern #274 num_stages=2 REALIZATION
# haircut -- a DIFFERENT quantity from this card's determinism-recovery phi;
# imported only for context / cross-reference, NOT used in the floor math).
PHI_LO_280_REALIZATION = 0.125
PHI_HI_280_REALIZATION = 0.735

# --- deployed M=8 verify SDPA launch geometry (PR #279 / splitkv_verify_patch / PR #39) --- #
M = 8                                # verify rows = K_spec(7) + 1 chain token, conc=1
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS          # 4
HEAD_DIM = 256                       # sliding head_dim (QK^T contraction = reduction axis)
NUM_LAYERS = 37
CTX = 528                            # time-avg decode KV length (roofline.json ctx)
BF16_BYTES = 2
NUM_SEQS = 1                         # concurrency=1 single sequence
BLOCK_M = 16                         # vLLM Triton BLOCK_M
NUM_PAR_SOFTMAX_SEGMENTS = 16        # 3D split-KV reduction segments
MIN_LAUNCH_GRID_SIZE_2D = 128
# A10G SM count. Repo-measured + deployed-patch authoritative = 80 (GA102, 80
# enabled). The PR #332 instruction text says 84 (full-GA102 figure); reported as
# a parallel column -- the verdict is invariant in [80, 84].
A10G_SMS = 80
A10G_SMS_INSTRUCTION = 84

# Measured deterministic-vs-adaptive anchor (deployed splitkv_verify_patch / PR #39):
# 2D verify attention 53us vs identical-bytes M=1 3D split-KV 12us (4.14x).
T_2D_VERIFY_US_39 = 53.0
T_3D_SPLITKV_US_39 = 12.0
SPLITKV_SPEEDUP_39 = 4.14

# wirbel #213 lambda=1 identity-overhead budget (a PERCENT; bar as fraction /100).
# This is the NO-REGRESSION bar: the omega that drops the 520.953 ceiling back to
# ~the 481.53 baseline. #327's phi* = 0.745 break-even is against THIS budget.
BUDGET_LAMBDA1_PCT_213 = 7.331808522875782
BUDGET_LAMBDA1_FRAC_213 = BUDGET_LAMBDA1_PCT_213 / 100.0

# advisor #192 >500 directive: under #192 the identity lane must CLEAR 500, not just
# avoid regression. At central rho the ceiling is LAMBDA1_CEIL*(1-floor), so clearing
# 500 needs floor <= 1 - 500/520.953 = 4.022% -- STRICTLY TIGHTER than no-regression.
# (defined below once TARGET_TPS / LAMBDA1_CEIL exist, see BUDGET_500_*.)

# denken #327 floor (the card this one refines) -- imported EXACT for round-trip.
FLOOR_COMBINED_FULL_327 = 0.09841249119201488    # floor at phi=1
# SDPA-only floor at phi=1: derived from the #291 anchors (== exposed/total) so it
# cannot drift from the import. Equals SDPA_share * (1 - BW_SDPA) by construction.
FLOOR_SDPA_FULL_327 = SDPA_EXPOSED_US / TOTAL_VERIFY_US_291   # ~0.094507
RECOVERY_NEEDED_TO_REVIVE_327 = 0.2549920813842095   # 1 - phi*
PHI_STAR_327 = 1.0 - RECOVERY_NEEDED_TO_REVIVE_327    # 0.7450079186157905

# wirbel #293 step constants.
OFFICIAL_TPS = 481.53
LAMBDA1_CEIL = 520.9527323111674
K_CAL = 125.268
STEP_US = 1218.2
TARGET_TPS = 500.0

# advisor #192 >500 budget (operative bar): floor must be <= this to clear 500 at
# central rho.  1 - 500/520.953 = 0.04022 -- tighter than the 7.332% no-regression bar.
BUDGET_500_FRAC_192 = 1.0 - TARGET_TPS / LAMBDA1_CEIL          # 0.040227
BUDGET_500_PCT_192 = BUDGET_500_FRAC_192 * 100.0              # 4.0227%
# phi* crossings for BOTH budgets, under #327's whole-slack convention floor=0.09841*phi
# (the convention that makes #327's phi*=0.745 round-trip).  phi*_500 < phi*_noreg.
PHI_STAR_500_COMBINED = BUDGET_500_FRAC_192 / FLOOR_COMBINED_FULL_327   # ~0.4087
PHI_STAR_NOREG_COMBINED = BUDGET_LAMBDA1_FRAC_213 / FLOOR_COMBINED_FULL_327  # ~0.745 (== #327)
# lm-head-FIXED convention (advisor's exact formula 14.51%*0.651*phi + 0.39%): only the
# SDPA split-KV term scales with phi; the lm_head GEMM locus (0.39%) + norms (0.0) stay
# fixed (split-K / fusion, NOT the split-KV reduction this card resolves). Parallel column.
_FLOOR_LM_FIXED = (LM_HEAD_EXPOSED_US + NORMS_EXPOSED_US) / TOTAL_VERIFY_US_291  # ~0.0039
_SDPA_SLOPE = (SDPA_PCT_OF_VERIFY / 100.0) * (1.0 - SDPA_BW_UTIL)               # 14.51%*0.651
PHI_STAR_500_LMFIXED = (BUDGET_500_FRAC_192 - _FLOOR_LM_FIXED) / _SDPA_SLOPE        # ~0.384
PHI_STAR_NOREG_LMFIXED = (BUDGET_LAMBDA1_FRAC_213 - _FLOOR_LM_FIXED) / _SDPA_SLOPE  # ~0.734

TOL_EXACT = 1e-6
TOL_BUDGET = 1e-6
TOL_ROOFLINE = 1e-6


# --------------------------------------------------------------------------- #
# Penalty model (imported from #327): pi(u) = 1 - BW_util, scaled by recovery phi.
# --------------------------------------------------------------------------- #
def bw_gap_penalty(bw_util: float) -> float:
    """Per-kernel determinism penalty fraction at FULL forgone slack (phi=1)."""
    return float(max(0.0, min(1.0, 1.0 - bw_util)))


def deterministic_penalty(bw_util: float, phi: float) -> float:
    """pi(phi) = (1 - BW_util) * phi -- the share of the kernel's own time paid as
    forgone reduction parallelism when a deterministic schedule recovers fraction
    (1 - phi) of it on non-reduction axes. phi in [0, 1]."""
    return bw_gap_penalty(bw_util) * float(max(0.0, min(1.0, phi)))


# --------------------------------------------------------------------------- #
# Step 1+2: reconstruct geometry + partition reduction vs non-reduction axes.
# --------------------------------------------------------------------------- #
def reconstruct_geometry() -> dict[str, Any]:
    # ---- byte model (reproduces #291/#280 SDPA row from first principles) ---- #
    kv_bytes_denken = 2 * BF16_BYTES * CTX * NUM_Q_HEADS * HEAD_DIM * NUM_LAYERS  # K,V
    kv_bytes_gqa = kv_bytes_denken / GQA_GROUP
    act_bytes = 2 * M * NUM_Q_HEADS * HEAD_DIM * BF16_BYTES * NUM_LAYERS          # Q in, out
    total_bytes = kv_bytes_denken + act_bytes
    roofline_us = total_bytes / (A10G_BW_GBPS * 1.0e3)        # bytes / (600 GB/s) -> us
    bw_util = roofline_us / SDPA_US_M8
    exposed_us = SDPA_US_M8 * (1.0 - bw_util)

    byte_resid = {
        "kv_bytes_denken": abs(kv_bytes_denken - KV_BYTES_DENKEN),
        "kv_bytes_gqa": abs(kv_bytes_gqa - KV_BYTES_GQA),
        "total_bytes": abs(total_bytes - SDPA_TOTAL_BYTES),
        "roofline_us": abs(roofline_us - SDPA_ROOFLINE_US),
        "bw_util": abs(bw_util - SDPA_BW_UTIL),
        "exposed_us": abs(exposed_us - SDPA_EXPOSED_US),
    }
    geometry_reproduces_291 = max(byte_resid.values()) < TOL_ROOFLINE

    # ---- launch grid (vLLM unified_attention; PR #279 grid logic) ---- #
    num_queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS          # 4
    block_q = max(1, BLOCK_M // num_queries_per_kv)           # 4
    total_num_q_blocks = M // block_q + NUM_SEQS              # 8//4 + 1 = 3
    seq_threshold_3d = MIN_LAUNCH_GRID_SIZE_2D // NUM_KV_HEADS  # 64
    n_nonreduction = total_num_q_blocks * NUM_KV_HEADS                       # 2D grid = 6
    n_full_3d = total_num_q_blocks * NUM_KV_HEADS * NUM_PAR_SOFTMAX_SEGMENTS  # 3D = 96
    # un-pack ceiling: non-reduction axes are M query-rows x q-heads (head_dim is
    # the QK^T contraction = a reduction, excluded). BLOCK_Q=1 + per-head tiling.
    n_unpack_max = M * NUM_Q_HEADS                            # 64

    return {
        "kv_bytes_denken": kv_bytes_denken, "kv_bytes_gqa": kv_bytes_gqa,
        "act_bytes": act_bytes, "total_bytes": total_bytes,
        "roofline_us": roofline_us, "bw_util": bw_util, "exposed_us": exposed_us,
        "byte_resid": byte_resid, "geometry_reproduces_291": geometry_reproduces_291,
        "gqa_group": GQA_GROUP, "num_queries_per_kv": num_queries_per_kv,
        "block_q": block_q, "total_num_q_blocks": total_num_q_blocks,
        "seq_threshold_3d": seq_threshold_3d,
        "n_nonreduction_ctas_2d": n_nonreduction,
        "n_full_3d_ctas": n_full_3d, "n_unpack_max_tiles": n_unpack_max,
        "num_par_softmax_segments": NUM_PAR_SOFTMAX_SEGMENTS,
    }


# --------------------------------------------------------------------------- #
# Step 3: occupancy -> geometric phi.  phi = forgone-parallelism fraction = the
# share of the machine the non-reduction (deterministic) schedule leaves idle.
# --------------------------------------------------------------------------- #
def occupancy_phi(n_nonreduction: int, sms: int) -> float:
    """phi = 1 - min(1, N_nonreduction / SMs). Non-reduction CTAs alone fill
    N_nonreduction/SMs of the machine; the rest is fillable ONLY by the forgone
    KV-split reduction axis -> that idle fraction is the forgone parallelism."""
    fill = min(1.0, float(n_nonreduction) / float(sms))
    return float(1.0 - fill)


# --------------------------------------------------------------------------- #
# Step 4: phi -> floor -> three-way verdict.
# --------------------------------------------------------------------------- #
def floor_sdpa_of_phi(phi: float) -> float:
    """floor_sdpa(phi) = SDPA_share * pi(phi). SDPA_share = pct/100."""
    sdpa_share = SDPA_PCT_OF_VERIFY / 100.0
    return sdpa_share * deterministic_penalty(SDPA_BW_UTIL, phi)


def floor_combined_of_phi(phi: float) -> float:
    """#327 convention: the whole forgone slack scales by phi -> round-trips
    floor_full at phi=1 and phi* = budget/floor_full = 0.745."""
    return FLOOR_COMBINED_FULL_327 * float(max(0.0, min(1.0, phi)))


def floor_lmhead_fixed_of_phi(phi: float) -> float:
    """Alternative convention: only the SDPA split-KV term scales by phi; the
    near-roofline lm_head locus (0.39%) + norms (0.0) stay FIXED (they are GEMM
    split-K / fusion, not the split-KV reduction this card resolves)."""
    floor_lm = LM_HEAD_EXPOSED_US / TOTAL_VERIFY_US_291
    floor_norms = NORMS_EXPOSED_US / TOTAL_VERIFY_US_291
    return floor_sdpa_of_phi(phi) + floor_lm + floor_norms


def three_way_verdict(phi: float, c326_config_only: float | None) -> str:
    """At recovery coefficient phi:
    (C) floor_combined(phi) > budget                          -> DEAD even det. SDPA.
    (B) floor_combined(phi) <= budget < #326 config-only      -> alive, needs kernel.
    (A) floor_combined(phi) <= budget AND #326 <= budget      -> alive, free knobs.
    When #326 unknown, the <=budget branch is 'B_alive_needs_kernel' (conservative:
    the recovery itself requires the UNBUILT un-packing kernel)."""
    floor = floor_combined_of_phi(phi)
    if floor > BUDGET_LAMBDA1_FRAC_213 + TOL_EXACT:
        return "C_dead"
    if c326_config_only is not None and c326_config_only <= BUDGET_LAMBDA1_FRAC_213 + TOL_EXACT:
        return "A_alive_free"
    return "B_alive_needs_kernel"


def five_hundred_band(phi: float) -> str:
    """advisor #192 three-way band at recovery phi (whole-slack floor=0.09841*phi).
    The >500 gate and the no-regression gate answer DIFFERENT questions; a custom
    kernel can be 'alive' by #327's 74.5% break-even yet still land UNDER 500.
      GREEN_500              floor <= 4.022%               -> clears 500, lane USEFUL.
      YELLOW_trap_misses_500 4.022% < floor <= 7.332%      -> no-regression but <500
                             (the new failure mode: 'B alive' that is NOT a >500 win).
      RED_regression         floor > 7.332%                -> regress below 481.53."""
    floor = floor_combined_of_phi(phi)
    if floor <= BUDGET_500_FRAC_192 + TOL_EXACT:
        return "GREEN_500"
    if floor <= BUDGET_LAMBDA1_FRAC_213 + TOL_EXACT:
        return "YELLOW_trap_misses_500"
    return "RED_regression"


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(c326_config_only: float | None = None) -> dict[str, Any]:
    geo = reconstruct_geometry()
    budget = BUDGET_LAMBDA1_FRAC_213

    # ---------- step 3: geometric phi (occupancy) + measured corroboration ---------- #
    phi_occupancy = occupancy_phi(geo["n_nonreduction_ctas_2d"], A10G_SMS)      # 0.925
    phi_occupancy_84 = occupancy_phi(geo["n_nonreduction_ctas_2d"], A10G_SMS_INSTRUCTION)
    # measured deterministic-vs-adaptive realized forgone fraction (PR #39).
    phi_measured_2dv3d = 1.0 - (T_3D_SPLITKV_US_39 / T_2D_VERIFY_US_39)          # 0.7736
    # the un-pack ceiling phi a CUSTOM kernel could reach.
    phi_unpack_floor = occupancy_phi(geo["n_unpack_max_tiles"], A10G_SMS)       # 0.20 (naive)
    geometric_phi_estimate = phi_occupancy                                      # HEADLINE

    # break-even CTA count: non-reduction tiles needed to drop phi to phi*.
    breakeven_ctas = (1.0 - PHI_STAR_327) * A10G_SMS                            # 20.4
    unpack_clears_breakeven = bool(geo["n_unpack_max_tiles"] > breakeven_ctas)   # 64 > 20.4
    natural_clears_breakeven = bool(geo["n_nonreduction_ctas_2d"] > breakeven_ctas)  # 6 > 20.4 -> False

    # ---------- the DECISIVE refinement: the un-pack (B) door is a mirage ---------- #
    # The naive un-pack reading (phi_unpack=0.20 < phi*=0.745 -> reopens (B)) assumes
    # the SDPA slack is OCCUPANCY-bound, so filling 64/80 SMs from non-reduction tiles
    # recovers it. It does NOT. The #291 SDPA (505us exposed, 34.9% BW) is measured on
    # the DEPLOYED served path, which is the 3D split-KV (splitkv_verify_patch / PR #39
    # routes M=8 verify to 3D). That adaptive path ALREADY launches N_full_3d = 96 CTAs
    # > 80 SMs -- the machine is OCCUPANCY-SATURATED -- and STILL sits at 34.9% BW. So
    # the 505us slack is the low arithmetic-intensity attention floor (AI = 7.88 flop/
    # byte << ridge 208), NOT removable by occupancy. A DETERMINISTIC kernel forgoes
    # the KV-split (the reduction axis), so its max non-reduction occupancy is the
    # un-pack ceiling 64 CTAs -- which is BELOW the adaptive 96. It therefore CANNOT
    # exceed the saturated 3D kernel's occupancy, cannot beat 34.9% BW, and pays >= the
    # full slack -> phi_realizable >= 1. The (B) door is geometrically CLOSED.
    adaptive_saturates_machine = bool(geo["n_full_3d_ctas"] > A10G_SMS)          # 96 > 80
    unpack_below_adaptive = bool(geo["n_unpack_max_tiles"] < geo["n_full_3d_ctas"])  # 64 < 96
    slack_is_low_ai_not_occupancy = bool(
        adaptive_saturates_machine and SDPA_BW_UTIL < 0.5 and SDPA_AI_FLOP_PER_BYTE < RIDGE_AI)
    unpack_door_refuted_by_saturated_3d = bool(
        adaptive_saturates_machine and unpack_below_adaptive and slack_is_low_ai_not_occupancy)
    # the realizable deterministic-kernel phi: pays >= the full slack (phi>=1) when the
    # un-pack door is refuted; otherwise the naive occupancy ceiling.
    phi_realizable_lower_bound = 1.0 if unpack_door_refuted_by_saturated_3d else phi_unpack_floor

    # ---------- step 4: floor at the geometric phi + the break-even ---------- #
    floor_at_geo = floor_combined_of_phi(geometric_phi_estimate)               # 0.0910
    floor_sdpa_at_geo = floor_sdpa_of_phi(geometric_phi_estimate)
    floor_at_measured = floor_combined_of_phi(phi_measured_2dv3d)              # 0.0762
    floor_at_unpack = floor_combined_of_phi(phi_unpack_floor)                  # 0.0197
    floor_at_phi1 = floor_combined_of_phi(1.0)                                 # 0.09841 (327 round-trip)
    floor_sdpa_at_phi1 = floor_sdpa_of_phi(1.0)                                # 0.09451 (327 round-trip)
    floor_lmfixed_at_geo = floor_lmhead_fixed_of_phi(geometric_phi_estimate)

    # phi where floor_combined crosses budget (must round-trip #327's 0.745).
    phi_cross_budget = budget / FLOOR_COMBINED_FULL_327                        # 0.745008
    # SDPA-only crossing (secondary convention).
    phi_cross_budget_sdpa_only = budget / FLOOR_SDPA_FULL_327                  # 0.7758
    recovery_needed_to_revive = 1.0 - phi_cross_budget                        # 0.255

    geo_recovery_fraction = 1.0 - geometric_phi_estimate                      # 0.075
    measured_recovery_fraction = 1.0 - phi_measured_2dv3d                     # 0.226
    geo_recovers_enough = bool(geo_recovery_fraction >= recovery_needed_to_revive)  # 0.075>=0.255 -> False
    measured_recovers_enough = bool(measured_recovery_fraction >= recovery_needed_to_revive)  # False

    # ---------- verdict ---------- #
    verdict_code = three_way_verdict(geometric_phi_estimate, c326_config_only)
    verdict_at_measured = three_way_verdict(phi_measured_2dv3d, c326_config_only)
    # NOTE: verdict_at_unpack is the NAIVE-occupancy reading (refuted below). The
    # realizable verdict uses phi_realizable_lower_bound (>=1 -> C_dead).
    verdict_at_unpack_naive = three_way_verdict(phi_unpack_floor, c326_config_only)
    verdict_at_realizable = three_way_verdict(phi_realizable_lower_bound, c326_config_only)
    identity_lane_verdict_at_geometric_phi = verdict_code
    lane_dead_at_geo = verdict_code == "C_dead"
    lane_dead_at_realizable = verdict_at_realizable == "C_dead"

    # compliant ceiling TPS = lambda1_ceil * (1 - floor) | identity == 1.0.
    ceiling_at_geo = LAMBDA1_CEIL * (1.0 - floor_at_geo)                       # ~473
    ceiling_at_unpack = LAMBDA1_CEIL * (1.0 - floor_at_unpack)                 # ~511
    ceiling_at_geo_clears_500 = bool(ceiling_at_geo >= TARGET_TPS)
    ceiling_at_unpack_clears_500 = bool(ceiling_at_unpack >= TARGET_TPS)

    # ---------- advisor #192: TWO-budget three-way band (>500 vs no-regression) -------- #
    # The >500 gate (4.022%) is STRICTLY tighter than no-regression (7.332%). Report the
    # band at each phi so a 'B alive' (by #327's 74.5% break-even) that lands in the
    # YELLOW trap is NOT mistaken for a >500 win.
    band_at_geo = five_hundred_band(geometric_phi_estimate)                    # RED (9.10% > 7.33%)
    band_at_measured = five_hundred_band(phi_measured_2dv3d)                   # RED (7.61% > 7.33%)
    band_at_realizable = five_hundred_band(phi_realizable_lower_bound)         # RED (>=9.84%)
    band_at_unpack_naive = five_hundred_band(phi_unpack_floor)                 # GREEN (1.97%) -- mirage
    geometric_phi_clears_500_budget = bool(floor_at_geo <= BUDGET_500_FRAC_192 + TOL_EXACT)
    geometric_phi_clears_noregression_budget = bool(
        floor_at_geo <= BUDGET_LAMBDA1_FRAC_213 + TOL_EXACT)
    # no DEFENSIBLE phi (geometric / measured / realizable) clears 500; only the REFUTED
    # naive-un-pack mirage lands GREEN -> the >500 door is closed for every realizable schedule.
    no_defensible_phi_clears_500 = bool(
        not geometric_phi_clears_500_budget
        and floor_at_measured > BUDGET_500_FRAC_192 + TOL_EXACT
        and floor_combined_of_phi(phi_realizable_lower_bound) > BUDGET_500_FRAC_192 + TOL_EXACT)
    naive_unpack_would_clear_500 = bool(floor_at_unpack <= BUDGET_500_FRAC_192 + TOL_EXACT)

    # 'light' now reflects the OPERATIVE #192 band at the geometric phi (not just the
    # no-regression verdict): RED regression / YELLOW <500 trap / GREEN >500.
    light = {"RED_regression": "RED", "YELLOW_trap_misses_500": "YELLOW",
             "GREEN_500": "GREEN"}[band_at_geo]
    verdict = _verdict(geometric_phi_estimate, phi_measured_2dv3d, phi_cross_budget,
                       floor_at_geo, budget, ceiling_at_geo, lane_dead_at_geo,
                       geo["n_nonreduction_ctas_2d"], geo["n_unpack_max_tiles"],
                       geo["n_full_3d_ctas"], unpack_door_refuted_by_saturated_3d,
                       band_at_geo, BUDGET_500_FRAC_192)
    handoff = _handoff(geometric_phi_estimate, phi_measured_2dv3d, phi_cross_budget,
                       floor_at_geo, budget, geo["n_nonreduction_ctas_2d"],
                       geo["n_unpack_max_tiles"], geo["n_full_3d_ctas"], ceiling_at_geo,
                       unpack_door_refuted_by_saturated_3d, verdict_code,
                       band_at_geo, BUDGET_500_FRAC_192, PHI_STAR_500_COMBINED)

    return {
        "step1_geometry": {
            "source": "denken #291 verify_compute_hideability + kanna #280 roofline.json "
                      "(measured M=8 SDPA, A10G) + PR #279 grid logic + splitkv_verify_patch/PR #39",
            "M": M, "num_q_heads": NUM_Q_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "gqa_group": geo["gqa_group"], "head_dim": HEAD_DIM, "num_layers": NUM_LAYERS,
            "ctx": CTX, "kv_bytes_denken": geo["kv_bytes_denken"],
            "kv_bytes_gqa": geo["kv_bytes_gqa"], "act_bytes": geo["act_bytes"],
            "total_bytes": geo["total_bytes"], "roofline_us": geo["roofline_us"],
            "bw_util": geo["bw_util"], "exposed_us": geo["exposed_us"],
            "byte_resid_max": max(geo["byte_resid"].values()),
            "geometry_reproduces_291": geo["geometry_reproduces_291"],
            "a10g_sms": A10G_SMS, "a10g_sms_instruction": A10G_SMS_INSTRUCTION,
        },
        "step2_partition": {
            "reduction_axis": "KV-sequence split (NUM_PAR_SOFTMAX_SEGMENTS=16 online-"
                              "softmax segments) -- FORGONE for determinism (fixed split + "
                              "ordered combine).",
            "non_reduction_axes": "M query-rows x q-heads (BLOCK_Q tiling) -- FREE to a "
                                  "deterministic schedule. head_dim=256 is the QK^T "
                                  "contraction (reduction), excluded.",
            "block_q": geo["block_q"], "total_num_q_blocks": geo["total_num_q_blocks"],
            "n_nonreduction_ctas_2d": geo["n_nonreduction_ctas_2d"],
            "n_full_3d_ctas": geo["n_full_3d_ctas"],
            "n_unpack_max_tiles": geo["n_unpack_max_tiles"],
            "num_par_softmax_segments": geo["num_par_softmax_segments"],
            "seq_threshold_3d": geo["seq_threshold_3d"],
            "partition_product_check": geo["n_nonreduction_ctas_2d"]
            * geo["num_par_softmax_segments"] == geo["n_full_3d_ctas"],
        },
        "step3_occupancy_phi": {
            "model": "phi = 1 - min(1, N_nonreduction / SMs) = forgone-parallelism "
                     "fraction = share of the machine the deterministic (non-reduction) "
                     "schedule leaves idle.",
            "geometric_phi_estimate": geometric_phi_estimate,
            "phi_occupancy_80sms": phi_occupancy,
            "phi_occupancy_84sms": phi_occupancy_84,
            "phi_measured_2dv3d": phi_measured_2dv3d,
            "phi_unpack_floor_naive_occupancy": phi_unpack_floor,
            "phi_realizable_lower_bound": phi_realizable_lower_bound,
            "phi_in_unit_interval": bool(0.0 <= geometric_phi_estimate <= 1.0),
            "t_2d_verify_us_39": T_2D_VERIFY_US_39, "t_3d_splitkv_us_39": T_3D_SPLITKV_US_39,
            "splitkv_speedup_39": SPLITKV_SPEEDUP_39,
            "breakeven_ctas": breakeven_ctas,
            "natural_clears_breakeven": natural_clears_breakeven,
            "unpack_clears_breakeven_naive": unpack_clears_breakeven,
            "adaptive_saturates_machine": adaptive_saturates_machine,
            "unpack_below_adaptive": unpack_below_adaptive,
            "slack_is_low_ai_not_occupancy": slack_is_low_ai_not_occupancy,
            "unpack_door_refuted_by_saturated_3d": unpack_door_refuted_by_saturated_3d,
            "sdpa_ai_flop_per_byte": SDPA_AI_FLOP_PER_BYTE, "ridge_ai": RIDGE_AI,
        },
        "step4_floor_verdict": {
            "phi_star_breakeven_combined": phi_cross_budget,
            "phi_star_breakeven_sdpa_only": phi_cross_budget_sdpa_only,
            "phi_star_327_roundtrip_resid": abs(phi_cross_budget - PHI_STAR_327),
            "recovery_fraction_needed_to_revive": recovery_needed_to_revive,
            "geo_recovery_fraction": geo_recovery_fraction,
            "measured_recovery_fraction": measured_recovery_fraction,
            "geo_recovers_enough_for_B": geo_recovers_enough,
            "measured_recovers_enough_for_B": measured_recovers_enough,
            "floor_at_geometric_phi": floor_at_geo,
            "floor_sdpa_at_geometric_phi": floor_sdpa_at_geo,
            "floor_lmfixed_at_geometric_phi": floor_lmfixed_at_geo,
            "floor_at_measured_phi": floor_at_measured,
            "floor_at_unpack_phi": floor_at_unpack,
            "floor_at_phi1_roundtrips_327": floor_at_phi1,
            "floor_sdpa_at_phi1_roundtrips_327": floor_sdpa_at_phi1,
            "floor_327_roundtrip_resid": abs(floor_at_phi1 - FLOOR_COMBINED_FULL_327),
            "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
            "budget_lambda1_frac_213": budget,
            # advisor #192: the OPERATIVE >500 budget (tighter than no-regression).
            "budget_500_pct_192": BUDGET_500_PCT_192,
            "budget_500_frac_192": BUDGET_500_FRAC_192,
            "phi_star_500_combined": PHI_STAR_500_COMBINED,
            "phi_star_noregression_combined": PHI_STAR_NOREG_COMBINED,
            "phi_star_500_lmfixed": PHI_STAR_500_LMFIXED,
            "phi_star_noregression_lmfixed": PHI_STAR_NOREG_LMFIXED,
            "phi_star_500_below_noregression": bool(
                PHI_STAR_500_COMBINED < PHI_STAR_NOREG_COMBINED),
            # the two-budget three-way band per advisor.
            "identity_half_500_band_at_geometric_phi": band_at_geo,
            "identity_half_500_band_at_measured_phi": band_at_measured,
            "identity_half_500_band_at_realizable_phi": band_at_realizable,
            "identity_half_500_band_at_unpack_naive_phi": band_at_unpack_naive,
            "geometric_phi_clears_500_budget": geometric_phi_clears_500_budget,
            "geometric_phi_clears_noregression_budget": geometric_phi_clears_noregression_budget,
            "no_defensible_phi_clears_500": no_defensible_phi_clears_500,
            "naive_unpack_would_clear_500": naive_unpack_would_clear_500,
            "identity_lane_verdict_at_geometric_phi": identity_lane_verdict_at_geometric_phi,
            "verdict_at_measured_phi": verdict_at_measured,
            "verdict_at_unpack_naive_phi": verdict_at_unpack_naive,
            "verdict_at_realizable_phi": verdict_at_realizable,
            "lane_dead_at_geometric_phi": lane_dead_at_geo,
            "lane_dead_at_realizable_phi": lane_dead_at_realizable,
            "verdict_light": light,
            "compliant_ceiling_tps_at_geo": ceiling_at_geo,
            "compliant_ceiling_tps_at_unpack": ceiling_at_unpack,
            "ceiling_at_geo_clears_500": ceiling_at_geo_clears_500,
            "ceiling_at_unpack_clears_500": ceiling_at_unpack_clears_500,
            "c326_config_only_ceiling": c326_config_only,
        },
        "context": {
            "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS,
            "lambda1_ceil": LAMBDA1_CEIL, "k_cal": K_CAL, "step_us": STEP_US,
            "floor_combined_full_327": FLOOR_COMBINED_FULL_327,
            "phi_star_327": PHI_STAR_327,
            "phi_lo_280_realization": PHI_LO_280_REALIZATION,
            "phi_hi_280_realization": PHI_HI_280_REALIZATION,
            "phi_280_is_different_quantity": "kanna #280's phi_lo/phi_hi is the fern #274 "
                                             "num_stages=2 REALIZATION haircut, NOT this "
                                             "card's determinism-recovery phi.",
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(phi_geo: float, phi_meas: float, phi_star: float, floor_geo: float,
             budget: float, ceiling: float, dead: bool, n_nonred: int, n_unpack: int,
             n_full_3d: int, refuted: bool, band: str, budget_500: float) -> str:
    head = ("IDENTITY-LANE-DEAD-AT-GEOMETRIC-PHI" if dead
            else "IDENTITY-LANE-ALIVE-AT-GEOMETRIC-PHI")
    door = ("the (B) door is geometrically CLOSED" if refuted
            else "the (B) door may be open via a custom kernel")
    band_txt = {"RED_regression": "RED (regress below the 481.53 baseline)",
                "YELLOW_trap_misses_500": "YELLOW (no-regression but UNDER 500 -- the trap)",
                "GREEN_500": "GREEN (clears 500)"}[band]
    return (f"{head}: the M=8 SDPA decode geometry fills only {n_nonred}/{A10G_SMS} SMs "
            f"from non-reduction (M x q-head) axes, so a deterministic (fixed-split, "
            f"ordered-combine) SDPA forgoes geometric_phi = {phi_geo*100:.1f}% of the "
            f"split-KV parallelism (measured 2D-vs-3D corroboration {phi_meas*100:.1f}%), "
            f"both ABOVE the #327 break-even phi* = {phi_star*100:.1f}% -> floor "
            f"{floor_geo*100:.3f}% > budget {budget*100:.3f}% -> {'DEAD (C)' if dead else 'alive'}. "
            f"#192 TWO-BUDGET BAND = {band_txt}: the floor clears NEITHER the operative >500 "
            f"bar ({budget_500*100:.3f}%) NOR the no-regression bar ({budget*100:.3f}%), so "
            f"even the BEST defensible reading is not a >500 win. "
            f"DECISIVE: {door} -- the deployed SDPA's exposed slack is measured on the 3D "
            f"split-KV path, which ALREADY launches {n_full_3d} CTAs > {A10G_SMS} SMs "
            f"(occupancy-SATURATED) yet stays at 34.9% BW, so the slack is the low-AI "
            f"attention floor (AI 7.88 << ridge 208), NOT occupancy-removable. A "
            f"deterministic kernel's un-pack ceiling ({n_unpack} tiles) is BELOW the "
            f"adaptive {n_full_3d}, so it cannot exceed that saturated occupancy -> pays "
            f">= the full slack -> phi_realizable >= 1. The naive un-pack reading "
            f"(phi={(1-min(1,n_unpack/A10G_SMS))*100:.0f}% < phi*) -- the ONLY reading that "
            f"would land GREEN-for-500 -- is the refuted mirage. Compliant "
            f"ceiling {ceiling:.0f} TPS (< 500). 0 TPS; analytic; NOT a launch.")


def _handoff(phi_geo: float, phi_meas: float, phi_star: float, floor_geo: float,
             budget: float, n_nonred: int, n_unpack: int, n_full_3d: int,
             ceiling_geo: float, refuted: bool, verdict_code: str,
             band: str, budget_500: float, phi_star_500: float) -> str:
    state = {"C_dead": "DEAD (C)", "B_alive_needs_kernel": "alive-needs-kernel (B)",
             "A_alive_free": "alive-with-knobs (A)"}[verdict_code]
    band_txt = {"RED_regression": "RED", "YELLOW_trap_misses_500": "YELLOW",
                "GREEN_500": "GREEN"}[band]
    return (
        f"resolving #327's free recovery coefficient phi from the measured M=8 SDPA "
        f"launch geometry: the non-reduction (M={M} x {NUM_Q_HEADS} q-head) axes fill "
        f"only {n_nonred}/{A10G_SMS} SMs, so a deterministic SDPA forgoes geometric_phi "
        f"= {phi_geo*100:.1f}% of the split-KV parallelism (PR #39's 2D-vs-3D ratio "
        f"corroborates at {phi_meas*100:.1f}%), both ABOVE the #327 break-even "
        f"phi* = {phi_star*100:.1f}% -> identity lane {state}: floor {floor_geo*100:.2f}% "
        f"> budget {budget*100:.2f}%, compliant ceiling {ceiling_geo:.0f} TPS. UNDER THE "
        f"#192 >500 DIRECTIVE this is reported against TWO budgets: the floor must clear the "
        f"OPERATIVE >500 bar ({budget_500*100:.3f}%, phi*_500 = {phi_star_500*100:.1f}%), "
        f"STRICTLY tighter than the {budget*100:.2f}% no-regression bar (phi* = "
        f"{phi_star*100:.1f}%) -> #192 band = {band_txt}: the geometric floor clears NEITHER, "
        f"so the identity half is a >500 RED, not merely a no-regression miss (the YELLOW "
        f"trap -- a 'B alive' under 500 -- is documented but the geometry does not even reach "
        f"it). This CLOSES "
        f"#327's open curvature toward (C) rather than reopening (B): the naive un-pack "
        f"door (recover the slack by filling {n_unpack}/{A10G_SMS} SMs from non-reduction "
        f"tiles) is REFUTED={refuted} -- the deployed SDPA slack is measured on the 3D "
        f"split-KV path, which already OVER-subscribes the machine ({n_full_3d} CTAs > "
        f"{A10G_SMS} SMs) yet still sits at 34.9% BW, proving the slack is the low arithmetic-"
        f"intensity floor (AI 7.88 << ridge 208), not occupancy. A deterministic kernel's "
        f"un-pack ceiling ({n_unpack} < adaptive {n_full_3d}) cannot exceed that saturated "
        f"occupancy, so phi_realizable >= 1 and the lane stays (C) for EVERY realizable "
        f"deterministic schedule. This pins #327's lower bracket from geometry (vs wirbel "
        f"#326's config-only UPPER ceiling, not re-measured here); no kernel is built here "
        f"(the custom SDPA reduction stays UNBUILT + human-approval-gated regardless). "
        f"0 TPS; analytic over banked numbers; BASELINE 481.53 "
        f"unchanged. NOT a launch. NOT a build. NOT a served-file change."
    )


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2 = syn["step1_geometry"], syn["step2_partition"]
    s3, s4 = syn["step3_occupancy_phi"], syn["step4_floor_verdict"]
    checks: dict[str, bool] = {}

    # (a) geometry reproduces #291/#280's measured SDPA row to tol.
    checks["a_geometry_reproduces_291"] = bool(s1["geometry_reproduces_291"])
    checks["a_byte_resid_below_tol"] = bool(s1["byte_resid_max"] < TOL_ROOFLINE)
    checks["a_kv_bytes_denken_exact"] = abs(s1["kv_bytes_denken"] - KV_BYTES_DENKEN) < 1.0
    checks["a_kv_bytes_gqa_exact"] = abs(s1["kv_bytes_gqa"] - KV_BYTES_GQA) < 1.0
    checks["a_total_bytes_exact"] = abs(s1["total_bytes"] - SDPA_TOTAL_BYTES) < 1.0
    checks["a_roofline_us_exact"] = abs(s1["roofline_us"] - SDPA_ROOFLINE_US) < TOL_ROOFLINE
    checks["a_bw_util_reproduces_349"] = abs(s1["bw_util"] - SDPA_BW_UTIL) < TOL_ROOFLINE
    checks["a_exposed_us_reproduces_505"] = abs(s1["exposed_us"] - SDPA_EXPOSED_US) < TOL_ROOFLINE
    checks["a_gqa_group_is_4"] = s1["gqa_group"] == 4

    # (b) launch grid partition (reduction vs non-reduction) is consistent.
    checks["b_block_q_is_4"] = s2["block_q"] == 4
    checks["b_total_q_blocks_is_3"] = s2["total_num_q_blocks"] == 3
    checks["b_n_nonreduction_is_6"] = s2["n_nonreduction_ctas_2d"] == 6
    checks["b_n_full_3d_is_96"] = s2["n_full_3d_ctas"] == 96
    checks["b_partition_product_consistent"] = bool(s2["partition_product_check"])
    checks["b_n_unpack_is_64"] = s2["n_unpack_max_tiles"] == 64
    checks["b_segments_is_16"] = s2["num_par_softmax_segments"] == 16

    # (c) geometric phi: well-formed, headline = occupancy, in [0,1].
    checks["c_phi_in_unit_interval"] = bool(s3["phi_in_unit_interval"])
    checks["c_geo_phi_is_occupancy"] = abs(
        s3["geometric_phi_estimate"] - s3["phi_occupancy_80sms"]) < TOL_EXACT
    checks["c_phi_occupancy_is_0p925"] = abs(s3["phi_occupancy_80sms"] - 0.925) < 1e-9
    checks["c_phi_measured_is_0p7736"] = abs(s3["phi_measured_2dv3d"] - (1 - 12/53)) < 1e-9
    checks["c_phi_unpack_is_0p20"] = abs(s3["phi_unpack_floor_naive_occupancy"] - 0.20) < 1e-9
    # SM-count invariance: verdict-relevant phi >> break-even at BOTH 80 and 84.
    checks["c_phi_sminvariant_above_breakeven"] = bool(
        s3["phi_occupancy_80sms"] > syn["step4_floor_verdict"]["phi_star_breakeven_combined"]
        and s3["phi_occupancy_84sms"] > syn["step4_floor_verdict"]["phi_star_breakeven_combined"])

    # (d) break-even CTA arithmetic + naive un-pack door.
    checks["d_breakeven_ctas_is_20p4"] = abs(s3["breakeven_ctas"] - (1 - PHI_STAR_327) * 80) < 1e-9
    checks["d_natural_below_breakeven"] = bool(not s3["natural_clears_breakeven"])  # 6 < 20.4
    checks["d_unpack_above_breakeven_naive"] = bool(s3["unpack_clears_breakeven_naive"])  # 64 > 20.4

    # (e) floor mapping round-trips #327 at phi=1 (combined AND sdpa-only).
    checks["e_floor_roundtrips_327_at_phi1"] = abs(
        s4["floor_at_phi1_roundtrips_327"] - FLOOR_COMBINED_FULL_327) < TOL_EXACT
    checks["e_floor_327_resid_below_tol"] = bool(s4["floor_327_roundtrip_resid"] < TOL_EXACT)
    checks["e_floor_full_is_9p841"] = abs(s4["floor_at_phi1_roundtrips_327"] - 0.09841249) < 1e-6
    checks["e_floor_sdpa_full_is_9p451"] = abs(
        s4["floor_sdpa_at_phi1_roundtrips_327"] - 0.0945069) < 1e-5

    # (f) phi* break-even round-trips #327's 0.745 (the "74.5% break-even").
    checks["f_phistar_roundtrips_327"] = abs(
        s4["phi_star_breakeven_combined"] - PHI_STAR_327) < TOL_EXACT
    checks["f_phistar_is_0p745"] = abs(s4["phi_star_breakeven_combined"] - 0.745008) < 1e-5
    checks["f_recovery_needed_is_0p255"] = abs(
        s4["recovery_fraction_needed_to_revive"] - RECOVERY_NEEDED_TO_REVIVE_327) < TOL_EXACT

    # (g) verdict monotone in phi: floor strictly increasing -> C-set is an upper interval.
    phis = [0.0, 0.20, phis_breakeven(s4), 0.7736, 0.925, 1.0]
    floors = [floor_combined_of_phi(p) for p in phis]
    checks["g_floor_monotone_in_phi"] = bool(
        all(floors[i] <= floors[i + 1] + 1e-15 for i in range(len(floors) - 1)))
    checks["g_verdict_monotone_C_is_upper_interval"] = _verdict_monotone(s4["c326_config_only_ceiling"])

    # (h) the decisive answer: geometric + measured phi land ABOVE break-even -> (C).
    checks["h_geo_phi_above_breakeven_dead"] = bool(
        s3["geometric_phi_estimate"] > s4["phi_star_breakeven_combined"]
        and s4["identity_lane_verdict_at_geometric_phi"] == "C_dead")
    checks["h_measured_phi_above_breakeven_dead"] = bool(
        s3["phi_measured_2dv3d"] > s4["phi_star_breakeven_combined"]
        and s4["verdict_at_measured_phi"] == "C_dead")
    # the NAIVE un-pack reading would revive (B) -- this is the mirage refuted in (k).
    checks["h_unpack_naive_would_revive_B"] = bool(
        s3["phi_unpack_floor_naive_occupancy"] < s4["phi_star_breakeven_combined"]
        and s4["verdict_at_unpack_naive_phi"] in ("B_alive_needs_kernel", "A_alive_free"))
    checks["h_geo_recovers_too_little"] = bool(not s4["geo_recovers_enough_for_B"])
    checks["h_floor_at_geo_busts_budget"] = bool(
        s4["floor_at_geometric_phi"] > s4["budget_lambda1_frac_213"])
    checks["h_ceiling_at_geo_below_500"] = bool(not s4["ceiling_at_geo_clears_500"])

    # (k) the DECISIVE refinement: the un-pack (B) door is a mirage; realizable phi -> (C).
    checks["k_adaptive_saturates_machine"] = bool(s3["adaptive_saturates_machine"])  # 96 > 80
    checks["k_unpack_below_adaptive"] = bool(s3["unpack_below_adaptive"])             # 64 < 96
    checks["k_slack_is_low_ai_not_occupancy"] = bool(s3["slack_is_low_ai_not_occupancy"])
    checks["k_unpack_door_refuted"] = bool(s3["unpack_door_refuted_by_saturated_3d"])
    checks["k_realizable_phi_at_least_1"] = bool(s3["phi_realizable_lower_bound"] >= 1.0 - TOL_EXACT)
    checks["k_realizable_verdict_is_C"] = bool(
        s4["verdict_at_realizable_phi"] == "C_dead" and s4["lane_dead_at_realizable_phi"])
    checks["k_ai_below_ridge"] = bool(s3["sdpa_ai_flop_per_byte"] < s3["ridge_ai"])

    # (l) advisor #192 TWO-budget three-way band: >500 (4.022%) tighter than no-reg (7.332%).
    checks["l_budget_500_is_4p022"] = abs(s4["budget_500_pct_192"] - 4.022) < 5e-3
    checks["l_budget_500_tighter_than_noregression"] = bool(
        s4["budget_500_frac_192"] < s4["budget_lambda1_frac_213"])
    # the >500 crossing must be BELOW the no-regression crossing (tighter bar -> lower phi*).
    checks["l_phi_star_500_below_noregression"] = bool(s4["phi_star_500_below_noregression"])
    checks["l_phi_star_500_combined_is_0p409"] = abs(s4["phi_star_500_combined"] - 0.4087) < 1e-3
    # advisor's exact lm-fixed formula 14.51%*0.651*phi + 0.39% = 4.022% -> phi ~ 0.384.
    checks["l_phi_star_500_lmfixed_is_0p384"] = abs(s4["phi_star_500_lmfixed"] - 0.3843) < 2e-3
    # at the geometric phi the floor busts BOTH budgets -> RED band, neither bool clears.
    checks["l_geo_clears_neither_budget"] = bool(
        not s4["geometric_phi_clears_500_budget"]
        and not s4["geometric_phi_clears_noregression_budget"])
    checks["l_geo_band_is_red"] = bool(
        s4["identity_half_500_band_at_geometric_phi"] == "RED_regression")
    checks["l_measured_band_is_red"] = bool(
        s4["identity_half_500_band_at_measured_phi"] == "RED_regression")
    checks["l_realizable_band_is_red"] = bool(
        s4["identity_half_500_band_at_realizable_phi"] == "RED_regression")
    # no DEFENSIBLE phi clears 500; only the REFUTED naive-un-pack mirage lands GREEN.
    checks["l_no_defensible_phi_clears_500"] = bool(s4["no_defensible_phi_clears_500"])
    checks["l_naive_unpack_mirage_is_green"] = bool(
        s4["identity_half_500_band_at_unpack_naive_phi"] == "GREEN_500"
        and s4["naive_unpack_would_clear_500"])
    # band is monotone: GREEN (low phi) -> YELLOW -> RED (high phi), no oscillation.
    checks["l_band_monotone_in_phi"] = _band_monotone()

    # (i) budget imported exact (#213) + constants.
    checks["i_budget_imported_exact"] = abs(
        s4["budget_lambda1_pct_213"] - 7.331808522875782) < TOL_BUDGET
    checks["i_constants_exact"] = bool(
        OFFICIAL_TPS == 481.53 and abs(LAMBDA1_CEIL - 520.9527323111674) < TOL_EXACT
        and K_CAL == 125.268 and TOTAL_VERIFY_US_291 == 5348.1268310546875
        and A10G_SMS == 80 and NUM_Q_HEADS == 8 and NUM_KV_HEADS == 2 and HEAD_DIM == 256)

    # (j) NaN-clean over the reported scalars.
    scalars = [s3["geometric_phi_estimate"], s3["phi_measured_2dv3d"],
               s3["phi_unpack_floor_naive_occupancy"], s3["phi_realizable_lower_bound"],
               s3["breakeven_ctas"], s4["floor_at_geometric_phi"],
               s4["floor_at_phi1_roundtrips_327"], s4["phi_star_breakeven_combined"],
               s4["compliant_ceiling_tps_at_geo"]]
    checks["j_nan_clean"] = all(math.isfinite(float(x)) for x in scalars)

    # the leg carries the 0-TPS + analytic + scope caveats.
    hl = syn["handoff_line"]
    checks["j_carries_caveats"] = bool(
        "0 TPS" in hl and "analytic" in hl and "NOT a launch" in hl
        and "NOT a build" in hl and "human-approval-gated" in hl and "UNBUILT" in hl)

    gate = bool(all(checks.values()))
    return {"phi_floor_self_test_passes": gate, "checks": checks}


def phis_breakeven(s4: dict[str, Any]) -> float:
    return float(s4["phi_star_breakeven_combined"])


def _verdict_monotone(c326: float | None) -> bool:
    """Over a phi grid, C_dead is an UPPER interval (phi > phi*): once dead, stays
    dead as phi rises; once alive, stays alive as phi falls. No oscillation."""
    grid = [i / 40.0 for i in range(41)]
    codes = [three_way_verdict(p, c326) for p in grid]
    dead = [c == "C_dead" for c in codes]
    # find first dead; everything from there up must be dead; everything below alive.
    if True not in dead:
        return True
    first = dead.index(True)
    return all(dead[first:]) and not any(dead[:first])


def _band_monotone() -> bool:
    """Over a phi grid the #192 band rank must be NON-DECREASING in phi:
    GREEN_500 (0) -> YELLOW_trap_misses_500 (1) -> RED_regression (2). No oscillation
    (floor=0.09841*phi is monotone, so the band can only worsen as phi rises)."""
    rank = {"GREEN_500": 0, "YELLOW_trap_misses_500": 1, "RED_regression": 2}
    grid = [i / 40.0 for i in range(41)]
    ranks = [rank[five_hundred_band(p)] for p in grid]
    return all(b >= a for a, b in zip(ranks, ranks[1:]))


def _band_code(band: str) -> int:
    """wandb-friendly band code: GREEN_500=2 (best) .. RED_regression=0 (worst)."""
    return {"GREEN_500": 2, "YELLOW_trap_misses_500": 1, "RED_regression": 0}[band]


# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: Any, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict, st: dict) -> None:
    s1, s2 = syn["step1_geometry"], syn["step2_partition"]
    s3, s4 = syn["step3_occupancy_phi"], syn["step4_floor_verdict"]
    print("\n" + "=" * 94, flush=True)
    print("EAGLE-3 DETERMINISTIC-SDPA phi-FLOOR (PR #332, denken) -- 0 GPU, 0 TPS", flush=True)
    print("=" * 94, flush=True)
    print("  (1) GEOMETRY  (denken #291 / kanna #280 measured M=8 SDPA, A10G; PR #279 grid)",
          flush=True)
    print(f"      GQA {NUM_Q_HEADS}q/{NUM_KV_HEADS}kv (group {s1['gqa_group']})  head_dim {HEAD_DIM}  "
          f"layers {NUM_LAYERS}  ctx {CTX}  SMs {s1['a10g_sms']} (instr {s1['a10g_sms_instruction']})",
          flush=True)
    print(f"      kv_bytes {s1['kv_bytes_denken']:.0f} (gqa {s1['kv_bytes_gqa']:.0f})  "
          f"roofline {s1['roofline_us']:.3f}us  BW {s1['bw_util']*100:.1f}%  "
          f"exposed {s1['exposed_us']:.2f}us", flush=True)
    print(f"      reproduces #291 to <=1e-6: {s1['geometry_reproduces_291']}  "
          f"(byte_resid_max {s1['byte_resid_max']:.2e})", flush=True)
    print("-" * 94, flush=True)
    print("  (2) PARTITION  reduction (KV-split, FORGONE) vs non-reduction (M x q-head, FREE)",
          flush=True)
    print(f"      2D non-reduction grid = {s2['total_num_q_blocks']} q-blocks x {NUM_KV_HEADS} kv "
          f"= {s2['n_nonreduction_ctas_2d']} CTAs   3D split-KV (+{s2['num_par_softmax_segments']} "
          f"segments) = {s2['n_full_3d_ctas']} CTAs", flush=True)
    print(f"      un-pack ceiling (BLOCK_Q=1 x per-head) = M x q-heads = {s2['n_unpack_max_tiles']} "
          f"tiles  (head_dim = QK^T reduction, excluded)", flush=True)
    print("-" * 94, flush=True)
    print("  (3) OCCUPANCY -> phi   phi = 1 - min(1, N_nonreduction / SMs)", flush=True)
    print(f"      geometric_phi_estimate = {s3['geometric_phi_estimate']*100:.1f}%  "
          f"(80 SMs; 84 SMs -> {s3['phi_occupancy_84sms']*100:.1f}%)", flush=True)
    print(f"      measured 2D-vs-3D (PR #39: {s3['t_2d_verify_us_39']:.0f}us vs "
          f"{s3['t_3d_splitkv_us_39']:.0f}us) -> phi {s3['phi_measured_2dv3d']*100:.1f}%   "
          f"naive un-pack ceiling -> phi {s3['phi_unpack_floor_naive_occupancy']*100:.1f}%",
          flush=True)
    print(f"      break-even {s3['breakeven_ctas']:.1f} CTAs:  natural {s2['n_nonreduction_ctas_2d']} "
          f"clears={s3['natural_clears_breakeven']}   un-pack {s2['n_unpack_max_tiles']} "
          f"clears={s3['unpack_clears_breakeven_naive']} (naive)", flush=True)
    print("-" * 94, flush=True)
    print("  (3b) DECISIVE: is the naive un-pack (B) door real?  -> REFUTED by saturated 3D",
          flush=True)
    print(f"      adaptive 3D launches {s2['n_full_3d_ctas']} CTAs > {s1['a10g_sms']} SMs "
          f"(saturates={s3['adaptive_saturates_machine']}) yet BW {s1['bw_util']*100:.1f}% "
          f"(AI {s3['sdpa_ai_flop_per_byte']:.2f} << ridge {s3['ridge_ai']:.0f})", flush=True)
    print(f"      => slack is low-AI floor, NOT occupancy-removable; un-pack {s2['n_unpack_max_tiles']} "
          f"< adaptive {s2['n_full_3d_ctas']} ({s3['unpack_below_adaptive']}) -> phi_realizable "
          f">= {s3['phi_realizable_lower_bound']:.2f}  (door refuted={s3['unpack_door_refuted_by_saturated_3d']})",
          flush=True)
    print("-" * 94, flush=True)
    print("  (4) phi -> FLOOR -> VERDICT vs lambda=1 BUDGET (#213)", flush=True)
    print(f"      phi* break-even = {s4['phi_star_breakeven_combined']*100:.1f}% "
          f"(round-trips #327; resid {s4['phi_star_327_roundtrip_resid']:.2e})  -> need "
          f"> {s4['recovery_fraction_needed_to_revive']*100:.1f}% recovery", flush=True)
    print(f"      floor@geo_phi = {s4['floor_at_geometric_phi']*100:.3f}%  "
          f"floor@measured = {s4['floor_at_measured_phi']*100:.3f}%  "
          f"floor@unpack = {s4['floor_at_unpack_phi']*100:.3f}%   vs budget "
          f"{s4['budget_lambda1_pct_213']:.3f}%", flush=True)
    print(f"      floor@phi=1 = {s4['floor_at_phi1_roundtrips_327']*100:.3f}% (round-trips #327 "
          f"9.841%)   compliant ceiling {s4['compliant_ceiling_tps_at_geo']:.0f} TPS "
          f"(unpack {s4['compliant_ceiling_tps_at_unpack']:.0f})", flush=True)
    print(f"      identity_lane_verdict_at_geometric_phi = "
          f"{s4['identity_lane_verdict_at_geometric_phi']}  [{s4['verdict_light']}]  "
          f"(measured -> {s4['verdict_at_measured_phi']}; realizable -> "
          f"{s4['verdict_at_realizable_phi']}; naive-unpack -> {s4['verdict_at_unpack_naive_phi']})",
          flush=True)
    print("-" * 94, flush=True)
    print("  (4b) advisor #192 TWO-BUDGET band: >500 ({:.3f}%) tighter than no-reg ({:.3f}%)"
          .format(s4['budget_500_pct_192'], s4['budget_lambda1_pct_213']), flush=True)
    print(f"      phi*_500 = {s4['phi_star_500_combined']*100:.1f}% (lm-fixed "
          f"{s4['phi_star_500_lmfixed']*100:.1f}%)  <  phi*_no-reg = "
          f"{s4['phi_star_noregression_combined']*100:.1f}%", flush=True)
    print(f"      band@geo_phi = {s4['identity_half_500_band_at_geometric_phi']}  "
          f"(measured -> {s4['identity_half_500_band_at_measured_phi']}; realizable -> "
          f"{s4['identity_half_500_band_at_realizable_phi']}; naive-unpack-mirage -> "
          f"{s4['identity_half_500_band_at_unpack_naive_phi']})", flush=True)
    print(f"      clears_500 = {s4['geometric_phi_clears_500_budget']}  "
          f"clears_no-regression = {s4['geometric_phi_clears_noregression_budget']}  "
          f"no_defensible_phi_clears_500 = {s4['no_defensible_phi_clears_500']}", flush=True)
    print("-" * 94, flush=True)
    print(f"  PRIMARY phi_floor_self_test_passes = {st['phi_floor_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 94, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[sdpa-phi-floor] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s2 = syn["step1_geometry"], syn["step2_partition"]
    s3, s4 = syn["step3_occupancy_phi"], syn["step4_floor_verdict"]
    run = init_wandb_run(
        job_type="eagle3-sdpa-phi-floor",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-sdpa-phi-floor", "deterministic-sdpa", "split-kv", "occupancy",
              "greedy-identity", "kernel-floor", "validity", "zero-tps"],
        config={
            "pr": 332, "analysis_only": True,
            "total_verify_us_291": TOTAL_VERIFY_US_291,
            "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
            "floor_combined_full_327": FLOOR_COMBINED_FULL_327,
            "phi_star_327": PHI_STAR_327, "a10g_sms": A10G_SMS,
            "a10g_sms_instruction": A10G_SMS_INSTRUCTION,
            "lambda1_ceil": LAMBDA1_CEIL, "official_tps": OFFICIAL_TPS,
            "target_tps": TARGET_TPS, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[sdpa-phi-floor] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    verdict_code = {"A_alive_free": 3, "B_alive_needs_kernel": 2, "C_dead": 0}[
        s4["identity_lane_verdict_at_geometric_phi"]]
    summary: dict[str, Any] = {
        # PRIMARY + TEST metrics.
        "phi_floor_self_test_passes": int(bool(st["phi_floor_self_test_passes"])),
        "geometric_phi_estimate": s3["geometric_phi_estimate"],
        "identity_lane_verdict_at_geometric_phi_code": verdict_code,
        # step1 geometry.
        "kv_bytes_denken": s1["kv_bytes_denken"], "kv_bytes_gqa": s1["kv_bytes_gqa"],
        "roofline_us": s1["roofline_us"], "bw_util": s1["bw_util"],
        "exposed_us": s1["exposed_us"], "byte_resid_max": s1["byte_resid_max"],
        "geometry_reproduces_291": int(bool(s1["geometry_reproduces_291"])),
        # step2 partition.
        "n_nonreduction_ctas_2d": s2["n_nonreduction_ctas_2d"],
        "n_full_3d_ctas": s2["n_full_3d_ctas"],
        "n_unpack_max_tiles": s2["n_unpack_max_tiles"],
        # step3 occupancy phi.
        "phi_occupancy_80sms": s3["phi_occupancy_80sms"],
        "phi_occupancy_84sms": s3["phi_occupancy_84sms"],
        "phi_measured_2dv3d": s3["phi_measured_2dv3d"],
        "phi_unpack_floor_naive_occupancy": s3["phi_unpack_floor_naive_occupancy"],
        "phi_realizable_lower_bound": s3["phi_realizable_lower_bound"],
        "breakeven_ctas": s3["breakeven_ctas"],
        "natural_clears_breakeven": int(bool(s3["natural_clears_breakeven"])),
        "unpack_clears_breakeven_naive": int(bool(s3["unpack_clears_breakeven_naive"])),
        # step3 DECISIVE: occupancy-saturated 3D path refutes the naive un-pack (B) door.
        "adaptive_saturates_machine": int(bool(s3["adaptive_saturates_machine"])),
        "unpack_below_adaptive": int(bool(s3["unpack_below_adaptive"])),
        "slack_is_low_ai_not_occupancy": int(bool(s3["slack_is_low_ai_not_occupancy"])),
        "unpack_door_refuted_by_saturated_3d": int(bool(s3["unpack_door_refuted_by_saturated_3d"])),
        "sdpa_ai_flop_per_byte": s3["sdpa_ai_flop_per_byte"], "ridge_ai": s3["ridge_ai"],
        # step4 floor + verdict.
        "phi_star_breakeven_combined": s4["phi_star_breakeven_combined"],
        "phi_star_breakeven_sdpa_only": s4["phi_star_breakeven_sdpa_only"],
        "recovery_fraction_needed_to_revive": s4["recovery_fraction_needed_to_revive"],
        "geo_recovery_fraction": s4["geo_recovery_fraction"],
        "measured_recovery_fraction": s4["measured_recovery_fraction"],
        "geo_recovers_enough_for_B": int(bool(s4["geo_recovers_enough_for_B"])),
        "floor_at_geometric_phi": s4["floor_at_geometric_phi"],
        "floor_at_measured_phi": s4["floor_at_measured_phi"],
        "floor_at_unpack_phi": s4["floor_at_unpack_phi"],
        "floor_at_phi1_roundtrips_327": s4["floor_at_phi1_roundtrips_327"],
        "floor_327_roundtrip_resid": s4["floor_327_roundtrip_resid"],
        "phi_star_327_roundtrip_resid": s4["phi_star_327_roundtrip_resid"],
        "budget_lambda1_pct_213": s4["budget_lambda1_pct_213"],
        # advisor #192 TWO-budget band (>500 vs no-regression).
        "budget_500_pct_192": s4["budget_500_pct_192"],
        "budget_500_frac_192": s4["budget_500_frac_192"],
        "phi_star_500_combined": s4["phi_star_500_combined"],
        "phi_star_noregression_combined": s4["phi_star_noregression_combined"],
        "phi_star_500_lmfixed": s4["phi_star_500_lmfixed"],
        "phi_star_500_below_noregression": int(bool(s4["phi_star_500_below_noregression"])),
        # band codes: GREEN_500=2, YELLOW_trap_misses_500=1, RED_regression=0.
        "identity_half_500_band_at_geometric_phi_code": _band_code(
            s4["identity_half_500_band_at_geometric_phi"]),
        "identity_half_500_band_at_measured_phi_code": _band_code(
            s4["identity_half_500_band_at_measured_phi"]),
        "identity_half_500_band_at_realizable_phi_code": _band_code(
            s4["identity_half_500_band_at_realizable_phi"]),
        "identity_half_500_band_at_unpack_naive_phi_code": _band_code(
            s4["identity_half_500_band_at_unpack_naive_phi"]),
        "geometric_phi_clears_500_budget": int(bool(s4["geometric_phi_clears_500_budget"])),
        "geometric_phi_clears_noregression_budget": int(bool(
            s4["geometric_phi_clears_noregression_budget"])),
        "no_defensible_phi_clears_500": int(bool(s4["no_defensible_phi_clears_500"])),
        "naive_unpack_would_clear_500": int(bool(s4["naive_unpack_would_clear_500"])),
        "lane_dead_at_geometric_phi": int(bool(s4["lane_dead_at_geometric_phi"])),
        "lane_dead_at_realizable_phi": int(bool(s4["lane_dead_at_realizable_phi"])),
        "compliant_ceiling_tps_at_geo": s4["compliant_ceiling_tps_at_geo"],
        "compliant_ceiling_tps_at_unpack": s4["compliant_ceiling_tps_at_unpack"],
        "ceiling_at_geo_clears_500": int(bool(s4["ceiling_at_geo_clears_500"])),
        "ceiling_at_unpack_clears_500": int(bool(s4["ceiling_at_unpack_clears_500"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_sdpa_phi_floor_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[sdpa-phi-floor] wandb logged (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--c326-config-only", type=float, default=None,
                    help="wirbel #326's config-only ceiling (fraction-of-step), if known")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-sdpa-phi-floor")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize(c326_config_only=args.c326_config_only)
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 332, "agent": "denken",
        "kind": "eagle3-sdpa-phi-floor", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[sdpa-phi-floor] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_sdpa_phi_floor_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[sdpa-phi-floor] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["phi_floor_self_test_passes"] and payload["nan_clean"]
        print(f"[sdpa-phi-floor] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
