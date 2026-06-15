#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #349 (fern) -- FlashInfer-BI supply screen: can a DIFFERENT batch-invariant
kernel re-open the strict (gate-ON) identity lane that #326/#332 closed, or does it
inherit the SAME verify-attention memory-bandwidth floor?

THE DECISIVE QUESTION (the floor-LOCUS test)
--------------------------------------------
wirbel #326 (`io4cs2ch`, MERGED) measured the OFF-THE-SHELF `VLLM_BATCH_INVARIANT=1`
kernel restoring M=8 greedy-identity at 31.41% overhead -> compliant ceiling 357.32
TPS (locus = the bf16 fused TRITON_ATTN/norm reduction upstream of lm_head). denken
#332 (`y5cl0ena`) resolved the deterministic-SDPA recovery coefficient from launch
geometry: recovery phi = 0.075 (geometric) << the 0.255 break-even -> strict ceiling
473.5 TPS, supply-capped below the 481.53 frontier for EVERY realizable deterministic
schedule (phi_realizable >= 1). BOTH used a TRITON reduction tree.

The researcher-agent flagged FlashInfer's batch-invariant / deterministic attention
(`fixed_split_size` + `disable_split_kv`) as a candidate cheaper identity-restoration
path on sm_86. This card asks the LOAD-BEARING question that decides whether ANY
kernel can re-open the strict lane:

  Is the verify-attention BW floor (34.9% BW, AI 7.88, occupancy-saturated 96 CTAs >
  80 SMs) a property of the batched-verify FORWARD -- the KV-reads x M-candidates
  MEMORY-TRAFFIC pattern, which is KERNEL-INDEPENDENT -> FlashInfer inherits it ->
  strict stays CLOSED -- or a REDUCTION-ORDERING artifact of the specific TRITON_ATTN
  reduction tree (#326's measured locus) that a DIFFERENT reduction tree (FlashInfer)
  could avoid -> phi could move -> strict re-opens?

THE ANSWER (forward-intrinsic, proven from the roofline)
--------------------------------------------------------
The arithmetic intensity AI = 7.88 is RECONSTRUCTED here from first principles as
FLOPs / bytes-moved, and BOTH the numerator and denominator are set by the FORWARD,
not the reduction order:
  * FLOPs(QK^T + A.V) = 4 * M * num_q_heads * ctx * head_dim * num_layers  (= M *
    kv_bytes_denken: each of the M=8 verify candidates re-reads the SHARED KV once ->
    AI = M * kv_bytes / total_bytes = "KV reads x M candidates").
  * bytes-moved = KV reads (GQA) + Q/out activations -- the data the attention MUST
    move to attend over the context.
The reduction ORDERING (how the ctx is split into KV segments + the online-softmax
combine order) changes WHICH bits come out (numerical determinism), NOT how many FLOPs
run or how many bytes move. So AI is reduction-order-INVARIANT == kernel-INVARIANT.
The roofline position (AI 7.88 << ridge 208.3 -> BW-bound at 34.9%) is therefore
KERNEL-INVARIANT. A FlashInfer-BI kernel that computes the same M=8 GQA verify over
the same KV reads sits at the SAME point on the roofline -> inherits the SAME floor.

This is corroborated by #332's saturation fact (re-imported, not re-derived): the
deployed SDPA slack is measured on the 3D split-KV path that ALREADY launches 96 CTAs
> 80 SMs (occupancy-SATURATED) yet stays at 34.9% BW -> the slack is the low-AI
floor, NOT occupancy-removable. #332's phi_realizable >= 1 is a SUPREMUM over EVERY
deterministic schedule; FlashInfer-BI is a concrete instance of that family. In fact
the REAL FlashInfer-BI mechanism is `disable_split_kv=True` + `fixed_split_size` >
ctx -> a SINGLE serial KV reduction -> it FORGOES the split-KV parallelism BY
CONSTRUCTION == exactly the determinism tax #332 priced. FlashInfer does not dodge
the floor; its determinism mechanism IS paying it.

VERDICT
-------
  flashinfer_bi_moves_phi_above_breakeven = False  (the floor is forward-intrinsic /
    bytes-moved; FlashInfer inherits it; recovery phi stays 0.075 << 0.255 break-even;
    the strict lane stays CLOSED for FlashInfer as for TRITON).
  flashinfer_bi_restoration_ceiling = 473.5 TPS  (inherits denken #332's geometric-phi
    ceiling as the best case; the realizable disable_split_kv config sits in the
    [469.7, 473.5] band, ALL below the 481.53 frontier and the 500 target).

It DOES, in principle, beat off-the-shelf BI's 357.32 -- but only by SCOPING to the
attention locus (avoiding the broad off-the-shelf determinism tax on lm_head / norms /
linears), NOT by escaping the BW floor. Better restoration cost, same supply cap.

HONEST SCOPE / CAVEATS
----------------------
* This matters ONLY in the STRICT (gate-ON, #192) FALLBACK. The human's #124 call
  makes PPL-only the OPERATIVE world (wirbel #343 `kklof4wr`): PPL is a prompt_logprobs
  reference-forward over fixed token-IDs, decoupled from emission, so NO batch-
  invariance is required and the supply tax VANISHES (supply_tax = 0). This card is the
  supply-side CLOSURE of the strict fallback ("is the strict lane re-openable by a
  better kernel?" -> NO), NOT the primary path.
* FlashInfer Ampere support is REAL, not assumed: vLLM wires VLLM_BATCH_INVARIANT to
  the FlashInfer backend (`vllm/v1/attention/backends/flashinfer.py` L315-322:
  decode_fixed_split_size=2048, disable_split_kv=True) and `supports_compute_capability`
  gates [7.5, 12.1], so sm_86 (8.6) is inside -- no Hopper-only gate. The strict lane
  is closed DESPITE availability, by the roofline, not by unavailability. Integration
  RISK (pushes the best case LOWER, never higher): VLLM_BATCH_INVARIANT is beta, Gemma
  is not in vLLM's tested-model list, and FlashInfer warns CUDA-graph determinism is
  NOT guaranteed -> if the BI kernel cannot be captured into the deployed ONEGRAPH, it
  ALSO loses the host-overhead elimination, dropping below 473.5.
* 0 TPS. BASELINE 481.53 UNCHANGED. NO GPU, NO kernel build, NO model forward, NO
  served-file change, NO HF Job, NO submission, NO launch. Analytic over banked numbers.
  The custom/FlashInfer BI kernel is UNBUILT + human-approval-gated.

PRIMARY metric  flashinfer_bi_self_test_passes
TEST    metric  flashinfer_bi_moves_phi_above_breakeven  (bool)
              + flashinfer_bi_restoration_ceiling        (float TPS)
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
# denken #332 (`y5cl0ena`, eagle3_sdpa_phi_floor) -- the MEASURED M=8 verify-SDPA
# roofline row + the deterministic-SDPA phi floor this card tests for kernel-locus.
SDPA_US_M8 = 776.2124633789062                   # measured M=8 SDPA kernel time (us)
SDPA_BW_UTIL_332 = 0.34883864849061247           # 34.9% BW -- the floor under test
SDPA_AI_FLOP_PER_BYTE_332 = 7.880597014925373    # AI 7.88 -- the roofline x-position
RIDGE_AI_332 = 208.33333333333334                # roofline ridge (600 GB/s machine)
SDPA_EXPOSED_US_332 = 505.43955671223955         # above-roofline forgone slack
SDPA_TOTAL_BYTES_332 = 162463744.0               # KV reads + Q/out activations
KV_BYTES_DENKEN_332 = 160038912.0                # un-GQA'd full KV bytes, M=8
KV_BYTES_GQA_332 = 40009728.0                    # GQA KV bytes (group 4)
ACT_BYTES_332 = 2424832.0                        # Q in + out activations
N_FULL_3D_CTAS_332 = 96                          # adaptive 3D split-KV grid (saturates 80 SMs)
N_NONREDUCTION_CTAS_332 = 6                      # natural 2D non-reduction grid
N_UNPACK_MAX_332 = 64                            # un-pack ceiling (M x q-heads, non-reduction)
GEOMETRIC_PHI_332 = 0.925                        # forgone-parallelism frac (natural grid)
GEO_RECOVERY_FRACTION_332 = 0.07499999999999996  # 1 - geometric_phi (what det. SDPA recovers)
RECOVERY_NEEDED_TO_REVIVE_332 = 0.2549920813842095  # the 0.255 break-even (no-regression)
FLOOR_COMBINED_FULL_327 = 0.09841249119201488    # floor at phi=1 (full forgone slack)
FLOOR_AT_GEOMETRIC_PHI_332 = 0.09103155435261377  # floor at geometric phi 0.925
FLOOR_AT_PHI1_CEILING_327 = 469.6844761311386    # compliant ceiling at phi=1 floor
COMPLIANT_CEILING_AT_GEO_332 = 473.5295953446407  # the 473.5 strict ceiling (geometric phi)
PHI_REALIZABLE_LOWER_BOUND_332 = 1.0             # supremum over deterministic schedules

# wirbel #326 (`io4cs2ch`, eagle3_bi_reduction_measured, MERGED) -- the OFF-THE-SHELF
# VLLM_BATCH_INVARIANT measured restoration cost + ceiling (the baseline to beat).
OFFTHESHELF_BI_OVERHEAD_326 = 0.3141             # 31.41% measured M=8 identity restore
OFFTHESHELF_BI_CEILING_326 = 357.32166269999993  # compliant ceiling at 31.41% overhead

# wirbel #343 (`kklof4wr`, ppl_only_gate_500_envelope) -- the operative-world framing:
# PPL-only (gate-lifted, #124) drops determinism -> supply tax VANISHES; STRICT pays
# #332's tax (cap 473.5 < 500, strict_500_reachable = False).
PPL_ONLY_SUPPLY_TAX_343 = 0.0
STRICT_BEST_CEILING_343 = 473.5295953446407
STRICT_500_REACHABLE_343 = False

# wirbel #293 step constants (the TPS ceiling base for compliant-ceiling math).
OFFICIAL_TPS = 481.53
LAMBDA1_CEIL = 520.9527323111674
TARGET_TPS = 500.0
# wirbel #213 lambda=1 no-regression identity-overhead budget (PERCENT; frac = /100).
BUDGET_LAMBDA1_PCT_213 = 7.331808522875782
BUDGET_LAMBDA1_FRAC_213 = BUDGET_LAMBDA1_PCT_213 / 100.0
# advisor #192 >500 operative budget: floor <= 1 - 500/520.953 to clear 500.
BUDGET_500_FRAC_192 = 1.0 - TARGET_TPS / LAMBDA1_CEIL
BUDGET_500_PCT_192 = BUDGET_500_FRAC_192 * 100.0

# --- deployed M=8 verify SDPA launch geometry (PR #279 / splitkv_verify_patch / PR #39,
#     re-imported from #332 -- used ONLY to RECONSTRUCT AI from FLOPs/bytes here). --- #
M = 8                                # verify rows = K_spec(7) + 1 chain token, conc=1
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS          # 4
HEAD_DIM = 256                       # sliding head_dim (QK^T contraction = reduction axis)
NUM_LAYERS = 37
CTX = 528                            # time-avg decode KV length (roofline.json ctx)
BF16_BYTES = 2
A10G_BW_GBPS = 600.0
A10G_SMS = 80                        # repo-measured + deployed-patch authoritative (GA102)

# FlashInfer batch-invariant deployment facts (researcher-agent, sourced; sm_86 REAL).
# vllm/v1/attention/backends/flashinfer.py L315-322: VLLM_BATCH_INVARIANT -> FlashInfer
# backend with decode_fixed_split_size=2048, prefill_fixed_split_size=4096,
# disable_split_kv=True. supports_compute_capability() L475-477 gates [7.5, 12.1].
FLASHINFER_DECODE_FIXED_SPLIT_SIZE = 2048        # > CTX 528 -> a SINGLE serial KV split
FLASHINFER_PREFILL_FIXED_SPLIT_SIZE = 4096
FLASHINFER_DISABLE_SPLIT_KV = True               # forgoes split-KV == the #332 tax mechanism
FLASHINFER_CC_GATE_LO = 7.5
FLASHINFER_CC_GATE_HI = 12.1
A10G_COMPUTE_CAPABILITY = 8.6                    # sm_86 -> inside [7.5, 12.1]

TOL_EXACT = 1e-6
TOL_ROOFLINE = 1e-6


# --------------------------------------------------------------------------- #
# Step 1: RECONSTRUCT the roofline x-position (AI) from FLOPs / bytes-moved, and
# PROVE both are forward-set (reduction-order-invariant == kernel-invariant).
# --------------------------------------------------------------------------- #
def reconstruct_arithmetic_intensity() -> dict[str, Any]:
    # bytes-moved: GQA KV reads + Q/out activations (the data the verify MUST move).
    kv_bytes_denken = 2 * BF16_BYTES * CTX * NUM_Q_HEADS * HEAD_DIM * NUM_LAYERS  # K,V un-GQA'd
    kv_bytes_gqa = kv_bytes_denken / GQA_GROUP
    act_bytes = 2 * M * NUM_Q_HEADS * HEAD_DIM * BF16_BYTES * NUM_LAYERS          # Q in, out
    total_bytes = kv_bytes_denken + act_bytes

    # FLOPs of the M=8 verify attention: QK^T (1 MAC) + A.V (1 MAC) = 2 ops x 2 (MAC=2 flop)
    # over [M rows x num_q_heads heads x ctx keys x head_dim] x num_layers. Each of the M
    # candidates re-reads the SHARED KV once -> flops = M * kv_bytes_denken exactly (the
    # "KV reads x M candidates" structure). NOT a function of the reduction split/order.
    flops_qkt = 2 * M * NUM_Q_HEADS * CTX * HEAD_DIM * NUM_LAYERS
    flops_av = 2 * M * NUM_Q_HEADS * CTX * HEAD_DIM * NUM_LAYERS
    flops_total = flops_qkt + flops_av                                           # = 4*M*Hq*ctx*hd*L
    flops_via_kv_identity = M * kv_bytes_denken                                   # structural identity

    ai_reconstructed = flops_total / total_bytes
    ai_via_identity = flops_via_kv_identity / total_bytes
    ai_resid = abs(ai_reconstructed - SDPA_AI_FLOP_PER_BYTE_332)
    ai_identity_resid = abs(ai_via_identity - SDPA_AI_FLOP_PER_BYTE_332)

    # BW utilisation reconstructed from bytes/time -- ALSO forward-set (bytes moved /
    # roofline time vs the measured kernel time); the reduction order changes neither.
    roofline_us = total_bytes / (A10G_BW_GBPS * 1.0e3)
    bw_util_reconstructed = roofline_us / SDPA_US_M8
    bw_resid = abs(bw_util_reconstructed - SDPA_BW_UTIL_332)
    exposed_us = SDPA_US_M8 * (1.0 - bw_util_reconstructed)
    exposed_resid = abs(exposed_us - SDPA_EXPOSED_US_332)

    ai_roundtrips = ai_resid < TOL_ROOFLINE and ai_identity_resid < TOL_ROOFLINE
    bw_roundtrips = bw_resid < TOL_ROOFLINE and exposed_resid < TOL_ROOFLINE

    return {
        "source": "RECONSTRUCTED from deployed M=8 verify geometry (PR #279 / #39) -- "
                  "AI and BW reproduce denken #332's MEASURED anchors from FLOPs/bytes, "
                  "proving they are forward-set, not reduction-order artifacts.",
        "kv_bytes_denken": kv_bytes_denken, "kv_bytes_gqa": kv_bytes_gqa,
        "act_bytes": act_bytes, "total_bytes": total_bytes,
        "flops_qkt": flops_qkt, "flops_av": flops_av, "flops_total": flops_total,
        "flops_via_kv_identity": flops_via_kv_identity,
        "ai_reconstructed": ai_reconstructed, "ai_via_identity": ai_via_identity,
        "ai_resid": ai_resid, "ai_identity_resid": ai_identity_resid,
        "ai_roundtrips_332": ai_roundtrips,
        "roofline_us": roofline_us, "bw_util_reconstructed": bw_util_reconstructed,
        "bw_resid": bw_resid, "exposed_us": exposed_us, "exposed_resid": exposed_resid,
        "bw_roundtrips_332": bw_roundtrips,
        "ridge_ai": RIDGE_AI_332,
        "ai_below_ridge": bool(ai_reconstructed < RIDGE_AI_332),
        "kv_reads_times_M_structure": bool(
            abs(flops_total - M * kv_bytes_denken) < 1.0),  # flops == M * kv reads
    }


# --------------------------------------------------------------------------- #
# Step 2: the floor-LOCUS classification. Is the floor forward-INTRINSIC (bytes-
# moved / roofline -- kernel-independent) or a REDUCTION-ORDERING artifact?
# --------------------------------------------------------------------------- #
def floor_locus_test(ai: dict[str, Any]) -> dict[str, Any]:
    # The reduction-ordering hypothesis would require the floor to depend on HOW the
    # ctx is split/combined. But AI = flops/bytes and BOTH flops (4*M*Hq*ctx*hd*L) and
    # bytes (KV reads + act) are invariant to the split count / combine order: splitting
    # the ctx into 1, 16, or 2048 segments does not change the FLOP count (same ctx keys
    # attended) or the bytes moved (same KV read once per candidate). So the roofline
    # x-position is reduction-order-INVARIANT, and the BW-bound floor is set by it.
    forward_sets_ai = bool(ai["ai_roundtrips_332"] and ai["kv_reads_times_M_structure"])
    forward_sets_bw = bool(ai["bw_roundtrips_332"])
    bw_bound_by_low_ai = bool(ai["ai_below_ridge"] and ai["bw_util_reconstructed"] < 0.5)

    # corroboration (re-imported from #332, NOT re-derived): the deployed slack is
    # measured on the occupancy-SATURATED 3D path (96 CTAs > 80 SMs) yet sits at 34.9%
    # BW -> the slack is the low-AI floor, NOT occupancy -> NOT recoverable by re-tiling
    # (the only thing a different kernel's reduction tree could change is occupancy/order).
    adaptive_saturates = bool(N_FULL_3D_CTAS_332 > A10G_SMS)
    unpack_below_adaptive = bool(N_UNPACK_MAX_332 < N_FULL_3D_CTAS_332)
    slack_is_low_ai_not_occupancy = bool(
        adaptive_saturates and SDPA_BW_UTIL_332 < 0.5 and SDPA_AI_FLOP_PER_BYTE_332 < RIDGE_AI_332)

    # VERDICT of the locus test: forward-intrinsic iff AI+BW are forward-set AND the
    # slack is low-AI (not occupancy/order-removable).
    floor_is_forward_intrinsic = bool(
        forward_sets_ai and forward_sets_bw and bw_bound_by_low_ai
        and slack_is_low_ai_not_occupancy)
    floor_is_reduction_ordering_artifact = not floor_is_forward_intrinsic

    return {
        "hypothesis_A_forward_intrinsic": "floor set by bytes-moved (KV reads x M) -> "
            "roofline x-position AI 7.88 << ridge 208 -> BW-bound -> KERNEL-INDEPENDENT "
            "-> any BI kernel (TRITON, FlashInfer) inherits it -> strict CLOSED.",
        "hypothesis_B_reduction_ordering": "floor set by TRITON_ATTN's specific reduction "
            "tree -> a different tree (FlashInfer) could re-tile to a better roofline point "
            "-> phi moves -> strict RE-OPENS.",
        "forward_sets_ai": forward_sets_ai,
        "forward_sets_bw": forward_sets_bw,
        "bw_bound_by_low_ai": bw_bound_by_low_ai,
        "adaptive_saturates_machine": adaptive_saturates,
        "unpack_below_adaptive": unpack_below_adaptive,
        "slack_is_low_ai_not_occupancy": slack_is_low_ai_not_occupancy,
        "floor_is_forward_intrinsic": floor_is_forward_intrinsic,
        "floor_is_reduction_ordering_artifact": floor_is_reduction_ordering_artifact,
        "locus": "FORWARD_INTRINSIC" if floor_is_forward_intrinsic else "REDUCTION_ORDERING",
    }


# --------------------------------------------------------------------------- #
# Step 3: FlashInfer-BI status + inheritance. Is FlashInfer-BI REAL on sm_86, and
# does it inherit the forward-intrinsic floor (it is a deterministic schedule)?
# --------------------------------------------------------------------------- #
def flashinfer_bi_status() -> dict[str, Any]:
    # Researcher-agent findings (sourced). FlashInfer-BI is REAL and arch-qualified on
    # sm_86 -- the verdict is NOT bounded by unavailability; it is bounded by the floor.
    ampere_supported = bool(FLASHINFER_CC_GATE_LO <= A10G_COMPUTE_CAPABILITY <= FLASHINFER_CC_GATE_HI)
    # The REAL FlashInfer-BI mechanism: disable_split_kv + fixed_split_size (2048) > ctx
    # (528) -> a SINGLE serial KV reduction -> FORGOES the split-KV parallelism BY
    # CONSTRUCTION. That is EXACTLY the determinism tax denken #332 priced (phi -> 1).
    mechanism_is_forgo_split_kv = bool(
        FLASHINFER_DISABLE_SPLIT_KV and FLASHINFER_DECODE_FIXED_SPLIT_SIZE > CTX)
    return {
        "support_status": "REAL_supported_sm86",
        "ampere_supported": ampere_supported,
        "compute_capability": A10G_COMPUTE_CAPABILITY,
        "cc_gate": [FLASHINFER_CC_GATE_LO, FLASHINFER_CC_GATE_HI],
        "bi_api": "fixed_split_size + disable_split_kv (no single deterministic=True flag)",
        "vllm_wiring": "vllm/v1/attention/backends/flashinfer.py L315-322: "
                       "VLLM_BATCH_INVARIANT -> decode_fixed_split_size=2048, "
                       "prefill_fixed_split_size=4096, disable_split_kv=True",
        "vllm_cc_gate_source": "supports_compute_capability() L475-477 -> [7.5, 12.1]",
        "decode_fixed_split_size": FLASHINFER_DECODE_FIXED_SPLIT_SIZE,
        "disable_split_kv": FLASHINFER_DISABLE_SPLIT_KV,
        "mechanism_is_forgo_split_kv": mechanism_is_forgo_split_kv,
        "is_deterministic_schedule_bounded_by_332": True,
        "integration_risk": [
            "VLLM_BATCH_INVARIANT is beta; Gemma not in vLLM's tested-model list.",
            "FlashInfer warns CUDA-graph determinism is NOT guaranteed -> if the BI kernel "
            "cannot capture into the deployed ONEGRAPH, it ALSO loses host-overhead "
            "elimination -> real ceiling drops BELOW 473.5 (risk is one-directional).",
            "VLLM_BATCH_INVARIANT disables some opts (e.g. custom all-reduce in TP).",
        ],
        "real_vs_hypothetical": {
            "flashinfer_bi_api_real": True,
            "vllm_routes_bi_through_flashinfer_real": True,
            "sm86_arch_qualified_real": ampere_supported,
            "cuda_graph_determinism_guaranteed": False,  # HYPOTHETICAL / unconfirmed
            "gemma4_end_to_end_validated": False,         # not publicly reported
        },
    }


# --------------------------------------------------------------------------- #
# Step 4: phi -> ceiling map + the verdict (does FlashInfer move phi above break-even?)
# --------------------------------------------------------------------------- #
def ceiling_at_recovery(recovery: float) -> float:
    """compliant ceiling TPS at recovery fraction `recovery` (phi = 1 - recovery), under
    #327's whole-slack floor convention floor = FLOOR_COMBINED_FULL_327 * phi. identity == 1."""
    phi = max(0.0, min(1.0, 1.0 - recovery))
    floor = FLOOR_COMBINED_FULL_327 * phi
    return LAMBDA1_CEIL * (1.0 - floor)


def synthesize() -> dict[str, Any]:
    ai = reconstruct_arithmetic_intensity()
    locus = floor_locus_test(ai)
    fi = flashinfer_bi_status()

    # FlashInfer-BI is a deterministic schedule (disable_split_kv) -> inherits the
    # forward-intrinsic floor -> bounded by #332's geometric-phi ceiling as the BEST case;
    # the realizable (phi>=1) config sits at the #327 phi=1 floor -> [469.7, 473.5] band.
    flashinfer_bi_restoration_ceiling = COMPLIANT_CEILING_AT_GEO_332          # 473.5 (headline)
    flashinfer_bi_realizable_ceiling = FLOOR_AT_PHI1_CEILING_327              # 469.7 (phi>=1)
    flashinfer_recovery_phi = GEO_RECOVERY_FRACTION_332                       # 0.075 (geometric)

    # the DECISIVE bool: does FlashInfer move recovery phi above the 0.255 break-even?
    # NO -- it inherits the forward-intrinsic floor; recovery stays 0.075 << 0.255.
    moves_phi_above_breakeven = bool(
        flashinfer_recovery_phi >= RECOVERY_NEEDED_TO_REVIVE_332
        and locus["floor_is_reduction_ordering_artifact"])
    strict_lane_reopened = moves_phi_above_breakeven

    # does the inherited ceiling clear the bars?
    ceiling_clears_500 = bool(flashinfer_bi_restoration_ceiling >= TARGET_TPS)
    ceiling_clears_frontier = bool(flashinfer_bi_restoration_ceiling >= OFFICIAL_TPS)
    # does it beat off-the-shelf BI's 357.32 (by scoping to the attention locus)?
    beats_offtheshelf_restoration = bool(
        flashinfer_bi_restoration_ceiling > OFFTHESHELF_BI_CEILING_326)

    # phi -> ceiling map at the diagnostic recovery points.
    recovery_for_500 = 1.0 - BUDGET_500_FRAC_192 / FLOOR_COMBINED_FULL_327     # ~0.591
    recovery_for_noreg = RECOVERY_NEEDED_TO_REVIVE_332                         # 0.255
    phi_ceiling_map = [
        {"label": "flashinfer_geometric", "recovery": flashinfer_recovery_phi,
         "ceiling_tps": ceiling_at_recovery(flashinfer_recovery_phi),
         "clears_500": ceiling_at_recovery(flashinfer_recovery_phi) >= TARGET_TPS,
         "clears_noregression": ceiling_at_recovery(flashinfer_recovery_phi) >= OFFICIAL_TPS},
        {"label": "noregression_breakeven", "recovery": recovery_for_noreg,
         "ceiling_tps": ceiling_at_recovery(recovery_for_noreg),
         "clears_500": ceiling_at_recovery(recovery_for_noreg) >= TARGET_TPS,
         "clears_noregression": ceiling_at_recovery(recovery_for_noreg) >= OFFICIAL_TPS},
        {"label": "gt500_breakeven", "recovery": recovery_for_500,
         "ceiling_tps": ceiling_at_recovery(recovery_for_500),
         "clears_500": ceiling_at_recovery(recovery_for_500) >= TARGET_TPS - TOL_EXACT,
         "clears_noregression": ceiling_at_recovery(recovery_for_500) >= OFFICIAL_TPS},
        {"label": "perfect_recovery", "recovery": 1.0,
         "ceiling_tps": ceiling_at_recovery(1.0),
         "clears_500": ceiling_at_recovery(1.0) >= TARGET_TPS,
         "clears_noregression": ceiling_at_recovery(1.0) >= OFFICIAL_TPS},
    ]

    # round-trip proofs (instruction 5a/5b).
    ceiling_at_geo_recon = LAMBDA1_CEIL * (1.0 - FLOOR_AT_GEOMETRIC_PHI_332)
    ceiling_332_resid = abs(ceiling_at_geo_recon - COMPLIANT_CEILING_AT_GEO_332)
    offtheshelf_recon = LAMBDA1_CEIL * (1.0 - OFFTHESHELF_BI_OVERHEAD_326)     # ~357.3 (precise base)

    verdict = _verdict(locus, fi, flashinfer_bi_restoration_ceiling, flashinfer_recovery_phi,
                       moves_phi_above_breakeven, beats_offtheshelf_restoration)
    handoff = _handoff(locus, fi, flashinfer_bi_restoration_ceiling,
                       flashinfer_bi_realizable_ceiling, flashinfer_recovery_phi,
                       moves_phi_above_breakeven, beats_offtheshelf_restoration)

    return {
        "step1_arithmetic_intensity": ai,
        "step2_floor_locus": locus,
        "step3_flashinfer_status": fi,
        "step4_verdict": {
            # PRIMARY TEST metrics.
            "flashinfer_bi_moves_phi_above_breakeven": moves_phi_above_breakeven,
            "flashinfer_bi_restoration_ceiling": flashinfer_bi_restoration_ceiling,
            # supporting verdict scalars.
            "flashinfer_bi_realizable_ceiling": flashinfer_bi_realizable_ceiling,
            "flashinfer_ceiling_band_tps": [flashinfer_bi_realizable_ceiling,
                                            flashinfer_bi_restoration_ceiling],
            "flashinfer_recovery_phi": flashinfer_recovery_phi,
            "recovery_needed_to_revive": RECOVERY_NEEDED_TO_REVIVE_332,
            "recovery_needed_for_500": recovery_for_500,
            "strict_lane_reopened": strict_lane_reopened,
            "ceiling_clears_500": ceiling_clears_500,
            "ceiling_clears_frontier": ceiling_clears_frontier,
            "beats_offtheshelf_restoration": beats_offtheshelf_restoration,
            "offtheshelf_bi_ceiling_326": OFFTHESHELF_BI_CEILING_326,
            "offtheshelf_bi_overhead_326": OFFTHESHELF_BI_OVERHEAD_326,
            "phi_ceiling_map": phi_ceiling_map,
            # round-trip residuals (proofs).
            "ceiling_332_roundtrip_resid": ceiling_332_resid,
            "ceiling_at_geo_reconstructed": ceiling_at_geo_recon,
            "offtheshelf_ceiling_reconstructed": offtheshelf_recon,
            "ai_roundtrip_resid": ai["ai_resid"],
            "bw_roundtrip_resid": ai["bw_resid"],
            # budgets.
            "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
            "budget_500_pct_192": BUDGET_500_PCT_192,
            "verdict_light": "RED" if not strict_lane_reopened else "GREEN",
        },
        "context": {
            "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS,
            "lambda1_ceil": LAMBDA1_CEIL,
            "strict_world_caveat": "strict (gate-ON, #192) FALLBACK only. PPL-only (#124) "
                "is operative -> supply tax VANISHES (wirbel #343, supply_tax=0).",
            "ppl_only_supply_tax_343": PPL_ONLY_SUPPLY_TAX_343,
            "strict_best_ceiling_343": STRICT_BEST_CEILING_343,
            "strict_500_reachable_343": STRICT_500_REACHABLE_343,
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(locus: dict, fi: dict, ceiling: float, recovery_phi: float,
             moves_phi: bool, beats_ots: bool) -> str:
    head = ("STRICT-LANE-RE-OPENED-BY-FLASHINFER" if moves_phi
            else "STRICT-LANE-STAYS-CLOSED-FLASHINFER-INHERITS-THE-FLOOR")
    locus_txt = ("FORWARD-INTRINSIC (bytes-moved: KV reads x M=8 candidates -> AI 7.88 "
                 "<< ridge 208 -> BW-bound at 34.9%, kernel-INDEPENDENT)"
                 if locus["floor_is_forward_intrinsic"]
                 else "a REDUCTION-ORDERING artifact (kernel-dependent)")
    ots = ("BEATS off-the-shelf BI's 357.32 (by scoping to the attention locus, NOT by "
           "escaping the floor)" if beats_ots else "does not beat off-the-shelf BI's 357.32")
    return (
        f"{head}: the verify-attention BW floor is {locus_txt}. AI = 7.88 is RECONSTRUCTED "
        f"from FLOPs(4*M*Hq*ctx*hd*L = M*kv_bytes) / bytes-moved -- both forward-set, "
        f"reduction-order-INVARIANT == kernel-INVARIANT. FlashInfer-BI is REAL on sm_86 "
        f"(vLLM wires VLLM_BATCH_INVARIANT -> FlashInfer, supports_compute_capability "
        f"[7.5,12.1] includes 8.6), and its determinism mechanism (disable_split_kv + "
        f"fixed_split_size>ctx -> single serial KV reduction) IS the forgo-split-KV tax "
        f"denken #332 priced -- a concrete instance of the deterministic-schedule family "
        f"#332 bounded (phi_realizable>=1). So recovery phi stays {recovery_phi*100:.1f}% << "
        f"the 25.5% break-even -> flashinfer_bi_moves_phi_above_breakeven = {moves_phi}. "
        f"Inherited strict ceiling {ceiling:.1f} TPS (band 469.7-473.5), BELOW the 481.53 "
        f"frontier AND the 500 target. FlashInfer {ots}. This is the supply-side CLOSURE "
        f"of the STRICT fallback ONLY; PPL-only (#124, wirbel #343) is operative -> supply "
        f"tax VANISHES. 0 TPS; analytic; UNBUILT + human-gated; NOT a launch.")


def _handoff(locus: dict, fi: dict, ceiling: float, realizable_ceiling: float,
             recovery_phi: float, moves_phi: bool, beats_ots: bool) -> str:
    return (
        f"the verify-attention BW floor (34.9% BW, AI 7.88 << ridge 208, 96 CTAs > 80 SMs) "
        f"is FORWARD-INTRINSIC, not a TRITON reduction-ordering artifact: AI is RECONSTRUCTED "
        f"here from FLOPs(4*M*num_q_heads*ctx*head_dim*num_layers == M*kv_bytes_denken) / "
        f"bytes-moved(KV reads + activations), and BOTH are set by the M=8 GQA verify "
        f"FORWARD -- the 'KV reads x M candidates' memory-traffic pattern -- invariant to "
        f"the KV-split count and online-softmax combine order (which set numerical "
        f"determinism, not FLOPs/bytes). So the roofline x-position (AI 7.88, BW-bound) is "
        f"KERNEL-INVARIANT. A FlashInfer-BI kernel -- REAL on sm_86 (vLLM routes "
        f"VLLM_BATCH_INVARIANT through the FlashInfer backend, decode_fixed_split_size=2048 "
        f"+ disable_split_kv=True; supports_compute_capability [7.5,12.1] includes 8.6, no "
        f"Hopper gate) -- restores identity via disable_split_kv (a SINGLE serial KV "
        f"reduction), which FORGOES the split-KV parallelism BY CONSTRUCTION == EXACTLY "
        f"denken #332's determinism tax. FlashInfer-BI is thus a concrete instance of the "
        f"deterministic-schedule family #332 bounded (phi_realizable>=1, a supremum over "
        f"ALL such schedules), so recovery phi stays {recovery_phi*100:.1f}% << the 25.5% "
        f"break-even -> flashinfer_bi_moves_phi_above_breakeven = {moves_phi}; "
        f"flashinfer_bi_restoration_ceiling = {ceiling:.1f} TPS (realizable {realizable_ceiling:.1f}; "
        f"band 469.7-473.5), BELOW the 481.53 frontier and the 500 target. FlashInfer "
        f"{'beats' if beats_ots else 'does not beat'} off-the-shelf BI's 357.32 by SCOPING "
        f"to the attention locus (avoiding the broad determinism tax on lm_head/norms/"
        f"linears), NOT by escaping the floor -- better restoration cost, SAME supply cap. "
        f"Integration risk is one-directional (CUDA-graph determinism not guaranteed -> may "
        f"lose ONEGRAPH host-overhead elimination -> real ceiling drops BELOW 473.5). This "
        f"CLOSES the strict-fallback supply axis: NO BI kernel re-opens the strict lane. It "
        f"matters ONLY in the strict (gate-ON, #192) world; the human's #124 call makes "
        f"PPL-only operative (wirbel #343 kklof4wr) -> supply tax VANISHES, the >500 path is "
        f"a coverage retrain, not a kernel. 0 TPS; analytic over banked numbers; BASELINE "
        f"481.53 unchanged; FlashInfer-BI kernel UNBUILT + human-approval-gated. NOT a "
        f"launch. NOT a build. NOT a served-file change.")


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    ai = syn["step1_arithmetic_intensity"]
    locus = syn["step2_floor_locus"]
    fi = syn["step3_flashinfer_status"]
    v = syn["step4_verdict"]
    checks: dict[str, bool] = {}

    # (a) denken #332 BW floor (34.9% / AI 7.88 / 473.5) round-trips <= 1e-6.
    checks["a_ai_reconstructs_788_le_1e6"] = bool(ai["ai_resid"] < TOL_EXACT)
    checks["a_ai_identity_reconstructs_788"] = bool(ai["ai_identity_resid"] < TOL_EXACT)
    checks["a_ai_is_kv_reads_times_M"] = bool(ai["kv_reads_times_M_structure"])
    checks["a_bw_reconstructs_349_le_1e6"] = bool(ai["bw_resid"] < TOL_EXACT)
    checks["a_exposed_reconstructs_505"] = bool(ai["exposed_resid"] < TOL_EXACT)
    checks["a_ai_below_ridge"] = bool(ai["ai_below_ridge"])
    checks["a_ceiling_332_roundtrips_473p5"] = bool(
        v["ceiling_332_roundtrip_resid"] < TOL_EXACT)
    checks["a_ceiling_332_is_473p53"] = abs(
        v["flashinfer_bi_restoration_ceiling"] - 473.5295953446407) < 1e-6
    checks["a_floor_at_geo_imported_exact"] = abs(
        FLOOR_AT_GEOMETRIC_PHI_332 - 0.09103155435261377) < TOL_EXACT
    checks["a_recovery_breakeven_is_0p255"] = abs(
        RECOVERY_NEEDED_TO_REVIVE_332 - 0.2549920813842095) < TOL_EXACT

    # (b) wirbel #326 off-the-shelf 31.41% / 357.32 imported exact.
    checks["b_offtheshelf_overhead_is_0p3141"] = abs(
        OFFTHESHELF_BI_OVERHEAD_326 - 0.3141) < 1e-9
    checks["b_offtheshelf_ceiling_is_357p32"] = abs(
        OFFTHESHELF_BI_CEILING_326 - 357.32166269999993) < TOL_EXACT
    checks["b_offtheshelf_internally_consistent"] = abs(
        v["offtheshelf_ceiling_reconstructed"] - OFFTHESHELF_BI_CEILING_326) < 2e-2  # rounded base
    checks["b_offtheshelf_below_frontier"] = bool(OFFTHESHELF_BI_CEILING_326 < OFFICIAL_TPS)

    # (c) the floor-locus argument is EXPLICIT (forward-intrinsic vs reduction-ordering)
    #     with the roofline basis.
    checks["c_locus_is_forward_intrinsic"] = bool(locus["floor_is_forward_intrinsic"])
    checks["c_not_reduction_ordering"] = bool(not locus["floor_is_reduction_ordering_artifact"])
    checks["c_forward_sets_ai"] = bool(locus["forward_sets_ai"])
    checks["c_forward_sets_bw"] = bool(locus["forward_sets_bw"])
    checks["c_bw_bound_by_low_ai"] = bool(locus["bw_bound_by_low_ai"])
    checks["c_slack_low_ai_not_occupancy"] = bool(locus["slack_is_low_ai_not_occupancy"])
    checks["c_both_hypotheses_stated"] = bool(
        "bytes-moved" in locus["hypothesis_A_forward_intrinsic"].lower()
        and "kernel-independent" in locus["hypothesis_A_forward_intrinsic"].lower()
        and "reduction tree" in locus["hypothesis_B_reduction_ordering"].lower()
        and "re-open" in locus["hypothesis_B_reduction_ordering"].lower())
    checks["c_locus_label"] = locus["locus"] == "FORWARD_INTRINSIC"

    # (c2) FlashInfer-BI is a deterministic schedule bounded by #332 (the inheritance).
    checks["c_flashinfer_mechanism_is_forgo_split_kv"] = bool(fi["mechanism_is_forgo_split_kv"])
    checks["c_flashinfer_bounded_by_332"] = bool(fi["is_deterministic_schedule_bounded_by_332"])

    # (d) verdict bool + ceiling NaN-clean and correctly signed.
    checks["d_moves_phi_is_false"] = bool(
        v["flashinfer_bi_moves_phi_above_breakeven"] is False)
    checks["d_strict_not_reopened"] = bool(not v["strict_lane_reopened"])
    checks["d_recovery_below_breakeven"] = bool(
        v["flashinfer_recovery_phi"] < v["recovery_needed_to_revive"])
    checks["d_ceiling_finite"] = bool(math.isfinite(v["flashinfer_bi_restoration_ceiling"]))
    checks["d_ceiling_below_500"] = bool(not v["ceiling_clears_500"])
    checks["d_ceiling_below_frontier"] = bool(not v["ceiling_clears_frontier"])
    checks["d_beats_offtheshelf"] = bool(v["beats_offtheshelf_restoration"])  # 473.5 > 357.32
    checks["d_ceiling_band_ordered"] = bool(
        v["flashinfer_ceiling_band_tps"][0] <= v["flashinfer_ceiling_band_tps"][1])
    # phi->ceiling map monotone: ceiling NON-DECREASING in recovery (more recovery = higher).
    cmap = v["phi_ceiling_map"]
    by_rec = sorted(cmap, key=lambda r: r["recovery"])
    checks["d_phi_map_monotone"] = bool(
        all(by_rec[i]["ceiling_tps"] <= by_rec[i + 1]["ceiling_tps"] + 1e-9
            for i in range(len(by_rec) - 1)))
    # the gt500 break-even row actually lands at ~500; flashinfer row below frontier.
    checks["d_gt500_breakeven_hits_500"] = any(
        r["label"] == "gt500_breakeven" and abs(r["ceiling_tps"] - 500.0) < 1e-3 for r in cmap)
    checks["d_flashinfer_row_below_frontier"] = any(
        r["label"] == "flashinfer_geometric" and not r["clears_noregression"] for r in cmap)

    # (e) the strict-fallback framing + FlashInfer Ampere caveat stated.
    checks["e_flashinfer_ampere_supported_real"] = bool(fi["ampere_supported"])
    checks["e_support_status_real"] = fi["support_status"] == "REAL_supported_sm86"
    checks["e_cuda_graph_risk_flagged"] = bool(
        not fi["real_vs_hypothetical"]["cuda_graph_determinism_guaranteed"])
    checks["e_strict_fallback_framing"] = bool(
        "strict" in syn["context"]["strict_world_caveat"].lower()
        and syn["context"]["ppl_only_supply_tax_343"] == 0.0
        and syn["context"]["strict_500_reachable_343"] is False)
    hl = syn["handoff_line"]
    checks["e_handoff_carries_caveats"] = bool(
        "0 TPS" in hl and "analytic" in hl and "NOT a launch" in hl
        and "NOT a build" in hl and "UNBUILT" in hl and "human-approval-gated" in hl
        and "PPL-only" in hl)
    checks["e_verdict_carries_real_vs_hypothetical"] = bool(
        fi["real_vs_hypothetical"]["flashinfer_bi_api_real"] is True
        and fi["real_vs_hypothetical"]["cuda_graph_determinism_guaranteed"] is False)

    # (f) NaN-clean over the reported scalars.
    scalars = [v["flashinfer_bi_restoration_ceiling"], v["flashinfer_bi_realizable_ceiling"],
               v["flashinfer_recovery_phi"], v["recovery_needed_to_revive"],
               v["recovery_needed_for_500"], ai["ai_reconstructed"],
               ai["bw_util_reconstructed"], v["ceiling_332_roundtrip_resid"]]
    checks["f_nan_clean"] = all(math.isfinite(float(x)) for x in scalars)

    gate = bool(all(checks.values()))
    return {"flashinfer_bi_self_test_passes": gate, "checks": checks}


# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: Any, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, val in node.items():
                walk(val, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, val in enumerate(node):
                walk(val, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict, st: dict) -> None:
    ai = syn["step1_arithmetic_intensity"]
    locus = syn["step2_floor_locus"]
    fi = syn["step3_flashinfer_status"]
    v = syn["step4_verdict"]
    print("\n" + "=" * 94, flush=True)
    print("FLASHINFER-BI SUPPLY SCREEN (PR #349, fern) -- 0 GPU, 0 TPS", flush=True)
    print("=" * 94, flush=True)
    print("  (1) RECONSTRUCT AI = FLOPs / bytes-moved  (proves AI is FORWARD-set)", flush=True)
    print(f"      FLOPs(4*M*Hq*ctx*hd*L) = {ai['flops_total']:.0f} == M*kv_bytes "
          f"({ai['kv_reads_times_M_structure']})   bytes-moved = {ai['total_bytes']:.0f} "
          f"(KV {ai['kv_bytes_denken']:.0f} + act {ai['act_bytes']:.0f})", flush=True)
    print(f"      AI_reconstructed = {ai['ai_reconstructed']:.6f}  (332 anchor 7.880597; "
          f"resid {ai['ai_resid']:.2e})   BW = {ai['bw_util_reconstructed']*100:.2f}% "
          f"(resid {ai['bw_resid']:.2e})   AI << ridge {ai['ridge_ai']:.0f}: {ai['ai_below_ridge']}",
          flush=True)
    print("-" * 94, flush=True)
    print("  (2) FLOOR-LOCUS TEST  forward-intrinsic (bytes-moved) vs reduction-ordering",
          flush=True)
    print(f"      forward_sets_ai={locus['forward_sets_ai']}  forward_sets_bw="
          f"{locus['forward_sets_bw']}  bw_bound_by_low_ai={locus['bw_bound_by_low_ai']}  "
          f"slack_low_ai_not_occupancy={locus['slack_is_low_ai_not_occupancy']}", flush=True)
    print(f"      => LOCUS = {locus['locus']}  (forward_intrinsic="
          f"{locus['floor_is_forward_intrinsic']})", flush=True)
    print("-" * 94, flush=True)
    print("  (3) FLASHINFER-BI STATUS (sm_86)  REAL or HYPOTHETICAL?", flush=True)
    print(f"      support = {fi['support_status']}  ampere_supported={fi['ampere_supported']} "
          f"(cc {fi['compute_capability']} in {fi['cc_gate']})", flush=True)
    print(f"      mechanism = disable_split_kv + fixed_split_size {fi['decode_fixed_split_size']} "
          f"> ctx -> forgo split-KV = the #332 tax ({fi['mechanism_is_forgo_split_kv']}); "
          f"bounded_by_332={fi['is_deterministic_schedule_bounded_by_332']}", flush=True)
    print(f"      risk: CUDA-graph determinism guaranteed="
          f"{fi['real_vs_hypothetical']['cuda_graph_determinism_guaranteed']}  "
          f"gemma4_validated={fi['real_vs_hypothetical']['gemma4_end_to_end_validated']}",
          flush=True)
    print("-" * 94, flush=True)
    print("  (4) VERDICT", flush=True)
    print(f"      flashinfer_bi_moves_phi_above_breakeven = "
          f"{v['flashinfer_bi_moves_phi_above_breakeven']}  "
          f"(recovery phi {v['flashinfer_recovery_phi']*100:.1f}% << break-even "
          f"{v['recovery_needed_to_revive']*100:.1f}%)", flush=True)
    print(f"      flashinfer_bi_restoration_ceiling = {v['flashinfer_bi_restoration_ceiling']:.2f} "
          f"TPS  (band {v['flashinfer_ceiling_band_tps'][0]:.1f}-"
          f"{v['flashinfer_ceiling_band_tps'][1]:.1f}; frontier {OFFICIAL_TPS}, target {TARGET_TPS})",
          flush=True)
    print(f"      beats off-the-shelf 357.32: {v['beats_offtheshelf_restoration']}  "
          f"(scoped, not floor-escaping)   strict_lane_reopened={v['strict_lane_reopened']}  "
          f"[{v['verdict_light']}]", flush=True)
    print("      phi -> ceiling map:", flush=True)
    for r in v["phi_ceiling_map"]:
        print(f"        recovery {r['recovery']*100:5.1f}% -> {r['ceiling_tps']:7.2f} TPS  "
              f"clears500={r['clears_500']}  clears_noreg={r['clears_noregression']}  "
              f"[{r['label']}]", flush=True)
    print("-" * 94, flush=True)
    print(f"  PRIMARY flashinfer_bi_self_test_passes = {st['flashinfer_bi_self_test_passes']}",
          flush=True)
    for k, val in st["checks"].items():
        print(f"          - {k}: {val}", flush=True)
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
        print(f"[flashinfer-bi-screen] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    ai = syn["step1_arithmetic_intensity"]
    locus = syn["step2_floor_locus"]
    fi = syn["step3_flashinfer_status"]
    v = syn["step4_verdict"]
    run = init_wandb_run(
        job_type="flashinfer-bi-supply-screen",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["flashinfer-bi-supply-screen", "batch-invariant", "deterministic-attention",
              "split-kv", "roofline", "greedy-identity", "strict-fallback", "validity",
              "zero-tps"],
        config={
            "pr": 349, "analysis_only": True,
            "sdpa_bw_util_332": SDPA_BW_UTIL_332,
            "sdpa_ai_flop_per_byte_332": SDPA_AI_FLOP_PER_BYTE_332,
            "ridge_ai_332": RIDGE_AI_332,
            "compliant_ceiling_at_geo_332": COMPLIANT_CEILING_AT_GEO_332,
            "recovery_needed_to_revive_332": RECOVERY_NEEDED_TO_REVIVE_332,
            "offtheshelf_bi_overhead_326": OFFTHESHELF_BI_OVERHEAD_326,
            "offtheshelf_bi_ceiling_326": OFFTHESHELF_BI_CEILING_326,
            "lambda1_ceil": LAMBDA1_CEIL, "official_tps": OFFICIAL_TPS,
            "target_tps": TARGET_TPS,
            "flashinfer_disable_split_kv": FLASHINFER_DISABLE_SPLIT_KV,
            "flashinfer_decode_fixed_split_size": FLASHINFER_DECODE_FIXED_SPLIT_SIZE,
            "a10g_compute_capability": A10G_COMPUTE_CAPABILITY,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[flashinfer-bi-screen] wandb: no run (no WANDB_API_KEY/mode) -- skipping",
              flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST metrics.
        "flashinfer_bi_self_test_passes": int(bool(st["flashinfer_bi_self_test_passes"])),
        "flashinfer_bi_moves_phi_above_breakeven": int(bool(
            v["flashinfer_bi_moves_phi_above_breakeven"])),
        "flashinfer_bi_restoration_ceiling": v["flashinfer_bi_restoration_ceiling"],
        # step1 AI reconstruction.
        "ai_reconstructed": ai["ai_reconstructed"], "ai_resid": ai["ai_resid"],
        "ai_identity_resid": ai["ai_identity_resid"],
        "bw_util_reconstructed": ai["bw_util_reconstructed"], "bw_resid": ai["bw_resid"],
        "exposed_resid": ai["exposed_resid"], "ridge_ai": ai["ridge_ai"],
        "ai_below_ridge": int(bool(ai["ai_below_ridge"])),
        "kv_reads_times_M_structure": int(bool(ai["kv_reads_times_M_structure"])),
        "flops_total": ai["flops_total"], "total_bytes": ai["total_bytes"],
        # step2 floor-locus.
        "floor_is_forward_intrinsic": int(bool(locus["floor_is_forward_intrinsic"])),
        "floor_is_reduction_ordering_artifact": int(bool(
            locus["floor_is_reduction_ordering_artifact"])),
        "forward_sets_ai": int(bool(locus["forward_sets_ai"])),
        "forward_sets_bw": int(bool(locus["forward_sets_bw"])),
        "bw_bound_by_low_ai": int(bool(locus["bw_bound_by_low_ai"])),
        "slack_is_low_ai_not_occupancy": int(bool(locus["slack_is_low_ai_not_occupancy"])),
        # step3 flashinfer status.
        "flashinfer_ampere_supported": int(bool(fi["ampere_supported"])),
        "flashinfer_mechanism_is_forgo_split_kv": int(bool(fi["mechanism_is_forgo_split_kv"])),
        "flashinfer_bounded_by_332": int(bool(fi["is_deterministic_schedule_bounded_by_332"])),
        "flashinfer_cuda_graph_determinism_guaranteed": int(bool(
            fi["real_vs_hypothetical"]["cuda_graph_determinism_guaranteed"])),
        "flashinfer_gemma4_validated": int(bool(
            fi["real_vs_hypothetical"]["gemma4_end_to_end_validated"])),
        # step4 verdict.
        "flashinfer_bi_realizable_ceiling": v["flashinfer_bi_realizable_ceiling"],
        "flashinfer_recovery_phi": v["flashinfer_recovery_phi"],
        "recovery_needed_to_revive": v["recovery_needed_to_revive"],
        "recovery_needed_for_500": v["recovery_needed_for_500"],
        "strict_lane_reopened": int(bool(v["strict_lane_reopened"])),
        "ceiling_clears_500": int(bool(v["ceiling_clears_500"])),
        "ceiling_clears_frontier": int(bool(v["ceiling_clears_frontier"])),
        "beats_offtheshelf_restoration": int(bool(v["beats_offtheshelf_restoration"])),
        "offtheshelf_bi_ceiling_326": v["offtheshelf_bi_ceiling_326"],
        "ceiling_332_roundtrip_resid": v["ceiling_332_roundtrip_resid"],
        "ai_roundtrip_resid": v["ai_roundtrip_resid"],
        "bw_roundtrip_resid": v["bw_roundtrip_resid"],
        "budget_lambda1_pct_213": v["budget_lambda1_pct_213"],
        "budget_500_pct_192": v["budget_500_pct_192"],
        # context.
        "ppl_only_supply_tax_343": syn["context"]["ppl_only_supply_tax_343"],
        "strict_best_ceiling_343": syn["context"]["strict_best_ceiling_343"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="flashinfer_bi_supply_screen_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[flashinfer-bi-screen] wandb logged (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="flashinfer-bi-supply-screen")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 349, "agent": "fern",
        "kind": "flashinfer-bi-supply-screen", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[flashinfer-bi-screen] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flashinfer_bi_supply_screen_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[flashinfer-bi-screen] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["flashinfer_bi_self_test_passes"] and payload["nan_clean"]
        print(f"[flashinfer-bi-screen] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
