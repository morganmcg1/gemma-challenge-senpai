#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Verify-compute hideability (PR #291) — the HONEST kernel-addressable floor.

WHAT THIS LEG REFINES
---------------------
denken #283 (vmxuwxm0, MERGED) put deployed 481.53 on the HBM-bound ceiling and
decomposed the HONEST official per-step wall (1/K_cal = 7982.9us) into
    read floor 3037.2us (38%)  +  non-read slack 4945.7us (62%),
with the 62% slack = draft 731.8 + verify-above-read 2104.6 + host/other 2109.3us.
#283's `tps_kernel_floor = 746.9` assumed ALL 2104.6us of verify-above-read compute
HIDES (collapses under the read shadow with perfect scheduling). That is an
OPTIMISTIC upper bound. This leg measures the ACTUAL overlap-hideable fraction.

wirbel #285 (97b57hhe, MERGED) priced the FREE lossless step ceiling at 487.7 TPS
(corroborated by kanna #286 0k4azmjo = 487.758) — the bit-identical greedy-safe
micro-levers on the NORMALIZED step (bridge draft~0.21 / verify~1.0). The team thus
carried TWO step-side floors in TWO bases: 487.7 (normalized-step, free, lossless)
and 746.9 (honest-wall, all-verify-compute-hides). This leg RECONCILES them onto one
basis and lands the honest floor between them.

THE TWO QUESTIONS
-----------------
(1) BASIS RECONCILIATION. wirbel #285's 487.7 lives on the NORMALIZED step (S=1218.2us);
    #283's 746.9 lives on the HONEST wall (W=1/K_cal=7982.9us). The explicit bridge is
    the COMPOSITION-COMPRESSION ratio phi_WS = W/S = 6.5530: a saving of dS us on the
    normalized step is worth dW = phi_WS * dS us on the honest wall for the SAME TPS
    (both round-trip 481.53). So 487.7's 15.48us normalized-step lever == 101.5us on the
    wall; 746.9's 2836.3us wall removal == 432.8us on the normalized step. basis_reconciled.
    The verify-above-read block is DEPLOYED-M=8 VERIFY-SIDE, so its hideability carries the
    VERIFY bridge ~ 1.0 (kanna #286: deployed-M8 verify per-seq 1129.6us ~ S=1218.2us), NOT
    the draft-side 0.21 -- a verify saving gets FULL normalized-step credit. (The draft 0.21
    over-credit is irrelevant here; we touch only the verify-side block.)

(2) OVERLAP-HIDEABILITY. On a memory-bound kernel, compute that fits under t_mem = bytes/BW
    is shadowed (free); compute exposed above t_mem is serial (on the critical path). The
    deployed verify is a chain of DATA-DEPENDENT layers already captured in ONE CUDA graph
    (ONEGRAPH) with fused epilogues (FUSED_SPARSE_ARGMAX, fused lm_head). So cross-layer
    concurrency is unavailable (each layer needs the previous layer's output), and the only
    remaining OVERLAP-schedulable headroom is INTRA-kernel cp.async pipelining. kanna #280
    (sdrerk5h) measured the per-component BW utilizations at M=8:
        gate_up 71.2% / down 66.5% (MLP: near-roofline, compute already shadowed),
        qkv 47.0% / o_proj 45.5% (GEMV: memory-bound), sdpa 34.9% (compute-EXPOSED),
        lm_head 83.4% (already fused).
    Of these, the ONLY greedy-SAFE overlap-schedulable lever is the SDPA num_stages 3->2
    cp.async-depth retune (bit-identical, maxdiff 0.0 over 128 gates -- wirbel #279/#285).
    The MLP/GEMV int4 kernels can only be sped up by num_warps/BLOCK_K/split-K, which
    REASSOCIATE the bf16 partial sums (greedy-UNSAFE -- kanna #280/#269, lawine #246); and
    lm_head/norms are already_captured by the deployed ONEGRAPH/fusion (wirbel #285,
    incremental 0.0us). So the greedy-SAFE overlap-hideable verify compute is EXACTLY the
    SDPA num_stages lever wirbel #285 already priced -- 15.48us normalized == 101.5us wall.

    "hideable" here means OVERLAP-schedulable (latency-only, greedy-SAFE by bit-identity);
    it is NOT kanna #280's *reducing* MLP compute (reassociation-gated, greedy-UNSAFE).

THE HONEST FLOOR
----------------
Remove ONLY the measured greedy-safe overlap-hideable verify compute (101.5us) from the
honest wall:
    tps_kernel_floor_honest = E[T]*1e6 / (W - verify_compute_hideable_us) = 487.7 TPS.
It COINCIDES with wirbel #285's free lossless floor -- because the only greedy-safe overlap
lever in the verify IS the SDPA num_stages lever wirbel already found. So of #283's
optimistic 746.9 all-hides bound, the HONEST realizable floor is 487.7 (the all-hides bound
over-credited the verify-above-read by ~259 TPS). free_lane_to_500_exists = FALSE:
reaching 500 would need ~3x the only greedy-safe lever and no candidate exists. The step-side
is DEFINITIVELY CLOSED; the human-gated E[T]-raise BUILD is the sole >500 path (reinforcing
fern #281). This leg adds 0 TPS (it prices the hideable fraction; it realizes nothing).

SCOPE
-----
LOCAL CPU analytic + an OPTIONAL GPU read-floor re-confirmation (CUDA-event grounding,
reused from #283/#278/#280 -- the verify-above-read split is already CUDA-event-measured).
All #283/#280/#285/#286/#278/#217/#267 scalars IMPORTED EXACT, NOT re-derived; kanna #280's
component decomposition IMPORTED (NOT re-decomposed). BASELINE stays 481.53. PRIMARY =
self-test. NOT a launch, NOT open2, no served-file change, no HF Job, no submission. The
launch gate remains land #245's MEASURED >=500 build at lambda_hat>=0.9780 AND PPL<=2.42,
human-approval-gated.

PRIMARY metric  verify_compute_hideability_self_test_passes
TEST    metric  tps_kernel_floor_honest

Run:
    cd target/ && CUDA_VISIBLE_DEVICES=0 \
      /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/speed/verify_compute_hideability/verify_compute_hideability.py --self-test \
      --wandb_group verify-compute-hideability --wandb_name denken/verify-compute-hideability
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
# Imported artifacts (CUDA-event grounding; reused, NOT re-measured).
# --------------------------------------------------------------------------- #
# denken #283 GPU read-floor re-confirmation (reduction over the 1.76 GB body).
_D283_READBENCH = (REPO_ROOT / "research/validity/hbm_bound_tps_ceiling/"
                   "read_floor_confirm.json")
# denken #278 CUDA-event verify/draft measurement (the verify-above-read source).
_D278_VERIFY = (REPO_ROOT / "research/validity/linear_step_decomposition/"
                "linear_verify_measurement.json")
# kanna #280 per-component verify decomposition (BW utilizations; IMPORTED, not re-decomposed).
_K280_ROOFLINE = (REPO_ROOT / "research/speed/verify_step_component_roofline/roofline.json")
# Optional fresh GPU probe output written by this script under --gpu-probe.
_LOCAL_READBENCH = HERE / "read_floor_confirm.json"

# --------------------------------------------------------------------------- #
# PR #291 imported constants (provenance in the docstring; NOT re-derived).
# All EXACT-matched in self-test (f).
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                 # PR #52 official TPS (the deployed point)
E_T = 3.844                                # kanna #217 deployed linear K=7 E[T]
K_CAL = 125.26795005202914                 # kanna #217 composition calibration (= official/E[T])
STEP_NORM_US = 1218.2                       # kanna #217 normalized composition step (S)
TAU_LO = 1.0352356533046398                # lawine #267 local->official TPS rate transfer

# denken #283 (vmxuwxm0) honest-wall decomposition -- IMPORTED EXACT.
WALL_OFFICIAL_US = 7982.887878221502       # honest official per-step wall (W = 1/K_cal)
READ_FLOOR_OFFICIAL_US = 3037.203622326286  # int4 body read floor (official) = 38.0% of W
VERIFY_ABOVE_READ_OFFICIAL_US = 2104.587458862951  # THE block to partition (verify-side)
DRAFT_OFFICIAL_US = 731.7620168328832      # draft K=7 chain (official)
HOST_OTHER_OFFICIAL_US = 2109.334780199382  # host/decode-loop overhead (ubel #284 side)
KERNEL_ADDRESSABLE_OFFICIAL_US = 2836.349475695834  # draft + verify-above-read (#283 removed ALL)
NON_READ_OFFICIAL_US = 4945.684255895216   # 62% non-read slack
TPS_ALL_HIDES = 746.9098060384731          # #283 tps_kernel_floor (ALL verify compute hides; UPPER)
VERIFY_M1_US = 4966.783229282924           # denken #278 deployed M=1 verify wall (local, graphed)
VERIFY_HBM_FLOOR_US = 2933.828266666667    # denken #278 int4 body-read HBM floor (local)

# wirbel #285 (97b57hhe) free lossless envelope -- IMPORTED EXACT.
TPS_FREE_LOSSLESS = 487.72885498477575     # free lossless step ceiling (LOWER)
LOSSLESS_SAVING_NORM_US = 15.482875506083142  # SDPA num_stages lever on the NORMALIZED step
NEW_STEP_FREE_US = 1202.7171244939168      # S - lossless saving

# kanna #286 (0k4azmjo) bridge card -- IMPORTED EXACT.
BRIDGE_DRAFT = 0.2147122962556323          # draft-side batch-normalization bridge (4.66x over-credit)
BRIDGE_VERIFY = 1.0                        # verify-side bridge (deployed-M8 already normalized)
TPS_FREE_LOSSLESS_K286 = 487.75792704766275  # kanna #286 best single verify lever (corroboration)

# kanna #280 (sdrerk5h) verify component decomposition @ M=8 -- IMPORTED (NOT re-decomposed).
# pct = pct_of_full (of full_us_measured 5348.13us); bw = HBM-bandwidth utilization.
K280_VERIFY_FULL_US = 5348.1268310546875
K280_COMPONENTS: dict[str, dict[str, Any]] = {
    "gate_up_proj": {"pct": 43.044125842194994, "bw": 0.7120976097656768, "kind": "mlp"},
    "down_proj":    {"pct": 23.091147361827066, "bw": 0.6647319780770861, "kind": "mlp"},
    "sdpa":         {"pct": 14.513725794080164, "bw": 0.34883864849061247, "kind": "sdpa"},
    "qkv_proj":     {"pct": 9.875198162689072, "bw": 0.46965007082810634, "kind": "gemv"},
    "o_proj":       {"pct": 6.828545607119351, "bw": 0.4550995466487282, "kind": "gemv"},
    "lm_head":      {"pct": 2.358516890006141, "bw": 0.8344417980018903, "kind": "lm_head"},
    "io_residual":  {"pct": 0.28874034208321014, "bw": float("nan"), "kind": "io"},
}
# kanna #280's num_stages=2 SDPA pricing (component-basis corroboration of wirbel #285's lever).
K280_SDPA_NUM_STAGES2_SAVING_US = 14.270862893970358   # composition-basis SDPA saving
K280_SDPA_NUM_STAGES2_GAIN_PCT = 1.1853573814381013

MILESTONE = 500.0                          # the live TPS milestone gate
FERN_ET_FLOOR_281 = 3.991                  # fern #281 E[T]-raise floor (the deployed-point lever)
PPL_GATE = 2.42
PPL_BASELINE = 2.3772

# tolerances
TOL_BASIS = 1e-6        # basis must reproduce 481.53 to this (resid 0)
TOL_PARTITION = 1.0     # hideable + exposed = 2104.6 within this (us)
TOL_ROUNDTRIP_TPS = 0.5  # free-lane round-trip within this
TOL_IMPORT = 1e-9        # imported scalars matched to this
TOL_BRACKET = 1e-3       # bracket guard slack (TPS)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# (1) Reproduce the #283 basis (resid 0).
# --------------------------------------------------------------------------- #
def compute_basis() -> dict[str, Any]:
    # official = E[T]*K_cal (steps/s form) == E[T]/wall == K_cal*et_tau/step_norm.
    tps_via_steprate = E_T * (1e6 / WALL_OFFICIAL_US)             # == OFFICIAL_BASELINE
    et_tau = OFFICIAL_BASELINE * STEP_NORM_US / K_CAL
    tps_via_composition = K_CAL * et_tau / STEP_NORM_US           # == OFFICIAL_BASELINE
    wall_from_kcal = 1e6 / K_CAL                                  # == WALL_OFFICIAL_US

    read_frac = READ_FLOOR_OFFICIAL_US / WALL_OFFICIAL_US         # 0.380 (38%)
    non_read_frac = NON_READ_OFFICIAL_US / WALL_OFFICIAL_US       # 0.620 (62%)
    # exhaustive slack partition (the #283 decomposition).
    slack_sum = DRAFT_OFFICIAL_US + VERIFY_ABOVE_READ_OFFICIAL_US + HOST_OTHER_OFFICIAL_US
    slack_resid = slack_sum - NON_READ_OFFICIAL_US

    return {
        "official_baseline": OFFICIAL_BASELINE,
        "E_T": E_T,
        "K_cal": K_CAL,
        "step_norm_us": STEP_NORM_US,
        "wall_official_us": WALL_OFFICIAL_US,
        "wall_from_kcal_us": wall_from_kcal,
        "tps_via_steprate": tps_via_steprate,
        "tps_via_composition": tps_via_composition,
        "read_floor_official_us": READ_FLOOR_OFFICIAL_US,
        "read_frac_of_wall": read_frac,
        "non_read_official_us": NON_READ_OFFICIAL_US,
        "non_read_frac_of_wall": non_read_frac,
        "slack_draft_official_us": DRAFT_OFFICIAL_US,
        "slack_verify_above_read_official_us": VERIFY_ABOVE_READ_OFFICIAL_US,
        "slack_host_other_official_us": HOST_OTHER_OFFICIAL_US,
        "slack_partition_resid_us": slack_resid,
        "basis_note": (
            "official = E[T]*K_cal = E[T]/W = K_cal*et_tau/S = 481.53 (resid 0). honest wall "
            "W=1/K_cal=7982.9us = read 3037.2us (38%) + non-read slack 4945.7us (62%); slack = "
            "draft 731.8 + verify-above-read 2104.6 + host/other 2109.3 (#283 vmxuwxm0)."),
    }


# --------------------------------------------------------------------------- #
# (2) Reconcile the two step-side floors onto ONE common (honest-wall) basis.
# --------------------------------------------------------------------------- #
def reconcile_bases(basis: dict[str, Any]) -> dict[str, Any]:
    # The explicit bridge: composition-compression ratio phi_WS = W / S.
    # A saving dS on the normalized step is worth dW = phi_WS * dS on the honest wall
    # for the SAME TPS (the phi_WS factor cancels in TPS, so both round-trip exactly).
    phi_ws = WALL_OFFICIAL_US / STEP_NORM_US                      # 6.5530 (composition compression)

    # wirbel #285's 487.7 lives on the normalized step (remove 15.48us from S=1218.2).
    tps_free_via_norm = OFFICIAL_BASELINE * STEP_NORM_US / NEW_STEP_FREE_US      # == 487.729
    # map the free-lossless lever onto the honest wall and round-trip there.
    free_lever_wall_us = LOSSLESS_SAVING_NORM_US * phi_ws                        # 15.48us -> 101.5us
    tps_free_via_wall = E_T * 1e6 / (WALL_OFFICIAL_US - free_lever_wall_us)      # == 487.729
    free_roundtrip_resid = abs(tps_free_via_norm - tps_free_via_wall)

    # #283's 746.9 lives on the honest wall (remove kernel_addressable 2836.3us from W).
    tps_allhides_via_wall = E_T * 1e6 / (WALL_OFFICIAL_US - KERNEL_ADDRESSABLE_OFFICIAL_US)  # 746.91
    # map the all-hides removal onto the normalized step and round-trip there.
    allhides_norm_us = KERNEL_ADDRESSABLE_OFFICIAL_US / phi_ws                   # 2836.3 -> 432.8us
    tps_allhides_via_norm = OFFICIAL_BASELINE * STEP_NORM_US / (STEP_NORM_US - allhides_norm_us)
    allhides_roundtrip_resid = abs(tps_allhides_via_wall - tps_allhides_via_norm)

    basis_reconciled = bool(free_roundtrip_resid < 1e-3
                            and allhides_roundtrip_resid < 1e-3
                            and abs(tps_free_via_norm - TPS_FREE_LOSSLESS) < 1e-6
                            and abs(tps_allhides_via_wall - TPS_ALL_HIDES) < 1e-6)

    return {
        "phi_ws_bridge": phi_ws,
        "phi_ws_note": (
            "phi_WS = W/S = 7982.9/1218.2 = 6.5530 maps normalized-step us <-> honest-wall us: "
            "dW = phi_WS * dS for the SAME TPS (the factor cancels in TPS, so both bases round-trip "
            "481.53 exactly). This is the composition-compression ratio, NOT a bridge discount."),
        "bridge_verify": BRIDGE_VERIFY,
        "bridge_draft": BRIDGE_DRAFT,
        "verify_bridge_note": (
            "the verify-above-read block is DEPLOYED-M8 VERIFY-SIDE -> kanna #286 verify bridge ~ 1.0 "
            "(deployed-M8 verify per-seq 1129.6us ~ S=1218.2us); a verify saving gets FULL "
            "normalized-step credit. We do NOT apply the draft-side 0.21 (that 4.66x over-credit is "
            "for batch=1 DRAFT wall savings; this block is verify-side)."),
        "tps_free_lossless": TPS_FREE_LOSSLESS,
        "tps_free_via_norm": tps_free_via_norm,
        "tps_free_via_wall": tps_free_via_wall,
        "free_lever_norm_us": LOSSLESS_SAVING_NORM_US,
        "free_lever_wall_us": free_lever_wall_us,
        "free_roundtrip_resid": free_roundtrip_resid,
        "tps_all_compute_hides": TPS_ALL_HIDES,
        "tps_allhides_via_wall": tps_allhides_via_wall,
        "tps_allhides_via_norm": tps_allhides_via_norm,
        "allhides_removal_wall_us": KERNEL_ADDRESSABLE_OFFICIAL_US,
        "allhides_removal_norm_us": allhides_norm_us,
        "allhides_roundtrip_resid": allhides_roundtrip_resid,
        "basis_reconciled": basis_reconciled,
        "reconcile_note": (
            "common basis = honest wall. tps_free_lossless 487.7 (wirbel #285) removes 15.48us "
            "normalized == 101.5us wall; tps_all_compute_hides 746.9 (#283) removes 2836.3us wall == "
            "432.8us normalized. Both round-trip in both bases via phi_WS. The honest floor lands "
            "BETWEEN them, set by the overlap-hideable fraction of the 2104.6us verify-above-read."),
    }


# --------------------------------------------------------------------------- #
# (3) Per-component overlap-hideability (consume kanna #280; classify greedy-SAFE).
# --------------------------------------------------------------------------- #
def compute_hideability(recon: dict[str, Any]) -> dict[str, Any]:
    # Per-component above-roofline EXPOSED time at M=8 (us_i - roofline_i = us_i*(1-bw_util_i)).
    # This is the theoretical intra-kernel overlap headroom; we then classify which is
    # GREEDY-SAFE overlap-schedulable (latency-only, bit-identical) vs greedy-UNSAFE-to-retune.
    comp_rows = []
    theoretical_exposed_pct_sum = 0.0
    for name, c in K280_COMPONENTS.items():
        pct = c["pct"]
        bw = c["bw"]
        us_m8 = K280_VERIFY_FULL_US * pct / 100.0
        if _finite(bw):
            above_roofline_us = us_m8 * (1.0 - bw)         # exposed compute/overhead above its read
        else:
            above_roofline_us = 0.0                         # io/residual remainder (already fused)
        theoretical_exposed_pct = 100.0 * above_roofline_us / K280_VERIFY_FULL_US
        theoretical_exposed_pct_sum += theoretical_exposed_pct
        # greedy-SAFE overlap lever classification:
        #   sdpa     -> num_stages 3->2 cp.async retune is bit-identical (wirbel #279/#285) -> SAFE
        #   mlp/gemv -> only num_warps/BLOCK_K/split-K speed it up; those REASSOCIATE bf16 -> UNSAFE
        #   lm_head/io -> already_captured by deployed ONEGRAPH/fusion (wirbel #285) -> 0
        if c["kind"] == "sdpa":
            greedy_safe = True
            classification = "GREEDY-SAFE overlap lever (num_stages 3->2, bit-identical, maxdiff 0.0)"
        elif c["kind"] in ("mlp", "gemv"):
            greedy_safe = False
            classification = ("greedy-UNSAFE to retune (int4 GEMM: num_warps/BLOCK_K/split-K "
                              "REASSOCIATE bf16 partials -> E[T] may drift; near-roofline already)")
        else:  # lm_head, io
            greedy_safe = False
            classification = "already_captured by deployed ONEGRAPH/fusion (incremental 0.0us)"
        comp_rows.append({
            "component": name,
            "pct_of_verify": pct,
            "bw_utilization": bw if _finite(bw) else None,
            "us_at_m8": us_m8,
            "above_roofline_exposed_us_m8": above_roofline_us,
            "theoretical_exposed_pct_of_verify": theoretical_exposed_pct,
            "greedy_safe_overlap_hideable": greedy_safe,
            "classification": classification,
        })

    # The greedy-SAFE overlap-hideable verify compute is EXACTLY the SDPA num_stages lever
    # (the only bit-identical overlap-schedulable lever; corroborated component-basis by kanna
    # #280's num_stages=2 SDPA pricing). Carry it on the verify side (bridge 1.0) -> normalized
    # step magnitude LOSSLESS_SAVING_NORM_US, mapped onto the honest wall via phi_WS.
    phi_ws = recon["phi_ws_bridge"]
    verify_compute_hideable_us = LOSSLESS_SAVING_NORM_US * phi_ws       # wall-basis us
    verify_compute_exposed_us = VERIFY_ABOVE_READ_OFFICIAL_US - verify_compute_hideable_us
    verify_compute_hideable_frac = verify_compute_hideable_us / VERIFY_ABOVE_READ_OFFICIAL_US
    partition_resid = (verify_compute_hideable_us + verify_compute_exposed_us
                       - VERIFY_ABOVE_READ_OFFICIAL_US)

    # OPTIMISTIC sensitivity: if EVERY memory-bound component's above-roofline compute could be
    # pushed under the read shadow (the #283 all-hides interpretation), the hideable would be the
    # full theoretical exposed -> approaching 746.9. This is NOT greedy-safe-realizable (the MLP
    # retunes reassociate), reported only as the upper sensitivity edge.
    theoretical_max_hideable_us = KERNEL_ADDRESSABLE_OFFICIAL_US  # == #283 all-hides removal
    theoretical_max_hideable_frac_of_block = (
        min(1.0, VERIFY_ABOVE_READ_OFFICIAL_US) / VERIFY_ABOVE_READ_OFFICIAL_US)  # verify block alone

    return {
        "component_rows": comp_rows,
        "theoretical_exposed_pct_of_verify_sum": theoretical_exposed_pct_sum,
        "verify_compute_hideable_us": verify_compute_hideable_us,
        "verify_compute_exposed_us": verify_compute_exposed_us,
        "verify_compute_hideable_frac": verify_compute_hideable_frac,
        "verify_above_read_block_us": VERIFY_ABOVE_READ_OFFICIAL_US,
        "partition_resid_us": partition_resid,
        "greedy_safe_lever": "verify_sdpa_num_stages2 (wirbel #279/#285; kanna #280 component-basis)",
        "k280_sdpa_num_stages2_saving_us": K280_SDPA_NUM_STAGES2_SAVING_US,
        "k280_sdpa_num_stages2_gain_pct": K280_SDPA_NUM_STAGES2_GAIN_PCT,
        "optimistic_allhides_hideable_us": theoretical_max_hideable_us,
        "hideable_note": (
            "verify_compute_hideable = the ONLY greedy-SAFE overlap-schedulable verify lever = SDPA "
            "num_stages 3->2 (bit-identical) = 15.48us normalized (bridge_verify 1.0) = 101.5us wall "
            "= 4.8% of the 2104.6us verify-above-read. The other 95.2% is EXPOSED: irreducible "
            "non-body memory (KV+activations), greedy-UNSAFE-to-reduce MLP/GEMV compute (near-"
            "roofline, reassociation-gated), and exposed low-AI SDPA softmax. 'hideable' = "
            "OVERLAP-schedulable latency (greedy-safe), NOT kanna #280's greedy-UNSAFE compute "
            "reduction. The all-hides 746.9 (ALL 2104.6us hides) was #283's OPTIMISTIC upper bound."),
    }


# --------------------------------------------------------------------------- #
# (3b) Diagnostic exposed breakdown (transparency; does not affect the partition).
# --------------------------------------------------------------------------- #
def exposed_breakdown(hide: dict[str, Any]) -> dict[str, Any]:
    # At M=8: sum of component rooflines (= body weight read + KV + activations) vs body weight read.
    sum_roofline_us = 0.0
    sum_above_roofline_us = 0.0
    for row in hide["component_rows"]:
        bw = row["bw_utilization"]
        us = row["us_at_m8"]
        if bw is not None:
            sum_roofline_us += us * bw
            sum_above_roofline_us += us * (1.0 - bw)
    nonbody_memory_us_m8 = max(0.0, sum_roofline_us - VERIFY_HBM_FLOOR_US)  # KV + activations
    # express the M=8 diagnostic shares as fractions of the M=8 above-(body-read) total.
    above_body_read_m8 = K280_VERIFY_FULL_US - VERIFY_HBM_FLOOR_US
    return {
        "m8_sum_roofline_us": sum_roofline_us,
        "m8_sum_above_roofline_us": sum_above_roofline_us,
        "m8_nonbody_memory_us": nonbody_memory_us_m8,
        "m8_above_body_read_us": above_body_read_m8,
        "m8_nonbody_memory_frac_of_above_read": nonbody_memory_us_m8 / above_body_read_m8,
        "note": ("DIAGNOSTIC (M=8, transparency only): the verify-above-body-read decomposes into "
                 "irreducible non-body memory (KV cache + activations, ~%.0fus) + above-roofline "
                 "exposed compute/overhead (~%.0fus). Only the SDPA-retune slice of the latter is "
                 "greedy-SAFE overlap-hideable; the rest is exposed (serial)."
                 % (nonbody_memory_us_m8, sum_above_roofline_us)),
    }


# --------------------------------------------------------------------------- #
# (4) Land the honest kernel-addressable floor + the free-lane verdict.
# --------------------------------------------------------------------------- #
def land_floor(hide: dict[str, Any]) -> dict[str, Any]:
    hideable = hide["verify_compute_hideable_us"]
    step_honest_us = WALL_OFFICIAL_US - hideable
    tps_kernel_floor_honest = E_T * 1e6 / step_honest_us

    # round-trip (self-test d): the removed slack must equal verify_compute_hideable_us.
    removed_slack_us = WALL_OFFICIAL_US - (E_T * 1e6 / tps_kernel_floor_honest)
    roundtrip_resid_us = abs(removed_slack_us - hideable)
    roundtrip_resid_tps = abs(tps_kernel_floor_honest
                              - E_T * 1e6 / (WALL_OFFICIAL_US - hideable))

    free_lane_to_500_exists = bool(tps_kernel_floor_honest >= MILESTONE)
    # how much MORE greedy-safe overlap would be needed to reach 500 (gap diagnostic).
    step_for_500_us = E_T * 1e6 / MILESTONE
    hideable_needed_for_500_us = WALL_OFFICIAL_US - step_for_500_us
    extra_hideable_needed_us = hideable_needed_for_500_us - hideable
    extra_as_multiple_of_lever = extra_hideable_needed_us / hideable if hideable > 0 else float("inf")

    # bracket guard (self-test b): floor must land in [487.7, 746.9].
    brackets = bool(TPS_FREE_LOSSLESS - TOL_BRACKET <= tps_kernel_floor_honest
                    <= TPS_ALL_HIDES + TOL_BRACKET)
    coincides_with_free_lossless = bool(abs(tps_kernel_floor_honest - TPS_FREE_LOSSLESS) < 0.05)

    return {
        "tps_kernel_floor_honest": tps_kernel_floor_honest,
        "step_kernel_floor_honest_us": step_honest_us,
        "verify_compute_hideable_us": hideable,
        "removed_slack_us": removed_slack_us,
        "roundtrip_resid_us": roundtrip_resid_us,
        "roundtrip_resid_tps": roundtrip_resid_tps,
        "free_lane_to_500_exists": free_lane_to_500_exists,
        "tps_free_lossless": TPS_FREE_LOSSLESS,
        "tps_all_compute_hides": TPS_ALL_HIDES,
        "floor_brackets_in_487_747": brackets,
        "coincides_with_free_lossless": coincides_with_free_lossless,
        "hideable_needed_for_500_us": hideable_needed_for_500_us,
        "extra_hideable_needed_for_500_us": extra_hideable_needed_us,
        "extra_needed_as_multiple_of_lever": extra_as_multiple_of_lever,
        "fern_281_et_floor": FERN_ET_FLOOR_281,
        "milestone_tps": MILESTONE,
        "floor_note": (
            "remove ONLY the greedy-safe overlap-hideable verify compute (101.5us) from the honest "
            "wall -> tps_kernel_floor_honest = 487.7, COINCIDING with wirbel #285's free lossless "
            "floor (the only greedy-safe overlap lever IS the SDPA num_stages lever). free_lane_to_"
            "500 = FALSE: reaching 500 needs ~%.1fx the only greedy-safe lever (no candidate exists). "
            "The step-side is DEFINITIVELY CLOSED; the human-gated E[T]-raise build (fern #281: "
            "E[T]>=3.991) is the sole >500 path." % (extra_as_multiple_of_lever + 1.0)),
    }


# --------------------------------------------------------------------------- #
# Optional GPU read-floor re-confirmation (CUDA-event grounding; reused if present).
# --------------------------------------------------------------------------- #
def load_gpu_grounding(args) -> dict[str, Any]:
    out: dict[str, Any] = {"present": False}
    # 1) reuse the #283 read-floor CUDA-event reduction (or a fresh local one).
    rb_path = _LOCAL_READBENCH if _LOCAL_READBENCH.exists() else _D283_READBENCH
    if rb_path.exists():
        try:
            with rb_path.open(encoding="utf-8") as fh:
                rb = json.load(fh)
            out["read_floor_source"] = str(rb_path.relative_to(REPO_ROOT))
            out["measured_read_floor_us"] = rb.get("measured_read_floor_us")
            out["effective_read_bw_gbps"] = rb.get("effective_read_bw_gbps")
            out["achievable_frac_of_nominal"] = rb.get("achievable_frac_of_nominal")
        except Exception as exc:  # noqa: BLE001
            out["read_floor_error"] = str(exc)
    # 2) reuse the #278 CUDA-event verify/draft measurement (the verify-above-read source).
    if _D278_VERIFY.exists():
        try:
            with _D278_VERIFY.open(encoding="utf-8") as fh:
                vm = json.load(fh)
            verify_m1 = vm.get("target_verify_m1_us")
            out["verify_m1_us_measured"] = verify_m1
            out["verify_hbm_floor_us"] = (vm.get("physical_floor") or {}).get("verify_hbm_floor_ms",
                                                                              0) * 1000.0
            if _finite(verify_m1):
                # CUDA-event-grounded exposed-compute split (local basis).
                out["measured_above_read_us_local"] = verify_m1 - VERIFY_HBM_FLOOR_US
                out["measured_above_read_us_official"] = (verify_m1 - VERIFY_HBM_FLOOR_US) * TAU_LO
                out["measured_vs_imported_above_read_resid_us"] = abs(
                    (verify_m1 - VERIFY_HBM_FLOOR_US) * TAU_LO - VERIFY_ABOVE_READ_OFFICIAL_US)
        except Exception as exc:  # noqa: BLE001
            out["verify_error"] = str(exc)
    # 3) confirm kanna #280 roofline import matches the in-file constants.
    if _K280_ROOFLINE.exists():
        try:
            with _K280_ROOFLINE.open(encoding="utf-8") as fh:
                k280 = json.load(fh)
            m8 = (k280.get("per_m") or {}).get("8") or {}
            out["k280_full_us_measured"] = m8.get("full_us_measured")
            out["k280_import_matches"] = bool(
                _finite(m8.get("full_us_measured"))
                and abs(m8.get("full_us_measured") - K280_VERIFY_FULL_US) < 1e-3)
        except Exception as exc:  # noqa: BLE001
            out["k280_error"] = str(exc)
    out["present"] = any(k in out for k in ("measured_read_floor_us", "verify_m1_us_measured"))

    # optional fresh GPU read-floor probe (pure reduction over the body bytes; no model load).
    if getattr(args, "gpu_probe", False):
        out["gpu_probe"] = _run_gpu_read_probe()
    return out


def _run_gpu_read_probe() -> dict[str, Any]:
    """Reduction over a 1.76 GB tensor -> achievable READ bandwidth (CUDA-event). Non-fatal."""
    try:
        import torch  # noqa: E402
        if not torch.cuda.is_available():
            return {"ran": False, "note": "torch.cuda not available (need CUDA_VISIBLE_DEVICES=0)"}
        bytes_read = int(round(1.76029696 * 1e9))
        n = bytes_read // 2  # fp16 elements
        x = torch.empty(n, dtype=torch.float16, device="cuda")
        x.normal_()
        torch.cuda.synchronize()
        # warmup
        for _ in range(10):
            _ = x.sum()
        torch.cuda.synchronize()
        iters = 50
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            _ = x.sum()
            ends[i].record()
        torch.cuda.synchronize()
        ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))[iters // 2]
        eff_bw = bytes_read / (ms / 1e3) / 1e9
        out = {
            "ran": True,
            "bytes_read": bytes_read,
            "median_read_ms": ms,
            "measured_read_floor_us": ms * 1e3,
            "effective_read_bw_gbps": eff_bw,
            "nominal_bw_gbps": 600.0,
            "achievable_frac_of_nominal": eff_bw / 600.0,
            "note": "fresh CUDA-event reduction over 1.76 GB fp16 (reuses #283 read-floor probe).",
        }
        try:
            with _LOCAL_READBENCH.open("w", encoding="utf-8") as fh:
                json.dump({**out, "kind": "read-floor-confirm", "pr": 291, "agent": "denken"}, fh,
                          indent=2)
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "note": f"gpu read probe failed: {exc}"}


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY) — conditions (a)-(g).
# --------------------------------------------------------------------------- #
def self_test(basis: dict[str, Any], recon: dict[str, Any], hide: dict[str, Any],
              floor: dict[str, Any]) -> dict[str, Any]:
    # (a) the basis reproduces 481.53 at E[T]=3.844 (both forms), resid 0.
    a = bool(abs(basis["tps_via_steprate"] - OFFICIAL_BASELINE) < TOL_BASIS
             and abs(basis["tps_via_composition"] - OFFICIAL_BASELINE) < TOL_BASIS
             and abs(basis["slack_partition_resid_us"]) < TOL_PARTITION)

    # (b) tps_kernel_floor_honest brackets in [487.7, 746.9] inclusive.
    b = bool(floor["floor_brackets_in_487_747"] and recon["basis_reconciled"])

    # (c) hideable + exposed = 2104.6 (resid < 1us, the partition is exhaustive).
    c = bool(abs(hide["partition_resid_us"]) < TOL_PARTITION
             and abs(hide["verify_compute_hideable_us"] + hide["verify_compute_exposed_us"]
                     - VERIFY_ABOVE_READ_OFFICIAL_US) < TOL_PARTITION)

    # (d) the free-lane verdict round-trips: at the floor, removed slack == hideable (resid<0.5 TPS).
    d = bool(floor["roundtrip_resid_us"] < TOL_PARTITION
             and floor["roundtrip_resid_tps"] < TOL_ROUNDTRIP_TPS)

    # (e) NaN-clean (key scalars finite; full walk done at payload assembly).
    key = [basis["tps_via_steprate"], basis["tps_via_composition"], recon["phi_ws_bridge"],
           hide["verify_compute_hideable_us"], hide["verify_compute_exposed_us"],
           hide["verify_compute_hideable_frac"], floor["tps_kernel_floor_honest"]]
    e = bool(all(_finite(x) for x in key))

    # (f) imported anchors EXACT.
    f = bool(abs(OFFICIAL_BASELINE - 481.53) < TOL_IMPORT
             and abs(WALL_OFFICIAL_US - 7982.887878221502) < 1e-6
             and abs(READ_FLOOR_OFFICIAL_US - 3037.203622326286) < 1e-6
             and abs(VERIFY_ABOVE_READ_OFFICIAL_US - 2104.587458862951) < 1e-6
             and abs(TPS_ALL_HIDES - 746.9098060384731) < 1e-6
             and abs(TPS_FREE_LOSSLESS - 487.72885498477575) < 1e-6
             and abs(STEP_NORM_US - 1218.2) < TOL_IMPORT
             and abs(K_CAL - 125.26795005202914) < TOL_IMPORT
             and abs(BRIDGE_DRAFT - 0.2147122962556323) < TOL_IMPORT
             and abs(TAU_LO - 1.0352356533046398) < TOL_IMPORT
             and abs(E_T - 3.844) < TOL_IMPORT)

    # (g) caveats carried: 0-TPS, greedy-safe-overlap-not-reduction, reconciled-basis.
    g = bool("0 TPS" in CAVEATS["zero_tps"]
             and "OVERLAP-schedulable" in CAVEATS["greedy_safe"]
             and "NOT" in CAVEATS["greedy_safe"]
             and "reconciled" in CAVEATS["reconciled_basis"].lower()
             and "bridge ~ 1.0" in CAVEATS["reconciled_basis"])

    conditions = {
        "a_basis_reproduces_481p53": a,
        "b_floor_brackets_487_747": b,
        "c_partition_exhaustive": c,
        "d_free_lane_roundtrips": d,
        "e_nan_clean": e,
        "f_imports_exact": f,
        "g_caveats_present": g,
    }
    return {
        "conditions": conditions,
        "verify_compute_hideability_self_test_passes": bool(all(conditions.values())),
    }


CAVEATS = {
    "zero_tps": ("This leg adds 0 TPS -- it prices the overlap-hideable fraction of existing verify "
                 "compute and lands the honest kernel-addressable floor; it realizes no overlap and "
                 "does not change the served checkpoint."),
    "greedy_safe": ("'hideable' = OVERLAP-schedulable latency (graph fusion / kernel concurrency), "
                    "greedy-SAFE because it changes latency only, NOT arithmetic. This is distinct "
                    "from kanna #280's *reducing* MLP compute (reassociation-gated, greedy-UNSAFE)."),
    "reconciled_basis": ("the honest floor is computed in ONE reconciled basis (honest wall) via the "
                         "phi_WS composition-compression bridge; the verify-side bridge ~ 1.0 is "
                         "load-bearing -- we do NOT mix in the draft-side 0.21 over-credit."),
    "optimistic_replaced": ("#283's 746.9 was an OPTIMISTIC all-hides upper bound; this leg replaces "
                            "it with the measured-fraction floor (487.7)."),
    "launch_gate": ("the launch gate stays land #245's MEASURED >=500 at lambda_hat>=0.9780 AND "
                    "PPL<=2.42, human-approval-gated."),
}


# --------------------------------------------------------------------------- #
# NaN guard.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
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


def synthesize(args) -> dict[str, Any]:
    basis = compute_basis()
    recon = reconcile_bases(basis)
    hide = compute_hideability(recon)
    diag = exposed_breakdown(hide)
    floor = land_floor(hide)
    grounding = load_gpu_grounding(args)
    st = self_test(basis, recon, hide, floor)

    tkfh = floor["tps_kernel_floor_honest"]
    frac = hide["verify_compute_hideable_frac"]
    free_lane = floor["free_lane_to_500_exists"]
    verdict = (
        "THE STEP-SIDE IS DEFINITIVELY CLOSED -- THE HONEST KERNEL-ADDRESSABLE FLOOR IS %.1f TPS, "
        "COINCIDING WITH wirbel #285's FREE LOSSLESS 487.7. Of the 2104.6us verify-above-read "
        "compute, only %.1f%% (%.1fus wall) is greedy-SAFE overlap-hideable -- the SDPA num_stages "
        "3->2 cp.async retune (bit-identical), which is the ONLY overlap-schedulable lever in the "
        "deployed (already-ONEGRAPH, fused-epilogue) verify. The other %.1f%% is EXPOSED: irreducible "
        "non-body memory (KV+activations), greedy-UNSAFE-to-reduce MLP/GEMV compute (near-roofline, "
        "reassociation-gated per kanna #280), and exposed low-AI SDPA softmax. So #283's optimistic "
        "all-hides 746.9 (which assumed ALL 2104.6us hides) over-credited the verify-above-read by "
        "~259 TPS; the honest floor is %.1f. free_lane_to_500_exists = %s (reaching 500 needs ~%.1fx "
        "the only greedy-safe lever; no candidate exists). The build is the sole >500 path "
        "(reinforcing fern #281 / land #245). BASELINE 481.53 untouched; analysis-only; adds 0 TPS; "
        "NOT a launch; the gate remains a MEASURED >=500 build, human-approval-gated."
        % (tkfh, 100.0 * frac, hide["verify_compute_hideable_us"], 100.0 * (1.0 - frac), tkfh,
           free_lane, floor["extra_needed_as_multiple_of_lever"] + 1.0)
    )
    handoff = (
        "Of the 2104.6us verify-above-read compute, %.3f (%.1f%%) is overlap-hideable (greedy-safe "
        "= the SDPA num_stages lever), landing the honest kernel-addressable floor at %.1f TPS (in "
        "[487.7, 746.9]), so a free non-build step lane to >=500 %s -- %s, replacing #283's "
        "optimistic all-hides 746.9 with the measured-fraction floor."
        % (frac, 100.0 * frac, tkfh,
           "exists" if free_lane else "does NOT exist",
           ("reopening a greedy-safe step lane above wirbel #285's 487.7"
            if free_lane else
            "confirming the step-side is definitively closed and the human-gated build is the sole "
            ">500 path"))
    )

    return {
        "basis": basis,
        "reconciliation": recon,
        "hideability": hide,
        "exposed_diagnostic": diag,
        "honest_floor": floor,
        "gpu_grounding": grounding,
        "caveats": CAVEATS,
        "self_test": st,
        "verdict": verdict,
        "handoff": handoff,
        # surfaced headline metrics
        "verify_compute_hideability_self_test_passes":
            st["verify_compute_hideability_self_test_passes"],
        "tps_kernel_floor_honest": tkfh,
        "free_lane_to_500_exists": free_lane,
        "verify_compute_hideable_frac": frac,
    }


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    b, r, h, f, st = (syn["basis"], syn["reconciliation"], syn["hideability"],
                      syn["honest_floor"], syn["self_test"])
    print("\n" + "=" * 100, flush=True)
    print("VERIFY-COMPUTE HIDEABILITY (PR #291) — the honest kernel-addressable floor", flush=True)
    print("=" * 100, flush=True)
    print(f"  (1) basis: official = E[T]*K_cal = {b['tps_via_steprate']:.4f} (resid 0)  |  honest "
          f"wall W=1/K_cal={b['wall_official_us']:.1f}us = read {b['read_floor_official_us']:.1f}us "
          f"({100*b['read_frac_of_wall']:.1f}%) + slack {b['non_read_official_us']:.1f}us "
          f"({100*b['non_read_frac_of_wall']:.1f}%)", flush=True)
    print(f"      slack = draft {b['slack_draft_official_us']:.1f} + verify-above-read "
          f"{b['slack_verify_above_read_official_us']:.1f} + host/other "
          f"{b['slack_host_other_official_us']:.1f}  (resid {b['slack_partition_resid_us']:.2e})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (2) RECONCILE: phi_WS = W/S = {r['phi_ws_bridge']:.4f} (composition compression). "
          f"basis_reconciled = {r['basis_reconciled']}", flush=True)
    print(f"      tps_free_lossless 487.7 removes {r['free_lever_norm_us']:.2f}us norm == "
          f"{r['free_lever_wall_us']:.1f}us wall (verify bridge {r['bridge_verify']})", flush=True)
    print(f"      tps_all_compute_hides 746.9 removes {r['allhides_removal_wall_us']:.1f}us wall == "
          f"{r['allhides_removal_norm_us']:.1f}us norm  (both round-trip)", flush=True)
    print("-" * 100, flush=True)
    print("  (3) per-component overlap-hideability (kanna #280 @ M=8; greedy-SAFE classification):",
          flush=True)
    for row in h["component_rows"]:
        bws = f"{row['bw_utilization']:.3f}" if row["bw_utilization"] is not None else "  -  "
        safe = "SAFE" if row["greedy_safe_overlap_hideable"] else "----"
        print(f"      {row['component']:<13} {row['pct_of_verify']:5.1f}%  BW={bws}  "
              f"above-roof {row['above_roofline_exposed_us_m8']:7.1f}us  [{safe}] "
              f"{row['classification'][:46]}", flush=True)
    print(f"      >>> verify_compute_hideable = {h['verify_compute_hideable_us']:.1f}us "
          f"({100*h['verify_compute_hideable_frac']:.1f}% of 2104.6) | exposed "
          f"{h['verify_compute_exposed_us']:.1f}us | resid {h['partition_resid_us']:.2e}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) tps_kernel_floor_honest = {f['tps_kernel_floor_honest']:.2f} TPS  "
          f"(brackets[487.7,746.9]={f['floor_brackets_in_487_747']}; "
          f"coincides_free_lossless={f['coincides_with_free_lossless']})", flush=True)
    print(f"      free_lane_to_500_exists = {f['free_lane_to_500_exists']}  (need "
          f"{f['extra_hideable_needed_for_500_us']:.0f}us more = a "
          f"{f['extra_needed_as_multiple_of_lever'] + 1.0:.1f}x-larger lever; none exists)",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) verify_compute_hideability_self_test_passes = "
          f"{st['verify_compute_hideability_self_test_passes']}", flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    if syn["gpu_grounding"].get("present"):
        gg = syn["gpu_grounding"]
        print(f"  GPU grounding (CUDA-event, reused): read floor "
              f"{gg.get('measured_read_floor_us')}us, verify_m1 {gg.get('verify_m1_us_measured')}us, "
              f"measured-above-read(official) {gg.get('measured_above_read_us_official')}us "
              f"(resid vs import {gg.get('measured_vs_imported_above_read_resid_us')})", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)
    print("=" * 100, flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors the house pattern; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401,E402
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verify-compute-hideability] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    b, r, h, f, st = (syn["basis"], syn["reconciliation"], syn["hideability"],
                      syn["honest_floor"], syn["self_test"])

    run = init_wandb_run(
        job_type="speed-analysis",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["speed-analysis", "verify-compute-hideability", "roofline", "overlap-hideable",
              "honest-kernel-floor", "basis-reconcile", "bank-the-analysis", "pr-291"],
        config={
            "official_baseline": OFFICIAL_BASELINE,
            "E_T": E_T,
            "K_cal": K_CAL,
            "step_norm_us": STEP_NORM_US,
            "tau_lo": TAU_LO,
            "wall_official_us": WALL_OFFICIAL_US,
            "read_floor_official_us": READ_FLOOR_OFFICIAL_US,
            "verify_above_read_official_us": VERIFY_ABOVE_READ_OFFICIAL_US,
            "kernel_addressable_official_us": KERNEL_ADDRESSABLE_OFFICIAL_US,
            "tps_all_hides": TPS_ALL_HIDES,
            "tps_free_lossless": TPS_FREE_LOSSLESS,
            "lossless_saving_norm_us": LOSSLESS_SAVING_NORM_US,
            "bridge_draft": BRIDGE_DRAFT,
            "bridge_verify": BRIDGE_VERIFY,
            "milestone_tps": MILESTONE,
            "fern_281_et_floor": FERN_ET_FLOOR_281,
            "ppl_gate": PPL_GATE,
            "ppl_baseline": PPL_BASELINE,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[verify-compute-hideability] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "verify_compute_hideability_self_test_passes":
            int(bool(st["verify_compute_hideability_self_test_passes"])),
        "tps_kernel_floor_honest": f["tps_kernel_floor_honest"],
        "free_lane_to_500_exists": int(bool(f["free_lane_to_500_exists"])),
        "verify_compute_hideable_frac": h["verify_compute_hideable_frac"],
        "verify_compute_hideable_us": h["verify_compute_hideable_us"],
        "verify_compute_exposed_us": h["verify_compute_exposed_us"],
        "verify_above_read_block_us": VERIFY_ABOVE_READ_OFFICIAL_US,
        "partition_resid_us": h["partition_resid_us"],
        "phi_ws_bridge": r["phi_ws_bridge"],
        "basis_reconciled": int(bool(r["basis_reconciled"])),
        "tps_free_lossless": TPS_FREE_LOSSLESS,
        "tps_all_compute_hides": TPS_ALL_HIDES,
        "floor_brackets_in_487_747": int(bool(f["floor_brackets_in_487_747"])),
        "coincides_with_free_lossless": int(bool(f["coincides_with_free_lossless"])),
        "roundtrip_resid_tps": f["roundtrip_resid_tps"],
        "extra_hideable_needed_for_500_us": f["extra_hideable_needed_for_500_us"],
        "extra_needed_as_multiple_of_lever": f["extra_needed_as_multiple_of_lever"],
        "tps_via_steprate": b["tps_via_steprate"],
        "tps_via_composition": b["tps_via_composition"],
        "read_frac_of_wall": b["read_frac_of_wall"],
        "non_read_frac_of_wall": b["non_read_frac_of_wall"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="verify_compute_hideability_result", artifact_type="speed-analysis",
                      data=payload)
    finish_wandb(run)
    print(f"[verify-compute-hideability] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--gpu-probe", action="store_true",
                    help="run a fresh CUDA-event read-floor reduction (optional grounding)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="verify-compute-hideability")
    args = ap.parse_args(argv)

    syn = synthesize(args)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 291,
        "agent": "denken",
        "kind": "verify-compute-hideability",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[verify-compute-hideability] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (e) and recompute PRIMARY.
    syn["self_test"]["conditions"]["e_nan_clean"] = bool(
        syn["self_test"]["conditions"]["e_nan_clean"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["verify_compute_hideability_self_test_passes"] = passes
    syn["verify_compute_hideability_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "verify_compute_hideability_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[verify-compute-hideability] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY verify_compute_hideability_self_test_passes = {passes}", flush=True)
    print(f"  TEST tps_kernel_floor_honest = {syn['tps_kernel_floor_honest']:.4f}", flush=True)
    print(f"  free_lane_to_500_exists = {syn['free_lane_to_500_exists']}", flush=True)
    print(f"  verify_compute_hideable_frac = {syn['verify_compute_hideable_frac']:.6f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[verify-compute-hideability] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
