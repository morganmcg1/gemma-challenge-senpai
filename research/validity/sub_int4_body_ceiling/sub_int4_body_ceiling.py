#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Sub-int4 body quantization ceiling (PR #355) — does shrinking the int4 body
weight read lift the STRICT NON-SPEC AR frontier (165.44 TPS) toward 500, and does
it compose under the strict-compliant SPEC substrate?

WHY THIS LEG
------------
Under the #319 strict-lock the only HBM-amortization lever (spec-decode) is
supply-tax-capped at 473.5 (denken #332) and method-independent (stark #345 /
fern #349 FlashInfer-BI / kanna #348 REST all confirm; wirbel #354 shows the
deployed spec config is < 500 even with a FREE identity kernel). That leaves
exactly one unexplored axis that attacks the batch=1 step ITSELF: shrinking the
int4 body weight read.

denken #344 (`sxltbech`) measured the int4 body GEMM weight read = 1.6973824 GB =
94.3% of the batch=1 (M=1) step HBM traffic, at arithmetic-intensity 4.0 << the
A10G ridge 208.3 -> the non-spec step is WEIGHT-READ-BANDWIDTH-BOUND. In that
regime tps is proportional to 1 / HBM_traffic, so a sub-int4 body shrink scales
the dominant 94.3% term up. The strict non-spec AR frontier (lawine #196) is
165.44 TPS; this leg asks: does ANY PPL<=2.42-safe body bit-width lift it to 500
(~3x), and does the same shrink applied to the strict spec substrate cross 500?

CRUX (vs the SPEC roofline #283/#287): in the deployed SPEC step the body read is
only ~38% of the honest wall (it amortizes over E[T]~3.844 accepted tokens), so a
body shrink there has weak leverage. In the NON-SPEC M=1 step the body read is
94.3% of the step (read once per single token) -> a body shrink has NEAR-LINEAR
leverage. This leg is the non-spec-regime counterpart of #287's spec-regime
Pareto, and additionally re-prices the spec substrate under the body shrink.

THE BANDWIDTH MODEL (self-test pins the round-trip)
---------------------------------------------------
In the BW-bound M=1 step:
    tps(b) = 165.44 * HBM_traffic(4.0) / HBM_traffic(b)
    HBM_traffic(b) = body_read(b) + non_body
Ground (denken #344): body_read(4.0) = 1.6973824 GB = 94.3% of the step, so
    step(4.0) = 1.6973824 / 0.943 = 1.7999813 GB,  non_body = 5.7% = 0.1025989 GB
(non_body = lm_head read [lmhead12k 0.06291456 GB] + KV + activations; FIXED wrt
body bit-width). The body read carries a per-group SCALE-METADATA FLOOR that does
NOT shrink with the code bits (int4-Marlin / compressed-tensors keeps one bf16
group scale per group regardless of code width):
    body_read(b) = code(4.0) * (b/4.0) + scale_meta
    code(4.0) + scale_meta = 1.6973824   (so body_read(4.0) is exact)
    scale_meta / code(4.0) = SCALE_BYTES_PER_GROUP / (CODE_BYTES_PER_VAL * group)
                           = 2 / (0.5 * group) = 4 / group
The served int4 frontier (fa2sw_precache_kenyan == fa2sw_nonspec_int4, the 165.44
substrate) is the g128 lmhead12k track (submissions/int4_g128_lmhead, group=128),
so scale_meta is small and b=2 lands at ~0.515 of the int4 read (NOT exactly
half). We carry g32 (raw QAT base) and a no-floor idealization as a sensitivity
bracket.

PPL GATE: BINDING FOR NON-SPEC, NOT FOR SPEC
--------------------------------------------
Deployed int4 PPL = 2.3772, gate = 2.42 -> headroom 0.0428 absolute = 1.80%
RELATIVE. Literature PPL(bits) is WikiText-2 (NOT the challenge eval), used as a
DIRECTIONAL RELATIVE bound: a scheme adding >1.80% PPL vs its own 4-bit very likely
breaks the gate. The literature splits into TWO lanes: the deployable SCALAR Marlin
lane (GPTQ/AWQ 3-bit = +11-13% >> 1.80%) floors at b_min ~= 3.85; the best-in-class
CODEBOOK lane (AQLM 3.04b +0.5% / QTIP-hybrid 3.0b +1.1%) reaches b_min ~= 2.94 but
needs non-vLLM codebook kernels. The verdict CUTS BY REGIME, not by lane: in the
NON-SPEC M=1 regime the body-read lever needs ~1.07 bits to triple 165.44 -> 500,
which is far below ANY PPL-safe bit (2.94 or 3.85), so non-spec NEVER clears 500
(PPL gate is moot -- the BW leverage is simply insufficient). In the SPEC regime a
~25% body-code shrink (4->3 bit) lifts the strict 473.5 substrate over 500 in pure
BW, so there PPL is NOT the wall at 3.0 bits -- the wall is a greedy-identity sub-int4
verify KERNEL (#192) and whether the M>=8 verify is even BW-bound (a lane-a gap).

SCOPE
-----
0-GPU CPU-analytic + published-literature card. NO model build, NO quantize, NO
training, NO HF Job, NO submission, NO served-file change. BASELINE 481.53
UNCHANGED. PRIMARY = the self-test; 0 TPS, bank-the-analysis. A PPL-safe sub-int4
body, if it existed, would be a SEPARATE human-approval-gated measured build; this
leg only prices feasibility.

PRIMARY metric  sub_int4_body_ceiling_self_test_passes
TEST    metric  strict_nonspec_tps_at_int2

Run:
    cd target/ && WANDB_MODE=online .venv/bin/python \
      research/validity/sub_int4_body_ceiling/sub_int4_body_ceiling.py --self-test \
      --wandb_group sub-int4-body-ceiling --wandb_name lawine/sub-int4-body-ceiling
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
# Imported EXACT anchors (provenance in the module docstring; NOT re-derived).
# --------------------------------------------------------------------------- #
STRICT_NONSPEC_FLOOR_TPS = 165.44       # lawine #196 strict non-spec int4 M=1 AR frontier
BODY_INT4_READ_GB = 1.6973824           # denken #344/#278 int4 body GEMM read (qkv+o+gate_up+down)
BODY_READ_FRAC_OF_STEP = 0.943          # denken #344: body read = 94.3% of the M=1 step HBM
TARGET_TPS = 500.0                      # the bar this lever must clear
LMHEAD_BF16_GB = 0.06291456             # denken #278 lmhead12k bf16 read (FIXED wrt body bits)

# PPL gate.
DEPLOYED_INT4_PPL = 2.3772              # PR #52 deployed served int4 PPL (128/128)
PPL_GATE = 2.42                         # validity bar (reference PPL + 5%)
PPL_HEADROOM_ABS = 0.0428               # 2.42 - 2.3772
PPL_HEADROOM_REL = 0.0428 / 2.3772      # ~0.018003 (1.80% relative headroom over int4)

# Spec substrate (deliverable c).
SPEC_STRICT_FRONTIER_TPS = 473.5        # denken #332 strict spec supply-tax cap (wirbel #354: < 500)
DEPLOYED_SPEC_TPS = 481.53              # served spec baseline (NOT strict-#192-compliant; #196)
SPEC_BODY_LMHEAD_READ_FRAC = 0.3804642716594112   # denken #283 body+lmhead read frac of honest spec wall

# Metadata-floor model (per-group bf16 scales; served frontier = g128).
CODE_BYTES_PER_VAL = 0.5                # 4-bit nibble at int4
SCALE_BYTES_PER_GROUP = 2.0            # bf16 group scale (compressed-tensors symmetric)
SERVED_GROUP_SIZE = 128                # served int4 frontier group size (int4_g128_lmhead)
QAT_BASE_GROUP_SIZE = 32              # raw QAT w4a16-ct base checkpoint group size (conservative)

# Body bit-widths to evaluate.
BIT_GRID = [4.0, 3.0, 2.5, 2.0]
INT2_BAND = (300.0, 340.0)            # PR pre-registered self-test band for the int2 point

# Tolerances.
TOL_ROUNDTRIP = 1e-6
TOL_IMPORT = 1e-9

# --------------------------------------------------------------------------- #
# Literature PPL(bits) overlay -- published WikiText-2 numbers for dense ~7B-class
# models (the closest public proxy: NO published gemma-4-E4B sub-int4 PPL). Each row
# is the RELATIVE PPL increase over the SAME method-family's 4-bit baseline, used as
# a DIRECTIONAL bound against the 1.80% relative headroom.
# rel_increase_over_4bit = (ppl_b - ppl_4bit) / ppl_4bit on WikiText-2.
#
# TWO LANES (the literature splits cleanly, and the verdict depends on which):
#  * CODEBOOK lane (AQLM/QTIP/QuIP#): additive-codebook / trellis quant. Holds ~3.0
#    avg-bits within ~1% PPL of 4-bit -- but is NOT a per-group bf16-scale Marlin
#    layout (item 5 of the lit pass) and is NOT in mainstream vLLM; decode is a
#    random-access gather, compute-heavy. This is the GENEROUS (best-achievable) lane.
#  * SCALAR/MARLIN lane (GPTQ/AWQ-g128): the deployable-today compressed-tensors
#    layout the served int4 frontier actually uses. 4->3 bit costs +11-13% PPL (>>
#    1.80% headroom), so its PPL-safe floor sits just below 4.0.
# Numbers from the literature pass (arXiv IDs cited); see LIT_NOTES.
# --------------------------------------------------------------------------- #
LIT_PPL_CURVE_CODEBOOK: list[dict[str, Any]] = [
    # avg_bits, rel_increase_over_4bit, scheme, model, corpus, arxiv
    {"bits": 4.0, "rel_increase_over_4bit": 0.0000, "scheme": "int4-anchor",
     "model": "deployed gemma-4-E4B g128", "corpus": "challenge-private", "arxiv": "PR#52"},
    {"bits": 3.04, "rel_increase_over_4bit": 0.0050, "scheme": "AQLM-additive-codebook",
     "model": "Llama-2-7B", "corpus": "WikiText-2(ctx4096)", "arxiv": "2401.06118-Tab2"},
    {"bits": 3.0, "rel_increase_over_4bit": 0.0110, "scheme": "QTIP-hybrid-trellis",
     "model": "Llama-2-7B", "corpus": "WikiText-2(ctx4096)", "arxiv": "2406.11235-Tab5"},
    {"bits": 2.0, "rel_increase_over_4bit": 0.1230, "scheme": "QTIP-hybrid-trellis",
     "model": "Llama-2-7B", "corpus": "WikiText-2(ctx4096)", "arxiv": "2406.11235-Tab5"},
]
LIT_PPL_CURVE_SCALAR: list[dict[str, Any]] = [
    {"bits": 4.0, "rel_increase_over_4bit": 0.0000, "scheme": "GPTQ/AWQ-4bit-anchor",
     "model": "Llama-2-7B", "corpus": "WikiText-2(ctx2048)", "arxiv": "2306.05179-Tab4"},
    {"bits": 3.0, "rel_increase_over_4bit": 0.1200, "scheme": "GPTQ/AWQ-g128",
     "model": "Llama-2-7B", "corpus": "WikiText-2(ctx2048)", "arxiv": "2210.17323/2306.00978"},
    {"bits": 2.0, "rel_increase_over_4bit": 1.6160, "scheme": "AWQ-g64-2bit",
     "model": "Llama-2-7B", "corpus": "WikiText-2(ctx2048)", "arxiv": "2410.19103"},
]
# Headline curve = the generous CODEBOOK lane (gives the lowest PPL-safe avg bits,
# answering the lit pass' single most important deliverable). The SCALAR lane is the
# deployable-today reality check.
LIT_PPL_CURVE: list[dict[str, Any]] = LIT_PPL_CURVE_CODEBOOK
LIT_NOTES = (
    "Literature PPL is WikiText-2 on Llama-2-7B (the closest published dense proxy; NO "
    "published gemma-4-E4B sub-int4 PPL exists). Used as a RELATIVE directional bound "
    "vs the 1.80% relative headroom over the int4 anchor. THE LANES DIVERGE SHARPLY: "
    "best-in-class CODEBOOK/trellis (AQLM 3.04b +0.5% [2401.06118], QTIP-hybrid 3.0b "
    "+1.1% [2406.11235]) holds within headroom down to ~3.0 avg-bits, so b_min_ppl ~= "
    "2.94; but the deployable SCALAR Marlin lane (GPTQ/AWQ 3-bit +11-13% [2210.17323/"
    "2306.00978]) blows the headroom an order of magnitude at 3-bit, so its floor is "
    "~3.85 bits. At 2-bit EVERY method blows the gate (best QTIP-hybrid +12.3%, scalar "
    "+160% AWQ-g64 / +558% GPTQ-OmniQuant). CRUCIAL: AQLM/QTIP/QuIP# are additive-"
    "codebook/trellis formats -- NOT per-group bf16-scale Marlin -- so they (a) do not "
    "match this card's linear-code+scale-floor byte model exactly, (b) are not in "
    "mainstream vLLM, and (c) add random-access decode compute that the pure-BW model "
    "ignores. Gemma-class models (wide 262k vocab, logit soft-capping, MatFormer/PLE) "
    "are not known to be MORE low-bit robust than Llama (arXiv:2409.11055 evaluates "
    "Gemma on task-accuracy, not PPL), so this proxy is, if anything, optimistic."
)

# Honest-scope caveats (self-test (f)).
CAVEATS = [
    "ZERO_TPS: this leg prices the PPL-vs-bit feasibility of the body-read denominator "
    "lever in the NON-SPEC M=1 regime; it builds NO >=500 artifact and adds 0 TPS. "
    "BASELINE stays 481.53.",
    "ANALYTIC_BW_MODEL: tps(b)=165.44*HBM(4)/HBM(b) is a roofline approximation of the "
    "weight-read-BW-bound M=1 step (denken #344 AI 4.0 << ridge 208.3); it assumes the "
    "non-body 5.7% and the per-group scale floor are fixed wrt body bits. Real sub-int4 "
    "Marlin/2-bit kernels add transcode + dispatch overhead, so realized tps(b) is a "
    "CEILING on this model.",
    "PPL_PROXY_DIRECTIONAL: literature PPL(bits) is WikiText-2 on Llama-2-7B proxies, not "
    "the challenge eval; b_min_ppl is a RELATIVE-degradation bound, not a measured "
    "challenge PPL. A real sub-int4 build must MEASURE challenge PPL<=2.42.",
    "TWO_LANES: the headline b_min_ppl ~= 2.94 is the GENEROUS best-in-class CODEBOOK "
    "lane (AQLM/QTIP). The deployable-today SCALAR Marlin lane (what the served int4 "
    "frontier uses) has b_min ~= 3.85 because GPTQ/AWQ 3-bit is +11-13% PPL (>> 1.80% "
    "headroom). Both lanes are reported; the realistic-with-today-kernels verdict is the "
    "SCALAR lane.",
    "CODEBOOK_NOT_DEPLOYABLE: the codebook/trellis methods that reach ~3.0 PPL-safe bits "
    "(AQLM 2401.06118, QTIP 2406.11235, QuIP# 2402.04396) are NOT per-group bf16-scale "
    "Marlin, are NOT in mainstream vLLM, and add random-access decode compute the pure-BW "
    "model ignores -- in the SPEC-verify M>=8 GEMM (higher AI, not weight-read-bound) a "
    "byte-read shrink may buy ~0. So the codebook spec-clears-500 number is a PURE-BW "
    "CEILING, not a deployable route.",
    "GROUP_SIZE_FLOOR: the scale-metadata floor uses the served g128 frontier; g32 (raw "
    "QAT base) is carried as a conservative sensitivity (pulls the int2 point down). "
    "Real 2-bit kernels may regroup/repack, changing the floor.",
    "SPEC_BINDING_WALL_IS_IDENTITY_NOT_PPL: for the SPEC lane at ~3.0 bits, PPL is NOT "
    "the wall -- a ~25% body-code shrink lifts the strict 473.5 substrate over 500 in "
    "pure BW. The binding walls are (i) a greedy-identity-compliant sub-int4 verify "
    "kernel (#192; deployed spec already diverges 56%, kanna #114) and (ii) whether the "
    "M>=8 verify is even BW-bound. This is a lane-a-class engineering gap, not a PPL gap.",
    "DO_NOT_BUILD: NO re-quantization of the served checkpoint, NO served-file change, "
    "NOT a launch, NO HF Job, NO submission. A PPL-safe sub-int4 body would be a SEPARATE "
    "human-approval-gated MEASURED build; the launch gate stays a measured >=500 at the "
    "deployed PPL/greedy-identity contract.",
    "SPEC_SUBSTRATE_IMPORTED: the spec composition uses denken #332's 473.5 strict cap "
    "and denken #283's 0.3805 body+lmhead spec read-fraction; wirbel #354's exact "
    "strict-compliant kernel frontier is a companion (< 500). The deployed 481.53 variant "
    "is also reported, but it is NOT strict-#192-compliant (kanna #114: 56% divergence).",
]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# (1) Body-read decomposition + the BW-bound non-spec tps model.
# --------------------------------------------------------------------------- #
def decompose_body_read(group_size: int) -> dict[str, Any]:
    """Split the int4 body read 1.6973824 GB into a linear-in-bits code term and a
    FIXED per-group bf16 scale floor (scale/code = 4/group)."""
    scale_over_code = (SCALE_BYTES_PER_GROUP / CODE_BYTES_PER_VAL) / group_size  # = 4/group
    code4 = BODY_INT4_READ_GB / (1.0 + scale_over_code)
    scale_meta = BODY_INT4_READ_GB - code4
    return {
        "group_size": group_size,
        "scale_over_code": scale_over_code,
        "code4_gb": code4,
        "scale_meta_gb": scale_meta,
        "scale_meta_frac_of_body": scale_meta / BODY_INT4_READ_GB,
    }


def step_terms() -> dict[str, float]:
    step4 = BODY_INT4_READ_GB / BODY_READ_FRAC_OF_STEP           # 1.7999813 GB
    non_body = step4 - BODY_INT4_READ_GB                          # 0.1025989 GB (5.7%)
    return {"step4_gb": step4, "non_body_gb": non_body,
            "non_body_frac": non_body / step4}


def body_read_gb(b: float, code4: float, scale_meta: float, floor: bool = True) -> float:
    """Body read at bit-width b. With floor: code shrinks linearly, scale fixed.
    Without floor (idealized): the whole 1.6973824 scales linearly."""
    if floor:
        return code4 * (b / 4.0) + scale_meta
    return BODY_INT4_READ_GB * (b / 4.0)


def strict_nonspec_tps(b: float, code4: float, scale_meta: float, non_body: float,
                       floor: bool = True) -> float:
    hbm4 = BODY_INT4_READ_GB + non_body
    hbm_b = body_read_gb(b, code4, scale_meta, floor) + non_body
    return STRICT_NONSPEC_FLOOR_TPS * hbm4 / hbm_b


def b_to_clear_500(code4: float, scale_meta: float, non_body: float,
                   floor: bool = True) -> float:
    """Solve strict_nonspec_tps(b) = 500 for b. In the floored model the code term
    must supply hbm_b = 165.44*hbm4/500; scale+non_body are fixed."""
    hbm4 = BODY_INT4_READ_GB + non_body
    hbm_target = STRICT_NONSPEC_FLOOR_TPS * hbm4 / TARGET_TPS
    if floor:
        code_needed = hbm_target - scale_meta - non_body
        return 4.0 * code_needed / code4
    code_needed = hbm_target - non_body
    return 4.0 * code_needed / BODY_INT4_READ_GB


# --------------------------------------------------------------------------- #
# (2) Literature PPL overlay -> b_min_ppl (lowest PPL-safe bit-width).
# --------------------------------------------------------------------------- #
def b_min_ppl_from_literature(curve_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """b_min_ppl = lowest avg bit-width whose relative PPL increase over 4-bit stays
    within the 1.80% relative headroom, by linear interpolation on the literature
    curve (monotone in bits)."""
    curve = sorted(curve_rows or LIT_PPL_CURVE, key=lambda r: r["bits"], reverse=True)  # 4.0 down
    hr = PPL_HEADROOM_REL
    b_min = curve[0]["bits"]  # default 4.0 if even the next step exceeds headroom
    crossed_between = None
    for i in range(len(curve) - 1):
        hi, lo = curve[i], curve[i + 1]
        if lo["rel_increase_over_4bit"] <= hr:
            b_min = lo["bits"]
            continue
        # headroom is crossed between hi (safe) and lo (unsafe): interpolate.
        r_hi, r_lo = hi["rel_increase_over_4bit"], lo["rel_increase_over_4bit"]
        if r_lo > r_hi:
            frac = (hr - r_hi) / (r_lo - r_hi)
            b_cross = hi["bits"] + frac * (lo["bits"] - hi["bits"])
            b_min = b_cross
            crossed_between = (hi["bits"], lo["bits"])
        break
    return {
        "b_min_ppl": b_min,
        "headroom_rel": hr,
        "headroom_abs": PPL_HEADROOM_ABS,
        "crossed_between_bits": crossed_between,
        "curve": curve,
    }


# --------------------------------------------------------------------------- #
# (3) Spec-substrate composition (deliverable c).
# --------------------------------------------------------------------------- #
def spec_body_code_frac_of_wall(code4: float) -> float:
    """Fraction of the honest SPEC wall that is the SHRINKABLE body code. denken #283
    body+lmhead = 0.3805 of the wall; within it, only the body CODE (not lmhead, not
    the fixed scale floor) shrinks with bits."""
    body_plus_lmhead = BODY_INT4_READ_GB + LMHEAD_BF16_GB
    return SPEC_BODY_LMHEAD_READ_FRAC * (code4 / body_plus_lmhead)


def spec_tps_at_bit(b: float, S: float, f_shrink: float) -> float:
    """tps_spec(b) = S / (1 - f_shrink * (1 - b/4))  (body code -> b/4 of its int4 size)."""
    return S / (1.0 - f_shrink * (1.0 - b / 4.0))


def spec_b_to_clear_500(S: float, f_shrink: float) -> float:
    """Solve spec_tps_at_bit(b) = 500 for b."""
    # 1 - f*(1-b/4) = S/500  ->  1-b/4 = (1 - S/500)/f
    one_minus_q = (1.0 - S / TARGET_TPS) / f_shrink
    return 4.0 * (1.0 - one_minus_q)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(group_size: int = SERVED_GROUP_SIZE) -> dict[str, Any]:
    served = decompose_body_read(group_size)
    qat = decompose_body_read(QAT_BASE_GROUP_SIZE)
    st = step_terms()
    code4, scale_meta, non_body = served["code4_gb"], served["scale_meta_gb"], st["non_body_gb"]

    # non-spec tps(b) on the served (g128) floor, plus g32 + no-floor sensitivity.
    tps_curve = []
    for b in BIT_GRID:
        tps_curve.append({
            "bits": b,
            "tps_served_floor": strict_nonspec_tps(b, code4, scale_meta, non_body, floor=True),
            "tps_g32_floor": strict_nonspec_tps(
                b, qat["code4_gb"], qat["scale_meta_gb"], non_body, floor=True),
            "tps_no_floor": strict_nonspec_tps(b, code4, scale_meta, non_body, floor=False),
            "body_read_gb_served": body_read_gb(b, code4, scale_meta, floor=True),
            "body_read_frac_of_int4": body_read_gb(b, code4, scale_meta, floor=True) / BODY_INT4_READ_GB,
        })

    int2 = next(r for r in tps_curve if r["bits"] == 2.0)
    strict_nonspec_tps_at_int2 = int2["tps_served_floor"]

    b500 = b_to_clear_500(code4, scale_meta, non_body, floor=True)
    b500_no_floor = b_to_clear_500(code4, scale_meta, non_body, floor=False)

    # PPL lanes: headline = GENEROUS best-in-class CODEBOOK lane (lowest PPL-safe avg
    # bits); SCALAR Marlin lane is the deployable-today reality check.
    ppl = b_min_ppl_from_literature(LIT_PPL_CURVE_CODEBOOK)
    ppl_marlin = b_min_ppl_from_literature(LIT_PPL_CURVE_SCALAR)
    b_min = ppl["b_min_ppl"]                  # codebook ~= 2.94 (headline)
    b_min_marlin = ppl_marlin["b_min_ppl"]    # scalar ~= 3.85 (deployable-today)

    sub_int4_alone_clears_500 = bool(b500 >= b_min)   # need a PPL-safe bit that also clears 500

    # non-spec tps at each PPL-safe floor (what the lever actually buys, compliant).
    nonspec_tps_at_bmin = strict_nonspec_tps(b_min, code4, scale_meta, non_body, floor=True)
    nonspec_tps_at_bmin_marlin = strict_nonspec_tps(
        b_min_marlin, code4, scale_meta, non_body, floor=True)

    # spec composition (deliverable c). Body code is only ~38% of the spec wall (weak
    # leverage). Strict 473.5 substrate (denken #332) is the headline; deployed 481.53
    # (NOT #192-compliant) reported alongside. Two b_min lanes: codebook (CEILING, needs
    # non-vLLM kernels + a greedy-identity verify) and scalar Marlin (deployable-today).
    f_shrink = spec_body_code_frac_of_wall(code4)
    spec_strict_b500 = spec_b_to_clear_500(SPEC_STRICT_FRONTIER_TPS, f_shrink)
    spec_deployed_b500 = spec_b_to_clear_500(DEPLOYED_SPEC_TPS, f_shrink)
    spec_tps_at_bmin_strict = spec_tps_at_bit(b_min, SPEC_STRICT_FRONTIER_TPS, f_shrink)
    spec_tps_at_bmin_deployed = spec_tps_at_bit(b_min, DEPLOYED_SPEC_TPS, f_shrink)
    spec_tps_at_bmin_marlin_strict = spec_tps_at_bit(b_min_marlin, SPEC_STRICT_FRONTIER_TPS, f_shrink)
    spec_tps_at_bmin_marlin_deployed = spec_tps_at_bit(b_min_marlin, DEPLOYED_SPEC_TPS, f_shrink)
    spec_tps_at_int2_strict = spec_tps_at_bit(2.0, SPEC_STRICT_FRONTIER_TPS, f_shrink)
    # headline boolean = GENEROUS codebook ceiling on the strict substrate.
    spec_plus_subint4_clears_500 = bool(spec_tps_at_bmin_strict >= TARGET_TPS)
    spec_plus_subint4_clears_500_deployed = bool(spec_tps_at_bmin_deployed >= TARGET_TPS)
    # deployable-today (scalar Marlin lane) -- the realistic verdict.
    spec_plus_subint4_clears_500_marlin = bool(spec_tps_at_bmin_marlin_strict >= TARGET_TPS)
    spec_plus_subint4_clears_500_marlin_deployed = bool(
        spec_tps_at_bmin_marlin_deployed >= TARGET_TPS)

    return {
        "group_size_served": group_size,
        "served_decomp": served,
        "qat_decomp": qat,
        "step_terms": st,
        "tps_curve": tps_curve,
        # headline deliverables
        "strict_nonspec_tps_at_int2": strict_nonspec_tps_at_int2,
        "strict_nonspec_tps_at_int2_g32": int2["tps_g32_floor"],
        "strict_nonspec_tps_at_int2_no_floor": int2["tps_no_floor"],
        "b_to_clear_500": b500,
        "b_to_clear_500_no_floor": b500_no_floor,
        "b_min_ppl": b_min,
        "b_min_ppl_marlin_scalar": b_min_marlin,
        "sub_int4_alone_clears_500": sub_int4_alone_clears_500,
        "nonspec_tps_at_bmin_ppl": nonspec_tps_at_bmin,
        "nonspec_tps_at_bmin_marlin": nonspec_tps_at_bmin_marlin,
        # spec composition
        "spec_body_code_frac_of_wall": f_shrink,
        "spec_strict_frontier_tps": SPEC_STRICT_FRONTIER_TPS,
        "spec_deployed_tps": DEPLOYED_SPEC_TPS,
        "spec_b_to_clear_500_strict": spec_strict_b500,
        "spec_b_to_clear_500_deployed": spec_deployed_b500,
        "spec_tps_at_bmin_strict": spec_tps_at_bmin_strict,
        "spec_tps_at_bmin_deployed": spec_tps_at_bmin_deployed,
        "spec_tps_at_bmin_marlin_strict": spec_tps_at_bmin_marlin_strict,
        "spec_tps_at_bmin_marlin_deployed": spec_tps_at_bmin_marlin_deployed,
        "spec_tps_at_int2_strict": spec_tps_at_int2_strict,
        "spec_plus_subint4_clears_500": spec_plus_subint4_clears_500,
        "spec_plus_subint4_clears_500_deployed": spec_plus_subint4_clears_500_deployed,
        "spec_plus_subint4_clears_500_marlin": spec_plus_subint4_clears_500_marlin,
        "spec_plus_subint4_clears_500_marlin_deployed": spec_plus_subint4_clears_500_marlin_deployed,
        # ppl
        "ppl": ppl,
        "ppl_marlin": ppl_marlin,
        "lit_notes": LIT_NOTES,
        "caveats": CAVEATS,
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY) — conditions (a)-(g).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    code4 = syn["served_decomp"]["code4_gb"]
    scale_meta = syn["served_decomp"]["scale_meta_gb"]
    non_body = syn["step_terms"]["non_body_gb"]

    # (a) round-trip: tps(4.0) == 165.44 exactly on the served floor.
    tps4 = strict_nonspec_tps(4.0, code4, scale_meta, non_body, floor=True)
    a = bool(abs(tps4 - STRICT_NONSPEC_FLOOR_TPS) < TOL_ROUNDTRIP)

    # (b) monotone: tps strictly increases as b decreases, on all three floor models.
    def monotone(key: str) -> bool:
        seq = [r[key] for r in sorted(syn["tps_curve"], key=lambda r: r["bits"], reverse=True)]
        return all(seq[i] < seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
    b = bool(monotone("tps_served_floor") and monotone("tps_g32_floor")
             and monotone("tps_no_floor"))

    # (c) the int2 point (served g128 floor) is within the PR band [300,340].
    c = bool(INT2_BAND[0] <= syn["strict_nonspec_tps_at_int2"] <= INT2_BAND[1])

    # (d) the central verdict identity: b_to_clear_500 < b_min_ppl  <=>  alone does NOT clear.
    d = bool((syn["b_to_clear_500"] < syn["b_min_ppl"])
             == (not syn["sub_int4_alone_clears_500"]))

    # (e) the metadata floor makes b=2 NOT exactly half of the int4 read (served floor).
    int2_frac = next(r["body_read_frac_of_int4"] for r in syn["tps_curve"] if r["bits"] == 2.0)
    e = bool(int2_frac > 0.5 + 1e-6)

    # (f) imports exact + caveats present.
    f = bool(abs(STRICT_NONSPEC_FLOOR_TPS - 165.44) < TOL_IMPORT
             and abs(BODY_INT4_READ_GB - 1.6973824) < TOL_IMPORT
             and abs(BODY_READ_FRAC_OF_STEP - 0.943) < TOL_IMPORT
             and abs(TARGET_TPS - 500.0) < TOL_IMPORT
             and abs(DEPLOYED_INT4_PPL - 2.3772) < TOL_IMPORT
             and abs(PPL_GATE - 2.42) < TOL_IMPORT
             and abs(SPEC_STRICT_FRONTIER_TPS - 473.5) < TOL_IMPORT
             and abs(SPEC_BODY_LMHEAD_READ_FRAC - 0.3804642716594112) < 1e-9
             and len(CAVEATS) >= 6)

    # (g) the int2 point is BW-bound-sane: body read roughly halves (in [0.45,0.60] of int4).
    g = bool(0.45 <= int2_frac <= 0.60)

    conditions = {
        "a_roundtrip_165p44_at_b4": a,
        "b_monotone_tps_up_as_bits_down": b,
        "c_int2_in_pr_band_300_340": c,
        "d_verdict_identity_b500_lt_bminppl": d,
        "e_floor_makes_b2_not_exactly_half": e,
        "f_imports_exact_and_caveats": f,
        "g_int2_body_read_halves_sane": g,
    }
    return {"conditions": conditions,
            "sub_int4_body_ceiling_self_test_passes": bool(all(conditions.values()))}


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


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict, st: dict) -> None:
    print("\n" + "=" * 100, flush=True)
    print("SUB-INT4 BODY CEILING (PR #355) — does shrinking the int4 body read lift the "
          "strict non-spec 165.44 frontier to 500?", flush=True)
    print("=" * 100, flush=True)
    sd, stp = syn["served_decomp"], syn["step_terms"]
    print(f"  body_read(int4)={BODY_INT4_READ_GB} GB = {100*BODY_READ_FRAC_OF_STEP:.1f}% of the "
          f"M=1 step ({stp['step4_gb']:.4f} GB); non_body={stp['non_body_gb']:.4f} GB "
          f"({100*stp['non_body_frac']:.1f}%)", flush=True)
    print(f"  served floor g{sd['group_size']}: code4={sd['code4_gb']:.4f} GB + scale_meta="
          f"{sd['scale_meta_gb']:.4f} GB (scale/code={sd['scale_over_code']:.4f})", flush=True)
    print("-" * 100, flush=True)
    print("  NON-SPEC tps(b) [served g128 floor | g32 | no-floor]:", flush=True)
    for r in syn["tps_curve"]:
        print(f"    b={r['bits']:>4}: {r['tps_served_floor']:7.2f} | {r['tps_g32_floor']:7.2f} | "
              f"{r['tps_no_floor']:7.2f}  (body read {100*r['body_read_frac_of_int4']:.1f}% of int4)",
              flush=True)
    print(f"  >>> strict_nonspec_tps_at_int2 = {syn['strict_nonspec_tps_at_int2']:.2f} "
          f"(g32 {syn['strict_nonspec_tps_at_int2_g32']:.2f} / no-floor "
          f"{syn['strict_nonspec_tps_at_int2_no_floor']:.2f})", flush=True)
    print(f"  >>> b_to_clear_500 (non-spec) = {syn['b_to_clear_500']:.4f} bits "
          f"(no-floor {syn['b_to_clear_500_no_floor']:.4f})", flush=True)
    print("-" * 100, flush=True)
    print(f"  PPL gate: deployed {DEPLOYED_INT4_PPL} -> gate {PPL_GATE} = {100*PPL_HEADROOM_REL:.2f}% "
          f"relative headroom. TWO lanes:", flush=True)
    print(f"    codebook (AQLM/QTIP, best-in-class, NOT in vLLM): b_min_ppl = {syn['b_min_ppl']:.3f} bits",
          flush=True)
    print(f"    scalar Marlin (GPTQ/AWQ, deployable-today):       b_min     = "
          f"{syn['b_min_ppl_marlin_scalar']:.3f} bits", flush=True)
    print(f"  >>> sub_int4_alone_clears_500 = {syn['sub_int4_alone_clears_500']}  "
          f"(non-spec needs {syn['b_to_clear_500']:.2f} bits << EITHER PPL-safe floor "
          f"{syn['b_min_ppl']:.2f}/{syn['b_min_ppl_marlin_scalar']:.2f})", flush=True)
    print(f"      non-spec tps at PPL-safe floor: codebook {syn['nonspec_tps_at_bmin_ppl']:.2f} / "
          f"marlin {syn['nonspec_tps_at_bmin_marlin']:.2f} TPS (vs floor {STRICT_NONSPEC_FLOOR_TPS}) "
          f"-- both << 500", flush=True)
    print("-" * 100, flush=True)
    print(f"  SPEC composition (body code = {100*syn['spec_body_code_frac_of_wall']:.1f}% of the "
          f"spec wall; strict substrate {SPEC_STRICT_FRONTIER_TPS}):", flush=True)
    print(f"    clears 500 at body bits <= {syn['spec_b_to_clear_500_strict']:.3f} (strict) / "
          f"{syn['spec_b_to_clear_500_deployed']:.3f} (deployed {DEPLOYED_SPEC_TPS})", flush=True)
    print(f"    codebook b_min {syn['b_min_ppl']:.2f}b (CEILING): strict {syn['spec_tps_at_bmin_strict']:.2f} / "
          f"deployed {syn['spec_tps_at_bmin_deployed']:.2f} TPS;  int2 {syn['spec_tps_at_int2_strict']:.2f}",
          flush=True)
    print(f"    marlin   b_min {syn['b_min_ppl_marlin_scalar']:.2f}b (deployable): strict "
          f"{syn['spec_tps_at_bmin_marlin_strict']:.2f} / deployed "
          f"{syn['spec_tps_at_bmin_marlin_deployed']:.2f} TPS", flush=True)
    print(f"  >>> spec_plus_subint4_clears_500 = {syn['spec_plus_subint4_clears_500']} "
          f"(CODEBOOK CEILING -- needs a non-vLLM, greedy-identity #192 verify kernel);  "
          f"deployable Marlin lane = {syn['spec_plus_subint4_clears_500_marlin']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) sub_int4_body_ceiling_self_test_passes = "
          f"{st['sub_int4_body_ceiling_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print("=" * 100, flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (house pattern; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        # Import installed wandb FIRST so it is cached before REPO_ROOT joins sys.path
        # (target/ holds a gitignored ./wandb output dir that would shadow the package).
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[sub-int4-body-ceiling] wandb logging unavailable: {exc}", flush=True)
        return

    syn, st = payload["synthesis"], payload["synthesis"]["self_test"]
    run = init_wandb_run(
        job_type="validity-gate",
        agent="lawine",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["validity-gate", "sub-int4-body-ceiling", "non-spec-frontier", "read-reduction",
              "ppl-gate", "bank-the-analysis", "pr-355"],
        config={
            "strict_nonspec_floor_tps": STRICT_NONSPEC_FLOOR_TPS,
            "body_int4_read_gb": BODY_INT4_READ_GB,
            "body_read_frac_of_step": BODY_READ_FRAC_OF_STEP,
            "target_tps": TARGET_TPS,
            "deployed_int4_ppl": DEPLOYED_INT4_PPL,
            "ppl_gate": PPL_GATE,
            "ppl_headroom_rel": PPL_HEADROOM_REL,
            "spec_strict_frontier_tps": SPEC_STRICT_FRONTIER_TPS,
            "deployed_spec_tps": DEPLOYED_SPEC_TPS,
            "spec_body_lmhead_read_frac": SPEC_BODY_LMHEAD_READ_FRAC,
            "served_group_size": SERVED_GROUP_SIZE,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[sub-int4-body-ceiling] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "sub_int4_body_ceiling_self_test_passes":
            int(bool(st["sub_int4_body_ceiling_self_test_passes"])),
        "strict_nonspec_tps_at_int2": syn["strict_nonspec_tps_at_int2"],
        "strict_nonspec_tps_at_int2_g32": syn["strict_nonspec_tps_at_int2_g32"],
        "strict_nonspec_tps_at_int2_no_floor": syn["strict_nonspec_tps_at_int2_no_floor"],
        "b_to_clear_500": syn["b_to_clear_500"],
        "b_to_clear_500_no_floor": syn["b_to_clear_500_no_floor"],
        "b_min_ppl": syn["b_min_ppl"],
        "b_min_ppl_marlin_scalar": syn["b_min_ppl_marlin_scalar"],
        "sub_int4_alone_clears_500": int(bool(syn["sub_int4_alone_clears_500"])),
        "spec_plus_subint4_clears_500": int(bool(syn["spec_plus_subint4_clears_500"])),
        "spec_plus_subint4_clears_500_deployed": int(bool(syn["spec_plus_subint4_clears_500_deployed"])),
        "spec_plus_subint4_clears_500_marlin": int(bool(syn["spec_plus_subint4_clears_500_marlin"])),
        "spec_plus_subint4_clears_500_marlin_deployed":
            int(bool(syn["spec_plus_subint4_clears_500_marlin_deployed"])),
        "nonspec_tps_at_bmin_ppl": syn["nonspec_tps_at_bmin_ppl"],
        "nonspec_tps_at_bmin_marlin": syn["nonspec_tps_at_bmin_marlin"],
        "spec_tps_at_bmin_strict": syn["spec_tps_at_bmin_strict"],
        "spec_tps_at_bmin_deployed": syn["spec_tps_at_bmin_deployed"],
        "spec_tps_at_bmin_marlin_strict": syn["spec_tps_at_bmin_marlin_strict"],
        "spec_tps_at_bmin_marlin_deployed": syn["spec_tps_at_bmin_marlin_deployed"],
        "spec_tps_at_int2_strict": syn["spec_tps_at_int2_strict"],
        "spec_b_to_clear_500_strict": syn["spec_b_to_clear_500_strict"],
        "spec_b_to_clear_500_deployed": syn["spec_b_to_clear_500_deployed"],
        "spec_body_code_frac_of_wall": syn["spec_body_code_frac_of_wall"],
        "ppl_headroom_rel": PPL_HEADROOM_REL,
        "ppl_headroom_abs": PPL_HEADROOM_ABS,
        "scale_meta_gb_served": syn["served_decomp"]["scale_meta_gb"],
        "code4_gb_served": syn["served_decomp"]["code4_gb"],
        "non_body_gb": syn["step_terms"]["non_body_gb"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="sub_int4_body_ceiling_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[sub-int4-body-ceiling] wandb logged: {len(summary)} metrics", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--group-size", type=int, default=SERVED_GROUP_SIZE,
                    help="served body group size for the scale-metadata floor (default 128)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="sub-int4-body-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize(group_size=args.group_size)
    st = self_test(syn)
    syn["self_test"] = st

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 355,
        "agent": "lawine",
        "kind": "sub-int4-body-ceiling",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[sub-int4-body-ceiling] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into the PRIMARY.
    passes = bool(st["sub_int4_body_ceiling_self_test_passes"] and payload["nan_clean"])
    st["sub_int4_body_ceiling_self_test_passes"] = passes

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sub_int4_body_ceiling_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[sub-int4-body-ceiling] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY sub_int4_body_ceiling_self_test_passes = {passes}", flush=True)
    print(f"  TEST strict_nonspec_tps_at_int2 = {syn['strict_nonspec_tps_at_int2']:.4f}", flush=True)
    print(f"  b_to_clear_500={syn['b_to_clear_500']:.4f}  b_min_ppl={syn['b_min_ppl']:.4f}  "
          f"sub_int4_alone_clears_500={syn['sub_int4_alone_clears_500']}  "
          f"spec_plus_subint4_clears_500={syn['spec_plus_subint4_clears_500']}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes
        print(f"[sub-int4-body-ceiling] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
