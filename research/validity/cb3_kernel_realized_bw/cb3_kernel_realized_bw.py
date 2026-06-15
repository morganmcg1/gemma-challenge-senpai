#!/usr/bin/env python3
"""PR #391 (lawine) -- cb3 kernel REALIZED bandwidth at the SERVED verify width (M=8/M=4),
extending #388's M=1-only card.

THE QUESTION (#388 follow-up #2, verbatim):
  #388 measured the cb3-vs-int4-Marlin body-GEMM speedup at M=1 only = 1.1234x realistic
  (beta=0.51), far below the 1.2744x byte roofline, BECAUSE int4-Marlin at M=1 runs at only
  25.6% HBM efficiency (m1_is_bw_bound=False): one activation row -> launch/dequant/occupancy
  overhead dominates, so the weight-byte shrink buys ~half its nominal value, and the strict
  measured-floor tier (1.0582x, +15.6 TPS) only STRADDLES #383's +17.22 floor.
  BUT the deployed stack serves MTP K=7 spec-decode (num_speculative_tokens=7): each step runs
  the body GEMM at M=8 verify width (7 draft + 1), not M=1. At M=8 the GEMM reuses the same
  weight matrix across 8 activation rows -> more compute per loaded weight tile -> better latency
  hiding -> the kernel approaches BANDWIDTH-BOUND -> the weight-byte shrink captures more of its
  roofline. So the SERVED-regime realized speedup is plausibly higher than the M=1 1.12x straddle,
  and M=8 (not M=1) is the operating point that sets the real supply lift. M=4 is the partial-
  accept width (some draft tokens rejected).

WHAT THIS IS / IS NOT:
  * GPU RESEARCH MICROBENCH -- profiling only. NOT a served-kernel patch, NO deployed-file
    change, NO competition submission, 0 official TPS, NO Hugging Face job. Keeps GPU usage
    on the single assigned A10G (CUDA_VISIBLE_DEVICES=0; the #358/#363 2nd-GPU gotcha).
  * No cb3/QTIP/QuIP#/AQLM kernel exists in this env (vLLM 0.22.0 ships only Marlin/AWQ/AQLM;
    QTIP/QuIP# are source-build-only, github.com/Cornell-RelaxML/qtip, no pip/vLLM). So the
    honest terminal result is a ROOFLINE bound: measure int4-Marlin's realized us/GEMM +
    effective GB/s at each width, then bound cb3 by its nominal byte count AT THE SAME achieved
    BW-efficiency. realized_is_roofline_bound = True. (#388 step 4: an acceptable terminal.)

KEY IDENTITY (the engine of the width-aware tiers):
  The count-weighted "measured fixed-overhead floor" speedup = 1/(r*eff + 1-eff), where r =
  byte_ratio (0.785) and eff = the count-weighted weight-read HBM efficiency. So beta_measured(M)
  == eff(M) EXACTLY (proof in floor_speedup_from_eff). Thus the M=1 0.256 efficiency is literally
  the strict byte-proportional fraction, and the M=8 question "does eff rise to >=0.5?" is the same
  as "does the strict-floor speedup tier climb from 1.058 toward the 1.274 roofline?". The QTIP
  realistic tier scales the same way: beta_qtip(M) = (0.51/eff_1)*eff_M (capped 1), recovering
  #388's 0.51 at M=1 and rising with the served width.

METHOD (extends #388; reuses the exact shape table + cb3 byte model + 3-tier machinery):
  (1) Microbench the 8 distinct Gemma-4-E4B body GEMM shapes at M in {1, 8, 4} under int4-Marlin
      (uint4b8 g128). Measure median us/GEMM, weight bytes, eff GB/s, weight-read BW-efficiency.
      THE new number: marlin_m8_hbm_eff / marlin_m4_hbm_eff next to the M=1 0.256.
  (2) Re-derive the 3 speedup tiers at each width: roofline 1.2744 (M-independent ceiling) /
      QTIP-empirical beta(M) (realistic) / measured-fixed-overhead-floor (strict).
  (3) Translate to realized_strict_base_lift_tps_m8 on #378's strict band [357.32, 469.68]
      using f_attn=0.0951 / f_lmhead=0.0224 / draft=0.1201, exactly as #388 did for M=1.
  (4) closes_383_robust_m8: does the STRICT (measured-floor) tier now clear #383's +17.22 floor
      AND +23.75 robust with MORE margin than M=1's straddle -- i.e. does the served regime
      upgrade the strict-tier straddle to a clean pass?

REPRODUCE (0-GPU analytic self-test):
  cd target/ && .venv/bin/python research/validity/cb3_kernel_realized_bw/cb3_kernel_realized_bw.py --self-test
GPU M=8/M=4 microbench (single A10G):
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      research/validity/cb3_kernel_realized_bw/cb3_kernel_realized_bw.py --gpu --m8 --m4 \
      --wandb_group cb3-m8-verify-body-speedup --wandb_name lawine/cb3-m8-verify-body-speedup
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ======================================================================================== #
# Constants -- read from the banked #372/#378/#383 anchors at runtime; hard fallbacks below
# (fallbacks let --self-test run 0-GPU even if a sibling JSON is absent).
# ======================================================================================== #
A10G_SMS = 80
A10G_HBM_PEAK_GBS = 600.0          # GA102 / A10G theoretical HBM bandwidth (datasheet)
BW_BOUND_EFF_THRESHOLD = 0.60      # >= this Marlin-eff => treat M=1 as BW-bound (roofline tight)
M8_BW_BOUND_THRESHOLD = 0.50       # PR #391 served-regime threshold: eff>=0.5 => M=8 is BW-bound
TOL = 1e-6

# ---- served verify widths (#391) ----------------------------------------------------- #
# Deployed PR #52 serves MTP K=7 spec-decode (num_speculative_tokens=7): each decode step runs
# the body GEMM at M = K+1 = 8 verify rows (7 draft + 1 bonus). M=4 is the partial-accept width
# (~half the draft tokens accepted). M=1 is the #388 baseline (pure single-token decode).
MTP_K = 7
SERVED_VERIFY_WIDTH = 8            # M=8: the actually-served verify width (the #391 headline op-point)
PARTIAL_ACCEPT_WIDTH = 4          # M=4: partial-accept width
DEFAULT_WIDTHS = [1, 8, 4]        # M=1 always run as the #388 baseline anchor

_VAL = Path(__file__).resolve().parents[1]
ANCHOR_372 = _VAL / "sub_int4_body_ceiling" / "measure_mixed_precision_results.json"
ANCHOR_378 = _VAL / "deployable_strict_served_tps" / "deployable_strict_served_tps_results.json"
ANCHOR_383 = _VAL / "demand_residual_honest_base" / "demand_residual_honest_base_results.json"

# ---- #372 mixed-precision body-shrink (the input being kernel-validated) -------------- #
INT4_BPW = 4.125                   # deployed: 4-bit + bf16 g128 scale (16/128 = 0.125)
CB3_BPW_UNIFORM = 3.125            # uniform cb3 (dim-2 Gaussian VQ K=64 + g128 incoherence)
CB3_BPW_EFF = 3.2368598382749325   # #372 mixed allocation (88.8% body params at cb3, rest int4)
BODY_BYTES_FRAC = 0.7846932941272564   # = CB3_BPW_EFF / INT4_BPW  (PR's "0.785")
PPL_GATE = 2.42
MIXED_GATE_PPL = 2.3811966031692555    # #372 measured; passes <= 2.42

# ---- #378 served-strict step decomposition (normalized us; sums to step_norm_us) ------ #
STEP_NORM_US = 1218.2
F_ATTN = 0.09506718019009251
F_BODY_STRICT = 0.76240970145034       # body GEMM weight read only (the HONEST shrinkable frac)
F_LMHEAD = 0.022428229458960704
F_DRAFT = 0.12009488890060672          # spec draft-model forward ("other"); does NOT body-shrink
BAND_OFF_THE_SHELF = 357.32166269999993    # #378 worse-case VBI=1 strict base (off-the-shelf #326)
BAND_FLOOR = 469.6847174760462             # #378 better-case VBI=1 strict base (first-principles #327)

# ---- #383 supply-side gap the body-shrink must close ---------------------------------- #
SUPPLY_FLOOR_JOINT_TPS = 17.216386736379093    # demand+gap joint floor
SUPPLY_ROBUST_ET_ONLY_TPS = 23.74874176829968  # E[T]-only robust target
DEPLOYABLE_FLOOR_383 = 469.68

# ---- published kernel evidence (literature cross-check; arxiv 2406.11235 QTIP Tab4/17) - #
# RTX6000-Ada batch=1: QTIP-3bit=2.88x fp16, int4-Marlin~2.0-2.5x fp16 => QTIP/Marlin in
# [1.15, 1.44]; conservative low end 1.15 used as the literature-anchored floor speedup.
LIT_QTIP3_OVER_MARLIN_LO = 1.15
LIT_QTIP3_OVER_MARLIN_HI = 1.28        # ~= the byte-ratio roofline (codebook does not eat savings)
# QTIP Table 4 batch=1 tok/s (RTX6000-Ada): 2-bit 188 / 3-bit 161 / 4-bit 140 / fp16 55.9.
# The realized speedup per bpw-step is FAR below the byte roofline -> batch=1 is overhead-heavy
# even for QTIP's own kernel. Solving realized_time_ratio = r*beta + (1-beta) on both the
# 4->3bit step (r=0.75, sp=1.15 => beta=0.52) and 2->4bit step (r=0.50, sp=1.343 => beta=0.51)
# gives a consistent byte-proportional fraction beta ~= 0.51. Pod-measured Marlin transfer/total
# on the dominant MLP shapes (~0.44) corroborates. This is the literature-anchored REALISTIC tier.
QTIP_BETA_BYTE_PROPORTIONAL = 0.51
# #388 banked count-weighted Marlin M=1 weight-read HBM efficiency (g5lfdpgw). Used (a) as the
# 0-GPU self-test eff_1 anchor and (b) as the QTIP ratio denominator when M=1 was not re-measured.
# In a GPU run with M=1 in the sweep, the FRESHLY measured eff_1 is used instead (self-consistent).
M1_MEASURED_HBM_EFF_388 = 0.25561637483960586

# ---- Marlin batch-size sweep literature (arxiv 2408.11743, A10, Llama-2-7B) -- #391 prior ---- #
# Table 2 end-to-end speedup vs FP16 is ~FLAT M=1 (2.93x) -> M=8 (2.90x): the int4 GEMM wall-clock
# time is ~flat across M=1..8 because Marlin is ALREADY software-pipelined at M=1. Roofline
# memory->compute crossover is M~=64 (Fig 11), so M=8 is solidly BANDWIDTH-bound as a REGIME.
# KEY CONSEQUENCE for this card: because Marlin time is ~flat M=1->8, the weight-read EFFICIENCY
# (wbytes/time) does NOT rise at M=8 -- the M=1 0.256 count-weighted number already sits near
# Marlin's achievable weight-read BW (held down by small launch-bound attention GEMMs + a peak-COPY
# reference above modest-GEMM achievable BW), NOT a latency headroom that M=8 recovers. QuIP# Table 5
# corroborates: codebook kernels reach only 29-57% of peak at M=1, rising with matrix SIZE not batch.
# So the honest #391 prior is: marlin_m8_hbm_eff ~= marlin_m1_hbm_eff (flat), the strict straddle
# stands, and "M=8 is BW-bound (regime)" does NOT imply "efficiency rises". Confounder for the QTIP
# realistic tier: QTIP/QuIP# Hadamard dequant is O(M*N) -> grows with M -> could SUPPRESS beta at
# M=8 (no codebook paper measures batch>1), so the beta(M) rise is weakly-supported / optimistic.
MARLIN_SPEEDUP_VS_FP16_M1 = 2.93
MARLIN_SPEEDUP_VS_FP16_M8 = 2.90
MARLIN_ROOFLINE_CROSSOVER_M = 64        # M < this => memory-bound regime on Ampere (Fig 11)

OFFICIAL_TPS = 481.53                  # PR #52 baseline (context only; this card is 0 official TPS)

# ======================================================================================== #
# Body GEMM shapes -- (out_features, in_features, total_instance_count). 8 distinct shapes.
# Extracted from the gemma-4-E4B-it-qat safetensors (42 layers; 24 own-KV + 18 KV-shared;
# full-attn @ {5,11,17,23,29,35,41}). Total body params ~= 3.890B (MLP ~= 84.9%).
# ======================================================================================== #
BODY_SHAPES: list[dict[str, Any]] = [
    {"name": "q_full",  "out": 4096,  "in": 2560,  "count": 7},
    {"name": "q_slide", "out": 2048,  "in": 2560,  "count": 35},
    {"name": "kv_full", "out": 1024,  "in": 2560,  "count": 8},    # k_full x4 + v_full x4
    {"name": "kv_slide","out": 512,   "in": 2560,  "count": 40},   # k_slide x20 + v_slide x20
    {"name": "o_full",  "out": 2560,  "in": 4096,  "count": 7},
    {"name": "o_slide", "out": 2560,  "in": 2048,  "count": 35},
    {"name": "gate_up", "out": 10240, "in": 2560,  "count": 84},   # gate x42 + up x42
    {"name": "down",    "out": 2560,  "in": 10240, "count": 42},
]


def _shape_params(s: dict[str, Any]) -> int:
    return s["out"] * s["in"] * s["count"]


def _int4_weight_bytes(out: int, inn: int) -> float:
    """int4-Marlin weight-read bytes for one GEMM (4.125 bpw = 4b weight + bf16 g128 scale)."""
    return out * inn * INT4_BPW / 8.0


# ======================================================================================== #
# Pure analytic core (0-GPU): byte ratio, roofline speedup, served-step translation, gate.
# ======================================================================================== #
def byte_ratio() -> float:
    return CB3_BPW_EFF / INT4_BPW


def roofline_speedup() -> float:
    """cb3 reads byte_ratio x the bytes; at equal BW-efficiency that is 1/byte_ratio faster."""
    return INT4_BPW / CB3_BPW_EFF


def qtip_empirical_speedup() -> float:
    """Literature-anchored REALISTIC speedup: only the byte-proportional fraction (beta~=0.51
    from QTIP batch=1 Table 4) of the step shrinks with the weight bytes; the rest is fixed
    launch/dequant/occupancy overhead. realized_time_ratio = r*beta + (1-beta)."""
    r = byte_ratio()
    beta = QTIP_BETA_BYTE_PROPORTIONAL
    return 1.0 / (r * beta + (1.0 - beta))


# ---- width-aware tiers (#391): same r*beta + (1-beta) engine, beta(M) driven by measured eff -- #
def speedup_from_beta(beta: float) -> float:
    """realized_time_ratio = r*beta + (1-beta); speedup = 1/ratio. beta in [0,1]:
    beta=0 -> 1.0x (all fixed overhead), beta=1 -> roofline 1/r (fully byte-proportional)."""
    r = byte_ratio()
    beta = min(max(beta, 0.0), 1.0)
    return 1.0 / (r * beta + (1.0 - beta))


def floor_speedup_from_eff(eff: float) -> float:
    """The count-weighted MEASURED fixed-overhead-floor speedup as a closed form of the measured
    weight-read HBM efficiency `eff`. Proof that beta_measured == eff exactly:
      per shape, cb3_fixed = r*t_transfer + t_overhead, t_transfer = wbytes/peak, t_overhead =
      t_meas - t_transfer.  Count-weighted: 1/speedup = (r*SUM c*t_transfer + SUM c*t_overhead)
      / SUM c*t_meas = r*B + (1-B) with B = SUM c*t_transfer / SUM c*t_meas
      = (SUM c*wbytes / peak) / SUM c*t_meas = agg_eff_gbs/peak = count_weighted HBM eff = eff.
    So the strict floor tier is exactly speedup_from_beta(eff)."""
    return speedup_from_beta(eff)


def qtip_beta_at_width(eff_m: float, eff_1: float,
                       beta1: float = QTIP_BETA_BYTE_PROPORTIONAL) -> float:
    """beta_qtip(M): scale the literature batch=1 QTIP beta (0.51) by the measured Marlin
    efficiency ratio eff_M/eff_1, i.e. assume QTIP keeps the SAME kernel-quality ratio
    (beta_qtip/beta_marlin_floor ~= 0.51/eff_1 ~= 2.0) over the strict floor at the served width
    as it had at M=1. Recovers 0.51 at eff_M==eff_1; capped at 1.0 (the roofline). The honest
    CONSERVATIVE alternative (QTIP gains nothing with M) holds beta fixed at 0.51 -- both reported."""
    if eff_1 <= 0:
        return min(max(beta1, 0.0), 1.0)
    return min((beta1 / eff_1) * eff_m, 1.0)


def width_speedup_tiers(eff_m: float, eff_1: float) -> dict[str, Any]:
    """The three realized-speedup tiers at a given verify width, from the measured HBM effs.
      * roofline      = 1/r (M-independent BW-bound ceiling)
      * qtip_empirical= realistic: beta_qtip(M) = (0.51/eff_1)*eff_M scaled (capped 1)
      * measured_floor= strict: beta_measured(M) = eff_M (the closed form above)
    Also reports the conservative fixed-beta=0.51 QTIP tier (no M gain) as the lower QTIP bracket."""
    beta_qtip = qtip_beta_at_width(eff_m, eff_1)
    return {
        "eff": eff_m,
        "beta_measured_floor": eff_m,
        "beta_qtip_scaled": beta_qtip,
        "beta_qtip_fixed": QTIP_BETA_BYTE_PROPORTIONAL,
        "roofline": roofline_speedup(),
        "qtip_empirical": speedup_from_beta(beta_qtip),
        "qtip_empirical_fixed_beta": qtip_empirical_speedup(),
        "measured_floor": floor_speedup_from_eff(eff_m),
    }


def lift_factor(speedup: float, f_body: float) -> float:
    """New/old served-TPS multiplier when the body-time fraction f_body speeds up by `speedup`."""
    r = 1.0 / speedup
    new_step = (1.0 - f_body) + f_body * r
    return 1.0 / new_step


def translate_band(speedup: float, f_body: float) -> dict[str, float]:
    lf = lift_factor(speedup, f_body)
    lo_new = BAND_OFF_THE_SHELF * lf
    hi_new = BAND_FLOOR * lf
    return {
        "lift_factor": lf,
        "base_off_the_shelf": BAND_OFF_THE_SHELF,
        "base_floor": BAND_FLOOR,
        "lifted_off_the_shelf": lo_new,
        "lifted_floor": hi_new,
        "delta_off_the_shelf": lo_new - BAND_OFF_THE_SHELF,
        "delta_floor": hi_new - BAND_FLOOR,
    }


def f_body_complement() -> float:
    """PR-spec body fraction = complement of attention + lm_head (lumps the draft/'other' in)."""
    return 1.0 - F_ATTN - F_LMHEAD


def activation_hadamard_tax_frac() -> dict[str, float]:
    """Analytic upper bound on the ONLY online RHT cost: a 1.2x-memcpy activation Hadamard on
    the GEMM input vector (dim = in_features), per token. Ratio vs the int4 weight read shows
    it is negligible at M=1 (weight read dominates). RHT on the weights is folded OFFLINE."""
    num = 0.0
    den = 0.0
    for s in BODY_SHAPES:
        act_bytes = 1.2 * s["in"] * 2.0          # bf16 activation vector, 1.2x memcpy overhead
        w_bytes = _int4_weight_bytes(s["out"], s["in"])
        num += s["count"] * act_bytes
        den += s["count"] * w_bytes
    return {"hadamard_over_weight_read_frac": num / den, "note": "online activation RHT only; <<1%"}


def supply_gate(lift_tps_conservative: float) -> dict[str, Any]:
    """closes_383_supply_gap_floor: does the (conservative) realized lift clear the #383 floor?"""
    clears_floor = lift_tps_conservative >= SUPPLY_FLOOR_JOINT_TPS
    clears_robust = lift_tps_conservative >= SUPPLY_ROBUST_ET_ONLY_TPS
    return {
        "supply_floor_joint_tps": SUPPLY_FLOOR_JOINT_TPS,
        "supply_robust_et_only_tps": SUPPLY_ROBUST_ET_ONLY_TPS,
        "conservative_lift_tps": lift_tps_conservative,
        "clears_floor": bool(clears_floor),
        "clears_robust": bool(clears_robust),
        "closes_383_supply_gap_floor": bool(clears_floor and clears_robust),
    }


def analytic_payload() -> dict[str, Any]:
    """Everything computable with no GPU: the speedup tiers + translation + #383 gate."""
    r = byte_ratio()
    S_roof = roofline_speedup()
    S_qtip = qtip_empirical_speedup()
    f_comp = f_body_complement()

    variants: dict[str, dict[str, float]] = {}
    for s_name, S in (("roofline", S_roof), ("qtip_empirical", S_qtip),
                      ("lit_conservative", LIT_QTIP3_OVER_MARLIN_LO)):
        for fb_name, fb in (("complement", f_comp), ("body_only", F_BODY_STRICT)):
            variants[f"{s_name}__{fb_name}"] = {"speedup": S, "f_body": fb, **translate_band(S, fb)}

    # #383 gate driver = REALISTIC (QTIP-empirical) speedup, body-only f_body, off-the-shelf base:
    # the literature-anchored estimate at the most conservative band/fraction cell.
    realistic_cons = variants["qtip_empirical__body_only"]["delta_off_the_shelf"]
    # Headline "realized" lift: REALISTIC speedup on the PR's complement f_body, off-the-shelf base
    # (the honest realized estimate -- NOT the roofline, which M=1-non-BW-bound makes optimistic).
    headline_lift = variants["qtip_empirical__complement"]["delta_off_the_shelf"]
    # PR literal-formula band (roofline speedup x complement) kept as a transparent UPPER bound.
    roofline_formula = variants["roofline__complement"]

    gate = supply_gate(realistic_cons)
    return {
        "byte_ratio": r,
        "byte_ratio_rounded3": round(r, 3),
        "realized_body_speedup": S_roof,
        "realized_body_speedup_qtip_empirical": S_qtip,
        "realized_is_roofline_bound": True,
        "f_body_complement_pr": f_comp,
        "f_body_strict_honest": F_BODY_STRICT,
        "lit_bracket_speedup": [LIT_QTIP3_OVER_MARLIN_LO, LIT_QTIP3_OVER_MARLIN_HI],
        "qtip_beta_byte_proportional": QTIP_BETA_BYTE_PROPORTIONAL,
        "translation_variants": variants,
        "headline_realized_strict_base_lift_tps": headline_lift,
        "roofline_formula_lift_off_the_shelf": roofline_formula["delta_off_the_shelf"],
        "roofline_formula_lift_floor": roofline_formula["delta_floor"],
        "supply_gate": gate,
        "activation_hadamard_tax": activation_hadamard_tax_frac(),
    }


def width_full_analysis(width: int, eff_m: float, eff_1: float) -> dict[str, Any]:
    """Full #391 per-width rollup: 3 tiers -> band translations (both f_body) -> #383 gates.
    HEADLINE realized_strict_base_lift_tps (exactly as #388): qtip_empirical x complement f_body,
    off-the-shelf base. closes_383_robust: driven by the STRICT measured-floor tier (body-only,
    off-the-shelf) -- this is the #391 question, whether the served width upgrades the M=1
    measured-floor STRADDLE to a clean pass clearing both +17.22 floor and +23.75 robust."""
    tiers = width_speedup_tiers(eff_m, eff_1)
    f_comp = f_body_complement()
    tier_map = (("roofline", tiers["roofline"]),
                ("qtip_empirical", tiers["qtip_empirical"]),
                ("qtip_fixed_beta", tiers["qtip_empirical_fixed_beta"]),
                ("measured_floor", tiers["measured_floor"]))
    variants: dict[str, dict[str, float]] = {}
    for s_name, S in tier_map:
        for fb_name, fb in (("complement", f_comp), ("body_only", F_BODY_STRICT)):
            variants[f"{s_name}__{fb_name}"] = {"speedup": S, "f_body": fb, **translate_band(S, fb)}

    headline_lift = variants["qtip_empirical__complement"]["delta_off_the_shelf"]
    realistic_cons = variants["qtip_empirical__body_only"]["delta_off_the_shelf"]
    strict_cons = variants["measured_floor__body_only"]["delta_off_the_shelf"]
    gate_qtip = supply_gate(realistic_cons)
    gate_strict = supply_gate(strict_cons)
    return {
        "width": width,
        "marlin_hbm_eff": eff_m,
        # measured-efficiency verdict (the PR's eff>=0.5 question -- "does efficiency RISE?")
        "is_bw_bound": bool(eff_m >= M8_BW_BOUND_THRESHOLD),
        "is_bw_bound_threshold": M8_BW_BOUND_THRESHOLD,
        # literature REGIME verdict (Marlin Fig 11: M < ~64 is memory-bound on Ampere). Distinct
        # from the efficiency-vs-peak-copy verdict: a kernel can be in the BW-bound regime yet sit
        # well below peak-copy efficiency (small/medium GEMMs, launch-bound attention shapes).
        "is_bw_bound_regime_literature": bool(width < MARLIN_ROOFLINE_CROSSOVER_M),
        "tiers": tiers,
        "realized_body_speedup": {  # the 3 named tiers (PR key realized_body_speedup_m8)
            "roofline": tiers["roofline"],
            "qtip_empirical": tiers["qtip_empirical"],
            "measured_floor": tiers["measured_floor"],
        },
        "translation_variants": variants,
        "realized_strict_base_lift_tps": headline_lift,           # qtip_empirical x complement
        "realized_strict_base_lift_tps_measured_floor": strict_cons,
        "supply_gate_qtip": gate_qtip,
        "supply_gate_strict_floor": gate_strict,
        # closes_383_robust: the STRICT-tier verdict (the #391 straddle->pass question)
        "closes_383_robust": bool(gate_strict["clears_floor"] and gate_strict["clears_robust"]),
        "closes_383_robust_qtip": bool(gate_qtip["clears_floor"] and gate_qtip["clears_robust"]),
    }


# ======================================================================================== #
# Self-test (0-GPU): asserts the PR #388 step-5 contract.
# ======================================================================================== #
def self_test() -> dict[str, Any]:
    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    r = byte_ratio()
    S = roofline_speedup()
    ap = analytic_payload()

    # 1. byte-count ratio == 3.2369/4.125 == 0.785 (PR step 5)
    chk("byte_ratio_eq_372", abs(r - BODY_BYTES_FRAC) < 1e-9, f"r={r:.12f}")
    chk("byte_ratio_rounds_0p785", round(r, 3) == 0.785, f"round(r,3)={round(r,3)}")
    # 2. roofline speedup is the byte-ratio-implied floor, and > 1
    chk("roofline_is_inv_byte_ratio", abs(S - 1.0 / r) < TOL, f"S={S:.10f} 1/r={1.0/r:.10f}")
    chk("roofline_gt_1", S > 1.0, f"S={S:.6f}")
    chk("roofline_eq_bpw_ratio", abs(S - INT4_BPW / CB3_BPW_EFF) < TOL, "")
    # 3. lift_factor bounded in (1, S) for both f_body variants (monotone, sane translation)
    for nm, fb in (("complement", f_body_complement()), ("body_only", F_BODY_STRICT)):
        lf = lift_factor(S, fb)
        chk(f"lift_factor_in_1_S_{nm}", 1.0 < lf < S, f"lf={lf:.6f} S={S:.6f}")
    # 4. lift is strictly positive on the band (speedup helps)
    tb = translate_band(S, f_body_complement())
    chk("band_lift_positive_off", tb["delta_off_the_shelf"] > 0, f"{tb['delta_off_the_shelf']:.3f}")
    chk("band_lift_positive_floor", tb["delta_floor"] > 0, f"{tb['delta_floor']:.3f}")
    # 5. #383 gate logic holds on the most-conservative cell
    g = ap["supply_gate"]
    chk("supply_gate_closes_floor", g["closes_383_supply_gap_floor"],
        f"cons_lift={g['conservative_lift_tps']:.3f} floor={g['supply_floor_joint_tps']:.3f}")
    # 6. activation-RHT online tax is negligible (<1% of weight read)
    tax = ap["activation_hadamard_tax"]["hadamard_over_weight_read_frac"]
    chk("hadamard_tax_negligible", tax < 0.01, f"tax={tax:.5f}")
    # 7. step fractions sum to 1 (decomposition consistency)
    fsum = F_ATTN + F_BODY_STRICT + F_LMHEAD + F_DRAFT
    chk("step_fractions_sum_1", abs(fsum - 1.0) < 1e-6, f"sum={fsum:.9f}")
    # 8. body shapes reconstruct ~3.89B params (shape table integrity)
    tot = sum(_shape_params(s) for s in BODY_SHAPES)
    chk("body_params_3p89B", 3.80e9 < tot < 3.95e9, f"total={tot/1e9:.4f}B")
    # 9. NaN/inf-clean across all numeric outputs
    nan_clean = all(_finite(v) for v in _iter_numeric(ap))
    chk("nan_inf_clean", nan_clean, "")
    # 10. #372 gate PPL passes (the body-shrink is PPL-feasible -- the precondition)
    chk("ppl_372_passes", MIXED_GATE_PPL <= PPL_GATE, f"ppl={MIXED_GATE_PPL} gate={PPL_GATE}")

    # ---- #391 width-machinery analytic checks (0-GPU; synthetic effs) ----------------- #
    e1 = M1_MEASURED_HBM_EFF_388
    # 11. beta engine bounds: beta=0 -> 1.0x (all overhead), beta=1 -> roofline (fully BW-bound)
    chk("speedup_beta0_is_1", abs(speedup_from_beta(0.0) - 1.0) < TOL, f"{speedup_from_beta(0.0):.6f}")
    chk("speedup_beta1_is_roofline", abs(speedup_from_beta(1.0) - roofline_speedup()) < TOL, "")
    # 12. floor speedup recovers #388's M=1 measured-floor 1.0582 from the banked eff
    chk("floor_recovers_388_m1", abs(floor_speedup_from_eff(e1) - 1.058241282158072) < 1e-6,
        f"floor(e1)={floor_speedup_from_eff(e1):.6f}")
    # 13. floor speedup STRICTLY rises with efficiency (served width helps iff eff rises)
    chk("floor_monotone_in_eff", floor_speedup_from_eff(0.50) > floor_speedup_from_eff(e1),
        f"f(0.5)={floor_speedup_from_eff(0.50):.4f} > f(e1)={floor_speedup_from_eff(e1):.4f}")
    # 14. qtip beta recovers the literature 0.51 at M=1 (eff_m == eff_1) and is capped at 1.0
    chk("qtip_beta_recovers_0p51", abs(qtip_beta_at_width(e1, e1) - QTIP_BETA_BYTE_PROPORTIONAL) < 1e-9,
        f"beta={qtip_beta_at_width(e1, e1):.6f}")
    chk("qtip_beta_capped_1", qtip_beta_at_width(0.95, e1) == 1.0, f"{qtip_beta_at_width(0.95, e1):.4f}")
    # 15. M=1 width_full_analysis reproduces #388: qtip headline 38.34, strict-floor STRADDLES
    #     (closes_383_robust False), and a BW-bound eff=0.5 upgrades the strict tier to a PASS.
    w1 = width_full_analysis(1, e1, e1)
    chk("m1_reproduces_388_headline", abs(w1["realized_strict_base_lift_tps"] - 38.34161969078741) < 1e-4,
        f"headline={w1['realized_strict_base_lift_tps']:.4f}")
    chk("m1_strict_straddles_388", (not w1["closes_383_robust"]) and (not w1["is_bw_bound"]),
        f"closes={w1['closes_383_robust']} bw_bound={w1['is_bw_bound']}")
    w_bw = width_full_analysis(8, 0.50, e1)
    chk("bwbound_upgrades_strict_to_pass", w_bw["closes_383_robust"] and w_bw["is_bw_bound"],
        f"strict_lift={w_bw['realized_strict_base_lift_tps_measured_floor']:.3f} "
        f"closes={w_bw['closes_383_robust']}")
    # 16. strict-tier lift is monotone in eff (higher served-width eff => bigger margin over #383)
    chk("strict_lift_monotone", (width_full_analysis(8, 0.50, e1)["realized_strict_base_lift_tps_measured_floor"]
                                 > w1["realized_strict_base_lift_tps_measured_floor"]),
        "")
    # 17. all width tiers NaN/inf-clean
    chk("width_tiers_finite", all(_finite(v) for v in _iter_numeric(width_full_analysis(8, 0.50, e1))), "")

    passes = all(c[1] for c in checks)
    return {
        "passes": bool(passes),
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c[1]),
        "checks": [{"name": n, "ok": ok, "detail": d} for (n, ok, d) in checks],
    }


def _iter_numeric(o: Any):
    if isinstance(o, dict):
        for v in o.values():
            yield from _iter_numeric(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _iter_numeric(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


# ======================================================================================== #
# GPU microbench (int4-Marlin M=1 per shape + device peak-copy BW reference).
# ======================================================================================== #
def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Launch with CUDA_VISIBLE_DEVICES=0 (the single-A10G pod default "
            "points at a non-existent 2nd GPU -- the #358/#363 gotcha).")
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "total_mem_gib": round(p.total_memory / (1024**3), 2),
        "is_a10g_80sm": bool(p.multi_processor_count == A10G_SMS and "A10G" in p.name),
        "is_ga102_sm86": bool(cc == (8, 6)),
    }


def _time_us(fn, iters: int, warmup: int) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # median, us


def _measure_peak_copy_gbs(dev, iters: int, warmup: int) -> dict[str, float]:
    """Achievable HBM bandwidth on THIS pod via a large bf16 device-to-device copy."""
    import torch
    n = 64 * 1024 * 1024  # 64M bf16 elems = 128 MiB; read+write = 256 MiB / copy
    x = torch.randn(n, dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(x)
    us = _time_us(lambda: y.copy_(x), iters, warmup)
    moved_bytes = 2 * x.numel() * 2  # read + write, 2 bytes/elem
    gbs = moved_bytes / (us * 1e-6) / 1e9
    return {"copy_us": us, "moved_bytes": float(moved_bytes), "peak_copy_gbs": gbs,
            "peak_theoretical_gbs": A10G_HBM_PEAK_GBS, "copy_eff_vs_theoretical": gbs / A10G_HBM_PEAK_GBS}


def _build_marlin_gemm(out: int, inn: int, dev, m: int = 1):
    """Return a 0-arg callable running one int4-Marlin (uint4b8 g128) GEMM at verify width M
    (M activation rows over the SAME weight matrix), + weight bytes. The weight read is
    M-independent (loaded once, reused across rows) -- that is exactly why cb3's byte shrink
    is the same absolute saving at every M; what changes with M is how BW-bound the kernel is."""
    import torch
    from vllm import _custom_ops as ops  # noqa: F401  (ensures custom ops registered)
    from vllm.scalar_type import scalar_types
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

    K, N = inn, out                 # Marlin weight is [K=in, N=out]
    wtype = scalar_types.uint4b8
    gs = 128
    w = (torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02)
    _w_ref, q_w, s, _g_idx, _sort, _rp = mt.marlin_quantize(w, wtype, gs, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    g_idx = torch.empty(0, dtype=torch.int, device=dev)
    sort_idx = torch.empty(0, dtype=torch.int, device=dev)
    x = torch.randn(m, K, dtype=torch.bfloat16, device=dev)

    def run():
        return mu.apply_gptq_marlin_linear(
            x, q_w, s, zp, g_idx, sort_idx, ws, wtype,
            output_size_per_partition=N, input_size_per_partition=K, is_k_full=True)

    # sanity: finite + correct shape on first call
    out_t = run()
    ok = bool(out_t.shape == (m, N) and torch.isfinite(out_t).all().item())
    return run, _int4_weight_bytes(out, inn), ok


def microbench_at_width(m: int, dev, peak_gbs: float, iters: int, warmup: int) -> dict[str, Any]:
    """int4-Marlin per-shape microbench at verify width M. The weight-read efficiency
    (count_weighted_marlin_bw_eff) is the headline marlin_m{M}_hbm_eff -- same weight-byte
    numerator as #388's M=1 0.256, new M-width denominator (median us/GEMM at M rows)."""
    per_shape: list[dict[str, Any]] = []
    tot_params = 0.0
    tot_int4_bytes = 0.0          # count-weighted total body weight bytes (int4); M-independent
    tot_total_bytes = 0.0        # count-weighted weight + act-read + out-write (BW-bound diagnostic)
    tot_marlin_time_us = 0.0     # count-weighted total body GEMM time at width M
    tot_cb3_fixed_us = 0.0       # count-weighted cb3 time under the FIXED-OVERHEAD model
    all_ok = True
    for sh in BODY_SHAPES:
        run, wbytes, ok = _build_marlin_gemm(sh["out"], sh["in"], dev, m)
        all_ok = all_ok and ok
        us = _time_us(run, iters, warmup)
        gbs = wbytes / (us * 1e-6) / 1e9                 # weight-read effective GB/s
        bw_eff = gbs / peak_gbs                           # weight-read HBM efficiency (THE metric)
        # total bytes moved at width M: weight (once) + M act rows in + M out rows out (bf16)
        act_out_bytes = m * sh["in"] * 2.0 + m * sh["out"] * 2.0
        total_bytes = wbytes + act_out_bytes
        total_gbs = total_bytes / (us * 1e-6) / 1e9
        total_bw_eff = total_gbs / peak_gbs               # incl act/out (secondary diagnostic)
        params = _shape_params(sh)
        cb3_bytes = wbytes * BODY_BYTES_FRAC
        cb3_roof_us = us * BODY_BYTES_FRAC                # roofline: same BW-eff => time scales w/ bytes
        # fixed-overhead model: only the WEIGHT transfer component shrinks (cb3 shrinks weights
        # only); launch/dequant/act/out overhead (the part that keeps it off the roofline) is held
        # fixed -> a measured-data FLOOR. At higher M, a BW-bound kernel has less fixed overhead
        # relative to the (now-dominant) weight transfer, so the floor speedup climbs.
        t_transfer_us = wbytes / (peak_gbs * 1e9) * 1e6
        t_overhead_us = max(us - t_transfer_us, 0.0)
        cb3_fixed_us = BODY_BYTES_FRAC * t_transfer_us + t_overhead_us
        sp_fixed = us / cb3_fixed_us if cb3_fixed_us > 0 else 1.0
        per_shape.append({
            "name": sh["name"], "out": sh["out"], "in": sh["in"], "count": sh["count"], "m": m,
            "params": params, "int4_weight_mib": wbytes / (1024**2),
            "cb3_weight_mib": cb3_bytes / (1024**2),
            "marlin_us": us, "marlin_eff_gbs": gbs, "marlin_bw_eff": bw_eff,
            "marlin_total_eff_gbs": total_gbs, "marlin_total_bw_eff": total_bw_eff,
            "cb3_roofline_us": cb3_roof_us,
            "transfer_us_at_peak": t_transfer_us, "overhead_us": t_overhead_us,
            "cb3_fixed_overhead_us": cb3_fixed_us, "fixed_overhead_speedup": sp_fixed,
            "finite_ok": ok,
        })
        tot_params += params
        tot_int4_bytes += sh["count"] * wbytes
        tot_total_bytes += sh["count"] * total_bytes
        tot_marlin_time_us += sh["count"] * us
        tot_cb3_fixed_us += sh["count"] * cb3_fixed_us

    agg_eff_gbs = tot_int4_bytes / (tot_marlin_time_us * 1e-6) / 1e9
    agg_bw_eff = agg_eff_gbs / peak_gbs
    agg_total_eff_gbs = tot_total_bytes / (tot_marlin_time_us * 1e-6) / 1e9
    agg_total_bw_eff = agg_total_eff_gbs / peak_gbs
    # BW-bound verdicts: 0.5 is the #391 served-regime threshold; 0.6 keeps #388 continuity.
    bw_bound = agg_bw_eff >= M8_BW_BOUND_THRESHOLD
    bw_bound_388 = agg_bw_eff >= BW_BOUND_EFF_THRESHOLD
    # count-weighted measured-data floor speedup (== speedup_from_beta(agg_bw_eff); see proof)
    measured_floor_speedup = tot_marlin_time_us / tot_cb3_fixed_us

    return {
        "width": m,
        "per_shape": per_shape,
        "aggregate": {
            "width": m,
            "total_body_params": tot_params,
            "total_int4_weight_gib": tot_int4_bytes / (1024**3),
            "count_weighted_marlin_eff_gbs": agg_eff_gbs,
            "count_weighted_marlin_bw_eff": agg_bw_eff,           # == marlin_m{M}_hbm_eff
            "count_weighted_total_eff_gbs": agg_total_eff_gbs,
            "count_weighted_total_bw_eff": agg_total_bw_eff,
            "is_bw_bound": bool(bw_bound),
            "is_bw_bound_388_threshold": bool(bw_bound_388),
            "bw_bound_threshold": M8_BW_BOUND_THRESHOLD,
            "bw_bound_threshold_388": BW_BOUND_EFF_THRESHOLD,
            "measured_floor_speedup": measured_floor_speedup,
            "roofline_speedup": roofline_speedup(),
        },
        "all_shapes_finite_ok": bool(all_ok),
    }


def gpu_microbench(widths: list[int], iters: int, warmup: int) -> dict[str, Any]:
    """Sweep the body-GEMM microbench across verify widths (M=1 baseline + served M=8 / M=4).
    Peak-copy BW and GPU facts are measured once. M=1 aggregate/per_shape are aliased to the top
    level for #388 backward-compat; all widths live under by_width."""
    import torch
    dev = _device()
    gpu = _gpu_facts(dev)
    peak = _measure_peak_copy_gbs(dev, iters, warmup)
    peak_gbs = peak["peak_copy_gbs"]

    by_width: dict[str, Any] = {}
    for m in widths:
        by_width[str(m)] = microbench_at_width(m, dev, peak_gbs, iters, warmup)

    base = by_width.get("1") or by_width[str(widths[0])]   # M=1 anchor (always present in practice)
    # keep #388's "m1_is_bw_bound" key (0.6 threshold) on the M=1 aggregate for continuity
    base["aggregate"]["m1_is_bw_bound"] = bool(base["aggregate"]["is_bw_bound_388_threshold"])
    return {
        "gpu": gpu,
        "peak_copy": peak,
        "widths": list(widths),
        "by_width": by_width,
        # ---- #388 backward-compat M=1 aliases ----
        "per_shape": base["per_shape"],
        "aggregate": base["aggregate"],
        "all_shapes_finite_ok": bool(all(by_width[w]["all_shapes_finite_ok"] for w in by_width)),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
    }


# ======================================================================================== #
# Reporting + wandb
# ======================================================================================== #
def build_payload(args, micro: dict[str, Any] | None, st: dict[str, Any]) -> dict[str, Any]:
    ap = analytic_payload()
    S_roof = ap["realized_body_speedup"]
    f_comp = ap["f_body_complement_pr"]

    headline = ap["headline_realized_strict_base_lift_tps"]
    gate = ap["supply_gate"]

    # Headline #383 gate stays on the literature-anchored REALISTIC (QTIP-empirical) speedup.
    # When the GPU ran, ALSO report the strict measured fixed-overhead floor as a sensitivity
    # (the most pessimistic cell -- it straddles the #383 floor, so the pass has thin margin).
    measured_floor_speedup = None
    supply_gate_strict_floor = None
    if micro is not None:
        measured_floor_speedup = micro["aggregate"]["measured_floor_speedup"]
        for fb_name, fb in (("complement", f_comp), ("body_only", F_BODY_STRICT)):
            ap["translation_variants"][f"measured_floor__{fb_name}"] = {
                "speedup": measured_floor_speedup, "f_body": fb,
                **translate_band(measured_floor_speedup, fb)}
        strict_cons = ap["translation_variants"]["measured_floor__body_only"]["delta_off_the_shelf"]
        supply_gate_strict_floor = supply_gate(strict_cons)

    # ---- #391 width-aware analysis: M=8 served verify + M=4 partial-accept ------------- #
    width_analysis: dict[str, Any] | None = None
    m8 = m4 = None
    m8_note = None
    if micro is not None and "by_width" in micro:
        bw = micro["by_width"]
        eff_1 = (bw["1"]["aggregate"]["count_weighted_marlin_bw_eff"] if "1" in bw
                 else M1_MEASURED_HBM_EFF_388)
        width_analysis = {}
        for w_str, blk in bw.items():
            eff_m = blk["aggregate"]["count_weighted_marlin_bw_eff"]
            width_analysis[w_str] = width_full_analysis(int(w_str), eff_m, eff_1)
        m8 = width_analysis.get(str(SERVED_VERIFY_WIDTH))
        m4 = width_analysis.get(str(PARTIAL_ACCEPT_WIDTH))
        if m8 is not None:
            eff8 = m8["marlin_hbm_eff"]
            strict8 = m8["realized_strict_base_lift_tps_measured_floor"]
            strict1 = (width_analysis.get("1", {}).get(
                "realized_strict_base_lift_tps_measured_floor", 15.65))
            if m8["is_bw_bound"]:
                m8_note = (
                    f"SERVED M=8 IS BW-BOUND: marlin_m8_hbm_eff={eff8:.3f} >= "
                    f"{M8_BW_BOUND_THRESHOLD} (up from M=1 {eff_1:.3f}). Strict measured-floor lift "
                    f"+{strict8:.1f} TPS clears #383 +17.22 floor AND +23.75 robust -> the served "
                    f"regime UPGRADES the M=1 straddle (+{strict1:.1f}) to a CLEAN PASS.")
            elif m8["closes_383_robust"]:
                m8_note = (
                    f"SERVED M=8 partially BW-bound: marlin_m8_hbm_eff={eff8:.3f} (< "
                    f"{M8_BW_BOUND_THRESHOLD} but up from M=1 {eff_1:.3f}); strict measured-floor "
                    f"lift +{strict8:.1f} TPS still clears #383 floor+robust -> straddle upgraded "
                    f"to a pass with margin.")
            else:
                delta_eff = eff8 - eff_1
                m8_note = (
                    f"SERVED M=8 WEIGHT-READ EFFICIENCY ~FLAT: marlin_m8_hbm_eff={eff8:.3f} vs M=1 "
                    f"{eff_1:.3f} (delta {delta_eff:+.3f}), < {M8_BW_BOUND_THRESHOLD}. This matches the "
                    f"Marlin prior (2408.11743 Tab2: int4 GEMM time ~flat M=1->8, speedup 2.93->2.90x): "
                    f"Marlin is ALREADY software-pipelined at M=1, so M=8 does more compute at ~flat "
                    f"weight-read time -> efficiency does NOT rise. M=8 IS bandwidth-bound as a REGIME "
                    f"(M<~64 crossover, Fig 11), but that does NOT add efficiency-vs-peak headroom. "
                    f"Strict measured-floor lift +{strict8:.1f} TPS (M=1 was +{strict1:.1f}) does NOT "
                    f"clear #383 +23.75 robust -> the supply lift does NOT improve at the served width; "
                    f"the strict-tier STRADDLE STANDS. HONEST NEGATIVE (#388 discipline): the served "
                    f"regime does not upgrade the M=1 straddle to a pass.")

    # if GPU ran, qualify the roofline with the measured M=1 BW-bound verdict
    realized_note = (
        "ROOFLINE BOUND: no cb3/QTIP kernel in env (Marlin/AWQ/AQLM only); cb3 bounded by its "
        "nominal 0.785x byte count at int4-Marlin's measured M=1 BW-efficiency. Tight iff cb3 "
        "matches that efficiency -- literature-supported (QTIP/QuIP# L1-resident codebook, RHT "
        "folded offline; only a <<1% online activation Hadamard).")
    if micro is not None:
        agg = micro["aggregate"]
        realized_note += (
            f" Measured count-weighted Marlin M=1 BW-eff = {agg['count_weighted_marlin_bw_eff']:.3f} "
            f"(peak-copy {micro['peak_copy']['peak_copy_gbs']:.1f} GB/s); "
            f"m1_is_bw_bound={agg['m1_is_bw_bound']}. "
            + ("BW-bound => roofline is tight." if agg["m1_is_bw_bound"]
               else "NOT fully BW-bound at M=1 (launch/latency-influenced) => roofline is an "
                    "OPTIMISTIC upper bound; literature-conservative 1.15x is the realized floor."))

    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 391, "base_pr": 388,
        "kind": "cb3-kernel-realized-bw",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # isolation flags (research microbench; NO served change/submission/HF job)
        "no_hf_job": True, "no_launch": True, "no_served_file_change": True,
        "no_kernel_rebuild": True, "analysis_only": True, "official_tps": 0.0,
        # ---- analytic core (M=1 #388 baseline) ----
        "byte_ratio": ap["byte_ratio"],
        "realized_body_speedup": S_roof,
        "realized_body_speedup_qtip_empirical": ap["realized_body_speedup_qtip_empirical"],
        "realized_body_speedup_measured_floor": measured_floor_speedup,
        "realized_is_roofline_bound": True,
        "lit_bracket_speedup": ap["lit_bracket_speedup"],
        "qtip_beta_byte_proportional": ap["qtip_beta_byte_proportional"],
        "f_body_complement_pr": f_comp,
        "f_body_strict_honest": F_BODY_STRICT,
        "translation_variants": ap["translation_variants"],
        "realized_strict_base_lift_tps": headline,
        "roofline_formula_lift_band_upper_bound": [ap["roofline_formula_lift_off_the_shelf"],
                                                   ap["roofline_formula_lift_floor"]],
        "closes_383_supply_gap_floor": gate["closes_383_supply_gap_floor"],
        "supply_gate": gate,
        "supply_gate_strict_floor": supply_gate_strict_floor,
        "activation_hadamard_tax": ap["activation_hadamard_tax"],
        "realized_note": realized_note,
        # ---- #391 served-width headline (M=8 verify / M=4 partial-accept) ----
        "served_verify_width": SERVED_VERIFY_WIDTH, "partial_accept_width": PARTIAL_ACCEPT_WIDTH,
        "mtp_k": MTP_K,
        "width_analysis": width_analysis,
        "marlin_m8_hbm_eff": (m8["marlin_hbm_eff"] if m8 else None),
        "marlin_m4_hbm_eff": (m4["marlin_hbm_eff"] if m4 else None),
        "realized_body_speedup_m8": (m8["realized_body_speedup"] if m8 else None),    # 3 tiers
        "realized_body_speedup_m4": (m4["realized_body_speedup"] if m4 else None),
        "realized_strict_base_lift_tps_m8": (m8["realized_strict_base_lift_tps"] if m8 else None),
        "realized_strict_base_lift_tps_m4": (m4["realized_strict_base_lift_tps"] if m4 else None),
        "realized_strict_base_lift_tps_m8_measured_floor": (
            m8["realized_strict_base_lift_tps_measured_floor"] if m8 else None),
        "closes_383_robust_m8": (m8["closes_383_robust"] if m8 else None),
        "closes_383_robust_m4": (m4["closes_383_robust"] if m4 else None),
        "m8_is_bw_bound": (m8["is_bw_bound"] if m8 else None),
        "m4_is_bw_bound": (m4["is_bw_bound"] if m4 else None),
        "m8_note": m8_note,
        "cb3_m8_microbench_self_test_passes": bool(st["passes"]),
        # ---- inputs (provenance) ----
        "inputs": {
            "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF, "cb3_bpw_uniform": CB3_BPW_UNIFORM,
            "mixed_gate_ppl_372": MIXED_GATE_PPL, "ppl_gate": PPL_GATE,
            "f_attn_378": F_ATTN, "f_lmhead_378": F_LMHEAD, "f_draft_378": F_DRAFT,
            "band_378": [BAND_OFF_THE_SHELF, BAND_FLOOR], "step_norm_us_378": STEP_NORM_US,
            "supply_floor_383": SUPPLY_FLOOR_JOINT_TPS, "supply_robust_383": SUPPLY_ROBUST_ET_ONLY_TPS,
            "mtp_k_52": MTP_K, "served_verify_width": SERVED_VERIFY_WIDTH,
            "m1_measured_hbm_eff_388": M1_MEASURED_HBM_EFF_388,
        },
        "selftest": st,
        "cb3_microbench_self_test_passes": bool(st["passes"]),
    }
    if micro is not None:
        payload["microbench"] = micro
        payload["gpu"] = micro["gpu"]
        payload["peak_mem_mib"] = micro["peak_mem_mib"]
        payload["m1_is_bw_bound"] = micro["aggregate"]["m1_is_bw_bound"]
    return payload


def print_report(p: dict[str, Any]) -> None:
    print("=" * 96)
    print(f"PR #391 lawine -- cb3 kernel REALIZED bandwidth @ SERVED M=8/M=4 verify "
          f"(extends #388 M=1)  ({p['created_at']})")
    if "gpu" in p:
        g = p["gpu"]
        print(f"  GPU {g['name']} sm{g['compute_capability']} x{g['sm_count']} "
              f"(a10g_80sm={g['is_a10g_80sm']})")
    print("-" * 96)
    print(f"  byte_ratio (cb3/int4)      = {p['byte_ratio']:.6f}   (PR target 0.785)")
    print(f"  realized_body_speedup      = {p['realized_body_speedup']:.6f}  ROOFLINE "
          f"(lit bracket {p['lit_bracket_speedup'][0]:.2f}-{p['lit_bracket_speedup'][1]:.2f}x)")
    print(f"  realized_is_roofline_bound = {p['realized_is_roofline_bound']}")
    if "microbench" in p:
        mb = p["microbench"]; agg = mb["aggregate"]
        print(f"  peak-copy HBM BW           = {mb['peak_copy']['peak_copy_gbs']:.1f} GB/s "
              f"({mb['peak_copy']['copy_eff_vs_theoretical']*100:.0f}% of {A10G_HBM_PEAK_GBS:.0f} theo)")
        print(f"  Marlin M=1 BW-eff (cwt)    = {agg['count_weighted_marlin_bw_eff']:.3f}  "
              f"({agg['count_weighted_marlin_eff_gbs']:.1f} GB/s)  m1_is_bw_bound={agg['m1_is_bw_bound']}")
        print(f"  speedup tiers: roofline {agg['roofline_speedup']:.3f}x (BW-bound ceiling) | "
              f"qtip-empirical {p['realized_body_speedup_qtip_empirical']:.3f}x (realistic, beta=0.51) | "
              f"measured-floor {agg['measured_floor_speedup']:.3f}x (strict fixed-overhead)")
        print("  per-shape (M=1 int4-Marlin):")
        for s in mb["per_shape"]:
            print(f"    {s['name']:>8} [{s['out']:>5}x{s['in']:>5}] x{s['count']:<3} "
                  f"{s['marlin_us']:>7.2f}us  {s['marlin_eff_gbs']:>6.1f}GB/s  eff={s['marlin_bw_eff']:.3f}")
        wa = p.get("width_analysis")
        if wa and "by_width" in mb:
            print("-" * 96)
            print("  *** #391 SERVED-WIDTH SWEEP (marlin_m{M}_hbm_eff = weight-read efficiency) ***")
            print(f"    {'M':>3} {'hbm_eff':>8} {'tot_eff':>8} {'bw_bound':>9} | "
                  f"{'roofline':>9} {'qtip':>7} {'floor':>7} | "
                  f"{'strict_lift':>11} {'closes_383':>11}")
            for w_str in sorted(wa.keys(), key=lambda x: int(x)):
                a = wa[w_str]; tiers = a["realized_body_speedup"]
                aggw = mb["by_width"][w_str]["aggregate"]
                print(f"    {a['width']:>3} {a['marlin_hbm_eff']:>8.3f} "
                      f"{aggw['count_weighted_total_bw_eff']:>8.3f} {str(a['is_bw_bound']):>9} | "
                      f"{tiers['roofline']:>9.3f} {tiers['qtip_empirical']:>7.3f} "
                      f"{tiers['measured_floor']:>7.3f} | "
                      f"+{a['realized_strict_base_lift_tps_measured_floor']:>10.1f} "
                      f"{str(a['closes_383_robust']):>11}")
            print("    (strict_lift = measured-floor tier, body-only, off-the-shelf base; "
                  "closes_383 = clears +17.22 floor AND +23.75 robust)")
            if p.get("m8_note"):
                print("  " + "-" * 92)
                for line in _wrap(p["m8_note"], 92):
                    print(f"  {line}")
    print("-" * 96)
    print("  realized_strict_base_lift_tps  (apply speedup to body fraction of #378 band):")
    for k, v in p["translation_variants"].items():
        print(f"    {k:>28}: x{v['lift_factor']:.4f}  band[{v['lifted_off_the_shelf']:.1f}, "
              f"{v['lifted_floor']:.1f}]  delta[+{v['delta_off_the_shelf']:.1f}, +{v['delta_floor']:.1f}]")
    g = p["supply_gate"]
    print("-" * 96)
    print(f"  #383 supply floor = +{g['supply_floor_joint_tps']:.2f} (joint) / "
          f"+{g['supply_robust_et_only_tps']:.2f} (robust)")
    print(f"  REALISTIC realized lift (qtip-empirical, body-only, off-the-shelf base) = "
          f"+{g['conservative_lift_tps']:.2f} TPS")
    print(f"  closes_383_supply_gap_floor = {g['closes_383_supply_gap_floor']}  "
          f"(clears_floor={g['clears_floor']} clears_robust={g['clears_robust']})")
    sf = p.get("supply_gate_strict_floor")
    if sf:
        print(f"  [strict fixed-overhead floor sensitivity] lift = +{sf['conservative_lift_tps']:.2f} TPS "
              f"-> closes_floor={sf['closes_383_supply_gap_floor']} (STRADDLES the floor; thin margin)")
    print(f"  activation Hadamard online tax = {p['activation_hadamard_tax']['hadamard_over_weight_read_frac']*100:.3f}% "
          f"of weight read (negligible)")
    ub = p["roofline_formula_lift_band_upper_bound"]
    print("-" * 96)
    print(f"  [M=1 #388 baseline] realized_strict_base_lift_tps = +{p['realized_strict_base_lift_tps']:.1f} "
          f"TPS (realistic; PR roofline-formula UPPER bound +{ub[0]:.1f}..+{ub[1]:.1f})")
    if p.get("marlin_m8_hbm_eff") is not None:
        t8 = p["realized_body_speedup_m8"]
        print("-" * 96)
        print(f"  *** #391 HEADLINE (SERVED M=8 verify) ***")
        print(f"    marlin_m8_hbm_eff           = {p['marlin_m8_hbm_eff']:.3f}  "
              f"(M=1 was {p['microbench']['by_width']['1']['aggregate']['count_weighted_marlin_bw_eff']:.3f}; "
              f"BW-bound>={M8_BW_BOUND_THRESHOLD}? {p['m8_is_bw_bound']})")
        print(f"    realized_body_speedup_m8    = roofline {t8['roofline']:.3f}x | "
              f"qtip {t8['qtip_empirical']:.3f}x | measured-floor {t8['measured_floor']:.3f}x")
        print(f"    realized_strict_base_lift_tps_m8 = +{p['realized_strict_base_lift_tps_m8']:.1f} TPS "
              f"(qtip realistic) | +{p['realized_strict_base_lift_tps_m8_measured_floor']:.1f} (strict floor)")
        print(f"    closes_383_robust_m8 (STRICT tier) = {p['closes_383_robust_m8']}")
        if p.get("marlin_m4_hbm_eff") is not None:
            print(f"    [M=4 partial-accept] hbm_eff={p['marlin_m4_hbm_eff']:.3f}  "
                  f"strict_lift=+{p['width_analysis']['4']['realized_strict_base_lift_tps_measured_floor']:.1f} "
                  f"closes_383={p['closes_383_robust_m4']}")
    print(f"  self-test: {p['selftest']['n_passed']}/{p['selftest']['n_checks']} "
          f"-> cb3_m8_microbench_self_test_passes = {p['cb3_m8_microbench_self_test_passes']}")
    print("=" * 96)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def maybe_log_wandb(payload: dict[str, Any], args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[cb3-bw] wandb helpers unavailable: {e}")
        return None
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["cb3-kernel-realized-bw", "roofline", "marlin-m8", "served-verify", "qtip-quip",
              "sub-int4-body", "pr-391", "pr-388-followup"],
        config={"pr": 391, "base_pr": 388, "kind": "cb3-kernel-realized-bw",
                "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF, "byte_ratio": byte_ratio(),
                "band_off_the_shelf": BAND_OFF_THE_SHELF, "band_floor": BAND_FLOOR,
                "supply_floor_383": SUPPLY_FLOOR_JOINT_TPS, "official_tps": 0.0,
                "mtp_k": MTP_K, "served_verify_width": SERVED_VERIFY_WIDTH,
                "partial_accept_width": PARTIAL_ACCEPT_WIDTH, "widths": str(DEFAULT_WIDTHS),
                "analysis_only": True, "no_hf_job": True},
    )
    if run is None:
        print("[cb3-bw] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "headline/realized_body_speedup": float(payload["realized_body_speedup"]),
        "headline/realized_body_speedup_qtip_empirical": float(payload["realized_body_speedup_qtip_empirical"]),
        "headline/realized_strict_base_lift_tps": float(payload["realized_strict_base_lift_tps"]),
        "headline/byte_ratio": float(payload["byte_ratio"]),
        "gate/closes_383_supply_gap_floor": float(payload["closes_383_supply_gap_floor"]),
        "gate/clears_floor": float(payload["supply_gate"]["clears_floor"]),
        "gate/clears_robust": float(payload["supply_gate"]["clears_robust"]),
        "gate/realistic_lift_tps": float(payload["supply_gate"]["conservative_lift_tps"]),
        "roofline/is_roofline_bound": float(payload["realized_is_roofline_bound"]),
        "hadamard/tax_frac": float(payload["activation_hadamard_tax"]["hadamard_over_weight_read_frac"]),
        "selftest/passes": float(payload["selftest"]["passes"]),
        "selftest/n_checks": float(payload["selftest"]["n_checks"]),
    }
    if payload.get("supply_gate_strict_floor"):
        sf = payload["supply_gate_strict_floor"]
        flat["gate_strict/closes_floor"] = float(sf["closes_383_supply_gap_floor"])
        flat["gate_strict/lift_tps"] = float(sf["conservative_lift_tps"])
    for k, v in payload["translation_variants"].items():
        flat[f"variant/{k}/lift_factor"] = float(v["lift_factor"])
        flat[f"variant/{k}/delta_off_the_shelf"] = float(v["delta_off_the_shelf"])
        flat[f"variant/{k}/delta_floor"] = float(v["delta_floor"])
    if "microbench" in payload:
        mb = payload["microbench"]; agg = mb["aggregate"]
        flat["gpu/sm_count"] = float(mb["gpu"]["sm_count"])
        flat["bw/peak_copy_gbs"] = float(mb["peak_copy"]["peak_copy_gbs"])
        flat["bw/marlin_cwt_eff_gbs"] = float(agg["count_weighted_marlin_eff_gbs"])
        flat["bw/marlin_cwt_bw_eff"] = float(agg["count_weighted_marlin_bw_eff"])
        flat["bw/m1_is_bw_bound"] = float(agg["m1_is_bw_bound"])
        flat["speedup/roofline"] = float(agg["roofline_speedup"])
        flat["speedup/measured_floor"] = float(agg["measured_floor_speedup"])
        for s in mb["per_shape"]:
            flat[f"shape/{s['name']}/marlin_us"] = float(s["marlin_us"])
            flat[f"shape/{s['name']}/eff_gbs"] = float(s["marlin_eff_gbs"])
            flat[f"shape/{s['name']}/bw_eff"] = float(s["marlin_bw_eff"])
        # ---- #391 per-width served sweep (M=1/M=8/M=4) ----
        for w_str, blk in mb.get("by_width", {}).items():
            aw = blk["aggregate"]
            flat[f"width/m{w_str}/marlin_hbm_eff"] = float(aw["count_weighted_marlin_bw_eff"])
            flat[f"width/m{w_str}/total_bw_eff"] = float(aw["count_weighted_total_bw_eff"])
            flat[f"width/m{w_str}/measured_floor_speedup"] = float(aw["measured_floor_speedup"])
            flat[f"width/m{w_str}/is_bw_bound"] = float(aw["is_bw_bound"])
            for s in blk["per_shape"]:
                flat[f"width/m{w_str}/shape/{s['name']}/us"] = float(s["marlin_us"])
                flat[f"width/m{w_str}/shape/{s['name']}/bw_eff"] = float(s["marlin_bw_eff"])
        wa = payload.get("width_analysis") or {}
        for w_str, a in wa.items():
            t = a["realized_body_speedup"]
            flat[f"m8sweep/m{w_str}/roofline_speedup"] = float(t["roofline"])
            flat[f"m8sweep/m{w_str}/qtip_speedup"] = float(t["qtip_empirical"])
            flat[f"m8sweep/m{w_str}/measured_floor_speedup"] = float(t["measured_floor"])
            flat[f"m8sweep/m{w_str}/strict_lift_tps"] = float(a["realized_strict_base_lift_tps_measured_floor"])
            flat[f"m8sweep/m{w_str}/realistic_lift_tps"] = float(a["realized_strict_base_lift_tps"])
            flat[f"m8sweep/m{w_str}/closes_383_robust"] = float(a["closes_383_robust"])
        # explicit M=8 headline scalars
        if payload.get("marlin_m8_hbm_eff") is not None:
            flat["headline_m8/marlin_m8_hbm_eff"] = float(payload["marlin_m8_hbm_eff"])
            flat["headline_m8/realized_strict_base_lift_tps_m8"] = float(payload["realized_strict_base_lift_tps_m8"])
            flat["headline_m8/strict_floor_lift_tps_m8"] = float(payload["realized_strict_base_lift_tps_m8_measured_floor"])
            flat["headline_m8/closes_383_robust_m8"] = float(payload["closes_383_robust_m8"])
            flat["headline_m8/m8_is_bw_bound"] = float(payload["m8_is_bw_bound"])
        if payload.get("marlin_m4_hbm_eff") is not None:
            flat["headline_m4/marlin_m4_hbm_eff"] = float(payload["marlin_m4_hbm_eff"])
            flat["headline_m4/closes_383_robust_m4"] = float(payload["closes_383_robust_m4"])
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="cb3_kernel_realized_bw", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[cb3-bw] wandb logged {len(flat)} keys (run {rid})")
    return rid


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float, str)) or o is None:
        return o
    return str(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (parity + M=8 checks)")
    ap.add_argument("--gpu", action="store_true", help="run the int4-Marlin microbench + roofline (M=1 always)")
    ap.add_argument("--m8", action="store_true", help="add the served M=8 verify leg (#391 headline)")
    ap.add_argument("--m4", action="store_true", help="add the M=4 partial-accept leg")
    ap.add_argument("--widths", type=str, default=None,
                    help="comma-separated verify widths to bench (overrides --m8/--m4; M=1 always added)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast GPU run to validate the path")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/cb3-m8-verify-body-speedup")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="cb3-m8-verify-body-speedup")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    st = self_test()

    # 0-GPU self-test path: no wandb, no SENPAI-RESULT (this is the reproduce gate).
    if args.self_test and not args.gpu:
        payload = build_payload(args, None, st)
        print_report(payload)
        out_path = Path(args.out_dir) / "cb3_kernel_realized_bw_selftest.json"
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        print(f"\n[cb3-bw] wrote {out_path}")
        print(f"\ncb3_microbench_self_test_passes = {st['passes']}")
        sys.exit(0 if st["passes"] else 1)

    if args.smoke:
        args.iters = min(args.iters, 20)
        args.warmup = min(args.warmup, 5)

    # Resolve verify widths: M=1 ALWAYS (the #388 baseline anchor + the QTIP eff_1 denominator);
    # --m8/--m4 add the served legs; --widths overrides. Dedup, keep M=1 first.
    if args.widths:
        widths = [int(x) for x in args.widths.split(",") if x.strip()]
    else:
        widths = []
        if args.m8:
            widths.append(SERVED_VERIFY_WIDTH)
        if args.m4:
            widths.append(PARTIAL_ACCEPT_WIDTH)
    widths = [1] + [w for w in widths if w != 1]
    seen: set[int] = set()
    widths = [w for w in widths if not (w in seen or seen.add(w))]

    micro = gpu_microbench(widths, args.iters, args.warmup)
    payload = build_payload(args, micro, st)
    print_report(payload)

    out_path = Path(args.out_dir) / "cb3_kernel_realized_bw_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[cb3-bw] wrote {out_path}")

    rid = None if args.smoke else maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    def _f(key):
        v = payload.get(key)
        return float(v) if v is not None else None

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        # ---- M=1 #388 baseline (unchanged) ----
        "realized_body_speedup": float(payload["realized_body_speedup"]),
        "realized_body_speedup_qtip_empirical": float(payload["realized_body_speedup_qtip_empirical"]),
        "realized_is_roofline_bound": bool(payload["realized_is_roofline_bound"]),
        "realized_strict_base_lift_tps": float(payload["realized_strict_base_lift_tps"]),
        "m1_is_bw_bound": bool(payload.get("m1_is_bw_bound", False)),
        # ---- #391 served-width headline (M=8 verify / M=4 partial-accept) ----
        "marlin_m8_hbm_eff": _f("marlin_m8_hbm_eff"),
        "marlin_m4_hbm_eff": _f("marlin_m4_hbm_eff"),
        "realized_body_speedup_m8": (payload["realized_body_speedup_m8"]
                                     if payload.get("realized_body_speedup_m8") else None),
        "realized_strict_base_lift_tps_m8": _f("realized_strict_base_lift_tps_m8"),
        "realized_strict_base_lift_tps_m8_measured_floor": _f("realized_strict_base_lift_tps_m8_measured_floor"),
        "closes_383_robust_m8": (bool(payload["closes_383_robust_m8"])
                                 if payload.get("closes_383_robust_m8") is not None else None),
        "closes_383_robust_m4": (bool(payload["closes_383_robust_m4"])
                                 if payload.get("closes_383_robust_m4") is not None else None),
        "m8_is_bw_bound": (bool(payload["m8_is_bw_bound"])
                           if payload.get("m8_is_bw_bound") is not None else None),
        "cb3_m8_microbench_self_test_passes": bool(st["passes"]),
        "official_tps": 0.0, "no_hf_job": True,
        "primary_metric": {"name": "marlin_m8_hbm_eff",
                           "value": _f("marlin_m8_hbm_eff")
                           if payload.get("marlin_m8_hbm_eff") is not None
                           else float(payload["realized_body_speedup"])},
        "test_metric": {"name": "realized_strict_base_lift_tps_m8",
                        "value": _f("realized_strict_base_lift_tps_m8")
                        if payload.get("realized_strict_base_lift_tps_m8") is not None
                        else float(payload["realized_strict_base_lift_tps"])},
    }))


if __name__ == "__main__":
    main()
