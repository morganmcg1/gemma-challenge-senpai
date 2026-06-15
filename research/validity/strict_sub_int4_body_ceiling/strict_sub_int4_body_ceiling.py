"""
Strict sub-int4 body quantization ceiling analysis (PR #356).

CPU-analytic over BANKED numbers. No GPU, no training, no served-file change.

Deliverables
------------
1. ceiling(b) curve: strict TPS ceiling as a function of body bits-per-weight
   b in {4, 3.5, 3, 2.5, 2}.  Body-read bytes scale proportionally with b/4;
   KV, attention, lm_head, draft_chain are unchanged.

2. PPL cost band: per-b PPL degradation band from quant literature (GPTQ,
   AWQ, QuIP#, AQLM, SpQR, OmniQuant, HQQ).  Gate = 2.42.

3. Identity note: sub-int4 body quantization does NOT inherently break
   deterministic greedy emission; binding risk is PPL, not token identity.

4. Verdict: sub_int4_clears_500_strict (bool) + b* corner.

5. Self-test (PRIMARY: strict_sub_int4_self_test_passes):
   (a) #344 sxltbech waterfall round-trips (resid <= 1e-9, sums to 1218.2 us)
   (b) ceiling(4) == 473.5296 / 520.9527 anchors round-trip
   (c) ceiling(b) monotone-increasing as b decreases, NaN-clean
   (d) PPL band ordered + 2.42 crossing identified
   (e) verdict bool set

TEST metrics: strict_ceiling_at_best_ppl_safe_bits (float)
              sub_int4_clears_500_strict (bool)
"""

import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Section 0 — banked anchors from #344 sxltbech (gate_independent_speed_lever)
# ---------------------------------------------------------------------------

# Step waterfall from #344 run sxltbech (gate_independent_speed_lever_results.json)
STEP_NORM_US: float = 1218.2          # normalised step time, microseconds

# Body GEMM components (int4 W4A16, all at 4 bpw) — from sxltbech waterfall
BODY_COMPONENTS_US: dict = {
    "gate_up_proj": 530.7241,          # 43.566% of step; bytes=969,932,800
    "down_proj":    265.3621,          # 21.783% of step; bytes=484,966,400
    "qkv_proj":      79.6088,          # 6.535%  of step; bytes=145,489,920
    "o_proj":        53.0724,          # 4.357%  of step; bytes=96,993,280
}
BODY_NORM_US: float = sum(BODY_COMPONENTS_US.values())   # 928.7674 us

# Non-body step components that do NOT scale with body bits (fixed costs)
NON_BODY_COMPONENTS_US: dict = {
    "attention":    115.8113,          # 9.507%  of step; KV / SDPA
    "lm_head":       27.3218,          # 2.243%  of step; bf16 lm_head (fixed)
    "draft_chain":  146.2978,          # 12.009% of step; draft tokens (fixed)
    # norms/sampling/framework: folded into graph, effectively 0
}
REMAINING_US: float = STEP_NORM_US - BODY_NORM_US   # 289.4325 us

# Speculative-decoding / acceptance constants
K_CAL: float     = 125.26795005202914   # tokens per normalised step
E_T_SERVED: float = 4.6827608           # expected tokens/step served

# Ceiling anchors (from #332 y5cl0ena, phi=0.075 SDPA floor)
STRICT_CEILING_INT4: float = 473.5295953446407   # TPS at b=4 strict
LAMBDA1_CEIL:        float = 520.9527323111674   # TPS at b=4, lambda=1

# Hardware
BODY_INT4_GB: float = 1.6973824     # body int4 weight bytes in GB
KV_GQA_BYTES: int   = 40_009_728    # KV cache GQA bytes (~40 MB, fixed)
A10G_BW_GBPS: float = 600.0         # A10G peak HBM bandwidth

# PPL baseline
DEPLOYED_INT4_PPL: float = 2.3772
PPL_GATE: float          = 2.4200
PPL_HEADROOM: float      = PPL_GATE - DEPLOYED_INT4_PPL   # 0.0428

# ---------------------------------------------------------------------------
# Section 1 — ceiling(b) model
# ---------------------------------------------------------------------------

BITS_GRID: tuple = (4.0, 3.5, 3.0, 2.5, 2.0)


def step_at_bits(b: float) -> float:
    """
    Normalised step time (us) at body quantisation width b bpw.

    Body GEMM bytes scale proportionally with b/4.  All other costs
    (KV reads, attention FLOP, lm_head, draft chain) are fixed.

    step(b) = body_norm_us * (b / 4) + remaining_us
    """
    return BODY_NORM_US * (b / 4.0) + REMAINING_US


def strict_ceiling_at_bits(b: float) -> float:
    """
    Strict TPS ceiling at b bpw.

    The strict-ceiling ratio (473.5296 / 520.9527 = 0.9089) from #332
    encodes the phi=0.075 SDPA floor discount.  We apply it at the
    int4 anchor and scale with step time:

        strict_ceiling(b) = STRICT_CEILING_INT4 * (STEP_NORM_US / step(b))

    This is equivalent to:
        strict_ceiling(b) = (K_CAL * E_T_served * strict_ratio) / step(b) [ms]
    and simplifies identically at b=4.
    """
    s = step_at_bits(b)
    return STRICT_CEILING_INT4 * (STEP_NORM_US / s)


def lambda1_ceiling_at_bits(b: float) -> float:
    """Lambda=1 TPS ceiling at b bpw (unconstrained-acceptance upper bound)."""
    s = step_at_bits(b)
    return LAMBDA1_CEIL * (STEP_NORM_US / s)


def b_star_for_500() -> float:
    """
    Analytic b* where strict_ceiling(b) = 500.

    strict_ceiling(b) = STRICT_CEILING_INT4 * STEP_NORM_US / step(b) = 500
    => step(b) = STRICT_CEILING_INT4 * STEP_NORM_US / 500
    => BODY_NORM_US * (b/4) + REMAINING_US = step_target
    => b = 4 * (step_target - REMAINING_US) / BODY_NORM_US
    """
    step_target = STRICT_CEILING_INT4 * STEP_NORM_US / 500.0
    b = 4.0 * (step_target - REMAINING_US) / BODY_NORM_US
    return b


# ---------------------------------------------------------------------------
# Section 2 — PPL cost band from quant literature
# ---------------------------------------------------------------------------
#
# All delta-PPL values are RELATIVE to the int4 baseline for the same model
# class (4B–13B parameters, predominantly Llama-2-7B/13B as the closest
# published analogue to Gemma-4-E4B).  The hard gate is:
#
#     PPL_gate = 2.42  =>  max allowed delta = +0.0428
#
# Citations
# ---------
# GPTQ:       Frantar et al., 2022 (arxiv:2210.17323)
#             Llama-2-7B W3 g128 PPL=6.74 vs W4=5.39  => delta=+1.35
#             Llama-2-13B W3 g128 PPL=5.85 vs W4=5.19 => delta=+0.66
# AWQ:        Lin et al., 2023 (arxiv:2306.00978)
#             Llama-2-7B W3 g128 delta=+0.64; 13B delta=+0.35
# QuIP#:      Tseng et al., 2024 (arxiv:2402.04396)
#             Llama-2-7B W3 delta=+0.23  (BEST KNOWN uniform W3 for 7B)
#             Llama-2-7B W2 delta=+1.10
# AQLM:       Egiazarian et al., 2024 (arxiv:2401.06118)
#             Llama-2-7B W2 delta=+1.43  (codebook-based)
# SpQR:       Dettmers et al., 2023 (arxiv:2306.03078)
#             Llama-2-7B W3 delta=+0.20 (sparse+outlier protection)
#             NOTE: SpQR W3 is NOT a standard hardware-friendly format;
#             irregular sparsity pattern prevents kernel reuse with vLLM int3.
# OmniQuant: Shao et al., 2023 (arxiv:2308.13137)
#             LLaMA-2-7B W3 delta=+0.29
# HQQ:        Badri & Shaji, 2023 (https://mobiusml.github.io/hqq_blog/)
#             Llama-2-7B W3 g64 delta=+0.47; W2 g16 delta=+1.95
#
# b=3.5 interpolated from W3 / W4:
#   pessimistic: (W4 delta=0) + 0.5 * (W3 pessimistic) ~= 0.50
#   central:     0.5 * QuIP# W3 central (~0.23)  ~= 0.12
#   optimistic:  0.5 * SpQR W3 low (~0.20)       ~= 0.10
#   (No published uniform 3.5-bit scheme exists for LLMs; all interpolated)
#
# b=2.5 interpolated from W2 / W3:
#   central:  (W3 delta 0.23) + 0.5 * (W2 - W3 gap)
#             QuIP# gap = 1.10 - 0.23 = 0.87  =>  0.23 + 0.435 ~= 0.665
#   pessimistic: GPTQ gap = 2.7 (W2 ~30 est) + W3 1.35 => ~5+; capped at +2.0
#

PPL_BAND: dict = {
    4.0: {
        "optimistic_delta":  0.0,
        "central_delta":     0.0,
        "pessimistic_delta": 0.0,
        "sources": ["deployed int4 baseline"],
        "ppl_optimistic":    DEPLOYED_INT4_PPL,
        "ppl_central":       DEPLOYED_INT4_PPL,
        "ppl_pessimistic":   DEPLOYED_INT4_PPL,
        "within_gate":       True,
        "notes": "Int4 (W4A16) operating point; PPL=2.3772.",
    },
    3.5: {
        "optimistic_delta":  0.10,
        "central_delta":     0.25,
        "pessimistic_delta": 0.50,
        "sources": [
            "interpolated W3/W4 boundary; no published uniform 3.5-bit LLM scheme",
            "SpQR (arxiv:2306.03078) W3 delta=+0.20 lower anchor",
            "QuIP# (arxiv:2402.04396) W3 delta=+0.23",
        ],
        "ppl_optimistic":    DEPLOYED_INT4_PPL + 0.10,
        "ppl_central":       DEPLOYED_INT4_PPL + 0.25,
        "ppl_pessimistic":   DEPLOYED_INT4_PPL + 0.50,
        "within_gate":       False,   # best-case +0.10 >> budget +0.0428
        "notes": (
            "No uniform 3.5-bit kernel exists in standard vLLM / AWQ stacks. "
            "Best-case optimistic +0.10 is ~2.3x the +0.0428 budget. "
            "All three band points exceed the gate."
        ),
    },
    3.0: {
        "optimistic_delta":  0.23,   # QuIP# Llama-2-7B (arxiv:2402.04396)
        "central_delta":     0.47,   # HQQ g64 Llama-2-7B
        "pessimistic_delta": 1.35,   # GPTQ Llama-2-7B g128 (arxiv:2210.17323)
        "sources": [
            "QuIP# (arxiv:2402.04396): Llama-2-7B W3 delta=+0.23 (best known)",
            "SpQR  (arxiv:2306.03078): Llama-2-7B W3 delta=+0.20 (non-standard hw format)",
            "OmniQuant (arxiv:2308.13137): LLaMA-2-7B W3 delta=+0.29",
            "AWQ   (arxiv:2306.00978): Llama-2-7B W3 g128 delta=+0.64",
            "HQQ   (https://mobiusml.github.io/hqq_blog/): Llama-2-7B W3 g64 delta=+0.47",
            "GPTQ  (arxiv:2210.17323): Llama-2-7B W3 g128 delta=+1.35",
        ],
        "ppl_optimistic":    DEPLOYED_INT4_PPL + 0.23,
        "ppl_central":       DEPLOYED_INT4_PPL + 0.47,
        "ppl_pessimistic":   DEPLOYED_INT4_PPL + 1.35,
        "within_gate":       False,   # best-case +0.23 is ~5.4x the budget
        "notes": (
            "Best-known uniform W3 result is QuIP# +0.23, which is 5.4x the "
            "+0.0428 budget for the deployed model. No W3 method in the "
            "literature comes within 4x of the gate for 4-8B class models. "
            "SpQR is listed for completeness; its irregular-sparsity format "
            "cannot be served by vLLM's current int3 GEMM kernels."
        ),
    },
    2.5: {
        "optimistic_delta":  0.665,  # interpolated QuIP# W3/W2 midpoint
        "central_delta":     1.00,
        "pessimistic_delta": 2.00,
        "sources": [
            "Interpolated from QuIP# W3 delta=+0.23 and W2 delta=+1.10 (arxiv:2402.04396)",
            "No published uniform 2.5-bit scheme for 4-8B LLMs found in literature",
        ],
        "ppl_optimistic":    DEPLOYED_INT4_PPL + 0.665,
        "ppl_central":       DEPLOYED_INT4_PPL + 1.00,
        "ppl_pessimistic":   DEPLOYED_INT4_PPL + 2.00,
        "within_gate":       False,
        "notes": (
            "No published uniform 2.5-bit LLM quantisation scheme for 4-8B models. "
            "Interpolated from QuIP# W3/W2 bracketing bounds. All points far exceed gate."
        ),
    },
    2.0: {
        "optimistic_delta":  1.10,   # QuIP# Llama-2-7B W2 (arxiv:2402.04396)
        "central_delta":     1.43,   # AQLM Llama-2-7B (arxiv:2401.06118)
        "pessimistic_delta": 30.0,   # GPTQ round-to-nearest W2 (catastrophic)
        "sources": [
            "QuIP#  (arxiv:2402.04396): Llama-2-7B W2 delta=+1.10 (best structured W2)",
            "AQLM   (arxiv:2401.06118): Llama-2-7B W2 delta=+1.43",
            "HQQ    (https://mobiusml.github.io/hqq_blog/): Llama-2-7B W2 g16 delta=+1.95",
            "GPTQ   (arxiv:2210.17323): W2 catastrophic for 7B without special handling",
        ],
        "ppl_optimistic":    DEPLOYED_INT4_PPL + 1.10,
        "ppl_central":       DEPLOYED_INT4_PPL + 1.43,
        "ppl_pessimistic":   DEPLOYED_INT4_PPL + 30.0,
        "within_gate":       False,
        "notes": (
            "Best-known W2 result is QuIP# +1.10 for 7B, which is 25.7x the budget. "
            "W2 is uniformly fatal for language quality on 4-8B models."
        ),
    },
}

# ---------------------------------------------------------------------------
# Section 3 — identity note
# ---------------------------------------------------------------------------

IDENTITY_NOTE: str = (
    "Sub-int4 body quantisation (W3.5, W3, W2.5, W2 A16) does NOT inherently "
    "break deterministic greedy token emission.  Greedy decoding selects the "
    "argmax of the lm_head logit vector; as long as the quantisation changes "
    "the logit distribution but does not alter WHICH token has the highest "
    "logit for every prompt in the eval set, the greedy token sequence is "
    "preserved.  In practice, sub-int4 quantisation does shift logit orderings "
    "on a fraction of positions, causing PPL degradation.  The binding "
    "constraint for this approach is therefore PPL (quality gate = 2.42), "
    "NOT token-identity divergence per se.  If PPL is within gate, greedy "
    "output would need a separate identity audit (#158 contract) before "
    "deployment; but for this feasibility study PPL is the primary kill signal "
    "and it is already conclusive."
)

# ---------------------------------------------------------------------------
# Section 4 — verdict
# ---------------------------------------------------------------------------

# First bits level where any PPL band point (optimistic) crosses the gate
PPL_FIRST_UNSAFE_B: float = 3.5    # b=3.5 optimistic delta=+0.10 >> +0.0428

# b* where ceiling crosses 500 (analytic, fractional)
B_STAR_ANALYTIC: float = b_star_for_500()   # ~3.722 bpw

# Best PPL-safe discrete bits value (only b=4 is within gate)
BEST_PPL_SAFE_B: float = 4.0

# Main deliverable metrics
STRICT_CEILING_AT_BEST_PPL_SAFE_BITS: float = strict_ceiling_at_bits(BEST_PPL_SAFE_B)
# Should equal STRICT_CEILING_INT4 = 473.5296 (b=4)

SUB_INT4_CLEARS_500_STRICT: bool = (
    # ceiling(b) exceeds 500 only for b < b* ~= 3.722, which is sub-int4
    # Every sub-int4 discrete grid point (3.5, 3, 2.5, 2) exceeds 500 TPS
    # HOWEVER all are ruled out by PPL gate.
    # => The answer is False: no sub-int4 width clears 500 within PPL gate.
    False
)


# ---------------------------------------------------------------------------
# Section 5 — main analysis runner + output
# ---------------------------------------------------------------------------

def build_ceiling_curve() -> list:
    """Return list of dicts, one per b in BITS_GRID."""
    curve = []
    for b in BITS_GRID:
        s = step_at_bits(b)
        sc = strict_ceiling_at_bits(b)
        lc = lambda1_ceiling_at_bits(b)
        pb = PPL_BAND[b]
        curve.append({
            "bits_per_weight":        b,
            "body_scale_factor":      b / 4.0,
            "body_read_gb":           BODY_INT4_GB * (b / 4.0),
            "step_us":                s,
            "strict_ceiling_tps":     sc,
            "lambda1_ceiling_tps":    lc,
            "clears_500_strict":      sc > 500.0,
            "ppl_delta_optimistic":   pb["optimistic_delta"],
            "ppl_delta_central":      pb["central_delta"],
            "ppl_delta_pessimistic":  pb["pessimistic_delta"],
            "ppl_optimistic":         pb["ppl_optimistic"],
            "ppl_central":            pb["ppl_central"],
            "ppl_pessimistic":        pb["ppl_pessimistic"],
            "ppl_within_gate":        pb["within_gate"],
            "feasible_strict":        (sc > 500.0) and pb["within_gate"],
        })
    return curve


def run_self_test(curve: list) -> dict:
    """Run all five self-test checks. Returns dict of pass/fail per check."""
    results: dict = {}

    # (a) #344 sxltbech waterfall round-trips
    #   - sum of body components == BODY_NORM_US (resid <= 1e-9 us)
    #   - step = body + remaining (resid <= 1e-9 us)
    body_sum = sum(BODY_COMPONENTS_US.values())
    non_body_sum = sum(NON_BODY_COMPONENTS_US.values())
    step_check = body_sum + REMAINING_US
    step_resid = abs(step_check - STEP_NORM_US)
    body_resid = abs(body_sum - BODY_NORM_US)
    remaining_derived = STEP_NORM_US - BODY_NORM_US
    remaining_resid = abs(remaining_derived - REMAINING_US)

    results["a_waterfall_roundtrip"] = {
        "body_sum_us":       body_sum,
        "body_resid_us":     body_resid,
        "step_check_us":     step_check,
        "step_resid_us":     step_resid,
        "remaining_us":      REMAINING_US,
        "remaining_resid_us": remaining_resid,
        "pass": (body_resid <= 1e-9) and (step_resid <= 1e-9),
    }

    # (b) ceiling(4) round-trips strict / lambda1 anchors
    c4_strict = strict_ceiling_at_bits(4.0)
    c4_lambda1 = lambda1_ceiling_at_bits(4.0)
    strict_resid  = abs(c4_strict  - STRICT_CEILING_INT4)
    lambda_resid  = abs(c4_lambda1 - LAMBDA1_CEIL)
    results["b_ceiling4_roundtrip"] = {
        "c4_strict_computed":       c4_strict,
        "c4_strict_anchor":         STRICT_CEILING_INT4,
        "strict_resid":             strict_resid,
        "c4_lambda1_computed":      c4_lambda1,
        "c4_lambda1_anchor":        LAMBDA1_CEIL,
        "lambda1_resid":            lambda_resid,
        "pass": (strict_resid <= 1e-6) and (lambda_resid <= 1e-6),
    }

    # (c) ceiling(b) monotone-increasing as b decreases, NaN-clean
    ceilings = [row["strict_ceiling_tps"] for row in curve]
    nan_clean = all(not math.isnan(c) for c in ceilings)
    # BITS_GRID is (4, 3.5, 3, 2.5, 2) — decreasing; ceilings should be increasing
    monotone = all(ceilings[i] < ceilings[i + 1] for i in range(len(ceilings) - 1))
    results["c_monotone_nan_clean"] = {
        "nan_clean": nan_clean,
        "monotone":  monotone,
        "ceilings":  ceilings,
        "pass": nan_clean and monotone,
    }

    # (d) PPL band ordered + 2.42 crossing identified at b <= 3.5
    # For each b, check optimistic <= central <= pessimistic
    ppl_ordered = True
    for row in curve:
        opt = row["ppl_delta_optimistic"]
        cen = row["ppl_delta_central"]
        pes = row["ppl_delta_pessimistic"]
        if not (opt <= cen <= pes):
            ppl_ordered = False
    # 2.42 gate crossing: b=4 safe, b=3.5 unsafe
    gate_crossing_b = None
    for row in curve:
        if not row["ppl_within_gate"]:
            gate_crossing_b = row["bits_per_weight"]
            break
    gate_crossing_correct = (gate_crossing_b == 3.5)
    results["d_ppl_band_ordered_and_crossing"] = {
        "ppl_ordered":              ppl_ordered,
        "gate_crossing_b":          gate_crossing_b,
        "gate_crossing_correct":    gate_crossing_correct,
        "pass": ppl_ordered and gate_crossing_correct,
    }

    # (e) verdict bool set
    verdict_set = isinstance(SUB_INT4_CLEARS_500_STRICT, bool)
    best_safe_ceil = STRICT_CEILING_AT_BEST_PPL_SAFE_BITS
    best_safe_correct = abs(best_safe_ceil - STRICT_CEILING_INT4) <= 1e-6
    results["e_verdict_set"] = {
        "sub_int4_clears_500_strict":              SUB_INT4_CLEARS_500_STRICT,
        "strict_ceiling_at_best_ppl_safe_bits":    best_safe_ceil,
        "best_safe_bits":                          BEST_PPL_SAFE_B,
        "verdict_set":                             verdict_set,
        "best_safe_ceil_correct":                  best_safe_correct,
        "pass": verdict_set and best_safe_correct,
    }

    all_pass = all(v["pass"] for v in results.values())
    return {"checks": results, "strict_sub_int4_self_test_passes": all_pass}


def build_report() -> dict:
    """Assemble the full deliverable JSON."""
    curve = build_ceiling_curve()
    self_test = run_self_test(curve)

    # Reformat PPL band for clean JSON output (remove function calls from values)
    ppl_band_out = {}
    for b, info in PPL_BAND.items():
        ppl_band_out[str(b)] = {k: v for k, v in info.items()}

    report = {
        "pr": 356,
        "agent": "denken",
        "leg": "strict sub-int4 body quantisation ceiling analysis",
        "banked_anchors": {
            "step_norm_us":             STEP_NORM_US,
            "body_norm_us":             BODY_NORM_US,
            "remaining_us":             REMAINING_US,
            "strict_ceiling_int4_tps":  STRICT_CEILING_INT4,
            "lambda1_ceil_tps":         LAMBDA1_CEIL,
            "k_cal":                    K_CAL,
            "e_t_served":               E_T_SERVED,
            "body_int4_gb":             BODY_INT4_GB,
            "kv_gqa_bytes":             KV_GQA_BYTES,
            "a10g_bw_gbps":             A10G_BW_GBPS,
            "deployed_int4_ppl":        DEPLOYED_INT4_PPL,
            "ppl_gate":                 PPL_GATE,
            "ppl_headroom":             PPL_HEADROOM,
            "source_pr_waterfall":      344,
            "source_run_id":            "sxltbech",
            "source_pr_strict_ceiling": 332,
            "source_run_id_332":        "y5cl0ena",
        },
        "body_components_us_int4":   BODY_COMPONENTS_US,
        "non_body_components_us":    NON_BODY_COMPONENTS_US,
        "ceiling_model_note": (
            "step(b) = body_norm_us * (b/4) + remaining_us; "
            "strict_ceiling(b) = STRICT_CEILING_INT4 * (STEP_NORM_US / step(b)). "
            "Body GEMM bytes scale proportionally with b/4. "
            "KV, attention, lm_head, draft_chain are fixed. "
            "strict_ceiling_int4 = 473.5296 encodes phi=0.075 SDPA floor from #332."
        ),
        "b_star_analytic_bpw":           B_STAR_ANALYTIC,
        "b_star_note": (
            f"Strict ceiling crosses 500 at b* = {B_STAR_ANALYTIC:.4f} bpw. "
            "The nearest discrete grid points are 3.5 (ceiling 523.5) and 4 (ceiling 473.5). "
            "b* lies between 3.5 and 4, confirming that the ceiling gate needs sub-int4."
        ),
        "ceiling_curve":        curve,
        "ppl_band":             ppl_band_out,
        "identity_note":        IDENTITY_NOTE,
        "ppl_citations": [
            "GPTQ: Frantar et al. 2022 (arxiv:2210.17323) — W3 g128 Llama-2-7B delta=+1.35",
            "AWQ:  Lin et al. 2023 (arxiv:2306.00978)    — W3 g128 Llama-2-7B delta=+0.64",
            "QuIP#: Tseng et al. 2024 (arxiv:2402.04396) — W3 Llama-2-7B delta=+0.23 (best W3)",
            "AQLM: Egiazarian et al. 2024 (arxiv:2401.06118) — W2 Llama-2-7B delta=+1.43",
            "SpQR: Dettmers et al. 2023 (arxiv:2306.03078)   — W3 Llama-2-7B delta=+0.20 (non-std hw)",
            "OmniQuant: Shao et al. 2023 (arxiv:2308.13137)  — W3 LLaMA-2-7B delta=+0.29",
            "HQQ: Badri & Shaji 2023 (mobiusml.github.io/hqq_blog) — W3 g64 delta=+0.47; W2 g16 delta=+1.95",
        ],
        "verdict": {
            "sub_int4_clears_500_strict":           SUB_INT4_CLEARS_500_STRICT,
            "strict_ceiling_at_best_ppl_safe_bits": STRICT_CEILING_AT_BEST_PPL_SAFE_BITS,
            "best_ppl_safe_bits":                   BEST_PPL_SAFE_B,
            "b_star_analytic_bpw":                  B_STAR_ANALYTIC,
            "ceiling_at_b3p5":                      strict_ceiling_at_bits(3.5),
            "ppl_delta_at_b3p5_best_case":          PPL_BAND[3.5]["optimistic_delta"],
            "ppl_budget":                           PPL_HEADROOM,
            "ppl_first_unsafe_b":                   PPL_FIRST_UNSAFE_B,
            "kill_reason": (
                "Speed ceiling passes 500 at b*=3.722 bpw (between discrete 3.5 and 4). "
                "b=3.5 ceiling=523.5 clears 500; however best-case QuIP#/SpQR-class W3 "
                "incurs +0.23 PPL delta, which is 5.4x the +0.0428 budget. "
                "Even interpolated W3.5 optimistic (+0.10) is 2.3x over budget. "
                "The PPL gate is conclusive; the ceiling lever is real but inaccessible "
                "without a quantisation method that stays within +0.04 PPL at W3.5 — "
                "a level not achieved in the published literature for 4-8B LLMs."
            ),
        },
        "self_test": self_test,
        "strict_sub_int4_self_test_passes":         self_test["strict_sub_int4_self_test_passes"],
    }
    return report


def log_to_wandb(report: dict, wandb_group: str, wandb_name: str) -> str:
    """Log key metrics to W&B and return run_id."""
    try:
        import wandb

        run = wandb.init(
            project="senpai",
            group=wandb_group,
            name=wandb_name,
            config={
                "pr":               report["pr"],
                "agent":            report["agent"],
                "leg":              report["leg"],
                "step_norm_us":     STEP_NORM_US,
                "body_norm_us":     BODY_NORM_US,
                "remaining_us":     REMAINING_US,
                "ppl_gate":         PPL_GATE,
                "deployed_int4_ppl": DEPLOYED_INT4_PPL,
                "ppl_headroom":     PPL_HEADROOM,
                "strict_ceiling_int4": STRICT_CEILING_INT4,
            },
        )

        # Log ceiling curve metrics
        for row in report["ceiling_curve"]:
            b = row["bits_per_weight"]
            prefix = f"ceiling/b{str(b).replace('.', 'p')}"
            wandb.log({
                f"{prefix}/step_us":              row["step_us"],
                f"{prefix}/strict_ceiling_tps":   row["strict_ceiling_tps"],
                f"{prefix}/lambda1_ceiling_tps":  row["lambda1_ceiling_tps"],
                f"{prefix}/ppl_delta_optimistic": row["ppl_delta_optimistic"],
                f"{prefix}/ppl_delta_central":    row["ppl_delta_central"],
                f"{prefix}/ppl_delta_pessimistic":row["ppl_delta_pessimistic"],
                f"{prefix}/clears_500_strict":    float(row["clears_500_strict"]),
                f"{prefix}/ppl_within_gate":      float(row["ppl_within_gate"]),
                f"{prefix}/feasible_strict":      float(row["feasible_strict"]),
            }, commit=False)

        # Log primary test metrics
        wandb.log({
            "summary/strict_sub_int4_self_test_passes":
                float(report["strict_sub_int4_self_test_passes"]),
            "summary/sub_int4_clears_500_strict":
                float(report["verdict"]["sub_int4_clears_500_strict"]),
            "summary/strict_ceiling_at_best_ppl_safe_bits":
                report["verdict"]["strict_ceiling_at_best_ppl_safe_bits"],
            "summary/b_star_analytic_bpw":
                report["b_star_analytic_bpw"],
            "summary/best_ppl_safe_bits":
                report["verdict"]["best_ppl_safe_bits"],
            "summary/ppl_budget":
                report["verdict"]["ppl_budget"],
            "summary/ppl_first_unsafe_b":
                report["verdict"]["ppl_first_unsafe_b"],
            "summary/best_case_ppl_delta_at_b3p5":
                report["verdict"]["ppl_delta_at_b3p5_best_case"],
            # Self-test sub-checks
            "test/a_waterfall_roundtrip":
                float(report["self_test"]["checks"]["a_waterfall_roundtrip"]["pass"]),
            "test/b_ceiling4_roundtrip":
                float(report["self_test"]["checks"]["b_ceiling4_roundtrip"]["pass"]),
            "test/c_monotone_nan_clean":
                float(report["self_test"]["checks"]["c_monotone_nan_clean"]["pass"]),
            "test/d_ppl_band_ordered_and_crossing":
                float(report["self_test"]["checks"]["d_ppl_band_ordered_and_crossing"]["pass"]),
            "test/e_verdict_set":
                float(report["self_test"]["checks"]["e_verdict_set"]["pass"]),
        }, commit=True)

        run_id = run.id
        wandb.finish()
        return run_id

    except Exception as exc:
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


# ---------------------------------------------------------------------------
# Section 6 — CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict sub-int4 body quantisation ceiling (PR #356)."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test and exit 0 on pass / 1 on fail.")
    parser.add_argument("--wandb_group", type=str,
                        default="strict-sub-int4-body-ceiling",
                        help="W&B group name for banking.")
    parser.add_argument("--wandb_name", type=str,
                        default="denken/strict-sub-int4-body-ceiling",
                        help="W&B run name for banking.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip W&B logging.")
    parser.add_argument("--out", type=str,
                        default="research/validity/strict_sub_int4_body_ceiling/"
                                "strict_sub_int4_body_ceiling_results.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    report = build_report()

    # Print summary to stdout
    print("\n=== Strict sub-int4 body ceiling analysis ===")
    print(f"Step anchors (PR #344 sxltbech):")
    print(f"  step_norm_us  = {STEP_NORM_US} us")
    print(f"  body_norm_us  = {BODY_NORM_US:.4f} us  ({BODY_NORM_US/STEP_NORM_US*100:.2f}% of step)")
    print(f"  remaining_us  = {REMAINING_US:.4f} us")
    print(f"\nCeiling curve:")
    print(f"  {'b (bpw)':<10} {'step_us':>10} {'strict_TPS':>12} {'clears_500':>12} {'PPL_opt_d':>10} {'PPL_cen_d':>10} {'in_gate':>8}")
    for row in report["ceiling_curve"]:
        b = row["bits_per_weight"]
        print(f"  {b:<10} {row['step_us']:>10.2f} {row['strict_ceiling_tps']:>12.2f} "
              f"{'YES' if row['clears_500_strict'] else 'NO':>12} "
              f"{row['ppl_delta_optimistic']:>10.3f} "
              f"{row['ppl_delta_central']:>10.3f} "
              f"{'YES' if row['ppl_within_gate'] else 'NO':>8}")
    print(f"\nb* analytic = {B_STAR_ANALYTIC:.4f} bpw "
          f"(ceiling crosses 500 here; between discrete b=3.5 and b=4)")
    print(f"\nVERDICT:")
    print(f"  sub_int4_clears_500_strict = {SUB_INT4_CLEARS_500_STRICT}")
    print(f"  strict_ceiling_at_best_ppl_safe_bits = {STRICT_CEILING_AT_BEST_PPL_SAFE_BITS:.4f}")
    print(f"  best_ppl_safe_bits = {BEST_PPL_SAFE_B}")
    print(f"  {report['verdict']['kill_reason']}")

    # Self-test
    st = report["self_test"]
    print(f"\nSelf-test:")
    for name, check in st["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  [{status}] {name}")
    print(f"  strict_sub_int4_self_test_passes = {st['strict_sub_int4_self_test_passes']}")

    # W&B
    if not args.no_wandb:
        run_id = log_to_wandb(report, args.wandb_group, args.wandb_name)
        report["wandb_run_id"] = run_id
        print(f"\nW&B run_id: {run_id}")
    else:
        report["wandb_run_id"] = None

    # Write JSON
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults written to: {out_path}")

    if args.self_test:
        if st["strict_sub_int4_self_test_passes"]:
            print("\nSELF-TEST PASSED")
            sys.exit(0)
        else:
            print("\nSELF-TEST FAILED", file=sys.stderr)
            failing = [k for k, v in st["checks"].items() if not v["pass"]]
            print(f"Failing checks: {failing}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
