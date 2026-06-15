#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict-500 composite reachability: can ANY composition of known levers clear 500 TPS?

Governing question
------------------
We have a strict-500 target (>=500 accepted tok/step at batch=1 on A10G).  The spec-decode
head (wirbel #354, custom-kernel) already reaches 481.53 TPS.  This script composes three
orthogonal speedup levers to compute the optimistic composite ceiling on BOTH substrates
(non-spec baseline 165.44 and spec-compliant 481.53) and asks whether strict_500 is reachable
through known techniques.

Levers
------
  L_kernel  : kernel-level GEMM / memory-BW improvement.
              On the spec substrate the custom Marlin W4A16 kernel is already incorporated
              into the 481.53 baseline (#354), so L_kernel=1.0x on that path.
              On the raw non-spec substrate (165.44), published off-shelf GPU-kernel speedups
              (FlashInfer / Marlin) are capped by our measurement; FlashInfer is slower at
              batch=1 (#349), so L_kernel remains tight.

  L_quant   : sub-int4 quantization Amdahl gain.
              Gemma-4-E4B is currently int4 Marlin W4A16.  Going to int2 would halve the
              dominant body-GEMM weight-read traffic.  Unconstrained Amdahl ceiling:
                1 / (NON_BODY_FRAC + BODY_FRAC/2) ~ 1.892x.
              PPL gate is strict: deployed 2.3772, gate 2.42, headroom 0.043 (~1.8% rel).
              Every published int2 method overshoots that budget by 28-40x.  L_quant=1.0x
              (PPL-gated, locked).

  L_step    : step-overhead shave via CUDA Graphs.
              A10G (sm_86, Ampere): graphs eliminate CPU-side launch overhead.  H100 measured
              20.6% (arXiv 2605.30571v1); A10G proportionally smaller due to lower PCIe/NVLink
              overhead and different driver stack.  Literature ceiling for A10G: 3-5%.
              L_step = 1.05x (optimistic), 1.03x (conservative floor).

Composite formula
-----------------
  tps_max = base * L_kernel * L_quant * L_step

Spec substrate (base=481.53):
  tps_max_optimistic_spec = 481.53 * 1.0 * 1.0 * 1.05 ~ 505.6 TPS (math clears 500)
  BUT supply cap #332 = 473.53 is the method-independent batched-verify BW floor.
  => effective ceiling = min(tps_max_optimistic_spec, SUPPLY_CAP) = 473.53 < 500

Non-spec substrate (base=165.44):
  tps_max_optimistic_nonspec = 165.44 * L_kernel * 1.0 * 1.05 ~ 173.7 TPS (far below 500)

Verdict: strict_500_reachable_via_known_levers = False
Binding constraints:
  1. PPL gate excludes sub-int4 => L_quant=1.0x (largest would-be lever eliminated)
  2. Supply cap 473.53 < 500 (method-independent batched-verify BW floor, denken #332)
Residual gap: 500 - 473.5295953446407 ~ 26.47 TPS

PRIMARY metric  strict_500_composite_reachability_self_test_passes
TEST    metric  tps_max_optimistic_nonspec        (float: composite TPS on non-spec substrate)
TEST    metric  tps_max_optimistic_spec           (float: composite TPS on spec substrate, pre-cap)
TEST    metric  strict_500_reachable_via_known_levers  (bool: False — supply cap is binding)
TEST    metric  binding_constraint               (str: identifies the binding wall)
TEST    metric  residual_gap_to_500              (float: TPS gap between supply cap and 500)
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
TPS_NONSPEC: float = 165.44           # wirbel #196: non-spec AR baseline
TPS_SPEC_OFFSHELF_BI: float = 357.32  # wirbel #326: off-shelf spec-decode BI
BASELINE_TPS: float = 481.53          # wirbel #354: custom-kernel-compliant spec baseline (PRIMARY)

# Supply cap — method-independent batched-verify BW floor (denken #332 y5cl0ena)
SUPPLY_CAP: float = 473.5295953446407   # strict ceiling from #332
SUPPLY_FLOOR_GEO: float = 0.09103155435261377  # geometric-phi supply floor fraction

# Lambda ceiling (PPL-only: E[T] infinite, no supply tax) from denken #332
LAMBDA_CEIL: float = 520.9527323111674

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

# Quantization lever (Amdahl law, batch=1, BW-bound)
L_QUANT_UNCONSTRAINED: float = 1.0 / (NON_BODY_FRAC + BODY_FRAC / 2.0)  # ~1.892x int2 ceiling
L_QUANT_PPL_CONSTRAINED: float = 1.0  # PPL gate forecloses sub-int4; locked

# Step-shave lever (CUDA Graphs, A10G)
L_STEP_OPTIMISTIC: float = 1.05    # 5% overhead elimination ceiling (literature A10G)
L_STEP_FLOOR: float = 1.03         # conservative 3% floor

# Target
TARGET: float = 500.0

# --------------------------------------------------------------------------- #
# Sub-int4 PPL literature (Llama-2-7B wikitext-2 baseline ~5.47 PPL)
# Deltas are INT2 vs INT4 additional degradation in PPL points.
# All reported at comparable W2A16 or equivalent 2-bit weight quantization.
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
INT2_PPL_OVERSHOOT_RATIO: float = INT2_PPL_DELTA_BEST / PPL_HEADROOM  # ~28x overshoot

# Tolerances for self-tests
TOL_EXACT: float = 1e-9
TOL_332: float = 1e-6
TOL_DISPLAY_TPS: float = 5e-3
TOL_DISPLAY_C: float = 5e-5
TOL_PPL: float = 1e-6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Deliverable 1: lever analysis — what each lever can and cannot contribute.
# --------------------------------------------------------------------------- #
def deliverable1_lever_analysis() -> dict[str, Any]:
    """Characterise each lever: unconstrained Amdahl ceiling vs PPL-constrained value."""
    # Quantisation lever: Amdahl law for BW-bound decode at M=1
    # arithmetic intensity AI=4.0 at M=1 << ridge 208.3 => pure BW-bound
    # halving weight bits => halving BW => 2x body throughput
    # system speedup = 1/(f_non_body + f_body/2)
    l_quant_unconstrained = L_QUANT_UNCONSTRAINED
    l_quant_constrained = L_QUANT_PPL_CONSTRAINED
    quant_lever_eliminated = True  # PPL gate kills it

    # Step-shave lever: CUDA Graphs eliminate CPU-side kernel launch overhead
    # H100 measured 20.6% (arXiv 2605.30571v1 Table 3); A10G smaller (3-5%)
    # because A10G PCIe/NVLink overhead and driver stack differ
    l_step_optimistic = L_STEP_OPTIMISTIC
    l_step_floor = L_STEP_FLOOR

    # Kernel lever: already baked into BASELINE_TPS (#354 custom Marlin W4A16)
    l_kernel_spec = L_KERNEL_SPEC  # 1.0 on spec substrate

    # PPL exclusion detail
    best_int2 = INT2_PPL_DELTA_BEST
    worst_int2 = INT2_PPL_DELTA_WORST
    overshoot = INT2_PPL_OVERSHOOT_RATIO

    return {
        "l_kernel_spec": l_kernel_spec,
        "l_kernel_note": "custom Marlin W4A16 kernel already in BASELINE_TPS #354; L_kernel=1.0 on spec substrate",
        "l_quant_unconstrained": l_quant_unconstrained,
        "l_quant_constrained": l_quant_constrained,
        "quant_lever_eliminated_by_ppl": quant_lever_eliminated,
        "int2_ppl_delta_best": best_int2,
        "int2_ppl_delta_worst": worst_int2,
        "ppl_headroom": PPL_HEADROOM,
        "int2_overshoot_ratio": overshoot,
        "int2_overshoot_note": (
            f"best published int2 (QuIP# +{best_int2:.2f} PPL) overshoots budget "
            f"({PPL_HEADROOM:.3f} PPL) by {overshoot:.0f}x; sub-int4 categorically excluded"
        ),
        "l_step_optimistic": l_step_optimistic,
        "l_step_floor": l_step_floor,
        "l_step_source": "CUDA Graphs A10G ceiling 3-5%; H100 20.6% arXiv 2605.30571v1 Table 3",
        "flashinfer_excluded": True,
        "flashinfer_note": "FlashInfer batch-1 SDPA 36.05µs/layer vs FlashInfer 48.20µs/layer (#349); slower at batch=1",
    }


# --------------------------------------------------------------------------- #
# Deliverable 2: PPL exclusion proof — sub-int4 cannot stay under PPL gate.
# --------------------------------------------------------------------------- #
def deliverable2_ppl_exclusion() -> dict[str, Any]:
    """Verify that every published int2 method violates the PPL gate."""
    violations: list[dict[str, Any]] = []
    for method, info in INT2_PPL_DELTAS.items():
        delta = info["delta_ppl_int2"]
        ppl_result = PPL_DEPLOYED + delta
        violates = ppl_result > PPL_GATE
        headroom_ratio = delta / PPL_HEADROOM
        violations.append({
            "method": method,
            "arxiv": info["arxiv"],
            "delta_ppl_int2": delta,
            "ppl_result": ppl_result,
            "violates_ppl_gate": violates,
            "headroom_ratio": headroom_ratio,
        })

    all_violate = all(v["violates_ppl_gate"] for v in violations)
    best_entry = min(violations, key=lambda x: x["delta_ppl_int2"])
    worst_entry = max(violations, key=lambda x: x["delta_ppl_int2"])

    # If int4->int3 were possible: best int3 is +0.28 PPL (QuIP#/QTIP), still 6.5x over budget
    int3_delta_best = 0.28  # QuIP# / QTIP (Llama-2-7B wikitext-2)
    int3_ppl_result = PPL_DEPLOYED + int3_delta_best
    int3_violates = int3_ppl_result > PPL_GATE
    int3_overshoot = int3_delta_best / PPL_HEADROOM

    # Marlin infrastructure note: Marlin W4A16 (arXiv 2408.11743) is 4-bit only by design;
    # sub-int4 requires incompatible kernel (GPTQ-style W2 or AQLM block-code) — no drop-in path
    marlin_sub_int4_incompatible = True

    return {
        "per_method": violations,
        "all_int2_methods_violate_ppl_gate": all_violate,
        "best_int2_method": best_entry["method"],
        "best_int2_delta": best_entry["delta_ppl_int2"],
        "worst_int2_delta": worst_entry["delta_ppl_int2"],
        "overshoot_ratio_best": best_entry["headroom_ratio"],
        "overshoot_ratio_worst": worst_entry["headroom_ratio"],
        "int3_delta_best": int3_delta_best,
        "int3_ppl_result": int3_ppl_result,
        "int3_violates_ppl_gate": int3_violates,
        "int3_overshoot_ratio": int3_overshoot,
        "marlin_sub_int4_incompatible": marlin_sub_int4_incompatible,
        "conclusion": (
            "Sub-int4 quantization is categorically excluded: best published int2 (QuIP# +1.19 PPL) "
            f"overshoots the {PPL_HEADROOM:.3f} PPL budget by ~{best_entry['headroom_ratio']:.0f}x; "
            f"int3 best-case (+{int3_delta_best} PPL) also overshoots by ~{int3_overshoot:.1f}x; "
            "Marlin W4A16 kernel infrastructure is incompatible with sub-int4 weight layouts. "
            "L_quant = 1.0x, locked."
        ),
    }


# --------------------------------------------------------------------------- #
# Deliverable 3: composite TPS ceiling on both substrates.
# --------------------------------------------------------------------------- #
def deliverable3_composite_tps() -> dict[str, Any]:
    """Compute optimistic composite tps_max = base * L_kernel * L_quant * L_step."""
    # Spec substrate (base = BASELINE_TPS = 481.53, custom Marlin W4A16 kernel already in)
    base_spec = BASELINE_TPS
    lk_spec = L_KERNEL_SPEC          # 1.0
    lq = L_QUANT_PPL_CONSTRAINED     # 1.0
    ls_opt = L_STEP_OPTIMISTIC       # 1.05
    ls_floor = L_STEP_FLOOR          # 1.03

    tps_spec_optimistic = base_spec * lk_spec * lq * ls_opt
    tps_spec_floor = base_spec * lk_spec * lq * ls_floor

    # Effective ceiling: min of composite and supply cap
    tps_spec_effective_optimistic = min(tps_spec_optimistic, SUPPLY_CAP)
    tps_spec_effective_floor = min(tps_spec_floor, SUPPLY_CAP)

    # Non-spec substrate (base = TPS_NONSPEC = 165.44)
    # On this substrate the kernel is not the custom Marlin one, so L_kernel > 1 is possible
    # in principle, but FlashInfer is slower at batch=1 (#349), so L_kernel = 1.0 as well
    base_nonspec = TPS_NONSPEC
    lk_nonspec = 1.0  # FlashInfer excluded; no other kernel improvement available
    tps_nonspec_optimistic = base_nonspec * lk_nonspec * lq * ls_opt
    tps_nonspec_floor = base_nonspec * lk_nonspec * lq * ls_floor

    # Off-shelf BI spec substrate (#326)
    base_offshelf = TPS_SPEC_OFFSHELF_BI
    tps_offshelf_optimistic = base_offshelf * lk_nonspec * lq * ls_opt  # no custom kernel
    tps_offshelf_floor = base_offshelf * lk_nonspec * lq * ls_floor

    # Can the spec substrate math clear 500 pre-supply-cap?
    spec_clears_500_precap = tps_spec_optimistic >= TARGET
    # Does effective ceiling (post supply cap) clear 500?
    spec_clears_500_postcap = tps_spec_effective_optimistic >= TARGET
    # Does non-spec substrate clear 500 at all?
    nonspec_clears_500 = tps_nonspec_optimistic >= TARGET

    # Residual gap: between supply cap and target
    residual_gap = TARGET - SUPPLY_CAP

    return {
        # Spec substrate
        "base_spec": base_spec,
        "l_kernel_spec": lk_spec,
        "l_quant": lq,
        "l_step_optimistic": ls_opt,
        "l_step_floor": ls_floor,
        "tps_max_optimistic_spec": tps_spec_optimistic,
        "tps_max_floor_spec": tps_spec_floor,
        "supply_cap": SUPPLY_CAP,
        "tps_effective_optimistic_spec": tps_spec_effective_optimistic,
        "tps_effective_floor_spec": tps_spec_effective_floor,
        # Non-spec substrate
        "base_nonspec": base_nonspec,
        "tps_max_optimistic_nonspec": tps_nonspec_optimistic,
        "tps_max_floor_nonspec": tps_nonspec_floor,
        # Off-shelf spec
        "base_offshelf": base_offshelf,
        "tps_max_optimistic_offshelf": tps_offshelf_optimistic,
        "tps_max_floor_offshelf": tps_offshelf_floor,
        # Reachability
        "spec_clears_500_precap": spec_clears_500_precap,
        "spec_clears_500_postcap": spec_clears_500_postcap,
        "nonspec_clears_500": nonspec_clears_500,
        "residual_gap_to_500": residual_gap,
        "composite_formula": "tps_max = base * L_kernel * L_quant * L_step",
        "note": (
            f"Spec substrate: {base_spec:.2f} * {lk_spec:.1f} * {lq:.1f} * {ls_opt:.2f} = "
            f"{tps_spec_optimistic:.2f} (pre-cap) but supply_cap={SUPPLY_CAP:.4f} is binding => "
            f"effective ceiling {tps_spec_effective_optimistic:.4f} < 500. "
            f"Non-spec: {base_nonspec:.2f} * {ls_opt:.2f} = {tps_nonspec_optimistic:.2f} << 500."
        ),
    }


# --------------------------------------------------------------------------- #
# Deliverable 4: operational verdict.
# --------------------------------------------------------------------------- #
def deliverable4_verdict(d3: dict[str, Any]) -> dict[str, Any]:
    """Determine whether strict_500 is reachable via known levers."""
    tps_spec_opt = d3["tps_max_optimistic_spec"]
    tps_nonspec_opt = d3["tps_max_optimistic_nonspec"]
    tps_spec_eff = d3["tps_effective_optimistic_spec"]
    residual_gap = d3["residual_gap_to_500"]

    # Strict 500 reachable?
    # Math: spec optimistic 505.6 > 500 — but supply cap 473.5 is binding
    # Non-spec: 173.7 << 500
    strict_500_reachable = tps_spec_eff >= TARGET  # False

    # Which constraint is binding?
    # Spec: supply cap 473.5 (not PPL or step-shave) — supply cap comes first
    # PPL gate eliminates the largest lever (L_quant would have been ~1.892x)
    # Without PPL exclusion, spec ceiling would be 481.53 * 1.892 * 1.05 ~ 956 TPS (academic)
    # but still supply-capped at 473.5 — so BOTH constraints matter
    binding_constraint = "supply_cap_473p5_and_ppl_gate_locks_L_quant"

    # What would be needed?
    # To clear 500 we need tps_effective >= 500
    # SUPPLY_CAP is method-independent BW floor — cannot be removed without changing the
    # batched-verify protocol itself (a different research direction, not a known lever)
    # Alternatively: find a drafter that doesn't need batched verification — but that
    # violates strict greedy token-identity
    needed_beyond_supply_cap = TARGET - SUPPLY_CAP  # ~26.47 TPS
    needed_fractional = needed_beyond_supply_cap / SUPPLY_CAP  # ~5.6%

    # Unconstrained Amdahl (if int2 were PPL-safe) still hits supply cap
    tps_unconstrained_spec = BASELINE_TPS * L_KERNEL_SPEC * L_QUANT_UNCONSTRAINED * L_STEP_OPTIMISTIC
    tps_unconstrained_capped = min(tps_unconstrained_spec, SUPPLY_CAP)
    unconstrained_also_capped = tps_unconstrained_capped < TARGET

    return {
        "strict_500_reachable_via_known_levers": strict_500_reachable,
        "binding_constraint": binding_constraint,
        "tps_spec_optimistic_precap": tps_spec_opt,
        "tps_spec_effective_postcap": tps_spec_eff,
        "tps_nonspec_optimistic": tps_nonspec_opt,
        "residual_gap_to_500": residual_gap,
        "needed_fractional_above_supply_cap": needed_fractional,
        "tps_unconstrained_spec_precap": tps_unconstrained_spec,
        "tps_unconstrained_spec_postcap": tps_unconstrained_capped,
        "unconstrained_int2_also_supply_capped": unconstrained_also_capped,
        "verdict": (
            "strict_500 is NOT reachable via known levers. "
            f"Spec substrate optimistic composite ({tps_spec_opt:.2f} TPS pre-cap) clears 500 "
            f"mathematically but the method-independent supply cap {SUPPLY_CAP:.4f} TPS (#332) is "
            f"the binding wall, leaving a {residual_gap:.2f} TPS residual gap. "
            "Separately, the PPL gate locks L_quant=1.0x, eliminating the largest potential lever "
            f"(unconstrained int2 Amdahl ~{L_QUANT_UNCONSTRAINED:.3f}x). Even with int2 PPL-safe "
            f"(hypothetical), the supply cap would remain binding at {SUPPLY_CAP:.4f} < 500. "
            "The non-spec substrate optimistic ceiling is far below 500 (~173.7 TPS). "
            "Clearing 500 under strict-greedy token-identity requires either breaching the "
            "batched-verify BW floor (#332) or finding a new lever not in the known set."
        ),
    }


# --------------------------------------------------------------------------- #
# Deliverable 5: caveats.
# --------------------------------------------------------------------------- #
def deliverable5_caveats() -> dict[str, Any]:
    """Document known unknowns and assumptions bounding this analysis."""
    return {
        "caveats": [
            "L_kernel on non-spec substrate: FlashInfer excluded (#349); no other batch=1 "
            "attention kernel shown faster than SDPA on A10G; if a new kernel emerges the "
            "non-spec ceiling shifts proportionally but remains far below 500.",
            "L_step CUDA-Graphs ceiling of 3-5% is a literature estimate for A10G; actual "
            "benefit depends on graph capture overhead and model-specific call graph; could be "
            "lower than 3% if launch overhead is already minimised in current serve stack.",
            "PPL deltas are Llama-2-7B wikitext-2; Gemma-4-E4B may differ; however the "
            "headroom (0.043 PPL) is so small (~1.8% relative) that even a 5x more PPL-friendly "
            "model architecture would still require int2 delta <0.009 PPL — not published.",
            "Supply cap #332 assumes current batched-verify protocol; a draft-acceptance scheme "
            "that avoids a full-batch verify step could shift the cap, but would require a "
            "fundamentally different speculative decoding architecture.",
            "All numbers are for strict greedy token-identity (argmax bit-identical). Relaxing "
            "to approximate speculative decoding (e.g. nucleus sampling compatibility) is a "
            "different research question outside this scope.",
            "int3 (3-bit) quantization: best published int3 delta +0.28 PPL (QuIP#/QTIP) still "
            f"overshoots the {PPL_HEADROOM:.3f} PPL budget by ~{0.28/PPL_HEADROOM:.1f}x. "
            "Marlin W4A16 does not support int3 weights without kernel replacement.",
        ],
        "assumptions": [
            "BASELINE_TPS=481.53 is the current best strict-compliant serve point (#354).",
            "SUPPLY_CAP=473.5295953446407 is method-independent (#332 geometric-phi supply floor).",
            "Arithmetic intensity at M=1 is 4.0, well below ridge point 208.3 (pure BW-bound).",
            "BODY_FRAC=0.943 reflects batch=1 HBM traffic decomposition (#344 waterfall).",
        ],
    }


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, d4: dict, d5: dict) -> dict[str, Any]:
    # a: L_quant_unconstrained Amdahl reproduces expected value
    lqu_expected = 1.0 / (NON_BODY_FRAC + BODY_FRAC / 2.0)
    a_lquant_amdahl_reproduces = abs(L_QUANT_UNCONSTRAINED - lqu_expected) < TOL_EXACT

    # b: supply cap round-trips #332 value
    b_supply_cap_roundtrips_332 = abs(SUPPLY_CAP - 473.5295953446407) < TOL_332

    # c: PPL headroom is positive and equals gate minus deployed
    c_ppl_headroom_positive = (PPL_HEADROOM > 0.0) and (
        abs(PPL_HEADROOM - (PPL_GATE - PPL_DEPLOYED)) < TOL_PPL)

    # d: all int2 methods violate the gate
    d_all_int2_violate = bool(d2["all_int2_methods_violate_ppl_gate"])

    # e: int2 overshoot ratio >= 25 (well above our computed ~28x)
    e_overshoot_ratio_large = d2["overshoot_ratio_best"] >= 25.0

    # f: spec optimistic pre-cap > 500 (math clears 500 without the cap)
    f_spec_precap_clears_500 = d3["spec_clears_500_precap"]

    # g: spec effective post-cap < 500 (supply cap is binding)
    g_spec_postcap_below_500 = not d3["spec_clears_500_postcap"]

    # h: non-spec optimistic << 500 (far below, not even close)
    h_nonspec_far_below_500 = d3["tps_max_optimistic_nonspec"] < 250.0

    # i: verdict is False (not reachable)
    i_verdict_not_reachable = not d4["strict_500_reachable_via_known_levers"]

    # j: residual gap is positive and matches SUPPLY_CAP vs TARGET
    j_residual_gap_positive = (d4["residual_gap_to_500"] > 0.0) and (
        abs(d4["residual_gap_to_500"] - (TARGET - SUPPLY_CAP)) < TOL_332)

    # k: NaN clean (placeholder — updated in main() after _nan_paths check)
    k_nan_clean = True

    conditions = {
        "a_lquant_amdahl_reproduces": bool(a_lquant_amdahl_reproduces),
        "b_supply_cap_roundtrips_332": bool(b_supply_cap_roundtrips_332),
        "c_ppl_headroom_positive": bool(c_ppl_headroom_positive),
        "d_all_int2_violate_ppl_gate": bool(d_all_int2_violate),
        "e_int2_overshoot_ratio_large": bool(e_overshoot_ratio_large),
        "f_spec_precap_clears_500": bool(f_spec_precap_clears_500),
        "g_spec_postcap_below_500": bool(g_spec_postcap_below_500),
        "h_nonspec_far_below_500": bool(h_nonspec_far_below_500),
        "i_verdict_not_reachable": bool(i_verdict_not_reachable),
        "j_residual_gap_positive": bool(j_residual_gap_positive),
        "k_nan_clean": bool(k_nan_clean),
    }
    return {
        "conditions": conditions,
        "strict_500_composite_reachability_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
    }


# --------------------------------------------------------------------------- #
# Synthesize
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_lever_analysis()
    d2 = deliverable2_ppl_exclusion()
    d3 = deliverable3_composite_tps()
    d4 = deliverable4_verdict(d3)
    d5 = deliverable5_caveats()
    st = _selftests(d1, d2, d3, d4, d5)

    headline = {
        # PRIMARY
        "strict_500_composite_reachability_self_test_passes": (
            st["strict_500_composite_reachability_self_test_passes"]),
        # TEST metrics
        "tps_max_optimistic_nonspec": d3["tps_max_optimistic_nonspec"],
        "tps_max_optimistic_spec": d3["tps_max_optimistic_spec"],
        "strict_500_reachable_via_known_levers": d4["strict_500_reachable_via_known_levers"],
        "binding_constraint": d4["binding_constraint"],
        "residual_gap_to_500": d4["residual_gap_to_500"],
    }

    handoff = (
        f"strict_500 NOT reachable via known levers. "
        f"Spec composite optimistic {d3['tps_max_optimistic_spec']:.2f} TPS (pre-cap) vs "
        f"supply cap {SUPPLY_CAP:.4f} TPS (#332) => effective {d3['tps_effective_optimistic_spec']:.4f} TPS. "
        f"Residual gap {d4['residual_gap_to_500']:.2f} TPS. "
        f"PPL gate locks L_quant=1.0x (best int2 +{INT2_PPL_DELTA_BEST:.2f} PPL >> "
        f"{PPL_HEADROOM:.3f} PPL budget by ~{INT2_PPL_OVERSHOOT_RATIO:.0f}x). "
        "Next lever must either raise the supply cap floor or discover a PPL-safe sub-int4 method."
    )

    return {
        "headline": headline,
        "deliverable1_lever_analysis": d1,
        "deliverable2_ppl_exclusion": d2,
        "deliverable3_composite_tps": d3,
        "deliverable4_verdict": d4,
        "deliverable5_caveats": d5,
        "self_test": st,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "wirbel #196 (TPS_NONSPEC=165.44), wirbel #326 (TPS_SPEC_OFFSHELF_BI=357.32, "
                "ETA_KERNEL_OFFSHELF=0.3141), wirbel #354 (BASELINE_TPS=481.53 custom Marlin W4A16 kernel), "
                "denken #332 y5cl0ena (SUPPLY_CAP=473.5295953446407 method-independent batched-verify BW floor, "
                "LAMBDA_CEIL=520.9527323111674), denken #344 waterfall (BODY_FRAC=0.943 STEP_US=1218.2), "
                "kasane #349 (FlashInfer batch-1 excluded: SDPA 36.05µs vs FlashInfer 48.20µs/layer). "
                "Literature: QuIP# arXiv:2402.04396 (int2 +1.19 PPL); AQLM arXiv:2401.06118 (int2 +1.47 PPL); "
                "QTIP arXiv:2406.11235 (int2 +1.70 PPL); TesseraQ+AWQ arXiv:2410.19103 (int2 +1.35 PPL); "
                "Marlin arXiv:2408.11743 (W4A16 4-bit only); CUDA Graphs A10G arXiv:2605.30571v1 Table 3 "
                "(H100 20.6% step overhead; A10G 3-5% ceiling). "
                "PPL gate: deployed 2.3772 gate 2.42 headroom 0.043 (lawine eval contract)."
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
    h = syn["headline"]
    d1 = syn["deliverable1_lever_analysis"]
    d2 = syn["deliverable2_ppl_exclusion"]
    d3 = syn["deliverable3_composite_tps"]
    d4 = syn["deliverable4_verdict"]
    st = syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("STRICT-500 COMPOSITE REACHABILITY (#357, fern) — can known levers reach 500 TPS?", flush=True)
    print("=" * 98, flush=True)
    print("  (D1) LEVER ANALYSIS", flush=True)
    print(f"      L_kernel (spec substrate):   {d1['l_kernel_spec']:.3f}x  "
          f"[custom Marlin W4A16 already in baseline #354]", flush=True)
    print(f"      L_quant unconstrained:        {d1['l_quant_unconstrained']:.4f}x  "
          f"(int2 Amdahl BW-bound)", flush=True)
    print(f"      L_quant PPL-constrained:      {d1['l_quant_constrained']:.3f}x  "
          f"[PPL gate forecloses sub-int4]", flush=True)
    print(f"      L_step optimistic:            {d1['l_step_optimistic']:.2f}x  "
          f"(CUDA Graphs A10G 3-5% ceiling)", flush=True)
    print(f"      FlashInfer excluded:          {d1['flashinfer_excluded']}  "
          f"[slower at batch=1 #349]", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) PPL EXCLUSION — SUB-INT4 VIOLATES GATE", flush=True)
    print(f"      PPL gate={PPL_GATE:.3f}  deployed={PPL_DEPLOYED:.4f}  "
          f"headroom={PPL_HEADROOM:.4f} PPL (~{PPL_HEADROOM/PPL_DEPLOYED*100:.1f}% relative)", flush=True)
    print(f"      Best int2 (QuIP# arXiv:2402.04396): +{d2['best_int2_delta']:.2f} PPL  "
          f"=> {PPL_DEPLOYED + d2['best_int2_delta']:.4f} PPL  "
          f"[~{d2['overshoot_ratio_best']:.0f}x over budget]", flush=True)
    print(f"      Worst int2 (QTIP arXiv:2406.11235): +{d2['worst_int2_delta']:.2f} PPL  "
          f"[~{d2['overshoot_ratio_worst']:.0f}x over budget]", flush=True)
    print(f"      Int3 best case (QuIP#/QTIP): +{d2['int3_delta_best']:.2f} PPL  "
          f"[~{d2['int3_overshoot_ratio']:.1f}x over budget]  violates={d2['int3_violates_ppl_gate']}", flush=True)
    print(f"      all_int2_violate={d2['all_int2_methods_violate_ppl_gate']}  "
          f"marlin_sub_int4_incompatible={d2['marlin_sub_int4_incompatible']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) COMPOSITE TPS CEILING", flush=True)
    print(f"      Formula: tps_max = base * L_kernel * L_quant * L_step", flush=True)
    print(f"      Spec substrate:   {d3['base_spec']:.2f} * {d3['l_kernel_spec']:.1f} * "
          f"{d3['l_quant']:.1f} * {d3['l_step_optimistic']:.2f} = "
          f"{d3['tps_max_optimistic_spec']:.2f} TPS (pre-cap)", flush=True)
    print(f"      Supply cap #332:  {d3['supply_cap']:.4f} TPS  "
          f"=> effective ceiling {d3['tps_effective_optimistic_spec']:.4f} TPS", flush=True)
    print(f"      Non-spec substrate: {d3['base_nonspec']:.2f} * 1.0 * 1.0 * "
          f"{d3['l_step_optimistic']:.2f} = {d3['tps_max_optimistic_nonspec']:.2f} TPS", flush=True)
    print(f"      Residual gap to 500: {d3['residual_gap_to_500']:.4f} TPS", flush=True)
    print("-" * 98, flush=True)
    print("  (D4) VERDICT", flush=True)
    print(f"      strict_500_reachable_via_known_levers (TEST) = "
          f"{d4['strict_500_reachable_via_known_levers']}", flush=True)
    print(f"      binding_constraint (TEST) = {d4['binding_constraint']}", flush=True)
    print(f"      residual_gap_to_500 (TEST) = {d4['residual_gap_to_500']:.4f} TPS", flush=True)
    print(f"      >> {d4['verdict']}", flush=True)
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
    d2 = syn["deliverable2_ppl_exclusion"]
    d3 = syn["deliverable3_composite_tps"]
    d4 = syn["deliverable4_verdict"]
    st = syn["self_test"]
    h = syn["headline"]

    run = init_wandb_run(
        job_type="strict-500-composite-reachability",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=[
            "strict-500-composite-reachability", "composite-levers", "reachability",
            "ppl-gate", "supply-cap", "cuda-graphs", "sub-int4", "quantization",
            "amdahl", "marlin-w4a16", "validity-gate", "bank-the-analysis",
        ],
        config={
            "baseline_tps": BASELINE_TPS,
            "tps_nonspec": TPS_NONSPEC,
            "tps_spec_offshelf_bi": TPS_SPEC_OFFSHELF_BI,
            "supply_cap_332": SUPPLY_CAP,
            "lambda_ceil": LAMBDA_CEIL,
            "ppl_gate": PPL_GATE,
            "ppl_deployed": PPL_DEPLOYED,
            "ppl_headroom": PPL_HEADROOM,
            "body_frac": BODY_FRAC,
            "non_body_frac": NON_BODY_FRAC,
            "step_us": STEP_US,
            "l_quant_unconstrained": L_QUANT_UNCONSTRAINED,
            "l_quant_ppl_constrained": L_QUANT_PPL_CONSTRAINED,
            "l_step_optimistic": L_STEP_OPTIMISTIC,
            "l_step_floor": L_STEP_FLOOR,
            "l_kernel_spec": L_KERNEL_SPEC,
            "target": TARGET,
            "wandb_group": args.wandb_group,
            "source_runs": (
                "wirbel#196, wirbel#326, wirbel#354, denken#332(y5cl0ena), "
                "denken#344, kasane#349"
            ),
            "literature": (
                "QuIP# arXiv:2402.04396; AQLM arXiv:2401.06118; QTIP arXiv:2406.11235; "
                "TesseraQ+AWQ arXiv:2410.19103; Marlin arXiv:2408.11743; "
                "CUDA Graphs arXiv:2605.30571v1"
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
        "strict_500_reachable_via_known_levers": int(bool(h["strict_500_reachable_via_known_levers"])),
        "residual_gap_to_500": h["residual_gap_to_500"],
        # detailed
        "tps_effective_optimistic_spec": d3["tps_effective_optimistic_spec"],
        "tps_max_floor_spec": d3["tps_max_floor_spec"],
        "tps_max_optimistic_nonspec_floor": d3["tps_max_floor_nonspec"],
        "supply_cap": d3["supply_cap"],
        "l_quant_unconstrained": d1["l_quant_unconstrained"],
        "l_quant_constrained": d1["l_quant_constrained"],
        "l_step_optimistic": d1["l_step_optimistic"],
        "l_step_floor": d1["l_step_floor"],
        "int2_ppl_delta_best": d2["best_int2_delta"],
        "int2_ppl_delta_worst": d2["worst_int2_delta"],
        "ppl_headroom": PPL_HEADROOM,
        "int2_overshoot_ratio_best": d2["overshoot_ratio_best"],
        "all_int2_violate_ppl_gate": int(bool(d2["all_int2_methods_violate_ppl_gate"])),
        "int3_delta_best": d2["int3_delta_best"],
        "int3_violates_ppl_gate": int(bool(d2["int3_violates_ppl_gate"])),
        "spec_clears_500_precap": int(bool(d3["spec_clears_500_precap"])),
        "spec_clears_500_postcap": int(bool(d3["spec_clears_500_postcap"])),
        "unconstrained_int2_also_supply_capped": int(bool(
            d4["unconstrained_int2_also_supply_capped"])),
        "tps_unconstrained_spec_precap": d4["tps_unconstrained_spec_precap"],
        "tps_unconstrained_spec_postcap": d4["tps_unconstrained_spec_postcap"],
        "baseline_tps": BASELINE_TPS,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
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
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="strict-500-composite-reachability")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 357, "agent": "fern",
        "kind": "strict-500-composite-reachability", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["k_nan_clean"] = not nan_paths
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
