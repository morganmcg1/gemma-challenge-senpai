#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #327 (denken) -- first-principles FLOOR of a batch-invariant bf16 lm_head+attn reduction.

THE DECISIVE QUESTION (#192 identity-half GO/NO-GO)
---------------------------------------------------
The strict-compliant >500 TPS lane needs an M-invariant reduction that RESTORES
M=8 greedy-identity. denken #232 (`nxwv6pam`) measured the 0.73% M=8 divergence;
lawine #288 (`i1e5054m`) + wirbel #324 (`pespixw1`) then root-caused it: the
int4-Marlin body GEMMs are **bit-exact across M** (max_abs_diff=0.0) -> they
contribute ZERO divergence and need NO fix. The divergence locus is the **bf16
tied lm_head + the bf16 SDPA/attention accumulation + norms** -- NOT the int4
split-K. This RE-SCOPES wirbel #216's "0.95% custom-kernel floor", which was
derived for the WRONG kernel (the now-known-M-invariant int4 split-K). The TRUE
floor -- the minimum overhead of a hand-written M-invariant **bf16 lm_head+attn**
reduction -- is what this card derives, answering: is the identity lane ALIVE
even with an IDEAL custom kernel (floor <= the lambda=1 budget 7.332%), or DEAD
even then?

THE MODEL (BW-gap -> forgone-parallelism), GROUNDED IN MEASURED PROFILING
-------------------------------------------------------------------------
Forcing an M-invariant (fixed-order / single-split / fixed-tree) reduction
forgoes the reduction parallelism the batch-variant split schedule (split-K for
GEMMs, split-KV for attention) was exploiting to lift occupancy. That forgone
parallelism is bounded by how far the kernel sits from the bandwidth roofline:
a near-roofline kernel (lm_head @ 83.4% BW) has little parallel-reduction
headroom to give up (cheap to determinize); a low-BW kernel (SDPA @ 34.9% BW)
has lots of split-reduction parallelism in play (costly). The per-component
determinism penalty is therefore the **above-roofline exposed slack** =
T_component * (1 - BW_util) -- the part of the kernel's time NOT explained by
streaming bytes at peak, i.e. the compute/occupancy/latency headroom an adaptive
split schedule was minimizing.

This is NOT a hand-waved coefficient: denken #291 (`verify_compute_hideability`,
the banked M=8 verify profiling card) measured exactly this per component on the
A10G -- `above_roofline_exposed_us_m8 == us_at_m8 * (1 - bw_utilization)` to
machine precision. We import those measured rows directly and verify the identity
holds <=1e-6 (the penalty model IS the measured BW-gap).

floor_overhead(component) = exposed_us(component) / step_us_total
                          = step_share(component) * (1 - BW_util(component))

WHAT THIS CARD DOES (CPU analytic over banked numbers; 0 GPU, 0 TPS)
-------------------------------------------------------------------
1. Partition the M=8 verify step (denken #291 decomposition) into the int4 body
   (gate_up+down+qkv+o_proj -- bit-exact across M per #288, OUT of scope) and the
   bf16 reduction-sensitive locus (tied lm_head + bf16 SDPA + norms). Report
   bf16_locus_step_share.
2. Per-component BW-gap floor for (a) lm_head only, (b) SDPA only, (c) combined
   (a)+(b)+norms. State the penalty model + assumptions.
3. Verdict vs the lambda=1 budget 7.332% (#213). Compliant ceiling TPS =
   520.953*(1-floor) | identity==1.0.
4. The decisive three-way (A)/(B)/(C) partition.
5. Honest caveats + the dominant uncertainty (the SDPA penalty-law curvature: the
   BW-gap floor charges the FULL forgone slack; an ideal kernel recovering
   parallelism on non-reduction axes pays less -> phi* recovery break-even).

HONEST SCOPE
------------
0 TPS. BASELINE 481.53 unchanged. NO new GPU measurement, NO model forward, NO
served-file change, NO HF Job, NO submission, NO build, NO launch. This prices
the IDENTITY-HALF step-inflation FLOOR from banked numbers; the custom kernel
itself is UNBUILT and human-approval-gated. The floor is a LOWER bracket for the
correctly-scoped problem (vs wirbel #326's config-only empirical UPPER bracket),
NOT a buildability proof. NOT a launch. NOT a build. NOT a served-file change.

PRIMARY metric  bi_reduction_floor_self_test_passes
TEST    metric  first_principles_floor_overhead  (combined fraction-of-step)
                + identity_lane_alive_with_custom_kernel (bool)
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
# denken #291 (`verify_compute_hideability`, PR #291) -- the MEASURED M=8 verify
# step decomposition on the A10G. Each row: pct_of_verify (share of the 5348.13us
# full M=8 verify forward), bw_utilization, us_at_m8, above_roofline_exposed_us.
# The exposed_us == us * (1 - bw_util) identity is the measured BW-gap = our
# penalty model. `int4` = Marlin int4 GEMM (bit-exact across M per lawine #288 ->
# OUT of the determinism-fix scope). `bf16_locus` = the reduction-sensitive locus
# (#288/#324: tied lm_head + bf16 SDPA accumulation + norms).
COMPONENT_ROWS_291: list[dict[str, Any]] = [
    {"component": "gate_up_proj", "pct_of_verify": 43.044125842194994,
     "bw_util": 0.7120976097656768, "us_at_m8": 2302.054443359375,
     "exposed_us": 662.7669766927085, "int4": True, "bf16_locus": False},
    {"component": "down_proj", "pct_of_verify": 23.091147361827066,
     "bw_util": 0.6647319780770861, "us_at_m8": 1234.9438476562498,
     "exposed_us": 414.03718098958325, "int4": True, "bf16_locus": False},
    {"component": "sdpa", "pct_of_verify": 14.513725794080164,
     "bw_util": 0.34883864849061247, "us_at_m8": 776.2124633789062,
     "exposed_us": 505.43955671223955, "int4": False, "bf16_locus": True},
    {"component": "qkv_proj", "pct_of_verify": 9.875198162689072,
     "bw_util": 0.46965007082810634, "us_at_m8": 528.1381225585939,
     "exposed_us": 280.0980158919271, "int4": True, "bf16_locus": False},
    {"component": "o_proj", "pct_of_verify": 6.828545607119351,
     "bw_util": 0.4550995466487282, "us_at_m8": 365.19927978515625,
     "exposed_us": 198.9972531184896, "int4": True, "bf16_locus": False},
    {"component": "lm_head", "pct_of_verify": 2.358516890006141,
     "bw_util": 0.8344417980018903, "us_at_m8": 126.136474609375,
     "exposed_us": 20.88292794270834, "int4": False, "bf16_locus": True},
    # io_residual = the io/residual/NORM catch-all. #291 measured its above-roofline
    # exposed slack at 0.0us ("already captured by deployed ONEGRAPH/fusion"). This
    # is our measured proxy for the NORM reduction term of the bf16 locus.
    {"component": "io_residual", "pct_of_verify": 0.28874034208321014,
     "bw_util": None, "us_at_m8": 15.44219970703125,
     "exposed_us": 0.0, "int4": False, "bf16_locus": True, "is_norms": True},
]
TOTAL_VERIFY_US_291 = 5348.1268310546875            # k280_full_us_measured (#291)
VERIFY_GEMM_COST_SHARE_OF_STEP_216 = 0.606620584396473   # #216 verify-gemm share

# wirbel #293 (`eagle3_step_overhead`) / wirbel #295 step constants.
OFFICIAL_TPS = 481.53                                # BASELINE (this card adds 0 TPS)
LAMBDA1_CEIL = 520.9527323111674                     # lambda=1 ceiling (#204/#293)
K_CAL = 125.268                                      # tokens/sec per E[T] (#293)
STEP_US = 1218.2                                     # normalized step (#293)
TAU = 1.218                                          # step in ms (#293)
ET_BUILT_611 = 6.1112149873699195                    # realistic built E[T] (#293/#325)
TARGET_TPS = 500.0

# wirbel #213 (`kernel_budget_lambda`) -- the lambda=1 identity-overhead budget.
# Value is a PERCENT; the comparison bar as a FRACTION is /100.
BUDGET_LAMBDA1_PCT_213 = 7.331808522875782           # 7.332% (display)
BUDGET_LAMBDA1_FRAC_213 = BUDGET_LAMBDA1_PCT_213 / 100.0   # 0.0733180852...

# wirbel #216 (via margingate #223 imported_anchors) -- the two #216 anchors.
OFFTHESHELF_VERIFYGEMM_PCT_216 = 31.410813860049373      # off-the-shelf UPPER anchor (%)
OFFTHESHELF_VERIFYGEMM_FRAC_216 = OFFTHESHELF_VERIFYGEMM_PCT_216 / 100.0
CUSTOM_KERNEL_FLOOR_MISSCOPED_PCT_216 = 0.9455349322572293   # the MIS-SCOPED int4 floor (%)
CUSTOM_KERNEL_FLOOR_MISSCOPED_FRAC_216 = CUSTOM_KERNEL_FLOOR_MISSCOPED_PCT_216 / 100.0

# denken #232 (`nxwv6pam`) measured M=8 divergence (the thing being fixed).
M8_DIVERGENCE_PCT_232 = 0.73

TOL_EXACT = 1e-6      # imported-constant / identity round-trip tolerance
TOL_BUDGET = 1e-6     # lambda-budget import tolerance


# --------------------------------------------------------------------------- #
# Penalty model: BW-gap -> forgone-parallelism. pi(u) = 1 - u (the above-roofline
# fraction). Monotone INCREASING in (1 - BW_util): a kernel further from the
# bandwidth roofline has more split-reduction parallelism to forgo.
# --------------------------------------------------------------------------- #
def bw_gap_penalty(bw_util: float) -> float:
    """Per-kernel determinism penalty as a fraction of the kernel's own time.

    pi(u) = 1 - u = the above-roofline exposed fraction = the forgone-parallelism
    headroom the M-adaptive split schedule was exploiting. Bounded to [0, 1].
    """
    return float(max(0.0, min(1.0, 1.0 - bw_util)))


def component_floor_share(row: dict[str, Any], total_us: float) -> float:
    """Floor overhead this component contributes, as a fraction of the step.

    = exposed_us / total_us = step_share * (1 - bw_util). Uses the MEASURED
    exposed_us (#291) so io_residual (bw_util None, exposed 0) is handled exactly.
    """
    return float(row["exposed_us"] / total_us)


# --------------------------------------------------------------------------- #
# Three-way GO/NO-GO partition (the decisive identity-half outcome).
# --------------------------------------------------------------------------- #
def three_way_outcome(floor: float, budget: float,
                      c326_config_only: float | None) -> str:
    """(A) floor<=budget AND #326<=budget  -> alive with EXISTING knobs.
    (B) floor<=budget < #326                -> alive but needs a hand-written kernel.
    (C) floor>budget                        -> DEAD even with an ideal custom kernel.

    #326 (config-only ceiling) is a SIBLING card not readable in this launch; when
    unknown, (C) still fires on floor alone (it does not depend on #326), and the
    floor<=budget branch is reported as 'A-or-B (pending #326)'.
    """
    if floor > budget + TOL_EXACT:
        return "C_dead_even_with_custom_kernel"
    if c326_config_only is None:
        return "AB_alive_pending_326_config_only"
    if c326_config_only <= budget + TOL_EXACT:
        return "A_alive_with_existing_knobs"
    return "B_alive_needs_custom_kernel"


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(c326_config_only: float | None = None) -> dict[str, Any]:
    rows = COMPONENT_ROWS_291
    total_us = TOTAL_VERIFY_US_291

    # ---------- step 1: partition int4 body vs bf16 reduction-sensitive locus ---------- #
    int4_rows = [r for r in rows if r["int4"]]
    locus_rows = [r for r in rows if r["bf16_locus"]]
    # measured shares (fraction of step), reproduced from us_at_m8 / total.
    for r in rows:
        r["step_share_recomputed"] = r["us_at_m8"] / total_us
    int4_body_step_share = sum(r["us_at_m8"] for r in int4_rows) / total_us       # ~0.8284
    bf16_locus_step_share = sum(r["us_at_m8"] for r in locus_rows) / total_us     # ~0.1716
    mlp_step_share = sum(r["us_at_m8"] for r in rows
                         if r["component"] in ("gate_up_proj", "down_proj")) / total_us  # 0.6614
    partition_exhaustive = abs((int4_body_step_share + bf16_locus_step_share) - 1.0) < TOL_EXACT
    # confirm the pct_of_verify rows reproduce the us-based shares (the #291 import is consistent).
    pct_vs_us_max_resid = max(abs(r["pct_of_verify"] / 100.0 - r["step_share_recomputed"])
                              for r in rows)

    # the penalty model IS the measured BW-gap: exposed_us == us*(1-bw_util) (<=1e-6).
    exposed_identity_max_resid = max(
        abs(r["exposed_us"] - r["us_at_m8"] * bw_gap_penalty(r["bw_util"]))
        for r in rows if r["bw_util"] is not None)

    # ---------- step 2: per-component BW-gap floor (a)(b)(c) ---------- #
    def floor_of(component: str) -> float:
        r = next(r for r in rows if r["component"] == component)
        return component_floor_share(r, total_us)

    floor_a_lm_head = floor_of("lm_head")                          # ~0.00390
    floor_b_sdpa = floor_of("sdpa")                                # ~0.09451
    floor_norms = floor_of("io_residual")                          # 0.0 (measured)
    floor_combined = floor_a_lm_head + floor_b_sdpa + floor_norms  # ~0.09841

    # per-component penalty fraction pi (of the kernel's own time), for the report.
    per_component = []
    for r in locus_rows:
        u = r["bw_util"]
        pi = (r["exposed_us"] / r["us_at_m8"]) if r["us_at_m8"] > 0 else 0.0
        per_component.append({
            "component": r["component"], "bw_util": u, "penalty_pi": pi,
            "step_share": r["step_share_recomputed"],
            "floor_contribution": component_floor_share(r, total_us),
            "is_norms": bool(r.get("is_norms", False)),
        })

    # cross-check: floor_combined == sum of locus exposed / total.
    floor_via_exposed = sum(r["exposed_us"] for r in locus_rows) / total_us
    floor_roundtrip_resid = abs(floor_combined - floor_via_exposed)

    # ---------- step 3: verdict vs lambda=1 budget + compliant ceiling ---------- #
    budget = BUDGET_LAMBDA1_FRAC_213
    identity_lane_alive = bool(floor_combined <= budget + TOL_EXACT)
    sdpa_alone_busts_budget = bool(floor_b_sdpa > budget)
    # compliant ceiling TPS = lambda1_ceil * (1 - floor) | identity == 1.0.
    def compliant_ceiling(floor: float) -> float:
        return LAMBDA1_CEIL * (1.0 - floor)
    ceiling_at_floor = compliant_ceiling(floor_combined)           # ~469.7
    ceiling_at_zero = compliant_ceiling(0.0)                       # 520.9527 (round-trip)
    ceiling_clears_500 = bool(ceiling_at_floor >= TARGET_TPS)      # False
    ceiling_holds_frontier = bool(ceiling_at_floor >= OFFICIAL_TPS)  # False

    # operative fern #325 composition note (step inflated by (1+floor)); reported,
    # NOT the headline -- the (1-floor) ceiling is the upper-bound view.
    operative_tps_625 = K_CAL * (ET_BUILT_611 / (1.0 + floor_combined))   # K_cal*E[T]/(1+floor)

    # ---------- sensitivity: the dominant uncertainty (SDPA penalty-law curvature) ---------- #
    # The BW-gap floor charges the FULL forgone slack (the pessimistic end of the
    # ideal-kernel range). An ideal hand-written kernel that recovers parallelism on
    # non-reduction axes (heads / vocab-tiles) pays only a fraction phi of the slack.
    # Break-even phi* where floor*phi == budget -> flips (C) -> alive.
    phi_star_recovery_breakeven = budget / floor_combined          # ~0.745
    recovery_needed_to_revive = 1.0 - phi_star_recovery_breakeven  # ~0.255 (>25.5% recovery)
    # convex-law illustration pi=(1-u)^2 (a more-ideal kernel): optimistic sub-floor.
    floor_convex_sub = sum(
        r["step_share_recomputed"] * (bw_gap_penalty(r["bw_util"]) ** 2)
        for r in locus_rows if r["bw_util"] is not None)
    convex_revives = bool(floor_convex_sub <= budget)

    # re-scoping #216: the correctly-scoped bf16 floor vs the mis-scoped int4 floor.
    rescope_lift_vs_216_misscoped = floor_combined / CUSTOM_KERNEL_FLOOR_MISSCOPED_FRAC_216  # ~10.4x
    floor_below_216_offtheshelf = bool(floor_combined <= OFFTHESHELF_VERIFYGEMM_FRAC_216)    # True
    floor_nonneg = bool(floor_combined >= 0.0)

    # ---------- step 4: the decisive three-way outcome ---------- #
    outcome = three_way_outcome(floor_combined, budget, c326_config_only)
    outcome_is_dead = outcome.startswith("C_")
    # consistency: (C) iff floor>budget; exhaustive over (floor, c326) plane.
    outcome_consistent_with_floor = bool(outcome_is_dead == (floor_combined > budget + TOL_EXACT))

    light = "RED" if outcome_is_dead else ("YELLOW" if "B_" in outcome or "AB_" in outcome
                                           else "GREEN")
    verdict = _verdict(floor_combined, budget, floor_b_sdpa, sdpa_alone_busts_budget,
                       identity_lane_alive, phi_star_recovery_breakeven, ceiling_at_floor)
    handoff = _handoff(floor_combined, budget, floor_b_sdpa, outcome,
                       phi_star_recovery_breakeven, recovery_needed_to_revive,
                       rescope_lift_vs_216_misscoped, ceiling_at_floor)

    return {
        "step1_partition": {
            "source_card": "denken #291 verify_compute_hideability (measured M=8 verify, A10G)",
            "total_verify_us": total_us,
            "int4_body_components": [r["component"] for r in int4_rows],
            "bf16_locus_components": [r["component"] for r in locus_rows],
            "int4_body_step_share": int4_body_step_share,
            "bf16_locus_step_share": bf16_locus_step_share,
            "mlp_step_share": mlp_step_share,
            "int4_mlp_excluded": True,
            "partition_exhaustive": partition_exhaustive,
            "pct_vs_us_max_resid": pct_vs_us_max_resid,
            "exposed_eq_bwgap_max_resid": exposed_identity_max_resid,
            "component_shares": {r["component"]: r["step_share_recomputed"] for r in rows},
            "component_bw_util": {r["component"]: r["bw_util"] for r in rows},
        },
        "step2_penalty_model": {
            "model": "pi(u) = 1 - BW_util  (above-roofline exposed slack = forgone "
                     "reduction parallelism); floor_component = step_share * pi(u) "
                     "= measured exposed_us / total_us",
            "monotone_in_one_minus_bwutil": True,
            "per_component": per_component,
            "floor_a_lm_head_only": floor_a_lm_head,
            "floor_b_sdpa_only": floor_b_sdpa,
            "floor_norms_io_residual": floor_norms,
            "floor_combined": floor_combined,
            "floor_via_exposed_us": floor_via_exposed,
            "floor_roundtrip_resid": floor_roundtrip_resid,
        },
        "step3_verdict_vs_budget": {
            "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
            "budget_lambda1_frac_213": budget,
            "first_principles_floor_overhead": floor_combined,
            "identity_lane_alive_with_custom_kernel": identity_lane_alive,
            "sdpa_alone_busts_budget": sdpa_alone_busts_budget,
            "compliant_ceiling_tps_at_floor": ceiling_at_floor,
            "compliant_ceiling_tps_at_zero": ceiling_at_zero,
            "ceiling_clears_500": ceiling_clears_500,
            "ceiling_holds_frontier_481p53": ceiling_holds_frontier,
            "operative_tps_via_1plusfloor_at_E611": operative_tps_625,
            "lambda1_ceil": LAMBDA1_CEIL,
        },
        "step3b_sensitivity": {
            "note": "the BW-gap floor charges the FULL forgone slack (pessimistic end "
                    "of the ideal-kernel range); the dominant uncertainty is the SDPA "
                    "penalty-law curvature -- how much split-KV parallelism an ideal "
                    "deterministic kernel recovers on non-reduction axes.",
            "phi_star_recovery_breakeven": phi_star_recovery_breakeven,
            "recovery_fraction_needed_to_revive": recovery_needed_to_revive,
            "floor_convex_sub_optimistic": floor_convex_sub,
            "convex_law_revives_lane": convex_revives,
            "custom_kernel_floor_misscoped_216": CUSTOM_KERNEL_FLOOR_MISSCOPED_FRAC_216,
            "rescope_lift_vs_216_misscoped": rescope_lift_vs_216_misscoped,
            "floor_below_216_offtheshelf_anchor": floor_below_216_offtheshelf,
            "offtheshelf_216_frac": OFFTHESHELF_VERIFYGEMM_FRAC_216,
            "floor_nonneg": floor_nonneg,
        },
        "step4_three_way": {
            "c326_config_only_ceiling": c326_config_only,
            "outcome": outcome,
            "outcome_light": light,
            "outcome_is_dead": outcome_is_dead,
            "outcome_consistent_with_floor": outcome_consistent_with_floor,
            "A_alive_with_existing_knobs": "floor<=budget AND #326<=budget",
            "B_alive_needs_custom_kernel": "floor<=budget < #326 config-only",
            "C_dead_even_with_custom_kernel": "floor>budget",
        },
        "context": {
            "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS,
            "m8_divergence_pct_232": M8_DIVERGENCE_PCT_232,
            "verify_gemm_cost_share_of_step_216": VERIFY_GEMM_COST_SHARE_OF_STEP_216,
            "K_cal": K_CAL, "step_us": STEP_US, "tau": TAU, "E_T_built_611": ET_BUILT_611,
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(floor: float, budget: float, sdpa: float, sdpa_busts: bool,
             alive: bool, phi_star: float, ceiling: float) -> str:
    head = ("IDENTITY-LANE-DEAD-AT-BWGAP-FLOOR" if not alive
            else "IDENTITY-LANE-ALIVE-AT-BWGAP-FLOOR")
    sd = ("SDPA-ALONE-BUSTS-BUDGET" if sdpa_busts else "SDPA-WITHIN-BUDGET")
    return (f"{head}: first-principles bf16 lm_head+attn reduction floor = "
            f"{floor*100:.3f}% of step vs the lambda=1 budget {budget*100:.3f}% "
            f"-> {'DEAD (C)' if not alive else 'alive'}; binding term is the bf16 "
            f"SDPA @ 34.9% BW ({sdpa*100:.3f}% alone, {sd}). Compliant ceiling caps "
            f"at {ceiling:.1f} TPS (< 500 and < 481.53 frontier). NOT robust: a custom "
            f"kernel recovering > {(1-phi_star)*100:.1f}% of forgone split-KV parallelism "
            f"flips it to alive-needs-kernel (B). 0 TPS; analytic; UNBUILT + human-gated; "
            f"NOT a launch.")


def _handoff(floor: float, budget: float, sdpa: float, outcome: str,
             phi_star: float, recovery_needed: float, rescope_lift: float,
             ceiling: float) -> str:
    state = ("dead" if outcome.startswith("C_")
             else ("alive-needs-custom-kernel" if "B_" in outcome else "alive-with-knobs"))
    return (
        f"the hand-written batch-invariant bf16 lm_head+attn reduction floor is "
        f"{floor*100:.2f}% of step (vs the lambda=1 budget {budget*100:.2f}%), so the "
        f"strict-compliant identity lane is {state} -- the binding term is the bf16 SDPA "
        f"@ 34.9% BW, whose above-roofline forgone-parallelism slack ({sdpa*100:.2f}% of "
        f"step) ALONE exceeds the entire budget. This bridges wirbel #326's config-only "
        f"measurement from below (first-principles LOWER bracket) and re-scopes wirbel "
        f"#216's mis-derived 0.95% int4-split-K floor UP by ~{rescope_lift:.1f}x to the "
        f"correct bf16 lm_head+attn locus. It feeds fern #325's joint envelope the "
        f"identity-half step-inflation input (compliant ceiling {ceiling:.0f} TPS at the "
        f"floor). Dominant uncertainty: the BW-gap floor charges the FULL forgone slack; an "
        f"ideal kernel recovering > {recovery_needed*100:.0f}% of the SDPA split-KV "
        f"parallelism on non-reduction axes would flip the verdict to alive-needs-kernel. "
        f"0 TPS; analytic over banked numbers; BASELINE 481.53 unchanged; the custom kernel "
        f"is UNBUILT + human-approval-gated. NOT a launch. NOT a build. NOT a served-file change."
    )


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2 = syn["step1_partition"], syn["step2_penalty_model"]
    s3, s3b, s4 = (syn["step3_verdict_vs_budget"], syn["step3b_sensitivity"],
                   syn["step4_three_way"])
    checks: dict[str, bool] = {}

    # (a) bf16-locus partition reproduces the banked step shares; int4 MLP excluded.
    checks["a_mlp_share_is_0p661"] = abs(s1["mlp_step_share"] - 0.661) < 1.5e-3
    checks["a_sdpa_share_is_0p145"] = abs(s1["component_shares"]["sdpa"] - 0.145) < 1e-3
    checks["a_lmhead_share_is_0p024"] = abs(s1["component_shares"]["lm_head"] - 0.0236) < 1e-3
    checks["a_int4_mlp_excluded"] = bool(
        "gate_up_proj" in s1["int4_body_components"]
        and "down_proj" in s1["int4_body_components"]
        and "gate_up_proj" not in s1["bf16_locus_components"])
    checks["a_locus_is_lmhead_sdpa_norms"] = bool(
        set(s1["bf16_locus_components"]) == {"sdpa", "lm_head", "io_residual"})
    checks["a_partition_exhaustive"] = bool(s1["partition_exhaustive"])
    checks["a_pct_vs_us_consistent"] = bool(s1["pct_vs_us_max_resid"] < 1e-4)
    # the penalty model IS the measured BW-gap (exposed_us == us*(1-u)).
    checks["a_penalty_is_measured_bwgap"] = bool(s1["exposed_eq_bwgap_max_resid"] < TOL_EXACT)

    # (b) penalty model monotone in (1 - BW_util) per component.
    bwutil_rows = [(r["bw_util"], bw_gap_penalty(r["bw_util"]))
                   for r in COMPONENT_ROWS_291 if r["bw_util"] is not None]
    bwutil_rows.sort(key=lambda t: (1.0 - t[0]))     # ascending (1 - u)
    pis = [pi for _, pi in bwutil_rows]
    checks["b_penalty_monotone_in_one_minus_bwutil"] = bool(
        all(pis[i] <= pis[i + 1] + 1e-12 for i in range(len(pis) - 1)))
    # SDPA (low BW) costlier per-unit than lm_head (near-roofline).
    pc = {p["component"]: p for p in s2["per_component"]}
    checks["b_sdpa_pricier_than_lmhead"] = bool(pc["sdpa"]["penalty_pi"] > pc["lm_head"]["penalty_pi"])
    checks["b_floor_roundtrips_via_exposed"] = bool(s2["floor_roundtrip_resid"] < TOL_EXACT)

    # (c) floor <= #216 off-the-shelf 31.41% upper anchor AND >= 0.
    checks["c_floor_below_216_offtheshelf"] = bool(s3b["floor_below_216_offtheshelf_anchor"])
    checks["c_floor_nonneg"] = bool(s3b["floor_nonneg"])
    checks["c_floor_below_31p41"] = bool(
        s3["first_principles_floor_overhead"] <= OFFTHESHELF_VERIFYGEMM_FRAC_216)

    # (d) lambda-budget 7.332% imported <= 1e-6 (reproduces #213's banked value).
    checks["d_budget_imported_exact"] = abs(
        s3["budget_lambda1_pct_213"] - 7.331808522875782) < TOL_BUDGET
    checks["d_budget_frac_consistent"] = abs(
        s3["budget_lambda1_frac_213"] - 7.331808522875782 / 100.0) < TOL_BUDGET

    # (e) compliant-ceiling formula round-trips 520.95 at floor=0.
    checks["e_ceiling_roundtrips_520p95"] = abs(
        s3["compliant_ceiling_tps_at_zero"] - LAMBDA1_CEIL) < TOL_EXACT
    checks["e_ceiling_at_floor_below_500"] = bool(not s3["ceiling_clears_500"])

    # (f) the three-way (A)/(B)/(C) partition is exhaustive and consistent.
    checks["f_outcome_consistent_with_floor"] = bool(s4["outcome_consistent_with_floor"])
    checks["f_partition_exhaustive_grid"] = _partition_is_exhaustive(s3["budget_lambda1_frac_213"])
    checks["f_dead_iff_floor_gt_budget"] = bool(
        s4["outcome_is_dead"] == (s3["first_principles_floor_overhead"]
                                  > s3["budget_lambda1_frac_213"] + TOL_EXACT))

    # (g) NaN-clean over reported scalars.
    scalars = [s1["bf16_locus_step_share"], s1["int4_body_step_share"],
               s2["floor_a_lm_head_only"], s2["floor_b_sdpa_only"], s2["floor_combined"],
               s3["first_principles_floor_overhead"], s3["compliant_ceiling_tps_at_floor"],
               s3b["phi_star_recovery_breakeven"], s3b["rescope_lift_vs_216_misscoped"]]
    checks["g_nan_clean"] = all(math.isfinite(float(x)) for x in scalars)

    # (h) constants imported EXACT.
    checks["h_constants_exact"] = bool(
        OFFICIAL_TPS == 481.53
        and abs(LAMBDA1_CEIL - 520.9527323111674) < TOL_EXACT
        and K_CAL == 125.268
        and STEP_US == 1218.2
        and abs(BUDGET_LAMBDA1_FRAC_213 - 0.07332) < 1e-4   # 7.332% display
        and TOTAL_VERIFY_US_291 == 5348.1268310546875
        and abs(sum(r["pct_of_verify"] for r in COMPONENT_ROWS_291) - 100.0) < 1e-6)

    # the leg carries the 0-TPS + analytic + scope caveats.
    hl = syn["handoff_line"]
    checks["h_carries_caveats"] = bool(
        "0 TPS" in hl and "analytic" in hl and "NOT a launch" in hl
        and "NOT a build" in hl and "UNBUILT" in hl and "human-approval-gated" in hl)

    gate = bool(all(checks.values()))
    return {"bi_reduction_floor_self_test_passes": gate, "checks": checks}


def _partition_is_exhaustive(budget: float) -> bool:
    """Over a grid of (floor, c326), exactly one of A/B/C (or the pending-326
    branch) fires -- the three-way partition leaves no gap and no overlap."""
    grid = [budget * f for f in (0.0, 0.5, 0.99, 1.0, 1.01, 1.5, 2.0)]
    for floor in grid:
        for c326 in (None, budget * 0.5, budget * 1.5):
            out = three_way_outcome(floor, budget, c326)
            tags = [out.startswith("C_"), out.startswith("A_"),
                    out.startswith("B_"), out.startswith("AB_")]
            if sum(tags) != 1:
                return False
    return True


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
    s1, s2 = syn["step1_partition"], syn["step2_penalty_model"]
    s3, s3b, s4 = (syn["step3_verdict_vs_budget"], syn["step3b_sensitivity"],
                   syn["step4_three_way"])
    print("\n" + "=" * 94, flush=True)
    print("EAGLE-3 BATCH-INVARIANT bf16 lm_head+attn REDUCTION FLOOR (PR #327, denken) -- 0 GPU",
          flush=True)
    print("=" * 94, flush=True)
    print("  (1) PARTITION  (denken #291 measured M=8 verify; int4 body bit-exact across M / #288)",
          flush=True)
    print(f"      int4 body (EXCLUDED)   = {s1['int4_body_components']}  "
          f"share {s1['int4_body_step_share']*100:.2f}%  (MLP {s1['mlp_step_share']*100:.2f}%)",
          flush=True)
    print(f"      bf16 locus (in scope)  = {s1['bf16_locus_components']}  "
          f"share {s1['bf16_locus_step_share']*100:.2f}%", flush=True)
    print(f"      exposed_us == us*(1-BW_util) resid = {s1['exposed_eq_bwgap_max_resid']:.2e}  "
          f"(penalty model IS the measured BW-gap)", flush=True)
    print("-" * 94, flush=True)
    print("  (2) BW-GAP PENALTY  pi(u)=1-BW_util  (floor = step_share * pi = exposed_us/total)",
          flush=True)
    for p in s2["per_component"]:
        bw = "n/a " if p["bw_util"] is None else f"{p['bw_util']*100:4.1f}%"
        print(f"      {p['component']:12s} BW {bw}  pi {p['penalty_pi']*100:5.1f}%  "
              f"share {p['step_share']*100:5.2f}%  -> floor {p['floor_contribution']*100:.3f}%"
              f"{'  [norms]' if p['is_norms'] else ''}", flush=True)
    print(f"      (a) lm_head only = {s2['floor_a_lm_head_only']*100:.3f}%   "
          f"(b) SDPA only = {s2['floor_b_sdpa_only']*100:.3f}%   "
          f"norms = {s2['floor_norms_io_residual']*100:.3f}%", flush=True)
    print(f"      (c) COMBINED FLOOR = {s2['floor_combined']*100:.3f}% of step", flush=True)
    print("-" * 94, flush=True)
    print("  (3) VERDICT vs lambda=1 BUDGET (#213)", flush=True)
    print(f"      first_principles_floor_overhead = {s3['first_principles_floor_overhead']*100:.3f}%  "
          f"vs budget {s3['budget_lambda1_pct_213']:.3f}%", flush=True)
    print(f"      identity_lane_alive_with_custom_kernel = "
          f"{s3['identity_lane_alive_with_custom_kernel']}   "
          f"(SDPA alone busts budget = {s3['sdpa_alone_busts_budget']})", flush=True)
    print(f"      compliant ceiling TPS = {s3['compliant_ceiling_tps_at_floor']:.1f} "
          f"(at floor) / {s3['compliant_ceiling_tps_at_zero']:.2f} (at 0; round-trips 520.95)  "
          f"clears500={s3['ceiling_clears_500']} holds481.53={s3['ceiling_holds_frontier_481p53']}",
          flush=True)
    print("-" * 94, flush=True)
    print("  (3b) DOMINANT UNCERTAINTY  (SDPA penalty-law curvature)", flush=True)
    print(f"      phi* recovery break-even = {s3b['phi_star_recovery_breakeven']*100:.1f}%  "
          f"-> need > {s3b['recovery_fraction_needed_to_revive']*100:.1f}% parallelism recovery "
          f"to revive (B)", flush=True)
    print(f"      convex sub-floor pi=(1-u)^2 = {s3b['floor_convex_sub_optimistic']*100:.3f}%  "
          f"revives lane = {s3b['convex_law_revives_lane']}", flush=True)
    print(f"      re-scope vs #216 mis-scoped 0.95% int4 floor = "
          f"x{s3b['rescope_lift_vs_216_misscoped']:.1f}  (below #216 off-the-shelf 31.41% = "
          f"{s3b['floor_below_216_offtheshelf_anchor']})", flush=True)
    print("-" * 94, flush=True)
    print("  (4) DECISIVE THREE-WAY OUTCOME", flush=True)
    print(f"      outcome = {s4['outcome']}  [{s4['outcome_light']}]  "
          f"(#326 config-only ceiling = {s4['c326_config_only_ceiling']})", flush=True)
    print("-" * 94, flush=True)
    print(f"  PRIMARY bi_reduction_floor_self_test_passes = "
          f"{st['bi_reduction_floor_self_test_passes']}", flush=True)
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
        print(f"[bi-reduction-floor] wandb logging unavailable: {exc}", flush=True)
        return None

    syn, st = payload["synthesis"], payload["self_test"]
    s1, s2 = syn["step1_partition"], syn["step2_penalty_model"]
    s3, s3b, s4 = (syn["step3_verdict_vs_budget"], syn["step3b_sensitivity"],
                   syn["step4_three_way"])
    run = init_wandb_run(
        job_type="eagle3-bi-reduction-floor",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-bi-reduction-floor", "batch-invariant", "greedy-identity",
              "bf16-lm_head-sdpa", "kernel-floor", "validity", "zero-tps"],
        config={
            "pr": 327, "analysis_only": True,
            "total_verify_us_291": TOTAL_VERIFY_US_291,
            "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
            "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL, "step_us": STEP_US,
            "official_tps": OFFICIAL_TPS, "target_tps": TARGET_TPS,
            "offtheshelf_216_pct": OFFTHESHELF_VERIFYGEMM_PCT_216,
            "custom_kernel_floor_misscoped_216_pct": CUSTOM_KERNEL_FLOOR_MISSCOPED_PCT_216,
            "m8_divergence_pct_232": M8_DIVERGENCE_PCT_232,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[bi-reduction-floor] wandb: no run (no WANDB_API_KEY/mode) -- skipping",
              flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + the two TEST metrics.
        "bi_reduction_floor_self_test_passes":
            int(bool(st["bi_reduction_floor_self_test_passes"])),
        "first_principles_floor_overhead": s3["first_principles_floor_overhead"],
        "identity_lane_alive_with_custom_kernel":
            int(bool(s3["identity_lane_alive_with_custom_kernel"])),
        # step 1 partition.
        "bf16_locus_step_share": s1["bf16_locus_step_share"],
        "int4_body_step_share": s1["int4_body_step_share"],
        "mlp_step_share": s1["mlp_step_share"],
        "exposed_eq_bwgap_max_resid": s1["exposed_eq_bwgap_max_resid"],
        "pct_vs_us_max_resid": s1["pct_vs_us_max_resid"],
        # step 2 per-component floor.
        "floor_a_lm_head_only": s2["floor_a_lm_head_only"],
        "floor_b_sdpa_only": s2["floor_b_sdpa_only"],
        "floor_norms_io_residual": s2["floor_norms_io_residual"],
        "floor_combined": s2["floor_combined"],
        "floor_roundtrip_resid": s2["floor_roundtrip_resid"],
        # step 3 verdict.
        "budget_lambda1_pct_213": s3["budget_lambda1_pct_213"],
        "budget_lambda1_frac_213": s3["budget_lambda1_frac_213"],
        "sdpa_alone_busts_budget": int(bool(s3["sdpa_alone_busts_budget"])),
        "compliant_ceiling_tps_at_floor": s3["compliant_ceiling_tps_at_floor"],
        "compliant_ceiling_tps_at_zero": s3["compliant_ceiling_tps_at_zero"],
        "ceiling_clears_500": int(bool(s3["ceiling_clears_500"])),
        "ceiling_holds_frontier_481p53": int(bool(s3["ceiling_holds_frontier_481p53"])),
        "operative_tps_via_1plusfloor_at_E611": s3["operative_tps_via_1plusfloor_at_E611"],
        # step 3b sensitivity.
        "phi_star_recovery_breakeven": s3b["phi_star_recovery_breakeven"],
        "recovery_fraction_needed_to_revive": s3b["recovery_fraction_needed_to_revive"],
        "floor_convex_sub_optimistic": s3b["floor_convex_sub_optimistic"],
        "convex_law_revives_lane": int(bool(s3b["convex_law_revives_lane"])),
        "rescope_lift_vs_216_misscoped": s3b["rescope_lift_vs_216_misscoped"],
        "floor_below_216_offtheshelf_anchor": int(bool(s3b["floor_below_216_offtheshelf_anchor"])),
        # step 4 three-way.
        "outcome_code": {"A_alive_with_existing_knobs": 3,
                         "B_alive_needs_custom_kernel": 2,
                         "AB_alive_pending_326_config_only": 1,
                         "C_dead_even_with_custom_kernel": 0}[s4["outcome"]],
        "outcome_is_dead": int(bool(s4["outcome_is_dead"])),
        "outcome_consistent_with_floor": int(bool(s4["outcome_consistent_with_floor"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_bi_reduction_floor_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[bi-reduction-floor] wandb logged (run {rid})", flush=True)
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
                    default="eagle3-bi-reduction-floor")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize(c326_config_only=args.c326_config_only)
    st = self_test(syn)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 327, "agent": "denken",
        "kind": "eagle3-bi-reduction-floor", "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[bi-reduction-floor] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_bi_reduction_floor_results.json"

    wid = None
    if not args.no_wandb:
        wid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = wid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[bi-reduction-floor] wrote {out_path}  (wandb run {wid})", flush=True)

    if args.self_test:
        ok = st["bi_reduction_floor_self_test_passes"] and payload["nan_clean"]
        print(f"[bi-reduction-floor] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
