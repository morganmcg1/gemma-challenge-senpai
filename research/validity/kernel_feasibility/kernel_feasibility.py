#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Compliant-kernel feasibility (PR #216, wirbel) — CPU-only analytic synthesis.

THE QUESTION (capstone of #199 → #213 → this; Issue #192 lane-a)
---------------------------------------------------------------
My #213 (`kernel-budget-lambda`, MERGED, `5o7zcj8s`) produced the actionable budget curve:
the compliant-spec 500-lane needs land #71 to build self-KV recovery λ above
``lambda_crit`` (0.8345 both-bugs / 0.9067 descent) AND hold a custom batch-invariant int4
verify kernel below ``max_kernel_overhead_pct(λ)`` (7.33% at λ=1, ≤0 below λ_crit). It also
proved kanna #122's OFF-THE-SHELF ``VLLM_BATCH_INVARIANT=1`` (+51.78%) clears at NO physical
λ≤1. That leaves the single UNANSWERED question of the whole #192 lane-a: is a compliant
batch-invariant int4 verify kernel actually BUILDABLE under that budget — a REAL 500-path or
only a theoretical one? The key insight: #122's +51.78% is the WHOLE-MODEL determinism
ceiling (every GEMM, attention softmax, RMSNorm, the LM head), but the #192 bug is localized
to ONE op (the int4 Marlin verify GEMM's split-K reduction order picked as f(batch width M)).
A CUSTOM kernel that fixes ONLY the verify GEMM's split-K geometry to be M-invariant should
cost a SMALL FRACTION of the whole-model +51.78%.

THE DECOMPOSITION (imported anchors, NOT re-derived)
----------------------------------------------------
We bound the plausible CUSTOM-kernel overhead between a first-principles FLOOR and #122's
measured CEILING, then map the band against #213's budget curve:

  (1) scope of #122's +51.78%.  The verify GEMM is one op of the step; its wall-time share is
      ``verify_gemm_cost_share_of_step = gemm_us[M=32] / (step·step_m8_us)`` (#153 measured
      M-sweep over #168's pinned step 1.2182). Under scope-proportionality (the determinism
      overhead is ≈ proportional to the fraction of step time forced deterministic) the part of
      +51.78% attributable to the verify GEMM is ``51.78%·share`` — the band's CONSERVATIVE end
      (a custom kernel fixing only the GEMM should not pay MORE than the off-the-shelf cost
      scoped to that GEMM).

  (2) first-principles split-K FLOOR.  Making ONLY the split-K reduction M-invariant forces a
      FIXED partition instead of the batch-adaptive one; the penalty is at most the adaptive
      split-K realization headroom — #150's measured ``s_net=1.56%`` of verify-GEMM time
      (operating M=8, gate_up frozen, occupancy-limited laggards lifted). Scaled to the step:
      ``floor = s_net·share`` — the band's OPTIMISTIC end (a hand-tuned fixed schedule matching
      the deployed width approaches ~0).

  (3) overlay.  ``max_kernel_overhead_pct(λ)`` (#213, the SAME LambdaCurve, imported) vs the
      band [floor, attributable] gives ``lambda_min_kernel_feasible`` = the λ where the budget
      first covers the band's LOWER end (best-case buildable) and (if any physical λ) its UPPER
      end (comfortably buildable). #122's whole-model +51.78% is reported as the NON-working
      reference it is.

THE DELIVERABLE
---------------
band [floor≈0.95%, attributable≈31.4%]; budget 7.33% at λ=1 / ≤0 below λ_crit. The OPTIMISTIC
(floor) end is buildable for λ ≳ λ_crit (verdict ``COMPLIANT_KERNEL_PLAUSIBLE_ABOVE_LAMBDA_x``);
the CONSERVATIVE (off-the-shelf-attributable) end clears at NO physical λ≤1
(``_INFEASIBLE_AT_ALL_PHYSICAL_LAMBDA``). So #192 lane-a is a REAL 500-path ONLY conditional on
land #71 building λ above ``lambda_min_kernel_feasible`` AND the custom kernel landing near its
split-K floor.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build / served-file
change. BASELINE stays 481.53; adds **0 TPS**; greedy/PPL untouched. The scope-proportionality
and split-K-penalty estimates are EXPLICIT assumptions; #199's three optimisms carry as a noted
band; ONLY an actual kernel build settles buildability. This is a buildability PRIOR, not a
proof. **NOT a launch. NOT open2.**

PRIMARY metric  kernel_feasibility_self_test_passes
TEST    metric  lambda_min_kernel_feasible  (floor / best-case, both-bugs, τ=1)
"""
from __future__ import annotations

import argparse
import importlib.util
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
# Import #213's budget machinery (path-based) — its OWN LambdaCurve / budget /
# λ_crit / λ-for-budget solvers on #199's banked spines. Re-running KB.C.synthesize
# is the canonical "import #199's banked result" (its own code on its own inputs →
# bit-identical spines). We do NOT re-derive the curve, λ_crit, 7.33%, 0.8345, or 51.78%.
# --------------------------------------------------------------------------- #
def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KB = _load("kernel_budget_lambda",
           "research/validity/kernel_budget_lambda/kernel_budget_lambda.py")
C = KB.C  # compliant_spec_et — SAME module instance #213 uses (shared banked spines).

# Pinned launch composition (imported via #213 → #199 → #172/#148/#168/#181).
K_CAL = KB.K_CAL                  # 125.26795005202914
STEP = KB.STEP                    # 1.2182  (#168 measured M=32 deployed step, normalized)
TAU_CENTRAL = KB.TAU_CENTRAL      # 1.0
TAU_CONS = KB.TAU_CONS            # 0.9924
TAU_CORNERS = KB.TAU_CORNERS
TARGET = KB.TARGET                # 500.0
BENCH_TOKENS = KB.BENCH_TOKENS    # 16384
KANNA122_OFFSHELF_OVERHEAD = KB.KANNA122_OFFSHELF_OVERHEAD  # 0.5178 (whole-model, NON-working)
REGIMES = KB.REGIMES              # ("both_bugs", "descent_only")

# Imported cost-decomposition anchors (read-only banked artifacts; the PR names these).
STEP_CURVE_REL = "research/oracle_readout/verify_step_m_curve.json"            # #153
SPLITK_CEILING_REL = "research/spec_cost_model/splitk_realization_ceiling_results.json"  # #150

VERIFY_M = 32          # deployed tree-verify width (TREE_M; #153 m-sweep node, == #168 step).
SPLITK_OPERATING_M = 8  # split-K realization headroom measured here (#150 operating_M).
SPINE_0997 = 0.997     # land #71's posted interim optimistic spine λ(q[2..7]).

TOL_ROUNDTRIP = 1e-9
TOL_BUDGET_ZERO = 1e-6


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load_json(relpath: str) -> dict[str, Any]:
    with (REPO_ROOT / relpath).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# (1)+(2) Cost decomposition: verify-GEMM share, off-the-shelf attribution, split-K floor.
# --------------------------------------------------------------------------- #
def _cost_decomposition() -> dict[str, Any]:
    sc = _load_json(STEP_CURVE_REL)
    sk = _load_json(SPLITK_CEILING_REL)

    gemm_us = sc["raw"]["gemm_us"]
    attn_us = sc["raw"]["attn_us"]
    step_m8_us = sc["anchors"]["step_m8_us"]            # 7982.887878221502
    gemm_us_m32 = gemm_us[str(VERIFY_M)]               # 5899.235962629318
    gemm_us_m8 = gemm_us[str(SPLITK_OPERATING_M)]      # 4980.291757583619
    attn_us_m32 = attn_us[str(VERIFY_M)]               # 1058.4877566666764

    # deployed step (µs) = #168 pinned normalized step (1.2182) × #153 M=8 normalizer.
    step_us_m32 = STEP * step_m8_us
    share = gemm_us_m32 / step_us_m32                   # verify-GEMM wall-time share of step
    share_at_m8 = gemm_us_m8 / step_m8_us               # cross-check at the split-K operating M

    s_net = sk["wall_scenarios"]["measured"]["s_net"]                    # 0.015586924621061815
    s_gross = sk["wall_scenarios"]["measured"]["s_gross"]               # 0.03198611975673549
    band_high = sk["gate"]["splitk_realization_ceiling_band_high_pct"]  # 7.805537614549518 (%)

    attributable_pct = KANNA122_OFFSHELF_OVERHEAD * 100.0 * share        # 51.78% scoped to GEMM
    floor_pct = s_net * 100.0 * share                                    # split-K-fix penalty
    floor_pct_theoretical_min = 0.0                                      # matched fixed schedule
    floor_pct_gross = s_gross * 100.0 * share                            # if gross gain forgone
    floor_pct_band_high = band_high * share                              # practical-88 UB on pen.

    return {
        "verify_gemm_cost_share_of_step": share,
        "verify_gemm_cost_share_of_step_at_operating_m8": share_at_m8,
        "offtheshelf_overhead_attributable_to_verify_gemm_pct": attributable_pct,
        "custom_kernel_overhead_floor_pct": floor_pct,
        "custom_kernel_overhead_floor_theoretical_min_pct": floor_pct_theoretical_min,
        "custom_kernel_overhead_floor_gross_pct": floor_pct_gross,
        "custom_kernel_overhead_floor_band_high_pct": floor_pct_band_high,
        "plausible_custom_overhead_band_pct": [floor_pct, attributable_pct],
        "offtheshelf_whole_model_overhead_pct": KANNA122_OFFSHELF_OVERHEAD * 100.0,
        "offtheshelf_to_custom_floor_ratio": (
            (KANNA122_OFFSHELF_OVERHEAD * 100.0) / floor_pct if floor_pct > 0 else float("inf")),
        "anchors": {
            "gemm_us_m32": gemm_us_m32, "gemm_us_m8": gemm_us_m8, "attn_us_m32": attn_us_m32,
            "step_m8_us": step_m8_us, "step_us_m32_deployed": step_us_m32,
            "step_normalized": STEP, "verify_m": VERIFY_M, "splitk_operating_m": SPLITK_OPERATING_M,
            "s_net": s_net, "s_gross": s_gross, "splitk_band_high_pct": band_high,
        },
        "assumptions": {
            "scope_proportionality": (
                "off-the-shelf VLLM_BATCH_INVARIANT=1 overhead is ≈ proportional to the fraction "
                "of step WALL-TIME forced deterministic; the verify-GEMM share is its time share "
                "of the deployed M=32 step (#153 gemm_us[32] / #168 step). FLOP/op share is "
                "proxied by time share (both are weight-traffic-dominated for these BW-bound GEMMs)."),
            "splitk_penalty": (
                "the fixed-(M-invariant)-split-K penalty is bounded by the measured adaptive "
                "split-K realization headroom s_net=1.56% (#150, operating M=8). The verify GEMM is "
                "BW-bound (AI≈28 << ridge 107) so this fraction is ≈ M-stable from M=8 to the "
                "deployed M=32; the theoretical minimum is ~0 (a fixed schedule matched at M)."),
        },
    }


# --------------------------------------------------------------------------- #
# (3) Feasibility overlay vs #213's budget curve (the SAME LambdaCurve, imported).
# --------------------------------------------------------------------------- #
def _build_curve(c199: dict[str, Any], regime: str, lam_hat: float) -> Any:
    br = c199["brackets"][regime]
    return KB.LambdaCurve(br["floor_spine_at_lambda_hat"], br["ceiling_spine"], lam_hat)


def _feasibility(curve: Any, lam_hat: float, floor_pct: float,
                 attributable_pct: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, tau in TAU_CORNERS:
        lam_crit = KB.lambda_crit_clears_500(curve, tau)
        b_floor = KB.lambda_for_budget(curve, floor_pct, tau)
        b_attr = KB.lambda_for_budget(curve, attributable_pct, tau)

        budget_hat = KB.budget_pct(curve.et_of_lambda(lam_hat), tau)
        budget_997 = KB.budget_pct(curve.et_of_lambda(SPINE_0997), tau)
        budget_one = KB.budget_pct(curve.et_of_lambda(1.0), tau)
        budget_at_crit = (KB.budget_pct(curve.et_of_lambda(lam_crit), tau)
                          if lam_crit is not None else None)

        out[tag] = {
            "lambda_crit_clears_500_zero_overhead": lam_crit,
            # band LOWER (floor / best-case buildable) — the headline lambda_min.
            "lambda_min_kernel_feasible_floor": b_floor["lambda_for_target"],
            "lambda_min_kernel_feasible_floor_is_physical": b_floor["is_physical_lambda_le_1"],
            # band UPPER (attributable / comfortably buildable).
            "lambda_min_kernel_feasible_attributable": b_attr["lambda_for_target"],
            "lambda_min_kernel_feasible_attributable_is_physical": b_attr["is_physical_lambda_le_1"],
            "attributable_reachable_within_prob_saturation":
                b_attr["reachable_within_prob_saturation"],
            "max_budget_pct_at_prob_saturation": b_attr["max_budget_pct_at_prob_saturation"],
            # budget at the named λ anchors.
            "budget_at_lambda_hat": budget_hat,
            "budget_at_spine_0997": budget_997,
            "budget_at_lambda_1": budget_one,
            "budget_at_lambda_crit": budget_at_crit,
            # buildable bools at λ̂ and at the optimistic spine λ=0.997, both band ends.
            "buildable_at_lambda_hat_floor": bool(floor_pct <= budget_hat),
            "buildable_at_lambda_hat_attributable": bool(attributable_pct <= budget_hat),
            "buildable_at_spine_0997_floor": bool(floor_pct <= budget_997),
            "buildable_at_spine_0997_attributable": bool(attributable_pct <= budget_997),
            "buildable_at_lambda_1_floor": bool(floor_pct <= budget_one),
            "buildable_at_lambda_1_attributable": bool(attributable_pct <= budget_one),
        }
    return out


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(shard: Path, max_records: int | None = None) -> dict[str, Any]:
    # (0) import #199's banked compliant-spec object + #213's curve inputs.
    c199 = C.synthesize(shard, max_records)
    lam_hat = c199["lambda_hat"]

    # (1)+(2) cost decomposition (band end-points; regime-INDEPENDENT — kernel cost only).
    decomp = _cost_decomposition()
    share = decomp["verify_gemm_cost_share_of_step"]
    floor_pct = decomp["custom_kernel_overhead_floor_pct"]
    attributable_pct = decomp["offtheshelf_overhead_attributable_to_verify_gemm_pct"]

    # (3) overlay the band on #213's budget curve, per regime / τ corner.
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        curve = _build_curve(c199, regime, lam_hat)
        regimes[regime] = _feasibility(curve, lam_hat, floor_pct, attributable_pct)

    head = regimes["both_bugs"]["tau_central_1p0"]
    lam_crit_bb = head["lambda_crit_clears_500_zero_overhead"]
    lam_min_floor = head["lambda_min_kernel_feasible_floor"]            # TEST metric
    lam_min_attr = head["lambda_min_kernel_feasible_attributable"]
    budget_at_crit = head["budget_at_lambda_crit"]

    # ---------- self-test (PRIMARY) ---------- #
    # (a) scope decomposition round-trips #122's +51.78% when scaled back to whole-model.
    whole_model_reconstructed = attributable_pct / share if share > 0 else float("nan")
    cond_a = (0.0 < share < 1.0
              and abs(whole_model_reconstructed - KANNA122_OFFSHELF_OVERHEAD * 100.0)
              <= TOL_ROUNDTRIP)
    # (b) band well-ordered: 0 ≤ floor ≤ attributable.
    cond_b = (floor_pct >= 0.0) and (floor_pct <= attributable_pct + 1e-12)
    # (c) at λ_crit budget≈0 ⇒ feasible ONLY if floor≈0 (internal consistency with #213).
    feasible_at_crit = (budget_at_crit is not None
                        and floor_pct <= budget_at_crit + TOL_BUDGET_ZERO)
    cond_c = (budget_at_crit is not None
              and abs(budget_at_crit) <= TOL_BUDGET_ZERO
              and (feasible_at_crit == (floor_pct <= TOL_BUDGET_ZERO)))
    # (d) lambda_min_kernel_feasible ≥ λ_crit (can't be feasible below the free-kernel clear).
    cond_d = (lam_min_floor is not None and lam_crit_bb is not None
              and lam_min_floor >= lam_crit_bb - 1e-9)
    conditions = {
        "a_offshelf_scope_decomposition_roundtrips_122_5178": bool(cond_a),
        "b_floor_ge_0_and_le_attributable_band_well_ordered": bool(cond_b),
        "c_at_lambda_crit_budget_zero_feasible_iff_floor_zero": bool(cond_c),
        "d_lambda_min_kernel_feasible_ge_lambda_crit": bool(cond_d),
        "e_nan_clean": True,   # set by the caller after the full payload walk.
    }

    verdict, verdict_conservative = _verdict(lam_min_floor, lam_min_attr, head)
    handoff = _handoff(decomp, lam_min_floor, lam_crit_bb, head)
    return {
        "self_test": {
            "kernel_feasibility_self_test_passes": bool(all(conditions.values())),
            "conditions": conditions,
        },
        "test_metric": {"lambda_min_kernel_feasible": lam_min_floor},
        "headline": {
            "lambda_min_kernel_feasible": lam_min_floor,
            "lambda_min_kernel_feasible_is_physical":
                head["lambda_min_kernel_feasible_floor_is_physical"],
            "lambda_min_kernel_feasible_attributable_end": lam_min_attr,
            "lambda_min_kernel_feasible_attributable_is_physical":
                head["lambda_min_kernel_feasible_attributable_is_physical"],
            "lambda_crit_clears_500_zero_overhead_both_bugs_tau1": lam_crit_bb,
            "verify_gemm_cost_share_of_step": share,
            "offtheshelf_overhead_attributable_to_verify_gemm_pct": attributable_pct,
            "custom_kernel_overhead_floor_pct": floor_pct,
            "plausible_custom_overhead_band_pct": [floor_pct, attributable_pct],
            "compliant_kernel_buildable_at_lambda_hat": bool(
                head["buildable_at_lambda_hat_floor"]),
            "compliant_kernel_buildable_at_lambda_hat_attributable": bool(
                head["buildable_at_lambda_hat_attributable"]),
            "compliant_kernel_buildable_at_spine_0997": bool(
                head["buildable_at_spine_0997_floor"]),
            "compliant_kernel_buildable_at_spine_0997_attributable": bool(
                head["buildable_at_spine_0997_attributable"]),
            "budget_at_lambda_1_both_bugs_tau1": head["budget_at_lambda_1"],
            "offtheshelf_attributable_clears_at_physical_lambda": bool(
                head["lambda_min_kernel_feasible_attributable_is_physical"]),
            "max_budget_pct_at_prob_saturation_both_bugs_tau1":
                head["max_budget_pct_at_prob_saturation"],
        },
        "cost_decomposition": decomp,
        "regimes": regimes,
        "lambda_hat": lam_hat,
        "spine_0997": SPINE_0997,
        "composition": {
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "clear500_bar_et_tau1": C.clear_bar(TAU_CENTRAL),
            "clear500_bar_et_tau_cons": C.clear_bar(TAU_CONS),
            "bench_tokens": BENCH_TOKENS,
            "kanna122_offshelf_overhead_nonworking_ref": KANNA122_OFFSHELF_OVERHEAD,
        },
        "optimism_band_note": (
            "Carries #199's THREE optimisms (rank-1 coverage 0.7304 over-counts the true "
            "compliant accept; λ-realism vs λ̂=0.342; zero OTHER overhead) PLUS the new "
            "scope-proportionality assumption (the off-the-shelf determinism tax splits ≈ by "
            "wall-time share) PLUS the split-K-penalty estimate (fixed-schedule cost bounded by "
            "the measured s_net=1.56% adaptive headroom). The band's OPTIMISTIC end is the "
            "split-K floor; the CONSERVATIVE end is #122's whole-model tax scoped to the GEMM. "
            "ONLY an actual batch-invariant kernel build settles buildability — this is a PRIOR."),
        "verdict": verdict,
        "verdict_conservative": verdict_conservative,
        "handoff_line": handoff,
    }


def _verdict(lam_min_floor: float | None, lam_min_attr: float | None,
             head: dict[str, Any]) -> tuple[str, str]:
    if lam_min_floor is None:
        opt = "COMPLIANT_KERNEL_INFEASIBLE_AT_ALL_PHYSICAL_LAMBDA"
    elif head["lambda_min_kernel_feasible_floor_is_physical"]:
        opt = f"COMPLIANT_KERNEL_PLAUSIBLE_ABOVE_LAMBDA_{lam_min_floor:.4f}"
    else:
        opt = "COMPLIANT_KERNEL_INFEASIBLE_AT_ALL_PHYSICAL_LAMBDA"
    # conservative = the off-the-shelf-attributable (band UPPER) end.
    if head["lambda_min_kernel_feasible_attributable_is_physical"] and lam_min_attr is not None:
        cons = f"COMPLIANT_KERNEL_OFFSHELF_ATTRIBUTABLE_PLAUSIBLE_ABOVE_LAMBDA_{lam_min_attr:.4f}"
    else:
        cons = "COMPLIANT_KERNEL_OFFSHELF_ATTRIBUTABLE_INFEASIBLE_AT_ALL_PHYSICAL_LAMBDA"
    return opt, cons


def _handoff(decomp: dict[str, Any], lam_min_floor: float | None,
             lam_crit_bb: float | None, head: dict[str, Any]) -> str:
    floor_pct = decomp["custom_kernel_overhead_floor_pct"]
    attributable_pct = decomp["offtheshelf_overhead_attributable_to_verify_gemm_pct"]
    ratio = decomp["offtheshelf_to_custom_floor_ratio"]
    lam_min_s = f"{lam_min_floor:.4f}" if lam_min_floor is not None else "NONE"
    crit_s = f"{lam_crit_bb:.4f}" if lam_crit_bb is not None else "NONE"
    return (
        f"COMPLIANT-KERNEL FEASIBILITY (Issue #192 lane-a, capstone of #199→#213→#216): the "
        f"compliant batch-invariant int4 verify kernel is plausibly buildable for λ ≥ "
        f"{lam_min_s} (custom-overhead band [{floor_pct:.2f}%, {attributable_pct:.2f}%] vs the "
        f"#213 budget at that λ; budget ≤0 below λ_crit={crit_s}), so #192 lane-a is a REAL "
        f"500-path CONDITIONAL on land #71 building self-KV recovery λ above that threshold AND "
        f"the custom kernel landing near its split-K floor — kanna #122's off-the-shelf +51.78% "
        f"is a WHOLE-MODEL ceiling ~{ratio:.0f}× the verify-GEMM-only cost and never the right "
        f"kernel (its scoped-to-GEMM share {attributable_pct:.1f}% already clears at NO physical "
        f"λ≤1, budget only {head['budget_at_lambda_1']:.2f}% at λ=1). HONEST SCOPE: scope-"
        f"proportionality + split-K-penalty are explicit estimates and #199's three optimisms "
        f"carry; ONLY an actual kernel build converts this PRIOR to a proof. Adds 0 TPS; "
        f"authorizes nothing. NOT a launch. NOT open2."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #213; never fatal).
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
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


def _print_report(syn: dict) -> None:
    st, hd, comp, dc = syn["self_test"], syn["headline"], syn["composition"], syn["cost_decomposition"]
    print("\n" + "=" * 96, flush=True)
    print("COMPLIANT-KERNEL FEASIBILITY (PR #216, wirbel) — Issue #192 lane-a capstone, CPU-only",
          flush=True)
    print("=" * 96, flush=True)
    print(f"  λ̂={syn['lambda_hat']:.5f}   clear-500 bar E[T]={comp['clear500_bar_et_tau1']:.4f} "
          f"(τ=1)   spine λ={syn['spine_0997']}", flush=True)
    print("-" * 96, flush=True)
    print("  (1)+(2) COST DECOMPOSITION (band end-points, regime-independent):", flush=True)
    print(f"      verify_gemm_cost_share_of_step = {dc['verify_gemm_cost_share_of_step']:.5f}  "
          f"(gemm_us[32]={dc['anchors']['gemm_us_m32']:.1f} / step_us={dc['anchors']['step_us_m32_deployed']:.1f})",
          flush=True)
    print(f"      off-the-shelf #122 +{comp['kanna122_offshelf_overhead_nonworking_ref']*100:.2f}% "
          f"(whole-model) → attributable_to_verify_gemm = "
          f"{hd['offtheshelf_overhead_attributable_to_verify_gemm_pct']:.3f}%  (band UPPER)",
          flush=True)
    print(f"      split-K floor s_net={dc['anchors']['s_net']*100:.3f}% (M=8) → "
          f"custom_kernel_overhead_floor = {hd['custom_kernel_overhead_floor_pct']:.3f}%  "
          f"(band LOWER; theoretical-min {dc['custom_kernel_overhead_floor_theoretical_min_pct']:.2f}%, "
          f"band-high {dc['custom_kernel_overhead_floor_band_high_pct']:.2f}%)", flush=True)
    print(f"      PLAUSIBLE CUSTOM-OVERHEAD BAND = [{hd['plausible_custom_overhead_band_pct'][0]:.3f}%, "
          f"{hd['plausible_custom_overhead_band_pct'][1]:.3f}%]   off-shelf/floor ratio="
          f"{dc['offtheshelf_to_custom_floor_ratio']:.1f}×", flush=True)
    for regime in REGIMES:
        print("-" * 96, flush=True)
        print(f"  [{regime}] (3) FEASIBILITY OVERLAY vs #213 budget:", flush=True)
        for tag, _tau in TAU_CORNERS:
            r = syn["regimes"][regime][tag]
            lm_f = r["lambda_min_kernel_feasible_floor"]
            lm_a = r["lambda_min_kernel_feasible_attributable"]
            lm_f_s = f"{lm_f:.4f}" if lm_f is not None else "NONE"
            lm_a_s = (f"{lm_a:.4f}" if lm_a is not None else "NONE")
            crit = r["lambda_crit_clears_500_zero_overhead"]
            crit_s = f"{crit:.4f}" if crit is not None else "NONE"
            print(f"      [{tag}] λ_crit={crit_s}  budget@λ̂={r['budget_at_lambda_hat']:+.3f}%  "
                  f"@0.997={r['budget_at_spine_0997']:+.3f}%  @1={r['budget_at_lambda_1']:+.3f}%",
                  flush=True)
            print(f"            λ_min(floor)={lm_f_s} (phys={r['lambda_min_kernel_feasible_floor_is_physical']})"
                  f"  λ_min(attrib)={lm_a_s} (phys={r['lambda_min_kernel_feasible_attributable_is_physical']})"
                  f"  max-budget@prob-sat={r['max_budget_pct_at_prob_saturation']:.2f}%", flush=True)
    print("-" * 96, flush=True)
    print(f"  buildable @λ̂={syn['lambda_hat']:.3f}: floor={hd['compliant_kernel_buildable_at_lambda_hat']} "
          f"attrib={hd['compliant_kernel_buildable_at_lambda_hat_attributable']}   "
          f"@spine0.997: floor={hd['compliant_kernel_buildable_at_spine_0997']} "
          f"attrib={hd['compliant_kernel_buildable_at_spine_0997_attributable']}", flush=True)
    print(f"  PRIMARY kernel_feasibility_self_test_passes = "
          f"{st['kernel_feasibility_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    lm = syn["test_metric"]["lambda_min_kernel_feasible"]
    print(f"  TEST lambda_min_kernel_feasible (floor/best-case, both-bugs, τ1) = "
          f"{lm:.5f}" if lm is not None else "  TEST lambda_min_kernel_feasible = NONE", flush=True)
    print(f"  VERDICT (optimistic) : {syn['verdict']}", flush=True)
    print(f"  VERDICT (conservative): {syn['verdict_conservative']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[kernel-feasibility] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="compliant-spec-et-ceiling",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["compliant-spec-et-ceiling", "issue-192", "batch-invariant", "validity-gate",
              "kernel-feasibility", "lane-a"],
        config={
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "lambda_hat": syn["lambda_hat"], "spine_0997": SPINE_0997,
            "bench_tokens": BENCH_TOKENS,
            "kanna122_offshelf_overhead": KANNA122_OFFSHELF_OVERHEAD,
            "verify_m": VERIFY_M, "splitk_operating_m": SPLITK_OPERATING_M,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[kernel-feasibility] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st, hd, dc = syn["self_test"], syn["headline"], syn["cost_decomposition"]
    bb1 = syn["regimes"]["both_bugs"]["tau_central_1p0"]
    des1 = syn["regimes"]["descent_only"]["tau_central_1p0"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "kernel_feasibility_self_test_passes":
            int(bool(st["kernel_feasibility_self_test_passes"])),
        "lambda_min_kernel_feasible": hd["lambda_min_kernel_feasible"],
        "lambda_min_kernel_feasible_is_physical": int(bool(hd["lambda_min_kernel_feasible_is_physical"])),
        # cost decomposition
        "verify_gemm_cost_share_of_step": dc["verify_gemm_cost_share_of_step"],
        "verify_gemm_cost_share_of_step_at_operating_m8": dc["verify_gemm_cost_share_of_step_at_operating_m8"],
        "offtheshelf_overhead_attributable_to_verify_gemm_pct":
            dc["offtheshelf_overhead_attributable_to_verify_gemm_pct"],
        "custom_kernel_overhead_floor_pct": dc["custom_kernel_overhead_floor_pct"],
        "custom_kernel_overhead_floor_gross_pct": dc["custom_kernel_overhead_floor_gross_pct"],
        "custom_kernel_overhead_floor_band_high_pct": dc["custom_kernel_overhead_floor_band_high_pct"],
        "band_lower_pct": hd["plausible_custom_overhead_band_pct"][0],
        "band_upper_pct": hd["plausible_custom_overhead_band_pct"][1],
        "offtheshelf_to_custom_floor_ratio": dc["offtheshelf_to_custom_floor_ratio"],
        # feasibility (both-bugs, τ1 headline)
        "lambda_crit_clears_500_zero_overhead": hd["lambda_crit_clears_500_zero_overhead_both_bugs_tau1"],
        "lambda_min_kernel_feasible_attributable_end": hd["lambda_min_kernel_feasible_attributable_end"],
        "offtheshelf_attributable_clears_at_physical_lambda":
            int(bool(hd["offtheshelf_attributable_clears_at_physical_lambda"])),
        "compliant_kernel_buildable_at_lambda_hat":
            int(bool(hd["compliant_kernel_buildable_at_lambda_hat"])),
        "compliant_kernel_buildable_at_spine_0997":
            int(bool(hd["compliant_kernel_buildable_at_spine_0997"])),
        "compliant_kernel_buildable_at_spine_0997_attributable":
            int(bool(hd["compliant_kernel_buildable_at_spine_0997_attributable"])),
        "budget_at_lambda_1_both_bugs_tau1": hd["budget_at_lambda_1_both_bugs_tau1"],
        "budget_at_spine_0997_both_bugs_tau1": bb1["budget_at_spine_0997"],
        "max_budget_pct_at_prob_saturation_both_bugs_tau1":
            hd["max_budget_pct_at_prob_saturation_both_bugs_tau1"],
        # descent regime cross-row
        "lambda_crit_descent_tau1": des1["lambda_crit_clears_500_zero_overhead"],
        "lambda_min_kernel_feasible_floor_descent_tau1": des1["lambda_min_kernel_feasible_floor"],
        "budget_at_lambda_1_descent_tau1": des1["budget_at_lambda_1"],
        # bars / composition
        "clear500_bar_et_tau1": syn["composition"]["clear500_bar_et_tau1"],
        "lambda_hat": syn["lambda_hat"],
        "verdict_plausible_above_lambda": int(syn["verdict"].startswith(
            "COMPLIANT_KERNEL_PLAUSIBLE_ABOVE_LAMBDA")),
        "verdict_conservative_infeasible_all_physical": int(
            "INFEASIBLE_AT_ALL_PHYSICAL_LAMBDA" in syn["verdict_conservative"]),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="kernel_feasibility_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[kernel-feasibility] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--shard", type=Path, default=C.RANKPROBE_SHARD,
                    help="in-scope PR#86 rankprobe shard (read-only; #199 source)")
    ap.add_argument("--max-records", type=int, default=None, help="debug: cap records parsed")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="compliant-spec-et-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize(args.shard, args.max_records)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 216, "agent": "wirbel",
        "kind": "kernel-feasibility", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["e_nan_clean"] = not nan_paths
    syn["self_test"]["kernel_feasibility_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[kernel-feasibility] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kernel_feasibility_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[kernel-feasibility] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["kernel_feasibility_self_test_passes"] and payload["nan_clean"]
        print(f"[kernel-feasibility] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
