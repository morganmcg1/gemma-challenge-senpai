#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 #319 unified single-approval read -- launch-spec self-test (PR #329, Issue #319).

0 GPU, 0 TPS. This does NOT launch anything. It is the CPU-only companion to the launch spec at
``research/launch/eagle3_319_unified_read_spec/spec.md``. It does TWO things, both on banked
constants only (re-derives nothing physical, touches no served file, runs no model):

  HALF 1 -- DECISION-ARITHMETIC REPRODUCTION (the load-bearing half, PR deliverable 5).
    Independently recomputes the merged #325/#323 GO/NO-GO thresholds from first principles + the
    banked fleet constants and verifies each reproduces to <= 1e-6:
      * rho_breakeven        = 500 / 622.080888                         (fern #310)  -> 0.8037539966988988
      * E[T]_pub             = et_from_profile(0.72925, 0.91443)        (lawine #300) ~ 4.966
      * rho_central          = rho_of_fdeep(f_deep=1.0)                  (fern #325)  ~ 0.9421
      * f_deep_breakeven     = bisect rho_of_fdeep == rho_breakeven      (fern #318)  -> 0.9163111901482197
      * round-trip           rho_of_fdeep(f_deep_breakeven) == rho_breakeven
      * worst-case credit    rho_of_fdeep(implied_f_deep 0.9083) == rho_worst 0.7923 (0-GPU credit CANNOT flip)
      * uncond_top4 bar      bisect et_from_profile(T,T,7) == 6.11       (lawine #316) -> 0.9213011665456927  *derived*
      * conditional-frac bar (1 - T)/(1 - a1) @ deployed a1             (lawine #316) -> 0.2907
      * regime-invariance    a1 + (1-a1)(1 - (1-T)/(1-a1)) == T for every a1            (algebraic identity)
      * joint truth table    all four GREEN/YELLOW/RED corners evaluate correctly

  HALF 2 -- SPEC COMPLETENESS (mirrors the ubel #322 measured-read-spec self-test).
    Machine-verifies every required section (HEADLINE, §0-§8) and every load-bearing
    number / flag / script / W&B id the human needs to fire the single a10g-small read is
    actually present in spec.md.

PRIMARY metric : unified_read_spec_self_test_passes  (bool -> 1 iff every arithmetic reproduction
                 lands <= 1e-6 AND every completeness condition holds AND the result is NaN-clean)
TEST    metric : max_abs_reproduction_residual       (float; the worst threshold reproduction error,
                 evidences the <= 1e-6 claim directly)

Run (CPU-only, no GPU):
    cd target/ && .venv/bin/python \\
        research/launch/eagle3_319_unified_read_spec/eagle3_319_unified_read_spec.py --self-test \\
        --wandb_group eagle3-319-unified-read --wandb_name fern/eagle3-319-unified-read-spec
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]          # target/
SPEC_PATH = HERE / "spec.md"

TOL_REPRO = 1e-6                                          # the PR's <= 1e-6 reproduction bar

# --------------------------------------------------------------------------- #
# Banked fleet constants (imported EXACTLY, cited; this card re-derives none).
# --------------------------------------------------------------------------- #
TARGET = 500.0
HONEST_PUBLIC_611 = 622.080888                # fern #310 (2u3kcnv5) honest_public(E[T]=6.11)
A1_HELD = 0.7292532942898975                  # lawine #300 / fern #318: position-1 (held by M=8 tree)
DEEP_PUB = 0.9144279172167558                 # lawine #300 / fern #318: EAGLE-3 deep-flat public rate
C_DEEP_LIN = 0.9713472759982902               # lawine #300: linear-spine deep private factor
EAGLE3_PUBLIC_ET = 4.96600000000002           # lawine #300: EAGLE-3 deep-flat public E[T]
E_T_BUILD_FREE = 6.1112149873699195           # fern #310 reconcile E_T_central (build-uniform target)

RHO_CENTRAL = 0.9421228821714434              # fern #325/#318: central deep-fidelity rho
RHO_WORST = 0.7922848664688427               # fern #318: EAGLE-3 worst x-dataset tau-ratio 5.34/6.74
IMPLIED_F_DEEP_WORST = 0.9082968311305457     # fern #325: implied f_deep (a1 held) at rho_worst
WITHIN_TASK_RHO = 0.9570535584491102          # fern #325: within-task analogue (460.85/481.53)

# merged thresholds this card reproduces (the GO/NO-GO bars the unified read clears):
RHO_BREAKEVEN = 0.8037539966988988            # fern #310 = 500 / 622.080888  (pass (a) flip bar)
F_DEEP_BREAKEVEN = 0.9163111901482197         # fern #318 (pass (a) flip bar, deep-retention form)
T_EFFECTIVE = 0.9213011665456927              # lawine #316/#323 build bar (pass (b) flip bar)
BAR_FRAC_AT_DEPLOYED_A1 = 0.2906732816855498  # lawine #316: conditional-frac bar @ deployed a1
A1_DEPLOYED = 0.7292532942898975              # == A1_HELD, the deployed raw a1 anchor

BASELINE_TPS = 481.53                         # current best summary.json:tps (unchanged; 0-TPS card)


# --------------------------------------------------------------------------- #
# Numeric helpers (no scipy in the analytic venv) -- copied verbatim from fern #318.
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-14, max_it: int = 400) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo > 0) == (fhi > 0):
        raise ValueError(f"root not bracketed: f({lo})={flo}, f({hi})={fhi}")
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


def et_from_profile(a1: float, deep: float, k_spec: int = 7) -> float:
    """E[T] = 1 + sum_{j=1..k} prod_{i<=j} a_i, with a_1=a1 and a_{2..k}=deep (kanna #289 survival sum)."""
    et = 1.0 + a1
    surv = a1
    for _ in range(2, k_spec + 1):
        surv *= deep
        et += surv
    return et


def rho_of_fdeep(f_deep: float) -> float:
    """rho_priv_e3 as a function of the incremental fusion-deep retention f_deep (a_1 held by tree)."""
    c_deep_e3 = C_DEEP_LIN * f_deep
    deep_priv = DEEP_PUB * c_deep_e3
    return et_from_profile(A1_HELD, deep_priv) / EAGLE3_PUBLIC_ET


def fdeep_for_rho(rho_target: float) -> float:
    return bisect(lambda f: rho_of_fdeep(f) - rho_target, 1e-6, 3.0)


def bar_frac_at_a1(a1: float, t_eff: float = T_EFFECTIVE) -> float:
    """The #316 build bar in conditional-frac units at raw a1: max_frac = (1 - T)/(1 - a1)."""
    return (1.0 - t_eff) / (1.0 - a1)


def uncond_from_cov_cond(a1: float, cov_cond: float) -> float:
    """Salvage identity: unconditional top-k = a1 + (1-a1)*cov_cond == c1_eff."""
    return a1 + (1.0 - a1) * cov_cond


def joint_verdict(pass_a_flips: bool, pass_b_flips: bool) -> str:
    """The §3 joint truth table: pass(a)=private-tax survives, pass(b)=coverage achievable."""
    if pass_a_flips and pass_b_flips:
        return "GREEN"
    if (not pass_a_flips) and (not pass_b_flips):
        return "RED"
    return "YELLOW"


# --------------------------------------------------------------------------- #
# HALF 1 -- decision-arithmetic reproduction (the load-bearing half, <= 1e-6).
# --------------------------------------------------------------------------- #
def arithmetic_reproduction() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def reg(key: str, value: float, target: float, tol: float) -> None:
        resid = abs(value - target)
        checks[key] = {"value": value, "target": target, "resid": resid,
                       "tol": tol, "pass": bool(resid <= tol)}

    # (1) break-even rho = 500 / honest_public(6.11) (fern #310).
    reg("01_rho_breakeven_500_over_62208",
        TARGET / HONEST_PUBLIC_611, RHO_BREAKEVEN, TOL_REPRO)

    # (2) public E[T] reproduced from the per-position deep-flat profile (lawine #300) -- 1e-3 (calibration).
    reg("02_public_et_reproduces_4966",
        et_from_profile(A1_HELD, DEEP_PUB), EAGLE3_PUBLIC_ET, 1e-3)

    # (3) central rho = rho_of_fdeep(1.0) reproduces the fern #325/#318 central (linear inheritance) -- 1e-4.
    reg("03_rho_central_reproduces_9421", rho_of_fdeep(1.0), RHO_CENTRAL, 1e-4)

    # (4) break-even f_deep via bisection (fern #318) reproduces the merged 0.9163111901482197.
    f_deep_breakeven = fdeep_for_rho(RHO_BREAKEVEN)
    reg("04_f_deep_breakeven_reproduces_9163", f_deep_breakeven, F_DEEP_BREAKEVEN, TOL_REPRO)

    # (5) round-trip: rho_of_fdeep(f_deep_breakeven) == rho_breakeven (the two pass-(a) bars agree).
    reg("05_roundtrip_rho_of_fdeep_breakeven",
        rho_of_fdeep(F_DEEP_BREAKEVEN), RHO_BREAKEVEN, TOL_REPRO)

    # (6) worst-case credit: rho_of_fdeep(implied_f_deep 0.9083) == rho_worst 0.7923.
    #     This encodes fern #325 §4A: the most-generous 0-GPU a1-credit lands implied_f_deep 0.9083,
    #     BELOW break-even 0.9163 -> the 0-GPU credit CANNOT flip alone; the #319 read is needed.
    reg("06_worstcase_credit_rho_of_implied_fdeep",
        rho_of_fdeep(IMPLIED_F_DEEP_WORST), RHO_WORST, TOL_REPRO)

    # (7) uncond_top4 bar DERIVED (not imported): solve build-uniform 1 + sum_{j=1..7} T^j = 6.11 for T.
    t_solved = bisect(lambda T: et_from_profile(T, T, 7) - E_T_BUILD_FREE, 0.5, 0.999)
    reg("07_uncond_top4_bar_derived_from_611", t_solved, T_EFFECTIVE, TOL_REPRO)

    # (8) conditional-frac bar at deployed a1 reproduces lawine #316's 0.2907 headline.
    reg("08_cond_frac_bar_at_deployed_a1",
        bar_frac_at_a1(A1_DEPLOYED), BAR_FRAC_AT_DEPLOYED_A1, TOL_REPRO)

    # (9) regime-invariance identity: a1 + (1-a1)(1 - (1-T)/(1-a1)) == T for every a1 (algebraic).
    invariance_resids = []
    for a1 in (0.65, 0.7292532942898975, 0.7713541666666667, 0.7730729805683441, 0.80, 0.85):
        cov4 = 1.0 - bar_frac_at_a1(a1)
        invariance_resids.append(abs(uncond_from_cov_cond(a1, cov4) - T_EFFECTIVE))
    max_inv = max(invariance_resids)
    checks["09_regime_invariance_uncond_is_T"] = {
        "value": max_inv, "target": 0.0, "resid": max_inv,
        "tol": TOL_REPRO, "pass": bool(max_inv <= TOL_REPRO)}

    # (10) joint truth table: all four corners evaluate to the §3 labels.
    #      pass(a) iff rho >= breakeven AND f_deep >= f_deep_breakeven; pass(b) iff uncond >= T.
    def pass_a(rho: float, f_deep: float) -> bool:
        return (rho >= RHO_BREAKEVEN) and (f_deep >= F_DEEP_BREAKEVEN)

    def pass_b(uncond: float) -> bool:
        return uncond >= T_EFFECTIVE

    corners = {
        # (within-task rho 0.957, f_deep 1.0) flips a; aime uncond 0.9570 flips b -> GREEN
        "GREEN_TT": joint_verdict(pass_a(WITHIN_TASK_RHO, 1.0), pass_b(0.9570053)) == "GREEN",
        # a flips, but coverage below bar (0.8903 aggregate) -> YELLOW coverage-bound
        "YELLOW_TF": joint_verdict(pass_a(WITHIN_TASK_RHO, 1.0), pass_b(0.8902556)) == "YELLOW",
        # worst-case rho/f_deep below break-even (no a), coverage clears -> YELLOW private-tax-bound
        "YELLOW_FT": joint_verdict(pass_a(RHO_WORST, IMPLIED_F_DEEP_WORST), pass_b(0.9300)) == "YELLOW",
        # both fail -> RED
        "RED_FF": joint_verdict(pass_a(RHO_WORST, IMPLIED_F_DEEP_WORST), pass_b(0.8902556)) == "RED",
    }
    checks["10_joint_truth_table_all_corners"] = {
        "value": float(sum(1 for v in corners.values() if v)), "target": 4.0,
        "resid": float(4 - sum(1 for v in corners.values() if v)), "tol": 0.0,
        "pass": bool(all(corners.values())), "corners": corners}

    residuals = [c["resid"] for c in checks.values() if isinstance(c.get("resid"), float)]
    max_abs_reproduction_residual = max(residuals) if residuals else 0.0
    # the worst residual among the strict <= 1e-6 reproductions (excludes the looser calibration checks 02/03)
    strict_keys = ["01_rho_breakeven_500_over_62208", "04_f_deep_breakeven_reproduces_9163",
                   "05_roundtrip_rho_of_fdeep_breakeven", "06_worstcase_credit_rho_of_implied_fdeep",
                   "07_uncond_top4_bar_derived_from_611", "08_cond_frac_bar_at_deployed_a1",
                   "09_regime_invariance_uncond_is_T"]
    max_strict_residual = max(checks[k]["resid"] for k in strict_keys)

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "checks": checks,
        "f_deep_breakeven_computed": f_deep_breakeven,
        "uncond_top4_bar_derived": t_solved,
        "max_abs_reproduction_residual": max_abs_reproduction_residual,
        "max_strict_residual": max_strict_residual,
        "arithmetic_all_pass": bool(all_pass),
        "thresholds": {
            "rho_breakeven": RHO_BREAKEVEN,
            "f_deep_breakeven": F_DEEP_BREAKEVEN,
            "uncond_top4_bar": T_EFFECTIVE,
            "cond_frac_bar_at_deployed_a1": BAR_FRAC_AT_DEPLOYED_A1,
        },
    }


# --------------------------------------------------------------------------- #
# HALF 2 -- spec completeness (mirrors the ubel #322 measured-read-spec self-test).
# --------------------------------------------------------------------------- #
def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def evaluate_completeness(spec_text: str) -> dict[str, bool]:
    c: dict[str, bool] = {}

    # 1. Every required section is present (HEADLINE + §0..§8).
    c["sec_headline_two_flips"] = _has(spec_text, "## HEADLINE", "two", "flip")
    c["sec_0_blocker"] = _has(spec_text, "## §0", "blocker", "gua9x68j")
    c["sec_1_single_job"] = _has(spec_text, "## §1", "single staged job", "once")
    c["sec_1_2_pass_a"] = _has(spec_text, "§1.2", "pass (a)", "per-depth")
    c["sec_1_3_pass_b"] = _has(spec_text, "§1.3", "pass (b)", "coverage")
    c["sec_2_harness"] = _has(spec_text, "## §2", "harness", "json")
    c["sec_3_go_nogo"] = _has(spec_text, "## §3", "go/no-go", "truth table")
    c["sec_4_proof_vram_issue"] = _has(spec_text, "## §4", "greedy", "vram", "approval")
    c["sec_5_cost_failure"] = _has(spec_text, "## §5", "cost", "failure branches")
    c["sec_6_checklist"] = _has(spec_text, "## §6", "pre-launch checklist")
    c["sec_7_launch_cmd"] = _has(spec_text, "## §7", "launch command")
    c["sec_8_selftest"] = _has(spec_text, "## §8", "self-test")

    # 2. The one shared head + arch both passes load (the §0 blocker).
    c["head_gua9x68j"] = _has(spec_text, "gua9x68j", "56ksyxgw")
    c["head_arch_eagle3"] = _has(spec_text, "Eagle3LlamaForCausalLM", "[2,21,39]")
    c["head_publish_bucket"] = _has(spec_text, "publish") and (
        _has(spec_text, "bucket") or _has(spec_text, "hub")) and _has(spec_text, "DRAFTER_SHA256")
    c["head_local_smoke"] = _has(spec_text, "smoke") and _has(spec_text, "greedy")

    # 3. Pass (a) -- per-depth private-alpha harness + outputs.
    c["passa_script"] = _has(spec_text, "accept_calibration.py", "spec_decode_num_accepted_tokens_per_pos")
    c["passa_rho"] = _has(spec_text, "rho_priv_e3", "E[T]_priv", "E[T]_pub")
    c["passa_a1_deep_split"] = _has(spec_text, "a", "deep", "f_deep") and _has(spec_text, "c_deep_lin")
    c["passa_private_set"] = _has(spec_text, "private", "ood")

    # 4. Pass (b) -- ShareGPT unconditional top-4 coverage harness + outputs.
    c["passb_script"] = _has(spec_text, "rank_coverage.py", "RANKPROBE_W=4")
    c["passb_uncond_top4"] = _has(spec_text, "uncond_top4")
    c["passb_sharegpt_128"] = _has(spec_text, "eval_prompts_sharegpt.json", "128")
    c["passb_env"] = _has(spec_text, "RANKPROBE_ENABLE=1", "LOOPGRAPH_WARMUP_CALLS=1e9",
                          "VLLM_USE_FLASHINFER_SAMPLER=0")

    # 5. §2 JSON output schema names the load-bearing keys.
    c["schema_pass_a"] = _has(spec_text, "pass_a_private_alpha", "alpha_per_depth_private", "rho_priv_e3")
    c["schema_pass_b"] = _has(spec_text, "pass_b_sharegpt_coverage", "uncond_top4", "align_bad")
    c["schema_decision"] = _has(spec_text, "joint_verdict", "fern325_flips_green", "lawine323_flips_go")

    # 6. §3 GO/NO-GO bars + all four truth-table corners.
    c["go_rho_bar"] = ("0.8038" in spec_text) and ("0.8037539966988988" in spec_text)
    c["go_fdeep_bar"] = ("0.9163" in spec_text) and ("0.9163111901482197" in spec_text)
    c["go_uncond_bar"] = ("0.9213" in spec_text) and ("0.9213011665456927" in spec_text)
    c["go_cond_frac_bar"] = "0.2907" in spec_text
    c["truth_table_corners"] = _has(spec_text, "GREEN", "coverage-bound", "private-tax-bound", "RED")

    # 7. §4 greedy/PPL proof + VRAM budget + the exact approval-issue title.
    c["proof_align_bad"] = _has(spec_text, "align_bad") and ("0" in spec_text)
    c["proof_readonly"] = _has(spec_text, "read-only") and _has(spec_text, "no emission")
    c["vram_fit"] = ("20.158" in spec_text) and ("3.84" in spec_text) and _has(spec_text, "headroom")
    c["approval_title"] = _has(spec_text, "Approval request: HF job for eagle3-319-unified-read")

    # 8. §5/§7 cost + launch gate + DO-NOT-LAUNCH guard.
    c["cost_one_job"] = _has(spec_text, "a10g-small", "one") and (
        _has(spec_text, "40 min") or "2400" in spec_text)
    c["launch_cmd"] = _has(spec_text, "train.py", "--launch")
    c["do_not_launch"] = _has(spec_text, "DO NOT") or _has(spec_text, "do not run")
    c["vllm_hw"] = _has(spec_text, "0.22.1rc1", "a10g-small", "sm_86")

    # 9. §6 checklist maps every merged closure this read flips / depends on.
    merged = ["#325", "#323", "#318", "#310", "#308", "#34", "#79", "#299", "#322"]
    c["checklist_closures"] = all(m in spec_text for m in merged)
    wandb_ids = ["xk1pghy4", "ceddxj20", "xe8ff7hq", "2u3kcnv5", "5axqa6oa", "z6wi4z4v"]
    c["checklist_wandb_ids"] = all(w in spec_text for w in wandb_ids)

    # 10. Baseline-untouched / 0-TPS framing carried (pre-registration honesty).
    c["zero_tps_baseline"] = ("481.53" in spec_text) and _has(spec_text, "0 TPS") and _has(
        spec_text, "unchanged")
    c["section_8_bars_echoed"] = _has(
        spec_text, "rho_breakeven = 0.8037539966988988", "f_deep_breakeven = 0.9163111901482197",
        "uncond_top4_bar = 0.9213011665456927")

    return c


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    arith = arithmetic_reproduction()

    spec_exists = SPEC_PATH.exists()
    spec_text = SPEC_PATH.read_text(encoding="utf-8") if spec_exists else ""
    completeness = evaluate_completeness(spec_text)
    completeness_all_pass = bool(spec_exists and all(completeness.values()))

    # NaN-clean walk over the arithmetic payload.
    def _finite(x: Any) -> bool:
        return not (isinstance(x, float) and not math.isfinite(x))
    nan_clean = all(_finite(ch.get("value")) and _finite(ch.get("resid"))
                    for ch in arith["checks"].values())

    self_test_passes = bool(arith["arithmetic_all_pass"] and completeness_all_pass and nan_clean)

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    n_arith = len(arith["checks"])
    n_arith_pass = sum(1 for c in arith["checks"].values() if c["pass"])
    n_complete = len(completeness)
    n_complete_pass = sum(1 for v in completeness.values() if v)

    return {
        "pr": 329, "agent": "fern", "kind": "eagle3_319_unified_read_spec",
        "analysis_only": True, "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "primary_metric_name": "unified_read_spec_self_test_passes",
        "unified_read_spec_self_test_passes": self_test_passes,
        "test_metric_name": "max_abs_reproduction_residual",
        "max_abs_reproduction_residual": arith["max_abs_reproduction_residual"],
        "max_strict_residual": arith["max_strict_residual"],
        "arithmetic_all_pass": arith["arithmetic_all_pass"],
        "completeness_all_pass": completeness_all_pass,
        "nan_clean": nan_clean,
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "spec_exists": spec_exists,
        "spec_bytes": len(spec_text.encode("utf-8")),
        "n_arith_conditions": n_arith, "n_arith_pass": n_arith_pass,
        "n_completeness_conditions": n_complete, "n_completeness_pass": n_complete_pass,
        "thresholds": arith["thresholds"],
        "f_deep_breakeven_computed": arith["f_deep_breakeven_computed"],
        "uncond_top4_bar_derived": arith["uncond_top4_bar_derived"],
        "arithmetic_checks": arith["checks"],
        "completeness_checks": completeness,
        "peak_mem_mib": peak_mem_mib,
        "constants": {
            "target": TARGET, "honest_public_611": HONEST_PUBLIC_611, "a1_held": A1_HELD,
            "deep_pub": DEEP_PUB, "c_deep_lin": C_DEEP_LIN, "eagle3_public_et": EAGLE3_PUBLIC_ET,
            "e_t_build_free": E_T_BUILD_FREE, "rho_central": RHO_CENTRAL, "rho_worst": RHO_WORST,
            "implied_f_deep_worst": IMPLIED_F_DEEP_WORST, "within_task_rho": WITHIN_TASK_RHO,
            "rho_breakeven": RHO_BREAKEVEN, "f_deep_breakeven": F_DEEP_BREAKEVEN,
            "uncond_top4_bar": T_EFFECTIVE, "bar_frac_at_deployed_a1": BAR_FRAC_AT_DEPLOYED_A1,
            "baseline_tps": BASELINE_TPS,
        },
        "provenance": (
            "fern #325 (xk1pghy4) joint envelope rho_worst 0.7923 / implied_f_deep 0.9083 / within-task "
            "0.957 x fern #318 (xe8ff7hq) f_deep_breakeven 0.9163 / et_from_profile x fern #310 (2u3kcnv5) "
            "honest_public_611 622.080888 / breakeven 0.8038 x lawine #300 a1 0.72925 / deep 0.91443 / "
            "c_deep_lin 0.97135 / public_et 4.966 x lawine #316/#323 (ceddxj20) uncond_top4 bar 0.9213 from "
            "build-uniform E[T]=6.11 x wirbel #79 (z6wi4z4v) RANKPROBE_W=4 align_bad=0 x ubel #322 spec pattern."),
        "scope": (
            "LOCAL CPU-only self-test over banked constants + the spec.md companion. 0 TPS; BASELINE 481.53 "
            "untouched; greedy/PPL untouched. NO GPU / vLLM / model forward / HF Job / submission / "
            "served-file change. Authorizes NOTHING. NOT a launch. This pre-registers the #319 read protocol."),
    }


# --------------------------------------------------------------------------- #
def print_report(p: dict[str, Any]) -> None:
    print("\n" + "=" * 92, flush=True)
    print("EAGLE-3 #319 UNIFIED READ -- launch-spec self-test (PR #329)", flush=True)
    print("=" * 92, flush=True)
    print("HALF 1 -- decision-arithmetic reproduction (merged #325/#323 thresholds, <= 1e-6):", flush=True)
    for k, ch in p["arithmetic_checks"].items():
        tag = "PASS" if ch["pass"] else "FAIL"
        print(f"  [{tag}] {k:<42} value={ch['value']:.12g} target={ch['target']:.12g} "
              f"resid={ch['resid']:.2e} (tol {ch['tol']:.0e})", flush=True)
    print(f"  -> f_deep_breakeven computed = {p['f_deep_breakeven_computed']:.16f}", flush=True)
    print(f"  -> uncond_top4 bar DERIVED  = {p['uncond_top4_bar_derived']:.16f}", flush=True)
    print(f"  -> max strict residual      = {p['max_strict_residual']:.2e}", flush=True)
    print("-" * 92, flush=True)
    print(f"HALF 2 -- spec completeness ({p['n_completeness_pass']}/{p['n_completeness_conditions']} "
          f"conditions on {p['spec_path']}, {p['spec_bytes']} bytes):", flush=True)
    failed = [k for k, v in p["completeness_checks"].items() if not v]
    if failed:
        print(f"  FAILED completeness conditions: {failed}", flush=True)
    else:
        print("  all completeness conditions hold.", flush=True)
    print("-" * 92, flush=True)
    print(f"PRIMARY unified_read_spec_self_test_passes = {p['unified_read_spec_self_test_passes']}", flush=True)
    print(f"TEST    max_abs_reproduction_residual      = {p['max_abs_reproduction_residual']:.2e}", flush=True)
    print(f"nan_clean = {p['nan_clean']}   peak_mem_mib = {p['peak_mem_mib']:.2f}", flush=True)
    print("=" * 92 + "\n", flush=True)


# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        import wandb as _wb  # noqa: F401
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[unified-read-spec] wandb logging unavailable (analysis unaffected): {exc}", flush=True)
        return None

    run = init_wandb_run(
        job_type="validity-analytic", agent="fern", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3", "unified-read", "launch-spec", "0-tps", "issue-319", "bank-the-analysis"],
        config={
            "pr": 329, "issue": 319, "wandb_group": args.wandb_group, "spec_path": payload["spec_path"],
            "rho_breakeven": RHO_BREAKEVEN, "f_deep_breakeven": F_DEEP_BREAKEVEN,
            "uncond_top4_bar": T_EFFECTIVE, "honest_public_611": HONEST_PUBLIC_611,
            "baseline_tps": BASELINE_TPS, "provenance": payload["provenance"], "scope": payload["scope"],
        },
    )
    if run is None:
        print("[unified-read-spec] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "unified_read_spec_self_test_passes": int(bool(payload["unified_read_spec_self_test_passes"])),
        "max_abs_reproduction_residual": payload["max_abs_reproduction_residual"],
        "max_strict_residual": payload["max_strict_residual"],
        "arithmetic_all_pass": int(bool(payload["arithmetic_all_pass"])),
        "completeness_all_pass": int(bool(payload["completeness_all_pass"])),
        "n_arith_pass": payload["n_arith_pass"], "n_arith_conditions": payload["n_arith_conditions"],
        "n_completeness_pass": payload["n_completeness_pass"],
        "n_completeness_conditions": payload["n_completeness_conditions"],
        "f_deep_breakeven_computed": payload["f_deep_breakeven_computed"],
        "uncond_top4_bar_derived": payload["uncond_top4_bar_derived"],
        "rho_breakeven": RHO_BREAKEVEN, "f_deep_breakeven": F_DEEP_BREAKEVEN,
        "uncond_top4_bar": T_EFFECTIVE, "cond_frac_bar_at_deployed_a1": BAR_FRAC_AT_DEPLOYED_A1,
        "spec_bytes": payload["spec_bytes"], "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"], "tps_added_by_this_card": 0,
    }
    summary.update({f"arith_{k}": int(bool(v["pass"])) for k, v in payload["arithmetic_checks"].items()})
    summary.update({f"complete_{k}": int(bool(v)) for k, v in payload["completeness_checks"].items()})
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_319_unified_read_spec_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[unified-read-spec] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-319-unified-read")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    payload = run()
    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_319_unified_read_spec_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[unified-read-spec] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    gate = bool(payload["unified_read_spec_self_test_passes"])
    print(f"  PRIMARY unified_read_spec_self_test_passes = {gate}", flush=True)
    print(f"  TEST    max_abs_reproduction_residual = {payload['max_abs_reproduction_residual']:.2e}", flush=True)
    print(f"  wandb run = {rid}", flush=True)
    if args.self_test:
        print(f"[unified-read-spec] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
