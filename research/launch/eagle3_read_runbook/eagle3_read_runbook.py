#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 measured-read RUNBOOK -- single-approval HF-job block for the #319 read (PR #350, Issue #319).

STRICT-IDENTITY WORLD. 0 GPU, 0 TPS. This does NOT launch anything, change no served file, file
no HF Job. It is the CPU-only companion to ``RUNBOOK.md`` -- the mechanically-ready single-approval
block the human can fire with ZERO further advisor round-trips. The human submits the job; this card
only PACKAGES it.

WHY STRICT: the human reversed issue #124 on issue #319 (2026-06-15T10:56:17Z): "No, ignore #124, we
want to ensure we stick with the strict greedy token matching". That re-instates issue #192 as the
live contract, so this runbook is rebuilt in the strict-identity world (advisor Option A, PR #350):

  * the coverage bar reverts to the IDENTITY bar 0.9213 (NOT the PPL-only c*=0.9089);
  * served greedy-token-identity == 1.0 becomes a co-equal HARD gate (NOT report-only);
  * the read is a DIAGNOSTIC, NOT a ">500 build GO/NO-GO": wirbel #343 (kklof4wr) + denken #332 cap
    the strict deployed substrate at 473.5 TPS < 500 for EVERY realizable deterministic schedule,
    regardless of coverage. The honest value is (a) served greedy-identity confirmation (the
    now-binding gate), (b) C2 (fp32->bf16 numerics + served greedy-identity) closure end-to-end, and
    (c) exact gap-to-0.9213 sizing on the existing head (~0.8903, P(clear)~0.06 -> expected NO-GO).

The card re-derives nothing physical. It (HALF 1) reproduces the strict-world GO/NO-GO arithmetic
from the banked fleet constants to <= 1e-6 and (HALF 2) machine-verifies RUNBOOK.md is a complete
single-approval block. Banked anchors assembled (cite, reuse exact):
  * ubel #322 (2nmem4dc) read spec / protocol           research/launch/eagle3_measured_read_spec.md
  * ubel #333 (quzi85y0) converter (convert_eagle3_to_safetensors.py)
  * ubel #338 (y4jj278b) vLLM load dry-run (C1/C3/C4 CLOSED-AT-0-GPU; only C2 GPU-residual)
  * wirbel #343 (kklof4wr) strict ceiling 473.5 / strict_500_reachable=False + PPL-only c* contingency
  * lawine #330 (hfrscdai) existing-head coverage prior 0.8903 / P(clear 0.9213)~0.06 / gap 0.031
  * lawine #339 (0aq16szh) retrain clears 0.9213 with P~0.843 (why gap-sizing is worth ~1 GPU-hr)
  * fern #34 (gua9x68j / train 56ksyxgw) the only trained {2,21,39} fusion head the read loads

PRIMARY metric : read_runbook_self_test_passes  (bool -> 1 iff every strict-world arithmetic
                 reproduction lands <= 1e-6 AND RUNBOOK.md is a complete single-approval block AND
                 the #338 pre-flight C1/C3/C4 is green AND the no_hf_job/no_launch/no_served_file
                 flags are recorded AND the payload is NaN-clean)
SECOND  metric : read_is_single_approval_ready   (bool -> the launch command + approval-issue title +
                 the 4 pre-launch gates + the §0 checkpoint-publish blocker are all present, so the
                 human can fire the one job with no further design work)
TEST    metric : max_abs_reproduction_residual   (float; worst strict-world threshold reproduction err)

Run (CPU-only, no GPU):
    cd target/ && python research/launch/eagle3_read_runbook/eagle3_read_runbook.py --self-test \\
        --wandb_group eagle3-read-runbook --wandb_name ubel/eagle3-read-runbook
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
RUNBOOK_PATH = HERE / "RUNBOOK.md"

TOL_REPRO = 1e-6                                          # the strict <= 1e-6 reproduction bar

# --------------------------------------------------------------------------- #
# Banked fleet constants (imported EXACTLY, cited; this card re-derives none).
# --------------------------------------------------------------------------- #
TARGET = 500.0
BASELINE_TPS = 481.53                          # PR #52 (2x9fm2zx) official frontier; unchanged (0-TPS card)

# ---- STRICT-IDENTITY WORLD: the LIVE contract (human #192 / #124-reversal on #319, 10:56:17Z) ----
IDENTITY_BAR = 0.9213011665456927              # lawine #330 (hfrscdai)/#316/#323 build bar; derived from E[T]=6.11
PPL_GATE = 2.42                                # program.md cap (ref 2.30 + 5%)
PPL_DEPLOYED = 2.3772                          # PR #52 frontier PPL; wirbel #343
PPL_MARGIN = PPL_GATE - PPL_DEPLOYED           # 0.0428
E_T_BUILD_FREE = 6.1112149873699195            # stark #337/#340: build-uniform E[T] target @ identity bar
SUPPLY_FLOOR = 0.09103155435261377             # denken #332 (y5cl0ena) geometric-phi supply (determinism) floor
LAMBDA_CENTRAL = 520.9527323111674             # stark #340/wirbel #343 demand-only central anchor @ identity
LAMBDA_WORST = 492.865273281899                # stark #340/wirbel #343 demand-only worst anchor @ identity
STRICT_CEILING_CENTRAL = 473.5295953446407     # wirbel #343 (kklof4wr): strict SUPREMUM over coverage (capped)
STRICT_CEILING_WORST = 447.99898136862197      # wirbel #343 worst-anchor strict ceiling
STRICT_500_REACHABLE = False                   # wirbel #343: supply-capped < 500 for every realizable schedule
STRICT_GAP_TO_500 = 26.47040465535929          # wirbel #343: 500 - strict ceiling
FERN349_RESTORATION_CEILING = 473.53           # fern #349 (u8vmtji0) FlashInfer-BI restoration_ceiling: an
                                               # INDEPENDENT third confirmation of the 473.5 cap (advisor-cited
                                               # 2026-06-15T11:29:49Z; corroborates wirbel #343 + denken #332)

# ---- the EXISTING in-repo head: the read's SUBJECT (lawine #330 hfrscdai) ----
EXISTING_HEAD_COV_PRIOR = 0.8902659519153152   # lawine #330 official-eval unconditional top-4 prior (0.8903)
EXISTING_HEAD_P_CLEARS_IDENTITY = 0.06031894029725235  # lawine #330: P(existing head clears 0.9213) ~ 0.06
GAP_TO_IDENTITY_BAR = 0.031035214630377506     # lawine #330 gap_to_bar magnitude (lift the read sizes)
RECORD_BERNOULLI_SE = 0.019995605787702757     # lawine #330 conservative record-Bernoulli SE
LIKELY_MISSES_P = 0.3                           # lawine #330 verdict threshold (P < 0.3 -> LIKELY-MISSES)

# ---- retrain context: WHY sizing the gap is worth ~1 GPU-hr (lawine #339 0aq16szh) ----
P_RETRAIN_CLEARS_IDENTITY_INDEP = 0.8426502591576528   # lawine #339: 4-lever retrain clears 0.9213 (independent)
P_RETRAIN_CLEARS_IDENTITY_CORR05 = 0.7941613476310673  # lawine #339: +0.5-correlated
RETRAIN_MIN_LIFT = 0.031035214630377506        # lawine #336/#330: min coverage lift to the identity bar

# ---- PPL-ONLY CONTINGENCY (carried ONLY for "if #192 is ever lifted"; #124 world) ----
# Imported verbatim from wirbel #343 (kklof4wr) deliverable2; NOT the live bars.
PPLONLY_COV_PRIOR_ROUNDED = 0.8903             # the rounded prior wirbel #343 differenced the c* lifts against
PPLONLY_CSTAR_CENTRAL = 0.9089363308345582     # wirbel #343: env_central(c*) = 500
PPLONLY_CSTAR_WORST = 0.925603648491971        # wirbel #343: env_worst(c*) = 500
PPLONLY_LIFT_CENTRAL = 0.01863633083455818     # wirbel #343: c*_central - 0.8903 (WITHIN +0.031 budget)
PPLONLY_LIFT_WORST = 0.035303648491970985      # wirbel #343: c*_worst - 0.8903 (OVER +0.031 budget)
PPLONLY_ENV_AT_PRIOR_CENTRAL = 470.347938447151    # wirbel #343: env_central(0.8903) < 500
PPLONLY_ENV_AT_PRIOR_WORST = 444.98886528896605    # wirbel #343: env_worst(0.8903) < 500
RETRAIN_LIFT_BUDGET_336 = 0.031                # lawine #336: retrain coverage-lift budget

# ---- banked W&B run ids (the anchors this runbook assembles; cite verbatim) ----
HEAD_WANDB = "gua9x68j"          # fern #34 head
HEAD_TRAIN_WANDB = "56ksyxgw"    # fern #34 training run
CONVERTER_WANDB = "quzi85y0"     # ubel #333 converter
DRYRUN_WANDB = "y4jj278b"        # ubel #338 vLLM load dry-run
READSPEC_WANDB = "2nmem4dc"      # ubel #322 read spec
WIRBEL343_WANDB = "kklof4wr"     # wirbel #343 strict ceiling + PPL-only c*
LAWINE330_WANDB = "hfrscdai"     # lawine #330 coverage prior
LAWINE339_WANDB = "0aq16szh"     # lawine #339 retrain clear probability
FRONTIER_WANDB = "2x9fm2zx"      # PR #52 frontier
FERN349_WANDB = "u8vmtji0"       # fern #349 FlashInfer-BI restoration_ceiling 473.53 (third cap corroboration)

# ---- advisor-provided DIRECTIONAL pointers (2026-06-15T11:29:49Z): where the live strict >500 search
#      actually moved, so this diagnostic points the right way. Cited as HANDED OFF -- this card does NOT
#      inspect those branches, re-derive, or reuse their numbers in any arithmetic check. ----
STRICT_500_FRONTIER_LADDER = [
    {"tps": 165.44, "lane": "non-spec", "ref": "lawine #196"},
    {"tps": 357.32, "lane": "off-shelf batch-invariant spec", "ref": "#326"},
    {"tps": 481.53, "lane": "custom reduction-invariant kernel (<=)", "ref": "wirbel #354"},
]
STRICT_500_INFLIGHT_LEVERS = [
    "wirbel #354 (custom-kernel compliance ceiling)",
    "lawine #355 (sub-int4 body)",
    "fern #357 (composite reachability)",
    "kanna #359 (identity-preserving step-shave)",
]

# ---- pre-flight gate ledger (ubel #338, y4jj278b): only C2 is GPU-residual ----
PREFLIGHT_LEDGER = {
    "C1": {"name": "config-field survival + class registration", "status": "CLOSED-AT-0-GPU", "closed": True},
    "C2": {"name": "fp32->bf16 inference numerics + served greedy-identity == 1.0",
           "status": "REQUIRES-GPU-SMOKE", "closed": False},
    "C3": {"name": "absent-d2t -> identity-map default", "status": "CLOSED-AT-0-GPU", "closed": True},
    "C4": {"name": "vLLM-fork version / schema pin", "status": "CLOSED-AT-0-GPU", "closed": True},
}
PREFLIGHT_GREEN_AT_0GPU = ["C1", "C3", "C4"]
C2_RESIDUAL = "C2"

# ---- the EXACT single-approval HF-job block (the deliverable) ----
SUBMISSION = "submissions/fa2sw_precache_kenyan"
METHOD_TAG = "ubel/eagle3-read-strict"
APPROVAL_TITLE = "Approval request: HF job for eagle3-read-strict"
LAUNCH_CMD = (
    "cd /workspace/senpai/target\n"
    "# (after the 4 pre-launch gates below PASS + human approval on issue #319)\n"
    "python train.py \\\n"
    f"  --submission {SUBMISSION} \\\n"
    f'  --method "{METHOD_TAG}" \\\n'
    "  --launch --wait"
)

# ---- no-side-effect flags (recorded per PR instruction #5e) ----
NO_GPU = True
NO_HF_JOB = True
NO_LAUNCH = True
NO_SERVED_FILE_CHANGE = True

# ---- the metrics the one a10g-small job RETURNS (strict world) ----
JOB_RETURNS = {
    "uncond_top4": {"bar": IDENTITY_BAR, "rule": ">=", "kind": "HARD", "note": "coverage vs identity bar"},
    "served_greedy_identity_rate": {"bar": 1.0, "rule": "==", "kind": "HARD",
                                    "note": "the now-binding gate (#192); C2 closure"},
    "ppl": {"bar": PPL_GATE, "rule": "<=", "kind": "HARD", "note": "program.md cap; deployed 2.3772"},
    "completed": {"bar": 128, "rule": "==", "kind": "HARD", "note": "all public prompts"},
    "alpha_per_depth": {"bar": None, "rule": "report", "kind": "DIAGNOSTIC",
                        "note": "[a1..a7]; sizes E[T] and the exact coverage gap to 0.9213"},
    "vram_peak_gib": {"bar": 24.0, "rule": "<=", "kind": "HARD", "note": "ubel #299/#306: 20.158 peak / 3.84 headroom"},
}


# --------------------------------------------------------------------------- #
# Numeric helpers (no scipy in the analytic venv).
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


def et_buildchain(c: float, k_spec: int = 7) -> float:
    """Build-uniform E[T] = 1 + sum_{j=1..k} c^j (stark #337 chain law; a_1 = deep = c)."""
    et = 1.0
    p = 1.0
    for _ in range(1, k_spec + 1):
        p *= c
        et += p
    return et


def envelope_central(c: float) -> float:
    """PPL-only demand-only envelope (central anchor): LAMBDA_CENTRAL * E[T](c)/E[T](identity)."""
    return LAMBDA_CENTRAL * et_buildchain(c) / et_buildchain(IDENTITY_BAR)


def envelope_worst(c: float) -> float:
    """PPL-only demand-only envelope (worst anchor): LAMBDA_WORST * E[T](c)/E[T](identity)."""
    return LAMBDA_WORST * et_buildchain(c) / et_buildchain(IDENTITY_BAR)


# --------------------------------------------------------------------------- #
# HALF 1 -- strict-world GO/NO-GO arithmetic reproduction (the load-bearing half).
# --------------------------------------------------------------------------- #
def arithmetic_reproduction() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def reg(key: str, value: float, target: float, tol: float) -> None:
        resid = abs(value - target)
        checks[key] = {"value": value, "target": target, "resid": resid,
                       "tol": tol, "pass": bool(resid <= tol)}

    def reg_bool(key: str, ok: bool) -> None:
        checks[key] = {"value": float(bool(ok)), "target": 1.0, "resid": 0.0 if ok else 1.0,
                       "tol": 0.0, "pass": bool(ok)}

    # (1) identity bar DERIVED (not just imported): solve build-uniform 1 + sum_{j=1..7} T^j = 6.11 for T.
    t_solved = bisect(lambda T: et_buildchain(T, 7) - E_T_BUILD_FREE, 0.5, 0.999)
    reg("01_identity_bar_derived_from_611", t_solved, IDENTITY_BAR, TOL_REPRO)

    # (2) strict ceiling (central) = lambda_central * (1 - supply_floor); roundtrips denken #332 / wirbel #343.
    reg("02_strict_ceiling_central_reproduces_473",
        LAMBDA_CENTRAL * (1.0 - SUPPLY_FLOOR), STRICT_CEILING_CENTRAL, TOL_REPRO)

    # (3) strict ceiling (worst) = lambda_worst * (1 - supply_floor).
    reg("03_strict_ceiling_worst_reproduces_448",
        LAMBDA_WORST * (1.0 - SUPPLY_FLOOR), STRICT_CEILING_WORST, TOL_REPRO)

    # (4) strict gap to 500 = 500 - strict ceiling central.
    reg("04_strict_gap_to_500_reproduces_26p47",
        TARGET - STRICT_CEILING_CENTRAL, STRICT_GAP_TO_500, TOL_REPRO)

    # (5) gap the read SIZES = identity bar - existing-head prior (lawine #330 gap_to_bar).
    reg("05_gap_to_identity_bar_reproduces_0p031",
        IDENTITY_BAR - EXISTING_HEAD_COV_PRIOR, GAP_TO_IDENTITY_BAR, TOL_REPRO)

    # (6) PPL-only CONTINGENCY import validated by roundtrip: env_central(c*_central) == 500 (wirbel #343).
    reg("06_pplonly_cstar_central_roundtrips_500", envelope_central(PPLONLY_CSTAR_CENTRAL), TARGET, 1e-6)

    # (7) env_worst(c*_worst) == 500.
    reg("07_pplonly_cstar_worst_roundtrips_500", envelope_worst(PPLONLY_CSTAR_WORST), TARGET, 1e-6)

    # (8) PPL-only lifts reproduce wirbel #343 (against the rounded 0.8903 prior wirbel used).
    reg("08a_pplonly_lift_central", PPLONLY_CSTAR_CENTRAL - PPLONLY_COV_PRIOR_ROUNDED,
        PPLONLY_LIFT_CENTRAL, TOL_REPRO)
    reg("08b_pplonly_lift_worst", PPLONLY_CSTAR_WORST - PPLONLY_COV_PRIOR_ROUNDED,
        PPLONLY_LIFT_WORST, TOL_REPRO)

    # (9) STRICT world: NOT a >500 gate -- strict ceiling < 500 AND strict_500_reachable is False.
    reg_bool("09_strict_500_not_reachable",
             (STRICT_CEILING_CENTRAL < TARGET) and (STRICT_500_REACHABLE is False))

    # (10) existing head LIKELY-MISSES strict -> expected NO-GO with a measured gap (lawine #330).
    reg_bool("10_existing_head_likely_misses_strict",
             (EXISTING_HEAD_COV_PRIOR < IDENTITY_BAR)
             and (EXISTING_HEAD_P_CLEARS_IDENTITY < LIKELY_MISSES_P))

    # (11) even gate-lifted (PPL-only), the existing head gives NO free 500 at its prior (both anchors < 500).
    env_c_prior = envelope_central(EXISTING_HEAD_COV_PRIOR)
    env_w_prior = envelope_worst(EXISTING_HEAD_COV_PRIOR)
    reg_bool("11_existing_head_no_free_500_even_gate_lifted",
             (env_c_prior < TARGET) and (env_w_prior < TARGET))

    # (12) PPL-only central lift within the +0.031 retrain budget; worst lift over it (wirbel #343 deliverable2).
    reg_bool("12_pplonly_central_within_budget_worst_over",
             (PPLONLY_LIFT_CENTRAL <= RETRAIN_LIFT_BUDGET_336)
             and (PPLONLY_LIFT_WORST > RETRAIN_LIFT_BUDGET_336))

    # (13) retrain ROI: a 4-lever retrain clears the identity bar with high probability (lawine #339)
    #      -- this is WHY the ~1 GPU-hr gap-sizing read is worth firing before a GPU-weeks retrain.
    reg_bool("13_retrain_clears_identity_high_prob",
             (P_RETRAIN_CLEARS_IDENTITY_INDEP > 0.5) and (P_RETRAIN_CLEARS_IDENTITY_CORR05 > 0.5))

    # (14) THIRD independent cap confirmation: fern #349 (u8vmtji0) FlashInfer-BI restoration_ceiling
    #      473.53 corroborates the wirbel #343 strict ceiling 473.53 to display precision (reg_bool so
    #      the rounded-vs-unrounded delta does NOT pollute the TEST max_abs_reproduction_residual).
    reg_bool("14_fern349_restoration_ceiling_corroborates_473",
             abs(STRICT_CEILING_CENTRAL - FERN349_RESTORATION_CEILING) < 0.01)

    residuals = [c["resid"] for c in checks.values() if isinstance(c.get("resid"), float)]
    max_abs_reproduction_residual = max(residuals) if residuals else 0.0
    strict_keys = ["01_identity_bar_derived_from_611", "02_strict_ceiling_central_reproduces_473",
                   "03_strict_ceiling_worst_reproduces_448", "04_strict_gap_to_500_reproduces_26p47",
                   "05_gap_to_identity_bar_reproduces_0p031", "06_pplonly_cstar_central_roundtrips_500",
                   "07_pplonly_cstar_worst_roundtrips_500", "08a_pplonly_lift_central",
                   "08b_pplonly_lift_worst"]
    max_strict_residual = max(checks[k]["resid"] for k in strict_keys)

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "checks": checks,
        "identity_bar_derived": t_solved,
        "envelope_central_at_prior": env_c_prior,
        "envelope_worst_at_prior": env_w_prior,
        "max_abs_reproduction_residual": max_abs_reproduction_residual,
        "max_strict_residual": max_strict_residual,
        "arithmetic_all_pass": bool(all_pass),
    }


# --------------------------------------------------------------------------- #
# HALF 2 -- RUNBOOK.md completeness (the single-approval block is human-ready).
# --------------------------------------------------------------------------- #
def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def evaluate_completeness(text: str) -> dict[str, bool]:
    c: dict[str, bool] = {}

    # 1. Strict-world header + the #124 reversal premise.
    c["hdr_strict_world"] = _has(text, "strict", "identity") and _has(text, "0.9213")
    c["hdr_124_reversal"] = ("#124" in text) and _has(text, "10:56") and ("#192" in text)
    c["hdr_diagnostic_not_gate"] = _has(text, "diagnostic") and _has(text, "not a") and _has(text, ">500")
    c["hdr_strict_cap_473"] = ("473.5" in text) and _has(text, "supply-cap") and ("< 500" in text)

    # 2. The EXACT single-approval HF-job command + the metrics it returns.
    c["cmd_launch"] = _has(text, "train.py", "--submission", SUBMISSION, "--launch", "--wait")
    c["cmd_method"] = METHOD_TAG in text
    c["cmd_do_not_run"] = _has(text, "do not") and (_has(text, "human") and _has(text, "approv"))
    c["job_returns_coverage"] = _has(text, "uncond_top4") or _has(text, "coverage")
    c["job_returns_greedy_rate"] = _has(text, "greedy", "identity", "rate")
    c["job_returns_per_depth"] = _has(text, "per-depth") and ("a1" in text.lower() or "alpha" in text.lower())

    # 3. STRICT GO/NO-GO bars (the two flips vs the PPL-only world).
    c["bar_coverage_identity"] = ("0.9213" in text) and ("0.9213011665456927" in text)
    c["bar_greedy_hard_gate"] = _has(text, "greedy") and _has(text, "== 1.0") and (
        _has(text, "hard") and _has(text, "co-equal"))
    c["bar_ppl"] = ("2.42" in text) and ("2.3772" in text)
    c["bar_completed_128"] = ("128" in text)
    c["bars_nan_clean_echo"] = _has(text, "0.9213011665456927") and ("473.5295953446407" in text)

    # 4. Pre-flight gate (#338): C1/C3/C4 green, C2 is the one this read closes.
    c["preflight_c1c3c4_green"] = all(g in text for g in ("C1", "C3", "C4")) and _has(text, "green")
    c["preflight_c2_residual"] = ("C2" in text) and _has(text, "bf16") and _has(text, "greedy")
    c["preflight_dryrun_cite"] = (DRYRUN_WANDB in text) and ("#338" in text)

    # 5. The read's honest VALUE (3 things) + expected NO-GO.
    c["value_greedy_confirm"] = _has(text, "greedy-identity") and _has(text, "confirm")
    c["value_c2_closure"] = _has(text, "C2") and _has(text, "clos")
    c["value_gap_sizing"] = _has(text, "gap") and ("0.031" in text) and _has(text, "siz")
    c["value_expected_nogo"] = _has(text, "0.8903") and ("0.06" in text) and _has(text, "no-go")

    # 6. PPL-only CONTINGENCY block (only if #192 is ever lifted).
    c["contingency_block"] = _has(text, "contingency") and _has(text, "if #192") and (
        "0.9089" in text) and ("0.9256" in text)
    c["contingency_labeled_not_live"] = _has(text, "not the live") or _has(text, "not live") or _has(
        text, "only if")

    # 7. Strict >500 reality + the ceiling-lift alternative (advisor honest-add).
    c["strict_500_reality"] = _has(text, "batch-invariant") and _has(text, "verify kernel") and _has(
        text, "e[t]")
    c["ceiling_lift_alt"] = _has(text, "ceiling-lift") and (_has(text, "denken") or _has(text, "stark"))
    # 7b. The advisor's 11:29:49Z pointers: fern #349 third-cap corroboration + where the live strict
    #     >500 search actually moved (the frontier ladder + the in-flight levers), so this diagnostic
    #     points the right way.
    c["strict_500_third_cap_fern349"] = (FERN349_WANDB in text) and ("fern #349" in text) and (
        "473.53" in text)
    c["strict_500_frontier_ladder"] = _has(text, "frontier") and ("357.32" in text) and ("165.44" in text)
    c["strict_500_inflight_levers"] = all(p in text for p in ("#354", "#355", "#357", "#359"))

    # 8. No-side-effect flags recorded.
    c["flags_no_hf_job"] = _has(text, "no_hf_job") or _has(text, "no hf job")
    c["flags_no_launch"] = _has(text, "no_launch") or _has(text, "no launch")
    c["flags_no_served"] = _has(text, "no_served_file_change") or _has(text, "no served-file change")
    c["flags_zero_tps"] = ("481.53" in text) and _has(text, "0 tps") and _has(text, "unchanged")

    # 9. §0 checkpoint-publish blocker + the 4 pre-launch gates (single-approval readiness).
    c["s0_publish_blocker"] = _has(text, "publish") and (HEAD_WANDB in text) and _has(text, "smoke")
    c["four_gates"] = _has(text, "DRAFTER_SHA256") and _has(text, "/v1/models") and _has(
        text, "approval") and _has(text, "manifest")
    c["approval_title"] = APPROVAL_TITLE in text

    # 10. Banked-anchor citations (assemble, do not re-derive).
    anchors = [HEAD_WANDB, HEAD_TRAIN_WANDB, CONVERTER_WANDB, DRYRUN_WANDB, READSPEC_WANDB,
               WIRBEL343_WANDB, LAWINE330_WANDB, LAWINE339_WANDB, FERN349_WANDB]
    c["cite_all_wandb_anchors"] = all(a in text for a in anchors)
    prs = ["#322", "#333", "#338", "#343", "#330", "#339", "#34", "#349"]
    c["cite_all_prs"] = all(p in text for p in prs)
    c["cite_converter_script"] = "convert_eagle3_to_safetensors.py" in text

    return c


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    arith = arithmetic_reproduction()

    runbook_exists = RUNBOOK_PATH.exists()
    runbook_text = RUNBOOK_PATH.read_text(encoding="utf-8") if runbook_exists else ""
    completeness = evaluate_completeness(runbook_text)
    completeness_all_pass = bool(runbook_exists and all(completeness.values()))

    # pre-flight gate: C1/C3/C4 green at 0-GPU per ubel #338.
    preflight_c1c3c4_green = all(PREFLIGHT_LEDGER[g]["closed"] for g in PREFLIGHT_GREEN_AT_0GPU)
    c2_is_the_residual = (not PREFLIGHT_LEDGER["C2"]["closed"]) and (
        "greedy-identity" in PREFLIGHT_LEDGER["C2"]["name"])

    # no-side-effect flags recorded.
    flags_recorded = bool(NO_HF_JOB and NO_LAUNCH and NO_SERVED_FILE_CHANGE and NO_GPU)

    # NaN-clean walk over the arithmetic payload.
    def _finite(x: Any) -> bool:
        return not (isinstance(x, float) and not math.isfinite(x))
    nan_clean = all(_finite(ch.get("value")) and _finite(ch.get("resid"))
                    for ch in arith["checks"].values()) and _finite(
        arith["max_abs_reproduction_residual"])

    read_runbook_self_test_passes = bool(
        arith["arithmetic_all_pass"] and completeness_all_pass and nan_clean
        and preflight_c1c3c4_green and c2_is_the_residual and flags_recorded)

    # single-approval readiness: the human can fire ONE job with no further design work.
    single_approval_keys = ["cmd_launch", "cmd_method", "cmd_do_not_run", "approval_title",
                            "s0_publish_blocker", "four_gates", "bar_coverage_identity",
                            "bar_greedy_hard_gate", "bar_ppl", "bar_completed_128"]
    single_approval_block_complete = all(completeness.get(k, False) for k in single_approval_keys)
    read_is_single_approval_ready = bool(
        read_runbook_self_test_passes and single_approval_block_complete)

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    n_arith = len(arith["checks"])
    n_arith_pass = sum(1 for c in arith["checks"].values() if c["pass"])
    n_complete = len(completeness)
    n_complete_pass = sum(1 for v in completeness.values() if v)

    return {
        "pr": 350, "issue": 319, "agent": "ubel", "kind": "eagle3_read_runbook",
        "world": "strict-identity (#192 live; #124 reversed on #319 2026-06-15T10:56:17Z)",
        "analysis_only": True, "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "primary_metric_name": "read_runbook_self_test_passes",
        "read_runbook_self_test_passes": read_runbook_self_test_passes,
        "read_is_single_approval_ready": read_is_single_approval_ready,
        "test_metric_name": "max_abs_reproduction_residual",
        "max_abs_reproduction_residual": arith["max_abs_reproduction_residual"],
        "max_strict_residual": arith["max_strict_residual"],
        "arithmetic_all_pass": arith["arithmetic_all_pass"],
        "completeness_all_pass": completeness_all_pass,
        "single_approval_block_complete": single_approval_block_complete,
        "preflight_c1c3c4_green": preflight_c1c3c4_green,
        "c2_is_the_residual": c2_is_the_residual,
        "flags_recorded": flags_recorded,
        "nan_clean": nan_clean,
        "runbook_path": str(RUNBOOK_PATH.relative_to(REPO_ROOT)),
        "runbook_exists": runbook_exists,
        "runbook_bytes": len(runbook_text.encode("utf-8")),
        "n_arith_conditions": n_arith, "n_arith_pass": n_arith_pass,
        "n_completeness_conditions": n_complete, "n_completeness_pass": n_complete_pass,
        "identity_bar_derived": arith["identity_bar_derived"],
        "envelope_central_at_prior": arith["envelope_central_at_prior"],
        "envelope_worst_at_prior": arith["envelope_worst_at_prior"],
        "arithmetic_checks": arith["checks"],
        "completeness_checks": completeness,
        "peak_mem_mib": peak_mem_mib,
        # ---- the single-approval block, machine-readable ----
        "single_approval_block": {
            "approval_title": APPROVAL_TITLE,
            "launch_cmd": LAUNCH_CMD,
            "submission": SUBMISSION,
            "method_tag": METHOD_TAG,
            "hardware": "a10g-small (sm_86), vLLM 0.22.1rc1.dev307+g3e8afdf78",
            "input_head": f"fern #34 {HEAD_WANDB} (train {HEAD_TRAIN_WANDB}); convert via "
                          f"convert_eagle3_to_safetensors.py (ubel #333 {CONVERTER_WANDB})",
            "job_returns": JOB_RETURNS,
            "no_gpu": NO_GPU, "no_hf_job": NO_HF_JOB, "no_launch": NO_LAUNCH,
            "no_served_file_change": NO_SERVED_FILE_CHANGE,
        },
        "go_no_go_strict": {
            "world": "strict-identity (LIVE)",
            "coverage_bar_uncond_top4": IDENTITY_BAR,
            "greedy_identity_rate_bar": 1.0,
            "greedy_identity_is_hard_co_equal_gate": True,
            "ppl_bar": PPL_GATE, "ppl_deployed": PPL_DEPLOYED, "ppl_margin": PPL_MARGIN,
            "completed_bar": 128,
            "strict_ceiling_central": STRICT_CEILING_CENTRAL,
            "strict_ceiling_worst": STRICT_CEILING_WORST,
            "strict_500_reachable": STRICT_500_REACHABLE,
            "strict_gap_to_500": STRICT_GAP_TO_500,
            "fern349_restoration_ceiling": FERN349_RESTORATION_CEILING,
            "cap_corroborated_by": "wirbel #343 (kklof4wr) + denken #332 + fern #349 (u8vmtji0) -- three independent",
            "read_is_diagnostic_not_500_gate": True,
            "c2_caveat": PREFLIGHT_LEDGER["C2"]["name"],
            "expected_outcome": "NO-GO (gap-sized): existing head ~0.8903 < 0.9213, P(clear)~0.06",
        },
        "ppl_only_contingency": {
            "label": "carried ONLY for 'if #192 is ever lifted' (#124 world); NOT the live bars",
            "c_star_central": PPLONLY_CSTAR_CENTRAL, "c_star_worst": PPLONLY_CSTAR_WORST,
            "lift_central": PPLONLY_LIFT_CENTRAL, "lift_worst": PPLONLY_LIFT_WORST,
            "env_at_prior_central": PPLONLY_ENV_AT_PRIOR_CENTRAL,
            "env_at_prior_worst": PPLONLY_ENV_AT_PRIOR_WORST,
            "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET_336,
        },
        "preflight_ledger": PREFLIGHT_LEDGER,
        "existing_head": {
            "cov_prior": EXISTING_HEAD_COV_PRIOR, "p_clears_identity": EXISTING_HEAD_P_CLEARS_IDENTITY,
            "gap_to_identity_bar": GAP_TO_IDENTITY_BAR, "record_bernoulli_se": RECORD_BERNOULLI_SE,
            "verdict": "LIKELY-MISSES (expected NO-GO with a measured gap)",
        },
        "retrain_context": {
            "p_clears_identity_independent": P_RETRAIN_CLEARS_IDENTITY_INDEP,
            "p_clears_identity_correlated_0p5": P_RETRAIN_CLEARS_IDENTITY_CORR05,
            "min_lift": RETRAIN_MIN_LIFT, "roi": "JUSTIFIED",
            "note": "the read sizes the exact starting gap before a P~0.84 retrain -- ~1 GPU-hr insurance",
        },
        "strict_500_search_pointers": {
            "label": "advisor-provided DIRECTIONAL pointers (2026-06-15T11:29:49Z); cited as handed off, "
                     "NOT inspected/re-derived/reused in arithmetic. This diagnostic confirms the EAGLE-3 "
                     "substrate's strict-identity status UNDERNEATH the live >500 search below.",
            "frontier_ladder": STRICT_500_FRONTIER_LADDER,
            "inflight_levers": STRICT_500_INFLIGHT_LEVERS,
            "third_cap_corroboration": f"fern #349 ({FERN349_WANDB}) FlashInfer-BI restoration_ceiling "
                                       f"{FERN349_RESTORATION_CEILING} confirms wirbel #343 strict ceiling",
        },
        "constants": {
            "target": TARGET, "baseline_tps": BASELINE_TPS, "identity_bar": IDENTITY_BAR,
            "ppl_gate": PPL_GATE, "ppl_deployed": PPL_DEPLOYED, "supply_floor": SUPPLY_FLOOR,
            "lambda_central": LAMBDA_CENTRAL, "lambda_worst": LAMBDA_WORST,
            "e_t_build_free": E_T_BUILD_FREE,
            "strict_ceiling_central": STRICT_CEILING_CENTRAL, "strict_ceiling_worst": STRICT_CEILING_WORST,
            "strict_500_reachable": STRICT_500_REACHABLE,
            "existing_head_cov_prior": EXISTING_HEAD_COV_PRIOR,
            "existing_head_p_clears_identity": EXISTING_HEAD_P_CLEARS_IDENTITY,
            "gap_to_identity_bar": GAP_TO_IDENTITY_BAR,
            "pplonly_cstar_central": PPLONLY_CSTAR_CENTRAL, "pplonly_cstar_worst": PPLONLY_CSTAR_WORST,
            "p_retrain_clears_identity_independent": P_RETRAIN_CLEARS_IDENTITY_INDEP,
        },
        "provenance": (
            f"STRICT bars: lawine #330 ({LAWINE330_WANDB}) identity bar 0.9213011665456927 / cov prior "
            f"0.8902659519153152 / P(clear)~0.0603 / gap 0.031035 x wirbel #343 ({WIRBEL343_WANDB}) strict "
            f"ceiling 473.5295953446407 (supply floor 0.09103, denken #332; corroborated by fern #349 "
            f"{FERN349_WANDB} FlashInfer-BI restoration_ceiling 473.53) / strict_500_reachable=False / "
            f"PPL-only c*_central 0.9089363308345582 / c*_worst 0.925603648491971 (CONTINGENCY) x lawine "
            f"#339 ({LAWINE339_WANDB}) retrain clears 0.9213 P~0.843 (indep). PRE-FLIGHT: ubel #338 "
            f"({DRYRUN_WANDB}) C1/C3/C4 CLOSED-AT-0-GPU, C2 (fp32->bf16 + served greedy-identity) GPU-residual; "
            f"ubel #333 ({CONVERTER_WANDB}) converter; ubel #322 ({READSPEC_WANDB}) read protocol; fern #34 "
            f"head {HEAD_WANDB} (train {HEAD_TRAIN_WANDB}); PR #52 frontier {FRONTIER_WANDB} 481.53. "
            f"All run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
        "scope": (
            "LOCAL CPU-only self-test over banked constants + the RUNBOOK.md companion. 0 TPS; BASELINE "
            "481.53 untouched; greedy/PPL untouched. NO GPU / vLLM / model forward / HF Job / submission / "
            "served-file change. Authorizes NOTHING. NOT a launch. This PACKAGES the #319 strict read so "
            "the human can fire ONE a10g-small job with zero further advisor round-trips."),
    }


# --------------------------------------------------------------------------- #
def print_report(p: dict[str, Any]) -> None:
    print("\n" + "=" * 94, flush=True)
    print("EAGLE-3 READ RUNBOOK -- single-approval HF-job block (PR #350, STRICT world)", flush=True)
    print("=" * 94, flush=True)
    print("HALF 1 -- strict-world GO/NO-GO arithmetic reproduction (<= 1e-6):", flush=True)
    for k, ch in p["arithmetic_checks"].items():
        tag = "PASS" if ch["pass"] else "FAIL"
        print(f"  [{tag}] {k:<46} value={ch['value']:.12g} target={ch['target']:.12g} "
              f"resid={ch['resid']:.2e} (tol {ch['tol']:.0e})", flush=True)
    print(f"  -> identity bar DERIVED from E[T]=6.11 = {p['identity_bar_derived']:.16f}", flush=True)
    print(f"  -> strict ceiling central / worst      = {p['go_no_go_strict']['strict_ceiling_central']:.4f}"
          f" / {p['go_no_go_strict']['strict_ceiling_worst']:.4f}  (< 500: supply-capped)", flush=True)
    print(f"  -> PPL-only env @prior central / worst = {p['envelope_central_at_prior']:.4f}"
          f" / {p['envelope_worst_at_prior']:.4f}  (both < 500: no free 500 even gate-lifted)", flush=True)
    print(f"  -> max strict residual                 = {p['max_strict_residual']:.2e}", flush=True)
    print("-" * 94, flush=True)
    print(f"HALF 2 -- RUNBOOK.md completeness ({p['n_completeness_pass']}/{p['n_completeness_conditions']} "
          f"conditions on {p['runbook_path']}, {p['runbook_bytes']} bytes):", flush=True)
    failed = [k for k, v in p["completeness_checks"].items() if not v]
    if failed:
        print(f"  FAILED completeness conditions: {failed}", flush=True)
    else:
        print("  all completeness conditions hold.", flush=True)
    print("-" * 94, flush=True)
    print(f"  pre-flight C1/C3/C4 green (ubel #338): {p['preflight_c1c3c4_green']}  "
          f"(C2 = {p['go_no_go_strict']['c2_caveat']})", flush=True)
    print(f"  no_hf_job/no_launch/no_served_file_change recorded: {p['flags_recorded']}", flush=True)
    print(f"PRIMARY read_runbook_self_test_passes   = {p['read_runbook_self_test_passes']}", flush=True)
    print(f"SECOND  read_is_single_approval_ready   = {p['read_is_single_approval_ready']}", flush=True)
    print(f"TEST    max_abs_reproduction_residual   = {p['max_abs_reproduction_residual']:.2e}", flush=True)
    print(f"nan_clean = {p['nan_clean']}   peak_mem_mib = {p['peak_mem_mib']:.2f}", flush=True)
    print("=" * 94 + "\n", flush=True)


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
        print(f"[read-runbook] wandb logging unavailable (analysis unaffected): {exc}", flush=True)
        return None

    run = init_wandb_run(
        job_type="validity-analytic", agent="ubel", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3", "read-runbook", "launch-spec", "strict-identity", "0-tps", "issue-319",
              "single-approval", "bank-the-analysis"],
        config={
            "pr": 350, "issue": 319, "wandb_group": args.wandb_group, "world": payload["world"],
            "runbook_path": payload["runbook_path"], "identity_bar": IDENTITY_BAR,
            "strict_ceiling_central": STRICT_CEILING_CENTRAL, "strict_500_reachable": STRICT_500_REACHABLE,
            "ppl_gate": PPL_GATE, "existing_head_cov_prior": EXISTING_HEAD_COV_PRIOR,
            "existing_head_p_clears_identity": EXISTING_HEAD_P_CLEARS_IDENTITY,
            "baseline_tps": BASELINE_TPS, "tps_added_by_this_card": 0,
            "provenance": payload["provenance"], "scope": payload["scope"],
        },
    )
    if run is None:
        print("[read-runbook] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "read_runbook_self_test_passes": int(bool(payload["read_runbook_self_test_passes"])),
        "read_is_single_approval_ready": int(bool(payload["read_is_single_approval_ready"])),
        "max_abs_reproduction_residual": payload["max_abs_reproduction_residual"],
        "max_strict_residual": payload["max_strict_residual"],
        "arithmetic_all_pass": int(bool(payload["arithmetic_all_pass"])),
        "completeness_all_pass": int(bool(payload["completeness_all_pass"])),
        "single_approval_block_complete": int(bool(payload["single_approval_block_complete"])),
        "preflight_c1c3c4_green": int(bool(payload["preflight_c1c3c4_green"])),
        "c2_is_the_residual": int(bool(payload["c2_is_the_residual"])),
        "flags_recorded": int(bool(payload["flags_recorded"])),
        "n_arith_pass": payload["n_arith_pass"], "n_arith_conditions": payload["n_arith_conditions"],
        "n_completeness_pass": payload["n_completeness_pass"],
        "n_completeness_conditions": payload["n_completeness_conditions"],
        "identity_bar": IDENTITY_BAR, "strict_ceiling_central": STRICT_CEILING_CENTRAL,
        "strict_ceiling_worst": STRICT_CEILING_WORST, "strict_gap_to_500": STRICT_GAP_TO_500,
        "ppl_gate": PPL_GATE, "ppl_deployed": PPL_DEPLOYED,
        "existing_head_cov_prior": EXISTING_HEAD_COV_PRIOR,
        "existing_head_p_clears_identity": EXISTING_HEAD_P_CLEARS_IDENTITY,
        "gap_to_identity_bar": GAP_TO_IDENTITY_BAR,
        "pplonly_cstar_central": PPLONLY_CSTAR_CENTRAL, "pplonly_cstar_worst": PPLONLY_CSTAR_WORST,
        "p_retrain_clears_identity_independent": P_RETRAIN_CLEARS_IDENTITY_INDEP,
        "envelope_central_at_prior": payload["envelope_central_at_prior"],
        "envelope_worst_at_prior": payload["envelope_worst_at_prior"],
        "runbook_bytes": payload["runbook_bytes"], "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"], "tps_added_by_this_card": 0,
    }
    summary.update({f"arith_{k}": int(bool(v["pass"])) for k, v in payload["arithmetic_checks"].items()})
    summary.update({f"complete_{k}": int(bool(v)) for k, v in payload["completeness_checks"].items()})
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_read_runbook_result", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[read-runbook] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-read-runbook")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    payload = run()
    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[read-runbook] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    payload["wandb_run_ids"] = [rid] if rid else []
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    gate = bool(payload["read_runbook_self_test_passes"])
    print(f"  PRIMARY read_runbook_self_test_passes = {gate}", flush=True)
    print(f"  SECOND  read_is_single_approval_ready = {payload['read_is_single_approval_ready']}", flush=True)
    print(f"  TEST    max_abs_reproduction_residual = {payload['max_abs_reproduction_residual']:.2e}", flush=True)
    print(f"  wandb run = {rid}", flush=True)
    if args.self_test:
        print(f"[read-runbook] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
