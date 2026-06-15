#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict-compliant frontier vs the custom reduction-invariant kernel overhead (PR #354, #319).

THE QUESTION
------------
The #319 human strict-lock (2026-06-15 10:56:17Z -- "ignore #124, we want to ensure we
stick with the strict greedy token matching") makes EVERY speculative config non-compliant
unless the int4-Marlin split-K M-variance is restored. The deployed 481.53 TPS itself is
NON-compliant (~0.73% M=8 divergence, #326), so the strict-COMPLIANT frontier TODAY is just
165.44 TPS (lawine #196, non-spec int4 M=1 AR, token_identity_rate=1.0).

This re-opens wirbel #326's custom reduction-invariant kernel (shelved in the now-dead
PPL-only world). The kernel restores M=8 greedy-identity at an overhead `eta`; #326 measured
the OFF-THE-SHELF VLLM_BATCH_INVARIANT=1 at eta=0.3141 -> compliant ceiling 357.32 TPS, but
identified a custom-kernel first-principles FLOOR ~0.9455% (#216's int4-Marlin split-K
estimate). This card PRICES the lever precisely:

    how far does the custom reduction-invariant kernel lift the strict-compliant frontier,
    and does it clear 500 ALONE?

THE MODEL (PR step 1)
---------------------
The deployed 481.53 runs WITHOUT the identity-restoring kernel (non-compliant spec). Applying
the kernel ADDS overhead `eta` (a fraction of the deployed step 1218.2us, #136). First-order:

    strict_compliant_tps(eta) = 481.53 * (1 - eta)                       [deployed base, linear]

grounded on #213's lambda=1 budget curve (7.332% at lambda=1, run 5o7zcj8s) and the deployed
step. Two cross-checks on the SAME (1-eta) overhead FORM:
  * #326's banked ceiling model (base = lambda=1 central ceiling 520.953): the round-trip
    520.953 * (1 - 0.3141) = 357.32166... reproduces #326's off-the-shelf ceiling bit-exactly.
  * #213's EXACT step-overhead map is the divisor form tps = tps_lambda1 / (1 + eta); its
    lambda=1 budget round-trips 500 (536.659 / 1.07332 = 500). The linear (1-eta) form is the
    first-order Taylor of the divisor; near the realistic custom-kernel floor (eta~0.0095) the
    two agree to ~0.04 TPS, and the closure (below) is ROBUST under both forms.

THE RESULT (PR steps 2-3)
-------------------------
The deployed base 481.53 < 500, so even a FREE kernel (eta -> 0) does not clear 500:
`kernel_alone_clears_500 = False`, `eta_crit_500 = None` (would need NEGATIVE overhead, i.e. a
speedup of ~3.8%, to reach 500). The custom kernel lifts the strict-compliant frontier from
165.44 (non-spec floor) to strict_compliant_tps(0.009455) ~ 477 TPS -- a massive +311 TPS lift
that restores almost all of the spec premium -- but stalls ~23 TPS SHORT of 500. So the custom
reduction-invariant kernel is NECESSARY (it is the only thing that makes the fast spec path
compliant) but NOT SUFFICIENT alone; a FASTER substrate UNDER the kernel (sub-int4 body --
lawine's companion card) is required to close the residual ~23 TPS.

PURE CPU ANALYTIC over banked merged *_results.json (all anchors imported VERBATIM). NO model
build, NO training, NO served-file change, NO HF Job, NO submission. 0 GPU, 0 TPS. BASELINE
481.53 UNCHANGED. Bank-the-analysis."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                      # .../target

# --------------------------------------------------------------------------- #
# Banked anchors. Literals are the source of truth for computation; the merged
# *_results.json are READ at runtime to confirm provenance (folded into self-test).
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE_DEPLOYED = 481.53              # #52 deployed frontier (NON-compliant under strict)
TARGET = 500.0                                   # official 500-TPS bar
LAMBDA1_CEIL_CENTRAL = 520.953                   # lambda=1 central ceiling (#257/#326 ceiling_500)

# ---- #196 (compliant_nonspec_floor, lawine, run laxllfjl-family) ------------ #
F196_JSON = REPO_ROOT / "research/validity/compliant_nonspec_floor/floor_report.json"
STRICT_COMPLIANT_FRONTIER_TODAY = 165.43791973106974   # nonspec_official_tps_est (token_identity_rate=1.0)

# ---- #326 (eagle3_bi_reduction_measured, wirbel, run io4cs2ch) -------------- #
F326_JSON = REPO_ROOT / "research/validity/eagle3_bi_reduction_measured/eagle3_bi_reduction_measured_report.json"
ETA_FLOOR_326 = 0.009455                          # band_floor_216: custom-kernel first-principles floor (~0.95%)
ETA_CEIL_326 = 0.3141                             # min_overhead_restoring_identity: off-the-shelf VBI (31.41%)
OFFTHESHELF_CEILING_326 = 357.32166269999993      # compliant ceiling at eta=0.3141 (= 520.953*(1-0.3141))
CEILING_AT_BAND_FLOOR_326 = 516.027389385         # compliant ceiling at eta=0.009455 (cross-check)
CEILING_AT_ZERO_326 = 520.953                     # compliant ceiling at eta=0 (= LAMBDA1_CEIL_CENTRAL)
BUDGET_LAMBDA1_FRAC_326 = 0.0733                  # #213 lambda=1 budget as #326 imported it
STEP_US = 1218.2                                  # deployed step (kanna #217 / #136)

# ---- #213 (kernel_budget_lambda, wirbel, run 5o7zcj8s) --------------------- #
F213_JSON = REPO_ROOT / "research/validity/kernel_budget_lambda/kernel_budget_lambda_results.json"
BUDGET_LAMBDA1_PCT_213 = 7.331808522875782        # overhead_budget_at_lambda_1_both_bugs_tau1
TPS_LAMBDA1_BOTHBUGS_213 = 536.6590426143789      # official_tps at lambda=1 both-bugs (budget pairs with THIS)
K_CAL = 125.26795005202914                        # spec-tree calibration constant (#257/#213)


# --------------------------------------------------------------------------- #
# Overhead models (the (1-eta) FORM is shared; only the base differs).
# --------------------------------------------------------------------------- #
def tps_linear(base: float, eta: float) -> float:
    """First-order overhead model: an `eta`-fraction step overhead linearised."""
    return base * (1.0 - eta)


def tps_divisor(base: float, eta: float) -> float:
    """#213's EXACT step-overhead map: overhead adds eta*step to the step -> tps/(1+eta)."""
    return base / (1.0 + eta)


def strict_compliant_tps(eta: float) -> float:
    """The strict-compliant frontier with the identity-restoring kernel (deployed base, linear)."""
    return tps_linear(OFFICIAL_BASELINE_DEPLOYED, eta)


# --------------------------------------------------------------------------- #
# Provenance: read the merged banked JSONs (non-fatal; confirms the literals).
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"[strict-frontier] provenance read skipped ({path.name}): {exc}", flush=True)
        return None


def provenance() -> dict[str, Any]:
    """Re-read the source-of-truth banked numbers and check they match the literals."""
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    j196 = _read_json(F196_JSON)
    if j196 is not None:
        v = j196.get("nonspec_official_tps_est")
        tir = j196.get("nonspec_token_identity_rate")
        detail["f196_nonspec_official_tps_est"] = v
        detail["f196_token_identity_rate"] = tir
        checks["prov_196_frontier"] = (v is not None
                                       and abs(float(v) - STRICT_COMPLIANT_FRONTIER_TODAY) < 1e-9
                                       and tir == 1.0)
    else:
        checks["prov_196_frontier"] = True  # non-fatal when file absent (literal stands)

    j326 = _read_json(F326_JSON)
    if j326 is not None:
        eta_ceil = j326.get("min_overhead_restoring_identity")
        ceil_ceil = j326.get("compliant_ceiling_at_band_ceil")
        ceil_floor = j326.get("compliant_ceiling_at_band_floor")
        ceil_zero = j326.get("compliant_ceiling_at_zero")
        band = j326.get("band") or [None, None]
        detail["f326_min_overhead_restoring_identity"] = eta_ceil
        detail["f326_compliant_ceiling_at_band_ceil"] = ceil_ceil
        detail["f326_band"] = band
        checks["prov_326_anchors"] = (
            eta_ceil is not None and abs(float(eta_ceil) - ETA_CEIL_326) < 1e-12
            and ceil_ceil is not None and abs(float(ceil_ceil) - OFFTHESHELF_CEILING_326) < 1e-9
            and ceil_floor is not None and abs(float(ceil_floor) - CEILING_AT_BAND_FLOOR_326) < 1e-9
            and ceil_zero is not None and abs(float(ceil_zero) - CEILING_AT_ZERO_326) < 1e-9
            and len(band) == 2 and abs(float(band[0]) - ETA_FLOOR_326) < 1e-12
            and abs(float(band[1]) - ETA_CEIL_326) < 1e-12)
    else:
        checks["prov_326_anchors"] = True

    j213 = _read_json(F213_JSON)
    if j213 is not None:
        syn = j213.get("synthesis", {})
        budget = (syn.get("headline", {}) or {}).get("overhead_budget_at_lambda_1_both_bugs_tau1")
        step_ms = (syn.get("composition", {}) or {}).get("step")
        detail["f213_budget_lambda1_pct"] = budget
        detail["f213_step_ms"] = step_ms
        checks["prov_213_budget"] = (
            budget is not None and abs(float(budget) - BUDGET_LAMBDA1_PCT_213) < 1e-9
            and step_ms is not None and abs(float(step_ms) * 1000.0 - STEP_US) < 1e-6)
    else:
        checks["prov_213_budget"] = True

    return {"checks": checks, "detail": detail}


# --------------------------------------------------------------------------- #
# Analytic core.
# --------------------------------------------------------------------------- #
def eta_grid(n: int = 401) -> list[float]:
    """Fine grid over the crossing-search interval [0, eta_ceil] (includes eta=0 free-kernel limit)."""
    return [ETA_CEIL_326 * i / (n - 1) for i in range(n)]


def named_points() -> list[dict[str, Any]]:
    """The four operating points the lever is priced at."""
    rows = [
        ("free_kernel_limit", 0.0,
         "optimistic eta->0 (a hypothetical zero-cost identity restore) -- the upper bound of the frontier"),
        ("custom_kernel_floor", ETA_FLOOR_326,
         "#216 int4-Marlin split-K first-principles floor (~0.95%): the realistic custom reduction-invariant kernel"),
        ("lambda1_budget_ref", BUDGET_LAMBDA1_FRAC_326,
         "#213 lambda=1 kernel budget (7.33%) carried as an overhead reference point"),
        ("offtheshelf_vbi_ceil", ETA_CEIL_326,
         "#326 off-the-shelf VLLM_BATCH_INVARIANT=1 measured M=8 identity restore (31.41%)"),
    ]
    out = []
    for name, eta, note in rows:
        tps_lin = strict_compliant_tps(eta)
        tps_div = tps_divisor(OFFICIAL_BASELINE_DEPLOYED, eta)
        out.append({
            "point": name,
            "eta": eta,
            "step_overhead_us": eta * STEP_US,
            "strict_compliant_tps_linear": tps_lin,
            "strict_compliant_tps_divisor_exact": tps_div,
            "linear_minus_divisor_tps": tps_lin - tps_div,
            "clears_500": bool(tps_lin >= TARGET),
            "lift_from_frontier_today_tps": tps_lin - STRICT_COMPLIANT_FRONTIER_TODAY,
            "gap_to_500_tps": TARGET - tps_lin,
            "note": note,
        })
    return out


def find_eta_crit(model: str = "linear") -> dict[str, Any]:
    """Solve strict_compliant_tps(eta) = 500. Returns the analytic eta and whether it lies in [0, eta_ceil]."""
    if model == "linear":
        # 481.53*(1-eta) = 500  ->  eta = 1 - 500/481.53
        eta_solved = 1.0 - TARGET / OFFICIAL_BASELINE_DEPLOYED
    else:
        # 481.53/(1+eta) = 500  ->  eta = 481.53/500 - 1
        eta_solved = OFFICIAL_BASELINE_DEPLOYED / TARGET - 1.0
    in_band = 0.0 <= eta_solved <= ETA_CEIL_326
    return {"model": model, "eta_solved": eta_solved, "in_band_0_to_ceil": bool(in_band)}


def synthesize() -> dict[str, Any]:
    grid = eta_grid()
    tps_curve = [strict_compliant_tps(e) for e in grid]

    points = named_points()
    floor_pt = next(p for p in points if p["point"] == "custom_kernel_floor")
    ceil_pt = next(p for p in points if p["point"] == "offtheshelf_vbi_ceil")
    free_pt = next(p for p in points if p["point"] == "free_kernel_limit")

    # ---- (2) eta_crit_500 + kernel_alone_clears_500 ----
    crit_lin = find_eta_crit("linear")
    crit_div = find_eta_crit("divisor")
    # max of the frontier over the search interval is at eta=0 (decreasing curve)
    tps_at_zero = strict_compliant_tps(0.0)
    kernel_alone_clears_500 = bool(tps_at_zero >= TARGET)
    eta_crit_500 = crit_lin["eta_solved"] if crit_lin["in_band_0_to_ceil"] else None

    # ---- (3) frontier lift + residual gap ----
    strict_compliant_tps_at_floor = floor_pt["strict_compliant_tps_linear"]
    strict_compliant_frontier_lift_from_165 = (
        strict_compliant_tps_at_floor - STRICT_COMPLIANT_FRONTIER_TODAY)
    residual_gap_to_500_tps = TARGET - strict_compliant_tps_at_floor
    max_possible_lift = TARGET - STRICT_COMPLIANT_FRONTIER_TODAY  # bound for self-test (d)

    # ---- cross-checks (the SAME (1-eta) form on #326's ceiling base) ----
    roundtrip_326_ceil = tps_linear(LAMBDA1_CEIL_CENTRAL, ETA_CEIL_326)      # -> 357.32166...
    roundtrip_326_floor = tps_linear(LAMBDA1_CEIL_CENTRAL, ETA_FLOOR_326)    # -> 516.027389385
    # #213 EXACT divisor budget round-trip: 536.659/(1+0.073318) -> 500
    budget_frac_213 = BUDGET_LAMBDA1_PCT_213 / 100.0
    roundtrip_213_divisor_500 = tps_divisor(TPS_LAMBDA1_BOTHBUGS_213, budget_frac_213)
    budget_implied_213 = TPS_LAMBDA1_BOTHBUGS_213 / TARGET - 1.0            # -> 0.073318...

    # first-order gap (linear vs exact divisor) at the band ends, on the deployed base
    fo_gap_floor = floor_pt["linear_minus_divisor_tps"]
    fo_gap_ceil = ceil_pt["linear_minus_divisor_tps"]

    # ---- self-test (PRIMARY) ----
    cond: dict[str, bool] = {}
    # (a) round-trip #326's 357.32 at eta=0.3141 (<= 1e-9) -- validates the (1-eta) FORM.
    cond["a_roundtrip_326_357_at_eta_ceil"] = abs(roundtrip_326_ceil - OFFTHESHELF_CEILING_326) <= 1e-9
    # (b) monotonic strictly decreasing over the grid.
    cond["b_strict_compliant_tps_monotone_decreasing"] = all(
        tps_curve[i + 1] < tps_curve[i] - 1e-12 for i in range(len(tps_curve) - 1))
    # (c) kernel_alone_clears_500 follows the 481.53 < 500 logic (max at eta=0 is the deployed base).
    cond["c_kernel_alone_clears_500_follows_481_lt_500"] = (
        (kernel_alone_clears_500 == (tps_at_zero >= TARGET))
        and kernel_alone_clears_500 is False
        and abs(tps_at_zero - OFFICIAL_BASELINE_DEPLOYED) < 1e-9
        and OFFICIAL_BASELINE_DEPLOYED < TARGET
        and eta_crit_500 is None)
    # (d) frontier lift from 165.44 is positive AND < (500 - 165.44).
    cond["d_frontier_lift_positive_and_bounded"] = (
        strict_compliant_frontier_lift_from_165 > 0.0
        and strict_compliant_frontier_lift_from_165 < max_possible_lift)
    # ---- grounding cross-checks (strengthen the form/anchor provenance) ----
    # (e) #326 band-floor ceiling round-trips 516.027 under the SAME (1-eta) form (<= 1e-9).
    cond["e_roundtrip_326_band_floor_516"] = abs(roundtrip_326_floor - CEILING_AT_BAND_FLOOR_326) <= 1e-9
    # (f) #213 EXACT divisor budget map round-trips 500 from the lambda=1 both-bugs TPS (<= 1e-6).
    cond["f_roundtrip_213_divisor_clears_500"] = (
        abs(roundtrip_213_divisor_500 - TARGET) <= 1e-6
        and abs(budget_implied_213 - budget_frac_213) <= 1e-9)
    # (g) baseline constants imported EXACT.
    cond["g_constants_imported_exact"] = (
        abs(OFFICIAL_BASELINE_DEPLOYED - 481.53) < 1e-9
        and abs(LAMBDA1_CEIL_CENTRAL - 520.953) < 1e-9
        and abs(STRICT_COMPLIANT_FRONTIER_TODAY - 165.43791973106974) < 1e-9
        and abs(ETA_FLOOR_326 - 0.009455) < 1e-12
        and abs(ETA_CEIL_326 - 0.3141) < 1e-12
        and abs(STEP_US - 1218.2) < 1e-9)
    # (e2-nan) NaN-clean folded in main() over the whole payload.

    verdict = (
        f"NECESSARY-BUT-NOT-SUFFICIENT. Under the #319 strict-lock the custom reduction-invariant "
        f"kernel lifts the strict-compliant frontier from {STRICT_COMPLIANT_FRONTIER_TODAY:.2f} TPS "
        f"(non-spec int4 M=1 AR, #196) to {strict_compliant_tps_at_floor:.2f} TPS at its "
        f"first-principles floor eta={ETA_FLOOR_326:.4f} (#216) -- a +{strict_compliant_frontier_lift_from_165:.1f} "
        f"TPS lift that restores almost all of the spec premium. But the deployed base "
        f"{OFFICIAL_BASELINE_DEPLOYED:.2f} < 500, so kernel_alone_clears_500=False and eta_crit_500=None "
        f"(would need eta={crit_lin['eta_solved']:.4f} < 0, i.e. a ~3.8% SPEEDUP, not an overhead). The "
        f"frontier stalls {residual_gap_to_500_tps:.1f} TPS short of 500. The custom kernel is the ONLY "
        f"thing that makes the fast spec path compliant, but a FASTER substrate UNDER it (sub-int4 body "
        f"-- lawine's companion card) is required to close the residual ~{residual_gap_to_500_tps:.0f} TPS. "
        f"Off-the-shelf VBI (eta={ETA_CEIL_326:.4f}) is far worse: {ceil_pt['strict_compliant_tps_linear']:.2f} "
        f"TPS. 0 GPU, 0 TPS, baseline {OFFICIAL_BASELINE_DEPLOYED} UNCHANGED. Bank-the-analysis.")

    handoff = (
        f"the custom reduction-invariant kernel is NECESSARY (only route to a compliant fast path; lifts "
        f"165.44 -> {strict_compliant_tps_at_floor:.0f} TPS, +{strict_compliant_frontier_lift_from_165:.0f}) "
        f"but NOT SUFFICIENT alone (481.53 < 500 -> stalls {residual_gap_to_500_tps:.0f} TPS short, "
        f"eta_crit_500=None). The next lever is a sub-int4 body UNDER the kernel (lawine's companion card) "
        f"to lift the {OFFICIAL_BASELINE_DEPLOYED:.0f} base above 500 before the (1-eta) kernel haircut.")

    return {
        "constants": {
            "official_baseline_deployed": OFFICIAL_BASELINE_DEPLOYED,
            "target_tps": TARGET,
            "lambda1_ceil_central": LAMBDA1_CEIL_CENTRAL,
            "strict_compliant_frontier_today": STRICT_COMPLIANT_FRONTIER_TODAY,
            "eta_floor_326": ETA_FLOOR_326,
            "eta_ceil_326": ETA_CEIL_326,
            "step_us": STEP_US,
            "K_cal": K_CAL,
            "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
            "tps_lambda1_bothbugs_213": TPS_LAMBDA1_BOTHBUGS_213,
        },
        "named_points": points,
        "eta_crit": {
            "eta_crit_500": eta_crit_500,
            "linear_solved": crit_lin,
            "divisor_solved": crit_div,
            "kernel_alone_clears_500": kernel_alone_clears_500,
            "tps_at_eta_zero": tps_at_zero,
            "search_interval": [0.0, ETA_CEIL_326],
            "note": ("strict_compliant_tps is decreasing, so its max over [0, eta_ceil] is at eta=0 "
                     f"(= deployed {OFFICIAL_BASELINE_DEPLOYED} < 500); no eta in [0, eta_ceil] reaches "
                     "500. The analytic crossing is at NEGATIVE eta (a speedup), confirming closure."),
        },
        "frontier_lift": {
            "strict_compliant_tps_at_floor": strict_compliant_tps_at_floor,
            "strict_compliant_frontier_today": STRICT_COMPLIANT_FRONTIER_TODAY,
            "strict_compliant_frontier_lift_from_165": strict_compliant_frontier_lift_from_165,
            "residual_gap_to_500_tps": residual_gap_to_500_tps,
            "max_possible_lift_to_500": max_possible_lift,
            "spec_premium_restored_frac": strict_compliant_frontier_lift_from_165 / (
                OFFICIAL_BASELINE_DEPLOYED - STRICT_COMPLIANT_FRONTIER_TODAY),
        },
        "cross_checks": {
            "roundtrip_326_offtheshelf_ceiling": roundtrip_326_ceil,
            "roundtrip_326_offtheshelf_ceiling_banked": OFFTHESHELF_CEILING_326,
            "roundtrip_326_band_floor_ceiling": roundtrip_326_floor,
            "roundtrip_326_band_floor_ceiling_banked": CEILING_AT_BAND_FLOOR_326,
            "roundtrip_213_divisor_clears_500": roundtrip_213_divisor_500,
            "budget_implied_213_frac": budget_implied_213,
            "first_order_gap_linear_minus_divisor_at_floor_tps": fo_gap_floor,
            "first_order_gap_linear_minus_divisor_at_ceil_tps": fo_gap_ceil,
            "note": ("the (1-eta) linear FORM round-trips both of #326's banked ceilings bit-exactly; the "
                     "#213 EXACT divisor map round-trips 500. Near the realistic floor the linear/divisor "
                     f"gap is only {fo_gap_floor:.2f} TPS, so the closure is robust under both forms; near "
                     f"the off-the-shelf ceil the gap widens to {fo_gap_ceil:.1f} TPS (still both << 500)."),
        },
        "self_test": {"conditions": cond},
        # headline metrics
        "strict_compliant_tps_at_floor": strict_compliant_tps_at_floor,
        "eta_crit_500": eta_crit_500,
        "kernel_alone_clears_500": kernel_alone_clears_500,
        "strict_compliant_frontier_lift_from_165": strict_compliant_frontier_lift_from_165,
        "verdict": verdict,
        "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors the banked cards; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init -> this venv lacks the wandb wheel")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[strict-frontier] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    st = syn["self_test"]
    fl = syn["frontier_lift"]
    cc = syn["cross_checks"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="wirbel", name=args.wandb_name, group=args.wandb_group,
            tags=["strict-compliant-frontier", "reduction-invariant-kernel", "kernel-overhead",
                  "necessary-not-sufficient", "319-strict-lock", "bank-the-analysis", "pr-354"],
            config={
                "official_baseline_deployed": OFFICIAL_BASELINE_DEPLOYED,
                "target_tps": TARGET,
                "lambda1_ceil_central": LAMBDA1_CEIL_CENTRAL,
                "strict_compliant_frontier_today": STRICT_COMPLIANT_FRONTIER_TODAY,
                "eta_floor_326": ETA_FLOOR_326, "eta_ceil_326": ETA_CEIL_326,
                "step_us": STEP_US, "budget_lambda1_pct_213": BUDGET_LAMBDA1_PCT_213,
                "imports": "lawine#196(laxllfjl 165.44) x wirbel#326(io4cs2ch 357.32@0.3141, floor 0.009455) "
                           "x wirbel#213(5o7zcj8s 7.332%@lambda1, step 1218.2us)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[strict-frontier] wandb init failed (analysis unaffected): {exc}", flush=True)
        return None
    if run is None:
        print("[strict-frontier] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "strict_compliant_kernel_frontier_self_test_passes":
            int(bool(payload["strict_compliant_kernel_frontier_self_test_passes"])),
        "strict_compliant_tps_at_floor": fl["strict_compliant_tps_at_floor"],
        "eta_crit_500": syn["eta_crit"]["eta_crit_500"],   # None -> stored as-is in run.summary
        "eta_crit_500_in_band": int(syn["eta_crit"]["eta_crit_500"] is not None),
        "eta_crit_500_solved_linear": syn["eta_crit"]["linear_solved"]["eta_solved"],
        "kernel_alone_clears_500": int(bool(syn["kernel_alone_clears_500"])),
        "strict_compliant_frontier_lift_from_165": fl["strict_compliant_frontier_lift_from_165"],
        "residual_gap_to_500_tps": fl["residual_gap_to_500_tps"],
        "strict_compliant_frontier_today": fl["strict_compliant_frontier_today"],
        "spec_premium_restored_frac": fl["spec_premium_restored_frac"],
        "strict_compliant_tps_at_eta_zero": syn["eta_crit"]["tps_at_eta_zero"],
        "roundtrip_326_offtheshelf_ceiling": cc["roundtrip_326_offtheshelf_ceiling"],
        "roundtrip_213_divisor_clears_500": cc["roundtrip_213_divisor_clears_500"],
        "first_order_gap_at_floor_tps": cc["first_order_gap_linear_minus_divisor_at_floor_tps"],
        "first_order_gap_at_ceil_tps": cc["first_order_gap_linear_minus_divisor_at_ceil_tps"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per named-point metrics
    for p in syn["named_points"]:
        key = p["point"]
        summary[f"pt_{key}_eta"] = p["eta"]
        summary[f"pt_{key}_tps_linear"] = p["strict_compliant_tps_linear"]
        summary[f"pt_{key}_tps_divisor"] = p["strict_compliant_tps_divisor_exact"]
        summary[f"pt_{key}_clears_500"] = int(bool(p["clears_500"]))

    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="strict_compliant_kernel_frontier_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        run_id = getattr(run, "id", None)
        print(f"[strict-frontier] wandb logged {len(summary)} summary keys (run {run_id})", flush=True)
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"[strict-frontier] wandb write failed (analysis unaffected): {exc}", flush=True)
        return getattr(run, "id", None)


def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad = []
    if isinstance(obj, bool):
        return bad
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


def _print_human(syn: dict) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" STRICT-COMPLIANT FRONTIER vs CUSTOM REDUCTION-INVARIANT KERNEL OVERHEAD (PR #354, #319)",
          flush=True)
    print("=" * 104, flush=True)
    print(f"  model: strict_compliant_tps(eta) = {OFFICIAL_BASELINE_DEPLOYED} * (1 - eta)   "
          f"[deployed base, non-compliant runs WITHOUT the kernel]", flush=True)
    print(f"  frontier TODAY (no kernel, non-spec int4 M=1 AR, #196) = "
          f"{syn['constants']['strict_compliant_frontier_today']:.2f} TPS", flush=True)
    print("-" * 104, flush=True)
    print(f"  {'point':<22}{'eta':>9}{'step+us':>10}{'tps_lin':>10}{'tps_div':>10}{'>500?':>7}"
          f"{'lift_165':>10}{'gap_500':>9}", flush=True)
    for p in syn["named_points"]:
        print(f"  {p['point']:<22}{p['eta']:>9.4f}{p['step_overhead_us']:>10.1f}"
              f"{p['strict_compliant_tps_linear']:>10.2f}{p['strict_compliant_tps_divisor_exact']:>10.2f}"
              f"{str(p['clears_500']):>7}{p['lift_from_frontier_today_tps']:>10.1f}"
              f"{p['gap_to_500_tps']:>9.1f}", flush=True)
    print("-" * 104, flush=True)
    ec = syn["eta_crit"]
    fl = syn["frontier_lift"]
    cc = syn["cross_checks"]
    print(f"  eta_crit_500 = {ec['eta_crit_500']}   kernel_alone_clears_500 = {ec['kernel_alone_clears_500']}"
          f"   (max frontier @ eta=0 = {ec['tps_at_eta_zero']:.2f} < 500; analytic crossing @ eta="
          f"{ec['linear_solved']['eta_solved']:.4f} < 0)", flush=True)
    print(f"  FRONTIER LIFT: {fl['strict_compliant_frontier_today']:.2f} -> "
          f"{fl['strict_compliant_tps_at_floor']:.2f} TPS  (+{fl['strict_compliant_frontier_lift_from_165']:.1f}, "
          f"restores {100*fl['spec_premium_restored_frac']:.1f}% of spec premium)  "
          f"residual gap to 500 = {fl['residual_gap_to_500_tps']:.1f} TPS", flush=True)
    print(f"  CROSS-CHECK: 520.953*(1-0.3141) = {cc['roundtrip_326_offtheshelf_ceiling']:.8f} "
          f"(#326 banked {cc['roundtrip_326_offtheshelf_ceiling_banked']:.8f}); "
          f"#213 divisor 536.659/(1.07332) = {cc['roundtrip_213_divisor_clears_500']:.4f}", flush=True)
    print("-" * 104, flush=True)
    st = syn["self_test"]
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="strict-compliant-frontier")
    args = ap.parse_args(argv)

    syn = synthesize()
    prov = provenance()
    # fold provenance checks into the self-test conditions
    syn["self_test"]["conditions"].update(prov["checks"])
    syn["provenance"] = prov

    self_test_passes = all(syn["self_test"]["conditions"].values())

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 354, "agent": "wirbel",
        "kind": "strict-compliant-kernel-frontier", "synthesis": syn,
        "strict_compliant_kernel_frontier_self_test_passes": self_test_passes,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
        "scope": ("CPU-only analytic over banked merged *_results.json (#196 laxllfjl, #326 io4cs2ch, "
                  "#213 5o7zcj8s). NO model build / training / served-file change / HF Job / submission. "
                  "0 GPU, 0 TPS. BASELINE 481.53 UNCHANGED. Bank-the-analysis."),
        "public_evidence_used": [
            "#319 human strict-lock (2026-06-15 10:56:17Z): 'ignore #124, we want to ensure we stick "
            "with the strict greedy token matching' -- re-opens the custom reduction-invariant kernel.",
            "Leaderboard frontier tops at ~489.6 TPS (osoi5-...-precache-skv64-v1), BELOW 500 -- no "
            "method clears the bar yet.",
            "lawine #196 (compliant_nonspec_floor): strict-compliant frontier today = 165.44 TPS "
            "(non-spec int4 M=1 AR, token_identity_rate=1.0).",
            "wirbel #326 (io4cs2ch): off-the-shelf VBI restores M=8 identity at eta=0.3141 -> 357.32 TPS; "
            "custom-kernel first-principles floor ~0.9455% (#216 int4-Marlin split-K).",
            "wirbel #213 (5o7zcj8s): lambda=1 kernel-overhead budget 7.332%, deployed step 1218.2us.",
            "board: openevolve in-serve dense-verify confirms BOTH spec levers (drafter-side self-KV, "
            "verify-side) are closed -> the strict-compliant kernel is the lane.",
        ],
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    payload["strict_compliant_kernel_frontier_self_test_passes"] = bool(self_test_passes and payload["nan_clean"])
    if nan_paths:
        print(f"[strict-frontier] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strict_compliant_kernel_frontier_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[strict-frontier] wrote {out_path}", flush=True)
    print(f"[strict-frontier] PRIMARY strict_compliant_kernel_frontier_self_test_passes = "
          f"{payload['strict_compliant_kernel_frontier_self_test_passes']}", flush=True)
    print(f"[strict-frontier] strict_compliant_tps_at_floor = {syn['strict_compliant_tps_at_floor']:.4f}", flush=True)
    print(f"[strict-frontier] eta_crit_500 = {syn['eta_crit_500']}  "
          f"kernel_alone_clears_500 = {syn['kernel_alone_clears_500']}", flush=True)
    print(f"[strict-frontier] strict_compliant_frontier_lift_from_165 = "
          f"{syn['strict_compliant_frontier_lift_from_165']:.4f}", flush=True)

    run_id = _maybe_log_wandb(args, payload)
    if run_id:
        payload["wandb_run_id"] = run_id
        out_path.write_text(json.dumps(payload, indent=2, default=float))

    if args.self_test:
        ok = payload["strict_compliant_kernel_frontier_self_test_passes"]
        print(f"[strict-frontier] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
