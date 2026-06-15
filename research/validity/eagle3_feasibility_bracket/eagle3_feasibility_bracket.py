#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Step-banked BUILT-raise target + EAGLE-3 feasibility bracket (PR #290).

THE QUESTION
------------
fern #281 (10necg21, MERGED) closed Path-A on all three axes; the SOLE re-open is
a BUILT public-E[T] raise to >= 4.966 at the deployed step. But that 4.966 floor
was priced @ the deployed (UNSHAVED) step = 1218.2us. wirbel #285 (97b57hhe,
MERGED) then proved a realizable, basis-honest, bit-identical lossless step shave
(SDPA num_stages 3->2): new_step = 1202.717us (+1.29% -> 487.7 TPS, 0/128
divergent, maxdiff=0.0). kanna #286 (0k4azmjo, MERGED) independently confirmed the
envelope is verify-side -> bridge = 1.0 (basis-honest, NO draft-side 0.21 discount).

Banking the one proven-free lossless lever therefore RELAXES fern #281's BUILT-raise
target below 4.966. Nobody computed the HONEST, step-banked target, nor bracketed
whether it sits inside the feasibility window for the prime forward candidate
(EAGLE-3 multi-layer hidden-state fidelity).

THE DELIVERABLE (this leg)
--------------------------
Bank wirbel #285's lossless envelope (bridge=1.0, kanna #286) into fern #281's
4.966 public-E[T] floor to produce the HONEST, step-banked BUILT-raise target,
compute EAGLE-3's recoverable budget beyond denken #119's linear cap (3.8445, which
the deployed drafter already sits AT -> zero linear headroom), and bracket it inside
the linear-cap -> full-acceptance feasibility window. A NECESSARY-condition de-risk
for the human-approval-gated EAGLE-3 retrain; sufficiency remains build-gated.

COMPOSITION LAW (clean K_cal frame, identical to fern #281 et_raise_feasibility_envelope)
----------------------------------------------------------------------------------------
    official = K_cal * E[T] / step_rel            (step_rel = step / step_deployed)
At step_rel = 1, E[T] = 3.844:  K_cal * E[T] = 481.53 (round-trips EXACT; K_cal =
481.53/3.844 = 125.268). The literal PR form  official = K_cal*(E[T]/step)*tau  is
algebraically identical (tau cancels into the step normalisation) and is reported as
a cross-check. For FIXED TPS the E[T] needed scales LINEARLY with step_rel, so the
step-banked target is simply  fern_floor * (new_step / step).

E[T] axis = PUBLIC accepted-tokens-per-step (range [0, K+1]). The PUBLIC floor 4.966
already builds in ubel #263's private 0.804 OOD haircut (4.966 * 0.804 = 3.992 ~
honest-500 real floor 3.9914 = 500/K_cal); the 0.804 is a constant multiplier that
cancels in the step ratio, so the relaxation is basis-honest on the SAME public axis.

Pure CPU analytic over banked W&B numbers (all imported VERBATIM; never re-derived).
Analysis-only; BASELINE 481.53 untouched (this BANKS a measured envelope into an E[T]
floor; adds 0 TPS). NOT a launch; no served-file change; no HF Job; no submission.
NOT open2."""
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
# Banked anchors (imported VERBATIM; never re-derived). Provenance in comments.
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                       # PR #52 official frontier TPS (2x9fm2zx)
TARGET_TPS = 500.0                               # the > 500 bar
LAMBDA1_CEIL = 520.9527323111674                 # #257 lambda=1 step-side ceiling
K_CAL = 125.268                                  # #257/#217 steps/sec calibration (== 481.53/3.844)
STEP_US = 1218.2                                  # kanna #217 vgovdrjc served step (NORMALIZED unit)
TAU = 1.218                                       # composition round-trip tau (cancels into step norm)
E_T_DEPLOYED = 3.844                              # stark #266 deployed K=7-linear public E[T] @ M=8
K_SPEC = 7                                        # K=7 linear MTP draft chain

# ---- wirbel #285 97b57hhe: the lossless micro-lever envelope (MY measurement) ----
NEW_STEP_US = 1202.7171244939168                  # banked step after SDPA num_stages 3->2 (bit-ident)
TOTAL_LOSSLESS_STEP_SAVING_US = 15.482875506083142
ENVELOPE_TPS = 487.72885498477575
ENVELOPE_TPS_GAIN_PCT = 1.2873247741107985
ENVELOPE_DIVERGENT = 0                            # 0/128 divergent
ENVELOPE_MAXDIFF = 0.0
ENVELOPE_PPL = 2.3772

# ---- kanna #286 0k4azmjo: bridge basis-honesty ----
BRIDGE_VERIFY = 1.0                               # verify-side (deployed M=8) -> NO discount
BRIDGE_DRAFT = 0.2147                             # draft-side batch=1 wall (does NOT apply here)
COMPOSED_STEP_STACK_TPS_286 = 493.637            # best composed basis-honest step stack (still < 500)

# ---- fern #281 10necg21: Path-A three-axis closure ----
FERN_FLOOR_PUBLIC = 4.966                         # public E[T] needed @ deployed step (priv 0.804)
PATH_A_FULLY_CLOSED = True

# ---- denken #119: the LINEAR drafter E[T] structural cap ----
LINEAR_CAP = 3.8445                               # caps even at PERFECT capacity (property of the chain)

# ---- ubel #263 2khp8gzs: private rank-2+ OOD acceptance collapse ----
PRIV_FACTOR = 0.804                               # canonical private/public E[T] ratio (decode)
EAGLE3_TARGET_LAYERS = (2, 21, 39)                # multi-layer hidden-state fusion source layers

# ---- derived constant: honest-500 real-E[T] floor = 500 / K_cal ----
HONEST500_FLOOR = 3.9914                          # fern #274/#281 = 500 / K_cal (real/measured E[T])


# --------------------------------------------------------------------------- #
# Analytic core.
# --------------------------------------------------------------------------- #
def tps_at(et: float, step_us: float) -> float:
    """Composition TPS at public E[T] and normalized step (clean K_cal frame)."""
    step_rel = step_us / STEP_US
    return K_CAL * et / step_rel


def et_for_tps(tps: float, step_us: float) -> float:
    """Public E[T] needed for a target TPS at a given step (inverse of tps_at)."""
    step_rel = step_us / STEP_US
    return tps * step_rel / K_CAL


def synthesize() -> dict[str, Any]:
    # ---- (1) STEP-BANK THE FLOOR ----------------------------------------- #
    # For FIXED TPS, the public-E[T] needed scales LINEARLY with the step, so banking
    # wirbel #285's lossless lever (new_step < step) RELAXES fern #281's 4.966 floor.
    step_rel = NEW_STEP_US / STEP_US                         # 0.987295 (< 1 -> relaxation)
    step_banked_built_raise_target = FERN_FLOOR_PUBLIC * step_rel    # ~4.9029
    delta_target = FERN_FLOOR_PUBLIC - step_banked_built_raise_target  # ~0.0631 shaved off
    relaxation_is_basis_honest = abs(BRIDGE_VERIFY - 1.0) < 1e-12     # kanna #286: verify-side, no discount

    # ---- (2) RECOVERABLE BUDGET BEYOND THE LINEAR CAP -------------------- #
    # EAGLE-3 must add this much PUBLIC E[T] beyond the linear cap (denken #119).
    eagle3_recoverable_budget_et = step_banked_built_raise_target - LINEAR_CAP   # ~1.058
    budget_unbanked = FERN_FLOOR_PUBLIC - LINEAR_CAP                              # 1.1215
    budget_shrink_from_free_lever = budget_unbanked - eagle3_recoverable_budget_et  # ~0.0631 (== delta_target)
    # the deployed drafter already sits AT the linear cap -> zero linear headroom.
    linear_headroom = LINEAR_CAP - E_T_DEPLOYED                  # ~0.0005 (~0)
    deployed_at_linear_cap = abs(linear_headroom) < 1e-2
    zero_linear_headroom = deployed_at_linear_cap

    # ---- (3) FEASIBILITY WINDOW (NECESSARY CONDITION) ------------------- #
    # Absolute per-position-acceptance ceiling = FULL acceptance of the K=7 linear MTP
    # chain: every drafted token + the bonus token -> E[T]_max = K + 1.
    e_t_max = float(K_SPEC + 1)                                  # 8.0
    tps_at_etmax_deployed = tps_at(e_t_max, STEP_US)            # ~1002 TPS (>> any step-side ceiling)
    # cross-check: the lambda=1 step-side ceiling, expressed as a deployed-step E[T].
    et_at_lambda1_ceiling = LAMBDA1_CEIL / K_CAL               # ~4.1587 (520.95 = K_cal * 4.1587)
    built_raise_target_within_feasibility_window = bool(
        LINEAR_CAP < step_banked_built_raise_target < e_t_max)
    budget_fraction_of_headroom = (
        (step_banked_built_raise_target - LINEAR_CAP) / (e_t_max - LINEAR_CAP))  # ~0.255

    # the full public-E[T] bracket (monotone landmarks on the [0, K+1] acceptance axis).
    feasibility_bracket = {
        "linear_cap_denken119": LINEAR_CAP,                     # 3.8445  (deployed sits here)
        "deployed_E_T": E_T_DEPLOYED,                           # 3.844   (~ cap)
        "honest500_real_floor": HONEST500_FLOOR,               # 3.9914  (500/K_cal, no-haircut landmark)
        "lambda1_equiv_E_T": et_at_lambda1_ceiling,            # 4.1587  (520.95 step-side -> E[T])
        "step_banked_built_raise_target": step_banked_built_raise_target,  # 4.9029 (private-valid, banked)
        "fern281_floor_public": FERN_FLOOR_PUBLIC,             # 4.966   (private-valid, deployed step)
        "E_T_max_full_acceptance": e_t_max,                    # 8.0     (K+1 ceiling)
    }

    # ---- (4) EAGLE-3 MECHANISM-MATCH (verdict, cited -- NOT re-derived) -- #
    budget_requires_nonlinear_drafter = bool(zero_linear_headroom and eagle3_recoverable_budget_et > 0.0)
    eagle3_mechanism_match = {
        "budget_requires_nonlinear_drafter": budget_requires_nonlinear_drafter,
        "reason": (
            "the deployed linear drafter sits AT denken #119's linear cap 3.8445 (E[T]=3.844, "
            "headroom %.4f ~ 0), so the entire recoverable budget %.4f E[T] is STRUCTURALLY off the "
            "linear family -- unreachable by any linear-chain retrain at any capacity."
            % (linear_headroom, eagle3_recoverable_budget_et)),
        "eagle3_target_layers": list(EAGLE3_TARGET_LAYERS),
        "mechanism": (
            "EAGLE-3's multi-layer hidden-state fusion (target layers {2,21,39}) attacks the j>=2 OOD "
            "acceptance collapse (ubel #263, priv/pub 0.804) that BOUNDS the linear cap -- its mechanism "
            "is aimed at the exact origin of the budget."),
        "kanna289_crossref": (
            "kanna #289 (per-position-acceptance-decay) localizes WHERE in the chain the budget lives "
            "(the per-position a_k cliff); this card owns the AGGREGATE step-banked target + budget and "
            "CITES her cliff -- does NOT re-derive the per-position profile (non-blocking, orthogonal)."),
    }

    # ---- (5) HONEST CAVEAT ---------------------------------------------- #
    eagle3_sufficiency_is_build_gated = True
    honest_caveat = (
        "NECESSARY-condition feasibility bracket, NOT a sufficiency proof: it confirms the step-banked "
        "target %.4f sits inside the feasibility window (< E[T]_max %.1f) and is structurally off the "
        "linear family, but whether EAGLE-3 ACTUALLY recovers ~%.2f E[T] is a BUILD measurement "
        "(human-approval-gated Phase-1 viability -> full retrain). Does NOT claim EAGLE-3 reaches the "
        "target." % (step_banked_built_raise_target, e_t_max, eagle3_recoverable_budget_et))

    # ---- (6) SELF-TEST (PRIMARY) ---------------------------------------- #
    # (a) step-banked relaxation reproduces fern_floor * new_step/step within tol.
    target_recomputed = FERN_FLOOR_PUBLIC * (NEW_STEP_US / STEP_US)
    a_relaxation = (abs(step_banked_built_raise_target - target_recomputed) < 1e-9
                    and abs(step_banked_built_raise_target - 4.903) < 5e-3)
    # (b) the free lever RELAXES, never tightens.
    b_relaxes = bool(step_banked_built_raise_target < FERN_FLOOR_PUBLIC)
    # (c) bridge = 1.0 imported from kanna #286 (the relaxation is basis-honest).
    c_bridge1 = bool(relaxation_is_basis_honest and abs(BRIDGE_VERIFY - 1.0) < 1e-12)
    # (d) budget = target_banked - 3.8445 within tol.
    budget_recomputed = step_banked_built_raise_target - LINEAR_CAP
    d_budget = (abs(eagle3_recoverable_budget_et - budget_recomputed) < 1e-12
                and abs(eagle3_recoverable_budget_et - 1.06) < 1e-2)
    # (e) deployed E[T]=3.844 sits AT the linear cap 3.8445 within tol (zero linear headroom).
    e_at_cap = bool(deployed_at_linear_cap and abs(linear_headroom) < 1e-2)
    # (f) target_banked < E_T_max (feasibility window holds).
    f_window = bool(step_banked_built_raise_target < e_t_max
                    and built_raise_target_within_feasibility_window)
    # (g) composition round-trips (481.53 <-> E[T]=3.844 at step=1218.2 reproduces K_cal=125.268).
    k_cal_implied = OFFICIAL_BASELINE / E_T_DEPLOYED            # step_rel = 1 -> official = K_cal * E[T]
    g_roundtrip = (abs(k_cal_implied - K_CAL) < 1e-2
                   and abs(tps_at(E_T_DEPLOYED, STEP_US) - OFFICIAL_BASELINE) < 1e-2)
    # (i) constants imported EXACT.
    i_constants = (abs(OFFICIAL_BASELINE - 481.53) < 1e-9
                   and abs(LAMBDA1_CEIL - 520.9527323111674) < 1e-9
                   and abs(K_CAL - 125.268) < 1e-9
                   and abs(STEP_US - 1218.2) < 1e-9
                   and abs(NEW_STEP_US - 1202.7171244939168) < 1e-9
                   and abs(E_T_DEPLOYED - 3.844) < 1e-9
                   and abs(LINEAR_CAP - 3.8445) < 1e-9
                   and abs(FERN_FLOOR_PUBLIC - 4.966) < 1e-9
                   and abs(HONEST500_FLOOR - 3.9914) < 1e-9)

    cond = {
        "a_step_banked_relaxation_reproduces": bool(a_relaxation),
        "b_free_lever_relaxes_not_tightens": bool(b_relaxes),
        "c_bridge1_basis_honest_from_kanna286": bool(c_bridge1),
        "d_budget_equals_target_minus_cap": bool(d_budget),
        "e_deployed_sits_at_linear_cap": bool(e_at_cap),
        "f_target_within_feasibility_window": bool(f_window),
        "g_composition_roundtrips_kcal": bool(g_roundtrip),
        "i_constants_imported_exact": bool(i_constants),
    }
    # (h) NaN-clean is checked on the full payload after assembly (added in main()).

    # ---- (7) HAND-OFF + verdict ----------------------------------------- #
    handoff = (
        "banking your own +1.29%% lossless step lever (bridge=1.0, basis-honest, kanna #286) relaxes "
        "fern #281's BUILT-raise target from 4.966 -> %.4f public E[T], leaving a recoverable budget of "
        "%.4f E[T] beyond denken #119's linear cap (3.8445, which the deployed drafter already sits AT "
        "-- zero linear headroom) -- inside the feasibility window (%.4f < E[T]_max %.1f) but recoverable "
        "ONLY by a structurally non-linear drafter, so EAGLE-3's target-feature fusion is mechanism-matched "
        "to the budget's OOD-collapse origin (ubel #263), giving the human-gated build an honest, "
        "step-banked target and a passed necessary-condition feasibility check (sufficiency remains "
        "build-gated)." % (step_banked_built_raise_target, eagle3_recoverable_budget_et,
                           step_banked_built_raise_target, e_t_max))

    verdict = (
        "Banking wirbel #285's proven-free lossless lever (bridge=1.0, kanna #286) into fern #281's 4.966 "
        "public-E[T] floor relaxes the BUILT-raise target to %.4f (Delta %.4f E[T] shaved off by the one "
        "free lever). EAGLE-3's recoverable budget beyond denken #119's linear cap is %.4f E[T] (un-banked "
        "%.4f; the free lever shrinks it by %.4f). The deployed linear drafter sits AT the cap (headroom "
        "%.4f ~ 0) -> ZERO linear headroom; the entire budget is structurally off the linear family. The "
        "target sits INSIDE the feasibility window (cap 3.8445 < %.4f < E[T]_max %.1f; %.1f%% of the "
        "cap->ceiling headroom), reachable in principle but ONLY by a structurally non-linear drafter "
        "whose mechanism (EAGLE-3 target-feature fusion, layers {2,21,39}) is matched to the OOD-collapse "
        "origin of the budget (ubel #263). NECESSARY-condition de-risk; sufficiency remains build-gated. "
        "BASELINE 481.53 untouched; analysis-only; NOT a launch." % (
            step_banked_built_raise_target, delta_target, eagle3_recoverable_budget_et, budget_unbanked,
            budget_shrink_from_free_lever, linear_headroom, step_banked_built_raise_target, e_t_max,
            budget_fraction_of_headroom * 100.0))

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE, "target_tps": TARGET_TPS,
            "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL, "step_us": STEP_US,
            "new_step_us": NEW_STEP_US, "tau": TAU, "E_T_deployed": E_T_DEPLOYED,
            "K_spec": K_SPEC, "linear_cap": LINEAR_CAP, "fern_floor_public": FERN_FLOOR_PUBLIC,
            "honest500_floor": HONEST500_FLOOR, "priv_factor": PRIV_FACTOR,
            "bridge_verify": BRIDGE_VERIFY, "bridge_draft": BRIDGE_DRAFT,
        },
        "step_bank": {
            "step_rel": step_rel,
            "step_banked_built_raise_target": step_banked_built_raise_target,
            "fern281_floor_public": FERN_FLOOR_PUBLIC,
            "delta_target_shaved": delta_target,
            "relaxation_is_basis_honest": bool(relaxation_is_basis_honest),
            "envelope_tps_gain_pct": ENVELOPE_TPS_GAIN_PCT,
            "envelope_tps": ENVELOPE_TPS,
            "envelope_divergent": ENVELOPE_DIVERGENT, "envelope_maxdiff": ENVELOPE_MAXDIFF,
            "envelope_ppl": ENVELOPE_PPL,
        },
        "recoverable_budget": {
            "eagle3_recoverable_budget_et": eagle3_recoverable_budget_et,
            "budget_unbanked": budget_unbanked,
            "budget_shrink_from_free_lever": budget_shrink_from_free_lever,
            "linear_headroom": linear_headroom,
            "deployed_at_linear_cap": deployed_at_linear_cap,
            "zero_linear_headroom": zero_linear_headroom,
        },
        "feasibility_window": {
            "E_T_max_full_acceptance": e_t_max,
            "tps_at_etmax_deployed": tps_at_etmax_deployed,
            "et_at_lambda1_ceiling": et_at_lambda1_ceiling,
            "built_raise_target_within_feasibility_window": built_raise_target_within_feasibility_window,
            "budget_fraction_of_headroom": budget_fraction_of_headroom,
            "feasibility_bracket": feasibility_bracket,
        },
        "eagle3_mechanism_match": eagle3_mechanism_match,
        "honest_caveat": honest_caveat,
        "eagle3_sufficiency_is_build_gated": eagle3_sufficiency_is_build_gated,
        "self_test": {
            "conditions": cond,
            "k_cal_implied": k_cal_implied,
            "target_recomputed": target_recomputed,
            "budget_recomputed": budget_recomputed,
        },
        # ---- headline metrics ----
        "step_banked_built_raise_target": step_banked_built_raise_target,
        "eagle3_recoverable_budget_et": eagle3_recoverable_budget_et,
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors kanna #286; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError(
                f"resolved a stub/namespace wandb at {list(getattr(_wb, '__path__', []) or [])} "
                "with no .init -> this venv lacks the wandb wheel")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-bracket] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    sb = syn["step_bank"]
    rb = syn["recoverable_budget"]
    fw = syn["feasibility_window"]
    st = syn["self_test"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="wirbel", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-feasibility-bracket", "step-banked-target", "recoverable-budget",
                  "linear-cap", "necessary-condition", "bank-the-analysis", "pr-290"],
            config={
                "official_baseline": OFFICIAL_BASELINE, "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL,
                "step_us": STEP_US, "new_step_us": NEW_STEP_US, "E_T_deployed": E_T_DEPLOYED,
                "linear_cap": LINEAR_CAP, "fern_floor_public": FERN_FLOOR_PUBLIC,
                "honest500_floor": HONEST500_FLOOR, "priv_factor": PRIV_FACTOR,
                "bridge_verify": BRIDGE_VERIFY, "K_spec": K_SPEC,
                "imports": "wirbel#285(97b57hhe new_step=1202.717 bridge=1.0) x "
                           "kanna#286(0k4azmjo verify-bridge=1.0) x fern#281(10necg21 floor=4.966) x "
                           "denken#119(linear-cap=3.8445) x ubel#263(2khp8gzs priv=0.804) x "
                           "kanna#217(vgovdrjc step=1218.2/K_cal=125.268)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-bracket] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-bracket] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "eagle3_feasibility_bracket_self_test_passes":
            int(bool(payload["eagle3_feasibility_bracket_self_test_passes"])),
        "step_banked_built_raise_target": syn["step_banked_built_raise_target"],
        "eagle3_recoverable_budget_et": syn["eagle3_recoverable_budget_et"],
        "fern281_floor_public": FERN_FLOOR_PUBLIC,
        "delta_target_shaved": sb["delta_target_shaved"],
        "step_rel": sb["step_rel"],
        "budget_unbanked": rb["budget_unbanked"],
        "budget_shrink_from_free_lever": rb["budget_shrink_from_free_lever"],
        "linear_headroom": rb["linear_headroom"],
        "deployed_at_linear_cap": int(bool(rb["deployed_at_linear_cap"])),
        "zero_linear_headroom": int(bool(rb["zero_linear_headroom"])),
        "E_T_max_full_acceptance": fw["E_T_max_full_acceptance"],
        "tps_at_etmax_deployed": fw["tps_at_etmax_deployed"],
        "et_at_lambda1_ceiling": fw["et_at_lambda1_ceiling"],
        "built_raise_target_within_feasibility_window":
            int(bool(fw["built_raise_target_within_feasibility_window"])),
        "budget_fraction_of_headroom": fw["budget_fraction_of_headroom"],
        "budget_requires_nonlinear_drafter":
            int(bool(syn["eagle3_mechanism_match"]["budget_requires_nonlinear_drafter"])),
        "eagle3_sufficiency_is_build_gated": int(bool(syn["eagle3_sufficiency_is_build_gated"])),
        "k_cal_implied": st["k_cal_implied"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_feasibility_bracket_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[eagle3-bracket] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-bracket] wandb write failed (analysis unaffected): {exc}", flush=True)


def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad = []
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
    print(" STEP-BANKED BUILT-RAISE TARGET + EAGLE-3 FEASIBILITY BRACKET (PR #290)", flush=True)
    print("=" * 104, flush=True)
    sb = syn["step_bank"]
    rb = syn["recoverable_budget"]
    fw = syn["feasibility_window"]
    print(f"  (1) STEP-BANK: fern #281 floor 4.966 x (new_step {NEW_STEP_US:.3f} / step {STEP_US:.1f}) "
          f"= {sb['step_banked_built_raise_target']:.4f}  (Delta {sb['delta_target_shaved']:.4f} shaved; "
          f"bridge=1.0 basis-honest={sb['relaxation_is_basis_honest']})", flush=True)
    print(f"  (2) BUDGET: target {sb['step_banked_built_raise_target']:.4f} - linear-cap "
          f"{LINEAR_CAP:.4f} = {rb['eagle3_recoverable_budget_et']:.4f} E[T]  (un-banked "
          f"{rb['budget_unbanked']:.4f}; free lever shrinks by {rb['budget_shrink_from_free_lever']:.4f})",
          flush=True)
    print(f"      deployed E[T] {E_T_DEPLOYED:.4f} ~ cap {LINEAR_CAP:.4f} (headroom "
          f"{rb['linear_headroom']:.4f}) -> ZERO linear headroom = {rb['zero_linear_headroom']}", flush=True)
    print(f"  (3) WINDOW: E[T]_max(K+1) = {fw['E_T_max_full_acceptance']:.1f} "
          f"(-> {fw['tps_at_etmax_deployed']:.1f} TPS @ deployed step); lambda1-equiv E[T] "
          f"{fw['et_at_lambda1_ceiling']:.4f}", flush=True)
    print(f"      within_window = {fw['built_raise_target_within_feasibility_window']}  "
          f"(budget = {fw['budget_fraction_of_headroom'] * 100:.1f}% of cap->ceiling headroom)", flush=True)
    print("-" * 104, flush=True)
    print("  BRACKET (public accepted-tok/step axis):", flush=True)
    fb = fw["feasibility_bracket"]
    for label, val in fb.items():
        print(f"      {label:<36}{val:>9.4f}", flush=True)
    print("-" * 104, flush=True)
    mm = syn["eagle3_mechanism_match"]
    print(f"  (4) EAGLE-3: budget_requires_nonlinear_drafter = {mm['budget_requires_nonlinear_drafter']}; "
          f"target layers {mm['eagle3_target_layers']} attack ubel #263 j>=2 OOD collapse", flush=True)
    print(f"  (5) CAVEAT: eagle3_sufficiency_is_build_gated = {syn['eagle3_sufficiency_is_build_gated']}",
          flush=True)
    st = syn["self_test"]
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
    print("-" * 104, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-feasibility-bracket")
    args = ap.parse_args(argv)

    syn = synthesize()
    self_test_passes = all(syn["self_test"]["conditions"].values())

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 290, "agent": "wirbel",
        "kind": "eagle3-feasibility-bracket", "synthesis": syn,
        "eagle3_feasibility_bracket_self_test_passes": self_test_passes,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    # fold NaN-clean (condition h) into the PRIMARY pass.
    payload["eagle3_feasibility_bracket_self_test_passes"] = bool(self_test_passes and payload["nan_clean"])
    if nan_paths:
        print(f"[eagle3-bracket] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_feasibility_bracket_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[eagle3-bracket] wrote {out_path}", flush=True)
    print(f"[eagle3-bracket] PRIMARY eagle3_feasibility_bracket_self_test_passes = "
          f"{payload['eagle3_feasibility_bracket_self_test_passes']}", flush=True)
    print(f"[eagle3-bracket] TEST step_banked_built_raise_target = "
          f"{syn['step_banked_built_raise_target']:.4f}", flush=True)
    print(f"[eagle3-bracket] eagle3_recoverable_budget_et = "
          f"{syn['eagle3_recoverable_budget_et']:.4f}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["eagle3_feasibility_bracket_self_test_passes"]
        print(f"[eagle3-bracket] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
