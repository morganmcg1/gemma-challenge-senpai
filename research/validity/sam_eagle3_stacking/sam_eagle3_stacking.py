#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #296 -- SAM x EAGLE-3 companion stacking: the true residual under a better drafter.

WHAT THIS ANSWERS
-----------------
lawine #292 (``3sqnkveo``, MERGED, MINE) closed SAM-Decoding as a +2-4% UNGATED
COMPANION and left a residual ``0.902`` E[T] for the gated EAGLE-3 retrain -- BUT
that residual assumed SAM's lift stacks CLEANLY (additively) on top of EAGLE-3.
The decisive #292 finding refutes clean additivity at the source: SAM retrieval
feeds on the SAME repetition substrate the deployed linear K=7 drafter already
exploits (``pearson(E[T], hit_rate)=+0.32576``; retrieval concentrates on the
already-FAST prompts -- high-E[T] decile hit 0.170 vs low-E[T] decile 0.007).

Extend the logic: EAGLE-3 is a STRUCTURALLY BETTER drafter (kanna #289 target:
lift j>=2 conditional acceptance to ~0.91, keep a_1>=0.73). A better drafter
accepts MORE of the recurrent spans SAM would retrieve -> under EAGLE-3, SAM's
retrieved candidates are even MORE redundant, and its +2-4% companion lift
SHRINKS. If so, the "residual 0.902" is OPTIMISTIC -- the true residual EAGLE-3
must cover ALONE is LARGER.

This leg PRICES the NON-ADDITIVITY of the SAM x EAGLE-3 stack: it models the
overlap fraction ``r_overlap`` (the share of SAM's hit-spans a better drafter
already covers) from the kanna #289 a_k lift + the #292 +0.326 correlation, then
solves the honest stacked E[T] and the TRUE residual EAGLE-3 must cover alone --
the number the human-gated EAGLE-3 decision actually depends on.

HOW (CPU analytic over banked constants + REUSE of the #292 recurrence join)
---------------------------------------------------------------------------
NO GPU, NO model forward, NO served change, NO HF Job, NO submission, NOT a build,
NOT open2. BASELINE stays 481.53; this leg adds 0 TPS. The #292 recurrence join
(prompt-side suffix-recurrence + per-prompt hit-rate join, 128/128) is REUSED by
LOADING the banked #292 report -- it is NOT re-measured. Both the SAM drafter swap
and the EAGLE-3 retrain stay human-approval-gated.

The overlap model
-----------------
``r_overlap`` = the fraction of SAM's retrieved hit-spans that an EAGLE-3 drafter
already accepts (redundant -> SAM contributes nothing there). Derived from two
banked signals, exactly as PR #296 step 1 prescribes:
  (1) the a_k LIFT (kanna #289): EAGLE-3 raises mean conditional acceptance from
      abar_lin to abar_e3, reducing the "drafter misses" events SAM was filling by
      ``r_accept = 1 - (1-abar_e3)/(1-abar_lin)`` (independent-position baseline).
  (2) the +0.326 E[T]~hit-rate correlation (#292): SAM hits CONCENTRATE where the
      drafter does well (high acceptance), so the true overlap exceeds the
      independent-position baseline. The correlation closes fraction ``rho`` of the
      remaining gap to full overlap: ``r_overlap = r_accept + (1-r_accept)*rho``.
``r_overlap`` is a MODELED point estimate inside the honest bracket [0,1]; the
residual is reported across the full bracket (r_overlap=0 reproduces #292's
additive 0.902; r_overlap=1 = SAM fully redundant).

Stacked E[T] and the true residual
----------------------------------
SAM's SURVIVING fractional lift under EAGLE-3: ``s = L_sam * (1 - r_overlap)``.
  - ``E[T]_stack = E[T]_eagle3 * (1 + s)``         (PR step 2: shrunk companion)
  - ``sam_marginal_under_eagle3_pct = 100 * s``    (TEST metric)
  - ``E[T]_sam_surviving = E_T_linear * (1 + s)``  (SAM's surviving lifted E[T] on
    the deployed drafter; the #292 residual basis, now shrunk by the overlap)
  - ``true_residual_for_eagle3 = target - E[T]_sam_surviving`` -- the E[T] budget
    EAGLE-3 must raise ALONE beyond SAM's shrunk companion. At r_overlap=0 this
    reproduces #292's 0.902; sub-additivity (r_overlap>0) can only RAISE it.
We ALSO report the operational ``eagle3_et_alone_for_target = target/(1+s)`` (how
good EAGLE-3's standalone drafter must actually be) for both the 4.9029
(step-banked, wirbel #290) and 6.1245 (step-overhead-corrected, wirbel #293)
targets.

Reproduce (analytic, fast; run under .venv/bin/python -- the server venv's wandb
import is shadowed by the local ./wandb dir):
  cd target/ && .venv/bin/python \\
    research/validity/sam_eagle3_stacking/sam_eagle3_stacking.py \\
    --self-test --wandb_group sam-eagle3-stacking \\
    --wandb_name lawine/sam-eagle3-stacking
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# research/validity/sam_eagle3_stacking/this.py -> repo root is 3 up.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Imported fleet anchors -- DO NOT re-derive (PR #296 "Import"). Edit => the
# self-test constant guard (check g) FAILS. All banked, W&B-verified.
# --------------------------------------------------------------------------- #
OFFICIAL_TPS = 481.53            # #52 official linear TPS (this leg adds 0)
K_CAL = 125.268                  # kanna #269: official = K_cal * E[T]
STEP_US = 1218.2                 # kanna #217 per-forward-pass time (us)
E_T = 3.844                      # denken #119 deployed linear served E[T]
E_T_LINEAR_CAP = 3.8445          # denken #119 linear-drafter E[T] cap (AT cap)
K_SPEC = 7                       # num_speculative_tokens (manifest)
E_T_MAX = float(K_SPEC + 1)      # 8.0 -- theoretical E[T] ceiling (K drafts + 1 verify)
HONEST_500_ET_FLOOR = 3.9914     # fern #281 honest E[T] floor for 500

# kanna #289 (fi34s269): per-position conditional acceptance a_k for the deployed
# linear K=7 drafter (EXACT banked array). Cliff at position 1; conditional
# acceptance INCREASES with depth. EAGLE-3 target = lift j>=2 to ~0.91, keep a_1.
AK_LINEAR = (
    0.7292532942898975,
    0.759556697719242,
    0.7929794882639035,
    0.8228,
    0.8348727920920435,
    0.8357919254658385,
    0.8464932652113331,
)
EAGLE3_AK_DEEP = 0.91            # kanna #289 EAGLE-3 target conditional accept j>=2
EAGLE3_A1_FLOOR = 0.73           # "keep a_1 >= 0.73" (deployed a_1 ~0.7293 ~ floor)

# wirbel #290 (ub3kpsso) step-banked target; wirbel #293 (abhoog1x)
# step-overhead-corrected target (the fusion-cost-tightened bracket). CITE both.
TARGET_STEP_BANKED = 4.9029                     # wirbel #290
TARGET_STEP_OVERHEAD_CORRECTED = 6.124544578534836  # wirbel #293

# lawine #292 (3sqnkveo, MINE, MERGED): SAM standalone-on-linear bracket + the
# recurrence join. Expected-value guards for the REUSED #292 report.
SAM_LIFT_PCT_LOWER = 2.0         # SAM standalone lift on the linear drafter, low
SAM_LIFT_PCT_UPPER = 4.0         # SAM standalone lift on the linear drafter, high
SAM_LIFT_PCT_BESTCASE = 6.0      # SAM[E2] MAT best case (retrieval-applicable)
RESIDUAL_292_OPTIMISTIC = 0.9022400000000004    # #292 additive residual (target 4.90)
PEARSON_ET_HITRATE_292 = 0.3257637378624572     # #292 pearson(E[T], hit_rate)
PROMPT_HIT_RATE_292 = 0.1612388250319285        # #292 n=3 token-weighted hit-rate
LIFTED_ET_UPPER_292 = 3.99776                   # #292 SAM standalone lifted E[T] upper
NUM_PROMPTS_EXPECTED = 128

# REUSE the #292 recurrence join (do NOT re-measure) by loading the banked report.
REPORT_292 = (
    ROOT / "research" / "validity" / "sam_decoding_acceptance_lift"
    / "sam_decoding_acceptance_lift_report.json"
)

# survives-materially threshold: SAM survives as a companion if it keeps at least
# this fraction of its standalone lift after EAGLE-3 absorbs the overlap.
SURVIVES_MIN_FRACTION = 0.25

OUT_DIR = ROOT / "research" / "validity" / "sam_eagle3_stacking"


# ========================================================================== #
# Reuse the #292 recurrence join (LOAD, do NOT re-measure)
# ========================================================================== #
def load_292_join() -> dict[str, Any]:
    """Load the banked #292 report and extract the reused recurrence-join numbers.

    Reusing (not re-measuring) the prompt-side suffix-recurrence + per-prompt
    hit-rate join is exactly what PR #296 prescribes. The expected-value guards
    below ensure the loaded join is the EXACT banked #292 result (a corrupted or
    re-measured report would fail the self-test constant guard, check g).
    """
    if not REPORT_292.exists():
        raise FileNotFoundError(
            f"missing #292 report (cannot reuse the recurrence join): {REPORT_292}"
        )
    r = json.loads(REPORT_292.read_text())
    lt = r.get("low_tail_coincidence", {})
    join = {
        "num_prompts": int(r.get("num_prompts", -1)),
        "pearson_et_vs_hitrate": float(lt.get("pearson_et_vs_hitrate", float("nan"))),
        "spearman_et_vs_hitrate": float(lt.get("spearman_et_vs_hitrate", float("nan"))),
        "low_et_decile_mean_hit_rate": float(lt.get("low_et_decile_mean_hit_rate", float("nan"))),
        "high_et_decile_mean_hit_rate": float(lt.get("high_et_decile_mean_hit_rate", float("nan"))),
        "prompt_recurrence_hit_rate": float(r.get("prompt_recurrence_hit_rate", float("nan"))),
        "lifted_et_upper": float(r.get("et_lift_bracket", {}).get("lifted_et_upper", float("nan"))),
        "residual_292": float(r.get("residual_et_budget_for_eagle3", float("nan"))),
        "sam_lift_pct_lower": float(r.get("sam_et_lift_pct_lower", float("nan"))),
        "sam_lift_pct_upper": float(r.get("sam_et_lift_pct_upper", float("nan"))),
    }
    return join


def join_matches_banked(join: dict[str, Any]) -> bool:
    """True iff the loaded #292 join equals the banked expected values (reuse-exact)."""
    return bool(
        join["num_prompts"] == NUM_PROMPTS_EXPECTED
        and abs(join["pearson_et_vs_hitrate"] - PEARSON_ET_HITRATE_292) <= 1e-9
        and abs(join["prompt_recurrence_hit_rate"] - PROMPT_HIT_RATE_292) <= 1e-9
        and abs(join["lifted_et_upper"] - LIFTED_ET_UPPER_292) <= 1e-5
        and abs(join["residual_292"] - RESIDUAL_292_OPTIMISTIC) <= 1e-6
        and join["sam_lift_pct_lower"] == SAM_LIFT_PCT_LOWER
        and join["sam_lift_pct_upper"] == SAM_LIFT_PCT_UPPER
    )


# ========================================================================== #
# Acceptance profiles + survival-model E[T] cross-check
# ========================================================================== #
def eagle3_ak() -> tuple[float, ...]:
    """EAGLE-3 target per-position conditional acceptance (kanna #289).

    Keep a_1 at the deployed cliff value (~0.7293, satisfies the >=0.73 floor);
    lift every deeper position j>=2 to ~0.91. This is the structurally-better
    drafter whose extra acceptance absorbs SAM's retrieval substrate.
    """
    a1 = AK_LINEAR[0]  # deployed cliff ~0.7293 ~ the >=0.73 floor
    return (a1,) + tuple(EAGLE3_AK_DEEP for _ in range(K_SPEC - 1))


def survival_et(ak: tuple[float, ...]) -> float:
    """E[T] = 1 + sum_k prod_{j<=k} a_j (greedy accept-run length + verify token)."""
    total = 0.0
    cum = 1.0
    for a in ak:
        cum *= a
        total += cum
    return 1.0 + total


# ========================================================================== #
# Overlap fraction r_overlap (PR step 1)
# ========================================================================== #
def overlap_model(ak_lin: tuple[float, ...], ak_e3: tuple[float, ...],
                  pearson: float) -> dict[str, Any]:
    """Model r_overlap from the a_k lift + the #292 +0.326 correlation.

    r_accept (independent-position baseline): EAGLE-3 raises mean conditional
    acceptance, reducing the drafter-miss events SAM fills by
    ``1 - (1-abar_e3)/(1-abar_lin)``. The +0.326 correlation says SAM hits
    concentrate where the drafter already accepts, so the true overlap EXCEEDS the
    independent baseline: the correlation closes fraction ``rho`` of the remaining
    gap to full overlap. Robustness anchor: a survival "room-fill" estimate
    ``(E[T]_e3 - E[T]_lin)/(E_T_max - E[T]_lin)`` is reported for context.
    """
    abar_lin = statistics.fmean(ak_lin)
    abar_e3 = statistics.fmean(ak_e3)
    # (1) a_k-lift miss-reduction (independent-position baseline).
    r_accept = 1.0 - (1.0 - abar_e3) / (1.0 - abar_lin)
    # (2) correlation tilt: SAM hits concentrate on high-acceptance regions.
    rho = max(0.0, pearson)  # only a POSITIVE correlation raises overlap
    r_tilt = (1.0 - r_accept) * rho
    r_overlap = r_accept + r_tilt
    r_overlap = min(1.0, max(0.0, r_overlap))
    # robustness: survival room-fill (EAGLE-3's E[T] gain as a fraction of the
    # head-room SAM had on the linear drafter, capped at the K+1 ceiling).
    et_lin = survival_et(ak_lin)
    et_e3 = survival_et(ak_e3)
    room_fill = (et_e3 - et_lin) / (E_T_MAX - et_lin) if E_T_MAX > et_lin else 0.0
    return {
        "abar_lin": abar_lin,
        "abar_e3": abar_e3,
        "r_accept_independent": r_accept,
        "pearson_used": rho,
        "r_tilt_correlation": r_tilt,
        "r_overlap_point": r_overlap,
        "r_overlap_bracket": [0.0, 1.0],
        "room_fill_robustness": room_fill,
        "eagle3_et_from_ak_survival": et_e3,
        "linear_et_from_ak_survival": et_lin,
    }


# ========================================================================== #
# Stacked E[T], surviving SAM marginal, true residual (PR steps 2-4)
# ========================================================================== #
def surviving_lift_frac(sam_lift_pct: float, r_overlap: float) -> float:
    """s = L_sam * (1 - r_overlap): SAM's SURVIVING fractional lift under EAGLE-3."""
    return (sam_lift_pct / 100.0) * (1.0 - r_overlap)


def stacked_et(et_eagle3: float, sam_lift_pct: float, r_overlap: float) -> float:
    """PR step 2: E[T]_stack = E[T]_eagle3 * (1 + L_sam*(1-r_overlap))."""
    return et_eagle3 * (1.0 + surviving_lift_frac(sam_lift_pct, r_overlap))


def et_sam_surviving(sam_lift_pct: float, r_overlap: float) -> float:
    """SAM's surviving lifted E[T] on the deployed drafter (the #292 residual basis)."""
    return E_T * (1.0 + surviving_lift_frac(sam_lift_pct, r_overlap))


def true_residual(target: float, sam_lift_pct: float, r_overlap: float) -> float:
    """target - E[T]_sam_surviving: the E[T] budget EAGLE-3 must raise ALONE.

    At r_overlap=0 this reproduces #292's additive residual (target - SAM-upper);
    sub-additivity (r_overlap>0) shrinks E[T]_sam_surviving and can only RAISE it.
    """
    return target - et_sam_surviving(sam_lift_pct, r_overlap)


def eagle3_alone_bar(target: float, sam_lift_pct: float, r_overlap: float) -> dict[str, float]:
    """Operational: how good must EAGLE-3's STANDALONE drafter be?

    Solve E[T]_eagle3 * (1+s) = target -> eagle3_et_alone_for_target = target/(1+s).
    """
    s = surviving_lift_frac(sam_lift_pct, r_overlap)
    et_alone = target / (1.0 + s)
    return {
        "surviving_lift_frac": s,
        "eagle3_et_alone_for_target": et_alone,
        "eagle3_raise_needed_from_deployed": et_alone - E_T,
    }


def residual_band(target: float, sam_lift_pct: float, r_overlap_point: float) -> dict[str, Any]:
    """Honest bracket (PR step 4): r_overlap=0 -> #292 additive; r_overlap=1 -> full raise."""
    res_additive = true_residual(target, sam_lift_pct, 0.0)        # r_overlap=0
    res_full_redundant = true_residual(target, sam_lift_pct, 1.0)  # r_overlap=1
    res_point = true_residual(target, sam_lift_pct, r_overlap_point)
    return {
        "target": target,
        "sam_lift_pct": sam_lift_pct,
        "residual_additive_r0": res_additive,          # ~= #292's 0.902 (for 4.9029)
        "residual_full_redundant_r1": res_full_redundant,  # full raise (SAM = nothing)
        "residual_point": res_point,
        "band": [res_additive, res_full_redundant],
        "band_ordered": bool(res_additive <= res_point <= res_full_redundant),
    }


# ========================================================================== #
# Stacking analysis (assembles the headline numbers)
# ========================================================================== #
def build_stacking(join: dict[str, Any]) -> dict[str, Any]:
    ak_lin = AK_LINEAR
    ak_e3 = eagle3_ak()
    pearson = join["pearson_et_vs_hitrate"]  # REUSED from #292 (do not re-derive)
    ov = overlap_model(ak_lin, ak_e3, pearson)
    r_overlap = ov["r_overlap_point"]

    # SAM's surviving companion lift (the shrunk %), lower & upper.
    sam_marginal_lower = SAM_LIFT_PCT_LOWER * (1.0 - r_overlap)
    sam_marginal_upper = SAM_LIFT_PCT_UPPER * (1.0 - r_overlap)
    surviving_fraction = 1.0 - r_overlap

    # PR step 2: stacked E[T] on top of an EAGLE-3 that reaches its survival E[T].
    et_e3 = ov["eagle3_et_from_ak_survival"]
    et_stack_lower = stacked_et(et_e3, SAM_LIFT_PCT_LOWER, r_overlap)
    et_stack_upper = stacked_et(et_e3, SAM_LIFT_PCT_UPPER, r_overlap)

    # PR steps 3-4: true residual + honest bracket, vs BOTH targets.
    band_banked = residual_band(TARGET_STEP_BANKED, SAM_LIFT_PCT_UPPER, r_overlap)
    band_banked_lower = residual_band(TARGET_STEP_BANKED, SAM_LIFT_PCT_LOWER, r_overlap)
    band_corrected = residual_band(TARGET_STEP_OVERHEAD_CORRECTED, SAM_LIFT_PCT_UPPER, r_overlap)

    # HEADLINE true_residual_for_eagle3: L_sam upper, point r_overlap, vs 4.9029 --
    # directly comparable to #292's 0.902 (same SAM-optimistic upper convention).
    true_residual_for_eagle3 = band_banked["residual_point"]

    # operational EAGLE-3-alone bar (multiplicative companion-on-EAGLE-3 reading).
    e3bar_banked = eagle3_alone_bar(TARGET_STEP_BANKED, SAM_LIFT_PCT_UPPER, r_overlap)
    e3bar_corrected = eagle3_alone_bar(TARGET_STEP_OVERHEAD_CORRECTED, SAM_LIFT_PCT_UPPER, r_overlap)

    # does EAGLE-3 alone (its survival E[T]) clear each target without SAM?
    eagle3_alone_clears_banked = bool(et_e3 >= TARGET_STEP_BANKED)
    eagle3_alone_clears_corrected = bool(et_e3 >= TARGET_STEP_OVERHEAD_CORRECTED)

    # survives verdict: keeps a material fraction of its standalone lift.
    sam_companion_survives = bool(
        surviving_fraction >= SURVIVES_MIN_FRACTION and sam_marginal_lower > 0.0
    )

    residual_rise_vs_292 = true_residual_for_eagle3 - RESIDUAL_292_OPTIMISTIC

    return {
        "overlap": ov,
        "r_overlap": r_overlap,
        "sam_marginal_under_eagle3_pct_lower": sam_marginal_lower,
        "sam_marginal_under_eagle3_pct_upper": sam_marginal_upper,
        "sam_marginal_under_eagle3_pct": sam_marginal_upper,  # TEST headline (upper)
        "sam_standalone_lift_pct_bracket": [SAM_LIFT_PCT_LOWER, SAM_LIFT_PCT_UPPER],
        "surviving_fraction_of_standalone": surviving_fraction,
        "eagle3_et_survival": et_e3,
        "et_stack_upper_on_eagle3": et_stack_upper,
        "et_stack_lower_on_eagle3": et_stack_lower,
        "sam_adds_on_eagle3_et_upper": et_stack_upper - et_e3,
        "true_residual_for_eagle3": true_residual_for_eagle3,
        "true_residual_for_eagle3_lower_lsam": band_banked_lower["residual_point"],
        "true_residual_for_eagle3_corrected_target": band_corrected["residual_point"],
        "residual_rise_vs_292_optimistic": residual_rise_vs_292,
        "residual_292_optimistic": RESIDUAL_292_OPTIMISTIC,
        "residual_band_banked_target": band_banked,
        "residual_band_corrected_target": band_corrected,
        "eagle3_alone_bar_banked": e3bar_banked,
        "eagle3_alone_bar_corrected": e3bar_corrected,
        "eagle3_alone_clears_banked_target": eagle3_alone_clears_banked,
        "eagle3_alone_clears_corrected_target": eagle3_alone_clears_corrected,
        "sam_companion_survives_eagle3": sam_companion_survives,
        "targets": {
            "step_banked_290": TARGET_STEP_BANKED,
            "step_overhead_corrected_293": TARGET_STEP_OVERHEAD_CORRECTED,
        },
    }


# ========================================================================== #
# Self-test (PRIMARY)
# ========================================================================== #
def build_self_test(stack: dict[str, Any], join: dict[str, Any]) -> dict[str, Any]:
    st: dict[str, Any] = {}
    r_overlap = stack["r_overlap"]
    band = stack["residual_band_banked_target"]

    # (a) r_overlap in [0,1].
    st["a_r_overlap_in_unit_interval"] = bool(
        0.0 <= r_overlap <= 1.0 and math.isfinite(r_overlap)
    )

    # (b) sam_marginal_under_eagle3_pct in [0, L_sam] (sub-additivity SHRINKS only).
    ml = stack["sam_marginal_under_eagle3_pct_lower"]
    mu = stack["sam_marginal_under_eagle3_pct_upper"]
    st["b_marginal_in_shrink_band"] = bool(
        0.0 <= ml <= SAM_LIFT_PCT_LOWER + 1e-12
        and 0.0 <= mu <= SAM_LIFT_PCT_UPPER + 1e-12
        and math.isfinite(ml) and math.isfinite(mu)
    )

    # (c) true_residual_for_eagle3 >= #292's optimistic 0.902 (sub-additivity can
    #     only RAISE the residual vs the additive optimism).
    st["c_residual_ge_292_optimistic"] = bool(
        stack["true_residual_for_eagle3"] >= RESIDUAL_292_OPTIMISTIC - 1e-9
    )

    # (d) residual band ordered: additive(r0) <= point <= full_redundant(r1).
    st["d_residual_band_ordered"] = bool(band["band_ordered"])

    # (e) lifted/stacked E[T] never exceeds the K+bonus ceiling (E_T_MAX = 8).
    max_stack = stacked_et(TARGET_STEP_OVERHEAD_CORRECTED, SAM_LIFT_PCT_UPPER, 0.0)
    st["e_stacked_et_below_ceiling"] = bool(
        stack["et_stack_upper_on_eagle3"] <= E_T_MAX
        and stack["eagle3_et_survival"] <= E_T_MAX
        and max_stack <= E_T_MAX
        and et_sam_surviving(SAM_LIFT_PCT_UPPER, 0.0) <= E_T_MAX
    )

    # (f) NaN-clean over every headline float + the reused 128-prompt join.
    floats: list[float] = [
        r_overlap,
        stack["overlap"]["r_accept_independent"],
        stack["overlap"]["r_tilt_correlation"],
        stack["overlap"]["abar_lin"], stack["overlap"]["abar_e3"],
        stack["eagle3_et_survival"],
        stack["sam_marginal_under_eagle3_pct_lower"],
        stack["sam_marginal_under_eagle3_pct_upper"],
        stack["et_stack_lower_on_eagle3"], stack["et_stack_upper_on_eagle3"],
        stack["true_residual_for_eagle3"],
        stack["true_residual_for_eagle3_lower_lsam"],
        stack["true_residual_for_eagle3_corrected_target"],
        band["residual_additive_r0"], band["residual_full_redundant_r1"],
        band["residual_point"],
        join["pearson_et_vs_hitrate"], join["prompt_recurrence_hit_rate"],
        join["lifted_et_upper"], join["residual_292"],
    ]
    st["n_prompts_joined"] = join["num_prompts"]
    st["n_nonfinite"] = sum(1 for x in floats if not (isinstance(x, float) and math.isfinite(x)))
    st["f_nan_clean"] = bool(
        join["num_prompts"] == NUM_PROMPTS_EXPECTED and st["n_nonfinite"] == 0
    )

    # (g) constants imported EXACT + the #292 join reused EXACT.
    st["g_constants_imported_exact"] = bool(
        OFFICIAL_TPS == 481.53
        and K_CAL == 125.268
        and abs(STEP_US - 1218.2) <= 1e-9
        and E_T == 3.844
        and K_SPEC == 7
        and abs(E_T_MAX - 8.0) <= 1e-9
        and AK_LINEAR[0] == 0.7292532942898975
        and AK_LINEAR[-1] == 0.8464932652113331
        and abs(EAGLE3_AK_DEEP - 0.91) <= 1e-12
        and abs(TARGET_STEP_BANKED - 4.9029) <= 1e-12
        and abs(TARGET_STEP_OVERHEAD_CORRECTED - 6.124544578534836) <= 1e-12
        and abs(K_CAL * E_T - OFFICIAL_TPS) < 1e-2          # composition round-trips
        and join_matches_banked(join)                       # #292 join reused EXACT
    )

    st["passes"] = bool(
        st["a_r_overlap_in_unit_interval"]
        and st["b_marginal_in_shrink_band"]
        and st["c_residual_ge_292_optimistic"]
        and st["d_residual_band_ordered"]
        and st["e_stacked_et_below_ceiling"]
        and st["f_nan_clean"]
        and st["g_constants_imported_exact"]
    )
    return st


# ========================================================================== #
# Handoff
# ========================================================================== #
def handoff_sentence(stack: dict[str, Any]) -> str:
    return (
        f"under a better EAGLE-3 drafter the SAM companion lift shrinks from "
        f"[{SAM_LIFT_PCT_LOWER:.0f},{SAM_LIFT_PCT_UPPER:.0f}]% to "
        f"[{stack['sam_marginal_under_eagle3_pct_lower']:.2f},"
        f"{stack['sam_marginal_under_eagle3_pct_upper']:.2f}]% "
        f"(overlap fraction {stack['r_overlap']:.3f}; SAM keeps only "
        f"{100.0 * stack['surviving_fraction_of_standalone']:.0f}% of its standalone "
        f"lift), raising the true residual EAGLE-3 must cover ALONE from #292's "
        f"optimistic {RESIDUAL_292_OPTIMISTIC:.3f} to "
        f"{stack['true_residual_for_eagle3']:.3f} E[T] (vs the 4.9029 step-banked "
        f"target; {stack['true_residual_for_eagle3_corrected_target']:.3f} vs the "
        f"6.1245 step-overhead-corrected target) -- so the human-gated retrain "
        f"{'can still' if stack['sam_companion_survives_eagle3'] else 'cannot'} lean "
        f"on SAM as a (weakened) free companion, but only for ~"
        f"{100.0 * stack['surviving_fraction_of_standalone']:.0f}% of its standalone "
        f"value, and the honest EAGLE-3-alone bar is "
        f"{stack['eagle3_alone_bar_banked']['eagle3_et_alone_for_target']:.3f} E[T] "
        f"(a +{stack['eagle3_alone_bar_banked']['eagle3_raise_needed_from_deployed']:.3f} "
        f"raise from the deployed 3.844)."
    )


def build_report(join: dict[str, Any]) -> dict[str, Any]:
    stack = build_stacking(join)
    st = build_self_test(stack, join)
    return {
        "pr": 296,
        "leg": "SAM x EAGLE-3 companion stacking: the true residual under a better drafter",
        "sam_eagle3_stacking_analysis_only": True,
        "tps_delta": 0.0,
        "baseline_official_tps": OFFICIAL_TPS,
        "reused_292_join": join,
        "imported": {
            "official_tps": OFFICIAL_TPS, "K_cal": K_CAL, "E_T": E_T,
            "e_t_linear_cap": E_T_LINEAR_CAP, "step_us": STEP_US,
            "K_spec": K_SPEC, "E_T_max": E_T_MAX,
            "honest_500_et_floor": HONEST_500_ET_FLOOR,
            "ak_linear_289": list(AK_LINEAR),
            "eagle3_ak_target": list(eagle3_ak()),
            "eagle3_ak_deep": EAGLE3_AK_DEEP, "eagle3_a1_floor": EAGLE3_A1_FLOOR,
            "target_step_banked_290": TARGET_STEP_BANKED,
            "target_step_overhead_corrected_293": TARGET_STEP_OVERHEAD_CORRECTED,
            "sam_lift_pct": [SAM_LIFT_PCT_LOWER, SAM_LIFT_PCT_UPPER],
            "sam_lift_pct_bestcase": SAM_LIFT_PCT_BESTCASE,
            "residual_292_optimistic": RESIDUAL_292_OPTIMISTIC,
            "pearson_et_hitrate_292": PEARSON_ET_HITRATE_292,
        },
        "stacking": stack,
        "self_test": st,
        "handoff": handoff_sentence(stack),
        # headline metrics
        "sam_eagle3_stacking_self_test_passes": st["passes"],
        "r_overlap": stack["r_overlap"],
        "sam_marginal_under_eagle3_pct": stack["sam_marginal_under_eagle3_pct"],
        "sam_marginal_under_eagle3_pct_lower": stack["sam_marginal_under_eagle3_pct_lower"],
        "true_residual_for_eagle3": stack["true_residual_for_eagle3"],
        "true_residual_for_eagle3_corrected_target": stack["true_residual_for_eagle3_corrected_target"],
        "residual_rise_vs_292_optimistic": stack["residual_rise_vs_292_optimistic"],
        "sam_companion_survives_eagle3": stack["sam_companion_survives_eagle3"],
        "eagle3_alone_clears_banked_target": stack["eagle3_alone_clears_banked_target"],
        "eagle3_alone_clears_corrected_target": stack["eagle3_alone_clears_corrected_target"],
    }


# ========================================================================== #
# W&B + CLI
# ========================================================================== #
def _flat_summary(report: dict[str, Any]) -> dict[str, Any]:
    s = report["stacking"]
    ov = s["overlap"]
    bb = s["residual_band_banked_target"]
    bc = s["residual_band_corrected_target"]
    return {
        "sam_eagle3_stacking_self_test_passes": report["self_test"]["passes"],
        "r_overlap": s["r_overlap"],
        "r_accept_independent": ov["r_accept_independent"],
        "r_tilt_correlation": ov["r_tilt_correlation"],
        "room_fill_robustness": ov["room_fill_robustness"],
        "pearson_et_hitrate_reused": ov["pearson_used"],
        "abar_lin": ov["abar_lin"],
        "abar_e3": ov["abar_e3"],
        "eagle3_et_survival": s["eagle3_et_survival"],
        "linear_et_survival": ov["linear_et_from_ak_survival"],
        "sam_marginal_under_eagle3_pct": s["sam_marginal_under_eagle3_pct"],
        "sam_marginal_under_eagle3_pct_lower": s["sam_marginal_under_eagle3_pct_lower"],
        "sam_marginal_under_eagle3_pct_upper": s["sam_marginal_under_eagle3_pct_upper"],
        "surviving_fraction_of_standalone": s["surviving_fraction_of_standalone"],
        "sam_standalone_lift_pct_lower": SAM_LIFT_PCT_LOWER,
        "sam_standalone_lift_pct_upper": SAM_LIFT_PCT_UPPER,
        "et_stack_lower_on_eagle3": s["et_stack_lower_on_eagle3"],
        "et_stack_upper_on_eagle3": s["et_stack_upper_on_eagle3"],
        "sam_adds_on_eagle3_et_upper": s["sam_adds_on_eagle3_et_upper"],
        "true_residual_for_eagle3": s["true_residual_for_eagle3"],
        "true_residual_for_eagle3_lower_lsam": s["true_residual_for_eagle3_lower_lsam"],
        "true_residual_for_eagle3_corrected_target": s["true_residual_for_eagle3_corrected_target"],
        "residual_292_optimistic": RESIDUAL_292_OPTIMISTIC,
        "residual_rise_vs_292_optimistic": s["residual_rise_vs_292_optimistic"],
        "residual_band_banked_r0": bb["residual_additive_r0"],
        "residual_band_banked_r1": bb["residual_full_redundant_r1"],
        "residual_band_banked_point": bb["residual_point"],
        "residual_band_corrected_r0": bc["residual_additive_r0"],
        "residual_band_corrected_r1": bc["residual_full_redundant_r1"],
        "residual_band_corrected_point": bc["residual_point"],
        "eagle3_et_alone_for_target_banked": s["eagle3_alone_bar_banked"]["eagle3_et_alone_for_target"],
        "eagle3_raise_needed_banked": s["eagle3_alone_bar_banked"]["eagle3_raise_needed_from_deployed"],
        "eagle3_et_alone_for_target_corrected": s["eagle3_alone_bar_corrected"]["eagle3_et_alone_for_target"],
        "eagle3_raise_needed_corrected": s["eagle3_alone_bar_corrected"]["eagle3_raise_needed_from_deployed"],
        "eagle3_alone_clears_banked_target": s["eagle3_alone_clears_banked_target"],
        "eagle3_alone_clears_corrected_target": s["eagle3_alone_clears_corrected_target"],
        "sam_companion_survives_eagle3": s["sam_companion_survives_eagle3"],
        "target_step_banked_290": TARGET_STEP_BANKED,
        "target_step_overhead_corrected_293": TARGET_STEP_OVERHEAD_CORRECTED,
        "baseline_official_tps": OFFICIAL_TPS,
        "tps_delta": 0.0,
    }


def _maybe_log_wandb(args, report: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # logging must never break the analysis
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="sam-eagle3-stacking",
        agent="senpai",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=[t for t in [args.wandb_group] if t],
        notes="PR #296 SAM x EAGLE-3 stacking: non-additivity overlap r_overlap -> shrunk companion lift + true residual EAGLE-3 must cover alone.",
        config={
            "pr": 296,
            "analysis_only": True,
            "reuses_292_join": True,
            "num_prompts": report["reused_292_join"]["num_prompts"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[wandb] no run (no creds / disabled)", flush=True)
        return
    log_summary(run, _flat_summary(report), step=0)
    run_id = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[wandb] logged run '{args.wandb_name}' (id={run_id}) group '{args.wandb_group}'", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="PRIMARY: run the analytic core + self-test; exit non-zero unless it passes")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    join = load_292_join()
    report = build_report(join)
    report["elapsed_s"] = time.time() - t0

    report_path = out_dir / "sam_eagle3_stacking_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    s = report["stacking"]
    ov = s["overlap"]
    print(json.dumps(report["self_test"], indent=2, sort_keys=True), flush=True)
    print("\nHEADLINE:", flush=True)
    print(f"  abar_lin / abar_e3                 = {ov['abar_lin']:.5f} / {ov['abar_e3']:.5f}", flush=True)
    print(f"  EAGLE-3 E[T] (survival, a_k)       = {s['eagle3_et_survival']:.4f}  "
          f"(clears 4.9029={s['eagle3_alone_clears_banked_target']}, "
          f"6.1245={s['eagle3_alone_clears_corrected_target']})", flush=True)
    print(f"  r_accept (a_k lift) + tilt (rho)   = {ov['r_accept_independent']:.4f} + "
          f"{ov['r_tilt_correlation']:.4f}  (room-fill robustness {ov['room_fill_robustness']:.4f})", flush=True)
    print(f"  r_overlap (point)                  = {s['r_overlap']:.4f}", flush=True)
    print(f"  SAM marginal under EAGLE-3 [lo,hi] = [{s['sam_marginal_under_eagle3_pct_lower']:.3f}, "
          f"{s['sam_marginal_under_eagle3_pct_upper']:.3f}]%  "
          f"(standalone [{SAM_LIFT_PCT_LOWER:.0f},{SAM_LIFT_PCT_UPPER:.0f}]%; "
          f"keeps {100.0 * s['surviving_fraction_of_standalone']:.0f}%)", flush=True)
    print(f"  true_residual_for_eagle3 (4.9029)  = {s['true_residual_for_eagle3']:.4f}  "
          f"(#292 optimistic {RESIDUAL_292_OPTIMISTIC:.4f}; rise +{s['residual_rise_vs_292_optimistic']:.4f})", flush=True)
    print(f"  true_residual_for_eagle3 (6.1245)  = {s['true_residual_for_eagle3_corrected_target']:.4f}", flush=True)
    bb = s["residual_band_banked_target"]
    print(f"  residual band (4.9029, L_sam=4%)   = [{bb['residual_additive_r0']:.4f}, "
          f"{bb['residual_full_redundant_r1']:.4f}]  point {bb['residual_point']:.4f}", flush=True)
    print(f"  EAGLE-3-alone bar (4.9029)         = {s['eagle3_alone_bar_banked']['eagle3_et_alone_for_target']:.4f}  "
          f"(+{s['eagle3_alone_bar_banked']['eagle3_raise_needed_from_deployed']:.4f} from 3.844)", flush=True)
    print(f"  sam_companion_survives_eagle3      = {s['sam_companion_survives_eagle3']}", flush=True)
    print(f"  self_test_passes                   = {report['self_test']['passes']}", flush=True)
    print(f"\n{report['handoff']}", flush=True)
    print(f"\n[report] -> {report_path}", flush=True)

    _maybe_log_wandb(args, report)

    if args.self_test and not report["self_test"]["passes"]:
        print("[self-test] FAIL", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
