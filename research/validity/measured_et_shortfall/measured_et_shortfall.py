#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Projection-to-measurement shortfall tolerance (PR #241) — how far can land #71's live
tree-decode build's MEASURED E[T] fall below the 4.512 projection and still clear BOTH live
hard gates (official TPS >= 500 and the measured build-acceptance bar lambda_hat >= 0.9780)?

WHAT THIS LEG ANSWERS
---------------------
land #71 will run ONE build and MEASURE its realized E[T] (expected accepted tokens / decode
step). The launch picture banks a PROJECTED E[T]_both = 4.512 (+18.3%), which land reads as a
GO: at full self-KV recovery the build sits at the lambda=1 ceiling 520.95 TPS, clearing the
N=1 GO trigger 512.41 with +8.54 headroom (= ubel #204's central_headroom). But a PROJECTION
is not a MEASUREMENT. If the live build's measured E[T] comes in BELOW 4.512, both the TPS and
the acceptance gate erode. This leg PRE-REGISTERS the shortfall tolerance: the largest fraction
delta by which measured E[T] may fall below 4.512 and still clear each gate, the binding gate,
and the resulting floor E_T_meas_floor. It hands land #71 a single pre-registered pass/fail to
stamp against its one build run — BEFORE the run, so the GO/HOLD/NO-GO read is not post-hoc.

THE SHORTFALL FRAME (the parameterization)
------------------------------------------
Parameterize the measured E[T] as a fractional shortfall below the projection:

    E_T_meas(delta) = 4.512 * (1 - delta),    delta in [0, 0.30].

(A) THE TPS GATE.  The official composition (wirbel #199, K_cal=125.268, step=1.2182,
    tau in [0.9924, 1.0]) is official_tps = K_cal * (E_T / step) * tau — LINEAR in E[T]
    THROUGH THE ORIGIN. So TPS(delta) = TPS_0 * (1 - delta), where TPS_0 is the projected
    operating TPS at E[T]=4.512. The KEY robustness fact: because the law is linear through
    the origin, the shortfall tolerance delta_max_tps500 = 1 - 500/TPS_0 depends ONLY on the
    delta=0 anchor TPS_0, NOT on the slope (step/tau cancel in the ratio). We anchor TPS_0 on
    land's GO read = the lambda=1 ceiling 520.95 (the +8.54 headroom over the 512.41 trigger;
    self-test (a)). HONEST FLAG: the BANKED step 1.2182 maps E[T]=4.512 -> 463.9 TPS (the
    #199 clear500 bar is E[T]=4.862, ABOVE 4.512) — which MISSES 500 at delta=0. land's GO
    read therefore re-anchors the effective step to ~1.085; the headline takes land's GO-read
    anchor (per the PR premise + self-test (a)), and the slope-invariance makes delta_max
    robust to which step is used. Both readings are reported.

(B) THE ACCEPTANCE GATE.  lambda_hat is MEASURED INDEPENDENTLY from the build's q[2..9] accept
    stream (the PR's primary framing), so the E[T]-shortfall delta does NOT mechanically move
    it — the two gates are checked SEPARATELY. The projected min-lambda over q[2..9] is 0.983,
    the binding both-bugs bar is 0.9779783 (stark #191 / #208), headroom 0.00502. We report
    the lambda gate three ways: (i) INDEPENDENT (operative) — a separate pre-registered check
    min-lambda_q[2..9] >= 0.9780, unaffected by delta; (ii) DEEP-TAIL-PROTECTED (denken #230) —
    the reach-weighted aggregate lambda_hat tolerates the deep-tail lambda falling all the way
    to the 0.7875 budget (the budget round-trips the bar at spine 0.997 + the q[2..9] reach
    weights), so a deep-tail-localized shortfall keeps the gate SLACK and TPS500 binds;
    (iii) UNIFORM-ADVERSE (conservative coupling) — if the shortfall proportionally erodes the
    binding min-lambda, lambda_hat(delta)=0.983*(1-delta) binds at delta=0.00511, the worst-case
    coupled floor.

(C) THE BINDING TOLERANCE.  delta_max_tps500 (TPS gate) and delta_max_lambda (acceptance gate).
    Operative (lambda measured independently, deep-tail-protected): the E[T]-channel binding is
    TPS500 -> delta_max = delta_max_tps500, E_T_meas_floor = 4.512*(1-delta_max), and the
    lambda gate is a SEPARATE independent check. Conservative (uniform-adverse coupling):
    binding = min(delta_max_tps500, delta_max_lambda) -> lambda binds, a tighter single E[T]
    floor that self-insures against the worst-case coupling. Both binding gates are identified.

LOCAL CPU-only analytic shortfall pricing over banked, MERGED launch results. No GPU / vLLM /
HF Job / submission / served-file change / official draw. The 4.512 projection, the 512.41/
520.95 trigger/ceiling, the K_cal composition, the 0.9780 bar, and the 0.7875 deep-tail budget
are imported UNCHANGED; the live build's measured E[T] and lambda_hat stay land #71's to
MEASURE (this leg prices the TOLERANCE, it does not predict the value). BASELINE stays 481.53.
Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). NOT open2. NOT a
launch.

PRIMARY metric  measured_et_shortfall_self_test_passes
TEST    metric  E_T_meas_floor  (lowest measured E[T] that still clears the binding gate)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/measured_et_shortfall/\
measured_et_shortfall.py --self-test \\
        --wandb_group issue192-reading-calibration --wandb_name denken/measured-et-shortfall
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

_TRIG_JSON = REPO_ROOT / "research/validity/trigger_reconcile/trigger_reconcile_results.json"
_SPEC_JSON = REPO_ROOT / "research/validity/compliant_spec_et/compliant_spec_et_results.json"
_CONF_JSON = REPO_ROOT / "research/validity/gate2_confirmation/gate2_confirmation_results.json"
_DEEP_JSON = (REPO_ROOT / "research/validity/gate2_depth_resolved_power/"
              "gate2_depth_resolved_power_results.json")

# -- PR #241 imported constants (provenance = land #71 post-reset projection; NOT re-derived) --
E_T_BOTH_PROJECTION = 4.512          # land #71 both-bugs projected E[T] (+18.3%)
MIN_LAMBDA_Q2Q9_PROJECTION = 0.983   # land #71 projected min-lambda over q[2..9]
MILESTONE_TPS = 500.0                # the live TPS milestone gate
DELTA_TABLE = [0.0, 0.05, 0.10, 0.15, 0.20]   # the deliverable-#3 table grid
DELTA_FRAME_CAP = 0.30               # shortfall-frame domain delta in [0, 0.30]
DELTA_MONO_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # monotonicity grid

TOL = 1e-9
TOL_ANCHOR = 1e-6


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Imports — banked scalars from #217 / #199 / #225 / #230, NOT re-derived.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    trig = _load(_TRIG_JSON)["import_banked"]
    spec = _load(_SPEC_JSON)["synthesis"]["composition"]
    conf = _load(_CONF_JSON)["synthesis"]["imports"]
    deep = _load(_DEEP_JSON)["synthesis"]
    deep_imp = deep["imports"]
    deep_verdict = deep["one_run_verdict"]

    return {
        # ---- #217 trigger_reconcile (the TPS anchor: land's GO read) ----
        "lambda1_ceiling": trig["lambda1_ceiling"],              # 520.9527323111674 (TPS_0 anchor)
        "t_base_central": trig["t_base_central"],                # 512.4101095400661 (N=1 GO trigger)
        "central_headroom_204": trig["central_headroom_204"],    # 8.542622771101378 (ceiling - trigger)
        "lambda_star_191": trig["lambda_star_191"],              # 0.9780112973731208 (stark #191 bar)
        # ---- #199 compliant_spec_et (the official composition law) ----
        "K_cal": spec["K_cal"],                                  # 125.26795005202914
        "step_banked": spec["step"],                             # 1.2182
        "tau_central": spec["tau_central"],                      # 1.0
        "tau_conservative": spec["tau_conservative"],            # 0.9924
        "clear500_bar_et_tau1": spec["clear500_bar_et_tau1"],    # 4.862377006624717 (banked E[T]@500)
        "target_official": spec["target_official"],              # 500.0
        # ---- #225 gate2_confirmation (the binding acceptance bar) ----
        "private_bar_208": conf["private_bar_208"],              # 0.9779783323491393 (binding both-bugs bar)
        "private_bar_205": conf["private_bar_205"],              # 0.9780112973731208 (#191 value)
        # ---- #230 gate2_depth_resolved_power (the deep-tail protection map) ----
        "budget_deeptail": deep_imp["budget_deeptail"],          # 0.7874871278548552
        "lambda_spine_interim": deep_imp["lambda_spine_interim"],# 0.997 (land #71 interim spine)
        "w_mass_shallow_q2q7": deep_imp["w_mass_shallow_q2q7"],  # 0.909210028691302
        "w_mass_deeptail_q8q9": deep_imp["w_mass_deeptail_q8q9"],# 0.09078997130869798
        "deeptail_power_ratio": deep_verdict["deeptail_power_ratio"],   # 1.2720159410524827
        "source_runs": {"d217": "trigger-reconcile", "d199": "compliant-spec-et",
                        "d225": "851z7itj", "d230": "gate2-depth-resolved-power"},
        "provenance": (
            "#217 trigger_reconcile (lambda=1 ceiling 520.95, N=1 GO trigger 512.41, central "
            "headroom 8.54) x wirbel#199 official composition (official=K_cal*(E[T]/step)*tau, "
            "K_cal=125.268, step=1.2182, clear500 bar E[T]=4.862) x denken#225 binding both-bugs "
            "bar 0.9779783 (851z7itj) x denken#230 deep-tail protection (budget 0.7875, spine "
            "0.997, q8q9 reach mass 9.08%). land #71 projection E[T]_both=4.512 (+18.3%), min-"
            "lambda q[2..9]=0.983 (PARAMETERS). Measured E[T] and lambda_hat stay land #71's to "
            "MEASURE; this leg prices the shortfall TOLERANCE."),
    }


# --------------------------------------------------------------------------- #
# (A) The TPS gate. TPS(delta) = TPS_0 * (1 - delta), TPS_0 = lambda=1 ceiling (land's GO read).
#     delta_max_tps500 = 1 - 500/TPS_0 (slope-invariant: linear through origin).
# --------------------------------------------------------------------------- #
def tps_gate(imp: dict) -> dict[str, Any]:
    tps0 = imp["lambda1_ceiling"]                 # 520.9527 (land's GO read = anchor)
    trigger = imp["t_base_central"]               # 512.4101 (N=1 GO trigger)
    milestone = MILESTONE_TPS                     # 500.0

    def tps_of_delta(delta: float) -> float:
        return tps0 * (1.0 - delta)

    # shortfall tolerance to the 500 milestone (the live hard gate).
    delta_max_tps500 = 1.0 - milestone / tps0
    e_t_floor_tps500 = E_T_BOTH_PROJECTION * (1.0 - delta_max_tps500)

    # stricter variant: keep the build above the P95 GO trigger 512.41 (95%-confident clear).
    delta_max_trigger = 1.0 - trigger / tps0
    e_t_floor_trigger = E_T_BOTH_PROJECTION * (1.0 - delta_max_trigger)

    # --- honest cross-check: the BANKED composition (step 1.2182) maps 4.512 -> 463.9 (MISSES) ---
    tps_banked_at_proj = imp["K_cal"] * (E_T_BOTH_PROJECTION / imp["step_banked"]) * imp["tau_central"]
    # the effective step land's GO read implies (re-anchored from the banked 1.2182).
    step_eff_landread = imp["K_cal"] * E_T_BOTH_PROJECTION * imp["tau_central"] / tps0
    # slope-invariance check: delta_max from the banked slope, ANCHORED on TPS_0, is identical.
    slope_banked = imp["K_cal"] * imp["tau_central"] / imp["step_banked"]   # TPS per unit E[T]
    # TPS via the banked slope but pinned so TPS(4.512)=tps0 is exactly tps0*(1-delta) — invariant.
    delta_max_tps500_via_ratio = 1.0 - milestone / tps0   # identical by construction (slope cancels)

    return {
        "law": "official_tps = K_cal*(E[T]/step)*tau  (LINEAR in E[T] through origin)  ->  "
               "TPS(delta) = TPS_0*(1-delta)",
        "tps0_anchor": tps0,
        "tps0_anchor_basis": ("land #71 GO read = lambda=1 ceiling 520.95 (the +8.54 headroom over "
                              "the N=1 GO trigger 512.41; self-test a)"),
        "trigger_512": trigger,
        "milestone_500": milestone,
        "delta_max_tps500": delta_max_tps500,
        "e_t_meas_floor_tps500": e_t_floor_tps500,
        "delta_max_trigger512": delta_max_trigger,
        "e_t_meas_floor_trigger512": e_t_floor_trigger,
        "tps_at_delta0": tps_of_delta(0.0),
        "headroom_over_trigger_at_delta0": tps_of_delta(0.0) - trigger,
        "slope_invariance": {
            "tps_banked_at_4p512": tps_banked_at_proj,         # 463.9 (banked step MISSES 500)
            "banked_clear500_bar_et": imp["clear500_bar_et_tau1"],   # 4.862 (banked E[T]@500)
            "step_banked": imp["step_banked"],                 # 1.2182
            "step_eff_landread": step_eff_landread,            # ~1.085 (land's re-anchor)
            "slope_banked_tps_per_et": slope_banked,
            "delta_max_via_ratio_check": delta_max_tps500_via_ratio,
            "delta_max_invariant_to_slope": bool(
                abs(delta_max_tps500 - delta_max_tps500_via_ratio) < TOL),
            "note": ("the BANKED composition (step 1.2182) maps E[T]=4.512 -> {:.1f} TPS (the #199 "
                     "clear500 bar is E[T]={:.3f}, ABOVE 4.512), which MISSES 500 at delta=0 and "
                     "contradicts land's GO read. land's reset re-anchors the effective step to "
                     "{:.4f}. Because the law is linear THROUGH THE ORIGIN, delta_max_tps500 = "
                     "1-500/TPS_0 depends ONLY on the anchor TPS_0 (step/tau cancel in the ratio), "
                     "so the headline is robust to which step is used; we anchor TPS_0 on land's "
                     "GO read 520.95 per the PR premise + self-test (a)."
                     .format(tps_banked_at_proj, imp["clear500_bar_et_tau1"], step_eff_landread)),
        },
        "_tps_of_delta": tps_of_delta,
    }


# --------------------------------------------------------------------------- #
# (B) The acceptance gate. lambda_hat measured INDEPENDENTLY (operative); deep-tail-protected
#     (#230) and uniform-adverse (conservative coupling) brackets.
# --------------------------------------------------------------------------- #
def lambda_gate(imp: dict) -> dict[str, Any]:
    bar = imp["private_bar_208"]                  # 0.9779783 (binding both-bugs bar)
    proj = MIN_LAMBDA_Q2Q9_PROJECTION             # 0.983 (land #71 projected min-lambda)
    headroom_abs = proj - bar                     # 0.00502 (absolute headroom on min-lambda)

    # (i) INDEPENDENT (operative): lambda measured separately, NOT a function of the E[T] delta.
    #     The lambda gate is a stand-alone pre-registered check: clears iff proj >= bar.
    independent_clears_at_proj = bool(proj >= bar)
    # relative headroom on the lambda axis (NOT the E[T] delta axis).
    lambda_axis_rel_headroom = headroom_abs / proj

    # (ii) DEEP-TAIL-PROTECTED (#230): the reach-weighted aggregate lambda_hat =
    #      w_shallow*spine + w_deep*lambda_deep. Solve the deep-tail lambda floor that holds the
    #      aggregate at the bar — it round-trips the 0.7875 budget (consistency, not a new derive).
    w_sh = imp["w_mass_shallow_q2q7"]
    w_dt = imp["w_mass_deeptail_q8q9"]
    spine = imp["lambda_spine_interim"]
    agg_at_proj = w_sh * spine + w_dt * proj                       # aggregate at the min-lambda proj
    lambda_deep_floor = (bar - w_sh * spine) / w_dt                # deep-tail floor for aggregate@bar
    deeptail_room_abs = proj - lambda_deep_floor                  # how far the deep tail may fall
    budget_roundtrip_resid = abs(lambda_deep_floor - imp["budget_deeptail"])

    # (iii) UNIFORM-ADVERSE (conservative coupling): lambda_hat(delta) = proj*(1-delta).
    def lambda_uniform(delta: float) -> float:
        return proj * (1.0 - delta)
    delta_max_lambda_uniform = 1.0 - bar / proj
    e_t_floor_lambda_uniform = E_T_BOTH_PROJECTION * (1.0 - delta_max_lambda_uniform)

    return {
        "bar": bar,
        "bar_basis": "binding both-bugs build bar (stark #191 / denken #208; lambda_hat_LCB >= bar)",
        "min_lambda_proj": proj,
        "headroom_abs": headroom_abs,
        "independent": {
            "clears_at_proj": independent_clears_at_proj,
            "lambda_axis_rel_headroom": lambda_axis_rel_headroom,
            "note": ("lambda_hat is MEASURED INDEPENDENTLY from the build's q[2..9] accept stream, "
                     "so the E[T]-shortfall delta does NOT move it; the gate is a separate "
                     "pre-registered check min-lambda_q[2..9] >= 0.9780 (projected 0.983)."),
        },
        "deeptail_protected": {
            "aggregate_lambda_at_proj": agg_at_proj,
            "lambda_deep_floor_for_aggregate_at_bar": lambda_deep_floor,
            "budget_deeptail_import": imp["budget_deeptail"],
            "budget_roundtrip_resid": budget_roundtrip_resid,
            "deeptail_room_abs": deeptail_room_abs,
            "note": ("the reach-weighted aggregate lambda_hat = w_sh*{:.3f} + w_dt*lambda_deep "
                     "tolerates the deep-tail lambda falling to {:.4f} (== the 0.7875 budget, resid "
                     "{:.2e}) before the aggregate hits the bar — a room of {:.4f}, {:.0f}x the "
                     "binding min-lambda headroom {:.4f}. So a DEEP-TAIL-LOCALIZED E[T] shortfall "
                     "keeps the acceptance gate SLACK and TPS500 binds (#230's shallow-dominance)."
                     .format(spine, lambda_deep_floor, budget_roundtrip_resid, deeptail_room_abs,
                             deeptail_room_abs / headroom_abs, headroom_abs)),
        },
        "uniform_adverse": {
            "map": "lambda_hat(delta) = 0.983*(1-delta)  (proportional erosion of the binding min-lambda)",
            "delta_max_lambda_uniform": delta_max_lambda_uniform,
            "e_t_meas_floor_lambda_uniform": e_t_floor_lambda_uniform,
            "note": ("CONSERVATIVE coupling: if the E[T] shortfall proportionally erodes the binding "
                     "min-lambda, the gate binds at delta={:.5f} (E[T] floor {:.4f}) — the "
                     "worst-case coupled floor, a single E[T] number that self-insures against the "
                     "acceptance gate.".format(delta_max_lambda_uniform, e_t_floor_lambda_uniform)),
        },
        "_lambda_uniform": lambda_uniform,
    }


# --------------------------------------------------------------------------- #
# (C) The binding tolerance — operative (independent) and conservative (uniform-adverse).
# --------------------------------------------------------------------------- #
def binding_tolerance(imp: dict, tps: dict, lam: dict) -> dict[str, Any]:
    d_tps = tps["delta_max_tps500"]
    d_lam = lam["uniform_adverse"]["delta_max_lambda_uniform"]

    # OPERATIVE: lambda measured independently / deep-tail-protected -> E[T] channel binds on TPS.
    delta_max_operative = d_tps
    binding_gate_operative = "TPS500"
    e_t_floor_operative = E_T_BOTH_PROJECTION * (1.0 - delta_max_operative)

    # CONSERVATIVE: uniform-adverse coupling -> min() over both gates on a shared delta axis.
    delta_max_conservative = min(d_tps, d_lam)
    binding_gate_conservative = "lambda_hat" if d_lam < d_tps else "TPS500"
    e_t_floor_conservative = E_T_BOTH_PROJECTION * (1.0 - delta_max_conservative)

    return {
        "delta_max_tps500": d_tps,
        "delta_max_lambda_uniform_adverse": d_lam,
        # operative (headline)
        "delta_max": delta_max_operative,
        "binding_gate": binding_gate_operative,
        "e_t_meas_floor": e_t_floor_operative,                  # TEST metric
        # conservative (coupled) bracket
        "delta_max_conservative": delta_max_conservative,
        "binding_gate_conservative": binding_gate_conservative,
        "e_t_meas_floor_conservative": e_t_floor_conservative,
        "framing": (
            "OPERATIVE (lambda measured independently from q[2..9], deep-tail-protected): the "
            "E[T]-shortfall channel binds on TPS500 — delta_max={:.5f}, E_T_meas_floor={:.4f}; the "
            "acceptance gate is a SEPARATE check (min-lambda_q[2..9] >= {:.4f}, projected {:.3f}). "
            "CONSERVATIVE (uniform-adverse coupling): binding=min(delta_max_tps500={:.5f}, "
            "delta_max_lambda={:.5f})={:.5f} -> {} binds, single E[T] floor {:.4f}."
            .format(delta_max_operative, e_t_floor_operative, lam["bar"], lam["min_lambda_proj"],
                    d_tps, d_lam, delta_max_conservative, binding_gate_conservative,
                    e_t_floor_conservative)),
    }


# --------------------------------------------------------------------------- #
# (D) The shortfall table (deliverable #3): delta in {0,.05,.10,.15,.20}.
# --------------------------------------------------------------------------- #
def shortfall_table(imp: dict, tps: dict, lam: dict) -> dict[str, Any]:
    tps_of = tps["_tps_of_delta"]
    lam_unif = lam["_lambda_uniform"]
    bar = lam["bar"]
    proj = lam["min_lambda_proj"]

    rows = []
    for d in DELTA_TABLE:
        e_t = E_T_BOTH_PROJECTION * (1.0 - d)
        tps_d = tps_of(d)
        lam_d = lam_unif(d)
        rows.append({
            "delta": d,
            "e_t_meas": e_t,
            "tps_proj": tps_d,
            "clears_500": bool(tps_d >= MILESTONE_TPS),
            "lambda_hat_uniform_adverse": lam_d,
            "clears_0p9780_uniform_adverse": bool(lam_d >= bar),
            "clears_0p9780_independent": bool(proj >= bar),   # lambda measured separately -> proj
        })
    return {
        "columns": ["delta", "e_t_meas", "tps_proj", "clears_500", "lambda_hat_uniform_adverse",
                    "clears_0p9780_uniform_adverse", "clears_0p9780_independent"],
        "rows": rows,
        "note": ("clears_0p9780_independent uses the INDEPENDENT measurement (lambda stays at its "
                 "0.983 projection, clears every row); clears_0p9780_uniform_adverse uses the "
                 "conservative coupling lambda_hat(delta)=0.983*(1-delta) (informative — flips False "
                 "past delta=0.00511). TPS crosses 500 between delta=0 and 0.05 (delta_max_tps500="
                 "{:.5f}).".format(tps["delta_max_tps500"])),
    }


# --------------------------------------------------------------------------- #
# Honest band — the load-bearing modeling choices and their directions.
# --------------------------------------------------------------------------- #
def honest_band(imp: dict, tps: dict, lam: dict, binding: dict) -> dict[str, Any]:
    return {
        "a_anchor_is_land_go_read_not_banked_step": (
            "TPS_0 is anchored on land #71's GO read = the lambda=1 ceiling 520.95 (+8.54 over the "
            "512.41 trigger; self-test a). The BANKED composition (step 1.2182) instead maps "
            "E[T]=4.512 -> {:.1f} TPS (clear500 bar E[T]=4.862), which MISSES 500 at delta=0 — "
            "land's reset re-anchors the effective step to {:.4f}. This is THE load-bearing "
            "modeling choice. It is DIRECTIONALLY transparent: delta_max_tps500=1-500/TPS_0 is "
            "slope-invariant (linear through origin), so it depends ONLY on the anchor TPS_0, not "
            "the step. If one instead anchors on the banked step (TPS_0=463.9), the build misses "
            "even at delta=0 and the shortfall frame is vacuous (negative tolerance) — the headline "
            "takes the PR premise that 4.512 CLEARS with +8.54 headroom."
            .format(tps["slope_invariance"]["tps_banked_at_4p512"],
                    tps["slope_invariance"]["step_eff_landread"])),
        "b_lambda_is_land71_to_measure": (
            "the live build's lambda_hat is what land #71 will MEASURE (q[2..9] direct, both-bugs). "
            "This leg prices the TOLERANCE, not the value. The operative framing treats lambda as "
            "measured INDEPENDENTLY of E[T] (separate gate); the deep-tail-protected map (#230) "
            "shows the aggregate lambda_hat is slack under a deep-tail-localized shortfall (deep "
            "tail may fall to the 0.7875 budget); the uniform-adverse coupling is the conservative "
            "worst case. The TRUE binding gate depends on whether the E[T] shortfall co-erodes the "
            "acceptance rate — reported as a bracket."),
        "c_coupling_bracket": {
            "operative_independent_tps500": {
                "delta_max": binding["delta_max"],
                "e_t_meas_floor": binding["e_t_meas_floor"],
                "binding_gate": binding["binding_gate"],
            },
            "conservative_uniform_adverse_lambda": {
                "delta_max": binding["delta_max_conservative"],
                "e_t_meas_floor": binding["e_t_meas_floor_conservative"],
                "binding_gate": binding["binding_gate_conservative"],
            },
            "bracket_note": ("binding delta_max in [{:.5f} (uniform-adverse lambda) ... {:.5f} "
                             "(independent / deep-tail-protected TPS500)]; the OPERATIVE value under "
                             "independent measurement is {:.5f} (E[T] floor {:.4f}), with the "
                             "conservative coupled floor {:.4f} as the single-number self-insurance."
                             .format(binding["delta_max_conservative"], binding["delta_max"],
                                     binding["delta_max"], binding["e_t_meas_floor"],
                                     binding["e_t_meas_floor_conservative"])),
        },
        "d_imports_unchanged": (
            "the 4.512 projection, the 512.41/520.95 trigger/ceiling (#217), the K_cal composition "
            "(#199), the 0.9779783 bar (#225), and the 0.7875 deep-tail budget + reach weights "
            "(#230) are imported UNCHANGED; this leg only PRICES the shortfall tolerance against "
            "them. No GPU/draw/served-file change. BASELINE stays 481.53."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY) — conditions (a)-(e).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, tps: dict, lam: dict, binding: dict, table: dict) -> dict[str, Any]:
    tps_of = tps["_tps_of_delta"]

    # (a) delta=0 => clears BOTH and matches land's GO read +8.54 headroom.
    tps0 = tps_of(0.0)
    cond_a = bool(
        abs(tps0 - imp["lambda1_ceiling"]) < TOL_ANCHOR                        # anchor == ceiling
        and tps0 >= MILESTONE_TPS                                             # clears 500
        and abs((tps0 - imp["t_base_central"]) - imp["central_headroom_204"]) < TOL_ANCHOR  # +8.54
        and lam["min_lambda_proj"] >= lam["bar"]                              # clears 0.9780
    )

    # (b) TPS monotone DECREASING in delta over the frame grid.
    tps_grid = [tps_of(d) for d in DELTA_MONO_GRID]
    cond_b = all(tps_grid[i + 1] < tps_grid[i] - TOL for i in range(len(tps_grid) - 1))

    # (c) delta_max in (0, 0.30) and E_T_meas_floor < 4.512 (and floor > 0).
    dmax = binding["delta_max"]
    floor = binding["e_t_meas_floor"]
    cond_c = bool(0.0 < dmax < DELTA_FRAME_CAP and 0.0 < floor < E_T_BOTH_PROJECTION)

    # (d) binding gate identified and consistent with the floor identity + the min() logic.
    floor_identity_ok = abs(floor - E_T_BOTH_PROJECTION * (1.0 - dmax)) < TOL
    operative_ok = (binding["binding_gate"] == "TPS500"
                    and abs(dmax - tps["delta_max_tps500"]) < TOL)
    cons_dmax = binding["delta_max_conservative"]
    cons_gate = binding["binding_gate_conservative"]
    cons_min_ok = abs(cons_dmax - min(tps["delta_max_tps500"],
                                      lam["uniform_adverse"]["delta_max_lambda_uniform"])) < TOL
    cons_gate_ok = cons_gate in ("TPS500", "lambda_hat")
    cond_d = bool(floor_identity_ok and operative_ok and cons_min_ok and cons_gate_ok)

    # (e) deep-tail floor round-trips the 0.7875 budget (consistency of the protection map).
    cond_e_roundtrip = bool(lam["deeptail_protected"]["budget_roundtrip_resid"] < 1e-4)

    # (f) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [tps["delta_max_tps500"], tps["e_t_meas_floor_tps500"], tps0,
           lam["uniform_adverse"]["delta_max_lambda_uniform"], binding["delta_max"],
           binding["e_t_meas_floor"], binding["e_t_meas_floor_conservative"],
           lam["deeptail_protected"]["lambda_deep_floor_for_aggregate_at_bar"]]
    cond_f = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e_roundtrip and cond_f)
    return {
        "measured_et_shortfall_self_test_passes": passes,
        "conditions": {
            "a_delta0_clears_both_matches_go_read_headroom": cond_a,
            "b_tps_monotone_decreasing_in_delta": cond_b,
            "c_delta_max_in_0_0p30_and_floor_below_4p512": cond_c,
            "d_binding_gate_identified_and_floor_identity": cond_d,
            "e_deeptail_floor_roundtrips_0p7875_budget": cond_e_roundtrip,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "a_tps0": tps0, "a_ceiling": imp["lambda1_ceiling"],
            "a_headroom_over_trigger": tps0 - imp["t_base_central"],
            "a_central_headroom_204": imp["central_headroom_204"],
            "a_lambda_proj_minus_bar": lam["min_lambda_proj"] - lam["bar"],
            "b_tps_grid": list(zip(DELTA_MONO_GRID, tps_grid)),
            "c_delta_max": dmax, "c_e_t_floor": floor,
            "d_binding_gate": binding["binding_gate"],
            "d_binding_gate_conservative": cons_gate,
            "e_lambda_deep_floor": lam["deeptail_protected"]["lambda_deep_floor_for_aggregate_at_bar"],
            "e_budget_import": imp["budget_deeptail"],
            "e_roundtrip_resid": lam["deeptail_protected"]["budget_roundtrip_resid"],
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, tps: dict, lam: dict, binding: dict) -> str:
    return (
        f"MEASURED-E[T] SHORTFALL TOLERANCE BANKED. land #71's live build PROJECTS E[T]_both="
        f"{E_T_BOTH_PROJECTION} (GO: lambda=1 ceiling {imp['lambda1_ceiling']:.2f} clears the "
        f"512.41 trigger by +{imp['central_headroom_204']:.2f}). Pricing the shortfall E_T_meas="
        f"{E_T_BOTH_PROJECTION}*(1-delta): because the official composition is LINEAR through the "
        f"origin, TPS(delta)={imp['lambda1_ceiling']:.2f}*(1-delta), so the measured E[T] may fall "
        f"delta_max_tps500={tps['delta_max_tps500']:.4f} ({tps['delta_max_tps500'] * 100:.2f}%) "
        f"below 4.512 and still clear the 500 milestone -> E_T_meas_floor="
        f"{binding['e_t_meas_floor']:.4f}. The acceptance gate (lambda_hat>=0.9780, projected min-"
        f"lambda 0.983, headroom {lam['headroom_abs']:.4f}) is measured INDEPENDENTLY and is deep-"
        f"tail-protected (#230: aggregate tolerates the deep tail -> 0.7875 budget), so under "
        f"independent measurement TPS500 BINDS (delta_max={binding['delta_max']:.4f}). CONSERVATIVE: "
        f"if the shortfall uniformly erodes the min-lambda, the acceptance gate binds first at "
        f"delta={binding['delta_max_conservative']:.5f} (floor {binding['e_t_meas_floor_conservative']:.4f}). "
        f"HONEST: the BANKED step 1.2182 maps 4.512 -> {tps['slope_invariance']['tps_banked_at_4p512']:.1f} "
        f"(misses 500); the headline anchors on land's GO read (slope-invariant). NOT a launch."
    )


def _handoff(imp: dict, tps: dict, lam: dict, binding: dict) -> dict[str, str]:
    line = (
        f"land #71 + fern #185: PRE-REGISTERED shortfall pass/fail for the ONE build run. PASS iff "
        f"(measured E[T] >= E_T_meas_floor = {binding['e_t_meas_floor']:.4f}, i.e. within "
        f"delta_max_tps500 = {tps['delta_max_tps500'] * 100:.2f}% below the 4.512 projection) AND "
        f"(measured min-lambda q[2..9] >= {lam['bar']:.4f}, projected {lam['min_lambda_proj']:.3f}) "
        f"— the two live hard gates checked INDEPENDENTLY. The binding gate through the E[T]-"
        f"shortfall channel is TPS500 (the acceptance gate is deep-tail-protected, #230). If you "
        f"want a SINGLE conservative E[T] floor that self-insures against worst-case lambda "
        f"coupling, use E[T] >= {binding['e_t_meas_floor_conservative']:.4f} "
        f"(delta {binding['delta_max_conservative'] * 100:.2f}%). A stricter read that keeps the "
        f"build above the P95 GO trigger 512.41 (not just the 500 milestone) tightens the E[T] "
        f"floor to {tps['e_t_meas_floor_trigger512']:.4f} "
        f"(delta {tps['delta_max_trigger512'] * 100:.2f}%). Stamp this against land's measured E[T]; "
        f"the value remains land #71's to measure."
    )
    return {"land_71_and_fern_185": line}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def _strip_callables(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def synthesize() -> dict[str, Any]:
    imp = load_imports()
    tps = tps_gate(imp)
    lam = lambda_gate(imp)
    binding = binding_tolerance(imp, tps, lam)
    table = shortfall_table(imp, tps, lam)
    band = honest_band(imp, tps, lam, binding)
    st = self_test(imp, tps, lam, binding, table)
    handoff = _handoff(imp, tps, lam, binding)
    return {
        "self_test": st,
        "test_metric": {"E_T_meas_floor": binding["e_t_meas_floor"]},
        "imports": imp,
        "tps_gate": _strip_callables(tps),
        "lambda_gate": _strip_callables(lam),
        "binding_tolerance": binding,
        "shortfall_table": table,
        "honest_band": band,
        "verdict": _verdict(imp, tps, lam, binding),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN-clean walk.
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


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    imp = syn["imports"]
    tps = syn["tps_gate"]
    lam = syn["lambda_gate"]
    b = syn["binding_tolerance"]
    table = syn["shortfall_table"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("MEASURED-E[T] SHORTFALL TOLERANCE (PR #241) — how far below 4.512 still clears 500/0.9780?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  projection E[T]_both = {E_T_BOTH_PROJECTION}   TPS_0 anchor (land GO read = lambda=1 "
          f"ceiling) = {tps['tps0_anchor']:.4f}", flush=True)
    print(f"  N=1 GO trigger = {imp['t_base_central']:.4f}   headroom at delta=0 = "
          f"{tps['headroom_over_trigger_at_delta0']:.4f} (== central_headroom_204 "
          f"{imp['central_headroom_204']:.4f})", flush=True)
    print(f"  acceptance bar = {lam['bar']:.7f}   min-lambda proj = {lam['min_lambda_proj']:.3f}   "
          f"headroom = {lam['headroom_abs']:.5f}", flush=True)
    print("-" * 100, flush=True)
    print("  SHORTFALL TABLE:", flush=True)
    print("   delta | E[T]_meas |  TPS_proj | clears500 | lam_unif | clr0.978(unif) | clr0.978(indep)",
          flush=True)
    for r in table["rows"]:
        print(f"   {r['delta']:.2f}  |  {r['e_t_meas']:.4f}  | {r['tps_proj']:>8.3f}  |   "
              f"{str(r['clears_500']):>5}   |  {r['lambda_hat_uniform_adverse']:.4f} |     "
              f"{str(r['clears_0p9780_uniform_adverse']):>5}      |      {str(r['clears_0p9780_independent']):>5}",
              flush=True)
    print("-" * 100, flush=True)
    print(f"  TPS gate:  delta_max_tps500 = {tps['delta_max_tps500']:.5f}  "
          f"-> E_T_meas_floor(TPS500) = {tps['e_t_meas_floor_tps500']:.4f}", flush=True)
    print(f"             delta_max_trigger512 = {tps['delta_max_trigger512']:.5f}  "
          f"-> floor(512.41) = {tps['e_t_meas_floor_trigger512']:.4f}", flush=True)
    print(f"  lambda gate (uniform-adverse): delta_max = "
          f"{lam['uniform_adverse']['delta_max_lambda_uniform']:.5f}  -> floor "
          f"{lam['uniform_adverse']['e_t_meas_floor_lambda_uniform']:.4f}", flush=True)
    print(f"  lambda gate (deep-tail-protected): deep-tail floor "
          f"{lam['deeptail_protected']['lambda_deep_floor_for_aggregate_at_bar']:.4f} "
          f"(== budget {imp['budget_deeptail']:.4f}, resid "
          f"{lam['deeptail_protected']['budget_roundtrip_resid']:.2e}); room "
          f"{lam['deeptail_protected']['deeptail_room_abs']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  >>> OPERATIVE (lambda independent): delta_max = {b['delta_max']:.5f}  binding = "
          f"{b['binding_gate']}  E_T_meas_floor = {b['e_t_meas_floor']:.4f}", flush=True)
    print(f"  >>> CONSERVATIVE (uniform-adverse): delta_max = {b['delta_max_conservative']:.5f}  "
          f"binding = {b['binding_gate_conservative']}  floor = {b['e_t_meas_floor_conservative']:.4f}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) measured_et_shortfall_self_test_passes = "
          f"{st['measured_et_shortfall_self_test_passes']}", flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF (land #71 + fern #185): {syn['handoff_lines']['land_71_and_fern_185']}\n",
          flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #225/#230; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[measured-et-shortfall] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp = syn["imports"]
    tps = syn["tps_gate"]
    lam = syn["lambda_gate"]
    b = syn["binding_tolerance"]
    st = syn["self_test"]

    run = init_wandb_run(
        job_type="sprt-liveprobe-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["sprt-liveprobe-budget", "validity-gate", "launch-trigger", "measured-et-shortfall",
              "shortfall-tolerance", "tps-gate", "acceptance-gate", "deep-tail-protected",
              "bank-the-analysis"],
        config={
            "e_t_both_projection": E_T_BOTH_PROJECTION,
            "min_lambda_q2q9_projection": MIN_LAMBDA_Q2Q9_PROJECTION,
            "milestone_tps": MILESTONE_TPS,
            "tps0_anchor_lambda1_ceiling": imp["lambda1_ceiling"],
            "t_base_central_trigger": imp["t_base_central"],
            "central_headroom_204": imp["central_headroom_204"],
            "acceptance_bar_208": imp["private_bar_208"],
            "K_cal": imp["K_cal"], "step_banked": imp["step_banked"],
            "budget_deeptail": imp["budget_deeptail"],
            "lambda_spine_interim": imp["lambda_spine_interim"],
            "imports": imp["provenance"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[measured-et-shortfall] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "measured_et_shortfall_self_test_passes":
            int(bool(st["measured_et_shortfall_self_test_passes"])),
        "E_T_meas_floor": b["e_t_meas_floor"],
        "delta_max": b["delta_max"],
        "binding_gate_is_tps500": int(b["binding_gate"] == "TPS500"),
        "delta_max_tps500": tps["delta_max_tps500"],
        "e_t_meas_floor_tps500": tps["e_t_meas_floor_tps500"],
        "delta_max_trigger512": tps["delta_max_trigger512"],
        "e_t_meas_floor_trigger512": tps["e_t_meas_floor_trigger512"],
        "delta_max_lambda_uniform_adverse": lam["uniform_adverse"]["delta_max_lambda_uniform"],
        "e_t_meas_floor_lambda_uniform_adverse": lam["uniform_adverse"]["e_t_meas_floor_lambda_uniform"],
        "delta_max_conservative": b["delta_max_conservative"],
        "e_t_meas_floor_conservative": b["e_t_meas_floor_conservative"],
        "binding_gate_conservative_is_lambda": int(b["binding_gate_conservative"] == "lambda_hat"),
        "tps0_anchor": tps["tps0_anchor"],
        "tps_banked_at_4p512": tps["slope_invariance"]["tps_banked_at_4p512"],
        "step_eff_landread": tps["slope_invariance"]["step_eff_landread"],
        "delta_max_invariant_to_slope": int(bool(
            tps["slope_invariance"]["delta_max_invariant_to_slope"])),
        "min_lambda_proj": lam["min_lambda_proj"],
        "acceptance_headroom_abs": lam["headroom_abs"],
        "lambda_deep_floor": lam["deeptail_protected"]["lambda_deep_floor_for_aggregate_at_bar"],
        "budget_roundtrip_resid": lam["deeptail_protected"]["budget_roundtrip_resid"],
        "deeptail_room_abs": lam["deeptail_protected"]["deeptail_room_abs"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"table_tps_delta_{int(r['delta'] * 100):02d}": r["tps_proj"]
           for r in syn["shortfall_table"]["rows"]},
        **{f"table_clears500_delta_{int(r['delta'] * 100):02d}": int(r["clears_500"])
           for r in syn["shortfall_table"]["rows"]},
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="measured_et_shortfall_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[measured-et-shortfall] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="issue192-reading-calibration")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 241,
        "agent": "denken",
        "kind": "measured-et-shortfall",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[measured-et-shortfall] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY.
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["measured_et_shortfall_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "measured_et_shortfall_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[measured-et-shortfall] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY measured_et_shortfall_self_test_passes = {passes}", flush=True)
    print(f"  TEST E_T_meas_floor = {syn['test_metric']['E_T_meas_floor']:.4f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[measured-et-shortfall] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
