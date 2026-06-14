#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""M16-measured -> official 500-shot go/no-go GATE (PR #142).

WHAT THIS IS
------------
A single validated gate that converts land #71's MEASURED M=16 descent-walk
numbers into one official-TPS go/no-go for the approval-gated 500-shot --
replacing fern #129's projection with a measured decision. It is the one call
land's measured run feeds directly, and it produces the evidence line for the
eventual `Approval request: HF job` issue. It does NOT authorize a launch (the
actual launch still goes through the human-approved-issue gate).

THE FOUR NUMBERS land's M=16 run reports (the gate's only free inputs):
  (1) accept_length          -- MEASURED E[T] (the numerator)
  (2) per_position_branch_hit -- measured rank-2 branch-hit rho2 (land's local gate)
  (3) measured_step_time     -- measured depth-9 decode step (lawine #136)
  (4) tau                    -- local->official transfer (lawine #116/#126; central 1.0)

THE MAP (banked supply cost model -- the #100 lever_composition compose + my
#125 realization-ceiling step model; identical figure of merit to #129/#134):
  official_TPS = K_cal * accept_length / measured_step_time * tau
  K_cal = 125.268 (= frontier 481.53 / E[T]_lin 3.844; #100 compose)
  measured_step_time: lawine #136 when it lands; until then the 1.2127 depth-9
    roofline (#125 W* step, measured 1.83x attn tax priced in) -> roofline-pending.

THE VERDICT (vs the 500 bar, + a 530 stretch), on the operative E[T] knife-edges:
  >= 4.841  clears 500 at the depth-9 step (fern #129 operative bar)
  >= 5.131  clears the 530 stretch
  == 5.207  rho-optimal supply ceiling (-> ~538, fern #125)
  GREEN = clears 500 robustly (conservative tau corner also >= 500, with margin)
  AMBER = straddles 500 (tau/step knife-edge)
  RED   = below 500 even at central tau

THE PRECONDITIONS (validity sanity-checks wired in as gate gates):
  branch-hit ~ rho2 = 0.4165 (land's local topology gate)
  tok/step > 3.844           (linear-MTP floor -- HARD ABORT below; the tree adds
                              nothing over the deployed linear chain there)
  PPL <= 2.42 captured        (quality gate)
  greedy token-IDs captured   (decode-audit evidence captured)
A GO needs a GREEN tps_verdict AND all preconditions passing.

SELF-TEST (the gate is valid iff it reproduces BOTH banked anchors within +-2%):
  (a) as-built oracle        E[T]=2.621 -> ~271  (RED)   [fern #134]
  (b) both-bugs-fixed ceiling E[T]=5.207 -> ~538 (GREEN)  [fern #125]

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no kernel
build. Reuses the banked #100 compose (lever_composition K_cal) and my #125 step
model verbatim (one source of truth per constant). Serves nothing -> greedy
identity untouched by construction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- banked models reused verbatim (single source of truth per constant) ----
lc = _load("lever_composition", os.path.join(_HERE, "lever_composition.py"))

K_CAL = lc.K_CAL                         # 125.268 (= 481.53 / 3.844; #100 compose)
E_T_LINEAR = lc.E_T_LINEAR               # 3.844  -- the linear-MTP floor (HARD abort below)
E_T_TREE = lc.E_T_TREE                   # 5.207  -- rho-optimal supply ceiling
FRONTIER_OFFICIAL = lc.FRONTIER_OFFICIAL  # 481.53
TARGET_OFFICIAL = lc.TARGET_OFFICIAL     # 500.0
TARGET_530 = 530.0                       # stretch target

# tau band: lawine #116/#126 local->official transfer (tight; central pinned at 1).
# FLOOR FIX (PR #155, ubel #148 flag): this gate projects a TREE (M=32 wide verify),
# so the conservative tau corner must be the TREE-class floor (lawine #126's
# derive_tau_tree_roofline 0.9924 -- the verify-GEMM/tree-mask compute-exposed
# fraction is wider than SplitK's M=8), NOT the SplitK-class 0.9983 that was here.
# The SplitK floor was ~0.6% optimistic on the tau-low corner for a tree projection.
# 0.9924318649123313 == #148 kcal_tree_transfer_band leg-A scale_floor (one source
# of truth). Central stays pinned at 1.0, so the 271/538 anchors are unchanged.
TAU = {"low": 0.9924318649123313, "central": 1.00, "high": 1.00}

# depth-9 W* decode step: the #125 realization-ceiling roofline (measured 1.83x
# attention tax already priced). lawine #136 will MEASURE this end-to-end; until
# then this roofline is the operative step and the gate flags roofline-pending.
STEP_ROOFLINE_DEPTH9 = 1.2127483746822987

# land's local topology gate: the measured rank-2 conditional branch-hit (rho2).
RHO2_BRANCH_HIT = 0.4165047789261015     # rho_cond_measured[0] (banked, #125/wirbel)
RHO2_TOL_DEFAULT = 0.05                  # branch-hit sanity-check tolerance

PPL_GATE = 2.42                          # quality gate (reference 2.30 + 5%)

# ---- self-test anchors: the gate is only valid if it reproduces BOTH ----
ANCHOR_RED_ET = 2.621                    # as-built oracle (BUG-1 + BUG-2 live; fern #134)
ANCHOR_RED_TPS = 271.0                   # -> must return ~271, RED
ANCHOR_GREEN_ET = 5.207                  # both-bugs-fixed rho-optimal ceiling (fern #125)
ANCHOR_GREEN_TPS = 538.0                 # -> must return ~538, GREEN
ANCHOR_TOL = 0.02                        # +-2%

# #134 recovery-matrix E[T] cells (committed advisor-branch anchors) -- the gate
# must reproduce their verdicts as a cross-check against the merged matrix.
MATRIX_CELLS = [
    (2.621, "cell1 both bugs live (measured)"),
    (2.7397, "cell2 only BUG-1 fixed (depth-1->0.7287, salvage broken)"),
    (5.0564, "cell3 only BUG-2 fixed (rho-opt descent, depth-1 still 0.679)"),
    (5.2068, "cell4 BOTH bugs fixed (rho-optimal ceiling)"),
]


# ----------------------------------------------------------------------------
# Core map (the #100/#129 figure of merit). The ONLY free input is accept_length;
# step_time and tau come from the measured M=16 run (or the banked roofline).
# ----------------------------------------------------------------------------
def official_tps_map(accept_length: float, step_time: float, tau: float) -> float:
    """official_TPS = K_cal * E[T] / step_time * tau."""
    return K_CAL * accept_length / step_time * tau


def accept_length_for_official(target_official: float, step_time: float, tau: float) -> float:
    """Invert the map: the accept_length at which official crosses `target`."""
    return target_official * step_time / (K_CAL * tau)


def _tps_verdict(central: float, taulow: float,
                 target: float = TARGET_OFFICIAL, knife: float = 1.0) -> str:
    """GREEN/AMBER/RED vs `target`, using the tau band for robustness.

    GREEN = clears `target` robustly: central > target+knife AND conservative
            corner (taulow) >= target.
    AMBER = within +-knife of the bar, OR central clears but the conservative
            corner does not (a tau/step knife-edge).
    RED   = central < target-knife.
    """
    if central < target - knife:
        return "RED"
    if central <= target + knife or taulow < target:
        return "AMBER"
    return "GREEN"


# ----------------------------------------------------------------------------
# THE GATE: one call land's measured M=16 run feeds directly.
# ----------------------------------------------------------------------------
def measured_m16_to_official(accept_length: float,
                             per_position_branch_hit: float | None,
                             measured_step_time: float = STEP_ROOFLINE_DEPTH9,
                             tau: float = TAU["central"],
                             *,
                             ppl: float | None = None,
                             greedy_token_ids_captured: bool | None = None,
                             step_is_roofline: bool = True,
                             rho2_tol: float = RHO2_TOL_DEFAULT) -> dict:
    """Map land's measured M=16 numbers -> official TPS + GREEN/AMBER/RED go/no-go.

    Headline output: `official_tps_central`. The dict also carries the verdict vs
    500 (+530 stretch), the operative E[T] knife-edge bars, and the validity
    preconditions, so a single call is a complete go/no-go.

    Parameters
    ----------
    accept_length            MEASURED E[T] (the numerator) [#1]
    per_position_branch_hit  measured rank-2 branch-hit rho2 (land's gate) [#2]
    measured_step_time       measured depth-9 step (lawine #136; default roofline) [#3]
    tau                      local->official transfer (lawine #116/#126) [#4]
    ppl                      captured PPL (None = not captured)
    greedy_token_ids_captured  decode-audit evidence captured (None = not captured)
    step_is_roofline         True => the step is the #125 roofline, not lawine's
                             measured anchor -> result flagged roofline-pending.
    """
    # --- core map: central + conservative (tau-low) corner ---
    official_central = official_tps_map(accept_length, measured_step_time, tau)
    official_taulow = official_tps_map(accept_length, measured_step_time, TAU["low"])

    # --- verdict vs 500 (with the 530 stretch) ---
    verdict = _tps_verdict(official_central, official_taulow, TARGET_OFFICIAL)
    stretch_530 = _tps_verdict(official_central, official_taulow, TARGET_530)
    clears_500_central = bool(official_central >= TARGET_OFFICIAL)
    clears_500_conservative = bool(official_taulow >= TARGET_OFFICIAL)
    clears_530_central = bool(official_central >= TARGET_530)
    clears_530_conservative = bool(official_taulow >= TARGET_530)

    # --- operative E[T] knife-edge bars at THIS step ---
    bar_500 = accept_length_for_official(TARGET_OFFICIAL, measured_step_time, tau)
    bar_530 = accept_length_for_official(TARGET_530, measured_step_time, tau)
    bar_frontier = accept_length_for_official(FRONTIER_OFFICIAL, measured_step_time, tau)
    knife_edges = {
        "accept_length_to_clear_500": bar_500,
        "accept_length_to_clear_530": bar_530,
        "accept_length_to_beat_frontier_481": bar_frontier,
        "linear_floor_et": E_T_LINEAR,
        "supply_ceiling_et": E_T_TREE,
        "supply_ceiling_official": official_tps_map(E_T_TREE, measured_step_time, tau),
    }

    # --- validity preconditions (wired in as gate gates) ---
    above_floor = bool(accept_length > E_T_LINEAR)             # HARD abort below 3.844
    if per_position_branch_hit is None:
        branch_ok = None
    else:
        branch_ok = bool(abs(per_position_branch_hit - RHO2_BRANCH_HIT) <= rho2_tol)
    ppl_captured = ppl is not None
    ppl_within_gate = (bool(ppl <= PPL_GATE) if ppl_captured else None)
    greedy_ok = (bool(greedy_token_ids_captured)
                 if greedy_token_ids_captured is not None else None)

    soft = [branch_ok, ppl_within_gate, greedy_ok]
    # all soft checks must be captured AND passing; the hard floor must pass.
    all_pass = bool(above_floor and all(c is True for c in soft))
    preconditions = {
        "tok_per_step": accept_length,
        "linear_floor": E_T_LINEAR,
        "tok_per_step_above_linear_floor": above_floor,    # HARD
        "branch_hit": per_position_branch_hit,
        "branch_hit_target_rho2": RHO2_BRANCH_HIT,
        "branch_hit_tol": rho2_tol,
        "branch_hit_matches_rho2": branch_ok,
        "ppl": ppl,
        "ppl_gate": PPL_GATE,
        "ppl_captured": ppl_captured,
        "ppl_within_gate": ppl_within_gate,
        "greedy_token_ids_captured": greedy_ok,
        "all_pass": all_pass,
    }

    # --- GO/NO-GO = GREEN tps_verdict AND all preconditions pass ---
    go_no_go = "GO" if (verdict == "GREEN" and all_pass) else "NO-GO"

    return {
        "inputs": {
            "accept_length": accept_length,
            "per_position_branch_hit": per_position_branch_hit,
            "measured_step_time": measured_step_time,
            "tau": tau,
            "ppl": ppl,
            "greedy_token_ids_captured": greedy_token_ids_captured,
        },
        "official_tps_central": official_central,
        "official_tps_taulow": official_taulow,
        "official_tps_band": [official_taulow, official_central],
        "tps_verdict": verdict,
        "clears_500_central": clears_500_central,
        "clears_500_conservative": clears_500_conservative,
        "clears_530_central": clears_530_central,
        "clears_530_conservative": clears_530_conservative,
        "stretch_530_verdict": stretch_530,
        "margin_to_500": official_central - TARGET_OFFICIAL,
        "margin_to_530": official_central - TARGET_530,
        "knife_edges": knife_edges,
        "preconditions": preconditions,
        "go_no_go": go_no_go,
        "step_time": measured_step_time,
        "step_is_roofline": bool(step_is_roofline),
        "roofline_pending": bool(step_is_roofline),
        "tau_band": TAU,
    }


# ----------------------------------------------------------------------------
# Self-test: the gate is valid iff it reproduces BOTH banked anchors within +-2%.
# Anchors are defined at the #125 depth-9 roofline step (271 = fern #134 as-built,
# 538 = fern #125 both-bugs-fixed ceiling), so the self-test always runs there.
# ----------------------------------------------------------------------------
def self_test(step_time: float = STEP_ROOFLINE_DEPTH9) -> dict:
    # Both anchors fed with passing soft preconditions so the only varying input is
    # E[T]; the RED anchor is below the 3.844 floor by construction (it IS the
    # broken as-built), which the precondition layer flags.
    red = measured_m16_to_official(
        ANCHOR_RED_ET, RHO2_BRANCH_HIT, step_time, TAU["central"],
        ppl=2.39, greedy_token_ids_captured=True, step_is_roofline=True)
    grn = measured_m16_to_official(
        ANCHOR_GREEN_ET, RHO2_BRANCH_HIT, step_time, TAU["central"],
        ppl=2.39, greedy_token_ids_captured=True, step_is_roofline=True)

    def _rel(v, ref):
        return abs(v - ref) / ref

    red_tps_rel = _rel(red["official_tps_central"], ANCHOR_RED_TPS)
    grn_tps_rel = _rel(grn["official_tps_central"], ANCHOR_GREEN_TPS)
    red_tps_ok = bool(red_tps_rel <= ANCHOR_TOL)
    grn_tps_ok = bool(grn_tps_rel <= ANCHOR_TOL)
    red_verdict_ok = bool(red["tps_verdict"] == "RED")
    grn_verdict_ok = bool(grn["tps_verdict"] == "GREEN")
    passes = bool(red_tps_ok and grn_tps_ok and red_verdict_ok and grn_verdict_ok)

    return {
        "passes": passes,
        "anchor_red": {
            "accept_length": ANCHOR_RED_ET,
            "expected_tps": ANCHOR_RED_TPS,
            "gate_tps_central": red["official_tps_central"],
            "rel_err": red_tps_rel,
            "within_2pct": red_tps_ok,
            "expected_verdict": "RED",
            "gate_verdict": red["tps_verdict"],
            "verdict_ok": red_verdict_ok,
            "go_no_go": red["go_no_go"],
            "tok_per_step_above_floor": red["preconditions"]["tok_per_step_above_linear_floor"],
        },
        "anchor_green": {
            "accept_length": ANCHOR_GREEN_ET,
            "expected_tps": ANCHOR_GREEN_TPS,
            "gate_tps_central": grn["official_tps_central"],
            "rel_err": grn_tps_rel,
            "within_2pct": grn_tps_ok,
            "expected_verdict": "GREEN",
            "gate_verdict": grn["tps_verdict"],
            "verdict_ok": grn_verdict_ok,
            "go_no_go": grn["go_no_go"],
            "clears_530": grn["clears_530_conservative"],
        },
        "tolerance": ANCHOR_TOL,
        "step_time": step_time,
        "note": ("self-test runs at the #125 depth-9 roofline step; the 271 / 538 "
                 "anchors are roofline-step numbers (fern #134 / #125). A measured "
                 "step from lawine #136 re-prices the live gate but not these anchors."),
    }


def recovery_matrix_crosscheck(step_time: float, tau: float) -> list[dict]:
    """Run the gate on the #134 recovery-matrix E[T] cells -- the gate's verdicts
    must agree with the merged matrix (cell1/2 RED < 500, cell3/4 GREEN >= 500)."""
    rows = []
    for et, label in MATRIX_CELLS:
        r = measured_m16_to_official(et, RHO2_BRANCH_HIT, step_time, tau,
                                     ppl=2.39, greedy_token_ids_captured=True,
                                     step_is_roofline=True)
        rows.append({
            "accept_length": et,
            "label": label,
            "official_tps_central": r["official_tps_central"],
            "tps_verdict": r["tps_verdict"],
            "clears_500": r["clears_500_central"],
            "clears_530": r["clears_530_central"],
            "go_no_go": r["go_no_go"],
        })
    return rows


def _load_measured_input(args) -> dict | None:
    """Assemble land's measured M=16 numbers from --measured-json or explicit flags."""
    if args.measured_json and os.path.exists(args.measured_json):
        with open(args.measured_json) as f:
            m = json.load(f)
        return {
            "accept_length": float(m["accept_length"]),
            "per_position_branch_hit": m.get("per_position_branch_hit", m.get("branch_hit")),
            "ppl": m.get("ppl"),
            "greedy_token_ids_captured": m.get("greedy_token_ids_captured"),
        }
    if args.measured_et is not None:
        return {
            "accept_length": float(args.measured_et),
            "per_position_branch_hit": args.measured_branch_hit,
            "ppl": args.measured_ppl,
            "greedy_token_ids_captured": args.measured_greedy_captured,
        }
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # measured step: lawine #136 when it lands; default = #125 depth-9 roofline.
    ap.add_argument("--measured-step", type=float, default=None,
                    help="lawine #136 MEASURED depth-9 step. Omit to use the 1.2127 "
                         "roofline (result flagged roofline-pending).")
    ap.add_argument("--tau", type=float, default=TAU["central"])
    # land's measured M=16 run (optional; until it lands the gate runs self-test + bands).
    ap.add_argument("--measured-json", default=None,
                    help="path to land's measured M=16 readout {accept_length, "
                         "per_position_branch_hit, ppl, greedy_token_ids_captured}.")
    ap.add_argument("--measured-et", type=float, default=None,
                    help="land's measured E[T] (alt to --measured-json).")
    ap.add_argument("--measured-branch-hit", type=float, default=None)
    ap.add_argument("--measured-ppl", type=float, default=None)
    ap.add_argument("--measured-greedy-captured", action="store_true", default=None)
    ap.add_argument("--out", default="research/oracle_readout/m16_measured_500_gate_results.json")
    ap.add_argument("--sample-out",
                    default="research/oracle_readout/m16_measured_input_sample.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/m16-measured-500-gate")
    ap.add_argument("--wandb-group", default="m16-measured-500-gate")
    args = ap.parse_args()

    # operative step + roofline-pending flag.
    if args.measured_step is not None:
        step_time = args.measured_step
        step_is_roofline = False
    else:
        step_time = STEP_ROOFLINE_DEPTH9
        step_is_roofline = True
    tau = args.tau

    # ---- self-test (PRIMARY metric source) -- always at the roofline anchors ----
    st = self_test()
    gate_self_test_passes = int(st["passes"])

    # ---- #134 recovery-matrix cross-check at the operative step ----
    xcheck = recovery_matrix_crosscheck(step_time, tau)

    # ---- operative bands at the live step ----
    bands = {
        "clear_500_bar": accept_length_for_official(TARGET_OFFICIAL, step_time, tau),
        "clear_530_bar": accept_length_for_official(TARGET_530, step_time, tau),
        "overtake_frontier_481_bar": accept_length_for_official(FRONTIER_OFFICIAL, step_time, tau),
        "supply_ceiling_et": E_T_TREE,
        "supply_ceiling_official_central": official_tps_map(E_T_TREE, step_time, TAU["central"]),
        "supply_ceiling_official_taulow": official_tps_map(E_T_TREE, step_time, TAU["low"]),
        "linear_floor_et": E_T_LINEAR,
        "step_time": step_time,
        "step_is_roofline": step_is_roofline,
    }

    # ---- live gate (if land's measured M=16 number is provided) ----
    measured = _load_measured_input(args)
    live = None
    land_measured_pending = measured is None
    if measured is not None:
        live = measured_m16_to_official(
            measured["accept_length"], measured.get("per_position_branch_hit"),
            step_time, tau, ppl=measured.get("ppl"),
            greedy_token_ids_captured=measured.get("greedy_token_ids_captured"),
            step_is_roofline=step_is_roofline)

    # ---- test metric: gate_ready_for_measured_build = 1 when validated + waiting ----
    gate_ready_for_measured_build = int(st["passes"] and land_measured_pending)

    # ---- top-line gate state ----
    if live is not None:
        gate_verdict = live["tps_verdict"]
        gate_go = live["go_no_go"]
        gate_label = (
            f"LIVE: land M=16 E[T]={measured['accept_length']:.3f} -> official "
            f"{live['official_tps_central']:.1f} (taulow {live['official_tps_taulow']:.1f}) "
            f"-> {gate_verdict} / {gate_go}"
            + ("  [STEP ROOFLINE-PENDING -- lawine #136 not yet measured]"
               if step_is_roofline else "  [step MEASURED, lawine #136]"))
    else:
        gate_verdict = "ARMED"
        gate_go = "PENDING"
        gate_label = (
            f"GATE ARMED + VALIDATED (self-test {'PASS' if st['passes'] else 'FAIL'}); "
            f"awaiting land #71's measured M=16 number. Operative clear-500 bar "
            f"E[T] >= {bands['clear_500_bar']:.3f}, clear-530 bar E[T] >= "
            f"{bands['clear_530_bar']:.3f}, supply ceiling E[T]=5.207 -> "
            f"{bands['supply_ceiling_official_central']:.1f}."
            + ("  [STEP ROOFLINE-PENDING -- lawine #136 not yet measured]"
               if step_is_roofline else "  [step MEASURED, lawine #136]"))

    # ---- decision-input line for the eventual `Approval request: HF job` issue ----
    decision_input_line = (
        f"M16-MEASURED-500-GATE (decision input ONLY; does NOT authorize a launch): "
        f"feed land #71's measured M=16 (accept_length E[T], branch-hit rho2, lawine "
        f"#136 step, tau) -> one official-TPS go/no-go. GO requires GREEN (official "
        f">= 500 robustly) AND all preconditions (E[T] > {E_T_LINEAR} floor, branch-hit "
        f"~ {RHO2_BRANCH_HIT:.4f}, PPL <= {PPL_GATE}, greedy IDs captured). At the "
        f"{'roofline' if step_is_roofline else 'measured'} depth-9 step "
        f"{step_time:.4f}: clear-500 needs E[T] >= {bands['clear_500_bar']:.3f}, "
        f"the 5.207 ceiling -> {bands['supply_ceiling_official_central']:.0f}.")

    out = {
        "primary_metric_name": "gate_self_test_passes",
        "gate_self_test_passes": gate_self_test_passes,
        "test_metric_name": "gate_ready_for_measured_build",
        "gate_ready_for_measured_build": gate_ready_for_measured_build,
        "gate_verdict": gate_verdict,
        "gate_go_no_go": gate_go,
        "gate_label": gate_label,
        "land_measured_pending": land_measured_pending,
        "roofline_pending": step_is_roofline,
        "decision_input_line": decision_input_line,
        "map": {
            "figure_of_merit": "official_TPS = K_cal * accept_length / step_time * tau",
            "K_cal": K_CAL,
            "step_time": step_time,
            "step_is_roofline": step_is_roofline,
            "step_roofline_depth9_125": STEP_ROOFLINE_DEPTH9,
            "tau_band": TAU,
            "frontier_official": FRONTIER_OFFICIAL,
            "target_official": TARGET_OFFICIAL,
            "target_530": TARGET_530,
            "linear_floor_et": E_T_LINEAR,
            "supply_ceiling_et": E_T_TREE,
        },
        "self_test": st,
        "recovery_matrix_crosscheck": xcheck,
        "operative_bands": bands,
        "live_gate": live,
        "preconditions_spec": {
            "branch_hit_target_rho2": RHO2_BRANCH_HIT,
            "branch_hit_tol": RHO2_TOL_DEFAULT,
            "linear_floor_hard_abort": E_T_LINEAR,
            "ppl_gate": PPL_GATE,
            "greedy_token_ids_required": True,
        },
        "measured_input_schema": {
            "accept_length": "float -- land's MEASURED E[T] (numerator)",
            "per_position_branch_hit": "float -- measured rank-2 branch-hit rho2 (~0.4165)",
            "ppl": "float -- captured PPL (<= 2.42)",
            "greedy_token_ids_captured": "bool -- decode-audit IDs captured",
            "_step_note": ("measured depth-9 step comes from lawine #136 via "
                           "--measured-step, NOT this file; tau via --tau."),
        },
        "provenance": (
            "extends fern #129/#134 oracle_readout_harness (figure of merit + #125 "
            "depth-9 step model) and the #100 lever_composition compose (K_cal). "
            "Anchors: as-built E[T]=2.621->271 (fern #134); both-bugs-fixed "
            "E[T]=5.207->538 (fern #125). branch-hit rho2=0.4165 + linear floor "
            "3.844 banked. measured step pending lawine #136; tau pinned lawine "
            "#116/#126. Decisive descent lever from fern #134 matrix + wirbel #135 DP."),
        "method": ("LOCAL CPU-only analytic gate; no GPU/vLLM/HF Job/submission/"
                   "kernel build. Produces the 500-shot DECISION INPUT only; does "
                   "NOT authorize a launch. Greedy identity untouched."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # write the measured-input template for land #71 to fill.
    os.makedirs(os.path.dirname(args.sample_out), exist_ok=True)
    sample = {
        "accept_length": 5.207,
        "per_position_branch_hit": RHO2_BRANCH_HIT,
        "ppl": 2.39,
        "greedy_token_ids_captured": True,
        "_note": ("EXAMPLE (the rho-optimal ceiling). Replace with land #71's MEASURED "
                  "M=16 numbers. Pass lawine #136's measured step via --measured-step."),
    }
    with open(args.sample_out, "w") as f:
        json.dump(sample, f, indent=2)

    # ------------------------------- console -------------------------------
    print("=" * 96)
    print("M16-MEASURED -> OFFICIAL 500-SHOT GO/NO-GO GATE (PR #142)")
    print("=" * 96)
    print(f"\nmap: official = K_cal*E[T]/step*tau  (K_cal={K_CAL:.3f}, step="
          f"{step_time:.4f}{' ROOFLINE-PENDING' if step_is_roofline else ' MEASURED'}, "
          f"tau_central={tau:.4f})")

    print(f"\n[SELF-TEST] gate valid iff it reproduces BOTH anchors within +-2%:")
    ar, ag = st["anchor_red"], st["anchor_green"]
    print(f"  (a) as-built    E[T]={ar['accept_length']:.3f} -> "
          f"{ar['gate_tps_central']:.1f} (exp ~{ar['expected_tps']:.0f}, "
          f"err {ar['rel_err'] * 100:.2f}%)  verdict {ar['gate_verdict']} "
          f"(exp {ar['expected_verdict']})  -> {'OK' if ar['within_2pct'] and ar['verdict_ok'] else 'FAIL'}")
    print(f"  (b) both-fixed  E[T]={ag['accept_length']:.3f} -> "
          f"{ag['gate_tps_central']:.1f} (exp ~{ag['expected_tps']:.0f}, "
          f"err {ag['rel_err'] * 100:.2f}%)  verdict {ag['gate_verdict']} "
          f"(exp {ag['expected_verdict']})  -> {'OK' if ag['within_2pct'] and ag['verdict_ok'] else 'FAIL'}")
    print(f"  => gate_self_test_passes = {gate_self_test_passes}")

    print(f"\n[X-CHECK] #134 recovery-matrix cells at step {step_time:.4f}:")
    print(f"  {'E[T]':>7s} {'official':>9s} {'verdict':>8s} {'>=500':>6s} {'>=530':>6s}  cell")
    for r in xcheck:
        print(f"  {r['accept_length']:7.3f} {r['official_tps_central']:9.1f} "
              f"{r['tps_verdict']:>8s} {'YES' if r['clears_500'] else 'no':>6s} "
              f"{'YES' if r['clears_530'] else 'no':>6s}  {r['label']}")

    print(f"\n[BANDS] operative ({'roofline' if step_is_roofline else 'measured'} "
          f"depth-9 step {step_time:.4f}):")
    print(f"  clear-500 bar   E[T] >= {bands['clear_500_bar']:.3f}")
    print(f"  clear-530 bar   E[T] >= {bands['clear_530_bar']:.3f}")
    print(f"  supply ceiling  E[T] = 5.207 -> {bands['supply_ceiling_official_central']:.1f} "
          f"(taulow {bands['supply_ceiling_official_taulow']:.1f})")

    if live is not None:
        print(f"\n[LIVE GATE] land M=16 E[T]={measured['accept_length']:.3f} -> official "
              f"{live['official_tps_central']:.1f} (taulow {live['official_tps_taulow']:.1f})  "
              f"verdict {live['tps_verdict']}  preconds_all_pass="
              f"{live['preconditions']['all_pass']}  -> {live['go_no_go']}")
    else:
        print(f"\n[LIVE GATE] PENDING -- no land #71 measured M=16 number yet. "
              f"gate_ready_for_measured_build = {gate_ready_for_measured_build}.")

    print(f"\n[PRIMARY] gate_self_test_passes = {gate_self_test_passes}")
    print(f"[TEST]    gate_ready_for_measured_build = {gate_ready_for_measured_build}")
    print(f"\n[GATE] {gate_verdict} / {gate_go} -- {gate_label}")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.sample_out} (measured-input template for land #71)")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "m16-measured-500-gate",
                                 "method": "cpu-analytic-gate-extends-100-129-134-125",
                                 "K_cal": K_CAL, "step_time": step_time,
                                 "step_is_roofline": step_is_roofline,
                                 "step_roofline_depth9": STEP_ROOFLINE_DEPTH9,
                                 "tau_low": TAU["low"], "tau_central": TAU["central"],
                                 "frontier_official": FRONTIER_OFFICIAL,
                                 "target_official": TARGET_OFFICIAL, "target_530": TARGET_530,
                                 "linear_floor_et": E_T_LINEAR, "supply_ceiling_et": E_T_TREE,
                                 "branch_hit_rho2": RHO2_BRANCH_HIT, "ppl_gate": PPL_GATE,
                                 "land_measured_pending": land_measured_pending})
        s = wandb.summary
        # PRIMARY + TEST
        s["gate_self_test_passes"] = gate_self_test_passes
        s["gate_ready_for_measured_build"] = gate_ready_for_measured_build
        # self-test detail
        s["selftest_anchor_red_tps"] = st["anchor_red"]["gate_tps_central"]
        s["selftest_anchor_red_relerr"] = st["anchor_red"]["rel_err"]
        s["selftest_anchor_red_verdict_ok"] = int(st["anchor_red"]["verdict_ok"])
        s["selftest_anchor_green_tps"] = st["anchor_green"]["gate_tps_central"]
        s["selftest_anchor_green_relerr"] = st["anchor_green"]["rel_err"]
        s["selftest_anchor_green_verdict_ok"] = int(st["anchor_green"]["verdict_ok"])
        # bands
        s["clear_500_bar_et"] = bands["clear_500_bar"]
        s["clear_530_bar_et"] = bands["clear_530_bar"]
        s["supply_ceiling_official_central"] = bands["supply_ceiling_official_central"]
        s["roofline_pending"] = int(step_is_roofline)
        s["gate_verdict"] = gate_verdict
        s["gate_go_no_go"] = gate_go
        s["gate_label"] = gate_label
        s["decision_input_line"] = decision_input_line
        if live is not None:
            s["live_official_tps_central"] = live["official_tps_central"]
            s["live_official_tps_taulow"] = live["official_tps_taulow"]
            s["live_tps_verdict"] = live["tps_verdict"]
            s["live_go_no_go"] = live["go_no_go"]
            s["live_preconditions_all_pass"] = int(live["preconditions"]["all_pass"])
        # self-test anchor table
        at = wandb.Table(columns=["anchor", "accept_length", "gate_tps", "expected_tps",
                                  "rel_err_pct", "within_2pct", "gate_verdict",
                                  "expected_verdict", "verdict_ok"])
        for key in ("anchor_red", "anchor_green"):
            a = st[key]
            at.add_data(key, a["accept_length"], a["gate_tps_central"], a["expected_tps"],
                        a["rel_err"] * 100, int(a["within_2pct"]), a["gate_verdict"],
                        a["expected_verdict"], int(a["verdict_ok"]))
        wandb.log({"self_test_anchors": at})
        # recovery-matrix cross-check table
        mt = wandb.Table(columns=["accept_length", "official_tps", "verdict",
                                  "clears_500", "clears_530", "go_no_go", "label"])
        for r in xcheck:
            mt.add_data(r["accept_length"], r["official_tps_central"], r["tps_verdict"],
                        int(r["clears_500"]), int(r["clears_530"]), r["go_no_go"], r["label"])
        wandb.log({"recovery_matrix_crosscheck": mt})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
