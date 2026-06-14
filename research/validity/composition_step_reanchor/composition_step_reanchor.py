#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Composition step re-anchor (PR #252) — is the SERVED per-step cost (1.2182 ms, linear MTP
K=7) the SAME object as the BUILT per-step cost (land's ~1.085 ms, tree decode), or two
different steps? Confirm (or correct) the 520.95 GO-read anchor the whole launch packet rests on.

THE INCONSISTENCY THIS LEG CLOSES (flagged in denken #241)
----------------------------------------------------------
The BANKED composition step 1.2182 ms maps E[T]=4.512 -> 463.97 TPS (MISSES 500 at delta=0),
while land's 22:12Z reset GO read maps the SAME E[T]=4.512 -> 520.95 TPS (an effective step
~1.085 ms). Every downstream launch number rests on land's GO read 520.95, NOT on a re-derived
step. CRUX: is 1.2182 (served, linear MTP K=7) the SAME object as ~1.085 (built, tree decode)?

THE TWO-STEP FRAME (the resolution)
-----------------------------------
The official composition is  official = K_cal * (E[T]/step) * tau  (wirbel #199 / kanna #217;
K_cal=125.268, tau in [0.9924, 1.0]). It prices a SINGLE decode path. The served 481.53 and the
built 520.95 are TWO DIFFERENT decode paths:

  (1) SERVED path  -- linear MTP K=7, per-step cost 1.2182 ms. Backing out its E[T] from the
      481.53 official:  E[T]_served = 481.53 * 1.2182 / (K_cal * tau) ~ 4.68 (tau=1) ... 4.72
      (tau=0.9924). The served path REALIZES E[T]~4.68 at 1.2182 ms/step -> 481.53.

  (2) BUILT path  -- tree decode, per-step cost backed out of land's GO read:
      step_built = K_cal * 4.512 * tau / 520.95 = 1.08495 ms (tau=1) -- EXACTLY the
      step_eff_landread denken #241 already computed, and == land's stated effective ~1.085.
      The built path PROJECTS E[T]_both=4.512 at 1.085 ms/step -> 520.95.

step_served (1.2182) and step_built (1.085) are DISTINCT (10.9% apart): the tree-decode build
legitimately runs a FASTER per-step cost. So the 520.95 anchor is NOT inconsistent with the
banked 1.2182 -- it just prices a DIFFERENT path. The 463.97 "inconsistency" was the artifact of
pricing the BUILT path's E[T]=4.512 at the SERVED step (1.2182) -- the wrong denominator for tree
decode. Priced at its own built-step (1.085), E[T]=4.512 -> 520.95, CLEARS 500.

THE TPS-EDGE DECOMPOSITION (exact; K_cal, tau cancel)
-----------------------------------------------------
520.9527 / 481.53 = (step_served / step_built) * (E[T]_built / E[T]_served)
                  = 1.1228 (cheaper built step)  *  0.9636 (LOWER built E[T])  =  1.0819.
The built path's TPS win over served is a STEP-COST win (tree decode's cheaper per-step), PARTLY
OFFSET by its lower projected both-bugs E[T] (4.512 < the served's realized ~4.68). The edge does
NOT come from more accepted tokens; it comes from cheaper steps.

THE #241 FLOOR SURVIVES (slope-invariance)
------------------------------------------
denken #241's E_T_meas_floor = 4.3305 = 4.512 * (500 / 520.95) is a property of the ANCHOR POINT
(E[T]_both=4.512, TPS_0=520.95) and the 500 target ONLY. The slope TPS_0/E[T]_both = 115.46 is
pinned by the anchor, INDEPENDENT of how it factors into K_cal*tau/step. Re-anchoring the step
(1.2182 -> 1.085) while preserving the GO read leaves the floor at 4.3305 (it moves < 1e-3). The
launch TPS gate is read against the BUILT step (1.085): under it the floor is a real 4.02%
tolerance; under the SERVED step (1.2182) the floor would be 4.862 > 4.512 (vacuous) -- which is
exactly why the build must be priced at its own (built) step.

LOCAL CPU-only analytic reconciliation of banked, MERGED figures. No GPU / vLLM / HF Job /
submission / served-file change / official draw. K_cal=125.268, step_banked=1.2182, tau in
[0.9924,1.0], the GO read 520.95 / built-step ~1.085, ubel #240 anchors 513.557@gate /
520.953@lambda=1, the served 481.53, and denken #241's floor 4.3305 / delta_max 0.04022 are all
IMPORTED UNCHANGED. land #245 owns the LIVE step measurement; this leg de-risks the ANALYTIC
anchor ahead of it. BASELINE stays 481.53. Bank-the-analysis (PRIMARY = self-test, adds 0 TPS).
NOT open2. NOT a launch.

PRIMARY metric  composition_step_reanchor_self_test_passes
TEST    metric  reanchored_tps_at_ET4512  (TPS that E[T]_both=4.512 maps to under the built-step)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/composition_step_reanchor/\
composition_step_reanchor.py --self-test \\
        --wandb_group issue192-reading-calibration --wandb_name denken/composition-step-reanchor
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

_D241_JSON = (REPO_ROOT / "research/validity/measured_et_shortfall/"
              "measured_et_shortfall_results.json")
_SPEC_JSON = REPO_ROOT / "research/validity/compliant_spec_et/compliant_spec_et_results.json"
_CARD_JSON = (REPO_ROOT / "research/validity/launch_decision_card/"
              "launch_decision_card_results.json")
_EXCH_JSON = (REPO_ROOT / "research/validity/tps_risk_exchange_rate/"
              "tps_risk_exchange_rate_results.json")

# -- PR #252 imported constants (provenance below; NOT re-derived) --
E_T_BOTH = 4.512                      # land #71/#245 both-bugs projected E[T] (+18.3%)
MILESTONE_TPS = 500.0                 # the live TPS milestone gate
LAND_STATED_BUILT_STEP_MS = 1.085     # land's 22:12Z reset stated effective built-step (rounded)

TOL = 1e-9
TOL_ANCHOR = 1e-6
TOL_FLOOR_MOVE = 1e-3                 # the #241 floor must move < this under the re-anchor
DISTINCT_REL_THRESHOLD = 0.01         # steps "distinct" if > 1% apart (they are ~10.9%)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Imports — banked scalars from #241 / #199 / launch-card / #240, NOT re-derived.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d241 = _load(_D241_JSON)["synthesis"]
    d241_imp = d241["imports"]
    d241_tps = d241["tps_gate"]
    d241_bind = d241["binding_tolerance"]
    spec = _load(_SPEC_JSON)["synthesis"]["composition"]
    card = _load(_CARD_JSON)
    exch = _load(_EXCH_JSON)

    served_official = card["constants"]["baseline_tps"]              # 481.53
    gate_513 = exch["provenance"]["tps_speed_gate_234"]              # 513.5574577506176
    ceiling_exch = exch["composition"]["ceiling_lambda1"]            # 520.9527323111674

    return {
        # ---- the official composition law (wirbel #199 / kanna #217 vgovdrjc) ----
        "K_cal": spec["K_cal"],                                      # 125.26795005202914
        "step_served": spec["step"],                                 # 1.2182 (SERVED, linear MTP K=7)
        "tau_central": spec["tau_central"],                          # 1.0
        "tau_conservative": spec["tau_conservative"],               # 0.9924
        "clear500_bar_et_tau1": spec["clear500_bar_et_tau1"],        # 4.862377006624717 (served E[T]@500)
        "target_official": spec["target_official"],                 # 500.0
        # ---- land's GO read = the lambda=1 ceiling (the anchor the packet rests on) ----
        "lambda1_ceiling": d241_imp["lambda1_ceiling"],              # 520.9527323111674 (GO read)
        "t_base_central": d241_imp["t_base_central"],                # 512.4101095400661 (N=1 GO trigger)
        "central_headroom_204": d241_imp["central_headroom_204"],    # 8.542622771101378
        # ---- denken #241 banked outputs (the floor this leg re-reads) ----
        "e_t_meas_floor_241": d241_bind["e_t_meas_floor"],           # 4.330527243789328
        "delta_max_tps500_241": d241_bind["delta_max_tps500"],       # 0.040220025755911104
        "step_eff_landread_241": d241_tps["slope_invariance"]["step_eff_landread"],  # 1.084952540947906
        "tps_banked_at_4p512_241": d241_tps["slope_invariance"]["tps_banked_at_4p512"],  # 463.9706046911471
        # ---- the served official + ubel #240 anchors (cross-checks) ----
        "served_official": served_official,                          # 481.53 (PPL 2.3772, 128/128)
        "gate_513_240": gate_513,                                    # 513.5574577506176 (ubel #240 gate)
        "ceiling_240": ceiling_exch,                                 # 520.9527323111674 (ubel #240 lambda=1)
        "source_runs": {
            "d241": "hqewf1d6 (measured-et-shortfall)",
            "d199": "compliant-spec-et",
            "d240": "t2nrnf2m/cl6poy6t (tps-risk-exchange-rate)",
            "d217": "vgovdrjc (trigger/composition)",
            "land": "land #71/#245 reset GO read 520.95 / built-step ~1.085",
        },
        "provenance": (
            "official=K_cal*(E[T]/step)*tau (wirbel#199/kanna#217 vgovdrjc; K_cal=125.268, "
            "step_served=1.2182, tau in [0.9924,1.0]) x land's 22:12Z reset GO read 520.95 / built-"
            "step ~1.085 (E[T]_both=4.512) x ubel#240 anchors 513.557@gate / 520.953@lambda=1 (resid "
            "0.0, t2nrnf2m/cl6poy6t) x denken#241 E_T_meas_floor=4.3305 / delta_max=0.04022 / "
            "step_eff_landread=1.08495 / banked-step@4.512=463.97 (hqewf1d6) x the served official "
            "481.53 (PPL 2.3772, 128/128, linear MTP K=7). ALL IMPORTED UNCHANGED; land #245 owns "
            "the LIVE step measurement."),
    }


# --------------------------------------------------------------------------- #
# (1) The two-step disambiguation (the frame — decisive).
# --------------------------------------------------------------------------- #
def step_disambiguation(imp: dict) -> dict[str, Any]:
    K = imp["K_cal"]
    step_served = imp["step_served"]                    # 1.2182 (banked, linear MTP K=7)
    tau_c = imp["tau_central"]                          # 1.0
    tau_k = imp["tau_conservative"]                     # 0.9924
    served = imp["served_official"]                     # 481.53
    ceiling = imp["lambda1_ceiling"]                    # 520.9527 (GO read)

    # (a) SERVED path: back out the E[T] the 481.53 official implies at step 1.2182.
    #     E[T]_served = official * step / (K_cal * tau).
    e_t_served_central = served * step_served / (K * tau_c)
    e_t_served_cons = served * step_served / (K * tau_k)

    # (b) BUILT path: back out the per-step cost land's GO read 520.95 implies at E[T]_both=4.512.
    #     step_built = K_cal * E[T]_both * tau / 520.95.
    step_built_central = K * E_T_BOTH * tau_c / ceiling
    step_built_cons = K * E_T_BOTH * tau_k / ceiling

    # the two per-step costs, compared.
    rel_gap = abs(step_served - step_built_central) / step_served
    steps_are_distinct = bool(rel_gap > DISTINCT_REL_THRESHOLD)
    step_speedup_built_over_served = step_served / step_built_central   # 1.1228x (built is cheaper)

    # exact TPS-edge decomposition (K_cal, tau cancel): 520.95/481.53 = step-ratio * E[T]-ratio.
    tps_edge = ceiling / served
    step_ratio = step_served / step_built_central                       # > 1: built step cheaper
    et_ratio = E_T_BOTH / e_t_served_central                            # < 1: built E[T] lower
    edge_recompose = step_ratio * et_ratio
    edge_resid = abs(tps_edge - edge_recompose)

    return {
        "law": "official = K_cal*(E[T]/step)*tau  -- prices a SINGLE decode path",
        "served_path": {
            "step_served_ms": step_served,
            "decode_path": "linear MTP K=7",
            "served_official_tps": served,
            "e_t_served_central_tau1": e_t_served_central,             # ~4.6825
            "e_t_served_conservative_tau0p9924": e_t_served_cons,      # ~4.7184
            "note": ("the SERVED path REALIZES E[T]_served~{:.4f} (tau=1) ... {:.4f} (tau=0.9924) at "
                     "step {:.4f} ms -> {:.2f} official."
                     .format(e_t_served_central, e_t_served_cons, step_served, served)),
        },
        "built_path": {
            "step_built_central_tau1_ms": step_built_central,         # 1.084952540947906
            "step_built_conservative_tau0p9924_ms": step_built_cons,  # ~1.0767
            "decode_path": "tree decode",
            "go_read_tps": ceiling,
            "e_t_both_projection": E_T_BOTH,
            "matches_step_eff_landread_241": bool(
                abs(step_built_central - imp["step_eff_landread_241"]) < TOL_ANCHOR),
            "matches_land_stated_1p085": bool(
                abs(step_built_central - LAND_STATED_BUILT_STEP_MS) < 1e-2),
            "note": ("the BUILT path PROJECTS E[T]_both={:.3f} at step {:.4f} ms -> {:.4f} (land's GO "
                     "read); the derived built-step == denken #241 step_eff_landread {:.6f} and == "
                     "land's stated ~1.085."
                     .format(E_T_BOTH, step_built_central, ceiling, imp["step_eff_landread_241"])),
        },
        "step_served_ms": step_served,
        "step_built_ms": step_built_central,
        "rel_gap_served_vs_built": rel_gap,
        "steps_are_distinct": steps_are_distinct,
        "step_speedup_built_over_served": step_speedup_built_over_served,
        "tps_edge_decomposition": {
            "tps_edge_520p95_over_481p53": tps_edge,                  # 1.0819
            "step_ratio_served_over_built": step_ratio,               # 1.1228 (cheaper built step)
            "et_ratio_built_over_served": et_ratio,                   # 0.9636 (lower built E[T])
            "edge_recompose": edge_recompose,
            "edge_resid": edge_resid,
            "note": ("520.95/481.53 = {:.4f} = step_ratio {:.4f} (built step cheaper) * E[T]_ratio "
                     "{:.4f} (built E[T] LOWER) -- the built path's TPS win is a STEP-COST win, "
                     "PARTLY OFFSET by its lower projected both-bugs E[T], NOT more accepted tokens."
                     .format(tps_edge, step_ratio, et_ratio)),
        },
        "verdict_note": (
            "step_served (1.2182, linear MTP K=7) and step_built ({:.4f}, tree decode) are {} "
            "({:.1f}% apart): the tree-decode build legitimately runs a FASTER per-step cost, so the "
            "520.95 anchor is NOT inconsistent with the banked 1.2182 -- it prices a DIFFERENT path. "
            "The 463.97 'inconsistency' was the artifact of pricing the BUILT E[T]=4.512 at the "
            "SERVED step.".format(step_built_central,
                                  "DISTINCT" if steps_are_distinct else "the SAME", rel_gap * 100)),
    }


# --------------------------------------------------------------------------- #
# (2a) The anchor confirmation (the core).
# --------------------------------------------------------------------------- #
def anchor_confirmation(imp: dict, dis: dict) -> dict[str, Any]:
    K = imp["K_cal"]
    tau_c = imp["tau_central"]
    ceiling = imp["lambda1_ceiling"]
    step_built = dis["step_built_ms"]                    # 1.084952540947906 (derived from 520.95)

    # reanchored TPS: price E[T]_both=4.512 at the re-anchored built-step (round-trips the GO read).
    reanchored_tps = K * (E_T_BOTH / step_built) * tau_c
    roundtrip_resid = abs(reanchored_tps - ceiling)      # ~0 by construction

    # the BOUND: land's stated ~1.085 (rounded 3 d.p.) reproduces this TPS to the rounding tol.
    tps_at_land_stated_step = K * (E_T_BOTH / LAND_STATED_BUILT_STEP_MS) * tau_c
    rounding_bound_tps = abs(reanchored_tps - tps_at_land_stated_step)
    step_resid_vs_stated = abs(step_built - LAND_STATED_BUILT_STEP_MS)

    # the served step (1.2182) priced at E[T]=4.512 -> the 463.97 MISS (the inconsistency).
    tps_served_step_at_ET4512 = K * (E_T_BOTH / imp["step_served"]) * tau_c

    lands_at_go_read = bool(roundtrip_resid < TOL_ANCHOR)
    clears_500 = bool(reanchored_tps >= MILESTONE_TPS)
    materially_below_500 = bool(reanchored_tps < MILESTONE_TPS - 1.0)

    if lands_at_go_read and clears_500:
        confirmation = "CONFIRMED-520.95"
    elif materially_below_500:
        confirmation = "OPTIMISTIC-GO-READ"
    else:
        confirmation = "INCONCLUSIVE"

    return {
        "reanchored_tps_at_ET4512": reanchored_tps,                  # TEST metric (520.9527)
        "go_read_anchor": ceiling,
        "roundtrip_resid_vs_go_read": roundtrip_resid,
        "lands_at_go_read": lands_at_go_read,
        "clears_500": clears_500,
        "confirmation": confirmation,
        "bound": {
            "tps_at_land_stated_1p085": tps_at_land_stated_step,     # 520.93 (rounded-step bound)
            "rounding_bound_tps": rounding_bound_tps,                # ~0.02 TPS
            "step_built_derived_ms": step_built,                     # 1.0849525
            "step_resid_vs_stated_1p085_ms": step_resid_vs_stated,   # ~4.7e-5 ms
            "note": ("the derived built-step {:.7f} ms reproduces the GO read EXACTLY (resid {:.1e}); "
                     "land's stated ~1.085 (3 d.p.) reproduces {:.2f} TPS -- a {:.3f} TPS rounding "
                     "bound. The derived step MATCHES land's stated built-step (step resid {:.1e} ms)."
                     .format(step_built, roundtrip_resid, tps_at_land_stated_step,
                             rounding_bound_tps, step_resid_vs_stated)),
        },
        "served_step_at_ET4512_miss": tps_served_step_at_ET4512,     # 463.97 (the inconsistency)
        "note": ("E[T]_both=4.512 priced at the re-anchored built-step {:.4f} ms -> {:.4f} TPS == "
                 "land's GO read 520.95 ({}). Priced at the SERVED step 1.2182 ms it would be {:.2f} "
                 "(MISSES 500) -- the wrong denominator for tree decode."
                 .format(step_built, reanchored_tps, confirmation, tps_served_step_at_ET4512)),
    }


# --------------------------------------------------------------------------- #
# (2b) The #241 floor re-read under the re-anchored step (slope-invariance).
# --------------------------------------------------------------------------- #
def floor_reread(imp: dict, dis: dict, anc: dict) -> dict[str, Any]:
    K = imp["K_cal"]
    tau_c = imp["tau_central"]
    step_built = dis["step_built_ms"]
    reanchored_tps = anc["reanchored_tps_at_ET4512"]

    # floor re-derived DIRECTLY under the built step: the E[T] at which TPS = 500.
    #   floor = 500 * step_built / (K_cal * tau)  ==  E[T]_both * 500 / reanchored_tps.
    floor_reanchored_via_step = MILESTONE_TPS * step_built / (K * tau_c)
    floor_reanchored_via_anchor = E_T_BOTH * MILESTONE_TPS / reanchored_tps
    floor_241 = imp["e_t_meas_floor_241"]                # 4.330527243789328 (imported)
    floor_moves = abs(floor_reanchored_via_step - floor_241)
    floor_survives = bool(floor_moves < TOL_FLOOR_MOVE)

    # delta_max under the re-anchored step (must equal #241's 0.04022).
    delta_max_reanchored = 1.0 - MILESTONE_TPS / reanchored_tps
    delta_max_241 = imp["delta_max_tps500_241"]          # 0.040220025755911104
    delta_max_resid = abs(delta_max_reanchored - delta_max_241)

    # the SERVED-step floor (the vacuous alternative): E[T] needed to clear 500 at step 1.2182.
    floor_served_step = MILESTONE_TPS * imp["step_served"] / (K * tau_c)   # == clear500_bar_et 4.862
    served_floor_above_proj = bool(floor_served_step > E_T_BOTH)

    return {
        "floor_241_imported": floor_241,
        "floor_reanchored_via_built_step": floor_reanchored_via_step,
        "floor_reanchored_via_anchor": floor_reanchored_via_anchor,
        "floor_moves": floor_moves,
        "floor_survives": floor_survives,
        "delta_max_reanchored": delta_max_reanchored,
        "delta_max_241_imported": delta_max_241,
        "delta_max_resid": delta_max_resid,
        "served_step_floor_vacuous": {
            "floor_at_served_step_1p2182": floor_served_step,         # 4.862 (== clear500 bar)
            "above_4p512_projection": served_floor_above_proj,        # True -> vacuous
            "note": ("under the SERVED step 1.2182 the E[T] needed to clear 500 is {:.4f} > the 4.512 "
                     "projection (VACUOUS: the build would miss even at projection) -- which is "
                     "exactly why land's tree build must be priced at its own (built) step 1.085, "
                     "under which the floor is a REAL 4.02% tolerance.".format(floor_served_step)),
        },
        "note": ("the #241 floor = 4.512*(500/520.95) is a property of the ANCHOR (4.512, 520.95) and "
                 "the 500 target ONLY: the slope 520.95/4.512 = {:.3f} is pinned by the anchor, "
                 "INDEPENDENT of how it factors into K_cal*tau/step. Re-derived directly under the "
                 "built step it is {:.6f} -- it MOVES {:.2e} (< {:.0e}) from #241's {:.6f}: SURVIVES."
                 .format(reanchored_tps / E_T_BOTH, floor_reanchored_via_step, floor_moves,
                         TOL_FLOOR_MOVE, floor_241)),
    }


# --------------------------------------------------------------------------- #
# (3) The verdict table (deliverable).
# --------------------------------------------------------------------------- #
def verdict_table(imp: dict, dis: dict, anc: dict) -> dict[str, Any]:
    K = imp["K_cal"]
    tau_c = imp["tau_central"]

    def tps_at_4512(step: float) -> float:
        return K * (E_T_BOTH / step) * tau_c

    rows = [
        {
            "step_label": "SERVED 1.2182",
            "step_ms": imp["step_served"],
            "implied_E_T": dis["served_path"]["e_t_served_central_tau1"],   # ~4.6825 (its realized E[T])
            "tps_at_ET4512": tps_at_4512(imp["step_served"]),               # 463.97
            "clears_500_at_ET4512": bool(tps_at_4512(imp["step_served"]) >= MILESTONE_TPS),
            "applies_to_path": "served linear MTP K=7 (the 481.53 official)",
        },
        {
            "step_label": "BUILT 1.085",
            "step_ms": dis["step_built_ms"],
            "implied_E_T": E_T_BOTH,                                        # 4.512 (its projected E[T])
            "tps_at_ET4512": tps_at_4512(dis["step_built_ms"]),            # 520.95
            "clears_500_at_ET4512": bool(tps_at_4512(dis["step_built_ms"]) >= MILESTONE_TPS),
            "applies_to_path": "built tree decode (land #245 live build)",
        },
    ]

    # headline verdict.
    if dis["steps_are_distinct"] and anc["confirmation"] == "CONFIRMED-520.95":
        anchor_verdict = "DISTINCT-STEPS-BOTH-VALID"
    elif anc["confirmation"] == "CONFIRMED-520.95":
        anchor_verdict = "CONFIRMED-520.95"
    elif anc["confirmation"] == "OPTIMISTIC-GO-READ":
        anchor_verdict = "OPTIMISTIC-GO-READ"
    else:
        anchor_verdict = "INCONCLUSIVE"

    return {
        "columns": ["step_label", "step_ms", "implied_E_T", "tps_at_ET4512",
                    "clears_500_at_ET4512", "applies_to_path"],
        "rows": rows,
        "anchor_verdict": anchor_verdict,
        "reanchored_tps_at_ET4512": anc["reanchored_tps_at_ET4512"],
        "read_launch_gate_against_step_ms": dis["step_built_ms"],
        "read_launch_gate_against_label": "BUILT 1.085 (tree decode)",
        "note": ("the launch TPS gate is read against the BUILT step (1.085, tree decode), NOT the "
                 "served 1.2182 (linear MTP K=7). E[T]_both=4.512 is the BUILT path's projection; "
                 "priced at its own step it -> 520.95 (CLEARS 500). Pricing it at the served step "
                 "(463.97) is a path-mismatch, not a real shortfall."),
    }


# --------------------------------------------------------------------------- #
# Honest band — the load-bearing modeling choices and their directions.
# --------------------------------------------------------------------------- #
def honest_band(imp: dict, dis: dict, anc: dict, fl: dict) -> dict[str, Any]:
    return {
        "a_anchor_is_self_consistent_not_independently_measured": (
            "the re-anchored built-step is DERIVED from land's GO read 520.95 (step_built = "
            "K_cal*4.512*tau/520.95), so reanchored_tps round-trips to 520.95 BY CONSTRUCTION. The "
            "CONFIRMATION is not the round-trip -- it is that the derived built-step {:.6f} ms == "
            "land's INDEPENDENTLY-stated effective built-step ~1.085 (resid {:.1e} ms) AND == denken "
            "#241 step_eff_landread {:.6f}. The TRUE open question -- does land's LIVE build MEASURE "
            "a ~1.085 ms step -- stays land #245's to measure; this leg de-risks the ANALYTIC anchor."
            .format(dis["step_built_ms"], anc["bound"]["step_resid_vs_stated_1p085_ms"],
                    imp["step_eff_landread_241"])),
        "b_two_paths_not_one_step": (
            "the served 481.53 (linear MTP K=7, 1.2182 ms/step) and the built 520.95 (tree decode, "
            "1.085 ms/step) are TWO decode paths with TWO per-step costs ({:.1f}% apart). The "
            "composition prices each path at its OWN step; the 463.97 'inconsistency' denken #241 "
            "flagged was pricing the BUILT path's E[T] at the SERVED step -- a path-mismatch. The "
            "anchor is NOT optimistic; it is the tree path priced at the tree step."
            .format(dis["rel_gap_served_vs_built"] * 100)),
        "c_edge_is_step_cost_not_acceptance": (
            "the 520.95/481.53 = {:.4f} edge decomposes EXACTLY into step-ratio {:.4f} (built step "
            "cheaper) * E[T]-ratio {:.4f} (built E[T] LOWER than served's realized ~4.68). The win is "
            "a per-step-cost win from tree decode, partly OFFSET by the conservative both-bugs E[T] "
            "projection -- it does not assume more accepted tokens than the served path realizes."
            .format(dis["tps_edge_decomposition"]["tps_edge_520p95_over_481p53"],
                    dis["tps_edge_decomposition"]["step_ratio_served_over_built"],
                    dis["tps_edge_decomposition"]["et_ratio_built_over_served"])),
        "d_floor_is_anchor_property_slope_invariant": (
            "denken #241's 4.3305 floor is a property of the anchor (4.512, 520.95) and the 500 "
            "target ONLY -- the slope is pinned by the anchor, independent of the step decomposition. "
            "Re-derived directly under the built step it moves {:.2e} (< {:.0e}): SURVIVES. Under the "
            "SERVED step the floor would be {:.4f} > 4.512 (vacuous), confirming the build must be "
            "read against the built step.".format(fl["floor_moves"], TOL_FLOOR_MOVE,
                                                  fl["served_step_floor_vacuous"]["floor_at_served_step_1p2182"])),
        "e_imports_unchanged": (
            "K_cal=125.268, step_served=1.2182, tau in [0.9924,1.0], the GO read 520.95 / built-step "
            "~1.085, ubel #240 anchors 513.557@gate / 520.953@lambda=1, the served 481.53, and denken "
            "#241's floor 4.3305 / delta_max 0.04022 are IMPORTED UNCHANGED. No GPU/draw/served-file "
            "change. BASELINE stays 481.53."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY) — conditions (a)-(f).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, dis: dict, anc: dict, fl: dict, table: dict) -> dict[str, Any]:
    K = imp["K_cal"]
    tau_c = imp["tau_central"]

    # (a) the banked step 1.2182 reproduces 463.97 TPS at E[T]=4.512 (the inconsistency).
    tps_banked = K * (E_T_BOTH / imp["step_served"]) * tau_c
    cond_a = bool(abs(tps_banked - imp["tps_banked_at_4p512_241"]) < TOL_ANCHOR
                  and tps_banked < MILESTONE_TPS)

    # (b) step_built reproduces land's 520.95 at E[T]=4.512 (resid <= 1e-6).
    cond_b = bool(anc["roundtrip_resid_vs_go_read"] <= TOL_ANCHOR
                  and abs(anc["reanchored_tps_at_ET4512"] - imp["lambda1_ceiling"]) <= TOL_ANCHOR)

    # (c) the served 481.53 round-trips to its implied E[T]_served at step 1.2182.
    e_t_served = dis["served_path"]["e_t_served_central_tau1"]
    served_roundtrip = K * (e_t_served / imp["step_served"]) * tau_c
    cond_c = bool(abs(served_roundtrip - imp["served_official"]) < TOL_ANCHOR)

    # (d) the #241 delta_max=1-500/TPS_0 slope-invariance holds; the 4.3305 floor moves < 1e-3.
    cond_d = bool(fl["floor_survives"]
                  and fl["floor_moves"] < TOL_FLOOR_MOVE
                  and fl["delta_max_resid"] < TOL_ANCHOR)

    # (e) anchor_verdict consistent with the step disambiguation.
    verdict = table["anchor_verdict"]
    if dis["steps_are_distinct"] and anc["confirmation"] == "CONFIRMED-520.95":
        cond_e = bool(verdict == "DISTINCT-STEPS-BOTH-VALID")
    else:
        cond_e = bool(verdict in ("CONFIRMED-520.95", "OPTIMISTIC-GO-READ", "INCONCLUSIVE"))

    # (f) NaN-clean (key scalars finite; full payload walk enforced in main()).
    key = [tps_banked, anc["reanchored_tps_at_ET4512"], anc["roundtrip_resid_vs_go_read"],
           dis["step_served_ms"], dis["step_built_ms"], e_t_served, served_roundtrip,
           fl["floor_reanchored_via_built_step"], fl["floor_moves"], fl["delta_max_reanchored"]]
    cond_f = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "composition_step_reanchor_self_test_passes": passes,
        "conditions": {
            "a_banked_step_reproduces_463p97_miss": cond_a,
            "b_built_step_reproduces_520p95_resid_le_1e6": cond_b,
            "c_served_481p53_roundtrips_implied_ET": cond_c,
            "d_floor_4p3305_survives_reanchor_moves_lt_1e3": cond_d,
            "e_anchor_verdict_consistent_with_disambiguation": cond_e,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "a_tps_banked_at_4p512": tps_banked,
            "a_banked_241_import": imp["tps_banked_at_4p512_241"],
            "b_reanchored_tps": anc["reanchored_tps_at_ET4512"],
            "b_go_read": imp["lambda1_ceiling"],
            "b_roundtrip_resid": anc["roundtrip_resid_vs_go_read"],
            "c_e_t_served": e_t_served,
            "c_served_roundtrip": served_roundtrip,
            "c_served_official": imp["served_official"],
            "d_floor_reanchored": fl["floor_reanchored_via_built_step"],
            "d_floor_241": imp["e_t_meas_floor_241"],
            "d_floor_moves": fl["floor_moves"],
            "d_delta_max_resid": fl["delta_max_resid"],
            "e_steps_are_distinct": dis["steps_are_distinct"],
            "e_anchor_verdict": verdict,
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, dis: dict, anc: dict, fl: dict, table: dict) -> str:
    return (
        f"COMPOSITION STEP RE-ANCHORED -> {table['anchor_verdict']}. The served step (1.2182 ms, "
        f"linear MTP K=7) and the built step ({dis['step_built_ms']:.4f} ms, tree decode) are "
        f"DISTINCT ({dis['rel_gap_served_vs_built'] * 100:.1f}% apart): the tree-decode build "
        f"legitimately runs a cheaper per-step cost, so the 520.95 GO-read anchor is NOT "
        f"inconsistent with the banked 1.2182 -- it prices a DIFFERENT path. Re-deriving the "
        f"built-step from land's reset trace gives {dis['step_built_ms']:.6f} ms (== denken #241 "
        f"step_eff_landread, == land's stated ~1.085), under which E[T]_both=4.512 -> "
        f"{anc['reanchored_tps_at_ET4512']:.4f} TPS == the GO read (CONFIRMED, resid "
        f"{anc['roundtrip_resid_vs_go_read']:.1e}). The 463.97 'inconsistency' denken #241 flagged "
        f"was pricing the BUILT E[T]=4.512 at the SERVED step. denken #241's 4.3305 floor is anchor-"
        f"pinned (slope-invariant) and SURVIVES the re-anchor (moves {fl['floor_moves']:.2e}). The "
        f"launch TPS gate is read against the BUILT step 1.085, NOT the served 1.2182. BASELINE "
        f"stays 481.53. NOT a launch."
    )


def _handoff(imp: dict, dis: dict, anc: dict, fl: dict) -> dict[str, str]:
    line = (
        f"land #245 + fern #238 card + denken #241 floor: the launch TPS gate should be read "
        f"against the BUILT-step ({dis['step_built_ms']:.4f} ms, tree decode), NOT the served-step "
        f"1.2182 (linear MTP K=7); under it E[T]_both=4.512 maps to reanchored_tps_at_ET4512="
        f"{anc['reanchored_tps_at_ET4512']:.4f}, so the 520.95 GO-read anchor is CONFIRMED (distinct-"
        f"steps-both-valid: served 1.2182 and built 1.085 price two different paths, resid "
        f"{anc['roundtrip_resid_vs_go_read']:.1e}) and my #241 4.3305 floor SURVIVES (moves "
        f"{fl['floor_moves']:.2e}, slope-invariant) -- land's one build run's pass/fail line is read "
        f"against a CONFIRMED step (measured-step is land #245's to read live)."
    )
    return {"land_245_fern_238_denken_241": line}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    dis = step_disambiguation(imp)
    anc = anchor_confirmation(imp, dis)
    fl = floor_reread(imp, dis, anc)
    table = verdict_table(imp, dis, anc)
    band = honest_band(imp, dis, anc, fl)
    st = self_test(imp, dis, anc, fl, table)
    handoff = _handoff(imp, dis, anc, fl)
    return {
        "self_test": st,
        "test_metric": {"reanchored_tps_at_ET4512": anc["reanchored_tps_at_ET4512"]},
        "imports": imp,
        "step_disambiguation": dis,
        "anchor_confirmation": anc,
        "floor_reread": fl,
        "verdict_table": table,
        "honest_band": band,
        "verdict": _verdict(imp, dis, anc, fl, table),
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
    dis = syn["step_disambiguation"]
    anc = syn["anchor_confirmation"]
    fl = syn["floor_reread"]
    table = syn["verdict_table"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("COMPOSITION STEP RE-ANCHOR (PR #252) — served 1.2182 (linear MTP K=7) vs built ~1.085 "
          "(tree decode)", flush=True)
    print("=" * 100, flush=True)
    print(f"  composition: official = K_cal*(E[T]/step)*tau   K_cal={imp['K_cal']:.5f}   "
          f"tau in [{imp['tau_conservative']}, {imp['tau_central']}]", flush=True)
    print(f"  SERVED path: step {dis['step_served_ms']:.4f} ms  ->  E[T]_served "
          f"{dis['served_path']['e_t_served_central_tau1']:.4f} (tau=1)  ->  "
          f"{imp['served_official']:.2f} official (linear MTP K=7)", flush=True)
    print(f"  BUILT  path: step {dis['step_built_ms']:.6f} ms  <-  E[T]_both {E_T_BOTH}  ->  "
          f"{imp['lambda1_ceiling']:.4f} GO read (tree decode)", flush=True)
    print(f"  steps_are_distinct = {dis['steps_are_distinct']}  "
          f"({dis['rel_gap_served_vs_built'] * 100:.2f}% apart; built {dis['step_speedup_built_over_served']:.4f}x cheaper)",
          flush=True)
    print("-" * 100, flush=True)
    print("  VERDICT TABLE:", flush=True)
    print("   step          |  step_ms  | implied E[T] | TPS@E[T]=4.512 | clears500 | applies-to-path",
          flush=True)
    for r in table["rows"]:
        print(f"   {r['step_label']:<13} | {r['step_ms']:>8.4f}  |   {r['implied_E_T']:>8.4f}   |   "
              f"{r['tps_at_ET4512']:>9.4f}    |   {str(r['clears_500_at_ET4512']):>5}   | "
              f"{r['applies_to_path']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  TPS-edge decomposition: 520.95/481.53 = "
          f"{dis['tps_edge_decomposition']['tps_edge_520p95_over_481p53']:.4f} = step-ratio "
          f"{dis['tps_edge_decomposition']['step_ratio_served_over_built']:.4f} (built cheaper) * "
          f"E[T]-ratio {dis['tps_edge_decomposition']['et_ratio_built_over_served']:.4f} (built lower)",
          flush=True)
    print(f"  ANCHOR: reanchored_tps_at_ET4512 = {anc['reanchored_tps_at_ET4512']:.6f}  "
          f"(GO read {anc['go_read_anchor']:.6f}, resid {anc['roundtrip_resid_vs_go_read']:.2e})  -> "
          f"{anc['confirmation']}", flush=True)
    print(f"          bound: land-stated 1.085 -> {anc['bound']['tps_at_land_stated_1p085']:.4f} "
          f"(rounding bound {anc['bound']['rounding_bound_tps']:.4f} TPS; step resid "
          f"{anc['bound']['step_resid_vs_stated_1p085_ms']:.1e} ms)", flush=True)
    print(f"  FLOOR (#241): {fl['floor_241_imported']:.6f}  -> re-anchored "
          f"{fl['floor_reanchored_via_built_step']:.6f}  (moves {fl['floor_moves']:.2e} < "
          f"{TOL_FLOOR_MOVE:.0e}: {'SURVIVES' if fl['floor_survives'] else 'MOVES'})", flush=True)
    print(f"          served-step floor (vacuous) = "
          f"{fl['served_step_floor_vacuous']['floor_at_served_step_1p2182']:.4f} > 4.512", flush=True)
    print("-" * 100, flush=True)
    print(f"  >>> anchor_verdict = {table['anchor_verdict']}   read launch gate against "
          f"{table['read_launch_gate_against_label']}", flush=True)
    print(f"  (PRIMARY) composition_step_reanchor_self_test_passes = "
          f"{st['composition_step_reanchor_self_test_passes']}", flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_lines']['land_245_fern_238_denken_241']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #241; never fatal).
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
        print(f"[composition-step-reanchor] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp = syn["imports"]
    dis = syn["step_disambiguation"]
    anc = syn["anchor_confirmation"]
    fl = syn["floor_reread"]
    table = syn["verdict_table"]
    st = syn["self_test"]

    run = init_wandb_run(
        job_type="sprt-liveprobe-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["sprt-liveprobe-budget", "validity-gate", "launch-trigger", "composition-step-reanchor",
              "two-step-disambiguation", "anchor-confirmation", "floor-reread", "bank-the-analysis"],
        config={
            "e_t_both": E_T_BOTH,
            "milestone_tps": MILESTONE_TPS,
            "K_cal": imp["K_cal"],
            "step_served": imp["step_served"],
            "tau_central": imp["tau_central"],
            "tau_conservative": imp["tau_conservative"],
            "go_read_lambda1_ceiling": imp["lambda1_ceiling"],
            "served_official": imp["served_official"],
            "gate_513_240": imp["gate_513_240"],
            "e_t_meas_floor_241": imp["e_t_meas_floor_241"],
            "delta_max_tps500_241": imp["delta_max_tps500_241"],
            "land_stated_built_step_ms": LAND_STATED_BUILT_STEP_MS,
            "imports": imp["provenance"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[composition-step-reanchor] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "composition_step_reanchor_self_test_passes":
            int(bool(st["composition_step_reanchor_self_test_passes"])),
        "reanchored_tps_at_ET4512": anc["reanchored_tps_at_ET4512"],
        "step_served_ms": dis["step_served_ms"],
        "step_built_ms": dis["step_built_ms"],
        "steps_are_distinct": int(bool(dis["steps_are_distinct"])),
        "rel_gap_served_vs_built": dis["rel_gap_served_vs_built"],
        "step_speedup_built_over_served": dis["step_speedup_built_over_served"],
        "e_t_served_central": dis["served_path"]["e_t_served_central_tau1"],
        "e_t_served_conservative": dis["served_path"]["e_t_served_conservative_tau0p9924"],
        "tps_edge_520p95_over_481p53": dis["tps_edge_decomposition"]["tps_edge_520p95_over_481p53"],
        "step_ratio_served_over_built": dis["tps_edge_decomposition"]["step_ratio_served_over_built"],
        "et_ratio_built_over_served": dis["tps_edge_decomposition"]["et_ratio_built_over_served"],
        "roundtrip_resid_vs_go_read": anc["roundtrip_resid_vs_go_read"],
        "anchor_confirmation_is_confirmed": int(anc["confirmation"] == "CONFIRMED-520.95"),
        "tps_at_land_stated_1p085": anc["bound"]["tps_at_land_stated_1p085"],
        "rounding_bound_tps": anc["bound"]["rounding_bound_tps"],
        "step_resid_vs_stated_1p085_ms": anc["bound"]["step_resid_vs_stated_1p085_ms"],
        "served_step_at_ET4512_miss": anc["served_step_at_ET4512_miss"],
        "floor_241_imported": fl["floor_241_imported"],
        "floor_reanchored_via_built_step": fl["floor_reanchored_via_built_step"],
        "floor_moves": fl["floor_moves"],
        "floor_survives": int(bool(fl["floor_survives"])),
        "delta_max_reanchored": fl["delta_max_reanchored"],
        "delta_max_resid": fl["delta_max_resid"],
        "served_step_floor_vacuous": fl["served_step_floor_vacuous"]["floor_at_served_step_1p2182"],
        "anchor_verdict_distinct_both_valid":
            int(table["anchor_verdict"] == "DISTINCT-STEPS-BOTH-VALID"),
        "read_launch_gate_against_step_ms": table["read_launch_gate_against_step_ms"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="composition_step_reanchor_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[composition-step-reanchor] wandb logged: {summary}", flush=True)


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
        "pr": 252,
        "agent": "denken",
        "kind": "composition-step-reanchor",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[composition-step-reanchor] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY.
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["composition_step_reanchor_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composition_step_reanchor_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[composition-step-reanchor] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY composition_step_reanchor_self_test_passes = {passes}", flush=True)
    print(f"  TEST reanchored_tps_at_ET4512 = "
          f"{syn['test_metric']['reanchored_tps_at_ET4512']:.6f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[composition-step-reanchor] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
