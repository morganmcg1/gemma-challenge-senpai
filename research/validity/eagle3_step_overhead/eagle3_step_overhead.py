#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 drafter step-overhead: re-bank the BUILT-raise target (PR #293).

THE QUESTION
------------
wirbel #290 (ub3kpsso, MERGED) priced the honest step-banked BUILT-raise target at
step_banked_built_raise_target = 4.9029 public E[T] -- but it banked the LINEAR
drafter's step (new_step = 1202.717us). EAGLE-3's drafter is NOT the deployed
single-projection linear MTP head: it does a MULTI-LAYER hidden-state fusion over
target layers {2,21,39} (3 fused layers + a fusion MLP) before the draft head. That
is a HEAVIER draft forward -> a HIGHER per-iteration step -> and since
    official = K_cal * (E[T] / step) * tau,
for FIXED TPS=500 the public-E[T] target scales LINEARLY with step. So the real
EAGLE-3 target is NOT 4.9029 at the linear step -- it is 4.9029 * (eagle3_step /
new_step), at the HEAVIER EAGLE-3 step. #290's feasibility bracket silently assumed
the cheap linear drafter's step; nobody priced the EAGLE-3 drafter's incremental
step cost and re-banked the target against it.

THE LOAD-BEARING GAP
--------------------
wirbel #285's free lossless lever shaved delta_target = 0.0631 off the target
(4.966 -> 4.9029). If the EAGLE-3 drafter's heavier draft forward ADDS MORE than
0.0631 to the step-banked target, the heavier drafter EATS the entire free-lever
relaxation and the net target is back at-or-above fern #281's un-banked 4.966. The
human-gated EAGLE-3 build is sized against the 4.9029 target -- if that target is
optimistically low because it ignores the heavier drafter's step, the Phase-1
viability gate is MIS-SIZED.

THE DRAFT-STEP DECOMPOSITION (banked, NOT re-derived)
-----------------------------------------------------
denken #278 (bu44n30q) decomposed the deployed LINEAR step's WALL components:
draft K=7 chain = 706.8555us, M=1 verify = 4966.78us, wall = 5673.6387us; the
1218.2us served step is a NORMALIZED (batch-amortized) composition unit
(= bridge x wall, bridge = 0.2147). The HONEST draft fraction of the step is the
WALL ratio draft/wall = 706.8555/5673.6387 = 0.124586 = (naive 0.5802) x (bridge
0.2147) -- i.e. the naive 58% over-credited fraction discounted by kanna #286's
draft-side bridge 0.2147 (4.66x over-credit). The TRUE draft step fraction is SMALL
(verify-side dominates); kanna #286's bound_direction shows the bridge is an UPPER
bound (batch=8 draft amortization could shrink it further), so the priced overhead
is CONSERVATIVE for a viability gate. denken #283 (vmxuwxm0) gives a sensitivity
FLOOR from the honest 1/K_cal=7982.9us wall frame: draft 731.76us / 7982.89us =
0.09167 (smaller, because it folds host/scheduling overhead into the wall).

THE DELIVERABLE (this leg)
--------------------------
Re-bank #290's step-banked target against the HEAVIER EAGLE-3 drafter's draft-step
overhead: price the deployed linear draft-step from the banked draft fraction, model
the EAGLE-3 multi-layer-fusion draft as a fusion-cost multiplier sweep
m_fuse in {2,3,4,6} on the linear draft, re-bank the target against the heavier step,
and decide (a) does the corrected target stay inside the feasibility window
(< E_T_max 8.0)? and (b) does the EAGLE-3 step overhead eat the #285 free-lever
relaxation (0.0631)? A NECESSARY de-optimism correction for the human-approval-gated
EAGLE-3 retrain target; the TRUE EAGLE-3 draft-step remains BUILD-measured.

Pure CPU analytic over banked W&B numbers (all imported VERBATIM; never re-derived).
Analysis-only; BASELINE 481.53 untouched (adds 0 TPS). NOT a launch; no served-file
change; no HF Job; no submission; NOT open2; NOT a build."""
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
E_T_MAX = float(K_SPEC + 1)                       # 8.0 full-acceptance (K+1) ceiling

# ---- wirbel #285 97b57hhe -> #290 ub3kpsso: lossless-banked step + step-banked target ----
NEW_STEP_US = 1202.7171244939168                  # banked step after SDPA num_stages 3->2 (bit-ident)
DELTA_TARGET_290 = 0.0631                          # free-lever relaxation (4.966 - 4.9029), imported EXACT
STEP_BANKED_TARGET_290 = 4.9029                    # #290 step-banked target @ the LINEAR step (displayed)

# ---- fern #281 10necg21: Path-A three-axis closure ----
FERN_FLOOR_PUBLIC = 4.966                         # public E[T] needed @ deployed step (priv 0.804)

# ---- denken #119: the LINEAR drafter E[T] structural cap ----
LINEAR_CAP = 3.8445                               # caps even at PERFECT capacity (property of the chain)
HONEST500_FLOOR = 3.9914                          # fern #274/#281 = 500 / K_cal (real/measured E[T])

# ---- denken #278 bu44n30q: deployed LINEAR step WALL decomposition (the draft fraction) ----
DRAFT_K7_CHAIN_US_278 = 706.8555014474051         # K=7 draft chain wall (graphed, CUDA-event)
STEP_WALL_MICRO_US_278 = 5673.638730730329        # draft + M=1 verify wall = step_norm / bridge
NAIVE_DRAFT_FRAC_278 = 0.5802458557276351         # draft / NORMALIZED step (the OVER-CREDITED 58%)
COMPOSITION_OVERCREDIT_278 = 4.817987532332126    # subtract-wall-from-norm over-credit factor

# ---- kanna #286 0k4azmjo: draft-side bridge (bounds the draft overhead from ABOVE) ----
BRIDGE_DRAFT = 0.2147122962556323                 # = step_norm / wall(draft+verify_m1); 4.66x over-credit
BRIDGE_VERIFY = 1.0                               # verify-side deployed-M8 is already normalized

# ---- denken #283 vmxuwxm0: honest 1/K_cal wall frame (sensitivity FLOOR draft fraction) ----
DRAFT_OFFICIAL_US_283 = 731.7620168328832         # draft in the official honest-wall basis
WALL_DEPLOYED_OFFICIAL_US_283 = 7982.887878221502  # 1/K_cal honest per-step wall (incl host overhead)

# ---- EAGLE-3 drafter geometry ----
EAGLE3_TARGET_LAYERS = (2, 21, 39)                # multi-layer hidden-state fusion source layers
L_FUSE_DEPLOYED = 3                               # |{2,21,39}| fused layers (the deployed L_fuse)
M_FUSE_DEPLOYED = 3                               # fusion-cost multiplier at L_fuse=3 (TEST point)
M_FUSE_SWEEP = (2, 3, 4, 6)                        # EAGLE-3 fusion-cost multiplier sweep
M_FUSE_IDENTITY = 1                               # linear-drafter identity (reproduces #290)


# --------------------------------------------------------------------------- #
# Analytic core.
# --------------------------------------------------------------------------- #
def corrected_target_at_step(step_us: float) -> float:
    """Re-bank fern #281's 4.966 floor against a (heavier) step: linear-in-step."""
    return FERN_FLOOR_PUBLIC * (step_us / STEP_US)


def tps_at(et: float, step_us: float) -> float:
    """Composition TPS at public E[T] and normalized step (clean K_cal frame)."""
    return K_CAL * et / (step_us / STEP_US)


def _sweep_rows(g_draft_frac: float, label: str) -> dict[str, Any]:
    """Price the linear draft-step at this draft fraction, sweep m_fuse, re-bank."""
    # ---- (1) PRICE THE LINEAR DRAFT-STEP (of the lossless-banked new_step) ----
    linear_draft_us = g_draft_frac * NEW_STEP_US
    verify_step_us = NEW_STEP_US - linear_draft_us            # unchanged by the drafter swap
    draft_is_minority = bool(linear_draft_us < verify_step_us)

    # #290 anchor recomputed exactly (the m_fuse=1 identity).
    target_290 = corrected_target_at_step(NEW_STEP_US)        # 4.90290 ~ 4.9029
    delta_target = FERN_FLOOR_PUBLIC - target_290            # 0.06310 ~ 0.0631 (== DELTA_TARGET_290)

    rows = []
    for m in (M_FUSE_IDENTITY, *M_FUSE_SWEEP):
        eagle3_draft_us = float(m) * linear_draft_us
        eagle3_step_us = verify_step_us + eagle3_draft_us
        d_step_us = eagle3_step_us - NEW_STEP_US               # step inflation vs the linear step
        # ---- (3) RE-BANK against the heavier step ----
        eagle3_corrected_target = corrected_target_at_step(eagle3_step_us)
        d_target_eagle3 = eagle3_corrected_target - target_290
        # ---- (4) eats the free lever? (identity: <=> corrected_target > fern 4.966) ----
        eats_free_lever = bool(d_target_eagle3 > DELTA_TARGET_290)
        above_fern_floor = bool(eagle3_corrected_target > FERN_FLOOR_PUBLIC)
        # ---- (5) feasibility window ----
        within_window = bool(eagle3_corrected_target < E_T_MAX)
        eroded_headroom = (eagle3_corrected_target - LINEAR_CAP) / (E_T_MAX - LINEAR_CAP)
        rows.append({
            "m_fuse": m,
            "is_eagle3_sweep": bool(m in M_FUSE_SWEEP),
            "eagle3_draft_us": eagle3_draft_us,
            "eagle3_step_us": eagle3_step_us,
            "d_step_us": d_step_us,
            "eagle3_corrected_target": eagle3_corrected_target,
            "d_target_eagle3": d_target_eagle3,
            "eagle3_step_eats_free_lever": eats_free_lever,
            "above_fern_floor": above_fern_floor,
            "eagle3_target_within_window": within_window,
            "eroded_headroom_fraction": eroded_headroom,
        })
    by_m = {r["m_fuse"]: r for r in rows}
    return {
        "label": label,
        "g_draft_frac": g_draft_frac,
        "linear_draft_us": linear_draft_us,
        "verify_step_us": verify_step_us,
        "draft_is_minority": draft_is_minority,
        "target_290_recomputed": target_290,
        "delta_target_recomputed": delta_target,
        "rows": rows,
        "by_m": by_m,
    }


def synthesize() -> dict[str, Any]:
    # ---- DRAFT FRACTION (banked, NOT re-derived) ------------------------- #
    # PRIMARY: denken #278 honest draft fraction of the step = draft_wall / step_wall
    #          (== naive 0.5802 x bridge 0.2147 == the bridge-discounted TRUE fraction).
    g_draft_frac = DRAFT_K7_CHAIN_US_278 / STEP_WALL_MICRO_US_278         # 0.124586
    g_draft_frac_via_bridge = NAIVE_DRAFT_FRAC_278 * BRIDGE_DRAFT          # 0.124586 (cross-check)
    # SENSITIVITY FLOOR: denken #283 honest 1/K_cal wall frame (smaller; folds host overhead).
    g_draft_frac_283 = DRAFT_OFFICIAL_US_283 / WALL_DEPLOYED_OFFICIAL_US_283  # 0.09167
    draft_frac_bounded_by_bridge = bool(g_draft_frac <= BRIDGE_DRAFT + 1e-12)

    primary = _sweep_rows(g_draft_frac, "denken278_bridge_bounded_PRIMARY")
    floor = _sweep_rows(g_draft_frac_283, "denken283_honest_wall_FLOOR")

    # ---- TEST / headline pulls at the DEPLOYED L_fuse=3 (m_fuse=3) -------- #
    r3 = primary["by_m"][M_FUSE_DEPLOYED]
    r3_floor = floor["by_m"][M_FUSE_DEPLOYED]
    eagle3_corrected_target = r3["eagle3_corrected_target"]               # TEST metric
    eagle3_step_eats_free_lever = r3["eagle3_step_eats_free_lever"]       # bool @ L_fuse=3
    eagle3_corrected_target_band = sorted(
        [r3_floor["eagle3_corrected_target"], r3["eagle3_corrected_target"]])

    # NET target = the heaviest-credible corrected target (most de-optimistic, bridge-bounded).
    net_target = max(r["eagle3_corrected_target"] for r in primary["rows"]
                     if r["m_fuse"] in M_FUSE_SWEEP)
    net_target_above_fern = bool(net_target > FERN_FLOOR_PUBLIC)
    window_holds_all_m = all(r["eagle3_target_within_window"]
                             for r in primary["rows"] if r["m_fuse"] in M_FUSE_SWEEP)
    eats_all_m = all(r["eagle3_step_eats_free_lever"]
                     for r in primary["rows"] if r["m_fuse"] in M_FUSE_SWEEP)

    # ---- (6) HONEST CAVEAT ---------------------------------------------- #
    eagle3_draft_step_is_build_measured = True
    honest_caveat = (
        "ANALYTIC step-overhead estimate: eagle3_step_us = verify_step + m_fuse x linear_draft, with "
        "the linear draft fraction (%.5f) from denken #278's banked WALL decomposition, bridge-bounded "
        "above by kanna #286 (0.2147). The TRUE EAGLE-3 draft-step is a BUILD measurement -- the "
        "{2,21,39} fused-layer forward + fusion MLP must be profiled on the A10G. "
        "eagle3_draft_step_is_build_measured = True: the modeled eagle3_step_us is NOT the realized "
        "step. The card DE-OPTIMISMS the target (a NECESSARY correction); it does not finalize it. "
        "Because the draft fraction is the bridge-bounded UPPER value (kanna #286 bound_direction: "
        "batch=8 draft amortization could shrink it further -> denken #283 floor %.5f), the priced "
        "overhead -- and hence the corrected target -- is a CONSERVATIVE upper estimate, the correct "
        "direction for a viability gate (it cannot UNDER-size the bar)."
        % (g_draft_frac, g_draft_frac_283))

    # ---- (7) SELF-TEST (PRIMARY) ---------------------------------------- #
    rows = primary["rows"]
    by_m = primary["by_m"]
    # (a) linear_draft < verify_step (draft is the minority; consistent with bridge 0.2147)
    #     AND the draft fraction is bridge-consistent + bridge-bounded.
    a_draft_minority = bool(
        primary["draft_is_minority"]
        and abs(g_draft_frac - g_draft_frac_via_bridge) < 1e-9
        and draft_frac_bounded_by_bridge)
    # (b) eagle3_step_us monotone increasing in m_fuse.
    steps_seq = [by_m[m]["eagle3_step_us"] for m in (1, 2, 3, 4, 6)]
    b_step_monotone = all(steps_seq[i] < steps_seq[i + 1] for i in range(len(steps_seq) - 1))
    # (c) eagle3_corrected_target monotone increasing AND >= 4.9029 for all m_fuse >= 1.
    tgt_seq = [by_m[m]["eagle3_corrected_target"] for m in (1, 2, 3, 4, 6)]
    c_target_monotone = (all(tgt_seq[i] < tgt_seq[i + 1] for i in range(len(tgt_seq) - 1))
                         and all(t >= STEP_BANKED_TARGET_290 - 1e-3 for t in tgt_seq))
    # (d) at m_fuse=1 (eagle3_step == new_step) the corrected target reproduces #290 = 4.9029.
    d_reproduces_290 = (abs(by_m[1]["eagle3_step_us"] - NEW_STEP_US) < 1e-6
                        and abs(by_m[1]["eagle3_corrected_target"] - STEP_BANKED_TARGET_290) < 1e-3)
    # (e) eats-free-lever reproduces: eats <=> (Delta > 0.0631) <=> (corrected_target > fern 4.966).
    #     The identity holds because target_290 + delta_target == fern 4.966 by construction; no swept
    #     row lands within tol of the 4.966 boundary, so the two predicates agree exactly.
    e_eats_logic = all(
        r["eagle3_step_eats_free_lever"] == (r["d_target_eagle3"] > DELTA_TARGET_290) for r in rows)
    e_identity = all(
        r["eagle3_step_eats_free_lever"] == (r["eagle3_corrected_target"] > FERN_FLOOR_PUBLIC)
        for r in rows)
    e_eats_at_m3 = bool(by_m[3]["eagle3_step_eats_free_lever"])
    e_eats = bool(e_eats_logic and e_identity and e_eats_at_m3)
    # (f) window verdict reproduces: corrected_target < 8.0 evaluated per row.
    f_window = all(
        r["eagle3_target_within_window"] == (r["eagle3_corrected_target"] < E_T_MAX) for r in rows)
    # (g) composition round-trips (481.53 <-> E[T]=3.844 at step=1218.2 reproduces K_cal=125.268).
    k_cal_implied = OFFICIAL_BASELINE / E_T_DEPLOYED
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
                   and abs(E_T_MAX - 8.0) < 1e-9
                   and abs(DELTA_TARGET_290 - 0.0631) < 1e-9
                   and abs(BRIDGE_DRAFT - 0.2147122962556323) < 1e-9)

    cond = {
        "a_draft_minority_bridge_consistent": bool(a_draft_minority),
        "b_eagle3_step_monotone_in_mfuse": bool(b_step_monotone),
        "c_corrected_target_monotone_geq_4p9029": bool(c_target_monotone),
        "d_mfuse1_reproduces_290_target": bool(d_reproduces_290),
        "e_eats_free_lever_logic_and_m3": bool(e_eats),
        "f_window_verdict_reproduces": bool(f_window),
        "g_composition_roundtrips_kcal": bool(g_roundtrip),
        "i_constants_imported_exact": bool(i_constants),
    }
    # (h) NaN-clean is checked on the full payload after assembly (added in main()).

    # ---- VERDICT + HAND-OFF --------------------------------------------- #
    verdict = (
        "Re-banking wirbel #290's step-banked BUILT-raise target (4.9029, priced at the LINEAR drafter's "
        "step) against the HEAVIER EAGLE-3 multi-layer-fusion draft forward. The deployed linear draft is "
        "a SMALL %.2f%% of the step (linear_draft = %.2fus < verify_step = %.2fus; bridge-bounded by kanna "
        "#286 0.2147), so verify dominates. Modeling the EAGLE-3 draft as m_fuse x linear_draft (L_fuse=3 "
        "-> m_fuse~3-4): at the DEPLOYED m_fuse=3 the draft inflates %.2f -> %.2fus (+%.2fus step), raising "
        "the corrected target 4.9029 -> %.4f public E[T] (Delta +%.4f). That overhead (+%.4f) is %.1fx the "
        "#285 free-lever relaxation (0.0631) -> the heavier drafter EATS the free lever at EVERY m_fuse>=2 "
        "(corrected target > fern #281's un-banked 4.966 at all m_fuse>=2). The corrected target stays "
        "INSIDE the feasibility window at every swept m_fuse (max %.4f at m_fuse=6 < E_T_max 8.0; %.1f%% of "
        "the cap->ceiling headroom eroded) -- the window HOLDS but the conservative m_fuse=6 bound nearly "
        "saturates it. Honest de-optimism: the step-overhead-corrected target the human-gated EAGLE-3 "
        "decision must size against is %.4f public E[T] (band [%.4f, %.4f] over the denken #283 floor "
        "-> denken #278 bridge-bounded draft fractions), NOT 4.9029. The modeled draft-step remains "
        "BUILD-measured. BASELINE 481.53 untouched; analysis-only; NOT a launch." % (
            g_draft_frac * 100.0, primary["linear_draft_us"], primary["verify_step_us"],
            primary["linear_draft_us"], r3["eagle3_draft_us"], r3["d_step_us"],
            eagle3_corrected_target, r3["d_target_eagle3"], r3["d_target_eagle3"],
            r3["d_target_eagle3"] / DELTA_TARGET_290, net_target,
            by_m[6]["eroded_headroom_fraction"] * 100.0, eagle3_corrected_target,
            eagle3_corrected_target_band[0], eagle3_corrected_target_band[1]))

    handoff = (
        "pricing the EAGLE-3 drafter's heavier multi-layer-fusion draft forward (L_fuse=3, m_fuse=3) "
        "inflates the deployed linear draft-step from %.2fus to %.2fus, raising your #290 step-banked "
        "target from 4.9029 -> %.4f public E[T] (Delta +%.4f), which DOES eat your own #285 free-lever "
        "relaxation (0.0631) and KEEPS the target INSIDE the feasibility window (%.4f < E_T_max 8.0) -- "
        "so the honest, step-overhead-corrected BUILT-raise target the human-gated EAGLE-3 decision must "
        "size against is %.4f public E[T] (band [%.4f, %.4f]; the modeled draft-step remains "
        "build-measured)." % (
            primary["linear_draft_us"], r3["eagle3_draft_us"], eagle3_corrected_target,
            r3["d_target_eagle3"], eagle3_corrected_target, eagle3_corrected_target,
            eagle3_corrected_target_band[0], eagle3_corrected_target_band[1]))

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE, "target_tps": TARGET_TPS,
            "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL, "step_us": STEP_US,
            "new_step_us": NEW_STEP_US, "tau": TAU, "E_T_deployed": E_T_DEPLOYED, "K_spec": K_SPEC,
            "E_T_max": E_T_MAX, "linear_cap": LINEAR_CAP, "fern_floor_public": FERN_FLOOR_PUBLIC,
            "honest500_floor": HONEST500_FLOOR, "step_banked_target_290": STEP_BANKED_TARGET_290,
            "delta_target_290": DELTA_TARGET_290, "bridge_draft": BRIDGE_DRAFT,
            "bridge_verify": BRIDGE_VERIFY, "eagle3_target_layers": list(EAGLE3_TARGET_LAYERS),
            "L_fuse_deployed": L_FUSE_DEPLOYED, "m_fuse_sweep": list(M_FUSE_SWEEP),
        },
        "draft_fraction": {
            "g_draft_frac_primary_denken278": g_draft_frac,
            "g_draft_frac_via_bridge_crosscheck": g_draft_frac_via_bridge,
            "g_draft_frac_floor_denken283": g_draft_frac_283,
            "naive_draft_frac_278": NAIVE_DRAFT_FRAC_278,
            "draft_frac_bounded_by_bridge": draft_frac_bounded_by_bridge,
            "draft_k7_chain_us_278": DRAFT_K7_CHAIN_US_278,
            "step_wall_micro_us_278": STEP_WALL_MICRO_US_278,
            "composition_overcredit_278": COMPOSITION_OVERCREDIT_278,
        },
        "primary": primary,
        "floor_sensitivity": floor,
        "headline": {
            "eagle3_corrected_target": eagle3_corrected_target,
            "eagle3_corrected_target_band": eagle3_corrected_target_band,
            "eagle3_step_eats_free_lever": eagle3_step_eats_free_lever,
            "net_target": net_target,
            "net_target_above_fern": net_target_above_fern,
            "window_holds_all_m": bool(window_holds_all_m),
            "eats_all_m": bool(eats_all_m),
            "linear_draft_us": primary["linear_draft_us"],
            "verify_step_us": primary["verify_step_us"],
        },
        "honest_caveat": honest_caveat,
        "eagle3_draft_step_is_build_measured": eagle3_draft_step_is_build_measured,
        "self_test": {"conditions": cond, "k_cal_implied": k_cal_implied},
        # ---- headline metrics ----
        "eagle3_corrected_target": eagle3_corrected_target,
        "eagle3_step_eats_free_lever": eagle3_step_eats_free_lever,
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors wirbel #290; never fatal).
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
        print(f"[eagle3-step-overhead] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    df = syn["draft_fraction"]
    hl = syn["headline"]
    pr = syn["primary"]
    st = syn["self_test"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="wirbel", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-step-overhead", "step-banked-target", "draft-step-overhead",
                  "fusion-multiplier", "de-optimism", "necessary-condition", "bank-the-analysis",
                  "pr-293"],
            config={
                "official_baseline": OFFICIAL_BASELINE, "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL,
                "step_us": STEP_US, "new_step_us": NEW_STEP_US, "E_T_deployed": E_T_DEPLOYED,
                "linear_cap": LINEAR_CAP, "fern_floor_public": FERN_FLOOR_PUBLIC,
                "step_banked_target_290": STEP_BANKED_TARGET_290, "delta_target_290": DELTA_TARGET_290,
                "bridge_draft": BRIDGE_DRAFT, "E_T_max": E_T_MAX, "K_spec": K_SPEC,
                "L_fuse_deployed": L_FUSE_DEPLOYED, "m_fuse_sweep": list(M_FUSE_SWEEP),
                "g_draft_frac_primary": df["g_draft_frac_primary_denken278"],
                "imports": "wirbel#290(ub3kpsso target=4.9029 new_step=1202.717 delta=0.0631) x "
                           "denken#278(bu44n30q draft=706.86 wall=5673.64 bridge=0.2147 overcredit=4.82) x "
                           "kanna#286(0k4azmjo draft-bridge=0.2147) x fern#281(10necg21 floor=4.966) x "
                           "denken#283(vmxuwxm0 draft=731.76 wall=7982.89) x denken#119(linear-cap=3.8445) x "
                           "kanna#217(vgovdrjc step=1218.2/K_cal=125.268)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-step-overhead] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-step-overhead] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "eagle3_step_overhead_self_test_passes":
            int(bool(payload["eagle3_step_overhead_self_test_passes"])),
        "eagle3_corrected_target": syn["eagle3_corrected_target"],
        "eagle3_step_eats_free_lever": int(bool(syn["eagle3_step_eats_free_lever"])),
        "step_banked_target_290": STEP_BANKED_TARGET_290,
        "delta_target_290": DELTA_TARGET_290,
        "g_draft_frac_primary": df["g_draft_frac_primary_denken278"],
        "g_draft_frac_floor_denken283": df["g_draft_frac_floor_denken283"],
        "linear_draft_us": hl["linear_draft_us"],
        "verify_step_us": hl["verify_step_us"],
        "net_target": hl["net_target"],
        "net_target_above_fern": int(bool(hl["net_target_above_fern"])),
        "window_holds_all_m": int(bool(hl["window_holds_all_m"])),
        "eats_all_m": int(bool(hl["eats_all_m"])),
        "k_cal_implied": st["k_cal_implied"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-m_fuse rows (full record for future analysis).
    for r in pr["rows"]:
        m = r["m_fuse"]
        summary[f"m{m}_eagle3_step_us"] = r["eagle3_step_us"]
        summary[f"m{m}_eagle3_corrected_target"] = r["eagle3_corrected_target"]
        summary[f"m{m}_d_target_eagle3"] = r["d_target_eagle3"]
        summary[f"m{m}_eats_free_lever"] = int(bool(r["eagle3_step_eats_free_lever"]))
        summary[f"m{m}_within_window"] = int(bool(r["eagle3_target_within_window"]))
        summary[f"m{m}_eroded_headroom_frac"] = r["eroded_headroom_fraction"]

    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_step_overhead_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[eagle3-step-overhead] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-step-overhead] wandb write failed (analysis unaffected): {exc}", flush=True)


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
    print(" EAGLE-3 DRAFTER STEP-OVERHEAD: RE-BANK THE BUILT-RAISE TARGET (PR #293)", flush=True)
    print("=" * 104, flush=True)
    df = syn["draft_fraction"]
    pr = syn["primary"]
    print(f"  (1) LINEAR DRAFT-STEP: g_draft_frac = {df['g_draft_frac_primary_denken278']:.5f} "
          f"(denken #278 draft {DRAFT_K7_CHAIN_US_278:.1f}/wall {STEP_WALL_MICRO_US_278:.1f}; "
          f"= naive {NAIVE_DRAFT_FRAC_278:.4f} x bridge {BRIDGE_DRAFT:.4f}; bounded<=bridge)", flush=True)
    print(f"      linear_draft = {pr['linear_draft_us']:.2f}us  <  verify_step = {pr['verify_step_us']:.2f}us "
          f"(draft is the MINORITY; verify dominates)  [floor frac {df['g_draft_frac_floor_denken283']:.5f}]",
          flush=True)
    print("-" * 104, flush=True)
    print(f"  (2-5) EAGLE-3 m_fuse SWEEP (re-bank: corrected = 4.966 x eagle3_step/{STEP_US:.1f}):", flush=True)
    print(f"      {'m_fuse':>7} {'eagle3_step_us':>15} {'d_step_us':>11} {'corrected_tgt':>14} "
          f"{'d_target':>9} {'eats?':>6} {'<8.0?':>6} {'eroded%':>8}", flush=True)
    for r in pr["rows"]:
        tag = "  (#290 anchor)" if r["m_fuse"] == 1 else (
            "  <- L_fuse=3 TEST" if r["m_fuse"] == M_FUSE_DEPLOYED else "")
        print(f"      {r['m_fuse']:>7} {r['eagle3_step_us']:>15.2f} {r['d_step_us']:>11.2f} "
              f"{r['eagle3_corrected_target']:>14.4f} {r['d_target_eagle3']:>9.4f} "
              f"{str(r['eagle3_step_eats_free_lever']):>6} {str(r['eagle3_target_within_window']):>6} "
              f"{r['eroded_headroom_fraction'] * 100:>7.1f}%{tag}", flush=True)
    hl = syn["headline"]
    print("-" * 104, flush=True)
    print(f"  TEST eagle3_corrected_target (L_fuse=3) = {hl['eagle3_corrected_target']:.4f}  "
          f"band [{hl['eagle3_corrected_target_band'][0]:.4f}, {hl['eagle3_corrected_target_band'][1]:.4f}]",
          flush=True)
    print(f"  eats_free_lever @ L_fuse=3 = {hl['eagle3_step_eats_free_lever']}  "
          f"(eats_all_m>=2 = {hl['eats_all_m']}); net_target = {hl['net_target']:.4f} "
          f"(> fern 4.966 = {hl['net_target_above_fern']}); window_holds_all_m = {hl['window_holds_all_m']}",
          flush=True)
    print(f"  (6) CAVEAT: eagle3_draft_step_is_build_measured = "
          f"{syn['eagle3_draft_step_is_build_measured']}", flush=True)
    st = syn["self_test"]
    print(f"  (7) SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
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
                    default="eagle3-step-overhead")
    args = ap.parse_args(argv)

    syn = synthesize()
    self_test_passes = all(syn["self_test"]["conditions"].values())

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 293, "agent": "wirbel",
        "kind": "eagle3-step-overhead", "synthesis": syn,
        "eagle3_step_overhead_self_test_passes": self_test_passes,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    # fold NaN-clean (condition h) into the PRIMARY pass.
    payload["eagle3_step_overhead_self_test_passes"] = bool(self_test_passes and payload["nan_clean"])
    if nan_paths:
        print(f"[eagle3-step-overhead] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_step_overhead_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[eagle3-step-overhead] wrote {out_path}", flush=True)
    print(f"[eagle3-step-overhead] PRIMARY eagle3_step_overhead_self_test_passes = "
          f"{payload['eagle3_step_overhead_self_test_passes']}", flush=True)
    print(f"[eagle3-step-overhead] TEST eagle3_corrected_target = "
          f"{syn['eagle3_corrected_target']:.4f}", flush=True)
    print(f"[eagle3-step-overhead] eagle3_step_eats_free_lever = "
          f"{syn['eagle3_step_eats_free_lever']}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["eagle3_step_overhead_self_test_passes"]
        print(f"[eagle3-step-overhead] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
