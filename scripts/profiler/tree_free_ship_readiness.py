#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-free-500 SHIP-READINESS gate (PR #109): the go/no-go decision doc that
turns denken #105's GREEN* tree-free ceiling into an *official-submission*
decision.

WHAT THIS ADDS OVER #105
------------------------
#105 answered "*can* the tree-free build-complete stack clear 500, and at what
SplitK?" -- and returned a CENTRAL SplitK-for-500 of 4.44% (GREEN*). But 4.44%
is a CENTRAL projection through two modeled official-side factors: the
local->official multiplier (lawine #99) and tau in [0.96,1.00] (assumed, not
measured). A SHIP decision cannot rest on the central; an official submission is
human-approval-gated and quota-limited, so we need the CONSERVATIVE CORNER to
clear 500 with margin BEFORE we spend the one shot. This module prices that.

THREE OUTPUTS (PR #109):
  Step 1  min_splitk_for_confident_ship_pct  (PRIMARY)
          The SplitK verify-GEMM speedup s at which the CONSERVATIVE CORNER of the
          tree-free projection still clears 500*(1+margin), for margin in
          {0, +1%, +2%}. Contrast vs #105's central 4.44%.
  Step 2  tau_official_reanchor_required     (TEST)
          Decision + rule: can we ship on lawine #99's LOCAL calibration alone
          (tau folded as [0.96,1.00] into the corner), or does the SplitK-modified
          kernel mix change tau enough to force ONE approval-gated official anchor?
  Step 3  the GO/HOLD submit-decision table over (measured SplitK%, tau) cells.

MODEL -- REUSED FROM #105 + lawine #99 (NOT re-derived)
-------------------------------------------------------
We import denken #105's calibrated composition model verbatim:
    official_TPS = K_cal * (E[T]/step) * tau,   K_cal = 481.53/3.844 = 125.268
    step = vg + attn + residual ;  vg = 0.53*(1-f_dq)/(1+s)
and lawine #99's calibrate() for the live multiplier + CI. Two HONEST
corrections to #105's lever bands, both documented inline:

  (1) BYTE-LEVER = wirbel PALETTE, NOT INT8 double-quant. wirbel #104 returned a
      KILL: the deployed FP16 g128 verify-GEMM scales do NOT double-quantize to
      INT8 bit-exactly (only 13.1% round-trip; best lossless hybrid is NET
      NEGATIVE -1.27% bytes; achievable lift ~ -0.02% TPS). It is an
      information-theoretic barrier (8-bit code < 10-bit FP16 mantissa over a
      ~3.6x within-block range), not a tuning issue. The byte lever that ACTUALLY
      exists is the lossless scale PALETTE/LUT (1,009 distinct FP16 values
      globally; a 10-bit index is bit-exact by construction, saves ~37.5% of
      scale bytes ~= 20 MB ~= 0.6% of the verify-GEMM stream -> ~0.3% TPS) -- but
      it is NOT BUILT (a #104 suggested follow-up, "worth a dedicated
      build-or-kill"). So the byte lever is PROJECTED-NOT-BANKED, central ~0.3%,
      CONSERVATIVE-CORNER 0. This is materially tighter than #105's INT8-dq band
      {0.4,0.5,1.1}% and is the dominant reason the corner SplitK bar lands well
      above #105's 4.44%.

  (2) MULTIPLIER factored EXPLICITLY with its CI. #105 baked the deployed
      local->official transfer (the locked-ref ratio 481.53/454.338 = 1.05985)
      into K_cal and carried tau separately. lawine #99 pins the multiplier at
      1.06019 (pooled-mean) with a measured LOCAL-side 95% CI [1.05999,1.06038]
      (+/-0.018%, the config-stable band) and an official-CV sensitivity ENVELOPE
      [1.03941,1.08097] (+/-1.96%). We keep #105's K_cal central (the 1.05985 vs
      1.06019 gap is +0.032% = the sub-MDE self-check residual; we do NOT
      re-center, preserving continuity with the merged #105 = 518.1), and use
      lawine #99 ONLY to supply the multiplier's CI band as a multiplicative
      correction mult/1.06019 (=1.0 at central). The corner uses the LOCAL-side CI
      low (negligible -0.019%).

NO DOUBLE-COUNT (load-bearing for Step 2): the multiplier's official-CV ENVELOPE
(+/-1.96%) and tau in [0.96,1.00] are the SAME risk (will the measured official
match the projection?) expressed twice. We carry that risk ONCE, in tau (the
PR-named gate), and use only the multiplier's genuinely-separate LOCAL-side CI in
the corner. We report the envelope-low as a clearly-labelled WORST-OF-WORST
double-stress sensitivity, and show tau=0.96 (-4%) is already STRICTLY more
conservative than the envelope-low (-1.96%), so the primary corner dominates it.

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no
served-file change. Greedy identity untouched by construction (SplitK 0-flip
kanna #87; palette bit-exact; LK prediction-only). Composition consumes committed
advisor-branch state only.
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


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- REUSE denken #105's calibrated tree-free composition model verbatim ----
tf = _load_module("tree_free_500_ceiling", os.path.join(_HERE, "tree_free_500_ceiling.py"))
# ---- REUSE lawine #99's live multiplier + CI ----
from scripts.profiler.local_official_projection import calibrate  # noqa: E402

TARGET = tf.TARGET_OFFICIAL  # 500.0
FRONTIER = tf.FRONTIER_OFFICIAL  # 481.53

# ---------------------------------------------------------------------------
# Lever bands for the SHIP corner (see module docstring for the two corrections)
# ---------------------------------------------------------------------------
# (1) byte lever = wirbel PALETTE (lossless LUT), NOT the KILLED INT8 double-quant.
#     central ~0.3% TPS (projected, build-or-kill pending); conservative corner 0.
PALETTE_TPS = {"low": 0.0, "central": 0.003, "high": 0.005}
# carried #105 bands (unchanged): LK #95 (projected, AMBER), tau #99, fp32 M8, persist #97.
LK_MULT = tf.LK_MULT          # {low 1.010, central 1.010, high 1.024}
TAU = tf.TAU                  # {low 0.96, central 1.00, high 1.00}
FP32_M8 = tf.FP32_M8          # {low 0, central 0, high 0.000102}
PERSIST = tf.PERSIST_RECLAIM  # {low 0, central 0, high 0.0217} upside-only
SPLITK_UBEL = tf.SPLITK_UBEL  # {low 0.05, central 0.085, high 0.12}
SPLITK_CEILING = tf.SPLITK_CEILING  # 0.2970 bandwidth-gap close
UBEL_PLAUSIBLE_MAX = 0.085    # PR #109 "a SplitK% ubel can plausibly hit (<= ~8.5%)"

# (2) multiplier (lawine #99), live.
_CALIB = calibrate()
MULT_CENTRAL = _CALIB.multiplier                      # 1.06019
MULT_LOCAL_CI = (_CALIB.mult_ci_local_lo, _CALIB.mult_ci_local_hi)   # ~[1.05999,1.06038]
MULT_ENV_CI = (_CALIB.mult_ci_env_lo, _CALIB.mult_ci_env_hi)         # ~[1.03941,1.08097]

MARGINS = [0.0, 0.01, 0.02]   # PR #109 Step 1: clear 500*(1+margin)


def palette_fdq(scenario_key: str) -> float:
    """Palette byte-reduction f_dq for a scenario (reuses #105's TPS->f_dq map)."""
    return tf.dq_tps_to_fdq(PALETTE_TPS[scenario_key])


def ship_point(scenario: str, mult_mode: str = "local") -> dict:
    """Build a ship scenario point. conservative MINIMISES official, optimistic
    MAXIMISES it. mult_mode in {"local","envelope"} selects which multiplier CI
    bound feeds the conservative/optimistic corners (default local; envelope is
    the worst-of-worst double-stress, see docstring)."""
    ci = MULT_LOCAL_CI if mult_mode == "local" else MULT_ENV_CI
    if scenario == "central":
        return {"lk_mult": LK_MULT["central"], "f_dq": palette_fdq("central"),
                "fp32_m8": FP32_M8["central"], "persist_reclaim": PERSIST["central"],
                "tau": TAU["central"], "mult": MULT_CENTRAL}
    if scenario == "conservative":
        return {"lk_mult": LK_MULT["low"], "f_dq": palette_fdq("low"),
                "fp32_m8": FP32_M8["high"], "persist_reclaim": PERSIST["low"],
                "tau": TAU["low"], "mult": ci[0]}
    if scenario == "optimistic":
        return {"lk_mult": LK_MULT["high"], "f_dq": palette_fdq("high"),
                "fp32_m8": FP32_M8["low"], "persist_reclaim": PERSIST["high"],
                "tau": TAU["high"], "mult": ci[1]}
    raise ValueError(scenario)


def official_ship(splitk_s: float, p: dict) -> float:
    """Tree-free official TPS at SplitK speedup s under ship point p.

    Reuses #105's compose() (= K_cal*(E[T]/step)*tau) and applies the lawine #99
    multiplier-CI correction mult/MULT_CENTRAL (=1.0 at central)."""
    base = tf.compose(splitk_s, p)["official_tps"]
    return base * (p["mult"] / MULT_CENTRAL)


def splitk_threshold_for(p: dict, target: float) -> float | None:
    """Minimum SplitK speedup s such that official_ship(s,p) >= target.

    Generalises #105's inverter to an arbitrary target (for the margin ladder) and
    folds the multiplier-CI correction. Returns 0.0 if cleared at s=0, or
    float('inf') if even full bandwidth-gap close cannot reach target."""
    e_t = tf.E_T_LINEAR * p["lk_mult"]
    eff_mult = p["mult"] / MULT_CENTRAL
    step_needed = tf.K_CAL * e_t * p["tau"] * eff_mult / target
    attn = tf.BUDGET["attention"] + p["fp32_m8"]
    residual = (1.0 - tf.BUDGET["verify_gemm"] - tf.BUDGET["attention"]) - p["persist_reclaim"]
    vg_needed = step_needed - attn - residual
    if vg_needed <= 0:
        return float("inf")
    vg_full = tf.BUDGET["verify_gemm"] * (1.0 - p["f_dq"])
    s = vg_full / vg_needed - 1.0
    return 0.0 if s <= 0 else s


def tau_required_at(splitk_s: float, p_levers: dict, target: float) -> float | None:
    """The tau that makes official == target at SplitK s holding the non-tau levers
    of p_levers fixed. Returns None if unreachable (needs tau>1 -> levers/SplitK
    insufficient) within tau<=1, or a value in (0,1]. The Step-2 driver."""
    p = dict(p_levers); p["tau"] = 1.0
    off_at_tau1 = official_ship(splitk_s, p)
    if off_at_tau1 <= 0:
        return None
    tau_need = target / off_at_tau1 * 1.0  # official is linear in tau
    return tau_need


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/spec_cost_model/tree_free_ship_readiness_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="denken/tree-free-ship-readiness")
    ap.add_argument("--wandb-group", default="tree-free-ship-readiness")
    args = ap.parse_args()

    p_cons = ship_point("conservative", "local")
    p_cent = ship_point("central", "local")
    p_opt = ship_point("optimistic", "local")
    p_cons_env = ship_point("conservative", "envelope")  # worst-of-worst double-stress

    # ===================== STEP 1: min_splitk_for_confident_ship =====================
    step1 = {"definition": "min SplitK verify-GEMM speedup s s.t. the CONSERVATIVE CORNER "
                           "clears 500*(1+margin); corner = tau=0.96, multiplier local-CI-low, "
                           "LK floor 1.010, byte-lever(palette)=0, fp32 worst, persist 0",
             "reference_105_central_splitk_for_500_pct": 4.443,
             "by_margin": []}
    for m in MARGINS:
        tgt = TARGET * (1.0 + m)
        s_cons = splitk_threshold_for(p_cons, tgt)
        s_cons_env = splitk_threshold_for(p_cons_env, tgt)
        s_cent = splitk_threshold_for(p_cent, tgt)
        step1["by_margin"].append({
            "margin_pct": m * 100.0, "target_tps": tgt,
            "conservative_corner_splitk_pct": s_cons,
            "conservative_corner_within_ubel_plausible": (
                s_cons is not None and s_cons != float("inf") and s_cons <= UBEL_PLAUSIBLE_MAX),
            "conservative_corner_within_gap_ceiling": (
                s_cons is not None and s_cons != float("inf") and s_cons <= SPLITK_CEILING),
            "central_splitk_pct": s_cent,
            "double_stress_envelope_splitk_pct": s_cons_env,
        })
    # primary scalar = the margin-0 conservative-corner SplitK threshold.
    primary = step1["by_margin"][0]["conservative_corner_splitk_pct"]
    step1["primary_min_splitk_for_confident_ship_pct"] = (
        None if primary in (None, float("inf")) else primary * 100.0)

    # ===================== STEP 2: tau_official_reanchor_required =====================
    # At the SplitK ubel can plausibly deliver, what tau is REQUIRED to clear 500
    # (and 505), holding the OTHER corner levers fixed? If tau_required <= 0.96
    # (band floor) -> the local-calibration worst case already clears -> NO reanchor.
    # If 0.96 < tau_required <= 1.00 -> we rely on tau above its floor; since SplitK
    # changes the kernel mix the deployed multiplier was measured on, the ~0.1-0.3%
    # mix-shift is no longer immaterial against that thin margin -> ONE official
    # anchor needed. If tau_required > 1.00 -> not a tau question (SplitK/levers
    # insufficient even at perfect realization).
    # Hold non-tau levers at the CORNER-conservative bundle (the ship-relevant
    # "projected levers don't realize" view) and at CENTRAL (context).
    s_plaus = SPLITK_UBEL["central"]   # 8.5%
    s_plaus_hi = SPLITK_UBEL["high"]   # 12%
    step2 = {"splitk_plausible_central_pct": s_plaus * 100.0,
             "splitk_plausible_high_pct": s_plaus_hi * 100.0,
             "tau_band": [TAU["low"], TAU["high"]],
             "mix_shift_estimate_pct": 0.3,
             "tau_required": {}}
    for s_label, s_val in (("ubel_central_8.5pct", s_plaus), ("ubel_high_12pct", s_plaus_hi)):
        for lev_label, lev in (("corner_levers", p_cons), ("central_levers", p_cent)):
            for tlabel, tgt in (("clear_500", TARGET), ("clear_505", TARGET * 1.01)):
                tau_need = tau_required_at(s_val, lev, tgt)
                step2["tau_required"][f"{s_label}|{lev_label}|{tlabel}"] = tau_need

    # decision rule applied at the headline point: ubel-central SplitK, corner levers, clear 500.
    tau_need_headline = tau_required_at(s_plaus, p_cons, TARGET)
    tau_need_central_levers = tau_required_at(s_plaus, p_cent, TARGET)
    tau_floor = TAU["low"]
    if tau_need_headline is None or tau_need_headline > 1.0 + 1e-9:
        # even tau=1 + corner levers can't clear at the plausible SplitK
        reanchor = "moot-need-more-splitk-or-levers"
        decision = ("HOLD-on-margin: at ubel-central SplitK the CORNER cannot clear 500 even at "
                    "tau=1.0; the gating lever is SplitK magnitude / palette+LK realization, not tau. "
                    "Re-evaluate tau only once a higher measured SplitK or a realized byte/LK lever "
                    "brings the corner within reach of tau<=1.0.")
    elif tau_need_headline <= tau_floor + 1e-9:
        reanchor = "no"
        decision = ("NO official re-anchor: the corner clears 500 at the tau band FLOOR 0.96, which "
                    "already covers the ~0.3% SplitK kernel-mix shift by >10x -> ship on lawine #99 "
                    "local calibration once ubel lands SplitK.")
    else:
        reanchor = "yes-one-official-anchor"
        decision = (f"YES, ONE official anchor: clearing 500 at the plausible SplitK needs tau >= "
                    f"{tau_need_headline:.3f} (> floor 0.96), i.e. we rely on tau near its ceiling. "
                    f"SplitK is a NEW verify-GEMM kernel -> a new kernel mix the deployed multiplier "
                    f"was never measured on; the ~0.1-0.3% mix-shift is no longer immaterial against "
                    f"this thin tau margin. Spend ONE approval-gated official anchor of the "
                    f"SplitK-built submission to convert tau from assumed-[0.96,1.0] to measured "
                    f"BEFORE trusting 500.")
    step2["tau_required_headline_ubel_central_corner_clear500"] = tau_need_headline
    step2["tau_required_central_levers_clear500"] = tau_need_central_levers
    step2["tau_official_reanchor_required"] = reanchor
    step2["decision"] = decision
    step2["rule"] = ("tau_required(plausible SplitK) <= 0.96 -> NO reanchor (floor already clears, "
                     "mix-shift immaterial) ; in (0.96,1.0] -> YES one official anchor (relying on tau "
                     "above floor; SplitK changes the kernel mix the multiplier was measured on) ; "
                     ">1.0 -> moot, SplitK/levers insufficient (a margin problem, not a tau problem).")

    # ===================== STEP 3: GO/HOLD submit-decision table =====================
    s_rows = sorted(set([0.0, 0.0444, 0.05, 0.065, SPLITK_UBEL["central"], 0.10,
                         SPLITK_UBEL["high"], 0.14, 0.17, 0.20, round(SPLITK_CEILING, 4)]))
    tau_cols = [0.96, 0.98, 1.00]
    # main table cells use the CORNER-conservative non-tau lever bundle (ship gate);
    # we report cons/central/opt lever bundles per (s,tau) in the JSON for completeness.
    def cell_official(s, tau, lever_point):
        p = dict(lever_point); p["tau"] = tau
        return official_ship(s, p)

    table = []
    for s in s_rows:
        row = {"splitk_pct": s * 100.0, "by_tau": {}}
        for tau in tau_cols:
            off_cons = cell_official(s, tau, p_cons)
            off_cent = cell_official(s, tau, p_cent)
            off_opt = cell_official(s, tau, p_opt)
            row["by_tau"][f"tau_{tau:.2f}"] = {
                "corner_conservative_tps": off_cons,
                "central_levers_tps": off_cent,
                "optimistic_tps": off_opt,
                "verdict_corner_vs_500": "GO" if off_cons >= TARGET else "HOLD",
                "verdict_corner_vs_505": "GO" if off_cons >= TARGET * 1.01 else "HOLD",
            }
        table.append(row)

    # the cell we are most likely to land in: ubel central SplitK x the tau band.
    expected_cell = {
        "splitk_pct": SPLITK_UBEL["central"] * 100.0,
        "tau_band": [TAU["low"], TAU["high"]],
        "corner_conservative_at_tau0.96": cell_official(SPLITK_UBEL["central"], 0.96, p_cons),
        "corner_conservative_at_tau1.00": cell_official(SPLITK_UBEL["central"], 1.00, p_cons),
        "central_levers_at_tau0.96": cell_official(SPLITK_UBEL["central"], 0.96, p_cent),
        "central_levers_at_tau1.00": cell_official(SPLITK_UBEL["central"], 1.00, p_cent),
        "optimistic_at_tau1.00": cell_official(SPLITK_UBEL["central"], 1.00, p_opt),
    }

    # ===================== overall ship verdict (GREEN / AMBER / RED) =====================
    s_cons_m0 = step1["by_margin"][0]["conservative_corner_splitk_pct"]
    s_cons_m1 = step1["by_margin"][1]["conservative_corner_splitk_pct"]
    # corner clears at a plausible (<=8.5%) SplitK at margin 0?
    corner_plausible_m0 = (s_cons_m0 not in (None, float("inf"))) and (s_cons_m0 <= UBEL_PLAUSIBLE_MAX)
    # corner reachable within the bandwidth-gap ceiling at all (margin 0)?
    corner_reachable = (s_cons_m0 not in (None, float("inf"))) and (s_cons_m0 <= SPLITK_CEILING)
    # RED: corner < 500 even at SplitK 12% (ubel high) AND tau=1.0 -> not safely reachable tree-free.
    off_corner_12_tau1 = cell_official(SPLITK_UBEL["high"], 1.00, p_cons)
    red = off_corner_12_tau1 < TARGET and (s_cons_m0 in (None, float("inf")) or s_cons_m0 > SPLITK_CEILING)

    if red:
        verdict = "RED"
        verdict_label = ("tree re-enters critical path: the conservative corner cannot clear 500 even "
                         "near the SplitK gap ceiling -> escalate to land #71 tree (lawine #107 step "
                         "ratio governs the tree corner).")
    elif corner_plausible_m0 and reanchor == "no":
        verdict = "GREEN"
        verdict_label = ("ship on local calibration once ubel lands: the conservative corner clears "
                         "500 at a SplitK ubel can plausibly hit (<=8.5%) AND no official re-anchor "
                         "is needed (tau floor already clears).")
    else:
        verdict = "AMBER"
        gating = []
        if not corner_plausible_m0:
            gating.append(f"corner clears 500 only at SplitK {s_cons_m0*100:.1f}% (> ubel-plausible 8.5%)")
        if reanchor == "yes-one-official-anchor":
            gating.append("one official tau-anchor of the SplitK-built submission is the gating measurement")
        if reanchor == "moot-need-more-splitk-or-levers":
            gating.append("at the plausible SplitK the corner needs tau>1.0 -> realize palette/LK or raise SplitK first")
        verdict_label = "HOLD the official shot until: " + " AND ".join(gating)

    gate = {
        "primary_metric_name": "min_splitk_for_confident_ship_pct",
        "min_splitk_for_confident_ship_pct": step1["primary_min_splitk_for_confident_ship_pct"],
        "test_metric_name": "tau_official_reanchor_required",
        "tau_official_reanchor_required": reanchor,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "ubel_must_beat_pct": step1["primary_min_splitk_for_confident_ship_pct"],
        "reference_105_central_splitk_pct": 4.443,
    }

    out = {
        "gate": gate,
        "step1_min_splitk_for_confident_ship": step1,
        "step2_tau_reanchor_decision": step2,
        "step3_go_hold_table": {"rows": table, "tau_cols": tau_cols,
                                "expected_cell": expected_cell,
                                "cell_lever_bundle": "main verdict uses corner-conservative non-tau levers"},
        "model": {
            "formula": "official = K_cal*(E[T]/step)*tau*(mult/mult_central); E[T]=3.844*lk_mult",
            "K_cal": tf.K_CAL, "E_T_linear": tf.E_T_LINEAR, "budget": tf.BUDGET,
            "frontier_official": FRONTIER, "target_official": TARGET,
            "reused_from": "denken #105 tree_free_500_ceiling.compose + lawine #99 calibrate",
        },
        "inputs": {
            "multiplier_central": MULT_CENTRAL,
            "multiplier_local_ci95": list(MULT_LOCAL_CI),
            "multiplier_envelope_ci95": list(MULT_ENV_CI),
            "tau_band": TAU,
            "lk_mult_band": LK_MULT,
            "palette_byte_tps_band": PALETTE_TPS,
            "palette_note": ("byte lever = wirbel #104 PALETTE/LUT (lossless, ~0.3% TPS, BUILD-OR-KILL "
                             "PENDING); INT8 double-quant KILLED (#104, greedy-lossy, net-negative bytes) "
                             "-> conservative corner byte=0"),
            "fp32_m8_abs": FP32_M8, "persist_reclaim_abs": PERSIST,
            "splitk_ubel_band": SPLITK_UBEL, "splitk_gap_ceiling": SPLITK_CEILING,
            "ubel_plausible_max_pct": UBEL_PLAUSIBLE_MAX * 100.0,
            "no_double_count_note": ("multiplier official-CV envelope (+/-1.96%) and tau[0.96,1.0] are the "
                                     "same official-side risk; carried ONCE in tau. corner uses multiplier "
                                     "LOCAL-side CI only. envelope-low reported as worst-of-worst stress."),
        },
        "public_evidence": {
            "leaderboard_frontier_tps": 489.63,
            "leaderboard_note": ("public #1 now frantic-penguin skv64 489.63; a cluster of SplitK/argmax-"
                                 "block-class submissions (byteshark splitkv-k7-argmaxblock64 484.62, "
                                 "need-for-speed mao-gemma-fast-skv64 488.07) sits at ~484-490 -- realized "
                                 "SplitK-class gains in the field are ~+0.6-1.7% over 481.53, BELOW the "
                                 "4.44% central, and NONE clear 500. Corroborates the conservative corner."),
            "digest": "GET /v1/digest?as=senpai 2026-06-14",
        },
        "method": ("CPU-only analytic; reuses denken #105 composition model + lawine #99 multiplier CI; "
                   "two documented band corrections (palette byte-lever, explicit multiplier CI). No GPU, "
                   "no served-file change, greedy identity untouched."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: (None if o == float("inf") else o))

    # ------------------------------- console -------------------------------
    def fmt(s):
        if s is None:
            return "clears@s=0"
        if s == float("inf"):
            return ">ceiling"
        return f"{s*100:.2f}%"

    print("=" * 90)
    print("TREE-FREE-500 SHIP-READINESS GATE (PR #109) -- the official-submission go/no-go")
    print("=" * 90)
    print(f"\nfrontier {FRONTIER} official | target {TARGET} | #105 central SplitK-for-500 = 4.44%")
    print(f"multiplier (lawine #99) {MULT_CENTRAL:.5f}  local-CI [{MULT_LOCAL_CI[0]:.5f},{MULT_LOCAL_CI[1]:.5f}] "
          f"envelope [{MULT_ENV_CI[0]:.5f},{MULT_ENV_CI[1]:.5f}]")
    print(f"byte lever = PALETTE (lossless, ~0.3% central, corner 0); INT8 double-quant KILLED (#104)")

    print(f"\n[STEP 1] min_splitk_for_confident_ship_pct  (CONSERVATIVE CORNER clears 500*(1+margin)):")
    print(f"  {'margin':>7s} {'target':>8s} {'corner s':>10s} {'<=ubel8.5%?':>12s} {'central s':>10s} {'env(stress)':>12s}")
    for r in step1["by_margin"]:
        print(f"  {r['margin_pct']:6.0f}% {r['target_tps']:8.1f} {fmt(r['conservative_corner_splitk_pct']):>10s} "
              f"{('YES' if r['conservative_corner_within_ubel_plausible'] else 'no'):>12s} "
              f"{fmt(r['central_splitk_pct']):>10s} {fmt(r['double_stress_envelope_splitk_pct']):>12s}")
    print(f"  >>> PRIMARY min_splitk_for_confident_ship (margin 0) = "
          f"{step1['primary_min_splitk_for_confident_ship_pct']:.2f}%  (vs #105 central 4.44%) "
          f"-- this is the bar ubel #84 must beat")

    print(f"\n[STEP 2] tau_official_reanchor_required (TEST):")
    print(f"  tau REQUIRED to clear 500 at ubel-central SplitK 8.5%, corner levers = "
          f"{tau_need_headline if tau_need_headline is None else round(tau_need_headline,3)}  (floor 0.96)")
    print(f"  tau REQUIRED at central levers = "
          f"{tau_need_central_levers if tau_need_central_levers is None else round(tau_need_central_levers,3)}")
    print(f"  >>> tau_official_reanchor_required = {reanchor}")
    print(f"      {decision}")

    print(f"\n[STEP 3] GO/HOLD submit-decision table (cells = CORNER-conservative levers; GO iff >=500):")
    print(f"  {'splitk%':>8s} | " + " | ".join(f"tau={t:.2f}" for t in tau_cols))
    for r in table:
        cells = []
        for t in tau_cols:
            c = r["by_tau"][f"tau_{t:.2f}"]
            cells.append(f"{c['corner_conservative_tps']:6.1f} {c['verdict_corner_vs_500']:>4s}")
        mark = "  <-- ubel central" if abs(r["splitk_pct"] - SPLITK_UBEL["central"]*100) < 0.01 else ""
        print(f"  {r['splitk_pct']:7.2f}% | " + " | ".join(cells) + mark)
    print(f"\n  expected cell (ubel 8.5% x tau band): corner {expected_cell['corner_conservative_at_tau0.96']:.1f} "
          f"(tau0.96) .. {expected_cell['corner_conservative_at_tau1.00']:.1f} (tau1.0) | "
          f"central-levers {expected_cell['central_levers_at_tau0.96']:.1f} .. "
          f"{expected_cell['central_levers_at_tau1.00']:.1f}")

    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "tree-free-ship-readiness",
                                 "method": "cpu-analytic-reuse-105-99",
                                 "frontier_official": FRONTIER, "target_official": TARGET,
                                 "multiplier_central": MULT_CENTRAL,
                                 "multiplier_local_ci": list(MULT_LOCAL_CI),
                                 "multiplier_envelope_ci": list(MULT_ENV_CI),
                                 "palette_byte_tps_band": PALETTE_TPS, "tau_band": TAU,
                                 "splitk_ubel": SPLITK_UBEL, "ubel_plausible_max": UBEL_PLAUSIBLE_MAX})

        def jnum(x):
            return 9.99 if x == float("inf") else (-1.0 if x is None else x)

        def pctj(x):  # fraction-or-sentinel -> percent (inf->999, None->-1)
            if x is None:
                return -1.0
            if x == float("inf"):
                return 999.0
            return x * 100.0

        s = wandb.summary
        s["min_splitk_for_confident_ship_pct"] = step1["primary_min_splitk_for_confident_ship_pct"]
        for r in step1["by_margin"]:
            tag = f"m{int(r['margin_pct'])}"
            s[f"splitk_corner_{tag}_pct"] = pctj(r["conservative_corner_splitk_pct"])
            s[f"splitk_central_{tag}_pct"] = pctj(r["central_splitk_pct"])
            s[f"splitk_env_stress_{tag}_pct"] = pctj(r["double_stress_envelope_splitk_pct"])
        s["tau_required_ubel_central_corner_clear500"] = jnum(tau_need_headline)
        s["tau_required_central_levers_clear500"] = jnum(tau_need_central_levers)
        s["tau_official_reanchor_required"] = reanchor
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label
        s["reference_105_central_splitk_pct"] = 4.443
        s["expected_cell_corner_tau096"] = expected_cell["corner_conservative_at_tau0.96"]
        s["expected_cell_corner_tau100"] = expected_cell["corner_conservative_at_tau1.00"]
        s["expected_cell_central_tau100"] = expected_cell["central_levers_at_tau1.00"]
        s["public_leaderboard_frontier_tps"] = 489.63

        # Step-1 margin table
        t1 = wandb.Table(columns=["margin_pct", "target_tps", "corner_splitk_pct",
                                  "within_ubel_plausible", "central_splitk_pct",
                                  "envelope_stress_splitk_pct"])
        for r in step1["by_margin"]:
            t1.add_data(r["margin_pct"], r["target_tps"], pctj(r["conservative_corner_splitk_pct"]),
                        bool(r["conservative_corner_within_ubel_plausible"]),
                        pctj(r["central_splitk_pct"]), pctj(r["double_stress_envelope_splitk_pct"]))
        wandb.log({"step1_margin_ladder": t1})

        # Step-3 GO/HOLD table (corner conservative)
        t3 = wandb.Table(columns=["splitk_pct"] + [f"corner_tau{t:.2f}" for t in tau_cols]
                         + [f"GO_tau{t:.2f}" for t in tau_cols])
        for r in table:
            vals = [r["splitk_pct"]]
            gos = []
            for t in tau_cols:
                c = r["by_tau"][f"tau_{t:.2f}"]
                vals.append(c["corner_conservative_tps"])
                gos.append(c["verdict_corner_vs_500"])
            t3.add_data(*(vals + gos))
            wandb.log({"table/splitk_pct": r["splitk_pct"],
                       "table/corner_tau096": r["by_tau"]["tau_0.96"]["corner_conservative_tps"],
                       "table/corner_tau100": r["by_tau"]["tau_1.00"]["corner_conservative_tps"],
                       "table/central_tau100": r["by_tau"]["tau_1.00"]["central_levers_tps"],
                       "table/target": TARGET})
        wandb.log({"step3_go_hold_table": t3})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
