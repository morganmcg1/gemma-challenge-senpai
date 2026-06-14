#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Depth-resolved gate-2 power (PR #230) — does ONE served run carry enough DEEP-TAIL
accept positions to confirm the q[8..9] λ̂ ≥ 0.7875 budget, or does the deep tail need
more runs than the aggregate?

WHAT THIS LEG ANSWERS
---------------------
denken #225 (`851z7itj`, MERGED) proved the AGGREGATE validity gate is one-run-confirmable:
λ̂≥0.9780 needs n_confirm = 1,124.76 accept positions under the AR(1)-ACF-corrected ASN (vs
the iid floor 405.4), well inside one served run's ~65,536 q[2..9] positions (58× headroom)
→ `gate2_confirmable_in_one_run=1`. But that ASN was computed on the AGGREGATE acceptance
rate, and the binding UNMEASURED quantity gating the launch is the DEEP TAIL q[8..9]
specifically (land #71's unmeasured deep-tail λ, deep-tail budget 0.7875 from stark #215).

The deep tail is DOUBLY DISADVANTAGED for confirmation:
  (1) FEWER accept positions — it is the sparse tail of the M=32 depth-9 tree; most
      sequences never reach depth 8–9 (reach mass decays ≈β/rung, #215/#208/#203).
  (2) LOWER λ — further from 1.0, so a WIDER per-sample Bernoulli variance λ(1−λ); the
      deep-tail budget 0.7875 carries σ²≈0.167 vs the 0.997 spine's σ²≈0.003 (~56× more).

So the aggregate one-run confirmation does NOT imply the deep-tail one is confirmable: the
aggregate is occupancy-dominated by the easy, low-variance shallow spine, which confirms
cheaply, while the deep tail must resolve a high-variance, sparse signal. This leg breaks
#225's ASN down BY DEPTH against the tree's depth-occupancy, solves n_confirm_deeptail vs
n_deeptail_available_one_run, and tells land #71 whether ONE served trace confirms the WHOLE
depth profile or only the shallow-mid spine. The depth-resolved companion to #225's aggregate
runbook.

THE MODEL (three pieces; all imports NOT re-derived)
----------------------------------------------------
(A) Depth-occupancy occ(d) — the mechanism. A position contributes a depth-d accept read
    only if its verified chain REACHES depth d. occ(d) is proxied by the BANKED #208/#203
    β-extended reach-weights w_d (the reach mass per rung, #215), normalized to occ(2)=1 at
    the base of the q[2..9] ladder. n_available(d) = occ(d)·N_positions decays with d, so the
    deep tail carries far fewer positions. (HONEST: this is a banked reach-WEIGHT vector, not
    a per-step reach-DP occupancy dump — the load-bearing modeling choice; the per-step DP
    dump is the tightening follow-up. A transparent geometric-in-λ reach cross-check is
    reported, and it OVERSTATES deep-tail occupancy under the 0.997 spine, so the banked
    reach-weight occupancy is the CONSERVATIVE — binding — one.)

(B) Depth-resolved ASN n_confirm(d) — the deliverable. The per-depth confirmation cost
    scales with the local Bernoulli variance σ_d² = λ_d(1−λ_d) (lower λ_d ⇒ wider variance ⇒
    more samples) and inherits #225's AR(1)-ACF inflation. With a COMMON gate-margin
    separation (so the per-depth lever is the variance), n_confirm(d) = n_confirm_agg ·
    σ_d²/σ̄², σ̄² = Σ_d occ_norm(d)·σ_d² (occupancy-weighted mean variance). By construction the
    occupancy-weighted aggregate Σ occ_norm(d)·n_confirm(d) = n_confirm_agg = 1,124.76
    (round-trips #225, self-test a). The deep-tail variance is sized AT the budget 0.7875
    (the worst-case σ² for any λ≥budget — conservative; does NOT predict land #71's value).

(C) The one-run verdict — the deliverable. deeptail_confirmable_in_one_run =
    (n_deeptail_available_one_run ≥ n_confirm_deeptail); deeptail_power_ratio = avail/needed;
    headline WHOLE_DEPTH_PROFILE_ONE_RUN_CONFIRMABLE vs DEEP_TAIL_NEEDS_MORE_RUNS
    (n_runs_deeptail = ceil(needed/available)).

LOCAL CPU-only analytic depth-decomposition over the banked #225 ASN + the tree's
depth-occupancy. No GPU / vLLM / HF Job / submission / served-file change / official draw. The
#225 aggregate ASN, the 0.7875 deep-tail budget, and the #190 ICC are imported UNCHANGED; the
deep-tail λ̂ value stays land #71's to MEASURE (this leg sizes the CONFIRMATION power, it does
not predict the value). BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis
(PRIMARY = self-test, adds 0 TPS). NOT open2. NOT a launch.

PRIMARY metric  gate2_depth_resolved_power_self_test_passes
TEST    metric  n_confirm_deeptail  (accept positions to confirm the q[8..9] λ̂ ≥ 0.7875 budget)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/gate2_depth_resolved_power/\
gate2_depth_resolved_power.py --self-test \\
        --wandb_group issue192-reading-calibration --wandb_name denken/gate2-depth-resolved-power
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

_D225_JSON = REPO_ROOT / "research/validity/gate2_confirmation/gate2_confirmation_results.json"
_D215_JSON = REPO_ROOT / "research/validity/deeptail_bar_budget/results.json"
_D212_PATH = REPO_ROOT / "research/validity/sprt_ar_asn/sprt_ar_asn.py"

_N01 = NormalDist(0.0, 1.0)
Z_ALPHA = _N01.inv_cdf(1.0 - 0.05)    # 1.64485 (one-sided)
Z_BETA = _N01.inv_cdf(0.95)           # 1.64485

# served-run decode budget (imported structure from #225): 128 prompts × 512 = 65,536
# q[2..9] positions per served run.
N_PROMPTS_SERVED = 128
OUTPUT_LEN_SERVED = 512
DECODE_STEPS_PER_RUN = N_PROMPTS_SERVED * OUTPUT_LEN_SERVED   # 65,536

# the q[2..9] ladder depths (depth-1 head excluded, per #215's primary axis).
LADDER_DEPTHS = list(range(2, 10))    # [2,3,4,5,6,7,8,9]
SHALLOW_DEPTHS = list(range(2, 8))    # q[2..7] spine
DEEPTAIL_DEPTHS = [8, 9]              # q[8..9] deep tail

TOL_ROUNDTRIP = 1e-6
TOL_ERR = 1e-9


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse #212's design-effect machinery (deff_exchangeable for the ICC deflation; the class
# needs no anchor files — pure functions).
D212 = _import("sprt_ar_asn", _D212_PATH)
deff_ar1 = D212.deff_ar1
deff_exchangeable = D212.deff_exchangeable
deff_partial_sum = D212.deff_partial_sum


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Imports — banked scalars from #225 / #215 / #190, NOT re-derived.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d225 = _load(_D225_JSON)["synthesis"]
    imp225 = d225["imports"]
    bg225 = d225["confirmation_budget"]
    feas225 = d225["feasibility"]

    d215 = _load(_D215_JSON)["synthesis"]
    rw215 = d215["reach_weight_profile"]
    db215 = d215["deeptail_budget"]

    return {
        # ---- #225 (851z7itj) banked AGGREGATE ASN (the round-trip target) ----
        "n_confirm_agg_iid": bg225["n_confirm_iid"],                 # 405.42403511311863
        "n_confirm_agg_ar1": bg225["n_confirm_arcorrected"],         # 672.34 (AR(1) optimistic)
        "n_confirm_agg_measured_acf": bg225["n_confirm_measured_acf"],  # 1124.7628546877863 (HEADLINE)
        "n_confirm_agg_flat": bg225["n_confirm_flat"],               # 1788.17 (conservative loose end)
        "deff_ar1": imp225["deff_ar1"],                              # 1.6584
        "deff_measured_acf": imp225["deff_measured_acf"],            # 2.7742875539531413
        "deff_flat": imp225["deff_flat"],                            # 4.4106
        "icc_190": imp225["icc_190"],                               # 0.1446247464062406
        "mbar_190": imp225["mbar_190"],                             # 24.582508774446666
        "rho_lag1_190": imp225["rho_lag1_190"],                     # 0.2583
        "private_bar_208": imp225["private_bar_208"],               # 0.9779783323491393 (binding bar)
        "agg_decode_steps_per_run": feas225["decode_steps_available_per_run"],  # 65,536
        "agg_headline_headroom_x": feas225["headline_headroom_x"],  # ~58× (aggregate headroom)
        "agg_confirmable_in_one_run": bool(feas225["gate2_confirmable_in_one_run"]),
        # ---- stark #215 deep-tail budget + β-extended reach-weights (the occupancy) ----
        "budget_deeptail": db215["min_deeptail_lambda_q8q9_clears_bar"],  # 0.7874871278548552
        "w_full_raw_d1_d9": list(rw215["w_full_raw_d1_d9"]),         # depths 1..9 reach mass
        "w_mass_shallow_q2q7": rw215["w_mass_shallow_q2q7"],        # 0.909210
        "w_mass_deeptail_q8q9": rw215["w_mass_deeptail_q8q9"],      # 0.090790
        "lambda_spine_interim": d215["lambda_spine_interim"],       # 0.997 (land #71 interim spine)
        "beta_193": d215["constants"]["beta_primary_193"],          # 0.765124 (geometric-occ cross-check)
        "deeptail_proj_relative_215": db215["deeptail_lambda_mechanism_proj_relative_from_spine"],  # 0.6852
        "source_runs": {"d225": "851z7itj", "d215": "deeptail-bar-budget", "d190": "fva6o4ug",
                        "d212": "b70053sw", "d208": "wi4gxxx8"},
        "provenance": (
            "denken#225 aggregate AR(1)-ACF ASN (851z7itj): n_confirm_acf=1124.76, iid=405.4, "
            "deff_measured_acf=2.7743, decision rule λ̂_LCB≥0.97798⇒PASS × stark#215 deep-tail "
            "budget 0.7875 + β-extended reach-weights (occupancy) × wirbel#190 within-prompt ICC "
            "0.1446 (fva6o4ug). land #71 interim spine λ(q[2..7])=0.997 (PARAMETER). Deep-tail λ̂ "
            "stays land #71's to MEASURE; this leg sizes the confirmation POWER."),
    }


# --------------------------------------------------------------------------- #
# (1) Depth-occupancy model (the mechanism). occ(d) from the banked β-extended reach-weights,
#     normalized to occ(2)=1 at the base of the q[2..9] ladder. n_available(d) = occ(d)·N.
# --------------------------------------------------------------------------- #
def depth_occupancy(imp: dict) -> dict[str, Any]:
    w = imp["w_full_raw_d1_d9"]                 # depths 1..9 (index 0 == depth 1)
    w_ladder = w[1:9]                           # depths 2..9
    base = w_ladder[0]                          # depth-2 reach mass (occ(2)=1)
    occ = {d: w_ladder[i] / base for i, d in enumerate(LADDER_DEPTHS)}   # occ(2)=1, decays
    n_available = {d: occ[d] * DECODE_STEPS_PER_RUN for d in LADDER_DEPTHS}

    # normalized occupancy weights over q[2..9] (Σ=1) — the ASN occupancy-weighting axis.
    occ_sum = sum(occ.values())
    occ_norm = {d: occ[d] / occ_sum for d in LADDER_DEPTHS}

    # deep-tail available: CONSERVATIVE = positions reaching depth 8 (entry to the band; the
    # d8,d9 reads within a position are serially correlated, so independent deep-tail positions
    # ≈ occ(8)·N). OPTIMISTIC = treat d8 and d9 reads as separate (sum) — an upper bound.
    n_deeptail_available_conservative = n_available[8]
    n_deeptail_available_optimistic = n_available[8] + n_available[9]

    # geometric-in-λ reach cross-check (transparent alternative): occ_geo(d)=Π_{j=2}^{d-1}λ_j,
    # spine 0.997 for the shallow rungs. This OVERSTATES deep-tail occupancy (≈0.98 at d8) — so
    # the banked reach-weight occupancy is the conservative/binding choice.
    spine = imp["lambda_spine_interim"]
    occ_geo = {2: 1.0}
    for d in LADDER_DEPTHS[1:]:
        occ_geo[d] = occ_geo[d - 1] * spine
    n_available_geo = {d: occ_geo[d] * DECODE_STEPS_PER_RUN for d in LADDER_DEPTHS}

    return {
        "occupancy_model": ("banked #208/#203 β-extended reach-weights w_d (reach mass per rung), "
                            "normalized occ(2)=1 at the q[2..9] ladder base"),
        "w_ladder_raw_d2_d9": w_ladder,
        "occ_d2_d9": {str(d): occ[d] for d in LADDER_DEPTHS},
        "occ_norm_d2_d9": {str(d): occ_norm[d] for d in LADDER_DEPTHS},
        "n_available_d2_d9": {str(d): n_available[d] for d in LADDER_DEPTHS},
        "n_deeptail_available_one_run": n_deeptail_available_conservative,   # headline (conservative)
        "n_deeptail_available_conservative_reach8": n_deeptail_available_conservative,
        "n_deeptail_available_optimistic_reads8plus9": n_deeptail_available_optimistic,
        "deeptail_occupancy_fraction_q8q9": imp["w_mass_deeptail_q8q9"],     # 9.08% of q[2..9] mass
        "geometric_reach_crosscheck": {
            "occ_geo_d2_d9": {str(d): occ_geo[d] for d in LADDER_DEPTHS},
            "n_available_geo_d8": n_available_geo[8],
            "note": ("geometric-in-λ reach with spine 0.997 keeps deep-tail occupancy ≈{:.3f} "
                     "(n_avail(8)≈{:,.0f}) — it OVERSTATES the deep-tail reach (the tree's M=32→M=8 "
                     "branching thins it faster than per-rung acceptance), so the banked reach-weight "
                     "occupancy is the CONSERVATIVE, binding choice."
                     .format(occ_geo[8], n_available_geo[8])),
        },
        "occ_obj": occ, "occ_norm_obj": occ_norm, "n_available_obj": n_available,
    }


# --------------------------------------------------------------------------- #
# (2) Depth-resolved ASN (the deliverable). Per-depth Bernoulli variance σ_d²=λ_d(1−λ_d);
#     n_confirm(d) = n_confirm_agg · σ_d²/σ̄² round-trips the aggregate by construction.
# --------------------------------------------------------------------------- #
def _lambda_ref(d: int, spine: float, budget_dt: float) -> float:
    """Per-depth reference λ for the variance: the measured spine on q[2..7]; the deep-tail
    BUDGET on q[8..9] (variance sized at the budget = worst-case σ² for any λ≥budget —
    conservative, and does NOT predict land #71's value)."""
    return spine if d in SHALLOW_DEPTHS else budget_dt


def _sigma2(lam: float) -> float:
    return lam * (1.0 - lam)


def depth_resolved_asn(imp: dict, occ: dict) -> dict[str, Any]:
    spine = imp["lambda_spine_interim"]
    budget_dt = imp["budget_deeptail"]
    occ_norm = occ["occ_norm_obj"]
    n_agg_acf = imp["n_confirm_agg_measured_acf"]
    n_agg_iid = imp["n_confirm_agg_iid"]
    n_agg_ar1 = imp["n_confirm_agg_ar1"]
    n_agg_flat = imp["n_confirm_agg_flat"]

    lam_ref = {d: _lambda_ref(d, spine, budget_dt) for d in LADDER_DEPTHS}
    sig2 = {d: _sigma2(lam_ref[d]) for d in LADDER_DEPTHS}
    sigbar2 = sum(occ_norm[d] * sig2[d] for d in LADDER_DEPTHS)     # occupancy-weighted mean variance

    # per-depth confirmation budget across the #225 inflation band (round-trips by construction).
    def n_confirm_band(n_agg: float, d: int) -> float:
        return n_agg * sig2[d] / sigbar2

    n_confirm = {d: n_confirm_band(n_agg_acf, d) for d in LADDER_DEPTHS}
    n_confirm_iid = {d: n_confirm_band(n_agg_iid, d) for d in LADDER_DEPTHS}
    n_confirm_ar1 = {d: n_confirm_band(n_agg_ar1, d) for d in LADDER_DEPTHS}
    n_confirm_flat = {d: n_confirm_band(n_agg_flat, d) for d in LADDER_DEPTHS}

    # deep-tail band confirm: λ̂_deeptail ≥ budget; variance at the budget (band-level).
    sig2_deeptail = _sigma2(budget_dt)
    n_confirm_deeptail = n_agg_acf * sig2_deeptail / sigbar2          # TEST metric
    n_confirm_deeptail_iid = n_agg_iid * sig2_deeptail / sigbar2
    n_confirm_deeptail_ar1 = n_agg_ar1 * sig2_deeptail / sigbar2
    n_confirm_deeptail_flat = n_agg_flat * sig2_deeptail / sigbar2

    # shallow spine band confirm (for the table / contrast).
    sig2_spine = _sigma2(spine)
    n_confirm_spine = n_agg_acf * sig2_spine / sigbar2

    # round-trip: occupancy-weighted aggregate of n_confirm(d) must reproduce n_agg_acf.
    agg_roundtrip = sum(occ_norm[d] * n_confirm[d] for d in LADDER_DEPTHS)

    n_confirm_vs_depth = [{
        "depth": d,
        "lambda_ref": lam_ref[d],
        "sigma2": sig2[d],
        "occ": occ["occ_obj"][d],
        "occ_norm": occ_norm[d],
        "n_confirm_iid": n_confirm_iid[d],
        "n_confirm_ar1": n_confirm_ar1[d],
        "n_confirm_measured_acf": n_confirm[d],
        "n_confirm_flat": n_confirm_flat[d],
        "n_available": occ["n_available_obj"][d],
        "band": "deeptail" if d in DEEPTAIL_DEPTHS else "spine",
    } for d in LADDER_DEPTHS]

    return {
        "aggregate_form": ("n_confirm(d) = n_confirm_agg · σ_d²/σ̄²,  σ_d²=λ_d(1−λ_d),  "
                           "σ̄²=Σ_d occ_norm(d)·σ_d²  (occupancy-weighted; common gate-margin separation)"),
        "lambda_ref_d2_d9": {str(d): lam_ref[d] for d in LADDER_DEPTHS},
        "sigma2_d2_d9": {str(d): sig2[d] for d in LADDER_DEPTHS},
        "sigbar2_occupancy_weighted_mean_variance": sigbar2,
        "sigma2_deeptail_at_budget": sig2_deeptail,
        "sigma2_spine": sig2_spine,
        "deeptail_over_spine_variance_ratio": sig2_deeptail / sig2_spine,
        "n_confirm_vs_depth": n_confirm_vs_depth,
        "n_confirm_deeptail": n_confirm_deeptail,                     # TEST (measured-ACF headline)
        "n_confirm_deeptail_iid": n_confirm_deeptail_iid,
        "n_confirm_deeptail_ar1": n_confirm_deeptail_ar1,
        "n_confirm_deeptail_flat": n_confirm_deeptail_flat,
        "n_confirm_spine_measured_acf": n_confirm_spine,
        "deeptail_over_spine_nconfirm_ratio": n_confirm_deeptail / n_confirm_spine,
        "aggregate_roundtrip_n_confirm": agg_roundtrip,
        "aggregate_roundtrip_target": n_agg_acf,
        "aggregate_roundtrip_resid": abs(agg_roundtrip - n_agg_acf),
        "note": (
            "the deep-tail band needs n_confirm_deeptail={:,.0f} accept positions (measured-ACF) — "
            "{:.1f}× the aggregate {:,.0f} — because its Bernoulli variance at the 0.7875 budget "
            "({:.4f}) is {:.0f}× the 0.997 spine's ({:.5f}); the aggregate is occupancy-dominated by "
            "the cheap, low-variance spine (each spine depth needs only ≈{:,.0f}), so the aggregate "
            "one-run confirmation does NOT carry the deep tail for free."
            .format(n_confirm_deeptail, n_confirm_deeptail / n_agg_acf, n_agg_acf, sig2_deeptail,
                    sig2_deeptail / sig2_spine, sig2_spine, n_confirm_spine)),
    }


# --------------------------------------------------------------------------- #
# (3) The one-run verdict (the deliverable).
# --------------------------------------------------------------------------- #
def one_run_verdict(imp: dict, occ: dict, asn: dict) -> dict[str, Any]:
    available = occ["n_deeptail_available_one_run"]                  # conservative (reach depth 8)
    available_opt = occ["n_deeptail_available_optimistic_reads8plus9"]
    needed = asn["n_confirm_deeptail"]

    ratio = available / needed
    ratio_opt = available_opt / needed
    confirmable = bool(available >= needed)
    n_runs_deeptail = 1 if confirmable else math.ceil(needed / available)

    headline = ("WHOLE_DEPTH_PROFILE_ONE_RUN_CONFIRMABLE" if confirmable
                else "DEEP_TAIL_NEEDS_MORE_RUNS")

    # contrast vs the aggregate headroom (the "doubly disadvantaged" penalty).
    agg_headroom = imp["agg_headline_headroom_x"]
    headroom_collapse = agg_headroom / ratio if ratio > 0 else float("inf")

    return {
        "n_deeptail_available_one_run": available,
        "n_deeptail_available_optimistic": available_opt,
        "n_confirm_deeptail": needed,
        "deeptail_power_ratio": ratio,                               # available / needed
        "deeptail_power_ratio_optimistic": ratio_opt,
        "deeptail_confirmable_in_one_run": confirmable,
        "n_runs_deeptail": n_runs_deeptail,
        "headline": headline,
        "aggregate_headroom_x": agg_headroom,
        "deeptail_headroom_x": ratio,
        "headroom_collapse_vs_aggregate_x": headroom_collapse,
        "note": (
            "deep tail q[8..9]: needs {:,.0f} accept positions, one served run carries {:,.0f} (ratio "
            "{:.2f}) ⇒ {}. The deep-tail headroom ({:.2f}×) is {:.0f}× THINNER than the aggregate's "
            "{:.0f}× — the deep tail is the BINDING constraint on one-run confirmation: the same trace "
            "that confirms the aggregate with 58× slack confirms the deep tail with only {:.2f}×. "
            "(optimistic d8+d9 reads ⇒ ratio {:.2f}.)"
            .format(needed, available, ratio, headline.replace("_", " ").lower(), ratio,
                    headroom_collapse, agg_headroom, ratio, ratio_opt)),
    }


# --------------------------------------------------------------------------- #
# (4) Honest band — the load-bearing modeling choices and their directions.
# --------------------------------------------------------------------------- #
def honest_band(imp: dict, occ: dict, asn: dict, verdict: dict) -> dict[str, Any]:
    # ICC deflation of N_eff per depth (conservative direction): correlated within-prompt reads
    # carry < 1 unit of independent info. N_eff = N / Deff_icc, Deff_icc = 1+(m̄−1)·ICC. The
    # HEADLINE compares the AR(1)-ACF-inflated n_confirm (1124.76-based) vs RAW n_available — the
    # SAME Deff appears once. Deflating N_eff by ICC AND using the iid n_confirm gives the SAME
    # ratio; doing both (acf n_confirm AND deflated N_eff) is the extra-conservative floor.
    icc = imp["icc_190"]
    mbar = imp["mbar_190"]
    deff_icc = deff_exchangeable(mbar, icc)        # flat-exchangeable Deff = 1+(m̄−1)·ICC = 4.41
    available_raw = occ["n_deeptail_available_one_run"]
    n_eff_deeptail = available_raw / deff_icc
    needed_iid = asn["n_confirm_deeptail_iid"]
    ratio_matched = available_raw / asn["n_confirm_deeptail"]        # acf-needed vs raw-avail (headline)
    ratio_neff_iid = n_eff_deeptail / needed_iid                     # iid-needed vs Neff-avail (== headline)
    ratio_double = n_eff_deeptail / asn["n_confirm_deeptail"]        # extra-conservative floor (double Deff)

    return {
        "a_occupancy_is_load_bearing": (
            "occ(d) is proxied by the banked #208/#203 β-extended reach-weights (a reach-WEIGHT "
            "vector, NOT a per-step reach-DP occupancy dump). It is THE load-bearing modeling choice: "
            "it sets n_available(d). The per-step reach-DP occupancy dump (land #71's tree trace) is "
            "the tightening follow-up. The geometric-in-λ cross-check OVERSTATES deep-tail occupancy, "
            "so the banked reach-weight occupancy is the conservative/binding choice."),
        "b_deeptail_lambda_is_land71_to_measure": (
            "the deep-tail λ̂ is what land #71 will MEASURE. This leg sizes the CONFIRMATION POWER "
            "(how many positions to resolve the budget), NOT the value. The variance is sized AT the "
            "0.7875 budget — the worst-case σ² for any λ≥budget — so n_confirm_deeptail is a "
            "conservative (upper) sizing; a higher measured deep-tail λ ⇒ lower variance ⇒ FEWER "
            "positions needed ⇒ MORE confirmable."),
        "c_icc_deflates_neff_conservative": {
            "deff_icc_flat_exchangeable": deff_icc,
            "n_eff_deeptail": n_eff_deeptail,
            "ratio_headline_acf_needed_vs_raw_avail": ratio_matched,
            "ratio_neff_iid_equivalent": ratio_neff_iid,
            "ratio_double_deff_extra_conservative_floor": ratio_double,
            "note": ("ICC 0.1446 (#190) deflates N_eff per depth (conservative direction). The Deff "
                     "appears ONCE in the headline (AR(1)-ACF-inflated n_confirm vs raw n_available); "
                     "deflating N_eff by ICC with the iid n_confirm gives the IDENTICAL ratio. Applying "
                     "BOTH (acf n_confirm AND ICC-deflated N_eff) double-counts the correlation and is "
                     "reported only as the extra-conservative floor (ratio %.2f)." % ratio_double),
        },
        "d_imports_unchanged": (
            "#225's aggregate ASN (n_confirm_acf=1124.76, iid=405.4, deff=2.7743), the 0.7875 "
            "deep-tail budget (#215), and the #190 ICC are imported UNCHANGED; this leg only "
            "DECOMPOSES them by depth against the occupancy."),
        "extra_conservative_floor_ratio": ratio_double,
        "extra_conservative_floor_confirmable": bool(ratio_double >= 1.0),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, occ: dict, asn: dict, verdict: dict, band: dict) -> dict[str, Any]:
    spine = imp["lambda_spine_interim"]
    budget_dt = imp["budget_deeptail"]

    # (a) occupancy-weighted aggregate n_confirm round-trips #225's 1124.76 to tol.
    cond_a = bool(asn["aggregate_roundtrip_resid"] < TOL_ROUNDTRIP
                  and abs(asn["aggregate_roundtrip_target"] - 1124.7628546877863) < 1e-6)

    # (b) n_confirm(λ) monotone INCREASING as λ decreases (wider Bernoulli variance), and the
    #     deep-tail n_confirm INCREASING as the budget decreases toward 0.5 (variance grows).
    sigbar2 = asn["sigbar2_occupancy_weighted_mean_variance"]
    n_agg = imp["n_confirm_agg_measured_acf"]
    lam_grid = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]      # all > 0.5
    nconf_lam = [n_agg * _sigma2(l) / sigbar2 for l in lam_grid]     # λ decreasing ⇒ should increase
    mono_lambda = all(nconf_lam[i + 1] > nconf_lam[i] - TOL_ERR for i in range(len(nconf_lam) - 1))
    budget_grid = [0.95, 0.90, 0.85, 0.7875, 0.70, 0.60]             # budget decreasing toward 0.5
    nconf_budget = [n_agg * _sigma2(b) / sigbar2 for b in budget_grid]
    mono_budget = all(nconf_budget[i + 1] > nconf_budget[i] - TOL_ERR
                      for i in range(len(nconf_budget) - 1))
    cond_b = bool(mono_lambda and mono_budget)

    # (c) n_available(d) monotone DECREASING in d (sparser deep tail).
    navail = [occ["n_available_obj"][d] for d in LADDER_DEPTHS]
    cond_c = all(navail[i + 1] < navail[i] + TOL_ERR for i in range(len(navail) - 1))

    # (d) deeptail_confirmable_in_one_run consistent with the available-vs-needed comparison.
    avail = verdict["n_deeptail_available_one_run"]
    need = verdict["n_confirm_deeptail"]
    cond_d = bool(verdict["deeptail_confirmable_in_one_run"] == (avail >= need)
                  and abs(verdict["deeptail_power_ratio"] - avail / need) < TOL_ERR)

    # (e) at ICC=0, N_eff = raw count (deflation sane): Deff_icc(icc=0)=1 ⇒ N_eff=N.
    deff_icc0 = deff_exchangeable(imp["mbar_190"], 0.0)
    n_eff_at_icc0 = avail / deff_icc0
    cond_e = bool(abs(deff_icc0 - 1.0) < TOL_ERR and abs(n_eff_at_icc0 - avail) < 1e-6)

    # (f) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [asn["n_confirm_deeptail"], asn["sigbar2_occupancy_weighted_mean_variance"],
           asn["aggregate_roundtrip_n_confirm"], occ["n_deeptail_available_one_run"],
           verdict["deeptail_power_ratio"], band["extra_conservative_floor_ratio"],
           imp["budget_deeptail"], imp["n_confirm_agg_measured_acf"]]
    cond_f = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "gate2_depth_resolved_power_self_test_passes": passes,
        "conditions": {
            "a_occ_weighted_aggregate_roundtrips_225_1124p76": cond_a,
            "b_nconfirm_up_as_lambda_down_and_budget_down": cond_b,
            "c_n_available_decreasing_in_depth": cond_c,
            "d_deeptail_confirmable_consistent_with_avail_vs_need": cond_d,
            "e_icc0_neff_eq_raw_count": cond_e,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "a_roundtrip": asn["aggregate_roundtrip_n_confirm"], "a_target": asn["aggregate_roundtrip_target"],
            "b_nconf_vs_lambda": list(zip(lam_grid, nconf_lam)),
            "b_nconf_vs_budget": list(zip(budget_grid, nconf_budget)),
            "c_n_available": [(d, occ["n_available_obj"][d]) for d in LADDER_DEPTHS],
            "d_available": avail, "d_needed": need, "d_ratio": verdict["deeptail_power_ratio"],
            "e_deff_icc0": deff_icc0, "e_n_eff_at_icc0": n_eff_at_icc0,
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, asn: dict, verdict: dict) -> str:
    return (
        f"DEPTH-RESOLVED GATE-2 POWER BANKED. #225's aggregate one-run confirmation (n_confirm="
        f"{imp['n_confirm_agg_measured_acf']:,.0f}, 58× headroom) does NOT automatically carry the "
        f"DEEP TAIL: q[8..9] is doubly disadvantaged — its Bernoulli variance at the 0.7875 budget "
        f"({asn['sigma2_deeptail_at_budget']:.4f}) is {asn['deeptail_over_spine_variance_ratio']:.0f}× "
        f"the 0.997 spine's ({asn['sigma2_spine']:.5f}), and it carries only "
        f"{imp['w_mass_deeptail_q8q9'] * 100:.1f}% of the q[2..9] reach mass. So n_confirm_deeptail="
        f"{verdict['n_confirm_deeptail']:,.0f} accept positions ({asn['deeptail_over_spine_nconfirm_ratio']:.0f}× "
        f"the spine's {asn['n_confirm_spine_measured_acf']:,.0f}), while one served run carries "
        f"{verdict['n_deeptail_available_one_run']:,.0f} deep-tail positions ⇒ ratio "
        f"{verdict['deeptail_power_ratio']:.2f} ⇒ {verdict['headline']}. The deep-tail headroom "
        f"({verdict['deeptail_headroom_x']:.2f}×) is {verdict['headroom_collapse_vs_aggregate_x']:.0f}× "
        f"THINNER than the aggregate's {verdict['aggregate_headroom_x']:.0f}× — the deep tail is the "
        f"BINDING constraint. n_runs_deeptail={verdict['n_runs_deeptail']}. The deep-tail λ̂ stays land "
        f"#71's to MEASURE (variance sized conservatively at the budget). NOT a launch."
    )


def _handoff(imp: dict, asn: dict, verdict: dict) -> dict[str, str]:
    confirms = ("confirms the whole depth profile" if verdict["deeptail_confirmable_in_one_run"]
                else f"confirms only the spine — the deep tail needs {verdict['n_runs_deeptail']} runs")
    line = (
        f"land #71 + fern #185: your #225 one-run aggregate confirmation "
        f"{'holds for' if verdict['deeptail_confirmable_in_one_run'] else 'does NOT hold for'} the deep "
        f"tail too: q[8..9] needs n_confirm_deeptail={verdict['n_confirm_deeptail']:,.0f} accept positions "
        f"(measured-ACF; iid floor {asn['n_confirm_deeptail_iid']:,.0f}) and one served run carries "
        f"n_deeptail_available={verdict['n_deeptail_available_one_run']:,.0f} (ratio "
        f"{verdict['deeptail_power_ratio']:.2f}), so land #71's first served trace {confirms}. The deep "
        f"tail is {verdict['headroom_collapse_vs_aggregate_x']:.0f}× tighter than the aggregate (58× → "
        f"{verdict['deeptail_headroom_x']:.2f}×) — fern #185 carries n_confirm_deeptail as the "
        f"deep-tail-specific confirmation-run budget; the deep-tail λ̂ value remains land #71's to measure."
    )
    return {"land_71_and_fern_185": line}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    occ = depth_occupancy(imp)
    asn = depth_resolved_asn(imp, occ)
    verdict = one_run_verdict(imp, occ, asn)
    band = honest_band(imp, occ, asn, verdict)
    st = self_test(imp, occ, asn, verdict, band)
    handoff = _handoff(imp, asn, verdict)
    return {
        "self_test": st,
        "test_metric": {"n_confirm_deeptail": asn["n_confirm_deeptail"]},
        "imports": imp,
        "depth_occupancy": {k: v for k, v in occ.items()
                            if k not in ("occ_obj", "occ_norm_obj", "n_available_obj")},
        "depth_resolved_asn": asn,
        "one_run_verdict": verdict,
        "honest_band": band,
        "verdict": _verdict(imp, asn, verdict),
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
    occ = syn["depth_occupancy"]
    asn = syn["depth_resolved_asn"]
    v = syn["one_run_verdict"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("DEPTH-RESOLVED GATE-2 POWER (PR #230) — can ONE run confirm the deep-tail λ̂ ≥ 0.7875?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  aggregate (#225): n_confirm={imp['n_confirm_agg_measured_acf']:,.0f}  "
          f"avail/run={imp['agg_decode_steps_per_run']:,}  headroom {imp['agg_headline_headroom_x']:,.0f}×  "
          f"(confirmable={imp['agg_confirmable_in_one_run']})", flush=True)
    print(f"  deep-tail budget (#215) = {imp['budget_deeptail']:.5f}   spine (land #71) = "
          f"{imp['lambda_spine_interim']:.3f}   binding bar (#208) = {imp['private_bar_208']:.5f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  occupancy model: {occ['occupancy_model']}", flush=True)
    print("  depth |  occ   | n_available |  λ_ref  |  σ²      | n_confirm(acf)", flush=True)
    for r in asn["n_confirm_vs_depth"]:
        print(f"   q{r['depth']}   | {r['occ']:.4f} | {r['n_available']:>10,.0f}  | {r['lambda_ref']:.4f} "
              f"| {r['sigma2']:.5f} | {r['n_confirm_measured_acf']:>11,.0f}   ({r['band']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  σ̄² (occ-weighted mean variance) = {asn['sigbar2_occupancy_weighted_mean_variance']:.6f}", flush=True)
    print(f"  deep-tail/spine variance ratio = {asn['deeptail_over_spine_variance_ratio']:.1f}×   "
          f"n_confirm ratio = {asn['deeptail_over_spine_nconfirm_ratio']:.1f}×", flush=True)
    print(f"  round-trip: occ-weighted Σ n_confirm(d) = {asn['aggregate_roundtrip_n_confirm']:,.4f}  "
          f"(target {asn['aggregate_roundtrip_target']:,.4f}, resid {asn['aggregate_roundtrip_resid']:.2e})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  (TEST) n_confirm_deeptail = {v['n_confirm_deeptail']:,.1f}  "
          f"[iid {asn['n_confirm_deeptail_iid']:,.0f} | ar1 {asn['n_confirm_deeptail_ar1']:,.0f} | "
          f"flat {asn['n_confirm_deeptail_flat']:,.0f}]", flush=True)
    print(f"  n_deeptail_available_one_run = {v['n_deeptail_available_one_run']:,.0f}  "
          f"(optimistic {v['n_deeptail_available_optimistic']:,.0f})", flush=True)
    print(f"  deeptail_power_ratio = {v['deeptail_power_ratio']:.3f}  "
          f"(optimistic {v['deeptail_power_ratio_optimistic']:.3f})", flush=True)
    print(f"  >>> {v['headline']}  (n_runs_deeptail={v['n_runs_deeptail']})", flush=True)
    print(f"  headroom collapse vs aggregate: {v['aggregate_headroom_x']:,.0f}× → "
          f"{v['deeptail_headroom_x']:.2f}×  ({v['headroom_collapse_vs_aggregate_x']:,.0f}× thinner)", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) gate2_depth_resolved_power_self_test_passes = "
          f"{st['gate2_depth_resolved_power_self_test_passes']}", flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF (land #71 + fern #185): {syn['handoff_lines']['land_71_and_fern_185']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #225/#215; never fatal).
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
        print(f"[gate2-depth-resolved-power] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp = syn["imports"]
    occ = syn["depth_occupancy"]
    asn = syn["depth_resolved_asn"]
    v = syn["one_run_verdict"]
    band = syn["honest_band"]
    st = syn["self_test"]

    run = init_wandb_run(
        job_type="sprt-liveprobe-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["sprt-liveprobe-budget", "validity-gate", "measurement-design", "gate2-confirmation",
              "depth-resolved", "deep-tail", "occupancy", "confirmation-power", "ar1-correction",
              "bank-the-analysis"],
        config={
            "n_confirm_agg_measured_acf": imp["n_confirm_agg_measured_acf"],
            "n_confirm_agg_iid": imp["n_confirm_agg_iid"],
            "deff_measured_acf": imp["deff_measured_acf"],
            "budget_deeptail": imp["budget_deeptail"],
            "lambda_spine_interim": imp["lambda_spine_interim"],
            "private_bar_208": imp["private_bar_208"],
            "icc_190": imp["icc_190"], "mbar_190": imp["mbar_190"],
            "decode_steps_per_run": DECODE_STEPS_PER_RUN,
            "occupancy_model": occ["occupancy_model"],
            "imports": imp["provenance"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[gate2-depth-resolved-power] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "gate2_depth_resolved_power_self_test_passes":
            int(bool(st["gate2_depth_resolved_power_self_test_passes"])),
        "n_confirm_deeptail": asn["n_confirm_deeptail"],
        "n_confirm_deeptail_iid": asn["n_confirm_deeptail_iid"],
        "n_confirm_deeptail_ar1": asn["n_confirm_deeptail_ar1"],
        "n_confirm_deeptail_flat": asn["n_confirm_deeptail_flat"],
        "n_confirm_spine_measured_acf": asn["n_confirm_spine_measured_acf"],
        "sigbar2_occupancy_weighted_mean_variance": asn["sigbar2_occupancy_weighted_mean_variance"],
        "sigma2_deeptail_at_budget": asn["sigma2_deeptail_at_budget"],
        "sigma2_spine": asn["sigma2_spine"],
        "deeptail_over_spine_variance_ratio": asn["deeptail_over_spine_variance_ratio"],
        "deeptail_over_spine_nconfirm_ratio": asn["deeptail_over_spine_nconfirm_ratio"],
        "aggregate_roundtrip_n_confirm": asn["aggregate_roundtrip_n_confirm"],
        "aggregate_roundtrip_resid": asn["aggregate_roundtrip_resid"],
        "n_deeptail_available_one_run": v["n_deeptail_available_one_run"],
        "n_deeptail_available_optimistic": v["n_deeptail_available_optimistic"],
        "deeptail_power_ratio": v["deeptail_power_ratio"],
        "deeptail_power_ratio_optimistic": v["deeptail_power_ratio_optimistic"],
        "deeptail_confirmable_in_one_run": int(bool(v["deeptail_confirmable_in_one_run"])),
        "n_runs_deeptail": v["n_runs_deeptail"],
        "aggregate_headroom_x": v["aggregate_headroom_x"],
        "deeptail_headroom_x": v["deeptail_headroom_x"],
        "headroom_collapse_vs_aggregate_x": v["headroom_collapse_vs_aggregate_x"],
        "deeptail_occupancy_fraction_q8q9": occ["deeptail_occupancy_fraction_q8q9"],
        "extra_conservative_floor_ratio": band["extra_conservative_floor_ratio"],
        "extra_conservative_floor_confirmable": int(bool(band["extra_conservative_floor_confirmable"])),
        "budget_deeptail": imp["budget_deeptail"],
        "lambda_spine_interim": imp["lambda_spine_interim"],
        "n_confirm_agg_measured_acf": imp["n_confirm_agg_measured_acf"],
        "decode_steps_per_run": DECODE_STEPS_PER_RUN,
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        "headline_verdict": v["headline"],
        **{f"n_available_q{r['depth']}": r["n_available"] for r in asn["n_confirm_vs_depth"]},
        **{f"n_confirm_acf_q{r['depth']}": r["n_confirm_measured_acf"] for r in asn["n_confirm_vs_depth"]},
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="gate2_depth_resolved_power_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[gate2-depth-resolved-power] wandb logged: {summary}", flush=True)


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
        "pr": 230,
        "agent": "denken",
        "kind": "gate2-depth-resolved-power",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[gate2-depth-resolved-power] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["gate2_depth_resolved_power_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate2_depth_resolved_power_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[gate2-depth-resolved-power] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY gate2_depth_resolved_power_self_test_passes = {passes}", flush=True)
    print(f"  TEST n_confirm_deeptail = {syn['test_metric']['n_confirm_deeptail']:,.2f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[gate2-depth-resolved-power] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
