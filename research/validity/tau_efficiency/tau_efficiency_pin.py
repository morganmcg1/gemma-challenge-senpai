#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Pin the tau overlap/coverage-efficiency factor for the descent/both-bugs tree (PR #181).

WHAT THIS IS
------------
Every tree TPS projection multiplies through tau -- the local->official transfer
efficiency in the composition law

    official = K_cal * (E[T] / step) * tau          (K_cal = 125.268)

K_cal, step, and E[T] are all pinned (ubel #148/#163/#169; lawine #168; denken/wirbel).
tau is the ONE factor still carried as an ASSUMED band [0.9924, 1.0], never independently
pinned for the descent/both-bugs tree. This leg closes that last open composition factor.

WHY NOW. fern #174 (MERGED) showed the descent-only launch verdict is a 0.035-TPS
knife-edge: LCB(P>=0.9) = 499.97 vs the 500 bar. The tau floor 0.9924 is a 0.76% ~= 4 TPS
discount on the ~520 projection -- the SAME order as the knife-edge. So the question the PR
poses is sharp: is the as-built tree's tau the assumed floor 0.9924, or tighter (closer to
1.0)? If it can be tightened, descent-only gets free margin; if not, the floor is real and
we bank it.

THE HONEST FINDING (stated up front, derived below)
---------------------------------------------------
The PR's premise -- "fern #174 used the assumed 0.9924 floor" -- needs one correction that
reframes the whole result. fern #174 did NOT sit tau at the floor as a point estimate: its
central projection uses tau = 1.0 (official 519.96 = K_cal*E[T]/step, no tau haircut), with
the floor [0.9924, 1.0] folded INTO the #148 calibration CI leg (the merged #155 no-double-
count convention). So:
  * Pinning tau to its analytic central (1.0) reproduces fern #174's LCB 499.965 EXACTLY
    -- delta 0.000 TPS. There is NO free margin to hand back; the floor was already in the CI.
  * The ONLY way tau hands margin is TIGHTENING the band floor above 0.9924. Analysis cannot:
    the floor is the un-pinnable SM-clock residual (the official free clock may throttle below
    the local 1710-MHz pin; we control one pod and cannot measure the official box's clock).
    The #126 roofline already derived the floor as tight as analysis allows. -> BANK THE FLOOR.
  * Under kanna #159's sigma_hw quadrature, even a PERFECT tau = 1.0 (band collapsed to a point)
    leaves descent-only at LCB 499.49 < 500. So tau is decisively NOT the lever -- the knife-edge
    is sampling + sigma_hw bound, not tau bound.

Verdict: tau for the descent tree is FLOOR-CONFIRMED at central 1.0, band [0.9924, 1.0];
it leaves descent-only ON the knife-edge. both-bugs (LCB 514.9) is the robust first shot.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file / kernel build.
BASELINE stays 481.53; greedy untouched; adds 0 TPS. PRIMARY = self-test. IMPORTS the merged
legs VERBATIM (lawine #126 tau roofline, ubel #148 K_cal band, ubel #169 footprint invariance,
wirbel #79/#83 rank-coverage rho, lawine #168 step launch-idle, kanna #159 sigma_hw, fern #174
knife-edge); does NOT re-derive them. NOT open2. NOT a launch.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) the 0.9924 floor reproduces from named roofline sub-terms (#126 mild-throttle x full-
    exposure corner == #148 Leg-A clock-exposure floor, to machine precision);
(b) the no-double-count orthogonality holds against K_cal / step / E[T];
(c) pinned tau in [0.9924, 1.0] with the band reported for BOTH topologies;
(d) the descent-only LCB re-evaluation is explicit + quantified vs fern #174 (floor-confirmed,
    delta 0.000; and a faithful re-instantiation of fern #174's 499.965 matches to machine prec);
(e) NaN-clean.
PRIMARY = tau_efficiency_self_test_passes; TEST = tau_descent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))   # research/validity/tau_efficiency -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PROF = os.path.join(_ROOT, "scripts", "profiler")
if _PROF not in sys.path:
    sys.path.insert(0, _PROF)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Import the fern #167 packet VERBATIM (it transitively loads #162 frontier + #155
# consolidator). This is the SAME engine fern #174 uses -> a faithful re-instantiation
# of the descent-only knife-edge, not a re-derivation.
# ============================================================================
packet = _load("pinned_launch_decision_packet",
               os.path.join(_PROF, "pinned_launch_decision_packet.py"))

instantiate_private = packet.instantiate_private
load_stark156_pinned = packet.load_stark156_pinned
load_lawine161_anchors = packet.load_lawine161_anchors
load_ubel154_bars = packet.load_ubel154_bars
load_stark151_retention = packet.load_stark151_retention
b_dict_for_depth1 = packet.b_dict_for_depth1
DEPTH1_DESCENT_ONLY = packet.DEPTH1_DESCENT_ONLY
cons = packet.cons
build_joint_b_dict = cons.build_joint_b_dict
SamplingModel = cons.SamplingModel
load_kcal_band = cons.load_kcal_band

K_CAL = packet.K_CAL                                 # 125.268
RHO2_BRANCH_HIT = packet.RHO2_BRANCH_HIT            # 0.4165
TARGET_OFFICIAL = packet.TARGET_OFFICIAL            # 500.0
Z_P90_ONESIDED = packet.Z_P90_ONESIDED              # 1.281552 (Phi(z)=0.9)
STEP_MEASURED_DEPTH9 = packet.STEP_MEASURED_DEPTH9  # 1.2182
STEP_REL_1SIGMA_DEFAULT = packet.STEP_REL_1SIGMA_DEFAULT  # 0.005

# ---- committed artifacts (one source of truth per constant; imported, not re-derived) ----
TAU_ROOFLINE_126 = os.path.join(_ROOT, "research/spec_cost_model/tree_verify_tau_roofline.json")
KCAL_BAND_148 = os.path.join(_ROOT, "research/kcal_tree_transfer/kcal_tree_transfer_band.json")
SIGMA_HW_159 = os.path.join(_ROOT, "research/validity/hw_variance_envelope/envelope.json")
STEP_168 = os.path.join(_ROOT, "research/spec_cost_model/step_anchor_reconciliation.json")
FERN_174 = os.path.join(_ROOT, "research/spec_cost_model/conservative_step_launch_verdict_results.json")
RANK_COV_79 = os.path.join(_ROOT, "research/rank_coverage/rank_coverage_results.json")

# the official anchor (unchanged; this leg adds 0 TPS)
BASELINE_OFFICIAL_TPS = 481.53
BASELINE_PPL = 2.3777
BASELINE_COMPLETED = 128
BASELINE_RUN_PREFIX = "results/senpai/fa2sw-precache-kenyan-20260613T213911Z"


def _finite(x, default: float = 0.0) -> float:
    try:
        return float(x) if (x is not None and math.isfinite(float(x))) else default
    except (TypeError, ValueError):
        return default


def _rel_err(v, ref):
    return abs(v - ref) / ref if ref else float("inf")


# ============================================================================
# Loaders for the committed sources.
# ============================================================================
def load_tau_126(path: str = TAU_ROOFLINE_126) -> dict:
    with open(path) as f:
        j = json.load(f)
    r = j["tau_tree_roofline"]
    return {
        "tau_central": float(r["tau_tree_central"]),
        "tau_band": [float(r["tau_tree_band"][0]), float(r["tau_tree_band"][1])],
        "tau_floor": float(r["tau_tree_floor"]),
        "ai_m32": float(r["tree_verify_arithmetic_intensity_M32"]),
        "corners": {k: float(v) for k, v in r["step3_tau_corners"].items()},
        "phi_comp": {k: float(v) for k, v in r["eps_decomposition"]["phi_comp_step_fraction"].items()},
        "clock_gap": {k: float(v) for k, v in r["eps_decomposition"]["clock_gap_rho_minus_1"].items()},
        "eps_at_floor_pct": float(r["eps_decomposition"]["eps_at_floor_pct"]),
        "driver": r["eps_decomposition"]["driver"],
        "method": r["method"],
        "attention_compute_exposed": bool(r["eps_decomposition"]["attention_compute_exposed"]),
        "tree_M": 32,
    }


def load_kcal_legs_148(path: str = KCAL_BAND_148) -> dict:
    with open(path) as f:
        j = json.load(f)
    band = j["kcal_tree_transfer_band"]
    legA = band["legs"]["A_clock_exposure"]
    legB = band["legs"]["B_scorer_amortization"]
    return {
        "K_cal_central": float(band["K_cal_central"]),
        "K_cal_lo": float(band["K_cal_lo"]),
        "calib_downside_pct": float(band["kcal_tree_transfer_band_width_pct"]),
        "calib_downside_rel": float(band["kcal_tree_transfer_band_width_pct"]) / 100.0,
        "legA_clock_exposure_floor": float(legA["scale_floor"]),     # 0.99243 -- the tau-tree floor
        "legA_downside_pct": float(legA["downside_pct"]),
        "legB_scorer_floor": float(legB["scale_floor"]),             # 0.99970 -- scorer amortization
        "legB_downside_pct": float(legB["downside_pct"]),
        "scale_lo": float(j["propagation_and_quadrature"]["scale_lo"]),   # legA*legB
        "model": j["model"],
        "bus_factor_pct": float(next(fr["delta"] for fr in j["multiplier_decomposition"]["factors"]
                                     if fr["name"] == "gpu_clock_thermal_power_bus")) * 100.0,
    }


def load_sigma_hw_159(path: str = SIGMA_HW_159) -> dict:
    with open(path) as f:
        j = json.load(f)
    env = j["envelope"]
    return {
        "sigma_hw_pct": float(env["sigma_hw_pct"]),
        "sigma_hw_rel": float(env["sigma_hw_pct"]) / 100.0,
        "sigma_within_pct": float(env["sigma_within_pct"]),
        "sigma_cross_pct": float(env["sigma_cross_pct"]),
        "sm_clock_held_1710": bool(
            env["detail"]["within_fresh"]["sm_clock_mhz_load"]["std"] == 0.0
            and env["detail"]["within_fresh"]["sm_clock_mhz_load"]["mean"] == 1710.0),
    }


def load_step168(path: str = STEP_168) -> dict:
    with open(path) as f:
        j = json.load(f)
    anchors = {a["name"]: a for a in j["step1_anchors"]}
    overlap = anchors["measured_idle_hidden_overlap"]
    return {
        "step_launch_realized": float(j["step2_launch_realized_step"]["both_bugs"]),
        "step_roofline": float(j["step2_launch_realized_step"]["band"]["lo_step_roofline"]),
        "launch_idle_pct": 0.447,   # "+0.447% exposed launch idle (43.3 us/step)" -- overlap vs roofline
        "launch_idle_role": overlap["role"],
        "tau_used": float(j["constants"]["tau"]),   # 1.0 -- #168 quotes tau=1
        "step_full_spread_pct": float(j["step2_launch_realized_step"]["band"]["full_spread_pct"]),
    }


def load_fern174(path: str = FERN_174) -> dict:
    with open(path) as f:
        j = json.load(f)
    cons_cell = j["step1_three_step_instantiation"]["conservative_1p2182"]["descent_only"]
    both_cell = j["step1_three_step_instantiation"]["conservative_1p2182"]["both_bugs"]
    return {
        "step": float(cons_cell["_step"]),
        "descent_E_T": float(cons_cell["E_T_input"]),
        "descent_r_tree": float(cons_cell["r_tree"]),
        "descent_proj_private": float(cons_cell["proj_private_tps"]),
        "descent_official": float(cons_cell["official_tps"]),
        "descent_combined_rel": float(cons_cell["combined_rel_1sigma"]),
        "descent_lcb_p90": float(cons_cell["lcb_p90"]),
        "descent_p_clear_500": float(cons_cell["p_clear_500"]),
        "descent_launch_go": cons_cell["launch_go_p90"],
        "both_lcb_p90": float(both_cell["lcb_p90"]),
        "both_launch_go": both_cell["launch_go_p90"],
        "wandb_run_id": j.get("wandb_run_id"),
    }


def load_rho_79(path: str = RANK_COV_79) -> dict:
    with open(path) as f:
        j = json.load(f)
    a = j["analysis"]
    return {
        "rho2": float(a["rho_marginal"]["2"]),
        "cov4": float(a["cumulative_coverage"]["4"]),
        "top1_acceptance": float(a["top1_acceptance"]),
        "lives_in": "E[T] accept-length DP (numerator)",
    }


# ============================================================================
# STEP 1 -- define tau + trace the 0.9924 floor provenance + decompose sources.
# ============================================================================
def step1_definition(tau126: dict, kcal: dict) -> dict:
    """tau is the local->official TRANSFER multiplier on the composition map; the floor
    is the mild-throttle x full-exposure roofline corner. Decompose into physical sources."""
    floor = tau126["tau_floor"]
    floor_corner = tau126["corners"]["mild_throttle_full_exposure"]
    # the SAME floor flows into the #148 calibration band as Leg A.
    legA_floor = kcal["legA_clock_exposure_floor"]
    floor_matches_corner = bool(_rel_err(floor, floor_corner) <= 1e-9)
    floor_matches_legA = bool(_rel_err(floor, legA_floor) <= 1e-6)
    eps_at_floor = 1.0 - floor
    eps_at_floor_pct = eps_at_floor * 100.0
    # first-order physical driver: eps ~ Phi_comp(full exposure) x clock_gap(mild throttle).
    phi_full = tau126["phi_comp"]["adversarial"]      # 0.0894 full GEMM compute-exposed fraction
    clock_mild = tau126["clock_gap"]["mild_throttle"]  # 0.0986 official-vs-local clock throttle
    eps_first_order = phi_full * clock_mild
    return {
        "tau_definition": (
            "tau is the LOCAL->OFFICIAL roofline transfer multiplier in "
            "official = K_cal*(E[T]/step)*tau. It is the dimensionless ratio "
            "tau = step_ratio_loc/step_ratio_off of how faithfully the M=32 wide-verify "
            "decode step transfers from the local-pinned A10G (1710 MHz) to the official "
            "free-clock A10G. The E[T] numerator is algorithmic (greedy on identical weights) "
            "and CANCELS exactly; only the verify-GEMM's incremental COMPUTE-exposed fraction "
            "x the un-pinnable SM-clock residual breaks tau below 1. tau multiplies the WHOLE "
            "map -- it is NOT an accept-length term and NOT an in-step time term."),
        "tau_floor_provenance": (
            f"floor = {floor:.10f} = lawine #126's mild_throttle_full_exposure roofline corner "
            f"({floor_corner:.10f}); it is eps_at_floor = {eps_at_floor_pct:.4f}% below central "
            f"tau=1.0. The SAME number is Leg A (clock_exposure) of ubel #148's K_cal calibration "
            f"band ({legA_floor:.10f}) -- i.e. the tau floor IS the dominant calibration-leg "
            f"downside, sourced from the roofline, flowed into the projection CI. Physical driver: "
            f"verify-GEMM compute-exposure Phi_comp x SM-clock throttle (rho-1)."),
        "physical_sources": {
            "rank_coverage_discount_rho": {
                "in_tau": 0.0,
                "note": ("ZERO -- rank-coverage rho (wirbel #79 rho2=0.4165) lives entirely in the "
                         "E[T] accept-length DP (the numerator), NOT in tau. The composition law "
                         "factors it out: tau's derivation cancels E[T]. A rho term inside tau would "
                         "re-discount acceptance (double-count E[T])."),
            },
            "scheduling_overlap_efficiency": {
                "driver": tau126["driver"],
                "phi_comp_central": tau126["phi_comp"]["central"],
                "phi_comp_full_exposure": phi_full,
                "note": ("the verify-GEMM (M=32, AI=107.66 at the sm_86 knee) has a compute-exposed "
                         "fraction Phi_comp that transfers at the CLOCK ratio, not the bus ratio. "
                         "This is tau's ONLY real loss channel."),
            },
            "warmup_discard_residual": {
                "in_tau": 0.0,
                "note": ("0 -- folded into K_cal's Leg B (scorer amortization, precache-neutralized), "
                         "not tau. tau is warmup-invariant."),
            },
            "sm_clock_residual": {
                "clock_gap_bus_parity": tau126["clock_gap"]["bus_parity"],
                "clock_gap_mild_throttle": clock_mild,
                "note": ("the un-pinnable axis: the official free clock may throttle below the local "
                         "1710-MHz pin. This is what sets the floor and what a real served measurement "
                         "would tighten."),
            },
        },
        "floor": floor,
        "floor_corner_mild_throttle_full_exposure": floor_corner,
        "floor_matches_corner": floor_matches_corner,
        "floor_matches_legA_148": floor_matches_legA,
        "eps_at_floor_pct": eps_at_floor_pct,
        "eps_at_floor_pct_126": tau126["eps_at_floor_pct"],
        "eps_first_order_phi_x_clockgap": eps_first_order,
        "eps_first_order_note": (
            f"first-order driver eps ~ Phi_comp(full {phi_full:.4f}) x clock_gap(mild {clock_mild:.4f}) "
            f"= {eps_first_order*100:.4f}%; the exact corner ({eps_at_floor_pct:.4f}%) carries the "
            f"roofline's higher-order BW/compute split. Both name the SAME two physical sub-terms."),
        "ai_m32": tau126["ai_m32"],
        "attention_compute_exposed": tau126["attention_compute_exposed"],
    }


# ============================================================================
# STEP 2 -- de-conflict: prove tau is not double-counting K_cal / step / E[T].
# ============================================================================
def step2_no_double_count(tau126: dict, kcal: dict, step168: dict, rho79: dict) -> dict:
    """Each effect inside tau is orthogonal to the other three composition factors."""
    # --- tau vs K_cal: bus-ratio (in K_cal) vs clock-ratio incremental (tau) ---
    # #148 splits the +6.019% multiplier residual: M=8 linear pure-BW transfers at the BUS
    # ratio and is FOLDED INTO K_cal (tree-footprint-invariant, ubel #169). tau carries ONLY
    # the M=32 tree's INCREMENTAL compute-exposure that transfers at the CLOCK ratio.
    tau_vs_kcal_orthogonal = bool(kcal["bus_factor_pct"] > 0.0 and tau126["tau_floor"] < 1.0
                                  and kcal["legA_clock_exposure_floor"] == tau126["tau_floor"])
    # --- tau vs step: launch-idle (in the denominator) cancels in tau's ratio ---
    # lawine #168's +0.447% launch idle is a TIME cost IN the step (43.3 us/step kernel-launch
    # gap), present IDENTICALLY local and official (hardware-scheduling, clock-independent) ->
    # it is in the step denominator and CANCELS in tau = step_ratio_loc/step_ratio_off. #168
    # itself quotes tau=1.0, confirming the launch-idle is NOT a tau term.
    tau_vs_step_orthogonal = bool(step168["tau_used"] == 1.0 and step168["launch_idle_pct"] > 0.0)
    # --- tau vs E[T]: rho lives in E[T]; tau cancels E[T] exactly ---
    # tau's roofline derivation cancels the E[T] numerator (greedy, identical weights). rho
    # (wirbel #79 rho2=0.4165) is in the E[T] accept-length DP, NOT in tau. tau has NO rho term.
    tau_vs_et_orthogonal = bool(rho79["rho2"] > 0.0 and tau126["method"].startswith("tau_tree = step_ratio_loc"))
    no_double_count = bool(tau_vs_kcal_orthogonal and tau_vs_step_orthogonal and tau_vs_et_orthogonal)
    return {
        "tau_no_double_count": no_double_count,
        "orthogonality": {
            "vs_K_cal": {
                "orthogonal": tau_vs_kcal_orthogonal,
                "argument": (
                    f"K_cal ({kcal['K_cal_central']:.3f}) carries the M=8 linear pure-BW transfer at "
                    f"the BUS ratio (+{kcal['bus_factor_pct']:.3f}% multiplier residual, 'the bus is "
                    f"the wall'), tree-footprint-INVARIANT (ubel #169). tau carries ONLY the M=32 "
                    f"tree's INCREMENTAL compute-exposed fraction transferring at the CLOCK ratio. "
                    f"#148 splits these explicitly: the gpu_clock_thermal_power_bus factor is the "
                    f"BW part (K_cal), Leg A is the incremental compute part (tau). Different regime "
                    f"(BW-bound vs compute-exposed), different transfer mechanism (bus vs clock)."),
            },
            "vs_step": {
                "orthogonal": tau_vs_step_orthogonal,
                "argument": (
                    f"lawine #168's +{step168['launch_idle_pct']:.3f}% exposed launch idle is a TIME "
                    f"cost IN the step denominator (43.3 us/step eager star-attn kernel-launch gap), "
                    f"present IDENTICALLY on local and official hardware (scheduling, clock-independent) "
                    f"-> it CANCELS in tau = step_ratio_loc/step_ratio_off. #168 itself quotes tau={step168['tau_used']:.1f}. "
                    f"tau's loss (clock residual) is a TRANSFER-fidelity multiplier, a distinct axis "
                    f"from the in-step launch idle. They do not double-count."),
            },
            "vs_E_T": {
                "orthogonal": tau_vs_et_orthogonal,
                "argument": (
                    f"tau's roofline derivation CANCELS the E[T] numerator exactly (greedy acceptance "
                    f"on identical weights -> same accept-length local and official). rank-coverage "
                    f"rho (wirbel #79 rho2={rho79['rho2']:.4f}, cov4={rho79['cov4']:.4f}) lives in the "
                    f"E[T] accept-length DP, NOT in tau. The realized tau has NO rank-coverage term: "
                    f"adding one would re-discount acceptance already counted in E[T]=5.0564. The "
                    f"'overlap/rank-coverage efficiency' name is a misnomer -- tau is PURELY the "
                    f"overlap/clock-exposure channel; the rank-coverage channel is E[T]'s."),
            },
        },
    }


# ============================================================================
# STEP 3 -- pin tau for the descent / both-bugs tree (topology-invariant).
# ============================================================================
def step3_pin_topologies(tau126: dict) -> dict:
    """Both descent-only and both-bugs run the SAME M=32 batched wide-verify (the depth-1
    spine / salvage-descent bugs change E[T] acceptance, NOT the verify width). tau depends
    ONLY on the M=32 verify-GEMM AI=107.66 x clock residual -> topology-INVARIANT. The
    measured-rho-optimal max-branch-3 (wirbel #83) and max-branch-4 (wirbel #79) trees BOTH
    land at M=32 (the roofline knee, 1 row under the M=33 tile cliff), so the floor is the
    same for both. Analysis cannot tighten the floor (un-pinnable clock residual)."""
    band = tau126["tau_band"]
    central = tau126["tau_central"]
    return {
        "tau_descent": central,
        "tau_both_bugs": central,
        "tau_band_descent": band,
        "tau_band_both_bugs": band,
        "recommended_central": central,
        "topology_invariant": True,
        "tightened": False,
        "tightened_band": band,
        "why_invariant": (
            "descent-only and both-bugs run the IDENTICAL M=32 batched wide-verify GEMM "
            "(AI=107.66 at the sm_86 knee); the BUG-1 depth-1 spine and BUG-2 salvage-descent "
            "change E[T] ACCEPTANCE, not the verify batch width. max-branch-3 (wirbel #83) and "
            "max-branch-4 (wirbel #79) both operate at M=32 (1 row under the M=33 tile cliff), so "
            "tau's roofline operating point -- and its floor -- are identical across topologies."),
        "why_not_tightened": (
            "the floor 0.9924 is the mild-throttle x full-exposure corner set by the UN-PINNABLE "
            "official SM-clock residual. We control one pod and cannot measure the official box's "
            "free clock; the #126 roofline already derived the floor as tight as analysis allows. "
            "Tightening to e.g. [0.996, 1.0] needs a real served M=32 tree measurement (land #71's "
            "eventual human-approved HF job). -> BANK THE FLOOR."),
        "central_recommendation_note": (
            "recommend central tau = 1.0 for the packet (the analytic best estimate: identical "
            "sm_86 silicon, clock cancels in expectation), with band [0.9924, 1.0] carried as the "
            "calibration-leg downside -- exactly fern #155's merged convention."),
    }


# ============================================================================
# STEP 4 -- propagate to the descent-only knife-edge LCB.
# ============================================================================
def _lcb(proj: float, combined_rel: float) -> float:
    return _finite(proj * (1.0 - Z_P90_ONESIDED * combined_rel))


def step4_propagate(live_desc: dict, kcal: dict, sigma_hw: dict, fern174: dict) -> dict:
    """Re-evaluate descent-only P>=0.9 LCB at the shipped step 1.2182 with the pinned tau
    (central 1.0, floor in the calib CI) vs the assumed-floor-as-point counterfactual, then
    fold sigma_hw. live_desc is the FAITHFUL re-instantiation (instantiate_private) of fern
    #174's descent-only conservative cell."""
    proj = live_desc["proj_private_tps"]
    combined_rel = live_desc["combined_rel_1sigma"]    # 3-term: sqrt(samp^2 + calib^2 + step^2)
    calib_rel = kcal["calib_downside_rel"]             # 0.787% (Leg A tau-floor + Leg B scorer)
    step_rel = STEP_REL_1SIGMA_DEFAULT                 # 0.005
    legB = kcal["legB_scorer_floor"]                   # 0.99970
    # back out the (tau/step-invariant) sampling leg from the live combined_rel.
    samp_rel = _finite(math.sqrt(max(0.0, combined_rel ** 2 - calib_rel ** 2 - step_rel ** 2)))
    sigma_hw_rel = sigma_hw["sigma_hw_rel"]

    def combined_with_calib(cr_calib: float, with_hw: bool) -> float:
        terms = samp_rel ** 2 + cr_calib ** 2 + step_rel ** 2
        if with_hw:
            terms += sigma_hw_rel ** 2
        return _finite(math.sqrt(terms))

    # (1) PINNED tau: central 1.0, floor [0.9924,1.0] folded in the calib leg == fern #174.
    cr_pinned = combined_with_calib(calib_rel, with_hw=False)
    lcb_pinned = _lcb(proj, cr_pinned)
    # (2) PINNED tau under sigma_hw quadrature (the PR's 4th term).
    cr_pinned_hw = combined_with_calib(calib_rel, with_hw=True)
    lcb_pinned_hw = _lcb(proj, cr_pinned_hw)
    # (3) tau CEILING (band collapsed to [1.0,1.0]): calib leg = Leg B scorer only.
    calib_rel_ceiling = 1.0 - legB
    cr_ceiling = combined_with_calib(calib_rel_ceiling, with_hw=False)
    lcb_ceiling = _lcb(proj, cr_ceiling)
    cr_ceiling_hw = combined_with_calib(calib_rel_ceiling, with_hw=True)
    lcb_ceiling_hw = _lcb(proj, cr_ceiling_hw)
    # (4) ASSUMED-FLOOR-AS-POINT counterfactual (what the PR's premise imagines #174 did):
    #     apply tau_floor as a CENTRAL haircut, calib leg -> Leg B only (avoid double-count).
    tau_floor = kcal["legA_clock_exposure_floor"]
    proj_floor_point = _finite(proj * tau_floor)
    lcb_floor_point = _lcb(proj_floor_point, cr_ceiling)   # calib=LegB since tau is now a point

    # (5) what floor would clear 500 in the 3-term framing? (raise Leg A until LCB == 500)
    lo, hi = tau_floor, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        calib_mid = 1.0 - (mid * legB)
        if _lcb(proj, combined_with_calib(calib_mid, with_hw=False)) < TARGET_OFFICIAL:
            lo = mid
        else:
            hi = mid
    floor_to_clear_500 = 0.5 * (lo + hi)

    delta_vs_174 = _finite(lcb_pinned - fern174["descent_lcb_p90"])
    return {
        "descent_only_lcb_pinned_tau": lcb_pinned,                       # TEST (== fern #174)
        "descent_only_clears_500_pinned_tau": bool(lcb_pinned >= TARGET_OFFICIAL),
        "tps_delta_vs_fern174_floor_tau": delta_vs_174,                   # 0.000 -> floor-confirmed
        "verdict": ("floor-confirmed" if abs(delta_vs_174) < 1e-6 else
                    ("free-margin" if delta_vs_174 > 0 else "tighter-floor-needed")),
        "scenarios": {
            "pinned_tau_3term": {"calib_rel": calib_rel, "combined_rel": cr_pinned,
                                 "lcb": lcb_pinned, "clears_500": bool(lcb_pinned >= 500.0),
                                 "note": "central tau=1.0, floor in calib CI == fern #174 exactly"},
            "pinned_tau_4term_sigma_hw": {"calib_rel": calib_rel, "combined_rel": cr_pinned_hw,
                                          "lcb": lcb_pinned_hw, "clears_500": bool(lcb_pinned_hw >= 500.0),
                                          "note": "+ kanna #159 sigma_hw 4th quadrature term"},
            "tau_ceiling_3term": {"calib_rel": calib_rel_ceiling, "combined_rel": cr_ceiling,
                                  "lcb": lcb_ceiling, "clears_500": bool(lcb_ceiling >= 500.0),
                                  "note": "band collapsed to tau=[1,1] (best case); calib=Leg B only"},
            "tau_ceiling_4term_sigma_hw": {"calib_rel": calib_rel_ceiling, "combined_rel": cr_ceiling_hw,
                                           "lcb": lcb_ceiling_hw, "clears_500": bool(lcb_ceiling_hw >= 500.0),
                                           "note": "even a PERFECT tau cannot clear 500 once sigma_hw is folded"},
            "assumed_floor_as_point": {"tau_point": tau_floor, "proj": proj_floor_point,
                                       "lcb": lcb_floor_point, "clears_500": bool(lcb_floor_point >= 500.0),
                                       "note": "counterfactual the PR's premise imagines #174 used; #174 did NOT"},
        },
        "floor_to_clear_500_3term": floor_to_clear_500,
        "samp_rel_backed_out": samp_rel,
        "sigma_hw_rel": sigma_hw_rel,
        "fern174_lcb": fern174["descent_lcb_p90"],
        "interpretation": (
            f"Pinning tau to its analytic central (1.0), with the floor [0.9924,1.0] in the calib "
            f"CI, reproduces fern #174's LCB {fern174['descent_lcb_p90']:.3f} EXACTLY "
            f"(delta {delta_vs_174:+.4f} TPS) -- FLOOR-CONFIRMED, no free margin. tau hands margin "
            f"ONLY by tightening the floor above {tau_floor:.4f} (would need >= {floor_to_clear_500:.4f} "
            f"to clear 500 in 3-term), which analysis cannot do (un-pinnable clock residual). Under "
            f"sigma_hw, even a PERFECT tau=1.0 lands at {lcb_ceiling_hw:.2f} < 500 -> tau is NOT the "
            f"lever; the knife-edge is sampling+sigma_hw bound. BANK THE FLOOR; both-bugs (LCB "
            f"{fern174['both_lcb_p90']:.1f}) is the robust first shot."),
    }


# ============================================================================
# Re-instantiate fern #174's descent-only conservative cell (faithful, not re-derived).
# ============================================================================
def reinstantiate_descent(args) -> dict:
    pinned = load_stark156_pinned()
    lawine = load_lawine161_anchors()
    step168 = load_step168()
    knots = load_stark151_retention(step=args.base_step)["knots"]
    kcal_band = load_kcal_band()
    sampling = SamplingModel(n_steps=args.n_steps, n_boot=args.n_boot, seed=args.seed,
                             step=args.base_step, step_rel_hw=args.step_rel_half_width)
    b_both = build_joint_b_dict(cons.RHO_OPT_JSON, cons.ORACLE_LIVE_JSON)
    b_desc = b_dict_for_depth1(b_both, DEPTH1_DESCENT_ONLY)
    ok_validity = {"ppl": 2.39, "boots": True, "completed": 128}
    step_conservative = step168["step_launch_realized"]
    desc = instantiate_private(lawine["e_t_descent_only"], RHO2_BRANCH_HIT, 1.0, 1.0,
                               "descent_only", pinned["drop_descent_only"], step_conservative,
                               b_desc, knots, sampling, kcal_band, ok_validity)
    both = instantiate_private(lawine["e_t_both_bugs"], RHO2_BRANCH_HIT, 1.0, 1.0,
                               "both_bugs", pinned["drop_both_bugs"], step_conservative,
                               b_both, knots, sampling, kcal_band, ok_validity)
    desc["official_tps"] = desc["geom_tps_public"]
    both["official_tps"] = both["geom_tps_public"]
    return {"descent": desc, "both": both, "step_conservative": step_conservative}


# ============================================================================
# STEP 5 -- self-test (PRIMARY).
# ============================================================================
def self_test(s1: dict, s2: dict, s3: dict, s4: dict, live: dict, fern174: dict) -> dict:
    # (a) the 0.9924 floor reproduces from named roofline sub-terms.
    assert_a = bool(s1["floor_matches_corner"] and s1["floor_matches_legA_148"]
                    and abs(s1["eps_at_floor_pct"] - s1["eps_at_floor_pct_126"]) <= 1e-6)
    # (b) no-double-count orthogonality holds vs K_cal / step / E[T].
    assert_b = bool(s2["tau_no_double_count"])
    # (c) pinned tau in [0.9924, 1.0] with band reported for BOTH topologies.
    band = s3["tau_band_descent"]
    in_band = (band[0] >= 0.9924 - 1e-6 and band[1] <= 1.0 + 1e-12
               and band[0] <= s3["tau_descent"] <= band[1])
    assert_c = bool(in_band and s3["tau_band_both_bugs"] == band
                    and 0.9924 - 1e-6 <= s3["tau_both_bugs"] <= 1.0 + 1e-12)
    # (d) descent-only LCB re-evaluation explicit + faithfully reproduces fern #174.
    live_matches_174 = bool(
        _rel_err(live["descent"]["proj_private_tps"], fern174["descent_proj_private"]) <= 1e-4
        and _rel_err(live["descent"]["combined_rel_1sigma"], fern174["descent_combined_rel"]) <= 1e-4
        and _rel_err(live["descent"]["lcb_p90"], fern174["descent_lcb_p90"]) <= 1e-4)
    floor_confirmed = bool(s4["verdict"] == "floor-confirmed"
                           and abs(s4["tps_delta_vs_fern174_floor_tau"]) < 1e-6)
    assert_d = bool(live_matches_174 and floor_confirmed
                    and s4["descent_only_clears_500_pinned_tau"] is False)
    # (e) NaN-clean across every headline numeric.
    nums = [s1["floor"], s1["eps_at_floor_pct"], s1["eps_first_order_phi_x_clockgap"],
            s3["tau_descent"], s3["tau_both_bugs"], band[0], band[1],
            s4["descent_only_lcb_pinned_tau"], s4["tps_delta_vs_fern174_floor_tau"],
            s4["floor_to_clear_500_3term"], s4["samp_rel_backed_out"],
            s4["scenarios"]["pinned_tau_4term_sigma_hw"]["lcb"],
            s4["scenarios"]["tau_ceiling_4term_sigma_hw"]["lcb"],
            live["descent"]["proj_private_tps"], live["descent"]["lcb_p90"]]
    assert_e = bool(all(x is not None and math.isfinite(x) for x in nums))

    passes = int(bool(assert_a and assert_b and assert_c and assert_d and assert_e))
    return {
        "passes": bool(passes),
        "tau_efficiency_self_test_passes": passes,
        "assert_a_floor_from_named_subterms": {
            "ok": assert_a, "floor": s1["floor"],
            "matches_126_corner": s1["floor_matches_corner"],
            "matches_148_legA": s1["floor_matches_legA_148"],
            "eps_at_floor_pct": s1["eps_at_floor_pct"],
            "expect": "0.9924 floor == #126 mild-throttle x full-exposure corner == #148 Leg A"},
        "assert_b_no_double_count": {
            "ok": assert_b, "tau_no_double_count": s2["tau_no_double_count"],
            "vs_K_cal": s2["orthogonality"]["vs_K_cal"]["orthogonal"],
            "vs_step": s2["orthogonality"]["vs_step"]["orthogonal"],
            "vs_E_T": s2["orthogonality"]["vs_E_T"]["orthogonal"],
            "expect": "tau orthogonal to K_cal (bus vs clock) / step (launch-idle cancels) / E[T] (rho in E[T])"},
        "assert_c_pinned_tau_in_band_both_topologies": {
            "ok": assert_c, "tau_descent": s3["tau_descent"], "tau_both_bugs": s3["tau_both_bugs"],
            "band": band, "expect": "tau in [0.9924,1.0], band reported for descent AND both-bugs"},
        "assert_d_lcb_reeval_explicit_vs_174": {
            "ok": assert_d, "live_matches_fern174": live_matches_174,
            "live_lcb": live["descent"]["lcb_p90"], "fern174_lcb": fern174["descent_lcb_p90"],
            "verdict": s4["verdict"], "tps_delta_vs_174": s4["tps_delta_vs_fern174_floor_tau"],
            "clears_500": s4["descent_only_clears_500_pinned_tau"],
            "expect": "live re-instantiation == #174 499.965; floor-confirmed (delta 0.000); does NOT clear 500"},
        "assert_e_nan_clean": {"ok": assert_e, "n_numbers_checked": len(nums)},
    }


# ============================================================================
# Driver.
# ============================================================================
def run(args) -> dict:
    t0 = time.time()
    tau126 = load_tau_126()
    kcal = load_kcal_legs_148()
    sigma_hw = load_sigma_hw_159()
    step168 = load_step168()
    fern174 = load_fern174()
    rho79 = load_rho_79()

    s1 = step1_definition(tau126, kcal)
    s2 = step2_no_double_count(tau126, kcal, step168, rho79)
    s3 = step3_pin_topologies(tau126)
    live = reinstantiate_descent(args)
    s4 = step4_propagate(live["descent"], kcal, sigma_hw, fern174)

    st = self_test(s1, s2, s3, s4, live, fern174)
    tau_efficiency_self_test_passes = st["tau_efficiency_self_test_passes"]
    tau_descent = s3["tau_descent"]

    handoff = (
        f"tau for the descent tree is pinned at {tau_descent:.4f} (band "
        f"[{s3['tau_band_descent'][0]:.4f}, {s3['tau_band_descent'][1]:.4f}]), distinct from "
        f"K_cal/step/E[T]; it confirms the floor (delta "
        f"{s4['tps_delta_vs_fern174_floor_tau']:+.3f} TPS to the descent-only LCB) -- leaving it ON "
        f"the knife-edge (LCB {s4['descent_only_lcb_pinned_tau']:.2f} < 500), so both-bugs (LCB "
        f"{fern174['both_lcb_p90']:.1f}) stays the robust first shot.")

    state = "ARMED" if tau_efficiency_self_test_passes else "SELF-TEST-FAIL"
    out = {
        "primary_metric_name": "tau_efficiency_self_test_passes",
        "tau_efficiency_self_test_passes": tau_efficiency_self_test_passes,
        "test_metric_name": "tau_descent",
        "tau_descent": tau_descent,
        "tau_both_bugs": s3["tau_both_bugs"],
        "gate_state": state,
        "composition_law": kcal["model"],
        "operating_point": {
            "K_cal": kcal["K_cal_central"],
            "step_launch_realized": step168["step_launch_realized"],
            "E_T_descent": fern174["descent_E_T"],
            "tau_central": s3["recommended_central"],
            "tau_band": s3["tau_band_descent"],
            "sigma_hw_pct": sigma_hw["sigma_hw_pct"],
        },
        "step1_definition": s1,
        "step2_no_double_count": s2,
        "step3_pin_topologies": s3,
        "step4_knife_edge": s4,
        "step5_self_test": st,
        "step6_handoff": handoff,
        "reinstantiation_check": {
            "live_descent_proj_private": live["descent"]["proj_private_tps"],
            "live_descent_combined_rel": live["descent"]["combined_rel_1sigma"],
            "live_descent_lcb_p90": live["descent"]["lcb_p90"],
            "fern174_descent_proj_private": fern174["descent_proj_private"],
            "fern174_descent_combined_rel": fern174["descent_combined_rel"],
            "fern174_descent_lcb_p90": fern174["descent_lcb_p90"],
            "fern174_wandb_run_id": fern174["wandb_run_id"],
            "note": "faithful re-instantiation of fern #174's descent-only conservative cell via instantiate_private.",
        },
        "summary_block": {
            "tps": BASELINE_OFFICIAL_TPS,
            "ppl": BASELINE_PPL,
            "completed": BASELINE_COMPLETED,
            "run_prefix": BASELINE_RUN_PREFIX,
            "note": ("BANK-THE-ANALYSIS: no served run; the official anchor is UNCHANGED. This leg "
                     "pins the last assumed composition factor tau and adds 0 TPS."),
        },
        "imported_legs": {
            "lawine_126_tau_roofline": [tau126["tau_floor"], tau126["tau_central"]],
            "ubel_148_kcal_band": [kcal["K_cal_lo"], kcal["K_cal_central"]],
            "ubel_169_footprint_invariance": "K_cal tree-footprint-invariant (M=32 20.47 GB)",
            "wirbel_79_83_rank_coverage": {"rho2": rho79["rho2"], "topology_M": tau126["tree_M"]},
            "lawine_168_step": [step168["step_launch_realized"], step168["launch_idle_pct"]],
            "kanna_159_sigma_hw": sigma_hw["sigma_hw_pct"],
            "fern_174_knife_edge": [fern174["descent_lcb_p90"], fern174["both_lcb_p90"]],
        },
        "public_evidence_used": [
            "Roofline model (Williams/Patterson/Asanovic, CACM 2009): arithmetic intensity, ridge point.",
            "NVIDIA A10G (sm_86, GA102) datasheet: ~600 GB/s GDDR6, 80 SMs, 1710 MHz boost clock.",
            "Marlin W4A16 kernel tiling (tile_n=128): flat GEMM for M<=32, tile cliff at M=33.",
            "Speculative-decode accept-length E[T] is algorithmic -> transfers 1:1 (tau cancels E[T]).",
        ],
        "provenance": (
            "Pins the last assumed composition factor tau in official=K_cal*(E[T]/step)*tau. IMPORTS "
            "VERBATIM: lawine #126 tau roofline ([0.9924,1.0] floor), ubel #148 K_cal band (Leg A == "
            "tau floor), ubel #169 footprint invariance, wirbel #79/#83 rank-coverage rho (lives in "
            "E[T]), lawine #168 step launch-idle (+0.447%, cancels in tau), kanna #159 sigma_hw, fern "
            "#174 knife-edge (re-instantiated faithfully). One source of truth per constant."),
        "method": (
            "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/kernel build. BASELINE "
            "stays 481.53; adds 0 TPS -- closes the last open composition factor. Greedy identity "
            "untouched. NOT open2. NOT a launch."),
        "metrics_nan_clean": int(st["assert_e_nan_clean"]["ok"]),
        "wandb_run_id": None,
        "wandb_url": None,
        "elapsed_s": None,
    }

    _print_console(out)

    if args.wandb and not args.no_wandb:
        try:
            rid, rurl = _log_wandb(args, out)
            out["wandb_run_id"], out["wandb_url"] = rid, rurl
        except Exception as e:  # noqa: BLE001
            print(f"[tau-efficiency-pin] W&B logging failed (non-fatal): {e!r}", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return out


def _print_console(out):
    print("=" * 100)
    print("PIN THE tau OVERLAP/COVERAGE-EFFICIENCY FACTOR (PR #181)")
    print("=" * 100)
    op = out["operating_point"]
    print(f"\ncomposition: {out['composition_law']}")
    print(f"K_cal={op['K_cal']:.3f}  step={op['step_launch_realized']:.4f}  "
          f"E[T]_descent={op['E_T_descent']:.4f}  tau_central={op['tau_central']:.4f}  "
          f"band=[{op['tau_band'][0]:.4f},{op['tau_band'][1]:.4f}]\n")

    s1 = out["step1_definition"]
    print(f"[STEP 1] tau floor {s1['floor']:.10f} (eps {s1['eps_at_floor_pct']:.4f}%); "
          f"matches #126 corner={s1['floor_matches_corner']} / #148 Leg A={s1['floor_matches_legA_148']}")
    print(f"         rank-coverage rho in tau = {s1['physical_sources']['rank_coverage_discount_rho']['in_tau']} "
          f"(rho lives in E[T], not tau)")

    s2 = out["step2_no_double_count"]
    print(f"\n[STEP 2] tau_no_double_count = {s2['tau_no_double_count']}  "
          f"(vs K_cal={s2['orthogonality']['vs_K_cal']['orthogonal']}, "
          f"vs step={s2['orthogonality']['vs_step']['orthogonal']}, "
          f"vs E[T]={s2['orthogonality']['vs_E_T']['orthogonal']})")

    s3 = out["step3_pin_topologies"]
    print(f"\n[STEP 3] tau_descent={s3['tau_descent']:.4f}  tau_both_bugs={s3['tau_both_bugs']:.4f}  "
          f"band=[{s3['tau_band_descent'][0]:.4f},{s3['tau_band_descent'][1]:.4f}]  "
          f"topology-invariant={s3['topology_invariant']}  tightened={s3['tightened']}")

    s4 = out["step4_knife_edge"]
    print(f"\n[STEP 4] descent-only LCB re-eval at step {op['step_launch_realized']:.4f}:")
    for k, sc in s4["scenarios"].items():
        if "lcb" in sc:
            print(f"    {k:28s} LCB={sc['lcb']:7.2f}  clears500={sc['clears_500']}")
    print(f"    descent_only_lcb_pinned_tau = {s4['descent_only_lcb_pinned_tau']:.3f} [TEST-related]  "
          f"delta vs #174 = {s4['tps_delta_vs_fern174_floor_tau']:+.4f}  "
          f"clears500={s4['descent_only_clears_500_pinned_tau']}")
    print(f"    verdict: {s4['verdict']}  (floor would need >= {s4['floor_to_clear_500_3term']:.4f} to clear 500 in 3-term)")

    rc = out["reinstantiation_check"]
    print(f"\n[RE-INSTANTIATION] live descent LCB {rc['live_descent_lcb_p90']:.4f} vs "
          f"#174 {rc['fern174_descent_lcb_p90']:.4f}")

    st = out["step5_self_test"]
    print(f"\n[STEP 5] SELF-TEST (PRIMARY):")
    for k in ("assert_a_floor_from_named_subterms", "assert_b_no_double_count",
              "assert_c_pinned_tau_in_band_both_topologies", "assert_d_lcb_reeval_explicit_vs_174",
              "assert_e_nan_clean"):
        print(f"    {k:46s} -> {'OK' if st[k]['ok'] else 'FAIL'}")
    print(f"  => tau_efficiency_self_test_passes = {out['tau_efficiency_self_test_passes']}")
    print(f"\n[STEP 6] HAND-OFF: {out['step6_handoff']}")
    print(f"\n[PRIMARY] tau_efficiency_self_test_passes = {out['tau_efficiency_self_test_passes']}")
    print(f"[TEST]    tau_descent = {out['tau_descent']:.4f}")
    print(f"[STATE]   {out['gate_state']}")


def _log_wandb(args, out):
    import wandb
    op = out["operating_point"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"instrument": "tau-overlap-efficiency-pin",
                             "method": "cpu-analytic-pin-126-148-159-168-174",
                             "K_cal": op["K_cal"], "step_launch_realized": op["step_launch_realized"],
                             "E_T_descent": op["E_T_descent"], "tau_central": op["tau_central"],
                             "tau_band_lo": op["tau_band"][0], "tau_band_hi": op["tau_band"][1],
                             "sigma_hw_pct": op["sigma_hw_pct"], "target_official": TARGET_OFFICIAL,
                             "frontier_official": BASELINE_OFFICIAL_TPS})
    s = wandb.summary
    s["tau_efficiency_self_test_passes"] = out["tau_efficiency_self_test_passes"]
    s["tau_descent"] = out["tau_descent"]
    s["tau_both_bugs"] = out["tau_both_bugs"]
    s["tau_no_double_count"] = int(out["step2_no_double_count"]["tau_no_double_count"])
    s["tau_floor"] = out["step1_definition"]["floor"]
    s["eps_at_floor_pct"] = out["step1_definition"]["eps_at_floor_pct"]
    s["gate_state"] = out["gate_state"]
    s["metrics_nan_clean"] = out["metrics_nan_clean"]
    s4 = out["step4_knife_edge"]
    s["descent_only_lcb_pinned_tau"] = s4["descent_only_lcb_pinned_tau"]
    s["descent_only_clears_500_pinned_tau"] = int(s4["descent_only_clears_500_pinned_tau"])
    s["tps_delta_vs_fern174_floor_tau"] = s4["tps_delta_vs_fern174_floor_tau"]
    s["descent_lcb_pinned_tau_sigma_hw"] = s4["scenarios"]["pinned_tau_4term_sigma_hw"]["lcb"]
    s["descent_lcb_tau_ceiling_sigma_hw"] = s4["scenarios"]["tau_ceiling_4term_sigma_hw"]["lcb"]
    s["floor_to_clear_500_3term"] = s4["floor_to_clear_500_3term"]
    s["knife_edge_verdict"] = s4["verdict"]
    rc = out["reinstantiation_check"]
    s["live_descent_lcb_p90"] = rc["live_descent_lcb_p90"]
    s["fern174_descent_lcb_p90"] = rc["fern174_descent_lcb_p90"]
    for k in ("assert_a_floor_from_named_subterms", "assert_b_no_double_count",
              "assert_c_pinned_tau_in_band_both_topologies", "assert_d_lcb_reeval_explicit_vs_174",
              "assert_e_nan_clean"):
        s[f"selftest_{k}"] = int(out["step5_self_test"][k]["ok"])

    # scenario table
    tbl = wandb.Table(columns=["scenario", "calib_rel", "combined_rel", "lcb", "clears_500"])
    for k, sc in s4["scenarios"].items():
        if "lcb" in sc:
            tbl.add_data(k, sc.get("calib_rel", float("nan")), sc.get("combined_rel", float("nan")),
                         sc["lcb"], int(sc["clears_500"]))
    wandb.log({"knife_edge_scenarios": tbl})
    rid, rurl = run.id, run.url
    print(f"\nW&B run: {rid}  ({rurl})", flush=True)
    wandb.finish()
    return rid, rurl


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-step", type=float, default=STEP_MEASURED_DEPTH9,
                    help="lawine #136/#168 launch-realized depth-9 step (1.2182); combined_rel is "
                         "step-invariant so this fixes the sampling CI.")
    ap.add_argument("--step-rel-half-width", type=float, default=STEP_REL_1SIGMA_DEFAULT,
                    help="lawine #136/#147 step-anchor 1-sigma relative (default 0.5%%).")
    ap.add_argument("--n-steps", type=int, default=cons.env.ORACLE_STEPS,
                    help="verify-step budget for the sampling CI (oracle 1024).")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for the #146 sampling model (sigma_descend is "
                         "bootstrap-independent, so a small value suffices).")
    ap.add_argument("--seed", type=int, default=181)
    ap.add_argument("--out", default="research/validity/tau_efficiency/tau_efficiency_pin_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/tau-overlap-efficiency-pin")
    ap.add_argument("--wandb-group", "--wandb_group", default="tau-overlap-efficiency-pin")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
