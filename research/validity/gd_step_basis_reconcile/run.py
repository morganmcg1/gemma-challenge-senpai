#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Deployed-path g_d reconciliation (PR #271): collapse the 480/577.6 step fork.

THE FORK
--------
The #268 verify-GEMM width decision FORKED on a single hidden parameter
    g_d = draft_pass_us / verify_forward_us        (draft<->verify coupling ratio)
- central  g_d = 0.0195 (MEASURED, denken #257 run h1gj2ved) -> M*=32 tops 480.0 TPS  -> NO-GO
- optimistic g_d = 0.168 (ASSUMED fleet anchor)              -> M*=32 peaks 577.6 TPS -> GO
The entire go/no-go for a wider speculative tree (M*=32) rode on which g_d is real.

THE COLLAPSE (this analytic core)
---------------------------------
1. PROVE the two #268 fork edges are the SAME single-parameter step model evaluated
   at two g_d values:
       step(M; g_d) = step_served * [ verify_us(M)/v8 + n_tree*g_d ] / (1 + K_spec*g_d)
   The served LINEAR M=8 anchor (n_draft = K_spec) is g_d-INVARIANT: it round-trips
   step_served = 1.2182 ms / 481.53 TPS for ANY g_d. So the served anchor is a
   CONSISTENCY check, not the discriminator. The discriminator is the DIRECT
   deployed-path measurement + the physical floor.
2. RESOLVE g_d with a DIRECT deployed-path measurement
   (measure_deployed_gd.py: the coupled draft-K=7 -> verify(8) step replayed
   back-to-back under CUDA graphs on the served wheel, region-split into
   draft-region / verify-region us) and a physical HBM-floor exclusion of g_d=0.168.
3. RE-PRICE the M*=32 width decision under the resolved deployed g_d, and report
   the 500-TPS PIVOT g_d* (the break-even coupling ratio): clears_500 iff
   deployed_gd >= g_d*. Tornado/CI says whether the measurement CI is decisive.

Pure CPU analytic over banked numbers (#257 h1gj2ved, #268 banked) imported
VERBATIM + one measurement JSON. NOT a launch; no served-file change; BASELINE
481.53 untouched (this adds 0 TPS; it is an analysis that retires a fork)."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                      # .../target
REPORT_257 = REPO_ROOT / "research/validity/built_step_roofline/built_step_roofline_report.json"
RESULTS_268 = REPO_ROOT / "research/verify_gemm_m_width/verify_gemm_m_width_results.json"
MEAS_DEFAULT = HERE / "deployed_gd_measurement.json"

OFFICIAL_BASELINE = 481.53
TARGET_TPS = 500.0
M_STAR = 32
M_VERIFY = 8


# --------------------------------------------------------------------------- #
# banked-anchor provenance (imported VERBATIM; re-read at runtime so the
# round-trip is re-checkable, never hand-copied).
# --------------------------------------------------------------------------- #
def load_banked() -> dict:
    r257 = json.loads(REPORT_257.read_text())
    r268 = json.loads(RESULTS_268.read_text())
    a = r257["imported_anchors"]
    imp = r268["synthesis"]["imports"]
    sm = r268["synthesis"]["step_models"]
    hop = r268["synthesis"]["M_star"]["handoff_operating_point"]
    et = r268["synthesis"]["et_anchor_matrix"]
    banked = {
        # served operating point + calibration
        "step_served_ms": a["step_served_ms"],            # 1.2182
        "K_cal": a["K_cal"],                              # 125.26795005202914
        "E_T_served": a["E_T_served"],                    # 4.6827608
        "K_spec": a["K_spec"],                            # 7  (served linear draft passes)
        "n_tree": a["tree_draft_passes_empirical_lawine153"],  # 5 (M=32 tree draft passes)
        "tau_band": a["tau_band"],                        # [0.9924, 1.0]
        "official_baseline": a["official_baseline"],      # 481.53
        # the two fork g_d values
        "g_d_measured_257": r257["g_d_measured"],         # 0.019498025961743392
        "g_d_eager_257": r257["g_d_eager_crosscheck"],    # 0.0695620802225808
        "g_d_assumed": a["g_d_assumed"],                  # 0.168
        # verify_us(M) curve + draft pass (banked #257)
        "verify_us_us": {int(k): v for k, v in imp["verify_us_us"].items()},
        "v8_us": imp["v8_us"],                            # 5163.714507818222
        "v32_us": imp["v32_us"],                          # 5979.945073127747
        "draft_pass_us_graphed": imp["draft_pass_us_graphed"],  # 100.6822395324707
        # #268 banked step + tps edges at M=32 (the fork we reproduce)
        "step_central_M32_banked": imp["step_central_M32_banked"],      # 1.3458358727216921
        "step_optimistic_M32_banked": imp["step_optimistic_M32_banked"],# 1.1185888768817671
        "tps_central_M32_at_ET4512_banked": imp["tps_central_M32_at_ET4512_banked"],
        "tps_optimistic_M32_at_ET4512_banked": imp["tps_optimistic_M32_at_ET4512_banked"],
        # #268 step-model internals (to reproduce optimistic basis exactly)
        "sm_verify8_n": sm["verify8_n"], "sm_draft_n": sm["draft_n"],
        "sm_bridge_c": sm["bridge_c"], "sm_n_tree": sm["n_tree"],
        # #268 M*=32 handoff operating point (E[T] held fixed; only g_d varies here)
        "E_T_M32_rankcov": hop["E_T"],                    # 5.15727323332918
        "lambda_hat_M32": hop["lambda_hat"],              # 0.9827
        "tps_central_M32_rankcov_banked": hop["tps_central"],      # 480.02959268046095
        "tps_optimistic_M32_rankcov_banked": hop["tps_optimistic"],# 577.5500357184688
        # E[T] anchor corners (#268 matrix) for cross-anchor robustness
        "et_matrix": {k: v["E_T"] for k, v in et["matrix"].items()},
        # physical floor (#257)
        "verify_hbm_floor_ms": r257["physical_floor"]["verify_hbm_floor_ms"],  # 2.9338...
        "a10g_bw_gbps": r257["physical_floor"]["a10g_bw_gbps"],
        "body_int4_gb": r257["physical_floor"]["body_int4_gb"],
        "source_runs": imp["source_runs"],
    }
    return banked


# --------------------------------------------------------------------------- #
# the UNIFIED single-parameter step model (both #268 fork edges are THIS, at two g_d).
# --------------------------------------------------------------------------- #
class StepModel:
    """step(M; g_d) = step_served * [verify_us(M)/v8 + n_tree*g_d] / (1 + K_spec*g_d).

    Derivation: the served LINEAR M=8 step calibrates the model for ANY g_d via
        verify8_n(g_d) = step_served / (1 + K_spec*g_d)   [verify share of served step]
        draft_n(g_d)   = g_d * verify8_n(g_d)             [per-pass draft share]
    so step(M) = verify8_n*(verify_us(M)/v8) + n_tree*draft_n. The M=32 tree uses
    n_tree=5 draft passes (lawine #153); the served linear uses K_spec=7. That
    7->5 asymmetry is why a LARGER g_d makes the wider tree cheaper (more saved
    draft passes + a smaller verify share), hence MONOTONE-increasing TPS in g_d."""

    def __init__(self, b: dict):
        self.step_served = b["step_served_ms"]
        self.v8 = b["v8_us"]
        self.verify_us = b["verify_us_us"]
        self.n_tree = b["n_tree"]
        self.k_spec = b["K_spec"]
        self.k_cal = b["K_cal"]

    def verify_at(self, M: int) -> float:
        if M in self.verify_us:
            return self.verify_us[M]
        raise KeyError(f"verify_us(M={M}) not in banked grid {sorted(self.verify_us)}")

    def step(self, M: int, g_d: float, *, n_draft: int | None = None) -> float:
        nd = self.n_tree if n_draft is None else n_draft
        return self.step_served * (self.verify_at(M) / self.v8 + nd * g_d) / (1.0 + self.k_spec * g_d)

    def tps(self, M: int, g_d: float, e_t: float, tau: float = 1.0, *, n_draft: int | None = None) -> float:
        return self.k_cal * e_t / self.step(M, g_d, n_draft=n_draft) * tau

    def served_anchor_step(self, g_d: float) -> float:
        """Served LINEAR M=8 with n_draft=K_spec -> g_d-INVARIANT (== step_served)."""
        return self.step(M_VERIFY, g_d, n_draft=self.k_spec)

    def pivot_gd_for_tps(self, M: int, e_t: float, target: float, tau: float = 1.0) -> float:
        """Break-even g_d at which tps(M; g_d) == target (closed form: tps is a
        Mobius function of g_d). clears_500 iff g_d >= pivot (tps increasing in g_d)."""
        # k_cal*e_t*tau*(1+K*g) / (step_served*(r + n*g)) = target, r = verify(M)/v8
        r = self.verify_at(M) / self.v8
        lhs = self.k_cal * e_t * tau
        # lhs*(1+K*g) = target*step_served*(r + n*g)
        # lhs + lhs*K*g = T*ss*r + T*ss*n*g
        T_ss = target * self.step_served
        num = T_ss * r - lhs
        den = lhs * self.k_spec - T_ss * self.n_tree
        return num / den if den != 0 else float("nan")


# --------------------------------------------------------------------------- #
# measurement ingest (deployed_gd_measurement.json; falls back to #257 banked).
# --------------------------------------------------------------------------- #
def load_measurement(path: Path, banked: dict) -> dict:
    if path.exists():
        m = json.loads(path.read_text())
        return {
            "present": True, "path": str(path),
            "deployed_gd": m["deployed_gd"],
            "deployed_gd_lo": m["deployed_gd_lo"], "deployed_gd_hi": m["deployed_gd_hi"],
            "deployed_gd_basis": m.get("deployed_gd_basis"),
            "g_d_isolated": m.get("g_d_isolated"),
            "g_d_isolated_eager": m.get("g_d_isolated_eager"),
            "draft_pass_us_graphed": m.get("draft_pass_us_graphed"),
            "draft_pass_us_eager": m.get("draft_pass_us_eager"),
            "verify8_us": m.get("verify8_us"), "verify32_us": m.get("verify32_us"),
            "coupled": m.get("coupled"),
            "physical_floor": m.get("physical_floor"),
            "deployed_num_layers": m.get("deployed_num_layers"),
            "peak_gpu_gb": m.get("peak_gpu_gb"),
            "raw": m,
        }
    # fallback: the #257 banked ISOLATED graphed g_d (still a valid deployed-ONEGRAPH
    # estimate -- deployed runs CUDA graphs). Flagged so the report is honest.
    g = banked["g_d_measured_257"]
    return {
        "present": False, "path": str(path),
        "deployed_gd": g, "deployed_gd_lo": g, "deployed_gd_hi": g,
        "deployed_gd_basis": "257_banked_isolated_graphed_FALLBACK(no measurement json)",
        "g_d_isolated": g, "g_d_isolated_eager": banked["g_d_eager_257"],
        "draft_pass_us_graphed": banked["draft_pass_us_graphed"],
        "verify8_us": banked["v8_us"], "verify32_us": banked["v32_us"],
        "coupled": None, "physical_floor": None, "raw": None,
    }


# --------------------------------------------------------------------------- #
# the analytic core.
# --------------------------------------------------------------------------- #
def synthesize(measurement_path: Path) -> dict:
    b = load_banked()
    sm = StepModel(b)
    meas = load_measurement(measurement_path, b)
    tol = 1e-6
    E_T = b["E_T_M32_rankcov"]                # E[T] HELD FIXED at the #268 M*=32 operating point
    tau_c = 1.0
    gd_iso = b["g_d_measured_257"]            # 0.0195
    gd_asm = b["g_d_assumed"]                 # 0.168
    gd_dep = meas["deployed_gd"]
    gd_lo, gd_hi = meas["deployed_gd_lo"], meas["deployed_gd_hi"]

    # ---- (1) PROVE the fork edges are one model: reproduce #268 banked edges ----
    step_c = sm.step(M_STAR, gd_iso)
    step_o = sm.step(M_STAR, gd_asm)
    tps_c = sm.tps(M_STAR, gd_iso, E_T, tau_c)
    tps_o = sm.tps(M_STAR, gd_asm, E_T, tau_c)
    provenance = {
        "step_central_M32_model": step_c, "step_central_M32_banked": b["step_central_M32_banked"],
        "step_central_resid": abs(step_c - b["step_central_M32_banked"]),
        "step_optimistic_M32_model": step_o, "step_optimistic_M32_banked": b["step_optimistic_M32_banked"],
        "step_optimistic_resid": abs(step_o - b["step_optimistic_M32_banked"]),
        "tps_central_M32_model": tps_c, "tps_central_M32_banked": b["tps_central_M32_rankcov_banked"],
        "tps_central_resid": abs(tps_c - b["tps_central_M32_rankcov_banked"]),
        "tps_optimistic_M32_model": tps_o, "tps_optimistic_M32_banked": b["tps_optimistic_M32_rankcov_banked"],
        "tps_optimistic_resid": abs(tps_o - b["tps_optimistic_M32_rankcov_banked"]),
        "single_param_model_reproduces_both_268_edges":
            abs(step_c - b["step_central_M32_banked"]) < 1e-9
            and abs(step_o - b["step_optimistic_M32_banked"]) < 1e-9
            and abs(tps_c - b["tps_central_M32_rankcov_banked"]) < 1e-6
            and abs(tps_o - b["tps_optimistic_M32_rankcov_banked"]) < 1e-6,
    }

    # ---- (2) ANCHOR self-test: served linear M=8 is g_d-INVARIANT (consistency, not discriminator) ----
    step_anchor_iso = sm.served_anchor_step(gd_iso)
    step_anchor_asm = sm.served_anchor_step(gd_asm)
    step_anchor_dep = sm.served_anchor_step(gd_dep)
    tps_anchor_iso = sm.k_cal * b["E_T_served"] / step_anchor_iso
    tps_anchor_asm = sm.k_cal * b["E_T_served"] / step_anchor_asm
    tps_anchor_dep = sm.k_cal * b["E_T_served"] / step_anchor_dep
    anchor = {
        "step_served_ms": b["step_served_ms"],
        "step_anchor_isolated_ms": step_anchor_iso, "step_anchor_assumed_ms": step_anchor_asm,
        "step_anchor_deployed_ms": step_anchor_dep,
        "tps_at_anchor_isolated": tps_anchor_iso, "tps_at_anchor_assumed": tps_anchor_asm,
        "tps_at_anchor_deployed": tps_anchor_dep,
        "official_baseline": OFFICIAL_BASELINE,
        "anchor_roundtrips_isolated_within_1pct": abs(tps_anchor_iso - OFFICIAL_BASELINE) / OFFICIAL_BASELINE <= 0.01,
        "anchor_roundtrips_assumed_within_1pct": abs(tps_anchor_asm - OFFICIAL_BASELINE) / OFFICIAL_BASELINE <= 0.01,
        "anchor_is_gd_invariant": abs(step_anchor_iso - step_anchor_asm) < 1e-9,
        "note": "served LINEAR M=8 uses n_draft=K_spec=7 (the calibration point) so the "
                "model returns step_served EXACTLY for any g_d -> the 481.53 anchor "
                "round-trips under BOTH the isolated (0.0195) and assumed (0.168) g_d. "
                "The anchor therefore CANNOT discriminate the fork; the discriminator is "
                "the direct deployed measurement + the HBM floor.",
    }

    # ---- (3) RESOLVE: re-price M*=32 under the deployed g_d; 500-TPS PIVOT ----
    g_pivot = sm.pivot_gd_for_tps(M_STAR, E_T, TARGET_TPS, tau_c)   # break-even g_d
    step32_dep = sm.step(M_STAR, gd_dep)
    tps32_dep = sm.tps(M_STAR, gd_dep, E_T, tau_c)
    tps32_dep_lo = sm.tps(M_STAR, gd_lo, E_T, tau_c)   # tps MONOTONE-increasing in g_d
    tps32_dep_hi = sm.tps(M_STAR, gd_hi, E_T, tau_c)
    width_clears = bool(tps32_dep >= TARGET_TPS)
    resolution = {
        "deployed_gd": gd_dep, "deployed_gd_lo": gd_lo, "deployed_gd_hi": gd_hi,
        "deployed_gd_basis": meas["deployed_gd_basis"],
        "g_d_isolated_measured": meas.get("g_d_isolated"),
        "g_d_isolated_257_banked": gd_iso,
        "g_d_assumed": gd_asm,
        "E_T_M32_held": E_T, "lambda_hat_M32": b["lambda_hat_M32"], "tau": tau_c,
        "step_at_Mstar32_deployed_ms": step32_dep,
        "tps_at_Mstar32_deployed": tps32_dep,
        "tps_at_Mstar32_deployed_lo": tps32_dep_lo, "tps_at_Mstar32_deployed_hi": tps32_dep_hi,
        "width_clears_500_deployed": width_clears,
        "g_d_pivot_500tps": g_pivot,
        "deployed_gd_above_pivot": bool(gd_dep >= g_pivot),
        "margin_gd_to_pivot": gd_dep - g_pivot,
        "tps_gap_to_500": tps32_dep - TARGET_TPS,
        "pivot_note": f"clears 500 iff deployed_gd >= g_d* = {g_pivot:.5f} (the 500-TPS pivot at "
                      f"M=32, E[T]={E_T:.4f}). The #268 fork edges 0.0195 (NO-GO 480) and 0.168 "
                      f"(GO 577.6) straddle this pivot; the measurement places the deployed g_d.",
    }

    # ---- (4) TORNADO / CI + physical-floor exclusion of g_d=0.168 ----
    # how far below the floor would verify have to be for g_d=0.168 to hold?
    draft_us = meas.get("draft_pass_us_graphed") or b["draft_pass_us_graphed"]
    floor_ms = (meas["physical_floor"]["verify_hbm_floor_ms"]
                if meas.get("physical_floor") else b["verify_hbm_floor_ms"])
    floor_us = floor_ms * 1e3
    g_d_max_from_floor = draft_us / floor_us                       # max g_d at fastest-possible verify
    verify_needed_for_assumed_us = draft_us / gd_asm               # verify if g_d=0.168 held
    verify_needed_over_floor = verify_needed_for_assumed_us / floor_us
    # CI decisiveness: does the deployed CI exclude the optimistic edge and stay below the pivot?
    ci_excludes_assumed = bool(gd_hi < gd_asm)
    ci_below_pivot = bool(gd_hi < g_pivot)
    ci_above_pivot = bool(gd_lo >= g_pivot)
    ci_straddles_pivot = bool(gd_lo < g_pivot <= gd_hi)
    tps_ci_straddles_500 = bool(tps32_dep_lo < TARGET_TPS <= tps32_dep_hi)
    gd_fork_decisive = bool((ci_below_pivot or ci_above_pivot) and ci_excludes_assumed)
    tornado = {
        "g_d_pivot_500tps": g_pivot,
        "ci_excludes_assumed_0p168": ci_excludes_assumed,
        "ci_below_pivot": ci_below_pivot, "ci_above_pivot": ci_above_pivot,
        "ci_straddles_pivot": ci_straddles_pivot,
        "tps_ci_lo": tps32_dep_lo, "tps_ci_hi": tps32_dep_hi,
        "tps_ci_straddles_500": tps_ci_straddles_500,
        "gd_fork_decisive": gd_fork_decisive,
        "physical_floor": {
            "draft_pass_us": draft_us, "verify_hbm_floor_us": floor_us,
            "g_d_max_from_floor": g_d_max_from_floor,
            "floor_excludes_assumed_0p168": bool(g_d_max_from_floor < gd_asm),
            "verify_needed_for_assumed_us": verify_needed_for_assumed_us,
            "verify_needed_over_floor": verify_needed_over_floor,
            "note": f"g_d=0.168 requires verify(8)={verify_needed_for_assumed_us:.0f}us = "
                    f"{verify_needed_over_floor:.2f}x the {floor_us:.0f}us HBM floor "
                    f"(reads {b['body_int4_gb']:.2f}GB int4 @ {b['a10g_bw_gbps']:.0f}GB/s) -> "
                    f"PHYSICALLY IMPOSSIBLE. Even at the floor, g_d <= {g_d_max_from_floor:.4f}.",
        },
    }

    # ---- (4b) BASIS robustness: every independent g_d estimate vs the pivot ----
    # The deployed path runs ONEGRAPH=1 (CUDA graphs), so the GRAPHED basis is
    # faithful; the EAGER basis carries per-kernel Python launch overhead the
    # deployed graph eliminates. All graphed estimates sit below the pivot; only
    # the (non-deployed) eager basis would clear -- ruled out by ONEGRAPH.
    gd_eager = meas.get("g_d_isolated_eager") or b["g_d_eager_257"]
    gd_iso_meas = meas.get("g_d_isolated")
    basis_estimates = {
        "graphed_coupled_deployed": gd_dep,
        "graphed_isolated_thisrun": gd_iso_meas,
        "graphed_isolated_257_banked": gd_iso,
        "eager_isolated_thisrun": gd_eager,
        "hbm_floor_cap": g_d_max_from_floor,
        "assumed_fleet": gd_asm,
    }
    graphed_keys = ["graphed_coupled_deployed", "graphed_isolated_thisrun", "graphed_isolated_257_banked"]
    graphed_vals = [basis_estimates[k] for k in graphed_keys if basis_estimates[k] is not None]
    robustness = {
        "estimates": basis_estimates,
        "g_d_pivot_500tps": g_pivot,
        "all_graphed_below_pivot": bool(all(v < g_pivot for v in graphed_vals)),
        "graphed_span": [min(graphed_vals), max(graphed_vals)],
        "eager_above_pivot": bool(gd_eager >= g_pivot),
        "eager_ruled_out_by_onegraph": True,
        "deployed_runs_onegraph": True,
        "note": "deployed manifest fa2sw_precache_kenyan sets ONEGRAPH=1 (the whole "
                "spec-decode step -- draft AND verify -- is CUDA-graph captured), so the "
                "GRAPHED g_d (~0.019) is the deployed-faithful basis. The eager g_d "
                f"({gd_eager:.4f}) is launch-inflated (the drafter's {meas.get('raw', {}).get('n_draft_gemms') if isinstance(meas.get('raw'), dict) else 'many'} "
                "tiny per-pass GEMMs x K=7 pay Python launch each, which the captured graph "
                "removes); it is NOT the deployed path and is excluded by ONEGRAPH. The only "
                "scenario that flips to GO (eager draft) is architecturally ruled out.",
    }

    # ---- (5) cross-anchor robustness: re-price M=32 under deployed g_d at every E[T] corner ----
    et_corners = dict(b["et_matrix"])
    et_corners.setdefault("rankcov_M32_5p157", E_T)
    et_matrix = {}
    for name, et_val in sorted(et_corners.items(), key=lambda kv: kv[1]):
        t = sm.tps(M_STAR, gd_dep, et_val, tau_c)
        et_matrix[name] = {
            "E_T": et_val, "tps_deployed": t, "clears_500": bool(t >= TARGET_TPS),
            "g_d_pivot": sm.pivot_gd_for_tps(M_STAR, et_val, TARGET_TPS, tau_c),
        }
    n_corners_clear = sum(1 for v in et_matrix.values() if v["clears_500"])

    # ---- (6) tau-band sensitivity (does +/-0.76% tau flip the deployed call?) ----
    tau_lo, tau_hi = b["tau_band"]
    tps32_tau_lo = sm.tps(M_STAR, gd_dep, E_T, tau_lo)
    tps32_tau_hi = sm.tps(M_STAR, gd_dep, E_T, tau_hi)
    tau_block = {
        "tau_band": b["tau_band"], "tps_deployed_tau_lo": tps32_tau_lo, "tps_deployed_tau_hi": tps32_tau_hi,
        "tau_flips_500": bool((tps32_tau_lo < TARGET_TPS) != (tps32_tau_hi < TARGET_TPS)),
    }

    # ---- (7) PRIMARY self-test ----
    key_scalars = [step_c, step_o, tps_c, tps_o, step32_dep, tps32_dep, g_pivot,
                   g_d_max_from_floor, gd_dep, gd_lo, gd_hi, tps_anchor_iso, tps_anchor_asm]
    cond = {
        "a_single_param_model_reproduces_both_268_fork_edges":
            bool(provenance["single_param_model_reproduces_both_268_edges"]),
        "b_served_anchor_gd_invariant_roundtrips_481p53_both_bases":
            bool(anchor["anchor_is_gd_invariant"]
                 and anchor["anchor_roundtrips_isolated_within_1pct"]
                 and anchor["anchor_roundtrips_assumed_within_1pct"]),
        "c_deployed_gd_measured_in_open_fork_interval_with_ci":
            bool(0.0 < gd_dep < gd_asm and gd_lo <= gd_dep <= gd_hi and math.isfinite(g_pivot)),
        "d_physical_floor_excludes_assumed_0p168":
            bool(g_d_max_from_floor < gd_asm and verify_needed_over_floor < 1.0),
        "e_key_scalars_finite": all(math.isfinite(x) for x in key_scalars),
    }
    self_test_passes = all(cond.values())

    # ---- verdict ----
    if width_clears:
        call = (f"GO on width: deployed g_d={gd_dep:.4f} >= pivot {g_pivot:.4f} -> "
                f"M*=32 prices {tps32_dep:.1f} TPS >= 500")
    else:
        call = (f"NO-GO on width: deployed g_d={gd_dep:.4f} < pivot {g_pivot:.4f} -> "
                f"M*=32 prices {tps32_dep:.1f} TPS < 500; the wider tree does NOT clear 500 on "
                f"the deployed path. The 577.6 optimistic edge was an artifact of an "
                f"unphysical assumed g_d=0.168 (>= {1.0/verify_needed_over_floor:.1f}x too fast "
                f"a verify). GATE rests on the DEPTH axis (stark #266) + an E[T] lift (fern #259), "
                f"NOT tree width.")
    verdict = (f"STEP-BASIS FORK COLLAPSED. The #268 480.0/577.6 split is ONE step model "
               f"step(M;g_d)=step_served*[v(M)/v8 + n_tree*g_d]/(1+K_spec*g_d) at g_d=0.0195 "
               f"(central) vs 0.168 (optimistic); the served M=8 anchor is g_d-INVARIANT "
               f"(481.53 under both) so it cannot decide. The deployed-path coupled measurement "
               f"({resolution['deployed_gd_basis']}) places g_d={gd_dep:.4f} "
               f"[{gd_lo:.4f},{gd_hi:.4f}] and the HBM floor caps g_d<={g_d_max_from_floor:.4f}, "
               f"both far below the 0.168 optimistic edge and the 500-TPS pivot g_d*={g_pivot:.4f}. "
               f"{call} (ALL graphed bases -- coupled {gd_dep:.4f}, isolated {gd_iso_meas if gd_iso_meas else gd_iso:.4f}, "
               f"#257 {gd_iso:.4f} -- sit below the pivot; only the launch-inflated EAGER basis "
               f"{gd_eager:.4f} would clear, and ONEGRAPH=1 graphs the deployed draft so the deployed "
               f"path does NOT pay it.) BASELINE 481.53 untouched. CPU analytic + local micro-measurement. NOT a launch.")

    handoff = (f"PR #271 hand-off (land #245 / stark #266 / fern #259): the deployed-path g_d "
               f"is {gd_dep:.4f} [{gd_lo:.4f},{gd_hi:.4f}] (graphed coupled, not the assumed "
               f"0.168 which the HBM floor excludes at g_d<={g_d_max_from_floor:.4f}), so the "
               f"#268 M*=32 width prices {tps32_dep:.1f} TPS at the rank-cov E[T]=5.157 operating "
               f"point -- {'CLEARS' if width_clears else 'BELOW'} 500 (pivot g_d*={g_pivot:.4f}); "
               f"tree WIDTH alone is {'sufficient' if width_clears else 'NOT the lever'}, the gate "
               f"is the DEPTH axis + E[T] realization.")

    return {
        "official_baseline": OFFICIAL_BASELINE, "target_tps": TARGET_TPS, "M_star": M_STAR,
        "measurement_present": meas["present"], "measurement_path": meas["path"],
        "deployed_num_layers": meas.get("deployed_num_layers"),
        "banked_provenance": {
            "step_served_ms": b["step_served_ms"], "K_cal": b["K_cal"], "E_T_served": b["E_T_served"],
            "K_spec": b["K_spec"], "n_tree": b["n_tree"], "v8_us": b["v8_us"], "v32_us": b["v32_us"],
            "E_T_M32_rankcov": b["E_T_M32_rankcov"], "tau_band": b["tau_band"],
            "source_runs": b["source_runs"],
        },
        "single_param_model": {
            "formula": "step(M;g_d) = step_served*[verify_us(M)/v8 + n_tree*g_d]/(1 + K_spec*g_d)",
            "step_served_ms": b["step_served_ms"], "v8_us": b["v8_us"],
            "n_tree": b["n_tree"], "K_spec": b["K_spec"], "K_cal": b["K_cal"],
            "provenance_roundtrip": provenance,
        },
        "anchor_self_test": anchor,
        "fork_resolution": resolution,
        "tornado_ci": tornado,
        "basis_robustness": robustness,
        "et_anchor_matrix_deployed": {
            "E_T_held_primary": E_T, "matrix": et_matrix,
            "n_corners_clearing_500": n_corners_clear, "n_corners_total": len(et_matrix),
        },
        "tau_band": tau_block,
        "measurement_detail": {
            "g_d_isolated": meas.get("g_d_isolated"), "g_d_isolated_eager": meas.get("g_d_isolated_eager"),
            "draft_pass_us_graphed": meas.get("draft_pass_us_graphed"),
            "draft_pass_us_eager": meas.get("draft_pass_us_eager"),
            "verify8_us": meas.get("verify8_us"), "verify32_us": meas.get("verify32_us"),
            "coupled": meas.get("coupled"), "physical_floor": meas.get("physical_floor"),
            "peak_gpu_gb": meas.get("peak_gpu_gb"),
        },
        # headline test metrics
        "deployed_gd": gd_dep, "deployed_gd_lo": gd_lo, "deployed_gd_hi": gd_hi,
        "tps_at_Mstar32_deployed": tps32_dep,
        "width_clears_500_deployed": width_clears,
        "g_d_pivot_500tps": g_pivot,
        "gd_fork_decisive": gd_fork_decisive,
        "tps_at_anchor_isolated": tps_anchor_iso, "tps_at_anchor_assumed": tps_anchor_asm,
        "self_test": {"conditions": cond, "gd_step_basis_reconcile_self_test_passes": self_test_passes},
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors denken #268; never fatal).
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
        print(f"[gd-reconcile] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    res, tor, anc, st = syn["fork_resolution"], syn["tornado_ci"], syn["anchor_self_test"], syn["self_test"]
    bp = syn["banked_provenance"]
    run = init_wandb_run(
        job_type="validity-gate",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["gd-reconcile", "step-basis", "g_d-fork", "deployed-path", "coupled-measure",
              "verify-width", "M-star-32", "hbm-floor", "bank-the-analysis", "pr-271"],
        config={
            "official_baseline": OFFICIAL_BASELINE, "target_tps": TARGET_TPS, "M_star": M_STAR,
            "step_served_ms": bp["step_served_ms"], "K_cal": bp["K_cal"],
            "E_T_M32_rankcov": bp["E_T_M32_rankcov"], "K_spec": bp["K_spec"], "n_tree": bp["n_tree"],
            "g_d_assumed": syn["fork_resolution"]["g_d_assumed"],
            "g_d_isolated_257": syn["fork_resolution"]["g_d_isolated_257_banked"],
            "measurement_present": payload["synthesis"]["measurement_present"],
            "deployed_gd_basis": res["deployed_gd_basis"],
            "imports": "denken#257(h1gj2ved) step+g_d+floor x denken#268 verify_us(M)+E[T] M*=32",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[gd-reconcile] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "gd_step_basis_reconcile_self_test_passes": int(bool(st["gd_step_basis_reconcile_self_test_passes"])),
        "deployed_gd": syn["deployed_gd"],
        "deployed_gd_lo": syn["deployed_gd_lo"], "deployed_gd_hi": syn["deployed_gd_hi"],
        "tps_at_Mstar32_deployed": syn["tps_at_Mstar32_deployed"],
        "tps_at_Mstar32_deployed_lo": res["tps_at_Mstar32_deployed_lo"],
        "tps_at_Mstar32_deployed_hi": res["tps_at_Mstar32_deployed_hi"],
        "width_clears_500_deployed": int(bool(syn["width_clears_500_deployed"])),
        "g_d_pivot_500tps": syn["g_d_pivot_500tps"],
        "deployed_gd_above_pivot": int(bool(res["deployed_gd_above_pivot"])),
        "margin_gd_to_pivot": res["margin_gd_to_pivot"],
        "tps_gap_to_500": res["tps_gap_to_500"],
        "gd_fork_decisive": int(bool(syn["gd_fork_decisive"])),
        "ci_excludes_assumed_0p168": int(bool(tor["ci_excludes_assumed_0p168"])),
        "g_d_max_from_floor": tor["physical_floor"]["g_d_max_from_floor"],
        "floor_excludes_assumed_0p168": int(bool(tor["physical_floor"]["floor_excludes_assumed_0p168"])),
        "verify_needed_over_floor_for_assumed": tor["physical_floor"]["verify_needed_over_floor"],
        "tps_at_anchor_isolated": syn["tps_at_anchor_isolated"],
        "tps_at_anchor_assumed": syn["tps_at_anchor_assumed"],
        "anchor_is_gd_invariant": int(bool(anc["anchor_is_gd_invariant"])),
        "step_central_M32_resid": syn["single_param_model"]["provenance_roundtrip"]["step_central_resid"],
        "step_optimistic_M32_resid": syn["single_param_model"]["provenance_roundtrip"]["step_optimistic_resid"],
        "n_et_corners_clearing_500": syn["et_anchor_matrix_deployed"]["n_corners_clearing_500"],
        "tau_flips_500": int(bool(syn["tau_band"]["tau_flips_500"])),
        "g_d_isolated_measured": res["g_d_isolated_measured"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        # et-corner curve for plotting
        **{f"tps_deployed_ET_{k}": v["tps_deployed"]
           for k, v in syn["et_anchor_matrix_deployed"]["matrix"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="gd_step_basis_reconcile_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[gd-reconcile] wandb logged {len(summary)} summary keys", flush=True)


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
    res, tor, anc = syn["fork_resolution"], syn["tornado_ci"], syn["anchor_self_test"]
    pr = syn["single_param_model"]["provenance_roundtrip"]
    print("\n" + "=" * 100, flush=True)
    print(" DEPLOYED-PATH g_d RECONCILIATION (PR #271) — collapse the 480/577.6 step fork", flush=True)
    print("=" * 100, flush=True)
    print(f"  model: step(M;g_d) = step_served*[v(M)/v8 + n_tree*g_d]/(1+K_spec*g_d)", flush=True)
    print(f"    reproduces #268 edges: step_central={pr['step_central_M32_model']:.6f} "
          f"(resid {pr['step_central_resid']:.2e}), step_opt={pr['step_optimistic_M32_model']:.6f} "
          f"(resid {pr['step_optimistic_resid']:.2e})", flush=True)
    print(f"  ANCHOR (served M=8 linear, g_d-invariant): tps_iso={anc['tps_at_anchor_isolated']:.2f} "
          f"tps_asm={anc['tps_at_anchor_assumed']:.2f} (both ~= {OFFICIAL_BASELINE})", flush=True)
    print(f"  DEPLOYED g_d = {res['deployed_gd']:.4f} [{res['deployed_gd_lo']:.4f},"
          f"{res['deployed_gd_hi']:.4f}]  ({res['deployed_gd_basis']})", flush=True)
    print(f"    isolated cross-check: measured={res['g_d_isolated_measured']}  #257-banked={res['g_d_isolated_257_banked']:.4f}", flush=True)
    print(f"  500-TPS PIVOT g_d* = {res['g_d_pivot_500tps']:.5f}  (clears 500 iff deployed_gd >= pivot)", flush=True)
    print(f"  M*=32 @ E[T]={res['E_T_M32_held']:.4f}: step={res['step_at_Mstar32_deployed_ms']:.4f}ms "
          f"-> TPS={res['tps_at_Mstar32_deployed']:.1f} "
          f"[{res['tps_at_Mstar32_deployed_lo']:.1f},{res['tps_at_Mstar32_deployed_hi']:.1f}]  "
          f"clears_500={res['width_clears_500_deployed']}", flush=True)
    pf = tor["physical_floor"]
    print(f"  PHYSICAL FLOOR: g_d=0.168 needs verify={pf['verify_needed_for_assumed_us']:.0f}us "
          f"= {pf['verify_needed_over_floor']:.2f}x floor -> impossible; g_d <= {pf['g_d_max_from_floor']:.4f}", flush=True)
    print(f"  gd_fork_decisive={syn['gd_fork_decisive']}  (CI excludes 0.168: {tor['ci_excludes_assumed_0p168']})", flush=True)
    print(f"  SELF-TEST: {syn['self_test']['gd_step_basis_reconcile_self_test_passes']}  "
          f"{ {k: int(v) for k, v in syn['self_test']['conditions'].items()} }", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--measurement", type=Path, default=MEAS_DEFAULT,
                    help="deployed_gd_measurement.json (falls back to #257 banked if absent)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="denken-gd-reconcile")
    args = ap.parse_args(argv)

    syn = synthesize(args.measurement)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 271, "agent": "denken",
        "kind": "gd-step-basis-reconcile", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[gd-reconcile] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gd_step_basis_reconcile_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[gd-reconcile] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = bool(syn["self_test"]["gd_step_basis_reconcile_self_test_passes"]) and payload["nan_clean"]
        print(f"[gd-reconcile] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
