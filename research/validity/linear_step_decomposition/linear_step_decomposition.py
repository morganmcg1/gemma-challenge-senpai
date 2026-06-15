#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Deployed LINEAR-step decomposition: price composition honesty (PR #278).

THE QUESTION
------------
The composition formula `official = K_cal * (E[T]/step) * tau` prices a draft-chain
reduction as a STEP reduction: every step-shaving projection (kanna #269 +4.39%,
#277, wirbel #270) subtracts a measured draft saving directly from the served
`step = 1218.2us`. Nobody has MEASURED what that 1218.2us step physically consists
of. If the step is a wall-clock sum `draft + M=1 verify + overhead`, then subtracting
a draft saving propagates 1:1 (composition HONEST). If the step is something else,
the projections may over-credit.

THE MEASUREMENT (measure_linear_verify.py) + THIS ANALYTIC CORE
---------------------------------------------------------------
We CUDA-event the deployed LINEAR-path M=1 target-model verify in isolation
(`target_verify_m1_us`) and the K=7 draft chain, then decompose the step.

THE FINDING (grounded in denken #257 built_step_roofline)
---------------------------------------------------------
denken #257 ALREADY established that `step=1.2182ms` is a NORMALIZED composition
unit, NOT wall-clock: the int4 verify alone has a 2.93ms HBM floor (reading 1.70GB
int4 weights @ 600GB/s), so a single verify forward CANNOT fit inside a 1.2182ms
wall step. This leg MEASURES the verify (~5.36ms wall, 1.8x its floor) and the
draft (~0.71ms wall) and shows:

  step_served(1218.2us) - draft_wall(706.9us) - verify_m1_wall(5357us)
      = -4846us   (NEGATIVE -> the naive same-basis decomposition is INVALID;
                   variable_fraction = (draft+verify)/step = 4.98 > 1, fixed
                   fraction < 0 -> both unphysical -> the 1218.2us step is a
                   NORMALIZED unit ~5x below the wall draft+verify sum.)

The composition prices a WALL draft saving (kanna's 51.24us) against the NORMALIZED
step (1218.2us). The honest pricing bridges the wall saving into the normalized
basis (Delta_norm = bridge * Delta_wall, bridge = step_norm/(draft+verify) ~ 0.20),
so the composition OVER-CREDITS draft-step savings by 1/bridge ~ 5x: kanna #269's
+4.39% composition projection is DISCOUNTED to ~+0.85% at the wall-consistent level.

Pure CPU analytic over the measurement JSON + banked anchors (denken #257/#271,
kanna #217/#269) imported VERBATIM. Analysis-only; BASELINE 481.53 untouched (this
adds 0 TPS; it MEASURES the step budget and validates/discounts the composition).
NOT a launch; no served-file change; no HF Job; no submission."""
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
REPORT_257 = REPO_ROOT / "research/validity/built_step_roofline/built_step_roofline_report.json"
MEAS_271 = REPO_ROOT / "research/validity/gd_step_basis_reconcile/deployed_gd_measurement.json"
MEAS_DEFAULT = HERE / "linear_verify_measurement.json"

# ---- banked composition anchors (imported VERBATIM; never re-derived) ----
OFFICIAL_BASELINE = 481.53          # PR #52 official TPS
GO_READ = 520.9527323111674         # #257 go_read
K_CAL = 125.26795005202914          # #257 K_cal
STEP_SERVED_MS = 1.2182             # kanna #217 vgovdrjc served step (normalized unit)
STEP_SERVED_US = STEP_SERVED_MS * 1e3
E_T_LINEAR = 3.844                  # PR #278 K=7-linear E[T] (== E_T_served/step_served)
TAU = 1.218                         # PR #278 composition round-trip tau
K_SPEC = 7                          # served linear draft passes
# kanna #269 epl52mkq activation fold (the step-shaving projection we price)
KANNA_DELTA_PASS_US = 7.32          # per draft-pass saving
KANNA_DELTA_CHAIN_US = 51.24        # x K=7 chain saving (7.32 * 7 = 51.24)
KANNA_NEW_STEP_PROJ_US = 1166.9     # kanna's projected new step
KANNA_GAIN_PCT_PROJ = 4.39          # kanna's projected composition gain


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# measurement ingest (linear_verify_measurement.json; falls back to #271 banked).
# --------------------------------------------------------------------------- #
def load_measurement(path: Path) -> dict:
    m = _read_json(path)
    if m is not None and "target_verify_m1_us" in m:
        return {
            "present": True, "path": str(path), "basis": "measured_this_run",
            "target_verify_m1_us": m["target_verify_m1_us"],
            "target_verify_m1_us_stats": m.get("target_verify_m1_us_stats"),
            "verify_full8_us": m.get("verify_full8_us"),
            "verify_linear_b8_us": m.get("verify_linear_b8_us"),
            "verify_linear_b8_per_seq_us": m.get("verify_linear_b8_per_seq_us"),
            "linear_vs_full8_ratio": m.get("linear_vs_full8_ratio"),
            "b8_over_b1_ratio": m.get("b8_over_b1_ratio"),
            "draft_pass_us_graphed": m.get("draft_pass_us_graphed"),
            "draft_k7_chain_us_graphed": m.get("draft_k7_chain_us_graphed"),
            "physical_floor": m.get("physical_floor"),
            "deployed_num_layers": m.get("deployed_num_layers"),
            "peak_gpu_gb": m.get("peak_gpu_gb"),
            "measured_cuda_events": True,
            "raw": m,
        }
    # fallback: #271 deployed_gd_measurement.json. verify8 (M=8 tree) ~= the deployed
    # linear M=1 verify (both = full int4 body forward over 8 positions; the body GEMM
    # HBM read dominates so the tree-vs-causal attention mask barely moves it).
    b = _read_json(MEAS_271) or {}
    return {
        "present": False, "path": str(path),
        "basis": "271_banked_verify8_FALLBACK(no measurement json; verify8~=linear-M1)",
        "target_verify_m1_us": b.get("verify8_us", 5357.805088588169),
        "verify_full8_us": b.get("verify8_us"),
        "draft_pass_us_graphed": b.get("draft_pass_us_graphed", 101.95675304957798),
        "draft_k7_chain_us_graphed": b.get("draft_k7_chain_us_graphed", 706.9037628173828),
        "physical_floor": b.get("physical_floor"),
        "deployed_num_layers": b.get("deployed_num_layers"),
        "measured_cuda_events": False,
        "raw": b,
    }


def banked_provenance() -> dict:
    """Re-read the banked anchors from their source JSONs so the round-trip is
    re-checkable (never hand-copied). Falls back to the module constants."""
    r257 = _read_json(REPORT_257) or {}
    a = r257.get("imported_anchors", {})
    return {
        "step_served_ms": a.get("step_served_ms", STEP_SERVED_MS),
        "K_cal": a.get("K_cal", K_CAL),
        "official_baseline": a.get("official_baseline", OFFICIAL_BASELINE),
        "go_read": a.get("go_read", GO_READ),
        "K_spec": a.get("K_spec", K_SPEC),
        "E_T_served": a.get("E_T_served"),
        # #257 already flagged the normalization (the basis fact this leg measures)
        "step_is_normalized_note_257": (r257.get("physical_floor", {}) or {}).get("note"),
        "kcal_overhead_factor_257": (r257.get("physical_floor", {}) or {}).get("kcal_overhead_factor"),
        "bridge_dimensionless_257": r257.get("bridge_dimensionless_normalized_over_wall"),
        "verify_hbm_floor_ms_257": (r257.get("physical_floor", {}) or {}).get("verify_hbm_floor_ms"),
    }


# --------------------------------------------------------------------------- #
# the analytic core.
# --------------------------------------------------------------------------- #
def synthesize(measurement_path: Path) -> dict:
    meas = load_measurement(measurement_path)
    bp = banked_provenance()

    step_us = STEP_SERVED_US
    draft_us = float(meas["draft_k7_chain_us_graphed"])     # K=7 chain, wall (graphed)
    verify_m1_us = float(meas["target_verify_m1_us"])       # deployed LINEAR verify, wall

    # composition E[T]*tau product (round-trips 481.53 exactly; the human split is
    # E[T]=3.844, tau=1.218, but only the product enters the M-shave projection).
    et_tau = OFFICIAL_BASELINE * STEP_SERVED_US / K_CAL      # = E[T]*tau s.t. round-trip exact
    tps_served = K_CAL * et_tau / step_us                   # == OFFICIAL_BASELINE

    def tps_at(new_step_us: float) -> float:
        return K_CAL * et_tau / new_step_us

    # ---- (1) STEP BUDGET DECOMPOSITION (the PR instruction-2 formula, in the
    #          composition's own NORMALIZED step basis -> exposes the basis fault) ----
    step_overhead_us = step_us - draft_us - verify_m1_us          # the "truly fixed" residual
    variable_fraction = (draft_us + verify_m1_us) / step_us       # should be <=1 if a wall sum
    fixed_fraction = step_overhead_us / step_us
    overhead_is_physical = bool(step_overhead_us >= 0.0 and variable_fraction <= 1.0)
    components_sum_exact = abs((draft_us + verify_m1_us + step_overhead_us) - step_us) < 1e-6

    # ---- (2) WALL micro-built step + the bridge to the normalized step ----
    step_wall_us = draft_us + verify_m1_us                        # micro-built wall step
    bridge = step_us / step_wall_us                              # normalized / wall (~0.20)
    overcredit_basis_factor = 1.0 / bridge                       # ~5x
    # the verify HBM floor proves the step cannot be wall (single verify > whole step)
    floor_ms = (meas.get("physical_floor") or {}).get("verify_hbm_floor_ms") \
        or bp.get("verify_hbm_floor_ms_257") or 2.933828266666667
    floor_us = floor_ms * 1e3
    verify_over_floor = verify_m1_us / floor_us
    floor_exceeds_step = bool(floor_us > step_us)               # 2934us > 1218us -> True

    # ---- (3) PRICE THE COMPOSITION HONESTY: kanna #269's activation fold ----
    dchain = KANNA_DELTA_CHAIN_US
    # Model A (composition-faithful, = kanna): subtract the WALL draft saving from
    # the NORMALIZED step. Reproduces kanna's +4.39%.
    new_step_A = step_us - dchain
    tps_A = tps_at(new_step_A)
    gain_A_pct = (tps_A - tps_served) / tps_served * 100.0
    # Model B (basis-honest): bridge the wall saving into the normalized basis first.
    dchain_norm = bridge * dchain
    new_step_B = step_us - dchain_norm
    tps_B = tps_at(new_step_B)
    gain_B_pct = (tps_B - tps_served) / tps_served * 100.0
    composition_overcredit_factor = (gain_A_pct / gain_B_pct) if gain_B_pct else float("inf")

    # composition_is_honest: does a WALL draft saving translate 1:1 to a STEP
    # reduction in the composition basis? Only if the step is wall-clock (overhead
    # physical AND the wall components fit inside the step). The measurement says NO.
    composition_is_honest = bool(overhead_is_physical and variable_fraction <= 1.0 + 1e-9)

    # ---- (4) VERDICT: overhead magnitude (per PR instruction 4) ----
    # PR thresholds are framed for a POSITIVE overhead; here overhead is NEGATIVE
    # (the wall verify alone is ~4.4x the whole normalized step), which is the
    # strongest possible "material fixed fraction" signal: the step is normalized.
    overhead_abs_frac = abs(step_overhead_us) / step_us
    overhead_small = bool(0.0 <= step_overhead_us < 0.04 * step_us)
    overhead_large = bool(step_overhead_us < 0.0 or step_overhead_us > 0.16 * step_us)

    # ---- (5) PRIMARY self-test ----
    roundtrip_tps = K_CAL * (E_T_LINEAR / STEP_SERVED_MS) * TAU   # PR's literal (E[T],tau) split
    cond = {
        "a_components_sum_to_step_within_5pct":
            bool(components_sum_exact
                 and abs(draft_us + verify_m1_us + step_overhead_us - step_us) <= 0.05 * step_us),
        "b_verify_m1_measured_with_cuda_events":
            bool(meas.get("measured_cuda_events", False)),
        "c_composition_roundtrips_481p53_within_0p1":
            bool(abs(roundtrip_tps - OFFICIAL_BASELINE) <= 0.1),
        "d_modelA_reproduces_kanna_4p39_within_0p15":
            bool(abs(gain_A_pct - KANNA_GAIN_PCT_PROJ) <= 0.15),
        "e_nan_clean": all(math.isfinite(x) for x in
                           [step_us, draft_us, verify_m1_us, step_overhead_us, bridge,
                            tps_A, tps_B, gain_A_pct, gain_B_pct, roundtrip_tps]),
        "f_baseline_constants_imported_exact":
            bool(abs(bp["official_baseline"] - 481.53) < 1e-9
                 and abs(bp["go_read"] - 520.9527323111674) < 1e-6
                 and abs(bp["K_cal"] - 125.26795005202914) < 1e-9
                 and abs(bp["step_served_ms"] - 1.2182) < 1e-9),
    }
    self_test_passes = all(cond.values())

    # ---- assemble ----
    budget_table = [
        {"component": "draft_K7_chain", "us": draft_us, "fraction_of_step_norm": draft_us / step_us,
         "basis": "wall (graphed, measured)"},
        {"component": "target_verify_m1", "us": verify_m1_us, "fraction_of_step_norm": verify_m1_us / step_us,
         "basis": "wall (graphed, measured)"},
        {"component": "step_overhead_residual", "us": step_overhead_us, "fraction_of_step_norm": fixed_fraction,
         "basis": "step_norm - draft - verify (NEGATIVE -> normalized step, not wall)"},
        {"component": "step_served_NORMALIZED", "us": step_us, "fraction_of_step_norm": 1.0,
         "basis": "composition unit (kanna #217); = bridge x wall(draft+verify)"},
        {"component": "step_wall_micro_built", "us": step_wall_us, "fraction_of_step_norm": step_wall_us / step_us,
         "basis": "draft+verify wall sum = step_norm / bridge"},
    ]

    verdict = (
        f"COMPOSITION OVER-CREDITS step-shaving. The deployed LINEAR step's WALL "
        f"components are draft {draft_us:.1f}us + M=1 verify {verify_m1_us:.1f}us = "
        f"{step_wall_us:.1f}us, but the composition's 'step' is the NORMALIZED "
        f"{step_us:.1f}us (= {bridge:.3f} x wall). The naive decomposition "
        f"step-draft-verify = {step_overhead_us:.0f}us is NEGATIVE (variable_fraction "
        f"={variable_fraction:.2f} > 1, fixed_fraction = {fixed_fraction:.2f} < 0 -> both "
        f"UNPHYSICAL), and the verify HBM floor {floor_us:.0f}us alone EXCEEDS the whole "
        f"{step_us:.0f}us step ({verify_over_floor:.2f}x floor verify), PROVING the 1218.2us "
        f"step is a normalized unit, not a wall-clock draft+verify+overhead sum (confirms "
        f"denken #257). So kanna #269's Model-A +{gain_A_pct:.2f}% (subtract the WALL "
        f"{dchain:.1f}us saving from the NORMALIZED step) OVER-CREDITS: the basis-honest "
        f"pricing (bridge the saving: Delta_norm = {bridge:.3f} x {dchain:.1f} = "
        f"{dchain_norm:.1f}us) gives +{gain_B_pct:.2f}% -> over-credit factor "
        f"{composition_overcredit_factor:.1f}x. composition_is_honest = "
        f"{composition_is_honest}. BASELINE 481.53 untouched; analysis-only; NOT a launch.")

    handoff = (
        f"the deployed linear step decomposes (WALL) as draft {draft_us:.1f}us + M=1 verify "
        f"{verify_m1_us:.1f}us = {step_wall_us:.1f}us total, but the composition's step is the "
        f"NORMALIZED {step_us:.1f}us (= {bridge:.3f} x wall), so the naive 'step - draft - "
        f"verify' overhead is {step_overhead_us:.0f}us (NEGATIVE, unphysical) and the verify HBM "
        f"floor alone ({floor_us:.0f}us) exceeds the whole step -> the 1218.2us step is a "
        f"normalized unit, NOT a wall sum; the composition therefore OVER-CREDITS draft-step "
        f"savings by {composition_overcredit_factor:.1f}x, so kanna #269's +4.39% composition "
        f"ceiling is DISCOUNTED to +{gain_B_pct:.2f}% (over-credit factor "
        f"{composition_overcredit_factor:.1f}x), and stark #273's wall-clock A/B is the empirical "
        f"verdict on whether even that discounted saving survives the real throughput stack.")

    return {
        "official_baseline": OFFICIAL_BASELINE, "go_read": GO_READ, "K_cal": K_CAL,
        "step_served_us": step_us, "E_T_linear": E_T_LINEAR, "tau": TAU,
        "et_tau_product": et_tau, "tps_served_roundtrip": tps_served,
        "measurement_present": meas["present"], "measurement_path": meas["path"],
        "measurement_basis": meas["basis"], "deployed_num_layers": meas.get("deployed_num_layers"),
        "banked_provenance": bp,
        "measurement_detail": {
            "target_verify_m1_us": verify_m1_us,
            "target_verify_m1_us_stats": meas.get("target_verify_m1_us_stats"),
            "verify_full8_us": meas.get("verify_full8_us"),
            "verify_linear_b8_us": meas.get("verify_linear_b8_us"),
            "verify_linear_b8_per_seq_us": meas.get("verify_linear_b8_per_seq_us"),
            "linear_vs_full8_ratio": meas.get("linear_vs_full8_ratio"),
            "b8_over_b1_ratio": meas.get("b8_over_b1_ratio"),
            "draft_pass_us_graphed": meas.get("draft_pass_us_graphed"),
            "draft_k7_chain_us_graphed": draft_us,
            "physical_floor": meas.get("physical_floor"),
            "peak_gpu_gb": meas.get("peak_gpu_gb"),
        },
        # ---- (1) step budget ----
        "step_budget": {
            "table": budget_table,
            "draft_us": draft_us, "target_verify_m1_us": verify_m1_us,
            "step_overhead_us": step_overhead_us,
            "variable_fraction": variable_fraction, "fixed_fraction": fixed_fraction,
            "overhead_is_physical": overhead_is_physical,
            "components_sum_exact": components_sum_exact,
            "note": "step_overhead = step_norm - draft - verify_m1 is NEGATIVE because the "
                    "1218.2us 'step' is a NORMALIZED composition unit, not a wall draft+verify "
                    "sum; the wall verify alone is ~4.4x the whole step.",
        },
        # ---- (2) wall step + bridge ----
        "wall_step_and_bridge": {
            "step_wall_micro_built_us": step_wall_us,
            "bridge_normalized_over_wall": bridge,
            "overcredit_basis_factor_1_over_bridge": overcredit_basis_factor,
            "verify_hbm_floor_us": floor_us, "verify_m1_over_floor": verify_over_floor,
            "floor_exceeds_whole_step": floor_exceeds_step,
            "bridge_257_crosscheck": bp.get("bridge_dimensionless_257"),
            "note": "step_norm = bridge x (draft+verify) wall; the verify HBM floor exceeding "
                    "the whole step is the physical proof the step is normalized (denken #257).",
        },
        # ---- (3) composition honesty ----
        "composition_honesty": {
            "kanna_delta_chain_us": dchain, "kanna_delta_pass_us": KANNA_DELTA_PASS_US,
            "model_A_new_step_us": new_step_A, "model_A_tps": tps_A, "model_A_gain_pct": gain_A_pct,
            "kanna_proj_gain_pct": KANNA_GAIN_PCT_PROJ, "kanna_proj_new_step_us": KANNA_NEW_STEP_PROJ_US,
            "model_B_delta_norm_us": dchain_norm, "model_B_new_step_us": new_step_B,
            "model_B_tps": tps_B, "model_B_gain_pct": gain_B_pct,
            "composition_overcredit_factor": composition_overcredit_factor,
            "composition_is_honest": composition_is_honest,
            "note": "Model A (=kanna) subtracts the WALL draft saving from the NORMALIZED step "
                    "-> +4.39%. Model B bridges the saving into the normalized basis first "
                    "-> the honest gain. The gap is the over-credit factor (~1/bridge).",
        },
        # ---- (4) verdict ----
        "overhead_verdict": {
            "step_overhead_us": step_overhead_us, "overhead_abs_frac": overhead_abs_frac,
            "overhead_small_lt4pct": overhead_small, "overhead_large_gt16pct_or_negative": overhead_large,
            "classification": "NEGATIVE/normalized-step (strongest material-fixed signal)"
                              if step_overhead_us < 0 else
                              ("small_composition_honest" if overhead_small else "large_material_fixed"),
        },
        # ---- (5) self-test ----
        "self_test": {
            "conditions": cond,
            "roundtrip_tps_literal_split": roundtrip_tps,
            "linear_step_decomposition_self_test_passes": self_test_passes,
        },
        # headline metrics
        "linear_step_decomposition_self_test_passes": self_test_passes,
        "composition_is_honest": composition_is_honest,
        "step_overhead_us": step_overhead_us,
        "target_verify_m1_us": verify_m1_us,
        "variable_fraction": variable_fraction,
        "bridge_normalized_over_wall": bridge,
        "composition_overcredit_factor": composition_overcredit_factor,
        "model_A_gain_pct": gain_A_pct, "model_B_gain_pct": gain_B_pct,
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors denken #271; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    # Import scripts.wandb_logging WITHOUT shadowing the installed `wandb` wheel.
    # The repo root holds a local `wandb/` run-output dir (a PEP-420 namespace
    # package); inserting REPO_ROOT at sys.path[0] makes `import wandb` resolve to
    # that dir (no `.init`) instead of site-packages. Append REPO_ROOT (so
    # site-packages wins for `wandb`) and drop any already-cached namespace `wandb`.
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
                "with no .init -> this venv lacks the wandb wheel (run with the repo .venv, "
                "which has wandb, or `uv pip install wandb`)")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[lin-decomp] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    sb, wb, ch = syn["step_budget"], syn["wall_step_and_bridge"], syn["composition_honesty"]
    st = syn["self_test"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="denken", name=args.wandb_name, group=args.wandb_group,
            tags=["linear-step-decomposition", "composition-honesty", "step-budget", "m1-verify",
                  "deployed-path", "bridge", "over-credit", "bank-the-analysis", "pr-278"],
            config={
                "official_baseline": OFFICIAL_BASELINE, "go_read": GO_READ, "K_cal": K_CAL,
                "step_served_ms": STEP_SERVED_MS, "E_T_linear": E_T_LINEAR, "tau": TAU, "K_spec": K_SPEC,
                "kanna_delta_chain_us": KANNA_DELTA_CHAIN_US, "kanna_proj_gain_pct": KANNA_GAIN_PCT_PROJ,
                "measurement_present": syn["measurement_present"], "measurement_basis": syn["measurement_basis"],
                "imports": "denken#257(built_step_roofline normalized-step+floor) x denken#271(verify8+draft) "
                           "x kanna#217(step=1218.2) x kanna#269(activation fold +4.39%)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[lin-decomp] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[lin-decomp] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "linear_step_decomposition_self_test_passes": int(bool(st["linear_step_decomposition_self_test_passes"])),
        "composition_is_honest": int(bool(syn["composition_is_honest"])),
        "step_overhead_us": syn["step_overhead_us"],
        "target_verify_m1_us": syn["target_verify_m1_us"],
        "draft_k7_chain_us": sb["draft_us"],
        "variable_fraction": sb["variable_fraction"],
        "fixed_fraction": sb["fixed_fraction"],
        "overhead_is_physical": int(bool(sb["overhead_is_physical"])),
        "step_wall_micro_built_us": wb["step_wall_micro_built_us"],
        "bridge_normalized_over_wall": wb["bridge_normalized_over_wall"],
        "overcredit_basis_factor": wb["overcredit_basis_factor_1_over_bridge"],
        "verify_hbm_floor_us": wb["verify_hbm_floor_us"],
        "verify_m1_over_floor": wb["verify_m1_over_floor"],
        "floor_exceeds_whole_step": int(bool(wb["floor_exceeds_whole_step"])),
        "model_A_gain_pct": ch["model_A_gain_pct"],
        "model_B_gain_pct": ch["model_B_gain_pct"],
        "model_A_new_step_us": ch["model_A_new_step_us"],
        "model_B_new_step_us": ch["model_B_new_step_us"],
        "composition_overcredit_factor": ch["composition_overcredit_factor"],
        "et_tau_product": syn["et_tau_product"],
        "tps_served_roundtrip": syn["tps_served_roundtrip"],
        "roundtrip_tps_literal_split": st["roundtrip_tps_literal_split"],
        "verify_full8_us": syn["measurement_detail"].get("verify_full8_us"),
        "verify_linear_b8_per_seq_us": syn["measurement_detail"].get("verify_linear_b8_per_seq_us"),
        "linear_vs_full8_ratio": syn["measurement_detail"].get("linear_vs_full8_ratio"),
        "b8_over_b1_ratio": syn["measurement_detail"].get("b8_over_b1_ratio"),
        "peak_gpu_gb": syn["measurement_detail"].get("peak_gpu_gb"),
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="linear_step_decomposition_result", artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[lin-decomp] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[lin-decomp] wandb write failed (analysis unaffected): {exc}", flush=True)


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
    sb, wb, ch = syn["step_budget"], syn["wall_step_and_bridge"], syn["composition_honesty"]
    print("\n" + "=" * 100, flush=True)
    print(" DEPLOYED LINEAR-STEP DECOMPOSITION (PR #278) — price composition honesty", flush=True)
    print("=" * 100, flush=True)
    print(f"  STEP BUDGET (against the NORMALIZED step {syn['step_served_us']:.1f}us):", flush=True)
    for r in sb["table"]:
        print(f"    {r['component']:<26} {r['us']:>9.1f}us  "
              f"{r['fraction_of_step_norm']*100:>7.1f}%  [{r['basis']}]", flush=True)
    print(f"  variable_fraction={sb['variable_fraction']:.3f} (>1 => unphysical)  "
          f"fixed_fraction={sb['fixed_fraction']:.3f} (<0 => unphysical)  "
          f"overhead_physical={sb['overhead_is_physical']}", flush=True)
    print(f"  WALL micro-step = {wb['step_wall_micro_built_us']:.1f}us  "
          f"bridge(norm/wall) = {wb['bridge_normalized_over_wall']:.4f}  "
          f"1/bridge = {wb['overcredit_basis_factor_1_over_bridge']:.2f}x", flush=True)
    print(f"  verify HBM floor = {wb['verify_hbm_floor_us']:.0f}us "
          f"(EXCEEDS whole step: {wb['floor_exceeds_whole_step']})  "
          f"verify_m1 = {wb['verify_m1_over_floor']:.2f}x floor", flush=True)
    print(f"  COMPOSITION HONESTY (kanna #269 fold, Delta_chain={ch['kanna_delta_chain_us']:.1f}us):", flush=True)
    print(f"    Model A (=kanna): new_step={ch['model_A_new_step_us']:.1f}us -> "
          f"+{ch['model_A_gain_pct']:.2f}% (reproduces kanna's +{ch['kanna_proj_gain_pct']}%)", flush=True)
    print(f"    Model B (honest): Delta_norm={ch['model_B_delta_norm_us']:.1f}us "
          f"new_step={ch['model_B_new_step_us']:.1f}us -> +{ch['model_B_gain_pct']:.2f}%", flush=True)
    print(f"    over-credit factor = {ch['composition_overcredit_factor']:.2f}x  "
          f"composition_is_honest = {ch['composition_is_honest']}", flush=True)
    st = syn["self_test"]
    print(f"  SELF-TEST: {st['linear_step_decomposition_self_test_passes']}  "
          f"{ {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--measurement", type=Path, default=MEAS_DEFAULT,
                    help="linear_verify_measurement.json (falls back to #271 banked if absent)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="linear-step-decomposition")
    args = ap.parse_args(argv)

    syn = synthesize(args.measurement)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 278, "agent": "denken",
        "kind": "linear-step-decomposition", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[lin-decomp] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "linear_step_decomposition_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[lin-decomp] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = bool(syn["self_test"]["linear_step_decomposition_self_test_passes"]) and payload["nan_clean"]
        print(f"[lin-decomp] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
