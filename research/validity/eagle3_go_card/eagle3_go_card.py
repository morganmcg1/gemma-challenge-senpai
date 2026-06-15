#!/usr/bin/env python3
"""PR #305 -- The EAGLE-3 GO/NO-GO decision card: tornado over the priced axes.

WHAT THIS LEG PRODUCES
----------------------
The build-economics matrix has priced six axes (step cost, companion floor, per-position
target, VRAM, build cost, read companion) and they all clear. Nobody has ASSEMBLED them into
the ONE artifact the human EAGLE-3 GO/NO-GO needs. This leg is that artifact:

  (1) DETERMINISTIC card -- reproduce (do not assume) the corrected-target invariant: at the
      wirbel #295 central corrected target E[T]=6.1112, the banked public projection equals the
      500 line BY CONSTRUCTION of the corrected target. Sweep E[T] across the #295 bracket
      [5.3636, 6.8588]; map public -> private via the banked OOD factor x0.804; report the E[T]
      at which PUBLIC crosses 500 and the E[T] at which PRIVATE crosses 500 (the binding one).
  (2) TORNADO -- rank each priced axis's banked uncertainty by its impact on projected PRIVATE
      TPS: {E[T] bracket, step-multiplier regime, private_factor, sigma_hw, SAM companion}.
      Headline the single axis whose uncertainty most threatens the 500 line.
  (3) P(clear 500) -- Monte-Carlo the joint over triangular banked bands; report P(public>=500),
      P(private>=500), and the conditional P(private>=500 | E[T] reaches its central 6.11).
  (4) HONEST caveat -- the card is CONDITIONAL on numerator reachability (kanna #294 / wirbel
      #303 / denken #304 in flight). It prices "IF reachable, here is the number and the risk,"
      NOT whether 6.11 is reachable. It is NOT a launch recommendation and NOT a measured TPS.

EVERYTHING is IMPORTED from merged banked runs (cited below) -- NOTHING re-derived. The core
projection reuses wirbel #295's banked `tps_public` / `corrected_target_at_step` verbatim, which
round-trip to 500 at the corrected target by construction.

IMPORTED CONSTANTS (run id -> value, self-tested to <=1e-6 against source JSON)
    official_baseline 481.53 ............ wirbel #295 eagle3_step_profile (c334qaqu)
    ceiling_lambda1   520.953 ........... wirbel #290 eagle3_feasibility_bracket
    K_cal 125.268 / step_us 1218.2 / tau 1.218 / E_T_anchor 3.844 .. wirbel #295
    honest500 E[T] floor 3.9914 ......... wirbel #290
    corrected target E[T] central 6.1112, bracket [5.3636, 6.8588] .. wirbel #295 regime_bracket
    step-multiplier band [1.745x, 4.161x] central ~2.95x ........... wirbel #295
    private_factor 0.804 (priv/pub E[T] 3.0898/3.844) ............... wirbel #290 / ubel #258 / kanna #289
    private E[T] band [3.0898 decode, 3.6554 pooled] ............... ubel #258 private_et_gap_decomp
    private_verified 460.85 ............. fern #302 read_cut_build_companion (8jewx2ur)
    read companion ratio -2.0177 ........ fern #302
    SAM companion residual 0.998, marginal [0.79, 1.59]% ........... lawine #296 sam_eagle3_stacking
    VRAM resident 20.10 GiB / headroom 3.90 GiB ................... ubel #299 (jnoss7id)
    build cost 107.47 GPU-hr / headroom 92.53 ..................... denken #301 (b4zg7b6c)
    sigma_hw 4.864 TPS .................. kanna #159 / launch_sigma_closure

SCOPE. LOCAL CPU-only synthesis. NO GPU / vLLM / HF Job / submission / served-file change /
official draw. 0 TPS added; BASELINE 481.53 untouched; greedy/PPL untouched. CONDITIONAL on
reachability (#294/#303/#304). Authorizes NOTHING. NOT a launch.

PRIMARY metric  eagle3_go_card_self_test_passes
TEST    metrics go_card_tornado_top_axis (str) + p_private_clears_500 (float) + et_private_cross_500 (float)

Run:
    python research/validity/eagle3_go_card/eagle3_go_card.py \
        --wandb_group eagle3-go-card --wandb_name fern/eagle3-go-card
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V = os.path.join(REPO_ROOT, "research", "validity")

# --------------------------------------------------------------------------- #
# Source banked runs (import-not-rederive). Each constant cites file + JSON path.
# --------------------------------------------------------------------------- #
STEP_PROFILE = os.path.join(V, "eagle3_step_profile/eagle3_step_profile_results.json")          # wirbel #295
FEAS = os.path.join(V, "eagle3_feasibility_bracket/eagle3_feasibility_bracket_results.json")    # wirbel #290
SAM = os.path.join(V, "sam_eagle3_stacking/sam_eagle3_stacking_report.json")                    # lawine #296
VRAM = os.path.join(V, "eagle3_vram_budget/eagle3_vram_budget_results.json")                    # ubel #299
BUILDCOST = os.path.join(V, "eagle3_build_cost/eagle3_build_cost_results.json")                 # denken #301
READCUT = os.path.join(V, "read_cut_build_companion/read_cut_build_companion_results.json")     # fern #302
SIGMA = os.path.join(V, "launch_sigma_closure/launch_sigma_closure_results.json")               # kanna #159/#201
PRIVGAP = os.path.join(V, "private_et_gap_decomp/private_et_gap_decomp_results.json")           # ubel #258
# wirbel #295 banked projection machinery (reused verbatim, NOT re-derived):
STEP_PROFILE_PY = os.path.join(V, "eagle3_step_profile/eagle3_step_profile.py")

TARGET = 500.0
IMPORT_TOL = 1e-6        # PR self-test (a): every import matches its source to <=1e-6
ROUNDTRIP_TOL = 1e-6     # PR self-test (b): deterministic projection reproduces 500
MC_DRAWS = 200_000       # PR step 4: >=100k


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dig(d: Any, path: str) -> Any:
    """Follow a '/'-separated JSON path; list indices are bare integers."""
    cur = d
    for seg in path.strip("/").split("/"):
        if isinstance(cur, list):
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            cur = cur[seg]
        else:
            raise KeyError(f"cannot descend into {seg!r} of {path!r}")
    return cur


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


# reuse wirbel #295's banked projection (module-level constants + pure functions; GPU only in run()).
MOD = _import("eagle3_step_profile", STEP_PROFILE_PY)
FERN_FLOOR = float(MOD.FERN_FLOOR_PUBLIC)     # 4.966 public-E[T] floor; tps_public(4.966, step_us) == 500


def tps_public(et: float, step_us: float) -> float:
    """Banked wirbel #295 projection: 500 * (et/4.966) * (step_us_anchor/step_us). Round-trips
    to 500 at the corrected target by construction (corrected_target_at_step is its inverse)."""
    return float(MOD.tps_public(et, step_us))


def step_for_corrected(corrected_et: float, step_us_anchor: float) -> float:
    """Inverse of corrected_target_at_step: the (heavier) step at which `corrected_et` nets 500."""
    return corrected_et * step_us_anchor / FERN_FLOOR


# --------------------------------------------------------------------------- #
# Step 1 -- import every banked constant + self-verify to <=1e-6.
# --------------------------------------------------------------------------- #
def load_constants() -> dict[str, Any]:
    # (cited value, source file, source JSON path). The cited value is the PR-stated import; the
    # loader reads the source and the self-test asserts |cited - source| <= 1e-6.
    spec: dict[str, tuple[float, str, str]] = {
        "official_baseline": (481.53, STEP_PROFILE, "/synthesis/constants/official_baseline"),
        "ceiling_lambda1":   (520.9527323111674, FEAS, "/synthesis/constants/lambda1_ceil"),
        "K_cal":             (125.268, STEP_PROFILE, "/synthesis/constants/K_cal"),
        "E_T_anchor":        (3.844, STEP_PROFILE, "/synthesis/constants/E_T_deployed"),
        "step_us":           (1218.2, STEP_PROFILE, "/synthesis/constants/step_us"),
        "tau":               (1.218, STEP_PROFILE, "/synthesis/constants/tau"),
        "honest500_floor":   (3.9914, FEAS, "/synthesis/constants/honest500_floor"),
        "E_T_central":       (6.1112149873699195, STEP_PROFILE, "/synthesis/regime_bracket/corrected_central"),
        "bracket_lo":        (5.363610726985671, STEP_PROFILE, "/synthesis/regime_bracket/bracket_lo"),
        "bracket_hi":        (6.858819247754167, STEP_PROFILE, "/synthesis/regime_bracket/bracket_hi"),
        "mult_lo":           (1.744676699335575, STEP_PROFILE, "/synthesis/measured_multiplier_faithful"),
        "mult_hi":           (4.161395297380165, STEP_PROFILE, "/synthesis/collapse_additive/multiplier_vs_linear"),
        "private_factor":    (0.804, FEAS, "/synthesis/constants/priv_factor"),
        "priv_ET_decode":    (3.0898055282313592, PRIVGAP, "/synthesis/axis1_per_position/priv_ET"),
        "priv_ET_pooled":    (3.6553609619014926, PRIVGAP, "/synthesis/reconciliation/sglang_anchors/priv_pooled"),
        "private_verified":  (460.85, READCUT, "/private_verified_tps"),
        "read_companion_ratio": (-2.01774820233227, READCUT, "/read_cut_realization_ratio"),
        "companion_residual":(0.9979165670218526, SAM, "/stacking/residual_band_banked_target/residual_point"),
        "sam_lift_lo_pct":   (0.7932288368645539, SAM, "/sam_marginal_under_eagle3_pct_lower"),
        "sam_lift_hi_pct":   (1.5864576737291078, SAM, "/sam_marginal_under_eagle3_pct"),
        "vram_resident_gib": (20.100143778324128, VRAM, "/synthesis/eagle3_build_resident_gb"),
        "vram_headroom_gib": (3.899856221675872, VRAM, "/synthesis/vram_headroom_gb"),
        "build_cost_gpu_hr": (107.46577676190476, BUILDCOST, "/synthesis/build_cost/total_build_gpu_hours"),
        "build_cost_headroom_gpu_hr": (92.53422323809524, BUILDCOST, "/synthesis/budget_verdict/budget_headroom_gpu_hours"),
        "sigma_hw":          (4.864468814937121, SIGMA, "/legs/sigma_hw"),
    }
    cache: dict[str, dict] = {}
    out: dict[str, Any] = {}
    verify: list[dict[str, Any]] = []
    for name, (cited, path, jpath) in spec.items():
        if path not in cache:
            cache[path] = _load(path)
        src_val = float(_dig(cache[path], jpath))
        err = abs(src_val - cited)
        out[name] = src_val                       # use the SOURCE value (cited is for audit)
        verify.append({
            "name": name, "cited": cited, "source_value": src_val, "abs_err": err,
            "matches_source": bool(err <= IMPORT_TOL),
            "source": os.path.relpath(path, REPO_ROOT) + "#" + jpath,
        })
    out["_verify"] = verify
    out["_all_match"] = bool(all(v["matches_source"] for v in verify))

    # --- private_factor band: grounded in the banked private E[T] pair (ubel #258). The deterministic
    #     card uses the CONSERVATIVE decode factor 0.804; the band extends UP to the pooled-protocol
    #     factor (milder OOD drop). Both edges are banked private E[T] / public E[T] ratios. ---
    out["private_factor_decode"] = out["priv_ET_decode"] / out["E_T_anchor"]    # ~0.8038 (== 0.804)
    out["private_factor_pooled"] = out["priv_ET_pooled"] / out["E_T_anchor"]    # ~0.9510 (milder)
    out["private_factor_band"] = [out["private_factor"],
                                  max(out["private_factor"], out["private_factor_pooled"])]
    out["deployed_realized_factor"] = out["private_verified"] / out["official_baseline"]  # 0.9571 context

    out["fern_floor_public"] = FERN_FLOOR
    out["provenance"] = (
        "wirbel#295 (c334qaqu) corrected target 6.1112 + bracket [5.3636,6.8588] + tps_public x "
        "wirbel#290 ceiling 520.953 / priv_factor 0.804 / honest500 3.9914 x ubel#258 private E[T] "
        "[3.0898 decode, 3.6554 pooled] x lawine#296 SAM residual 0.998 / marginal [0.79,1.59]% x "
        "ubel#299 (jnoss7id) VRAM 20.10/3.90 x denken#301 (b4zg7b6c) build 107.47 GPU-hr x "
        "fern#302 (8jewx2ur) read companion -2.0177 / private_verified 460.85 x kanna#159 sigma_hw 4.864.")
    return out


# --------------------------------------------------------------------------- #
# Step 2 -- the deterministic card (the corrected-target invariant + the E[T] sweep).
# --------------------------------------------------------------------------- #
def build_deterministic_card(C: dict) -> dict[str, Any]:
    et_c = C["E_T_central"]
    step_us = C["step_us"]
    pf = C["private_factor"]
    step_central = step_for_corrected(et_c, step_us)        # the heavier step at which 6.1112 nets 500

    # (b) reproduce -- do not assume -- the corrected-target invariant.
    public_at_central = tps_public(et_c, step_central)
    invariant_resid = abs(public_at_central - TARGET)

    # sweep E[T] across the #295 bracket at the central (corrected) denominator.
    lo, hi = C["bracket_lo"], C["bracket_hi"]
    grid = []
    n = 25
    pinned = [lo, et_c, hi, C["honest500_floor"]]           # pin the bracket edges + honest floor
    ets = sorted(set([lo + (hi - lo) * i / (n - 1) for i in range(n)] + [p for p in pinned if lo <= p <= hi]))
    for et in ets:
        pub = tps_public(et, step_central)
        grid.append({
            "E_T": et,
            "public_tps": pub,
            "private_tps": pf * pub,
            "public_clears_500": bool(pub >= TARGET),
            "private_clears_500": bool(pf * pub >= TARGET),
        })

    # closed-form crossings (public linear in E[T] at fixed step: pub = TARGET * E_T / et_c).
    et_public_cross_500 = et_c                              # by construction
    et_private_cross_500 = et_c / pf                        # private = pf * TARGET * et / et_c
    private_at_central = pf * TARGET
    private_at_bracket_hi = pf * tps_public(hi, step_central)

    return {
        "step_central_us": step_central,
        "public_at_central_E_T": public_at_central,
        "invariant_reproduces_500": bool(invariant_resid <= ROUNDTRIP_TOL),
        "invariant_residual": invariant_resid,
        "sweep_vs_E_T": grid,
        "et_public_cross_500": et_public_cross_500,
        "et_private_cross_500": et_private_cross_500,            # TEST metric (the binding one)
        "private_cross_above_bracket": bool(et_private_cross_500 > hi),
        "private_at_central_tps": private_at_central,
        "private_at_bracket_hi_tps": private_at_bracket_hi,
        "private_binding": bool(et_private_cross_500 > et_public_cross_500),
        "statement": (
            "Public clears 500 at E[T]=%.4f BY CONSTRUCTION of the corrected target (invariant "
            "residual %.2e). Under the worst-case OOD private factor x%.3f, PRIVATE needs E[T]=%.4f "
            "-- %s the #295 bracket top %.4f -- so PRIVATE is the binding axis: at the central "
            "corrected target the private projection is %.1f TPS, and even at the bracket top it is "
            "only %.1f TPS." % (
                et_public_cross_500, invariant_resid, pf, et_private_cross_500,
                "ABOVE" if et_private_cross_500 > hi else "within",
                hi, private_at_central, private_at_bracket_hi)),
    }


# --------------------------------------------------------------------------- #
# Step 3 -- the tornado (rank each priced axis by its impact on projected PRIVATE TPS).
# --------------------------------------------------------------------------- #
def _project_private(C: dict, et: float, step_us_den: float, pf: float, comp_lift_pct: float) -> float:
    return pf * tps_public(et * (1.0 + comp_lift_pct / 100.0), step_us_den)


def build_tornado(C: dict) -> dict[str, Any]:
    et_c = C["E_T_central"]
    step_us = C["step_us"]
    step_central = step_for_corrected(et_c, step_us)
    pf_c = C["private_factor"]
    central = _project_private(C, et_c, step_central, pf_c, 0.0)   # 402.0 (companion excluded at center)

    axes: list[dict[str, Any]] = []

    # (A) achieved E[T] across the #295 corrected-target bracket (step held central).
    lo_priv = _project_private(C, C["bracket_lo"], step_central, pf_c, 0.0)
    hi_priv = _project_private(C, C["bracket_hi"], step_central, pf_c, 0.0)
    axes.append({
        "axis": "E_T", "band": [C["bracket_lo"], C["bracket_hi"]], "central": et_c,
        "private_at_lo": lo_priv, "private_at_hi": hi_priv,
        "delta_tps": abs(hi_priv - lo_priv), "favorable_dir": "higher E[T]",
        "sign_ok": bool(hi_priv > lo_priv),     # higher E[T] -> higher TPS
        "note": "build's achieved public E[T] across the #295 bracket (step at central regime).",
    })

    # (B) step-multiplier regime band [1.745x, 4.161x] -> heavier step lowers TPS. Parametrised by
    #     the equivalent corrected-target c in [bracket_lo, bracket_hi]; reported also as a multiplier.
    step_light = step_for_corrected(C["bracket_lo"], step_us)   # multiplicative-lower regime (light step)
    step_heavy = step_for_corrected(C["bracket_hi"], step_us)   # additive-upper regime (heavy step)
    light_priv = _project_private(C, et_c, step_light, pf_c, 0.0)
    heavy_priv = _project_private(C, et_c, step_heavy, pf_c, 0.0)
    axes.append({
        "axis": "step_multiplier", "band": [C["mult_lo"], C["mult_hi"]], "central": 0.5 * (C["mult_lo"] + C["mult_hi"]),
        "step_band_us": [step_light, step_heavy],
        "private_at_light_step": light_priv, "private_at_heavy_step": heavy_priv,
        "delta_tps": abs(light_priv - heavy_priv), "favorable_dir": "lighter step",
        "sign_ok": bool(light_priv > heavy_priv),   # heavier step -> lower TPS
        "note": "EAGLE-3 draft-step regime [multiplicative 1.745x, additive 4.161x]; CORRELATED with "
                "the E_T axis (both trace to the #295 corrected-target bracket) -- see honest caveat.",
    })

    # (C) private_factor band [0.804 decode, 0.951 pooled] (public held at central 500).
    pub_c = tps_public(et_c, step_central)
    pf_lo, pf_hi = C["private_factor_band"]
    axes.append({
        "axis": "private_factor", "band": [pf_lo, pf_hi], "central": pf_c,
        "private_at_lo": pf_lo * pub_c, "private_at_hi": pf_hi * pub_c,
        "delta_tps": abs(pf_hi - pf_lo) * pub_c, "favorable_dir": "milder OOD drop (higher factor)",
        "sign_ok": bool(pf_hi > pf_lo),
        "note": "priv/pub E[T] ratio; [decode 0.804 conservative, pooled 0.951 milder] (ubel #258).",
    })

    # (D) sigma_hw single-draw hardware variance (+- on the realized TPS).
    axes.append({
        "axis": "sigma_hw", "band": [-C["sigma_hw"], C["sigma_hw"]], "central": 0.0,
        "private_at_lo": central - C["sigma_hw"], "private_at_hi": central + C["sigma_hw"],
        "delta_tps": 2.0 * C["sigma_hw"], "favorable_dir": "favorable hardware draw",
        "sign_ok": True,
        "note": "kanna #159 single-draw hardware sigma 4.864 TPS (additive on the measured TPS).",
    })

    # (E) SAM companion marginal E[T] lift [0.79, 1.59]% (residual EAGLE-3 must cover alone 0.998).
    comp_lo = _project_private(C, et_c, step_central, pf_c, C["sam_lift_lo_pct"])
    comp_hi = _project_private(C, et_c, step_central, pf_c, C["sam_lift_hi_pct"])
    axes.append({
        "axis": "sam_companion", "band": [C["sam_lift_lo_pct"], C["sam_lift_hi_pct"]],
        "central": C["sam_lift_lo_pct"], "residual_eagle3_alone": C["companion_residual"],
        "private_at_lo": comp_lo, "private_at_hi": comp_hi,
        "delta_tps": abs(comp_hi - comp_lo), "favorable_dir": "more companion lift",
        "sign_ok": bool(comp_hi > comp_lo),
        "note": "lawine #296: SAM adds [0.79,1.59]% E[T] under a better drafter; residual 0.998 "
                "means EAGLE-3 covers ~all of the raise alone.",
    })

    ranked = sorted(axes, key=lambda a: a["delta_tps"], reverse=True)
    top = ranked[0]["axis"]
    return {
        "central_private_tps": central,
        "axes": ranked,
        "top_axis": top,                                     # TEST metric (go_card_tornado_top_axis)
        "top_axis_delta_tps": ranked[0]["delta_tps"],
        "all_signs_correct": bool(all(a["sign_ok"] for a in axes)),
        "ranking": [(a["axis"], a["delta_tps"]) for a in ranked],
        "statement": (
            "The axis whose banked uncertainty most threatens the 500 line is '%s' (Delta %.1f private "
            "TPS). The E_T bracket and step_multiplier are correlated views of the same #295 step-"
            "regime uncertainty; private_factor is the largest INDEPENDENT axis." % (top, ranked[0]["delta_tps"])),
    }


# --------------------------------------------------------------------------- #
# Step 4 -- P(clear 500) Monte-Carlo over the triangular banked bands.
# --------------------------------------------------------------------------- #
def _triangular(rng, lo, mode, hi, n):
    import numpy as np
    if hi - lo < 1e-12:
        return np.full(n, mode)
    mode = min(max(mode, lo), hi)
    return rng.triangular(lo, mode, hi, size=n)


def monte_carlo_pclear(C: dict, n: int, seed: int = 20260615) -> dict[str, Any]:
    import numpy as np
    rng = np.random.default_rng(seed)
    et_c = C["E_T_central"]
    step_us = C["step_us"]
    lo, hi = C["bracket_lo"], C["bracket_hi"]
    pf_lo, pf_hi = C["private_factor_band"]
    floor_anchor = FERN_FLOOR

    # draws (independent triangulars per PR step 4).
    et = _triangular(rng, lo, et_c, hi, n)                                   # achieved public E[T]
    creg = _triangular(rng, lo, et_c, hi, n)                                 # step-regime corrected-target
    pf = _triangular(rng, pf_lo, pf_lo, pf_hi, n)                            # OOD factor, mode at conservative
    comp = _triangular(rng, C["sam_lift_lo_pct"], C["sam_lift_hi_pct"], C["sam_lift_hi_pct"], n) / 100.0
    hw_pub = rng.normal(0.0, C["sigma_hw"], n)
    hw_pri = rng.normal(0.0, C["sigma_hw"], n)

    step = creg * step_us / floor_anchor                                    # heavier step from the regime draw
    public = TARGET * (et * (1.0 + comp) / floor_anchor) * (step_us / step)  # banked tps_public, vectorised
    public_obs = public + hw_pub
    private_obs = pf * public + hw_pri

    p_public = float(np.mean(public_obs >= TARGET))
    p_private = float(np.mean(private_obs >= TARGET))
    # conditional: E[T] reaches its central 6.11 (upper half of the bracket).
    cond = et >= et_c
    p_private_cond = float(np.mean(private_obs[cond] >= TARGET)) if cond.any() else float("nan")

    # COUPLED sensitivity (honest caveat): achieved E[T] tracks the regime's corrected target
    # (the "exactly reaches the corrected target" reading) -> public pins at 500, regime cancels.
    public_coupled = TARGET * (et * (1.0 + comp) / floor_anchor) * (step_us / (et * step_us / floor_anchor))
    private_coupled = pf * public_coupled + hw_pri
    p_public_coupled = float(np.mean(public_coupled + hw_pub >= TARGET))
    p_private_coupled = float(np.mean(private_coupled >= TARGET))

    monotone = _mc_monotone_in_et(C, rng)
    return {
        "n_draws": int(n),
        "p_public_clears_500": p_public,
        "p_private_clears_500": p_private,                       # TEST metric
        "p_private_clears_500_given_E_T_central": p_private_cond,
        "coupled_p_public_clears_500": p_public_coupled,
        "coupled_p_private_clears_500": p_private_coupled,
        "public_mean_tps": float(np.mean(public)),
        "public_p05_tps": float(np.percentile(public, 5)),
        "public_p95_tps": float(np.percentile(public, 95)),
        "private_mean_tps": float(np.mean(pf * public)),
        "private_p05_tps": float(np.percentile(pf * public, 5)),
        "private_p95_tps": float(np.percentile(pf * public, 95)),
        "p_private_monotone_in_E_T": monotone,
        "statement": (
            "Independent triangular bands (PR step 4): P(public>=500)=%.3f, P(private>=500)=%.3f, "
            "P(private>=500 | E[T]>=6.11)=%.3f. COUPLED (E[T] tracks the regime's corrected target): "
            "P(public>=500)=%.3f, P(private>=500)=%.3f -- the regime cancels and private is pinned "
            "sub-500, so the independent number is an UPPER bound on the favorable-corner probability." % (
                p_public, p_private, p_private_cond, p_public_coupled, p_private_coupled)),
    }


def _mc_monotone_in_et(C: dict, rng) -> bool:
    """P(private>=500) is monotone non-decreasing as the E[T] band shifts up (self-test d)."""
    import numpy as np
    step_central = step_for_corrected(C["E_T_central"], C["step_us"])
    pf = C["private_factor"]
    ps = []
    for shift in (-0.5, -0.25, 0.0, 0.25, 0.5):
        et = _triangular(rng, C["bracket_lo"] + shift, C["E_T_central"] + shift, C["bracket_hi"] + shift, 40_000)
        public = TARGET * (et / FERN_FLOOR) * (C["step_us"] / step_central)
        ps.append(float(np.mean(pf * public >= TARGET)))
    return bool(all(ps[i + 1] >= ps[i] - 1e-3 for i in range(len(ps) - 1)))


# --------------------------------------------------------------------------- #
# Step 5 -- honest caveat + the priced-axes scorecard (context: every priced axis cleared).
# --------------------------------------------------------------------------- #
def build_scorecard(C: dict) -> dict[str, Any]:
    rows = [
        {"axis": "(a) STEP cost", "source": "wirbel #295 (c334qaqu)", "status": "PRICED",
         "value": "corrected target E[T]=%.4f, bracket [%.4f, %.4f]" % (C["E_T_central"], C["bracket_lo"], C["bracket_hi"])},
        {"axis": "(b) COMPANION floor", "source": "lawine #296", "status": "PRICED",
         "value": "SAM marginal [0.79,1.59]%%; residual EAGLE-3 alone %.4f" % C["companion_residual"]},
        {"axis": "(c) PER-POSITION target", "source": "kanna #289 / denken #297", "status": "PRICED",
         "value": "uniform spec, cliff at position 1 (prompt-invariant)"},
        {"axis": "(e) VRAM fit", "source": "ubel #299 (jnoss7id)", "status": "GREEN",
         "value": "resident %.2f GiB, headroom %.2f GiB vs 24 hard" % (C["vram_resident_gib"], C["vram_headroom_gib"])},
        {"axis": "(h) BUILD cost", "source": "denken #301 (b4zg7b6c)", "status": "GO",
         "value": "%.2f A10G-GPU-hr, headroom %.2f under <=200 lane" % (C["build_cost_gpu_hr"], C["build_cost_headroom_gpu_hr"])},
        {"axis": "(i) READ companion", "source": "fern #302 (8jewx2ur)", "status": "BUILD-ALONE",
         "value": "read-cut realization ratio %.4f (regresses on wall) -> build alone" % C["read_companion_ratio"]},
    ]
    return {"priced_axes": rows, "n_priced": len(rows)}


def build_honest_caveat(C: dict, card: dict, mc: dict) -> dict[str, Any]:
    return {
        "conditional_on_reachability": (
            "This card is CONDITIONAL on numerator reachability: whether a trained EAGLE-3 drafter "
            "ACTUALLY delivers public E[T]=6.1112 is the in-flight kanna #294 / wirbel #303 / denken "
            "#304 lane. The card prices 'IF reachable, here is the number and the risk', NOT whether "
            "6.11 is reachable."),
        "private_binding_under_0804": (
            "Under the conservative OOD factor x0.804 (priv/pub E[T] 3.0898/3.844, ubel #258), PRIVATE "
            "is the binding constraint: it crosses 500 only at public E[T]=%.4f, ABOVE the #295 bracket "
            "top %.4f and pressing the K+1=8 acceptance ceiling. The private upside comes from a milder "
            "OOD drop (pooled-protocol factor %.4f) -- the largest INDEPENDENT tornado axis." % (
                card["et_private_cross_500"], C["bracket_hi"], C["private_factor_pooled"])),
        "et_step_correlation": (
            "The E_T-bracket and step-multiplier tornado axes are CORRELATED views of the SAME #295 "
            "step-regime uncertainty (the bracket was derived by mapping the regime through the "
            "corrected-target formula). The independent-MC P(clear) treats them as orthogonal (the PR "
            "step-4 convention, conservative on variance / optimistic on the favorable corner); the "
            "COUPLED-MC pins public at 500 and gives P(private>=500)=%.3f -- both reported." % mc["coupled_p_private_clears_500"]),
        "private_factor_band_basis": (
            "No banked sampling CI exists on the OOD factor; the band [%.4f, %.4f] is BOUNDED by the "
            "two banked private E[T] protocols (decode 3.0898 conservative, pooled 3.6554 milder, ubel "
            "#258), not a sampling interval. The deployed-realized public->private drop was milder "
            "still (%.4f = 460.85/481.53), reported for context." % (
                C["private_factor_band"][0], C["private_factor_band"][1], C["deployed_realized_factor"])),
        "deterministic_convention": (
            "The build projection uses wirbel #295's banked tps_public (the 4.966 public-E[T] floor "
            "convention that DEFINES the corrected target and round-trips to 500). The deployed anchor "
            "481.53 @ E[T]=3.844 is the K_cal convention; honest500 E[T]=3.9914 is the deployed-step "
            "E[T] for 500 (the corrected targets add the heavier-step inflation)."),
        "scope": (
            "LOCAL CPU-only synthesis. 0 TPS added; BASELINE 481.53 untouched; greedy/PPL untouched. "
            "NO GPU / vLLM / HF Job / submission / served-file change / official draw. Authorizes "
            "NOTHING. NOT a launch recommendation and NOT a measured TPS."),
    }


# --------------------------------------------------------------------------- #
# Step 6 -- self-test (PRIMARY metric).
# --------------------------------------------------------------------------- #
def self_test(C: dict, card: dict, tornado: dict, mc: dict) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # (a) every imported constant matches its source <=1e-6.
    a_ok = C["_all_match"]
    results["a_imports_match_source"] = {
        "pass": bool(a_ok),
        "max_abs_err": max((v["abs_err"] for v in C["_verify"]), default=0.0),
        "n_constants": len(C["_verify"]),
        "mismatches": [v["name"] for v in C["_verify"] if not v["matches_source"]],
    }

    # (b) deterministic projection at 6.1112 reproduces the 500 line.
    b_ok = card["invariant_reproduces_500"] and card["invariant_residual"] <= ROUNDTRIP_TOL
    results["b_invariant_reproduces_500"] = {
        "pass": bool(b_ok), "public_at_central": card["public_at_central_E_T"],
        "residual": card["invariant_residual"]}

    # (c) tornado signs: higher E[T] -> higher TPS; heavier step -> lower TPS; all axes signed right.
    c_ok = tornado["all_signs_correct"]
    results["c_tornado_signs_correct"] = {
        "pass": bool(c_ok), "signs": {a["axis"]: a["sign_ok"] for a in tornado["axes"]}}

    # (d) P(clear) in [0,1] and monotone non-decreasing in E[T].
    probs = [mc["p_public_clears_500"], mc["p_private_clears_500"],
             mc["coupled_p_public_clears_500"], mc["coupled_p_private_clears_500"]]
    d_ok = all(0.0 <= p <= 1.0 for p in probs) and mc["p_private_monotone_in_E_T"]
    results["d_pclear_bounded_and_monotone"] = {
        "pass": bool(d_ok), "probs_in_unit_interval": bool(all(0.0 <= p <= 1.0 for p in probs)),
        "monotone_in_E_T": mc["p_private_monotone_in_E_T"]}

    # (e) public->private mapping consistent (x0.804): private == private_factor * public at central.
    et_c, step_us = C["E_T_central"], C["step_us"]
    step_central = step_for_corrected(et_c, step_us)
    pub = tps_public(et_c, step_central)
    e_ok = abs(card["private_at_central_tps"] - C["private_factor"] * pub) <= 1e-9
    results["e_public_private_x0804_consistent"] = {
        "pass": bool(e_ok), "private_factor": C["private_factor"],
        "private_at_central": card["private_at_central_tps"], "factor_times_public": C["private_factor"] * pub}

    # (f) NaN-clean across every reported numeric.
    payload_numeric = {"card": card, "tornado": tornado, "mc": mc,
                       "constants": {k: v for k, v in C.items() if not k.startswith("_") and _finite(v)}}
    nan_paths = _nan_paths(payload_numeric, "selftest")
    f_ok = (len(nan_paths) == 0)
    results["f_nan_clean"] = {"pass": bool(f_ok), "nan_paths": nan_paths}

    # (g) caveats carried (conditional-on-reachability + not-a-launch present, BASELINE untouched).
    cav = build_honest_caveat(C, card, mc)
    g_ok = ("CONDITIONAL" in cav["conditional_on_reachability"]
            and "NOT a launch" in cav["scope"] and "481.53 untouched" in cav["scope"])
    results["g_caveats_carried"] = {"pass": bool(g_ok)}

    passes = bool(all(v["pass"] for v in results.values()))
    return {"eagle3_go_card_self_test_passes": passes, "conditions": results}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(args) -> dict[str, Any]:
    t0 = time.time()
    C = load_constants()
    card = build_deterministic_card(C)
    tornado = build_tornado(C)
    mc = monte_carlo_pclear(C, args.mc_draws)
    scorecard = build_scorecard(C)
    caveat = build_honest_caveat(C, card, mc)
    st = self_test(C, card, tornado, mc)

    constants_public = {k: v for k, v in C.items() if not k.startswith("_")}
    payload = {
        "pr": 305, "agent": "fern", "kind": "eagle3_go_card",
        "primary_metric_name": "eagle3_go_card_self_test_passes",
        "eagle3_go_card_self_test_passes": st["eagle3_go_card_self_test_passes"],
        "test_metric_names": ["go_card_tornado_top_axis", "p_private_clears_500", "et_private_cross_500"],
        "go_card_tornado_top_axis": tornado["top_axis"],
        "p_private_clears_500": mc["p_private_clears_500"],
        "et_private_cross_500": card["et_private_cross_500"],
        "deterministic_card": card,
        "tornado": tornado,
        "monte_carlo": mc,
        "priced_axes_scorecard": scorecard,
        "honest_caveat": caveat,
        "self_test": st,
        "import_verification": C["_verify"],
        "all_imports_match_source": C["_all_match"],
        "constants": constants_public,
        "provenance": C["provenance"],
        "elapsed_sec": time.time() - t0,
        "peak_mem_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "nan_clean": len(_nan_paths({"card": card, "tornado": tornado, "mc": mc})) == 0,
    }
    return payload


# --------------------------------------------------------------------------- #
# wandb + main
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if args.no_wandb:
        return
    try:
        import wandb
    except Exception as exc:                       # noqa: BLE001
        print(f"[wandb] unavailable ({exc}); skipping.", file=sys.stderr)
        return
    C = payload["constants"]
    card, tor, mc = payload["deterministic_card"], payload["tornado"], payload["monte_carlo"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group,
                     config={
                         "pr": 305, "agent": "fern", "kind": "eagle3_go_card",
                         "official_baseline": C["official_baseline"], "ceiling_lambda1": C["ceiling_lambda1"],
                         "E_T_central": C["E_T_central"], "bracket_lo": C["bracket_lo"], "bracket_hi": C["bracket_hi"],
                         "private_factor": C["private_factor"], "private_factor_pooled": C["private_factor_pooled"],
                         "sigma_hw": C["sigma_hw"], "companion_residual": C["companion_residual"],
                         "vram_resident_gib": C["vram_resident_gib"], "build_cost_gpu_hr": C["build_cost_gpu_hr"],
                         "read_companion_ratio": C["read_companion_ratio"], "mc_draws": payload["monte_carlo"]["n_draws"],
                         "provenance": payload["provenance"],
                         "scope": payload["honest_caveat"]["scope"],
                     })
    summary = {
        "eagle3_go_card_self_test_passes": payload["eagle3_go_card_self_test_passes"],
        "go_card_tornado_top_axis": payload["go_card_tornado_top_axis"],
        "p_private_clears_500": payload["p_private_clears_500"],
        "et_private_cross_500": payload["et_private_cross_500"],
        "et_public_cross_500": card["et_public_cross_500"],
        "p_public_clears_500": mc["p_public_clears_500"],
        "p_private_clears_500_given_E_T_central": mc["p_private_clears_500_given_E_T_central"],
        "coupled_p_private_clears_500": mc["coupled_p_private_clears_500"],
        "top_axis_delta_tps": tor["top_axis_delta_tps"],
        "private_at_central_tps": card["private_at_central_tps"],
        "private_at_bracket_hi_tps": card["private_at_bracket_hi_tps"],
        "invariant_residual": card["invariant_residual"],
        "all_imports_match_source": payload["all_imports_match_source"],
        "max_import_abs_err": payload["self_test"]["conditions"]["a_imports_match_source"]["max_abs_err"],
        "elapsed_sec": payload["elapsed_sec"], "peak_mem_mib": payload["peak_mem_mib"],
    }
    for k, v in list(summary.items()):
        if isinstance(v, float) and not math.isfinite(v):
            summary[k] = None
    wandb.log(summary)
    wandb.summary.update({k: v for k, v in summary.items() if v is not None})

    tor_tbl = wandb.Table(columns=["axis", "delta_private_tps", "band_lo", "band_hi", "sign_ok"])
    for a in tor["axes"]:
        tor_tbl.add_data(a["axis"], a["delta_tps"], a["band"][0], a["band"][1], bool(a["sign_ok"]))
    sweep_tbl = wandb.Table(columns=["E_T", "public_tps", "private_tps", "public_clears_500", "private_clears_500"])
    for r in card["sweep_vs_E_T"]:
        sweep_tbl.add_data(r["E_T"], r["public_tps"], r["private_tps"],
                           bool(r["public_clears_500"]), bool(r["private_clears_500"]))
    imp_tbl = wandb.Table(columns=["constant", "cited", "source_value", "abs_err", "matches", "source"])
    for v in payload["import_verification"]:
        imp_tbl.add_data(v["name"], v["cited"], v["source_value"], v["abs_err"], bool(v["matches_source"]), v["source"])
    wandb.log({"tornado": tor_tbl, "deterministic_sweep": sweep_tbl, "import_verification": imp_tbl})

    art = wandb.Artifact("eagle3_go_card_results", type="analysis")
    with art.new_file("eagle3_go_card_results.json", mode="w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    run.log_artifact(art)
    wandb.finish()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR #305 EAGLE-3 GO/NO-GO decision card")
    ap.add_argument("--out", default="research/validity/eagle3_go_card/eagle3_go_card_results.json")
    ap.add_argument("--mc-draws", "--mc_draws", type=int, default=MC_DRAWS)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/eagle3-go-card")
    ap.add_argument("--wandb-group", "--wandb_group", default="eagle3-go-card")
    args = ap.parse_args(argv)

    payload = run(args)
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)

    st = payload["self_test"]
    print("eagle3_go_card_self_test_passes =", st["eagle3_go_card_self_test_passes"])
    for cond, v in st["conditions"].items():
        print(f"  {cond}: {'PASS' if v['pass'] else 'FAIL'}")
    print("\nTEST metrics:")
    print("  go_card_tornado_top_axis =", payload["go_card_tornado_top_axis"])
    print("  p_private_clears_500     =", round(payload["p_private_clears_500"], 4))
    print("  et_private_cross_500     =", round(payload["et_private_cross_500"], 4))
    print("\nDETERMINISTIC:", payload["deterministic_card"]["statement"])
    print("\nTORNADO ranking (axis: Delta private TPS):")
    for axis, d in payload["tornado"]["ranking"]:
        print(f"    {axis:<16} {d:8.2f}")
    print("\nMONTE-CARLO:", payload["monte_carlo"]["statement"])
    nan_paths = _nan_paths(payload, "payload")
    if nan_paths:
        print("\n[WARN] NaN paths:", nan_paths, file=sys.stderr)
    _maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
