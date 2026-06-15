#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #310 (student fern) -- Does E[T]=6.11 clear PRIVATE 500 under the per-position model?

WHAT THIS LEG RECONCILES
------------------------
Two MERGED banked runs price the EAGLE-3 build's PRIVATE projection with conventions that give
OPPOSITE GO/NO-GO verdicts on the sole >500 path:

  * fern #305 GO-card (m4nmtdl9): scalar OOD factor x0.804 -> "private 402 @ E[T]=6.11, sub-500,
    PRIVATE binding, NO-GO" (private-500 crossing only at public E[T]=7.601, beyond the #295 bracket).
  * lawine #300 (8t5q6sr0): the 0.804 is the RAW width-1 spine ratio; the deployed M=8 verify-tree
    recovers deep-position fidelity to a PER-POSITION ratio rho_priv_e3=0.9421 (c_1=1.0 held,
    c_{j>=2}=0.97135), under which private-500 needs only public E[T]~=4.19 -> the build's 6.11 clears.

This leg re-pins the GO-card's private axis under #300's per-position model and reports private TPS
at E[T]=6.11 under BOTH conventions side by side, on a SINGLE consistent public base.

THE RECONCILIATION (the crux: a hidden double-application of 0.804 in #305's "public" axis)
-------------------------------------------------------------------------------------------
#305's tps_public is anchored to fern #281's 4.966 "public-E[T] floor". But the banked constants
make 4.966 the SCALAR-PRIVATE-500 floor, not a public-500 floor:

    honest public-500 floor  = 500 / K_cal       = 3.99146   (= banked honest500_floor 3.9914)
    fern #281 "public" floor  = 3.99146 / 0.804   = 4.9645    (~ banked fern_floor_public 4.966)

So #305's "public" already carries ONE 0.804 OOD haircut: tps_public(E[T]=3.844) = 387.0 =
0.804 x 481.53 (the MEASURED public). #305 then multiplies by private_factor 0.804 a SECOND time,
so its "private 402 @ 6.11" = honest_public(622) x 0.804^2 -- the OOD tax applied TWICE.

The HONEST single-tax reconciliation puts BOTH private models on the K_cal public projection at the
#295 corrected (heavier) step, honest_public(E[T]) = K_cal * E[T] * step_us/step_central:

    honest_public(6.1112)            = 622.08 TPS         (NOT 500; the deployed measured priv/pub is
                                                           0.957 tree-recovered, not the raw 0.804)
    scalar      0.804  x 622.08      = 500.2  (break-even, ~the bar)
    per-position 0.9421 x 622.08     = 586.1  (CLEARS, +86 / +17.2%)   <-- headline
    tree-recov  0.955  x 622.08      = 594.1  (clears)
    raw/no-rec  0.7797 x 622.08      = 485.1  (MISS, -15)
    break-even rho                   = 500/622.08 = 0.8037 (~ the scalar/raw boundary)

VERDICT: under the per-position model the build E[T]=6.11 CLEARS private 500 (586.1, +17.2%) -> GO
on private. This CONFIRMS lawine #300 and REFUTES the GO-card's "sub-500 NO-GO" (402), whose
pessimism is a DOUBLE-COUNT of the raw width-1 spine tax. The NO-GO survives ONLY under the
raw/no-tree-recovery worst case (0.7797 -> 485).

HONEST CAVEAT (carried): rho_priv_e3=0.9421 is MODELED from the deployed LINEAR spine's deep-position
fidelity (lawine #300); a trained {2,21,39}-fusion EAGLE-3 draft may carry a DIFFERENT deep-position
private tax. This reconciles the two banked conventions; it does NOT measure the fusion draft's
private tax (checkpoint-gated follow-up). The build's E[T]=6.11 reachability is CONDITIONAL (denken
#304 a1-cliff break, out of scope). Greedy-identity / PPL compliance is orthogonal (Issue #192).

SCOPE. LOCAL CPU-only reconciliation over banked constants. 0 TPS added; BASELINE 481.53 untouched;
greedy/PPL untouched. NO GPU / vLLM / HF Job / submission / served-file change. Authorizes NOTHING.

PRIMARY metric  private_perposition_reconcile_self_test_passes
TEST    metrics private_tps_at_611_perposition (float) + build_clears_private500_perposition (bool)

Run:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_private_perposition_reconcile/eagle3_private_perposition_reconcile.py \\
        --wandb_group eagle3-private-reconcile --wandb_name fern/eagle3-private-reconcile
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V = REPO_ROOT / "research" / "validity"

# Source banked runs (import-not-rederive). Each constant cites file + JSON path.
STEP_PROFILE = V / "eagle3_step_profile/eagle3_step_profile_results.json"        # wirbel #295
FEAS = V / "eagle3_feasibility_bracket/eagle3_feasibility_bracket_results.json"  # wirbel #290
GO_CARD = V / "eagle3_go_card/eagle3_go_card_results.json"                       # fern #305
PRIV_BAR = V / "private_bar_eagle3/private_bar_eagle3_results.json"              # lawine #300
PRIV_BAR_PY = V / "private_bar_eagle3/private_bar_eagle3.py"                     # #300 live machinery

TARGET = 500.0
IMPORT_TOL = 1e-6     # self-test (1): every imported constant matches its source to <=1e-6
REPRO_TOL = 1e-6      # self-test (a)/(b): reproduce #305 / #300 banked values to <=1e-6
ETBAR_TOL = 1e-3      # self-test (c): per-position private-500 public E[T] within 1e-3 of #300's 4.19


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dig(d: Any, path: str) -> Any:
    cur = d
    for seg in path.strip("/").split("/"):
        cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
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


# --------------------------------------------------------------------------- #
# Step 1 -- import every banked constant + self-verify to <=1e-6.
# --------------------------------------------------------------------------- #
def load_constants() -> dict[str, Any]:
    # (cited value, source file, source JSON path). The loader reads the source and the self-test
    # asserts |cited - source| <= 1e-6; the SOURCE value is used downstream.
    spec: dict[str, tuple[float, Path, str]] = {
        "official_baseline": (481.53, STEP_PROFILE, "/synthesis/constants/official_baseline"),
        "K_cal":             (125.268, STEP_PROFILE, "/synthesis/constants/K_cal"),
        "step_us":           (1218.2, STEP_PROFILE, "/synthesis/constants/step_us"),
        "tau":               (1.218, STEP_PROFILE, "/synthesis/constants/tau"),
        "E_T_deployed":      (3.844, STEP_PROFILE, "/synthesis/constants/E_T_deployed"),
        "fern_floor_public": (4.966, STEP_PROFILE, "/synthesis/constants/fern_floor_public"),
        "E_T_central":       (6.1112149873699195, STEP_PROFILE, "/synthesis/regime_bracket/corrected_central"),
        "bracket_lo":        (5.363610726985671, STEP_PROFILE, "/synthesis/regime_bracket/bracket_lo"),
        "bracket_hi":        (6.858819247754167, STEP_PROFILE, "/synthesis/regime_bracket/bracket_hi"),
        "honest500_floor":   (3.9914, FEAS, "/synthesis/constants/honest500_floor"),
        "private_factor":    (0.804, FEAS, "/synthesis/constants/priv_factor"),
        "private_verified":  (460.85, PRIV_BAR, "/synthesis/imported/private_verified_tps"),
        # lawine #300 per-position model anchors (the reconciliation target).
        "rho_priv_e3":       (0.9421228821714434, PRIV_BAR, "/synthesis/step3_eagle3/rho_priv_e3"),
        "rho_priv_e3_raw":   (0.7797221674962985, PRIV_BAR, "/synthesis/step3_eagle3/rho_priv_e3_raw"),
        "rho_deployed_tree": (0.955267407218773, PRIV_BAR, "/synthesis/step1_anchor/rho_deployed_tree_recovered"),
        "c_deep_300":        (0.9713472759982902, PRIV_BAR, "/synthesis/step1_anchor/c_deep_calibrated"),
        "public_etbar_priv500_300": (4.194301321486128, PRIV_BAR, "/synthesis/step2_invert/public_etbar_for_private_500"),
        "eagle3_public_et_300": (4.96600000000002, PRIV_BAR, "/synthesis/step3_eagle3/eagle3_public_et"),
        # fern #305 scalar projection banked values (the reproduce-round-trip target).
        "go305_private_central": (402.0, GO_CARD, "/deterministic_card/private_at_central_tps"),
        "go305_private_brackethi": (451.17793160534995, GO_CARD, "/deterministic_card/private_at_bracket_hi_tps"),
        "go305_et_private_cross": (7.601013665882984, GO_CARD, "/deterministic_card/et_private_cross_500"),
        "go305_public_central": (499.99999999999994, GO_CARD, "/deterministic_card/public_at_central_E_T"),
    }
    cache: dict[Path, dict] = {}
    out: dict[str, Any] = {}
    verify: list[dict[str, Any]] = []
    for name, (cited, path, jpath) in spec.items():
        if path not in cache:
            cache[path] = _load(path)
        src_val = float(_dig(cache[path], jpath))
        err = abs(src_val - cited)
        out[name] = src_val
        verify.append({
            "name": name, "cited": cited, "source_value": src_val, "abs_err": err,
            "matches_source": bool(err <= IMPORT_TOL),
            "source": str(path.relative_to(REPO_ROOT)) + "#" + jpath,
        })
    out["_verify"] = verify
    out["_all_match"] = bool(all(v["matches_source"] for v in verify))

    # The collapse bracket the robustness sweep spans (lawine #300 banked): raw/no-recovery
    # (EAGLE-3 gets NO tree salvage) -> per-position (deployed-effective) -> tree-recovered
    # (the deployed system's actual measured ratio).
    out["collapse_bracket"] = {
        "raw_no_recovery": out["rho_priv_e3_raw"],     # 0.7797
        "scalar_0804": out["private_factor"],          # 0.804  (raw width-1 spine aggregate)
        "per_position": out["rho_priv_e3"],            # 0.9421 (PR headline)
        "tree_recovered": out["rho_deployed_tree"],    # 0.955
    }
    # The #295 corrected (heavier-EAGLE-3-step) regime denominator fern #305 used.
    out["step_central_us"] = out["E_T_central"] * out["step_us"] / out["fern_floor_public"]   # 1499.13
    out["K_cal_eagle3"] = out["K_cal"] * out["step_us"] / out["step_central_us"]              # 101.79
    out["deployed_measured_priv_over_pub"] = out["private_verified"] / out["official_baseline"]  # 0.957
    out["provenance"] = (
        "wirbel#295 (c334qaqu) K_cal 125.268 / step_us 1218.2 / tau 1.218 / E_T_dep 3.844 / "
        "fern_floor 4.966 / corrected target 6.1112 bracket [5.3636,6.8588] x wirbel#290 honest500 "
        "3.9914 / priv_factor 0.804 x lawine#300 (8t5q6sr0) rho_priv_e3 0.9421 / c_deep 0.97135 / "
        "raw 0.7797 / tree 0.955 / etbar 4.194 x fern#305 (m4nmtdl9) scalar private 402.0/451.2 / "
        "cross 7.601 x private_verified 460.85.")
    return out


# --------------------------------------------------------------------------- #
# Step 2 -- reproduce fern #305's scalar-private projection (round-trip <=1e-6).
# --------------------------------------------------------------------------- #
def public_305(C: dict, et: float) -> float:
    """fern #305's banked tps_public at the corrected (heavier) step, where it round-trips to 500 at
    the corrected central target by construction: 500 * (et / E_T_central). NOTE this 'public' is
    anchored to fern #281's 4.966 floor, which is the SCALAR-PRIVATE-500 floor (= 3.99146/0.804),
    so this curve already carries one 0.804 OOD haircut relative to the K_cal public projection."""
    return TARGET * et / C["E_T_central"]


def reproduce_305_scalar(C: dict) -> dict[str, Any]:
    pf = C["private_factor"]
    et_c, hi = C["E_T_central"], C["bracket_hi"]

    public_central = public_305(C, et_c)                  # 500.0 by construction
    private_central = pf * public_305(C, et_c)            # 402.0
    private_brackethi = pf * public_305(C, hi)            # 451.18
    et_private_cross = et_c / pf                          # 7.601 (public E[T] where scalar priv = 500)

    repro = {
        "public_at_central": public_central,
        "private_at_central": private_central,
        "private_at_bracket_hi": private_brackethi,
        "et_private_cross_500": et_private_cross,
        "private_cross_above_bracket": bool(et_private_cross > hi),
        # round-trip residuals vs #305's banked JSON values (<=1e-6).
        "resid_private_central": abs(private_central - C["go305_private_central"]),
        "resid_private_brackethi": abs(private_brackethi - C["go305_private_brackethi"]),
        "resid_et_cross": abs(et_private_cross - C["go305_et_private_cross"]),
        "resid_public_central": abs(public_central - C["go305_public_central"]),
    }
    repro["reproduces_305"] = bool(
        repro["resid_private_central"] <= REPRO_TOL
        and repro["resid_private_brackethi"] <= REPRO_TOL
        and repro["resid_et_cross"] <= REPRO_TOL
        and repro["resid_public_central"] <= REPRO_TOL)
    return repro


# --------------------------------------------------------------------------- #
# Step 3 -- reproduce lawine #300's per-position model (rho_priv_e3=0.9421 <=1e-6).
# --------------------------------------------------------------------------- #
def reproduce_300_perposition(C: dict) -> tuple[dict[str, Any], Any, dict]:
    """Import #300's live machinery and re-run its synthesis; reproduce rho_priv_e3, c_deep, and the
    private-500 public E[T]bar (~4.19). Returns (repro dict, module, synthesis dict)."""
    sys.path.insert(0, str(REPO_ROOT))
    m = _import("private_bar_eagle3", PRIV_BAR_PY)
    syn = m.synthesize()
    s1, s2, s3 = syn["step1_anchor"], syn["step2_invert"], syn["step3_eagle3"]

    rho = s3["rho_priv_e3"]
    rho_raw = s3["rho_priv_e3_raw"]
    c_deep = s1["c_deep_calibrated"]
    etbar = s2["public_etbar_for_private_500"]

    repro = {
        "rho_priv_e3": rho,
        "rho_priv_e3_raw": rho_raw,
        "c_deep_calibrated": c_deep,
        "c1_held": 1.0,
        "eagle3_public_et": s3["eagle3_public_et"],
        "eagle3_private_et_deployed": s3["eagle3_private_et_deployed"],
        "public_etbar_for_private_500": etbar,
        "private_tps_deployed_anchor": s1["private_tps_deployed"],   # ~460.85
        # round-trip residuals vs #300's banked JSON values (<=1e-6).
        "resid_rho_priv_e3": abs(rho - C["rho_priv_e3"]),
        "resid_c_deep": abs(c_deep - C["c_deep_300"]),
        "resid_etbar": abs(etbar - C["public_etbar_priv500_300"]),
    }
    repro["reproduces_300_rho"] = bool(repro["resid_rho_priv_e3"] <= REPRO_TOL)
    repro["reproduces_300_etbar"] = bool(repro["resid_etbar"] <= ETBAR_TOL)
    return repro, m, syn


def perposition_ratio_at_public_et(m, c_deep: float, public_et: float) -> float:
    """PROFILE-RECOMPUTED per-position ratio at a target public (physical-acceptance) E[T]: build the
    smallest-deviation acceptance profile achieving public_et (deep-flat if <=ceiling, else a1-lifted
    uniform), apply #300's deployed-effective collapse (c_1=1.0 held, c_deep on j>=2), return
    private_E[T]/public_E[T]. (The flat 0.9421 anchor is this ratio AT public_et=4.966.)"""
    prof = m.profile_for_public_et(public_et)
    priv_cond = m.apply_deployed_collapse(prof["cond"], c_deep)
    return m.et_from_cond(priv_cond) / m.et_from_cond(prof["cond"])


# --------------------------------------------------------------------------- #
# Step 4 -- the HONEST single-tax reconciliation (the headline).
# --------------------------------------------------------------------------- #
def honest_public(C: dict, et: float) -> float:
    """K_cal public projection at the #295 corrected (heavier-EAGLE-3) step -- NO OOD haircut:
    honest_public(et) = K_cal * et * (step_us / step_central). At et=6.1112 this is 622.08 TPS
    (= deployed E[T]=4.966-equivalent public; the corrected target was derived to net this)."""
    return C["K_cal"] * et * (C["step_us"] / C["step_central_us"])


def reconcile(C: dict, m, c_deep: float) -> dict[str, Any]:
    et_c = C["E_T_central"]
    hp_c = honest_public(C, et_c)                          # 622.08
    bracket = C["collapse_bracket"]

    # Confirm the hidden haircut: #305's "public" = private_factor x honest_public (one 0.804 tax).
    haircut_ratio = public_305(C, et_c) / hp_c             # ~0.804
    # And #305's reported private double-taxes: honest x 0.804^2.
    go305_private_as_double_tax = hp_c * C["private_factor"] ** 2

    # Apply each private model ONCE to the honest public base.
    def private_at(rho: float, et: float) -> float:
        return rho * honest_public(C, et)

    def et_cross(rho: float) -> float:
        # honest_public(et)*rho = 500 -> et = 500 / (rho * K_cal_eagle3)
        return TARGET / (rho * C["K_cal_eagle3"])

    models = {}
    for label, rho in [("raw_no_recovery", bracket["raw_no_recovery"]),
                       ("scalar_0804", bracket["scalar_0804"]),
                       ("per_position", bracket["per_position"]),
                       ("tree_recovered", bracket["tree_recovered"])]:
        ptps = private_at(rho, et_c)
        models[label] = {
            "rho": rho,
            "private_tps_at_611": ptps,
            "clears_private_500": bool(ptps >= TARGET),
            "margin_tps": ptps - TARGET,
            "margin_pct": 100.0 * (ptps - TARGET) / TARGET,
            "et_private_cross_500": et_cross(rho),
            "cross_within_bracket": bool(et_cross(rho) <= C["bracket_hi"]),
            "cross_below_bracket_lo": bool(et_cross(rho) < C["bracket_lo"]),
        }

    # PROFILE-RECOMPUTED per-position (honesty cross-check; the build's 6.11 exceeds the deep-flat
    # ceiling 6.105 so the profile is a1-lifted/uniform -> rho differs slightly from the flat 0.9421).
    rho_recomp_611 = perposition_ratio_at_public_et(m, c_deep, et_c)
    private_perpos_recomp_611 = rho_recomp_611 * hp_c

    # Map #300's "private-500 needs public E[T]~=4.19" into the build's frame: the build's honest
    # public 622.08 == deployed E[T]-equivalent 622.08/K_cal = 4.966, which is >> 4.19 -> clears.
    build_deployed_equiv_et = hp_c / C["K_cal"]            # 4.966
    confirms_300_etbar = bool(build_deployed_equiv_et > C["public_etbar_priv500_300"])

    # Break-even private model: rho at which private-at-6.11 == 500.
    breakeven_rho = TARGET / hp_c                          # 0.8037

    head_rho = bracket["per_position"]
    headline_private_611 = private_at(head_rho, et_c)      # 586.06
    return {
        "honest_public_at_611": hp_c,
        "honest_public_definition": "K_cal * E[T] * step_us/step_central (corrected heavier step)",
        "go305_public_haircut_ratio": haircut_ratio,       # ~0.804 (one hidden tax in #305's 'public')
        "go305_private_402_is_honest_x_0804_squared": go305_private_as_double_tax,
        "models": models,
        "headline_private_tps_at_611_perposition": headline_private_611,
        "build_clears_private500_perposition": bool(headline_private_611 >= TARGET),
        "perposition_margin_tps": headline_private_611 - TARGET,
        "perposition_margin_pct": 100.0 * (headline_private_611 - TARGET) / TARGET,
        "perposition_profile_recomputed_rho_at_611": rho_recomp_611,
        "perposition_profile_recomputed_private_611": private_perpos_recomp_611,
        "build_deployed_equiv_public_et": build_deployed_equiv_et,
        "private500_public_etbar_300": C["public_etbar_priv500_300"],
        "confirms_300_private500_needs_4p19": confirms_300_etbar,
        "breakeven_rho": breakeven_rho,
        "breakeven_at_scalar_anchor": bool(abs(breakeven_rho - C["private_factor"]) < 0.01),
        "statement": (
            "Honest public(6.11)=%.2f TPS (K_cal, corrected step). Apply the OOD tax ONCE: scalar "
            "0.804 -> %.1f (break-even), per-position 0.9421 -> %.1f (CLEARS +%.1f%%), tree 0.955 -> "
            "%.1f, raw 0.7797 -> %.1f (miss). #305's 'public 500' = honest x %.4f (one hidden 0.804 "
            "haircut via the fern #281 4.966 floor), so its 'private 402' = honest x 0.804^2 "
            "(double-count). Under the per-position model the build E[T]=6.11 CLEARS private 500." % (
                hp_c, models["scalar_0804"]["private_tps_at_611"],
                models["per_position"]["private_tps_at_611"], models["per_position"]["margin_pct"],
                models["tree_recovered"]["private_tps_at_611"],
                models["raw_no_recovery"]["private_tps_at_611"], haircut_ratio)),
    }


# --------------------------------------------------------------------------- #
# Step 5 -- robustness: sweep the private-model bracket x the #295 E[T] bracket.
# --------------------------------------------------------------------------- #
def robustness(C: dict) -> dict[str, Any]:
    et_c, lo, hi = C["E_T_central"], C["bracket_lo"], C["bracket_hi"]
    bracket = C["collapse_bracket"]
    rho_grid = [bracket["raw_no_recovery"], bracket["scalar_0804"],
                bracket["per_position"], bracket["tree_recovered"]]

    # (1) private-model sweep at the central build target 6.11.
    sweep_models = []
    for rho in sorted(set(rho_grid)):
        ptps = rho * honest_public(C, et_c)
        sweep_models.append({"rho": rho, "private_tps_at_611": ptps, "clears": bool(ptps >= TARGET)})
    clears_flags = [r["clears"] for r in sweep_models]
    robust_across_models = bool(all(clears_flags))
    holds_for_milder_only = bool(any(clears_flags) and not all(clears_flags))

    # (2) E[T] bracket sweep at the per-position rho (does the build clear across [5.36, 6.86]?).
    n = 13
    sweep_et = []
    rho_pp = bracket["per_position"]
    for i in range(n):
        et = lo + (hi - lo) * i / (n - 1)
        pub = honest_public(C, et)
        sweep_et.append({
            "E_T": et, "honest_public": pub,
            "private_perpos": rho_pp * pub, "private_scalar": C["private_factor"] * pub,
            "perpos_clears": bool(rho_pp * pub >= TARGET),
            "scalar_clears": bool(C["private_factor"] * pub >= TARGET),
        })
    perpos_clears_whole_bracket = bool(all(r["perpos_clears"] for r in sweep_et))

    # (3) break-even private model (rho where private-at-6.11 == 500) and its bracket membership.
    breakeven_rho = TARGET / honest_public(C, et_c)
    breakeven_in_bracket = bool(bracket["raw_no_recovery"] - 1e-9 <= breakeven_rho
                                <= bracket["tree_recovered"] + 1e-9)

    return {
        "private_model_sweep_at_611": sweep_models,
        "robust_across_collapse_bracket": robust_across_models,
        "holds_for_milder_factors_only": holds_for_milder_only,
        "clears_for_rho_at_or_above": breakeven_rho,
        "et_bracket_sweep_perposition": sweep_et,
        "perposition_clears_whole_et_bracket": perpos_clears_whole_bracket,
        "breakeven_rho": breakeven_rho,
        "breakeven_in_collapse_bracket": breakeven_in_bracket,
        "statement": (
            "Across the collapse bracket [raw 0.7797, scalar 0.804, per-pos 0.9421, tree 0.955] the "
            "build clears private-500 @ 6.11 for every model with rho >= break-even %.4f -- i.e. for "
            "per-position and tree-recovered (CLEAR) but NOT raw/no-tree-recovery (485, MISS). The "
            "per-position projection clears across the ENTIRE #295 E[T] bracket [%.4f, %.4f] (cross "
            "at E[T]=%.3f, below the bracket floor). The NO-GO is NOT robust: it requires both the "
            "double-count AND the no-tree-recovery worst case." % (
                breakeven_rho, lo, hi, TARGET / (rho_pp * C["K_cal_eagle3"]))),
    }


# --------------------------------------------------------------------------- #
# Step 6 -- honest caveats (carried explicitly).
# --------------------------------------------------------------------------- #
def caveats() -> dict[str, str]:
    return {
        "rho_modeled_from_linear_spine": (
            "rho_priv_e3=0.9421 is MODELED from the deployed LINEAR spine's deep-position fidelity "
            "(lawine #300: c_1=1.0 held by the M=8 tree, c_deep=0.97135 on j>=2 calibrated to the "
            "organizer-verified 460.85). A trained {2,21,39}-fusion EAGLE-3 draft may carry a "
            "DIFFERENT deep-position private tax."),
        "reconciles_not_measures": (
            "This leg RECONCILES the two banked private conventions onto one honest public base; it "
            "does NOT measure the fusion draft's private tax. That is the checkpoint-gated follow-up "
            "scoped by lawine's fusion-rank-coverage probe."),
        "reachability_conditional": (
            "The build's E[T]=6.11 reachability is CONDITIONAL and OUT OF SCOPE (denken #304 a1-cliff "
            "break; the deep-flat ceiling is 6.105 so 6.11 needs an a1 lift). This leg prices the "
            "PRIVATE projection IF 6.11 is reached, not whether it is reachable."),
        "greedy_ppl_orthogonal": (
            "Greedy-identity / PPL compliance is orthogonal (Issue #192 lane) and OUT OF SCOPE."),
        "scope": (
            "LOCAL CPU-only reconciliation over banked constants. 0 TPS added; BASELINE 481.53 "
            "untouched; greedy/PPL untouched. NO GPU / vLLM / HF Job / submission / served-file "
            "change. Authorizes NOTHING. NOT a launch."),
    }


# --------------------------------------------------------------------------- #
# Step 7 -- self-test (PRIMARY metric).
# --------------------------------------------------------------------------- #
def self_test(C: dict, rep305: dict, rep300: dict, rec: dict, rob: dict, cav: dict) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # (1) every imported constant matches its source <=1e-6.
    results["01_imports_match_source"] = {
        "pass": bool(C["_all_match"]),
        "max_abs_err": max((v["abs_err"] for v in C["_verify"]), default=0.0),
        "mismatches": [v["name"] for v in C["_verify"] if not v["matches_source"]],
    }

    # (a) scalar projection reproduces #305's 402.0 / 451.2 + 7.601 cross <=1e-6.
    results["a_scalar_reproduces_305"] = {
        "pass": bool(rep305["reproduces_305"]),
        "resid_private_central": rep305["resid_private_central"],
        "resid_private_brackethi": rep305["resid_private_brackethi"],
        "resid_et_cross": rep305["resid_et_cross"],
    }

    # (b) per-position model reproduces #300's rho_priv_e3=0.9421 <=1e-6.
    results["b_perposition_reproduces_300_rho"] = {
        "pass": bool(rep300["reproduces_300_rho"]),
        "resid_rho_priv_e3": rep300["resid_rho_priv_e3"],
        "rho_priv_e3": rep300["rho_priv_e3"],
    }

    # (c) per-position private-500 public E[T]bar reproduces #300's ~4.19 within tol.
    results["c_perposition_etbar_reproduces_419"] = {
        "pass": bool(rep300["reproduces_300_etbar"]),
        "resid_etbar": rep300["resid_etbar"],
        "etbar": rep300["public_etbar_for_private_500"],
    }

    # (d) both maps monotone non-decreasing in E[T].
    grid = [C["bracket_lo"] + (C["bracket_hi"] - C["bracket_lo"]) * i / 19 for i in range(20)]
    scal = [C["private_factor"] * honest_public(C, et) for et in grid]
    perp = [C["rho_priv_e3"] * honest_public(C, et) for et in grid]
    mono_scal = all(scal[i + 1] >= scal[i] - 1e-9 for i in range(len(scal) - 1))
    mono_perp = all(perp[i + 1] >= perp[i] - 1e-9 for i in range(len(perp) - 1))
    results["d_both_maps_monotone_in_et"] = {
        "pass": bool(mono_scal and mono_perp), "scalar_monotone": bool(mono_scal),
        "perposition_monotone": bool(mono_perp)}

    # (e) break-even private model lies in the banked collapse bracket [0.7797, 0.955] and ~0.804.
    be = rec["breakeven_rho"]
    in_bracket = bool(C["rho_priv_e3_raw"] - 1e-9 <= be <= C["rho_deployed_tree"] + 1e-9)
    near_scalar = bool(abs(be - C["private_factor"]) < 0.01)
    results["e_breakeven_in_bracket"] = {
        "pass": bool(in_bracket and near_scalar), "breakeven_rho": be,
        "in_collapse_bracket": in_bracket, "near_scalar_anchor_0804": near_scalar}

    # (f) NaN-clean across every reported numeric.
    payload_numeric = {"rep305": rep305, "rep300": rep300, "reconcile": rec, "robustness": rob,
                       "constants": {k: v for k, v in C.items() if not k.startswith("_") and _finite(v)}}
    nan_paths = _nan_paths(payload_numeric, "selftest")
    results["f_nan_clean"] = {"pass": bool(len(nan_paths) == 0), "nan_paths": nan_paths}

    # (g) caveats carried (modeled-from-linear-spine + reconciles-not-measures + reachability +
    #     greedy/PPL orthogonal + BASELINE untouched).
    g_ok = ("MODELED" in cav["rho_modeled_from_linear_spine"]
            and "does NOT measure" in cav["reconciles_not_measures"]
            and "CONDITIONAL" in cav["reachability_conditional"]
            and "orthogonal" in cav["greedy_ppl_orthogonal"]
            and "481.53" in cav["scope"] and "NOT a launch" in cav["scope"])
    results["g_caveats_carried"] = {"pass": bool(g_ok)}

    passes = bool(all(v["pass"] for v in results.values()))
    return {"private_perposition_reconcile_self_test_passes": passes, "conditions": results}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    C = load_constants()
    rep305 = reproduce_305_scalar(C)
    rep300, m300, _syn300 = reproduce_300_perposition(C)
    rec = reconcile(C, m300, rep300["c_deep_calibrated"])
    rob = robustness(C)
    cav = caveats()
    st = self_test(C, rep305, rep300, rec, rob, cav)

    constants_public = {k: v for k, v in C.items() if not k.startswith("_")}
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 310, "agent": "fern", "kind": "eagle3_private_perposition_reconcile",
        "analysis_only": True,
        "primary_metric_name": "private_perposition_reconcile_self_test_passes",
        "private_perposition_reconcile_self_test_passes": st["private_perposition_reconcile_self_test_passes"],
        "test_metric_names": ["private_tps_at_611_perposition", "build_clears_private500_perposition"],
        "private_tps_at_611_perposition": rec["headline_private_tps_at_611_perposition"],
        "build_clears_private500_perposition": rec["build_clears_private500_perposition"],
        "reproduce_305_scalar": rep305,
        "reproduce_300_perposition": rep300,
        "reconcile_honest_single_tax": rec,
        "robustness": rob,
        "caveats": cav,
        "self_test": st,
        "import_verification": C["_verify"],
        "all_imports_match_source": C["_all_match"],
        "constants": constants_public,
        "provenance": C["provenance"],
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    payload["nan_clean"] = len(_nan_paths({"r": rep305, "p": rep300, "c": rec, "b": rob})) == 0
    return payload


# --------------------------------------------------------------------------- #
# console report
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    rec, rob, st = payload["reconcile_honest_single_tax"], payload["robustness"], payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PR #310 -- Does E[T]=6.11 clear PRIVATE 500 under the per-position model?", flush=True)
    print("=" * 100, flush=True)
    print("REPRODUCE #305 scalar (round-trip <=1e-6):", flush=True)
    r5 = payload["reproduce_305_scalar"]
    print(f"  public(6.11)={r5['public_at_central']:.4f}  private(6.11)={r5['private_at_central']:.4f}"
          f"  private(6.8588)={r5['private_at_bracket_hi']:.4f}  cross@E[T]={r5['et_private_cross_500']:.4f}"
          f"  [reproduces_305={r5['reproduces_305']}]", flush=True)
    print("REPRODUCE #300 per-position (<=1e-6):", flush=True)
    r3 = payload["reproduce_300_perposition"]
    print(f"  rho_priv_e3={r3['rho_priv_e3']:.7f}  c_deep={r3['c_deep_calibrated']:.7f}"
          f"  etbar(priv-500)={r3['public_etbar_for_private_500']:.4f}"
          f"  [rho_ok={r3['reproduces_300_rho']} etbar_ok={r3['reproduces_300_etbar']}]", flush=True)
    print("-" * 100, flush=True)
    print(f"HONEST public(6.11) = {rec['honest_public_at_611']:.2f} TPS "
          f"(#305 'public 500' = honest x {rec['go305_public_haircut_ratio']:.4f}; one hidden 0.804 haircut)",
          flush=True)
    print("  model            rho      private@6.11   clears   margin     cross E[T]", flush=True)
    for label, mm in rec["models"].items():
        print(f"  {label:<15} {mm['rho']:.4f}   {mm['private_tps_at_611']:8.2f}    "
              f"{str(mm['clears_private_500']):<5}   {mm['margin_tps']:+7.1f}   {mm['et_private_cross_500']:.4f}",
              flush=True)
    print(f"  HEADLINE per-position private@6.11 = {payload['private_tps_at_611_perposition']:.2f} "
          f"-> CLEARS={payload['build_clears_private500_perposition']} "
          f"(+{rec['perposition_margin_pct']:.1f}%)  [profile-recomputed {rec['perposition_profile_recomputed_private_611']:.1f}]",
          flush=True)
    print(f"  confirms #300 'priv-500 needs public E[T]~4.19': build deployed-equiv E[T]="
          f"{rec['build_deployed_equiv_public_et']:.4f} > 4.194 -> {rec['confirms_300_private500_needs_4p19']}",
          flush=True)
    print(f"  break-even rho = {rec['breakeven_rho']:.4f} (~scalar anchor 0.804)", flush=True)
    print("-" * 100, flush=True)
    print("ROBUSTNESS:", rob["statement"], flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) private_perposition_reconcile_self_test_passes = "
          f"{st['private_perposition_reconcile_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"   - {k}: {'PASS' if v['pass'] else 'FAIL'}", flush=True)
    print(f"nan_clean={payload['nan_clean']}  peak_mem_mib={payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# wandb logging (robust; never fatal)
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[reconcile] wandb logging unavailable: {exc}", flush=True)
        return None

    C = payload["constants"]
    rec, rob, st = payload["reconcile_honest_single_tax"], payload["robustness"], payload["self_test"]
    run = init_wandb_run(
        job_type="validity-analytic", agent="fern", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-private-reconcile", "validity-analytic", "private-bar", "eagle3",
              "per-position-collapse", "go-no-go", "bank-the-analysis"],
        config={
            "pr": 310, "K_cal": C["K_cal"], "step_us": C["step_us"], "step_central_us": C["step_central_us"],
            "E_T_central": C["E_T_central"], "fern_floor_public": C["fern_floor_public"],
            "honest500_floor": C["honest500_floor"], "private_factor": C["private_factor"],
            "rho_priv_e3": C["rho_priv_e3"], "rho_priv_e3_raw": C["rho_priv_e3_raw"],
            "rho_deployed_tree": C["rho_deployed_tree"], "private_verified": C["private_verified"],
            "provenance": payload["provenance"], "scope": payload["caveats"]["scope"],
        },
    )
    if run is None:
        print("[reconcile] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "private_perposition_reconcile_self_test_passes":
            int(bool(st["private_perposition_reconcile_self_test_passes"])),
        "private_tps_at_611_perposition": payload["private_tps_at_611_perposition"],
        "build_clears_private500_perposition": int(bool(payload["build_clears_private500_perposition"])),
        "honest_public_at_611": rec["honest_public_at_611"],
        "go305_public_haircut_ratio": rec["go305_public_haircut_ratio"],
        "perposition_margin_tps": rec["perposition_margin_tps"],
        "perposition_margin_pct": rec["perposition_margin_pct"],
        "perposition_profile_recomputed_private_611": rec["perposition_profile_recomputed_private_611"],
        "build_deployed_equiv_public_et": rec["build_deployed_equiv_public_et"],
        "confirms_300_private500_needs_4p19": int(bool(rec["confirms_300_private500_needs_4p19"])),
        "breakeven_rho": rec["breakeven_rho"],
        "robust_across_collapse_bracket": int(bool(rob["robust_across_collapse_bracket"])),
        "perposition_clears_whole_et_bracket": int(bool(rob["perposition_clears_whole_et_bracket"])),
        "scalar_private_at_611": rec["models"]["scalar_0804"]["private_tps_at_611"],
        "raw_private_at_611": rec["models"]["raw_no_recovery"]["private_tps_at_611"],
        "tree_private_at_611": rec["models"]["tree_recovered"]["private_tps_at_611"],
        "max_import_abs_err": st["conditions"]["01_imports_match_source"]["max_abs_err"],
        "nan_clean": int(bool(payload["nan_clean"])), "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v["pass"])) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_private_perposition_reconcile_result",
                      artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[reconcile] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PR #310 EAGLE-3 private per-position reconcile")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-private-reconcile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    payload = run()
    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_private_perposition_reconcile_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[reconcile] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    gate = bool(payload["private_perposition_reconcile_self_test_passes"])
    print(f"  PRIMARY private_perposition_reconcile_self_test_passes = {gate}", flush=True)
    print(f"  TEST private_tps_at_611_perposition = "
          f"{payload['private_tps_at_611_perposition']:.2f}", flush=True)
    print(f"  TEST build_clears_private500_perposition = "
          f"{payload['build_clears_private500_perposition']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)
    if args.self_test:
        print(f"[reconcile] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
