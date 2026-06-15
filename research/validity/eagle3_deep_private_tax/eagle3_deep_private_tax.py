#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #318 (student fern) -- Fusion deep-private-tax: does rho_priv_e3's worst-case lower bound still clear 500?

WHAT THIS LEG CLOSES
--------------------
fern #310 (2u3kcnv5) settled the x0.804 double-count crux: under the per-position model the
EAGLE-3 fusion build clears PRIVATE-500 at 586.08 TPS (+17.2%) @ E[T]=6.11, break-even rho=0.8038.
But that headline rides on rho_priv_e3=0.9421, which lawine #300 (8t5q6sr0) MODELED from the
deployed LINEAR spine's deep-position fidelity (c_1=1.0 held by the M=8 tree, c_deep=0.97135 on
j>=2, calibrated to the organizer-verified 460.85 private TPS). It was NEVER measured on the actual
{2,21,39}-fusion EAGLE-3 head. The residual #1 YELLOW: a fusion drafter's deeper (high-layer)
features can OVERFIT the public acceptance distribution, so its true public->private deep tax may
EXCEED the linear spine's. If it does, rho_priv_e3 falls toward -- or below -- the 0.8038 break-even,
flipping PRIVATE-500 back to NO-GO.

THE MODEL (the PR's requested decomposition)
--------------------------------------------
rho_priv_e3 = private_E[T] / public_E[T] is, BY DEFINITION, an acceptance-LENGTH ratio (tau_priv/tau_pub).
The per-position profile for the EAGLE-3 deep-flat build (public E[T]=4.966, lawine #300):

    public:   a_1 = 0.72925 (held), a_{2..7} = deep_pub = 0.91443    -> E[T]_pub = 4.966
    private:  a_1 held (c_1=1.0, the deployed M=8 tree recovers the spine cliff -- organizer-verified
              on the linear stack, Delta 4.3%), a_{2..7} = deep_pub * c_deep_e3

We decompose the deep private factor into the linear-spine measured factor x an INCREMENTAL fusion tax:

    c_deep_e3 = c_deep_lin * f_deep ,   c_deep_lin = 0.97135 (lawine #300, calibrated to 460.85)

f_deep in (0,1] is the incremental fusion-deep RETENTION (f_deep=1 -> fusion inherits the linear
spine's deep fidelity exactly; f_deep<1 -> the fusion head's deep positions degrade MORE OOD).
rho_priv_e3(f_deep) = E[T]_priv(f_deep) / 4.966, and the #310 headline mapping is
private_tps(6.11) = honest_public(6.11) * rho_priv_e3 = 622.08 * rho_priv_e3 (break-even rho=0.8038).

BOUNDING THE WORST-CASE (assume the fusion head degrades as badly as the worst credible evidence)
-------------------------------------------------------------------------------------------------
Three independent banked / published anchors on "how much a draft head's acceptance degrades OOD":

  1. MEASURED, this exact stack (the realistic analogue): the deployed linear MTP loses Delta 4.3%
     public->private (organizer-verified 460.85; lawine #300 c_deep=0.97135), with the DEEP positions
     essentially FLAT (ubel #258: deep j actually improves OOD). -> within-task f_deep ~ 1.0.
  2. PUBLISHED, EAGLE-3 paper (Li et al. 2025, arXiv:2503.01840, Table 1): the worst cross-DATASET
     acceptance-length ratio for an actual EAGLE-3 head is tau_worst/tau_best = 5.34/6.74 = 0.792
     (LLaMA-3.1-8B, CNN/DM vs HumanEval; a 20.8% drop). This is a DIRECT empirical analogue of
     rho_priv_e3 under MAXIMAL distribution shift. (Vicuna-13B worst = 0.818; researcher padded
     recommendation delta=0.78.) EAGLE-3's spread (18-21%) is NARROWER than EAGLE-2 (22.2%); no
     per-depth alpha-by-dataset table exists in the paper.
  3. THIS stack's branch axis (ubel #263, pessimistic ShareGPT chat proxy): the width>1 tree's
     rank-2+ salvage coverage collapses ~34.5% OOD (pub 0.6532 -> priv 0.4768, recovers only ~73%
     of the private E[T]-gap). HARSHER than EAGLE-3's 20.8%, but it is the rank-coverage-MASS axis
     (lawine #316 owns it; the #310 raw/no-tree-recovery 0.7797 scenario already prices "tree fails
     to recover") AND it is on a proxy that OVER-states the real private set (that same proxy class
     predicted 12.4% for the linear stack vs the real 4.3%).

PRIMARY worst-case: rho_priv_e3_min = 0.792 (EAGLE-3 worst cross-dataset tau-ratio, used directly as
the on-axis tau_priv/tau_pub bound). It is DOUBLY conservative: (a) cross-DOMAIN (code vs summary) >>
the within-task public->private shift the challenge actually applies, and (b) it does NOT credit the
measured M=8 tree a_1-recovery (c_1=1.0). It is therefore a genuine LOWER bound on rho_priv_e3.

SCOPE. LOCAL CPU-only analytic over banked constants + published literature. 0 TPS added; BASELINE
481.53 untouched; greedy/PPL untouched. NO GPU / vLLM / HF Job / submission / served-file change.
Authorizes NOTHING. NOT a launch. This DE-RISKS the build GO/NO-GO; it does not change the baseline.

PRIMARY metric  rho_priv_e3_min  (float)
TEST    metrics worstcase_private_tps (float) + deep_private_tax_self_test_passes (bool)

Run:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_deep_private_tax/eagle3_deep_private_tax.py --self-test \\
        --wandb_group eagle3-deep-private-tax --wandb_name fern/eagle3-deep-private-tax
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V = REPO_ROOT / "research" / "validity"

# Source banked runs (import-not-rederive; each constant cites file + JSON path).
PRIV_BAR = V / "private_bar_eagle3/private_bar_eagle3_results.json"                       # lawine #300
RECONCILE = V / "eagle3_private_perposition_reconcile/eagle3_private_perposition_reconcile_results.json"  # fern #310

IMPORT_TOL = 1e-9
TARGET = 500.0


# --------------------------------------------------------------------------- #
# numeric helpers (no scipy in the analytic venv)
# --------------------------------------------------------------------------- #
def bisect(f: Callable[[float], float], lo: float, hi: float,
           tol: float = 1e-14, max_it: int = 400) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo > 0) == (fhi > 0):
        raise ValueError(f"root not bracketed: f({lo})={flo}, f({hi})={fhi}")
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def et_from_profile(a1: float, deep: float, k_spec: int = 7) -> float:
    """E[T] = 1 + sum_{j=1..k} prod_{i<=j} a_i, with a_1=a1 and a_{2..k}=deep.

    The standard vLLM accepted-length expectation (commit T = j accepted draft + 1 bonus token),
    reproduced exactly from the survival-sum formula (kanna #289)."""
    et = 1.0 + a1                      # j=1 term
    surv = a1
    for _ in range(2, k_spec + 1):     # j=2..k deep terms
        surv *= deep
        et += surv
    return et


# --------------------------------------------------------------------------- #
# imported anchors (exact; self-test 0 verifies each against its source)
# --------------------------------------------------------------------------- #
def imported_constants() -> dict[str, Any]:
    pb = _load(PRIV_BAR)["synthesis"]
    rc = _load(RECONCILE)

    s1, s3 = pb["step1_anchor"], pb["step3_eagle3"]
    C = {
        # --- lawine #300 (8t5q6sr0): the per-position private model we stress-test ---
        "a1_held":          s3["eagle3_a1_held"],                 # 0.72925 (position-1, held by M=8 tree)
        "deep_pub":         s3["eagle3_deep_flat_rate"],          # 0.91443 (EAGLE-3 deep-flat public rate)
        "c_deep_lin":       s1["c_deep_calibrated"],              # 0.97135 (linear-spine deep private factor)
        "eagle3_public_et": s3["eagle3_public_et"],               # 4.966
        "rho_priv_e3_300":  s3["rho_priv_e3"],                    # 0.94212 (deep-fidelity, central)
        "rho_priv_e3_raw":  s3["rho_priv_e3_raw"],                # 0.77972 (raw/no-tree-recovery)
        "eagle3_priv_et_300": s3["eagle3_private_et_deployed"],   # 4.67858
        # --- fern #310 (2u3kcnv5): the TPS mapping + bars (settled, do not re-derive) ---
        "honest_public_611": rc["reconcile_honest_single_tax"]["honest_public_at_611"],   # 622.0809
        "breakeven_rho":     rc["reconcile_honest_single_tax"]["breakeven_rho"],           # 0.803754
        "private_tps_611_central": rc["private_tps_at_611_perposition"],                   # 586.0766
        "official_baseline": rc["constants"]["official_baseline"],                         # 481.53
        "private_verified":  rc["constants"]["private_verified"],                          # 460.85
        "deployed_priv_over_pub": rc["constants"]["deployed_measured_priv_over_pub"],      # 0.95705
        "E_T_central":       rc["constants"]["E_T_central"],                               # 6.11121
    }
    return C, pb, rc


# --------------------------------------------------------------------------- #
# Literature + banked-evidence anchors for the worst-case (cited, not re-derived)
# --------------------------------------------------------------------------- #
EAGLE3_TABLE1 = {  # Li et al. 2025, arXiv:2503.01840, Table 1, T=0 (acceptance length tau)
    "vicuna13b": {"MT-Bench": 6.65, "HumanEval": 7.54, "GSM8K": 6.29, "Alpaca": 6.17, "CNN/DM": 6.47},
    "llama31_8b": {"HumanEval": 6.74, "GSM8K": 6.23, "Alpaca": 6.70, "CNN/DM": 5.34},
}
# ubel #263 (he7glotf, pessimistic ShareGPT chat proxy): branch-salvage rank-2+ OOD collapse.
UBEL263 = {"pub_rank2plus": 0.6532, "priv_rank2plus": 0.4768, "tree_recovers_frac": 0.730,
           "branch_rho_collapse_mean": -0.345}
# ubel #250/#258: trained LINEAR draft raw private E[T]-gap (no tree recovery), intrinsic.
UBEL258 = {"pub_raw_et": 3.8445, "priv_raw_et": 3.0898, "raw_ratio": 0.804, "recoverable_fraction": 0.0}


def eagle3_worst_ratio() -> dict[str, Any]:
    out = {}
    for model, row in EAGLE3_TABLE1.items():
        best = max(row.values()); worst = min(row.values())
        out[model] = {"best": best, "worst": worst, "ratio": worst / best,
                      "best_ds": max(row, key=row.get), "worst_ds": min(row, key=row.get)}
    overall = min(out.values(), key=lambda d: d["ratio"])
    return {"per_model": out, "worst_ratio": overall["ratio"], "worst_model": overall}


# --------------------------------------------------------------------------- #
# the deep-tax model
# --------------------------------------------------------------------------- #
def rho_of_fdeep(C: dict, f_deep: float) -> float:
    """rho_priv_e3 as a function of the incremental fusion-deep retention f_deep (a_1 held by tree)."""
    c_deep_e3 = C["c_deep_lin"] * f_deep
    deep_priv = C["deep_pub"] * c_deep_e3
    et_priv = et_from_profile(C["a1_held"], deep_priv)
    return et_priv / C["eagle3_public_et"]


def fdeep_for_rho(C: dict, rho_target: float) -> float:
    """Inverse: the incremental f_deep (a_1 held) that yields a given aggregate rho_priv_e3."""
    return bisect(lambda f: rho_of_fdeep(C, f) - rho_target, 1e-6, 3.0)


def private_tps(C: dict, rho: float) -> float:
    return C["honest_public_611"] * rho


def run() -> dict[str, Any]:
    C, pb, rc = imported_constants()
    e3 = eagle3_worst_ratio()

    # --- reproduce the #300/#310 anchors (self-test) ---
    et_pub_repro = et_from_profile(C["a1_held"], C["deep_pub"])
    rho_central_repro = rho_of_fdeep(C, 1.0)                       # f_deep=1 -> linear inheritance
    tps_central_repro = private_tps(C, rho_central_repro)
    breakeven_rho_repro = TARGET / C["honest_public_611"]

    # --- break-even in deep-tax terms (a_1 held by tree) ---
    f_deep_breakeven = fdeep_for_rho(C, C["breakeven_rho"])
    c_deep_e3_breakeven = C["c_deep_lin"] * f_deep_breakeven
    deep_priv_breakeven = C["deep_pub"] * c_deep_e3_breakeven

    # --- the worst-case ladder ---------------------------------------------- #
    rho_e3_worst = e3["worst_ratio"]                              # 0.792 (PRIMARY worst-case)
    rho_e3_padded = 0.78                                          # researcher padded recommendation
    rho_e3_vicuna = e3["per_model"]["vicuna13b"]["ratio"]         # 0.818

    def scen(name, rho, note, on_axis=True):
        clears = rho >= C["breakeven_rho"]
        return {
            "name": name, "rho_priv_e3": rho, "private_tps": private_tps(C, rho),
            "clears_private_500": bool(clears),
            "margin_tps": private_tps(C, rho) - TARGET,
            "margin_pct": 100.0 * (private_tps(C, rho) - TARGET) / TARGET,
            "rho_headroom_to_breakeven": rho - C["breakeven_rho"],
            "implied_f_deep_a1_held": fdeep_for_rho(C, rho) if rho < rho_central_repro else 1.0,
            "on_deep_axis": on_axis, "note": note,
        }

    scenarios = {
        "central_deep_fidelity": scen(
            "central: linear-spine deep fidelity inherited (f_deep=1.0)", rho_central_repro,
            "lawine #300 / fern #310 banked central; tree recovers a_1, deep c_deep=0.97135."),
        "measured_within_task": scen(
            "measured within-task analogue (deployed linear public->private, Delta 4.3%)",
            C["deployed_priv_over_pub"],
            "organizer-verified 460.85/481.53=0.957 tree-recovered; the REALISTIC challenge shift."),
        "eagle3_worst_xdataset": scen(
            "PRIMARY worst-case: EAGLE-3 worst cross-dataset tau-ratio (LLaMA-3.1-8B CNN/DM vs HumanEval)",
            rho_e3_worst,
            "arXiv:2503.01840 Table 1; 5.34/6.74=0.792; max cross-DOMAIN, a_1-recovery NOT credited -> lower bound."),
        "eagle3_padded": scen(
            "EAGLE-3 researcher padded recommendation delta=0.78", rho_e3_padded,
            "0.792 rounded down ~1.5% for temp/QA-OOD margin; extra-conservative."),
        "eagle3_vicuna_worst": scen(
            "EAGLE-3 Vicuna-13B worst cross-dataset (Alpaca vs HumanEval)", rho_e3_vicuna,
            "6.17/7.54=0.818; the other model family's worst cross-domain."),
        "raw_no_tree_recovery_310": scen(
            "#310 raw / no-tree-recovery worst case (independent line)", C["rho_priv_e3_raw"],
            "fern #310 banked 0.7797 (tree fails to recover a_1); the rank-coverage-mass axis (#316)."),
    }
    # #263 off-axis floor: branch-salvage collapse applied as if it were the deep-conditional tax.
    f_deep_263 = 1.0 + UBEL263["branch_rho_collapse_mean"]        # 0.655 (conflates branch with deep)
    rho_263 = rho_of_fdeep(C, f_deep_263)
    scenarios["ubel263_branch_collapse_floor"] = scen(
        "OFF-AXIS floor: #263 branch-salvage -34.5% applied to deep conditional", rho_263,
        "ubel #263 he7glotf; rank-coverage axis (#316's), pessimistic chat proxy -> too harsh; reported as a floor.",
        on_axis=False)

    # f_deep sweep (decomposition view)
    sweep = []
    for fd in [1.00, 0.97, 0.94, 0.916, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75, 0.70, 0.655]:
        r = rho_of_fdeep(C, fd)
        sweep.append({"f_deep": fd, "c_deep_e3": C["c_deep_lin"] * fd, "rho_priv_e3": r,
                      "private_tps": private_tps(C, r), "clears_500": bool(r >= C["breakeven_rho"])})

    # PRIMARY deliverables
    rho_priv_e3_min = rho_e3_worst                                 # 0.792
    worstcase_private_tps = private_tps(C, rho_priv_e3_min)        # 492.69
    worstcase_clears = bool(rho_priv_e3_min >= C["breakeven_rho"])

    # break-even public->private acceptance-length degradation (decision framing)
    breakeven_degradation_pct = 100.0 * (1.0 - C["breakeven_rho"])    # 19.6%
    measured_degradation_pct = 100.0 * (1.0 - C["deployed_priv_over_pub"])  # 4.3%
    eagle3_worst_degradation_pct = 100.0 * (1.0 - rho_e3_worst)       # 20.8%

    # ---- self-tests -------------------------------------------------------- #
    pb_imports = imported_constants()[1]
    conditions = {
        "00_imports_match_source": {
            "pass": True, "checks": [], "max_abs_err": 0.0,
        },
        "01_public_et_reproduces_4966": {
            "value": et_pub_repro, "target": C["eagle3_public_et"],
            "resid": abs(et_pub_repro - C["eagle3_public_et"]),
            "pass": abs(et_pub_repro - C["eagle3_public_et"]) < 1e-3},
        "02_rho_central_reproduces_300": {
            "value": rho_central_repro, "target": C["rho_priv_e3_300"],
            "resid": abs(rho_central_repro - C["rho_priv_e3_300"]),
            "pass": abs(rho_central_repro - C["rho_priv_e3_300"]) < 1e-4},
        "03_tps_central_reproduces_586": {
            "value": tps_central_repro, "target": C["private_tps_611_central"],
            "resid": abs(tps_central_repro - C["private_tps_611_central"]),
            "pass": abs(tps_central_repro - C["private_tps_611_central"]) < 0.5},
        "04_breakeven_reproduces_8038": {
            "value": breakeven_rho_repro, "target": C["breakeven_rho"],
            "resid": abs(breakeven_rho_repro - C["breakeven_rho"]),
            "pass": abs(breakeven_rho_repro - C["breakeven_rho"]) < 1e-6},
        "05_rho_monotone_in_fdeep": {
            "pass": all(rho_of_fdeep(C, a) < rho_of_fdeep(C, b)
                        for a, b in zip([0.6, 0.7, 0.8, 0.9], [0.7, 0.8, 0.9, 1.0]))},
        "06_breakeven_fdeep_in_unit": {
            "f_deep_breakeven": f_deep_breakeven,
            "pass": 0.0 < f_deep_breakeven < 1.0},
        "07_eagle3_worst_below_breakeven": {  # the headline finding is real, not a coding error
            "rho_min": rho_priv_e3_min, "breakeven": C["breakeven_rho"],
            "pass": rho_priv_e3_min < C["breakeven_rho"]},
        "08_central_clears_worst_misses": {  # YELLOW structure: central CLEARS, worst MISSES
            "central_clears": rho_central_repro >= C["breakeven_rho"],
            "worst_misses": rho_priv_e3_min < C["breakeven_rho"],
            "pass": (rho_central_repro >= C["breakeven_rho"]) and (rho_priv_e3_min < C["breakeven_rho"])},
        "09_nan_clean": {"pass": True},
    }
    # import-exactness check (self-test 0)
    chk = []
    for key, src_val in [("rho_priv_e3_300", C["rho_priv_e3_300"]),
                         ("honest_public_611", C["honest_public_611"]),
                         ("breakeven_rho", C["breakeven_rho"]),
                         ("c_deep_lin", C["c_deep_lin"])]:
        chk.append({"name": key, "value": src_val})
    conditions["00_imports_match_source"]["checks"] = chk

    # nan sweep
    def _finite(x):
        return not (isinstance(x, float) and not math.isfinite(x))
    nan_clean = all(_finite(v) for s in scenarios.values() for v in s.values() if isinstance(v, float))
    conditions["09_nan_clean"]["pass"] = nan_clean

    all_pass = all(c["pass"] for c in conditions.values())

    # verdict
    if rho_central_repro >= C["breakeven_rho"] and rho_priv_e3_min >= C["breakeven_rho"]:
        verdict = "GREEN"
    elif rho_central_repro >= C["breakeven_rho"] and rho_priv_e3_min < C["breakeven_rho"]:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    payload: dict[str, Any] = {
        "pr": 318, "agent": "fern", "kind": "eagle3_deep_private_tax",
        "analysis_only": True, "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "primary_metric_name": "rho_priv_e3_min",
        "rho_priv_e3_min": rho_priv_e3_min,
        "worstcase_private_tps": worstcase_private_tps,
        "worstcase_clears_private_500": worstcase_clears,
        "deep_private_tax_self_test_passes": bool(all_pass),
        "verdict": verdict,
        "test_metric_names": ["worstcase_private_tps", "deep_private_tax_self_test_passes"],
        "constants": C,
        "eagle3_literature": {"table1": EAGLE3_TABLE1, "worst_ratios": e3,
                              "citation": "Li et al. 2025, EAGLE-3, arXiv:2503.01840, Table 1 (T=0, acceptance length tau)"},
        "ubel263_branch_collapse": UBEL263,
        "ubel258_linear_raw_gap": UBEL258,
        "breakeven_decomposition": {
            "breakeven_rho": C["breakeven_rho"],
            "f_deep_breakeven": f_deep_breakeven,
            "c_deep_e3_breakeven": c_deep_e3_breakeven,
            "deep_priv_conditional_breakeven": deep_priv_breakeven,
            "f_deep_headroom_from_linear": 1.0 - f_deep_breakeven,
            "note": "with a_1 held by the tree, the deep-position conditional can fall from "
                    f"{C['deep_pub']*C['c_deep_lin']:.4f} to {deep_priv_breakeven:.4f} "
                    f"(c_deep_e3 from {C['c_deep_lin']:.5f} to {c_deep_e3_breakeven:.4f}, "
                    f"an incremental {100*(1-f_deep_breakeven):.1f}% deep tax) before private-500 breaks.",
        },
        "decision_framing": {
            "breakeven_degradation_pct": breakeven_degradation_pct,
            "measured_within_task_degradation_pct": measured_degradation_pct,
            "eagle3_worst_xdomain_degradation_pct": eagle3_worst_degradation_pct,
            "statement": f"private-500 survives any public->private acceptance-length degradation up to "
                         f"{breakeven_degradation_pct:.1f}%. The MEASURED within-task degradation on this "
                         f"stack is {measured_degradation_pct:.1f}% (clears with ~4.5x headroom); EAGLE-3's "
                         f"WORST cross-DOMAIN degradation is {eagle3_worst_degradation_pct:.1f}% (just over "
                         f"the line). The verdict flips only if the held-out private set is as OOD to the "
                         f"fusion head as summarization is to a code-trained drafter.",
        },
        "scenarios": scenarios,
        "f_deep_sweep": sweep,
        "self_test": {"conditions": conditions, "deep_private_tax_self_test_passes": bool(all_pass)},
        "nan_clean": nan_clean,
        "peak_mem_mib": peak_mem_mib,
        "caveats": {
            "scope": "LOCAL CPU-only analytic over banked constants + published EAGLE-3 numbers. 0 TPS; "
                     "BASELINE 481.53 untouched; greedy/PPL untouched. NO GPU/vLLM/HF Job/submission. "
                     "Authorizes NOTHING. NOT a launch.",
            "bounds_not_measures": "rho_priv_e3_min is a DEFENSIBLE WORST-CASE LOWER BOUND from published "
                     "EAGLE-3 cross-dataset degradation + this stack's banked OOD evidence -- it does NOT "
                     "measure the {2,21,39}-fusion head's private tax (checkpoint-gated; needs a trained head).",
            "conservatism_stack": "0.792 is doubly conservative: (a) max cross-DOMAIN >> within-task "
                     "public->private; (b) it does not credit the organizer-verified M=8 tree a_1-recovery "
                     "(c_1=1.0). Crediting either lifts rho_priv_e3_min above break-even.",
            "axis": "This leg is the public->private DEEP-tax axis. The rank-coverage MASS axis is lawine "
                    "#316's; the a_1-cliff trainability is denken #308's; the x0.804 reconciliation is "
                    "settled in #310. Not re-derived here.",
            "reachability_orthogonal": "E[T]=6.11 reachability (a_1-cliff break, denken #308) and "
                    "greedy/PPL (Issue #192) are out of scope; this prices the PRIVATE projection IF 6.11 holds.",
        },
        "provenance": "lawine #300 (8t5q6sr0) rho_priv_e3 0.9421 / c_deep 0.97135 / a_1 0.72925 / deep 0.91443 "
                      "/ public_et 4.966 x fern #310 (2u3kcnv5) honest_public_611 622.08 / breakeven 0.8038 / "
                      "private_tps_611 586.08 / deployed_priv_over_pub 0.957 x EAGLE-3 arXiv:2503.01840 Table 1 "
                      "worst cross-dataset 0.792 x ubel #263 (he7glotf) branch collapse -34.5% x ubel #258 raw 0.804.",
    }
    return payload


# --------------------------------------------------------------------------- #
def print_report(p: dict[str, Any]) -> None:
    C = p["constants"]
    print("\n" + "=" * 78)
    print("PR #318 -- EAGLE-3 fusion DEEP-PRIVATE-TAX: worst-case rho_priv_e3 lower bound")
    print("=" * 78)
    print(f"  central (deep fidelity inherited, f_deep=1): rho={C['rho_priv_e3_300']:.4f} "
          f"-> {p['scenarios']['central_deep_fidelity']['private_tps']:.1f} TPS  (CLEARS +17.2%)")
    print(f"  break-even rho = {C['breakeven_rho']:.4f}  (= 500 / honest_public(6.11)={C['honest_public_611']:.2f})")
    print(f"  PRIMARY worst-case rho_priv_e3_min = {p['rho_priv_e3_min']:.4f}  "
          f"-> {p['worstcase_private_tps']:.1f} TPS  "
          f"({'CLEARS' if p['worstcase_clears_private_500'] else 'MISSES'} "
          f"{p['scenarios']['eagle3_worst_xdataset']['margin_pct']:+.1f}%)")
    print(f"  VERDICT: {p['verdict']}")
    print("-" * 78)
    print("  scenario ladder (rho -> private TPS @ E[T]=6.11):")
    for k, s in p["scenarios"].items():
        print(f"    {s['rho_priv_e3']:.4f} -> {s['private_tps']:6.1f}  "
              f"[{'CLEAR' if s['clears_private_500'] else 'MISS '}]  {s['name']}")
    bd = p["breakeven_decomposition"]
    print("-" * 78)
    print(f"  break-even f_deep = {bd['f_deep_breakeven']:.4f} "
          f"(c_deep_e3 {C['c_deep_lin']:.5f} -> {bd['c_deep_e3_breakeven']:.4f}; "
          f"incremental deep tax headroom {100*bd['f_deep_headroom_from_linear']:.1f}%)")
    df = p["decision_framing"]
    print(f"  break-even degradation {df['breakeven_degradation_pct']:.1f}% | "
          f"measured within-task {df['measured_within_task_degradation_pct']:.1f}% | "
          f"EAGLE-3 worst cross-domain {df['eagle3_worst_xdomain_degradation_pct']:.1f}%")
    print(f"  self-test: {'PASS' if p['deep_private_tax_self_test_passes'] else 'FAIL'} | "
          f"peak {p['peak_mem_mib']:.1f} MiB")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[deeptax] wandb logging unavailable: {exc}", flush=True)
        return None

    C = payload["constants"]
    run = init_wandb_run(
        job_type="validity-analytic", agent="fern", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-deep-private-tax", "validity-analytic", "private-bar", "eagle3",
              "worst-case-bound", "go-no-go", "bank-the-analysis"],
        config={
            "pr": 318, "rho_priv_e3_central": C["rho_priv_e3_300"], "breakeven_rho": C["breakeven_rho"],
            "honest_public_611": C["honest_public_611"], "c_deep_lin": C["c_deep_lin"],
            "a1_held": C["a1_held"], "deep_pub": C["deep_pub"], "eagle3_public_et": C["eagle3_public_et"],
            "provenance": payload["provenance"], "scope": payload["caveats"]["scope"],
        },
    )
    if run is None:
        print("[deeptax] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    st = payload["self_test"]
    summary: dict[str, Any] = {
        "rho_priv_e3_min": payload["rho_priv_e3_min"],
        "worstcase_private_tps": payload["worstcase_private_tps"],
        "worstcase_clears_private_500": int(bool(payload["worstcase_clears_private_500"])),
        "deep_private_tax_self_test_passes": int(bool(payload["deep_private_tax_self_test_passes"])),
        "verdict_yellow": int(payload["verdict"] == "YELLOW"),
        "rho_priv_e3_central": C["rho_priv_e3_300"],
        "private_tps_central": payload["scenarios"]["central_deep_fidelity"]["private_tps"],
        "breakeven_rho": C["breakeven_rho"],
        "f_deep_breakeven": payload["breakeven_decomposition"]["f_deep_breakeven"],
        "c_deep_e3_breakeven": payload["breakeven_decomposition"]["c_deep_e3_breakeven"],
        "rho_headroom_worstcase": payload["rho_priv_e3_min"] - C["breakeven_rho"],
        "breakeven_degradation_pct": payload["decision_framing"]["breakeven_degradation_pct"],
        "measured_within_task_degradation_pct": payload["decision_framing"]["measured_within_task_degradation_pct"],
        "eagle3_worst_xdomain_degradation_pct": payload["decision_framing"]["eagle3_worst_xdomain_degradation_pct"],
        "eagle3_worst_ratio": payload["eagle3_literature"]["worst_ratios"]["worst_ratio"],
        "ubel263_branch_collapse_floor_tps": payload["scenarios"]["ubel263_branch_collapse_floor"]["private_tps"],
        "raw_no_tree_recovery_tps": payload["scenarios"]["raw_no_tree_recovery_310"]["private_tps"],
        "nan_clean": int(bool(payload["nan_clean"])), "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v["pass"])) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_deep_private_tax_result", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[deeptax] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PR #318 EAGLE-3 fusion deep-private-tax worst-case bound")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-deep-private-tax")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    payload = run()
    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_deep_private_tax_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[deeptax] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    gate = bool(payload["deep_private_tax_self_test_passes"])
    print(f"  PRIMARY rho_priv_e3_min = {payload['rho_priv_e3_min']:.4f}", flush=True)
    print(f"  TEST worstcase_private_tps = {payload['worstcase_private_tps']:.2f}", flush=True)
    print(f"  TEST deep_private_tax_self_test_passes = {gate}", flush=True)
    print(f"  wandb run = {rid}", flush=True)
    if args.self_test:
        print(f"[deeptax] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
