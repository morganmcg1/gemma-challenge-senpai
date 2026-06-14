#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""AR(1)-corrected ASN (PR #212) — tighten the SPRT liveprobe realism band.

THE OVER-COUNTED SEAM
---------------------
denken #205 (`eijqklu2`, MERGED) collapsed the liveprobe certification cost ~75×
(E[N]_nogo=405.42 / nearbar=14,915.06 / worstcase=24,398.04 vs #197's truth-independent
fixed-N 30,455.40) by exploiting that the grounded NO-GO truth (β=0.765 → λ_eq=0.539 ≪
bar 0.9780) drifts the Wald LLR hard toward the NO-GO boundary. But it priced the
sequential accumulation's realism band with a CONSERVATIVE FLAT variance-inflation
×Deff=4.41 — the #190 design effect 1+(m̄−1)·ICC applied as if every within-prompt trial
were EQUALLY correlated / exchangeable.

The real within-prompt structure is NOT exchangeable: #190 measured a DECAYING serial
ACF (ρ(1)=0.258, ρ(2)=0.168, ρ(3)=0.118 …) over within-prompt clusters of size m̄=24.58.
A decaying ACF spreads less total correlation than a flat-exchangeable ICC, so the
effective information per trial is HIGHER than the flat ×4.41 implies, and the realistic
expected-N is TIGHTER (lower) than the flat-Deff inflation. This leg folds the AR(1)/ACF
correlation into the SPRT's partial-sum variance and reports how far the realism band
tightens below ×4.41.

THE MECHANISM (partial-sum variance, triangular/Bartlett weight)
---------------------------------------------------------------
For a stationary within-cluster series with lag-k autocorrelation c(k), the variance of
the sum of m terms is

    Var(Σ_m) = σ²·[m + 2·Σ_{k=1}^{m−1}(m−k)·c(k)]   ⇒   Deff(m) = 1 + 2·Σ_{k=1}^{m−1}(1−k/m)·c(k)

(this is #190's `deff_acf` triangular weight). The within-prompt correlation is the
correlation horizon: trials in different prompts are independent, so the SPRT partial sum
over N trials (≈ N/m̄ independent prompt-clusters) has Var = N·σ²·Deff(m̄). Each actual
trial therefore carries 1/Deff(m̄) of the IID information, and the Wald ASN — whose
boundaries are on the LLR sum and whose per-trial DRIFT is unchanged by correlation —
scales E[N]_real ≈ E[N]_iid · Deff(m̄) (the same "effective-sample-size" multiplier #205
applied; only its VALUE changes here, 4.41 → Deff_AR).

    • AR(1):           c(k)=ρ^k        → Deff_AR(m̄),  asymptote Deff_AR(∞)=(1+ρ)/(1−ρ)
    • exchangeable:    c(k)=ICC const  → Deff(m̄)=1+(m̄−1)·ICC = 4.41  (#205's flat band)
    • ρ→0:             c(k)=0          → Deff=1                      (#205's IID ASN)

HONEST ENVELOPE: the measured ACF decays SLOWER than pure AR(1) (ρ(2)=0.168 ≫ ρ(1)²=0.067,
a fat tail), so the empirical-ACF Deff (#190's banked `acf_deff_at_mbar`=2.77, the measured
ρ(k) in the same Bartlett sum) is LARGER than the AR(1)-parametric Deff (~1.66). The AR(1)
geometric decay is therefore an OPTIMISTIC lower bound; the measured-ACF value is the
data-grounded estimate. Both sit strictly below the flat-exchangeable 4.41, so the flat
band is confirmed CONSERVATIVE (the loose end). The decaying-ACF correction only SHARPENS
the 75× collapse; it never reverses it.

LOCAL CPU-only analytic synthesis over banked #205 SPRT + #190 ACF. No GPU / vLLM / HF Job
/ submission / served-file change / official draw. It MODELS the realism band; it takes NO
draws and authorizes none. BASELINE stays 481.53. Greedy/PPL untouched. The AR correction
moves E[N] (the certification COST), NOT the (α,power) error rates or the binding bar
0.9780 (imported). Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). NOT a launch.

PRIMARY metric  sprt_ar_self_test_passes
TEST    metric  expected_n_nogo_ar  (AR(1)-corrected expected SPRT trials to certify the NO-GO)

Run:
    python research/validity/sprt_ar_asn/sprt_ar_asn.py --self-test \\
        --wandb-name denken/sprt-ar-asn --wandb-group sprt-liveprobe-budget
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
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

_D205_PATH = REPO_ROOT / "research/oracle_readout/sprt_liveprobe_budget/sprt_liveprobe_budget.py"
_D205_JSON = REPO_ROOT / "research/oracle_readout/sprt_liveprobe_budget/sprt_liveprobe_budget_results.json"
_D190_JSON = REPO_ROOT / "research/validity/icc_neff/icc_neff_results.json"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- reuse #205's exact Wald SPRT machinery (the Sprt class needs only banked scalars:
#      bar, mu1, n_fss — NO anchor files at import or instantiation) -------------------- #
D205 = _import("sprt_liveprobe_budget", _D205_PATH)
Sprt = D205.Sprt
ALPHA = D205.ALPHA                    # 0.05
POWER = D205.POWER                    # 0.95
TARGET_OFFICIAL = D205.TARGET_OFFICIAL  # 500.0

TOL_REPRO = 1e-4
TOL_DEFF = 5e-3                       # exchangeable round-trip rel-tol vs #190's flat 4.41
TOL_ERR = 1e-6


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Imports — banked scalars, NOT re-derived.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d205 = json.load(open(_D205_JSON))
    syn = d205["synthesis"]
    h, oc = syn["headline"], syn["operating_characteristic"]
    sb = syn["sprt_setup"]["sprt_boundaries"]

    d190 = json.load(open(_D190_JSON))
    ie = d190["icc_estimate"]
    rc = d190["realistic_ci"]

    return {
        # ---- #205 (eijqklu2) banked SPRT ----
        "expected_n_nogo_iid": h["expected_n_sprt_nogo"],            # 405.424
        "expected_n_nearbar_iid": h["expected_n_sprt_nearbar"],      # 14,915.06
        "expected_n_worstcase_iid": oc["worst_case_expected_n"],     # 24,398.04
        "n_fss_197": h["n_fixed_z95_197"],                           # 30,455.40
        "sprt_A_upper": sb["A_upper_decide_go"],                     # +2.9444
        "sprt_B_lower": sb["B_lower_decide_nogo"],                   # -2.9444
        "realized_alpha_205": oc["realized_alpha_false_nogo_at_mu1"],  # 0.05
        "realized_power_205": oc["realized_power_decide_nogo_at_bar"],  # 0.95
        "private_bar_both": syn["imports"]["private_bar_both_0p9780"],  # 0.9780
        "beta_nogo": h["beta_nogo"],                                  # 0.7651
        "lambda_eq_nogo": h["lambda_eq_nogo"],                       # 0.5396
        "private_lcb_nogo_tps": h["private_lcb_nogo_tps"],           # 419.6
        "savings_vs_fixed_n_nogo_iid": h["savings_vs_fixed_n_nogo"],  # 75.12
        "mu1_go_anchor": 1.0,
        # ---- #190 (fva6o4ug) banked within-prompt ACF / Deff ----
        "rho_lag1_190": ie["acf_rho_lags_1to8"][0],                  # 0.2583  (lag-1 ACF ρ)
        "rho_lags_1to8_190": ie["acf_rho_lags_1to8"],
        "deff_empirical_acf_190": ie["acf_deff_at_mbar"],            # 2.7743 (measured ρ(k) Bartlett)
        "icc_190": ie["icc_hat"],                                    # 0.1446
        "icc_ci_190": ie["icc_ci"],                                  # [0.1043, 0.1857]
        "mbar_190": rc["m_bar"],                                     # 24.5825 (cluster size)
        "deff_flat_441": rc["design_effect_hat"],                   # 4.4106  (1+(m̄−1)·ICC)
        "source_runs": {"d205": "eijqklu2", "d190": "fva6o4ug",
                        "d197": "wqr94io4", "d191": "jeclr39w", "d193": "2clxvlr8"},
    }


# --------------------------------------------------------------------------- #
# (1) Partial-sum variance design effect (the mechanism). Triangular/Bartlett weight
#     over the within-prompt cluster of size m — IDENTICAL form to #190's deff_acf.
# --------------------------------------------------------------------------- #
def deff_partial_sum(m: float, corr: Callable[[int], float]) -> float:
    """Deff(m) = Var(Σ_m)/(m·σ²) = 1 + 2·Σ_{k=1}^{⌊m⌋}(1 − k/m)·corr(k).

    corr(k) = lag-k autocorrelation. The within-prompt cluster of size m is the correlation
    horizon (prompts independent), so this is the per-trial variance-inflation applied to the
    SPRT partial sum. Reproduces #190's `deff_acf` triangular weighting exactly."""
    kmax = int(math.floor(m - 1e-9))
    s = 0.0
    for k in range(1, kmax + 1):
        s += (1.0 - k / m) * corr(k)
    return 1.0 + 2.0 * s


def deff_ar1(m: float, rho: float) -> float:
    """AR(1) cluster design effect: c(k)=ρ^k."""
    return deff_partial_sum(m, lambda k: rho ** k)


def deff_ar1_asymptote(rho: float) -> float:
    """n→∞ AR(1) variance-inflation (1+ρ)/(1−ρ) — the long-run information-rate deflation."""
    return (1.0 + rho) / (1.0 - rho)


def deff_exchangeable(m: float, c: float) -> float:
    """Exchangeable (compound-symmetry) cluster design effect: c(k)=c const → 1+(m−1)·c.
    The partial-sum machinery `deff_partial_sum(m, k→c)` reproduces this (self-test b)."""
    return 1.0 + (m - 1.0) * c


# --------------------------------------------------------------------------- #
# (1) Deff comparison table (AR(1) vs measured-ACF vs flat-exchangeable).
# --------------------------------------------------------------------------- #
def deff_comparison(imp: dict) -> dict[str, Any]:
    m = imp["mbar_190"]
    rho = imp["rho_lag1_190"]
    icc = imp["icc_190"]

    deff_ar_mbar = deff_ar1(m, rho)
    deff_ar_inf = deff_ar1_asymptote(rho)
    deff_exch_machine = deff_partial_sum(m, lambda k: icc)      # exchangeable via partial-sum
    deff_exch_closed = deff_exchangeable(m, icc)                 # = imported 4.41
    deff_emp = imp["deff_empirical_acf_190"]                     # measured ρ(k), banked 2.77
    deff_flat = imp["deff_flat_441"]

    return {
        "rho_lag1_190": rho,
        "mbar_cluster_size": m,
        "deff_ar_at_mbar": deff_ar_mbar,
        "deff_ar_asymptote": deff_ar_inf,
        "deff_empirical_acf_measured": deff_emp,
        "deff_exchangeable_machine_check": deff_exch_machine,
        "deff_exchangeable_closed_form": deff_exch_closed,
        "deff_flat_441": deff_flat,
        "ordering_ar_lt_emp_lt_flat": bool(deff_ar_mbar < deff_emp < deff_flat),
        "ar_underestimates_fat_tail": bool(deff_emp > deff_ar_mbar),
        "tightening_ar_vs_flat": deff_flat / deff_ar_mbar,
        "tightening_empirical_vs_flat": deff_flat / deff_emp,
        "note": (
            f"AR(1) ρ={rho:.4f} → Deff_AR(m̄)={deff_ar_mbar:.4f} (asymptote {deff_ar_inf:.4f}); the "
            f"MEASURED ACF decays slower than ρ^k (ρ(2)={imp['rho_lags_1to8_190'][1]:.3f} ≫ ρ(1)²="
            f"{rho**2:.3f}), so the empirical-ACF Deff {deff_emp:.4f} > AR(1) {deff_ar_mbar:.4f}. Both ≪ "
            f"the flat-exchangeable {deff_flat:.4f}. AR(1) is the optimistic lower bound; the measured "
            f"ACF is the data-grounded realistic Deff; flat 4.41 is the conservative ceiling."),
    }


# --------------------------------------------------------------------------- #
# (2) AR-corrected expected-N (the deliverable). E[N]_real = E[N]_iid · Deff.
# --------------------------------------------------------------------------- #
def expected_n_corrected(imp: dict, dc: dict) -> dict[str, Any]:
    iid = {
        "nogo": imp["expected_n_nogo_iid"],
        "nearbar": imp["expected_n_nearbar_iid"],
        "worstcase": imp["expected_n_worstcase_iid"],
    }
    deff_ar = dc["deff_ar_at_mbar"]
    deff_emp = dc["deff_empirical_acf_measured"]
    deff_flat = dc["deff_flat_441"]

    def band(name: str) -> dict[str, Any]:
        e = iid[name]
        return {
            "iid": e,
            "ar1": e * deff_ar,
            "empirical_acf": e * deff_emp,
            "flat_441": e * deff_flat,
            "tightening_ar_vs_flat": deff_flat / deff_ar,
            "tightening_empirical_vs_flat": deff_flat / deff_emp,
        }

    rows = {k: band(k) for k in iid}
    # the N_fss/E[N] savings ratio is Deff-INVARIANT: the #197 fixed-N reference and the SPRT E[N]
    # both inflate by the SAME cluster Deff, so the ratio cancels it. The AR correction changes the
    # ABSOLUTE realism-inflated counts, never the headline collapse.
    n_fss = imp["n_fss_197"]
    savings = {
        "n_fss_197": n_fss,
        "savings_ratio_iid": n_fss / iid["nogo"],
        "savings_ratio_under_ar": (n_fss * deff_ar) / rows["nogo"]["ar1"],
        "savings_ratio_under_empirical_acf": (n_fss * deff_emp) / rows["nogo"]["empirical_acf"],
        "savings_ratio_under_flat_441": (n_fss * deff_flat) / rows["nogo"]["flat_441"],
        "deff_invariant": bool(
            abs((n_fss * deff_ar) / rows["nogo"]["ar1"] - n_fss / iid["nogo"]) < 1e-6
            and abs((n_fss * deff_flat) / rows["nogo"]["flat_441"] - n_fss / iid["nogo"]) < 1e-6),
        "note": ("N_fss/E[N] = 75.12× is Deff-invariant: both the fixed-N reference and the SPRT E[N] "
                 "scale by the same cluster Deff. The AR correction tightens the ABSOLUTE realism band "
                 "(flat → AR / measured-ACF) but leaves the 75× collapse untouched."),
    }
    return {
        "rows": rows,
        "expected_n_nogo_ar": rows["nogo"]["ar1"],
        "expected_n_nearbar_ar": rows["nearbar"]["ar1"],
        "expected_n_worstcase_ar": rows["worstcase"]["ar1"],
        "expected_n_nogo_empirical_acf": rows["nogo"]["empirical_acf"],
        "expected_n_nogo_flat_441": rows["nogo"]["flat_441"],
        "deff_ar_at_mbar": deff_ar,
        "deff_empirical_acf": deff_emp,
        "deff_flat_441": deff_flat,
        "savings_invariance": savings,
    }


# --------------------------------------------------------------------------- #
# (3) Realism band from the ρ-CI (honest envelope). ρ(1)-CI inherits #190's banked
#     prompt-level cluster-bootstrap ICC CI, scaled to the lag-1 ACF (ρ ∝ correlation
#     strength). The loose (high-ρ) AR end is confirmed below the flat-exchangeable 4.41.
# --------------------------------------------------------------------------- #
def realism_band(imp: dict, dc: dict) -> dict[str, Any]:
    m = imp["mbar_190"]
    rho = imp["rho_lag1_190"]
    icc = imp["icc_190"]
    icc_lo, icc_hi = imp["icc_ci_190"]
    e_nogo = imp["expected_n_nogo_iid"]
    deff_flat = imp["deff_flat_441"]
    deff_emp = imp["deff_empirical_acf_190"]

    # map the banked ICC CI to a ρ(1) CI (proportional to correlation strength)
    rho_lo = rho * (icc_lo / icc)
    rho_hi = rho * (icc_hi / icc)

    deff_ar_lo = deff_ar1(m, rho_lo)    # tight (low-ρ) AR Deff
    deff_ar_hi = deff_ar1(m, rho_hi)    # loose (high-ρ) AR Deff

    en_tight = e_nogo * deff_ar_lo
    en_loose = e_nogo * deff_ar_hi
    en_flat = e_nogo * deff_flat

    # flat 4.41 is conservative iff even the loose AR end AND the measured ACF stay below it
    flat_is_conservative = bool(deff_ar_hi < deff_flat and deff_emp < deff_flat)

    return {
        "rho_lag1_hat": rho,
        "rho_ci": [rho_lo, rho_hi],
        "rho_ci_method": ("inherits #190 prompt-level cluster-bootstrap ICC CI "
                          f"[{icc_lo:.4f},{icc_hi:.4f}], scaled to lag-1 ACF (ρ∝corr-strength)"),
        "deff_ar_ci": [deff_ar_lo, deff_ar_hi],
        "expected_n_nogo_ar_tight": en_tight,
        "expected_n_nogo_ar_loose": en_loose,
        "expected_n_nogo_band": [en_tight, en_loose],
        "expected_n_nogo_empirical_acf": e_nogo * deff_emp,
        "expected_n_nogo_flat_441_loose_end": en_flat,
        "flat_441_is_conservative": flat_is_conservative,
        "loose_ar_below_flat_margin": deff_flat - deff_ar_hi,
        "note": (
            f"ρ(1)∈[{rho_lo:.4f},{rho_hi:.4f}] → Deff_AR∈[{deff_ar_lo:.4f},{deff_ar_hi:.4f}] → "
            f"E[N]_nogo∈[{en_tight:.0f},{en_loose:.0f}]. The loose AR end {deff_ar_hi:.3f} and the "
            f"measured-ACF Deff {deff_emp:.3f} both sit below the flat-exchangeable {deff_flat:.3f} → "
            f"#205's flat ×4.41 (E[N]_nogo {en_flat:.0f}) is the CONSERVATIVE (loose) end."),
    }


# --------------------------------------------------------------------------- #
# (d) Operating characteristic is Deff-INVARIANT. Recompute #205's OC from the reused
#     Sprt (boundaries + drift only) to confirm the AR correction moves E[N], not (α,power).
# --------------------------------------------------------------------------- #
def operating_characteristic_invariance(imp: dict) -> dict[str, Any]:
    sp = Sprt(bar=imp["private_bar_both"], mu1=imp["mu1_go_anchor"], n_fss=imp["n_fss_197"])
    realized_alpha = 1.0 - sp.p_accept_go(sp.mu1)     # P(decide NO-GO | truly GO μ1)
    realized_power = 1.0 - sp.p_accept_go(sp.mu0)     # P(decide NO-GO | at bar μ0)
    # cross-check: the reused Sprt reproduces #205's banked IID ASN exactly
    asn_nogo_repro = sp.asn(imp["lambda_eq_nogo"])
    asn_peak_repro = sp.peak_asn()
    # drift-dominance diagnostic (researcher caveat on the E[N]·Deff scaling): the NO-GO truth
    # sits FAR below the indifference midpoint μ_mid → drift-dominated, where the Deff multiplier
    # is asymptotically accurate; near-bar/worst-case sit AT μ_mid (variance-dominated), where the
    # Deff scaling is one-sided CONSERVATIVE (it overestimates E[N]).
    lam_eq_nogo = imp["lambda_eq_nogo"]
    gap_nogo = abs(lam_eq_nogo - sp.mu_mid)
    p_go_nogo = sp.p_accept_go(lam_eq_nogo)
    return {
        "A_upper": sp.A, "B_lower": sp.B, "mu_mid": sp.mu_mid, "margin_m": sp.m,
        "realized_alpha": realized_alpha, "realized_power": realized_power,
        "realized_alpha_205_import": imp["realized_alpha_205"],
        "realized_power_205_import": imp["realized_power_205"],
        "alpha_unchanged": bool(abs(realized_alpha - imp["realized_alpha_205"]) < TOL_ERR),
        "power_unchanged": bool(abs(realized_power - imp["realized_power_205"]) < TOL_ERR),
        "asn_nogo_iid_reproduced": asn_nogo_repro,
        "asn_peak_iid_reproduced": asn_peak_repro,
        "reproduces_205_nogo": bool(abs(asn_nogo_repro - imp["expected_n_nogo_iid"]) < 1e-3),
        "reproduces_205_worstcase": bool(abs(asn_peak_repro - imp["expected_n_worstcase_iid"]) < 1e-3),
        "nogo_gap_from_indifference_lambda": gap_nogo,
        "nogo_gap_in_margins": gap_nogo / sp.m,
        "nogo_p_decide_go": p_go_nogo,
        "nogo_is_drift_dominated": bool(p_go_nogo < ALPHA and gap_nogo > 5.0 * sp.m),
        "regime_caveat": (
            f"E[N]·Deff is asymptotically exact in the DRIFT-dominated regime and one-sided "
            f"conservative (overestimates) in the variance-dominated regime. The NO-GO truth "
            f"(λ_eq={lam_eq_nogo:.3f}) sits {gap_nogo / sp.m:.0f}× the {sp.m:.3f} margin below the "
            f"indifference midpoint μ_mid={sp.mu_mid:.3f} (P(decide GO)={p_go_nogo:.2e}) → strongly "
            f"drift-dominated, so the TEST metric expected_n_nogo_ar is on solid ground. The near-bar "
            f"(β=1) and worst-case (indifference) truths are variance-dominated, so their AR-corrected "
            f"E[N] are CONSERVATIVE upper bounds — the realism tightening there is, if anything, "
            f"understated."),
        "note": ("the AR correction rescales the per-trial INFORMATION (variance), not the per-trial "
                 "DRIFT or the Wald boundaries A,B — so the OC and realized (α,power) are unchanged; "
                 "only the ASN E[N] is multiplied by Deff."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, dc: dict, en: dict, band: dict, oc: dict) -> dict[str, Any]:
    m = imp["mbar_190"]
    rho = imp["rho_lag1_190"]
    icc = imp["icc_190"]
    deff_flat = imp["deff_flat_441"]

    # (a) ρ→0 reproduces #205's flat-independent ASN (Deff_AR=1 → E[N]=E[N]_iid).
    deff_ar_zero = deff_ar1(m, 0.0)
    en_nogo_at_zero = imp["expected_n_nogo_iid"] * deff_ar_zero
    cond_a = bool(abs(deff_ar_zero - 1.0) < TOL_ERR
                  and abs(en_nogo_at_zero - imp["expected_n_nogo_iid"]) < 1e-6)

    # (b) FLAT-exchangeable limit (ρ const over all lags = ICC) round-trips ×Deff=4.41.
    deff_exch = deff_partial_sum(m, lambda k: icc)
    cond_b = bool(abs(deff_exch - deff_flat) <= TOL_DEFF * deff_flat)

    # (c) Deff_AR monotone INCREASING in ρ (more correlation → larger design effect).
    grid = [deff_ar1(m, r) for r in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]]
    cond_c = all(grid[i + 1] > grid[i] - TOL_ERR for i in range(len(grid) - 1)) and grid[0] <= grid[-1]

    # (d) realized (α,power) UNCHANGED — the AR correction moves E[N], not the error rates.
    cond_d = bool(oc["alpha_unchanged"] and oc["power_unchanged"]
                  and oc["reproduces_205_nogo"] and oc["reproduces_205_worstcase"])

    # (e) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [dc["deff_ar_at_mbar"], dc["deff_ar_asymptote"], dc["deff_empirical_acf_measured"],
           dc["deff_flat_441"], en["expected_n_nogo_ar"], en["expected_n_nearbar_ar"],
           en["expected_n_worstcase_ar"], band["expected_n_nogo_ar_tight"],
           band["expected_n_nogo_ar_loose"], band["rho_ci"][0], band["rho_ci"][1]]
    cond_e = all(_finite(x) for x in key)

    # (f) the AR correction TIGHTENS: every realistic Deff < flat 4.41 (the hypothesis check).
    cond_f = bool(dc["deff_ar_at_mbar"] < deff_flat
                  and dc["deff_empirical_acf_measured"] < deff_flat
                  and band["flat_441_is_conservative"])

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "sprt_ar_self_test_passes": passes,
        "conditions": {
            "a_rho0_reproduces_205_iid_asn_deff1": cond_a,
            "b_exchangeable_limit_roundtrips_flat_441": cond_b,
            "c_deff_ar_monotone_increasing_in_rho": bool(cond_c),
            "d_realized_alpha_power_unchanged": cond_d,
            "e_key_scalars_finite": cond_e,
            "f_ar_correction_tightens_below_flat_441": cond_f,
        },
        "evidence": {
            "a_deff_ar_at_rho0": deff_ar_zero, "a_en_nogo_at_rho0": en_nogo_at_zero,
            "a_en_nogo_iid": imp["expected_n_nogo_iid"],
            "b_deff_exchangeable_machine": deff_exch, "b_deff_flat_441": deff_flat,
            "b_rel_err": abs(deff_exch - deff_flat) / deff_flat,
            "c_deff_ar_grid": grid,
            "d_realized_alpha": oc["realized_alpha"], "d_realized_power": oc["realized_power"],
            "f_deff_ar": dc["deff_ar_at_mbar"], "f_deff_empirical": dc["deff_empirical_acf_measured"],
            "f_deff_flat": deff_flat,
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, dc: dict, en: dict, band: dict) -> str:
    return (
        f"AR-CORRECTED. #205 priced the liveprobe realism band with a CONSERVATIVE flat "
        f"×Deff={dc['deff_flat_441']:.2f} (exchangeable ICC={imp['icc_190']:.4f}). Folding #190's "
        f"DECAYING serial ACF (ρ(1)={dc['rho_lag1_190']:.4f}) into the partial-sum variance gives "
        f"Deff_AR(m̄)={dc['deff_ar_at_mbar']:.3f} (asymptote {dc['deff_ar_asymptote']:.3f}), a "
        f"{dc['tightening_ar_vs_flat']:.2f}× tightening — E[N]_nogo drops from the flat "
        f"{en['expected_n_nogo_flat_441']:,.0f} to {en['expected_n_nogo_ar']:,.0f}. HONEST CAVEAT: the "
        f"MEASURED ACF decays slower than pure AR(1), so the data-grounded empirical-ACF Deff "
        f"{dc['deff_empirical_acf_measured']:.3f} (E[N]_nogo {en['expected_n_nogo_empirical_acf']:,.0f}, "
        f"{dc['tightening_empirical_vs_flat']:.2f}× tighter) is the realistic value; AR(1) is the "
        f"optimistic lower bound. The ρ-CI band is [{band['expected_n_nogo_ar_tight']:,.0f},"
        f"{band['expected_n_nogo_ar_loose']:,.0f}]; even the loose end stays below the flat-Deff "
        f"{en['expected_n_nogo_flat_441']:,.0f}, so flat ×4.41 is the CONSERVATIVE (loose) end "
        f"(flat_441_is_conservative={band['flat_441_is_conservative']}). The decaying-ACF correction "
        f"only SHARPENS #205's 75× collapse; realized (α,power)=(0.05,0.95) and the bar 0.9780 are "
        f"untouched. NOT a launch."
    )


def _handoff(imp: dict, dc: dict, en: dict, band: dict) -> dict[str, str]:
    fern = (
        f"fern #185: the AR(1)-corrected liveprobe certification cost is E[N]_nogo="
        f"{en['expected_n_nogo_ar']:,.0f} (AR(1) optimistic lower bound) / "
        f"{en['expected_n_nogo_empirical_acf']:,.0f} (measured-ACF, data-grounded), vs #205's "
        f"conservative flat-Deff {en['expected_n_nogo_flat_441']:,.0f} and the IID floor "
        f"{imp['expected_n_nogo_iid']:,.0f}; realized (0.05,0.95) unchanged. Carry the tightened "
        f"expected-N band [{band['expected_n_nogo_ar_tight']:,.0f} tight … "
        f"{en['expected_n_nogo_flat_441']:,.0f} flat-loose] as the measurement-cost row. The "
        f"{imp['savings_vs_fixed_n_nogo_iid']:.0f}× collapse vs #197's fixed-N is Deff-INVARIANT "
        f"(fixed-N and SPRT E[N] scale by the SAME cluster Deff), so the decaying-ACF correction "
        f"leaves the 75× headline intact while SHARPENING the absolute realism band below the "
        f"conservative flat 4.41; it never reverses the NO-GO-is-cheap conclusion. flat ×4.41 is the "
        f"loose end."
    )
    return {"fern_185": fern}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    dc = deff_comparison(imp)
    en = expected_n_corrected(imp, dc)
    oc = operating_characteristic_invariance(imp)
    band = realism_band(imp, dc)
    st = self_test(imp, dc, en, band, oc)
    handoff = _handoff(imp, dc, en, band)
    return {
        "self_test": st,
        "test_metric": {"expected_n_nogo_ar": en["expected_n_nogo_ar"]},
        "imports": imp,
        "deff_comparison": dc,
        "expected_n_corrected": en,
        "operating_characteristic_invariance": oc,
        "realism_band": band,
        "verdict": _verdict(imp, dc, en, band),
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
    imp, dc = syn["imports"], syn["deff_comparison"]
    en, band, oc = syn["expected_n_corrected"], syn["realism_band"], syn["operating_characteristic_invariance"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("AR(1)-CORRECTED ASN (PR #212) — tighten the SPRT liveprobe realism band", flush=True)
    print("=" * 96, flush=True)
    print(f"  ρ(1)_190 = {dc['rho_lag1_190']:.4f}   m̄ = {dc['mbar_cluster_size']:.4f}   "
          f"ICC = {imp['icc_190']:.4f}   private bar = {imp['private_bar_both']:.4f}", flush=True)
    print("-" * 96, flush=True)
    print("  DESIGN EFFECT (within-prompt cluster m̄):", flush=True)
    print(f"      flat-exchangeable (ICC)        Deff = {dc['deff_flat_441']:.4f}   (#205 conservative)", flush=True)
    print(f"      measured-ACF (Bartlett, #190)  Deff = {dc['deff_empirical_acf_measured']:.4f}   "
          f"(data-grounded; {dc['tightening_empirical_vs_flat']:.2f}× tighter)", flush=True)
    print(f"      AR(1)  ρ^k  at m̄               Deff = {dc['deff_ar_at_mbar']:.4f}   "
          f"(asymptote {dc['deff_ar_asymptote']:.4f}; {dc['tightening_ar_vs_flat']:.2f}× tighter)", flush=True)
    print("-" * 96, flush=True)
    print(f"  (TEST) expected_n_nogo_ar = {en['expected_n_nogo_ar']:,.1f}   "
          f"(flat-Deff {en['rows']['nogo']['flat_441']:,.0f} | measured-ACF "
          f"{en['expected_n_nogo_empirical_acf']:,.0f} | IID floor {imp['expected_n_nogo_iid']:,.0f})", flush=True)
    print("  AR-corrected E[N] table  (E[N]_iid · Deff):", flush=True)
    for k, r in en["rows"].items():
        print(f"      {k:9s}  iid={r['iid']:>10,.0f}  AR(1)={r['ar1']:>10,.0f}  "
              f"emp-ACF={r['empirical_acf']:>10,.0f}  flat={r['flat_441']:>11,.0f}", flush=True)
    print("-" * 96, flush=True)
    print(f"  ρ-CI band: ρ(1)∈[{band['rho_ci'][0]:.4f},{band['rho_ci'][1]:.4f}]  → "
          f"E[N]_nogo∈[{band['expected_n_nogo_ar_tight']:,.0f},{band['expected_n_nogo_ar_loose']:,.0f}]", flush=True)
    print(f"  flat_441_is_conservative = {band['flat_441_is_conservative']}  "
          f"(loose AR Deff {band['deff_ar_ci'][1]:.3f} < flat {dc['deff_flat_441']:.3f})", flush=True)
    print("-" * 96, flush=True)
    print(f"  realized (α,power) = ({oc['realized_alpha']:.4f},{oc['realized_power']:.4f})  "
          f"UNCHANGED (α {oc['alpha_unchanged']} / power {oc['power_unchanged']}); "
          f"reproduces #205 nogo={oc['reproduces_205_nogo']} worstcase={oc['reproduces_205_worstcase']}", flush=True)
    print("-" * 96, flush=True)
    print(f"  (PRIMARY) sprt_ar_self_test_passes = {st['sprt_ar_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (fern #185): {syn['handoff_lines']['fern_185']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #205; never fatal).
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
        print(f"[sprt-ar-asn] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp, dc = syn["imports"], syn["deff_comparison"]
    en, band, oc = syn["expected_n_corrected"], syn["realism_band"], syn["operating_characteristic_invariance"]
    st = syn["self_test"]

    run = init_wandb_run(
        job_type="sprt-liveprobe-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["sprt-liveprobe-budget", "validity-gate", "measurement-design", "sequential-test",
              "wald-sprt", "expected-n", "ar1-correction", "design-effect", "bank-the-analysis"],
        config={
            "target_official": TARGET_OFFICIAL, "alpha": ALPHA, "power": POWER,
            "rho_lag1_190": dc["rho_lag1_190"], "mbar_190": dc["mbar_cluster_size"],
            "icc_190": imp["icc_190"], "deff_flat_441": dc["deff_flat_441"],
            "private_bar_both": imp["private_bar_both"], "beta_nogo": imp["beta_nogo"],
            "imports": "denken#205 SPRT (eijqklu2) + wirbel#190 ACF (fva6o4ug)",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[sprt-ar-asn] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "sprt_ar_self_test_passes": int(bool(st["sprt_ar_self_test_passes"])),
        "expected_n_nogo_ar": en["expected_n_nogo_ar"],
        "expected_n_nearbar_ar": en["expected_n_nearbar_ar"],
        "expected_n_worstcase_ar": en["expected_n_worstcase_ar"],
        "expected_n_nogo_empirical_acf": en["expected_n_nogo_empirical_acf"],
        "expected_n_nogo_flat_441": en["expected_n_nogo_flat_441"],
        "expected_n_nogo_iid": imp["expected_n_nogo_iid"],
        "rho_lag1_190": dc["rho_lag1_190"],
        "deff_ar_at_mbar": dc["deff_ar_at_mbar"],
        "deff_ar_asymptote": dc["deff_ar_asymptote"],
        "deff_empirical_acf": dc["deff_empirical_acf_measured"],
        "deff_flat_441": dc["deff_flat_441"],
        "tightening_ar_vs_flat": dc["tightening_ar_vs_flat"],
        "tightening_empirical_vs_flat": dc["tightening_empirical_vs_flat"],
        "rho_ci_lo": band["rho_ci"][0], "rho_ci_hi": band["rho_ci"][1],
        "expected_n_nogo_ar_tight": band["expected_n_nogo_ar_tight"],
        "expected_n_nogo_ar_loose": band["expected_n_nogo_ar_loose"],
        "flat_441_is_conservative": int(bool(band["flat_441_is_conservative"])),
        "realized_alpha": oc["realized_alpha"], "realized_power": oc["realized_power"],
        "n_fss_197": imp["n_fss_197"], "icc_190": imp["icc_190"],
        "mbar_190": dc["mbar_cluster_size"], "private_bar_both": imp["private_bar_both"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="sprt_ar_asn_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[sprt-ar-asn] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default="sprt-liveprobe-budget")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 212,
        "agent": "denken",
        "kind": "sprt-ar-asn",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[sprt-ar-asn] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (e) and recompute PRIMARY
    syn["self_test"]["conditions"]["e_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["e_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["sprt_ar_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sprt_ar_asn_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[sprt-ar-asn] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY sprt_ar_self_test_passes = {passes}", flush=True)
    print(f"  TEST expected_n_nogo_ar = {syn['test_metric']['expected_n_nogo_ar']:,.2f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[sprt-ar-asn] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
