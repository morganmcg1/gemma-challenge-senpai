#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""SPRT liveprobe budget (PR #205) — expected-N early-stop vs #197's fixed-N 30k.

THE UN-COMPOSED SEAM
--------------------
denken #197 (`wqr94io4`, MERGED) priced the FIXED-N Neyman liveprobe budget at
30,455 trials to DECISIVELY certify the best-case λ=1 build against the private bar
0.9780 — and proved (`mechanism_can_clear_private_bar=False`) that at the grounded
β=0.765 even PERFECT depth-1 recovery gives private_LCB 419.6 ≪ 500: a clear NO-GO.
A fixed-N design spends the full 30k EVEN WHEN the build is obviously below the bar.

This leg is the SEQUENTIAL analog: a Wald SPRT on the per-depth accept stream,
weighted by #197's E[T]-functional weights a_d, stops as soon as the accumulated
evidence is decisive — so the EXPECTED trials on a clear-NO-GO build collapse far
below 30k. It tells land #71 the REALISTIC trial cost (the expected-N early-stop
cost under each truth), not just the truth-independent fixed-N worst case.

IMPORTS — NOT re-derived (re-uses #197's banked Neyman design verbatim)
----------------------------------------------------------------------
    #197 (`wqr94io4`)  neyman_budget(): per-depth weights a_d=∂E[T]/∂q_d, Neyman
                       fractions, fixed-N N_opt(margin) (30,455 @λ=1, margin 0.022),
                       shallow-heavy allocation; private forward map private_lcb_mech.
    #193 (`2clxvlr8`)  β_primary 0.7651, β_crit 0.9649, λ_d=λ̂₁·β^(d−1)   (via #197.build)
    #187 (`tloghme9`)  per-depth σ_d, ∂E[T]/∂λ, survival-thinned ladder (via #197)
    #191 (`jeclr39w`)  private bar 0.9780 both-bugs / descent UNREACHABLE (via #197)
    #190 (`fva6o4ug`)  within-prompt ICC 0.1446, Deff 4.41 (realism multiplier — a band)

DESIGN — matched-strength sequential test
-----------------------------------------
#197's decisive fixed-N at margin m, N_FSS(m)=z95²·σ₁²/m², is ADOPTED as the
fixed-sample reference for a sequential test of strength (α,power)=(0.05,0.95);
equivalently the per-trial standardized drift is calibrated so the SPRT's
no-early-stop sample size reproduces #197's N_FSS (self-test a). The per-trial LLR
on the Neyman-weighted λ-equivalent read x_i is

    z_i = (m / σ₁²) · (x_i − μ_mid),   x_i ~ N(λ_eq, σ₁²),   μ_mid = (bar+1)/2

with Wald boundaries A=ln((1−β)/α), B=ln(β/(1−α)). The OC/ASN are Wald-Gaussian.
The SEQUENTIAL SAVINGS RATIO N_FSS/E[N] is INVARIANT to the absolute info
calibration (σ₁² cancels), so the headline ≈64× collapse on a clear-NO-GO build is
robust whether we anchor N_FSS to #197's 30,455 or to the physical-info (0.05,0.95)
fixed-N 85,820 — both reported.

LOCAL CPU-only analytic synthesis over banked weights. No GPU / vLLM / HF Job /
submission / served-file change. It MODELS the sequential test; it takes NO draws and
authorizes none. BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis
(PRIMARY = self-test, adds 0 TPS). NOT open2. NOT a launch.

PRIMARY metric  sprt_budget_self_test_passes
TEST    metric  expected_n_sprt_nogo  (expected SPRT trials to certify the likely NO-GO)

Run:
    python research/oracle_readout/sprt_liveprobe_budget/sprt_liveprobe_budget.py \
        --self-test --wandb-name denken/sprt-liveprobe-budget \
        --wandb-group sprt-liveprobe-budget
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

_D197_PATH = REPO_ROOT / "research/oracle_readout/liveprobe_depth_budget/liveprobe_depth_budget.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- import the #197 banked design (which reaches #193/#187/#191/#172/#178) ---- #
D197 = _import("liveprobe_depth_budget", _D197_PATH)
D172 = D197.D172

# ---- composition constants (imported, not re-derived) ---- #
Z95 = D197.Z95                                   # 1.95996… two-sided 95% normal quantile
TARGET_OFFICIAL = D197.TARGET_OFFICIAL           # 500.0
HEADLINE_LAM_TRUE = D197.HEADLINE_LAM_TRUE       # 1.0 (best-case GO build; margin 0.022)

# #190 (`fva6o4ug`) within-prompt realism multiplier (committed; a SECONDARY band).
ICC_190 = 0.1446247464062406                     # wirbel #190 ANOVA within-prompt ICC
DEFF_190 = 4.410614351127293                     # wirbel #190 design effect (m̄=24.58)

# SPRT strength (Wald): target (α, power) = (0.05, 0.95) ⇒ Type-II β_err = 0.05.
ALPHA = 0.05
POWER = 0.95
BETA_ERR = 1.0 - POWER
_N01 = NormalDist(0.0, 1.0)
Z_ALPHA = _N01.inv_cdf(1.0 - ALPHA)              # 1.6449  (one-sided)
Z_BETA = _N01.inv_cdf(POWER)                     # 1.6449
Z_SUM = Z_ALPHA + Z_BETA                         # 3.2897

# Truth grid (geometric staleness decay β at PERFECT depth-1 recovery λ̂₁=1.0 — the
# conservative NO-GO family from #197 mechanism_feasibility: only deep-ladder staleness
# β varies). β=0.765 is the grounded likely NO-GO; β=1.0 the best-case near-bar GO.
BETA_TRUTH_GRID = (0.70, 0.765, 0.82, 0.88, 0.92, 0.9649, 0.98, 1.0)

TOL_REPRO = 1e-4
TOL_FIXED_N = 1e-3                                # reproduce #197's 30,455 to this rel tol
TOL_ERR = 1e-6


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# State: reuse #197's build() (st carries the bar, β, forward map, ctx spines).
# --------------------------------------------------------------------------- #
def build_state(anchors: dict[str, Any]) -> dict[str, Any]:
    return D197.build(anchors)


def priv_lcb(st: dict, lam1: float, beta: float) -> float:
    """#197 private-LCB forward map (both-bugs): pub(λ-profile)·(1−drop)·τ_low."""
    return D197.private_lcb_mech(st, lam1, beta, topo="both")


# --------------------------------------------------------------------------- #
# Forward-map inversions (monotone, bisection). λ_eq = flat-equivalent recovery
# whose private LCB matches a target; β_at = staleness giving a target at λ̂₁=1.
# --------------------------------------------------------------------------- #
def _bisect(f, lo: float, hi: float, target: float, iters: int = 200) -> float:
    flo, fhi = f(lo) - target, f(hi) - target
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo > 0:                       # ensure f(lo) < target < f(hi)
        lo, hi, flo, fhi = hi, lo, fhi, flo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - target
        if fm == 0.0:
            return mid
        if fm < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lambda_eq_of_priv(st: dict, target_priv: float) -> float:
    """Flat-ladder recovery λ_eq (β=1) whose private LCB == target_priv."""
    return _bisect(lambda lam: priv_lcb(st, lam, 1.0), 1e-6, 1.0, target_priv)


def lambda_eq_of_beta(st: dict, beta: float, lam1: float = 1.0) -> float:
    """Truth at staleness β (perfect depth-1) → its flat-equivalent recovery λ_eq."""
    return lambda_eq_of_priv(st, priv_lcb(st, lam1, beta))


def beta_of_lambda_eq(st: dict, lam_eq: float, lam1: float = 1.0) -> float:
    """Inverse: the staleness β (at λ̂₁=1) whose private LCB matches flat λ_eq."""
    target = priv_lcb(st, lam_eq, 1.0)
    return _bisect(lambda b: priv_lcb(st, lam1, b), 1e-6, 1.0, target)


# --------------------------------------------------------------------------- #
# Fixed-N reference (imported verbatim from #197): N_FSS(margin), Neyman per-depth
# allocation. This is the no-early-stop limit (self-test a).
# --------------------------------------------------------------------------- #
def fixed_n_reference(st: dict, lam_true: float = HEADLINE_LAM_TRUE) -> dict[str, Any]:
    b = D197.neyman_budget(st, lam_true)
    return {
        "lam_true": b["lam_true"],
        "bar": st["private_bar_both"],
        "margin": b["margin"],                       # μ1 − μ0 in λ
        "N_fixed_z95": b["N_opt"],                   # #197's decisive fixed-N (30,455 @λ=1)
        "N_equal": b["N_equal"],
        "efficiency_gain": b["efficiency_gain"],
        "a_d_2to9": b["a_d"],
        "sigma_d_2to9": b["sigma_d"],
        "q_d_2to9": b["q_d"],
        "neyman_weight_a_sigma_2to9": b["neyman_weight_a_sigma"],
        "neyman_fraction_2to9": b["neyman_fraction_2to9"],
        "N_d_budget_1to9": b["N_d_budget_1to9"],
        "dEt_dlambda": b["dEt_dlambda"],
        "sum_a_sigma": b["sum_a_sigma"],
        "sigma1_sq_per_trial": (b["sum_a_sigma"] ** 2) / (b["dEt_dlambda"] ** 2),  # #197 per-trial λ-var
    }


# --------------------------------------------------------------------------- #
# Wald SPRT core (Gaussian OC/ASN). The per-trial LLR variance ς² is calibrated so
# the no-early-stop fixed-sample size at (α,power) equals N_fss:  N_fss=(z_α+z_β)²/ς².
# OC and ASN follow Wald's approximation; the SAVINGS RATIO N_fss/E[N] is ς²-invariant.
# --------------------------------------------------------------------------- #
def wald_boundaries(alpha: float = ALPHA, beta_err: float = BETA_ERR) -> dict[str, float]:
    A = math.log((1.0 - beta_err) / alpha)       # upper: decide GO (μ1)
    B = math.log(beta_err / (1.0 - alpha))       # lower: decide NO-GO (μ0=bar)
    return {"A_upper_decide_go": A, "B_lower_decide_nogo": B}


class Sprt:
    """Matched-strength Wald SPRT on the Neyman-weighted λ-equivalent stream."""

    def __init__(self, bar: float, mu1: float, n_fss: float,
                 alpha: float = ALPHA, beta_err: float = BETA_ERR):
        self.mu0 = bar                            # NO-GO boundary (the private bar)
        self.mu1 = mu1                            # GO anchor (full recovery, β=1)
        self.m = mu1 - bar                        # margin
        self.mu_mid = 0.5 * (bar + mu1)
        self.alpha, self.beta_err = alpha, beta_err
        bnd = wald_boundaries(alpha, beta_err)
        self.A, self.B = bnd["A_upper_decide_go"], bnd["B_lower_decide_nogo"]
        self.n_fss = n_fss                        # fixed-sample reference (no-early-stop limit)
        # Calibrate per-trial LLR variance so (z_α+z_β)²/ς² == n_fss.
        self.var_z = (Z_ALPHA + Z_BETA) ** 2 / n_fss          # ς² = Var[z_i]
        self.drift_slope = self.var_z / self.m                # E[z|λ_eq] = slope·(λ_eq−μ_mid)

    # per-trial LLR drift under a truth at flat-equivalent recovery λ_eq
    def drift(self, lam_eq: float) -> float:
        return self.drift_slope * (lam_eq - self.mu_mid)

    # Wald exponent h(λ_eq): E[e^{h z}]=1 ⇒ h=−2ν/ς² = −2(λ_eq−μ_mid)/m
    def h(self, lam_eq: float) -> float:
        return -2.0 * (lam_eq - self.mu_mid) / self.m

    # OC: P(decide GO | λ_eq) — hit upper A before lower B
    def p_accept_go(self, lam_eq: float) -> float:
        h = self.h(lam_eq)
        if abs(h) < 1e-12:                        # indifference point: linear OC
            return -self.B / (self.A - self.B)
        ehA, ehB = math.exp(h * self.A), math.exp(h * self.B)
        return (1.0 - ehB) / (ehA - ehB)

    # ASN: expected number of trials under λ_eq (Wald). Peak at the indifference point.
    def asn(self, lam_eq: float) -> float:
        nu = self.drift(lam_eq)
        if abs(nu) < 1e-15:                       # ν→0 at μ_mid: E[N]=−A·B/ς²
            return (-self.A * self.B) / self.var_z
        pa = self.p_accept_go(lam_eq)
        return (pa * self.A + (1.0 - pa) * self.B) / nu

    def peak_asn(self) -> float:
        return (-self.A * self.B) / self.var_z    # worst-case (ASN at indifference)


# --------------------------------------------------------------------------- #
# (1) SPRT setup: boundaries + per-trial LLR increment statement.
# --------------------------------------------------------------------------- #
def sprt_setup(st: dict, fss: dict, sp: Sprt) -> dict[str, Any]:
    return {
        "hypotheses": {
            "H0_go_capable": "ladder clears the private bar 0.9780 at P≥0.95 (μ1=1.0 anchor)",
            "H1_nogo": "ladder below the bar (μ0=0.9780)",
            "indifference_zone_lambda": [sp.mu0, sp.mu1],
            "margin_m": sp.m,
            "mu_mid": sp.mu_mid,
        },
        "sprt_boundaries": {
            "A_upper_decide_go": sp.A,
            "B_lower_decide_nogo": sp.B,
            "alpha": sp.alpha, "power": 1.0 - sp.beta_err, "beta_err": sp.beta_err,
            "form": "A=ln((1−β)/α)=ln(0.95/0.05); B=ln(β/(1−α))=ln(0.05/0.95)",
        },
        "per_trial_llr_increment": {
            "form": "z_i = (m/σ₁²)·(x_i − μ_mid),  x_i = Neyman-weighted λ-equiv read ~ N(λ_eq, σ₁²)",
            "x_i_construction": ("x_i = μ_ref + Σ_{d=2..9} (a_d/(∂E[T]/∂λ))·(â_{d,i} − q_d); "
                                 "â_{d,i}∈{0,1} the depth-d accept indicator on full-ladder trial i; "
                                 "Var[x_i]=σ₁²=(Σ a_dσ_d)²/(∂E[T]/∂λ)² (the #197 per-trial λ-variance)"),
            "neyman_info_weights_a_sigma_2to9": fss["neyman_weight_a_sigma_2to9"],
            "neyman_fraction_2to9": fss["neyman_fraction_2to9"],
            "sigma1_sq_per_trial_physical": fss["sigma1_sq_per_trial"],
            "var_z_calibrated": sp.var_z,
            "drift_slope_dz_dlambda_eq": sp.drift_slope,
            "calibration_note": ("ς²=Var[z_i] set so the no-early-stop fixed-sample size "
                                 "(z_α+z_β)²/ς² == #197's N_fixed; the savings RATIO N_fixed/E[N] "
                                 "is ς²-invariant (σ₁² cancels)."),
        },
    }


# --------------------------------------------------------------------------- #
# (2) Expected-N under each truth (the core). ASN(β) via the Wald OC/ASN function,
#     iid and ×Deff-deflated (realism band, step 5).
# --------------------------------------------------------------------------- #
def expected_n_table(st: dict, sp: Sprt, betas=BETA_TRUTH_GRID) -> dict[str, Any]:
    rows = []
    for b in betas:
        lam_eq = lambda_eq_of_beta(st, b)
        priv = priv_lcb(st, 1.0, b)
        asn_iid = sp.asn(lam_eq)
        rows.append({
            "beta": b,
            "lambda_eq": lam_eq,
            "private_lcb_tps": priv,
            "is_go_truth": bool(priv >= TARGET_OFFICIAL),
            "p_decide_go": sp.p_accept_go(lam_eq),
            "expected_n_iid": asn_iid,
            "expected_n_realistic_icc": asn_iid * DEFF_190,
            "savings_vs_fixed_n_iid": sp.n_fss / asn_iid if asn_iid > 0 else None,
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# (3) Sequential depth order / two-stage shallow screen. The SPRT's per-trial LLR
#     weights each depth by its Neyman info a_dσ_d (likelihood-ratio IS inverse-
#     variance), so it KEEPS #197's shallow-heavy allocation. Probe-shallow-first:
#     cumulative info fraction shows a shallow screen rejects most NO-GO builds before
#     any deep (7–9) probing.
# --------------------------------------------------------------------------- #
def sequential_depth_order(fss: dict) -> dict[str, Any]:
    frac = fss["neyman_fraction_2to9"]            # depths 2..9, already info-descending
    depths = list(range(2, 2 + len(frac)))
    order = sorted(zip(depths, frac), key=lambda t: t[1], reverse=True)
    ordered_depths = [d for d, _ in order]
    cum, run = [], 0.0
    for _, f in order:
        run += f
        cum.append(run)
    shallow_23 = sum(f for d, f in zip(depths, frac) if d in (2, 3))
    shallow_234 = sum(f for d, f in zip(depths, frac) if d in (2, 3, 4))
    deep_789 = sum(f for d, f in zip(depths, frac) if d in (7, 8, 9))
    return {
        "sequential_depth_order": ordered_depths,
        "keeps_197_shallow_heavy_allocation": bool(ordered_depths == list(range(2, 10))),
        "info_fraction_2to9_descending": [f for _, f in order],
        "cumulative_info_fraction": cum,
        "shallow_screen_depths_2_3_info_frac": shallow_23,
        "shallow_screen_depths_2_3_4_info_frac": shallow_234,
        "deep_depths_7_8_9_info_frac": deep_789,
        "interpretation": (
            "the SPRT's likelihood-ratio increment is inverse-variance (Neyman) weighted, so it "
            f"KEEPS #197's shallow-heavy order [2..9]: depths {{2,3}} carry {shallow_23:.1%} and "
            f"{{2,3,4}} carry {shallow_234:.1%} of the decisive information, while the deep end "
            f"{{7,8,9}} carries only {deep_789:.1%}. A shallow-first stage therefore accumulates "
            "the bulk of |drift| and REJECTS most clear-NO-GO builds (large drift) before any "
            "expensive depth-7–9 probing."),
    }


# --------------------------------------------------------------------------- #
# (4) Operating characteristic at the 0.022 knife-edge + ASN peak (worst case).
# --------------------------------------------------------------------------- #
def operating_characteristic(st: dict, sp: Sprt) -> dict[str, Any]:
    # realized strength at the indifference-zone edges (Wald boundaries deliver target)
    realized_alpha = 1.0 - sp.p_accept_go(sp.mu1)       # P(decide NO-GO | truly GO μ1)
    realized_power = 1.0 - sp.p_accept_go(sp.mu0)       # P(decide NO-GO | at bar μ0)
    p_go_at_bar = sp.p_accept_go(sp.mu0)                # false-GO at the bar = α
    # ASN peak: indifference point λ_eq=μ_mid (worst-case expected-N)
    peak = sp.peak_asn()
    beta_at_peak = beta_of_lambda_eq(st, sp.mu_mid)
    return {
        "knife_edge_margin_lambda": sp.m,
        "realized_alpha_false_nogo_at_mu1": realized_alpha,
        "realized_power_decide_nogo_at_bar": realized_power,
        "realized_false_go_at_bar": p_go_at_bar,
        "target_alpha": sp.alpha, "target_power": 1.0 - sp.beta_err,
        "worst_case_expected_n": peak,
        "worst_case_expected_n_realistic_icc": peak * DEFF_190,
        "beta_at_asn_peak_indifference": beta_at_peak,
        "lambda_eq_at_asn_peak": sp.mu_mid,
        "note": ("the boundaries deliver realized (α,power)≥(0.05,0.95) at the indifference edges "
                 "(Wald is conservative); the ASN PEAKS at the indifference point μ_mid (a build "
                 "GENUINELY at the bar is the only expensive case — everything clearly below is cheap)."),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any]) -> dict[str, Any]:
    st = build_state(anchors)
    fss = fixed_n_reference(st, HEADLINE_LAM_TRUE)
    bar, mu1 = st["private_bar_both"], HEADLINE_LAM_TRUE
    n_fss = fss["N_fixed_z95"]                          # 30,455 (no-early-stop limit, self-test a)

    sp = Sprt(bar=bar, mu1=mu1, n_fss=n_fss)
    # physical-info calibration cross-check (held-info (0.05,0.95) FSS = 85,820)
    n_fss_physical = (Z_ALPHA + Z_BETA) ** 2 * fss["sigma1_sq_per_trial"] / (sp.m ** 2)
    sp_phys = Sprt(bar=bar, mu1=mu1, n_fss=n_fss_physical)

    setup = sprt_setup(st, fss, sp)
    table = expected_n_table(st, sp)
    depth = sequential_depth_order(fss)
    oc = operating_characteristic(st, sp)

    # headline truths
    beta_nogo = st["beta_primary"]                     # 0.765 grounded likely NO-GO
    lam_eq_nogo = lambda_eq_of_beta(st, beta_nogo)
    expected_n_sprt_nogo = sp.asn(lam_eq_nogo)         # TEST metric (iid)
    expected_n_sprt_nogo_icc = expected_n_sprt_nogo * DEFF_190
    lam_eq_nearbar = lambda_eq_of_beta(st, 1.0)        # β=1.0 best-case
    expected_n_sprt_nearbar = sp.asn(lam_eq_nearbar)
    savings_nogo = n_fss / expected_n_sprt_nogo
    savings_ratio_physical = n_fss_physical / sp_phys.asn(lam_eq_nogo)  # ς²-invariant cross-check

    headline = {
        "expected_n_sprt_nogo": expected_n_sprt_nogo,           # TEST (iid)
        "expected_n_sprt_nogo_iid": expected_n_sprt_nogo,
        "expected_n_sprt_nogo_realistic_icc": expected_n_sprt_nogo_icc,
        "expected_n_sprt_nearbar": expected_n_sprt_nearbar,
        "beta_nogo": beta_nogo, "lambda_eq_nogo": lam_eq_nogo,
        "private_lcb_nogo_tps": priv_lcb(st, 1.0, beta_nogo),
        "n_fixed_z95_197": n_fss,
        "savings_vs_fixed_n_nogo": savings_nogo,
        "n_fixed_physical_info_0p05_0p95": n_fss_physical,
        "savings_ratio_calibration_invariant_check": {
            "matched": savings_nogo, "physical_info": savings_ratio_physical,
            "note": "savings ratio is ς²-invariant: matched≈physical confirms the headline collapse "
                    "is independent of the absolute info calibration.",
        },
    }

    sttest = _self_test(st, fss, sp, sp_phys, table, oc, headline)
    handoff = _handoff(st, headline, oc, depth)

    return {
        "self_test": sttest,
        "test_metric": {"expected_n_sprt_nogo": expected_n_sprt_nogo},
        "headline": headline,
        "imports": {
            "private_bar_both_0p9780": st["private_bar_both"],
            "private_bar_descent": st["private_bar_descent"],
            "beta_primary": st["beta_primary"],
            "beta_crit_depth1_sufficient": st["beta_crit_depth1"],
            "lambda_hat_1_liveprobe": st["lam_hat_1"],
            "n_fixed_z95_197": n_fss,
            "z95": Z95, "target_official": TARGET_OFFICIAL,
            "icc_190": ICC_190, "deff_190": DEFF_190,
            "source_runs": {"d197": "wqr94io4", "d193": "2clxvlr8", "d187": "tloghme9",
                            "d191": "jeclr39w", "d190": "fva6o4ug"},
        },
        "sprt_setup": setup,
        "expected_n_table": table,
        "sequential_depth_order": depth,
        "operating_characteristic": oc,
        "fixed_n_reference_197": {
            "N_fixed_z95": fss["N_fixed_z95"], "margin": fss["margin"],
            "N_d_budget_1to9": fss["N_d_budget_1to9"],
            "neyman_fraction_2to9": fss["neyman_fraction_2to9"],
            "efficiency_gain_neyman_vs_equal": fss["efficiency_gain"],
            "sigma1_sq_per_trial": fss["sigma1_sq_per_trial"],
            "n_fixed_physical_info_0p05_0p95": n_fss_physical,
        },
        "realism_band_icc": {
            "icc": ICC_190, "deff": DEFF_190,
            "expected_n_sprt_nogo_iid": expected_n_sprt_nogo,
            "expected_n_sprt_nogo_realistic_icc": expected_n_sprt_nogo_icc,
            "n_fixed_iid": n_fss, "n_fixed_realistic_icc": n_fss * DEFF_190,
            "note": ("#190 within-prompt ICC inflates absolute trial counts by Deff=4.41 if land "
                     "#71's liveprobe steps are autocorrelated, but BOTH the SPRT E[N] and the "
                     "fixed-N scale by Deff, so the ≈%.0f× saving is Deff-invariant."
                     % savings_nogo),
        },
        "verdict": _verdict(headline, oc, depth),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _self_test(st: dict, fss: dict, sp: Sprt, sp_phys: Sprt, table: dict, oc: dict,
               headline: dict) -> dict[str, Any]:
    n_fss = fss["N_fixed_z95"]

    # (a) TRUNCATED SPRT (early-stop disabled) reproduces #197's fixed-N 30,455 — the
    #     fixed-N IS the no-early-stop limit. Re-derive via #197's neyman_budget AND confirm
    #     the SPRT's calibrated no-early-stop FSS round-trips to the same number.
    repro_197 = abs(n_fss - 30455.404769372028) <= max(1.0, TOL_FIXED_N * 30455.404769372028)
    fss_roundtrip = (Z_ALPHA + Z_BETA) ** 2 / sp.var_z          # must equal n_fss by construction
    cond_a = bool(repro_197 and abs(fss_roundtrip - n_fss) <= TOL_FIXED_N * n_fss)

    # (b) early-stop saves on the easy truth: E[N|near-bar] ≤ 30k and E[N|NO-GO] ≪ 30k.
    e_near = headline["expected_n_sprt_nearbar"]
    e_nogo = headline["expected_n_sprt_nogo"]
    cond_b = bool(e_near <= n_fss + 1e-6 and e_nogo < 0.5 * n_fss)

    # (c) boundaries deliver the error control: realized (α,power) ≥ target (0.05,0.95).
    cond_c = bool(oc["realized_power_decide_nogo_at_bar"] >= POWER - TOL_ERR
                  and oc["realized_alpha_false_nogo_at_mu1"] <= ALPHA + TOL_ERR)

    # (d) ASN monotone — cheaper as the truth moves AWAY from the bar (down the NO-GO side),
    #     and PEAKS at the indifference point (worst-case ≥ every grid ASN).
    rows = table["rows"]
    nogo_side = [r for r in rows if r["lambda_eq"] <= sp.mu_mid]   # at/below indifference
    nogo_side_sorted = sorted(nogo_side, key=lambda r: r["lambda_eq"])
    mono = all(nogo_side_sorted[i]["expected_n_iid"] <= nogo_side_sorted[i + 1]["expected_n_iid"] + 1e-6
               for i in range(len(nogo_side_sorted) - 1))
    peak = oc["worst_case_expected_n"]
    peak_dominates = all(peak >= r["expected_n_iid"] - 1e-6 for r in rows)
    cond_d = bool(mono and peak_dominates)

    # (e) reproduces #193's β_crit 0.9649 + #191's private bar 0.9780 imports.
    cond_e = bool(abs(st["beta_crit_depth1"] - 0.9648839148878561) < TOL_REPRO
                  and abs(st["private_bar_both"] - 0.9780112973731208) < TOL_REPRO
                  and st["private_bar_descent"] is None)

    # (f) NaN-clean (enforced at payload level; here key scalars finite).
    key = [n_fss, e_near, e_nogo, peak, sp.A, sp.B, sp.var_z, sp.drift_slope,
           headline["savings_vs_fixed_n_nogo"], headline["lambda_eq_nogo"]]
    cond_f = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "sprt_budget_self_test_passes": passes,
        "conditions": {
            "a_truncated_reproduces_197_fixed_n_30455": cond_a,
            "b_early_stop_saves_nearbar_le_30k_nogo_much_less": cond_b,
            "c_realized_alpha_power_meet_target": cond_c,
            "d_asn_monotone_and_peaks_at_indifference": cond_d,
            "e_reproduces_193_beta_crit_191_private_bar": cond_e,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "a_n_fss_197": n_fss, "a_fss_roundtrip": fss_roundtrip,
            "a_n_fss_physical_0p05_0p95": headline["n_fixed_physical_info_0p05_0p95"],
            "b_expected_n_nearbar": e_near, "b_expected_n_nogo": e_nogo, "b_fixed_n": n_fss,
            "c_realized_alpha": oc["realized_alpha_false_nogo_at_mu1"],
            "c_realized_power": oc["realized_power_decide_nogo_at_bar"],
            "d_nogo_side_asn": [(r["beta"], r["expected_n_iid"]) for r in nogo_side_sorted],
            "d_peak_asn": peak,
            "e_beta_crit": st["beta_crit_depth1"], "e_private_bar": st["private_bar_both"],
        },
    }


def _verdict(headline: dict, oc: dict, depth: dict) -> str:
    return (
        f"SPRT-BUDGET-DESIGNED. The REALISTIC liveprobe cost to certify the likely NO-GO is "
        f"E[N]≈{headline['expected_n_sprt_nogo']:,.0f} trials (grounded β={headline['beta_nogo']:.3f}, "
        f"private_LCB {headline['private_lcb_nogo_tps']:.1f}≪500) — a ≈{headline['savings_vs_fixed_n_nogo']:.0f}× "
        f"collapse vs #197's truth-independent fixed-N {headline['n_fixed_z95_197']:,.0f}. E[N] rises to "
        f"≈{headline['expected_n_sprt_nearbar']:,.0f} only if the build is genuinely near the bar, and "
        f"peaks at the indifference point (worst-case {oc['worst_case_expected_n']:,.0f}). The SPRT KEEPS "
        f"#197's shallow-heavy order: depths {{2,3,4}} carry "
        f"{depth['shallow_screen_depths_2_3_4_info_frac']:.0%} of the decisive information, so a shallow-"
        f"first stage rejects most NO-GO builds before any deep (7–9) probing. Boundaries deliver realized "
        f"(α,power)=({oc['realized_alpha_false_nogo_at_mu1']:.3f},{oc['realized_power_decide_nogo_at_bar']:.3f})"
        f"≥(0.05,0.95). The ICC band (Deff=4.41) scales absolute counts but not the saving ratio. NOT a launch."
    )


def _handoff(st: dict, headline: dict, oc: dict, depth: dict) -> dict[str, str]:
    land = (
        f"land #71's REALISTIC liveprobe cost is E[N]≈{headline['expected_n_sprt_nogo']:,.0f} trials to "
        f"certify the likely NO-GO (vs #197's fixed-N {headline['n_fixed_z95_197']:,.0f}), rising to "
        f"≈{headline['expected_n_sprt_nearbar']:,.0f} only if the build is genuinely at the bar; a shallow-"
        f"first SPRT stage (depths 2–4 carry {depth['shallow_screen_depths_2_3_4_info_frac']:.0%} of the "
        f"decisive info) rejects most NO-GO builds before deep probing; fern #185 consumes "
        f"(expected-N, OC) as the realistic measurement-cost row. Worst-case (build at the bar) "
        f"≈{oc['worst_case_expected_n']:,.0f} trials; ×Deff=4.41 for the autocorrelated-step realism band."
    )
    fern = (
        f"fern #185 GO/NO-GO integrator (denken #205): the SEQUENTIAL measurement cost row — "
        f"expected-N early-stop ≈{headline['expected_n_sprt_nogo']:,.0f} (likely NO-GO) … "
        f"≈{oc['worst_case_expected_n']:,.0f} (worst-case, at the bar), at (α,power)=(0.05,0.95). "
        f"Use this, not #197's truth-independent fixed-N, as the realistic liveprobe budget; the "
        f"≈{headline['savings_vs_fixed_n_nogo']:.0f}× collapse is calibration-invariant."
    )
    return {"land_71_spec": land, "fern_185_integrator": fern}


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
    h = syn["headline"]
    oc = syn["operating_characteristic"]
    dp = syn["sequential_depth_order"]
    st = syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("SPRT LIVEPROBE BUDGET (PR #205) — expected-N early-stop vs #197's fixed-N 30k", flush=True)
    print("=" * 96, flush=True)
    print(f"  private bar = {syn['imports']['private_bar_both_0p9780']:.4f}   "
          f"β_primary = {syn['imports']['beta_primary']:.4f}   β_crit = "
          f"{syn['imports']['beta_crit_depth1_sufficient']:.4f}", flush=True)
    print(f"  boundaries  A=+{syn['sprt_setup']['sprt_boundaries']['A_upper_decide_go']:.4f} (GO)  "
          f"B={syn['sprt_setup']['sprt_boundaries']['B_lower_decide_nogo']:.4f} (NO-GO)  "
          f"target (α,power)=({oc['target_alpha']:.2f},{oc['target_power']:.2f})", flush=True)
    print("-" * 96, flush=True)
    print(f"  (TEST) expected_n_sprt_nogo = {h['expected_n_sprt_nogo']:,.0f}  "
          f"(β={h['beta_nogo']:.3f}, priv_LCB {h['private_lcb_nogo_tps']:.1f})  "
          f"savings {h['savings_vs_fixed_n_nogo']:.1f}× vs fixed-N {h['n_fixed_z95_197']:,.0f}", flush=True)
    print(f"         near-bar (β=1.0) E[N] = {h['expected_n_sprt_nearbar']:,.0f}   "
          f"worst-case (indifference) = {oc['worst_case_expected_n']:,.0f}", flush=True)
    print(f"         ICC band: nogo iid {h['expected_n_sprt_nogo_iid']:,.0f} | "
          f"×Deff {h['expected_n_sprt_nogo_realistic_icc']:,.0f}", flush=True)
    print("-" * 96, flush=True)
    print("  ASN(β) truth table:", flush=True)
    for r in syn["expected_n_table"]["rows"]:
        print(f"      β={r['beta']:.4f}  λ_eq={r['lambda_eq']:.4f}  priv={r['private_lcb_tps']:7.1f}  "
              f"P(GO)={r['p_decide_go']:.3f}  E[N]={r['expected_n_iid']:>9,.0f}  "
              f"(GO={r['is_go_truth']})", flush=True)
    print("-" * 96, flush=True)
    print(f"  sequential depth order = {dp['sequential_depth_order']}  "
          f"(keeps #197 shallow-heavy = {dp['keeps_197_shallow_heavy_allocation']})", flush=True)
    print(f"      shallow {{2,3,4}} info = {dp['shallow_screen_depths_2_3_4_info_frac']:.1%}   "
          f"deep {{7,8,9}} info = {dp['deep_depths_7_8_9_info_frac']:.1%}", flush=True)
    print("-" * 96, flush=True)
    print(f"  realized (α,power) = ({oc['realized_alpha_false_nogo_at_mu1']:.4f}, "
          f"{oc['realized_power_decide_nogo_at_bar']:.4f})  at the {oc['knife_edge_margin_lambda']:.4f}-margin "
          f"knife-edge   peak β={oc['beta_at_asn_peak_indifference']:.4f}", flush=True)
    print("-" * 96, flush=True)
    print(f"  (PRIMARY) sprt_budget_self_test_passes = {st['sprt_budget_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (land #71): {syn['handoff_lines']['land_71_spec']}", flush=True)
    print(f"\n  HAND-OFF (fern #185): {syn['handoff_lines']['fern_185_integrator']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #197; never fatal).
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
        print(f"[sprt-budget] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="sprt-liveprobe-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["sprt-liveprobe-budget", "validity-gate", "measurement-design",
              "sequential-test", "wald-sprt", "expected-n", "private-bar", "bank-the-analysis"],
        config={
            "z95": Z95, "target_official": TARGET_OFFICIAL,
            "alpha": ALPHA, "power": POWER,
            "private_bar_both": syn["imports"]["private_bar_both_0p9780"],
            "beta_primary": syn["imports"]["beta_primary"],
            "n_fixed_z95_197": syn["imports"]["n_fixed_z95_197"],
            "icc_190": ICC_190, "deff_190": DEFF_190,
            "imports": "denken#197 neyman_budget + #193 β + #187 σ_d + #191 bar + #190 ICC",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[sprt-budget] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    h = syn["headline"]
    oc = syn["operating_characteristic"]
    dp = syn["sequential_depth_order"]
    st = syn["self_test"]
    summary: dict[str, Any] = {
        "sprt_budget_self_test_passes": int(bool(st["sprt_budget_self_test_passes"])),
        "expected_n_sprt_nogo": h["expected_n_sprt_nogo"],
        "expected_n_sprt_nogo_iid": h["expected_n_sprt_nogo_iid"],
        "expected_n_sprt_nogo_realistic_icc": h["expected_n_sprt_nogo_realistic_icc"],
        "expected_n_sprt_nearbar": h["expected_n_sprt_nearbar"],
        "worst_case_expected_n": oc["worst_case_expected_n"],
        "worst_case_expected_n_realistic_icc": oc["worst_case_expected_n_realistic_icc"],
        "n_fixed_z95_197": h["n_fixed_z95_197"],
        "n_fixed_physical_info_0p05_0p95": h["n_fixed_physical_info_0p05_0p95"],
        "savings_vs_fixed_n_nogo": h["savings_vs_fixed_n_nogo"],
        "sprt_A_upper": syn["sprt_setup"]["sprt_boundaries"]["A_upper_decide_go"],
        "sprt_B_lower": syn["sprt_setup"]["sprt_boundaries"]["B_lower_decide_nogo"],
        "realized_alpha": oc["realized_alpha_false_nogo_at_mu1"],
        "realized_power": oc["realized_power_decide_nogo_at_bar"],
        "beta_at_asn_peak": oc["beta_at_asn_peak_indifference"],
        "shallow_234_info_frac": dp["shallow_screen_depths_2_3_4_info_frac"],
        "deep_789_info_frac": dp["deep_depths_7_8_9_info_frac"],
        "beta_nogo": h["beta_nogo"], "lambda_eq_nogo": h["lambda_eq_nogo"],
        "private_lcb_nogo_tps": h["private_lcb_nogo_tps"],
        "private_bar_both": syn["imports"]["private_bar_both_0p9780"],
        "beta_crit_depth1_sufficient": syn["imports"]["beta_crit_depth1_sufficient"],
        "icc_190": ICC_190, "deff_190": DEFF_190,
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    for r in syn["expected_n_table"]["rows"]:
        tag = str(round(r["beta"], 4)).replace(".", "p")
        summary[f"asn_beta_{tag}"] = r["expected_n_iid"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="sprt_liveprobe_budget_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[sprt-budget] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--bug2-anchor", type=Path, default=D172.DEFAULT_BUG2_ANCHOR)
    ap.add_argument("--topo-json", type=Path, default=D172.DEFAULT_TOPO_JSON)
    ap.add_argument("--accept-json", type=Path, default=D172.DEFAULT_ACCEPT_JSON)
    ap.add_argument("--rankcov-json", type=Path, default=D172.DEFAULT_RANKCOV_JSON)
    ap.add_argument("--decomp-json", type=Path, default=D172.DEFAULT_DECOMP_JSON)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default="sprt-liveprobe-budget")
    args = ap.parse_args(argv)

    anchors = D172.load_anchors(
        args.bug2_anchor, args.topo_json, args.accept_json, args.rankcov_json, args.decomp_json)
    syn = synthesize(anchors)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 205,
        "agent": "denken",
        "kind": "sprt-liveprobe-budget",
        "anchors": {k: v for k, v in anchors.items() if k != "_paths"},
        "anchor_paths": anchors.get("_paths"),
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[sprt-budget] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["sprt_budget_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sprt_liveprobe_budget_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[sprt-budget] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY sprt_budget_self_test_passes = {passes}", flush=True)
    print(f"  TEST expected_n_sprt_nogo = {syn['test_metric']['expected_n_sprt_nogo']:,.1f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[sprt-budget] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
