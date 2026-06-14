#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Gate-2 confirmation runbook (PR #225) — the sequential test land #71 runs to PROVE
measured λ̂_built ≥ 0.9780 (both-bugs, q[2..9] direct) at (α=0.05, power=0.95).

WHAT IS BEING CONSOLIDATED
--------------------------
The launch is HELD on three hard gates: (1) land #71 builds the tree, (2) **measured
λ̂_built ≥ 0.9780 both-bugs q[2..9] direct** (THIS gate), (3) issue #192. denken's
live-probe lane already banked the two halves of gate-2's machinery:

    #205 (`eijqklu2`)  the Wald SPRT on the Neyman-weighted q[2..9] accept stream:
                       boundaries A=+2.944 / B=−2.944, indifference zone [bar, 1.0],
                       realized (α,power)=(0.05,0.95), IID expected-N.
    #212 (`b70053sw`)  the AR(1)-corrected ASN: the within-prompt design-effect band
                       that turns the IID expected-N into a realistic DECODE-STEP cost
                       (IID 405 → AR(1) 672 → measured-ACF 1,125 → flat-loose 1,788).

What did NOT yet exist as ONE banked artifact is the gate-2 CONFIRMATION PROCEDURE:
given the deployed tree emits per-position accept/reject data, the exact sequential
test (boundaries, stopping rule, AR(1)-corrected sample size) that CONFIRMS the bar,
the GO/HOLD/NO-GO decision rule, and the decode-step budget it costs — i.e. whether
gate-2 is measurable from the SAME launch-candidate run. This leg is that runbook. It
consolidates #205+#212 into the operational test; it does NOT re-derive them.

wirbel #216 made the gate sharper: a compliant int4 kernel is buildable only if λ ≥
0.857 (`lambda_min_kernel_feasible`), and below λ_crit=0.8345 (#213) even a FREE kernel
misses 500 — but the BINDING private-grade bar is stark #208's worst-case 0.97798
(`wi4gxxx8`, worst realizable domain blend). So gate-2 must cleanly separate three
verdicts: λ̂ ≥ 0.978 (GO-eligible / PASS), λ̂ ∈ [0.857, 0.978) (kernel-feasible but
below the private bar / HOLD), λ̂ < 0.857 (NO-GO — no buildable kernel clears).

THE TEST (one-sided composite confirmation of an upper-bounded parameter)
------------------------------------------------------------------------
    H0 : λ_built < 0.9780                  (NO-GO — the bar is the null boundary μ0)
    H1 : λ_built ≥ 0.9780 + margin         (GO-eligible — full-recovery anchor μ1=1.0)

statistic  per decode step i the deployed tree yields one both-bugs q[2..9] Neyman-
           weighted λ-equivalent accept read x_i ~ N(λ_built, σ²) (the #205 construction
           x_i = μ_ref + Σ_{d=2..9}(a_d/∂E[T]∂λ)(â_{d,i}−q_d)); the Wald LLR accumulates
           z_i = (m/σ²)(x_i − μ_mid) and stops at A (GO) or B (NO-GO).
caveat     the Wald boundaries control α POINTWISE at μ0=bar; the composite null λ<bar is
           covered by OC-monotonicity (worst case at λ→bar⁻). For an exactly-uniform always-
           valid guarantee the one-sided mSPRT / confidence sequence is the textbook-correct
           instrument with the SAME λ̂_LCB read-out (see composite_null_caveat). Bar, (α,power),
           and the AR(1) budget are unchanged by the instrument choice.

THE TWO COST AXES (both reported; both honest)
----------------------------------------------
(A) TRUTH-DEPENDENT expected-N (the SPRT's defining property). Rejecting a build that
    sits CLEARLY below the bar is cheap (large |drift| → fast hit of B); confirming a
    build inside the THIN GO sliver [0.978, 1.0] is dear (truth near the indifference
    midpoint μ_mid=0.989 → variance-dominated). The GO region is only m=0.022 wide
    because λ is upper-bounded at 1.0, so the GO-confirm cost can never be drift-cheap.
        reject-clear-NO-GO  (β=0.765, λ_eq=0.54)  E[N]_iid = 405  → measured-ACF 1,125  ← HEADLINE
        confirm-GO-at-anchor (λ=1.0, near-bar)     E[N]_iid = 14,915 → measured-ACF 41,379
        knife-edge-at-bar   (λ=μ_mid, peak)        E[N]_iid = 24,398 → measured-ACF 67,687
    The headline n_confirm is the drift-dominated band — the realistic cost given #215's
    shallow-only λ̂≈0.9065 (which the SPRT rejects in the drift-dominated regime). The
    near-bar/peak numbers are disclosed as the honest variance-dominated worst case.

(B) SEPARATION cost (the design knob): fixed-n ∝ 1/δ², δ = |bar − alternative|. Choosing
    a WIDE alternative (kernel floor 0.857, δ=0.121) is cheap (~1k trials); a NARROW one
    (0.97, δ=0.008) is ruinous (~231k). This is why the gate-2 zone anchors μ1 at the
    full-recovery 1.0 (δ=m=0.022 → fixed-n = #205's 30,455), not at a near-bar point.

The SPRT collapses the fixed-n worst case ~75× in the drift-dominated regime; that 75×
is Deff-invariant (#212). The AR(1)/measured-ACF inflation rests on #212's banked
within-prompt correlation (ICC 0.1446, ρ(1)=0.258, m̄=24.6); the flat ×4.41 is the
conservative loose end, measured-ACF ×2.77 the data-grounded value, AR(1) ×1.66 the
optimistic floor.

LOCAL CPU-only analytic synthesis over banked #205 SPRT + #212 ASN + #208 bar + #216
floors. No GPU / vLLM / HF Job / submission / served-file change / official draw. It
MODELS the measurement; it takes NO draws and authorizes none. BASELINE stays 481.53.
Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). This produces
the MEASUREMENT plan only: the build is land #71's, the GO/NO-GO integration is fern
#185's. NOT open2. NOT a launch.

PRIMARY metric  gate2_confirmation_self_test_passes
TEST    metric  n_confirm_measured_acf  (data-grounded AR(1)-corrected decode-step budget)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/gate2_confirmation/gate2_confirmation.py \\
        --self-test --wandb_group sprt-ar-asn --wandb_name denken/gate2-confirmation
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

_D212_PATH = REPO_ROOT / "research/validity/sprt_ar_asn/sprt_ar_asn.py"
_D212_JSON = REPO_ROOT / "research/validity/sprt_ar_asn/sprt_ar_asn_results.json"
_D208_JSON = REPO_ROOT / "research/validity/multivertex_realizability/results.json"
_D216_JSON = REPO_ROOT / "research/validity/kernel_feasibility/kernel_feasibility_results.json"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- reuse #212's exact Wald SPRT machinery (which itself wraps #205's `Sprt`). The
#      class needs only banked scalars (bar, mu1, n_fss) — NO anchor files at import or
#      instantiation — and `deff_ar1` reproduces the within-prompt design effect. ------- #
D212 = _import("sprt_ar_asn", _D212_PATH)
Sprt = D212.Sprt
deff_ar1 = D212.deff_ar1
ALPHA = D212.ALPHA                    # 0.05
POWER = D212.POWER                    # 0.95

_N01 = NormalDist(0.0, 1.0)
Z_ALPHA = _N01.inv_cdf(1.0 - ALPHA)   # 1.6449 (one-sided)
Z_BETA = _N01.inv_cdf(POWER)          # 1.6449
Z_SUM = Z_ALPHA + Z_BETA              # 3.2897

# served-run decode budget: 128 public prompts × output_len 512 (q[2..9] subset).
N_PROMPTS_SERVED = 128
OUTPUT_LEN_SERVED = 512
DECODE_STEPS_PER_RUN = N_PROMPTS_SERVED * OUTPUT_LEN_SERVED   # 65,536 positions

# named gate-2 thresholds (the binding bar + the two #216/#213 floors).
NEARBAR_ALT = 0.97                    # the PR's "near-bar" narrow-separation alternative
GO_ANCHOR = 1.0                       # full-recovery GO anchor (the SPRT μ1)

TOL_REPRO = 1e-6
TOL_DEFF = 1e-6
TOL_BAR = 5e-5                         # 0.97798 round-trip (stark #208 rounds to 5 dp)
TOL_FSS_REL = 1e-9                     # separation-cost round-trip vs #205 n_fss


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Imports — banked scalars from #212/#208/#216, NOT re-derived.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d212 = json.load(open(_D212_JSON))["synthesis"]
    enc = d212["expected_n_corrected"]
    dc = d212["deff_comparison"]
    im = d212["imports"]
    oc = d212["operating_characteristic_invariance"]

    d208 = json.load(open(_D208_JSON))["synthesis"]["beta_robustness"]
    bar_208 = d208["beta_primary"]["worstcase_bar"]              # 0.9779783… (≈0.97798)
    bar_208_band = d208["worstcase_bar_band"]

    d216 = json.load(open(_D216_JSON))["synthesis"]["headline"]
    lam_min_feasible = d216["lambda_min_kernel_feasible"]        # 0.85715… (#216 buildable floor)
    lam_crit = d216["lambda_crit_clears_500_zero_overhead_both_bugs_tau1"]  # 0.83445 (#213 free-kernel floor)

    return {
        # ---- #212 (b70053sw) banked SPRT expected-N band (the gate-2 budget) ----
        "n_confirm_iid": enc["rows"]["nogo"]["iid"],            # 405.42403511311863
        "n_confirm_ar1": enc["rows"]["nogo"]["ar1"],            # 672.3420962564048
        "n_confirm_measured_acf": enc["rows"]["nogo"]["empirical_acf"],  # 1124.7628546877863 (TEST)
        "n_confirm_flat": enc["rows"]["nogo"]["flat_441"],      # 1788.1690675618568
        "n_go_anchor_iid": enc["rows"]["nearbar"]["iid"],       # 14915.057585591705 (confirm GO@1.0)
        "n_go_anchor_ar1": enc["rows"]["nearbar"]["ar1"],       # 24734.648699561607
        "n_go_anchor_measured_acf": enc["rows"]["nearbar"]["empirical_acf"],  # 41378.65862620145
        "n_go_anchor_flat": enc["rows"]["nearbar"]["flat_441"],  # 65784.56703490077
        "n_peak_iid": enc["rows"]["worstcase"]["iid"],          # 24398.04273973794 (build at bar)
        "n_peak_ar1": enc["rows"]["worstcase"]["ar1"],          # 40460.92431498759
        "n_peak_measured_acf": enc["rows"]["worstcase"]["empirical_acf"],  # 67687.18631367176
        "n_peak_flat": enc["rows"]["worstcase"]["flat_441"],    # 107610.35744730521
        # ---- #212 design-effect band (within-prompt correlation inflation) ----
        "deff_ar1": dc["deff_ar_at_mbar"],                      # 1.658367630002036
        "deff_measured_acf": dc["deff_empirical_acf_measured"],  # 2.7742875539531413
        "deff_flat": dc["deff_flat_441"],                       # 4.410614351127293
        "deff_ar1_asymptote": dc["deff_ar_asymptote"],          # 1.6965728308550085
        "rho_lag1_190": dc["rho_lag1_190"],                     # 0.2583178258286258
        "mbar_190": dc["mbar_cluster_size"],                    # 24.582508774446666
        "icc_190": im["icc_190"],                               # 0.1446247464062406
        # ---- #205 SPRT setup (boundaries + indifference zone + calibration) ----
        "private_bar_205": im["private_bar_both"],              # 0.9780112973731208 (#191 value used by #205/#212)
        "n_fss_205": im["n_fss_197"],                           # 30455.404769372028 (fixed-N worst case)
        "sprt_A_upper": im["sprt_A_upper"],                     # +2.9444389791664403
        "sprt_B_lower": im["sprt_B_lower"],                     # −2.9444389791664394
        "mu1_go_anchor": im["mu1_go_anchor"],                   # 1.0
        "mu_mid_205": oc["mu_mid"],                             # 0.9890056486865604
        "margin_m_205": oc["margin_m"],                         # 0.021988702626879242
        "lambda_eq_nogo": im["lambda_eq_nogo"],                 # 0.5395958435497274
        "beta_nogo": im["beta_nogo"],                           # 0.765124365433998
        "realized_alpha_205": oc["realized_alpha"],             # 0.05
        "realized_power_205": oc["realized_power"],             # 0.95
        "savings_75x": d212["expected_n_corrected"]["savings_invariance"]["savings_ratio_iid"],  # 75.12
        # ---- stark #208 (wi4gxxx8) binding gate-2 bar (worst realizable blend) ----
        "private_bar_208": bar_208,                             # 0.9779783323491393 (≈0.97798) BINDING
        "private_bar_208_band": bar_208_band,                   # [0.9779783…, 0.9780155…]
        # ---- wirbel #216 / #213 kernel floors ----
        "lambda_min_kernel_feasible_216": lam_min_feasible,     # 0.8571542761568587 (≈0.857)
        "lambda_crit_free_kernel_213": lam_crit,                # 0.8344533978886615 (≈0.8345)
        "source_runs": {"d205": "eijqklu2", "d212": "b70053sw", "d208": "wi4gxxx8",
                        "d216": "kernel-feasibility", "d213": "5o7zcj8s", "d190": "fva6o4ug"},
    }


# --------------------------------------------------------------------------- #
# The Wald SPRT, reconstructed from banked scalars (no anchors). Reproduces #205/#212.
# --------------------------------------------------------------------------- #
def build_sprt(imp: dict) -> "Sprt":
    return Sprt(bar=imp["private_bar_205"], mu1=imp["mu1_go_anchor"], n_fss=imp["n_fss_205"])


# --------------------------------------------------------------------------- #
# (1) The gate-2 test definition (setup + regime recommendation).
# --------------------------------------------------------------------------- #
def gate2_test_definition(imp: dict, sp: "Sprt") -> dict[str, Any]:
    bar = imp["private_bar_208"]
    return {
        "hypotheses": {
            "H0_nogo": f"λ_built < {bar:.5f} (below the binding private bar — NO-GO; null boundary μ0)",
            "H1_go_eligible": f"λ_built ≥ {bar:.5f} + margin (GO-eligible; full-recovery anchor μ1=1.0)",
            "kind": "one-sided composite confirmation of an upper-bounded parameter (λ ≤ 1.0)",
            "indifference_zone_lambda": [bar, GO_ANCHOR],
            "margin_m": imp["margin_m_205"],
            "mu_mid": imp["mu_mid_205"],
        },
        "test_statistic": {
            "per_step_read": ("x_i = μ_ref + Σ_{d=2..9}(a_d/∂E[T]∂λ)(â_{d,i}−q_d): the both-bugs "
                              "q[2..9] Neyman-weighted λ-equivalent accept read on full-ladder decode "
                              "step i (the #205 construction), x_i ~ N(λ_built, σ²)"),
            "wald_llr_increment": "z_i = (m/σ²)(x_i − μ_mid); accumulate ΣZ, stop at A (GO) or B (NO-GO)",
            "boundaries": {"A_upper_decide_go": sp.A, "B_lower_decide_nogo": sp.B,
                           "form": "A=ln((1−β)/α)=ln(0.95/0.05); B=ln(β/(1−α))=ln(0.05/0.95)"},
            "alpha": ALPHA, "power": POWER,
            "within_prompt_correlation": {"icc_190": imp["icc_190"], "rho_lag1_190": imp["rho_lag1_190"],
                                          "mbar_190": imp["mbar_190"],
                                          "note": "prompts independent; within-prompt reads autocorrelated "
                                                  "(decaying ACF) → design-effect inflation on E[N] (#212)"},
        },
        "regime_recommendation": "SPRT (sequential)",
        "regime_rationale": (
            "Recommend the Wald SPRT over a truth-independent fixed-n test: the deployed build is "
            "expected to sit CLEARLY on one side of the bar (#215's shallow-only λ̂≈0.9065 misses; a "
            "passing build clears with margin), so the sequential test stops ~75× earlier than the "
            "fixed-n worst case. The LCB decision rule (below) is the natural read-out: a build whose "
            "always-valid λ̂ lower-confidence-bound clears the bar is exactly the SPRT GO event. Fall "
            "back to the fixed-n n_fss only as the truth-independent ceiling (build genuinely at the bar)."),
        "composite_null_caveat": (
            "The Wald boundaries A,B control type-I error POINTWISE at the null boundary μ0=bar (where "
            "the SPRT is calibrated). The null H0: λ_built<bar is COMPOSITE; in the Gaussian known-σ² "
            "MLR construction here the OC is monotone in λ, so sup_{λ<bar} P(decide GO)=P_{μ0}(decide "
            "GO)≤α — the worst-case false-GO is at λ→bar⁻ and the boundary value bounds the whole "
            "composite null (the test is STRICTLY more conservative further below the bar, where |drift| "
            "is larger). The residual approximations are (i) the Wald overshoot (boundaries ignore excess "
            "over A,B → realized α≈0.05, asymptotically exact), and (ii) the per-step read is treated as "
            "exactly Gaussian. For an EXACTLY-uniform, assumption-agnostic always-valid guarantee, the "
            "one-sided mSPRT / confidence sequence (mixture-LLR; Robbins; Johari et al. 2016; Waudby-Smith "
            "& Ramdas 2022; Fischer & Ramdas 2024) is the textbook-correct instrument and yields the SAME "
            "λ̂_LCB read-out; the Wald SPRT is its drift-dominated practical approximation, which is what "
            "prices the E[N] band here. This caveat changes the INSTRUMENT, not the bar, the (α,power) "
            "target, or the AR(1)-corrected budget."),
    }


# --------------------------------------------------------------------------- #
# (2) The confirmation sample size (the core). Reuse #212's banked E[N] band; report
#     the truth-dependent regimes + the fixed-n worst case.
# --------------------------------------------------------------------------- #
def confirmation_budget(imp: dict) -> dict[str, Any]:
    rows = {
        "reject_clear_nogo": {  # drift-dominated — the HEADLINE confirmation budget
            "truth": f"build clearly below bar (β={imp['beta_nogo']:.3f}, λ_eq={imp['lambda_eq_nogo']:.3f})",
            "regime": "drift-dominated (cheap)",
            "iid": imp["n_confirm_iid"], "ar1": imp["n_confirm_ar1"],
            "measured_acf": imp["n_confirm_measured_acf"], "flat": imp["n_confirm_flat"],
        },
        "confirm_go_at_anchor": {  # variance-dominated — the thin GO sliver
            "truth": "build at full-recovery anchor λ=1.0 (only m=0.022 above the bar)",
            "regime": "variance-dominated (dear; GO region is a thin sliver)",
            "iid": imp["n_go_anchor_iid"], "ar1": imp["n_go_anchor_ar1"],
            "measured_acf": imp["n_go_anchor_measured_acf"], "flat": imp["n_go_anchor_flat"],
        },
        "knife_edge_at_bar": {  # peak ASN — build exactly at the bar (measure-zero)
            "truth": "build exactly at the bar λ=μ_mid (indifference point)",
            "regime": "peak (worst case)",
            "iid": imp["n_peak_iid"], "ar1": imp["n_peak_ar1"],
            "measured_acf": imp["n_peak_measured_acf"], "flat": imp["n_peak_flat"],
        },
    }
    return {
        "rows": rows,
        # headline scalars (the PR's requested reports)
        "n_confirm_iid": imp["n_confirm_iid"],                       # 405 floor (import anchor)
        "n_confirm_arcorrected": imp["n_confirm_ar1"],               # 672 (AR(1) optimistic)
        "n_confirm_measured_acf": imp["n_confirm_measured_acf"],     # 1125 (TEST headline, data-grounded)
        "n_confirm_flat": imp["n_confirm_flat"],                     # 1788 (conservative loose end)
        "fixed_n_worstcase_truth_independent": imp["n_fss_205"],     # 30,455 (fixed-N, build at bar)
        "sprt_peak_at_bar_iid": imp["n_peak_iid"],                   # 24,398 (SPRT peak ASN at bar)
        "sprt_collapse_vs_fixed_n": imp["savings_75x"],              # 75.12× (Deff-invariant)
        "deff_band": {"ar1": imp["deff_ar1"], "measured_acf": imp["deff_measured_acf"],
                      "flat": imp["deff_flat"]},
        "note": (
            "HEADLINE n_confirm = the drift-dominated band (build clearly off the bar): IID 405 → "
            "AR(1) 672 → measured-ACF 1,125 → flat 1,788. The GO-side confirm of a build genuinely in "
            "the thin [0.978,1.0] sliver is the variance-dominated case (14,915 IID / 41,379 measured-"
            "ACF at λ=1.0); the build-exactly-at-bar knife-edge peaks at 24,398 IID / 67,687 measured-"
            "ACF. The SPRT collapses the fixed-n 30,455 ~75× when drift-dominated (Deff-invariant)."),
    }


# --------------------------------------------------------------------------- #
# (2b) Separation cost (the design knob). fixed-n ∝ 1/δ², δ = |bar − alternative|.
#      σ²_λ banked from #205's (n_fss, m): n_fixed(δ) = n_fss·(m/δ)² (round-trips at δ=m).
# --------------------------------------------------------------------------- #
def _n_fixed_of_sep(imp: dict, delta: float) -> float:
    """Fixed-sample one-sided (α,power)=(0.05,0.95) size to distinguish the bar from an
    alternative at separation δ. n_fixed = z_sum²·σ²_λ/δ² = n_fss·(m/δ)² (banked calibration)."""
    m = imp["margin_m_205"]
    return imp["n_fss_205"] * (m / delta) ** 2


def separation_cost(imp: dict) -> dict[str, Any]:
    bar = imp["private_bar_208"]                       # binding bar for the named deltas
    sigma_sq_lambda = imp["n_fss_205"] * imp["margin_m_205"] ** 2 / (Z_SUM ** 2)

    named = {
        "kernel_floor_0p857": {
            "alternative": imp["lambda_min_kernel_feasible_216"],
            "delta": bar - imp["lambda_min_kernel_feasible_216"],
            "width": "WIDE", "cost": "cheap"},
        "go_anchor_1p0": {
            "alternative": GO_ANCHOR, "delta": GO_ANCHOR - bar,
            "width": "gate-2 zone margin", "cost": "the standard zone"},
        "near_bar_0p97": {
            "alternative": NEARBAR_ALT, "delta": bar - NEARBAR_ALT,
            "width": "NARROW", "cost": "expensive"},
    }
    for v in named.values():
        v["n_fixed_iid"] = _n_fixed_of_sep(imp, abs(v["delta"]))
        v["n_fixed_measured_acf"] = v["n_fixed_iid"] * imp["deff_measured_acf"]

    # monotone separation grid (alternative sweeping UP toward the bar → δ↓ → cost↑)
    grid_alts = [0.84, 0.857, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.973, 0.976]
    grid = [{"alternative": a, "delta": bar - a, "n_fixed_iid": _n_fixed_of_sep(imp, bar - a)}
            for a in grid_alts if bar - a > 0]

    return {
        "sigma_sq_lambda_per_step": sigma_sq_lambda,
        "formula": "n_fixed(δ) = z_sum²·σ²_λ/δ² = n_fss·(m/δ)²,  δ = |bar − alternative|",
        "named_separations": named,
        "monotone_grid_alt_to_nfixed": grid,
        "separation_cost_ratio_near_vs_wide": (named["near_bar_0p97"]["n_fixed_iid"]
                                               / named["kernel_floor_0p857"]["n_fixed_iid"]),
        "note": (
            f"distinguishing the bar {bar:.5f} from the kernel floor 0.857 (δ="
            f"{named['kernel_floor_0p857']['delta']:.4f}, WIDE) costs only "
            f"{named['kernel_floor_0p857']['n_fixed_iid']:,.0f} fixed-n IID trials; from the near-bar "
            f"0.97 (δ={named['near_bar_0p97']['delta']:.4f}, NARROW) it costs "
            f"{named['near_bar_0p97']['n_fixed_iid']:,.0f} — a "
            f"{named['near_bar_0p97']['n_fixed_iid'] / named['kernel_floor_0p857']['n_fixed_iid']:,.0f}× "
            f"blow-up. The SPRT collapses each ~75× when drift-dominated; the gate-2 zone anchors μ1 at "
            f"the full-recovery 1.0 (δ=m=0.022 → fixed-n {imp['n_fss_205']:,.0f}), never a near-bar point."),
    }


# --------------------------------------------------------------------------- #
# (3) The decision rule (the deliverable): boundaries + 3-way GO/HOLD/NO-GO mapping.
# --------------------------------------------------------------------------- #
def decision_rule(imp: dict, sp: "Sprt") -> dict[str, Any]:
    bar = imp["private_bar_208"]
    lam_feasible = imp["lambda_min_kernel_feasible_216"]
    lam_crit = imp["lambda_crit_free_kernel_213"]
    return {
        "stopping_boundaries": {
            "A_upper_decide_go": sp.A, "B_lower_decide_nogo": sp.B,
            "rule": "accumulate ΣZ over q[2..9] reads; ΣZ≥A ⇒ GO-eligible, ΣZ≤B ⇒ NO-GO, else continue",
        },
        "lcb_three_way_mapping": {
            "PASS_go_eligible": f"λ̂_LCB ≥ {bar:.5f}  (clears the binding private bar — gate-2 PASS)",
            "HOLD_kernel_feasible_below_bar": (
                f"λ̂ ∈ [{lam_feasible:.5f}, {bar:.5f})  (a compliant int4 kernel is buildable per #216, "
                f"but below the private-grade bar — HOLD, not GO)"),
            "NOGO_no_buildable_kernel": (
                f"λ̂ < {lam_feasible:.5f}  (no buildable kernel clears 500; deeper still, λ̂ < "
                f"{lam_crit:.5f} misses even with a FREE zero-overhead kernel, #213/#216) — NO-GO"),
        },
        "thresholds": {"private_bar_go": bar, "kernel_feasible_floor": lam_feasible,
                       "free_kernel_floor_lambda_crit": lam_crit},
        "decode_steps_available_per_run": DECODE_STEPS_PER_RUN,
        "decode_steps_basis": f"{N_PROMPTS_SERVED} prompts × output_len {OUTPUT_LEN_SERVED} (q[2..9] subset)",
        "note": ("the LCB read-out is coherent with the SPRT GO event: a build whose always-valid λ̂ "
                 "lower-confidence-bound clears the bar is exactly the ΣZ≥A decision. The q[2..9] reads "
                 "are dominated by the shallow rungs (depths 2–4 carry the bulk of the Neyman info, "
                 "#205), which nearly every decode position reaches, so the effective read count ≈ the "
                 f"full {DECODE_STEPS_PER_RUN:,}."),
    }


# --------------------------------------------------------------------------- #
# (4) Feasibility (SECONDARY): does ONE served run supply ≥ n_confirm decode steps?
# --------------------------------------------------------------------------- #
def feasibility(imp: dict, budget: dict) -> dict[str, Any]:
    avail = DECODE_STEPS_PER_RUN
    checks = {
        "headline_drift_dominated_measured_acf": budget["n_confirm_measured_acf"],
        "headline_flat_loose_end": budget["n_confirm_flat"],
        "confirm_go_at_anchor_measured_acf": imp["n_go_anchor_measured_acf"],
        "knife_edge_at_bar_measured_acf": imp["n_peak_measured_acf"],
        "fixed_n_worstcase_iid": imp["n_fss_205"],
    }
    fits = {k: bool(v <= avail) for k, v in checks.items()}
    # gate-2 is confirmable in one run iff every REALISTIC regime fits (headline + GO-confirm);
    # only the measure-zero knife-edge (build exactly at the bar) under measured-ACF inflation can spill.
    confirmable = bool(fits["headline_drift_dominated_measured_acf"]
                       and fits["headline_flat_loose_end"]
                       and fits["confirm_go_at_anchor_measured_acf"])
    return {
        "decode_steps_available_per_run": avail,
        "n_confirm_vs_available": checks,
        "fits_one_run": fits,
        "gate2_confirmable_in_one_run": confirmable,
        "knife_edge_fits": fits["knife_edge_at_bar_measured_acf"],
        "knife_edge_overflow_ratio": imp["n_peak_measured_acf"] / avail,
        "headline_headroom_x": avail / budget["n_confirm_measured_acf"],
        "note": (
            f"one served run = {avail:,} q[2..9] positions. The HEADLINE drift-dominated budget "
            f"{budget['n_confirm_measured_acf']:,.0f} (measured-ACF) fits with {avail / budget['n_confirm_measured_acf']:,.0f}× "
            f"headroom; even the GO-confirm-at-anchor {imp['n_go_anchor_measured_acf']:,.0f} fits. Only the "
            f"measure-zero knife-edge (build EXACTLY at the bar, {imp['n_peak_measured_acf']:,.0f} measured-"
            f"ACF) marginally exceeds one run (×{imp['n_peak_measured_acf'] / avail:.2f}) — and #215's "
            f"shallow-only λ̂≈0.9065 puts the build firmly in the drift-dominated regime, so gate-2 is "
            f"measurable from the SAME launch-candidate run, no extra draw."),
    }


# --------------------------------------------------------------------------- #
# (d) Operating characteristic — reproduce #205/#212's realized (α,power) from the Sprt.
# --------------------------------------------------------------------------- #
def operating_characteristic(imp: dict, sp: "Sprt") -> dict[str, Any]:
    realized_alpha = 1.0 - sp.p_accept_go(sp.mu1)    # P(decide NO-GO | truly GO μ1)
    realized_power = 1.0 - sp.p_accept_go(sp.mu0)    # P(decide NO-GO | at bar μ0)
    asn_nogo_repro = sp.asn(imp["lambda_eq_nogo"])   # must reproduce the imported 405 floor
    return {
        "A_upper": sp.A, "B_lower": sp.B, "mu_mid": sp.mu_mid, "margin_m": sp.m,
        "realized_alpha": realized_alpha, "realized_power": realized_power,
        "realized_alpha_205_import": imp["realized_alpha_205"],
        "realized_power_205_import": imp["realized_power_205"],
        "asn_nogo_iid_reproduced": asn_nogo_repro,
        "reproduces_212_nogo_floor": bool(abs(asn_nogo_repro - imp["n_confirm_iid"]) < TOL_REPRO),
        "note": ("the (α,power) live on the Wald boundaries A,B (unchanged by the within-prompt "
                 "correlation, which only rescales E[N] via Deff); realized (0.05,0.95) at the "
                 "indifference edges, reproduced from #205/#212."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, sp: "Sprt", budget: dict, sep: dict, oc: dict) -> dict[str, Any]:
    # (a) IID limit reproduces #212's 405 floor (reproduce the SPRT ASN at the NO-GO truth).
    asn_repro = oc["asn_nogo_iid_reproduced"]
    cond_a = bool(abs(asn_repro - imp["n_confirm_iid"]) < TOL_REPRO
                  and abs(imp["n_confirm_iid"] - 405.42403511311863) < 1e-3)

    # (b) AR(1)/measured-ACF inflation round-trips #212's 672/1125 (E[N]_iid·Deff), and the
    #     AR(1) design effect re-derives from the banked ρ(1)/m̄.
    n_ar = imp["n_confirm_iid"] * imp["deff_ar1"]
    n_emp = imp["n_confirm_iid"] * imp["deff_measured_acf"]
    deff_ar_rederive = deff_ar1(imp["mbar_190"], imp["rho_lag1_190"])
    cond_b = bool(abs(n_ar - imp["n_confirm_ar1"]) < 1e-6
                  and abs(n_emp - imp["n_confirm_measured_acf"]) < 1e-6
                  and abs(deff_ar_rederive - imp["deff_ar1"]) < TOL_DEFF)

    # (c) n_confirm monotone ↑ as the alternative approaches the bar (separation cost), AND the
    #     fixed-n round-trips #205's n_fss at δ = the gate-2 margin m.
    grid = sep["monotone_grid_alt_to_nfixed"]
    mono = all(grid[i + 1]["n_fixed_iid"] > grid[i]["n_fixed_iid"] - 1e-9 for i in range(len(grid) - 1))
    n_at_margin = _n_fixed_of_sep(imp, imp["margin_m_205"])
    roundtrip_fss = abs(n_at_margin - imp["n_fss_205"]) <= TOL_FSS_REL * imp["n_fss_205"]
    # cost strictly increases as alternative rises toward the bar (δ shrinks)
    strict_up = all(grid[i + 1]["alternative"] > grid[i]["alternative"] for i in range(len(grid) - 1))
    cond_c = bool(mono and strict_up and roundtrip_fss)

    # (d) realized (α,power) = (0.05,0.95) at the reported n.
    cond_d = bool(abs(oc["realized_alpha"] - 0.05) < TOL_REPRO
                  and abs(oc["realized_power"] - 0.95) < TOL_REPRO
                  and oc["reproduces_212_nogo_floor"])

    # (e) the bar 0.97798 round-trips stark #208 (worst realizable blend), inside its band.
    bar = imp["private_bar_208"]
    lo, hi = imp["private_bar_208_band"]
    cond_e = bool(abs(bar - 0.97798) < TOL_BAR and (lo - 1e-12) <= bar <= (hi + 1e-12))

    # (f) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [budget["n_confirm_iid"], budget["n_confirm_arcorrected"], budget["n_confirm_measured_acf"],
           budget["n_confirm_flat"], budget["fixed_n_worstcase_truth_independent"],
           sep["named_separations"]["kernel_floor_0p857"]["n_fixed_iid"],
           sep["named_separations"]["near_bar_0p97"]["n_fixed_iid"],
           imp["private_bar_208"], imp["lambda_min_kernel_feasible_216"], imp["lambda_crit_free_kernel_213"]]
    cond_f = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "gate2_confirmation_self_test_passes": passes,
        "conditions": {
            "a_iid_limit_reproduces_212_405_floor": cond_a,
            "b_ar1_measured_acf_roundtrip_212_672_1125": cond_b,
            "c_n_confirm_monotone_as_alt_to_bar_and_fss_roundtrip": cond_c,
            "d_realized_alpha_power_0p05_0p95": cond_d,
            "e_bar_0p97798_roundtrips_stark_208": cond_e,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "a_asn_reproduced": asn_repro, "a_n_confirm_iid_import": imp["n_confirm_iid"],
            "b_n_ar_computed": n_ar, "b_n_ar_import": imp["n_confirm_ar1"],
            "b_n_emp_computed": n_emp, "b_n_emp_import": imp["n_confirm_measured_acf"],
            "b_deff_ar_rederive": deff_ar_rederive, "b_deff_ar_import": imp["deff_ar1"],
            "c_n_fixed_at_margin": n_at_margin, "c_n_fss_205": imp["n_fss_205"],
            "c_grid_alt_nfixed": [(g["alternative"], g["n_fixed_iid"]) for g in grid],
            "d_realized_alpha": oc["realized_alpha"], "d_realized_power": oc["realized_power"],
            "e_bar_208": bar, "e_bar_208_band": imp["private_bar_208_band"],
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, budget: dict, sep: dict, feas: dict) -> str:
    return (
        f"GATE-2 RUNBOOK BANKED. The confirmation procedure is a Wald SPRT (boundaries A=+{imp['sprt_A_upper']:.3f}/"
        f"B={imp['sprt_B_lower']:.3f}, indifference zone [{imp['private_bar_208']:.5f}, 1.0]) on the deployed "
        f"tree's both-bugs q[2..9] Neyman-weighted accept stream, realized (α,power)=(0.05,0.95). The "
        f"AR(1)-corrected confirmation budget is n_confirm = {budget['n_confirm_iid']:,.0f} IID → "
        f"{budget['n_confirm_arcorrected']:,.0f} AR(1) → {budget['n_confirm_measured_acf']:,.0f} measured-ACF "
        f"(HEADLINE, data-grounded) → {budget['n_confirm_flat']:,.0f} flat, vs the truth-independent fixed-n "
        f"ceiling {budget['fixed_n_worstcase_truth_independent']:,.0f} (~{budget['sprt_collapse_vs_fixed_n']:.0f}× "
        f"SPRT collapse). HONEST: that headline is the drift-dominated cost (build clearly off the bar); a "
        f"GO-confirm of a build genuinely in the thin [0.978,1.0] sliver costs {imp['n_go_anchor_measured_acf']:,.0f} "
        f"measured-ACF, and the build-exactly-at-bar knife-edge peaks at {imp['n_peak_measured_acf']:,.0f}. "
        f"Separation cost: distinguishing the bar from the kernel floor 0.857 (wide) is "
        f"{sep['named_separations']['kernel_floor_0p857']['n_fixed_iid']:,.0f} IID, from 0.97 (narrow) "
        f"{sep['named_separations']['near_bar_0p97']['n_fixed_iid']:,.0f} — a "
        f"{sep['separation_cost_ratio_near_vs_wide']:,.0f}× swing. Decision rule: λ̂_LCB ≥ "
        f"{imp['private_bar_208']:.5f} ⇒ PASS; [{imp['lambda_min_kernel_feasible_216']:.3f},"
        f"{imp['private_bar_208']:.3f}) ⇒ HOLD; < {imp['lambda_min_kernel_feasible_216']:.3f} ⇒ NO-GO. "
        f"gate2_confirmable_in_one_run = {feas['gate2_confirmable_in_one_run']} (headline fits with "
        f"{feas['headline_headroom_x']:,.0f}× headroom in one served run's {feas['decode_steps_available_per_run']:,} "
        f"q[2..9] positions). NOT a launch."
    )


def _handoff(imp: dict, budget: dict, feas: dict) -> dict[str, str]:
    land_fern = (
        f"gate-2 (measured λ̂_built ≥ {imp['private_bar_208']:.5f} both-bugs, q[2..9] direct) is confirmed by a "
        f"Wald SPRT over the deployed tree's q[2..9] Neyman-weighted accept stream (boundaries A=+"
        f"{imp['sprt_A_upper']:.3f}/B={imp['sprt_B_lower']:.3f}, (α,power)=(0.05,0.95)): AR(1)-corrected "
        f"confirmation budget n_confirm = {budget['n_confirm_measured_acf']:,.0f} decode steps (measured-ACF; "
        f"AR(1) {budget['n_confirm_arcorrected']:,.0f}; IID floor {budget['n_confirm_iid']:,.0f}; flat "
        f"{budget['n_confirm_flat']:,.0f}), which DOES fit one served run's ~{feas['decode_steps_available_per_run']:,} "
        f"q[2..9] positions — so gate-2 is measurable from the SAME launch-candidate run with the decision rule "
        f"λ̂_LCB ≥ {imp['private_bar_208']:.5f} ⇒ PASS, [{imp['lambda_min_kernel_feasible_216']:.3f},"
        f"{imp['private_bar_208']:.3f}) ⇒ HOLD (kernel-feasible, below private bar), < "
        f"{imp['lambda_min_kernel_feasible_216']:.3f} ⇒ NO-GO; fern #185 reads gate-2 as a single PASS/HOLD/NO-GO "
        f"flag. CAVEAT: confirming a build genuinely in the thin [0.978,1.0] GO sliver is the variance-dominated "
        f"case ({imp['n_go_anchor_measured_acf']:,.0f} measured-ACF), still inside one run."
    )
    return {"land_71_and_fern_185": land_fern}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    sp = build_sprt(imp)
    test_def = gate2_test_definition(imp, sp)
    budget = confirmation_budget(imp)
    sep = separation_cost(imp)
    rule = decision_rule(imp, sp)
    oc = operating_characteristic(imp, sp)
    feas = feasibility(imp, budget)
    st = self_test(imp, sp, budget, sep, oc)
    handoff = _handoff(imp, budget, feas)
    return {
        "self_test": st,
        "test_metric": {"n_confirm_measured_acf": budget["n_confirm_measured_acf"]},
        "imports": imp,
        "gate2_test_definition": test_def,
        "confirmation_budget": budget,
        "separation_cost": sep,
        "gate2_decision_rule": rule,
        "operating_characteristic": oc,
        "feasibility": feas,
        "verdict": _verdict(imp, budget, sep, feas),
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
    td = syn["gate2_test_definition"]
    bg = syn["confirmation_budget"]
    sep = syn["separation_cost"]
    rule = syn["gate2_decision_rule"]
    feas = syn["feasibility"]
    oc = syn["operating_characteristic"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("GATE-2 CONFIRMATION RUNBOOK (PR #225) — prove measured λ̂_built ≥ 0.9780 (both-bugs q[2..9])",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  binding bar (stark #208) = {imp['private_bar_208']:.7f}   zone [bar,1.0]  margin m = "
          f"{imp['margin_m_205']:.5f}   μ_mid = {imp['mu_mid_205']:.5f}", flush=True)
    print(f"  boundaries  A=+{oc['A_upper']:.4f} (GO)  B={oc['B_lower']:.4f} (NO-GO)   "
          f"realized (α,power)=({oc['realized_alpha']:.4f},{oc['realized_power']:.4f})   "
          f"regime: {td['regime_recommendation']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (TEST) n_confirm_measured_acf = {bg['n_confirm_measured_acf']:,.2f}   "
          f"[IID {bg['n_confirm_iid']:,.0f} | AR(1) {bg['n_confirm_arcorrected']:,.0f} | "
          f"flat {bg['n_confirm_flat']:,.0f}]", flush=True)
    print(f"  fixed-n worst case (truth-indep) = {bg['fixed_n_worstcase_truth_independent']:,.0f}   "
          f"SPRT collapse ≈ {bg['sprt_collapse_vs_fixed_n']:.1f}×", flush=True)
    print("  truth-dependent E[N] regimes (IID → measured-ACF):", flush=True)
    for k, r in bg["rows"].items():
        print(f"      {k:22s} {r['regime']:34s} iid={r['iid']:>10,.0f}  meas-ACF={r['measured_acf']:>11,.0f}",
              flush=True)
    print("-" * 100, flush=True)
    print("  SEPARATION cost  n_fixed(δ) = n_fss·(m/δ)²:", flush=True)
    for k, v in sep["named_separations"].items():
        print(f"      {k:18s} alt={v['alternative']:.4f}  δ={v['delta']:+.4f}  ({v['width']:>16s})  "
              f"n_fixed_iid={v['n_fixed_iid']:>12,.0f}", flush=True)
    print(f"      near/wide cost ratio = {sep['separation_cost_ratio_near_vs_wide']:,.0f}×", flush=True)
    print("-" * 100, flush=True)
    print("  DECISION RULE (λ̂ LCB three-way):", flush=True)
    for k, v in rule["lcb_three_way_mapping"].items():
        print(f"      {k:32s} : {v}", flush=True)
    print(f"  decode steps available / served run = {rule['decode_steps_available_per_run']:,}  "
          f"({rule['decode_steps_basis']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (SECONDARY) gate2_confirmable_in_one_run = {feas['gate2_confirmable_in_one_run']}  "
          f"(headline headroom {feas['headline_headroom_x']:,.0f}×; knife-edge fits={feas['knife_edge_fits']} "
          f"×{feas['knife_edge_overflow_ratio']:.2f})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) gate2_confirmation_self_test_passes = {st['gate2_confirmation_self_test_passes']}",
          flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF (land #71 + fern #185): {syn['handoff_lines']['land_71_and_fern_185']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #205/#212; never fatal).
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
        print(f"[gate2-confirmation] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp = syn["imports"]
    bg = syn["confirmation_budget"]
    sep = syn["separation_cost"]
    feas = syn["feasibility"]
    oc = syn["operating_characteristic"]
    st = syn["self_test"]

    run = init_wandb_run(
        job_type="sprt-liveprobe-budget",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["sprt-liveprobe-budget", "validity-gate", "measurement-design", "sequential-test",
              "wald-sprt", "expected-n", "ar1-correction", "gate2-confirmation", "confirmation-runbook",
              "bank-the-analysis"],
        config={
            "alpha": ALPHA, "power": POWER,
            "private_bar_208": imp["private_bar_208"], "private_bar_205": imp["private_bar_205"],
            "lambda_min_kernel_feasible_216": imp["lambda_min_kernel_feasible_216"],
            "lambda_crit_free_kernel_213": imp["lambda_crit_free_kernel_213"],
            "n_fss_205": imp["n_fss_205"], "deff_measured_acf": imp["deff_measured_acf"],
            "decode_steps_per_run": DECODE_STEPS_PER_RUN,
            "imports": "denken#205 SPRT (eijqklu2) + denken#212 ASN (b70053sw) + stark#208 bar (wi4gxxx8) "
                       "+ wirbel#216 kernel floors + #213 lambda_crit",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[gate2-confirmation] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    nm = sep["named_separations"]
    summary: dict[str, Any] = {
        "gate2_confirmation_self_test_passes": int(bool(st["gate2_confirmation_self_test_passes"])),
        "n_confirm_measured_acf": bg["n_confirm_measured_acf"],
        "n_confirm_arcorrected": bg["n_confirm_arcorrected"],
        "n_confirm_iid": bg["n_confirm_iid"],
        "n_confirm_flat": bg["n_confirm_flat"],
        "fixed_n_worstcase": bg["fixed_n_worstcase_truth_independent"],
        "sprt_peak_at_bar_iid": bg["sprt_peak_at_bar_iid"],
        "sprt_collapse_vs_fixed_n": bg["sprt_collapse_vs_fixed_n"],
        "n_go_anchor_measured_acf": imp["n_go_anchor_measured_acf"],
        "n_peak_measured_acf": imp["n_peak_measured_acf"],
        "deff_ar1": imp["deff_ar1"], "deff_measured_acf": imp["deff_measured_acf"],
        "deff_flat": imp["deff_flat"],
        "sep_n_fixed_kernel_floor_0p857": nm["kernel_floor_0p857"]["n_fixed_iid"],
        "sep_n_fixed_go_anchor_1p0": nm["go_anchor_1p0"]["n_fixed_iid"],
        "sep_n_fixed_near_bar_0p97": nm["near_bar_0p97"]["n_fixed_iid"],
        "separation_cost_ratio_near_vs_wide": sep["separation_cost_ratio_near_vs_wide"],
        "sigma_sq_lambda_per_step": sep["sigma_sq_lambda_per_step"],
        "private_bar_208": imp["private_bar_208"],
        "lambda_min_kernel_feasible_216": imp["lambda_min_kernel_feasible_216"],
        "lambda_crit_free_kernel_213": imp["lambda_crit_free_kernel_213"],
        "decode_steps_available_per_run": feas["decode_steps_available_per_run"],
        "gate2_confirmable_in_one_run": int(bool(feas["gate2_confirmable_in_one_run"])),
        "headline_headroom_x": feas["headline_headroom_x"],
        "knife_edge_overflow_ratio": feas["knife_edge_overflow_ratio"],
        "realized_alpha": oc["realized_alpha"], "realized_power": oc["realized_power"],
        "sprt_A_upper": oc["A_upper"], "sprt_B_lower": oc["B_lower"],
        "margin_m": oc["margin_m"], "mu_mid": oc["mu_mid"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="gate2_confirmation_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[gate2-confirmation] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default="sprt-ar-asn")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 225,
        "agent": "denken",
        "kind": "gate2-confirmation",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[gate2-confirmation] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["gate2_confirmation_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate2_confirmation_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[gate2-confirmation] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY gate2_confirmation_self_test_passes = {passes}", flush=True)
    print(f"  TEST n_confirm_measured_acf = {syn['test_metric']['n_confirm_measured_acf']:,.2f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[gate2-confirmation] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
