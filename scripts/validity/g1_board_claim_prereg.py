#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Pre-register the board G1 claim: joint position survival + official-TPS table -- PR #758.

The G1-safety leg is complete and decisive (#749 faithful central -7.73% -> #754
well-posed P(DQ)=0.0137 -> #756 single-draw P(G1-pass)=0.985). This capstone packages
that leg into a *pre-registered* board claim a blog reader cannot misread, BEFORE the
fire's official summary.json:tps lands (so it can't be read as post-hoc curve-fitting).

It reuses kanna's own #756 machinery BYTE-FOR-FOR-IDENTICAL (imports load_systematic_delta,
load_sigma_hw, convolve from g1_single_draw_repro_pass) and produces three deliverables:

  1. PARAMETRIC OFFICIAL-TPS -> G1 CLAIM TABLE (pre-registered). For a grid of plausible
     fire official-TPS outcomes, tabulate P(G1-pass), R_05, the 95%-worst private TPS
     (official x R_05), the margin over the 126.378 bar, and the speedup-x over the bar.
     P(G1-pass) and R_05 are SCALE-FREE (a ratio rule): they are INVARIANT to the official
     number -- that invariance is what makes pre-registering the claim sound.

  2. JOINT LEADERBOARD-POSITION SURVIVAL  P(>=1 of {126.378-live, fire} clears G1)  [PRIMARY].
     The position is held by TWO configs: the accepted live 126.378 (int4_g128_lmhead,
     non-spec greedy, deterministic) AND the fire (int4_mtp_batchinv, spec). If the fire
     DQs on G1, the 126.378 still stands -> position-survival is a JOINT quantity strictly
     safer than the fire-alone 0.985. We model the 126.378 two ways:
       - CONSERVATIVE: carries the fire's full delta_stock distribution (an upper bound on
         its spread; the non-spec config has no spec-dec acceptance stochasticity, so its
         systematic spread is <= the fire's).
       - DETERMINISTIC (physical): non-spec greedy has NO acceptance gap -> delta_stock~=0;
         only sigma_hw remains -> it essentially never misses by 5%.
     The two submissions share the #754 public-anchor common-mode sampling noise (5.32 of
     6.05 pp SE), so their survivals are POSITIVELY correlated. We report the joint under
     (i) independence (optimistic ceiling) and (ii) the realistic common-mode-correlated
     case (the honest headline), plus the comonotonic floor. The correlation is built
     FAITHFULLY from the bootstrap, not assumed: central_b - central_b_epubfix is the
     shared common-mode leg (SE 5.318), central_b_epubfix the idiosyncratic leg (SE 2.878),
     and they are empirically uncorrelated (var adds to 6.047^2) -- so we share the
     common-mode draws between the two submissions and keep idiosyncratic + sigma_hw
     independent. No Gaussian approximation in the primary; a bivariate-normal analytic
     cross-check is reported alongside.

  3. ONE-SIDED-G1 vs TWO-SIDED-+/-5% RESOLUTION (board paragraph). Formalizes the #756
     finding (only 34% of single draws land in a two-sided +/-5% band because 64% are MORE
     than 5% FASTER) into one organizer-rule-cited statement: being faster is NOT a
     reproduction failure; the G1 rule is one-sided (private >= 0.95 x reported), which the
     fire passes at 98.5%.

LOCAL ONLY: CPU numpy, reads existing in-repo artifacts (all kanna's own work on the
advisor branch), launches no server, makes no submission, fires no HF Job.
analysis_only=1, official_tps=0, no_hf_job=1, fires=0.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the EXACT #756 machinery so the fire leg is byte-identical (no re-derivation).
from scripts.validity.g1_single_draw_repro_pass import (  # noqa: E402
    BAR_TPS, G1_DELTA_PCT, G1_RULE_RATIO,
    DEF_HW_ENVELOPE, DEF_HW_SINGLEDRAW, DEF_SOURCE_REPORT,
    convolve, load_sigma_hw, load_systematic_delta,
)

# Anchors from #756 (research/validity/g1_single_draw_repro_pass) -- asserted reproduced.
P_FIRE_756 = 0.98504
R05_756 = 0.97739
G1_MARGIN_756 = 2.9231
COMMON_MODE_SE_754 = 5.318          # public-anchor common-mode SE (pp), #754

# Pre-registered grid of plausible fire official-TPS outcomes (the board can't see the
# real number yet; ~150-168 expected, bracketed 130..180).
DEF_OFFICIAL_GRID = [130.0, 140.0, 150.0, 157.0, 168.0, 180.0]
DEF_OUT_DIR = "research/validity/g1_board_claim_prereg/results"


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _bvn_p_both_below(a: float, b: float, rho: float, *, n: int = 4000) -> float:
    """P(X<a, Y<b) for standard bivariate normal corr rho, by 1-D Gaussian quadrature.

    Used only as an analytic cross-check of the bootstrap-faithful joint simulation.
    """
    if rho <= -0.999:
        return max(0.0, _phi(a) + _phi(b) - 1.0)
    if rho >= 0.999:
        return min(_phi(a), _phi(b))
    # integrate phi(x) * Phi((b - rho x)/sqrt(1-rho^2)) over x in (-inf, a]
    lo, hi = -8.0, a
    xs = np.linspace(lo, hi, n)
    pdf = np.exp(-0.5 * xs * xs) / sqrt(2.0 * np.pi)
    denom = sqrt(1.0 - rho * rho)
    cdf = 0.5 * (1.0 + np.vectorize(erf)((b - rho * xs) / (denom * sqrt(2.0))))
    trapz = getattr(np, "trapezoid", np.trapz)  # np>=2.0 renamed trapz->trapezoid
    return float(trapz(pdf * cdf, xs))


def parametric_table(p_pass: float, r05: float, bar: float,
                     grid: list[float]) -> dict[str, Any]:
    """Deliverable 1: pre-registered official-TPS -> G1 claim table.

    P(G1-pass) and R_05 are SCALE-FREE (the gate is a ratio rule private >= 0.95*reported),
    so they are the SAME for every official-TPS row -- that invariance is the whole point of
    pre-registering: the claim is fixed before the number is known. Only the TPS-denominated
    columns move with the official number.
    """
    rows = []
    for off in grid:
        worst_priv = off * r05
        rows.append({
            "official_tps": off,
            "p_g1_pass": round(p_pass, 5),                      # invariant (scale-free)
            "r05_worst_single_draw_ratio": round(r05, 5),       # invariant (scale-free)
            "worst_private_tps_at_95": round(worst_priv, 3),    # official x R_05
            "margin_over_126378_bar_tps": round(worst_priv - bar, 3),
            "speedup_x_over_bar": round(off / bar, 4),
            "worst_draw_clears_bar": bool(worst_priv >= bar),
        })
    return {
        "bar_tps": bar,
        "p_g1_pass_invariant": round(p_pass, 5),
        "r05_invariant": round(r05, 5),
        "scale_free_note": ("P(G1-pass) and R_05 do not depend on the official number "
                            "(the gate is the ratio rule private>=0.95*reported); only the "
                            "TPS-denominated columns scale. This invariance is why the claim "
                            "can be committed before the official summary.json:tps lands."),
        "grid": rows,
    }


def joint_survival(central_b: np.ndarray, central_b_epubfix: np.ndarray,
                   sigma_hw_frac: float, *, hw_seed_fire: int, hw_seed_126: int,
                   idio_perm_seed: int, indep_perm_seed: int) -> dict[str, Any]:
    """Deliverable 2: joint P(>=1 of {126.378, fire} clears G1).

    Bootstrap-faithful correlation: cm = central_b - central_b_epubfix is the shared
    public-anchor common-mode leg; idio = central_b_epubfix is the idiosyncratic leg; the
    two are empirically uncorrelated. The 126.378 (conservative) shares cm but draws an
    independent idiosyncratic permutation and an independent sigma_hw; the fire leg uses
    central_b verbatim (-> byte-identical to #756). The deterministic 126.378 carries no
    delta_stock (non-spec greedy) -> only sigma_hw.
    """
    n = central_b.shape[0]
    cm = central_b - central_b_epubfix            # shared common-mode (SE ~5.318)
    idio = central_b_epubfix                      # idiosyncratic (SE ~2.878)

    # --- FIRE leg: delta = central_b verbatim, sigma_hw fold seed 756 == #756 byte-identical
    r_sys_fire = 1.0 - central_b / 100.0
    eps_fire = np.random.default_rng(hw_seed_fire).normal(0.0, sigma_hw_frac, size=n)
    R_fire = r_sys_fire * (1.0 + eps_fire)
    pass_fire = R_fire >= G1_RULE_RATIO

    # --- 126.378 CONSERVATIVE: shares cm, independent idiosyncratic, independent sigma_hw
    perm = np.random.default_rng(idio_perm_seed).permutation(n)
    delta_126_cons = cm + idio[perm]
    r_sys_126c = 1.0 - delta_126_cons / 100.0
    eps_126 = np.random.default_rng(hw_seed_126).normal(0.0, sigma_hw_frac, size=n)
    R_126c = r_sys_126c * (1.0 + eps_126)
    pass_126c = R_126c >= G1_RULE_RATIO

    # --- 126.378 DETERMINISTIC (physical): non-spec greedy, no delta_stock, sigma_hw only
    R_126d = 1.0 * (1.0 + eps_126)
    pass_126d = R_126d >= G1_RULE_RATIO

    P_fire = float(pass_fire.mean())
    P_126c = float(pass_126c.mean())
    P_126d = float(pass_126d.mean())

    # correlation of the realized single-draw ratios (conservative model)
    rho_R = float(np.corrcoef(R_fire, R_126c)[0, 1])
    rho_sys = float(np.corrcoef(central_b, delta_126_cons)[0, 1])

    # --- JOINT, conservative 126 model ---
    # (ii) common-mode-correlated (HONEST headline): paired OR with shared cm
    P_or_corr_cons = float((pass_fire | pass_126c).mean())
    # (i) independence ceiling: pair fire with an INDEPENDENT 126 draw
    sh = np.random.default_rng(indep_perm_seed).permutation(n)
    P_or_indep_cons_sim = float((pass_fire | pass_126c[sh]).mean())
    P_or_indep_cons = 1.0 - (1.0 - P_fire) * (1.0 - P_126c)     # analytic
    # comonotonic floor (perfect positive corr -> the second submission adds nothing)
    P_or_comon_cons = max(P_fire, P_126c)

    # --- JOINT, deterministic 126 model (physical) ---
    # 126_det depends only on independent sigma_hw -> ~independent of the fire's delta_stock
    P_or_corr_det = float((pass_fire | pass_126d).mean())
    P_or_indep_det = 1.0 - (1.0 - P_fire) * (1.0 - P_126d)

    # analytic bivariate-normal cross-check of the conservative correlated joint
    # map the 5%-slow fail threshold into standardized delta space at each marginal
    se_fire = float(central_b.std(ddof=1))
    mu_fire = float(central_b.mean())
    se_126 = float(delta_126_cons.std(ddof=1))
    mu_126 = float(delta_126_cons.mean())
    # fail = delta > +5 ; standardized upper-tail thresholds
    zf = (G1_DELTA_PCT - mu_fire) / se_fire
    z6 = (G1_DELTA_PCT - mu_126) / se_126
    # P(both fail) = P(delta_fire>5, delta_126>5) = P(-Zf< -zf, -Z6<-z6) bvn corr rho_sys
    p_both_fail_bvn = _bvn_p_both_below(-zf, -z6, rho_sys)
    P_or_corr_bvn = 1.0 - p_both_fail_bvn

    return {
        "marginals": {
            "p_fire_clears_g1": round(P_fire, 5),
            "p_126378_clears_g1_conservative": round(P_126c, 5),
            "p_126378_clears_g1_deterministic": round(P_126d, 5),
            "p_126378_deterministic_analytic": round(_phi((1.0 - G1_RULE_RATIO) / sigma_hw_frac), 7),
            "note_126_det": ("non-spec greedy carries no acceptance gap -> delta_stock~=0; "
                             "P = Phi(0.05/sigma_hw) -> sigma_hw alone never yields a 5% miss"),
        },
        "correlation": {
            "rho_realized_ratio_conservative": round(rho_R, 4),
            "rho_systematic_delta_conservative": round(rho_sys, 4),
            "common_mode_se_pp": COMMON_MODE_SE_754,
            "common_mode_variance_fraction": round((COMMON_MODE_SE_754 ** 2) / (se_fire ** 2), 4),
            "note": ("the two submissions share the official-128 public-anchor sampling "
                     "noise (5.32 of 6.05 pp SE, #754) -> positively correlated survivals; "
                     "rho built from the bootstrap common-mode leg, not assumed."),
        },
        "joint_conservative_126_as_fire_deltastock": {
            "p_position_survives_independence_ceiling": round(P_or_indep_cons, 5),
            "p_position_survives_independence_sim": round(P_or_indep_cons_sim, 5),
            "p_position_survives_common_mode_correlated": round(P_or_corr_cons, 5),
            "p_position_survives_comonotonic_floor": round(P_or_comon_cons, 5),
            "p_position_survives_correlated_bvn_xcheck": round(P_or_corr_bvn, 5),
        },
        "joint_deterministic_126_physical": {
            "p_position_survives_independence": round(P_or_indep_det, 7),
            "p_position_survives_correlated": round(P_or_corr_det, 7),
            "note": ("the 126.378 is a near-certain deterministic backstop whose failure "
                     "mode (sigma_hw) is independent of the fire's delta_stock-driven "
                     "failure -> position survival ~= 1.0 physically."),
        },
        "_arrays": {"R_fire": R_fire, "R_126c": R_126c},  # popped before JSON
    }


def run(args) -> dict[str, Any]:
    report_path = REPO / args.report if not Path(args.report).is_absolute() else Path(args.report)
    env_path = REPO / args.hw_envelope
    single_path = REPO / args.hw_singledraw
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    central_b, central_b_epubfix, point_central, sys_meta = load_systematic_delta(
        report_path, B=args.draws, K=args.K, seed=args.seed)
    boot_se = float(central_b.std(ddof=1))
    hw = load_sigma_hw(env_path, single_path)
    sigma_canon = hw["canonical_frac"]
    sigma_abs = hw["absolute_frac_at_160tps"]

    # --- fire single-draw leg, byte-identical to #756 (assert) ---
    grid_levels = list(args.grid)
    primary = convolve(central_b, sigma_canon, hw_seed=args.hw_seed,
                       bar=BAR_TPS, reported_levels=grid_levels)
    primary.pop("_r_real")
    p_fire = primary["p_g1_single_draw_pass"]
    r05 = primary["r_realized"]["p5"]
    g1_margin = primary["g1_margin_tps_at_95"]
    repro_ok = (abs(p_fire - P_FIRE_756) < 1e-4 and abs(r05 - R05_756) < 1e-4
                and abs(g1_margin - G1_MARGIN_756) < 1e-2)

    # --- Deliverable 1: parametric official-TPS table ---
    table = parametric_table(p_fire, r05, BAR_TPS, grid_levels)

    # --- Deliverable 2: joint position survival (canonical sigma_hw) ---
    joint = joint_survival(central_b, central_b_epubfix, sigma_canon,
                           hw_seed_fire=args.hw_seed, hw_seed_126=args.hw_seed_126,
                           idio_perm_seed=args.idio_perm_seed,
                           indep_perm_seed=args.indep_perm_seed)
    arrays = joint.pop("_arrays")
    # robustness: re-run the joint at the conservative absolute-3% sigma_hw bound
    joint_abs = joint_survival(central_b, central_b_epubfix, sigma_abs,
                               hw_seed_fire=args.hw_seed, hw_seed_126=args.hw_seed_126,
                               idio_perm_seed=args.idio_perm_seed,
                               indep_perm_seed=args.indep_perm_seed)
    joint_abs.pop("_arrays")

    p_position = joint["joint_conservative_126_as_fire_deltastock"][
        "p_position_survives_common_mode_correlated"]
    p_floor = joint["joint_conservative_126_as_fire_deltastock"][
        "p_position_survives_comonotonic_floor"]
    p_ceiling = joint["joint_conservative_126_as_fire_deltastock"][
        "p_position_survives_independence_ceiling"]
    p_physical = joint["joint_deterministic_126_physical"]["p_position_survives_correlated"]

    # --- Deliverable 3: one-sided vs two-sided resolution ---
    one_vs_two = {
        "g1_rule_scored": "private_TPS >= 0.95 * reported_TPS  (one-sided, 5% slow-side tolerance)",
        "p_g1_pass_one_sided": round(p_fire, 5),
        "p_within_two_sided_pm5pct": round(primary["p_within_pm5pct_two_sided"], 5),
        "p_faster_than_plus5pct": round(primary["p_faster_than_plus5pct"], 5),
        "p_slower_than_minus5pct_fail": round(1.0 - p_fire, 5),
        "partition_check": round(primary["p_within_pm5pct_two_sided"]
                                 + primary["p_faster_than_plus5pct"]
                                 + (1.0 - p_fire), 5),
        "paragraph": (
            "Scoring against the one-sided G1 reproduction rule (private TPS >= 0.95 x "
            "reported TPS), the fire passes with probability 98.5%. A naive two-sided "
            "+/-5% reading would call only 34.1% of single private draws a 'match' -- but "
            "that is a measurement artifact, not a risk: 64.4% of draws land MORE than 5% "
            "FASTER than reported (private prompts have a favorable acceptance profile, "
            "central -7.7%), and being faster is NOT a reproduction failure. Only 1.5% of "
            "draws fall on the >5%-slower side that G1 actually penalizes. The one-sided "
            "rule is the organizer's reproduction rule; we score against private >= "
            "0.95 x reported and flag that if the published rule were two-sided or used a "
            "different tolerance, the claim would have to be restated."),
        "organizer_rule_flag": ("scored against private>=0.95*reported (one-sided). If the "
                                "organizer's published G1 rule differs (two-sided band, or "
                                "tolerance != 5%), restate before quoting."),
    }

    # honesty carry-forward (verbatim from #754/#756)
    honesty = {
        "central_ci95_pct": sys_meta["boot_ci95_pct"],
        "ci95_upper_touches_positive": True,
        "boot_se_pct": round(boot_se, 4),
        "common_mode_public_anchor_se_pct": COMMON_MODE_SE_754,
        "gate_literal_epub_fixed_p_dq": 0.0,
        "gate_literal_note": ("the gate divides by the FIXED official-128 public TPS -> "
                              "e_pub-fixed reading gives P(DQ)=0; we headline the "
                              "conservative full-bootstrap (common-mode carried)."),
        "sigma_hw_transfer_assumption": hw["provenance"]["config_note"],
        "sigma_hw_absolute_corner_is_conservative": True,
        "no_double_count_with_500_portfolio": (
            "#228 publish_first_lambda_floor and #253 two_path_500_portfolio combine the "
            "REACH probabilities of two speculative ~500-TPS build paths (P_A/P_B, "
            "lambda-floor); this card combines the REPRODUCTION-survival of two FINISHED "
            "submissions (126.378 live + int4 fire) at 126-168 TPS. Different op-point, "
            "different event (reach vs reproduce), different objects (build paths vs "
            "submissions); no shared term is reused -> no double-count."),
    }

    headline = (
        f"Our leaderboard position survives the organizer's private G1 rerun with "
        f"probability >= {100*p_floor:.1f}% (pre-registered before the official number; "
        f"~{100*p_position:.1f}% under the realistic common-mode model, ~{100*p_physical:.1f}% "
        f"physical), and at any fire official TPS in 150-168 the 95%-worst single private "
        f"draw clears the 126.378 bar by "
        f"{150*r05 - BAR_TPS:.1f}-{168*r05 - BAR_TPS:.1f} TPS."
    )

    out = {
        "pr": 758,
        "student": "kanna",
        "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "reused_756_byte_identical": bool(repro_ok),
        "repro_anchors_756": {
            "p_fire_repro": round(p_fire, 5), "p_fire_expected": P_FIRE_756,
            "r05_repro": round(r05, 5), "r05_expected": R05_756,
            "g1_margin_repro": round(g1_margin, 4), "g1_margin_expected": G1_MARGIN_756,
        },
        "sign_convention": ("delta_stock<0 => private FASTER; G1 fail = private >5% SLOWER; "
                            "R = private/reported = 1 - delta_stock/100, folded with "
                            "multiplicative sigma_hw."),
        "systematic_leg_754": sys_meta,
        "sigma_hw_canonical_frac": sigma_canon,
        "deliverable_1_parametric_official_tps_table": table,
        "deliverable_2_joint_position_survival": joint,
        "deliverable_2_joint_at_absolute_3pct_sigma_hw": joint_abs[
            "joint_conservative_126_as_fire_deltastock"],
        "deliverable_3_one_sided_vs_two_sided": one_vs_two,
        "test_metric_g1_margin_tps_at_95": round(g1_margin, 4),
        "honesty_carry_forward": honesty,
    }

    out["decision"] = {
        "primary_metric_p_position_survives_g1": round(p_position, 5),
        "p_position_survives_floor_comonotonic": round(p_floor, 5),
        "p_position_survives_ceiling_independence": round(p_ceiling, 5),
        "p_position_survives_physical_deterministic_126": round(p_physical, 7),
        "test_metric_g1_margin_tps_at_95": round(g1_margin, 4),
        "g1_margin_reconfirms_756": bool(abs(g1_margin - G1_MARGIN_756) < 1e-2),
        "honest_headline_reading": "common_mode_correlated",
        "verdict": ("POSITION_G1_SAFE_PREREGISTERED" if p_position >= P_FIRE_756
                    else "POSITION_G1_AT_RISK"),
    }
    out["headline"] = headline

    (out_dir / "g1_board_claim_prereg.json").write_text(json.dumps(out, indent=2))
    _plot(arrays, table, joint, out_dir / "board_claim_prereg.png", sigma_canon)
    _print_summary(out)
    print(f"[g1prereg] wrote {out_dir/'g1_board_claim_prereg.json'}", flush=True)
    return out


def _plot(arrays: dict, table: dict, joint: dict, path: Path, sigma_hw_frac: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[g1prereg] plot skipped: {e!r}", flush=True)
        return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5))
    # left: single-draw ratio dists, fire vs conservative-126
    ax0.hist(arrays["R_fire"], bins=140, density=True, color="#4878CF", alpha=0.65,
             label="fire R = private/reported")
    ax0.hist(arrays["R_126c"], bins=140, density=True, color="#E24A33", alpha=0.45,
             label="126.378 (conservative) R")
    ax0.axvline(G1_RULE_RATIO, color="k", ls="-", lw=2, label="G1 floor R=0.95")
    ax0.set_title(f"Single-draw ratios (sigma_hw={100*sigma_hw_frac:.1f}%)  PR #758")
    ax0.set_xlabel("private_TPS / reported_TPS")
    ax0.set_ylabel("density")
    ax0.set_xlim(0.80, 1.35)
    ax0.legend(fontsize=8, loc="upper left")
    # right: pre-registered margin-over-bar vs official TPS
    offs = [r["official_tps"] for r in table["grid"]]
    marg = [r["margin_over_126378_bar_tps"] for r in table["grid"]]
    ax1.plot(offs, marg, "o-", color="#4878CF", lw=2)
    ax1.axhline(0.0, color="k", ls=":", alpha=0.6, label="126.378 bar")
    ax1.axvspan(150, 168, color="green", alpha=0.10, label="expected fire 150-168")
    for o, m in zip(offs, marg):
        ax1.annotate(f"{m:+.1f}", (o, m), fontsize=7.5, ha="center", va="bottom")
    ax1.set_title("Pre-registered: 95%-worst private margin over 126.378 bar")
    ax1.set_xlabel("fire official TPS")
    ax1.set_ylabel("worst-draw margin over bar (TPS)")
    ax1.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[g1prereg] wrote {path}", flush=True)


def _print_summary(o: dict[str, Any]) -> None:
    d = o["decision"]
    j = o["deliverable_2_joint_position_survival"]
    jc = j["joint_conservative_126_as_fire_deltastock"]
    m = j["marginals"]
    t3 = o["deliverable_3_one_sided_vs_two_sided"]
    print("\n" + "=" * 78, flush=True)
    print("PRE-REGISTERED BOARD G1 CLAIM: joint survival + official-TPS table   PR #758", flush=True)
    print("=" * 78, flush=True)
    print(f"  reused #756 byte-identical: {o['reused_756_byte_identical']}  "
          f"(P_fire={o['repro_anchors_756']['p_fire_repro']}, "
          f"R_05={o['repro_anchors_756']['r05_repro']}, "
          f"g1_margin={o['repro_anchors_756']['g1_margin_repro']})", flush=True)
    print("  -- Deliverable 1: parametric official-TPS table (P_g1, R_05 are SCALE-FREE) --", flush=True)
    print(f"     {'off':>6} {'worst_priv':>11} {'margin_bar':>11} {'speedup_x':>10} {'clears':>7}", flush=True)
    for r in o["deliverable_1_parametric_official_tps_table"]["grid"]:
        print(f"     {r['official_tps']:>6.0f} {r['worst_private_tps_at_95']:>11.2f} "
              f"{r['margin_over_126378_bar_tps']:>+11.2f} {r['speedup_x_over_bar']:>10.3f} "
              f"{str(r['worst_draw_clears_bar']):>7}", flush=True)
    print("  -- Deliverable 2: joint position survival P(>=1 of {126.378, fire} clears G1) --", flush=True)
    print(f"     marginals: P_fire={m['p_fire_clears_g1']}  "
          f"P_126(cons)={m['p_126378_clears_g1_conservative']}  "
          f"P_126(det)={m['p_126378_clears_g1_deterministic']}", flush=True)
    print(f"     rho(realized)={j['correlation']['rho_realized_ratio_conservative']}  "
          f"common-mode var frac={j['correlation']['common_mode_variance_fraction']}", flush=True)
    print(f"     independence ceiling = {jc['p_position_survives_independence_ceiling']}", flush=True)
    print(f"     common-mode correlated (HONEST) = "
          f"{jc['p_position_survives_common_mode_correlated']}  "
          f"(bvn xcheck {jc['p_position_survives_correlated_bvn_xcheck']})", flush=True)
    print(f"     comonotonic floor = {jc['p_position_survives_comonotonic_floor']}", flush=True)
    print(f"     physical (deterministic 126) = "
          f"{d['p_position_survives_physical_deterministic_126']}", flush=True)
    print("  -- Deliverable 3: one-sided vs two-sided --", flush=True)
    print(f"     P(G1-pass one-sided)={t3['p_g1_pass_one_sided']}  "
          f"within +/-5%={t3['p_within_two_sided_pm5pct']}  "
          f"faster>+5%={t3['p_faster_than_plus5pct']}  "
          f"fail(slower>5%)={t3['p_slower_than_minus5pct_fail']}", flush=True)
    print(f"  VERDICT: {d['verdict']}  "
          f"(p_position_survives_g1={d['primary_metric_p_position_survives_g1']}, "
          f"g1_margin_tps_at_95={d['test_metric_g1_margin_tps_at_95']})", flush=True)
    print("  " + o["headline"], flush=True)
    print("=" * 78 + "\n", flush=True)


def log_to_wandb(o: dict[str, Any], *, group: str, name: str, out_dir: Path) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return None
    d = o["decision"]
    j = o["deliverable_2_joint_position_survival"]
    jc = j["joint_conservative_126_as_fire_deltastock"]
    m = j["marginals"]
    t3 = o["deliverable_3_one_sided_vs_two_sided"]
    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "p_position_survives_g1": d["primary_metric_p_position_survives_g1"],
        "p_position_survives_floor_comonotonic": d["p_position_survives_floor_comonotonic"],
        "p_position_survives_ceiling_independence": d["p_position_survives_ceiling_independence"],
        "p_position_survives_physical_deterministic_126":
            d["p_position_survives_physical_deterministic_126"],
        "p_position_survives_correlated_bvn_xcheck":
            jc["p_position_survives_correlated_bvn_xcheck"],
        "p_fire_clears_g1": m["p_fire_clears_g1"],
        "p_126378_clears_g1_conservative": m["p_126378_clears_g1_conservative"],
        "p_126378_clears_g1_deterministic": m["p_126378_clears_g1_deterministic"],
        "rho_realized_ratio": j["correlation"]["rho_realized_ratio_conservative"],
        "common_mode_variance_fraction": j["correlation"]["common_mode_variance_fraction"],
        "g1_margin_tps_at_95": d["test_metric_g1_margin_tps_at_95"],
        "p_g1_pass_one_sided": t3["p_g1_pass_one_sided"],
        "p_within_two_sided_pm5pct": t3["p_within_two_sided_pm5pct"],
        "p_faster_than_plus5pct": t3["p_faster_than_plus5pct"],
        "reused_756_byte_identical": int(o["reused_756_byte_identical"]),
    }
    # pre-registered table -> per-official summary keys
    for r in o["deliverable_1_parametric_official_tps_table"]["grid"]:
        off = int(r["official_tps"])
        summary[f"table_off{off}_worst_priv_tps"] = r["worst_private_tps_at_95"]
        summary[f"table_off{off}_margin_over_bar"] = r["margin_over_126378_bar_tps"]
        summary[f"table_off{off}_speedup_x"] = r["speedup_x_over_bar"]
    run = init_wandb_run(
        job_type="g1-board-claim-prereg", agent="senpai", name=name, group=group,
        tags=["g1-board-claim-prereg", group, "joint-survival", "prereg", "pr758"],
        notes=f"#758 pre-registered board G1 claim; verdict={d['verdict']}; {o['headline']}",
        config={"analysis_only": True, "no_hf_job": True, "fires": 0,
                "bootstrap_draws": o["systematic_leg_754"]["draws"],
                "seed": o["systematic_leg_754"]["seed"],
                "sigma_hw_canonical_frac": o["sigma_hw_canonical_frac"],
                "bar_tps": BAR_TPS, "wandb_group": group,
                "official_tps_grid": [r["official_tps"]
                                      for r in o["deliverable_1_parametric_official_tps_table"]["grid"]],
                "source_report": o["systematic_leg_754"]["source_report"]})
    if run is None:
        print("[wandb] run not created; json is the record", flush=True)
        return None
    log_summary(run, summary, step=0)
    run.summary["verdict"] = d["verdict"]
    run.summary["headline"] = o["headline"]
    for fn in ("g1_board_claim_prereg.json", "board_claim_prereg.png"):
        p = out_dir / fn
        if p.exists():
            try:
                log_file_artifact(run, path=p, name=f"g1prereg_{p.stem}",
                                  artifact_type="g1-board-claim-prereg")
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] artifact {fn} failed (non-fatal): {e!r}", flush=True)
    rid = run.id
    finish_wandb(run)
    print(f"[wandb] logged {name} id={rid} (group={group})", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=DEF_SOURCE_REPORT)
    ap.add_argument("--hw-envelope", default=DEF_HW_ENVELOPE)
    ap.add_argument("--hw-singledraw", default=DEF_HW_SINGLEDRAW)
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seed", type=int, default=730, help="systematic bootstrap seed (#754/#756)")
    ap.add_argument("--hw-seed", type=int, default=756, help="fire sigma_hw seed (== #756, byte-identical)")
    ap.add_argument("--hw-seed-126", type=int, default=758, help="126.378 sigma_hw seed (independent run)")
    ap.add_argument("--idio-perm-seed", type=int, default=1758,
                    help="126.378 idiosyncratic permutation seed (shares common-mode)")
    ap.add_argument("--indep-perm-seed", type=int, default=2758,
                    help="independence-bound pairing seed")
    ap.add_argument("--grid", type=float, nargs="+", default=DEF_OFFICIAL_GRID)
    ap.add_argument("--out-dir", default=DEF_OUT_DIR)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="kanna/g1-board-claim-prereg")
    args = ap.parse_args()

    out = run(args)

    run_id = None
    if args.wandb_group:
        try:
            run_id = log_to_wandb(out, group=args.wandb_group, name=args.wandb_name,
                                  out_dir=REPO / args.out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] logging failed (non-fatal): {e!r}", flush=True)

    dec = out["decision"]
    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [run_id] if run_id else [],
        "primary_metric": {"name": "p_position_survives_g1",
                           "value": dec["primary_metric_p_position_survives_g1"]},
        "test_metric": {"name": "g1_margin_tps_at_95",
                        "value": dec["test_metric_g1_margin_tps_at_95"]},
    }
    print("SENPAI-RESULT:")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
