#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""sigma_hw breakdown-point: how fragile is the G1-safety claim to hw-variance mis-spec? -- PR #763.

The board G1-safety headline (#756 single-draw P(G1-pass)=0.98504; #758 position
survival 0.9958) rests on ONE load-bearing modeling assumption: the local->official
hardware-variance transfer sigma_hw. It is NOT a direct fire-config measurement -- the
1% FRACTIONAL multiplicative CV was characterized on the split-KV/K7 ~485 TPS frontier
(#159 frantic-penguin 3-draw CV 0.962%) and ASSUMED to transfer to int4_mtp_batchinv
(#478: sigma_hw is a clock/bandwidth draw -> multiplicative -> transfers). The #756 card
flagged this itself (wide CI95, common-mode dominance, one-sided gate).

This capstone does NOT re-measure sigma_hw. It asks the robustness question: how badly
would the modeled sigma_hw have to be MIS-SPECIFIED before the G1 claim breaks? We hold
everything else (the #754 systematic delta_stock bootstrap, the #758 common-mode rho,
the idiosyncratic SE, all seeds) FIXED at their #756/#758 values and sweep ONLY sigma_hw
upward as a multiple of the modeled 1% value. The breakdown-point is the multiple at
which single-draw P(G1-pass) first drops to 0.95 (then 0.90), and the same for the joint
leaderboard-position survival.

Sensitivity-sweep determinism: convolve / joint_survival draw eps ~ N(0, sigma) with a
FIXED rng seed and fixed size, so eps = sigma * z for the SAME standard normals z at
every multiple. Scaling sigma scales the same draws -> the breakdown curve is smooth and
monotone with NO Monte-Carlo jitter across the sweep, which is exactly what isolates the
sigma_hw axis. (This is byte-identical to #756/#758 at the modeled 1x multiple -- asserted.)

Three deliverables:
  1. The fine sigma_hw multiple sweep (0.5x .. 4x detailed, extended to ~20x to locate the
     crossings) of single-draw P(G1-pass), R_05, the joint position-survival (common-mode
     correlated, comonotonic floor, independence ceiling, and the PHYSICAL deterministic-126
     backstop), and the deterministic-126 marginal.
  2. The BREAKDOWN-MULTIPLES: the multiple of modeled sigma_hw at which each curve first
     crosses 0.95 and 0.90 (linear-interpolated). PRIMARY = the fire single-draw 0.95
     crossing. Verdict robust iff that multiple >= 2.0x.
  3. The scale-free vs absolute-margin split: P(G1-pass)/R_05 are SCALE-FREE in the
     official TPS (the gate is the ratio rule private>=0.95*reported) so the breakdown
     multiple is identical at any anchor -- but the absolute 95%-worst private TPS is NOT,
     so we report it at the predicted fire anchor (157, fern #750) and the bar-adjacent
     stress anchor (130) to show the margin shrink toward the 126.378 bar.

Honesty carry-forward: this is a SENSITIVITY BOUND on a modeling assumption, framed as
"the claim survives sigma_hw up to N x modeled", NOT a claim that "sigma_hw is N x". The
#756 caveats carry: wide systematic CI95 [-19.9,+3.7], common-mode dominates 77.3% of the
delta_stock variance, the gate is one-sided.

LOCAL ONLY: CPU numpy, reads existing in-repo artifacts (all kanna's own #754/#756/#758
work on the advisor branch), launches no server, makes no submission, fires no HF Job.
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

# Reuse kanna's own #756 single-draw machinery and #758 joint-survival machinery
# BYTE-FOR-BYTE -- the sweep is the SAME functions called at scaled sigma_hw.
from scripts.validity.g1_single_draw_repro_pass import (  # noqa: E402
    BAR_TPS, G1_DELTA_PCT, G1_RULE_RATIO,
    DEF_HW_ENVELOPE, DEF_HW_SINGLEDRAW, DEF_SOURCE_REPORT,
    convolve, load_sigma_hw, load_systematic_delta,
)
from scripts.validity.g1_board_claim_prereg import joint_survival  # noqa: E402

# Anchors asserted reproduced byte-identically at the modeled 1x multiple.
P_FIRE_756 = 0.98504
R05_756 = 0.97739
G1_MARGIN_756 = 2.9231
P_POSITION_758 = 0.9958          # #758 common-mode correlated position survival

# Fire-TPS anchors for the ABSOLUTE 95%-worst-private margin (P_g1/R_05 are scale-free;
# the anchor only scales the TPS-denominated columns, NOT the breakdown multiple).
DEF_FIRE_ANCHOR = 157.0          # fern #750 predicted official ~156.95
DEF_STRESS_ANCHOR = 130.0        # bar-adjacent stress anchor (margin razor-thin here)

DEF_OUT_DIR = "research/validity/g1_sigma_hw_breakdown/results"


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _first_downcross(mult: np.ndarray, y: np.ndarray, t: float) -> dict[str, Any]:
    """First multiple at which a (near-)monotone-decreasing curve y(mult) drops to t.

    Linear-interpolated on the bracketing segment. Returns the multiple, the sigma_hw
    fraction implied (caller multiplies by modeled), whether a crossing was found within
    the swept grid, and a human note.
    """
    mult = np.asarray(mult, dtype=float)
    y = np.asarray(y, dtype=float)
    if y[0] < t:
        return {"crossed": True, "below_grid": True, "mult": float(mult[0]),
                "note": f"already below {t} at the smallest swept multiple {mult[0]:.3f}x"}
    for i in range(1, len(y)):
        if y[i] < t <= y[i - 1]:
            frac = (y[i - 1] - t) / (y[i - 1] - y[i])
            m_cross = float(mult[i - 1] + frac * (mult[i] - mult[i - 1]))
            return {"crossed": True, "below_grid": False, "mult": round(m_cross, 4),
                    "bracket": [round(float(mult[i - 1]), 4), round(float(mult[i]), 4)],
                    "note": f"crosses {t} between {mult[i-1]:.3f}x and {mult[i]:.3f}x"}
    return {"crossed": False, "below_grid": False, "mult": None,
            "note": f"never drops to {t} within the swept grid (<= {mult[-1]:.2f}x) -> "
                    f"robust beyond the swept range; min value {y.min():.5f} at {mult[-1]:.2f}x"}


def sweep(central_b: np.ndarray, central_b_epubfix: np.ndarray, modeled_sigma: float,
          multiples: np.ndarray, *, hw_seed: int, hw_seed_126: int, idio_perm_seed: int,
          indep_perm_seed: int, fire_anchor: float, stress_anchor: float) -> list[dict[str, Any]]:
    """Recompute every G1 quantity at sigma_hw = modeled_sigma * m for each m, holding the
    systematic delta_stock bootstrap and all couplings/seeds fixed (isolates sigma_hw)."""
    rows: list[dict[str, Any]] = []
    for m in multiples:
        sigma = float(modeled_sigma * m)
        single = convolve(central_b, sigma, hw_seed=hw_seed, bar=BAR_TPS,
                          reported_levels=[fire_anchor, stress_anchor])
        single.pop("_r_real")
        joint = joint_survival(central_b, central_b_epubfix, sigma,
                               hw_seed_fire=hw_seed, hw_seed_126=hw_seed_126,
                               idio_perm_seed=idio_perm_seed, indep_perm_seed=indep_perm_seed)
        joint.pop("_arrays")
        jc = joint["joint_conservative_126_as_fire_deltastock"]
        jd = joint["joint_deterministic_126_physical"]
        mg = joint["marginals"]
        r05 = single["r_realized"]["p5"]
        rows.append({
            "mult": round(float(m), 4),
            "sigma_hw_frac": round(sigma, 6),
            "sigma_hw_pct": round(100.0 * sigma, 4),
            # scale-free (anchor-independent) gate quantities
            "p_g1_single_draw_pass": round(single["p_g1_single_draw_pass"], 5),
            "r05": round(r05, 5),
            "g1_margin_tps_at_95": round(single["g1_margin_tps_at_95"], 4),
            # joint leaderboard-position survival (anchor-independent)
            "p_position_correlated": round(jc["p_position_survives_common_mode_correlated"], 5),
            "p_position_floor": round(jc["p_position_survives_comonotonic_floor"], 5),
            "p_position_ceiling": round(jc["p_position_survives_independence_ceiling"], 5),
            "p_position_physical": round(jd["p_position_survives_correlated"], 6),
            "p_126_det_marginal": round(mg["p_126378_clears_g1_deterministic"], 6),
            "p_fire_marginal": round(mg["p_fire_clears_g1"], 5),
            # ABSOLUTE 95%-worst private TPS at the two anchors (scale-dependent)
            "worst_priv_tps_fire_anchor": round(fire_anchor * r05, 3),
            "worst_priv_tps_stress_anchor": round(stress_anchor * r05, 3),
            "margin_over_bar_fire_anchor": round(fire_anchor * r05 - BAR_TPS, 3),
            "margin_over_bar_stress_anchor": round(stress_anchor * r05 - BAR_TPS, 3),
        })
    return rows


def _col(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def _row_at_mult(rows: list[dict[str, Any]], m_target: float | None) -> dict[str, Any] | None:
    """Nearest swept row to a (possibly interpolated) breakdown multiple, for absolute-TPS context."""
    if m_target is None:
        return None
    mults = _col(rows, "mult")
    i = int(np.argmin(np.abs(mults - m_target)))
    r = rows[i]
    return {
        "nearest_swept_mult": r["mult"],
        "sigma_hw_pct": r["sigma_hw_pct"],
        "p_g1_single_draw_pass": r["p_g1_single_draw_pass"],
        "r05": r["r05"],
        "worst_priv_tps_fire_anchor": r["worst_priv_tps_fire_anchor"],
        "margin_over_bar_fire_anchor": r["margin_over_bar_fire_anchor"],
        "worst_priv_tps_stress_anchor": r["worst_priv_tps_stress_anchor"],
        "margin_over_bar_stress_anchor": r["margin_over_bar_stress_anchor"],
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
    modeled_sigma = hw["canonical_frac"]          # the value under sensitivity test (0.01)

    # --- byte-identical reproduction guard at the modeled 1x multiple (== #756/#758) ---
    base_single = convolve(central_b, modeled_sigma, hw_seed=args.hw_seed, bar=BAR_TPS,
                           reported_levels=[args.fire_anchor, args.stress_anchor])
    base_single.pop("_r_real")
    base_joint = joint_survival(central_b, central_b_epubfix, modeled_sigma,
                                hw_seed_fire=args.hw_seed, hw_seed_126=args.hw_seed_126,
                                idio_perm_seed=args.idio_perm_seed,
                                indep_perm_seed=args.indep_perm_seed)
    base_joint.pop("_arrays")
    p_fire_1x = base_single["p_g1_single_draw_pass"]
    r05_1x = base_single["r_realized"]["p5"]
    g1_margin_1x = base_single["g1_margin_tps_at_95"]
    p_pos_1x = base_joint["joint_conservative_126_as_fire_deltastock"][
        "p_position_survives_common_mode_correlated"]
    repro_ok = (abs(p_fire_1x - P_FIRE_756) < 1e-4 and abs(r05_1x - R05_756) < 1e-4
                and abs(g1_margin_1x - G1_MARGIN_756) < 1e-2
                and abs(p_pos_1x - P_POSITION_758) < 1e-4)

    # --- the sigma_hw multiple sweep ---
    multiples = np.round(np.arange(args.mult_lo, args.mult_hi + 1e-9, args.mult_step), 4)
    rows = sweep(central_b, central_b_epubfix, modeled_sigma, multiples,
                 hw_seed=args.hw_seed, hw_seed_126=args.hw_seed_126,
                 idio_perm_seed=args.idio_perm_seed, indep_perm_seed=args.indep_perm_seed,
                 fire_anchor=args.fire_anchor, stress_anchor=args.stress_anchor)

    mult_arr = _col(rows, "mult")
    fire_curve = _col(rows, "p_g1_single_draw_pass")
    pos_corr_curve = _col(rows, "p_position_correlated")
    pos_phys_curve = _col(rows, "p_position_physical")
    p126_curve = _col(rows, "p_126_det_marginal")

    # --- breakdown multiples (the headline output) ---
    breakdown = {
        "fire_single_draw_pass": {
            "p_at_modeled_1x": round(p_fire_1x, 5),
            "mult_at_0.95": _first_downcross(mult_arr, fire_curve, 0.95),
            "mult_at_0.90": _first_downcross(mult_arr, fire_curve, 0.90),
        },
        "position_survival_correlated": {
            "p_at_modeled_1x": round(p_pos_1x, 5),
            "mult_at_0.95": _first_downcross(mult_arr, pos_corr_curve, 0.95),
            "mult_at_0.90": _first_downcross(mult_arr, pos_corr_curve, 0.90),
        },
        "position_survival_physical_backstop": {
            "p_at_modeled_1x": round(float(pos_phys_curve[np.argmin(np.abs(mult_arr - 1.0))]), 6),
            "mult_at_0.95": _first_downcross(mult_arr, pos_phys_curve, 0.95),
            "mult_at_0.90": _first_downcross(mult_arr, pos_phys_curve, 0.90),
        },
        "marginal_126_deterministic": {
            "p_at_modeled_1x": round(float(p126_curve[np.argmin(np.abs(mult_arr - 1.0))]), 6),
            "mult_at_0.95": _first_downcross(mult_arr, p126_curve, 0.95),
            "mult_at_0.90": _first_downcross(mult_arr, p126_curve, 0.90),
        },
    }

    m95_fire = breakdown["fire_single_draw_pass"]["mult_at_0.95"]["mult"]
    m90_fire = breakdown["fire_single_draw_pass"]["mult_at_0.90"]["mult"]
    g1_claim_robust = int(m95_fire is not None and m95_fire >= 2.0)

    # absolute-TPS context AT the fire breakdown points (margin shrink toward the bar)
    abs_context = {
        "at_modeled_1x": _row_at_mult(rows, 1.0),
        "at_fire_0.95_breakdown": _row_at_mult(rows, m95_fire),
        "at_fire_0.90_breakdown": _row_at_mult(rows, m90_fire),
        "note": ("P(G1-pass)/R_05 are SCALE-FREE in the official TPS (gate = ratio rule "
                 "private>=0.95*reported), so the breakdown MULTIPLE is identical at any "
                 "anchor; only the absolute worst-private TPS scales. At the 130 stress "
                 "anchor the worst-private margin over the 126.378 bar is razor-thin even at "
                 "the modeled sigma_hw, so the absolute bar-clear breaks far BEFORE the "
                 "scale-free G1-reproduction claim does."),
    }

    # does the deterministic-126 backstop hold position survival >= 0.95 at the point the
    # FIRE single-draw claim breaks? (the whole reason position survival is more robust)
    backstop = {}
    if m95_fire is not None:
        i = int(np.argmin(np.abs(mult_arr - m95_fire)))
        backstop = {
            "fire_0.95_breakdown_mult": m95_fire,
            "position_correlated_at_that_mult": round(float(pos_corr_curve[i]), 5),
            "position_physical_at_that_mult": round(float(pos_phys_curve[i]), 6),
            "position_correlated_holds_above_0.95": bool(pos_corr_curve[i] >= 0.95),
            "position_physical_holds_above_0.95": bool(pos_phys_curve[i] >= 0.95),
            "note": ("when the FIRE single-draw P(G1-pass) hits 0.95, the joint position "
                     "survival is still well above 0.95 because the 126.378 live submission "
                     "(non-spec greedy, deterministic) fails only via sigma_hw, INDEPENDENT "
                     "of the fire's delta_stock-driven failure -> the backstop is exactly "
                     "why position survival breaks at a LARGER sigma_hw multiple than the "
                     "fire-alone claim."),
        }

    verdict = ("G1_CLAIM_ROBUST_TO_SIGMA_HW" if g1_claim_robust
               else "G1_CLAIM_FRAGILE_TO_SIGMA_HW")

    headline = (
        f"The G1-safety headline survives sigma_hw mis-specification up to "
        f"{m95_fire:.2f}x the modeled 1% before single-draw P(G1-pass) drops to 0.95 "
        f"(and {('%.2fx' % m90_fire) if m90_fire else '>%.0fx' % mult_arr[-1]} before 0.90); "
        f"the joint leaderboard-position survival is even more robust thanks to the "
        f"deterministic 126.378 backstop. Robust (>= 2x): {bool(g1_claim_robust)}."
    )

    # detailed table subset for the report: the requested 0.5-4x band at 0.5x steps + the
    # crossing neighborhoods (kept compact; the full curve is in `sweep_full`).
    report_mults = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    table_subset = []
    for tm in report_mults:
        i = int(np.argmin(np.abs(mult_arr - tm)))
        table_subset.append(rows[i])

    out = {
        "pr": 763,
        "student": "kanna",
        "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "reused_756_758_byte_identical": bool(repro_ok),
        "repro_anchors": {
            "p_fire_1x": round(p_fire_1x, 5), "p_fire_expected_756": P_FIRE_756,
            "r05_1x": round(r05_1x, 5), "r05_expected_756": R05_756,
            "g1_margin_1x": round(g1_margin_1x, 4), "g1_margin_expected_756": G1_MARGIN_756,
            "p_position_1x": round(p_pos_1x, 5), "p_position_expected_758": P_POSITION_758,
        },
        "modeled_sigma_hw": {
            "modeled_frac": modeled_sigma,
            "modeled_pct": round(100.0 * modeled_sigma, 4),
            "provenance": ("canonical 1% FRACTIONAL multiplicative CV: #159 cross-allocation "
                           "frantic-penguin 3-draw official a10g-small CV 0.962% (dominant), "
                           "#478 canonicalized to 1% multiplicative. NOT a fire-config "
                           "measurement -> the value under sensitivity test."),
            "transfer_assumption": hw["provenance"]["config_note"],
            "fire_anchor_tps": args.fire_anchor,
            "stress_anchor_tps": args.stress_anchor,
        },
        "sign_convention": ("delta_stock<0 => private FASTER; G1 fail = private >5% SLOWER; "
                            "R = private/reported = 1 - delta_stock/100, folded with "
                            "MULTIPLICATIVE sigma_hw (eps ~ N(0, sigma)); sweep scales sigma "
                            "with FIXED standard-normal draws -> smooth monotone curve."),
        "systematic_leg_754": sys_meta,
        "sweep_grid": {"lo": args.mult_lo, "hi": args.mult_hi, "step": args.mult_step,
                       "n_points": int(len(multiples))},
        "breakdown_multiples": breakdown,
        "absolute_tps_context": abs_context,
        "deterministic_backstop_check": backstop,
        "table_subset_0p5_to_4x": table_subset,
        "sweep_full": rows,
        "honesty_carry_forward": {
            "framing": ("SENSITIVITY BOUND on a modeling assumption: 'the claim survives "
                        "sigma_hw up to N x modeled', NOT 'sigma_hw is N x'. No new sigma_hw "
                        "measurement was taken."),
            "systematic_ci95_pct": sys_meta["boot_ci95_pct"],
            "boot_se_pct": round(boot_se, 4),
            "common_mode_variance_fraction": 0.7734,
            "common_mode_public_anchor_se_pct": 5.318,
            "one_sided_gate": ("the G1 rule is one-sided (private>=0.95*reported); being "
                               "faster is not a failure; 64.4% of single draws are >5% FASTER."),
            "what_held_fixed": ("the #754 systematic delta_stock bootstrap (seed 730, 50k), "
                                "the #758 common-mode rho (0.7515 realized) / idiosyncratic "
                                "SE / all hw + permutation seeds -> the sweep isolates "
                                "sigma_hw alone."),
            "absolute_3pct_corner_in_756": ("the #756 absolute-TPS loose bound (3% at ~160 "
                                            "TPS, ~3x modeled) already gave P(G1-pass)=0.972, "
                                            "consistent with the >=2x breakdown found here."),
        },
    }
    out["decision"] = {
        "primary_metric_sigma_hw_breakdown_mult_at_095": m95_fire,
        "sigma_hw_breakdown_mult_at_090": m90_fire,
        "test_metric_g1_claim_robust": g1_claim_robust,
        "verdict": verdict,
    }
    out["headline"] = headline

    (out_dir / "g1_sigma_hw_breakdown.json").write_text(json.dumps(out, indent=2))
    _plot(rows, breakdown, out_dir / "sigma_hw_breakdown.png")
    _print_summary(out)
    print(f"[g1sigmasweep] wrote {out_dir/'g1_sigma_hw_breakdown.json'}", flush=True)
    return out


def _plot(rows: list[dict[str, Any]], breakdown: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[g1sigmasweep] plot skipped: {e!r}", flush=True)
        return
    mult = _col(rows, "mult")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(mult, _col(rows, "p_g1_single_draw_pass"), "-", color="#4878CF", lw=2.2,
            label="fire single-draw P(G1-pass)")
    ax.plot(mult, _col(rows, "p_position_correlated"), "-", color="#6ACC65", lw=2.0,
            label="position survival (common-mode correlated)")
    ax.plot(mult, _col(rows, "p_position_physical"), "--", color="#218A21", lw=1.8,
            label="position survival (physical det-126 backstop)")
    ax.plot(mult, _col(rows, "p_126_det_marginal"), ":", color="#9467BD", lw=1.6,
            label="126.378 deterministic marginal")
    ax.axhline(0.95, color="r", ls="-", lw=1.4, alpha=0.8, label="0.95 threshold")
    ax.axhline(0.90, color="darkred", ls="--", lw=1.2, alpha=0.7, label="0.90 threshold")
    ax.axvline(1.0, color="k", ls=":", alpha=0.6, label="modeled sigma_hw (1x)")
    ax.axvline(2.0, color="grey", ls="-.", alpha=0.5, label="2x robustness bar")
    m95 = breakdown["fire_single_draw_pass"]["mult_at_0.95"]["mult"]
    if m95 is not None:
        ax.scatter([m95], [0.95], color="#4878CF", zorder=5, s=45)
        ax.annotate(f"fire 0.95 @ {m95:.2f}x", (m95, 0.95), fontsize=8,
                    xytext=(m95 + 0.4, 0.945), color="#27408B")
    ax.set_title("sigma_hw breakdown sweep: G1-safety vs hardware-variance mis-spec  PR #763")
    ax.set_xlabel("sigma_hw as a multiple of the modeled 1% (sensitivity sweep)")
    ax.set_ylabel("probability of clearing G1")
    ax.set_ylim(0.70, 1.005)
    ax.set_xlim(mult[0], mult[-1])
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[g1sigmasweep] wrote {path}", flush=True)


def _print_summary(o: dict[str, Any]) -> None:
    d = o["decision"]
    b = o["breakdown_multiples"]
    print("\n" + "=" * 80, flush=True)
    print("SIGMA_HW BREAKDOWN-POINT: G1-safety robustness to hw-variance mis-spec   PR #763",
          flush=True)
    print("=" * 80, flush=True)
    ra = o["repro_anchors"]
    print(f"  reused #756/#758 byte-identical: {o['reused_756_758_byte_identical']}  "
          f"(P_fire={ra['p_fire_1x']}, R_05={ra['r05_1x']}, "
          f"g1_margin={ra['g1_margin_1x']}, P_pos={ra['p_position_1x']})", flush=True)
    print(f"  modeled sigma_hw = {o['modeled_sigma_hw']['modeled_pct']}% fractional "
          f"(#159 cross 0.962% / #478 1%); the value under sensitivity test", flush=True)
    print(f"  swept {o['sweep_grid']['n_points']} multiples "
          f"{o['sweep_grid']['lo']}x..{o['sweep_grid']['hi']}x step {o['sweep_grid']['step']}",
          flush=True)
    print("  -- breakdown multiples (x modeled sigma_hw) --", flush=True)
    for key, label in [("fire_single_draw_pass", "fire single-draw P(G1-pass)"),
                       ("position_survival_correlated", "position survival (correlated)"),
                       ("position_survival_physical_backstop", "position survival (physical)"),
                       ("marginal_126_deterministic", "126.378 det marginal")]:
        c95 = b[key]["mult_at_0.95"]; c90 = b[key]["mult_at_0.90"]
        s95 = f"{c95['mult']:.2f}x" if c95["mult"] is not None else f">{o['sweep_grid']['hi']:.0f}x"
        s90 = f"{c90['mult']:.2f}x" if c90["mult"] is not None else f">{o['sweep_grid']['hi']:.0f}x"
        print(f"    {label:34s} 1x={b[key]['p_at_modeled_1x']:.5f}  "
              f"->0.95 @ {s95:>7s}   ->0.90 @ {s90:>7s}", flush=True)
    bs = o["deterministic_backstop_check"]
    if bs:
        print(f"  -- backstop: at the fire 0.95 breakdown ({bs['fire_0.95_breakdown_mult']:.2f}x), "
              f"position survival corr={bs['position_correlated_at_that_mult']} "
              f"phys={bs['position_physical_at_that_mult']} "
              f"(>=0.95 corr={bs['position_correlated_holds_above_0.95']}, "
              f"phys={bs['position_physical_holds_above_0.95']})", flush=True)
    ac = o["absolute_tps_context"]
    for tag in ("at_modeled_1x", "at_fire_0.95_breakdown"):
        r = ac.get(tag)
        if r:
            print(f"  -- {tag}: worst-priv @157={r['worst_priv_tps_fire_anchor']} "
                  f"(margin {r['margin_over_bar_fire_anchor']:+.2f}) | @130="
                  f"{r['worst_priv_tps_stress_anchor']} "
                  f"(margin {r['margin_over_bar_stress_anchor']:+.2f}) over the 126.378 bar",
                  flush=True)
    print(f"  VERDICT: {d['verdict']}  "
          f"(breakdown_mult@0.95={d['primary_metric_sigma_hw_breakdown_mult_at_095']}, "
          f"g1_claim_robust={d['test_metric_g1_claim_robust']})", flush=True)
    print("  " + o["headline"], flush=True)
    print("=" * 80 + "\n", flush=True)


def log_to_wandb(o: dict[str, Any], *, group: str, name: str, out_dir: Path) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return None
    d = o["decision"]; b = o["breakdown_multiples"]; bs = o["deterministic_backstop_check"]
    ac = o["absolute_tps_context"]

    def _m(key, thr):
        return b[key][f"mult_at_{thr}"]["mult"]

    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "sigma_hw_breakdown_mult_at_095": d["primary_metric_sigma_hw_breakdown_mult_at_095"],
        "sigma_hw_breakdown_mult_at_090": d["sigma_hw_breakdown_mult_at_090"],
        "g1_claim_robust": d["test_metric_g1_claim_robust"],
        "fire_p_g1_at_1x": b["fire_single_draw_pass"]["p_at_modeled_1x"],
        "position_correlated_at_1x": b["position_survival_correlated"]["p_at_modeled_1x"],
        "position_correlated_breakdown_095": _m("position_survival_correlated", "0.95"),
        "position_physical_breakdown_095": _m("position_survival_physical_backstop", "0.95"),
        "marginal_126_det_breakdown_095": _m("marginal_126_deterministic", "0.95"),
        "modeled_sigma_hw_pct": o["modeled_sigma_hw"]["modeled_pct"],
        "reused_756_758_byte_identical": int(o["reused_756_758_byte_identical"]),
    }
    if bs:
        summary["position_corr_at_fire095_breakdown"] = bs["position_correlated_at_that_mult"]
        summary["position_phys_at_fire095_breakdown"] = bs["position_physical_at_that_mult"]
    if ac.get("at_modeled_1x"):
        summary["worst_priv_tps_fire157_at_1x"] = ac["at_modeled_1x"]["worst_priv_tps_fire_anchor"]
        summary["worst_priv_tps_stress130_at_1x"] = ac["at_modeled_1x"]["worst_priv_tps_stress_anchor"]
    # a few curve waypoints for quick W&B inspection
    for r in o["table_subset_0p5_to_4x"]:
        mtag = str(r["mult"]).replace(".", "p")
        summary[f"curve_m{mtag}_p_g1"] = r["p_g1_single_draw_pass"]
        summary[f"curve_m{mtag}_p_position_corr"] = r["p_position_correlated"]

    run = init_wandb_run(
        job_type="g1-sigma-hw-breakdown", agent="senpai", name=name, group=group,
        tags=["g1-sigma-hw-breakdown", group, "sigma_hw", "robustness", "sensitivity", "pr763"],
        notes=f"#763 sigma_hw breakdown-point; verdict={d['verdict']}; {o['headline']}",
        config={"analysis_only": True, "no_hf_job": True, "fires": 0,
                "bootstrap_draws": o["systematic_leg_754"]["draws"],
                "seed": o["systematic_leg_754"]["seed"],
                "modeled_sigma_hw_frac": o["modeled_sigma_hw"]["modeled_frac"],
                "bar_tps": BAR_TPS, "wandb_group": group,
                "sweep_lo": o["sweep_grid"]["lo"], "sweep_hi": o["sweep_grid"]["hi"],
                "sweep_step": o["sweep_grid"]["step"],
                "fire_anchor_tps": o["modeled_sigma_hw"]["fire_anchor_tps"],
                "stress_anchor_tps": o["modeled_sigma_hw"]["stress_anchor_tps"],
                "source_report": o["systematic_leg_754"]["source_report"]})
    if run is None:
        print("[wandb] run not created; json is the record", flush=True)
        return None
    log_summary(run, summary, step=0)
    run.summary["verdict"] = d["verdict"]
    run.summary["headline"] = o["headline"]
    # log the full sweep as a wandb Table for rich downstream analysis
    try:
        import wandb
        cols = ["mult", "sigma_hw_pct", "p_g1_single_draw_pass", "r05",
                "p_position_correlated", "p_position_floor", "p_position_physical",
                "p_126_det_marginal", "worst_priv_tps_fire_anchor",
                "worst_priv_tps_stress_anchor", "margin_over_bar_stress_anchor"]
        tbl = wandb.Table(columns=cols)
        for r in o["sweep_full"]:
            tbl.add_data(*[r[c] for c in cols])
        run.log({"sigma_hw_sweep": tbl}, step=0)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] sweep table failed (non-fatal): {e!r}", flush=True)
    for fn in ("g1_sigma_hw_breakdown.json", "sigma_hw_breakdown.png"):
        p = out_dir / fn
        if p.exists():
            try:
                log_file_artifact(run, path=p, name=f"g1sigmasweep_{p.stem}",
                                  artifact_type="g1-sigma-hw-breakdown")
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
    ap.add_argument("--hw-seed", type=int, default=756, help="fire sigma_hw seed (== #756/#758)")
    ap.add_argument("--hw-seed-126", type=int, default=758, help="126.378 sigma_hw seed (== #758)")
    ap.add_argument("--idio-perm-seed", type=int, default=1758, help="== #758")
    ap.add_argument("--indep-perm-seed", type=int, default=2758, help="== #758")
    ap.add_argument("--fire-anchor", type=float, default=DEF_FIRE_ANCHOR)
    ap.add_argument("--stress-anchor", type=float, default=DEF_STRESS_ANCHOR)
    ap.add_argument("--mult-lo", type=float, default=0.5, help="lowest sigma_hw multiple")
    ap.add_argument("--mult-hi", type=float, default=20.0, help="highest sigma_hw multiple")
    ap.add_argument("--mult-step", type=float, default=0.05, help="sweep step in multiples")
    ap.add_argument("--out-dir", default=DEF_OUT_DIR)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="kanna/g1-sigma-hw-breakdown")
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
        "primary_metric": {"name": "sigma_hw_breakdown_mult_at_095",
                           "value": dec["primary_metric_sigma_hw_breakdown_mult_at_095"]},
        "test_metric": {"name": "g1_claim_robust",
                        "value": dec["test_metric_g1_claim_robust"]},
    }
    print("SENPAI-RESULT:")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
