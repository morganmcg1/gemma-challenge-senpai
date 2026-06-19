#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Single-draw P(G1-pass) for the fire: acceptance-gap (delta_stock) x sigma_hw -- PR #756.

The well-posed #754 card answered P(DQ)=P(delta_stock > +5%) = 0.0137 -- the chance
the SYSTEMATIC public->private acceptance gap alone makes the private run >5% slower.
But the literal G1 gate is a SINGLE private reproduction draw, whose realized TPS
carries TWO downside sources:

  (1) SYSTEMATIC -- the #754 delta_stock distribution (private prompts have a
      different acceptance profile; central -7.73%, favorable; only slow corner is
      free-response math at +19.7%).
  (2) ALEATORIC -- single-draw run-to-run hardware/scheduling jitter sigma_hw (which
      physical A10G the official scorer lands you on; clock/bandwidth/contention),
      the variance characterized in kanna #159 (hw_variance_envelope) and
      canonicalized in kanna #478 (single_draw_risk) as a ~1% FRACTIONAL,
      MULTIPLICATIVE per-draw CV.

This card folds (2) onto (1) to produce the literal gate quantity:

    p_g1_single_draw_pass = P( private_TPS >= 0.95 * reported_TPS )            [PRIMARY]

and, inverted, the TPS headroom the fire's official number must clear to pass G1 at
95% confidence:

    g1_margin_tps_at_95   = 126.378 * (1 / R_05 - 1)                            [TEST]

where R = private_TPS / reported_TPS and R_05 is its single-draw 5th percentile.

Sign convention (carried from #754): delta_stock < 0 => private FASTER;
P(DQ) = P(delta_stock > +5%) = private >5% SLOWER = the reproduction-FAIL side.
TPS ratio mapping (verified in the #749 report: delta_stock_pct_by_tps ~=
delta_stock_pct_by_eaccept to <1pt): R_systematic = 1 - delta_stock/100.

Arm A -- reuse the EXACT #754 well-posed bootstrap (seed 730, 50k block-bootstrap
  draws over the #749 faithful serving run's ~10s spec-log windows) to get the
  SYSTEMATIC R distribution. No new HF/serving job.
Arm B -- convolve with sigma_hw MULTIPLICATIVELY: R_realized = R_sys * (1 + eps),
  eps ~ N(0, sigma_hw_frac). Report p_g1_single_draw_pass, the two-sided
  P(within +/-5%), and the move vs #754's systematic-only 1-0.0137=0.9863.
Arm C -- invert: R_05 -> g1_margin_tps_at_95 over the 126.378 bar, plus the
  scale-free reproduction margin and per-reported-level reproduction headroom.

LOCAL ONLY: reads existing in-repo artifacts (all kanna's own work, advisor branch),
launches no server, makes no submission, fires no HF Job.
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

# Reuse the EXACT #754 bootstrap machinery so the systematic leg is byte-identical.
from scripts.validity.deltastock_wellposed import (  # noqa: E402
    SUBSET_ORDER, W_GPQA_F, W_MATH_F, W_MMLU_F,
    bin_intervals, boot_eaccept, parse_intervals, subset_accept_boundaries,
)

# --- fixed challenge constants ---
BAR_TPS = 126.378           # int4_g128_lmhead official bar (PPL 2.019, W&B 905tbujn)
G1_RULE_RATIO = 0.95        # private_TPS >= 0.95 * reported_TPS (the 5% reproduction rule)
G1_DELTA_PCT = 5.0          # equivalently delta_stock <= +5% on the slow side
SYSTEMATIC_ONLY_754_PASS = 1.0 - 0.0137  # #754 bootstrap P(DQ)=0.0137 -> 0.9863

# Default source artifacts (all in-repo, all kanna's own work on the advisor branch).
DEF_SOURCE_REPORT = ("research/validity/deltastock_measure/"
                     "749faithful-20260619T160229Z/report.json")
DEF_HW_ENVELOPE = "research/validity/hw_variance_envelope/envelope.json"          # #159
DEF_HW_SINGLEDRAW = ("research/equivalence_escalation/single_draw_risk_474/"
                     "single_draw_risk_474_results.json")                          # #478
DEF_OUT_DIR = "research/validity/g1_single_draw_repro_pass/results"


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def load_systematic_delta(report_path: Path, *, B: int, K: int, seed: int):
    """Regenerate the #754 well-posed bootstrap systematic delta_stock draws (in %).

    Returns (central_b, central_b_epubfix, point_central, meta). central_b is the
    conservative reading (public anchor jointly bootstrapped -> common-mode noise
    carried); central_b_epubfix is the gate-literal reading (official-128 public TPS
    held FIXED, since the gate divides by it).
    """
    report = json.loads(report_path.read_text())
    log_text = (report_path.parent / "server.log").read_text()
    acc, drf = parse_intervals(log_text)
    boundaries = subset_accept_boundaries(report)
    bins = bin_intervals(acc, drf, boundaries)

    rng = np.random.default_rng(seed)
    e_b = {n: boot_eaccept(bins[n]["a"], bins[n]["d"], B, K, rng) for n in SUBSET_ORDER}

    # conservative (public jointly bootstrapped)
    e_pub_b = e_b["public"]
    d_mmlu = (1.0 - e_b["knowledge_mmlupro"] / e_pub_b) * 100.0
    d_gpqa = (1.0 - e_b["gpqa_diamond"] / e_pub_b) * 100.0
    d_math = (1.0 - e_b["reasoning_math"] / e_pub_b) * 100.0
    central_b = W_MMLU_F * d_mmlu + W_GPQA_F * d_gpqa + W_MATH_F * d_math

    # gate-literal (public e_accept fixed at the point estimate)
    e_pub_point = report["e_accept_public"]
    d_mmlu_fx = (1.0 - e_b["knowledge_mmlupro"] / e_pub_point) * 100.0
    d_gpqa_fx = (1.0 - e_b["gpqa_diamond"] / e_pub_point) * 100.0
    d_math_fx = (1.0 - e_b["reasoning_math"] / e_pub_point) * 100.0
    central_b_epubfix = W_MMLU_F * d_mmlu_fx + W_GPQA_F * d_gpqa_fx + W_MATH_F * d_math_fx

    p_mmlu = report["delta_stock_pct_by_eaccept"]["knowledge_mmlupro"]
    p_gpqa = report["delta_stock_pct_by_eaccept"]["gpqa_diamond"]
    p_math = report["delta_stock_pct_by_eaccept"]["reasoning_math"]
    point_central = W_MMLU_F * p_mmlu + W_GPQA_F * p_gpqa + W_MATH_F * p_math

    meta = {
        "source_report": str(report_path),
        "draws": B, "K": K, "seed": seed,
        "n_windows_per_subset": {n: int(len(bins[n]["a"])) for n in SUBSET_ORDER},
        "point_central_pct": round(point_central, 4),
        "boot_mean_pct": round(float(central_b.mean()), 4),
        "boot_se_pct": round(float(central_b.std(ddof=1)), 4),
        "boot_ci95_pct": [round(float(np.percentile(central_b, 2.5)), 3),
                          round(float(np.percentile(central_b, 97.5)), 3)],
        "p_dq_repro_check": round(float(np.mean(central_b > G1_DELTA_PCT)), 5),
        "epubfix_se_pct": round(float(central_b_epubfix.std(ddof=1)), 4),
        "delta_stock_pct_by_tps_corners": report["delta_stock_pct_by_tps"],
    }
    return central_b, central_b_epubfix, float(point_central), meta


def load_sigma_hw(env_path: Path, single_path: Path) -> dict[str, Any]:
    """Reload kanna's own sigma_hw (single-draw hardware-allocation CV) from #159+#478.

    Canonical = 0.01 FRACTIONAL (the #478 multiplicative convention). Cross-allocation
    dominated by frantic-penguin's same-submission official 3-draw (#159). Absolute
    upper bound = 4.8153 TPS (the #478 conservative bound), re-expressed as a fraction
    of a ~160 TPS fire op-point.
    """
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    sd = json.loads(single_path.read_text()) if single_path.exists() else {}

    sigma_hw_pct_159 = (env.get("envelope") or {}).get("sigma_hw_pct")         # 0.9624
    cross = ((env.get("envelope") or {}).get("detail") or {}).get("sigma_cross") or {}
    fp = cross.get("frantic_penguin") or {}
    sig_inputs = sd.get("sigma_hw_inputs") or {}

    canonical = sd.get("sigma_frac_canonical", 0.01)                            # 0.01
    cross_frac = sig_inputs.get("sigma_between_frac",
                                (sigma_hw_pct_159 / 100.0) if sigma_hw_pct_159 else 0.009623)
    oneshot_frac = sig_inputs.get("sigma_oneshot_frac_measured", 0.010128)
    abs_tps = sd.get("sigma_abs_canonical", 4.8153)
    # absolute bound re-expressed at a ~160 TPS private fire op-point (loose upper)
    fire_oppoint_tps = 160.0
    abs_frac_at_fire = abs_tps / fire_oppoint_tps

    return {
        "canonical_frac": float(canonical),
        "cross_frac_frantic_penguin": float(cross_frac),
        "oneshot_meas_frac": float(oneshot_frac),
        "within_frac_159": float(sig_inputs.get("sigma_within_frac", 0.000726)),
        "absolute_tps_bound": float(abs_tps),
        "absolute_frac_at_160tps": float(abs_frac_at_fire),
        "sigma_hw_pct_159": sigma_hw_pct_159,
        "frantic_penguin_draws": fp.get("draws"),
        "provenance": {
            "same_hardware_a10g_sm86": True,
            "within_basis": "measured n=12 fresh-server local A10G sm_86 (#159), ~0.011%",
            "cross_basis": ("frantic-penguin 3 same-submission OFFICIAL a10g-small draws "
                            "489.63/483.80/480.41, CV 0.962% (#159) -- the dominant term"),
            "same_config_as_fire": False,
            "config_note": ("cross measured on the split-KV/K7 ~485 TPS frontier, NOT "
                            "int4_mtp_batchinv; #478 establishes sigma_hw is a "
                            "MULTIPLICATIVE clock/bandwidth draw -> the ~1% FRACTIONAL "
                            "model transfers across configs (a model assumption, not a "
                            "direct fire-config measurement). Absolute-TPS (3% at "
                            "~160 TPS) is the conservative loose bound."),
            "common_mode_with_delta_stock": False,
            "common_mode_note": ("sigma_hw is hardware-allocation scatter; the #754 "
                                 "delta_stock common-mode noise is public-anchor PROMPT "
                                 "sampling -> independent sources -> quadrature, no "
                                 "double-count."),
        },
    }


def convolve(central_b: np.ndarray, sigma_hw_frac: float, *, hw_seed: int,
             bar: float, reported_levels: list[float]) -> dict[str, Any]:
    """Fold sigma_hw onto a systematic delta_stock draw set; compute the single-draw
    realized ratio R = private_TPS/reported_TPS distribution and all G1 quantities."""
    r_sys = 1.0 - central_b / 100.0
    rng = np.random.default_rng(hw_seed)
    eps = rng.normal(0.0, sigma_hw_frac, size=r_sys.shape)
    r_real = r_sys * (1.0 + eps)

    p_pass = float(np.mean(r_real >= G1_RULE_RATIO))
    p_within = float(np.mean((r_real >= 0.95) & (r_real <= 1.05)))
    p_faster_than_5 = float(np.mean(r_real > 1.05))
    sys_only_pass = float(np.mean(r_sys >= G1_RULE_RATIO))

    r05 = float(np.percentile(r_real, 5.0))
    r01 = float(np.percentile(r_real, 1.0))
    r50 = float(np.percentile(r_real, 50.0))
    # Arm C margin: reported number must beat the bar by this so the 95%-worst single
    # private draw still clears the bar:  reported* = bar / R_05.
    g1_margin_tps_at_95 = bar * (1.0 / r05 - 1.0)
    required_reported_at_95 = bar / r05

    # reproduction-rule TPS headroom at candidate reported levels (scale-free in ratio)
    repro_headroom = []
    for lvl in reported_levels:
        worst_priv = lvl * r05
        repro_headroom.append({
            "reported_tps": lvl,
            "worst_case_private_tps_at_95": round(worst_priv, 3),
            "repro_headroom_tps_at_95": round(worst_priv - G1_RULE_RATIO * lvl, 3),
            "clears_bar_at_95": bool(worst_priv >= bar),
        })

    return {
        "sigma_hw_frac": sigma_hw_frac,
        "p_g1_single_draw_pass": p_pass,
        "p_within_pm5pct_two_sided": p_within,
        "p_faster_than_plus5pct": p_faster_than_5,
        "systematic_only_pass": sys_only_pass,
        "delta_from_754_systematic_only": p_pass - sys_only_pass,
        "r_realized": {
            "mean": round(float(r_real.mean()), 5),
            "p1": round(r01, 5), "p5": round(r05, 5), "p50": round(r50, 5),
            "std": round(float(r_real.std(ddof=1)), 5),
        },
        "repro_margin_pp_at_95": round((r05 - G1_RULE_RATIO) * 100.0, 4),
        "g1_margin_tps_at_95": round(g1_margin_tps_at_95, 4),
        "required_reported_tps_at_95": round(required_reported_at_95, 4),
        "repro_headroom_by_reported": repro_headroom,
        "_r_real": r_real,  # popped before JSON dump; kept for plotting
    }


def sym_normal_crosscheck(point_central: float, boot_se: float,
                          sigma_hw_frac: float) -> dict[str, Any]:
    """Analytic Normal cross-check of p_g1_single_draw_pass.

    delta_realized ~= delta_sys + hw, with the hw term mapped into delta-% space at the
    central ratio R0 = 1 - point_central/100:  sigma_hw_delta = 100 * R0 * sigma_hw_frac.
    P(fail) = P(delta_realized > 5); pass = 1 - P(fail).
    """
    r0 = 1.0 - point_central / 100.0
    sigma_hw_delta = 100.0 * r0 * sigma_hw_frac
    sigma_tot = sqrt(boot_se ** 2 + sigma_hw_delta ** 2)
    z = (G1_DELTA_PCT - point_central) / sigma_tot
    p_fail = 1.0 - _phi(z)
    return {
        "point_central_pct": round(point_central, 4),
        "boot_se_pct": round(boot_se, 4),
        "sigma_hw_delta_pp": round(sigma_hw_delta, 4),
        "sigma_total_pp": round(sigma_tot, 4),
        "p_g1_pass_symnormal": round(1.0 - p_fail, 5),
        "p_dq_symnormal": round(p_fail, 5),
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
    epubfix_se = float(central_b_epubfix.std(ddof=1))
    hw = load_sigma_hw(env_path, single_path)

    reported_levels = [BAR_TPS, 150.0, 157.0, 167.8]  # bar + plausible int4 fire levels

    # PRIMARY convolution at the canonical sigma_hw
    sigma_canon = hw["canonical_frac"]
    primary = convolve(central_b, sigma_canon, hw_seed=args.hw_seed,
                       bar=BAR_TPS, reported_levels=reported_levels)
    r_real_primary = primary.pop("_r_real")

    # gate-literal (e_pub-fixed) systematic leg, same hw fold -> a higher (less
    # conservative) reading since the common-mode public-anchor noise is removed
    gate_literal = convolve(central_b_epubfix, sigma_canon, hw_seed=args.hw_seed,
                            bar=BAR_TPS, reported_levels=reported_levels)
    gate_literal.pop("_r_real")

    # sigma_hw robustness bracket (low cross CV .. canonical .. one-shot .. absolute)
    bracket = {
        "cross_cv_159_0p962pct": hw["cross_frac_frantic_penguin"],
        "canonical_1pct_478": hw["canonical_frac"],
        "oneshot_meas_478": hw["oneshot_meas_frac"],
        "absolute_bound_3pct_at_160tps": hw["absolute_frac_at_160tps"],
    }
    sweep = {}
    for name, s in bracket.items():
        c = convolve(central_b, s, hw_seed=args.hw_seed, bar=BAR_TPS,
                     reported_levels=reported_levels)
        c.pop("_r_real")
        sweep[name] = {
            "sigma_hw_frac": s,
            "p_g1_single_draw_pass": c["p_g1_single_draw_pass"],
            "p_within_pm5pct_two_sided": c["p_within_pm5pct_two_sided"],
            "r05": c["r_realized"]["p5"],
            "g1_margin_tps_at_95": c["g1_margin_tps_at_95"],
        }

    symnorm = sym_normal_crosscheck(point_central, boot_se, sigma_canon)

    out = {
        "pr": 756,
        "student": "kanna",
        "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "sign_convention": ("delta_stock<0 => private FASTER; P(DQ)=P(delta_stock>+5%)="
                            "private >5% SLOWER = reproduction-FAIL side; "
                            "R_systematic = 1 - delta_stock/100."),
        "systematic_leg_754": sys_meta,
        "sigma_hw": hw,
        "arm_a_systematic_ratio": {
            "r_systematic_mean": round(float((1.0 - central_b / 100.0).mean()), 5),
            "r_systematic_p5": round(float(np.percentile(1.0 - central_b / 100.0, 5)), 5),
            "r_systematic_p50": round(float(np.percentile(1.0 - central_b / 100.0, 50)), 5),
            "p_dq_systematic_only": round(float(np.mean(central_b > G1_DELTA_PCT)), 5),
            "systematic_only_pass_0p9863": round(SYSTEMATIC_ONLY_754_PASS, 5),
        },
        "arm_b_single_draw_pass": {
            "p_g1_single_draw_pass": round(primary["p_g1_single_draw_pass"], 5),
            "p_within_pm5pct_two_sided": round(primary["p_within_pm5pct_two_sided"], 5),
            "p_faster_than_plus5pct": round(primary["p_faster_than_plus5pct"], 5),
            "systematic_only_pass": round(primary["systematic_only_pass"], 5),
            "sigma_hw_moves_pass_by": round(primary["delta_from_754_systematic_only"], 5),
            "r_realized": primary["r_realized"],
            "gate_literal_epubfixed": {
                "p_g1_single_draw_pass": round(gate_literal["p_g1_single_draw_pass"], 5),
                "p_within_pm5pct_two_sided": round(gate_literal["p_within_pm5pct_two_sided"], 5),
                "note": ("public anchor held fixed (the gate divides by the fixed "
                         "official-128 public TPS); removes common-mode noise -> upper "
                         "reading on p_g1."),
            },
            "sym_normal_crosscheck": symnorm,
        },
        "arm_c_invert_margin": {
            "r05_single_draw": primary["r_realized"]["p5"],
            "repro_margin_pp_at_95": primary["repro_margin_pp_at_95"],
            "g1_margin_tps_at_95": round(primary["g1_margin_tps_at_95"], 4),
            "required_reported_tps_at_95": round(primary["required_reported_tps_at_95"], 4),
            "bar_tps": BAR_TPS,
            "margin_definition": ("reported number must beat the 126.378 bar by "
                                  "g1_margin_tps_at_95 so the 95%-worst single private "
                                  "draw (R_05) still clears the bar: reported* = bar/R_05."),
            "repro_headroom_by_reported": primary["repro_headroom_by_reported"],
        },
        "sigma_hw_robustness_sweep": sweep,
        "honesty_carry_forward": {
            "central_ci95_pct": sys_meta["boot_ci95_pct"],
            "boot_se_pct": round(boot_se, 4),
            "common_mode_public_anchor_se_pct": 5.318,
            "epub_fixed_se_pct": round(epubfix_se, 4),
            "common_mode_note": ("the 6.05% systematic SE is dominated by the shared "
                                 "official-128 public-anchor sampling noise (5.32pp "
                                 "common-mode); the gate-literal e_pub-fixed SE is "
                                 f"{epubfix_se:.2f}pp -> p_g1 even higher."),
            "window_block_bootstrap_conservatism": ("the ~10s spec-log window block "
                                                    "bootstrap is conservative vs an "
                                                    "i.i.d.-prompt resample."),
            "sigma_hw_provenance": hw["provenance"],
            "no_double_count": ("sigma_hw (hardware-allocation) and delta_stock common-"
                                "mode (prompt sampling) are independent -> added in "
                                "quadrature, not double-counted."),
        },
    }

    verdict = ("SINGLE_DRAW_G1_SAFE" if primary["p_g1_single_draw_pass"] >= 0.95
               else "SINGLE_DRAW_G1_AT_RISK")
    out["decision"] = {
        "primary_metric_p_g1_single_draw_pass": round(primary["p_g1_single_draw_pass"], 5),
        "test_metric_g1_margin_tps_at_95": round(primary["g1_margin_tps_at_95"], 4),
        "g1_safe_single_draw": bool(primary["p_g1_single_draw_pass"] >= 0.95),
        "verdict": verdict,
    }
    headline = (
        f"On a single private reproduction draw, the int4_mtp_batchinv fire has "
        f"P(G1-pass)={100*primary['p_g1_single_draw_pass']:.1f}%, clearing the 5% "
        f"reproduction rule; the 95%-worst draw still runs at {100*primary['r_realized']['p5']:.1f}% "
        f"of reported, so the official number need only beat the 126.378 bar by "
        f"{primary['g1_margin_tps_at_95']:.1f} TPS to clear G1 at 95% confidence."
    )
    out["headline"] = headline

    (out_dir / "g1_single_draw_repro_pass.json").write_text(json.dumps(out, indent=2))
    _plot(r_real_primary, primary, out_dir / "single_draw_distribution.png", sigma_canon)
    _print_summary(out)
    print(f"[g1single] wrote {out_dir/'g1_single_draw_repro_pass.json'}", flush=True)
    return out


def _plot(r_real: np.ndarray, primary: dict, path: Path, sigma_hw_frac: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[g1single] plot skipped: {e!r}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(r_real, bins=160, density=True, color="#4878CF", alpha=0.75,
            label="single-draw R = private/reported (delta_stock (x) sigma_hw)")
    ax.axvline(0.95, color="r", ls="-", lw=2,
               label="G1 reproduction floor R=0.95")
    r05 = primary["r_realized"]["p5"]
    ax.axvline(r05, color="darkorange", ls="--", lw=1.6,
               label=f"R_05 (95%-worst draw) = {r05:.3f}")
    ax.axvline(1.0, color="k", ls=":", alpha=0.5, label="R=1 (exact reproduce)")
    ax.axvline(primary["r_realized"]["mean"], color="green", ls="-.", alpha=0.7,
               label=f"mean R = {primary['r_realized']['mean']:.3f}")
    p_pass = primary["p_g1_single_draw_pass"]
    ax.set_title(f"Single-draw P(G1-pass) = {100*p_pass:.1f}%  "
                 f"(sigma_hw={100*sigma_hw_frac:.1f}% fractional)  PR #756")
    ax.set_xlabel("private_TPS / reported_TPS  (single private reproduction draw)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlim(0.80, 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[g1single] wrote {path}", flush=True)


def _print_summary(o: dict[str, Any]) -> None:
    a = o["arm_a_systematic_ratio"]; b = o["arm_b_single_draw_pass"]
    c = o["arm_c_invert_margin"]; d = o["decision"]; sm = b["sym_normal_crosscheck"]
    print("\n" + "=" * 74, flush=True)
    print("SINGLE-DRAW P(G1-pass)  delta_stock (x) sigma_hw   PR #756", flush=True)
    print("=" * 74, flush=True)
    sys_meta = o["systematic_leg_754"]
    print(f"  systematic (#754): central={sys_meta['point_central_pct']}%  "
          f"SE={sys_meta['boot_se_pct']}%  CI95={sys_meta['boot_ci95_pct']}  "
          f"P(DQ)check={sys_meta['p_dq_repro_check']}", flush=True)
    print(f"  sigma_hw canonical = {100*o['sigma_hw']['canonical_frac']:.2f}% fractional "
          f"(#159 cross 0.962% / #478 conv 1%); cross-dominated, multiplicative", flush=True)
    print(f"  Arm A: R_sys mean={a['r_systematic_mean']}  p5={a['r_systematic_p5']}  "
          f"systematic-only pass={a['systematic_only_pass_0p9863']}", flush=True)
    print(f"  Arm B: p_g1_single_draw_pass = {b['p_g1_single_draw_pass']}  "
          f"(sym-normal {sm['p_g1_pass_symnormal']})", flush=True)
    print(f"         P(within +/-5%) = {b['p_within_pm5pct_two_sided']}  "
          f"P(faster than +5%) = {b['p_faster_than_plus5pct']}", flush=True)
    print(f"         sigma_hw moved pass by {b['sigma_hw_moves_pass_by']:+.5f} vs 0.9863  "
          f"| gate-literal = {b['gate_literal_epubfixed']['p_g1_single_draw_pass']}", flush=True)
    print(f"  Arm C: R_05={c['r05_single_draw']}  repro margin={c['repro_margin_pp_at_95']}pp  "
          f"g1_margin_tps_at_95={c['g1_margin_tps_at_95']} TPS over {c['bar_tps']}", flush=True)
    print(f"         required reported >= {c['required_reported_tps_at_95']} TPS for 95% bar-clear", flush=True)
    print("  sigma_hw robustness:", flush=True)
    for name, s in o["sigma_hw_robustness_sweep"].items():
        print(f"    {name:34s} sigma={100*s['sigma_hw_frac']:.2f}%  "
              f"p_g1={s['p_g1_single_draw_pass']:.4f}  R05={s['r05']:.3f}  "
              f"margin={s['g1_margin_tps_at_95']:.2f}TPS", flush=True)
    print(f"  VERDICT: {d['verdict']}  (g1_safe={d['g1_safe_single_draw']})", flush=True)
    print("  " + o["headline"], flush=True)
    print("=" * 74 + "\n", flush=True)


def log_to_wandb(o: dict[str, Any], *, group: str, name: str, out_dir: Path) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return None
    b = o["arm_b_single_draw_pass"]; c = o["arm_c_invert_margin"]
    a = o["arm_a_systematic_ratio"]; dec = o["decision"]; sw = o["sigma_hw_robustness_sweep"]
    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "p_g1_single_draw_pass": b["p_g1_single_draw_pass"],
        "p_within_pm5pct_two_sided": b["p_within_pm5pct_two_sided"],
        "p_faster_than_plus5pct": b["p_faster_than_plus5pct"],
        "p_g1_symnormal": b["sym_normal_crosscheck"]["p_g1_pass_symnormal"],
        "p_g1_gate_literal_epubfixed": b["gate_literal_epubfixed"]["p_g1_single_draw_pass"],
        "systematic_only_pass": b["systematic_only_pass"],
        "sigma_hw_moves_pass_by": b["sigma_hw_moves_pass_by"],
        "p_dq_systematic_only": a["p_dq_systematic_only"],
        "r05_single_draw": c["r05_single_draw"],
        "repro_margin_pp_at_95": c["repro_margin_pp_at_95"],
        "g1_margin_tps_at_95": c["g1_margin_tps_at_95"],
        "required_reported_tps_at_95": c["required_reported_tps_at_95"],
        "sigma_hw_canonical_frac": o["sigma_hw"]["canonical_frac"],
        "r_realized_mean": b["r_realized"]["mean"],
        "r_realized_std": b["r_realized"]["std"],
        "g1_safe_single_draw": int(dec["g1_safe_single_draw"]),
        "central_ci95_lo": o["systematic_leg_754"]["boot_ci95_pct"][0],
        "central_ci95_hi": o["systematic_leg_754"]["boot_ci95_pct"][1],
    }
    for nm, s in sw.items():
        summary[f"sweep_{nm}_p_g1"] = s["p_g1_single_draw_pass"]
    run = init_wandb_run(
        job_type="g1-single-draw-repro-pass", agent="senpai", name=name, group=group,
        tags=["g1-single-draw-repro-pass", group, "sigma_hw", "delta_stock", "pr756"],
        notes=f"#756 single-draw P(G1-pass); verdict={dec['verdict']}; {o['headline']}",
        config={"analysis_only": True, "no_hf_job": True, "fires": 0,
                "bootstrap_draws": o["systematic_leg_754"]["draws"],
                "seed": o["systematic_leg_754"]["seed"],
                "sigma_hw_canonical_frac": o["sigma_hw"]["canonical_frac"],
                "bar_tps": BAR_TPS, "wandb_group": group,
                "source_report": o["systematic_leg_754"]["source_report"]})
    if run is None:
        print("[wandb] run not created; json is the record", flush=True)
        return None
    log_summary(run, summary, step=0)
    run.summary["verdict"] = dec["verdict"]
    run.summary["headline"] = o["headline"]
    for fn in ("g1_single_draw_repro_pass.json", "single_draw_distribution.png"):
        p = out_dir / fn
        if p.exists():
            try:
                log_file_artifact(run, path=p, name=f"g1single_{p.stem}",
                                  artifact_type="g1-single-draw-repro-pass")
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] artifact {fn} failed (non-fatal): {e!r}", flush=True)
    rid = run.id
    finish_wandb(run)
    print(f"[wandb] logged {name} id={rid} (group={group})", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=DEF_SOURCE_REPORT,
                    help="#749 faithful delta_stock report.json (server.log alongside)")
    ap.add_argument("--hw-envelope", default=DEF_HW_ENVELOPE)
    ap.add_argument("--hw-singledraw", default=DEF_HW_SINGLEDRAW)
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seed", type=int, default=730, help="systematic bootstrap seed (#754)")
    ap.add_argument("--hw-seed", type=int, default=756, help="aleatoric sigma_hw jitter seed")
    ap.add_argument("--out-dir", default=DEF_OUT_DIR)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="kanna/g1-single-draw-repro-pass")
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
        "primary_metric": {"name": "p_g1_single_draw_pass",
                           "value": dec["primary_metric_p_g1_single_draw_pass"]},
        "test_metric": {"name": "g1_margin_tps_at_95",
                        "value": dec["test_metric_g1_margin_tps_at_95"]},
    }
    print("SENPAI-RESULT:")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
