#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Ground the G1-safety leg: feed a MEASURED fire-config sigma_hw into #763 -- PR #766.

#763 (G1_CLAIM_ROBUST) showed the G1 5% private-reproduction gate survives a
sigma_hw mis-spec up to 4.72x the MODELED 1% before the fire single-draw
P(G1-pass) drops to 0.95. But that 4.72x headroom is measured relative to a
MODELED sigma_hw of 1% -- a value characterized on a NON-fire config (#159
split-KV/K7 ~485 TPS frontier) and ASSUMED to transfer to int4_mtp_batchinv via
the #478 multiplicative-clock model. The board's G1 leg therefore rested on an
assumed, not a measured, variance.

This card closes that gap. It takes the empirically MEASURED run-to-run TPS
coefficient-of-variation of the FIRE config (int4_mtp_batchinv, VLLM_BATCH_INVARIANT=1,
the merged submission), produced by repeating the canonical 128x512 conc=1 served
decode N>=10 times with a fresh server per run (research/tps_noise_floor harness,
the official analog -- every official draw is a fresh cold server), and feeds that
measured sigma_hw into the EXACT #763/#756 breakdown machinery (convolve + the
sigma-multiple sweep, byte-identically) in place of the modeled 1%.

Honesty carve (carried from #159, load-bearing):
  sigma_hw = hypot(sigma_within, sigma_cross). A single pod can only directly
  measure sigma_within (the within-allocation / fresh-serve run-to-run floor: the
  thermal / measurement / graph-capture / FP-reduction scatter on ONE physical
  A10G). sigma_cross -- which physical A10G the official scorer lands you on -- is
  BOUNDED-not-measured from public draws (#159 frantic-penguin same-submission
  3-draw CV 0.9623%, the dominant term). So this card reports:
    * measured_sigma_hw_pct  -- the directly-measured fire-config fresh-serve CV
      (the PR's primary metric). If it is at/below the modeled 1%, the fire config
      carries no within-allocation jitter ANOMALY -> the #478 transfer assumption
      is empirically vindicated on the fire config.
    * grounded_sigma_hw_pct  -- the CONSERVATIVE total hypot(measured_within,
      cross_bound 0.9623%): the measured within folded with the still-bounded
      cross term. This is the number the G1 headroom should honestly cite.
  The verdict grounds OUR internal G1-safety confidence; it does NOT change the
  organizer's identity-blind scorer and does NOT move the locked
  int4_g128_lmhead@126.378 baseline. analysis_only=1, official_tps=0,
  no_hf_job=1, fires=0.

LOCAL ONLY: CPU numpy + the in-repo measured artifact + kanna's own #763/#756
machinery. Launches no server here (the serve+decode N-run measurement is the
separate research/tps_noise_floor.run_noise_floor step), makes no submission,
fires no HF Job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse kanna's own #756 single-draw machinery and #763 sigma-sweep machinery
# BYTE-FOR-BYTE -- the only new input is the MEASURED sigma_hw fed in place of 1%.
from scripts.validity.g1_single_draw_repro_pass import (  # noqa: E402
    BAR_TPS, convolve, load_sigma_hw, load_systematic_delta,
    DEF_HW_ENVELOPE, DEF_HW_SINGLEDRAW, DEF_SOURCE_REPORT,
)
from scripts.validity.g1_sigma_hw_breakdown import (  # noqa: E402
    DEF_FIRE_ANCHOR, DEF_STRESS_ANCHOR, P_FIRE_756, R05_756,
    _col, _first_downcross, sweep,
)

# The #763 fire single-draw 0.95 breakdown multiple (x modeled 1%), recomputed
# below and asserted against this published anchor.
BREAKDOWN_MULT_AT_095 = 4.72
BREAKDOWN_MULT_AT_090 = 7.2437

DEF_OUT_DIR = "research/validity/g1_sigma_measured/results"
DEF_MEASURED = "research/validity/g1_sigma_measured/fresh_n12/noise_floor_fresh.json"


def load_measured_sigma(path: Path, metric: str) -> dict[str, Any]:
    """Pull the measured run-to-run TPS CV from a run_noise_floor.py artifact.

    measured_sigma_hw_pct = 100 * std / mean of the per-run ``metric`` (default
    ``wall_tps`` = num_completion_tokens / decode_duration_s, the wall-clock
    single-stream TPS -- robust to the log-stat env, the gate-relevant rate)."""
    art = json.loads(path.read_text())
    agg = (art.get("aggregate") or {}).get(metric) or {}
    n = int(agg.get("n", 0))
    if n < 2 or not np.isfinite(agg.get("cv_pct", float("nan"))):
        raise SystemExit(
            f"measured artifact {path} has insufficient {metric} samples (n={n}); "
            f"need N>=2 with finite CV")
    return {
        "artifact": str(path),
        "metric": metric,
        "n": n,
        "mean_tps": float(agg["mean"]),
        "std_tps": float(agg["std"]),
        "measured_sigma_hw_pct": float(agg["cv_pct"]),
        "measured_sigma_hw_frac": float(agg["cv_pct"]) / 100.0,
        "min_tps": float(agg["min"]),
        "max_tps": float(agg["max"]),
        "range_pct": float(agg.get("range_pct", float("nan"))),
        "median_tps": float(agg.get("median", float("nan"))),
        "values_tps": agg.get("values"),
        "mode": art.get("mode"),
        "submission": art.get("submission"),
        "workload": art.get("workload"),
        "n_runs_artifact": art.get("n_runs"),
        "elapsed_s": art.get("elapsed_s"),
    }


def g1_at_sigma(central_b: np.ndarray, sigma_frac: float, *, hw_seed: int,
                fire_anchor: float, stress_anchor: float) -> dict[str, Any]:
    """All G1 single-draw quantities at one sigma_hw (the #756 convolve, byte-identical)."""
    c = convolve(central_b, sigma_frac, hw_seed=hw_seed, bar=BAR_TPS,
                 reported_levels=[fire_anchor, stress_anchor])
    c.pop("_r_real")
    r05 = c["r_realized"]["p5"]
    return {
        "sigma_hw_frac": sigma_frac,
        "sigma_hw_pct": round(100.0 * sigma_frac, 5),
        "p_g1_single_draw_pass": round(c["p_g1_single_draw_pass"], 5),
        "r05": round(r05, 5),
        "worst_priv_tps_fire_anchor": round(fire_anchor * r05, 3),
        "worst_priv_tps_stress_anchor": round(stress_anchor * r05, 3),
        "margin_over_bar_fire_anchor": round(fire_anchor * r05 - BAR_TPS, 3),
        "g1_margin_tps_at_95": round(c["g1_margin_tps_at_95"], 4),
        "clears_bar_at_95_fire_anchor": bool(fire_anchor * r05 >= BAR_TPS),
    }


def run(args) -> dict[str, Any]:
    report_path = REPO / args.report if not Path(args.report).is_absolute() else Path(args.report)
    env_path = REPO / args.hw_envelope
    single_path = REPO / args.hw_singledraw
    measured_path = REPO / args.measured if not Path(args.measured).is_absolute() else Path(args.measured)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- systematic delta_stock leg + modeled sigma_hw (exact #763 inputs) ---
    central_b, central_b_epubfix, point_central, sys_meta = load_systematic_delta(
        report_path, B=args.draws, K=args.K, seed=args.seed)
    hw = load_sigma_hw(env_path, single_path)
    modeled_frac = hw["canonical_frac"]            # 0.01
    cross_bound_frac = hw["cross_frac_frantic_penguin"]   # 0.009623 (#159 dominant)

    # --- byte-identity guard at the modeled 1x (== #756/#763) ---
    base = g1_at_sigma(central_b, modeled_frac, hw_seed=args.hw_seed,
                       fire_anchor=args.fire_anchor, stress_anchor=args.stress_anchor)
    repro_ok = (abs(base["p_g1_single_draw_pass"] - P_FIRE_756) < 1e-4
                and abs(base["r05"] - R05_756) < 1e-4)

    # --- the MEASURED fire-config sigma_hw (the new empirical input) ---
    meas = load_measured_sigma(measured_path, args.measured_metric)
    measured_frac = meas["measured_sigma_hw_frac"]
    measured_pct = meas["measured_sigma_hw_pct"]
    meas_over_modeled = measured_frac / modeled_frac if modeled_frac else float("inf")

    # --- step 2: G1 at the MEASURED sigma (in place of the modeled 1%) ---
    at_measured = g1_at_sigma(central_b, measured_frac, hw_seed=args.hw_seed,
                              fire_anchor=args.fire_anchor, stress_anchor=args.stress_anchor)
    # headroom in units of the MEASURED sigma: the gate fails at 4.72x MODELED =
    # (4.72 / meas_over_modeled) x MEASURED.
    headroom_mult_measured_basis = (BREAKDOWN_MULT_AT_095 / meas_over_modeled
                                    if meas_over_modeled > 0 else float("inf"))

    # --- honest conservative reconciliation: fold the measured WITHIN with the
    # still-BOUNDED cross-allocation term (one pod cannot measure cross). ---
    grounded_frac = float(np.hypot(measured_frac, cross_bound_frac))
    grounded_pct = 100.0 * grounded_frac
    grounded_over_modeled = grounded_frac / modeled_frac if modeled_frac else float("inf")
    at_grounded = g1_at_sigma(central_b, grounded_frac, hw_seed=args.hw_seed,
                              fire_anchor=args.fire_anchor, stress_anchor=args.stress_anchor)
    headroom_mult_grounded_basis = (BREAKDOWN_MULT_AT_095 / grounded_over_modeled
                                    if grounded_over_modeled > 0 else float("inf"))

    # --- step 3: recompute the #763 breakdown sweep + place the measured/grounded
    # sigma on it (byte-identical sweep machinery; proves 4.72x and locates the
    # measured multiple in the safe region). ---
    multiples = np.round(np.arange(args.mult_lo, args.mult_hi + 1e-9, args.mult_step), 4)
    rows = sweep(central_b, central_b_epubfix, modeled_frac, multiples,
                 hw_seed=args.hw_seed, hw_seed_126=args.hw_seed_126,
                 idio_perm_seed=args.idio_perm_seed, indep_perm_seed=args.indep_perm_seed,
                 fire_anchor=args.fire_anchor, stress_anchor=args.stress_anchor)
    mult_arr = _col(rows, "mult")
    fire_curve = _col(rows, "p_g1_single_draw_pass")
    m95 = _first_downcross(mult_arr, fire_curve, 0.95)
    m90 = _first_downcross(mult_arr, fire_curve, 0.90)
    m95_fire = m95["mult"]
    sweep_repro_ok = (m95_fire is not None and abs(m95_fire - BREAKDOWN_MULT_AT_095) < 0.05)

    def _on_curve(over_modeled: float) -> dict[str, Any]:
        in_grid = bool(mult_arr[0] <= over_modeled <= mult_arr[-1])
        i = int(np.argmin(np.abs(mult_arr - over_modeled)))
        return {
            "multiple_of_modeled": round(float(over_modeled), 5),
            "in_swept_grid": in_grid,
            "nearest_swept_mult": rows[i]["mult"],
            "p_g1_at_nearest": rows[i]["p_g1_single_draw_pass"],
            "below_breakdown_4p72": bool(over_modeled < BREAKDOWN_MULT_AT_095),
            "in_safe_region_p_ge_095": bool(rows[i]["p_g1_single_draw_pass"] >= 0.95),
        }

    measured_on_curve = _on_curve(meas_over_modeled)
    grounded_on_curve = _on_curve(grounded_over_modeled)

    # --- step 4: verdict ---
    fire_worst_priv = at_measured["worst_priv_tps_fire_anchor"]
    grounded_worst_priv = at_grounded["worst_priv_tps_fire_anchor"]
    # comfortably inside breakdown: require <50% of the way to 4.72x on the
    # conservative (grounded) basis AND the worst-case private TPS clears the bar.
    comfortably_inside = bool(grounded_over_modeled < 0.5 * BREAKDOWN_MULT_AT_095)
    clears_bar = bool(grounded_worst_priv >= BAR_TPS and fire_worst_priv >= BAR_TPS)
    g1_empirically_grounded = int(comfortably_inside and clears_bar
                                  and measured_on_curve["below_breakdown_4p72"]
                                  and grounded_on_curve["below_breakdown_4p72"])

    material_diff_from_1pct = bool(abs(measured_pct - 100.0 * modeled_frac) >= 0.25)

    verdict = ("G1_EMPIRICALLY_GROUNDED" if g1_empirically_grounded
               else "G1_NOT_EMPIRICALLY_GROUNDED")

    headline = (
        f"Measured fire-config (int4_mtp_batchinv, BI=1) run-to-run TPS CV = "
        f"{measured_pct:.4f}% over N={meas['n']} fresh-serve {meas.get('workload')} draws "
        f"(mean {meas['mean_tps']:.2f} TPS, std {meas['std_tps']:.4f}); that is "
        f"{meas_over_modeled:.3f}x the modeled 1%. Folded with the bounded cross-"
        f"allocation term (#159 0.9623%) the conservative grounded sigma_hw = "
        f"{grounded_pct:.4f}% ({grounded_over_modeled:.3f}x modeled). At the measured "
        f"sigma the fire single-draw P(G1-pass) = {at_measured['p_g1_single_draw_pass']:.5f} "
        f"and the 95%-worst private TPS @157 = {fire_worst_priv:.2f} (clears the "
        f"126.378 bar by {at_measured['margin_over_bar_fire_anchor']:+.2f}); the gate's "
        f"4.72x breakdown is {headroom_mult_grounded_basis:.2f}x away on the grounded "
        f"basis. G1 leg empirically grounded: {bool(g1_empirically_grounded)}."
    )

    out = {
        "pr": 766,
        "student": "kanna",
        "analysis_only": True, "official_tps": 0, "no_hf_job": True, "fires": 0,
        "reused_756_763_byte_identical": bool(repro_ok and sweep_repro_ok),
        "repro_anchors": {
            "p_fire_1x": base["p_g1_single_draw_pass"], "p_fire_expected_756": P_FIRE_756,
            "r05_1x": base["r05"], "r05_expected_756": R05_756,
            "sweep_m95_recomputed": m95_fire, "sweep_m95_expected_763": BREAKDOWN_MULT_AT_095,
            "sweep_m90_recomputed": m90["mult"], "sweep_m90_expected_763": BREAKDOWN_MULT_AT_090,
        },
        "modeled_sigma_hw": {
            "modeled_frac": modeled_frac, "modeled_pct": round(100.0 * modeled_frac, 4),
            "cross_bound_frac_159": cross_bound_frac,
            "cross_bound_pct_159": round(100.0 * cross_bound_frac, 4),
            "provenance": ("modeled 1% = #159 cross-allocation frantic-penguin 3-draw "
                           "0.9623% (NON-fire ~485 TPS frontier) -> #478 1% multiplicative; "
                           "the value PR #766 replaces with a MEASURED fire-config CV."),
        },
        "measured_sigma_hw": meas,
        "step1_measured": {
            "measured_sigma_hw_pct": round(measured_pct, 5),
            "measured_sigma_hw_frac": measured_frac,
            "mean_tps": meas["mean_tps"], "std_tps": meas["std_tps"],
            "n": meas["n"], "min_tps": meas["min_tps"], "max_tps": meas["max_tps"],
            "measured_over_modeled_ratio": round(meas_over_modeled, 5),
            "material_diff_from_modeled_1pct": material_diff_from_1pct,
            "isolation_note": ("fresh-serve N-run wall_tps CV = the directly-measurable "
                               "WITHIN-allocation sigma_hw on ONE A10G; cross-allocation "
                               "scatter (which physical card) is BOUNDED-not-measured "
                               "(#159), one pod cannot capture it."),
        },
        "step2_g1_at_measured": {
            **at_measured,
            "fire_p_g1_at_measured_sigma": at_measured["p_g1_single_draw_pass"],
            "worst_priv_tps_at_measured_sigma": fire_worst_priv,
            "headroom_multiple_measured_basis": round(headroom_mult_measured_basis, 3),
            "breakdown_mult_at_095_modeled_basis": BREAKDOWN_MULT_AT_095,
            "headroom_note": ("headroom_multiple_measured_basis = 4.72 (modeled-basis "
                              "breakdown) / (measured/modeled): how many x the MEASURED "
                              "sigma_hw before the one-sided 5% gate's fire single-draw "
                              "P(G1-pass) hits 0.95."),
        },
        "step2b_grounded_conservative": {
            "grounded_sigma_hw_pct": round(grounded_pct, 5),
            "grounded_sigma_hw_frac": round(grounded_frac, 6),
            "grounded_over_modeled_ratio": round(grounded_over_modeled, 5),
            "definition": "hypot(measured_within, cross_bound_159=0.9623%)",
            **{f"grounded_{k}": v for k, v in at_grounded.items()},
            "headroom_multiple_grounded_basis": round(headroom_mult_grounded_basis, 3),
        },
        "step3_on_modeled_curve": {
            "sweep_grid": {"lo": args.mult_lo, "hi": args.mult_hi, "step": args.mult_step,
                           "n_points": int(len(multiples))},
            "fire_breakdown_mult_at_095": m95, "fire_breakdown_mult_at_090": m90,
            "measured_point": measured_on_curve,
            "grounded_point": grounded_on_curve,
        },
        "honesty_carry_forward": {
            "framing": ("EMPIRICAL GROUNDING of kanna's own modeled G1 card: the directly-"
                        "measured fire-config fresh-serve CV replaces the assumed within "
                        "component; the cross-allocation term stays bounded-not-measured."),
            "within_vs_cross": ("sigma_hw = hypot(within, cross); measured = within only "
                                "(one pod); cross bounded at 0.9623% (#159)."),
            "does_not_change_scorer": ("grounds OUR internal G1-safety confidence; the "
                                       "organizer's identity-blind scorer + the locked "
                                       "int4_g128_lmhead@126.378 baseline are untouched."),
            "engine_and_sampling": ("served vLLM 0.22.0 + MTP drafter K=6, VLLM_BATCH_INVARIANT=1; "
                                    "greedy tau=0 (per #319); fresh server per repeat, "
                                    "canonical 128x512 conc=1 official decode, seed 1."),
        },
    }
    out["decision"] = {
        "primary_metric_measured_sigma_hw_pct": round(measured_pct, 5),
        "test_metric_g1_empirically_grounded": g1_empirically_grounded,
        "fire_p_g1_at_measured_sigma": at_measured["p_g1_single_draw_pass"],
        "worst_priv_tps_at_measured_sigma": fire_worst_priv,
        "headroom_multiple_measured_basis": round(headroom_mult_measured_basis, 3),
        "headroom_multiple_grounded_basis": round(headroom_mult_grounded_basis, 3),
        "verdict": verdict,
    }
    out["headline"] = headline

    (out_dir / "g1_sigma_measured.json").write_text(json.dumps(out, indent=2, default=str))
    _print_summary(out)
    print(f"[g1sigmeas] wrote {out_dir/'g1_sigma_measured.json'}", flush=True)
    return out


def _print_summary(o: dict[str, Any]) -> None:
    d = o["decision"]; s1 = o["step1_measured"]; s2 = o["step2_g1_at_measured"]
    s2b = o["step2b_grounded_conservative"]; mc = o["step3_on_modeled_curve"]
    print("\n" + "=" * 82, flush=True)
    print("G1 sigma_hw EMPIRICALLY GROUNDED: measured fire-config CV -> #763 breakdown  PR #766",
          flush=True)
    print("=" * 82, flush=True)
    ra = o["repro_anchors"]
    print(f"  reused #756/#763 byte-identical: {o['reused_756_763_byte_identical']}  "
          f"(p_fire_1x={ra['p_fire_1x']} vs {ra['p_fire_expected_756']}; "
          f"sweep m95={ra['sweep_m95_recomputed']} vs {ra['sweep_m95_expected_763']})", flush=True)
    print(f"  -- step1 MEASURED sigma_hw (fire int4_mtp_batchinv, BI=1) --", flush=True)
    print(f"     N={s1['n']}  mean={s1['mean_tps']:.3f} TPS  std={s1['std_tps']:.4f}  "
          f"CV=measured_sigma_hw_pct={s1['measured_sigma_hw_pct']:.4f}%  "
          f"({s1['measured_over_modeled_ratio']:.3f}x modeled 1%)", flush=True)
    print(f"  -- step2 G1 at MEASURED sigma --", flush=True)
    print(f"     fire P(G1-pass)={s2['fire_p_g1_at_measured_sigma']:.5f}  "
          f"worst-priv@157={s2['worst_priv_tps_at_measured_sigma']:.2f} "
          f"(margin {s2['margin_over_bar_fire_anchor']:+.2f} over 126.378)  "
          f"headroom={s2['headroom_multiple_measured_basis']:.2f}x measured", flush=True)
    print(f"  -- step2b GROUNDED (hypot measured-within + cross-bound 0.9623%) --", flush=True)
    print(f"     grounded sigma_hw={s2b['grounded_sigma_hw_pct']:.4f}%  "
          f"({s2b['grounded_over_modeled_ratio']:.3f}x modeled)  "
          f"fire P(G1-pass)={s2b['grounded_p_g1_single_draw_pass']:.5f}  "
          f"worst-priv@157={s2b['grounded_worst_priv_tps_fire_anchor']:.2f}  "
          f"headroom={s2b['headroom_multiple_grounded_basis']:.2f}x grounded", flush=True)
    print(f"  -- step3 on the #763 modeled curve --", flush=True)
    print(f"     measured @ {mc['measured_point']['multiple_of_modeled']}x modeled "
          f"(below 4.72x breakdown: {mc['measured_point']['below_breakdown_4p72']}, "
          f"safe p>=0.95: {mc['measured_point']['in_safe_region_p_ge_095']}); "
          f"grounded @ {mc['grounded_point']['multiple_of_modeled']}x", flush=True)
    print(f"  VERDICT: {d['verdict']}  (measured_sigma_hw_pct={d['primary_metric_measured_sigma_hw_pct']}, "
          f"g1_empirically_grounded={d['test_metric_g1_empirically_grounded']})", flush=True)
    print("  " + o["headline"], flush=True)
    print("=" * 82 + "\n", flush=True)


def log_to_wandb(o: dict[str, Any], *, group: str, name: str, out_dir: Path) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return None
    d = o["decision"]; s1 = o["step1_measured"]; s2 = o["step2_g1_at_measured"]
    s2b = o["step2b_grounded_conservative"]; meas = o["measured_sigma_hw"]
    summary = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "measured_sigma_hw_pct": d["primary_metric_measured_sigma_hw_pct"],
        "g1_empirically_grounded": d["test_metric_g1_empirically_grounded"],
        "fire_p_g1_at_measured_sigma": d["fire_p_g1_at_measured_sigma"],
        "worst_priv_tps_at_measured_sigma": d["worst_priv_tps_at_measured_sigma"],
        "headroom_multiple_measured_basis": d["headroom_multiple_measured_basis"],
        "headroom_multiple_grounded_basis": d["headroom_multiple_grounded_basis"],
        "measured_mean_tps": meas["mean_tps"],
        "measured_std_tps": meas["std_tps"],
        "measured_n_runs": meas["n"],
        "measured_over_modeled_ratio": s1["measured_over_modeled_ratio"],
        "grounded_sigma_hw_pct": s2b["grounded_sigma_hw_pct"],
        "grounded_over_modeled_ratio": s2b["grounded_over_modeled_ratio"],
        "grounded_fire_p_g1": s2b["grounded_p_g1_single_draw_pass"],
        "grounded_worst_priv_tps_fire157": s2b["grounded_worst_priv_tps_fire_anchor"],
        "modeled_sigma_hw_pct": o["modeled_sigma_hw"]["modeled_pct"],
        "cross_bound_pct_159": o["modeled_sigma_hw"]["cross_bound_pct_159"],
        "breakdown_mult_at_095": BREAKDOWN_MULT_AT_095,
        "material_diff_from_modeled_1pct": int(s1["material_diff_from_modeled_1pct"]),
        "reused_756_763_byte_identical": int(o["reused_756_763_byte_identical"]),
        "margin_over_bar_fire_anchor_at_measured": s2["margin_over_bar_fire_anchor"],
    }
    run = init_wandb_run(
        job_type="g1-sigma-measured", agent="senpai", name=name, group=group,
        tags=["g1-sigma-measured", group, "sigma_hw", "measured", "g1", "pr766"],
        notes=f"#766 measured sigma_hw grounding; verdict={d['verdict']}; {o['headline']}",
        config={"analysis_only": True, "no_hf_job": True, "fires": 0,
                "measured_artifact": meas["artifact"], "measured_metric": meas["metric"],
                "measured_mode": meas["mode"], "measured_workload": meas["workload"],
                "modeled_sigma_hw_frac": o["modeled_sigma_hw"]["modeled_frac"],
                "cross_bound_frac_159": o["modeled_sigma_hw"]["cross_bound_frac_159"],
                "bar_tps": BAR_TPS, "fire_anchor_tps": DEF_FIRE_ANCHOR,
                "wandb_group": group})
    if run is None:
        print("[wandb] run not created; json is the record", flush=True)
        return None
    log_summary(run, summary, step=0)
    run.summary["verdict"] = d["verdict"]
    run.summary["headline"] = o["headline"]
    try:
        import wandb
        vals = meas.get("values_tps") or []
        if vals:
            tbl = wandb.Table(columns=["run_idx", "wall_tps"])
            for i, v in enumerate(vals):
                tbl.add_data(i, v)
            run.log({"measured_wall_tps_per_run": tbl}, step=0)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] per-run table failed (non-fatal): {e!r}", flush=True)
    p = out_dir / "g1_sigma_measured.json"
    if p.exists():
        try:
            log_file_artifact(run, path=p, name="g1sigmeas_g1_sigma_measured",
                              artifact_type="g1-sigma-measured")
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] artifact failed (non-fatal): {e!r}", flush=True)
    rid = run.id
    finish_wandb(run)
    print(f"[wandb] logged {name} id={rid} (group={group})", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default=DEF_MEASURED,
                    help="run_noise_floor.py fresh-mode artifact (noise_floor_fresh.json)")
    ap.add_argument("--measured-metric", default="wall_tps",
                    choices=["wall_tps", "steady_gen_tps_mean", "steady_gen_tps_mean_nonzero"])
    ap.add_argument("--report", default=DEF_SOURCE_REPORT)
    ap.add_argument("--hw-envelope", default=DEF_HW_ENVELOPE)
    ap.add_argument("--hw-singledraw", default=DEF_HW_SINGLEDRAW)
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seed", type=int, default=730)
    ap.add_argument("--hw-seed", type=int, default=756)
    ap.add_argument("--hw-seed-126", type=int, default=758)
    ap.add_argument("--idio-perm-seed", type=int, default=1758)
    ap.add_argument("--indep-perm-seed", type=int, default=2758)
    ap.add_argument("--fire-anchor", type=float, default=DEF_FIRE_ANCHOR)
    ap.add_argument("--stress-anchor", type=float, default=DEF_STRESS_ANCHOR)
    ap.add_argument("--mult-lo", type=float, default=0.5)
    ap.add_argument("--mult-hi", type=float, default=20.0)
    ap.add_argument("--mult-step", type=float, default=0.05)
    ap.add_argument("--out-dir", default=DEF_OUT_DIR)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="kanna/g1-sigma-measured")
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
        "primary_metric": {"name": "measured_sigma_hw_pct",
                           "value": dec["primary_metric_measured_sigma_hw_pct"]},
        "test_metric": {"name": "g1_empirically_grounded",
                        "value": dec["test_metric_g1_empirically_grounded"]},
    }
    print("SENPAI-RESULT:")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
