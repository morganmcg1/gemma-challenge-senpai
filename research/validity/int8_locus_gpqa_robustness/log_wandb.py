#!/usr/bin/env python3
"""PR #717 -- INT8-LOCUS GPQA-Diamond 30-seed robustness verdict + W&B log.

Reads the banked aggregate produced LOCALLY on the assigned A10G:
  - gpqa_int8locus_30seed.json  (aggregate_gpqa.py: int8-locus GPQA-D #31-sampled, 30 seeds)

Emits the robustness verdict:
  INT8_GPQA_COMFORTABLE  -- point clears 0.471 AND pooled-Wilson + seed-mean LB both clear
                            AND the point is stable/rising across @10/@21/@30.
  INT8_GPQA_MARGINAL_TIE -- point clears but >=1 CI lens straddles below and/or the point
                            drifts down (the int4-body #696 pathology, reproduced on int8-locus).
  INT8_GPQA_REAL_BELOW   -- the point itself fails the gate.

LOCAL only: analysis_only=1, official_tps=0, no_hf_job=1, fires=0. NO HF Job, NO submission,
served file untouched. NO FIRE.

The call is reported against EVERY CI lens (no thumb on the scale): pooled Wilson (the PR's
literal primary metric gpqa_d_int8locus_sampled_wilson_lo_30seed; anti-conservative),
seed-mean t-CI (the fixed-benchmark gate CI), clustered bootstrap (generalize beyond Diamond).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../target (holds scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATE = 0.471

# lawine #715 = the full 4-leg #515 panel (GSM8K/MMLU-Pro/GPQA-D/AIME) on this SAME int8-locus
# recovery (fern #659). Its GPQA-D leg is a shallow point check; #717 is the 30-seed depth that
# panel-leg lacks. Reconciliation = does the robust 30-seed instrument CONFIRM or REFINE #715's
# GPQA-D pass? The #715 GPQA-D leg point (if banked) is passed via --lawine715-gpqa-point.
LAWINE715_REF = "lawine #715 (#515 4-leg panel on the int8-locus recovery, fern #659)"


def _adjudication(g: dict) -> dict:
    """Side-by-side int8-locus vs #696 int4-body, the load-bearing comparison: does
    deepening body precision (int8 on L14-27) lift GPQA-D OUT of the int4-body marginal
    tie, or leave it stranded in the same pathology?"""
    ref = g.get("int4_body_reference", {})
    return {
        "int8_locus": {
            "point_30seed": g["point_item_mean"],
            "pooled_wilson_lo_30seed": g["pooled_wilson_lo"],
            "seed_mean_lo_30seed": g["seed_mean_lo"],
            "greedy_point": g.get("greedy_point"),
            "point_at_10": g["point_at_10"], "point_at_21": g["point_at_21"],
            "point_at_30": g["point_at_30"],
            "pooled_clears": g["pooled_wilson_clears_0471"],
            "seed_mean_clears": g["seed_mean_clears_0471"],
            "point_stable_or_rising": g["point_stable_or_rising"],
            "verdict": g["verdict"],
        },
        "int4_body_696": ref,
        "delta_point": g["int8_vs_int4_body_gpqa_delta"],
        "delta_pooled_lo": g.get("int8_vs_int4_body_pooled_lo_delta"),
        "delta_seed_mean_lo": g.get("int8_vs_int4_body_seed_mean_lo_delta"),
        "moves_up_and_out_of_int4_tie": g.get("moves_up_and_out_of_int4_tie"),
        "interpretation": (
            "int8-on-L14-27 LIFTS GPQA-D out of the int4-body marginal tie "
            "(both CI lenses now clear, point stable/rising)."
            if g.get("moves_up_and_out_of_int4_tie")
            else "int8-on-L14-27 does NOT rescue GPQA-D: it stays in the same marginal-tie "
                 "pathology as the int4-body arm (point clears but >=1 CI lens straddles "
                 "below and/or the point drifts down)."),
    }


def build_reconciliation(g: dict, lawine715_point: float | None) -> dict:
    """How the 30-seed GPQA-D robustness instrument relates to #715's panel GPQA-D leg."""
    pt = g["point_item_mean"]
    rec = {
        "lawine715_ref": LAWINE715_REF,
        "what_715_measured": "GPQA-D as one leg of the 4-benchmark #515 quality panel on the "
                             "int8-locus arm -- a shallow pass/fail point (the depth a single "
                             "panel pass cannot give).",
        "what_717_adds": "the 30-seed pooled-Wilson + seed-mean-t-CI + @10/@21/@30 drift "
                         "instrument (#696 verbatim) on the IDENTICAL int8-locus arm -- "
                         "turns #715's GPQA-D point into a robustness verdict.",
        "int8locus_gpqa_point_30seed": pt,
        "int8locus_gpqa_pooled_wilson_lo_30seed": g["pooled_wilson_lo"],
        "int8locus_gpqa_seed_mean_lo_30seed": g["seed_mean_lo"],
        "verdict_30seed": g["verdict"],
    }
    if lawine715_point is not None:
        rec["lawine715_gpqa_point"] = lawine715_point
        rec["point_delta_717_minus_715"] = pt - lawine715_point
        rec["reconciles"] = bool(abs(pt - lawine715_point) < 0.02)  # within ~1 item of n=198
        rec["note"] = (
            f"#717 30-seed point {pt:.4f} vs #715 panel GPQA-D leg {lawine715_point:.4f} "
            f"(delta {pt - lawine715_point:+.4f}); "
            + ("CONSISTENT -- the panel pass holds at depth."
               if abs(pt - lawine715_point) < 0.02
               else "DIVERGES -- the panel point is NOT robust at 30 seeds.")
            + f" Robustness verdict: {g['verdict']}."
        )
    else:
        rec["lawine715_gpqa_point"] = None
        rec["note"] = (
            "lawine #715 GPQA-D leg point not supplied; reconciliation is structural -- "
            f"#717 supplies the 30-seed depth (#715's panel leg lacks it). "
            f"Robustness verdict: {g['verdict']}."
        )
    return rec


def build_synthesis(g: dict, lawine715_point: float | None) -> dict:
    straddle = g.get("straddling_lenses", [])
    syn = {
        "pr": 717,
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # ---- headline ----
        "verdict": g["verdict"],
        "int8_gpqa_comfortable": g["int8_gpqa_comfortable"],
        # ---- PR primary + test metrics ----
        "gpqa_d_int8locus_sampled_wilson_lo_30seed": g["gpqa_d_int8locus_sampled_wilson_lo_30seed"],
        # ---- gate anchors ----
        "gpqa_gate_abs": GATE, "gpqa_base_sampled_3seed": g["base_sampled_3seed"],
        "arm": g["arm"], "fern_recipe_ref": g.get("fern_recipe_ref"),
        "int8_group_size": g.get("int8_group_size"),
        # ---- point + all CI lenses ----
        "gpqa_K_seeds": g["K_seeds"], "gpqa_n_items": g["n_items"],
        "gpqa_n_prompt_mismatch": g["n_prompt_mismatch"],
        "gpqa_sampled_point": g["point_item_mean"], "gpqa_sampled_pct_of_base": g["pct_of_base"],
        "gpqa_mean_acc": g["mean_acc"], "gpqa_std_acc": g["std_acc"],
        "gpqa_min_seed_acc": g["min_seed_acc"], "gpqa_max_seed_acc": g["max_seed_acc"],
        "gpqa_n_seeds_below_gate": g["n_seeds_below_gate"],
        "gpqa_pooled_wilson_lo": g["pooled_wilson_lo"], "gpqa_pooled_wilson_hi": g["pooled_wilson_hi"],
        "gpqa_pooled_wilson_lo_pct_of_base": g["pooled_wilson_lo_pct_of_base"],
        "gpqa_seed_mean_lo": g["seed_mean_lo"], "gpqa_seed_mean_hi": g["seed_mean_hi"],
        "gpqa_seed_mean_half_width": g["seed_mean_half_width"],
        "gpqa_bootstrap_items_lo": g["bootstrap_items_lo"], "gpqa_bootstrap_items_hi": g["bootstrap_items_hi"],
        "gpqa_wilson_n198_lo": g["wilson_n198_lo"], "gpqa_wilson_n198_hi": g["wilson_n198_hi"],
        # ---- clears bools ----
        "gpqa_point_clears_0471": g["point_clears_0471"],
        "gpqa_pooled_wilson_clears_0471": g["pooled_wilson_clears_0471"],
        "gpqa_seed_mean_clears_0471": g["seed_mean_clears_0471"],
        "gpqa_bootstrap_items_clears_0471": g["bootstrap_items_clears_0471"],
        "gpqa_worst_seed_clears_0471": g["worst_seed_clears_0471"],
        "gpqa_both_ci_lenses_clear": g["both_ci_lenses_clear"],
        "gpqa_straddling_lenses": straddle,
        "gpqa_ci_untightenable_on_diamond_population": g["ci_untightenable_on_diamond_population"],
        # ---- margins ----
        "gpqa_point_margin": g["point_margin"], "gpqa_pooled_wilson_margin": g["pooled_wilson_margin"],
        "gpqa_seed_mean_margin": g["seed_mean_margin"],
        # ---- trajectory / drift (the marginal-tie signature) ----
        "gpqa_point_at_10": g["point_at_10"], "gpqa_point_at_21": g["point_at_21"],
        "gpqa_point_at_30": g["point_at_30"],
        "gpqa_point_drift_10_to_30": g["point_drift_10_to_30"],
        "gpqa_point_stable_or_rising": g["point_stable_or_rising"],
        "gpqa_trajectory": g["trajectory"],
        # ---- cross-arm delta + adjudication vs #696 int4-body (g5lma5qf) ----
        "int4_body_reference": g.get("int4_body_reference"),
        "int4_body_point_30seed": g["int4_body_point_30seed"],
        "int8_vs_int4_body_gpqa_delta": g["int8_vs_int4_body_gpqa_delta"],
        "int8_vs_int4_body_pooled_lo_delta": g.get("int8_vs_int4_body_pooled_lo_delta"),
        "int8_vs_int4_body_seed_mean_lo_delta": g.get("int8_vs_int4_body_seed_mean_lo_delta"),
        "moves_up_and_out_of_int4_tie": g.get("moves_up_and_out_of_int4_tie"),
        "adjudication": _adjudication(g),
        # ---- lawine #715 reconciliation ----
        "reconciliation": build_reconciliation(g, lawine715_point),
        "wandb_run_ids_legs": {"gpqa_aggregate": "gpqa_int8locus_30seed.json"},
    }
    return syn


def nan_clean(obj):
    if isinstance(obj, dict):
        return {k: nan_clean(x) for k, x in obj.items()}
    if isinstance(obj, list):
        return [nan_clean(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def log_wandb(syn: dict, args) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                            log_json_artifact, log_summary)
    except Exception as exc:
        print(f"[717] wandb_logging import failed: {exc}; skipping W&B.", flush=True)
        return None
    run = init_wandb_run(
        job_type="analysis",
        agent="wirbel",
        name=args.wandb_name or "wirbel/int8-locus-gpqa-robustness",
        group=args.wandb_group or "int8-locus-gpqa-robustness-wirbel",
        notes="PR #717: does GPQA-Diamond clear 0.471 ROBUSTLY at 30 seeds on fern #659's "
              "int8-locus recovery, or reproduce the int4-body MARGINAL-TIE? LOCAL "
              "analysis_only, in-memory RTN fake-quant, NO FIRE.",
        tags=["int8-locus", "31basis", "gpqa-diamond", "30seed", "robustness",
              "analysis_only", "no_fire"],
        config={
            "pr": 717, "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "gpqa_gate_abs": GATE, "arm": syn["arm"], "fern_recipe_ref": syn.get("fern_recipe_ref"),
            "int8_group_size": syn.get("int8_group_size"),
            "protocol": "generation_config.json #31 (T=1.0 top_p=0.95 top_k=64, min_tokens=8), "
                        "#696 instrument verbatim",
        },
    )
    if run is None:
        print("[717] W&B not initialised (no key/mode); verdict saved locally.", flush=True)
        return None
    log_summary(run, nan_clean(syn), step=0, run_prefix="")
    log_json_artifact(run, name="int8_locus_gpqa_robustness_verdict", artifact_type="analysis",
                      data=nan_clean(syn))
    rid = run.id
    finish_wandb(run)
    return rid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--gpqa", type=Path, default=HERE / "gpqa_int8locus_30seed.json")
    ap.add_argument("--lawine715-gpqa-point", dest="lawine715_point", type=float, default=None,
                    help="optional #715 panel GPQA-D leg point for the numeric reconciliation")
    ap.add_argument("--out", type=Path, default=HERE / "verdict.json")
    args = ap.parse_args(argv)

    g = json.load(open(args.gpqa))
    syn = build_synthesis(g, args.lawine715_point)

    args.out.write_text(json.dumps(nan_clean(syn), indent=2))
    print(f"[717] verdict written -> {args.out}", flush=True)
    print(json.dumps({k: syn[k] for k in (
        "verdict", "int8_gpqa_comfortable", "gpqa_d_int8locus_sampled_wilson_lo_30seed",
        "gpqa_sampled_point", "gpqa_sampled_pct_of_base", "gpqa_point_clears_0471",
        "gpqa_pooled_wilson_clears_0471", "gpqa_seed_mean_clears_0471",
        "gpqa_straddling_lenses", "gpqa_point_at_10", "gpqa_point_at_21", "gpqa_point_at_30",
        "gpqa_point_drift_10_to_30", "gpqa_point_stable_or_rising",
        "int8_vs_int4_body_gpqa_delta",
    )}, indent=2), flush=True)

    rid = log_wandb(syn, args)
    if rid:
        print(f"[717] W&B run id: {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
