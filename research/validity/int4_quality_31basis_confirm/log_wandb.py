#!/usr/bin/env python3
"""PR #696 -- joint {greedy,#31-sampled} x {GPQA-D,AIME} basis-correction verdict + W&B log.

Reads the two banked aggregates produced LOCALLY on the assigned A10G:
  - gpqa_30seed.json  (aggregate_gpqa.py: int4-body GPQA-Diamond #31-sampled, target 30 seeds)
  - aime_sampled.json (aggregate_aime.py: int4-body AIME #31-sampled + greedy anchor, same arm)

Builds the decision-grade two-gate table and emits the verdict string:
  BASIS_CLEARS_BOTH / BASIS_GPQA_CLEARS_AIME_REAL / BASIS_BOTH_REAL / BASIS_INDETERMINATE.

LOCAL only: analysis_only=1, official_tps=0, no_hf_job=1, fires=0. NO HF Job, NO submission,
served file untouched, locked int4_g128_lmhead @126.378 untouched. NO FIRE.

The GPQA "comfortable clear" call is reported against EVERY CI lens (no thumb on the scale):
  - pooled Wilson (PR's literal primary metric gpqa_d_sampled_wilson_lo_30seed; anti-conservative)
  - seed-mean t-CI (conditions on the fixed Diamond-198 instrument; the fixed-benchmark gate CI)
  - clustered bootstrap over items (generalize beyond Diamond; UN-tightenable at n=198)
The headline verdict follows the PR's literal primary metric, with the honest caveats flagged.
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

GPQA_GATE = 0.471
AIME_GATE = 0.420
AIME_BASE_GREEDY = 0.4667

# two-sided 95% t critical values (df = K-1); -> 1.96 for large K.
_TCRIT = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
    10: 2.262, 15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045, 35: 2.032, 40: 2.023,
    50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 150: 1.976, 200: 1.972,
}


def _tcrit(K: int) -> float:
    if K in _TCRIT:
        return _TCRIT[K]
    keys = sorted(_TCRIT)
    lo = max([k for k in keys if k <= K], default=keys[0])
    return _TCRIT[lo]


def seeds_to_resolve_seedmean(point: float, std: float, gate: float, kmax: int = 400):
    """Smallest K such that the seed-mean t-CI LB (point - t(K-1)*std/sqrt(K)) >= gate, IF the
    point estimate and per-seed std hold. None if point<=gate (LB converges to point, never clears).
    Caveat (flagged in the report): the GPQA-D point has DRIFTED DOWN with seeds (0.4894@10 ->
    0.4822@21 -> 0.4783@30), so 'point holds' is optimistic; more seeds may not resolve it."""
    if point <= gate:
        return None
    for K in range(2, kmax + 1):
        lo = point - _tcrit(K) * std / math.sqrt(K)
        if lo >= gate:
            return K
    return None


def decide_verdict(g: dict, a: dict) -> dict:
    # ---- GPQA-D: does the #31 basis comfortably clear 0.471? ----
    g_pooled_lo = g["pooled_wilson_lo"]          # PR literal primary metric
    g_seed_lo = g["seed_mean_lo"]                # fixed-benchmark CI
    g_boot_lo = g["bootstrap_items_lo"]          # generalization CI (untightenable)
    g_point = g["point_item_mean"]
    g_pooled_clears = bool(g_pooled_lo >= GPQA_GATE)
    g_seed_clears = bool(g_seed_lo >= GPQA_GATE)
    # "comfortable" = the PR primary metric clears AND the honest fixed-benchmark CI clears too.
    gpqa_comfortable = bool(g_pooled_clears and g_seed_clears)
    gpqa_straddles = bool(g_point >= GPQA_GATE and not g_pooled_clears)

    # ---- AIME: does it recover toward 90% on the sampled basis, or stay below? ----
    a_point = a["point_item_mean"]
    a_pct = a["pct_of_base_greedy"]              # PR test metric: aime_sampled_pct_of_base
    a_pooled_lo = a["pooled_wilson_lo"]
    a_point_clears_gate = bool(a_point >= AIME_GATE)
    a_pooled_clears_gate = bool(a_pooled_lo >= AIME_GATE)
    a_clears_90 = bool(a_pct >= 90.0)
    # AIME "comfortably clears": >=90% of base AND the Wilson LB clears the 0.420 gate.
    aime_comfortable = bool(a_clears_90 and a_pooled_clears_gate)
    # AIME "survives below" (genuinely-harder leg): point/CI stays under the gate.
    aime_real_below = bool(not a_point_clears_gate or not a_clears_90)

    # Per-leg classification (each leg gets its own honest label) ----------------
    #   GPQA: CLEARS = point clears 0.471 AND pooled-Wilson + seed-mean LB both clear;
    #         INDETERMINATE = point clears but a CI LB straddles below; REAL = point itself below.
    #   AIME: CLEARS = >=90% AND Wilson LB clears 0.420; REAL = point/pct below the gate;
    #         INDETERMINATE = neither (point clears the abs gate but pct<90 or LB straddles).
    if gpqa_comfortable:
        gpqa_leg = "CLEARS"
    elif gpqa_straddles:
        gpqa_leg = "INDETERMINATE"
    else:
        gpqa_leg = "REAL"
    if aime_comfortable:
        aime_leg = "CLEARS"
    elif aime_real_below:
        aime_leg = "REAL"
    else:
        aime_leg = "INDETERMINATE"

    # Headline verdict over the 4 canonical buckets, with an honest composite when the two
    # legs disagree in a way the 4-bucket taxonomy does not name (GPQA marginal-tie + AIME real).
    if gpqa_leg == "CLEARS" and aime_leg == "CLEARS":
        verdict = "BASIS_CLEARS_BOTH"
        closest_canonical = "BASIS_CLEARS_BOTH"
    elif gpqa_leg == "CLEARS" and aime_leg == "REAL":
        verdict = "BASIS_GPQA_CLEARS_AIME_REAL"
        closest_canonical = "BASIS_GPQA_CLEARS_AIME_REAL"
    elif gpqa_leg == "REAL" and aime_leg == "REAL":
        verdict = "BASIS_BOTH_REAL"
        closest_canonical = "BASIS_BOTH_REAL"
    elif gpqa_leg == "INDETERMINATE" and aime_leg == "REAL":
        # Composite: GPQA-D clears on POINT (91.3%) but every CI LB straddles 0.471 at 30 seeds
        # (BASIS_INDETERMINATE leg); AIME does NOT recover and stays below 0.420 (the REAL leg).
        # AIME is the binding blocker regardless of how the GPQA marginal-tie resolves.
        verdict = "BASIS_GPQA_INDETERMINATE_AIME_REAL"
        closest_canonical = "BASIS_GPQA_CLEARS_AIME_REAL (only if GPQA 'clears' is read as point-clears; its Wilson LB straddles)"
    elif gpqa_leg == "INDETERMINATE" and aime_leg in ("CLEARS", "INDETERMINATE"):
        verdict = "BASIS_INDETERMINATE"
        closest_canonical = "BASIS_INDETERMINATE"
    else:
        verdict = f"BASIS_GPQA_{gpqa_leg}_AIME_{aime_leg}"
        closest_canonical = verdict

    return {
        "verdict": verdict,
        "closest_canonical": closest_canonical,
        "gpqa_leg": gpqa_leg,
        "aime_leg": aime_leg,
        "gpqa_pooled_wilson_clears_0471": g_pooled_clears,
        "gpqa_seed_mean_clears_0471": g_seed_clears,
        "gpqa_comfortable_clear": gpqa_comfortable,
        "gpqa_point_clears_only": gpqa_straddles,
        "aime_clears_90pct": a_clears_90,
        "aime_point_clears_0420": a_point_clears_gate,
        "aime_pooled_clears_0420": a_pooled_clears_gate,
        "aime_pooled_wilson_hi": a.get("pooled_wilson_hi"),
        "aime_pooled_hi_clears_gate": bool(a.get("pooled_wilson_hi", 1.0) >= AIME_GATE),
        "aime_comfortable_clear": aime_comfortable,
        "aime_survives_below_gate": aime_real_below,
    }


def build_synthesis(g: dict, a: dict) -> dict:
    v = decide_verdict(g, a)
    # GPQA-D INDETERMINATE deliverable: residual half-width + seeds needed to resolve.
    g_point = g["point_item_mean"]
    g_std = g["std_acc"]
    g_K = g["K_seeds"]
    seedmean_hw = _tcrit(g_K) * g_std / math.sqrt(g_K)          # current seed-mean half-width
    seeds_needed_seedmean = seeds_to_resolve_seedmean(g_point, g_std, GPQA_GATE)
    n_pooled_pass = g.get("n_for_pooled_wilson_pass_at_point")
    seeds_needed_pooled = (round(n_pooled_pass / g["n_items"]) if n_pooled_pass else None)
    gpqa_resolve = {
        "K_now": g_K,
        "point": g_point,
        "point_margin_above_gate": g_point - GPQA_GATE,
        "seed_mean_half_width_now": seedmean_hw,
        "seed_mean_lo_now": g["seed_mean_lo"],
        "pooled_wilson_lo_now": g["pooled_wilson_lo"],
        "seeds_needed_seedmean_lens_if_point_holds": seeds_needed_seedmean,
        "seeds_needed_pooled_wilson_lens_if_point_holds": seeds_needed_pooled,
        "point_drift_note": "0.4894@10seed(renewed) -> 0.4822@21 -> 0.4783@30; downward drift "
                            "means 'point holds' is optimistic, resolution not guaranteed",
    }
    # the {greedy, #31-sampled} x {GPQA-D, AIME} table
    table = {
        "gpqa_greedy_point": 0.4697,            # #692 same-arm greedy anchor (FAILS 0.471)
        "gpqa_greedy_clears_0471": False,
        "gpqa_sampled_point": g["point_item_mean"],
        "gpqa_sampled_pct_of_base": g["pct_of_base"],
        "gpqa_sampled_pooled_wilson_lo": g["pooled_wilson_lo"],
        "gpqa_sampled_seed_mean_lo": g["seed_mean_lo"],
        "gpqa_sampled_bootstrap_items_lo": g["bootstrap_items_lo"],
        "aime_greedy_point_same_arm": a.get("greedy_anchor_same_arm"),
        "aime_greedy_point_xarm_692": 0.3500,   # #692 g128 int4ar greedy (cross-arm)
        "aime_sampled_point": a["point_item_mean"],
        "aime_sampled_pct_of_base_greedy": a["pct_of_base_greedy"],
        "aime_sampled_pooled_wilson_lo": a["pooled_wilson_lo"],
        "aime_sampled_seed_mean_lo": a["seed_mean_lo"],
    }
    syn = {
        "pr": 696,
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # ---- headline ----
        "verdict": v["verdict"],
        # ---- PR primary + test metrics ----
        "gpqa_d_sampled_wilson_lo_30seed": g["pooled_wilson_lo"],
        "aime_sampled_pct_of_base": a["pct_of_base_greedy"],
        # ---- gate anchors ----
        "gpqa_gate_abs": GPQA_GATE, "gpqa_base_sampled_3seed": g["base_sampled_3seed"],
        "aime_gate_abs": AIME_GATE, "aime_base_greedy": AIME_BASE_GREEDY,
        "aime_base_bf16_control": a.get("base_bf16_control"),
        # ---- GPQA-D leg (all CI lenses) ----
        "gpqa_K_seeds": g["K_seeds"], "gpqa_n_items": g["n_items"],
        "gpqa_sampled_point": g["point_item_mean"], "gpqa_sampled_pct_of_base": g["pct_of_base"],
        "gpqa_pooled_wilson_lo": g["pooled_wilson_lo"],
        "gpqa_pooled_wilson_lo_pct_of_base": g["pooled_wilson_lo_pct_of_base"],
        "gpqa_seed_mean_lo": g["seed_mean_lo"], "gpqa_bootstrap_items_lo": g["bootstrap_items_lo"],
        "gpqa_wilson_n198_lo": g["wilson_n198_lo"],
        "gpqa_ci_untightenable_on_diamond_population": g["ci_untightenable_on_diamond_population"],
        "gpqa_n_for_pooled_wilson_pass_at_point": g["n_for_pooled_wilson_pass_at_point"],
        "gpqa_seeds_to_resolve": gpqa_resolve,
        # ---- AIME leg ----
        "aime_K_seeds": a["K_seeds"], "aime_n_items": a["n_items"],
        "aime_greedy_anchor_same_arm": a.get("greedy_anchor_same_arm"),
        "aime_sampled_point": a["point_item_mean"],
        "aime_sampled_pct_of_base_greedy": a["pct_of_base_greedy"],
        "aime_sampled_pct_of_base_bf16": a["pct_of_base_bf16"],
        "aime_pooled_wilson_lo": a["pooled_wilson_lo"],
        "aime_seed_mean_lo": a["seed_mean_lo"], "aime_bootstrap_items_lo": a["bootstrap_items_lo"],
        # ---- decision booleans ----
        **{k: v[k] for k in v if k != "verdict"},
        # ---- the 2x2 table (flattened) ----
        **{f"table_{k}": val for k, val in table.items()},
        "two_gate_table": table,
        "wandb_run_ids_legs": {"gpqa_aggregate": "gpqa_30seed.json",
                               "aime_aggregate": "aime_sampled.json"},
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
        print(f"[696] wandb_logging import failed: {exc}; skipping W&B.", flush=True)
        return None
    run = init_wandb_run(
        job_type="analysis",
        agent="wirbel",
        name=args.wandb_name or "wirbel/int4-quality-31basis-confirm",
        group=args.wandb_group or "int4-quality-31basis-confirm-wirbel",
        notes="PR #696: does the int4-body quantitative quality wall SURVIVE the gate-faithful "
              "#31 SAMPLED basis on BOTH gates (GPQA-D + AIME)? LOCAL analysis_only, NO FIRE.",
        tags=["int4-body", "31basis", "gpqa-diamond", "aime", "two-gate",
              "analysis_only", "no_fire"],
        config={
            "pr": 696, "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "gpqa_gate_abs": GPQA_GATE, "aime_gate_abs": AIME_GATE,
            "arm": "int4-body-isolated (g32 QAT body + bf16 262k head)",
            "protocol": "generation_config.json #31 (T=1.0 top_p=0.95 top_k=64, min_tokens=8)",
        },
    )
    if run is None:
        print("[696] W&B not initialised (no key/mode); verdict saved locally.", flush=True)
        return None
    log_summary(run, nan_clean(syn), step=0, run_prefix="")
    log_json_artifact(run, name="int4_quality_31basis_verdict", artifact_type="analysis",
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
    ap.add_argument("--gpqa", type=Path, default=HERE / "gpqa_30seed.json")
    # AIME primary leg = THINK regime (numerator protocol matched to the 0.420 = 0.90x0.4667
    # thinking-base gate). The nothink aggregate (#580 floor, gate 0.090) is folded in as a
    # protocol cross-check if present.
    ap.add_argument("--aime", type=Path, default=HERE / "aime_sampled_think.json")
    ap.add_argument("--aime-nothink", dest="aime_nothink", type=Path,
                    default=HERE / "aime_sampled_nothink.json")
    ap.add_argument("--out", type=Path, default=HERE / "joint_verdict.json")
    args = ap.parse_args(argv)

    g = json.load(open(args.gpqa))
    a = json.load(open(args.aime))
    syn = build_synthesis(g, a)
    if args.aime_nothink and args.aime_nothink.exists():
        an = json.load(open(args.aime_nothink))
        syn["aime_nothink_cross_check"] = {
            "regime": "nothink", "gate_abs": an["gate_abs"], "base_greedy": an["base_greedy"],
            "K_seeds": an["K_seeds"], "greedy_anchor_same_arm": an.get("greedy_anchor_same_arm"),
            "sampled_point": an["point_item_mean"],
            "sampled_pct_of_base": an["pct_of_base_greedy"],
            "pooled_wilson_lo": an["pooled_wilson_lo"],
            "point_clears_90pct": an.get("point_clears_90pct_greedy"),
        }
        syn["aime_nothink_sampled_point"] = an["point_item_mean"]
        syn["aime_nothink_sampled_pct_of_base"] = an["pct_of_base_greedy"]

    args.out.write_text(json.dumps(nan_clean(syn), indent=2))
    print(f"[696] joint verdict written -> {args.out}", flush=True)
    print(json.dumps({k: syn[k] for k in (
        "verdict", "gpqa_d_sampled_wilson_lo_30seed", "gpqa_sampled_point",
        "gpqa_sampled_pct_of_base", "gpqa_pooled_wilson_clears_0471",
        "gpqa_seed_mean_clears_0471", "gpqa_comfortable_clear",
        "aime_sampled_pct_of_base", "aime_sampled_point", "aime_greedy_anchor_same_arm",
        "aime_clears_90pct", "aime_survives_below_gate",
    )}, indent=2), flush=True)

    rid = log_wandb(syn, args)
    if rid:
        print(f"[696] W&B run id: {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
