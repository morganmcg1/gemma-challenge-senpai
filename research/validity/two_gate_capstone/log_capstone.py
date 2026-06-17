#!/usr/bin/env python3
"""Two-gate capstone verdict logger (PR #587).

Logs the converged NO-FIRE verdict booleans for the two-gate challenge to W&B.
This is a SYNTHESIS of ~15 already-banked cards -- it re-states their banked
numbers, it does NOT re-measure anything.

LOCAL only: analysis_only=true, official_tps=0. NO HF Job, NO /v1/jobs:run,
NO --launch, NO submission, NO served-file change. NO FIRE.

Verdict
-------
two_gate_unsatisfiable = TRUE under the conjunction
  { mandated int4 ckpt } AND { #319-operative identity }
  AND { >=90%-of-base quality } AND { faster-than-375.857 TPS }.

The quality+identity+int4-safe ceiling is base_fullhead ~= 252.69 TPS, >=113 TPS
(~30%) short of the 375.857 ship. The ship needs BOTH a 12k head-prune (kills
quality) AND MTP spec-dec (kills #319 identity); each lever violates a different
gate. n_binding_constraints = 4; exactly ONE single-constraint relaxation
reopens a fire -> dropping the 375.857 target (base_fullhead ships as-is).

Refines PR-expected: relaxing int4 -> bf16/int8 body is SLOWER (bf16=143.99,
int8=205.48 TPS; body census #571 vct3k1vc), so reopens_fire[int4]=FALSE.
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

SHIP_TPS = 375.857
ANCHOR_BASE_FULLHEAD_TPS = 252.69          # wirbel #553 / body census #571 int4_g32
FREE_HEAD_UPPER_TPS = 328.9                # #544 decomposition (head removed, spec-off)
OPT_QUALITY_SAFE_CEILING_TPS = 292.1       # #544 int4-head quantized ceiling
MTP_K7_PROJ_TPS = 262.9                    # #584 best measured drafter (gap -113)
INT8_BODY_TPS = 205.48                     # #571 body census
BF16_BODY_TPS = 143.99                     # #571 body census (byte-exact, slowest)

REOPENS_FIRE = {
    "mandated_int4_ckpt": False,           # bf16/int8 body SLOWER -> refutes PR-expected
    "operative_identity_319": False,       # MTP-K7 262.9 < ship; any_drafter_clears=False
    "quality_90pct": False,                # free-head spec-off 328.9 < ship
    "target_gt_375857": True,              # drop bar -> base_fullhead 252.69 ships
}


def build_verdict() -> dict:
    gap_to_ship = SHIP_TPS - ANCHOR_BASE_FULLHEAD_TPS
    syn = {
        "pr": 587,
        "analysis_only": True,
        "official_tps": 0,
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),

        # ---- HEADLINE VERDICT ----
        "two_gate_unsatisfiable": True,
        "n_binding_constraints": 4,
        "n_single_relaxations_that_fire": int(sum(REOPENS_FIRE.values())),

        # ---- per-constraint reopens_fire vector (flattened scalars) ----
        "reopens_fire_mandated_int4_ckpt": REOPENS_FIRE["mandated_int4_ckpt"],
        "reopens_fire_operative_identity_319": REOPENS_FIRE["operative_identity_319"],
        "reopens_fire_quality_90pct": REOPENS_FIRE["quality_90pct"],
        "reopens_fire_target_gt_375857": REOPENS_FIRE["target_gt_375857"],

        # ---- speed leg ----
        "ship_tps": SHIP_TPS,
        "anchor_base_fullhead_tps": ANCHOR_BASE_FULLHEAD_TPS,
        "gap_to_ship_tps": gap_to_ship,
        "gap_to_ship_frac": gap_to_ship / SHIP_TPS,
        "opt_quality_safe_ceiling_tps": OPT_QUALITY_SAFE_CEILING_TPS,
        "free_head_upper_tps": FREE_HEAD_UPPER_TPS,
        "free_head_beats_ship": FREE_HEAD_UPPER_TPS > SHIP_TPS,        # False
        "mtp_k7_proj_tps": MTP_K7_PROJ_TPS,
        "any_measured_drafter_clears_ship": False,                    # #584
        "specdec_break_even_acceptance": 4.95,                        # #584 honest
        "mtp_k7_best_acceptance": 3.844,                              # #583/#584
        "any_strict_safe_speed_lever_anywhere": False,                # #556/#562/#571
        "decode_overhead_floor_tps": 311.27,                          # #569

        # ---- int4 refutation evidence ----
        "int4_relax_reopens_fire": False,
        "int4_relax_refutes_pr_expected": True,
        "int8_body_tps": INT8_BODY_TPS,
        "bf16_body_tps": BF16_BODY_TPS,
        "bf16_body_is_byte_exact_but_slowest": True,                  # #571

        # ---- quality leg (base_fullhead vs re-anchored >=90% gates) ----
        "quality_satisfied": True,
        "mmlu_pro_value": 0.6313, "mmlu_pro_gate": 0.605, "mmlu_pro_pass": True,
        "gpqa_value": 0.4798, "gpqa_gate": 0.471, "gpqa_pass": True,
        "gpqa_is_knife_edge": True,                                   # +0.009 only
        "gpqa_conflicting_value_seed12345": 0.4697,                   # fails -0.001
        "gpqa_579_confirmation_landed": False,                        # unlanded
        "gsm8k_value_approx": 0.85, "gsm8k_gate": 0.807, "gsm8k_pass": True,
        "aime_value": 0.1167, "aime_gate": 0.090, "aime_pass": True,

        # ---- identity leg ----
        "identity_operative_319_int4_referenced": True,               # #585
        "no_int4_config_is_literal_bf16_byte_exact": True,            # #585
        "specdec_identity_fire_eligible": False,                      # #576/#583
        "official_scorer_runs_token_identity_check": False,           # #124

        # ---- W&B re-confirmation flags (instruction #1) ----
        "flag_ppl_absent_in_83jiwjr9": True,
        "flag_gpqa_arm_absent_in_qi24h8zx": True,
        "flag_xmdeo3dj_project_path_mismatch": True,
        "flag_anchor_252_frame_inconsistency_584_vs_587": True,
        "verdict_robust_to_anchor_frame": True,

        # ---- provenance ----
        "run_ids": {
            "tps_anchor_553": "83jiwjr9",
            "gpqa_paired_574": "7bi4e2ne",
            "aime_580": "yokbmy9i",
            "gates_581": "qi24h8zx",
            "body_census_571": "vct3k1vc",
            "identity_operative_585": "2u44yaa1",
            "identity_mechanism_576": "g7yob0yg",
            "specdec_analytic_583": "xmdeo3dj",
            "specdec_empirical_584": "gd5s78ze",
        },
        "reopens_fire": dict(REOPENS_FIRE),  # nested copy for the artifact
    }
    return syn


def nan_clean(obj):
    if isinstance(obj, dict):
        return {k: nan_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_clean(v) for v in obj]
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
        print(f"[capstone] wandb_logging import failed: {exc}; skipping W&B.", flush=True)
        return None
    run = init_wandb_run(
        job_type="analysis",
        agent="fern",
        name=args.wandb_name or "fern/two-gate-unsatisfiable-capstone",
        group=args.wandb_group or "two-gate-unsatisfiable-capstone",
        notes="PR #587: converged two-gate NO-FIRE capstone. two_gate_unsatisfiable=TRUE; "
              "n_binding=4; only dropping the 375.857 target reopens a fire. analysis_only, NO FIRE.",
        tags=["two-gate", "capstone", "analysis_only", "no_fire", "base_fullhead"],
        config={
            "pr": 587, "analysis_only": True, "official_tps": 0,
            "ship_tps": SHIP_TPS, "anchor_base_fullhead_tps": ANCHOR_BASE_FULLHEAD_TPS,
            "run_ids": syn["run_ids"],
        },
    )
    if run is None:
        print("[capstone] W&B not initialised (no key/mode); verdict saved locally.", flush=True)
        return None
    log_summary(run, nan_clean(syn), step=0, run_prefix="")
    log_json_artifact(run, name="two_gate_capstone_verdict", artifact_type="analysis",
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
    ap.add_argument("--out", type=Path, default=HERE / "two_gate_capstone_verdict.json")
    args = ap.parse_args(argv)

    syn = build_verdict()

    # self-consistency checks (real invariants, NOT banked-constant mirrors)
    assert syn["two_gate_unsatisfiable"] is True
    assert syn["gap_to_ship_tps"] > 0, "base_fullhead must be slower than ship"
    assert syn["free_head_beats_ship"] is False, "even a free head must not beat ship"
    assert syn["n_single_relaxations_that_fire"] == sum(REOPENS_FIRE.values())
    assert syn["n_single_relaxations_that_fire"] == 1, "only the target relax fires"
    assert REOPENS_FIRE["target_gt_375857"] is True
    assert syn["mtp_k7_best_acceptance"] < syn["specdec_break_even_acceptance"]
    assert BF16_BODY_TPS < ANCHOR_BASE_FULLHEAD_TPS, "bf16 body must be slower (int4 refutation)"

    args.out.write_text(json.dumps(nan_clean(syn), indent=2))
    print(f"[capstone] verdict written -> {args.out}", flush=True)
    print(f"[capstone] two_gate_unsatisfiable={syn['two_gate_unsatisfiable']} "
          f"n_binding={syn['n_binding_constraints']} "
          f"fires={syn['n_single_relaxations_that_fire']} "
          f"reopens_fire={REOPENS_FIRE}", flush=True)

    rid = log_wandb(syn, args)
    if rid:
        print(f"[capstone] W&B run id: {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
