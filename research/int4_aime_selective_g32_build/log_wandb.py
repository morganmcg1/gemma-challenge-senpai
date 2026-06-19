#!/usr/bin/env python3
"""PR #702 W&B logger: push the selective-g32 AIME verdict to the
`int4-aime-selective-g32-build-ubel` group.

Reads selective_g32_summary.json (aggregate.py output) and logs:
  * the explicit guard scalars analysis_only=1, official_tps=0, no_hf_job=1, fires=0;
  * the per-arm AIME table {full_g128, selective, full_g32} with pooled accuracy,
    n_correct/n, 5-seed-pooled Wilson95 CIs, and per-seed [min,mean,max];
  * PRIMARY metric selective_g32_aime_compliant and TEST metric
    selective_recovery_fraction;
  * the verdict string + note + control-validity flags;
  * the 48-module activation-critical subset manifest as an artifact.

LOCAL analysis only. Entity/project/key come from the env (WANDB_ENTITY,
WANDB_PROJECT, WANDB_API_KEY already exported in this container).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
GROUP = "int4-aime-selective-g32-build-ubel"
ARMS = ["full_g128", "selective", "full_g32"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(HERE / "selective_g32_summary.json"))
    ap.add_argument("--manifest", default=str(HERE / "subset48_manifest.json"))
    ap.add_argument("--name", default="ubel/selective-g32-build")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + print the payload but do not call wandb.init")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    arms = s.get("arms", {})

    config = {
        "analysis_only": s.get("analysis_only", 1),
        "official_tps": s.get("official_tps", 0),
        "no_hf_job": s.get("no_hf_job", 1),
        "fires": s.get("fires", 0),
        "pr": 702,
        "bar": s.get("bar"),
        "bf16_base": s.get("bf16_base"),
        "g128_ref": s.get("g128_ref"),
        "g32_ref": s.get("g32_ref"),
        "protocol": s.get("protocol"),
        "subset_source_run": "vjhzcvmu",   # #700 activation-localization
        "subset_n_modules": 48,
        "subset_f_param": 0.013526888815572418,
        "byte_law_tps_projected": 126.275,  # #700 byte-law, NOT measured here
    }

    # flat summary scalars (the verdict-bearing numbers)
    flat = {
        "selective_g32_aime_compliant": s.get("selective_g32_aime_compliant"),
        "selective_wilson_hi": s.get("selective_wilson_hi"),
        "selective_recovery_fraction": s.get("selective_recovery_fraction"),
        # int4-Marlin-scale re-projection (the scale the 0.420 gate actually lives on)
        "selective_int4scale_projection": s.get("selective_int4scale_projection"),
        "selective_int4scale_clears_gate": s.get("selective_int4scale_clears_gate"),
        "rf_clear_threshold": s.get("rf_clear_threshold"),
        "verdict": s.get("verdict"),
        "verdict_note": s.get("verdict_note"),
        "analysis_only": s.get("analysis_only", 1),
        "official_tps": s.get("official_tps", 0),
        "no_hf_job": s.get("no_hf_job", 1),
        "fires": s.get("fires", 0),
    }
    # propagated recovery_fraction CI + probabilistic clearing (07:24 PR-comment commitment)
    mc = s.get("recovery_fraction_mc") or {}
    for src, dst in (
        ("rf_median", "recovery_fraction_mc_median"),
        ("rf_ci_lo", "recovery_fraction_mc_lo"),
        ("rf_ci_hi", "recovery_fraction_mc_hi"),
        ("proj_median", "int4scale_proj_mc_median"),
        ("proj_ci_lo", "int4scale_proj_mc_lo"),
        ("proj_ci_hi", "int4scale_proj_mc_hi"),
        ("p_projection_clears_gate", "p_projection_clears_gate"),
        ("p_control_separation", "p_control_separation"),
        ("separation_median", "control_separation_median"),
    ):
        if src in mc:
            flat[dst] = mc[src]
    for a in ARMS:
        r = arms.get(a)
        if not r:
            continue
        flat[f"{a}_pooled_acc"] = r.get("pooled_accuracy")
        flat[f"{a}_wilson_lo"] = r.get("wilson95_lo")
        flat[f"{a}_wilson_hi"] = r.get("wilson95_hi")
        flat[f"{a}_n_correct"] = r.get("pooled_n_correct")
        flat[f"{a}_n"] = r.get("pooled_n")
        flat[f"{a}_perseed_mean"] = r.get("per_seed_acc_mean")
    for k, v in (s.get("controls") or {}).items():
        flat[f"control_{k}"] = v

    print("=== config ===")
    print(json.dumps(config, indent=2, default=str))
    print("=== summary scalars ===")
    print(json.dumps(flat, indent=2, default=str))

    if args.dry_run:
        print("[dry-run] not logging to wandb")
        return

    # Resume the proof-of-life run (live_log.py) so the verdict lands in the same
    # run the advisor is already watching, rather than spawning a second run.
    rid_file = HERE / "_wandb_run_id.txt"
    rid = rid_file.read_text().strip() if rid_file.exists() else None
    run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                     id=rid, resume="allow", group=GROUP, name=args.name,
                     job_type="analysis", config=config,
                     tags=["pr702", "int4-aime", "selective-g32",
                           "analysis_only", "quality-recovery"])

    # per-arm AIME table
    cols = ["arm", "n_seeds", "pooled_acc", "n_correct", "n",
            "wilson95_lo", "wilson95_hi", "perseed_min", "perseed_mean", "perseed_max"]
    tbl = wandb.Table(columns=cols)
    for a in ARMS:
        r = arms.get(a)
        if not r:
            continue
        tbl.add_data(a, r.get("n_seeds"), r.get("pooled_accuracy"),
                     r.get("pooled_n_correct"), r.get("pooled_n"),
                     r.get("wilson95_lo"), r.get("wilson95_hi"),
                     r.get("per_seed_acc_min"), r.get("per_seed_acc_mean"),
                     r.get("per_seed_acc_max"))
    run.log({"aime_arm_table": tbl})

    # verdict + numbers into run.summary so they show in the runs table
    clean = {k: v for k, v in flat.items() if not (isinstance(v, float) and math.isnan(v))}
    run.summary.update(clean)

    # artifact: the 48-module subset manifest + the pooled summary
    art = wandb.Artifact("selective_g32_subset48", type="manifest",
                         metadata={"pr": 702, "source_run": "vjhzcvmu",
                                   "verdict": s.get("verdict")})
    if Path(args.manifest).exists():
        art.add_file(args.manifest)
    if Path(args.summary).exists():
        art.add_file(args.summary)
    run.log_artifact(art)

    print(f"[wandb] logged run {run.id} group={GROUP} verdict={s.get('verdict')}")
    run.finish()


if __name__ == "__main__":
    main()
