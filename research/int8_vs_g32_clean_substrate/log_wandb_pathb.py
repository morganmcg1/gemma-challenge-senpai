#!/usr/bin/env python3
"""PR #726 W&B logger: push the Path-B clean-substrate int8-vs-g32 verdict to the
`ubel-int8-vs-g32-clean` group.

Reads pathb_summary.json (aggregate_pathb.py) + xcheck_g32.json (substrate proof)
and logs the guard scalars, per-arm AIME table, the PRIMARY paired delta + verdict,
the ceiling delta, the substrate gate, and the weight-space cross-check. LOCAL
analysis only -- no HF Job, no submission. Entity/project from env or defaults.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
GROUP = "ubel-int8-vs-g32-clean"
ARMS = ["full_g32", "int8_locus", "bf16_locus", "int8_full"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(HERE / "pathb_summary.json"))
    ap.add_argument("--xcheck", default=str(HERE / "xcheck_g32.json"))
    ap.add_argument("--name", default="ubel/int8-vs-g32-clean-substrate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    arms = s.get("arms", {})
    xc = json.load(open(args.xcheck)) if Path(args.xcheck).exists() else {}
    prim = s.get("primary_int8_locus_minus_g32") or {}
    ceil = s.get("secondary_int8_minus_bf16_locus") or {}
    gate = s.get("substrate_gate") or {}

    config = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "pr": 726, "path": "B", "substrate_master": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
        "locus": s.get("locus"), "protocol": s.get("protocol"), "g32_anchor": s.get("g32_anchor"),
        "decode": "sampled T=1.0 top_p=0.95 top_k=64 (lewtun #31 gate-relevant regime)",
    }

    flat = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # PRIMARY / TEST
        "int8_locus_minus_g32_clean": prim.get("delta"),
        "int8_edge_replicates": int(prim.get("replicate", 0)),
        "primary_verdict": prim.get("verdict"),
        "primary_mcnemar_p_exact": prim.get("p_exact"),
        "primary_mcnemar_p_chi2cc": prim.get("p_chi2_cc"),
        "primary_discordant_b_int8win": prim.get("b"),
        "primary_discordant_c_g32win": prim.get("c"),
        # SECONDARY ceiling
        "int8_minus_bf16_locus": ceil.get("delta"),
        "ceiling_noise_signature": int(ceil.get("noise_signature", 0)) if ceil else None,
        "ceiling_mcnemar_p_exact": ceil.get("p_exact"),
        # GATE
        "substrate_gate_pass": int(gate.get("pass", 0)) if gate else None,
        "full_g32_pooled": gate.get("full_g32_pooled") if gate else None,
        "full_g32_anchor_in_ci": int(gate.get("anchor_in_ci", 0)) if gate else None,
        # cross-check
        "xcheck_substrate_is_master": int(xc.get("substrate_is_master", 0)) if xc else None,
        "xcheck_rel_err_max": (xc.get("rel_err_all") or {}).get("max") if xc else None,
        "xcheck_rel_err_mean": (xc.get("rel_err_all") or {}).get("mean") if xc else None,
    }
    if "extend_int8_full_minus_g32" in s:
        flat["int8_full_minus_g32_clean"] = s["extend_int8_full_minus_g32"].get("delta")
        flat["extend_mcnemar_p_exact"] = s["extend_int8_full_minus_g32"].get("p_exact")
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

    print("=== config ==="); print(json.dumps(config, indent=2, default=str))
    print("=== summary scalars ==="); print(json.dumps(flat, indent=2, default=str))
    if args.dry_run:
        print("[dry-run] not logging to wandb")
        return

    run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                     group=GROUP, name=args.name, job_type="analysis", config=config,
                     tags=["pr726", "int4-aime", "int8-locus", "clean-substrate",
                           "analysis_only", "paired-mcnemar"])

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
                     r.get("per_seed_acc_min"), r.get("per_seed_acc_mean"), r.get("per_seed_acc_max"))
    run.log({"aime_arm_table": tbl})

    clean = {k: v for k, v in flat.items()
             if v is not None and not (isinstance(v, float) and math.isnan(v))}
    run.summary.update(clean)

    art = wandb.Artifact("pathb_int8_vs_g32_clean", type="analysis",
                         metadata={"pr": 726, "verdict": prim.get("verdict"),
                                   "substrate_is_master": xc.get("substrate_is_master")})
    for f in (args.summary, args.xcheck):
        if Path(f).exists():
            art.add_file(f)
    run.log_artifact(art)

    (HERE / "_wandb_run_id.txt").write_text(run.id)
    print(f"[wandb] logged run {run.id} group={GROUP} verdict={prim.get('verdict')}")
    run.finish()


if __name__ == "__main__":
    main()
