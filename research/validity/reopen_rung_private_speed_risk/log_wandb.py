#!/usr/bin/env python
"""PR #522 (kanna): log reopen-rung private SPEED-drift risk to W&B.

group=reopen-rung-private-speed-risk. analysis_only=true; official_tps=0.
Reads rung_private_speed_risk_table.json (produced by
reopen_rung_private_speed_risk.py) and logs the decision-tree feed table +
key scalar outputs. No serving, no HF Job.
"""

from __future__ import annotations

import json
import os

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "reopen-rung-private-speed-risk"
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    d = json.load(open(os.path.join(HERE, "rung_private_speed_risk_table.json")))
    acc = d["acceptance_distribution"]
    fw = d["framework"]
    rows = d["rung_private_speed_risk_table"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="kanna/reopen-rung-private-speed-risk",
        job_type="analysis",
        config={
            "analysis_only": True,
            "official_tps": 0,
            "no_serve": True,
            "no_hf_job": True,
            "no_launch": True,
            "no_submission": True,
            "pr": 522,
            "boundary": d["boundary"],
            "PF": fw["PF"],
            "breach_central": fw["breach_central"],
            "sigma_hw_frac": fw["sigma_hw_frac"],
            "sigma_hw_abs_at_481_provenance_only": fw["sigma_hw_abs_at_481_provenance_only"],
            "sigma_accdraw_used": fw["sigma_accdraw_used"],
            "n_acceptance_draws": acc["n_draws"],
            "shared_drafter_sha256": "ed159e334999fd6b5f2d0dbad026346d4efac89eb7c6f55c5cdb042eca5dd18e",
            "spec_method": "mtp",
            "num_speculative_tokens": 7,
            "source_runs": fw["source_runs"],
        },
    )

    wandb.summary.update({
        # ---- KEY OUTPUTS (PR-required) ----
        "splitkv399_projected_private_tps": d["splitkv399_projected_private_tps"],
        "splitkv399_private_tps_worstcase": d["splitkv399_private_tps_worstcase"],
        "surgical357_private_tps_worstcase": d["surgical357_private_tps_worstcase"],
        "frontier457_projected_private_tps": d["frontier457_projected_private_tps"],
        "frontier457_private_tps_worstcase": d["frontier457_private_tps_worstcase"],
        "best_riskadj_rung": d["best_riskadj_rung"],
        "best_riskadj_worstcase_floor": d["best_riskadj_worstcase_floor"],
        # ---- shared multiplier (load-bearing) ----
        "shared_mult_central": rows[0]["mult_central"],
        "shared_mult_worstcase": rows[0]["mult_worst"],
        # ---- acceptance variance (measured, reused) ----
        "acc_R_ea_mean": acc["R_ea_mean"],
        "acc_R_ea_sd": acc["R_ea_sd"],
        "acc_R_ea_min": acc["R_ea_min"],
        "acc_R_ea_max": acc["R_ea_max"],
        "acc_breach_mean": acc["breach_acc_mean"],
        "acc_breach_sd_sigma_accdraw": acc["breach_acc_sd"],
        "acc_R_tps_mean": acc["R_tps_mean"],
        # ---- control reproduction / floor ----
        "floorlock_tps": d["floorlock_tps"],
        "self_test_passes": d["self_test"]["passes"],
        "nan_clean": d["self_test"]["checks"]["nan_clean"],
        "verdict_oneline": d["verdict_oneline"],
    })

    # decision-tree feed table
    cols = ["rung", "role", "public_tps", "public_anchor_kind", "loadable",
            "projected_private_tps_mean", "private_tps_band95_lo",
            "private_tps_band95_hi", "private_tps_worstcase",
            "proxy_pessimistic_private_tps_mean", "proxy_worstdraw_private_tps",
            "denken24_refuted_private_tps", "quality_verdict"]
    tbl = wandb.Table(columns=cols)
    for x in rows:
        tbl.add_data(*[x.get(c) for c in cols])
    wandb.log({"rung_private_speed_risk_table": tbl})

    # acceptance draws table (audit trail of the reused measurement)
    dcols = ["domain", "ea_pub", "ea_pri", "R_ea", "breach_acc", "tps_pub", "tps_pri", "R_tps"]
    dtbl = wandb.Table(columns=dcols)
    for dr in d["acceptance_draws"]:
        dtbl.add_data(*[dr.get(c) for c in dcols])
    wandb.log({"acceptance_draws": dtbl})

    print("W&B run:", run.url)
    print("run id:", run.id)
    run.finish()


if __name__ == "__main__":
    main()
