#!/usr/bin/env python3
"""Log the #822 AdaEDL early-stop study to W&B (rich record + citable run id).

Run under SYSTEM python3 from a neutral cwd (e.g. /tmp) so the local target/wandb/
data dir does not shadow the wandb package (memory: wandb_logging_shadow).
All numbers are LOCAL A10G exploratory proxies (single-stream HTTP-serial decode),
NOT the official a10g-small leaderboard metric.
"""
import json
import sys

import wandb

SUMMARY = "/workspace/senpai/target/research/adaedl_822/summary.json"


def main():
    s = json.load(open(SUMMARY))
    tps = s["tps_table"]
    prem = s["premise_step1"]
    cf = s["counterfactual_offline"]
    gi = s["greedy_identity_noise_floor_relative"]

    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="gemma-challenge-senpai",
        name="land/adaedl-earlystop-822",
        group="adaedl_822",
        job_type="local_serve_study",
        config={
            "submission": "int4_mtp_bi0_int4head",
            "hypothesis": "AdaEDL entropy-gated draft early-stop (arXiv:2410.18351)",
            "K_num_speculative_tokens": prem["K"],
            "decode": s["decode_config"],
            "local_exploratory": True,
            "hardware_note": s["hardware"],
            "thresholds_swept": [2.477, 1.449, 0.727, 0.402],
        },
    )

    # headline scalars
    wandb.summary["ppl"] = s["ppl"]["value"]
    wandb.summary["ppl_n_records"] = s["ppl"]["n_records"]
    wandb.summary["E_accept_control"] = prem["E_accept"]
    wandb.summary["entropy_separation_reject_minus_accept"] = prem["entropy_separation"]
    wandb.summary["premise_holds"] = prem["premise_holds"]
    wandb.summary["tps_unpatched_ship"] = tps["unpatched_ship"]["tps_mean"]
    wandb.summary["tps_patched_inf_control"] = tps["patched_inf_life2"]["tps_mean"]
    wandb.summary["tps_best_tau"] = s["best_tau"]["tps_mean"]
    wandb.summary["best_tau_thresh"] = s["best_tau"]["thresh"]
    wandb.summary["best_tau_pct_vs_ship"] = s["best_tau"]["pct_vs_ship"]
    wandb.summary["machinery_tax_pct_vs_ship"] = tps["patched_inf_life2"]["pct_vs_ship"]
    wandb.summary["verdict"] = s["verdict"]

    # TPS sweep table
    t = wandb.Table(columns=["config", "thresh", "tps_mean", "tps_std", "pct_vs_ship", "pct_vs_patched_inf"])
    for k in ["unpatched_ship", "patched_inf_life2", "tau2477", "tau1449", "tau0727", "tau0402"]:
        e = tps[k]
        t.add_data(k, str(e.get("thresh")), e["tps_mean"], e["tps_std"],
                   e.get("pct_vs_ship", 0.0), e.get("pct_vs_patched_inf", float("nan")))
    wandb.log({"tps_sweep": t})

    # per-position accept rate
    pp = wandb.Table(columns=["position", "accept_rate"])
    for j, r in sorted(prem["per_position_accept_rate"].items(), key=lambda kv: int(kv[0])):
        pp.add_data(int(j), r)
    wandb.log({"per_position_accept_rate": pp})

    # offline counterfactual
    ct = wandb.Table(columns=["tau", "fwd_per_step", "fwd_saved", "real_acc", "acc_loss_pct", "stop_rate"])
    for tau, row in sorted(cf.items(), key=lambda kv: -float(kv[0])):
        ct.add_data(float(tau), row["fwd_per_step"], row["fwd_saved"], row["real_acc"],
                    row["acc_loss_pct"], row["stop_rate"])
    wandb.log({"offline_counterfactual": ct})

    # greedy identity noise-floor
    gt = wandb.Table(columns=["comparison", "positionwise_agree_pct", "exact", "prompts"])
    for k, v in gi.items():
        gt.add_data(k, v["positionwise_agree_pct"], v["exact"], v["prompts"])
    wandb.log({"greedy_identity_noise_floor": gt})

    rid = run.id
    url = run.url
    run.finish()
    print(f"WANDB_RUN_ID={rid}")
    print(f"WANDB_URL={url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
