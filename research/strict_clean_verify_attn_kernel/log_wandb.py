#!/usr/bin/env python
"""PR #747 wirbel — log the verify-attention M=1 byte-exactness microbench to
W&B group strict-clean-verify-attn-kernel-wirbel. Two runs: BI=1 (the PR's
reference stack) and BI=0 (the served-manifest config). ANALYSIS ONLY."""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean

import wandb

HERE = Path(__file__).resolve().parent
ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "strict-clean-verify-attn-kernel-wirbel"

RUNS = [
    ("bi1", "verify_attn_report.json", "wirbel/verify-attn-m1-byteexact-bi1"),
    ("bi0", "verify_attn_report_bi0.json", "wirbel/verify-attn-m1-byteexact-bi0"),
]


def per_verify_wall_ms(d, K=6):
    """Faithful forced-M=1 per-verify wall (ms), mean over timed layers at K."""
    ts = [t for t in d["timing"] if t["K"] == K]
    return mean(t["c_ms"] for t in ts) if ts else float("nan")


def batched_wall_ms(d, K=6):
    ts = [t for t in d["timing"] if t["K"] == K]
    return mean(t["a_ms"] for t in ts) if ts else float("nan")


def main():
    ids = []
    for tag, fname, rname in RUNS:
        d = json.loads((HERE / fname).read_text())
        bi = int(d["batch_invariant"])
        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP, name=rname,
            job_type="analysis", reinit=True,
            config={
                "pr": 747, "lane": "strict-clean-verify-attn-kernel",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "batch_invariant": bi, "backend": d["backend"],
                "device": d["device"], "capability": d["capability"],
                "ks": d["ks"], "n_rows": d["n_rows"],
                "n_layers_captured": d["n_layers_captured"],
                "model": "int4_g128_lmhead", "head_dims": "256 sliding / 512 global",
            },
        )
        ids.append(run.id)
        twall = per_verify_wall_ms(d)
        awall = batched_wall_ms(d)
        summary = {
            # primary route-b metric: faithful forced-M=1 (real decode path)
            "verify_attn_forced_m1_max_bitdiff": d["verify_attn_forced_m1_max_bitdiff"],
            # a 2D-only verify kernel (what NOT to build)
            "verify_attn_forced_m1_2d_max_bitdiff": d["verify_attn_forced_m1_2d_max_bitdiff"],
            # batched M=K verify (the deployed spec path; reproduces land #680)
            "verify_attn_batched_max_bitdiff": d["verify_attn_batched_max_bitdiff"],
            # pure query-batching effect (M=K vs M=1, same 2D path)
            "verify_attn_batching_effect_max_bitdiff": d["verify_attn_batching_effect_max_bitdiff"],
            "sanity_c_vs_ref_max_bitdiff": d["sanity_c_vs_ref_max_bitdiff"],
            "verify_attn_per_verify_wall_ms": twall,      # test_metric (K=6, faithful)
            "verify_attn_batched_wall_ms": awall,
            "verify_attn_m1_tax_ratio": twall / awall if awall else float("nan"),
            "verdict": d["verdict"],
        }
        # per-(K, ltype) detail tables
        kt = wandb.Table(columns=["K", "a_batched_max", "b_m1_2d_max",
                                  "c_m1_faithful_max", "ab_batching_max",
                                  "a_maxd", "c_maxd"])
        for K in d["ks"]:
            pk = d["per_k"][str(K)]
            kt.add_data(K, pk["a_batched_vs_ar"]["max_nbit"],
                        pk["b_forcedm1_2d_vs_ar"]["max_nbit"],
                        pk["c_forcedm1_faithful_vs_ar"]["max_nbit"],
                        pk["ab_batching_effect"]["max_nbit"],
                        pk["a_batched_vs_ar"]["max_maxd"],
                        pk["c_forcedm1_faithful_vs_ar"]["max_maxd"])
        tt = wandb.Table(columns=["layer", "ltype", "head_size", "K",
                                  "a_batched_us", "c_forcedm1_us", "ratio"])
        for t in d["timing"]:
            tt.add_data(t["layer"], t["ltype"], t["head_size"], t["K"],
                        round(t["a_ms"] * 1000, 2), round(t["c_ms"] * 1000, 2),
                        round(t["c_ms"] / t["a_ms"], 3))
        run.log({"per_k": kt, "timing": tt, **summary})
        run.summary.update(summary)
        print(f"[wandb] {tag}: run {run.id} verdict={d['verdict']} "
              f"forced_m1={summary['verify_attn_forced_m1_max_bitdiff']} "
              f"batched={summary['verify_attn_batched_max_bitdiff']} "
              f"wall_ms={twall:.3f}")
        run.finish()
    print("WANDB_RUN_IDS=" + ",".join(ids))


if __name__ == "__main__":
    raise SystemExit(main())
