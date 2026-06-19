#!/usr/bin/env python3
"""Log the PR #700 activation-weighted output-impact localization to W&B.

analysis_only=true, official_tps=0, no_hf_job=1, fires=0 (explicit scalars).
Logs: the activation-weighted per-module ranking (Table + artifact), the impact
localization Pareto curve, side-by-side weight-vs-activation localization stats,
all summary scalars incl. the verdict + validity caveat, and the full JSON
artifacts.  Group: int4-aime-activation-localization-ubel.
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

HERE = Path(__file__).parent
IL = json.load(open(HERE / "impact_localization.json"))
ACT = json.load(open(HERE / "act_norms.json"))
S = IL["summary"]
ROWS = IL["rows"]
CURVE = IL["stats"]["impact"]["_curve"]

TPS_FLOOR = 126.378
BYTE_K = 0.06005


def main():
    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="gemma-challenge-senpai",
        group="int4-aime-activation-localization-ubel",
        name="ubel/activation-impact-localization",
        job_type="analysis",
        config={
            "card": "PR-700 int4-AIME recipe terminator: activation-weighted localization",
            "step": "instr_1-3_activation_norm_probe + impact rerank + byte-law",
            "analysis_only": True,
            "official_tps": 0,
            "no_hf_job": 1,
            "gpu_used": True,
            "gpu": "A10G (local pod), CUDA_VISIBLE_DEVICES override 4->0",
            "impact_definition": "impact = rel_div(#695 weight-error) x act_norm(AIME input L2)",
            "act_checkpoint": ACT["meta"]["checkpoint"],
            "act_calib": f"AIME-2024 x{ACT['meta']['n_prompts']} (prefill) + "
                         f"decode-subset x{ACT['meta']['decode_subset']} x{ACT['meta']['decode_tokens']}tok",
            "rel_div_source": "#695 sqnr_probe.json (34sd7dod), reused verbatim",
            "byte_law": "TPS(f)=126.378/(1+0.06005*f); f=body-param-fraction on g32",
            "byte_law_note": "TPS(f)<=126.378 for all f>=0 (g32 finer => more scale bytes); "
                             "speed-free REVIVE => clearing footprint cost within anchor noise",
            "tps_floor_anchor": TPS_FLOOR,
            "byte_k": BYTE_K,
            "clearing_coverage_assumption": S["clearing_coverage_assumption"],
            "aime_g128": 0.350, "aime_g32_uniform": 0.438, "aime_bar": 0.420, "aime_base": 0.4667,
            "n_body_modules": len(ROWS),
        },
    )

    # ---- activation-weighted per-module ranking (the card's required Table) ----
    cols = ["module", "layer", "proj", "params", "in_dim", "rel_div", "act_norm",
            "act_norm_decode", "impact", "abs_out", "phys_out"]
    tbl = wandb.Table(columns=cols)
    for r in sorted(ROWS, key=lambda r: r["impact"], reverse=True):
        tbl.add_data(*[r[c] for c in cols])
    run.log({"activation_weighted_per_module": tbl})

    # ---- impact localization Pareto curve (greedy impact-energy-per-param) ----
    ccols = ["rank", "f_param", "f_impact_energy", "tps_proj", "module"]
    ctbl = wandb.Table(columns=ccols)
    for i, c in enumerate(CURVE):
        ctbl.add_data(i + 1, c["f_param"], c["f_energy"], c["tps_proj"], c["module"])
    run.log({"impact_localization_pareto_curve": ctbl})
    xs = [c["f_param"] for c in CURVE]
    ys = [c["f_energy"] for c in CURVE]
    run.log({"impact_loc_curve_plot": wandb.plot.line_series(
        xs=[xs, xs], ys=[ys, xs],
        keys=["activation_weighted_impact_energy_removed", "diffuse_baseline (y=x)"],
        title="Activation-weighted localization: impact-energy removed vs g32 footprint",
        xname="f_param (g32 footprint = speed cost)")})

    # ---- side-by-side weight-space (#695) vs activation-weighted (#700) ----
    scols = ["axis", "top16_energy_share", "energy_gini_like", "cv", "clearing_tps"]
    stbl = wandb.Table(columns=scols)
    stbl.add_data("weight-space diff_norm (#695)", S["top16_energy_share_695_diffnorm"],
                  S["energy_gini_like_695_diffnorm"], S["rel_div_cv_695"], S["tps_at_50pct_695"])
    stbl.add_data("weight-space rel_div (no act)", S["reldiv_top16_energy_share"],
                  S["reldiv_energy_gini_like"], S["reldiv_cv"], None)
    stbl.add_data("activation-weighted impact (#700)", S["impact_top16_energy_share"],
                  S["impact_energy_gini_like"], S["impact_cv"], S["activation_critical_tps_at_clearing_footprint"])
    stbl.add_data("abs_out diff_norm x act", S["absout_top16_energy_share"],
                  S["absout_energy_gini_like"], None, S["absout_tps_at_clearing"])
    stbl.add_data("phys_out (/sqrt dim)", S["physout_top16_energy_share"],
                  S["physout_energy_gini_like"], None, S["physout_tps_at_clearing"])
    stbl.add_data("activation impact ex-PLE", S["exclude_ple_top16_energy_share"],
                  S["exclude_ple_energy_gini_like"], None, S["exclude_ple_clearing_tps"])
    run.log({"weight_vs_activation_localization": stbl})

    # ---- summary scalars ----
    skip = {"top16_modules_by_impact", "clearing_subset_composition"}
    for k, v in S.items():
        if k not in skip:
            run.summary[k] = v
    run.summary["clearing_subset_composition"] = json.dumps(S["clearing_subset_composition"])

    # ---- artifacts ----
    art = wandb.Artifact(
        "activation_impact_localization", type="analysis",
        description="PR#700 activation-weighted output-impact localization (343 body "
                    "modules): per-module impact=rel_div x act_norm ranking, localization "
                    "stats, byte-law footprint->TPS, prefill/decode act-norms.")
    for f in ("impact_localization.json", "act_norms.json", "impact_localization.py",
              "activation_probe.py"):
        art.add_file(str(HERE / f))
    run.log_artifact(art)

    print("WANDB_RUN_ID", run.id)
    print("WANDB_RUN_URL", run.url)
    print("VERDICT", S["verdict"])
    print("PRIMARY activation_weighted_top16_energy_share", S["activation_weighted_top16_energy_share"])
    print("TEST activation_critical_tps_at_clearing_footprint", S["activation_critical_tps_at_clearing_footprint"])
    run.finish()


if __name__ == "__main__":
    main()
