#!/usr/bin/env python3
"""Log the PR #695 per-layer SQNR localization probe to W&B.

analysis_only=true, official_tps=0, no_hf_job=1 (explicit scalars).
Logs: per-layer sensitivity ranking (Table), the energy-localization curve
(Table, the speed/quality Pareto SHAPE), all localization summary scalars, and
the full probe JSON as an artifact.  Group: int4-selective-grid-aime-ubel.
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

HERE = Path(__file__).parent
PROBE = json.load(open(HERE / "sqnr_probe.json"))
S = PROBE["summary"]
ROWS = PROBE["rows"]
CURVE = PROBE["loc_curve"]

TPS_FLOOR = 126.378
BYTE_K = 0.06005


def tps_at(f):
    return TPS_FLOOR / (1.0 + BYTE_K * f)


def main():
    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="gemma-challenge-senpai",
        group="int4-selective-grid-aime-ubel",
        name="ubel/sqnr-localization-probe",
        job_type="analysis",
        config={
            "card": "PR-695 selective mixed-grid int4",
            "step": "instruction_1_per_layer_sqnr_localization_probe",
            "analysis_only": True,
            "official_tps": 0,
            "no_hf_job": 1,
            "basis": "weight-space SQNR (basis-INDEPENDENT; not a decode basis)",
            "method": "dequant(official g32 qat-w4a16-ct) vs dequant(int4_g128 anchor), "
                      "per-module relative L2 = g128 excess-error over QAT-native g32",
            "g32_ref": "google/gemma-4-E4B-it-qat-w4a16-ct (group_size=32, QAT-native)",
            "g128_anchor": "int4_g128_lmhead PR#4 (group_size=128, byte-floor, 126.378)",
            "byte_law": "TPS(f)=126.378/(1+0.06005*f); f=body-param-fraction on g32",
            "tps_floor_anchor": TPS_FLOOR,
            "byte_k": BYTE_K,
            "aime_bar": 0.420,
            "gpu_used": False,
            "n_body_modules": S["n_modules"],
        },
    )

    # ---- per-layer sensitivity ranking table (the W&B artifact-grade record) ----
    cols = ["module", "layer", "proj", "params", "rel_div", "sqnr_db",
            "energy", "energy_per_param", "scale_cv_g128grp", "gs32", "gs128"]
    tbl = wandb.Table(columns=cols)
    for r in sorted(ROWS, key=lambda r: r["rel_div"], reverse=True):
        tbl.add_data(*[r[c] for c in cols])
    run.log({"per_layer_sensitivity": tbl})

    # ---- localization / Pareto-shape curve (greedy energy-per-param) ----
    ccols = ["rank", "f_param", "f_energy_removed", "tps_proj", "module"]
    ctbl = wandb.Table(columns=ccols)
    for i, c in enumerate(CURVE):
        ctbl.add_data(i + 1, c["f_param"], c["f_energy"], c["tps_proj"], c["module"])
    run.log({"localization_pareto_curve": ctbl})

    # also a wandb line plot of f_energy vs f_param (tight curve bows up; diffuse=diagonal)
    xs = [c["f_param"] for c in CURVE]
    ys = [c["f_energy"] for c in CURVE]
    run.log({"loc_curve_plot": wandb.plot.line_series(
        xs=[xs, xs], ys=[ys, xs],
        keys=["g128_excess_energy_removed", "diffuse_baseline (y=x)"],
        title="Localization: energy removed vs param-fraction (g32 footprint)",
        xname="f_param (g32 footprint = speed cost)")})

    # ---- summary scalars (machine-checkable verdict discipline, #679 standard) ----
    topk = S["topk_energy_share"]
    diffuse = S["energy_gini_like"] < 0.5 and (S["rel_div_std"] / S["rel_div_mean"]) < 0.25
    summary = {
        # explicit boundary scalars the card demands
        "analysis_only": 1,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": 0,
        # localization verdict scalars
        "localization_verdict": "DIFFUSE" if diffuse else "LOCALIZED",
        "selective_clears_speedsafe": 0,  # ruled out by diffuse localization
        "rel_div_mean": S["rel_div_mean"],
        "rel_div_std": S["rel_div_std"],
        "rel_div_cv": S["rel_div_std"] / S["rel_div_mean"],
        "rel_div_min": S["rel_div_min"],
        "rel_div_max": S["rel_div_max"],
        "rel_div_p90": S["rel_div_p90"],
        "rel_div_p99": S["rel_div_p99"],
        "energy_gini_like": S["energy_gini_like"],
        "f_param_at_50pct_energy": S["f_param_at_50pct_energy"],
        "f_param_at_80pct_energy": S["f_param_at_80pct_energy"],
        "f_param_at_90pct_energy": S["f_param_at_90pct_energy"],
        "tps_at_50pct_energy": S["tps_at_50pct_energy"],
        "tps_at_80pct_energy": S["tps_at_80pct_energy"],
        "tps_at_90pct_energy": S["tps_at_90pct_energy"],
        "top1_energy_share": topk["1"],
        "top8_energy_share": topk["8"],
        "top16_energy_share": topk["16"],
        "top32_energy_share": topk["32"],
        # best-case: smallest f that even stays within ~anchor noise (remove<=10% energy)
        "bestcase_f_at_10pct_energy": next(c["f_param"] for c in CURVE if c["f_energy"] >= 0.10),
        "bestcase_tps_at_10pct_energy": next(c["tps_proj"] for c in CURVE if c["f_energy"] >= 0.10),
    }
    for k, v in summary.items():
        run.summary[k] = v

    # ---- artifact: full per-layer probe JSON ----
    art = wandb.Artifact("sqnr_localization_probe", type="analysis",
                         description="PR#695 per-layer g32->g128 SQNR localization probe "
                                     "(343 body modules); per-layer ranking + Pareto curve.")
    art.add_file(str(HERE / "sqnr_probe.json"))
    art.add_file(str(HERE / "sqnr_probe.py"))
    run.log_artifact(art)

    print("WANDB_RUN_ID", run.id)
    print("WANDB_RUN_URL", run.url)
    print("localization_verdict", summary["localization_verdict"])
    run.finish()


if __name__ == "__main__":
    main()
