#!/usr/bin/env python
"""PR #805 — log the int4head+PLE-dequant experiment to W&B.

The in-harness validate wandb hook ran in the server venv (no wandb installed),
so it silently skipped. This logs ONE rich run from the collected JSON artifacts
(TPS A/B, PPL/greedy/gate, GSM8K, dispatch proof, xsession determinism) into the
challenge project so the experiment leaves a queryable record.

Run with a python that has full wandb (e.g. .venv/bin/python), NOT the server venv.
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

ROOT = Path("/workspace/senpai/target/research/int4head_ple_dequant")
ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    tps = jload(ROOT / "tps" / "tps_ab_summary.json")
    qual = jload(ROOT / "quality" / "quality_summary.json")
    gsm_g = jload(ROOT / "quality" / "gsm8k" / "pledequant_greedy_greedy.json")
    gsm_s = jload(ROOT / "quality" / "gsm8k" / "pledequant_sampled_sampled.json")
    xses = jload(ROOT / "xsession" / "xsession_summary.json")
    pub = jload(ROOT / "publish_hub.json")

    config = {
        "experiment": "int4head + per_layer_input_gate de-quant (bf16/cuBLAS)",
        "pr": 805,
        "wandb_group": "bi0-int4head-ple-dequant",
        "base_submission": "int4_mtp_bi0_int4head",
        "delta": "42x per_layer_input_gate (N=256) int4-Marlin -> bf16 cuBLAS (UnquantizedLinearMethod)",
        "ple_source": "dequant (q_int4 * scale -> bf16; exact served values, kernel-isolated)",
        "ignore_added": "re:.*per_layer_input_gate",
        "lm_head": "int4 g32 (untied)",
        "body": "int4 W4A16 (byte-identical to google/gemma-4-E4B-it-qat-w4a16-ct)",
        "drafter": "MTP K=6 (gemma-4-E4B-it-qat-q4_0-unquantized-assistant)",
        "control": "int4_mtp_bi0_int4head (within-job A/B)",
        "harness": "128x512 warm decode, 3 reps interleaved, conc=1",
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name="ubel/pledequant",
        group="bi0-int4head-ple-dequant",
        job_type="serve-measure",
        tags=["pr805", "int4head", "ple-dequant", "kernel-lever", "fire-track"],
        config=config,
    )

    summary = {}
    if tps:
        pd, ih = tps["pledequant"], tps["int4head_control"]
        summary.update({
            "tps/pledequant_median": pd["median"],
            "tps/pledequant_mean": pd["mean"],
            "tps/pledequant_cv_pct": pd["cv_pct"],
            "tps/pledequant_e_accept": pd["e_accept_median"],
            "tps/int4head_control_median": ih["median"],
            "tps/int4head_control_mean": ih["mean"],
            "tps/int4head_control_cv_pct": ih["cv_pct"],
            "tps/int4head_control_e_accept": ih["e_accept_median"],
            "tps/delta_abs": tps["delta_abs_tps"],
            "tps/delta_pct": tps["delta_pct"],
        })
        # per-rep table
        tbl = wandb.Table(columns=["arm", "rep", "tps"])
        for arm, key in (("pledequant", "pledequant"), ("int4head_control", "int4head_control")):
            for i, v in enumerate(tps[key]["reps"], 1):
                tbl.add_data(arm, i, v)
        run.log({"tps/per_rep": tbl})
    if qual:
        ev = qual.get("evidence", {})
        ab = qual.get("ab", {})
        summary.update({
            "quality/ppl": ev.get("ppl"),
            "quality/completed": ev.get("completed"),
            "quality/all_modalities_loaded": ev.get("all_modalities_loaded"),
            "quality/official_gate": ev.get("official_gate"),
            "quality/greedy_verdict_specon_vs_specoff": ev.get("greedy_verdict"),
            "quality/greedy_specon_num_divergent": ev.get("greedy_onset", {}).get("num_divergent"),
            "quality/ab_vs_int4head_num_divergent": ab.get("num_divergent"),
            "quality/ab_vs_int4head_first_token_flips": ab.get("first_token_flips"),
            "quality/ab_vs_int4head_onset_median": ab.get("onset", {}).get("onset_median"),
        })
    if gsm_g:
        summary["quality/gsm8k_greedy_acc"] = gsm_g.get("accuracy")
        summary["quality/gsm8k_greedy_n"] = gsm_g.get("n_problems")
    if gsm_s:
        summary["quality/gsm8k_sampled_acc"] = gsm_s.get("accuracy")
    if xses:
        for label, c in xses.get("cases", {}).items():
            summary[f"xsession/{label}_specoff_AvsB_divergent"] = c.get("num_divergent")
            summary[f"xsession/{label}_specoff_AvsB_verdict"] = c.get("verdict")
    if pub:
        summary["publish/repo_id"] = pub.get("repo_id")
        summary["publish/sha"] = pub.get("sha")
        summary["publish/private"] = pub.get("private")

    # Baselines for context (from PR body).
    summary.update({
        "baseline/int4head_published_tps": 256.74,
        "baseline/int4head_ppl": 2.0029,
        "baseline/int4head_gsm8k": 0.9150,
        "baseline/bi0_shipped_tps": 218.02,
        "baseline/projection_pct": 5.3,
        "floors/gsm8k": 0.807,
    })

    run.summary.update({k: v for k, v in summary.items() if v is not None})
    print("logged run:", run.id, run.url)
    print(json.dumps({k: v for k, v in summary.items() if v is not None}, indent=2))
    run.finish()
    Path(ROOT / "wandb_run_id.txt").write_text(run.id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
