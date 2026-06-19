#!/usr/bin/env python3
"""Assemble the fire served-quality dossier (PR #753) and log it to W&B.

Reads the per-arm/per-task eval JSONs produced by run_dossier.py:
  fire arm  : fire_sampled.json (GSM8K), fire_mmlu_pro.json, fire_aime.json
  base arm  : base_sampled.json (GSM8K), base_mmlu_pro.json, base_aime.json
where `base` is the SAME int4_mtp_batchinv submission served drafter-OFF
(SENPAI_REFERENCE_MODE=1) -- a matched denominator that isolates the speculative
drafter. Computes per-task absolute accuracy (fire) + %-of-base, writes
dossier.json, and logs a flat metric set + a per-task table to W&B group
fire_served_quality_dossier. LOCAL analysis only -- no HF Job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP = "fire_served_quality_dossier"


def _load(p: Path):
    return json.load(open(p)) if p.exists() else None


def _gsm8k_acc(d):
    if not d:
        return None
    return {"acc": d.get("accuracy"), "n_correct": d.get("n_correct"),
            "n": d.get("n_problems"), "min_tokens": (d.get("sampling") or {}).get("min_tokens")}


def _mmlu_acc(d):
    if not d:
        return None
    return {"acc": d.get("accuracy"), "n_correct": d.get("n_correct"),
            "n": d.get("n_scored"), "min_tokens": d.get("min_tokens"),
            "empty_rate": d.get("empty_rate")}


def _aime_acc(d):
    if not d:
        return None
    return {"maj_k_acc": d.get("maj_k_accuracy"), "mean_pass_rate": d.get("mean_pass_rate"),
            "n_correct_maj": d.get("n_correct_maj"), "n": d.get("n_problems"),
            "k": d.get("maj_k"), "min_tokens": (d.get("sampling") or {}).get("min_tokens")}


def pct(fire, base):
    if fire is None or base in (None, 0):
        return None
    return 100.0 * fire / base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--name", default="ubel/fire-served-quality-dossier")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)

    fire = {
        "gsm8k": _gsm8k_acc(_load(d / "fire_sampled.json")),
        "mmlu_pro": _mmlu_acc(_load(d / "fire_mmlu_pro.json")),
        "aime": _aime_acc(_load(d / "fire_aime.json")),
    }
    base = {
        "gsm8k": _gsm8k_acc(_load(d / "base_sampled.json")),
        "mmlu_pro": _mmlu_acc(_load(d / "base_mmlu_pro.json")),
        "aime": _aime_acc(_load(d / "base_aime.json")),
    }

    gsm_f = (fire["gsm8k"] or {}).get("acc")
    gsm_b = (base["gsm8k"] or {}).get("acc")
    mm_f = (fire["mmlu_pro"] or {}).get("acc")
    mm_b = (base["mmlu_pro"] or {}).get("acc")
    aime_f = (fire["aime"] or {}).get("maj_k_acc")
    aime_b = (base["aime"] or {}).get("maj_k_acc")
    aime_pf = (fire["aime"] or {}).get("mean_pass_rate")
    aime_pb = (base["aime"] or {}).get("mean_pass_rate")

    dossier = {
        "fire": fire,
        "base": base,
        "pct_of_base": {
            "gsm8k": pct(gsm_f, gsm_b),
            "mmlu_pro": pct(mm_f, mm_b),
            "aime_maj_k": pct(aime_f, aime_b),
            "aime_mean_pass_rate": pct(aime_pf, aime_pb),
        },
        "protocol": {
            "decode": "sampled generation_config.json (T=1.0, top_p=0.95, top_k=64)",
            "eos_guard": "min_tokens=8 (#541)",
            "base_denominator": "same submission, drafter OFF (SENPAI_REFERENCE_MODE=1) "
                                "-> isolates the speculative drafter on an identical "
                                "int4 W4A16 / vLLM 0.22.0 stack",
        },
    }

    pcts = [v for v in dossier["pct_of_base"].values() if v is not None]
    panel_mean = sum(pcts) / len(pcts) if pcts else None
    if panel_mean is not None:
        dossier["headline_panel_mean_pct_of_base"] = panel_mean

    out = d / "dossier.json"
    out.write_text(json.dumps(dossier, indent=2))
    print(json.dumps(dossier, indent=2))
    print(f"[dossier] wrote {out}")

    if args.no_wandb:
        return 0

    import wandb

    flat = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0, "pr": 753,
        "mmlu_pro_acc": mm_f, "gsm8k_acc": gsm_f, "aime_maj_k_acc": aime_f,
        "aime_mean_pass_rate": aime_pf,
        "base_mmlu_pro_acc": mm_b, "base_gsm8k_acc": gsm_b, "base_aime_maj_k_acc": aime_b,
        "base_aime_mean_pass_rate": aime_pb,
        "pct_of_base_mmlu_pro": dossier["pct_of_base"]["mmlu_pro"],
        "pct_of_base_gsm8k": dossier["pct_of_base"]["gsm8k"],
        "pct_of_base_aime_maj_k": dossier["pct_of_base"]["aime_maj_k"],
        "pct_of_base_aime_mean_pass_rate": dossier["pct_of_base"]["aime_mean_pass_rate"],
        "panel_mean_pct_of_base": panel_mean,
        "fire_ppl_gate": 2.0055150113084133, "ppl_gate_cap": 2.42,
    }
    config = {
        "submission": "submissions/int4_mtp_batchinv", "arm_fire": "drafter ON (K=6)",
        "arm_base": "drafter OFF (SENPAI_REFERENCE_MODE=1)",
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "decode": "sampled T=1.0 top_p=0.95 top_k=64 (lewtun #31)", "eos_guard": "min_tokens=8 (#541)",
        "batch_invariant": 1,
        "gsm8k_n": (fire["gsm8k"] or {}).get("n"), "mmlu_n": (fire["mmlu_pro"] or {}).get("n"),
        "aime_n": (fire["aime"] or {}).get("n"), "aime_k": (fire["aime"] or {}).get("k"),
    }
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"), project=os.environ.get("WANDB_PROJECT"),
        group=GROUP, name=args.name, config=config, job_type="served-quality-eval",
    )
    table = wandb.Table(columns=["task", "metric", "fire", "base", "pct_of_base"])
    table.add_data("gsm8k", "acc", gsm_f, gsm_b, dossier["pct_of_base"]["gsm8k"])
    table.add_data("mmlu_pro", "acc", mm_f, mm_b, dossier["pct_of_base"]["mmlu_pro"])
    table.add_data("aime", "maj@k", aime_f, aime_b, dossier["pct_of_base"]["aime_maj_k"])
    table.add_data("aime", "mean_pass_rate", aime_pf, aime_pb,
                   dossier["pct_of_base"]["aime_mean_pass_rate"])
    wandb.log({**flat, "dossier_table": table})
    print(f"[dossier] W&B run id: {run.id}  url: {run.url}")
    (d / "_wandb_run_id.txt").write_text(run.id)
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
