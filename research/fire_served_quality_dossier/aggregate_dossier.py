#!/usr/bin/env python3
"""Assemble the fire served-quality dossier (PR #753 -> #757) and log it to W&B.

Reads the per-arm/per-task eval JSONs produced by run_dossier.py for three arms:
  fire arm  : fire_sampled.json (GSM8K), fire_mmlu_pro.json, fire_aime.json
              -- int4 W4A16 target + MTP spec-dec drafter ON (the as-fired submission).
  base arm  : base_*.json  -- SAME int4 submission, drafter OFF (SENPAI_REFERENCE_MODE=1).
  bf16 arm  : bf16_*.json  -- full-precision bf16 google/gemma-4-E4B-it, drafter OFF.

Per-task decomposition (exact identity per task; panel-mean is the mean of the
four per-task percentages, matching #753):

  specdec_factor     = fire / base   -- speculative-drafter cost (isolated, #753 ~100.2%).
  int4_quant_factor  = base / bf16   -- int4-QAT W4A16 quantization cost vs the original.
  fire_pct_of_bf16   = fire / bf16   -- complete %-of-original retained (PRIMARY).
                     = int4_quant_factor x specdec_factor   (within rounding).

The bf16 arm completes the quality claim: it turns #753's "% of the int4 base"
into the citable "% of the original full-precision model" the blog wants. LOCAL
analysis only -- no HF Job. Degrades gracefully to the 2-arm specdec view when the
bf16 JSONs are absent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP = "fire_served_quality_dossier"

# The four panel metrics: (label, task, field). AIME contributes two (maj@k and
# mean pass-rate), matching the #753 panel so specdec_factor reproduces 100.21%.
METRICS = [
    ("gsm8k", "gsm8k", "acc"),
    ("mmlu_pro", "mmlu_pro", "acc"),
    ("aime_maj_k", "aime", "maj_k_acc"),
    ("aime_mean_pass_rate", "aime", "mean_pass_rate"),
]


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


def _arm(d: Path, prefix: str) -> dict:
    return {
        "gsm8k": _gsm8k_acc(_load(d / f"{prefix}_sampled.json")),
        "mmlu_pro": _mmlu_acc(_load(d / f"{prefix}_mmlu_pro.json")),
        "aime": _aime_acc(_load(d / f"{prefix}_aime.json")),
    }


def pct(num, den):
    if num is None or den in (None, 0):
        return None
    return 100.0 * num / den


def _panel_mean(rows: dict, key: str):
    xs = [r[key] for r in rows.values() if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--name", default="ubel/fire-bf16-fullprecision-denominator")
    ap.add_argument("--bf16-model", default="google/gemma-4-E4B-it",
                    help="the full-precision denominator served for the bf16 arm")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)

    arms = {"fire": _arm(d, "fire"), "base": _arm(d, "base"), "bf16": _arm(d, "bf16")}
    have_bf16 = any(arms["bf16"].get(t) for t in ("gsm8k", "mmlu_pro", "aime"))

    def val(arm: str, task: str, field: str):
        return (arms[arm].get(task) or {}).get(field)

    per_task: dict[str, dict] = {}
    for label, task, field in METRICS:
        f, b, v = val("fire", task, field), val("base", task, field), val("bf16", task, field)
        per_task[label] = {
            "fire": f, "base": b, "bf16": v,
            "specdec_factor": pct(f, b),      # fire / base   (spec-dec drafter cost)
            "int4_quant_factor": pct(b, v),   # base / bf16   (W4A16+QAT quant cost)
            "fire_pct_of_bf16": pct(f, v),    # fire / bf16   (complete %-of-original)
            # per-task exact product check: int4_quant_factor x specdec_factor / 100
            "product_check": (pct(b, v) * pct(f, b) / 100.0)
            if (pct(b, v) is not None and pct(f, b) is not None) else None,
        }

    headline = {
        "specdec_factor_panel_mean": _panel_mean(per_task, "specdec_factor"),
        "int4_quant_factor_panel_mean": _panel_mean(per_task, "int4_quant_factor"),
        "fire_pct_of_bf16_panel_mean": _panel_mean(per_task, "fire_pct_of_bf16"),
    }
    iq, sd = headline["int4_quant_factor_panel_mean"], headline["specdec_factor_panel_mean"]
    # Product of the panel means: only ~= the panel mean of fire_pct_of_bf16,
    # since mean-of-products != product-of-means. Reported as a coarse cross-check;
    # the exact identity is the per-task product_check above.
    headline["product_of_panel_means"] = (iq * sd / 100.0) if (iq and sd) else None

    dossier = {
        "arms": arms,
        "per_task": per_task,
        "headline_panel_mean": headline,
        "primary_metric": {"name": "fire_pct_of_bf16_panel_mean",
                           "value": headline["fire_pct_of_bf16_panel_mean"]},
        "denominator": {
            "bf16_model_id": args.bf16_model,
            "why": "google/gemma-4-E4B-it is the original full-precision instruct model "
                   "Google released; the int4-QAT submission (google/gemma-4-E4B-it-qat-"
                   "w4a16-ct) descends from it. It is the denominator a skeptical blog "
                   "reader means by 'the original model', so fire_pct_of_bf16 = '% of the "
                   "original full-precision model'.",
            "alternative": "google/gemma-4-E4B-it-qat-q4_0-unquantized -- the QAT bf16 "
                           "checkpoint the W4A16 was *directly* quantized from. Against it, "
                           "int4_quant_factor would isolate W4A16 rounding ALONE; against "
                           "the plain -it it bundles QAT adaptation + W4A16 (the headline "
                           "'int4-QAT quantization' cost).",
        },
        "protocol": {
            "decode": "sampled generation_config.json (T=1.0, top_p=0.95, top_k=64)",
            "eos_guard": "min_tokens=8 (#541)",
            "stack": "vLLM 0.22.0 api_server, dtype=bf16, VLLM_BATCH_INVARIANT=1, native "
                     "torch sampler; one arm at a time on the A10G.",
            "arms": {
                "fire": "int4 W4A16 target + MTP drafter ON (num_spec_tokens=6).",
                "base": "same int4 submission, drafter OFF (SENPAI_REFERENCE_MODE=1).",
                "bf16": f"MODEL_ID={args.bf16_model}, drafter OFF, native bf16 (no Marlin).",
            },
        },
    }

    out = d / "dossier.json"
    out.write_text(json.dumps(dossier, indent=2))
    print(json.dumps(dossier, indent=2))
    print(f"[dossier] wrote {out}  (have_bf16={have_bf16})")

    if args.no_wandb:
        return 0

    import wandb

    def g(label, field):
        return per_task[label][field]

    flat = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0, "pr": 757,
        # raw arm accuracies
        "fire_mmlu_pro_acc": g("mmlu_pro", "fire"), "fire_gsm8k_acc": g("gsm8k", "fire"),
        "fire_aime_maj_k_acc": g("aime_maj_k", "fire"),
        "fire_aime_mean_pass_rate": g("aime_mean_pass_rate", "fire"),
        "base_mmlu_pro_acc": g("mmlu_pro", "base"), "base_gsm8k_acc": g("gsm8k", "base"),
        "base_aime_maj_k_acc": g("aime_maj_k", "base"),
        "base_aime_mean_pass_rate": g("aime_mean_pass_rate", "base"),
        "bf16_mmlu_pro_acc": g("mmlu_pro", "bf16"), "bf16_gsm8k_acc": g("gsm8k", "bf16"),
        "bf16_aime_maj_k_acc": g("aime_maj_k", "bf16"),
        "bf16_aime_mean_pass_rate": g("aime_mean_pass_rate", "bf16"),
        # per-task factors
        "specdec_factor_gsm8k": g("gsm8k", "specdec_factor"),
        "specdec_factor_mmlu_pro": g("mmlu_pro", "specdec_factor"),
        "specdec_factor_aime_maj_k": g("aime_maj_k", "specdec_factor"),
        "specdec_factor_aime_mean_pass_rate": g("aime_mean_pass_rate", "specdec_factor"),
        "int4_quant_factor_gsm8k": g("gsm8k", "int4_quant_factor"),
        "int4_quant_factor_mmlu_pro": g("mmlu_pro", "int4_quant_factor"),
        "int4_quant_factor_aime_maj_k": g("aime_maj_k", "int4_quant_factor"),
        "int4_quant_factor_aime_mean_pass_rate": g("aime_mean_pass_rate", "int4_quant_factor"),
        "fire_pct_of_bf16_gsm8k": g("gsm8k", "fire_pct_of_bf16"),
        "fire_pct_of_bf16_mmlu_pro": g("mmlu_pro", "fire_pct_of_bf16"),
        "fire_pct_of_bf16_aime_maj_k": g("aime_maj_k", "fire_pct_of_bf16"),
        "fire_pct_of_bf16_aime_mean_pass_rate": g("aime_mean_pass_rate", "fire_pct_of_bf16"),
        # panel-mean headline
        "specdec_factor_panel_mean": headline["specdec_factor_panel_mean"],
        "int4_quant_factor_panel_mean": headline["int4_quant_factor_panel_mean"],
        "fire_pct_of_bf16_panel_mean": headline["fire_pct_of_bf16_panel_mean"],
        "product_of_panel_means": headline["product_of_panel_means"],
        "fire_ppl_gate": 2.0055150113084133, "ppl_gate_cap": 2.42,
    }
    config = {
        "fire": "submissions/int4_mtp_batchinv, drafter ON (K=6)",
        "base": "submissions/int4_mtp_batchinv, drafter OFF (SENPAI_REFERENCE_MODE=1)",
        "bf16": f"{args.bf16_model}, drafter OFF, native bf16",
        "fire_model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "bf16_model_id": args.bf16_model,
        "bf16_alternative_denominator": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
        "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "decode": "sampled T=1.0 top_p=0.95 top_k=64 (lewtun #31)", "eos_guard": "min_tokens=8 (#541)",
        "batch_invariant": 1,
        "gsm8k_n": val("bf16", "gsm8k", "n") or val("fire", "gsm8k", "n"),
        "mmlu_n": val("bf16", "mmlu_pro", "n") or val("fire", "mmlu_pro", "n"),
        "aime_n": val("bf16", "aime", "n") or val("fire", "aime", "n"),
        "aime_k": val("bf16", "aime", "k") or val("fire", "aime", "k"),
    }
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"), project=os.environ.get("WANDB_PROJECT"),
        group=GROUP, name=args.name, config=config, job_type="served-quality-eval",
    )
    table = wandb.Table(columns=["task", "metric", "fire", "base", "bf16",
                                 "specdec_factor", "int4_quant_factor", "fire_pct_of_bf16"])
    pretty = {"gsm8k": ("gsm8k", "acc"), "mmlu_pro": ("mmlu_pro", "acc"),
              "aime_maj_k": ("aime", "maj@k"), "aime_mean_pass_rate": ("aime", "mean_pass_rate")}
    for label in [m[0] for m in METRICS]:
        r = per_task[label]
        task, metric = pretty[label]
        table.add_data(task, metric, r["fire"], r["base"], r["bf16"],
                       r["specdec_factor"], r["int4_quant_factor"], r["fire_pct_of_bf16"])
    wandb.log({**flat, "dossier_table": table})
    print(f"[dossier] W&B run id: {run.id}  url: {run.url}")
    (d / "_wandb_run_id.txt").write_text(run.id)
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
