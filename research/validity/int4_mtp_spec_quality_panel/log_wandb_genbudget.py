#!/usr/bin/env python
"""PR #612 (fern) — log the Option-B GPQA generation-budget retry to W&B.

Tests the human's #481 question: is stark #605's GPQA-Diamond FAIL (pooled 0.4141,
3 seeds @ max_tokens=3072 on vLLM 0.22.0) a generation-budget TRUNCATION artifact?
Re-run on dev307 @ max_model_len=8192 with an explicit generous budget, logging the
finish_reason / length_stop_rate diagnostics #605 lacked.

LOCAL ONLY, analysis_only=True, official_tps=0. Group `optionb-gpqa-genbudget-retry`.
This config is NOT #319 byte-identical (spec) -> SURFACE-to-human, never auto-fire.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb

GROUP = "optionb-gpqa-genbudget-retry"
HERE = Path(__file__).resolve().parent
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
BAR = 0.471
IMPLIED_BASE = 0.523
# stark #605 baseline (the number this card overturns or confirms).
BASELINE_605 = {"pooled_accuracy": 0.4141, "n": 594, "max_tokens": 3072,
                "max_model_len": 6144, "vllm_version": "0.22.0", "stderr": 0.0202}

COMMON = {
    "config_name": "int4_g128_lmhead + MTP-K7 spec (fern #597)",
    "model_id": "/workspace/gemma_build/int4_g128_lmhead",
    "drafter": "/tmp/qat-assistant",
    "num_speculative_tokens": 7,
    "vllm_batch_invariant": 1,
    "vllm_version": "0.22.1rc1.dev307+g3e8afdf78",
    "max_model_len": 8192,
    "max_num_seqs": 16,
    "min_tokens": 8,
    "sampling": "T=1.0, top_p=0.95, top_k=64 (generation_config.json)",
    "is_319_identical": False,
    "analysis_only": True,
    "official_tps": 0,
    "baseline_605": BASELINE_605,
}


def log_arm(pooled_path: Path) -> dict:
    d = json.loads(pooled_path.read_text())
    tag = d["tag"]
    mt = d["per_seed"][0]["max_tokens"] if d["per_seed"] else None
    acc = d["pooled_accuracy"]
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name=f"fern/gpqa-genbudget-{tag}",
        group=GROUP, job_type="quality-eval", reinit=True,
        config={**COMMON, "arm_tag": tag, "max_tokens": mt, "seeds": d["seeds"],
                "bar_90pct": BAR, "implied_base": IMPLIED_BASE},
    )
    retro = d.get("retro_truncation_at_cap", {})
    log = {
        "gpqa_accuracy": acc,
        "max_tokens": mt,
        "bar_90pct": BAR,
        "margin_over_bar": acc - BAR,
        "pct_of_base": acc / IMPLIED_BASE,
        "pass": int(d["pass"]),
        "n_scored": d["n_scored"],
        "n_correct": d["n_correct"],
        "stderr": d["stderr"],
        "ci95_lo_normal": d["ci95_normal"][0],
        "ci95_hi_normal": d["ci95_normal"][1],
        "ci95_lo_wilson": d["ci95_wilson"][0],
        "ci95_hi_wilson": d["ci95_wilson"][1],
        "sigma_vs_bar": d["sigma_vs_bar"],
        "length_stop_rate": d["pooled_length_stop_rate"],
        "n_length_truncated": d["n_length_truncated"],
        "completion_tokens_mean": d["completion_tokens_mean_weighted"],
        "delta_vs_605": acc - BASELINE_605["pooled_accuracy"],
        # retro truncation: frac of NATURAL completions that exceed a tighter cap
        # (= length-stop rate #605 would have suffered at that cap, same generations)
        "retro_trunc_at_605cap_3072": retro.get("3072", {}).get("frac_over"),
        "retro_trunc_at_2048": retro.get("2048", {}).get("frac_over"),
        "retro_trunc_at_4096": retro.get("4096", {}).get("frac_over"),
    }
    wandb.log(log)
    run.summary["gpqa_accuracy"] = acc
    run.summary["pass"] = bool(d["pass"])
    run.summary["length_stop_rate"] = d["pooled_length_stop_rate"]
    run.finish()
    print(f"[wandb] {tag}: acc={acc:.4f} (n={d['n_scored']}) len_stop_rate={d['pooled_length_stop_rate']:.4f} "
          f"-> {'PASS' if d['pass'] else 'FAIL'} vs bar {BAR}")
    return d


def log_verdict(arms: dict) -> None:
    gb = arms.get("gb6144") or next(iter(arms.values()))
    acc = gb["pooled_accuracy"]
    resurrected = bool(gb["pass"])
    verdict = "OPTION-B-RESURRECTED" if resurrected else "OPTION-B-DEAD-TRUNCATION-RULED-OUT"
    retro = gb.get("retro_truncation_at_cap", {})
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name="fern/gpqa-genbudget-VERDICT",
        group=GROUP, job_type="verdict", reinit=True,
        config={**COMMON, "bar_90pct": BAR, "implied_base": IMPLIED_BASE,
                "verdict_arm": "gb6144"},
    )
    log = {
        "verdict_resurrected": int(resurrected),
        "gpqa_accuracy_generous": acc,
        "bar_90pct": BAR,
        "margin_over_bar": acc - BAR,
        "delta_vs_605_0p4141": acc - BASELINE_605["pooled_accuracy"],
        "generous_length_stop_rate": gb["pooled_length_stop_rate"],
        # decisive truncation delta: old 3072 cap (retro) -> new 6144 budget (measured)
        "retro_trunc_at_605cap_3072": retro.get("3072", {}).get("frac_over"),
        "trunc_delta_old3072_to_new6144": (
            (retro.get("3072", {}).get("frac_over") or 0.0) - gb["pooled_length_stop_rate"]),
    }
    for tag, d in arms.items():
        log[f"{tag}_accuracy"] = d["pooled_accuracy"]
        log[f"{tag}_length_stop_rate"] = d["pooled_length_stop_rate"]
    wandb.log(log)
    run.summary["verdict"] = verdict
    run.summary["surface_to_human"] = True  # non-#319-identical: never auto-fire
    run.finish()
    print(f"\n[VERDICT] {verdict}  generous-budget GPQA={acc:.4f} vs bar {BAR} "
          f"(Δ vs #605 0.4141 = {acc-BASELINE_605['pooled_accuracy']:+.4f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["gb6144"])
    args = ap.parse_args()
    arms = {}
    for tag in args.tags:
        p = HERE / f"{tag}_pooled.json"
        if p.exists():
            arms[tag] = log_arm(p)
        else:
            print(f"[wandb] skip {tag}: {p} missing")
    if arms:
        log_verdict(arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
