#!/usr/bin/env python
"""PR #653 (lawine) -- log the 3-arm AIME g32-recovery panel to W&B.

Reads panel_summary.json (written by aggregate.py) and logs:
  * one run per arm  : lawine/aime-{arm}  -- acc, Wilson95, %-of-bf16, bar pass,
                       extract_fail, truncation, per-year.
  * one panel run    : lawine/aime-g32-recovery-panel -- the paired deltas
                       (group-size HEADLINE + recipe cross-check, McNemar+Newcombe),
                       the GPQA-vs-AIME contrast, and the ubel-0.350 vs denken-0.400
                       reconciliation.

LOCAL ONLY: analysis_only=True, official_tps=0, NO HF Job, NO submission, NO
served-file change. Group `aime-g32-recovery-lawine`.

Run under the repo .venv (has wandb), NOT the serve venv:
  ./.venv/bin/python research/validity/aime_g32_recovery_653/log_wandb.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

GROUP = "aime-g32-recovery-lawine"
HERE = Path(__file__).resolve().parent
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

# byte-for-byte the ubel #628/#638 denominator config; only MODEL_ID changes per arm.
COMMON = {
    "vllm_version": "0.22.0",
    "engine": "manifest-pinned 0.22.0 (/tmp/vllm0220-srv)",
    "vllm_batch_invariant": 1,
    "max_model_len": 8192,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.90,
    "use_flashinfer_sampler": 0,
    "min_tokens": 8,
    "max_tokens": 6144,
    "decode": "greedy temp=0 top_p=1.0 top_k=-1, M=1 AR (no drafter)",
    "eval_seed": 1234,
    "client_concurrency": 1,
    "dataset": "AIME n=60 (2024 + 2025-I + 2025-II), maj@1",
    "serve_path": "submissions/bf16_base_aime/serve.py (MODEL_ID repoint, --dtype bfloat16)",
    "analysis_only": True,
    "official_tps": 0,
    "anchor_bf16_base_aime": 0.4667,
    "bar_90pct": 0.420,
}

ARM_MODEL = {
    "shipped_g128": "int4_g128_lmhead (W4A16 g128 body + untied int4 g128 head, minmax) -- the LIVE submission body",
    "ours_g32": "int4_g32_lmhead (#639 Arm-2: body+untied int4 head, group_size 128->32, minmax) -- the attribution arm",
    "official_g32": "google/gemma-4-E4B-it-qat-w4a16-ct (g32, tied bf16 head) -- Google's recipe cross-check",
}


def log_arm(label: str, a: dict, panel: dict) -> str:
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name=f"lawine/aime-{label}",
        group=GROUP, job_type="quality-int4-aime-recovery", reinit=True,
        config={**COMMON, "arm": label, "model": ARM_MODEL.get(label, label)},
    )
    acc = a["acc"]
    log = {
        "aime/accuracy": acc,
        "aime/n_correct": a["n_correct"],
        "aime/n": a["n"],
        "aime/wilson95_lo": a["wilson"][0],
        "aime/wilson95_hi": a["wilson"][1],
        "aime/pct_of_bf16": a["pct_of_bf16"],
        "aime/clears_90pct_bar": int(a["clears_bar"]),
        "aime/extract_fail": a["extract_fail"],
        "aime/extract_fail_rate": a["extract_fail_rate"],
        "aime/truncation": a["trunc"],
        "aime/truncation_rate": a["trunc_rate"],
        "aime/censored_acc": a["censored_acc"],
        "aime/censored_n_finished": a["n_nontrunc"],
        "aime/censored_pct_of_bf16": a["censored_pct_of_bf16"],
        "aime/wall_min": a["wall_min"],
    }
    for y, (c, t) in a["per_year"].items():
        log[f"aime/year_{y}/n_correct"] = c
        log[f"aime/year_{y}/n"] = t
        log[f"aime/year_{y}/acc"] = (c / t) if t else 0.0
    wandb.log(log)
    for kk, vv in log.items():
        run.summary[kk] = vv
    run.summary["arm"] = label
    run.summary["aime_accuracy"] = acc
    run.summary["pct_of_bf16"] = a["pct_of_bf16"]
    run.summary["clears_90pct_bar"] = bool(a["clears_bar"])
    rid = run.id
    run.finish()
    print(f"[wandb] arm={label} acc={acc:.4f} pct_bf16={a['pct_of_bf16']*100:.1f}% "
          f"clears={a['clears_bar']} id={rid}")
    return rid


def log_panel(panel: dict) -> str:
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name="lawine/aime-g32-recovery-panel",
        group=GROUP, job_type="quality-int4-aime-recovery-panel", reinit=True,
        config={**COMMON, "arm": "panel",
                "gpqa_groupsize_delta_ref": panel.get("gpqa_groupsize_delta")},
    )
    log: dict = {}
    arms = panel.get("arms", {})
    for label, a in arms.items():
        log[f"panel/{label}/acc"] = a["acc"]
        log[f"panel/{label}/pct_of_bf16"] = a["pct_of_bf16"]
        log[f"panel/{label}/clears_bar"] = int(a["clears_bar"])

    deltas = panel.get("deltas", {})
    gs = deltas.get("groupsize_ours_g32_minus_shipped_g128")
    if gs:
        log.update({
            "delta_groupsize/value": gs["delta"],
            "delta_groupsize/newcombe95_lo": gs["newcombe95"][0],
            "delta_groupsize/newcombe95_hi": gs["newcombe95"][1],
            "delta_groupsize/mcnemar_b_ours_only": gs["mcnemar_b"],
            "delta_groupsize/mcnemar_c_shipped_only": gs["mcnemar_c"],
            "delta_groupsize/mcnemar_p": gs["mcnemar_p"],
            "delta_groupsize/cells_both": gs["e_both"],
            "delta_groupsize/cells_neither": gs["h_neither"],
        })
    rc = deltas.get("recipe_official_g32_minus_shipped_g128")
    if rc:
        log.update({
            "delta_recipe/value": rc["delta"],
            "delta_recipe/newcombe95_lo": rc["newcombe95"][0],
            "delta_recipe/newcombe95_hi": rc["newcombe95"][1],
            "delta_recipe/mcnemar_p": rc["mcnemar_p"],
        })
    if "gpqa_vs_aime_fraction" in panel:
        log["contrast/gpqa_groupsize_delta"] = panel["gpqa_groupsize_delta"]
        log["contrast/aime_groupsize_delta"] = gs["delta"] if gs else None
        log["contrast/aime_as_fraction_of_gpqa_move"] = panel["gpqa_vs_aime_fraction"]

    rec = panel.get("reconciliation")
    if rec:
        log.update({
            "reconcile/n_common": rec["n_common"],
            "reconcile/my_shipped_acc": rec["my_acc"],
            "reconcile/denken_ar_acc": rec["denken_acc"],
            "reconcile/answer_divergent_items": rec["answer_divergent_items"],
            "reconcile/correctness_flips": rec["correctness_flips"],
        })
    wandb.log(log)
    for kk, vv in log.items():
        if vv is not None:
            run.summary[kk] = vv
    run.summary["surface_to_human"] = True
    rid = run.id
    run.finish()
    print(f"[wandb] panel logged id={rid} group={GROUP}")
    return rid


def main() -> int:
    panel = json.loads((HERE / "panel_summary.json").read_text())
    ids = {}
    for label, a in panel.get("arms", {}).items():
        ids[label] = log_arm(label, a, panel)
    ids["panel"] = log_panel(panel)
    (HERE / "wandb_run_ids.json").write_text(json.dumps(ids, indent=2))
    print(f"[wandb] all runs: {ids}")
    print(f"[wandb] project={ENTITY}/{PROJECT}  group={GROUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
