#!/usr/bin/env python3
"""PR #627 -- log the int4 GPQA crater config-axis sweep to W&B.

Reads every gpqa_<label>.json the sweep produced (run_eval.py output), plus a
small arm-metadata JSON (label -> serve_config/conc/axis/note), and logs ONE
analysis-only run (group optionb-int4-crater-config-axis, official_tps=0). Per arm
we log accuracy, finish_length_rate (length_stop_rate), completion-token stats,
stop-reason counts, and a prompt_sha parity digest (sha256 of the sorted
(id,prompt_sha) pairs) so the byte-identity of prompts across arms is auditable
(axis e). The verdict fields are passed on the CLI.

Run under the repo .venv (has wandb). WANDB_DIR set off-tree to dodge the local
./wandb shadow.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

RUNS = Path("research/validity/optionb_crater_config_axis/runs")


def prompt_sha_digest(per_sample) -> tuple[str, int]:
    pairs = sorted((str(r.get("id")), str(r.get("prompt_sha"))) for r in per_sample)
    h = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()[:16]
    return h, len(pairs)


def load_arm(label: str) -> dict | None:
    p = RUNS / f"gpqa_{label}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="arm metadata json: {label:{serve,conc,axis,note}}")
    ap.add_argument("--group", default="optionb-int4-crater-config-axis")
    ap.add_argument("--name", default="lawine/optionb-int4-crater-config-axis")
    ap.add_argument("--crater-controlling-axis", default="")
    ap.add_argument("--gpqa-healthy-config", default="")
    ap.add_argument("--gpqa-healthy-acc", type=float, default=float("nan"))
    ap.add_argument("--gpqa-cratered-config", default="")
    ap.add_argument("--gpqa-cratered-acc", type=float, default=float("nan"))
    ap.add_argument("--submission-config-gpqa", type=float, default=float("nan"))
    ap.add_argument("--submission-config-finish-length-rate", type=float, default=float("nan"))
    ap.add_argument("--crater-at-submission-config", default="")  # "true"/"false"/""
    ap.add_argument("--verdict", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    metrics = {}
    table_rows = []
    digests = {}
    arm_idsha = {}
    for label, m in meta.items():
        d = load_arm(label)
        if d is None:
            print(f"[log] MISSING arm {label}")
            continue
        psd, npairs = prompt_sha_digest(d.get("per_sample", []))
        digests[label] = psd
        arm_idsha[label] = {str(r.get("id")): str(r.get("prompt_sha"))
                            for r in d.get("per_sample", [])}
        acc = d.get("accuracy")
        lsr = d.get("length_stop_rate")
        base = f"arm/{label}"
        metrics[f"{base}/accuracy"] = acc
        metrics[f"{base}/finish_length_rate"] = lsr
        metrics[f"{base}/n_scored"] = d.get("n_scored")
        metrics[f"{base}/max_tokens"] = d.get("max_tokens")
        metrics[f"{base}/min_tokens"] = d.get("min_tokens")
        metrics[f"{base}/conc"] = m.get("conc")
        metrics[f"{base}/ctok_mean"] = d.get("completion_tokens_mean")
        metrics[f"{base}/ctok_p95"] = d.get("completion_tokens_p95")
        metrics[f"{base}/n_stop_max_tokens"] = d.get("n_stop_max_tokens")
        metrics[f"{base}/prompt_sha_digest"] = psd
        table_rows.append([
            label, m.get("axis", ""), m.get("serve", ""), m.get("conc"),
            d.get("max_tokens"), d.get("min_tokens"), d.get("n_scored"),
            round(acc, 4) if isinstance(acc, float) else acc,
            round(lsr, 4) if isinstance(lsr, float) else lsr,
            d.get("completion_tokens_mean"), psd, m.get("note", ""),
        ])

    # prompt_sha parity across arms that claim the same construction (seed12345).
    # Compare per-id, NOT whole-arm digests: a limit=100 arm is a strict subset of
    # the n=198 id-set, so its full-set digest necessarily differs even when every
    # shared prompt is byte-identical. True divergence = an id whose prompt_sha
    # disagrees across the arms that contain it.
    seed_labels = [k for k in arm_idsha if meta.get(k, {}).get("prompt") in (None, "seed12345")]
    id_to_shas: dict[str, set] = {}
    for k in seed_labels:
        for i, sha in arm_idsha[k].items():
            id_to_shas.setdefault(i, set()).add(sha)
    n_mismatch = sum(1 for shas in id_to_shas.values() if len(shas) > 1)
    parity = (n_mismatch == 0) if id_to_shas else None

    print(f"[log] {len(table_rows)} arms; prompt_sha parity (seed12345 arms) = {parity} "
          f"(ids={len(id_to_shas)} divergent={n_mismatch})")
    for r in sorted(table_rows):
        print("   ", r)
    print(f"[log] digests: {digests}")

    if args.dry_run:
        return 0

    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    crater_bool = {"true": True, "false": False}.get(args.crater_at_submission_config.lower(), None)
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        group=args.group,
        name=args.name,
        config={
            "config_under_test": "int4_g128_lmhead (shipped submission == /workspace/gemma_build/int4_g128_lmhead)",
            "stack": "vLLM 0.22.0 (submission pin), all LOCAL A10G",
            "decode": "GREEDY (temp=0) GPQA-Diamond n=198",
            "analysis_only": True,
            "official_tps": 0,
            "gpqa_bar": 0.471,
            "gpqa_bar_strict_clean": 0.4782,
            "crater_controlling_axis": args.crater_controlling_axis,
            "gpqa_healthy_config": args.gpqa_healthy_config,
            "gpqa_healthy_acc": args.gpqa_healthy_acc,
            "gpqa_cratered_config": args.gpqa_cratered_config,
            "gpqa_cratered_acc": args.gpqa_cratered_acc,
            "submission_config_gpqa": args.submission_config_gpqa,
            "submission_config_finish_length_rate": args.submission_config_finish_length_rate,
            "crater_at_submission_config": crater_bool,
            "verdict": args.verdict,
            "prompt_sha_parity_seed12345_arms": parity,
        },
    )
    wandb.log(metrics)
    wandb.summary.update({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})
    cols = ["arm", "axis", "serve", "conc", "max_tok", "min_tok", "n", "accuracy",
            "finish_len_rate", "ctok_mean", "prompt_sha", "note"]
    try:
        wandb.log({"arm_table": wandb.Table(columns=cols, data=sorted(table_rows))})
    except Exception as e:
        print(f"[log] table log skipped: {e}")
    print(f"\n[wandb] logged run {run.id} (group={args.group})")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
