#!/usr/bin/env python
"""PR #646 — post-hoc W&B relog for the int4->int8->bf16 reasoning ladder.

WHY THIS EXISTS: the vLLM-0.22.0 eval venv (/tmp/senpai-venvs/20f658587e8a6643,
py3.12) has NO wandb installed, and the local ./wandb output dir shadows the import
anyway (eval_ladder.py does sys.path.insert(0, ROOT)). So eval_ladder.log_wandb
silently fails every cell — the group stays empty even though the per-cell
summary_{body}_{kind}.json files are correct and complete. This script re-logs those
local summaries to W&B so the deliverable has grouped runs + run_ids.

RUN IT WITH SYSTEM PYTHON FROM A NON-TARGET CWD (so `import wandb` resolves to the
installed 0.27.0, not the ./wandb output dir):

    cd research/validity/int8_bf16_reasoning_ladder      # no ./wandb here
    WANDB_DIR=$PWD/results /usr/bin/python3 relog_wandb.py [--dry-run]

ANALYSIS-ONLY. Creates W&B runs only; never touches the served file or HF.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GROUP = "int8-bf16-reasoning-ladder-fern"
EXPECTED_N = {"gpqa": 198, "aime": 60}

# Cells this script is responsible for logging (the genuinely-measured rungs).
CELLS = [("int8", "gpqa"), ("int8", "aime"), ("int4", "gpqa")]

CONFIG_COMMON = {
    "pr": 646, "decode": "greedy_t0", "max_model_len": 8192, "max_tokens": 6144,
    "min_tokens": 8, "max_num_seqs": 1, "batch_invariant": 1, "vllm": "0.22.0",
    "spec": "off_AR_M1", "analysis_only": True, "official_tps": 0,
    "wandb_group": GROUP, "relogged_via": "system_python_3.10_wandb0.27",
    "concurrency": 1,
}
QUANT = {
    "int8": ("int8_w8a16_g128_lmhead", "google/gemma-4-E4B-it@fee6332c (plain bf16)"),
    "int4": ("int4_w4a16_g128_lmhead", "int4_g128_lmhead (live served-rung body)"),
}


def load_summary(body: str, kind: str) -> dict | None:
    p = RES / f"summary_{body}_{kind}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="log even if n < expected (debug only)")
    args = ap.parse_args()

    planned = []
    for body, kind in CELLS:
        summ = load_summary(body, kind)
        if summ is None:
            print(f"[skip] {body}/{kind}: no summary yet")
            continue
        n = summ.get("n", 0)
        exp = EXPECTED_N[kind]
        if n < exp and not args.allow_partial:
            print(f"[skip] {body}/{kind}: partial n={n} < {exp} (not complete) — not logging")
            continue
        if summ.get("wandb_run_id"):
            print(f"[skip] {body}/{kind}: already has wandb_run_id={summ['wandb_run_id']}")
            continue
        planned.append((body, kind, summ))
        print(f"[plan] {body}/{kind}: n={n} acc={summ.get('acc'):.4f} "
              f"pct_bf16={summ.get('pct_of_bf16')} clears90={summ.get('clears_90pct_bar')}")

    if args.dry_run:
        print(f"[dry-run] would log {len(planned)} cell(s); no W&B runs created")
        return 0
    if not planned:
        print("[done] nothing to log")
        return 0

    import wandb  # noqa: PLC0415  (system python only)
    if not hasattr(wandb, "init"):
        print(f"[fatal] wandb has no init() — wrong module shadowed it: "
              f"{getattr(wandb, '__file__', '?')}")
        return 2

    for body, kind, summ in planned:
        qname, qbase = QUANT[body]
        cfg = dict(CONFIG_COMMON)
        cfg.update({"body": body, "eval": kind, "quant": qname, "source_base": qbase})
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=GROUP, name=f"fern/{body}-{kind}-greedy-ladder",
            config=cfg, reinit=True,
        )
        log = {f"ladder/{k}": v for k, v in summ.items()
               if isinstance(v, (int, float, bool)) or v is None}
        wandb.log(log)
        for k, v in summ.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                run.summary[k] = v
        rid = run.id
        wandb.finish()
        summ["wandb_run_id"] = rid
        (RES / f"summary_{body}_{kind}.json").write_text(json.dumps(summ, indent=2))
        print(f"[logged] {body}/{kind} -> run {rid}")

    print("[done] relog complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
