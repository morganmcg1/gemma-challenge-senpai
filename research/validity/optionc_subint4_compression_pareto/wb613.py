#!/usr/bin/env python3
"""W&B run for PR #613 — Option-C(a) 2:4-sparse W4A16 kernel grep (terminal).

Phase 0 was decisive: the sparse-Marlin path is fully removed from vLLM dev307,
so no build/serve/measure follows. This run records the terminal kernel-grep
finding for the optionc-sparse24-w4a16 group.

LOCAL / analysis_only / official_tps=0 / NO FIRE.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ID_FILE = HERE / "run_id_613.txt"
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
GROUP = "optionc-sparse24-w4a16"
NAME = "stark/optionc-sparse24-w4a16-build"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=HERE, text=True,
                              capture_output=True, check=False).stdout.strip()
    except Exception:
        return ""


def main():
    import wandb
    rid = RUN_ID_FILE.read_text().strip() if RUN_ID_FILE.exists() else wandb.util.generate_id()
    run = wandb.init(
        project=PROJECT, entity=ENTITY, id=rid, resume="allow",
        name=NAME, group=GROUP, job_type="optionc-sparse24-kernel-grep",
        tags=["gemma-challenge", "optionc", "sparse24", "w4a16", "sparse-marlin",
              "kernel-grep", "analysis-only", "pr-613", "no-fire", "terminal"],
        config={
            "pr": 613, "analysis_only": True, "official_tps": 0, "no_fire": True,
            "engine": "vllm-dev307 (0.22.1rc1.dev307+g3e8afdf78)",
            "phase": 0, "decisive_gate": "sparse-Marlin kernel presence",
            "baseline_tps": 126.378, "baseline_ppl": 2.019,
            "baseline_bytes_per_token_gib": 9.85, "gate_ppl_max": 2.42,
            "gate_bars": {"gsm8k": 0.807, "mmlu_pro": 0.605,
                          "gpqa_diamond": 0.471, "aime": 0.090},
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
        },
    )
    RUN_ID_FILE.write_text(run.id)

    s = run.summary
    s["phase0/kernel_present"] = False
    s["phase0/dense_marlin_gemm_present"] = True
    s["phase0/sparse_gptq_marlin_24_gemm_present"] = False
    s["phase0/sparse_w4a16_24_scheme_present"] = False
    s["phase0/config_parser_rejects_sparsity"] = True
    s["phase0/config_parser_locus"] = (
        "compressed_tensors.py:288-293 _parse_sparsity_config raises DeprecationWarning")
    s["phase0/build_attempted"] = False
    s["phase0/disk_free_gb_at_pickup"] = 131
    s["verdict/optionc_a_dead"] = True
    s["verdict/optionc_closed"] = True
    s["verdict/ar_ceiling_tps"] = 126.378
    s["verdict/optionc_verdict"] = (
        "kernel_present=false: sparse-Marlin path (gptq_marlin_24_gemm + "
        "W4A16Sparse24/CompressedTensors24) fully removed from vLLM dev307 "
        "(scheme class, compiled op, and config parser all gone/rejecting); dense "
        "WNA16->Marlin baseline intact. Option-C(a) DEAD; option C CLOSED; 126.378 "
        "stands as the quality-safe AR weight-compression ceiling.")
    run.log({"global_step": 0, "stage": 0, "kernel_present": 0,
             "event/name": "phase0_terminal_kernel_absent"})
    print(json.dumps({"run_id": run.id, "url": run.url, "group": GROUP}))
    run.finish()


if __name__ == "__main__":
    main()
