#!/usr/bin/env python
"""Log the int4_g128+MTP greedy-identity gate results to W&B.

Reads the run_identity / selfdet result.json files produced under runs/ and
logs each as a W&B run in the PR #597 group so the identity-gate + TPS evidence
and the batch-invariance root-cause are preserved for analysis. LOCAL ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb

GROUP = "int4g128-mtp-identity-tps"
HERE = Path(__file__).resolve().parent

# Root-cause constants shared across runs (PR #597 finding).
ROOT_CAUSE = {
    "mtp_verify_is_exact_greedy": True,  # rejection_greedy_sample_kernel: accept iff draft==target argmax
    "attention_backend": "TRITON_ATTN",  # forced for Gemma4 heterogeneous head dims (256/512)
    "attention_batch_invariant": True,   # triton_unified_attention num_segments=1 under VLLM_BATCH_INVARIANT
    "lmhead_matmul_batch_invariant": True,  # aten::linear override
    "int4_marlin_gemm_batch_invariant": False,  # gptq_marlin_gemm: custom kernel, M-dependent, NOT covered
    "divergence_mechanism": (
        "int4 W4A16 Marlin GEMM is not covered by VLLM_BATCH_INVARIANT; the M=K+1 "
        "spec-verify forward uses different Marlin tiling than the M=1 AR forward, so "
        "verify logits != AR logits -> target argmax flips -> divergent decode."
    ),
}


def _log_identity(result: dict, name: str) -> None:
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=name, group=GROUP, job_type="identity-gate", reinit=True,
        config={
            "model_id": result.get("model_id"),
            "drafter": result.get("drafter"),
            "k": result.get("k"),
            "batch_invariant": result.get("batch_invariant"),
            "num_prompts": result.get("num_prompts"),
            "output_len": result.get("output_len"),
            "tau_lo": result.get("tau_lo"),
            **ROOT_CAUSE,
        },
    )
    onset = result.get("onset") or {}
    wandb.log({
        "freerun_seq_exact": result.get("freerun_seq_exact"),
        "freerun_token_identity": result.get("freerun_token_identity"),
        "num_identical": result.get("num_identical"),
        "num_divergent": result.get("num_divergent"),
        "total_tokens_compared": result.get("total_tokens_compared"),
        "total_divergent_tokens": result.get("total_divergent_tokens"),
        "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"),
        "local_decode_tps_single_stream": result.get("local_decode_tps_single_stream"),
        "official_proxy_tps": result.get("official_proxy_tps"),
        "beats_126_378": int(bool(result.get("beats_126_378"))),
        "verdict_greedy_identical": int(result.get("verdict") == "GREEDY_IDENTICAL"),
    })
    run.summary["verdict"] = result.get("verdict")
    run.finish()
    print(f"[wandb] logged identity run '{name}': seq_exact={result.get('freerun_seq_exact')}")


def _log_selfdet(result: dict, name: str) -> None:
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=name, group=GROUP, job_type="selfdet", reinit=True,
        config={
            "model_id": result.get("model_id"),
            "drafter": result.get("drafter"),
            "batch_invariant": result.get("batch_invariant"),
            "num_prompts": result.get("num_prompts"),
            "output_len": result.get("output_len"),
            "note": "spec-OFF reference self-determinism; within=pass1-vs-pass2 same server, cross=fresh server",
        },
    )
    within = result.get("within_server") or {}
    cross = result.get("cross_server") or {}
    wandb.log({
        "within_server_seq_exact": within.get("seq_exact"),
        "within_server_token_identity": within.get("token_identity"),
        "cross_server_seq_exact": cross.get("seq_exact"),
        "cross_server_token_identity": cross.get("token_identity"),
    })
    run.summary["within_verdict"] = within.get("verdict")
    run.summary["cross_verdict"] = cross.get("verdict")
    run.finish()
    print(f"[wandb] logged selfdet run '{name}': within={within.get('seq_exact')} cross={cross.get('seq_exact')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = ap.parse_args()

    for result_path in sorted(args.runs_dir.glob("*/result.json")):
        data = json.loads(result_path.read_text())
        _log_identity(data, f"fern/{data.get('label', result_path.parent.name)}")

    for selfdet_path in sorted(args.runs_dir.glob("*/selfdet_result.json")):
        data = json.loads(selfdet_path.read_text())
        _log_selfdet(data, f"fern/{data.get('label', selfdet_path.parent.name)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
