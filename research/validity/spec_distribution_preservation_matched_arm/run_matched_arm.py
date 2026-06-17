#!/usr/bin/env python
"""PR #620 — matched-arm spec-vs-AR downstream-quality driver.

Tests whether int4+MTP-K7 speculative decoding is DISTRIBUTION-PRESERVING under
SAMPLED decoding (T=1.0/top_p=0.95/top_k=64, lewtun #31). Two arms on the SAME
int4 body substrate (/workspace/gemma_build/int4_g128_lmhead), served via the SAME
submission (int4_mtp_batchinv, VLLM_BATCH_INVARIANT=1, MAX_MODEL_LEN=6144):

  * spec arm: NUM_SPECULATIVE_TOKENS=7, drafter /tmp/qat-assistant (fern #597's
    427.7 official-proxy candidate).
  * ar   arm: NUM_SPECULATIVE_TOKENS=0 -> plain int4 M=1 AR (drafter OFF). Per the
    submission serve.py docstring this is "the exact reference; reference and
    candidate then differ ONLY in the drafter" -> one-variable matched arm.

By the rejection-sampling theorem (Leviathan 2023 / Chen 2023) the spec arm samples
EXACTLY from the target (int4-body, top_k/top_p-truncated) distribution, so spec
accuracy should equal AR accuracy within sampling noise. This driver produces the
paired per-item evidence; analyze_matched_arm.py does the McNemar + cluster-bootstrap.

ANALYSIS-ONLY. Local A10G. NO HF Job, NO submission, NO served-file change.
analysis_only=True, official_tps=0.

Idempotent: skips any (arm,eval,seed) whose output JSON already exists, so a crashed
run resumes cleanly. Writes each run to results/ as it completes.

Usage:
  run_matched_arm.py --arms spec,ar --mode full
  run_matched_arm.py --arms spec     --mode smoke   # tiny limit, 1 seed
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"

# vLLM 0.22.0 submission stack (lawine #606: dev307 is NOT a faithful proxy).
SERVER_PY = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")
EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")

BODY = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER = "/tmp/qat-assistant"
PORT = 8000
MODEL = "gemma-4-e4b-it"

# 5 dataset (choice-shuffle) seeds for GPQA: de-biases position AND gives 5 paired
# layouts per question. First 3 reused from stark #605 / kanna #610 for continuity.
GPQA_SEEDS = [12345, 23456, 34567, 45678, 56789]
# 5 sampling seeds for GSM8K (fixed 500-subset+fewshot at --seed 1234, vary the draw).
GSM8K_SAMPLING_SEEDS = [1234, 2345, 3456, 4567, 5678]


def _spec_env(arm: str, max_num_seqs: int) -> dict[str, str]:
    """extra_env for the int4_mtp_batchinv submission. Toggles ONLY speculation."""
    num_spec = 7 if arm == "spec" else 0
    return {
        "MODEL_ID": BODY,
        "DRAFTER_MODEL": DRAFTER,
        "NUM_SPECULATIVE_TOKENS": str(num_spec),
        "VLLM_BATCH_INVARIANT": "1",  # M=K+1 verify numerically == M=1 (fern #597)
        "MAX_MODEL_LEN": "6144",       # land #598
        "MAX_NUM_SEQS": str(max_num_seqs),
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # sampled top_k/top_p: avoid curand JIT
    }


def _run(cmd: list[str], log_path: Path, tag: str) -> int:
    print(f"[driver] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[driver] {tag} rc={rc} wall={time.time()-t0:.0f}s {time.strftime('%H:%M:%S')}", flush=True)
    return rc


def run_gpqa(arm: str, seed: int, max_tokens: int, limit: int, conns: int) -> None:
    out = RES / f"{arm}_gpqa_s{seed}.json"
    if out.exists():
        print(f"[driver] SKIP existing {out.name}", flush=True)
        return
    log_dir = RES / f"_inspect_{arm}_gpqa_s{seed}"
    cmd = [
        str(EVAL_PY), str(ROOT / "research/validity/downstream_quality_eval/run_eval.py"),
        "--task", "gpqa_diamond", "--arm", arm, "--out", str(out),
        "--seed", str(seed), "--sampling-seed", str(seed),
        "--max-tokens", str(max_tokens), "--max-connections", str(conns),
        "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64", "--min-tokens", "8",
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", MODEL,
        "--log-dir", str(log_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    _run(cmd, RES / f"_{arm}_gpqa_s{seed}.out", f"gpqa arm={arm} seed={seed}")


def run_gsm8k(arm: str, sseed: int, n: int, conns: int) -> None:
    out_dir = RES / f"gsm8k_{arm}_ss{sseed}"
    # gsm8k_eval writes {label}_{regime}_s{sampling_seed}.json when --sampling-seed set.
    out = out_dir / f"{arm}_sampled_s{sseed}.json"
    if out.exists():
        print(f"[driver] SKIP existing {out.relative_to(RES)}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EVAL_PY), str(ROOT / "research/downstream_quality_gsm8k/gsm8k_eval.py"),
        "--base-url", f"http://127.0.0.1:{PORT}", "--model", MODEL,
        "--label", arm, "--regimes", "sampled",
        "--n", str(n), "--n-shot", "8", "--seed", "1234", "--sampling-seed", str(sseed),
        "--top-p", "0.95", "--top-k", "64", "--max-tokens", "512", "--min-tokens", "8",
        "--concurrency", str(conns), "--max-num-seqs", str(conns),
        "--out-dir", str(out_dir),
    ]
    _run(cmd, RES / f"_{arm}_gsm8k_ss{sseed}.out", f"gsm8k arm={arm} sseed={sseed}")


def serve_and_eval(arm: str, mode: str) -> None:
    smoke = mode == "smoke"
    max_tokens = 4096            # PR guard #590 (>=4096); 6144 model-len admits it
    conns = 16
    gpqa_seeds = GPQA_SEEDS[:1] if smoke else GPQA_SEEDS
    gsm_seeds = GSM8K_SAMPLING_SEEDS[:1] if smoke else GSM8K_SAMPLING_SEEDS
    gpqa_limit = 4 if smoke else 0
    gsm_n = 16 if smoke else 500

    extra_env = _spec_env(arm, conns)
    log_path = RES / f"_serve_{arm}.log"
    print(f"[driver] === ARM {arm} (mode={mode}) === {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[driver] serve env: {extra_env}", flush=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=PORT,
        log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        print(f"[driver] {arm} ready at {srv.base_url} model={srv.served_model_name}", flush=True)
        for seed in gpqa_seeds:
            run_gpqa(arm, seed, max_tokens, gpqa_limit, conns)
        for sseed in gsm_seeds:
            run_gsm8k(arm, sseed, gsm_n, conns)
    print(f"[driver] === ARM {arm} DONE === {time.strftime('%H:%M:%S')}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="spec,ar")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm:
            serve_and_eval(arm, args.mode)
    print(f"[driver] ALL ARMS COMPLETE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
