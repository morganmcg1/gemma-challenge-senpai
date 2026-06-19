#!/usr/bin/env python3
"""Served quality dossier for the as-fired int4+MTP-spec-dec fire (PR #753/#757).

Serves ONE arm on the local A10G through the real serve.py / vLLM 0.22.0
api_server, then runs the MMLU-Pro / GSM8K / AIME panel against the live endpoint
under the lewtun #31 downstream sampling protocol (generation_config.json:
T=1.0, top_p=0.95, top_k=64) + the #541 min_tokens=8 EOS-guard.

Three arms, identical eval knobs / vLLM 0.22.0 stack / dtype=bf16:

  * --arm fire  : submissions/int4_mtp_batchinv as-fired (int4 W4A16 target,
                  drafter ON, NUM_SPECULATIVE_TOKENS=6).
  * --arm base  : SAME int4 submission, SENPAI_REFERENCE_MODE=1 forces
                  num_speculative_tokens=0 (drafter OFF, plain int4 M=1 AR).
  * --arm bf16  : full-precision bf16 denominator -- MODEL_ID overridden to the
                  original released instruct model (google/gemma-4-E4B-it,
                  --bf16-model), drafter OFF, served through the SAME serve.py
                  (native bf16, no Marlin). This is the "% of the original
                  full-precision model" denominator the blog wants.

fire/base differ ONLY in the speculative drafter (specdec_factor = fire/base);
base/bf16 differ ONLY in W4A16+QAT quantization (int4_quant_factor = base/bf16);
the product gives fire_pct_of_bf16 = the complete %-of-original retained by the
fast submission.

One server load drives all three evals via --base-url, so one arm at a time (the
16 GB bf16 weights need more KV headroom than the ~4 GB int4 target -- use
--gpu-mem-util / a smaller --max-num-seqs if the bf16 arm OOMs). MAX_NUM_SEQS is
raised for eval tractability; VLLM_BATCH_INVARIANT=1 (set by the manifest) +
per-request seeds keep each request's decode batch-invariant, so the score is
unchanged by batch size and the arms are matched. VLLM_USE_FLASHINFER_SAMPLER=0
selects the torch-native top-k/top-p sampler (this box's CUDA toolkit ships no
curand.h for the flashinfer JIT sampler; native sampling is numerically standard).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness  # noqa: E402

# Eval-client python: has inspect_ai/inspect_evals (MMLU-Pro) + stdlib (gsm8k/aime).
EVAL_PY = "/tmp/eval-serve-venv/bin/python"
SUBMISSION = ROOT / "submissions/int4_mtp_batchinv"
GSM8K = ROOT / "research/downstream_quality_gsm8k/gsm8k_eval.py"
AIME = ROOT / "research/downstream_quality_aime/aime_eval.py"
MMLU = ROOT / "research/validity/downstream_quality_eval/run_eval.py"


def smoke_completion(base: str) -> str:
    payload = {
        "model": "gemma-4-e4b-it",
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 64,
        "max_tokens": 16, "min_tokens": 8, "seed": 1234, "stream": False,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"]


def run(cmd: list[str], log: Path) -> int:
    print(f"[dossier] $ {' '.join(cmd)}\n[dossier]   -> log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[dossier] rc={rc} wall={time.time()-t0:.0f}s", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["fire", "base", "bf16"])
    ap.add_argument("--smoke", action="store_true", help="tiny limits to de-risk the stack")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--bf16-model", default="google/gemma-4-E4B-it",
                    help="full-precision bf16 denominator for --arm bf16: the original "
                         "released instruct model the int4-QAT submission descends from")
    ap.add_argument("--gpu-mem-util", default=None,
                    help="override GPU_MEMORY_UTILIZATION (e.g. 0.92); the 16 GB bf16 "
                         "weights need more KV headroom than the ~4 GB int4 target")
    ap.add_argument("--mmlu-n", type=int, default=250)
    ap.add_argument("--gsm8k-n", type=int, default=300)
    ap.add_argument("--aime-years", default="2024")
    ap.add_argument("--aime-client-concurrency", type=int, default=None,
                    help="client_concurrency for the AIME phase (default: --max-num-seqs). "
                         "AIME issues n=k samples PER request, so client_concurrency*k decode "
                         "sequences compete for MAX_NUM_SEQS server slots. On a slow arm "
                         "(native bf16, no Marlin) client_concurrency=max_num_seqs over-"
                         "subscribes k-fold (16*8=128 seqs for 16 slots) and a single request's "
                         "wait can exceed --aime-request-timeout-s. Set 2 so 2*8=16 == "
                         "max_num_seqs (exactly fills the server, ~1-wave per-request latency). "
                         "Score is client_concurrency-invariant (VLLM_BATCH_INVARIANT=1), so "
                         "this is a pure dispatch knob -- the arm stays matched to fire/base.")
    ap.add_argument("--aime-request-timeout-s", type=int, default=1200,
                    help="per-request client timeout for the AIME phase; slow arms need headroom.")
    ap.add_argument("--tasks", default="gsm8k,mmlu,aime",
                    help="comma list subset of {gsm8k,mmlu,aime}; AIME last (most expensive)")
    ap.add_argument("--outdir", default=str(ROOT / "research/fire_served_quality_dossier"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}{'_smoke' if args.smoke else ''}"
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    harness.ensure_serving_http_compat(Path(server_python))

    extra_env = {
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # The container maps the A10G as NVML index 0, but the host-level
        # CUDA_VISIBLE_DEVICES is inherited as 4 -> vLLM's NVML lookup raises
        # NVMLError_InvalidArgument. Pin to the container-local index (matches
        # research/validity/downstream_quality_eval/start_server.sh).
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if args.arm == "base":
        extra_env["SENPAI_REFERENCE_MODE"] = "1"  # drafter OFF -> matched denominator
    if args.arm == "bf16":
        # Full-precision bf16 denominator: serve the original released instruct
        # model through the SAME vLLM 0.22.0 api_server / serve.py / dtype=bf16
        # stack, drafter OFF. Overriding MODEL_ID is enough -- the bf16 checkpoint
        # carries no quantization_config, so vLLM loads native bf16 (no Marlin);
        # the spec-decode attn-group patch is a no-op with the drafter off.
        extra_env["SENPAI_REFERENCE_MODE"] = "1"  # drafter OFF
        extra_env["MODEL_ID"] = args.bf16_model
    if args.gpu_mem_util:
        extra_env["GPU_MEMORY_UTILIZATION"] = str(args.gpu_mem_util)

    server_log = outdir / f"server_{tag}.log"
    print(f"[dossier] arm={args.arm} smoke={args.smoke} extra_env={extra_env}", flush=True)
    print(f"[dossier] server_python={server_python} log={server_log}", flush=True)

    t_load = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=server_python, port=8000,
        startup_timeout_s=1800, log_path=server_log, extra_env=extra_env,
    ) as srv:
        base = srv.base_url  # http://127.0.0.1:8000
        print(f"[dossier] server ready in {time.time()-t_load:.0f}s at {base}", flush=True)
        print(f"[dossier] SMOKE sampled completion: {smoke_completion(base)!r}", flush=True)

        limit_mmlu = ["--limit", "3"] if args.smoke else []
        limit_gsm = ["--limit", "3"] if args.smoke else []
        limit_aime = ["--limit", "3"] if args.smoke else []
        rcs = {}

        if "gsm8k" in tasks:
            rcs["gsm8k"] = run([
                EVAL_PY, str(GSM8K), "--base-url", base, "--model", "gemma-4-e4b-it",
                "--label", tag, "--regimes", "sampled",
                "--n", str(args.gsm8k_n), "--seed", "1234",
                "--top-p", "0.95", "--top-k", "64", "--max-tokens", "512",
                "--min-tokens", "8", "--concurrency", str(args.max_num_seqs),
                "--max-num-seqs", str(args.max_num_seqs),
                "--out-dir", str(outdir), *limit_gsm,
            ], outdir / f"_eval_gsm8k_{tag}.log")

        if "mmlu" in tasks:
            rcs["mmlu"] = run([
                EVAL_PY, str(MMLU), "--task", "mmlu_pro", "--arm", tag,
                "--out", str(outdir / f"{tag}_mmlu_pro.json"),
                "--n", str(args.mmlu_n), "--seed", "12345", "--max-tokens", "2048",
                "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
                "--min-tokens", "8", "--max-connections", str(args.max_num_seqs),
                "--base-url", base.rstrip("/") + "/v1", "--model", "gemma-4-e4b-it",
                "--log-dir", str(outdir / f"_inspect_{tag}"), *limit_mmlu,
            ], outdir / f"_eval_mmlu_{tag}.log")

        if "aime" in tasks:
            aime_cc = args.aime_client_concurrency or args.max_num_seqs
            rcs["aime"] = run([
                EVAL_PY, str(AIME), "--base-url", base, "--model", "gemma-4-e4b-it",
                "--label", tag, "--years", args.aime_years, "--k", "8",
                "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
                "--max-tokens", "3072", "--min-tokens", "8", "--seed", "1234",
                "--no-thinking", "--max-num-seqs", str(args.max_num_seqs),
                "--client-concurrency", str(aime_cc),
                "--request-timeout-s", str(args.aime_request_timeout_s),
                "--out", str(outdir / f"{tag}_aime.json"), *limit_aime,
            ], outdir / f"_eval_aime_{tag}.log")

    print(f"[dossier] DONE arm={args.arm} rcs={rcs}", flush=True)
    return 0 if all(v == 0 for v in rcs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
