#!/usr/bin/env python3
"""PR #762 wirbel -- non-strict (BI=0) rung downstream-quality dossier.

Prices the missing leg of the strict/non-strict fork (fern #750): the BI=0
non-strict rung is +73 TPS faster than the BI=1 strict fire (229.85 vs 156.95
anchored) and PPL-parity (2.0056 vs 2.0057, PR #751), so the ONLY thing it gives
up is the internal literal-#319 byte-exact bar (NOT a DQ -- the organizer scorer
is identity-blind). This card measures whether that fast path is also
*downstream-quality-equivalent* to the strict fire.

Reproduces the ubel #753 served-quality panel (MMLU-Pro / GSM8K / AIME maj@8)
under the lewtun #31 sampled protocol (generation_config.json: T=1.0, top_p=0.95,
top_k=64) + the #541 min_tokens=8 EOS-guard, on the real serve.py / vLLM 0.22.0
api_server (the 0.22.0 accuracy engine; lawine #606 invalidates dev307 accuracy).

Three arms, identical checkpoint / kernels / drafter -- the only moved variable
per comparison is named:

  * bi1_fire     submissions/int4_mtp_batchinv      VLLM_BATCH_INVARIANT=1, drafter ON
  * bi0_nonstrict submissions/int4_mtp_bi0_surgattn VLLM_BATCH_INVARIANT=0 + surgical
                  force-2D TRITON_ATTN patch, drafter ON
  * int4_base    submissions/int4_mtp_batchinv + SENPAI_REFERENCE_MODE=1
                  (num_speculative_tokens forced to 0 -> plain int4 M=1 AR): the
                  matched %-of-base denominator, exactly as ubel #753.

One server load drives all three evals via --base-url (the int4 target + KV cache
fills the A10G, so one arm at a time). MAX_NUM_SEQS is raised to 16 for eval
tractability and VLLM_USE_FLASHINFER_SAMPLER=0 selects the torch-native sampler
(this box's CUDA toolkit ships no curand.h for the flashinfer JIT sampler) --
both exactly as run_dossier.py so bi1_fire/int4_base reproduce ubel #753.

CAVEAT (bi0_nonstrict only): AIME/MMLU/GSM8K run client-concurrency 16; under
VLLM_BATCH_INVARIANT=1 (bi1_fire, int4_base) batched decode == single-stream
decode so the score is batch-size-invariant. bi0_nonstrict is batch-NON-invariant
on the non-attention GEMMs/RMSNorm/softmax, so its concurrency-16 score may carry
a small batch-dependent component the single-stream served path (MAX_NUM_SEQS=1,
TPS=229.85) would not. This can only ADD noise to bi0, so equivalence measured
here holds a fortiori for the single-stream served rung. LOCAL ONLY: no HF Job,
no served-file change, no submission, no --launch.
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
GSM8K = ROOT / "research/downstream_quality_gsm8k/gsm8k_eval.py"
AIME = ROOT / "research/downstream_quality_aime/aime_eval.py"
MMLU = ROOT / "research/validity/downstream_quality_eval/run_eval.py"

# arm -> (submission dir, reference_mode). reference_mode=True forces the drafter
# OFF (plain int4 M=1 AR) on the batchinv submission = the matched denominator.
ARMS = {
    "bi1_fire": (ROOT / "submissions/int4_mtp_batchinv", False),
    "bi0_nonstrict": (ROOT / "submissions/int4_mtp_bi0_surgattn", False),
    "int4_base": (ROOT / "submissions/int4_mtp_batchinv", True),
}


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
    print(f"[panel] $ {' '.join(cmd)}\n[panel]   -> log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[panel] rc={rc} wall={time.time()-t0:.0f}s", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--smoke", action="store_true", help="tiny limits to de-risk the stack")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--mmlu-n", type=int, default=250)
    ap.add_argument("--gsm8k-n", type=int, default=300)
    ap.add_argument("--aime-years", default="2024")
    ap.add_argument("--tasks", default="gsm8k,mmlu,aime",
                    help="comma list subset of {gsm8k,mmlu,aime}; AIME last (most expensive)")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "out"))
    args = ap.parse_args()

    submission, reference_mode = ARMS[args.arm]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}{'_smoke' if args.smoke else ''}"
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    harness.ensure_serving_http_compat(Path(server_python))

    extra_env = {
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Container maps the A10G as NVML index 0 but host CUDA_VISIBLE_DEVICES is
        # inherited as 4 -> vLLM NVML lookup raises; pin to the container-local idx.
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if reference_mode:
        extra_env["SENPAI_REFERENCE_MODE"] = "1"  # drafter OFF -> matched denominator

    server_log = outdir / f"server_{tag}.log"
    print(f"[panel] arm={args.arm} submission={submission.name} reference_mode={reference_mode} "
          f"smoke={args.smoke} extra_env={extra_env}", flush=True)
    print(f"[panel] server_python={server_python} log={server_log}", flush=True)

    t_load = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=8000,
        startup_timeout_s=1800, log_path=server_log, extra_env=extra_env,
    ) as srv:
        base = srv.base_url  # http://127.0.0.1:8000
        print(f"[panel] server ready in {time.time()-t_load:.0f}s at {base}", flush=True)
        print(f"[panel] SMOKE sampled completion: {smoke_completion(base)!r}", flush=True)

        limit = ["--limit", "3"] if args.smoke else []
        rcs = {}

        if "gsm8k" in tasks:
            rcs["gsm8k"] = run([
                EVAL_PY, str(GSM8K), "--base-url", base, "--model", "gemma-4-e4b-it",
                "--label", tag, "--regimes", "sampled",
                "--n", str(args.gsm8k_n), "--seed", "1234",
                "--top-p", "0.95", "--top-k", "64", "--max-tokens", "512",
                "--min-tokens", "8", "--concurrency", str(args.max_num_seqs),
                "--max-num-seqs", str(args.max_num_seqs),
                "--out-dir", str(outdir), *limit,
            ], outdir / f"_eval_gsm8k_{tag}.log")

        if "mmlu" in tasks:
            rcs["mmlu"] = run([
                EVAL_PY, str(MMLU), "--task", "mmlu_pro", "--arm", tag,
                "--out", str(outdir / f"{tag}_mmlu_pro.json"),
                "--n", str(args.mmlu_n), "--seed", "12345", "--max-tokens", "2048",
                "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
                "--min-tokens", "8", "--max-connections", str(args.max_num_seqs),
                "--base-url", base.rstrip("/") + "/v1", "--model", "gemma-4-e4b-it",
                "--log-dir", str(outdir / f"_inspect_{tag}"), *limit,
            ], outdir / f"_eval_mmlu_{tag}.log")

        if "aime" in tasks:
            rcs["aime"] = run([
                EVAL_PY, str(AIME), "--base-url", base, "--model", "gemma-4-e4b-it",
                "--label", tag, "--years", args.aime_years, "--k", "8",
                "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
                "--max-tokens", "3072", "--min-tokens", "8", "--seed", "1234",
                "--no-thinking", "--max-num-seqs", str(args.max_num_seqs),
                "--client-concurrency", str(args.max_num_seqs),
                "--out", str(outdir / f"{tag}_aime.json"), *limit,
            ], outdir / f"_eval_aime_{tag}.log")

    print(f"[panel] DONE arm={args.arm} rcs={rcs}", flush=True)
    return 0 if all(v == 0 for v in rcs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
