#!/usr/bin/env python3
"""PR #557 — one-arm vanilla-base serve + downstream-quality eval driver. LOCAL, NO FIRE.

Serves the STOCK gemma-4-E4B int4 checkpoint on the dev307 build under a chosen
attention-backend configuration and runs ubel #511's inspect_evals greedy harness
(run_eval.py) at client conc=32 on the byte-identical seeded item set, so every cell
is directly comparable to #542's banked cells (prompt_sha asserted, n_prompt_mismatch=0).

Arm configs (only the attention routing moves; ckpt / dtype / seqs / greedy all fixed):
  triton_default  : pure vanilla vLLM. dev307 Gemma4Config forces TRITON_ATTN for the
                    heterogeneous head dims -> the #542 broken base (re-confirm cell).
  global_fa       : VLLM_ATTENTION_BACKEND=FLASH_ATTN (explicit -> bypasses the forced
                    TRITON). Expected to FAIL at init: FA rejects the 512-dim full layers.
  fa_sliding      : FA_SLIDING=1 -> route ONLY the head_dim=256 sliding layers to
                    FLASH_ATTN; full (512) layers keep TRITON. The healthy denominator.
  surgical_attn   : fa_sliding + SURGICAL_ATTN_USE_3D_OFF=1 (2D order-preserving path on
                    the TRITON full layers). base_fullhead's exact attention, no speed stack.

Same server flags as #542's start_base_server: VLLM_USE_FLASHINFER_SAMPLER=0,
--max-num-seqs 16 (ubel #511 reference serve), greedy temp=0, max-model-len 4096.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/vanilla_base_serve_regression"
QE = ROOT / "research/validity/downstream_quality_eval"
SERVE_INJECT = HERE / "serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")  # dev307 build
EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")                  # ubel inspect client
STOCK = "/tmp/gemma4-e4b-qat-w4a16-ct"
PORT = 8000
SEED = 12345

ARM_ENV = {
    "triton_default": {},
    "global_fa": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
    "fa_sliding": {"FA_SLIDING": "1"},
    "surgical_attn": {"FA_SLIDING": "1", "SURGICAL_ATTN_USE_3D_OFF": "1"},
    # ROOT-CAUSE FIX arm: plain vanilla serve (no MTP/2D/splitkv/onegraph) + ONLY the
    # vLLM-native PLE embed-scale fold. dev307's gemma4.py get_per_layer_inputs dropped
    # the runtime ×16 (sqrt(256)) per-layer-embedding scale and gated it behind this env
    # (model_loader/utils.py:130); a vanilla serve without it feeds 16×-too-small
    # per-layer embeddings into every decoder layer -> long-CoT degeneration. This arm is
    # the recovered healthy denominator.
    "ple_fold": {"PLE_FOLD_EMBED_SCALE": "1", "PLE_FOLD_TARGET_MODEL": STOCK},
}


def wait_gpu_free(threshold_mib=1500, timeout_s=180):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            used = max(int(x) for x in out.split())
            if used < threshold_mib:
                print(f"[gpu] free ({used} MiB) — proceeding", flush=True)
                return
            print(f"[gpu] waiting for release: {used} MiB used", flush=True)
        except Exception as exc:
            print(f"[gpu] nvidia-smi probe failed: {exc!r}", flush=True)
        time.sleep(5)
    print(f"[gpu] WARN: still busy after {timeout_s}s — proceeding anyway", flush=True)


def start_server(arm: str, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # serve_inject FIRST so OUR sitecustomize wins (prometheus shim + gated attn patches).
    env["PYTHONPATH"] = str(SERVE_INJECT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env["PR557_PATCH_DIR"] = str(SUBMISSION)
    for k, v in ARM_ENV[arm].items():
        env[k] = v
    cmd = [
        str(SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", STOCK, "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", "16",
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", '{"temperature":0.0,"top_p":1.0,"top_k":0}',
    ]
    flags = {k: env.get(k) for k in ("FA_SLIDING", "SURGICAL_ATTN_USE_3D_OFF",
                                     "VLLM_ATTENTION_BACKEND", "PLE_FOLD_EMBED_SCALE")}
    print(f"[serve] arm={arm} flags={ {k: v for k, v in flags.items() if v} }", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def wait_ready(proc: subprocess.Popen, timeout_s=1200) -> None:
    base = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early code={proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("endpoint not ready")


def run_cell(arm: str, task: str, out: Path, *, n=None, max_tokens=2048, conc=32, limit=0) -> dict:
    cmd = [
        str(EVAL_PY), str(QE / "run_eval.py"), "--task", task, "--arm", arm,
        "--out", str(out), "--seed", str(SEED), "--max-tokens", str(max_tokens),
        "--max-connections", str(conc), "--base-url", f"http://127.0.0.1:{PORT}/v1",
        "--model", "gemma-4-e4b-it",
    ]
    if task == "mmlu_pro":
        cmd += ["--n", str(n)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[eval] {arm}/{task} limit={limit or 'full'} ->", " ".join(cmd[-6:]), flush=True)
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] {arm}/{task} acc={d['accuracy']:.4f} scored={d['n_scored']} "
          f"correct={d['n_correct']} empty={d.get('n_empty')} err={d['n_error']}", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_ENV))
    ap.add_argument("--mmlu-n", type=int, default=500)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap each axis to first N")
    ap.add_argument("--tasks", default="mmlu_pro,gpqa_diamond")
    ap.add_argument("--tag", default="", help="filename suffix")
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    suffix = (f".{args.tag}" if args.tag else "") + (".smoke" if args.limit else "")
    log = HERE / f"server_{args.arm}{suffix}.log"

    wait_gpu_free()
    proc = start_server(args.arm, log)
    results = {}
    try:
        wait_ready(proc)
        print(f"[driver] arm={args.arm} READY", flush=True)
        for task in args.tasks.split(","):
            task = task.strip()
            mt = 2048 if task == "mmlu_pro" else 3072
            out = HERE / f"{args.arm}_{task}{suffix}.json"
            results[task] = run_cell(args.arm, task, out, n=args.mmlu_n,
                                     max_tokens=mt, conc=args.conc, limit=args.limit)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    print(f"[driver] arm={args.arm} COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
