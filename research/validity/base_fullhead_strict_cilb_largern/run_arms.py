#!/usr/bin/env python3
"""PR #564 — base_fullhead vs ple_fold downstream quality at LARGER n. LOCAL, NO FIRE.

Settles the #557 strict-Wilson-CI-lb miss (MMLU near-miss 0.593 vs 0.596, GPQA miss
0.401 vs 0.436 at n=500/198) by re-measuring BOTH arms at larger MMLU-Pro n under the
EXACT #542/#557 greedy harness (run_eval.py, ubel #511), then recomputing whether
base_fullhead's Wilson CI-lb clears 0.90 x the ple_fold point once the CIs tighten.

Two arms, dev307 build, seqs=16, greedy temp=0, byte-identical seeded item set. ONLY
the attention reduction path moves between them (both carry the PLE embed-scale fold,
the #557 correctness gate):

  base_fullhead : FA_SLIDING=1 + SURGICAL_ATTN_USE_3D_OFF=1 + PLE_FOLD_EMBED_SCALE=1.
                  The submission's exact surgical 2D order-preserving attention (the
                  #557 `surgical_attn` arm) PLUS the fold -> the healthy base_fullhead
                  quality config (#542 measured 0.636/0.4697 with this attention; the
                  MTP/split-KV/onegraph speed stack is greedy-identity-preserving and
                  batch-invariant, so seqs=16 no-MTP reproduces the same greedy tokens
                  while being fast enough for larger n). The quality numerator.
  ple_fold      : PLE_FOLD_EMBED_SCALE=1 ONLY (plain vanilla 3D-default TRITON serve).
                  The #557 recovered healthy denominator (0.662/0.4848). MUST set the
                  fold or it is the broken 0.43/0.31 base (#557 root cause).

GPQA-Diamond is the FULL 198-item set in both #542/#557 and here -> its Wilson CI width
is fixed at the dataset ceiling and CANNOT be tightened by larger n. The larger-n lever
is MMLU-Pro only (12032-item pool). That asymmetry is a reported finding, not a bug.

Reuses #557's serve_inject (sitecustomize: prometheus shim + gated surgical-attn patches)
and PR557_PATCH_DIR=submissions/fa2sw_strict_surgical357 so the surgical attention arms.
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
HERE = ROOT / "research/validity/base_fullhead_strict_cilb_largern"
QE = ROOT / "research/validity/downstream_quality_eval"
# Reuse the #557 serve_inject (gated surgical-attn + prometheus shim) and submission patches.
SERVE_INJECT = ROOT / "research/validity/vanilla_base_serve_regression/serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")  # dev307 build
EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")                  # ubel #511 inspect client
STOCK = "/tmp/gemma4-e4b-qat-w4a16-ct"                             # stock int4, native 262k head
PORT = 8000
SEED = 12345

ARM_ENV = {
    # base_fullhead = surgical 2D attention (the submission's exact attention path) + fold.
    "base_fullhead": {
        "FA_SLIDING": "1",
        "SURGICAL_ATTN_USE_3D_OFF": "1",
        "PLE_FOLD_EMBED_SCALE": "1",
        "PLE_FOLD_TARGET_MODEL": STOCK,
    },
    # ple_fold = plain vanilla serve (default 3D TRITON) + ONLY the embed-scale fold.
    "ple_fold": {
        "PLE_FOLD_EMBED_SCALE": "1",
        "PLE_FOLD_TARGET_MODEL": STOCK,
    },
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
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu] nvidia-smi probe failed: {exc!r}", flush=True)
        time.sleep(5)
    print(f"[gpu] WARN: still busy after {timeout_s}s — proceeding anyway", flush=True)


def start_server(arm: str, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # serve_inject FIRST so our sitecustomize wins (prometheus shim + gated attn patches).
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
                                     "PLE_FOLD_EMBED_SCALE", "VLLM_ATTENTION_BACKEND")}
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
    print(f"[eval] {arm}/{task} n={n if task=='mmlu_pro' else 198} limit={limit or 'full'} START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] {arm}/{task} acc={d['accuracy']:.4f} scored={d['n_scored']} "
          f"correct={d['n_correct']} empty={d.get('n_empty')} err={d['n_error']} "
          f"dt={time.time()-t0:.0f}s", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base_fullhead,ple_fold")
    ap.add_argument("--mmlu-n", type=int, default=1500)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--tasks", default="mmlu_pro,gpqa_diamond")
    ap.add_argument("--smoke", action="store_true", help="limit each axis to a few items, then exit")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    suffix = ".smoke" if (args.smoke or args.limit) else ""
    lim_mmlu = (8 if args.smoke else args.limit)
    lim_gpqa = (5 if args.smoke else args.limit)

    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm not in ARM_ENV:
            raise SystemExit(f"unknown arm {arm!r}; choices={list(ARM_ENV)}")
        log = HERE / f"server_{arm}{suffix}.log"
        wait_gpu_free()
        proc = start_server(arm, log)
        try:
            wait_ready(proc)
            print(f"[driver] arm={arm} READY {time.strftime('%H:%M:%S')}", flush=True)
            for task in args.tasks.split(","):
                task = task.strip()
                mt = 2048 if task == "mmlu_pro" else 3072
                lim = lim_mmlu if task == "mmlu_pro" else lim_gpqa
                out = HERE / f"{arm}_{task}{suffix}.json"
                run_cell(arm, task, out, n=args.mmlu_n, max_tokens=mt, conc=args.conc, limit=lim)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except Exception:
                pass
            try:
                proc.wait(timeout=60)
            except Exception:
                pass
        print(f"[driver] arm={arm} COMPLETE {time.strftime('%H:%M:%S')}", flush=True)

    print(f"[driver] ALL ARMS COMPLETE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
