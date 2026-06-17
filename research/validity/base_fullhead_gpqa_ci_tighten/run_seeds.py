#!/usr/bin/env python3
"""PR #589 — base_fullhead GPQA-Diamond CI-tighten under the SAMPLING protocol. LOCAL, NO FIRE.

Settles whether base_fullhead's GPQA-D gate margin (point 0.4798 vs the >=0.471 bar, +0.009,
the thinnest of the 4 quality gates) is robust once measured under the actual downstream
SAMPLING protocol (generation_config.json: do_sample=True temp=1.0 top_p=0.95 top_k=64;
lewtun #31) across MANY seeds, with min_tokens=8 (wirbel #541) EOS-guard.

Serves base_fullhead ONCE on the faithful #564 surgical+fold stack (dev307 build +
serve_inject sitecustomize: prometheus shim + FA_SLIDING flash routing + surgical 2D
order-preserving attention), checkpoint = the local int4-W4A16-g32 QAT snapshot (native 262k
bf16 lm_head -- lm_head is in the quant `ignore` list), then loops run_eval.py GPQA-Diamond
(full 198) over sampling seeds. Server stays up across seeds (serve-once / eval-many).

NO HF FIRE: analysis_only, official_tps=0. Only the assigned local A10G.
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
HERE = ROOT / "research/validity/base_fullhead_gpqa_ci_tighten"
RES = HERE / "results"
QE = ROOT / "research/validity/downstream_quality_eval"
SERVE_INJECT = ROOT / "research/validity/vanilla_base_serve_regression/serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")   # dev307 build (#564)
EVAL_PY = Path("/tmp/land-inspect/bin/python")                      # land inspect client (0.3.240/0.14.0)
# local int4-W4A16-g32 QAT snapshot, native 262k head (lm_head NOT quantized) = base_fullhead body
STOCK = Path(
    os.environ.get(
        "STOCK_CKPT",
        str(Path.home()
            / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct"
              "/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"),
    )
)
PORT = int(os.environ.get("PORT", "8000"))
DATASET_SEED = 12345  # #547/#563 byte-identical GPQA-Diamond choice-shuffle seed
# generation_config.json (lewtun #31) sampling protocol
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
GPQA_MAX_TOKENS = 3072
MIN_TOKENS = 8  # wirbel #541 EOS-guard (proven mechanical no-op on GPQA CoT by #563)

# base_fullhead surgical+fold serve env (#564 ARM_ENV["base_fullhead"]).
ARM_ENV = {
    "FA_SLIDING": "1",
    "SURGICAL_ATTN_USE_3D_OFF": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": str(STOCK),
}


def start_server(log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # curand-less box: force native torch sampler
    env["PYTHONPATH"] = str(SERVE_INJECT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env["PR557_PATCH_DIR"] = str(SUBMISSION)
    env.update(ARM_ENV)
    cmd = [
        str(SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(STOCK), "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", "16",
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", json.dumps(SAMPLING),
    ]
    print(f"[serve] base_fullhead surgical+fold flags={ARM_ENV} stock={STOCK}", flush=True)
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


def run_seed(seed: int, out: Path, limit: int = 0) -> dict:
    cmd = [
        str(EVAL_PY), str(QE / "run_eval.py"),
        "--task", "gpqa_diamond", "--arm", "base_fullhead",
        "--out", str(out), "--seed", str(DATASET_SEED),
        "--max-tokens", str(GPQA_MAX_TOKENS), "--max-connections", "16",
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", "gemma-4-e4b-it",
        "--temperature", str(SAMPLING["temperature"]),
        "--top-p", str(SAMPLING["top_p"]), "--top-k", str(SAMPLING["top_k"]),
        "--sampling-seed", str(seed), "--min-tokens", str(MIN_TOKENS),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[eval] seed={seed} limit={limit or 'full'} START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] seed={seed} acc={d['accuracy']:.4f} scored={d['n_scored']} "
          f"correct={d['n_correct']} empty={d.get('n_empty')} err={d['n_error']} "
          f"dt={time.time()-t0:.0f}s", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--smoke", action="store_true", help="limit to 3 items on seed 0, then exit")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    log = HERE / ("server_smoke.log" if args.smoke else "server.log")
    proc = start_server(log)
    try:
        wait_ready(proc)
        print(f"[driver] READY {time.strftime('%H:%M:%S')}", flush=True)
        if args.smoke:
            run_seed(0, RES / "_smoke_gpqa_s0.json", limit=3)
            print("[driver] SMOKE OK", flush=True)
            return 0
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        for s in seeds:
            out = RES / f"bf_gpqa_sampled_mt8_s{s}.json"
            if out.exists():
                print(f"[driver] seed={s} SKIP existing {out.name}", flush=True)
                continue
            run_seed(s, out)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    print(f"[driver] ALL SEEDS COMPLETE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
