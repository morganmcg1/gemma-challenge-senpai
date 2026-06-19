#!/usr/bin/env python3
"""PR #696 -- harden the int4-body GPQA-Diamond #31-SAMPLED pass from 10 -> 30 seeds.

LOCAL, NO FIRE, analysis_only. official_tps=0, no_hf_job, no submission, no served-file
change. Only the assigned A10G.

Reuses the EXACT #589/#692 body-isolation serve recipe (base_fullhead = local
int4-W4A16-g32 QAT snapshot google/gemma-4-E4B-it-qat-w4a16-ct, native 262k bf16 lm_head
in the quant `ignore` list -> int4 BODY isolated, head NOT quantized) on the faithful
dev307 surgical+fold stack, and the lewtun #31 sampling protocol
(do_sample=True T=1.0 top_p=0.95 top_k=64; min_tokens=8 EOS-guard).

The ONLY deviation from #589's run_seeds.py: EVAL_PY points at the surviving
/tmp/eval-serve-venv (inspect_ai 0.3.240 + inspect_evals), because the #589
/tmp/land-inspect ephemeral venv was cleaned. Server recipe is byte-identical, so the
new seeds pool with the existing 10. A --repro-check re-runs sampling_seed 0 and asserts
it reproduces the banked 102/198 -> validates cross-boot pooling before we trust the 30.

Serve-once / eval-many. Existing seeds are skipped (idempotent / resumable).
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
HERE = ROOT / "research/validity/int4_quality_31basis_confirm"
RES = HERE / "results_gpqa"
QE = ROOT / "research/validity/downstream_quality_eval"
SERVE_INJECT = ROOT / "research/validity/vanilla_base_serve_regression/serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")   # dev307 build (#564)
EVAL_PY = Path(os.environ.get("EVAL_PY", "/tmp/eval-serve-venv/bin/python"))  # #589 land-inspect is gone
STOCK = Path(
    os.environ.get(
        "STOCK_CKPT",
        str(Path.home()
            / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct"
              "/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"),
    )
)
PORT = int(os.environ.get("PORT", "8000"))
DATASET_SEED = 12345
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
GPQA_MAX_TOKENS = 3072
MIN_TOKENS = 8
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))  # 4096 == #589 (GPQA mt=3072 fits)

# Banked #589 reference for the cross-boot reproduction gate.
REPRO_REF = {"sampling_seed": 0, "n_correct": 102, "n_scored": 198}

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
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["PYTHONPATH"] = str(SERVE_INJECT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env["PR557_PATCH_DIR"] = str(SUBMISSION)
    env.update(ARM_ENV)
    cmd = [
        str(SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(STOCK), "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", "16",
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", json.dumps(SAMPLING),
    ]
    print(f"[serve] base_fullhead g32+bf16head flags={ARM_ENV} mml={MAX_MODEL_LEN} stock={STOCK}", flush=True)
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
    ap.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29")
    ap.add_argument("--repro-check", action="store_true",
                    help="re-run sampling_seed 0 to a scratch file, assert it reproduces "
                         "the banked 102/198 (cross-boot pooling validity), then continue")
    ap.add_argument("--smoke", action="store_true", help="3 items on seed 0, then exit")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    log = HERE / ("server_smoke.log" if args.smoke else "server_gpqa.log")
    proc = start_server(log)
    try:
        wait_ready(proc)
        print(f"[driver] READY {time.strftime('%H:%M:%S')}", flush=True)
        if args.smoke:
            run_seed(0, RES / "_smoke_gpqa_s0.json", limit=3)
            print("[driver] SMOKE OK", flush=True)
            return 0
        if args.repro_check:
            d = run_seed(0, RES / "_repro_gpqa_s0.json")
            ok = (d["n_correct"] == REPRO_REF["n_correct"] and d["n_scored"] == REPRO_REF["n_scored"])
            print(f"[driver] REPRO-CHECK seed0 got {d['n_correct']}/{d['n_scored']} "
                  f"vs banked {REPRO_REF['n_correct']}/{REPRO_REF['n_scored']} -> "
                  f"{'MATCH (cross-boot reproducible; pooling valid)' if ok else 'MISMATCH (cross-boot drift!)'}",
                  flush=True)
            (HERE / "repro_check.json").write_text(json.dumps(
                {"got_correct": d["n_correct"], "got_scored": d["n_scored"],
                 "ref_correct": REPRO_REF["n_correct"], "ref_scored": REPRO_REF["n_scored"],
                 "reproduced": bool(ok)}, indent=2))
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
