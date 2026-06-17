#!/usr/bin/env python3
"""PR #542 — base_fullhead MMLU-Pro + GPQA-Diamond 2x2 driver. LOCAL, NO FIRE.

Serves two arms on the SAME stock int4 checkpoint and runs ubel #511/#527's
inspect_evals greedy harness (run_eval.py) against each at client conc=32 on a
byte-identical item set:

  base_fullhead : submissions/fa2sw_strict_surgical357 full fast stack (surgical
                  2D attn + MTP K7 + split-KV + onegraph + PLE fold), repointed
                  off the osoi5 substrate onto the stock ckpt with no head prune
                  (LM_HEAD_PRUNE=0, PCK04_KEEPSET="") -> native 262k BF16 head.
  plain base    : the SAME stock ckpt served vanilla vLLM (no fast stack). Only
                  moved variable vs base_fullhead = the fast kernels.

Both servers: greedy temp=0, VLLM_USE_FLASHINFER_SAMPLER=0, same challenge vLLM
wheel. base_fullhead keeps its recipe MAX_NUM_SEQS=1 (deterministic single-stream
via MTP); plain base uses MAX_NUM_SEQS=32 so conc=32 genuinely batches (the vanilla
served regime). Hard-reject guard: raise if lm_head rows < 262144.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

HERE = ROOT / "research/validity/base_fullhead_shortchain_quality"
QE = ROOT / "research/validity/downstream_quality_eval"  # reuse run_eval.py (ubel #511)
EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")
PROM_SHIM = QE / "pck04_inject"  # prometheus route-name compat shim (numerics-orthogonal)
STOCK = "/tmp/gemma4-e4b-qat-w4a16-ct"
SUBMISSION = "fa2sw_strict_surgical357"
PORT = 8000
SEED = 12345

# base_fullhead = fern #535 recipe: repoint off osoi5 substrate, no head prune.
FULLHEAD_OVERRIDES = {
    "LOCAL_MODEL_DIR": STOCK,
    "PLE_FOLD_TARGET_MODEL": STOCK,
    "LM_HEAD_PRUNE": "0",
    "LM_HEAD_PRUNE_REQUIRE": "0",
    "PCK04_KEEPSET": "",
}


def assert_full_head(model_dir: str) -> dict:
    """HARD REJECT if the head is not the full native 262k head."""
    with open(Path(model_dir) / "model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    lm = hdr.get("lm_head.weight")
    rows = lm["shape"][0] if lm else 0
    if rows < 262144:
        raise SystemExit(
            f"HARD REJECT: lm_head rows {rows} < 262144 — not the full native head "
            f"(silent 16k/12k fallback). Refusing to mislabel a pruned head as base_fullhead."
        )
    print(f"[guard] lm_head.weight shape={lm['shape']} dtype={lm['dtype']} rows>=262144 OK", flush=True)
    return {"lm_head_shape": lm["shape"], "lm_head_dtype": lm["dtype"]}


def wait_gpu_free(threshold_mib: int = 1500, timeout_s: int = 180) -> None:
    """Poll nvidia-smi until GPU memory drops below threshold (prev arm fully released)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            used = max(int(x) for x in out.split())
            if used < threshold_mib:
                print(f"[gpu] free ({used} MiB < {threshold_mib}) — proceeding", flush=True)
                return
            print(f"[gpu] waiting for release: {used} MiB used", flush=True)
        except Exception as exc:
            print(f"[gpu] nvidia-smi probe failed: {exc!r}", flush=True)
        time.sleep(5)
    print(f"[gpu] WARN: still busy after {timeout_s}s — proceeding anyway", flush=True)


def wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: int = 1200) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early code={proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"endpoint not ready at {base_url}")


def start_base_server(server_python: Path, log_path: Path, max_num_seqs: int = 32) -> subprocess.Popen:
    """Vanilla vLLM on the stock ckpt — NO fast stack (PYTHONPATH excludes the submission)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # only the prometheus route-name compat shim (orthogonal to numerics); NO submission patches
    env["PYTHONPATH"] = str(PROM_SHIM) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    cmd = [
        str(server_python), "-m", "vllm.entrypoints.openai.api_server",
        "--model", STOCK, "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", '{"temperature":0.0,"top_p":1.0,"top_k":0}',
    ]
    print(f"[base] {' '.join(cmd)}", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


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
    print(f"[eval] {arm}/{task} ->", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] {arm}/{task} acc={d['accuracy']:.4f} scored={d['n_scored']} "
          f"correct={d['n_correct']} err={d['n_error']}", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fullhead serve + 5q MMLU + 5q GPQA, then exit")
    ap.add_argument("--mmlu-n", type=int, default=500)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--arms", default="fullhead,base", help="comma list: fullhead,base")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)
    HERE.mkdir(parents=True, exist_ok=True)

    hd = assert_full_head(STOCK)
    (HERE / "head_guard.json").write_text(json.dumps(hd, indent=2))

    submission_dir = (ROOT / "submissions" / SUBMISSION).resolve()
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[driver] server_python={server_python}", flush=True)

    arms = args.arms.split(",")
    base_url = f"http://127.0.0.1:{PORT}"

    # ---- base_fullhead arm (full fast stack on stock ckpt) ----
    if "fullhead" in arms:
        log = HERE / "server_fullhead.log"
        with harness.LocalServer(
            submission_dir, server_python=server_python, port=PORT,
            log_path=log, extra_env=FULLHEAD_OVERRIDES,
        ) as server:
            print(f"[driver] base_fullhead ready model_id={server.model_id}", flush=True)
            if args.smoke:
                run_cell("fullhead", "mmlu_pro", HERE / "_smoke_fullhead_mmlu.json",
                         n=5, max_tokens=2048, conc=args.conc, limit=5)
                run_cell("fullhead", "gpqa_diamond", HERE / "_smoke_fullhead_gpqa.json",
                         max_tokens=3072, conc=args.conc, limit=5)
                print("[driver] SMOKE OK (fullhead)", flush=True)
                return 0
            run_cell("fullhead", "mmlu_pro", HERE / "fullhead_mmlu_pro.json",
                     n=args.mmlu_n, max_tokens=2048, conc=args.conc)
            run_cell("fullhead", "gpqa_diamond", HERE / "fullhead_gpqa.json",
                     max_tokens=3072, conc=args.conc)

    # ---- plain base arm (vanilla vLLM on the SAME stock ckpt) ----
    if "base" in arms and not args.smoke:
        log = HERE / "server_base.log"
        wait_gpu_free()  # ensure fullhead arm fully released the GPU before base serve
        proc = start_base_server(server_python, log, max_num_seqs=32)
        try:
            wait_ready(base_url, proc)
            print("[driver] plain base ready", flush=True)
            run_cell("base", "mmlu_pro", HERE / "base_mmlu_pro.json",
                     n=args.mmlu_n, max_tokens=2048, conc=args.conc)
            run_cell("base", "gpqa_diamond", HERE / "base_gpqa.json",
                     max_tokens=3072, conc=args.conc)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except Exception:
                pass
            proc.wait(timeout=60)

    print("[driver] 2x2 COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
