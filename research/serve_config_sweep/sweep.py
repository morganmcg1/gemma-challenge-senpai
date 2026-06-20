#!/usr/bin/env python3
"""PR #811 int4head serve-config byte-exact TPS sweep (LOCAL A10G only).

One-at-a-time A/B of byte-exact-safe serve-config knobs vs the int4head control,
on the conc=1 / output_len=512 single-stream decode workload (128 ShareGPT
prompts, the #788 audit set). Launches the REAL int4head submissions/serve.py
(same patches: force-2D attn + attn-group backport + prometheus guard) with
controlled per-knob env, then scores with the OFFICIAL decode_outputs.py /
ppl_endpoint.py. No manifest mutation, no HF Job.

TPS = decode num_completion_tokens / duration_s (local exploratory proxy, NOT the
official a10g-small number). With ignore_eos + output_len=512 every prompt emits
exactly 512 tokens, so completion length is constant across configs and the TPS
delta is purely duration. Byte-exact parity = identical per-prompt
completion_token_sha256 vs the control reference (a single combined hash per run).

Usage:
  sweep.py <label> [<label> ...] [--num-prompts N] [--output-len L] [--with-ppl]
Each positional <label> is ONE fresh-server run (repeat a label for reps). Results
append to research/serve_config_sweep/results.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
SUBMISSION = ROOT / "submissions/int4_mtp_bi0_int4head"
SERVE_PY = SUBMISSION / "serve.py"
VENV = "/tmp/senpai-venvs/20f658587e8a6643/bin/python"
HARNESS = ROOT / "official/main_bucket/shared_resources/speed_benchmark"
DECODE_SCRIPT = HARNESS / "scripts/decode_outputs.py"
PPL_SCRIPT = HARNESS / "scripts/ppl_endpoint.py"
EVAL_DATASET = HARNESS / "data/eval_prompts_sharegpt.json"
PPL_DATASET = HARNESS / "data/ppl_ground_truth_tokens.jsonl"
OUTROOT = ROOT / "research/serve_config_sweep"
RESULTS = OUTROOT / "results.jsonl"
MODEL_ID = "/workspace/gemma_build/bi0_int4head_g32"
SERVED = "gemma-4-e4b-it"

# Control = int4head shipped manifest env (the ~255-256.74 reference config).
BASE_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",  # native sampler (cuRAND headers missing); logits/PPL unaffected, both arms use it
    "MODEL_ID": MODEL_ID,
    "SERVED_MODEL_NAME": SERVED,
    "VLLM_BATCH_INVARIANT": "0",
    "DRAFTER_MODEL": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
    "NUM_SPECULATIVE_TOKENS": "6",
    "MAX_MODEL_LEN": "4096",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "MAX_NUM_BATCHED_TOKENS": "512",
    "MAX_NUM_SEQS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HOST": "127.0.0.1",
}

# One-at-a-time byte-exact-safe knob overrides vs control.
CONFIGS: dict[str, dict[str, str]] = {
    "control": {},
    # MAX_MODEL_LEN right-sizing. MEASURED (promptlen.py, seed=1, n=128): input tokens
    # max=2427, median=230 -> longest total = 2427+512 = 2939. So the true floor is
    # 2939, NOT the 640 the PR premised (prompts are NOT ~128 tok). maxlen1024 truncates
    # 6/128 prompts and maxlen2048 truncates 1/128 -> both FAIL (non-byte-exact, 400).
    # The ONLY valid right-size is >=2939; maxlen3072 is the legit 4096->3072 test (~25% cut).
    "maxlen3072": {"MAX_MODEL_LEN": "3072"},
    "maxlen2048": {"MAX_MODEL_LEN": "2048"},  # INVALID: truncates the 2427-tok prompt
    "maxlen1024": {"MAX_MODEL_LEN": "1024"},  # INVALID: truncates 6/128 prompts
    # Scheduler budget: engine WARNS to raise max_num_batched_tokens for spec slots.
    "batched2048": {"MAX_NUM_BATCHED_TOKENS": "2048"},
    "batched1024": {"MAX_NUM_BATCHED_TOKENS": "1024"},
    # KV headroom (already 82x unused at conc=1; cheap confirm).
    "gpumem095": {"GPU_MEMORY_UTILIZATION": "0.95"},
    # CUDA-graph capture set tuned to the conc=1 M=7 path (drop never-hit 2,4; keep 1 + the >=7 size 8).
    "cgsizes_1_8": {"CUDAGRAPH_CAPTURE_SIZES": "1 8"},
    "cgsizes_8": {"CUDAGRAPH_CAPTURE_SIZES": "8"},
    # Marlin split-K atomic-add reduce: STRUCTURALLY a no-op on sm_86+bf16
    # (marlin_utils.should_use_atomic_add_reduce gates device_capability[0]<9 & bf16 -> False).
    # Included as an empirical confirm of the source-level null.
    "marlin_atomic": {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
}


def parity_hash(decode_jsonl: Path) -> tuple[str, int, int]:
    """Combined byte-exact signature: sha256 over sorted
    'prompt_token_sha256:completion_token_sha256' lines. Returns (hash, n_prompts,
    total_completion_tokens)."""
    pairs = []
    total = 0
    for line in decode_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pairs.append(f"{row['prompt_token_sha256']}:{row['completion_token_sha256']}")
        total += int(row["num_completion_tokens"])
    pairs.sort()
    h = hashlib.sha256("\n".join(pairs).encode()).hexdigest()
    return h, len(pairs), total


def wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited (code {proc.returncode}) before ready; see log")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as r:
                if r.status == 200:
                    return
                last = f"status={r.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = str(e)
        time.sleep(5)
    raise RuntimeError(f"endpoint not ready: {last}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def run_one(label: str, rep: int, port: int, num_prompts: int, output_len: int, with_ppl: bool) -> dict:
    if label not in CONFIGS:
        raise SystemExit(f"unknown config '{label}'; known: {sorted(CONFIGS)}")
    outdir = OUTROOT / f"{label}_rep{rep}"
    outdir.mkdir(parents=True, exist_ok=True)
    server_log = outdir / "serve.log"
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(CONFIGS[label])
    env["PORT"] = str(port)

    t0 = time.time()
    rec: dict = {"label": label, "rep": rep, "port": port, "config": CONFIGS[label],
                 "num_prompts": num_prompts, "output_len": output_len,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log = open(server_log, "w")
    proc = subprocess.Popen([VENV, str(SERVE_PY)], cwd=str(SUBMISSION), env=env,
                            stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        print(f"[sweep] {label} rep{rep} port {port}: waiting for server ...", flush=True)
        wait_ready(base_url, proc, timeout_s=900)
        rec["startup_s"] = round(time.time() - t0, 1)
        print(f"[sweep] {label} rep{rep}: ready in {rec['startup_s']}s; decoding {num_prompts} prompts", flush=True)

        decode_out = outdir / "decode_outputs.jsonl"
        decode_sum = outdir / "decode_summary.json"
        rc = subprocess.run([VENV, str(DECODE_SCRIPT), "--base-url", base_url, "--model", SERVED,
                             "--dataset-path", str(EVAL_DATASET), "--output-file", str(decode_out),
                             "--summary-file", str(decode_sum), "--num-prompts", str(num_prompts),
                             "--output-len", str(output_len), "--request-timeout-s", "300"]).returncode
        if rc != 0:
            raise RuntimeError(f"decode stage failed rc={rc}")
        ds = json.loads(decode_sum.read_text())
        dur = ds.get("duration_s") or 0.0
        toks = ds.get("num_completion_tokens") or 0
        ph, npairs, total = parity_hash(decode_out)
        rec.update({"tps": round(toks / dur, 4) if dur else 0.0, "decode_duration_s": round(dur, 2),
                    "num_completion_tokens": toks, "decode_num_records": ds.get("num_records"),
                    "parity_hash": ph, "parity_n_prompts": npairs})

        if with_ppl:
            ppl_out = outdir / "ppl_results.jsonl"
            ppl_sum = outdir / "ppl_summary.json"
            rc = subprocess.run([VENV, str(PPL_SCRIPT), "--base-url", base_url, "--model", SERVED,
                                 "--dataset-path", str(PPL_DATASET), "--output-file", str(ppl_out),
                                 "--summary-file", str(ppl_sum), "--request-timeout-s", "300"]).returncode
            if rc != 0:
                raise RuntimeError(f"ppl stage failed rc={rc}")
            ps = json.loads(ppl_sum.read_text())
            rec.update({"ppl": ps.get("ppl"), "ppl_num_records": ps.get("num_records"),
                        "ppl_num_tokens": ps.get("num_tokens")})
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = str(e)
        print(f"[sweep] {label} rep{rep} FAILED: {e}", flush=True)
    finally:
        terminate(proc)
        log.close()
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"SENPAI-SWEEP label={label} rep={rep} ok={rec.get('ok')} tps={rec.get('tps')} "
          f"ppl={rec.get('ppl')} parity={rec.get('parity_hash', '')[:12]} "
          f"completion_toks={rec.get('num_completion_tokens')} dur={rec.get('decode_duration_s')}s "
          f"wall={rec['wall_s']}s", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--with-ppl", action="store_true")
    ap.add_argument("--port", type=int, default=8033)
    args = ap.parse_args()
    # rep index per label = count of prior successful+failed runs of that label in results.jsonl
    seen: dict[str, int] = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                lbl = json.loads(line).get("label")
                seen[lbl] = seen.get(lbl, 0) + 1
    for label in args.labels:
        rep = seen.get(label, 0)
        seen[label] = rep + 1
        run_one(label, rep, args.port, args.num_prompts, args.output_len, args.with_ppl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
