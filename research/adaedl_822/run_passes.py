#!/usr/bin/env python3
"""Drive repeated decode (and optional PPL) passes against ONE persistent
int4_mtp_bi0_int4head serve process, for the AdaEDL early-stop study (#822).

local_prevalidate.py tears the server down after a single PPL+decode cycle. This
driver keeps the server alive so a single model load can sweep many AdaEDL
thresholds: between passes it rewrites ADAEDL_THRESH_FILE (the patch re-reads it,
mtime-cached, at the top of every propose) and toggles ADAEDL_LOG_FLAG (logged
records vs clean zero-overhead TPS). All numbers are LOCAL A10G exploratory
proxies, never official.

Pass spec (``--passes`` JSON list), each item:
  {"name": str, "thresh": "inf"|float|null, "logged": bool, "n": int}
  thresh=null  -> leave ADAEDL_THRESH_FILE untouched (unpatched / static)
  logged=true  -> touch ADAEDL_LOG_FLAG before the pass (records on)
  logged=false -> remove ADAEDL_LOG_FLAG (clean TPS)
  n            -> repeat the decode this many times (>=1); per-rep TPS recorded

Writes <out>/<name>/rep<k>/decode_{outputs.jsonl,summary.json} and a top-level
<out>/passes_summary.json with per-pass per-rep tps + the mean/std.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HARNESS = ROOT / "official/main_bucket/shared_resources/speed_benchmark"
DECODE_SCRIPT = HARNESS / "scripts/decode_outputs.py"
PPL_SCRIPT = HARNESS / "scripts/ppl_endpoint.py"
EVAL_DATASET = HARNESS / "data/eval_prompts_sharegpt.json"
PPL_DATASET = HARNESS / "data/ppl_ground_truth_tokens.jsonl"
SUBMISSION = ROOT / "submissions/int4_mtp_bi0_int4head"
MANIFEST = json.loads((SUBMISSION / "manifest.json").read_text())


def build_env(venv_python: Path, port: int, draft_stop_entropy: str | None,
              adaedl_out: str | None, thresh_file: str | None,
              log_flag: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for k, v in (MANIFEST.get("env") or {}).items():
        env[str(k)] = str(v)
    venv_bin = str(venv_python.parent)
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HOST"] = "0.0.0.0"
    env["PORT"] = str(port)
    # Local-only requirements (memory: flashinfer JIT crash; private-repo 401).
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    # Force device 0: the inherited CUDA_VISIBLE_DEVICES=7 is stale (only index 0
    # exists on this pod) and crashes nvml at startup. Memory: gpu_env.
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # AdaEDL overrides (only present when running the patched server).
    if draft_stop_entropy is not None:
        env["DRAFT_STOP_ENTROPY"] = draft_stop_entropy
    else:
        env.pop("DRAFT_STOP_ENTROPY", None)
    if adaedl_out is not None:
        env["ADAEDL_OUT"] = adaedl_out
    else:
        env.pop("ADAEDL_OUT", None)
    if thresh_file is not None:
        env["ADAEDL_THRESH_FILE"] = thresh_file
    else:
        env.pop("ADAEDL_THRESH_FILE", None)
    if log_flag is not None:
        env["ADAEDL_LOG_FLAG"] = log_flag
    else:
        env.pop("ADAEDL_LOG_FLAG", None)
    return env


def wait_for_models(base_url: str, timeout_s: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before readiness (code {proc.returncode}); see server log")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as resp:
                if resp.status == 200:
                    return
                last = f"status={resp.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(5)
    raise RuntimeError(f"endpoint not ready at {base_url}/v1/models: {last}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=40)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def run_decode(venv_python: Path, base_url: str, model: str, out_dir: Path,
               num_prompts: int, output_len: int, timeout_s: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "decode_summary.json"
    cmd = [str(venv_python), str(DECODE_SCRIPT),
           "--base-url", base_url, "--model", model,
           "--dataset-path", str(EVAL_DATASET),
           "--output-file", str(out_dir / "decode_outputs.jsonl"),
           "--summary-file", str(summary_file),
           "--num-prompts", str(num_prompts),
           "--output-len", str(output_len),
           "--request-timeout-s", str(timeout_s)]
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"decode pass failed rc={rc} ({out_dir})")
    s = json.loads(summary_file.read_text())
    dur = s.get("duration_s") or 0.0
    toks = s.get("num_completion_tokens") or 0
    s["tps"] = (toks / dur) if dur else 0.0
    return s


def run_ppl(venv_python: Path, base_url: str, model: str, out_dir: Path,
            timeout_s: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "ppl_summary.json"
    cmd = [str(venv_python), str(PPL_SCRIPT),
           "--base-url", base_url, "--model", model,
           "--dataset-path", str(PPL_DATASET),
           "--output-file", str(out_dir / "ppl_results.jsonl"),
           "--summary-file", str(summary_file),
           "--request-timeout-s", str(timeout_s)]
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"PPL pass failed rc={rc} ({out_dir})")
    return json.loads(summary_file.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--venv-python", default="/tmp/senpai-venvs/20f658587e8a6643/bin/python")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--draft-stop-entropy", default=None,
                    help="DRAFT_STOP_ENTROPY env (set 'inf' for patched; omit for unpatched)")
    ap.add_argument("--adaedl-out", default=None, help="records JSONL path (enables logging capability)")
    ap.add_argument("--thresh-file", default=None, help="ADAEDL_THRESH_FILE path")
    ap.add_argument("--log-flag", default=None, help="ADAEDL_LOG_FLAG path")
    ap.add_argument("--passes", required=True, help="JSON list of pass specs")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--do-ppl", action="store_true", help="run a 128-record PPL pass first")
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    ap.add_argument("--request-timeout-s", type=int, default=180)
    ap.add_argument("--server-log", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # serve.py runs with cwd=submission_dir, so these env paths MUST be absolute.
    if args.adaedl_out:
        args.adaedl_out = str(Path(args.adaedl_out).resolve())
    if args.thresh_file:
        args.thresh_file = str(Path(args.thresh_file).resolve())
    if args.log_flag:
        args.log_flag = str(Path(args.log_flag).resolve())
    passes = json.loads(args.passes)
    venv_python = Path(args.venv_python)
    model = str(MANIFEST.get("served_model_name", "gemma-4-e4b-it"))
    base_url = f"http://127.0.0.1:{args.port}"
    server_log = args.server_log or str(out_dir / "server.log")

    env = build_env(venv_python, args.port, args.draft_stop_entropy,
                    args.adaedl_out, args.thresh_file, args.log_flag)
    serve_cmd = [str(venv_python), "serve.py"]
    print(f"[driver:{args.label}] starting server (log {server_log})", flush=True)
    log = open(server_log, "w")
    proc = subprocess.Popen(serve_cmd, cwd=str(SUBMISSION), env=env,
                            stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    results: dict = {"label": args.label, "base_url": base_url,
                     "draft_stop_entropy": args.draft_stop_entropy,
                     "num_prompts": args.num_prompts, "output_len": args.output_len,
                     "local_exploratory": True, "passes": []}
    try:
        t0 = time.time()
        wait_for_models(base_url, args.startup_timeout_s, proc)
        print(f"[driver:{args.label}] ready in {time.time()-t0:.1f}s", flush=True)

        if args.do_ppl:
            print(f"[driver:{args.label}] PPL 128-record validity gate", flush=True)
            ppl = run_ppl(venv_python, base_url, model, out_dir / "ppl", args.request_timeout_s)
            results["ppl"] = ppl.get("ppl")
            results["ppl_num_records"] = ppl.get("num_records")
            results["ppl_num_tokens"] = ppl.get("num_tokens")
            print(f"[driver:{args.label}] PPL={results['ppl']} n={results['ppl_num_records']}", flush=True)

        for spec in passes:
            name = spec["name"]
            thresh = spec.get("thresh")
            logged = bool(spec.get("logged", False))
            n = int(spec.get("n", 1))
            # threshold file
            if thresh is not None and args.thresh_file:
                Path(args.thresh_file).write_text(f"{thresh}\n")
                time.sleep(0.05)  # ensure mtime tick
            # logging flag
            if args.log_flag:
                if logged:
                    Path(args.log_flag).touch()
                else:
                    try:
                        Path(args.log_flag).unlink()
                    except FileNotFoundError:
                        pass
            tps_list = []
            for k in range(n):
                s = run_decode(venv_python, base_url, model,
                               out_dir / name / f"rep{k}", args.num_prompts,
                               args.output_len, args.request_timeout_s)
                tps_list.append(s["tps"])
                print(f"[driver:{args.label}] pass={name} rep={k} "
                      f"thresh={thresh} logged={logged} tps={s['tps']:.4f} "
                      f"toks={s['num_completion_tokens']} dur={s['duration_s']:.2f}", flush=True)
            entry = {"name": name, "thresh": thresh, "logged": logged, "n": n,
                     "tps_list": tps_list,
                     "tps_mean": statistics.mean(tps_list),
                     "tps_std": statistics.pstdev(tps_list) if len(tps_list) > 1 else 0.0}
            results["passes"].append(entry)
            (out_dir / "passes_summary.json").write_text(json.dumps(results, indent=2))

        print(f"[driver:{args.label}] DONE", flush=True)
        return 0
    finally:
        terminate(proc)
        print(f"[driver:{args.label}] server terminated", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
