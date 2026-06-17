#!/usr/bin/env python
"""PR #549 fern — Stage 1: attribute the base_fullhead -> osoi5 TPS gap into
head vs body, at the M=1 single-stream decode operating point (local, analysis-only).

The full-precision 262k ``lm_head`` (bf16, 1.342 GB) is the fattest *quality-free*
TPS target on ``base_fullhead`` (full head, no prune). Stage 1 asks the decisive,
cheapest question first: **how much TPS is actually recoverable if the head read
went to ~0?** Three numbers:

  * ``head_byte_frac``        -- lm_head GEMM bytes / total per-decode-token weight-read
    bytes (static; embedding/PLE gathers and unused vision/audio towers excluded).
    The pure-read-bound CEILING on what dropping the head could buy.
  * ``head_time_frac``        -- MEASURED head-GEMM GPU time / per-verify-step wall.
    The Amdahl term: only a component's measured wall-fraction is creditable to TPS
    (the deployed serve point is ~38% HBM-read-bound, denken #283 -- so time_frac is
    strictly below byte_frac, and that gap IS the non-read-bound tax).
  * ``head_attributable_tps`` -- ``TPS * head_time_frac/(1-head_time_frac)``: the TPS
    rise if the head GEMM time were removed from the critical path (upper bound,
    assumes head fully on the critical path).

GO/NO-GO (PR #549): if ``head_attributable_tps < 20``, the head is not the lever ->
clean NO-GO, stop before Stage 2.

Instrumentation is in-context and default-off:
  * ``usercustomize.py`` (FULLHEAD_HOOK=1) CUDA-event-times ``Gemma4ForCausalLM.
    compute_logits`` (the 262k head GEMM) per call, bucketed by M. Immune to a head
    ABLATION's confound: it times the head GEMM inside the UNCHANGED base_fullhead
    step, so speculative acceptance / step structure are unperturbed (a head-prune
    ablation would change which tokens are accepted -> different E[accept] -> TPS not
    comparable; CUDA-event timing sidesteps that entirely).
  * ``steptime_patch.py`` (STEPTIME=1, already in the submission) CUDA-event-times the
    whole ``execute_model`` (verify) step and the drafter ``propose``.

Two serve passes:
  * Pass A (timing, decisive): FULLHEAD_DUMP=0 -> the hidden-state D2H copy is OFF, so
    head timing and TPS are unperturbed. Yields all three head_* numbers + E[accept].
  * Pass B (capture, GATED on Pass A GO): FULLHEAD_DUMP=1 -> dump <=60k held-out
    decode-position hidden states to /tmp for the Stage-2 offline miss-rate(K) leg.

Cross-check: an isolation head-GEMM microbench ([M,2560]@[2560,262144] bf16, M in
{1,8}) grounds the in-context decode_head_ms against raw A10G HBM bandwidth.

LOCAL only: analysis_only, official_tps=0, no HF Job, no --launch, no submission, no
served-file change (base_fullhead is reached purely by serve-env overrides + the
default-off probe).

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/fullhead_candidate_verify/stage1_driver.py --smoke --no-wandb
Full Stage-1:
  CUDA_VISIBLE_DEVICES=0 python research/fullhead_candidate_verify/stage1_driver.py \
    --wandb_name fern/fullhead-stage1 --wandb_group fullhead-candidate-verify
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# When re-invoked under the SERVER venv (microbench mode) avoid a stray cwd/script
# entry shadowing stdlib; re-add ROOT for `from scripts.local_validation import ...`.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
MEASURE_FLOOR = ROOT / "research" / "base_int4_floor_tps" / "measure_floor.py"
OUT_ROOT = HERE

# fern's own readable w4a16-ct snapshot (full 262k bf16 head; int4 body).
MODEL_DIR = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)

HIDDEN = 2560
VOCAB = 262144
HEAD_BYTES = HIDDEN * VOCAB * 2  # bf16 lm_head weight = 1.342 GB

NUM_PROMPTS = 128
OUTPUT_LEN = 512
SEED = 1

# Cited anchors (NOT re-derived here).
BASE_FULLHEAD_TPS_ANCHOR = 252.49   # report_base_fullhead.json warm-median (#535/#544)
OSOI5_PRUNED_TPS_ANCHOR = 353.73    # osoi5-12k local warm-median (PR #549 context)
A10G_HBM_GBPS = 600.0               # A10G HBM2 ~600 GB/s (read-bound floor context)

NO_GO_TPS = 20.0                    # PR #549 Stage-1 gate
DUMP_PATH = "/tmp/fullhead_hidden_fern.pt"
DUMP_MAX = 60000


# ========================================================================== #
# base_fullhead serve recipe + instrumentation env
# ========================================================================== #
def build_extra_env(*, dump: bool, model_dir: str) -> dict[str, str]:
    """report_base_fullhead.json recipe (full 262k head, prune OFF) + probes.

    Everything else is the submission's manifest default (spec-alive MTP K=7,
    ONEGRAPH, fused-sparse-argmax on the DRAFTER, precache, fa-sliding, splitkv).
    SURGICAL_ATTN_USE_3D_OFF is left UNSET -> stock parent compute path.
    """
    env: dict[str, str] = {
        # --- full-head base recipe (overrides the osoi5 12k-prune manifest) ---
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",
        "LOCAL_MODEL_DIR": model_dir,
        "PLE_FOLD_TARGET_MODEL": model_dir,
        "LM_HEAD_FULL_REQUIRE": "1",   # fail-closed: assert 262144-row head loaded
        # --- in-context probes (default-off; gated here) ---
        # research-dir FIRST so `import sitecustomize` resolves to our shim (the
        # vLLM worker disables user-site, so usercustomize won't load). serve.py
        # leaves PYTHONPATH untouched because the package dir is already present.
        "PYTHONPATH": f"{HERE}{os.pathsep}{SUBMISSION}",
        "FULLHEAD_PKG_DIR": str(SUBMISSION),
        "FULLHEAD_HOOK": "1",
        "FULLHEAD_DUMP": "1" if dump else "0",
        "FULLHEAD_DUMP_PATH": DUMP_PATH,
        "FULLHEAD_DUMP_MAX": str(DUMP_MAX),
        "FULLHEAD_DECODE_M_MAX": "8",
        "FULLHEAD_WARMUP_SKIP": "64",
        "FULLHEAD_REPORT_EVERY": "2000",
        "STEPTIME": "1",
        "STEPTIME_WARMUP_SKIP": "64",
        "STEPTIME_REPORT_EVERY": "1024",
    }
    return env


# ========================================================================== #
# static head_byte_frac from the safetensors header
# ========================================================================== #
def _safetensors_header(model_path: Path) -> dict[str, Any]:
    with open(model_path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def _is_per_decode_read(name: str) -> bool:
    """True if this tensor's full bytes are read on EVERY decode token.

    Excludes lookup tables (token/PLE/multimodal embeddings are GATHERED, not
    full-read) and the unused vision/audio towers. Includes every transformer
    backbone matmul weight (+ its int4 scales/zeros), the layernorms, the final
    norm, and the lm_head.
    """
    low = name.lower()
    exclude = (
        "embed_tokens", "per_layer", "ple", "vision", "audio",
        "multimodal", "multi_modal", "mm_", "embed_audio", "embed_vision",
        "altup",  # per-token routing tables in gemma3n; not a dense matmul read
    )
    if any(tok in low for tok in exclude):
        # lm_head sometimes shares the "embed" stem in tied models -> keep it.
        if "lm_head" not in low:
            return False
    return True


def head_byte_frac(model_dir: str) -> dict[str, Any]:
    path = Path(model_dir) / "model.safetensors"
    if not path.exists():
        return {"head_byte_frac": None, "note": f"missing {path}"}
    header = _safetensors_header(path)
    head_bytes = 0
    body_bytes = 0
    total_bytes = 0
    n_head = 0
    for name, meta in header.items():
        off = meta.get("data_offsets")
        if not off:
            continue
        nbytes = int(off[1]) - int(off[0])
        total_bytes += nbytes
        if "lm_head" in name.lower():
            head_bytes += nbytes
            n_head += 1
        elif _is_per_decode_read(name):
            body_bytes += nbytes
    denom = head_bytes + body_bytes
    frac = head_bytes / denom if denom else None
    return {
        "head_byte_frac": frac,
        "head_bytes": head_bytes,
        "body_read_bytes": body_bytes,
        "per_decode_read_bytes": denom,
        "total_checkpoint_bytes": total_bytes,
        "n_head_tensors": n_head,
        "head_gb": head_bytes / 1e9,
        "per_decode_read_gb": denom / 1e9,
    }


# ========================================================================== #
# isolation head-GEMM microbench (runs UNDER the server venv: needs torch+cuda)
# ========================================================================== #
def _microbench(out_file: str) -> int:
    import torch

    dev = "cuda"
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    # lm_head weight [vocab, hidden]; logits = x @ W.T  (x is [M, hidden]).
    W = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=dev)
    out: dict[str, Any] = {"device": name, "hidden": HIDDEN, "vocab": VOCAB,
                           "head_bytes": HEAD_BYTES, "per_M": {}}
    for M in (1, 8):
        x = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)
        for _ in range(25):
            _ = x @ W.t()
        torch.cuda.synchronize()
        iters = 200
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            _ = x @ W.t()
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / iters
        achieved_gbps = HEAD_BYTES / (ms / 1e3) / 1e9
        out["per_M"][str(M)] = {
            "head_gemm_ms": ms,
            "achieved_read_gbps": achieved_gbps,
        }
        print(f"[microbench] M={M} head_gemm_ms={ms:.4f} achieved_read={achieved_gbps:.1f} GB/s",
              flush=True)
    out["read_floor_ms"] = HEAD_BYTES / (A10G_HBM_GBPS * 1e9) * 1e3
    Path(out_file).write_text(json.dumps(out, indent=2))
    return 0


def run_microbench(server_python: Path) -> dict[str, Any]:
    out_file = OUT_ROOT / "microbench.json"
    cmd = [str(server_python), str(Path(__file__).resolve()), "--microbench",
           "--out-file", str(out_file)]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    env["PATH"] = f"{server_python.parent}{os.pathsep}{env.get('PATH', '')}"
    print(f"[stage1] head-GEMM isolation microbench -> {out_file}", flush=True)
    subprocess.run(cmd, check=True, timeout=600, env=env)
    return json.loads(out_file.read_text())


# ========================================================================== #
# server-log parsing (formats are emitted by usercustomize.py / steptime_patch.py)
# ========================================================================== #
_RE_FULLHEAD_FINAL = re.compile(
    r"\[fullhead\] FINAL decode_verify_calls=(\d+) "
    r"decode_head_ms_sum=([\d.]+) decode_head_ms_mean=([\d.eE+-]+|nan)"
)
_RE_FULLHEAD_AGG = re.compile(
    r"\[fullhead\] agg decode_calls=(\d+) decode_head_ms_mean=([\d.eE+-]+|nan) "
    r"decode_head_ms_sum=([\d.]+) \| prefill_calls=(\d+) "
    r"prefill_head_ms_mean=([\d.eE+-]+|nan) \| hidden_rows=(\d+) m_hist=(\{[^}]*\})"
)
_RE_STEPTIME_AGG = re.compile(r"\[steptime\] agg n=(\d+) kind=(\w+) (.*)")
_RE_METRIC = re.compile(r"(\w+) p50=([\d.eE+-]+) p90=([\d.eE+-]+) mean=([\d.eE+-]+)")


def _f(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_server_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    res: dict[str, Any] = {
        "fullhead_shim_ran": "[fullhead-shim] ran submission sitecustomize" in text,
        "fullhead_finder_registered": "[fullhead] finder registered" in text,
        "fullhead_wrapped": "[fullhead] wrapped Gemma4ForCausalLM.compute_logits" in text,
        "lmhead_full_verified": bool(re.search(r"verified full lm_head: 262144 rows", text)),
        "steptime_exec_active": "[steptime] execute_model wrapper active" in text,
        "steptime_propose_active": "[steptime] propose wrapper active" in text,
    }
    # The probe runs in BOTH the APIServer parent (imports gemma4 but never decodes
    # -> 0 calls) and the EngineCore worker (the real decode loop). Take the record
    # with the MOST decode calls (the worker); a plain last-match grabs the parent's
    # zero-call atexit line and shadows the worker's real timing with nan.
    fm = list(_RE_FULLHEAD_FINAL.finditer(text))
    if fm:
        m = max(fm, key=lambda mm: int(mm.group(1)))
        res["decode_verify_calls"] = int(m.group(1))
        res["decode_head_ms_sum"] = _f(m.group(2))
        res["decode_head_ms_mean"] = _f(m.group(3))
    am = list(_RE_FULLHEAD_AGG.finditer(text))
    if am:
        m = max(am, key=lambda mm: int(mm.group(1)))
        res["agg_decode_calls"] = int(m.group(1))
        res["agg_decode_head_ms_mean"] = _f(m.group(2))
        res["agg_decode_head_ms_sum"] = _f(m.group(3))
        res["prefill_calls"] = int(m.group(4))
        res["prefill_head_ms_mean"] = _f(m.group(5))
        res["hidden_rows"] = int(m.group(6))
        try:
            res["m_hist"] = ast.literal_eval(m.group(7))
        except Exception:
            res["m_hist"] = {}
        # The worker atexit FINAL line can be SIGKILLed before flush, leaving only the
        # parent's zero-call FINAL; the periodic worker AGG line is the reliable timing
        # source. Promote it to the canonical decode_head_ms_* when it has more calls.
        if res["agg_decode_calls"] > (res.get("decode_verify_calls") or 0):
            res["decode_verify_calls"] = res["agg_decode_calls"]
            res["decode_head_ms_sum"] = res["agg_decode_head_ms_sum"]
            res["decode_head_ms_mean"] = res["agg_decode_head_ms_mean"]
    # last steptime agg per kind
    step: dict[str, dict[str, float]] = {}
    for m in _RE_STEPTIME_AGG.finditer(text):
        kind = m.group(2)
        body = m.group(3)
        metrics = {mm.group(1): _f(mm.group(4)) for mm in _RE_METRIC.finditer(body)}
        metrics["_n"] = float(m.group(1))
        step[kind] = metrics  # keep last (cumulative) occurrence
    res["steptime"] = step
    return res


# ========================================================================== #
# attribution math
# ========================================================================== #
def compute_attribution(tps: float, parsed: dict[str, Any], byte_info: dict[str, Any]) -> dict[str, Any]:
    head_ms = parsed.get("decode_head_ms_mean", float("nan"))
    step = parsed.get("steptime", {})
    exec_m = step.get("exec", {})
    draft_m = step.get("draft", {})
    exec_gap = exec_m.get("gap", float("nan"))
    exec_cpu = exec_m.get("cpu", float("nan"))
    exec_gpu = exec_m.get("gpu", float("nan"))
    draft_cpu = draft_m.get("cpu", 0.0)
    draft_gpu = draft_m.get("gpu", 0.0)

    # Per-verify-step wall cadence: gap (covers the standalone drafter propose +
    # scheduling between exec calls in this wheel) + this exec's cpu wall.
    wall_step_ms = exec_gap + exec_cpu
    head_time_frac = head_ms / wall_step_ms if wall_step_ms and math.isfinite(wall_step_ms) else float("nan")
    # GPU-only fraction (context): head vs the summed GPU work of verify+draft.
    gpu_step_ms = exec_gpu + draft_gpu
    head_gpu_frac = head_ms / gpu_step_ms if gpu_step_ms else float("nan")

    if math.isfinite(head_time_frac) and head_time_frac < 1.0:
        head_attributable_tps = tps * head_time_frac / (1.0 - head_time_frac)
    else:
        head_attributable_tps = float("nan")

    eacc_wall = tps * wall_step_ms / 1e3 if math.isfinite(wall_step_ms) else float("nan")

    byte_frac = byte_info.get("head_byte_frac")
    tps_ceiling_readbound = (
        tps / (1.0 - byte_frac) if isinstance(byte_frac, float) and byte_frac < 1.0 else None
    )
    head_read_floor_ms = HEAD_BYTES / (A10G_HBM_GBPS * 1e9) * 1e3

    return {
        "tps_measured": tps,
        "head_byte_frac": byte_frac,
        "head_time_frac": head_time_frac,
        "head_gpu_frac": head_gpu_frac,
        "head_attributable_tps": head_attributable_tps,
        "head_gemm_ms": head_ms,
        "head_read_floor_ms": head_read_floor_ms,
        "wall_per_step_ms": wall_step_ms,
        "exec_gap_ms": exec_gap,
        "exec_cpu_ms": exec_cpu,
        "exec_gpu_ms": exec_gpu,
        "draft_cpu_ms": draft_cpu,
        "draft_gpu_ms": draft_gpu,
        "gpu_per_step_ms": gpu_step_ms,
        "eacc_wall_estimate": eacc_wall,
        "tps_ceiling_readbound": tps_ceiling_readbound,
        "go": bool(math.isfinite(head_attributable_tps) and head_attributable_tps >= NO_GO_TPS),
    }


# ========================================================================== #
# one serve pass (launch -> warm -> decode -> aggregate -> parse log)
# ========================================================================== #
def run_pass(mf: Any, harness: Any, paths: Any, *, server_python: Path, submission: Path,
             extra_env: dict[str, str], label: str, num_prompts: int, output_len: int,
             port: int, request_timeout_s: int) -> dict[str, Any]:
    log_path = OUT_ROOT / f"server_{label}.log"
    pass_file = OUT_ROOT / f"{label}_pass.json"

    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    peak = {"mib": 0.0}
    stop = threading.Event()

    def _sample_vram() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
                if vals:
                    peak["mib"] = max(peak["mib"], max(vals))
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(2.0)

    sampler = threading.Thread(target=_sample_vram, daemon=True)
    sampler.start()
    measured: dict[str, Any] = {"label": label, "num_prompts": num_prompts, "output_len": output_len}
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            measured["model_id"] = srv.model_id
            measured["served_model_name"] = srv.served_model_name
            print(f"[stage1] warming server ({srv.base_url})", flush=True)
            mf._warm_server(srv.base_url, srv.served_model_name, n=mf.WARMUP_REQUESTS)
            summary = mf._run_decode_pass(
                server_python, worker_env, base_url=srv.base_url, model=srv.served_model_name,
                out_file=pass_file, num_prompts=num_prompts, output_len=output_len,
                dataset_path=paths.EVAL_PROMPTS, tokenizer=paths.TOKENIZER,
                request_timeout_s=request_timeout_s,
            )
            measured["tps"] = mf._aggregate(summary)
    finally:
        stop.set()
        sampler.join(timeout=5)

    measured["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    measured["parsed_log"] = parse_server_log(log_path)
    measured["log_path"] = str(log_path)
    return measured


# ========================================================================== #
# wandb
# ========================================================================== #
def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[stage1] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile",
        agent="fern",
        name=args.wandb_name or "fern/fullhead-stage1",
        group=args.wandb_group or "fullhead-candidate-verify",
        tags=["fullhead", "candidate-verify", "stage1", "head-attribution",
              "local-a10g", "analysis-only"],
        notes="PR #549 Stage 1: head-vs-body TPS attribution on base_fullhead 262k head",
        config={
            "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
            "num_prompts": report["measured_A"]["num_prompts"],
            "output_len": report["measured_A"]["output_len"],
            "concurrency": 1, "seed": SEED, "no_go_tps": NO_GO_TPS,
            "hidden": HIDDEN, "vocab": VOCAB,
        },
    )
    if run is None:
        return None
    summary = {k: v for k, v in report["attribution"].items()
               if isinstance(v, (int, float, bool)) and (not isinstance(v, float) or math.isfinite(v))}
    summary["peak_vram_gb"] = report["measured_A"].get("peak_vram_gb")
    summary["analysis_only"] = True
    summary["official_tps"] = 0
    summary["primary_metric"] = report["attribution"].get("head_attributable_tps")
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="fullhead-stage1-report", artifact_type="stage1-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(report: dict[str, Any]) -> None:
    a = report["attribution"]
    line = "=" * 12 + " PR #549 STAGE 1 — base_fullhead HEAD-vs-BODY ATTRIBUTION " + "=" * 12
    print("\n" + line, flush=True)
    print(f"  TPS (Pass A, warm-median)      = {a['tps_measured']:.2f}", flush=True)
    print(f"  head_byte_frac (static read)   = {a['head_byte_frac']}", flush=True)
    print(f"  head_gemm_ms (in-context)      = {a['head_gemm_ms']:.4f}  "
          f"(read-floor {a['head_read_floor_ms']:.3f} ms)", flush=True)
    print(f"  wall_per_step_ms (gap+cpu)     = {a['wall_per_step_ms']:.4f}  "
          f"(gpu/step {a['gpu_per_step_ms']:.4f})", flush=True)
    print(f"  head_time_frac (MEASURED)      = {a['head_time_frac']:.4f}  "
          f"(gpu_frac {a['head_gpu_frac']:.4f})", flush=True)
    print(f"  E[accept] (wall-derived)       = {a['eacc_wall_estimate']:.3f}", flush=True)
    print(f"  >>> head_attributable_tps      = {a['head_attributable_tps']:.2f}  "
          f"(read-bound ceiling {a['tps_ceiling_readbound']})", flush=True)
    print(f"  GO (>= {NO_GO_TPS:.0f} TPS)?            = {a['go']}", flush=True)
    mb = report.get("microbench", {}).get("per_M", {})
    if mb:
        print(f"  microbench head_gemm_ms        = M1 {mb.get('1', {}).get('head_gemm_ms', float('nan')):.4f} / "
              f"M8 {mb.get('8', {}).get('head_gemm_ms', float('nan')):.4f}", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# main
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="4x32 plumbing check (Pass A only)")
    ap.add_argument("--no-dump", action="store_true", help="skip Pass B hidden-state capture even if GO")
    ap.add_argument("--dump-only", action="store_true",
                    help="run ONLY Pass B hidden-state capture (skip Pass A/attribution; GO already established)")
    ap.add_argument("--no-microbench", action="store_true")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=300)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    # internal microbench mode (runs under the server venv)
    ap.add_argument("--microbench", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args(argv)

    if args.microbench:
        return _microbench(args.out_file)

    import importlib.util
    spec = importlib.util.spec_from_file_location("measure_floor", str(MEASURE_FLOOR))
    mf = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mf)
    from scripts.local_validation import harness, paths

    for note in paths.prepare_local_gpu_env():
        print(f"[stage1] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])

    num_prompts = 4 if args.smoke else NUM_PROMPTS
    output_len = 32 if args.smoke else OUTPUT_LEN
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    byte_info = head_byte_frac(args.model_dir)
    print(f"[stage1] head_byte_frac={byte_info.get('head_byte_frac')} "
          f"(head {byte_info.get('head_gb')} GB / per-decode-read {byte_info.get('per_decode_read_gb')} GB)",
          flush=True)

    if args.dump_only:
        print("[stage1] --dump-only: skipping Pass A; running Pass B hidden-state capture", flush=True)
        extra_b = build_extra_env(dump=True, model_dir=args.model_dir)
        measured_b = run_pass(
            mf, harness, paths, server_python=server_python, submission=SUBMISSION,
            extra_env=extra_b, label="passB_dump",
            num_prompts=num_prompts, output_len=output_len, port=args.port,
            request_timeout_s=args.request_timeout_s,
        )
        pb = measured_b["parsed_log"]
        dump_exists = Path(DUMP_PATH).exists()
        dump_sz = Path(DUMP_PATH).stat().st_size if dump_exists else 0
        print(f"[stage1] dump-only: exists={dump_exists} size={dump_sz/1e6:.1f}MB "
              f"hidden_rows={pb.get('hidden_rows')} -> {DUMP_PATH}", flush=True)
        (OUT_ROOT / "stage1_dumponly.json").write_text(json.dumps(
            {"measured_B": measured_b, "dump_path": DUMP_PATH, "dump_exists": dump_exists,
             "dump_bytes": dump_sz, "hidden_rows": pb.get("hidden_rows")},
            indent=2, sort_keys=True, default=str))
        return 0 if dump_exists else 1

    # --- Pass A: clean timing (dump OFF) ---
    extra_a = build_extra_env(dump=False, model_dir=args.model_dir)
    measured_a = run_pass(
        mf, harness, paths, server_python=server_python, submission=SUBMISSION,
        extra_env=extra_a, label="passA_smoke" if args.smoke else "passA",
        num_prompts=num_prompts, output_len=output_len, port=args.port,
        request_timeout_s=args.request_timeout_s,
    )
    parsed_a = measured_a["parsed_log"]
    tps_a = measured_a["tps"]["warm_median_tps"]

    # hard plumbing checks
    plumbing = {
        "fullhead_wrapped": parsed_a.get("fullhead_wrapped"),
        "lmhead_full_verified": parsed_a.get("lmhead_full_verified"),
        "steptime_exec_active": parsed_a.get("steptime_exec_active"),
        "has_decode_head_ms": "decode_head_ms_mean" in parsed_a,
        "has_exec_steptime": "exec" in parsed_a.get("steptime", {}),
    }
    print(f"[stage1] plumbing: {plumbing}", flush=True)

    attribution = compute_attribution(tps_a, parsed_a, byte_info)

    microbench = {}
    if not args.no_microbench:
        try:
            microbench = run_microbench(server_python)
        except Exception as exc:
            print(f"[stage1] microbench FAILED (non-fatal): {exc!r}", flush=True)

    # --- Pass B: hidden-state capture, GATED on GO (skip in smoke) ---
    measured_b = None
    if not args.smoke and not args.no_dump and attribution["go"]:
        print(f"[stage1] GO (head_attributable_tps={attribution['head_attributable_tps']:.1f}) "
              f"-> Pass B hidden-state capture for Stage 2", flush=True)
        extra_b = build_extra_env(dump=True, model_dir=args.model_dir)
        measured_b = run_pass(
            mf, harness, paths, server_python=server_python, submission=SUBMISSION,
            extra_env=extra_b, label="passB_dump",
            num_prompts=num_prompts, output_len=output_len, port=args.port,
            request_timeout_s=args.request_timeout_s,
        )
        pb = measured_b["parsed_log"]
        measured_b["dump_exists"] = Path(DUMP_PATH).exists()
        measured_b["dump_path"] = DUMP_PATH
        measured_b["hidden_rows"] = pb.get("hidden_rows")
        print(f"[stage1] Pass B dump exists={measured_b['dump_exists']} "
              f"rows={measured_b['hidden_rows']} -> {DUMP_PATH}", flush=True)
    elif not attribution["go"]:
        print(f"[stage1] NO-GO (head_attributable_tps={attribution['head_attributable_tps']:.1f} "
              f"< {NO_GO_TPS}) -> stopping at Stage 1, no Pass B", flush=True)

    report: dict[str, Any] = {
        "analysis_only": True,
        "official_tps": 0,
        "tps_delta": 0.0,
        "pr": 549,
        "stage": 1,
        "submission": str(SUBMISSION),
        "model_dir": args.model_dir,
        "byte_info": byte_info,
        "attribution": attribution,
        "plumbing": plumbing,
        "microbench": microbench,
        "measured_A": measured_a,
        "measured_B": measured_b,
        "anchors": {
            "base_fullhead_tps": BASE_FULLHEAD_TPS_ANCHOR,
            "osoi5_pruned_tps": OSOI5_PRUNED_TPS_ANCHOR,
            "observed_prune_gain_tps": OSOI5_PRUNED_TPS_ANCHOR - BASE_FULLHEAD_TPS_ANCHOR,
        },
    }

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report["created_at"] = stamp
    if not args.no_wandb and not args.smoke:
        report["wandb_run_id"] = log_wandb(report, args)

    out_name = "stage1_smoke.json" if args.smoke else "stage1_report.json"
    (OUT_ROOT / out_name).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(report)
    print(f"[stage1] report: {OUT_ROOT / out_name}", flush=True)

    if not plumbing["fullhead_wrapped"] or not plumbing["has_decode_head_ms"]:
        print("[stage1] FAIL: fullhead hook did not fire / no decode head timing captured", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
