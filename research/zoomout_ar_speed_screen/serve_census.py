#!/usr/bin/env python
"""PR #630 ZOOM-OUT AR speed screen — local byte-exact knob A/B (analysis-only).

Serves the strict ``int4_g128_lmhead`` rung locally on the assigned A10G and, for
each candidate M=1 single-stream backend knob, measures two things on ONE live
server session per config:

  * **warm-median single-stream decode TPS** — the SAME conc=1, seed=1,
    ``ignore_eos`` recipe the base-int4 floor used (#533, ``measure_floor.py``),
    so the numbers are directly comparable to the 95.78 local floor anchor.
  * **byte-exact greedy census** — per-prompt completion-token sha256 at
    ``temperature=0`` / ``ignore_eos``, captured via the OFFICIAL
    ``decode_outputs.py`` request path. Compared two ways:
      (a) r1-vs-r2 within the reference config  -> M=1 self-determinism, and
      (b) each knob-vs-reference                -> the #319 byte-exactness contract.

The locked 126.378 rung is the OFFICIAL a10g-small score (sglang bench in an HF
Job); it CANNOT be reproduced locally. We report the stock warm-median as the
on-harness anchor plus the local->official map (tau_lo, #267) for CONTEXT, and we
answer the decisive speed question LOCALLY and apples-to-apples: does any
byte-exact knob beat the stock reference by >= +10 TPS on the SAME harness?

Knobs are injected as serve env (vLLM reads ``VLLM_ATTENTION_BACKEND`` natively;
``serve.py`` execs vLLM with the full environment). CUDA-graph stays ON for every
config (stock ``serve.py`` passes no ``--enforce-eager``); the roofline already
prices the eager axis. The reference is whatever backend vLLM auto-selects for the
heterogeneous head_dim (256 local / 512 global) — captured from the server log.

LOCAL only: ``analysis_only=true``, ``official_tps=0``, NO HF Job, NO ``--launch``,
NO submission, NO served-file change. PPL / greedy-token identity are properties
of the model+config; a knob that changes them is exactly what the census detects.

Run (smoke plumbing first):
  CUDA_VISIBLE_DEVICES=0 python research/zoomout_ar_speed_screen/serve_census.py --smoke --no-wandb
Full screen (stock + attention-backend swaps):
  CUDA_VISIBLE_DEVICES=0 python research/zoomout_ar_speed_screen/serve_census.py \
    --wandb_name wirbel/zoomout-ar-speed-census --wandb_group zoomout-ar-speed-screen
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Re-invoked as --decode-worker under the SERVER venv. Drop bare cwd / script-dir
# from sys.path so a stray module never shadows stdlib (transformers -> dynamo ->
# cProfile does `import profile`); re-add ROOT for `from scripts.local_validation`.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBMISSION = ROOT / "submissions" / "int4_g128_lmhead"
BUILT_CKPT = Path("/workspace/gemma_build/int4_g128_lmhead")
OUT_ROOT = ROOT / "research" / "zoomout_ar_speed_screen" / "out"

NUM_PROMPTS = 64          # PR #630: byte-exact census n >= 64 prompts
OUTPUT_LEN = 512          # official decode depth (amortizes prefill in the TPS too)
SEED = 1
WARMUP_REQUESTS = 4       # official sglang.bench_serving discards this many from timing

# Reference rung + speed contract (all on-branch, cited in the roofline).
INT4_HEAD_OFFICIAL_TPS = 126.378     # locked int4_g128_lmhead rung (PR #4), OFFICIAL a10g-small
BASE_INT4_LOCAL_FLOOR_TPS = 95.77683  # base-int4 warm-median, SAME local harness (#533, b9j1z40d)
TPS_GAIN_TARGET = 10.0               # #481 ask: >= +10 TPS over 126.378
# local->official map CONTEXT only (advisor applies it; not a claim of this leg).
TAU_LO = 481.53 / 465.14047160458415  # #267 deployed-ship anchor ratio ~= 1.0352

# Candidate single-stream knobs. CONFIGS[0] is the byte-exact REFERENCE.
# attention-backend swaps are the one env-injectable knob and the exact axis the
# #393/#498 byte-exact pin is about (heterogeneous head_dim forces a Triton
# sliding-window path; alternatives either can't serve head_dim 512 or flip the
# split-KV reduction order -> non-byte-exact). CUDA-graph stays on for all.
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("stock", {}),
    ("flashinfer", {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}),
    ("flash_attn", {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}),
]
REFERENCE = "stock"


# ========================================================================== #
# Decode worker (runs UNDER the server venv: needs transformers tokenizer)
# ========================================================================== #
def _decode_worker(args: argparse.Namespace) -> int:
    import importlib.util

    from scripts.local_validation import paths  # noqa: E402

    spec = importlib.util.spec_from_file_location("official_decode", str(paths.DECODE_SCRIPT))
    od = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(od)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = od.read_sharegpt_prompts(Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed)
    if len(records) != args.num_prompts:
        raise ValueError(f"expected {args.num_prompts} prompts, found {len(records)}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        t0 = time.perf_counter()
        prompt_token_ids = od.encode_prompt(tok, record["prompt_text"])
        t1 = time.perf_counter()
        response = od.request_decode(
            base_url=args.base_url, model=args.model,
            prompt_token_ids=prompt_token_ids, output_len=args.output_len,
            timeout_s=args.request_timeout_s,
        )
        t2 = time.perf_counter()
        choice = od.choice_from_response(response)
        completion_token_ids, _, _ = od.extract_generated_token_ids(response, choice, prompt_token_ids)
        rows.append({
            "index": index,
            "id": record["id"],
            "t_tokenize_s": t1 - t0,
            "t_request_s": t2 - t1,
            "num_prompt_tokens": len(prompt_token_ids),
            "num_completion_tokens": len(completion_token_ids),
            "prompt_token_sha256": od.sha256_tokens(prompt_token_ids),
            "completion_token_sha256": od.sha256_tokens(completion_token_ids),
            "completion_token_ids": completion_token_ids,  # kept for first-divergence diag
        })
        print(f"[worker] {index + 1}/{len(records)} req_ms={1000.0 * (t2 - t1):.1f} "
              f"comp={len(completion_token_ids)} csha={rows[-1]['completion_token_sha256'][:12]}",
              flush=True)

    out = {"output_len": args.output_len, "num_records": len(records), "per_request": rows}
    Path(args.out_file).write_text(json.dumps(out))
    return 0


# ========================================================================== #
# Aggregation (TPS) — mirrors measure_floor.py so floors are comparable
# ========================================================================== #
def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, k))]


def _aggregate_tps(summary: dict[str, Any], warmup: int) -> dict[str, Any]:
    rows = summary["per_request"]
    n = len(rows)
    t_req = [r["t_request_s"] for r in rows]
    n_comp = [r["num_completion_tokens"] for r in rows]
    warm_idx = list(range(min(warmup, n), n))
    T_req_warm = sum(t_req[i] for i in warm_idx)
    N_warm = sum(n_comp[i] for i in warm_idx)
    per_req_tps_warm = sorted(n_comp[i] / t_req[i] for i in warm_idx if t_req[i] > 0)
    warm_median_tps = statistics.median(per_req_tps_warm) if per_req_tps_warm else float("nan")
    warm_aggregate_tps = N_warm / T_req_warm if T_req_warm else float("nan")
    comp = [r["num_completion_tokens"] for r in rows]
    return {
        "warm_median_tps": warm_median_tps,
        "warm_aggregate_tps": warm_aggregate_tps,
        "warm_mean_tps": statistics.fmean(per_req_tps_warm) if per_req_tps_warm else float("nan"),
        "warm_tps_p10": _percentile(per_req_tps_warm, 0.10),
        "warm_tps_p90": _percentile(per_req_tps_warm, 0.90),
        "num_warm_records": len(warm_idx),
        "warmup_discarded": min(warmup, n),
        "mean_request_warm_ms": (1000.0 * statistics.fmean(t_req[i] for i in warm_idx)) if warm_idx else float("nan"),
        "completion_tokens_min": min(comp) if comp else 0,
        "completion_tokens_median": sorted(comp)[len(comp) // 2] if comp else 0,
        "completion_tokens_max": max(comp) if comp else 0,
        "all_full_length": all(c == summary["output_len"] for c in comp),
    }


# ========================================================================== #
# Census comparison (byte-exactness)
# ========================================================================== #
def _census_compare(ref_rows: list[dict[str, Any]], cmp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(ref_rows), len(cmp_rows))
    prompt_sha_mismatch = 0
    token_mismatch_idx: list[int] = []
    first_div_pos: list[int] = []
    for i in range(n):
        r, c = ref_rows[i], cmp_rows[i]
        if r["prompt_token_sha256"] != c["prompt_token_sha256"]:
            prompt_sha_mismatch += 1
        if r["completion_token_sha256"] != c["completion_token_sha256"]:
            ri, ci = r.get("completion_token_ids") or [], c.get("completion_token_ids") or []
            m = min(len(ri), len(ci))
            pos = next((k for k in range(m) if ri[k] != ci[k]), m)
            token_mismatch_idx.append(i)
            first_div_pos.append(pos)
    return {
        "n_compared": n,
        "n_prompt_sha_mismatch": prompt_sha_mismatch,
        "prompt_sha_parity": prompt_sha_mismatch == 0,
        "n_token_mismatch": len(token_mismatch_idx),
        "byte_exact": len(token_mismatch_idx) == 0 and prompt_sha_mismatch == 0,
        "mismatch_indices": token_mismatch_idx[:32],
        "first_divergence_positions": first_div_pos[:32],
        "min_first_divergence": min(first_div_pos) if first_div_pos else None,
    }


# ========================================================================== #
# Orchestration
# ========================================================================== #
def _warm_server(base_url: str, model: str, n: int = WARMUP_REQUESTS, tokens: int = 16) -> None:
    import urllib.request
    payload = {"model": model, "prompt": "Warm up the server.", "max_tokens": tokens,
               "temperature": 0.0, "stream": False, "ignore_eos": True}
    body = json.dumps(payload).encode()
    for _ in range(n):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
        except Exception as exc:
            print(f"[census] warmup request failed (non-fatal): {exc!r}", flush=True)


def _run_pass(server_python: Path, worker_env: dict[str, str], *, base_url: str, model: str,
              out_file: Path, num_prompts: int, output_len: int, dataset_path: Path,
              tokenizer: str, request_timeout_s: int) -> dict[str, Any]:
    cmd = [
        str(server_python), str(Path(__file__).resolve()), "--decode-worker",
        "--base-url", base_url, "--model", model, "--dataset-path", str(dataset_path),
        "--tokenizer", tokenizer, "--num-prompts", str(num_prompts), "--output-len", str(output_len),
        "--seed", str(SEED), "--out-file", str(out_file), "--request-timeout-s", str(request_timeout_s),
    ]
    print(f"[census] decode pass {num_prompts}x{output_len} conc=1 -> {out_file.name}", flush=True)
    subprocess.run(cmd, check=True, timeout=7200, env=worker_env)
    return json.loads(out_file.read_text())


# vLLM 0.22.0 logs the resolved attention backend as
#   "Using AttentionBackendEnum.TRITON_ATTN backend."
# and, for gemma4's heterogeneous head_dim, hardcodes
#   "Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence."
_BACKEND_RE = re.compile(
    r"Using AttentionBackendEnum\.(\w+) backend"
    r"|Forcing (\w+) backend"
    r"|Using (\w+) backend"
)
_FORCED_RE = re.compile(r"Forcing (\w+) backend to prevent")


def _detect_backend(log_path: Path) -> dict[str, Any]:
    try:
        txt = log_path.read_text(errors="replace")
    except OSError:
        return {"resolved": None, "forced": None}
    hits: list[str] = []
    for m in _BACKEND_RE.finditer(txt):
        g = next((x for x in m.groups() if x), None)
        if g:
            hits.append(g)
    forced = _FORCED_RE.search(txt)
    return {"resolved": hits[-1] if hits else None,
            "forced": forced.group(1) if forced else None}


def _server_failed_reason(log_path: Path) -> str:
    try:
        txt = log_path.read_text(errors="replace")
    except OSError:
        return "no server log"
    tail = txt.strip().splitlines()[-8:]
    return " | ".join(t.strip() for t in tail if t.strip())[:600]


def run_config(name: str, knob: dict[str, str], *, server_python: Path, worker_env: dict[str, str],
               num_prompts: int, output_len: int, warmup: int, args: argparse.Namespace,
               want_r2: bool) -> dict[str, Any]:
    from scripts.local_validation import harness, paths  # noqa: E402

    extra_env = {"MODEL_ID": str(BUILT_CKPT), **knob}
    log_path = OUT_ROOT / f"server_{name}.log"
    result: dict[str, Any] = {"name": name, "knob": knob, "served": False}

    peak = {"mib": 0.0}
    stop = threading.Event()

    def _sample_vram() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
                if vals:
                    peak["mib"] = max(peak["mib"], max(vals))
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(2.0)

    sampler = threading.Thread(target=_sample_vram, daemon=True)
    sampler.start()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=args.port,
            startup_timeout_s=args.startup_timeout_s, log_path=log_path, extra_env=extra_env,
        ) as srv:
            result["served"] = True
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            _warm_server(srv.base_url, srv.served_model_name, n=warmup or WARMUP_REQUESTS)

            r1_file = OUT_ROOT / f"pass_{name}_r1.json"
            s1 = _run_pass(server_python, worker_env, base_url=srv.base_url,
                           model=srv.served_model_name, out_file=r1_file,
                           num_prompts=num_prompts, output_len=output_len,
                           dataset_path=paths.EVAL_PROMPTS, tokenizer=paths.TOKENIZER,
                           request_timeout_s=args.request_timeout_s)
            result["tps"] = _aggregate_tps(s1, warmup)
            result["rows_r1"] = s1["per_request"]
            if want_r2:
                r2_file = OUT_ROOT / f"pass_{name}_r2.json"
                s2 = _run_pass(server_python, worker_env, base_url=srv.base_url,
                               model=srv.served_model_name, out_file=r2_file,
                               num_prompts=num_prompts, output_len=output_len,
                               dataset_path=paths.EVAL_PROMPTS, tokenizer=paths.TOKENIZER,
                               request_timeout_s=args.request_timeout_s)
                result["rows_r2"] = s2["per_request"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["server_tail"] = _server_failed_reason(log_path)
        print(f"[census] config {name!r} did not complete: {result['error']}\n"
              f"         server tail: {result['server_tail']}", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=5)
    result["attention_backend"] = _detect_backend(log_path)
    result["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    return result


def build_report(configs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_name = {c["name"]: c for c in configs}
    ref = by_name.get(REFERENCE)
    ref_rows = (ref or {}).get("rows_r1")

    self_det = None
    if ref and ref.get("rows_r2") is not None and ref_rows is not None:
        self_det = _census_compare(ref_rows, ref["rows_r2"])

    ref_tps = (ref or {}).get("tps", {}).get("warm_median_tps") if ref else None

    per_config: list[dict[str, Any]] = []
    faster_byte_exact_plus10 = False
    faster_not_319 = False
    for c in configs:
        tps = c.get("tps", {})
        wm = tps.get("warm_median_tps")
        delta_local = (wm - ref_tps) if (isinstance(wm, float) and isinstance(ref_tps, float)) else None
        census = None
        if c["name"] != REFERENCE and ref_rows is not None and c.get("rows_r1") is not None:
            census = _census_compare(ref_rows, c["rows_r1"])
        entry = {
            "name": c["name"],
            "knob": c["knob"],
            "served": c.get("served", False),
            "error": c.get("error"),
            "attention_backend": c.get("attention_backend"),
            "warm_median_tps_local": wm,
            "warm_aggregate_tps_local": tps.get("warm_aggregate_tps"),
            "tps_delta_vs_stock_local": delta_local,
            "byte_exact_vs_stock": (census or {}).get("byte_exact") if census else (c["name"] == REFERENCE),
            "census_vs_stock": census,
            "all_full_length": tps.get("all_full_length"),
            "peak_vram_gb": c.get("peak_vram_gb"),
        }
        per_config.append(entry)
        if c["name"] != REFERENCE and isinstance(delta_local, float):
            is_exact = bool((census or {}).get("byte_exact"))
            # local +10 target with the official map as context (advisor applies map).
            if delta_local >= TPS_GAIN_TARGET and is_exact:
                faster_byte_exact_plus10 = True
            if delta_local > 0 and not is_exact:
                faster_not_319 = True

    if faster_byte_exact_plus10:
        verdict = "ZOOMOUT_LEVER_FOUND"
    elif faster_not_319:
        verdict = "ZOOMOUT_FASTER_BUT_NOT_319"
    else:
        verdict = "ZOOMOUT_NO_CHEAP_LEVER"

    return {
        "kind": "zoomout-ar-speed-census",
        "pr": 630,
        "analysis_only": True,
        "official_tps": 0,
        "engine_or_lever": "vllm-0.22.0-M1-attention-backend-knob",
        "reference_config": REFERENCE,
        "built_checkpoint": str(BUILT_CKPT),
        "num_prompts": args.num_prompts if not args.smoke else 4,
        "output_len": args.output_len if not args.smoke else 16,
        "concurrency": 1,
        "seed": SEED,
        # anchors
        "int4_head_official_tps_ref": INT4_HEAD_OFFICIAL_TPS,
        "base_int4_local_floor_tps": BASE_INT4_LOCAL_FLOOR_TPS,
        "stock_warm_median_tps_local": ref_tps,
        "implied_official_from_stock_local": (ref_tps * TAU_LO) if isinstance(ref_tps, float) else None,
        "tau_lo_context": TAU_LO,
        "tps_gain_target": TPS_GAIN_TARGET,
        # results
        "self_determinism_r1_vs_r2": self_det,
        "per_config": per_config,
        "verdict": verdict,
    }


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:
        print(f"[census] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-screen", agent="wirbel",
        name=args.wandb_name or "wirbel/zoomout-ar-speed-census",
        group=args.wandb_group or "zoomout-ar-speed-screen",
        tags=["zoomout-ar-speed", "byte-exact-census", "local-a10g", "analysis-only", "pr630"],
        notes="PR #630: local M=1 byte-exact attention-backend knob A/B on the int4_g128_lmhead rung",
        config={
            "num_prompts": report["num_prompts"], "output_len": report["output_len"],
            "concurrency": 1, "seed": SEED, "reference_config": REFERENCE,
            "configs": [c[0] for c in CONFIGS], "tau_lo": TAU_LO,
            "int4_head_official_tps_ref": INT4_HEAD_OFFICIAL_TPS,
        },
    )
    if run is None:
        print("[census] wandb init returned None — skipping", flush=True)
        return None
    flat = {k: v for k, v in report.items() if not isinstance(v, (dict, list)) and v is not None}
    for c in report["per_config"]:
        nm = c["name"]
        for key in ("warm_median_tps_local", "tps_delta_vs_stock_local", "byte_exact_vs_stock"):
            if c.get(key) is not None:
                flat[f"{nm}__{key}"] = c[key]
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="zoomout-ar-speed-census", artifact_type="speed-census", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(report: dict[str, Any]) -> None:
    line = "=" * 12 + " ZOOM-OUT AR SPEED CENSUS (PR #630) " + "=" * 12
    print("\n" + line, flush=True)
    sd = report.get("self_determinism_r1_vs_r2")
    rt = report.get("stock_warm_median_tps_local")
    print(f"  reference int4_g128_lmhead OFFICIAL = {report['int4_head_official_tps_ref']} TPS "
          f"(a10g-small; NOT locally reproducible)", flush=True)
    if isinstance(rt, float):
        print(f"  stock LOCAL warm-median = {rt:.3f} TPS  "
              f"(implied official ~ {report['implied_official_from_stock_local']:.1f}; "
              f"base-int4 local floor {report['base_int4_local_floor_tps']:.2f})", flush=True)
    if sd is not None:
        print(f"  M=1 self-determinism (r1 vs r2): byte_exact={sd['byte_exact']} "
              f"mismatch={sd['n_token_mismatch']}/{sd['n_compared']}", flush=True)
    for c in report["per_config"]:
        wm = c.get("warm_median_tps_local")
        d = c.get("tps_delta_vs_stock_local")
        wm_s = f"{wm:.2f}" if isinstance(wm, float) else "  -  "
        d_s = f"{d:+.2f}" if isinstance(d, float) else "  -  "
        be = c.get("byte_exact_vs_stock")
        ab = c.get("attention_backend") or {}
        bk = ab.get("resolved")
        print(f"  [{c['name']:>11}] served={c['served']!s:>5} backend={bk} "
              f"tps={wm_s} d_vs_stock={d_s} byte_exact_vs_stock={be} "
              f"{'ERR:' + c['error'] if c.get('error') else ''}", flush=True)
    print(f"  VERDICT: {report['verdict']}", flush=True)
    print("=" * len(line) + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="4x16 stock-only plumbing check")
    ap.add_argument("--only", default=None, help="comma list of config names to run (reference always included)")
    ap.add_argument("--no-selfcheck", action="store_true", help="skip the reference r2 self-determinism pass")
    ap.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    # decode-worker (internal; runs under the server venv)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-file")
    args = ap.parse_args(argv)

    if args.decode_worker:
        return _decode_worker(args)

    from scripts.local_validation import harness, paths  # noqa: E402

    for note in paths.prepare_local_gpu_env():
        print(f"[census] {note}", flush=True)
    if not BUILT_CKPT.exists():
        print(f"[census] FAIL: built checkpoint missing at {BUILT_CKPT}", flush=True)
        return 1

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])

    num_prompts = 4 if args.smoke else args.num_prompts
    output_len = 16 if args.smoke else args.output_len
    warmup = 0 if args.smoke else WARMUP_REQUESTS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Pin the decode worker subprocess to the SERVER venv.
    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    selected = [REFERENCE] if args.smoke else [c[0] for c in CONFIGS]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        want.add(REFERENCE)
        selected = [c[0] for c in CONFIGS if c[0] in want]
    knob_by_name = dict(CONFIGS)

    configs: list[dict[str, Any]] = []
    for name in selected:
        want_r2 = (name == REFERENCE) and not args.no_selfcheck and not args.smoke
        print(f"\n[census] === config {name} knob={knob_by_name[name]} (r2={want_r2}) ===", flush=True)
        configs.append(run_config(
            name, knob_by_name[name], server_python=server_python, worker_env=worker_env,
            num_prompts=num_prompts, output_len=output_len, warmup=warmup, args=args, want_r2=want_r2,
        ))

    report = build_report(configs, args)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report["created_at"] = stamp
    if not args.no_wandb and not args.smoke:
        report["wandb_run_id"] = log_wandb(report, args)

    # Drop the bulky per-row token-id payloads from the on-disk report (keep shas
    # inside census diagnostics; raw rows already persisted under out/pass_*.json).
    slim = json.loads(json.dumps(report))
    out_name = "census_smoke.json" if args.smoke else "census_report.json"
    (OUT_ROOT / out_name).write_text(json.dumps(slim, indent=2, sort_keys=True))
    _print_summary(report)
    print(f"[census] report: {OUT_ROOT / out_name}", flush=True)

    # NaN guard on the reference TPS (smoke uses warmup=0 -> nan is acceptable).
    rt = report.get("stock_warm_median_tps_local")
    if not args.smoke and not (isinstance(rt, float) and math.isfinite(rt)):
        print("[census] FAIL: non-finite stock warm-median TPS", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
