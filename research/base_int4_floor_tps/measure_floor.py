#!/usr/bin/env python
"""PR #533 extension — base-int4 standard-serve TPS FLOOR (local, analysis-only).

The advisor (#533, 2026-06-16) asked for the **quality-safe ship FLOOR**: the
warm-median TPS of the plain ``base-int4`` serve at the official 128x512 config,
with **no surgical layer-drop and no fast kernels** -- the guaranteed lower
bracket of what a quality-safe ship costs in TPS (fern #535's base-int4 + fast
kernels probes the upper bracket; together they bound the quality-safe ship).
Re-confirms GSM8K(base) accuracy on the SAME live serve so the floor TPS and the
gate denominator share one server session.

LOCAL only: ``analysis_only=true``, ``official_tps=0``, no HF Job, no ``--launch``,
no submission, no served-file change. PPL / greedy identity are properties of the
model+config, both unchanged.

Methodology -- mirrors the banked fleet local-TPS recipe
(``research/systems/local_official_tps_transfer/profile.py``
``_decode_worker`` / ``_aggregate_pass``) so the floor is directly comparable to
the deployed-ship LOCAL anchor (465.14 TPS, #246):

  * Official decode loop: the SAME ``decode_outputs.py`` request path the official
    ``hf_bucket_single_job.py`` uses -- 128 ShareGPT prompts, output_len 512,
    conc=1, seed=1, ``ignore_eos`` (so every request emits exactly 512 tokens),
    chat-template prompt tokenization, ``temperature=0``.
  * The official ``summary.json:tps`` is ``sglang.bench_serving``
    ``output_throughput`` (conc=1, 128x512, seed=1, 4 warmup requests DISCARDED,
    request-time only -- tokenization/IO outside the timer). We time tokenize /
    request separately per request and reconstruct that boundary on LOCAL hw, then
    add the per-request **warm MEDIAN** the advisor named.

Three clearly-labelled flavours are reported; ``base_int4_floor_tps`` is the
warm-median (the advisor's wording):

  * ``warm_median_tps``     -- median over warm requests of completion_tokens/req_time.
  * ``warm_aggregate_tps``  -- N_warm / T_req_warm == the official output_throughput
                               boundary reconstructed on LOCAL hw (conc=1).
  * ``wall_tps``            -- N_all / (T_tokenize + T_request), the #246-anchor-style
                               wall metric (here without per-request file IO).

The local->official map (tau_lo = 481.53/465.14 ~= 1.0352, #267) is reported as
CONTEXT only -- the advisor applies the map; this leg does not claim an official
number.

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/base_int4_floor_tps/measure_floor.py --smoke --no-wandb
Full floor + GSM8K re-confirm:
  CUDA_VISIBLE_DEVICES=0 python research/base_int4_floor_tps/measure_floor.py \
    --wandb_name wirbel/base-int4-floor-tps --wandb_group gsm8k-base-ship-fullhead
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# This script re-invokes itself as a --decode-worker under the SERVER venv, where
# transformers -> torch._dynamo -> cProfile does `import profile`. Drop the bare
# cwd / script-dir entries so a stray module never shadows stdlib; re-add ROOT for
# `from scripts.local_validation import ...`.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBMISSION = ROOT / "submissions" / "int4_base_aime"
OUT_ROOT = ROOT / "research" / "base_int4_floor_tps"
GSM8K_EVAL = ROOT / "research" / "downstream_quality_gsm8k" / "gsm8k_eval.py"

NUM_PROMPTS = 128
OUTPUT_LEN = 512
SEED = 1
# official sglang.bench_serving discards this many warmup requests from timing.
WARMUP_REQUESTS = 4

# Imported context only (PR #267; NOT re-derived, NOT a claim of this leg).
LOCAL_ANCHOR_TPS = 465.14047160458415  # deployed ship LOCAL warm anchor (#246)
OFFICIAL_ANCHOR_TPS = 481.53           # deployed ship OFFICIAL (#52)
TAU_LO = OFFICIAL_ANCHOR_TPS / LOCAL_ANCHOR_TPS
OFFICIAL_GATE_TPS = 500.0

# GSM8K base gate denominator (PR #533, run mo0ci0yl): sampled 0.878 / greedy 0.896.
GSM8K_BASE_SAMPLED = 0.878


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
            base_url=args.base_url,
            model=args.model,
            prompt_token_ids=prompt_token_ids,
            output_len=args.output_len,
            timeout_s=args.request_timeout_s,
        )
        t2 = time.perf_counter()
        choice = od.choice_from_response(response)
        completion_token_ids, _, _ = od.extract_generated_token_ids(response, choice, prompt_token_ids)
        rows.append({
            "index": index,
            "t_tokenize_s": t1 - t0,
            "t_request_s": t2 - t1,
            "num_prompt_tokens": len(prompt_token_ids),
            "num_completion_tokens": len(completion_token_ids),
        })
        print(f"[worker] {index + 1}/{len(records)} req_ms={1000.0 * (t2 - t1):.1f} "
              f"comp={len(completion_token_ids)} prompt={len(prompt_token_ids)}", flush=True)

    out = {"output_len": args.output_len, "num_records": len(records), "per_request": rows}
    Path(args.out_file).write_text(json.dumps(out, indent=2))
    return 0


# ========================================================================== #
# Aggregation
# ========================================================================== #
def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, k))]


def _aggregate(summary: dict[str, Any], warmup: int = WARMUP_REQUESTS) -> dict[str, Any]:
    rows = summary["per_request"]
    n = len(rows)
    t_tok = [r["t_tokenize_s"] for r in rows]
    t_req = [r["t_request_s"] for r in rows]
    n_comp = [r["num_completion_tokens"] for r in rows]

    warm_idx = list(range(min(warmup, n), n))
    T_req_warm = sum(t_req[i] for i in warm_idx)
    N_warm = sum(n_comp[i] for i in warm_idx)
    per_req_tps_warm = sorted(n_comp[i] / t_req[i] for i in warm_idx if t_req[i] > 0)

    warm_aggregate_tps = N_warm / T_req_warm if T_req_warm else float("nan")
    warm_median_tps = statistics.median(per_req_tps_warm) if per_req_tps_warm else float("nan")
    warm_mean_tps = statistics.fmean(per_req_tps_warm) if per_req_tps_warm else float("nan")

    T_wall = sum(t_tok) + sum(t_req)
    N_all = sum(n_comp)
    wall_tps = N_all / T_wall if T_wall else float("nan")

    mean_req_warm_ms = 1000.0 * statistics.fmean(t_req[i] for i in warm_idx) if warm_idx else float("nan")
    cold = t_req[:min(warmup, n)]
    mean_req_cold_ms = 1000.0 * statistics.fmean(cold) if cold else float("nan")

    comp_sorted = sorted(n_comp)
    return {
        "output_len": summary["output_len"],
        "num_records": n,
        "num_warm_records": len(warm_idx),
        "warmup_discarded": min(warmup, n),
        # headline flavours
        "warm_median_tps": warm_median_tps,
        "warm_aggregate_tps": warm_aggregate_tps,
        "warm_mean_tps": warm_mean_tps,
        "wall_tps": wall_tps,
        # spread of the per-request warm distribution
        "warm_tps_p10": _percentile(per_req_tps_warm, 0.10),
        "warm_tps_p90": _percentile(per_req_tps_warm, 0.90),
        "warm_tps_min": per_req_tps_warm[0] if per_req_tps_warm else float("nan"),
        "warm_tps_max": per_req_tps_warm[-1] if per_req_tps_warm else float("nan"),
        # request-time detail
        "mean_request_warm_ms": mean_req_warm_ms,
        "mean_request_cold_ms": mean_req_cold_ms,
        "warmup_drag_ms_per_req": mean_req_cold_ms - mean_req_warm_ms,
        # completion-token sanity (ignore_eos => expect == output_len)
        "completion_tokens_min": comp_sorted[0] if comp_sorted else 0,
        "completion_tokens_median": comp_sorted[len(comp_sorted) // 2] if comp_sorted else 0,
        "completion_tokens_max": comp_sorted[-1] if comp_sorted else 0,
        "completion_tokens_total": N_all,
        "all_full_length": all(c == summary["output_len"] for c in n_comp),
        "T_request_warm_s": T_req_warm,
        "T_wall_s": T_wall,
    }


# ========================================================================== #
# Orchestration
# ========================================================================== #
def _warm_server(base_url: str, model: str, n: int = 4, tokens: int = 16) -> None:
    """Fire a few short /v1/completions so first-request lazy init is paid before
    the timed pass (belt-and-braces on top of the warm slice)."""
    import urllib.request

    payload = {
        "model": model, "prompt": "Warm up the server.", "max_tokens": tokens,
        "temperature": 0.0, "stream": False, "ignore_eos": True,
    }
    body = json.dumps(payload).encode()
    for _ in range(n):
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
        except Exception as exc:  # warmup failure is non-fatal
            print(f"[floor] warmup request failed (non-fatal): {exc!r}", flush=True)


def _run_decode_pass(server_python: Path, worker_env: dict[str, str], *, base_url: str,
                     model: str, out_file: Path, num_prompts: int, output_len: int,
                     dataset_path: Path, tokenizer: str, request_timeout_s: int) -> dict[str, Any]:
    cmd = [
        str(server_python), str(Path(__file__).resolve()), "--decode-worker",
        "--base-url", base_url, "--model", model,
        "--dataset-path", str(dataset_path), "--tokenizer", tokenizer,
        "--num-prompts", str(num_prompts), "--output-len", str(output_len),
        "--seed", str(SEED), "--out-file", str(out_file),
        "--request-timeout-s", str(request_timeout_s),
    ]
    print(f"[floor] decode pass: {num_prompts}x{output_len} conc=1 -> {out_file}", flush=True)
    subprocess.run(cmd, check=True, timeout=5400, env=worker_env)
    return json.loads(out_file.read_text())


def _gsm8k_reconfirm(base_url: str, out_dir: Path, *, n: int, label: str) -> dict[str, Any] | None:
    """Re-run GSM8K base sampled against the SAME live serve (base-url mode)."""
    cmd = [
        sys.executable, str(GSM8K_EVAL), "--base-url", base_url,
        "--label", label, "--regimes", "sampled", "--n", str(n), "--seed", "1234",
        "--concurrency", "32", "--max-tokens", "512", "--out-dir", str(out_dir),
    ]
    print(f"[floor] GSM8K re-confirm (base, sampled, n={n}) on the same serve", flush=True)
    rc = subprocess.run(cmd, check=False, timeout=3600).returncode
    out_path = out_dir / f"{label}_sampled.json"
    if rc != 0 or not out_path.exists():
        print(f"[floor] GSM8K re-confirm rc={rc}; out exists={out_path.exists()}", flush=True)
        return None
    d = json.loads(out_path.read_text())
    return {
        "accuracy": d["accuracy"], "n_correct": d["n_correct"], "n_problems": d["n_problems"],
        "strict_rate": d["strict_rate"], "extract_fail_rate": d["extract_fail_rate"],
        "truncation_rate": d["truncation_rate"], "wall_s": d["wall_s"],
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.local_validation import harness, paths  # noqa: E402

    # PR #541: optionally profile a DIFFERENT submission with extra serve-env
    # overrides (the base_fullhead fast-upper anchor = surgical kernels on the stock
    # 262k-head base-int4). Default stays the int4_base_aime base-int4 floor.
    submission = args.submission or SUBMISSION
    label = args.label or submission.name
    serve_env: dict[str, str] = {}
    for kv in args.serve_env:
        k, _, v = kv.partition("=")
        serve_env[k.strip()] = v

    for note in paths.prepare_local_gpu_env():
        print(f"[floor] {note}", flush=True)
    manifest = harness.load_manifest(submission)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])

    num_prompts = 4 if args.smoke else NUM_PROMPTS
    output_len = 32 if args.smoke else OUTPUT_LEN
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / f"server_{label}_floor.log"

    # Pin the worker subprocess to the SERVER venv (mirror harness._participant_env).
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

    measured: dict[str, Any] = {
        "submission": str(submission),
        "label": label,
        "serve_env": serve_env,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "concurrency": 1,
        "seed": SEED,
        "warmup_requests_discarded": WARMUP_REQUESTS,
    }
    sampler = threading.Thread(target=_sample_vram, daemon=True)
    sampler.start()
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=args.port,
            startup_timeout_s=1800, log_path=log_path,
            extra_env=serve_env,
        ) as srv:
            measured["model_id"] = srv.model_id
            measured["served_model_name"] = srv.served_model_name
            print(f"[floor] warming server ({srv.base_url})", flush=True)
            _warm_server(srv.base_url, srv.served_model_name, n=WARMUP_REQUESTS)

            pass_file = OUT_ROOT / ("smoke_pass.json" if args.smoke else f"{label}_floor_pass.json")
            summary = _run_decode_pass(
                server_python, worker_env, base_url=srv.base_url, model=srv.served_model_name,
                out_file=pass_file, num_prompts=num_prompts, output_len=output_len,
                dataset_path=paths.EVAL_PROMPTS, tokenizer=paths.TOKENIZER,
                request_timeout_s=args.request_timeout_s,
            )
            measured["tps"] = _aggregate(summary)

            if not args.smoke and not args.no_gsm8k:
                gsm_n = args.gsm8k_n
                measured["gsm8k_reconfirm"] = _gsm8k_reconfirm(
                    srv.base_url, OUT_ROOT, n=gsm_n, label="base_reconfirm"
                )
    finally:
        stop.set()
        sampler.join(timeout=5)

    measured["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    return measured


def build_report(measured: dict[str, Any]) -> dict[str, Any]:
    tps = measured["tps"]
    floor = tps["warm_median_tps"]
    gsm = measured.get("gsm8k_reconfirm") or {}
    gsm_acc = gsm.get("accuracy")
    report = {
        "analysis_only": True,
        "official_tps": 0,
        "tps_delta": 0.0,
        "base_int4_floor_tps": floor,                       # headline = warm-median (advisor wording)
        "warm_median_tps": floor,
        "warm_aggregate_tps": tps["warm_aggregate_tps"],    # official output_throughput boundary (local)
        "wall_tps": tps["wall_tps"],
        "warm_mean_tps": tps["warm_mean_tps"],
        "warm_tps_p10": tps["warm_tps_p10"],
        "warm_tps_p90": tps["warm_tps_p90"],
        "mean_request_warm_ms": tps["mean_request_warm_ms"],
        "all_full_length": tps["all_full_length"],
        "completion_tokens_median": tps["completion_tokens_median"],
        "num_warm_records": tps["num_warm_records"],
        "peak_vram_gb": measured.get("peak_vram_gb"),
        "model_id": measured.get("model_id"),
        "submission": measured.get("submission"),
        # local->official map CONTEXT ONLY (advisor applies it; not a claim here)
        "tau_lo": TAU_LO,
        "implied_official_from_warm_median_tps": floor * TAU_LO if isinstance(floor, float) else None,
        "implied_official_from_warm_aggregate_tps": tps["warm_aggregate_tps"] * TAU_LO,
        "official_gate_tps": OFFICIAL_GATE_TPS,
        # GSM8K re-confirm on the SAME serve
        "gsm8k_base_reconfirm_acc": gsm_acc,
        "gsm8k_base_gate_denominator": GSM8K_BASE_SAMPLED,
        "gsm8k_reconfirm_matches": (
            abs(gsm_acc - GSM8K_BASE_SAMPLED) <= 0.02 if isinstance(gsm_acc, float) else None
        ),
        "gsm8k_reconfirm_detail": gsm,
        "measured": measured,
    }
    return report


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # pragma: no cover
        print(f"[floor] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile",
        agent="wirbel",
        name=args.wandb_name or "wirbel/base-int4-floor-tps",
        group=args.wandb_group or "gsm8k-base-ship-fullhead",
        tags=["base-int4-floor", "quality-safe-floor", "local-a10g", "tps-floor", "analysis-only"],
        notes="PR #533 ext: base-int4 standard-serve warm-median TPS floor + GSM8K(base) re-confirm",
        config={
            "submission": report["measured"].get("submission", str(SUBMISSION)),
            "label": report["measured"].get("label"),
            "serve_env": report["measured"].get("serve_env", {}),
            "num_prompts": report["measured"]["num_prompts"],
            "output_len": report["measured"]["output_len"],
            "concurrency": 1, "seed": SEED, "warmup_requests": WARMUP_REQUESTS,
            "tau_lo": TAU_LO,
        },
    )
    if run is None:
        print("[floor] wandb init returned None — skipping", flush=True)
        return None
    summary = {k: v for k, v in report.items() if not isinstance(v, (dict, list)) and v is not None}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="base-int4-floor-report", artifact_type="floor-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(report: dict[str, Any]) -> None:
    m = report["measured"]
    t = m["tps"]
    line = "=" * 14 + " BASE-INT4 STANDARD-SERVE TPS FLOOR (PR #533) " + "=" * 14
    print("\n" + line, flush=True)
    print(f"  submission: {report['submission']}  model: {report['model_id']}", flush=True)
    print(f"  config: {m['num_prompts']}x{m['output_len']} conc=1 seed={SEED} "
          f"(warm={t['num_warm_records']}, {WARMUP_REQUESTS} discarded)", flush=True)
    print(f"  base_int4_floor_tps (warm-median) = {report['base_int4_floor_tps']:.3f} TPS", flush=True)
    print(f"    warm-aggregate (official boundary) = {t['warm_aggregate_tps']:.3f}  "
          f"wall = {t['wall_tps']:.3f}  warm-mean = {t['warm_mean_tps']:.3f}", flush=True)
    print(f"    warm per-req TPS p10/p90 = {t['warm_tps_p10']:.2f}/{t['warm_tps_p90']:.2f}  "
          f"mean_req_warm = {t['mean_request_warm_ms']:.1f} ms", flush=True)
    print(f"    completion tokens: median={t['completion_tokens_median']} "
          f"all_full_length={t['all_full_length']}  peak_vram={report['peak_vram_gb']:.2f} GiB", flush=True)
    print(f"  context map (tau_lo={TAU_LO:.5f}): implied official from warm-median "
          f"~ {report['implied_official_from_warm_median_tps']:.1f} TPS "
          f"(gate {OFFICIAL_GATE_TPS:.0f})", flush=True)
    g = report.get("gsm8k_base_reconfirm_acc")
    if g is not None:
        print(f"  GSM8K(base) re-confirm (same serve) = {g:.4f}  "
              f"(denominator {GSM8K_BASE_SAMPLED}; matches={report['gsm8k_reconfirm_matches']})", flush=True)
    print("=" * len(line) + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="4x32 plumbing check, no GSM8K")
    ap.add_argument("--no-gsm8k", action="store_true", help="skip the GSM8K re-confirm")
    ap.add_argument("--gsm8k-n", type=int, default=500, help="GSM8K re-confirm item count")
    ap.add_argument("--submission", type=Path, default=None,
                    help="submission dir to profile (default: int4_base_aime base-int4 floor). "
                         "PR #541 reuses this for the base_fullhead fast-upper anchor.")
    ap.add_argument("--serve-env", action="append", default=[], metavar="KEY=VAL",
                    help="extra serve env override for the submission (repeatable), e.g. LM_HEAD_PRUNE=0")
    ap.add_argument("--label", default=None, help="output/report name suffix (default: derived from submission)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=300)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")

    # decode-worker (internal; runs under the server venv)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-file")
    args = ap.parse_args(argv)

    if args.decode_worker:
        return _decode_worker(args)

    measured = run_measurement(args)
    report = build_report(measured)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report["created_at"] = stamp
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.no_wandb and not args.smoke:
        report["wandb_run_id"] = log_wandb(report, args)
    # Default base-int4 floor keeps the canonical report.json (PR #533). A
    # non-default submission (PR #541 base_fullhead anchor) writes a label-specific
    # report so it never clobbers the banked floor artifact.
    if args.smoke:
        out_name = "smoke_report.json"
    elif args.submission is not None:
        out_name = f"report_{report['measured'].get('label') or 'custom'}.json"
    else:
        out_name = "report.json"
    (OUT_ROOT / out_name).write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_summary(report)
    print(f"[floor] report: {OUT_ROOT / out_name}", flush=True)
    # NaN guard on the headline floor.
    floor = report["base_int4_floor_tps"]
    if not (isinstance(floor, float) and math.isfinite(floor)):
        print("[floor] FAIL: non-finite base_int4_floor_tps", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
