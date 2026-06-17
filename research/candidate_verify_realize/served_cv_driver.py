#!/usr/bin/env python
"""PR #566 fern — SERVED candidate-verify realization: wire the int4-nominator+verify
head into the LIVE vLLM sampler logits path on base_fullhead and MEASURE it end-to-end
(local A10G, analysis-only, NO HF fire). Converts #560's microbench+reprojection
(cv_served_tps_is_measured=False, projected 291.36 TPS) into a real served number.

Three passes on the SAME base_fullhead serve recipe (full 262k bf16 head, prune OFF,
spec-alive MTP K=7 -> M=8 verify, the wirbel #553 / lawine #544 252.31 anchor config):

  * REF   (CV_HEAD=0): unmodified bf16 full head. Measures the reference served TPS on
    THIS pod RIGHT NOW + captures the greedy completion token ids (the identity oracle).
  * CVSP  (CV_HEAD=1, audit off): compute_logits REPLACED by int4-nominator GEMV ->
    top-8 -> bf16 verify -> scatter into [M,vocab] (the server's own argmax then picks).
    Measures the HEADLINE served CV TPS + captures completion token ids.
  * CVAUD (CV_HEAD=1, audit on, SHORT): per decode step also runs the bf16 oracle and
    compares argmax -> live per-step identity rate at matched states.

Identity (headline): REF vs CVSP completion token ids, per request -> byte-exact
sequence match + first-divergence localization. Cross-checked by CVAUD per-step rate.

LOCAL only: analysis_only, official_tps=0, no HF Job, no --launch, no submission, no
served-file change (base_fullhead + the CV head are reached purely by serve-env
overrides + the default-off sitecustomize probe).

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/candidate_verify_realize/served_cv_driver.py --smoke --no-wandb
Full:
  CUDA_VISIBLE_DEVICES=0 python research/candidate_verify_realize/served_cv_driver.py \
    --num-prompts 24 --audit-prompts 8 \
    --wandb_name fern/served-cv-realize --wandb_group served-cv-realize
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
MEASURE_FLOOR = ROOT / "research" / "base_int4_floor_tps" / "measure_floor.py"
OUT_ROOT = HERE

MODEL_DIR = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)

HIDDEN = 2560
VOCAB = 262144
HEAD_BYTES_BF16 = HIDDEN * VOCAB * 2          # 1.342 GB
KSAFE = 8
GROUP = 128
SEED = 1
# On the 23 GB A10G the shipped base_fullhead serves conc=1 with only ~+0.13 GB KV
# headroom at util=0.90; the +0.36 GB int4 nominator pushes KV to -0.23 GB (boot fails).
# Raise util to fund the int4 (each 0.01 util ~= 0.225 GB here). Applied to BOTH REF and
# CV passes so the served TPS delta stays apples-to-apples; util only changes KV capacity,
# not single-stream decode speed. Override via CV_GPU_MEM_UTIL.
GPU_MEM_UTIL = os.environ.get("CV_GPU_MEM_UTIL", "0.92")

# Cited anchors (NOT re-derived here).
ANCHOR_BASE_FULLHEAD = 252.31                  # wirbel #553 / lawine #544 base_fullhead
REPROJ_560_CENTRAL = 291.36223894548687        # #560 stage2 cv_realized_quality_safe_tps (252 basis)
REPROJ_560_GAIN = 39.05223894548686            # #560 stage2 gain on 252.31
PROJ_549_BAND = (28.251525006837994, 43.68323839224104)  # #549 int4_g128 [pess,opt] gain band
SURGICAL_357_SHIP = 375.857                    # speed gate the flip-condition requires exceeding
OFFICIAL_1 = 481.53                            # public #1 (fa2sw_precache_kenyan, PR #52)
A10G_HBM_GBPS = 600.0


def build_env(*, cv: bool, audit: bool, model_dir: str, audit_out: str | None,
              ksafe: int = KSAFE) -> dict[str, str]:
    """base_fullhead serve recipe (full 262k head, prune OFF) + CV-head probe env."""
    env: dict[str, str] = {
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",
        "LOCAL_MODEL_DIR": model_dir,
        "PLE_FOLD_TARGET_MODEL": model_dir,
        "LM_HEAD_FULL_REQUIRE": "1",
        "GPU_MEMORY_UTILIZATION": GPU_MEM_UTIL,
        # research-dir FIRST so `import sitecustomize` resolves to our CV shim
        "PYTHONPATH": f"{HERE}{os.pathsep}{SUBMISSION}",
        "CV_PKG_DIR": str(SUBMISSION),
        "CV_HEAD": "1" if cv else "0",
        "CV_AUDIT": "1" if audit else "0",
        "CV_KSAFE": str(ksafe),
        "CV_GROUP": str(GROUP),
        "CV_WARMUP_SKIP": "64",
        "CV_REPORT_EVERY": "2000",
        "CV_DECODE_M_MAX": "8",
    }
    if audit_out:
        env["CV_AUDIT_OUT"] = audit_out
    return env


# ========================================================================== #
# token-capturing decode worker (runs UNDER the server venv)
# ========================================================================== #
def _decode_worker(args: argparse.Namespace) -> int:
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
            "t_tokenize_s": t1 - t0,
            "t_request_s": t2 - t1,
            "num_prompt_tokens": len(prompt_token_ids),
            "num_completion_tokens": len(completion_token_ids),
            "completion_token_ids": list(completion_token_ids),
        })
        print(f"[cv-worker] {index + 1}/{len(records)} req_ms={1000.0 * (t2 - t1):.1f} "
              f"comp={len(completion_token_ids)} prompt={len(prompt_token_ids)}", flush=True)

    out = {"output_len": args.output_len, "num_records": len(records), "per_request": rows}
    Path(args.out_file).write_text(json.dumps(out))
    return 0


# ========================================================================== #
# server pass
# ========================================================================== #
def run_pass(mf: Any, harness: Any, paths: Any, *, server_python: Path, label: str,
             extra_env: dict[str, str], num_prompts: int, output_len: int, port: int,
             request_timeout_s: int) -> dict[str, Any]:
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
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            measured["model_id"] = srv.model_id
            measured["served_model_name"] = srv.served_model_name
            print(f"[cv] [{label}] warming server ({srv.base_url})", flush=True)
            mf._warm_server(srv.base_url, srv.served_model_name, n=mf.WARMUP_REQUESTS)
            cmd = [
                str(server_python), str(Path(__file__).resolve()), "--decode-worker",
                "--base-url", srv.base_url, "--model", srv.served_model_name,
                "--dataset-path", str(paths.EVAL_PROMPTS), "--tokenizer", paths.TOKENIZER,
                "--num-prompts", str(num_prompts), "--output-len", str(output_len),
                "--seed", str(SEED), "--out-file", str(pass_file),
                "--request-timeout-s", str(request_timeout_s),
            ]
            print(f"[cv] [{label}] decode pass {num_prompts}x{output_len} conc=1 -> {pass_file}", flush=True)
            subprocess.run(cmd, check=True, timeout=5400, env=worker_env)
            summary = json.loads(pass_file.read_text())
            try:
                measured["tps"] = mf._aggregate(summary)
            except Exception as exc:  # small-prompt audit passes have no warm rows
                measured["tps"] = {"warm_median_tps": float("nan"), "aggregate_error": repr(exc)}
            measured["per_request"] = summary["per_request"]
    finally:
        stop.set()
        sampler.join(timeout=5)

    measured["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    measured["log_path"] = str(log_path)
    return measured


def grep_log(log_path: str, needles: list[str]) -> dict[str, bool]:
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return {n: False for n in needles}
    return {n: (n in text) for n in needles}


# ========================================================================== #
# identity (REF vs CV completion token ids)
# ========================================================================== #
def compare_identity(ref_rows: list[dict], cv_rows: list[dict]) -> dict[str, Any]:
    by_ref = {r["index"]: r for r in ref_rows}
    by_cv = {r["index"]: r for r in cv_rows}
    common = sorted(set(by_ref) & set(by_cv))
    seq_exact = 0
    total_tokens = 0
    matched_tokens = 0
    matched_prefix_tokens = 0   # tokens before the first divergence (cascade-aware)
    first_divergences: list[dict] = []
    for idx in common:
        a = by_ref[idx]["completion_token_ids"]
        b = by_cv[idx]["completion_token_ids"]
        n = min(len(a), len(b))
        total_tokens += max(len(a), len(b))
        eqmask = [a[i] == b[i] for i in range(n)]
        matched_tokens += sum(eqmask)
        # first divergence
        div = next((i for i in range(n) if a[i] != b[i]), None)
        if div is None and len(a) == len(b):
            seq_exact += 1
            matched_prefix_tokens += n
        else:
            dpos = div if div is not None else n
            matched_prefix_tokens += dpos
            if len(first_divergences) < 16:
                first_divergences.append({
                    "index": idx, "first_divergence_token": dpos,
                    "len_ref": len(a), "len_cv": len(b),
                    "ref_tok": a[dpos] if dpos < len(a) else None,
                    "cv_tok": b[dpos] if dpos < len(b) else None,
                })
    return {
        "n_sequences": len(common),
        "n_sequences_byte_exact": seq_exact,
        "sequence_exact_rate": seq_exact / len(common) if common else None,
        "total_tokens": total_tokens,
        "matched_tokens": matched_tokens,
        "token_identity_rate": matched_tokens / total_tokens if total_tokens else None,
        "matched_prefix_tokens": matched_prefix_tokens,
        "prefix_identity_rate": matched_prefix_tokens / total_tokens if total_tokens else None,
        "first_divergences": first_divergences,
    }


# ========================================================================== #
# synthesis
# ========================================================================== #
def synthesize(ref: dict, cvsp: dict, ident: dict, audit: dict | None) -> dict[str, Any]:
    ref_tps = ref["tps"]["warm_median_tps"]
    cv_tps = cvsp["tps"]["warm_median_tps"]
    gain = cv_tps - ref_tps
    speedup = cv_tps / ref_tps if ref_tps else None
    # gain re-based onto the cited 252.31 anchor (same wall-speedup applied to the anchor)
    gain_on_anchor = ANCHOR_BASE_FULLHEAD * (speedup - 1.0) if speedup else None
    cv_tps_on_anchor = ANCHOR_BASE_FULLHEAD * speedup if speedup else None
    in_549_band = (PROJ_549_BAND[0] <= gain <= PROJ_549_BAND[1]) if gain is not None else None
    audit_rate = audit.get("identity_rate") if audit else None
    return {
        "ref_served_tps_measured": ref_tps,
        "cv_served_tps_measured": cv_tps,
        "cv_served_tps_is_measured": True,
        "served_gain_tps": gain,
        "served_wall_speedup": speedup,
        "served_gain_on_252_anchor": gain_on_anchor,
        "cv_served_tps_on_252_anchor": cv_tps_on_anchor,
        "reproj_560_central_252": REPROJ_560_CENTRAL,
        "reproj_560_gain_252": REPROJ_560_GAIN,
        "cv_realized_vs_reproj560_delta": (cv_tps_on_anchor - REPROJ_560_CENTRAL)
                                          if cv_tps_on_anchor else None,
        "in_549_gain_band": in_549_band,
        "proj_549_band": list(PROJ_549_BAND),
        "anchor_base_fullhead_252": ANCHOR_BASE_FULLHEAD,
        "surgical_357_ship": SURGICAL_357_SHIP,
        "clears_surgical_357_ship": (cv_tps_on_anchor > SURGICAL_357_SHIP)
                                    if cv_tps_on_anchor else None,
        "official_1_tps": OFFICIAL_1,
        "gap_to_surgical_357": (SURGICAL_357_SHIP - cv_tps_on_anchor)
                               if cv_tps_on_anchor else None,
        # identity
        "cv_served_argmax_identity_rate": ident["token_identity_rate"],
        "cv_served_sequence_exact_rate": ident["sequence_exact_rate"],
        "cv_served_n_sequences_byte_exact": ident["n_sequences_byte_exact"],
        "cv_served_n_sequences": ident["n_sequences"],
        "cv_audit_per_step_identity_rate": audit_rate,
        "tie_break_convention": "server-native argmax over scattered [M,vocab] verify "
                                "logits == lowest-vocab-index (self-consistent, K_safe=8 "
                                "miss_rate=0 -> winner always in shortlist)",
        "peak_vram_gb": max(ref.get("peak_vram_gb", 0.0), cvsp.get("peak_vram_gb", 0.0)),
        "analysis_only": True,
        "official_tps": 0,
    }


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[cv] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile", agent="fern",
        name=args.wandb_name or "fern/served-cv-realize",
        group=args.wandb_group or "served-cv-realize",
        tags=["candidate-verify", "served", "head-ceiling", "local-a10g",
              "analysis-only", "pr566"],
        notes="PR #566: realize the candidate-verify head served end-to-end (convert "
              "#560's 291.36 reprojection into a measured served TPS + live identity)",
        config={
            "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
            "num_prompts": args.num_prompts, "audit_prompts": args.audit_prompts,
            "output_len": args.output_len, "concurrency": 1, "seed": SEED,
            "k_safe": KSAFE, "group": GROUP, "hidden": HIDDEN, "vocab": VOCAB,
        },
    )
    if run is None:
        return None
    s = report["synthesis"]
    summary = {k: v for k, v in s.items()
               if isinstance(v, (int, float, bool)) and (not isinstance(v, float) or math.isfinite(v))}
    summary["primary_metric"] = s["cv_served_tps_measured"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="served-cv-realize-report", artifact_type="cv-served-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(s: dict[str, Any]) -> None:
    line = "=" * 12 + " PR #566 — SERVED CANDIDATE-VERIFY REALIZATION " + "=" * 12
    print("\n" + line, flush=True)
    print(f"  REF served TPS (this pod)      = {s['ref_served_tps_measured']:.2f}", flush=True)
    print(f"  CV  served TPS (MEASURED)      = {s['cv_served_tps_measured']:.2f}  "
          f"(speedup {s['served_wall_speedup']:.4f}x, +{s['served_gain_tps']:.2f})", flush=True)
    print(f"  CV  TPS re-based on 252.31     = {s['cv_served_tps_on_252_anchor']:.2f}  "
          f"(+{s['served_gain_on_252_anchor']:.2f})", flush=True)
    print(f"  #560 reproj (252 basis)        = {s['reproj_560_central_252']:.2f}  "
          f"(measured-vs-reproj {s['cv_realized_vs_reproj560_delta']:+.2f})", flush=True)
    print(f"  in #549 gain band [{s['proj_549_band'][0]:.1f},{s['proj_549_band'][1]:.1f}]? = {s['in_549_gain_band']}", flush=True)
    print(f"  clears surgical-357 ship 375.857? = {s['clears_surgical_357_ship']}  "
          f"(gap {s['gap_to_surgical_357']:+.2f})", flush=True)
    print(f"  >>> live argmax identity       = {s['cv_served_argmax_identity_rate']}  "
          f"(seq byte-exact {s['cv_served_n_sequences_byte_exact']}/{s['cv_served_n_sequences']})", flush=True)
    print(f"  >>> audit per-step identity    = {s['cv_audit_per_step_identity_rate']}", flush=True)
    print(f"  peak VRAM                      = {s['peak_vram_gb']:.2f} GB", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# main
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="4x32 plumbing check (REF + CVSP + tiny CVAUD)")
    ap.add_argument("--num-prompts", type=int, default=24, help="prompts for the timed REF/CVSP passes")
    ap.add_argument("--audit-prompts", type=int, default=8, help="prompts for the CVAUD identity pass")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--no-audit", action="store_true", help="skip the CVAUD per-step identity pass")
    ap.add_argument("--audit-only", action="store_true",
                    help="run ONLY a CVAUD per-step identity pass (for a K_safe sweep)")
    ap.add_argument("--cv-ksafe", type=int, default=KSAFE,
                    help="shortlist width for --audit-only (default 8)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    # internal worker mode (runs under the server venv)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer"); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-file")
    args = ap.parse_args(argv)

    if args.decode_worker:
        # reuse --num-prompts as the worker prompt count
        return _decode_worker(args)

    mf_spec = importlib.util.spec_from_file_location("measure_floor", str(MEASURE_FLOOR))
    mf = importlib.util.module_from_spec(mf_spec)
    assert mf_spec and mf_spec.loader
    mf_spec.loader.exec_module(mf)
    from scripts.local_validation import harness, paths

    for note in paths.prepare_local_gpu_env():
        print(f"[cv] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- SMOKE: a single CVAUD pass de-risks everything in one server start: server
    # boots with the CV head + base_fullhead recipe, int4 nominator builds, compute_logits
    # is REPLACED, tokens flow through the live decode loop, AND the per-step audit reports
    # identity vs the bf16 oracle (no separate REF run needed). TPS here is meaningless. ---
    if args.smoke:
        audit_out = str(OUT_ROOT / "cv_audit_smoke.json")
        cvaud = run_pass(mf, harness, paths, server_python=server_python, label="cvaud_smoke",
                         extra_env=build_env(cv=True, audit=True, model_dir=args.model_dir, audit_out=audit_out),
                         num_prompts=4, output_len=48, port=args.port,
                         request_timeout_s=args.request_timeout_s)
        plumbing = grep_log(cvaud["log_path"], [
            "[cv-head] finder registered", "[cv-head] REPLACED Gemma4ForCausalLM.compute_logits",
            "[cv-head] built int4 nominator"])
        audit = json.loads(Path(audit_out).read_text()) if Path(audit_out).exists() else None
        print(f"\n[cv] SMOKE plumbing={plumbing}", flush=True)
        print(f"[cv] SMOKE audit={audit and {k: audit[k] for k in ('identity_rate','audit_total','audit_mismatch','tie_positions','cv_head_ms_mean','m_hist') if k in audit}}", flush=True)
        ok = all(plumbing.values()) and audit is not None and audit.get("audit_mismatch") == 0
        print(f"[cv] SMOKE {'PASS' if ok else 'CHECK'} ({time.time()-t_start:.0f}s)", flush=True)
        return 0 if ok else 1

    # --- AUDIT-ONLY: a single CVAUD per-step identity pass at a chosen K_safe (K sweep to
    # characterize whether a wider shortlist recovers the bf16-tie misses). No TPS, no REF. ---
    if args.audit_only:
        k = args.cv_ksafe
        audit_out = str(OUT_ROOT / f"cv_audit_k{k}.json")
        cvaud = run_pass(mf, harness, paths, server_python=server_python, label=f"cvaud_k{k}",
                         extra_env=build_env(cv=True, audit=True, model_dir=args.model_dir,
                                             audit_out=audit_out, ksafe=k),
                         num_prompts=args.audit_prompts, output_len=args.output_len, port=args.port,
                         request_timeout_s=args.request_timeout_s)
        audit = json.loads(Path(audit_out).read_text()) if Path(audit_out).exists() else None
        print(f"\n[cv] AUDIT-ONLY K_safe={k}: "
              f"identity={audit and audit.get('identity_rate')} "
              f"rows={audit and audit.get('audit_total')} "
              f"mismatch={audit and audit.get('audit_mismatch')} "
              f"ties={audit and audit.get('tie_positions')} "
              f"m_hist={audit and audit.get('m_hist')} ({time.time()-t_start:.0f}s)", flush=True)
        return 0

    num_prompts = args.num_prompts
    output_len = args.output_len
    audit_prompts = args.audit_prompts

    # --- REF: unmodified bf16 full head ---
    ref = run_pass(mf, harness, paths, server_python=server_python, label="ref",
                   extra_env=build_env(cv=False, audit=False, model_dir=args.model_dir, audit_out=None),
                   num_prompts=num_prompts, output_len=output_len, port=args.port,
                   request_timeout_s=args.request_timeout_s)
    print(f"[cv] REF warm_median_tps={ref['tps']['warm_median_tps']:.2f} "
          f"peak={ref['peak_vram_gb']:.2f}GB ({time.time()-t_start:.0f}s elapsed)", flush=True)

    # --- CVSP: CV head, speed mode ---
    cvsp = run_pass(mf, harness, paths, server_python=server_python, label="cvsp",
                    extra_env=build_env(cv=True, audit=False, model_dir=args.model_dir, audit_out=None),
                    num_prompts=num_prompts, output_len=output_len, port=args.port,
                    request_timeout_s=args.request_timeout_s)
    cvsp["plumbing"] = grep_log(cvsp["log_path"], [
        "[cv-head] finder registered", "[cv-head] REPLACED Gemma4ForCausalLM.compute_logits",
        "[cv-head] built int4 nominator"])
    print(f"[cv] CVSP warm_median_tps={cvsp['tps']['warm_median_tps']:.2f} "
          f"peak={cvsp['peak_vram_gb']:.2f}GB plumbing={cvsp['plumbing']} "
          f"({time.time()-t_start:.0f}s elapsed)", flush=True)

    ident = compare_identity(ref["per_request"], cvsp["per_request"])
    print(f"[cv] IDENTITY seq_exact={ident['n_sequences_byte_exact']}/{ident['n_sequences']} "
          f"token_rate={ident['token_identity_rate']}", flush=True)

    # --- CVAUD: CV head, audit mode (per-step identity), SHORT ---
    audit = None
    if not args.no_audit:
        audit_out = str(OUT_ROOT / "cv_audit_detail.json")
        cvaud = run_pass(mf, harness, paths, server_python=server_python, label="cvaud",
                         extra_env=build_env(cv=True, audit=True, model_dir=args.model_dir, audit_out=audit_out),
                         num_prompts=audit_prompts, output_len=output_len, port=args.port,
                         request_timeout_s=args.request_timeout_s)
        if Path(audit_out).exists():
            audit = json.loads(Path(audit_out).read_text())
            print(f"[cv] CVAUD per-step identity={audit.get('identity_rate')} "
                  f"rows={audit.get('audit_total')} mismatch={audit.get('audit_mismatch')} "
                  f"ties={audit.get('tie_positions')}", flush=True)
        else:
            print("[cv] CVAUD: no audit detail emitted (check server log)", flush=True)

    synthesis = synthesize(ref, cvsp, ident, audit)
    report = {
        "pr": 566, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "submission": str(SUBMISSION), "model_dir": args.model_dir,
        "num_prompts": num_prompts, "audit_prompts": audit_prompts, "output_len": output_len,
        "k_safe": KSAFE, "group": GROUP,
        "ref": {k: v for k, v in ref.items() if k != "per_request"},
        "cvsp": {k: v for k, v in cvsp.items() if k != "per_request"},
        "identity": ident,
        "audit": audit,
        "synthesis": synthesis,
        "elapsed_s": time.time() - t_start,
    }
    out_file = OUT_ROOT / ("served_cv_smoke.json" if args.smoke else "served_cv_report.json")
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[cv] report -> {out_file} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    rid = None
    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[cv] wandb run id={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
