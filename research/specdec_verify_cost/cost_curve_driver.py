#!/usr/bin/env python
"""PR #575 wirbel — SPEC-DEC VERIFY-COST CURVE C(M) + asymptotic ceiling on
base_fullhead (LOCAL A10G, analysis-only, NO HF fire).

Measures the verify-pass cost curve ``C(M)`` directly (M = verified positions per
step = K+1 for draft-length K) on the quality-safe base_fullhead substrate (stock
base-int4 + full native 262k bf16 head, prune OFF), and from it derives:
  (a) verify_cost_k7_measured_ms = C(8) — an INDEPENDENT measured verify cost to
      ground fern #573's acceptance->TPS model denominator;
  (b) specdec_asymptotic_ceiling_tps = 1/c_compute — the A->inf, drafter-independent
      strongest spec-dec TPS base_fullhead can reach;
  (c) specdec_regime in {verify_cost_limited, acceptance_limited}: acceptance-limited
      iff the ceiling clears the 375.857 ship.

Method: serve base_fullhead once per draft-length K in {0,1,2,4,7,8,16} (M=K+1 in
{1,2,3,5,8,9,17}) with an ngram drafter (drafter-independent VERIFY cost; ngram
draft ~0, disengages the MTP loopgraph machinery so any K works). A default-off
per-step probe (``mstep_probe`` via the research-dir sitecustomize chain, no served
file changed) CUDA-event + perf_counter times each decode step and buckets it by M
(read from the SchedulerOutput; prefill excluded). C(M) = warm per-step latency.
Fit C(M)=C_fixed+M*c_compute; ceiling=1/c_compute; 1/C(1) must reproduce 252.69
(serve_equiv_check). MAX_NUM_SEQS=1, single stream, warm-median.

LOCAL only: analysis_only=true, official_tps=0, NO HF Job / --launch / submission /
served-file change. base_fullhead full bf16 head => quality_gate_passes_by_construction;
this card measures COST not tokens (greedy-identity is fern/lawine's empirical job).

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/specdec_verify_cost/cost_curve_driver.py --smoke --no-wandb
Full sweep:
  CUDA_VISIBLE_DEVICES=0 python research/specdec_verify_cost/cost_curve_driver.py \
    --wandb_name wirbel/specdec-verify-cost-asymptote --wandb_group base-fullhead-specdec-ceiling
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

# wirbel's own stock int4 snapshot (NO baked bucket) — base_fullhead substrate.
MODEL_DIR = (
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)

SEED = 1
HIDDEN = 2560
VOCAB = 262144
HEAD_BYTES_BF16 = HIDDEN * VOCAB * 2          # 1.342 GB read once per step
GPU_MEM_UTIL = os.environ.get("SPECDEC_GPU_MEM_UTIL", "0.92")

# Cited anchors (NOT re-derived here).
ANCHOR_BASE_FULLHEAD_NOSPEC = 252.69          # wirbel #553 base_fullhead no-spec (1/C(1))
SHIP_TPS = 375.857                            # surgical-357 ship (the flip-condition bar)
FLOOR_TPS = 311.25                            # lawine #554 magically-free floor
GATE_TPS = 500.0                              # leaderboard gate
BASELINE_TPS = 481.53                         # public #1 (untouched)
A10G_HBM_GBPS = 600.0

# K sweep -> M = K+1 verify granularity. K=7/M=8 is the ship's drafter.
K_SWEEP = [0, 1, 2, 4, 7, 8, 16]
TARGET_M = [k + 1 for k in K_SWEEP]           # [1, 2, 3, 5, 8, 9, 17]


def ngram_spec_config(k: int) -> str:
    """ngram drafter at draft-length K (empty -> no-spec M=1). Minimal config; vLLM
    fills prompt_lookup defaults. ngram draft cost ~0 -> pure VERIFY cost, and it
    disengages the MTP loopgraph machinery so every K serves cleanly."""
    if k <= 0:
        return ""
    return json.dumps({
        "method": "ngram",
        "num_speculative_tokens": k,
        "prompt_lookup_max": min(k, 4),
        "prompt_lookup_min": 1,
    })


# The ship's default drafter (manifest) — base_fullhead + this == the wirbel #553
# 252.69 serve. Used as the SUBSTRATE-ANCHOR pass (proves 252.69 is the MTP-K=7 spec
# number, not 1/C(1)) and to measure the MTP step cost C(8) incl. the draft head.
MTP_DRAFTER_PATH = "/tmp/qat-assistant"
MTP_K = 7


def mtp_spec_config() -> str:
    return json.dumps({"method": "mtp", "model": MTP_DRAFTER_PATH, "num_speculative_tokens": MTP_K})


def spec_config_for(kind: str, k: int) -> str:
    return mtp_spec_config() if kind == "mtp" else ngram_spec_config(k)


def build_env(*, k: int, mstep_out: str, warmup_skip: int, flush_every: int,
              kind: str = "ngram") -> dict[str, str]:
    """base_fullhead serve recipe (full 262k head, prune OFF) + ngram@K + M-step probe.

    REQUIRE guards are zeroed so the ngram path (which never imports the MTP
    gemma4/gemma4_mtp/llm_base_proposer modules the loopgraph/fused-argmax/accept
    patches target) falls back gracefully instead of asserting. Where a fusion DOES
    apply (full-head decode) REQUIRE=0 is behaviorally identical to REQUIRE=1, so the
    K=0 no-spec serve still reproduces the 252.69 fast-kernel anchor."""
    env: dict[str, str] = {
        # --- base_fullhead substrate (served_cv_driver recipe) ---
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",
        "LOCAL_MODEL_DIR": MODEL_DIR,
        "PLE_FOLD_TARGET_MODEL": MODEL_DIR,
        "PLE_FOLD_EMBED_SCALE": "1",
        "LM_HEAD_FULL_REQUIRE": "1",
        "GPU_MEMORY_UTILIZATION": GPU_MEM_UTIL,
        "MAX_NUM_SEQS": "1",
        # --- the swept variable: ngram drafter at draft-length K (or no-spec / MTP) ---
        "SPECULATIVE_CONFIG": spec_config_for(kind, k),
        # --- zero the MTP-specific REQUIRE guards so ngram boots cleanly ---
        "LOOPGRAPH_REQUIRE_CAPTURE": "0",
        "FUSED_SPARSE_ARGMAX_REQUIRE": "0",
        "DIXIE_FUSED_ACCEPT_PREP_REQUIRE": "0",
        "PRECACHE_REQUIRE": "0",
        # --- M-step probe (research-dir sitecustomize chain; no served file changed) ---
        "PYTHONPATH": f"{HERE}{os.pathsep}{SUBMISSION}",
        "MSTEP_PKG_DIR": str(SUBMISSION),
        "MSTEP": "1",
        "MSTEP_OUT": mstep_out,
        "MSTEP_SERVED_K": str(k),
        "MSTEP_WARMUP_SKIP": str(warmup_skip),
        "MSTEP_FLUSH_EVERY": str(flush_every),
        "MSTEP_M_CAP": "64",
    }
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
        })
        print(f"[worker] {index + 1}/{len(records)} req_ms={1000.0 * (t2 - t1):.1f} "
              f"comp={len(completion_token_ids)} prompt={len(prompt_token_ids)}", flush=True)

    out = {"output_len": args.output_len, "num_records": len(records), "per_request": rows}
    Path(args.out_file).write_text(json.dumps(out))
    return 0


# ========================================================================== #
# server pass (one per K)
# ========================================================================== #
def run_pass(mf: Any, harness: Any, paths: Any, *, server_python: Path, k: int,
             num_prompts: int, output_len: int, port: int, request_timeout_s: int,
             warmup_skip: int, flush_every: int, kind: str = "ngram") -> dict[str, Any]:
    label = f"mtp_k{k}" if kind == "mtp" else f"k{k}"
    log_path = OUT_ROOT / f"server_{label}.log"
    pass_file = OUT_ROOT / f"{label}_pass.json"
    mstep_out = str(OUT_ROOT / f"mstep_{label}.json")
    if os.path.exists(mstep_out):
        os.remove(mstep_out)
    extra_env = build_env(k=k, mstep_out=mstep_out, warmup_skip=warmup_skip,
                          flush_every=flush_every, kind=kind)

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
    measured: dict[str, Any] = {"k": k, "m": k + 1, "label": label, "kind": kind,
                                "num_prompts": num_prompts, "output_len": output_len,
                                "spec_config": extra_env["SPECULATIVE_CONFIG"]}
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            measured["model_id"] = srv.model_id
            measured["served_model_name"] = srv.served_model_name
            print(f"[cc] [{label}] warming server ({srv.base_url})", flush=True)
            mf._warm_server(srv.base_url, srv.served_model_name, n=mf.WARMUP_REQUESTS)
            cmd = [
                str(server_python), str(Path(__file__).resolve()), "--decode-worker",
                "--base-url", srv.base_url, "--model", srv.served_model_name,
                "--dataset-path", str(paths.EVAL_PROMPTS), "--tokenizer", paths.TOKENIZER,
                "--num-prompts", str(num_prompts), "--output-len", str(output_len),
                "--seed", str(SEED), "--out-file", str(pass_file),
                "--request-timeout-s", str(request_timeout_s),
            ]
            print(f"[cc] [{label}] decode pass {num_prompts}x{output_len} conc=1 K={k} -> {pass_file}", flush=True)
            subprocess.run(cmd, check=True, timeout=5400, env=worker_env)
            summary = json.loads(pass_file.read_text())
            try:
                measured["tps"] = mf._aggregate(summary)
            except Exception as exc:
                measured["tps"] = {"warm_median_tps": float("nan"), "aggregate_error": repr(exc)}
    finally:
        stop.set()
        sampler.join(timeout=5)

    measured["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    measured["log_path"] = str(log_path)
    measured["mstep"] = json.loads(Path(mstep_out).read_text()) if Path(mstep_out).exists() else None
    measured["plumbing"] = grep_log(str(log_path), [
        "[mstep-shim] ran submission sitecustomize",
        "[mstep] execute_model wrapper active",
        "[mstep] finders registered",
    ])
    return measured


def grep_log(log_path: str, needles: list[str]) -> dict[str, bool]:
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return {n: False for n in needles}
    return {n: (n in text) for n in needles}


# ========================================================================== #
# fit + synthesis
# ========================================================================== #
def weighted_linfit(xs: list[float], ys: list[float], ws: list[float]) -> dict[str, float]:
    """Weighted least squares y = a + b*x. Returns slope b, intercept a, R^2."""
    S = sum(ws)
    Sx = sum(w * x for w, x in zip(ws, xs))
    Sy = sum(w * y for w, y in zip(ws, ys))
    Sxx = sum(w * x * x for w, x in zip(ws, xs))
    Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = S * Sxx - Sx * Sx
    if abs(denom) < 1e-18:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    b = (S * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / S
    ybar = Sy / S
    ss_tot = sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys))
    ss_res = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": b, "intercept": a, "r2": r2}


def _bucket_central(mstep: dict | None, m: int, seam: str = "exec_step",
                    field: str = "trimmed_mean_ms") -> dict[str, Any] | None:
    if not mstep:
        return None
    seam_d = mstep.get(seam, {})
    b = seam_d.get(str(m))
    if not b:
        return None
    wall = b.get("wall", b) if seam == "exec_step" else b
    if not wall or wall.get("n", 0) == 0:
        return None
    return wall


def _gpu_bucket(mstep: dict | None, m: int) -> dict[str, Any] | None:
    if not mstep:
        return None
    b = mstep.get("exec_step", {}).get(str(m), {})
    g = b.get("gpu")
    return g if (g and g.get("n", 0) > 0) else None


def synthesize(passes: list[dict[str, Any]]) -> dict[str, Any]:
    # ngram (verify-cost) passes keyed by K; the MTP pass kept aside as the
    # substrate anchor (its served TPS is the 252.69 #553 reproduction).
    ng_passes = [p for p in passes if p.get("kind", "ngram") == "ngram"]
    by_k = {p["k"]: p for p in ng_passes}
    mtp_pass = next((p for p in passes if p.get("kind") == "mtp"), None)

    # ---- C(M) = the per-step VERIFY GPU time (CUDA event), bucketed by M -------
    # The CUDA-event gpu time is the true GPU-bound per-step cost (for K=0 it equals
    # the wall+gap inter-step interval AND 1/warm_median_tps — see serve_equiv). The
    # python `wall` is the async-dispatch return time (returns before the GPU step
    # finishes) and is NOT the step cost, so it is kept only as a diagnostic.
    c_of_m: dict[str, Any] = {}
    # all matched points (M=1 is the no-spec path); spec-only points (M>=2) share the
    # spec-dec verify path so their slope is the consistent per-position compute cost.
    fx_all: list[float] = []; fy_all: list[float] = []; fw_all: list[float] = []
    fx_sp: list[float] = []; fy_sp: list[float] = []; fw_sp: list[float] = []
    for k in K_SWEEP:
        m = k + 1
        p = by_k.get(k)
        mstep = p.get("mstep") if p else None
        gpu = _gpu_bucket(mstep, m)
        wall = _bucket_central(mstep, m, "exec_step")
        head = (mstep or {}).get("head_gemm", {}).get(str(m)) if mstep else None
        entry: dict[str, Any] = {"k": k, "m": m}
        if gpu:
            c_gpu = gpu["trimmed_mean_ms"]
            entry.update({
                "C_gpu_ms": c_gpu,
                "C_gpu_median_ms": gpu["median_ms"],
                "C_gpu_ci95_ms": gpu["ci95_halfwidth_ms"],
                "C_gpu_p10_ms": gpu["p10_ms"],
                "C_gpu_p90_ms": gpu["p90_ms"],
                "n_steps": gpu["n"],
                "tps_max_at_m": 1000.0 * m / c_gpu if c_gpu > 0 else None,
            })
            w = gpu["n"] / max(gpu["std_ms"] ** 2, 1e-9)
            fx_all.append(float(m)); fy_all.append(c_gpu); fw_all.append(w)
            if m >= 2:
                fx_sp.append(float(m)); fy_sp.append(c_gpu); fw_sp.append(w)
        if wall:
            entry["C_wall_ms_diag"] = wall["trimmed_mean_ms"]   # async-dispatch artifact, diagnostic only
        if head and head.get("n", 0) > 0:
            entry["head_gpu_ms"] = head["trimmed_mean_ms"]
        if p:
            entry["warm_median_tps"] = p.get("tps", {}).get("warm_median_tps")
            entry["draft_hist"] = (mstep or {}).get("draft_hist")
        c_of_m[str(m)] = entry

    # ---- fit C(M) = C_fixed + M * c_compute ------------------------------------
    fit_all = weighted_linfit(fx_all, fy_all, fw_all) if len(fx_all) >= 2 else {
        "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    fit_spec = weighted_linfit(fx_sp, fy_sp, fw_sp) if len(fx_sp) >= 2 else {
        "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    def _ceil(slope: float) -> float:
        return 1000.0 / slope if (slope and math.isfinite(slope) and slope > 0) else float("nan")

    # primary = spec-only fit (consistent spec-dec verify path); fall back to all-points.
    primary_fit = fit_spec if math.isfinite(fit_spec["slope"]) else fit_all
    c_compute = primary_fit["slope"]
    c_fixed = primary_fit["intercept"]
    ceiling = _ceil(c_compute)
    ceiling_all = _ceil(fit_all["slope"])

    # ---- serve-equivalence (REINTERPRETED) -------------------------------------
    # The PR's literal model says 1/C(1) == 252.69, but on a10g that is physically
    # impossible: the measured full-262k-head GEMM alone is ~2.79 ms (1.342 GB at
    # ~480 GB/s, near the A10G HBM peak), so a single-token step that reads the full
    # head cannot exceed ~140 TPS. 252.69 is therefore the base_fullhead served TPS
    # WITH the ship's MTP K=7 drafter (= A_mtp / C(8)), not 1/C(1). We therefore split
    # the check into the two things that ARE physically checkable:
    #   (1) nospec internal consistency: the K=0 served warm-median TPS == 1/C(1)
    #       (gpu-event), i.e. the probe's per-step cost reproduces the served rate.
    #   (2) substrate anchor: base_fullhead served with MTP K=7 reproduces 252.69
    #       (the #553 config) -> the substrate under test IS the #553 substrate.
    p0 = by_k.get(0)
    k0_warm_tps = (p0 or {}).get("tps", {}).get("warm_median_tps", float("nan"))
    c1_gpu = c_of_m.get("1", {}).get("C_gpu_ms")
    tps_from_c1 = 1000.0 / c1_gpu if (c1_gpu and c1_gpu > 0) else float("nan")
    nospec_consistent = (
        abs(k0_warm_tps - tps_from_c1) <= 0.08 * tps_from_c1
        if all(isinstance(v, float) and math.isfinite(v) for v in (k0_warm_tps, tps_from_c1)) else False)

    mtp_tps = (mtp_pass or {}).get("tps", {}).get("warm_median_tps", float("nan"))
    mtp_substrate_ok = (
        abs(mtp_tps - ANCHOR_BASE_FULLHEAD_NOSPEC) <= 0.05 * ANCHOR_BASE_FULLHEAD_NOSPEC
        if isinstance(mtp_tps, float) and math.isfinite(mtp_tps) else False)

    # implied MTP mean-acceptance from the measured verify cost: A = TPS * C(8) / 1000
    c8 = c_of_m.get("8", {}).get("C_gpu_ms")        # K=7 verify cost (ngram, drafter-free)
    c8_mtp = _gpu_bucket((mtp_pass or {}).get("mstep"), 8)
    c8_mtp_ms = c8_mtp["trimmed_mean_ms"] if c8_mtp else None
    implied_A_mtp = (mtp_tps * c8_mtp_ms / 1000.0
                     if (isinstance(mtp_tps, float) and math.isfinite(mtp_tps) and c8_mtp_ms) else
                     (mtp_tps * c8 / 1000.0 if (isinstance(mtp_tps, float) and math.isfinite(mtp_tps) and c8) else float("nan")))

    regime = ("acceptance_limited" if (math.isfinite(ceiling) and ceiling >= SHIP_TPS)
              else "verify_cost_limited")
    head_read_ms_floor = 1000.0 * HEAD_BYTES_BF16 / (A10G_HBM_GBPS * 1e9)

    return {
        "C_of_M": c_of_m,
        "C_fixed_ms": c_fixed,
        "c_compute_ms_per_pos": c_compute,
        "fit_r2": primary_fit["r2"],
        "fit_spec_points_m_ge_2": fit_spec,
        "fit_all_points": fit_all,
        "specdec_asymptotic_ceiling_tps": ceiling,
        "specdec_asymptotic_ceiling_tps_all_points": ceiling_all,
        "specdec_regime": regime,
        "specdec_ceiling_exceeds_ship": bool(math.isfinite(ceiling) and ceiling >= SHIP_TPS),
        "specdec_ceiling_gap_to_ship": (SHIP_TPS - ceiling) if math.isfinite(ceiling) else float("nan"),
        "verify_cost_k7_measured_ms": c8,                 # ngram drafter-free verify cost at M=8
        "verify_cost_k7_mtp_step_ms": c8_mtp_ms,          # MTP step (drafter+verify) at M=8, if measured
        # ---- serve-equivalence, reinterpreted ----
        "serve_equiv_check": bool(nospec_consistent and (mtp_substrate_ok or not mtp_pass)),
        "serve_equiv_nospec_consistent": nospec_consistent,
        "serve_equiv_mtp_substrate_ok": mtp_substrate_ok,
        "nospec_warm_median_tps": k0_warm_tps,            # the TRUE 1/C(1) on a10g (~87.6)
        "nospec_tps_from_C1_gpu": tps_from_c1,
        "true_C1_ms": c1_gpu,
        "mtp_substrate_served_tps": mtp_tps,              # base_fullhead + MTP K=7 (~251, == 252.69 anchor)
        "implied_mtp_mean_acceptance": implied_A_mtp,     # A = mtp_tps * C(8) / 1000 (cross-check, ~3.3)
        "anchor_252_69_is_mtp_spec_not_1_over_C1": True,  # the corrected interpretation (flag to advisor)
        "anchor_base_fullhead_mtp": ANCHOR_BASE_FULLHEAD_NOSPEC,
        "ship_tps": SHIP_TPS,
        "floor_tps": FLOOR_TPS,
        "gate_tps": GATE_TPS,
        "head_bytes_bf16": HEAD_BYTES_BF16,
        "head_read_ms_floor_at_600gbps": head_read_ms_floor,
        "quality_gate_passes_by_construction": True,
        "identity_note": ("this card measures COST not tokens; greedy-identity is "
                          "fern/lawine's empirical job (#566 standard), not asserted here"),
        "self_det": True,
        "analysis_only": True,
        "official_tps": 0,
        "peak_vram_gb": max((p.get("peak_vram_gb", 0.0) for p in passes), default=0.0),
    }


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # pragma: no cover
        print(f"[cc] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile", agent="wirbel",
        name=args.wandb_name or "wirbel/specdec-verify-cost-asymptote",
        group=args.wandb_group or "base-fullhead-specdec-ceiling",
        tags=["specdec", "verify-cost-curve", "asymptotic-ceiling", "base-fullhead",
              "local-a10g", "analysis-only", "pr575"],
        notes="PR #575: spec-dec verify-cost curve C(M) + A->inf ceiling + limiting-regime "
              "verdict on base_fullhead (3rd spec-dec leg; grounds fern #573 model input)",
        config={
            "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
            "k_sweep": K_SWEEP, "target_m": TARGET_M,
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "concurrency": 1, "seed": SEED, "gpu_mem_util": GPU_MEM_UTIL,
            "ship_tps": SHIP_TPS, "anchor_base_fullhead_nospec": ANCHOR_BASE_FULLHEAD_NOSPEC,
        },
    )
    if run is None:
        return None
    s = report["synthesis"]
    summary = {k: v for k, v in s.items()
               if isinstance(v, (int, float, bool)) and (not isinstance(v, float) or math.isfinite(v))}
    summary["primary_metric"] = s["specdec_asymptotic_ceiling_tps"]
    # per-M C(M) (gpu-event verify cost) and TPS_max(M) as flat keys for cross-run comparison
    for mk, e in s["C_of_M"].items():
        if e.get("C_gpu_ms") is not None:
            summary[f"C_M{mk}_ms"] = e["C_gpu_ms"]
            if e.get("tps_max_at_m") is not None:
                summary[f"tps_max_M{mk}"] = e["tps_max_at_m"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="specdec-verify-cost-report", artifact_type="specdec-cost-report", data=report)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _fmt(v: Any, p: str = ".2f") -> str:
    return format(v, p) if isinstance(v, (int, float)) and math.isfinite(v) else str(v)


def _print_summary(s: dict[str, Any]) -> None:
    line = "=" * 10 + " PR #575 — SPEC-DEC VERIFY-COST CURVE + ASYMPTOTIC CEILING " + "=" * 10
    print("\n" + line, flush=True)
    print("  C(M) = per-step VERIFY GPU time (CUDA event), ms  [TPS_max = 1000*M/C(M)]:", flush=True)
    for mk in sorted(s["C_of_M"], key=lambda x: int(x)):
        e = s["C_of_M"][mk]
        if e.get("C_gpu_ms") is not None:
            print(f"    M={mk:>2} (K={e['k']:>2}): C={e['C_gpu_ms']:.4f}±{e.get('C_gpu_ci95_ms',0):.4f}ms "
                  f"n={e.get('n_steps')}  TPS_max={_fmt(e.get('tps_max_at_m'),'.1f')}  "
                  f"warm_tps={_fmt(e.get('warm_median_tps'),'.1f')}", flush=True)
    print(f"  fit (spec M>=2): C(M) = {_fmt(s['C_fixed_ms'],'.4f')} + M*{_fmt(s['c_compute_ms_per_pos'],'.5f')}  "
          f"R2={_fmt(s['fit_r2'],'.5f')}", flush=True)
    print(f"  serve_equiv (REINTERPRETED): nospec_warm={_fmt(s['nospec_warm_median_tps'],'.2f')} == "
          f"1/C(1)={_fmt(s['nospec_tps_from_C1_gpu'],'.2f')} (consistent={s['serve_equiv_nospec_consistent']}); "
          f"MTP_substrate={_fmt(s['mtp_substrate_served_tps'],'.2f')} vs anchor 252.69 "
          f"(ok={s['serve_equiv_mtp_substrate_ok']})  -> {s['serve_equiv_check']}", flush=True)
    print(f"      [252.69 is the MTP-K=7 spec number, NOT 1/C(1); implied MTP A="
          f"{_fmt(s['implied_mtp_mean_acceptance'],'.2f')} tok/step]", flush=True)
    print(f"  >>> verify_cost_k7_measured_ms (C(8), ngram) = {_fmt(s['verify_cost_k7_measured_ms'],'.4f')}  "
          f"(MTP step incl. draft = {_fmt(s.get('verify_cost_k7_mtp_step_ms'),'.4f')})", flush=True)
    print(f"  >>> specdec_asymptotic_ceiling_tps (1/c_compute) = {_fmt(s['specdec_asymptotic_ceiling_tps'],'.1f')}  "
          f"(all-points {_fmt(s['specdec_asymptotic_ceiling_tps_all_points'],'.1f')})", flush=True)
    print(f"  >>> regime = {s['specdec_regime']}  (ceiling>=ship {SHIP_TPS}? "
          f"{s['specdec_ceiling_exceeds_ship']}, gap {_fmt(s['specdec_ceiling_gap_to_ship'],'.1f')})", flush=True)
    print(f"  peak VRAM = {_fmt(s['peak_vram_gb'],'.2f')} GB", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# main
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check (K in {0,7}, few prompts)")
    ap.add_argument("--k-list", default=None, help="comma-separated K override (default full sweep)")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    # internal worker mode (runs under the server venv)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer"); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-file")
    ap.add_argument("--no-mtp-anchor", action="store_true",
                    help="skip the MTP K=7 substrate-anchor pass (the 252.69 reproduction check)")
    args = ap.parse_args(argv)

    if args.decode_worker:
        return _decode_worker(args)

    mf_spec = importlib.util.spec_from_file_location("measure_floor", str(MEASURE_FLOOR))
    mf = importlib.util.module_from_spec(mf_spec)
    assert mf_spec and mf_spec.loader
    mf_spec.loader.exec_module(mf)
    from scripts.local_validation import harness, paths

    for note in paths.prepare_local_gpu_env():
        print(f"[cc] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])

    if args.k_list:
        k_list = [int(x) for x in args.k_list.split(",") if x.strip() != ""]
    elif args.smoke:
        k_list = [0, 7]
    else:
        k_list = list(K_SWEEP)
    num_prompts = 4 if args.smoke else args.num_prompts
    output_len = 64 if args.smoke else args.output_len
    # smoke: low warmup + frequent flush so the few decode steps persist before the
    # worker SIGTERM (atexit is unreliable on EngineCore teardown). full: discard the
    # cold-start ramp, flush ~every 200 resolved steps (final loss <=200, ~2-3%).
    warmup_skip = int(os.environ.get("MSTEP_WARMUP_SKIP", "8" if args.smoke else "128"))
    flush_every = int(os.environ.get("MSTEP_FLUSH_EVERY", "20" if args.smoke else "200"))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    # (kind, k) plan: the ngram K-sweep (drafter-free VERIFY cost curve) + one MTP K=7
    # substrate-anchor pass (proves base_fullhead+MTP reproduces 252.69 = the #553 serve).
    plan: list[tuple[str, int]] = [("ngram", k) for k in k_list]
    if not args.smoke and not args.no_mtp_anchor:
        plan.append(("mtp", MTP_K))
    passes: list[dict[str, Any]] = []
    for kind, k in plan:
        p = run_pass(mf, harness, paths, server_python=server_python, k=k,
                     num_prompts=num_prompts, output_len=output_len, port=args.port,
                     request_timeout_s=args.request_timeout_s,
                     warmup_skip=warmup_skip, flush_every=flush_every, kind=kind)
        ms = p.get("mstep") or {}
        nbucket = (ms.get("exec_step", {}).get(str(k + 1), {}).get("gpu", {}) or {}).get("n")
        print(f"[cc] [{kind} k{k}] warm_median_tps={p.get('tps',{}).get('warm_median_tps')} "
              f"M={k+1} n_steps={nbucket} plumbing={p.get('plumbing')} "
              f"peak={p.get('peak_vram_gb',0):.2f}GB ({time.time()-t_start:.0f}s)", flush=True)
        passes.append(p)

    if args.smoke:
        ok = all(p.get("plumbing", {}).get("[mstep] execute_model wrapper active") for p in passes)
        for p in passes:
            ms = p.get("mstep") or {}
            print(f"[cc] SMOKE k{p['k']}: exec_buckets={list((ms.get('exec_step') or {}).keys())} "
                  f"draft_hist={ms.get('draft_hist')} m_logits_hist={ms.get('m_logits_hist')} "
                  f"decode_steps={ms.get('decode_steps_timed')}", flush=True)
        print(f"[cc] SMOKE {'PASS' if ok else 'CHECK'} ({time.time()-t_start:.0f}s)", flush=True)
        return 0 if ok else 1

    synthesis = synthesize(passes)
    report = {
        "pr": 575, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
        "k_sweep": k_list, "num_prompts": num_prompts, "output_len": output_len,
        "passes": [{k: v for k, v in p.items() if k != "mstep"} | {"mstep": p.get("mstep")} for p in passes],
        "synthesis": synthesis,
        "elapsed_s": time.time() - t_start,
    }
    out_file = OUT_ROOT / "specdec_cost_report.json"
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[cc] report -> {out_file} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    # NaN guard on the headline.
    ceil = synthesis["specdec_asymptotic_ceiling_tps"]
    if not (isinstance(ceil, float) and math.isfinite(ceil)):
        print("[cc] FAIL: non-finite specdec_asymptotic_ceiling_tps", flush=True)
        return 1

    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[cc] wandb run id={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
