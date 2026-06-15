#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #359 — strict-step-shave-stack: GPU-backed per-step micro-lever measurement.

The strict >500 program needs the *measured* (not analytic) per-decode-step cost
of the byte-strict, non-speculative int4 M=1 AR base, and how much of that step an
identity-preserving micro-lever stack can actually shave on real A10G silicon.

This is **local inference profiling on the pod GPU** — NOT an HF Job, NOT a
submission. It serves the strict base (``submissions/fa2sw_nonspec_int4``, lawine
#196: byte-identical to the deployed ``fa2sw_precache_kenyan`` EXCEPT
``SPECULATIVE_CONFIG`` blanked, so every decode step is a plain int4 M=1 AR
target forward — official ~165.44 TPS) and measures, one lever at a time:

  * (c) attention-backend swap   — the ONE custom lever that is *active* on the
        non-spec base. ``FA_SLIDING`` swaps ~16 gemma4 sliding-window TARGET
        layers onto FlashAttention; ``VLLM_ATTENTION_BACKEND`` forces a single
        backend across all layers (TRITON_ATTN / FLASH_ATTN). Toggled via the
        ``LocalServer(extra_env=...)`` override — no submission edit.
  * (a) custom CUDA-graph  (``ONEGRAPH``)            — patches the spec-decode
  * (b) custom kernel-fusion(``FUSED_SPARSE_ARGMAX``)  proposer / drafter sparse
        argmax. Both are measured here to PROVE they are inert on the non-spec
        base (the drafter never runs when ``SPECULATIVE_CONFIG`` is blank), which
        is itself a load-bearing finding: the spec-decode step-shave levers do
        NOT transfer to the strict base.

vLLM's *native* decode-step cudagraph and compilation fusion are ON in the base
(no ``--enforce-eager``) but are not env-isolatable through the submission's
serve.py, so their standalone delta is out of scope here (reported, not measured;
a follow-up would need a profiling-only serve flag).

Method (per config, one server session each, levers consumed at engine init):
  per-step µs via two-length request-time differencing — mean warm per-request
  latency at L_long minus L_short, divided by (L_long - L_short). Prefill / TTFT /
  client tokenization / IO are per-request constants independent of output length,
  so the difference isolates the steady-state decode STEP. For M=1 AR one output
  token == one decode step, so this is the per-forward-step time directly.
  Greedy-token-identity is verified by per-prompt ``completion_token_sha256`` vs
  the base config (#319 HARD gate): any lever that changes an emitted token is
  reported and EXCLUDED from the compliant stack.

Local TPS is a RELATIVE screen (local != official a10g scale); the per-step µs
*deltas* and the identity verdicts are what transfer. The implied official lift
applies the measured local µs-delta to the official per-step time anchored at the
165.44 strict-base TPS, and reports the fraction of the 500-TPS gap it closes.

Changes NOTHING served: no submission edit, no HF Job, no submission. 0 official
TPS. This file is the deliverable.

Reproduce (full matrix, local A10G):
  CUDA_VISIBLE_DEVICES=0 python research/validity/strict_step_shave_stack/strict_step_shave_stack.py \
    --measure --wandb_group strict-step-shave-359 \
    --wandb_name kanna/strict-step-shave-stack
Add ``--self-test`` to also assert ``strict_step_shave_stack_self_test_passes``.
Use ``--smoke`` for a tiny base-only plumbing run (no wandb).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
# This script is re-invoked as the --decode-worker under the SERVER venv. Python
# auto-prepends the script's own directory (and a bare cwd entry) to sys.path,
# which would shadow the `scripts.local_validation` import the worker needs.
# Scrub them and re-add ROOT explicitly (mirrors local_official_tps_transfer
# profile.py). PYTHONSAFEPATH=1 in the worker env is a second guard.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Imported anchors — DO NOT re-derive (self-test constant guard checks these).
# --------------------------------------------------------------------------- #
# Official launch gate.
TARGET_TPS = 500.0
# Strict non-spec base, lawine #196: byte-strict int4 M=1 AR, official a10g TPS.
STRICT_BASE_OFFICIAL_TPS_196 = 165.44
# Deployed spec-decode anchor (fa2sw_precache_kenyan) — context only, NOT the base.
DEPLOYED_SPEC_OFFICIAL_TPS = 481.53
# Local->official transfer factor (lawine #267, spec-path calibrated). Used to
# sanity-map local readings; the µs DELTAS are the transferable quantity.
TAU_LO_267 = 1.0352356533046398
LOCAL_BAR_FOR_500_267 = 482.9818200367414
# Analytic prior (this PR's preserved card, strict_step_shave_stack_analytic.py):
# predicted identity-preserving step-shave fraction, realizable vs optimistic.
PRED_REALIZABLE_FRAC = 0.01271
PRED_OPTIMISTIC_FRAC = 0.017794

SUBMISSION = ROOT / "submissions" / "fa2sw_nonspec_int4"
OUT_ROOT = ROOT / "research" / "validity" / "strict_step_shave_stack"

# official sglang.bench_serving discards this many warmup requests from timing;
# we discard the same count per measured pass for steady-state.
WARMUP_DISCARD = 4
# Full-matrix measurement defaults (cheap; deltas are stable in warm steady state).
FULL_NUM_PROMPTS = 24
FULL_L_SHORT = 64
FULL_L_LONG = 256
# Smoke (plumbing-only) defaults.
SMOKE_NUM_PROMPTS = 6
SMOKE_L_SHORT = 16
SMOKE_L_LONG = 48


# ========================================================================== #
# Lever config matrix
# ========================================================================== #
# Precache is disabled in every config: the manifest points PRECACHE_DATASET at
# /harness/data/eval_prompts_sharegpt.json which does not exist in this pod
# container, and PRECACHE_REQUIRE=1 would abort startup. Precache is a KV/prefix
# warmup (identity-neutral; does not change emitted tokens), so disabling it does
# not affect the warm steady-state per-step time we measure.
COMMON_OVERRIDES: dict[str, str] = {
    "PRECACHE_BENCH": "0",
    "PRECACHE_REQUIRE": "0",
}


def cfg_env(
    *,
    onegraph: bool = True,
    fa_sliding: bool = True,
    fused_argmax: bool = True,
    enforce_eager: bool = False,
    compilation_config: str | None = None,
    vllm_attention_backend: str | None = None,
) -> dict[str, str]:
    """Build the LocalServer extra_env that toggles one lever off the base.

    The base manifest has ONEGRAPH=1, FA_SLIDING=1, FUSED_SPARSE_ARGMAX=1 with
    their REQUIRE guards on. When a flag is turned OFF we also clear its REQUIRE
    guard so a (legitimately) skipped capture/fusion does not abort startup.

    The advisor's named NATIVE step-shave levers (#359, GPU-pivot comment) are
    vLLM-engine levers, not the spec-decode custom kernels:
      * (a) native decode CUDA-graph capture -> ``enforce_eager`` (also drops
            torch.compile fusion) and ``compilation_config`` (cudagraph_mode only).
      * (b) native kernel fusion (norm/act/residual) -> the inductor/compile part
            of the same toggle; ``enforce_eager`` removes both (a)+(b),
            ``compilation_config={"cudagraph_mode":"NONE"}`` removes ONLY (a) so
            (b) can be isolated as eager-minus-cudagraph_off.
      * (c) attention-backend swap -> FA_SLIDING (FA2-on-sliding vs all-Triton)
            and VLLM_ATTENTION_BACKEND. (The model forces TRITON_ATTN globally for
            its heterogeneous head dims, so an all-FlashAttention swap is expected
            to be overridden -- itself a measured ceiling on (c).)
    The spec-custom ONEGRAPH / FUSED_SPARSE_ARGMAX toggles are kept purely as
    inert-proof controls: the drafter never runs on the non-spec base, so they are
    expected to move the step by ~0 (a load-bearing finding -- the spec stack's
    custom step-shaves do NOT transfer to the strict non-spec greedy path).
    """
    env = dict(COMMON_OVERRIDES)
    # spec-proposer custom CUDA-graph (inert-proof control on non-spec base).
    env["ONEGRAPH"] = "1" if onegraph else "0"
    if not onegraph:
        env["LOOPGRAPH_REQUIRE_CAPTURE"] = "0"
    # spec-drafter custom fused sparse argmax (inert-proof control).
    env["FUSED_SPARSE_ARGMAX"] = "1" if fused_argmax else "0"
    if not fused_argmax:
        env["FUSED_SPARSE_ARGMAX_REQUIRE"] = "0"
    # (a)+(b) native engine cudagraph + compile fusion.
    if enforce_eager:
        env["ENFORCE_EAGER"] = "1"
    if compilation_config:
        env["COMPILATION_CONFIG"] = compilation_config
    # (c) attention-backend levers.
    env["FA_SLIDING"] = "1" if fa_sliding else "0"
    if vllm_attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = vllm_attention_backend
    return env


# Each config = one server bring-up (levers are consumed at engine init). The
# venv + weights + int4 bake + lm_head prune are cached after the first config,
# so only the first bring-up pays the full download/bake cost.
def build_configs() -> dict[str, dict[str, Any]]:
    return {
        # THE strict base + greedy-identity reference. FA2-SW on sliding layers,
        # native cudagraph + native fusion on, drafter off (non-spec).
        "base": {
            "extra_env": cfg_env(),
            "lever": None,
            "desc": "strict non-spec int4 M=1 AR base (manifest defaults)",
        },
        # (a)+(b) NATIVE engine cudagraph capture + torch.compile fusion OFF (eager
        # forward). base-minus-this = the combined identity-safe step-shave the
        # native cudagraph+fusion bank into the 165.44 strict base.
        "eager": {
            "extra_env": cfg_env(enforce_eager=True),
            "lever": "cudagraph_fusion_native",
            "desc": "ENFORCE_EAGER=1 (native decode cudagraph + compile fusion OFF)",
        },
        # (a) NATIVE cudagraph capture OFF but inductor fusion ON. base-minus-this =
        # the cudagraph-only worth; eager-minus-this = the fusion-only worth.
        "cudagraph_off": {
            "extra_env": cfg_env(compilation_config='{"cudagraph_mode": "NONE"}'),
            "lever": "cudagraph_native",
            "desc": 'COMPILATION_CONFIG cudagraph_mode=NONE (cudagraph OFF, fusion ON)',
        },
        # (c) attention: disable the FA2-SW per-layer swap -> sliding layers fall
        # back to vLLM's default (TRITON) backend. Isolates the FA_SLIDING lever µs.
        "fa_sliding_off": {
            "extra_env": cfg_env(fa_sliding=False),
            "lever": "attention",
            "desc": "FA_SLIDING=0 (sliding layers -> vLLM default backend)",
        },
        # (c) attention ceiling: try to force ALL layers onto FlashAttention. The
        # model forces TRITON_ATTN for heterogeneous head dims, so this is expected
        # to be overridden (measured ceiling on the attention-backend swap).
        "attn_flash_attn_all": {
            "extra_env": cfg_env(fa_sliding=False, vllm_attention_backend="FLASH_ATTN"),
            "lever": "attention",
            "desc": "VLLM_ATTENTION_BACKEND=FLASH_ATTN, FA_SLIDING=0 (try all FA2)",
        },
        # STACKED identity-safe step-shave: native cudagraph+fusion AND attention
        # all OFF at once. base-minus-this = the measured composed stack worth (no
        # naive-sum composition assumption; overlap is captured directly).
        "all_shave_off": {
            "extra_env": cfg_env(enforce_eager=True, fa_sliding=False),
            "lever": "stacked",
            "desc": "ENFORCE_EAGER=1 + FA_SLIDING=0 (full identity-safe stack OFF)",
        },
        # spec-custom inert-proof controls: the drafter never runs on the non-spec
        # base, so these custom spec-decode step-shaves are expected to move the
        # step by ~0 (proof they do NOT transfer to the strict greedy path).
        "onegraph_off": {
            "extra_env": cfg_env(onegraph=False),
            "lever": "spec_custom_inert",
            "desc": "ONEGRAPH=0 (spec-proposer cudagraph; inert-proof on non-spec)",
        },
        "fused_argmax_off": {
            "extra_env": cfg_env(fused_argmax=False),
            "lever": "spec_custom_inert",
            "desc": "FUSED_SPARSE_ARGMAX=0 (drafter sparse argmax; inert-proof)",
        },
    }


# ========================================================================== #
# Decode worker — runs UNDER the server venv (has transformers/torch).
# ========================================================================== #
def _decode_worker(args: argparse.Namespace) -> int:
    """Faithful to the official decode_outputs.py timed loop, but times
    tokenization / request / IO per request and records the per-prompt completion
    token sha256 so the orchestrator can (1) difference request time across output
    lengths for the per-step µs and (2) gate greedy-token-identity vs the base."""
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

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_completion = 0
    for index, record in enumerate(records):
        prompt_text = record["prompt_text"]
        t0 = time.perf_counter()
        prompt_token_ids = od.encode_prompt(tok, prompt_text)
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
        completion_token_ids, _source, _kind = od.extract_generated_token_ids(
            response, choice, prompt_token_ids
        )
        t3 = time.perf_counter()
        total_completion += len(completion_token_ids)
        rows.append({
            "id": record["id"],
            "index": index,
            "num_prompt_tokens": len(prompt_token_ids),
            "num_completion_tokens": len(completion_token_ids),
            "completion_token_sha256": od.sha256_tokens(completion_token_ids),
            "t_tokenize_s": t1 - t0,
            "t_request_s": t2 - t1,
            "t_io_s": t3 - t2,
        })

    summary = {
        "output_len": args.output_len,
        "num_records": len(records),
        "num_completion_tokens": total_completion,
        "per_request": rows,
    }
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[worker] output_len={args.output_len} records={len(records)} "
          f"completion_tokens={total_completion}", flush=True)
    return 0


def _warm_request_stats(summary: dict[str, Any]) -> dict[str, Any]:
    """Warm (post-discard) per-request timing + per-prompt token identity."""
    rows = summary["per_request"]
    # Discard JIT/cache-cold warmup requests, but always keep >=2 warm samples
    # (small smoke passes have few prompts).
    discard = min(WARMUP_DISCARD, max(0, len(rows) - 2))
    warm = rows[discard:]
    t_req_warm = [r["t_request_s"] for r in warm]
    n_comp_warm = [r["num_completion_tokens"] for r in warm]
    return {
        "output_len": summary["output_len"],
        "num_records": len(rows),
        "num_warm": len(warm),
        "mean_request_warm_s": statistics.fmean(t_req_warm) if t_req_warm else float("nan"),
        "stdev_request_warm_s": statistics.pstdev(t_req_warm) if len(t_req_warm) > 1 else 0.0,
        "mean_completion_tokens_warm": statistics.fmean(n_comp_warm) if n_comp_warm else float("nan"),
        # official-boundary-style request-time-only TPS (warm), cross-check.
        "request_only_tps_warm": (
            sum(n_comp_warm) / sum(t_req_warm) if sum(t_req_warm) > 0 else float("nan")
        ),
        # per-prompt token identity keyed by prompt id (ALL records, not just warm).
        "token_sha_by_id": {r["id"]: r["completion_token_sha256"] for r in rows},
    }


# ========================================================================== #
# Orchestrator — serve each config, run the two-length passes.
# ========================================================================== #
def _measure_one_config(
    name: str,
    spec: dict[str, Any],
    *,
    server_python: Path,
    paths_mod: Any,
    harness_mod: Any,
    run_dir: Path,
    port: int,
    num_prompts: int,
    l_short: int,
    l_long: int,
    worker_env: dict[str, str],
    vram_peak: dict[str, float],
) -> dict[str, Any]:
    import subprocess as sp

    cfg_dir = run_dir / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg_dir / "server.log"
    extra_env = spec["extra_env"]
    print(f"\n[measure] ==== config {name} ==== {spec['desc']}", flush=True)
    print(f"[measure] extra_env={extra_env}", flush=True)

    # warmup pass (L_long, fully discarded) then measured L_short + L_long.
    passes = [("warmup", l_long, max(6, WARMUP_DISCARD + 2)),
              ("short", l_short, num_prompts),
              ("long", l_long, num_prompts)]
    pass_summaries: dict[str, dict[str, Any]] = {}
    t_server0 = time.time()
    with harness_mod.LocalServer(
        SUBMISSION, server_python=server_python, port=port,
        startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
    ) as srv:
        bringup_s = time.time() - t_server0
        served_model = srv.served_model_name
        for tag, L, n in passes:
            out_file = cfg_dir / f"pass_{tag}_L{L}.json"
            cmd = [
                str(server_python), str(Path(__file__).resolve()), "--decode-worker",
                "--base-url", srv.base_url,
                "--model", served_model,
                "--dataset-path", str(paths_mod.EVAL_PROMPTS),
                "--tokenizer", paths_mod.TOKENIZER,
                "--num-prompts-worker", str(n),
                "--output-len", str(L),
                "--seed", str(paths_mod.SEED),
                "--out-file", str(out_file),
            ]
            print(f"[measure] {name}: pass {tag} output_len={L} n={n}", flush=True)
            sp.run(cmd, check=True, timeout=3600, env=worker_env)
            if tag == "warmup":
                continue
            pass_summaries[tag] = _warm_request_stats(json.loads(out_file.read_text()))

    short_s = pass_summaries["short"]
    long_s = pass_summaries["long"]
    # per-step µs by two-length differencing of warm request time.
    d_req_s = long_s["mean_request_warm_s"] - short_s["mean_request_warm_s"]
    d_steps = l_long - l_short
    per_step_ms = 1000.0 * d_req_s / d_steps if d_steps else float("nan")
    per_step_us = 1000.0 * per_step_ms
    decode_tps_local = 1000.0 / per_step_ms if per_step_ms and math.isfinite(per_step_ms) else float("nan")
    return {
        "name": name,
        "lever": spec["lever"],
        "desc": spec["desc"],
        "extra_env": extra_env,
        "bringup_s": bringup_s,
        "served_model": served_model,
        "l_short": l_short,
        "l_long": l_long,
        "mean_request_warm_short_ms": 1000.0 * short_s["mean_request_warm_s"],
        "mean_request_warm_long_ms": 1000.0 * long_s["mean_request_warm_s"],
        "per_step_ms": per_step_ms,
        "per_step_us": per_step_us,
        "decode_tps_local": decode_tps_local,
        "request_only_tps_warm_long": long_s["request_only_tps_warm"],
        "num_warm_short": short_s["num_warm"],
        "num_warm_long": long_s["num_warm"],
        "peak_vram_gb": (vram_peak.get("mib", 0.0) or 0.0) / 1024.0,
        # identity uses the LONG pass (longest greedy sequences = strongest gate).
        "token_sha_by_id": long_s["token_sha_by_id"],
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    """Serve each config once; measure per-step µs + capture token identity."""
    import os
    import subprocess as sp
    import threading

    from scripts.local_validation import harness as harness_mod
    from scripts.local_validation import paths as paths_mod

    for note in paths_mod.prepare_local_gpu_env():
        print(f"[measure] {note}", flush=True)

    manifest = harness_mod.load_manifest(SUBMISSION)
    server_python = args.server_python or harness_mod.ensure_server_venv(manifest["dependencies"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = OUT_ROOT / f"measure-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Worker subprocess pinned to the SERVER venv (mirrors harness._participant_env):
    # VIRTUAL_ENV/PATH to the server venv, no inherited PYTHONPATH, SAFEPATH on.
    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    # background VRAM sampler.
    def _vram_mib() -> float | None:
        try:
            out = sp.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, sp.SubprocessError):
            return None
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else None

    vram_peak = {"mib": 0.0}
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            m = _vram_mib()
            if m:
                vram_peak["mib"] = max(vram_peak["mib"], m)
            stop.wait(2.0)

    configs = build_configs()
    if args.only:
        wanted = set(args.only.split(","))
        configs = {k: v for k, v in configs.items() if k in wanted}
        if "base" not in configs:
            raise ValueError("--only must include 'base' (identity + step reference)")

    num_prompts = args.num_prompts
    l_short, l_long = args.l_short, args.l_long

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    results: dict[str, dict[str, Any]] = {}
    try:
        for name, spec in configs.items():
            try:
                results[name] = _measure_one_config(
                    name, spec, server_python=server_python, paths_mod=paths_mod,
                    harness_mod=harness_mod, run_dir=run_dir, port=args.port,
                    num_prompts=num_prompts, l_short=l_short, l_long=l_long,
                    worker_env=worker_env, vram_peak=vram_peak,
                )
            except Exception as exc:  # noqa: BLE001
                # base failure is fatal (no identity/step reference). Any other
                # config can fail in isolation (e.g. an attention-backend that the
                # model rejects) without losing the rest of the matrix -- record it.
                if name == "base":
                    raise
                print(f"[measure] config {name} FAILED: {exc!r}", flush=True)
                results[name] = {
                    "name": name, "lever": spec["lever"], "desc": spec["desc"],
                    "extra_env": spec["extra_env"], "error": repr(exc),
                    "per_step_us": float("nan"), "decode_tps_local": float("nan"),
                    "token_sha_by_id": {}, "peak_vram_gb": 0.0,
                }
    finally:
        stop.set()
        sampler.join(timeout=5)

    return {
        "run_dir": str(run_dir),
        "num_prompts": num_prompts,
        "l_short": l_short,
        "l_long": l_long,
        "peak_vram_gb": (vram_peak.get("mib", 0.0) or 0.0) / 1024.0,
        "configs": results,
    }


# ========================================================================== #
# Analysis
# ========================================================================== #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def analyze(measured: dict[str, Any]) -> dict[str, Any]:
    """Per-lever µs worth, identity verdict, stacked identity-safe step-shave.

    Convention: ``base`` is the fully-optimized strict non-spec int4 base (165.44
    official; native cudagraph + compile fusion + FA2-sliding all ON). Every other
    config turns ONE identity-safe lever OFF, so:

      lever_worth_us = per_step_us(lever OFF) - per_step_us(base)

    POSITIVE lever_worth = the lever, ON in the base, SAVES that many µs/step (the
    base "banks" it). The dual-signed ``shave_us_vs_base = -lever_worth_us`` keeps
    the original sign (POSITIVE only if a config is genuinely FASTER than the
    already-optimized base, i.e. a real *additional* shave the base is missing --
    expected ~none, since the base already turns every identity-safe lever on).

    The stacked identity-safe step-shave is measured DIRECTLY from the
    ``all_shave_off`` config (no naive-sum composition): the µs the base banks by
    having the whole native-cudagraph+fusion+attention stack on at once.
    """
    configs = measured["configs"]
    base = configs["base"]
    base_step_us = base["per_step_us"]
    base_sha = base["token_sha_by_id"]

    # official per-step time of the strict base, anchored at the 165.44 TPS a10g
    # measurement (non-spec M=1 AR -> 1 token/step -> official_step_us = 1e6/TPS).
    official_base_step_us = 1e6 / STRICT_BASE_OFFICIAL_TPS_196
    gap_tps_to_500 = TARGET_TPS - STRICT_BASE_OFFICIAL_TPS_196

    levers: dict[str, Any] = {}
    for name, cfg in configs.items():
        if name == "base":
            continue
        cfg_step = cfg.get("per_step_us", float("nan"))
        # identity vs base, by prompt id (long pass = strongest greedy gate).
        sha = cfg.get("token_sha_by_id", {})
        ids = sorted(set(base_sha) & set(sha))
        matched = sum(1 for i in ids if base_sha[i] == sha[i])
        identity_frac = matched / len(ids) if ids else float("nan")
        identity_preserved = bool(ids) and matched == len(ids)
        # lever worth: µs the lever (ON in base, OFF here) banks. +ve => helps base.
        if _finite(cfg_step) and _finite(base_step_us):
            lever_worth_us = cfg_step - base_step_us
            lever_worth_frac = lever_worth_us / base_step_us if base_step_us else float("nan")
        else:
            lever_worth_us = float("nan")
            lever_worth_frac = float("nan")
        shave_us = -lever_worth_us if _finite(lever_worth_us) else float("nan")
        shave_frac = -lever_worth_frac if _finite(lever_worth_frac) else float("nan")
        # official: strip this lever off the official step (add back its worth).
        if _finite(lever_worth_us):
            official_step_without = official_base_step_us + lever_worth_us
            official_tps_without = 1e6 / official_step_without if official_step_without > 0 else float("nan")
            official_tps_lift_from_lever = STRICT_BASE_OFFICIAL_TPS_196 - official_tps_without
        else:
            official_tps_without = float("nan")
            official_tps_lift_from_lever = float("nan")
        levers[name] = {
            "lever": cfg["lever"],
            "desc": cfg["desc"],
            "per_step_us": cfg_step,
            "decode_tps_local": cfg.get("decode_tps_local", float("nan")),
            "lever_worth_us": lever_worth_us,
            "lever_worth_frac": lever_worth_frac,
            "shave_us_vs_base": shave_us if _finite(shave_us) else 0.0,
            "shave_frac_vs_base": shave_frac if _finite(shave_frac) else 0.0,
            "identity_frac": identity_frac,
            "identity_preserved": identity_preserved,
            "num_prompts_compared": len(ids),
            "official_tps_without_lever": official_tps_without,
            "official_tps_lift_from_lever": official_tps_lift_from_lever,
            "error": cfg.get("error"),
        }

    # Banked native step-shave levers (one-at-a-time): identity-preserving, finite,
    # positive worth, excluding the stacked config and the spec-custom inert controls.
    NATIVE = {"cudagraph_fusion_native", "cudagraph_native", "attention"}
    banked = {
        n: v for n, v in levers.items()
        if v["lever"] in NATIVE and v["identity_preserved"]
        and _finite(v["lever_worth_us"]) and v["lever_worth_us"] > 0
    }
    highest_yield_shave = max(banked, key=lambda n: banked[n]["lever_worth_us"]) if banked else None

    # Stacked identity-safe step-shave: prefer the DIRECT all_shave_off measurement;
    # fall back to composing the banked one-at-a-time worths if it is missing.
    # When composing, reduce by lever FAMILY, not by config: several configs can
    # measure the SAME physical lever (e.g. fa_sliding_off and attn_flash_attn_all
    # are two attention-backend variants of the single `attention` lever), so a raw
    # config-sum would double-count that family. Take the max worth per family, then
    # sum across distinct families. `naive_sum_worth_us` keeps the raw config-sum so
    # `composed_lt_naive_sum` is a meaningful (non-trivial) assertion.
    stacked = levers.get("all_shave_off")
    if stacked and stacked["identity_preserved"] and _finite(stacked["lever_worth_us"]) and stacked["lever_worth_us"] > 0:
        stacked_worth_us = stacked["lever_worth_us"]
        stacked_source = "measured_all_shave_off"
    else:
        by_family: dict[str, float] = {}
        for v in banked.values():
            fam = str(v["lever"])
            by_family[fam] = max(by_family.get(fam, 0.0), v["lever_worth_us"])
        stacked_worth_us = sum(by_family.values())
        stacked_source = "composed_from_banked_levers_by_family"
    stacked_worth_frac = stacked_worth_us / official_base_step_us if official_base_step_us else 0.0
    naive_sum_worth_us = sum(v["lever_worth_us"] for v in banked.values())

    # Genuine ADDITIONAL shave the base is missing (any identity-preserving config
    # FASTER than the already-optimized base). Expected ~none on the strict base.
    extra = {
        n: v for n, v in levers.items()
        if v["identity_preserved"] and _finite(v["shave_us_vs_base"]) and v["shave_us_vs_base"] > 0
        and n != "all_shave_off"
    }
    if extra:
        extra_name = max(extra, key=lambda n: extra[n]["shave_us_vs_base"])
        additional_identity_shave_us = extra[extra_name]["shave_us_vs_base"]
    else:
        extra_name = None
        additional_identity_shave_us = 0.0

    # The fully-shaved strict base IS 165.44 (it already banks the stack); any
    # genuine additional shave lifts it marginally. This is the "after-shave" TPS.
    official_step_after = official_base_step_us - additional_identity_shave_us
    official_tps_after = 1e6 / official_step_after if official_step_after > 0 else STRICT_BASE_OFFICIAL_TPS_196
    # hypothetical stripped base (whole identity-safe stack OFF) for the lift story.
    official_tps_base_stripped = 1e6 / (official_base_step_us + stacked_worth_us) if (official_base_step_us + stacked_worth_us) > 0 else float("nan")
    official_tps_lift_from_stack = STRICT_BASE_OFFICIAL_TPS_196 - official_tps_base_stripped if _finite(official_tps_base_stripped) else float("nan")
    frac_of_gap = (official_tps_after - STRICT_BASE_OFFICIAL_TPS_196) / gap_tps_to_500 if gap_tps_to_500 else 0.0

    # inert levers: |worth| within ~1% of the base step (e.g. spec-custom controls).
    inert = [n for n, v in levers.items() if _finite(v["lever_worth_frac"]) and abs(v["lever_worth_frac"]) < 0.01]
    failed = [n for n, v in levers.items() if v.get("error")]
    # a config that FAILED to boot is a failure, not an identity-breaker.
    identity_breakers = [n for n, v in levers.items() if not v["identity_preserved"] and not v.get("error")]

    step_shave_closes_500_gap = bool(_finite(official_tps_after) and official_tps_after >= TARGET_TPS)

    return {
        "base_per_step_us_local": base_step_us,
        "base_decode_tps_local": base["decode_tps_local"],
        "base_official_tps_196": STRICT_BASE_OFFICIAL_TPS_196,
        "official_base_step_us": official_base_step_us,
        "gap_tps_to_500": gap_tps_to_500,
        "levers": levers,
        # headline (PRIMARY deliverables) -----------------------------------
        "max_identity_step_shave_us": stacked_worth_us,
        "max_identity_step_shave_frac": stacked_worth_frac,
        "stacked_worth_source": stacked_source,
        "naive_sum_worth_us": naive_sum_worth_us,
        "composed_lt_naive_sum": bool(_finite(stacked_worth_us) and stacked_worth_us <= naive_sum_worth_us + 1e-6),
        "highest_yield_shave": highest_yield_shave,
        "additional_identity_shave_us": additional_identity_shave_us,
        "additional_shave_config": extra_name,
        "official_tps_after_shave_if_applied": official_tps_after,
        "official_tps_base_stripped": official_tps_base_stripped,
        "official_tps_lift_from_stack": official_tps_lift_from_stack,
        "frac_of_gap_to_500_closed": frac_of_gap,
        "step_shave_closes_500_gap": step_shave_closes_500_gap,
        # diagnostics --------------------------------------------------------
        "banked_native_levers": sorted(banked),
        "inert_levers": inert,
        "identity_breaking_levers": identity_breakers,
        "failed_levers": failed,
        "pred_realizable_frac": PRED_REALIZABLE_FRAC,
        "pred_optimistic_frac": PRED_OPTIMISTIC_FRAC,
    }


def build_self_test(measured: dict[str, Any] | None, analysis: dict[str, Any] | None) -> dict[str, Any]:
    """PRIMARY: strict_step_shave_stack_self_test_passes.

    A MEASUREMENT harness self-test: the run is well-formed and the deliverables
    are computable. It does NOT require any lever to help (a measured null is a
    valid scientific result).

    (a) base measured: finite per_step_us > 0 and a non-empty identity reference.
    (b) every non-base config produced a per-prompt token sha map (identity gate
        is computable for each lever).
    (c) every lever has a finite shave_us and a bool identity verdict.
    (d) the headline deliverables are finite.
    (e) NaN-clean over headline floats.
    (f) constants imported exactly.
    """
    st: dict[str, Any] = {}
    measured_ok = measured is not None and analysis is not None
    st["measured_present"] = bool(measured_ok)

    if measured_ok:
        configs = measured["configs"]
        base = configs.get("base", {})
        base_us = base.get("per_step_us", float("nan"))
        base_sha = base.get("token_sha_by_id", {})
        st["base_step_us_ok"] = bool(isinstance(base_us, float) and math.isfinite(base_us) and base_us > 0)
        st["base_identity_ref_ok"] = bool(len(base_sha) > 0)
        # (b)+(c)
        lever_sha_ok = True
        lever_delta_ok = True
        n_measured_levers = 0
        for name, cfg in configs.items():
            if name == "base":
                continue
            # A config that failed to boot is recorded honestly (error set, no sha
            # map); it must NOT sink the PRIMARY self-test. Only require a sha map
            # for configs that were actually measured.
            if not cfg.get("error"):
                if not cfg.get("token_sha_by_id"):
                    lever_sha_ok = False
                else:
                    n_measured_levers += 1
            v = analysis["levers"].get(name, {})
            sv = v.get("shave_us_vs_base")
            if not isinstance(sv, float) or not math.isfinite(sv):
                lever_delta_ok = False
            if not isinstance(v.get("identity_preserved"), bool):
                lever_delta_ok = False
        st["lever_identity_maps_ok"] = bool(lever_sha_ok and n_measured_levers >= 1)
        st["lever_deltas_ok"] = bool(lever_delta_ok)
        # (d)+(e)
        headline = [
            analysis["base_per_step_us_local"],
            analysis["max_identity_step_shave_us"],
            analysis["max_identity_step_shave_frac"],
            analysis["official_tps_after_shave_if_applied"],
            analysis["frac_of_gap_to_500_closed"],
            analysis["official_base_step_us"],
            analysis["gap_tps_to_500"],
        ]
        st["headline_finite_ok"] = all(
            isinstance(x, float) and math.isfinite(x) for x in headline
        )
        st["nan_clean_ok"] = st["headline_finite_ok"]
    else:
        st["base_step_us_ok"] = False
        st["base_identity_ref_ok"] = False
        st["lever_identity_maps_ok"] = False
        st["lever_deltas_ok"] = False
        st["headline_finite_ok"] = False
        st["nan_clean_ok"] = False

    # (f) constants imported exactly.
    st["constants_ok"] = bool(
        TARGET_TPS == 500.0
        and STRICT_BASE_OFFICIAL_TPS_196 == 165.44
        and DEPLOYED_SPEC_OFFICIAL_TPS == 481.53
        and TAU_LO_267 == 1.0352356533046398
        and LOCAL_BAR_FOR_500_267 == 482.9818200367414
    )

    st["passes"] = bool(
        st["measured_present"]
        and st["base_step_us_ok"]
        and st["base_identity_ref_ok"]
        and st["lever_identity_maps_ok"]
        and st["lever_deltas_ok"]
        and st["headline_finite_ok"]
        and st["nan_clean_ok"]
        and st["constants_ok"]
    )
    return st


# ========================================================================== #
# Report + wandb
# ========================================================================== #
def build_report(measured: dict[str, Any] | None) -> dict[str, Any]:
    analysis = analyze(measured) if measured else None
    self_test = build_self_test(measured, analysis)
    report = {
        "strict_step_shave_stack_analysis_only": True,
        "baseline_official_tps": DEPLOYED_SPEC_OFFICIAL_TPS,
        "strict_base_official_tps": STRICT_BASE_OFFICIAL_TPS_196,
        "tps_delta": 0.0,
        "target_tps": TARGET_TPS,
        "tau_lo": TAU_LO_267,
        "analysis": analysis,
        "self_test": self_test,
        "strict_step_shave_stack_self_test_passes": self_test["passes"],
        "measured": measured,
    }
    return report


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="validity-profile",
        agent="kanna",
        name=args.wandb_name or "kanna/strict-step-shave-stack",
        group=args.wandb_group or "strict-step-shave-359",
        tags=["strict-step-shave", "step-us-measure", "non-spec-int4", "local-a10g",
              "identity-gate", "pr-359"],
        notes="PR #359 GPU-backed per-step micro-lever measurement on the strict non-spec base",
        config={
            "submission": str(SUBMISSION),
            "strict_base_official_tps_196": STRICT_BASE_OFFICIAL_TPS_196,
            "target_tps": TARGET_TPS,
            "tau_lo_267": TAU_LO_267,
            "num_prompts": (report.get("measured") or {}).get("num_prompts"),
            "l_short": (report.get("measured") or {}).get("l_short"),
            "l_long": (report.get("measured") or {}).get("l_long"),
        },
    )
    if run is None:
        print("[wandb] init returned None — skipping", flush=True)
        return None
    a = report.get("analysis") or {}
    st = report["self_test"]
    summary: dict[str, Any] = {
        "self_test_passes": int(bool(st["passes"])),
        "strict_step_shave_stack_analysis_only": 1,
        "tps_delta": 0.0,
        "baseline_official_tps": DEPLOYED_SPEC_OFFICIAL_TPS,
    }
    if a:
        summary.update({
            # PRIMARY + advisor-named emit set (kanna #359) -----------------
            "strict_step_shave_stack_self_test_passes": int(bool(st["passes"])),
            "max_identity_step_shave_frac": a["max_identity_step_shave_frac"],
            "tps_after_shave": a["official_tps_after_shave_if_applied"],
            "step_shave_closes_500_gap": int(bool(a["step_shave_closes_500_gap"])),
            "highest_yield_shave": a["highest_yield_shave"] or "none",
            # supporting -----------------------------------------------------
            "base_per_step_us_local": a["base_per_step_us_local"],
            "base_decode_tps_local": a["base_decode_tps_local"],
            "official_base_step_us": a["official_base_step_us"],
            "gap_tps_to_500": a["gap_tps_to_500"],
            "max_identity_step_shave_us": a["max_identity_step_shave_us"],
            "stacked_worth_source": a["stacked_worth_source"],
            "naive_sum_worth_us": a["naive_sum_worth_us"],
            "composed_lt_naive_sum": int(bool(a["composed_lt_naive_sum"])),
            "additional_identity_shave_us": a["additional_identity_shave_us"],
            "official_tps_after_shave_if_applied": a["official_tps_after_shave_if_applied"],
            "official_tps_base_stripped": a["official_tps_base_stripped"],
            "official_tps_lift_from_stack": a["official_tps_lift_from_stack"],
            "frac_of_gap_to_500_closed": a["frac_of_gap_to_500_closed"],
            "n_banked_native_levers": len(a["banked_native_levers"]),
            "n_inert_levers": len(a["inert_levers"]),
            "n_identity_breaking_levers": len(a["identity_breaking_levers"]),
            "n_failed_levers": len(a["failed_levers"]),
            "peak_vram_gb": (report.get("measured") or {}).get("peak_vram_gb"),
        })
        # per-lever metrics (flattened).
        for name, v in a["levers"].items():
            summary[f"lever.{name}.per_step_us"] = v["per_step_us"]
            summary[f"lever.{name}.lever_worth_us"] = v["lever_worth_us"]
            summary[f"lever.{name}.lever_worth_frac"] = v["lever_worth_frac"]
            summary[f"lever.{name}.shave_us"] = v["shave_us_vs_base"]
            summary[f"lever.{name}.identity_frac"] = v["identity_frac"]
            summary[f"lever.{name}.identity_preserved"] = int(bool(v["identity_preserved"]))
    summary = {k: v for k, v in summary.items() if v is not None}
    log_summary(run, summary, step=0)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(report: dict[str, Any]) -> None:
    a = report.get("analysis")
    st = report["self_test"]
    line = "=" * 14 + " STRICT STEP-SHAVE STACK (PR #359) " + "=" * 14
    print("\n" + line, flush=True)
    if a:
        print(f"  strict base: {a['base_per_step_us_local']:.1f} µs/step local "
              f"({a['base_decode_tps_local']:.2f} local TPS); official anchor "
              f"{a['base_official_tps_196']:.2f} TPS = {a['official_base_step_us']:.1f} µs/step",
              flush=True)
        print(f"  gap to {TARGET_TPS:.0f}: {a['gap_tps_to_500']:.2f} official TPS", flush=True)
        print("  levers (per-step µs; lever_worth = µs the lever banks into base when ON):", flush=True)
        for name, v in a["levers"].items():
            if v.get("error"):
                print(f"    {name:<22}   FAILED: {v['error']}", flush=True)
                continue
            ident = "IDENTICAL" if v["identity_preserved"] else f"DIVERGENT({v['identity_frac']:.3f})"
            print(f"    {name:<22} {v['per_step_us']:>9.1f} us  "
                  f"worth {v['lever_worth_us']:>+8.1f} us ({v['lever_worth_frac']:>+6.2%})  "
                  f"{ident}", flush=True)
        print(f"  highest-yield identity-safe lever: {a['highest_yield_shave']}  "
              f"(banked native levers: {', '.join(a['banked_native_levers']) or 'none'})", flush=True)
        print(f"  STACKED identity-safe step-shave: {a['max_identity_step_shave_us']:.1f} us "
              f"({a['max_identity_step_shave_frac']:.3%} of official step)  "
              f"[source={a['stacked_worth_source']}; naive-sum {a['naive_sum_worth_us']:.1f} us; "
              f"composed<=sum={a['composed_lt_naive_sum']}]", flush=True)
        print(f"  [pred realizable {a['pred_realizable_frac']:.3%} / optimistic {a['pred_optimistic_frac']:.3%}]",
              flush=True)
        print(f"  strict base (fully shaved) {a['base_official_tps_196']:.2f} TPS  vs  "
              f"stripped (stack OFF) {a['official_tps_base_stripped']:.2f} TPS  "
              f"=> stack lifts +{a['official_tps_lift_from_stack']:.2f} TPS", flush=True)
        print(f"  additional shave beyond base: {a['additional_identity_shave_us']:.1f} us "
              f"=> after-shave {a['official_tps_after_shave_if_applied']:.2f} TPS "
              f"(closes {a['frac_of_gap_to_500_closed']:.2%} of gap)  "
              f"closes_500={a['step_shave_closes_500_gap']}", flush=True)
        if a["inert_levers"]:
            print(f"  INERT on non-spec base (|worth|<1%): {', '.join(a['inert_levers'])}", flush=True)
        if a["identity_breaking_levers"]:
            print(f"  EXCLUDED (break identity): {', '.join(a['identity_breaking_levers'])}", flush=True)
        if a["failed_levers"]:
            print(f"  FAILED to boot/measure: {', '.join(a['failed_levers'])}", flush=True)
    print(f"\n  SELF-TEST strict_step_shave_stack_self_test_passes = {st['passes']}", flush=True)
    for k in ("measured_present", "base_step_us_ok", "base_identity_ref_ok",
              "lever_identity_maps_ok", "lever_deltas_ok", "headline_finite_ok",
              "nan_clean_ok", "constants_ok"):
        print(f"    {k} = {st[k]}", flush=True)
    print("=" * len(line) + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true",
                    help="serve the strict base + lever configs and measure per-step µs")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny base-only plumbing run (implies --measure --only base, no wandb)")
    ap.add_argument("--reanalyze", type=Path, default=None,
                    help="0-GPU: re-derive analysis+self-test from a saved results/report "
                         "JSON's measured block and rewrite results.json (no serving, no wandb)")
    ap.add_argument("--self-test", action="store_true",
                    help="exit non-zero unless strict_step_shave_stack_self_test_passes")
    ap.add_argument("--only", default=None,
                    help="comma-separated config names to run (must include 'base')")
    ap.add_argument("--num-prompts", type=int, default=FULL_NUM_PROMPTS)
    ap.add_argument("--l-short", type=int, default=FULL_L_SHORT)
    ap.add_argument("--l-long", type=int, default=FULL_L_LONG)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")

    # decode-worker (internal, runs under the server venv)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--num-prompts-worker", dest="num_prompts_worker", type=int, default=None)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--out-file")
    args = ap.parse_args(argv)

    if args.decode_worker:
        # the worker reuses --num-prompts-worker as its count.
        if args.num_prompts_worker is not None:
            args.num_prompts = args.num_prompts_worker
        return _decode_worker(args)

    if args.smoke:
        args.measure = True
        args.only = "base"
        args.no_wandb = True
        args.num_prompts = min(args.num_prompts, SMOKE_NUM_PROMPTS)
        args.l_short = SMOKE_L_SHORT
        args.l_long = SMOKE_L_LONG

    measured = None
    if args.reanalyze:
        saved = json.loads(Path(args.reanalyze).read_text())
        measured = saved.get("measured")
        if not measured:
            print(f"[reanalyze] no 'measured' block in {args.reanalyze}", flush=True)
            return 1
        args.no_wandb = True  # re-derivation is a 0-GPU pure-analysis pass.
        print(f"[reanalyze] re-deriving analysis from {args.reanalyze} (0-GPU, no serving)", flush=True)
    elif args.measure:
        measured = run_measurement(args)

    report = build_report(measured)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if measured and not args.no_wandb:
        report["wandb_run_id"] = log_wandb(report, args)
    elif args.reanalyze:
        # Carry the raw-measurement run id forward; the re-derivation logs no new
        # wandb run, so the corrected card stays pinned to its source measurement.
        report["wandb_run_id"] = saved.get("wandb_run_id")
        report["reanalyzed_from"] = {
            "path": str(args.reanalyze),
            "source_created_at": saved.get("created_at"),
        }
    report["created_at"] = stamp
    (OUT_ROOT / "strict_step_shave_stack_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    if measured:
        (Path(measured["run_dir"]) / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_summary(report)
    print(f"[report] {OUT_ROOT / 'strict_step_shave_stack_results.json'}", flush=True)

    if args.self_test:
        return 0 if report["strict_step_shave_stack_self_test_passes"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
