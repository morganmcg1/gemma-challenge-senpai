#!/usr/bin/env python
"""Synthetic batched-verify cost model for int4 Gemma decode on A10G (PR #18).

WHAT THIS MEASURES
------------------
Speculative decoding amortizes one weight read over many tokens: the verify step
forwards M = K+1 query positions for ONE sequence (1 bonus + K draft tokens)
against a shared KV context, computes logits at all M positions, and rejection-
samples. At concurrency=1 the int4 base decode is ~weight-GEMM / bandwidth bound,
so the per-step latency should scale SUB-linearly with M up to a knee M*, making
accepted tokens "nearly free" until the lm_head (262k-vocab projection over M
positions) or attention starts to bite.

This is a LATENCY/THROUGHPUT microbenchmark with SYNTHETIC candidate tokens. No
drafter, no greedy gate, no correctness check, no HF Job. It drives vLLM's own
`GPUModelRunner._dummy_run` (the blessed "run a forward of N tokens" primitive)
to issue the exact 1-request, M-query-token decode shape, with `profile_seq_lens`
set to ctx+M so attention attends over a realistic KV context. The lm_head is
timed separately via `model.compute_logits` over all M positions (the term that
grows with M). Component shares come from torch.profiler self-device time, same
categorisation as the official gemma_decode_profiler.

The per-verify-step latency is t_step(M) = t_forward(M) + t_lmhead(M).

Run M=1 in graph mode and confirm it reproduces the PR #7 int4 base (~10.3 ms/tok
== 96.89 TPS at ctx~256); that calibrates the harness.

OUTPUT
------
results.json: raw per-(mode,ctx,M) latencies + component shares + derived cost
model (ideal/realistic TPS ceilings, knee M*, optimal K*). Optionally a W&B run
with profiling tables.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

# Must be set before importing torch/vllm. See research/spec_cost_model/report.md.
# This A10G node inherits CUDA_VISIBLE_DEVICES=5 (host physical id), but the
# in-container GPU is index 0 — the inherited value makes torch.cuda unavailable,
# so force 0 (single-GPU node) rather than setdefault, which would keep the 5.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process => profiler sees kernels
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")     # avoid flashinfer JIT (curand.h missing)

import numpy as np
import torch

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"

# Kernel-name -> category (lowercase substring match). Order matters: GEMM first
# so Marlin/cutlass matmuls win before generic buckets. Mirrors the official
# gemma_decode_profiler graph-mode categories.
CATEGORIES = [
    ("matmul_gemm", ["marlin", "gptq", "gemm", "cutlass", "wmma", "gemv", "splitk",
                     "split_k", "ampere", "s16816", "s1688", "dot", "cublas", "cijk"]),
    ("attention", ["attn", "_fwd", "flash", "paged", "unified_attention",
                   "reshape_and_cache", "rotary", "rope"]),
    ("sampling_lmhead", ["log_softmax", "logsoftmax", "argmax", "topk", "top_k",
                         "softmax", "sample", "logits", "cumsum", "sort"]),
    ("norm", ["rms", "layernorm", "layer_norm", "norm_kernel"]),
    ("activation", ["silu", "gelu", "swiglu", "act_and_mul", "geglu"]),
    ("elementwise_copy", ["elementwise", "copy", "cast", "convert", "memcpy",
                          "fill", "index", "vectorized", "_to_copy", "add_kernel",
                          "mul_kernel"]),
]


def categorize(name: str) -> str:
    n = name.lower()
    if "marlin" in n or "gemv" in n or "gemm" in n:
        return "matmul_gemm"
    for cat, subs in CATEGORIES:
        if any(s in n for s in subs):
            return cat
    return "other"


def self_dev_us(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def find_runner(obj, depth=0, seen=None):
    """Walk the in-process engine object graph to the GPUModelRunner."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 10:
        return None
    seen.add(id(obj))
    if isinstance(obj, GPUModelRunner):
        return obj
    for attr in ("llm_engine", "engine_core", "engine", "model_executor", "executor",
                 "driver_worker", "worker", "model_runner", "core", "engines"):
        child = getattr(obj, attr, None)
        if child is not None:
            r = find_runner(child, depth + 1, seen)
            if r is not None:
                return r
    return None


def build_llm(model: str, enforce_eager: bool, m_sweep: list[int], max_ctx: int):
    from vllm import LLM
    max_batched = max(2048, max(m_sweep) + 8)
    kwargs = dict(
        model=model,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=max(4096, max_ctx + max(m_sweep) + 64),
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=max_batched,
        max_num_seqs=1,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        disable_log_stats=True,
        seed=0,
    )
    if not enforce_eager:
        # With max_num_seqs=1 the default max capture size is 2, so M>=4 would fall
        # back to eager and wreck the graph-mode curve. Pin capture sizes to the
        # M sweep so every M hits an exact piecewise graph.
        kwargs["compilation_config"] = {
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": sorted(set(m_sweep)),
        }
    return LLM(**kwargs)


def make_step_fns(runner, ctx: int, mode: str):
    """Return (forward_fn, lmhead_fn) for a single-request M-token verify step."""
    from vllm.config import CUDAGraphMode
    model = runner.model
    cg_mode = CUDAGraphMode.NONE if mode == "eager" else None  # None => dispatcher picks PIECEWISE

    def forward(M: int):
        hs, _ = runner._dummy_run(
            num_tokens=M,
            cudagraph_runtime_mode=cg_mode,
            force_attention=True,
            uniform_decode=False,
            skip_eplb=True,
            is_profile=False,
            profile_seq_lens=ctx + M,
        )
        return hs

    def lmhead(hs):
        logits = model.compute_logits(hs)        # [M, vocab] — verify needs all M
        _ = logits.argmax(dim=-1)                # greedy target token per position
        return logits

    return forward, lmhead


def _pipelined(forward, lmhead, M, steps):
    """Per-step GPU times (ms) under back-to-back enqueue with NO per-step sync.

    The CPU runs ahead and the GPU stays saturated — the same condition a real
    async-scheduled drafter (vLLM async scheduling is on) achieves, and the same
    condition under which the PR #7 reference 96.89 TPS was measured (256 tokens
    from one generate). Per-step deltas are read from the CUDA-event timeline AFTER
    a single final sync, so they are free of the CPU-prep bubble that a per-step
    sync would expose. This is the correct latency basis for a *throughput ceiling*.
    """
    ev = lambda: torch.cuda.Event(enable_timing=True)
    e0 = [ev() for _ in range(steps)]   # before forward
    e1 = [ev() for _ in range(steps)]   # after forward / before lm_head
    e2 = [ev() for _ in range(steps)]   # after lm_head
    torch.cuda.synchronize()
    for i in range(steps):
        e0[i].record()
        hs = forward(M)
        e1[i].record()
        lmhead(hs)
        e2[i].record()
    torch.cuda.synchronize()
    f_ms = [e0[i].elapsed_time(e1[i]) for i in range(steps)]
    l_ms = [e1[i].elapsed_time(e2[i]) for i in range(steps)]
    s_ms = [e0[i].elapsed_time(e2[i]) for i in range(steps)]
    return f_ms, l_ms, s_ms


def _serialized(forward, lmhead, M, steps):
    """Isolated per-step latency: sync after every step (CPU+GPU serialized).

    Larger than pipelined by the per-step CPU-prep bubble that real async serving
    overlaps away; reported as a diagnostic only, not used for the ceiling."""
    ev = lambda: torch.cuda.Event(enable_timing=True)
    s_ms = []
    for _ in range(steps):
        a, b = ev(), ev()
        a.record()
        lmhead(forward(M))
        b.record()
        torch.cuda.synchronize()
        s_ms.append(a.elapsed_time(b))
    return s_ms


def time_config(runner, M, ctx, mode, steps, warmup):
    """Median per-step latency (ms) for forward, lmhead, and full verify step.

    Primary t_step_ms is the PIPELINED (throughput) time — the basis for the TPS
    ceiling and for the M=1 calibration against PR #7's 96.89 TPS. A serialized
    isolated-latency number is recorded alongside as a diagnostic."""
    forward, lmhead = make_step_fns(runner, ctx, mode)
    pct = lambda xs, p: float(np.percentile(xs, p))
    with torch.inference_mode():
        for _ in range(warmup):
            lmhead(forward(M))
        torch.cuda.synchronize()
        f_ms, l_ms, s_ms = _pipelined(forward, lmhead, M, steps)
        ser_ms = _serialized(forward, lmhead, M, max(20, steps // 4))

    med_pipe = statistics.median(s_ms)
    med_ser = statistics.median(ser_ms)
    return {
        "t_forward_ms": statistics.median(f_ms),
        "t_lmhead_ms": statistics.median(l_ms),
        "t_step_ms": med_pipe,
        "t_step_p25_ms": pct(s_ms, 25),
        "t_step_p75_ms": pct(s_ms, 75),
        "t_forward_p25_ms": pct(f_ms, 25),
        "t_forward_p75_ms": pct(f_ms, 75),
        "t_step_serialized_ms": med_ser,
        "pipeline_speedup": (med_ser / med_pipe) if med_pipe else None,
    }


def profile_config(runner, M, ctx, mode, profile_steps):
    """Self-device-time category shares of the FORWARD (transformer) pass, plus the
    lm_head self-device time, so we can split attention / GEMM / lm_head / overhead."""
    from torch.profiler import profile, ProfilerActivity
    forward, lmhead = make_step_fns(runner, ctx, mode)
    with torch.inference_mode():
        for _ in range(5):
            lmhead(forward(M))
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     acc_events=True) as prof:
            for _ in range(profile_steps):
                forward(M)
            torch.cuda.synchronize()
        fwd_rows = [(e.key, self_dev_us(e)) for e in prof.key_averages()]

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     acc_events=True) as prof2:
            for _ in range(profile_steps):
                hs = forward(M)
                lmhead(hs)
            torch.cuda.synchronize()
        all_rows = [(e.key, self_dev_us(e)) for e in prof2.key_averages()]

    def cats(rows):
        c: dict[str, float] = {}
        for name, us in rows:
            if us > 0:
                c[categorize(name)] = c.get(categorize(name), 0.0) + us
        return c

    fwd_cats = cats(fwd_rows)
    fwd_busy = sum(fwd_cats.values()) or 1.0
    all_cats = cats(all_rows)
    all_busy = sum(all_cats.values()) or 1.0
    return {
        "forward_busy_us": fwd_busy,
        "forward_category_pct": {k: 100 * v / fwd_busy for k, v in fwd_cats.items()},
        "step_busy_us": all_busy,
        "step_category_pct": {k: 100 * v / all_busy for k, v in all_cats.items()},
    }


def _install_splitkv(patch_path: str):
    """Import + install the #43 split-KV verify patch (PR #51 re-grounding).

    Returns the patch module so the caller can read its redirect counter after the
    sweep. Must run AFTER build_llm so the vLLM ops module is already imported and
    the patch swaps unified_attention in place (rather than only arming the
    import-time finder)."""
    import importlib.util as _ilu
    patch_path = os.path.abspath(patch_path)
    spec = _ilu.spec_from_file_location("splitkv_verify_patch", patch_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    active = mod.install()
    print(f"[cost] splitkv patch {patch_path} install()={active} "
          f"SPLITKV_VERIFY={os.environ.get('SPLITKV_VERIFY', '1')} "
          f"max_q<={getattr(mod, 'SPLITKV_VERIFY_MAX_Q', '?')}", flush=True)
    return mod


def run_mode(model, mode, m_sweep, ctx_sweep, steps, warmup, profile_steps,
             splitkv_patch=None):
    print(f"\n[cost] ===== MODE={mode} =====", flush=True)
    t0 = time.time()
    llm = build_llm(model, enforce_eager=(mode == "eager"), m_sweep=m_sweep,
                    max_ctx=max(ctx_sweep))
    print(f"[cost] {mode} LLM ready in {time.time()-t0:.1f}s", flush=True)
    splitkv_mod = None
    if splitkv_patch:
        splitkv_mod = _install_splitkv(splitkv_patch)
    runner = find_runner(llm)
    if runner is None:
        raise RuntimeError("could not locate GPUModelRunner")
    # Read hidden size from an actual forward's hidden-state shape (config layout
    # varies for multimodal Gemma4Config — text_config.hidden_size etc.).
    hidden = None
    try:
        fwd, _ = make_step_fns(runner, ctx_sweep[0], mode)
        with torch.inference_mode():
            hidden = int(fwd(1).shape[-1])
    except Exception:
        pass

    def _redirected():
        return int(splitkv_mod._stats["redirected"]) if splitkv_mod else 0

    rows = []
    for ctx in ctx_sweep:
        for M in m_sweep:
            r0 = _redirected()
            t = time_config(runner, M, ctx, mode, steps, warmup)
            p = profile_config(runner, M, ctx, mode, profile_steps)
            redir = _redirected() - r0
            row = {"mode": mode, "ctx": ctx, "M": M, **t, **p,
                   "splitkv_redirected": redir}
            rows.append(row)
            attn = p["step_category_pct"].get("attention", 0.0)
            sk = f" splitkv_redir={redir}" if splitkv_mod else ""
            print(f"[cost] {mode} ctx={ctx:4d} M={M:2d}: step={t['t_step_ms']:7.3f}ms "
                  f"(fwd={t['t_forward_ms']:7.3f} lmhead={t['t_lmhead_ms']:6.3f}) "
                  f"attn={attn:4.1f}% | TPS_ideal(K={M-1 if M>1 else 1})="
                  f"{(M-1 if M>1 else 1)/(t['t_step_ms']/1000):6.1f}{sk}", flush=True)

    splitkv_info = None
    if splitkv_mod:
        splitkv_info = {"active": True, "patch_path": os.path.abspath(splitkv_patch),
                        "total_redirected": _redirected(),
                        "max_q": getattr(splitkv_mod, "SPLITKV_VERIFY_MAX_Q", None)}
        print(f"[cost] {mode} splitkv TOTAL redirected={splitkv_info['total_redirected']} "
              f"(verify batches routed to 3D split-KV)", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return rows, hidden, splitkv_info


# ---------------------------------------------------------------------------
# Cost model: ideal + realistic TPS ceilings, knee, optimal K.
# ---------------------------------------------------------------------------
def expected_accepted(model_spec, K):
    """E[emitted tokens per verify step] for a draft of length K (M=K+1 positions)."""
    kind, val = model_spec
    if kind == "flat":
        # Constant expected accepted (clamped to <= K+1, the hard ceiling).
        return min(val, K + 1)
    if kind == "geom":
        p = val
        # Leviathan/Chen i.i.d. acceptance: (1 - p^(K+1)) / (1 - p). At p->1 => K+1.
        return (1 - p ** (K + 1)) / (1 - p)
    raise ValueError(kind)


def build_cost_model(rows, accept_models):
    """For each (mode, ctx): latency(M), marginal cost, ideal & realistic TPS, knee, K*."""
    out = {}
    by_key: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        by_key.setdefault((r["mode"], r["ctx"]), {})[r["M"]] = r
    for (mode, ctx), mrows in by_key.items():
        Ms = sorted(mrows)
        lat = {M: mrows[M]["t_step_ms"] for M in Ms}
        # marginal per-token cost between consecutive M points
        marginal = {}
        for i in range(1, len(Ms)):
            a, b = Ms[i - 1], Ms[i]
            marginal[b] = (lat[b] - lat[a]) / (b - a)
        # Knee M*: the deepest M for which the verify step is still "nearly free",
        # i.e. latency(M) <= latency(M=1) * (1 + knee_tol). At conc=1 the step is
        # weight-bandwidth bound, so latency is ~flat (often sub-noise, occasionally
        # negative marginal) until compute starts to bite; a marginal-ratio rule is
        # not robust there. This rule reports the last M before per-step cost rises
        # materially over the single-token cost -> "propose up to K* = M*-1".
        knee_tol = 0.10
        l1 = lat[Ms[0]]
        knee = Ms[0]
        for M in Ms:
            if lat[M] <= l1 * (1 + knee_tol):
                knee = M
            else:
                break
        # Ideal TPS(K) ceiling, two conventions:
        #  - ideal (PR formula):  K     / latency(M=K+1)  -> counts only the K
        #    drafted tokens emitted when all accepted.
        #  - ideal_bonus (true):  (K+1) / latency(M=K+1)  == M / latency(M)  -> the
        #    real all-accepted ceiling, which also emits the 1 bonus token at the
        #    last verified position. This is the valid UPPER BOUND on the realistic
        #    curves below (the geometric model includes the bonus via 1-p^(K+1)),
        #    so it is the honest "max TPS any drafter can reach" headline.
        ideal = {}
        ideal_bonus = {}
        for M in Ms:
            K = M - 1
            if K >= 1:
                ideal[K] = K / (lat[M] / 1000.0)
                ideal_bonus[K] = (K + 1) / (lat[M] / 1000.0)
        ideal_kstar = max(ideal, key=ideal.get) if ideal else None
        ideal_bonus_kstar = max(ideal_bonus, key=ideal_bonus.get) if ideal_bonus else None
        # realistic TPS per acceptance model
        realistic = {}
        for label, spec in accept_models.items():
            tps_by_K = {}
            for M in Ms:
                K = M - 1
                if K >= 1:
                    tps_by_K[K] = expected_accepted(spec, K) / (lat[M] / 1000.0)
            if tps_by_K:
                kstar = max(tps_by_K, key=tps_by_K.get)
                realistic[label] = {
                    "tps_by_K": tps_by_K,
                    "K_star": kstar,
                    "tps_at_Kstar": tps_by_K[kstar],
                }
        # attention-share crossings (% of full step GPU-busy)
        attn_cross = {"gt5pct_at_M": None, "gt10pct_at_M": None, "gt15pct_at_M": None}
        for M in Ms:
            a = mrows[M]["step_category_pct"].get("attention", 0.0)
            if attn_cross["gt5pct_at_M"] is None and a > 5:
                attn_cross["gt5pct_at_M"] = M
            if attn_cross["gt10pct_at_M"] is None and a > 10:
                attn_cross["gt10pct_at_M"] = M
            if attn_cross["gt15pct_at_M"] is None and a > 15:
                attn_cross["gt15pct_at_M"] = M
        out[f"{mode}|ctx{ctx}"] = {
            "mode": mode, "ctx": ctx,
            "latency_ms_by_M": lat,
            "t_forward_ms_by_M": {M: mrows[M]["t_forward_ms"] for M in Ms},
            "t_lmhead_ms_by_M": {M: mrows[M]["t_lmhead_ms"] for M in Ms},
            "lmhead_share_by_M": {M: mrows[M]["t_lmhead_ms"] / lat[M] for M in Ms},
            "attention_pct_step_by_M": {
                M: mrows[M]["step_category_pct"].get("attention", 0.0) for M in Ms},
            "gemm_pct_step_by_M": {
                M: mrows[M]["step_category_pct"].get("matmul_gemm", 0.0) for M in Ms},
            "marginal_ms_per_token_by_M": marginal,
            "knee_Mstar": knee,
            "ideal_tps_by_K": ideal,
            "ideal_K_star": ideal_kstar,
            "ideal_tps_at_Kstar": ideal.get(ideal_kstar) if ideal_kstar else None,
            "ideal_tps_bonus_by_K": ideal_bonus,
            "ideal_bonus_K_star": ideal_bonus_kstar,
            "ideal_tps_bonus_at_Kstar": ideal_bonus.get(ideal_bonus_kstar) if ideal_bonus_kstar else None,
            "realistic": realistic,
            "attention_share_crossings": attn_cross,
        }
    return out


def parse_accept_models(s):
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        kind, val = tok.split(":")
        out[tok] = (kind, float(val))
    return out


def _build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4-base", default=DEFAULT_MODEL)
    ap.add_argument("--m-sweep", default="1,2,4,6,8,10,12,16")
    ap.add_argument("--ctx-sweep", default="256,512")
    ap.add_argument("--modes", default="eager,graph")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--profile-steps", type=int, default=30)
    ap.add_argument("--accept-models",
                    default="flat:2.2,flat:3.3,flat:4.5,geom:0.6,geom:0.7,geom:0.8")
    ap.add_argument("--output", default="research/spec_cost_model/results.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="spec-cost-model")
    ap.add_argument("--wandb_name", default="spec-cost-model-int4")
    ap.add_argument("--no-wandb", action="store_true")
    # Internal worker flags: run ONE mode in an isolated process and dump a
    # partial JSON. vLLM V1 in-process does not release GPU memory on `del llm`,
    # so two LLM() instances cannot coexist; running each mode in its own
    # subprocess lets process exit free the GPU between modes.
    ap.add_argument("--single-mode", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--partial-out", default=None, help=argparse.SUPPRESS)
    # PR #51: re-ground the verify cost curve on the MERGED #43 split-KV patch.
    # Installs splitkv_verify_patch.install() after the engine is built so the
    # verify-attention op (1<M<=64 query rows) routes through vLLM's 3D split-KV
    # (FlashDecoding) path -- the only thing #43 changes. The patch's redirect
    # counter is recorded in the output so "patch active" is provable, not assumed.
    ap.add_argument("--splitkv-patch", default=None,
                    help="path to splitkv_verify_patch.py; install it before the "
                         "M-sweep so the verify forward uses the #43 3D split-KV path")
    # Tree-causal mask attention microbenchmark (PR #33). Independent of the
    # M-sweep orchestrator: isolates the only mask-addressable term (core
    # QK^T-softmax-V) at the real Gemma-4-E4B decoder geometry and measures the
    # dense-causal vs sparse tree-causal saving, then folds it into the merged
    # PR #28 dense t_step curve.
    ap.add_argument("--tree-mask", action="store_true",
                    help="run the tree-causal mask attention microbenchmark and exit")
    ap.add_argument("--tree-M", default="25,33,49",
                    help="verify-position counts M=K*W+1 to profile (tree shapes)")
    ap.add_argument("--tree-W", type=int, default=4, help="tree width W")
    ap.add_argument("--tree-ctx", type=int, default=256, help="shared KV context length")
    ap.add_argument("--tree-iters", type=int, default=300, help="timed attn iters")
    ap.add_argument("--tree-warmup", type=int, default=50, help="warmup attn iters")
    ap.add_argument("--dense-curve-json",
                    default="research/spec_cost_model/results_msweep.json",
                    help="PR #28 merged curve for dense t_step(M) + attention share")
    ap.add_argument("--tree-output",
                    default="research/spec_cost_model/results_tree_mask.json")
    return ap


def _run_worker(args):
    """Worker: run exactly ONE mode, write a partial JSON, exit (frees the GPU)."""
    mode = args.single_mode
    m_sweep = [int(x) for x in args.m_sweep.split(",")]
    ctx_sweep = [int(x) for x in args.ctx_sweep.split(",")]
    device = torch.cuda.get_device_name(0)
    print(f"[cost] WORKER mode={mode} device={device} torch={torch.__version__} "
          f"M={m_sweep} ctx={ctx_sweep} steps={args.steps} warmup={args.warmup}", flush=True)
    rows, hidden, splitkv_info = run_mode(
        args.int4_base, mode, m_sweep, ctx_sweep,
        args.steps, args.warmup, args.profile_steps,
        splitkv_patch=args.splitkv_patch)
    with open(args.partial_out, "w") as f:
        json.dump({"mode": mode, "rows": rows, "hidden": hidden, "device": device,
                   "splitkv_info": splitkv_info}, f)
    print(f"[cost] WORKER mode={mode} wrote {args.partial_out} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Tree-causal mask attention microbenchmark (PR #33)
# ---------------------------------------------------------------------------
# The verify-step latency t_step(M) = QKV/O/MLP GEMM (mask-INDEPENDENT, processes
# all M tokens through the int4 weights) + core attention softmax(QK^T+mask)V
# (the ONLY mask-addressable term) + RoPE/KV-write (O(M), mask-independent) +
# lm_head (O(M), mask-independent). A width-W tree-causal mask shrinks ONLY the
# among-token block of core attention (M*(M+1)/2 dense-causal pairs -> sum of
# root-to-node path lengths), leaving the M*ctx cross-context block and every
# GEMM untouched.
#
# vLLM's paged-FlashAttention `_dummy_run` path cannot consume an arbitrary mask
# (PR #28 report sec.5), and injecting one would be fragile enough to invalidate
# the measurement. So we ISOLATE the mask-addressable term in a standalone
# attention microbenchmark at the real Gemma-4-E4B text-decoder geometry and
# fold the measured saving into PR #28's dense t_step curve. Three lenses:
#   - SDPA + boolean mask: the PRODUCTION-realistic path (SpecInfer/EAGLE/vLLM
#     tree verify is dense attention + topology mask; saving is ~0 by
#     construction because the dense score matrix is materialised regardless).
#   - FlexAttention block-sparse (best-effort): the only kernel that *could*
#     skip masked work, but its 128-token block granularity is coarser than the
#     M<=49 tree sparsity, so realised saving is also ~0.
#   - analytic FLOP-ideal: the unrealisable theoretical ceiling (perfect
#     element-sparse kernel) = dense_core_attn * (1 - tree_pairs/dense_pairs).
DEFAULT_ATTN_GEOM = dict(layers=42, hq=8, hkv=2, head_dim=256)  # gemma-4-E4B text


def _load_attn_geometry(model: str) -> dict:
    """(layers, q-heads, kv-heads, head_dim) of the TEXT decoder (the only stack
    the verify forward runs). Falls back to known gemma-4-E4B values."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        g = dict(
            layers=int(tc.num_hidden_layers),
            hq=int(tc.num_attention_heads),
            hkv=int(getattr(tc, "num_key_value_heads", tc.num_attention_heads)),
            head_dim=int(getattr(tc, "head_dim",
                                 tc.hidden_size // tc.num_attention_heads)),
            sliding_window=int(getattr(tc, "sliding_window", 0) or 0),
        )
        return g
    except Exception as e:  # noqa: BLE001
        print(f"[tree-mask] AutoConfig failed ({e!r}); using gemma-4-E4B defaults",
              flush=True)
        return dict(**DEFAULT_ATTN_GEOM, sliding_window=512)


def _tree_among(M: int, W: int):
    """Among-token tree-causal structure for a balanced W-ary tree (parent of
    node i is (i-1)//W). Returns (adj[M,M] bool, parent list, depth list,
    pair_count) where adj[i,j]=True iff j is an ancestor of i (incl. self+root),
    exactly the PR #33 mask construction."""
    parent = [-1] * M
    for i in range(1, M):
        parent[i] = (i - 1) // W
    adj = np.zeros((M, M), dtype=bool)
    for i in range(M):
        cur = i
        while cur > 0:
            adj[i, cur] = True
            cur = parent[cur]
        adj[i, 0] = True
    depth = [0] * M
    for i in range(1, M):
        depth[i] = depth[parent[i]] + 1
    return adj, parent, depth, int(adj.sum())


def _curve_interp(path: str, key: str = "graph|ctx256"):
    """Return (lat_at, attn_pct_at) piecewise-linear interpolators over the
    merged PR #28 dense curve, plus the raw dicts."""
    d = json.load(open(path))
    node = d["cost_model"][key]
    lat = {int(k): float(v) for k, v in node["latency_ms_by_M"].items()}
    apc = {int(k): float(v) for k, v in node["attention_pct_step_by_M"].items()}

    def mk(table):
        xs = sorted(table)

        def at(M):
            if M <= xs[0]:
                return table[xs[0]]
            if M >= xs[-1]:
                return table[xs[-1]]
            lo = max(x for x in xs if x <= M)
            hi = min(x for x in xs if x >= M)
            if lo == hi:
                return table[lo]
            t = (M - lo) / (hi - lo)
            return table[lo] * (1 - t) + table[hi] * t
        return at
    return mk(lat), mk(apc), lat, apc


def _time_gpu(fn, iters: int, warmup: int) -> float:
    """Median per-call ms, pipelined (CPU runs ahead, single final sync) — the
    same throughput basis as the main profiler's _pipelined()."""
    import torch as T
    for _ in range(warmup):
        fn()
    T.cuda.synchronize()
    e0 = [T.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [T.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        fn()
        e1[i].record()
    T.cuda.synchronize()
    return statistics.median([e0[i].elapsed_time(e1[i]) for i in range(iters)])


def _tree_mask_main(args):
    import torch as T
    import torch.nn.functional as Fn
    dev = "cuda"
    geom = _load_attn_geometry(args.int4_base)
    L, Hq, Hkv, D = geom["layers"], geom["hq"], geom["hkv"], geom["head_dim"]
    ctx, W = args.tree_ctx, args.tree_W
    Ms = [int(x) for x in args.tree_M.split(",")]
    lat_at, apc_at, lat_raw, apc_raw = _curve_interp(args.dense_curve_json)
    sw = geom.get("sliding_window", 0)

    print(f"[tree-mask] geom: L={L} Hq={Hq} Hkv={Hkv} D={D} sliding_window={sw}",
          flush=True)
    print(f"[tree-mask] ctx={ctx} W={W} M={Ms} | iters={args.tree_iters} "
          f"warmup={args.tree_warmup}", flush=True)
    if sw and ctx + max(Ms) >= sw:
        print(f"[tree-mask] WARNING ctx+maxM={ctx+max(Ms)} >= sliding_window={sw}: "
              f"local layers truncate ctx (delta still valid; ctx is common-mode)",
              flush=True)
    else:
        print(f"[tree-mask] ctx+maxM={ctx+max(Ms)} < sliding_window={sw}: all {L} "
              f"layers attend full ctx+M (uniform geometry)", flush=True)

    # FlexAttention (best-effort): the only kernel that could skip masked blocks.
    flex = None
    try:
        from torch.nn.attention.flex_attention import (
            flex_attention, create_block_mask)
        flex = torch.compile(flex_attention, dynamic=False)
        _flex_create = create_block_mask
    except Exception as e:  # noqa: BLE001
        print(f"[tree-mask] FlexAttention unavailable ({e!r}); SDPA + analytic only",
              flush=True)

    rows = []
    for M in Ms:
        K = (M - 1) // W
        adj_np, parent, depth, tree_pairs = _tree_among(M, W)
        dense_pairs = M * (M + 1) // 2
        ctx_pairs = M * ctx
        dense_total = ctx_pairs + dense_pairs
        tree_total = ctx_pairs + tree_pairs
        flop_ratio = tree_total / dense_total           # tree / dense core-attn FLOPs
        ideal_save_frac = 1.0 - flop_ratio              # fraction of core attn removed

        # Random bf16 Q/K/V; batch dim = #layers so one kernel call ~ one verify
        # step's total core attention (graph-mode amortises per-layer launches).
        g = T.Generator(device=dev).manual_seed(0)
        q = T.randn(L, Hq, M, D, dtype=T.bfloat16, device=dev, generator=g)
        k = T.randn(L, Hkv, ctx + M, D, dtype=T.bfloat16, device=dev, generator=g)
        v = T.randn(L, Hkv, ctx + M, D, dtype=T.bfloat16, device=dev, generator=g)
        # Manual GQA replication (avoids the enable_gqa flex gotcha; identical
        # math for both paths so the delta is clean).
        rep = Hq // Hkv
        kr = k.repeat_interleave(rep, dim=1)
        vr = v.repeat_interleave(rep, dim=1)

        # Boolean masks [M, ctx+M] (True = attend): ctx fully visible; among-M is
        # causal (dense) vs ancestor-only (tree).
        among_idx = T.arange(ctx + M, device=dev) - ctx          # key's among-index
        in_ctx = (T.arange(ctx + M, device=dev) < ctx)           # [ctx+M]
        qi = T.arange(M, device=dev)
        dense_among = (among_idx[None, :] <= qi[:, None]) & (~in_ctx)[None, :]
        dense_mask = in_ctx[None, :] | dense_among               # [M, ctx+M]
        adj_t = T.from_numpy(adj_np).to(dev)
        amax = adj_t[:, T.clamp(among_idx, 0, M - 1)]            # [M, ctx+M]
        tree_among = amax & (~in_ctx)[None, :]
        tree_mask = in_ctx[None, :] | tree_among                 # [M, ctx+M]
        dm = dense_mask[None, None]
        tm = tree_mask[None, None]

        def sdpa(mask):
            return Fn.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)

        sdpa_dense = _time_gpu(lambda: sdpa(dm), args.tree_iters, args.tree_warmup)
        sdpa_tree = _time_gpu(lambda: sdpa(tm), args.tree_iters, args.tree_warmup)

        flex_dense = flex_tree = None
        if flex is not None:
            try:
                def dense_mod(b, h, qd, kv):
                    return (kv < ctx) | ((kv >= ctx) & ((kv - ctx) <= qd))

                def tree_mod(b, h, qd, kv):
                    j = T.clamp(kv - ctx, 0, M - 1)
                    return (kv < ctx) | ((kv >= ctx) & adj_t[qd, j])
                bm_d = _flex_create(dense_mod, None, None, M, ctx + M, device=dev)
                bm_t = _flex_create(tree_mod, None, None, M, ctx + M, device=dev)
                flex_dense = _time_gpu(
                    lambda: flex(q, kr, vr, block_mask=bm_d),
                    args.tree_iters, args.tree_warmup)
                flex_tree = _time_gpu(
                    lambda: flex(q, kr, vr, block_mask=bm_t),
                    args.tree_iters, args.tree_warmup)
            except Exception as e:  # noqa: BLE001
                print(f"[tree-mask] M={M} FlexAttention failed ({e!r}); skipping",
                      flush=True)
                flex_dense = flex_tree = None

        # Fold into PR #28 dense t_step. delta_sdpa is the production-realistic
        # saving; delta_flex the best-effort block-sparse saving; delta_flopideal
        # the unrealisable ceiling (calibrated against the vLLM attention bucket
        # so it never under-states the saving the GEMM-bound ramp could hide).
        t_step_dense = lat_at(M)
        attn_bucket_ms = (apc_at(M) / 100.0) * t_step_dense     # incl RoPE/KV-write
        delta_sdpa = sdpa_dense - sdpa_tree
        delta_flex = (flex_dense - flex_tree) if flex_dense is not None else None
        delta_flopideal_vllm = ideal_save_frac * attn_bucket_ms  # bucket >= core
        delta_flopideal_sdpa = ideal_save_frac * sdpa_dense      # measured dense core
        row = {
            "M": M, "K": K, "W": W, "ctx": ctx, "depth_max": max(depth),
            "dense_pairs_amongM": dense_pairs, "tree_pairs_amongM": tree_pairs,
            "ctx_pairs": ctx_pairs, "flop_ratio_tree_over_dense": flop_ratio,
            "ideal_save_frac_of_core_attn": ideal_save_frac,
            "t_step_dense_ms": t_step_dense,
            "attn_bucket_pct": apc_at(M), "attn_bucket_ms": attn_bucket_ms,
            "sdpa_dense_core_ms": sdpa_dense, "sdpa_tree_core_ms": sdpa_tree,
            "delta_sdpa_ms": delta_sdpa,
            "flex_dense_core_ms": flex_dense, "flex_tree_core_ms": flex_tree,
            "delta_flex_ms": delta_flex,
            "delta_flopideal_vllmcal_ms": delta_flopideal_vllm,
            "delta_flopideal_sdpacal_ms": delta_flopideal_sdpa,
            # tree-masked t_step under each lens (saving subtracted from dense)
            "t_step_tree_sdpa_ms": t_step_dense - max(0.0, delta_sdpa),
            "t_step_tree_flex_ms": (t_step_dense - max(0.0, delta_flex))
                                   if delta_flex is not None else None,
            "t_step_tree_flopideal_ms": t_step_dense - delta_flopideal_vllm,
        }
        rows.append(row)
        fx = f"{delta_flex:+.4f}" if delta_flex is not None else "n/a"
        print(f"[tree-mask] M={M:2d} K={K:2d}: dense_pairs={dense_pairs} "
              f"tree_pairs={tree_pairs} ideal_save={ideal_save_frac*100:4.1f}%core | "
              f"t_step_dense={t_step_dense:6.3f}ms attn_bucket={attn_bucket_ms:5.3f}ms | "
              f"sdpa d/t={sdpa_dense:.4f}/{sdpa_tree:.4f} (Δ{delta_sdpa:+.4f}) "
              f"flexΔ={fx} | FLOP-ideal Δ={delta_flopideal_vllm:.4f}ms "
              f"-> tree_masked t_step(ideal)={row['t_step_tree_flopideal_ms']:.3f}ms",
              flush=True)
        del q, k, v, kr, vr
        gc.collect()
        T.cuda.empty_cache()

    payload = {
        "config": {
            "model": args.int4_base, "geometry": geom, "ctx": ctx, "W": W,
            "tree_M": Ms, "iters": args.tree_iters, "warmup": args.tree_warmup,
            "dense_curve_json": args.dense_curve_json,
            "device": T.cuda.get_device_name(0),
            "note": ("component microbench of the mask-addressable core-attention "
                     "term; GEMM/RoPE/KV-write/lm_head are mask-independent and "
                     "taken from the PR #28 dense curve"),
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.tree_output) or ".", exist_ok=True)
    with open(args.tree_output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[tree-mask] wrote {args.tree_output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb_treemask(args, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[tree-mask] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[tree-mask] DONE", flush=True)


def _log_wandb_treemask(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name or "tree-mask-attn",
                     job_type="profiling", config=payload["config"])
    cols = ["M", "K", "dense_pairs_amongM", "tree_pairs_amongM",
            "ideal_save_frac_of_core_attn", "t_step_dense_ms", "attn_bucket_ms",
            "sdpa_dense_core_ms", "sdpa_tree_core_ms", "delta_sdpa_ms",
            "flex_dense_core_ms", "flex_tree_core_ms", "delta_flex_ms",
            "delta_flopideal_vllmcal_ms", "t_step_tree_flopideal_ms"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"tree_mask_table": tbl})
    summary = {}
    for r in payload["rows"]:
        tag = f"M{r['M']}"
        summary[f"delta_sdpa_ms_{tag}"] = r["delta_sdpa_ms"]
        summary[f"delta_flopideal_ms_{tag}"] = r["delta_flopideal_vllmcal_ms"]
        summary[f"ideal_save_frac_core_{tag}"] = r["ideal_save_frac_of_core_attn"]
        summary[f"t_step_tree_flopideal_ms_{tag}"] = r["t_step_tree_flopideal_ms"]
        if r["delta_flex_ms"] is not None:
            summary[f"delta_flex_ms_{tag}"] = r["delta_flex_ms"]
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    run.finish()
    print(f"[tree-mask] W&B run: {run.url}", flush=True)


def main():
    args = _build_argparser().parse_args()

    # Tree-causal mask attention microbenchmark (PR #33), independent path.
    if args.tree_mask:
        _tree_mask_main(args)
        return

    # Worker path: a single isolated mode (spawned by the orchestrator below).
    if args.single_mode:
        _run_worker(args)
        return

    # Orchestrator path: CUDA-free. Spawn one worker subprocess per mode so the
    # GPU is fully released between modes, then merge partials and build the model.
    m_sweep = [int(x) for x in args.m_sweep.split(",")]
    ctx_sweep = [int(x) for x in args.ctx_sweep.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    accept_models = parse_accept_models(args.accept_models)

    print(f"[cost] orchestrator model={args.int4_base} M={m_sweep} ctx={ctx_sweep} "
          f"modes={modes} steps={args.steps} warmup={args.warmup}", flush=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_rows, hidden, device = [], None, None
    splitkv_info: dict = {}
    for mode in modes:
        fd, partial_path = tempfile.mkstemp(prefix=f"spec_cost_{mode}_", suffix=".json")
        os.close(fd)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--single-mode", mode, "--partial-out", partial_path,
               "--int4-base", args.int4_base,
               "--m-sweep", args.m_sweep, "--ctx-sweep", args.ctx_sweep,
               "--steps", str(args.steps), "--warmup", str(args.warmup),
               "--profile-steps", str(args.profile_steps), "--no-wandb"]
        if args.splitkv_patch:
            cmd += ["--splitkv-patch", args.splitkv_patch]
        print(f"[cost] launching worker for mode={mode}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"worker mode={mode} failed with exit code {rc}")
        with open(partial_path) as f:
            part = json.load(f)
        os.unlink(partial_path)
        all_rows.extend(part["rows"])
        hidden = hidden or part.get("hidden")
        device = device or part.get("device")
        if part.get("splitkv_info"):
            splitkv_info[mode] = part["splitkv_info"]
        # incremental save so a later-mode crash can't lose earlier data
        _save(args, m_sweep, ctx_sweep, accept_models, all_rows, hidden, device=device,
              splitkv_info=splitkv_info or None)

    cost_model = build_cost_model(all_rows, accept_models)
    payload = _save(args, m_sweep, ctx_sweep, accept_models, all_rows, hidden,
                    device=device, cost_model=cost_model,
                    splitkv_info=splitkv_info or None)

    # Calibration line: graph M=1 ctx=256 should reproduce ~96.89 TPS (~10.3 ms).
    for r in all_rows:
        if r["mode"] == "graph" and r["M"] == 1 and r["ctx"] == 256:
            tps = 1.0 / (r["t_step_ms"] / 1000.0)
            print(f"\n[cost] CALIBRATION graph M=1 ctx=256: {r['t_step_ms']:.3f} ms "
                  f"=> {tps:.2f} TPS (PR#7 ref 96.89)", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as e:
            print(f"[cost] W&B logging failed (non-fatal): {e}", flush=True)

    print(f"\n[cost] wrote {args.output}", flush=True)
    print("[cost] DONE", flush=True)


def _save(args, m_sweep, ctx_sweep, accept_models, rows, hidden, device=None,
          cost_model=None, splitkv_info=None):
    payload = {
        "config": {
            "model": args.int4_base, "m_sweep": m_sweep, "ctx_sweep": ctx_sweep,
            "steps": args.steps, "warmup": args.warmup,
            "accept_models": {k: list(v) for k, v in accept_models.items()},
            "device": device or "unknown", "hidden_size": hidden,
            "env": {k: os.environ.get(k) for k in
                    ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ENABLE_V1_MULTIPROCESSING",
                     "CUDA_VISIBLE_DEVICES", "SPLITKV_VERIFY", "SPLITKV_VERIFY_MAX_Q")},
            "splitkv_info": splitkv_info,
        },
        "rows": rows,
    }
    if cost_model is not None:
        payload["cost_model"] = cost_model
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    # Per-config table
    cols = ["mode", "ctx", "M", "t_step_ms", "t_forward_ms", "t_lmhead_ms",
            "attn_pct_step", "gemm_pct_step", "lmhead_pct_step"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        sc = r.get("step_category_pct", {})
        tbl.add_data(r["mode"], r["ctx"], r["M"], r["t_step_ms"], r["t_forward_ms"],
                     r["t_lmhead_ms"], sc.get("attention", 0.0),
                     sc.get("matmul_gemm", 0.0), sc.get("sampling_lmhead", 0.0))
    run.log({"cost_table": tbl})
    # Headline summary metrics
    cm = payload.get("cost_model", {})
    summary = {}
    g = cm.get("graph|ctx256") or next(iter(cm.values()), None)
    if g:
        summary["tps_ceiling_ideal_at_kstar"] = g.get("ideal_tps_at_Kstar")
        summary["ideal_K_star"] = g.get("ideal_K_star")
        summary["tps_ceiling_bonus_at_kstar"] = g.get("ideal_tps_bonus_at_Kstar")
        summary["ideal_bonus_K_star"] = g.get("ideal_bonus_K_star")
        summary["knee_Mstar"] = g.get("knee_Mstar")
        summary["t_step_ms_M1"] = g.get("latency_ms_by_M", {}).get(1)
        if g.get("latency_ms_by_M", {}).get(1):
            summary["calib_tps_M1"] = 1000.0 / g["latency_ms_by_M"][1]
        for label in ("geom:0.7", "geom:0.6", "geom:0.8", "flat:3.3", "flat:4.5"):
            r = g.get("realistic", {}).get(label, {})
            if r:
                summary[f"optimal_k_{label}"] = r.get("K_star")
                summary[f"tps_real_{label}_at_kstar"] = r.get("tps_at_Kstar")
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    # Line series: ideal TPS vs K for graph ctx256
    if g and g.get("ideal_tps_by_K"):
        for K, tps in sorted(g["ideal_tps_by_K"].items(), key=lambda x: int(x[0])):
            run.log({"K": int(K), "ideal_tps": tps})
    run.finish()
    print(f"[cost] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
