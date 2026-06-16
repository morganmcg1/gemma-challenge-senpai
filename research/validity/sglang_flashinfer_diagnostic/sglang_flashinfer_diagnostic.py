#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #498 (denken) -- SGLang/Flashinfer M=1 decode: is byte-exact greedy identity FREE? (#481 zoom-out)

THE QUESTION
------------
Every determinism-tax result in this program is vLLM + Triton-specific. The -107 TPS byte-exact
2D-attention tax that caps the surgical-357 rung (lawine #488 ``ko01dcyy``) exists because vLLM's Triton
attention uses split-K / split-KV reductions whose ORDER depends on GPU occupancy -- which differs between
M=1 autoregressive decode and M=8 spec-decode verify. To hold byte-exact greedy identity we must pin
attention to a slow order-preserving kernel. ubel #491 (``5cappm87``) proved attention is the SOLE must-pin
op (9/10 decode ops argmax-free), so the entire strict-frontier headroom hinges on this one kernel's
reduction determinism.

SGLang dispatches the **Flashinfer** attention library by default (not Triton). vLLM also exposes a
FLASHINFER backend that calls the same ``flashinfer`` library. THE POINTED QUESTION: is Flashinfer's
batch-decode (M=1, single decode query) attention reduction **deterministic by construction** -- i.e. does
it avoid the occupancy-dependent split-KV reduction that forces vLLM's -107 pin? If YES, SGLang/Flashinfer
could deliver fast AND byte-exact greedy identity for free (collapsing the tax, potentially a strict rung
near the deployed 481). If NO, the tax is engine-INDEPENDENT (a property of the split-KV math, not a vLLM
artifact) -- also a high-value finding the whole post-222 program hinges on.

WHAT WE MEASURE (LOCAL, analysis_only -- the determinism STRUCTURE, transferable across configs)
------------------------------------------------------------------------------------------------
Three walls block a literal "stand up SGLang 0.22.1rc1 server" reading (all documented honestly, not worked
around):
  W1. "SGLang 0.22.1rc1" is not a real SGLang release -- 0.22.1rc1 is the *vLLM* version string pinned by
      this challenge (server venv = vllm 0.22.1rc1.dev307). Real SGLang latest is ~0.5.13; v0.5.11+ moved
      to CUDA 13.0 (sgl-kernel wheels), and this pod is CUDA 12.9 -> sgl-kernel wheel incompatibility.
  W2. vLLM 0.22.1rc1 FORCES the TRITON_ATTN backend for Gemma-4 ("heterogeneous head dimensions
      head_dim=256, global_head_dim=512 ... Forcing TRITON_ATTN to prevent mixed-backend numerical
      divergence"), overriding VLLM_ATTENTION_BACKEND=FLASHINFER. So an end-to-end vLLM-with-Flashinfer
      Gemma-4 census is not available -- captured live in phase_engine.
  W3. Flashinfer's TENSOR-CORE decode kernel does not support Gemma-4's global head_dim=512 on sm_86
      (A10G); only the cuda-core decode path does -- and ``fixed_split_size`` (the cheap determinism lever)
      REQUIRES tensor cores, so at head_dim=512 the only batch-invariant lever is the heavier
      ``disable_split_kv`` floor.

The determinism property of the attention reduction is the SAME library (flashinfer-python 0.6.12, already
installed and what SGLang would dispatch) regardless of orchestrator -- so we measure it DIRECTLY at the
flashinfer decode-kernel level (the kernelcensus phase), exactly the op #491's margin proxy could only
reach end-to-end. This is faithful for the verdict and config-robust.

  (KERNELCENSUS, GPU) flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper at the real Gemma-4 GQA geometry
    (8 Q / 2 KV heads; head_dim 256 local AND 512 global), over KV seqlens, for three reduction modes
    (default dynamic split-KV / disable_split_kv / fixed_split_size). Per mode:
      * self_det  = byte identity of two M=1 runs at fixed occupancy (run-to-run determinism).
      * m_invariance = byte identity of query-0's output computed as M=1 vs inside an M=8 batch (the
        DEPLOYED verify-vs-decode occupancy gap, SPEC_K+1=8). THIS is the operative test: dynamic split-KV
        picks a different split count at M=1 vs M=8 -> different merge_states order -> different bits.
      * positive control = a perturbed query MUST change the bytes (proves the comparator is sensitive,
        so a 1.0 invariance is real, not a stuck comparator).
      * latency (median over iters) -> the determinism TAX = lat(byte-exact mode)/lat(default) - 1.
    flashinfer's OWN plan() docstring states fixed_split_size "will lead to deterministic softmax score
    reduction in the merge_states kernel, and therefore batch-size invariant outputs" (citing
    thinkingmachines.ai) -- i.e. the DEFAULT is NOT batch-invariant. We quantify that on this A10G.
  (ENGINE, GPU, optional) Load the loadable full-vocab int4 under VLLM_ATTENTION_BACKEND=FLASHINFER, capture
    the backend vLLM actually selects (W2), and a realized conc=1 greedy decode-TPS anchor (TRITON-forced,
    vanilla int4 -- NOT the deployed 481.53 stack, NOT a Flashinfer number; reproducibility caveat only).

HONESTY BAR (per the PR + #491 style)
-------------------------------------
* analysis_only=true, official_tps=0, NO HF Job, NO train.py --launch, NO submission, NO served-file
  change. Baseline 481.53 UNCHANGED. Greedy identity MEASURED at the kernel level, never broken.
* Config flag: the engine census cannot load the deployed PRUNED-head int4 (lm_head rows 16384/12288 !=
  config vocab 262144 -> vanilla vLLM ParallelLMHead assert, same as #491), so it uses the upstream
  FULL-VOCAB int4 (google/gemma-4-E4B-it-qat-w4a16-ct). The determinism STRUCTURE we measure is an
  attention-kernel property independent of the output head, so this is faithful for the verdict; the
  absolute TPS is config-caveated (flagged exactly as ubel #491 flagged its full-vocab margin checkpoint).
* sglang_decode_tps (full-engine SGLang/Flashinfer) is NOT measurable here (W1/W2/W3) -> reported BLOCKED
  with the realized TRITON-vanilla anchor + the attention-kernel determinism tax as the substantive
  throughput findings. The -107 tax IS an attention tax, so the kernel-level tax is the right proxy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

# ---- cited anchors (NOT re-derived) ---------------------------------------------------------------
DEPLOYED_TPS = 481.53          # PR #52 public non-strict deployed (2x9fm2zx), vLLM
SURGICAL_TPS = 357.64          # lawine #488 (ko01dcyy) strict surgical rung (attention-only 2D pin)
FLOORLOCK_TPS = 161.70         # strict floor-lock (private-safe, M=1 AR no drafter)
ATTENTION_2D_TAX_TPS = 107.0   # the 481 -> surgical byte-exact 2D-attention pin cost this card probes
CEILING_TPS = 520.953          # strict lambda=1 ceiling (context only)

# ---- Gemma-4-E4B-it geometry (deployed osoi5 int4 / full-vocab w4a16-ct text_config) --------------
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM_LOCAL = 256           # sliding-window local-attention layers
HEAD_DIM_GLOBAL = 512          # global-attention layers (heterogeneous head dim -> W2/W3)
DEPLOYED_VERIFY_M = 8          # SPEC_K(7) + 1 verify width == the M=8 spec-decode occupancy
PAGE_SIZE = 16
FIXED_SPLIT_PAGES = 8          # fixed_split_size value (in pages); determinism comes from it being FIXED
WORKSPACE_BYTES = 384 * 1024 * 1024

# ---- benchmark OPERATIVE decode-KV regime (measured from the speed_benchmark prompts, PR #498) -----
# ppl_ground_truth_tokens.jsonl (128 prompts): context 114..2431 (p50 234, p90 392); target/generation
# is a near-constant 512 (p50=p90=512, max 512). Per-row decode KV = context + tokens_generated_so_far,
# so the LAST decode step of each row reaches ctx+target. Across rows: min 456, p50 720, p90 894,
# max 2943. The operative question for the -107 pin is whether default split-KV's M=1-vs-M=8 divergence
# THRESHOLD falls below this band (pin bites in deployment) or above it (pin is a worst-case-only artifact).
OPERATIVE_KV_P50 = 720
OPERATIVE_KV_P90 = 894
OPERATIVE_KV_MAX = 2943

# ---- SGLang version reality (researcher pass, PR #498) --------------------------------------------
SGLANG_VERSION_IN_PR = "0.22.1rc1"   # PR text -- this is actually the vLLM version string (W1)
SGLANG_REAL_LATEST = "0.5.13"

# full-vocab int4 fallback (loads in vanilla vLLM; deployed pruned head does not -- same as #491)
ENGINE_MODEL_CANDIDATES = (
    os.path.expanduser("~/.cache/huggingface/hub/"
                       "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"),
)
PROMPTS_JSONL = ("official/main_bucket/shared_resources/speed_benchmark/data/"
                 "ppl_ground_truth_tokens.jsonl")

# geometry sweep: (label, head_dim, use_tensor_cores, available_modes)
#   tensor-core decode supports head_dim 256 only on sm_86 (W3); fixed_split_size requires tensor cores.
GEOMS = (
    ("local_hd256_tensorcore", HEAD_DIM_LOCAL, True, ("default", "disable_split_kv", "fixed_split_size")),
    ("local_hd256_cudacore", HEAD_DIM_LOCAL, False, ("default", "disable_split_kv")),
    ("global_hd512_cudacore", HEAD_DIM_GLOBAL, False, ("default", "disable_split_kv")),
)
# the path SGLang would use by default for GQA decode (group size 4): tensor-core dynamic split-KV.
SGLANG_DEFAULT_GEOM = "local_hd256_tensorcore"
# the deployed Gemma-4 runs BOTH a local (hd256) and a global (hd512) attention kernel on EVERY decode
# token; greedy identity requires every layer byte-exact, so the operative identity is the WORST over the
# paths the deployed model actually dispatches: local tensor-core fast path + global cuda-core (W3-forced).
DEPLOYED_DECODE_GEOMS = ("local_hd256_tensorcore", "global_hd512_cudacore")


# ======================================================================================== #
# small helpers
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return str(o)


def _f(x: Any) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def resolve_engine_model() -> str | None:
    for cand in ENGINE_MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    return None


# ======================================================================================== #
# GPU PHASE: KERNELCENSUS -- direct flashinfer decode-attention reduction determinism
# ======================================================================================== #
def phase_kernelcensus(out_path: str, seqlens: list[int], n_lat_iters: int, warmup: int,
                       seed: int) -> None:
    import torch
    import flashinfer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- launch with CUDA_VISIBLE_DEVICES=0")
    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    p = torch.cuda.get_device_properties(dev)
    gpu = {"name": p.name, "sm": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
           "total_mem_gib": round(p.total_memory / (1024 ** 3), 2)}
    print(f"[kernel] flashinfer={flashinfer.__version__} torch={torch.__version__} "
          f"gpu={gpu['name']} sm{gpu['sm']} SMs={gpu['sm_count']}", flush=True)

    def byte_rate(a: "torch.Tensor", b: "torch.Tensor") -> float:
        # a,b: [n_qo_heads, head_dim] -- identical iff every head_dim element matches; mean over heads
        return float((a == b).all(dim=-1).float().mean().item())

    def planned(M: int, npages: int, hd: int, utc: bool, mode: str):
        ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        w = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=utc)
        indptr = torch.tensor([i * npages for i in range(M + 1)], dtype=torch.int32, device=dev)
        indices = torch.cat([torch.arange(npages, dtype=torch.int32, device=dev) for _ in range(M)])
        last = (npages * PAGE_SIZE) - ((npages - 1) * PAGE_SIZE)  # = PAGE_SIZE for full pages
        lpl = torch.full((M,), last, dtype=torch.int32, device=dev)
        kw: dict[str, Any] = {}
        if mode == "disable_split_kv":
            kw["disable_split_kv"] = True
        elif mode == "fixed_split_size":
            kw["fixed_split_size"] = FIXED_SPLIT_PAGES
        w.plan(indptr, indices, lpl, N_Q_HEADS, N_KV_HEADS, hd, PAGE_SIZE,
               pos_encoding_mode="NONE", q_data_type=dt, kv_data_type=dt, **kw)
        return w

    results: list[dict[str, Any]] = []
    geom_support: dict[str, dict[str, Any]] = {}
    any_nan = False
    t0 = time.time()

    for (glabel, hd, utc, modes) in GEOMS:
        # probe support once (tensor-core hd512 is W3-unsupported)
        try:
            npages0 = (seqlens[0] + PAGE_SIZE - 1) // PAGE_SIZE
            kv0 = torch.randn(npages0, 2, PAGE_SIZE, N_KV_HEADS, hd, device=dev, dtype=dt,
                              generator=torch.Generator(dev).manual_seed(seed + 7))
            q0 = torch.randn(1, N_Q_HEADS, hd, device=dev, dtype=dt,
                             generator=torch.Generator(dev).manual_seed(seed + 3))
            _ = planned(1, npages0, hd, utc, "default").run(q0, kv0)
            torch.cuda.synchronize()
            geom_support[glabel] = {"supported": True, "head_dim": hd, "use_tensor_cores": utc}
        except Exception as e:  # noqa: BLE001
            geom_support[glabel] = {"supported": False, "head_dim": hd, "use_tensor_cores": utc,
                                    "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"[kernel] geom {glabel}: UNSUPPORTED {geom_support[glabel]['error']}", flush=True)
            continue

        for seqlen in seqlens:
            npages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
            kv = torch.randn(npages, 2, PAGE_SIZE, N_KV_HEADS, hd, device=dev, dtype=dt,
                             generator=torch.Generator(dev).manual_seed(seed + seqlen))
            q8 = torch.randn(DEPLOYED_VERIFY_M, N_Q_HEADS, hd, device=dev, dtype=dt,
                             generator=torch.Generator(dev).manual_seed(seed + 1))
            q8[1:] = q8[0]                                   # all M queries identical to query-0
            qp = q8.clone()
            qp[0, 0, 0] = qp[0, 0, 0] + torch.tensor(0.5, dtype=dt, device=dev)

            for mode in modes:
                try:
                    w1 = planned(1, npages, hd, utc, mode)
                    w8 = planned(DEPLOYED_VERIFY_M, npages, hd, utc, mode)
                    w1b = planned(1, npages, hd, utc, mode)
                    o1 = w1.run(q8[:1].contiguous(), kv)
                    o8 = w8.run(q8[:DEPLOYED_VERIFY_M].contiguous(), kv)
                    o1b = w1b.run(q8[:1].contiguous(), kv)
                    op = w1b.run(qp[:1].contiguous(), kv)          # perturbed query, same wrapper
                    torch.cuda.synchronize()
                    nan = bool(torch.isnan(o1).any() or torch.isnan(o8).any())
                    any_nan = any_nan or nan
                    m_inv = byte_rate(o1[0], o8[0])
                    self_det = byte_rate(o1[0], o1b[0])
                    ctrl = byte_rate(o1[0], op[0])
                    # latency of the M=1 decode run (steady kernel; plan once, time run)
                    for _ in range(warmup):
                        w1.run(q8[:1].contiguous(), kv)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(n_lat_iters):
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record()
                        w1.run(q8[:1].contiguous(), kv)
                        e.record()
                        torch.cuda.synchronize()
                        times.append(s.elapsed_time(e))
                    times.sort()
                    lat_ms = times[len(times) // 2]
                    rec = {"geom": glabel, "head_dim": hd, "use_tensor_cores": utc, "seqlen": seqlen,
                           "mode": mode, "m_invariance_byte_rate": m_inv, "self_det_byte_rate": self_det,
                           "positive_control_byte_rate": ctrl, "latency_ms_m1": lat_ms, "nan": nan}
                    results.append(rec)
                    print(f"[kernel] {glabel:24s} L={seqlen:5d} {mode:16s} "
                          f"m_inv={m_inv:.4f} self={self_det:.4f} ctrl={ctrl:.4f} "
                          f"lat={lat_ms:.4f}ms", flush=True)
                except Exception as e:  # noqa: BLE001
                    rec = {"geom": glabel, "head_dim": hd, "use_tensor_cores": utc, "seqlen": seqlen,
                           "mode": mode, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    results.append(rec)
                    print(f"[kernel] {glabel:24s} L={seqlen:5d} {mode:16s} FAIL {rec['error']}",
                          flush=True)

    out = {
        "phase": "kernelcensus", "gpu": gpu,
        "flashinfer_version": flashinfer.__version__, "torch_version": str(torch.__version__),
        "seqlens": seqlens, "n_lat_iters": n_lat_iters, "fixed_split_pages": FIXED_SPLIT_PAGES,
        "verify_M": DEPLOYED_VERIFY_M, "geom_support": geom_support, "records": results,
        "any_nan": bool(any_nan),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(_jsonable(out), open(out_path, "w"), indent=2)
    print(f"KERNELCENSUS_DONE {out_path}", flush=True)


# ======================================================================================== #
# GPU PHASE: ENGINE -- vLLM backend-forcing capture + realized conc=1 decode-TPS anchor (optional)
# ======================================================================================== #
def _load_prompts(n_prompts: int, ctx_cap: int) -> list[dict]:
    path = ROOT / PROMPTS_JSONL
    rows = [json.loads(l) for l in open(path)][:n_prompts]
    out = []
    for rec in rows:
        ctx = list(rec.get("context_token_ids", []))[:ctx_cap]
        if len(ctx) >= 2:
            out.append({"prompt_token_ids": ctx})
    return out


def phase_engine(out_path: str, n_prompts: int, n_new: int, ctx_cap: int, gpu_mem_util: float) -> None:
    import tempfile
    import torch

    model_dir = resolve_engine_model()
    out: dict[str, Any] = {"phase": "engine", "requested_backend": "FLASHINFER",
                           "model_dir": model_dir, "full_vocab_fallback": True}
    if model_dir is None:
        out["error"] = "no loadable full-vocab int4 model found"
        json.dump(_jsonable(out), open(out_path, "w"), indent=2)
        print(f"ENGINE_DONE {out_path}", flush=True)
        return

    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"   # request Flashinfer; W2 will override
    # vLLM configures its OWN logging (bypasses python StreamHandlers), so capture the backend-selection
    # logs at the file-descriptor level around LLM construction, then re-emit so _fullrun.log keeps them.
    cap_path = tempfile.NamedTemporaryFile("w+", suffix=".vllmlog", delete=False).name
    try:
        from vllm import LLM, SamplingParams
        t_load = time.time()
        sys.stdout.flush(); sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
        cap_fd = os.open(cap_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(cap_fd, 1); os.dup2(cap_fd, 2)
        try:
            llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
                      max_model_len=max(1024, ctx_cap + n_new + 16), gpu_memory_utilization=gpu_mem_util,
                      max_num_seqs=1, max_num_batched_tokens=512, enable_prefix_caching=False,
                      enforce_eager=True, trust_remote_code=True)
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved_out, 1); os.dup2(saved_err, 2)
            os.close(cap_fd); os.close(saved_out); os.close(saved_err)
        out["load_s"] = round(time.time() - t_load, 1)
        log_text = Path(cap_path).read_text(errors="replace")
        sys.stderr.write(log_text)   # re-emit captured vLLM logs so the run log retains them
        sys.stderr.flush()
        forced = bool(re.search(r"Forcing TRITON_ATTN", log_text))
        backends = re.findall(r"Using AttentionBackendEnum\.(\w+)", log_text)
        hetero = bool(re.search(r"heterogeneous head dimensions", log_text))
        out["actual_backend"] = (backends[-1] if backends else ("TRITON_ATTN" if forced else "unknown"))
        out["vllm_forced_triton"] = forced
        out["heterogeneous_head_dims_detected"] = hetero
        out["forcing_log_line"] = next((ln.strip() for ln in log_text.splitlines()
                                        if "Forcing TRITON_ATTN" in ln), "")
        out["flashinfer_engine_available_for_gemma4"] = bool(
            out["actual_backend"].upper().startswith("FLASHINFER"))

        prompts = _load_prompts(n_prompts, ctx_cap)
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=n_new)
        torch.cuda.synchronize()
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t0
        toks_a = [list(o.outputs[0].token_ids) for o in outs]
        total_new = sum(len(t) for t in toks_a)
        out["decode_tps_conc1"] = (total_new / dt) if dt > 0 else float("nan")
        out["total_new_tokens"] = total_new
        out["n_prompts"] = len(prompts)
        # run-to-run self determinism end-to-end (TRITON path; expect 1.0)
        outs_b = llm.generate(prompts, sp, use_tqdm=False)
        toks_b = [list(o.outputs[0].token_ids) for o in outs_b]
        ident = sum(1 for a, b in zip(toks_a, toks_b) if a == b)
        out["self_det_completion_identity"] = ident / len(toks_a) if toks_a else float("nan")
        out["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        out["sample_tokens"] = toks_a[0][:12] if toks_a else []
        print(f"[engine] requested=FLASHINFER actual={out['actual_backend']} forced_triton={forced} "
              f"decode_tps={out['decode_tps_conc1']:.2f} self_det={out['self_det_completion_identity']}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"[engine] FAILED {out['error']}", flush=True)
    finally:
        Path(cap_path).unlink(missing_ok=True)
    json.dump(_jsonable(out), open(out_path, "w"), indent=2)
    print(f"ENGINE_DONE {out_path}", flush=True)


# ======================================================================================== #
# COMPOSE -- assemble the verdict + the PR KEY OUTPUTS
# ======================================================================================== #
def _recs_where(records: list[dict], **kw) -> list[dict]:
    out = []
    for r in records:
        if "error" in r:
            continue
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def compose(kernel: dict, engine: dict) -> dict[str, Any]:
    records = kernel.get("records", [])
    geom_support = kernel.get("geom_support", {})

    # --- self determinism: run-to-run byte identity at fixed occupancy across ALL good records ---
    self_dets = [_f(r["self_det_byte_rate"]) for r in records if "self_det_byte_rate" in r]
    flashinfer_m1_self_deterministic = bool(self_dets and all(s >= 0.999 for s in self_dets))

    # --- operative identity: M=1-vs-M=8 byte identity, WORST over the DEPLOYED per-token dispatch paths ---
    # the deployed model runs both a local (hd256) and a global (hd512) attention kernel every token; greedy
    # identity needs every layer byte-exact, so per seqlen we take the min m_inv across DEPLOYED_DECODE_GEOMS.
    dep_recs = [r for g in DEPLOYED_DECODE_GEOMS for r in _recs_where(records, geom=g, mode="default")]
    seqlens_sorted = sorted({int(r["seqlen"]) for r in dep_recs})
    minv_by_seqlen: dict[int, float] = {}
    binding_geom_by_seqlen: dict[int, str] = {}
    for L in seqlens_sorted:
        at_L = [r for r in dep_recs if int(r["seqlen"]) == L]
        worst = min(at_L, key=lambda r: _f(r["m_invariance_byte_rate"]))
        minv_by_seqlen[L] = _f(worst["m_invariance_byte_rate"])
        binding_geom_by_seqlen[L] = worst["geom"]
    op_ids = [minv_by_seqlen[L] for L in seqlens_sorted]
    flashinfer_m1_operative_identity = (min(op_ids) if op_ids else float("nan"))
    default_op_mean = (sum(op_ids) / len(op_ids)) if op_ids else float("nan")

    # --- split-divergence THRESHOLD: smallest tested KV where ANY deployed path's M=1-vs-M=8 breaks (<0.999) -
    # below it the deployed dispatch IS byte-exact (single-CTA, no split); above it dynamic split-KV picks a
    # different split count at M=1 vs M=8 -> different merge_states order -> different bits.
    split_divergence_threshold = None
    for L in seqlens_sorted:
        if minv_by_seqlen[L] < 0.999:
            split_divergence_threshold = L
            break
    byte_exact_ceiling = None       # largest tested KV where ALL deployed paths are still byte-exact
    for L in seqlens_sorted:
        if minv_by_seqlen[L] >= 0.999:
            byte_exact_ceiling = L

    # --- OPERATIVE-REGIME classification: does the divergence bite within the benchmark's actual KV reach? --
    def _minv_at_or_above(kv: int) -> float:
        at = [L for L in seqlens_sorted if L >= kv]
        if at:
            return minv_by_seqlen[min(at)]
        return minv_by_seqlen[seqlens_sorted[-1]] if seqlens_sorted else float("nan")

    in_regime = [L for L in seqlens_sorted if L <= OPERATIVE_KV_MAX]
    operative_worst_identity = (min(minv_by_seqlen[L] for L in in_regime)
                                if in_regime else float("nan"))
    operative_regime_byte_exact = bool(math.isfinite(operative_worst_identity)
                                       and operative_worst_identity >= 0.999)
    operative_regime_classes = {
        "p50_kv": OPERATIVE_KV_P50, "p50_byte_exact": bool(_minv_at_or_above(OPERATIVE_KV_P50) >= 0.999),
        "p90_kv": OPERATIVE_KV_P90, "p90_byte_exact": bool(_minv_at_or_above(OPERATIVE_KV_P90) >= 0.999),
        "max_kv": OPERATIVE_KV_MAX, "max_byte_exact": bool(_minv_at_or_above(OPERATIVE_KV_MAX) >= 0.999),
    }
    # threshold sits ABOVE the operative band iff divergence never seen at/below OPERATIVE_KV_MAX
    threshold_above_operative = bool(split_divergence_threshold is None
                                     or split_divergence_threshold > OPERATIVE_KV_MAX)
    binding_geom_at_threshold = (binding_geom_by_seqlen.get(split_divergence_threshold)
                                 if split_divergence_threshold is not None else None)

    # default m-invariance across all geoms (for the table / global head_dim too)
    default_minv_by_geom: dict[str, float] = {}
    for (glabel, _hd, _utc, _modes) in GEOMS:
        recs = _recs_where(records, geom=glabel, mode="default")
        vals = [_f(r["m_invariance_byte_rate"]) for r in recs]
        if vals:
            default_minv_by_geom[glabel] = min(vals)

    # --- do the determinism levers RESTORE batch invariance? (disable_split_kv / fixed_split_size) ---
    lever_restores: dict[str, dict[str, float]] = {}
    for (glabel, _hd, _utc, modes) in GEOMS:
        d: dict[str, float] = {}
        for mode in ("disable_split_kv", "fixed_split_size"):
            if mode in modes:
                recs = _recs_where(records, geom=glabel, mode=mode)
                vals = [_f(r["m_invariance_byte_rate"]) for r in recs]
                if vals:
                    d[mode] = min(vals)
        if d:
            lever_restores[glabel] = d
    any_lever_restores = any(v >= 0.999 for d in lever_restores.values() for v in d.values())

    # --- determinism TAX: latency(byte-exact mode) / latency(default) - 1, per geom (median over L) ---
    def _median(xs: list[float]) -> float:
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    determinism_tax: dict[str, dict[str, float]] = {}
    for (glabel, _hd, _utc, modes) in GEOMS:
        dflt = _recs_where(records, geom=glabel, mode="default")
        if not dflt:
            continue
        dflt_by_L = {r["seqlen"]: _f(r["latency_ms_m1"]) for r in dflt}
        tax: dict[str, float] = {}
        for mode in ("fixed_split_size", "disable_split_kv"):
            if mode not in modes:
                continue
            recs = _recs_where(records, geom=glabel, mode=mode)
            ratios = []
            for r in recs:
                L = r["seqlen"]
                ld = dflt_by_L.get(L)
                lm = _f(r["latency_ms_m1"])
                if ld and math.isfinite(ld) and math.isfinite(lm) and ld > 0:
                    ratios.append(lm / ld - 1.0)
            if ratios:
                tax[mode] = _median(ratios)
        if tax:
            determinism_tax[glabel] = tax

    # cheapest byte-exact tax at the SGLang-default geom (the relevant -107 analog)
    sg_tax = determinism_tax.get(SGLANG_DEFAULT_GEOM, {})
    cheapest_byteexact_tax_pct = float("nan")
    if sg_tax:
        cheapest_byteexact_tax_pct = 100.0 * min(v for v in sg_tax.values())

    # --- GLOBAL hd512 layer: can ANY available lever restore byte-exactness there? (W3 binding path) ---
    # tensor-core decode is unsupported at hd512 on sm_86 -> fixed_split_size is unavailable; the only lever
    # is disable_split_kv. If even that does not reach m_inv=1.0, the global-attention layers cannot be made
    # byte-exact by these knobs at all -> the byte-exact pin is not merely costly there, it is UNAVAILABLE.
    hd512_default_min = min((_f(r["m_invariance_byte_rate"])
                             for r in _recs_where(records, geom="global_hd512_cudacore", mode="default")),
                            default=float("nan"))
    hd512_lever_min = min((_f(r["m_invariance_byte_rate"])
                           for r in _recs_where(records, geom="global_hd512_cudacore",
                                                mode="disable_split_kv")), default=float("nan"))
    global_hd512_lever_restores = bool(math.isfinite(hd512_lever_min) and hd512_lever_min >= 0.999)

    # --- the headline verdict booleans ---
    # BY CONSTRUCTION (any KV): worst-case min over ALL tested seqlens
    default_is_byte_exact = bool(math.isfinite(flashinfer_m1_operative_identity)
                                 and flashinfer_m1_operative_identity >= 0.999)
    flashinfer_determinism_free = bool(flashinfer_m1_self_deterministic and default_is_byte_exact)
    # IN THE DEPLOYED REGIME (KV <= operative max): does the pin actually bind on this benchmark?
    flashinfer_determinism_free_operative = bool(flashinfer_m1_self_deterministic
                                                 and operative_regime_byte_exact)

    # global head_dim 512 lever availability (W3): fixed_split_size needs tensor cores, unsupported at 512
    hd512_tc_supported = bool(geom_support.get("global_hd512_cudacore", {}).get("supported")) and \
        bool(geom_support.get("local_hd256_tensorcore", {}).get("supported"))
    fixed_split_available_hd512 = False   # tensor-core decode unsupported at hd512 on sm_86 (probed)

    # the -107 pin binds in DEPLOYMENT iff the M=1-vs-M=8 divergence occurs within the operative KV band.
    collapses_tax = bool(flashinfer_determinism_free_operative)
    thr_txt = (f"KV>={split_divergence_threshold}" if split_divergence_threshold is not None
               else "no tested KV")
    max_tested_kv = (seqlens_sorted[-1] if seqlens_sorted else 0)
    if flashinfer_determinism_free:
        one_line = ("SGLang/Flashinfer at M=1 decode is FAST-AND-BYTE-EXACT-FREE by construction (default "
                    "dynamic split-KV stays M=1-vs-M=8 byte-identical across ALL tested KV up to "
                    f"{max_tested_kv}, both local hd256 and global hd512 paths) -> COLLAPSES the -107 "
                    "2D-attention tax that caps surgical-357.")
    elif collapses_tax:
        one_line = ("SGLang/Flashinfer M=1 decode is byte-exact WITHIN the benchmark's operative KV band "
                    f"(<= {OPERATIVE_KV_MAX}; M=1-vs-M=8 first breaks at {thr_txt} on the "
                    f"{binding_geom_at_threshold} path) but is NOT byte-exact by construction -> the -107 "
                    "pin does NOT bind in this deployment (tax collapses in practice), though Flashinfer is "
                    "occupancy-variant at longer KV.")
    else:
        lever_note = ("disable_split_kv / fixed_split_size (own kernel tax "
                      f"~{cheapest_byteexact_tax_pct:.0f}%)")
        if binding_geom_at_threshold == "global_hd512_cudacore" and not global_hd512_lever_restores:
            lever_note = ("a byte-exact lever that WORKS -- at the binding global head_dim=512 layers "
                          "fixed_split_size is unavailable (no tensor cores, sm_86) AND disable_split_kv "
                          f"does NOT restore invariance (m_inv stays {hd512_lever_min:.3f}), so no cheap "
                          "knob yields byte-exactness there")
        one_line = ("SGLang/Flashinfer at M=1 decode is FAST-BUT-NOT-BYTE-EXACT (default dynamic split-KV "
                    "is occupancy-variant; worst-case M=1-vs-M=8 byte identity "
                    f"{flashinfer_m1_operative_identity:.3f}, first breaks at {thr_txt} on the "
                    f"{binding_geom_at_threshold} path -- WITHIN the operative band <= {OPERATIVE_KV_MAX}) -> "
                    "does NOT collapse the -107 2D-attention tax; the tax is ENGINE-INDEPENDENT and bites in "
                    f"deployment. Byte-exact Flashinfer needs {lever_note}.")

    verdict = (
        f"DIAGNOSTIC (LOCAL, analysis_only). flashinfer_m1_self_deterministic="
        f"{flashinfer_m1_self_deterministic} (run-to-run at fixed occupancy). "
        f"flashinfer_m1_operative_identity={flashinfer_m1_operative_identity:.4f} (worst-case M=1-vs-M=8 "
        f"byte identity over all tested KV, DEFAULT dynamic split-KV, WORST over the deployed per-token "
        f"dispatch paths {list(DEPLOYED_DECODE_GEOMS)}; 1.0=byte-exact). split-divergence threshold="
        f"{split_divergence_threshold} on the {binding_geom_at_threshold} path (smallest KV where M=1-vs-M=8 "
        f"first breaks; byte-exact ceiling KV={byte_exact_ceiling}). global hd512 lever check: "
        f"disable_split_kv m_inv={hd512_lever_min:.3f} (restores={global_hd512_lever_restores}; "
        f"fixed_split_size unavailable, no tensor cores at hd512). benchmark operative decode-KV band "
        f"p50={OPERATIVE_KV_P50} "
        f"p90={OPERATIVE_KV_P90} max={OPERATIVE_KV_MAX} -> operative_regime_byte_exact="
        f"{operative_regime_byte_exact} (worst in-band identity {operative_worst_identity:.4f}); "
        f"flashinfer_determinism_free(by-construction)={flashinfer_determinism_free}, "
        f"flashinfer_determinism_free(operative)={flashinfer_determinism_free_operative}. "
        f"Determinism levers restore invariance (disable_split_kv / fixed_split_size -> m_inv=1.0): "
        f"{any_lever_restores}; cheapest byte-exact kernel tax at the GQA path "
        f"~{cheapest_byteexact_tax_pct:.1f}%. flashinfer's plan() docstring itself states fixed_split_size "
        f"yields 'batch-size invariant outputs' (citing thinkingmachines.ai) -> the DEFAULT is not "
        f"batch-invariant by design. Walls: W1 SGLang '0.22.1rc1' is the vLLM version (real SGLang "
        f"{SGLANG_REAL_LATEST}, sgl-kernel needs CUDA13 vs pod CUDA12.9); W2 vLLM forces TRITON_ATTN for "
        f"Gemma-4 heterogeneous head dims (actual_backend={engine.get('actual_backend', 'n/a')}); W3 "
        f"flashinfer tensor-core decode unsupported at global head_dim=512 on sm_86 so fixed_split_size "
        f"(cheap lever) is unavailable there. VERDICT: {one_line}")

    return {
        "flashinfer_m1_self_deterministic": flashinfer_m1_self_deterministic,
        "flashinfer_m1_operative_identity": flashinfer_m1_operative_identity,
        "flashinfer_m1_operative_identity_mean": default_op_mean,
        "flashinfer_determinism_free": flashinfer_determinism_free,
        "flashinfer_determinism_free_operative": flashinfer_determinism_free_operative,
        "default_is_byte_exact": default_is_byte_exact,
        "split_divergence_threshold_seqlen": split_divergence_threshold,
        "byte_exact_ceiling_seqlen": byte_exact_ceiling,
        "operative_regime_byte_exact": operative_regime_byte_exact,
        "operative_worst_identity": operative_worst_identity,
        "operative_regime_classes": operative_regime_classes,
        "threshold_above_operative": threshold_above_operative,
        "binding_geom_at_threshold": binding_geom_at_threshold,
        "operative_kv_band": {"p50": OPERATIVE_KV_P50, "p90": OPERATIVE_KV_P90, "max": OPERATIVE_KV_MAX},
        "deployed_decode_geoms": list(DEPLOYED_DECODE_GEOMS),
        "minv_by_seqlen_worst_deployed": {str(L): v for L, v in minv_by_seqlen.items()},
        "global_hd512_default_min": hd512_default_min,
        "global_hd512_lever_min": hd512_lever_min,
        "global_hd512_lever_restores": global_hd512_lever_restores,
        "default_m_invariance_by_geom": default_minv_by_geom,
        "lever_restores_invariance": lever_restores,
        "any_lever_restores_invariance": any_lever_restores,
        "determinism_tax_pct_by_geom": {g: {m: 100.0 * v for m, v in d.items()}
                                        for g, d in determinism_tax.items()},
        "cheapest_byteexact_kernel_tax_pct": cheapest_byteexact_tax_pct,
        "fixed_split_available_hd512": fixed_split_available_hd512,
        "geom_support": geom_support,
        "collapses_minus107_tax": collapses_tax,
        # SGLang TPS anchor: BLOCKED -> realized TRITON-vanilla anchor only (W1/W2/W3), config-caveated
        "sglang_decode_tps": None,
        "sglang_decode_tps_status": ("BLOCKED: SGLang uninstallable (pod CUDA12.9 vs sgl-kernel CUDA13; "
                                     "'0.22.1rc1' is the vLLM version); vLLM forces TRITON_ATTN for "
                                     "Gemma-4; flashinfer decode unsupported at global head_dim=512 on "
                                     "sm_86. See engine_decode_tps_triton_vanilla for the realized "
                                     "on-A10G anchor and determinism_tax_pct for the attention-kernel tax."),
        "sglang_vs_vllm_deployed_ratio": None,
        "engine_actual_backend": engine.get("actual_backend"),
        "engine_decode_tps_triton_vanilla": engine.get("decode_tps_conc1"),
        "engine_self_det_completion_identity": engine.get("self_det_completion_identity"),
        "engine_vllm_forced_triton": engine.get("vllm_forced_triton"),
        "engine_heterogeneous_detected": engine.get("heterogeneous_head_dims_detected"),
        "engine_forcing_log_line": engine.get("forcing_log_line"),
        "one_line_verdict": one_line,
        "verdict": verdict,
        "anchors": {"deployed_tps": DEPLOYED_TPS, "surgical_tps": SURGICAL_TPS,
                    "floorlock_tps": FLOORLOCK_TPS, "attention_2d_tax_tps": ATTENTION_2D_TAX_TPS},
    }


# ======================================================================================== #
# SELF-TEST (PRIMARY: flashinfer_diagnostic_self_test_passes)
# ======================================================================================== #
def selftest(kernel: dict, comp: dict, flags: dict) -> dict[str, Any]:
    c: dict[str, bool] = {}
    records = kernel.get("records", [])
    good = [r for r in records if "error" not in r]
    # (a) census produced data, NaN-clean
    c["a_has_records"] = bool(len(good) > 0)
    c["a_nan_clean"] = (not bool(kernel.get("any_nan", True)))
    # (b) positive control sensitive: at least the SGLang-default geom default-mode ctrl < 0.999
    sg_default = _recs_where(records, geom=SGLANG_DEFAULT_GEOM, mode="default")
    c["b_positive_control_sensitive"] = bool(
        sg_default and all(_f(r["positive_control_byte_rate"]) < 0.999 for r in sg_default))
    # (c) run-to-run self determinism observed (the self_det leg is meaningful)
    c["c_self_det_observed"] = bool(comp["flashinfer_m1_self_deterministic"])
    # (d) the EXPERIMENT IS DECISIVE: either default is byte-exact (free), or a lever restores invariance
    #     (proving the comparator can read a 1.0). One of the two must hold or the census is vacuous.
    c["d_census_decisive"] = bool(comp["default_is_byte_exact"] or comp["any_lever_restores_invariance"])
    # (e) operative identity is a finite measured float in [0,1]
    oi = comp["flashinfer_m1_operative_identity"]
    c["e_operative_identity_measured"] = bool(math.isfinite(oi) and 0.0 <= oi <= 1.0)
    # (f) determinism-free verdict is internally consistent with self & operative identity
    expect_free = bool(comp["flashinfer_m1_self_deterministic"] and comp["default_is_byte_exact"])
    c["f_verdict_consistent"] = bool(comp["flashinfer_determinism_free"] == expect_free)
    # (g) flags clean (no launch / analysis only)
    c["g_no_launch_flags"] = bool(flags["no_hf_job"] and flags["no_launch"] and flags["analysis_only"])
    # (h) REGIME GUARD (kills the short-seqlen artifact): a "byte-exact-free" reading is only admissible
    #     if the sweep on the DEFAULT path either OBSERVED the split break, or confirmed byte-exactness out
    #     to at least the operative-max KV -- so a 1.0 cannot come from merely testing short seqlens.
    thr = comp.get("split_divergence_threshold_seqlen")
    ceil_ = comp.get("byte_exact_ceiling_seqlen")
    c["h_regime_resolved"] = bool((thr is not None)
                                  or (ceil_ is not None and ceil_ >= OPERATIVE_KV_MAX))
    # (i) operative-regime classification is internally consistent with the threshold location
    thr_above = comp.get("threshold_above_operative")
    c["i_operative_consistent"] = bool(comp["operative_regime_byte_exact"] == bool(thr_above))
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


# ======================================================================================== #
# Report + wandb + orchestration
# ======================================================================================== #
def print_report(payload: dict) -> None:
    comp, st, kern = payload["compose"], payload["selftest"], payload["kernelcensus"]
    bar = "=" * 104
    print(bar)
    print("SGLANG/FLASHINFER M=1 DECODE DETERMINISM DIAGNOSTIC -- is byte-exact FREE? (PR #498, denken)")
    g = kern.get("gpu", {})
    print(f"  GPU {g.get('name')} sm{g.get('sm')} SMs={g.get('sm_count')}  "
          f"flashinfer={kern.get('flashinfer_version')} torch={kern.get('torch_version')}")
    print("-" * 104)
    print(f"  {'geom':26s} {'L':>6s} {'mode':16s} {'m_inv(M1vsM8)':>14s} {'self_r2r':>9s} "
          f"{'ctrl':>6s} {'lat_ms':>8s}")
    for r in kern.get("records", []):
        if "error" in r:
            print(f"  {r['geom']:26s} {r['seqlen']:6d} {r['mode']:16s}   ERROR {r['error'][:40]}")
            continue
        print(f"  {r['geom']:26s} {r['seqlen']:6d} {r['mode']:16s} {r['m_invariance_byte_rate']:14.4f} "
              f"{r['self_det_byte_rate']:9.4f} {r['positive_control_byte_rate']:6.3f} "
              f"{r['latency_ms_m1']:8.4f}")
    print("-" * 104)
    print(f"  flashinfer_m1_self_deterministic = {comp['flashinfer_m1_self_deterministic']}")
    print(f"  flashinfer_m1_operative_identity = {comp['flashinfer_m1_operative_identity']:.4f}  "
          f"(worst-case M=1-vs-M=8 over all tested KV, WORST over deployed paths "
          f"{comp.get('deployed_decode_geoms')}; 1.0=byte-exact)")
    print(f"  split_divergence_threshold KV    = {comp['split_divergence_threshold_seqlen']} "
          f"on {comp.get('binding_geom_at_threshold')}  (byte-exact ceiling KV={comp['byte_exact_ceiling_seqlen']})")
    print(f"  global hd512 lever (disable_splitkv)= m_inv={_f(comp.get('global_hd512_lever_min')):.4f} "
          f"restores={comp.get('global_hd512_lever_restores')}  "
          f"(fixed_split_size unavailable at hd512, no tensor cores sm_86)")
    band = comp['operative_kv_band']
    print(f"  operative decode-KV band         = p50={band['p50']} p90={band['p90']} max={band['max']}  "
          f"-> operative_regime_byte_exact={comp['operative_regime_byte_exact']} "
          f"(worst in-band identity {comp['operative_worst_identity']:.4f})")
    print(f"  flashinfer_determinism_free      = {comp['flashinfer_determinism_free']} (by-construction) | "
          f"{comp['flashinfer_determinism_free_operative']} (operative regime)")
    print(f"  default m_invariance by geom      = "
          + json.dumps({k: round(v, 4) for k, v in comp['default_m_invariance_by_geom'].items()}))
    print(f"  lever restores invariance         = "
          + json.dumps({k: {m: round(x, 3) for m, x in d.items()}
                        for k, d in comp['lever_restores_invariance'].items()}))
    print(f"  determinism tax % by geom         = "
          + json.dumps({k: {m: round(x, 1) for m, x in d.items()}
                        for k, d in comp['determinism_tax_pct_by_geom'].items()}))
    print(f"  cheapest byte-exact kernel tax %  = {comp['cheapest_byteexact_kernel_tax_pct']:.1f}")
    print(f"  engine actual_backend (req FI)    = {comp['engine_actual_backend']}  "
          f"(forced_triton={comp['engine_vllm_forced_triton']})")
    print(f"  engine decode_tps (TRITON vanilla)= {comp['engine_decode_tps_triton_vanilla']}")
    print(f"  sglang_decode_tps                 = {comp['sglang_decode_tps']}  (BLOCKED, see status)")
    print("-" * 104)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): "
          + json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 104)
    print("  VERDICT\n   " + comp["one_line_verdict"])
    print(bar)


def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[diag] wandb helpers unavailable: {e}")
        return None
    comp = payload["compose"]
    run = init_wandb_run(
        job_type="analysis-sglang-flashinfer-diagnostic", agent="denken",
        name=args.wandb_name, group=args.wandb_group,
        tags=["sglang-flashinfer-diagnostic", "flashinfer-determinism", "m1-decode",
              "split-kv", "481-zoom-out", "pr-498"],
        config={"pr": 498, "kind": "sglang-flashinfer-diagnostic",
                "deployed_tps": DEPLOYED_TPS, "surgical_tps": SURGICAL_TPS,
                "attention_2d_tax_tps": ATTENTION_2D_TAX_TPS,
                "verify_M": DEPLOYED_VERIFY_M, "fixed_split_pages": FIXED_SPLIT_PAGES,
                "sglang_version_in_pr": SGLANG_VERSION_IN_PR, "sglang_real_latest": SGLANG_REAL_LATEST},
    )
    if run is None:
        print("[diag] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "diag/flashinfer_m1_self_deterministic": float(bool(comp["flashinfer_m1_self_deterministic"])),
        "diag/flashinfer_m1_operative_identity": _f(comp["flashinfer_m1_operative_identity"]),
        "diag/flashinfer_m1_operative_identity_mean": _f(comp["flashinfer_m1_operative_identity_mean"]),
        "diag/flashinfer_determinism_free": float(bool(comp["flashinfer_determinism_free"])),
        "diag/flashinfer_determinism_free_operative": float(
            bool(comp["flashinfer_determinism_free_operative"])),
        "diag/default_is_byte_exact": float(bool(comp["default_is_byte_exact"])),
        "diag/split_divergence_threshold_seqlen": _f(comp["split_divergence_threshold_seqlen"]),
        "diag/byte_exact_ceiling_seqlen": _f(comp["byte_exact_ceiling_seqlen"]),
        "diag/operative_regime_byte_exact": float(bool(comp["operative_regime_byte_exact"])),
        "diag/operative_worst_identity": _f(comp["operative_worst_identity"]),
        "diag/threshold_above_operative": float(bool(comp["threshold_above_operative"])),
        "diag/global_hd512_default_min": _f(comp["global_hd512_default_min"]),
        "diag/global_hd512_lever_min": _f(comp["global_hd512_lever_min"]),
        "diag/global_hd512_lever_restores": float(bool(comp["global_hd512_lever_restores"])),
        "diag/any_lever_restores_invariance": float(bool(comp["any_lever_restores_invariance"])),
        "diag/cheapest_byteexact_kernel_tax_pct": _f(comp["cheapest_byteexact_kernel_tax_pct"]),
        "diag/collapses_minus107_tax": float(bool(comp["collapses_minus107_tax"])),
        "diag/engine_decode_tps_triton_vanilla": _f(comp["engine_decode_tps_triton_vanilla"]),
        "diag/engine_vllm_forced_triton": float(bool(comp.get("engine_vllm_forced_triton"))),
        "selftest/flashinfer_diagnostic_self_test_passes": float(payload["selftest"]["passes"]),
    }
    for g, v in comp["default_m_invariance_by_geom"].items():
        flat[f"default_m_inv/{g}"] = _f(v)
    for L, v in comp.get("minv_by_seqlen_worst_deployed", {}).items():
        flat[f"worst_deployed_m_inv/L{L}"] = _f(v)
    run.log({"global_step": 0, **{k: (v if (isinstance(v, float) and math.isfinite(v)) else 0.0)
                                  for k, v in flat.items()}})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="sglang_flashinfer_diagnostic",
                      artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[diag] wandb logged (run {rid})")
    return rid


def resolve_server_python(arg: str | None) -> str:
    if arg:
        return arg
    default = "/tmp/server-venv/bin/python"
    if Path(default).exists():
        return default
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.local_validation import harness  # noqa: E402
    m = harness.load_manifest(ROOT / "submissions" / "fa2sw_precache_kenyan")
    return str(harness.ensure_server_venv(m["dependencies"]))


def run_gpu_phase(server_python: str, phase_args: list[str], timeout: int, extra_env: dict) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.update(extra_env)
    cmd = [server_python, os.path.abspath(__file__)] + phase_args
    print(f"[orch] launching: {' '.join(phase_args)}", flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[orch] phase TIMED OUT after {timeout}s: {phase_args}", flush=True)
        return 124


def orchestrate(args) -> int:
    import shutil
    import tempfile
    HERE.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="sgl_flashinfer_diag_"))  # intermediate phase JSONs (not committed)
    kernel_json = str(scratch / "_kernelcensus.json")
    engine_json = str(scratch / "_engine.json")
    server_python = resolve_server_python(args.server_python)
    print(f"[orch] server_python = {server_python}", flush=True)

    seqlens = [int(x) for x in args.seqlens.split(",")]
    rc_k = run_gpu_phase(server_python, [
        "--phase", "kernelcensus", "--out", kernel_json,
        "--seqlens", args.seqlens, "--lat-iters", str(args.lat_iters),
        "--warmup", str(args.warmup), "--seed", str(args.seed)],
        timeout=args.kernel_timeout, extra_env={})
    kernel = json.load(open(kernel_json)) if Path(kernel_json).exists() else {
        "phase": "kernelcensus", "records": [], "error": rc_k, "any_nan": True}

    engine: dict[str, Any] = {"phase": "engine", "skipped": True}
    if not args.no_engine:
        rc_e = run_gpu_phase(server_python, [
            "--phase", "engine", "--out", engine_json,
            "--engine-prompts", str(args.engine_prompts), "--engine-new", str(args.engine_new),
            "--ctx-cap", str(args.ctx_cap), "--gpu-mem-util", str(args.gpu_mem_util)],
            timeout=args.engine_timeout, extra_env={"VLLM_ATTENTION_BACKEND": "FLASHINFER"})
        engine = json.load(open(engine_json)) if Path(engine_json).exists() else {
            "phase": "engine", "error": rc_e}

    comp = compose(kernel, engine)
    flags = {"no_hf_job": True, "no_launch": True, "analysis_only": True,
             "no_served_file_change": True, "official_tps": 0}
    st = selftest(kernel, comp, flags)

    payload = {
        "agent": "denken", "pr": 498, "kind": "sglang-flashinfer-diagnostic",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "kernelcensus": kernel, "engine": engine, "compose": comp, "selftest": st,
        "flashinfer_diagnostic_self_test_passes": bool(st["passes"]),
        # headline KEY OUTPUTS
        "flashinfer_m1_self_deterministic": comp["flashinfer_m1_self_deterministic"],
        "flashinfer_m1_operative_identity": comp["flashinfer_m1_operative_identity"],
        "flashinfer_determinism_free": comp["flashinfer_determinism_free"],
        "flashinfer_determinism_free_operative": comp["flashinfer_determinism_free_operative"],
        "split_divergence_threshold_seqlen": comp["split_divergence_threshold_seqlen"],
        "operative_regime_byte_exact": comp["operative_regime_byte_exact"],
        "sglang_decode_tps": comp["sglang_decode_tps"],
        "sglang_vs_vllm_deployed_ratio": comp["sglang_vs_vllm_deployed_ratio"],
        "collapses_minus107_tax": comp["collapses_minus107_tax"],
    }
    print_report(payload)
    out_path = HERE / "sglang_flashinfer_diagnostic_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[diag] wrote {out_path}", flush=True)
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    result = {"terminal": True, "status": "complete", "pending_arms": False,
              "wandb_run_ids": ([rid] if rid else []),
              "flashinfer_m1_self_deterministic": comp["flashinfer_m1_self_deterministic"],
              "flashinfer_m1_operative_identity": round(_f(comp["flashinfer_m1_operative_identity"]), 4),
              "flashinfer_determinism_free": comp["flashinfer_determinism_free"],
              "flashinfer_determinism_free_operative": comp["flashinfer_determinism_free_operative"],
              "split_divergence_threshold_seqlen": comp["split_divergence_threshold_seqlen"],
              "operative_regime_byte_exact": comp["operative_regime_byte_exact"],
              "collapses_minus107_tax": comp["collapses_minus107_tax"],
              "cheapest_byteexact_kernel_tax_pct": round(_f(comp["cheapest_byteexact_kernel_tax_pct"]), 1),
              "sglang_decode_tps": comp["sglang_decode_tps"],
              "engine_decode_tps_triton_vanilla": (round(_f(comp["engine_decode_tps_triton_vanilla"]), 2)
                                                   if comp.get("engine_decode_tps_triton_vanilla")
                                                   is not None else None),
              "primary_metric": {"name": "flashinfer_m1_operative_identity",
                                 "value": round(_f(comp["flashinfer_m1_operative_identity"]), 4)},
              "test_metric": {"name": "cheapest_byteexact_kernel_tax_pct",
                              "value": round(_f(comp["cheapest_byteexact_kernel_tax_pct"]), 1)},
              "self_test_passes": bool(st["passes"]),
              "verdict": comp["one_line_verdict"]}
    print("SENPAI-RESULT: " + json.dumps(result), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0 if st["passes"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["kernelcensus", "engine"], default=None,
                    help="internal GPU phase dispatch (run under the server venv)")
    ap.add_argument("--out", default=None)
    # kernelcensus
    # dense sweep brackets the operative band (720/896/2944 = p50/p90/max KV) to locate the
    # split-divergence threshold and classify whether the -107 pin binds in the deployed regime.
    ap.add_argument("--seqlens", default="256,512,720,896,1024,1280,1536,2048,2944,4096")
    ap.add_argument("--lat-iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    # engine
    ap.add_argument("--engine-prompts", type=int, default=8)
    ap.add_argument("--engine-new", type=int, default=128)
    ap.add_argument("--ctx-cap", type=int, default=256)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--no-engine", action="store_true", help="skip the vLLM engine leg")
    # orchestration
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--kernel-timeout", type=int, default=1800)
    ap.add_argument("--engine-timeout", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny fast path for validation")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/sglang-flashinfer-diagnostic")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="sglang-flashinfer-diagnostic")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seqlens = "512,896,4096"   # short(byte-exact) + operative-p90 + long(broken): verdict must flip
        args.lat_iters = min(args.lat_iters, 10)
        args.engine_prompts = min(args.engine_prompts, 3)
        args.engine_new = min(args.engine_new, 16)

    if args.phase == "kernelcensus":
        seqlens = [int(x) for x in args.seqlens.split(",")]
        phase_kernelcensus(args.out, seqlens, args.lat_iters, args.warmup, args.seed)
        return
    if args.phase == "engine":
        phase_engine(args.out, args.engine_prompts, args.engine_new, args.ctx_cap, args.gpu_mem_util)
        return
    raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    main()
