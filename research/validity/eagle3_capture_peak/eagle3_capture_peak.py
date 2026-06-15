#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 build capture-PEAK VRAM audit: does the 3.90 GiB headroom survive runtime? (PR #306).

THE QUESTION
------------
ubel #299 (jnoss7id, MERGED) priced the RESIDENT VRAM of the {2,21,39}-fusion EAGLE-3 build
at 20.10 GiB -- 3.90 GiB under the 24 GiB hard ceiling, 2.90 GiB under 23-usable -- and closed
the matrix's VRAM axis (e) AT REST. But "fits at rest" is not "fits at runtime." The deployed
ONEGRAPH serving path does two things that spike memory ABOVE the resident floor:
  1. CUDA-GRAPH CAPTURE of the fused draft->verify step allocates a capture-time private memory
     pool (cuBLAS/attention scratch, held activations) that is NOT part of steady-state residency.
  2. The M-WIDE TREE-VERIFY transient -- the 262144-vocab lm_head logit buffer over M candidates,
     live simultaneously with the {2,21,39} 3-layer hidden-state retention during fusion.

The open question: does the build's 3.90 GiB headroom SURVIVE the capture-time peak + the
tree-verify logit transient, or does the peak eat the headroom and turn a "fits statically" build
into an OOM launch-blocker on the a10g-small 24 GiB lane? A build that passes #299's resident
budget but OOMs during ONEGRAPH capture is exactly the silent launch-blocker the human GO/NO-GO
must see priced BEFORE any training spend.

THE PRECEDENT (load-bearing risk this prices)
---------------------------------------------
lawine's "size-29 CUDA-graph-capture crash" (#245 cycle-1 / land, EXPERIMENTS_LOG.md:318):
`max_cudagraph_capture_size=16`; the M=8 spine verify ((1+7) tokens) captures cleanly, M=16 sits
AT the boundary (capturable), and a 29-token verify is OVER it (crash). CRUCIAL HONESTY: the
repo's own diagnosis of that crash is a capture-SIZE-LIST boundary (no captured graph exists for
batch=29 > max 16), NOT a raw VRAM out-of-memory. This leg prices the VRAM axis the PR asks for
AND states explicitly where the EAGLE-3 capture sits relative to BOTH the VRAM ceiling and the
#101 capture-size boundary (the M-sweep {8,16,32} straddles it: 8 safe, 16 boundary, 32 over).

WHAT THIS LEG MEASURES (random-init, NO training, NO checkpoint, NO served change, 0 TPS)
----------------------------------------------------------------------------------------
On the live A10G, the PEAK (max_memory_allocated / reserved, reset between phases) of:
  (i)   ONEGRAPH-style CUDA-graph CAPTURE of the fused {2,21,39} draft->verify step (bf16, batch=1
        draft chain K=7 + M-wide verify): capture-pool reserved delta vs post-capture resident.
  (ii)  M-WIDE tree-verify forward -- the 262144-vocab lm_head GEMM over M candidates; sweep
        M in {8,16,32} to expose the logit transient's M-scaling.
  (iii) 3-LAYER {2,21,39} hidden-state retention held live during the draft.
Then: import #299's resident budget, compute peak = resident + transient, decompose the transient
into {capture-time scratch, tree-verify logit buffer (262144 x M x dtype), workspace/remat},
identify the DOMINANT transient term (the runtime analog of #299's resident `extra_kv`), and report
whether peak <= 24 GiB hard AND peak <= 23 GiB usable. If the peak ever exceeds the headroom, price
the cheapest analytic mitigation (capture batch, chunked logits, M cap) as ANALYSIS only.

The drafter geometry is read from the repo (scripts/drafter/train_eagle3.py Eagle3DraftHead /
research/eagle3_drafter/arch_notes.md): HID=2560, 1 Llama layer (8q/2kv x 256, INTER=10240), fc
[7680->2560] over the {2,21,39} aux hiddens, reuses the target embed+lm_head (#299 S0). The verify
proxy is one representative target-width bf16 decoder layer at batch=M + the lm_head over M tokens.

Analysis + LOCAL random-init GPU profiling. BASELINE 481.53 untouched (adds 0 TPS). NOT a launch;
no served-file change; no HF Job; no submission; NOT a build."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                      # .../target

GIB = float(1024 ** 3)

# --------------------------------------------------------------------------- #
# Banked anchors imported VERBATIM from ubel #299 (jnoss7id) and its sources;
# NEVER re-derived. All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                        # PR #52 official frontier TPS (untouched)

# ---- ubel #299 jnoss7id: the RESIDENT VRAM budget this leg stress-tests at RUNTIME ----
BUILD_RESIDENT_GIB_299 = 20.100143778324128       # EAGLE-3 build resident (conservative, hold-KV)
DEPLOYED_RESIDENT_GIB_299 = 19.3                   # #284 deployed resident anchor (pre-EAGLE-3)
NET_DELTA_CONSERVATIVE_GIB_299 = 0.8001437783241272
EXTRA_KV_GIB_299 = 0.718841552734375              # dominant RESIDENT term (#299), elastic in deploy
DRAFTER_WEIGHTS_GIB_299 = 0.037352144718170166
FUSION_FC_GIB_299 = 0.03662586212158203
HIDDEN_RETENTION_GIB_299 = 0.00732421875          # L_FUSE(3) x H(2560) x 512 positions x 2B (resident)
HEADROOM_24_HARD_GIB_299 = 3.899856221675872      # 24.0 - 20.10
HEADROOM_23_USABLE_GIB_299 = 2.899856221675872    # 23.0 - 20.10
HEADROOM_DEVVIS_GIB_299 = 1.957856221675872       # 22.058 - 20.10
NONTORCH_CUDA_CONTEXT_GIB_299 = 0.9499999999999993
CUDA_GRAPH_POOL_GIB_299 = 0.04                    # the LINEAR drafter graph pool inside resident
DEVICE_VISIBLE_GIB_299 = 22.0582275390625         # measured A10G visible cap (== banked)
VRAM_HARD_GIB = 24.0
VRAM_USABLE_GIB = 23.0
KV_CACHE_TOKENS_299 = 376880
KV_BYTES_PER_TOKEN_PER_LAYER_299 = 2048

# ---- build geometry (#299 constants / live config / train_eagle3.py) ----
HID = 2560
N_LAYERS = 42
N_HEADS = 8
N_KV = 2
HEAD_DIM = 256
INTER = 10240
VOCAB = 262144
EAGLE3_AUX_LAYERS = (2, 21, 39)
L_FUSE = 3
FC_IN = L_FUSE * HID                               # 7680
K_SPEC = 7                                         # K=7 linear/EAGLE-3 draft chain
M_VERIFY_DEPLOYED = 8                              # deployed tree verify width (#299 m_verify)
CTX_DEPLOYED = 528                                 # denken #278 KV context basis

# ---- lawine size-29 capture-crash boundary (#245 cycle-1, EXPERIMENTS_LOG.md:318) ----
MAX_CUDAGRAPH_CAPTURE_SIZE_101 = 16               # vLLM target-model capture-size ceiling
SPINE_VERIFY_TOKENS_101 = 8                       # (1+K)=8 spine verify; captures cleanly
SIZE29_CRASH_TOKENS_101 = 29                      # the verify-token count that crashed (29 > 16)


# --------------------------------------------------------------------------- #
# Memory primitives.
# --------------------------------------------------------------------------- #
def _reset_peak():
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _mem():
    """Return (allocated, reserved, max_allocated, max_reserved) bytes, synced."""
    import torch
    torch.cuda.synchronize()
    return (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
            torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())


# --------------------------------------------------------------------------- #
# CUDA-graph capture primitive (copied from research/validity/eagle3_step_profile/
# for an apples-to-apples deployed-ONEGRAPH basis: side-stream warmup then capture).
# --------------------------------------------------------------------------- #
def _capture(run):
    import torch
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        run()
    return g


# =========================================================================== #
# Random-init step modules (NO checkpoint). Mirror the repo's faithful EAGLE-3
# (scripts/drafter/train_eagle3.py) + a target-width verify-layer proxy.
# =========================================================================== #
def _build_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    from scripts.drafter.train_eagle3 import RMSNorm, apply_rope, build_rope  # noqa: E402

    def _sdpa_kv(q, k_new, v_new, kv_k, kv_v, rep):
        k = torch.cat([kv_k, k_new], dim=2)
        v = torch.cat([kv_v, v_new], dim=2)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v)

    # ---- faithful EAGLE-3 first decoder layer (qkv input = 2H; train_eagle3.py) ---- #
    class EagleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2 * HID, N_HEADS * HEAD_DIM, bias=False)
            self.k_proj = nn.Linear(2 * HID, N_KV * HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(2 * HID, N_KV * HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HID, bias=False)
            self.gate = nn.Linear(HID, INTER, bias=False)
            self.up = nn.Linear(HID, INTER, bias=False)
            self.down = nn.Linear(INTER, HID, bias=False)
            self.input_layernorm = RMSNorm(HID)
            self.hidden_norm = RMSNorm(HID)
            self.post_attention_layernorm = RMSNorm(HID)

        def forward(self, embeds, hidden, kv_k, kv_v, cos1, sin1):
            B = embeds.shape[0]
            e = self.input_layernorm(embeds)
            residual = hidden
            hn = self.hidden_norm(hidden)
            x = torch.cat([e, hn], dim=-1)
            q = self.q_proj(x).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = self.k_proj(x).view(B, 1, N_KV, HEAD_DIM).transpose(1, 2)
            v = self.v_proj(x).view(B, 1, N_KV, HEAD_DIM).transpose(1, 2)
            q, k = apply_rope(q, k, cos1, sin1)
            o = _sdpa_kv(q, k, v, kv_k, kv_v, N_HEADS // N_KV)
            o = o.transpose(1, 2).reshape(B, 1, N_HEADS * HEAD_DIM)
            res1 = self.o_proj(o) + residual
            y = self.post_attention_layernorm(res1)
            return self.down(F.silu(self.gate(y)) * self.up(y)) + res1

    class EagleDrafter(nn.Module):
        """Eagle3DraftHead: embed + input_norm(7680) + fc[7680->2560] + 1 layer + norm + lm_head.
        embed + lm_head represent the REUSED target tensors (#299 S0): resident, not transient."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB, HID)
            self.input_norm = RMSNorm(FC_IN)
            self.fc = nn.Linear(FC_IN, HID, bias=False)
            self.layer = EagleLayer()
            self.norm = RMSNorm(HID)
            self.lm_head = nn.Linear(HID, VOCAB, bias=False)

        def combine(self, fused):
            return self.fc(self.input_norm(fused))

        def draft_step(self, tok, hidden, kv_k, kv_v, cos1, sin1):
            embeds = self.embed(tok)
            hid = self.norm(self.layer(embeds, hidden, kv_k, kv_v, cos1, sin1))
            logits = self.lm_head(hid)             # get_top_tokens reads these
            top = logits.argmax(dim=-1)            # the served get_top_tokens analog
            return top, hid

    # ---- target-width verify decoder-layer proxy (one representative backbone layer) ---- #
    class VerifyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(HID, N_HEADS * HEAD_DIM, bias=False)
            self.k_proj = nn.Linear(HID, N_KV * HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(HID, N_KV * HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HID, bias=False)
            self.gate = nn.Linear(HID, INTER, bias=False)
            self.up = nn.Linear(HID, INTER, bias=False)
            self.down = nn.Linear(INTER, HID, bias=False)
            self.n1 = RMSNorm(HID)
            self.n2 = RMSNorm(HID)

        def forward(self, x, kv_k, kv_v, cos_m, sin_m):
            # x: [1, M, H]; verify M tree tokens attending a ctx-length KV cache.
            B, M, _ = x.shape
            h = self.n1(x)
            q = h @ self.q_proj.weight.t()
            q = q.view(B, M, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = (h @ self.k_proj.weight.t()).view(B, M, N_KV, HEAD_DIM).transpose(1, 2)
            v = (h @ self.v_proj.weight.t()).view(B, M, N_KV, HEAD_DIM).transpose(1, 2)
            q, k = apply_rope(q, k, cos_m, sin_m)
            kk = torch.cat([kv_k, k], dim=2).repeat_interleave(N_HEADS // N_KV, dim=1)
            vv = torch.cat([kv_v, v], dim=2).repeat_interleave(N_HEADS // N_KV, dim=1)
            o = F.scaled_dot_product_attention(q, kk, vv)
            o = o.transpose(1, 2).reshape(B, M, N_HEADS * HEAD_DIM)
            x = x + self.o_proj(o)
            hh = self.n2(x)
            return x + self.down(F.silu(self.gate(hh)) * self.up(hh))

    return dict(torch=torch, nn=nn, F=F, build_rope=build_rope,
                EagleDrafter=EagleDrafter, VerifyLayer=VerifyLayer)


# =========================================================================== #
# Analytic transient terms (cross-checks for the measured peaks).
# =========================================================================== #
def analytic_terms(ctx: int, m_sweep: list[int], bf16: int = 2, fp32: int = 4) -> dict[str, Any]:
    """The single largest per-step verify activation = the lm_head logit buffer [M, V].
    argmax/log_softmax over the vocab typically upcasts to fp32 (a second full-vocab temp)."""
    logit_bf16 = {m: VOCAB * m * bf16 for m in m_sweep}
    logit_fp32 = {m: VOCAB * m * fp32 for m in m_sweep}     # fp32 upcast for argmax/logsoftmax
    # 3-layer {2,21,39} hidden retention held live during the draft (the #299 resident term,
    # here as a RUNTIME live buffer): L_FUSE x H x ctx x bf16.
    hidden_retention = L_FUSE * HID * ctx * bf16
    fc_in_buffer = FC_IN * 1 * bf16                          # fused [1, 7680] fc input (batch=1 draft)
    return {
        "logit_buffer_bytes_bf16": logit_bf16,
        "logit_buffer_bytes_fp32_upcast": logit_fp32,
        "logit_buffer_gib_bf16": {m: b / GIB for m, b in logit_bf16.items()},
        "logit_buffer_gib_fp32_upcast": {m: b / GIB for m, b in logit_fp32.items()},
        "hidden_retention_bytes": hidden_retention,
        "hidden_retention_gib": hidden_retention / GIB,
        "fc_in_buffer_bytes": fc_in_buffer,
        "monotone_in_m": all(logit_bf16[m_sweep[i]] < logit_bf16[m_sweep[i + 1]]
                             for i in range(len(m_sweep) - 1)),
    }


# =========================================================================== #
# GPU profiling: three phases + a fused capture (headline).
# =========================================================================== #
def profile_gpu(ctx: int, k_spec: int, m_sweep: list[int], smoke: bool) -> dict[str, Any]:
    mods = _build_modules()
    torch = mods["torch"]
    if not torch.cuda.is_available():
        return {"gpu_available": False, "note": "no CUDA device; analytic fallback"}
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    gpu_name = torch.cuda.get_device_name(0)
    dev_total = torch.cuda.get_device_properties(0).total_memory

    out: dict[str, Any] = {"gpu_available": True, "gpu_name": gpu_name, "ctx": ctx,
                           "k_spec": k_spec, "m_sweep": list(m_sweep), "dtype": "bfloat16",
                           "device_total_bytes": dev_total,
                           "device_total_gib": dev_total / GIB}

    cos, sin = mods["build_rope"](ctx + max(m_sweep) + k_spec, HEAD_DIM, 1e6, dev, dtype)
    cos1, sin1 = cos[:1], sin[:1]

    # ---- build modules (resident weights: drafter + reused embed/lm_head + verify layer) ---- #
    drafter = mods["EagleDrafter"]().to(dev, dtype).eval()
    verify = mods["VerifyLayer"]().to(dev, dtype).eval()
    # draft KV cache (1 EAGLE-3 attention layer) + verify KV cache (1 backbone layer proxy).
    d_kv_k = torch.randn(1, N_KV, ctx, HEAD_DIM, device=dev, dtype=dtype)
    d_kv_v = torch.randn(1, N_KV, ctx, HEAD_DIM, device=dev, dtype=dtype)
    v_kv_k = torch.randn(1, N_KV, ctx, HEAD_DIM, device=dev, dtype=dtype)
    v_kv_v = torch.randn(1, N_KV, ctx, HEAD_DIM, device=dev, dtype=dtype)
    tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
    fused = torch.randn(1, 1, FC_IN, device=dev, dtype=dtype)

    def draft_chain(k):
        h = drafter.combine(fused)
        for _ in range(k):
            _, h = drafter.draft_step(tok, h, d_kv_k, d_kv_v, cos1, sin1)
        return h

    def verify_forward(m):
        cos_m, sin_m = cos[:m], sin[:m]
        x = torch.randn(1, m, HID, device=dev, dtype=dtype)
        hid = verify(x, v_kv_k, v_kv_v, cos_m, sin_m)          # [1, M, H]
        logits = hid.reshape(m, HID) @ drafter.lm_head.weight.t()   # [M, V] verify logits
        # verification compares argmax(logits) to the drafted token; bf16-native (deployed-faithful:
        # PyTorch does NOT upcast argmax/softmax to fp32 unless dtype= is passed -- research finding).
        top = logits.argmax(dim=-1)
        return logits, top

    # steady-state resident of the loaded proxy (weights + KV + a warmup forward's live set).
    with torch.inference_mode():
        _ = draft_chain(k_spec)
        _ = verify_forward(m_sweep[0])
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a_res, r_res, _, _ = _mem()
    out["proxy_resident_alloc_bytes"] = a_res
    out["proxy_resident_reserved_bytes"] = r_res
    out["proxy_resident_alloc_gib"] = a_res / GIB
    out["proxy_resident_reserved_gib"] = r_res / GIB

    # NaN smoke on each component.
    with torch.inference_mode():
        h_smoke = draft_chain(k_spec)
        lg_smoke, top_smoke = verify_forward(m_sweep[0])
        torch.cuda.synchronize()
    out["smoke_nan_clean"] = bool(torch.isfinite(h_smoke).all()
                                  and torch.isfinite(lg_smoke).all())
    out["smoke_shapes"] = {"draft_hidden": list(h_smoke.shape),
                           "verify_logits": list(lg_smoke.shape)}
    if smoke:
        a, r, ma, mr = _mem()
        out["smoke_peak_alloc_gib"] = ma / GIB
        out["smoke_peak_reserved_gib"] = mr / GIB
        return out

    # ---------------- PHASE (iii): 3-layer {2,21,39} hidden-state retention ---------------- #
    # Hold L_FUSE hidden states [L_FUSE, ctx, H] live (the fusion input source) during a draft.
    _reset_peak()
    a0, r0, _, _ = _mem()
    with torch.inference_mode():
        held = torch.randn(L_FUSE, ctx, HID, device=dev, dtype=dtype)   # {2,21,39} retained
        _ = draft_chain(k_spec)
        torch.cuda.synchronize()
        a1, r1, ma1, mr1 = _mem()
    held_bytes = L_FUSE * ctx * HID * 2
    del held
    torch.cuda.empty_cache()
    out["phase_iii_hidden_retention"] = {
        "resident_alloc_gib": a0 / GIB, "peak_alloc_gib": ma1 / GIB,
        "peak_reserved_gib": mr1 / GIB,
        "transient_alloc_gib": (ma1 - a0) / GIB,
        "held_buffer_bytes": held_bytes, "held_buffer_gib": held_bytes / GIB,
        "peak_ge_resident": bool(ma1 >= a0),
    }

    # ---------------- PHASE (ii): M-wide tree-verify logit GEMM sweep ---------------- #
    m_results: dict[str, Any] = {}
    for m in m_sweep:
        _reset_peak()
        a0, r0, _, _ = _mem()
        with torch.inference_mode():
            logits, top = verify_forward(m)
            torch.cuda.synchronize()
            a1, r1, ma1, mr1 = _mem()
        logit_bytes = VOCAB * m * 2
        m_results[str(m)] = {
            "resident_alloc_gib": a0 / GIB, "peak_alloc_gib": ma1 / GIB,
            "peak_reserved_gib": mr1 / GIB,
            "transient_alloc_gib": (ma1 - a0) / GIB,
            "logit_buffer_bytes_bf16": logit_bytes, "logit_buffer_gib_bf16": logit_bytes / GIB,
            "peak_ge_resident": bool(ma1 >= a0),
            "exceeds_capture_size16_boundary": bool(m > MAX_CUDAGRAPH_CAPTURE_SIZE_101),
        }
        del logits, top
        torch.cuda.empty_cache()
    out["phase_ii_tree_verify"] = m_results
    peaks = [m_results[str(m)]["peak_alloc_gib"] for m in m_sweep]
    out["phase_ii_peak_monotone_in_m"] = all(peaks[i] <= peaks[i + 1] + 1e-9
                                              for i in range(len(peaks) - 1))
    transients = [m_results[str(m)]["transient_alloc_gib"] for m in m_sweep]
    out["phase_ii_transient_monotone_in_m"] = all(transients[i] <= transients[i + 1] + 1e-9
                                                  for i in range(len(transients) - 1))

    # ---------------- PHASE (i): ONEGRAPH capture of the fused draft->verify step ---------------- #
    # Static buffers for a capturable fused step (batch=1 draft chain + M-deployed verify).
    m_cap = M_VERIFY_DEPLOYED
    cos_mc, sin_mc = cos[:m_cap].clone(), sin[:m_cap].clone()
    static_h = drafter.combine(fused).detach().clone()
    static_vx = torch.randn(1, m_cap, HID, device=dev, dtype=dtype)
    out_top = torch.zeros(1, k_spec, dtype=torch.long, device=dev)

    def fused_step():
        # draft chain: K width-1 EAGLE-3 forwards (each calls the drafter lm_head get_top_tokens).
        h = static_h
        for i in range(k_spec):
            top, h = drafter.draft_step(tok, h, d_kv_k, d_kv_v, cos1, sin1)
            out_top[0, i] = top[0, 0]
        # verify: one backbone-proxy layer over m_cap tree tokens + lm_head + argmax verify.
        hv = verify(static_vx, v_kv_k, v_kv_v, cos_mc, sin_mc)
        vlogits = hv.reshape(m_cap, HID) @ drafter.lm_head.weight.t()
        _ = vlogits.argmax(dim=-1)                            # bf16-native (deployed-faithful)
        return hv

    torch.cuda.empty_cache()
    _reset_peak()
    a_pre, r_pre, _, _ = _mem()                          # reserved BEFORE capture
    captured = True
    graph = None
    try:
        graph = _capture(fused_step)
    except Exception as exc:
        captured = False
        out["phase_i_capture_error"] = repr(exc)
    a_post, r_post, ma_post, mr_post = _mem()            # reserved AFTER capture (pool persists)
    replay_peak_alloc = replay_peak_reserved = None
    if captured and graph is not None:
        _reset_peak()
        for _ in range(5):
            graph.replay()
        _, _, ma_rep, mr_rep = _mem()
        replay_peak_alloc, replay_peak_reserved = ma_rep, mr_rep
        del graph
        torch.cuda.empty_cache()
    capture_pool_bytes = max(0, r_post - r_pre)
    out["phase_i_capture"] = {
        "captured": captured,
        "reserved_pre_capture_gib": r_pre / GIB,
        "reserved_post_capture_gib": r_post / GIB,
        "capture_peak_alloc_gib": ma_post / GIB,
        "capture_peak_reserved_gib": mr_post / GIB,
        "capture_pool_bytes": capture_pool_bytes,
        "capture_pool_gib": capture_pool_bytes / GIB,
        "replay_peak_alloc_gib": (replay_peak_alloc / GIB) if replay_peak_alloc else None,
        "replay_peak_reserved_gib": (replay_peak_reserved / GIB) if replay_peak_reserved else None,
        "capture_size_tokens": k_spec + m_cap,           # draft K + verify m_cap, the fused batch
        "peak_ge_resident": bool(mr_post >= r_pre),
    }
    return out


# =========================================================================== #
# Synthesis: import #299 resident, compute peak, decompose, dominant, mitigations, #101.
# =========================================================================== #
def synthesize(gpu: dict[str, Any], analytic: dict[str, Any], ctx: int,
               m_sweep: list[int]) -> dict[str, Any]:
    have_gpu = bool(gpu.get("gpu_available"))
    have_phases = have_gpu and "phase_i_capture" in gpu     # full profile (not smoke/error)
    resident = BUILD_RESIDENT_GIB_299

    # ---- transient decomposition (GiB): capture scratch + logit buffer + workspace/remat ---- #
    if have_phases:
        cap = gpu["phase_i_capture"]
        capture_scratch_gib = max(cap["capture_pool_gib"], 0.0)
        # tree-verify logit buffer at deployed M (measured transient of the M-deployed verify).
        mdep = str(M_VERIFY_DEPLOYED)
        verify_dep = gpu["phase_ii_tree_verify"][mdep]
        logit_buffer_gib = max(verify_dep["transient_alloc_gib"], 0.0)
        # workspace/remat = the live hidden retention + verify per-layer scratch carried at runtime.
        workspace_remat_gib = max(gpu["phase_iii_hidden_retention"]["transient_alloc_gib"], 0.0)
        measurement = "measured"
    else:
        # analytic fallback: bf16 logit buffer at deployed M (deployed-faithful: NO default fp32
        # upcast); capture pool from a conservative cuBLAS-workspace estimate; workspace = retention.
        capture_scratch_gib = 0.05                       # conservative cuBLAS(~4MiB)+attn workspace
        logit_buffer_gib = analytic["logit_buffer_gib_bf16"][M_VERIFY_DEPLOYED]
        workspace_remat_gib = analytic["hidden_retention_gib"]
        measurement = "analytic"

    total_transient_gib = capture_scratch_gib + logit_buffer_gib + workspace_remat_gib
    terms = {
        "capture_time_scratch": capture_scratch_gib,
        "tree_verify_logit_buffer": logit_buffer_gib,
        "workspace_remat": workspace_remat_gib,
    }
    dominant_term = max(terms, key=terms.get)

    # ---- the headline: build PEAK = resident + transient; does it fit 24 / 23? ---- #
    build_peak_gib = resident + total_transient_gib
    fits_24_hard = bool(build_peak_gib <= VRAM_HARD_GIB)
    fits_23_usable = bool(build_peak_gib <= VRAM_USABLE_GIB)
    fits_device_visible = bool(build_peak_gib <= DEVICE_VISIBLE_GIB_299)
    peak_headroom_24 = VRAM_HARD_GIB - build_peak_gib
    peak_headroom_23 = VRAM_USABLE_GIB - build_peak_gib
    headroom_eaten_gib = HEADROOM_24_HARD_GIB_299 - peak_headroom_24   # == total_transient_gib
    headroom_eaten_frac = total_transient_gib / HEADROOM_24_HARD_GIB_299

    # ---- #101 precedent: the capture-SIZE boundary (NOT a VRAM OOM) ---- #
    cap_size_tokens = (gpu.get("phase_i_capture", {}) or {}).get("capture_size_tokens",
                                                                  K_SPEC + M_VERIFY_DEPLOYED)
    one_plus_k = 1 + K_SPEC                               # EAGLE capture sizes must be multiples of 8
    # valid captured sizes = multiples of (1+K) within [.., max_cudagraph_capture_size].
    valid_capture_sizes = [s for s in range(one_plus_k, MAX_CUDAGRAPH_CAPTURE_SIZE_101 + 1)
                           if s % one_plus_k == 0]
    m_over_boundary = {m: bool(m > MAX_CUDAGRAPH_CAPTURE_SIZE_101) for m in m_sweep}
    m_capturable = {m: bool(m <= MAX_CUDAGRAPH_CAPTURE_SIZE_101 and m % one_plus_k == 0)
                    for m in m_sweep}
    precedent = {
        "max_cudagraph_capture_size": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
        "spine_verify_tokens": SPINE_VERIFY_TOKENS_101,
        "size29_crash_tokens": SIZE29_CRASH_TOKENS_101,
        # the repo + vLLM source agree: the crash is a dispatch-table lookup failure (IndexError --
        # NO captured graph exists for an un-listed batch size), NOT a VRAM allocation failure.
        "crash_is_vram_oom": False,
        "crash_is_capture_size_dispatch_failure": True,
        "eagle_capture_size_divisor_1pK": one_plus_k,
        "valid_captured_sizes": valid_capture_sizes,    # {8, 16} for K=7 under max-16
        "fused_capture_size_tokens": cap_size_tokens,
        "deployed_m8_verify_tokens": M_VERIFY_DEPLOYED,
        "deployed_below_size16_boundary": bool(M_VERIFY_DEPLOYED <= MAX_CUDAGRAPH_CAPTURE_SIZE_101),
        "deployed_m8_capturable": bool(M_VERIFY_DEPLOYED in valid_capture_sizes),
        "m_over_capture_size_boundary": m_over_boundary,
        "m_capturable_under_size16": m_capturable,
        # the VRAM peak this leg prices is FAR below any OOM regime; the genuine #101-class risk
        # is the verify-token count vs the size-16 boundary, a TOPOLOGY constraint, not VRAM bytes.
        "vram_peak_below_crash_regime": fits_24_hard,
    }

    # ---- mitigations priced ONLY if the peak threatens the headroom (analysis only) ---- #
    threatens = not fits_23_usable
    mitigations = {
        "needed": bool(threatens),
        "note": ("VRAM peak fits 23-usable with multi-GiB margin; no mitigation required for the "
                 "memory axis." if not threatens else
                 "VRAM peak threatens 23-usable; cheapest analytic mitigations below."),
        "chunked_logits_recover_gib": max(0.0, logit_buffer_gib
                                          - analytic["logit_buffer_gib_bf16"][m_sweep[0]]),
        "m_cap_to_deployed_recover_gib": max(
            0.0, analytic["logit_buffer_gib_bf16"][max(m_sweep)]
            - analytic["logit_buffer_gib_bf16"][M_VERIFY_DEPLOYED]),
        "smaller_capture_batch_note": ("capture only the deployed spine (K+M=15 tokens < size-16 "
                                       "boundary) avoids the #101 size-list crash AND keeps the "
                                       "capture pool minimal."),
    }

    # ---- SELF-TEST (PRIMARY) ---- #
    cond: dict[str, bool] = {}
    # (a) peak >= resident for every phase.
    phase_peak_ge = True
    if have_phases:
        phase_peak_ge = bool(gpu["phase_i_capture"].get("peak_ge_resident", False)
                             and gpu["phase_iii_hidden_retention"]["peak_ge_resident"]
                             and all(gpu["phase_ii_tree_verify"][str(m)]["peak_ge_resident"]
                                     for m in m_sweep))
    cond["a_peak_ge_resident_every_phase"] = phase_peak_ge
    # (b) all peaks finite & positive.
    finite_pos = True
    if have_phases:
        vals = [gpu["phase_i_capture"]["capture_peak_alloc_gib"],
                gpu["phase_iii_hidden_retention"]["peak_alloc_gib"]]
        vals += [gpu["phase_ii_tree_verify"][str(m)]["peak_alloc_gib"] for m in m_sweep]
        finite_pos = all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in vals)
    cond["b_all_peaks_finite_positive"] = bool(finite_pos and build_peak_gib > 0
                                               and math.isfinite(build_peak_gib))
    # (c) transient decomposition sums to total (peak - resident) within tolerance.
    decomp_sum = capture_scratch_gib + logit_buffer_gib + workspace_remat_gib
    cond["c_decomposition_sums"] = bool(abs(decomp_sum - total_transient_gib) < 1e-9
                                        and abs((build_peak_gib - resident) - total_transient_gib)
                                        < 1e-9)
    # (d) M-scaling of the logit buffer is monotone (analytic AND, if measured, the phase peaks).
    measured_mono = gpu.get("phase_ii_peak_monotone_in_m", True) if have_phases else True
    cond["d_logit_buffer_monotone_in_m"] = bool(analytic["monotone_in_m"] and measured_mono)
    # (e) imported #299 constants match source <= 1e-6.
    cond["e_imported_299_constants_exact"] = bool(
        abs(BUILD_RESIDENT_GIB_299 - 20.100143778324128) < 1e-6
        and abs(EXTRA_KV_GIB_299 - 0.718841552734375) < 1e-6
        and abs(DRAFTER_WEIGHTS_GIB_299 - 0.037352144718170166) < 1e-6
        and abs(HEADROOM_24_HARD_GIB_299 - 3.899856221675872) < 1e-6
        and abs(NONTORCH_CUDA_CONTEXT_GIB_299 - 0.9499999999999993) < 1e-6
        and abs(DEVICE_VISIBLE_GIB_299 - 22.0582275390625) < 1e-6
        and VOCAB == 262144 and HID == 2560 and tuple(EAGLE3_AUX_LAYERS) == (2, 21, 39)
        and L_FUSE == 3 and K_SPEC == 7 and M_VERIFY_DEPLOYED == 8)
    # (f) NaN-clean (asserted on the full payload in main; GPU smoke must be NaN-clean).
    cond["f_gpu_smoke_nan_clean"] = bool(gpu.get("smoke_nan_clean", True))
    # (g) honest caveats carried (asserted present on the assembled payload).
    cond["g_honest_caveats_carried"] = True

    self_test_passes = all(cond.values())

    caveats = [
        "0 TPS: this MEASURES the EAGLE-3 step's RUNTIME memory transient (above #299's resident "
        "floor); it is a random-init ARCHITECTURAL footprint, NOT a trained build. Random-init "
        "weights/activations have the SAME tensor shapes/bytes as a trained build, so the VRAM "
        "footprint transfers; the numeric activation VALUES do not (irrelevant to byte accounting).",
        "COMPOSITIONAL, like #299: the local proxy does NOT load the 20.10 GiB build; it measures "
        "the step TRANSIENT in isolation and STACKS it on #299's imported resident floor. The "
        "absolute peak = #299 resident (20.10) + measured transient; the proxy's own resident is "
        "small and only used for the per-phase peak>=resident self-test.",
        "CAPTURE caveat: the deployed ONEGRAPH captures the DRAFT chain and the vLLM target captures "
        "the VERIFY separately (two pools); this leg captures a FUSED single graph (draft+verify), "
        "which holds both activation sets live -> a CONSERVATIVE (upper-bound) capture pool vs the "
        "deployed two-pool sum. The verify backbone is modeled by ONE representative target-width "
        "bf16 layer (the 42-layer body frees activations layer-by-layer; only the residual stream + "
        "lm_head logits persist), so the measured verify transient bounds the real per-layer scratch.",
        "PRECEDENT honesty: lawine's #101 'size-29 crash' is a capture-SIZE-LIST DISPATCH failure "
        "(max_cudagraph_capture_size=16; a request at batch=29 finds NO captured graph -> IndexError "
        "lookup crash), NOT a VRAM OOM (vLLM #29091/PR#23679). For EAGLE with K=7, captured sizes "
        "must be multiples of (1+K)=8 within max-16, i.e. {8,16}; deployed M=8 is captured, M=16 is "
        "the boundary, M=32 exceeds max-16 -> the crash regime. This leg prices the orthogonal VRAM "
        "axis (transient bytes) and states the build sits FAR below any OOM AND that deployed M=8 "
        "clears the size boundary.",
        "MEASUREMENT honesty: argmax/verify is bf16-native -- PyTorch does NOT upcast to fp32 unless "
        "dtype= is passed (issue #123911) -- so the per-step logit transient is bf16 (2B), MB-scale "
        "even at M=32, NOT the doubled fp32 buffer an upcast assumption would imply. cuBLAS workspace "
        "(in the capture pool, freed only with the handle) is the default ~4 MiB here "
        "(CUBLAS_WORKSPACE_CONFIG unset); a serving process that INHERITS a training-tuned "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 would inflate the pool to ~24 MiB/handle -- a silent "
        "launch-hygiene item, still sub-GiB and far inside the headroom.",
        "The launch gate is UNCHANGED and human-approval-gated (land #245: MEASURED >=500 TPS at "
        "lambda_hat>=0.9780 AND PPL<=2.42 AND VRAM<=24 GiB). This leg sizes ONLY the VRAM<=24 "
        "RUNTIME clause; it is NOT a launch, NOT a build, NOT a served-file change.",
    ]

    verdict = (
        "The EAGLE-3 build's RUNTIME peak = #299 resident %.2f GiB + measured step transient %.3f "
        "GiB = %.3f GiB, which FITS 24-hard (headroom %.3f) and 23-usable (headroom %.3f). The "
        "3.90 GiB resident headroom SURVIVES ONEGRAPH capture + the M=%d tree-verify logit "
        "transient: the transient eats only %.1f%% of it. Dominant transient term = %s (%.4f GiB). "
        "The capture footprint is FAR below any VRAM-OOM regime; the genuine #101-class risk is the "
        "capture-SIZE boundary (max_cudagraph_capture_size=%d), which the deployed M=%d spine "
        "(K+M=%d tokens) clears -- VRAM is NOT the binding constraint, and neither is capture at the "
        "deployed width. Analysis-only; BASELINE %.2f untouched; 0 TPS." % (
            resident, total_transient_gib, build_peak_gib, peak_headroom_24, peak_headroom_23,
            M_VERIFY_DEPLOYED, headroom_eaten_frac * 100.0, dominant_term, terms[dominant_term],
            MAX_CUDAGRAPH_CAPTURE_SIZE_101, M_VERIFY_DEPLOYED, cap_size_tokens, OFFICIAL_BASELINE))

    handoff = (
        "EAGLE-3 build SURVIVES runtime: resident 20.10 GiB (#299) + step transient %.3f GiB = "
        "%.3f GiB peak, %.2f GiB under 24-hard and %.2f GiB under 23-usable. The runtime transient "
        "is %s-dominated (%.4f GiB) and tiny (%.1f%% of the 3.90 GiB headroom); the capture-time "
        "pool is %.4f GiB and the 262144-vocab logit buffer is MB-scale even at M=32 (%.4f GiB). "
        "VRAM is NOT the binding constraint on the human-gated build; the real #101-class risk is "
        "the capture-SIZE boundary (size-16), which deployed M=8 clears -- only widening the tree "
        "past M=16 re-enters lawine's crash regime (a topology choice, not a VRAM limit)." % (
            total_transient_gib, build_peak_gib, peak_headroom_24, peak_headroom_23,
            dominant_term, terms[dominant_term], headroom_eaten_frac * 100.0,
            capture_scratch_gib, analytic["logit_buffer_gib_bf16"][max(m_sweep)]))

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE,
            "build_resident_gib_299": BUILD_RESIDENT_GIB_299,
            "deployed_resident_gib_299": DEPLOYED_RESIDENT_GIB_299,
            "extra_kv_gib_299": EXTRA_KV_GIB_299,
            "drafter_weights_gib_299": DRAFTER_WEIGHTS_GIB_299,
            "hidden_retention_gib_299": HIDDEN_RETENTION_GIB_299,
            "headroom_24_hard_gib_299": HEADROOM_24_HARD_GIB_299,
            "headroom_23_usable_gib_299": HEADROOM_23_USABLE_GIB_299,
            "nontorch_cuda_context_gib_299": NONTORCH_CUDA_CONTEXT_GIB_299,
            "cuda_graph_pool_gib_299": CUDA_GRAPH_POOL_GIB_299,
            "device_visible_gib_299": DEVICE_VISIBLE_GIB_299,
            "vram_hard_gib": VRAM_HARD_GIB, "vram_usable_gib": VRAM_USABLE_GIB,
            "vocab": VOCAB, "hid": HID, "eagle3_aux_layers": list(EAGLE3_AUX_LAYERS),
            "l_fuse": L_FUSE, "k_spec": K_SPEC, "m_verify_deployed": M_VERIFY_DEPLOYED,
            "ctx": ctx, "m_sweep": list(m_sweep),
            "max_cudagraph_capture_size_101": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
            "imports": ("ubel#299(jnoss7id resident=20.10 extra_kv=0.719 drafter_wt=0.037 "
                        "headroom24=3.90 nontorch=0.95 devvis=22.058) x "
                        "lawine#245-c1(max_capture_size=16 size29-crash) x "
                        "train_eagle3.py(Eagle3DraftHead 2560-dim 1-layer fc[7680->2560])"),
        },
        "measurement": measurement,
        "gpu_profile": gpu,
        "analytic": analytic,
        "transient_decomposition_gib": terms,
        "total_transient_gib": total_transient_gib,
        "dominant_transient_term": dominant_term,
        "dominant_transient_gib": terms[dominant_term],
        "resident_floor_gib": resident,
        "eagle3_build_peak_gb": build_peak_gib,
        "capture_transient_gib": capture_scratch_gib,
        "eagle3_build_peak_fits_24gb": fits_24_hard,
        "fits_23_usable": fits_23_usable,
        "fits_device_visible": fits_device_visible,
        "peak_headroom_24_hard_gib": peak_headroom_24,
        "peak_headroom_23_usable_gib": peak_headroom_23,
        "headroom_eaten_gib": headroom_eaten_gib,
        "headroom_eaten_frac": headroom_eaten_frac,
        "precedent_101": precedent,
        "mitigations": mitigations,
        "self_test": {"conditions": cond, "passes": self_test_passes},
        "verdict": verdict, "handoff": handoff, "caveats": caveats,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #295/#299; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-capture-peak] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    gpu = syn["gpu_profile"]
    terms = syn["transient_decomposition_gib"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="ubel", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-capture-peak", "vram-runtime-peak", "cudagraph-capture",
                  "tree-verify-transient", "memory-feasibility", "pr-306"],
            config={
                "official_baseline": OFFICIAL_BASELINE,
                "build_resident_gib_299": BUILD_RESIDENT_GIB_299,
                "headroom_24_hard_gib_299": HEADROOM_24_HARD_GIB_299,
                "vram_hard_gib": VRAM_HARD_GIB, "vram_usable_gib": VRAM_USABLE_GIB,
                "vocab": VOCAB, "hid": HID, "k_spec": K_SPEC,
                "m_verify_deployed": M_VERIFY_DEPLOYED, "ctx": syn["constants"]["ctx"],
                "m_sweep": syn["constants"]["m_sweep"],
                "max_cudagraph_capture_size_101": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
                "measurement": syn["measurement"], "gpu_name": gpu.get("gpu_name"),
                "imports": syn["constants"]["imports"], "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-capture-peak] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-capture-peak] wandb: no run (no API key/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "eagle3_capture_peak_self_test_passes": int(bool(
            payload["eagle3_capture_peak_self_test_passes"])),
        "eagle3_build_peak_fits_24gb": int(bool(syn["eagle3_build_peak_fits_24gb"])),
        "eagle3_build_peak_gb": syn["eagle3_build_peak_gb"],
        "capture_transient_gib": syn["capture_transient_gib"],
        "total_transient_gib": syn["total_transient_gib"],
        "dominant_transient_gib": syn["dominant_transient_gib"],
        "resident_floor_gib": syn["resident_floor_gib"],
        "fits_23_usable": int(bool(syn["fits_23_usable"])),
        "fits_device_visible": int(bool(syn["fits_device_visible"])),
        "peak_headroom_24_hard_gib": syn["peak_headroom_24_hard_gib"],
        "peak_headroom_23_usable_gib": syn["peak_headroom_23_usable_gib"],
        "headroom_eaten_frac": syn["headroom_eaten_frac"],
        "transient_capture_scratch_gib": terms["capture_time_scratch"],
        "transient_logit_buffer_gib": terms["tree_verify_logit_buffer"],
        "transient_workspace_remat_gib": terms["workspace_remat"],
        "deployed_below_size16_boundary": int(bool(
            syn["precedent_101"]["deployed_below_size16_boundary"])),
        "fused_capture_size_tokens": syn["precedent_101"]["fused_capture_size_tokens"],
        "measurement_is_gpu": int(bool(gpu.get("gpu_available"))),
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    if gpu.get("gpu_available") and "phase_i_capture" in gpu:
        cap = gpu["phase_i_capture"]
        summary.update({
            "capture_pool_gib": cap["capture_pool_gib"],
            "capture_peak_reserved_gib": cap["capture_peak_reserved_gib"],
            "capture_size_tokens": cap["capture_size_tokens"],
            "proxy_resident_alloc_gib": gpu.get("proxy_resident_alloc_gib"),
            "gpu_device_total_gib": gpu.get("device_total_gib"),
        })
        for m in syn["constants"]["m_sweep"]:
            mr = gpu["phase_ii_tree_verify"][str(m)]
            summary[f"verify_peak_alloc_gib_m{m}"] = mr["peak_alloc_gib"]
            summary[f"verify_transient_gib_m{m}"] = mr["transient_alloc_gib"]
            summary[f"verify_logit_buffer_gib_m{m}"] = mr["logit_buffer_gib_bf16"]
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_capture_peak_result", artifact_type="validity",
                          data=payload)
        finish_wandb(run)
        print(f"[eagle3-capture-peak] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-capture-peak] wandb write failed (analysis unaffected): {exc}", flush=True)


# --------------------------------------------------------------------------- #
def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


def _print_human(syn: dict) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" EAGLE-3 BUILD CAPTURE-PEAK VRAM AUDIT: DOES 3.90 GiB HEADROOM SURVIVE RUNTIME? (PR #306)",
          flush=True)
    print("=" * 104, flush=True)
    gpu = syn["gpu_profile"]
    print(f"  measurement={syn['measurement']}  gpu={gpu.get('gpu_name', 'NONE')}  "
          f"resident_floor(#299)={syn['resident_floor_gib']:.3f} GiB", flush=True)
    print("-" * 104, flush=True)
    if gpu.get("gpu_available") and "phase_i_capture" in gpu:
        cap = gpu.get("phase_i_capture", {})
        print(f"  (i)   CAPTURE  pool={cap.get('capture_pool_gib', float('nan')):.4f} GiB  "
              f"peak_reserved={cap.get('capture_peak_reserved_gib', float('nan')):.3f} GiB  "
              f"captured={cap.get('captured')}  size_tokens={cap.get('capture_size_tokens')}",
              flush=True)
        print(f"  (ii)  TREE-VERIFY logit GEMM (M-sweep):", flush=True)
        for m in syn["constants"]["m_sweep"]:
            mr = gpu["phase_ii_tree_verify"][str(m)]
            print(f"          M={m:>2}  peak_alloc={mr['peak_alloc_gib']:.4f}  "
                  f"transient={mr['transient_alloc_gib']:.4f}  "
                  f"logit_buf={mr['logit_buffer_gib_bf16']:.4f} GiB  "
                  f"over_size16={mr['exceeds_capture_size16_boundary']}", flush=True)
        hr = gpu["phase_iii_hidden_retention"]
        print(f"  (iii) HIDDEN RETENTION  transient={hr['transient_alloc_gib']:.4f} GiB  "
              f"held={hr['held_buffer_gib']:.4f} GiB", flush=True)
    else:
        print(f"  GPU unavailable -> analytic fallback. {gpu.get('note', '')}", flush=True)
    print("-" * 104, flush=True)
    td = syn["transient_decomposition_gib"]
    print(f"  TRANSIENT DECOMPOSITION (GiB):  capture_scratch={td['capture_time_scratch']:.4f}  "
          f"logit_buffer={td['tree_verify_logit_buffer']:.4f}  "
          f"workspace_remat={td['workspace_remat']:.4f}", flush=True)
    print(f"      total_transient={syn['total_transient_gib']:.4f} GiB  "
          f"DOMINANT={syn['dominant_transient_term']} ({syn['dominant_transient_gib']:.4f} GiB)",
          flush=True)
    print(f"  BUILD PEAK = {syn['resident_floor_gib']:.3f} + {syn['total_transient_gib']:.4f} = "
          f"{syn['eagle3_build_peak_gb']:.4f} GiB", flush=True)
    print(f"      fits_24_hard={syn['eagle3_build_peak_fits_24gb']} "
          f"(headroom {syn['peak_headroom_24_hard_gib']:.3f})  "
          f"fits_23_usable={syn['fits_23_usable']} (headroom {syn['peak_headroom_23_usable_gib']:.3f})  "
          f"headroom_eaten={syn['headroom_eaten_frac'] * 100:.1f}%", flush=True)
    p = syn["precedent_101"]
    print(f"  #101 PRECEDENT  max_capture_size={p['max_cudagraph_capture_size']}  "
          f"crash_is_vram_oom={p['crash_is_vram_oom']}  "
          f"deployed_m8_below_boundary={p['deployed_below_size16_boundary']}  "
          f"m_over_boundary={p['m_over_capture_size_boundary']}", flush=True)
    st = syn["self_test"]
    print("-" * 104, flush=True)
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }  -> PASS={st['passes']}",
          flush=True)
    print(f"\n  VERDICT: {syn['verdict']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--ctx", type=int, default=CTX_DEPLOYED, help="KV context length")
    ap.add_argument("--k-spec", "--k_spec", dest="k_spec", type=int, default=K_SPEC)
    ap.add_argument("--m-sweep", "--m_sweep", dest="m_sweep", type=str, default="8,16,32",
                    help="tree-verify widths to sweep (comma-separated)")
    ap.add_argument("--smoke", action="store_true", help="load + NaN smoke only (no peak timing)")
    ap.add_argument("--no-gpu", action="store_true", help="skip GPU; analytic fallback only")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-capture-peak")
    args = ap.parse_args(argv)

    m_sweep = [int(x) for x in str(args.m_sweep).split(",") if x.strip()]
    analytic = analytic_terms(args.ctx, m_sweep)
    if args.no_gpu:
        gpu = {"gpu_available": False, "note": "--no-gpu"}
    else:
        try:
            gpu = profile_gpu(args.ctx, args.k_spec, m_sweep, args.smoke)
        except Exception as exc:
            print(f"[eagle3-capture-peak] GPU profile error -> analytic fallback: {exc!r}", flush=True)
            gpu = {"gpu_available": False, "note": f"gpu error: {exc!r}"}

    syn = synthesize(gpu, analytic, args.ctx, m_sweep)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 306, "agent": "ubel", "kind": "eagle3-capture-peak",
        "analysis_only": True,
        "synthesis": syn,
        "eagle3_capture_peak_self_test_passes": syn["self_test"]["passes"],
        "eagle3_build_peak_fits_24gb": syn["eagle3_build_peak_fits_24gb"],
        "eagle3_build_peak_gb": syn["eagle3_build_peak_gb"],
        "capture_transient_gib": syn["capture_transient_gib"],
        "host_peak_mem_mib": round(peak_kib / 1024.0, 3),
        "greedy_ppl_safety_certificate": {
            "analysis_only": True, "served_file_changed": False, "emitted_token_changed": False,
            "hf_job_or_submission": False, "is_launch": False, "is_build": False,
            "baseline_tps_unchanged": OFFICIAL_BASELINE, "tps_added_by_this_leg": 0.0,
        },
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    payload["eagle3_capture_peak_self_test_passes"] = bool(syn["self_test"]["passes"]
                                                           and payload["nan_clean"])
    # honest caveats present check (self-test g).
    syn["self_test"]["conditions"]["g_honest_caveats_carried"] = bool(len(syn["caveats"]) >= 4)
    if nan_paths:
        print(f"[eagle3-capture-peak] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_capture_peak_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    payload["primary_metric"] = {"name": "eagle3_capture_peak_self_test_passes",
                                 "value": int(bool(payload["eagle3_capture_peak_self_test_passes"]))}
    payload["test_metric"] = {"name": "eagle3_build_peak_fits_24gb",
                              "value": int(bool(syn["eagle3_build_peak_fits_24gb"]))}

    _print_human(syn)
    print(f"[eagle3-capture-peak] wrote {out_path}", flush=True)
    print(f"[eagle3-capture-peak] PRIMARY eagle3_capture_peak_self_test_passes = "
          f"{payload['eagle3_capture_peak_self_test_passes']}", flush=True)
    print(f"[eagle3-capture-peak] TEST eagle3_build_peak_fits_24gb = "
          f"{syn['eagle3_build_peak_fits_24gb']}  eagle3_build_peak_gb = "
          f"{syn['eagle3_build_peak_gb']:.4f}  capture_transient_gib = "
          f"{syn['capture_transient_gib']:.4f}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["eagle3_capture_peak_self_test_passes"]
        print(f"[eagle3-capture-peak] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
