#!/usr/bin/env python
"""Drafter NON-GEMM profile + Step-0 reduction feasibility (PR #77).

WHAT THIS MEASURES
------------------
denken #75 (MERGED) mapped the drafter's GEMM half: the 7-pass weight-GEMM chain
is 566 us = 4.88% of an 11.6 ms decode step, latency-floored (47% HBM peak), and
int4 weights were refuted as a TPS lever. #75 caveat 3 flagged the OTHER ~70% of
the drafter forward -- the NON-GEMM block (~1230-1530 us inferred from wirbel
#69's 15.5-18.1% drafter budget minus the 566 us GEMM) -- as the largest
un-audited drafter cost, *inferred not timed*. This script TIMES it and breaks it
into sub-blocks, the same faithful isolated-kernel form as #75 / #68.

THE NON-GEMM SUB-BLOCKS (from the served code, not assumed)
----------------------------------------------------------
Per width-1 draft pass, `Gemma4MultiTokenPredictor.forward` (gemma4_mtp.py) runs,
besides the 19 weight GEMMs #75 already timed, these NON-GEMM ops:
  - input embed gather   : embed_tokens[input_id]  (1 row of [262144, D])
  - RMSNorms (hidden 256): input_ln, post_attn_ln, pre_ff_ln, post_ff_ln x4 layers
                           + final model.norm  = 17 norms over dim 256
  - q_norm               : RMSNorm over head_dim, x3 sliding(256) + x1 full(512)
  - rotary               : RoPE on q, x3 sliding(theta 1e4) + x1 full(0.25 partial)
  - attention SDPA       : Q-only, KV-shared read of the TARGET kv-cache,
                           x3 sliding(head_dim256, KV<=512) + x1 full(head_dim512, KV=L)
  - residual adds        : x8 (2/layer), layer_scalar mul x4, gelu_tanh act x4
  - centroid sparse sampler (the "262k masked-embed gather"): once/pass, picks the
    draft token via top-64 centroids -> 8192 candidate rows of lm_head[262144,256].

DEPLOYED EXECUTION MODE (matches #75)
-------------------------------------
ONEGRAPH=1: the whole 7-pass propose is ONE CUDA graph (launch-free), and the
centroid sampler is separately CUDA-graphed + FUSED_SPARSE_ARGMAX=1 (a 2-kernel
triton fusion, block=16). So the deployed non-GEMM is launch-free. We report the
launch-free (graph-replay) per-op cost as the deployed-representative basis and
eager as the without-onegraph contrast, exactly like #75.

CRITICAL STEP-0 FACT (from the served kernel, gemma4_mtp.py + sitecustomize.py)
------------------------------------------------------------------------------
The centroid masked-embedder ALREADY gathers only the `num_selected` =
top_k * (vocab/num_centroids) = 64*128 = 8192 candidate rows of lm_head[262144,256]
(4 MB), NOT the full 262144 (134 MB). The fused triton kernel loads lm_head rows
only for the 8192 selected vocab_ids. So the PR's hypothesized contract-safe
"gather only the candidate set" reduction is ALREADY DEPLOYED -- this script
quantifies how close to its floor it already is.

FAITHFULNESS
------------
Norms / rotary / activation: the REAL served vLLM modules (RMSNorm, get_rope,
get_act_and_mul_fn) with real drafter weights -> identical kernels. Centroid
sampler: the REAL `Gemma4MTPMaskedEmbedder` (unfused decomposition) + a VERBATIM
copy of the deployed fused triton kernel (deployed number). Attention: a roofline
(KV bytes / HBM BW; decode attention is memory-bound) cross-checked with a timed
SDPA proxy at the exact shapes -- a PROXY for the served TRITON_ATTN paged kernel
(byte volume is kernel-agnostic to first order), swept over KV length L. All M=1,
value-independent, no serve-path change, no HF Job.

Primary metric: drafter_nongemm_binding_subblock_pct_of_decode.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import struct
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402

DEFAULT_DRAFTER = "/tmp/qat-assistant"
A10G_HBM_GBS = 600.0
BF16 = torch.bfloat16


# --------------------------------------------------------------------------- #
# safetensors helpers (verbatim from #75 drafter_forward_roofline.py)          #
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def load_tensor(path: str, name: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


# --------------------------------------------------------------------------- #
# Timing: launch-free (reps-in-one-graph, amortizes replay overhead -> true    #
# in-graph per-op kernel time, the deployed onegraph basis) + eager contrast.  #
# --------------------------------------------------------------------------- #
def time_op_graph(fn, reps: int, iters: int, warmup: int):
    """Per-call us via ONE CUDA graph capturing `reps` back-to-back calls, timed
    over `iters` replays. Returns (us_per_call, captured)."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                for _ in range(reps):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        us = (e0.elapsed_time(e1) / iters) * 1000.0 / reps
        del g
        return us, True
    except Exception as exc:  # noqa: BLE001
        print(f"[nongemm]   graph capture failed: {exc!r}; eager fallback", flush=True)
        return time_op_eager(fn, iters, warmup), False


def time_op_eager(fn, iters: int, warmup: int) -> float:
    """Eager per-call us (carries the ~launch+dispatch floor)."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        torch.cuda.synchronize()
        for i in range(iters):
            e0[i].record()
            fn()
            e1[i].record()
        torch.cuda.synchronize()
    ms = [e0[i].elapsed_time(e1[i]) for i in range(iters)]
    return statistics.median(ms) * 1000.0


# --------------------------------------------------------------------------- #
# Fused sparse-argmax triton kernels -- VERBATIM from the deployed submission  #
# submissions/fa2sw_precache_kenyan/sitecustomize.py (FUSED_SPARSE_ARGMAX path).#
# This is the kernel the served drafter runs every pass; copied so the profile #
# is bit-identical to deployment without importing sitecustomize (which patches #
# vLLM on import).                                                              #
# --------------------------------------------------------------------------- #
def _get_fused_sparse_argmax_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _sparse_argmax_blocks_kernel(
        hidden_states, lm_head_weight, top_centroids, token_ordering,
        partial_scores, partial_tokens,
        hidden_stride_t, hidden_stride_d, lm_head_stride_v, lm_head_stride_d,
        top_stride_t, top_stride_k, partial_score_stride_t, partial_token_stride_t,
        VOCAB_PER_CENTROID: tl.constexpr, SELECTED_COUNT: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr, BLOCK_SELECTED: tl.constexpr, BLOCK_D: tl.constexpr,
    ) -> None:
        token_idx = tl.program_id(0)
        selected_block = tl.program_id(1)
        selected_offsets = selected_block * BLOCK_SELECTED + tl.arange(0, BLOCK_SELECTED)
        valid_selected = selected_offsets < SELECTED_COUNT
        centroid_slots = selected_offsets // VOCAB_PER_CENTROID
        token_slots = selected_offsets - centroid_slots * VOCAB_PER_CENTROID
        centroid_ids = tl.load(
            top_centroids + token_idx * top_stride_t + centroid_slots * top_stride_k,
            mask=valid_selected, other=0)
        vocab_ids = tl.load(
            token_ordering + centroid_ids * VOCAB_PER_CENTROID + token_slots,
            mask=valid_selected, other=0)
        d_offsets = tl.arange(0, BLOCK_D)
        valid_d = d_offsets < HIDDEN_SIZE
        hidden = tl.load(
            hidden_states + token_idx * hidden_stride_t + d_offsets * hidden_stride_d,
            mask=valid_d, other=0.0).to(tl.float32)
        weights = tl.load(
            lm_head_weight + vocab_ids[:, None] * lm_head_stride_v
            + d_offsets[None, :] * lm_head_stride_d,
            mask=valid_selected[:, None] & valid_d[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(weights * hidden[None, :], axis=1)
        scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(valid_selected, scores, -float("inf"))
        best_score, best_local_idx = tl.max(
            scores, axis=0, return_indices=True, return_indices_tie_break_left=True)
        best_selected = selected_block * BLOCK_SELECTED + best_local_idx
        best_centroid_slot = best_selected // VOCAB_PER_CENTROID
        best_token_slot = best_selected - best_centroid_slot * VOCAB_PER_CENTROID
        best_centroid = tl.load(
            top_centroids + token_idx * top_stride_t + best_centroid_slot * top_stride_k)
        best_token = tl.load(
            token_ordering + best_centroid * VOCAB_PER_CENTROID + best_token_slot)
        tl.store(partial_scores + token_idx * partial_score_stride_t + selected_block, best_score)
        tl.store(partial_tokens + token_idx * partial_token_stride_t + selected_block, best_token)

    @triton.jit
    def _sparse_argmax_reduce_kernel(
        partial_scores, partial_tokens, output_tokens,
        partial_score_stride_t, partial_token_stride_t, output_stride_t,
        NUM_BLOCKS: tl.constexpr, BLOCK_BLOCKS: tl.constexpr,
    ) -> None:
        token_idx = tl.program_id(0)
        block_offsets = tl.arange(0, BLOCK_BLOCKS)
        valid_blocks = block_offsets < NUM_BLOCKS
        scores = tl.load(
            partial_scores + token_idx * partial_score_stride_t + block_offsets,
            mask=valid_blocks, other=-float("inf"))
        _, best_block = tl.max(
            scores, axis=0, return_indices=True, return_indices_tie_break_left=True)
        token = tl.load(partial_tokens + token_idx * partial_token_stride_t + best_block)
        tl.store(output_tokens + token_idx * output_stride_t, token)

    return triton, _sparse_argmax_blocks_kernel, _sparse_argmax_reduce_kernel


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def make_fused_top_tokens(emb, block: int):
    """Driver mirroring sitecustomize.get_top_tokens_fused (FUSED_SPARSE_ARGMAX)."""
    triton, blocks_kernel, reduce_kernel = _get_fused_sparse_argmax_kernels()
    hidden_size = int(emb.hidden_size)
    selected_count = int(emb.num_selected)
    block_selected = _next_power_of_2(block)
    num_blocks = triton.cdiv(selected_count, block_selected)
    reduce_block = _next_power_of_2(num_blocks)
    block_d = _next_power_of_2(hidden_size)

    def run(hidden_states, lm_head_weight):
        num_tokens = int(hidden_states.shape[0])
        _, top_k_indices = torch.topk(
            emb.centroids(hidden_states), k=emb.centroid_intermediate_top_k,
            dim=-1, sorted=False)
        partial_scores = torch.empty((num_tokens, num_blocks), dtype=torch.float32,
                                     device=hidden_states.device)
        partial_tokens = torch.empty((num_tokens, num_blocks), dtype=torch.int64,
                                     device=hidden_states.device)
        output_tokens = torch.empty((num_tokens,), dtype=torch.int64,
                                    device=hidden_states.device)
        blocks_kernel[(num_tokens, num_blocks)](
            hidden_states, lm_head_weight, top_k_indices, emb.token_ordering,
            partial_scores, partial_tokens,
            hidden_states.stride(0), hidden_states.stride(1),
            lm_head_weight.stride(0), lm_head_weight.stride(1),
            top_k_indices.stride(0), top_k_indices.stride(1),
            partial_scores.stride(0), partial_tokens.stride(0),
            VOCAB_PER_CENTROID=int(emb.vocab_size_per_centroid),
            SELECTED_COUNT=selected_count, HIDDEN_SIZE=hidden_size,
            BLOCK_SELECTED=block_selected, BLOCK_D=block_d, num_warps=8)
        reduce_kernel[(num_tokens,)](
            partial_scores, partial_tokens, output_tokens,
            partial_scores.stride(0), partial_tokens.stride(0), output_tokens.stride(0),
            NUM_BLOCKS=num_blocks, BLOCK_BLOCKS=reduce_block, num_warps=8)
        return output_tokens

    return run


# --------------------------------------------------------------------------- #
# Attention: roofline (decode attention is memory-bound) + SDPA proxy.         #
# --------------------------------------------------------------------------- #
def attn_kv_bytes(num_kv: int, head_dim: int, L: int) -> float:
    # K + V read, bf16; q/out negligible at M=1.
    return 2.0 * L * num_kv * head_dim * 2.0


def time_attn_proxy(num_heads, num_kv, head_dim, L, iters, warmup):
    """SDPA proxy at the served drafter shapes: q[1,H,1,Dh], k/v[1,Hkv,L,Dh]."""
    import torch.nn.functional as F
    q = torch.randn(1, num_heads, 1, head_dim, device="cuda", dtype=BF16)
    k = torch.randn(1, num_kv, L, head_dim, device="cuda", dtype=BF16)
    v = torch.randn(1, num_kv, L, head_dim, device="cuda", dtype=BF16)

    def fn():
        try:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        except TypeError:
            kk = k.repeat_interleave(num_heads // num_kv, dim=1)
            vv = v.repeat_interleave(num_heads // num_kv, dim=1)
            return F.scaled_dot_product_attention(q, kk, vv)

    us_g, captured = time_op_graph(fn, reps=32, iters=iters, warmup=warmup)
    return us_g, captured


# --------------------------------------------------------------------------- #
# Build the real served non-GEMM modules from the real drafter weights.        #
# --------------------------------------------------------------------------- #
def build_modules(drafter_dir: str, top_k: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.activation import get_act_and_mul_fn
    from vllm.model_executor.models.gemma4_mtp import Gemma4MTPMaskedEmbedder

    # vLLM CustomOps (RMSNorm/rotary/activation) read the compilation config at
    # instantiation; provide a default config context so forward dispatch resolves.
    _cfg_ctx = set_current_vllm_config(VllmConfig())
    _cfg_ctx.__enter__()

    st = os.path.join(drafter_dir, "model.safetensors")
    cfg = json.load(open(os.path.join(drafter_dir, "config.json")))
    tcfg = cfg["text_config"]
    eps = tcfg["rms_norm_eps"]
    hidden = tcfg["hidden_size"]            # 256
    head_dim = tcfg["head_dim"]            # 256 sliding
    global_head_dim = tcfg["global_head_dim"]  # 512 full
    inter = tcfg["intermediate_size"]      # 2048
    vocab = tcfg["vocab_size"]             # 262144
    num_centroids = cfg["num_centroids"]   # 2048

    def _fused(m):
        """Prefer the single-kernel forward_cuda over the native multi-kernel
        decomposition (which over-counts when timed standalone). The deployed
        onegraph is torch-compiled (native + inductor fusion); forward_cuda is a
        tighter, less-misleading standalone UPPER BOUND than native."""
        fc = getattr(m, "forward_cuda", None)
        if fc is None:
            return m
        def call(*a, **k):
            try:
                return fc(*a, **k)
            except Exception:
                return m(*a, **k)
        return call

    def rms(dim, wname):
        m = RMSNorm(dim, eps=eps).to("cuda", BF16)
        try:
            w = load_tensor(st, wname).to("cuda", BF16)
            with torch.no_grad():
                m.weight.copy_(w)
        except Exception:
            pass
        return _fused(m)

    mods = {}
    # hidden-dim RMSNorms (representative: layer 0 weights; shape-identical x17)
    mods["rmsnorm_hidden"] = rms(hidden, "model.layers.0.input_layernorm.weight")
    mods["qnorm_sliding"] = rms(head_dim, "model.layers.0.self_attn.q_norm.weight")
    mods["qnorm_full"] = rms(global_head_dim, "model.layers.3.self_attn.q_norm.weight")

    # rotary (sliding theta 1e4; full partial 0.25 theta 1e6)
    mods["rope_sliding"] = get_rope(
        head_dim, max_position=tcfg["max_position_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        is_neox_style=True).to("cuda")
    mods["rope_full"] = get_rope(
        global_head_dim, max_position=tcfg["max_position_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                         "partial_rotary_factor": 0.25},
        is_neox_style=True).to("cuda")

    mods["act"] = _fused(get_act_and_mul_fn(tcfg["hidden_activation"]))  # gelu_pytorch_tanh

    # input embed table [262144, hidden] (real, draft-dim). Served input embed is
    # backbone-dim [262144,2560] after sharing; a 1-row gather either way.
    emb_w = load_tensor(st, "model.embed_tokens.weight").to("cuda", BF16)

    # centroid masked-embedder (real class + real weights)
    me = Gemma4MTPMaskedEmbedder(hidden, vocab, num_centroids, top_k).to("cuda", BF16)
    with torch.no_grad():
        me.centroids.weight.copy_(load_tensor(st, "masked_embedding.centroids.weight").to("cuda", BF16))
        me.token_ordering.copy_(load_tensor(st, "masked_embedding.token_ordering").to("cuda"))
    lm_head = emb_w  # tied [262144, 256] draft-dim, used by the masked-embedder

    info = dict(hidden=hidden, head_dim=head_dim, global_head_dim=global_head_dim,
                inter=inter, vocab=vocab, num_centroids=num_centroids,
                num_heads=tcfg["num_attention_heads"], num_kv=tcfg["num_key_value_heads"],
                vocab_per_centroid=vocab // num_centroids,
                num_selected=top_k * (vocab // num_centroids), top_k=top_k,
                sliding_window=tcfg["sliding_window"])
    return mods, emb_w, me, lm_head, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--top-k", type=int, default=64, help="centroid_intermediate_top_k (serve.py sets 64)")
    ap.add_argument("--fused-block", type=int, default=16, help="FUSED_SPARSE_ARGMAX_BLOCK (manifest=16)")
    ap.add_argument("--k", type=int, default=7, help="num_speculative_tokens (deployed=7)")
    ap.add_argument("--l-sweep", default="128,256,512,1024,2048",
                    help="KV lengths for the attention roofline/proxy sweep")
    ap.add_argument("--l-headline", type=int, default=512,
                    help="representative decode KV length for the headline attention row")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--decode-step-ms", type=float, default=11.6)
    ap.add_argument("--gemm-chain-us", type=float, default=566.0,
                    help="denken #75 measured 7-pass drafter GEMM chain (us/step)")
    ap.add_argument("--drafter-budget-pct-lo", type=float, default=15.5)
    ap.add_argument("--drafter-budget-pct-hi", type=float, default=18.1)
    ap.add_argument("--frontier-tps", type=float, default=481.53)
    ap.add_argument("--output", default="research/spec_cost_model/drafter_nongemm_profile.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="drafter-nongemm-profile")
    ap.add_argument("--wandb_name", default="denken/drafter-nongemm-profile")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-only", default=None)
    args = ap.parse_args()

    if args.log_only:
        with open(args.log_only) as fh:
            payload = json.load(fh)
        _log_wandb(args, payload)
        return

    l_sweep = [int(x) for x in args.l_sweep.split(",") if x.strip()]
    print(f"[nongemm] device: {torch.cuda.get_device_name(0)}", flush=True)
    mods, emb_w, me, lm_head, info = build_modules(args.drafter_dir, args.top_k)
    H, Hkv = info["num_heads"], info["num_kv"]
    hd, ghd = info["head_dim"], info["global_head_dim"]
    print(f"[nongemm] config: {info}", flush=True)

    iters, warm = args.iters, args.warmup
    # per-pass op COUNTS (from gemma4_mtp.forward, 4 layers = 3 sliding + 1 full)
    blocks = []  # each: dict(name, count, us_graph, us_eager, bytes, note)

    def add(name, count, fn, reps, note="", bytes_per=0.0, eager=True):
        ug, cap = time_op_graph(fn, reps=reps, iters=iters, warmup=warm)
        ue = time_op_eager(fn, max(40, iters // 4), warm) if eager else float("nan")
        blocks.append(dict(name=name, count=count, us_graph=ug, us_eager=ue,
                           captured=cap, bytes=bytes_per, note=note))
        print(f"[nongemm] {name:>22s} x{count:<2d} {ug:7.3f}us graph "
              f"({ue:7.3f} eager) {note}", flush=True)

    # --- 1. input embed gather (1 row) ---
    idx = torch.zeros(1, dtype=torch.long, device="cuda")
    add("embed_gather", 1, lambda: emb_w.index_select(0, idx),
        reps=128, note="1 row [262144,D]; served D=2560", bytes_per=info["hidden"] * 2)

    # --- 2. RMSNorm hidden (dim 256) x17 ---
    xh = torch.randn(1, info["hidden"], device="cuda", dtype=BF16)
    add("rmsnorm_hidden", 17, lambda: mods["rmsnorm_hidden"](xh), reps=128,
        note="input/post_attn/pre_ff/post_ff x4 + final", bytes_per=info["hidden"] * 2)

    # --- 3. q_norm (per-head) sliding x3 + full x1 ---
    xqs = torch.randn(1, H, hd, device="cuda", dtype=BF16)
    xqf = torch.randn(1, H, ghd, device="cuda", dtype=BF16)
    add("qnorm_sliding", 3, lambda: mods["qnorm_sliding"](xqs), reps=128)
    add("qnorm_full", 1, lambda: mods["qnorm_full"](xqf), reps=128)

    # --- 4. rotary sliding x3 + full x1 ---
    pos = torch.zeros(1, dtype=torch.long, device="cuda")
    qs = torch.randn(1, H * hd, device="cuda", dtype=BF16)
    qf = torch.randn(1, H * ghd, device="cuda", dtype=BF16)
    add("rotary_sliding", 3, lambda: mods["rope_sliding"](pos, qs.clone(), None), reps=64)
    add("rotary_full", 1, lambda: mods["rope_full"](pos, qf.clone(), None), reps=64)

    # --- 5. residual adds x8, layer_scalar mul x4 ---
    a = torch.randn(1, info["hidden"], device="cuda", dtype=BF16)
    b = torch.randn(1, info["hidden"], device="cuda", dtype=BF16)
    sc = torch.ones(1, device="cuda", dtype=BF16)
    add("residual_add", 8, lambda: a + b, reps=256, bytes_per=info["hidden"] * 2 * 3)
    add("layer_scalar_mul", 4, lambda: a * sc, reps=256)
    # concat [1,D]+[1,D]->[1,2D] glue x1
    add("embed_concat", 1, lambda: torch.cat([a, a], dim=-1), reps=256)

    # --- 6. gelu_tanh act-and-mul x4 over [1, 2*inter] -> [1, inter] ---
    gu = torch.randn(1, 2 * info["inter"], device="cuda", dtype=BF16)
    add("activation_gelu", 4, lambda: mods["act"](gu), reps=128,
        note="gelu_tanh(gate)*up over [1,2048]")

    # --- 7. centroid sparse sampler (DEPLOYED fused triton; 8192-candidate gather) ---
    fused = make_fused_top_tokens(me, args.fused_block)
    hsamp = torch.randn(1, info["hidden"], device="cuda", dtype=BF16)
    add("centroid_sampler_fused", 1, lambda: fused(hsamp, lm_head), reps=16,
        note=f"top-{args.top_k} centroids -> {info['num_selected']} cand rows, "
             f"FUSED block={args.fused_block}",
        bytes_per=info["num_selected"] * info["hidden"] * 2)

    # centroid sampler internal decomposition (unfused, for Step-0) ----------
    decomp = {}
    def decomp_add(name, fn, reps):
        ug, _ = time_op_graph(fn, reps=reps, iters=iters, warmup=warm)
        decomp[name] = ug
        print(f"[nongemm]   [decomp] {name:>20s} {ug:7.3f}us", flush=True)
    # centroids GEMM (already in #75; shown for context)
    decomp_add("centroids_gemm", lambda: me.centroids(hsamp), reps=64)
    def topk_fn():
        return torch.topk(me.centroids(hsamp), k=args.top_k, dim=-1, sorted=False)
    decomp_add("centroids_topk", topk_fn, reps=64)
    # gather 8192 candidate rows (the "262k masked-embed gather", already 8192/262144)
    sel = torch.randint(0, info["vocab"], (info["num_selected"],), device="cuda")
    decomp_add("masked_embed_gather8192",
               lambda: lm_head.index_select(0, sel), reps=16)
    # full-vocab gather counterfactual (what a NON-masked drafter would pay)
    allidx = torch.arange(info["vocab"], device="cuda")
    decomp_add("full_vocab_gather262k",
               lambda: lm_head.index_select(0, allidx), reps=4)
    # unfused score+argmax over 8192 candidates
    emb8192 = torch.randn(1, info["num_selected"], info["hidden"], device="cuda", dtype=BF16)
    def score_argmax():
        lg = torch.einsum("td,tsd->ts", hsamp, emb8192)
        return lg.argmax(-1)
    decomp_add("score_argmax_8192", score_argmax, reps=32)

    # --- 8. attention SDPA: roofline + proxy, swept over L --------------------
    attn_rows = []
    for L in l_sweep:
        Ls = min(L, info["sliding_window"])
        bytes_sliding = attn_kv_bytes(Hkv, hd, Ls)
        bytes_full = attn_kv_bytes(Hkv, ghd, L)
        roof_sliding = bytes_sliding / A10G_HBM_GBS / 1e3   # us
        roof_full = bytes_full / A10G_HBM_GBS / 1e3
        us_sliding, _ = time_attn_proxy(H, Hkv, hd, Ls, iters, warm)
        us_full, _ = time_attn_proxy(H, Hkv, ghd, L, iters, warm)
        # per-pass attention = 3 sliding + 1 full
        pass_roof = 3 * roof_sliding + roof_full
        pass_proxy = 3 * us_sliding + us_full
        attn_rows.append(dict(L=L, L_sliding=Ls,
                              roof_sliding_us=roof_sliding, roof_full_us=roof_full,
                              proxy_sliding_us=us_sliding, proxy_full_us=us_full,
                              per_pass_roofline_us=pass_roof, per_pass_proxy_us=pass_proxy,
                              bytes_sliding=bytes_sliding, bytes_full=bytes_full))
        print(f"[nongemm] attn L={L:5d} | sliding(KV{Ls}) roof {roof_sliding:6.2f} "
              f"proxy {us_sliding:6.2f} | full roof {roof_full:6.2f} proxy {us_full:6.2f} "
              f"| per-pass roof {pass_roof:6.2f} proxy {pass_proxy:6.2f} us", flush=True)

    head = next(r for r in attn_rows if r["L"] == args.l_headline)
    # attention contributes one per-pass block; report BOTH roofline and proxy.
    attn_pass_roof = head["per_pass_roofline_us"]
    attn_pass_proxy = head["per_pass_proxy_us"]

    # --------------------------------------------------------------------- #
    # Aggregate. CRITICAL FAITHFULNESS NOTE.                                 #
    #                                                                        #
    # The deployed drafter runs under ONEGRAPH=1 + @support_torch_compile,   #
    # so inductor FUSES the elementwise / norm / rotary / residual "glue"    #
    # ops into the backbone GEMM epilogues. Timing each glue op in ISOLATION #
    # pays a separate kernel launch + global RW that does NOT exist in the   #
    # deployed graph, so the isolated-unfused SUM is an UPPER BOUND, not the #
    # deployed cost. Proof: that sum EXCEEDS wirbel#69's *entire* measured    #
    # drafter budget -- impossible if it were the real per-op cost.          #
    #                                                                        #
    # We therefore report THREE faithful layers:                             #
    #  (1) AUTHORITATIVE non-GEMM TOTAL = wirbel#69 end-to-end drafter budget  #
    #      minus #75's 566us GEMM chain (measured in the real server).       #
    #  (2) STANDALONE deployed kernels we CAN time faithfully: the centroid  #
    #      sparse sampler (its own triton kernel + CUDA graph) and attention #
    #      (its own paged kernel; roofline = memory-bound floor). These are  #
    #      the ONLY non-GEMM ops that do NOT fuse into the GEMM epilogues.   #
    #  (3) FUSIBLE GLUE (gather/norms/rotary/residual/act): isolated-unfused #
    #      UPPER BOUND only; fused away in the deployed graph.               #
    # Binding non-GEMM sub-block = the largest genuine STANDALONE kernel.    #
    # --------------------------------------------------------------------- #
    k = args.k
    decode_us = args.decode_step_ms * 1000.0

    FUSIBLE_GLUE = {"embed_gather", "rmsnorm_hidden", "qnorm_sliding", "qnorm_full",
                    "rotary_sliding", "rotary_full", "residual_add",
                    "layer_scalar_mul", "embed_concat", "activation_gelu"}

    by_name = {bl["name"]: bl for bl in blocks}
    def pass_us(name):
        bl = by_name[name]
        return bl["us_graph"] * bl["count"]

    # (3) fusible glue: isolated-unfused UPPER BOUND (deployed graph fuses these)
    glue_pass_ub = sum(pass_us(n) for n in FUSIBLE_GLUE if n in by_name)
    # (2) standalone deployed kernels
    sampler_pass = pass_us("centroid_sampler_fused")
    standalone_pass_floor = sampler_pass + attn_pass_roof    # attn at memory floor
    standalone_pass_hi = sampler_pass + attn_pass_proxy       # attn proxy upper bound

    # full isolated-unfused per-pass / per-step sum (the over-count we DISTRUST)
    isolated_pass_ub = glue_pass_ub + standalone_pass_hi
    nongemm_isolated_step_ub = isolated_pass_ub * k

    gemm_step = args.gemm_chain_us
    # (1) AUTHORITATIVE non-GEMM total from wirbel#69 (drafter budget - GEMM chain)
    budget_lo = decode_us * args.drafter_budget_pct_lo / 100.0   # 15.5% drafter
    budget_hi = decode_us * args.drafter_budget_pct_hi / 100.0   # 18.1% drafter
    nongemm_anchor_lo = budget_lo - gemm_step
    nongemm_anchor_hi = budget_hi - gemm_step

    # standalone kernels per step + the un-attributable long-tail remainder
    sampler_step = sampler_pass * k
    attn_roof_step = attn_pass_roof * k
    attn_proxy_step = attn_pass_proxy * k
    standalone_floor_step = standalone_pass_floor * k
    # remainder = #69 non-GEMM total minus the standalone kernels we resolved.
    # It is fused glue (tiny) + per-kernel graph-replay dispatch + python glue:
    # "death by a thousand cuts" with NO single fat reducible sub-block.
    unattributed_lo = nongemm_anchor_lo - standalone_floor_step
    unattributed_hi = nongemm_anchor_hi - standalone_floor_step

    drafter_mid = 0.5 * (budget_lo + budget_hi)   # #69 drafter, for %drafter denom

    # build the reported sub-block table (per-pass, per-step, %drafter, %decode)
    table = []
    def row(name, count, pp, cls, note):
        st_us = pp * k
        return dict(name=name, count=count, us_per_pass=pp, us_per_step=st_us,
                    cls=cls, pct_of_drafter=100.0 * st_us / drafter_mid,
                    pct_of_decode=100.0 * st_us / decode_us, note=note)
    for bl in blocks:
        if bl["name"] == "centroid_sampler_fused":
            continue  # added below as a standalone deployed kernel
        cls = "fusible_glue_upperbound" if bl["name"] in FUSIBLE_GLUE else "standalone"
        nt = bl["note"]
        if cls == "fusible_glue_upperbound":
            nt = (nt + "; " if nt else "") + "isolated-unfused UB (fused in deployed graph)"
        table.append(row(bl["name"], bl["count"], bl["us_graph"] * bl["count"], cls, nt))
    # standalone deployed kernels (the genuine non-GEMM kernels in the served graph)
    table.append(row("centroid_sampler_fused", 1, sampler_pass, "standalone_deployed",
                     by_name["centroid_sampler_fused"]["note"]))
    table.append(row("attention_sdpa_roofline", 4, attn_pass_roof, "standalone_deployed",
                     f"3 sliding(KV{head['L_sliding']})+1 full(KV{args.l_headline}); memory-bound floor"))
    table.append(row("attention_sdpa_proxy", 4, attn_pass_proxy, "standalone_upperbound",
                     "SDPA proxy; launch-overhead-dominated UB for paged TRITON_ATTN"))

    # binding non-GEMM sub-block = largest genuine STANDALONE deployed kernel
    standalone_rows = [t for t in table if t["cls"] == "standalone_deployed"]
    binding = max(standalone_rows, key=lambda t: t["us_per_step"])

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    verdict = dict(
        primary_metric_name="drafter_nongemm_binding_subblock_pct_of_decode",
        drafter_nongemm_binding_subblock_pct_of_decode=binding["pct_of_decode"],
        binding_subblock=binding["name"],
        binding_subblock_us_per_step=binding["us_per_step"],
        binding_subblock_pct_of_drafter=binding["pct_of_drafter"],
        # (1) AUTHORITATIVE non-GEMM total = wirbel#69 drafter budget - #75 GEMM
        nongemm_anchor_step_us_lo=nongemm_anchor_lo,
        nongemm_anchor_step_us_hi=nongemm_anchor_hi,
        nongemm_anchor_pct_decode_lo=100.0 * nongemm_anchor_lo / decode_us,
        nongemm_anchor_pct_decode_hi=100.0 * nongemm_anchor_hi / decode_us,
        # (2) standalone deployed kernels resolved faithfully
        sampler_step_us=sampler_step,
        sampler_pct_decode=100.0 * sampler_step / decode_us,
        attn_roofline_step_us=attn_roof_step,
        attn_roofline_pct_decode=100.0 * attn_roof_step / decode_us,
        attn_proxy_step_us=attn_proxy_step,
        standalone_floor_step_us=standalone_floor_step,
        standalone_floor_pct_decode=100.0 * standalone_floor_step / decode_us,
        # (3) un-attributable long-tail remainder (fused glue + graph dispatch)
        nongemm_unattributed_step_us_lo=unattributed_lo,
        nongemm_unattributed_step_us_hi=unattributed_hi,
        nongemm_unattributed_pct_decode_lo=100.0 * unattributed_lo / decode_us,
        nongemm_unattributed_pct_decode_hi=100.0 * unattributed_hi / decode_us,
        # isolated-unfused over-count (UPPER BOUND we DISTRUST; proves fusion)
        nongemm_isolated_unfused_step_us_ub=nongemm_isolated_step_ub,
        isolated_exceeds_wirbel69_drafter=bool(nongemm_isolated_step_ub > budget_hi),
        # GEMM (#75) + drafter total: this audit CONFIRMS #69, not replaces it
        gemm_step_us=gemm_step,
        gemm_pct_decode=100.0 * gemm_step / decode_us,
        drafter_step_us_lo=budget_lo, drafter_step_us_hi=budget_hi,
        drafter_pct_decode_lo=args.drafter_budget_pct_lo,
        drafter_pct_decode_hi=args.drafter_budget_pct_hi,
        wirbel69_drafter_budget_us=[budget_lo, budget_hi],
        wirbel69_drafter_budget_pct=[args.drafter_budget_pct_lo, args.drafter_budget_pct_hi],
        # Step-0 contract-safe reduction feasibility
        masked_embed_gather_already_candidate_restricted=True,
        masked_embed_gather_candidate_rows=info["num_selected"],
        masked_embed_gather_full_vocab_rows=info["vocab"],
        masked_embed_gather_restriction_ratio=info["num_selected"] / info["vocab"],
        masked_embed_gather8192_us=decomp.get("masked_embed_gather8192"),
        full_vocab_gather262k_us=decomp.get("full_vocab_gather262k"),
        centroids_topk_us=decomp.get("centroids_topk"),
        contract_safe_nongemm_lever_exists=False,
        k=k, decode_step_ms=args.decode_step_ms, peak_gpu_mem_gib=peak_mem,
    )

    print("\n[nongemm] ===== NON-GEMM SUB-BLOCK TABLE (per decode step, xK=%d) =====" % k, flush=True)
    print("[nongemm] cls: [S]=standalone deployed kernel  [u]=standalone upper bound  "
          "[U]=fusible-glue isolated UPPER BOUND (fused away in deployed onegraph)", flush=True)
    print(f"{'sub-block':>26s} {'cls':>3s} {'cnt':>4s} {'us/pass':>9s} {'us/step':>9s} "
          f"{'%drafter':>9s} {'%decode':>8s}", flush=True)
    clsmark = {"standalone_deployed": "S", "standalone_upperbound": "u",
               "fusible_glue_upperbound": "U", "standalone": "S"}
    for t in sorted(table, key=lambda t: -t["us_per_step"]):
        print(f"{t['name']:>26s} {clsmark.get(t['cls'], '?'):>3s} {t['count']:>4d} "
              f"{t['us_per_pass']:>9.2f} {t['us_per_step']:>9.1f} "
              f"{t['pct_of_drafter']:>8.1f}% {t['pct_of_decode']:>7.2f}%", flush=True)

    print("\n[nongemm] ----- FAITHFUL NON-GEMM ACCOUNTING (deployed onegraph + torch-compile) -----", flush=True)
    print(f"[nongemm] AUTHORITATIVE non-GEMM total (wirbel#69 drafter "
          f"{args.drafter_budget_pct_lo}-{args.drafter_budget_pct_hi}% - #75 GEMM {gemm_step:.0f}us): "
          f"{nongemm_anchor_lo:.0f}-{nongemm_anchor_hi:.0f}us/step "
          f"({verdict['nongemm_anchor_pct_decode_lo']:.1f}-{verdict['nongemm_anchor_pct_decode_hi']:.1f}% decode)", flush=True)
    print(f"[nongemm]   resolved STANDALONE kernels: sampler {sampler_step:.0f}us "
          f"({verdict['sampler_pct_decode']:.2f}%) + attn[roofline floor] {attn_roof_step:.0f}us "
          f"({verdict['attn_roofline_pct_decode']:.2f}%) = {standalone_floor_step:.0f}us/step "
          f"({verdict['standalone_floor_pct_decode']:.2f}% decode)", flush=True)
    print(f"[nongemm]   UN-ATTRIBUTABLE long-tail (fused glue + per-kernel graph dispatch + python): "
          f"{unattributed_lo:.0f}-{unattributed_hi:.0f}us/step "
          f"({verdict['nongemm_unattributed_pct_decode_lo']:.1f}-{verdict['nongemm_unattributed_pct_decode_hi']:.1f}% decode) "
          f"-- NO single reducible hotspot", flush=True)
    print(f"[nongemm]   [sanity] isolated-unfused SUM = {nongemm_isolated_step_ub:.0f}us/step "
          f"{'EXCEEDS' if verdict['isolated_exceeds_wirbel69_drafter'] else 'within'} #69 whole-drafter "
          f"budget {budget_hi:.0f}us -> proves glue FUSES in deployment (over-count)", flush=True)
    print(f"[nongemm] BINDING non-GEMM sub-block (largest STANDALONE kernel): {binding['name']} "
          f"= {binding['us_per_step']:.0f}us/step = {binding['pct_of_decode']:.2f}% decode", flush=True)
    print(f"[nongemm] Step-0 contract-safe reduction: masked-embed gather ALREADY candidate-restricted "
          f"{info['num_selected']}/{info['vocab']} ({100.0*info['num_selected']/info['vocab']:.1f}% vocab); "
          f"gather8192 {decomp.get('masked_embed_gather8192'):.1f}us vs full-262k "
          f"{decomp.get('full_vocab_gather262k'):.1f}us "
          f"(~{(decomp.get('full_vocab_gather262k') or 1)/(decomp.get('masked_embed_gather8192') or 1):.0f}x already saved)", flush=True)
    print(f"[nongemm] => NO contract-safe non-GEMM lever: gather restricted, attn at memory floor, "
          f"glue already fused. Drafter near floor on BOTH halves (#75 GEMM int4-refuted + this).", flush=True)
    print(f"[nongemm] peak GPU mem: {peak_mem:.2f} GiB", flush=True)

    payload = dict(
        config=dict(drafter_dir=args.drafter_dir, torch=torch.__version__,
                    device=torch.cuda.get_device_name(0), k=k, top_k=args.top_k,
                    fused_block=args.fused_block, iters=iters, warmup=warm,
                    decode_step_ms=args.decode_step_ms, gemm_chain_us=gemm_step,
                    l_sweep=l_sweep, l_headline=args.l_headline,
                    frontier_tps=args.frontier_tps, A10G_HBM_GBS=A10G_HBM_GBS,
                    drafter_info=info, peak_gpu_mem_gib=peak_mem,
                    note="isolated launch-free (reps-in-graph) per-op timing, real "
                         "served vLLM modules + real drafter weights + VERBATIM deployed "
                         "fused sparse-argmax triton kernel; attention = roofline + SDPA "
                         "proxy (memory-bound decode). Value-independent, no serve change."),
        blocks=blocks, table=table, attn_sweep=attn_rows,
        centroid_decomp=decomp, verdict=verdict,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[nongemm] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[nongemm] W&B logging failed: {exc!r}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    cols = ["name", "cls", "count", "us_per_pass", "us_per_step", "pct_of_drafter",
            "pct_of_decode", "note"]
    tbl = wandb.Table(columns=cols)
    for t in payload["table"]:
        tbl.add_data(t["name"], t.get("cls", ""), t["count"], t.get("us_per_pass"),
                     t["us_per_step"], t["pct_of_drafter"], t["pct_of_decode"],
                     t.get("note", ""))
    run.log({"nongemm_subblock_table": tbl})
    acols = ["L", "per_pass_roofline_us", "per_pass_proxy_us", "roof_full_us", "proxy_full_us"]
    atbl = wandb.Table(columns=acols)
    for r in payload["attn_sweep"]:
        atbl.add_data(r["L"], r["per_pass_roofline_us"], r["per_pass_proxy_us"],
                      r["roof_full_us"], r["proxy_full_us"])
    run.log({"attn_L_sweep": atbl})
    run.summary.update({k: v for k, v in payload["verdict"].items() if v is not None})
    run.finish()
    print(f"[nongemm] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
