"""Tile-config microbench for the MTP K=7 drafter's drafter-SPECIFIC Triton kernels.

PR #449 (lawine/drafter-kernel-tile-profile). LOCAL A10G (sm_86) only, byte-exact,
NO served-file change, NO HF submission.

The deployed drafter (submissions/fa2sw_precache_kenyan, ONEGRAPH=1) runs K=7 width-1
MTP-head iterations per decode step. Each iteration dispatches:
  * ~10 int4 Marlin GEMMs (pre/post-proj, q/o-proj x4, MLP x4)  -> stark's int4-GEMM domain, NOT here
  * Q-only KV-shared attention (TRITON_ATTN)                    -> verify-attention domain (wirbel #442)
  * centroids() Linear [256->2048] + topk(64)                   -> cublas + torch.topk, not Triton
  * the FUSED_SPARSE_ARGMAX Triton kernels                      <-- the ONLY drafter-SPECIFIC Triton kernels

This script benches ONLY the drafter-specific Triton path: `_sparse_argmax_blocks_kernel`
(grid (num_tokens, num_blocks)) + `_sparse_argmax_reduce_kernel` (grid (num_tokens,)),
extracted VERBATIM from submissions/fa2sw_precache_kenyan/sitecustomize.py, parametrized so
the served-frozen launch config (BLOCK_SELECTED=next_pow2(16)=16, num_warps=8, num_stages=triton
default) can be swept against {BLOCK_SELECTED, num_warps, num_stages}.

Real shapes (from /tmp/qat-assistant/config.json + target vocab):
  num_tokens=1 (width-1 decode), hidden_size(draft-dim)=256, vocab_size=262144,
  num_centroids=2048, vocab_size_per_centroid=128, centroid_intermediate_top_k=64,
  num_selected=64*128=8192, lm_head_weight [262144, 256] bf16.

Correctness: every swept config must return token IDs byte-identical to the served default
(same math + tie_break_left) -> a faster argmax cannot change the proposed draft token, and
verify (target argmax via the accept-prep kernel, land #420) is the sole arbiter of emitted
tokens => greedy identity is FREE by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import triton
import triton.language as tl


# --- kernels: VERBATIM from sitecustomize.py _get_fused_sparse_argmax_kernels ---
@triton.jit
def _sparse_argmax_blocks_kernel(
    hidden_states,
    lm_head_weight,
    top_centroids,
    token_ordering,
    partial_scores,
    partial_tokens,
    hidden_stride_t,
    hidden_stride_d,
    lm_head_stride_v,
    lm_head_stride_d,
    top_stride_t,
    top_stride_k,
    partial_score_stride_t,
    partial_token_stride_t,
    VOCAB_PER_CENTROID: tl.constexpr,
    SELECTED_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SELECTED: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    token_idx = tl.program_id(0)
    selected_block = tl.program_id(1)

    selected_offsets = selected_block * BLOCK_SELECTED + tl.arange(0, BLOCK_SELECTED)
    valid_selected = selected_offsets < SELECTED_COUNT
    centroid_slots = selected_offsets // VOCAB_PER_CENTROID
    token_slots = selected_offsets - centroid_slots * VOCAB_PER_CENTROID

    centroid_ids = tl.load(
        top_centroids + token_idx * top_stride_t + centroid_slots * top_stride_k,
        mask=valid_selected,
        other=0,
    )
    vocab_ids = tl.load(
        token_ordering + centroid_ids * VOCAB_PER_CENTROID + token_slots,
        mask=valid_selected,
        other=0,
    )

    d_offsets = tl.arange(0, BLOCK_D)
    valid_d = d_offsets < HIDDEN_SIZE
    hidden = tl.load(
        hidden_states + token_idx * hidden_stride_t + d_offsets * hidden_stride_d,
        mask=valid_d,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        lm_head_weight
        + vocab_ids[:, None] * lm_head_stride_v
        + d_offsets[None, :] * lm_head_stride_d,
        mask=valid_selected[:, None] & valid_d[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(weights * hidden[None, :], axis=1)
    scores = scores.to(tl.bfloat16).to(tl.float32)
    scores = tl.where(valid_selected, scores, -float("inf"))
    best_score, best_local_idx = tl.max(
        scores, axis=0, return_indices=True, return_indices_tie_break_left=True
    )

    best_selected = selected_block * BLOCK_SELECTED + best_local_idx
    best_centroid_slot = best_selected // VOCAB_PER_CENTROID
    best_token_slot = best_selected - best_centroid_slot * VOCAB_PER_CENTROID
    best_centroid = tl.load(
        top_centroids + token_idx * top_stride_t + best_centroid_slot * top_stride_k
    )
    best_token = tl.load(token_ordering + best_centroid * VOCAB_PER_CENTROID + best_token_slot)
    tl.store(partial_scores + token_idx * partial_score_stride_t + selected_block, best_score)
    tl.store(partial_tokens + token_idx * partial_token_stride_t + selected_block, best_token)


@triton.jit
def _sparse_argmax_reduce_kernel(
    partial_scores,
    partial_tokens,
    output_tokens,
    partial_score_stride_t,
    partial_token_stride_t,
    output_stride_t,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
) -> None:
    token_idx = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    valid_blocks = block_offsets < NUM_BLOCKS
    scores = tl.load(
        partial_scores + token_idx * partial_score_stride_t + block_offsets,
        mask=valid_blocks,
        other=-float("inf"),
    )
    _, best_block = tl.max(
        scores, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    token = tl.load(partial_tokens + token_idx * partial_token_stride_t + best_block)
    tl.store(output_tokens + token_idx * output_stride_t, token)


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


# --- real model dims ---
VOCAB_SIZE = 262144
NUM_CENTROIDS = 2048
VOCAB_PER_CENTROID = VOCAB_SIZE // NUM_CENTROIDS  # 128
CENTROID_TOP_K = 64
NUM_SELECTED = CENTROID_TOP_K * VOCAB_PER_CENTROID  # 8192
HIDDEN_SIZE = 256  # drafter draft-dim (text_config.hidden_size)
NUM_TOKENS = 1  # width-1 decode


def build_inputs(device: str = "cuda", seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=g)
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=g)
    # token_ordering: a permutation of [0, vocab). The grouping into centroids is by
    # contiguous runs of VOCAB_PER_CENTROID, so the gather is 64 contiguous 128-row runs.
    token_ordering = torch.randperm(VOCAB_SIZE, device=device, generator=g).to(torch.int64)
    # top_k_indices: the 64 selected centroid ids per token (random, distinct per token).
    top_k_indices = torch.stack(
        [torch.randperm(NUM_CENTROIDS, device=device, generator=g)[:CENTROID_TOP_K] for _ in range(NUM_TOKENS)]
    ).to(torch.int64)
    return hidden_states, lm_head_weight, token_ordering, top_k_indices


def run_fused(hidden_states, lm_head_weight, token_ordering, top_k_indices,
              block_arg: int, num_warps: int, num_stages: int | None):
    """One full fused sparse-argmax dispatch (blocks + reduce). Returns output_tokens."""
    num_tokens = int(hidden_states.shape[0])
    block_selected = _next_power_of_2(block_arg)
    num_blocks = triton.cdiv(NUM_SELECTED, block_selected)
    reduce_block = _next_power_of_2(num_blocks)
    block_d = _next_power_of_2(HIDDEN_SIZE)

    partial_scores = torch.empty((num_tokens, num_blocks), dtype=torch.float32, device=hidden_states.device)
    partial_tokens = torch.empty((num_tokens, num_blocks), dtype=torch.int64, device=hidden_states.device)
    output_tokens = torch.empty((num_tokens,), dtype=torch.int64, device=hidden_states.device)

    extra = {} if num_stages is None else {"num_stages": num_stages}
    _sparse_argmax_blocks_kernel[(num_tokens, num_blocks)](
        hidden_states, lm_head_weight, top_k_indices, token_ordering,
        partial_scores, partial_tokens,
        hidden_states.stride(0), hidden_states.stride(1),
        lm_head_weight.stride(0), lm_head_weight.stride(1),
        top_k_indices.stride(0), top_k_indices.stride(1),
        partial_scores.stride(0), partial_tokens.stride(0),
        VOCAB_PER_CENTROID=VOCAB_PER_CENTROID, SELECTED_COUNT=NUM_SELECTED,
        HIDDEN_SIZE=HIDDEN_SIZE, BLOCK_SELECTED=block_selected, BLOCK_D=block_d,
        num_warps=num_warps, **extra,
    )
    _sparse_argmax_reduce_kernel[(num_tokens,)](
        partial_scores, partial_tokens, output_tokens,
        partial_scores.stride(0), partial_tokens.stride(0), output_tokens.stride(0),
        NUM_BLOCKS=num_blocks, BLOCK_BLOCKS=reduce_block,
        num_warps=num_warps, **extra,
    )
    return output_tokens


def reference_torch(hidden_states, lm_head_weight, token_ordering, top_k_indices):
    """Exact PyTorch sparse path (the fused kernel's spec), for correctness anchoring."""
    clusters = token_ordering.view(NUM_CENTROIDS, VOCAB_PER_CENTROID)
    selected = clusters[top_k_indices]  # [t, top_k, vpc]
    selected = selected.reshape(NUM_TOKENS, -1)  # [t, num_selected]
    emb = lm_head_weight[selected.reshape(-1)].view(NUM_TOKENS, NUM_SELECTED, HIDDEN_SIZE)
    # bf16 logits materialized before argmax (matches kernel .to(bf16))
    logits = torch.einsum("td,tsd->ts", hidden_states.float(), emb.float()).to(torch.bfloat16).float()
    best = logits.argmax(dim=-1)
    return selected.gather(1, best.unsqueeze(1)).squeeze(1)


def bench_config(inputs, block_arg, num_warps, num_stages, rep_ms=200):
    fn = lambda: run_fused(*inputs, block_arg=block_arg, num_warps=num_warps, num_stages=num_stages)
    fn()  # compile/warm
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=50, rep=rep_ms, return_mode="median")
    return ms * 1000.0  # -> microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--rep-ms", type=int, default=200)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA"
    dev = torch.cuda.get_device_name(0)
    inputs = build_inputs()

    # correctness: reference vs default kernel config
    ref = reference_torch(*inputs)

    # served default launch: BLOCK_SELECTED=next_pow2(16)=16, num_warps=8, num_stages=triton default
    DEFAULT_BLOCK_ARG, DEFAULT_WARPS, DEFAULT_STAGES = 16, 8, None
    default_tokens = run_fused(*inputs, block_arg=DEFAULT_BLOCK_ARG, num_warps=DEFAULT_WARPS, num_stages=DEFAULT_STAGES)
    torch.cuda.synchronize()
    default_correct = bool(torch.equal(default_tokens, ref))

    default_us = bench_config(inputs, DEFAULT_BLOCK_ARG, DEFAULT_WARPS, DEFAULT_STAGES, args.rep_ms)

    # sweep grid (adapted to this kernel's tile params; BLOCK_D is pinned to next_pow2(hidden)=256)
    block_args = [8, 16, 32, 64, 128]
    warps_grid = [2, 4, 8]
    stages_grid = [2, 3, 4]

    results = []
    for ba in block_args:
        for w in warps_grid:
            for s in stages_grid:
                try:
                    toks = run_fused(*inputs, block_arg=ba, num_warps=w, num_stages=s)
                    torch.cuda.synchronize()
                    correct = bool(torch.equal(toks, ref))
                    us = bench_config(inputs, ba, w, s, args.rep_ms)
                    block_selected = _next_power_of_2(ba)
                    num_blocks = triton.cdiv(NUM_SELECTED, block_selected)
                    results.append({
                        "block_arg": ba, "block_selected": block_selected, "num_blocks": num_blocks,
                        "num_warps": w, "num_stages": s, "us": us, "correct": correct,
                        "speedup_vs_default": default_us / us,
                    })
                    print(f"BLOCK_SELECTED={block_selected:4d} (arg {ba:3d}, blocks {num_blocks:4d}) "
                          f"warps={w} stages={s}: {us:7.2f}us  x{default_us/us:5.3f}  correct={correct}")
                except Exception as exc:  # noqa: BLE001
                    print(f"block_arg={ba} warps={w} stages={s}: FAILED {exc!r}")
                    results.append({"block_arg": ba, "num_warps": w, "num_stages": s,
                                    "us": None, "correct": None, "error": repr(exc)})

    ok = [r for r in results if r.get("us") is not None and r.get("correct")]
    best = min(ok, key=lambda r: r["us"]) if ok else None

    summary = {
        "device": dev,
        "dims": {"num_tokens": NUM_TOKENS, "hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE,
                 "num_centroids": NUM_CENTROIDS, "vocab_per_centroid": VOCAB_PER_CENTROID,
                 "centroid_top_k": CENTROID_TOP_K, "num_selected": NUM_SELECTED},
        "served_default": {"block_arg": DEFAULT_BLOCK_ARG, "block_selected": _next_power_of_2(DEFAULT_BLOCK_ARG),
                           "num_warps": DEFAULT_WARPS, "num_stages": "triton_default",
                           "us": default_us, "correct": default_correct},
        "lm_head_bytes_per_call": NUM_SELECTED * HIDDEN_SIZE * 2,
        "best": best,
        "best_speedup": (default_us / best["us"]) if best else None,
        "best_delta_us": (default_us - best["us"]) if best else None,
        "n_iters_per_decode_step": 7,
        "results": results,
    }
    print("\n=== SUMMARY ===")
    print(f"device={dev}")
    print(f"served default (BLOCK_SELECTED=16, warps=8, stages=default): {default_us:.2f}us  correct={default_correct}")
    if best:
        print(f"best correct config: BLOCK_SELECTED={best['block_selected']} warps={best['num_warps']} "
              f"stages={best['num_stages']}: {best['us']:.2f}us")
        print(f"best speedup vs served default: x{default_us/best['us']:.4f}  "
              f"(delta {default_us-best['us']:+.2f}us per call, "
              f"x7/decode = {7*(default_us-best['us']):+.2f}us on D)")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"microbench-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
