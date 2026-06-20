"""Fused Triton sparse-argmax for the Gemma4 MTP drafter centroid top-token.

Ported from ``fa2sw_precache_kenyan/sitecustomize.py`` (the fused-sparse-argmax
subsystem only). Runs on STOCK ``vllm==0.22.0``. Applied from ``sitecustomize.py``
on first import of ``vllm.model_executor.models.gemma4_mtp``.

What it replaces: ``Gemma4MTPMaskedEmbedder.get_top_tokens`` — the drafter's sparse
centroid argmax. The stock path materializes a (num_tokens, num_selected) bf16
sparse-logit tensor via an einsum, then argmaxes. The fused path computes the same
bf16-rounded scores and argmax in two Triton kernels (block-parallel partial argmax
+ reduction), skipping the intermediate logit tensor. It returns the same vocab
token id the stock argmax would (modulo bf16-tie ordering — see below).

Greedy-identity safety: this only changes the DRAFTER's proposed token. The verifier
still emits the target argmax at T=0, so the served greedy token sequence is
identical regardless of which token the drafter proposes; only acceptance rate / TPS
can move. The freq-bias and dixie-accept experiment paths from the donor are NOT
ported, so the stock ``_select_and_score`` (and therefore the PyTorch fallback) is
left byte-exact. On any unsupported shape/dtype/device the patch falls back to the
stock ``get_top_tokens`` (or raises if FUSED_SPARSE_ARGMAX_REQUIRE=1).
"""

from __future__ import annotations

import os
import sys
from typing import Any


FUSED_SPARSE_ARGMAX = os.environ.get("FUSED_SPARSE_ARGMAX", "1") == "1"
FUSED_SPARSE_ARGMAX_REQUIRE = os.environ.get("FUSED_SPARSE_ARGMAX_REQUIRE") == "1"
FUSED_SPARSE_ARGMAX_BLOCK = int(os.environ.get("FUSED_SPARSE_ARGMAX_BLOCK", "16"))
_FUSED_SPARSE_ARGMAX_KERNELS: Any | None = None


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def _get_fused_sparse_argmax_kernels() -> Any:
    global _FUSED_SPARSE_ARGMAX_KERNELS
    if _FUSED_SPARSE_ARGMAX_KERNELS is not None:
        return _FUSED_SPARSE_ARGMAX_KERNELS

    import triton
    import triton.language as tl

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

        selected_offsets = selected_block * BLOCK_SELECTED + tl.arange(
            0, BLOCK_SELECTED
        )
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
        # The PyTorch sparse path materializes bf16 logits before argmax.
        scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(valid_selected, scores, -float("inf"))
        best_score, best_local_idx = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        best_selected = selected_block * BLOCK_SELECTED + best_local_idx
        best_centroid_slot = best_selected // VOCAB_PER_CENTROID
        best_token_slot = best_selected - best_centroid_slot * VOCAB_PER_CENTROID
        best_centroid = tl.load(
            top_centroids + token_idx * top_stride_t + best_centroid_slot * top_stride_k
        )
        best_token = tl.load(
            token_ordering + best_centroid * VOCAB_PER_CENTROID + best_token_slot
        )
        tl.store(
            partial_scores + token_idx * partial_score_stride_t + selected_block,
            best_score,
        )
        tl.store(
            partial_tokens + token_idx * partial_token_stride_t + selected_block,
            best_token,
        )

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
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        token = tl.load(
            partial_tokens + token_idx * partial_token_stride_t + best_block
        )
        tl.store(output_tokens + token_idx * output_stride_t, token)

    _FUSED_SPARSE_ARGMAX_KERNELS = (
        triton,
        _sparse_argmax_blocks_kernel,
        _sparse_argmax_reduce_kernel,
    )
    return _FUSED_SPARSE_ARGMAX_KERNELS


def _fallback_sparse_argmax(
    self: Any,
    original_get_top_tokens: Any,
    hidden_states: Any,
    lm_head_weight: Any,
    reason: Exception,
) -> Any:
    if FUSED_SPARSE_ARGMAX_REQUIRE:
        raise RuntimeError(
            "FUSED_SPARSE_ARGMAX_REQUIRE=1 but fusion failed"
        ) from reason
    if not getattr(self, "_pupa_fused_sparse_argmax_warned", False):
        self._pupa_fused_sparse_argmax_warned = True
        print(
            f"[bi0-fused-sparse-argmax] falling back to PyTorch path: {reason!r}",
            file=sys.stderr,
            flush=True,
        )
    return original_get_top_tokens(self, hidden_states, lm_head_weight)


def apply(module: Any) -> None:
    import torch

    embedder_cls = module.Gemma4MTPMaskedEmbedder
    original_get_top_tokens = embedder_cls.get_top_tokens

    def get_top_tokens_fused(self: Any, hidden_states: Any, lm_head_weight: Any) -> Any:
        if not FUSED_SPARSE_ARGMAX:
            return original_get_top_tokens(self, hidden_states, lm_head_weight)
        try:
            if (
                hidden_states.device.type != "cuda"
                or lm_head_weight.device.type != "cuda"
            ):
                raise RuntimeError("fusion requires CUDA tensors")
            if (
                hidden_states.dtype != torch.bfloat16
                or lm_head_weight.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "fusion currently preserves exact PyTorch argmax only for bf16"
                )
            hidden_size = int(self.hidden_size)
            if hidden_size <= 0 or hidden_size > 1024:
                raise RuntimeError(f"unsupported hidden_size={hidden_size}")

            triton, blocks_kernel, reduce_kernel = _get_fused_sparse_argmax_kernels()
            num_tokens = int(hidden_states.shape[0])
            selected_count = int(self.num_selected)
            block_selected = _next_power_of_2(FUSED_SPARSE_ARGMAX_BLOCK)
            num_blocks = triton.cdiv(selected_count, block_selected)
            reduce_block = _next_power_of_2(num_blocks)
            block_d = _next_power_of_2(hidden_size)

            _, top_k_indices = torch.topk(
                self.centroids(hidden_states),
                k=self.centroid_intermediate_top_k,
                dim=-1,
                sorted=False,
            )
            partial_scores = torch.empty(
                (num_tokens, num_blocks),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            partial_tokens = torch.empty(
                (num_tokens, num_blocks),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            output_tokens = torch.empty(
                (num_tokens,),
                dtype=torch.int64,
                device=hidden_states.device,
            )

            blocks_kernel[(num_tokens, num_blocks)](
                hidden_states,
                lm_head_weight,
                top_k_indices,
                self.token_ordering,
                partial_scores,
                partial_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                lm_head_weight.stride(0),
                lm_head_weight.stride(1),
                top_k_indices.stride(0),
                top_k_indices.stride(1),
                partial_scores.stride(0),
                partial_tokens.stride(0),
                VOCAB_PER_CENTROID=int(self.vocab_size_per_centroid),
                SELECTED_COUNT=selected_count,
                HIDDEN_SIZE=hidden_size,
                BLOCK_SELECTED=block_selected,
                BLOCK_D=block_d,
                num_warps=8,
            )
            reduce_kernel[(num_tokens,)](
                partial_scores,
                partial_tokens,
                output_tokens,
                partial_scores.stride(0),
                partial_tokens.stride(0),
                output_tokens.stride(0),
                NUM_BLOCKS=num_blocks,
                BLOCK_BLOCKS=reduce_block,
                num_warps=8,
            )
            return output_tokens
        except Exception as exc:
            return _fallback_sparse_argmax(
                self,
                original_get_top_tokens,
                hidden_states,
                lm_head_weight,
                exc,
            )

    embedder_cls.get_top_tokens = get_top_tokens_fused
    print(
        f"[bi0-fused-sparse-argmax] patched Gemma4MTPMaskedEmbedder top-token path "
        f"in pid {os.getpid()} (enabled={FUSED_SPARSE_ARGMAX}, "
        f"require={FUSED_SPARSE_ARGMAX_REQUIRE}, block={FUSED_SPARSE_ARGMAX_BLOCK})",
        file=sys.stderr,
        flush=True,
    )
