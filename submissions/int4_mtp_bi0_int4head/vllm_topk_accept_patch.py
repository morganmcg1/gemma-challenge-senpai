"""Top-k-match speculative-decode accept-branch (PR #816).

vLLM v1's greedy rejection sampler accepts a draft token ONLY when it exactly
equals the target model's argmax. The greedy kernel
(``vllm/v1/sample/rejection_sampler.py``, ``rejection_greedy_sample_kernel``) hard-
wires ``rejected = draft_token_id != target_argmax_id`` and emits the argmax, so
the served output is byte-identical to plain greedy AR of the int4 target. This
patch relaxes that test, behind the ``TOPK_ACCEPT_K`` env var, to:

    accept the draft token iff it is in topk(target_logits, K) at that position.

Larger K accepts more draft tokens -> longer mean accepted length (E_accept) ->
the int4 body verify-GEMM and int4 lm_head are amortized over more output tokens
-> higher TPS. The cost is quality: an accepted token is no longer guaranteed to
be the greedy argmax, so the output diverges from greedy. Byte-identity is
explicitly WAIVED for this phase under Issue #784 ("Byte-identical outputs are
not sacred ... if a variant keeps quality inside the 5% band, treat it as worth
testing"). PR #816 measures realized E_accept(K) and TPS(K) against the full
quality panel (PPL / AIME / MMLU-Pro / GPQA-Diamond / GSM8K) to find the largest
K whose quality stays inside the #784 floors.

Mechanism (NO new kernel). ``rejection_sample`` (the torch-level driver, line ~392)
already holds the full target logits and computes
``target_argmax = target_logits.argmax(dim=-1)``, which it feeds to the unmodified
greedy Triton kernel; that kernel accepts position ``pos`` iff
``draft[pos] == target_argmax[pos]`` and stores ``target_argmax[pos]``. We rebind
the module-level ``rejection_sample`` to a wrapper that, in the pure-greedy path,
hands the SAME kernel an EFFECTIVE argmax:

    effective_argmax[pos] = draft[pos]       if draft[pos] in topk(logits[pos], K)
                          = real_argmax[pos]  otherwise

With this substitution the unmodified kernel:
  * ACCEPTS (``rejected`` stays False, stores ``effective_argmax == draft``) at every
    position whose draft is in the target top-K, emitting that draft token, and
  * REJECTS (``effective_argmax == real_argmax != draft``, since a draft outside the
    top-K can never equal the rank-0 argmax) at the first position whose draft is
    NOT in the top-K, emitting the true argmax and stopping,
i.e. it accepts exactly "the longest prefix of draft tokens that all fall in the
target top-K", which is the intended top-k-match rule. The kernel's bonus-token
append (the genuine target argmax at the post-draft position) is unchanged.

K=1 is the exact original: a draft is in top-1 iff it equals the argmax, so
``effective_argmax == real_argmax`` and the kernel reduces to the upstream exact-
match branch, bit-for-bit. K>=2 is a strict superset of the K=1 accepts (every
exact-match position still accepts), so E_accept is monotonic in K.

No-op contract. K is read from the environment at apply() time. If ``TOPK_ACCEPT_K``
is unset or <= 1 we do NOT rebind ``rejection_sample`` at all -- the leaderboard
serving path is the untouched upstream function, byte-identical to the shipped
submission. The wrapper (installed only for K >= 2) ALSO re-reads K per call and
delegates to the captured original for any K<=1 / synthetic-oracle / non-greedy
call, so it can never alter the random-sampling or #813 synthetic-oracle paths.

Active-path note: the shipped submission serves via the classic
``vllm.v1.worker.gpu_model_runner.GPUModelRunner`` (the module the sibling
``vllm_attn_group_patch`` patches), which imports
``vllm.v1.sample.rejection_sampler.RejectionSampler`` -- whose ``forward`` calls the
module-global ``rejection_sample`` we rebind here. The newer
``vllm.v1.worker.gpu.spec_decode.rejection_sampler`` is a different, unused runner
on this pinned vLLM 0.22.0 stack.

Same ``apply(module)`` contract as ``vllm_attn_group_patch`` / ``vllm_force2d_attn_patch``,
installed by ``sitecustomize.py`` on first import of ``vllm.v1.sample.rejection_sampler``.
"""

from __future__ import annotations

import functools
import os
import sys

# Marker set on the rejection_sampler module once handled, so apply() is
# idempotent across the multiple processes / repeated plugin loads that touch the
# same module object.
_PATCH_FLAG = "_int4_mtp_topk_accept_patch_applied"
# Marker set on the wrapper itself, so we never double-wrap.
_WRAPPER_FLAG = "_int4_mtp_topk_accept_wrapper"
_ENV = "TOPK_ACCEPT_K"


def _read_k() -> int:
    """Read the top-k accept width from the environment (default 1 == exact)."""
    try:
        return int(os.environ.get(_ENV, "1"))
    except (TypeError, ValueError):
        return 1


def apply(rs) -> bool:
    """Rebind ``rs.rejection_sample`` to the top-k-match wrapper when
    ``TOPK_ACCEPT_K >= 2``.

    ``rs`` is the imported ``vllm.v1.sample.rejection_sampler`` module. All free
    symbols are resolved from that live module so we never guess against a vLLM
    version. Returns True if a wrapper was installed, False if left as a no-op
    (K<=1) or already handled.
    """
    if getattr(rs, _PATCH_FLAG, False):
        return False

    k_apply = _read_k()
    if k_apply <= 1:
        # Exact-argmax greedy accept: do NOT rebind. Leaderboard path untouched.
        setattr(rs, _PATCH_FLAG, True)
        print(
            f"[int4_mtp_topk_accept] {_ENV}={os.environ.get(_ENV)!r} (<=1): no-op, "
            "exact-argmax greedy accept (byte-identical to shipped submission)",
            file=sys.stderr,
            flush=True,
        )
        return False

    orig = rs.rejection_sample
    if getattr(orig, _WRAPPER_FLAG, False):
        setattr(rs, _PATCH_FLAG, True)
        return False

    import torch

    @functools.wraps(orig)
    def rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode: bool = False,
        synthetic_conditional_rates=None,
    ):
        k = _read_k()
        # Relax ONLY the pure-greedy, non-synthetic path (the benchmark decode
        # condition: temp=0 -> all_greedy, no synthetic oracle). Anything else
        # falls through to the exact upstream implementation, untouched.
        if k <= 1 or synthetic_mode or not sampling_metadata.all_greedy:
            return orig(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
                synthetic_mode=synthetic_mode,
                synthetic_conditional_rates=synthetic_conditional_rates,
            )

        batch_size = len(num_draft_tokens)
        num_tokens = draft_token_ids.shape[0]
        vocab_size = target_logits.shape[-1]
        device = target_logits.device
        k_eff = min(k, vocab_size)

        # Rank-0 exact-match choice the upstream kernel would use.
        target_argmax = target_logits.argmax(dim=-1)  # [num_tokens], int64
        # Where the draft is within the target top-K, point the kernel's "argmax"
        # AT the draft so the unmodified kernel accepts + emits it; elsewhere keep
        # the true argmax so the kernel rejects at that position and stops.
        if num_tokens > 0:
            topk_ids = target_logits.topk(k_eff, dim=-1).indices  # [num_tokens, k]
            in_topk = (
                topk_ids == draft_token_ids.to(topk_ids.dtype).unsqueeze(1)
            ).any(dim=1)
            effective_argmax = torch.where(
                in_topk,
                draft_token_ids.to(target_argmax.dtype),
                target_argmax,
            )
        else:
            effective_argmax = target_argmax

        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            rs.PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=device,
        )
        # all_greedy here -> is_greedy=None (kernel treats None as all-greedy);
        # uniform/synthetic pointers None, SYNTHETIC_MODE False: byte-identical
        # kernel launch to the upstream greedy fast-path, only target_argmax swapped
        # for effective_argmax. The all_greedy branch returns immediately upstream
        # (the random kernel is never reached), so we mirror that and return here.
        rs.rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            effective_argmax,
            bonus_token_ids,
            None,
            max_spec_len,
            None,
            None,
            SYNTHETIC_MODE=False,
        )
        return output_token_ids

    setattr(rejection_sample, _WRAPPER_FLAG, True)
    rs.rejection_sample = rejection_sample
    setattr(rs, _PATCH_FLAG, True)
    print(
        f"[int4_mtp_topk_accept] rejection_sample wrapped: greedy accept relaxed to "
        f"draft in topk(target_logits, K={k_apply}); byte-identity WAIVED under #784",
        file=sys.stderr,
        flush=True,
    )
    return True
