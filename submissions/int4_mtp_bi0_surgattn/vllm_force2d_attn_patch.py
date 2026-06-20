"""Surgical force-2D attention patch: byte-exact, split-invariant attention.

This submission runs the engine under ``VLLM_BATCH_INVARIANT=0`` (fast,
non-deterministic-reduction GEMM/RMSNorm/softmax kernels) to avoid the global
"BI-tax" that the batch-invariant baseline (``int4_mtp_batchinv``) pays on every
op. The one place batch-invariance is actually load-bearing for this stack is the
attention reduction order: it is what makes the spec-decode M=K verify forward
byte-identical to the M=1 AR decode forward, which the strict greedy-identity gate
requires. We recover exactly that property -- and nothing else from BI=1 -- with a
single surgical change to the TRITON_ATTN path.

Mechanism. ``vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward``
calls the bare module-level name ``unified_attention`` (imported into the backend
module's globals at import time). We rebind that name to a thin wrapper that nulls
the three ``softmax_segm_{output,max,expsum}`` scratch buffers on every call. Per
the kernel's own launch gate (``triton_unified_attention.py`` ~L918-932, condition
#1: "No intermediate tiled softmax buffers for the 3D kernel have been
allocated"), passing ``softmax_segm_* = None`` forces ``use_3d = False`` -- the 2D
single-pass branch -- for BOTH the q=1 decode and the q=K spec-verify forwards.
The 2D path then reuses ``out`` as a never-written placeholder for the segm
pointers and never launches ``reduce_segments``.

Why this is byte-exact (strict greedy identity holds under BI=0):

* The 2D single-pass output of a query row is provably invariant to how many query
  rows share the launch (M): the inner tile loop keeps per-row ``(max, expsum,
  acc)`` state with no cross-row data dependency, so an M=K batched launch and K
  serial M=1 launches hit the same tiles in the same order and accumulate
  identically. (Measured: PR #747, batched-2D vs serial-M=1-2D = 0 bit-diff.)
* It is also invariant to the per-sequence ``seq_len`` passed (verify passes the
  full length L for all K rows; each row's decode counterpart at position p passed
  p+1): keys beyond a row's causal position get logit ``-inf`` -> ``exp(-inf - m)
  = 0.0`` exactly (IEEE-754), a true no-op on that row's accumulator, so the tail
  length is numerically irrelevant.
* Under BI=0 the only other decode-vs-verify divergent op would be the GEMM, but
  the int4 Marlin GEMV is bit-exactly M-invariant (PR #736), and RMSNorm is
  per-token (row-independent). So once attention is forced 2D everywhere, the
  decode and verify forwards are byte-identical end-to-end.

The 3D split-KV (flash-decoding) decode path we give up only ever helped fill SMs
at long context on a single stream; attention is a small fraction of decode wall
time here, so the BI-tax we avoid on every GEMM/norm dominates the trade.

The wrapper is a no-op for any call that already has no segm buffers (prefill,
or an engine that didn't allocate them), so it never changes numerics on paths
that were already 2D.
"""

from __future__ import annotations

import functools
import os
import sys

# Marker set on the backend module once patched, so apply() is idempotent across
# the multiple processes / repeated plugin loads that touch the same module.
_PATCH_FLAG = "_int4_mtp_force2d_attn_patch_applied"
# Marker set on the wrapper itself, so we never double-wrap if apply() somehow
# runs against an already-wrapped binding.
_WRAPPER_FLAG = "_int4_mtp_force2d_wrapper"

# A/B toggle (PR #794, surgattn-combine attribution). The shipped submission
# forces 2D for byte-exact greedy identity; this is the DEFAULT. Setting
# ``VLLM_SURGATTN=0`` (or off/false/no) disables the force, leaving vanilla vLLM
# dispatch so the M=1 decode/drafter forwards take the 3D split-KV path (the
# surgattn-OFF +6.69% arm). It exists ONLY to measure the 2D-vs-3D delta on the
# bi0 stack; the shipped leaderboard path is unchanged (toggle absent => force-2D).
_SURGATTN_ENV = "VLLM_SURGATTN"


def _surgattn_enabled() -> bool:
    return os.environ.get(_SURGATTN_ENV, "1").strip().lower() not in (
        "0", "off", "false", "no", ""
    )


def apply(triton_attn) -> bool:
    """Rebind ``triton_attn.unified_attention`` to the force-2D wrapper.

    ``triton_attn`` is the imported ``vllm.v1.attention.backends.triton_attn``
    module. Returns ``True`` if the patch was applied, ``False`` if it was
    already present or intentionally disabled via ``VLLM_SURGATTN=0``.
    """
    if getattr(triton_attn, _PATCH_FLAG, False):
        return False

    if not _surgattn_enabled():
        # surgattn OFF: leave vanilla unified_attention untouched so M=1 forwards
        # take the 3D split-KV path. Mark the module so we don't re-evaluate.
        setattr(triton_attn, _PATCH_FLAG, True)
        print(
            "[int4_mtp_force2d] VLLM_SURGATTN=0: force-2D DISABLED — M=1 forwards "
            "use vanilla 3D split-KV (surgattn-OFF A/B arm; NOT byte-identical)",
            file=sys.stderr,
            flush=True,
        )
        return False

    orig = triton_attn.unified_attention
    if getattr(orig, _WRAPPER_FLAG, False):
        setattr(triton_attn, _PATCH_FLAG, True)
        return False

    @functools.wraps(orig)
    def unified_attention(*args, **kwargs):
        # Null the 3D scratch -> kernel launch gate selects the 2D single-pass
        # path for q=1 decode and q=K verify alike. See module docstring for the
        # byte-exactness argument. The served forward() passes these by keyword.
        kwargs["softmax_segm_output"] = None
        kwargs["softmax_segm_max"] = None
        kwargs["softmax_segm_expsum"] = None
        return orig(*args, **kwargs)

    setattr(unified_attention, _WRAPPER_FLAG, True)
    triton_attn.unified_attention = unified_attention
    setattr(triton_attn, _PATCH_FLAG, True)
    # One positive line per process so server.log proves the 2D force is live in
    # the worker (silence would otherwise be ambiguous with "patch never ran").
    print(
        "[int4_mtp_force2d] unified_attention wrapped: forcing 2D single-pass "
        "attention (use_3d=False) for decode and verify under BI=0",
        file=sys.stderr,
        flush=True,
    )
    return True
