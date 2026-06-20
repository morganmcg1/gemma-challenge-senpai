"""Forward-type-gated force-2D attention patch (surgattn attribution card).

This is the shipped ``int4_mtp_bi0_surgattn`` force-2D patch, extended with a
*forward-type discriminator* so the 2D force can be applied to the main-model
forward only while letting the kernel gate pick the 3D split-KV path on the
drafter proposer forwards. See ``_surgattn_arm`` for the full argument; the short
version:

``vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward`` calls the
bare module-level name ``unified_attention``. We rebind that name to a wrapper
that, **only when the arm says this forward must be 2D**, nulls the three
``softmax_segm_{output,max,expsum}`` scratch buffers. Per the kernel's launch
gate (``triton_unified_attention.py`` ~L918-932), passing any ``softmax_segm_* =
None`` forces ``use_3d = False`` (the 2D single-pass branch). Leaving them intact
lets the gate pick 3D when its own conditions hold (``max_seqlen_q <= 1`` etc.).

Byte-exactness of the 2D path (unchanged from bi0): the 2D single-pass output of
a query row is provably invariant to the query width M and to the per-sequence
seq_len, so an M=1 decode forward and the M=K verify forward accumulate
identically. The arm logic only changes *which* forwards are kept on that
byte-exact 2D path:

* ``control_2d``      every M=1 forward (== shipped bi0).
* ``drafter_only_3d`` the main-model forwards (byte-identical emitted tokens),
                      while the drafter proposer forwards may go 3D — those only
                      produce draft proposals, which at temp=0 cannot change the
                      emitted target-argmax sequence.
* ``all_3d``          nothing forced (surgattn OFF; the identity-breaking anchor).

The wrapper is otherwise a no-op for any call that already has no segm buffers
(prefill, or an engine that didn't allocate them).
"""

from __future__ import annotations

import functools
import sys

import _surgattn_arm as _arm

# Marker set on the backend module once patched, so apply() is idempotent across
# the multiple processes / repeated plugin loads that touch the same module.
_PATCH_FLAG = "_int4_mtp_force2d_attn_patch_applied"
# Marker set on the wrapper itself, so we never double-wrap if apply() somehow
# runs against an already-wrapped binding.
_WRAPPER_FLAG = "_int4_mtp_force2d_wrapper"

# One-shot dispatch-confirmation logging, so server.log proves the discriminator
# routed each forward type to the intended kernel path (the smoke check reads
# these). Keyed (in_drafter, forced_2d) -> already-logged.
_seen: set[tuple[bool, bool]] = set()


def _log_first(in_drafter: bool, forced_2d: bool, segm_present: bool) -> None:
    key = (in_drafter, forced_2d)
    if key in _seen:
        return
    _seen.add(key)
    where = "drafter-proposer" if in_drafter else "main-model"
    if forced_2d:
        msg = f"forcing 2D (use_3d=False) on {where} forward"
    else:
        # Not forced: 3D engages iff the gate's own conditions hold, which needs
        # the segm buffers to have been allocated by the caller.
        gate = "segm buffers PRESENT -> gate may pick 3D" if segm_present else (
            "segm buffers ABSENT -> stays 2D regardless"
        )
        msg = f"allowing gate choice on {where} forward ({gate})"
    print(
        f"[surgattn-attrib arm={_arm.ARM}] {msg}",
        file=sys.stderr,
        flush=True,
    )


def apply(triton_attn) -> bool:
    """Rebind ``triton_attn.unified_attention`` to the arm-gated wrapper.

    ``triton_attn`` is the imported ``vllm.v1.attention.backends.triton_attn``
    module. Returns ``True`` if the patch was applied, ``False`` if it was
    already present.
    """
    if getattr(triton_attn, _PATCH_FLAG, False):
        return False

    orig = triton_attn.unified_attention
    if getattr(orig, _WRAPPER_FLAG, False):
        setattr(triton_attn, _PATCH_FLAG, True)
        return False

    @functools.wraps(orig)
    def unified_attention(*args, **kwargs):
        in_drafter = _arm.in_drafter_propose()
        forced_2d = _arm.should_force_2d()
        _log_first(in_drafter, forced_2d, kwargs.get("softmax_segm_output") is not None)
        if forced_2d:
            # Null the 3D scratch -> kernel launch gate selects the 2D
            # single-pass path. The served forward() passes these by keyword.
            kwargs["softmax_segm_output"] = None
            kwargs["softmax_segm_max"] = None
            kwargs["softmax_segm_expsum"] = None
        return orig(*args, **kwargs)

    setattr(unified_attention, _WRAPPER_FLAG, True)
    triton_attn.unified_attention = unified_attention
    setattr(triton_attn, _PATCH_FLAG, True)
    print(
        f"[surgattn-attrib arm={_arm.ARM}] unified_attention wrapped: "
        "force-2D gated on forward type (drafter-proposer vs main-model) under BI=0",
        file=sys.stderr,
        flush=True,
    )
    return True
