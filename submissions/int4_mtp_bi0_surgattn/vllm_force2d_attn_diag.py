"""Output-neutral use_3d confirmation diagnostic (PR #785).

Opt-in via ``VLLM_SURGATTN_DIAG=1``. Installed ONLY in the surgattn-OFF variant
(``VLLM_SURGATTN=0``), where the force-2D patch is absent and the TRITON_ATTN
kernel's own launch gate (``triton_unified_attention.py`` ~L918-932) decides
3D split-KV vs 2D single-pass per forward. This wrapper answers the open #785
question -- *does the 3D split-KV branch actually fire at bi0's served KV
lengths?* -- by logging the gate's ``use_3d`` decision once per distinct
``max_seqlen_q`` seen.

It is **output-neutral**: it never mutates ``args``/``kwargs`` and always
forwards them unchanged to the real ``unified_attention``. It only *reads* the
kwargs the served ``forward()`` passes by keyword (``max_seqlen_q``,
``seqused_k``, ``seq_threshold_3D``, ``num_par_softmax_segments``,
``softmax_segm_*``) and reproduces the kernel's exact ``use_3d`` predicate for
logging. After the first 8 distinct outcomes it stops logging and is a pure
pass-through, so it adds no measurable decode overhead (and during CUDA-graph
replay it does not run at all).

Expected log (single-stream serving, BI=0, spec K=6):
* ``max_seqlen_q=1``  (M=1 target decode / MTP drafter) -> ``use_3d=True``  (3D)
* ``max_seqlen_q=7``  (M=K spec verify, K+1=7)           -> ``use_3d=False`` (2D)

i.e. surgattn's force-2D only ever changes the M=1 forwards; the M=K verify is
2D either way because the gate disqualifies 3D whenever ``max_seqlen_q > 1``.
"""

from __future__ import annotations

import functools
import sys

_PATCH_FLAG = "_int4_mtp_surgattn_diag_applied"
_WRAPPER_FLAG = "_int4_mtp_surgattn_diag_wrapper"
_MAX_DISTINCT_LOGS = 8


def apply(triton_attn) -> bool:
    """Wrap ``triton_attn.unified_attention`` with the pass-through logger."""
    if getattr(triton_attn, _PATCH_FLAG, False):
        return False

    orig = triton_attn.unified_attention
    if getattr(orig, _WRAPPER_FLAG, False):
        setattr(triton_attn, _PATCH_FLAG, True)
        return False

    # Same source the kernel reads for the gate's batch-invariance term.
    try:
        from vllm import envs

        is_batch_invariant = bool(envs.VLLM_BATCH_INVARIANT)
    except Exception:
        is_batch_invariant = False

    seen: set = set()

    @functools.wraps(orig)
    def unified_attention(*args, **kwargs):
        # READ-ONLY inspection; never mutate args/kwargs -> byte-identical to the
        # unwrapped call (numerics come entirely from orig()).
        if len(seen) < _MAX_DISTINCT_LOGS:
            try:
                msq = kwargs.get("max_seqlen_q")
                segm_o = kwargs.get("softmax_segm_output")
                segm_m = kwargs.get("softmax_segm_max")
                segm_e = kwargs.get("softmax_segm_expsum")
                st3d = kwargs.get("seq_threshold_3D")
                nps = kwargs.get("num_par_softmax_segments")
                seqused_k = kwargs.get("seqused_k")
                num_seqs = len(seqused_k) if seqused_k is not None else None
                # Exact replica of the kernel launch gate (L923-932).
                use_3d = not (
                    st3d is None
                    or nps is None
                    or segm_o is None
                    or segm_m is None
                    or segm_e is None
                    or (msq is not None and msq > 1)
                    or (num_seqs is not None and st3d is not None and num_seqs > st3d)
                    or is_batch_invariant
                )
                key = (msq, use_3d)
                if key not in seen:
                    seen.add(key)
                    print(
                        f"[surgattn_diag] unified_attention: max_seqlen_q={msq} "
                        f"num_seqs={num_seqs} seq_threshold_3D={st3d} "
                        f"num_par_softmax_segments={nps} segm_allocated={segm_o is not None} "
                        f"is_batch_invariant={is_batch_invariant} -> use_3d={use_3d} "
                        f"({'3D split-KV' if use_3d else '2D single-pass'})",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception:
                pass
        return orig(*args, **kwargs)

    setattr(unified_attention, _WRAPPER_FLAG, True)
    triton_attn.unified_attention = unified_attention
    setattr(triton_attn, _PATCH_FLAG, True)
    print(
        "[surgattn_diag] output-neutral use_3d confirmation wrapper installed "
        "(VLLM_SURGATTN_DIAG=1); force-2D patch is OFF so the kernel launch gate "
        "selects the branch -- logging its use_3d decision per distinct max_seqlen_q",
        file=sys.stderr,
        flush=True,
    )
    return True
