"""splitkv-verify: route small multi-query-row (spec-verify) attention batches
to vLLM's 3D split-KV (FlashDecoding) path.

vLLM's Triton ``unified_attention`` (vllm/v1/attention/ops/triton_unified_attention.py)
gates the 3D split-KV path OFF whenever ``max_seqlen_q > 1``. That guard is a
*performance heuristic* for large prefill batches that already saturate the 2D
launch grid -- it is NOT a correctness requirement: split-KV partitions the KV
(reduction) axis, orthogonal to the query rows, and the per-segment online-
softmax merge (``reduce_segments``, launched once per query token) is already
multi-row-correct.

At concurrency=1 the speculative *verify* step issues M = num_speculative+1
query rows (M=8 for this submission). The 2D grid is then
(ceil(M/BLOCK_Q), num_kv_heads) ~= a handful of CTAs, badly under-occupying the
A10G's 80 SMs. PR #39 measured 53 us for this 2D verify-attention vs 12 us for
the identical-bytes M=1 3D path (4.14x). This patch redirects the verify range
to 3D split-KV, which fills the GPU with ``num_par_softmax_segments`` extra CTAs.

Mechanism: in this pinned vLLM build ``max_seqlen_q`` is consumed ONLY by the
2D-vs-3D dispatch test inside ``unified_attention`` (verified: it is never
forwarded to the kernel launch; the kernel drives true multi-row work from
``cu_seqlens_q`` / ``BLOCK_Q``). So for a verify batch that is provably safe for
3D we override ``max_seqlen_q`` to 1, which selects the 3D path while leaving the
actual computation untouched.

Safety gates (all required before redirecting; otherwise stock dispatch runs):
  * ``SPLITKV_VERIFY=1`` (env, default on);
  * ``1 < max_seqlen_q <= SPLITKV_VERIFY_MAX_Q`` (default 64): only the
    spec-verify / tiny-prefill regime, where 2D is occupancy-bound. Large
    prefill (prompt_len > 64) keeps 2D, which already saturates the grid.
  * ``q.shape[0] <= softmax_segm_output.shape[0]``: the per-segment buffers are
    indexed by global query-token row, so they must hold every row. (For M=1
    this is exactly the stock ``num_seqs <= seq_threshold_3D`` gate, since then
    q.shape[0] == num_seqs.)
  * ``num_seqs <= seq_threshold_3D``: preserve the stock 2D-occupancy threshold.
  * not batch-invariant.
Any failure in the decision path falls through to the stock dispatch (fail-open).

Greedy identity: 3D split-KV is the same math as single-pass attention; only
bf16 rounding order differs (PR #39 measured max_abs_err 6.1e-5 for the M=1 3D
path vs an SDPA reference at these context lengths, well under the 1e-4 SWA
tolerance). The M=1 *decode* path is ALREADY 3D in the baseline, so this merely
makes the verify step numerically consistent with it. Greedy token identity is
validated separately (gen_greedy_reference + validate_submission).
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import os
import sys
from typing import Any

OPS_TARGET = "vllm.v1.attention.ops.triton_unified_attention"
BACKEND_TARGET = "vllm.v1.attention.backends.triton_attn"

SPLITKV_VERIFY = os.environ.get("SPLITKV_VERIFY", "1") == "1"
SPLITKV_VERIFY_MAX_Q = int(os.environ.get("SPLITKV_VERIFY_MAX_Q", "64"))
_LOG_LIMIT = int(os.environ.get("SPLITKV_VERIFY_LOG", "5"))

_stats = {"redirected": 0}


def _batch_invariant() -> bool:
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua

        return bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:  # noqa: BLE001 - fail-open
        return False


def would_redirect(
    *,
    q_rows: int,
    max_seqlen_q: int,
    segm_rows: int | None,
    seq_threshold_3D: int | None,
    num_seqs: int,
) -> bool:
    """Pure predicate: would a batch with these shapes be routed to 3D split-KV?

    Exposed so the profiling harness can report ``used_3d_split_kv`` accurately
    without having to re-run the dispatch.
    """
    if not SPLITKV_VERIFY:
        return False
    if seq_threshold_3D is None or segm_rows is None:
        return False
    if not (1 < int(max_seqlen_q) <= SPLITKV_VERIFY_MAX_Q):
        return False
    if int(q_rows) > int(segm_rows):
        return False
    if int(num_seqs) > int(seq_threshold_3D):
        return False
    if _batch_invariant():
        return False
    return True


def _should_redirect(kw: dict) -> bool:
    mq = kw.get("max_seqlen_q")
    q = kw.get("q")
    segm = kw.get("softmax_segm_output")
    thr = kw.get("seq_threshold_3D")
    seqused = kw.get("seqused_k")
    if mq is None or q is None or segm is None or thr is None or seqused is None:
        return False
    try:
        return would_redirect(
            q_rows=int(q.shape[0]),
            max_seqlen_q=int(mq),
            segm_rows=int(segm.shape[0]),
            seq_threshold_3D=int(thr),
            num_seqs=int(seqused.shape[0]),
        )
    except Exception:  # noqa: BLE001 - fail-open
        return False


def _make_wrapper(orig: Any) -> Any:
    @functools.wraps(orig)
    def unified_attention(*args, **kwargs):
        # The vLLM Triton backend and the profiling harness both call this with
        # all-keyword arguments. Only the all-keyword fast path is redirected;
        # any positional call passes straight through (stock dispatch -> safe).
        if not args and SPLITKV_VERIFY:
            try:
                if _should_redirect(kwargs):
                    kwargs = dict(kwargs)
                    mq = kwargs["max_seqlen_q"]
                    kwargs["max_seqlen_q"] = 1
                    _stats["redirected"] += 1
                    if _stats["redirected"] <= _LOG_LIMIT:
                        print(
                            f"[splitkv-verify] verify batch M={mq} "
                            f"q_rows={int(kwargs['q'].shape[0])} -> 3D split-KV "
                            f"(n={_stats['redirected']})",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001 - fail-open
                print(
                    f"[splitkv-verify] redirect skipped, baseline kept: {exc!r}",
                    flush=True,
                )
        return orig(*args, **kwargs)

    unified_attention._splitkv_verify_wrapped = True  # type: ignore[attr-defined]
    unified_attention._splitkv_orig = orig  # type: ignore[attr-defined]
    return unified_attention


def _patch_backend_ref(wrapped: Any) -> None:
    """triton_attn binds ``unified_attention`` at import (``from ops import ...``),
    so if that backend module is already loaded we must swap its reference too."""
    mod = sys.modules.get(BACKEND_TARGET)
    if mod is None:
        return
    cur = getattr(mod, "unified_attention", None)
    if cur is not None and not getattr(cur, "_splitkv_verify_wrapped", False):
        mod.unified_attention = wrapped


def _patch_ops_module(module: Any) -> None:
    orig = getattr(module, "unified_attention", None)
    if orig is None or getattr(orig, "_splitkv_verify_wrapped", False):
        return
    wrapped = _make_wrapper(orig)
    module.unified_attention = wrapped
    _patch_backend_ref(wrapped)
    print(
        "[splitkv-verify] wrapped unified_attention "
        f"(redirect 1<M<={SPLITKV_VERIFY_MAX_Q} verify batches to 3D split-KV)",
        flush=True,
    )


def _patch_ops_module_safe(module: Any) -> None:
    try:
        _patch_ops_module(module)
    except Exception as exc:  # noqa: BLE001 - fail-open
        print(f"[splitkv-verify] patch error, baseline kept: {exc!r}", flush=True)


def install() -> bool:
    """In-process install (profiling harness / tests). Returns True if active."""
    if not SPLITKV_VERIFY:
        return False
    try:
        import vllm.v1.attention.ops.triton_unified_attention as ua
    except Exception as exc:  # noqa: BLE001
        print(f"[splitkv-verify] vLLM ops module unavailable: {exc!r}", flush=True)
        return False
    _patch_ops_module_safe(ua)
    return True


# --- import-time meta-path finder (served subprocess via sitecustomize) -------
class _ChainLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _patch_ops_module_safe(module)


class _ChainFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str) -> None:
        self._target = target
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _ChainLoader(spec.loader)
        return spec


if SPLITKV_VERIFY:
    # If the ops module is already imported (harness path), patch it in place;
    # otherwise register a finder so it is patched the moment vLLM imports it
    # (before triton_attn binds the name via ``from ops import unified_attention``).
    if OPS_TARGET in sys.modules:
        _patch_ops_module_safe(sys.modules[OPS_TARGET])
    else:
        sys.meta_path.insert(0, _ChainFinder(OPS_TARGET))
    print(
        f"[splitkv-verify] armed (SPLITKV_VERIFY=1, max_q<={SPLITKV_VERIFY_MAX_Q})",
        flush=True,
    )
