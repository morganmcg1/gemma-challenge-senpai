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
# LOCAL probe only: one-shot dump of the exact kernel inputs for wide (tree)
# verify batches, to confirm the causal boundary (context_len = seqused_k -
# cur_batch_query_len) the kernel computes. Byte-identical deployed path when unset.
_KERNEL_DBG = os.environ.get("TREE_VERIFY_KERNEL_DBG") == "1"
_kernel_dbg_stats = {"logged": 0}

_stats = {"redirected": 0}

# --- PR #71 Component 2: tree-causal qq_bias mask DISPATCH (LOCAL probe) -------
# Env-gated. When TREE_QQ_BIAS_PROBE=1, thread an [M,M] fp32 tree-causal qq_bias
# into the redirected verify attention so star-attention is DISPATCHED for the
# tree rows (chiku-inu missing-half #1). The bias = where(ancestor-or-self,0,-inf),
# added to the score AFTER the kernel's causal mask
# (triton_unified_attention.py:525) and UPSTREAM of the IS_3D split (:548), so it
# reaches the 3D split-KV verify path. Indexed [query_pos, key_rel_pos] with
# qq_bias_stride_0 = stride(0) doubling as the column bound -> tensor MUST be a
# contiguous [M,M] fp32 (verified against the installed kernel).
#
#   parent=linear  -> the degenerate tree's mask IS the standard lower-triangular
#                     causal mask, so the bias is a mathematical NO-OP on the
#                     already-causal M=8 verify (ancestor -> +0; non-ancestor was
#                     already -inf -> +(-inf) still -inf). PPL/decode MUST be
#                     IDENTICAL; a transposed/mis-indexed tensor would mask real
#                     ancestors and break PPL -> the identity run validates the
#                     [query_pos, key_rel_pos] convention + no-corruption plumbing.
#   parent=canary  -> diagonal-only (each row attends prefix+self, NO ancestor
#                     draft rows). This is deliberately NOT a no-op: it must CHANGE
#                     PPL/decode -> empirical positive proof that qq_bias actually
#                     reaches and affects the kernel (DISPATCH confirmed).
#   parent=m16/m32 -> the real tree mask (for the later salvage probe, once the
#                     verify is widened to M=16/32 with node-order rows).
#
# Deployed path is byte-identical when TREE_QQ_BIAS_PROBE is unset (all logic is
# inside `if TREE_QQ_BIAS:` guards; kwargs untouched otherwise).
TREE_QQ_BIAS = os.environ.get("TREE_QQ_BIAS_PROBE") == "1"
# Only width-TREE_QQ_BIAS_M verify batches get the bias (8 = deployed K=7+1
# verify; isolates the canary's PPL delta to the verify, leaving prefill correct).
TREE_QQ_BIAS_M = int(os.environ.get("TREE_QQ_BIAS_M", "8") or "8")
TREE_QQ_BIAS_PARENT = os.environ.get("TREE_QQ_BIAS_PARENT", "linear")
_qq_stats = {"dispatched": 0}
_qq_cache: dict = {}


def _linear_parent(m: int) -> list:
    """Degenerate (chain) tree: node i's parent is i-1; ancestors-or-self={0..i}."""
    return [-1] + list(range(m - 1))


def _load_tree_spec_for_qq():
    """Load the validated CPU tree-spec reference (single source of truth for the
    tree-causal mask). Same module scripts/profiler/tree_spec.py the Component-1
    probe uses -> no structural drift."""
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    ts_path = repo_root / "scripts" / "profiler" / "tree_spec.py"
    spec = importlib.util.spec_from_file_location("_pr71_tree_spec_qq", ts_path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: tree_spec uses @dataclass (resolves annotations via
    # sys.modules[cls.__module__].__dict__).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_qq_bias(m: int, device: Any) -> Any:
    """Contiguous [M,M] fp32 tree-causal qq_bias = where(ancestor-or-self,0,-inf),
    in node order, on ``device``. Cached per (M, device, parent)."""
    import torch

    key = (int(m), str(device), TREE_QQ_BIAS_PARENT)
    cached = _qq_cache.get(key)
    if cached is not None:
        return cached
    if TREE_QQ_BIAS_PARENT == "canary":
        # diagonal-only: each row attends prefix(other=0) + itself only. NOT a
        # no-op -> proves qq_bias is applied if PPL/decode changes vs control.
        mask_t = torch.eye(m, dtype=torch.bool, device=device)
    else:
        ts = _load_tree_spec_for_qq()
        if TREE_QQ_BIAS_PARENT == "m16":
            parent = ts.PARENT_M16
        elif TREE_QQ_BIAS_PARENT == "m32":
            parent = ts.PARENT_M32
        else:
            parent = _linear_parent(m)
        tree = ts.TreeSpec(parent)
        if tree.num_nodes != m:
            raise ValueError(
                f"qq_bias parent M={tree.num_nodes} != verify M={m} "
                f"(parent={TREE_QQ_BIAS_PARENT})"
            )
        mask_rows = ts.tree_causal_mask(tree)  # list[list[bool]], ancestor-or-self
        mask_t = torch.tensor(mask_rows, dtype=torch.bool, device=device)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    neg = torch.full((), float("-inf"), dtype=torch.float32, device=device)
    qq = torch.where(mask_t, zero, neg).contiguous()
    _qq_cache[key] = qq
    return qq


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
                    if (
                        _KERNEL_DBG
                        and int(mq) >= 9
                        and _kernel_dbg_stats["logged"] < 4
                    ):
                        _kernel_dbg_stats["logged"] += 1

                        def _tl(x: Any) -> Any:
                            try:
                                return x.tolist() if hasattr(x, "tolist") else x
                            except Exception:
                                return x

                        cu = kwargs.get("cu_seqlens_q")
                        sk = kwargs.get("seqused_k")
                        bt = kwargs.get("block_table")
                        qb = kwargs.get("qq_bias")
                        print(
                            f"[kernel-dbg] mq={int(mq)} q.shape={tuple(kwargs['q'].shape)} "
                            f"cu_seqlens_q={_tl(cu)} seqused_k={_tl(sk)} "
                            f"max_seqlen_k={_tl(kwargs.get('max_seqlen_k'))} "
                            f"causal={kwargs.get('causal')} "
                            f"qq_bias={'None' if qb is None else tuple(qb.shape)} "
                            f"bt.shape={None if bt is None else tuple(bt.shape)}",
                            flush=True,
                        )
                    _stats["redirected"] += 1
                    if _stats["redirected"] <= _LOG_LIMIT:
                        print(
                            f"[splitkv-verify] verify batch M={mq} "
                            f"q_rows={int(kwargs['q'].shape[0])} -> 3D split-KV "
                            f"(n={_stats['redirected']})",
                            flush=True,
                        )
                    # Component 2: DISPATCH the tree-causal qq_bias for the
                    # width-TREE_QQ_BIAS_M verify batch (LOCAL probe, env-gated).
                    if (
                        TREE_QQ_BIAS
                        and int(mq) == TREE_QQ_BIAS_M
                        and kwargs.get("qq_bias") is None
                    ):
                        q = kwargs["q"]
                        q_rows = int(q.shape[0])
                        kwargs["qq_bias"] = _build_qq_bias(q_rows, q.device)
                        _qq_stats["dispatched"] += 1
                        if (
                            _qq_stats["dispatched"] <= _LOG_LIMIT
                            or _qq_stats["dispatched"] % 50 == 0
                        ):
                            print(
                                f"[tree-qq-bias] DISPATCHED [{q_rows}x{q_rows}] fp32 "
                                f"parent={TREE_QQ_BIAS_PARENT} -> 3D split-KV verify "
                                f"(n={_qq_stats['dispatched']})",
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
