"""byteexact-splitkv: pin vLLM's 3D split-KV to a FIXED tiles-per-segment so the
M=8 spec-verify attention reduction is byte-identical to the M=1 AR decode of the
same token -- WITHOUT installing the global batch-invariant matmul tax.

Background (lawine PR #496, run ``42qroec1``; packaged here for PR #500)
-----------------------------------------------------------------------
The deployed fast decode path is vLLM's 3D split-KV (FlashDecoding): the KV axis
is cut into ``num_par_softmax_segments`` parallel segments that each reduce a
contiguous span of keys, then ``reduce_segments`` merges them with an online
softmax. The per-call span width is **adaptive**::

    tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)

Because it divides by ``seq_len``, the segment boundaries move as the context
grows. The M=8 spec-verify (seq_len = L) and the M=1 AR decode of that same token
(seq_len = L) can therefore land their segment cuts at different absolute key
positions whenever L sweeps across a tile multiple -- so the float reduction order
differs and the bf16 result flips (lawine #496 microbench: 6/8 residual byte-flips
on straddle positions for the adaptive path).

Pinning ``tiles_per_segment = T`` (a fixed constant) makes every segment boundary
fall at a FIXED ABSOLUTE key position -- segment ``s`` always covers keys
``[s*T*TILE_SIZE, (s+1)*T*TILE_SIZE)`` regardless of seq_len. The reduction order
at any given absolute key position is then identical for M=8 verify and M=1 AR ->
byte-exact, M-invariant (Thinking-Machines "fix the split SIZE, not the split
COUNT"). #496 measured this fixed scheme realizes **399.75 TPS byte-exact** (+42
over the surgical-357 2D rung) at PPL 2.3767 with 0/8 kernel flips, and it keeps
the fast Marlin MLP/QKV/lm_head path (no global matmul tax, unlike the full
``VLLM_BATCH_INVARIANT=1`` flag).

Mechanism (stock-wheel monkeypatch -- NO installed-wheel file edit)
------------------------------------------------------------------
Two import-time transforms on the STOCK wheel, mirroring stark's
``surgical_attn_patch.py`` and the parent ``splitkv_verify_patch.py`` discipline
(meta-path finder with a ``_busy`` guard, ``install()`` in-process path, fail-open):

  1. ``vllm.v1.attention.ops.triton_unified_attention`` -- both ``@triton.jit``
     kernels (``kernel_unified_attention`` and ``reduce_segments``) have their
     single adaptive ``tiles_per_segment = cdiv_fn(seq_len, ... * TILE_SIZE)``
     line rewritten to the literal ``tiles_per_segment = T`` (T =
     ``BYTEEXACT_FIXED_TPS``). We re-exec the edited source in the stock module's
     ``__dict__`` so the ``@triton.jit`` decorator and every helper global
     (``cdiv_fn``, ``compute_tile_loop_bounds``, ``tl`` ...) re-resolve, and the
     kernel recompiles with T as a compile-time constant -- byte-identical to
     #496's ``FIXED_TILES_PER_SEGMENT`` constexpr build. ``tiles_per_segment`` is
     computed once per kernel and flows into ``compute_tile_loop_bounds`` (the
     per-segment key range) and ``act_num_segments``; the single-line rewrite
     therefore fixes the whole segmentation.
  2. ``vllm.v1.attention.backends.triton_attn`` -- the module global
     ``NUM_PAR_SOFTMAX_SEGMENTS`` is raised from 16 to S (``BYTEEXACT_NUM_SEGMENTS``)
     so the fixed CHUNK = T*TILE_SIZE keys still covers the full context
     (coverage = S*T*TILE_SIZE keys; T=4,S=64,TILE_SIZE=16 -> 4096 = max_model_len)
     while keeping split parallelism/occupancy. The backend reads this global at
     attention-metadata ``__init__`` (after this patch runs), and sizes the
     per-segment scratch buffers from it.

Composition with the parent stack
---------------------------------
``splitkv_verify_patch`` routes the M=8 verify batch onto this same 3D split-KV
path (it forces ``max_seqlen_q=1`` so the 1<M<=64 verify range selects 3D); this
patch makes that 3D path byte-exact. The two finders compose (each wraps the
other's loader; the real module code runs first, then each patch's exec hook).
This patch does NOT force the 2D path (that is surgical-357) and does NOT set
``VLLM_BATCH_INVARIANT``/``is_batch_invariant``, so ``splitkv_verify``'s
``_batch_invariant()`` gate stays False (verify keeps redirecting to 3D) and
``init_batch_invariance()`` never installs the matmul tax.

Gating
------
Off by default. Active only when ``BYTEEXACT_FIXED_TPS`` is a positive int (the
packaged manifest sets ``BYTEEXACT_FIXED_TPS=4`` and ``BYTEEXACT_NUM_SEGMENTS=64``).
When unset/0 this module is not even imported (see ``sitecustomize.py``), so the
served compute path is byte-identical to the parent ``fa2sw_precache_kenyan`` stack.

Fail-open: any error leaves the stock dispatch untouched (the served stack then
runs the fast adaptive 3D split-KV -- fast but NOT byte-exact, so the operative
identity gate would catch it; it never silently degrades correctness).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import inspect
import os
import sys
import tempfile
import textwrap
from typing import Any

OPS_TARGET = "vllm.v1.attention.ops.triton_unified_attention"
BACKEND_TARGET = "vllm.v1.attention.backends.triton_attn"

# Re-jitted kernel source files are written here and KEPT for the process lifetime:
# triton's ``@jit`` reads the function source via ``inspect`` at compile time (first
# kernel call), so the backing file must remain readable. (Module-level list also
# prevents premature GC / lets a teardown remove them if ever desired.)
_TMP_FILES: list[str] = []

# TILE_SIZE_DECODE for the Gemma-4-E4B global full-attention layers (head_dim 512,
# no sliding window -> not the gemma3 windowed path). Used only for the log line's
# CHUNK / coverage arithmetic; the kernel uses its own compile-time TILE_SIZE.
_TILE_SIZE = 16

try:
    FIXED_TPS = int(os.environ.get("BYTEEXACT_FIXED_TPS", "0") or "0")
except ValueError:
    FIXED_TPS = 0
try:
    NUM_SEGMENTS = int(os.environ.get("BYTEEXACT_NUM_SEGMENTS", "64") or "64")
except ValueError:
    NUM_SEGMENTS = 64

ENABLED = FIXED_TPS > 0

# The two single-line adaptive computations to pin (one per @triton.jit kernel).
# Each appears exactly once in its function body; the replacement keeps the line's
# leading indentation because the anchor has none of its own.
_KERNEL_EDITS = {
    "kernel_unified_attention": (
        "tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)",
        "tiles_per_segment = {T}",
    ),
    "reduce_segments": (
        "tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)",
        "tiles_per_segment = {T}",
    ),
}

_state = {"ops_done": False, "backend_done": False, "logged": False}


def _log_armed() -> None:
    if _state["logged"]:
        return
    coverage = FIXED_TPS * NUM_SEGMENTS * _TILE_SIZE
    print(
        f"[byteexact] fixed split-KV armed: tiles_per_segment={FIXED_TPS} "
        f"num_par_softmax_segments={NUM_SEGMENTS} "
        f"(coverage {coverage} keys)",
        flush=True,
    )
    _state["logged"] = True


def _rebuild_jit_kernel(module: Any, fn_name: str, anchor: str, repl: str) -> None:
    """Source-transform a single ``@triton.jit`` kernel: replace its one adaptive
    ``tiles_per_segment = ...`` line with the fixed literal and re-exec the
    function in the stock module ``__dict__`` (re-applies ``@triton.jit`` and
    re-resolves every helper global). Raises if the anchor is missing so a stale
    wheel fails loud rather than silently shipping the adaptive (non-exact) path."""
    jitfn = getattr(module, fn_name, None)
    if jitfn is None:
        raise RuntimeError(f"{fn_name} not found in {OPS_TARGET}")
    src_fn = getattr(jitfn, "fn", jitfn)  # triton JITFunction wraps the real fn
    src = textwrap.dedent(inspect.getsource(src_fn))
    def_idx = src.index(f"def {fn_name}")
    body = src[def_idx:]  # drop any existing decorator lines; re-add exactly one
    if anchor not in body:
        raise RuntimeError(f"anchor not found in {fn_name}: {anchor!r}")
    new_body = body.replace(anchor, repl, 1)
    new_src = "@triton.jit\n" + new_body
    # triton's @jit rejects functions whose source is not in a real .py file (it
    # re-reads the source via inspect at compile time), so back the exec with a
    # persistent temp file and compile against THAT path; keep it for process life.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_byteexact_{fn_name}.py", delete=False
    )
    try:
        tmp.write(new_src)
    finally:
        tmp.close()
    _TMP_FILES.append(tmp.name)
    code = compile(new_src, tmp.name, "exec")
    exec(code, module.__dict__)  # rebinds module.<fn_name> to the fixed kernel


def _patch_ops_module(module: Any) -> None:
    if _state["ops_done"]:
        return
    if "triton" not in module.__dict__:
        raise RuntimeError("triton not in ops module globals; cannot re-jit")
    for fn_name, (anchor, repl_tmpl) in _KERNEL_EDITS.items():
        _rebuild_jit_kernel(module, fn_name, anchor, repl_tmpl.format(T=FIXED_TPS))
    module.__dict__["_byteexact_fixed_tps"] = FIXED_TPS  # marker / introspection
    _state["ops_done"] = True
    _log_armed()
    print(
        f"[byteexact] re-jitted {', '.join(_KERNEL_EDITS)} with "
        f"tiles_per_segment={FIXED_TPS} (was adaptive cdiv)",
        flush=True,
    )


def _patch_backend_module(module: Any) -> None:
    if _state["backend_done"]:
        return
    prior = getattr(module, "NUM_PAR_SOFTMAX_SEGMENTS", None)
    module.NUM_PAR_SOFTMAX_SEGMENTS = NUM_SEGMENTS
    _state["backend_done"] = True
    print(
        f"[byteexact] set triton_attn.NUM_PAR_SOFTMAX_SEGMENTS={NUM_SEGMENTS} "
        f"(was {prior!r})",
        flush=True,
    )


def _patch_ops_safe(module: Any) -> None:
    try:
        _patch_ops_module(module)
    except Exception as exc:  # noqa: BLE001 - fail-open
        print(f"[byteexact] ops patch error, baseline kept: {exc!r}", flush=True)


def _patch_backend_safe(module: Any) -> None:
    try:
        _patch_backend_module(module)
    except Exception as exc:  # noqa: BLE001 - fail-open
        print(f"[byteexact] backend patch error, baseline kept: {exc!r}", flush=True)


def install() -> bool:
    """In-process install (profiling harness / tests). Returns True if active."""
    if not ENABLED:
        return False
    try:
        import vllm.v1.attention.ops.triton_unified_attention as ua
    except Exception as exc:  # noqa: BLE001
        print(f"[byteexact] vLLM ops module unavailable: {exc!r}", flush=True)
        return False
    _patch_ops_safe(ua)
    try:
        import vllm.v1.attention.backends.triton_attn as ta
    except Exception as exc:  # noqa: BLE001
        print(f"[byteexact] vLLM backend module unavailable: {exc!r}", flush=True)
        return _state["ops_done"]
    _patch_backend_safe(ta)
    return _state["ops_done"] and _state["backend_done"]


# --- import-time meta-path finder (served subprocess via sitecustomize) -------
# Same _busy-guarded chain pattern as splitkv_verify_patch / surgical_attn_patch:
# when several finders are registered for the same target they compose (each wraps
# the other's loader), so the module's real code runs first, then each patch's exec
# hook fires. One finder instance per target so the _busy guards never overlap.
class _ChainLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, hook: Any) -> None:
        self._inner = inner
        self._hook = hook

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._hook(module)


class _ChainFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, hook: Any) -> None:
        self._target = target
        self._hook = hook
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
        spec.loader = _ChainLoader(spec.loader, self._hook)
        return spec


def _arm() -> None:
    # ops kernels
    if OPS_TARGET in sys.modules:
        _patch_ops_safe(sys.modules[OPS_TARGET])
    else:
        sys.meta_path.insert(0, _ChainFinder(OPS_TARGET, _patch_ops_safe))
    # backend segment-count global
    if BACKEND_TARGET in sys.modules:
        _patch_backend_safe(sys.modules[BACKEND_TARGET])
    else:
        sys.meta_path.insert(0, _ChainFinder(BACKEND_TARGET, _patch_backend_safe))


if ENABLED:
    _arm()
    print(
        f"[byteexact] armed (BYTEEXACT_FIXED_TPS={FIXED_TPS}, "
        f"BYTEEXACT_NUM_SEGMENTS={NUM_SEGMENTS}): fixed-order 3D split-KV "
        f"-> M-invariant byte-exact attention; matmul tax NOT installed",
        flush=True,
    )
