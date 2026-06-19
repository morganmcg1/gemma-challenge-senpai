"""Serve-boot attention pin + pin-active recorder for the strict-#319 served-head cert (PR #690).

This module is imported by the *copied* submission's ``sitecustomize.py`` (one
appended ``import _stark_pin`` line; the original submission is never touched). It
runs in every process of the vLLM server tree (api_server, EngineCore, worker)
because the copy dir is on ``PYTHONPATH`` exactly like the submission's own
sitecustomize boot-patch (kanna #177 precedent). ENABLE_USER_SITE is False in the
serve venv, so usercustomize is unavailable — a sitecustomize-chained import is the
only universal hook.

It does two things, both gated by env so it is a no-op outside this cert:

1. ``STARK_PIN_MODE=fixed2d`` -> force the M=1 AR decode forward onto the 2D
   single-pass kernel (matching the M>=2 verify forward) by overriding
   ``seq_threshold_3D=0`` at the ``unified_attention`` call site, the authoritative
   point where vLLM recomputes ``use_3d`` from that argument (triton_unified_attention
   lines 923-932): with threshold 0, ``num_seqs(>=1) > 0`` makes ``use_3d=False``.
   We ALSO set ``triton_attn.MIN_LAUNCH_GRID_SIZE_2D = 0`` at module import (land
   #684's lever), but that alone does NOT reach the deployed decode: the metadata
   builder derives ``seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv`` and
   then, with decode CUDA graphs enabled (no ``--enforce-eager``), rounds it to the
   nearest capture size (>=1) -- so the num_seqs=1 decode never reaches threshold 0
   and stayed on 3D split-KV (PR #690 first cut: observed threshold 7, break 53.78%
   ~= the un-pinned 51.70%). The call-site override is the robust fix. NOT a kernel
   rebuild. The sole CUDA call site (triton_attn.forward) passes every argument by
   keyword, so the override is a single ``kwargs`` write.

2. Pin-active PROOF (the lawine #681 load-bearing requirement). When
   ``vllm.v1.attention.ops.triton_unified_attention`` imports we record its
   import-time ``is_batch_invariant`` (= ``envs.VLLM_BATCH_INVARIANT`` frozen at
   import) and wrap ``unified_attention`` to record, for the first few distinct
   forward shapes, the EXACT 2D-vs-3D branch the served forward takes
   (num_seqs, max_seqlen_q, seq_threshold_3D, is_batch_invariant, use_3d). This is
   a direct measurement that the pin is live inside the serving forward, not just in
   the outer env. The wrapper calls straight through; it never changes numerics.

All proof is written as small JSON to ``STARK_PIN_PROOF_DIR`` (one file per pid).
Every hook is wrapped in try/except so a recorder bug can never break serving.
"""
from __future__ import annotations

import json
import os
import sys
import time

_PROOF_DIR = os.environ.get("STARK_PIN_PROOF_DIR")
_MODE = os.environ.get("STARK_PIN_MODE", "none")
_TRITON_ATTN = "vllm.v1.attention.backends.triton_attn"
_UNIFIED = "vllm.v1.attention.ops.triton_unified_attention"

# Bound the per-process branch recordings so we never spam disk: record at most one
# row per distinct (max_seqlen_q, num_seqs) shape, capped.
_seen_shapes: set = set()
_MAX_SHAPE_ROWS = 24


def _proof_write(name: str, payload: dict) -> None:
    if not _PROOF_DIR:
        return
    try:
        os.makedirs(_PROOF_DIR, exist_ok=True)
        path = os.path.join(_PROOF_DIR, f"{name}_{os.getpid()}.json")
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _proof_append(name: str, payload: dict) -> None:
    if not _PROOF_DIR:
        return
    try:
        os.makedirs(_PROOF_DIR, exist_ok=True)
        path = os.path.join(_PROOF_DIR, f"{name}_{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _patch_triton_attn(module) -> None:
    """Pin MIN_LAUNCH_GRID_SIZE_2D=0 for fixed2d; always record the live constant."""
    before = getattr(module, "MIN_LAUNCH_GRID_SIZE_2D", None)
    after = before
    try:
        if _MODE == "fixed2d":
            module.MIN_LAUNCH_GRID_SIZE_2D = 0
            after = module.MIN_LAUNCH_GRID_SIZE_2D
    except Exception:
        pass
    _proof_write(
        "pin_triton_attn",
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "module": _TRITON_ATTN,
            "stark_pin_mode": _MODE,
            "min_launch_before": before,
            "min_launch_after": after,
            "serve_env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        },
    )


def _record_branch(module, kwargs: dict, forced: bool) -> None:
    """Record the EFFECTIVE (post-override) 2D/3D branch for one distinct shape.

    Reads straight from ``kwargs`` because the sole CUDA call site
    (triton_attn.forward, vllm 0.22.0 line 634) passes every argument by keyword.
    ``use_3d`` is an exact replica of the selector in
    triton_unified_attention.unified_attention (lines 923-932): True -> 3D split-KV,
    False -> 2D single-pass. When ``forced`` is True the ``seq_threshold_3D`` read
    here is already 0, so a num_seqs>=1 decode records as 2D_single_pass."""
    num_seqs = len(kwargs["seqused_k"])
    max_seqlen_q = int(kwargs["max_seqlen_q"])
    shape = (max_seqlen_q, num_seqs)
    if shape in _seen_shapes:
        return
    st3 = kwargs.get("seq_threshold_3D")
    nps = kwargs.get("num_par_softmax_segments")
    so = kwargs.get("softmax_segm_output")
    sm = kwargs.get("softmax_segm_max")
    se = kwargs.get("softmax_segm_expsum")
    ibi = bool(getattr(module, "is_batch_invariant", False))
    use_3d = not (
        st3 is None
        or nps is None
        or so is None
        or sm is None
        or se is None
        or max_seqlen_q > 1
        or num_seqs > st3
        or ibi
    )
    _seen_shapes.add(shape)
    _proof_append(
        "pin_branch",
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "stark_pin_mode": _MODE,
            "num_seqs": num_seqs,
            "max_seqlen_q": max_seqlen_q,
            "seq_threshold_3D": st3,
            "seq_threshold_3D_forced_to_0": bool(forced),
            "is_batch_invariant": ibi,
            "use_3d_split_kv": bool(use_3d),
            "kernel": "3D_split_kv" if use_3d else "2D_single_pass",
        },
    )


def _wrap_unified_attention(module) -> None:
    """Record import-time is_batch_invariant and wrap unified_attention to (a) in
    fixed2d mode force the M=1 decode onto 2D at the call site, and (b) log the
    actual 2D/3D branch the served forward takes (direct pin-active evidence)."""
    is_bi = bool(getattr(module, "is_batch_invariant", False))
    _proof_write(
        "pin_unified_import",
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "module": _UNIFIED,
            "stark_pin_mode": _MODE,
            "is_batch_invariant_at_import": is_bi,
            "serve_env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        },
    )

    orig = getattr(module, "unified_attention", None)
    if orig is None or getattr(orig, "_stark_wrapped", False):
        return

    def wrapper(*args, **kwargs):
        # fixed2d pin (load-bearing): unified_attention recomputes use_3d from the
        # seq_threshold_3D ARGUMENT (vllm 0.22.0 lines 923-932), so forcing it to 0
        # makes the num_seqs>=1 decode take the 2D single-pass path (use_3d=False),
        # matching the M>=2 verify path -> spec==AR byte-exact. The sole call site
        # passes every arg by keyword, so this is a single kwargs write. land #684's
        # MIN_LAUNCH_GRID_SIZE_2D=0 alone can't reach threshold 0: the builder rounds
        # it to the nearest cudagraph capture size (>=1), so the M=1 decode stayed on
        # 3D split-KV (observed threshold 7, break ~unchanged).
        forced = False
        if _MODE == "fixed2d":
            st3 = kwargs.get("seq_threshold_3D")
            if isinstance(st3, int) and st3 != 0:
                kwargs["seq_threshold_3D"] = 0
                forced = True
        if len(_seen_shapes) < _MAX_SHAPE_ROWS:
            try:
                _record_branch(module, kwargs, forced)
            except Exception:
                pass
        return orig(*args, **kwargs)

    try:
        wrapper._stark_wrapped = True  # type: ignore[attr-defined]
        module.unified_attention = wrapper
    except Exception:
        pass


def _apply(module) -> None:
    try:
        name = getattr(module, "__name__", "")
        if name == _TRITON_ATTN:
            _patch_triton_attn(module)
        elif name == _UNIFIED:
            _wrap_unified_attention(module)
    except Exception:
        pass


def _install() -> None:
    # Patch any already-imported targets, then install a one-shot finder for the
    # rest (mirrors the submission sitecustomize's spec-decode finder pattern).
    for target in (_TRITON_ATTN, _UNIFIED):
        if target in sys.modules:
            _apply(sys.modules[target])

    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    targets = {_TRITON_ATTN, _UNIFIED}

    class _StarkPinFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets:
                return None
            # Temporarily remove ourselves so the nested find_spec resolves the
            # real loader without recursing.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            try:
                spec = find_spec(fullname)
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(mod, _orig=orig_exec):
                _orig(mod)
                _apply(mod)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _StarkPinFinder())


try:
    _install()
    _proof_write(
        "pin_installed",
        {"ts": time.time(), "pid": os.getpid(), "stark_pin_mode": _MODE},
    )
except Exception:
    pass
