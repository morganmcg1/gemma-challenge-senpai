#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #755 lawine -- FORCE num_splits=1 on the served attention (instruction 2).

Realizes land #743's offline ``num_splits=1`` lever in the DEPLOYABLE served path.
The split gate (triton_unified_attention.py:923-931) collapses to the 2D one-shot
reduction (num_segments=1) whenever the module global ``is_batch_invariant`` is
True. ``enable_batch_invariant_mode()`` (matmul/softmax only) NEVER sets that
global -- it is frozen once at import from ``envs.VLLM_BATCH_INVARIANT``. So the
clean, surgical force is to pin that global True in EVERY server process the moment
``triton_unified_attention`` loads.

Two layers of belt-and-suspenders so the force is robust even if some path imported
the kernel before the env was visible:
  (1) set ``triton_unified_attention.is_batch_invariant = True`` on load; AND
  (2) wrap ``triton_attn.unified_attention`` so the kernel module's global is
      re-pinned True immediately before every real call (covers a stale freeze).

EMPIRICAL NOTE (PR #755 served probe, run 2026-06-19): under the publishable-K4
config ``VLLM_BATCH_INVARIANT=1`` the served worker ALREADY reports
``is_batch_invariant=True`` with BOTH the M=1 decode and the M=5 verify at
``use3d=0, nseg=1``. So under BI=1 this force is a confirmed NO-OP (it pins a global
that is already True) -- which is exactly why it cannot move #752's 24/128: the
divergence is not the attention split. The module is kept (a) as the literal
realization of the instruction-2 lever, and (b) because it DOES bite under a
non-BI serve, where the M=1 decode would otherwise take the 3D nseg>1 split.

NO other numerics change. LOCAL A10G only.
"""
from __future__ import annotations

import os
import sys

_INSTALLED = {"v": False}


def _log(msg: str) -> None:
    try:
        print(f"[pr755-force pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _pin_kernel_global(kernel_mod) -> bool:
    before = bool(getattr(kernel_mod, "is_batch_invariant", False))
    kernel_mod.is_batch_invariant = True
    _log(f"pinned triton_unified_attention.is_batch_invariant: {before} -> True")
    return before


def _make_wrapper(real_fn, kernel_mod):
    def wrapper(*args, **kwargs):
        # re-pin in case anything reset it; cheap attribute write
        if not getattr(kernel_mod, "is_batch_invariant", False):
            kernel_mod.is_batch_invariant = True
        return real_fn(*args, **kwargs)
    wrapper.__name__ = getattr(real_fn, "__name__", "unified_attention")
    wrapper._numsplits_force = True
    return wrapper


def _patch_module(triton_attn_mod) -> None:
    import vllm.v1.attention.ops.triton_unified_attention as kernel_mod
    _pin_kernel_global(kernel_mod)
    real = getattr(triton_attn_mod, "unified_attention", None)
    if real is not None and not getattr(real, "_numsplits_force", False):
        triton_attn_mod.unified_attention = _make_wrapper(real, kernel_mod)


def install() -> None:
    if _INSTALLED["v"]:
        return
    _INSTALLED["v"] = True
    _log("install()")

    # If the kernel module is already imported, pin immediately.
    kmod = "vllm.v1.attention.ops.triton_unified_attention"
    if kmod in sys.modules:
        _pin_kernel_global(sys.modules[kmod])
    target = "vllm.v1.attention.backends.triton_attn"
    if target in sys.modules:
        _patch_module(sys.modules[target])
        return

    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _Finder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.v1.attention.backends.triton_attn":
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                try:
                    _patch_module(module)
                except Exception:
                    _log("patch failed")

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
