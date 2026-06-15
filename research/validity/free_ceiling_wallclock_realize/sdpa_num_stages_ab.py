"""Env-gated ``num_stages`` injector for the deployed verify SDPA (PR #298, stark).

LOCAL wall-clock A/B toggle ONLY. When ``SDPA_NUM_STAGES_AB`` is set to an int N
(e.g. ``2``), every launch of
``vllm.v1.attention.ops.triton_unified_attention.kernel_unified_attention`` is
forced to compile/run with ``num_stages=N`` instead of the triton-default
``num_stages=3`` (the deployed 481.53 baseline, confirmed bare-@triton.jit by
wirbel #270/#279 self-test lynchpin). This is the SDPA ``num_stages`` 3->2 lever:
a bit-identical cp.async pipeline-depth change (NOT MMA/K-reduction order ->
maxdiff 0.0, greedy-safe), the ONLY incremental free lossless lever in wirbel
#285's envelope and denken #291's kernel-event floor.

Mechanism: reuse the submission's own deferred-patch idiom (``_TargetFinder`` /
``_PatchingLoader`` meta-path finder in ``sitecustomize.py``) so the patch lands
the moment vLLM imports the kernel module -- i.e. BEFORE the ONEGRAPH/LOOPGRAPH
capture, so the ``num_stages=N`` cubin is the one baked into the captured graph.
The module global ``kernel_unified_attention`` is wrapped (not the JITFunction
class), so ONLY the unified attention kernel is retuned -- the int4 GEMMs and
every other triton kernel are untouched (their retune is greedy-UNSAFE and out of
scope; denken #291).

This file is loaded ONLY via a temporary env-gated hook appended to
``sitecustomize.py`` during the PR #298 wall A/B by
``free_ceiling_wallclock_realize.py``; the hook is reverted and NEVER submitted.
With ``SDPA_NUM_STAGES_AB`` unset the deployed stack is byte-identical (this file
is not even imported). Forced launches are counted and logged to stderr so the
A/B can PROVE the candidate arm actually ran ``num_stages=N`` (guards against a
silent no-op that would fake an ``over_credits`` verdict).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "vllm.v1.attention.ops.triton_unified_attention"
_RAW = os.environ.get("SDPA_NUM_STAGES_AB", "").strip()

# shared mutable state (this module is exec'd into sitecustomize's namespace; the
# closures below capture this dict so the forced-launch counter survives).
_AB_STATE = {"forced": 0, "stages": None, "wrapped": False, "target_seen": False}


def _log(msg: str) -> None:
    print(f"[sdpa-ab] {msg}", file=sys.stderr, flush=True)


def _make_wrapper(inner_kernel, num_stages):
    """Wrap the module-global ``kernel_unified_attention`` so ``kern[grid](...)``
    injects ``num_stages`` when the caller did not pass one (the deployed
    ``unified_attention`` never passes it -> triton default 3)."""

    class _StagesForcingKernel:
        def __getitem__(self, grid):
            launcher = inner_kernel[grid]  # functools.partial(inner_kernel.run, grid=grid)

            def _call(*args, **kwargs):
                if kwargs.get("num_stages") is None:
                    kwargs["num_stages"] = num_stages
                    _AB_STATE["forced"] += 1
                    c = _AB_STATE["forced"]
                    if c in (1, 2, 5, 20, 100) or c % 500 == 0:
                        _log(f"forced num_stages={num_stages} on kernel_unified_attention (count={c})")
                return launcher(*args, **kwargs)

            return _call

        def __getattr__(self, name):  # forward everything else to the real kernel
            return getattr(inner_kernel, name)

    return _StagesForcingKernel()


def _patch_module(module, num_stages):
    if _AB_STATE["wrapped"]:
        return
    kern = getattr(module, "kernel_unified_attention", None)
    if kern is None:
        _log("WARN: target module has no kernel_unified_attention; not patched")
        return
    module.kernel_unified_attention = _make_wrapper(kern, num_stages)
    _AB_STATE["wrapped"] = True
    _AB_STATE["stages"] = num_stages
    _AB_STATE["target_seen"] = True
    _log(f"PATCHED {_TARGET}.kernel_unified_attention -> num_stages={num_stages}")


def _install(num_stages):
    _AB_STATE["stages"] = num_stages
    # belt+suspenders: if vLLM already imported the kernel module, patch in place.
    if _TARGET in sys.modules:
        _patch_module(sys.modules[_TARGET], num_stages)
        return

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            _patch_module(module, num_stages)

    class _TargetFinder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self._busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET or self._busy:
                return None
            self._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PatchingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _TargetFinder())
    _log(f"meta-path finder installed for {_TARGET} (num_stages={num_stages})")


if _RAW and _RAW.isdigit() and int(_RAW) != 3:
    _install(int(_RAW))
elif _RAW:
    _log(f"no-op: SDPA_NUM_STAGES_AB={_RAW!r} (== deployed default 3 or invalid)")
