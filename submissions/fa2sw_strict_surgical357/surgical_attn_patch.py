"""surgical-attn: force the byte-exact 2D order-preserving attention path on the
full-attention reductions WITHOUT installing the global batch-invariant matmul tax.

Background (lawine PR #488, run ``ko01dcyy``)
---------------------------------------------
On this A10G (sm_86 -> SM80 family) the shipped strict flag ``VLLM_BATCH_INVARIANT=1``
collapses the deployed ~481 TPS to ~222 because it does TWO *independent* things:

  1. ``init_batch_invariance()`` installs ``matmul_persistent`` Triton overrides on
     every aten::mm/addmm/linear (~48% tax -- the majority of the 481->222 collapse), AND
  2. forces the full_attention reductions onto vLLM's order-preserving 2D single-segment
     sequential-KV path (``use_3d=False`` in ``triton_unified_attention.unified_attention``)
     -- the *only* part that buys byte-exact greedy identity (the M=8 spec-verify reduces
     in the same order as its own M=1 AR decode).

Item 2 is the load-bearing identity lever; item 1 is identity-unnecessary (lawine #488
measured only 9/128 residual flips between the surgical and full-flag arms, all bf16-ULP
near-ties per merged PR #461 -- 0 semantic). This patch reaches item 2 *without* item 1.

Mechanism
---------
``triton_unified_attention`` computes, per call,
``use_3d = not (... or is_batch_invariant)`` where ``is_batch_invariant`` is a module
global initialised to ``envs.VLLM_BATCH_INVARIANT`` (False here -- we never set that env).
Setting the module global ``is_batch_invariant = True`` makes ``use_3d`` False for every
attention call (the 2D order-preserving path) -- byte-identical effect to lawine's local
serve-venv one-line edit, applied at runtime instead of editing the installed wheel.

It also composes with ``splitkv_verify_patch``: that patch's ``_batch_invariant()`` reads
the same ``getattr(triton_unified_attention, "is_batch_invariant", False)`` flag, so once
this patch sets it True the spec-verify 3D split-KV redirect short-circuits to fail-open --
i.e. the M=8 verify takes the 2D order-preserving path too, exactly as in the full flag.
Because ``envs.VLLM_BATCH_INVARIANT`` stays unset, ``init_batch_invariance()`` never runs
and the matmul tax is never installed: MLP / QKV / lm_head keep the fast Marlin path.

Gating
------
Off by default. Active only when ``SURGICAL_ATTN_USE_3D_OFF=1`` (the strict-draw manifest
sets it). When unset, this module is not even imported (see sitecustomize.py), so the served
compute path is byte-identical to the parent ``fa2sw_precache_kenyan`` deployed stack.

Fail-open: any error leaves the stock dispatch untouched.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from typing import Any

OPS_TARGET = "vllm.v1.attention.ops.triton_unified_attention"

SURGICAL_ATTN_USE_3D_OFF = os.environ.get("SURGICAL_ATTN_USE_3D_OFF", "0") == "1"


def _force_batch_invariant(module: Any) -> None:
    """Set the module-level ``is_batch_invariant`` flag True so every attention call
    takes the 2D order-preserving (``use_3d=False``) path. Idempotent."""
    prior = getattr(module, "is_batch_invariant", None)
    if prior is True:
        return
    module.is_batch_invariant = True
    print(
        "[surgical-attn] forced triton_unified_attention.is_batch_invariant=True "
        f"(was {prior!r}) -> 2D order-preserving attention, global matmul tax OFF",
        flush=True,
    )


def _force_safe(module: Any) -> None:
    try:
        _force_batch_invariant(module)
    except Exception as exc:  # noqa: BLE001 - fail-open
        print(f"[surgical-attn] patch error, baseline kept: {exc!r}", flush=True)


def install() -> bool:
    """In-process install (tests / harness). Returns True if the flag is now set."""
    if not SURGICAL_ATTN_USE_3D_OFF:
        return False
    try:
        import vllm.v1.attention.ops.triton_unified_attention as ua
    except Exception as exc:  # noqa: BLE001
        print(f"[surgical-attn] vLLM ops module unavailable: {exc!r}", flush=True)
        return False
    _force_safe(ua)
    return bool(getattr(ua, "is_batch_invariant", False))


# --- import-time meta-path finder (served subprocess via sitecustomize) -------
# Same _busy-guarded chain pattern as splitkv_verify_patch: when both finders are
# registered for OPS_TARGET they compose (each wraps the other's loader), so the
# module's real code runs first, then each patch's exec hook fires.
class _ChainLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _force_safe(module)


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


if SURGICAL_ATTN_USE_3D_OFF:
    # If the ops module is already imported, set the flag in place; otherwise
    # register a finder so it is set the moment vLLM imports it.
    if OPS_TARGET in sys.modules:
        _force_safe(sys.modules[OPS_TARGET])
    else:
        sys.meta_path.insert(0, _ChainFinder(OPS_TARGET))
    print(
        "[surgical-attn] armed (SURGICAL_ATTN_USE_3D_OFF=1): full-attention reductions "
        "-> 2D order-preserving sequential-KV; matmul tax NOT installed",
        flush=True,
    )
