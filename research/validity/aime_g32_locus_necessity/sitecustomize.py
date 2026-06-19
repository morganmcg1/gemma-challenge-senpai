#!/usr/bin/env python
"""PR #713 — spawn-safe in-memory int4 fake-quant injection (engine-core child).

Loaded by the vLLM engine-core CHILD process via PYTHONPATH at interpreter
startup. The V1 OpenAI *async* server always runs the engine core in a separate
process (AsyncMPClient) and initializes CUDA in the api_server parent, so that
child MUST spawn — a fork hits "Cannot re-initialize CUDA in forked subprocess".
A spawned child is a fresh interpreter, so any runtime monkeypatch set in the
serve.py parent is LOST. Instead we register a MetaPathFinder HERE (proven peer
pattern: submissions/fa2sw_strict_m1ar_int4/sitecustomize.py) that wraps
GPUModelRunner.load_model AFTER vLLM imports it in the child:

    original load_model  -> bf16 weights on GPU
    then fake-quant the selected text-decoder Linear weights in place
    (dequant->requant int4 group: g32 on FQ_G32_LAYERS, g128 elsewhere).

Gated on FQ_APPLY=1 (set by serve.py) so a plain vLLM launch in this dir is
untouched. The "[fq-serve] APPLIED ..." line is printed from the child pid and
is the smoke-test proof that the patch fired inside the engine core.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

RUNNER_TARGET = "vllm.v1.worker.gpu_model_runner"
_STATE = {"applied": False}


def _truthy(v: str) -> bool:
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def _apply_fakequant_runner_patch(module) -> None:
    runner_cls = module.GPUModelRunner
    original_load_model = runner_cls.load_model

    def load_model(self, *a, **k):
        original_load_model(self, *a, **k)
        if _STATE["applied"]:
            return
        # bf16 base control (kanna #699 engine-health gate): serve the UNMODIFIED bf16
        # master with NO fake-quant, so this engine's greedy AIME can be reconciled
        # against the bf16 endpoint (coherent ~0.4667, not a repetition-to-cap collapse).
        # No-op for every fake-quant cell (FQ_BASE unset/0) — the campaign is untouched.
        if _truthy(os.environ.get("FQ_BASE", "0")):
            print(f"[fq-serve] FQ_BASE=1 -> NO fake-quant applied (bf16 base control) "
                  f"pid={os.getpid()}", flush=True)
            _STATE["applied"] = True
            return
        import fakequant as fq  # serve dir is on PYTHONPATH in this child

        g32_spec = os.environ.get("FQ_G32_LAYERS", "") or ""
        g32 = int(os.environ.get("FQ_G32_GROUP", "32") or "32")
        g128 = int(os.environ.get("FQ_G128_GROUP", "128") or "128")
        quant_head = _truthy(os.environ.get("FQ_QUANT_HEAD", "1"))
        g32_layers = fq.parse_layers(g32_spec)
        raw = self.get_model()  # unwraps CUDAGraphWrapper -> raw nn.Module
        # One-time census so the child log proves which modules were targeted.
        n_quant = n_head = 0
        for name, mod in raw.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not hasattr(w, "shape") or w.dim() != 2:
                continue
            if fq.is_quant_target(name):
                n_quant += 1
            elif quant_head and fq.is_head_target(name):
                n_head += 1
        print(
            f"[fq-serve] FQ_G32_LAYERS={g32_spec!r} -> {sorted(g32_layers)} "
            f"g32={g32} g128={g128} quant_head={quant_head} "
            f"census(body={n_quant} head={n_head}) pid={os.getpid()}",
            flush=True,
        )
        rep = fq.apply_fake_quant(
            raw, g32_layers, g32=g32, g128=g128, quant_head=quant_head,
            log=lambda m: print(m, flush=True),
        )
        print(f"[fq-serve] APPLIED {rep} pid={os.getpid()}", flush=True)
        _STATE["applied"] = True

    runner_cls.load_model = load_model
    print(f"[fq-serve] patched GPUModelRunner.load_model in pid {os.getpid()}", flush=True)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner, patch_fn) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _TargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch_fn) -> None:
        self._target = target
        self._patch_fn = patch_fn
        self._busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._patch_fn)
        return spec


if _truthy(os.environ.get("FQ_APPLY", "0")):
    sys.meta_path.insert(0, _TargetFinder(RUNNER_TARGET, _apply_fakequant_runner_patch))
    print(
        f"[fq-serve] registered fakequant MetaPathFinder for {RUNNER_TARGET} "
        f"(pid {os.getpid()})",
        flush=True,
    )
