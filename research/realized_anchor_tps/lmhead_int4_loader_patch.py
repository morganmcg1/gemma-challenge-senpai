"""PR #553 Stage 2 loader shim (LOCAL, analysis-only) -- meta-path variant.

The int4-lm_head experimental checkpoint symlinks the original 11.5GB blob as
shard-1, which still PHYSICALLY contains the bf16 ``lm_head.weight``. vLLM's
weight loader yields every physical key, so that bf16 tensor reaches
AutoWeightsLoader and crashes it ("no module or parameter named
language_model.lm_head.weight") because lm_head is now a quantized ParallelLMHead
whose only params are weight_packed / weight_scale / weight_shape.

This module drops EXACTLY that one bf16 tensor at the single chokepoint every
weight iterator funnels through (``DefaultModelLoader.get_all_weights``). vLLM's
default loader uses the MULTI-THREAD safetensors iterator, which bypasses
``should_skip_weight``, so the get_all_weights wrapper is the load-bearing patch.

Why a meta-path finder rather than an eager import: the submission's own
``sitecustomize.py`` installs lazy meta-path finders that surgically patch
``vllm.v1.spec_decode.gemma4`` / ``gpu_model_runner`` / ``gemma4_mtp`` /
``llm_base_proposer`` / ``vllm.attention.layer`` when those modules are first
imported. This module is triggered from a ``.pth`` (site processing, BEFORE the
submission's sitecustomize runs), so importing vllm eagerly here is avoided --
instead we install a finder that defers our patch until ``default_loader`` is
naturally imported during engine init, by which point the submission's finders
are already in ``sys.meta_path``. (Verified: importing default_loader pulls in
none of the submission's finder targets, so the two patch sets never collide.)

Gated on LMHEAD_INT4_SKIP_STRAY=1 so it is a complete no-op for every other
process / serve config; only the Stage-2 int4 serve sets it.
"""
import importlib.abc
import importlib.util
import os
import sys

_STRAY = "lm_head.weight"
_TARGET = "vllm.model_executor.model_loader.default_loader"


def _patch_default_loader(module) -> None:
    dl = module
    if getattr(dl.DefaultModelLoader, "_lmhead_int4_gaw_patched", False):
        return
    _orig_gaw = dl.DefaultModelLoader.get_all_weights

    def _gaw(self, model_config, model):
        for name, w in _orig_gaw(self, model_config, model):
            if name == _STRAY:
                continue
            yield name, w

    dl.DefaultModelLoader.get_all_weights = _gaw
    dl.DefaultModelLoader._lmhead_int4_gaw_patched = True
    print(f"[lmhead-int4-shim] get_all_weights patched (pid {os.getpid()}): "
          f"dropping stray bf16 {_STRAY}", flush=True)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self._orig = orig_loader

    def create_module(self, spec):
        return self._orig.create_module(spec)

    def exec_module(self, module):
        self._orig.exec_module(module)
        try:
            _patch_default_loader(module)
        except Exception as exc:  # pragma: no cover
            print(f"[lmhead-int4-shim] patch FAILED (pid {os.getpid()}): {exc!r}",
                  flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        # Resolve the real spec without recursing into ourselves.
        sys.meta_path.remove(self)
        try:
            real = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if real is None or real.loader is None:
            return None
        real.loader = _PatchingLoader(real.loader)
        return real


def _install() -> None:
    if os.environ.get("LMHEAD_INT4_SKIP_STRAY") != "1":
        return
    # Already imported (e.g. re-exec): patch in place; else install the finder.
    if _TARGET in sys.modules:
        _patch_default_loader(sys.modules[_TARGET])
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


_install()
