"""osoi-v0: env-gated identity-skip of Gemma4 decoder layers.

LSK_SKIP_LAYERS="2,3,7" makes the listed TARGET layers return their input
unchanged (each Gemma4DecoderLayer is residual-closed, so identity-skip is
exact layer removal numerically). Applied via meta-path hook on module load
so it works inside the EngineCore process.

Used in two roles:
  1. Screening: toggle skip sets on the unmodified checkpoint via env.
  2. Cross-check: the baked (surgically layer-removed) checkpoint must score
     identically to runtime-skip on the original checkpoint.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_SKIP = {
    int(x)
    for x in os.environ.get("LSK_SKIP_LAYERS", "").replace(" ", "").split(",")
    if x
}
_TARGET = "vllm.model_executor.models.gemma4"


def _apply(module):
    cls = module.Gemma4DecoderLayer
    orig = cls.forward

    def forward(self, positions, hidden_states, residual, per_layer_input=None, **kw):
        if self.layer_idx in _SKIP:
            return hidden_states, None
        return orig(self, positions, hidden_states, residual, per_layer_input, **kw)

    cls.forward = forward
    print(
        f"[osoi-lsk] identity-skip active for target layers {sorted(_SKIP)} "
        f"(pid {os.getpid()})",
        file=sys.stderr,
        flush=True,
    )


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _apply(module)


class _Finder(importlib.abc.MetaPathFinder):
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
        spec.loader = _Loader(spec.loader)
        return spec


if _SKIP:
    if _TARGET in sys.modules:
        _apply(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
