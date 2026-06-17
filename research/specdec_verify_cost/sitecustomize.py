"""PR #575 wirbel — research-dir sitecustomize shim (M-step verify-cost probe).

The vLLM worker disables user-site import, so ``usercustomize`` never loads — but
``sitecustomize`` IS imported by ``site`` at interpreter start. To inject the
``mstep_probe`` timing hooks without editing any shipped file, the driver puts
THIS directory FIRST on PYTHONPATH (so ``import sitecustomize`` resolves here) and
passes ``MSTEP_PKG_DIR`` = the submission dir. This shim then:

  1. Executes the submission's real ``sitecustomize.py`` by explicit path, so the
     entire served stack (loopgraph / fused-argmax / fastrender / pck04 / detok /
     fa-sliding / splitkv / precache / router-guard) installs exactly as unshadowed.
     (Under the ngram drafter the MTP-specific finders target modules that never
     import, so they are inert; the generic runner/renderer patches still apply —
     this keeps the fast kernels that give the 252.69 base_fullhead anchor.)
  2. Imports ``mstep_probe`` AFTER the package finders, so the probe's finders sit
     frontmost and its execute_model / compute_logits wraps are the OUTERMOST wrap
     (it therefore times the fully-patched served step — the real per-step cost).
  3. Rebinds ``sys.modules['sitecustomize']`` to the submission module so runtime
     consumers of ``import sitecustomize`` (serve.py's lazy helper lookups) resolve
     the real helper set.

Default-off: with MSTEP_PKG_DIR unset this shim is never the one site imports, and
the probe itself is gated on MSTEP=1. No shipped submission file is modified.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_PKG = os.environ.get("MSTEP_PKG_DIR")
_mod = sys.modules.get("_mstep_pkg_sitecustomize")
if _PKG and _mod is None:
    _path = os.path.join(_PKG, "sitecustomize.py")
    if os.path.exists(_path):
        try:
            _spec = importlib.util.spec_from_file_location("_mstep_pkg_sitecustomize", _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["_mstep_pkg_sitecustomize"] = _mod
            assert _spec and _spec.loader
            _spec.loader.exec_module(_mod)
            print(f"[mstep-shim] ran submission sitecustomize {_path} pid={os.getpid()}", flush=True)
        except Exception as _exc:  # pragma: no cover
            print(f"[mstep-shim] submission sitecustomize FAILED: {_exc!r}", flush=True)
            raise

try:
    import mstep_probe  # noqa: F401  (self-registers when MSTEP=1)
except Exception as _exc:  # pragma: no cover
    print(f"[mstep-shim] mstep_probe import failed: {_exc!r}", flush=True)

if _mod is not None:
    sys.modules["sitecustomize"] = _mod
    print(f"[mstep-shim] rebound sys.modules['sitecustomize'] -> submission module pid={os.getpid()}", flush=True)
