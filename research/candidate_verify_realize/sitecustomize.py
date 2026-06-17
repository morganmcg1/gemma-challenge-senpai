"""PR #566 fern — research-dir sitecustomize shim (CV-head probe injection).

The vLLM worker disables user-site import, so ``usercustomize`` never loads — but
``sitecustomize`` IS imported by ``site`` at interpreter start. To inject the
CV_HEAD replacement without editing any shipped file, the driver puts THIS
directory FIRST on PYTHONPATH (so ``import sitecustomize`` resolves here) and
passes ``CV_PKG_DIR`` = the submission dir. This shim then:

  1. Executes the submission's real ``sitecustomize.py`` by explicit path, so the
     entire served stack (loopgraph / fused-argmax / fastrender / pck04 / detok /
     steptime / fa-sliding / splitkv / precache / router-guard) installs exactly as
     unshadowed.
  2. Imports ``cv_head_probe`` (self-registers its meta-path finder only when
     CV_HEAD=1; otherwise inert) AFTER the package finders, so the CV finder sits
     frontmost and its compute_logits replacement is the OUTERMOST wrap.
  3. Rebinds ``sys.modules['sitecustomize']`` to the submission module so runtime
     consumers of ``import sitecustomize`` (serve.py's lazy decode-time helper
     lookups) resolve the real helper set.

Default-off: with CV_PKG_DIR unset this shim is never the one site imports, and the
probe itself is gated on CV_HEAD. No shipped submission file is modified.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_PKG = os.environ.get("CV_PKG_DIR")
_mod = sys.modules.get("_cv_pkg_sitecustomize")
if _PKG and _mod is None:
    _path = os.path.join(_PKG, "sitecustomize.py")
    if os.path.exists(_path):
        try:
            _spec = importlib.util.spec_from_file_location("_cv_pkg_sitecustomize", _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["_cv_pkg_sitecustomize"] = _mod
            assert _spec and _spec.loader
            _spec.loader.exec_module(_mod)
            print(f"[cv-shim] ran submission sitecustomize {_path} pid={os.getpid()}", flush=True)
        except Exception as _exc:  # pragma: no cover
            print(f"[cv-shim] submission sitecustomize FAILED: {_exc!r}", flush=True)
            raise

try:
    import cv_head_probe  # noqa: F401  (self-registers when CV_HEAD=1)
except Exception as _exc:  # pragma: no cover
    print(f"[cv-shim] cv_head_probe import failed: {_exc!r}", flush=True)

if _mod is not None:
    sys.modules["sitecustomize"] = _mod
    print(f"[cv-shim] rebound sys.modules['sitecustomize'] -> submission module pid={os.getpid()}", flush=True)
