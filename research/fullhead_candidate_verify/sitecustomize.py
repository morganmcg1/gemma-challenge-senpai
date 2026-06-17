"""PR #549 fern — research-dir sitecustomize shim (probe injection, env-gated).

The vLLM worker disables user-site import, so ``usercustomize`` never loads — but
``sitecustomize`` IS reliably imported by ``site`` at interpreter start (the
submission's own serve patches depend on it). To inject the FULLHEAD_HOOK probe
without editing any shipped file, the driver puts THIS directory FIRST on
PYTHONPATH (so ``import sitecustomize`` resolves here) and passes
``FULLHEAD_PKG_DIR`` = the submission dir. This shim then:

  1. Executes the submission's real ``sitecustomize.py`` by explicit path, so the
     entire served stack (loopgraph / fused-argmax / fastrender / pck04 / detok /
     steptime / fa-sliding / splitkv / precache / router-guard) still installs
     exactly as it would unshadowed. Runs once (site never imports the package one
     directly because this dir shadows it).
  2. Imports ``fullhead_probe`` (which self-registers its meta-path finder only
     when FULLHEAD_HOOK=1; otherwise inert), AFTER the package finders, so the
     fullhead finder sits frontmost and composes on top of pck04's gemma4 finder.

Default-off: with FULLHEAD_PKG_DIR unset this shim is never the one site imports
(the driver only sets the research-first PYTHONPATH when probing), and the probe
itself is gated on FULLHEAD_HOOK. No shipped submission file is modified.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_PKG = os.environ.get("FULLHEAD_PKG_DIR")
_mod = sys.modules.get("_fullhead_pkg_sitecustomize")
if _PKG and _mod is None:
    _path = os.path.join(_PKG, "sitecustomize.py")
    if os.path.exists(_path):
        try:
            _spec = importlib.util.spec_from_file_location(
                "_fullhead_pkg_sitecustomize", _path
            )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["_fullhead_pkg_sitecustomize"] = _mod
            assert _spec and _spec.loader
            _spec.loader.exec_module(_mod)
            print(
                f"[fullhead-shim] ran submission sitecustomize {_path} pid={os.getpid()}",
                flush=True,
            )
        except Exception as _exc:  # pragma: no cover
            print(f"[fullhead-shim] submission sitecustomize FAILED: {_exc!r}", flush=True)
            raise

try:
    import fullhead_probe  # noqa: F401  (self-registers when FULLHEAD_HOOK=1)
except Exception as _exc:  # pragma: no cover
    print(f"[fullhead-shim] fullhead_probe import failed: {_exc!r}", flush=True)

# Rebind sys.modules['sitecustomize'] to the SUBMISSION module so runtime consumers of
# ``import sitecustomize`` resolve the real helper set. serve.py's rejection-sampler
# patch calls ``_gemma_sitecustomize._dixie_fused_accept_prep(...)`` at DECODE time
# (serve.py:427, lazy import inside forward) — that runs long after this site-init shim,
# so the rebind below is already in place. This research shim only needs to RUN once at
# site-init to register the probe finder (now on sys.meta_path: global + persistent);
# it must not remain the bound ``sitecustomize`` module or that helper lookup would
# AttributeError and kill the EngineCore (DIXIE_FUSED_ACCEPT_PREP_REQUIRE=1).
if _mod is not None:
    sys.modules["sitecustomize"] = _mod
    print(
        "[fullhead-shim] rebound sys.modules['sitecustomize'] -> submission module "
        f"pid={os.getpid()}",
        flush=True,
    )
