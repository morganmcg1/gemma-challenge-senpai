"""Auto-loaded at interpreter startup (Python imports ``sitecustomize`` during
``site`` initialization for every process whose ``sys.path`` contains this file).

``serve.py`` prepends this submission directory to ``PYTHONPATH`` before launching
the vLLM OpenAI server, so this module runs in every process in the server tree:
the ``api_server`` process, the (forked or spawned) ``EngineCore`` process, and the
worker process where ``GPUModelRunner`` actually builds attention groups. That is
the only place the ``num_heads`` attention-group fix has to take effect, and the
``EngineCore`` start method may be ``spawn`` (vLLM forces spawn when CUDA is already
initialized in the parent), so a parent-process monkeypatch would not propagate --
``PYTHONPATH`` + ``sitecustomize`` reaches every process regardless of fork/spawn.

We do NOT import vLLM here: ``sitecustomize`` runs at startup for *every* Python
process that uses this venv (pip, helper scripts, the benchmark client), and a full
vLLM/torch import there would be slow and could fail off-GPU. Instead we install a
one-shot ``sys.meta_path`` finder that applies the patch the moment
``vllm.v1.worker.gpu_model_runner`` is first imported, and is a no-op otherwise.
"""

import os
import sys

_TARGET = "vllm.v1.worker.gpu_model_runner"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _apply(module) -> None:
    try:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import vllm_attn_group_patch

        vllm_attn_group_patch.apply(module)
    except Exception:
        import logging

        logging.getLogger("int4_mtp_drafter.patch").exception(
            "failed to apply attention-group num_heads patch"
        )


if _TARGET in sys.modules:
    _apply(sys.modules[_TARGET])
else:
    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _SpecDecodePatchFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            # One-shot: drop ourselves so the real loaders resolve the spec and
            # we never recurse through the find_spec() call below.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec_module = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec_module):
                _orig(module)
                _apply(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _SpecDecodePatchFinder())


# --- Output-neutral prometheus _IncludedRouter / missing-`.path` startup-500 guard ---
# vLLM 0.22.0 floors ``fastapi>=0.115`` and ``prometheus-fastapi-instrumentator>=7.0.0``
# by lower bound only. A fresh runner resolve pulls ``fastapi>=0.118`` / ``starlette>=1``,
# whose ``include_router`` appends a pathless ``_IncludedRouter`` route to ``app.routes``.
# The instrumentator's ``_get_route_name`` does ``route.path`` on it -> AttributeError ->
# HTTP 500 on EVERY request -> ``/v1/models`` never becomes ready -> the benchmark job
# aborts in ``wait_for_models``. This wraps ``_get_route_name`` to return None on that
# AttributeError (the metric route-name label is simply dropped for the unlabelable
# route) and is a byte-verbatim no-op on any normal path-bearing route. It touches ONLY
# the prometheus metrics-middleware route lookup: model weights, logits, sampling, PPL,
# and greedy decode are numerically unaffected. Validated output-neutral in this repo
# (kanna PR #177; W&B bjtwr9jn). Eager import is safe -- the instrumentator is a
# lightweight pure-Python dep (no torch/GPU) always present in the serve venv. The
# 3-arg signature matches prometheus_fastapi_instrumentator 8.x's
# ``_get_route_name(scope, routes, route_name=None)`` so the in-module recursion through
# Mount sub-routes also stays guarded.
try:
    import prometheus_fastapi_instrumentator.routing as _prom_routing

    _prom_orig_get_route_name = _prom_routing._get_route_name

    def _prom_guarded_get_route_name(scope, routes, route_name=None):
        try:
            return _prom_orig_get_route_name(scope, routes, route_name)
        except AttributeError:
            return None

    _prom_routing._get_route_name = _prom_guarded_get_route_name
except Exception:
    import logging as _logging

    _logging.getLogger("int4_mtp_drafter.prometheus_guard").exception(
        "failed to apply prometheus _IncludedRouter route-name guard"
    )
