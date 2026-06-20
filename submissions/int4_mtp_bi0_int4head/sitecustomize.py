"""Auto-loaded at interpreter startup (Python imports ``sitecustomize`` during
``site`` initialization for every process whose ``sys.path`` contains this file).

``serve.py`` prepends this submission directory to ``PYTHONPATH`` before launching
the vLLM OpenAI server, so this module runs in every process in the server tree:
the ``api_server`` process, the (forked or spawned) ``EngineCore`` process, and the
worker process where ``GPUModelRunner`` builds attention groups and the
``TritonAttentionImpl`` runs. The ``EngineCore`` start method may be ``spawn``
(vLLM forces spawn when CUDA is already initialized in the parent), so a
parent-process monkeypatch would not propagate -- ``PYTHONPATH`` + ``sitecustomize``
reaches every process regardless of fork/spawn.

We install two runtime patches, each applied the moment its target vLLM module is
first imported:

* ``vllm_attn_group_patch`` on ``vllm.v1.worker.gpu_model_runner`` -- backports the
  upstream attention-group ``num_heads`` dedup so the 4-head MTP drafter and the
  8-head int4 target land in separate attention groups (no-op without speculation).
* ``vllm_force2d_attn_patch`` on ``vllm.v1.attention.backends.triton_attn`` -- forces
  the 2D single-pass attention path for both decode and spec-verify, recovering the
  byte-exact greedy identity that this BI=0 submission would otherwise only get from
  the global batch-invariant ("BI-tax") kernels.

We do NOT import vLLM here: ``sitecustomize`` runs at startup for *every* Python
process that uses this venv (pip, helper scripts, the benchmark client), and a full
vLLM/torch import there would be slow and could fail off-GPU. Instead we install
one-shot ``sys.meta_path`` finders that apply each patch on first import of its
target module, and are no-ops otherwise.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _make_applier(patch_module_name, label):
    """Build an applier that imports ``patch_module_name`` and calls its
    ``apply(module)`` against the freshly-imported vLLM target module, logging
    (but not raising) on failure so a patch bug can never wedge the server."""

    def _apply(module) -> None:
        try:
            if _HERE not in sys.path:
                sys.path.insert(0, _HERE)
            patch = __import__(patch_module_name)
            patch.apply(module)
        except Exception:
            import logging

            logging.getLogger("int4_mtp_drafter.patch").exception(
                "failed to apply %s", label
            )

    return _apply


def _install_hook(target, applier):
    """Apply ``applier`` to ``target`` now if already imported, else install a
    one-shot meta-path finder that applies it on first import of ``target``."""
    if target in sys.modules:
        applier(sys.modules[target])
        return

    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _PatchFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target_module=None):
            if fullname != target:
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
                applier(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchFinder())


_install_hook(
    "vllm.v1.worker.gpu_model_runner",
    _make_applier("vllm_attn_group_patch", "attention-group num_heads patch"),
)
_install_hook(
    "vllm.v1.attention.backends.triton_attn",
    _make_applier("vllm_force2d_attn_patch", "force-2D attention patch"),
)
# AdaEDL entropy-gated draft early-stop. No-op unless DRAFT_STOP_ENTROPY is set,
# so the shipped submission stays byte-identical when the env is unset.
_install_hook(
    "vllm.v1.spec_decode.gemma4",
    _make_applier("adaedl_earlystop_patch", "AdaEDL entropy-gated draft early-stop"),
)

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
