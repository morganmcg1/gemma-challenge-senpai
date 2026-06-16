"""Serve-side injector for the downstream-quality eval, auto-imported by Python's
`site` in EVERY process started with this directory on PYTHONPATH — including the
vLLM v1 EngineCore worker that multiprocessing spawns (it inherits PYTHONPATH and
re-runs site init). This is the same mechanism the real submission uses to install
its patches into the spawned model-runner.

Two independent jobs:

1. Prometheus route-name compat shim (ALWAYS, both arms). vLLM 0.22.1rc1
   unconditionally mounts `prometheus_fastapi_instrumentator`, whose
   `routing._get_route_name` does `route.path` on every entry of `app.routes`.
   Under this box's newer FastAPI some entries are `_IncludedRouter` objects with
   no `.path`, so EVERY request (incl. `/v1/models` readiness and the eval
   completions) 500s with `AttributeError: '_IncludedRouter' object has no
   attribute 'path'`. We replace `_get_route_name` with a `getattr(route,"path")`
   -guarded version that descends into sub-routers. This touches only the metrics
   middleware's request labeling — it is completely orthogonal to model numerics,
   so it is safe to apply identically to the pure-vanilla base arm.

2. pck04 lm_head patch (ONLY when PCK04_KEEPSET is set -> the ship arm). Imports
   `serve_patch_pck04` from the submission dir, which registers a meta-path finder
   that rebuilds Gemma4ForCausalLM.lm_head to K=len(keep_ids) rows (matching the
   pruned osoi5-12k checkpoint, which vanilla vLLM otherwise asserts on) and
   scatters the [M, K] logits back to full vocab with -inf at non-kept positions.
   No-op when unset, so the base arm stays numerically vanilla.
"""
import os
import sys


def _install_prometheus_route_compat():
    """Make prometheus-fastapi-instrumentator tolerate routes without `.path`."""
    try:
        import prometheus_fastapi_instrumentator.routing as _r
        from starlette.routing import Match, Mount
    except Exception:
        return  # instrumentator/starlette not present -> nothing to patch

    def _safe_get_route_name(scope, routes, route_name=None):
        for route in routes:
            try:
                match, child_scope = route.matches(scope)
            except Exception:
                continue
            path = getattr(route, "path", None)
            sub = getattr(route, "routes", None)
            if match == Match.FULL:
                if path is None:
                    # router-like wrapper (e.g. FastAPI _IncludedRouter): descend.
                    if sub:
                        return _safe_get_route_name({**scope, **child_scope}, sub, route_name)
                    return route_name
                route_name = path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child = _safe_get_route_name(child_scope, route.routes, route_name)
                    route_name = None if child is None else route_name + child
                return route_name
            elif match == Match.PARTIAL and route_name is None and path is not None:
                route_name = path
        return None

    # `get_route_name` (the public entry) calls `_get_route_name` via a module-global
    # lookup at call time, so replacing this single name fixes every caller binding.
    _r._get_route_name = _safe_get_route_name
    print("[sitecustomize] prometheus route-name compat shim installed "
          f"pid={os.getpid()}", file=sys.stderr, flush=True)


_install_prometheus_route_compat()

_keepset = os.environ.get("PCK04_KEEPSET", "")
if _keepset:
    _sub = os.environ.get(
        "PCK04_PATCH_DIR",
        "/workspace/senpai/target/submissions/fa2sw_strict_surgical357",
    )
    if _sub not in sys.path:
        sys.path.insert(0, _sub)
    import serve_patch_pck04  # noqa: F401  (registers the meta-path finder on import)

    print(
        f"[sitecustomize] pck04 inject ACTIVE keepset={_keepset} "
        f"patch_dir={_sub} pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
