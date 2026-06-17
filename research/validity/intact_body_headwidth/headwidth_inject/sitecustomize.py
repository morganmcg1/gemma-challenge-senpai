"""Serve-side injector for the intact-body head-width sweep (PR #547).

Auto-imported by Python's `site` in EVERY process started with this dir on
PYTHONPATH (including the vLLM v1 EngineCore worker spawned by multiprocessing,
which inherits PYTHONPATH and re-runs site init).

Two jobs:

1. Prometheus route-name compat shim (ALWAYS). Identical to the
   downstream_quality_eval/pck04_inject shim: vLLM 0.22.1rc1 mounts
   prometheus_fastapi_instrumentator whose `_get_route_name` does `route.path`
   on every `app.routes` entry; newer FastAPI has `_IncludedRouter` entries with
   no `.path`, 500-ing every request. We descend safely instead. Orthogonal to
   model numerics.

2. head-width lm_head knob (ONLY when HEADWIDTH_KEEPSET is set). Imports
   serve_patch_headwidth from the experiment dir, which registers a meta-path
   finder that patches Gemma4ForCausalLM.compute_logits to either mask (quality,
   bit-faithful to a row-pruned head) or slice (genuine pruned-head GEMV for TPS)
   the output vocabulary to the keepset. No-op when unset -> pure-vanilla vLLM.
"""
import os
import sys


def _install_prometheus_route_compat():
    try:
        import prometheus_fastapi_instrumentator.routing as _r
        from starlette.routing import Match, Mount
    except Exception:
        return

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

    _r._get_route_name = _safe_get_route_name
    print("[sitecustomize-hw] prometheus route-name compat shim installed "
          f"pid={os.getpid()}", file=sys.stderr, flush=True)


_install_prometheus_route_compat()

_keepset = os.environ.get("HEADWIDTH_KEEPSET", "")
if _keepset:
    _patch_dir = os.environ.get(
        "HEADWIDTH_PATCH_DIR",
        os.path.dirname(os.path.abspath(__file__)).replace("/headwidth_inject", ""),
    )
    if _patch_dir not in sys.path:
        sys.path.insert(0, _patch_dir)
    import serve_patch_headwidth  # noqa: F401  (registers the meta-path finder)

    print(f"[sitecustomize-hw] head-width inject ACTIVE keepset={_keepset} "
          f"mode={os.environ.get('HEADWIDTH_MODE','mask')} patch_dir={_patch_dir} "
          f"pid={os.getpid()}", file=sys.stderr, flush=True)
