"""Serve-side injector for PR #563 (base_fullhead quality under sampling).

Auto-imported by Python's `site` in EVERY process started with this dir on
PYTHONPATH (including the vLLM v1 EngineCore worker spawned by multiprocessing).

ONE job, orthogonal to model numerics: the prometheus route-name compat shim
(identical to the #547 headwidth_inject / downstream_quality_eval shim). vLLM
0.22.1rc1 mounts prometheus_fastapi_instrumentator whose `_get_route_name` calls
`route.path` on every `app.routes` entry; newer FastAPI has `_IncludedRouter`
entries with no `.path`, which 500s EVERY request (incl. /v1/models and
/v1/chat/completions). We descend safely instead. No head patch here: both arms
serve the full-vocab head on pure-vanilla vLLM, differing only in checkpoint.
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
    print("[sitecustomize-563] prometheus route-name compat shim installed "
          f"pid={os.getpid()}", file=sys.stderr, flush=True)


_install_prometheus_route_compat()
