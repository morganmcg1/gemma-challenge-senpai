"""Serve-side injector for the vanilla-base serve-regression card (PR #557).

Auto-imported by Python's `site` in EVERY process started with this dir on
PYTHONPATH — including the vLLM v1 EngineCore worker that multiprocessing spawns
(it inherits PYTHONPATH and re-runs site init). Same mechanism the real
submission and ubel #511's pck04_inject use.

Two jobs, both orthogonal-or-identical to ubel #511's base serve EXCEPT the
attention-backend routing this card is studying:

1. Prometheus route-name compat shim (ALWAYS). Identical to
   research/validity/downstream_quality_eval/pck04_inject/sitecustomize.py — vLLM
   0.22.1rc1 mounts prometheus_fastapi_instrumentator whose
   `routing._get_route_name` does `route.path` on `_IncludedRouter` wrappers that
   lack `.path`, 500-ing every request. Touches only metrics request-labeling;
   orthogonal to model numerics.

2. The surgical attention patches, env-gated, EXACTLY as base_fullhead loads them:
   - `fa_sliding_patch`   (FA_SLIDING=1): route the head_dim=256 SLIDING layers to
     FLASH_ATTN (full head_dim=512 layers stay on the forced TRITON_ATTN), the
     sidestep of vLLM dev307's Gemma4 forced-TRITON regression.
   - `surgical_attn_patch` (SURGICAL_ATTN_USE_3D_OFF=1): force the 2D
     order-preserving path on the TRITON full-attention reductions; no matmul tax.
   Both are no-ops when their env flag is unset (Stage-1 / Stage-2 base arms keep
   pure-vanilla routing).
"""
import os
import sys


def _install_prometheus_route_compat():
    """Make prometheus-fastapi-instrumentator tolerate routes without `.path`.
    Copied verbatim from pck04_inject/sitecustomize.py (numerics-orthogonal)."""
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
    print("[sitecustomize-557] prometheus route-name compat shim installed "
          f"pid={os.getpid()}", file=sys.stderr, flush=True)


_install_prometheus_route_compat()

# Make the submission's patch modules importable (same files base_fullhead uses).
_SUB = os.environ.get(
    "PR557_PATCH_DIR",
    "/workspace/senpai/target/submissions/fa2sw_strict_surgical357",
)
if _SUB not in sys.path:
    sys.path.insert(0, _SUB)

if os.environ.get("FA_SLIDING", "0") == "1":
    import fa_sliding_patch  # noqa: F401  (registers Attention.__init__ wrapper on import)
    print(f"[sitecustomize-557] fa_sliding_patch imported pid={os.getpid()}",
          file=sys.stderr, flush=True)

if os.environ.get("SURGICAL_ATTN_USE_3D_OFF", "0") == "1":
    import surgical_attn_patch  # noqa: F401  (forces 2D order-preserving path on import)
    print(f"[sitecustomize-557] surgical_attn_patch imported pid={os.getpid()}",
          file=sys.stderr, flush=True)
