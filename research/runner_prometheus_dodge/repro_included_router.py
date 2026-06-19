"""Reproduce the runner ``/v1/models`` 500 (the prometheus ``_IncludedRouter``
missing-``.path`` bug that errored the int4_mtp_batchinv HF job) on a REAL ASGI
server, CPU-only, and prove the submission-side dodges turn it back into a 200.

Why CPU-only is faithful: the failure is 100% in the web stack
(fastapi ``include_router`` -> ``_IncludedRouter`` route object;
``prometheus_fastapi_instrumentator.routing._get_route_name`` does ``route.path``
on it). It is independent of vLLM/torch/the model -- the model loaded fine on the
runner; only ``GET /v1/models`` 500'd inside the metrics middleware. So a minimal
FastAPI app that mirrors how vLLM's OpenAI ``api_server`` wires its router +
prometheus instrumentation reproduces the exact ``AttributeError``.

vLLM 0.22.0's ``vllm.entrypoints.openai.api_server`` builds ``router = APIRouter()``
with the ``/v1/models`` handler, then ``app.include_router(router)`` on the parent
``FastAPI`` app, then instruments it with ``prometheus_fastapi_instrumentator``.
On fastapi>=0.118 / starlette>=1 ``include_router`` appends an ``_IncludedRouter``
(a ``BaseRoute`` subclass with no ``.path``) to ``app.routes`` instead of copying
the sub-routes inline -- the runner's exact drift (a fresh 2026-06-19 resolve of
vLLM's open ``fastapi>=0.115`` / ``prometheus-fastapi-instrumentator>=7`` bounds
pulls fastapi 0.137 / starlette 1.3.1).

Modes (REPRO_MODE env, default unpatched):
  unpatched -> bare app. 500 on the buggy stack; 200 on a stack predating
               ``_IncludedRouter`` (e.g. fastapi<0.116 -- which is dodge path a).
  guard     -> apply darwin's sitecustomize-style guard wrapping
               ``_get_route_name`` to return None on AttributeError (dodge path
               b). Expect 200 on every stack.

Prints one JSON line. Exit 0 always (verdict is in the JSON).
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def build_app():
    """A minimal mirror of vLLM's api_server router wiring + prometheus instrument."""
    from fastapi import APIRouter, FastAPI
    from prometheus_fastapi_instrumentator import Instrumentator

    router = APIRouter()

    @router.get("/v1/models")
    async def show_models():  # noqa: ANN202 - mirrors vLLM handler shape
        return {"object": "list", "data": [{"id": "gemma-4-e4b-it", "object": "model"}]}

    app = FastAPI()
    # The trigger: parent app includes the sub-router. On fastapi>=0.118 this
    # appends an _IncludedRouter (no .path) to app.routes.
    app.include_router(router)

    # vLLM wires the instrumentator unconditionally; .instrument() installs the
    # PrometheusFastApiInstrumentator HTTP middleware whose _get_handler ->
    # routing.get_route_name -> _get_route_name iterates app.routes doing
    # route.path.
    Instrumentator().instrument(app).expose(app)
    return app


def apply_guard() -> dict:
    """Darwin's PR-#177 guard: wrap prometheus _get_route_name so a pathless
    (``_IncludedRouter``) matched route returns None instead of raising. Output-
    neutral: byte-verbatim no-op on a normal (path-bearing) route. Touches ONLY
    the metrics-middleware route-name lookup -- nothing in the app, routes,
    handlers, or response path."""
    import prometheus_fastapi_instrumentator.routing as _r

    _orig = _r._get_route_name

    def _guarded(scope, routes, route_name=None):
        try:
            return _orig(scope, routes, route_name)
        except AttributeError:
            return None

    _r._get_route_name = _guarded
    return {"guard_applied": True, "wrapped": "prometheus_fastapi_instrumentator.routing._get_route_name"}


def main() -> int:
    import fastapi
    import starlette
    import prometheus_fastapi_instrumentator as pfi

    mode = os.environ.get("REPRO_MODE", "unpatched")
    out: dict = {
        "mode": mode,
        "fastapi": fastapi.__version__,
        "starlette": starlette.__version__,
        "prometheus_fastapi_instrumentator": pfi.__version__,
    }

    # Does this stack even build the pathless _IncludedRouter route? (informational)
    has_included_router = hasattr(fastapi.routing, "_IncludedRouter")
    out["stack_has_IncludedRouter"] = has_included_router

    if mode == "guard":
        out["guard"] = apply_guard()

    app = build_app()

    # Inspect the route objects the prometheus middleware will iterate.
    route_types = []
    pathless_routes = []
    for rt in app.routes:
        tn = type(rt).__name__
        route_types.append(tn)
        if not hasattr(rt, "path"):
            pathless_routes.append(tn)
    out["app_route_types"] = route_types
    out["pathless_route_types"] = sorted(set(pathless_routes))

    # Drive a real request through the full ASGI middleware stack (this is what
    # the runner's readiness probe `wait_for_models` does: GET /v1/models).
    from fastapi.testclient import TestClient

    captured_exc = None
    status = None
    try:
        # raise_server_exceptions=False -> mirror uvicorn: a handler/middleware
        # exception becomes a 500 response (what the runner saw), not a client
        # raise, so we can read the real status code + the server-side traceback.
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        status = resp.status_code
    except Exception as exc:  # pragma: no cover - belt & suspenders
        captured_exc = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    # Re-run with raise_server_exceptions=True to capture the actual exception
    # type/message the middleware raised (the AttributeError), for evidence.
    exc_type = None
    exc_msg = None
    if status == 500:
        try:
            client2 = TestClient(app, raise_server_exceptions=True)
            client2.get("/v1/models")
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc)

    out["status_code"] = status
    out["client_exception"] = captured_exc
    out["server_exception_type"] = exc_type
    out["server_exception_msg"] = exc_msg
    out["models_endpoint_ok"] = status == 200
    out["reproduced_runner_500"] = (
        status == 500
        and exc_type == "AttributeError"
        and exc_msg is not None
        and "path" in exc_msg
    )

    print("REPRO_JSON " + json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
