"""PR #549 fern — in-context lm_head probe (env-gated, FULLHEAD_HOOK=1).

Loaded into the vLLM worker via the research-dir ``sitecustomize.py`` chain (NOT
``usercustomize`` — user-site import is disabled in the vLLM worker, so the
always-imported ``sitecustomize`` channel is used instead). Registers a meta-path
finder that, after ``vllm.model_executor.models.gemma4`` loads, wraps
``Gemma4ForCausalLM.compute_logits`` (the 262k full-head GEMM boundary) to:

  1. Time the call with a CUDA-event pair (deferred resolution, no hot-path sync)
     -> per-call head GPU ms, bucketed by M (decode verify: M<=DECODE_M_MAX;
     prefill: larger). The CUDA-event time brackets ONLY the head GEMM call, so it
     is immune to any host-side hidden-state copy below.
  2. Count calls per bucket -> lets the driver recover E[accept] =
     decode_tokens / decode_verify_calls (head runs once per verify step).
  3. Optionally (FULLHEAD_DUMP=1) collect the input hidden_states for decode-shaped
     calls into a CPU bf16 ring, capped at FULLHEAD_DUMP_MAX rows, dumped via
     torch.save at process exit -> the held-out decode-position stream for the
     Stage-2 offline candidate miss-rate(K) measurement.

Default-off: with FULLHEAD_HOOK unset this module registers nothing, so the served
path is byte-identical. No shipped submission file is modified.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.util
import os
import sys
from collections import deque
from typing import Any

_ENABLED = os.environ.get("FULLHEAD_HOOK", "0") == "1"
_DUMP = os.environ.get("FULLHEAD_DUMP", "0") == "1"
_DUMP_PATH = os.environ.get("FULLHEAD_DUMP_PATH", "/tmp/fullhead_hidden.pt")
_DUMP_MAX = int(os.environ.get("FULLHEAD_DUMP_MAX", "60000"))
_DECODE_M_MAX = int(os.environ.get("FULLHEAD_DECODE_M_MAX", "8"))
_REPORT_EVERY = int(os.environ.get("FULLHEAD_REPORT_EVERY", "2000"))
_WARMUP_SKIP = int(os.environ.get("FULLHEAD_WARMUP_SKIP", "64"))

_TARGET = "vllm.model_executor.models.gemma4"

_state: dict[str, Any] = {
    "i": 0,
    "pending": deque(),
    # bucket -> [count, sum_ms]
    "decode": [0, 0.0],
    "prefill": [0, 0.0],
    "reported": 0,
    "hidden": [],        # list[cpu bf16 [m,h]]
    "hidden_rows": 0,
    "dumped": False,
    "m_hist": {},        # M -> count
}


def _resolve(force: bool = False) -> None:
    pend = _state["pending"]
    while pend:
        rec = pend[0]
        ev0, ev1, m, i = rec
        if not force and not ev1.query():
            break
        if force:
            ev1.synchronize()
        ms = ev0.elapsed_time(ev1)
        pend.popleft()
        if i >= _WARMUP_SKIP:
            b = _state["decode"] if m <= _DECODE_M_MAX else _state["prefill"]
            b[0] += 1
            b[1] += ms
        done = _state["decode"][0] + _state["prefill"][0]
        if done and done % _REPORT_EVERY == 0 and done != _state["reported"]:
            _state["reported"] = done
            _report()


def _report() -> None:
    d, p = _state["decode"], _state["prefill"]
    dm = d[1] / d[0] if d[0] else float("nan")
    pm = p[1] / p[0] if p[0] else float("nan")
    print(
        f"[fullhead] agg decode_calls={d[0]} decode_head_ms_mean={dm:.4f} "
        f"decode_head_ms_sum={d[1]:.1f} | prefill_calls={p[0]} "
        f"prefill_head_ms_mean={pm:.4f} | hidden_rows={_state['hidden_rows']} "
        f"m_hist={dict(sorted(_state['m_hist'].items()))}",
        flush=True,
    )


def _dump() -> None:
    if _state["dumped"] or not _state["hidden"]:
        return
    _state["dumped"] = True
    try:
        import torch
        H = torch.cat(_state["hidden"], dim=0)
        torch.save({"hidden": H, "dtype": "bf16", "rows": H.shape[0]}, _DUMP_PATH)
        print(f"[fullhead] dumped hidden states {tuple(H.shape)} -> {_DUMP_PATH}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[fullhead] dump FAILED: {exc!r}", flush=True)


def _atexit() -> None:
    _resolve(force=True)
    _report()
    _dump()
    d = _state["decode"]
    print(
        f"[fullhead] FINAL decode_verify_calls={d[0]} decode_head_ms_sum={d[1]:.2f} "
        f"decode_head_ms_mean={(d[1]/d[0] if d[0] else float('nan')):.4f}",
        flush=True,
    )


def _wrap(module: Any) -> None:
    import torch

    cls = getattr(module, "Gemma4ForCausalLM", None)
    if cls is None:
        print("[fullhead] WARN: Gemma4ForCausalLM not found; hook inert", flush=True)
        return
    base = cls.compute_logits

    def compute_logits(self_model: Any, hidden_states: torch.Tensor, *a: Any, **kw: Any) -> Any:
        m = int(hidden_states.shape[0]) if hidden_states is not None and hidden_states.dim() >= 1 else 0
        _state["m_hist"][m] = _state["m_hist"].get(m, 0) + 1
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = base(self_model, hidden_states, *a, **kw)
        ev1.record()
        _state["pending"].append((ev0, ev1, m, _state["i"]))
        _state["i"] += 1
        if _DUMP and m <= _DECODE_M_MAX and _state["hidden_rows"] < _DUMP_MAX and hidden_states is not None:
            try:
                h = hidden_states.detach().to("cpu", torch.bfloat16)
                _state["hidden"].append(h)
                _state["hidden_rows"] += h.shape[0]
            except Exception:
                pass
        # Dump the moment the ring fills — the EngineCore worker is hard-killed on
        # server shutdown and never runs atexit, so an atexit-only dump is lost. The
        # one-time torch.save spike lands in the dump pass (not the clean-timing pass).
        if _DUMP and not _state["dumped"] and _state["hidden_rows"] >= _DUMP_MAX:
            _dump()
        _resolve()
        return out

    cls.compute_logits = compute_logits
    print(
        f"[fullhead] wrapped Gemma4ForCausalLM.compute_logits "
        f"(dump={_DUMP} max={_DUMP_MAX} decode_M<={_DECODE_M_MAX}) pid={os.getpid()}",
        flush=True,
    )


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _wrap(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != _TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if _ENABLED:
    sys.meta_path.insert(0, _Finder())
    atexit.register(_atexit)
    print(f"[fullhead] finder registered (dump={_DUMP}) pid={os.getpid()}", flush=True)
