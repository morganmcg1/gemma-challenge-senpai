"""PR #789 — drafter CUDA-graph dispatch probe (env-gated, CGPROBE=1).

Settles the Step-1 question: at runtime, are the 6 MTP-drafter proposer passes
dispatched as a captured CUDA graph (PIECEWISE/FULL) or run eager (NONE)?

Wraps three seams in the EngineCore worker process, all at the python call
boundary (OUTSIDE any graph capture, so capture/replay is undisturbed):

  - vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer
      ._determine_batch_execution_and_padding
      -> logs, per draft forward pass: input num_tokens, the dispatched
         CUDAGraphMode (NONE=eager / PIECEWISE / FULL), and the padded
         num_tokens. THIS is the decisive signal.
  - vllm.v1.spec_decode.gemma4.Gemma4Proposer.propose
      -> propose() CPU wall + CUDA-event GPU span (draft pass-block timing).
  - vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model
      -> verify-pass CPU wall + GPU span (the GPU-bound captured anchor).

Output goes to stdout (=> server.log):
  [cgprobe] dispatch call=<n> ctx=<first|loop> num_tokens=<in> mode=<MODE> padded=<n>
  [cgprobe] timing kind=<draft|exec> i=<step> cpu=<ms> gpu=<ms>
  [cgprobe] agg dispatch mode counts: {MODE: count, ...}  (every CGPROBE_REPORT)
  [cgprobe] agg timing kind=<k> n=<n> cpu p50=.. gpu p50=..

No files written; no served-model behavior altered. Byte-verbatim no-op unless
CGPROBE=1.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import time
from collections import deque
from typing import Any

CGPROBE = os.environ.get("CGPROBE", "0") == "1"
WARMUP_SKIP = int(os.environ.get("CGPROBE_WARMUP_SKIP", "60"))
RAW_COUNT = int(os.environ.get("CGPROBE_RAW_COUNT", "60"))
REPORT_EVERY = int(os.environ.get("CGPROBE_REPORT_EVERY", "120"))

PROPOSE_TARGET = "vllm.v1.spec_decode.gemma4"
RUNNER_TARGET = "vllm.v1.worker.gpu_model_runner"
BASE_TARGET = "vllm.v1.spec_decode.llm_base_proposer"

_state: dict[str, Any] = {
    "exec_i": 0,
    "last_ret": None,
    "cur_draft": None,
    "pending": deque(),
    "timing": {},          # kind -> list[(cpu, gpu)]
    "dispatch_n": 0,       # _determine_* call count (post-warmup, draft-only best-effort)
    "dispatch_total": 0,   # all calls incl warmup
    "mode_counts": {},     # mode_name -> count (post-warmup)
    "shape_counts": {},    # (num_tokens_in -> padded, mode) -> count (post-warmup)
}


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _resolve_pending(force: bool = False) -> None:
    pend = _state["pending"]
    while pend:
        rec = pend[0]
        ev1 = rec["ev1"]
        if not force and not ev1.query():
            break
        if force:
            ev1.synchronize()
        gpu = rec["ev0"].elapsed_time(ev1)
        pend.popleft()
        kind, i = rec["kind"], rec["i"]
        if i >= WARMUP_SKIP:
            _state["timing"].setdefault(kind, []).append((rec["cpu"], gpu))
        if WARMUP_SKIP <= i < WARMUP_SKIP + RAW_COUNT:
            print(
                f"[cgprobe] timing kind={kind} i={i} cpu={rec['cpu']:.3f} gpu={gpu:.3f}",
                flush=True,
            )


def _report_timing() -> None:
    for kind, rows in _state["timing"].items():
        if not rows:
            continue
        cpu = [r[0] for r in rows]
        gpu = [r[1] for r in rows]
        print(
            f"[cgprobe] agg timing kind={kind} n={len(rows)} "
            f"cpu p50={_pct(cpu,50):.3f} p90={_pct(cpu,90):.3f} mean={sum(cpu)/len(cpu):.3f} | "
            f"gpu p50={_pct(gpu,50):.3f} p90={_pct(gpu,90):.3f} mean={sum(gpu)/len(gpu):.3f}",
            flush=True,
        )


def _wrap_determine(module: Any) -> None:
    cls = module.SpecDecodeBaseProposer
    base = cls._determine_batch_execution_and_padding

    def wrapped(self: Any, num_tokens: int, *a: Any, **k: Any) -> Any:
        out = base(self, num_tokens, *a, **k)
        _state["dispatch_total"] += 1
        try:
            mode = out[0]
            padded = out[1]
            mode_name = getattr(mode, "name", str(mode))
        except Exception:
            return out
        # Skip warmup/capture-era calls. After WARMUP_SKIP total calls treat as
        # steady decode (capture happens once at startup, before any decode).
        if _state["dispatch_total"] > WARMUP_SKIP:
            n = _state["dispatch_n"]
            _state["dispatch_n"] = n + 1
            _state["mode_counts"][mode_name] = _state["mode_counts"].get(mode_name, 0) + 1
            key = f"{num_tokens}->{padded}:{mode_name}"
            _state["shape_counts"][key] = _state["shape_counts"].get(key, 0) + 1
            if n < RAW_COUNT:
                print(
                    f"[cgprobe] dispatch call={n} num_tokens={num_tokens} "
                    f"mode={mode_name} padded={padded}",
                    flush=True,
                )
            if _state["dispatch_n"] % REPORT_EVERY == 0:
                print(
                    f"[cgprobe] agg dispatch n={_state['dispatch_n']} "
                    f"mode_counts={_state['mode_counts']} shapes={_state['shape_counts']}",
                    flush=True,
                )
                _report_timing()
        return out

    cls._determine_batch_execution_and_padding = wrapped
    print("[cgprobe] _determine_batch_execution_and_padding wrapper active", flush=True)


def _wrap_propose(module: Any) -> None:
    import torch

    cls = module.Gemma4Proposer
    base = cls.propose

    def propose(self: Any, *a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        try:
            return base(self, *a, **k)
        finally:
            ev1.record()
            _state["pending"].append(
                {"kind": "draft", "i": _state["exec_i"], "cpu": (time.perf_counter() - t0) * 1e3,
                 "ev0": ev0, "ev1": ev1}
            )
            _resolve_pending()

    cls.propose = propose
    print("[cgprobe] propose wrapper active", flush=True)


def _wrap_execute(module: Any) -> None:
    import torch

    cls = module.GPUModelRunner
    base = cls.execute_model

    def execute_model(self: Any, *a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        try:
            return base(self, *a, **k)
        finally:
            ev1.record()
            _state["pending"].append(
                {"kind": "exec", "i": _state["exec_i"], "cpu": (time.perf_counter() - t0) * 1e3,
                 "ev0": ev0, "ev1": ev1}
            )
            _state["exec_i"] += 1
            _resolve_pending()

    cls.execute_model = execute_model
    print("[cgprobe] execute_model wrapper active", flush=True)


class _ChainLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, patch_fn: Any) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _ChainFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch_fn: Any) -> None:
        self._target = target
        self._patch_fn = patch_fn
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _ChainLoader(spec.loader, self._patch_fn)
        return spec


if CGPROBE:
    sys.meta_path.insert(0, _ChainFinder(BASE_TARGET, _wrap_determine))
    sys.meta_path.insert(0, _ChainFinder(PROPOSE_TARGET, _wrap_propose))
    sys.meta_path.insert(0, _ChainFinder(RUNNER_TARGET, _wrap_execute))
    print("[cgprobe] finders registered", flush=True)
