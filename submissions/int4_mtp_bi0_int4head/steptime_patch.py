"""agent-smith: per-step timeline probe (env-gated, STEPTIME=1).

Wraps two seams in the worker process with perf_counter + CUDA event pairs:

  - vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model
      -> per step: gap_ms (CPU time since previous call returned),
         cpu_ms (call wall), gpu_ms (CUDA-event elapsed inside the call)
  - vllm.v1.spec_decode.gemma4.Gemma4Proposer.propose
      -> draft_cpu_ms / draft_gpu_ms, attributed to the enclosing step

Events are recorded at the python call boundary, OUTSIDE any CUDA-graph
capture performed inside the call, so the loopgraph/onegraph machinery is
undisturbed. Composes with the package's existing meta-path finders: our
finder sits in front, re-resolves the spec through the remaining finders
(their _busy-guard pattern), and applies our patch after theirs.

Output goes to stdout (=> job_logs.txt):
  [steptime] raw i=<step> spec=<0|1> gap=.. cpu=.. gpu=.. dcpu=.. dgpu=..
      (for a window of steps, default 40..200)
  [steptime] agg n=.. spec=<0|1> gap p50/p90/mean=.. cpu .. gpu .. dgpu ..
      (every STEPTIME_REPORT_EVERY resolved steps, cumulative, warmup excluded)

No files are written; no behavior of the served model is altered.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import time
from collections import deque
from typing import Any

STEPTIME = os.environ.get("STEPTIME", "0") == "1"
RAW_START = int(os.environ.get("STEPTIME_RAW_START", "40"))
RAW_COUNT = int(os.environ.get("STEPTIME_RAW_COUNT", "160"))
REPORT_EVERY = int(os.environ.get("STEPTIME_REPORT_EVERY", "1024"))
WARMUP_SKIP = int(os.environ.get("STEPTIME_WARMUP_SKIP", "64"))

RUNNER_TARGET = "vllm.v1.worker.gpu_model_runner"
PROPOSE_TARGET = "vllm.v1.spec_decode.gemma4"

_state: dict[str, Any] = {
    "i": 0,                 # execute_model call index
    "last_ret": None,       # perf_counter at previous execute_model return
    "cur_draft": None,      # draft measurement of the in-flight step
    "pending": deque(),     # unresolved records (with CUDA events)
    "agg": {},              # kind -> list of (gap, cpu, gpu, dcpu, dgpu)
    "reported": 0,
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
        dev1 = rec["dev1"]
        if not force and not ev1.query():
            break
        if force:
            ev1.synchronize()
        gpu = rec["ev0"].elapsed_time(ev1)
        dgpu = rec["dev0"].elapsed_time(dev1) if dev1 is not None else 0.0
        pend.popleft()
        i = rec["i"]
        kind = rec.get("kind", "exec")
        if i >= WARMUP_SKIP:
            _state["agg"].setdefault(kind, []).append(
                (rec["gap"], rec["cpu"], gpu, rec["dcpu"], dgpu)
            )
        if RAW_START <= i < RAW_START + RAW_COUNT:
            print(
                f"[steptime] raw i={i} kind={kind} gap={rec['gap']:.3f} "
                f"cpu={rec['cpu']:.3f} gpu={gpu:.3f} dcpu={rec['dcpu']:.3f} "
                f"dgpu={dgpu:.3f}",
                flush=True,
            )
        done = sum(len(v) for v in _state["agg"].values())
        if done and done % REPORT_EVERY == 0 and done != _state["reported"]:
            _state["reported"] = done
            _report()


def _report() -> None:
    for kind, rows in _state["agg"].items():
        if not rows:
            continue
        cols = list(zip(*rows))
        names = ("gap", "cpu", "gpu", "dcpu", "dgpu")
        parts = []
        for name, vals in zip(names, cols):
            v = list(vals)
            parts.append(
                f"{name} p50={_pct(v,50):.3f} p90={_pct(v,90):.3f} "
                f"mean={sum(v)/len(v):.3f}"
            )
        print(
            f"[steptime] agg n={len(rows)} kind={kind} " + " | ".join(parts),
            flush=True,
        )


def _wrap_execute_model(module: Any) -> None:
    import torch

    runner_cls = module.GPUModelRunner
    base = runner_cls.execute_model

    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        now = time.perf_counter()
        gap = 0.0 if _state["last_ret"] is None else (now - _state["last_ret"]) * 1e3
        _state["cur_draft"] = [0.0, None, None]  # dcpu, dev0, dev1
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        try:
            out = base(self, *args, **kwargs)
        finally:
            ev1.record()
            ret = time.perf_counter()
            dcpu, dev0, dev1 = _state["cur_draft"]
            _state["pending"].append(
                {
                    "i": _state["i"],
                    "gap": gap,
                    "cpu": (ret - now) * 1e3,
                    "ev0": ev0,
                    "ev1": ev1,
                    "dcpu": dcpu,
                    "dev0": dev0,
                    "dev1": dev1,
                }
            )
            _state["i"] += 1
            _state["last_ret"] = ret
            _state["cur_draft"] = None
            _resolve_pending()
        return out

    runner_cls.execute_model = execute_model
    print("[steptime] execute_model wrapper active", flush=True)


def _wrap_propose(module: Any) -> None:
    import torch

    proposer_cls = module.Gemma4Proposer
    base = proposer_cls.propose

    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        cur = _state["cur_draft"]
        if cur is None:
            # v1 finding: this wheel calls propose OUTSIDE execute_model —
            # record it as a standalone 'draft' record instead.
            t0 = time.perf_counter()
            dev0 = torch.cuda.Event(enable_timing=True)
            dev1 = torch.cuda.Event(enable_timing=True)
            dev0.record()
            try:
                return base(self, *args, **kwargs)
            finally:
                dev1.record()
                _state["pending"].append(
                    {
                        "i": _state["i"],
                        "kind": "draft",
                        "gap": 0.0,
                        "cpu": (time.perf_counter() - t0) * 1e3,
                        "ev0": dev0,
                        "ev1": dev1,
                        "dcpu": 0.0,
                        "dev0": None,
                        "dev1": None,
                    }
                )
        t0 = time.perf_counter()
        dev0 = torch.cuda.Event(enable_timing=True)
        dev1 = torch.cuda.Event(enable_timing=True)
        dev0.record()
        try:
            return base(self, *args, **kwargs)
        finally:
            dev1.record()
            cur[0] += (time.perf_counter() - t0) * 1e3
            if cur[1] is None:
                cur[1] = dev0
            cur[2] = dev1

    proposer_cls.propose = propose
    print("[steptime] propose wrapper active", flush=True)


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


if STEPTIME:
    sys.meta_path.insert(0, _ChainFinder(RUNNER_TARGET, _wrap_execute_model))
    sys.meta_path.insert(0, _ChainFinder(PROPOSE_TARGET, _wrap_propose))
    print("[steptime] finders registered", flush=True)
