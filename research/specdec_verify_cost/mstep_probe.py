"""PR #575 wirbel — per-step VERIFY-cost probe, bucketed by verify granularity M.

Measures the speculative-decode verify-step cost curve ``C(M)`` directly on the
served base_fullhead stack (full 262k bf16 head, prune OFF). M = the number of
verified positions in a decode step = K+1 for draft-length K. The PR cost model:

    per verify step reads the weights ONCE and computes logits at M positions,
    cost C(M); A (1<=A<=M) tokens are emitted; served TPS ~= A / C(M).
    No-spec: A=M=1 -> TPS_base = 1/C(1) must reproduce 252.69 (wirbel #553).
    Asymptote: M->inf -> M/C(M) -> 1/(dC/dM); if C(M)=C_fixed+M*c_compute the
    drafter-independent ceiling is 1/c_compute.

Loaded into the vLLM worker via the research-dir ``sitecustomize.py`` chain (the
worker disables user-site import, so ``sitecustomize`` is the injection channel).
Env-gated on ``MSTEP=1``; unset -> this module registers nothing -> served path
byte-identical. No shipped submission file is modified.

Two seams, both timed with CUDA-event pairs recorded at the python call boundary
(OUTSIDE any CUDA-graph capture inside the call, so the served machinery is
undisturbed), and bucketed by M:

  - vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model
      -> PRIMARY: per-step wall_ms (perf_counter) + gpu_ms (CUDA event), the full
         verify-step latency C(M). M is read from the SchedulerOutput
         (``total_num_scheduled_tokens``); prefill steps (new request being
         prefilled, or M > MSTEP_M_CAP) are excluded so the small-M decode buckets
         stay clean.
  - vllm.model_executor.models.gemma4.Gemma4ForCausalLM.compute_logits
      -> DIAGNOSTIC: head-GEMM gpu_ms bucketed by m=hidden_states.shape[0]. The
         262k bf16 head reads 1.342 GB once per step regardless of M, so this
         bucket should be ~flat in M (the physics that makes the ceiling high).
         Also a cross-check that compute_logits M == scheduler M on decode steps.

Output JSON -> ``MSTEP_OUT`` (flushed every MSTEP_FLUSH_EVERY resolved decode
steps + atexit, so the last flush survives the EngineCore worker SIGTERM). The
process that never decoded (the API-server process imports this too) writes
nothing (guarded), so it never clobbers the worker's real data.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import deque
from typing import Any

_ENABLED = os.environ.get("MSTEP", "0") == "1"
_OUT = os.environ.get("MSTEP_OUT", "")
_PKG = os.environ.get("MSTEP_PKG_DIR", "")
_SERVED_K = int(os.environ.get("MSTEP_SERVED_K", "-1"))
_WARMUP_SKIP = int(os.environ.get("MSTEP_WARMUP_SKIP", "64"))
_M_CAP = int(os.environ.get("MSTEP_M_CAP", "64"))          # bucket only M <= cap (drop prefill)
_MAX_PER_BUCKET = int(os.environ.get("MSTEP_MAX_PER_BUCKET", "60000"))
_FLUSH_EVERY = int(os.environ.get("MSTEP_FLUSH_EVERY", "200"))
_REPORT_EVERY = int(os.environ.get("MSTEP_REPORT_EVERY", "4000"))

_RUNNER_TARGET = "vllm.v1.worker.gpu_model_runner"
_HEAD_TARGET = "vllm.model_executor.models.gemma4"

_state: dict[str, Any] = {
    "i": 0,                       # execute_model call index (all steps)
    "decode_done": 0,             # resolved qualifying decode steps
    "last_ret": None,             # perf_counter at previous execute_model return
    "pending": deque(),           # (ev0, ev1, M, wall_ms, gap_ms, i)
    "exec_buckets": {},           # M -> {"gpu": [...], "wall": [...], "gap": [...]}
    "head_pending": deque(),      # (ev0, ev1, m)
    "head_buckets": {},           # m -> [gpu_ms, ...]
    "m_logits_hist": {},          # m -> count (every compute_logits call)
    "draft_hist": {},             # num_draft_tokens -> count (decode steps)
    "prefill_steps": 0,
    "reported": 0,
    "wrote": False,
}


def _bucket(d: dict[int, dict[str, list]], m: int) -> dict[str, list]:
    b = d.get(m)
    if b is None:
        b = {"gpu": [], "wall": [], "gap": []}
        d[m] = b
    return b


def _read_M(args: tuple, kwargs: dict) -> tuple[int | None, bool, int]:
    """(M, is_prefill, num_draft) from the SchedulerOutput passed to execute_model."""
    so = args[0] if args else kwargs.get("scheduler_output")
    if so is None:
        return None, False, 0
    M = getattr(so, "total_num_scheduled_tokens", None)
    new_reqs = getattr(so, "scheduled_new_reqs", None)
    is_prefill = bool(new_reqs)
    spec = getattr(so, "scheduled_spec_decode_tokens", None)
    num_draft = 0
    if spec:
        try:
            num_draft = sum(len(v) for v in spec.values())
        except (TypeError, AttributeError):
            num_draft = 0
    return (int(M) if M is not None else None), is_prefill, num_draft


def _resolve_exec(force: bool = False) -> None:
    pend = _state["pending"]
    while pend:
        ev0, ev1, M, wall_ms, gap_ms, i = pend[0]
        if not force and not ev1.query():
            break
        if force:
            ev1.synchronize()
        gpu_ms = ev0.elapsed_time(ev1)
        pend.popleft()
        if i >= _WARMUP_SKIP and 0 < M <= _M_CAP:
            b = _bucket(_state["exec_buckets"], M)
            if len(b["gpu"]) < _MAX_PER_BUCKET:
                b["gpu"].append(gpu_ms)
                b["wall"].append(wall_ms)
                b["gap"].append(gap_ms)
            _state["decode_done"] += 1
            done = _state["decode_done"]
            if done % _REPORT_EVERY == 0 and done != _state["reported"]:
                _state["reported"] = done
                _report()
            if done % _FLUSH_EVERY == 0:
                _write()


def _resolve_head(force: bool = False) -> None:
    pend = _state["head_pending"]
    while pend:
        ev0, ev1, m = pend[0]
        if not force and not ev1.query():
            break
        if force:
            ev1.synchronize()
        gpu_ms = ev0.elapsed_time(ev1)
        pend.popleft()
        if 0 < m <= _M_CAP:
            lst = _state["head_buckets"].setdefault(m, [])
            if len(lst) < _MAX_PER_BUCKET:
                lst.append(gpu_ms)


def _summ(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    if n == 0:
        return {"n": 0}
    s = sorted(vals)
    mean = statistics.fmean(s)
    std = statistics.pstdev(s) if n > 1 else 0.0

    def _pct(p: float) -> float:
        k = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[k]

    # 5%-trimmed mean (robust to GC/scheduler spikes), for the linear fit
    lo = int(0.05 * n)
    hi = n - lo
    trimmed = s[lo:hi] if hi > lo else s
    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": statistics.median(s),
        "trimmed_mean_ms": statistics.fmean(trimmed),
        "std_ms": std,
        "p10_ms": _pct(0.10),
        "p90_ms": _pct(0.90),
        "min_ms": s[0],
        "max_ms": s[-1],
        "ci95_halfwidth_ms": (1.96 * std / (n ** 0.5)) if n > 1 else 0.0,
    }


def _report() -> None:
    parts = []
    for m in sorted(_state["exec_buckets"]):
        b = _state["exec_buckets"][m]
        if b["wall"]:
            parts.append(f"M={m} n={len(b['wall'])} "
                         f"wall_med={statistics.median(b['wall']):.4f} "
                         f"gpu_med={statistics.median(b['gpu']):.4f}")
    print(f"[mstep] agg decode_done={_state['decode_done']} K={_SERVED_K} "
          f"prefill={_state['prefill_steps']} | " + " | ".join(parts), flush=True)


def _write() -> None:
    if not _OUT:
        return
    # guard: only a process that actually timed decode steps writes (the API-server
    # process imports this module but never calls execute_model with decode work).
    if _state["decode_done"] == 0 and not _state["exec_buckets"]:
        return
    try:
        exec_summary = {
            str(m): {
                "wall": _summ(b["wall"]),
                "gpu": _summ(b["gpu"]),
                "gap": _summ(b["gap"]),
            }
            for m, b in sorted(_state["exec_buckets"].items())
        }
        head_summary = {str(m): _summ(v) for m, v in sorted(_state["head_buckets"].items())}
        payload = {
            "served_k": _SERVED_K,
            "warmup_skip": _WARMUP_SKIP,
            "m_cap": _M_CAP,
            "decode_steps_timed": _state["decode_done"],
            "prefill_steps_excluded": _state["prefill_steps"],
            "exec_step": exec_summary,           # PRIMARY: C(M) full verify-step latency
            "head_gemm": head_summary,           # DIAGNOSTIC: 262k head gpu_ms by M
            "m_logits_hist": {str(k): v for k, v in sorted(_state["m_logits_hist"].items())},
            "draft_hist": {str(k): v for k, v in sorted(_state["draft_hist"].items())},
            "pid": os.getpid(),
        }
        # atomic: write a temp file then rename, so a SIGKILL mid-write can never
        # corrupt a previously-good flush (the worker is SIGTERM->SIGKILLed at teardown).
        tmp = f"{_OUT}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, _OUT)
        _state["wrote"] = True
    except Exception as exc:  # pragma: no cover
        print(f"[mstep] write failed: {exc!r}", flush=True)


def _wrap_runner(module: Any) -> None:
    import torch

    runner_cls = module.GPUModelRunner
    base = runner_cls.execute_model

    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        now = time.perf_counter()
        gap = 0.0 if _state["last_ret"] is None else (now - _state["last_ret"]) * 1e3
        M, is_prefill, num_draft = _read_M(args, kwargs)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        try:
            out = base(self, *args, **kwargs)
        finally:
            ev1.record()
            ret = time.perf_counter()
            wall = (ret - now) * 1e3
            _state["last_ret"] = ret
            qualifies = (M is not None) and (not is_prefill) and (0 < M <= _M_CAP)
            if qualifies:
                _state["pending"].append((ev0, ev1, M, wall, gap, _state["i"]))
                _state["draft_hist"][num_draft] = _state["draft_hist"].get(num_draft, 0) + 1
            elif is_prefill or (M is not None and M > _M_CAP):
                _state["prefill_steps"] += 1
            _state["i"] += 1
            _resolve_exec()
            _resolve_head()
        return out

    runner_cls.execute_model = execute_model
    print(f"[mstep] execute_model wrapper active (K={_SERVED_K}, m_cap={_M_CAP}, "
          f"warmup_skip={_WARMUP_SKIP}) pid={os.getpid()}", flush=True)


def _wrap_head(module: Any) -> None:
    import torch

    cls = getattr(module, "Gemma4ForCausalLM", None)
    if cls is None:
        print("[mstep] WARN: Gemma4ForCausalLM not found; head diagnostic inert", flush=True)
        return
    base = cls.compute_logits

    def compute_logits(self_model: Any, hidden_states: Any, *a: Any, **kw: Any) -> Any:
        if hidden_states is None or not hasattr(hidden_states, "shape") or hidden_states.dim() < 1:
            return base(self_model, hidden_states, *a, **kw)
        m = int(hidden_states.shape[0])
        _state["m_logits_hist"][m] = _state["m_logits_hist"].get(m, 0) + 1
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = base(self_model, hidden_states, *a, **kw)
        ev1.record()
        _state["head_pending"].append((ev0, ev1, m))
        return out

    cls.compute_logits = compute_logits
    print(f"[mstep] compute_logits head-diagnostic wrapper active pid={os.getpid()}", flush=True)


def _atexit() -> None:
    _resolve_exec(force=True)
    _resolve_head(force=True)
    _report()
    _write()
    if _state["wrote"]:
        print(f"[mstep] FINAL decode_steps={_state['decode_done']} -> {_OUT}", flush=True)


class _ChainLoader(importlib.abc.Loader):
    def __init__(self, inner: Any, patch_fn: Any) -> None:
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


if _ENABLED:
    sys.meta_path.insert(0, _ChainFinder(_HEAD_TARGET, _wrap_head))
    sys.meta_path.insert(0, _ChainFinder(_RUNNER_TARGET, _wrap_runner))
    atexit.register(_atexit)
    print(f"[mstep] finders registered (K={_SERVED_K}) pid={os.getpid()}", flush=True)
