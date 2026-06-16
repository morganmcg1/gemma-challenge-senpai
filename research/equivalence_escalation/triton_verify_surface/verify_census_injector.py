"""Env-gated served-stack routing+timing census for the M=8 verify attention surface
(PR #459, wirbel).

LOCAL analysis ONLY. When ``WIRBEL_VCENSUS`` is set (=1) this wraps the two attention
entry points the served Gemma-4-E4B stack can route a verify call through, COUNTS each
launch by (backend, head_dim, is_3d, q_rows), and (bounded, fail-open) CUDA-event TIMES a
sample of them. It does NOT change any output -- it is a pure observer (the wrapped call
is forwarded unchanged). With ``WIRBEL_VCENSUS`` unset this file is never imported and the
deployed stack is byte-identical.

WHY (the #459 reconciliation crux): #447 assumed FA_SLIDING routes ALL 30 head-256
sliding layers to FA2 so only 7 head-512 global layers keep Triton at verify (-> "Triton
verify surface = 1.27%"). #442's served census showed head-256 sliding DOES reach the
Triton 3D split-KV kernel at the M=8 verify. This census NAILS the count: how many
head-256 sliding layers route Triton (n256) vs FA2 (n256_fa2) per verify forward, using
the 7 head-512 global Triton launches as a per-forward CLOCK (global layers always keep
Triton). That n256 sizes the true Triton verify surface.

The two wrapped entry points:
  * ``vllm.v1.attention.ops.triton_unified_attention.kernel_unified_attention`` -- the
    Triton 3D split-KV kernel (deployed verify path, SPLITKV_VERIFY=1). Same deferred
    meta-path-finder idiom as the deployed splitkv_verify_patch / #442 bm4 injector, so
    the wrap lands the moment vLLM imports the kernel module.
  * ``vllm.v1.attention.backends.flash_attn.flash_attn_varlen_func`` -- the FA2 entry the
    FA_SLIDING-flipped head-256 layers call (best-effort; fail-open). Completeness check:
    n256_triton + n256_fa2 should equal the 30 sliding layers.

Verify regime = ``is_3d AND q_rows >= 2``: the M=8 verify runs 3D split-KV with q.shape[0]
== 8 (the M rows; splitkv_verify overrides max_seqlen_q=1 to select 3D but the 8 query
rows remain). M=1 drafter decode (q_rows==1) and prefill (is_3d False) are excluded.

Loaded ONLY via a temporary env-gated sitecustomize hook (apply->measure->revert by
verify_surface_census.py); NEVER submitted. NOT an HF Job, NOT a submission, NOT a launch.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.util
import json
import os
import sys
import threading

_TRITON_TARGET = "vllm.v1.attention.ops.triton_unified_attention"
_FA2_BACKEND_TARGET = "vllm.v1.attention.backends.flash_attn"
_RAW = os.environ.get("WIRBEL_VCENSUS", "").strip()
_OUT_BASE = os.environ.get("WIRBEL_VCENSUS_OUT", "/tmp/wirbel_verify_census.json")
# PER-PROCESS output path: vLLM serves from an API-server parent + an EngineCore worker
# child (sitecustomize runs in BOTH). A single shared path lets the model-less parent
# (counts=0) clobber the worker's real census. Suffix the PID so the driver can read every
# process's file and aggregate (the empty parent contributes 0).
def _pid_path(base: str) -> str:
    import pathlib
    p = pathlib.Path(base)
    return str(p.with_name(f"{p.stem}.{os.getpid()}{p.suffix}"))
_OUT_PATH = _pid_path(_OUT_BASE)
_TIME_ENABLED = os.environ.get("WIRBEL_VCENSUS_TIME", "1").strip() not in ("", "0", "false", "False")

_M_VERIFY = 8
_SEQ_RECORD_CAP = 2000        # full per-call sequence recorded up to this many verify calls
_TIME_SAMPLES_PER_BUCKET = 96  # bounded async CUDA-event samples per (backend,head)
# Write JSON early+often so data survives a SIGKILL of the worker (the M=8 verify may be
# graph-captured -> the wrapper only fires during the brief capture warmup, so the file
# must be flushed before the worker is torn down). Dense early, sparse once saturated.
def _flush_due(vc: int) -> bool:
    return (vc <= 512 and vc % 8 == 0) or (vc % 512 == 0)

_LOCK = threading.Lock()
_STATE = {
    "triton_calls": 0, "fa2_calls": 0, "verify_calls": 0,
    # counts[backend][head][is_3d][q_rows] = n
    "counts": {},
    # counts_by_m[backend][head][per_seq_M] = n  (per_seq_M = q_rows // num_seqs);
    # the batch-robust discriminant: verify=M8, drafter/decode=M1.
    "counts_by_m": {},
    # full ordered sequence of verify-regime calls (capped)
    "sequence": [],
    # per-bucket served CUDA-event us samples: key "backend/head/q_rows" -> [us,...]
    "time_samples": {},
    # pending (start_evt, end_evt, bucket) awaiting sync
    "_pending_events": [],
    "wrapped_triton": False, "wrapped_fa2": False,
    "fa2_wrap_error": None,
}


def _log(msg: str) -> None:
    print(f"[vcensus] {msg}", file=sys.stderr, flush=True)


def _bump(counts, backend, head, is_3d, q_rows):
    b = counts.setdefault(backend, {})
    h = b.setdefault(str(head), {})
    d = h.setdefault("3d" if is_3d else "2d", {})
    k = str(q_rows)
    d[k] = d.get(k, 0) + 1


def _bump_m(counts_by_m, backend, head, m):
    b = counts_by_m.setdefault(backend, {})
    h = b.setdefault(str(head), {})
    k = str(m)
    h[k] = h.get(k, 0) + 1


def _per_seq_m(q_rows, num_seqs):
    try:
        if num_seqs and num_seqs > 0:
            return q_rows // num_seqs
    except Exception:  # noqa: BLE001
        pass
    return q_rows


def _bucket_key(backend, head, q_rows):
    return f"{backend}/{head}/{q_rows}"


def _maybe_time(bucket_key):
    """Return (start_evt, end_evt) to bracket a launch, or (None, None). Bounded per
    bucket; fail-open (timing never breaks the served forward)."""
    if not _TIME_ENABLED:
        return None, None
    try:
        import torch
        have = len(_STATE["time_samples"].get(bucket_key, []))
        pend = sum(1 for _, _, bk in _STATE["_pending_events"] if bk == bucket_key)
        if have + pend >= _TIME_SAMPLES_PER_BUCKET:
            return None, None
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        return s, e
    except Exception:  # noqa: BLE001
        return None, None


def _flush_events():
    """Synchronize pending CUDA events and reduce to per-bucket us samples. Fail-open."""
    if not _STATE["_pending_events"]:
        return
    try:
        import torch
        torch.cuda.synchronize()
        for s, e, bk in _STATE["_pending_events"]:
            try:
                us = float(s.elapsed_time(e) * 1e3)  # ms -> us
                if us > 0:
                    _STATE["time_samples"].setdefault(bk, []).append(us)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        _STATE["_pending_events"] = []


def _summarize_times():
    import statistics
    out = {}
    for bk, xs in _STATE["time_samples"].items():
        if not xs:
            continue
        xs_sorted = sorted(xs)
        out[bk] = {
            "n": len(xs), "median_us": statistics.median(xs_sorted),
            "mean_us": sum(xs_sorted) / len(xs_sorted),
            "min_us": xs_sorted[0], "max_us": xs_sorted[-1],
        }
    return out


def _write(final=False):
    try:
        if final:
            _flush_events()
        payload = {
            "experiment": "verify_surface_census", "pr": 459, "student": "wirbel",
            "pid": os.getpid(), "out_base": _OUT_BASE,
            "triton_calls": _STATE["triton_calls"], "fa2_calls": _STATE["fa2_calls"],
            "verify_calls": _STATE["verify_calls"],
            "wrapped_triton": _STATE["wrapped_triton"], "wrapped_fa2": _STATE["wrapped_fa2"],
            "fa2_wrap_error": _STATE["fa2_wrap_error"],
            "time_enabled": _TIME_ENABLED,
            "counts": _STATE["counts"],
            "counts_by_m": _STATE["counts_by_m"],
            "served_per_call_us": _summarize_times(),
            "sequence_len": len(_STATE["sequence"]),
            "sequence_head": _STATE["sequence"][:120],
            "final": final,
        }
        with open(_OUT_PATH, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN census write failed: {exc!r}")


def _record(backend, head, is_3d, q_rows, num_seqs, tile, block_m, block_q, evt_pair):
    with _LOCK:
        _bump(_STATE["counts"], backend, head, is_3d, q_rows)
        m = _per_seq_m(q_rows, num_seqs)
        # Batch-robust verify gate: per-seq M>=2 (verify M=8) vs drafter/decode M=1.
        is_verify = bool(is_3d and m >= 2 and head in (256, 512))
        if head in (256, 512):
            _bump_m(_STATE["counts_by_m"], backend, head, m)
        if is_verify:
            _STATE["verify_calls"] += 1
            if len(_STATE["sequence"]) < _SEQ_RECORD_CAP:
                _STATE["sequence"].append(
                    {"backend": backend, "head": head, "is_3d": is_3d,
                     "q_rows": q_rows, "num_seqs": num_seqs, "m": m,
                     "tile": tile, "block_m": block_m, "block_q": block_q})
            if evt_pair[0] is not None:
                _STATE["_pending_events"].append(
                    (evt_pair[0], evt_pair[1], _bucket_key(backend, head, m)))
            if _flush_due(_STATE["verify_calls"]):
                _flush_events()
                _write(final=False)


# ---------------------------------------------------------------------------
# Triton kernel_unified_attention wrap (the decisive routing observer).
# ---------------------------------------------------------------------------
def _make_triton_wrapper(inner_kernel):
    class _CensusKernel:
        def __getitem__(self, grid):
            def _call(*args, **kwargs):
                # Deployed unified_attention launches all-kwargs with a tuple grid.
                rec = (not args) and isinstance(grid, tuple)
                s = e = None
                if rec:
                    try:
                        head = int(kwargs.get("HEAD_SIZE"))
                        is_3d = bool(kwargs.get("IS_3D"))
                        q = kwargs.get("query_ptr")
                        q_rows = int(q.shape[0])
                        num_seqs = int(kwargs.get("num_seqs"))
                        tile = int(kwargs.get("TILE_SIZE")) if kwargs.get("TILE_SIZE") is not None else None
                        block_m = kwargs.get("BLOCK_M")
                        block_q = kwargs.get("BLOCK_Q")
                        _STATE["triton_calls"] += 1
                        if is_3d and _per_seq_m(q_rows, num_seqs) >= 2 and head in (256, 512):
                            s, e = _maybe_time(_bucket_key("triton", head, _per_seq_m(q_rows, num_seqs)))
                    except Exception:  # noqa: BLE001 - never break the served launch
                        rec = False
                if rec and s is not None:
                    try:
                        s.record()
                        out = inner_kernel[grid](*args, **kwargs)
                        e.record()
                    except Exception:  # noqa: BLE001
                        s = e = None
                        out = inner_kernel[grid](*args, **kwargs)
                else:
                    out = inner_kernel[grid](*args, **kwargs)
                if rec:
                    try:
                        _record("triton", head, is_3d, q_rows, num_seqs, tile,
                                block_m, block_q, (s, e))
                    except Exception:  # noqa: BLE001
                        pass
                return out
            return _call

        def __getattr__(self, name):
            return getattr(inner_kernel, name)

    return _CensusKernel()


def _patch_triton(module):
    if _STATE["wrapped_triton"]:
        return
    kern = getattr(module, "kernel_unified_attention", None)
    if kern is None:
        _log("WARN: triton module has no kernel_unified_attention; not wrapped")
        return
    module.kernel_unified_attention = _make_triton_wrapper(kern)
    _STATE["wrapped_triton"] = True
    _log(f"WRAPPED {_TRITON_TARGET}.kernel_unified_attention (routing+timing census)")


# ---------------------------------------------------------------------------
# FA2 flash_attn_varlen_func wrap (completeness check; best-effort, fail-open).
# ---------------------------------------------------------------------------
def _make_fa2_wrapper(inner_fn):
    def _wrapped(*args, **kwargs):
        s = e = None
        head = q_rows = None
        num_seqs = 1
        try:
            q = kwargs.get("q", args[0] if args else None)
            if q is not None and hasattr(q, "shape") and q.ndim >= 2:
                head = int(q.shape[-1])
                q_rows = int(q.shape[0])
                # varlen passes cu_seqlens_q (len = num_seqs + 1); use it for per-seq M.
                cu = kwargs.get("cu_seqlens_q")
                try:
                    if cu is not None and hasattr(cu, "shape") and cu.shape[0] >= 2:
                        num_seqs = int(cu.shape[0]) - 1
                except Exception:  # noqa: BLE001
                    num_seqs = 1
                _STATE["fa2_calls"] += 1
                if _per_seq_m(q_rows, num_seqs) >= 2:
                    s, e = _maybe_time(_bucket_key("fa2", head, _per_seq_m(q_rows, num_seqs)))
        except Exception:  # noqa: BLE001
            head = None
        if s is not None:
            try:
                s.record()
                out = inner_fn(*args, **kwargs)
                e.record()
            except Exception:  # noqa: BLE001
                s = e = None
                out = inner_fn(*args, **kwargs)
        else:
            out = inner_fn(*args, **kwargs)
        if head is not None and q_rows is not None:
            try:
                # FA2 has no 3D/2D split; treat its sliding verify calls as "verify"
                # when per-seq M>=2 so n256_fa2 lands in the same per-forward accounting.
                _record("fa2", head, True, q_rows, num_seqs, None, None, None, (s, e))
            except Exception:  # noqa: BLE001
                pass
        return out
    return _wrapped


def _patch_fa2(module):
    if _STATE["wrapped_fa2"]:
        return
    fn = getattr(module, "flash_attn_varlen_func", None)
    if fn is None or not callable(fn):
        _STATE["fa2_wrap_error"] = "no flash_attn_varlen_func symbol in backend module"
        return
    module.flash_attn_varlen_func = _make_fa2_wrapper(fn)
    _STATE["wrapped_fa2"] = True
    _log(f"WRAPPED {_FA2_BACKEND_TARGET}.flash_attn_varlen_func (FA2 completeness census)")


# ---------------------------------------------------------------------------
# Deferred meta-path finders (land before ONEGRAPH capture).
# ---------------------------------------------------------------------------
def _install_finder(target, patch_fn):
    if target in sys.modules:
        try:
            patch_fn(sys.modules[target])
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN in-place patch {target} failed: {exc!r}")
        return

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                patch_fn(module)
            except Exception as exc:  # noqa: BLE001
                _log(f"WARN patch-on-exec {target} failed: {exc!r}")

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self._busy = False

        def find_spec(self, fullname, path=None, target_=None):
            if fullname != target or self._busy:
                return None
            self._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            except Exception:  # noqa: BLE001
                return None
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PatchingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())


def _install():
    _install_finder(_TRITON_TARGET, _patch_triton)
    try:
        _install_finder(_FA2_BACKEND_TARGET, _patch_fa2)
    except Exception as exc:  # noqa: BLE001
        _STATE["fa2_wrap_error"] = repr(exc)[:160]
    atexit.register(lambda: _write(final=True))
    _write(final=False)
    _log(f"installed pid={os.getpid()} (out={_OUT_PATH}, time={_TIME_ENABLED}); "
         f"finders for triton kernel + FA2 backend")


if _RAW and _RAW not in ("0", "", "false", "False"):
    _install()
elif _RAW:
    _log(f"no-op: WIRBEL_VCENSUS={_RAW!r} (disabled)")
