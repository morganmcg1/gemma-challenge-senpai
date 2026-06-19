#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #755 lawine -- SERVED-path num_splits probe (reconcile #747 <-> #752).

Instruction 1: instrument the LIVE served vLLM 0.22.0 stack to log, per attention
forward, the actual ``num_splits`` (= ``num_segments``, the split-KV reduction
fan-out) the kernel picks for the M=K+1 spec-VERIFY batch vs the M=1 AR/decode
batch, and whether ``is_batch_invariant`` (the ``VLLM_BATCH_INVARIANT=1`` gate) is
actually live in the worker process. This is the hinge that explains why land #743 /
wirbel #747's offline ``enforce_eager`` "BI=1 -> already byte-exact" does (or does
NOT) transfer to the served 512-token run (#752 strict 24/128).

MECHANISM PINNED BY SOURCE READ (vllm 0.22.0):
  * ``triton_unified_attention.py:34``  ``is_batch_invariant = envs.VLLM_BATCH_INVARIANT``
    -- a module global frozen ONCE at import time.
  * ``enable_batch_invariant_mode()`` (batch_invariant.py:905) patches ONLY the
    matmul family + softmax/bmm; it NEVER re-sets ``is_batch_invariant`` in the
    attention kernel. So the attention split gate is governed solely by that
    import-time frozen global.
  * ``unified_attention`` use_3d gate (triton_unified_attention.py:923-931):
        use_3d = not (... or max_seqlen_q > 1 or num_seqs > seq_threshold_3D
                      or is_batch_invariant)
        num_segments = num_par_softmax_segments if use_3d else 1
    => under is_batch_invariant True, use_3d is False for EVERY shape -> nseg=1
       for both M=1 decode and M=K+1 verify (the wirbel #747 claim). Under False,
       M=1 decode (num_seqs<=threshold) takes 3D nseg>1 while M=K+1 verify takes
       2D nseg=1 -> a reduction-order split (the #752 24/128 candidate).

We wrap ``vllm.v1.attention.backends.triton_attn.unified_attention`` (the served
TRITON_ATTN entry; called all-kwargs at triton_attn.py:638, NO torch-custom-op
wrapper) and recompute the kernel's own use_3d / num_segments from the passed args
+ the live module global ``is_batch_invariant`` -- a read-only mirror of the real
decision -- then call the real kernel unchanged. Aggregated by
(max_seqlen_q, is_local_window, use_3d, num_segments).

ROBUSTNESS (why the first #755 probe read calls=0): vLLM v1 serves on a multi-proc
tree (api_server + EngineCore worker). sitecustomize runs in EVERY process, so the
front-end ALSO installs this probe but never runs attention; its atexit dump
(calls=0) was clobbering the worker's data on a single shared path. Fixes here:
  (1) dump to a PER-PID path ``<out>.pid<pid>.json`` (no cross-proc clobber); the
      runner reads all pid files and keeps the one with the most attention calls.
  (2) at the moment ``triton_attn`` loads, log to STDERR (-> server.log) the pid +
      the live ``is_batch_invariant`` -- so the worker's value is captured even if
      a SIGKILL at serve teardown skips atexit.

NO numerics change -- pure observation. LOCAL A10G only.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from collections import defaultdict

_LOCK = threading.Lock()
_STATE: dict[str, object] = {
    "installed": False,
    "out_path": None,
    "flush_every": 500,
    "n_calls": 0,
    # key "msq=<>|local=<>|use3d=<>|nseg=<>" -> aggregate
    "buckets": defaultdict(lambda: {
        "count": 0, "seqlen_k_min": None, "seqlen_k_max": None,
        "num_seqs_min": None, "num_seqs_max": None,
    }),
    "is_batch_invariant_seen": None,
    "logged_buckets": set(),
}


def _log(msg: str) -> None:
    try:
        print(f"[pr755-probe pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _pid_path() -> str | None:
    out = _STATE["out_path"]
    if not out:
        return None
    return f"{out}.pid{os.getpid()}.json"


def _upd_range(d: dict, key: str, val) -> None:
    lo, hi = d[key + "_min"], d[key + "_max"]
    d[key + "_min"] = val if lo is None else min(lo, val)
    d[key + "_max"] = val if hi is None else max(hi, val)


def _dump() -> None:
    out = _pid_path()
    if not out:
        return
    with _LOCK:
        buckets = {k: dict(v) for k, v in _STATE["buckets"].items()}
        payload = {
            "phase": "served_numsplits_probe",
            "pr": 755,
            "pid": os.getpid(),
            "is_batch_invariant_live": _STATE["is_batch_invariant_seen"],
            "n_attention_calls": _STATE["n_calls"],
            "buckets": buckets,
            "legend": "key = msq(max_seqlen_q: 1=decode, K+1=verify, >K+1=prefill)|"
                      "local(sliding-window layer)|use3d|nseg(num_splits the kernel runs)",
        }
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, out)


def _make_wrapper(real_fn, kernel_mod):
    def wrapper(*args, **kwargs):
        try:
            _observe(kwargs, kernel_mod)
        except Exception:  # never let the probe perturb serving
            pass
        return real_fn(*args, **kwargs)
    wrapper.__name__ = getattr(real_fn, "__name__", "unified_attention")
    wrapper._numsplits_probe = True  # idempotency marker
    return wrapper


def _observe(kwargs: dict, kernel_mod) -> None:
    max_seqlen_q = kwargs.get("max_seqlen_q")
    seqused_k = kwargs.get("seqused_k")
    max_seqlen_k = kwargs.get("max_seqlen_k")
    window_size = kwargs.get("window_size")
    seq_threshold_3D = kwargs.get("seq_threshold_3D")
    nseg_cfg = kwargs.get("num_par_softmax_segments")
    so = kwargs.get("softmax_segm_output")
    sm = kwargs.get("softmax_segm_max")
    se = kwargs.get("softmax_segm_expsum")
    if max_seqlen_q is None or seqused_k is None:
        return
    try:
        num_seqs = int(seqused_k.shape[0])
    except Exception:
        num_seqs = len(seqused_k)
    is_bi = bool(getattr(kernel_mod, "is_batch_invariant", False))
    # exact replica of triton_unified_attention.unified_attention use_3d gate (L923-931)
    use_3d = not (
        seq_threshold_3D is None
        or nseg_cfg is None
        or so is None
        or sm is None
        or se is None
        or int(max_seqlen_q) > 1
        or (seq_threshold_3D is not None and num_seqs > seq_threshold_3D)
        or is_bi
    )
    num_segments = int(nseg_cfg) if (use_3d and nseg_cfg is not None) else 1
    # sliding-window layers pass window_size[0] >= 0; global layers pass -1
    try:
        is_local = bool(window_size is not None and int(window_size[0]) >= 0)
    except Exception:
        is_local = False
    key = f"msq={int(max_seqlen_q)}|local={int(is_local)}|use3d={int(use_3d)}|nseg={num_segments}"
    new_bucket = False
    with _LOCK:
        _STATE["is_batch_invariant_seen"] = is_bi
        b = _STATE["buckets"][key]
        if b["count"] == 0:
            new_bucket = True
        b["count"] += 1
        if max_seqlen_k is not None:
            _upd_range(b, "seqlen_k", int(max_seqlen_k))
        _upd_range(b, "num_seqs", num_seqs)
        _STATE["n_calls"] += 1
        need_dump = (_STATE["n_calls"] % int(_STATE["flush_every"]) == 0)
    if new_bucket:
        _log(f"is_batch_invariant={is_bi} new attn bucket: {key} "
             f"(seqlen_k={max_seqlen_k}, num_seqs={num_seqs})")
    if need_dump:
        _dump()


def _patch_module(triton_attn_mod) -> bool:
    real = getattr(triton_attn_mod, "unified_attention", None)
    if real is None or getattr(real, "_numsplits_probe", False):
        return False
    # the kernel module that owns the is_batch_invariant global + the gate constants
    import vllm.v1.attention.ops.triton_unified_attention as kernel_mod
    triton_attn_mod.unified_attention = _make_wrapper(real, kernel_mod)
    _STATE["is_batch_invariant_seen"] = bool(getattr(kernel_mod, "is_batch_invariant", False))
    _log(f"installed unified_attention wrapper; "
         f"triton_unified_attention.is_batch_invariant={_STATE['is_batch_invariant_seen']}")
    _dump()  # persist the is_batch_invariant value immediately (survives SIGKILL)
    return True


def install(out_path: str | None = None, flush_every: int = 500) -> None:
    """Install the probe. Patches triton_attn now if imported, else on first import."""
    if _STATE["installed"]:
        return
    _STATE["installed"] = True
    _STATE["out_path"] = out_path or os.environ.get("SENPAI_NUMSPLITS_PROBE")
    _STATE["flush_every"] = max(1, int(flush_every))
    atexit.register(_dump)
    _log(f"install() out={_STATE['out_path']}")

    target = "vllm.v1.attention.backends.triton_attn"
    if target in sys.modules:
        _patch_module(sys.modules[target])
        return

    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _Finder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.v1.attention.backends.triton_attn":
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                try:
                    _patch_module(module)
                except Exception:
                    _log("patch failed")

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
