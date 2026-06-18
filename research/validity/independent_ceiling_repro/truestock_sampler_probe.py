#!/usr/bin/env python
"""Empirically confirm the TRUE-STOCK sampler default is unservable in THIS container.

The clean-room K-sweep (clean_room_kceil.py) forces VLLM_USE_FLASHINFER_SAMPLER=0
(the documented repo-wide container shim, paths.default_native_sampler). A reviewer
may object that the sampler is one of the three regime knobs land #664 is pinning,
so pinning it = not "stock". This probe substantiates that the native sampler is
FORCED by the container, not a regime choice: it boots the canonical
int4_mtp_batchinv K=6 stack with the sampler env var UNSET (true vLLM default,
which selects the flashinfer sampler since flashinfer-python 0.6.12 is installed)
and shows the engine dies at memory-profiling on the cuRAND JIT, before any token.

ANALYSIS-ONLY, no submission. ~90-120s (crashes early, before graph capture).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

# True stock: ensure the parent env does NOT pin the sampler so vLLM auto-selects
# its real default (flashinfer, because flashinfer-python is installed).
os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)

from scripts.local_validation import harness, paths  # noqa: E402

DEV307 = Path("/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python")
SUB = ROOT / "submissions" / "int4_mtp_batchinv"
LOG = ROOT / "research" / "validity" / "independent_ceiling_repro" / "run" / "_truestock_probe.boot"

note = paths.normalize_cuda_visible_devices()
if note:
    print(f"[gpu] {note}", flush=True)

print("[probe] booting int4_mtp_batchinv K=6 with VLLM_USE_FLASHINFER_SAMPLER UNSET "
      "(true stock -> flashinfer sampler)", flush=True)
served = False
err = None
try:
    srv = harness.LocalServer(
        SUB, server_python=DEV307, port=8000, log_path=LOG, startup_timeout_s=420,
        extra_env={"NUM_SPECULATIVE_TOKENS": "6", "DRAFTER_MODEL": "/tmp/qat-assistant"})
    srv.__enter__()
    served = True
    print("[probe] UNEXPECTED: true-stock served OK (no crash)", flush=True)
    try:
        srv.__exit__(None, None, None)
    except Exception:
        pass
except Exception as exc:
    err = str(exc)
    print(f"[probe] true-stock boot FAILED: {err[:200]}", flush=True)

# audit the boot log for the documented cuRAND/flashinfer JIT failure signature
txt = LOG.read_text(errors="replace") if LOG.exists() else ""
curand = "curand.h" in txt
ninja = "Ninja build failed" in txt or "RuntimeError: Error building extension" in txt
fi = "flashinfer" in txt.lower()
print(f"[probe] served={served} curand_h_missing={curand} ninja_build_failed={ninja} "
      f"flashinfer_mentioned={fi}", flush=True)
# surface the most relevant crash lines
for line in txt.splitlines():
    low = line.lower()
    if any(k in low for k in ("curand.h", "ninja build failed", "error building extension",
                              "no such file", "flashinfer", "memory profil")):
        print("   |", line[:200], flush=True)
print("VERDICT:",
      "TRUE_STOCK_CRASHES_native_shim_FORCED" if (not served and curand)
      else ("TRUE_STOCK_SERVES_shim_is_noop" if served else "CRASH_unconfirmed_signature"),
      flush=True)
