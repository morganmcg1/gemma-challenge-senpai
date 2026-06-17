#!/usr/bin/env python3
"""PR #557 Stage-1 — global_fa load-failure probe. LOCAL, NO FIRE.

Starts the STOCK int4 base serve with VLLM_ATTENTION_BACKEND=FLASH_ATTN (a stock
GLOBAL override that bypasses dev307's forced TRITON_ATTN). The hypothesis: a global
FlashAttention CANNOT serve this model because FA's kernel rejects the head_dim=512
full layers (head_size <= 256 limit) — so a stock global override is NOT the fix and
the per-reduction-path lever (surgical_attn) is the real recovery. Writes
global_fa_loadtest.json = {"load_failed": bool, "reason": str, "build": ...}.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_arm  # noqa: E402

HERE = run_arm.HERE
REJECT_PAT = re.compile(
    r"(head[_ ]?size|head_dim|512|not support|unsupported|invalid|assert|ValueError|"
    r"RuntimeError|FlashAttention|no.*backend|Cannot use)", re.IGNORECASE)


def main() -> int:
    run_arm.wait_gpu_free()
    log = HERE / "server_global_fa_loadtest.log"
    proc = run_arm.start_server("global_fa", log)
    load_failed = None
    reason = ""
    try:
        try:
            run_arm.wait_ready(proc, timeout_s=300)
            # If it somehow came up, the override did NOT fail — record that honestly.
            load_failed = False
            reason = "server reached /v1/models READY under global FLASH_ATTN (unexpected)"
        except RuntimeError as exc:
            load_failed = True
            reason = str(exc)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            pass
    # Pull the FA-rejection line out of the server log for the cause string.
    reject_lines = []
    try:
        for ln in log.read_text(errors="replace").splitlines():
            if ("FLASH" in ln.upper() or "TRITON" in ln.upper() or "head" in ln.lower()
                    or "Error" in ln or "Traceback" in ln) and REJECT_PAT.search(ln):
                reject_lines.append(ln.strip())
    except Exception:
        pass
    out = {
        "arm": "global_fa",
        "build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "env": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        "load_failed": load_failed,
        "server_exit_code": proc.returncode,
        "reason": reason,
        "reject_log_lines": reject_lines[-12:],
        "log": str(log),
    }
    (HERE / "global_fa_loadtest.json").write_text(json.dumps(out, indent=2))
    print(f"[global_fa] load_failed={load_failed} exit={proc.returncode} reason={reason!r}")
    for ln in reject_lines[-12:]:
        print("   ", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
