#!/usr/bin/env python
"""PR #656 — serve one arm of the AR-vs-spec GPQA 10-seed contrast.

Single-variable contrast: same int4_g128_lmhead body+head, same dev307 engine,
gb6144, BI=1, max_num_seqs=16. The ONLY difference between arms is speculation:

  --k 6  -> SPEC arm (as-served Option-B = shipped int4_mtp_batchinv manifest,
            NUM_SPECULATIVE_TOKENS=6 + MTP drafter /tmp/qat-assistant).
  --k 0  -> AR  arm (NUM_SPECULATIVE_TOKENS=0 -> serve.py serves plain int4 M=1,
            speculative_config=None, drafter OFF; the only removed variable is
            speculation, per int4_mtp_batchinv/serve.py docstring).

Reuses the official submission serve path (int4_mtp_batchinv: MTP drafter +
attn-group patch + sitecustomize) through scripts.local_validation.harness, and
the PR #612 dev307 engine pin (same cached venv hash, no rebuild).
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
HERE = Path(__file__).resolve().parent

# PR #612 dev307 engine pin (matches the already-built cached venv; no rebuild).
DEV307_DEPS = [
    "MarkupSafe==3.0.3",
    "https://wheels.vllm.ai/3e8afdf78598afc8be999a6a049be3a5fe182e48/"
    "vllm-0.22.1rc1.dev307%2Bg3e8afdf78.cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
    "jinja2==3.1.6",
    "transformers==5.9.0",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-id", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--drafter", default="/tmp/qat-assistant")
    ap.add_argument("--k", type=int, required=True,
                    help="6 = SPEC (shipped manifest); 0 = AR (drafter OFF).")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--batch-invariant", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    server_python = harness.ensure_server_venv(DEV307_DEPS)
    arm = "spec_k%d" % args.k if args.k > 0 else "ar_m1"
    print(f"[serve] arm={arm} engine=dev307 server_python={server_python}", flush=True)

    extra_env = {
        "MODEL_ID": args.model_id,
        "DRAFTER_MODEL": args.drafter,
        "NUM_SPECULATIVE_TOKENS": str(args.k),
        "VLLM_BATCH_INVARIANT": str(args.batch_invariant),
        "MAX_MODEL_LEN": str(args.max_model_len),
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": str(args.max_num_batched_tokens),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }

    stop = {"flag": False}

    def _handler(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    ready_file = HERE / "serve.ready"
    ready_file.unlink(missing_ok=True)
    log_path = HERE / f"serve_{arm}.log"

    with harness.LocalServer(
        SUBMISSION, server_python=server_python, port=args.port,
        log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        ready_file.write_text(f"{srv.base_url}\n{srv.served_model_name}\n{arm}\n")
        print(f"READY arm={arm} base_url={srv.base_url} model={srv.served_model_name}", flush=True)
        print(f"[serve] config: {extra_env}", flush=True)
        while not stop["flag"]:
            if srv.proc is not None and srv.proc.poll() is not None:
                print(f"[serve] server process exited rc={srv.proc.returncode}", flush=True)
                ready_file.unlink(missing_ok=True)
                return 1
            time.sleep(2)
    ready_file.unlink(missing_ok=True)
    print("[serve] stopped cleanly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
