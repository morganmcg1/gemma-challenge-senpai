#!/usr/bin/env python
"""Serve the int4_g128_lmhead + MTP-K7 spec config (fern #597) for the #605 quality panel.

Persistent foreground server (run in background, kill via SIGTERM to tear down cleanly).
Reuses scripts.local_validation.harness so the serve path is the official submission
serve.py (int4_mtp_batchinv: MTP drafter + attn-group patch + sitecustomize).

Config = fern #597 `int4g128_k7_bi1_n16` EXCEPT the two documented guard changes:
  MAX_MODEL_LEN 4096 -> 6144 (land #598), and MAX_NUM_SEQS 1 -> N for eval concurrency.
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

# PR #612 engine directive: serve on vLLM dev307 (0.22.1rc1.dev307+g3e8afdf78), NOT
# the submission manifest's pinned 0.22.0 (which #547 found craters MMLU on this int4
# model). The submission's served files are UNCHANGED; only the harness venv selection
# is overridden here (research-side). These deps hash to the already-built cached venv
# /tmp/senpai-venvs/a341b8bdf5ec1fe0 (no rebuild). NOTE: stark #605 (the 0.4141 GPQA
# baseline) ran on 0.22.0; the dev307+3072 replicate arm bridges the engine gap.
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
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--batch-invariant", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--engine", choices=["dev307", "manifest"], default="dev307",
                    help="dev307 (PR #612 directive) or the submission manifest pin (0.22.0).")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    engine_deps = DEV307_DEPS if args.engine == "dev307" else manifest["dependencies"]
    server_python = harness.ensure_server_venv(engine_deps)
    print(f"[serve] engine={args.engine} server_python={server_python}", flush=True)

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
    log_path = HERE / "serve.log"

    with harness.LocalServer(
        SUBMISSION, server_python=server_python, port=args.port,
        log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        ready_file.write_text(f"{srv.base_url}\n{srv.served_model_name}\n")
        print(f"READY base_url={srv.base_url} model={srv.served_model_name}", flush=True)
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
