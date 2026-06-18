#!/usr/bin/env python
"""Serve one arm of the spec-break quality-materiality census (PR #682, wirbel).

ANALYSIS-ONLY, LOCAL single-A10G. No weights changed, NO HF job, NO submission
file change. official_tps=0.

Body-matched AR-vs-SPEC isolation. BOTH arms serve the SAME int4 body
(int4_g128_lmhead, the strict-#319 ship anchor / PR #4) with the SAME engine
(dev307), BI=1, native sampler. The ONLY removed variable is speculation:

  --arm spec :  NUM_SPECULATIVE_TOKENS=K (default 6, the wirbel #671 ~170 band
                drafter /tmp/qat-assistant).
  --arm ar   :  NUM_SPECULATIVE_TOKENS=0 -> submission serve.py serves plain int4
                M=1 AR greedy, "the exact-greedy reference the strict-#319 gate
                compares against" (submissions/int4_mtp_batchinv/serve.py:12-14).

So the spec-vs-AR quality delta isolates the int4-Marlin verify-width break
(kanna #673) from the int4 body's own pre-existing quality gap.

Persistent foreground server (run in background; SIGTERM to tear down). Reuses
scripts.local_validation.harness so the serve path is the official submission
serve.py. Mirrors research/validity/int4_mtp_spec_quality_panel/serve_spec.py
config EXCEPT: ready/log land in THIS card's dir, and --arm/--k drive spec on/off.
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

# vLLM dev307 (0.22.1rc1.dev307+g3e8afdf78): the engine wirbel #671's ~170 band
# and serve_spec.py both ran on. NOT the manifest 0.22.0 pin (#547 found 0.22.0
# craters MMLU on this int4 model). These deps hash to the already-built cached
# venv /tmp/senpai-venvs/a341b8bdf5ec1fe0 (no rebuild).
DEV307_DEPS = [
    "MarkupSafe==3.0.3",
    "https://wheels.vllm.ai/3e8afdf78598afc8be999a6a049be3a5fe182e48/"
    "vllm-0.22.1rc1.dev307%2Bg3e8afdf78.cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
    "jinja2==3.1.6",
    "transformers==5.9.0",
]

BODY = "/workspace/gemma_build/int4_g128_lmhead"  # strict-#319 ship anchor (PR #4)
DRAFTER = "/tmp/qat-assistant"  # wirbel #671 ~170-band QAT-matched MTP drafter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ar", "spec"], required=True)
    ap.add_argument("--k", type=int, default=None,
                    help="num_speculative_tokens; default 6 for spec, 0 for ar")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-id", default=BODY)
    ap.add_argument("--drafter", default=DRAFTER)
    # 8192 = the #515 quality-panel context (max_tokens=6144 gb6144 budget +
    # up to ~2048 prompt). max_model_len=6144 collides with max_tokens=6144 for
    # any non-trivial prompt -> vLLM 400 Bad Request on every request.
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--batch-invariant", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = ap.parse_args()

    k = args.k if args.k is not None else (6 if args.arm == "spec" else 0)
    if args.arm == "ar" and k != 0:
        raise SystemExit(f"--arm ar requires k=0 (got {k})")

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    server_python = harness.ensure_server_venv(DEV307_DEPS)
    print(f"[serve] arm={args.arm} k={k} body={args.model_id} server_python={server_python}",
          flush=True)

    extra_env = {
        "MODEL_ID": args.model_id,
        "DRAFTER_MODEL": args.drafter,
        "NUM_SPECULATIVE_TOKENS": str(k),
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

    ready_file = HERE / f"serve_{args.arm}.ready"
    ready_file.unlink(missing_ok=True)
    log_path = HERE / f"serve_{args.arm}.log"

    with harness.LocalServer(
        SUBMISSION, server_python=server_python, port=args.port,
        log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        ready_file.write_text(f"{srv.base_url}\n{srv.served_model_name}\n")
        print(f"READY arm={args.arm} base_url={srv.base_url} model={srv.served_model_name}",
              flush=True)
        print(f"[serve] config: {extra_env}", flush=True)
        while not stop["flag"]:
            if srv.proc is not None and srv.proc.poll() is not None:
                print(f"[serve] server process exited rc={srv.proc.returncode}", flush=True)
                ready_file.unlink(missing_ok=True)
                return 1
            time.sleep(2)
    ready_file.unlink(missing_ok=True)
    print(f"[serve] arm={args.arm} stopped cleanly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
