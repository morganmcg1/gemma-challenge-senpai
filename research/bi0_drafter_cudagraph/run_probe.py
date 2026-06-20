"""PR #789 — drive the bi0 control server with CGPROBE=1 and a short decode to
capture the drafter per-pass CUDA-graph dispatch mode (eager NONE vs PIECEWISE).

LOCAL A10G probe only. No HF Job.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_surgattn"
SERVER_PY = Path("/senpai-run/home/student-stark/.venvs/vllm022/bin/python")
OUT = Path(__file__).resolve().parent / "runs" / "control_probe"


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "server.log"
    extra_env = {
        "CGPROBE": "1",
        "CGPROBE_DIR": str(Path(__file__).resolve().parent),
        "CGPROBE_WARMUP_SKIP": "60",
        "CGPROBE_RAW_COUNT": "40",
        "CGPROBE_REPORT_EVERY": "200",
        "DISABLE_LOG_STATS": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=8000,
        log_path=log_path, extra_env=extra_env, startup_timeout_s=1200,
    ) as srv:
        print(f"[probe] ready in {time.time()-t0:.0f}s; driving decode", flush=True)
        summary = harness.capture_decode(
            SERVER_PY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=OUT / "decode_outputs.jsonl", summary_file=OUT / "decode_summary.json",
            num_prompts=8, output_len=192, timeout_s=2400,
        )
        print(f"[probe] decode summary: {summary}", flush=True)
    print("[probe] server stopped; see server.log for [cgprobe] lines", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
