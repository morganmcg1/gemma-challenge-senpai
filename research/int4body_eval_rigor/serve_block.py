"""Persistent local serve of the shipped int4_g128_lmhead submission.

Stands up the EXACT shipped manifest config (vllm 0.22.0, MAX_MODEL_LEN=4096,
MAX_NUM_BATCHED_TOKENS=512) via the proven LocalServer path, then blocks so the
endpoint stays up while we drive AIME + gpqa_diamond evals against --base-url.
Writes SERVER_READY when /v1/models is live. Numerics-neutral local shims only
(CUDA_VISIBLE_DEVICES=0, native sampler) — does not touch the served distribution.
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

WD = ROOT / "research/int4body_eval_rigor"


def main() -> None:
    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)
    sub = ROOT / "submissions/int4_g128_lmhead"
    manifest = harness.load_manifest(sub)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[serve] server_python={server_python}", flush=True)
    # Local QUALITY-eval override only: bump MAX_MODEL_LEN so gpqa_diamond can be
    # scored at the truncation-clean max_tokens=4096 basis the #515 base gate was
    # set on (bars_verdict #614: 6144 model_len / 4096 max_tokens; <=2048 depresses
    # GPQA 0.07-0.14). This changes ONLY sequence capacity -- weights, kernels,
    # dtype, sampler and per-token distribution are untouched, so greedy identity
    # and int4-body quality are preserved. The shipped submission/manifest (4096)
    # is NOT modified; this is an analysis-only serve knob.
    extra_env = {}
    override_mml = os.environ.get("OVERRIDE_MAX_MODEL_LEN")
    if override_mml:
        extra_env["MAX_MODEL_LEN"] = override_mml
        print(f"[serve] OVERRIDE MAX_MODEL_LEN={override_mml} (quality-eval basis)", flush=True)
    srv = harness.LocalServer(
        sub,
        server_python=server_python,
        port=8000,
        startup_timeout_s=1800,
        log_path=WD / "out" / "serve.log",
        extra_env=extra_env,
    )
    srv.__enter__()
    print(f"[serve] READY base_url={srv.base_url} model={srv.served_model_name}", flush=True)
    (WD / "SERVER_READY").write_text(f"{srv.base_url}\n{srv.served_model_name}\n")
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        srv.__exit__(None, None, None)


if __name__ == "__main__":
    main()
