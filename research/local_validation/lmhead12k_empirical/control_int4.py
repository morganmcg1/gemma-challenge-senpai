#!/usr/bin/env python
"""Stage 2 CONTROL: unpruned int4 served-vs-served greedy gate + matched TPS.

Mirrors the pruned authoritative gate's methodology on the UNPRUNED int4 base to
establish the served-path false-divergence FLOOR: serve the int4 checkpoint twice
through the SAME plain vLLM api_server path (cudagraph ON, single-stream), capture
both, and compare with the official greedy verifier. Two SEPARATE server instances
(A=reference, B=candidate) mirror the pruned gate (gen_greedy_reference serve #1 +
validate_submission serve #2), so the control is apples-to-apples. Expectation:
~128/128 GREEDY_IDENTICAL -> the served-vs-served method does NOT manufacture
divergence (unlike the batched-offline reference, which gave 21/128). Also probes
single-stream decode TPS with the SAME harness.probe_tps validate_submission uses
for the pruned model, so the isolated prune TPS delta is measured identically.

The lmhead12k plugin is installed in the server venv; it must be DISABLED here
(VLLM_PLUGINS="") or it would register the custom scatter class for the unpruned
model and fail to find kept_ids.json. Run from the repo root with the server venv:
    CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python \
        research/local_validation/lmhead12k_empirical/control_int4.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

INT4 = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
BASELINE_SUB = ROOT / "submissions" / "vllm_baseline"
VENVPY = Path("/tmp/server-venv/bin/python")
OUT = ROOT / "research" / "local_validation" / "lmhead12k_empirical" / "control_int4_served"


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def serve_and_capture(tag: str, out_file: Path, summary_file: Path, *, want_tps: bool):
    env = {"MODEL_ID": INT4, "VLLM_PLUGINS": "", "SERVED_MODEL_NAME": paths.DEFAULT_SERVED_NAME}
    tps = None
    with harness.LocalServer(
        BASELINE_SUB, server_python=VENVPY, port=8000,
        log_path=OUT / f"serve_{tag}.log", extra_env=env,
    ) as srv:
        print(f"[control] serve {tag} ready model_id={srv.model_id} {ts()}", flush=True)
        harness.capture_decode(
            VENVPY, base_url=srv.base_url, model=srv.served_model_name,
            out_file=out_file, summary_file=summary_file,
        )
        if want_tps:
            tps = harness.probe_tps(srv.base_url, srv.served_model_name, decode_tokens=256)
            print(f"[control] unpruned TPS(single-stream)={tps['decode_tps_single_stream']:.2f} {ts()}", flush=True)
    return tps


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[control] {note}", flush=True)

    print(f"=== CONTROL int4 served-vs-served start {ts()} ===", flush=True)
    # serve #1 -> A (reference); also grab the matched single-stream TPS here.
    tps = serve_and_capture("A", OUT / "int4_A.jsonl", OUT / "int4_A_summary.json", want_tps=True)
    # serve #2 -> B (candidate); fresh server instance, same config.
    serve_and_capture("B", OUT / "int4_B.jsonl", OUT / "int4_B_summary.json", want_tps=False)

    report = greedy_gate.compare(OUT / "int4_A.jsonl", OUT / "int4_B.jsonl")
    onset = greedy_gate.onset_summary(report)
    result = {
        "control_verdict": report.verdict,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "num_prompts_compared": report.num_prompts_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "divergence_onset": onset,
        "unpruned_int4_tps_single_stream": (tps or {}).get("decode_tps_single_stream"),
        "tps_detail": tps,
        "model_id": INT4,
        "method": "two separate int4 serves (vllm_baseline, VLLM_PLUGINS='', cudagraph ON), "
                  "official greedy verifier A-vs-B",
        "created_at": ts(),
    }
    (OUT / "control_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("=== CONTROL RESULT ===", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
