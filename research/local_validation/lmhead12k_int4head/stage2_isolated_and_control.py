#!/usr/bin/env python
"""Stage 2 (PR #36): unpruned-int4 control + same-session isolated head-dtype TPS
delta + bandwidth-model refit for the int4-head submission.

Three checkpoints, ONE serving stack. The installed lmhead12k plugin is
checkpoint-driven (it reads the head quant scheme from each checkpoint's
``quantization_config``), so the *only* variable across the three TPS points is
the ``lm_head`` dtype / per-token byte read:

    unpruned-int4 base : bf16 FULL 262k head (1.342 GB)  control + bandwidth anchor
    bf16-head pruned   : bf16 12k  head      (0.0629 GB) PR #14 rung, same-session
    int4-head pruned   : int4 12k  head      (0.0162 GB) THIS PR (TPS from stage1)

Phase 1 CONTROL (mirrors PR #14 ``control_int4.py``): serve the UNPRUNED int4 base
twice through plain vLLM (``VLLM_PLUGINS=''`` so the scatter class is NOT used),
capture both, official greedy verifier A-vs-B. Expect GREEDY_IDENTICAL 128/128 ->
the served-vs-served method has a zero false-divergence FLOOR (so the int4-head
128/128 self-consistency gate is meaningful, not rigged). Also yields the
same-session unpruned-int4 single-stream TPS (bandwidth ``B+1.342`` anchor).

Phase 2 ISOLATED: serve the bf16-head pruned rung (``lmhead12k_empirical``) through
the SAME installed plugin (head loads bf16 because that checkpoint ignores
``lm_head``) and probe the SAME ``harness.probe_tps``. This is the clean single-
variable partner for the int4-head TPS: identical body, identical kept_ids,
identical plugin/stack -- ONLY the head dtype (bf16 62.9 MB vs int4 16.2 MB) moves.

Phase 3 ANALYSIS: read the int4-head TPS from stage1 evidence, compute the isolated
head-dtype delta, and refit the lm_head-bytes <-> TPS bandwidth model on the three
same-session points; write the analysis + a consolidated ``head_bytes.json``.

Run from repo root with the server venv (GPU must be free):
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /tmp/server-venv/bin/python \
        research/local_validation/lmhead12k_int4head/stage2_isolated_and_control.py
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
EMPIRICAL_SUB = ROOT / "submissions" / "lmhead12k_empirical"
EMPIRICAL_CKPT = "/workspace/gemma_build/lmhead12k_empirical"
VENVPY = Path("/tmp/server-venv/bin/python")

BASE = ROOT / "research" / "local_validation" / "lmhead12k_int4head"
CTRL_OUT = BASE / "control_int4_served"
ISO_OUT = BASE / "isolated_tps"
STAGE1_EVID = BASE / "stage1_evidence" / "evidence.json"
SERVED_HEAD_BYTES = BASE / "smoke" / "served_head_bytes.json"

HIDDEN = 2560
FULL_VOCAB = 262144
KEPT = 12288
# Per-token lm_head weight read (GB) at M=1 decode, by checkpoint.
HEAD_GB = {
    "unpruned_int4": FULL_VOCAB * HIDDEN * 2 / 1e9,   # bf16 full 262k head  = 1.342 GB
    "bf16_head_12k": KEPT * HIDDEN * 2 / 1e9,          # bf16 12k head        = 0.0629 GB
    # int4_head_12k filled from the measured served bytes below.
}


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def serve_and_capture(sub: Path, model_id: str, tag: str, out_file: Path,
                      summary_file: Path, *, want_tps: bool, plugins_off: bool,
                      head_bytes_log: Path | None = None):
    env = {"MODEL_ID": model_id, "SERVED_MODEL_NAME": paths.DEFAULT_SERVED_NAME}
    if plugins_off:
        env["VLLM_PLUGINS"] = ""
    if head_bytes_log is not None:
        env["SENPAI_HEAD_BYTES_LOG"] = str(head_bytes_log)
    tps = None
    with harness.LocalServer(
        sub, server_python=VENVPY, port=8000,
        log_path=out_file.parent / f"serve_{tag}.log", extra_env=env,
    ) as srv:
        print(f"[stage2] serve {tag} ready model_id={srv.model_id} {ts()}", flush=True)
        if out_file is not None and summary_file is not None:
            harness.capture_decode(
                VENVPY, base_url=srv.base_url, model=srv.served_model_name,
                out_file=out_file, summary_file=summary_file,
            )
        if want_tps:
            tps = harness.probe_tps(srv.base_url, srv.served_model_name, decode_tokens=256)
            print(f"[stage2] {tag} TPS(single-stream)={tps['decode_tps_single_stream']:.2f} {ts()}", flush=True)
    return tps


def probe_only(sub: Path, model_id: str, tag: str, log_dir: Path,
               head_bytes_log: Path | None = None):
    """Serve + probe_tps with NO 128-record decode (cheap same-session TPS anchor)."""
    env = {"MODEL_ID": model_id, "SERVED_MODEL_NAME": paths.DEFAULT_SERVED_NAME}
    if head_bytes_log is not None:
        env["SENPAI_HEAD_BYTES_LOG"] = str(head_bytes_log)
    log_dir.mkdir(parents=True, exist_ok=True)
    with harness.LocalServer(
        sub, server_python=VENVPY, port=8000,
        log_path=log_dir / f"serve_{tag}.log", extra_env=env,
    ) as srv:
        print(f"[stage2] serve {tag} ready model_id={srv.model_id} {ts()}", flush=True)
        tps = harness.probe_tps(srv.base_url, srv.served_model_name, decode_tokens=256)
        print(f"[stage2] {tag} TPS(single-stream)={tps['decode_tps_single_stream']:.2f} {ts()}", flush=True)
    return tps


def fit_bandwidth(points: list[tuple[float, float]]) -> dict:
    """Fit TPS = BW / (B + head_GB) on (head_GB, TPS) points.

    1/TPS = (1/BW)*head_GB + (B/BW)  ->  linear least squares in head_GB.
    Returns BW (GB/s) and B (body GB).
    """
    n = len(points)
    xs = [p[0] for p in points]
    ys = [1.0 / p[1] for p in points]
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom          # = 1/BW
    intercept = (sy - slope * sx) / n            # = B/BW
    bw = 1.0 / slope
    body = intercept * bw
    return {"bw_gb_s": bw, "body_gb": body, "slope_inv_bw": slope, "intercept_b_over_bw": intercept}


def main() -> int:
    CTRL_OUT.mkdir(parents=True, exist_ok=True)
    ISO_OUT.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[stage2] {note}", flush=True)

    # ---- Phase 1: unpruned-int4 control (false-divergence floor) ----
    print(f"=== PHASE 1 CONTROL unpruned-int4 served-vs-served start {ts()} ===", flush=True)
    ctrl_tps = serve_and_capture(
        BASELINE_SUB, INT4, "A", CTRL_OUT / "int4_A.jsonl", CTRL_OUT / "int4_A_summary.json",
        want_tps=True, plugins_off=True,
    )
    serve_and_capture(
        BASELINE_SUB, INT4, "B", CTRL_OUT / "int4_B.jsonl", CTRL_OUT / "int4_B_summary.json",
        want_tps=False, plugins_off=True,
    )
    report = greedy_gate.compare(CTRL_OUT / "int4_A.jsonl", CTRL_OUT / "int4_B.jsonl")
    onset = greedy_gate.onset_summary(report)
    ctrl_result = {
        "control_verdict": report.verdict,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "num_prompts_compared": report.num_prompts_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "divergence_onset": onset,
        "unpruned_int4_tps_single_stream": (ctrl_tps or {}).get("decode_tps_single_stream"),
        "tps_detail": ctrl_tps,
        "model_id": INT4,
        "method": "two separate int4 serves (vllm_baseline, VLLM_PLUGINS='', cudagraph ON), "
                  "official greedy verifier A-vs-B",
        "created_at": ts(),
    }
    (CTRL_OUT / "control_result.json").write_text(json.dumps(ctrl_result, indent=2, sort_keys=True))
    print(f"[stage2] CONTROL verdict={report.verdict} "
          f"{report.num_identical}/{report.num_prompts_compared} "
          f"unpruned_tps={ctrl_result['unpruned_int4_tps_single_stream']}", flush=True)

    # ---- Phase 2: same-session bf16-head TPS (isolated single-variable partner) ----
    print(f"=== PHASE 2 ISOLATED bf16-head same-session TPS start {ts()} ===", flush=True)
    bf16_tps = probe_only(
        EMPIRICAL_SUB, EMPIRICAL_CKPT, "bf16head", ISO_OUT,
        head_bytes_log=ISO_OUT / "bf16_served_head_bytes.json",
    )
    (ISO_OUT / "bf16_head_tps.json").write_text(json.dumps(bf16_tps, indent=2, sort_keys=True))

    # ---- Phase 3: analysis (isolated delta + bandwidth refit) ----
    print(f"=== PHASE 3 ANALYSIS {ts()} ===", flush=True)
    int4_tps = None
    if STAGE1_EVID.exists():
        ev = json.loads(STAGE1_EVID.read_text())
        int4_tps = ev.get("tps_single_stream_a10g")
    served_head_bytes = None
    if SERVED_HEAD_BYTES.exists():
        sh = json.loads(SERVED_HEAD_BYTES.read_text())
        served_head_bytes = sh.get("served_head_bytes")
    int4_head_gb = (served_head_bytes or 16220176) / 1e9
    HEAD_GB["int4_head_12k"] = int4_head_gb

    unpruned_tps = ctrl_result["unpruned_int4_tps_single_stream"]
    bf16_head_tps = (bf16_tps or {}).get("decode_tps_single_stream")

    isolated = None
    if int4_tps is not None and bf16_head_tps is not None:
        isolated = {
            "bf16_head_tps_same_session": bf16_head_tps,
            "int4_head_tps_same_session": int4_tps,
            "abs_delta_tps": int4_tps - bf16_head_tps,
            "pct_delta": 100.0 * (int4_tps - bf16_head_tps) / bf16_head_tps,
            "bf16_head_bytes": int(HEAD_GB["bf16_head_12k"] * 1e9),
            "int4_head_bytes": served_head_bytes,
            "note": "single variable: identical int4 body + kept_ids + plugin/stack; "
                    "ONLY lm_head dtype changes (bf16 62.9 MB -> int4 16.2 MB).",
        }

    band = None
    pts = []
    if unpruned_tps is not None:
        pts.append(("unpruned_int4", HEAD_GB["unpruned_int4"], unpruned_tps))
    if bf16_head_tps is not None:
        pts.append(("bf16_head_12k", HEAD_GB["bf16_head_12k"], bf16_head_tps))
    if int4_tps is not None:
        pts.append(("int4_head_12k", HEAD_GB["int4_head_12k"], int4_tps))
    if len(pts) >= 2:
        fit = fit_bandwidth([(g, t) for _, g, t in pts])
        preds = {name: fit["bw_gb_s"] / (fit["body_gb"] + g) for name, g, _ in pts}
        band = {
            "points": [{"name": n, "head_gb": g, "tps_measured": t,
                        "tps_pred_from_fit": preds[n]} for n, g, t in pts],
            "fit": fit,
            "note": "TPS = BW/(B+head_GB); least-squares on 1/TPS vs head_GB over "
                    "same-session points. PR projection used BW~483 GB/s, B~3.61 GB.",
        }

    analysis = {
        "created_at": ts(),
        "head_gb_by_checkpoint": HEAD_GB,
        "tps_same_session": {
            "unpruned_int4": unpruned_tps,
            "bf16_head_12k": bf16_head_tps,
            "int4_head_12k": int4_tps,
        },
        "isolated_head_dtype_delta": isolated,
        "bandwidth_model": band,
    }
    (ISO_OUT / "isolated_and_bandwidth.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    print("=== STAGE2 ANALYSIS ===", flush=True)
    print(json.dumps(analysis, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
