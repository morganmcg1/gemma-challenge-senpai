#!/usr/bin/env python3
"""PR #791 step-3: firm up the surgattn-OFF (3D-on-M=1) decode-TPS A/B with reps.

#785 measured this n=1/arm (variant 224.55 / control 210.48 -> +6.69%) and flagged
a noise component (the delta exceeds the sub-1% attn-compute ceiling). This script
re-runs the SAME decode benchmark (speed_benchmark/decode_outputs.py, 32 prompts x
512 output tok, the exact source of the #785 anchors) but R reps against ONE warm
server per arm, so the only changed variable across arms is VLLM_SURGATTN and the
within-arm spread quantifies the noise floor.

  * control = shipped bi0 (force-2D ON, byte-identical), VLLM_SURGATTN unset/default.
  * variant = VLLM_SURGATTN=0 (kernel gate picks 3D split-KV on the M=1 forwards).

TPS = num_completion_tokens / duration_s (matches local_prevalidate's definition).
Shipped serve config (MAX_MODEL_LEN=4096, MAX_NUM_SEQS=1 => every decode M=1) is
preserved. LOCAL ONLY -- no HF job, exploratory proxy (official a10g A/B is the real
speed test, human-gated, out of scope here).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
SERVER_PY = ROOT / ".venvs" / "vllm022" / "bin" / "python"
CLIENT_PY = ROOT / ".venvs" / "vllm022" / "bin" / "python"
HARNESS = ROOT / "official" / "main_bucket" / "shared_resources" / "speed_benchmark"
DECODE_SCRIPT = HARNESS / "scripts" / "decode_outputs.py"
EVAL_DATASET = HARNESS / "data" / "eval_prompts_sharegpt.json"
SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_surgattn"
PORT = 8000

COMMON_SERVER_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",  # native sampler; avoids cuRAND JIT; logit-identical
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
ARM_ENV = {"control": {}, "variant": {"VLLM_SURGATTN": "0"}}
ARM_LOG_REQUIRE = {
    "control": "[int4_mtp_force2d] unified_attention wrapped",
    "variant": "[int4_mtp_surgattn] VLLM_SURGATTN=0",
}
ARM_LOG_FORBID = {
    "control": "[int4_mtp_surgattn] VLLM_SURGATTN=0",
    "variant": "[int4_mtp_force2d] unified_attention wrapped",
}


def _wait_log(log_path: Path, needle: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if needle in log_path.read_text(errors="replace"):
                return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    return False


def _decode_once(out_dir: Path, tag: str, *, num_prompts: int, output_len: int) -> dict:
    summary_file = out_dir / f"decode_{tag}.json"
    cmd = [
        str(CLIENT_PY), str(DECODE_SCRIPT),
        "--base-url", f"http://127.0.0.1:{PORT}", "--model", "gemma-4-e4b-it",
        "--dataset-path", str(EVAL_DATASET),
        "--output-file", str(out_dir / f"decode_{tag}.jsonl"),
        "--summary-file", str(summary_file),
        "--num-prompts", str(num_prompts), "--output-len", str(output_len),
        "--seed", "1", "--request-timeout-s", "180",
    ]
    subprocess.run(cmd, check=True)
    d = json.loads(summary_file.read_text())
    dur = d.get("duration_s") or 0.0
    toks = d.get("num_completion_tokens") or 0
    d["tps"] = (toks / dur) if dur else 0.0
    return d


def run_arm(arm: str, *, out_dir: Path, reps: int, num_prompts: int, output_len: int) -> dict:
    extra = {**COMMON_SERVER_ENV, **ARM_ENV[arm]}
    log_path = out_dir / f"tps_server_{arm}.log"
    print(f"\n===== ARM {arm}  extra_env={ARM_ENV[arm]} =====", flush=True)
    res: dict = {"arm": arm, "extra_env": ARM_ENV[arm], "reps": []}
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=PORT, log_path=log_path,
        extra_env=extra, startup_timeout_s=1800,
    ):
        need, forbid = ARM_LOG_REQUIRE[arm], ARM_LOG_FORBID[arm]
        if not _wait_log(log_path, need, timeout_s=120):
            raise RuntimeError(f"[{arm}] expected server-log marker absent: {need!r}")
        if forbid in log_path.read_text(errors="replace"):
            raise RuntimeError(f"[{arm}] forbidden marker present (wrong toggle): {forbid!r}")
        print(f"[{arm}] toggle proven: present={need!r} absent={forbid!r}", flush=True)
        res["toggle_proven"] = True

        print(f"[{arm}] warmup (discarded) ...", flush=True)
        _decode_once(out_dir, f"{arm}_warmup", num_prompts=4, output_len=output_len)

        for r in range(reps):
            d = _decode_once(out_dir, f"{arm}_rep{r}", num_prompts=num_prompts, output_len=output_len)
            print(f"[{arm}] rep{r}: tps={d['tps']:.2f} "
                  f"(ctok={d['num_completion_tokens']} dur={d['duration_s']:.2f}s recs={d['num_records']})",
                  flush=True)
            res["reps"].append({"rep": r, "tps": d["tps"], "duration_s": d["duration_s"],
                                "num_completion_tokens": d["num_completion_tokens"],
                                "num_records": d["num_records"]})
    tps = [x["tps"] for x in res["reps"]]
    res["tps_mean"] = statistics.mean(tps)
    res["tps_min"] = min(tps)
    res["tps_max"] = max(tps)
    res["tps_stdev"] = statistics.stdev(tps) if len(tps) > 1 else 0.0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="control,variant")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--out-dir", default=str(HERE / "tps"))
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    results: dict[str, dict] = {}
    for arm in arms:
        results[arm] = run_arm(arm, out_dir=out_dir, reps=args.reps,
                               num_prompts=args.num_prompts, output_len=args.output_len)
        (out_dir / "tps_summary.json").write_text(json.dumps(
            {"served_config": "MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 (every decode M=1)",
             "benchmark": f"decode_outputs.py num_prompts={args.num_prompts} output_len={args.output_len} seed=1",
             "anchors_785": {"variant": 224.55, "control": 210.48, "delta_pct": 6.69},
             "arms": results}, indent=2))

    if "variant" in results and "control" in results:
        v, c = results["variant"]["tps_mean"], results["control"]["tps_mean"]
        delta = 100.0 * (v - c) / c if c else 0.0
        print(f"\n[A/B] variant_mean={v:.2f} control_mean={c:.2f} delta={delta:+.2f}%", flush=True)
        summ = json.loads((out_dir / "tps_summary.json").read_text())
        summ["ab_delta_pct"] = delta
        summ["variant_mean_tps"] = v
        summ["control_mean_tps"] = c
        (out_dir / "tps_summary.json").write_text(json.dumps(summ, indent=2))

    print("\n========== TPS SUMMARY ==========", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
