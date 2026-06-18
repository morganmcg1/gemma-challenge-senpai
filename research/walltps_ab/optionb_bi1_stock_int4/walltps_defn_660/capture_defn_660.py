"""PR #660 wall_tps-definition reconcile capture (LOCAL, analysis-only).

Re-runs the un-rescued served decode at K in {6, 5} on the EXACT #632 stack
(submission int4_mtp_batchinv, VLLM_BATCH_INVARIANT=1, MAX_NUM_SEQS=1, greedy,
128x512, seed=1, drafter /tmp/qat-assistant) with full per-step timing so the
offline analyzer can price wall_tps under BOTH PR-named definitions.

Per K (one fresh server) it captures three things:
  * Pass-1 (non-streaming, official decode_outputs.py via harness.capture_decode):
    the CANONICAL ``duration_s`` -> full_e2e wall_tps. This must reproduce the
    #632 headline (K6 ~170.21) -- the anchor for everything else.
  * the server-log "Avg generation throughput" interval series -> the steady-state
    generation-phase meter (excludes prefill/warmup ramp).
  * Pass-2 (streaming stream_decode.py): per-request TTFT (prefill) + per-token
    arrival times -> an INDEPENDENT full_e2e and steady computation from the
    timing splits, plus the cold-start (server_ready_s) for a boot-inclusive wall.

NO server-file changes, NO submission, NO HF job. official_tps=0, analysis_only.
Writes walltps_defn_capture.json for analyze_defn_reconcile.py.

Run under the repo python (has scripts.local_validation):
    python research/walltps_ab/optionb_bi1_stock_int4/walltps_defn_660/capture_defn_660.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tps_noise_floor.run_noise_floor import preflight_gpu  # noqa: E402
from scripts.local_validation import harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
DRAFTER = "/tmp/qat-assistant"          # #632 used this local drafter, not the hub default
SWEEP_KS = [6, 5]                        # K6 = headline (the 9.4% gap is at K6); K5 = check
GEN_TPS_RE = re.compile(r"Avg generation throughput:\s*([0-9.]+) tokens/s")
ESPEC_RE = re.compile(r"(?:Mean acceptance length|mean acceptance length):\s*([0-9.]+)")


def parse_gen_meter(log_path: Path) -> dict[str, Any]:
    """Steady-state generation-phase throughput from vLLM's interval meter."""
    txt = log_path.read_text(errors="replace") if log_path.exists() else ""
    vals = [float(x) for x in GEN_TPS_RE.findall(txt)]
    nz = [v for v in vals if v > 60.0]          # drop cold-start ramp + idle-zero intervals
    espec = [float(x) for x in ESPEC_RE.findall(txt)]
    import statistics
    return {
        "gen_tps_series_all": vals,
        "gen_tps_series_steady": nz,
        "steady_gen_tps_mean": statistics.fmean(nz) if nz else None,
        "steady_gen_tps_median": statistics.median(nz) if nz else None,
        "steady_gen_tps_n": len(nz),
        "first_interval_tps": vals[0] if vals else None,
        "espec_mean": statistics.fmean(espec) if espec else None,
    }


def run_k(server_python: Path, k: int, *, num_prompts: int, output_len: int, tag: str) -> dict[str, Any]:
    out_dir = HERE / f"k{k}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"
    extra_env = {
        "CUDA_VISIBLE_DEVICES": "0",          # inherited =7 is stale; only GPU is index 0
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",             # emit the gen-throughput + spec-accept meters
        "VLLM_BATCH_INVARIANT": "1",
        "DRAFTER_MODEL": DRAFTER,
        "NUM_SPECULATIVE_TOKENS": str(k),
        "MAX_NUM_SEQS": "1",
    }
    print(f"\n===== K={k} : fresh server =====", flush=True)
    preflight_gpu()
    t_boot0 = time.time()
    rec: dict[str, Any] = {"K": k, "t_start_utc": datetime.now(timezone.utc).isoformat()}
    with harness.LocalServer(
        SUBMISSION, server_python=server_python, log_path=server_log, extra_env=extra_env
    ) as server:
        server_ready_s = time.time() - t_boot0
        rec["server_ready_s"] = server_ready_s
        rec["served_model_name"] = server.served_model_name

        # Pass-1: canonical non-streaming -> full_e2e duration_s (reproduces #632)
        p1_out = out_dir / "pass1_nonstream.jsonl"
        p1_sum = out_dir / "pass1_nonstream.summary.json"
        t1 = time.time()
        summ = harness.capture_decode(
            server_python, base_url=server.base_url, model=server.served_model_name,
            out_file=p1_out, summary_file=p1_sum,
            num_prompts=num_prompts, output_len=output_len, seed=paths.SEED,
        )
        rec["pass1_wall_around_s"] = time.time() - t1
        rec["pass1_num_completion_tokens"] = int(summ["num_completion_tokens"])
        rec["pass1_duration_s"] = float(summ["duration_s"])
        rec["pass1_full_e2e_wall_tps"] = rec["pass1_num_completion_tokens"] / rec["pass1_duration_s"]

        # per-request token-id shas from Pass-1 (identity anchor)
        try:
            shas = [json.loads(l)["completion_token_sha256"]
                    for l in p1_out.read_text().splitlines() if l.strip()]
            rec["pass1_completion_sha_concat"] = __import__("hashlib").sha256(
                ",".join(shas).encode()).hexdigest()
        except Exception as e:
            rec["pass1_completion_sha_concat"] = f"ERR:{e}"

        # Pass-2: streaming instrumented -> per-request TTFT + per-token timing
        p2_out = out_dir / "pass2_stream.jsonl"
        p2_sum = out_dir / "pass2_stream.summary.json"
        try:
            cmd = [
                str(server_python), str(HERE / "stream_decode.py"),
                "--base-url", server.base_url, "--model", server.served_model_name,
                "--dataset-path", str(paths.EVAL_PROMPTS),
                "--decode-script", str(paths.DECODE_SCRIPT),
                "--output-file", str(p2_out), "--summary-file", str(p2_sum),
                "--tokenizer", paths.TOKENIZER,
                "--num-prompts", str(num_prompts),
                "--output-len", str(output_len), "--seed", str(paths.SEED),
            ]
            print("[stream]", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, timeout=3600)
            rec["pass2_stream"] = json.loads(p2_sum.read_text())
        except Exception as e:
            print(f"[stream] WARN pass-2 failed: {e}", flush=True)
            rec["pass2_stream"] = {"error": str(e)}

    rec["gen_meter"] = parse_gen_meter(server_log)
    # boot-inclusive ("cold job wall") denominator
    rec["cold_job_wall_tps"] = rec["pass1_num_completion_tokens"] / (
        rec["pass1_duration_s"] + rec["server_ready_s"])
    return rec


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default=",".join(str(k) for k in SWEEP_KS),
                    help="comma-separated K values (default 6,5)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--tag", default="", help="output subdir/file suffix (e.g. _smoke)")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    server_python = harness.ensure_server_venv(["vllm==0.22.0", "transformers==5.9.0"])
    print(f"[venv] server_python={server_python}", flush=True)
    out: dict[str, Any] = {
        "pr": 660, "analysis_only": True, "official_tps": 0,
        "stack": "int4_mtp_batchinv", "drafter": DRAFTER,
        "config": {"batch_invariant": 1, "max_num_seqs": 1, "greedy": True,
                   "num_prompts": args.num_prompts, "output_len": args.output_len,
                   "seed": paths.SEED, "vllm": "0.22.0"},
        "rows": [],
    }
    for k in ks:
        rec = run_k(server_python, k, num_prompts=args.num_prompts,
                    output_len=args.output_len, tag=args.tag)
        out["rows"].append(rec)
        print(f"\n[K={k}] full_e2e={rec['pass1_full_e2e_wall_tps']:.3f}  "
              f"steady_gen_meter={rec['gen_meter']['steady_gen_tps_mean']}  "
              f"cold_job_wall={rec['cold_job_wall_tps']:.3f}  "
              f"server_ready_s={rec['server_ready_s']:.1f}", flush=True)
        sp = rec.get("pass2_stream", {})
        if "stream_full_e2e_wall_tps" in sp:
            print(f"[K={k}] stream_full_e2e={sp['stream_full_e2e_wall_tps']:.3f}  "
                  f"stream_steady={sp['stream_steady_wall_tps']:.3f}  "
                  f"mean_ttft_s={sp['mean_ttft_s']:.4f}", flush=True)
        (HERE / f"walltps_defn_capture{args.tag}.json").write_text(
            json.dumps(out, indent=2, default=str))
    print(f"\n[done] wrote {HERE / f'walltps_defn_capture{args.tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
