#!/usr/bin/env python
"""Calibration probe (PR #814 follow-up): serve ONE body-group-size checkpoint
through the EXACT int4head serve path (MAX_NUM_SEQS=1, same as measure_arm.py)
and report single-stream decode TPS via harness.probe_tps, PLUS the prefill
components (wall_1tok / wall_Ntok / ttft) so the probe number can be diagnosed.

WHY: measure_arm.py reports an 8-prompt-sequential wall_tps (prefill-INCLUDED)
~234-240, while probe_single_stream.py reported g128=347 (prefill-EXCLUDED).
Both were naively compared to the SAME 256.74 int4head anchor with contradictory
verdicts (-6% vs +35%). They measure different things, so at most one is
apples-to-apples with the anchor. This script re-probes each arm at one fixed
config so g32-vs-g128 on probe_tps is a clean within-harness A/B, independent of
the 256.74 anchor's provenance.

LOCAL single-A10G only. No HF Job, no submission.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

INT4HEAD_DIR = ROOT / "submissions" / "int4_mtp_bi0_int4head"
SERVED_NAME = "gemma-4-e4b-it"
INT4HEAD_ANCHOR = 256.74  # PR #814 "256.74 local single-stream TPS" reference


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--body-group-size", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "calibrate_results.jsonl"))
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[calib] {note}", flush=True)

    ckpt = Path(args.ckpt).resolve()
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    manifest = harness.load_manifest(INT4HEAD_DIR)
    server_py = harness.ensure_server_venv(manifest["dependencies"])
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = Path(__file__).resolve().parent / "logs" / f"calib_{args.arm}_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rec: dict = {
        "arm": args.arm, "ckpt": str(ckpt), "body_group_size": args.body_group_size,
        "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "config": "int4head-serve-path MAX_NUM_SEQS=1 (measure_arm parity)",
        "served_ok": False, "error": None,
        "int4head_anchor": INT4HEAD_ANCHOR,
        "reps": args.reps, "decode_tokens": args.decode_tokens,
        "probe_samples": [],
    }
    try:
        t0 = time.time()
        with harness.LocalServer(
            INT4HEAD_DIR, server_python=server_py, port=args.port,
            log_path=log_path, extra_env={"MODEL_ID": str(ckpt)},
        ):
            rec["ready_s"] = round(time.time() - t0, 1)
            rec["served_ok"] = True
            for r in range(args.reps):
                res = harness.probe_tps(base_url, SERVED_NAME, decode_tokens=args.decode_tokens)
                rec["probe_samples"].append(res)
                print(
                    f"  rep{r}: decode_tps_single_stream={res['decode_tps_single_stream']:.2f} "
                    f"naive_tps={res['naive_tps']:.2f} ttft={res['ttft_s_approx']:.3f}s "
                    f"wall1={res['wall_1tok_s']:.3f}s wallN={res['wall_ntok_s']:.3f}s n={res['decode_tokens']}",
                    flush=True,
                )
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"[calib] ERROR: {exc}", flush=True)

    ss = [s["decode_tps_single_stream"] for s in rec["probe_samples"]
          if s["decode_tps_single_stream"] == s["decode_tps_single_stream"]]
    rec["probe_tps_mean"] = statistics.fmean(ss) if ss else None
    rec["probe_tps_std"] = statistics.pstdev(ss) if len(ss) > 1 else 0.0
    rec["probe_tps_median"] = statistics.median(ss) if ss else None
    if rec["probe_tps_mean"] is not None:
        rec["delta_pct_vs_anchor"] = round(
            100.0 * (rec["probe_tps_mean"] - INT4HEAD_ANCHOR) / INT4HEAD_ANCHOR, 3)
    rec["t_end_utc"] = datetime.now(timezone.utc).isoformat()

    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    print("\n[calib] ===== RESULT =====", flush=True)
    print(f"  arm={rec['arm']} body_gs={rec['body_group_size']} served={rec['served_ok']} error={rec['error']}", flush=True)
    if rec["probe_tps_mean"] is not None:
        print(f"  probe_tps_single_stream mean={rec['probe_tps_mean']:.2f} std={rec['probe_tps_std']:.2f} "
              f"median={rec['probe_tps_median']:.2f} (n={len(ss)})", flush=True)
        print(f"  vs 256.74 anchor: delta={rec['delta_pct_vs_anchor']}%", flush=True)
    return 0 if rec["served_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
