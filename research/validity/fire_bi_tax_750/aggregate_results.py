#!/usr/bin/env python3
"""Aggregate all PR #750 raw stage outputs into runs/RESULTS.json.

Re-derives every scalar from the raw sglang bench logs + decode summaries +
server logs so the deliverable is reproducible from artifacts (not hand-typed),
then preserves any pre-existing hand-banked keys it does not recompute. Run AFTER
run_all.sh finishes all four stages, BEFORE compute_projection.py.

Meters (kept apples-to-apples):
  * tps_BIx        = sglang "Output token throughput" (the OFFICIAL summary.json
                     basis; same meter as the int4_qat anchor) from bench_bix.log.
  * local_int4_qat = same sglang meter on int4_qat (the official-anchor denominator).
  * *_decode_tps   = decode-pass throughput (decode_outputs.py sequential calls);
                     a SEPARATE, slower meter — used only for the spec-OFF refs and
                     flagged as such (do NOT cross the two meters).
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

D = Path("runs")


def parse_bench(log: Path) -> dict:
    """sglang bench_serving summary block -> dict."""
    if not log.exists():
        return {}
    t = log.read_text()
    def grab(pat):
        m = re.search(pat, t)
        return float(m.group(1)) if m else None
    return {
        "output_tps": grab(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
        "total_tps": grab(r"Total token throughput \(tok/s\):\s+([0-9.]+)"),
        "duration_s": grab(r"Benchmark duration \(s\):\s+([0-9.]+)"),
        "completed": grab(r"Successful requests:\s+([0-9]+)"),
        "gen_tokens": grab(r"Total generated tokens:\s+([0-9]+)"),
    }


def decode_tps(summary: Path) -> dict:
    if not summary.exists():
        return {}
    s = json.loads(summary.read_text())
    dur = s.get("duration_s")
    ntok = s.get("num_completion_tokens")
    return {
        "duration_s": dur,
        "num_completion_tokens": ntok,
        "tps": (ntok / dur) if (dur and ntok) else None,
        "num_records": s.get("num_records"),
    }


def mean_acceptance(server_log: Path) -> float | None:
    if not server_log.exists():
        return None
    vals = [float(m) for m in re.findall(
        r"Mean acceptance length:\s+([0-9.]+)", server_log.read_text())]
    return round(statistics.fmean(vals), 3) if vals else None


def peak_gpu_mib(*sample_files: Path) -> int | None:
    peak = 0
    for f in sample_files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            for n in re.findall(r"(\d+)\s*MiB", line):
                peak = max(peak, int(n))
            line = line.strip()
            if line.isdigit():
                peak = max(peak, int(line))
    return peak or None


def main() -> None:
    R = {}
    rj = D / "RESULTS.json"
    if rj.exists():
        R = json.loads(rj.read_text())  # preserve hand-banked keys

    # ---- sglang bench arms (OFFICIAL-basis TPS meter) ----
    for arm, bi in (("bi1", 1), ("bi0", 0)):
        b = parse_bench(D / f"bench_{arm}.log")
        if b.get("output_tps") is not None:
            R[f"tps_BI{bi}"] = b["output_tps"]
            R[f"tps_BI{bi}_total_tps"] = b["total_tps"]
            R[f"tps_BI{bi}_bench_duration_s"] = b["duration_s"]
            R[f"tps_BI{bi}_completed"] = int(b["completed"]) if b["completed"] else None
            R[f"tps_BI{bi}_total_output_tokens"] = int(b["gen_tokens"]) if b["gen_tokens"] else None
        acc = mean_acceptance(D / f"server_{arm}_specon.log")
        if acc is not None:
            R[f"tps_BI{bi}_mean_acceptance_len"] = acc

    # ---- int4_qat official anchor (same sglang meter, same checkpoint) ----
    iq = parse_bench(D / "bench_int4qat.log")
    if iq.get("output_tps") is not None:
        R["local_int4_qat_tps"] = iq["output_tps"]
        R["local_int4_qat_bench_duration_s"] = iq["duration_s"]
        R["local_int4_qat_completed"] = int(iq["completed"]) if iq["completed"] else None

    # ---- spec-OFF references (decode-pass meter; DO NOT cross with sglang) ----
    r1 = decode_tps(D / "decode_ref_bi1_summary.json")
    if r1.get("tps") is not None:
        R["local_fire_specOFF_BI1_decode_tps"] = r1["tps"]
        R["ref_bi1_decode_duration_s"] = r1["duration_s"]
    r0 = decode_tps(D / "decode_ref_bi0_summary.json")
    if r0.get("tps") is not None:
        R["local_fire_specOFF_BI0_decode_tps"] = r0["tps"]
        R["ref_bi0_decode_duration_s"] = r0["duration_s"]
    c1 = decode_tps(D / "decode_cand_bi1_summary.json")
    if c1.get("tps") is not None:
        R["cand_bi1_decode_duration_s"] = c1["duration_s"]
        R["cand_bi1_decode_tps_xcheck"] = c1["tps"]
    c0 = decode_tps(D / "decode_cand_bi0_summary.json")
    if c0.get("tps") is not None:
        R["cand_bi0_decode_duration_s"] = c0["duration_s"]
        R["cand_bi0_decode_tps_xcheck"] = c0["tps"]

    # ---- peak GPU ----
    pk = peak_gpu_mib(D / "gpu_mem_samples.txt", D / "gpu_mem_samples_l2.txt",
                      D / "mem_bi0.txt")
    if pk:
        R["peak_gpu_mib"] = pk

    rj.write_text(json.dumps(R, indent=2, sort_keys=True))
    print(f"[aggregate] wrote {rj} with {len(R)} keys")
    for k in ("tps_BI1", "tps_BI0", "local_int4_qat_tps",
              "local_fire_specOFF_BI1_decode_tps", "local_fire_specOFF_BI0_decode_tps",
              "tps_BI1_mean_acceptance_len", "tps_BI0_mean_acceptance_len",
              "identity_BI1", "identity_BI0", "peak_gpu_mib"):
        print(f"  {k:38s} = {R.get(k)}")


if __name__ == "__main__":
    main()
