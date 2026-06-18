#!/usr/bin/env python3
"""Finalize PR #675: attach per-candidate achieved GB/s (advisor ask, #675
comment 2026-06-18) + the v0220 ship-venv determinism control + denken #676
roofline cross-check to results.json.

achieved bandwidth (wall basis) = W_bytes_per_token * wall_tps, where
W_bytes_per_token = the text-decoder weight footprint streamed per M=1 text
token (header-derived; int4 body + int4 lm_head; excludes embed tables [row
gather] + vision/audio towers [silent on text]). At conc=1 the server-log
steady-state generation throughput (~127 tok/s) == wall_tps, so wall_tps is the
decode-step rate (prefill amortization negligible) and this GB/s is a decode-step
bandwidth, not merely an end-to-end lower bound.

denken #676 (relayed by advisor, cross-read #666, W&B vwiqwzvk): GEMV-isolated
469.3 GB/s = 90.7% of empirical read-peak 517.7 GB/s (== 86.3% of A10G 600 GB/s
spec). All GB use 1e9 (decimal), matching the 600 GB/s spec sheet.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results.json"
WF = HERE / "weight_footprint.json"
V0220 = HERE / "results_v0220.json"

DENKEN_GEMV_GBPS = 469.3        # GEMV-isolated achieved (denken #676 / vwiqwzvk)
READ_PEAK_GBPS = 517.7         # empirical read-peak (denken #676)
A10G_SPEC_GBPS = 600.0         # A10G GDDR6 spec memory bandwidth

res = json.loads(RES.read_text())
wf = json.loads(WF.read_text())
v0220 = json.loads(V0220.read_text())

W_GB = wf["W_GB_per_token"]   # 1e9-basis GB streamed per M=1 text token

def gbps(tps):
    return None if tps is None else W_GB * tps

# Per-candidate achieved bandwidth (wall == decode-step basis).
bw = {}
for arm, d in res["arms"].items():
    tps = d.get("wall_tps")
    if d.get("status") == "decode_timeout":
        ub = d.get("partial_wall_tps_upper_bound")
        bw[arm] = {
            "wall_tps": None, "tps_upper_bound": ub,
            "achieved_gbps_upper_bound": gbps(ub),
            "pct_read_peak_upper_bound": (gbps(ub) / READ_PEAK_GBPS * 100) if ub else None,
            "active_kernel": d.get("active_kernel"), "status": d["status"],
        }
    elif tps is None:
        bw[arm] = {"wall_tps": None, "achieved_gbps": None,
                   "active_kernel": d.get("active_kernel"), "status": d.get("status")}
    else:
        g = gbps(tps)
        bw[arm] = {
            "wall_tps": tps,
            "achieved_gbps": g,
            "pct_read_peak": g / READ_PEAK_GBPS * 100,
            "pct_a10g_spec": g / A10G_SPEC_GBPS * 100,
            "active_kernel": d.get("active_kernel"),
            "status": d.get("status"),
        }

# ship-venv anchor
for arm in ("v0220_a", "v0220_b"):
    tps = v0220["wall_tps"][arm]
    g = gbps(tps)
    bw[arm] = {"wall_tps": tps, "achieved_gbps": g,
               "pct_read_peak": g / READ_PEAK_GBPS * 100,
               "pct_a10g_spec": g / A10G_SPEC_GBPS * 100,
               "active_kernel": "MarlinLinearKernel",
               "status": "ok_shipvenv_0.22.0"}

marlin_med_gbps = gbps(res["anchor_wall_tps_median_of_base"])
triton_ub_gbps = bw["triton1"]["achieved_gbps_upper_bound"]

res["weight_footprint"] = {
    "W_bytes_per_token": wf["W_bytes_per_token"],
    "W_GB_per_token": W_GB,
    "components": {
        "int4_body_text_decoder_GB": wf["by_bucket_bytes"]["text_decoder"] / 1e9,
        "int4_lm_head_GB": wf["by_bucket_bytes"]["lm_head"] / 1e9,
    },
    "excluded_GB": {k: v / 1e9 for k, v in wf["excluded_bytes"].items()},
    "note": wf["note"],
}
res["bandwidth"] = {
    "basis": "achieved GB/s = W_GB_per_token * wall_tps (1e9 GB). conc=1 "
             "server-log generation throughput == wall_tps -> decode-step basis.",
    "denken676_anchor": {
        "gemv_isolated_gbps": DENKEN_GEMV_GBPS,
        "read_peak_gbps": READ_PEAK_GBPS,
        "pct_read_peak": DENKEN_GEMV_GBPS / READ_PEAK_GBPS * 100,
        "a10g_spec_gbps": A10G_SPEC_GBPS,
        "source": "denken #676 / W&B vwiqwzvk, relayed by advisor (cross-read #666)",
    },
    "per_candidate": bw,
    "marlin_median_gbps": marlin_med_gbps,
    "marlin_pct_read_peak": marlin_med_gbps / READ_PEAK_GBPS * 100,
    "marlin_vs_triton_speedup": (marlin_med_gbps / triton_ub_gbps) if triton_ub_gbps else None,
    "implied_gemv_wall_share": marlin_med_gbps / DENKEN_GEMV_GBPS,
    "crosscheck_note": (
        f"Marlin wall/decode-step bandwidth {marlin_med_gbps:.1f} GB/s "
        f"({marlin_med_gbps / READ_PEAK_GBPS * 100:.1f}% of {READ_PEAK_GBPS} read-peak) is a "
        f"strict lower bound on denken's GEMV-isolated {DENKEN_GEMV_GBPS} GB/s; ratio "
        f"{marlin_med_gbps / DENKEN_GEMV_GBPS:.3f} => the weight GEMV occupies "
        f"{marlin_med_gbps / DENKEN_GEMV_GBPS * 100:.1f}% of M=1 per-token time, the rest "
        f"attention/sampler/host/PLE-gather. Consistent, not contradictory. No byte-identical "
        f"kernel/knob raises achieved bandwidth above Marlin; Triton is "
        f"{(marlin_med_gbps / triton_ub_gbps):.1f}x slower. Selection-side confirmation of "
        f"'byte-identically realizable headroom ~= 0'."
    ),
}
res["v0220_ship_control"] = {
    "vllm_version": v0220["vllm_version"],
    "wall_tps": v0220["wall_tps"],
    "ship_self_break_rate": v0220["ship_self_determinism"]["break_rate"],
    "ship_env_deterministic": v0220["ship_env_deterministic"],
    "crossversion_dev307_vs_0220_break_rate": v0220["crossversion_v0220a_vs_dev307base1"]["break_rate"],
    "interpretation": (
        "On the SHIP vLLM 0.22.0 venv two fresh-server reps are byte-identical "
        "(break_rate=0.0) => #319 holds in production and the dev307 0.906 break_rate "
        "between provably-identical Marlin runs is a LOCAL dev307 autotune artifact "
        "(#601), not a kernel-induced identity break and not a ship-gate risk. "
        "Cross-version dev307-vs-0.22.0 diverges (expected: different vLLM numerics). "
        "Ship anchor 127.018/126.960 reproduces the dev307 127.098 wall_tps."
    ),
}

RES.write_text(json.dumps(res, indent=2))

# console table
print(f"W_bytes_per_token = {W_GB:.4f} GB (int4 body {wf['by_bucket_bytes']['text_decoder']/1e9:.3f} "
      f"+ int4 lm_head {wf['by_bucket_bytes']['lm_head']/1e9:.3f})")
print(f"denken #676 GEMV-isolated: {DENKEN_GEMV_GBPS} GB/s = "
      f"{DENKEN_GEMV_GBPS/READ_PEAK_GBPS*100:.1f}% of {READ_PEAK_GBPS} read-peak\n")
print(f"{'candidate':12s} {'kernel':24s} {'wall_tps':>9s} {'GB/s':>9s} {'%peak':>7s}  status")
order = ["base2", "base3", "base1", "v0220_a", "v0220_b", "atomicadd1", "baseBI1", "triton1", "humming1"]
for a in order:
    b = bw[a]
    tps = b.get("wall_tps")
    g = b.get("achieved_gbps") or b.get("achieved_gbps_upper_bound")
    pk = b.get("pct_read_peak") or b.get("pct_read_peak_upper_bound")
    tps_s = f"{tps:9.3f}" if tps else (f"<{b.get('tps_upper_bound'):8.2f}" if b.get('tps_upper_bound') else f"{'n/a':>9s}")
    g_s = f"{g:9.1f}" if g else f"{'n/a':>9s}"
    pk_s = f"{pk:6.1f}%" if pk else f"{'n/a':>7s}"
    ub = "<=" if b.get("status") == "decode_timeout" else "  "
    print(f"{a:12s} {str(b.get('active_kernel')):24s} {tps_s} {ub}{g_s} {pk_s}  {b.get('status')}")
print(f"\nMarlin median {marlin_med_gbps:.1f} GB/s ({marlin_med_gbps/READ_PEAK_GBPS*100:.1f}% read-peak); "
      f"Marlin/Triton = {marlin_med_gbps/triton_ub_gbps:.1f}x; "
      f"implied GEMV wall-share vs denken = {marlin_med_gbps/DENKEN_GEMV_GBPS*100:.1f}%")
print(f"v0220 ship self break_rate = {v0220['ship_self_determinism']['break_rate']}  "
      f"ship_env_deterministic = {v0220['ship_env_deterministic']}")
print(f"VERDICT: {res['verdict']}")
