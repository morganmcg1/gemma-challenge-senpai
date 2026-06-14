#!/usr/bin/env python3
"""Assemble the PR #90 MTP draft-length K sweep into a single wall_tps-vs-K curve.

Reads the per-arm ``paired_ab.json`` produced by ``paired_tps_ab.py`` for each K in
{5,6,8,9} (each measured against the shared fresh K=7 baseline) and prints:

  * the median wall_tps per K (the #72/#82 protocol metric) + Δ% vs K=7 + REAL/NULL,
  * a "warm" median that drops the first-run-of-a-new-config cold-cache run
    (identified by an anomalous ``server_ready_s``) as a robustness cross-check,
  * E[accept] (accepted tok/step) per K to validate the cost-model K* prediction,
  * the GATE: K=7 CONFIRMED OPTIMAL (best or tie within NULL) vs FREE WIN.

The cold-cache run is a known artifact: the deployed K=7 graphs are already cached
so its 3 runs are warm, but each novel K recompiles loopgraph/CUDA-graph on its first
fresh server (``server_ready_s`` jumps ~90s -> ~145s), depressing only that run's
wall_tps. The median-of-3 already excludes that single low outlier; the warm median
makes the exclusion explicit and auditable.

Usage:
    .venv/bin/python research/walltps_ab/analyze_k_sweep.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "research" / "walltps_ab"

# K=7 is the deployed baseline (shared, run fresh once inside the K=6 arm).
BASELINE_K = 7
# K -> the per-arm result dir / paired_ab.json that holds that K's candidate arm.
K_ARMS = {6: "mtp_k6", 5: "mtp_k5", 8: "mtp_k8", 9: "mtp_k9"}
# A fresh server that recompiles a novel-K graph runs noticeably longer than the
# warm steady ~90s; flag runs above this as cold-cache-contaminated for the warm median.
COLD_READY_S = 120.0
# #72/#82 operative REAL/NULL bar at N=3.
OP_THRESHOLD_PCT = 0.10


def _load(arm_dir: str) -> dict | None:
    p = SWEEP / arm_dir / "paired_ab.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _warm_median(records: list[dict]) -> tuple[float | None, int]:
    """Median wall_tps over runs whose server_ready_s looks warm (drops cold-cache
    first-runs). Returns (median, n_cold_dropped)."""
    warm = [r["wall_tps"] for r in records
            if r.get("wall_tps") is not None
            and (r.get("server_ready_s") or 0) <= COLD_READY_S]
    n_cold = len(records) - len(warm)
    if not warm:
        return None, n_cold
    return statistics.median(warm), n_cold


def main() -> int:
    # Pull the shared K=7 baseline arm from any completed candidate result.
    base = None
    for arm_dir in K_ARMS.values():
        d = _load(arm_dir)
        if d:
            base = d["arms"]["baseline"]
            break
    if base is None:
        print("[k-sweep] no completed arm results yet under", SWEEP)
        return 1

    base_med = base["wall_tps"]["median"]
    base_warm, base_cold = _warm_median(base["records"])
    base_eacc = base["e_accept_exact"]["mean"]

    rows = []
    # K=7 baseline row.
    rows.append({
        "K": BASELINE_K, "median": base_med, "warm_median": base_warm,
        "n_cold": base_cold, "e_accept": base_eacc, "cv_pct": base["wall_tps"].get("cv_pct"),
        "delta_pct": 0.0, "verdict": "REF", "ready_s": [round(r.get("server_ready_s", 0)) for r in base["records"]],
        "vals": [round(v, 2) for v in base["wall_tps"]["values"]],
    })
    for K, arm_dir in sorted(K_ARMS.items()):
        d = _load(arm_dir)
        if not d:
            rows.append({"K": K, "median": None, "pending": True})
            continue
        cand = d["arms"]["candidate"]
        med = cand["wall_tps"]["median"]
        warm, n_cold = _warm_median(cand["records"])
        delta = 100.0 * (med - base_med) / base_med
        verdict = "REAL" if abs(delta) >= OP_THRESHOLD_PCT else "NULL"
        rows.append({
            "K": K, "median": med, "warm_median": warm, "n_cold": n_cold,
            "e_accept": cand["e_accept_exact"]["mean"], "cv_pct": cand["wall_tps"].get("cv_pct"),
            "delta_pct": delta, "verdict": verdict,
            "ready_s": [round(r.get("server_ready_s", 0)) for r in cand["records"]],
            "vals": [round(v, 2) for v in cand["wall_tps"]["values"]],
        })

    rows.sort(key=lambda r: r["K"])
    print("\n===== PR #90 MTP draft-length K sweep — wall_tps vs K =====")
    print(f"{'K':>3}  {'median':>9}  {'Δ% vs K7':>9}  {'verdict':>7}  {'warm_med':>9}  "
          f"{'cold':>4}  {'E[accept]':>9}  {'cv%':>7}  ready_s / vals")
    for r in rows:
        if r.get("pending"):
            print(f"{r['K']:>3}  {'PENDING':>9}")
            continue
        wm = f"{r['warm_median']:.3f}" if r.get("warm_median") is not None else "—"
        cv = f"{r['cv_pct']:.4f}" if isinstance(r.get("cv_pct"), (int, float)) else "—"
        print(f"{r['K']:>3}  {r['median']:>9.3f}  {r['delta_pct']:>+8.3f}%  {r['verdict']:>7}  "
              f"{wm:>9}  {r['n_cold']:>4}  {r['e_accept']:>9.4f}  {cv:>7}  "
              f"{r['ready_s']} / {r['vals']}")

    # Gate on the protocol median (warm median is the audit cross-check).
    done = [r for r in rows if not r.get("pending")]
    best = max(done, key=lambda r: r["median"])
    k7 = next(r for r in done if r["K"] == BASELINE_K)
    # K=7 confirmed optimal if it is the best, or no other K beats it by >= the REAL bar.
    better = [r for r in done if r["K"] != BASELINE_K
              and (r["median"] - k7["median"]) / k7["median"] * 100.0 >= OP_THRESHOLD_PCT]
    k7_optimal = len(better) == 0
    print()
    if k7_optimal:
        print(f">>> GATE: K=7 CONFIRMED OPTIMAL — best K={best['K']} "
              f"(median {best['median']:.3f}); no K is REAL-better than K=7. "
              f"mtp_k7_confirmed_optimal_bool=1")
    else:
        win = max(better, key=lambda r: r["median"])
        print(f">>> GATE: FREE WIN — K={win['K']} is REAL-better than K=7 "
              f"(+{win['delta_pct']:.3f}%, median {win['median']:.3f}). "
              f"mtp_k7_confirmed_optimal_bool=0")
    print(f">>> primary mtp_k_optimal_wall_tps = {best['median']:.3f} (K={best['K']})")
    print(f">>> K=7 locked local reference wall_tps = {k7['median']:.3f}")
    n_pending = sum(1 for r in rows if r.get("pending"))
    if n_pending:
        print(f"\n[note] {n_pending} arm(s) still PENDING — gate is provisional until all complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
