#!/usr/bin/env python3
"""Assemble the PR #138 block64 MTP draft-length K sweep into the wall_tps-vs-K curve.

Reads the per-arm ``paired_ab.json`` produced by ``paired_tps_ab.py``:

  * ``k7_block64`` — baseline arm = the FRESH K7-block16 anchor (must reproduce the
    lawine #90 locked 454.338), candidate arm = K7-block64.
  * ``k6_block64`` / ``k8_block64`` / ``k9_block64`` — candidate arm = K{6,8,9}-block64
    (each measured against the reused K7-block16 anchor baseline).

Prints the PR #138 table (``K | block | wall_tps | Δ% vs K7-block16 | Δ% vs K7-block64``),
plus E[accept] and a derived per-step time ``step_time_ms = 1000 * E[accept] / wall_tps``
(block size leaves E[accept] invariant at fixed K — bit-identical drafter proposals — so
wall_tps moves are pure step-time moves), and the GATE:

  * K* = argmax median wall_tps at block64;
  * does any K beat K7-block64 by >= the operative N=3 bar (0.10%)? -> K* shifted up;
  * else K=7 remains optimal even with the cheaper block64 argmax.

A warm median (drops the cold-cache first-run of each novel-K graph, flagged by an
anomalous ``server_ready_s``) is reported as an audit cross-check, exactly as in the
lawine #90 ``analyze_k_sweep.py``.

Usage:
    .venv/bin/python research/walltps_ab/analyze_k_sweep_block64.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "research" / "walltps_ab"

BLOCK = 64
# K -> per-arm result dir holding that K's block64 candidate arm.
K_ARMS = {6: "k6_block64", 7: "k7_block64", 8: "k8_block64", 9: "k9_block64"}
# The anchor (K7-block16) lives in the k7_block64 arm's *baseline* slot.
ANCHOR_ARM = "k7_block64"
LAWINE_K7_BLOCK16_REF = 454.338  # PR #138 / lawine #90 locked reference.
# A fresh server recompiling a novel-K graph runs noticeably longer than the warm ~90s.
COLD_READY_S = 120.0
# #72/#82 operative REAL/NULL bar at N=3.
OP_THRESHOLD_PCT = 0.10


def _load(arm_dir: str) -> dict | None:
    p = SWEEP / arm_dir / "paired_ab.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _warm_median(records: list[dict]) -> tuple[float | None, int]:
    warm = [r["wall_tps"] for r in records
            if r.get("wall_tps") is not None
            and (r.get("server_ready_s") or 0) <= COLD_READY_S]
    n_cold = len(records) - len(warm)
    if not warm:
        return None, n_cold
    return statistics.median(warm), n_cold


def _step_time_ms(wall_tps: float | None, e_accept: float | None) -> float | None:
    """Effective per-verify-step wall time. throughput = accepted_tok / wall_s and
    steps = accepted_tok / E[accept] => step_time_s = E[accept] / wall_tps."""
    if not wall_tps or not e_accept:
        return None
    return 1000.0 * e_accept / wall_tps


def main() -> int:
    anchor = _load(ANCHOR_ARM)
    if not anchor:
        print("[k64] no anchor arm yet under", SWEEP / ANCHOR_ARM)
        return 1

    base = anchor["arms"]["baseline"]          # K7-block16 anchor
    k7b64 = anchor["arms"]["candidate"]        # K7-block64 reference

    anchor_med = base["wall_tps"]["median"]
    anchor_eacc = base["e_accept_exact"]["mean"]
    k7b64_med = k7b64["wall_tps"]["median"]
    k7b64_eacc = k7b64["e_accept_exact"]["mean"]

    # Anchor reproduction self-check.
    anchor_rel_err = 100.0 * abs(anchor_med - LAWINE_K7_BLOCK16_REF) / LAWINE_K7_BLOCK16_REF
    print("\n===== PR #138 block64 K-sweep — wall_tps vs K =====")
    print(f"anchor K7-block16 median = {anchor_med:.3f} "
          f"(lawine ref {LAWINE_K7_BLOCK16_REF:.3f}, Δ{anchor_rel_err:.3f}%, "
          f"{'REPRODUCES' if anchor_rel_err <= 2.0 else 'OUT-OF-BAND ±2%!!'})")
    # E[accept] block-invariance check at K7 (bit-identical proposals => identical accept).
    eacc_rel = (100.0 * abs(k7b64_eacc - anchor_eacc) / anchor_eacc) if anchor_eacc else float("nan")
    print(f"E[accept] block-invariance @K7: block16={anchor_eacc:.4f} block64={k7b64_eacc:.4f} "
          f"(Δ{eacc_rel:.3f}%, {'INVARIANT' if eacc_rel <= 0.5 else 'DRIFTED?!'})")
    print(f"K7-block64 reference median = {k7b64_med:.3f}\n")

    # Per-K block64 rows.
    rows = []
    for K, arm_dir in sorted(K_ARMS.items()):
        if K == 7:
            med, eacc, recs = k7b64_med, k7b64_eacc, k7b64["records"]
        else:
            d = _load(arm_dir)
            if not d:
                rows.append({"K": K, "pending": True})
                continue
            cand = d["arms"]["candidate"]
            med, eacc, recs = cand["wall_tps"]["median"], cand["e_accept_exact"]["mean"], cand["records"]
        warm, n_cold = _warm_median(recs)
        rows.append({
            "K": K, "median": med, "warm_median": warm, "n_cold": n_cold,
            "e_accept": eacc, "step_time_ms": _step_time_ms(med, eacc),
            "d_vs_b16": 100.0 * (med - anchor_med) / anchor_med,
            "d_vs_k7b64": 100.0 * (med - k7b64_med) / k7b64_med,
            "ready_s": [round(r.get("server_ready_s", 0)) for r in recs],
            "vals": [round(v, 2) for v in (recs and [r["wall_tps"] for r in recs] or [])],
        })

    print(f"{'K':>3}  {'block':>5}  {'wall_tps':>9}  {'Δ% K7b16':>9}  {'Δ% K7b64':>9}  "
          f"{'verdict':>7}  {'E[accept]':>9}  {'step_ms':>8}  {'warm':>9}  cold  ready_s")
    done = [r for r in rows if not r.get("pending")]
    for r in rows:
        if r.get("pending"):
            print(f"{r['K']:>3}  {BLOCK:>5}  {'PENDING':>9}")
            continue
        v = "REF" if r["K"] == 7 else ("REAL" if abs(r["d_vs_k7b64"]) >= OP_THRESHOLD_PCT else "NULL")
        wm = f"{r['warm_median']:.3f}" if r.get("warm_median") is not None else "—"
        print(f"{r['K']:>3}  {BLOCK:>5}  {r['median']:>9.3f}  {r['d_vs_b16']:>+8.3f}%  "
              f"{r['d_vs_k7b64']:>+8.3f}%  {v:>7}  {r['e_accept']:>9.4f}  "
              f"{r['step_time_ms']:>8.4f}  {wm:>9}  {r['n_cold']:>4}  {r['ready_s']}")

    if not done:
        print("\n[k64] no completed arms yet.")
        return 0

    # GATE: K* and whether it is REAL-better than K7-block64.
    best = max(done, key=lambda r: r["median"])
    k7 = next(r for r in done if r["K"] == 7)
    better = [r for r in done if r["K"] != 7
              and (r["median"] - k7["median"]) / k7["median"] * 100.0 >= OP_THRESHOLD_PCT]
    print()
    if better:
        win = max(better, key=lambda r: r["median"])
        print(f">>> GATE: K* SHIFTED — K={win['K']} is REAL-better than K7 at block64 "
              f"(+{win['d_vs_k7b64']:.3f}% vs K7b64, median {win['median']:.3f}). "
              f"k_star_block64={win['K']}")
    else:
        print(f">>> GATE: K=7 STILL OPTIMAL at block64 — best K={best['K']} "
              f"(median {best['median']:.3f}); no K beats K7-block64 by >= {OP_THRESHOLD_PCT:.2f}%. "
              f"k_star_block64=7")
    print(f">>> primary k_optimal_wall_tps_block64 = {best['median']:.3f} (K={best['K']})")
    print(f">>> K7-block64 reference wall_tps = {k7['median']:.3f}")
    print(f">>> block16->block64 step-time effect @K7 = {k7['d_vs_b16']:+.3f}% wall_tps "
          f"({_step_time_ms(anchor_med, anchor_eacc):.4f} -> {k7['step_time_ms']:.4f} ms/step)")
    n_pending = sum(1 for r in rows if r.get("pending"))
    if n_pending:
        print(f"\n[note] {n_pending} arm(s) PENDING — gate provisional until all complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
