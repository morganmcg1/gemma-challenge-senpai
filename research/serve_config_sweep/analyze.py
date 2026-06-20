#!/usr/bin/env python3
"""PR #811 sweep analysis: per-knob TPS delta table + byte-exact parity verdict.

Reads research/serve_config_sweep/results.jsonl. Control = label 'control'.
Median TPS per label (drops the FIRST rep of the session as the cold cache/clock
warmup, like the #797 rep0-drop convention, unless only one rep exists). Byte-exact
parity = variant parity_hash == control reference parity_hash. Cap rule (#784):
a knob is a WIN only if byte-exact AND median TPS delta >= +1% vs control.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path("/workspace/senpai/target/research/serve_config_sweep/results.jsonl")


def main() -> int:
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("ok")]
    by: dict[str, list[dict]] = {}
    for r in ok:
        by.setdefault(r["label"], []).append(r)

    def warm_tps(runs: list[dict]) -> list[float]:
        runs = sorted(runs, key=lambda r: r["rep"])
        warm = runs[1:] if len(runs) > 1 else runs  # drop rep0 (cold) when >1 rep
        return [r["tps"] for r in warm if r.get("tps")]

    if "control" not in by:
        print("no control runs yet")
        return 0
    ctrl_runs = sorted(by["control"], key=lambda r: r["rep"])
    ctrl_tps = warm_tps(ctrl_runs)
    ctrl_med = statistics.median(ctrl_tps) if ctrl_tps else 0.0
    ctrl_ref_hash = ctrl_runs[0]["parity_hash"]  # rep0 reference token stream
    # parity self-consistency across control reps:
    ctrl_parity_consistent = len({r["parity_hash"] for r in ctrl_runs}) == 1

    print(f"CONTROL: median_tps={ctrl_med:.4f}  warm_reps={ctrl_tps}  "
          f"all_reps_tps={[r['tps'] for r in ctrl_runs]}  "
          f"parity_consistent_across_reps={ctrl_parity_consistent}")
    if ctrl_tps:
        spread = (max(ctrl_tps) - min(ctrl_tps)) / ctrl_med * 100 if ctrl_med else 0
        print(f"CONTROL warm-rep spread: {spread:.2f}% of median  (noise band)")
    print(f"CONTROL ref parity_hash: {ctrl_ref_hash[:16]}  ppl={ctrl_runs[-1].get('ppl')}")
    print()
    hdr = f"{'label':14} {'n':>2} {'med_tps':>9} {'dTPS':>7} {'d%':>7} {'byte_exact':>10} {'ppl':>7}  knob"
    print(hdr); print("-" * len(hdr))
    for label in sorted(by):
        runs = by[label]
        tps_list = warm_tps(runs)
        med = statistics.median(tps_list) if tps_list else 0.0
        d = med - ctrl_med
        dp = (d / ctrl_med * 100) if ctrl_med else 0.0
        be = all(r["parity_hash"] == ctrl_ref_hash for r in runs)
        ppl = next((r.get("ppl") for r in reversed(runs) if r.get("ppl") is not None), None)
        knob = json.dumps(runs[0].get("config", {})) or "(control)"
        win = "  <== WIN" if (be and label != "control" and dp >= 1.0) else ""
        print(f"{label:14} {len(runs):>2} {med:>9.3f} {d:>+7.3f} {dp:>+6.2f}% "
              f"{str(be):>10} {str(ppl):>7}  {knob}{win}")
    print()
    print("WIN criterion (#784): byte_exact==True AND d% >= +1.00%. Else null/within-noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
