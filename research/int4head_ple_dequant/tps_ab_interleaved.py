#!/usr/bin/env python
"""PR #805 Step 3 (tie-breaker) — INTERLEAVED decode-TPS A/B.

The back-to-back harness (tps_ab.py) measures all pledequant reps, THEN all
control reps. If the GPU drifts (thermal) over the ~45-min run, the arm measured
second is systematically penalized -> a confound when the true delta (~+3%) is
comparable to the cross-run substrate variance (~2.6% observed: run1 pledequant
median 258.66 vs rerun ~265.5).

This variant INTERLEAVES: rep1 pledequant, rep1 control, rep2 pledequant, rep2
control, ... so both arms see the SAME slow thermal trajectory. The paired
per-rep delta (pledequant_i - control_i) then cancels any monotonic drift, and
the median of paired deltas is the drift-robust kernel-isolation estimate.

Workload identical to tps_ab.py: official 128 prompts x 512 tokens, conc=1,
CUDA graphs ON, measured_steady_gen_tps (vLLM's own whole-run engine meter).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ARMS = [
    ("pledequant", ROOT / "submissions" / "int4_mtp_bi0_int4head_pledequant"),
    ("int4head_control", ROOT / "submissions" / "int4_mtp_bi0_int4head"),
]
OUT = ROOT / "research" / "int4head_ple_dequant" / "tps_interleaved"
OUT.mkdir(parents=True, exist_ok=True)


def cv(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return 100.0 * statistics.pstdev(vals) / statistics.fmean(vals)


def measure(label: str, sub: Path, server_python: str, rep: int) -> tuple[float, float]:
    out_dir = OUT / label / f"rep{rep}"
    t0 = time.time()
    print(f"\n##### {label} rep{rep}/{REPS} -> {out_dir} #####", flush=True)
    report = serve_profile.run(
        sub, server_python, out_dir,
        num_prompts=paths.NUM_PROMPTS, output_len=paths.OUTPUT_LEN,
        kernel_window_tokens=256, variants=["frontier"], do_kernel=False,
        wandb_name=None, wandb_group="bi0-int4head-ple-dequant",
    )
    a = report["analysis"]
    tps = float(a["tps"]["measured_steady_gen_tps"])
    ea = float(a["e_accept"])
    print(f"[tps] {label} rep{rep}: steady_gen_tps={tps:.2f}  E_accept={ea:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return tps, ea


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[tps] {note}", flush=True)
    server_python = harness.ensure_server_venv(
        harness.load_manifest(ARMS[0][1])["dependencies"]
    )
    print(f"[tps] INTERLEAVED  server_python={server_python}  reps={REPS}", flush=True)

    results: dict[str, list[float]] = {a: [] for a, _ in ARMS}
    eaccept: dict[str, list[float]] = {a: [] for a, _ in ARMS}
    paired: list[dict] = []
    for rep in range(1, REPS + 1):
        rep_tps: dict[str, float] = {}
        for label, sub in ARMS:  # pledequant then control, each rep
            tps, ea = measure(label, sub, server_python, rep)
            results[label].append(tps)
            eaccept[label].append(ea)
            rep_tps[label] = tps
        d = rep_tps["pledequant"] - rep_tps["int4head_control"]
        dp = 100.0 * d / rep_tps["int4head_control"]
        paired.append({"rep": rep, "pledequant": rep_tps["pledequant"],
                       "control": rep_tps["int4head_control"],
                       "delta_abs": d, "delta_pct": dp})
        print(f"[pair] rep{rep}: pledequant={rep_tps['pledequant']:.2f}  "
              f"control={rep_tps['int4head_control']:.2f}  "
              f"delta={d:+.2f} ({dp:+.2f}%)", flush=True)

    print("\n" + "=" * 64)
    print("INTERLEAVED DECODE TPS A/B (measured_steady_gen_tps, tok/s)")
    print("=" * 64)
    summary = {"reps": REPS, "paired": paired}
    for label, _ in ARMS:
        v = results[label]
        summary[label] = {
            "reps": v, "median": statistics.median(v), "mean": statistics.fmean(v),
            "cv_pct": cv(v),
            "e_accept_median": statistics.median(eaccept[label]),
        }
        print(f"  {label:18s} reps={['%.2f'%x for x in v]}  "
              f"median={summary[label]['median']:.2f}  CV={summary[label]['cv_pct']:.2f}%  "
              f"E_accept~{summary[label]['e_accept_median']:.3f}")

    paired_deltas = [p["delta_abs"] for p in paired]
    paired_pcts = [p["delta_pct"] for p in paired]
    med_pair_abs = statistics.median(paired_deltas)
    med_pair_pct = statistics.median(paired_pcts)
    # median-of-medians delta (back-to-back style) for cross-check
    med_delta_abs = summary["pledequant"]["median"] - summary["int4head_control"]["median"]
    med_delta_pct = 100.0 * med_delta_abs / summary["int4head_control"]["median"]
    summary["paired_delta_median_abs"] = med_pair_abs
    summary["paired_delta_median_pct"] = med_pair_pct
    summary["paired_delta_mean_pct"] = statistics.fmean(paired_pcts)
    summary["median_of_medians_delta_pct"] = med_delta_pct
    print("-" * 64)
    print(f"  PAIRED delta (drift-cancelled): median={med_pair_abs:+.2f} tok/s "
          f"({med_pair_pct:+.2f}%)  mean={summary['paired_delta_mean_pct']:+.2f}%")
    print(f"  median-of-medians delta (cross-check) = {med_delta_abs:+.2f} tok/s "
          f"({med_delta_pct:+.2f}%)")
    print(f"  PR projection +5.3%; confirm gate >= +4%; cap if < +2%")
    (OUT / "tps_ab_interleaved_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[tps] summary -> {OUT / 'tps_ab_interleaved_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
