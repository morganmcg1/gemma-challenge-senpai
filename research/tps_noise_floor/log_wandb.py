"""Log the TPS noise-floor deliverable (PR #72) to W&B as ONE rich run.

Decoupled from ``run_noise_floor.py`` on purpose: the timing loop is expensive
(~50 min) and a W&B hiccup must never cost a re-run, so wandb logging is a
separate replay over the saved artifacts. Reads:

  * ``<mode>_n<N>/noise_floor_<mode>.json``  (per-run records + aggregate)
  * ``analysis.json``                        (decomposition + warmup + MDE)

and emits a single grouped run with per-run series, the warmup / MDE / median-of-n
tables, the headline summary scalars (primary: ``tps_noise_floor_cv``), and the
analysis + protocol artifacts.

Run under the repo .venv (has wandb). ``import wandb`` is done FIRST so the real
package is cached in sys.modules before the repo root (which holds a ./wandb run
dir that would otherwise shadow the import) is added to the path.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import wandb  # noqa: F401  (import first to win over the ./wandb shadow dir)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402


def _windowed(series: list[float] | None, w: int) -> float | None:
    s = [float(x) for x in (series or []) if isinstance(x, (int, float))]
    return statistics.fmean(s[w:]) if len(s) > w else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise-floor", type=Path,
                    default=ROOT / "research/tps_noise_floor/fresh_n12/noise_floor_fresh.json")
    ap.add_argument("--analysis", type=Path,
                    default=ROOT / "research/tps_noise_floor/analysis.json")
    ap.add_argument("--protocol", type=Path,
                    default=ROOT / "research/tps_noise_floor/PROTOCOL.md")
    ap.add_argument("--name", default="lawine/tps-noise-floor")
    ap.add_argument("--group", default="tps-noise-floor")
    args = ap.parse_args(argv)

    nf = json.loads(args.noise_floor.read_text())
    analysis = json.loads(args.analysis.read_text())
    mode = nf.get("mode", "fresh")
    records = nf["records"]
    agg = nf["aggregate"]
    a = analysis["modes"][mode]
    best_w = a["best_w_head"]

    run = wandb_logging.init_wandb_run(
        job_type="tps-noise-floor",
        agent="lawine",
        name=args.name,
        group=args.group,
        tags=["tps-noise-floor", mode, nf.get("submission", "fa2sw_precache_kenyan"),
              "measurement-protocol"],
        notes="TPS measurement noise-floor characterization + hardened protocol (PR #72)",
        config={
            "submission": nf.get("submission"),
            "mode": mode,
            "n_runs": nf.get("n_runs"),
            **nf.get("workload", {}),
            "steptime": nf.get("steptime"),
            "recommended_w_head": best_w,
            "recommended_metric": "wall_tps",
        },
    )
    if run is None:
        print("[log_wandb] wandb disabled (no API key / WANDB_DISABLED); nothing logged")
        return 1

    # ---- per-run time series (step = run_idx) ----
    for rec in records:
        series = rec.get("gen_tps_series") or []
        clock = rec.get("clock") or {}
        metrics = {
            "run/steady_gen_tps_mean": rec.get("steady_gen_tps_mean"),
            "run/wall_tps": rec.get("wall_tps"),
            "run/windowed_w%d" % best_w: _windowed(series, best_w),
            "run/first_interval_tps": series[0] if series else None,
            "run/e_accept_exact": rec.get("e_accept_exact"),
            "run/decode_duration_s": rec.get("decode_duration_s"),
            "run/server_ready_s": rec.get("server_ready_s"),
            "run/sm_clock_mhz_load": (clock.get("sm_clock_mhz_load") or {}).get("mean"),
            "run/temp_c_max": (clock.get("temp_c") or {}).get("max"),
            "run/power_w_load": (clock.get("power_w_load") or {}).get("mean"),
            "run/n_intervals": len(series),
        }
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb_logging.log_event(run, "noise_run", step=rec["run_idx"], metrics=metrics)

    # ---- tables ----
    per_run_tbl = wandb.Table(columns=[
        "run_idx", "steady_gen_tps_mean", "wall_tps", f"windowed_w{best_w}",
        "first_interval_tps", "e_accept_exact", "temp_c_max", "sm_clock_mhz"])
    for rec in records:
        series = rec.get("gen_tps_series") or []
        clock = rec.get("clock") or {}
        per_run_tbl.add_data(
            rec["run_idx"], rec.get("steady_gen_tps_mean"), rec.get("wall_tps"),
            _windowed(series, best_w), series[0] if series else None,
            rec.get("e_accept_exact"), (clock.get("temp_c") or {}).get("max"),
            (clock.get("sm_clock_mhz_load") or {}).get("mean"))

    warm_tbl = wandb.Table(columns=["w_head", "mean_tps", "cv_pct"])
    for row in a["warmup"]["sweep"]:
        warm_tbl.add_data(row["w_head"], row["mean_tps"], row["cv_pct"])

    mde_tbl = wandb.Table(columns=[
        "n_per_arm", "wall_mde_pct", "wall_mde_pow_pct",
        "windowed_mde_pct", "windowed_mde_pow_pct"])
    wall_levels = {lv["n_per_arm"]: lv for lv in a["mde_wall_median"]["levels"]}
    win_levels = {lv["n_per_arm"]: lv for lv in a["mde_median"]["levels"]}
    for n in sorted(set(wall_levels) | set(win_levels)):
        w_lv, i_lv = wall_levels.get(n, {}), win_levels.get(n, {})
        mde_tbl.add_data(n, w_lv.get("mde_interleaved_pct"), w_lv.get("mde_interleaved_pow_pct"),
                         i_lv.get("mde_interleaved_pct"), i_lv.get("mde_interleaved_pow_pct"))

    run.log({"tables/per_run": per_run_tbl, "tables/warmup_sweep": warm_tbl,
             "tables/mde": mde_tbl, "global_step": nf.get("n_runs", len(records))})

    # ---- headline summary scalars ----
    nf_raw = a["noise_floor"]["steady_gen_tps_mean_raw"]
    nf_wall = a["noise_floor"]["wall_tps"]
    nf_win = a["noise_floor"]["windowed_best_w"]
    warm = a["warmup"]
    drift = a["drift_windowed"]
    tok = a["token_nondeterminism"]
    mde_wall_n1 = a["mde_wall_median"]["levels"][0]
    summary = {
        # PRIMARY METRIC: residual CV after the recommended protocol (wall_tps)
        "tps_noise_floor_cv": nf_wall.get("cv_pct"),
        "tps_noise_floor_cv_windowed_w%d" % best_w: nf_win.get("cv_pct"),
        "tps_noise_floor_cv_raw_steady": nf_raw.get("cv_pct"),
        "mde_wall_pct_n1": mde_wall_n1.get("mde_interleaved_pct"),
        "mde_wall_pct_n1_powered": mde_wall_n1.get("mde_interleaved_pow_pct"),
        "wall_tps_mean": nf_wall.get("mean"),
        "wall_tps_std": nf_wall.get("std"),
        "wall_tps_range_pct": nf_wall.get("range_pct"),
        "steady_raw_mean": nf_raw.get("mean"),
        "steady_raw_range_pct": nf_raw.get("range_pct"),
        "warmup_deficit_pct": warm.get("warmup_deficit_pct"),
        "warmup_recommended_w": best_w,
        "drift_slope_tps_per_run": (drift.get("tps_vs_runidx") or {}).get("slope"),
        "drift_r": (drift.get("tps_vs_runidx") or {}).get("r"),
        "sm_clock_drift_mhz_per_min": (drift.get("sm_clock_vs_wallmin") or {}).get("slope"),
        "temp_drift_c_per_min": (drift.get("temp_vs_wallmin") or {}).get("slope"),
        "e_accept_cv_pct": (tok.get("e_accept") or {}).get("cv_pct"),
        "pearson_tps_vs_e_accept": tok.get("pearson_tps_vs_e_accept"),
        "pr56_steady_swing_pct": 4.42,
        "pr56_wall_swing_pct": 0.01,
    }
    summary = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    wandb_logging.log_summary(run, summary, step=nf.get("n_runs", len(records)))

    # ---- artifacts ----
    wandb_logging.log_json_artifact(run, name="tps_noise_floor_analysis",
                                    artifact_type="analysis", data=analysis)
    wandb_logging.log_json_artifact(run, name="tps_noise_floor_records",
                                    artifact_type="noise-floor", data=nf)
    if args.protocol.exists():
        wandb_logging.log_file_artifact(run, path=args.protocol,
                                        name="tps_measurement_protocol",
                                        artifact_type="protocol")

    wandb_logging.finish_wandb(run)
    print(f"[log_wandb] logged run '{args.name}' (group={args.group}); "
          f"tps_noise_floor_cv={summary.get('tps_noise_floor_cv'):.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
