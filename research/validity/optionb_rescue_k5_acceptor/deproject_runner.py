#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #642 (stark) -- de-project #636's recompute acceptor: measure the REAL served
wall-TPS cost of the gap-flagged M=1 recompute, not the additive projection.

#636 projected ``rescued_wall_tps = 1/(1/152.291 + ftr/126.378)`` -- assuming each
recompute adds cleanly to the amortized spec loop AND costs ``1/126.378`` (the WRONG
checkpoint: that is the int4_g128_lmhead AR-rung forward, not the w4a16-ct target
width-1 forward the acceptor actually runs). This runner MEASURES the real hit by
serving ``int4_mtp_batchinv`` (the Option-B spec stack) UNCHANGED except for two
env levers the submission already honors:

  * ``SENPAI_REFERENCE_MODE=1``  -> arm (d): spec OFF, w4a16-ct **M=1 AR** served.
    Its median wall_tps is the *true full-context per-recompute width-1 forward
    cost* (corrects #636's g128 assumption) AND its decode jsonl is R_served, the
    served M=1 AR trajectory the make-or-break identity scan walks.
  * ``SENPAI_RECOMPUTE_RATE=r``  -> rate arms: the recompute-acceptor SPEED patch
    fires ``r * emitted`` real width-1 target forwards into the live spec loop, so
    the int4 GEMM weight-read + CUDA-graph-break / serialization the projection
    ignored is paid on the wall clock. ``r=0`` reproduces the un-rescued ceiling
    (arm c); ``r in {0.05,0.10,0.20}`` is the slope sweep that fits the real
    in-loop marginal cost C; ``r=flag_trigger_rate`` is the de-projected acceptor
    (arm a).

The measurement itself is the validated PR #72/#82 protocol REUSED verbatim
(``run_arm`` -> ``timed_decode`` -> median wall_tps over N fresh serves, conc=1,
output_len 512, 128 prompts, ``wall_tps = completion_tokens/decode_s``); the only
new logic here is the arm loop + the additive-cost slope fit. analysis_only,
official_tps=0, NO HF Job.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.profiler.paired_tps_ab import ArmSpec, run_arm, arm_stats  # noqa: E402

LOCKED_319_AR_TPS = 126.378          # AR rung official a10g-small (PR #4)
STARK_636_UNRESCUED_CEILING = 152.291  # #636 spec ceiling anchor (K=7 analysis)
STARK_636_PROJECTED = 139.20           # #636 projected rescued wall_tps (+10.1%)


def parse_env(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--extra-env expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v
    return out


def build_args(a) -> SimpleNamespace:
    """The minimal arg surface run_arm/build_serve_env/timed_decode read."""
    return SimpleNamespace(
        n=a.n, num_prompts=a.num_prompts, output_len=a.output_len, seed=a.seed,
        clock_interval_ms=a.clock_interval_ms, settle_s=a.settle_s, steptime=True,
    )


def median_wall_tps(stats: dict[str, Any]) -> float | None:
    w = stats.get("wall_tps") or {}
    return w.get("median")


def fit_additive_cost(rate_to_tps: dict[float, float]) -> dict[str, Any] | None:
    """Fit the additive serving model ``1/wall_tps(r) = 1/tps0 + C*r`` by OLS on the
    (rate, 1/wall_tps) points. C [sec/recompute, normalized per emitted token] is the
    REAL in-loop marginal recompute cost -- the number #636 assumed to be 1/126.378.
    Returns the fit + the predicted wall_tps at the un-rescued ceiling (r=0)."""
    pts = sorted(rate_to_tps.items())
    if len(pts) < 2:
        return None
    xs = [r for r, _ in pts]
    ys = [1.0 / t for _, t in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    slope = sxy / sxx            # C  (sec/recompute per token)
    intercept = my - slope * mx  # 1/tps0
    # R^2
    yhat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    tps0_fit = (1.0 / intercept) if intercept > 0 else None
    return {
        "model": "1/wall_tps = 1/tps0 + C*rate",
        "C_sec_per_recompute": slope,
        "intercept_inv_tps0": intercept,
        "tps0_fit": tps0_fit,
        "r2": r2,
        "points": [{"rate": r, "wall_tps": t, "inv_wall_tps": 1.0 / t} for r, t in pts],
        # #636 assumed C = 1/126.378 (g128 AR-rung forward). Ratio>1 => real cost
        # exceeds the projection's assumption => projection was optimistic.
        "C_over_636_assumption": slope * LOCKED_319_AR_TPS,
    }


def predict_acceptor_tps(tps0: float, cost_per_recompute: float, ftr: float) -> float:
    """Additive serving model evaluated at the acceptor's flag-trigger rate."""
    return 1.0 / (1.0 / tps0 + cost_per_recompute * ftr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default="int4_mtp_batchinv")
    ap.add_argument("--mode", choices=["reference", "rate"], required=True,
                    help="reference: one SENPAI_REFERENCE_MODE=1 arm (d / R_served). "
                         "rate: one SENPAI_RECOMPUTE_RATE arm per --rates value.")
    ap.add_argument("--rates", default="0.0,0.05,0.10,0.20",
                    help="comma rates for --mode rate (slope sweep). r=0 == un-rescued ceiling.")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="KEY=VALUE applied to every arm (e.g. NUM_SPECULATIVE_TOKENS=5 for K=5)")
    ap.add_argument("--n", type=int, default=3, help="fresh serves per arm (median-of-N)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--ftr", type=float, default=None,
                    help="flag_trigger_rate (from the identity scan) -> predict arm (a) tps")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="optionb-rescue-deproject-stark")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[deproject] {note}", flush=True)

    extra = parse_env(a.extra_env)
    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build arms.
    arms: list[ArmSpec] = []
    if a.mode == "reference":
        arms.append(ArmSpec("ref_d", a.submission, {"SENPAI_REFERENCE_MODE": "1", **extra}))
    else:
        rates = [float(x) for x in a.rates.split(",") if x.strip() != ""]
        for r in rates:
            lbl = f"r{r:g}".replace(".", "p")
            env = {**extra}
            if r > 0:
                env["SENPAI_RECOMPUTE_RATE"] = repr(r)
            arms.append(ArmSpec(lbl, a.submission, env))

    sub_dir = (ROOT / "submissions" / a.submission).resolve()
    if not sub_dir.exists():
        raise SystemExit(f"submission not found: {sub_dir}")
    manifest = harness.load_manifest(sub_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    rargs = build_args(a)
    print(f"[deproject] mode={a.mode} submission={a.submission} extra_env={extra} "
          f"arms={[arm.label for arm in arms]} n={a.n} "
          f"workload={a.num_prompts}x{a.output_len} -> {out_dir}", flush=True)

    t0 = time.time()
    per_arm: dict[str, dict[str, Any]] = {}
    rate_to_tps: dict[float, float] = {}
    records_path = out_dir / "records.jsonl"
    with open(records_path, "w") as fh:
        for arm in arms:
            recs = run_arm(arm, rargs, server_python, out_dir, fh)
            st = arm_stats(recs)
            med = median_wall_tps(st)
            ea = (st.get("e_accept_exact") or {}).get("mean")
            per_arm[arm.label] = {
                "label": arm.label, "override_env": arm.override_env,
                "wall_tps_median": med,
                "wall_tps_cv_pct": (st.get("wall_tps") or {}).get("cv_pct"),
                "wall_tps_values": (st.get("wall_tps") or {}).get("values"),
                "e_accept_exact_mean": ea,
                "n": st.get("wall_tps", {}).get("n"),
            }
            if a.mode == "rate" and "SENPAI_RECOMPUTE_RATE" in arm.override_env:
                rate_to_tps[float(eval(arm.override_env["SENPAI_RECOMPUTE_RATE"]))] = med
            elif a.mode == "rate":
                rate_to_tps[0.0] = med
            print(f"[deproject] arm {arm.label}: median wall_tps={med:.3f} "
                  f"cv={per_arm[arm.label]['wall_tps_cv_pct']} E[accept]={ea}", flush=True)
    elapsed = time.time() - t0

    fit = fit_additive_cost(rate_to_tps) if a.mode == "rate" else None
    acceptor_pred = None
    if fit and a.ftr is not None and fit["tps0_fit"]:
        acceptor_pred = {
            "ftr": a.ftr,
            "wall_tps_from_slope": predict_acceptor_tps(
                fit["tps0_fit"], fit["C_sec_per_recompute"], a.ftr),
        }

    result = {
        "pr": 642, "leg": f"deproject-{a.mode}", "analysis_only": True,
        "official_tps": 0, "no_hf_job": True,
        "submission": a.submission, "extra_env": extra,
        "n": a.n, "workload": {"num_prompts": a.num_prompts, "output_len": a.output_len,
                               "seed": a.seed},
        "elapsed_s": elapsed,
        "arms": per_arm,
        "rate_to_wall_tps": {str(k): v for k, v in sorted(rate_to_tps.items())},
        "additive_cost_fit": fit,
        "acceptor_prediction": acceptor_pred,
        "anchors": {"locked_ar_tps": LOCKED_319_AR_TPS,
                    "stark_636_unrescued_ceiling": STARK_636_UNRESCUED_CEILING,
                    "stark_636_projected": STARK_636_PROJECTED},
    }
    result_path = out_dir / f"deproject_{a.mode}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[deproject] ===== {a.mode} done in {elapsed/60:.1f} min =====", flush=True)
    for lbl, info in per_arm.items():
        print(f"  {lbl:10s} wall_tps={info['wall_tps_median']} "
              f"E[accept]={info['e_accept_exact_mean']}", flush=True)
    if fit:
        print(f"  additive fit: C={fit['C_sec_per_recompute']:.6e} sec/recompute "
              f"(={fit['C_over_636_assumption']:.3f}x the #636 1/126.378 assumption) "
              f"tps0_fit={fit['tps0_fit']:.3f} R2={fit['r2']:.4f}", flush=True)
    if acceptor_pred:
        print(f"  acceptor @ ftr={acceptor_pred['ftr']}: "
              f"wall_tps_from_slope={acceptor_pred['wall_tps_from_slope']:.3f}", flush=True)
    print(f"[deproject] artifacts -> {result_path}", flush=True)

    if not a.no_wandb:
        _log_wandb(a, result)
    return 0


def _log_wandb(a, result: dict[str, Any]) -> None:
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] import failed: {exc!r}; JSON only", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.wandb_name or f"stark/deproject-{a.mode}", group=a.wandb_group,
        notes=f"PR#642 de-project #636 recompute acceptor ({a.mode}): real served wall_tps cost",
        config={"pr": 642, "mode": a.mode, "submission": a.submission,
                "extra_env": result["extra_env"], "n": a.n,
                "num_prompts": a.num_prompts, "output_len": a.output_len},
    )
    if run is None:
        print("[wandb] disabled; JSON only", flush=True)
        return
    summary: dict[str, Any] = {}
    for lbl, info in result["arms"].items():
        if isinstance(info.get("wall_tps_median"), (int, float)):
            summary[f"arm/{lbl}/wall_tps"] = info["wall_tps_median"]
        if isinstance(info.get("e_accept_exact_mean"), (int, float)):
            summary[f"arm/{lbl}/e_accept"] = info["e_accept_exact_mean"]
    fit = result.get("additive_cost_fit")
    if fit:
        summary["fit/C_sec_per_recompute"] = fit["C_sec_per_recompute"]
        summary["fit/C_over_636_assumption"] = fit["C_over_636_assumption"]
        summary["fit/tps0_fit"] = fit["tps0_fit"]
        summary["fit/r2"] = fit["r2"]
    ap_ = result.get("acceptor_prediction")
    if ap_:
        summary["acceptor/ftr"] = ap_["ftr"]
        summary["acceptor/wall_tps_from_slope"] = ap_["wall_tps_from_slope"]
    for k, v in summary.items():
        run.summary[k] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
