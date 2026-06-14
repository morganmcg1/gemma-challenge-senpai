"""Paired ``wall_tps`` A/B runner — operationalizes the PR #72 protocol (PR #82).

LOCAL-only, contract-neutral. READS ``submissions/<dir>`` but never writes to a
served file. For each of two arms (baseline, candidate) it serves the submission
UNCHANGED except for optional *serve-time env config-overrides* (e.g.
``MAX_NUM_BATCHED_TOKENS=2048``), runs N fresh decode-only timed runs, computes the
**median ``wall_tps``** per arm, the paired delta %, a CI, and prints a **verdict
against the #72 MDE table**.

Why ``wall_tps``: PR #72 proved the team's status-quo local A/B metric
(``steady_gen_tps_mean``, the vLLM interval-meter mean) is *fragile* (CV 0.33%, a
fake 4.4% same-config swing) AND off-spec, whereas
``wall_tps = num_completion_tokens / decode_duration_s`` is the **official
leaderboard ``output_throughput`` definition** and has CV 0.035% (~125x tighter).
This runner makes that protocol a one-command paired comparison so every
lever-builder (land #71 tree-verify, denken #81 prompt-lookup, stark #78 GEMM
fusion, ubel #36 int4 lm_head) decides on the same correct number.

Protocol (research/tps_noise_floor/PROTOCOL.md, PR #72):
  * Metric: **median ``wall_tps`` of N fresh runs**, decode-only (the PPL validity
    pass is run separately so ``prompt_logprobs`` never perturbs the timing window).
  * Operative decision thresholds (conservative, team-adopted, ~2x the raw powered
    MDE): a delta is **REAL** at ``>= 0.20%`` with N=1, ``>= 0.10%`` with N>=3;
    below that it is **NULL / within noise**.
  * Sequential fresh-per-run is unbiased here: PR #72 measured across-run drift at
    0.000 tps/run (A10G SM clock pinned 1710 MHz) and showed server restart adds no
    measurable throughput variance, so ``--reuse-baseline-from`` (below) is valid.

Reuses the #72 harness end-to-end (``timed_decode`` / ``build_serve_env`` /
``preflight_gpu`` / ``aggregate`` + the analyze MDE stats); it does NOT reinvent the
measurement.

Run under the repo ``.venv`` (has wandb); serve/decode subprocs use the submission's
own serve venv. Canonical copy-paste entrypoints (these are what PROTOCOL.md points
to)::

    # A=B self-null (proves the runner is unbiased -> delta ~= 0, verdict NULL):
    .venv/bin/python scripts/profiler/paired_tps_ab.py \
        --baseline fa2sw_precache_kenyan --candidate fa2sw_precache_kenyan \
        --n 3 --wandb-name lawine/ab-selfnull --wandb-group walltps-ab-runner

    # config-override candidate (same served files, serve-time env change only):
    .venv/bin/python scripts/profiler/paired_tps_ab.py \
        --baseline fa2sw_precache_kenyan --candidate fa2sw_precache_kenyan \
        --candidate-env MAX_NUM_BATCHED_TOKENS=2048 \
        --n 3 --wandb-name lawine/ab-mbt2048 --wandb-group walltps-ab-runner

    # reuse a prior baseline measurement (valid: restart-invariant per #72) so a
    # re-screen of several candidates does not re-run the baseline each time:
    ... --reuse-baseline-from research/walltps_ab/ab-selfnull/paired_ab.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
# Reuse the #72 measurement harness verbatim -- do NOT reinvent the measurement.
from research.tps_noise_floor.run_noise_floor import (  # noqa: E402
    aggregate,
    build_serve_env,
    preflight_gpu,
    timed_decode,
)
from research.tps_noise_floor.analyze_noise_floor import (  # noqa: E402
    Z_DETECT,
    Z_POWER,
    bootstrap_stat_cv,
)

OUT_ROOT = ROOT / "research" / "walltps_ab"

# Canonical wall_tps per-run noise floor characterized at N=12 in PR #72
# (research/tps_noise_floor/PROTOCOL.md). Used as the prior for the MDE so a 3-run
# A/B does not have to re-estimate sigma from 3 noisy points, and as the yardstick
# for the per-arm floor sanity check below.
WALL_TPS_FLOOR_CV_PCT = 0.035
# An arm whose own N-run CV exceeds this many x the characterized floor is flagged
# as a measurement anomaly (the canonical MDE no longer applies to it -> re-run).
FLOOR_ANOMALY_FACTOR = 5.0


# ---------------------------------------------------------------------------
# Arm specification
# ---------------------------------------------------------------------------
@dataclass
class ArmSpec:
    """One side of the A/B: a submission served with optional serve-time env
    config-overrides. The served *files* are never modified; overrides are merged
    on top of the manifest env at ``LocalServer`` launch (exactly how the manifest
    itself ships ``MAX_NUM_BATCHED_TOKENS`` etc.), so the comparison stays
    contract-neutral."""

    label: str
    submission: str
    override_env: dict[str, str] = field(default_factory=dict)

    @property
    def submission_dir(self) -> Path:
        return (ROOT / "submissions" / self.submission).resolve()

    def describe(self) -> str:
        if not self.override_env:
            return self.submission
        ov = ",".join(f"{k}={v}" for k, v in sorted(self.override_env.items()))
        return f"{self.submission}[{ov}]"


def parse_env_overrides(items: list[str] | None) -> dict[str, str]:
    """``["KEY=VAL", ...]`` -> ``{KEY: VAL}``. A config-override is a serve-time env
    var the submission's serve.py already consumes (e.g. ``MAX_NUM_BATCHED_TOKENS``,
    ``SPECULATIVE_CONFIG``); the runner does not validate the key against the
    submission, it just sets it in the server env."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--*-env expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"--*-env has empty key: {item!r}")
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Run one arm (N fresh decode-only runs)
# ---------------------------------------------------------------------------
def run_arm(arm: ArmSpec, args, server_python: Path, out_dir: Path,
            records_fh) -> list[dict[str, Any]]:
    """N fresh-server, decode-only timed runs of ``arm``. Mirrors
    ``run_noise_floor.run_fresh`` (fresh server per run = the operationally-relevant
    A/B floor) but tags each record with the arm + its config-override and never
    runs PPL inside the timing loop."""
    arm_dir = out_dir / arm.label
    arm_dir.mkdir(parents=True, exist_ok=True)
    # Measurement env (DISABLE_LOG_STATS=0 + native sampler + steptime, matching the
    # canonical serve_profile / #56 regime) with the arm's config-override on top.
    serve_env = build_serve_env(args)
    serve_env.update(arm.override_env)

    records: list[dict[str, Any]] = []
    for i in range(args.n):
        server_log = arm_dir / f"server_run{i:02d}.log"
        print(f"\n[ab:{arm.label}] === run {i+1}/{args.n} ({arm.describe()}) ===", flush=True)
        preflight_gpu()
        t_load0 = time.time()
        with harness.LocalServer(arm.submission_dir, server_python=server_python,
                                 log_path=server_log, extra_env=serve_env) as server:
            server_ready_s = time.time() - t_load0
            rec = timed_decode(
                server, server_python, arm_dir, i,
                num_prompts=args.num_prompts, output_len=args.output_len,
                seed=args.seed, log_offset=0,
                clock_interval_ms=args.clock_interval_ms, settle_s=args.settle_s,
            )
            rec["server_ready_s"] = server_ready_s
        rec["arm"] = arm.label
        rec["submission"] = arm.submission
        rec["override_env"] = dict(arm.override_env)
        records.append(rec)
        records_fh.write(json.dumps(rec) + "\n")
        records_fh.flush()
        _print_run(arm.label, rec)
    return records


def _print_run(label: str, rec: dict[str, Any]) -> None:
    wall = rec.get("wall_tps")
    print(
        f"[ab:{label}] run {rec['run_idx']:02d}: wall_tps={wall:.3f} "
        f"steady={rec.get('steady_gen_tps_mean')} E[accept]={rec.get('e_accept_exact')} "
        f"ready={rec.get('server_ready_s', 0):.0f}s "
        f"sm={(rec.get('clock') or {}).get('sm_clock_mhz_load', {}).get('mean')}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Per-arm stats + paired verdict
# ---------------------------------------------------------------------------
def arm_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "wall_tps": aggregate(records, "wall_tps"),
        "steady_gen_tps_mean": aggregate(records, "steady_gen_tps_mean"),
        "e_accept_exact": aggregate(records, "e_accept_exact"),
        "server_ready_s": aggregate(records, "server_ready_s"),
    }


def operative_threshold_pct(n: int) -> float:
    """The conservative, team-adopted REAL/NULL bar from PR #72: >=0.20% at N=1,
    >=0.10% at N>=3 (N=2 interpolated to 0.15%). These sit ~2x above the raw powered
    MDE as a safety margin, and are the thresholds the PR #82 instructions name."""
    if n <= 1:
        return 0.20
    if n == 2:
        return 0.15
    return 0.10


def _median_se_tps(values: list[float], n: int) -> float | None:
    """Bootstrap SE of the median-of-n estimator from the observed runs (the #72
    estimator). Weak at n=3 but a useful observed cross-check on the floor prior."""
    boot = bootstrap_stat_cv(values, n, stat="median")
    return boot["se"] if boot else None


def paired_verdict(base: dict[str, Any], cand: dict[str, Any], n: int) -> dict[str, Any]:
    """Paired wall_tps verdict: median delta vs the operative #72 threshold, plus an
    observed-variance CI and a floor sanity check.

    Decision (primary): ``|delta_median_pct| >= operative_threshold_pct(n)`` -> REAL,
    else NULL. The MDE is a *property of the protocol* (the characterized 0.035% floor
    -> the 0.20%/0.10% operative bar), not re-derived from 3 noisy points; the observed
    CI and floor check are reported alongside so an anomalously noisy arm is caught."""
    wa, wc = base["wall_tps"], cand["wall_tps"]
    med_a, med_c = wa.get("median"), wc.get("median")
    mean_a, mean_c = wa.get("mean"), wc.get("mean")
    std_a, std_c = wa.get("std", 0.0), wc.get("std", 0.0)
    vals_a, vals_c = wa.get("values") or [], wc.get("values") or []

    if not med_a or not med_c:
        return {"error": "missing wall_tps medians", "verdict": "ERROR"}

    delta_med_tps = med_c - med_a
    delta_med_pct = 100.0 * delta_med_tps / med_a
    delta_mean_pct = (100.0 * (mean_c - mean_a) / mean_a) if mean_a else None

    op_thresh = operative_threshold_pct(n)
    verdict = "REAL" if abs(delta_med_pct) >= op_thresh else "NULL"

    # Raw powered MDE from the characterized per-run floor (the protocol prior):
    # SE(mean-of-n) ~ floor_sigma / sqrt(n); paired -> x sqrt(2); detect/powered z.
    floor_sigma_tps = med_a * WALL_TPS_FLOOR_CV_PCT / 100.0
    se_diff_floor = floor_sigma_tps / math.sqrt(n) * math.sqrt(2.0)
    raw_mde_detect_pct = 100.0 * Z_DETECT * se_diff_floor / med_a
    raw_mde_powered_pct = 100.0 * (Z_DETECT + Z_POWER) * se_diff_floor / med_a

    # Observed-variance two-sample CI (cross-check; uses THIS A/B's spread).
    se_mean_a = std_a / math.sqrt(n) if n else 0.0
    se_mean_c = std_c / math.sqrt(n) if n else 0.0
    se_diff_obs = math.hypot(se_mean_a, se_mean_c)
    ci95_tps = Z_DETECT * se_diff_obs
    ci95_pct = 100.0 * ci95_tps / med_a
    observed_significant = abs(med_c - med_a) > ci95_tps if se_diff_obs > 0 else None
    se_med_a = _median_se_tps(vals_a, n)
    se_med_c = _median_se_tps(vals_c, n)

    # Floor sanity: did either arm wander far above the characterized 0.035% CV?
    cv_a, cv_c = wa.get("cv_pct"), wc.get("cv_pct")
    max_cv = max([c for c in (cv_a, cv_c) if isinstance(c, (int, float))], default=None)
    floor_exceeded = (max_cv is not None
                      and max_cv > FLOOR_ANOMALY_FACTOR * WALL_TPS_FLOOR_CV_PCT)

    human = (
        f"A={med_a:.2f} B={med_c:.2f} Δ={delta_med_pct:+.3f}% [{verdict}] "
        f"(op@N{n}={op_thresh:.2f}%, raw-powered-MDE={raw_mde_powered_pct:.3f}%)"
    )
    return {
        "metric": "wall_tps",
        "n_per_arm": n,
        "baseline_median_wall_tps": med_a,
        "candidate_median_wall_tps": med_c,
        "baseline_mean_wall_tps": mean_a,
        "candidate_mean_wall_tps": mean_c,
        "delta_median_tps": delta_med_tps,
        "delta_median_pct": delta_med_pct,
        "delta_mean_pct": delta_mean_pct,
        "operative_threshold_pct": op_thresh,
        "raw_mde_detect_pct": raw_mde_detect_pct,
        "raw_mde_powered_pct": raw_mde_powered_pct,
        "floor_cv_pct": WALL_TPS_FLOOR_CV_PCT,
        "se_diff_floor_tps": se_diff_floor,
        "se_diff_observed_tps": se_diff_obs,
        "ci95_observed_pct": ci95_pct,
        "observed_significant": observed_significant,
        "median_se_baseline_tps": se_med_a,
        "median_se_candidate_tps": se_med_c,
        "baseline_cv_pct": cv_a,
        "candidate_cv_pct": cv_c,
        "floor_exceeded": floor_exceeded,
        "verdict": verdict,
        "human_verdict": human,
    }


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, baseline: ArmSpec, candidate: ArmSpec,
               base_recs: list[dict], cand_recs: list[dict],
               result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[ab] wandb_logging import failed ({exc}); skipping wandb", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="walltps-ab", agent="lawine",
            name=args.wandb_name or f"lawine/ab-{candidate.label}",
            group=args.wandb_group,
            tags=["walltps-ab-runner", baseline.submission, candidate.submission],
            config={
                "baseline": baseline.describe(),
                "candidate": candidate.describe(),
                "baseline_override_env": baseline.override_env,
                "candidate_override_env": candidate.override_env,
                "n": args.n, "num_prompts": args.num_prompts,
                "output_len": args.output_len, "seed": args.seed,
                "reused_baseline": bool(args.reuse_baseline_from),
            },
        )
    except Exception as exc:
        print(f"[ab] wandb init failed ({exc}); skipping wandb", flush=True)
        return
    if run is None:
        print("[ab] wandb disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return
    try:
        step = 0
        for label, recs in (("baseline", base_recs), ("candidate", cand_recs)):
            for rec in recs:
                metrics = {
                    f"{label}/wall_tps": rec.get("wall_tps"),
                    f"{label}/steady_gen_tps_mean": rec.get("steady_gen_tps_mean"),
                    f"{label}/e_accept_exact": rec.get("e_accept_exact"),
                    f"{label}/server_ready_s": rec.get("server_ready_s"),
                }
                metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb_logging.log_event(run, f"{label}_run", step=step, metrics=metrics)
                step += 1
        v = result["verdict"]
        flat = {f"verdict/{k}": val for k, val in v.items()
                if isinstance(val, (int, float, bool))}
        flat["verdict/is_real"] = 1.0 if v.get("verdict") == "REAL" else 0.0
        wandb_logging.log_summary(run, flat, step=step)
        wandb_logging.log_json_artifact(
            run, name=f"paired_ab_{candidate.label}", artifact_type="walltps-ab",
            data=result,
        )
    except Exception as exc:
        print(f"[ab] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _load_baseline_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    recs = (data.get("arms", {}).get("baseline", {}) or {}).get("records")
    if not recs:
        raise SystemExit(f"--reuse-baseline-from has no baseline records: {path}")
    print(f"[ab] reusing {len(recs)} baseline records from {path} "
          "(valid: restart-invariant per PR #72)", flush=True)
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="fa2sw_precache_kenyan",
                    help="baseline submission dir name under submissions/")
    ap.add_argument("--candidate", default=None,
                    help="candidate submission dir (default = baseline -> self-null)")
    ap.add_argument("--baseline-env", action="append", default=[],
                    help="serve-time config-override KEY=VALUE for the baseline arm (repeatable)")
    ap.add_argument("--candidate-env", action="append", default=[],
                    help="serve-time config-override KEY=VALUE for the candidate arm (repeatable)")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--candidate-label", default=None)
    ap.add_argument("--n", type=int, default=3, help="fresh runs per arm (median-of-N)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--steptime", dest="steptime", action="store_true", default=True,
                    help="enable the per-step STEPTIME probe (default; matches the canonical "
                         "serve_profile / #56 measurement env build_serve_env reuses)")
    ap.add_argument("--no-steptime", dest="steptime", action="store_false")
    ap.add_argument("--reuse-baseline-from", type=Path, default=None,
                    help="load the baseline arm's records from a prior paired_ab.json instead "
                         "of re-running it (valid here: PR #72 restart-invariance)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default=None, help="output subdir name (default = candidate label)")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="walltps-ab-runner")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[ab] {note}", flush=True)

    candidate_sub = args.candidate or args.baseline
    base_override = parse_env_overrides(args.baseline_env)
    cand_override = parse_env_overrides(args.candidate_env)
    cand_label = args.candidate_label or (
        "candidate" if (candidate_sub != args.baseline or not cand_override)
        else "_".join(f"{k}{v}" for k, v in sorted(cand_override.items()))[:40] or "candidate"
    )
    baseline = ArmSpec(args.baseline_label, args.baseline, base_override)
    candidate = ArmSpec(cand_label, candidate_sub, cand_override)

    for arm in (baseline, candidate):
        if not arm.submission_dir.exists():
            raise SystemExit(f"submission not found: {arm.submission_dir}")

    # One server venv per submission dependency set (shared if identical manifests).
    manifest = harness.load_manifest(baseline.submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    tag = args.tag or candidate.label
    out_dir = (args.out_dir or (OUT_ROOT / tag)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ab] baseline={baseline.describe()} candidate={candidate.describe()} "
          f"n={args.n} workload={args.num_prompts}x{args.output_len} seed={args.seed} -> {out_dir}",
          flush=True)

    t0 = time.time()
    records_path = out_dir / "records.jsonl"
    with open(records_path, "w") as records_fh:
        if args.reuse_baseline_from:
            base_recs = _load_baseline_records(args.reuse_baseline_from)
            for rec in base_recs:
                records_fh.write(json.dumps(rec) + "\n")
            records_fh.flush()
        else:
            base_recs = run_arm(baseline, args, server_python, out_dir, records_fh)
        cand_recs = run_arm(candidate, args, server_python, out_dir, records_fh)
    elapsed = time.time() - t0

    base_stats = arm_stats(base_recs)
    cand_stats = arm_stats(cand_recs)
    verdict = paired_verdict(base_stats, cand_stats, args.n)

    result = {
        "runner": "paired_tps_ab", "pr": 82, "metric": "wall_tps",
        "baseline": {"submission": baseline.submission, "label": baseline.label,
                     "override_env": baseline.override_env, "describe": baseline.describe()},
        "candidate": {"submission": candidate.submission, "label": candidate.label,
                      "override_env": candidate.override_env, "describe": candidate.describe()},
        "n": args.n,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed},
        "reused_baseline_from": str(args.reuse_baseline_from) if args.reuse_baseline_from else None,
        "elapsed_s": elapsed,
        "git": _git_info(),
        "arms": {
            "baseline": {**base_stats, "records": base_recs},
            "candidate": {**cand_stats, "records": cand_recs},
        },
        "verdict": verdict,
    }
    result_path = out_dir / "paired_ab.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))

    _print_summary(baseline, candidate, base_stats, cand_stats, verdict, elapsed)
    print(f"[ab] artifacts -> {result_path}", flush=True)
    _log_wandb(args, baseline, candidate, base_recs, cand_recs, result)
    return 0


def _git_info() -> dict[str, Any]:
    try:
        from scripts import wandb_logging
        return wandb_logging.git_info()
    except Exception:
        return {}


def _fmt(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "—"


def _print_summary(baseline: ArmSpec, candidate: ArmSpec, base: dict, cand: dict,
                   verdict: dict, elapsed: float) -> None:
    print(f"\n[ab] ===== PAIRED wall_tps A/B — {elapsed/60:.1f} min =====", flush=True)
    for label, arm, st in (("A baseline", baseline, base), ("B candidate", candidate, cand)):
        w = st["wall_tps"]
        print(f"  {label:12s} {arm.describe()}", flush=True)
        if w.get("n"):
            print(f"     wall_tps  n={w['n']} median={_fmt(w.get('median'),3)} "
                  f"mean={_fmt(w.get('mean'),3)} std={_fmt(w.get('std'),3)} "
                  f"CV={_fmt(w.get('cv_pct'),3)}% range={_fmt(w.get('min'),2)}..{_fmt(w.get('max'),2)}",
                  flush=True)
        ea = st["e_accept_exact"]
        if ea.get("n"):
            print(f"     E[accept] mean={_fmt(ea.get('mean'),4)} CV={_fmt(ea.get('cv_pct'),3)}%", flush=True)
    print(f"  ---- verdict: raw-powered-MDE={_fmt(verdict.get('raw_mde_powered_pct'))}% "
          f"op-threshold={_fmt(verdict.get('operative_threshold_pct'),2)}% "
          f"observed-CI95=±{_fmt(verdict.get('ci95_observed_pct'))}%", flush=True)
    if verdict.get("floor_exceeded"):
        print(f"  ---- WARN: an arm CV exceeded {FLOOR_ANOMALY_FACTOR}x the characterized "
              f"{WALL_TPS_FLOOR_CV_PCT}% floor — treat verdict with caution / re-run", flush=True)
    print(f"\n  >>> {verdict.get('human_verdict')}\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
