"""Empirical sigma_hw driver (PR #467) — measured served-TPS envelope vs the 1% convention.

LOCAL-only, MEASUREMENT + analysis ONLY. Serves the deployed
``fa2sw_precache_kenyan`` stack (the 481.53 baseline: MTP K=7 + M=8 decode,
ONEGRAPH=1, PR #52) **UNCHANGED** and runs the canonical 128-prompt x 512-token,
conc=1 decode N>=8 times in FRESH processes to measure the empirical run-to-run
served-TPS envelope (``summary.json:tps`` == ``wall_tps`` ==
num_completion_tokens / decode_duration_s, the official ``output_throughput``
definition — see PR #72 ``research/tps_noise_floor``).

NO served-file edit, NO kernel rebuild, NO HF job, NO submission, NO deploy.
This produces no new operating point — it measures the variance of code that
already exists, to replace the ASSERTED ``sigma_hw = 4.8153 TPS`` (= 1% x 481.53,
land #451) with a MEASURED envelope.

Clean-room hygiene per run (mirrors the N=7 read-peak hygiene of #463):
  * fresh CUDA context  : a brand-new server process per run (preflight reaps any
    lingering vLLM + waits for VRAM to drain before each load),
  * distinct seed       : run i uses seed = base_seed + i (the envelope is NOT
    artificially narrowed by pinning one seed; greedy decode is near-deterministic
    so this can only widen, never shrink, the measured band),
  * boost-clock warmup  : a small throwaway decode before each TIMED decode spins
    the SM clock to boost (1710 MHz on this A10G) and warms the batch=1 CUDA graph
    / KV path, so the timed window starts at steady clock (the official harness
    likewise discards 4 warmup requests).

Reuses the validated PR #72 primitives verbatim (``run_noise_floor.timed_decode``
/ ``build_serve_env`` / ``preflight_gpu`` / ``aggregate``) so the measured
``wall_tps`` is byte-for-byte the same statistic the fleet A/Bs on.

Run under the repo .venv (has wandb); serve/decode subprocs use the submission's
serve venv. Example::

    .venv/bin/python -m research.empirical_sigma_hw.run_sigma_hw \
        --submission fa2sw_precache_kenyan --n-runs 10 \
        --wandb-name lawine/empirical-sigma-hw \
        --wandb-group equivalence-escalation-anchors
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from research.tps_noise_floor import run_noise_floor as nf  # noqa: E402

OUT_ROOT = ROOT / "research" / "empirical_sigma_hw"

# --------------------------------------------------------------------------- #
# Program constants under test (PR #467 baseline section)                      #
# --------------------------------------------------------------------------- #
CONVENTION_SIGMA_HW = 4.8153          # = 1% x 481.53, land #451 (c675zor8) — UNDER TEST
DEPLOYED_OFFICIAL_TPS = 481.53        # PR #52 (2x9fm2zx), official a10g-small
STRICT_FRONTIER = 467.14              # blanket-strict realized frontier (denken #423)
STRICT_DEPLOYED_GAP = 14.39           # = 481.53 - 467.14
UNIFIED_CEILING = 510.87              # unified ceiling +-4.82
CEILING_REANCHOR_463 = 510.654        # my #463 re-anchored read-peak ceiling
MATERIALITY_BAR_TPS = 2.0             # the program's "+2 TPS materiality" bar
TAU_LO = 1.03524                      # local->official transfer (PR #267); official ~= tau_lo*local
LOCAL_ANCHOR_PR72 = 454.12            # PR #72 N=12 decode_outputs.py wall_tps mean (local A10G)
PPL_ANCHOR = 2.3772                   # official PPL anchor (local 2.3767)
PPL_CAP = 2.42                        # validity gate

# Two-sided 95% Student-t critical values by degrees of freedom (n-1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    # nearest tabulated df below, else the asymptotic z
    keys = sorted(k for k in _T95 if k <= df)
    return _T95[keys[-1]] if keys else 1.960


# --------------------------------------------------------------------------- #
# Peak GPU memory (single query; vLLM pre-allocates the KV pool at startup so   #
# memory.used is flat at the reserved peak through the whole decode).           #
# --------------------------------------------------------------------------- #
def _gpu_mem_used_mib() -> int | None:
    return nf._gpu_mem_used_mib()


# --------------------------------------------------------------------------- #
# Boost-clock warmup: a tiny throwaway decode to pin the SM clock + warm the    #
# batch=1 CUDA graph before the TIMED decode.                                   #
# --------------------------------------------------------------------------- #
def warmup_decode(server, runner_python: Path, out_dir: Path, run_idx: int,
                  *, num_prompts: int, output_len: int) -> float:
    warm_out = out_dir / "warmup" / f"run{run_idx:02d}.jsonl"
    warm_sum = out_dir / "warmup" / f"run{run_idx:02d}.summary.json"
    t0 = time.time()
    try:
        harness.capture_decode(
            runner_python, base_url=server.base_url, model=server.served_model_name,
            out_file=warm_out, summary_file=warm_sum,
            num_prompts=num_prompts, output_len=output_len, seed=0,
        )
    except Exception as exc:  # warmup must never abort a timed run
        print(f"[sigma] WARN warmup decode failed: {exc}", flush=True)
    return time.time() - t0


# --------------------------------------------------------------------------- #
# The fresh-per-run loop                                                        #
# --------------------------------------------------------------------------- #
def run_fresh_sigma(args, submission_dir: Path, server_python: Path, out_dir: Path,
                    records_fh) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int | None]:
    records: list[dict[str, Any]] = []
    ppl_result: dict[str, Any] | None = None
    peak_mem_mib: int | None = None
    for i in range(args.n_runs):
        seed = args.seed + i  # DISTINCT seed per run
        server_log = out_dir / f"server_fresh_run{i:02d}.log"
        print(f"\n[sigma] === run {i + 1}/{args.n_runs} (fresh server, seed={seed}) ===", flush=True)
        nf.preflight_gpu()
        t_load0 = time.time()
        with harness.LocalServer(submission_dir, server_python=server_python,
                                 log_path=server_log,
                                 extra_env=nf.build_serve_env(args)) as server:
            server_ready_s = time.time() - t_load0
            warm_s = warmup_decode(server, server_python, out_dir, i,
                                   num_prompts=args.warmup_prompts,
                                   output_len=args.warmup_len)
            rec = nf.timed_decode(
                server, server_python, out_dir, i,
                num_prompts=args.num_prompts, output_len=args.output_len,
                seed=seed, log_offset=nf._server_log_len(server_log),
                clock_interval_ms=args.clock_interval_ms, settle_s=args.settle_s,
            )
            rec["server_ready_s"] = server_ready_s
            rec["warmup_s"] = warm_s
            mem = _gpu_mem_used_mib()
            rec["mem_used_mib"] = mem
            if mem is not None:
                peak_mem_mib = mem if peak_mem_mib is None else max(peak_mem_mib, mem)
            # One PPL validity pass on the LAST fresh server (timing already
            # captured; PPL never perturbs a timed window).
            if i == args.n_runs - 1 and not args.no_ppl:
                ppl_result = _ppl_pass(server, server_python, out_dir)
        records.append(rec)
        records_fh.write(json.dumps(rec) + "\n")
        records_fh.flush()
        nf._print_run(rec)
    return records, ppl_result, peak_mem_mib


def _ppl_pass(server, server_python: Path, out_dir: Path) -> dict[str, Any] | None:
    try:
        print("\n[sigma] PPL validity pass (timed loop already done)...", flush=True)
        ppl = harness.run_ppl(
            server_python, base_url=server.base_url, model=server.served_model_name,
            out_file=out_dir / "ppl_check.jsonl",
            summary_file=out_dir / "ppl_check.summary.json",
        )
        print(f"[sigma] PPL: ppl={ppl.get('ppl')} num_records={ppl.get('num_records')}", flush=True)
        return ppl
    except Exception as exc:
        print(f"[sigma] WARN PPL validity pass failed: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------- #
# sigma_hw analysis + materiality restatement + self-test                      #
# --------------------------------------------------------------------------- #
def analyze(records: list[dict[str, Any]], ppl_result: dict[str, Any] | None,
            submission_clean: bool) -> dict[str, Any]:
    wall = [r["wall_tps"] for r in records
            if isinstance(r.get("wall_tps"), (int, float)) and r["wall_tps"] == r["wall_tps"]]
    n = len(wall)
    out: dict[str, Any] = {"n_served_repeats": n}
    if n < 2:
        out["error"] = "need n>=2 wall_tps samples"
        return out

    median = statistics.median(wall)
    mean = statistics.fmean(wall)
    sigma_hw_local = statistics.pstdev(wall)        # POPULATION sd == sigma_hw (per PR)
    sample_sd = statistics.stdev(wall)              # sample sd
    lo, hi = min(wall), max(wall)
    frac = sigma_hw_local / median if median else float("nan")     # empirical_sigma_hw_frac
    # 95% CI on the MEAN (sample sd / sqrt n, Student-t).
    se = sample_sd / math.sqrt(n)
    ci_half = t95(n - 1) * se
    # sigma_hw projected onto the OFFICIAL 481.53 operating point. The local->official
    # gap (PR #267) is a STABLE multiplicative scalar (hardware/clock+harness), so the
    # fractional run-to-run noise is scale-invariant and transfers: official sigma_hw
    # = frac * 481.53. (We cannot measure the official board's own envelope without
    # repeated HF jobs, which are out of scope; but 1% was never measured either.)
    empirical_sigma_hw_tps = frac * DEPLOYED_OFFICIAL_TPS

    # --- the 1% convention test ---
    convention_frac = CONVENTION_SIGMA_HW / DEPLOYED_OFFICIAL_TPS   # 0.01 by construction
    ratio_to_convention = frac / convention_frac if convention_frac else float("nan")
    signed_drift_tps = empirical_sigma_hw_tps - CONVENTION_SIGMA_HW  # <0 => measured tighter
    signed_drift_frac = frac - convention_frac
    # "convention holds" iff the measured fraction is within a factor of 2 of 1%.
    sigma_hw_convention_holds = bool(0.5 * convention_frac <= frac <= 2.0 * convention_frac)

    # --- materiality restatements under the MEASURED envelope ---
    strict_gap_in_empirical_sigma = STRICT_DEPLOYED_GAP / empirical_sigma_hw_tps \
        if empirical_sigma_hw_tps else float("nan")
    strict_gap_in_convention_sigma = STRICT_DEPLOYED_GAP / CONVENTION_SIGMA_HW
    ceiling_delta = abs(UNIFIED_CEILING - CEILING_REANCHOR_463)     # 0.216
    ceiling_delta_in_empirical_sigma = ceiling_delta / empirical_sigma_hw_tps \
        if empirical_sigma_hw_tps else float("nan")
    # two-estimate (combined) uncertainty = sqrt(2)*sigma for two independent reads
    ceiling_delta_in_combined_sigma = ceiling_delta / (math.sqrt(2) * empirical_sigma_hw_tps) \
        if empirical_sigma_hw_tps else float("nan")
    # PR's "hold within sigma_hw" == within 1 sigma_hw of the per-estimate band.
    ceiling_holds_under_empirical_sigma = bool(ceiling_delta <= empirical_sigma_hw_tps)
    materiality_bar_in_empirical_sigma = MATERIALITY_BAR_TPS / empirical_sigma_hw_tps \
        if empirical_sigma_hw_tps else float("nan")
    materiality_bar_in_convention_sigma = MATERIALITY_BAR_TPS / CONVENTION_SIGMA_HW
    recommended_bar_3sigma_tps = 3.0 * empirical_sigma_hw_tps

    ppl = ppl_result.get("ppl") if ppl_result else None

    # --- self-test ---
    checks: dict[str, bool] = {
        "n_ge_8": n >= 8,
        "all_runs_128x512": all(int(r.get("num_completion_tokens", 0)) == paths.NUM_PROMPTS * paths.OUTPUT_LEN
                                for r in records),
        "median_reproduces_local_anchor_2pct":
            abs(median - LOCAL_ANCHOR_PR72) / LOCAL_ANCHOR_PR72 < 0.02,
        "sigma_hw_frac_finite_pos": math.isfinite(frac) and frac > 0,
        "empirical_sigma_hw_tps_roundtrip":
            math.isfinite(empirical_sigma_hw_tps)
            and abs(empirical_sigma_hw_tps - frac * DEPLOYED_OFFICIAL_TPS) < 1e-9,
        "strict_gap_material_gt_3sigma":
            math.isfinite(strict_gap_in_empirical_sigma) and strict_gap_in_empirical_sigma > 3.0,
        "wall_tps_nan_clean": all(math.isfinite(v) for v in wall),
        "no_served_file_change": submission_clean,
        "ppl_within_cap": (ppl is not None and ppl <= PPL_CAP and abs(ppl - PPL_ANCHOR) < 0.02),
    }
    sigma_hw_self_test_passes = all(checks.values())

    out.update({
        "empirical_served_tps_median": median,
        "empirical_served_tps_mean": mean,
        "empirical_sigma_hw_local_tps": sigma_hw_local,
        "empirical_sigma_hw_sample_sd_local_tps": sample_sd,
        "empirical_served_tps_min": lo,
        "empirical_served_tps_max": hi,
        "empirical_served_tps_range_tps": hi - lo,
        "empirical_served_tps_ci95_halfwidth_tps": ci_half,
        "empirical_served_tps_ci95": [mean - ci_half, mean + ci_half],
        "empirical_sigma_hw_frac": frac,
        "empirical_sigma_hw_frac_pct": 100.0 * frac,
        "empirical_sigma_hw_tps": empirical_sigma_hw_tps,
        "convention_sigma_hw": CONVENTION_SIGMA_HW,
        "convention_sigma_hw_frac": convention_frac,
        "empirical_over_convention_ratio": ratio_to_convention,
        "convention_over_empirical_ratio": (1.0 / ratio_to_convention) if ratio_to_convention else float("nan"),
        "sigma_hw_signed_drift_tps": signed_drift_tps,
        "sigma_hw_signed_drift_frac": signed_drift_frac,
        "sigma_hw_convention_holds": sigma_hw_convention_holds,
        # materiality restatements
        "strict_deployed_gap_tps": STRICT_DEPLOYED_GAP,
        "strict_gap_in_empirical_sigma": strict_gap_in_empirical_sigma,
        "strict_gap_in_convention_sigma": strict_gap_in_convention_sigma,
        "ceiling_delta_tps": ceiling_delta,
        "ceiling_delta_in_empirical_sigma": ceiling_delta_in_empirical_sigma,
        "ceiling_delta_in_combined_sigma": ceiling_delta_in_combined_sigma,
        "ceiling_holds_under_empirical_sigma": ceiling_holds_under_empirical_sigma,
        "materiality_bar_tps": MATERIALITY_BAR_TPS,
        "materiality_bar_in_empirical_sigma": materiality_bar_in_empirical_sigma,
        "materiality_bar_in_convention_sigma": materiality_bar_in_convention_sigma,
        "recommended_materiality_bar_3sigma_tps": recommended_bar_3sigma_tps,
        # transfer + anchors
        "tau_lo_local_to_official": TAU_LO,
        "median_projected_official_tps": median * TAU_LO,
        "local_anchor_pr72_wall_tps": LOCAL_ANCHOR_PR72,
        # ppl + validity
        "ppl": ppl,
        "ppl_cap": PPL_CAP,
        "no_served_file_change": submission_clean,
        "no_submission": True,
        # self-test
        "self_test_checks": checks,
        "sigma_hw_self_test_passes": sigma_hw_self_test_passes,
    })
    return out


# --------------------------------------------------------------------------- #
# git cleanliness of the served submission dir                                 #
# --------------------------------------------------------------------------- #
def _submission_clean(submission: str) -> bool:
    try:
        rel = f"submissions/{submission}"
        r = subprocess.run(["git", "status", "--porcelain", "--", rel],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and r.stdout.strip() == ""
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# wandb                                                                         #
# --------------------------------------------------------------------------- #
def _to_num(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    return v


def _log_wandb(args, records: list[dict[str, Any]], agg: dict[str, Any],
               analysis: dict[str, Any], peak_mem_mib: int | None) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[sigma] wandb_logging import failed ({exc}); skipping wandb", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="empirical-sigma-hw", agent="lawine",
            name=args.wandb_name or "lawine/empirical-sigma-hw",
            group=args.wandb_group,
            tags=["empirical-sigma-hw", "equivalence-escalation-anchors", args.submission],
            config={
                "submission": args.submission, "n_runs": args.n_runs,
                "num_prompts": args.num_prompts, "output_len": args.output_len,
                "base_seed": args.seed, "distinct_seeds": True,
                "warmup_prompts": args.warmup_prompts, "warmup_len": args.warmup_len,
                "convention_sigma_hw": CONVENTION_SIGMA_HW,
                "deployed_official_tps": DEPLOYED_OFFICIAL_TPS,
            },
        )
    except Exception as exc:
        print(f"[sigma] wandb init failed ({exc}); skipping wandb", flush=True)
        return None
    if run is None:
        print("[sigma] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        for rec in records:
            metrics = {
                "run/wall_tps": rec.get("wall_tps"),
                "run/steady_gen_tps_mean": rec.get("steady_gen_tps_mean"),
                "run/e_accept_exact": rec.get("e_accept_exact"),
                "run/decode_duration_s": rec.get("decode_duration_s"),
                "run/seed": rec.get("seed"),
                "run/server_ready_s": rec.get("server_ready_s"),
                "run/warmup_s": rec.get("warmup_s"),
                "run/sm_clock_mhz_load": (rec.get("clock") or {}).get("sm_clock_mhz_load", {}).get("mean"),
                "run/temp_c_max": (rec.get("clock") or {}).get("temp_c", {}).get("max"),
                "run/mem_used_mib": rec.get("mem_used_mib"),
            }
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, "sigma_run", step=rec["run_idx"], metrics=metrics)
        # Flat summary: aggregate stats + the full analysis (bools -> int).
        flat: dict[str, Any] = {}
        for mkey, a in agg.items():
            for stat in ("mean", "std", "cv_pct", "min", "max", "range_pct", "median"):
                if isinstance(a, dict) and isinstance(a.get(stat), (int, float)):
                    flat[f"{mkey}/{stat}"] = a[stat]
        for k, v in analysis.items():
            if isinstance(v, bool):
                flat[k] = int(v)
            elif isinstance(v, (int, float)) and math.isfinite(v):
                flat[k] = v
        for k, v in analysis.get("self_test_checks", {}).items():
            flat[f"self_test/{k}"] = int(bool(v))
        if peak_mem_mib is not None:
            flat["peak_mem_mib"] = peak_mem_mib
        wandb_logging.log_summary(run, flat, step=args.n_runs)
        wandb_logging.log_json_artifact(
            run, name="empirical_sigma_hw", artifact_type="sigma-hw",
            data={"records": records, "aggregate": agg, "analysis": analysis,
                  "peak_mem_mib": peak_mem_mib},
        )
    except Exception as exc:
        print(f"[sigma] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


# --------------------------------------------------------------------------- #
# print                                                                         #
# --------------------------------------------------------------------------- #
def _print_analysis(a: dict[str, Any]) -> None:
    print("\n[sigma] ============ EMPIRICAL sigma_hw ============", flush=True)
    print(f"  n_served_repeats          = {a.get('n_served_repeats')}", flush=True)
    print(f"  median served wall_tps    = {a.get('empirical_served_tps_median'):.4f} (local A10G)", flush=True)
    print(f"  range [min,max]           = [{a.get('empirical_served_tps_min'):.4f}, "
          f"{a.get('empirical_served_tps_max'):.4f}]  (= {a.get('empirical_served_tps_range_tps'):.4f} TPS)", flush=True)
    print(f"  sigma_hw (pstdev, local)  = {a.get('empirical_sigma_hw_local_tps'):.4f} TPS", flush=True)
    print(f"  sample-sd (local)         = {a.get('empirical_sigma_hw_sample_sd_local_tps'):.4f} TPS", flush=True)
    print(f"  95% CI halfwidth (mean)   = +-{a.get('empirical_served_tps_ci95_halfwidth_tps'):.4f} TPS", flush=True)
    print(f"  empirical_sigma_hw_frac   = {a.get('empirical_sigma_hw_frac_pct'):.4f} %", flush=True)
    print(f"  empirical_sigma_hw_tps    = {a.get('empirical_sigma_hw_tps'):.4f} TPS  (frac x 481.53)", flush=True)
    print(f"  convention_sigma_hw       = {a.get('convention_sigma_hw'):.4f} TPS  (1% x 481.53)", flush=True)
    print(f"  convention/empirical      = {a.get('convention_over_empirical_ratio'):.1f}x  "
          f"(convention is this many x too loose)", flush=True)
    print(f"  sigma_hw_convention_holds = {a.get('sigma_hw_convention_holds')}", flush=True)
    print(f"  signed drift              = {a.get('sigma_hw_signed_drift_tps'):+.4f} TPS "
          f"({a.get('sigma_hw_signed_drift_frac')*100:+.4f} pp)", flush=True)
    print("  -- materiality under the MEASURED envelope --", flush=True)
    print(f"  strict 14.39 gap          = {a.get('strict_gap_in_empirical_sigma'):.1f} sigma "
          f"(was {a.get('strict_gap_in_convention_sigma'):.2f} sigma on convention)", flush=True)
    print(f"  ceiling delta 0.216       = {a.get('ceiling_delta_in_empirical_sigma'):.2f} sigma "
          f"(combined {a.get('ceiling_delta_in_combined_sigma'):.2f} sigma); "
          f"holds_within_1sigma={a.get('ceiling_holds_under_empirical_sigma')}", flush=True)
    print(f"  +2 TPS materiality bar    = {a.get('materiality_bar_in_empirical_sigma'):.1f} sigma "
          f"(was {a.get('materiality_bar_in_convention_sigma'):.2f} sigma on convention)", flush=True)
    print(f"  recommended 3sigma bar    = {a.get('recommended_materiality_bar_3sigma_tps'):.3f} TPS", flush=True)
    print(f"  ppl                       = {a.get('ppl')}  (cap {a.get('ppl_cap')})", flush=True)
    print(f"  no_served_file_change     = {a.get('no_served_file_change')}", flush=True)
    print(f"  sigma_hw_self_test_passes = {a.get('sigma_hw_self_test_passes')}", flush=True)
    if not a.get("sigma_hw_self_test_passes"):
        print(f"    failing checks: {[k for k, v in a.get('self_test_checks', {}).items() if not v]}", flush=True)


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default="fa2sw_precache_kenyan")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED, help="BASE seed; run i uses seed+i")
    ap.add_argument("--warmup-prompts", type=int, default=8)
    ap.add_argument("--warmup-len", type=int, default=64)
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--steptime", dest="steptime", action="store_true", default=True)
    ap.add_argument("--no-steptime", dest="steptime", action="store_false")
    ap.add_argument("--no-ppl", action="store_true", help="skip the PPL validity pass")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default="lawine/empirical-sigma-hw")
    ap.add_argument("--wandb-group", default="equivalence-escalation-anchors")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[sigma] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    submission_clean = _submission_clean(args.submission)
    if not submission_clean:
        print(f"[sigma] WARN submissions/{args.submission} is NOT git-clean — "
              f"this run would NOT be on the unchanged deployed config!", flush=True)

    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[sigma] submission={submission_dir.name} server_python={server_python} "
          f"clean={submission_clean}", flush=True)

    out_dir = (args.out_dir or (OUT_ROOT / f"fresh_n{args.n_runs}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    print(f"[sigma] n_runs={args.n_runs} workload={args.num_prompts}x{args.output_len} "
          f"base_seed={args.seed} (distinct per run) -> {out_dir}", flush=True)

    t0 = time.time()
    with open(records_path, "w") as records_fh:
        records, ppl_result, peak_mem_mib = run_fresh_sigma(
            args, submission_dir, server_python, out_dir, records_fh)
    elapsed = time.time() - t0

    agg = {
        "wall_tps": nf.aggregate(records, "wall_tps"),
        "steady_gen_tps_mean": nf.aggregate(records, "steady_gen_tps_mean"),
        "e_accept_exact": nf.aggregate(records, "e_accept_exact"),
    }
    analysis = analyze(records, ppl_result, submission_clean)
    run_id = _log_wandb(args, records, agg, analysis, peak_mem_mib)
    analysis["wandb_run_id"] = run_id

    result = {
        "submission": args.submission,
        "n_runs": args.n_runs,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "base_seed": args.seed, "distinct_seeds": True},
        "warmup": {"prompts": args.warmup_prompts, "len": args.warmup_len},
        "elapsed_s": elapsed,
        "peak_mem_mib": peak_mem_mib,
        "ppl_result": ppl_result,
        "git": nf._git_info(),
        "aggregate": agg,
        "analysis": analysis,
        "records": records,
    }
    result_path = out_dir / "sigma_hw.json"
    result_path.write_text(json.dumps(result, indent=2))

    nf._print_summary("sigma_hw", agg, elapsed)
    _print_analysis(analysis)
    print(f"\n[sigma] peak_mem_mib={peak_mem_mib} wandb_run_id={run_id}", flush=True)
    print(f"[sigma] artifacts -> {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
