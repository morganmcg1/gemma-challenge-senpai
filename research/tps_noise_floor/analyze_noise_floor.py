"""Analyze the TPS noise-floor records (PR #72): variance decomposition +
hardened-protocol validation + minimum-detectable-effect (MDE).

Consumes one or more ``noise_floor_<mode>.json`` produced by
``run_noise_floor.py`` and emits ``analysis.json`` + a printed report covering:

  1. Empirical noise floor table (mean / std / CV / range, per mode & metric).
  2. Variance decomposition:
       (a) warmup transient   — first-interval throughput deficit vs steady,
       (b) steady-state jitter — within-run std of the steady-window intervals,
       (c) thermal/clock drift — per-run TPS & SM-clock vs run-index / wall-time
                                  (slope, r) + lag-1 autocorrelation,
       (d) token nondeterminism — Pearson(per-run steady TPS, per-run E[accept]).
  3. Protocol validation: warmup-discard W sweep, median-of-N bootstrap CV(n),
     and sequential-vs-interleaved A/B MDE under the measured drift.

Pure-stdlib (statistics + random); no numpy/scipy dependency.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

random.seed(12345)

# z for a two-sided test at alpha and the (1-beta) power add-on.
Z_DETECT = 1.959963985  # alpha=0.05 two-sided
Z_POWER = 0.841621234   # power=0.80 add-on


# ---------------------------------------------------------------------------
# stdlib stats
# ---------------------------------------------------------------------------
def _finite(xs: list[float]) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and x == x]


def cv_pct(xs: list[float]) -> float | None:
    xs = _finite(xs)
    if len(xs) < 2:
        return None
    m = statistics.fmean(xs)
    return 100.0 * statistics.stdev(xs) / m if m else None


def summarize(xs: list[float]) -> dict[str, Any]:
    xs = _finite(xs)
    if not xs:
        return {"n": 0}
    n = len(xs)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    s = sorted(xs)
    return {
        "n": n, "mean": m, "std": sd,
        "cv_pct": 100.0 * sd / m if m else None,
        "min": s[0], "max": s[-1],
        "range_pct": 100.0 * (s[-1] - s[0]) / m if m else None,
        "median": statistics.median(xs),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys)
             if isinstance(a, (int, float)) and isinstance(b, (int, float))
             and a == a and b == b]
    if len(pairs) < 3:
        return None
    xs2, ys2 = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = statistics.fmean(xs2), statistics.fmean(ys2)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / (sx * sy)


def linreg(xs: list[float], ys: list[float]) -> dict[str, Any] | None:
    """OLS slope/intercept/r for y ~ x."""
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys)
             if a == a and b == b]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = statistics.fmean([p[0] for p in pairs])
    my = statistics.fmean([p[1] for p in pairs])
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    if sxx == 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    slope = sxy / sxx
    return {"slope": slope, "intercept": my - slope * mx, "r": pearson(
        [p[0] for p in pairs], [p[1] for p in pairs]), "n": n}


def lag1_autocorr(xs: list[float]) -> float | None:
    xs = _finite(xs)
    if len(xs) < 4:
        return None
    m = statistics.fmean(xs)
    num = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(len(xs) - 1))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den else None


def bootstrap_stat_cv(values: list[float], n: int, stat: str = "median",
                      reps: int = 4000) -> dict[str, float] | None:
    """CV of the (median|mean)-of-n estimator, by resampling n runs w/ replacement."""
    values = _finite(values)
    if len(values) < 2 or n < 1:
        return None
    ests = []
    for _ in range(reps):
        sample = [random.choice(values) for _ in range(n)]
        ests.append(statistics.median(sample) if stat == "median" else statistics.fmean(sample))
    m = statistics.fmean(ests)
    sd = statistics.stdev(ests) if len(ests) > 1 else 0.0
    return {"n": n, "mean": m, "se": sd, "cv_pct": 100.0 * sd / m if m else None}


# ---------------------------------------------------------------------------
# windowed per-run TPS (warmup discard)
# ---------------------------------------------------------------------------
def windowed_tps(series: list[float], w_head: int, w_tail: int = 0) -> float | None:
    s = _finite(series or [])
    if not s:
        return None
    end = len(s) - w_tail if w_tail else len(s)
    win = s[w_head:end]
    return statistics.fmean(win) if win else None


def warmup_sweep(records: list[dict], max_w: int = 6, plateau_tol_pct: float = 0.5) -> dict[str, Any]:
    """Sweep warmup-discard W (intervals) and recommend the protocol window.

    Two criteria, because minimizing CV alone is a trap: the cold-start interval
    is LOW, and if it is *consistently* low across runs it can show a small across-
    run CV at W=0 while biasing the reported mean far below the true steady plateau
    (smoke: 400 vs 440). So we first find the smallest W whose windowed mean has
    reached the plateau (within ``plateau_tol_pct`` of the max windowed mean = the
    asymptotic steady value), then among W>=plateau_w pick the min-CV window. That
    yields an UNBIASED *and* tight steady-state estimator.
    """
    rows = []
    best = None  # global min-CV (kept for transparency; may sit inside warmup)
    for w in range(0, max_w + 1):
        per_run = _finite([windowed_tps(r.get("gen_tps_series"), w) for r in records])
        if len(per_run) < 2:
            continue
        cv = cv_pct(per_run)
        mean = statistics.fmean(per_run)
        rows.append({"w_head": w, "n": len(per_run), "mean_tps": mean, "cv_pct": cv})
        if cv is not None and (best is None or cv < best["cv_pct"]):
            best = {"w_head": w, "cv_pct": cv, "mean_tps": mean}

    # Plateau = the asymptotic steady mean (warmup only drags the mean DOWN, so the
    # max windowed mean across W is the unbiased steady value). Smallest W within
    # tol of it is where bias is gone.
    plateau_w, recommended = None, None
    if rows:
        plateau_mean = max(r["mean_tps"] for r in rows)
        thresh = plateau_mean * (1.0 - plateau_tol_pct / 100.0)
        for r in rows:
            if r["mean_tps"] >= thresh:
                plateau_w = r["w_head"]
                break
        if plateau_w is None:
            plateau_w = rows[-1]["w_head"]
        # min-CV among windows at/after the plateau -> the recommended protocol W
        for r in rows:
            if r["w_head"] >= plateau_w and r["cv_pct"] is not None:
                if recommended is None or r["cv_pct"] < recommended["cv_pct"]:
                    recommended = {"w_head": r["w_head"], "cv_pct": r["cv_pct"],
                                   "mean_tps": r["mean_tps"]}

    # Warmup deficit: first-interval throughput vs the recommended steady window.
    rw = (recommended or {}).get("w_head", plateau_w or 1)
    first_vals, steady_vals = [], []
    for r in records:
        s = _finite(r.get("gen_tps_series") or [])
        if len(s) > rw:
            first_vals.append(s[0])
            steady_vals.append(statistics.fmean(s[rw:]))
    deficit_pct = None
    if first_vals and steady_vals:
        mf, ms = statistics.fmean(first_vals), statistics.fmean(steady_vals)
        deficit_pct = 100.0 * (ms - mf) / ms if ms else None
    return {
        "sweep": rows, "best_min_cv": best,
        "plateau_w": plateau_w, "plateau_tol_pct": plateau_tol_pct,
        "recommended": recommended,
        "first_interval_mean": statistics.fmean(first_vals) if first_vals else None,
        "steady_mean_at_recommended_w": statistics.fmean(steady_vals) if steady_vals else None,
        "warmup_deficit_pct": deficit_pct,
    }


# ---------------------------------------------------------------------------
# drift + clock + token-nondeterminism
# ---------------------------------------------------------------------------
def _wall_minutes(records: list[dict]) -> list[float]:
    ts = []
    for r in records:
        try:
            ts.append(datetime.fromisoformat(r["t_start_utc"]).timestamp())
        except Exception:
            ts.append(float("nan"))
    if not ts or ts[0] != ts[0]:
        return [float(i) for i in range(len(records))]
    t0 = ts[0]
    return [(t - t0) / 60.0 for t in ts]


def _clock_load_mean(r: dict) -> float | None:
    return ((r.get("clock") or {}).get("sm_clock_mhz_load") or {}).get("mean")


def _temp_max(r: dict) -> float | None:
    return ((r.get("clock") or {}).get("temp_c") or {}).get("max")


def drift_analysis(records: list[dict], metric: str, w_head: int) -> dict[str, Any]:
    idx = [r.get("run_idx", i) for i, r in enumerate(records)]
    mins = _wall_minutes(records)
    tps = [windowed_tps(r.get("gen_tps_series"), w_head) if metric == "windowed"
           else r.get(metric) for r in records]
    sm = [_clock_load_mean(r) for r in records]
    temp = [_temp_max(r) for r in records]
    return {
        "metric": metric, "w_head": w_head,
        "tps_vs_runidx": linreg([float(i) for i in idx], tps),
        "tps_vs_wallmin": linreg(mins, tps),
        "tps_lag1_autocorr": lag1_autocorr(_finite(tps)),
        "sm_clock_vs_wallmin": linreg(mins, sm),
        "temp_vs_wallmin": linreg(mins, temp),
        "tps_vs_smclock_pearson": pearson(tps, sm),
        "tps_vs_temp_pearson": pearson(tps, temp),
        "wall_minutes_total": mins[-1] if mins else None,
        "tps_values": _finite(tps),
        "sm_clock_values": _finite(sm),
        "temp_values": _finite(temp),
    }


def token_nondeterminism(records: list[dict], w_head: int) -> dict[str, Any]:
    tps = [windowed_tps(r.get("gen_tps_series"), w_head) for r in records]
    eacc = [r.get("e_accept_exact") for r in records]
    return {
        "e_accept": summarize(eacc),
        "pearson_tps_vs_e_accept": pearson(tps, eacc),
        "note": ("if |r| is high and significant, run-to-run TPS swing is partly "
                 "driven by acceptance-pattern nondeterminism (algorithm-level, "
                 "partly irreducible); if ~0, the swing is hardware/scheduler timing."),
    }


# ---------------------------------------------------------------------------
# MDE: sequential vs interleaved
# ---------------------------------------------------------------------------
def mde_analysis(records: list[dict], w_head: int, stat: str = "median",
                 values: list[float] | None = None,
                 metric_name: str = "windowed") -> dict[str, Any]:
    """MDE for an A/B where each arm uses (stat)-of-n runs of a per-run TPS metric.

    By default the metric is the warmup-discarded windowed interval mean; pass
    ``values`` (one per run, e.g. ``wall_tps``) to compute the MDE on the
    recommended headline metric instead.
    """
    per_run = _finite(values) if values is not None else _finite(
        [windowed_tps(r.get("gen_tps_series"), w_head) for r in records])
    if len(per_run) < 2:
        return {"error": "insufficient runs"}
    mean = statistics.fmean(per_run)
    sigma = statistics.stdev(per_run)  # residual run-to-run std after warmup discard

    # drift slope per run (for the sequential bias term)
    idx = list(range(len(per_run)))
    lr = linreg([float(i) for i in idx], per_run)
    slope_per_run = abs(lr["slope"]) if lr else 0.0

    out_levels = []
    for n in (1, 2, 3, 4, 6, 8):
        boot = bootstrap_stat_cv(per_run, n, stat=stat)
        se_arm = boot["se"] if boot else sigma / math.sqrt(n)
        se_diff = se_arm * math.sqrt(2.0)
        # interleaved cancels slow drift -> random-only MDE
        mde_inter = Z_DETECT * se_diff
        mde_inter_pow = (Z_DETECT + Z_POWER) * se_diff
        # sequential (all-A-then-all-B): adds a drift bias over the n-run gap
        drift_bias = slope_per_run * n
        mde_seq = Z_DETECT * se_diff + drift_bias
        mde_seq_pow = (Z_DETECT + Z_POWER) * se_diff + drift_bias
        out_levels.append({
            "n_per_arm": n,
            "se_arm": se_arm, "se_diff": se_diff,
            "drift_bias_tps": drift_bias,
            "mde_interleaved_tps": mde_inter, "mde_interleaved_pct": 100.0 * mde_inter / mean,
            "mde_interleaved_pow_pct": 100.0 * mde_inter_pow / mean,
            "mde_sequential_tps": mde_seq, "mde_sequential_pct": 100.0 * mde_seq / mean,
            "mde_sequential_pow_pct": 100.0 * mde_seq_pow / mean,
        })
    return {
        "stat": stat, "w_head": w_head, "mean_tps": mean, "metric": metric_name,
        "residual_sigma": sigma, "residual_cv_pct": 100.0 * sigma / mean,
        "drift_slope_per_run_tps": slope_per_run,
        "levels": out_levels,
        "note": ("MDE = detectable A-B gap. interleaved (ABAB) cancels slow drift; "
                 "sequential (AAAA then BBBB) carries a drift bias = |slope|*n."),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def analyze_mode(records: list[dict]) -> dict[str, Any]:
    raw_steady = [r.get("steady_gen_tps_mean") for r in records]
    raw_steady_nz = [r.get("steady_gen_tps_mean_nonzero") for r in records]
    wall = [r.get("wall_tps") for r in records]
    warm = warmup_sweep(records)
    best_w = ((warm.get("recommended") or {}).get("w_head")
              if warm.get("recommended") else None)
    if best_w is None:
        best_w = warm.get("plateau_w") or 1

    # median-of-n CV curve on the windowed metric
    per_run_windowed = _finite([windowed_tps(r.get("gen_tps_series"), best_w) for r in records])
    median_curve = [bootstrap_stat_cv(per_run_windowed, n, stat="median")
                    for n in range(1, min(len(per_run_windowed), 8) + 1)]
    mean_curve = [bootstrap_stat_cv(per_run_windowed, n, stat="mean")
                  for n in range(1, min(len(per_run_windowed), 8) + 1)]
    # ...and on wall_tps (the recommended headline metric = the official
    # output_throughput definition: num_completion_tokens / decode duration_s)
    wall_vals = _finite([r.get("wall_tps") for r in records])
    wall_median_curve = [bootstrap_stat_cv(wall_vals, n, stat="median")
                         for n in range(1, min(len(wall_vals), 8) + 1)]

    return {
        "n_runs": len(records),
        "noise_floor": {
            "steady_gen_tps_mean_raw": summarize(raw_steady),
            "steady_gen_tps_mean_nonzero": summarize(raw_steady_nz),
            "wall_tps": summarize(wall),
            "windowed_best_w": summarize(per_run_windowed),
        },
        "best_w_head": best_w,
        "warmup": warm,
        "drift_windowed": drift_analysis(records, "windowed", best_w),
        "drift_raw_steady": drift_analysis(records, "steady_gen_tps_mean", 0),
        "token_nondeterminism": token_nondeterminism(records, best_w),
        "median_of_n_cv": median_curve,
        "mean_of_n_cv": mean_curve,
        "wall_median_of_n_cv": wall_median_curve,
        "mde_median": mde_analysis(records, best_w, stat="median"),
        "mde_mean": mde_analysis(records, best_w, stat="mean"),
        "mde_wall_median": mde_analysis(records, best_w, stat="median",
                                        values=wall_vals, metric_name="wall_tps"),
    }


def cross_mode_decomposition(modes: dict[str, Any]) -> dict[str, Any] | None:
    """Isolate the cold-start/restart variance component (PR #72 task 2a).

    fresh-mode = full operational noise (model-load + CUDA-graph capture + start
    thermal + steady jitter + token nondeterminism); reuse-mode = same MINUS the
    per-run restart. So var(fresh) - var(reuse) attributes the restart/cold-start
    share of the operational floor. Done on both the raw steady metric (the 428.37
    headline) and the warmup-discarded windowed metric.
    """
    if "fresh" not in modes or "reuse" not in modes:
        return None

    def _var(mode_key: str, metric_key: str) -> tuple[float | None, float | None, float | None]:
        s = modes[mode_key]["noise_floor"].get(metric_key, {})
        sd, mean = s.get("std"), s.get("mean")
        var = sd * sd if isinstance(sd, (int, float)) else None
        return var, sd, mean

    block: dict[str, Any] = {}
    for label, metric_key in (("raw_steady", "steady_gen_tps_mean_raw"),
                              ("windowed", "windowed_best_w")):
        vf, sf, mf = _var("fresh", metric_key)
        vr, sr, mr = _var("reuse", metric_key)
        if vf is None or vr is None:
            continue
        cold = max(0.0, vf - vr)
        block[label] = {
            "fresh_std": sf, "fresh_cv_pct": (100.0 * sf / mf if mf else None),
            "reuse_std": sr, "reuse_cv_pct": (100.0 * sr / mr if mr else None),
            "within_session_var": vr, "cold_start_var": cold, "operational_var": vf,
            "cold_start_share_pct": (100.0 * cold / vf if vf else None),
            "within_session_share_pct": (100.0 * vr / vf if vf else None),
            "cold_start_std_equiv": math.sqrt(cold),
        }
    block["note"] = ("var(fresh)-var(reuse) = restart/cold-start variance; the "
                     "remainder (=var(reuse)) is the irreducible within-session "
                     "measurement+thermal+token floor. Variances add only if the two "
                     "components are independent — a first-order attribution.")
    return block


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="noise_floor_<mode>.json files")
    ap.add_argument("--out", type=Path, default=Path("research/tps_noise_floor/analysis.json"))
    args = ap.parse_args(argv)

    out: dict[str, Any] = {"modes": {}}
    for path in args.inputs:
        data = json.loads(Path(path).read_text())
        mode = data.get("mode", Path(path).stem)
        records = data.get("records", [])
        out["modes"][mode] = analyze_mode(records)
        out["modes"][mode]["source"] = str(path)
        out["modes"][mode]["steptime"] = data.get("steptime")

    decomp = cross_mode_decomposition(out["modes"])
    if decomp:
        out["cross_mode_decomposition"] = decomp

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    _print_report(out)
    print(f"\n[analyze] -> {args.out}", flush=True)
    return 0


def _fmt(v, p=2):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "—"


def _print_report(out: dict[str, Any]) -> None:
    cmd = out.get("cross_mode_decomposition")
    if cmd:
        print("\n========== VARIANCE DECOMPOSITION (fresh vs reuse) ==========")
        for label in ("raw_steady", "windowed"):
            b = cmd.get(label)
            if not b:
                continue
            print(f"-- {label}: operational CV(fresh)={_fmt(b['fresh_cv_pct'])}% "
                  f"within-session CV(reuse)={_fmt(b['reuse_cv_pct'])}%")
            print(f"     cold-start share={_fmt(b['cold_start_share_pct'])}%  "
                  f"within-session share={_fmt(b['within_session_share_pct'])}%  "
                  f"(cold-start std-equiv {_fmt(b['cold_start_std_equiv'])} tps)")
    for mode, a in out["modes"].items():
        print(f"\n========== MODE: {mode} (n={a['n_runs']}, steptime={a.get('steptime')}) ==========")
        nf = a["noise_floor"]
        print("-- noise floor (per-run) --")
        for k, s in nf.items():
            if s.get("n"):
                print(f"   {k:30s} n={s['n']:2d} mean={_fmt(s['mean'])} std={_fmt(s['std'])} "
                      f"CV={_fmt(s['cv_pct'])}% range={_fmt(s['min'])}..{_fmt(s['max'])} "
                      f"({_fmt(s['range_pct'])}%)")
        w = a["warmup"]
        print(f"-- warmup: recommended W={a['best_w_head']} intervals "
              f"(plateau_w={w.get('plateau_w')}, global-min-CV W={(w.get('best_min_cv') or {}).get('w_head')}); "
              f"first-interval={_fmt(w.get('first_interval_mean'))} vs "
              f"steady={_fmt(w.get('steady_mean_at_recommended_w'))} "
              f"(deficit {_fmt(w.get('warmup_deficit_pct'))}%)")
        for row in w.get("sweep", []):
            print(f"     W={row['w_head']}: mean={_fmt(row['mean_tps'])} CV={_fmt(row['cv_pct'])}%")
        d = a["drift_windowed"]
        print(f"-- drift (windowed): TPS~runidx slope={_fmt((d.get('tps_vs_runidx') or {}).get('slope'),3)} "
              f"r={_fmt((d.get('tps_vs_runidx') or {}).get('r'),3)}; lag1={_fmt(d.get('tps_lag1_autocorr'),3)}; "
              f"TPS~SMclock r={_fmt(d.get('tps_vs_smclock_pearson'),3)}; "
              f"SMclock~min slope={_fmt((d.get('sm_clock_vs_wallmin') or {}).get('slope'),2)}MHz/min; "
              f"temp~min slope={_fmt((d.get('temp_vs_wallmin') or {}).get('slope'),2)}C/min")
        t = a["token_nondeterminism"]
        print(f"-- token nondet: E[accept] CV={_fmt((t.get('e_accept') or {}).get('cv_pct'),3)}%, "
              f"pearson(TPS,E[accept])={_fmt(t.get('pearson_tps_vs_e_accept'),3)}")
        print("-- median-of-n residual CV --")
        for b in a["median_of_n_cv"]:
            if b:
                print(f"     n={b['n']}: CV={_fmt(b['cv_pct'])}%  SE={_fmt(b['se'],3)}")
        for mde_key, label in (("mde_wall_median", "wall_tps [RECOMMENDED]"),
                               ("mde_median", "windowed interval-mean")):
            mde = a.get(mde_key, {})
            if not mde.get("levels"):
                continue
            print(f"-- MDE on {label} (median-of-n, residual CV={_fmt(mde.get('residual_cv_pct'))}%, "
                  f"drift {_fmt(mde.get('drift_slope_per_run_tps'),3)} tps/run) --")
            for lv in mde.get("levels", []):
                print(f"     n={lv['n_per_arm']}: interleaved MDE={_fmt(lv['mde_interleaved_pct'])}% "
                      f"(pow {_fmt(lv['mde_interleaved_pow_pct'])}%)  | "
                      f"sequential MDE={_fmt(lv['mde_sequential_pct'])}% "
                      f"(bias {_fmt(lv['drift_bias_tps'],2)} tps)")


if __name__ == "__main__":
    raise SystemExit(main())
