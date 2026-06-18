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


def read_acceptor_stats(stat_dir: Path) -> dict[str, Any]:
    """Aggregate the recompute-acceptor sidecar JSONs a tau-mode arm wrote (one set
    per serve PID). Returns the run-summed realized firing/flag rate (the live, data-
    driven rate the gap-gate actually fired at) and the cudagraph dispatch resolution
    (whether a width-1 recompute replays a captured graph or falls to eager)."""
    out: dict[str, Any] = {"stat_dir": str(stat_dir)}
    if not stat_dir.exists():
        return out
    emitted = positions = flagged = fired = 0
    dispatch = None
    for p in sorted(stat_dir.glob("recompute_progress_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        emitted += int(d.get("emitted_total", 0))
        positions += int(d.get("positions_total", 0))
        flagged += int(d.get("flagged_total", 0))
        fired += int(d.get("fired_total", 0))
    for p in sorted(stat_dir.glob("recompute_dispatch_resolve_*.json")):
        try:
            dispatch = json.loads(p.read_text())
            break
        except Exception:
            continue
    out.update({
        "emitted_total": emitted, "positions_total": positions,
        "flagged_total": flagged, "fired_total": fired,
        "realized_flag_rate": (flagged / positions) if positions else None,
        "realized_fire_rate": (fired / emitted) if emitted else None,
        "dispatch_resolve": dispatch,
        "is_captured_replay": (dispatch or {}).get("is_captured_replay"),
    })
    return out


def read_livecert_summary(stat_dir: Path) -> dict[str, Any]:
    """Read the live-cert sidecars (one per worker PID) and return the dominant one
    (most pre-fork draft positions). The patch writes the COMPOSED cert summary at every
    progress boundary + at exit, so the file already holds per-tau flag/break rates, the
    min break-free tau, position counts and the rule-of-three UB."""
    best: dict[str, Any] = {}
    best_n = -1
    if stat_dir.exists():
        for p in sorted(stat_dir.glob("livecert_summary_*.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            n = int(d.get("prefork_draft_positions", 0) or 0)
            if n > best_n:
                best_n, best = n, d
    return best


def _run_one(arm: ArmSpec, n: int, rargs, server_python: Path, out_dir: Path,
             fh) -> dict[str, Any]:
    """Run a single arm with N fresh serves; return its per-arm stat dict."""
    rargs_n = SimpleNamespace(**{**vars(rargs), "n": n})
    recs = run_arm(arm, rargs_n, server_python, out_dir, fh)
    st = arm_stats(recs)
    return {
        "label": arm.label, "override_env": arm.override_env,
        "wall_tps_median": median_wall_tps(st),
        "wall_tps_cv_pct": (st.get("wall_tps") or {}).get("cv_pct"),
        "wall_tps_values": (st.get("wall_tps") or {}).get("values"),
        "e_accept_exact_mean": (st.get("e_accept_exact") or {}).get("mean"),
        "n": (st.get("wall_tps") or {}).get("n"),
    }


def run_livecert(a, extra: dict[str, str], out_dir: Path, server_python: Path,
                 rargs) -> dict[str, Any]:
    """PR #669 -- LIVE de-teacher-force identity cert + Item-3 efficient-acceptor speed.

    Phase A: SENPAI_REFERENCE_MODE=1 arm -> served M=1-AR trajectory R_served jsonl
             (skipped if --ref-jsonl reuses an existing one).
    Phase B: instrumented live K-spec arm (SENPAI_LIVECERT_REF_JSONL) -> the patch
             records the per-position verify gap + de-teacher-forced flip along the LIVE
             trajectory, partitioned pre-fork vs global, and writes the cert summary
             (min_safe flag rate, served_rescued_break_rate, position count, RoT UB).
    Phase C (unless --cert-only): per-position RATE arms at {0, min_safe, 2*min_safe} ->
             efficient_acceptor_tps (the Item-1 sync-strip makes this a measurement);
             r=0 is the un-rescued ceiling. Item-4 decisive: official-equiv vs 126.378
             (+10 bar 136.378)."""
    sub = a.submission
    records_path = out_dir / "records.jsonl"
    per_arm: dict[str, dict[str, Any]] = {}
    with open(records_path, "w") as fh:
        # ---- Phase A: R_served ----
        if a.ref_jsonl is not None:
            ref_jsonl = a.ref_jsonl.resolve()
            print(f"[livecert] reusing R_served: {ref_jsonl}", flush=True)
        else:
            ref_arm = ArmSpec("ref_d", sub, {"SENPAI_REFERENCE_MODE": "1", **extra})
            per_arm["ref_d"] = _run_one(ref_arm, 1, rargs, server_python, out_dir, fh)
            ref_jsonl = out_dir / "ref_d" / "decode" / "run00.jsonl"
            print(f"[livecert] R_served generated: {ref_jsonl} "
                  f"(ar wall_tps={per_arm['ref_d']['wall_tps_median']})", flush=True)
        if not ref_jsonl.exists():
            raise SystemExit(f"[livecert] R_served jsonl missing: {ref_jsonl}")

        # ---- Phase B: instrumented live cert arm ----
        stat_dir = (out_dir / "livecert_stat").resolve()
        lc_env = {
            **extra,
            "SENPAI_LIVECERT_REF_JSONL": str(ref_jsonl),
            "SENPAI_LIVECERT_STAT_DIR": str(stat_dir),
            "SENPAI_LIVECERT_N_PROMPTS": str(a.num_prompts),
            # Write the composed cert sidecar frequently so a short run (which never
            # crosses a big boundary) still leaves a summary on disk; the SIGTERM/atexit
            # flush writes the final authoritative one. JSON dump of a small dict is cheap.
            "SENPAI_RECOMPUTE_LOG_EVERY": "512",
        }
        lc_arm = ArmSpec("livecert", sub, lc_env)
        per_arm["livecert"] = _run_one(lc_arm, 1, rargs, server_python, out_dir, fh)
        cert = read_livecert_summary(stat_dir)
        if not cert:
            raise SystemExit(f"[livecert] no cert summary written to {stat_dir}")
        min_safe_draft = cert.get("min_safe_live_flag_rate_per_draft")
        min_safe_emit = cert.get("min_safe_live_flag_rate_per_emit")
        # Draft-only partition (PR #669): the acceptor recomputes DRAFT positions; no-gap
        # prefill (output-pos-0) flips are structurally un-flaggable (no draft row) and
        # are reported separately. The efficient-acceptor rate is set by the draft-only
        # min-safe tau (the strict one is None whenever any prefill flip occurs, which
        # would spuriously gate off the whole speed leg).
        min_safe_draft_do = cert.get("min_safe_live_flag_rate_per_draft_draftonly")
        min_safe_emit_do = cert.get("min_safe_live_flag_rate_per_emit_draftonly")
        eff_min_safe_draft = min_safe_draft_do if min_safe_draft_do else min_safe_draft
        print(f"[livecert] CERT: min_safe_tau(strict)={cert.get('min_safe_tau')} "
              f"min_safe_tau(draftonly)={cert.get('min_safe_tau_draftonly')} "
              f"flag_rate_do(per_draft)={min_safe_draft_do} (per_emit)={min_safe_emit_do} "
              f"break_rate_do={cert.get('served_rescued_break_rate_at_min_safe_draftonly')} "
              f"n_prefill_flips={cert.get('n_prefill_flips')} "
              f"n_draft_flips={cert.get('n_draft_flips')} "
              f"prefork_draft_positions={cert.get('prefork_draft_positions')} "
              f"RoT_UB={cert.get('rule_of_three_ub_over_draft')} "
              f"flip_gap_max={cert.get('flip_gap_max')} "
              f"flip_gap_max_finite={cert.get('flip_gap_max_finite')} "
              f"matched={cert.get('n_matched')}/{cert.get('n_reqs_seen')}", flush=True)

        # ---- Phase C: efficient-acceptor speed at the min-safe per-position rate ----
        speed: dict[str, Any] = {}
        if not a.cert_only and eff_min_safe_draft and eff_min_safe_draft > 0:
            if a.speed_rates:
                rates = [float(x) for x in a.speed_rates.split(",") if x.strip() != ""]
            else:
                rates = [0.0, round(eff_min_safe_draft, 6),
                         round(2 * eff_min_safe_draft, 6)]
            rate_to_tps: dict[float, float] = {}
            for r in rates:
                lbl = f"r{r:g}".replace(".", "p").replace("-", "m")
                env = {**extra}
                if r > 0:
                    env["SENPAI_RECOMPUTE_RATE"] = repr(r)
                    env["SENPAI_RECOMPUTE_STAT_DIR"] = str(
                        (out_dir / f"speed_stat_{lbl}").resolve())
                arm = ArmSpec(f"speed_{lbl}", sub, env)
                per_arm[arm.label] = _run_one(arm, a.n, rargs, server_python, out_dir, fh)
                rate_to_tps[r] = per_arm[arm.label]["wall_tps_median"]
                print(f"[livecert] speed r={r}: wall_tps="
                      f"{per_arm[arm.label]['wall_tps_median']}", flush=True)
            fit = fit_additive_cost(rate_to_tps)
            unrescued = rate_to_tps.get(0.0)
            eff_rate = round(eff_min_safe_draft, 6)
            eff_tps = rate_to_tps.get(eff_rate)
            ratio_ar = (eff_tps / a.ar_rung_local) if eff_tps else None
            official_equiv = (ratio_ar * LOCKED_319_AR_TPS) if ratio_ar else None
            speed = {
                "rate_to_wall_tps": {str(k): v for k, v in sorted(rate_to_tps.items())},
                "additive_cost_fit": fit,
                "unrescued_ceiling_local": unrescued,
                "unrescued_k5_anchor": a.unrescued_k5_local,
                "efficient_acceptor_rate_per_draft": eff_rate,
                "efficient_acceptor_rate_basis": (
                    "draftonly" if min_safe_draft_do else "strict"),
                "efficient_acceptor_flag_rate_per_emit": (
                    min_safe_emit_do if min_safe_draft_do else min_safe_emit),
                "efficient_acceptor_tps": eff_tps,
                "ar_rung_local": a.ar_rung_local,
                "ar_rung_official": LOCKED_319_AR_TPS,
                "acceptor_over_ar_ratio": ratio_ar,
                "official_equiv_tps": official_equiv,
                "clears_rung_by_tps": (official_equiv - LOCKED_319_AR_TPS)
                                      if official_equiv else None,
                "clears_plus10_bar": (official_equiv >= LOCKED_319_AR_TPS + 10.0)
                                     if official_equiv else None,
                "acceptor_over_unrescued": (eff_tps / unrescued)
                                           if (eff_tps and unrescued) else None,
            }
            print(f"[livecert] SPEED: eff_acceptor_tps={eff_tps} "
                  f"unrescued={unrescued} official_equiv={official_equiv} "
                  f"clears_+10={speed.get('clears_plus10_bar')}", flush=True)

        verdict = _livecert_verdict(cert, speed, a)
        result = {
            "pr": 669, "leg": "livecert", "analysis_only": True,
            "official_tps": 0, "no_hf_job": True,
            "submission": sub, "extra_env": extra,
            "n": a.n, "live_tau_scan": bool(a.live_tau_scan),
            "workload": {"num_prompts": a.num_prompts, "output_len": a.output_len,
                         "seed": a.seed},
            "ref_jsonl": str(ref_jsonl),
            "cert": cert,
            "speed": speed,
            "verdict": verdict,
            "arms": per_arm,
            "anchors": {"locked_ar_tps": LOCKED_319_AR_TPS,
                        "ar_rung_local": a.ar_rung_local,
                        "unrescued_k5_local": a.unrescued_k5_local,
                        "plus10_bar_official": LOCKED_319_AR_TPS + 10.0},
        }
        result_path = out_dir / "livecert.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"[livecert] verdict={verdict['verdict']} -> {result_path}", flush=True)
    return result


def _livecert_verdict(cert: dict, speed: dict, a) -> dict[str, Any]:
    """Map the cert + speed numbers onto the PR's three verdicts.

    Reducibility is decided on the DRAFT-ONLY partition: the recompute-acceptor flags
    and recomputes draft (verify) positions, so the relevant question is the min
    break-free flag rate over draft positions vs the un-rescued live realized rate
    (~0.1126/emit per the cost probe). No-gap prefill (output-pos-0) flips are
    structurally outside the acceptor's domain and are surfaced as a separate identity
    caveat (n_prefill_flips), not folded into reducibility. If reducible, the speed
    decides HOLDS (official-equiv >= +10 bar) vs MARGINAL (clears the rung but not +10)
    vs below-rung (reducible but eroded)."""
    # draft-only flag rate / break (acceptor domain); fall back to strict if the
    # partition is absent (older cert summaries).
    flag_emit_do = cert.get("min_safe_live_flag_rate_per_emit_draftonly")
    flag_emit_strict = cert.get("min_safe_live_flag_rate_per_emit")
    flag_emit = flag_emit_do if flag_emit_do is not None else flag_emit_strict
    break_rate = cert.get("served_rescued_break_rate_at_min_safe_draftonly")
    if break_rate is None:
        break_rate = cert.get("served_rescued_break_rate_at_min_safe")
    n_prefill_flips = cert.get("n_prefill_flips")
    LIVE_UNRESCUED_FIRE_PER_EMIT = 0.1126  # cost-probe realized fire/emit (#663 full_k5)
    reducible = (
        flag_emit is not None
        and flag_emit < 0.75 * LIVE_UNRESCUED_FIRE_PER_EMIT
        and (break_rate == 0 or break_rate is None or break_rate == 0.0)
    )
    official_equiv = speed.get("official_equiv_tps")
    plus10 = LOCKED_319_AR_TPS + 10.0
    if not reducible:
        v = "LIVE_FLAG_IRREDUCIBLE"
    elif official_equiv is None:
        v = "LIVE_FLAG_REDUCIBLE_CERT_ONLY"
    elif official_equiv >= plus10:
        v = "LIVE_FLAG_REDUCIBLE_HOLDS"
    elif official_equiv >= LOCKED_319_AR_TPS:
        v = "LIVE_FLAG_REDUCIBLE_MARGINAL"
    else:
        v = "LIVE_FLAG_REDUCIBLE_BELOW_RUNG"
    return {
        "verdict": v,
        "reducible": reducible,
        "reducibility_basis": "draftonly" if flag_emit_do is not None else "strict",
        "min_safe_flag_rate_per_emit": flag_emit,
        "min_safe_flag_rate_per_emit_strict": flag_emit_strict,
        "n_prefill_flips": n_prefill_flips,
        "live_unrescued_fire_per_emit": LIVE_UNRESCUED_FIRE_PER_EMIT,
        "served_rescued_break_rate": break_rate,
        "official_equiv_tps": official_equiv,
        "plus10_bar": plus10,
        "rung": LOCKED_319_AR_TPS,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default="int4_mtp_batchinv")
    ap.add_argument("--mode", choices=["reference", "rate", "tau", "livecert"],
                    required=True,
                    help="reference: one SENPAI_REFERENCE_MODE=1 arm (d / R_served). "
                         "rate: one SENPAI_RECOMPUTE_RATE arm per --rates value. "
                         "tau: #663 REAL gap-flag acceptor -- one un-rescued arm + one "
                         "SENPAI_ACCEPTOR_TAU arm per --taus value (data-driven firing). "
                         "livecert: PR #669 -- LIVE de-teacher-force identity cert "
                         "(ref arm -> R_served, instrumented live arm -> min_safe flag "
                         "rate + break_rate, then per-position rate arms -> efficient "
                         "acceptor TPS at the min-safe rate).")
    ap.add_argument("--rates", default="0.0,0.05,0.10,0.20",
                    help="comma rates for --mode rate (slope sweep). r=0 == un-rescued ceiling.")
    ap.add_argument("--taus", default="0.3",
                    help="comma tau-flag thresholds for --mode tau (real gap-gated acceptor).")
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
    ap.add_argument("--ar-rung-local", type=float, default=126.75,
                    help="arm (b): local AR rung wall_tps (#642 6uepftr6 = 126.75). "
                         "acceptor_over_ar_ratio = arm(a)/this; official-equiv = ratio*126.378")
    # ---- livecert mode ----
    ap.add_argument("--live-tau-scan", action="store_true",
                    help="livecert: sweep the flag predicate loose->tight on the live "
                         "loop (the patch sweeps an internal tau grid; this flag is the "
                         "explicit opt-in and is recorded in the result).")
    ap.add_argument("--ref-jsonl", type=Path, default=None,
                    help="livecert: reuse an existing R_served decode jsonl instead of "
                         "generating one (default: run a SENPAI_REFERENCE_MODE=1 arm).")
    ap.add_argument("--cert-only", action="store_true",
                    help="livecert: stop after the cert (skip the Item-3 speed rate arms).")
    ap.add_argument("--speed-rates", default=None,
                    help="livecert: comma per-position fire rates for the Item-3 speed "
                         "arms (default: 0.0,<min_safe_per_draft>,<2x>).")
    ap.add_argument("--unrescued-k5-local", type=float, default=159.333,
                    help="livecert: K=5 un-rescued ceiling anchor (full_k5 tau-mode).")
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

    sub_dir = (ROOT / "submissions" / a.submission).resolve()
    if not sub_dir.exists():
        raise SystemExit(f"submission not found: {sub_dir}")

    # ---- PR #669 livecert mode: dedicated orchestration (ref -> live cert -> speed) ----
    if a.mode == "livecert":
        manifest = harness.load_manifest(sub_dir)
        server_python = harness.ensure_server_venv(manifest["dependencies"])
        rargs = build_args(a)
        print(f"[livecert] submission={a.submission} extra_env={extra} "
              f"workload={a.num_prompts}x{a.output_len} cert_only={a.cert_only} -> {out_dir}",
              flush=True)
        t0 = time.time()
        result = run_livecert(a, extra, out_dir, server_python, rargs)
        result["elapsed_s"] = time.time() - t0
        print(f"[livecert] ===== done in {result['elapsed_s']/60:.1f} min =====", flush=True)
        if not a.no_wandb:
            _log_wandb_livecert(a, result)
        return 0

    # Build arms.
    arms: list[ArmSpec] = []
    if a.mode == "reference":
        arms.append(ArmSpec("ref_d", a.submission, {"SENPAI_REFERENCE_MODE": "1", **extra}))
    elif a.mode == "tau":
        taus = [float(x) for x in a.taus.split(",") if x.strip() != ""]
        # Arm (c): un-rescued ceiling -- acceptor OFF, otherwise identical stack.
        arms.append(ArmSpec("unrescued", a.submission, {**extra}))
        # Arm (a): the REAL gap-flag acceptor -- firing emerges from live verify gaps.
        for t in taus:
            lbl = f"tau{t:g}".replace(".", "p")
            env = {**extra, "SENPAI_ACCEPTOR_TAU": repr(t),
                   "SENPAI_RECOMPUTE_STAT_DIR": str((out_dir / f"stat_{lbl}").resolve())}
            arms.append(ArmSpec(lbl, a.submission, env))
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
            if a.mode == "tau" and "SENPAI_ACCEPTOR_TAU" in arm.override_env:
                stat_dir = Path(arm.override_env["SENPAI_RECOMPUTE_STAT_DIR"])
                acc_stats = read_acceptor_stats(stat_dir)
                per_arm[arm.label]["acceptor_stats"] = acc_stats
                print(f"[deproject] arm {arm.label}: realized_flag_rate="
                      f"{acc_stats.get('realized_flag_rate')} fired="
                      f"{acc_stats.get('fired_total')} captured_replay="
                      f"{acc_stats.get('is_captured_replay')}", flush=True)
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

    tau_summary = None
    if a.mode == "tau":
        unrescued_tps = (per_arm.get("unrescued") or {}).get("wall_tps_median")
        tau_summary = {"unrescued_ceiling": unrescued_tps,
                       "ar_rung_local": a.ar_rung_local,
                       "ar_rung_official": LOCKED_319_AR_TPS, "arms": {}}
        for lbl, info in per_arm.items():
            if "SENPAI_ACCEPTOR_TAU" not in info.get("override_env", {}):
                continue
            a_tps = info["wall_tps_median"]
            ratio_ar = (a_tps / a.ar_rung_local) if a_tps else None
            tau_summary["arms"][lbl] = {
                "acceptor_walltps_real_live": a_tps,
                "acceptor_K5_over_ar_ratio": ratio_ar,
                "official_equiv_tps": (ratio_ar * LOCKED_319_AR_TPS) if ratio_ar else None,
                "clears_rung_by_tps": (ratio_ar * LOCKED_319_AR_TPS - LOCKED_319_AR_TPS)
                                      if ratio_ar else None,
                "acceptor_over_unrescued": (a_tps / unrescued_tps)
                                           if (a_tps and unrescued_tps) else None,
                "realized_flag_rate": (info.get("acceptor_stats") or {}).get(
                    "realized_flag_rate"),
                "is_captured_replay": (info.get("acceptor_stats") or {}).get(
                    "is_captured_replay"),
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
        "tau_summary": tau_summary,
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
    if tau_summary:
        print(f"  [tau] un-rescued ceiling (c)={tau_summary['unrescued_ceiling']} "
              f"AR rung local (b)={tau_summary['ar_rung_local']}", flush=True)
        for lbl, s in tau_summary["arms"].items():
            print(f"  [tau] {lbl}: acceptor(a)={s['acceptor_walltps_real_live']} "
                  f"ratio(a/b)={s['acceptor_K5_over_ar_ratio']} "
                  f"official_equiv={s['official_equiv_tps']} "
                  f"clears_rung_by={s['clears_rung_by_tps']} TPS | "
                  f"realized_flag_rate={s['realized_flag_rate']} "
                  f"captured_replay={s['is_captured_replay']}", flush=True)
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
    extra_env = result["extra_env"]
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.wandb_name or f"stark/deproject-{a.mode}", group=a.wandb_group,
        notes=f"PR#663 K=5 real gap-flag recompute acceptor ({a.mode}): live served "
              f"wall_tps (harness lineage #642 de-projection)",
        config={"pr": 663, "harness_lineage_pr": 642, "mode": a.mode,
                "submission": a.submission, "extra_env": extra_env, "n": a.n,
                "num_prompts": a.num_prompts, "output_len": a.output_len,
                "num_speculative_tokens": extra_env.get("NUM_SPECULATIVE_TOKENS"),
                "analysis_only": True, "official_tps": 0},
    )
    if run is None:
        print("[wandb] disabled; JSON only", flush=True)
        return
    summary: dict[str, Any] = {"analysis_only": 1, "official_tps": 0}
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
    ts = result.get("tau_summary")
    if ts:
        if isinstance(ts.get("unrescued_ceiling"), (int, float)):
            summary["tau/unrescued_ceiling"] = ts["unrescued_ceiling"]
        for lbl, s in ts.get("arms", {}).items():
            for key in ("acceptor_walltps_real_live", "acceptor_K5_over_ar_ratio",
                        "official_equiv_tps", "clears_rung_by_tps",
                        "acceptor_over_unrescued", "realized_flag_rate"):
                if isinstance(s.get(key), (int, float)):
                    summary[f"tau/{lbl}/{key}"] = s[key]
            if s.get("is_captured_replay") is not None:
                summary[f"tau/{lbl}/is_captured_replay"] = int(bool(s["is_captured_replay"]))
    for k, v in summary.items():
        run.summary[k] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def _log_wandb_livecert(a, result: dict[str, Any]) -> None:
    """PR #669 -- log the LIVE cert + speed as EXPLICIT, machine-checkable W&B summary
    scalars (in #663 the break_rate was prose-only; here served_rescued_break_rate, the
    pre-fork position count and the rule-of-three UB are logged scalars)."""
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] import failed: {exc!r}; JSON only", flush=True)
        return
    extra_env = result["extra_env"]
    cert = result.get("cert") or {}
    speed = result.get("speed") or {}
    verdict = result.get("verdict") or {}
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.wandb_name or "stark/k5-livecert", group=a.wandb_group,
        notes="PR#669 K=5 LIVE de-teacher-force identity cert + efficient-acceptor "
              "speed (Item-1 sync strip, per-position firing). analysis_only.",
        config={"pr": 669, "harness_lineage_pr": 642, "mode": "livecert",
                "submission": a.submission, "extra_env": extra_env, "n": a.n,
                "num_prompts": a.num_prompts, "output_len": a.output_len,
                "num_speculative_tokens": extra_env.get("NUM_SPECULATIVE_TOKENS"),
                "live_tau_scan": bool(a.live_tau_scan),
                "ref_jsonl": result.get("ref_jsonl"),
                "analysis_only": True, "official_tps": 0},
    )
    if run is None:
        print("[wandb] disabled; JSON only", flush=True)
        return
    summary: dict[str, Any] = {"analysis_only": 1, "official_tps": 0}
    # ---- REQUIRED machine-checkable cert scalars (Item 2) ----
    summary["cert/served_rescued_break_rate"] = cert.get(
        "served_rescued_break_rate_at_min_safe")
    summary["cert/prefork_draft_positions"] = cert.get("prefork_draft_positions")
    summary["cert/prefork_emit_positions"] = cert.get("prefork_emit_positions")
    summary["cert/rule_of_three_ub"] = cert.get("rule_of_three_ub_over_draft")
    summary["cert/min_safe_tau"] = cert.get("min_safe_tau")
    summary["cert/min_safe_live_flag_rate_per_draft"] = cert.get(
        "min_safe_live_flag_rate_per_draft")
    summary["cert/min_safe_live_flag_rate_per_emit"] = cert.get(
        "min_safe_live_flag_rate_per_emit")
    summary["cert/n_flips_prefork"] = cert.get("n_flips_prefork")
    summary["cert/flip_gap_max"] = cert.get("flip_gap_max")
    summary["cert/n_matched"] = cert.get("n_matched")
    summary["cert/n_forked"] = cert.get("n_forked")
    summary["cert/n_reqs_seen"] = cert.get("n_reqs_seen")
    summary["cert/global_draft_positions"] = cert.get("global_draft_positions")
    # ---- draft-only partition scalars (PR #669: prefill flips are un-flaggable and
    # reported separately from the acceptor-domain draft-only min-safe flag rate) ----
    summary["cert/n_prefill_flips"] = cert.get("n_prefill_flips")
    summary["cert/n_draft_flips"] = cert.get("n_draft_flips")
    summary["cert/flip_gap_max_finite"] = cert.get("flip_gap_max_finite")
    summary["cert/min_safe_tau_draftonly"] = cert.get("min_safe_tau_draftonly")
    summary["cert/min_safe_live_flag_rate_per_draft_draftonly"] = cert.get(
        "min_safe_live_flag_rate_per_draft_draftonly")
    summary["cert/min_safe_live_flag_rate_per_emit_draftonly"] = cert.get(
        "min_safe_live_flag_rate_per_emit_draftonly")
    summary["cert/served_rescued_break_rate_draftonly"] = cert.get(
        "served_rescued_break_rate_at_min_safe_draftonly")
    # per-tau curves -> scalars (flag rates pre-fork & global, break rate)
    for tk, s in (cert.get("per_tau") or {}).items():
        for key in ("prefork_flag_rate_per_emit", "prefork_flag_rate_per_draft",
                    "global_flag_rate_per_draft", "prefork_break_rate_over_draft",
                    "prefork_break_count", "prefork_break_count_draftonly",
                    "prefork_break_rate_draftonly_over_draft"):
            v = s.get(key)
            if isinstance(v, (int, float)):
                summary[f"cert/tau{tk}/{key}"] = v
    # ---- speed (Items 3-4) ----
    for key in ("efficient_acceptor_tps", "unrescued_ceiling_local",
                "efficient_acceptor_rate_per_draft", "acceptor_over_ar_ratio",
                "official_equiv_tps", "clears_rung_by_tps", "acceptor_over_unrescued",
                "ar_rung_local"):
        v = speed.get(key)
        if isinstance(v, (int, float)):
            summary[f"speed/{key}"] = v
    if speed.get("clears_plus10_bar") is not None:
        summary["speed/clears_plus10_bar"] = int(bool(speed["clears_plus10_bar"]))
    fit = speed.get("additive_cost_fit")
    if fit:
        summary["speed/fit_C_sec_per_recompute"] = fit.get("C_sec_per_recompute")
        summary["speed/fit_tps0"] = fit.get("tps0_fit")
        summary["speed/fit_r2"] = fit.get("r2")
    # ---- verdict + per-arm wall_tps ----
    summary["verdict"] = verdict.get("verdict")
    summary["verdict/reducible"] = int(bool(verdict.get("reducible")))
    if verdict.get("reducibility_basis") is not None:
        summary["verdict/reducibility_basis"] = verdict.get("reducibility_basis")
    if isinstance(verdict.get("n_prefill_flips"), (int, float)):
        summary["verdict/n_prefill_flips"] = verdict.get("n_prefill_flips")
    if isinstance(verdict.get("min_safe_flag_rate_per_emit"), (int, float)):
        summary["verdict/min_safe_flag_rate_per_emit"] = verdict.get(
            "min_safe_flag_rate_per_emit")
    if speed.get("efficient_acceptor_rate_basis") is not None:
        summary["speed/efficient_acceptor_rate_basis"] = speed.get(
            "efficient_acceptor_rate_basis")
    if isinstance(speed.get("efficient_acceptor_flag_rate_per_emit"), (int, float)):
        summary["speed/efficient_acceptor_flag_rate_per_emit"] = speed.get(
            "efficient_acceptor_flag_rate_per_emit")
    for lbl, info in result["arms"].items():
        if isinstance(info.get("wall_tps_median"), (int, float)):
            summary[f"arm/{lbl}/wall_tps"] = info["wall_tps_median"]
        if isinstance(info.get("e_accept_exact_mean"), (int, float)):
            summary[f"arm/{lbl}/e_accept"] = info["e_accept_exact_mean"]
    for k, v in summary.items():
        if v is not None:
            run.summary[k] = v
    finish_wandb(run)
    print(f"[wandb] logged livecert run {run.id} verdict={verdict.get('verdict')}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
