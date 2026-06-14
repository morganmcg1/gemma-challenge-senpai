"""TPS measurement noise-floor characterization harness (PR #72).

LOCAL-only. Serves the deployed ``fa2sw_precache_kenyan`` stack UNCHANGED and
runs the canonical 128-prompt x 512-token, conc=1 decode N times, capturing for
each run the noise-floor evidence the measurement-protocol deliverable needs:

  * per-run **steady-state TPS** (mean of vLLM's "Avg generation throughput"
    interval meter -- the metric behind the 428.37 headline) and **wall TPS**
    (num_completion_tokens / decode duration_s),
  * the full **per-interval throughput series** within each run (warmup ramp +
    drift), parsed from the server log slice for that run,
  * per-run **E[accept]** (mean acceptance length) -- to correlate token-level
    nondeterminism with the TPS swing,
  * a continuous **GPU SM-clock / temp / power** sample stream during the decode
    (thermal/clock drift attribution).

Two modes isolate the two halves of the variance:
  * ``--mode fresh``  : a brand-new server per run (matches how real A/B arms run
    -- includes model-load + CUDA-graph-capture + start-thermal variance). This is
    the *operationally relevant* noise floor (comparable to #56's 429.04 vs 448.01).
  * ``--mode reuse``  : one server, N back-to-back decodes (removes restart
    variance -> the irreducible within-session measurement+thermal+token floor).

Fixed seed (1) every run => identical workload => any TPS swing is pure
measurement / hardware / FP-reduction noise, which is exactly the noise floor.

No served-file changes. Decode-only timing; an optional single PPL pass
(``--ppl-once``, reuse mode) is the unchanged-stack validity check, run AFTER the
timing loop so it never perturbs a timed run.

Run under the repo .venv (has wandb); serve/decode subprocs use the submission's
serve venv. Example::

    .venv/bin/python -m research.tps_noise_floor.run_noise_floor \
        --submission fa2sw_precache_kenyan --mode fresh --n-runs 12 \
        --wandb-name lawine/noise-floor-fresh --wandb-group tps-noise-floor
"""
from __future__ import annotations

import argparse
import json
import os
import signal
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

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

OUT_ROOT = ROOT / "research" / "tps_noise_floor"


# ---------------------------------------------------------------------------
# Server-log slice -> steady TPS series + E[accept]
# ---------------------------------------------------------------------------
def parse_run_slice(slice_text: str, num_spec: int | None = None) -> dict[str, Any]:
    """Per-run metrics from this run's slice of the vLLM server log.

    Reuses serve_profile's validated parser for the scalar method that defines
    the 428.37 headline (``steady_gen_tps_mean``), then additionally exposes the
    full per-interval throughput series (the warmup/drift signal the scalar hides)
    and an idle-filtered steady mean (reuse mode logs "0.0" idle lines between
    decodes).

    ``num_spec`` (K) is parsed ONCE from the full server log by the caller and
    injected here: in reuse mode the per-run slice starts AFTER the server-startup
    ``num_speculative_tokens`` line, so serve_profile.parse_spec_log can't recover
    K from the slice alone and would leave ``e_accept_exact`` (the per-run E[accept]
    PR #72 task 2 correlates against TPS) unset. The SpecDecoding Accepted/Drafted
    token counts ARE in every slice, so injecting K is all that's missing.
    """
    scalar = serve_profile.parse_spec_log(slice_text)
    gen_series_all = [float(x) for x in serve_profile._GEN_TPS_RE.findall(slice_text)]
    gen_series = [v for v in gen_series_all if v > 0.0]  # drop idle 0.0 ticks
    out: dict[str, Any] = dict(scalar)
    if out.get("e_accept_exact") is None and num_spec:
        total_acc = out.get("total_accepted_tokens") or 0
        total_draft = out.get("total_drafted_tokens") or 0
        if total_draft:
            out["num_speculative_tokens"] = out.get("num_speculative_tokens") or num_spec
            out["e_accept_exact"] = 1.0 + num_spec * total_acc / total_draft
            out["draft_acceptance_rate"] = total_acc / total_draft
    out["gen_tps_series"] = gen_series
    out["gen_tps_series_n"] = len(gen_series)
    # Idle-filtered steady mean (the honest steady-state number when a shared
    # server logs idle ticks before/after the timed decode).
    out["steady_gen_tps_mean_nonzero"] = (
        statistics.fmean(gen_series) if gen_series else None
    )
    return out


# ---------------------------------------------------------------------------
# GPU clock / temp / power sampler (read-only nvidia-smi; negligible load)
# ---------------------------------------------------------------------------
_SMI_FIELDS = ["timestamp", "clocks.sm", "clocks.mem", "temperature.gpu",
               "power.draw", "utilization.gpu"]


class GpuSampler:
    """Background nvidia-smi loop -> per-run CSV; summarized on stop()."""

    def __init__(self, csv_path: Path, interval_ms: int = 250, gpu_index: int = 0) -> None:
        self.csv_path = csv_path
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.proc: subprocess.Popen | None = None
        self._fh = None

    def __enter__(self) -> "GpuSampler":
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.csv_path, "w")
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(_SMI_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-i", str(self.gpu_index),
            "-lms", str(self.interval_ms),
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.DEVNULL,
                                         text=True, preexec_fn=os.setsid)
        except OSError:
            self.proc = None
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass

    def summarize(self, util_min: float = 1.0) -> dict[str, Any]:
        """Mean/min/max of SM clock, temp, power. ``*_load`` = util>util_min rows
        (clocks while the decode is actually running)."""
        rows: list[dict[str, float]] = []
        try:
            for line in self.csv_path.read_text().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(_SMI_FIELDS):
                    continue
                try:
                    rows.append({
                        "sm": float(parts[1]), "mem": float(parts[2]),
                        "temp": float(parts[3]), "power": float(parts[4]),
                        "util": float(parts[5]),
                    })
                except ValueError:
                    continue
        except OSError:
            return {"n_samples": 0}
        if not rows:
            return {"n_samples": 0}

        def agg(key: str, subset: list[dict[str, float]]) -> dict[str, float]:
            vals = [r[key] for r in subset]
            if not vals:
                return {"mean": None, "min": None, "max": None}
            return {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}

        load = [r for r in rows if r["util"] >= util_min] or rows
        return {
            "n_samples": len(rows),
            "n_load_samples": len(load),
            "sm_clock_mhz": agg("sm", rows),
            "sm_clock_mhz_load": agg("sm", load),
            "mem_clock_mhz_load": agg("mem", load),
            "temp_c": agg("temp", rows),
            "power_w_load": agg("power", load),
            "util_pct": agg("util", rows),
        }


# ---------------------------------------------------------------------------
# One timed decode against a live server
# ---------------------------------------------------------------------------
def timed_decode(
    server: harness.LocalServer,
    runner_python: Path,
    out_dir: Path,
    run_idx: int,
    *,
    num_prompts: int,
    output_len: int,
    seed: int,
    log_offset: int,
    clock_interval_ms: int,
    settle_s: float,
) -> dict[str, Any]:
    """Run one decode, slice this run's server-log + GPU-clock window, return the
    per-run record. ``log_offset`` is the server-log byte length BEFORE this run
    (so a reused server log can be sliced per run)."""
    decode_out = out_dir / "decode" / f"run{run_idx:02d}.jsonl"
    decode_summary = out_dir / "decode" / f"run{run_idx:02d}.summary.json"
    clock_csv = out_dir / "clocks" / f"run{run_idx:02d}.csv"

    t_start = datetime.now(timezone.utc).isoformat()
    with GpuSampler(clock_csv, interval_ms=clock_interval_ms) as sampler:
        t0 = time.time()
        summary = harness.capture_decode(
            runner_python,
            base_url=server.base_url,
            model=server.served_model_name,
            out_file=decode_out,
            summary_file=decode_summary,
            num_prompts=num_prompts,
            output_len=output_len,
            seed=seed,
        )
        wall_around_decode_s = time.time() - t0
        # Record the slice end NOW (decode just finished) so the trailing partial /
        # idle throughput window — logged on the next ~10s tick, after the decode —
        # does NOT contaminate this run's steady series (matters in reuse mode where
        # the server stays up between runs). Brief flush first for any in-flight line.
        time.sleep(0.4)
        log_end = _server_log_len(server.log_path)
        time.sleep(max(0.0, settle_s - 0.4))  # let the GPU sampler catch the cooldown tail
    clock_stats = sampler.summarize()

    # Slice the server log for just this run: [log_offset, log_end). K
    # (num_speculative_tokens) is logged once at server startup, which a reuse-mode
    # run slice excludes — so recover it from the FULL log and inject it, otherwise
    # per-run E[accept] can't be computed (PR #72 task 2's TPS-vs-E[accept] signal).
    log_text = ""
    num_spec = None
    if server.log_path and Path(server.log_path).exists():
        data = Path(server.log_path).read_bytes()
        log_text = data[log_offset:log_end].decode("utf-8", "replace")
        m = serve_profile._NUM_SPEC_RE.search(data.decode("utf-8", "replace"))
        num_spec = int(m.group(1)) if m else None
    parsed = parse_run_slice(log_text, num_spec=num_spec)

    n_tok = int(summary.get("num_completion_tokens", 0))
    decode_dur_s = float(summary.get("duration_s", wall_around_decode_s))
    wall_tps = n_tok / decode_dur_s if decode_dur_s > 0 else float("nan")

    return {
        "run_idx": run_idx,
        "t_start_utc": t_start,
        "seed": seed,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "num_completion_tokens": n_tok,
        "decode_duration_s": decode_dur_s,
        "wall_around_decode_s": wall_around_decode_s,
        "wall_tps": wall_tps,
        "steady_gen_tps_mean": parsed.get("steady_gen_tps_mean"),
        "steady_gen_tps_mean_nonzero": parsed.get("steady_gen_tps_mean_nonzero"),
        "steady_gen_tps_n": parsed.get("steady_gen_tps_n"),
        "gen_tps_series": parsed.get("gen_tps_series"),
        "e_accept_exact": parsed.get("e_accept_exact"),
        "e_accept_interval_mean": parsed.get("e_accept_interval_mean"),
        "num_speculative_tokens": parsed.get("num_speculative_tokens"),
        "total_accepted_tokens": parsed.get("total_accepted_tokens"),
        "total_drafted_tokens": parsed.get("total_drafted_tokens"),
        "intervals": parsed.get("intervals"),
        "clock": clock_stats,
        "log_offset_start": log_offset,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [r[key] for r in records if isinstance(r.get(key), (int, float))
            and r[key] == r[key]]  # drop None / NaN
    if not vals:
        return {"n": 0}
    n = len(vals)
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    s = sorted(vals)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv_pct": 100.0 * std / mean if mean else float("nan"),
        "min": s[0],
        "max": s[-1],
        "range_pct": 100.0 * (s[-1] - s[0]) / mean if mean else float("nan"),
        "median": statistics.median(vals),
        "p25": s[int(0.25 * (n - 1))],
        "p75": s[int(0.75 * (n - 1))],
        "values": vals,
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _server_log_len(log_path: Path | None) -> int:
    try:
        return log_path.stat().st_size if log_path else 0
    except OSError:
        return 0


def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", "0"], capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def preflight_gpu(port: int = 8000, mem_threshold_mib: int = 1500, timeout_s: int = 180) -> None:
    """Guarantee a clean serving slot before LocalServer.__enter__.

    A prior server that connected to (or left behind) a vLLM ``api_server`` /
    orphaned ``EngineCore`` would (a) hold port 8000 so ``_wait_ready`` silently
    binds to the STALE server, and (b) hold VRAM so the next load OOMs. This
    single-GPU pod only ever runs this harness, so it is safe to reap any
    lingering vLLM process and then wait for VRAM to drain. Caught a stale
    ``--disable-log-stats`` server during smoke that produced 0 throughput lines.
    """
    patterns = ["vllm.entrypoints.openai.api_server", "VLLM::EngineCore",
                "multiprocessing.resource_tracker"]
    reaped = False
    for pat in patterns:
        r = subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        reaped = reaped or (r.returncode == 0)
    if reaped:
        print(f"[noise] preflight: reaped lingering vLLM process(es)", flush=True)
        time.sleep(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_mem_used_mib()
        if used is None or used < mem_threshold_mib:
            if used is not None:
                print(f"[noise] preflight: GPU free ({used} MiB used)", flush=True)
            return
        time.sleep(3)
    print(f"[noise] preflight: WARN GPU still busy after {timeout_s}s "
          f"({_gpu_mem_used_mib()} MiB)", flush=True)


def build_serve_env(args) -> dict[str, str]:
    """Measurement env mirroring serve_profile.run_timing_pass (the canonical
    path that produced the 428.37 / #56 429.04-vs-448.01 numbers we characterize):

      * ``DISABLE_LOG_STATS=0`` re-enables vLLM's stat logger so the "Avg
        generation throughput" meter + SpecDecoding (E[accept]) lines are emitted
        (the manifest ships =1 for leaderboard speed). Host-side only; no GPU
        compute change.
      * ``VLLM_USE_FLASHINFER_SAMPLER=0`` — local cuRAND-JIT shim (greedy/PPL
        unaffected).
      * ``STEPTIME`` per-step probe — ON by default to match the #56 regime
        exactly; ``--no-steptime`` drops it (a candidate protocol lever: does
        removing per-step instrumentation tighten the floor?).
    """
    env = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
    }
    if args.steptime:
        expected_steps = max(64, args.num_prompts * args.output_len // 2)
        env.update(serve_profile._steptime_env(expected_steps))
    return env


def run_reuse(args, submission_dir: Path, server_python: Path, out_dir: Path,
              records_fh, ppl_after: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    server_log = out_dir / "server_reuse.log"
    preflight_gpu()
    with harness.LocalServer(submission_dir, server_python=server_python,
                             log_path=server_log, extra_env=build_serve_env(args)) as server:
        for i in range(args.n_runs):
            offset = _server_log_len(server_log)
            print(f"\n[noise:reuse] === run {i+1}/{args.n_runs} (offset={offset}) ===", flush=True)
            rec = timed_decode(
                server, server_python, out_dir, i,
                num_prompts=args.num_prompts, output_len=args.output_len,
                seed=args.seed, log_offset=offset,
                clock_interval_ms=args.clock_interval_ms, settle_s=args.settle_s,
            )
            records.append(rec)
            records_fh.write(json.dumps(rec) + "\n")
            records_fh.flush()
            _print_run(rec)
        if ppl_after:
            _ppl_validity(server, server_python, out_dir)
    return records


def run_fresh(args, submission_dir: Path, server_python: Path, out_dir: Path,
              records_fh) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(args.n_runs):
        server_log = out_dir / f"server_fresh_run{i:02d}.log"
        print(f"\n[noise:fresh] === run {i+1}/{args.n_runs} (fresh server) ===", flush=True)
        preflight_gpu()
        t_load0 = time.time()
        with harness.LocalServer(submission_dir, server_python=server_python,
                                 log_path=server_log, extra_env=build_serve_env(args)) as server:
            server_ready_s = time.time() - t_load0
            rec = timed_decode(
                server, server_python, out_dir, i,
                num_prompts=args.num_prompts, output_len=args.output_len,
                seed=args.seed, log_offset=0,
                clock_interval_ms=args.clock_interval_ms, settle_s=args.settle_s,
            )
            rec["server_ready_s"] = server_ready_s
        records.append(rec)
        records_fh.write(json.dumps(rec) + "\n")
        records_fh.flush()
        _print_run(rec)
    return records


def _ppl_validity(server, server_python, out_dir: Path) -> None:
    """One PPL pass (unchanged-stack validity check). Run AFTER the timed loop."""
    try:
        print("\n[noise] PPL validity check (decode-loop already done)...", flush=True)
        ppl = harness.run_ppl(
            server_python, base_url=server.base_url, model=server.served_model_name,
            out_file=out_dir / "ppl_check.jsonl",
            summary_file=out_dir / "ppl_check.summary.json",
        )
        print(f"[noise] PPL validity: ppl={ppl.get('ppl')} num_records={ppl.get('num_records')}", flush=True)
    except Exception as exc:  # validity check must never discard the timing data
        print(f"[noise] WARN PPL validity check failed: {exc}", flush=True)


def _print_run(rec: dict[str, Any]) -> None:
    series = rec.get("gen_tps_series") or []
    print(
        f"[noise] run {rec['run_idx']:02d}: steady_tps={rec.get('steady_gen_tps_mean')} "
        f"steady_nz={rec.get('steady_gen_tps_mean_nonzero')} wall_tps={rec.get('wall_tps'):.2f} "
        f"E[accept]={rec.get('e_accept_exact')} intervals={len(series)} "
        f"sm_load={(rec.get('clock') or {}).get('sm_clock_mhz_load', {}).get('mean')} "
        f"temp_max={(rec.get('clock') or {}).get('temp_c', {}).get('max')}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, records: list[dict[str, Any]], agg: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[noise] wandb_logging import failed ({exc}); skipping wandb", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="tps-noise-floor", agent="lawine",
            name=args.wandb_name or f"lawine/noise-floor-{args.mode}",
            group=args.wandb_group,
            tags=["tps-noise-floor", args.mode, args.submission],
            config={
                "submission": args.submission, "mode": args.mode,
                "n_runs": args.n_runs, "num_prompts": args.num_prompts,
                "output_len": args.output_len, "seed": args.seed,
                "steptime": args.steptime,
            },
        )
    except Exception as exc:
        print(f"[noise] wandb init failed ({exc}); skipping wandb", flush=True)
        return
    if run is None:
        print("[noise] wandb disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return
    try:
        for rec in records:
            metrics = {
                "run/steady_gen_tps_mean": rec.get("steady_gen_tps_mean"),
                "run/steady_gen_tps_mean_nonzero": rec.get("steady_gen_tps_mean_nonzero"),
                "run/wall_tps": rec.get("wall_tps"),
                "run/e_accept_exact": rec.get("e_accept_exact"),
                "run/decode_duration_s": rec.get("decode_duration_s"),
                "run/sm_clock_mhz_load": (rec.get("clock") or {}).get("sm_clock_mhz_load", {}).get("mean"),
                "run/temp_c_max": (rec.get("clock") or {}).get("temp_c", {}).get("max"),
            }
            if "server_ready_s" in rec:
                metrics["run/server_ready_s"] = rec["server_ready_s"]
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, "noise_run", step=rec["run_idx"], metrics=metrics)
        flat = {}
        for mkey, a in agg.items():
            for stat in ("mean", "std", "cv_pct", "min", "max", "range_pct", "median"):
                if isinstance(a, dict) and isinstance(a.get(stat), (int, float)):
                    flat[f"{mkey}/{stat}"] = a[stat]
        wandb_logging.log_summary(run, flat, step=args.n_runs)
        wandb_logging.log_json_artifact(
            run, name=f"noise_floor_{args.mode}", artifact_type="noise-floor",
            data={"records": records, "aggregate": agg},
        )
    except Exception as exc:
        print(f"[noise] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default="fa2sw_precache_kenyan")
    ap.add_argument("--mode", choices=["fresh", "reuse"], required=True)
    ap.add_argument("--n-runs", type=int, default=12)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--steptime", dest="steptime", action="store_true", default=True,
                    help="enable the per-step STEPTIME probe (default; matches the "
                         "canonical serve_profile / #56 measurement env)")
    ap.add_argument("--no-steptime", dest="steptime", action="store_false",
                    help="drop STEPTIME (candidate protocol lever: less per-step perturbation)")
    ap.add_argument("--ppl-once", action="store_true",
                    help="run ONE PPL validity pass after the timed loop (reuse mode)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="tps-noise-floor")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[noise] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[noise] submission={submission_dir.name} server_python={server_python}", flush=True)

    out_dir = (args.out_dir or (OUT_ROOT / f"{args.mode}_n{args.n_runs}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    print(f"[noise] mode={args.mode} n_runs={args.n_runs} workload={args.num_prompts}x{args.output_len} "
          f"seed={args.seed} -> {out_dir}", flush=True)

    t0 = time.time()
    with open(records_path, "w") as records_fh:
        if args.mode == "reuse":
            records = run_reuse(args, submission_dir, server_python, out_dir,
                                records_fh, ppl_after=args.ppl_once)
        else:
            records = run_fresh(args, submission_dir, server_python, out_dir, records_fh)
    elapsed = time.time() - t0

    agg = {
        "steady_gen_tps_mean": aggregate(records, "steady_gen_tps_mean"),
        "steady_gen_tps_mean_nonzero": aggregate(records, "steady_gen_tps_mean_nonzero"),
        "wall_tps": aggregate(records, "wall_tps"),
        "e_accept_exact": aggregate(records, "e_accept_exact"),
    }
    result = {
        "mode": args.mode,
        "submission": args.submission,
        "n_runs": args.n_runs,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed},
        "steptime": args.steptime,
        "elapsed_s": elapsed,
        "git": _git_info(),
        "aggregate": agg,
        "records": records,
    }
    result_path = out_dir / f"noise_floor_{args.mode}.json"
    result_path.write_text(json.dumps(result, indent=2))

    _print_summary(args.mode, agg, elapsed)
    print(f"[noise] artifacts -> {result_path}", flush=True)
    _log_wandb(args, records, agg)
    return 0


def _git_info() -> dict[str, str]:
    try:
        from scripts import wandb_logging
        return wandb_logging.git_info()
    except Exception:
        return {}


def _print_summary(mode: str, agg: dict[str, Any], elapsed: float) -> None:
    print(f"\n[noise] ===== NOISE FLOOR ({mode}) — {elapsed/60:.1f} min =====", flush=True)
    for key in ("steady_gen_tps_mean", "steady_gen_tps_mean_nonzero", "wall_tps", "e_accept_exact"):
        a = agg.get(key, {})
        if a.get("n"):
            print(f"  {key:30s} n={a['n']:2d} mean={a['mean']:.3f} std={a['std']:.3f} "
                  f"CV={a['cv_pct']:.2f}% range={a['min']:.2f}..{a['max']:.2f} "
                  f"({a['range_pct']:.2f}%) median={a['median']:.3f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
