"""bi0 speculative-token depth (K) sweep — PR #774.

LOCAL-only, contract-neutral. Serves ``submissions/int4_mtp_bi0_surgattn``
UNCHANGED except ``NUM_SPECULATIVE_TOKENS`` (a serve-time env the submission's
``serve.py`` already consumes — no served-file edit). The hypothesis: bi0 ships
K=6, but K=6 may not be the TPS-optimal acceptance/overhead trade-off at the
batch-1 (MAX_NUM_SEQS=1) decode the leaderboard scores; the sweep finds the knee.

For each K the driver:
  * preflights the GPU (reap any stale vLLM + wait for VRAM to drain),
  * serves bi0 with ``NUM_SPECULATIVE_TOKENS=K`` under the PR #72 measurement env
    (``DISABLE_LOG_STATS=0`` so vLLM emits the SpecDecoding acceptance lines;
    PyTorch-native sampler for this container's cuRAND-JIT shim — neither changes
    which tokens are emitted, so greedy identity / PPL are untouched),
  * runs 1 warmup decode (discarded) + N>=3 timed decodes on the SAME server and
    reports the **median wall_tps with CV** (the #72 within-session floor:
    ``wall_tps = num_completion_tokens / decode_duration_s`` is the official
    leaderboard ``output_throughput`` definition),
  * captures the decoded **completion-token fingerprint** (sha256 over the
    per-prompt completion-token sha256 vector) for the cross-K greedy-identity
    proof, and verifies the N reps are byte-identical to each other (intra-K
    determinism),
  * reads ``e_accept_exact`` / ``draft_acceptance_rate`` from the server-log
    SpecDecoding counters AND cross-checks them against the Prometheus
    ``/metrics`` ``vllm:spec_decode_*`` counters,
  * runs ONE PPL validity pass AFTER the timing loop (so ``prompt_logprobs`` never
    perturbs a timed window),
  * writes a per-K artifact and ONE wandb run (group ``bi0-k-sweep``).

K=0 = speculation OFF = plain int4 W4A16 AR decode = the **exact-greedy
reference**. At temperature 0 vLLM's rejection sampler short-circuits to
target-argmax, so every K>0 MUST emit byte-identical greedy tokens to K=0; any
cross-K divergence is a BUG (kernel non-determinism), not a quality knob. The K=0
and K>0 arms run on the SAME engine/kernels/quant, so the ONLY removed variable is
speculation depth — a clean controlled comparison.

Reuses the proven local-validation harness end-to-end (``harness.LocalServer`` /
``timed_decode`` / ``build_serve_env`` / ``preflight_gpu`` / ``aggregate`` and
``accept_calibration.parse_prometheus``); it does NOT reinvent the measurement.

Run under the repo ``.venv`` (has wandb). Each K is a self-contained server
lifecycle, so the sweep is RESUMABLE — a completed K banks
``research/bi0_k_sweep/k{K}/arm.json`` and is skipped on re-run, which respects the
hard 90-min/process budget. Entrypoints::

    # single K (one bounded process; the canonical per-K run):
    .venv/bin/python scripts/profiler/bi0_k_sweep.py --k 6

    # full sweep, each K as its OWN child process (hard per-K isolation), then the
    # no-GPU cross-K identity + TPS-vs-K table:
    .venv/bin/python scripts/profiler/bi0_k_sweep.py --k-list 0,2,4,6,8

    # no-GPU plan validation (pre-launch):
    .venv/bin/python scripts/profiler/bi0_k_sweep.py --dry-run

    # no-GPU rollup of already-banked arms:
    .venv/bin/python scripts/profiler/bi0_k_sweep.py --analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
# Reuse the #72 measurement harness verbatim — do NOT reinvent the measurement.
from research.tps_noise_floor.run_noise_floor import (  # noqa: E402
    aggregate,
    build_serve_env,
    preflight_gpu,
    timed_decode,
)
# Reuse the validated Prometheus spec-decode scrape (exact vLLM 0.22.0 metric names).
from scripts.profiler.accept_calibration import parse_prometheus  # noqa: E402

DEFAULT_SUBMISSION = "int4_mtp_bi0_surgattn"
OUT_ROOT = ROOT / "research" / "bi0_k_sweep"
WANDB_GROUP = "bi0-k-sweep"
AGENT = "fern"
# 20% slack over bi0's PPL 2.0058 (PR #774). A K>0 arm that stays greedy-identical
# to K=0 is PPL-identical by construction; this guards a measurement anomaly.
PPL_BAR = 2.42


# ---------------------------------------------------------------------------
# Greedy fingerprint over decoded completion tokens
# ---------------------------------------------------------------------------
def fingerprint_decode_jsonl(decode_jsonl: Path) -> dict[str, Any]:
    """Byte-exact greedy fingerprint of one decode pass.

    Reads the official ``decode_outputs.py`` per-prompt rows (each carries
    ``completion_token_sha256`` over its emitted token-id list) and folds them, in
    prompt ``index`` order, into a single sha256. Two decode passes share a
    fingerprint iff they emitted byte-identical greedy tokens for every prompt — so
    comparing a K>0 fingerprint to K=0's IS the challenge's greedy-identity proof.
    Also returns the per-record (index -> completion sha) map so a divergence can be
    localized to the first divergent prompt.
    """
    rows: list[dict[str, Any]] = []
    for line in decode_jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("index", r.get("dataset_index", 0))))
    per_record: dict[int, str] = {}
    h = hashlib.sha256()
    for r in rows:
        idx = int(r.get("index", r.get("dataset_index", 0)))
        csha = str(r.get("completion_token_sha256", ""))
        psha = str(r.get("prompt_token_sha256", r.get("prompt_sha256", "")))
        per_record[idx] = csha
        h.update(f"{idx}:{psha}:{csha}\n".encode("ascii"))
    return {
        "fingerprint": h.hexdigest(),
        "n_records": len(rows),
        "per_record": per_record,
    }


def first_divergence(ref: dict[int, str], cand: dict[int, str]) -> dict[str, Any] | None:
    """First prompt index where two per-record completion-sha maps differ."""
    for idx in sorted(set(ref) | set(cand)):
        if ref.get(idx) != cand.get(idx):
            return {
                "index": idx,
                "reference_completion_sha256": ref.get(idx),
                "candidate_completion_sha256": cand.get(idx),
            }
    return None


# ---------------------------------------------------------------------------
# Prometheus /metrics spec-decode scrape (cross-check of the log-derived accept)
# ---------------------------------------------------------------------------
def scrape_metrics(base_url: str, K: int, *, timeout_s: float = 30.0) -> dict[str, Any]:
    """GET ``{base_url}/metrics`` and parse the vLLM spec-decode counters.

    Read-only; cumulative over every decode this server served. Empty/absent on
    K=0 (no speculation) or on wheels that don't populate the counters — treated as
    a cross-check, never the sole acceptance source (the server-log parse is)."""
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"populated": False, "error": str(exc)}
    return parse_prometheus(text, max(K, 1))


# ---------------------------------------------------------------------------
# One K arm: serve bi0 @ NUM_SPECULATIVE_TOKENS=K, warmup + N timed reps, PPL
# ---------------------------------------------------------------------------
def _log_len(server: harness.LocalServer) -> int:
    try:
        return Path(server.log_path).stat().st_size if server.log_path else 0
    except OSError:
        return 0


def run_one_k(args, K: int) -> dict[str, Any]:
    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    out_dir = (OUT_ROOT / f"k{K}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"

    # Measurement env (#72: DISABLE_LOG_STATS=0 + native sampler + steptime) with the
    # K override on top. NUM_SPECULATIVE_TOKENS is exactly what bi0's serve.py reads;
    # K=0 -> serve.py skips --speculative-config -> plain int4 AR reference.
    serve_env = build_serve_env(args)
    serve_env["NUM_SPECULATIVE_TOKENS"] = str(K)

    manifest_env = manifest.get("env") or {}
    print(f"\n[ksweep] ===== K={K} ({args.submission}) — serve env override "
          f"NUM_SPECULATIVE_TOKENS={K}, VLLM_BATCH_INVARIANT="
          f"{manifest_env.get('VLLM_BATCH_INVARIANT')}, MAX_NUM_SEQS="
          f"{manifest_env.get('MAX_NUM_SEQS')} =====", flush=True)

    t0 = time.time()
    preflight_gpu()
    t_load0 = time.time()
    records: list[dict[str, Any]] = []
    fp_reps: list[dict[str, Any]] = []
    prom: dict[str, Any] = {}
    ppl_summary: dict[str, Any] = {}
    records_path = out_dir / "records.jsonl"

    with harness.LocalServer(submission_dir, server_python=server_python,
                             log_path=server_log, extra_env=serve_env) as server:
        server_ready_s = time.time() - t_load0
        peak_mem_mib = _gpu_mem_used_mib()

        # --- warmup (discarded): settle CUDA graphs / SM clocks before timing ---
        if args.warmup_prompts > 0:
            print(f"[ksweep:k{K}] warmup decode "
                  f"{args.warmup_prompts}x{args.warmup_len} (discarded)", flush=True)
            try:
                harness.capture_decode(
                    server_python, base_url=server.base_url,
                    model=server.served_model_name,
                    out_file=out_dir / "warmup.jsonl",
                    summary_file=out_dir / "warmup.summary.json",
                    num_prompts=args.warmup_prompts, output_len=args.warmup_len,
                    seed=args.seed,
                )
            except Exception as exc:  # warmup failure must not kill the arm
                print(f"[ksweep:k{K}] WARN warmup failed: {exc}", flush=True)

        # --- N timed reps on the same server (within-session wall_tps floor) ---
        with open(records_path, "w") as fh:
            for i in range(args.n):
                offset = _log_len(server)
                print(f"[ksweep:k{K}] timed rep {i+1}/{args.n} (offset={offset})", flush=True)
                rec = timed_decode(
                    server, server_python, out_dir, i,
                    num_prompts=args.num_prompts, output_len=args.output_len,
                    seed=args.seed, log_offset=offset,
                    clock_interval_ms=args.clock_interval_ms, settle_s=args.settle_s,
                )
                rec["K"] = K
                # timed_decode returns the raw SpecDecoding counters but not the
                # ratio; derive draft_acceptance_rate = accepted/drafted here so it
                # aggregates (e_accept_exact = 1 + K*draft_acceptance_rate).
                ta, td = rec.get("total_accepted_tokens"), rec.get("total_drafted_tokens")
                if isinstance(ta, (int, float)) and isinstance(td, (int, float)) and td:
                    rec["draft_acceptance_rate"] = ta / td
                records.append(rec)
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                peak_mem_mib = max(peak_mem_mib or 0, _gpu_mem_used_mib() or 0)
                # Fingerprint this rep's decoded tokens (intra-K determinism check).
                decode_jsonl = out_dir / "decode" / f"run{i:02d}.jsonl"
                if decode_jsonl.exists():
                    fp = fingerprint_decode_jsonl(decode_jsonl)
                    fp["rep"] = i
                    fp_reps.append(fp)
                print(f"[ksweep:k{K}] rep {i:02d}: wall_tps="
                      f"{_fmt(rec.get('wall_tps'))} E[accept]={_fmt(rec.get('e_accept_exact'))} "
                      f"draft_accept={_fmt(rec.get('draft_acceptance_rate'))} "
                      f"fp={(fp_reps[-1]['fingerprint'][:12] if fp_reps else 'NA')}", flush=True)

        # --- /metrics scrape (cumulative spec-decode counters; cross-check) ---
        prom = scrape_metrics(server.base_url, K)

        # --- PPL validity pass AFTER the timing loop ---
        if args.no_ppl:
            print(f"[ksweep:k{K}] PPL pass skipped (--no-ppl)", flush=True)
            ppl_summary = {"skipped": True}
        else:
            try:
                ppl_summary = harness.run_ppl(
                    server_python, base_url=server.base_url,
                    model=server.served_model_name,
                    out_file=out_dir / "ppl.jsonl",
                    summary_file=out_dir / "ppl.summary.json",
                )
            except Exception as exc:  # validity check must not discard timing data
                print(f"[ksweep:k{K}] WARN PPL pass failed: {exc}", flush=True)
                ppl_summary = {"error": str(exc)}

    elapsed_s = time.time() - t0

    # Intra-K determinism: every rep must be byte-identical (greedy => deterministic).
    intra_k_identical = bool(fp_reps) and all(
        f["fingerprint"] == fp_reps[0]["fingerprint"] for f in fp_reps
    )
    fingerprint = fp_reps[0]["fingerprint"] if fp_reps else None
    per_record = fp_reps[0]["per_record"] if fp_reps else {}

    wall = aggregate(records, "wall_tps")
    e_accept = aggregate(records, "e_accept_exact")
    draft_accept = aggregate(records, "draft_acceptance_rate")
    ppl_val = ppl_summary.get("ppl") if isinstance(ppl_summary, dict) else None

    arm: dict[str, Any] = {
        "schema": "bi0_k_sweep_arm/v1",
        "complete": True,
        "K": K,
        "submission": args.submission,
        "n": args.n,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "warmup": f"{args.warmup_prompts}x{args.warmup_len}"},
        "measurement_env": serve_env,
        "manifest_env": manifest_env,
        "server_ready_s": server_ready_s,
        "elapsed_s": elapsed_s,
        "peak_gpu_mem_mib": peak_mem_mib,
        "wall_tps": wall,
        "e_accept_exact": e_accept,
        "draft_acceptance_rate": draft_accept,
        "prometheus_spec_decode": prom,
        "ppl": ppl_val,
        "ppl_summary": ppl_summary,
        "ppl_bar": PPL_BAR,
        "ppl_within_bar": (ppl_val is not None and ppl_val <= PPL_BAR),
        "fingerprint": fingerprint,
        "fingerprint_per_record": {str(k): v for k, v in per_record.items()},
        "fingerprint_reps": [{"rep": f["rep"], "fingerprint": f["fingerprint"],
                              "n_records": f["n_records"]} for f in fp_reps],
        "intra_k_identical": intra_k_identical,
        "git": _git_info(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    arm_path = out_dir / "arm.json"
    arm_path.write_text(json.dumps(arm, indent=2, default=str))
    print(f"[ksweep:k{K}] wall_tps median={_fmt(wall.get('median'))} "
          f"CV={_fmt(wall.get('cv_pct'))}% E[accept]={_fmt(e_accept.get('mean'))} "
          f"ppl={_fmt(ppl_val)} intra_k_identical={intra_k_identical} "
          f"peak_mem={peak_mem_mib}MiB -> {arm_path}", flush=True)
    _log_wandb_arm(args, arm)
    return arm


def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", "0"], capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# wandb (one run per K)
# ---------------------------------------------------------------------------
def _log_wandb_arm(args, arm: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[ksweep] wandb_logging import failed ({exc}); skipping wandb", flush=True)
        return
    K = arm["K"]
    try:
        run = wandb_logging.init_wandb_run(
            job_type="bi0-k-sweep", agent=AGENT,
            name=args.wandb_name or f"{AGENT}/bi0-ksweep-k{K}",
            group=args.wandb_group,
            tags=["bi0-k-sweep", args.submission, f"k{K}"],
            config={
                "K": K, "submission": args.submission, "n": args.n,
                "num_prompts": args.num_prompts, "output_len": args.output_len,
                "seed": args.seed,
                "warmup": arm["workload"]["warmup"],
                "VLLM_BATCH_INVARIANT": arm["manifest_env"].get("VLLM_BATCH_INVARIANT"),
                "MAX_NUM_SEQS": arm["manifest_env"].get("MAX_NUM_SEQS"),
                "ppl_bar": PPL_BAR,
            },
        )
    except Exception as exc:
        print(f"[ksweep] wandb init failed ({exc}); skipping wandb", flush=True)
        return
    if run is None:
        print("[ksweep] wandb disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return
    try:
        # per-rep series
        records_path = (OUT_ROOT / f"k{K}" / "records.jsonl")
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                metrics = {
                    "rep/wall_tps": rec.get("wall_tps"),
                    "rep/e_accept_exact": rec.get("e_accept_exact"),
                    "rep/draft_acceptance_rate": rec.get("draft_acceptance_rate"),
                    "rep/decode_duration_s": rec.get("decode_duration_s"),
                    "rep/sm_clock_mhz_load": (rec.get("clock") or {}).get(
                        "sm_clock_mhz_load", {}).get("mean"),
                }
                metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb_logging.log_event(run, "ksweep_rep", step=int(rec.get("run_idx", 0)),
                                        metrics=metrics)
        flat: dict[str, Any] = {"K": K}
        for mkey in ("wall_tps", "e_accept_exact", "draft_acceptance_rate"):
            a = arm.get(mkey) or {}
            for stat in ("median", "mean", "std", "cv_pct", "min", "max"):
                if isinstance(a.get(stat), (int, float)):
                    flat[f"{mkey}/{stat}"] = a[stat]
        prom = arm.get("prometheus_spec_decode") or {}
        for pk in ("mean_tokens_per_step_E_T", "num_drafts", "num_accepted_tokens",
                   "num_draft_tokens"):
            if isinstance(prom.get(pk), (int, float)):
                flat[f"prometheus/{pk}"] = prom[pk]
        for k_, v_ in (("ppl", arm.get("ppl")),
                       ("ppl_within_bar", 1.0 if arm.get("ppl_within_bar") else 0.0),
                       ("intra_k_identical", 1.0 if arm.get("intra_k_identical") else 0.0),
                       ("peak_gpu_mem_mib", arm.get("peak_gpu_mem_mib")),
                       ("server_ready_s", arm.get("server_ready_s"))):
            if isinstance(v_, (int, float)):
                flat[k_] = v_
        wandb_logging.log_summary(run, flat, step=args.n)
        if arm.get("fingerprint"):
            run.summary["fingerprint"] = arm["fingerprint"]
        wandb_logging.log_json_artifact(
            run, name=f"bi0_ksweep_k{K}", artifact_type="bi0-k-sweep", data=arm)
    except Exception as exc:
        print(f"[ksweep] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# analyze: cross-K greedy identity + TPS-vs-K table
# ---------------------------------------------------------------------------
def load_arms() -> dict[int, dict[str, Any]]:
    arms: dict[int, dict[str, Any]] = {}
    if not OUT_ROOT.exists():
        return arms
    for arm_path in sorted(OUT_ROOT.glob("k*/arm.json")):
        try:
            arm = json.loads(arm_path.read_text())
        except Exception:
            continue
        if arm.get("complete") and isinstance(arm.get("K"), int):
            arms[arm["K"]] = arm
    return arms


def analyze(args) -> dict[str, Any]:
    arms = load_arms()
    if not arms:
        raise SystemExit(f"no banked arms under {OUT_ROOT} (run --k / --k-list first)")
    Ks = sorted(arms)
    ref_K = 0 if 0 in arms else Ks[0]
    ref_arm = arms[ref_K]
    ref_per_record = {int(k): v for k, v in (ref_arm.get("fingerprint_per_record") or {}).items()}
    ref_fp = ref_arm.get("fingerprint")
    if ref_K != 0:
        print(f"[ksweep:analyze] WARN K=0 reference absent; using K={ref_K} as identity "
              "reference (cross-K identity is then relative, not vs spec-off)", flush=True)

    rows: list[dict[str, Any]] = []
    for K in Ks:
        arm = arms[K]
        cand_per_record = {int(k): v for k, v in (arm.get("fingerprint_per_record") or {}).items()}
        identical = (arm.get("fingerprint") == ref_fp) if ref_fp else None
        divergence = None if (identical or K == ref_K) else first_divergence(ref_per_record, cand_per_record)
        w = arm.get("wall_tps") or {}
        ea = arm.get("e_accept_exact") or {}
        da = arm.get("draft_acceptance_rate") or {}
        prom = arm.get("prometheus_spec_decode") or {}
        rows.append({
            "K": K,
            "wall_tps_median": w.get("median"),
            "wall_tps_cv_pct": w.get("cv_pct"),
            "wall_tps_n": w.get("n"),
            "e_accept_exact_mean": ea.get("mean"),
            "draft_acceptance_rate_mean": da.get("mean"),
            "prom_E_T": prom.get("mean_tokens_per_step_E_T"),
            "prom_populated": prom.get("populated"),
            "ppl": arm.get("ppl"),
            "ppl_within_bar": arm.get("ppl_within_bar"),
            "greedy_identical_to_ref": identical,
            "intra_k_identical": arm.get("intra_k_identical"),
            "first_divergence": divergence,
            "peak_gpu_mem_mib": arm.get("peak_gpu_mem_mib"),
        })

    # Per-arm quality tier (PR #774 + human directive #784):
    #   * "strict-safe": byte-identical greedy to the K=0 reference AND PPL<=bar —
    #     the hard quality-safe class (no behavioural change at all).
    #   * "candidate"  : NOT byte-identical BUT PPL<=bar. bi0 runs BI=0 + int4
    #     Marlin, which is not batch-invariant across the AR-vs-verify forward, so a
    #     cross-K argmax drift is EXPECTED, not necessarily a bug; per #784 we report
    #     such an arm as a speed candidate rather than discarding it. It still needs
    #     a downstream task-quality (5%-band) check before an HF submission.
    #   * "ppl-fail"   : PPL over bar — disqualified.
    for r in rows:
        if not r["ppl_within_bar"]:
            r["tier"] = "ppl-fail"
        elif r["greedy_identical_to_ref"]:
            r["tier"] = "strict-safe"
        else:
            r["tier"] = "candidate"

    def _has_tps(r: dict[str, Any]) -> bool:
        return isinstance(r["wall_tps_median"], (int, float))

    # Strict winner = fastest byte-identical + PPL-safe arm (K=0 ref is identical to
    # itself but never the speed winner). Candidate winner = fastest PPL-safe arm
    # regardless of identity (#784); equals the strict winner when identity holds.
    strict_safe = [r for r in rows if r["tier"] == "strict-safe" and _has_tps(r)]
    ppl_safe = [r for r in rows if r["ppl_within_bar"] and _has_tps(r)]
    best = max(strict_safe, key=lambda r: r["wall_tps_median"]) if strict_safe else None
    best_candidate = max(ppl_safe, key=lambda r: r["wall_tps_median"]) if ppl_safe else None
    baseline_row = next((r for r in rows if r["K"] == 6), None)

    summary = {
        "schema": "bi0_k_sweep_summary/v2",
        "reference_K": ref_K,
        "reference_fingerprint": ref_fp,
        "ppl_bar": PPL_BAR,
        "ks": Ks,
        "rows": rows,
        # strict quality-safe winner (byte-identical greedy + PPL<=bar)
        "best_K": (best["K"] if best else None),
        "best_wall_tps_median": (best["wall_tps_median"] if best else None),
        "best_ppl": (best["ppl"] if best else None),
        # #784 relaxed candidate winner (PPL<=bar; byte-identity NOT required)
        "best_candidate_K": (best_candidate["K"] if best_candidate else None),
        "best_candidate_wall_tps_median": (best_candidate["wall_tps_median"] if best_candidate else None),
        "best_candidate_ppl": (best_candidate["ppl"] if best_candidate else None),
        "best_candidate_greedy_identical": (best_candidate["greedy_identical_to_ref"] if best_candidate else None),
        "baseline_k6_wall_tps_median": (baseline_row.get("wall_tps_median") if baseline_row else None),
        "all_greedy_identical": all(
            (r["greedy_identical_to_ref"] is True) for r in rows if r["K"] != ref_K) if len(rows) > 1 else None,
        "git": _git_info(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    _print_table(summary)
    print(f"[ksweep:analyze] summary -> {summary_path}", flush=True)
    if not args.no_wandb and not args.no_rollup_wandb:
        _log_wandb_rollup(args, summary)
    return summary


def _delta_vs_k6(summary: dict[str, Any], tps: Any) -> str:
    base = summary.get("baseline_k6_wall_tps_median")
    return (f"{100.0 * (tps - base) / base:+.2f}% vs K=6"
            if isinstance(base, (int, float)) and isinstance(tps, (int, float)) and base else "")


def _print_table(summary: dict[str, Any]) -> None:
    print("\n[ksweep] ===== TPS vs K (median wall_tps) =====", flush=True)
    hdr = (f"{'K':>3} {'wall_tps':>10} {'CV%':>7} {'E[accept]':>10} {'draft_acc':>10} "
           f"{'prom_E_T':>9} {'PPL':>7} {'identical':>10} {'tier':>11}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in summary["rows"]:
        ident = ("ref" if r["K"] == summary["reference_K"]
                 else ("YES" if r["greedy_identical_to_ref"] else "NO"))
        print(f"{r['K']:>3} {_fmt(r['wall_tps_median'],2):>10} {_fmt(r['wall_tps_cv_pct'],3):>7} "
              f"{_fmt(r['e_accept_exact_mean'],3):>10} {_fmt(r['draft_acceptance_rate_mean'],3):>10} "
              f"{_fmt(r['prom_E_T'],3):>9} {_fmt(r['ppl'],4):>7} {ident:>10} "
              f"{str(r.get('tier','—')):>11}", flush=True)
        if r.get("first_divergence"):
            print(f"      first divergence @ prompt index {r['first_divergence']['index']}", flush=True)
    if summary.get("best_K") is not None:
        print(f"\n  >>> best STRICT quality-safe (byte-identical + PPL<=bar) K={summary['best_K']} "
              f"wall_tps={_fmt(summary.get('best_wall_tps_median'),2)} "
              f"ppl={_fmt(summary.get('best_ppl'),4)} "
              f"{_delta_vs_k6(summary, summary.get('best_wall_tps_median'))}", flush=True)
    else:
        print("\n  >>> no STRICT (byte-identical + PPL-safe) K found "
              "(expected if BI=0 int4 Marlin drifts across K)", flush=True)
    # #784 candidate winner: fastest PPL-safe arm even if it drifted from K=0.
    cand_K = summary.get("best_candidate_K")
    if cand_K is not None and cand_K != summary.get("best_K"):
        ident = summary.get("best_candidate_greedy_identical")
        note = "byte-identical" if ident else "DRIFTED (needs 5%-band task-quality check)"
        print(f"  >>> best #784 CANDIDATE (PPL<=bar, identity optional) K={cand_K} "
              f"wall_tps={_fmt(summary.get('best_candidate_wall_tps_median'),2)} "
              f"ppl={_fmt(summary.get('best_candidate_ppl'),4)} [{note}] "
              f"{_delta_vs_k6(summary, summary.get('best_candidate_wall_tps_median'))}", flush=True)


def _log_wandb_rollup(args, summary: dict[str, Any]) -> None:
    try:
        from scripts import wandb_logging
    except Exception:
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="bi0-k-sweep-rollup", agent=AGENT,
            name=f"{AGENT}/bi0-ksweep-rollup", group=args.wandb_group,
            tags=["bi0-k-sweep", "rollup", args.submission],
            config={"ks": summary["ks"], "reference_K": summary["reference_K"],
                    "ppl_bar": PPL_BAR, "submission": args.submission},
        )
    except Exception:
        return
    if run is None:
        return
    try:
        for r in summary["rows"]:
            metrics = {k: v for k, v in {
                "K": r["K"],
                "wall_tps_median": r["wall_tps_median"],
                "e_accept_exact_mean": r["e_accept_exact_mean"],
                "draft_acceptance_rate_mean": r["draft_acceptance_rate_mean"],
                "prom_E_T": r["prom_E_T"],
                "ppl": r["ppl"],
                "greedy_identical": 1.0 if r["greedy_identical_to_ref"] else 0.0,
            }.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, "ksweep_K", step=r["K"], metrics=metrics)
        flat = {k: v for k, v in {
            "best_K": summary.get("best_K"),
            "best_wall_tps_median": summary.get("best_wall_tps_median"),
            "best_ppl": summary.get("best_ppl"),
            "best_candidate_K": summary.get("best_candidate_K"),
            "best_candidate_wall_tps_median": summary.get("best_candidate_wall_tps_median"),
            "best_candidate_ppl": summary.get("best_candidate_ppl"),
            "best_candidate_greedy_identical": 1.0 if summary.get("best_candidate_greedy_identical") else 0.0,
            "baseline_k6_wall_tps_median": summary.get("baseline_k6_wall_tps_median"),
            "all_greedy_identical": 1.0 if summary.get("all_greedy_identical") else 0.0,
        }.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, flat, step=len(summary["rows"]))
        wandb_logging.log_json_artifact(run, name="bi0_ksweep_summary",
                                        artifact_type="bi0-k-sweep", data=summary)
    except Exception as exc:
        print(f"[ksweep] WARN rollup wandb error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# dry-run: no-GPU plan validation
# ---------------------------------------------------------------------------
def dry_run(args, k_list: list[int]) -> None:
    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    exists = submission_dir.exists()
    print("[ksweep:dry-run] ===== PLAN (no GPU, no serve) =====", flush=True)
    print(f"  submission     : {args.submission}  (exists={exists})", flush=True)
    print(f"  K list         : {k_list}", flush=True)
    print(f"  workload       : {args.num_prompts} prompts x {args.output_len} tokens, "
          f"seed={args.seed}, N={args.n} timed reps, warmup={args.warmup_prompts}x{args.warmup_len}", flush=True)
    print(f"  output root    : {OUT_ROOT}", flush=True)
    print(f"  wandb          : group={args.wandb_group} agent={AGENT} "
          f"names={[f'{AGENT}/bi0-ksweep-k{K}' for K in k_list]} (disabled={args.no_wandb})", flush=True)
    print(f"  PPL bar        : <= {PPL_BAR}", flush=True)
    # Show the per-K serve env override that LocalServer would apply on top of the
    # manifest env (the only changed variable across arms is NUM_SPECULATIVE_TOKENS).
    base_env = build_serve_env(args)
    print(f"  measurement env: {base_env}", flush=True)
    if exists:
        manifest = harness.load_manifest(submission_dir)
        print(f"  manifest env   : {manifest.get('env')}", flush=True)
        print(f"  serve cmd      : {manifest.get('serve')}  model_id={manifest.get('model_id')}", flush=True)
    for K in k_list:
        env_k = dict(base_env)
        env_k["NUM_SPECULATIVE_TOKENS"] = str(K)
        banked = (OUT_ROOT / f"k{K}" / "arm.json")
        status = "BANKED (would skip unless --force)" if banked.exists() else "pending"
        spec = "spec OFF (int4 AR reference)" if K == 0 else f"spec depth {K}"
        print(f"    K={K:<2} NUM_SPECULATIVE_TOKENS={K} [{spec}] -> {status}", flush=True)
    # Prove the fingerprint folder is correct on synthetic decode rows (no GPU).
    _selftest_fingerprint()
    print("[ksweep:dry-run] OK — plan valid, fingerprint self-test passed. No GPU touched.", flush=True)


def _selftest_fingerprint() -> None:
    import tempfile
    rows_a = [{"index": 0, "prompt_token_sha256": "p0", "completion_token_sha256": "c0"},
              {"index": 1, "prompt_token_sha256": "p1", "completion_token_sha256": "c1"}]
    rows_b = [{"index": 1, "prompt_token_sha256": "p1", "completion_token_sha256": "c1"},
              {"index": 0, "prompt_token_sha256": "p0", "completion_token_sha256": "c0"}]
    rows_c = [{"index": 0, "prompt_token_sha256": "p0", "completion_token_sha256": "cX"},
              {"index": 1, "prompt_token_sha256": "p1", "completion_token_sha256": "c1"}]
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        for name, rows in (("a", rows_a), ("b", rows_b), ("c", rows_c)):
            (dd / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        fa = fingerprint_decode_jsonl(dd / "a.jsonl")
        fb = fingerprint_decode_jsonl(dd / "b.jsonl")
        fc = fingerprint_decode_jsonl(dd / "c.jsonl")
    assert fa["fingerprint"] == fb["fingerprint"], "fingerprint must be order-invariant"
    assert fa["fingerprint"] != fc["fingerprint"], "fingerprint must catch a token diff"
    div = first_divergence(fa["per_record"], fc["per_record"])
    assert div and div["index"] == 0, "first_divergence must localize the diff"
    print(f"  fingerprint self-test: order-invariant OK, diff-sensitive OK, "
          f"first-divergence@{div['index']} OK", flush=True)


# ---------------------------------------------------------------------------
# k-list driver: each K as its own child process (hard per-K isolation), resumable
# ---------------------------------------------------------------------------
def run_k_list(args, k_list: list[int]) -> int:
    start = time.time()
    done: list[int] = []
    for K in k_list:
        banked = OUT_ROOT / f"k{K}" / "arm.json"
        if banked.exists() and not args.force:
            try:
                if json.loads(banked.read_text()).get("complete"):
                    print(f"[ksweep] K={K} already banked ({banked}); skipping (resume). "
                          "Use --force to re-run.", flush=True)
                    done.append(K)
                    continue
            except Exception:
                pass
        if args.max_minutes and (time.time() - start) / 60.0 >= args.max_minutes:
            print(f"[ksweep] budget {args.max_minutes} min reached; stopping before K={K}. "
                  f"Re-invoke to resume (banked: {done}).", flush=True)
            break
        child = [sys.executable, str(Path(__file__).resolve()), "--k", str(K),
                 "--submission", args.submission, "--n", str(args.n),
                 "--num-prompts", str(args.num_prompts), "--output-len", str(args.output_len),
                 "--seed", str(args.seed), "--warmup-prompts", str(args.warmup_prompts),
                 "--warmup-len", str(args.warmup_len), "--wandb-group", args.wandb_group]
        if args.no_wandb:
            child.append("--no-wandb")
        if args.force:
            child.append("--force")
        if args.no_ppl:
            child.append("--no-ppl")
        print(f"[ksweep] === launching child for K={K}: {' '.join(child)} ===", flush=True)
        rc = subprocess.run(child, check=False).returncode
        if rc != 0:
            print(f"[ksweep] WARN child for K={K} exited rc={rc}; continuing to next K "
                  "(banked arms preserved).", flush=True)
        else:
            done.append(K)
    print(f"[ksweep] k-list complete; banked K={done}. Running analyze...", flush=True)
    analyze(args)
    return 0


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def _git_info() -> dict[str, Any]:
    try:
        from scripts import wandb_logging
        return wandb_logging.git_info()
    except Exception:
        return {}


def _fmt(v: Any, p: int = 3) -> str:
    return f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "—"


def parse_k_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit(f"--k-list parsed empty from {spec!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=DEFAULT_SUBMISSION,
                    help="submission dir under submissions/ (default: int4_mtp_bi0_surgattn = bi0)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--k", type=int, default=None, help="run a single K arm (bounded process)")
    mode.add_argument("--k-list", default="0,2,4,6,8",
                      help="comma K list; each K runs as its OWN child process, then analyze "
                           "(default: 0,2,4,6,8; K=0=spec-off reference)")
    # --analyze / --dry-run are no-GPU actions that compose with --k / --k-list
    # (e.g. `--k 6 --dry-run` previews a single-K plan), so they are NOT in the
    # mutually-exclusive serving group above.
    ap.add_argument("--analyze", action="store_true", help="no-GPU rollup of banked arms")
    ap.add_argument("--dry-run", action="store_true", help="no-GPU plan validation + self-test")
    ap.add_argument("--n", type=int, default=3, help="timed reps per K (median-of-N; >=3)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS,
                    help="prompts per timed decode (default 128 = official audit set)")
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--warmup-prompts", type=int, default=8, help="warmup decode prompts (discarded)")
    ap.add_argument("--warmup-len", type=int, default=128, help="warmup decode output_len (discarded)")
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    # build_serve_env(args) reads args.steptime / args.num_prompts / args.output_len.
    ap.add_argument("--steptime", dest="steptime", action="store_true", default=True)
    ap.add_argument("--no-steptime", dest="steptime", action="store_false")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="in --k-list mode, stop launching new K once elapsed exceeds this "
                         "(0=run all; resumable since each K banks its artifact)")
    ap.add_argument("--force", action="store_true", help="re-run a K even if already banked")
    ap.add_argument("--no-ppl", action="store_true",
                    help="skip the per-K PPL validity pass (smoke/debug only; full runs keep it)")
    ap.add_argument("--wandb-name", default=None, help="override the single-K wandb run name")
    ap.add_argument("--wandb-group", default=WANDB_GROUP)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-rollup-wandb", action="store_true",
                    help="skip the analyze-mode rollup wandb run (per-K runs still log)")
    args = ap.parse_args(argv)

    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    if args.dry_run:
        dry_run(args, parse_k_list(args.k_list) if args.k is None else [args.k])
        return 0
    if args.analyze:
        analyze(args)
        return 0

    for note in paths.prepare_local_gpu_env():
        print(f"[ksweep] {note}", flush=True)

    if args.k is not None:
        run_one_k(args, args.k)
        return 0
    return run_k_list(args, parse_k_list(args.k_list))


if __name__ == "__main__":
    raise SystemExit(main())
