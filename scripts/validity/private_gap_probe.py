#!/usr/bin/env python
"""Predict a submission's public->private TPS gap LOCALLY, before spending an HF Job.

WHY (PR #44): above ~286 TPS the binding constraint is the challenge's **private
re-run**. Honest speculative-decoder stacks lose 4-9% TPS on the private prompt
set and die on the 5% reproduction rule (a 7.2% public->private drop just
invalidated `firfir-cast`). Today that gap is only discovered by burning HF-Jobs
quota and getting invalidated. This probe measures the gap on the assigned A10G
using the *official* benchmark path (sglang bench_serving, single-stream,
output_len 512) on two prompt sets that differ ONLY in distribution:

  * public  : the official 128 reasoning prompts (eval_prompts_sharegpt.json)
  * private : a chat-heavy ShareGPT proxy, length-matched per-prompt to public
              (built by scripts/validity/build_private_proxy.py)

The absolute local A10G TPS is exploratory-only; the RELATIVE public->private
ratio is the signal -- and it is exactly what the private re-run gates on.

TWO gap mechanisms are separated:
  1. drafter acceptance  -- E_accept (mean acceptance length, tokens/target-step)
     measured from vLLM's spec-decode counters; the chat distribution is the
     drafter's weakest, so E_accept drops vs reasoning.
  2. public-overfit precache -- fa2sw_precache_kenyan replays the *public* bench
     prompts into the prefix cache during warmup; on a novel (private) set those
     hits vanish. Controlled here via the `precache` mode of each scenario.

Scenarios (each on a fresh server, the only changed variable named):
  leaderboard   precache=public, bench=public   -> the number you'd submit
  private_rerun precache=off,    bench=private   -> the number the verifier sees
  public_cold   precache=off,    bench=public    -> isolates the precache benefit

Headline gap   = (leaderboard - private_rerun) / leaderboard.
Precache benefit = (leaderboard - public_cold) / leaderboard.
Distribution gap = (public_cold - private_rerun) / public_cold  (both precache-cold;
                   the pure drafter/length effect, explained by the E_accept drop).

LOCAL ONLY. This launches no HF Job and makes no submission.

Honesty caveats (carried from PR #38):
  * the proxy is NOT the real private set -- this is a calibrated early warning,
    not an exact predictor; if the proxy says <5% but real stacks die >5%, the
    proxy is too easy and should be pushed toward the chat tail.
  * served greedy decode on this A10G is run-to-run non-deterministic
    (PR #38: FA_SLIDING reduction noise); TPS/E_accept are stable run aggregates,
    but treat sub-1% TPS deltas as noise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

# Official benchmark protocol (hf_bucket_single_job.py) -- must match exactly so
# the local ratio lines up with the official public/private re-run.
BENCHMARK_DEPENDENCIES = [
    "sglang==0.5.2",
    "transformers==5.9.0",
    "jinja2==3.1.6",
    "pybase64==1.4.3",
    "pydantic==2.13.4",
]
OUTPUT_LEN = 512
MAX_CONCURRENCY = 1
REQUEST_RATE = "inf"
WARMUP_REQUESTS = 4
SEED = 1
# osoi5-v0-baked carries the unmodified gemma-4-E4B-it tokenizer (vocab 262144);
# using the local copy keeps the probe hermetic. Falls back to the hub id.
DEFAULT_TOKENIZER = "/tmp/osoi5-v0-baked"
PRIVATE_DEFAULT = REPO / "data" / "private_proxy_sharegpt.json"


def _bench_env(bench_python: Path) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(bench_python.parent.parent)
    env["PATH"] = f"{bench_python.parent}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_bench(
    bench_python: Path, bench_env: dict[str, str], *, base_url: str, model: str,
    dataset: Path, num_prompts: int, output_file: Path, tokenizer: str,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    """Run the official sglang bench_serving exactly as the HF Job does and return
    the parsed last-line result (output_throughput is the official `tps`)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bench_python), "-m", "sglang.bench_serving",
        "--backend", "vllm-chat",
        "--base-url", base_url.rstrip("/"),
        "--model", model,
        "--tokenizer", tokenizer,
        "--dataset-name", "sharegpt",
        "--dataset-path", str(dataset),
        "--sharegpt-output-len", str(OUTPUT_LEN),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(MAX_CONCURRENCY),
        "--request-rate", REQUEST_RATE,
        "--warmup-requests", str(WARMUP_REQUESTS),
        "--seed", str(SEED),
        "--extra-request-body", json.dumps({"ignore_eos": True}),
        "--output-file", str(output_file),
        "--output-details", "--disable-stream", "--disable-tqdm",
    ]
    print("[bench]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=bench_env, timeout=timeout_s,
                          capture_output=True, text=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-12:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_serving rc={proc.returncode}\nSTDOUT(tail):\n{tail}\n"
            f"STDERR(tail):\n{chr(10).join((proc.stderr or '').splitlines()[-12:])}")
    lines = output_file.read_text().strip().splitlines()
    result = json.loads(lines[-1])
    total_tps = (result["total_input_tokens"] + result["total_output_tokens"]) / result["duration"]
    return {
        "tps": result["output_throughput"],
        "total_tps": total_tps,
        "completed": result["completed"],
        "duration_s": result["duration"],
        "total_input_tokens": result["total_input_tokens"],
        "total_output_tokens": result["total_output_tokens"],
        "mean_e2e_latency_ms": result.get("mean_e2e_latency_ms"),
        "request_throughput_req_s": result.get("request_throughput"),
    }


def scrape_counters(base_url: str) -> dict[str, Any]:
    try:
        text = serve_profile._get_text(f"{base_url}/metrics")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return serve_profile.parse_spec_metrics(text)


def acceptance_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Per-bench E_accept from cumulative spec-decode counter deltas.

    E_accept (mean acceptance length) = 1 + accepted/drafts = mean tokens emitted
    per target forward pass (1..1+K). accepted_per_step = E_accept - 1 is the PR's
    'mean accepted-tok/step'. accept_rate = accepted/draft_tokens."""
    def d(key: str) -> float | None:
        a, b = after.get(key), before.get(key)
        if a is None or b is None:
            return None
        return a - b

    d_drafts, d_acc, d_dtok = d("num_drafts"), d("num_accepted_tokens"), d("num_draft_tokens")
    out: dict[str, Any] = {
        "delta_num_drafts": d_drafts,
        "delta_num_accepted_tokens": d_acc,
        "delta_num_draft_tokens": d_dtok,
    }
    if d_drafts and d_acc is not None:
        out["e_accept"] = 1.0 + d_acc / d_drafts
        out["accepted_per_step"] = d_acc / d_drafts
    if d_dtok and d_acc is not None:
        out["accept_rate"] = d_acc / d_dtok
    return out


def precache_env(mode: str, public: Path, private: Path, num_prompts: int) -> dict[str, str]:
    """Override the manifest precache so each scenario names its one changed variable.
    Locally the manifest's PRECACHE_DATASET (/harness/...) does not exist, so the
    stack ungates WITHOUT precache by default; we point it explicitly per scenario."""
    if mode == "off":
        return {"PRECACHE_BENCH": "0"}
    target = public if mode == "public" else private
    return {
        "PRECACHE_BENCH": "1",
        "PRECACHE_REQUIRE": "1",
        "PRECACHE_DATASET": str(Path(target).resolve()),
        "PRECACHE_NUM_PROMPTS": str(num_prompts),
    }


def run_scenario(
    submission: Path, server_python: Path, bench_python: Path, bench_env: dict[str, str],
    out_dir: Path, *, label: str, precache_mode: str, dataset: Path,
    public: Path, private: Path, num_prompts: int, tokenizer: str, run_ppl: bool,
) -> dict[str, Any]:
    log_path = out_dir / f"server_{label}.log"
    extra_env = {
        "DISABLE_LOG_STATS": "0",            # expose spec_decode_* counters (host-side only)
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # cuRAND-free sampler (does not touch logits)
        **precache_env(precache_mode, public, private, num_prompts),
    }
    res: dict[str, Any] = {
        "label": label, "precache_mode": precache_mode, "dataset": str(dataset),
        "num_prompts": num_prompts,
    }
    t_serve = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        res["serve_ready_s"] = round(time.time() - t_serve, 1)
        before = scrape_counters(srv.base_url)
        bench_out = out_dir / f"bench_{label}.jsonl"
        res["bench"] = run_bench(
            bench_python, bench_env, base_url=srv.base_url, model=srv.served_model_name,
            dataset=dataset, num_prompts=num_prompts, output_file=bench_out,
            tokenizer=tokenizer,
        )
        after = scrape_counters(srv.base_url)
        res["acceptance"] = acceptance_delta(before, after)
        res["counters_before"], res["counters_after"] = before, after
        if run_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    bench_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_dir / f"ppl_{label}.jsonl",
                    summary_file=out_dir / f"ppl_{label}.summary.json",
                )
                res["ppl"] = ppl_summary.get("ppl")
                res["ppl_num_tokens"] = ppl_summary.get("num_tokens")
            except Exception as exc:  # noqa: BLE001
                res["ppl_error"] = str(exc)
    # whole-server-run E_accept from vLLM's own log lines, as a cross-check
    try:
        res["spec_log_xcheck"] = serve_profile.parse_spec_log(log_path.read_text())
    except Exception:  # noqa: BLE001
        pass
    b = res["bench"]
    a = res.get("acceptance", {})
    print(f"[scenario {label}] tps={b['tps']:.2f} completed={b['completed']} "
          f"E_accept={a.get('e_accept')} ppl={res.get('ppl')}", flush=True)
    return res


def pct_gap(hi: float, lo: float) -> float:
    return (hi - lo) / hi * 100.0 if hi else float("nan")


def log_report_to_wandb(
    report: dict[str, Any], *, submission: str, wandb_name: str, wandb_group: str,
    report_path: Path | None = None,
) -> None:
    """Push the probe's headline scalars + report.json to W&B (PR #44 optional record).

    Lazy + defensive: never raise, so a missing/disabled W&B can't fail the probe.
    Re-runnable on a saved report.json (no GPU work), mirroring log_greedy_gate_wandb."""
    headline = report.get("headline_public_to_private_gap_pct")
    tps, eacc, ppl = report.get("tps", {}), report.get("e_accept", {}), report.get("ppl", {})
    summary: dict[str, Any] = {
        # primary_metric name the PR gates on, plus the verbose alias
        "private_gap_pct": headline,
        "headline_public_to_private_gap_pct": headline,
        "tps_leaderboard_public": tps.get("leaderboard_public"),
        "tps_private_rerun": tps.get("private_rerun"),
        "tps_public_cold": tps.get("public_cold"),
        "e_accept_public": eacc.get("public"),
        "e_accept_private": eacc.get("private"),
        "e_accept_delta": eacc.get("delta"),
        "ppl_public": ppl.get("public"),
        "ppl_private": ppl.get("private"),
        "ppl_cap": report.get("ppl_cap"),
        "num_prompts": report.get("num_prompts"),
        "would_fail_5pct": bool(headline is not None and headline > 5.0),
        "verdict_5pct_rule": report.get("verdict_5pct_rule"),
    }
    if "decomposition" in report:
        dec = report["decomposition"]
        summary["precache_benefit_on_public_pct"] = dec.get("precache_benefit_on_public_pct")
        summary["distribution_gap_precache_neutral_pct"] = dec.get(
            "distribution_gap_precache_neutral_pct")
    try:
        from scripts.wandb_logging import (
            finish_wandb, init_wandb_run, log_file_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - logging must never break the probe
        print(f"[wandb] logging unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="private-gap-probe",
        agent="senpai",
        name=wandb_name,
        tags=["private-gap-probe", *([wandb_group] if wandb_group else [])],
        config={
            "submission": submission,
            "num_prompts": report.get("num_prompts"),
            "smoke": report.get("smoke"),
            "public_evidence_band": report.get("public_evidence_band"),
            "wandb_group": wandb_group,
            "timestamp": report.get("timestamp"),
        },
    )
    if run is None:
        print("[wandb] run not created (no creds/disabled); report.json is the record", flush=True)
        return
    log_summary(run, summary, step=0)
    if report_path is not None:
        log_file_artifact(run, path=Path(report_path), name="private_gap_report",
                          artifact_type="private-gap-probe-report")
    finish_wandb(run)
    print(f"[wandb] logged run {wandb_name} (group={wandb_group})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=str(REPO / "submissions/fa2sw_precache_kenyan"))
    ap.add_argument("--public", default=str(paths.EVAL_PROMPTS))
    ap.add_argument("--private", default=str(PRIVATE_DEFAULT))
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--smoke", action="store_true",
                    help="8+8 prompts, leaderboard+private_rerun only, to validate plumbing")
    ap.add_argument("--no-decompose", action="store_true",
                    help="skip the public_cold precache-decomposition scenario")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None,
                    help="optional: log the headline gap + report.json to W&B under this group "
                         "(e.g. private-gap-probe). No GPU cost; off by default.")
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None,
                    help="optional W&B run name (default kanna/private-gap-<submission>)")
    args = ap.parse_args()

    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[gpu] {n}", flush=True)

    submission = Path(args.submission)
    public = Path(args.public)
    private = Path(args.private)
    tokenizer = args.tokenizer if Path(args.tokenizer).exists() else paths.TOKENIZER
    num_prompts = 8 if args.smoke else args.num_prompts
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "research/validity/private_gap_probe" / (("smoke-" if args.smoke else "") + ts))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] out_dir={out_dir}", flush=True)

    manifest = harness.load_manifest(submission)
    print("[probe] building/locating server venv (custom vLLM wheel)", flush=True)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print("[probe] building/locating bench venv (sglang)", flush=True)
    bench_python = harness.ensure_server_venv(BENCHMARK_DEPENDENCIES)
    bench_env = _bench_env(bench_python)

    scenarios = [
        ("leaderboard", "public", public, True),
        ("private_rerun", "off", private, True),
    ]
    if not args.smoke and not args.no_decompose:
        scenarios.append(("public_cold", "off", public, False))

    results: dict[str, Any] = {}
    for label, precache_mode, dataset, do_ppl in scenarios:
        results[label] = run_scenario(
            submission, server_python, bench_python, bench_env, out_dir,
            label=label, precache_mode=precache_mode, dataset=dataset,
            public=public, private=private, num_prompts=num_prompts,
            tokenizer=tokenizer, run_ppl=do_ppl,
        )

    lb = results["leaderboard"]["bench"]["tps"]
    pr = results["private_rerun"]["bench"]["tps"]
    eacc_pub = results["leaderboard"]["acceptance"].get("e_accept")
    eacc_priv = results["private_rerun"]["acceptance"].get("e_accept")
    headline = pct_gap(lb, pr)
    report: dict[str, Any] = {
        "timestamp": ts,
        "submission": str(submission),
        "num_prompts": num_prompts,
        "smoke": args.smoke,
        "tps": {"leaderboard_public": lb, "private_rerun": pr},
        "headline_public_to_private_gap_pct": headline,
        "e_accept": {"public": eacc_pub, "private": eacc_priv,
                     "delta": (eacc_pub - eacc_priv) if (eacc_pub and eacc_priv) else None},
        "ppl": {"public": results["leaderboard"].get("ppl"),
                "private": results["private_rerun"].get("ppl")},
        "ppl_cap": 2.42,
        "verdict_5pct_rule": (
            "WOULD-FAIL (>5% private TPS drop -> INVALID)" if headline > 5.0
            else "would-pass (<=5% private TPS drop)"),
        "public_evidence_band": "honest stacks 4-9%; firfir-cast cap 7.2%; "
                                "kenyan-duma claimed precache ~1% to private",
        "scenarios": results,
    }
    if "public_cold" in results:
        pc = results["public_cold"]["bench"]["tps"]
        report["tps"]["public_cold"] = pc
        report["decomposition"] = {
            "precache_benefit_on_public_pct": pct_gap(lb, pc),
            "distribution_gap_precache_neutral_pct": pct_gap(pc, pr),
            "note": "headline = precache_benefit + distribution_gap (approx); "
                    "distribution_gap is the pure drafter/length effect both-cold",
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"[probe] report -> {report_path}", flush=True)

    if args.wandb_group:
        wandb_name = args.wandb_name or f"kanna/private-gap-{submission.name}"
        try:
            log_report_to_wandb(report, submission=str(submission), wandb_name=wandb_name,
                                wandb_group=args.wandb_group, report_path=report_path)
        except Exception as e:  # noqa: BLE001
            # report.json is already written above; optional telemetry must not fail a
            # completed probe (e.g. a broken/partial wandb in the submission venv).
            print(f"[probe] W&B logging failed (non-fatal): {e!r}", flush=True)
    return 0


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else str(x)


def _print_summary(r: dict[str, Any]) -> None:
    t = r["tps"]
    print("\n" + "=" * 64, flush=True)
    print(f"PRIVATE-GAP PROBE  ({'SMOKE ' if r['smoke'] else ''}n={r['num_prompts']})", flush=True)
    print("=" * 64, flush=True)
    print(f"  public  TPS (leaderboard, precache=public) : {_fmt(t['leaderboard_public'])}", flush=True)
    if "public_cold" in t:
        print(f"  public  TPS (cold, precache=off)           : {_fmt(t['public_cold'])}", flush=True)
    print(f"  private TPS (re-run, precache=off)         : {_fmt(t['private_rerun'])}", flush=True)
    print(f"  HEADLINE public->private gap               : {_fmt(r['headline_public_to_private_gap_pct'])} %", flush=True)
    e = r["e_accept"]
    print(f"  E_accept  public / private                 : {_fmt(e['public'])} / {_fmt(e['private'])}  (Δ {_fmt(e['delta'])})", flush=True)
    p = r["ppl"]
    print(f"  PPL       public / private (cap {r['ppl_cap']})       : {_fmt(p['public'])} / {_fmt(p['private'])}", flush=True)
    if "decomposition" in r:
        d = r["decomposition"]
        print(f"  precache benefit on public                 : {_fmt(d['precache_benefit_on_public_pct'])} %", flush=True)
        print(f"  distribution gap (precache-neutral)        : {_fmt(d['distribution_gap_precache_neutral_pct'])} %", flush=True)
    print(f"  VERDICT (5% rule)                          : {r['verdict_5pct_rule']}", flush=True)
    print("=" * 64 + "\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
