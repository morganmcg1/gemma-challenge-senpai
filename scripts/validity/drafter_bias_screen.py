#!/usr/bin/env python
"""Screen the drafter token-frequency logit bias (PR #48) on E_accept.

Hypothesis: adding +b to the top-K frequent tokens in the DRAFTER's candidate
scores raises acceptance rate (E_accept) and thus TPS, with no PPL/greedy-identity
risk (the verifier still emits its own argmax).

DECISION METRIC = E_accept (mean accepted tokens / target step), read straight
from vLLM's spec-decode counters. E_accept is onegraph/CUDA-graph invariant (the
graph only changes HOW the same drafted tokens are produced, not WHICH), so it is
the clean, hardware-independent test of the claim: if the bias does not raise
E_accept it cannot raise TPS. Baseline (PR #44, fused path, precache-off public):
E_accept 4.071.

Each arm = one fresh server with exactly one changed variable (DRAFTER_FREQ_BIAS),
benchmarked on the public prompts via the official sglang protocol (reused from
private_gap_probe). LOCAL ONLY: no HF Job, no submission.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.validity.private_gap_probe import (  # noqa: E402
    BENCHMARK_DEPENDENCIES,
    DEFAULT_TOKENIZER,
    _bench_env,
    acceptance_delta,
    run_bench,
    scrape_counters,
)

DEFAULT_TOKENS = REPO / "research/drafter_freq_bias/freq_top_tokens.json"
_CAP_OK = re.compile(r"\[onegraph\] captured")
_CAP_FAIL = re.compile(r"\[onegraph\] capture failed")
_BIAS_BUILT = re.compile(r"\[drafter-freq-bias\] built bias table")


def _scan_log(path: Path) -> dict[str, bool]:
    try:
        txt = path.read_text()
    except Exception:  # noqa: BLE001
        return {}
    return {
        "onegraph_captured": bool(_CAP_OK.search(txt)),
        "onegraph_capture_failed": bool(_CAP_FAIL.search(txt)),
        "bias_table_built": bool(_BIAS_BUILT.search(txt)),
    }


def run_arm(
    *, bias: float, submission: Path, server_python: Path, bench_python: Path,
    bench_env: dict[str, str], out_dir: Path, public: Path, tokens: Path,
    topk: int, num_prompts: int, tokenizer: str, require_capture: bool,
) -> dict[str, Any]:
    label = f"b{bias}".replace(".", "p")
    log_path = out_dir / f"server_{label}.log"
    extra_env = {
        "PRECACHE_BENCH": "0",               # drop precache var (match public_cold control)
        "DISABLE_LOG_STATS": "0",            # expose spec_decode_* counters
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # cuRAND-free sampler (does not touch logits)
        "LOOPGRAPH_REQUIRE_CAPTURE": "1" if require_capture else "0",
        "DRAFTER_FREQ_BIAS": str(bias),
        "DRAFTER_FREQ_BIAS_TOPK": str(topk),
        # b==0 -> empty path -> _freq_bias_active() False -> stock fused drafter path
        "DRAFTER_FREQ_BIAS_TOKENS": "" if bias == 0.0 else str(tokens.resolve()),
    }
    res: dict[str, Any] = {"label": label, "bias": bias, "topk": topk, "num_prompts": num_prompts}
    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        res["serve_ready_s"] = round(time.time() - t0, 1)
        before = scrape_counters(srv.base_url)
        bench_out = out_dir / f"bench_{label}.jsonl"
        res["bench"] = run_bench(
            bench_python, bench_env, base_url=srv.base_url, model=srv.served_model_name,
            dataset=public, num_prompts=num_prompts, output_file=bench_out, tokenizer=tokenizer,
        )
        after = scrape_counters(srv.base_url)
        res["acceptance"] = acceptance_delta(before, after)
    res["log"] = _scan_log(log_path)
    b, a = res["bench"], res["acceptance"]
    print(
        f"[arm {label}] tps={b['tps']:.2f} completed={b['completed']} "
        f"E_accept={a.get('e_accept')} accept_rate={a.get('accept_rate')} "
        f"capture={res['log'].get('onegraph_captured')} bias_built={res['log'].get('bias_table_built')}",
        flush=True,
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=str(REPO / "submissions/fa2sw_precache_kenyan"))
    ap.add_argument("--public", default=str(paths.EVAL_PROMPTS))
    ap.add_argument("--tokens", default=str(DEFAULT_TOKENS))
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--arms", default="0,0.5,1.0,2.0", help="comma-separated DRAFTER_FREQ_BIAS values")
    ap.add_argument("--require-capture", action="store_true",
                    help="fail the arm if onegraph capture fails (default: allow eager fallback; "
                         "E_accept is capture-invariant so the screen stays valid)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    args = ap.parse_args()

    for n in paths.prepare_local_gpu_env():
        print(f"[gpu] {n}", flush=True)

    submission = Path(args.submission)
    public = Path(args.public)
    tokens = Path(args.tokens)
    tokenizer = args.tokenizer if Path(args.tokenizer).exists() else paths.TOKENIZER
    arms = [float(x) for x in args.arms.split(",") if x.strip() != ""]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "research/drafter_freq_bias" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[screen] out_dir={out_dir} arms={arms} topk={args.topk} n={args.num_prompts}", flush=True)

    manifest = harness.load_manifest(submission)
    print("[screen] building/locating server venv (custom vLLM wheel)", flush=True)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print("[screen] building/locating bench venv (sglang)", flush=True)
    bench_python = harness.ensure_server_venv(BENCHMARK_DEPENDENCIES)
    bench_env = _bench_env(bench_python)

    arm_results: dict[str, Any] = {}
    for bias in arms:
        r = run_arm(
            bias=bias, submission=submission, server_python=server_python,
            bench_python=bench_python, bench_env=bench_env, out_dir=out_dir, public=public,
            tokens=tokens, topk=args.topk, num_prompts=args.num_prompts, tokenizer=tokenizer,
            require_capture=args.require_capture,
        )
        arm_results[r["label"]] = r

    base = next((r for r in arm_results.values() if r["bias"] == 0.0), None)
    base_eacc = base["acceptance"].get("e_accept") if base else None
    base_tps = base["bench"]["tps"] if base else None
    rows = []
    for r in sorted(arm_results.values(), key=lambda x: x["bias"]):
        eacc = r["acceptance"].get("e_accept")
        tps = r["bench"]["tps"]
        rows.append({
            "bias": r["bias"], "e_accept": eacc, "accept_rate": r["acceptance"].get("accept_rate"),
            "tps": tps, "completed": r["bench"]["completed"],
            "d_e_accept_vs_base": (eacc - base_eacc) if (eacc and base_eacc) else None,
            "d_tps_pct_vs_base": ((tps - base_tps) / base_tps * 100.0) if (tps and base_tps) else None,
            "onegraph_captured": r["log"].get("onegraph_captured"),
            "bias_table_built": r["log"].get("bias_table_built"),
        })
    best = max((row for row in rows if row["e_accept"]), key=lambda x: x["e_accept"], default=None)
    verdict = "NO-GAIN (bias does not raise E_accept)"
    if best and base_eacc and best["bias"] != 0.0 and best["e_accept"] > base_eacc + 1e-6:
        verdict = f"GAIN at bias={best['bias']} (E_accept {base_eacc:.4f} -> {best['e_accept']:.4f})"

    report = {
        "timestamp": ts, "submission": str(submission), "tokens": str(tokens),
        "topk": args.topk, "num_prompts": args.num_prompts, "arms": arms,
        "baseline_e_accept": base_eacc, "baseline_tps": base_tps,
        "rows": rows, "verdict": verdict, "results": arm_results,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72, flush=True)
    print(f"DRAFTER FREQ-BIAS SCREEN  (topk={args.topk}, n={args.num_prompts})", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'bias':>5}  {'E_accept':>9}  {'dE':>8}  {'accept%':>8}  {'tps':>8}  {'dTPS%':>7}  cap", flush=True)
    for row in rows:
        de = row["d_e_accept_vs_base"]
        dt = row["d_tps_pct_vs_base"]
        ar = row["accept_rate"]
        print(
            f"  {row['bias']:>5}  {row['e_accept'] or float('nan'):>9.4f}  "
            f"{(de if de is not None else float('nan')):>+8.4f}  "
            f"{(ar*100 if ar else float('nan')):>7.2f}%  {row['tps']:>8.2f}  "
            f"{(dt if dt is not None else float('nan')):>+7.2f}  {str(row['onegraph_captured'])[0]}",
            flush=True,
        )
    print(f"  VERDICT: {verdict}", flush=True)
    print("=" * 72 + "\n", flush=True)
    print(f"[screen] report -> {report_path}", flush=True)

    if args.wandb_group:
        _log_wandb(report, wandb_group=args.wandb_group,
                   wandb_name=args.wandb_name or "kanna/drafter-freq-bias-screen",
                   report_path=report_path)
    return 0


def _log_wandb(report: dict[str, Any], *, wandb_group: str, wandb_name: str, report_path: Path) -> None:
    try:
        from scripts.wandb_logging import (
            finish_wandb, init_wandb_run, log_file_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="drafter-freq-bias-screen", agent="senpai", name=wandb_name,
        tags=["drafter-freq-bias", wandb_group],
        config={"submission": report["submission"], "topk": report["topk"],
                "num_prompts": report["num_prompts"], "arms": report["arms"],
                "tokens": report["tokens"], "wandb_group": wandb_group},
    )
    if run is None:
        print("[wandb] run not created; report.json is the record", flush=True)
        return
    summary: dict[str, Any] = {
        "baseline_e_accept": report["baseline_e_accept"],
        "baseline_tps": report["baseline_tps"],
        "verdict": report["verdict"],
        "gain": report["verdict"].startswith("GAIN"),
    }
    for row in report["rows"]:
        tag = f"b{row['bias']}".replace(".", "p")
        summary[f"e_accept/{tag}"] = row["e_accept"]
        summary[f"accept_rate/{tag}"] = row["accept_rate"]
        summary[f"tps/{tag}"] = row["tps"]
        summary[f"d_e_accept/{tag}"] = row["d_e_accept_vs_base"]
    log_summary(run, summary, step=0)
    log_file_artifact(run, path=report_path, name="drafter_freq_bias_report",
                      artifact_type="drafter-freq-bias-screen-report")
    finish_wandb(run)
    print(f"[wandb] logged {wandb_name} (group={wandb_group})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
