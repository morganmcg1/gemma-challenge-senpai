#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Measure the REAL stock-drafter delta_stock — the #730 G1 fire-gate resolver (PR #739).

delta_stock = the public->private TPS/acceptance drift of the EXACT fire candidate
`submissions/int4_mtp_batchinv` (un-rescued K=6 MTP target + stock-Hub drafter
google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant), on vLLM 0.22.0 with
VLLM_BATCH_INVARIANT=1, conc=1, enforce-eager.

We have no official private set, so delta_stock is estimated via a
prompt-distribution-shift proxy. For each disjoint prompt subset we measure the
per-block acceptance e_accept (the #737 block-efficiency INCLUDING the always-
accepted bonus token):

    e_accept = 1 + (delta num_accepted_tokens) / (delta num_drafts)

from vLLM's cumulative spec_decode_* Prometheus counters, scraped before/after
each subset on one long-lived server (counter deltas isolate each subset; the
drafter's per-step acceptance does not leak across disjoint prompt sets).

#737 finding: single-stream TPS is linear in e_accept (h == delta), so

    TPS_subset = TPS_public * (e_accept_subset / e_accept_public)
    delta_stock(subset) = 1 - TPS_subset / TPS_public
                        = 1 - e_accept_subset / e_accept_public

The PUBLIC anchor is the official 128 benchmark prompts (57 MMLU-Pro + 57
GPQA-Diamond + 14 AIME). The shifted subsets span the plausible private mixes:
same-family held-out knowledge/reasoning (the likely private case) and pure-chat /
code / multilingual (the documented "wide/chat-heavy" worst corner, kanna #44).

LOCAL ONLY: launches no HF Job, makes no submission. Local A10G TPS is exploratory
(non-deterministic per PR #38); the e_accept RATIO is the gated signal.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SUBMISSION = REPO / "submissions" / "int4_mtp_batchinv"

# name -> (dataset path, kind). "public" is the anchor; the rest are shifts.
SUBSETS: dict[str, tuple[Path, str]] = {
    "public": (paths.EVAL_PROMPTS, "anchor:mmlu_pro+gpqa+aime (official 128)"),
    "knowledge_mmlupro": (REPO / "data/deltastock_knowledge_mmlupro.json", "same-family held-out MMLU-Pro MCQ"),
    "gpqa_diamond": (REPO / "data/deltastock_gpqa_diamond.json", "same-family held-out GPQA-D MCQ (harder MCQ half, #749)"),
    "reasoning_math": (REPO / "data/deltastock_reasoning_math.json", "AIME2024+gsm8k free-response"),
    "chat_casual": (REPO / "data/private_proxy_native_casual.json", "ShareGPT pure-chat (worst-corner dir)"),
    "code": (REPO / "data/private_proxy_native_code.json", "ShareGPT code-heavy"),
    "multilingual": (REPO / "data/private_proxy_native_multilingual.json", "non-latin script"),
}


def load_prompts(path: Path, n: int) -> list[str]:
    data = json.loads(Path(path).read_text())
    out = []
    for x in data[:n]:
        out.append(x["conversations"][0]["value"])
    return out


def scrape(base_url: str) -> dict[str, Any]:
    return serve_profile.parse_spec_metrics(serve_profile._get_text(f"{base_url}/metrics"))


def completion(base_url: str, model: str, prompt: str, max_tokens: int,
               timeout_s: int = 600) -> dict[str, Any]:
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": False, "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def out_tokens(resp: dict[str, Any]) -> int:
    u = resp.get("usage") or {}
    if isinstance(u.get("completion_tokens"), int):
        return u["completion_tokens"]
    ch = (resp.get("choices") or [{}])[0]
    t = ch.get("token_ids")
    return len(t) if isinstance(t, list) else 0


def run_subset(base_url: str, model: str, prompts: list[str], max_tokens: int,
               client_conc: int) -> dict[str, Any]:
    before = scrape(base_url)
    t0 = time.time()
    total_out = 0
    errors = 0

    def one(p: str) -> int:
        try:
            return out_tokens(completion(base_url, model, p, max_tokens))
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] completion failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
            return -1

    with ThreadPoolExecutor(max_workers=client_conc) as ex:
        for n in ex.map(one, prompts):
            if n < 0:
                errors += 1
            else:
                total_out += n
    wall = time.time() - t0
    after = scrape(base_url)

    d_drafts = (after.get("num_drafts") or 0) - (before.get("num_drafts") or 0)
    d_acc = (after.get("num_accepted_tokens") or 0) - (before.get("num_accepted_tokens") or 0)
    d_dtok = (after.get("num_draft_tokens") or 0) - (before.get("num_draft_tokens") or 0)
    e_accept = (1.0 + d_acc / d_drafts) if d_drafts else None
    accept_rate = (d_acc / d_dtok) if d_dtok else None
    tps = total_out / wall if wall > 0 else None
    return {
        "n_prompts": len(prompts), "errors": errors, "max_tokens": max_tokens,
        "wall_s": round(wall, 1), "total_output_tokens": total_out,
        "tps_local": tps, "e_accept": e_accept, "accept_rate": accept_rate,
        "delta_num_drafts": d_drafts, "delta_num_accepted_tokens": d_acc,
        "delta_num_draft_tokens": d_dtok,
        "counters_before": before, "counters_after": after,
    }


def p_dq_from_subsets(deltas: dict[str, float], *, frac_per_subset: float = 1.0) -> dict[str, Any]:
    """Convert the measured per-subset delta_stock distribution into P(delta_stock>5%).

    Two readings, both reported:
      * empirical_frac : fraction of shifted subsets whose delta exceeds 5% (the raw
        'fraction > 5%' the PR asks for).
      * gaussian_tail  : model the realistic private delta as N(mu, sigma) over the
        shifted subsets and return P(>5%); a smooth estimator that does not collapse
        to 0/1 when all subsets sit on one side of the line.
    """
    vals = [v for k, v in deltas.items() if k != "public"]
    n = len(vals)
    emp = sum(1 for v in vals if v > 5.0) / n if n else float("nan")
    mu = statistics.mean(vals) if vals else float("nan")
    sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    if sigma > 0:
        from math import erf, sqrt
        z = (5.0 - mu) / sigma
        gauss = 0.5 * (1 - erf(z / sqrt(2)))
    else:
        gauss = 1.0 if mu > 5.0 else 0.0
    return {
        "empirical_frac_gt5": emp, "mean_delta": mu, "std_delta": sigma,
        "gaussian_tail_p_gt5": gauss, "n_shifted_subsets": n,
        "worst_corner_delta": max(vals) if vals else float("nan"),
        "min_delta": min(vals) if vals else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-python", default=str(REPO / ".venvs/vllm022/bin/python"),
                    help="python for serve.py; default reuses validated .venvs/vllm022")
    ap.add_argument("--max-tokens", type=int, default=512, help="decode length per prompt")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--client-conc", type=int, default=8,
                    help="client-side pipelining; server still serializes at MAX_NUM_SEQS=1")
    ap.add_argument("--only", default=None, help="comma-sep subset names (default all)")
    ap.add_argument("--smoke", action="store_true", help="public+chat only, 4 prompts, 32 tokens")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    args = ap.parse_args()

    notes = paths.prepare_local_gpu_env()
    for nnote in notes:
        print(f"[gpu] {nnote}", flush=True)

    max_tokens = 32 if args.smoke else args.max_tokens
    num_prompts = 4 if args.smoke else args.num_prompts
    names = (["public", "chat_casual"] if args.smoke
             else (args.only.split(",") if args.only else list(SUBSETS)))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "research/validity/deltastock_measure" / (("smoke-" if args.smoke else "") + ts))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] out_dir={out_dir}  subsets={names}  n={num_prompts} max_tok={max_tokens}", flush=True)

    server_python = Path(args.server_python)
    log_path = out_dir / "server.log"
    # enforce-eager (PR), expose spec_decode_* counters; manifest already pins
    # VLLM_BATCH_INVARIANT=1, drafter, K=6, conc=1.
    extra_env = {"ENFORCE_EAGER": "1", "DISABLE_LOG_STATS": "0",
                 "VLLM_USE_FLASHINFER_SAMPLER": "0"}

    results: dict[str, Any] = {}
    with harness.LocalServer(SUBMISSION, server_python=server_python, port=8000,
                             log_path=log_path, extra_env=extra_env,
                             startup_timeout_s=1800) as srv:
        model = srv.served_model_name
        print(f"[probe] served model={model} at {srv.base_url}", flush=True)
        for name in names:
            path, kind = SUBSETS[name]
            prompts = load_prompts(path, num_prompts)
            print(f"\n[subset {name}] {kind}  ({len(prompts)} prompts)", flush=True)
            r = run_subset(srv.base_url, model, prompts, max_tokens, args.client_conc)
            r["kind"] = kind
            r["dataset"] = str(path)
            results[name] = r
            print(f"[subset {name}] e_accept={r['e_accept']!r} accept_rate={r['accept_rate']!r} "
                  f"tps_local={r['tps_local']!r} wall={r['wall_s']}s err={r['errors']}", flush=True)

    # whole-run spec-log cross-check
    try:
        spec_xcheck = serve_profile.parse_spec_log(log_path.read_text())
    except Exception:  # noqa: BLE001
        spec_xcheck = None

    e_pub = results.get("public", {}).get("e_accept")
    tps_pub = results.get("public", {}).get("tps_local")
    deltas_eacc: dict[str, float] = {}
    deltas_tps: dict[str, float] = {}
    for name, r in results.items():
        if e_pub and r.get("e_accept"):
            deltas_eacc[name] = round((1 - r["e_accept"] / e_pub) * 100, 3)
        if tps_pub and r.get("tps_local"):
            deltas_tps[name] = round((1 - r["tps_local"] / tps_pub) * 100, 3)

    agg = p_dq_from_subsets(deltas_eacc) if len(deltas_eacc) > 1 else {}
    report = {
        "timestamp": ts, "submission": str(SUBMISSION), "smoke": args.smoke,
        "num_prompts": num_prompts, "max_tokens": max_tokens,
        "server_python": str(server_python),
        "config": {"model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
                   "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
                   "num_speculative_tokens": 6, "VLLM_BATCH_INVARIANT": 1,
                   "enforce_eager": True, "max_num_seqs": 1},
        "e_accept_public": e_pub, "tps_public_local": tps_pub,
        "delta_stock_pct_by_eaccept": deltas_eacc,
        "delta_stock_pct_by_tps": deltas_tps,
        "aggregate": agg,
        "spec_log_xcheck": spec_xcheck,
        "subsets": results,
        "anchors": {"flagship_52_valid_pct": 4.3, "kanna_44_chat_upper_pct": 12.4,
                    "honest_band_pct": [4, 9], "g1_threshold_pct": 5.0},
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"[probe] report -> {report_path}", flush=True)

    if args.wandb_group:
        try:
            log_to_wandb(report, wandb_name=args.wandb_name or "kanna/deltastock-measure",
                         wandb_group=args.wandb_group, report_path=report_path)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] W&B logging failed (non-fatal): {e!r}", flush=True)
    return 0


def _print_summary(r: dict[str, Any]) -> None:
    print("\n" + "=" * 70, flush=True)
    print(f"DELTA_STOCK MEASURE  ({'SMOKE ' if r['smoke'] else ''}n={r['num_prompts']} "
          f"max_tok={r['max_tokens']})", flush=True)
    print("=" * 70, flush=True)
    print(f"  e_accept(public) = {r['e_accept_public']}   tps_local(public) = {r['tps_public_local']}", flush=True)
    print(f"  {'subset':22s} {'e_accept':>9s} {'accept%':>8s} {'d_stock(eacc)%':>14s} {'d_stock(tps)%':>13s}", flush=True)
    for name, s in r["subsets"].items():
        ea = s.get("e_accept"); ar = s.get("accept_rate")
        de = r["delta_stock_pct_by_eaccept"].get(name)
        dt = r["delta_stock_pct_by_tps"].get(name)
        print(f"  {name:22s} {ea if ea is None else round(ea,4):>9} "
              f"{ar if ar is None else round(ar*100,2):>8} "
              f"{de if de is None else round(de,2):>14} {dt if dt is None else round(dt,2):>13}", flush=True)
    a = r.get("aggregate") or {}
    if a:
        print(f"\n  shifted-subset delta_stock: mean={a['mean_delta']:.2f}% "
              f"worst={a['worst_corner_delta']:.2f}% min={a['min_delta']:.2f}%", flush=True)
        print(f"  fraction>5% (empirical) = {a['empirical_frac_gt5']:.2f}   "
              f"P(delta>5%) gaussian-tail = {a['gaussian_tail_p_gt5']:.3f}", flush=True)
    print("=" * 70 + "\n", flush=True)


def log_to_wandb(report: dict[str, Any], *, wandb_name: str, wandb_group: str,
                 report_path: Path) -> None:
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return
    agg = report.get("aggregate") or {}
    central = agg.get("mean_delta")
    summary = {
        "p_dq_g1_at_measured_delta_stock": agg.get("gaussian_tail_p_gt5"),
        "delta_stock_central_pct": central,
        "delta_stock_worst_corner_pct": agg.get("worst_corner_delta"),
        "delta_stock_min_pct": agg.get("min_delta"),
        "empirical_frac_gt5": agg.get("empirical_frac_gt5"),
        "e_accept_public": report.get("e_accept_public"),
        "tps_public_local": report.get("tps_public_local"),
        "g1_clears": bool(central is not None and central < 5.0),
        **{f"delta_stock__{k}": v for k, v in report.get("delta_stock_pct_by_eaccept", {}).items()},
        **{f"e_accept__{k}": (s.get("e_accept")) for k, s in report.get("subsets", {}).items()},
    }
    run = init_wandb_run(
        job_type="deltastock-measure", agent="senpai", name=wandb_name,
        group=wandb_group, tags=["deltastock-measure", wandb_group],
        config={"submission": report["submission"], "num_prompts": report["num_prompts"],
                "max_tokens": report["max_tokens"], "wandb_group": wandb_group,
                "smoke": report["smoke"], **report["config"]})
    if run is None:
        print("[wandb] run not created; report.json is the record", flush=True)
        return
    log_summary(run, summary, step=0)
    log_file_artifact(run, path=report_path, name="deltastock_report",
                      artifact_type="deltastock-measure-report")
    finish_wandb(run)
    print(f"[wandb] logged {wandb_name} (group={wandb_group})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
