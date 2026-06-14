#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Measure the REAL per-rank acceptance on the deployed MTP K=7 chain (PR #76).

WHAT THIS ANSWERS
-----------------
PR #74 projected a +18-20% tree-verify TPS gain whose single largest uncertainty is
the deployed chain's REAL served per-position acceptance. Two numbers disagree and
bracket the gain:

  * my #49 acceptance model used top-1 = 0.6792 -- but that scalar was measured on an
    EAGLE-3 drafter (PR #16/#26), NOT the deployed MTP drafter.
  * denken #68 noted the deployed chain emits ~3.8 tok/step, which under a GEOMETRIC
    (constant per-position p) linear chain implies top-1 ~ 0.775.

This script measures the authoritative per-position acceptance + mean accepted
tokens/step ON THE CURRENT DEPLOYED STACK (`submissions/fa2sw_precache_kenyan`:
linear MTP K=7, M=8 verify, #43 split-KV) by reading vLLM's OWN spec-decode counters.

HOW (contract-safe, measurement-only)
--------------------------------------
The served accept rule is byte-identical: we launch the UNMODIFIED submission via the
local harness (`scripts/local_validation/harness.LocalServer`) and only override one
env var -- DISABLE_LOG_STATS=0 -- so vLLM re-registers its stat loggers. The
leaderboard manifest ships --disable-log-stats, which de-registers them; re-enabling
stats is a handful of host-side counter increments per step (<0.1 ms, no GPU compute)
and does NOT change which tokens are accepted or emitted (greedy identity untouched).

vLLM V1's scheduler computes, per decode step, num_accepted = len(generated)-1 from
the ACTUAL emitted tokens (vllm/v1/core/sched/scheduler.py), independent of the
submission's monkeypatched fused-accept kernel, and feeds SpecDecodingStats. So:

  * `vllm:spec_decode_num_accepted_tokens_per_pos{position=k}` / num_drafts = the
    CUMULATIVE acceptance C[k+1] = P(accept >= k+1 drafts) -- exact, whole-run.
  * mean acceptance length E[T] = 1 + num_accepted_tokens / num_drafts.

Two independent reads, cross-validated:
  (A) server-log "SpecDecoding metrics: ... Per-position acceptance rate: ..." lines
      (always emitted while spec-decode runs), aggregated draft-weighted over the run.
  (B) Prometheus /metrics counters (exact cumulative; may be empty on some wheels --
      treated as a cross-check, never the sole source).

OUTPUT
------
Per-position cumulative C[1..7] and conditional p[k]=C[k]/C[k-1], mean tokens/step,
draft acceptance rate. Primary metric: deployed_chain_mean_tokens_per_step.

LOCAL ONLY. Single assigned GPU. No HF Job, no submission launch, no served-file
change. Reads the served stack read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# repo root on sys.path so we can import the proven local-validation harness verbatim
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
DEFAULT_OUTPUT = ROOT / "research" / "accept_calibration" / "accept_calibration_results.json"


# --------------------------------------------------------------------------- #
# (A) server-log parse -> whole-run draft-weighted per-position acceptance
# --------------------------------------------------------------------------- #
# One vLLM LoggingStatLogger interval line. The per-position vector is vLLM's
# CUMULATIVE acceptance rate C[k] = P(accept >= k drafts) (metrics.py increments
# num_accepted_tokens_per_pos[0..num_accepted-1] each step, then divides the
# per-interval sum by that interval's num_drafts). Counts are per-interval (the
# logger reset()s each window), so we re-weight each interval by its own num_drafts.
_LINE_RE = re.compile(
    r"Mean acceptance length:\s*([\d.]+).*?"
    r"Accepted:\s*(\d+)\s*tokens,\s*Drafted:\s*(\d+)\s*tokens,\s*"
    r"Per-position acceptance rate:\s*([0-9.,\s]+?),\s*Avg Draft acceptance rate:\s*([\d.]+)%"
)
_NUM_SPEC_RE = re.compile(r"num_speculative_tokens[\"']?\s*:\s*(\d+)")


def parse_log_per_position(log_text: str) -> dict[str, Any]:
    """Whole-run cumulative C[k] + conditional p[k] + E[T] from vLLM's own log."""
    k_match = _NUM_SPEC_RE.search(log_text)
    num_spec = int(k_match.group(1)) if k_match else None
    intervals = []
    for m in _LINE_RE.finditer(log_text):
        cum = [float(x) for x in m.group(4).split(",")]
        intervals.append({
            "mean_acceptance_length": float(m.group(1)),
            "accepted": int(m.group(2)),
            "drafted": int(m.group(3)),
            "cumulative_rate": cum,
            "avg_draft_acceptance_pct": float(m.group(5)),
        })
    if not intervals:
        return {"intervals": 0, "error": "no SpecDecoding metric lines in server log"}
    K = num_spec or len(intervals[0]["cumulative_rate"])
    total_acc = sum(it["accepted"] for it in intervals)
    total_drf = sum(it["drafted"] for it in intervals)
    # num_drafts (steps) per interval = drafted / K; weight each interval's cumulative
    # rate vector by its num_drafts to recover the whole-run cumulative acceptance.
    num = [0.0] * K
    den = 0.0
    for it in intervals:
        nd = it["drafted"] / K
        den += nd
        for k in range(K):
            num[k] += it["cumulative_rate"][k] * nd
    C = [num[k] / den for k in range(K)]
    cond = [C[0]] + [C[k] / C[k - 1] if C[k - 1] > 0 else float("nan") for k in range(1, K)]
    num_drafts = total_drf / K
    return {
        "intervals": len(intervals),
        "num_speculative_tokens": K,
        "total_accepted_tokens": total_acc,
        "total_drafted_tokens": total_drf,
        "num_drafts": num_drafts,
        "mean_tokens_per_step_E_T": 1.0 + total_acc / num_drafts if num_drafts else None,
        "draft_acceptance_rate": total_acc / total_drf if total_drf else None,
        "cumulative_acceptance_C": C,
        "conditional_acceptance_p": cond,
        "per_interval": intervals,
    }


# --------------------------------------------------------------------------- #
# (B) Prometheus /metrics -> exact whole-run cumulative counters (cross-check)
# --------------------------------------------------------------------------- #
def _prom_sum(text: str, metric: str) -> float | None:
    """Sum a counter across all (non-position) label sets; None if absent."""
    pat = re.compile(
        rf"^{re.escape(metric)}(?:_total)?(?:\{{(?![^}}]*position=)[^}}]*\}})?\s+([\d.eE+-]+)$",
        re.M,
    )
    total, found = 0.0, False
    for m in pat.finditer(text):
        try:
            total += float(m.group(1))
            found = True
        except ValueError:
            pass
    return total if found else None


def _prom_per_pos(text: str, metric: str, K: int) -> list[float] | None:
    """Extract a per-position counter vector keyed by the position="k" label."""
    pat = re.compile(
        rf"^{re.escape(metric)}(?:_total)?\{{([^}}]*)\}}\s+([\d.eE+-]+)$", re.M
    )
    acc: dict[int, float] = {}
    for m in pat.finditer(text):
        labels, val = m.group(1), m.group(2)
        pm = re.search(r'position="(\d+)"', labels)
        if not pm:
            continue
        try:
            acc[int(pm.group(1))] = acc.get(int(pm.group(1)), 0.0) + float(val)
        except ValueError:
            pass
    if not acc:
        return None
    return [acc.get(i, 0.0) for i in range(K)]


def parse_prometheus(text: str, K: int) -> dict[str, Any]:
    drafts = _prom_sum(text, "vllm:spec_decode_num_drafts")
    accepted = _prom_sum(text, "vllm:spec_decode_num_accepted_tokens")
    draft_tokens = _prom_sum(text, "vllm:spec_decode_num_draft_tokens")
    per_pos = _prom_per_pos(text, "vllm:spec_decode_num_accepted_tokens_per_pos", K)
    out: dict[str, Any] = {
        "populated": bool(drafts),
        "num_drafts": drafts,
        "num_accepted_tokens": accepted,
        "num_draft_tokens": draft_tokens,
    }
    if drafts and accepted is not None:
        out["mean_tokens_per_step_E_T"] = 1.0 + accepted / drafts
    if drafts and per_pos is not None:
        C = [x / drafts for x in per_pos]
        out["accepted_per_pos"] = per_pos
        out["cumulative_acceptance_C"] = C
        out["conditional_acceptance_p"] = (
            [C[0]] + [C[k] / C[k - 1] if C[k - 1] > 0 else float("nan") for k in range(1, K)]
        )
    return out


def _get_text(url: str, timeout_s: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return r.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(submission: Path, *, num_prompts: int, output_len: int, out_path: Path,
        seed: int, dataset: Path | None = None, tag: str = "") -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    log_path = out_path.parent / f"server_accept_calibration{suffix}.log"

    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    # Re-enable vLLM stat loggers (the manifest ships --disable-log-stats); native
    # PyTorch sampler to dodge this container's FlashInfer cuRAND JIT. Neither
    # changes accepted tokens / greedy identity.
    extra_env = {"DISABLE_LOG_STATS": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}

    report: dict[str, Any] = {
        "submission": str(submission),
        "server_python": str(server_python),
        "num_prompts": num_prompts,
        "output_len": output_len,
        "seed": seed,
        "conc": 1,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset) if dataset else str(paths.EVAL_PROMPTS),
        "tag": tag,
    }

    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        decode_out = out_path.parent / f"decode_accept_calibration{suffix}.jsonl"
        decode_summary = out_path.parent / f"decode_accept_calibration{suffix}.summary.json"
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_summary,
            num_prompts=num_prompts, output_len=output_len, seed=seed, timeout_s=3600,
            dataset=dataset,
        )
        report["decode_summary"] = summary
        try:
            report["prometheus_raw_available"] = True
            metrics_text = _get_text(f"{srv.base_url}/metrics")
        except (urllib.error.URLError, OSError) as exc:
            report["prometheus_raw_available"] = False
            report["prometheus_error"] = str(exc)
            metrics_text = ""
    report["decode_wall_s"] = time.time() - t0

    log_text = log_path.read_text()
    log_res = parse_log_per_position(log_text)
    report["server_log_metrics"] = log_res
    K = log_res.get("num_speculative_tokens") or 7
    report["prometheus_metrics"] = parse_prometheus(metrics_text, K) if metrics_text else {
        "populated": False, "note": "metrics endpoint unreachable"
    }
    report["server_log"] = str(log_path)

    # Authoritative numbers come from the server-log aggregate (proven to populate
    # on this stack); Prometheus per-pos is the exact cross-check when present.
    E_T = log_res.get("mean_tokens_per_step_E_T")
    report["primary_metric"] = {
        "name": "deployed_chain_mean_tokens_per_step", "value": E_T,
    }
    report["headline"] = {
        "deployed_chain_mean_tokens_per_step": E_T,
        "top1_first_position_acceptance": (log_res.get("cumulative_acceptance_C") or [None])[0],
        "cumulative_acceptance_C": log_res.get("cumulative_acceptance_C"),
        "conditional_acceptance_p": log_res.get("conditional_acceptance_p"),
        "draft_acceptance_rate": log_res.get("draft_acceptance_rate"),
        "num_drafts": log_res.get("num_drafts"),
    }
    out_path.write_text(json.dumps(report, indent=2))
    return report


def log_wandb(report: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[accept-calib] wandb unavailable ({exc})", flush=True)
        return None
    try:
        log_res = report["server_log_metrics"]
        prom = report.get("prometheus_metrics", {})
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="profiling",
            config={
                "submission": report["submission"], "num_prompts": report["num_prompts"],
                "output_len": report["output_len"], "conc": 1, "seed": report["seed"],
                "num_speculative_tokens": log_res.get("num_speculative_tokens"),
            },
        )
        C = log_res.get("cumulative_acceptance_C") or []
        p = log_res.get("conditional_acceptance_p") or []
        flat: dict[str, Any] = {
            "primary/deployed_chain_mean_tokens_per_step": report["primary_metric"]["value"],
            "deployed_chain_E_T": log_res.get("mean_tokens_per_step_E_T"),
            "top1_first_position_acceptance": C[0] if C else None,
            "draft_acceptance_rate": log_res.get("draft_acceptance_rate"),
            "num_drafts": log_res.get("num_drafts"),
            "total_accepted_tokens": log_res.get("total_accepted_tokens"),
            "total_drafted_tokens": log_res.get("total_drafted_tokens"),
            "prometheus_populated": prom.get("populated"),
            "prometheus_E_T": prom.get("mean_tokens_per_step_E_T"),
        }
        for k, c in enumerate(C, start=1):
            flat[f"cumulative_C/pos{k}"] = c
        for k, pc in enumerate(p, start=1):
            flat[f"conditional_p/pos{k}"] = pc
        run.summary.update(flat)
        tbl = wandb.Table(columns=["position", "cumulative_C", "conditional_p"])
        for k in range(len(C)):
            tbl.add_data(k + 1, C[k], p[k] if k < len(p) else None)
        run.log({"per_position_acceptance": tbl})
        rid = run.id
        print(f"[accept-calib] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[accept-calib] wandb log failed ({exc})", flush=True)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="prompts dataset (sharegpt json); default = public eval prompts. "
                         "Point at data/private_proxy_sharegpt.json to measure the "
                         "private-proxy per-position ladder (PR #151).")
    ap.add_argument("--tag", default="",
                    help="suffix for server-log / decode artifact filenames so a "
                         "public and a private run in one session do not clobber.")
    ap.add_argument("--wandb-name", default="wirbel/deployed-chain-acceptance")
    ap.add_argument("--wandb-group", default="acceptance-calibration")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[accept-calib] {note}", flush=True)

    report = run(args.submission.resolve(), num_prompts=args.num_prompts,
                 output_len=args.output_len, out_path=args.output.resolve(), seed=args.seed,
                 dataset=args.dataset.resolve() if args.dataset else None, tag=args.tag)

    wid = None
    if not args.no_wandb:
        wid = log_wandb(report, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    args.output.resolve().write_text(json.dumps(report, indent=2))

    h = report["headline"]
    print("\n========== DEPLOYED-CHAIN ACCEPTANCE CALIBRATION ==========", flush=True)
    print(f"submission           : {report['submission']}", flush=True)
    print(f"workload             : conc=1, {args.num_prompts} prompts, output_len {args.output_len}", flush=True)
    print(f"num decode drafts    : {h['num_drafts']:.0f}", flush=True)
    print(f"E[T] mean tok/step   : {h['deployed_chain_mean_tokens_per_step']:.4f}", flush=True)
    print(f"top-1 (pos-1) accept : {h['top1_first_position_acceptance']:.4f}", flush=True)
    print(f"draft acceptance rate: {h['draft_acceptance_rate']:.4f}", flush=True)
    C = h["cumulative_acceptance_C"]; p = h["conditional_acceptance_p"]
    print("cumulative C[k]      : " + ", ".join(f"{x:.4f}" for x in C), flush=True)
    print("conditional p[k]     : " + ", ".join(f"{x:.4f}" for x in p), flush=True)
    prom = report.get("prometheus_metrics", {})
    if prom.get("populated") and prom.get("mean_tokens_per_step_E_T"):
        print(f"prometheus E[T] xchk : {prom['mean_tokens_per_step_E_T']:.4f} "
              f"(populated={prom.get('populated')})", flush=True)
    else:
        print(f"prometheus per-pos   : not populated (log is authoritative)", flush=True)
    print(f"wandb run            : {wid}", flush=True)
    print(f"artifacts            : {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
