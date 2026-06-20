#!/usr/bin/env python
"""Benchmark-faithful 128-prompt TPS for the bi0 fire candidates (PR #825).

Re-measures ONE fire candidate (int4head or PLE-dequant) on the EXACT official
decode workload so the local number predicts the official HF ``summary.json:tps``:

  * the official ``decode_outputs.py`` prompt set (128 ShareGPT prompts, seed=1),
  * conc=1 sequential requests (one /v1/completions at a time, MAX_NUM_SEQS=1),
  * temp=0, ignore_eos, output_len=512, return_token_ids -- identical payload,
  * the aggregate is token-weighted ``sum(completion_tokens) / wall_clock`` over
    the WHOLE 128-prompt loop, i.e. PREFILL-INCLUDED, exactly like the official
    summary's ``num_completion_tokens / duration_s``.

On top of the official shape it adds the #814 diagnostics the official script
cannot give: PER-PROMPT wall TPS + PER-PROMPT MTP acceptance (E_accept), read
from the vLLM ``/metrics`` spec-decode counters before/after each request, and an
acceptance-variance bootstrap CI on the 128-prompt aggregate.

Token-free serve path (``LMHEAD_QUANT_AT_STARTUP=1``, public base
``google/gemma-4-E4B-it-qat-w4a16-ct``, head group_size 32, ``--dequant-ple`` for
the pledequant arm) -- byte-identical to the banked #817 / #805 builds, no
private-repo auth. The gated profiling knobs (REJECTRANK_ENABLE, synthetic
acceptance, cudagraph-capture-sizes) are left UNSET, so the served path is the
shipped fire-candidate path.

LOCAL single-A10G only. NO HF Job, NO submission, NO leaderboard change, NO code
change to submissions/**.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

BASE_W4A16 = "google/gemma-4-E4B-it-qat-w4a16-ct"  # public base for token-free startup quant
SERVED_NAME = "gemma-4-e4b-it"
PPL_CAP = 2.42
TARGET_TPS = 250.0

# Candidate registry. extra_env OVERRIDES the manifest env to activate the
# token-free startup-quant serve path against the public base. No code change to
# the submission; only env (exactly what harness.LocalServer.extra_env is for).
CANDIDATES: dict[str, dict[str, Any]] = {
    "int4head": {
        "dir": ROOT / "submissions" / "int4_mtp_bi0_int4head",
        "extra_env": {
            "MODEL_ID": BASE_W4A16,
            "LMHEAD_QUANT_AT_STARTUP": "1",
            "LMHEAD_QUANT_GROUP_SIZE": "32",
        },
        "headline_tps": 256.74,
        "headline_kind": "prefill-EXCLUDED single-stream probe_tps (#814 caveat)",
        "headline_ppl": 2.0029,
    },
    "pledequant": {
        "dir": ROOT / "submissions" / "int4_mtp_bi0_int4head_pledequant",
        "extra_env": {
            "MODEL_ID": BASE_W4A16,
            "LMHEAD_QUANT_AT_STARTUP": "1",
            "LMHEAD_QUANT_GROUP_SIZE": "32",
            "LMHEAD_QUANT_DEQUANT_PLE": "1",
        },
        "headline_tps": 265.61,
        "headline_kind": "local headline (#805); basis unconfirmed",
        "headline_ppl": 2.0031,
    },
}


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def load_official_decode_module():
    """Import the official decode_outputs.py by path for byte-faithful workload."""
    spec = importlib.util.spec_from_file_location("official_decode", str(paths.DECODE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- per-prompt MTP acceptance via vLLM /metrics (proven in #814 accept_diag) ---
import re  # noqa: E402

_COUNTER_RE = {
    "accepted": re.compile(r"^vllm:spec_decode_num_accepted_tokens(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+\-.]+)"),
    "draft": re.compile(r"^vllm:spec_decode_num_draft_tokens(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+\-.]+)"),
    "drafts": re.compile(r"^vllm:spec_decode_num_drafts(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+\-.]+)"),
}


def read_counters(base_url: str) -> dict[str, float]:
    out = {k: 0.0 for k in _COUNTER_RE}
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return out
    for line in text.splitlines():
        s = line.strip()
        for key, rx in _COUNTER_RE.items():
            m = rx.match(s)
            if m:
                out[key] += float(m.group(1))
    return out


def _dist(values: list[float]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None and v == v]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }


def bootstrap_aggregate_ci(
    per_prompt_tokens: list[int], per_prompt_wall: list[float], *,
    iters: int = 10000, seed: int = 12345, alpha: float = 0.05,
) -> dict[str, float | None]:
    """Bootstrap CI for the token-weighted 128-prompt aggregate TPS.

    Resamples the (tokens, wall) prompt pairs WITH REPLACEMENT and recomputes
    ``sum(tokens)/sum(wall)`` each time. This is the acceptance-lottery CI: it
    estimates how the prefill-included aggregate would move if the 128 official
    prompts were a different random draw, i.e. the #814 per-prompt variance
    propagated into the official-style aggregate.
    """
    pairs = [(t, w) for t, w in zip(per_prompt_tokens, per_prompt_wall) if w and w > 0]
    if len(pairs) < 2:
        return {"lo": None, "hi": None, "median": None, "iters": 0}
    rng = random.Random(seed)
    n = len(pairs)
    aggs: list[float] = []
    for _ in range(iters):
        tok_sum = 0.0
        wall_sum = 0.0
        for _ in range(n):
            t, w = pairs[rng.randrange(n)]
            tok_sum += t
            wall_sum += w
        aggs.append(tok_sum / wall_sum)
    aggs.sort()
    lo_i = int((alpha / 2) * iters)
    hi_i = int((1 - alpha / 2) * iters) - 1
    return {
        "lo": aggs[lo_i],
        "hi": aggs[hi_i],
        "median": aggs[iters // 2],
        "iters": iters,
        "alpha": alpha,
    }


def run_instrumented_rep(
    decode_mod, base_url: str, prompts: list[dict[str, Any]], output_len: int,
    request_timeout_s: int,
) -> dict[str, Any]:
    """One full 128-prompt sequential pass with per-prompt timing + acceptance."""
    per_prompt: list[dict[str, Any]] = []
    loop_t0 = time.time()
    for rec in prompts:
        ptids = rec["prompt_token_ids"]
        c0 = read_counters(base_url)
        t0 = time.time()
        resp = decode_mod.request_decode(
            base_url=base_url, model=SERVED_NAME, prompt_token_ids=ptids,
            output_len=output_len, timeout_s=request_timeout_s,
        )
        t1 = time.time()
        c1 = read_counters(base_url)
        choice = decode_mod.choice_from_response(resp)
        comp_ids, _src, _kind = decode_mod.extract_generated_token_ids(resp, choice, ptids)
        wall = t1 - t0
        n_out = len(comp_ids)
        d_acc = c1["accepted"] - c0["accepted"]
        d_drf = c1["draft"] - c0["draft"]
        d_drafts = c1["drafts"] - c0["drafts"]
        accept_rate = (d_acc / d_drf) if d_drf > 0 else None
        # E_accept = mean tokens EMITTED per verify step = (accepted + 1 bonus/step)/steps
        e_accept = ((d_acc + d_drafts) / d_drafts) if d_drafts > 0 else None
        per_prompt.append({
            "id": rec["id"],
            "n_prompt_tokens": len(ptids),
            "n_out": n_out,
            "wall_s": wall,
            "tps": (n_out / wall) if wall > 0 else None,
            "accepted": d_acc, "draft": d_drf, "drafts": d_drafts,
            "accept_rate": accept_rate, "e_accept": e_accept,
        })
    loop_wall = time.time() - loop_t0
    tot_tokens = sum(p["n_out"] for p in per_prompt)
    tight_wall = sum(p["wall_s"] for p in per_prompt)
    return {
        "per_prompt": per_prompt,
        "tot_tokens": tot_tokens,
        "loop_wall_s": loop_wall,          # whole-loop, includes /metrics scrapes (faithful upper)
        "tight_wall_s": tight_wall,        # request-only, scrape-free (faithful lower)
        "wall_tps_loopclock": tot_tokens / loop_wall if loop_wall > 0 else None,
        "wall_tps_tight": tot_tokens / tight_wall if tight_wall > 0 else None,
    }


def measure(args) -> dict[str, Any]:
    cand = CANDIDATES[args.candidate]
    sub_dir = cand["dir"]
    out_dir = Path(args.work_dir or (ROOT / "research" / "bi0_firecand_tps_faithful" / "runs" / args.candidate))
    out_dir.mkdir(parents=True, exist_ok=True)

    decode_mod = load_official_decode_module()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.TOKENIZER)
    raw = decode_mod.read_sharegpt_prompts(paths.EVAL_PROMPTS, num_prompts=args.num_prompts, seed=args.seed)
    prompts = []
    for r in raw:
        prompts.append({
            "id": r["id"],
            "dataset_index": r["dataset_index"],
            "prompt_token_ids": decode_mod.encode_prompt(tok, r["prompt_text"]),
        })
    print(f"[measure] candidate={args.candidate} prompts={len(prompts)} "
          f"output_len={args.output_len} reps={args.reps}", flush=True)

    manifest = harness.load_manifest(sub_dir)
    server_py = harness.ensure_server_venv(manifest["dependencies"])

    rec: dict[str, Any] = {
        "candidate": args.candidate,
        "submission_dir": str(sub_dir),
        "serve_mode": "token-free startup-quant (LMHEAD_QUANT_AT_STARTUP=1)",
        "extra_env": cand["extra_env"],
        "headline_tps": cand["headline_tps"],
        "headline_kind": cand["headline_kind"],
        "headline_ppl": cand["headline_ppl"],
        "num_prompts": args.num_prompts, "output_len": args.output_len,
        "warmup_prompts": args.warmup_prompts, "reps": args.reps, "seed": args.seed,
        "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "served_ok": False, "error": None,
        "reps_detail": [], "official_crosscheck": None, "probe_tps": None,
        "ppl": None, "ppl_error": None, "gpu_mem_used_mib": None, "ready_s": None,
    }
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = out_dir / "server.log"
    try:
        t0 = time.time()
        with harness.LocalServer(
            sub_dir, server_python=server_py, port=args.port,
            log_path=log_path, extra_env=dict(cand["extra_env"]),
        ) as srv:
            rec["ready_s"] = round(time.time() - t0, 1)
            rec["served_ok"] = True
            rec["served_model_id"] = srv.model_id

            # Warmup: hot the cudagraph / compile / KV caches.
            for i in range(min(args.warmup_prompts, len(prompts))):
                decode_mod.request_decode(
                    base_url=base_url, model=SERVED_NAME,
                    prompt_token_ids=prompts[i]["prompt_token_ids"],
                    output_len=args.output_len, timeout_s=args.request_timeout_s,
                )
            rec["gpu_mem_used_mib"] = gpu_mem_used_mib()

            for r in range(args.reps):
                print(f"[measure] --- rep {r} ---", flush=True)
                rd = run_instrumented_rep(
                    decode_mod, base_url, prompts, args.output_len, args.request_timeout_s
                )
                (out_dir / f"rep{r}_per_prompt.json").write_text(json.dumps(rd["per_prompt"], indent=2))
                # keep aggregates + drop bulky per_prompt from the in-memory rec for reps>0
                summary = {k: v for k, v in rd.items() if k != "per_prompt"}
                summary["rep"] = r
                rec["reps_detail"].append(summary)
                rd["per_prompt_keep"] = rd["per_prompt"]
                if r == 0:
                    rec["_rep0"] = rd  # full detail used for distributions
                print(f"[measure] rep{r} wall_tps loopclock={rd['wall_tps_loopclock']:.2f} "
                      f"tight={rd['wall_tps_tight']:.2f} tot_tokens={rd['tot_tokens']}", flush=True)
                # accumulate per-prompt walls across reps for stable per-prompt dist
                if "_walls_by_prompt" not in rec:
                    rec["_walls_by_prompt"] = [[] for _ in prompts]
                    rec["_eacc_by_prompt"] = [[] for _ in prompts]
                    rec["_tokens_by_prompt"] = [p["n_out"] for p in rd["per_prompt"]]
                for i, p in enumerate(rd["per_prompt"]):
                    rec["_walls_by_prompt"][i].append(p["wall_s"])
                    if p["e_accept"] is not None:
                        rec["_eacc_by_prompt"][i].append(p["e_accept"])

            # Official capture_decode cross-check (no instrumentation -> gold n/d).
            if not args.no_official_crosscheck:
                try:
                    s = harness.capture_decode(
                        server_py, base_url=base_url, model=SERVED_NAME,
                        out_file=out_dir / "official.jsonl",
                        summary_file=out_dir / "official.summary.json",
                        num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
                    )
                    n = int(s.get("num_completion_tokens", 0))
                    d = float(s.get("duration_s", 0.0))
                    rec["official_crosscheck"] = {
                        "num_completion_tokens": n, "duration_s": d,
                        "wall_tps": (n / d) if d > 0 else None,
                    }
                    print(f"[measure] official capture_decode wall_tps="
                          f"{rec['official_crosscheck']['wall_tps']:.2f}", flush=True)
                except Exception as exc:
                    rec["official_crosscheck"] = {"error": str(exc)}

            # Prefill-EXCLUDED single-stream probe (reconcile the 256.74/265.61 headline).
            try:
                rec["probe_tps"] = harness.probe_tps(base_url, SERVED_NAME, decode_tokens=256)
            except Exception as exc:
                rec["probe_tps"] = {"error": str(exc)}

            if not args.no_ppl:
                try:
                    ppl = harness.run_ppl(
                        server_py, base_url=base_url, model=SERVED_NAME,
                        out_file=out_dir / "ppl.jsonl", summary_file=out_dir / "ppl.summary.json",
                    )
                    rec["ppl"] = ppl.get("ppl")
                except Exception as exc:
                    rec["ppl_error"] = str(exc)
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"[measure] ERROR: {exc}", flush=True)

    rec["t_end_utc"] = datetime.now(timezone.utc).isoformat()
    _finalize_stats(rec, args)
    return rec


def _finalize_stats(rec: dict[str, Any], args) -> None:
    reps = rec.get("reps_detail") or []
    loop_tps = [r["wall_tps_loopclock"] for r in reps if r.get("wall_tps_loopclock")]
    tight_tps = [r["wall_tps_tight"] for r in reps if r.get("wall_tps_tight")]
    # rep-level repeatability CI (measurement noise across reps)
    rec["agg_wall_tps_loopclock_mean"] = statistics.fmean(loop_tps) if loop_tps else None
    rec["agg_wall_tps_loopclock_std"] = statistics.pstdev(loop_tps) if len(loop_tps) > 1 else 0.0
    rec["agg_wall_tps_tight_mean"] = statistics.fmean(tight_tps) if tight_tps else None
    rec["agg_wall_tps_tight_std"] = statistics.pstdev(tight_tps) if len(tight_tps) > 1 else 0.0
    rec["rep_loopclock_tps"] = loop_tps
    rec["rep_tight_tps"] = tight_tps

    # scrape overhead = how much the per-prompt /metrics reads inflate loopclock vs tight
    if rec["agg_wall_tps_loopclock_mean"] and rec["agg_wall_tps_tight_mean"]:
        rec["scrape_overhead_pct"] = round(
            100.0 * (rec["agg_wall_tps_tight_mean"] - rec["agg_wall_tps_loopclock_mean"])
            / rec["agg_wall_tps_tight_mean"], 3)

    # Stable per-prompt aggregate from mean wall across reps.
    walls_by = rec.pop("_walls_by_prompt", None)
    eacc_by = rec.pop("_eacc_by_prompt", None)
    tokens_by = rec.pop("_tokens_by_prompt", None)
    rec.pop("_rep0", None)
    if walls_by and tokens_by:
        mean_wall = [statistics.fmean(w) if w else None for w in walls_by]
        pp_tps = [(t / w) if w else None for t, w in zip(tokens_by, mean_wall)]
        pp_eacc = [statistics.fmean(e) if e else None for e in eacc_by] if eacc_by else []
        rec["per_prompt_tps_dist"] = _dist(pp_tps)
        rec["per_prompt_e_accept_dist"] = _dist(pp_eacc)
        # token-weighted aggregate from the stable per-prompt walls
        valid = [(t, w) for t, w in zip(tokens_by, mean_wall) if w]
        if valid:
            rec["agg_wall_tps_from_meanwall"] = sum(t for t, _ in valid) / sum(w for _, w in valid)
        rec["bootstrap_ci_95"] = bootstrap_aggregate_ci(
            [t for t, w in zip(tokens_by, mean_wall) if w],
            [w for w in mean_wall if w],
        )

    # Verdict vs 250: take the most conservative defensible lower bound.
    lowers = []
    boot = rec.get("bootstrap_ci_95") or {}
    if boot.get("lo"):
        lowers.append(("bootstrap_p2.5", boot["lo"]))
    if rec.get("agg_wall_tps_loopclock_mean") is not None:
        lo2 = rec["agg_wall_tps_loopclock_mean"] - 2 * (rec.get("agg_wall_tps_loopclock_std") or 0.0)
        lowers.append(("rep_mean_minus_2std_loopclock", lo2))
    rec["lower_bounds"] = dict(lowers)
    if lowers:
        worst_name, worst_val = min(lowers, key=lambda kv: kv[1])
        rec["conservative_lower_bound"] = worst_val
        rec["conservative_lower_bound_source"] = worst_name
        rec["clears_250_robust"] = bool(worst_val > TARGET_TPS)
    # headline reconciliation
    probe = rec.get("probe_tps") or {}
    rec["probe_decode_tps_single_stream"] = probe.get("decode_tps_single_stream")
    rec["probe_naive_tps"] = probe.get("naive_tps")


def log_wandb(args, rec: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[measure] wandb import failed ({exc}); skipping", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="bi0-firecand-tps-faithful", agent="wirbel",
        name=args.wandb_name or f"wirbel/firecand-faithful-{args.candidate}",
        group=args.wandb_group,
        tags=["bi0-firecand-tps-faithful", "analysis-only", "pr825", args.candidate],
        config={
            "analysis_only": True, "official_tps": 0,
            "candidate": args.candidate,
            "serve_mode": rec["serve_mode"],
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "reps": args.reps, "seed": args.seed,
            "headline_tps": rec["headline_tps"], "headline_ppl": rec["headline_ppl"],
            "target_tps": TARGET_TPS, "ppl_cap": PPL_CAP,
        },
    )
    if run is None:
        print("[measure] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = None
    try:
        run_id = run.id
        boot = rec.get("bootstrap_ci_95") or {}
        ppd = rec.get("per_prompt_tps_dist") or {}
        ead = rec.get("per_prompt_e_accept_dist") or {}
        off = rec.get("official_crosscheck") or {}
        metrics = {
            "faithful/agg_wall_tps_loopclock_mean": rec.get("agg_wall_tps_loopclock_mean"),
            "faithful/agg_wall_tps_loopclock_std": rec.get("agg_wall_tps_loopclock_std"),
            "faithful/agg_wall_tps_tight_mean": rec.get("agg_wall_tps_tight_mean"),
            "faithful/agg_wall_tps_from_meanwall": rec.get("agg_wall_tps_from_meanwall"),
            "faithful/official_crosscheck_wall_tps": off.get("wall_tps"),
            "faithful/scrape_overhead_pct": rec.get("scrape_overhead_pct"),
            "faithful/bootstrap_lo_95": boot.get("lo"),
            "faithful/bootstrap_hi_95": boot.get("hi"),
            "faithful/bootstrap_median": boot.get("median"),
            "faithful/conservative_lower_bound": rec.get("conservative_lower_bound"),
            "faithful/clears_250_robust": 1.0 if rec.get("clears_250_robust") else 0.0,
            "faithful/probe_decode_tps_single_stream": rec.get("probe_decode_tps_single_stream"),
            "faithful/probe_naive_tps": rec.get("probe_naive_tps"),
            "perprompt/tps_min": ppd.get("min"), "perprompt/tps_median": ppd.get("median"),
            "perprompt/tps_max": ppd.get("max"), "perprompt/tps_std": ppd.get("std"),
            "perprompt/tps_mean": ppd.get("mean"),
            "perprompt/e_accept_min": ead.get("min"), "perprompt/e_accept_median": ead.get("median"),
            "perprompt/e_accept_max": ead.get("max"), "perprompt/e_accept_std": ead.get("std"),
            "perprompt/e_accept_mean": ead.get("mean"),
            "faithful/ppl": rec.get("ppl"),
            "faithful/headline_tps": rec.get("headline_tps"),
            "faithful/gpu_mem_used_mib": rec.get("gpu_mem_used_mib"),
            "faithful/ready_s": rec.get("ready_s"),
        }
        metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, metrics, step=0)
        wandb_logging.log_json_artifact(
            run, name=f"firecand_faithful_{args.candidate}",
            artifact_type="bi0-firecand-tps-faithful", data=rec,
        )
    except Exception as exc:
        print(f"[measure] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--request-timeout-s", type=int, default=300)
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--no-official-crosscheck", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="bi0-firecand-tps-faithful")
    ap.add_argument("--out", default=None, help="results jsonl to append to")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2 prompts / 16 tokens / 1 rep / no ppl / no crosscheck / no wandb")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = 2
        args.output_len = 16
        args.reps = 1
        args.warmup_prompts = 1
        args.no_ppl = True
        args.no_official_crosscheck = True
        args.no_wandb = True

    for note in paths.prepare_local_gpu_env():
        print(f"[measure] {note}", flush=True)

    rec = measure(args)
    run_id = None if args.smoke else log_wandb(args, rec)
    rec["wandb_run_id"] = run_id

    out = Path(args.out or (ROOT / "research" / "bi0_firecand_tps_faithful" /
                            ("smoke.jsonl" if args.smoke else "results.jsonl")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}) + "\n")

    print("\n[measure] ===== RESULT =====", flush=True)
    print(f"  candidate={rec['candidate']} served={rec['served_ok']} error={rec['error']}", flush=True)
    print(f"  agg_wall_tps loopclock mean={rec.get('agg_wall_tps_loopclock_mean')} "
          f"std={rec.get('agg_wall_tps_loopclock_std')} reps={rec.get('rep_loopclock_tps')}", flush=True)
    print(f"  agg_wall_tps tight   mean={rec.get('agg_wall_tps_tight_mean')} "
          f"(scrape_overhead={rec.get('scrape_overhead_pct')}%)", flush=True)
    off = rec.get("official_crosscheck") or {}
    print(f"  official_crosscheck wall_tps={off.get('wall_tps')}", flush=True)
    print(f"  bootstrap_ci_95={rec.get('bootstrap_ci_95')}", flush=True)
    print(f"  conservative_lower_bound={rec.get('conservative_lower_bound')} "
          f"({rec.get('conservative_lower_bound_source')}) clears_250={rec.get('clears_250_robust')}", flush=True)
    print(f"  per_prompt_tps_dist={rec.get('per_prompt_tps_dist')}", flush=True)
    print(f"  per_prompt_e_accept_dist={rec.get('per_prompt_e_accept_dist')}", flush=True)
    print(f"  probe (prefill-excl) decode_tps={rec.get('probe_decode_tps_single_stream')} "
          f"naive_tps={rec.get('probe_naive_tps')}  headline={rec.get('headline_tps')} "
          f"({rec.get('headline_kind')})", flush=True)
    print(f"  ppl={rec.get('ppl')} (cap {PPL_CAP})  mem={rec.get('gpu_mem_used_mib')}MiB "
          f"ready={rec.get('ready_s')}s wandb={run_id}", flush=True)
    return 0 if rec["served_ok"] and not rec["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
