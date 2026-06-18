#!/usr/bin/env python
"""PR #673 -- drafter-FREE speculative decoding wall-TPS / acceptance / byte-identity screen.

ANALYSIS-ONLY. Serves the shipped ``int4_g128_lmhead`` body on vLLM dev307 with
``VLLM_BATCH_INVARIANT=1`` + greedy and sweeps drafter-free speculative proposers
(ngram/prompt-lookup, Arctic suffix) at several ``num_speculative_tokens`` K. For
each cell it measures, on the OFFICIAL TPS prompt distribution (128 ShareGPT
reasoning prompts, output 512, conc=1, temp 0, ignore_eos -- the exact
``decode_outputs.py`` methodology the leaderboard scores):

  * un-rescued wall-TPS = num_completion_tokens / duration_s  (the #72 robust local
    metric; median over ``--repeats`` fresh servers),
  * acceptance = vLLM mean acceptance length (1 + accepted/drafts) and draft
    acceptance rate (accepted/proposed) -- the harness-independent mechanism signal,
  * greedy byte-identity vs the AR anchor (per-prompt completion-token sha256) ->
    break_rate; lossless verify should give 0.

The AR anchor cell (spec OFF, same body+engine) is the matched control AND the
byte-identity reference; it should reproduce ~126.94 local (=126.378 official).

Decision (logged as W&B summary scalars, with analysis_only=true / official_tps=0):
  official-equiv = local_walltps * 0.870 (stark tax, drafter-MTP-calibrated; a
  caveat for drafter-free is reported). drafterfree_best_walltps (official-equiv)
  vs SHIP 126.378 and +10 136.378 -> DRAFTERFREE_CLEARS_10 / _PARTIAL / _DEAD.

NO HF Job, NO submission, NO change to submissions/int4_g128_lmhead/serve.py.

Example::

    /tmp/senpai-venvs/.../bin/python -m research.validity.drafterfree_specdec.run_drafterfree \
        --cells ar,ngram:5,ngram:6 --repeats 1 --wandb-name kanna/drafterfree-screen
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE  # this dir holds manifest.json + serve_drafterfree.py

# Decision constants (PR #673 / project memory).
STARK_TAX = 0.870          # official-equiv = local_spec_walltps * tax (MTP-calibrated)
SHIP_BAR = 126.378         # shipped int4_g128_lmhead official TPS (fire bar)
PLUS10_BAR = 136.378       # SHIP + 10 (2nd +10 lever bar)
AR_LOCAL_ANCHOR = 126.94   # expected local AR M=1 (sanity target for the anchor cell)


# ---------------------------------------------------------------------------
# GPU preflight: reap any stale vLLM and wait for VRAM to drain (single-GPU pod).
# ---------------------------------------------------------------------------
def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def preflight_gpu(mem_threshold_mib: int = 1500, timeout_s: int = 180) -> None:
    patterns = ["vllm.entrypoints.openai.api_server", "VLLM::EngineCore",
                "multiprocessing.resource_tracker"]
    reaped = False
    for pat in patterns:
        r = subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        reaped = reaped or (r.returncode == 0)
    if reaped:
        print("[df] preflight: reaped lingering vLLM process(es)", flush=True)
        time.sleep(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_mem_used_mib()
        if used is None or used < mem_threshold_mib:
            print(f"[df] preflight: GPU ready ({used} MiB used)", flush=True)
            return
        time.sleep(3)
    print(f"[df] preflight: WARN GPU still busy ({_gpu_mem_used_mib()} MiB)", flush=True)


# ---------------------------------------------------------------------------
# Cell -> speculative-config JSON (drafter-free). "" => plain AR anchor.
# ---------------------------------------------------------------------------
def build_spec_json(method: str, k: int, pl_min: int, pl_max: int) -> str:
    if method == "ar":
        return ""
    if method == "ngram":
        return json.dumps({
            "method": "ngram",
            "num_speculative_tokens": k,
            "prompt_lookup_min": pl_min,
            "prompt_lookup_max": pl_max,
        })
    if method == "suffix":
        return json.dumps({"method": "suffix", "num_speculative_tokens": k})
    raise ValueError(f"unknown method {method!r}")


def parse_cell(cell: str) -> tuple[str, int]:
    """'ar' -> ('ar',0); 'ngram:6' -> ('ngram',6); 'suffix:5' -> ('suffix',5)."""
    if cell == "ar":
        return "ar", 0
    method, _, kstr = cell.partition(":")
    return method, int(kstr)


# ---------------------------------------------------------------------------
# Prometheus /metrics acceptance scrape (whole-run, exact).
# ---------------------------------------------------------------------------
def scrape_metrics(base_url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=30) as r:
            return serve_profile.parse_spec_metrics(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError) as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Byte-identity: per-prompt completion-token sha256 of a cell vs the AR reference.
# ---------------------------------------------------------------------------
def load_completion_shas(decode_jsonl: Path) -> dict[str, str]:
    shas: dict[str, str] = {}
    with decode_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            shas[str(row["id"])] = row["completion_token_sha256"]
    return shas


def byte_identity(ref_shas: dict[str, str], cell_jsonl: Path) -> dict[str, Any]:
    cell_shas = load_completion_shas(cell_jsonl)
    common = sorted(set(ref_shas) & set(cell_shas))
    mism = [pid for pid in common if ref_shas[pid] != cell_shas[pid]]
    n = len(common)
    return {
        "n_compared": n,
        "n_mismatch": len(mism),
        "break_rate": (len(mism) / n) if n else None,
        "byte_exact": (len(mism) == 0 and n > 0),
        "mismatch_ids": mism[:10],
    }


# ---------------------------------------------------------------------------
# One fresh server + one official decode pass -> per-run record.
# ---------------------------------------------------------------------------
def run_one(cell: str, method: str, k: int, repeat: int, out_dir: Path, server_python: Path,
            *, num_prompts: int, output_len: int, pl_min: int, pl_max: int,
            batch_invariant: int, max_model_len: int, max_num_seqs: int) -> dict[str, Any]:
    spec_json = build_spec_json(method, k, pl_min, pl_max)
    extra_env = {
        "VLLM_BATCH_INVARIANT": str(batch_invariant),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",         # emit SpecDecoding + gen-throughput log lines
        "MAX_MODEL_LEN": str(max_model_len),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "GPU_MEMORY_UTILIZATION": "0.90",
        "SPEC_CONFIG_JSON": spec_json,
        "MODEL_ID": "/workspace/gemma_build/int4_g128_lmhead",
    }
    server_log = out_dir / f"server_{cell.replace(':', '')}_r{repeat}.log"
    decode_jsonl = out_dir / f"decode_{cell.replace(':', '')}_r{repeat}.jsonl"
    decode_summary = out_dir / f"decode_{cell.replace(':', '')}_r{repeat}.summary.json"

    preflight_gpu()
    print(f"\n[df] === cell={cell} repeat={repeat} spec={spec_json or 'AR'} ===", flush=True)
    t_load0 = time.time()
    rec: dict[str, Any] = {"cell": cell, "method": method, "k": k, "repeat": repeat,
                           "spec_config_json": spec_json}
    with harness.LocalServer(
        SUBMISSION, server_python=server_python, port=8000, log_path=server_log,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        rec["server_ready_s"] = time.time() - t_load0
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_jsonl, summary_file=decode_summary,
            num_prompts=num_prompts, output_len=output_len, timeout_s=5400,
        )
        rec["metrics"] = scrape_metrics(srv.base_url)
    # wall-TPS (#72 protocol)
    n_tok = int(summary.get("num_completion_tokens", 0))
    dur = float(summary.get("duration_s", 0.0))
    rec["num_completion_tokens"] = n_tok
    rec["duration_s"] = dur
    rec["wall_tps"] = (n_tok / dur) if dur > 0 else float("nan")
    rec["decode_jsonl"] = str(decode_jsonl)
    rec["num_records"] = int(summary.get("num_records", 0))
    # acceptance from server log (cross-check / fallback to /metrics)
    try:
        rec["spec_log"] = serve_profile.parse_spec_log(server_log.read_text())
    except OSError as exc:
        rec["spec_log"] = {"error": str(exc)}
    print(f"[df] cell={cell} r{repeat}: wall_tps={rec['wall_tps']:.2f} "
          f"tokens={n_tok} dur={dur:.1f}s ready={rec['server_ready_s']:.0f}s "
          f"accept_len={(rec['metrics'] or {}).get('e_accept_mean_acceptance_length')}", flush=True)
    return rec


def _acceptance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best acceptance estimate per cell: /metrics first, server-log fallback."""
    al_metrics, dar_metrics, al_log = [], [], []
    for r in records:
        m = r.get("metrics") or {}
        if isinstance(m.get("e_accept_mean_acceptance_length"), (int, float)):
            al_metrics.append(m["e_accept_mean_acceptance_length"])
        if isinstance(m.get("draft_acceptance_rate"), (int, float)):
            dar_metrics.append(m["draft_acceptance_rate"])
        sl = r.get("spec_log") or {}
        if isinstance(sl.get("e_accept_exact"), (int, float)):
            al_log.append(sl["e_accept_exact"])
    def _mean(xs):
        return statistics.fmean(xs) if xs else None
    return {
        "mean_acceptance_length_metrics": _mean(al_metrics),
        "draft_acceptance_rate_metrics": _mean(dar_metrics),
        "mean_acceptance_length_serverlog": _mean(al_log),
    }


def aggregate_cell(records: list[dict[str, Any]], ar_median: float | None) -> dict[str, Any]:
    tps = [r["wall_tps"] for r in records if isinstance(r.get("wall_tps"), (int, float))
           and r["wall_tps"] == r["wall_tps"]]
    med = statistics.median(tps) if tps else float("nan")
    acc = _acceptance(records)
    out = {
        "n_runs": len(records),
        "wall_tps_values": tps,
        "wall_tps_median": med,
        "wall_tps_mean": statistics.fmean(tps) if tps else float("nan"),
        "wall_tps_min": min(tps) if tps else None,
        "wall_tps_max": max(tps) if tps else None,
        "official_equiv_tax": med * STARK_TAX if med == med else float("nan"),
        **acc,
    }
    if ar_median and med == med:
        out["speedup_vs_ar"] = med / ar_median
        out["official_equiv_ratio"] = SHIP_BAR * (med / ar_median)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="ar,ngram:5,ngram:6",
                    help="comma list: ar, ngram:K, suffix:K")
    ap.add_argument("--repeats", type=int, default=1, help="fresh servers per cell (median)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--prompt-lookup-min", type=int, default=2)
    ap.add_argument("--prompt-lookup-max", type=int, default=6)
    ap.add_argument("--batch-invariant", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=HERE / "_runs")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="drafterfree-specdec")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[df] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[df] server_python={server_python}", flush=True)

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_fh = (out_dir / "records.jsonl").open("w")

    # AR cell must run first (matched control + byte-identity reference).
    if "ar" in cells:
        cells = ["ar"] + [c for c in cells if c != "ar"]

    all_records: dict[str, list[dict[str, Any]]] = {}
    failed_cells: dict[str, str] = {}
    ar_ref_shas: dict[str, str] | None = None
    t0 = time.time()
    for cell in cells:
        method, k = parse_cell(cell)
        recs: list[dict[str, Any]] = []
        try:
            for rep in range(args.repeats):
                rec = run_one(
                    cell, method, k, rep, out_dir, server_python,
                    num_prompts=args.num_prompts, output_len=args.output_len,
                    pl_min=args.prompt_lookup_min, pl_max=args.prompt_lookup_max,
                    batch_invariant=args.batch_invariant, max_model_len=args.max_model_len,
                    max_num_seqs=args.max_num_seqs,
                )
                recs.append(rec)
                records_fh.write(json.dumps(rec) + "\n")
                records_fh.flush()
        except Exception as exc:  # one cell (e.g. suffix unavailable) must not abort the sweep
            print(f"[df] cell={cell} FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed_cells[cell] = f"{type(exc).__name__}: {exc}"
            preflight_gpu()  # reap any half-dead server before the next cell
            if not recs:
                continue
        if not recs:
            continue
        all_records[cell] = recs
        # byte-identity wiring
        first_jsonl = Path(recs[0]["decode_jsonl"])
        if cell == "ar":
            ar_ref_shas = load_completion_shas(first_jsonl)
        elif ar_ref_shas is not None:
            bi = byte_identity(ar_ref_shas, first_jsonl)
            for r in recs:
                r["byte_identity"] = bi
            print(f"[df] cell={cell} byte_identity: break_rate={bi['break_rate']} "
                  f"({bi['n_mismatch']}/{bi['n_compared']})", flush=True)
        # self-consistency (r_i vs r_0, fresh servers, same config) -- isolates
        # engine/run-to-run nondeterminism from genuine spec divergence. A ~0
        # AR self-consistency validates the byte-identity reference; a non-zero
        # one means the break_rate vs AR is confounded by engine noise.
        if len(recs) > 1:
            r0_shas = load_completion_shas(Path(recs[0]["decode_jsonl"]))
            sc_runs = [byte_identity(r0_shas, Path(recs[i]["decode_jsonl"]))
                       for i in range(1, len(recs))]
            sc_rates = [s["break_rate"] for s in sc_runs if s["break_rate"] is not None]
            sc_mean = statistics.fmean(sc_rates) if sc_rates else None
            sc = {"runs": sc_runs, "mean_break_rate": sc_mean, "n_fresh_servers": len(recs)}
            for r in recs:
                r["self_consistency"] = sc
            print(f"[df] cell={cell} self_consistency: mean_break_rate={sc_mean} "
                  f"(r_i vs r_0 across {len(recs)} fresh servers)", flush=True)
    records_fh.close()
    elapsed = time.time() - t0

    # ---- aggregate + verdict ----
    ar_median = None
    if "ar" in all_records:
        ar_tps = [r["wall_tps"] for r in all_records["ar"]
                  if isinstance(r.get("wall_tps"), (int, float))]
        ar_median = statistics.median(ar_tps) if ar_tps else None

    cell_agg: dict[str, Any] = {}
    for cell, recs in all_records.items():
        agg = aggregate_cell(recs, ar_median)
        if cell != "ar" and ar_median is not None and recs[0].get("byte_identity"):
            agg["byte_identity"] = recs[0]["byte_identity"]
        if recs[0].get("self_consistency"):
            agg["self_consistency"] = recs[0]["self_consistency"]
        cell_agg[cell] = agg

    spec_cells = {c: a for c, a in cell_agg.items() if c != "ar"}
    best_cell, best_off_equiv = None, float("-inf")
    for c, a in spec_cells.items():
        oe = a.get("official_equiv_tax")
        if isinstance(oe, (int, float)) and oe == oe and oe > best_off_equiv:
            best_off_equiv, best_cell = oe, c

    if best_cell is None:
        verdict = "DRAFTERFREE_DEAD"
    elif best_off_equiv >= PLUS10_BAR:
        verdict = "DRAFTERFREE_CLEARS_10"
    elif best_off_equiv > SHIP_BAR:
        verdict = "DRAFTERFREE_PARTIAL"
    else:
        verdict = "DRAFTERFREE_DEAD"

    result = {
        "analysis_only": True,
        "official_tps": 0,
        "stark_tax": STARK_TAX,
        "ship_bar": SHIP_BAR,
        "plus10_bar": PLUS10_BAR,
        "ar_local_median": ar_median,
        "ar_local_anchor_expected": AR_LOCAL_ANCHOR,
        "cells": cell_agg,
        "drafterfree_best_cell": best_cell,
        "drafterfree_best_walltps_official_equiv": best_off_equiv if best_cell else None,
        "verdict": verdict,
        "failed_cells": failed_cells,
        "elapsed_s": elapsed,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "conc": 1, "temp": 0.0, "ignore_eos": True,
                     "batch_invariant": args.batch_invariant,
                     "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
                     "prompt_lookup_min": args.prompt_lookup_min,
                     "prompt_lookup_max": args.prompt_lookup_max},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    _print_summary(result)
    if not args.no_wandb:
        _log_wandb(result, args)
    return 0


def _print_summary(result: dict[str, Any]) -> None:
    print("\n[df] ===================== DRAFTER-FREE SPEC SCREEN =====================", flush=True)
    print(f"[df] AR local median = {result['ar_local_median']} (expected ~{result['ar_local_anchor_expected']})",
          flush=True)
    print(f"[df] bars: SHIP={result['ship_bar']}  +10={result['plus10_bar']}  tax={result['stark_tax']}",
          flush=True)
    for cell, a in result["cells"].items():
        bi = a.get("byte_identity") or {}
        sc = a.get("self_consistency") or {}
        print(f"[df] {cell:12s} wall_tps_med={a['wall_tps_median']:.2f} "
              f"off_equiv_tax={a.get('official_equiv_tax', float('nan')):.2f} "
              f"speedup={a.get('speedup_vs_ar', float('nan')):.3f} "
              f"accept_len={a.get('mean_acceptance_length_metrics')} "
              f"break_rate={bi.get('break_rate')} "
              f"self_break={sc.get('mean_break_rate')}", flush=True)
    print(f"[df] VERDICT: {result['verdict']} (best={result['drafterfree_best_cell']} "
          f"@ {result['drafterfree_best_walltps_official_equiv']})", flush=True)


def _log_wandb(result: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[df] wandb import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=args.wandb_name or "kanna/drafterfree-screen",
            group=args.wandb_group, job_type="analysis-only",
            tags=["drafterfree-specdec", "analysis-only", "pr673", "ngram", "suffix"],
            config={
                "analysis_only": True, "cells": args.cells, "repeats": args.repeats,
                "num_prompts": args.num_prompts, "output_len": args.output_len,
                "batch_invariant": args.batch_invariant, "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs, "stark_tax": STARK_TAX,
                "prompt_lookup_min": args.prompt_lookup_min,
                "prompt_lookup_max": args.prompt_lookup_max,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[df] wandb init failed ({exc}); skipping", flush=True)
        return
    # Machine-checkable no-fire guard + decision scalars.
    summary: dict[str, Any] = {
        "analysis_only": True,
        "official_tps": 0,
        "verdict": result["verdict"],
        "drafterfree_best_cell": result["drafterfree_best_cell"],
        "drafterfree_best_walltps": result["drafterfree_best_walltps_official_equiv"],
        "ar_local_median": result["ar_local_median"],
        "ship_bar": SHIP_BAR, "plus10_bar": PLUS10_BAR, "stark_tax": STARK_TAX,
    }
    for cell, a in result["cells"].items():
        tag = cell.replace(":", "_")
        for key in ("wall_tps_median", "wall_tps_mean", "official_equiv_tax",
                    "official_equiv_ratio", "speedup_vs_ar",
                    "mean_acceptance_length_metrics", "draft_acceptance_rate_metrics",
                    "mean_acceptance_length_serverlog"):
            v = a.get(key)
            if isinstance(v, (int, float)) and v == v:
                summary[f"{tag}/{key}"] = v
        bi = a.get("byte_identity") or {}
        if isinstance(bi.get("break_rate"), (int, float)):
            summary[f"{tag}/break_rate"] = bi["break_rate"]
            summary[f"{tag}/byte_exact"] = 1 if bi.get("byte_exact") else 0
        sc = a.get("self_consistency") or {}
        if isinstance(sc.get("mean_break_rate"), (int, float)):
            summary[f"{tag}/self_break_rate"] = sc["mean_break_rate"]
    run.summary.update(summary)
    # Per-cell table
    tbl = wandb.Table(columns=["cell", "wall_tps_median", "official_equiv_tax",
                               "speedup_vs_ar", "mean_acceptance_length", "break_rate"])
    for cell, a in result["cells"].items():
        bi = a.get("byte_identity") or {}
        tbl.add_data(cell, a.get("wall_tps_median"), a.get("official_equiv_tax"),
                     a.get("speedup_vs_ar"), a.get("mean_acceptance_length_metrics"),
                     bi.get("break_rate"))
    run.log({"cells": tbl})
    print(f"[df] wandb run: {run.url} (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
