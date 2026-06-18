#!/usr/bin/env python
"""g32 official-QAT recipe speed gate (PR #649).

ANALYSIS-ONLY, LOCAL single-A10G. No HF jobs, no submissions, no leaderboard
change. Does NOT touch the locked ``int4_g128_lmhead`` served file.

lawine #639 showed Google's official ``gemma-4-E4B-it-qat-w4a16-ct`` (g32 body +
**tied bf16 lm_head**) RECOVERS GPQA quality (93.6% of bf16 base, clears the 90%
bar where our shipped g128 body fails ~86%). This card is the SPEED half: does
that quality-recovering recipe still beat the **126.378** official anchor at
strict-#319?

The locked rung took TWO speed-trades vs the official recipe:
  1. group_size 32 -> 128 (fewer scales = faster Marlin, larger quant error)
  2. untie + int4-quantize lm_head  (official keeps it tied/bf16)

This profiler serves each checkpoint through the **exact #640 config-0 manifest**
(vLLM 0.22.0, ``speculative_config=None`` pure autoregressive) and measures the
in-harness single-stream ``wall_tps``. Local->official projection re-anchors the
locked g128 rung **in-session** to its known official TPS, then scales by ratio::

    official_proj = OFFICIAL_ANCHOR_TPS * wall_tps_cell / wall_tps_g128anchor

so the absolute local number need not equal #640's 127.683 (it should land near
it; that is a sanity check, not the calibration).

Cells form a 2x2 body x head factorial::

    body \\ head | int4-untied      | tied-bf16
    ------------+------------------+------------------
    g128        | g128_int4head (D)| g128_bf16head (C)
    g32         | g32_int4head (B) | g32_bf16head  (A)

D = locked rung (the anchor). A = g32 official recipe. C is a cheap safetensors
surgery (drop int4 lm_head, tie bf16 embed). Speed-trade attribution telescopes
on A,C,D with NO additivity assumption::

    int4-head bought  = D - C   (head precision, at g128 body)
    g128-group bought = C - A   (group size, at bf16 head)
    total             = D - A   = (D-C) + (C-A)

(card corner B, g32+int4-head, would only supply the alternative split D-B; it
needs a real lm_head re-quantization, not cheap, and is redundant for the total.)

Strict-#319 self-consistency: g32 is a *different weight set*, so this is g32's
OWN greedy reproducibility (per-prompt completion sha invariant across reps x
seeds), NOT byte-exact agreement with the g128 rung. Greedy is argmax -> seed
independent by construction, so a flip across seeds/reps is genuine int4-Marlin
nondeterminism.

Example::

    .venv/bin/python -m scripts.profiler.g32_recipe_speed_gate \
        --num-prompts 8 --output-len 512 --warmups 1 --reps 3 --seeds 1,2 \
        --cells g32_bf16head --ppl-cells g128_int4head,g32_bf16head \
        --selfconsist-cells g32_bf16head \
        --wandb-name wirbel/g32-recipe-speed-gate \
        --wandb-group g32-recipe-speed-gate-wirbel
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
# Reuse the #640 serve/measure plumbing verbatim so the serve path is identical.
from scripts.profiler.serveconfig_tps_sweep import (  # noqa: E402
    BASE_ENV,
    DEPS,
    OFFICIAL_ANCHOR_TPS,
    REFERENCE_PPL,
    SERVED_NAME,
    _gpu_mem_used_mib,
    _kill,
    fingerprint,
    preflight,
    render_cmd,
    serve,
)

# --- checkpoint registry (2x2 corners) -------------------------------------
G32_SNAPSHOT = (
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
ANCHOR_CELL = "g128_int4head"
CELLS: dict[str, dict[str, str]] = {
    # D: locked rung; the in-session projection anchor (official 126.378).
    "g128_int4head": {"model": "/workspace/gemma_build/int4_g128_lmhead",
                      "body": "g128", "head": "int4-untied",
                      "note": "LOCKED rung (PR #4): g128 body + int4-untied lm_head"},
    # A: Google official QAT recipe (lawine #639 quality-recoverer).
    "g32_bf16head": {"model": G32_SNAPSHOT,
                     "body": "g32", "head": "tied-bf16",
                     "note": "official gemma-4-E4B-it-qat-w4a16-ct: g32 body + tied bf16 lm_head"},
    # C: cheap surgery corner (built by build_g128_bf16head.py).
    "g128_bf16head": {"model": "/workspace/gemma_build/g128_bf16head",
                      "body": "g128", "head": "tied-bf16",
                      "note": "surgery: locked g128 body + tied bf16 lm_head (head-precision isolate)"},
    # B: would need real lm_head re-quantization (not built by default).
    "g32_int4head": {"model": "/workspace/gemma_build/g32_int4head",
                     "body": "g32", "head": "int4-untied",
                     "note": "g32 body + int4-untied lm_head (group-size isolate; needs requant)"},
}

REFERENCE_PPL_640 = 2.0187   # #640 config-0 local PPL
OFFICIAL_PPL_CAP = 2.42
STRICT_319_N = 8             # prompts in the self-consistency census
OUT_ROOT = ROOT / "research" / "g32_recipe_speed_gate"


def cell_flags(model_path: str, port: int) -> dict[str, Any]:
    """The EXACT #640 config-0 serve flags, with --model parameterized.

    config-0 uses vLLM's default max_num_seqs, which #640 proved is the
    single-stream optimum (max_num_seqs=1 was 0.54% slower). All cells share
    these flags so the local->official ratio is config-invariant.
    """
    return {
        "--model": model_path,
        "--served-model-name": SERVED_NAME,
        "--host": "127.0.0.1",
        "--port": str(port),
        "--dtype": "bfloat16",
        "--max-model-len": "4096",
        "--gpu-memory-utilization": "0.90",
        "--max-num-batched-tokens": "512",
        "--trust-remote-code": None,
        "--no-enable-log-requests": None,
    }


def _read_by_dataset_index(jsonl_path: Path) -> dict[int, str]:
    """Map ``dataset_index -> completion_token_sha256`` (prompt-keyed).

    Self-consistency MUST key on ``dataset_index``, not the request-order
    ``index`` that :func:`fingerprint` sorts on. The decode seed permutes which
    dataset rows are sampled, so request-order position *i* is a DIFFERENT prompt
    at each seed; comparing position *i* across seeds compares unlike prompts and
    spuriously reports divergence. Keying on the stable ``dataset_index`` makes a
    capture comparable to any other capture of the same prompt.
    """
    out: dict[int, str] = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[int(o["dataset_index"])] = str(o["completion_token_sha256"])
    return out


def _capture(server_py: Path, base_url: str, of: Path, sf: Path, *,
             num_prompts: int, output_len: int, seed: int) -> tuple[float, dict[int, str], list[int]]:
    """One decode capture; returns (wall_tps, {dataset_index: sha}, completion counts)."""
    s = harness.capture_decode(
        server_py, base_url=base_url, model=SERVED_NAME,
        out_file=of, summary_file=sf,
        num_prompts=num_prompts, output_len=output_len, seed=seed,
    )
    n = int(s.get("num_completion_tokens", 0))
    d = float(s.get("duration_s", 0.0))
    tps = n / d if d > 0 else float("nan")
    by_idx = _read_by_dataset_index(of)
    _, counts = fingerprint(of)
    return tps, by_idx, counts


def measure_cell(name: str, server_py: Path, out_dir: Path, *,
                 num_prompts: int, output_len: int, seeds: list[int],
                 warmups: int, reps: int, port: int,
                 do_selfconsist: bool, do_ppl: bool) -> dict[str, Any]:
    spec = CELLS[name]
    model_path = spec["model"]
    base = out_dir / name
    base.mkdir(parents=True, exist_ok=True)
    flags = cell_flags(model_path, port)
    env = os.environ.copy()
    env.update(BASE_ENV)
    cmd = render_cmd(server_py, flags)
    (base / "cmd.txt").write_text(" ".join(cmd) + "\n")

    rec: dict[str, Any] = {
        "name": name, "model": model_path, "body": spec["body"], "head": spec["head"],
        "note": spec["note"], "flags": flags,
        "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "served_ok": False, "error": None,
        "rep_wall_tps": [], "wall_tps": None, "wall_tps_cv_pct": None,
        "ready_s": None, "gpu_mem_used_mib": None,
        "rep0_completion_counts": None, "full_length": None,
        # strict-#319 self-consistency
        "selfconsist_n_captures": 0, "selfconsist_seeds": [],
        "passes_strict_319": None, "strict_319_n_const": None, "strict_319_n_total": None,
        # ppl
        "ppl": None, "ppl_error": None,
    }
    used_before = preflight()
    rec["gpu_mem_used_before_mib"] = used_before
    base_url = f"http://127.0.0.1:{port}"
    log_path = base / "server.log"
    proc = None
    log = None
    # one {dataset_index: sha} map per capture (reps + per-seed warmed sc captures)
    sha_samples: list[dict[int, str]] = []
    try:
        t0 = time.time()
        proc, log = serve(cmd, env, log_path, port)
        rec["ready_s"] = round(time.time() - t0, 1)
        rec["served_ok"] = True

        primary_seed = seeds[0]
        for w in range(warmups):
            harness.capture_decode(
                server_py, base_url=base_url, model=SERVED_NAME,
                out_file=base / f"warm{w}.jsonl", summary_file=base / f"warm{w}.summary.json",
                num_prompts=num_prompts, output_len=output_len, seed=primary_seed,
            )
        rec["gpu_mem_used_mib"] = _gpu_mem_used_mib()

        # Timed reps at the primary seed -> wall_tps. The primary seed is already
        # cache-warm (the warmups above ran at primary_seed), so these reps are a
        # valid warm-state determinism sample. Capture sha on EVERY rep.
        for r in range(reps):
            tps, by_idx, counts = _capture(
                server_py, base_url, base / f"rep{r}.jsonl", base / f"rep{r}.summary.json",
                num_prompts=num_prompts, output_len=output_len, seed=primary_seed,
            )
            rec["rep_wall_tps"].append(tps)
            sha_samples.append(by_idx)
            if r == 0:
                rec["rep0_completion_counts"] = counts
                rec["full_length"] = all(c == output_len for c in counts) if counts else False
        vals = [v for v in rec["rep_wall_tps"] if v == v]
        rec["wall_tps"] = statistics.median(vals) if vals else float("nan")
        rec["wall_tps_min"] = min(vals) if vals else None
        rec["wall_tps_max"] = max(vals) if vals else None
        rec["wall_tps_cv_pct"] = (100.0 * statistics.pstdev(vals) / statistics.fmean(vals)
                                  if len(vals) > 1 and statistics.fmean(vals) else 0.0)

        # strict-#319 self-consistency (warm-state byte-exact greedy determinism).
        # Each non-primary seed gets its OWN warmup so its compared reps are
        # cache-warm. With prefix caching on (config-0), the FIRST cold decode of
        # a prompt can differ from later warm decodes (an r0!=r1==r2 cold/warm
        # boundary flip) -- a serve-path artifact, NOT weight nondeterminism. We
        # then check sha invariance PER dataset_index across every capture of that
        # prompt. Union over seeds widens prompt coverage beyond a single 8-set.
        if do_selfconsist:
            used_seeds = [primary_seed] * reps
            for sd in seeds[1:]:
                for w in range(max(1, warmups)):
                    harness.capture_decode(
                        server_py, base_url=base_url, model=SERVED_NAME,
                        out_file=base / f"sc_seed{sd}_warm{w}.jsonl",
                        summary_file=base / f"sc_seed{sd}_warm{w}.summary.json",
                        num_prompts=num_prompts, output_len=output_len, seed=sd,
                    )
                for k in range(reps):
                    _, by_idx, _ = _capture(
                        server_py, base_url,
                        base / f"sc_seed{sd}_r{k}.jsonl", base / f"sc_seed{sd}_r{k}.summary.json",
                        num_prompts=num_prompts, output_len=output_len, seed=sd,
                    )
                    sha_samples.append(by_idx)
                    used_seeds.append(sd)
            rec["selfconsist_n_captures"] = len(sha_samples)
            rec["selfconsist_seeds"] = sorted(set(used_seeds))
            # union of all prompts seen; a prompt is constant iff every capture of
            # it agrees (captures that don't include that prompt are skipped).
            all_idx = sorted({i for s in sha_samples for i in s})
            n_const = 0
            per_prompt = []
            for i in all_idx:
                vals_i = {s[i] for s in sha_samples if i in s}
                n_caps = sum(1 for s in sha_samples if i in s)
                const = len(vals_i) == 1
                n_const += 1 if const else 0
                per_prompt.append({"dataset_index": i, "constant": const,
                                   "n_distinct": len(vals_i), "n_captures": n_caps})
            rec["strict_319_n_const"] = n_const
            rec["strict_319_n_total"] = len(all_idx)
            rec["passes_strict_319"] = (n_const == len(all_idx) and len(all_idx) > 0)
            rec["strict_319_per_prompt"] = per_prompt

        if do_ppl:
            try:
                ppl = harness.run_ppl(
                    server_py, base_url=base_url, model=SERVED_NAME,
                    out_file=base / "ppl.jsonl", summary_file=base / "ppl.summary.json",
                )
                rec["ppl"] = ppl.get("ppl")
            except Exception as exc:
                rec["ppl_error"] = str(exc)
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"  [cell:{name}] ERROR: {exc}", flush=True)
    finally:
        _kill(proc)
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
    return rec


def attribution(by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Telescoping 2x2 attribution from whichever corners ran."""
    def tps(n: str) -> float | None:
        r = by_name.get(n)
        v = r.get("wall_tps") if r else None
        return v if (isinstance(v, (int, float)) and v == v) else None
    D = tps("g128_int4head")
    A = tps("g32_bf16head")
    C = tps("g128_bf16head")
    B = tps("g32_int4head")
    out: dict[str, Any] = {"D_g128_int4head": D, "A_g32_bf16head": A,
                           "C_g128_bf16head": C, "B_g32_int4head": B}
    if D and A:
        out["total_trade_tps"] = round(D - A, 3)
        out["total_trade_pct_of_D"] = round(100.0 * (D - A) / D, 3)
    if D and C:
        out["int4head_bought_tps"] = round(D - C, 3)          # head precision @ g128 body
        out["int4head_bought_pct_of_D"] = round(100.0 * (D - C) / D, 3)
    if C and A:
        out["g128group_bought_tps_at_bf16head"] = round(C - A, 3)  # group size @ bf16 head
        out["g128group_bought_pct_of_D"] = round(100.0 * (C - A) / D, 3) if D else None
    if D and B:
        out["g128group_bought_tps_at_int4head"] = round(D - B, 3)  # group size @ int4 head (card split)
        out["int4head_bought_tps_at_g32body"] = round(B - A, 3) if A else None
    return out


def build_summary(records: list[dict[str, Any]], anchor_local_tps: float | None) -> dict[str, Any]:
    by_name = {r["name"]: r for r in records}
    # projection: anchor the in-session g128 rung to its known official TPS.
    for r in records:
        wt = r.get("wall_tps")
        if anchor_local_tps and wt and anchor_local_tps == anchor_local_tps and wt == wt:
            r["delta_pct_vs_anchor"] = round(100.0 * (wt - anchor_local_tps) / anchor_local_tps, 3)
            r["official_proj_tps"] = round(OFFICIAL_ANCHOR_TPS * wt / anchor_local_tps, 3)
        else:
            r["delta_pct_vs_anchor"] = None
            r["official_proj_tps"] = None

    g32 = by_name.get("g32_bf16head")
    attrib = attribution(by_name)

    verdict = "INCOMPLETE"
    g32_proj = g32.get("official_proj_tps") if g32 else None
    g32_319 = g32.get("passes_strict_319") if g32 else None
    g32_ppl = g32.get("ppl") if g32 else None
    ppl_ok = (isinstance(g32_ppl, (int, float)) and g32_ppl <= OFFICIAL_PPL_CAP)
    beats_anchor = (isinstance(g32_proj, (int, float)) and g32_proj >= OFFICIAL_ANCHOR_TPS)
    if g32 and g32.get("served_ok"):
        if beats_anchor and g32_319 and ppl_ok:
            verdict = "G32_SPEED_VIABLE"
        elif ppl_ok and g32_319 and not beats_anchor:
            verdict = "G32_QUALITY_SAFE_BUT_SLOW"
        elif not beats_anchor and not g32_319:
            verdict = "G32_NO_RECIPE_UPSIDE"
        else:
            verdict = "G32_MIXED"  # e.g. fast but #319-unstable, or quality-fail
    return {
        "verdict": verdict,
        "anchor_local_wall_tps": anchor_local_tps,
        "anchor_local_wall_tps_640": 127.683,
        "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
        "official_ppl_cap": OFFICIAL_PPL_CAP,
        "reference_ppl": REFERENCE_PPL,
        "g32_official_proj_tps": g32_proj,
        "g32_wall_tps_cv_pct": g32.get("wall_tps_cv_pct") if g32 else None,
        "g32_passes_strict_319": g32_319,
        "g32_strict_319_n": f"{g32.get('strict_319_n_const')}/{g32.get('strict_319_n_total')}" if g32 else None,
        "g32_ppl": g32_ppl,
        "g32_ppl_ok": ppl_ok,
        "g32_beats_anchor": beats_anchor,
        "attribution": attrib,
    }


def _log_wandb(args, records: list[dict[str, Any]], summary: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[g32] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="g32-recipe-speed-gate", agent="wirbel",
        name=args.wandb_name or "wirbel/g32-recipe-speed-gate",
        group=args.wandb_group,
        tags=["g32-recipe-speed-gate", "analysis-only", summary["verdict"]],
        config={
            "analysis_only": True, "official_tps": 0,
            "official_anchor_tps": OFFICIAL_ANCHOR_TPS, "official_ppl_cap": OFFICIAL_PPL_CAP,
            "reference_ppl": REFERENCE_PPL,
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "seeds": args.seeds, "warmups": args.warmups, "reps": args.reps,
            "cells": [ANCHOR_CELL] + [c for c in args.cells.split(",") if c],
        },
    )
    if run is None:
        print("[g32] wandb disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return None
    run_id = None
    try:
        run_id = run.id
        for i, r in enumerate(records):
            metrics = {
                "cell/wall_tps": r.get("wall_tps"),
                "cell/wall_tps_cv_pct": r.get("wall_tps_cv_pct"),
                "cell/delta_pct_vs_anchor": r.get("delta_pct_vs_anchor"),
                "cell/official_proj_tps": r.get("official_proj_tps"),
                "cell/ready_s": r.get("ready_s"),
                "cell/gpu_mem_used_mib": r.get("gpu_mem_used_mib"),
                "cell/passes_strict_319": (1 if r.get("passes_strict_319") else 0)
                if r.get("passes_strict_319") is not None else None,
                "cell/strict_319_n_const": r.get("strict_319_n_const"),
                "cell/ppl": r.get("ppl"),
            }
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"cell_{r['name']}", step=i, metrics=metrics,
                                    data={"cell": r["name"], "body": r["body"], "head": r["head"]})
        flat: dict[str, Any] = {}
        for k, v in summary.items():
            if isinstance(v, (int, float, str, bool)):
                flat[f"verdict/{k}"] = v
        for k, v in (summary.get("attribution") or {}).items():
            if isinstance(v, (int, float)):
                flat[f"attribution/{k}"] = v
        wandb_logging.log_summary(run, flat, step=len(records))
        wandb_logging.log_json_artifact(
            run, name="g32_recipe_speed_gate", artifact_type="g32-recipe-speed-gate",
            data={"summary": summary, "records": records},
        )
    except Exception as exc:
        print(f"[g32] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


def _f(v: Any) -> str:
    if isinstance(v, (int, float)) and v == v:
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seeds", default="1,2", help="comma list; first is the timed seed")
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cells", default="g32_bf16head",
                    help=f"comma list of non-anchor cells from {list(CELLS)}; anchor {ANCHOR_CELL} always first")
    ap.add_argument("--ppl-cells", default="g128_int4head,g32_bf16head")
    ap.add_argument("--selfconsist-cells", default="g32_bf16head")
    ap.add_argument("--max-minutes", type=float, default=85.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="g32-recipe-speed-gate-wirbel")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[g32] {note}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    requested = [c.strip() for c in args.cells.split(",") if c.strip()]
    # anchor always runs first (in-session projection denominator)
    plan = [ANCHOR_CELL] + [c for c in requested if c != ANCHOR_CELL]
    for c in plan:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c!r}; known: {list(CELLS)}")
    ppl_set = {c.strip() for c in args.ppl_cells.split(",") if c.strip()}
    sc_set = {c.strip() for c in args.selfconsist_cells.split(",") if c.strip()}

    out_dir = (args.out_dir or (OUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[g32] plan={plan} seeds={seeds} workload={args.num_prompts}x{args.output_len} "
          f"warmups={args.warmups} reps={args.reps} -> {out_dir}", flush=True)
    print(f"[g32] ppl_cells={sorted(ppl_set)} selfconsist_cells={sorted(sc_set)}", flush=True)

    if args.dry_run:
        for c in plan:
            print(f"\n=== {c} [{CELLS[c]['body']}/{CELLS[c]['head']}] ===\n  {CELLS[c]['note']}", flush=True)
            print("  " + " ".join(render_cmd(Path("<server_py>"), cell_flags(CELLS[c]["model"], args.port))), flush=True)
            print(f"  model exists: {Path(CELLS[c]['model']).exists()}", flush=True)
        return 0

    # fail fast if a planned checkpoint dir is missing
    for c in plan:
        if not Path(CELLS[c]["model"]).exists():
            raise SystemExit(f"checkpoint for cell {c!r} not found: {CELLS[c]['model']}")

    server_py = harness.ensure_server_venv(DEPS)
    print(f"[g32] server_python={server_py}", flush=True)

    records: list[dict[str, Any]] = []
    records_path = out_dir / "records.jsonl"
    t_start = time.time()
    with open(records_path, "w") as fh:
        for c in plan:
            elapsed_min = (time.time() - t_start) / 60.0
            if c != ANCHOR_CELL and elapsed_min > args.max_minutes:
                print(f"[g32] budget {args.max_minutes}min reached ({elapsed_min:.1f}); stop before {c}", flush=True)
                break
            print(f"\n[g32] === {c} [{CELLS[c]['body']}/{CELLS[c]['head']}] (t+{elapsed_min:.1f}min) ===", flush=True)
            rec = measure_cell(
                c, server_py, out_dir,
                num_prompts=args.num_prompts, output_len=args.output_len, seeds=seeds,
                warmups=args.warmups, reps=args.reps, port=args.port,
                do_selfconsist=(c in sc_set), do_ppl=(c in ppl_set),
            )
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  wall_tps={_f(rec.get('wall_tps'))} cv={_f(rec.get('wall_tps_cv_pct'))}% "
                  f"reps={[round(v,2) for v in rec.get('rep_wall_tps',[])]} "
                  f"#319={rec.get('passes_strict_319')} ({rec.get('strict_319_n_const')}/{rec.get('strict_319_n_total')}) "
                  f"ppl={_f(rec.get('ppl'))} ready={_f(rec.get('ready_s'))}s mem={rec.get('gpu_mem_used_mib')}MiB", flush=True)

    anchor = next((r for r in records if r["name"] == ANCHOR_CELL and r.get("wall_tps")), None)
    anchor_local = anchor.get("wall_tps") if anchor else None
    summary = build_summary(records, anchor_local)
    summary["elapsed_min"] = round((time.time() - t_start) / 60.0, 2)
    summary["workload"] = {"num_prompts": args.num_prompts, "output_len": args.output_len,
                           "seeds": seeds, "warmups": args.warmups, "reps": args.reps}
    try:
        from scripts import wandb_logging
        summary["git"] = wandb_logging.git_info()
    except Exception:
        summary["git"] = {}

    result_path = out_dir / "summary.json"
    result_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    run_id = _log_wandb(args, records, summary)
    summary["wandb_run_id"] = run_id

    _print_summary(summary, records)
    result_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"[g32] artifacts -> {result_path}", flush=True)
    return 0


def _print_summary(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    print("\n[g32] ===================== VERDICT =====================", flush=True)
    print(f"  VERDICT: {summary['verdict']}", flush=True)
    print(f"  anchor local wall_tps = {_f(summary['anchor_local_wall_tps'])} (#640: 127.683)", flush=True)
    print(f"  g32 official_proj_tps = {_f(summary['g32_official_proj_tps'])}  (anchor {OFFICIAL_ANCHOR_TPS})", flush=True)
    print(f"  g32 wall_tps CV%      = {_f(summary['g32_wall_tps_cv_pct'])}", flush=True)
    print(f"  g32 strict-#319       = {summary['g32_strict_319_n']} passes={summary['g32_passes_strict_319']}", flush=True)
    print(f"  g32 PPL               = {_f(summary['g32_ppl'])} (cap {OFFICIAL_PPL_CAP}, ref {REFERENCE_PPL}) ok={summary['g32_ppl_ok']}", flush=True)
    print(f"  attribution           = {json.dumps(summary['attribution'])}", flush=True)
    print("\n  per-cell:", flush=True)
    for r in records:
        print(f"    {r['name']:16s} [{r['body']:4s}/{r['head']:11s}] "
              f"tps={_f(r.get('wall_tps')):>8} d={_f(r.get('delta_pct_vs_anchor')):>7}% "
              f"proj={_f(r.get('official_proj_tps')):>8} #319={str(r.get('passes_strict_319')):>5} "
              f"ppl={_f(r.get('ppl')):>7} ready={_f(r.get('ready_s'))}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
