#!/usr/bin/env python
"""Serve-config TPS sweep for the locked int4_g128_lmhead submission (PR #640).

ANALYSIS-ONLY, LOCAL single-A10G. No weights changed, no HF jobs, no submissions.

Measures **config-0** (the exact shipped manifest serve flags) as the in-harness
``wall_tps`` anchor, then sweeps **Arm-A FREE** serve knobs one-at-a-time vs
config-0. For every arm the byte-exact greedy identity vs config-0 is checked for
free from the decode capture (the per-prompt ``completion_token_sha256`` the
official ``decode_outputs.py`` already writes) -- so a knob that *secretly*
perturbs numerics (e.g. a prefill-chunk change) is auto-detected and demoted to
Arm-B rather than mistaken for a free win.

Local -> official projection is a simple ratio anchored on the locked rung::

    official_proj = OFFICIAL_ANCHOR_TPS * wall_tps_arm / wall_tps_config0

NOT the spec-frontier projection module (that anchors on 481.53 and assumes a
drafter; this rung is pure autoregressive, ``speculative_config=None``).

A byte-exact arm preserves strict-#319 AND PPL (2.019) BY CONSTRUCTION: identical
output token ids -> identical greedy identity -> identical teacher-forced PPL. So
an Arm-A win needs no separate PPL pass; only an Arm-B numerics arm must re-earn
both. (``--ppl-anchor`` optionally runs one PPL pass on config-0 to confirm the
local stack reproduces 2.019.)

Run under the repo .venv; the served subprocess + decode use the deps-keyed serve
venv (with the fastapi<0.116 HTTP-compat fix applied). Example::

    .venv/bin/python -m scripts.profiler.serveconfig_tps_sweep \
        --num-prompts 8 --warmups 1 --reps 3 \
        --wandb-name wirbel/serveconfig-sweep --wandb-group locked-ar-serveconfig-sweep
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
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

# --- locked rung facts (PR #640) -------------------------------------------
CKPT = "/workspace/gemma_build/int4_g128_lmhead"
SERVED_NAME = "gemma-4-e4b-it"
OFFICIAL_ANCHOR_TPS = 126.378  # locked int4_g128_lmhead official output_throughput
REFERENCE_PPL = 2.019
DEPS = ["vllm==0.22.0", "transformers==5.9.0"]
OUT_ROOT = ROOT / "research" / "serveconfig_sweep"

# Host-only env shared by every config (mirrors the manifest + container shims).
BASE_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# Free-win gate: a win-candidate must beat config-0 by at least this much
# (local wall_tps) to count as a real free win rather than measurement noise.
WIN_THRESHOLD_PCT = 2.0


def base_flags(port: int) -> dict[str, Any]:
    """The exact shipped serve.py command line (config-0). Order preserved."""
    return {
        "--model": CKPT,
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


# Arm registry. ``patch`` adds/replaces flags on a copy of base_flags; ``drop``
# removes; ``env`` overrides BASE_ENV. ``arm_class``:
#   anchor           - config-0, the #319 reference + projection denominator
#   bound            - enforce_eager: a SLOWER lower bound that quantifies how
#                      much config-0's CUDA graphs + compile already save
#   win_candidate    - a FREE knob expected byte-exact (preserves #319/PPL)
#   numerics_detector- a knob that likely perturbs numerics; included to PROVE
#                      the in-loop #319 check flags it (then it would be Arm-B)
ARMS: list[dict[str, Any]] = [
    {"name": "config-0", "arm_class": "anchor", "patch": {}, "drop": [], "env": {},
     "note": "exact shipped manifest serve flags (inductor VLLM_COMPILE + FULL_AND_PIECEWISE cudagraph + prefix/chunked-prefill)"},
    {"name": "enforce_eager", "arm_class": "bound", "patch": {"--enforce-eager": None}, "drop": [], "env": {},
     "note": "disable cudagraph+compile; SLOWER bound -> quantifies the graph/compile savings already baked into config-0"},
    {"name": "max_num_seqs_1", "arm_class": "win_candidate", "patch": {"--max-num-seqs": "1"}, "drop": [], "env": {},
     "note": "match single-stream exactly (default ~256); free -- scheduler/batch bookkeeping only, no numerics"},
    {"name": "block_size_32", "arm_class": "win_candidate", "patch": {"--block-size": "32"}, "drop": [], "env": {},
     "note": "larger KV paging block (default 16); free -- same attention math, fewer page-table ops"},
    {"name": "no_prefix_caching", "arm_class": "win_candidate", "patch": {"--no-enable-prefix-caching": None}, "drop": [], "env": {},
     "note": "drop prefix-cache hashing (distinct prompts never reuse); free -- decode path unchanged"},
    {"name": "cudagraph_sizes_min", "arm_class": "win_candidate", "patch": {"--cudagraph-capture-sizes": [1, 2, 4, 8]}, "drop": [], "env": {},
     "note": "tighter cudagraph capture set (default ladder 1..512); free -- M=1 replays only the bs=1 graph; probes 'tighter capture set' (O3==O2 so no -O headroom; config-0 already captures a FULL bs=1 decode graph)"},
    {"name": "max_model_len_2048", "arm_class": "win_candidate", "patch": {"--max-model-len": "2048"}, "drop": [], "env": {},
     "note": "smaller KV reservation (>= 512-out workload fits); free -- decode weight reads unchanged"},
    {"name": "flashinfer_sampler", "arm_class": "win_candidate", "patch": {}, "drop": [], "env": {"VLLM_USE_FLASHINFER_SAMPLER": "1"},
     "note": "flashinfer sampler backend; temp=0 greedy is argmax so byte-exact expected; MAY FAIL (cuRAND JIT missing in container)"},
    {"name": "mbt_2048", "arm_class": "numerics_detector", "patch": {"--max-num-batched-tokens": "2048"}, "drop": [], "env": {},
     "note": "prefill chunk size 512->2048; check_greedy_identity warns argmax near-ties CASCADE -> expect #319 FAIL (Arm-B)"},
]

# The win-candidates that drive the stop-early gate (config-0 + these three).
GATE_WIN_CANDIDATES = ["max_num_seqs_1", "block_size_32", "no_prefix_caching"]


# ---------------------------------------------------------------------------
# GPU + process plumbing
# ---------------------------------------------------------------------------
def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def preflight(threshold_mib: int = 1500, timeout_s: int = 180) -> int | None:
    """Reap any lingering vLLM process and wait for VRAM to drain.

    A prior server that left an orphaned ``api_server`` / ``EngineCore`` would
    hold port 8000 (so readiness binds the STALE server) and VRAM (so the next
    load OOMs). This single-GPU pod only runs this harness, so reaping is safe.
    """
    reaped = False
    for pat in ["vllm.entrypoints.openai.api_server", "VLLM::EngineCore"]:
        r = subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        reaped = reaped or (r.returncode == 0)
    if reaped:
        time.sleep(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_mem_used_mib()
        if used is None or used < threshold_mib:
            return used
        time.sleep(3)
    return _gpu_mem_used_mib()


def _kill(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=20)
    except Exception:
        pass


def render_cmd(server_py: Path, flags: dict[str, Any]) -> list[str]:
    cmd = [str(server_py), "-m", "vllm.entrypoints.openai.api_server"]
    for k, v in flags.items():
        if v is None:
            cmd.append(k)
        elif isinstance(v, (list, tuple)):
            cmd.append(k)
            cmd.extend(str(x) for x in v)
        else:
            cmd.extend([k, str(v)])
    return cmd


def build_flags(arm: dict[str, Any], port: int) -> dict[str, Any]:
    flags = base_flags(port)
    for d in arm.get("drop", []):
        flags.pop(d, None)
    flags.update(arm.get("patch", {}))
    return flags


def serve(cmd: list[str], env: dict[str, str], log_path: Path, port: int,
          startup_timeout_s: int = 600) -> tuple[subprocess.Popen, Any]:
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=os.setsid)
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            try:
                log.flush()
            except Exception:
                pass
            raise RuntimeError(f"server exited early rc={proc.returncode} (see {log_path})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return proc, log
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(3)
    _kill(proc)
    try:
        log.close()
    except Exception:
        pass
    raise RuntimeError(f"server not ready within {startup_timeout_s}s (see {log_path})")


# ---------------------------------------------------------------------------
# #319 fingerprint from a decode capture
# ---------------------------------------------------------------------------
def fingerprint(jsonl_path: Path) -> tuple[list[str], list[int]]:
    """Ordered per-prompt (completion_token_sha256, num_completion_tokens).

    The sha list IS the byte-exact greedy-identity fingerprint: two configs
    agree token-for-token on every prompt iff their sha lists are equal.
    """
    rows: list[tuple[int, str, int]] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rows.append((int(o["index"]), str(o["completion_token_sha256"]), int(o["num_completion_tokens"])))
    rows.sort()
    return [s for _, s, _ in rows], [n for _, _, n in rows]


def compare_fp(ref: list[str] | None, cur: list[str]) -> dict[str, Any]:
    if ref is None:
        return {"n_match": None, "n_total": len(cur), "passes_strict_319": None}
    n = min(len(ref), len(cur))
    n_match = sum(1 for a, b in zip(ref, cur) if a == b)
    return {
        "n_match": n_match,
        "n_total": len(cur),
        "passes_strict_319": (len(ref) == len(cur) and n_match == len(cur)),
    }


# ---------------------------------------------------------------------------
# Measure one config: serve -> warmup -> timed reps -> kill
# ---------------------------------------------------------------------------
def measure(arm: dict[str, Any], server_py: Path, out_dir: Path, *,
            num_prompts: int, output_len: int, seed: int, warmups: int, reps: int,
            port: int, ppl_anchor: bool = False) -> dict[str, Any]:
    base = out_dir / arm["name"]
    base.mkdir(parents=True, exist_ok=True)
    flags = build_flags(arm, port)
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(arm.get("env", {}))
    cmd = render_cmd(server_py, flags)
    env_overrides = {**BASE_ENV, **arm.get("env", {})}
    (base / "cmd.txt").write_text(" ".join(cmd) + "\n\nENV_OVERRIDES:\n" +
                                  json.dumps(env_overrides, indent=2) + "\n")

    rec: dict[str, Any] = {
        "name": arm["name"], "arm_class": arm["arm_class"], "note": arm["note"],
        "flags": {k: v for k, v in flags.items()}, "env_overrides": env_overrides,
        "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "served_ok": False, "error": None,
        "rep_wall_tps": [], "wall_tps": None, "ready_s": None,
        "gpu_mem_used_mib": None, "fingerprint": None, "completion_counts": None,
        "ppl": None,
    }

    used_before = preflight()
    rec["gpu_mem_used_before_mib"] = used_before
    base_url = f"http://127.0.0.1:{port}"
    log_path = base / "server.log"
    proc = None
    log = None
    try:
        t0 = time.time()
        proc, log = serve(cmd, env, log_path, port)
        rec["ready_s"] = round(time.time() - t0, 1)
        rec["served_ok"] = True

        for w in range(warmups):
            harness.capture_decode(
                server_py, base_url=base_url, model=SERVED_NAME,
                out_file=base / f"warm{w}.jsonl", summary_file=base / f"warm{w}.summary.json",
                num_prompts=num_prompts, output_len=output_len, seed=seed,
            )
        # Serving footprint after warmup (model fully resident + graphs captured).
        rec["gpu_mem_used_mib"] = _gpu_mem_used_mib()

        for r in range(reps):
            of = base / f"rep{r}.jsonl"
            sf = base / f"rep{r}.summary.json"
            s = harness.capture_decode(
                server_py, base_url=base_url, model=SERVED_NAME,
                out_file=of, summary_file=sf,
                num_prompts=num_prompts, output_len=output_len, seed=seed,
            )
            n = int(s.get("num_completion_tokens", 0))
            d = float(s.get("duration_s", 0.0))
            tps = n / d if d > 0 else float("nan")
            rec["rep_wall_tps"].append(tps)
            if r == 0:
                fp, counts = fingerprint(of)
                rec["fingerprint"] = fp
                rec["completion_counts"] = counts
                rec["full_length"] = all(c == output_len for c in counts) if counts else False
        vals = [v for v in rec["rep_wall_tps"] if v == v]
        rec["wall_tps"] = statistics.median(vals) if vals else float("nan")
        rec["wall_tps_min"] = min(vals) if vals else None
        rec["wall_tps_max"] = max(vals) if vals else None
        rec["wall_tps_cv_pct"] = (100.0 * statistics.pstdev(vals) / statistics.fmean(vals)
                                  if len(vals) > 1 and statistics.fmean(vals) else 0.0)

        if ppl_anchor:
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
        print(f"  [arm:{arm['name']}] ERROR: {exc}", flush=True)
    finally:
        _kill(proc)
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
    return rec


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def annotate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {r["name"]: r for r in records}
    anchor = by_name.get("config-0")
    ref_fp = anchor.get("fingerprint") if anchor else None
    base_tps = anchor.get("wall_tps") if anchor else None

    for r in records:
        cmp = compare_fp(ref_fp, r.get("fingerprint") or []) if r.get("fingerprint") else \
            {"n_match": None, "n_total": None, "passes_strict_319": None}
        r.update(cmp)
        if r["name"] == "config-0":
            r["passes_strict_319"] = True  # self
        wt = r.get("wall_tps")
        if base_tps and wt and base_tps == base_tps and wt == wt:
            r["delta_pct"] = round(100.0 * (wt - base_tps) / base_tps, 3)
            r["official_proj_tps"] = round(OFFICIAL_ANCHOR_TPS * wt / base_tps, 3)
        else:
            r["delta_pct"] = None
            r["official_proj_tps"] = None

    # win-candidates that ran, stayed byte-exact, and have a delta
    wins = [r for r in records if r["arm_class"] == "win_candidate"
            and r.get("served_ok") and r.get("passes_strict_319") and r.get("delta_pct") is not None]
    best = max(wins, key=lambda r: r["wall_tps"], default=None)
    best_delta = best["delta_pct"] if best else 0.0

    # stop-early spread over {config-0} + the three gate win-candidates that ran
    gate_set = [anchor] + [by_name[n] for n in GATE_WIN_CANDIDATES if n in by_name and by_name[n].get("wall_tps")]
    gate_tps = [r["wall_tps"] for r in gate_set if r and r.get("wall_tps") and r["wall_tps"] == r["wall_tps"]]
    spread_pct = (100.0 * (max(gate_tps) - min(gate_tps)) / min(gate_tps)) if len(gate_tps) >= 2 else None

    # numerics arms that, if they re-earned #319, would be Arm-B candidates
    numerics_win = [r for r in records if r["arm_class"] in ("numerics_detector",)
                    and r.get("served_ok") and r.get("delta_pct") and r["delta_pct"] >= WIN_THRESHOLD_PCT]

    if best and best_delta >= WIN_THRESHOLD_PCT:
        verdict = "SERVECONFIG_FREE_TPS_WIN"
    elif numerics_win:
        verdict = "SERVECONFIG_NUMERICS_ONLY"
    else:
        verdict = "SERVECONFIG_FLAT"

    primary = best["wall_tps"] if best else (base_tps if base_tps else None)
    test = best["official_proj_tps"] if best else (OFFICIAL_ANCHOR_TPS if base_tps else None)
    fire_candidates = [
        {"name": r["name"], "wall_tps": r["wall_tps"], "delta_pct": r["delta_pct"],
         "official_proj_tps": r["official_proj_tps"], "passes_strict_319": r["passes_strict_319"]}
        for r in records
        if r.get("official_proj_tps") and r["official_proj_tps"] > OFFICIAL_ANCHOR_TPS
        and r.get("passes_strict_319") and r["arm_class"] in ("win_candidate", "anchor")
    ]
    return {
        "verdict": verdict,
        "config0_wall_tps": base_tps,
        "best_win_candidate": best["name"] if best else None,
        "best_arm_a_local_wall_tps": primary,
        "best_arm_a_official_proj_tps": test,
        "best_delta_pct": best_delta,
        "gate_spread_pct": spread_pct,
        "win_threshold_pct": WIN_THRESHOLD_PCT,
        "fire_candidates": fire_candidates,
    }


def gate_tripped(records: list[dict[str, Any]]) -> bool:
    """True once config-0 + all three gate win-candidates have run and no win
    candidate beats config-0 by >= threshold AND their spread is < threshold."""
    by_name = {r["name"]: r for r in records}
    if "config-0" not in by_name or not by_name["config-0"].get("wall_tps"):
        return False
    if not all(n in by_name and by_name[n].get("wall_tps") for n in GATE_WIN_CANDIDATES):
        return False
    summ = annotate(records)
    spread = summ["gate_spread_pct"]
    return (summ["best_delta_pct"] < WIN_THRESHOLD_PCT) and (spread is not None and spread < WIN_THRESHOLD_PCT)


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, records: list[dict[str, Any]], summary: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[sweep] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="serveconfig-tps-sweep", agent="wirbel",
        name=args.wandb_name or "wirbel/serveconfig-sweep",
        group=args.wandb_group,
        tags=["serveconfig-sweep", "int4_g128_lmhead", "analysis-only", summary["verdict"]],
        config={
            "submission": "int4_g128_lmhead", "checkpoint": CKPT,
            "official_anchor_tps": OFFICIAL_ANCHOR_TPS, "reference_ppl": REFERENCE_PPL,
            "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed,
            "warmups": args.warmups, "reps": args.reps, "win_threshold_pct": WIN_THRESHOLD_PCT,
            "analysis_only": True,
        },
    )
    if run is None:
        print("[sweep] wandb disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return None
    run_id = None
    try:
        run_id = run.id
        for i, r in enumerate(records):
            metrics = {
                "arm/wall_tps": r.get("wall_tps"),
                "arm/delta_pct": r.get("delta_pct"),
                "arm/official_proj_tps": r.get("official_proj_tps"),
                "arm/ready_s": r.get("ready_s"),
                "arm/gpu_mem_used_mib": r.get("gpu_mem_used_mib"),
                "arm/n_match": r.get("n_match"),
                "arm/passes_strict_319": 1 if r.get("passes_strict_319") else 0,
                "arm/wall_tps_cv_pct": r.get("wall_tps_cv_pct"),
            }
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"arm_{r['name']}", step=i, metrics=metrics,
                                    data={"arm_name": r["name"], "arm_class": r["arm_class"]})
        flat = {f"verdict/{k}": v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}
        wandb_logging.log_summary(run, flat, step=len(records))
        wandb_logging.log_json_artifact(
            run, name="serveconfig_sweep", artifact_type="serveconfig-sweep",
            data={"summary": summary, "records": records},
        )
    except Exception as exc:
        print(f"[sweep] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-prompts", type=int, default=8, help="screening workload prompts (ratio is workload-invariant)")
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-minutes", type=float, default=80.0, help="wall budget; stop before starting a config past this")
    ap.add_argument("--only", default=None, help="comma-separated arm names to run (else all)")
    ap.add_argument("--no-gate", action="store_true", help="run all arms; do not stop early on the FLAT gate")
    ap.add_argument("--ppl-anchor", action="store_true", help="run one PPL pass on config-0 to confirm local 2.019")
    ap.add_argument("--dry-run", action="store_true", help="print planned configs + commands, do not serve")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="locked-ar-serveconfig-sweep")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[sweep] {note}", flush=True)

    only = set(a.strip() for a in args.only.split(",")) if args.only else None
    plan = [a for a in ARMS if (only is None or a["name"] in only)]
    # config-0 must run first (it is the #319 reference + projection denominator)
    plan.sort(key=lambda a: 0 if a["name"] == "config-0" else 1)

    out_dir = (args.out_dir or (OUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] checkpoint={CKPT}", flush=True)
    print(f"[sweep] workload={args.num_prompts}x{args.output_len} seed={args.seed} "
          f"warmups={args.warmups} reps={args.reps} -> {out_dir}", flush=True)
    print(f"[sweep] plan: {[a['name'] for a in plan]}", flush=True)

    if args.dry_run:
        for a in plan:
            print(f"\n=== {a['name']} [{a['arm_class']}] ===\n  {a['note']}", flush=True)
            print("  " + " ".join(render_cmd(Path("<server_py>"), build_flags(a, args.port))), flush=True)
            print(f"  env: {{**BASE_ENV, {a.get('env', {})}}}", flush=True)
        return 0

    server_py = harness.ensure_server_venv(DEPS)
    print(f"[sweep] server_python={server_py}", flush=True)

    records: list[dict[str, Any]] = []
    records_path = out_dir / "records.jsonl"
    t_start = time.time()
    with open(records_path, "w") as fh:
        for a in plan:
            elapsed_min = (time.time() - t_start) / 60.0
            if a["name"] != "config-0" and elapsed_min > args.max_minutes:
                print(f"[sweep] budget {args.max_minutes}min reached ({elapsed_min:.1f}); stopping before {a['name']}", flush=True)
                break
            print(f"\n[sweep] === {a['name']} [{a['arm_class']}] (t+{elapsed_min:.1f}min) ===\n  {a['note']}", flush=True)
            rec = measure(
                a, server_py, out_dir,
                num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
                warmups=args.warmups, reps=args.reps, port=args.port,
                ppl_anchor=(args.ppl_anchor and a["name"] == "config-0"),
            )
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            _print_arm(rec, records)
            if not args.no_gate and gate_tripped(records):
                print(f"[sweep] STOP-EARLY GATE TRIPPED: config-0 + {GATE_WIN_CANDIDATES} within "
                      f"{WIN_THRESHOLD_PCT}% -> SERVECONFIG_FLAT", flush=True)
                break

    summary = annotate(records)
    summary["elapsed_min"] = round((time.time() - t_start) / 60.0, 2)
    summary["workload"] = {"num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed,
                           "warmups": args.warmups, "reps": args.reps}
    summary["n_arms_run"] = len(records)
    try:
        from scripts import wandb_logging
        summary["git"] = wandb_logging.git_info()
    except Exception:
        summary["git"] = {}

    result_path = out_dir / "summary.json"
    result_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    run_id = _log_wandb(args, records, summary)
    summary["wandb_run_id"] = run_id

    _print_summary(summary, records, out_dir)
    # rewrite with run id included
    result_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"[sweep] artifacts -> {result_path}", flush=True)
    return 0


def _print_arm(rec: dict[str, Any], records: list[dict[str, Any]]) -> None:
    summ = annotate(records)
    r = next(x for x in records if x["name"] == rec["name"])
    print(f"  wall_tps={_f(r.get('wall_tps'))} reps={[round(v,2) for v in r.get('rep_wall_tps',[])]} "
          f"cv={_f(r.get('wall_tps_cv_pct'))}% delta={_f(r.get('delta_pct'))}% "
          f"official_proj={_f(r.get('official_proj_tps'))} #319={r.get('passes_strict_319')} "
          f"(n_match={r.get('n_match')}/{r.get('n_total')}) ready={_f(r.get('ready_s'))}s "
          f"mem={r.get('gpu_mem_used_mib')}MiB", flush=True)


def _print_summary(summary: dict[str, Any], records: list[dict[str, Any]], out_dir: Path) -> None:
    print("\n[sweep] ===================== VERDICT =====================", flush=True)
    print(f"  VERDICT: {summary['verdict']}", flush=True)
    print(f"  config-0 wall_tps        = {_f(summary['config0_wall_tps'])}", flush=True)
    print(f"  best win-candidate       = {summary['best_win_candidate']}", flush=True)
    print(f"  best_arm_a_local_wall_tps= {_f(summary['best_arm_a_local_wall_tps'])}", flush=True)
    print(f"  best_arm_a_official_proj = {_f(summary['best_arm_a_official_proj_tps'])} (anchor {OFFICIAL_ANCHOR_TPS})", flush=True)
    print(f"  best_delta_pct           = {_f(summary['best_delta_pct'])}%  (win threshold {WIN_THRESHOLD_PCT}%)", flush=True)
    print(f"  gate_spread_pct          = {_f(summary['gate_spread_pct'])}%", flush=True)
    print(f"  fire_candidates          = {summary['fire_candidates']}", flush=True)
    print(f"  elapsed                  = {summary.get('elapsed_min')} min  arms_run={summary['n_arms_run']}", flush=True)
    print("\n  per-arm:", flush=True)
    for r in records:
        print(f"    {r['name']:20s} [{r['arm_class']:16s}] tps={_f(r.get('wall_tps')):>8} "
              f"d={_f(r.get('delta_pct')):>7}% proj={_f(r.get('official_proj_tps')):>8} "
              f"#319={str(r.get('passes_strict_319')):>5} ready={_f(r.get('ready_s'))}s", flush=True)


def _f(v: Any) -> str:
    if isinstance(v, (int, float)) and v == v:
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
