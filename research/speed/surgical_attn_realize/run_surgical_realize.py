"""PR #488 — Surgical attention-only strict realization: real rung above 222, or mirage?

LOCAL MEASUREMENT ONLY. ``analysis_only=true``, ``official_tps=0``. No HF job, no
submission, no served-file change to the deployed stack. A LOCAL serve-venv vLLM
prototype edit (``triton_unified_attention.py`` honoring ``SURGICAL_ATTN_USE_3D_OFF=1``)
isolates the lever; it is backed up and restored after measurement.

The question
------------
The shipped strict config (``VLLM_BATCH_INVARIANT=1``) collapses 481->~222 TPS because
on this A10G (sm_86 -> SM80 family) the flag does TWO independent things:

  1. installs ``matmul_persistent`` Triton overrides on every aten::mm/addmm/linear
     (~48% tax -- this is what kills 481->222), AND
  2. forces the 7 full_attention reductions onto the order-preserving 2D single-segment
     sequential-KV path (``use_3d=False``) -- the *only* part that buys byte-exact
     greedy identity (M=8 verify == own M=1 AR).

Item 2 is config-reachable WITHOUT item 1: set the attention module's
``is_batch_invariant=True`` while leaving ``envs.VLLM_BATCH_INVARIANT`` unset, so
``init_batch_invariance()`` (gated on the env) never installs the matmul tax. The
deployed ``splitkv_verify_patch`` already keys off that same flag (``_batch_invariant()``
-> no 3D redirect), so the M=8 verify takes the byte-exact 2D path automatically.

This harness measures, single-stream, same pod, back-to-back, the real served e2e TPS of:

  (a) deployed    : no flag                       -- ~481 reference (3D split-KV, identity 0.9966)
  (b) full_flag   : VLLM_BATCH_INVARIANT=1         -- the ~222 ship floor to beat
  (c) surgical    : SURGICAL_ATTN_USE_3D_OFF=1     -- attn-only 2D, matmul tax OFF  <-- THE TEST

Per arm: median ``wall_tps`` over N back-to-back decodes (the robust, official-spec
metric = num_completion_tokens / decode duration_s), PPL, 128/128 completion, peak GPU
mem, and a server-log mechanism check (3D-redirect count, ONEGRAPH capture, no traceback).
Round-0 token IDs are kept for a cross-arm served token diff (C vs B must be byte-identical;
C vs A reveals the deployed flips). token_identity_rate vs M=1 AR is measured separately by
the eager locus census (deployed_flip_attribution.py) -- reduction order is graph-invariant.

Run under the repo .venv (has wandb); serve/decode subprocs use the submission serve venv::

    .venv/bin/python -m research.speed.surgical_attn_realize.run_surgical_realize \
        --n-decodes 3 --wandb-name lawine/surgical-realize \
        --wandb-group surgical-attention-realization
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from research.tps_noise_floor.run_noise_floor import (  # noqa: E402
    preflight_gpu,
    _gpu_mem_used_mib,
)

OUT_ROOT = ROOT / "research" / "speed" / "surgical_attn_realize"

# Banked anchors (PR #488 body / stark #472 / lawine #438). Between-session hardware
# noise and the 222 floor are the bars the verdict is graded against.
SIGMA_HW = 4.864          # between-session wall_tps sigma (fresh server per arm)
MATERIALITY_TPS = 2.0     # lift over the 222 floor must clear this to count as a real rung
DEPLOYED_REF_TPS = 481.53 # PR #52 deployed FAST baseline (non-equivalent, identity 0.9966)
STRICT_FLOOR_REF = 222.0  # shipped global-strict floor (the realization floor to beat)
SURGICAL_TARGET = 457.0   # stark #466/#472 modeled isolated-locus realization (the "realizes" bucket)
PPL_GATE = 2.42

ARMS: list[dict[str, Any]] = [
    {
        "name": "deployed",
        "extra_env": {},
        "label": "(a) deployed 3D split-KV (no flag, ~481 ref)",
    },
    {
        "name": "full_flag",
        "extra_env": {"VLLM_BATCH_INVARIANT": "1"},
        "label": "(b) global VLLM_BATCH_INVARIANT=1 (~222 ship floor)",
    },
    {
        "name": "surgical",
        "extra_env": {"SURGICAL_ATTN_USE_3D_OFF": "1"},
        "label": "(c) surgical attn-only 2D order-preserving (matmul tax OFF)  <-- TEST",
    },
]


# ---------------------------------------------------------------------------
# Server-log mechanism evidence
# ---------------------------------------------------------------------------
def grep_log(log_path: Path) -> dict[str, Any]:
    """Pull the lever-mechanism signals out of one arm's server log.

    ``splitkv_redirects`` = count of "verify batch ... -> 3D split-KV" lines (the deployed
    fast path). For surgical/full_flag this MUST be 0 (the 2D order-preserving path was
    taken instead). ``graph_capture`` confirms ONEGRAPH CUDA-graph capture ran (the
    realization question: does forcing 2D break capture e2e?). ``traceback`` flags any
    crash that would invalidate the arm.
    """
    out = {
        "splitkv_armed": False,
        "splitkv_redirects": 0,
        "graph_capture_lines": 0,
        "onegraph_captured": False,
        "fatal_traceback": False,
        "n_tracebacks": 0,
        "benign_usage_tracebacks": 0,
        "batch_invariant_mentions": 0,
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return out
    # In the served subprocess the ops module is already imported, so the patch
    # logs "wrapped unified_attention" (not the import-time "armed" line).
    out["splitkv_armed"] = ("[splitkv-verify] wrapped" in text) or ("[splitkv-verify] armed" in text)
    out["splitkv_redirects"] = text.count("-> 3D split-KV")
    out["graph_capture_lines"] = text.count("Capturing CUDA graph") + text.count("Capturing cudagraph")
    # The authoritative ONEGRAPH realization signal (does forcing use_3d=False break capture?).
    out["onegraph_captured"] = "[onegraph] captured" in text
    n_tb = text.count("Traceback (most recent call last)")
    # vLLM's _report_usage_worker telemetry thread crashes on this pod (cpuinfo JSON
    # parse fails); it is identical across all arms and unrelated to the lever. Only a
    # traceback in EXCESS of that benign one (or a CUDA error) is fatal.
    n_usage = text.count("_report_usage_worker")
    out["n_tracebacks"] = n_tb
    out["benign_usage_tracebacks"] = n_usage
    out["fatal_traceback"] = ("CUDA error" in text) or (n_tb > n_usage)
    low = text.lower()
    out["batch_invariant_mentions"] = low.count("batch_invariant") + low.count("batch-invariant")
    return out


# ---------------------------------------------------------------------------
# One arm: fresh server, N back-to-back decodes (median wall_tps), one PPL pass
# ---------------------------------------------------------------------------
def run_arm(
    arm: dict[str, Any],
    submission_dir: Path,
    server_python: Path,
    out_dir: Path,
    *,
    n_decodes: int,
    num_prompts: int,
    output_len: int,
    seed: int,
    do_ppl: bool,
    records_fh,
) -> dict[str, Any]:
    name = arm["name"]
    arm_dir = out_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_log = arm_dir / "server.log"
    print(f"\n[surgical] ===== ARM {name} :: {arm['label']} =====", flush=True)
    print(f"[surgical] extra_env={arm['extra_env']}", flush=True)

    preflight_gpu()
    decodes: list[dict[str, Any]] = []
    peak_mem_mib = 0
    server_ready_s = None
    ppl_summary: dict[str, Any] | None = None
    first_decode_out: Path | None = None

    t_load0 = time.time()
    with harness.LocalServer(
        submission_dir,
        server_python=server_python,
        log_path=server_log,
        extra_env=arm["extra_env"],
    ) as server:
        server_ready_s = time.time() - t_load0
        print(f"[surgical] {name}: server ready in {server_ready_s:.0f}s", flush=True)
        m = _gpu_mem_used_mib()
        if m:
            peak_mem_mib = max(peak_mem_mib, m)

        for i in range(n_decodes):
            decode_out = arm_dir / f"decode_round{i:02d}.jsonl"
            decode_summary = arm_dir / f"decode_round{i:02d}.summary.json"
            if i == 0:
                first_decode_out = decode_out
            t0 = time.time()
            summary = harness.capture_decode(
                server_python,
                base_url=server.base_url,
                model=server.served_model_name,
                out_file=decode_out,
                summary_file=decode_summary,
                num_prompts=num_prompts,
                output_len=output_len,
                seed=seed,
            )
            wall_around = time.time() - t0
            n_tok = int(summary.get("num_completion_tokens", 0))
            dur = float(summary.get("duration_s", wall_around))
            wall_tps = n_tok / dur if dur > 0 else float("nan")
            n_completed = int(summary.get("num_records", 0))
            rec = {
                "arm": name,
                "round": i,
                "wall_tps": wall_tps,
                "num_completion_tokens": n_tok,
                "decode_duration_s": dur,
                "wall_around_decode_s": wall_around,
                "num_completed_prompts": n_completed,
                "expected_tokens": num_prompts * output_len,
            }
            decodes.append(rec)
            print(
                f"[surgical] {name} round {i}: wall_tps={wall_tps:.2f} "
                f"tok={n_tok}/{num_prompts * output_len} dur={dur:.1f}s "
                f"completed={n_completed}",
                flush=True,
            )
            mm = _gpu_mem_used_mib()
            if mm:
                peak_mem_mib = max(peak_mem_mib, mm)

        if do_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    server_python,
                    base_url=server.base_url,
                    model=server.served_model_name,
                    out_file=arm_dir / "ppl.jsonl",
                    summary_file=arm_dir / "ppl.summary.json",
                )
                print(
                    f"[surgical] {name}: PPL={ppl_summary.get('ppl')} "
                    f"records={ppl_summary.get('num_records')}",
                    flush=True,
                )
            except Exception as exc:  # PPL must never discard the timing data
                print(f"[surgical] {name}: WARN PPL failed: {exc}", flush=True)

    mech = grep_log(server_log)
    wall_tps_vals = [d["wall_tps"] for d in decodes if d["wall_tps"] == d["wall_tps"]]
    median_tps = statistics.median(wall_tps_vals) if wall_tps_vals else float("nan")
    arm_rec = {
        "arm": name,
        "label": arm["label"],
        "extra_env": arm["extra_env"],
        "median_wall_tps": median_tps,
        "wall_tps_values": wall_tps_vals,
        "wall_tps_n": len(wall_tps_vals),
        "wall_tps_std": statistics.stdev(wall_tps_vals) if len(wall_tps_vals) > 1 else 0.0,
        "server_ready_s": server_ready_s,
        "peak_gpu_mem_mib": peak_mem_mib,
        "ppl": (ppl_summary or {}).get("ppl"),
        "ppl_num_records": (ppl_summary or {}).get("num_records"),
        "num_completed_prompts": decodes[0]["num_completed_prompts"] if decodes else None,
        "completion_full": bool(decodes and decodes[0]["num_completion_tokens"] == num_prompts * output_len),
        "mechanism": mech,
        "first_decode_out": str(first_decode_out) if first_decode_out else None,
        "decodes": decodes,
    }
    records_fh.write(json.dumps(arm_rec) + "\n")
    records_fh.flush()
    _print_arm(arm_rec)
    return arm_rec


def _print_arm(rec: dict[str, Any]) -> None:
    mech = rec.get("mechanism") or {}
    print(
        f"[surgical] ARM {rec['arm']} SUMMARY: median_wall_tps={rec['median_wall_tps']:.2f} "
        f"(n={rec['wall_tps_n']}, std={rec['wall_tps_std']:.2f}) PPL={rec['ppl']} "
        f"completed={rec['num_completed_prompts']} full={rec['completion_full']} "
        f"peak_mem={rec['peak_gpu_mem_mib']}MiB | "
        f"splitkv_redirects={mech.get('splitkv_redirects')} "
        f"onegraph_captured={mech.get('onegraph_captured')} fatal_traceback={mech.get('fatal_traceback')}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Cross-arm served token diff (corroboration, not the headline identity)
# ---------------------------------------------------------------------------
def _load_token_seqs(path: Path | None) -> dict[str, list[int]] | None:
    """Map prompt-id -> generated token-id list from a decode_outputs.jsonl."""
    if not path or not Path(path).exists():
        return None
    seqs: dict[str, list[int]] = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj.get("id", obj.get("dataset_index", obj.get("index", len(seqs)))))
            toks = obj.get("completion_token_ids")
            if isinstance(toks, list):
                seqs[key] = [int(t) for t in toks]
    except Exception as exc:  # noqa: BLE001
        print(f"[surgical] token-seq load failed for {path}: {exc}", flush=True)
        return None
    return seqs or None


def cross_arm_token_diff(a: Path | None, b: Path | None, label: str) -> dict[str, Any]:
    """Per-token identity between two served arms' round-0 outputs."""
    sa, sb = _load_token_seqs(a), _load_token_seqs(b)
    if not sa or not sb:
        return {"label": label, "available": False}
    common = sorted(set(sa) & set(sb))
    total = 0
    matched = 0
    n_flipped_seqs = 0
    first_div = []
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        seq_flips = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - seq_flips
        if seq_flips or len(ta) != len(tb):
            n_flipped_seqs += 1
            for i in range(n):
                if ta[i] != tb[i]:
                    first_div.append({"prompt": k, "pos": i, "a": ta[i], "b": tb[i]})
                    break
    return {
        "label": label,
        "available": True,
        "n_prompts_compared": len(common),
        "n_tokens_compared": total,
        "n_tokens_matched": matched,
        "token_identity_rate": (matched / total) if total else None,
        "n_sequences_with_any_flip": n_flipped_seqs,
        "first_divergences": first_div[:10],
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def build_verdict(arm_recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dep = arm_recs.get("deployed", {})
    full = arm_recs.get("full_flag", {})
    surg = arm_recs.get("surgical", {})
    surg_tps = surg.get("median_wall_tps")
    full_tps = full.get("median_wall_tps")
    dep_tps = dep.get("median_wall_tps")

    lift_vs_222 = None
    if isinstance(surg_tps, (int, float)) and isinstance(full_tps, (int, float)):
        lift_vs_222 = surg_tps - full_tps

    # Recovery fraction relative to the SAME-SESSION deployed (A) and 222 floor (B) — basis-independent
    # (local deployed wall_tps ~450 differs from the 481.53 banked anchor, so absolute 457-thresholds
    # would mis-bucket; the recovery fraction asks "how much of the 222-collapse does surgical undo?").
    recovery = None
    if all(isinstance(x, (int, float)) for x in (surg_tps, full_tps, dep_tps)) and (dep_tps - full_tps) > 0:
        recovery = (surg_tps - full_tps) / (dep_tps - full_tps)

    landing = "unknown"
    if recovery is not None:
        if recovery >= 0.80:
            landing = "realizes_near_deployed"   # surgical ~ deployed minus only the small attn tax
        elif recovery >= 0.30:
            landing = "partial"
        else:
            landing = "collapses_~222"            # matmul tax was not the cause -> mirage

    lift_clears = bool(
        isinstance(lift_vs_222, (int, float)) and lift_vs_222 > max(MATERIALITY_TPS, SIGMA_HW)
    )
    realizes = bool(lift_clears and recovery is not None and recovery >= 0.30)

    return {
        "surgical_realized_tps": surg_tps,
        "full_flag_222_floor_tps": full_tps,
        "deployed_ref_tps": dep_tps,
        "surgical_lift_vs_222": lift_vs_222,
        "materiality_bar_tps": MATERIALITY_TPS,
        "sigma_hw": SIGMA_HW,
        "lift_clears_materiality_and_sigma": bool(
            isinstance(lift_vs_222, (int, float)) and lift_vs_222 > max(MATERIALITY_TPS, SIGMA_HW)
        ),
        "surgical_landing_bucket": landing,
        "surgical_realizes_above_222": realizes,
        "surgical_recovery_fraction_of_deployed": (
            (surg_tps - full_tps) / (dep_tps - full_tps)
            if all(isinstance(x, (int, float)) for x in (surg_tps, full_tps, dep_tps))
            and (dep_tps - full_tps)
            else None
        ),
        "surgical_ppl": surg.get("ppl"),
        "surgical_ppl_passes_gate": (
            isinstance(surg.get("ppl"), (int, float)) and surg["ppl"] <= PPL_GATE
        ),
        "surgical_completion_full": surg.get("completion_full"),
        "surgical_no_3d_redirect": (
            (surg.get("mechanism") or {}).get("splitkv_redirects") == 0
        ),
        "surgical_graph_capture_ok": (
            (surg.get("mechanism") or {}).get("onegraph_captured", False)
            and not (surg.get("mechanism") or {}).get("fatal_traceback", False)
        ),
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
    }


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def log_wandb(args, arm_recs: dict[str, dict[str, Any]], verdict: dict[str, Any],
              diffs: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[surgical] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="surgical-attention-realization",
            agent="lawine",
            name=args.wandb_name or "lawine/surgical-realize",
            group=args.wandb_group,
            tags=["surgical-attention-realization", "pr488", "analysis-only"],
            config={
                "n_decodes": args.n_decodes,
                "num_prompts": args.num_prompts,
                "output_len": args.output_len,
                "seed": args.seed,
                "sigma_hw": SIGMA_HW,
                "materiality_tps": MATERIALITY_TPS,
                "deployed_ref_tps": DEPLOYED_REF_TPS,
                "strict_floor_ref": STRICT_FLOOR_REF,
                "analysis_only": True,
                "official_tps": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[surgical] wandb init failed ({exc}); skipping", flush=True)
        return None
    if run is None:
        print("[surgical] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        for i, name in enumerate(["deployed", "full_flag", "surgical"]):
            rec = arm_recs.get(name)
            if not rec:
                continue
            metrics = {
                f"arm/{name}/median_wall_tps": rec.get("median_wall_tps"),
                f"arm/{name}/wall_tps_std": rec.get("wall_tps_std"),
                f"arm/{name}/ppl": rec.get("ppl"),
                f"arm/{name}/peak_gpu_mem_mib": rec.get("peak_gpu_mem_mib"),
                f"arm/{name}/num_completed_prompts": rec.get("num_completed_prompts"),
                f"arm/{name}/splitkv_redirects": (rec.get("mechanism") or {}).get("splitkv_redirects"),
                f"arm/{name}/server_ready_s": rec.get("server_ready_s"),
            }
            metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"arm_{name}", step=i, metrics=metrics)
        flat = {f"verdict/{k}": v for k, v in verdict.items() if isinstance(v, (int, float, bool))}
        wandb_logging.log_summary(run, flat, step=len(ARMS))
        wandb_logging.log_json_artifact(
            run,
            name="surgical_attn_realize",
            artifact_type="surgical-attention-realization",
            data={"arms": arm_recs, "verdict": verdict, "cross_arm_token_diffs": diffs},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[surgical] WARN wandb logging error: {exc}", flush=True)
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
    ap.add_argument("--submission", default="fa2sw_precache_kenyan")
    ap.add_argument("--arms", default="deployed,full_flag,surgical",
                    help="comma list subset of arm names to run")
    ap.add_argument("--n-decodes", type=int, default=3,
                    help="back-to-back decodes per arm (median wall_tps)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-ppl", dest="do_ppl", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny serve+decode sanity (8 prompts x 16 tok, 1 decode, no ppl)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="surgical-attention-realization")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 8)
        args.output_len = min(args.output_len, 16)
        args.n_decodes = 1
        args.do_ppl = False
        args.no_wandb = True

    for note in paths.prepare_local_gpu_env():
        print(f"[surgical] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[surgical] submission={submission_dir.name} server_python={server_python}", flush=True)

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in ARMS if a["name"] in want]
    if not arms:
        raise SystemExit(f"no arms selected from {args.arms!r}")

    out_dir = (args.out_dir or (OUT_ROOT / ("smoke" if args.smoke else "run"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "arm_records.jsonl"
    print(f"[surgical] arms={[a['name'] for a in arms]} n_decodes={args.n_decodes} "
          f"workload={args.num_prompts}x{args.output_len} seed={args.seed} -> {out_dir}", flush=True)

    t0 = time.time()
    arm_recs: dict[str, dict[str, Any]] = {}
    with open(records_path, "w") as records_fh:
        for arm in arms:
            rec = run_arm(
                arm, submission_dir, server_python, out_dir,
                n_decodes=args.n_decodes, num_prompts=args.num_prompts,
                output_len=args.output_len, seed=args.seed,
                do_ppl=args.do_ppl, records_fh=records_fh,
            )
            arm_recs[arm["name"]] = rec
    elapsed = time.time() - t0

    # Cross-arm served token diff (corroboration of byte-exactness on the served path).
    diffs: dict[str, Any] = {}
    p = {n: Path(r["first_decode_out"]) if r.get("first_decode_out") else None
         for n, r in arm_recs.items()}
    if "surgical" in arm_recs and "full_flag" in arm_recs:
        diffs["surgical_vs_full_flag"] = cross_arm_token_diff(
            p.get("surgical"), p.get("full_flag"), "surgical_vs_full_flag (expect identical)")
    if "surgical" in arm_recs and "deployed" in arm_recs:
        diffs["surgical_vs_deployed"] = cross_arm_token_diff(
            p.get("surgical"), p.get("deployed"), "surgical_vs_deployed (expect deployed flips)")

    verdict = build_verdict(arm_recs)
    try:
        from scripts import wandb_logging
        git = wandb_logging.git_info()
    except Exception:
        git = {}

    result = {
        "pr": 488,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "submission": args.submission,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "n_decodes": args.n_decodes},
        "elapsed_s": elapsed,
        "git": git,
        "arms": arm_recs,
        "cross_arm_token_diffs": diffs,
        "verdict": verdict,
    }
    run_id = None
    if not args.smoke:
        run_id = log_wandb(args, arm_recs, verdict, diffs)
    result["wandb_run_id"] = run_id
    result_path = out_dir / "surgical_realize_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    _print_final(verdict, diffs, elapsed, result_path, run_id)
    return 0


def _print_final(verdict: dict[str, Any], diffs: dict[str, Any], elapsed: float,
                 result_path: Path, run_id: str | None) -> None:
    print(f"\n[surgical] ================= VERDICT ({elapsed/60:.1f} min) =================", flush=True)
    print(f"  deployed_ref_tps         = {verdict.get('deployed_ref_tps')}", flush=True)
    print(f"  full_flag_222_floor_tps  = {verdict.get('full_flag_222_floor_tps')}", flush=True)
    print(f"  surgical_realized_tps    = {verdict.get('surgical_realized_tps')}", flush=True)
    print(f"  surgical_lift_vs_222     = {verdict.get('surgical_lift_vs_222')}  "
          f"(materiality>{MATERIALITY_TPS}, sigma_hw {SIGMA_HW})", flush=True)
    print(f"  surgical_landing_bucket  = {verdict.get('surgical_landing_bucket')}", flush=True)
    print(f"  surgical_recovery_frac   = {verdict.get('surgical_recovery_fraction_of_deployed')}", flush=True)
    print(f"  surgical_ppl             = {verdict.get('surgical_ppl')} "
          f"(gate<={PPL_GATE}: {verdict.get('surgical_ppl_passes_gate')})", flush=True)
    print(f"  surgical_no_3d_redirect  = {verdict.get('surgical_no_3d_redirect')}", flush=True)
    print(f"  surgical_graph_capture_ok= {verdict.get('surgical_graph_capture_ok')}", flush=True)
    print(f"  >>> surgical_realizes_above_222 = {verdict.get('surgical_realizes_above_222')} <<<", flush=True)
    for k, d in diffs.items():
        if d.get("available"):
            print(f"  token-diff {k}: identity_rate={d.get('token_identity_rate')} "
                  f"flips_seqs={d.get('n_sequences_with_any_flip')}/{d.get('n_prompts_compared')}", flush=True)
    print(f"[surgical] artifacts -> {result_path}  wandb_run_id={run_id}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
