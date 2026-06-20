#!/usr/bin/env python
"""Serve ONE body-group-size checkpoint through the EXACT int4head fire-candidate
path and measure single-stream decode TPS + official 128-prompt PPL (PR #814).

The serve path is the shipped ``submissions/int4_mtp_bi0_int4head`` submission
(its serve.py wires the gemma4_assistant MTP drafter at NUM_SPECULATIVE_TOKENS=6,
the surgical force-2D / attn-group patches via PYTHONPATH, VLLM_BATCH_INVARIANT=0)
with ONE override: ``MODEL_ID`` points at the body-group-size checkpoint under
test. So the only variable vs int4head (256.74 TPS / PPL 2.0029) is the body
weight group_size baked into the checkpoint.

LOCAL single-A10G only. No HF Job, no submission, no leaderboard change.
"""
from __future__ import annotations

import argparse
import json
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

INT4HEAD_DIR = ROOT / "submissions" / "int4_mtp_bi0_int4head"
SERVED_NAME = "gemma-4-e4b-it"
REFERENCE_PPL = 2.42  # official cap
INT4HEAD_TPS = 256.74
INT4HEAD_PPL = 2.0029


def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _scrape_spec_metrics(base_url: str) -> dict[str, float]:
    """Best-effort vLLM spec-decode acceptance metrics from /metrics."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in ("spec_decode_num_accepted_tokens", "spec_decode_num_draft_tokens",
                    "spec_decode_num_emitted_tokens", "spec_decode_draft_acceptance_rate",
                    "spec_decode_efficiency"):
            if key in line:
                try:
                    out[key] = out.get(key, 0.0) + float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
    return out


def _sample_text(out_file: Path, runner_python: Path, n_chars: int = 240) -> str:
    """Decode the first captured completion's token ids for a sanity check."""
    try:
        first = json.loads(out_file.read_text().splitlines()[0])
        ids = first.get("completion_token_ids") or first.get("token_ids")
        if not ids:
            return "<no token ids>"
        code = (
            "import sys,json;from transformers import AutoTokenizer;"
            f"t=AutoTokenizer.from_pretrained({paths.TOKENIZER!r});"
            "print(t.decode(json.load(sys.stdin)['ids'])[:%d])" % n_chars
        )
        p = subprocess.run([str(runner_python), "-c", code],
                           input=json.dumps({"ids": ids[:64]}), capture_output=True,
                           text=True, timeout=120)
        return (p.stdout or p.stderr).strip().replace("\n", " ")[:n_chars]
    except Exception as exc:
        return f"<decode failed: {exc}>"


def measure(args) -> dict[str, Any]:
    ckpt = Path(args.ckpt).resolve()
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    out_dir = Path(args.work_dir or (ROOT / "research" / "bi0_body_groupsize" / "runs" / args.arm))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = harness.load_manifest(INT4HEAD_DIR)
    server_py = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[measure] arm={args.arm} ckpt={ckpt}", flush=True)
    print(f"[measure] server_python={server_py}", flush=True)

    rec: dict[str, Any] = {
        "arm": args.arm, "ckpt": str(ckpt), "body_group_size": args.body_group_size,
        "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "served_ok": False, "error": None,
        "num_prompts": args.num_prompts, "output_len": args.output_len,
        "warmups": args.warmups, "reps": args.reps, "seed": args.seed,
        "rep_wall_tps": [], "wall_tps_mean": None, "wall_tps_std": None,
        "wall_tps_median": None, "wall_tps_cv_pct": None,
        "ppl": None, "ppl_error": None, "sample_text": None,
        "gpu_mem_used_mib": None, "ready_s": None, "spec_metrics": {},
    }
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = out_dir / "server.log"
    extra_env = {"MODEL_ID": str(ckpt)}
    try:
        t0 = time.time()
        with harness.LocalServer(
            INT4HEAD_DIR, server_python=server_py, port=args.port,
            log_path=log_path, extra_env=extra_env,
        ) as srv:
            rec["ready_s"] = round(time.time() - t0, 1)
            rec["served_ok"] = True
            rec["model_id"] = srv.model_id

            for w in range(args.warmups):
                harness.capture_decode(
                    server_py, base_url=base_url, model=SERVED_NAME,
                    out_file=out_dir / f"warm{w}.jsonl",
                    summary_file=out_dir / f"warm{w}.summary.json",
                    num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
                )
            rec["gpu_mem_used_mib"] = _gpu_mem_used_mib()

            for r in range(args.reps):
                of = out_dir / f"rep{r}.jsonl"
                s = harness.capture_decode(
                    server_py, base_url=base_url, model=SERVED_NAME,
                    out_file=of, summary_file=out_dir / f"rep{r}.summary.json",
                    num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
                )
                n = int(s.get("num_completion_tokens", 0))
                d = float(s.get("duration_s", 0.0))
                tps = n / d if d > 0 else float("nan")
                rec["rep_wall_tps"].append(tps)
                if r == 0:
                    rec["sample_text"] = _sample_text(of, server_py)
                    rec["rep0_completion_tokens"] = n
            vals = [v for v in rec["rep_wall_tps"] if v == v]
            if vals:
                rec["wall_tps_mean"] = statistics.fmean(vals)
                rec["wall_tps_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                rec["wall_tps_median"] = statistics.median(vals)
                rec["wall_tps_cv_pct"] = (100.0 * rec["wall_tps_std"] / rec["wall_tps_mean"]
                                          if rec["wall_tps_mean"] else None)
            rec["spec_metrics"] = _scrape_spec_metrics(base_url)

            if not args.no_ppl:
                try:
                    ppl = harness.run_ppl(
                        server_py, base_url=base_url, model=SERVED_NAME,
                        out_file=out_dir / "ppl.jsonl",
                        summary_file=out_dir / "ppl.summary.json",
                    )
                    rec["ppl"] = ppl.get("ppl")
                except Exception as exc:
                    rec["ppl_error"] = str(exc)
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"[measure] ERROR: {exc}", flush=True)

    rec["t_end_utc"] = datetime.now(timezone.utc).isoformat()
    if rec["wall_tps_mean"]:
        rec["delta_pct_vs_int4head"] = round(100.0 * (rec["wall_tps_mean"] - INT4HEAD_TPS) / INT4HEAD_TPS, 3)
    return rec


def _log_wandb(args, rec: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[measure] wandb import failed ({exc}); skipping", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="bi0-body-groupsize", agent="wirbel",
        name=args.wandb_name or f"wirbel/body-groupsize-{args.arm}",
        group=args.wandb_group,
        tags=["bi0-body-groupsize", "analysis-only", args.arm],
        config={
            "analysis_only": True, "official_tps": 0,
            "arm": args.arm, "body_group_size": args.body_group_size,
            "head_group_size": 32, "int4head_tps": INT4HEAD_TPS, "int4head_ppl": INT4HEAD_PPL,
            "official_ppl_cap": REFERENCE_PPL,
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "warmups": args.warmups, "reps": args.reps, "seed": args.seed,
        },
    )
    if run is None:
        print("[measure] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = None
    try:
        run_id = run.id
        metrics = {
            "arm/wall_tps_mean": rec.get("wall_tps_mean"),
            "arm/wall_tps_std": rec.get("wall_tps_std"),
            "arm/wall_tps_median": rec.get("wall_tps_median"),
            "arm/wall_tps_cv_pct": rec.get("wall_tps_cv_pct"),
            "arm/delta_pct_vs_int4head": rec.get("delta_pct_vs_int4head"),
            "arm/ppl": rec.get("ppl"),
            "arm/gpu_mem_used_mib": rec.get("gpu_mem_used_mib"),
            "arm/ready_s": rec.get("ready_s"),
            "arm/body_group_size": rec.get("body_group_size"),
        }
        for i, v in enumerate(rec.get("rep_wall_tps", [])):
            metrics[f"arm/rep{i}_wall_tps"] = v
        for k, v in (rec.get("spec_metrics") or {}).items():
            metrics[f"spec/{k}"] = v
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, metrics, step=0)
        wandb_logging.log_json_artifact(
            run, name=f"body_groupsize_{args.arm}",
            artifact_type="bi0-body-groupsize", data=rec,
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True, help="label e.g. g32/g64/g128")
    ap.add_argument("--body-group-size", type=int, required=True)
    ap.add_argument("--out", required=True, help="results jsonl to append to")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="bi0-int4head-body-groupsize")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[measure] {note}", flush=True)

    rec = measure(args)
    run_id = _log_wandb(args, rec)
    rec["wandb_run_id"] = run_id

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    print("\n[measure] ===== RESULT =====", flush=True)
    print(f"  arm={rec['arm']} body_gs={rec['body_group_size']} served={rec['served_ok']} "
          f"error={rec['error']}", flush=True)
    print(f"  wall_tps mean={rec['wall_tps_mean']} std={rec['wall_tps_std']} "
          f"cv={rec['wall_tps_cv_pct']}% reps={[round(v,2) for v in rec['rep_wall_tps']]}", flush=True)
    print(f"  delta_vs_int4head={rec.get('delta_pct_vs_int4head')}%  ppl={rec['ppl']} "
          f"(cap {REFERENCE_PPL})", flush=True)
    print(f"  ready={rec['ready_s']}s mem={rec['gpu_mem_used_mib']}MiB wandb={run_id}", flush=True)
    print(f"  sample_text: {rec.get('sample_text')}", flush=True)
    print(f"  spec_metrics: {rec.get('spec_metrics')}", flush=True)
    return 0 if rec["served_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
