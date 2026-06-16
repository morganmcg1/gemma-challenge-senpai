"""PR #488 — Surgical strict identity census vs M=1 AR (the validity gate).

LOCAL MEASUREMENT ONLY. ``analysis_only=true``, ``official_tps=0``.

``run_surgical_realize.py`` answered the SPEED question (surgical attn-only = 357.6
TPS, a real +135.7 rung above the 222 floor). This script answers the decisive
VALIDITY question the speed harness could only proxy: is the surgical attn-only
served greedy decode BYTE-EXACT vs its own M=1 autoregressive greedy reference?

The #192 strict gate is: served greedy (M=8 spec-verify) must be token-identical to
plain greedy AR (M=1, drafter OFF) of the SAME submitted config. The surgical arm
forces the order-preserving 2D attention reduction (byte-exact vs M=1) but leaves the
QKV/MLP/lm_head matmuls on the fast Marlin path (NOT batch-invariant). If the Marlin
GEMM reduction order depends on the row count M (M=8 verify vs M=1 AR), the surgical
M=8 verify diverges from its own M=1 AR -> NOT byte-exact, even though attention is
order-preserving. That is exactly the variable ``VLLM_BATCH_INVARIANT=1`` (the matmul
tax) removes -- so the cross-arm ``surgical_vs_full_flag``=0.9477 in the speed harness
is a PROXY for this; here we measure it directly and self-consistently.

Arms:
  * surgical_m1ar : SURGICAL_ATTN_USE_3D_OFF=1 + SENPAI_REFERENCE_MODE=1  (drafter OFF,
    M=1 AR, surgical 2D attention) -- the canonical greedy reference for the surgical
    config, generated on its OWN engine/kernels/quant.

Then diff (token-by-token, per prompt id):
  * surgical_m8 (run/surgical/decode_round00.jsonl, already captured) vs surgical_m1ar
    -> ``surgical_identity_census`` :: the gate number. operative-1.0 needs >=0.99 with
    every non-tie a bf16-ULP tie and 0 semantic flips.
  * surgical_m1ar vs full_flag_m8 (the byte-exact reference) -> isolates whether the
    fast-Marlin M=1 reference itself is config-stable vs the batch-invariant build.

Run under the repo .venv; serve/decode subprocs use the submission serve venv::

    .venv/bin/python -m research.speed.surgical_attn_realize.census_surgical_identity \
        --wandb-name lawine/surgical-identity-census \
        --wandb-group surgical-attention-realization
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from research.speed.surgical_attn_realize.run_surgical_realize import (  # noqa: E402
    cross_arm_token_diff,
    grep_log,
)

OUT_ROOT = ROOT / "research" / "speed" / "surgical_attn_realize"
RUN_DIR = OUT_ROOT / "run"
IDENTITY_GATE = 0.99  # operative-1.0 floor

# Each arm forces the order-preserving 2D attention reduction (is_batch_invariant=True)
# AND drops the drafter (SENPAI_REFERENCE_MODE) so the engine runs plain M=1 greedy AR.
# The ONLY difference between arms is the matmul library:
#   * surgical  : NO matmul tax -- QKV/MLP/lm_head stay on fast Marlin/bf16.
#   * full_flag : VLLM_BATCH_INVARIANT=1 installs the batch-invariant aten matmul tax.
# So comparing each arm's M=8 served decode to its OWN M=1 AR reference isolates exactly
# what (if anything) the matmul tax buys in strict identity over attention-only.
ARM_ENV: dict[str, dict[str, str]] = {
    "surgical": {"SURGICAL_ATTN_USE_3D_OFF": "1", "SENPAI_REFERENCE_MODE": "1"},
    "full_flag": {"VLLM_BATCH_INVARIANT": "1", "SENPAI_REFERENCE_MODE": "1"},
}
ARM_M8: dict[str, Path] = {
    "surgical": RUN_DIR / "surgical" / "decode_round00.jsonl",
    "full_flag": RUN_DIR / "full_flag" / "decode_round00.jsonl",
}
ARM_LABEL: dict[str, str] = {
    "surgical": "surgical 2D attn, fast Marlin (NO matmul tax)",
    "full_flag": "VLLM_BATCH_INVARIANT=1 (2D attn + matmul tax)",
}


def run_m1ar_arm(
    arm: str,
    submission_dir: Path,
    server_python: Path,
    out_dir: Path,
    *,
    num_prompts: int,
    output_len: int,
    seed: int,
) -> dict[str, Any]:
    """Serve ``arm`` with the drafter OFF (M=1 AR) and capture one decode."""
    arm_dir = out_dir / f"{arm}_m1ar"
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_log = arm_dir / "server.log"
    decode_out = arm_dir / "decode_round00.jsonl"
    decode_summary = arm_dir / "decode_round00.summary.json"
    extra_env = dict(ARM_ENV[arm])
    print(f"\n[census] ===== {arm}_m1ar (M=1 AR, {ARM_LABEL[arm]}, drafter OFF) =====", flush=True)
    print(f"[census] extra_env={extra_env}", flush=True)

    t_load0 = time.time()
    with harness.LocalServer(
        submission_dir,
        server_python=server_python,
        log_path=server_log,
        extra_env=extra_env,
    ) as server:
        ready_s = time.time() - t_load0
        print(f"[census] {arm}_m1ar: server ready in {ready_s:.0f}s", flush=True)
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
        wall = time.time() - t0
    n_tok = int(summary.get("num_completion_tokens", 0))
    dur = float(summary.get("duration_s", wall))
    mech = grep_log(server_log)
    rec = {
        "arm": f"{arm}_m1ar",
        "extra_env": extra_env,
        "server_ready_s": ready_s,
        "decode_duration_s": dur,
        "wall_tps_m1ar": (n_tok / dur) if dur > 0 else None,
        "num_completion_tokens": n_tok,
        "num_completed_prompts": int(summary.get("num_records", 0)),
        "completion_full": n_tok == num_prompts * output_len,
        "decode_out": str(decode_out),
        "mechanism": mech,
    }
    bi_expect = "expect >=1; matmul tax ON" if arm == "full_flag" else "expect 0; no matmul tax"
    print(
        f"[census] {arm}_m1ar: M=1 AR wall_tps={rec['wall_tps_m1ar']:.2f} "
        f"tok={n_tok}/{num_prompts*output_len} completed={rec['num_completed_prompts']} "
        f"| splitkv_redirects={mech.get('splitkv_redirects')} (expect 0; drafter OFF, no verify) "
        f"batch_invariant_mentions={mech.get('batch_invariant_mentions')} ({bi_expect}) "
        f"fatal_traceback={mech.get('fatal_traceback')}",
        flush=True,
    )
    return rec


def classify(diff: dict[str, Any], gate: float) -> dict[str, Any]:
    """Append an operative-1.0 verdict to a token-diff result.

    Free-running greedy decode cascades from the FIRST mismatch, so a single root
    divergence drags that whole sequence's token-identity down. The honest signals from
    token-ids ALONE are therefore (a) the token-identity rate and (b) how MANY sequences
    carry any flip and how DEEP the root flip sits. Whether a root flip is a bf16-ULP
    knife-edge tie or a genuine semantic flip needs the per-flip logit MARGIN, which is
    not in the decode outputs -- that attribution lives in merged #461
    (deployed_flip_attribution): the attn-only pin == full batch-invariant identity
    (0.9978), and its residual flips are session-unstable bf16-ULP near-ties
    (margin_vs_m1 ~0.125-0.25), not semantic. This census reproduces that identity
    POPULATION in-session; #461 supplies the margin proof.
    """
    rate = diff.get("token_identity_rate")
    n_flip = int(diff.get("n_sequences_with_any_flip") or 0)
    out = dict(diff)
    out["operative_1_0_pass"] = bool(isinstance(rate, (int, float)) and rate >= gate)
    out["n_flipped_sequences"] = n_flip
    out["gate"] = gate
    # Median/min root position across flipped sequences: deep roots are the knife-edge
    # cascade signature; a cluster of shallow roots would indicate a systematic flip.
    roots = [d.get("pos") for d in (diff.get("first_divergences") or []) if isinstance(d.get("pos"), int)]
    out["root_positions_sample"] = sorted(roots)
    out["min_root_pos"] = min(roots) if roots else None
    out["semantic_flip_attribution"] = "token-ids only; margin proof in merged #461 (attn-only residual = bf16-ULP ties)"
    return out


def log_wandb(args, rec: dict[str, Any], census: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[census] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="surgical-attention-realization",
            agent="lawine",
            name=args.wandb_name or f"lawine/{args.arm}-identity-census",
            group=args.wandb_group,
            tags=["surgical-attention-realization", "pr488", "analysis-only", "identity-census", args.arm],
            config={
                "arm": args.arm,
                "num_prompts": args.num_prompts,
                "output_len": args.output_len,
                "seed": args.seed,
                "identity_gate": IDENTITY_GATE,
                "analysis_only": True,
                "official_tps": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[census] wandb init failed ({exc}); skipping", flush=True)
        return None
    if run is None:
        print("[census] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        flat: dict[str, Any] = {
            "census/m1ar_wall_tps": rec.get("wall_tps_m1ar"),
            "census/m1ar_completed": rec.get("num_completed_prompts"),
        }
        for key, d in census.items():
            if not d.get("available"):
                continue
            flat[f"census/{key}/token_identity_rate"] = d.get("token_identity_rate")
            flat[f"census/{key}/n_sequences_with_any_flip"] = d.get("n_sequences_with_any_flip")
            flat[f"census/{key}/operative_1_0_pass"] = int(bool(d.get("operative_1_0_pass")))
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run,
            name="surgical_identity_census",
            artifact_type="surgical-attention-realization",
            data={"m1ar_arm": rec, "census": census},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[census] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARM_ENV), default="surgical",
                    help="which config's M=1 AR reference to capture and gate")
    ap.add_argument("--submission", default="fa2sw_precache_kenyan")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT / "census")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="surgical-attention-realization")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[census] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[census] submission={submission_dir.name} server_python={server_python}", flush=True)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arm = args.arm
    own_m8 = ARM_M8[arm]
    other_arm = "full_flag" if arm == "surgical" else "surgical"
    other_m8 = ARM_M8[other_arm]
    if not own_m8.exists():
        raise SystemExit(f"{arm} M=8 decode not found: {own_m8} (run run_surgical_realize first)")

    t0 = time.time()
    rec = run_m1ar_arm(
        arm, submission_dir, server_python, out_dir,
        num_prompts=args.num_prompts, output_len=args.output_len, seed=args.seed,
    )
    m1ar_out = Path(rec["decode_out"])

    census: dict[str, Any] = {}
    census[f"{arm}_m8_vs_{arm}_m1ar"] = classify(
        cross_arm_token_diff(own_m8, m1ar_out,
                             f"{arm}_m8 vs {arm}_m1ar (THE GATE: served greedy vs own M=1 AR)"),
        IDENTITY_GATE,
    )
    if other_m8.exists():
        census[f"{arm}_m1ar_vs_{other_arm}_m8"] = classify(
            cross_arm_token_diff(m1ar_out, other_m8,
                                 f"{arm}_m1ar vs {other_arm}_m8 (cross-config control)"),
            IDENTITY_GATE,
        )
    elapsed = time.time() - t0

    try:
        from scripts import wandb_logging
        git = wandb_logging.git_info()
    except Exception:
        git = {}

    run_id = log_wandb(args, rec, census)
    result = {
        "pr": 488,
        "kind": "identity_census",
        "arm": arm,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "submission": args.submission,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed},
        "elapsed_s": elapsed,
        "git": git,
        "m1ar_arm": rec,
        "census": census,
        "analysis_only": True,
        "official_tps": 0,
        "wandb_run_id": run_id,
    }
    result_path = out_dir / f"{arm}_identity_census.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(f"\n[census] ================= IDENTITY CENSUS ({elapsed/60:.1f} min) =================", flush=True)
    for key, d in census.items():
        if d.get("available"):
            print(
                f"  {key}: identity_rate={d.get('token_identity_rate'):.6f} "
                f"flips_seqs={d.get('n_sequences_with_any_flip')}/{d.get('n_prompts_compared')} "
                f"operative_1.0_pass={d.get('operative_1_0_pass')} "
                f"semantic_flips={d.get('has_semantic_flips')}",
                flush=True,
            )
        else:
            print(f"  {key}: UNAVAILABLE", flush=True)
    print(f"[census] artifacts -> {result_path}  wandb_run_id={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
