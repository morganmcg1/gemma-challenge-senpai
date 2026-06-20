#!/usr/bin/env python
"""PR #807 W4A8-body A/B driver: serve the SAME int4 body checkpoint twice and
toggle ONLY the body activation precision via VLLM_MARLIN_INPUT_DTYPE.

Submission ``submissions/int4_mtp_bi0_w4a8body`` bakes VLLM_MARLIN_INPUT_DTYPE=int8
(W4A8 candidate). The W4A16 control is the identical serve with that env var
forced empty (get_marlin_input_dtype -> None -> bf16 activations). Everything else
-- checkpoint (google/gemma-4-E4B-it-qat-w4a16-ct), MTP drafter, bi0 BI=0 +
force-2D patches, serve flags -- is byte-identical, so candidate-minus-control
isolates the body-W4A8 lever.

Gated, cheapest-first (mirrors the PR):

  --step 1  VIABILITY KILL-GATE. Plain int4 AR (NUM_SPECULATIVE_TOKENS=0), serve
            BOTH arms, greedy-generate a few prompts. PASS iff W4A8 serves +
            generates finite/sane text. Also greedy-compares W4A8 vs W4A16 tokens:
            if they DIFFER, int8 activations are genuinely live (not a silent
            W4A16 fallback). If W4A8 engine-init crashes / "no kernel" -> KILL.

  --step 2  PPL + SPEED (only if step 1 passed). Full MTP config (manifest K=6).
            For each arm: official 128-prompt PPL (gate <=2.42, 128/128) + a
            single-stream decode TPS probe. The PPL DELTA (W4A8 slightly worse)
            is the behavioural proof int8 is active; the TPS DELTA is the lever
            verdict (physics predicts ~0 or negative at M=1-8 decode).

Run from repo root with the server venv python (harness picks it; just use any
python that can import scripts.local_validation):
  CUDA_VISIBLE_DEVICES=0 python research/validity/w4a8_body_viability/run_w4a8_ab.py --step 1
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402
from scripts import wandb_logging  # noqa: E402


def _summ(vals: list[float]) -> dict:
    """mean/std/min/max over finite reps (PR asks for spread, not a point probe)."""
    good = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not good:
        return {"reps": vals, "n": 0}
    return {
        "mean": statistics.fmean(good),
        "std": statistics.pstdev(good) if len(good) > 1 else 0.0,
        "min": min(good),
        "max": max(good),
        "n": len(good),
        "reps": good,
    }


def _gpu_mem_used_mib() -> float | None:
    """Steady-state GPU memory proxy (vLLM reserves by util, so ~equal across arms)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, text=True, capture_output=True, timeout=30,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_w4a8body"
HERE = Path(__file__).resolve().parent
MODEL_NAME = "gemma-4-e4b-it"

# arm -> the ONLY env that differs. None UNSETS the manifest-baked var, so
# envs.VLLM_MARLIN_INPUT_DTYPE is None -> get_marlin_input_dtype()->None -> bf16
# activations (W4A16). Empty string is INVALID (env_with_choices rejects "";
# valid options are int8/fp8) and crashed engine-init in the first run.
ARM_ENV = {
    "w4a16": {"VLLM_MARLIN_INPUT_DTYPE": None},
    "w4a8": {"VLLM_MARLIN_INPUT_DTYPE": "int8"},
}

SMOKE_PROMPTS = [
    "Explain step by step how a transformer decodes one token at a time.",
    "What is the capital of France, and why is it historically significant?",
    "Write a short Python function that returns the nth Fibonacci number.",
]


def server_python() -> Path:
    manifest = harness.load_manifest(SUBMISSION)
    return harness.ensure_server_venv(manifest["dependencies"])


def step1(port: int) -> int:
    """Viability kill-gate: plain int4 AR, both arms, greedy generate + compare."""
    py = server_python()
    print(f"[step1] server_python={py}", flush=True)
    out: dict = {"step": 1, "submission": str(SUBMISSION), "arms": {}}

    for arm in ("w4a16", "w4a8"):
        extra_env = dict(ARM_ENV[arm])
        extra_env["NUM_SPECULATIVE_TOKENS"] = "0"  # plain AR: isolate the body GEMM
        log_path = HERE / f"step1_{arm}_serve.log"
        print(f"\n[step1:{arm}] extra_env={extra_env}", flush=True)
        arm_rec: dict = {"extra_env": extra_env, "served": False, "generations": []}
        try:
            with harness.LocalServer(
                SUBMISSION, server_python=py, port=port,
                log_path=log_path, extra_env=extra_env, startup_timeout_s=1200,
            ) as srv:
                arm_rec["served"] = True
                arm_rec["base_url"] = srv.base_url
                for p in SMOKE_PROMPTS:
                    resp = harness._completion(srv.base_url, MODEL_NAME, p, 64)
                    ch = (resp.get("choices") or [{}])[0]
                    text = ch.get("text", "")
                    usage = resp.get("usage") or {}
                    arm_rec["generations"].append({
                        "prompt": p,
                        "text": text,
                        "completion_tokens": usage.get("completion_tokens"),
                        "finish_reason": ch.get("finish_reason"),
                    })
                    print(f"[step1:{arm}] +{usage.get('completion_tokens')} tok :: "
                          f"{text[:120]!r}", flush=True)
        except Exception as e:  # engine-init crash / no-kernel / dispatch error
            arm_rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[step1:{arm}] FAILED {arm_rec['error']}", flush=True)
        out["arms"][arm] = arm_rec

    # Verdict
    w8 = out["arms"]["w4a8"]
    w16 = out["arms"]["w4a16"]
    w8_served = w8.get("served") and not w8.get("error")
    w8_sane = w8_served and all(
        (g.get("completion_tokens") or 0) > 0 and g.get("text", "").strip()
        for g in w8["generations"]
    )
    texts_w8 = [g["text"] for g in w8.get("generations", [])]
    texts_w16 = [g["text"] for g in w16.get("generations", [])]
    differ = (len(texts_w8) == len(texts_w16) and len(texts_w8) > 0
              and any(a != b for a, b in zip(texts_w8, texts_w16)))
    out["verdict"] = {
        "w4a8_serves": bool(w8_served),
        "w4a8_sane_text": bool(w8_sane),
        "int8_active_vs_w4a16_greedy_differs": bool(differ),
        "kill_gate": "PASS" if (w8_served and w8_sane) else "KILL",
    }
    (HERE / "step1_result.json").write_text(json.dumps(out, indent=2))
    print(f"\n[step1] VERDICT = {out['verdict']}", flush=True)
    print(f"[step1] wrote {HERE / 'step1_result.json'}", flush=True)
    return 0 if out["verdict"]["kill_gate"] == "PASS" else 2


def step2(port: int, num_prompts: int, tps_reps: int, wandb_run) -> int:
    """PPL + TPS A/B on the full MTP config (manifest K=6).

    TPS is measured over ``tps_reps`` independent probes per arm so the verdict
    carries a spread (physics predicts the W4A8 body lever is ~0 / negative at
    M=1-8 decode; the effect is small relative to A10G run-to-run noise, so a
    single point probe is not enough to call sign).
    """
    py = server_python()
    print(f"[step2] server_python={py} tps_reps={tps_reps}", flush=True)
    out: dict = {"step": 2, "submission": str(SUBMISSION), "num_prompts": num_prompts,
                 "tps_reps": tps_reps, "arms": {}}

    for arm in ("w4a16", "w4a8"):
        extra_env = dict(ARM_ENV[arm])  # MTP stays at manifest K=6
        log_path = HERE / f"step2_{arm}_serve.log"
        print(f"\n[step2:{arm}] extra_env={extra_env}", flush=True)
        arm_rec: dict = {"extra_env": extra_env}
        adir = HERE / f"step2_{arm}"
        adir.mkdir(parents=True, exist_ok=True)
        try:
            with harness.LocalServer(
                SUBMISSION, server_python=py, port=port,
                log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
            ) as srv:
                t0 = time.time()
                tps_reps_list = []
                first_probe = None
                for rep in range(tps_reps):
                    probe = harness.probe_tps(srv.base_url, MODEL_NAME, decode_tokens=256)
                    first_probe = first_probe or probe
                    v = probe.get("decode_tps_single_stream")
                    tps_reps_list.append(v)
                    print(f"[step2:{arm}] tps rep {rep + 1}/{tps_reps} = {v:.3f}", flush=True)
                tps_summary = _summ(tps_reps_list)
                arm_rec["tps"] = first_probe
                arm_rec["tps_summary"] = tps_summary
                arm_rec["gpu_mem_used_mib"] = _gpu_mem_used_mib()
                print(f"[step2:{arm}] decode_tps mean={tps_summary.get('mean'):.3f} "
                      f"std={tps_summary.get('std'):.3f} "
                      f"min={tps_summary.get('min'):.3f} max={tps_summary.get('max'):.3f}",
                      flush=True)
                ppl = harness.run_ppl(
                    py, base_url=srv.base_url, model=MODEL_NAME,
                    out_file=adir / "ppl_results.jsonl",
                    summary_file=adir / "ppl_summary.json",
                )
                arm_rec["ppl_summary"] = ppl
                arm_rec["ppl"] = ppl.get("ppl")
                arm_rec["ppl_completed"] = (
                    ppl.get("num_records") or ppl.get("num_completed") or ppl.get("completed"))
                arm_rec["elapsed_s"] = time.time() - t0
                print(f"[step2:{arm}] ppl={arm_rec['ppl']} "
                      f"completed={arm_rec['ppl_completed']}", flush=True)
                if wandb_run is not None:
                    m = {
                        f"{arm}/tps_mean": tps_summary.get("mean"),
                        f"{arm}/tps_std": tps_summary.get("std"),
                        f"{arm}/tps_min": tps_summary.get("min"),
                        f"{arm}/tps_max": tps_summary.get("max"),
                        f"{arm}/ppl": arm_rec["ppl"],
                        f"{arm}/ppl_completed": arm_rec["ppl_completed"],
                        f"{arm}/gpu_mem_used_mib": arm_rec["gpu_mem_used_mib"],
                    }
                    wandb_logging.log_event(
                        wandb_run, f"arm_{arm}", step=(1 if arm == "w4a8" else 0),
                        metrics={k: v for k, v in m.items() if v is not None},
                    )
        except Exception as e:
            arm_rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[step2:{arm}] FAILED {arm_rec['error']}", flush=True)
        out["arms"][arm] = arm_rec

    w8 = out["arms"]["w4a8"]
    w16 = out["arms"]["w4a16"]

    def g(d, *ks):
        for k in ks:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return d

    tps8 = g(w8, "tps_summary", "mean")
    tps16 = g(w16, "tps_summary", "mean")
    ppl8, ppl16 = w8.get("ppl"), w16.get("ppl")
    out["delta"] = {
        "tps_w4a16_mean": tps16, "tps_w4a8_mean": tps8,
        "tps_w4a16_std": g(w16, "tps_summary", "std"),
        "tps_w4a8_std": g(w8, "tps_summary", "std"),
        "tps_pct_change": ((tps8 - tps16) / tps16 * 100.0)
        if (tps8 and tps16) else None,
        "tps_abs_delta": (tps8 - tps16) if (tps8 and tps16) else None,
        "ppl_w4a16": ppl16, "ppl_w4a8": ppl8,
        "ppl_delta": (ppl8 - ppl16) if (ppl8 and ppl16) else None,
        "ppl_gate_2p42_w4a8": (ppl8 is not None and ppl8 <= 2.42),
        "ppl_completed_w4a8": w8.get("ppl_completed"),
        "int4head_base_tps_ref": 256.74,
    }
    (HERE / "step2_result.json").write_text(json.dumps(out, indent=2))
    if wandb_run is not None:
        wandb_logging.log_event(wandb_run, "delta", step=2,
                                metrics={k: v for k, v in out["delta"].items()
                                         if isinstance(v, (int, float))})
        for k, v in out["delta"].items():
            wandb_run.summary[f"delta/{k}"] = v
        wandb_logging.log_json_artifact(
            wandb_run, name="w4a8_step2_result", artifact_type="ab-result", data=out)
    print(f"\n[step2] DELTA = {json.dumps(out['delta'], indent=2)}", flush=True)
    print(f"[step2] wrote {HERE / 'step2_result.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2], required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--tps-reps", type=int, default=5,
                    help="independent decode-TPS probes per arm (spread)")
    ap.add_argument("--wandb-name", default="wirbel/w4a8-body-viability-step2")
    ap.add_argument("--wandb-group", default="w4a8-body-viability")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    if args.step == 1:
        return step1(args.port)

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_logging.init_wandb_run(
            job_type="w4a8-body-ab",
            agent="wirbel",
            name=args.wandb_name,
            group=args.wandb_group,
            notes=("PR #807 W4A8-body A/B: same int4 w4a16-ct checkpoint + MTP K=6, "
                   "toggle ONLY VLLM_MARLIN_INPUT_DTYPE (int8 vs unset). "
                   "TPS + 128-prompt PPL per arm."),
            tags=["w4a8", "marlin-input-dtype", "pr807"],
            config={
                "submission": str(SUBMISSION),
                "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "num_speculative_tokens": 6,
                "tps_reps": args.tps_reps,
                "num_prompts": args.num_prompts,
                "ppl_gate": 2.42,
                "int4head_base_tps_ref": 256.74,
                "control_arm": "w4a16 (VLLM_MARLIN_INPUT_DTYPE unset, bf16 acts)",
                "treatment_arm": "w4a8 (VLLM_MARLIN_INPUT_DTYPE=int8, int8 acts)",
            },
        )
        print(f"[wandb] run={'live' if wandb_run is not None else 'disabled/unavailable'}",
              flush=True)
    try:
        return step2(args.port, args.num_prompts, args.tps_reps, wandb_run)
    finally:
        wandb_logging.finish_wandb(wandb_run)


if __name__ == "__main__":
    raise SystemExit(main())
