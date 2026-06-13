#!/usr/bin/env python3
"""Verify-rollback gate driver (PR #24). LOCAL ONLY (A10G); NOT an HF Job.

Orchestrates the two real-model arms on the int4 target and assembles the
verify-rollback metrics by composition (see ``paper_notes.md``):

  ref  arm  = spec OFF (M=1 AR)   -> the verify-rollback committed output AND the
                                     per-token re-verify cost.  TPS_AR.
  cand arm  = spec ON  (K)        -> the discardable speculative proposal.  TPS_spec,
                                     and flip_rate(cand vs ref) = p.

Then:
  * verify-rollback output := ref            -> GREEDY_IDENTICAL vs ref, flip = 0.
  * rollback_rate/step      := 1-(1-p)^K     (derived from measured p; paper §2).
  * TPS_VR                  := 1/(1/TPS_AR + 1/TPS_spec)   (paper §2.2), < TPS_AR.

Reuses the proven int4 server harness (launch_server.sh / wait_ready.sh /
capture_decode.sh) and the official greedy-identity verifier. Logs one W&B run
in group ``verify-rollback-gate`` and writes a local <arm>_vr_summary.json.

Example:
  ARM=int4_VR .venvs/vllm022/bin/python research/verify_rollback/run_vr_arm.py \
      --model google/gemma-4-E4B-it-qat-w4a16-ct --K 6 --nprompts 16 \
      --mode eager --out-dir research/verify_rollback/arms \
      --wandb-group verify-rollback-gate --wandb-name int4_VR
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/workspace/senpai/target")
HARNESS = ROOT / "research" / "int4_mtp_batchinv"
VENV_PY = ROOT / ".venvs" / "vllm022" / "bin" / "python"
VERIFIER = (ROOT / "official" / "main_bucket" / "shared_resources"
            / "gemma_greedy_identity_verifier_flowian-powers")
BASE = "http://127.0.0.1:8000"
MODEL_NAME = "gemma-4-e4b-it"

sys.path.insert(0, str(HERE))
import verify_rollback_patch as vr  # noqa: E402


def gpu_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def wait_gpu_free(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if gpu_used_mib() < 800:
            return True
        time.sleep(3)
    return False


def serve_and_capture(tag, nspec, model_id, mode, nprompts, outdir, inv, probe):
    """Launch one server phase, capture decode, kill server. Returns summary dict."""
    log = outdir / f"{tag}_server.log"
    out_prefix = outdir / f"{tag}_decode"
    env = dict(os.environ)
    env.update({
        "MODEL_ID": model_id,
        "VLLM_BATCH_INVARIANT": str(inv),
        "NUM_SPECULATIVE_TOKENS": str(nspec),
        "ENFORCE_EAGER": "1" if mode == "eager" else "0",
        "MAX_MODEL_LEN": "4096",
    })
    if probe:
        env["VR_PROBE"] = "1"
        env["VR_LOG"] = str(outdir / f"{tag}_accept.jsonl")
        # HERE holds usercustomize.py, which Python auto-imports at startup for
        # every process in the server tree and which imports verify_rollback_patch
        # (the accept-step probe) when VR_PROBE=1. Behavior-preserving.
        env["PYTHONPATH"] = f"{HERE}{os.pathsep}" + env.get("PYTHONPATH", "")
    print(f"=== [{tag}] launch nspec={nspec} mode={mode} inv={inv} "
          f"model={model_id} {time.strftime('%H:%M:%S')} ===", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(
            ["setsid", "bash", str(HARNESS / "launch_server.sh")],
            stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(HARNESS))
    time.sleep(1)
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    summary = None
    try:
        rc = subprocess.call(["bash", str(HARNESS / "wait_ready.sh"), str(log), "900"])
        if rc != 0:
            print(f"[{tag}] SERVER NOT READY (rc={rc})", flush=True)
            return None
        t0 = time.time()
        rc = subprocess.call(
            ["bash", str(HARNESS / "capture_decode.sh"), BASE, MODEL_NAME,
             str(out_prefix), str(nprompts)],
            stdout=open(outdir / f"{tag}_decode.capture.log", "w"),
            stderr=subprocess.STDOUT)
        wall = time.time() - t0
        if rc != 0:
            print(f"[{tag}] DECODE FAILED (rc={rc})", flush=True)
            return None
        sfile = Path(f"{out_prefix}.summary.json")
        if sfile.exists():
            summary = json.loads(sfile.read_text())
            summary["client_wall_s"] = wall
    finally:
        if pgid:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                pass
        try:
            proc.terminate()
        except Exception:
            pass
        wait_gpu_free()
    return summary


def tps_of(summary) -> float:
    if not summary:
        return float("nan")
    d = summary.get("duration_s") or summary.get("client_wall_s")
    n = summary.get("num_completion_tokens")
    return (n / d) if (d and n) else float("nan")


def run_verifier(ref, cand):
    rep = subprocess.run(
        [str(VENV_PY), "check_greedy_identity.py", "--reference", str(ref),
         "--candidate", str(cand), "--json"],
        cwd=str(VERIFIER), capture_output=True, text=True)
    try:
        return json.loads(rep.stdout)
    except Exception:
        return {"verdict": "INCOMPARABLE", "stderr": rep.stderr[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--nprompts", type=int, default=16)
    ap.add_argument("--mode", choices=["eager", "cudagraph"], default="eager")
    ap.add_argument("--inv", type=int, default=0,
                    help="VLLM_BATCH_INVARIANT (0 = plain stack; VR needs no invariant)")
    ap.add_argument("--out-dir", default=str(HERE / "arms"))
    ap.add_argument("--arm", default=os.environ.get("ARM", "int4_VR"))
    ap.add_argument("--probe", action="store_true",
                    help="enable the behavior-preserving accept-step probe on cand")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing ref/cand decode jsonl if present")
    ap.add_argument("--wandb-group", default="verify-rollback-gate")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    arm = args.arm
    tag_ref = f"{arm}_{args.mode}_ref"
    tag_cand = f"{arm}_{args.mode}_cand"
    ref_jsonl = outdir / f"{tag_ref}_decode.jsonl"
    cand_jsonl = outdir / f"{tag_cand}_decode.jsonl"
    ref_sum = outdir / f"{tag_ref}_decode.summary.json"
    cand_sum = outdir / f"{tag_cand}_decode.summary.json"

    ref_summary = cand_summary = None
    if args.reuse and ref_jsonl.exists() and cand_jsonl.exists():
        print(f"[{arm}] reusing existing decode streams", flush=True)
        ref_summary = json.loads(ref_sum.read_text()) if ref_sum.exists() else None
        cand_summary = json.loads(cand_sum.read_text()) if cand_sum.exists() else None
    else:
        wait_gpu_free()
        ref_summary = serve_and_capture(
            tag_ref, 0, args.model, args.mode, args.nprompts, outdir, args.inv, False)
        if ref_summary is None and not ref_jsonl.exists():
            print(f"[{arm}] REF arm failed", flush=True)
            return 1
        cand_summary = serve_and_capture(
            tag_cand, args.K, args.model, args.mode, args.nprompts, outdir,
            args.inv, args.probe)
        if cand_summary is None and not cand_jsonl.exists():
            print(f"[{arm}] CAND arm failed", flush=True)
            return 1

    # --- metrics -----------------------------------------------------------
    tps_ar = tps_of(ref_summary)
    tps_spec = tps_of(cand_summary)
    tps_vr = vr.compose_tps(tps_ar, tps_spec) if (
        tps_ar == tps_ar and tps_spec == tps_spec) else float("nan")

    # p and rollback via reconstruction; also build the VR output file (== ref).
    vr_out = outdir / f"{arm}_{args.mode}_vr_decode.jsonl"
    st = vr.reconstruct_vr(str(ref_jsonl), str(cand_jsonl), args.K,
                           vr_out_path=str(vr_out),
                           cand_jsonl_for_ids=str(cand_jsonl))

    # official verifier: cand-vs-ref (the spec flip) and VR-vs-ref (must be identical).
    cand_vs_ref = run_verifier(ref_jsonl, cand_jsonl)
    vr_vs_ref = run_verifier(ref_jsonl, vr_out) if vr_out.exists() else {}

    result = {
        "arm": arm, "mode": args.mode, "model": args.model, "K": args.K,
        "nprompts": args.nprompts, "inv": args.inv,
        "tps_ar_int4_specoff": tps_ar,
        "tps_spec_int4_K": tps_spec,
        "tps_vr_composed": tps_vr,
        "flip_rate_spec_per_token": st.flip_rate_per_token,
        "flip_rate_spec_ci95": list(st.flip_ci95),
        "flip_events": st.flip_events, "geom_trials": st.geom_trials,
        "rollback_rate_per_step_derived": st.rollback_rate_per_step_derived,
        "observed_first_rollback_step_rate_lb": st.observed_first_rollback_step_rate_lb,
        "vr_flip_rate_per_token": st.vr_flip_rate_per_token,
        "cand_vs_ref_verdict": cand_vs_ref.get("verdict"),
        "cand_identical": cand_vs_ref.get("num_identical"),
        "vr_vs_ref_verdict": vr_vs_ref.get("verdict"),
        "vr_identical": vr_vs_ref.get("num_identical"),
        "num_prompts_compared": cand_vs_ref.get("num_prompts_compared"),
    }
    (outdir / f"{arm}_{args.mode}_vr_summary.json").write_text(
        json.dumps(result, indent=2))
    print("VR_RESULT " + json.dumps(result), flush=True)

    # --- W&B ---------------------------------------------------------------
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                group=args.wandb_group,
                name=args.wandb_name or f"{arm}-{args.mode}",
                job_type="verify-rollback-gate",
                config={"arm": arm, "mode": args.mode, "K": args.K,
                        "model": args.model, "nprompts": args.nprompts,
                        "inv": args.inv, "pr": 24})
            wandb.log({k: v for k, v in result.items()
                       if isinstance(v, (int, float))})
            wandb.summary.update(result)
            print("WANDB_RUN_ID " + run.id, flush=True)
            run.finish()
        except Exception as exc:
            print(f"[wandb] logging failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
