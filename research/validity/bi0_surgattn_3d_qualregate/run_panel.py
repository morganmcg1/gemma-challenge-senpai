#!/usr/bin/env python3
"""PR #791 step-2 full panel: surgattn-OFF (3D-on-M=1) vs shipped bi0 (force-2D).

Step-1 MMLU-Pro kill-gate already PASSED (run_quality.py): variant 0.64 vs
same-session control 0.62 (Δ +0.02, n.s.; 12/100 symmetric answer flips). This
script runs the remaining 3 axes of the #784 quality band, reusing the exact
harnesses #773/#762/#605 used:

  * GSM8K  (research/downstream_quality_gsm8k/gsm8k_eval.py) -- n=1/request, so
    every decode is M=1 at the shipped MAX_NUM_SEQS=1 serve config => the variant's
    3D-on-M=1 divergence is fully exercised.  PAIRED control+variant.
  * GPQA-Diamond (downstream_quality_eval/run_eval.py) -- n=1/request, M=1. PAIRED.
  * AIME-2024 (research/downstream_quality_aime/aime_eval.py) -- maj@k via n=k in
    ONE request; coarse (n=30), the #784 bar is just "no collapse" (>=8/30).
    Run VARIANT-only as a no-collapse check vs the bi0 10/30 anchor; wrapped in
    try/except (n>1 + spec-dec is a fragile combination).

Same LocalServer + server-log toggle proof as run_quality.py.  Shipped serve
config (MAX_MODEL_LEN=4096, MAX_NUM_SEQS=1) is preserved so the only changed
variable is VLLM_SURGATTN.  LOCAL ONLY -- no HF job, analysis_only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
SERVER_PY = ROOT / ".venvs" / "vllm022" / "bin" / "python"
EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")
RUN_EVAL = ROOT / "research" / "validity" / "downstream_quality_eval" / "run_eval.py"
GSM8K_EVAL = ROOT / "research" / "downstream_quality_gsm8k" / "gsm8k_eval.py"
AIME_EVAL = ROOT / "research" / "downstream_quality_aime" / "aime_eval.py"
SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_surgattn"
PORT = 8000

COMMON_SERVER_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",  # native sampler; avoids cuRAND JIT; logit-identical
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
ARM_ENV = {"control": {}, "variant": {"VLLM_SURGATTN": "0"}}
ARM_LOG_REQUIRE = {
    "control": "[int4_mtp_force2d] unified_attention wrapped",
    "variant": "[int4_mtp_surgattn] VLLM_SURGATTN=0",
}
ARM_LOG_FORBID = {
    "control": "[int4_mtp_surgattn] VLLM_SURGATTN=0",
    "variant": "[int4_mtp_force2d] unified_attention wrapped",
}


def _wait_log(log_path: Path, needle: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if needle in log_path.read_text(errors="replace"):
                return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    return False


def _run(cmd: list[str], env: dict, timeout: int, tag: str) -> bool:
    t0 = time.time()
    print(f"\n[{tag}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env, timeout=timeout)
        print(f"[{tag}] OK wall={time.time()-t0:.0f}s", flush=True)
        return True
    except Exception as e:  # noqa: BLE001 -- one task failing must not abort the panel
        print(f"[{tag}] FAILED after {time.time()-t0:.0f}s: {e!r}", flush=True)
        return False


def gsm8k_cmd(arm: str, out_dir: Path, *, n: int) -> tuple[list[str], Path]:
    out = out_dir / f"gsm8k_{arm}_sampled_s0.json"
    cmd = [
        str(EVAL_PY), str(GSM8K_EVAL),
        "--base-url", f"http://127.0.0.1:{PORT}", "--model", "gemma-4-e4b-it",
        "--label", f"gsm8k_{arm}", "--regimes", "sampled",
        "--n", str(n), "--n-shot", "8", "--seed", "1234", "--sampling-seed", "0",
        "--top-p", "0.95", "--top-k", "64", "--max-tokens", "512", "--min-tokens", "8",
        "--concurrency", "16", "--out-dir", str(out_dir),
    ]
    return cmd, out


def gpqa_cmd(arm: str, out_dir: Path, *, seed: int, max_tokens: int) -> tuple[list[str], Path]:
    out = out_dir / f"gpqa_{arm}_s{seed}_t{max_tokens}.json"
    cmd = [
        str(EVAL_PY), str(RUN_EVAL),
        "--task", "gpqa_diamond", "--arm", "int4_mtp_bi0_surgattn",
        "--out", str(out),
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", "gemma-4-e4b-it",
        "--seed", str(seed),
        "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
        "--max-tokens", str(max_tokens), "--sampling-seed", "0", "--max-connections", "16",
    ]
    return cmd, out


def aime_cmd(arm: str, out_dir: Path, *, k: int, max_tokens: int) -> tuple[list[str], Path]:
    out = out_dir / f"aime_{arm}_2024_k{k}.json"
    cmd = [
        str(EVAL_PY), str(AIME_EVAL),
        "--base-url", f"http://127.0.0.1:{PORT}", "--model", "gemma-4-e4b-it",
        "--years", "2024", "--k", str(k),
        "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
        "--max-tokens", str(max_tokens), "--min-tokens", "8", "--no-thinking",
        "--seed", "1234", "--save-text", "--label", f"aime_{arm}", "--out", str(out),
    ]
    return cmd, out


def run_arm(arm: str, *, out_dir: Path, gsm8k_n: int, gpqa_seed: int, gpqa_maxtok: int,
            aime_k: int, aime_maxtok: int, do_aime: bool) -> dict:
    extra = {**COMMON_SERVER_ENV, **ARM_ENV[arm]}
    log_path = out_dir / f"panel_server_{arm}.log"
    print(f"\n===== ARM {arm}  extra_env={ARM_ENV[arm]} =====", flush=True)
    res: dict = {"arm": arm, "extra_env": ARM_ENV[arm], "tasks": {}}
    eval_env = os.environ.copy()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=PORT, log_path=log_path,
        extra_env=extra, startup_timeout_s=1800,
    ):
        need, forbid = ARM_LOG_REQUIRE[arm], ARM_LOG_FORBID[arm]
        if not _wait_log(log_path, need, timeout_s=120):
            raise RuntimeError(f"[{arm}] expected server-log marker absent: {need!r}")
        if forbid in log_path.read_text(errors="replace"):
            raise RuntimeError(f"[{arm}] forbidden marker present (wrong toggle): {forbid!r}")
        print(f"[{arm}] toggle proven: present={need!r} absent={forbid!r}", flush=True)
        res["toggle_proven"] = True

        cmd, out = gsm8k_cmd(arm, out_dir, n=gsm8k_n)
        if _run(cmd, eval_env, timeout=7200, tag=f"{arm}/gsm8k") and out.exists():
            d = json.loads(out.read_text())
            res["tasks"]["gsm8k"] = {"accuracy": d["accuracy"], "n": d["n_problems"],
                                     "n_correct": d["n_correct"], "trunc": d.get("truncation_rate"),
                                     "out": str(out)}

        cmd, out = gpqa_cmd(arm, out_dir, seed=gpqa_seed, max_tokens=gpqa_maxtok)
        if _run(cmd, eval_env, timeout=10800, tag=f"{arm}/gpqa") and out.exists():
            d = json.loads(out.read_text())
            res["tasks"]["gpqa_diamond"] = {"accuracy": d["accuracy"], "n": d["n_scored"],
                                            "n_correct": d["n_correct"], "n_error": d["n_error"],
                                            "trunc": d.get("length_stop_rate"), "out": str(out)}

        if do_aime:
            cmd, out = aime_cmd(arm, out_dir, k=aime_k, max_tokens=aime_maxtok)
            if _run(cmd, eval_env, timeout=10800, tag=f"{arm}/aime") and out.exists():
                d = json.loads(out.read_text())
                res["tasks"]["aime"] = {"maj_k_accuracy": d["maj_k_accuracy"], "n": d["n_problems"],
                                        "n_correct_maj": d["n_correct_maj"], "k": d["maj_k"],
                                        "out": str(out)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="variant,control")
    ap.add_argument("--aime-arms", default="variant", help="arms to also run AIME on")
    ap.add_argument("--gsm8k-n", type=int, default=300)
    ap.add_argument("--gpqa-seed", type=int, default=12345)
    ap.add_argument("--gpqa-maxtok", type=int, default=3072)
    ap.add_argument("--aime-k", type=int, default=8)
    ap.add_argument("--aime-maxtok", type=int, default=3072)
    ap.add_argument("--out-dir", default=str(HERE / "runs"))
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    aime_arms = {a.strip() for a in args.aime_arms.split(",") if a.strip()}

    results: dict[str, dict] = {}
    for arm in arms:
        results[arm] = run_arm(
            arm, out_dir=out_dir, gsm8k_n=args.gsm8k_n, gpqa_seed=args.gpqa_seed,
            gpqa_maxtok=args.gpqa_maxtok, aime_k=args.aime_k, aime_maxtok=args.aime_maxtok,
            do_aime=(arm in aime_arms),
        )
        # write incrementally so a budget cut still leaves a usable summary
        (out_dir / "panel_summary.json").write_text(json.dumps(
            {"served_config": "MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 (every n=1 decode M=1)",
             "sampling": "T=1.0/top_p=0.95/top_k=64",
             "gsm8k_n": args.gsm8k_n, "gpqa_seed": args.gpqa_seed,
             "gpqa_maxtok": args.gpqa_maxtok, "aime_k": args.aime_k,
             "arms": results}, indent=2))

    print("\n========== PANEL SUMMARY ==========", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
