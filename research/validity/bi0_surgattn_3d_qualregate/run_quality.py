#!/usr/bin/env python3
"""PR #791 -- surgattn-OFF (3D-on-M=1) vs shipped bi0 (force-2D) quality re-gate.

Base = shipped bi0 ``submissions/int4_mtp_bi0_surgattn`` (force-2D, byte-exact
greedy identity). Variant = the SAME submission with ``VLLM_SURGATTN=0`` (the
#785 toggle), which skips the force-2D patch so the TRITON_ATTN kernel gate is
free to pick the 3D split-KV path on the M=1 decode forwards. Under the shipped
serve config (``MAX_NUM_SEQS=1``) every decode step is M=1, so the variant's
3D-on-M=1 divergence is *fully* exercised -- a worst-case quality test.

This launches ONE server per arm via the battle-tested ``harness.LocalServer``,
proves which kernel path is live from the server log (the force-2D patch prints a
positive line; the disabled path prints the ``VLLM_SURGATTN=0`` line), then drives
the same ``run_eval.py`` MMLU-Pro task #773 used. Paired arms (control + variant)
on byte-identical prompts make the arm DELTA the sensitive signal -- most
completions are identical, so the paired difference has far lower variance than
two independent draws and is not confounded by the #773 0.644-vs-0.57
budget/anchor mismatch.

Sampler: ``VLLM_USE_FLASHINFER_SAMPLER=0`` (native) avoids the cuRAND JIT in this
container and is logit-identical (quality unaffected); both arms match.

LOCAL ONLY -- no HF job.
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
SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_surgattn"
PORT = 8000

# Server-process env that is the SAME for both arms (so the only changed variable
# is VLLM_SURGATTN). Native sampler avoids cuRAND JIT; offline avoids HF round-trips
# for the already-cached model/drafter. These do NOT touch logits.
COMMON_SERVER_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

# Per-arm server env. control = shipped (force-2D ON). variant = VLLM_SURGATTN=0.
ARM_ENV = {
    "control": {},
    "variant": {"VLLM_SURGATTN": "0"},
}
# Substring that MUST be in the server log for the arm to be the intended one, and
# one that must be ABSENT (so a mis-toggled arm aborts instead of producing a
# silently-wrong number).
ARM_LOG_REQUIRE = {
    "control": "[int4_mtp_force2d] unified_attention wrapped",
    "variant": "[int4_mtp_surgattn] VLLM_SURGATTN=0",
}
ARM_LOG_FORBID = {
    "control": "[int4_mtp_surgattn] VLLM_SURGATTN=0",
    "variant": "[int4_mtp_force2d] unified_attention wrapped",
}


def _mmlu_cmd(out: Path, *, n: int, seed: int, max_tokens: int, limit: int) -> list[str]:
    cmd = [
        str(EVAL_PY), str(RUN_EVAL),
        "--task", "mmlu_pro", "--arm", "int4_mtp_bi0_surgattn",
        "--out", str(out),
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", "gemma-4-e4b-it",
        "--n", str(n), "--seed", str(seed),
        "--temperature", "1.0", "--top-p", "0.95", "--top-k", "64",
        "--max-tokens", str(max_tokens), "--sampling-seed", "0",
        "--max-connections", "16",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


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


def run_arm(arm: str, *, n: int, seed: int, max_tokens: int, smoke_limit: int,
            out_dir: Path) -> dict:
    extra = {**COMMON_SERVER_ENV, **ARM_ENV[arm]}
    log_path = out_dir / f"server_{arm}.log"
    print(f"\n===== ARM {arm}  extra_env={ARM_ENV[arm]} =====", flush=True)
    result: dict = {"arm": arm, "extra_env": ARM_ENV[arm]}
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=PORT, log_path=log_path,
        extra_env=extra, startup_timeout_s=1800,
    ) as srv:
        # Prove the intended kernel path is live before spending the eval budget.
        need = ARM_LOG_REQUIRE[arm]
        if not _wait_log(log_path, need, timeout_s=120):
            raise RuntimeError(f"[{arm}] expected server-log marker absent: {need!r}")
        forbid = ARM_LOG_FORBID[arm]
        if forbid in log_path.read_text(errors="replace"):
            raise RuntimeError(f"[{arm}] forbidden marker present (wrong toggle): {forbid!r}")
        print(f"[{arm}] toggle proven from server log: present={need!r} absent={forbid!r}",
              flush=True)
        result["toggle_proven"] = True

        eval_env = os.environ.copy()  # clean env for the inspect harness (no offline force)

        # Smoke: a couple items to confirm the harness talks to this server.
        if smoke_limit:
            smoke_out = out_dir / f"mmlu_{arm}_smoke.json"
            print(f"[{arm}] smoke MMLU-Pro limit={smoke_limit}", flush=True)
            subprocess.run(
                _mmlu_cmd(smoke_out, n=n, seed=seed, max_tokens=max_tokens, limit=smoke_limit),
                check=True, env=eval_env, timeout=1800,
            )
            sd = json.loads(smoke_out.read_text())
            # score_on_error=True scores an errored sample as incorrect, so n_scored
            # stays > 0 even when every request 400s. Guard on the actual error count:
            # a clean smoke must have NO errored and at least one non-error sample.
            n_smoke = sd.get("n_samples", 0)
            n_err = sd.get("n_error", 0)
            if n_smoke <= 0:
                raise RuntimeError(f"[{arm}] smoke produced 0 samples")
            if n_err > 0:
                raise RuntimeError(
                    f"[{arm}] smoke had {n_err}/{n_smoke} ERRORED samples "
                    f"(see {smoke_out}); aborting before the full run. "
                    f"Most likely max_tokens too large vs max_model_len=4096."
                )
            print(f"[{arm}] smoke OK: scored={sd['n_scored']} err=0 acc={sd['accuracy']:.3f}", flush=True)

        # Full kill-gate run.
        full_out = out_dir / f"mmlu_{arm}_n{n}_s{seed}_t{max_tokens}.json"
        t0 = time.time()
        print(f"[{arm}] FULL MMLU-Pro n={n} seed={seed} max_tokens={max_tokens}", flush=True)
        subprocess.run(
            _mmlu_cmd(full_out, n=n, seed=seed, max_tokens=max_tokens, limit=0),
            check=True, env=eval_env, timeout=7200,
        )
        d = json.loads(full_out.read_text())
        d["_wall_s"] = round(time.time() - t0, 1)
        if d["n_error"] > 0:
            print(f"[{arm}] WARNING: full run had {d['n_error']}/{d['n_samples']} "
                  f"errored samples", flush=True)
        result.update({
            "mmlu_out": str(full_out),
            "accuracy": d["accuracy"], "n_scored": d["n_scored"], "n_correct": d["n_correct"],
            "n_error": d["n_error"], "n_empty": d["n_empty"],
            "length_stop_rate": d["length_stop_rate"], "ctok_mean": d["completion_tokens_mean"],
            "finish_length_rate": d["finish_length_rate"], "wall_s": d["_wall_s"],
            "per_sample": d["per_sample"],
        })
        _ctok = d["completion_tokens_mean"]
        _ctok_s = f"{_ctok:.0f}" if _ctok is not None else "na"
        print(f"[{arm}] FULL done acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) "
              f"err={d['n_error']} trunc_rate={d['length_stop_rate']:.3f} ctok_mean={_ctok_s} "
              f"wall={result['wall_s']}s", flush=True)
    return result


def paired_delta(control: dict, variant: dict) -> dict:
    """McNemar-style paired comparison on the byte-identical prompt set."""
    cmap = {r["id"]: r for r in control["per_sample"]}
    vmap = {r["id"]: r for r in variant["per_sample"]}
    ids = sorted(set(cmap) & set(vmap))
    # prompt_sha identity proves both arms saw byte-identical prompts.
    sha_mismatch = [i for i in ids if cmap[i].get("prompt_sha") != vmap[i].get("prompt_sha")]
    same_answer = sum(1 for i in ids if cmap[i]["answer"] == vmap[i]["answer"])
    c_correct_v_wrong = sum(1 for i in ids if cmap[i]["correct"] and not vmap[i]["correct"])
    v_correct_c_wrong = sum(1 for i in ids if vmap[i]["correct"] and not cmap[i]["correct"])
    return {
        "n_paired": len(ids),
        "prompt_sha_mismatch": len(sha_mismatch),
        "same_answer_text": same_answer,
        "answer_flip": len(ids) - same_answer,
        "control_right_variant_wrong": c_correct_v_wrong,
        "variant_right_control_wrong": v_correct_c_wrong,
        "net_variant_minus_control_correct": v_correct_c_wrong - c_correct_v_wrong,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="variant,control",
                    help="comma list; variant first (kill-gate leads with the variant)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    # max_model_len=4096 (serve.py); MMLU-Pro #773 used 2048 (leaves >=2048 for
    # input). 4096 here = the full ctx, so 0 tokens remain for any prompt -> every
    # request 400s ("upper bound for 0 input tokens"). Must stay <= ~3500.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--smoke-limit", type=int, default=2)
    ap.add_argument("--out-dir", default=str(HERE / "runs"))
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results: dict[str, dict] = {}
    for arm in arms:
        results[arm] = run_arm(
            arm, n=args.n, seed=args.seed, max_tokens=args.max_tokens,
            smoke_limit=args.smoke_limit, out_dir=out_dir,
        )

    summary = {
        "task": "mmlu_pro", "n": args.n, "seed": args.seed, "max_tokens": args.max_tokens,
        "sampling": "T=1.0/top_p=0.95/top_k=64/sampling-seed=0",
        "served_config": "MAX_NUM_SEQS=1 (every decode M=1)",
        "arms": {a: {k: v for k, v in r.items() if k != "per_sample"} for a, r in results.items()},
    }
    if "control" in results and "variant" in results:
        summary["paired"] = paired_delta(results["control"], results["variant"])
        ca, va = results["control"]["accuracy"], results["variant"]["accuracy"]
        summary["delta_variant_minus_control"] = round(va - ca, 4)
    (out_dir / "mmlu_killgate_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n========== MMLU-Pro KILL-GATE SUMMARY ==========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
