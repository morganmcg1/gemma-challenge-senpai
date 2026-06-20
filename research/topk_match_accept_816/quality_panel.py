#!/usr/bin/env python
"""PR #816 — Downstream quality panel for the top-k-match accept-branch.

For a given TOPK_ACCEPT_K, serve int4head ONCE (with the topk patch live) and run
the in-band quality gates against that single endpoint: MMLU-Pro, GPQA-Diamond,
AIME (greedy maj@1 and optional maj@8), GSM8K. These accuracies are what decide
whether a k stays inside the #784 floors:
  MMLU-Pro >= 0.572, GPQA-Diamond >= 0.471, GSM8K >= 0.807, AIME >= 0.090,
  PPL <= 2.42, 128/128.

Accuracy is concurrency-independent for this batch-invariant submission (force-2D
attention + M-invariant Marlin GEMV), so we serve at MAX_NUM_SEQS=32 to finish the
panel inside the run budget; the conc=1 TPS number comes from topk_sweep.py, not
here. Each task runs as its own subprocess against the live --base-url and is
isolated in try/except so one failure cannot sink the panel.

LOCAL A10G only. No HF job. Run (background), one k per invocation:
  CUDA_VISIBLE_DEVICES=0 uv run python research/topk_match_accept_816/quality_panel.py \
    --k 2 --tasks mmlu_pro,gpqa_diamond,aime_greedy,gsm8k \
    --wandb-group bi0-int4head-topk-accept
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "int4_mtp_bi0_int4head"
SERVER_PY = ROOT / ".venv" / "bin" / "python"
RUN_EVAL = ROOT / "research" / "validity" / "downstream_quality_eval" / "run_eval.py"
AIME_EVAL = ROOT / "research" / "downstream_quality_aime" / "aime_eval.py"
GSM8K_EVAL = ROOT / "research" / "downstream_quality_gsm8k" / "gsm8k_eval.py"

# #784 in-band floors + the #795 int4head starting panel (k=1 reference).
FLOORS = {"mmlu_pro": 0.572, "gpqa_diamond": 0.471, "gsm8k": 0.807, "aime": 0.090}
K1_REFERENCE_795 = {
    "mmlu_pro": 0.692, "gpqa_diamond": 0.5030, "gsm8k": 0.915,
    "aime_greedy": 0.300, "aime_maj8": 0.400, "ppl": 2.00256,
}


def _run(cmd: list[str], log: Path, timeout_s: int) -> int:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    with open(log, "w") as fh:
        return subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s
        ).returncode


def task_mmlu_pro(base_url, out_dir, k, *, n, seed, max_tokens) -> dict[str, Any]:
    out = out_dir / f"mmlu_pro_k{k}.json"
    rc = _run(
        [str(SERVER_PY), str(RUN_EVAL), "--task", "mmlu_pro",
         "--arm", f"k{k}", "--out", str(out), "--base-url", f"{base_url}/v1",
         "--n", str(n), "--seed", str(seed), "--max-tokens", str(max_tokens)],
        out_dir / f"mmlu_pro_k{k}.log", timeout_s=5400,
    )
    d = json.loads(out.read_text()) if out.exists() else {}
    return {"rc": rc, "accuracy": d.get("accuracy"), "n_scored": d.get("n_scored"),
            "empty_rate": d.get("empty_rate"), "out": str(out)}


def task_gpqa(base_url, out_dir, k, *, seed, max_tokens) -> dict[str, Any]:
    out = out_dir / f"gpqa_diamond_k{k}.json"
    rc = _run(
        [str(SERVER_PY), str(RUN_EVAL), "--task", "gpqa_diamond",
         "--arm", f"k{k}", "--out", str(out), "--base-url", f"{base_url}/v1",
         "--seed", str(seed), "--max-tokens", str(max_tokens)],
        out_dir / f"gpqa_diamond_k{k}.log", timeout_s=5400,
    )
    d = json.loads(out.read_text()) if out.exists() else {}
    return {"rc": rc, "accuracy": d.get("accuracy"), "n_scored": d.get("n_scored"),
            "empty_rate": d.get("empty_rate"), "out": str(out)}


def task_aime(base_url, out_dir, k, *, maj_k, temperature, limit, max_tokens) -> dict[str, Any]:
    tag = "greedy" if maj_k == 1 else f"maj{maj_k}"
    out = out_dir / f"aime_{tag}_k{k}.json"
    # EXACT #795 firegate protocol (research/validity/int4head_firegate_panel/run_aime.sh):
    # --no-thinking + --min-tokens 8 + --seed 1234. Without --no-thinking the model
    # spends the 3072-token budget in the reasoning channel and truncates before the
    # final \boxed{}, collapsing greedy maj@1 from the #795 ref 0.300 -> 0.100 (the
    # extractor then falls back to a stray intermediate integer). This bug sank the
    # first k=1 panel; --no-thinking restores the reference.
    cmd = [str(SERVER_PY), str(AIME_EVAL), "--base-url", base_url,
           "--k", str(maj_k), "--temperature", str(temperature),
           "--max-tokens", str(max_tokens), "--years", "2024",
           "--no-thinking", "--min-tokens", "8", "--seed", "1234",
           "--save-text", "--label", f"k{k}_{tag}", "--out", str(out)]
    if maj_k == 1:  # pure greedy (top_k=-1 disables; matches #795 exactly)
        cmd += ["--top-k", "-1", "--top-p", "1.0", "--client-concurrency", "8"]
    else:  # sampled maj@k (#795: T=1.0 top_p=0.95 top_k=64)
        cmd += ["--top-p", "0.95", "--top-k", "64", "--client-concurrency", "16"]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _run(cmd, out_dir / f"aime_{tag}_k{k}.log", timeout_s=5400)
    d = json.loads(out.read_text()) if out.exists() else {}
    # aime_eval reports maj@k accuracy under "maj_k_accuracy"; resolve leniently.
    acc = (d.get("maj_k_accuracy") or d.get("maj_at_k_accuracy")
           or d.get("accuracy") or d.get("maj_accuracy"))
    return {"rc": rc, "accuracy": acc, "out": str(out), "raw_keys": list(d.keys())[:12]}


def task_gsm8k(base_url, out_dir, k, *, n, max_tokens) -> dict[str, Any]:
    label = f"k{k}"
    rc = _run(
        [str(SERVER_PY), str(GSM8K_EVAL), "--base-url", base_url,
         "--regimes", "greedy", "--n", str(n), "--label", label,
         "--out-dir", str(out_dir), "--max-tokens", str(max_tokens),
         "--concurrency", "32"],
        out_dir / f"gsm8k_k{k}.log", timeout_s=5400,
    )
    out = out_dir / f"{label}_greedy.json"
    d = json.loads(out.read_text()) if out.exists() else {}
    return {"rc": rc, "accuracy": d.get("accuracy"), "n": d.get("n"),
            "out": str(out), "raw_keys": list(d.keys())[:12]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="TOPK_ACCEPT_K to serve")
    ap.add_argument("--tasks", default="mmlu_pro,gpqa_diamond,aime_greedy,gsm8k",
                    help="comma list from {mmlu_pro,gpqa_diamond,aime_greedy,aime_maj8,gsm8k}")
    ap.add_argument("--num-spec", type=int, default=6)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="server context. 4096 fits aime(3072)+gsm8k(512); use 12288 "
                         "for greedy MC (gpqa generates up to 6144), matching #795.")
    ap.add_argument("--mmlu-n", type=int, default=250)
    ap.add_argument("--gsm8k-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mc-max-tokens", type=int, default=4096)
    ap.add_argument("--gpqa-max-tokens", type=int, default=6144,
                    help="#795 used 6144 for GPQA-Diamond greedy generations.")
    ap.add_argument("--aime-max-tokens", type=int, default=3072)
    ap.add_argument("--aime-limit", type=int, default=0)
    ap.add_argument("--wandb-group", default="bi0-int4head-topk-accept")
    ap.add_argument("--wandb-prefix", default="stark/topk-quality")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[gpu-env] {n}", flush=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = HERE / "runs" / f"quality_k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"server_k{args.k}.log"
    extra_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TOPK_ACCEPT_K": str(args.k),
        "NUM_SPECULATIVE_TOKENS": str(args.num_spec),
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        "MAX_MODEL_LEN": str(args.max_model_len),
    }
    print(f"[quality] k={args.k} tasks={tasks} max_num_seqs={args.max_num_seqs} "
          f"out={out_dir}", flush=True)

    rec: dict[str, Any] = {"k": args.k, "tasks": {}, "floors": FLOORS,
                           "k1_reference_795": K1_REFERENCE_795}
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION, server_python=SERVER_PY, port=args.port, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        rec["boot_s"] = time.time() - t0
        rec["model_id"] = srv.model_id
        base_url = srv.base_url
        model = srv.served_model_name
        print(f"[quality] serving {model} at {base_url}", flush=True)

        runners = {
            "mmlu_pro": lambda: task_mmlu_pro(
                base_url, out_dir, args.k, n=args.mmlu_n, seed=args.seed,
                max_tokens=args.mc_max_tokens),
            "gpqa_diamond": lambda: task_gpqa(
                base_url, out_dir, args.k, seed=args.seed,
                max_tokens=args.gpqa_max_tokens),
            "aime_greedy": lambda: task_aime(
                base_url, out_dir, args.k, maj_k=1, temperature=0.0,
                limit=args.aime_limit, max_tokens=args.aime_max_tokens),
            "aime_maj8": lambda: task_aime(
                base_url, out_dir, args.k, maj_k=8, temperature=1.0,
                limit=args.aime_limit, max_tokens=args.aime_max_tokens),
            "gsm8k": lambda: task_gsm8k(
                base_url, out_dir, args.k, n=args.gsm8k_n,
                max_tokens=512),
        }
        for t in tasks:
            if t not in runners:
                print(f"[quality] unknown task {t!r}; skipping", flush=True)
                continue
            print(f"\n----- {t} (k={args.k}) -----", flush=True)
            ts = time.time()
            try:
                res = runners[t]()
            except Exception as exc:  # noqa: BLE001
                res = {"error": repr(exc)}
            res["wall_s"] = time.time() - ts
            rec["tasks"][t] = res
            (out_dir / "panel.json").write_text(json.dumps(rec, indent=2, default=str))
            print(f"[quality] {t}: {res.get('accuracy')} (rc={res.get('rc')}, "
                  f"{res.get('wall_s', 0):.0f}s)", flush=True)

    (out_dir / "panel.json").write_text(json.dumps(rec, indent=2, default=str))

    # In-band verdict per measured task.
    print(f"\n===== QUALITY PANEL k={args.k} =====", flush=True)
    in_band = True
    for t, res in rec["tasks"].items():
        acc = res.get("accuracy")
        floor_key = "aime" if t.startswith("aime") else t
        floor = FLOORS.get(floor_key)
        ok = (acc is not None and floor is not None and acc >= floor)
        if floor is not None and t != "aime_maj8":
            in_band = in_band and ok
        ref = K1_REFERENCE_795.get(t)
        print(f"  {t:14s} acc={acc} floor={floor} ref795={ref} "
              f"{'PASS' if ok else 'FAIL/NA'}", flush=True)
    rec["in_band"] = in_band
    print(f"VERDICT k={args.k}: {'IN-BAND' if in_band else 'OUT-OF-BAND/incomplete'}",
          flush=True)
    (out_dir / "panel.json").write_text(json.dumps(rec, indent=2, default=str))

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                group=args.wandb_group, name=f"{args.wandb_prefix}-k{args.k}",
                reinit=True,
                config={"pr": 816, "experiment": "topk-match-accept-quality",
                        "local_a10g": True, "topk_accept_k": args.k,
                        "max_num_seqs": args.max_num_seqs, "model_id": rec.get("model_id")},
            )
            summ = {"topk_accept_k": args.k, "in_band": in_band}
            for t, res in rec["tasks"].items():
                summ[f"{t}_accuracy"] = res.get("accuracy")
            run.summary.update(summ)
            print(f"[wandb] logged {run.id}", flush=True)
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] log failed ({exc})", flush=True)


if __name__ == "__main__":
    main()
