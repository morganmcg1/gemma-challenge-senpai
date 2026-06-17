#!/usr/bin/env python3
"""PR #598 -- larger-instrument GPQA CI-robustness on gpqa_main (n=448). LOCAL, NO FIRE.

#589 proved base_fullhead's GPQA-Diamond gate margin is NOT CI-robust on the n=198
Diamond ceiling (the binomial item-sampling floor dominates; a Wilson-lb clears only
at n~3758). The only path to a CI-robust GPQA certification is a LARGER-n instrument.
GPQA-Main (n=448) has 2.26x Diamond's items. This driver measures TWO configs on
gpqa_main under the actual downstream SAMPLING protocol (generation_config.json:
temp=1.0/top_p=0.95/top_k=64, lewtun #31), min_tokens=8 (#541), across K seeds:

  (a) base          = UNQUANTIZED bf16 google/gemma-4-E4B-it (the gate denominator).
  (b) base_fullhead = int4-W4A16-g32 QAT body + native 262k bf16 lm_head.

Both arms serve on the IDENTICAL faithful #589 stack (dev307 build + serve_inject
sitecustomize: prometheus shim + FA_SLIDING flash routing + SURGICAL_ATTN_USE_3D_OFF
2D order-preserving attention + PLE_FOLD_EMBED_SCALE). The ONLY difference between
arms is the checkpoint (bf16 body vs int4 body); the native 262k head, attention,
fold, sampler, and dataset are held constant -- so the base<->base_fullhead contrast
isolates the int4-body-quantization effect (the "is the int4 body worse?" question)
and the McNemar pairing satisfies "only the model differs".

The dev307 build serves any Gemma4 ckpt 16x-degenerate unless PLE_FOLD_EMBED_SCALE=1
(the #557 root cause). PLE_FOLD matches PLE_FOLD_TARGET_MODEL to --model by EXACT
string and HARD-RAISES at load if the fold cannot apply -- so a mis-folded denominator
cannot silently pass; it crashes. We pin PLE_FOLD_TARGET_MODEL == --model per arm.

Serve ONCE per config, loop run_eval.py gpqa_main over sampling seeds (serve-once /
eval-many). Per-seed result files are resumable (skip-if-exists) so the sweep can be
split across bounded invocations. NO HF FIRE: analysis_only, official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/gpqa_larger_instrument_ci"
RES = HERE / "results"
QE = ROOT / "research/validity/downstream_quality_eval"
SERVE_INJECT = ROOT / "research/validity/vanilla_base_serve_regression/serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")  # dev307 build (#557/#564/#589)
EVAL_PY = Path("/tmp/land-inspect/bin/python")                     # inspect client (0.3.240 / 0.14.0)
HF_HUB = Path.home() / ".cache/huggingface/hub"

# Resolved local snapshots (both already cached). base = UNQUANTIZED bf16; the gate
# denominator. base_fullhead body = int4-W4A16-g32 QAT (native 262k bf16 lm_head; the
# lm_head is in the quant `ignore` list).
CKPT = {
    "base": str(HF_HUB / "models--google--gemma-4-E4B-it/snapshots/"
                "fee6332c1abaafb77f6f9624236c63aa2f1d0187"),
    "base_fullhead": str(HF_HUB / "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
                         "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"),
}

PORT = int(os.environ.get("PORT", "8000"))
DATASET_SEED = 12345  # fixed gpqa_main choice-shuffle seed -> byte-identical item set across arms+seeds
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}  # generation_config.json (lewtun #31)
GPQA_MAX_TOKENS = 3072
MIN_TOKENS = 8  # wirbel #541 EOS-guard (proven mechanical no-op on GPQA CoT by #563)
GPQA_SPLIT = "main"  # n=448 (extended n=546 fallback exists but Main is available)
# #589 served Diamond at max_model_len=4096; that admission cap (4096-3072=1024 input
# budget) silently TRUNCATED items whose prompt exceeded ~1020 tokens -> vLLM returns a
# 400 at admission, inspect scores the sample wrong (score_on_error). On Diamond that hit
# 1 item/seed (n_error=1 in every #589 result file). GPQA-Main has 2 such items (measured
# max prompt 2418 tok on the long molecular-biology DNA item recnTTKdBzfuoZ7w7; 2nd=1021),
# so 4096 biases BOTH arms down by 2/448 every seed. We raise the cap to 6144 = covers the
# 2418-tok prompt + the FULL faithful 3072 gen budget with margin. This changes nothing for
# items that already fit (identical KV blocks, sampler, batching): worst-case concurrent KV
# (top-16 prompts + full gen ~= 59k tok) stays far under the 88,116-tok pool, so 16-way
# batching is preserved and no preemption occurs -- it only converts the previously
# force-wrong truncated items into real answers. Everything else is the byte-faithful #589
# stack. (Server log is asserted post-hoc for 0 preemptions + 0 admission errors.)
MAX_MODEL_LEN = 6144


def arm_env(config: str) -> dict:
    """Faithful #589 base_fullhead serve env, with PLE_FOLD pinned to this arm's ckpt."""
    return {
        "FA_SLIDING": "1",
        "SURGICAL_ATTN_USE_3D_OFF": "1",
        "PLE_FOLD_EMBED_SCALE": "1",
        "PLE_FOLD_TARGET_MODEL": CKPT[config],  # MUST equal --model (exact-string fold guard)
    }


def start_server(config: str, log_path: Path, *, gpu_mem_util: float, max_num_seqs: int
                 ) -> subprocess.Popen:
    ckpt = CKPT[config]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # curand-less box: force native torch sampler
    env["PYTHONPATH"] = str(SERVE_INJECT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env["PR557_PATCH_DIR"] = str(SUBMISSION)
    env.update(arm_env(config))
    cmd = [
        str(SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", ckpt, "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(gpu_mem_util), "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", json.dumps(SAMPLING),
    ]
    print(f"[serve] config={config} ckpt={ckpt} flags={arm_env(config)} "
          f"gpu_mem_util={gpu_mem_util} max_num_seqs={max_num_seqs}", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def wait_ready(proc: subprocess.Popen, timeout_s=1800) -> None:
    base = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early code={proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("endpoint not ready")


def run_seed(config: str, seed: int, out: Path, *, conc: int, limit: int = 0) -> dict:
    cmd = [
        str(EVAL_PY), str(QE / "run_eval.py"),
        "--task", "gpqa_main", "--gpqa-split", GPQA_SPLIT, "--arm", config,
        "--out", str(out), "--seed", str(DATASET_SEED),
        "--max-tokens", str(GPQA_MAX_TOKENS), "--max-connections", str(conc),
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", "gemma-4-e4b-it",
        "--temperature", str(SAMPLING["temperature"]),
        "--top-p", str(SAMPLING["top_p"]), "--top-k", str(SAMPLING["top_k"]),
        "--sampling-seed", str(seed), "--min-tokens", str(MIN_TOKENS),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[eval] config={config} seed={seed} limit={limit or 'full'} "
          f"START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] config={config} seed={seed} acc={d['accuracy']:.4f} "
          f"scored={d['n_scored']} correct={d['n_correct']} empty={d.get('n_empty')} "
          f"err={d['n_error']} dt={time.time()-t0:.0f}s", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CKPT))
    ap.add_argument("--seeds", default="0,1,2,3,4")  # K=5
    ap.add_argument("--conc", type=int, default=16)         # faithful #589 max-connections
    ap.add_argument("--max-num-seqs", type=int, default=16)  # faithful #589 serve width
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--smoke", action="store_true", help="limit to 3 items on seed 0, then exit")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    log = HERE / f"server_{args.config}_{tag}.log"
    proc = start_server(args.config, log, gpu_mem_util=args.gpu_mem_util,
                        max_num_seqs=args.max_num_seqs)
    try:
        wait_ready(proc)
        print(f"[driver] config={args.config} READY {time.strftime('%H:%M:%S')}", flush=True)
        if args.smoke:
            run_seed(args.config, 0, RES / f"_smoke_{args.config}_s0.json",
                     conc=args.conc, limit=3)
            print(f"[driver] SMOKE OK config={args.config}", flush=True)
            return 0
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        for s in seeds:
            out = RES / f"{args.config}_gpqa_main_mt8_s{s}.json"
            if out.exists():
                print(f"[driver] config={args.config} seed={s} SKIP existing {out.name}",
                      flush=True)
                continue
            run_seed(args.config, s, out, conc=args.conc)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    print(f"[driver] config={args.config} ALL SEEDS COMPLETE {time.strftime('%H:%M:%S')}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
