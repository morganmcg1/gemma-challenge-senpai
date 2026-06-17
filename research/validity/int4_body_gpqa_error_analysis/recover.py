#!/usr/bin/env python3
"""PR #619 -- max_tokens mitigation test for the int4-body gpqa_main truncation deficit.

#598's gpqa_main sweep ran at max_tokens=3072. The failure-mode analysis (analyze.py)
found that the int4 body TRUNCATES more than the bf16 base (313 vs 239 cells over the
2240-cell sweep) and that EVERY truncated cell is scored wrong (it never emits the
final 'ANSWER: X' line). The truncated completions are coherent long CoT, NOT degenerate
repetition loops -- so the truncation is a generation-BUDGET artifact, not an int4
pathology. This driver tests the single most-promising recoverable knob: raise the
generation budget and re-measure whether the truncated cells recover (reach a correct
answer) and the deficit shrinks.

FAIR by construction: BOTH arms are re-served on the byte-identical faithful #589 stack
(same dev307 build, serve_inject, FA_SLIDING + SURGICAL_ATTN_USE_3D_OFF + PLE_FOLD,
sampler, dataset seed) -- the ONLY changes vs #598 are max_tokens 3072->RECOVER_MAX_TOKENS
and the matching max_model_len bump. We re-run ONLY the cells that truncated in EITHER
arm on a given seed (per-seed ids-file in ids/trunc_s{seed}.json) -- the minimal set that
can change. Non-truncated cells are unchanged from #598 (same seed+prompt+model -> the
token stream before the old 3072 cap is identical), so the recovery aggregate substitutes
just these cells' new outcomes into the stored #598 result and recomputes the deficit.

LOCAL, analysis_only, NO HF FIRE. Run with the dev307 server venv path baked below.
  python recover.py --config base_fullhead --seeds 0,1,2,3,4
  python recover.py --config base          --seeds 0,1,2,3,4
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
HERE = ROOT / "research/validity/int4_body_gpqa_error_analysis"
RES = HERE / "recover_results"
IDS = HERE / "ids"
QE = ROOT / "research/validity/downstream_quality_eval"
SERVE_INJECT = ROOT / "research/validity/vanilla_base_serve_regression/serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")  # dev307 build (#557/#564/#589)
EVAL_PY = Path("/tmp/land-inspect/bin/python")
HF_HUB = Path.home() / ".cache/huggingface/hub"

CKPT = {
    "base": str(HF_HUB / "models--google--gemma-4-E4B-it/snapshots/"
                "fee6332c1abaafb77f6f9624236c63aa2f1d0187"),
    "base_fullhead": str(HF_HUB / "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
                         "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"),
}

PORT = int(os.environ.get("PORT", "8000"))
DATASET_SEED = 12345  # MUST match #598 -> byte-identical item set + choice shuffle
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}  # generation_config.json
MIN_TOKENS = 8  # #541 EOS guard (held constant with #598)
# THE KNOB UNDER TEST: 3072 (#598) -> 6144 (2x). max_model_len must cover the longest
# gpqa_main prompt (measured 2418 tok on recnTTKdBzfuoZ7w7) + the full new gen budget.
RECOVER_MAX_TOKENS = 6144
MAX_MODEL_LEN = 8960  # >= 2418 + 6144 with margin


def arm_env(config: str) -> dict:
    return {
        "FA_SLIDING": "1",
        "SURGICAL_ATTN_USE_3D_OFF": "1",
        "PLE_FOLD_EMBED_SCALE": "1",
        "PLE_FOLD_TARGET_MODEL": CKPT[config],
    }


def start_server(config: str, log_path: Path, *, gpu_mem_util: float, max_num_seqs: int
                 ) -> subprocess.Popen:
    ckpt = CKPT[config]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
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
          f"max_model_len={MAX_MODEL_LEN} max_num_seqs={max_num_seqs}", flush=True)
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


def run_seed(config: str, seed: int, out: Path, ids_file: Path, *, conc: int) -> dict:
    cmd = [
        str(EVAL_PY), str(QE / "run_eval.py"),
        "--task", "gpqa_main", "--gpqa-split", "main", "--arm", config,
        "--out", str(out), "--seed", str(DATASET_SEED),
        "--max-tokens", str(RECOVER_MAX_TOKENS), "--max-connections", str(conc),
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", "gemma-4-e4b-it",
        "--temperature", str(SAMPLING["temperature"]),
        "--top-p", str(SAMPLING["top_p"]), "--top-k", str(SAMPLING["top_k"]),
        "--sampling-seed", str(seed), "--min-tokens", str(MIN_TOKENS),
        "--ids-file", str(ids_file),
    ]
    print(f"[eval] config={config} seed={seed} ids_file={ids_file.name} "
          f"max_tokens={RECOVER_MAX_TOKENS} START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] config={config} seed={seed} n={d['n_samples']} acc={d['accuracy']:.4f} "
          f"correct={d['n_correct']} empty={d.get('n_empty')} err={d['n_error']} "
          f"dt={time.time()-t0:.0f}s", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CKPT))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--smoke", action="store_true", help="seed 0, first ids-file only, exit")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    log = HERE / f"server_{args.config}_{tag}.log"
    proc = start_server(args.config, log, gpu_mem_util=args.gpu_mem_util,
                        max_num_seqs=args.max_num_seqs)
    try:
        wait_ready(proc)
        print(f"[driver] config={args.config} READY {time.strftime('%H:%M:%S')}", flush=True)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        if args.smoke:
            seeds = seeds[:1]
        for s in seeds:
            ids_file = (IDS / "smoke.json") if args.smoke else (IDS / f"trunc_s{s}.json")
            out = RES / f"{'_smoke_' if args.smoke else ''}{args.config}_recover_s{s}.json"
            if out.exists():
                print(f"[driver] config={args.config} seed={s} SKIP existing {out.name}", flush=True)
                continue
            run_seed(args.config, s, out, ids_file, conc=args.conc)
            if args.smoke:
                print(f"[driver] SMOKE OK config={args.config}", flush=True)
                return 0
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    print(f"[driver] config={args.config} DONE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
