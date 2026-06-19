#!/usr/bin/env python3
"""PR #717 -- 30-seed GPQA-Diamond #31-SAMPLED robustness sweep on fern #659's INT8-LOCUS arm.

LOCAL, NO FIRE, analysis_only. official_tps=0, no_hf_job, no submission, no served-file
change. Only the assigned A10G.

Reuses the EXACT #696 GPQA-Diamond instrument VERBATIM -- same dev307 surgical+fold serve
stack, same lewtun #31 sampling protocol (do_sample=True T=1.0 top_p=0.95 top_k=64;
min_tokens=8 EOS-guard; DATASET_SEED=12345; max_tokens=3072), same inspect_evals eval
(run_eval.py --task gpqa_diamond), same 30-seed pooled-Wilson + seed-mean-t-CI adjudication.

The ONLY change is the quantized model under test. #696 served the int4-W4A16-g32 QAT
snapshot directly (int4 BODY, native bf16 head). #717 instead serves the bf16
qat-unquantized base with an IN-MEMORY RTN fake-quant injected at vLLM weight-load
(research/validity/int8_locus_gpqa_robustness/sitecustomize.py), reproducing fern #659's
nmjvtfov int8-locus recipe:
    - int4-g128 on the 343 body Linear modules (the int4_g128 skeleton),
    - int8 (group FAKEQUANT_INT8_GROUP) on language_model.layers[14..27] (fern's upgrade),
    - synthetic int4-g128 lm_head from bf16 embed_tokens (embed itself stays bf16).
Served bf16 (dequantized) -- FAKE-quant reproduces the rounding error faithfully, the SAME
in-memory RTN fern used from the qat-unquantized source.

VALIDITY GATE: after the server is ready we grep its log for the "[sc-717] RTN APPLIED"
marker (int4=225 int8=118). A missing marker means the injector silently no-op'd and we
would be serving full-precision bf16 -> inflated GPQA. We FAIL LOUD in that case.

Serve-once / eval-many. Existing seeds are skipped (idempotent / resumable across wakeups).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/int8_locus_gpqa_robustness"
RES = HERE / "results_gpqa"
QE = ROOT / "research/validity/downstream_quality_eval"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
MODULE_LIST = ROOT / "submissions/int4_g128_lmhead/official_quantized_modules.json"
MAKE_SERVE = HERE / "make_serve_dir.py"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")   # dev307 build (#564)
EVAL_PY = Path(os.environ.get("EVAL_PY", "/tmp/eval-serve-venv/bin/python"))
SERVE_DIR = Path(os.environ.get("SERVE_DIR", "/tmp/wirbel_int8locus_serve"))
PORT = int(os.environ.get("PORT", "8000"))
DATASET_SEED = 12345
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
GPQA_MAX_TOKENS = 3072
MIN_TOKENS = 8
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))  # 4096 == #696 (GPQA mt=3072 fits)
INT8_LAYERS = os.environ.get("FAKEQUANT_INT8_LAYERS", "14-27")  # fern's L14-27 int8 upgrade
INT8_GROUP = os.environ.get("FAKEQUANT_INT8_GROUP", "128")
RTN_MARKER = "[sc-717] RTN APPLIED"

# fake-quant injection + #696 surgical/fold serve recipe. PLE_FOLD_TARGET_MODEL must equal
# the --model path (vLLM gates the Gemma4 PLE fold on model_config.model == this).
ARM_ENV = {
    "FA_SLIDING": "1",
    "SURGICAL_ATTN_USE_3D_OFF": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": str(SERVE_DIR),
    "FAKEQUANT_MODULE_LIST": str(MODULE_LIST),
    "FAKEQUANT_INT8_LAYERS": INT8_LAYERS,
    "FAKEQUANT_INT8_GROUP": INT8_GROUP,
    "PR557_PATCH_DIR": str(SUBMISSION),
}


def build_serve_dir() -> None:
    env = dict(os.environ)
    env["SERVE_DIR"] = str(SERVE_DIR)
    print(f"[driver] building tiny serve-dir {SERVE_DIR} (symlink base; tie=false) ...", flush=True)
    subprocess.run([str(SERVER_PY), str(MAKE_SERVE)], env=env, check=True)


def start_server(log_path: Path, disable_fakequant: bool = False) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # OUR sitecustomize (the RTN injector) must win -> HERE first on PYTHONPATH.
    env["PYTHONPATH"] = str(HERE) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env.update(ARM_ENV)
    if disable_fakequant:
        env["FAKEQUANT_DISABLE"] = "1"   # full-bf16 control (deliberate no-op)
    cmd = [
        str(SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(SERVE_DIR), "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", "16",
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", json.dumps(SAMPLING),
    ]
    tag = "FULL-BF16 CONTROL" if disable_fakequant else f"int8-locus RTN (L{INT8_LAYERS} int8 g{INT8_GROUP})"
    print(f"[serve] {tag} flags={ARM_ENV} mml={MAX_MODEL_LEN} serve_dir={SERVE_DIR}", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def wait_ready(proc: subprocess.Popen, timeout_s=1200) -> None:
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


def assert_rtn_applied(log_path: Path, disable_fakequant: bool = False) -> str:
    """Validity gate: confirm the in-memory RTN fake-quant actually ran (or, in control
    mode, that it was deliberately disabled). A silent no-op would serve full bf16 and
    inflate GPQA -> we refuse to eval without the marker."""
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    want = "[sc-717] RTN fake-quant DISABLED" if disable_fakequant else RTN_MARKER
    line = next((ln for ln in text.splitlines() if want in ln), None)
    if line is None:
        raise RuntimeError(
            f"VALIDITY GATE FAILED: '{want}' marker absent from {log_path.name}. "
            "The injector did not run -- refusing to eval (would serve full bf16)."
        )
    print(f"[driver] RTN VALIDITY GATE OK: {line.strip()}", flush=True)
    return line.strip()


def run_seed(seed: int, out: Path, limit: int = 0, greedy: bool = False) -> dict:
    # greedy precision anchor (instruction #5): temperature=0 deterministic. The server's
    # override-generation-config default (T=1.0 sampling) is overridden per-request by the
    # explicit temperature=0 in GenerateConfig -- the same single sampling-configured server
    # serves both the #31-sampled seeds and this greedy anchor (exactly as #696 did).
    temp = "0.0" if greedy else str(SAMPLING["temperature"])
    top_p = "1.0" if greedy else str(SAMPLING["top_p"])
    top_k = "0" if greedy else str(SAMPLING["top_k"])
    cmd = [
        str(EVAL_PY), str(QE / "run_eval.py"),
        "--task", "gpqa_diamond", "--arm", "int8_locus" + ("_greedy" if greedy else ""),
        "--out", str(out), "--seed", str(DATASET_SEED),
        "--max-tokens", str(GPQA_MAX_TOKENS), "--max-connections", "16",
        "--base-url", f"http://127.0.0.1:{PORT}/v1", "--model", "gemma-4-e4b-it",
        "--temperature", temp, "--top-p", top_p, "--top-k", top_k,
        "--sampling-seed", str(seed), "--min-tokens", str(MIN_TOKENS),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    tag = "greedy-anchor" if greedy else f"seed={seed}"
    print(f"[eval] {tag} limit={limit or 'full'} START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[eval] {tag} acc={d['accuracy']:.4f} scored={d['n_scored']} "
          f"correct={d['n_correct']} empty={d.get('n_empty')} err={d['n_error']} "
          f"dt={time.time()-t0:.0f}s", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(30)))
    ap.add_argument("--smoke", action="store_true", help="4 items on seed 0, then exit")
    ap.add_argument("--control", action="store_true",
                    help="FAKEQUANT_DISABLE=1 full-bf16 control (no fake-quant) -- a "
                         "ceiling sanity check, NOT the int8-locus arm")
    ap.add_argument("--no-greedy-anchor", action="store_true",
                    help="skip the instruction-#5 greedy precision anchor (full GPQA-D "
                         "temperature=0); by default it runs once before the sampled seeds")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    build_serve_dir()
    suffix = "_control" if args.control else ""
    log = HERE / (f"server_smoke{suffix}.log" if args.smoke else f"server_gpqa{suffix}.log")
    proc = start_server(log, disable_fakequant=args.control)
    try:
        wait_ready(proc)
        print(f"[driver] READY {time.strftime('%H:%M:%S')}", flush=True)
        assert_rtn_applied(log, disable_fakequant=args.control)
        if args.smoke:
            run_seed(0, RES / f"_smoke_gpqa{suffix}_s0.json", limit=4)
            print("[driver] SMOKE OK", flush=True)
            return 0
        # greedy precision anchor (instruction #5, mirrors #696): full-instrument
        # temperature=0 deterministic GPQA-D on the SAME sampling-configured server
        # (per-request override). Run FIRST so it is captured early and survives wakeup
        # cutoffs; idempotent. Also the basis-degeneration contingency reference.
        if not args.no_greedy_anchor:
            ganchor = RES / f"gpqa_int8locus{suffix}_greedy.json"
            if ganchor.exists():
                print(f"[driver] greedy-anchor SKIP existing {ganchor.name}", flush=True)
            else:
                run_seed(0, ganchor, greedy=True)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        for s in seeds:
            out = RES / f"gpqa_int8locus{suffix}_sampled_s{s}.json"
            if out.exists():
                print(f"[driver] seed={s} SKIP existing {out.name}", flush=True)
                continue
            run_seed(s, out)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    print(f"[driver] ALL SEEDS COMPLETE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
