#!/usr/bin/env python3
"""PR #696 -- int4-body AIME on the gate-faithful #31 SAMPLED basis (the TEST leg).

LOCAL, NO FIRE, analysis_only. official_tps=0, no_hf_job, no submission.

#692 only had AIME on GREEDY (0.3500, 75% of the 0.4667 vanilla base) -- and that number
was the g128-SERVED int4ar arm (int4 g128 body + int4 g128 head) on vLLM 0.22.0. This card
re-measures AIME on the SAME body-isolation arm the GPQA leg uses -- base_fullhead = the
int4-W4A16-g32 QAT body + native bf16 262k head (head NOT quantized) -- on the SAME dev307
surgical+fold stack, so the joint {GPQA,AIME} verdict is config- AND engine-consistent.
We measure int4-body AIME GREEDY and #31-SAMPLED on this identical arm so the greedy->sampled
recovery is a clean within-config contrast (exactly the "does the wall survive the basis
correction" question).

Serve = base_fullhead (mml=8192 so the gb6144 AIME budget fits; GPQA used mml=4096).
Eval  = research/downstream_quality_aime/aime_eval.py against the live endpoint, full 60
        (2024 + 2025-I + 2025-II), k=1, min_tokens=8, --no-thinking, max_tokens=6144.
  greedy : T=0.0 top_p=1.0 top_k=-1   (matches the 0.4667 base denominator harness)
  #31    : T=1.0 top_p=0.95 top_k=64  (generation_config.json), one --seed per call.
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
HERE = ROOT / "research/validity/int4_quality_31basis_confirm"
RES = HERE / "results_aime"
AIME_EVAL = ROOT / "research/downstream_quality_aime/aime_eval.py"
SERVE_INJECT = ROOT / "research/validity/vanilla_base_serve_regression/serve_inject"
SUBMISSION = ROOT / "submissions/fa2sw_strict_surgical357"
SERVER_PY = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")
CLIENT_PY = Path(os.environ.get("CLIENT_PY", "/tmp/eval-serve-venv/bin/python"))
STOCK = Path(
    os.environ.get(
        "STOCK_CKPT",
        str(Path.home()
            / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct"
              "/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"),
    )
)
PORT = int(os.environ.get("PORT", "8001"))
MAX_MODEL_LEN = 8192
AIME_MT = 6144
# Canonical 60 = 2024(30) + 2025(30) via the single reachable math-ai/aime25 mirror
# (the inspect_evals aime2025 source). opencompass AIME2025-I is server-side 500-down,
# so the per-part split is avoided; problems are byte-identical (ubel #567 cross-check),
# so this matches the #638 base denominator's 60-problem set.
YEARS = "2024,2025"
CONC = int(os.environ.get("AIME_CONC", "16"))
# base_fullhead generation_config override is sampling; greedy is forced per-request by T=0.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

ARM_ENV = {
    "FA_SLIDING": "1",
    "SURGICAL_ATTN_USE_3D_OFF": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": str(STOCK),
}


def start_server(log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["PYTHONPATH"] = str(SERVE_INJECT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env["PR557_PATCH_DIR"] = str(SUBMISSION)
    env.update(ARM_ENV)
    cmd = [
        str(SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(STOCK), "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", "16",
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", json.dumps(SAMPLING),
    ]
    print(f"[serve] base_fullhead g32+bf16head mml={MAX_MODEL_LEN} (AIME) stock={STOCK}", flush=True)
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


def run_aime(label: str, out: Path, *, greedy: bool, seed: int, thinking: bool, limit: int = 0) -> dict:
    if greedy:
        samp = ["--temperature", "0.0", "--top-p", "1.0", "--top-k", "-1"]
    else:
        samp = ["--temperature", str(SAMPLING["temperature"]),
                "--top-p", str(SAMPLING["top_p"]), "--top-k", str(SAMPLING["top_k"])]
    # PR #696 fix: the aggregate gate (0.420 = 0.90 x 0.4667) is a THINKING-enabled base
    # denominator, so a no-thinking numerator vs that gate is apples-to-oranges. Protocol is
    # now explicit: --thinking matches the cited 0.4667/0.3500 wall regime; default no-thinking
    # matches the #580 floor regime (base 0.10). Aggregation pairs each with a MATCHED base.
    think_flag = [] if thinking else ["--no-thinking"]
    cmd = [
        str(CLIENT_PY), str(AIME_EVAL),
        # aime_eval.py appends '/v1/chat/completions' to --base-url (unlike the
        # inspect openai-api harness, which wants the '/v1' suffix). Pass the server
        # ROOT here, else the request path doubles to '/v1/v1/...' -> HTTP 404.
        "--base-url", f"http://127.0.0.1:{PORT}", "--model", "gemma-4-e4b-it",
        "--years", YEARS, "--k", "1", *samp,
        "--max-tokens", str(AIME_MT), "--min-tokens", "8", *think_flag,
        "--seed", str(seed), "--client-concurrency", str(CONC),
        "--save-text", "--label", label, "--out", str(out),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[aime] {label} greedy={greedy} seed={seed} limit={limit or 'full'} "
          f"START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[aime] {label} maj@1={d.get('maj_k_accuracy'):.4f} "
          f"correct={d.get('n_correct_maj')}/{d.get('n_problems')} "
          f"extract_fail={d.get('extract_fail_rate')} dt={time.time()-t0:.0f}s", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", action="store_true", help="run the greedy anchor")
    ap.add_argument("--sampled-seeds", default="", help="comma list of #31 sampling seeds")
    ap.add_argument("--thinking", action="store_true",
                    help="enable_thinking=True (the cited 0.4667/0.3500 wall regime). "
                         "Default off = #580 floor regime (no-thinking, base 0.10).")
    ap.add_argument("--smoke", action="store_true", help="2 problems, greedy, then exit")
    args = ap.parse_args()

    tag = "think" if args.thinking else "nothink"
    RES.mkdir(parents=True, exist_ok=True)
    log = HERE / (f"server_aime_smoke_{tag}.log" if args.smoke else f"server_aime_{tag}.log")
    proc = start_server(log)
    try:
        wait_ready(proc)
        print(f"[driver] AIME READY tag={tag} {time.strftime('%H:%M:%S')}", flush=True)
        if args.smoke:
            run_aime(f"aime_smoke_{tag}", RES / f"_smoke_aime_{tag}.json",
                     greedy=True, seed=1234, thinking=args.thinking, limit=2)
            print("[driver] AIME SMOKE OK", flush=True)
            return 0
        if args.greedy:
            out = RES / f"int4body_aime_{tag}_greedy.json"
            if out.exists():
                print(f"[driver] {tag} greedy SKIP existing {out.name}", flush=True)
            else:
                run_aime(f"int4body_aime_{tag}_greedy", out, greedy=True, seed=1234,
                         thinking=args.thinking)
        for s in [int(x) for x in args.sampled_seeds.split(",") if x.strip() != ""]:
            out = RES / f"int4body_aime_{tag}_sampled_s{s}.json"
            if out.exists():
                print(f"[driver] {tag} sampled seed={s} SKIP existing {out.name}", flush=True)
                continue
            run_aime(f"int4body_aime_{tag}_sampled_s{s}", out, greedy=False, seed=s,
                     thinking=args.thinking)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    print(f"[driver] AIME DONE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
