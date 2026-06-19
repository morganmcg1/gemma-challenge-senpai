#!/usr/bin/env python
"""PR #720 -- dual-substrate #319 self-consistency profiler.

Question (#319, program.md L27-28/L311): is a recovery checkpoint's *served-fast*
greedy decode token-identical to its OWN served plain-AR greedy decode? This is
SELF-consistency of the submitted checkpoint, NOT drop-in identity vs the anchor.

Three legs, all single-stream **cc=1** (the official ``decode_outputs.py`` fires
one request at a time) and **BI=1** (``VLLM_BATCH_INVARIANT=1``) per kanna #699's
batch-width caution (cc>1 greedy is corrupt on the substitute venv, so the
faithful reference is cc=1/BI=1):

  ar_ref      = config served on **vLLM 0.22.0 --enforce-eager**, spec-off,
                free-running greedy. The canonical plain-AR reference (BASELINE L10:
                served spec-off, eager = most deterministic AR path).
  dev307_fast = same config on **vLLM 0.22.1rc1.dev307, CUDA-graphs ON** (the
                deployed speed substrate). dev307==ar_ref 128/128 ⇒ the speed
                substrate preserves identity.
  eager_floor = a SECOND independent 0.22.0 --enforce-eager run (determinism
                floor; catches int4 exact-tie run-to-run non-determinism, cf #654).

Verdict (per config), keyed on ``dataset_index`` (seed permutes request order, so
``index`` is not stable; ``dataset_index`` is):
  dev307_fast == ar_ref (all)                       -> RECOVERY_319_SELF_CONSISTENT
  dev307 diverges, eager_floor == ar_ref (all)      -> RECOVERY_319_NEEDS_EAGER
  eager_floor also diverges                          -> RECOVERY_319_SELF_INCONSISTENT

Resumable: a leg whose ``<label>.<leg>.jsonl`` already exists is reused unless
``--force``. Writes ``<out>/result.json`` for a downstream wandb logger.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

V0220 = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")          # vLLM 0.22.0
VDEV307 = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")        # 0.22.1rc1.dev307
SERVED = paths.DEFAULT_SERVED_NAME                                      # gemma-4-e4b-it
PORT = 8127

LEG_SPEC = {
    # label -> (venv_python, enforce_eager)
    "ar_ref": (V0220, True),          # 0.22.0 eager -- canonical plain-AR reference (BASELINE L10)
    "dev307_fast": (VDEV307, False),  # dev307 CUDA-graphs ON -- deployed speed substrate
    "eager_floor": (V0220, True),     # 0.22.0 eager #2 -- determinism floor (int4 tie nondet, #654)
    "dev307_eager": (VDEV307, True),  # dev307 eager -- isolates CUDA-graph break vs version-shift
}


# --------------------------------------------------------------------------- serve
def _gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def _used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        used, _free = out.stdout.strip().splitlines()[0].split(",")
        return int(used)
    except Exception:
        return -1


def preflight(wait_free_mib: int = 18000, timeout_s: int = 120) -> None:
    """Reap stray api_server procs and wait for VRAM to drain before a serve."""
    # api_server is the parent; VLLM::EngineCore is the GPU-holding child that
    # survives if the parent is killed uncleanly (its cmdline doesn't contain the
    # api_server pattern), so reap both or VRAM never drains for the next leg.
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints.openai.api_server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], capture_output=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free = _gpu_free_mib()
        if free < 0 or free >= wait_free_mib:
            return
        time.sleep(3)
    print(f"[preflight] WARN: GPU free {_gpu_free_mib()} MiB < {wait_free_mib} after {timeout_s}s", flush=True)


def serve_env(bi: bool) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # avoid cuRAND JIT (paths.default_native_sampler)
    env["VLLM_BATCH_INVARIANT"] = "1" if bi else "0"
    env.setdefault("HF_HOME", str(Path(os.environ["HOME"]) / ".cache" / "huggingface"))
    return env


def serve(venv_py: Path, model_dir: Path, *, enforce_eager: bool, bi: bool, log_path: Path,
          port: int = PORT, max_model_len: int = 4096, gmu: float = 0.90, mnbt: int = 512,
          startup_timeout_s: int = 1200):
    # Repair the HTTP layer if this venv resolved Starlette>=1 (drops
    # _IncludedRouter.path, 500s /v1/models). Idempotent via per-venv marker;
    # HTTP-only, numerics (logits/greedy decode) unaffected. dev307 needs this.
    harness.ensure_serving_http_compat(venv_py)
    args = [
        str(venv_py), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_dir), "--served-model-name", SERVED,
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16", "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gmu),
        "--trust-remote-code", "--no-enable-log-requests",
        "--max-num-batched-tokens", str(mnbt),
    ]
    if enforce_eager:
        args.append("--enforce-eager")
    env = serve_env(bi)
    log = open(log_path, "w")
    print(f"[serve] {Path(venv_py).parent.parent.name} eager={enforce_eager} bi={bi} cvd={env['CUDA_VISIBLE_DEVICES']} -> {model_dir.name}", flush=True)
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log.flush()
            raise RuntimeError(f"server exited code {proc.returncode} before ready; see {log_path}")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"[serve] ready in {time.time()-t0:.0f}s", flush=True)
                    return proc, base, log
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    _terminate(proc, log)
    raise RuntimeError(f"endpoint not ready after {startup_timeout_s}s; see {log_path}")


def _terminate(proc, log=None) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    if log is not None:
        try:
            log.close()
        except Exception:
            pass


def cudagraph_captured(log_path: Path) -> bool:
    try:
        txt = log_path.read_text(errors="ignore").lower()
    except Exception:
        return False
    return ("capturing cudagraph" in txt) or ("graph capturing finished" in txt) or ("capturing the model" in txt)


# --------------------------------------------------------------- fingerprint / compare
def read_by_dataset_index(jsonl: Path) -> dict:
    out = {}
    for line in Path(jsonl).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[int(r["dataset_index"])] = r["completion_token_sha256"]
    return out


def compare(ref: dict, cur: dict) -> dict:
    keys = sorted(set(ref) & set(cur))
    mism = [k for k in keys if ref[k] != cur[k]]
    return {
        "n_total": len(keys),
        "n_match": len(keys) - len(mism),
        "n_mismatch": len(mism),
        "n_ref_only": len(set(ref) - set(cur)),
        "n_cur_only": len(set(cur) - set(ref)),
        "first_mismatch_dataset_indices": mism[:12],
        "all_match": (len(keys) > 0 and len(mism) == 0),
    }


# --------------------------------------------------------------- batch-invariance health
def _complete(base: str, prompt: str, max_tokens: int, timeout_s: int = 120):
    payload = {"model": SERVED, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": False, "ignore_eos": True, "return_token_ids": True}
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def _out_tokens(resp) -> list:
    ch = (resp.get("choices") or [{}])[0]
    for k in ("token_ids", "completion_token_ids"):
        if isinstance(ch.get(k), list):
            return ch[k]
    return []


def batch_invariance_health(base: str, n: int = 8, max_tokens: int = 64) -> dict:
    """kanna #699 confound control: does the batched (cc=n) greedy path == cc=1?

    Fires n distinct prompts once sequentially (cc=1) and once all-at-once
    (cc=n concurrent). If the engine's batched decode is corrupt (kanna #699),
    the two disagree -- which would let a self-consistency 'break' be
    mis-attributed to the recovery config. We only ever *measure* at cc=1, so
    this is informational hardening, not part of the verdict.
    """
    prompts = [f"In exactly one paragraph, explain idea number {i}: " for i in range(n)]
    seq = []
    for p in prompts:
        seq.append(_out_tokens(_complete(base, p, max_tokens)))
    with ThreadPoolExecutor(max_workers=n) as ex:
        con = list(ex.map(lambda p: _out_tokens(_complete(base, p, max_tokens)), prompts))
    agree = sum(1 for a, b in zip(seq, con) if a == b and len(a) > 0)
    return {"n": n, "max_tokens": max_tokens, "n_agree": agree, "healthy": agree == n}


# ----------------------------------------------------------------------------- legs
def run_leg(label: str, model_dir: Path, out_dir: Path, *, num_prompts: int, output_len: int,
            seed: int, bi: bool, force: bool, health: bool) -> dict:
    venv_py, enforce_eager = LEG_SPEC[label]
    jsonl = out_dir / f"{label}.jsonl"
    summ = out_dir / f"{label}.summary.json"
    log_path = out_dir / f"{label}.serve.log"
    meta_path = out_dir / f"{label}.meta.json"
    if jsonl.exists() and summ.exists() and meta_path.exists() and not force:
        print(f"[leg {label}] reuse existing {jsonl.name}", flush=True)
        return json.loads(meta_path.read_text())

    preflight()
    proc, base, log = serve(venv_py, model_dir, enforce_eager=enforce_eager, bi=bi, log_path=log_path)
    meta = {"label": label, "venv": Path(venv_py).parent.parent.name, "enforce_eager": enforce_eager,
            "bi": bi, "num_prompts": num_prompts, "output_len": output_len, "seed": seed}
    try:
        time.sleep(2)
        meta["cudagraph_captured"] = cudagraph_captured(log_path)
        if health:
            try:
                meta["batch_invariance"] = batch_invariance_health(base)
            except Exception as e:  # health is best-effort
                meta["batch_invariance"] = {"error": str(e)}
        t0 = time.time()
        summary = harness.capture_decode(
            venv_py, base_url=base, model=SERVED, out_file=jsonl, summary_file=summ,
            num_prompts=num_prompts, output_len=output_len, seed=seed,
            timeout_s=max(600, int(num_prompts * output_len / 8) + 600),
        )
        meta["decode_wall_s"] = round(time.time() - t0, 1)
        meta["tps"] = summary.get("tps")
        meta["completed"] = summary.get("completed")
        meta["gpu_used_mib_peak"] = _used_mib()
    finally:
        _terminate(proc, log)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def classify(present: dict) -> dict:
    """Full 4-leg comparison matrix + PR-headline verdict + break attribution.

    Legs (any subset present):
      ar_ref       0.22.0 eager  -- canonical plain-AR reference
      dev307_fast  dev307 graphs -- deployed speed substrate
      eager_floor  0.22.0 eager  -- determinism floor (rerun of ar_ref's engine)
      dev307_eager dev307 eager  -- same deployed build, graphs OFF

    Comparisons:
      dev307_vs_ar          dev307_fast vs ar_ref  (PR primary: speed substrate vs canonical AR)
      floor_vs_ar           eager_floor vs ar_ref  (0.22.0 run-to-run determinism)
      dev307graph_vs_eager  dev307_fast vs dev307_eager (PURE CUDA-graph effect, same build)
      dev307eager_vs_ar     dev307_eager vs ar_ref (PURE 0.22.0-vs-dev307 version effect)

    The PURE-graph and PURE-version splits disambiguate *why* dev307_fast diverges
    from the 0.22.0-eager reference: a cross-version reference can make a config
    look non-self-consistent when, on its own deployed engine (dev307), served-fast
    greedy actually equals dev307 plain-AR (graphs preserve identity) -- the gap is
    purely the 0.22.0->dev307 build change, not the submission.
    """
    legs = {k: read_by_dataset_index(present[k]) for k in LEG_SPEC if k in present}
    res = {}

    def cmp_if(a, b, key):
        if a in legs and b in legs:
            res[key] = compare(legs[b], legs[a])  # compare(ref=b, cur=a)

    cmp_if("dev307_fast", "ar_ref", "dev307_vs_ar")
    cmp_if("eager_floor", "ar_ref", "floor_vs_ar")
    cmp_if("dev307_fast", "dev307_eager", "dev307graph_vs_eager")
    cmp_if("dev307_eager", "ar_ref", "dev307eager_vs_ar")

    # PR-headline verdict: keyed on dev307_fast vs the canonical 0.22.0-eager AR ref.
    verdict = "INCOMPLETE"
    if "dev307_vs_ar" in res:
        if res["dev307_vs_ar"]["all_match"]:
            verdict = "RECOVERY_319_SELF_CONSISTENT"
        elif "floor_vs_ar" in res:
            verdict = ("RECOVERY_319_NEEDS_EAGER" if res["floor_vs_ar"]["all_match"]
                       else "RECOVERY_319_SELF_INCONSISTENT")
        else:
            verdict = "RECOVERY_319_DEV307_DIVERGES_FLOOR_PENDING"
    res["verdict"] = verdict

    # Break attribution: when dev307_fast diverges from the 0.22.0-eager ref, is it
    # the CUDA graphs, the 0.22.0->dev307 version change, or intrinsic tie nondet?
    attribution = None
    if "dev307_vs_ar" in res and not res["dev307_vs_ar"]["all_match"]:
        causes = []
        if "dev307graph_vs_eager" in res and not res["dev307graph_vs_eager"]["all_match"]:
            causes.append("cuda_graphs")
        if "dev307eager_vs_ar" in res and not res["dev307eager_vs_ar"]["all_match"]:
            causes.append("version_0220_vs_dev307")
        if "floor_vs_ar" in res and not res["floor_vs_ar"]["all_match"]:
            causes.append("intrinsic_tie_nondet")
        attribution = causes or ["unresolved"]
    res["break_attribution"] = attribution

    # Deployment-relevant self-consistency: on the DEPLOYED engine (dev307), does
    # served-fast (graphs) == that engine's own plain-AR (eager)? This is the
    # truest #319 reading for a submission that ships on dev307.
    if "dev307graph_vs_eager" in res:
        res["dev307_self_consistent"] = res["dev307graph_vs_eager"]["all_match"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", type=Path, required=True)
    ap.add_argument("--label", required=True, help="config label, e.g. g32_locus")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--legs", default="ar_ref,dev307_fast", help="comma list subset of ar_ref,dev307_fast,eager_floor")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-bi", action="store_true", help="disable VLLM_BATCH_INVARIANT (default on)")
    ap.add_argument("--no-health", action="store_true", help="skip the batch-invariance health check")
    ap.add_argument("--force", action="store_true", help="re-run legs even if jsonl exists")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # inherited launch value is a stale host index
    args.out_dir.mkdir(parents=True, exist_ok=True)
    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    assert all(l in LEG_SPEC for l in legs), f"bad legs {legs}"
    bi = not args.no_bi

    metas = {}
    present = {}
    for leg in legs:
        m = run_leg(leg, args.config_dir, args.out_dir, num_prompts=args.num_prompts,
                    output_len=args.output_len, seed=args.seed, bi=bi, force=args.force,
                    health=not args.no_health)
        metas[leg] = m
        present[leg] = args.out_dir / f"{leg}.jsonl"

    cls = classify(present)
    result = {
        "label": args.label, "config_dir": str(args.config_dir),
        "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed,
        "bi": bi, "legs": legs, "legs_meta": metas, **cls,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("\n===== RESULT =====", flush=True)
    print(json.dumps({k: result[k] for k in ("label", "verdict", "break_attribution", "dev307_self_consistent") if k in result}, indent=2), flush=True)
    for k in ("dev307_vs_ar", "floor_vs_ar", "dev307graph_vs_eager", "dev307eager_vs_ar"):
        if k in result:
            c = result[k]
            print(f"  {k}: {c['n_match']}/{c['n_total']} match (mismatch {c['n_mismatch']})", flush=True)
    for leg, m in metas.items():
        print(f"  leg {leg}: tps={m.get('tps')} wall={m.get('decode_wall_s')}s cudagraph={m.get('cudagraph_captured')} health={m.get('batch_invariance',{}).get('healthy') if isinstance(m.get('batch_invariance'),dict) else None}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
