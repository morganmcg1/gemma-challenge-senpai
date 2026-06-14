#!/usr/bin/env python
"""N-run greedy-determinism capture harness for `fa2sw_precache_kenyan` (PR #73).

WHY. program.md (lines 27-28, 324) requires "greedy decode token-identical to
plain greedy autoregressive decode". PR #38 found the deployed frontier stack is
NON-reproducible run-to-run on the A10G (two fresh reloads of the byte-identical
greedy config diverge ~28/32). This harness scales that to N reloads and isolates
the source, so a sibling analyzer (`analyze_determinism.py`) can deliver the
contract verdict: is greedy-identity a BIT-EXACT property the stack can satisfy,
or a DISTRIBUTIONAL one the PPL gate already enforces?

WHAT. Serves the DEPLOYED stack UNCHANGED, N times with fresh reloads, capturing
the greedy spec-ON decode token IDs each reload via the official decode_outputs.py
(sequential, M=1). Source-isolation configs toggle ONE factor at a time, via
extra_env only (NO served-file change):

  default          deployed (FA_SLIDING=1, SPLITKV_VERIFY=1, atomic-add off)
  fa_sliding_off   FA_SLIDING=0                  FA2 sliding-window kernel -> captured graph
  splitkv_off      SPLITKV_VERIFY=0              #43 3D split-KV verify reduction off
  atomic_on        VLLM_MARLIN_USE_ATOMIC_ADD=1  int4 Marlin atomic-add reduction on

--spec-off (PR #114) injects SENPAI_REFERENCE_MODE=1 (drafter OFF, plain M=1 AR) on
top of any config and writes to a sibling ``<config>__specoff/`` dir. Comparing a
``<config>`` spec-ON run to its ``<config>__specoff`` M=1 run with the OFFICIAL
verifier (greedy_gate.compare) IS the self-referential greedy gate: speculation is
the only changed variable, so a DIVERGENT verdict is the M=K+1-verify-vs-M=1-decode
batch-width effect, not an env/precache artifact.

Each fresh serve OPTIONALLY also runs the official sglang TPS bench (--bench) and
the teacher-forced PPL gate (--ppl) against the SAME reload, so one reload yields
token-identity + TPS + PPL together. The FA_SLIDING=0 TPS cost is the load-bearing
unmeasured number from the PR #38 follow-up.

Runs accumulate under a stable --out-root (captures/<config>/run_XX/), so N can be
gathered across several bounded invocations (single GPU, sequential reloads) and
stay inside SENPAI_TIMEOUT_MINUTES per call.

LOCAL ONLY. No HF Job, no submission, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

# Official sglang benchmark protocol (hf_bucket_single_job.py) -- must match
# private_gap_probe.py so the local TPS lines up with the official re-run.
BENCHMARK_DEPENDENCIES = [
    "sglang==0.5.2",
    "transformers==5.9.0",
    "jinja2==3.1.6",
    "pybase64==1.4.3",
    "pydantic==2.13.4",
]
OUTPUT_LEN = paths.OUTPUT_LEN
MAX_CONCURRENCY = 1
REQUEST_RATE = "inf"
WARMUP_REQUESTS = 4
# osoi5-v0-baked carries the unmodified gemma-4-E4B-it tokenizer (vocab 262144);
# use the local copy so capture is hermetic (no Hub fetch). The SAME tokenizer is
# used for every run, so prompt encodings are identical run-to-run and only the
# decode (completion) varies -- which is exactly what we measure.
LOCAL_TOKENIZER = "/tmp/osoi5-v0-baked"

# One changed variable per config. BASE_ENV is shared by all (hermetic + counters).
BASE_ENV = {
    "PRECACHE_BENCH": "0",               # manifest precache path (/harness/...) is absent locally
    "DISABLE_LOG_STATS": "0",            # expose spec_decode_* counters for E_accept
    "VLLM_USE_FLASHINFER_SAMPLER": "0",  # cuRAND-free sampler (does not touch logits/argmax)
}
CONFIGS: dict[str, dict[str, str]] = {
    "default": {},
    "fa_sliding_off": {"FA_SLIDING": "0"},
    "splitkv_off": {"SPLITKV_VERIFY": "0"},
    "atomic_on": {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
}


def _bench_env(bench_python: Path) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(bench_python.parent.parent)
    env["PATH"] = f"{bench_python.parent}{__import__('os').pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_bench(
    bench_python: Path, bench_env: dict[str, str], *, base_url: str, model: str,
    dataset: Path, num_prompts: int, output_len: int, output_file: Path,
    tokenizer: str, timeout_s: int = 3600,
) -> dict[str, Any]:
    """Official sglang bench_serving -> output_throughput is the canonical local `tps`."""
    import subprocess

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bench_python), "-m", "sglang.bench_serving",
        "--backend", "vllm-chat",
        "--base-url", base_url.rstrip("/"),
        "--model", model,
        "--tokenizer", tokenizer,
        "--dataset-name", "sharegpt",
        "--dataset-path", str(dataset),
        "--sharegpt-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(MAX_CONCURRENCY),
        "--request-rate", REQUEST_RATE,
        "--warmup-requests", str(WARMUP_REQUESTS),
        "--seed", str(paths.SEED),
        "--extra-request-body", json.dumps({"ignore_eos": True}),
        "--output-file", str(output_file),
        "--output-details", "--disable-stream", "--disable-tqdm",
    ]
    print("[bench]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=bench_env, timeout=timeout_s, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        err = "\n".join((proc.stderr or "").splitlines()[-12:])
        raise RuntimeError(f"bench_serving rc={proc.returncode}\nSTDOUT:\n{tail}\nSTDERR:\n{err}")
    result = json.loads(output_file.read_text().strip().splitlines()[-1])
    return {
        "tps": result["output_throughput"],
        "completed": result["completed"],
        "duration_s": result["duration"],
        "total_output_tokens": result["total_output_tokens"],
        "mean_e2e_latency_ms": result.get("mean_e2e_latency_ms"),
    }


def scrape_counters(base_url: str) -> dict[str, Any]:
    try:
        return serve_profile.parse_spec_metrics(serve_profile._get_text(f"{base_url}/metrics"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def acceptance_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """E_accept (mean acceptance length) from cumulative spec-decode counter deltas."""
    def d(key: str) -> float | None:
        a, b = after.get(key), before.get(key)
        return None if (a is None or b is None) else a - b

    d_drafts, d_acc, d_dtok = d("num_drafts"), d("num_accepted_tokens"), d("num_draft_tokens")
    out: dict[str, Any] = {}
    if d_drafts and d_acc is not None:
        out["e_accept"] = 1.0 + d_acc / d_drafts
    if d_dtok and d_acc is not None:
        out["accept_rate"] = d_acc / d_dtok
    return out


def capture_run(
    submission: Path, server_python: Path, bench_python: Path, bench_env: dict[str, str],
    out_root: Path, *, config_name: str, run_idx: int, num_prompts: int, output_len: int,
    public: Path, tokenizer: str, do_bench: bool, do_ppl: bool, spec_off: bool = False,
) -> dict[str, Any]:
    """One fresh reload: capture token IDs (+ optional sglang TPS + PPL).

    spec_off=True injects SENPAI_REFERENCE_MODE=1 so a drafter submission's serve.py
    clears SPECULATIVE_CONFIG and serves plain M=1 AR (the canonical greedy reference
    — see scripts/local_validation/gen_greedy_reference.py). Captures land in a
    sibling ``<config>__specoff/`` dir so the spec-ON and spec-OFF runs differ in
    EXACTLY one variable (speculation) under the same BASE_ENV/tokenizer/harness;
    that is the only clean way to read the self-referential greedy gate (PR #114).
    """
    spec_extra = {paths.REFERENCE_MODE_ENV: "1"} if spec_off else {}
    extra_env = {**BASE_ENV, **CONFIGS[config_name], **spec_extra}
    cfg_tag = f"{config_name}__specoff" if spec_off else config_name
    run_dir = out_root / cfg_tag / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    meta: dict[str, Any] = {
        "config": config_name,
        "config_tag": cfg_tag,
        "spec_off": spec_off,
        "run_idx": run_idx,
        "extra_env": extra_env,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "tokenizer": tokenizer,
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    print(f"\n{'='*64}\n[run] config={cfg_tag} idx={run_idx} "
          f"env={ {**CONFIGS[config_name], **spec_extra} or '{}'}\n{'='*64}", flush=True)
    try:
        t_serve = time.time()
        with harness.LocalServer(
            submission, server_python=server_python, port=8000, log_path=log_path,
            extra_env=extra_env, startup_timeout_s=1800,
        ) as srv:
            meta["serve_ready_s"] = round(time.time() - t_serve, 1)
            before = scrape_counters(srv.base_url)
            t0 = time.time()
            decode_summary = harness.capture_decode(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=run_dir / "decode_outputs.jsonl",
                summary_file=run_dir / "decode_summary.json",
                num_prompts=num_prompts, output_len=output_len, tokenizer=tokenizer,
            )
            meta["decode_capture_s"] = round(time.time() - t0, 1)
            meta["decode_num_records"] = decode_summary.get("num_records")
            meta["decode_completion_tokens"] = decode_summary.get("num_completion_tokens")
            meta["decode_duration_s"] = decode_summary.get("duration_s")
            after = scrape_counters(srv.base_url)
            meta["acceptance"] = acceptance_delta(before, after)
            if do_bench:
                meta["bench"] = run_bench(
                    bench_python, bench_env, base_url=srv.base_url, model=srv.served_model_name,
                    dataset=public, num_prompts=num_prompts, output_len=output_len,
                    output_file=run_dir / "bench.jsonl", tokenizer=tokenizer,
                )
                print(f"[run] bench tps={meta['bench']['tps']:.2f} completed={meta['bench']['completed']}", flush=True)
            if do_ppl:
                ppl_summary = harness.run_ppl(
                    bench_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=run_dir / "ppl.jsonl", summary_file=run_dir / "ppl.summary.json",
                )
                meta["ppl"] = ppl_summary.get("ppl")
                meta["ppl_num_tokens"] = ppl_summary.get("num_tokens")
                print(f"[run] ppl={meta['ppl']} num_tokens={meta['ppl_num_tokens']}", flush=True)
        meta["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- record + continue so a chunk survives one bad reload
        meta["ok"] = False
        meta["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[run] ERROR config={config_name} idx={run_idx}: {meta['error']}", flush=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=str(REPO / "submissions/fa2sw_precache_kenyan"))
    ap.add_argument("--config", choices=list(CONFIGS), default="default")
    ap.add_argument("--runs", type=int, default=1, help="number of fresh reloads this invocation")
    ap.add_argument("--start-idx", type=int, default=0, help="first run index (accumulate across calls)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--public", default=str(paths.EVAL_PROMPTS))
    ap.add_argument("--tokenizer", default=LOCAL_TOKENIZER)
    ap.add_argument("--bench", action="store_true", help="also run the sglang TPS bench each reload")
    ap.add_argument("--ppl", action="store_true", help="also run the teacher-forced PPL gate each reload")
    ap.add_argument("--spec-off", action="store_true",
                    help="inject SENPAI_REFERENCE_MODE=1 (drafter OFF, plain M=1 AR reference); "
                         "captures land in <config>__specoff/ so the self-referential greedy gate "
                         "(spec-ON vs this) isolates speculation as the only changed variable (PR #114)")
    ap.add_argument("--out-root", default=str(REPO / "research/validity/greedy_determinism/captures"))
    ap.add_argument("--smoke", action="store_true", help="4 prompts x 64 tok, 1 run -- plumbing check")
    args = ap.parse_args()

    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[gpu] {n}", flush=True)

    submission = Path(args.submission)
    public = Path(args.public)
    tokenizer = args.tokenizer if Path(args.tokenizer).exists() else paths.TOKENIZER
    num_prompts = 4 if args.smoke else args.num_prompts
    output_len = 64 if args.smoke else args.output_len
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[harness] out_root={out_root} config={args.config} spec_off={args.spec_off} "
          f"runs={args.runs} start_idx={args.start_idx} n={num_prompts} olen={output_len} "
          f"bench={args.bench} ppl={args.ppl}", flush=True)

    manifest = harness.load_manifest(submission)
    print("[harness] locating server venv (custom vLLM wheel)", flush=True)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    bench_python = server_python
    bench_env: dict[str, str] = {}
    if args.bench:
        print("[harness] locating bench venv (sglang)", flush=True)
        bench_python = harness.ensure_server_venv(BENCHMARK_DEPENDENCIES)
        bench_env = _bench_env(bench_python)

    metas = []
    for k in range(args.runs):
        idx = args.start_idx + k
        metas.append(capture_run(
            submission, server_python, bench_python, bench_env, out_root,
            config_name=args.config, run_idx=idx, num_prompts=num_prompts,
            output_len=output_len, public=public, tokenizer=tokenizer,
            do_bench=args.bench, do_ppl=args.ppl, spec_off=args.spec_off,
        ))

    ok = sum(1 for m in metas if m.get("ok"))
    print(f"\n[harness] DONE config={args.config}: {ok}/{len(metas)} reloads ok", flush=True)
    for m in metas:
        tps = (m.get("bench") or {}).get("tps")
        print(f"  run_{m['run_idx']:02d} ok={m.get('ok')} ready={m.get('serve_ready_s')}s "
              f"decode={m.get('decode_capture_s')}s tps={tps} ppl={m.get('ppl')} "
              f"E_accept={(m.get('acceptance') or {}).get('e_accept')}", flush=True)
    return 0 if ok == len(metas) else 1


if __name__ == "__main__":
    raise SystemExit(main())
