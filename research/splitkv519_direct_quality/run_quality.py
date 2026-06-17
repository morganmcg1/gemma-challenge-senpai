#!/usr/bin/env python3
"""PR #568 -- direct served downstream-quality of the EXACT #519 split-KV submission.

Serves ``submissions/fa2sw_strict_byteexact_splitkv399`` (the as-submitted #519
substrate: ``osoi5-v0-baked`` body + 12k pck04 head-prune + MTP K=7 drafter +
FIXED-segment byte-exact split-KV attention) via the local-validation
``LocalServer`` -- i.e. the submission's OWN serve.py with its manifest env -- then
measures MMLU-Pro / GPQA-Diamond / AIME directly against the live endpoint.

LOCAL ONLY. analysis_only. NO HF Job, NO ``--launch``, NO submission, NO served-file
change. The point of the PR: convert #524's *transferred* (byte-exactness-argued)
quality numbers for the #519 config into DIRECTLY-MEASURED ones on the exact
substrate.

  * MMLU-Pro / GPQA  -- ubel #511's inspect_evals greedy harness (run_eval.py), on
    the byte-identical seeded item sets. prompt_sha is recorded so the aggregate can
    assert it equals the banked base run (apples-to-apples).
  * AIME             -- fern #514's harness (aime_eval.py), GREEDY (k=1, T=0), with
    the MANDATORY ``min_tokens=8`` EOS-guard, on the full 60 (2024 + 2025-I + 2025-II).

One server, three axes. Serving overrides touch only HF-Job-only paths (PRECACHE) and
decode concurrency (MAX_NUM_SEQS) -- the #519 attention is fixed-segment byte-exact and
batch-invariant by construction, so per-sequence greedy output is unchanged by batch
size (wirbel #533 measured bs=1==bs=32 on this exact osoi5 substrate).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/splitkv519_direct_quality"
SUBMISSION = ROOT / "submissions/fa2sw_strict_byteexact_splitkv399"
EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")           # inspect_ai / inspect_evals
RUN_EVAL = ROOT / "research/validity/downstream_quality_eval/run_eval.py"
AIME_EVAL = ROOT / "research/downstream_quality_aime/aime_eval.py"


def _ts() -> str:
    return time.strftime("%H:%M:%SZ", time.gmtime())


def run_mmlu_gpqa(task: str, base_url_v1: str, out: Path, *, n: int, max_tokens: int,
                  conc: int, min_tokens: int, limit: int, tag: str) -> dict:
    cmd = [
        str(EVAL_PY), str(RUN_EVAL), "--task", task, "--arm", f"splitkv519{tag}",
        "--out", str(out), "--seed", "12345", "--max-tokens", str(max_tokens),
        "--min-tokens", str(min_tokens), "--max-connections", str(conc),
        "--base-url", base_url_v1, "--model", "gemma-4-e4b-it",
    ]
    if task == "mmlu_pro":
        cmd += ["--n", str(n)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[{_ts()}] === {task} (min_tokens={min_tokens}) ===\n    $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    d = json.load(open(out)) if out.exists() else {}
    d["_rc"] = rc
    print(f"[{_ts()}] {task} rc={rc} acc={d.get('accuracy')} scored={d.get('n_scored')} "
          f"empty={d.get('n_empty')} empty_rate={d.get('empty_rate')}", flush=True)
    return d


def run_aime(base_url: str, out: Path, *, years: str, min_tokens: int, max_tokens: int,
             limit: int, tag: str) -> dict:
    cmd = [
        sys.executable, str(AIME_EVAL), "--base-url", base_url, "--model", "gemma-4-e4b-it",
        "--years", years, "--k", "1", "--temperature", "0.0", "--top-p", "1.0",
        "--top-k", "-1", "--max-tokens", str(max_tokens), "--min-tokens", str(min_tokens),
        "--no-thinking", "--seed", "1234", "--save-text",
        "--label", f"splitkv519_aime_min8{tag}", "--out", str(out),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[{_ts()}] === AIME greedy min_tokens={min_tokens} years={years} ===\n    $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    d = json.load(open(out)) if out.exists() else {}
    d["_rc"] = rc
    print(f"[{_ts()}] AIME rc={rc} maj@1={d.get('maj_k_accuracy')} "
          f"correct={d.get('n_correct_maj')}/{d.get('n_problems')} "
          f"extract_fail_rate={d.get('extract_fail_rate')}", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap each axis to first N (0=full)")
    ap.add_argument("--max-num-seqs", type=int, default=16, help="server decode concurrency override")
    ap.add_argument("--conc", type=int, default=16, help="client connections for MMLU/GPQA")
    ap.add_argument("--mmlu-n", type=int, default=500, help="MMLU-Pro seeded subset size (matched to ubel #511 base)")
    ap.add_argument("--aime-years", default="2024,2025", help="full 60 = 2024(30) + 2025 I+II(30, math-ai/aime25 mirror; opencompass per-part is 500-down)")
    ap.add_argument("--aime-min-tokens", type=int, default=8, help="MANDATORY EOS-guard on AIME")
    ap.add_argument("--skip", default="", help="comma list of axes to skip: mmlu,gpqa,aime")
    ap.add_argument("--tag", default="", help="suffix for arm labels / output files")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    tag = (f"_{args.tag}" if args.tag else "") + ("_smoke" if args.limit else "")
    HERE.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from scripts.local_validation import harness, paths  # noqa: E402

    for note in paths.prepare_local_gpu_env():
        print(f"[setup] {note}", flush=True)
    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    # Serving overrides: disable HF-Job-only precache, raise decode concurrency. The
    # #519 attention is fixed-absolute-segment byte-exact -> M-invariant AND
    # batch-invariant, so these do not move the per-sequence greedy distribution.
    overrides = {
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_519_no_precache.json",
        "MAX_NUM_SEQS": str(args.max_num_seqs),
    }
    log_path = HERE / f"server{tag}.log"
    base_url_v1 = f"http://127.0.0.1:{args.port}/v1"
    base_url = f"http://127.0.0.1:{args.port}"

    summary = {
        "submission": str(SUBMISSION),
        "manifest_name": manifest.get("name"),
        "serve_overrides": overrides,
        "limit": args.limit or None,
        "max_num_seqs": args.max_num_seqs,
        "conc": args.conc,
        "mmlu_n": args.mmlu_n,
        "aime_years": args.aime_years,
        "aime_min_tokens": args.aime_min_tokens,
        "tag": tag,
        "started_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "axes": {},
    }
    print(f"[{_ts()}] serving {SUBMISSION.name} (overrides={overrides}; log {log_path})", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        SUBMISSION,
        server_python=server_python,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        log_path=log_path,
        extra_env=overrides,
    ) as srv:
        print(f"[{_ts()}] READY model={srv.served_model_name} in {time.time()-t0:.0f}s", flush=True)
        if "mmlu" not in skip:
            d = run_mmlu_gpqa("mmlu_pro", base_url_v1, HERE / f"mmlu_pro{tag}.json",
                              n=args.mmlu_n, max_tokens=2048, conc=args.conc,
                              min_tokens=0, limit=args.limit, tag=tag)
            summary["axes"]["mmlu_pro"] = {k: d.get(k) for k in
                                           ("accuracy", "n_scored", "n_correct", "n_empty",
                                            "empty_rate", "n_error", "_rc")}
        if "gpqa" not in skip:
            d = run_mmlu_gpqa("gpqa_diamond", base_url_v1, HERE / f"gpqa{tag}.json",
                              n=args.mmlu_n, max_tokens=3072, conc=args.conc,
                              min_tokens=0, limit=args.limit, tag=tag)
            summary["axes"]["gpqa_diamond"] = {k: d.get(k) for k in
                                               ("accuracy", "n_scored", "n_correct", "n_empty",
                                                "empty_rate", "n_error", "_rc")}
        if "aime" not in skip:
            d = run_aime(base_url, HERE / f"aime_min8{tag}.json",
                         years=args.aime_years, min_tokens=args.aime_min_tokens,
                         max_tokens=3072, limit=args.limit, tag=tag)
            summary["axes"]["aime"] = {k: d.get(k) for k in
                                       ("maj_k_accuracy", "n_correct_maj", "n_problems",
                                        "extract_fail_rate", "_rc")}

    summary["finished_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary["wall_s"] = round(time.time() - t0, 1)
    spath = HERE / f"run_summary{tag}.json"
    spath.write_text(json.dumps(summary, indent=2))
    (HERE / f"DONE{tag}").write_text(summary["finished_at"] + "\n")
    print(f"[{_ts()}] ALL DONE wall={summary['wall_s']}s -> {spath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
