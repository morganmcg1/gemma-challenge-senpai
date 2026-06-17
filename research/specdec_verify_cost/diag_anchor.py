#!/usr/bin/env python
"""PR #575 wirbel — anchor diagnostic (NO probe): what does 252.69 mean?

Serves base_fullhead three ways and reports warm-median served TPS (output_len=256,
MAX_NUM_SEQS=1), with NO mstep probe attached, to ground-truth the 252.69 anchor:
  - nospec : SPECULATIVE_CONFIG=""  -> true 1/C(1) no-spec single-token rate
  - mtp    : the ship's MTP K=7 drafter (the #553 base_fullhead serve)
  - ngram7 : ngram K=7 (the sweep's drafter) -> ngram acceptance on this substrate

If mtp ~= 252.69 and nospec ~= 88-144, then 252.69 is a SPEC number (A/C(8)), not
1/C(1), and the PR's serve-equiv identity needs reinterpretation.

Run: CUDA_VISIBLE_DEVICES=0 python research/specdec_verify_cost/diag_anchor.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
MEASURE_FLOOR = ROOT / "research" / "base_int4_floor_tps" / "measure_floor.py"
MODEL_DIR = (
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
SEED = 1


def base_fullhead_env() -> dict[str, str]:
    return {
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",
        "LOCAL_MODEL_DIR": MODEL_DIR,
        "PLE_FOLD_TARGET_MODEL": MODEL_DIR,
        "PLE_FOLD_EMBED_SCALE": "1",
        "LM_HEAD_FULL_REQUIRE": "1",
        "GPU_MEMORY_UTILIZATION": "0.92",
        "MAX_NUM_SEQS": "1",
        # zero MTP-specific REQUIRE guards so non-MTP configs boot cleanly
        "LOOPGRAPH_REQUIRE_CAPTURE": "0",
        "FUSED_SPARSE_ARGMAX_REQUIRE": "0",
        "DIXIE_FUSED_ACCEPT_PREP_REQUIRE": "0",
        "PRECACHE_REQUIRE": "0",
    }


def spec_config(kind: str) -> str:
    if kind == "nospec":
        return ""
    if kind == "mtp":
        # the ship's default drafter (manifest) on the base_fullhead substrate
        return json.dumps({"method": "mtp", "model": "/tmp/qat-assistant", "num_speculative_tokens": 7})
    if kind == "ngram7":
        return json.dumps({"method": "ngram", "num_speculative_tokens": 7,
                           "prompt_lookup_max": 4, "prompt_lookup_min": 1})
    raise ValueError(kind)


def _decode_worker(args: argparse.Namespace) -> int:
    from scripts.local_validation import paths
    spec = importlib.util.spec_from_file_location("official_decode", str(paths.DECODE_SCRIPT))
    od = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(od)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = od.read_sharegpt_prompts(Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed)
    rows = []
    for i, record in enumerate(records):
        t0 = time.perf_counter()
        pids = od.encode_prompt(tok, record["prompt_text"])
        t1 = time.perf_counter()
        resp = od.request_decode(base_url=args.base_url, model=args.model,
                                 prompt_token_ids=pids, output_len=args.output_len,
                                 timeout_s=args.request_timeout_s)
        t2 = time.perf_counter()
        choice = od.choice_from_response(resp)
        comp, _, _ = od.extract_generated_token_ids(resp, choice, pids)
        rows.append({"index": i, "t_tokenize_s": t1 - t0, "t_request_s": t2 - t1,
                     "num_prompt_tokens": len(pids), "num_completion_tokens": len(comp)})
        print(f"[worker] {i+1}/{len(records)} req_ms={1000*(t2-t1):.1f} comp={len(comp)} prompt={len(pids)}", flush=True)
    Path(args.out_file).write_text(json.dumps(
        {"output_len": args.output_len, "num_records": len(records), "per_request": rows}))
    return 0


def serve_and_measure(mf, harness, paths, *, server_python, kind, num_prompts, output_len, port):
    label = f"diag_{kind}"
    log_path = HERE / f"{label}.log"
    pass_file = HERE / f"{label}_pass.json"
    extra_env = base_fullhead_env()
    extra_env["SPECULATIVE_CONFIG"] = spec_config(kind)
    # NO probe: MSTEP unset, no PYTHONPATH injection.

    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH','')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"

    peak = {"mib": 0.0}; stop = threading.Event()
    def _vram():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
                vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
                if vals: peak["mib"] = max(peak["mib"], max(vals))
            except (OSError, subprocess.SubprocessError): pass
            stop.wait(2.0)
    sampler = threading.Thread(target=_vram, daemon=True); sampler.start()
    res: dict[str, Any] = {"kind": kind, "spec_config": extra_env["SPECULATIVE_CONFIG"]}
    try:
        with harness.LocalServer(SUBMISSION, server_python=server_python, port=port,
                                 startup_timeout_s=1800, log_path=log_path, extra_env=extra_env) as srv:
            print(f"[diag] [{kind}] warming ({srv.base_url})", flush=True)
            mf._warm_server(srv.base_url, srv.served_model_name, n=mf.WARMUP_REQUESTS)
            cmd = [str(server_python), str(Path(__file__).resolve()), "--decode-worker",
                   "--base-url", srv.base_url, "--model", srv.served_model_name,
                   "--dataset-path", str(paths.EVAL_PROMPTS), "--tokenizer", paths.TOKENIZER,
                   "--num-prompts", str(num_prompts), "--output-len", str(output_len),
                   "--seed", str(SEED), "--out-file", str(pass_file), "--request-timeout-s", "600"]
            print(f"[diag] [{kind}] decode {num_prompts}x{output_len}", flush=True)
            subprocess.run(cmd, check=True, timeout=5400, env=worker_env)
            summary = json.loads(pass_file.read_text())
            try:
                res["tps"] = mf._aggregate(summary)
            except Exception as exc:
                res["tps"] = {"warm_median_tps": float("nan"), "err": repr(exc)}
    finally:
        stop.set(); sampler.join(timeout=5)
    res["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="nospec,mtp,ngram7")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer"); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-file"); ap.add_argument("--request-timeout-s", type=int, default=600)
    args = ap.parse_args(argv)
    if args.decode_worker:
        return _decode_worker(args)

    mf_spec = importlib.util.spec_from_file_location("measure_floor", str(MEASURE_FLOOR))
    mf = importlib.util.module_from_spec(mf_spec); assert mf_spec and mf_spec.loader
    mf_spec.loader.exec_module(mf)
    from scripts.local_validation import harness, paths
    for note in paths.prepare_local_gpu_env():
        print(f"[diag] {note}", flush=True)
    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    results = []
    for kind in [k for k in args.kinds.split(",") if k.strip()]:
        r = serve_and_measure(mf, harness, paths, server_python=server_python, kind=kind,
                              num_prompts=args.num_prompts, output_len=args.output_len, port=args.port)
        wm = r.get("tps", {}).get("warm_median_tps")
        print(f"[diag] >>> {kind}: warm_median_tps={wm} peak={r.get('peak_vram_gb',0):.2f}GB "
              f"spec={r['spec_config'][:60]!r}", flush=True)
        results.append(r)
    (HERE / "diag_anchor_result.json").write_text(json.dumps(results, indent=2, default=str))
    print("\n===== ANCHOR DIAGNOSTIC =====", flush=True)
    for r in results:
        print(f"  {r['kind']:>8}: warm_median_tps={r.get('tps',{}).get('warm_median_tps')}", flush=True)
    print("  (252.69 = wirbel #553 base_fullhead anchor)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
