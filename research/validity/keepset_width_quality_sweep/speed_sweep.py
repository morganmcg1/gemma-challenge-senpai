#!/usr/bin/env python3
"""Speed leg of the keepset-width sweep (PR #527): warm-median 128x512 served TPS
per lm_head keepset width K, on the DEPLOYED surgical-357 fast stack.

Reuses research/speed/surgical_attn_realize/run_surgical_realize.run_arm VERBATIM
(the harness that produced the 357.06 surgical-357 anchor, W&B j7qao5e9) so every
width is measured apples-to-apples with that anchor (fresh server, 3 back-to-back
128x512 decodes, all-rounds median wall_tps, peak GPU mem).

Only the lm_head keepset width moves. Per width we point the SAME submission
(fa2sw_strict_surgical357) at the pre-baked width-K checkpoint via extra_env
(highest precedence in harness.LocalServer): LM_HEAD_PRUNE=0 (skip the stack's own
12k re-prune; our checkpoint is already width-K) + LOCAL_MODEL_DIR / PCK04_KEEPSET /
PLE_FOLD_TARGET_MODEL -> the baked-K dir. The full deployed fast path (surgical
attn 2D order-preserving, split-KV verify, MTP-K7 spec-dec, onegraph, fa-sliding,
fused sparse argmax, precache) is otherwise identical across widths.

LOCAL A10G measurement + analysis ONLY. analysis_only=true, official_tps=0,
no HF job, no submission/served-file change. Single assigned GPU.

  speed_sweep.py [--widths 12k,16k,32k,full] [--n-decodes 3]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = Path(__file__).resolve().parent
SUBMISSION = "fa2sw_strict_surgical357"

WIDTHS = {
    "12k":  ("/tmp/osoi5-12k-baked", 12288),
    "16k":  ("/tmp/osoi5-v0-baked", 16384),
    "32k":  ("/tmp/osoi5-32k-baked", 32768),
    "full": ("/tmp/osoi5-full-baked", 262144),
}


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="12k,16k,32k,full")
    ap.add_argument("--n-decodes", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=None)
    ap.add_argument("--output-len", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results" / "_speed.json"))
    a = ap.parse_args()

    srz = _load_module("run_surgical_realize",
                       "research/speed/surgical_attn_realize/run_surgical_realize.py")
    harness = srz.harness
    paths = srz.paths

    for note in paths.prepare_local_gpu_env():
        print(f"[speed] {note}", flush=True)

    submission_dir = (ROOT / "submissions" / SUBMISSION).resolve()
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[speed] submission={submission_dir.name} server_python={server_python}", flush=True)

    num_prompts = a.num_prompts or paths.NUM_PROMPTS
    output_len = a.output_len or paths.OUTPUT_LEN
    seed = paths.SEED
    out_dir = (HERE / "results" / "speed").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    want = [w.strip() for w in a.widths.split(",") if w.strip()]
    speed = {}
    if os.path.exists(a.out):
        try:
            speed = json.load(open(a.out))
        except Exception:
            speed = {}

    records_path = out_dir / "speed_records.jsonl"
    t0 = time.time()
    with open(records_path, "a") as fh:
        for w in want:
            baked, K = WIDTHS[w]
            if not os.path.isdir(baked):
                print(f"[speed] {w}: checkpoint {baked} missing -> skip", flush=True)
                continue
            keep = os.path.join(baked, "pck04_keepset.json")
            arm = {
                "name": f"surgical_{w}",
                "label": f"surgical @ K={K} ({w}) deployed fast stack, baked head",
                "extra_env": {
                    "SURGICAL_ATTN_USE_3D_OFF": "1",
                    "LM_HEAD_PRUNE": "0",
                    "LM_HEAD_PRUNE_REQUIRE": "0",
                    "LOCAL_MODEL_DIR": baked,
                    "PLE_FOLD_TARGET_MODEL": baked,
                    "PCK04_KEEPSET": keep,
                },
            }
            # full vocab is not a sparse keepset; the fused-sparse-argmax fast path
            # assumes a pruned head, so disable it for full and fall back to dense argmax.
            if w == "full":
                arm["extra_env"]["FUSED_SPARSE_ARGMAX"] = "0"
            print(f"\n[speed] ===== width {w} K={K} baked={baked} =====", flush=True)
            try:
                rec = srz.run_arm(
                    arm, submission_dir, server_python, out_dir / w,
                    n_decodes=a.n_decodes, num_prompts=num_prompts,
                    output_len=output_len, seed=seed, do_ppl=False, records_fh=fh,
                )
                speed[w] = {
                    "K": K,
                    "warm_median_tps": rec.get("median_wall_tps"),
                    "wall_tps_values": rec.get("wall_tps_values"),
                    "wall_tps_std": rec.get("wall_tps_std"),
                    "peak_gpu_mem_mib": rec.get("peak_gpu_mem_mib"),
                    "server_ready_s": rec.get("server_ready_s"),
                    "completion_full": rec.get("completion_full"),
                    "mechanism": rec.get("mechanism"),
                }
                print(f"[speed] {w} K={K}: warm_median_tps={speed[w]['warm_median_tps']} "
                      f"std={speed[w]['wall_tps_std']} peak_mem={speed[w]['peak_gpu_mem_mib']}MiB", flush=True)
            except Exception as exc:
                print(f"[speed] {w} K={K}: FAILED: {exc!r}", flush=True)
                speed[w] = {"K": K, "warm_median_tps": None, "error": repr(exc)}
            with open(a.out, "w") as f:
                json.dump(speed, f, indent=2)

    print(f"\n[speed] DONE in {time.time()-t0:.0f}s -> {a.out}", flush=True)
    print(json.dumps(speed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
