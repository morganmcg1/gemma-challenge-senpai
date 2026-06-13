"""Run the official decode op-profiler locally on a checkpoint.

Drives ``gemma_decode_profiler_claudecode/profile_graph.py`` (CUDA graphs ON,
real serving config -> clean single-stream TPS + GPU-busy composition) and/or
``profile_eager.py`` (graphs OFF -> faithful per-kernel compute composition).
Both need ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so an in-process torch.profiler
actually captures device kernels; this runner sets it for them.

Each profiler loads the model, profiles a single-stream decode, prints a
categorized breakdown, writes JSON, and exits — there is no server.

    /tmp/server-venv/bin/python -m scripts.local_validation.profile_decode \\
        --model-id google/gemma-4-E4B-it-qat-w4a16-ct --mode both
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import paths

PROFILERS = {
    "graph": (paths.PROFILE_GRAPH, "graph_profile.json"),
    "eager": (paths.PROFILE_EAGER, "profile_breakdown.json"),
}


def run_profiler(
    runner_python: Path, mode: str, *, model_id: str, out_dir: Path, env_extra: dict[str, str] | None = None
) -> dict:
    script, out_name = PROFILERS[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    import os

    env = os.environ.copy()
    env.update(
        {
            "MODEL_ID": model_id,
            "STATE_DIR": str(out_dir),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        }
    )
    env.update(env_extra or {})
    log_path = out_dir / f"profile_{mode}.log"
    print(f"[profile:{mode}] {runner_python} {script} (model={model_id}) -> {out_dir}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        # The profilers sys.exit(0) on success; surface their stdout to our log.
        proc = subprocess.run([str(runner_python), str(script)], env=env, stdout=log, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    result_path = out_dir / out_name
    if proc.returncode != 0 or not result_path.exists():
        tail = "\n".join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(f"profiler '{mode}' failed (rc={proc.returncode}); log tail:\n{tail}")
    data = json.loads(result_path.read_text())
    print(f"[profile:{mode}] done in {dur:.0f}s -> {result_path}", flush=True)
    _print_summary(mode, data)
    return data


def _print_summary(mode: str, data: dict) -> None:
    pct = data.get("category_pct", {})
    if mode == "graph":
        print(f"  graph-mode TPS: {data.get('graph_tps', float('nan')):.2f} tok/s "
              f"(GPU-busy {data.get('gpu_busy_share_of_wall_pct', float('nan')):.1f}% of wall)", flush=True)
    else:
        print(f"  eager TPS: {data.get('eager_tps', float('nan')):.2f} tok/s (absolute eager-inflated)", flush=True)
    for cat, p in sorted(pct.items(), key=lambda x: -x[1]):
        print(f"    {cat:18s} {p:5.1f}%", flush=True)


def _resolve_server_python(arg: Path | None, submission: Path) -> Path:
    """Use the given server-python if it exists, else build/reuse the hash-keyed
    venv from the submission's manifest deps (the documented /tmp/server-venv path
    is a convenience alias; the real venv lives under /tmp/senpai-venvs/<hash>)."""
    from . import harness
    if arg and arg.exists():
        return arg
    manifest = harness.load_manifest(submission)
    py = harness.ensure_server_venv(manifest["dependencies"])
    if arg and arg != py:
        print(f"[profile] --server-python {arg} not found; using resolved venv {py}", flush=True)
    return py


def _run_serving_profile(args) -> int:
    """PR #30 path: profile the *real* served frontier stack (see serve_profile)."""
    from . import serve_profile
    submission = Path(args.submission).resolve()
    out_dir = (args.out_dir or (paths.ROOT / "research" / "profiling" / "frontier_decode")).resolve()
    server_python = _resolve_server_python(args.server_python, submission)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in serve_profile.VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(serve_profile.VARIANTS)}")
    print(f"[profile] serving-stack profile: submission={submission} "
          f"server_python={server_python} variants={variants}", flush=True)
    serve_profile.run(
        submission, server_python, out_dir,
        num_prompts=args.num_prompts, output_len=args.output_len,
        iso_num_prompts=args.iso_num_prompts, iso_output_len=args.iso_output_len,
        kernel_window_tokens=args.kernel_window_tokens, variants=variants,
        do_kernel=not args.no_kernel, wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
    )
    return 0


def _run_attention_detail(args) -> int:
    """PR #39 path: op-level deep-profile of the decode-attention kernel.

    No server, no submission — drives the *real* vLLM Triton attention kernel
    (plus FA2 / SDPA bake-off) on a paged KV cache with L2-defeating rotation,
    and quantifies the 19.6% attention lever vs the HBM-bandwidth floor."""
    from . import profile_attention
    out_path = (Path(args.output) if args.output else
                paths.ROOT / "research" / "profiling" / "fa2sw_attention" / "attention_detail.json")
    fd = paths.ROOT / "research" / "profiling" / "frontier_decode"
    profile_json = fd / "frontier_decode_profile.json"
    traces = sorted(fd.glob("trace_frontier/*.pt.trace.json.gz"))
    trace_path = traces[0] if traces else None
    decode_jsonl = fd / "decode_frontier.jsonl"
    baked = Path("/tmp/osoi5-v0-baked")
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    print(f"[profile] attention-detail: M={m_values} -> {out_path}", flush=True)
    profile_attention.run(
        out_path, m_values, profile_json=profile_json, trace_path=trace_path,
        decode_jsonl=decode_jsonl if decode_jsonl.exists() else None,
        eval_prompts=paths.EVAL_PROMPTS if paths.EVAL_PROMPTS.exists() else None,
        baked_model=baked if baked.exists() else None,
        n_iter=args.attn_iters,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # No-server op-profiler (PR #8): bare in-process LLM, vanilla model.
    ap.add_argument("--model-id", default=paths.INT4_MODEL)
    ap.add_argument("--mode", choices=["graph", "eager", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=None)
    # Attention deep-profile (PR #39): op-level kernel bandwidth bake-off.
    ap.add_argument("--profile-mode", choices=["op", "attention-detail"], default="op",
                    help="'op' = legacy graph/eager profilers; "
                         "'attention-detail' = PR #39 attention kernel microbench")
    ap.add_argument("--M-values", dest="m_values", default="1,7,17,25,45",
                    help="comma list of query widths M for the attention sweep")
    ap.add_argument("--output", default=None,
                    help="output JSON path for --profile-mode attention-detail")
    ap.add_argument("--attn-iters", type=int, default=100,
                    help="profiled iterations per attention op measurement")
    ap.add_argument("--runner-python", type=Path, default=Path(sys.executable),
                    help="python with vLLM (default: current interpreter)")
    # Serving-stack profiler (PR #30): real serve.py + spec-decode + isolation.
    ap.add_argument("--submission", default=None,
                    help="profile this submission's live serving stack (enables PR #30 path)")
    ap.add_argument("--server-python", type=Path, default=None,
                    help="python with the submission's vLLM venv (auto-resolved if absent)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--iso-num-prompts", type=int, default=32,
                    help="prompts for the isolation variants (spec_off/lmhead_off); "
                         "kept small since they only feed verify_gpu_ms p50 + a GEMM trace")
    ap.add_argument("--iso-output-len", type=int, default=256,
                    help="output_len for the isolation variants")
    ap.add_argument("--kernel-window-tokens", type=int, default=256,
                    help="tokens to decode while the torch profiler is recording")
    ap.add_argument("--variants", default="frontier,spec_off,lmhead_off",
                    help="comma list of isolation variants (frontier,spec_off,lmhead_off)")
    ap.add_argument("--no-kernel", action="store_true",
                    help="skip the torch-profiler kernel pass (timing/steptime only)")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="frontier-decode-profile")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[profile] {note}", flush=True)

    if args.profile_mode == "attention-detail":
        return _run_attention_detail(args)

    if args.submission:
        return _run_serving_profile(args)

    out_dir = args.out_dir or (paths.LOCALRUN_ROOT / f"profile-{paths.model_tag(args.model_id)}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    modes = ["graph", "eager"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        results[mode] = run_profiler(args.runner_python, mode, model_id=args.model_id, out_dir=out_dir)
    (out_dir / "profile_index.json").write_text(json.dumps({"model_id": args.model_id, "modes": list(results)}, indent=2))
    print(f"[profile] artifacts in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
