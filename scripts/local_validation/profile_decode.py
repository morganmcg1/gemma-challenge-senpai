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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=paths.INT4_MODEL)
    ap.add_argument("--mode", choices=["graph", "eager", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--runner-python", type=Path, default=Path(sys.executable),
                    help="python with vLLM (default: current interpreter)")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[profile] {note}", flush=True)

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
