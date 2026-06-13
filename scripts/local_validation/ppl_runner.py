"""Local PPL runner — wrap the official ppl_endpoint.py with correct headroom.

Scores the fixed ``ppl_ground_truth_tokens.jsonl`` continuations under a served
endpoint. Two modes:

  * ``--base-url URL``  : score an endpoint you already have running.
  * ``--submission DIR``: serve the submission locally (with the PPL memory
                          headroom env applied) and then score it.

The headroom env (``MAX_NUM_BATCHED_TOKENS=512``, ``GPU_MEMORY_UTILIZATION=0.90``,
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``) bounds the full-vocab
float32 ``log_softmax`` peak that ``prompt_logprobs`` materialises, so the long
ground-truth records don't OOM. A correctly served bf16 E4B scores PPL ≈ 2.30;
the int4 QAT base ≈ 2.01.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import harness, paths

HEADROOM_ENV = {
    "MAX_NUM_BATCHED_TOKENS": "512",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _headroom_overrides(manifest_env: dict) -> dict[str, str]:
    """Fill in PPL headroom defaults only where the manifest hasn't set them."""
    return {k: v for k, v in HEADROOM_ENV.items() if k not in (manifest_env or {})}


def score_endpoint(
    base_url: str,
    model: str,
    *,
    out_dir: Path,
    dataset: Path | None = None,
    runner_python: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = harness.run_ppl(
        runner_python or Path(sys.executable),
        base_url=base_url,
        model=model,
        out_file=out_dir / "ppl_results.jsonl",
        summary_file=out_dir / "ppl_summary.json",
        dataset=dataset,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base-url", help="score an already-running endpoint")
    src.add_argument("--submission", type=Path, help="serve this submission dir, then score it")
    ap.add_argument("--model", default=paths.DEFAULT_SERVED_NAME, help="served model name (for --base-url)")
    ap.add_argument("--dataset", type=Path, default=None, help="PPL ground-truth jsonl (default: official mirror)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir for ppl_summary.json")
    ap.add_argument("--server-python", type=Path, default=None, help="python with vLLM (default: build from manifest deps)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-ensure-headroom", dest="ensure_headroom", action="store_false")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[ppl] {note}", flush=True)

    dataset = args.dataset or paths.ppl_dataset()
    out_dir = args.out_dir or (paths.LOCALRUN_ROOT / f"ppl-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")

    if args.base_url:
        summary = score_endpoint(args.base_url, args.model, out_dir=out_dir, dataset=dataset)
        print(f"PPL={summary['ppl']:.4f} num_tokens={summary['num_tokens']} -> {out_dir/'ppl_summary.json'}", flush=True)
        return 0

    # Serve-then-score mode.
    manifest = harness.load_manifest(args.submission)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    overrides = _headroom_overrides(manifest.get("env", {})) if args.ensure_headroom else {}
    if overrides:
        print(f"[ppl] injecting headroom env: {overrides}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "server.log"
    with harness.LocalServer(
        args.submission, server_python=server_python, port=args.port, log_path=log, extra_env=overrides
    ) as srv:
        summary = score_endpoint(
            srv.base_url, srv.served_model_name, out_dir=out_dir, dataset=dataset, runner_python=server_python
        )
    print(f"PPL={summary['ppl']:.4f} num_tokens={summary['num_tokens']} model={summary['model']} -> {out_dir/'ppl_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
