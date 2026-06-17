#!/usr/bin/env python
"""Reference self-determinism probe for the greedy-identity measurement method.

Before any ref-vs-candidate divergence number can be trusted, the M=1 AR
reference must reproduce. The base_fullhead #572 puzzle was partly a *noisy
reference*: the comparison method's own self-determinism floor was only 0.508
seq, so the 0.4777 candidate "divergence" was confounded by reference noise.

This probe captures the SAME M=1 AR (spec-OFF, BI=1) decode three ways and
gates the copies against each other:
  - within-server : two captures from one served process (refA1 vs refA2).
                    Isolates pure decode determinism (same CUDA context/alloc).
  - cross-server  : a capture from a second, freshly launched process (refB).
                    Isolates run-to-run determinism the way the real gate does
                    (ref process != cand process: different context/alloc).

If within-server seq_exact < 1.0, decode itself is non-deterministic and EVERY
divergence number inherits that floor. If within-server == 1.0 but cross-server
< 1.0, the noise lives in process/context setup, not the spec stack. If both ==
1.0, the reference is solid and any ref-vs-candidate divergence is a REAL
property of the candidate (spec verify), not measurement noise.

LOCAL ONLY — no HF Job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402


def _ref_env(model_id: str, drafter: str, batch_invariant: int) -> dict[str, str]:
    return {
        "MODEL_ID": model_id,
        "DRAFTER_MODEL": drafter,
        "VLLM_BATCH_INVARIANT": str(batch_invariant),
        "MAX_NUM_SEQS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "512",
        "SENPAI_REFERENCE_MODE": "1",
        "NUM_SPECULATIVE_TOKENS": "0",
    }


def _capture(srv, server_python, out_file: Path, num_prompts: int, output_len: int) -> None:
    harness.capture_decode(
        server_python,
        base_url=srv.base_url,
        model=srv.served_model_name,
        out_file=out_file,
        summary_file=out_file.parent / (out_file.stem + ".summary.json"),
        num_prompts=num_prompts,
        output_len=output_len,
        seed=paths.SEED,
    )


def _gate(a: Path, b: Path) -> dict:
    rep = greedy_gate.compare(a, b)
    n = rep.num_prompts_compared or 1
    tok = rep.total_tokens_compared or 1
    return {
        "verdict": rep.verdict,
        "seq_exact": rep.num_identical / n,
        "token_identity": 1.0 - rep.total_divergent_tokens / tok,
        "num_identical": rep.num_identical,
        "num_prompts_compared": rep.num_prompts_compared,
        "total_divergent_tokens": rep.total_divergent_tokens,
        "total_tokens_compared": rep.total_tokens_compared,
        "onset": greedy_gate.onset_summary(rep),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submissions" / "int4_mtp_batchinv")
    ap.add_argument("--model-id", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--drafter", default="/tmp/qat-assistant")
    ap.add_argument("--batch-invariant", type=int, default=1, choices=[0, 1])
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--label", default="selfdet_qat_bi1_n16")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--port-a", type=int, default=8021)
    ap.add_argument("--port-b", type=int, default=8022)
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[selfdet] {note}", flush=True)

    manifest = harness.load_manifest(args.submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[selfdet] server_python={server_python}", flush=True)

    run_dir = args.out_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    refA1 = run_dir / "refA1.jsonl"
    refA2 = run_dir / "refA2.jsonl"
    refB = run_dir / "refB.jsonl"
    env = _ref_env(args.model_id, args.drafter, args.batch_invariant)

    # --- server A: two captures (within-server determinism) ---
    print("[selfdet] === server A: refA1, refA2 (within-server) ===", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        args.submission, server_python=server_python, port=args.port_a,
        log_path=run_dir / "serverA.log", extra_env=env,
    ) as srv:
        print(f"[selfdet] server A ready in {time.time()-t0:.0f}s", flush=True)
        _capture(srv, server_python, refA1, args.num_prompts, args.output_len)
        _capture(srv, server_python, refA2, args.num_prompts, args.output_len)

    # --- server B: one capture (cross-server determinism) ---
    print("[selfdet] === server B: refB (cross-server) ===", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        args.submission, server_python=server_python, port=args.port_b,
        log_path=run_dir / "serverB.log", extra_env=env,
    ) as srv:
        print(f"[selfdet] server B ready in {time.time()-t0:.0f}s", flush=True)
        _capture(srv, server_python, refB, args.num_prompts, args.output_len)

    within = _gate(refA1, refA2)
    cross = _gate(refA1, refB)
    result = {
        "label": args.label,
        "model_id": args.model_id,
        "drafter": args.drafter,
        "batch_invariant": args.batch_invariant,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "within_server": within,
        "cross_server": cross,
    }
    (run_dir / "selfdet_result.json").write_text(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 60, flush=True)
    print(f"[SELFDET] {args.label}", flush=True)
    print(f"  within-server seq_exact: {within['seq_exact']:.4f}  ({within['num_identical']}/{within['num_prompts_compared']})  tok_id={within['token_identity']:.6f}  verdict={within['verdict']}", flush=True)
    print(f"  cross-server  seq_exact: {cross['seq_exact']:.4f}  ({cross['num_identical']}/{cross['num_prompts_compared']})  tok_id={cross['token_identity']:.6f}  verdict={cross['verdict']}", flush=True)
    print(f"  result -> {run_dir / 'selfdet_result.json'}", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
