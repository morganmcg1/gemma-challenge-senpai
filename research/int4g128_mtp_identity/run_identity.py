#!/usr/bin/env python
"""Served free-run greedy-identity + TPS probe for an int4 target + MTP-K drafter.

Reuses the local-validation harness primitives so the capture/gate path is the
official one: serve a submission's serve.py (its drafter + attention-group patch),
capture 128x512 greedy decode through the official decode_outputs.py, and compare
the candidate (spec ON) token sequences against the M=1 AR reference (spec OFF,
SENPAI_REFERENCE_MODE=1) of the SAME checkpoint+kernels via the official
greedy_identity verifier.

This isolates speculation as the only removed variable. Reports:
  - freerun_seq_exact   = num_identical / num_compared  (the #566 gate)
  - freerun_token_id    = 1 - total_divergent_tokens / total_tokens_compared
  - divergence onset distribution (early+wide = structural; late+stochastic = FP)
  - local served decode TPS (single-stream) + official proxy = local * tau

It varies env on a single submission dir (int4_mtp_batchinv) so K,
VLLM_BATCH_INVARIANT, DRAFTER_MODEL, MODEL_ID can be swept without editing the
committed manifest. LOCAL ONLY — no HF Job.
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

TAU_LO = 1.035  # banked #594 local->official scalar


def _base_env(model_id: str, drafter: str, batch_invariant: int) -> dict[str, str]:
    return {
        "MODEL_ID": model_id,
        "DRAFTER_MODEL": drafter,
        "VLLM_BATCH_INVARIANT": str(batch_invariant),
        "MAX_NUM_SEQS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "512",
    }


def capture(
    submission: Path,
    server_python: Path,
    out_file: Path,
    *,
    extra_env: dict[str, str],
    num_prompts: int,
    output_len: int,
    port: int,
    probe_tps: bool,
) -> dict:
    """Serve with extra_env, capture decode, optionally probe TPS. Returns summary."""
    log_path = out_file.parent / (out_file.stem + ".server.log")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info: dict = {}
    with harness.LocalServer(
        submission,
        server_python=server_python,
        port=port,
        log_path=log_path,
        extra_env=extra_env,
    ) as srv:
        info["serve_ready_s"] = time.time() - t0
        summary = harness.capture_decode(
            server_python,
            base_url=srv.base_url,
            model=srv.served_model_name,
            out_file=out_file,
            summary_file=out_file.parent / (out_file.stem + ".summary.json"),
            num_prompts=num_prompts,
            output_len=output_len,
            seed=paths.SEED,
        )
        info.update(summary)
        if probe_tps:
            # multiple decode lengths to separate prefill from steady-state decode
            info["tps_probe"] = harness.probe_tps(
                srv.base_url, srv.served_model_name, decode_tokens=output_len
            )
    info["capture_wall_s"] = time.time() - t0
    info["server_log"] = str(log_path)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submissions" / "int4_mtp_batchinv",
                    help="submission dir providing serve.py + attn-group patch")
    ap.add_argument("--model-id", required=True, help="int4 target checkpoint (hub id or local path)")
    ap.add_argument("--drafter", default="/tmp/qat-assistant", help="MTP drafter path/id")
    ap.add_argument("--k", type=int, default=7, help="num_speculative_tokens")
    ap.add_argument("--batch-invariant", type=int, default=1, choices=[0, 1])
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--ref-port", type=int, default=8011)
    ap.add_argument("--cand-port", type=int, default=8012)
    ap.add_argument("--reuse-ref", action="store_true",
                    help="skip reference capture if ref.jsonl already exists")
    ap.add_argument("--skip-tps", action="store_true")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[id] {note}", flush=True)

    manifest = harness.load_manifest(args.submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[id] server_python={server_python}", flush=True)

    run_dir = args.out_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    ref_file = run_dir / "ref.jsonl"
    cand_file = run_dir / "cand.jsonl"

    base = _base_env(args.model_id, args.drafter, args.batch_invariant)

    # --- reference: M=1 AR (spec OFF via SENPAI_REFERENCE_MODE), same kernels ---
    if args.reuse_ref and ref_file.exists():
        print(f"[id] reusing existing reference {ref_file}", flush=True)
        ref_info = json.loads((run_dir / "ref_info.json").read_text()) if (run_dir / "ref_info.json").exists() else {}
    else:
        ref_env = {**base, "SENPAI_REFERENCE_MODE": "1", "NUM_SPECULATIVE_TOKENS": "0"}
        print(f"[id] === REFERENCE capture (M=1 AR, BI={args.batch_invariant}) ===", flush=True)
        ref_info = capture(
            args.submission, server_python, ref_file,
            extra_env=ref_env, num_prompts=args.num_prompts,
            output_len=args.output_len, port=args.ref_port, probe_tps=False,
        )
        (run_dir / "ref_info.json").write_text(json.dumps(ref_info, indent=2, default=str))

    # --- candidate: spec ON, K, same kernels ---
    cand_env = {**base, "NUM_SPECULATIVE_TOKENS": str(args.k)}
    print(f"[id] === CANDIDATE capture (MTP-K{args.k}, BI={args.batch_invariant}) ===", flush=True)
    cand_info = capture(
        args.submission, server_python, cand_file,
        extra_env=cand_env, num_prompts=args.num_prompts,
        output_len=args.output_len, port=args.cand_port, probe_tps=not args.skip_tps,
    )
    (run_dir / "cand_info.json").write_text(json.dumps(cand_info, indent=2, default=str))

    # --- gate: official greedy_identity verifier ---
    report = greedy_gate.compare(ref_file, cand_file)
    rd = report.to_dict()
    onset = greedy_gate.onset_summary(report)
    n_cmp = report.num_prompts_compared or 1
    seq_exact = report.num_identical / n_cmp
    tok_total = report.total_tokens_compared or 1
    tok_id = 1.0 - report.total_divergent_tokens / tok_total

    tps_probe = cand_info.get("tps_probe", {})
    local_tps = tps_probe.get("decode_tps_single_stream")
    proxy_tps = local_tps * TAU_LO if isinstance(local_tps, (int, float)) else None

    result = {
        "label": args.label,
        "model_id": args.model_id,
        "drafter": args.drafter,
        "k": args.k,
        "batch_invariant": args.batch_invariant,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "verdict": report.verdict,
        "freerun_seq_exact": seq_exact,
        "freerun_token_identity": tok_id,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "num_prompts_compared": report.num_prompts_compared,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "onset": {k: onset.get(k) for k in ("onset_min", "onset_median", "onset_max", "num_divergent")},
        "local_decode_tps_single_stream": local_tps,
        "official_proxy_tps": proxy_tps,
        "tau_lo": TAU_LO,
        "beats_126_378": (proxy_tps > 126.378) if isinstance(proxy_tps, (int, float)) else None,
        "tps_probe": tps_probe,
        "ref_server_log": ref_info.get("server_log"),
        "cand_server_log": cand_info.get("server_log"),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 60, flush=True)
    print(f"[RESULT] {args.label}", flush=True)
    print(f"  verdict:              {report.verdict}", flush=True)
    print(f"  freerun_seq_exact:    {seq_exact:.4f}  ({report.num_identical}/{report.num_prompts_compared})", flush=True)
    print(f"  freerun_token_id:     {tok_id:.6f}  ({report.total_divergent_tokens}/{report.total_tokens_compared} divergent)", flush=True)
    print(f"  {greedy_gate.onset_line(onset, args.output_len)}", flush=True)
    print(f"  local_decode_tps:     {local_tps}", flush=True)
    print(f"  official_proxy_tps:   {proxy_tps}  (>126.378: {result['beats_126_378']})", flush=True)
    print(f"  result -> {run_dir / 'result.json'}", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
