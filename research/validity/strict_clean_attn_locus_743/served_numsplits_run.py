#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #755 lawine -- run the SERVED num_splits probe (instruction 1).

Serve the publishable-K4-BI1 config two ways on the faithful vLLM 0.22.0 engine,
both with the read-only num_splits probe loaded via the submission sitecustomize
(env-gated SENPAI_NUMSPLITS_PROBE), and compare the kernel's actual ``num_splits``
(= num_segments) for:
  * AR reference  (drafter OFF) -> the TARGET **M=1 decode** forward
  * spec K=4      (drafter ON)  -> the TARGET **M=K+1 verify** forward
plus the live ``is_batch_invariant`` flag in the worker. This pins whether the
served path's verify/decode actually share one reduction order under BI=1 (=> the
#752 strict 24/128 cannot be num_splits, look elsewhere) or split apart (=> land
#743's num_splits=1 fix is the cure).

LOCAL A10G only. analysis_only=1. NO HF Job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "research" / "spec_achievable_ceiling"
for p in (str(SPEC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_sweep as rs  # noqa: E402
from run_pubdrafter_sweep import PUB_DRAFTER  # noqa: E402
from scripts.local_validation import paths  # noqa: E402


def _read_probe(probe_path: Path) -> dict:
    """Read the per-PID probe dumps and keep the one with the most attention calls
    (the EngineCore worker; the api_server front-end dumps calls=0 and must not
    clobber it). Falls back to a legacy single-path dump if present."""
    best: dict = {}
    best_calls = -1
    for f in sorted(probe_path.parent.glob(probe_path.name + ".pid*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        calls = d.get("n_attention_calls") or 0
        if calls > best_calls:
            best, best_calls = d, calls
    if not best and probe_path.exists():
        try:
            best = json.loads(probe_path.read_text())
        except Exception:
            best = {}
    return best


def serve_with_probe(server_python, *, label, run_dir, base, extra, probe_path, port,
                     num_prompts, output_len):
    # clear stale pid dumps so we only read this run's worker
    for f in probe_path.parent.glob(probe_path.name + ".pid*.json"):
        try:
            f.unlink()
        except OSError:
            pass
    env = {**base, **extra,
           "SENPAI_PR755_DIR": str(HERE),
           "SENPAI_NUMSPLITS_PROBE": str(probe_path)}
    info = rs.serve_capture(
        rs.SUBMISSION, server_python, label=label, run_dir=run_dir, extra_env=env,
        port=port, num_prompts=num_prompts, output_len=output_len,
        do_ppl=False, do_logprobs=False, ref_recs=None, startup_timeout_s=1800,
    )
    probe = _read_probe(probe_path)
    return info, probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4, help="spec width (publishable rung K=4)")
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--drafter", default=PUB_DRAFTER)
    ap.add_argument("--model-id", default=rs.MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs" / "numsplits_probe")
    args = ap.parse_args()

    run_dir = args.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[probe] {note}", flush=True)

    from scripts.local_validation import harness
    manifest = harness.load_manifest(rs.SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    print(f"[probe] server_python={server_python} vllm={vllm_ver}", flush=True)

    base = rs.base_env(args.model_id, args.drafter, batch_invariant=1)
    t0 = time.time()

    # --- AR reference: TARGET M=1 decode ---
    print(f"\n[probe] === AR ref (drafter OFF, BI=1): TARGET M=1 decode, {args.num_prompts}x{args.output_len} ===", flush=True)
    ref_info, ref_probe = serve_with_probe(
        server_python, label="probe_ref", run_dir=run_dir,
        base=base, extra={"SENPAI_REFERENCE_MODE": "1", "NUM_SPECULATIVE_TOKENS": "0"},
        probe_path=run_dir / "probe_ref.numsplits.json", port=8031,
        num_prompts=args.num_prompts, output_len=args.output_len)

    # --- spec K: TARGET M=K+1 verify ---
    print(f"\n[probe] === SPEC K={args.k} (drafter ON, BI=1): TARGET M={args.k+1} verify ===", flush=True)
    spec_info, spec_probe = serve_with_probe(
        server_python, label=f"probe_spec_k{args.k}", run_dir=run_dir,
        base=base, extra={"NUM_SPECULATIVE_TOKENS": str(args.k)},
        probe_path=run_dir / f"probe_spec_k{args.k}.numsplits.json", port=8032,
        num_prompts=args.num_prompts, output_len=args.output_len)

    def buckets(probe):
        return probe.get("buckets", {}) if isinstance(probe, dict) else {}

    report = {
        "pr": 755, "analysis_only": True, "official_tps": 0,
        "vllm_version": vllm_ver, "k": args.k,
        "num_prompts": args.num_prompts, "output_len": args.output_len,
        "ar_ref": {
            "is_batch_invariant_live": ref_probe.get("is_batch_invariant_live"),
            "n_attention_calls": ref_probe.get("n_attention_calls"),
            "wall_tps": ref_info.get("wall_tps"),
            "buckets": buckets(ref_probe),
        },
        "spec": {
            "is_batch_invariant_live": spec_probe.get("is_batch_invariant_live"),
            "n_attention_calls": spec_probe.get("n_attention_calls"),
            "wall_tps": spec_info.get("wall_tps"),
            "buckets": buckets(spec_probe),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    def show(name, side):
        print(f"\n[{name}] is_batch_invariant_live={side['is_batch_invariant_live']} "
              f"calls={side['n_attention_calls']} wall_tps={side['wall_tps']}", flush=True)
        for key, b in sorted(side["buckets"].items()):
            print(f"    {key:40s} count={b['count']:>7} "
                  f"seqlen_k=[{b.get('seqlen_k_min')}..{b.get('seqlen_k_max')}] "
                  f"num_seqs=[{b.get('num_seqs_min')}..{b.get('num_seqs_max')}]", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"[PR755 num_splits PROBE] vllm={vllm_ver} BI=1 conc=1 K={args.k}", flush=True)
    show("AR ref  (TARGET M=1 decode)", report["ar_ref"])
    show(f"spec K={args.k} (TARGET M={args.k+1} verify)", report["spec"])
    print(f"\n  report -> {run_dir / 'report.json'}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
