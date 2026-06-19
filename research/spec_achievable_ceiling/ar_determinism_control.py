#!/usr/bin/env python
"""PR #728 lawine — AR-vs-AR determinism control.

The K-sweep's strict greedy gate reads DIVERGENT for every spec config (seq_exact
~0.18) while the tau=0.3 rescue shows 0 confident genuine flips (all onset gaps are
int4-grid ties). That says the divergence is benign FP, but the sweep INFERS it
rather than demonstrating it. This control closes the loop: serve the SAME AR
config a second time (drafter OFF, SENPAI_REFERENCE_MODE=1, NUM_SPECULATIVE_TOKENS=0,
BI=1) and byte-compare its free-run greedy to the first AR reference (ref.jsonl).

  * AR2-vs-AR1 128/128 identical  -> the M=1 path is run-to-run deterministic, so
    100% of the spec-vs-AR divergence is the M=K+1 verify-vs-M=1 batching effect on
    the un-BI-patched int4 Marlin GEMV (benign ties; project_byteexact_census_finding).
  * AR2-vs-AR1 DIVERGENT at a similar rate -> the base itself carries a run-to-run
    FP floor and the spec config diverges no more than the base does from itself.

Either way the spec config's DIVERGENT verdict is bounded by a benign FP envelope,
not a lossy optimization. LOCAL ONLY: analysis_only=1, official_tps=0, no HF Job.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
MODEL_DIR = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER = "/tmp/qat-assistant"
RUN_DIR = HERE / "runs" / "sweep"          # reuse the sweep's ref.jsonl as AR1
REF1 = RUN_DIR / "ref.jsonl"


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[ctrl] {note}", flush=True)
    if not REF1.exists():
        print(f"[ctrl] ERROR: AR1 reference missing at {REF1}", flush=True)
        return 1

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    print(f"[ctrl] server_python={server_python} vllm={vllm_ver}", flush=True)

    ar_env = {
        "MODEL_ID": MODEL_DIR, "DRAFTER_MODEL": DRAFTER,
        "VLLM_BATCH_INVARIANT": "1", "MAX_NUM_SEQS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0", "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90", "MAX_NUM_BATCHED_TOKENS": "512",
        "SENPAI_REFERENCE_MODE": "1", "NUM_SPECULATIVE_TOKENS": "0",
    }
    out_file = RUN_DIR / "ref2.jsonl"
    summary_file = RUN_DIR / "ref2.summary.json"
    log_path = RUN_DIR / "ref2.server.log"
    t0 = time.time()
    print("[ctrl] === AR2 (identical config to AR1): decode 128x512 ===", flush=True)
    with harness.LocalServer(SUBMISSION, server_python=server_python, port=8023,
                             startup_timeout_s=1800, log_path=log_path, extra_env=ar_env) as srv:
        print(f"[ctrl] ready in {time.time()-t0:.0f}s; decoding 128x512", flush=True)
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=out_file, summary_file=summary_file,
            num_prompts=paths.NUM_PROMPTS, output_len=paths.OUTPUT_LEN, seed=paths.SEED,
        )

    dur = summary.get("duration_s")
    ntok = summary.get("num_completion_tokens")
    wt = (ntok / dur) if dur and ntok else None
    report = greedy_gate.compare(REF1, out_file)
    onset = greedy_gate.onset_summary(report)
    n_cmp = report.num_prompts_compared or 1
    tok_total = report.total_tokens_compared or 1
    result = {
        "pr": 728, "analysis_only": True, "official_tps": 0,
        "control": "AR2_vs_AR1_determinism",
        "vllm_version": vllm_ver,
        "ar2_wall_tps_local": wt,
        "ar1_wall_tps_local": 106.02275748221821,
        "strict_verdict": report.verdict,
        "seq_exact": report.num_identical / n_cmp,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "token_identity": 1.0 - report.total_divergent_tokens / tok_total,
        "total_divergent_tokens": report.total_divergent_tokens,
        "total_tokens_compared": report.total_tokens_compared,
        "onset": {k: onset.get(k) for k in ("onset_min", "onset_median", "onset_max", "num_divergent")},
    }
    (RUN_DIR / "ar_determinism_control.json").write_text(json.dumps(result, indent=2, default=str))
    print("\n" + "=" * 70, flush=True)
    print(f"[PR728 CONTROL] AR2-vs-AR1 (M=1 spec-off, BI=1, vllm={vllm_ver})", flush=True)
    print(f"  ar2_wall_tps={wt} (ar1=106.02)", flush=True)
    print(f"  strict_verdict={report.verdict} seq_exact={result['seq_exact']:.4f} "
          f"token_identity={result['token_identity']:.4f} "
          f"num_divergent={report.num_divergent}/{n_cmp}", flush=True)
    print(f"  onset min/med/max = {onset.get('onset_min')}/{onset.get('onset_median')}/{onset.get('onset_max')}", flush=True)
    print(f"  -> {RUN_DIR / 'ar_determinism_control.json'}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
