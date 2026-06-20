"""Determinism control for PR #792: re-run the ctk32 CONTROL config a second time
(decode-only) and compare token-for-token to the first ctk32 run.

WHY: the sweep found ctk64/ctk128 are NOT byte-identical to the ctk32 control
(9/10 of 128 prompts diverge). The PR's primary premise is that widening the
drafter's centroid_top_k is output-neutral under greedy verify, so any
divergence must be explained. Two candidate causes:
  (A) drafter-induced: different proposals -> different M=7 verify-GEMM batch
      composition -> ULP-level argmax flips at near-ties on the non-batch-
      invariant Marlin kernel (bi0 serves VLLM_BATCH_INVARIANT=0) -> cascade.
  (B) run-to-run nondeterminism of the stack itself (atomics / non-deterministic
      reductions), independent of the drafter.

This control distinguishes them. SAME drafter (ctk32), SAME serve env, SAME
decode workload (128x512, seed 1):
  * ctk32b == ctk32 (128/128)  => stack is deterministic given the drafter =>
    the sweep's 9/10 divergences are GENUINELY drafter-induced (cause A).
  * ctk32b diverges too        => cause B; byte-identity is unmeasurable at this
    granularity and impossible on this stack regardless of the lever.

Decode-only (no PPL): PPL is teacher-forced and drafter-invariant; already
confirmed identical at 2.0053 across all three points.

    cd target && .venv/bin/python -m research.bi0_mtp_accept.repeat_det
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from research.bi0_mtp_accept import sweep  # noqa: E402

OUT_DIR = sweep.OUT_DIR


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)
    snapshot = sweep.find_drafter_snapshot()
    # Re-stage ctk32 identically to the original control run.
    drafter_dir = sweep.stage_drafter(snapshot, 32)
    manifest = harness.load_manifest(sweep.SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[main] serve venv python = {server_python}", flush=True)

    log_path = OUT_DIR / "server_ctk32b.log"
    decode_out = OUT_DIR / "decode_ctk32b.jsonl"
    decode_sum = OUT_DIR / "decode_ctk32b.summary.json"
    extra_env = {"DRAFTER_MODEL": str(drafter_dir), "VLLM_USE_FLASHINFER_SAMPLER": "0"}

    t0 = time.time()
    with harness.LocalServer(
        sweep.SUBMISSION,
        server_python=server_python,
        port=8000,
        log_path=log_path,
        extra_env=extra_env,
        startup_timeout_s=1800,
    ) as srv:
        print(f"[serve] ready in {time.time()-t0:.0f}s", flush=True)
        harness.capture_decode(
            server_python,
            base_url=srv.base_url,
            model=srv.served_model_name,
            out_file=decode_out,
            summary_file=decode_sum,
            num_prompts=paths.NUM_PROMPTS,
            output_len=paths.OUTPUT_LEN,
            timeout_s=3600,
        )

    def load_sha(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[str(r["id"])] = r["completion_token_sha256"]
        return out

    a = load_sha(OUT_DIR / "decode_ctk32.jsonl")
    b = load_sha(decode_out)
    ids = sorted(set(a) & set(b))
    mism = sorted(i for i in ids if a[i] != b[i])
    matched = len(ids) - len(mism)
    print("\n========== ctk32 DETERMINISM CONTROL (ctk32b vs ctk32) ==========", flush=True)
    print(f"compared={len(ids)} matched={matched} mismatched={len(mism)}", flush=True)
    print(f"identical={len(mism) == 0 and len(ids) > 0}", flush=True)
    if mism:
        print(f"mismatched ids: {mism}", flush=True)
    verdict = (
        "STACK DETERMINISTIC given drafter -> sweep divergences are drafter-induced (cause A)"
        if not mism
        else "STACK NONDETERMINISTIC run-to-run (cause B) -> byte-identity unmeasurable"
    )
    print(f"VERDICT: {verdict}", flush=True)
    (OUT_DIR / "determinism_control.json").write_text(
        json.dumps(
            {
                "compared": len(ids),
                "matched": matched,
                "mismatched": len(mism),
                "mismatched_ids": mism,
                "identical": len(mism) == 0 and len(ids) > 0,
                "verdict": verdict,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
