#!/usr/bin/env python
"""Noise-floor control for the int4head within-job greedy gate (PR #801).

The within-job spec-off vs spec-on gate read 19/128 (109 divergent). To know
whether that is a SPEC-PATH divergence or just the A10G run-to-run FP-noise
floor (served greedy decode is non-deterministic at output_len 512 — FA_SLIDING
reduction noise + argmax ties), we need the M=1-AR-vs-M=1-AR control: serve
int4head spec-OFF a SECOND time and compare to the first spec-off run.

Decomposition:
  noise_floor   = specoff_A vs specoff_B   (two M=1 AR runs, same engine)
  spec_vs_refA  = specon  vs specoff_A     (already 19/128)
  spec_vs_refB  = specon  vs specoff_B     (cross-check vs the 2nd reference)

If noise_floor ~ spec_vs_ref, the 109 divergent is the irreducible hardware
floor and greedy-identity is not a discriminating fire gate (leaderboard gate is
PPL<=2.42 + 128/128 + modalities + quality, all of which int4head passes). If
noise_floor is clean (~128/128) but spec_vs_ref is divergent, the MTP verify
path genuinely diverges from plain AR.

LOCAL ONLY. No HF job.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

BASE = REPO / "research/validity/int4head_firevalidity_801"
WJ = BASE / "withinjob"
OUT = BASE / "noisefloor"


def capture_specoff_b(submission: Path, server_python: Path) -> Path:
    stage = OUT / "specoff_ref_B"
    stage.mkdir(parents=True, exist_ok=True)
    out = stage / "decode_outputs.jsonl"
    summary_file = stage / "decode_summary.json"
    log_path = stage / "server.log"
    print("[specoff_B] serving int4head spec-OFF (2nd independent M=1 AR run) ...", flush=True)
    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env={paths.REFERENCE_MODE_ENV: "1"}, startup_timeout_s=1800,
    ) as srv:
        print(f"[specoff_B] ready in {time.time()-t0:.0f}s; decoding 128x512 (temp=0)", flush=True)
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=out, summary_file=summary_file, num_prompts=paths.NUM_PROMPTS,
            output_len=paths.OUTPUT_LEN, seed=paths.SEED,
        )
    print(f"[specoff_B] records={summary['num_records']} tokens={summary['num_completion_tokens']}", flush=True)
    return out


def cmp(ref: Path, cand: Path, label: str) -> dict:
    r = greedy_gate.compare(ref, cand)
    d = r.to_dict()
    tot = d.get("total_tokens_compared", 0) or 0
    div = d.get("total_divergent_tokens", 0) or 0
    onset = greedy_gate.onset_summary(r)
    summary = {
        "label": label,
        "verdict": r.verdict,
        "num_identical": r.num_identical,
        "num_divergent": r.num_divergent,
        "num_prompts_compared": r.num_prompts_compared,
        "total_tokens_compared": tot,
        "total_divergent_tokens": div,
        "flip_rate_per_token": (div / tot) if tot else 0.0,
        "onset": onset,
    }
    print(f"\n[{label}] {r.verdict}: {r.num_identical}/{r.num_prompts_compared} identical, "
          f"flip_rate={summary['flip_rate_per_token']:.6f}", flush=True)
    print(f"    {greedy_gate.onset_line(onset, paths.OUTPUT_LEN)}", flush=True)
    return summary


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    submission = REPO / "submissions/int4_mtp_bi0_int4head"
    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    specoff_a = WJ / "specoff_ref" / "decode_outputs.jsonl"
    specon = WJ / "specon_cand" / "decode_outputs.jsonl"
    assert specoff_a.exists() and specon.exists(), "run within_job_greedy_gate.py first"

    specoff_b = capture_specoff_b(submission, server_python)

    results = {
        "noise_floor_AvsB": cmp(specoff_a, specoff_b, "noise_floor specoff_A vs specoff_B"),
        "spec_vs_refA": cmp(specoff_a, specon, "spec-on vs specoff_A"),
        "spec_vs_refB": cmp(specoff_b, specon, "spec-on vs specoff_B"),
    }
    (OUT / "control.json").write_text(json.dumps(results, indent=2))

    nf = results["noise_floor_AvsB"]
    sp = results["spec_vs_refA"]
    print("\n" + "=" * 70, flush=True)
    print("NOISE-FLOOR CONTROL (int4head, PR #801)", flush=True)
    print("=" * 70, flush=True)
    print(f"  noise floor  (M=1 AR vs M=1 AR): {nf['num_identical']}/128 identical, "
          f"flip {nf['flip_rate_per_token']:.4f}", flush=True)
    print(f"  spec-on vs spec-off  (refA)    : {sp['num_identical']}/128 identical, "
          f"flip {sp['flip_rate_per_token']:.4f}", flush=True)
    print(f"  spec-on vs spec-off  (refB)    : {results['spec_vs_refB']['num_identical']}/128 identical", flush=True)
    print("=" * 70, flush=True)
    print(f"[done] -> {OUT / 'control.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
