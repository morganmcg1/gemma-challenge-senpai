#!/usr/bin/env python
"""PR #805 — cross-session greedy-determinism confirmatory test.

Decisive disambiguation for the Stage-3 113/128 "DIVERGENT" greedy verdict:

  H_crosssession (benign): the bf16 per_layer_input_gate GEMV (cuBLAS) is
    cross-session NON-deterministic (memory: bf16 GEMV flips greedy argmax
    ~9-13% across serve processes), so ANY two-process comparison of pledequant
    diverges — independent of speculation. MTP spec decode is exonerated.

  H_specbreak (concerning): the bf16 PLE makes MTP speculative decode lossy, so
    spec-OFF decode is deterministic and two spec-OFF runs are IDENTICAL.

These predict OPPOSITE outcomes for spec-OFF run A vs spec-OFF run B (same
checkpoint, M=1 AR, drafter OFF — speculation is NOT involved at all):
  H_crosssession -> ~110/128 divergent (and int4head control -> ~0/128)
  H_specbreak    -> 0/128 divergent

Run A = the existing canonical spec-off references (generated earlier this job).
Run B = a fresh spec-off serve process, written to a separate --out path.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
PY = "/tmp/senpai-venvs/20f658587e8a6643/bin/python"

from scripts.local_validation import gen_greedy_reference as gr  # noqa: E402
from scripts.local_validation import greedy_gate  # noqa: E402

OUT = ROOT / "research" / "int4head_ple_dequant" / "xsession"
OUT.mkdir(parents=True, exist_ok=True)
N = 128

CASES = {
    "pledequant": ROOT / "submissions" / "int4_mtp_bi0_int4head_pledequant",
    "int4head": ROOT / "submissions" / "int4_mtp_bi0_int4head",
}


def sh(cmd: list[str], log: Path) -> int:
    print(f"\n$ {' '.join(cmd)}\n  -> log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    return rc


def run_b(label: str, sub: Path) -> Path:
    out_b = OUT / f"{label}_specoff_B" / "decode_outputs.jsonl"
    out_b.parent.mkdir(parents=True, exist_ok=True)
    rc = sh([PY, "-m", "scripts.local_validation.gen_greedy_reference",
             "--mode", "served", "--submission", str(sub), "--spec-off",
             "--num-prompts", str(N), "--port", "8001", "--out", str(out_b)],
            OUT / f"genref_{label}_B.log")
    assert rc == 0 and out_b.exists(), f"{label} run B failed rc={rc}"
    return out_b


def compare(label: str, ref_a: Path, ref_b: Path) -> dict:
    rep = greedy_gate.compare(ref_a, ref_b)  # A=reference, B=candidate (symmetric counts)
    per = [p for p in rep.per_prompt if not p.identical]
    first_tok = sum(1 for p in per if p.first_divergence_index == 0)
    onset = greedy_gate.onset_summary(rep)
    res = {
        "label": f"{label}_specoff_A_vs_B",
        "verdict": rep.verdict,
        "num_prompts_compared": rep.num_prompts_compared,
        "num_identical": rep.num_identical,
        "num_divergent": rep.num_divergent,
        "total_tokens_compared": rep.total_tokens_compared,
        "total_divergent_tokens": rep.total_divergent_tokens,
        "first_token_flips": first_tok,
        "onset_line": greedy_gate.onset_line(onset, 512),
        "ref_a": str(ref_a),
        "ref_b": str(ref_b),
    }
    print(f"\n[xsession {label}] specoff-A vs specoff-B: verdict={res['verdict']} "
          f"divergent={res['num_divergent']}/{res['num_prompts_compared']} "
          f"first_tok_flips={first_tok}  {res['onset_line']}", flush=True)
    return res


def main() -> int:
    summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": {}}
    for label, sub in CASES.items():
        ref_a = gr.served_reference_path(gr.reference_key_for_submission(sub))
        assert ref_a.exists(), f"{label} run-A canonical ref missing: {ref_a}"
        print(f"\n===== {label}: run-A canonical ref = {ref_a} =====", flush=True)
        ref_b = run_b(label, sub)
        summary["cases"][label] = compare(label, ref_a, ref_b)
        (OUT / "xsession_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== VERDICT =====", flush=True)
    pd = summary["cases"]["pledequant"]
    ih = summary["cases"]["int4head"]
    print(f"pledequant specoff A-vs-B: {pd['num_divergent']}/128 divergent", flush=True)
    print(f"int4head   specoff A-vs-B: {ih['num_divergent']}/128 divergent", flush=True)
    if pd["num_divergent"] > 20 and ih["num_divergent"] <= 2:
        print("=> CONFIRMED H_crosssession: bf16 PLE reintroduces cross-session greedy "
              "nondeterminism; int4 is bit-exact; MTP spec decode exonerated.", flush=True)
    elif pd["num_divergent"] <= 2:
        print("=> H_specbreak: pledequant spec-off is deterministic across sessions; "
              "the Stage-3 divergence came from speculation -> investigate MTP.", flush=True)
    else:
        print("=> AMBIGUOUS: see counts above.", flush=True)
    (OUT / "xsession_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[xsession] summary -> {OUT / 'xsession_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
