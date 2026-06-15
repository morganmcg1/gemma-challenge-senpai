#!/usr/bin/env python3
"""Official greedy-gate summary for the static-K wall-clock A/B (PR #273).

The PR premise was "greedy-safe by construction: emitted token-ids are identical
across all K (128/128)". That is empirically FALSE for *every* config on this
int4 + vLLM stack: floating-point reduction order in the verify step is not
bit-stable, so argmax ties flip and the greedy stream diverges run-to-run. This is
the int4+vLLM nondeterminism that the competition's greedy gate is documented to
tolerate (it compares *within-stack*, and run-to-run FP drift is not a blocker).

What actually matters for a fair A/B is **greedy-validity parity**: changing K must
not push the served stream OUT of the deployed K=7's benign-FP regime into an
early/lossy divergence regime (which would mean K changed what the model really
emits). This script runs the OFFICIAL ``greedy_gate.py`` for each K's served decode
against the committed canonical M=1 autoregressive reference and summarizes the
divergence regime (verdict, divergent-token %, onset distribution). If every
candidate K matches the deployed K=7's regime, the A/B is comparing like-for-like
greedy-valid configs.

No GPU: pure file comparison over already-captured decode jsonl.

Usage:
    .venv/bin/python research/validity/static_k_wallclock_ab/greedy_gate_summary.py \
        --seed 1 --ks 3 4 5 6 7
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTROOT = ROOT / "research" / "validity" / "static_k_wallclock_ab"
# Canonical served-mode, spec-off M=1 AR reference for this exact submission (committed).
REF = (ROOT / "research" / "greedy_reference"
       / "workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it"
       / "decode_outputs.jsonl")
REF_K = 7
EARLY_ONSET = 16  # first-divergence index below this == early/lossy-style onset


def _decode_for(seed: int, k: int) -> Path | None:
    p = OUTROOT / f"seed{seed}_mtp_k{k}" / f"mtp_k{k}" / "decode" / "run00.jsonl"
    if p.exists():
        return p
    if k == REF_K:  # K=7 reference is saved as the baseline arm of the fresh candidate
        for cand in sorted(OUTROOT.glob(f"seed{seed}_mtp_k*/mtp_k7/decode/run00.jsonl")):
            return cand
    return None


def _run_gate(cand: Path) -> dict[str, Any]:
    r = subprocess.run(
        [".venv/bin/python", "-m", "scripts.local_validation.greedy_gate",
         "--candidate", str(cand), "--reference", str(REF), "--json"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    txt = r.stdout.strip()
    start = txt.find("{")
    if start < 0:
        raise RuntimeError(f"greedy_gate produced no json for {cand}:\n{r.stderr[-2000:]}")
    return json.loads(txt[start:])


def summarize(seed: int, ks: list[int]) -> dict[str, Any]:
    if not REF.exists():
        return {"error": f"committed M=1 AR reference not found: {REF}"}
    summary: dict[str, Any] = {}
    for k in ks:
        d = _decode_for(seed, k)
        if d is None:
            summary[str(k)] = {"pending": True}
            continue
        g = _run_gate(d)
        onsets = [p["first_divergence_index"] for p in g["per_prompt"]
                  if p.get("first_divergence_index") is not None]
        summary[str(k)] = {
            "verdict": g["verdict"],
            "num_identical": g["num_identical"],
            "num_prompts": g["num_prompts_compared"],
            "num_divergent_prompts": g["num_divergent"],
            "total_divergent_tokens": g["total_divergent_tokens"],
            "total_tokens": g["total_tokens_compared"],
            "divergent_token_pct": round(100 * g["total_divergent_tokens"]
                                         / max(1, g["total_tokens_compared"]), 3),
            "onset_min": min(onsets) if onsets else None,
            "onset_median": statistics.median(onsets) if onsets else None,
            "onset_max": max(onsets) if onsets else None,
            "n_early_onset_lt16": sum(1 for o in onsets if o < EARLY_ONSET),
            "integrity_failures": g.get("integrity_failures", []),
            "decode_file": str(d.relative_to(ROOT)),
            "is_reference_k": (k == REF_K),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    summary = summarize(args.seed, args.ks)
    out = args.out or (OUTROOT / f"greedy_gate_vs_m1ar_seed{args.seed}.json")
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n===== Official greedy-gate vs M=1 AR (seed={args.seed}) =====")
    if "error" in summary:
        print("  ERROR:", summary["error"])
        return 1
    print(f"  reference (M=1 AR, served spec-off): {REF.relative_to(ROOT)}")
    print(f"{'K':>3}  {'verdict':>10}  {'ident':>7}  {'div-tok%':>9}  "
          f"{'onset med':>10}  {'early<16':>9}  {'integrity':>10}")
    for k in sorted(summary, key=int):
        s = summary[k]
        if s.get("pending"):
            print(f"{k:>3}  {'PENDING':>10}")
            continue
        tag = " (REF K)" if s.get("is_reference_k") else ""
        print(f"{k:>3}  {s['verdict']:>10}  {s['num_identical']:>3}/{s['num_prompts']:<3}  "
              f"{s['divergent_token_pct']:>8.2f}%  {str(s['onset_median']):>10}  "
              f"{s['n_early_onset_lt16']:>9}  {'OK' if not s['integrity_failures'] else 'FAIL':>10}{tag}")
    print(f"\n  >>> summary -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
