#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #764 (land) -- analyze the independent fire literal-identity cross-validation.

Reads the arm dirs written by run_xcheck.py:
  spec_on        = fire config (BI=1, MTP drafter num_spec=6) -- the CANDIDATE
  spec_off_ref   = same serve.py + SENPAI_REFERENCE_MODE=1 (M=1 AR, drafter OFF) -- MY reference
  spec_off_repB  = a 2nd fresh M=1 AR run (N=32) -- the AR-vs-AR determinism control run B

PRIMARY: fire_literal_greedy_identity = fraction of the 128 prompts byte-exact (spec_on vs
spec_off_ref, completion_token_sha256 equality, strict-#319 all-or-nothing greedy identity). The
only within-pair difference is speculation on/off (BI, model, kernels, backend held identical), so a
DIVERGENT prompt is attributable to the MTP spec verify path, not cross-engine noise.

SUPPORTING (mechanism evidence for #751's int4 quant-grid near-tie argmax flips):
  - frac_diverging          = n_diverge / n_total  (#751 saw ~0.84 = 108/128)
  - first_div_pos_histogram = where the first flip lands per diverging prompt (SPREAD across the
    512-token rollout, NOT root-clustered, is the near-tie-cascade signature)
DETERMINISM CONTROL: spec_off_ref vs spec_off_repB over the shared 32 prompts. 32/32 (#751 saw 0/32
divergence) proves the M=1 AR stack is bit-reproducible, so the spec_on<128/128 is a real per-step
reduction-order divergence, not run-to-run noise -- the control that makes the primary interpretable.

VERDICT: xcheck_consistent_with_751 = 1 iff |n_identical - 20| <= TOL (default TOL=10/128), i.e. my
independent number reproduces wirbel #751's 0.156 within tolerance -> the "self-consistent + PPL-clean
but NOT literal-byte-exact vs an independent AR reference" claim is independently corroborated (board
honesty hardened). 0 -> harness-dependence, a board blocker to surface BEFORE the post.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

# reuse the MERGED #748 identity machinery (my own work)
P748 = ROOT / "research" / "validity" / "strict_clean_served_byteexact_748"
sys.path.insert(0, str(P748))
from analyze import first_div, identity, load_arm  # noqa: E402

WIRBEL_751_IDENTICAL = 20      # 20/128 identical
WIRBEL_751_FRAC = 0.156        # literal greedy identity
WIRBEL_751_DIVERGE_FRAC = 0.84  # ~108/128 diverge at all
DEFAULT_TOL = 10               # +/-10/128 on the identical count


def first_div_histogram(diverging: list[dict], output_len: int) -> dict:
    """Bin per-prompt first-divergence positions to show the flip is SPREAD, not root-clustered."""
    edges = [0, 1, 4, 16, 64, 128, 256, output_len + 1]
    labels = ["0", "1-3", "4-15", "16-63", "64-127", "128-255", f"256-{output_len}"]
    counts = {lab: 0 for lab in labels}
    for d in diverging:
        fd = d["first_div_pos"]
        for j in range(len(edges) - 1):
            if edges[j] <= fd < edges[j + 1]:
                counts[labels[j]] += 1
                break
    fds = sorted(d["first_div_pos"] for d in diverging)
    return {
        "bins": counts,
        "min": (fds[0] if fds else None),
        "median": (fds[len(fds) // 2] if fds else None),
        "max": (fds[-1] if fds else None),
        "frac_first_div_after_tok16": (
            round(sum(1 for f in fds if f >= 16) / len(fds), 4) if fds else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--spec-on", default="spec_on")
    ap.add_argument("--spec-off", default="spec_off_ref")
    ap.add_argument("--spec-off-repB", default="spec_off_repB")
    ap.add_argument("--tol", type=int, default=DEFAULT_TOL)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "analysis.json")
    args = ap.parse_args()

    cand = load_arm(args.runs / args.spec_on)
    ref = load_arm(args.runs / args.spec_off)

    # PRIMARY: spec_on (candidate) vs spec_off M=1 AR (my independent reference)
    ident = identity(cand, ref)
    n_match, n_total, n_div = ident["n_match"], ident["n_total"], ident["n_diverge"]
    frac = ident["frac"]
    frac_diverging = (n_div / n_total) if n_total else float("nan")
    hist = first_div_histogram(ident["diverging"], ref["summary"]["output_len"])

    # DETERMINISM CONTROL: spec_off_ref vs a 2nd fresh M=1 AR run over the shared (32) prompts
    determinism = None
    repB_dir = args.runs / args.spec_off_repB
    if (repB_dir / "decode_outputs.jsonl").exists():
        repB = load_arm(repB_dir)
        det = identity(repB, ref)  # identity() intersects on shared ids
        determinism = {
            "identity": {k: det[k] for k in ("n_match", "n_total", "frac", "n_diverge",
                                             "per_token_flip_hazard")},
            "deterministic": bool(det["n_match"] == det["n_total"] and det["n_total"] > 0),
            "floor_frac": det["frac"],
            "n_shared_prompts": det["n_total"],
            "first_div_of_any_nondeterministic": [d["first_div_pos"] for d in det["diverging"]],
        }

    # VERDICT
    n_off = abs(n_match - WIRBEL_751_IDENTICAL)
    xcheck = bool(n_off <= args.tol)
    band_lo, band_hi = WIRBEL_751_IDENTICAL - args.tol, WIRBEL_751_IDENTICAL + args.tol

    if xcheck:
        verdict = "XCHECK_REPRODUCES_751_NOT_BYTEEXACT"
        honest = (
            f"Independent harness reproduces wirbel #751: fire literal served-greedy identity "
            f"{n_match}/{n_total} (frac {frac:.4f}) is within +/-{args.tol}/128 of #751's 20/128 "
            f"(0.156). The fire (full BI=1, MTP spec) is self-consistent-gate + PPL-clean but NOT "
            f"literal-byte-exact vs an INDEPENDENT served spec-off M=1 AR reference -- corroborated "
            f"on a second harness. This is a DIFFERENT notion of strict than denken's 128/128 "
            f"self-consistent gate (whose reference differs from mine); BOTH hold, and this does NOT "
            f"overturn the gate. NOT a DQ (PPL 2.0057 / 128-128 completion / all modalities hold).")
    else:
        verdict = "XCHECK_HARNESS_DEPENDENCE"
        honest = (
            f"Independent harness does NOT reproduce wirbel #751: fire literal identity {n_match}/"
            f"{n_total} (frac {frac:.4f}) is OUTSIDE +/-{args.tol}/128 of #751's 20/128 (0.156). "
            f"This is a harness-dependence that MUST be surfaced to the board BEFORE the post and "
            f"quantified ({'near-128/128 -- my harness sees byte-exactness #751 did not' if n_match > band_hi else 'near-0 -- my harness sees MORE divergence than #751'}).")

    result = {
        "pr": 764,
        "verdict": verdict,
        "primary_metric": {"name": "fire_literal_greedy_identity", "value": frac,
                           "n_match": n_match, "n_total": n_total, "n_diverge": n_div},
        "test_metric": {"name": "xcheck_consistent_with_751", "value": int(xcheck)},
        "xcheck": {
            "consistent": xcheck,
            "n_identical": n_match,
            "wirbel_751_identical": WIRBEL_751_IDENTICAL,
            "abs_diff_identical": n_off,
            "tolerance_identical": args.tol,
            "consistency_band_identical": [band_lo, band_hi],
            "my_frac": round(frac, 4), "wirbel_751_frac": WIRBEL_751_FRAC,
        },
        "divergence": {
            "frac_diverging": round(frac_diverging, 4),
            "n_diverging": n_div,
            "wirbel_751_diverge_frac": WIRBEL_751_DIVERGE_FRAC,
            "per_token_flip_hazard": ident["per_token_flip_hazard"],
            "first_div_pos_histogram": hist,
        },
        "determinism_control": determinism,
        "honest_read": honest,
        "arms": {
            "spec_on": {k: cand["summary"].get(k) for k in (
                "tag", "kind", "num_speculative_tokens", "output_tps", "n_prompts",
                "output_len", "peak_gpu_mem_mib", "boot_s", "server_backend_line", "server_spec_line")},
            "spec_off_ref": {k: ref["summary"].get(k) for k in (
                "tag", "kind", "num_speculative_tokens", "output_tps", "n_prompts",
                "output_len", "peak_gpu_mem_mib", "boot_s", "server_backend_line",
                "server_refmode_line")},
        },
        "tps_note": ("output_tps is a raw full-vocab api_server LOCAL probe (no deployed pruned-16k "
                     "lm_head / fa2sw / precache / onegraph); it does NOT transfer to official TPS. "
                     "official_tps=0, no_hf_job=1, fires=0; locked int4_g128_lmhead@126.378 untouched. "
                     "Only the identity / divergence-distribution / determinism results transfer."),
    }
    args.out.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print(f"  fire_literal_greedy_identity (spec_on vs my spec_off M=1 AR ref): "
          f"{n_match}/{n_total}  frac={frac:.4f}  hazard={ident['per_token_flip_hazard']*100:.3f}%/tok")
    print(f"  vs wirbel #751: 20/128 (0.156)  |  abs_diff={n_off} identical  "
          f"tol=+/-{args.tol}  band=[{band_lo},{band_hi}]  -> consistent={xcheck}")
    print(f"  frac_diverging={frac_diverging:.4f} (n_div={n_div})  vs #751 ~0.84")
    print(f"  first_div histogram: {hist['bins']}")
    print(f"    first_div min/median/max = {hist['min']}/{hist['median']}/{hist['max']}  "
          f"frac_after_tok16={hist['frac_first_div_after_tok16']}")
    if determinism is not None:
        di = determinism["identity"]
        print(f"  DETERMINISM CONTROL (spec_off rep B vs ref, {determinism['n_shared_prompts']} shared): "
              f"{di['n_match']}/{di['n_total']}  floor_frac={di['frac']:.4f}  "
              f"-> {'DETERMINISTIC (rollout test valid)' if determinism['deterministic'] else 'NON-REPRODUCIBLE'}")
    else:
        print("  DETERMINISM CONTROL: pending (spec_off_repB not run)")
    print(f"  TPS (LOCAL probe, non-transferable): spec_on={cand['summary']['output_tps']:.2f}  "
          f"spec_off_ref={ref['summary']['output_tps']:.2f}")
    print(f"  -> {args.out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
