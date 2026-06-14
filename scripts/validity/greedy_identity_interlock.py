#!/usr/bin/env python
"""Step 4 pre-quota interlock (PR #96): greedy-token identity, baseline vs the
REAL composed frontier (#71 tree M-widen × #84 SplitK W4A16) on the served stack.

PR #96's emulation (greedy_compounding.py) bounds the *upstream* network-wide
compounding of per-GEMM ≤1-ULP reduction-order perturbations on a fixed greedy
trajectory. This script is the ground-truth complement land #71 / ubel #84 run on
the actual composed kernels before spending landing quota: it serves each stack
with the proven #73 harness, captures greedy completions over N fresh reloads, and
diffs them token-for-token (sha256). It answers the one question that decides the
merge: does the composed stack emit byte-identical greedy tokens to the baseline?

It reuses, verbatim, the two proven #73 primitives:
  * greedy_determinism.py  — served LocalServer reload + capture_decode (subprocess)
  * analyze_determinism.py — load_runs / pair_stats / prompt_pair token diff

Three checks, in #96-Step-1 order (a non-deterministic baseline makes the identity
comparison meaningless, so it is a hard precondition — see PR #38: served greedy
decode can be non-deterministic run-to-run on A10G; pin a reproducible --config if
the baseline self-check fails):
  1. baseline self-determinism  (run-to-run byte-identical)   -> precondition
  2. composed self-determinism  (run-to-run byte-identical)
  3. baseline-vs-composed identity (every cross pair byte-identical, 0 divergent)

GREEN iff all three hold. Otherwise RED/INCONCLUSIVE with onset diagnostics
(first divergent token position) so the failure can be localized.

Single command (capture + diff):

  python scripts/validity/greedy_identity_interlock.py \
      --baseline-submission submissions/<current deployed build> \
      --composed-submission submissions/<#71×#84 composed build> \
      --runs 3 --config default

Diff pre-captured roots only (no GPU; re-analysis / offline self-test):

  python scripts/validity/greedy_identity_interlock.py --skip-capture \
      --baseline-root <dir> --composed-root <dir> --config default
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.validity.analyze_determinism import load_runs, pair_stats  # noqa: E402

DET_SCRIPT = REPO / "scripts/validity/greedy_determinism.py"
IDENTITY_MIN = 0.999  # mean byte-identical frac >= this == "identical" (allows no slack)


def _capture(submission: Path, out_root: Path, *, config: str, runs: int,
             num_prompts: int, output_len: int) -> None:
    """Serve `submission` and capture `runs` fresh reloads into out_root/{config}/."""
    cmd = [
        sys.executable, str(DET_SCRIPT),
        "--submission", str(submission),
        "--config", config,
        "--runs", str(runs),
        "--num-prompts", str(num_prompts),
        "--output-len", str(output_len),
        "--out-root", str(out_root),
    ]
    print(f"[interlock] capture: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _self_determinism(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """All run-to-run pairs within one stack must be byte-identical."""
    if len(runs) < 2:
        return {"num_runs": len(runs), "checkable": False, "min_byte_identical_frac": None,
                "deterministic": None, "num_divergent_pairs": 0, "onsets": []}
    fracs, onsets, ndiv = [], [], 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            s = pair_stats(runs[i]["rows"], runs[j]["rows"])
            fracs.append(s["byte_identical_frac"])
            onsets.extend(s["onsets"])
            ndiv += s["num_divergent"]
    mn = min(fracs) if fracs else None
    return {"num_runs": len(runs), "checkable": True,
            "min_byte_identical_frac": mn,
            "deterministic": (mn is not None and mn >= IDENTITY_MIN),
            "num_divergent_pairs": ndiv, "onsets": sorted(onsets)}


def _identity(base_runs: list[dict[str, Any]], comp_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Every (baseline run × composed run) pair must be byte-identical."""
    fracs, onsets, ndiv, npairs = [], [], 0, 0
    for b, c in product(base_runs, comp_runs):
        s = pair_stats(b["rows"], c["rows"])
        fracs.append(s["byte_identical_frac"])
        onsets.extend(s["onsets"])
        ndiv += s["num_divergent"]
        npairs += 1
    mean_frac = sum(fracs) / len(fracs) if fracs else None
    onsets.sort()
    return {"num_pairs": npairs, "mean_byte_identical_frac": mean_frac,
            "num_divergent_prompt_pairs": ndiv,
            "identical": (mean_frac is not None and mean_frac >= IDENTITY_MIN and ndiv == 0),
            "onset_min": onsets[0] if onsets else None,
            "onset_max": onsets[-1] if onsets else None,
            "onsets_head": onsets[:32]}


def interlock(base_root: Path, comp_root: Path, config: str, output_len: int) -> dict[str, Any]:
    base_runs = load_runs(base_root, config)
    comp_runs = load_runs(comp_root, config)
    if not base_runs or not comp_runs:
        return {"verdict": "INCONCLUSIVE",
                "reason": f"missing captures (baseline runs={len(base_runs)}, "
                          f"composed runs={len(comp_runs)}) under config '{config}'",
                "baseline_root": str(base_root), "composed_root": str(comp_root)}

    base_self = _self_determinism(base_runs)
    comp_self = _self_determinism(comp_runs)
    ident = _identity(base_runs, comp_runs)

    # #96-Step-1 precondition: a non-deterministic baseline (PR #38 served wobble)
    # makes the identity comparison meaningless -> INCONCLUSIVE, pin a config.
    if base_self["checkable"] and not base_self["deterministic"]:
        verdict = "INCONCLUSIVE"
        reason = (f"baseline NOT self-deterministic run-to-run "
                  f"(min byte-identical {base_self['min_byte_identical_frac']:.4f}); served "
                  f"nondeterminism (PR #38) swamps the identity test — pin a reproducible "
                  f"--config (e.g. fa_sliding_off) and re-run")
    elif ident["identical"] and (comp_self["deterministic"] is not False):
        verdict = "GREEN"
        reason = ("composed frontier emits byte-identical greedy tokens to baseline "
                  f"({ident['num_pairs']} cross pairs, 0 divergent) and is self-deterministic")
    else:
        verdict = "RED"
        bits = [f"{ident['num_divergent_prompt_pairs']} divergent baseline-vs-composed prompt-pairs"]
        if comp_self["deterministic"] is False:
            bits.append(f"composed self-nondeterministic (min {comp_self['min_byte_identical_frac']:.4f})")
        if ident["onset_min"] is not None:
            bits.append(f"first divergence at token {ident['onset_min']}")
        reason = "; ".join(bits)

    return {
        "verdict": verdict,
        "reason": reason,
        "config": config,
        "identity_min_threshold": IDENTITY_MIN,
        "baseline_root": str(base_root),
        "composed_root": str(comp_root),
        "baseline_self_determinism": base_self,
        "composed_self_determinism": comp_self,
        "baseline_vs_composed_identity": ident,
        "primary_metric": {"name": "baseline_vs_composed_divergent_prompt_pairs",
                           "value": ident["num_divergent_prompt_pairs"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-submission", default=None)
    ap.add_argument("--composed-submission", default=None)
    ap.add_argument("--config", default="default",
                    help="served config to pin for BOTH stacks (default/fa_sliding_off/"
                         "splitkv_off/atomic_on); pin a reproducible one if baseline wobbles")
    ap.add_argument("--runs", type=int, default=3, help="fresh reloads per stack")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--out-root", default=None,
                    help="parent dir for baseline/ and composed/ capture roots")
    ap.add_argument("--baseline-root", default=None, help="explicit baseline capture root")
    ap.add_argument("--composed-root", default=None, help="explicit composed capture root")
    ap.add_argument("--skip-capture", action="store_true",
                    help="diff existing --baseline-root/--composed-root only (no GPU)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="4 prompts x 32 tok, 1 run, baseline-vs-baseline plumbing check")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts, args.output_len, args.runs = 4, 32, 1
        if args.composed_submission is None:
            args.composed_submission = args.baseline_submission

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root) if args.out_root else (
        REPO / "research/validity/greedy_compounding" / f"interlock-{ts}")
    base_root = Path(args.baseline_root) if args.baseline_root else out_root / "baseline"
    comp_root = Path(args.composed_root) if args.composed_root else out_root / "composed"

    if not args.skip_capture:
        if not args.baseline_submission or not args.composed_submission:
            ap.error("--baseline-submission and --composed-submission required unless --skip-capture")
        _capture(Path(args.baseline_submission), base_root, config=args.config, runs=args.runs,
                 num_prompts=args.num_prompts, output_len=args.output_len)
        _capture(Path(args.composed_submission), comp_root, config=args.config, runs=args.runs,
                 num_prompts=args.num_prompts, output_len=args.output_len)

    report = interlock(base_root, comp_root, args.config, args.output_len)
    report_path = Path(args.report) if args.report else out_root / "interlock_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74, flush=True)
    print("GREEDY-IDENTITY INTERLOCK (PR #96 Step 4) — baseline vs composed frontier", flush=True)
    print("=" * 74, flush=True)
    bs, cs, idn = (report.get("baseline_self_determinism"), report.get("composed_self_determinism"),
                   report.get("baseline_vs_composed_identity"))
    if bs is not None:
        print(f"  baseline self-determ : runs={bs['num_runs']} "
              f"min_byte_identical={bs['min_byte_identical_frac']} det={bs['deterministic']}", flush=True)
        print(f"  composed self-determ : runs={cs['num_runs']} "
              f"min_byte_identical={cs['min_byte_identical_frac']} det={cs['deterministic']}", flush=True)
        print(f"  base-vs-composed     : pairs={idn['num_pairs']} "
              f"mean_byte_identical={idn['mean_byte_identical_frac']} "
              f"divergent_pairs={idn['num_divergent_prompt_pairs']} "
              f"onset_min={idn['onset_min']}", flush=True)
    print("-" * 74, flush=True)
    print(f"VERDICT: {report['verdict']}  — {report['reason']}", flush=True)
    print(f"[interlock] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
