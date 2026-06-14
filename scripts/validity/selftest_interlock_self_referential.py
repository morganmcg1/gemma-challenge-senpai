#!/usr/bin/env python
"""Offline self-test for the PR #114 self-referential interlock verdict logic.

Builds synthetic capture trees in a tempdir (no GPU, no serving) in the exact
layout greedy_determinism.py writes — out_root/<config>/run_XX and
out_root/<config>__specoff/run_XX, each holding a decode_outputs.jsonl keyed by
record ``id`` (the official verifier's key) + ``index`` (load_runs' key) — and
asserts greedy_identity_interlock.interlock() returns the right verdict for:

  GREEN        : spec-ON self-deterministic, spec-OFF self-deterministic,
                 spec-ON == spec-OFF token-for-token (the stack reproduces its own AR)
  RED          : spec-ON self-deterministic but DIVERGENT from its own M=1 AR
                 (a near-tie flip at a known onset) -> greedy-safe-by-construction fails
  INCONCLUSIVE : spec-ON NOT self-deterministic run-to-run (PR #38 served wobble)
  INCONCLUSIVE : spec-OFF reference missing entirely

Also exercises the real CLI once (--self-referential --skip-capture) so the
argument plumbing and report writing are covered, not just the pure function.

Run: python scripts/validity/selftest_interlock_self_referential.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.validity import greedy_identity_interlock as gi  # noqa: E402

OUTPUT_LEN = 32
NUM_PROMPTS = 4


def _base_tokens(seed: int) -> list[list[int]]:
    """Deterministic per-prompt token lists (NUM_PROMPTS x OUTPUT_LEN)."""
    return [[(seed * 1000 + p * OUTPUT_LEN + t) % 256 for t in range(OUTPUT_LEN)]
            for p in range(NUM_PROMPTS)]


def _flip_at(tokens: list[list[int]], prompt: int, pos: int) -> list[list[int]]:
    """Copy of `tokens` with one argmax flip at (prompt, pos) — simulates a
    near-tie batched-verify divergence at a known onset."""
    out = [list(row) for row in tokens]
    out[prompt][pos] = (out[prompt][pos] + 1) % 256
    return out


def _write_run(run_dir: Path, tokens: list[list[int]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "decode_outputs.jsonl").open("w") as fh:
        for p, ids in enumerate(tokens):
            # `id` is the official verifier's match key; `index` is load_runs' key.
            fh.write(json.dumps({"id": f"prompt_{p}", "index": p,
                                 "completion_token_ids": ids}) + "\n")
    (run_dir / "meta.json").write_text(json.dumps({"run_idx": int(run_dir.name.split("_")[-1])}))


def _build(root: Path, *, spec_runs: list[list[list[int]]],
           ar_runs: list[list[list[int]]], config: str = "default") -> None:
    for i, toks in enumerate(spec_runs):
        _write_run(root / config / f"run_{i:02d}", toks)
    for i, toks in enumerate(ar_runs):
        _write_run(root / f"{config}{gi.SPECOFF_SUFFIX}" / f"run_{i:02d}", toks)


def _verdict(root: Path, config: str = "default") -> dict:
    spec = gi._runs_from(root, config, None)
    ar = gi._runs_from(root, config + gi.SPECOFF_SUFFIX, None)
    return gi.interlock(spec, ar, config, OUTPUT_LEN)


def main() -> int:
    ar = _base_tokens(1)
    fails = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # GREEN: two identical spec-ON runs, identical spec-OFF reference.
        green = tmp / "green"
        _build(green, spec_runs=[ar, [list(r) for r in ar]], ar_runs=[ar])
        rep = _verdict(green)
        ok = (rep["verdict"] == "GREEN"
              and rep["self_referential_gate_confirmed"] == "yes"
              and rep["primary_metric"]["value"] == 0
              and rep["self_consistency_gate"]["all_greedy_identical"] is True)
        print(f"[GREEN]        verdict={rep['verdict']:12s} "
              f"confirmed={rep['self_referential_gate_confirmed']} "
              f"div_runs={rep['primary_metric']['value']}  -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(("GREEN", rep))

        # RED: spec-ON self-deterministic but DIVERGENT from own AR at prompt 2 pos 7.
        red_tokens = _flip_at(ar, prompt=2, pos=7)
        red = tmp / "red"
        _build(red, spec_runs=[red_tokens, [list(r) for r in red_tokens]], ar_runs=[ar])
        rep = _verdict(red)
        sc = rep["self_consistency_gate"]
        ok = (rep["verdict"] == "RED"
              and rep["self_referential_gate_confirmed"] == "no"
              and rep["primary_metric"]["value"] == 2          # both spec-ON runs diverge
              and sc["onset_min"] == 7                         # flip onset localized
              and sc["num_divergent_runs"] == 2)
        print(f"[RED]          verdict={rep['verdict']:12s} "
              f"confirmed={rep['self_referential_gate_confirmed']} "
              f"div_runs={rep['primary_metric']['value']} onset_min={sc['onset_min']} "
              f"sig={sc['onset_signature']!r}  -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(("RED", rep))

        # INCONCLUSIVE: spec-ON not self-deterministic (run_00 != run_01).
        wob = tmp / "wobble"
        _build(wob, spec_runs=[ar, _flip_at(ar, prompt=0, pos=3)], ar_runs=[ar])
        rep = _verdict(wob)
        ok = (rep["verdict"] == "INCONCLUSIVE"
              and rep["self_referential_gate_confirmed"] == "inconclusive"
              and rep["spec_on_self_determinism"]["deterministic"] is False)
        print(f"[WOBBLE]       verdict={rep['verdict']:12s} "
              f"confirmed={rep['self_referential_gate_confirmed']} "
              f"spec_on_det={rep['spec_on_self_determinism']['deterministic']}  "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(("WOBBLE", rep))

        # INCONCLUSIVE: spec-OFF reference missing entirely.
        miss = tmp / "missing"
        _build(miss, spec_runs=[ar, [list(r) for r in ar]], ar_runs=[])
        rep = _verdict(miss)
        ok = rep["verdict"] == "INCONCLUSIVE" and "missing captures" in rep["reason"]
        print(f"[MISSING-AR]   verdict={rep['verdict']:12s} "
              f"reason={rep['reason'][:46]!r}...  -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(("MISSING-AR", rep))

        # CLI end-to-end: --self-referential --skip-capture on the GREEN tree.
        report_path = tmp / "cli_report.json"
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts/validity/greedy_identity_interlock.py"),
             "--self-referential", "--skip-capture",
             "--spec-root", str(green / "default"),
             "--ar-root", str(green / f"default{gi.SPECOFF_SUFFIX}"),
             "--config", "default", "--output-len", str(OUTPUT_LEN),
             "--report", str(report_path)],
            capture_output=True, text=True)
        cli_rep = json.loads(report_path.read_text()) if report_path.exists() else {}
        ok = proc.returncode == 0 and cli_rep.get("verdict") == "GREEN"
        print(f"[CLI skip-cap] rc={proc.returncode} verdict={cli_rep.get('verdict')}  "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(("CLI", {"rc": proc.returncode, "stderr": proc.stderr[-400:]}))

    print("-" * 60)
    if fails:
        print(f"SELFTEST FAILED ({len(fails)} case(s)): {[f[0] for f in fails]}")
        for name, rep in fails:
            print(f"  --- {name} ---\n{json.dumps(rep, indent=2)[:800]}")
        return 1
    print("SELFTEST PASSED — all verdicts correct (GREEN/RED/INCONCLUSIVE x2 + CLI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
