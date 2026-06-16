"""PR #488 diagnostic: where does the census M=1-AR divergence come from?

CPU-only token diffs over already-captured decode jsonl files. No GPU, no serve.
Isolates whether the 0.42-0.44 census identity is the matmul-library (the thing the
gate tries to measure) or the ONEGRAPH/M=1-batching confound (the thing the M=1 AR
reference accidentally also changes).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run"
CEN = ROOT / "census"


def load(path: Path) -> dict[str, list[int]]:
    seqs: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = str(obj.get("id", obj.get("dataset_index", obj.get("index", len(seqs)))))
        toks = obj.get("completion_token_ids")
        if isinstance(toks, list):
            seqs[key] = [int(t) for t in toks]
    return seqs


def diff(a: Path, b: Path, label: str) -> dict:
    sa, sb = load(a), load(b)
    common = sorted(set(sa) & set(sb))
    total = matched = nflip = 0
    roots = []
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        sf = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - sf
        if sf or len(ta) != len(tb):
            nflip += 1
            for i in range(n):
                if ta[i] != tb[i]:
                    roots.append(i)
                    break
    rate = matched / total if total else None
    roots.sort()
    med_root = roots[len(roots) // 2] if roots else None
    print(
        f"{label:48s} rate={rate:.6f} flips={nflip:3d}/{len(common)} "
        f"min_root={roots[0] if roots else None} med_root={med_root}"
    )
    return {"label": label, "rate": rate, "flips": nflip, "n": len(common),
            "min_root": roots[0] if roots else None, "med_root": med_root}


def main() -> None:
    print("=== PR488 reference-confound diagnostic (CPU-only token diffs) ===\n")
    print("[A] CONSISTENCY: do the two M=1 AR references agree? (M=1, ONEGRAPH off,")
    print("    drafter off; ONLY matmul lib differs -> a no-op at M=1 -> expect ~1.0)")
    diff(CEN / "surgical_m1ar" / "decode_round00.jsonl",
         CEN / "full_flag_m1ar" / "decode_round00.jsonl",
         "surgical_m1ar vs full_flag_m1ar")

    print("\n[B] SERVED DETERMINISM: same config, two separate decode runs (expect ~1.0")
    print("    if the M=8 served path is run-to-run deterministic)")
    for arm in ("deployed", "full_flag", "surgical"):
        diff(RUN / arm / "decode_round00.jsonl",
             RUN / arm / "decode_round01.jsonl",
             f"{arm}_m8 r0 vs r1")

    print("\n[C] SAME-CONFIG cross-arm at M=8 (vary ONLY matmul lib; ONEGRAPH on, drafter")
    print("    on, M=8 -> isolates the matmul-tax identity effect)")
    diff(RUN / "surgical" / "decode_round00.jsonl",
         RUN / "full_flag" / "decode_round00.jsonl",
         "surgical_m8 vs full_flag_m8")

    print("\n[D] THE CONFOUNDED GATE (M=8 served vs M=1 AR; changes matmul lib AND")
    print("    ONEGRAPH AND batching all at once)")
    diff(RUN / "surgical" / "decode_round00.jsonl",
         CEN / "surgical_m1ar" / "decode_round00.jsonl",
         "surgical_m8 vs surgical_m1ar")
    diff(RUN / "full_flag" / "decode_round00.jsonl",
         CEN / "full_flag_m1ar" / "decode_round00.jsonl",
         "full_flag_m8 vs full_flag_m1ar")


if __name__ == "__main__":
    main()
