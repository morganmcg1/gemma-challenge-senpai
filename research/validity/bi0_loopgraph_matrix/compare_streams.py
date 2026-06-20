"""Greedy token-stream divergence comparator for the bi0-loopgraph matrix (PR #771).

LOCAL-only, read-only. Compares decode JSONL files produced by the matrix harness
by matching records on ``dataset_index`` and comparing ``completion_token_ids``.
For each pair it reports: #identical / #divergent prompts, total divergent tokens,
and the first-divergence index (onset) distribution.

The point of this script: the bi0 control runs ``VLLM_BATCH_INVARIANT=0`` and is
therefore non-deterministic run-to-run (non-associative FP reductions flip argmax
at near-ties). So "greedy identity vs the bi0 control" cannot be an exact-byte
claim against a fixed reference -- the reference itself moves. The honest proof is:
(1) the control's own rep-to-rep self-divergence ENVELOPE, and (2) that the variant
diverges from a control rep by NO MORE than that envelope, while (3) loopgraph_only
and loopgraph_fused are byte-identical to each other (fused-argmax is argmax-
preserving -> adds zero divergence).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median


def load(path: Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["dataset_index"]] = r["completion_token_ids"]
    return out


def compare(a: dict[int, list[int]], b: dict[int, list[int]]) -> dict:
    keys = sorted(set(a) & set(b))
    n_ident = n_div = 0
    tot_tokens = tot_div_tokens = 0
    onsets: list[int] = []
    for k in keys:
        ta, tb = a[k], b[k]
        m = min(len(ta), len(tb))
        tot_tokens += m
        first = None
        ndiff = 0
        for i in range(m):
            if ta[i] != tb[i]:
                ndiff += 1
                if first is None:
                    first = i
        if first is None and len(ta) == len(tb):
            n_ident += 1
        else:
            n_div += 1
            onsets.append(first if first is not None else m)
            tot_div_tokens += ndiff
    return {
        "n_compared": len(keys),
        "n_identical": n_ident,
        "n_divergent": n_div,
        "total_tokens": tot_tokens,
        "total_divergent_tokens": tot_div_tokens,
        "onset_min": min(onsets) if onsets else None,
        "onset_median": int(median(onsets)) if onsets else None,
        "onset_max": max(onsets) if onsets else None,
    }


def main(argv: list[str]) -> int:
    # argv: pairs of "label=path"; prints a comparison for every unordered pair.
    streams: dict[str, dict[int, list[int]]] = {}
    for item in argv:
        label, path = item.split("=", 1)
        streams[label] = load(Path(path))
        print(f"[loaded] {label}: {len(streams[label])} records  ({path})")
    labels = list(streams)
    print()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            c = compare(streams[la], streams[lb])
            verdict = "IDENTICAL" if c["n_divergent"] == 0 else "DIVERGENT"
            print(
                f"{la:24s} vs {lb:24s} [{verdict}]  "
                f"ident={c['n_identical']:3d}/{c['n_compared']:3d}  "
                f"div_tok={c['total_divergent_tokens']:6d}/{c['total_tokens']}  "
                f"onset(min/med/max)={c['onset_min']}/{c['onset_median']}/{c['onset_max']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
