#!/usr/bin/env python3
"""PR #581 diagnostic: capture GSM8K greedy generations (--save-text) against a
running server, serving the dataset from the prefetched cache (no datasets-server
hit). Used to A/B the serve.sh override server vs the clean int4-recipe server on
the SAME problems and characterize WHY the bf16 base scores below int4 on GSM8K.

No edit to the shared gsm8k_eval harness: we monkeypatch _load_split to read the
local cache, then call gsm8k_eval.main with --save-text so per_problem carries the
raw completion text.

Usage:
  diag_gsm8k.py --base-url http://127.0.0.1:8000 --label diag_override --n 100 [--regimes greedy]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GSM8K_SRC = ROOT / "research" / "downstream_quality_gsm8k" / "gsm8k_eval.py"


def load_mod():
    spec = importlib.util.spec_from_file_location("gsm8k_eval", GSM8K_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--regimes", default="greedy")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    a = ap.parse_args()

    ge = load_mod()
    cache_path = Path(a.out_dir) / "_gsm8k_dataset_cache.json"
    cache = json.loads(cache_path.read_text())

    def cached_load_split(split: str, n):
        rows = cache[split]
        return rows[:n] if (n is not None and n >= 0) else list(rows)

    ge._load_split = cached_load_split

    argv = ["--base-url", a.base_url, "--model", "gemma-4-e4b-it",
            "--label", a.label, "--regimes", a.regimes, "--limit", str(a.n),
            "--n-shot", "8", "--seed", str(a.seed), "--concurrency", "32",
            "--save-text", "--out-dir", a.out_dir]
    return ge.main(argv) or 0


if __name__ == "__main__":
    raise SystemExit(main())
