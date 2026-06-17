#!/usr/bin/env python3
"""PR #581: cached GSM8K driver for the unquantized bf16 base denominator.

The shared ``research/downstream_quality_gsm8k/gsm8k_eval.py`` re-fetches the full
GSM8K test + train splits from the HF ``datasets-server`` /rows API at the start of
EVERY run. Four back-to-back arms (one no-guard sampled+greedy at seed 1234, plus
min_tokens=8 sampled at seeds 1234/1235/1236) issue ~60 paginated /rows calls in a
couple of minutes and trip a ``429 Too Many Requests`` rate limit, which kills a run
on dataset load.

This wrapper fetches each split ONCE (with exponential backoff), caches it to a
local JSON, monkeypatches ``gsm8k_eval._load_split`` to serve from that cache, then
runs each requested cell against an already-serving ``--base-url``. No edit to the
shared harness, so every arm scores through the identical 8-shot CoT / strict-match
path and stays comparable to the team's prior GSM8K cells.

Usage:
  run_gsm8k_cached.py --base-url http://127.0.0.1:8000 --out-dir <dir> [--n 500]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # research/validity/base_quality_denominator -> repo root
GSM8K_SRC = ROOT / "research" / "downstream_quality_gsm8k" / "gsm8k_eval.py"


def _load_gsm8k_module():
    spec = importlib.util.spec_from_file_location("gsm8k_eval", GSM8K_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))  # so its `from scripts...` import path resolves if ever hit
    spec.loader.exec_module(mod)
    return mod


def prefetch(ge, cache_path: Path) -> dict:
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        print(f"[cache] reuse {cache_path} "
              f"(test={len(data.get('test', []))} train={len(data.get('train', []))})",
              flush=True)
        return data
    cache: dict = {}
    # test: full split (n=None -> all 1319); train: enough for an 8-shot draw (>=64)
    for split, n in (("test", None), ("train", 256)):
        last = None
        for attempt in range(8):
            try:
                cache[split] = ge._load_split(split, n)
                print(f"[cache] fetched {split}: {len(cache[split])} rows", flush=True)
                break
            except Exception as exc:  # transient 429 / datasets-server hiccup
                last = exc
                wait = min(90, 8 * (attempt + 1))
                print(f"[cache] {split} attempt {attempt} failed: {exc!r}; sleep {wait}s",
                      flush=True)
                time.sleep(wait)
        else:
            raise SystemExit(f"[cache] FATAL: prefetch failed for {split}: {last!r}")
    cache_path.write_text(json.dumps(cache))
    print(f"[cache] wrote {cache_path}", flush=True)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="server root, e.g. http://127.0.0.1:8000")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seeds", default="1234,1235,1236", help="min_tokens=8 sampled seeds")
    ap.add_argument("--anchor-seed", type=int, default=1234,
                    help="seed for the no-guard sampled+greedy anchor (apples-to-apples subset)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ge = _load_gsm8k_module()
    cache = prefetch(ge, out_dir / "_gsm8k_dataset_cache.json")

    def cached_load_split(split: str, n):
        rows = cache[split]
        return rows[:n] if (n is not None and n >= 0) else list(rows)

    ge._load_split = cached_load_split  # monkeypatch: all runs now read from cache

    common = ["--base-url", args.base_url, "--model", args.model,
              "--n", str(args.n), "--n-shot", str(args.n_shot),
              "--concurrency", str(args.concurrency), "--out-dir", str(out_dir)]

    cells: list[list[str]] = []
    # no-guard sampled + greedy at the anchor seed (directly comparable to int4 base subset)
    cells.append(["--label", f"base_bf16_s{args.anchor_seed}_noguard",
                  "--seed", str(args.anchor_seed), "--regimes", "sampled,greedy"] + common)
    # min_tokens=8 sampled across seeds (the #581 mandated protocol)
    for s in [x.strip() for x in args.seeds.split(",") if x.strip()]:
        cells.append(["--label", f"base_bf16_s{s}_mt8",
                      "--seed", s, "--regimes", "sampled", "--min-tokens", "8"] + common)

    rc = 0
    for argv in cells:
        label = argv[argv.index("--label") + 1]
        print(f"\n[driver] === cell {label} :: {' '.join(argv)} ===", flush=True)
        try:
            r = ge.main(argv)
            rc = rc or r
        except SystemExit as e:
            rc = rc or (e.code or 0)
        except Exception as exc:
            print(f"[driver] cell {label} raised: {exc!r}", flush=True)
            rc = 1
    print(f"[driver] ALL CELLS DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
