#!/usr/bin/env python
"""Pairwise byte-exact census across all captured passes (PR #630 determinism matrix).

Loads the per-request token-id JSONs in out/ and computes a full pairwise
comparison so we can separate three determinism axes cleanly:

  * stock r1 vs r2          : prefix-cache ON, warm-pass self-determinism  (the 34/64 finding)
  * pcache_off r1 vs r2     : prefix-cache OFF, warm-pass self-determinism (mechanism test)
  * pcache_on r1 vs r2      : standalone control reproducing the ON condition in-harness
  * pcache_off r1 vs stock r1 : does disabling caching change the cold reference?
  * stock/flashinfer/flash_attn r1 : cold-pass cross-config (all TRITON_ATTN -> expect identical)

Pure analysis; no GPU, no serve. Run after the serve passes complete.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load_rows(name: str):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text())["per_request"]


def census(a, b):
    n = min(len(a), len(b))
    psha = sum(a[i]["prompt_token_sha256"] != b[i]["prompt_token_sha256"] for i in range(n))
    tok_mism, firstdiv = [], []
    for i in range(n):
        if a[i]["completion_token_sha256"] != b[i]["completion_token_sha256"]:
            ai = a[i].get("completion_token_ids") or []
            bi = b[i].get("completion_token_ids") or []
            m = min(len(ai), len(bi))
            pos = next((k for k in range(m) if ai[k] != bi[k]), m)
            tok_mism.append(i)
            firstdiv.append(pos)
    return {
        "n_compared": n,
        "n_prompt_sha_mismatch": psha,
        "prompt_sha_parity": psha == 0,
        "n_token_mismatch": len(tok_mism),
        "byte_exact": len(tok_mism) == 0 and psha == 0,
        "min_first_divergence": min(firstdiv) if firstdiv else None,
        "median_first_divergence": sorted(firstdiv)[len(firstdiv)//2] if firstdiv else None,
    }


PAIRS = [
    ("stock_r1", "stock_r2", "prefix-cache ON  : warm-pass self-determinism (main census)"),
    ("pcache_on_r1", "pcache_on_r2", "prefix-cache ON  : standalone control, warm-pass self-det"),
    ("pcache_off_r1", "pcache_off_r2", "prefix-cache OFF : warm-pass self-determinism (mechanism test)"),
    ("pcache_off_r1", "stock_r1", "cold cross-config: caching OFF vs stock cold reference"),
    ("pcache_on_r1", "stock_r1", "cold cross-config: standalone ON vs serve.py stock (harness check)"),
    ("flashinfer_r1", "stock_r1", "cold cross-config: FLASHINFER env vs stock (both TRITON_ATTN)"),
    ("flash_attn_r1", "stock_r1", "cold cross-config: FLASH_ATTN env vs stock (both TRITON_ATTN)"),
]


def main() -> int:
    results = {}
    print(f"{'pair':<34} {'byte_exact':>10} {'mismatch':>10} {'psha_par':>9} {'min_div':>8} {'med_div':>8}  note")
    for a, b, note in PAIRS:
        ra, rb = load_rows(f"pass_{a}.json") or load_rows(f"{a}.json"), load_rows(f"pass_{b}.json") or load_rows(f"{b}.json")
        if ra is None or rb is None:
            print(f"{a} vs {b:<20} {'--MISSING--':>10}  ({'have ' + a if ra else 'missing ' + a}/{'have ' + b if rb else 'missing ' + b})")
            continue
        c = census(ra, rb)
        results[f"{a}__vs__{b}"] = c
        print(f"{a+' vs '+b:<34} {str(c['byte_exact']):>10} "
              f"{str(c['n_token_mismatch'])+'/'+str(c['n_compared']):>10} "
              f"{str(c['prompt_sha_parity']):>9} {str(c['min_first_divergence']):>8} "
              f"{str(c['median_first_divergence']):>8}  {note}")
    (OUT / "determinism_matrix.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nwrote {OUT / 'determinism_matrix.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
