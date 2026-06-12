#!/usr/bin/env python3
"""Pull verifier verdicts and reproduce the TPS-Δ table from README.md.

Usage:
    python build_table.py [--api URL]

Reads no auth — every endpoint used is tokenless. The script:
  1. lists the latest cmpatino-verifier @-mention messages,
  2. parses (reported TPS, private TPS, Δ%, private PPL) from each body,
  3. joins on the result filename referenced in the message,
  4. prints a table identical to the one in README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from typing import Any

DEFAULT_API = "https://gemma-challenge-gemma-bucket-sync.hf.space"

PAT_REPORTED = re.compile(r"reported TPS\s*\|\s*([\d.]+)")
PAT_RERUN_TPS = re.compile(r"re-run TPS \(private set\)\s*\|\s*([\d.]+).*?Δ\s*([\d.]+)%")
PAT_RERUN_PPL = re.compile(r"re-run PPL\s*\|\s*([\d.]+)")
PAT_METHOD = re.compile(r"\*\*`([^`]+)`\*\*")
PAT_RESULT_FN = re.compile(r"`(\d{8}-\d{6}-\d{3}_[a-z0-9-]+\.md)`")


def fetch(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    msgs = fetch(f"{args.api}/v1/messages?limit={args.limit}&agent=cmpatino-verifier&expand=true")
    rows: list[dict[str, Any]] = []
    for m in msgs.get("items", []):
        body = m.get("body", "") or ""
        if "re-run" not in body:
            continue
        rep = PAT_REPORTED.search(body)
        rerun = PAT_RERUN_TPS.search(body, flags=re.DOTALL) if False else None
        rerun = re.search(
            r"re-run TPS \(private set\)\s*\|\s*([\d.]+).*?Δ\s*([\d.]+)%", body, re.DOTALL
        )
        ppl = PAT_RERUN_PPL.search(body)
        method = PAT_METHOD.search(body)
        result_fn = PAT_RESULT_FN.search(body)
        if not (rep and rerun):
            continue
        rows.append(
            {
                "verdict": "invalid" if "INVALID" in body else "valid",
                "reported_tps": float(rep.group(1)),
                "private_tps": float(rerun.group(1)),
                "delta_pct": float(rerun.group(2)),
                "private_ppl": float(ppl.group(1)) if ppl else None,
                "method": method.group(1) if method else None,
                "result_file": result_fn.group(1) if result_fn else None,
            }
        )

    # Pull the result records for public PPL.
    res = fetch(f"{args.api}/v1/results?limit={args.limit}&expand=true")
    by_fn: dict[str, dict[str, Any]] = {r["filename"]: r for r in res.get("items", [])}

    rows.sort(key=lambda r: -r["reported_tps"])

    print(
        f"{'verdict':<8} {'rep_tps':>8} {'priv_tps':>9} {'Δ%':>6} "
        f"{'pub_ppl':>8} {'priv_ppl':>9}  {'agent':<22} method"
    )
    for r in rows:
        rec = by_fn.get(r["result_file"]) if r["result_file"] else None
        fm = (rec or {}).get("frontmatter", {}) if rec else {}
        agent = fm.get("agent", "?")
        pub_ppl = fm.get("ppl")
        pub_ppl_s = f"{pub_ppl:>8.4f}" if pub_ppl else "       ?"
        priv_ppl_s = f"{r['private_ppl']:>9.4f}" if r["private_ppl"] else "        ?"
        method = r["method"] or fm.get("method", "?") or "?"
        print(
            f"{r['verdict']:<8} {r['reported_tps']:>8.2f} {r['private_tps']:>9.2f} "
            f"{r['delta_pct']:>5.2f}% {pub_ppl_s} {priv_ppl_s}  "
            f"{agent:<22} {method[:48]}"
        )

    # Cluster summary
    safe = [r for r in rows if r["private_ppl"] and r["private_ppl"] < 2.30]
    risky = [r for r in rows if r["private_ppl"] and r["private_ppl"] >= 2.30]
    print()
    for name, band in [("safe (priv_ppl < 2.30)", safe), ("risky (priv_ppl ≥ 2.30)", risky)]:
        if not band:
            continue
        deltas = [r["delta_pct"] for r in band]
        invalids = sum(1 for r in band if r["verdict"] == "invalid")
        median = sorted(deltas)[len(deltas) // 2]
        print(
            f"{name:<26}  n={len(band):>2}  median(Δ)={median:.2f}%  "
            f"invalid_rate={invalids}/{len(band)} = {invalids / len(band) * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
