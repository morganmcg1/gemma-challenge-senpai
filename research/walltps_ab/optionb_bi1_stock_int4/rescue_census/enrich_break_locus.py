"""PR #651: enrich the on-AR-head break-locus census with token strings + a_pos rank.

rescue_capture stored, for each break, the recompute top-N as a STRING-keyed logprob dict
(vLLM returns top_logprobs by token string, not id). To answer PR step 3 'served-AR token
vs M=1-recompute token' we detokenize a_pos/r_pos/s_pos via the ref server's /detokenize and
look a_pos up in the stored break_top to get its logprob, rank, and gap below the recompute
top1. If a_pos is the recompute runner-up at ~0.125 nat, the 'wide' break is a 2-way int4
tie-flip (a_pos right behind r_pos), not a confident miss. analysis_only.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"


def detok(ids):
    out = {}
    for tid in ids:
        body = json.dumps({"model": MODEL, "tokens": [tid]}).encode()
        req = urllib.request.Request(f"{BASE}/detokenize", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            out[tid] = json.loads(r.read().decode())["prompt"]
    return out


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = [json.loads(l) for l in (HERE / f"rescue_k{k}.jsonl").read_text().splitlines() if l.strip()]
    head_brk = [r for r in rows if r["is_fire"] and r["pre_div"] and not r["rescue"]]
    ids = sorted({r["a_pos"] for r in head_brk} | {r["r_pos"] for r in head_brk}
                 | {r["s_pos"] for r in head_brk})
    s = detok(ids)

    enriched = []
    a_is_top2 = a_in_topN = a_missing = 0
    for r in head_brk:
        top = r.get("break_top") or {}
        vals = sorted(top.values(), reverse=True)
        top1 = vals[0] if vals else None
        a_str = s[r["a_pos"]]
        a_lp = top.get(a_str)
        rank = None
        if a_lp is not None:
            rank = 1 + sum(1 for v in top.values() if v > a_lp)
            if rank == 2:
                a_is_top2 += 1
            a_in_topN += 1
        else:
            a_missing += 1
        enriched.append({
            "id": r["id"], "pos": r["pos"], "verify_margin": r["verify_margin"],
            "recompute_margin": r["recompute_margin"],
            "a_pos": r["a_pos"], "a_str": a_str, "r_pos": r["r_pos"], "r_str": s[r["r_pos"]],
            "s_pos": r["s_pos"], "a_logprob": a_lp, "a_rank_in_recompute": rank,
            "top1_minus_a": (top1 - a_lp) if (a_lp is not None and top1 is not None) else None,
            "wide": bool(r["recompute_margin"] is not None and r["recompute_margin"] > 1e-6),
        })

    wide = [e for e in enriched if e["wide"]]
    print(f"[enrich K={k}] {len(head_brk)} on-AR-head breaks ; wide={len(wide)}")
    print(f"  a_pos present in recompute top-N: {a_in_topN}/{len(head_brk)} "
          f"(missing {a_missing}) ; a_pos is recompute top2: {a_is_top2}")
    print(f"\n  WIDE head breaks (a_pos vs r_pos with gap):")
    print(f"  {'id':26} {'pos':>4} {'vmarg':>6} {'rcmarg':>6} {'a_rank':>6} {'top1-a':>7}  a_str | r_str")
    for e in sorted(wide, key=lambda x: x["pos"]):
        t1a = f"{e['top1_minus_a']:.4f}" if e["top1_minus_a"] is not None else "NA"
        ar = e["a_rank_in_recompute"] if e["a_rank_in_recompute"] is not None else "miss"
        print(f"  {e['id']:26} {e['pos']:>4} {e['verify_margin']:>6.3f} {e['recompute_margin']:>6.3f} "
              f"{str(ar):>6} {t1a:>7}  {e['a_str']!r} | {e['r_str']!r}")
    out = HERE / f"break_locus_enriched_k{k}.json"
    out.write_text(json.dumps({"k": k, "n_head_breaks": len(head_brk), "n_wide": len(wide),
                               "a_in_topN": a_in_topN, "a_is_top2": a_is_top2,
                               "a_missing": a_missing, "enriched": enriched}, indent=2))
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
