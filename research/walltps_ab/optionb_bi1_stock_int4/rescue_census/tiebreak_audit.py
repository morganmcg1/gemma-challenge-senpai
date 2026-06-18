"""PR #654 Part 1: lowest-token-index tie-break direction audit on the 60 wide head ties.

The recompute acceptor resolves an EXACT tie by lowest token index (torch.argmax returns the
first index on equal values). At each on-AR-head "wide" break (recompute_margin in 0.125-0.25
nat = one-to-two int4 quanta, i.e. an int4-quantum tie per #651), we stored both candidate ids:
  a_pos = served-AR token id (== served == ar_ref_bi1 on the head)
  r_pos = M=1 prefill-recompute argmax id
If the candidates are treated as tied (int4 quantum apart), the acceptor's lowest-index rule
selects min(a_pos, r_pos). "AR wins" iff a_pos < r_pos -> the acceptor would have emitted the
AR token by its OWN deterministic rule -> identity-safe at that tie by construction.

Offline: reads break_locus_enriched_k{3,5,7}.json (no server). analysis_only.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = (3, 5, 7)


def main() -> int:
    per_k = {}
    all_wide = []
    for k in KS:
        p = HERE / f"break_locus_enriched_k{k}.json"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        d = json.loads(p.read_text())
        wide = [e for e in d["enriched"] if e.get("wide")]
        win = lose = degen = 0
        losses = []
        for e in wide:
            a, r = e["a_pos"], e["r_pos"]
            e["_k"] = k
            if a < r:
                win += 1
            elif a > r:
                lose += 1
                losses.append(e)
            else:
                degen += 1
            all_wide.append(e)
        per_k[k] = {"n_wide": len(wide), "AR_wins": win, "AR_loses": lose,
                    "degenerate": degen, "losses": losses}
        print(f"[K={k}] wide={len(wide):2d}  AR_wins(a<r)={win:2d}  "
              f"AR_loses(a>r)={lose}  degenerate(a==r)={degen}")

    tot = len(all_wide)
    win = sum(p["AR_wins"] for p in per_k.values())
    lose = sum(p["AR_loses"] for p in per_k.values())
    degen = sum(p["degenerate"] for p in per_k.values())
    print(f"\n[TOTAL] wide ties = {tot}  ->  AR wins lowest-index = {win}/{tot}  "
          f"AR loses = {lose}  degenerate = {degen}")

    if lose:
        print(f"\n=== {lose} LOSS(es): wide ties where the served-AR token does NOT win the "
              f"lowest-index tie-break (a_pos > r_pos) ===")
        print(f"  {'K':>2} {'id':26} {'pos':>4} {'vmarg':>6} {'rcmarg':>6} "
              f"{'a_pos':>8} {'r_pos':>8}  a_str | r_str")
        for e in all_wide:
            if e["a_pos"] > e["r_pos"]:
                print(f"  {e['_k']:>2} {e['id']:26} {e['pos']:>4} "
                      f"{e['verify_margin']:>6.3f} {e['recompute_margin']:>6.3f} "
                      f"{e['a_pos']:>8} {e['r_pos']:>8}  {e['a_str']!r} | {e['r_str']!r}")

    out = HERE / "tiebreak_audit.json"
    out.write_text(json.dumps({
        "n_wide_total": tot, "AR_wins_tiebreak": win, "AR_loses_tiebreak": lose,
        "degenerate": degen, "per_k": {str(k): {kk: vv for kk, vv in v.items() if kk != "losses"}
                                       for k, v in per_k.items()},
        "losses": [{"k": e["_k"], "id": e["id"], "pos": e["pos"], "a_pos": e["a_pos"],
                    "r_pos": e["r_pos"], "a_str": e["a_str"], "r_str": e["r_str"],
                    "verify_margin": e["verify_margin"],
                    "recompute_margin": e["recompute_margin"]}
                   for e in all_wide if e["a_pos"] > e["r_pos"]],
    }, indent=2))
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
