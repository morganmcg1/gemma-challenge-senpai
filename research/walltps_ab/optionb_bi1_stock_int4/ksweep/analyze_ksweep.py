#!/usr/bin/env python3
"""PR #632 K-sweep analysis (CPU only, reads the A/B artifacts).

For each K in the sweep it collects the n=3 fresh-server records (median wall_tps,
mean acceptance length, accept/draft token counts), then:
  * net wall_tps vs K  ->  K* = argmax
  * cycle_time(K) = e_accept / wall_tps  (s/cycle); a linear fit cycle_time = K*c_draft
    + c_verify empirically splits the per-draft-forward (M=1, BI-taxed) cost from the
    verify+overhead cost -- the physical reason a lower K can net more TPS.
  * acceptance structure: accepted-draft/cycle, per-draft accept rate, implied alpha.
  * draft-forward burden K/e_accept (M=1 BI-taxed forwards per emitted token).

Records are pulled from:
  * k3/records.jsonl            -> arms 'k7' (baseline) and 'k3' (candidate)
  * k4/paired_ab.json,...       -> arms.candidate.records  (k7 baseline reused)
  * k2/paired_ab.json (if run)
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

KS = Path(__file__).resolve().parent
ANCHOR_K7 = 152.29          # #623 banked BI=1 K=7 median
LOCKED_RUNG = 126.378       # strict-#319 AR official rung


def _recs_from_jsonl(p: Path):
    out = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.setdefault(int(r["num_speculative_tokens"]), []).append(r)
    return out


def _recs_from_paired(p: Path):
    out = {}
    if not p.exists():
        return out
    data = json.loads(p.read_text())
    for side in ("baseline", "candidate"):
        for r in (data.get("arms", {}).get(side, {}) or {}).get("records", []) or []:
            out.setdefault(int(r["num_speculative_tokens"]), []).append(r)
    return out


def collect():
    byk: dict[int, list] = {}
    def merge(d):
        for k, rs in d.items():
            byk.setdefault(k, [])
            # de-dup by (arm, run_idx, t_start) so reused baselines don't double count
            seen = {(x.get("arm"), x.get("run_idx"), x.get("t_start_utc")) for x in byk[k]}
            for r in rs:
                key = (r.get("arm"), r.get("run_idx"), r.get("t_start_utc"))
                if key not in seen:
                    byk[k].append(r); seen.add(key)
    merge(_recs_from_jsonl(KS / "k3" / "records.jsonl"))
    for kk in (2, 4, 5, 6):
        merge(_recs_from_paired(KS / f"k{kk}" / "paired_ab.json"))
        merge(_recs_from_jsonl(KS / f"k{kk}" / "records.jsonl"))
    return byk


def summarize(byk):
    rows = []
    for k in sorted(byk):
        rs = byk[k]
        wtps = [r["wall_tps"] for r in rs]
        eacc = [r["e_accept_exact"] for r in rs]
        ta = sum(r.get("total_accepted_tokens", 0) for r in rs)
        td = sum(r.get("total_drafted_tokens", 0) for r in rs)
        med = statistics.median(wtps)
        cyc = statistics.mean(eacc) / med            # s/cycle at the median tps
        rows.append({
            "K": k, "n": len(wtps),
            "wall_tps_median": med,
            "wall_tps_mean": statistics.mean(wtps),
            "wall_tps_std": statistics.pstdev(wtps) if len(wtps) > 1 else 0.0,
            "e_accept_mean": statistics.mean(eacc),
            "accept_per_draft": (ta / td) if td else float("nan"),
            "accept_draft_per_cycle": statistics.mean(eacc) - 1.0,   # minus bonus token
            "draft_fwd_per_tok": k / statistics.mean(eacc),
            "cycle_time_ms": cyc * 1e3,
        })
    return rows


def fit_cost(rows):
    """cycle_time = K*c_draft + c_verify  (least squares over available K)."""
    if len(rows) < 2:
        return None
    xs = [r["K"] for r in rows]
    ys = [r["cycle_time_ms"] for r in rows]
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    den = n*sxx - sx*sx
    if den == 0:
        return None
    c_draft = (n*sxy - sx*sy) / den
    c_verify = (sy - c_draft*sx) / n
    return {"c_draft_ms": c_draft, "c_verify_ms": c_verify}


def main():
    byk = collect()
    if not byk:
        print("no records yet"); return 0
    rows = summarize(byk)
    print(f"{'K':>3} {'n':>2} {'wall_tps_med':>12} {'mean±std':>14} {'e_accept':>9} "
          f"{'acc/draft':>9} {'cyc_ms':>7} {'draftfwd/tok':>12}")
    for r in rows:
        print(f"{r['K']:>3} {r['n']:>2} {r['wall_tps_median']:>12.3f} "
              f"{r['wall_tps_mean']:>8.2f}±{r['wall_tps_std']:>4.2f} {r['e_accept_mean']:>9.4f} "
              f"{r['accept_per_draft']:>9.4f} {r['cycle_time_ms']:>7.2f} {r['draft_fwd_per_tok']:>12.3f}")
    kstar = max(rows, key=lambda r: r["wall_tps_median"])
    k7 = next((r for r in rows if r["K"] == 7), None)
    print(f"\nK* (argmax median wall_tps) = {kstar['K']}  -> {kstar['wall_tps_median']:.3f} tps")
    if k7:
        d = kstar["wall_tps_median"] - k7["wall_tps_median"]
        print(f"K=7 median = {k7['wall_tps_median']:.3f} tps  (#623 anchor {ANCHOR_K7})")
        print(f"K* vs K7: {d:+.3f} tps ({100*d/k7['wall_tps_median']:+.2f}%)  beats_k7={kstar['K']!=7 and d>0}")
        print(f"K* vs locked-rung {LOCKED_RUNG}: {kstar['wall_tps_median']-LOCKED_RUNG:+.3f} tps "
              f"({100*(kstar['wall_tps_median']-LOCKED_RUNG)/LOCKED_RUNG:+.2f}%)")
    cost = fit_cost(rows)
    if cost:
        cd, cv = cost["c_draft_ms"], cost["c_verify_ms"]
        print(f"\ncost fit: cycle_time(K) = {cd:.3f}*K + {cv:.3f} ms  "
              f"(c_draft={cd:.3f} ms/M1-fwd, c_verify+overhead={cv:.3f} ms)")
        print(f"  draft/verify cost ratio r = {cd/cv:.3f}")
        # continuous-optimum scan with measured costs + measured e_accept(K)
        ea = {r["K"]: r["e_accept_mean"] for r in rows}
        print("  model net_tps(K) = e_accept(K)/(c_draft*K + c_verify):")
        for r in rows:
            k = r["K"]; pred = ea[k]/(cd*k + cv)*1e3
            print(f"    K={k}: predicted {pred:.2f} tps   measured {r['wall_tps_median']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
