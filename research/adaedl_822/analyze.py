#!/usr/bin/env python3
"""Step-1 analysis of AdaEDL draft-entropy records (#822).

Reads the JSONL emitted by adaedl_earlystop_patch.py (ADAEDL_OUT) when run in
the CONTROL (DRAFT_STOP_ENTROPY=inf, never stops -> full K=6 draft every step).
Each record:
  {"step", "K", "draft_len", "accept_length", "stopped", "thresh",
   "H": [H_1 .. H_{draft_len}]}   # per-position drafter entropy (nats)

Validates the AdaEDL premise (rejected tail positions have HIGH drafter entropy)
and computes the offline early-stop counterfactual for a grid of thresholds tau:
  L(tau)        = first position j with H_j > tau, else K          (realized draft len)
  realized_acc  = min(accept_length, L(tau))                       (accept under early-stop)
  forwards_done = L(tau)                                           (drafter forwards run)
Early-stop is accept-free iff L(tau) >= accept_length+1 (we only stop at/after the
reject point). The sweet spot: forwards_done << K with accept_loss ~ 0.
"""
import argparse
import json
import math
from collections import defaultdict


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--taus", default=None,
                    help="comma list of thresholds to tabulate (default: derived percentiles)")
    args = ap.parse_args()

    recs = load(args.records)
    full = [r for r in recs if r.get("H") and len(r["H"]) == r.get("draft_len")]
    if not recs:
        print("NO RECORDS")
        return
    K = max(r["K"] for r in recs)
    print(f"records={len(recs)}  usable(H present)={len(full)}  K={K}")

    # control sanity: in control inf, draft_len should == K every step
    dls = [r["draft_len"] for r in recs]
    print(f"draft_len: min={min(dls)} max={max(dls)} mean={sum(dls)/len(dls):.3f} "
          f"(control inf expects all == K={K})")

    accs = [r["accept_length"] for r in recs]
    E_accept = sum(accs) / len(accs)
    print(f"E_accept (mean accept_length) = {E_accept:.4f}   r = E/K = {E_accept/K:.4f}")
    # accept_length histogram
    hist = defaultdict(int)
    for a in accs:
        hist[a] += 1
    print("accept_length hist: " + "  ".join(f"{a}:{hist[a]}" for a in sorted(hist)))

    # ---- per-position accept rate + entropy (fern #774 shape) ----
    print("\n--- per-position (j=1..K): accept rate + mean drafter entropy ---")
    print(f"{'pos':>3} {'n_prop':>7} {'P(acc_j)':>9} {'meanH_j':>9} {'p50H':>8} {'p85H':>8}")
    per_pos_H = defaultdict(list)
    for r in full:
        a = r["accept_length"]
        for j, h in enumerate(r["H"], start=1):
            per_pos_H[j].append((h, 1 if j <= a else 0))
    for j in range(1, K + 1):
        rows = per_pos_H.get(j, [])
        if not rows:
            continue
        Hs = sorted(h for h, _ in rows)
        pacc = sum(acc for _, acc in rows) / len(rows)
        meanH = sum(Hs) / len(Hs)
        print(f"{j:>3} {len(rows):>7} {pacc:>9.4f} {meanH:>9.4f} "
              f"{pct(Hs,50):>8.4f} {pct(Hs,85):>8.4f}")

    # ---- mean H: accepted positions (j<=a) vs the reject point (j==a+1) ----
    H_acc, H_rej = [], []
    for r in full:
        a = r["accept_length"]
        for j, h in enumerate(r["H"], start=1):
            if j <= a:
                H_acc.append(h)
            elif j == a + 1:
                H_rej.append(h)
    if H_acc and H_rej:
        ma = sum(H_acc) / len(H_acc)
        mr = sum(H_rej) / len(H_rej)
        print(f"\nmean H @ accepted positions (j<=a) = {ma:.4f}  (n={len(H_acc)})")
        print(f"mean H @ reject point      (j==a+1) = {mr:.4f}  (n={len(H_rej)})")
        print(f"separation (reject - accept)        = {mr-ma:+.4f}   "
              f"{'PREMISE HOLDS (reject higher)' if mr>ma else 'PREMISE FAILS'}")

    # ---- P(accept_j | H_j bucket): pooled predictive check ----
    all_HA = [(h, acc) for j in per_pos_H for (h, acc) in per_pos_H[j]]
    Hs_all = sorted(h for h, _ in all_HA)
    edges = [pct(Hs_all, p) for p in (0, 20, 40, 60, 80, 100)]
    print("\n--- P(accept_j | H_j bucket) pooled over positions ---")
    print(f"{'H bucket':>22} {'n':>7} {'P(acc)':>8} {'meanH':>8}")
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        sel = [(h, a) for h, a in all_HA if (h >= lo and (h < hi or b == len(edges) - 2))]
        if not sel:
            continue
        pacc = sum(a for _, a in sel) / len(sel)
        mh = sum(h for h, _ in sel) / len(sel)
        print(f"[{lo:>8.3f},{hi:>8.3f}] {len(sel):>7} {pacc:>8.4f} {mh:>8.4f}")

    # ---- early-stop counterfactual over thresholds ----
    if args.taus:
        taus = [float(x) for x in args.taus.split(",")]
    else:
        ph = sorted(h for h, _ in all_HA)
        taus = sorted(set(round(pct(ph, p), 3) for p in (30, 50, 60, 70, 80, 85, 90, 95)))
    print("\n--- early-stop counterfactual (offline, from control H sequences) ---")
    print(f"baseline: E_accept={E_accept:.4f}  forwards/step=K={K}")
    print(f"{'tau':>8} {'fwd/step':>9} {'fwd_saved':>10} {'real_acc':>9} "
          f"{'acc_loss':>9} {'acc_loss%':>9} {'stop_rate':>9}")
    for tau in taus:
        fwds, racc, stopped = [], [], 0
        for r in full:
            H = r["H"]
            a = r["accept_length"]
            L = K
            for j, h in enumerate(H, start=1):
                if h > tau:
                    L = j
                    break
            if L < K:
                stopped += 1
            fwds.append(L)
            racc.append(min(a, L))
        mf = sum(fwds) / len(fwds)
        mra = sum(racc) / len(racc)
        loss = E_accept - mra
        print(f"{tau:>8.3f} {mf:>9.4f} {K-mf:>10.4f} {mra:>9.4f} "
              f"{loss:>9.4f} {100*loss/E_accept:>8.3f}% {stopped/len(full):>9.4f}")


if __name__ == "__main__":
    main()
