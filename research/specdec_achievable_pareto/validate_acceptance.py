#!/usr/bin/env python
"""Independent cross-check of the ngram offline-acceptance sweep (PR #584).

Re-implements vLLM-NgramProposer exact-greedy-verify semantics from scratch and
runs it over the SAVED base_fullhead no-spec reference decode, so the verdict
number (ngram_max_acceptance) does not depend on the still-running driver.
Pure CPU; does not touch the GPU.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WARM = 4  # driver warm_discarded
# #575 measured verify-cost C(M) ms (drafter-independent; M=K+1)
C_MEASURED = {1: 11.4960, 2: 12.9621, 3: 13.0065, 5: 13.1367, 8: 13.1369, 9: 13.3716, 17: 14.2623}
C_FIT_INTERCEPT, C_FIT_SLOPE = 12.704375493740388, 0.08462453266874165
TAU_LO = 481.53 / 465.14047160458415
SHIP_TPS = 375.857
A_SHIP_573 = 2.6805993589363384


def c_of_m(m):
    return C_MEASURED[m] if m in C_MEASURED else C_FIT_INTERCEPT + C_FIT_SLOPE * m


def ngram_propose(context, n_max, n_min, k):
    L = len(context)
    for n in range(n_max, n_min - 1, -1):
        if L < n + 1:
            continue
        pattern = context[L - n:]
        for s in range(L - n - 1, -1, -1):
            if context[s:s + n] == pattern:
                draft = context[s + n:s + n + k]
                if draft:
                    return draft
    return []


def sim_seq(prompt_ids, completion_ids, n_max, n_min, k):
    context = list(prompt_ids)
    G = completion_ids
    n = len(G)
    i = draft_steps = no_draft_steps = accepted = 0
    while i < n:
        draft = ngram_propose(context, n_max, n_min, k)
        if not draft:
            no_draft_steps += 1
            context.append(G[i]); i += 1; continue
        a = 0
        cap = min(len(draft), n - i)
        while a < cap and draft[a] == G[i + a]:
            a += 1
        draft_steps += 1
        accepted += a
        emit = min(a + 1, n - i)
        context.extend(G[i:i + emit]); i += emit
    return draft_steps, no_draft_steps, accepted, n


def main():
    ref = json.load(open(HERE / "ref_pass0.json"))
    rows = ref["per_request"][WARM:]
    print(f"warm rows: {len(rows)} (dropped first {WARM})")
    n_list = [2, 3, 4]
    k_list = [3, 5, 7, 10]
    # ref no-spec local TPS from the saved report
    rep = json.load(open(HERE / "pareto_report.json"))
    ref_local = rep["ngram_served"]["3"]["tps"]  # placeholder; use ref below
    ref_local = rep.get("ref_local_tps")
    t1_ms = 1000.0 / ref_local if ref_local else float("nan")
    print(f"ref no-spec local TPS = {ref_local:.3f}  (t1 = {t1_ms:.3f} ms/step)\n")
    print(f"{'n':>2} {'k':>3} {'M':>3} {'e_accept':>9} {'coverage':>9} {'avg_tok/s':>9} "
          f"{'C(M)ms':>7} {'real_loc':>9} {'proj_off':>9} {'ge2.68?':>7}")
    best = (-1, None)
    grid = {}
    for n in n_list:
        for k in k_list:
            tot_d = tot_u = tot_a = 0
            for r in rows:
                d, u, a, _ = sim_seq(r["prompt_token_ids"], r["completion_token_ids"], n, 2, k)
                tot_d += d; tot_u += u; tot_a += a
            e_acc = 1.0 + tot_a / tot_d if tot_d else float("nan")
            cov = tot_d / (tot_d + tot_u) if (tot_d + tot_u) else float("nan")
            m = k + 1
            c_ms = c_of_m(m)
            avg_tok = cov * e_acc + (1 - cov) * 1.0
            avg_ms = cov * c_ms + (1 - cov) * t1_ms
            real_loc = 1000.0 * avg_tok / avg_ms if avg_ms > 0 else float("nan")
            proj_off = real_loc * TAU_LO
            grid[(n, k)] = dict(e_accept=e_acc, coverage=cov, real_loc=real_loc, proj_off=proj_off, c_ms=c_ms)
            if e_acc > best[0]:
                best = (e_acc, (n, k))
            print(f"{n:>2} {k:>3} {m:>3} {e_acc:>9.4f} {cov:>9.4f} {avg_tok:>9.4f} "
                  f"{c_ms:>7.3f} {real_loc:>9.2f} {proj_off:>9.2f} {str(e_acc >= A_SHIP_573):>7}")
    ngram_max_acc = best[0]
    best_proj = max(g["proj_off"] for g in grid.values())
    print(f"\nngram_max_acceptance = {ngram_max_acc:.4f}  at (n,k)={best[1]}")
    print(f"ngram_clears_268 (>= {A_SHIP_573:.4f}) = {ngram_max_acc >= A_SHIP_573}")
    print(f"best_ngram_projected_off_tps = {best_proj:.2f}   ship = {SHIP_TPS}")
    print(f"any ngram clears ship (clean frame, real_loc*tau > ship) = "
          f"{any(g['proj_off'] > SHIP_TPS for g in grid.values())}")
    # cross-check vs served ngram TPS (n=2 configs)
    print("\n--- cross-check: offline energy-reconstructed vs SERVED ngram TPS (n=2) ---")
    for k in k_list:
        served = rep["ngram_served"].get(str(k), {}).get("tps", {}).get("warm_median_tps")
        recon = grid[(2, k)]["real_loc"]
        print(f"  k={k:>2}: served_warm_median={served:>7.2f}  offline_recon={recon:>7.2f}  "
              f"ratio={recon/served if served else float('nan'):.3f}")


if __name__ == "__main__":
    main()
