#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""entropy_controller (PR #54): drafter-entropy-keyed dynamic draft-length (K),
the AdaEDL (arXiv:2410.18351) early-draft-stop lever -- designed + simulated
against a per-position trace that now carries the drafter's DRAFT-TIME entropy
(captured by `eval_eagle3.py --confidence`), and scored on the post-#43 split-KV
verify-cost curve for the DEPLOYED *linear* MTP K=7 chain.

WHY (the corrected accepthist lever).  PR #51 established, on the post-#43 curve,
that the dynamic-K *oracle* ceiling is real but modest and that the realizable
acceptance-HISTORY controller captures almost none of it (window-mean->next-run
r~0.32; ~0.7% of oracle).  #51's own #1 follow-up: acceptance history is the wrong
state -- the drafter's entropy AT DRAFT TIME is a strictly stronger, contemporaneous
per-step predictor of whether the chain is about to be rejected (AdaEDL; the public
top-3 rock-ai/pupa/need-for-speed key dynamic-K on it).  This module answers the
single decisive question #51 scoped out:

    does corr(drafter-entropy, run-length) materially beat the r~0.32 of
    acceptance history -- enough to recover a meaningful slice of the oracle?

DEPLOYED STRUCTURE (premise corrected by wirbel, propagated to #51).  The served
`fa2sw_precache_kenyan` drafter is a LINEAR MTP K=7 chain (verify M = K_drafted+1,
W=1), NOT a width-4 tree (vLLM 0.22 has no tree-verify path).  So the controller
here EARLY-STOPS a linear chain: draft tokens greedily at chain positions
0..Kmax-1; after proposing token s, look at the drafter's entropy H_s; if it says
"uncertain" (a rejection is coming) stop, capping K_drafted=s+1.  Verify M=K+1,
accept the longest greedy-matching prefix, emit min(run,K)+1.  Because the verifier
still emits exactly the greedy continuation, token identity (hence PPL=2.377) is
preserved -> leaderboard-legal (official gate = PPL+completion+modalities; #38).

PREDICTOR (Task 2, the gate).  For every trace position j with signal S_j and
acceptance a_j=(hit_rank_j==1) we report Pearson(S_j,a_j), the AUC of S_j as an
accept/reject classifier, and Pearson(S_j, remaining-run-from-j) -- head to head
across S in {entropy(full), entropy64(sparse-head realistic), top1p(Max-Conf-SPD),
margin}, and against the acceptance-history window-mean->next-run r recomputed on
THIS linear trace (apples to apples).  AdaEDL's Pinsker stop  1-sqrt(g*H)<lam  is
algebraically an entropy threshold  H > (1-lam)^2/g, so Variant-B reduces to the
entropy threshold sweep; reported as such.

CONTROLLER (Task 3).  Sweep the stop threshold for each signal; score steady-state
TPS on the measured curve vs (a) static K=7, (b) best static K*, (c) the #51
acceptance-history controllers (AIMD / window-mean, re-scored on the linear trace),
(d) the clairvoyant oracle (DP).  Primary metric = fraction of the linear-chain
oracle ceiling recovered.  Drafter cost is modelled BOTH linear-in-K (sequential
MTP modules: early-stop saves drafter passes too) and fixed (parallel MTP: early-
stop saves only verify) to bracket the served regime.  Anchoring: raw trace level
+ optional conservative i.i.d.-thinned lower levels (Task: served e_accept~3.82).

CPU-ONLY.  Inputs: the entropy trace JSONL + a measured cost-curve JSON.  No GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_acceptance_model import (  # noqa: E402
    LatencyCurve,
    load_latency_curve,
    load_trace,
)
from accepthist_controller import AIMD, WindowMeanLinear, WindowMeanLUT, learn_lut  # noqa: E402


# --------------------------------------------------------------------------- #
# trace -> per-position arrays
# --------------------------------------------------------------------------- #
SIGNALS = ("entropy", "entropy64", "top1p", "margin")
# orientation: does a LARGER value mean MORE likely to be accepted?  entropy and
# entropy64 are inverse (high entropy -> reject); top1p / margin are direct.
SIGNAL_DIRECTION = {"entropy": -1, "entropy64": -1, "top1p": +1, "margin": +1}


def load_entropy_trace(path: str):
    traces, meta = load_trace(path)
    seqs = []
    for tr in traces:
        hr = tr.get("hit_rank") or []
        if not hr:
            continue
        rec = {"hit_rank": hr, "accept": [1 if r == 1 else 0 for r in hr]}  # W=1 linear
        for s in SIGNALS:
            if s in tr:
                rec[s] = tr[s]
        seqs.append(rec)
    have_conf = all(all(s in seq for s in SIGNALS) for seq in seqs) and bool(seqs)
    return seqs, meta, have_conf


def remaining_run(accept: list[int], cap: int | None = None) -> list[int]:
    """remaining_run[j] = # consecutive accepts starting at j (optionally capped)."""
    n = len(accept)
    run = [0] * n
    nxt = 0
    for i in range(n - 1, -1, -1):
        nxt = nxt + 1 if accept[i] else 0
        run[i] = nxt if cap is None else min(nxt, cap)
    return run


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx * dy > 0 else float("nan")


def auc_mann_whitney(scores, labels):
    """AUC of `scores` predicting label==1, oriented so higher score => label 1.
    Rank-based (handles ties by mid-rank).  Returns AUC in [0,1]."""
    pairs = sorted(zip(scores, labels), key=lambda z: z[0])
    n = len(pairs)
    # mid-ranks
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0  # 1-indexed mid-rank
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(1 for _, l in pairs if l == 1)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    sum_pos = sum(rk for rk, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def decile_calibration(sig, accept, nbins=10):
    """mean accept rate per signal-quantile bin (ascending signal)."""
    order = sorted(range(len(sig)), key=lambda i: sig[i])
    n = len(order)
    out = []
    for b in range(nbins):
        lo, hi = b * n // nbins, (b + 1) * n // nbins
        idx = order[lo:hi]
        if not idx:
            out.append(None)
            continue
        out.append({
            "sig_lo": round(sig[idx[0]], 4), "sig_hi": round(sig[idx[-1]], 4),
            "accept_rate": round(sum(accept[i] for i in idx) / len(idx), 4),
            "n": len(idx)})
    return out


def predictor_report(seqs):
    """Task 2: entropy & proxy predictors of per-token accept and remaining-run,
    head-to-head, plus the acceptance-history window-mean baseline (#51)."""
    Kmax = 7
    # pooled per-position arrays
    acc_all, sig_all = [], {s: [] for s in SIGNALS}
    run_all = []
    for seq in seqs:
        a = seq["accept"]
        acc_all += a
        run_all += remaining_run(a, cap=Kmax)
        for s in SIGNALS:
            sig_all[s] += seq[s]
    rep = {"n_positions": len(acc_all), "base_accept_top1": sum(acc_all) / len(acc_all)}
    rep["signals"] = {}
    for s in SIGNALS:
        x = sig_all[s]
        d = SIGNAL_DIRECTION[s]
        oriented = [d * v for v in x]                  # higher => more likely accept
        rep["signals"][s] = {
            "pearson_accept": pearson(x, acc_all),     # signed, native orientation
            "pearson_run": pearson(x, run_all),
            "auc_accept": auc_mann_whitney(oriented, acc_all),
            "mean": round(st.mean(x), 4), "sd": round(st.pstdev(x), 4),
            "calibration_decile": decile_calibration(x, acc_all),
        }
    # ---- acceptance-history baseline (#51), recomputed on THIS linear trace ---
    # per-invoke accepted_len under static K=Kmax, then window-mean -> next run
    hist = {}
    for w in (2, 4, 8, 16):
        xs, ys = [], []
        for seq in seqs:
            a = seq["accept"]
            # walk linear chain at static Kmax to get per-invoke accepted_len
            lens = []
            pos, n = 0, len(a)
            while pos < n:
                m = 0
                while m < Kmax and pos + m < n and a[pos + m]:
                    m += 1
                lens.append(m)
                pos += min(m + 1, n - pos)
            for i in range(w, len(lens)):
                xs.append(sum(lens[i - w:i]) / w)
                ys.append(lens[i])
        hist[f"w{w}"] = {"r": pearson(xs, ys), "n": len(xs)}
    rep["accepthist_window_run_r"] = hist
    rep["accepthist_best_r"] = max((v["r"] for v in hist.values()
                                    if not math.isnan(v["r"])), default=float("nan"))
    return rep


# --------------------------------------------------------------------------- #
# controllers on the linear chain (early-stop) + cost model
# --------------------------------------------------------------------------- #
def drafter_ms(K, step_ms, mode, Kmax):
    """linear: sequential MTP modules, cost ~ K steps.  fixed: parallel MTP head,
    one pass regardless of K (early-stop saves only verify)."""
    return step_ms * K if mode == "linear" else step_ms * Kmax


def verify_ms(K, curve):
    return curve.at(K + 1)          # M = K_drafted + 1 (W=1 linear)


def run_static(accept, K):
    """traverse one sequence at fixed draft depth K; return per-invoke K_drafted."""
    n = len(accept)
    pos = 0
    Ks = []
    while pos < n:
        m = 0
        while m < K and pos + m < n and accept[pos + m]:
            m += 1
        Ks.append(K)
        pos += min(m + 1, n - pos)
    return Ks


def run_entropy_stop(accept, sig, direction, thr, Kmax):
    """Early-stop chain.  Propose token s (pay drafter); if signal says 'uncertain'
    (direction*sig[pos+s] < direction*thr i.e. crossed the stop side) OR s+1==Kmax,
    stop with K_drafted=s+1.  Returns per-invoke K_drafted list."""
    n = len(accept)
    pos = 0
    Ks = []
    while pos < n:
        s = 0
        while True:
            j = pos + s
            stop_here = (j >= n - 1) or (s + 1 >= Kmax)
            if not stop_here:
                # uncertain if the (oriented) signal has crossed below threshold
                if direction * sig[j] < direction * thr:
                    stop_here = True
            if stop_here:
                K = s + 1
                break
            s += 1
        Ks.append(K)
        m = 0
        while m < K and pos + m < n and accept[pos + m]:
            m += 1
        pos += min(m + 1, n - pos)
    return Ks


def score_Ks(accept, Ks, curve, step_ms, mode, Kmax):
    """given per-invoke K_drafted, accumulate tokens + time for one sequence."""
    n = len(accept)
    pos = 0
    tot_ms = 0.0
    for K in Ks:
        tot_ms += drafter_ms(K, step_ms, mode, Kmax) + verify_ms(K, curve)
        m = 0
        while m < K and pos + m < n and accept[pos + m]:
            m += 1
        pos += min(m + 1, n - pos)
    return n, tot_ms, Ks


def tps_of(seqs, runner, curve, step_ms, mode, Kmax):
    tok = ms = 0.0
    allK = []
    for seq in seqs:
        Ks = runner(seq)
        t, m, _ = score_Ks(seq["accept"], Ks, curve, step_ms, mode, Kmax)
        tok += t
        ms += m
        allK += Ks
    return {
        "tps": tok / (ms / 1000.0) if ms > 0 else 0.0,
        "tokens": tok, "ms": ms,
        "mean_K": st.mean(allK) if allK else 0.0,
        "sd_K": st.pstdev(allK) if len(allK) > 1 else 0.0,
        "invokes": len(allK),
    }


def oracle_linear(seqs, curve, step_ms, mode, Kmax):
    """clairvoyant per-invoke optimal K (DP) on the linear chain."""
    tok = ms = 0.0
    for seq in seqs:
        a = seq["accept"]
        n = len(a)
        run = remaining_run(a)                      # uncapped
        INF = float("inf")
        dp = [INF] * (n + 1)
        dp[n] = 0.0
        for pos in range(n - 1, -1, -1):
            r = run[pos]
            best = INF
            for K in range(1, Kmax + 1):
                consumed = min(min(r, K) + 1, n - pos)
                c = drafter_ms(K, step_ms, mode, Kmax) + verify_ms(K, curve) + dp[pos + consumed]
                if c < best:
                    best = c
            dp[pos] = best
        tok += n
        ms += dp[0]
    return tok / (ms / 1000.0) if ms > 0 else 0.0


# --------------------------------------------------------------------------- #
# bake-off
# --------------------------------------------------------------------------- #
def thresholds_for(seqs, signal, n=25):
    vals = sorted(v for seq in seqs for v in seq[signal])
    if not vals:
        return []
    qs = [vals[min(len(vals) - 1, int(len(vals) * i / (n + 1)))] for i in range(1, n + 1)]
    return sorted(set(qs))


def evaluate(seqs, curve, step_ms, mode, Kmax, K_static, label):
    res = {"label": label, "drafter_step_ms": step_ms, "drafter_mode": mode,
           "Kmax": Kmax, "K_static": K_static}
    # static curve
    static_rows = {}
    bK, bT = 1, -1.0
    for K in range(1, Kmax + 1):
        s = tps_of(seqs, lambda seq, K=K: run_static(seq["accept"], K),
                   curve, step_ms, mode, Kmax)
        static_rows[K] = s["tps"]
        if s["tps"] > bT:
            bK, bT = K, s["tps"]
    res["static_curve"] = static_rows
    res["best_static_K"], res["best_static_tps"] = bK, bT
    s_stat = tps_of(seqs, lambda seq: run_static(seq["accept"], K_static),
                    curve, step_ms, mode, Kmax)
    res["static_K_static"] = {"K": K_static, **s_stat}
    res["e_accept"] = (s_stat["tokens"] / s_stat["invokes"]) - 1.0  # emitted/invoke - bonus

    # oracle
    res["oracle_tps"] = oracle_linear(seqs, curve, step_ms, mode, Kmax)
    res["oracle_gain_vs_static_pct"] = 100 * (res["oracle_tps"] / s_stat["tps"] - 1)

    # entropy / proxy early-stop controllers
    res["controllers"] = {}
    best_by_signal = {}
    for sgl in SIGNALS:
        if not all(sgl in seq for seq in seqs):
            continue
        d = SIGNAL_DIRECTION[sgl]
        best = None
        rows = []
        for thr in thresholds_for(seqs, sgl):
            s = tps_of(seqs, lambda seq, sgl=sgl, d=d, thr=thr:
                       run_entropy_stop(seq["accept"], seq[sgl], d, thr, Kmax),
                       curve, step_ms, mode, Kmax)
            rows.append({"thr": round(thr, 5), "tps": s["tps"],
                         "mean_K": round(s["mean_K"], 3), "sd_K": round(s["sd_K"], 3)})
            if best is None or s["tps"] > best["tps"]:
                best = {"thr": round(thr, 5), **s}
        res["controllers"][sgl] = {"best": best, "sweep": rows}
        best_by_signal[sgl] = best["tps"]
    res["best_signal_tps"] = best_by_signal

    # acceptance-history controllers (#51), re-scored on linear trace
    ah = {}
    best_ah = None
    for K0 in (bK, 3, 5):
        for inc in (1, 2):
            for dec in (0.5, 0.6, 0.7):
                pol = AIMD(1, Kmax, inc, dec, K0)
                s = tps_of(seqs, lambda seq, pol=pol: _accepthist_Ks(seq["accept"], pol, Kmax),
                           curve, step_ms, mode, Kmax)
                if best_ah is None or s["tps"] > best_ah[1]:
                    best_ah = (f"aimd_K0{K0}_inc{inc}_dec{dec}", s["tps"], s)
    for window in (2, 4, 8):
        for a in (0.5, 1.0, 1.5):
            for b in (1, 2, 3):
                pol = WindowMeanLinear(window, a, b, 1, Kmax, bK)
                s = tps_of(seqs, lambda seq, pol=pol: _accepthist_Ks(seq["accept"], pol, Kmax),
                           curve, step_ms, mode, Kmax)
                if best_ah is None or s["tps"] > best_ah[1]:
                    best_ah = (f"winlin_w{window}_a{a}_b{b}", s["tps"], s)
    ah["best"] = {"tag": best_ah[0], "tps": best_ah[1],
                  "mean_K": round(best_ah[2]["mean_K"], 3),
                  "sd_K": round(best_ah[2]["sd_K"], 3)}
    res["accepthist_controller"] = ah

    # headline: best realizable entropy controller vs static + frac of oracle
    best_ent_sig = max(best_by_signal, key=best_by_signal.get) if best_by_signal else None
    best_ent_tps = best_by_signal.get(best_ent_sig, float("nan")) if best_ent_sig else float("nan")
    res["best_entropy_signal"] = best_ent_sig
    res["best_entropy_tps"] = best_ent_tps
    denom = res["oracle_tps"] - s_stat["tps"]
    res["entropy_gain_vs_static_pct"] = 100 * (best_ent_tps / s_stat["tps"] - 1)
    res["entropy_frac_of_oracle"] = (best_ent_tps - s_stat["tps"]) / denom if denom > 0 else float("nan")
    res["accepthist_gain_vs_static_pct"] = 100 * (best_ah[1] / s_stat["tps"] - 1)
    res["accepthist_frac_of_oracle"] = (best_ah[1] - s_stat["tps"]) / denom if denom > 0 else float("nan")
    return res


def _accepthist_Ks(accept, policy, Kmax):
    """run a #51 history policy (callable accepted_lens->K) on the linear chain."""
    if hasattr(policy, "reset"):
        policy.reset()
    n = len(accept)
    pos = 0
    Ks, lens = [], []
    while pos < n:
        K = max(1, min(Kmax, int(policy(lens))))
        Ks.append(K)
        m = 0
        while m < K and pos + m < n and accept[pos + m]:
            m += 1
        lens.append(m)
        pos += min(m + 1, n - pos)
    return Ks


# --------------------------------------------------------------------------- #
def thin_to_eaccept(seqs, target_E, Kmax, seed=0):
    """i.i.d.-thin accept 1->0 so static-Kmax E[emitted/invoke]-1 matches target_E.
    CONSERVATIVE for the entropy predictor: random flips ERODE the entropy<->accept
    relationship, so a win that survives thinning is a lower bound."""
    import random
    rng = random.Random(seed)

    def E_at(ss):
        tok = inv = 0
        for seq in ss:
            Ks = run_static(seq["accept"], Kmax)
            t, _, _ = score_Ks(seq["accept"], Ks, LatencyCurve({1: 1, 2: 1}), 0, "fixed", Kmax)
            tok += t
            inv += len(Ks)
        return tok / inv - 1.0
    lo, hi = 0.0, 0.95
    for _ in range(40):
        d = (lo + hi) / 2
        thinned = [dict(seq, accept=[0 if (x and rng.random() < d) else x
                                     for x in seq["accept"]]) for seq in seqs]
        if E_at(thinned) > target_E:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return [dict(seq, accept=[0 if (x and rng.random() < d) else x
                              for x in seq["accept"]]) for seq in seqs], d


def print_report(pred, results):
    print("\n" + "=" * 80)
    print("[entropy] ===== Task 2: predictor (drafter-entropy vs acceptance-history) =====")
    print(f"  positions={pred['n_positions']}  base top-1 accept={pred['base_accept_top1']:.4f}")
    print(f"  acceptance-history window->next-run r (best): {pred['accepthist_best_r']:.4f}"
          f"   [#51 baseline ~0.32]")
    print(f"  {'signal':10s} {'r(accept)':>10s} {'r(run)':>9s} {'AUC':>7s}")
    for s, v in pred["signals"].items():
        print(f"  {s:10s} {v['pearson_accept']:>10.4f} {v['pearson_run']:>9.4f} "
              f"{v['auc_accept']:>7.4f}")
    for res in results:
        print("\n" + "-" * 80)
        print(f"[entropy] CURVE/REGIME = {res['label']}  "
              f"(drafter={res['drafter_mode']} {res['drafter_step_ms']}ms/step, Kmax={res['Kmax']})")
        s = res["static_K_static"]
        print(f"  static K={s['K']:<2}     : {s['tps']:7.1f} TPS  (e_accept={res['e_accept']:.2f}, "
              f"mean_K={s['mean_K']:.2f})")
        print(f"  best static K*={res['best_static_K']:<2}: {res['best_static_tps']:7.1f} TPS")
        print(f"  ORACLE        : {res['oracle_tps']:7.1f} TPS  "
              f"(+{res['oracle_gain_vs_static_pct']:.1f}% vs static K={s['K']})")
        for sgl, c in res["controllers"].items():
            b = c["best"]
            print(f"  stop[{sgl:9s}]: {b['tps']:7.1f} TPS  thr={b['thr']:.4f} "
                  f"mean_K={b['mean_K']:.2f} sd_K={b['sd_K']:.2f}")
        ah = res["accepthist_controller"]["best"]
        print(f"  accepthist    : {ah['tps']:7.1f} TPS  mean_K={ah['mean_K']:.2f} [{ah['tag']}]")
        print(f"  --> best entropy ({res['best_entropy_signal']}): {res['best_entropy_tps']:7.1f} TPS  "
              f"({res['entropy_gain_vs_static_pct']:+.2f}% vs static)  "
              f"recovers {100*res['entropy_frac_of_oracle']:.1f}% of oracle")
        print(f"      (accepthist recovers {100*res['accepthist_frac_of_oracle']:.1f}% of oracle)")


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace",
                    default="research/eagle3_drafter/eval_traces/entropy_trace_debug1k2ep.jsonl")
    ap.add_argument("--Kmax", type=int, default=7, help="linear MTP draft depth (deployed=7)")
    ap.add_argument("--K-static", "--K_static", type=int, default=7,
                    help="deployed static operating point to beat")
    ap.add_argument("--drafter-step-ms", "--drafter_step_ms", type=float, default=0.2,
                    help="per-token drafter cost (linear-mode); #51 total 1.4ms/7~0.2")
    ap.add_argument("--curve",
                    default="research/spec_cost_model/results_pr51_splitkv_longctx.json|graph|ctx512")
    ap.add_argument("--anchor-eaccept", "--anchor_eaccept", type=float, nargs="*", default=None,
                    help="also run conservative i.i.d.-thinned variants at these E[accept]")
    ap.add_argument("--output", default="research/entropy_dynamic_k/entropy_sim.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT",
                                                              "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="entropy-dynamic-k")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    seqs, meta, have_conf = load_entropy_trace(args.trace)
    print(f"[entropy] trace={args.trace}  seqs={len(seqs)}  "
          f"positions={sum(len(s['accept']) for s in seqs)}  conf={have_conf}")
    if not have_conf:
        raise SystemExit("[entropy] trace has no confidence fields; regenerate with "
                         "`eval_eagle3.py --confidence`.")

    pred = predictor_report(seqs)

    path, key = args.curve.split("|", 1)
    curve = LatencyCurve(load_latency_curve(path, key))

    results = []
    for mode in ("linear", "fixed"):
        r = evaluate(seqs, curve, args.drafter_step_ms, mode, args.Kmax, args.K_static,
                     f"ctx512-splitkv drafter={mode}")
        r["curve"], r["curve_key"], r["thinned"] = path, key, None
        results.append(r)

    if args.anchor_eaccept:
        for E in args.anchor_eaccept:
            thinned, d = thin_to_eaccept(seqs, E, args.Kmax)
            r = evaluate(thinned, curve, args.drafter_step_ms, "linear", args.Kmax,
                         args.K_static, f"ctx512-splitkv drafter=linear THINNED->E={E}")
            r["curve"], r["curve_key"], r["thinned"] = path, key, {"target_E": E, "drop": d}
            results.append(r)

    print_report(pred, results)

    payload = {"config": vars(args), "trace_meta": meta, "predictor": pred,
               "results": results}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[entropy] wrote {args.output}")

    if not args.no_wandb and args.wandb_name:
        try:
            log_wandb(args, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[entropy] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[entropy] DONE")


def log_wandb(args, payload):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="analysis", config=payload["config"])
    pred = payload["predictor"]
    summary = {
        "n_positions": pred["n_positions"],
        "base_accept_top1": pred["base_accept_top1"],
        "accepthist_window_run_r": pred["accepthist_best_r"],
    }
    for s, v in pred["signals"].items():
        summary[f"{s}_pearson_accept"] = v["pearson_accept"]
        summary[f"{s}_pearson_run"] = v["pearson_run"]
        summary[f"{s}_auc_accept"] = v["auc_accept"]
    for r in payload["results"]:
        tag = r["label"].replace(" ", "_").replace("=", "").replace(">", "")
        summary[f"{tag}_static_tps"] = r["static_K_static"]["tps"]
        summary[f"{tag}_oracle_tps"] = r["oracle_tps"]
        summary[f"{tag}_oracle_gain_pct"] = r["oracle_gain_vs_static_pct"]
        summary[f"{tag}_best_entropy_tps"] = r["best_entropy_tps"]
        summary[f"{tag}_entropy_gain_pct"] = r["entropy_gain_vs_static_pct"]
        summary[f"{tag}_entropy_frac_of_oracle"] = r["entropy_frac_of_oracle"]
        summary[f"{tag}_accepthist_frac_of_oracle"] = r["accepthist_frac_of_oracle"]
    run.summary.update(summary)
    run.finish()


if __name__ == "__main__":
    main()
