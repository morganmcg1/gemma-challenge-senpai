#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""accepthist: per-step dynamic draft-length (K) controller, designed + simulated
against the measured per-position acceptance trace and the measured int4 verify
cost curve (PR #51).

WHY (PR #51).  The static operating point is K*=11 (M=45, width-4 tree): the K
that maximizes steady-state TPS = E[tokens/invoke] / ((drafter_ms + verify(M))/1000)
under the measured cost curve.  But the per-invoke accepted length is NOT
stationary -- it is sharply BIMODAL (many invokes miss on token 0; many saturate
the full K) with positive short-lag autocorrelation.  A clairvoyant controller
that picks K per-invoke from the upcoming run length beats static K*=11 by a wide
margin (the ORACLE ceiling, computed here by DP).  The question this module
answers: how much of that ceiling can a REALIZABLE controller -- one that sees
only PAST accepted lengths -- capture?

DELIVERABLES
  Task 2 (instrument): per-position accept autocorrelation; per-invoke
    accepted_len distribution + autocorrelation; acceptance vs sequence position;
    rolling-window predictiveness corr(mean(prev w), next).
  Task 3 (controller): realizable controllers (AIMD, window-mean->K linear/lookup)
    simulated on the trace, scored on the MEASURED cost curve, vs static K=11,
    static argmax-K*, and the clairvoyant oracle.  Sweep window/bounds.  Run on
    BOTH the pre-#43 and post-#43 (split-KV) curves; report which curve, if any,
    lets dynamic-K open a gap.

STEADY-STATE TPS IDENTITY.  For a FIXED per-position accept sequence acc[], every
policy that runs to the end emits exactly len(acc) tokens (each position leaves as
an accept or as the verifier's bonus).  So maximizing TPS == MINIMIZING total
verify+drafter time = sum over invokes of (drafter_ms + curve.at(K_t*W + 1)).
This makes the oracle a clean DP and makes all policies directly comparable.

SERVING INTEGRATION (the `--accepthist --accepthist-window N` hook, default off).
The realizable controllers here are pure functions of the PAST accepted-length
history -- exactly the state a served decode loop already has.  To wire into the
vLLM spec-decode path one would, each step, before the drafter proposes:
    K_t = controller(recent_accepted_lens)          # this module's policy objects
    proposer.num_speculative_tokens = K_t           # per-step draft depth
    # (tree builder uses M_t = W*K_t + 1 verify positions)
and after verify, append the observed accepted_len to the history ring buffer.
`--accepthist-window N` is the ring-buffer length feeding the window-mean policy;
default-off means K stays at the static `--K-static`.  Because the verifier still
accepts only greedy-correct tokens, changing K per step is GREEDY-EXACT (token
identity preserved -> PPL unchanged), hence leaderboard-legal (official gate =
PPL+completion+modalities, not token-identity; kanna #38).  This module SIMULATES
that loop on the measured acceptance trace + measured cost curve; the served hook
itself is left as the design above because the simulation (below) shows the
acceptance-history signal yields ~0 TPS over a correctly-tuned static K.

CPU-ONLY.  Inputs: the hit-rank trace + a measured cost curve JSON.  No GPU/vLLM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

# reuse the committed cost-curve + trace loaders / accept semantics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_acceptance_model import (  # noqa: E402
    LatencyCurve,
    load_latency_curve,
    load_trace,
)


# --------------------------------------------------------------------------- #
# accept protocol (per-invoke), generalised to a time-varying K policy
# --------------------------------------------------------------------------- #
def acc_indicator(hit_rank: list[int], W: int) -> list[int]:
    return [1 if 1 <= r <= W else 0 for r in hit_rank]


def run_policy(acc: list[int], W: int, policy) -> dict:
    """Traverse one sequence under `policy` (a callable history->K).  Returns the
    per-invoke accepted_len + chosen-K traces and the total verify cost pieces.
    `policy` is reset per sequence via policy.reset() if present."""
    if hasattr(policy, "reset"):
        policy.reset()
    n = len(acc)
    pos = invokes = 0
    accepted_lens: list[int] = []
    Ks: list[int] = []
    while pos < n:
        K = int(policy(accepted_lens))
        m = 0
        while m < K and pos + m < n and acc[pos + m]:
            m += 1
        emitted = min(m + 1, n - pos)        # accepted + 1 bonus, capped at end
        accepted_lens.append(m)
        Ks.append(K)
        pos += emitted
        invokes += 1
    return {"accepted_lens": accepted_lens, "Ks": Ks, "tokens": n, "invokes": invokes}


def score_policy(acc_seqs, W, policy, curve, drafter_ms):
    """Steady-state TPS of `policy` over all sequences on the measured curve."""
    tot_tok = 0
    tot_ms = 0.0
    all_lens, all_Ks = [], []
    for acc in acc_seqs:
        r = run_policy(acc, W, policy)
        tot_tok += r["tokens"]
        for K in r["Ks"]:
            tot_ms += drafter_ms + curve.at(K * W + 1)
        all_lens += r["accepted_lens"]
        all_Ks += r["Ks"]
    tps = tot_tok / (tot_ms / 1000.0) if tot_ms > 0 else 0.0
    return {
        "tps": tps, "tokens": tot_tok, "verify_ms": tot_ms,
        "mean_K": st.mean(all_Ks) if all_Ks else 0.0,
        "sd_K": st.pstdev(all_Ks) if len(all_Ks) > 1 else 0.0,
        "mean_accepted_len": st.mean(all_lens) if all_lens else 0.0,
        "invokes": len(all_Ks),
    }


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
class StaticK:
    def __init__(self, K: int):
        self.K = K

    def __call__(self, hist):
        return self.K


class AIMD:
    """Additive-increase / multiplicative-decrease on draft depth.

    After an invoke that SATURATED (accepted_len == current K -> the run was at
    least K long, we under-drafted): K += inc.  After an invoke that MISSED early
    (accepted_len < K): K = max(Kmin, round(K * dec)).  Bounded [Kmin, Kmax].
    Mirrors AdaEDL/AIMD congestion control: ride long runs up, back off on misses.
    """

    def __init__(self, Kmin, Kmax, inc, dec, K0):
        self.Kmin, self.Kmax, self.inc, self.dec, self.K0 = Kmin, Kmax, inc, dec, K0
        self.K = K0

    def reset(self):
        self.K = self.K0

    def __call__(self, hist):
        if hist:
            last = hist[-1]
            if last >= self.K:
                self.K = min(self.Kmax, self.K + self.inc)
            else:
                self.K = max(self.Kmin, int(round(self.K * self.dec)))
        return self.K


class WindowMeanLinear:
    """K_next = clip(round(a * mean(prev `window` accepted_len) + b))."""

    def __init__(self, window, a, b, Kmin, Kmax, K0):
        self.window, self.a, self.b = window, a, b
        self.Kmin, self.Kmax, self.K0 = Kmin, Kmax, K0

    def reset(self):
        pass

    def __call__(self, hist):
        if not hist:
            return self.K0
        w = hist[-self.window:]
        K = int(round(self.a * (sum(w) / len(w)) + self.b))
        return max(self.Kmin, min(self.Kmax, K))


class WindowMeanLUT:
    """Cliff-aware: bucket the window-mean predictor, apply a per-bucket optimal K
    learned on a training split (passed in)."""

    def __init__(self, window, lut, Kmin, Kmax, K0):
        self.window, self.lut = window, lut          # lut: bucket(int) -> K
        self.Kmin, self.Kmax, self.K0 = Kmin, Kmax, K0

    def reset(self):
        pass

    def __call__(self, hist):
        if not hist:
            return self.K0
        w = hist[-self.window:]
        b = int(round(sum(w) / len(w)))
        return self.lut.get(b, self.K0)


# --------------------------------------------------------------------------- #
# clairvoyant oracle (DP) + best static
# --------------------------------------------------------------------------- #
def oracle_dp(acc, W, curve, drafter_ms, Kmin, Kmax):
    n = len(acc)
    run = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        run[i] = run[i + 1] + 1 if acc[i] else 0
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[n] = 0.0
    choice = [Kmin] * (n + 1)
    for pos in range(n - 1, -1, -1):
        r = run[pos]
        best, bestK = INF, Kmin
        for K in range(Kmin, Kmax + 1):
            consumed = min(r, K) + 1
            nxt = min(pos + consumed, n)
            c = drafter_ms + curve.at(K * W + 1) + dp[nxt]
            if c < best:
                best, bestK = c, K
        dp[pos], choice[pos] = best, bestK
    return n, dp[0]


def best_static(acc_seqs, W, curve, drafter_ms, Kmin, Kmax):
    rows = {}
    bK, bT = Kmin, -1.0
    for K in range(Kmin, Kmax + 1):
        s = score_policy(acc_seqs, W, StaticK(K), curve, drafter_ms)
        rows[K] = s["tps"]
        if s["tps"] > bT:
            bK, bT = K, s["tps"]
    return bK, bT, rows


# --------------------------------------------------------------------------- #
# stats helpers (Task 2 instrumentation)
# --------------------------------------------------------------------------- #
def autocorr(seq, lag):
    n = len(seq)
    if n <= lag:
        return float("nan")
    m = sum(seq) / n
    num = sum((seq[i] - m) * (seq[i + lag] - m) for i in range(n - lag))
    den = sum((x - m) ** 2 for x in seq)
    return num / den if den > 0 else float("nan")


def pooled_autocorr(seqs, lag):
    vals = [autocorr(s, lag) for s in seqs if len(s) > lag + 5]
    vals = [v for v in vals if not math.isnan(v)]
    return (sum(vals) / len(vals)) if vals else float("nan"), len(vals)


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx * dy > 0 else float("nan")


def instrument(acc_seqs, W, K_static, curve, drafter_ms):
    out = {}
    all_acc = [a for s in acc_seqs for a in s]
    out["base_accept_rate"] = sum(all_acc) / len(all_acc)
    out["n_positions"] = len(all_acc)
    out["n_sequences"] = len(acc_seqs)
    out["pos_autocorr"] = {f"lag{lag}": pooled_autocorr(acc_seqs, lag)[0]
                           for lag in (1, 2, 3, 5, 10)}
    inv_seqs = [run_policy(s, W, StaticK(K_static))["accepted_lens"] for s in acc_seqs]
    out["invoke_autocorr"] = {f"lag{lag}": pooled_autocorr(inv_seqs, lag)[0]
                              for lag in (1, 2, 3)}
    flat = [x for s in inv_seqs for x in s]
    out["accepted_len"] = {
        "mean": st.mean(flat), "sd": st.pstdev(flat),
        "min": min(flat), "max": max(flat), "n_invokes": len(flat),
        "hist": {str(k): flat.count(k) / len(flat) for k in range(0, K_static + 1)
                 if flat.count(k)},
        "frac_miss0": flat.count(0) / len(flat),
        "frac_saturate": flat.count(K_static) / len(flat),
    }
    # acceptance vs normalized sequence position (deciles)
    dec = [[] for _ in range(10)]
    for s in acc_seqs:
        n = len(s)
        for j, a in enumerate(s):
            dec[min(9, int(10 * j / n))].append(a)
    out["accept_by_decile"] = [round(sum(d) / len(d), 4) if d else None for d in dec]
    # rolling-window predictiveness
    out["window_predictiveness"] = {}
    for w in (2, 4, 8, 16):
        xs, ys = [], []
        for s in inv_seqs:
            for i in range(w, len(s)):
                xs.append(sum(s[i - w:i]) / w)
                ys.append(s[i])
        out["window_predictiveness"][f"w{w}"] = {"r": pearson(xs, ys), "n": len(xs)}
    return out, inv_seqs


# --------------------------------------------------------------------------- #
# learn a window-mean LUT on the trace (cliff-aware conditional-optimal K)
# --------------------------------------------------------------------------- #
def learn_lut(acc_seqs, W, window, curve, drafter_ms, Kmin, Kmax):
    """For each window-mean bucket b, choose the K minimizing the per-invoke
    verify+drafter cost-per-token AVERAGED over invokes whose predictor==b.
    Greedy/local (each invoke scored independently given its true upcoming run)."""
    # collect (bucket, run_length) per invoke under a neutral static traversal
    samples = {}  # bucket -> list of upcoming run lengths
    for acc in acc_seqs:
        lens = run_policy(acc, W, StaticK(Kmax))["accepted_lens"]  # max-K reveals full runs
        # re-derive per-invoke upcoming run with positions:
        n = len(acc)
        run = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            run[i] = run[i + 1] + 1 if acc[i] else 0
        pos = 0
        hist = []
        while pos < n:
            b = int(round(sum(hist[-window:]) / len(hist[-window:]))) if hist else None
            r = run[pos]
            if b is not None:
                samples.setdefault(b, []).append(r)
            # advance using the eventual accepted_len at max K to build history
            m = min(r, Kmax)
            hist.append(m)
            pos += min(m + 1, n - pos)
    lut = {}
    for b, runs in samples.items():
        bestK, bestcost = Kmin, float("inf")
        for K in range(Kmin, Kmax + 1):
            # expected cost-per-token if we use K on invokes with these run lengths
            tok = sum(min(r, K) + 1 for r in runs)
            cost = len(runs) * (drafter_ms + curve.at(K * W + 1))
            cpt = cost / max(1, tok)
            if cpt < bestcost:
                bestcost, bestK = cpt, K
        lut[b] = bestK
    return lut


# --------------------------------------------------------------------------- #
# acceptance-level anchoring (trace is optimistic vs real served e_accept)
# --------------------------------------------------------------------------- #
def thin_to_eaccept(acc_seqs, W, target_E, curve, drafter_ms, K_ref=11, seed=0):
    """i.i.d.-thin accept=1 -> 0 with prob d so static-K_ref E matches target_E.
    NOTE: i.i.d. thinning ERODES autocorrelation, so it is a CONSERVATIVE (lower-
    bound) test of whether the dynamic-K relative win survives at lower acceptance.
    """
    import random
    rng = random.Random(seed)

    def E_at(seqs):
        tok = inv = 0
        for s in seqs:
            r = run_policy(s, W, StaticK(K_ref))
            tok += r["tokens"]
            inv += r["invokes"]
        return tok / inv
    lo, hi = 0.0, 0.95
    for _ in range(40):
        d = (lo + hi) / 2
        thinned = [[0 if (a and rng.random() < d) else a for a in s] for s in acc_seqs]
        E = E_at(thinned)
        if E > target_E:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return [[0 if (a and rng.random() < d) else a for a in s] for s in acc_seqs], d


# --------------------------------------------------------------------------- #
def evaluate_curve(acc_seqs, W, curve, drafter_ms, Kmin, Kmax, K_static,
                   windows, label):
    """Full controller bake-off on one cost curve.  Returns a results dict."""
    res = {"label": label, "drafter_ms": drafter_ms, "W": W,
           "Kmin": Kmin, "Kmax": Kmax}
    bK, bT, static_rows = best_static(acc_seqs, W, curve, drafter_ms, Kmin, Kmax)
    res["static_curve"] = static_rows
    res["best_static_K"] = bK
    res["best_static_tps"] = bT
    s11 = score_policy(acc_seqs, W, StaticK(K_static), curve, drafter_ms)
    res["static_K_static"] = {"K": K_static, **s11}
    # oracle
    tok = ms = 0.0
    for acc in acc_seqs:
        t, m = oracle_dp(acc, W, curve, drafter_ms, Kmin, Kmax)
        tok += t
        ms += m
    res["oracle_tps"] = tok / (ms / 1000.0)
    res["oracle_gain_vs_static11_pct"] = 100 * (res["oracle_tps"] / s11["tps"] - 1)
    res["oracle_gain_vs_beststatic_pct"] = 100 * (res["oracle_tps"] / bT - 1)

    # realizable controllers
    controllers = {}

    # AIMD sweep
    best_aimd = None
    for K0 in (bK, 6, 8):
        for inc in (1, 2, 3):
            for dec in (0.5, 0.6, 0.7):
                pol = AIMD(Kmin, Kmax, inc, dec, K0)
                s = score_policy(acc_seqs, W, pol, curve, drafter_ms)
                tag = f"aimd_K0{K0}_inc{inc}_dec{dec}"
                controllers[tag] = s["tps"]
                if best_aimd is None or s["tps"] > best_aimd[1]:
                    best_aimd = (tag, s["tps"], {"K0": K0, "inc": inc, "dec": dec}, s)
    res["best_aimd"] = {"tag": best_aimd[0], "tps": best_aimd[1],
                        "params": best_aimd[2], "mean_K": best_aimd[3]["mean_K"],
                        "sd_K": best_aimd[3]["sd_K"]}

    # WindowMeanLinear sweep (a,b) per window
    best_lin = None
    for window in windows:
        for a in (0.5, 1.0, 1.5, 2.0):
            for b in (1, 2, 3, 4):
                pol = WindowMeanLinear(window, a, b, Kmin, Kmax, bK)
                s = score_policy(acc_seqs, W, pol, curve, drafter_ms)
                tag = f"winlin_w{window}_a{a}_b{b}"
                controllers[tag] = s["tps"]
                if best_lin is None or s["tps"] > best_lin[1]:
                    best_lin = (tag, s["tps"], {"window": window, "a": a, "b": b}, s)
    res["best_winlin"] = {"tag": best_lin[0], "tps": best_lin[1],
                          "params": best_lin[2], "mean_K": best_lin[3]["mean_K"],
                          "sd_K": best_lin[3]["sd_K"]}

    # WindowMeanLUT (cliff-aware, learned on same trace -> optimistic/in-sample)
    best_lut = None
    for window in windows:
        lut = learn_lut(acc_seqs, W, window, curve, drafter_ms, Kmin, Kmax)
        pol = WindowMeanLUT(window, lut, Kmin, Kmax, bK)
        s = score_policy(acc_seqs, W, pol, curve, drafter_ms)
        tag = f"winlut_w{window}"
        controllers[tag] = s["tps"]
        if best_lut is None or s["tps"] > best_lut[1]:
            best_lut = (tag, s["tps"], {"window": window, "lut": lut}, s)
    res["best_winlut"] = {"tag": best_lut[0], "tps": best_lut[1],
                          "params": {"window": best_lut[2]["window"]},
                          "lut": best_lut[2]["lut"],
                          "mean_K": best_lut[3]["mean_K"],
                          "sd_K": best_lut[3]["sd_K"]}

    # headline: best realizable vs static-K_static
    best_real_tps = max(res["best_aimd"]["tps"], res["best_winlin"]["tps"],
                        res["best_winlut"]["tps"])
    res["best_realizable_tps"] = best_real_tps
    res["best_realizable_gain_vs_static11_pct"] = 100 * (best_real_tps / s11["tps"] - 1)
    res["best_realizable_gain_vs_beststatic_pct"] = 100 * (best_real_tps / bT - 1)
    res["realizable_fraction_of_oracle"] = (
        (best_real_tps - bT) / (res["oracle_tps"] - bT)
        if res["oracle_tps"] > bT else float("nan"))
    res["all_controllers"] = controllers
    return res


def print_curve_report(res):
    print(f"\n{'='*78}\n[accepthist] CURVE = {res['label']}  "
          f"(W={res['W']}, drafter={res['drafter_ms']}ms, K∈[{res['Kmin']},{res['Kmax']}])")
    s11 = res["static_K_static"]
    print(f"  static K={s11['K']:>2}        : {s11['tps']:7.1f} TPS  "
          f"(mean_accepted_len={s11['mean_accepted_len']:.2f})")
    print(f"  best static  K*={res['best_static_K']:>2}: {res['best_static_tps']:7.1f} TPS")
    print(f"  ORACLE (clairvoyant): {res['oracle_tps']:7.1f} TPS  "
          f"(+{res['oracle_gain_vs_static11_pct']:.1f}% vs K=11, "
          f"+{res['oracle_gain_vs_beststatic_pct']:.1f}% vs best-static)")
    for name in ("best_aimd", "best_winlin", "best_winlut"):
        r = res[name]
        print(f"  {name:14s}: {r['tps']:7.1f} TPS  mean_K={r['mean_K']:.2f} "
              f"sd_K={r['sd_K']:.2f}  [{r.get('tag','')}]")
    print(f"  --> best realizable : {res['best_realizable_tps']:7.1f} TPS  "
          f"(+{res['best_realizable_gain_vs_static11_pct']:+.2f}% vs K=11, "
          f"+{res['best_realizable_gain_vs_beststatic_pct']:+.2f}% vs best-static)")
    print(f"      captures {100*res['realizable_fraction_of_oracle']:.1f}% of the "
          f"oracle ceiling")


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace",
                    default="research/eagle3_drafter/eval_traces/topk_trace_debug1k2ep.jsonl")
    ap.add_argument("--W", type=int, default=4, help="tree width")
    ap.add_argument("--drafter-ms", "--drafter_ms", type=float, default=1.4)
    ap.add_argument("--K-static", "--K_static", type=int, default=11,
                    help="the static operating point to beat (PR baseline K*=11)")
    ap.add_argument("--Kmin", type=int, default=1)
    ap.add_argument("--Kmax", type=int, default=16)
    ap.add_argument("--windows", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--curves", nargs="+",
                    default=["pre43=research/spec_cost_model/results_msweep.json|graph|ctx256"],
                    help="label=path|costkey entries; multiple allowed")
    ap.add_argument("--anchor-eaccept", "--anchor_eaccept", type=float, default=None,
                    help="also run a conservative i.i.d.-thinned variant at this E")
    ap.add_argument("--output", default="research/accepthist/accepthist_sim.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT",
                                                              "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="accepthist-dynamic-k")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap


def parse_curve(spec):
    label, rest = spec.split("=", 1)
    parts = rest.split("|")
    path = parts[0]
    key = "|".join(parts[1:]) if len(parts) > 1 else "graph|ctx256"
    return label, path, key


def main():
    args = build_argparser().parse_args()
    traces, meta = load_trace(args.trace)
    acc_seqs = [acc_indicator(tr["hit_rank"], args.W) for tr in traces if tr.get("hit_rank")]
    print(f"[accepthist] trace={args.trace}  seqs={len(acc_seqs)}  "
          f"positions={sum(len(s) for s in acc_seqs)}  "
          f"top-{args.W}_accept={meta.get('top_acc',{}).get(str(args.W),'?')}")

    instr, _ = instrument(acc_seqs, args.W, args.K_static, None, args.drafter_ms)
    print("\n[accepthist] ===== Task 2: acceptance instrumentation =====")
    print(f"  accept(top-{args.W}) = {instr['base_accept_rate']:.4f}  "
          f"positions={instr['n_positions']}  seqs={instr['n_sequences']}")
    print("  position-level accept autocorr:", {k: round(v, 4)
          for k, v in instr["pos_autocorr"].items()})
    print("  per-invoke accepted_len autocorr:", {k: round(v, 4)
          for k, v in instr["invoke_autocorr"].items()})
    al = instr["accepted_len"]
    print(f"  accepted_len: mean={al['mean']:.3f} sd={al['sd']:.3f} "
          f"frac_miss0={al['frac_miss0']:.3f} frac_saturate(K={args.K_static})="
          f"{al['frac_saturate']:.3f}")
    print("  accept by decile:", instr["accept_by_decile"])
    print("  window predictiveness:", {k: round(v["r"], 4)
          for k, v in instr["window_predictiveness"].items()})

    curve_results = []
    for spec in args.curves:
        label, path, key = parse_curve(spec)
        if not os.path.exists(path):
            print(f"[accepthist] SKIP curve {label}: {path} not found", flush=True)
            continue
        curve = LatencyCurve(load_latency_curve(path, key))
        r = evaluate_curve(acc_seqs, args.W, curve, args.drafter_ms, args.Kmin,
                           args.Kmax, args.K_static, args.windows,
                           f"{label} ({key})")
        r["curve_path"], r["curve_key"], r["curve_label"] = path, key, label
        print_curve_report(r)
        curve_results.append(r)

        if args.anchor_eaccept:
            thinned, d = thin_to_eaccept(acc_seqs, args.W, args.anchor_eaccept,
                                         curve, args.drafter_ms, args.K_static)
            ra = evaluate_curve(thinned, args.W, curve, args.drafter_ms, args.Kmin,
                                args.Kmax, args.K_static, args.windows,
                                f"{label} ({key}) THINNED→E={args.anchor_eaccept}")
            ra["thin_drop_prob"] = d
            ra["curve_path"], ra["curve_key"], ra["curve_label"] = path, key, label + "_thin"
            print_curve_report(ra)
            curve_results.append(ra)

    payload = {"config": vars(args), "trace_meta": meta,
               "instrument": instr, "curves": curve_results}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[accepthist] wrote {args.output}")

    if not args.no_wandb and args.wandb_name:
        try:
            log_wandb(args, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[accepthist] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[accepthist] DONE")


def log_wandb(args, payload):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="analysis", config=payload["config"])
    instr = payload["instrument"]
    summary = {
        "accept_rate": instr["base_accept_rate"],
        "n_positions": instr["n_positions"],
        "frac_miss0": instr["accepted_len"]["frac_miss0"],
        "frac_saturate": instr["accepted_len"]["frac_saturate"],
        "accepted_len_mean": instr["accepted_len"]["mean"],
        "accepted_len_sd": instr["accepted_len"]["sd"],
    }
    for k, v in instr["pos_autocorr"].items():
        summary[f"pos_autocorr_{k}"] = v
    for k, v in instr["invoke_autocorr"].items():
        summary[f"invoke_autocorr_{k}"] = v
    for k, v in instr["window_predictiveness"].items():
        summary[f"winpred_{k}_r"] = v["r"]
    for r in payload["curves"]:
        lab = r["curve_label"].replace("|", "_")
        summary[f"{lab}_static11_tps"] = r["static_K_static"]["tps"]
        summary[f"{lab}_beststatic_K"] = r["best_static_K"]
        summary[f"{lab}_beststatic_tps"] = r["best_static_tps"]
        summary[f"{lab}_oracle_tps"] = r["oracle_tps"]
        summary[f"{lab}_oracle_gain_vs11_pct"] = r["oracle_gain_vs_static11_pct"]
        summary[f"{lab}_realizable_tps"] = r["best_realizable_tps"]
        summary[f"{lab}_realizable_gain_vs11_pct"] = r["best_realizable_gain_vs_static11_pct"]
        summary[f"{lab}_realizable_frac_of_oracle"] = r["realizable_fraction_of_oracle"]
    run.summary.update(summary)
    run.finish()


if __name__ == "__main__":
    main()
