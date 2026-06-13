#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-salvage acceptance model: width-W tree vs linear K (PR #26).

Extends the PR #18 int4 verify-cost curve (`scripts/profiler/spec_cost_model.py`,
`research/spec_cost_model/results.json`) from LINEAR speculative decoding to
WIDTH-W TREE decoding, using the measured top-1..top-K acceptance of our EAGLE-3
debug head (PR #16, measured by `scripts/drafter/eval_eagle3.py --top_k`).

Question (PR #26): given our drafter (EAGLE-3, top-1 tf_acc ~0.68, top-4 ~0.86),
does a width-4 tree raise E[accepted tok/invoke] enough — after the extra
tree-verify positions — to lift the realistic TPS ceiling above linear K=6 and
toward the >500 TPS drafter-ladder target at full-scale acceptance?

TWO ACCEPTANCE MODELS
  - i.i.d. (Step 2): geometric, E = (1 - q^(K+1))/(1-q); q = per-position accept.
      linear  q = p              (top-1 acceptance)
      tree-W  q = p + rescue*(1-p)  (rescue measured; == measured top-W at the
              debug p, extrapolated to higher p targets). q_indep = 1-(1-p)^W is
              also reported as an independence bound.
  - empirical (Step 3): simulate the spec-decode accept protocol on the measured
      per-position hit-rank trace (no i.i.d. assumption — captures the easy/hard
      span correlation that the geometric model ignores).

TWO VERIFY-LATENCY MODELS (per invoke; M = K*W + 1 query positions verified)
  - measured (PRIMARY): latency(M) interpolated from the PR #18 graph|ctx256 curve.
      A width-W depth-K tree verifies K*W candidates + 1 root in ONE 1-request,
      M-query-token forward — the exact primitive PR #18 measured as ~flat to
      M=16. So the tree costs the same as a linear verify of the same total M:
      this is the PR's "better estimate" V_tree(K,W) ~= V_linear(M=K*W).
  - additive (BOUND): V = (K*W) * (verify_base_ms / 6), the naive "tree = W x
      linear" penalty. Surfaced as a pessimistic bound; it ignores PR #18's
      measured bandwidth-bound flat floor, so it OVER-states tree-verify cost.

TPS = E / ((drafter_ms + V_ms)/1000). A verify-only ceiling (drafter_ms=0) is
also reported to line up with PR #18's numbers.

LOCAL, CPU-ONLY. No GPU, no vLLM, no HF Job. Inputs: measured acceptance scalars
+ the PR #18 results.json + (optionally) the Step-1 hit-rank trace.
"""
from __future__ import annotations

import argparse
import json
import os

FABLEOUS_RESCUE = 0.431  # public: width-4 tree rescues 43.1% of linear misses


# --------------------------------------------------------------------------- #
# PR #18 measured verify-latency curve
# --------------------------------------------------------------------------- #
def load_latency_curve(path: str, key: str) -> dict[int, float]:
    d = json.load(open(path))
    cm = d.get("cost_model", {})
    node = cm.get(key)
    if node is None:
        raise SystemExit(f"cost-model key {key!r} not in {path}; have {list(cm)}")
    return {int(m): float(v) for m, v in node["latency_ms_by_M"].items()}


class LatencyCurve:
    """latency(M) from PR #18 graph|ctx256 with interpolation + flat-tail extrapolation."""

    def __init__(self, lat: dict[int, float]):
        self.lat = lat
        self.M = sorted(lat)
        self.mmin, self.mmax = self.M[0], self.M[-1]
        a, b = self.M[-2], self.M[-1]
        self.tail_slope = (lat[b] - lat[a]) / (b - a)  # ms per added query token

    def at(self, M: float) -> float:
        M = max(1.0, M)
        if M <= self.mmin:
            return self.lat[self.mmin]
        if M >= self.mmax:
            return self.lat[self.mmax] + self.tail_slope * (M - self.mmax)
        lo = max(m for m in self.M if m <= M)
        hi = min(m for m in self.M if m >= M)
        if lo == hi:
            return self.lat[lo]
        t = (M - lo) / (hi - lo)
        return self.lat[lo] * (1 - t) + self.lat[hi] * t

    def in_range(self, M: float) -> bool:
        return self.mmin <= M <= self.mmax


# --------------------------------------------------------------------------- #
# Acceptance + expected-emitted models
# --------------------------------------------------------------------------- #
def E_iid(q: float, K: int) -> float:
    """E[emitted tok/invoke] for i.i.d. per-position accept prob q, draft depth K.

    Geometric (Leviathan/Chen): includes the +1 bonus token. q->1 => K+1."""
    q = min(max(q, 0.0), 1.0 - 1e-9)
    return (1.0 - q ** (K + 1)) / (1.0 - q)


def q_rescue(p: float, W: int, rescue: float) -> float:
    """Width-W per-position accept prob via rescue-rate extrapolation.

    rescue = fraction of top-1 misses caught by widening to top-W. At the measured
    p this reproduces the measured top-W exactly; for higher-p targets it assumes
    the rescue fraction is regime-stable (same lever, stronger base)."""
    if W <= 1:
        return p
    return min(1.0 - 1e-9, p + rescue * (1.0 - p))


def q_indep(p: float, W: int) -> float:
    """Independence-across-candidates bound: at least one of W i.i.d. draws hits."""
    return 1.0 - (1.0 - p) ** W


# --------------------------------------------------------------------------- #
# Empirical spec-decode simulation (Step 3) — no i.i.d. assumption
# --------------------------------------------------------------------------- #
def simulate_sequence(hit_rank: list[int], K: int, W: int) -> tuple[int, int]:
    """One held-out sequence under the spec-decode accept protocol.

    accept[j] = reference token is in the draft's top-W at position j
              = (1 <= hit_rank[j] <= W).
    Each invoke drafts up to K positions, accepts the longest matched prefix, then
    emits +1 bonus (the verifier's correct token at the first mismatch, or the free
    token after K accepts). Returns (tokens_emitted, n_invokes)."""
    n = len(hit_rank)
    acc = [1 if 1 <= r <= W else 0 for r in hit_rank]
    pos = invokes = 0
    while pos < n:
        matched = 0
        while matched < K and pos + matched < n and acc[pos + matched]:
            matched += 1
        emitted = min(matched + 1, n - pos)  # +1 bonus, capped at sequence end
        pos += emitted
        invokes += 1
    return n, invokes


def empirical_E(traces: list[dict], K: int, W: int) -> dict:
    tot_tok = tot_inv = 0
    for tr in traces:
        hr = tr.get("hit_rank") or []
        if not hr:
            continue
        t, i = simulate_sequence(hr, K, W)
        tot_tok += t
        tot_inv += i
    return {"E": tot_tok / max(1, tot_inv), "tokens": tot_tok, "invokes": tot_inv}


def trace_aggregate_accept(traces: list[dict], W: int) -> float:
    """Mean per-position accept rate (ref in top-W) over the whole trace."""
    hit = tot = 0
    for tr in traces:
        for r in tr.get("hit_rank") or []:
            tot += 1
            if 1 <= r <= W:
                hit += 1
    return hit / max(1, tot)


def load_trace(path: str) -> tuple[list[dict], dict]:
    meta, traces = {}, []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "meta" in d:
                meta = d["meta"]
            elif "hit_rank" in d:
                traces.append(d)
    return traces, meta


# --------------------------------------------------------------------------- #
# Cost-model assembly
# --------------------------------------------------------------------------- #
def tps(E: float, V_ms: float, drafter_ms: float) -> float:
    denom = (drafter_ms + V_ms) / 1000.0
    return E / denom if denom > 0 else 0.0


def sweep_tps(curve, p, W, rescue, K_range, drafter_ms, verify_base_ms,
              tree_mults):
    """For one (p, W): TPS vs K under the measured curve, additive bound, and (for
    tree) the multiplier bounds. Returns per-K rows + the argmax-K* summaries."""
    q = q_rescue(p, W, rescue)
    v_per_tok = verify_base_ms / 6.0  # PR #18 said ~7ms ~= K=6 verify -> 1.17ms/tok
    rows = []
    for K in range(K_range[0], K_range[1] + 1):
        M = K * W + 1
        E = E_iid(q, K)
        V_meas = curve.at(M)
        V_add = (K * W) * v_per_tok
        row = {
            "K": K, "W": W, "p": p, "q": q, "M": M, "E": E,
            "V_meas_ms": V_meas, "V_add_ms": V_add,
            "M_in_range": curve.in_range(M),
            "tps_meas": tps(E, V_meas, drafter_ms),
            "tps_meas_verifyonly": tps(E, V_meas, 0.0),
            "tps_add": tps(E, V_add, drafter_ms),
        }
        if W > 1:
            # multiplier framing: V_tree = mult * V_linear(M=K+1) measured
            V_lin = curve.at(K + 1)
            for m in tree_mults:
                row[f"tps_mult{m:g}x"] = tps(E, m * V_lin, drafter_ms)
        rows.append(row)

    def kstar(field):
        best = max(rows, key=lambda r: r[field])
        return {"K_star": best["K"], "tps": best[field], "E": best["E"],
                "q": best["q"], "M": best["M"], "M_in_range": best["M_in_range"]}

    summ = {"q": q, "meas": kstar("tps_meas"),
            "meas_verifyonly": kstar("tps_meas_verifyonly"),
            "add": kstar("tps_add")}
    if W > 1:
        for m in tree_mults:
            summ[f"mult{m:g}x"] = kstar(f"tps_mult{m:g}x")
    return rows, summ


# --------------------------------------------------------------------------- #
# Plotting (optional)
# --------------------------------------------------------------------------- #
def make_plots(all_rows, p_list, widths, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[tree] matplotlib unavailable, skipping plots ({e!r})", flush=True)
        return {}
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    # E[accept] vs K and TPS vs K (measured, with drafter)
    for metric, ylab, fname in [("E", "E[accepted tok/invoke]", "E_vs_K.png"),
                                ("tps_meas", "TPS ceiling (measured verify + drafter)",
                                 "tps_vs_K.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for p in p_list:
            for W in widths:
                xs = [r["K"] for r in all_rows if r["p"] == p and r["W"] == W]
                ys = [r[metric] for r in all_rows if r["p"] == p and r["W"] == W]
                style = "-" if W > 1 else "--"
                ax.plot(xs, ys, style, label=f"W={W} p={p:.3f}")
        ax.set_xlabel("draft depth K")
        ax.set_ylabel(ylab)
        ax.set_title(ylab + " vs K")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        path = os.path.join(outdir, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths[fname] = path
    return paths


# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1-acc", "--top1_acc", dest="top1_acc", type=float,
                    required=True, help="measured top-1 acceptance (debug head)")
    ap.add_argument("--top4-acc", "--top4_acc", dest="top4_acc", type=float,
                    required=True, help="measured top-W acceptance (W=widths max)")
    ap.add_argument("--rescue-rate", "--rescue_rate", dest="rescue_rate", type=float,
                    default=None, help="default = (top4-top1)/(1-top1)")
    ap.add_argument("--drafter-ms", "--drafter_ms", dest="drafter_ms", type=float,
                    default=1.4)
    ap.add_argument("--verify-base-ms", "--verify_base_ms", dest="verify_base_ms",
                    type=float, default=7.0, help="additive model: ~K=6 verify ms")
    ap.add_argument("--cost-model-json", "--cost_model_json", dest="cost_model_json",
                    default="research/spec_cost_model/results.json")
    ap.add_argument("--cost-key", "--cost_key", dest="cost_key", default="graph|ctx256")
    ap.add_argument("--K-range", "--K_range", dest="K_range", type=int, nargs=2,
                    default=[1, 20])
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--p-list", "--p_list", dest="p_list", type=float, nargs="+",
                    default=None, help="top-1 acceptance scenarios; default "
                                       "[measured, 0.78, 0.85]")
    ap.add_argument("--tree-mults", "--tree_mults", dest="tree_mults", type=float,
                    nargs="+", default=[1.5, 4.0])
    ap.add_argument("--sim-K", "--sim_K", dest="sim_K", type=int, default=6,
                    help="headline depth for the empirical simulation")
    ap.add_argument("--trace", default=None, help="Step-1 hit-rank JSONL (Step 3)")
    ap.add_argument("--output", default="research/spec_cost_model/tree_results.json")
    ap.add_argument("--plot-dir", "--plot_dir", dest="plot_dir",
                    default="research/spec_cost_model/tree_plots")
    ap.add_argument("--wandb_project", "--wandb-project", dest="wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    ap.add_argument("--wandb_entity", "--wandb-entity", dest="wandb_entity",
                    default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="tree-salvage-acceptance-model")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    top1, top4 = args.top1_acc, args.top4_acc
    rescue = args.rescue_rate
    if rescue is None:
        rescue = (top4 - top1) / (1.0 - top1) if top1 < 1.0 else 0.0
    Wmax = max(args.widths)
    p_list = args.p_list or [round(top1, 4), 0.78, 0.85]

    curve = LatencyCurve(load_latency_curve(args.cost_model_json, args.cost_key))
    print("=" * 78, flush=True)
    print(f"[tree] measured: top1={top1:.4f} top{Wmax}={top4:.4f} rescue={rescue:.4f} "
          f"(fableous {FABLEOUS_RESCUE:.3f}, delta={rescue - FABLEOUS_RESCUE:+.4f})",
          flush=True)
    print(f"[tree] verify curve {args.cost_key}: M={curve.mmin}..{curve.mmax} "
          f"lat={curve.lat[curve.mmin]:.2f}..{curve.lat[curve.mmax]:.2f}ms "
          f"tail_slope={curve.tail_slope:.4f}ms/tok; drafter={args.drafter_ms}ms", flush=True)
    print(f"[tree] p scenarios={p_list}  widths={args.widths}  K={args.K_range}", flush=True)
    print("=" * 78, flush=True)

    # ---- Step 2: i.i.d. model sweep over (p, W, K) ----
    all_rows = []
    summary_by_pw = {}
    for p in p_list:
        for W in args.widths:
            rows, summ = sweep_tps(curve, p, W, rescue, args.K_range,
                                   args.drafter_ms, args.verify_base_ms, args.tree_mults)
            all_rows.extend(rows)
            summary_by_pw[f"p{p:.4f}|W{W}"] = summ

    # Headline at fixed K = sim_K (default 6): linear vs tree, every p.
    K0 = args.sim_K
    headline = {}
    for p in p_list:
        ql, qt = q_rescue(p, 1, rescue), q_rescue(p, Wmax, rescue)
        El, Et = E_iid(ql, K0), E_iid(qt, K0)
        Ml, Mt = 1 * K0 + 1, Wmax * K0 + 1
        Vl, Vt = curve.at(Ml), curve.at(Mt)
        headline[f"p{p:.4f}"] = {
            "q_lin": ql, "q_tree": qt, "q_tree_indep": q_indep(p, Wmax),
            "E_lin": El, "E_tree": Et, "E_ratio": Et / El,
            "M_lin": Ml, "M_tree": Mt, "M_tree_in_range": curve.in_range(Mt),
            "V_lin_ms": Vl, "V_tree_ms": Vt, "V_tree_over_lin": Vt / Vl,
            "tps_lin_meas": tps(El, Vl, args.drafter_ms),
            "tps_tree_meas": tps(Et, Vt, args.drafter_ms),
            "tps_lin_verifyonly": tps(El, Vl, 0.0),
            "tps_tree_verifyonly": tps(Et, Vt, 0.0),
        }
        h = headline[f"p{p:.4f}"]
        h["tps_gain_meas"] = h["tps_tree_meas"] / h["tps_lin_meas"]

    print(f"\n[tree] ===== i.i.d. headline at K={K0} (linear vs width-{Wmax} tree) =====",
          flush=True)
    print(f"{'p(top1)':>8} {'q_tree':>7} {'E_lin':>6} {'E_tree':>7} {'Erat':>5} "
          f"{'Vlin':>6} {'Vtree':>6} {'TPSlin':>7} {'TPStree':>8} {'gain':>5} {'tree M':>8}",
          flush=True)
    for p in p_list:
        h = headline[f"p{p:.4f}"]
        flag = "" if h["M_tree_in_range"] else "*"
        print(f"{p:8.4f} {h['q_tree']:7.4f} {h['E_lin']:6.3f} {h['E_tree']:7.3f} "
              f"{h['E_ratio']:5.2f} {h['V_lin_ms']:6.2f} {h['V_tree_ms']:6.2f} "
              f"{h['tps_lin_meas']:7.1f} {h['tps_tree_meas']:8.1f} {h['tps_gain_meas']:5.2f} "
              f"{h['M_tree']:6d}{flag:>2}", flush=True)
    print("  (* tree M beyond PR#18 measured M<=16 -> verify latency extrapolated)",
          flush=True)

    # ---- optimal K* table (measured verify, with drafter) ----
    print(f"\n[tree] ===== optimal K* (measured verify + {args.drafter_ms}ms drafter) =====",
          flush=True)
    print(f"{'p':>8} {'W':>2} {'K*':>3} {'TPS@K*':>8} {'E@K*':>6} {'verify-only K*/TPS':>20}",
          flush=True)
    for p in p_list:
        for W in args.widths:
            s = summary_by_pw[f"p{p:.4f}|W{W}"]
            vo = s["meas_verifyonly"]
            print(f"{p:8.4f} {W:2d} {s['meas']['K_star']:3d} {s['meas']['tps']:8.1f} "
                  f"{s['meas']['E']:6.3f}   K*={vo['K_star']:2d}/{vo['tps']:7.1f}", flush=True)

    # ---- verify-overhead sensitivity at K = sim_K (the verdict hinges on this) ----
    # The naive fear is V_tree = W x V_linear (4x). PR #18 measured that the verify
    # forward is bandwidth-bound and ~flat in M, and a width-W tree verifies all K*W
    # candidates in ONE M=K*W+1 forward -> the MEASURED overhead is curve.at(K*W+1) /
    # curve.at(K+1), far below 4x. This table makes the dependence explicit.
    print(f"\n[tree] ===== verify-overhead sensitivity at K={K0} (width-{Wmax} tree) =====",
          flush=True)
    print(f"{'p':>8} {'TPSlin':>7} | {'measured':>9} {'1.5x':>7} {'4.0x':>7} {'additive':>9} "
          f"| {'meas ovh':>9}", flush=True)
    overhead = {}
    for p in p_list:
        lin = next(r for r in all_rows if r["p"] == p and r["W"] == 1 and r["K"] == K0)
        tr = next(r for r in all_rows if r["p"] == p and r["W"] == Wmax and r["K"] == K0)
        meas_ovh = tr["V_meas_ms"] / lin["V_meas_ms"]
        overhead[f"p{p:.4f}"] = {
            "tps_lin_meas": lin["tps_meas"], "tps_tree_meas": tr["tps_meas"],
            "tps_tree_1.5x": tr.get("tps_mult1.5x"), "tps_tree_4x": tr.get("tps_mult4x"),
            "tps_tree_additive": tr["tps_add"], "measured_overhead_x": meas_ovh,
            "tree_beats_linear_measured": tr["tps_meas"] > lin["tps_meas"],
            "tree_beats_linear_4x": tr.get("tps_mult4x", 0) > lin["tps_meas"],
        }
        o = overhead[f"p{p:.4f}"]
        print(f"{p:8.4f} {lin['tps_meas']:7.1f} | {tr['tps_meas']:9.1f} "
              f"{o['tps_tree_1.5x']:7.1f} {o['tps_tree_4x']:7.1f} {tr['tps_add']:9.1f} "
              f"| {meas_ovh:8.3f}x", flush=True)
    print("  measured overhead << 4x: the tree verifies K*W candidates in ONE flat-cost "
          "forward (PR #18).", flush=True)

    # ---- Step 3: empirical simulation from the trace ----
    empirical = None
    if args.trace and os.path.exists(args.trace):
        traces, tmeta = load_trace(args.trace)
        emp = {}
        for W in args.widths:
            e = empirical_E(traces, K0, W)
            agg = trace_aggregate_accept(traces, W)
            e_iid = E_iid(agg, K0)
            emp[f"W{W}"] = {**e, "agg_accept": agg, "E_iid_same_q": e_iid,
                            "emp_over_iid": e["E"] / e_iid if e_iid else None}
        El_emp = emp["W1"]["E"]
        Et_emp = emp[f"W{Wmax}"]["E"]
        rescue_emp = ((emp[f"W{Wmax}"]["agg_accept"] - emp["W1"]["agg_accept"]) /
                      (1.0 - emp["W1"]["agg_accept"]))
        Ml, Mt = 1 * K0 + 1, Wmax * K0 + 1
        empirical = {
            "n_sequences": len(traces), "sim_K": K0,
            "by_width": emp,
            "E_linear": El_emp, "E_tree": Et_emp, "E_ratio": Et_emp / El_emp,
            "rescue_rate_empirical": rescue_emp,
            "rescue_vs_fableous_delta": rescue_emp - FABLEOUS_RESCUE,
            "tps_linear_meas": tps(El_emp, curve.at(Ml), args.drafter_ms),
            "tps_tree_meas": tps(Et_emp, curve.at(Mt), args.drafter_ms),
            "trace_meta": tmeta,
        }
        empirical["tps_gain_meas"] = (empirical["tps_tree_meas"] /
                                      empirical["tps_linear_meas"])
        print(f"\n[tree] ===== empirical simulation (trace, {len(traces)} seqs, K={K0}) =====",
              flush=True)
        print(f"  linear (top-1): E_emp={El_emp:.3f}  vs i.i.d.={emp['W1']['E_iid_same_q']:.3f} "
              f"(emp/iid={emp['W1']['emp_over_iid']:.3f})", flush=True)
        print(f"  tree-{Wmax} (top-{Wmax}): E_emp={Et_emp:.3f}  vs i.i.d."
              f"={emp[f'W{Wmax}']['E_iid_same_q']:.3f} "
              f"(emp/iid={emp[f'W{Wmax}']['emp_over_iid']:.3f})", flush=True)
        print(f"  E_tree/E_linear (empirical) = {empirical['E_ratio']:.3f}", flush=True)
        print(f"  rescue_rate_empirical = {rescue_emp:.4f} "
              f"(fableous {FABLEOUS_RESCUE:.3f}, delta {rescue_emp - FABLEOUS_RESCUE:+.4f})",
              flush=True)
        print(f"  TPS(meas+drafter): linear={empirical['tps_linear_meas']:.1f} "
              f"tree={empirical['tps_tree_meas']:.1f} gain={empirical['tps_gain_meas']:.2f}x",
              flush=True)
    else:
        if args.trace:
            print(f"[tree] trace {args.trace} not found; skipping empirical sim", flush=True)

    # ---- verdict ----
    h78 = headline.get("p0.7800")
    verdict = {
        "tree_net_positive_measured": all(
            headline[f"p{p:.4f}"]["tps_gain_meas"] > 1.0 for p in p_list),
        "tps_ceiling_tree_at_full_scale": h78["tps_tree_meas"] if h78 else None,
        "exceeds_500_at_full_scale_verifyonly": (
            h78["tps_tree_verifyonly"] > 500.0 if h78 else None),
        "exceeds_500_at_full_scale_withdrafter": (
            h78["tps_tree_meas"] > 500.0 if h78 else None),
    }
    print(f"\n[tree] VERDICT: tree net-positive (measured, all p) = "
          f"{verdict['tree_net_positive_measured']}", flush=True)
    if h78:
        print(f"[tree] full-scale (p=0.78) tree TPS ceiling: "
              f"{h78['tps_tree_meas']:.1f} (with drafter) / "
              f"{h78['tps_tree_verifyonly']:.1f} (verify-only); >500 target "
              f"{'MET' if verdict['exceeds_500_at_full_scale_verifyonly'] else 'not met'} "
              f"(verify-only)", flush=True)

    # ---- save + plots + W&B ----
    payload = {
        "config": {
            "top1_acc": top1, "top4_acc": top4, "rescue_rate": rescue,
            "fableous_rescue": FABLEOUS_RESCUE, "drafter_ms": args.drafter_ms,
            "verify_base_ms": args.verify_base_ms, "cost_model_json": args.cost_model_json,
            "cost_key": args.cost_key, "p_list": p_list, "widths": args.widths,
            "K_range": args.K_range, "sim_K": K0, "tree_mults": args.tree_mults,
            "latency_curve": {str(k): v for k, v in curve.lat.items()},
        },
        "headline_iid": headline,
        "verify_overhead_sensitivity": overhead,
        "optimal_kstar": summary_by_pw,
        "empirical": empirical,
        "verdict": verdict,
        "rows": all_rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[tree] wrote {args.output}", flush=True)

    plot_paths = make_plots(all_rows, p_list, args.widths, args.plot_dir)

    if not args.no_wandb and args.wandb_name:
        try:
            log_wandb(args, payload, plot_paths, p_list, headline, empirical)
        except Exception as e:  # noqa: BLE001
            print(f"[tree] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[tree] DONE", flush=True)


def log_wandb(args, payload, plot_paths, p_list, headline, empirical):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    summary = {}
    cfg = payload["config"]
    summary["top1_acc"] = cfg["top1_acc"]
    summary["top4_acc"] = cfg["top4_acc"]
    summary["rescue_rate"] = cfg["rescue_rate"]
    summary["rescue_vs_fableous_delta"] = cfg["rescue_rate"] - cfg["fableous_rescue"]
    for p in p_list:
        h = headline[f"p{p:.4f}"]
        tag = f"p{p:.3f}".replace(".", "_")
        summary[f"E_lin_K{cfg['sim_K']}_{tag}"] = h["E_lin"]
        summary[f"E_tree_K{cfg['sim_K']}_{tag}"] = h["E_tree"]
        summary[f"E_ratio_K{cfg['sim_K']}_{tag}"] = h["E_ratio"]
        summary[f"tps_lin_meas_{tag}"] = h["tps_lin_meas"]
        summary[f"tps_tree_meas_{tag}"] = h["tps_tree_meas"]
        summary[f"tps_gain_meas_{tag}"] = h["tps_gain_meas"]
    if empirical:
        summary["E_accept_linear_emp"] = empirical["E_linear"]
        summary["E_accept_tree4_emp"] = empirical["E_tree"]
        summary["E_accept_tree4_over_linear_ratio"] = empirical["E_ratio"]
        summary["rescue_rate_empirical"] = empirical["rescue_rate_empirical"]
        summary["rescue_rate_empirical_vs_fableous_delta"] = \
            empirical["rescue_vs_fableous_delta"]
        summary["tps_linear_emp_meas"] = empirical["tps_linear_meas"]
        summary["tps_tree_emp_meas"] = empirical["tps_tree_meas"]
        summary["tps_gain_emp_meas"] = empirical["tps_gain_meas"]
    summary.update({f"verdict_{k}": v for k, v in payload["verdict"].items()})
    run.summary.update({k: v for k, v in summary.items() if v is not None})

    # TPS-vs-K and E-vs-K line tables
    cols = ["K", "W", "p", "E", "M", "tps_meas", "tps_meas_verifyonly", "tps_add"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"tree_cost_table": tbl})
    for name, path in plot_paths.items():
        run.log({name.replace(".png", ""): wandb.Image(path)})
    run.finish()
    print(f"[tree] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
