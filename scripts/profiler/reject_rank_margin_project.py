#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Offline margin-gated top-k accept projection (PR #823) -- the projection twin of #820.

#820 settled the relaxation frontier's first question: uniform top-k accept is a big
speed lever (+19.9% at k=2 -> +64.4% at k=16) but its quality cost is unmeasured, and
it proved **entropy is the wrong gating signal** on Gemma's peaked 262k-vocab verifier
(the rescuable rank-2 mass sits at LOW entropy, so FLy is inverted).

This script tests #820's own follow-up #2: **gate on the prob MARGIN, not entropy.**

    margin(d) = dp(d) / ap(d)  in (0, 1]
      dp = verifier softmax prob at the DRAFT token
      ap = verifier softmax prob at the ARGMAX (top-1)

A rank-2 reject with ``margin ~= 1`` is a genuine near-tie -- the verifier is nearly
indifferent between the draft and its argmax, so emitting the draft is almost free
quality-wise. A rank-2 reject with ``margin << 1`` is the verifier confidently
preferring its argmax -- emitting the draft is a real error. **Entropy cannot tell
these apart on a peaked verifier; the margin can, by construction** (it directly
measures how far below the argmax the draft sits).

Margin-gated accept criterion (drop-in addition to stark #816's top-k test):

    accept(d) = strict-argmax (acc==1)  OR  ( rank(d) <= k  AND  margin(d) >= m )

so ``m = 0`` reduces EXACTLY to uniform top-k (#820's lever), and raising ``m`` refuses
the low-margin (genuine-error) relaxes while keeping the high-margin (near-tie) ones.

Reuses #820's faithful block-replay accounting and stark's oracle TPS curve
``TPS = a + b*L`` (read off the SAME anchors as #820) so every number here is directly
comparable to #820. NO GPU, NO kernel change, NO re-serve: reads the JSONL that
``reject_rank_capture.py`` already wrote (per draft position: acc, rk, H, dp, ap).

  Step 1  margin distribution at reject positions + the DECISIVE cross-tab vs entropy:
          margin-conditioned reject-rank CDF (does high margin isolate the rank-2
          near-misses where entropy did not?) shown beside #820's entropy table, plus
          P(rank<=k | margin decile).
  Step 2  the (k, margin) grid in {2,5,16} x {0.0, 0.5, 0.8, 0.95}: realized
          E_accept/TPS (Delta% vs ~253 anchor AND vs the uniform-k point at the same k),
          and the quality proxy -- how many of uniform-k's relaxes are LOW-margin
          (genuine-error risk, removed by the gate) vs HIGH-margin (benign near-tie).
  Step 3  does margin DOMINATE uniform-k? Find the single (k, margin) that keeps
          >= 90% of uniform-k's gain while cutting the risky low-margin relaxes the
          most. If no margin setting separates -> clean negative, ship plain uniform-k.

    python -m scripts.profiler.reject_rank_margin_project \
        --in-dir research/reject_rank_entropy/int4head \
        --wandb-group bi0-margin-gate-accept
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse #820's VALIDATED pure helpers so the method is identical and comparable.
from scripts.profiler.reject_rank_project import (  # noqa: E402
    BASELINE_E_ACCEPT, BASELINE_R, BASELINE_TPS, DEFAULT_ORACLE_POINTS,
    FLY_THETA, GEMMA_VOCAB, QUALITY_FLOORS,
    _accepted_len_uniform, _entropy_describe, _linfit, _parse_oracle_points,
    _project_uniform, _quantiles, _reject_rank_cdf, _strict_stats, _tps_of_L,
)

DEFAULT_K_LIST = "2,5,16"
DEFAULT_MARGIN_LIST = "0.0,0.5,0.8,0.95"
# Extra thresholds (beyond the grid) purely to RESOLVE the margin-conditioned CDF /
# decile separation in step 1; not part of the (k, margin) decision grid.
DEFAULT_MARGIN_CDF_THRESH = "0.1,0.3,0.5,0.8,0.95"
# #820's entropy thresholds, re-emitted for the side-by-side contrast.
DEFAULT_H_NATS_LIST = "0.1,0.25,0.5,1.0,2.0,3.743"
# The #820 "52-65% low-entropy risky" headline came from this nats boundary.
H_NATS_820_HEADLINE = 1.0
RETENTION_DOMINATE_PCT = 90.0   # gate must keep >= this % of uniform-k's gain to "dominate"
DEFAULT_GO_PCT = 10.0


# --------------------------------------------------------------------------- #
# Load (margin-aware: requires dp/ap in addition to #820's acc/rk/H)
# --------------------------------------------------------------------------- #
def _load_blocks_margin(shards: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in shards:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n = rec["n"]
                if not (len(rec["acc"]) == len(rec["rk"]) == len(rec["H"])
                        == len(rec["dp"]) == len(rec["ap"]) == n):
                    continue
                blocks.append(rec)
    return blocks


def _read_vocab(shards: list[str]) -> int | None:
    for path in shards:
        meta = path + ".meta.json"
        if Path(meta).exists():
            try:
                v = json.loads(Path(meta).read_text()).get("vocab")
                if v:
                    return int(v)
            except Exception:  # noqa: BLE001
                pass
    return None


def _margin(dp: float, ap: float) -> float:
    """margin = dp/ap in [0,1]. ap is the verifier top-1 prob (>0 over 262k vocab);
    dp <= ap by definition so the ratio is <=1 (clamped for float-rounding ties)."""
    if ap <= 0.0:
        return 0.0
    return min(1.0, dp / ap)


# --------------------------------------------------------------------------- #
# Block replay -- margin-gated top-k
# --------------------------------------------------------------------------- #
def _accepted_len_margin(rk: list[int], dp: list[float], ap: list[float],
                         acc: list[int], k: int, margin: float) -> int:
    """accept(d) = strict-argmax (acc==1) OR (rank<=k AND margin>=m).

    margin=0 collapses to uniform top-k (#820): (rank<=k AND margin>=0) == (rank<=k).
    """
    for d in range(len(rk)):
        keep = (acc[d] == 1) or (rk[d] <= k and _margin(dp[d], ap[d]) >= margin)
        if not keep:
            return d
    return len(rk)


def _project_margin_gated(blocks: list[dict[str, Any]], k: int, margin: float,
                          fit: dict[str, float], baseline_tps: float) -> dict[str, Any]:
    """Margin-gated top-k E_accept/TPS. relaxes_below_margin must be 0 by construction
    (counted to PROVE it on the realized data); relaxes_at_or_above_margin are the
    near-tie emissions the gate keeps."""
    sum_len = 0.0
    relaxes_below = relaxes_atabove = 0
    for b in blocks:
        rk, dp, ap, acc = b["rk"], b["dp"], b["ap"], b["acc"]
        fd = _accepted_len_margin(rk, dp, ap, acc, k, margin)
        sum_len += fd + 1
        for d in range(fd):
            if acc[d] == 0:
                if _margin(dp[d], ap[d]) >= margin:
                    relaxes_atabove += 1
                else:
                    relaxes_below += 1
    n_blocks = len(blocks)
    e_acc = sum_len / n_blocks if n_blocks else float("nan")
    tps = _tps_of_L(e_acc, fit)
    return {"k": k, "margin": margin, "E_accept": e_acc, "tps": tps,
            "tps_gain_abs": tps - baseline_tps,
            "tps_gain_pct": 100.0 * (tps - baseline_tps) / baseline_tps,
            "relaxes_below_margin": relaxes_below,
            "relaxes_atabove_margin": relaxes_atabove}


def _uniform_relax_split_margin(blocks: list[dict[str, Any]], k: int,
                                margin: float) -> tuple[int, int]:
    """(risky, safe) = uniform(k) relaxed-accepts with margin<m (verifier confident,
    genuine-error risk) vs margin>=m (near-tie, benign). The margin gate refuses the
    risky ones; safe ones it keeps (modulo cascade truncation, handled in replay)."""
    risky = safe = 0
    for b in blocks:
        rk, dp, ap, acc = b["rk"], b["dp"], b["ap"], b["acc"]
        fd = _accepted_len_uniform(rk, k)
        for d in range(fd):
            if acc[d] == 0:
                if _margin(dp[d], ap[d]) >= margin:
                    safe += 1
                else:
                    risky += 1
    return risky, safe


# --------------------------------------------------------------------------- #
# Step 1 -- margin distribution + margin-conditioned CDF + decile table
# --------------------------------------------------------------------------- #
def _margin_describe(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    all_m: list[float] = []
    rank2_m: list[float] = []
    for b in blocks:
        acc, rk, dp, ap = b["acc"], b["rk"], b["dp"], b["ap"]
        for d in range(b["n"]):
            if acc[d] == 0:
                m = _margin(dp[d], ap[d])
                all_m.append(m)
                if rk[d] == 2:
                    rank2_m.append(m)
    # fraction of rejects at/above key margin thresholds (the grid points)
    frac_atabove = {}
    for thr in (0.5, 0.8, 0.95):
        c = sum(1 for m in all_m if m >= thr)
        frac_atabove[f"ge_{thr}"] = {"count": c,
                                     "frac": (c / len(all_m)) if all_m else float("nan")}
    return {
        "n_rejects": len(all_m),
        "all_rejects_margin": _quantiles(all_m),
        "rank2_rejects_margin": _quantiles(rank2_m),
        "n_rank2_rejects": len(rank2_m),
        "frac_rejects_at_or_above": frac_atabove,
    }


def _reject_rank_cdf_margin(blocks: list[dict[str, Any]], ks: list[int],
                            margin_thr: float) -> dict[str, Any]:
    """Reject-rank CDF split by HIGH margin (>= thr, near-tie) vs LOW margin (< thr).
    The margin analog of #820's entropy-conditioned CDF -- the decisive contrast:
    does high margin isolate the rank-2 near-misses where entropy did not?"""
    hi_ranks: list[int] = []
    lo_ranks: list[int] = []
    for b in blocks:
        acc, rk, dp, ap = b["acc"], b["rk"], b["dp"], b["ap"]
        for d in range(b["n"]):
            if acc[d] == 0:
                (hi_ranks if _margin(dp[d], ap[d]) >= margin_thr else lo_ranks).append(rk[d])

    def _cdf(ranks: list[int]) -> dict[str, float]:
        n = len(ranks)
        return {f"le_{k}": (sum(1 for r in ranks if r <= k) / n if n else float("nan"))
                for k in ks}

    return {"margin_thr": margin_thr,
            "n_rejects_high_margin": len(hi_ranks),
            "n_rejects_low_margin": len(lo_ranks),
            "cdf_high_margin_rejects": _cdf(hi_ranks),
            "cdf_low_margin_rejects": _cdf(lo_ranks)}


def _margin_decile_table(blocks: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    """For each margin decile [lo,hi) of dp/ap at reject positions: count, P(rank==2),
    P(rank<=k). Shows where the rank-2 near-misses concentrate in margin."""
    buckets = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    rows = []
    for lo, hi in buckets:
        ranks: list[int] = []
        for b in blocks:
            acc, rk, dp, ap = b["acc"], b["rk"], b["dp"], b["ap"]
            for d in range(b["n"]):
                if acc[d] == 0:
                    m = _margin(dp[d], ap[d])
                    # last bucket is closed on the right so margin==1.0 lands in [0.9,1.0]
                    if (lo <= m < hi) or (hi == 1.0 and m == 1.0):
                        ranks.append(rk[d])
        n = len(ranks)
        row = {"lo": lo, "hi": hi, "count": n,
               "p_rank2": (sum(1 for r in ranks if r == 2) / n) if n else float("nan")}
        for k in ks:
            row[f"p_le_{k}"] = (sum(1 for r in ranks if r <= k) / n) if n else float("nan")
        rows.append(row)
    return rows


def _quality_proxy_crosstab(blocks: list[dict[str, Any]], k: int,
                            margin_thr: float, h_nats: float) -> dict[str, Any]:
    """2x2 cross-tab of uniform(k)'s relaxed-accepts by margin (the TRUE risk signal)
    vs entropy (#820's refuted signal), at matched boundaries. Connects to #820's
    "52-65% low-entropy" headline and shows what the margin gate actually removes."""
    # cells: [low_margin][low_entropy] etc. low_margin == genuine-error risk (gate drops).
    lm_le = lm_he = hm_le = hm_he = 0
    for b in blocks:
        rk, dp, ap, acc, H = b["rk"], b["dp"], b["ap"], b["acc"], b["H"]
        fd = _accepted_len_uniform(rk, k)
        for d in range(fd):
            if acc[d] == 0:
                low_margin = _margin(dp[d], ap[d]) < margin_thr
                low_entropy = H[d] <= h_nats
                if low_margin and low_entropy:
                    lm_le += 1
                elif low_margin and not low_entropy:
                    lm_he += 1
                elif not low_margin and low_entropy:
                    hm_le += 1
                else:
                    hm_he += 1
    total = lm_le + lm_he + hm_le + hm_he
    low_margin_total = lm_le + lm_he
    low_entropy_total = lm_le + hm_le
    return {"k": k, "margin_thr": margin_thr, "h_nats": h_nats,
            "uniform_relaxes_total": total,
            "low_margin_low_entropy": lm_le, "low_margin_high_entropy": lm_he,
            "high_margin_low_entropy": hm_le, "high_margin_high_entropy": hm_he,
            "low_margin_total": low_margin_total,
            "low_entropy_total": low_entropy_total,
            "frac_low_margin": (low_margin_total / total) if total else float("nan"),
            "frac_low_entropy": (low_entropy_total / total) if total else float("nan")}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path,
                    default=ROOT / "research" / "reject_rank_entropy" / "int4head")
    ap.add_argument("--records-base", type=str, default="rejectrank_records.jsonl")
    ap.add_argument("--k-list", type=str, default=DEFAULT_K_LIST)
    ap.add_argument("--margin-list", type=str, default=DEFAULT_MARGIN_LIST)
    ap.add_argument("--margin-cdf-thresh", type=str, default=DEFAULT_MARGIN_CDF_THRESH)
    ap.add_argument("--h-nats-list", type=str, default=DEFAULT_H_NATS_LIST)
    ap.add_argument("--h-nats-headline", type=float, default=H_NATS_820_HEADLINE)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--oracle-points", type=str, default=DEFAULT_ORACLE_POINTS)
    ap.add_argument("--baseline-tps", type=float, default=BASELINE_TPS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", type=str, default="bi0-margin-gate-accept")
    ap.add_argument("--wandb-name", type=str, default="fern/margin-gate-accept-project")
    args = ap.parse_args(argv)

    baseline_tps = args.baseline_tps
    in_dir = args.in_dir.resolve()
    out_dir = (args.out_dir or in_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = str(in_dir / args.records_base)
    shards = sorted(s for s in glob.glob(base + ".*") if not s.endswith(".meta.json"))
    if not shards:
        print(f"[margin] ERROR: no record shards at {base}.* — run reject_rank_capture first",
              flush=True)
        return 1
    blocks = _load_blocks_margin(shards)
    if not blocks:
        print("[margin] ERROR: shards present but no parseable blocks (need acc/rk/H/dp/ap)",
              flush=True)
        return 1

    vocab = args.vocab or _read_vocab(shards) or GEMMA_VOCAB
    log_v = math.log(vocab)

    ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    margins = [float(x) for x in args.margin_list.split(",") if x.strip()]
    margin_cdf_thr = [float(x) for x in args.margin_cdf_thresh.split(",") if x.strip()]
    h_nats_list = [float(x) for x in args.h_nats_list.split(",") if x.strip()]
    fit = _linfit(_parse_oracle_points(args.oracle_points))

    # ---- self-consistency (must reproduce #820 / the shipped strict path) ----
    strict = _strict_stats(blocks)
    strict_tps = _tps_of_L(strict["E_accept"], fit)
    e_acc_ok = abs(strict["E_accept"] - BASELINE_E_ACCEPT) <= 0.15
    r_ok = abs(strict["r"] - BASELINE_R) <= 0.03

    # ---- Step 1: distributions + the decisive contrast ----
    cdf = _reject_rank_cdf(blocks, ks)
    margin_desc = _margin_describe(blocks)
    ent = _entropy_describe(blocks, log_v)
    margin_cdf = [_reject_rank_cdf_margin(blocks, ks, m) for m in margin_cdf_thr]
    entropy_cdf = [{"h_nats": h, "h_norm": h / log_v, **_reject_rank_cdf(blocks, ks, h_nats=h)}
                   for h in h_nats_list]
    decile_table = _margin_decile_table(blocks, ks)

    # ---- uniform projection (same numbers as #820; the speed reference) ----
    uniform = [_project_uniform(blocks, k, fit, baseline_tps) for k in ks]
    uni_by_k = {u["k"]: u for u in uniform}
    best_uniform = max(uniform, key=lambda r: r["tps"])

    # ---- Step 2: (k, margin) grid ----
    grid: list[dict[str, Any]] = []
    for k in ks:
        u = uni_by_k[k]
        for m in margins:
            g = _project_margin_gated(blocks, k, m, fit, baseline_tps)
            risky, safe = _uniform_relax_split_margin(blocks, k, m)
            gain_u = u["tps_gain_abs"]
            gain_g = g["tps_gain_abs"]
            grid.append({
                "k": k, "margin": m,
                "tps_uniform": u["tps"], "tps_gated": g["tps"],
                "E_accept_uniform": u["E_accept"], "E_accept_gated": g["E_accept"],
                "gain_uniform": gain_u, "gain_gated": gain_g,
                "gain_pct_vs_anchor": g["tps_gain_pct"],
                "retention_pct_same_k": (100.0 * gain_g / gain_u) if gain_u > 0 else float("nan"),
                "retention_pct_vs_best_uniform": (
                    100.0 * gain_g / best_uniform["tps_gain_abs"]
                    if best_uniform["tps_gain_abs"] > 0 else float("nan")),
                "uniform_relaxes_risky_low_margin": risky,   # the gate refuses these
                "uniform_relaxes_safe_high_margin": safe,
                "gated_relaxes_below_margin": g["relaxes_below_margin"],   # must be 0
                "gated_relaxes_atabove_margin": g["relaxes_atabove_margin"],
            })

    # ---- Step 2 quality proxy: cross-tab vs #820's entropy headline ----
    crosstabs = [_quality_proxy_crosstab(blocks, k, m, args.h_nats_headline)
                 for k in ks for m in (mm for mm in margins if mm > 0)]

    # ---- Step 3: does margin DOMINATE uniform-k? ----
    # qualifying = keeps >= RETENTION_DOMINATE_PCT of the SAME-k uniform gain while
    # removing >0 genuine-error (low-margin) relaxes. Among those pick the one cutting
    # the MOST risky relaxes (tie-break: faster).
    qualifying = [row for row in grid
                  if row["margin"] > 0
                  and row["uniform_relaxes_risky_low_margin"] > 0
                  and row["retention_pct_same_k"] >= RETENTION_DOMINATE_PCT]
    recommended = (max(qualifying,
                       key=lambda r: (r["uniform_relaxes_risky_low_margin"], r["tps_gated"]))
                   if qualifying else None)
    margin_dominates = recommended is not None

    # best retention achievable at ANY margin>0 (how close we got to the 90% bar)
    margin_rows = [r for r in grid if r["margin"] > 0]
    best_retention_row = max(margin_rows, key=lambda r: r["retention_pct_same_k"]) \
        if margin_rows else None

    # ---- verdict ----
    best_gain_pct = best_uniform["tps_gain_pct"]
    lever_verdict = ("GO" if best_gain_pct >= DEFAULT_GO_PCT
                     else ("AMBER" if best_gain_pct >= 3.0 else "NO-GO"))
    # ship recommendation: the margin-gated config if it dominates, else plain uniform-k.
    if margin_dominates:
        ship = {"kind": "margin_gated", "k": recommended["k"],
                "margin": recommended["margin"], "tps": recommended["tps_gated"],
                "gain_pct": recommended["gain_pct_vs_anchor"]}
    else:
        ship = {"kind": "uniform_topk", "k": best_uniform["k"], "margin": 0.0,
                "tps": best_uniform["tps"], "gain_pct": best_uniform["tps_gain_pct"]}

    result = {
        "pr": 823, "in_dir": str(in_dir), "shards": shards, "n_blocks": len(blocks),
        "vocab": vocab, "log_vocab": log_v,
        "oracle_fit": fit, "baseline_tps": baseline_tps,
        "strict_self_consistency": {
            **strict, "tps_at_strict": strict_tps,
            "E_accept_matches_baseline": e_acc_ok, "r_matches_baseline": r_ok,
            "baseline_E_accept": BASELINE_E_ACCEPT, "baseline_r": BASELINE_R,
        },
        "margin_describe": margin_desc,
        "entropy_describe": ent,
        "reject_rank_cdf": cdf,
        "margin_conditioned_cdf": margin_cdf,
        "entropy_conditioned_cdf": entropy_cdf,
        "margin_decile_table": decile_table,
        "projection_uniform_topk": uniform,
        "margin_grid": grid,
        "quality_proxy_crosstabs": crosstabs,
        "best_uniform": best_uniform,
        "recommended_margin_gated": recommended,
        "best_retention_row": best_retention_row,
        "margin_dominates_uniform": margin_dominates,
        "lever_verdict": lever_verdict,
        "ship_recommendation": ship,
        "h_nats_headline": args.h_nats_headline,
        "quality_floors": QUALITY_FLOORS,
        "retention_dominate_pct": RETENTION_DOMINATE_PCT,
    }
    (out_dir / "reject_rank_margin_projection.json").write_text(json.dumps(result, indent=2))
    _write_markdown(out_dir / "reject_rank_margin_projection.md", result, ks, margins)
    _print_summary(result, ks)
    if args.wandb:
        _maybe_log_wandb(result, args)

    print(f"\n[margin] artifacts -> {out_dir}", flush=True)
    if not (e_acc_ok and r_ok):
        print("[margin] WARNING: strict self-consistency vs baseline is OFF — "
              "probe may not have seen the true greedy path (or sample too small).", flush=True)
    return 0


def _print_summary(r: dict[str, Any], ks: list[int]) -> None:
    s = r["strict_self_consistency"]
    md = r["margin_describe"]
    print("\n========== MARGIN-GATED ACCEPT PROJECTION (PR #823) ==========", flush=True)
    print(f"blocks={r['n_blocks']}  vocab={r['vocab']}  "
          f"oracle TPS={r['oracle_fit']['intercept']:.2f}+{r['oracle_fit']['slope']:.2f}*L "
          f"(R2={r['oracle_fit']['r2']:.4f})", flush=True)
    print(f"strict E_accept={s['E_accept']:.4f} (base {s['baseline_E_accept']}, "
          f"ok={s['E_accept_matches_baseline']})  r={s['r']:.4f} "
          f"(base {s['baseline_r']}, ok={s['r_matches_baseline']})  TPS@strict={s['tps_at_strict']:.1f}",
          flush=True)
    am = md["all_rejects_margin"]
    r2 = md["rank2_rejects_margin"]
    print(f"margin=dp/ap @ rejects: median={am['p50']:.3f} mean={am['mean']:.3f} "
          f"p90={am['p90']:.3f} max={am['max']:.3f}  |  @rank-2: median={r2['p50']:.3f} "
          f"max={r2['max']:.3f}", flush=True)
    fa = md["frac_rejects_at_or_above"]
    print("frac rejects margin>=: " + "  ".join(
        f"{thr}:{fa[f'ge_{thr}']['frac']:.3f}({fa[f'ge_{thr}']['count']})"
        for thr in (0.5, 0.8, 0.95)), flush=True)

    print("\nUNIFORM top-k (=margin 0; same as #820):", flush=True)
    for u in r["projection_uniform_topk"]:
        print(f"  k={u['k']:>2}  E_accept={u['E_accept']:.3f}  TPS={u['tps']:.1f} "
              f"({u['tps_gain_pct']:+.1f}%)  relaxes={u['relaxes_total']}", flush=True)

    print("\n(k, margin) grid:", flush=True)
    print("   k  margin  TPS_uni  TPS_gate  ret%(k)  ret%(best)  risky_drop  safe_keep", flush=True)
    for row in r["margin_grid"]:
        print(f"  {row['k']:>2}  {row['margin']:>5.2f}  {row['tps_uniform']:>7.1f}  "
              f"{row['tps_gated']:>7.1f}  {row['retention_pct_same_k']:>6.0f}  "
              f"{row['retention_pct_vs_best_uniform']:>8.0f}  "
              f"{row['uniform_relaxes_risky_low_margin']:>9}  "
              f"{row['uniform_relaxes_safe_high_margin']:>8}", flush=True)

    if r["recommended_margin_gated"]:
        g = r["recommended_margin_gated"]
        print(f"\nMARGIN DOMINATES: k={g['k']} margin={g['margin']:.2f} -> "
              f"TPS={g['tps_gated']:.1f} ({g['retention_pct_same_k']:.0f}% of uniform-k gain), "
              f"removes {g['uniform_relaxes_risky_low_margin']} risky low-margin relaxes",
              flush=True)
    else:
        br = r["best_retention_row"]
        print(f"\nMARGIN DOES NOT DOMINATE: no (k, margin>0) keeps >= "
              f"{r['retention_dominate_pct']:.0f}% of uniform-k's gain. "
              f"Best retention any margin>0: k={br['k']} margin={br['margin']:.2f} "
              f"-> {br['retention_pct_same_k']:.0f}% (TPS {br['tps_gated']:.1f}). "
              f"=> ship plain uniform-k (quality cost is intrinsic, must be measured).",
              flush=True)

    sh = r["ship_recommendation"]
    print(f"\nSHIP: {sh['kind']} k={sh['k']} margin={sh['margin']:.2f} "
          f"TPS={sh['tps']:.1f} ({sh['gain_pct']:+.1f}%)   "
          f"lever_verdict={r['lever_verdict']}  margin_dominates={r['margin_dominates_uniform']}",
          flush=True)
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [],
        "margin_dominates_uniform": r["margin_dominates_uniform"],
        "ship": f"{sh['kind']}_k{sh['k']}_m{sh['margin']}",
        "primary_metric": {"name": "ship_projected_tps", "value": round(sh["tps"], 2)},
        "test_metric": {"name": "best_margin_retention_pct_same_k",
                        "value": round(r["best_retention_row"]["retention_pct_same_k"], 2)
                        if r["best_retention_row"] else 0.0},
    }
    print("\nSENPAI-RESULT: " + json.dumps(marker, separators=(",", ":")), flush=True)


def _write_markdown(path: Path, r: dict[str, Any], ks: list[int],
                    margins: list[float]) -> None:
    s = r["strict_self_consistency"]
    md = r["margin_describe"]
    L = ["# Margin-gated top-k accept projection (PR #823) — twin of #820\n"]
    L.append(f"- blocks: **{r['n_blocks']}**  vocab: {r['vocab']}  "
             f"(reuses #820 capture `0r80mau9`, NO re-serve, 0 GPU)")
    L.append(f"- oracle fit (same anchors as #820): `TPS = {r['oracle_fit']['intercept']:.2f} + "
             f"{r['oracle_fit']['slope']:.2f}*L`  (R²={r['oracle_fit']['r2']:.4f})")
    L.append(f"- **strict self-consistency**: E_accept={s['E_accept']:.4f} "
             f"(baseline {s['baseline_E_accept']}, ok={s['E_accept_matches_baseline']}), "
             f"r={s['r']:.4f} (baseline {s['baseline_r']}, ok={s['r_matches_baseline']}), "
             f"TPS@strict={s['tps_at_strict']:.1f}")
    am, r2 = md["all_rejects_margin"], md["rank2_rejects_margin"]
    fa = md["frac_rejects_at_or_above"]
    L.append(f"- **margin = dp/ap at reject positions**: median **{am['p50']:.3f}**, "
             f"mean {am['mean']:.3f}, p90 {am['p90']:.3f}, max {am['max']:.3f}  |  "
             f"at rank-2 rejects: median **{r2['p50']:.3f}**, max **{r2['max']:.3f}**")
    L.append(f"- fraction of all rejects with margin ≥ "
             + ", ".join(f"{thr}: **{fa[f'ge_{thr}']['frac']:.3f}** ({fa[f'ge_{thr}']['count']})"
                         for thr in (0.5, 0.8, 0.95)) + "\n")

    # Step 1 contrast: margin-conditioned CDF
    L.append("## Step 1 — margin-conditioned reject-rank CDF (HIGH=near-tie vs LOW)\n")
    L.append("| margin thr | subset | " + " | ".join(f"≤{k}" for k in ks) + " | n |")
    L.append("|---|---|" + "---|" * (len(ks) + 1))
    for e in r["margin_conditioned_cdf"]:
        L.append(f"| {e['margin_thr']:.2f} | high (m≥thr, near-tie) | "
                 + " | ".join(f"{e['cdf_high_margin_rejects'][f'le_{k}']:.3f}" for k in ks)
                 + f" | {e['n_rejects_high_margin']} |")
        L.append(f"| {e['margin_thr']:.2f} | low (m<thr) | "
                 + " | ".join(f"{e['cdf_low_margin_rejects'][f'le_{k}']:.3f}" for k in ks)
                 + f" | {e['n_rejects_low_margin']} |")
    L.append("")

    # the side-by-side entropy table (#820, refuted) for the explicit contrast
    L.append("### Side-by-side: entropy-conditioned CDF (#820, REFUTED — does NOT separate)\n")
    L.append("| H thr (nats) | subset | " + " | ".join(f"≤{k}" for k in ks) + " | n |")
    L.append("|---|---|" + "---|" * (len(ks) + 1))
    for e in r["entropy_conditioned_cdf"]:
        L.append(f"| {e['h_nats']:.2f} | high (H>thr) | "
                 + " | ".join(f"{e['cdf_high_entropy_rejects'][f'le_{k}']:.3f}" for k in ks)
                 + f" | {e['n_rejects_high_entropy']} |")
        L.append(f"| {e['h_nats']:.2f} | low (H≤thr) | "
                 + " | ".join(f"{e['cdf_low_entropy_rejects'][f'le_{k}']:.3f}" for k in ks)
                 + f" | {e['n_rejects_low_entropy']} |")
    L.append("")

    # margin decile table
    L.append("### P(rank≤k | margin decile) — where do the near-misses live?\n")
    L.append("| margin bucket | count | P(rank=2) | " + " | ".join(f"P(≤{k})" for k in ks) + " |")
    L.append("|---|---|---|" + "---|" * len(ks))
    for row in r["margin_decile_table"]:
        L.append(f"| [{row['lo']:.1f},{row['hi']:.1f}) | {row['count']} | {row['p_rank2']:.3f} | "
                 + " | ".join(f"{row[f'p_le_{k}']:.3f}" for k in ks) + " |")
    L.append("")

    # Step 2 uniform reference
    L.append("## Step 2 — uniform top-k (= margin 0; identical to #820)\n")
    L.append("| k | E_accept | TPS | Δ% vs ~253 | non-argmax emitted |")
    L.append("|---|---|---|---|---|")
    for u in r["projection_uniform_topk"]:
        L.append(f"| {u['k']} | {u['E_accept']:.3f} | {u['tps']:.1f} | {u['tps_gain_pct']:+.1f} | "
                 f"{u['relaxes_total']} |")
    L.append("")

    # Step 2 grid
    L.append("## Step 2 — (k, margin) grid\n")
    L.append("| k | margin | TPS_uniform | TPS_gated | Δ% vs ~253 | retain% (same k) | "
             "retain% (best uni) | risky low-m relaxes dropped | safe high-m kept |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in r["margin_grid"]:
        L.append(f"| {row['k']} | {row['margin']:.2f} | {row['tps_uniform']:.1f} | "
                 f"{row['tps_gated']:.1f} | {row['gain_pct_vs_anchor']:+.1f} | "
                 f"{row['retention_pct_same_k']:.0f} | {row['retention_pct_vs_best_uniform']:.0f} | "
                 f"{row['uniform_relaxes_risky_low_margin']} | "
                 f"{row['uniform_relaxes_safe_high_margin']} |")
    L.append("")

    # Step 2 quality proxy crosstab
    L.append(f"## Step 2 — quality proxy: uniform-k relaxes by margin × entropy "
             f"(H thr = {r['h_nats_headline']:.2f} nats, #820's headline boundary)\n")
    L.append("| k | margin thr | total relaxes | low-margin (risky, gate drops) | "
             "frac low-margin | frac low-entropy (#820) | low-m∧low-H | low-m∧high-H |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in r["quality_proxy_crosstabs"]:
        L.append(f"| {c['k']} | {c['margin_thr']:.2f} | {c['uniform_relaxes_total']} | "
                 f"{c['low_margin_total']} | {c['frac_low_margin']:.3f} | "
                 f"{c['frac_low_entropy']:.3f} | {c['low_margin_low_entropy']} | "
                 f"{c['low_margin_high_entropy']} |")
    L.append("")

    # Step 3 decision
    L.append("## Step 3 — does margin DOMINATE uniform-k?\n")
    if r["recommended_margin_gated"]:
        g = r["recommended_margin_gated"]
        L.append(f"**YES.** Recommended drop-in for stark #816: **k={g['k']}, margin={g['margin']:.2f}** "
                 f"(`rank ≤ {g['k']} AND dp/ap ≥ {g['margin']:.2f}`) → **{g['tps_gated']:.1f} TPS** "
                 f"({g['gain_pct_vs_anchor']:+.1f}% vs ~253), keeping "
                 f"**{g['retention_pct_same_k']:.0f}%** of uniform-k's gain while removing "
                 f"**{g['uniform_relaxes_risky_low_margin']}** genuine-error (low-margin) relaxes.\n")
    else:
        br = r["best_retention_row"]
        L.append(f"**NO — clean negative.** No `(k, margin>0)` keeps ≥ "
                 f"{r['retention_dominate_pct']:.0f}% of uniform-k's gain. The best any margin>0 "
                 f"achieves is **{br['retention_pct_same_k']:.0f}%** retention "
                 f"(k={br['k']}, margin={br['margin']:.2f}, TPS {br['tps_gated']:.1f}). Uniform-k's "
                 f"speed lives in the LOW-margin (genuine-disagreement) region, so a quality-safe "
                 f"margin gate cannot keep the speed.\n")
        L.append("**=> Ship plain uniform-k.** The relaxation lever's quality cost is intrinsic "
                 "(coupled to the speed) and must be MEASURED on the real quality floors, not "
                 "gated away by margin.\n")

    sh = r["ship_recommendation"]
    L.append("## Verdict\n")
    L.append(f"- relaxation-lever ceiling (best uniform, unchanged from #820): "
             f"**{r['best_uniform']['tps']:.1f} TPS** "
             f"({r['best_uniform']['tps_gain_pct']:+.1f}% vs ~253), E_accept={r['best_uniform']['E_accept']:.3f}")
    L.append(f"- **margin dominates uniform-k: {r['margin_dominates_uniform']}**")
    L.append(f"- **SHIP to stark #816: {sh['kind']}** k={sh['k']} margin={sh['margin']:.2f} "
             f"→ {sh['tps']:.1f} TPS ({sh['gain_pct']:+.1f}%)")
    L.append(f"- quality floors to protect (NOT re-evaluated here): {r['quality_floors']}")
    path.write_text("\n".join(L) + "\n")


def _maybe_log_wandb(r: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        from scripts.wandb_logging import (init_wandb_run, log_json_artifact,
                                           log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[margin] wandb logging skipped (import): {exc!r}", flush=True)
        return
    cfg = {"n_blocks": r["n_blocks"], "vocab": r["vocab"],
           "oracle_slope": r["oracle_fit"]["slope"],
           "oracle_intercept": r["oracle_fit"]["intercept"],
           "baseline_tps": r["baseline_tps"], "pr": 823,
           "reuses_capture": "0r80mau9", "h_nats_headline": r["h_nats_headline"]}
    run = init_wandb_run(job_type="projection", agent="fern",
                         name=args.wandb_name, group=args.wandb_group, config=cfg,
                         tags=["pr823", "margin-gate", "reject-rank", "topk-accept",
                               "quality-safe", "twin-of-820"])
    if run is None:
        print("[margin] wandb unavailable; JSON/markdown still written.", flush=True)
        return
    s = r["strict_self_consistency"]
    md = r["margin_describe"]
    summary: dict[str, Any] = {
        "margin_dominates_uniform": int(r["margin_dominates_uniform"]),
        "lever_verdict": r["lever_verdict"],
        "strict_E_accept": s["E_accept"], "strict_r": s["r"],
        "strict_E_accept_matches_baseline": int(s["E_accept_matches_baseline"]),
        "margin_median_all_rejects": md["all_rejects_margin"]["p50"],
        "margin_mean_all_rejects": md["all_rejects_margin"]["mean"],
        "margin_median_rank2": md["rank2_rejects_margin"]["p50"],
        "margin_max_rank2": md["rank2_rejects_margin"]["max"],
        "frac_rejects_margin_ge_0.5": md["frac_rejects_at_or_above"]["ge_0.5"]["frac"],
        "frac_rejects_margin_ge_0.8": md["frac_rejects_at_or_above"]["ge_0.8"]["frac"],
        "frac_rejects_margin_ge_0.95": md["frac_rejects_at_or_above"]["ge_0.95"]["frac"],
        "best_uniform_tps": r["best_uniform"]["tps"],
        "best_uniform_gain_pct": r["best_uniform"]["tps_gain_pct"],
        "best_uniform_k": r["best_uniform"]["k"],
        "ship_kind": r["ship_recommendation"]["kind"],
        "ship_k": r["ship_recommendation"]["k"],
        "ship_margin": r["ship_recommendation"]["margin"],
        "ship_tps": r["ship_recommendation"]["tps"],
    }
    for row in r["margin_grid"]:
        tag = f"k{row['k']}_m{str(row['margin']).replace('.', 'p')}"
        summary[f"grid_{tag}_tps"] = row["tps_gated"]
        summary[f"grid_{tag}_retention_same_k"] = row["retention_pct_same_k"]
        summary[f"grid_{tag}_risky_dropped"] = row["uniform_relaxes_risky_low_margin"]
        summary[f"grid_{tag}_safe_kept"] = row["uniform_relaxes_safe_high_margin"]
    if r["best_retention_row"]:
        br = r["best_retention_row"]
        summary["best_margin_retention_pct_same_k"] = br["retention_pct_same_k"]
        summary["best_margin_retention_k"] = br["k"]
        summary["best_margin_retention_margin"] = br["margin"]
    if r["recommended_margin_gated"]:
        g = r["recommended_margin_gated"]
        summary.update({"rec_k": g["k"], "rec_margin": g["margin"],
                        "rec_tps": g["tps_gated"],
                        "rec_retention_pct": g["retention_pct_same_k"],
                        "rec_risky_removed": g["uniform_relaxes_risky_low_margin"]})
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="reject_rank_margin_projection",
                      artifact_type="projection", data=r)
    rid = getattr(run, "id", None)
    print(f"[margin] wandb run id: {rid}", flush=True)
    try:
        run.finish()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    raise SystemExit(main())
