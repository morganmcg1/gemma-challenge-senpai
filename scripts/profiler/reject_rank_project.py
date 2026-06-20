#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Offline reject-rank + entropy CDF and top-k E_accept/TPS projection (PR #820).

Reads the per-draft-position JSONL that ``reject_rank_capture.py`` wrote on the
deployed int4head greedy verify path and produces, with NO GPU, the viability
gate for relaxing the strict ``draft == verifier-argmax`` accept criterion:

  (2) reject-rank CDF -- fraction of strict rejects at rank <= {2,3,5,8,16}; and
      that CDF split by HIGH vs LOW verifier entropy across a swept threshold.
  (3) projected realized E_accept(k) and TPS(k) for k in {2,3,5,8,16} under BOTH
        (A) uniform top-k     : accept draft if rank(draft) <= k everywhere; and
        (B) entropy-gated top-k: relax to top-k ONLY where H > H_thresh, keep
            strict argmax (the shipped criterion) where the verifier is confident
            (FLy-inspired, arXiv:2511.22972).
      Projection replays each REAL verify block under the relaxed criterion. This
      is exactly faithful WITHIN a block: vLLM computes all K draft-position logits
      in one teacher-forced pass, so the rank/entropy at depth d is fixed
      regardless of the accept decision at depths < d. The only approximation is
      cross-block context drift -- the SAME assumption stark's oracle TPS curve
      makes. TPS is read off the oracle's near-linear acceptance-length->TPS fit.
  (3b) quality proxy. At each swept H_thresh we set the gating threshold EQUAL to a
      risk boundary at the same H, so:
        * uniform(k) makes some relaxed-accepts (non-argmax tokens emitted) at
          H <= H_thresh -- the RISKY ones (verifier confident, a non-argmax accept
          is likely a real error); and some at H > H_thresh -- the safe ones.
        * gated(k,H_thresh) makes the SAME high-entropy relaxes but refuses every
          low-entropy one -> zero risky relaxes by construction.
      "Gain retention" = (gated_gain)/(uniform_gain): if the rank<=k rescues are
      concentrated at HIGH entropy, gating keeps almost all the speed while
      eliminating the risky relaxes -> B dominates A. If uniform's gain comes from
      LOW-entropy relaxes, gating costs real speed -> the quality risk is intrinsic.
  (4) go/no-go: does the best projected variant clear a meaningful gain over the
      ~253 baseline, and does (B) dominate (A)?

Block-replay accounting. Per decode step the engine emits ``accepted_len + 1``
tokens (the accepted draft prefix plus one bonus/correction token). The mean over
blocks is E_accept; the strict-criterion mean must reproduce the baseline 3.379
(self-consistency, asserted on load).

    python -m scripts.profiler.reject_rank_project \
        --in-dir research/reject_rank_entropy/int4head \
        --wandb-group bi0-reject-rank-entropy

Oracle anchors / quality floors default to the PR #820 baseline block; override on
the CLI. Entropy thresholds sweep in NATS (this 262k-vocab model's verifier
entropy is tiny -- median ~0.3 nats -- so FLy's normalized theta=0.3 maps to
~3.74 nats and is effectively inert here; we report normalized h alongside).
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

# --- PR #820 baseline constants (from the PR body; sourced from the assignment) ---
DEFAULT_ORACLE_POINTS = "4.355:309.96,5.20:362.37,6.10:421.29,7.0:471.09"
BASELINE_E_ACCEPT = 3.379          # strict draft==argmax mean acceptance length
BASELINE_R = 0.397                 # r = accepted_drafts / K = (E_accept-1)/K
BASELINE_TPS = 253.0               # ~local int4head greedy TPS (W&B 7ntx4nrn)
DEFAULT_K_LIST = "2,3,5,8,16"
# Entropy thresholds in NATS (data-relevant for this peaked 262k-vocab verifier).
# 3.743 nats == FLy normalized theta=0.3 on V=262144 (log V = 12.477), kept as a
# reference point to show it is inert on this model.
DEFAULT_H_NATS_LIST = "0.1,0.25,0.5,1.0,2.0,3.743"
FLY_THETA = 0.3
GEMMA_VOCAB = 262144               # gemma-4-E4B; fallback if probe meta is absent
RETENTION_DOMINATE_PCT = 90.0      # gated must keep >= this % of uniform's gain
DEFAULT_GO_PCT = 10.0              # best projected gain to justify a kernel patch
QUALITY_FLOORS = {
    "ppl_max": 2.42, "aime_min": 0.090, "modality": "128/128",
    "mmlu_pro_min": 0.572, "gpqa_min": 0.471, "gsm8k_min": 0.807,
}


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def _load_blocks(shards: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in shards:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n = rec["n"]
                if not (len(rec["acc"]) == len(rec["rk"]) == len(rec["H"]) == n):
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


# --------------------------------------------------------------------------- #
# Oracle TPS curve
# --------------------------------------------------------------------------- #
def _parse_oracle_points(spec: str) -> list[tuple[float, float]]:
    pts = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        l_str, t_str = tok.split(":")
        pts.append((float(l_str), float(t_str)))
    return sorted(pts)


def _linfit(points: list[tuple[float, float]]) -> dict[str, float]:
    """OLS TPS = a + b*L over the oracle anchors, with R^2."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in points)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {"intercept": a, "slope": b, "r2": r2,
            "l_min": min(p[0] for p in points), "l_max": max(p[0] for p in points)}


def _tps_of_L(L: float, fit: dict[str, float]) -> float:
    return fit["intercept"] + fit["slope"] * L


# --------------------------------------------------------------------------- #
# Block replay
# --------------------------------------------------------------------------- #
def _accepted_len_uniform(rk: list[int], k: int) -> int:
    for d, r in enumerate(rk):
        if r > k:
            return d
    return len(rk)


def _accepted_len_gated(rk: list[int], H: list[float], acc: list[int],
                        k: int, h_nats: float) -> int:
    """accept(d) = strict-argmax (acc==1) OR (H>h_nats AND rank<=k)."""
    for d in range(len(rk)):
        if not ((acc[d] == 1) or (H[d] > h_nats and rk[d] <= k)):
            return d
    return len(rk)


def _strict_stats(blocks: list[dict[str, Any]]) -> dict[str, float]:
    """Reproduce the shipped strict criterion (self-consistency).

    Baseline r~0.397 is accepted-drafts/K (mean accepted draft tokens per block
    over the K proposed), NOT the per-position argmax-match rate: positions AFTER a
    block's first reject still get logits in the teacher-forced pass and often
    match argmax, so per-position match is much higher and is only a descriptive
    upper bound. We derive r = (E_accept-1)/K and report per-position match too.
    """
    n_blocks = len(blocks)
    n_pos = n_match = accepted_drafts = 0
    sum_len = 0.0
    max_n = 0
    for b in blocks:
        acc = b["acc"]
        n_pos += b["n"]
        n_match += sum(acc)
        max_n = max(max_n, b["n"])
        fd = b["n"]
        for d, a in enumerate(acc):
            if a == 0:
                fd = d
                break
        accepted_drafts += fd
        sum_len += fd + 1
    K = max_n or 1
    e_accept = (sum_len / n_blocks) if n_blocks else float("nan")
    return {
        "n_blocks": n_blocks, "n_draft_positions": n_pos, "K_inferred": K,
        "per_position_match_rate": (n_match / n_pos) if n_pos else float("nan"),
        "accepted_drafts_mean": (accepted_drafts / n_blocks) if n_blocks else float("nan"),
        "r": ((e_accept - 1.0) / K) if n_blocks else float("nan"),
        "E_accept": e_accept,
    }


def _project_uniform(blocks: list[dict[str, Any]], k: int, fit: dict[str, float],
                     baseline_tps: float) -> dict[str, Any]:
    """Uniform top-k E_accept/TPS + total non-argmax tokens emitted (relaxes)."""
    sum_len = 0.0
    relaxes = 0
    for b in blocks:
        rk, acc = b["rk"], b["acc"]
        fd = _accepted_len_uniform(rk, k)
        sum_len += fd + 1
        relaxes += sum(1 for d in range(fd) if acc[d] == 0)
    n_blocks = len(blocks)
    e_acc = sum_len / n_blocks if n_blocks else float("nan")
    tps = _tps_of_L(e_acc, fit)
    return {"k": k, "E_accept": e_acc, "tps": tps,
            "tps_gain_abs": tps - baseline_tps,
            "tps_gain_pct": 100.0 * (tps - baseline_tps) / baseline_tps,
            "relaxes_total": relaxes}


def _uniform_relax_split(blocks: list[dict[str, Any]], k: int, h_nats: float) -> tuple[int, int]:
    """(risky, safe) = uniform(k) relaxed-accepts at H<=h_nats vs H>h_nats."""
    risky = safe = 0
    for b in blocks:
        rk, H, acc = b["rk"], b["H"], b["acc"]
        fd = _accepted_len_uniform(rk, k)
        for d in range(fd):
            if acc[d] == 0:
                if H[d] > h_nats:
                    safe += 1
                else:
                    risky += 1
    return risky, safe


def _project_gated(blocks: list[dict[str, Any]], k: int, h_nats: float,
                   fit: dict[str, float], baseline_tps: float) -> dict[str, Any]:
    """Entropy-gated top-k E_accept/TPS. low-entropy relaxes are 0 by construction;
    counted to PROVE that on the realized data."""
    sum_len = 0.0
    relaxes_low = relaxes_high = 0
    for b in blocks:
        rk, H, acc = b["rk"], b["H"], b["acc"]
        fd = _accepted_len_gated(rk, H, acc, k, h_nats)
        sum_len += fd + 1
        for d in range(fd):
            if acc[d] == 0:
                if H[d] > h_nats:
                    relaxes_high += 1
                else:
                    relaxes_low += 1
    n_blocks = len(blocks)
    e_acc = sum_len / n_blocks if n_blocks else float("nan")
    tps = _tps_of_L(e_acc, fit)
    return {"k": k, "h_nats": h_nats, "E_accept": e_acc, "tps": tps,
            "tps_gain_abs": tps - baseline_tps,
            "tps_gain_pct": 100.0 * (tps - baseline_tps) / baseline_tps,
            "relaxes_low_entropy": relaxes_low, "relaxes_high_entropy": relaxes_high}


# --------------------------------------------------------------------------- #
# CDFs / entropy description
# --------------------------------------------------------------------------- #
def _reject_rank_cdf(blocks: list[dict[str, Any]], ks: list[int],
                     h_nats: float | None = None) -> dict[str, Any]:
    """Fraction of STRICT rejects (acc==0) with rank <= k -- over all rejects and
    over the first-reject-per-block (truncation-relevant). If ``h_nats`` given,
    also split by HIGH vs LOW entropy."""
    all_ranks: list[int] = []
    first_ranks: list[int] = []
    hi_ranks: list[int] = []
    lo_ranks: list[int] = []
    for b in blocks:
        acc, rk, H = b["acc"], b["rk"], b["H"]
        seen_first = False
        for d in range(b["n"]):
            if acc[d] == 0:
                all_ranks.append(rk[d])
                if h_nats is not None:
                    (hi_ranks if H[d] > h_nats else lo_ranks).append(rk[d])
                if not seen_first:
                    first_ranks.append(rk[d])
                    seen_first = True

    def _cdf(ranks: list[int]) -> dict[str, float]:
        n = len(ranks)
        return {f"le_{k}": (sum(1 for r in ranks if r <= k) / n if n else float("nan"))
                for k in ks}

    out: dict[str, Any] = {
        "n_rejects_all": len(all_ranks),
        "n_rejects_first_per_block": len(first_ranks),
        "cdf_all_rejects": _cdf(all_ranks),
        "cdf_first_reject_per_block": _cdf(first_ranks),
    }
    if h_nats is not None:
        out.update({"h_nats": h_nats,
                    "n_rejects_high_entropy": len(hi_ranks),
                    "n_rejects_low_entropy": len(lo_ranks),
                    "cdf_high_entropy_rejects": _cdf(hi_ranks),
                    "cdf_low_entropy_rejects": _cdf(lo_ranks)})
    return out


def _quantiles(xs: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    s = sorted(xs)
    out = {f"p{int(q*100)}": s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))] for q in qs}
    out["mean"] = sum(s) / len(s)
    out["max"] = s[-1]
    return out


def _entropy_describe(blocks: list[dict[str, Any]], log_v: float | None) -> dict[str, Any]:
    all_H: list[float] = []
    rej_H: list[float] = []
    for b in blocks:
        for d in range(b["n"]):
            all_H.append(b["H"][d])
            if b["acc"][d] == 0:
                rej_H.append(b["H"][d])
    out = {"all_positions_nats": _quantiles(all_H), "reject_positions_nats": _quantiles(rej_H)}
    if log_v:
        out["log_vocab"] = log_v
        out["fly_theta_nats"] = FLY_THETA * log_v
    return out


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
    ap.add_argument("--h-nats-list", type=str, default=DEFAULT_H_NATS_LIST,
                    help="verifier-entropy thresholds in NATS to sweep")
    ap.add_argument("--vocab", type=int, default=None,
                    help="override vocab (else probe .meta.json, else gemma 262144)")
    ap.add_argument("--oracle-points", type=str, default=DEFAULT_ORACLE_POINTS)
    ap.add_argument("--baseline-tps", type=float, default=BASELINE_TPS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", type=str, default="bi0-reject-rank-entropy")
    ap.add_argument("--wandb-name", type=str, default="fern/reject-rank-entropy-project")
    args = ap.parse_args(argv)

    baseline_tps = args.baseline_tps
    in_dir = args.in_dir.resolve()
    out_dir = (args.out_dir or in_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = str(in_dir / args.records_base)
    shards = sorted(s for s in glob.glob(base + ".*") if not s.endswith(".meta.json"))
    if not shards:
        print(f"[project] ERROR: no record shards at {base}.* — run capture first", flush=True)
        return 1
    blocks = _load_blocks(shards)
    if not blocks:
        print("[project] ERROR: shards present but no parseable blocks", flush=True)
        return 1

    vocab = args.vocab or _read_vocab(shards) or GEMMA_VOCAB
    vocab_src = ("cli" if args.vocab else
                 ("meta" if _read_vocab(shards) else "fallback_gemma_262144"))
    log_v = math.log(vocab)

    ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    h_nats_list = [float(x) for x in args.h_nats_list.split(",") if x.strip()]
    fit = _linfit(_parse_oracle_points(args.oracle_points))

    # ---- self-consistency ----
    strict = _strict_stats(blocks)
    strict_tps = _tps_of_L(strict["E_accept"], fit)
    e_acc_ok = abs(strict["E_accept"] - BASELINE_E_ACCEPT) <= 0.15
    r_ok = abs(strict["r"] - BASELINE_R) <= 0.03

    # ---- CDFs ----
    cdf = _reject_rank_cdf(blocks, ks)
    ent = _entropy_describe(blocks, log_v)
    entropy_cdf = [{"h_nats": h, "h_norm": h / log_v, **_reject_rank_cdf(blocks, ks, h_nats=h)}
                   for h in h_nats_list]

    # ---- uniform projection (E_accept/TPS depends only on k) ----
    uniform = [_project_uniform(blocks, k, fit, baseline_tps) for k in ks]
    uni_by_k = {u["k"]: u for u in uniform}
    best_uniform = max(uniform, key=lambda r: r["tps"])

    # ---- A-vs-B comparison grid: gating threshold == risk boundary == h_nats ----
    grid: list[dict[str, Any]] = []
    for k in ks:
        u = uni_by_k[k]
        for h in h_nats_list:
            g = _project_gated(blocks, k, h, fit, baseline_tps)
            risky, safe = _uniform_relax_split(blocks, k, h)
            gain_u = u["tps_gain_abs"]
            gain_g = g["tps_gain_abs"]
            grid.append({
                "k": k, "h_nats": h, "h_norm": h / log_v,
                "tps_uniform": u["tps"], "tps_gated": g["tps"],
                "gain_uniform": gain_u, "gain_gated": gain_g,
                "E_accept_uniform": u["E_accept"], "E_accept_gated": g["E_accept"],
                "retention_pct": (100.0 * gain_g / gain_u) if gain_u > 0 else float("nan"),
                "uniform_relaxes_risky_lowH": risky,   # gating refuses these
                "uniform_relaxes_safe_highH": safe,
                "gated_relaxes_lowH": g["relaxes_low_entropy"],   # must be 0
                "gated_relaxes_highH": g["relaxes_high_entropy"],
            })

    # FLy literal point (theta=0.3) -> ~3.743 nats; show it is inert here.
    fly_nats = FLY_THETA * log_v
    fly_rows = [row for row in grid if abs(row["h_nats"] - fly_nats) < 0.05]
    fly_point = max(fly_rows, key=lambda r: r["tps_gated"]) if fly_rows else None

    # Recommended gated: most risk eliminated while keeping >=RETENTION_DOMINATE_PCT
    # of uniform's gain. Among qualifying rows pick the fastest shippable config.
    qualifying = [row for row in grid
                  if row["uniform_relaxes_risky_lowH"] > 0
                  and row["retention_pct"] >= RETENTION_DOMINATE_PCT]
    recommended_gated = max(qualifying, key=lambda r: r["tps_gated"]) if qualifying else None

    # ---- verdict ----
    best_overall = best_uniform  # uniform is the relaxation-lever ceiling
    best_gain_pct = best_overall["tps_gain_pct"]
    verdict = "GO" if best_gain_pct >= DEFAULT_GO_PCT else ("AMBER" if best_gain_pct >= 3.0 else "NO-GO")
    # B dominates A: a gated config keeps >=90% of the BEST uniform gain while
    # eliminating >0 risky low-entropy relaxes.
    b_dominates = bool(recommended_gated
                       and recommended_gated["gain_gated"] >= 0.90 * best_uniform["tps_gain_abs"]
                       and recommended_gated["uniform_relaxes_risky_lowH"] > 0)

    result = {
        "in_dir": str(in_dir), "shards": shards, "n_blocks": len(blocks),
        "vocab": vocab, "vocab_source": vocab_src, "log_vocab": log_v,
        "oracle_fit": fit, "baseline_tps": baseline_tps,
        "strict_self_consistency": {
            **strict, "tps_at_strict": strict_tps,
            "E_accept_matches_baseline": e_acc_ok, "r_matches_baseline": r_ok,
            "baseline_E_accept": BASELINE_E_ACCEPT, "baseline_r": BASELINE_R,
        },
        "entropy_describe": ent,
        "reject_rank_cdf": cdf,
        "entropy_conditioned_cdf": entropy_cdf,
        "projection_uniform_topk": uniform,
        "comparison_grid": grid,
        "best_uniform": best_uniform,
        "fly_theta_point": fly_point, "fly_theta_nats": fly_nats,
        "recommended_gated": recommended_gated,
        "verdict": verdict, "best_overall": best_overall, "B_dominates_A": b_dominates,
        "fly_theta": FLY_THETA, "quality_floors": QUALITY_FLOORS,
    }
    (out_dir / "reject_rank_projection.json").write_text(json.dumps(result, indent=2))
    _write_markdown(out_dir / "reject_rank_projection.md", result, ks, h_nats_list)
    _print_summary(result, ks)
    if args.wandb:
        _maybe_log_wandb(result)

    print(f"\n[project] artifacts -> {out_dir}", flush=True)
    if not (e_acc_ok and r_ok):
        print("[project] WARNING: strict self-consistency vs baseline is OFF — "
              "probe may not have seen the true greedy path (or sample too small).", flush=True)
    return 0


def _print_summary(r: dict[str, Any], ks: list[int]) -> None:
    s = r["strict_self_consistency"]
    print("\n========== REJECT-RANK PROJECTION (PR #820) ==========", flush=True)
    print(f"blocks={r['n_blocks']}  vocab={r['vocab']} ({r['vocab_source']})  "
          f"oracle TPS={r['oracle_fit']['intercept']:.2f}+{r['oracle_fit']['slope']:.2f}*L "
          f"(R2={r['oracle_fit']['r2']:.4f})", flush=True)
    print(f"strict E_accept={s['E_accept']:.4f} (base {s['baseline_E_accept']}, ok={s['E_accept_matches_baseline']})  "
          f"r={s['r']:.4f} (base {s['baseline_r']}, ok={s['r_matches_baseline']})  "
          f"per_pos_match={s['per_position_match_rate']:.4f}  TPS@strict={s['tps_at_strict']:.1f}", flush=True)
    eH = r["entropy_describe"]
    print(f"verifier entropy (all pos) median={eH['all_positions_nats']['p50']:.3f} "
          f"p90={eH['all_positions_nats']['p90']:.3f} max={eH['all_positions_nats']['max']:.3f} nats  "
          f"| FLy theta=0.3 -> {r['fly_theta_nats']:.2f} nats", flush=True)
    c = r["reject_rank_cdf"]["cdf_all_rejects"]
    print("reject-rank CDF (all rejects): " + "  ".join(f"<={k}:{c[f'le_{k}']:.3f}" for k in ks), flush=True)
    print("\nUNIFORM top-k:", flush=True)
    for u in r["projection_uniform_topk"]:
        print(f"  k={u['k']:>2}  E_accept={u['E_accept']:.3f}  TPS={u['tps']:.1f} "
              f"({u['tps_gain_pct']:+.1f}%)  relaxes={u['relaxes_total']}", flush=True)
    print("\nA-vs-B grid (gate threshold == risk boundary):", flush=True)
    print("   k  H_nats  h_norm  TPS_A   TPS_B   retain%  risky(A)  safe(A)", flush=True)
    for row in r["comparison_grid"]:
        print(f"  {row['k']:>2}  {row['h_nats']:>5.2f}  {row['h_norm']:>5.3f}  "
              f"{row['tps_uniform']:>6.1f}  {row['tps_gated']:>6.1f}  {row['retention_pct']:>6.0f}  "
              f"{row['uniform_relaxes_risky_lowH']:>7}  {row['uniform_relaxes_safe_highH']:>6}", flush=True)
    if r["fly_theta_point"]:
        f = r["fly_theta_point"]
        print(f"\nFLy literal theta=0.3 ({r['fly_theta_nats']:.2f} nats): k={f['k']} "
              f"TPS_gated={f['tps_gated']:.1f} retain={f['retention_pct']:.0f}% "
              f"(near-strict => inert on this 262k-vocab model)", flush=True)
    if r["recommended_gated"]:
        g = r["recommended_gated"]
        print(f"\nRECOMMENDED gated: k={g['k']} H_thresh={g['h_nats']:.2f} nats "
              f"(h={g['h_norm']:.3f}) -> TPS={g['tps_gated']:.1f} "
              f"(+{g['gain_gated']:.1f}, {g['retention_pct']:.0f}% of uniform), "
              f"eliminates {g['uniform_relaxes_risky_lowH']} risky low-H relaxes", flush=True)
    else:
        print("\nRECOMMENDED gated: NONE clears the retention bar — uniform's gain "
              "lives in the low-entropy (risky) region.", flush=True)
    print(f"\nVERDICT: {r['verdict']}  best={r['best_overall']['tps']:.1f} TPS "
          f"({r['best_overall']['tps_gain_pct']:+.1f}%)  B_dominates_A={r['B_dominates_A']}", flush=True)
    primary = round(r["best_overall"]["tps"], 2)
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [], "decision": r["verdict"], "b_dominates_a": r["B_dominates_A"],
        "primary_metric": {"name": "projected_best_tps", "value": primary},
        "test_metric": {"name": "projected_best_tps_gain_pct",
                        "value": round(r["best_overall"]["tps_gain_pct"], 2)},
    }
    print("\nSENPAI-RESULT: " + json.dumps(marker, separators=(",", ":")), flush=True)


def _write_markdown(path: Path, r: dict[str, Any], ks: list[int], h_nats_list: list[float]) -> None:
    s = r["strict_self_consistency"]
    L = ["# Reject-rank + entropy CDF / top-k projection (PR #820)\n"]
    L.append(f"- blocks: **{r['n_blocks']}**  vocab: {r['vocab']} ({r['vocab_source']})")
    L.append(f"- oracle fit: `TPS = {r['oracle_fit']['intercept']:.2f} + "
             f"{r['oracle_fit']['slope']:.2f}*L`  (R²={r['oracle_fit']['r2']:.4f}, "
             f"anchors L∈[{r['oracle_fit']['l_min']},{r['oracle_fit']['l_max']}])")
    L.append(f"- **strict self-consistency**: E_accept={s['E_accept']:.4f} "
             f"(baseline {s['baseline_E_accept']}, ok={s['E_accept_matches_baseline']}), "
             f"r=(E_accept-1)/K={s['r']:.4f} (baseline {s['baseline_r']}, ok={s['r_matches_baseline']}), "
             f"per-position match={s['per_position_match_rate']:.4f}, TPS@strict={s['tps_at_strict']:.1f}")
    eH = r["entropy_describe"]["all_positions_nats"]
    L.append(f"- verifier entropy (all positions, nats): median={eH['p50']:.3f}, p90={eH['p90']:.3f}, "
             f"p99={eH['p99']:.3f}, max={eH['max']:.3f} — FLy θ=0.3 → **{r['fly_theta_nats']:.2f} nats** "
             f"(inert on this peaked 262k-vocab verifier)\n")

    L.append("## Reject-rank CDF (fraction of strict rejects at rank ≤ k)\n")
    L.append("| subset | " + " | ".join(f"≤{k}" for k in ks) + " | n |")
    L.append("|---|" + "---|" * (len(ks) + 1))
    ca = r["reject_rank_cdf"]
    L.append("| all rejects | " + " | ".join(f"{ca['cdf_all_rejects'][f'le_{k}']:.3f}" for k in ks)
             + f" | {ca['n_rejects_all']} |")
    L.append("| first reject/block | " + " | ".join(
        f"{ca['cdf_first_reject_per_block'][f'le_{k}']:.3f}" for k in ks)
        + f" | {ca['n_rejects_first_per_block']} |")
    L.append("")

    L.append("## Entropy-conditioned reject-rank CDF (HIGH vs LOW entropy)\n")
    L.append("| H_thresh (nats) | h=H/logV | subset | " + " | ".join(f"≤{k}" for k in ks) + " | n |")
    L.append("|---|---|---|" + "---|" * (len(ks) + 1))
    for e in r["entropy_conditioned_cdf"]:
        L.append(f"| {e['h_nats']:.2f} | {e['h_norm']:.3f} | high (H>thr) | "
                 + " | ".join(f"{e['cdf_high_entropy_rejects'][f'le_{k}']:.3f}" for k in ks)
                 + f" | {e['n_rejects_high_entropy']} |")
        L.append(f"| {e['h_nats']:.2f} | {e['h_norm']:.3f} | low (H≤thr) | "
                 + " | ".join(f"{e['cdf_low_entropy_rejects'][f'le_{k}']:.3f}" for k in ks)
                 + f" | {e['n_rejects_low_entropy']} |")
    L.append("")

    L.append("## (A) Uniform top-k projection\n")
    L.append("| k | E_accept | TPS | Δ% | non-argmax tokens emitted |")
    L.append("|---|---|---|---|---|")
    for u in r["projection_uniform_topk"]:
        L.append(f"| {u['k']} | {u['E_accept']:.3f} | {u['tps']:.1f} | {u['tps_gain_pct']:+.1f} | "
                 f"{u['relaxes_total']} |")
    L.append("")

    L.append("## (A vs B) grid — gating threshold == risk boundary\n")
    L.append("| k | H_thresh | h | TPS_A | TPS_B | retain% | risky relaxes A drops | safe relaxes |")
    L.append("|---|---|---|---|---|---|---|---|")
    for row in r["comparison_grid"]:
        L.append(f"| {row['k']} | {row['h_nats']:.2f} | {row['h_norm']:.3f} | "
                 f"{row['tps_uniform']:.1f} | {row['tps_gated']:.1f} | {row['retention_pct']:.0f} | "
                 f"{row['uniform_relaxes_risky_lowH']} | {row['uniform_relaxes_safe_highH']} |")
    L.append("")

    if r["recommended_gated"]:
        g = r["recommended_gated"]
        L.append("## Recommended entropy-gated point\n")
        L.append(f"- **k={g['k']}, H_thresh={g['h_nats']:.2f} nats (h={g['h_norm']:.3f})** → "
                 f"**{g['tps_gated']:.1f} TPS** (+{g['gain_gated']:.1f}, "
                 f"{g['retention_pct']:.0f}% of uniform's gain), eliminating "
                 f"**{g['uniform_relaxes_risky_lowH']}** risky low-entropy relaxed-accepts.\n")
    else:
        L.append("## Recommended entropy-gated point: NONE\n")
        L.append("No threshold keeps ≥90% of uniform's gain while eliminating risky "
                 "low-entropy relaxes — uniform's speed lives in the low-entropy region.\n")

    L.append(f"## Verdict: **{r['verdict']}**  (B dominates A: **{r['B_dominates_A']}**)\n")
    bo = r["best_overall"]
    L.append(f"- relaxation-lever ceiling (best uniform): **{bo['tps']:.1f} TPS** "
             f"({bo['tps_gain_pct']:+.1f}% vs ~{r['baseline_tps']:.0f}), E_accept={bo['E_accept']:.3f}")
    if r["fly_theta_point"]:
        f = r["fly_theta_point"]
        L.append(f"- FLy literal θ=0.3 ({r['fly_theta_nats']:.2f} nats): TPS_gated={f['tps_gated']:.1f} "
                 f"(retain {f['retention_pct']:.0f}%) — inert here, threshold must be recalibrated to nats")
    L.append(f"- quality floors to protect (NOT re-evaluated here): {r['quality_floors']}")
    path.write_text("\n".join(L) + "\n")


def _maybe_log_wandb(r: dict[str, Any]) -> None:
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                           log_json_artifact)
    except Exception as exc:  # noqa: BLE001
        print(f"[project] wandb logging skipped (import): {exc!r}", flush=True)
        return
    cfg = {"n_blocks": r["n_blocks"], "vocab": r["vocab"],
           "oracle_slope": r["oracle_fit"]["slope"], "oracle_intercept": r["oracle_fit"]["intercept"],
           "baseline_tps": r["baseline_tps"], "fly_theta": r["fly_theta"]}
    run = init_wandb_run(job_type="projection", agent="fern",
                         name="fern/reject-rank-entropy-project",
                         group="bi0-reject-rank-entropy", config=cfg,
                         tags=["pr820", "reject-rank", "entropy-gated", "fly", "topk-accept"])
    if run is None:
        print("[project] wandb unavailable; JSON/markdown still written.", flush=True)
        return
    s = r["strict_self_consistency"]
    summary: dict[str, Any] = {
        "verdict": r["verdict"], "B_dominates_A": int(r["B_dominates_A"]),
        "strict_E_accept": s["E_accept"], "strict_r": s["r"],
        "strict_E_accept_matches_baseline": int(s["E_accept_matches_baseline"]),
        "strict_per_position_match_rate": s["per_position_match_rate"],
        "best_uniform_tps": r["best_uniform"]["tps"],
        "best_uniform_gain_pct": r["best_uniform"]["tps_gain_pct"],
        "best_uniform_k": r["best_uniform"]["k"],
    }
    # reject-rank CDF (all rejects) — queryable without the artifact.
    for kk, v in r["reject_rank_cdf"]["cdf_all_rejects"].items():
        summary[f"reject_rank_cdf_{kk}"] = v
    # per-k uniform projection (TPS / E_accept / gain / non-argmax emissions).
    for u in r["projection_uniform_topk"]:
        summary[f"uniform_k{u['k']}_tps"] = u["tps"]
        summary[f"uniform_k{u['k']}_e_accept"] = u["E_accept"]
        summary[f"uniform_k{u['k']}_gain_pct"] = u["tps_gain_pct"]
        summary[f"uniform_k{u['k']}_relaxes"] = u["relaxes_total"]
    if r["recommended_gated"]:
        g = r["recommended_gated"]
        summary.update({"rec_gated_k": g["k"], "rec_gated_h_nats": g["h_nats"],
                        "rec_gated_tps": g["tps_gated"], "rec_gated_retention_pct": g["retention_pct"],
                        "rec_gated_risky_eliminated": g["uniform_relaxes_risky_lowH"]})
    if r["fly_theta_point"]:
        summary["fly_theta_tps_gated"] = r["fly_theta_point"]["tps_gated"]
        summary["fly_theta_retention_pct"] = r["fly_theta_point"]["retention_pct"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="reject_rank_projection", artifact_type="projection", data=r)
    rid = getattr(run, "id", None)
    print(f"[project] wandb run id: {rid}", flush=True)
    try:
        run.finish()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    raise SystemExit(main())
