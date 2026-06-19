#!/usr/bin/env python3
"""PR #726 aggregator: paired AIME verdict on the provenance-clean bf16 substrate.

For each arm in {full_g32, int8_locus, bf16_locus, int8_full} read 5 per-seed AIME
JSONs (aime_eval.py, k=1 sampled) and key per item by (seed, problem_id). Then:

  PRIMARY  int8_locus_minus_g32_clean = pooled(int8_locus) - pooled(full_g32),
           paired McNemar over the 300 shared (seed,id) items.
           REPLICATE iff delta>0 AND McNemar p<0.05 (mirrors fern #659 int8>g32,
           p=0.0248). Else INVERT.

  SECOND   int8_minus_bf16_locus = pooled(int8_locus) - pooled(bf16_locus).
           bf16 ⊇ int8 information, so delta>=0 is the NOISE signature (int8 cannot
           beat its own information ceiling on a clean substrate).

  GATE     full_g32 must reproduce #702 nqk9izab 0.3867 (substrate sanity). Pass iff
           0.3867 lies inside full_g32's pooled Wilson95 CI.

McNemar uses the EXACT two-sided binomial on the discordant pairs (b,c) -- robust at
small discordant counts -- plus a continuity-corrected chi2(df=1) for reference.
Pure stdlib (no scipy). analysis_only=1, official_tps=0, no_hf_job=1, fires=0. LOCAL.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["full_g32", "int8_locus", "bf16_locus", "int8_full"]
G32_ANCHOR = 0.3867          # #702 nqk9izab full_g32 pooled (116/300)
ALPHA = 0.05


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def mcnemar(b: int, c: int) -> dict:
    """Paired test on discordant counts b (X>Y) and c (Y>X)."""
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_exact": 1.0, "p_chi2_cc": 1.0, "chi2_cc": 0.0}
    k = min(b, c)
    p_exact = min(1.0, 2.0 * binom_cdf(k, nd, 0.5))
    chi2 = (abs(b - c) - 1) ** 2 / nd          # continuity-corrected, df=1
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))  # chi2(df=1) survival = erfc(sqrt(x/2))
    return {"b": b, "c": c, "n_discordant": nd, "p_exact": p_exact,
            "p_chi2_cc": p_chi2, "chi2_cc": chi2}


def load_arm(arm: str) -> dict | None:
    files = sorted(glob.glob(str(HERE / "results" / f"{arm}_seed*.json")))
    if not files:
        return None
    items: dict[tuple, bool] = {}           # (seed, id) -> maj_correct
    per_seed_acc, seeds_meta = [], []
    n_correct_pool = n_total_pool = empty = extract_fail = 0
    for fp in files:
        d = json.load(open(fp))
        seed = d.get("sampling", {}).get("seed")
        if seed is None:                     # fall back to filename ..._seed<N>.json
            seed = int(Path(fp).stem.split("seed")[-1])
        pp = d.get("per_problem", [])
        nc = 0
        for r in pp:
            key = (seed, r["id"])
            ok = bool(r["maj_correct"])
            items[key] = ok
            nc += int(ok)
            for a in r.get("answers", []):
                if a is None:
                    extract_fail += 1
            for t in (r.get("texts") or []):
                if not str(t).strip():
                    empty += 1
        n = len(pp)
        per_seed_acc.append(nc / n if n else float("nan"))
        n_correct_pool += nc
        n_total_pool += n
        seeds_meta.append({"file": Path(fp).name, "seed": seed,
                           "acc": nc / n if n else None, "n_correct": nc, "n": n,
                           "wall_s": d.get("wall_s")})
    p = n_correct_pool / n_total_pool if n_total_pool else float("nan")
    lo, hi = wilson(n_correct_pool, n_total_pool)
    return {
        "arm": arm, "n_seeds": len(files), "items": items,
        "pooled_accuracy": p, "pooled_n_correct": n_correct_pool, "pooled_n": n_total_pool,
        "wilson95_lo": lo, "wilson95_hi": hi,
        "per_seed_acc": per_seed_acc,
        "per_seed_acc_mean": sum(per_seed_acc) / len(per_seed_acc) if per_seed_acc else float("nan"),
        "per_seed_acc_min": min(per_seed_acc) if per_seed_acc else float("nan"),
        "per_seed_acc_max": max(per_seed_acc) if per_seed_acc else float("nan"),
        "pooled_empty": empty, "extract_fail": extract_fail, "seeds": seeds_meta,
    }


def paired(x: dict, y: dict) -> dict:
    """Paired comparison X - Y over shared (seed,id) items. Returns delta, contingency, McNemar."""
    keys = sorted(set(x["items"]) & set(y["items"]))
    a = b = c = dd = 0
    for k in keys:
        xi, yi = x["items"][k], y["items"][k]
        if xi and yi: a += 1
        elif xi and not yi: b += 1
        elif (not xi) and yi: c += 1
        else: dd += 1
    n = len(keys)
    corr_x = a + b
    corr_y = a + c
    delta = (corr_x - corr_y) / n if n else float("nan")
    mc = mcnemar(b, c)
    return {"n_paired": n, "x_correct": corr_x, "y_correct": corr_y,
            "delta": delta, "contingency": {"both": a, "x_only": b, "y_only": c, "neither": dd},
            **mc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "pathb_summary.json"))
    args = ap.parse_args()

    arms = {a: load_arm(a) for a in ARMS}
    arms = {a: r for a, r in arms.items() if r}

    out = {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
           "g32_anchor": G32_ANCHOR,
           "protocol": "AIME #31 gate basis: years=2024,2025-I,2025-II n=60, k=1 sampled "
                       "(T=1.0 top_p=0.95 top_k=64), max_tokens=12288, min_tokens=8, 5-seed "
                       "pooled (300); paired McNemar on (seed,id); built from qat_unq bf16 master.",
           "locus": "L14-27 (118 body Linear); lm_head int4-g128 locked; rest int4-g32.",
           "arms": {}}

    for a, r in arms.items():
        out["arms"][a] = {k: v for k, v in r.items() if k != "items"}

    # --- substrate gate: full_g32 reproduces 0.3867? ---
    g32 = arms.get("full_g32")
    gate = None
    if g32:
        lo, hi = g32["wilson95_lo"], g32["wilson95_hi"]
        gate = {"full_g32_pooled": g32["pooled_accuracy"], "anchor": G32_ANCHOR,
                "wilson95": [lo, hi], "anchor_in_ci": bool(lo <= G32_ANCHOR <= hi),
                "abs_delta": abs(g32["pooled_accuracy"] - G32_ANCHOR)}
        gate["pass"] = bool(gate["anchor_in_ci"] or gate["abs_delta"] <= 0.05)
    out["substrate_gate"] = gate

    # --- PRIMARY: int8_locus - full_g32 ---
    primary = None
    if "int8_locus" in arms and "full_g32" in arms:
        pr = paired(arms["int8_locus"], arms["full_g32"])
        replicate = bool(pr["delta"] > 0 and pr["p_exact"] < ALPHA)
        primary = {**pr, "metric": "int8_locus_minus_g32_clean",
                   "replicate": replicate,
                   "verdict": "REPLICATE" if replicate else
                              ("INVERT" if pr["delta"] <= 0 else "NULL_NS")}
    out["primary_int8_locus_minus_g32"] = primary

    # --- SECONDARY ceiling: int8_locus - bf16_locus ---
    ceiling = None
    if "int8_locus" in arms and "bf16_locus" in arms:
        cp = paired(arms["int8_locus"], arms["bf16_locus"])
        ceiling = {**cp, "metric": "int8_minus_bf16_locus",
                   "noise_signature": bool(cp["delta"] >= 0)}
    out["secondary_int8_minus_bf16_locus"] = ceiling

    # --- EXTEND context: int8_full - full_g32 (headroom) ---
    if "int8_full" in arms and "full_g32" in arms:
        out["extend_int8_full_minus_g32"] = {**paired(arms["int8_full"], arms["full_g32"]),
                                             "metric": "int8_full_minus_g32_clean"}

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    # ---- print ----
    print("=" * 96)
    print(f"PR #726  Path-B clean substrate (qat_unq)  AIME sampled n=300  anchor full_g32={G32_ANCHOR}")
    print("-" * 96)
    print(f"{'arm':12} {'seeds':>5} {'pooled':>7} {'n_corr':>9} {'wilson95':>18} {'perseed[min,mean,max]':>26}")
    for a in ARMS:
        r = arms.get(a)
        if not r:
            continue
        ci = f"[{r['wilson95_lo']:.4f},{r['wilson95_hi']:.4f}]"
        ps = f"[{r['per_seed_acc_min']:.3f},{r['per_seed_acc_mean']:.3f},{r['per_seed_acc_max']:.3f}]"
        print(f"{a:12} {r['n_seeds']:>5} {r['pooled_accuracy']:>7.4f} "
              f"{r['pooled_n_correct']:>4}/{r['pooled_n']:<4} {ci:>18} {ps:>26}")
    print("-" * 96)
    if gate:
        print(f"SUBSTRATE GATE: full_g32 {gate['full_g32_pooled']:.4f} vs anchor {G32_ANCHOR} "
              f"(|Δ|={gate['abs_delta']:.4f}, anchor∈CI={gate['anchor_in_ci']}) -> PASS={gate['pass']}")
    if primary:
        print(f"PRIMARY  int8_locus - full_g32  delta={primary['delta']:+.4f}  "
              f"(int8={primary['x_correct']}/{primary['n_paired']}, g32={primary['y_correct']}/{primary['n_paired']})  "
              f"discordant b={primary['b']} c={primary['c']}  McNemar p_exact={primary['p_exact']:.4f} "
              f"p_chi2cc={primary['p_chi2_cc']:.4f}  ->  {primary['verdict']}  (replicate={int(primary['replicate'])})")
    if ceiling:
        print(f"CEILING  int8_locus - bf16_locus  delta={ceiling['delta']:+.4f}  "
              f"discordant b={ceiling['b']} c={ceiling['c']}  McNemar p_exact={ceiling['p_exact']:.4f}  "
              f"noise_signature(delta>=0)={ceiling['noise_signature']}")
    if "extend_int8_full_minus_g32" in out:
        e = out["extend_int8_full_minus_g32"]
        print(f"EXTEND   int8_full - full_g32  delta={e['delta']:+.4f}  McNemar p_exact={e['p_exact']:.4f}")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
