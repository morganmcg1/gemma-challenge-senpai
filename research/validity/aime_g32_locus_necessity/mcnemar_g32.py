#!/usr/bin/env python
"""PR #713 — paired McNemar: g32-on-locus (or a sub-locus) vs int4-N=0, same seeds.

The load-bearing recovery test. A standalone sampled Wilson CI vs the greedy-defined
0.420 bar is biased to FAIL by construction (sampled<=greedy; bar set on greedy).
Pairing each (item,seed) outcome between the g32 variant and the int4-N=0 anchor
cancels the common sampling penalty and directly answers "does the within-mandate g32
upgrade on the locus lift AIME?". ubel #650 / #659 precedent.

Reads sampled per-seed jsonl written by eval_g32.py:
  variant:  {variant_prefix}_s{seed}_aime.jsonl
  anchor:   {anchor_prefix}_s{seed}_aime.jsonl
Pools the 2x2 discordant counts across all seeds and reports per-seed + pooled acc,
paired diff + Wald CI, exact (binomial) McNemar p, and continuity-corrected chi^2.

Unlike #659 (prose-only), this LOGS the paired per-(seed,item) answer vectors AND the
McNemar stats to W&B as a Table + Artifact, so the decisive instrument is auditable.
test_metric = g32_locus_mcnemar_paired_diff.

Usage:
  mcnemar_g32.py --variant fqg32_L14-27 --anchor fqg32_N0 \
      --seeds 12345 23456 34567 45678 56789 [--wandb-group ...] [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def load_correct(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        out[str(r["id"])] = 1 if r.get("correct") else 0
    return out


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    """Exact binomial (sign) test on discordant pairs; p = 0.5 under H0."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="variant cell prefix, e.g. fqg32_L14-27")
    ap.add_argument("--anchor", default="fqg32_N0", help="anchor cell prefix (int4-N=0)")
    ap.add_argument("--kind", default="aime")
    ap.add_argument("--seeds", nargs="+", default=["12345", "23456", "34567", "45678", "56789"])
    ap.add_argument("--pr", type=int, default=713)
    ap.add_argument("--wandb-group", default="aime-g32-locus-necessity-fern")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    a = b = c = d = 0  # both-correct / variant-only / anchor-only / both-wrong
    per_seed_delta = []
    rows = []  # paired per-(seed,item) vectors for the W&B table
    print(f"{'seed':>7} {'paired':>6} {'var':>6} {'anc':>6} {'delta':>7}  (var-only/anc-only)")
    for s in args.seeds:
        vmap = load_correct(RES / f"{args.variant}_s{s}_{args.kind}.jsonl")
        amap = load_correct(RES / f"{args.anchor}_s{s}_{args.kind}.jsonl")
        common = sorted(set(vmap) & set(amap))
        if not common:
            print(f"{s:>7}  no overlap (var={len(vmap)} anc={len(amap)})")
            continue
        sa = sb = sc = sd = 0
        for iid in common:
            x, y = vmap[iid], amap[iid]
            if x and y:
                sa += 1; disc = "both_correct"
            elif x and not y:
                sb += 1; disc = "variant_only"
            elif (not x) and y:
                sc += 1; disc = "anchor_only"
            else:
                sd += 1; disc = "both_wrong"
            rows.append([s, iid, x, y, disc])
        np_ = len(common)
        v_acc = (sa + sb) / np_
        a_acc = (sa + sc) / np_
        print(f"{s:>7} {np_:>6} {v_acc:>6.3f} {a_acc:>6.3f} {v_acc - a_acc:>+7.3f}  ({sb}/{sc})")
        per_seed_delta.append(v_acc - a_acc)
        a += sa; b += sb; c += sc; d += sd

    n_pairs = a + b + c + d
    if n_pairs == 0:
        print("no paired data")
        return 1
    pv = (a + b) / n_pairs
    pa = (a + c) / n_pairs
    diff = pv - pa  # == (b - c)/n_pairs
    var = ((b + c) - (b - c) ** 2 / n_pairs) / (n_pairs ** 2)
    se = math.sqrt(max(var, 0.0))
    lo95, hi95 = diff - 1.96 * se, diff + 1.96 * se
    lo90, hi90 = diff - 1.645 * se, diff + 1.645 * se
    p_exact = exact_mcnemar_two_sided(b, c)
    chi2_cc = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    md = (sum(per_seed_delta) / len(per_seed_delta)) if per_seed_delta else 0.0
    pos = sum(1 for x in per_seed_delta if x > 0)
    lift_sig = bool(p_exact < 0.05 and diff > 0)
    ci_excl_zero = bool(lo95 > 0 or hi95 < 0)

    print("-" * 72)
    print(f"POOLED pairs={n_pairs}  2x2: both_ok={a} variant_only={b} anchor_only={c} both_wrong={d}")
    print(f"  variant({args.variant}) acc = {pv:.4f}   anchor({args.anchor}) acc = {pa:.4f}")
    print(f"  paired diff (variant - anchor) = {diff:+.4f}")
    print(f"    Wald95 [{lo95:+.4f}, {hi95:+.4f}]   Wald90 [{lo90:+.4f}, {hi90:+.4f}]")
    print(f"  discordant: variant_only(b)={b}  anchor_only(c)={c}")
    print(f"  exact McNemar (binomial, two-sided) p = {p_exact:.4f}")
    print(f"  McNemar chi^2 (cont-corrected) = {chi2_cc:.3f}  (crit_0.05=3.841)")
    print(f"  per-seed delta: mean={md:+.4f}  seeds_variant>anchor={pos}/{len(per_seed_delta)}")
    print(f"  RECOVERY_LIFT_SIGNIFICANT_0.05 = {lift_sig}")
    print(f"  diff_CI95_excludes_zero = {ci_excl_zero}")

    stats = {
        "pr": args.pr, "variant": args.variant, "anchor": args.anchor, "kind": args.kind,
        "seeds": ",".join(args.seeds), "n_pairs": n_pairs,
        "both_correct": a, "variant_only": b, "anchor_only": c, "both_wrong": d,
        "variant_acc": pv, "anchor_acc": pa,
        "g32_locus_mcnemar_paired_diff": diff,
        "paired_diff_wald95_lo": lo95, "paired_diff_wald95_hi": hi95,
        "paired_diff_wald90_lo": lo90, "paired_diff_wald90_hi": hi90,
        "mcnemar_exact_p": p_exact, "mcnemar_chi2_cc": chi2_cc,
        "per_seed_mean_delta": md, "seeds_variant_gt_anchor": pos,
        "n_seeds": len(per_seed_delta),
        "recovery_lift_significant_0p05": lift_sig,
        "diff_ci95_excludes_zero": ci_excl_zero,
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
    }
    (RES / f"mcnemar_{args.variant}_vs_{args.anchor}.json").write_text(json.dumps(stats, indent=2))

    if not args.no_wandb:
        try:
            import wandb
        except ImportError:
            print("[wandb] not available — skipping artifact log", flush=True)
            return 0
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=f"fern/mcnemar-{args.variant}-vs-{args.anchor}",
            config={**stats, "wandb_group": args.wandb_group,
                    "instrument": "paired_mcnemar_sampled", "substrate": "bf16_fakequant_inmemory"},
            reinit=True,
        )
        table = wandb.Table(columns=["seed", "item_id", "variant_correct", "anchor_correct", "discordant"])
        for r in rows:
            table.add_data(*r)
        run.log({"mcnemar/paired_vectors": table})
        for k, v in stats.items():
            if isinstance(v, (int, float, bool)):
                run.summary[k] = v
        art = wandb.Artifact(f"mcnemar_paired_{args.variant}_vs_{args.anchor}", type="paired_eval")
        with art.new_file("paired_rows.jsonl", mode="w") as fh:
            for r in rows:
                fh.write(json.dumps({"seed": r[0], "id": r[1], "variant_correct": r[2],
                                     "anchor_correct": r[3], "discordant": r[4]}) + "\n")
        with art.new_file("mcnemar_stats.json", mode="w") as fh:
            fh.write(json.dumps(stats, indent=2))
        run.log_artifact(art)
        rid = run.id
        wandb.finish()
        print(f"[wandb] logged paired McNemar table+artifact -> run {rid}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
