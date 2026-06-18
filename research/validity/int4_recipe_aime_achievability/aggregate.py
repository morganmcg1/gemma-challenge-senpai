#!/usr/bin/env python
"""Aggregate the group-size AIME bands (PR #679) and decide the verdict.

For each arm reads <arm>_session*.json (the #672 harness output), computes the
multi-session band exactly like #672:
    upper95 = mean + t(0.975, n-1) * (sample_std / sqrt(n))
and compares mean/upper95 to the 0.420 AIME bar (bf16-base 0.4833 reference).

Also computes the int4 decode memory-traffic byte delta vs g128 (scale bytes
per body weight = 2 / group_size; packed weight is 0.5 byte/weight), which sets
the expected decode-TPS cost of a finer grid.

Run: python aggregate.py [--wandb]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = 0.420
BF16_REF = 0.4833
ANCHOR_TPS = 126.378
LOCAL_TO_OFFICIAL = 0.870

# Student-t 0.975 quantiles by dof (n-1); matches #672's n=4 -> 3.182 -> 0.388.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228}

# body weight element counts (from the official 343-module set; total int4 body
# params) -> used for scale-byte traffic. Derived from the g128 PPL-scan:
# g128 scale_elems = 31_037_440 (one bf16 scale per 128-elem group) => body has
# 31_037_440 * 128 = 3_972_792_320 quantized body weight elements.
BODY_W_ELEMS = 31_037_440 * 128


def band(accs: list[float]) -> dict:
    n = len(accs)
    mean = sum(accs) / n
    if n >= 2:
        var = sum((a - mean) ** 2 for a in accs) / (n - 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
        t = T975.get(n - 1, 1.96)
        upper95 = mean + t * se
        lower95 = mean - t * se
    else:
        sd = se = 0.0
        upper95 = lower95 = mean
        t = float("nan")
    return {
        "n_sessions": n, "accs": accs, "mean": mean,
        "min": min(accs), "max": max(accs), "std": sd, "se": se,
        "t975": t, "upper95": upper95, "lower95": lower95,
        # For a quality FLOOR (must be ABOVE the bar) the honest decisions are:
        #   confidently_clears: even the pessimistic edge beats the bar (lower95 > bar)
        #   clears_bar_mean   : the central estimate beats the bar (mean > bar)
        #   confidently_fails : even the optimistic edge is below the bar (upper95 < bar)  <- #672's g128 verdict
        # upper95 > bar alone is NOT "clears"; it only means "not a confident fail".
        "clears_bar_mean": mean > BAR,
        "confidently_clears": lower95 > BAR,
        "confidently_fails": upper95 < BAR,
        "clears_bar_upper95": upper95 > BAR,
    }


def load_arm(arm: str) -> dict | None:
    files = sorted(glob.glob(str(HERE / f"{arm}_session*.json")))
    if not files:
        return None
    accs, fails, ntok = [], [], []
    for fp in files:
        d = json.load(open(fp))
        accs.append(d["maj_k_accuracy"])
        fails.append(d.get("extract_fail_rate", 0.0))
        ntok.append(d.get("mean_completion_tokens", d.get("mean_output_tokens", 0)))
    b = band(accs)
    b["files"] = [os.path.basename(f) for f in files]
    b["extract_fail_rate_max"] = max(fails)
    # build provenance (rel_err) if present
    meta_map = {"g128": "/workspace/gemma_build/int4_g128_lmhead/_build_meta.json",
                "g64": "/workspace/gemma_build/int4_g64body_lmhead/_build_meta.json",
                "g32": "/workspace/gemma_build/int4_g32body_lmhead/_build_meta.json"}
    mp = meta_map.get(arm)
    if mp and Path(mp).exists():
        b["build_meta"] = json.load(open(mp))
    return b


def byte_traffic(group_size: int) -> dict:
    """Per-body-weight decode bytes and total body scale MB at this group_size."""
    packed_bpw = 0.5                     # 4-bit packed weight
    scale_bpw = 2.0 / group_size         # one bf16 scale per group element
    total_bpw = packed_bpw + scale_bpw
    scale_mb = BODY_W_ELEMS * scale_bpw / 1e6
    return {"group_size": group_size, "packed_bpw": packed_bpw,
            "scale_bpw": scale_bpw, "total_bpw": total_bpw, "scale_mb": round(scale_mb, 2)}


def contrast(a: dict, b: dict) -> dict:
    """Two-sample (Welch) contrast of session-accuracy means, a - b.

    Tells us whether a finer grid genuinely lifts AIME over the g128 control,
    rather than reading one noisy arm's wide t-interval in isolation. With the
    0.828 cross-session non-determinism floor a single arm's band is wide; the
    paired-arm delta is the cleaner signal of a real recipe effect.
    """
    delta = a["mean"] - b["mean"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    t = delta / se if se > 0 else float("nan")
    na, nb = a["n_sessions"], b["n_sessions"]
    if a["se"] > 0 and b["se"] > 0 and na > 1 and nb > 1:
        num = (a["se"] ** 2 + b["se"] ** 2) ** 2
        den = (a["se"] ** 4) / (na - 1) + (b["se"] ** 4) / (nb - 1)
        dof = num / den
    else:
        dof = float("nan")
    return {"delta": delta, "se_delta": se, "t": t, "dof": dof,
            "lift_over_2se": delta > 2 * se}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    arms = {}
    for a in ["g128", "g64", "g32"]:
        r = load_arm(a)
        if r:
            arms[a] = r

    # byte traffic table vs g128
    g128_bpw = byte_traffic(128)["total_bpw"]
    traffic = {a: byte_traffic(gs) for a, gs in [("g128", 128), ("g64", 64), ("g32", 32)]}
    for a, t in traffic.items():
        t["bpw_delta_vs_g128"] = t["total_bpw"] - g128_bpw
        t["bpw_pct_vs_g128"] = 100.0 * (t["total_bpw"] / g128_bpw - 1.0)

    # The verdict question is whether a RECIPE CHANGE clears AIME, so the
    # candidates are the FINER grids (g64, g32); g128 is the shipped control we
    # are trying to beat, not a candidate. "Clears" = the CENTRAL estimate beats
    # the floor (mean > bar). upper95 > bar is only "not a confident fail" and
    # over-fires on a noisy arm (g128 here: mean 0.350 fails, but a lucky 0.433
    # session pushes its upper95 over the bar) -- so we do NOT use it to decide.
    FINER = ["g64", "g32"]
    finer = {a: arms[a] for a in FINER if a in arms}
    central_clear = {a: r for a, r in finer.items() if r["clears_bar_mean"]}
    if central_clear:
        best_arm = max(central_clear, key=lambda a: central_clear[a]["mean"])
    elif finer:
        best_arm = max(finer, key=lambda a: finer[a]["mean"])
    elif arms:
        best_arm = max(arms, key=lambda a: arms[a]["mean"])
    else:
        best_arm = None

    # PR-contract scalar: best recipe's upper95 (reported as defined). The
    # decision below leads with the mean, not this.
    int4_recipe_aime_best = arms[best_arm]["upper95"] if best_arm else float("nan")
    quality_clears = bool(central_clear)  # a finer recipe's central estimate beats the bar

    # finer-vs-g128 two-sample contrasts (is the lift real above the session floor?)
    contrasts = {}
    if "g128" in arms:
        for a in FINER:
            if a in arms:
                contrasts[a] = contrast(arms[a], arms["g128"])

    # QUALITY leg only. INT4_RECIPE_CLEARS_AIME additionally requires speed >
    # anchor (measured separately by _speed_ab.py); INT4_RECIPE_AIME_BOUND needs
    # NO finer recipe to clear. The speed leg is folded into the final verdict in
    # the PR report.
    if not quality_clears:
        verdict = "INT4_RECIPE_AIME_BOUND"
    else:
        verdict = "INT4_RECIPE_CLEARS_AIME(quality-leg; speed pending)"

    best_mean = arms[best_arm]["mean"] if best_arm else float("nan")
    best_low95 = arms[best_arm]["lower95"] if best_arm else float("nan")
    summary = {
        "bar": BAR, "bf16_ref": BF16_REF, "anchor_tps": ANCHOR_TPS,
        "arms": arms, "traffic": traffic, "contrasts_vs_g128": contrasts,
        "best_arm": best_arm, "int4_recipe_aime_best": int4_recipe_aime_best,
        "best_mean": best_mean, "best_lower95": best_low95,
        "quality_clears_central": quality_clears,
        "best_confidently_clears": bool(best_arm and arms[best_arm]["confidently_clears"]),
        "verdict_provisional": verdict,
    }
    out = HERE / "band_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    def status(r: dict) -> str:
        # vs the 0.420 quality FLOOR: CLEAR* = confident, clear = central only,
        # FAIL = confident fail (#672-style), fail = central fail / straddle.
        if r["confidently_clears"]:
            return "CLEAR*"
        if r["confidently_fails"]:
            return "FAIL"
        return "clear" if r["clears_bar_mean"] else "fail"

    print("=" * 84)
    print(f"AIME bar={BAR}  bf16-base ref={BF16_REF}  anchor_tps={ANCHOR_TPS}")
    print("-" * 84)
    print(f"{'arm':6} {'n':>2} {'accs':28} {'mean':>7} {'lo95':>6} {'up95':>6} "
          f"{'vs-bar':>7} {'bpw%':>7} {'rel_err':>8}")
    for a in ["g128", "g64", "g32"]:
        if a not in arms:
            print(f"{a:6} -- (no sessions yet)")
            continue
        r = arms[a]
        accs = ",".join(f"{x:.3f}" for x in r["accs"])
        rel = r.get("build_meta", {}).get("body_rel_err_mean")
        rel_s = f"{rel:.5f}" if rel is not None else "shipped"
        print(f"{a:6} {r['n_sessions']:>2} {accs:28} {r['mean']:>7.4f} "
              f"{r['lower95']:>6.3f} {r['upper95']:>6.3f} {status(r):>7} "
              f"{traffic[a]['bpw_pct_vs_g128']:>+6.1f}% {rel_s:>8}")
    print("-" * 84)
    for a, c in contrasts.items():
        print(f"contrast {a}-g128: delta={c['delta']:+.4f}  se={c['se_delta']:.4f}  "
              f"t={c['t']:.2f}  lift>2se={c['lift_over_2se']}")
    print(f"best_arm={best_arm}  mean={best_mean:.4f}  lower95={best_low95:.4f}  "
          f"int4_recipe_aime_best(up95)={int4_recipe_aime_best:.4f}")
    print(f"quality_clears_central={quality_clears}  "
          f"best_confidently_clears={summary['best_confidently_clears']}")
    print(f"PROVISIONAL VERDICT (quality leg): {verdict}")
    print(f"[wrote] {out}")

    if args.wandb:
        import wandb
        run = wandb.init(
            project="gemma-challenge-senpai",
            name="ubel/int4-recipe-aime-achievability",
            group="int4-recipe-aime-achievability-ubel",
            config={"analysis_only": 1, "official_tps": 0, "fires": 0,
                    "bar": BAR, "bf16_ref": BF16_REF, "anchor_tps": ANCHOR_TPS,
                    "protocol": "int4-AR greedy AIME @12288 conc=16 60q (2024+2025-I+2025-II), fresh-process band"},
        )
        flat = {"analysis_only": 1, "official_tps": 0, "fires": 0,
                "int4_recipe_aime_best": int4_recipe_aime_best,
                "best_arm": best_arm, "best_mean": best_mean, "best_lower95": best_low95,
                "quality_clears_central": int(quality_clears),
                "best_confidently_clears": int(summary["best_confidently_clears"]),
                "verdict": verdict}
        for a, r in arms.items():
            for key in ["mean", "min", "max", "lower95", "upper95", "std", "se", "n_sessions"]:
                flat[f"aime_{a}_{key}"] = r[key]
            flat[f"aime_{a}_confidently_fails"] = int(r["confidently_fails"])
            flat[f"bpw_pct_{a}_vs_g128"] = traffic[a]["bpw_pct_vs_g128"]
            if "build_meta" in r:
                flat[f"body_rel_err_mean_{a}"] = r["build_meta"]["body_rel_err_mean"]
        for a, c in contrasts.items():
            flat[f"aime_{a}_minus_g128_delta"] = c["delta"]
            flat[f"aime_{a}_minus_g128_t"] = c["t"]
        run.summary.update(flat)
        art = wandb.Artifact("int4_recipe_aime_band", type="analysis")
        art.add_file(str(out))
        run.log_artifact(art)
        print(f"[wandb] run {run.id}")
        run.finish()


if __name__ == "__main__":
    main()
