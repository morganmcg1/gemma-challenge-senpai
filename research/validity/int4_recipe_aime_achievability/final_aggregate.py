#!/usr/bin/env python
"""PR #679 FINAL aggregation: fold the three legs into one verdict + W&B run.

Legs:
  1. group-size band sweep  -> g128/g64/g32 minmax AIME bands (band_summary.json)
  2. calibration sweep       -> g128mse (speed-free probe) / g32mse AIME bands
                                (g128mse_session*.json, g32mse_session*.json)
  3. speed gate              -> g64/g32 official-equiv TPS + PPL (speed_ab_summary.json)

Verdict logic (the PR's two single-line options + the honest middle):
  * A finer-grid OR mse recipe whose central AIME mean > 0.420 AND whose
    official-equiv TPS is >= the 126.378 anchor  -> INT4_RECIPE_CLEARS_AIME
    (the speed-free mse@g128 probe is the only zero-cost path to this).
  * If SOME recipe clears AIME centrally but EVERY clearing recipe is slower
    than the anchor -> the AIME failure is recipe-RECOVERABLE on quality
    (NOT fundamental to int4) but has no speed-competitive recipe: we report
    INT4_RECIPE_AIME_BOUND(speed) and state plainly that the bound is a SPEED
    bound, not the "fundamental-to-int4 quality" bound the PR hypothesised.
  * If NO recipe clears AIME centrally -> INT4_RECIPE_AIME_BOUND (quality):
    corroborates #672, int4 W4A16 fundamentally fails AIME.

Run: python final_aggregate.py [--wandb]
ANALYSIS-ONLY, LOCAL.
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

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def band(accs: list[float]) -> dict:
    n = len(accs)
    mean = sum(accs) / n
    if n >= 2:
        sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (n - 1))
        se = sd / math.sqrt(n)
        t = T975.get(n - 1, 1.96)
        up, lo = mean + t * se, mean - t * se
    else:
        sd = se = 0.0
        up = lo = mean
        t = float("nan")
    return {"n_sessions": n, "accs": accs, "mean": mean, "min": min(accs),
            "max": max(accs), "std": sd, "se": se, "t975": t,
            "upper95": up, "lower95": lo,
            "clears_bar_mean": mean > BAR, "confidently_clears": lo > BAR,
            "confidently_fails": up < BAR}


def load_arm(arm: str) -> dict | None:
    files = sorted(glob.glob(str(HERE / f"{arm}_session*.json")))
    if not files:
        return None
    accs, fails = [], []
    for fp in files:
        d = json.load(open(fp))
        accs.append(d["maj_k_accuracy"])
        fails.append(d.get("extract_fail_rate", 0.0))
    b = band(accs)
    b["files"] = [os.path.basename(f) for f in files]
    b["extract_fail_rate_max"] = max(fails)
    return b


def contrast(a: dict, b: dict) -> dict:
    delta = a["mean"] - b["mean"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return {"delta": delta, "se_delta": se,
            "t": delta / se if se > 0 else float("nan"),
            "lift_over_2se": delta > 2 * se}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    # ---- AIME bands for every arm we have sessions for ----
    arm_order = ["g128", "g64", "g32", "g128mse", "g32mse"]
    arms = {a: load_arm(a) for a in arm_order}
    arms = {a: r for a, r in arms.items() if r}

    contrasts = {}
    if "g128" in arms:
        for a in ["g64", "g32", "g128mse", "g32mse"]:
            if a in arms:
                contrasts[a] = contrast(arms[a], arms["g128"])

    # ---- speed leg ----
    speed = {}
    sp_path = HERE / "speed_ab_summary.json"
    if sp_path.exists():
        sp = json.loads(sp_path.read_text())["summary"]
        speed["anchor_local_wall_tps"] = sp["anchor_local_wall_tps"]
        speed["anchor_ppl"] = sp.get("anchor_ppl")
        for c, v in sp["variants"].items():
            key = "g64" if "g64" in c else ("g32" if "g32" in c else c)
            speed[key] = {"official_proj_tps": v["official_proj_tps"],
                          "delta_pct": v["delta_pct_vs_anchor_local"],
                          "beats_anchor": v["beats_anchor"], "ppl": v["ppl"]}
    # mse@g128 is byte-identical to the g128 anchor (same #scales + packed
    # bytes); only scale VALUES differ -> decode traffic, hence TPS, is the
    # anchor's by construction. So its official-equiv == ANCHOR_TPS (speed-free).
    g128mse_official = ANCHOR_TPS if "g128mse" in arms else None
    # mse@g32 shares g32's byte layout -> g32's measured speed.
    g32mse_official = speed.get("g32", {}).get("official_proj_tps") if "g32mse" in arms else None

    # ---- per-recipe (cleared?, speed, official-equiv) ----
    def official_of(arm: str):
        return {"g64": speed.get("g64", {}).get("official_proj_tps"),
                "g32": speed.get("g32", {}).get("official_proj_tps"),
                "g128mse": g128mse_official,
                "g32mse": g32mse_official,
                "g128": ANCHOR_TPS}.get(arm)

    recipes = {}
    for a, r in arms.items():
        off = official_of(a)
        cleared = r["clears_bar_mean"]
        # speed-OK if official-equiv >= anchor (mse@g128 ties => speed-neutral win)
        speed_ok = (off is not None) and (off >= ANCHOR_TPS - 1e-6)
        recipes[a] = {"aime_mean": r["mean"], "aime_upper95": r["upper95"],
                      "aime_lower95": r["lower95"], "clears_aime": cleared,
                      "official_equiv_tps": off, "speed_ok": speed_ok,
                      "quality_safe_fast": bool(cleared and speed_ok)}

    # candidate recipes = the non-shipped ones (g128 minmax is the control)
    candidates = {a: v for a, v in recipes.items() if a != "g128"}
    clearing = {a: v for a, v in candidates.items() if v["clears_aime"]}
    winners = {a: v for a, v in candidates.items() if v["quality_safe_fast"]}

    # best recipe for the PR scalar: prefer a quality-safe-fast winner (highest
    # AIME mean); else the highest-AIME clearing recipe; else highest AIME.
    pool = winners or clearing or candidates
    best_arm = max(pool, key=lambda a: pool[a]["aime_mean"]) if pool else None
    int4_recipe_aime_best = arms[best_arm]["upper95"] if best_arm else float("nan")
    int4_recipe_tps_official_equiv = recipes[best_arm]["official_equiv_tps"] if best_arm else 0

    # ---- verdict ----
    if winners:
        verdict = "INT4_RECIPE_CLEARS_AIME"
        verdict_note = (f"{best_arm} clears AIME (mean {arms[best_arm]['mean']:.3f}) "
                        f"at official-equiv {int4_recipe_tps_official_equiv:.2f} "
                        f">= anchor {ANCHOR_TPS} -> quality-safe int4 body.")
    elif clearing:
        verdict = "INT4_RECIPE_AIME_BOUND(speed)"
        verdict_note = ("A finer-grid recipe clears the AIME quality bar (so the "
                        "failure is NOT fundamental to int4), but every AIME-clearing "
                        "recipe is slower than the 126.378 anchor -> no speed-competitive "
                        "quality-safe int4 body. The bound is a SPEED bound, not a quality one.")
    else:
        verdict = "INT4_RECIPE_AIME_BOUND"
        verdict_note = ("No in-recipe change clears the AIME bar centrally -> int4 W4A16 "
                        "fundamentally fails AIME; corroborates #672.")

    # Calibration leg = the speed-free mse@g128 probe (the only zero-cost path to
    # CLEARS_AIME). Arm B (mse@g32) is deliberately skipped: g32 is already
    # speed-disqualified (119.22 < 126.378 anchor), so an mse-on-g32 recipe has
    # ZERO verdict leverage; the chain predicts mse ~no-ops the QAT-native g32 grid
    # (rel_err already ~0.0667 uniform); and disk at 99% forbids a 2nd ~10GB build.
    # The leg is complete once the decisive g128mse arm has its >=3-session band.
    g128mse_sessions = arms.get("g128mse", {}).get("n_sessions", 0)
    calib_pending = g128mse_sessions < 3
    calib_armB_g32mse_skipped = {
        "skipped": True,
        "reason": ("g32 already fails the speed gate (119.22 < 126.378 anchor) so no "
                   "g32-grid recipe can reach CLEARS_AIME; mse on the QAT-native g32 grid "
                   "is predicted ~no-op; disk at 99% forbids a 2nd ~10GB build."),
    }

    summary = {
        "bar": BAR, "bf16_ref": BF16_REF, "anchor_tps": ANCHOR_TPS,
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "arms": arms, "contrasts_vs_g128": contrasts, "speed": speed,
        "recipes": recipes,
        "best_arm": best_arm,
        "int4_recipe_aime_best": int4_recipe_aime_best,
        "int4_recipe_tps_official_equiv": int4_recipe_tps_official_equiv,
        "quality_safe_fast_winners": list(winners),
        "aime_clearing_recipes": list(clearing),
        "verdict": verdict, "verdict_note": verdict_note,
        "calib_pending": calib_pending,
        "calib_armB_g32mse_skipped": calib_armB_g32mse_skipped,
    }
    out = HERE / "final_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # ---- print table ----
    print("=" * 92)
    print(f"PR #679 FINAL  AIME bar={BAR}  bf16-base={BF16_REF}  anchor_tps={ANCHOR_TPS}")
    print("-" * 92)
    print(f"{'recipe':9} {'n':>2} {'accs':30} {'mean':>6} {'lo95':>6} {'up95':>6} "
          f"{'clrAIME':>7} {'offTPS':>7} {'spdOK':>5} {'WIN':>4}")
    for a in arm_order:
        if a not in arms:
            continue
        r, rc = arms[a], recipes[a]
        accs = ",".join(f"{x:.3f}" for x in r["accs"])
        off = rc["official_equiv_tps"]
        off_s = f"{off:.2f}" if off is not None else "  n/a"
        print(f"{a:9} {r['n_sessions']:>2} {accs:30} {r['mean']:>6.3f} "
              f"{r['lower95']:>6.3f} {r['upper95']:>6.3f} {str(rc['clears_aime']):>7} "
              f"{off_s:>7} {str(rc['speed_ok']):>5} {str(rc['quality_safe_fast']):>4}")
    print("-" * 92)
    for a, c in contrasts.items():
        print(f"contrast {a}-g128: delta={c['delta']:+.4f} se={c['se_delta']:.4f} "
              f"t={c['t']:.2f} lift>2se={c['lift_over_2se']}")
    print("-" * 92)
    print(f"best_recipe={best_arm}  int4_recipe_aime_best(up95)={int4_recipe_aime_best:.4f}  "
          f"int4_recipe_tps_official_equiv={int4_recipe_tps_official_equiv}")
    print(f"clearing={list(clearing)}  winners={list(winners)}  calib_pending={calib_pending}")
    print(f"VERDICT: {verdict}")
    print(f"  {verdict_note}")
    print(f"[wrote] {out}")

    if args.wandb:
        import wandb
        run = wandb.init(
            project="gemma-challenge-senpai",
            name="ubel/int4-recipe-aime-achievability",
            group="int4-recipe-aime-achievability-ubel",
            config={"analysis_only": 1, "official_tps": 0, "fires": 0,
                    "bar": BAR, "bf16_ref": BF16_REF, "anchor_tps": ANCHOR_TPS,
                    "protocol": "int4-AR greedy AIME @12288 conc=16 60q fresh-process band; "
                                "speed: served wall-TPS in-session ratio x official 126.378, np=8 ol=512 reps=3"},
        )
        flat = {"analysis_only": 1, "official_tps": 0, "fires": 0,
                "int4_recipe_aime_best": int4_recipe_aime_best,
                "int4_recipe_tps_official_equiv": int4_recipe_tps_official_equiv,
                "best_recipe": best_arm, "verdict": verdict,
                "n_clearing_recipes": len(clearing), "n_quality_safe_fast": len(winners)}
        for a, r in arms.items():
            for k in ["mean", "min", "max", "lower95", "upper95", "std", "n_sessions"]:
                flat[f"aime_{a}_{k}"] = r[k]
            flat[f"aime_{a}_clears_bar_mean"] = int(r["clears_bar_mean"])
            flat[f"aime_{a}_confidently_fails"] = int(r["confidently_fails"])
            off = recipes[a]["official_equiv_tps"]
            if off is not None:
                flat[f"official_equiv_tps_{a}"] = off
            flat[f"quality_safe_fast_{a}"] = int(recipes[a]["quality_safe_fast"])
        for a, c in contrasts.items():
            flat[f"aime_{a}_minus_g128_delta"] = c["delta"]
            flat[f"aime_{a}_minus_g128_t"] = c["t"]
        for k in ["g64", "g32"]:
            if k in speed and isinstance(speed[k], dict):
                flat[f"ppl_{k}"] = speed[k]["ppl"]
        if speed.get("anchor_ppl") is not None:
            flat["ppl_g128_anchor"] = speed["anchor_ppl"]
        run.summary.update(flat)
        art = wandb.Artifact("int4_recipe_aime_final", type="analysis")
        art.add_file(str(out))
        for extra in ["band_summary.json", "speed_ab_summary.json"]:
            if (HERE / extra).exists():
                art.add_file(str(HERE / extra))
        run.log_artifact(art)
        print(f"[wandb] run {run.id}")
        run.finish()


if __name__ == "__main__":
    main()
