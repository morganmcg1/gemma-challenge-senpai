#!/usr/bin/env python3
"""PR#670 de-risk analysis: 2x2 espec/wall_tps decomposition + projection + verdict.

Reads every records.jsonl under derisk_670/ (each line is one paired-run record with
fields: arm, e_accept_exact, wall_tps), groups by arm, and computes:
  - 2x2 cell means/spreads (stock|ftv1 x topk32|topk64)
  - main effects (retrain, top_k) + interaction on espec and wall_tps
  - headline edge (ftv1@64 - stock@32) attribution into top_k + retrain + interaction
  - rescued-official-equiv projection via stark tax 0.870 (authorized #666), vs +10 bar
Optionally reads sub64/*/records.jsonl for the subsample (seed-resampling) CI on the delta.
"""
import json, glob, math, statistics as st, sys, pathlib

D = pathlib.Path("/workspace/senpai/target/research/walltps_ab/optionb_bi1_stock_int4/derisk_670")
STARK_TAX = 0.870          # stark #663 captured-rescued/un-rescued, authorized via #666
LOCKED_ANCHOR = 126.378    # locked int4_g128_lmhead official TPS
PLUS10_BAR = 136.378       # +10 fire bar (official)

def load_records(globpat):
    """All paired-run records across derisk_670 subdirs, deduped by (arm, t_start_utc).
    Carries num_prompts + seed so the population grid (128) and the subsample (64) stay
    separate even though they share arm labels (stock_topk32 / ftv1_topk64)."""
    recs = []
    seen = set()  # reused-baseline records are copied across headline/ and headline_v2/
    for f in sorted(glob.glob(str(globpat))):
        for line in pathlib.Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            arm = r.get("arm")
            if arm is None or r.get("e_accept_exact") is None:
                continue
            key = (arm, r.get("t_start_utc"))
            if key in seen:
                continue
            seen.add(key)
            recs.append({
                "arm": arm, "espec": r["e_accept_exact"], "wtps": r["wall_tps"],
                "num_prompts": r.get("num_prompts"), "seed": r.get("seed"), "file": f,
            })
    return recs

def cells_from(recs, num_prompts):
    cells = {}  # arm -> list of (espec, wall_tps, file)
    for r in recs:
        if r["num_prompts"] != num_prompts:
            continue
        cells.setdefault(r["arm"], []).append((r["espec"], r["wtps"], r["file"]))
    return cells

def agg(vals):
    xs = [v[0] for v in vals]; ws = [v[1] for v in vals]
    return {
        "n": len(xs),
        "espec_mean": st.mean(xs) if xs else float("nan"),
        "espec_sd": (st.pstdev(xs) if len(xs) > 1 else 0.0),
        "wtps_mean": st.mean(ws) if ws else float("nan"),
        "wtps_sd": (st.pstdev(ws) if len(ws) > 1 else 0.0),
        "espec_vals": [round(x, 4) for x in xs],
        "wtps_vals": [round(w, 3) for w in ws],
    }

def proj(wtps):
    resc = wtps * STARK_TAX
    return resc, resc / LOCKED_ANCHOR - 1.0, resc >= PLUS10_BAR

def main():
    recs = load_records(D / "*" / "records.jsonl")
    cells = cells_from(recs, 128)   # population 2x2 grid = full 128-prompt eval set only
    # canonical arm labels
    want = {"stock_topk32": "stock@32", "stock_topk64": "stock@64",
            "ftv1_topk32": "ftv1@32", "ftv1_topk64": "ftv1@64"}
    A = {}
    print("=== cells present ===")
    for arm in sorted(cells):
        a = agg(cells[arm]); A[arm] = a
        tag = want.get(arm, arm)
        print(f"  {tag:10s} ({arm:14s}) n={a['n']} espec={a['espec_mean']:.4f}±{a['espec_sd']:.4f} "
              f"wtps={a['wtps_mean']:.3f}±{a['wtps_sd']:.3f}  espec_vals={a['espec_vals']}")

    need = ["stock_topk32", "stock_topk64", "ftv1_topk32", "ftv1_topk64"]
    have = [k for k in need if k in A and A[k]["n"] > 0]
    if len(have) < 4:
        print(f"\n[partial] have {have}; missing {[k for k in need if k not in have]} — rerun when complete.")
        # still show headline if both endpoints present
        if "stock_topk32" in A and "ftv1_topk64" in A:
            de = A["ftv1_topk64"]["espec_mean"] - A["stock_topk32"]["espec_mean"]
            dw = A["ftv1_topk64"]["wtps_mean"] - A["stock_topk32"]["wtps_mean"]
            print(f"[headline] ftv1@64 - stock@32: dEspec={de:+.4f} dWtps={dw:+.3f} "
                  f"({dw/A['stock_topk32']['wtps_mean']*100:+.2f}%)")
        return

    s32, s64 = A["stock_topk32"], A["stock_topk64"]
    f32, f64 = A["ftv1_topk32"], A["ftv1_topk64"]

    def line(name, a):
        r, p, clr = proj(a["wtps_mean"])
        print(f"  {name:10s} espec {a['espec_mean']:.4f} | un-rescued wtps {a['wtps_mean']:.3f} "
              f"| rescued-equiv {r:.2f} ({p*100:+.2f}% vs {LOCKED_ANCHOR}) "
              f"| clears +10: {'YES' if clr else 'no'}")
    print("\n=== 2x2 grid (espec | un-rescued wall_tps | rescued-equiv = x0.870) ===")
    for nm, a in (("stock@32", s32), ("stock@64", s64), ("ftv1@32", f32), ("ftv1@64", f64)):
        line(nm, a)

    print("\n=== espec decomposition ===")
    topk_on_stock = s64["espec_mean"] - s32["espec_mean"]
    topk_on_ftv1  = f64["espec_mean"] - f32["espec_mean"]
    retrain_at_32 = f32["espec_mean"] - s32["espec_mean"]
    retrain_at_64 = f64["espec_mean"] - s64["espec_mean"]
    headline      = f64["espec_mean"] - s32["espec_mean"]
    topk_main = 0.5 * (topk_on_stock + topk_on_ftv1)
    retrain_main = 0.5 * (retrain_at_32 + retrain_at_64)
    interaction = headline - (topk_on_stock + retrain_at_32)  # path-difference
    print(f"  top_k effect  (stock 32->64): {topk_on_stock:+.4f}")
    print(f"  top_k effect  (ftv1  32->64): {topk_on_ftv1:+.4f}")
    print(f"  retrain effect (@topk32 stock->ftv1): {retrain_at_32:+.4f}")
    print(f"  retrain effect (@topk64 stock->ftv1): {retrain_at_64:+.4f}")
    print(f"  -- main effects: top_k={topk_main:+.4f}  retrain={retrain_main:+.4f}  interaction={interaction:+.4f}")
    print(f"  HEADLINE edge ftv1@64 - stock@32 espec = {headline:+.4f}")
    if abs(headline) > 1e-9:
        print(f"     attribution: top_k {topk_on_stock/headline*100:.0f}% + retrain {retrain_at_32/headline*100:.0f}% "
              f"+ interaction {interaction/headline*100:.0f}%  (stock-first path)")

    print("\n=== wall_tps decomposition (un-rescued K6) ===")
    hw = f64["wtps_mean"] - s32["wtps_mean"]
    tw = s64["wtps_mean"] - s32["wtps_mean"]
    rw = f32["wtps_mean"] - s32["wtps_mean"]
    print(f"  top_k (stock 32->64): {tw:+.3f} | retrain (@32): {rw:+.3f} | "
          f"interaction: {hw-tw-rw:+.3f} | HEADLINE: {hw:+.3f} ({hw/s32['wtps_mean']*100:+.2f}%)")

    print("\n=== verdict inputs ===")
    for nm, a in (("stock@32 (ships)", s32), ("ftv1@64 (deployed)", f64),
                  ("stock@64 (free knob)", s64)):
        r, p, clr = proj(a["wtps_mean"])
        print(f"  {nm:22s} rescued-equiv {r:.2f} -> {'CLEARS +10' if clr else 'under +10'} ({p*100:+.2f}%)")

    subsample_ci(recs)

def subsample_ci(recs):
    """Prompt-resampling CI on the headline delta (ftv1@64 - stock@32). The eval set is a
    hard 128-prompt population, so we resample 64-prompt draws via --seed {1,2,3}; each seed
    is a genuinely different draw. Stable delta across draws => not a single-seed artifact."""
    sub = {}  # seed -> arm -> (espec, wtps)
    for r in recs:
        if r["num_prompts"] != 64:
            continue
        sub.setdefault(r["seed"], {})[r["arm"]] = (r["espec"], r["wtps"])
    print("\n=== subsample resampling CI (n=64 x seeds, prompt-resampling robustness) ===")
    if not sub:
        print("  [pending] no n=64 subsample records yet — rerun after sub64_seed* complete.")
        return
    de, dw = [], []
    for s in sorted(sub):
        arms = sub[s]
        b = arms.get("stock_topk32"); c = arms.get("ftv1_topk64")
        if not (b and c):
            print(f"  seed {s}: incomplete pair {list(arms)} — skipping"); continue
        d_e = c[0] - b[0]; d_w = c[1] - b[1]
        de.append(d_e); dw.append(d_w)
        print(f"  seed {s}: stock@32 espec {b[0]:.4f} | ftv1@64 espec {c[0]:.4f} "
              f"| Δespec {d_e:+.4f} | Δwtps {d_w:+.3f}")
    if len(de) >= 2:
        m, sd = st.mean(de), st.stdev(de)
        # t-based 95% CI on the per-seed delta mean (small-n honest interval)
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(de), 2.776)
        half = tcrit * sd / math.sqrt(len(de))
        mw = st.mean(dw)
        print(f"  -- Δespec across {len(de)} resamples: mean {m:+.4f} sd {sd:.4f} "
              f"95%CI [{m-half:+.4f}, {m+half:+.4f}]  (mean Δwtps {mw:+.3f})")
        print(f"  -- single-seed-artifact check: CI {'EXCLUDES 0 -> edge robust to resampling' if (m-half) > 0 else 'includes 0 -> fragile'}")

if __name__ == "__main__":
    main()
