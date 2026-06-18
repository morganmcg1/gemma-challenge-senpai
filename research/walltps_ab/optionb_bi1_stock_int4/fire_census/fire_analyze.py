"""PR #648 analysis: served recompute-fire census across ALL served positions.

Reads per_prompt_margins_k{K}.jsonl (one row/prompt: margins[per-position top1-top2
gap], completion_token_ids, sha_ok). For each K computes:

 (1) served_fire_frac = positions with gap < TAU(0.5) / total positions, with a
     PROMPT-LEVEL bootstrap 95% CI (positions within a prompt are correlated, so a
     naive binomial CI understates width; we resample the 128 prompts).
 (2) per-K table + K-independence read.
 (3) clustering: per-prompt fire-count distribution + mean inter-fire gap.
 (4) pre/post-divergence decomposition vs the M=1 AR ref (ar_ref_bi1):
        pre-div positions share AR context => the teacher-forced-equivalent rate
        (built-in cross-check of stark's 7.80% TF); post-div positions are the
        off-AR served trajectory (the NEW quantity). Amplification iff post >> pre.
 (5) tax cross-check: back out stark's implied per-fire overhead ratio r from his
        endpoints (Option-B base 172.74 -> projected 139.20 @ his 7.80% fire), then
        apply r to the MEASURED served_fire_frac to get the cross-check wall_tps.

analysis_only. No HF Job, no generation here (pure offline read of the replay).
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "ksweep"
AR_REF = KS / "ar_ref_bi1" / "decode_outputs.jsonl"
OUT = HERE / "fire_census_result.json"

TAU = 0.5                 # stark #636 recompute-flag threshold (nat)
STARK_TF_FIRE = 0.0780    # his teacher-forced fire fraction
OPTIONB_BASE_TPS = 172.74 # land #632 no-recompute Option-B K=5 local wall_tps
STARK_PROJ_TPS = 139.20   # stark #636 projected wall_tps on his TF cost model
LOCKED_TPS = 126.378      # locked int4_g128_lmhead served file (the >baseline bar)
N_BOOT = 20000
SEED = 648


def load_ar_first_div():
    ar = {}
    with AR_REF.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                ar[d["id"]] = d["completion_token_ids"]
    return ar


def first_divergence(served_ids, ar_ids):
    n = min(len(served_ids), len(ar_ids))
    for i in range(n):
        if served_ids[i] != ar_ids[i]:
            return i
    return n  # identical up to compared length -> never diverges


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def census_one_k(k, ar):
    path = HERE / f"per_prompt_margins_k{k}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    per_prompt = []  # (n_pos, n_fire, n_pre_pos, n_pre_fire, n_post_pos, n_post_fire, fire_positions, sha_ok)
    fires_per_prompt = []
    inter_fire_gaps = []
    for r in rows:
        margins = r["margins"]
        served_ids = r["completion_token_ids"]
        fdiv = first_divergence(served_ids, ar.get(r["id"], []))
        fire_pos = [i for i, g in enumerate(margins) if g is not None and g < TAU]
        n_pos = len(margins)
        n_fire = len(fire_pos)
        n_pre_pos = min(fdiv, n_pos)
        n_post_pos = n_pos - n_pre_pos
        n_pre_fire = sum(1 for i in fire_pos if i < fdiv)
        n_post_fire = n_fire - n_pre_fire
        per_prompt.append({
            "id": r["id"], "n_pos": n_pos, "n_fire": n_fire, "fdiv": fdiv,
            "n_pre_pos": n_pre_pos, "n_pre_fire": n_pre_fire,
            "n_post_pos": n_post_pos, "n_post_fire": n_post_fire,
            "sha_ok": r["sha_ok"],
        })
        fires_per_prompt.append(n_fire)
        for a, b in zip(fire_pos, fire_pos[1:]):
            inter_fire_gaps.append(b - a)

    tot_pos = sum(p["n_pos"] for p in per_prompt)
    tot_fire = sum(p["n_fire"] for p in per_prompt)
    pre_pos = sum(p["n_pre_pos"] for p in per_prompt)
    pre_fire = sum(p["n_pre_fire"] for p in per_prompt)
    post_pos = sum(p["n_post_pos"] for p in per_prompt)
    post_fire = sum(p["n_post_fire"] for p in per_prompt)

    fire_frac = tot_fire / tot_pos

    # prompt-level bootstrap CI (resample prompts, recompute ratio)
    rng = random.Random(SEED + k)
    npr = len(per_prompt)
    boots = []
    fires_arr = [p["n_fire"] for p in per_prompt]
    pos_arr = [p["n_pos"] for p in per_prompt]
    for _ in range(N_BOOT):
        sf = 0
        sp = 0
        for _ in range(npr):
            j = rng.randrange(npr)
            sf += fires_arr[j]
            sp += pos_arr[j]
        boots.append(sf / sp)
    boots.sort()
    ci_lo = boots[int(0.025 * N_BOOT)]
    ci_hi = boots[int(0.975 * N_BOOT)]

    # byte-exact subset (sha_ok) robustness
    bx = [p for p in per_prompt if p["sha_ok"]]
    bx_pos = sum(p["n_pos"] for p in bx)
    bx_fire = sum(p["n_fire"] for p in bx)

    sf = sorted(fires_per_prompt)
    return {
        "k": k,
        "n_prompts": npr,
        "total_positions": tot_pos,
        "total_fires": tot_fire,
        "served_fire_frac": fire_frac,
        "ci95_boot": [ci_lo, ci_hi],
        "ci95_wilson_naive": list(wilson(tot_fire, tot_pos)),
        "n_sha_ok": len(bx),
        "byte_exact_fire_frac": (bx_fire / bx_pos) if bx_pos else None,
        # pre/post divergence decomposition
        "pre_div_positions": pre_pos, "pre_div_fires": pre_fire,
        "pre_div_fire_frac": (pre_fire / pre_pos) if pre_pos else None,
        "post_div_positions": post_pos, "post_div_fires": post_fire,
        "post_div_fire_frac": (post_fire / post_pos) if post_pos else None,
        # clustering
        "fires_per_prompt_mean": statistics.fmean(fires_per_prompt),
        "fires_per_prompt_median": statistics.median(fires_per_prompt),
        "fires_per_prompt_min": min(fires_per_prompt),
        "fires_per_prompt_max": max(fires_per_prompt),
        "fires_per_prompt_p95": sf[max(0, math.ceil(0.95 * len(sf)) - 1)],
        "n_prompts_zero_fire": sum(1 for x in fires_per_prompt if x == 0),
        "mean_inter_fire_gap": statistics.fmean(inter_fire_gaps) if inter_fire_gaps else None,
        "median_inter_fire_gap": statistics.median(inter_fire_gaps) if inter_fire_gaps else None,
        "fires_per_prompt_hist": fires_per_prompt,
    }


def tax_crosscheck(fire_frac):
    """Back out stark's implied per-fire overhead ratio r from his endpoints, then
    apply r to the measured fire_frac.  base/(1+f*r)=proj  =>  r=(base/proj-1)/f_tf."""
    r = (OPTIONB_BASE_TPS / STARK_PROJ_TPS - 1.0) / STARK_TF_FIRE
    wall = OPTIONB_BASE_TPS / (1.0 + fire_frac * r)
    # fire_frac that would drop wall to the locked bar
    f_breakeven = (OPTIONB_BASE_TPS / LOCKED_TPS - 1.0) / r
    return {
        "stark_implied_overhead_ratio_r": r,
        "crosscheck_wall_tps": wall,
        "delta_vs_stark_proj": wall - STARK_PROJ_TPS,
        "stays_above_locked_126p378": wall > LOCKED_TPS,
        "fire_frac_breakeven_to_locked": f_breakeven,
    }


def main() -> int:
    ar = load_ar_first_div()
    results = {}
    for k in (3, 4, 5, 6, 7):
        r = census_one_k(k, ar)
        if r:
            r["tax_crosscheck"] = tax_crosscheck(r["served_fire_frac"])
            results[f"k{k}"] = r

    ks = sorted(int(kk[1:]) for kk in results)
    fracs = {k: results[f"k{k}"]["served_fire_frac"] for k in ks}

    # K-independence read: spread of per-K fire-frac vs the CIs
    if len(ks) >= 2:
        fr_vals = [fracs[k] for k in ks]
        k_spread = max(fr_vals) - min(fr_vals)
        # do all per-K bootstrap CIs overlap a common point? (use widest CI as ref)
        cis = [results[f"k{k}"]["ci95_boot"] for k in ks]
        common_lo = max(c[0] for c in cis)
        common_hi = min(c[1] for c in cis)
        k_independent = common_lo <= common_hi
    else:
        k_spread = 0.0
        k_independent = None

    # headline K (prefer K=5 to match stark #642)
    hk = 5 if 5 in ks else ks[len(ks) // 2]
    head = results[f"k{hk}"]
    overall_frac = head["served_fire_frac"]

    # verdict
    pre = head["pre_div_fire_frac"]
    post = head["post_div_fire_frac"]
    amp = (post / pre) if (pre and post) else None
    if k_spread > 0.02:  # >2 pp swing across K
        verdict = "FIRE_TAX_K_DEPENDENT"
    elif overall_frac <= STARK_TF_FIRE * 1.15:  # within ~15% of his 7.80% (or lower)
        verdict = "FIRE_TAX_CHEAP"
    elif head["tax_crosscheck"]["stays_above_locked_126p378"]:
        verdict = "FIRE_TAX_AMPLIFIED"  # higher than TF but speed leg still clears
    else:
        verdict = "FIRE_TAX_AMPLIFIED"  # and breaks the >126.378 leg (noted in numbers)

    out = {
        "tau": TAU,
        "stark_tf_fire_frac": STARK_TF_FIRE,
        "headline_k": hk,
        "served_fire_frac_overall": overall_frac,
        "served_fire_frac_ci95": head["ci95_boot"],
        "per_k_fire_frac": fracs,
        "k_spread_pp": k_spread,
        "k_independent": k_independent,
        "post_over_pre_amplification": amp,
        "crosscheck_wall_tps": head["tax_crosscheck"]["crosscheck_wall_tps"],
        "stays_above_locked": head["tax_crosscheck"]["stays_above_locked_126p378"],
        "verdict": verdict,
        "per_k": results,
    }
    OUT.write_text(json.dumps(out, indent=2))

    print("\n=========== SERVED RECOMPUTE-FIRE CENSUS (PR #648) ===========")
    print(f"tau = {TAU} nat ; stark TF fire = {STARK_TF_FIRE*100:.2f}%")
    print(f"\n  K |  positions |  fires  | fire_frac |   boot95 CI    | pre-div | post-div | sha_ok")
    for k in ks:
        r = results[f"k{k}"]
        print(f"  {k} | {r['total_positions']:9d} | {r['total_fires']:6d} | "
              f"{r['served_fire_frac']*100:7.3f}% | "
              f"[{r['ci95_boot'][0]*100:.2f},{r['ci95_boot'][1]*100:.2f}]% | "
              f"{(r['pre_div_fire_frac'] or 0)*100:6.3f}% | {(r['post_div_fire_frac'] or 0)*100:7.3f}% | "
              f"{r['n_sha_ok']}/{r['n_prompts']}")
    print(f"\nK-spread = {k_spread*100:.3f} pp ; K-independent(CI overlap) = {k_independent}")
    h = head
    print(f"\nHeadline K={hk}:")
    print(f"  served_fire_frac = {overall_frac*100:.3f}%  (stark TF {STARK_TF_FIRE*100:.2f}%) "
          f"=> ratio {overall_frac/STARK_TF_FIRE:.2f}x")
    print(f"  pre-div(AR-eq) fire = {(pre or 0)*100:.3f}%  | post-div(off-AR) fire = {(post or 0)*100:.3f}%  "
          f"| amplification post/pre = {amp:.2f}x" if amp else "")
    print(f"  fires/prompt: mean {h['fires_per_prompt_mean']:.2f} median {h['fires_per_prompt_median']} "
          f"max {h['fires_per_prompt_max']} ; zero-fire prompts {h['n_prompts_zero_fire']}/{h['n_prompts']}")
    print(f"  mean inter-fire gap = {h['mean_inter_fire_gap']:.1f} tokens "
          f"(median {h['median_inter_fire_gap']})")
    tc = h["tax_crosscheck"]
    print(f"\nTax cross-check (stark implied r = {tc['stark_implied_overhead_ratio_r']:.3f} per-fire token-times):")
    print(f"  Option-B base {OPTIONB_BASE_TPS} -> crosscheck wall_tps = {tc['crosscheck_wall_tps']:.2f} "
          f"(stark proj {STARK_PROJ_TPS}; delta {tc['delta_vs_stark_proj']:+.2f})")
    print(f"  stays above locked {LOCKED_TPS}: {tc['stays_above_locked_126p378']} "
          f"(breakeven fire_frac to locked = {tc['fire_frac_breakeven_to_locked']*100:.2f}%)")
    print(f"\nVERDICT: {verdict}")
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
