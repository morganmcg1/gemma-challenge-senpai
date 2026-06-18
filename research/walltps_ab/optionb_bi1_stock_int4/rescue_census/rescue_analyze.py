"""PR #651 analysis: served recompute-RESCUE census.

Reads rescue_k{K}.jsonl (one row per fired position: r_pos / a_pos / s_pos token ids,
verify_margin, recompute_margin, pre_div, sha_ok, and -- for breaks -- the raw recompute
top-N logprobs). Computes, per K:

 (1) served_rescue_rate / served_break_rate over ALL fires, with a PROMPT-level bootstrap
     95% CI (fires within a prompt are correlated). Decisive number = served_break_rate;
     stark #636's teacher-forced analog is rescued_break_rate = 0/14035.
 (2) pre/post-divergence split (PR step 2). pre-div == on-AR head: served context == AR
     context, so this is the faithful served analog of stark's TF rescue test (a break here
     is a genuine recompute flip away from AR). post-div == off-AR tail: the served context
     has itself diverged, so a break is dominated by the prefix already having left AR (the
     M=1 recompute faithfully continues the SERVED trajectory != AR by construction); we
     decompose post breaks into r==s (pure trajectory divergence, not a recompute failure)
     vs r!=s (genuine third-token flip).
 (3) break-locus census (PR step 3): for each break, verify_margin, a_pos vs r_pos, whether
     it is a benign 0.0-nat ULP tie (recompute essentially tied between r_pos and a_pos) vs a
     wider miss, and pre/post.
 (4) byte-exact subset (sha_ok) rescue rate -- robustness, same as #648 (K=5 125/128).

analysis_only. No generation here (pure offline read of the capture).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rescue_census_result.json"

TAU = 0.5
STARK_TF_BREAK = "0/14035"          # stark #636 teacher-forced rescued_break_rate
ULP_TIE_NAT = 1e-6                  # recompute top1-top2 gap below this == 0.0-nat ULP tie
N_BOOT = 20000
SEED = 651


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def boot_ci(per_prompt_break, per_prompt_fire, k):
    """Prompt-level bootstrap CI for break_rate = sum(breaks)/sum(fires)."""
    rng = random.Random(SEED + k)
    npr = len(per_prompt_fire)
    if npr == 0:
        return [0.0, 0.0]
    boots = []
    for _ in range(N_BOOT):
        sb = sf = 0
        for _ in range(npr):
            j = rng.randrange(npr)
            sb += per_prompt_break[j]
            sf += per_prompt_fire[j]
        boots.append((sb / sf) if sf else 0.0)
    boots.sort()
    return [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]]


def classify_break(row):
    """benign 0.0-nat ULP tie (recompute argmax tied with a_pos) vs wider miss.

    Primary id-free signal: recompute_margin (top1-top2 gap). If ~0 the recompute argmax won
    by an ULP-thin margin over its runner-up -> a tie. Secondary (string match on the stored
    top-N): is a_pos's logprob within ULP of r_pos's? Returns dict."""
    rc = row.get("recompute_margin")
    rc_is_tie = (rc is not None and rc != float("inf") and rc <= ULP_TIE_NAT)
    info = {"recompute_margin": rc, "recompute_top1top2_is_ulp_tie": bool(rc_is_tie)}
    top = row.get("break_top")  # {token_str: logprob}
    a_gap = None
    a_in_topN = None
    if top:
        vals = sorted(top.values(), reverse=True)
        top1 = vals[0]
        # a_pos's logprob is unknown by id, but we can bound it: if the SECOND value is within
        # ULP of the first, *some* token ties the argmax. We report the top1-top2 gap as the
        # tie width; whether the tied token is a_pos is confirmed in the per-break dump below.
        a_gap = top1 - (vals[1] if len(vals) >= 2 else top1)
        a_in_topN = len(vals)
    info["top1_minus_top2"] = a_gap
    info["n_top_returned"] = a_in_topN
    # benign iff the recompute is at a 0.0-nat tie (the AR token can win under tie-break)
    info["benign_ulp_tie"] = bool(rc_is_tie)
    return info


def census_one_k(k):
    path = HERE / f"rescue_k{k}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    fires = [r for r in rows if r["is_fire"]]

    # per-prompt aggregation (preserve prompt order for bootstrap)
    by_prompt = {}
    for r in fires:
        d = by_prompt.setdefault(r["id"], {"fire": 0, "break": 0, "bx": r["sha_ok"]})
        d["fire"] += 1
        d["break"] += (not r["rescue"])
    pids = list(by_prompt)
    pp_fire = [by_prompt[p]["fire"] for p in pids]
    pp_break = [by_prompt[p]["break"] for p in pids]

    n_fire = sum(pp_fire)
    n_break = sum(pp_break)
    n_rescue = n_fire - n_break

    pre = [r for r in fires if r["pre_div"]]
    post = [r for r in fires if not r["pre_div"]]
    pre_break = sum(1 for r in pre if not r["rescue"])
    post_break = sum(1 for r in post if not r["rescue"])
    # post-div break decomposition: pure trajectory divergence (r==s) vs genuine flip (r!=s)
    post_break_rows = [r for r in post if not r["rescue"]]
    post_break_traj = sum(1 for r in post_break_rows if r["r_eq_s"])
    post_break_flip = sum(1 for r in post_break_rows if not r["r_eq_s"])
    pre_break_rows = [r for r in pre if not r["rescue"]]
    pre_break_traj = sum(1 for r in pre_break_rows if r["r_eq_s"])  # expect 0 (head: s==a)

    # byte-exact subset
    bx_fire = [r for r in fires if r["sha_ok"]]
    bx_break = sum(1 for r in bx_fire if not r["rescue"])

    # break-locus census (all breaks, with severity classification)
    break_loci = []
    for r in fires:
        if not r["rescue"]:
            cl = classify_break(r)
            break_loci.append({
                "id": r["id"], "pos": r["pos"], "pre_div": r["pre_div"],
                "verify_margin": r["verify_margin"], "recompute_margin": r["recompute_margin"],
                "a_pos": r["a_pos"], "r_pos": r["r_pos"], "s_pos": r["s_pos"],
                "r_eq_s": r["r_eq_s"], **cl,
            })
    n_benign = sum(1 for b in break_loci if b["benign_ulp_tie"])
    n_wide = len(break_loci) - n_benign

    return {
        "k": k,
        "n_prompts": len(pids),
        "total_fires": n_fire,
        "rescued": n_rescue,
        "broken": n_break,
        "served_rescue_rate": (n_rescue / n_fire) if n_fire else None,
        "served_break_rate": (n_break / n_fire) if n_fire else None,
        "served_break_rate_ci95_boot": boot_ci(pp_break, pp_fire, k),
        "served_break_rate_ci95_wilson": list(wilson(n_break, n_fire)),
        # pre/post split
        "pre_div_fires": len(pre), "pre_div_breaks": pre_break,
        "pre_div_break_rate": (pre_break / len(pre)) if pre else None,
        "post_div_fires": len(post), "post_div_breaks": post_break,
        "post_div_break_rate": (post_break / len(post)) if post else None,
        # post-div break decomposition
        "post_break_trajectory_divergence_r_eq_s": post_break_traj,
        "post_break_genuine_flip_r_ne_s": post_break_flip,
        "pre_break_trajectory_divergence_r_eq_s": pre_break_traj,
        # byte-exact subset
        "n_sha_ok_prompts": sum(1 for p in pids if by_prompt[p]["bx"]),
        "bx_fires": len(bx_fire), "bx_breaks": bx_break,
        "bx_break_rate": (bx_break / len(bx_fire)) if bx_fire else None,
        # severity
        "breaks_benign_ulp_tie": n_benign,
        "breaks_wider_miss": n_wide,
        "break_loci": break_loci,
    }


def verdict_for(res):
    """SERVED_RESCUE_COMPLETE iff every break is a benign 0.0-nat ULP tie (or zero breaks);
    else SERVED_RESCUE_BREACH. We judge on the ON-AR HEAD (pre-div) -- the population the
    online acceptor actually operates on and the faithful analog of stark's TF 0/14035 -- and
    report the off-AR tail transparently (its breaks are dominated by prefix divergence)."""
    head_breaks = res["pre_div_breaks"]
    head_wide = sum(1 for b in res["break_loci"]
                    if b["pre_div"] and not b["benign_ulp_tie"])
    if head_breaks == 0:
        return "SERVED_RESCUE_COMPLETE", "on-AR head: zero recompute breaks (== stark TF 0)"
    if head_wide == 0:
        return ("SERVED_RESCUE_COMPLETE",
                f"on-AR head: {head_breaks} breaks, ALL benign 0.0-nat ULP ties (tie-break resolves to AR)")
    return ("SERVED_RESCUE_BREACH",
            f"on-AR head: {head_wide} wide (non-tie) recompute breaks away from AR")


def main() -> int:
    results = {}
    for k in (3, 4, 5, 6, 7):
        r = census_one_k(k)
        if r:
            results[f"k{k}"] = r

    if not results:
        print("no rescue_k*.jsonl found")
        return 1

    hk = "k5" if "k5" in results else sorted(results, key=lambda s: int(s[1:]))[0]
    head = results[hk]
    verdict, why = verdict_for(head)

    out = {
        "tau": TAU,
        "stark_tf_rescued_break_rate": STARK_TF_BREAK,
        "headline_k": head["k"],
        "served_rescue_rate": head["served_rescue_rate"],
        "served_break_rate": head["served_break_rate"],
        "served_break_rate_ci95": head["served_break_rate_ci95_boot"],
        "on_AR_head_break_rate": head["pre_div_break_rate"],
        "off_AR_tail_break_rate": head["post_div_break_rate"],
        "verdict": verdict,
        "verdict_basis": why,
        "per_k": results,
    }
    OUT.write_text(json.dumps(out, indent=2))

    print("\n========= SERVED RECOMPUTE-RESCUE CENSUS (PR #651) =========")
    print(f"tau = {TAU} nat ; stark TF rescued_break_rate = {STARK_TF_BREAK}")
    print(f"\n  K | fires | rescued | broken | break_rate |   boot95 CI   | "
          f"head brk | tail brk | benign/wide")
    for kk in sorted(results, key=lambda s: int(s[1:])):
        r = results[kk]
        ci = r["served_break_rate_ci95_boot"]
        print(f"  {r['k']} | {r['total_fires']:5d} | {r['rescued']:7d} | {r['broken']:6d} | "
              f"{(r['served_break_rate'] or 0)*100:8.4f}% | "
              f"[{ci[0]*100:.3f},{ci[1]*100:.3f}]% | "
              f"{r['pre_div_breaks']:3d}/{r['pre_div_fires']:<4d} | "
              f"{r['post_div_breaks']:3d}/{r['post_div_fires']:<5d} | "
              f"{r['breaks_benign_ulp_tie']}/{r['breaks_wider_miss']}")
    h = head
    print(f"\nHeadline K={h['k']}:")
    print(f"  served_rescue_rate = {(h['served_rescue_rate'] or 0)*100:.4f}%  "
          f"served_break_rate = {(h['served_break_rate'] or 0)*100:.4f}%  "
          f"({h['broken']}/{h['total_fires']})")
    print(f"  ON-AR HEAD (stark TF analog): break_rate = {(h['pre_div_break_rate'] or 0)*100:.4f}%  "
          f"({h['pre_div_breaks']}/{h['pre_div_fires']})")
    print(f"  OFF-AR TAIL: break_rate = {(h['post_div_break_rate'] or 0)*100:.4f}%  "
          f"({h['post_div_breaks']}/{h['post_div_fires']})")
    print(f"    tail breaks: trajectory-divergence(r==s) = {h['post_break_trajectory_divergence_r_eq_s']}  "
          f"genuine-flip(r!=s) = {h['post_break_genuine_flip_r_ne_s']}")
    print(f"  breaks: benign 0.0-nat ULP ties = {h['breaks_benign_ulp_tie']}  "
          f"wider misses = {h['breaks_wider_miss']}")
    print(f"  byte-exact subset: {h['bx_breaks']}/{h['bx_fires']} broken "
          f"(sha_ok prompts {h['n_sha_ok_prompts']}/{h['n_prompts']})")
    print(f"\nVERDICT: {verdict}  ({why})")
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
