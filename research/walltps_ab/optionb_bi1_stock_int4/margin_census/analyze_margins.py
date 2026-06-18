"""PR #645 step 3: margin histogram + verdict from the captured root-fork logprobs.

Reads margin_records.jsonl (one row per ROOT fork, raw_top = top-N [token_str,
logprob] from the served K=7 verify forward). Maps both the spec token A and the
AR token B (ids) to vLLM's string-keyed logprobs via the tokenizer, then:

  margin_AB = logp(A) - logp(B)   # PR instr #2: spec-emitted vs AR-arg gap
  margin_AC = logp(top1) - logp(top2)   # literal M=8 top1-top2 gap (gap-acceptor
                                          # threshold quantity)

At a ROOT fork the spec-accepted prefix is byte-identical to AR, so the M=8 slot
shares the M=1 causal context (differs only by varlen width -> FP). So B should be
the M=8 runner-up; we verify that empirically and report agreement.

Two of the 128 prompts (idx 0,1) are near-tie sensitive: requesting logprobs (the
#632 capture did NOT) tips an earlier near-tie and perturbs the trajectory. We
handle them faithfully (see rematch_flipped.py / forced_prefix_margins.py):
  * fork1 mmlu_pro-006f3a2112 p=161: replay diverges EXACTLY at 161 (prefix[:161]
    byte-identical to #632) -> on-#632-context. It is a PERFECT 0.0-nat tie; the
    argmax flips A<->B purely by reduction order. margin = 0.0, faithful.
  * fork2 mmlu_pro-012f0d5c8d p=214: replay diverges at 54 (off-context at 214).
    Best stored-context estimate via forced-prefix (forced_prefix_records.jsonl):
    gap 0.125 nat, A on top. Tagged OFFTRAJ_ESTIMATE; <0.5 by every measurement
    and by root-fork construction. Reported in full hist; excluded from strict hist.

Coverage of stark #636's tau=0.5 recompute flag (flags positions with gap < tau):
a fork is CAUGHT iff gap < 0.5, a HOLE iff gap >= 0.5.
served_min_tau_for_zero_break = max gap over forks (smallest tau that flags all).
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "margin_records.jsonl"
FORCED = HERE / "forced_prefix_records.jsonl"
OUT = HERE / "margin_census_result.json"
TOKENIZER = "google/gemma-4-E4B-it"

# fork classification (see module docstring)
FORK1 = "mmlu_pro-006f3a2112"   # on-#632-context perfect tie (faithful, gap 0.0)
FORK2 = "mmlu_pro-012f0d5c8d"   # off-context; forced-prefix stored-context estimate


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    k = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
    return s[k]


def lookup(raw, s):
    """Return (logprob, rank) of decoded string s in raw_top, or (None, None)."""
    for i, (t, lp) in enumerate(raw):
        if t == s:
            return lp, i
    return None, None


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def dec(tid):
        return tok.decode([tid])

    rows = {json.loads(l)["id"]: json.loads(l)
            for l in REC.read_text().splitlines() if l.strip()}
    forced = {json.loads(l)["id"]: json.loads(l)
              for l in FORCED.read_text().splitlines() if l.strip()} if FORCED.exists() else {}

    # splice fork2 with its forced-prefix stored-context record (on the right prefix)
    if FORK2 in forced:
        rows[FORK2] = forced[FORK2]

    rows = list(rows.values())
    print(f"[load] {len(rows)} root-fork records")

    n_faithful = sum(bool(r.get("sha_ok") and r.get("emitted_eq_A")) for r in rows)
    print(f"[class] on-context faithful (sha_ok & emit==A): {n_faithful}")
    print(f"[class] fork1 {FORK1}: on-#632-context perfect tie (faithful gap 0.0)")
    print(f"[class] fork2 {FORK2}: OFFTRAJ_ESTIMATE via forced-prefix")

    n_decode_mapping_ok = 0
    results = []
    b_not_found = []
    for r in rows:
        A, B = r["A"], r["B"]
        raw = r["raw_top"]  # [[token_str, logprob], ...] desc
        top1_str, logp_top1 = raw[0]
        top2_str, logp_top2 = raw[1]
        A_str, B_str = dec(A), dec(B)
        logp_A, A_rank = lookup(raw, A_str)
        logp_B, B_rank = lookup(raw, B_str)
        # A must be present (it is the emitted/argmax or a co-leader of a tie)
        if logp_A is None:
            logp_A = r.get("logp_A")  # emitted-token logprob fallback
            A_rank = 0
        mapping_ok = (A_str == top1_str)        # A is literal top-1
        n_decode_mapping_ok += mapping_ok

        # classification
        if r["id"] == FORK2:
            cls = "offtraj_estimate"
        elif (r.get("sha_ok") and r.get("emitted_eq_A")):
            cls = "faithful"
        else:
            cls = "on_context_tie"   # fork1: prefix matched, perfect tie

        margin_AC = logp_top1 - logp_top2       # literal top1-top2 gap
        if logp_B is None:
            b_not_found.append(r["id"])
            margin_AB = None
            margin_AB_lb = logp_A - raw[-1][1]  # A beats 20th-best by >= this
            B_eq_C = False
        else:
            margin_AB = logp_A - logp_B
            margin_AB_lb = margin_AB
            # B is the runner-up if it shares the top2 logprob (handles ties)
            B_eq_C = (B_rank == 1) or (logp_B == logp_top2)
        C_str = top2_str
        logp_C = logp_top2

        results.append({
            "id": r["id"], "root_pos": r["root_pos"], "cls": cls,
            "A": A, "B": B, "A_str": A_str, "B_str": B_str, "C_str": C_str,
            "logp_A": logp_A, "logp_B": logp_B, "logp_C": logp_C,
            "A_rank": A_rank, "B_rank": B_rank, "B_found": logp_B is not None,
            "B_eq_C": B_eq_C, "margin_AB": margin_AB, "margin_AB_lb": margin_AB_lb,
            "margin_AC": margin_AC, "mapping_ok": mapping_ok,
        })

    print(f"[map] A id->string decode == vLLM top-1 key on {n_decode_mapping_ok}/{len(rows)} forks "
          f"(fork1 is a tie where B sorts above A; expected)")
    if b_not_found:
        print(f"[warn] B not in top-20 on {len(b_not_found)} forks: {b_not_found[:10]}")

    def gaps_for(rs, use_AB):
        if use_AB:
            return [x["margin_AB"] if x["margin_AB"] is not None else x["margin_AB_lb"] for x in rs]
        return [x["margin_AC"] for x in rs]

    def binify(gaps):
        n = len(gaps)
        sub = sum(g < 0.5 for g in gaps)
        mid = sum(0.5 <= g < 1.0 for g in gaps)
        tail = sum(g >= 1.0 for g in gaps)
        return {
            "n": n,
            "frac_sub_0p5": sub / n, "n_sub_0p5": sub,
            "frac_0p5_to_1p0": mid / n, "n_0p5_to_1p0": mid,
            "frac_ge_1p0": tail / n, "n_ge_1p0": tail,
            "min": min(gaps), "median": statistics.median(gaps),
            "p95": pctl(gaps, 0.95), "max": max(gaps),
            "mean": statistics.fmean(gaps),
        }

    hist_AB = binify(gaps_for(results, True))
    hist_AC = binify(gaps_for(results, False))
    # strict subset: drop the 1 offtraj estimate (fork2)
    strict = [x for x in results if x["cls"] != "offtraj_estimate"]
    hist_AB_strict = binify(gaps_for(strict, True))
    hist_AC_strict = binify(gaps_for(strict, False))

    n_B_eq_C = sum(x["B_eq_C"] for x in results)
    served_min_tau = max(x["margin_AC"] for x in results)
    holes = [x for x in results if x["margin_AC"] >= 0.5]
    verdict = "FLAG_COVERS_ALL" if not holes else (
        "SERVED_TAU_HIGHER" if served_min_tau > 0.5 else "FLAG_HAS_HOLE")

    worst = sorted(results, key=lambda x: -x["margin_AC"])[:8]

    out = {
        "n_prompts_diverged": len(rows),
        "n_root_forks": len(rows),
        "n_on_context_faithful": n_faithful,
        "n_on_context_tie": sum(x["cls"] == "on_context_tie" for x in results),
        "n_offtraj_estimate": sum(x["cls"] == "offtraj_estimate" for x in results),
        "hist_AB_pr_instruction": hist_AB,
        "hist_AC_top1_top2_gap": hist_AC,
        "hist_AB_strict_107": hist_AB_strict,
        "hist_AC_strict_107": hist_AC_strict,
        "frac_sub_0p5": hist_AB["frac_sub_0p5"],
        "frac_0p5_to_1p0": hist_AB["frac_0p5_to_1p0"],
        "frac_ge_1p0": hist_AB["frac_ge_1p0"],
        "margin_median": hist_AB["median"],
        "margin_p95": hist_AB["p95"],
        "margin_max": hist_AB["max"],
        "served_min_tau_for_zero_break": served_min_tau,
        "stark_teacher_forced_min_tau": 0.5,
        "transfers_to_served": served_min_tau < 0.5,
        "n_B_eq_C": n_B_eq_C, "frac_B_eq_C": n_B_eq_C / len(rows),
        "n_B_not_found_in_top20": len(b_not_found),
        "b_not_found_ids": b_not_found,
        "n_decode_mapping_ok": n_decode_mapping_ok,
        "verdict": verdict,
        "worst_forks": [
            {"id": w["id"], "root_pos": w["root_pos"], "cls": w["cls"], "A_str": w["A_str"],
             "B_str": w["B_str"], "C_str": w["C_str"], "margin_AC": w["margin_AC"],
             "margin_AB": w["margin_AB"], "B_eq_C": w["B_eq_C"]}
            for w in worst
        ],
    }
    OUT.write_text(json.dumps(out, indent=2))
    (HERE / "margin_per_fork.jsonl").write_text("\n".join(json.dumps(x) for x in results) + "\n")

    print("\n========== MARGIN CENSUS ==========")
    print(f"n_root_forks = {out['n_root_forks']}  "
          f"(faithful {out['n_on_context_faithful']}, tie {out['n_on_context_tie']}, "
          f"estimate {out['n_offtraj_estimate']})")
    print(f"B == M=8 runner-up (B_eq_C): {n_B_eq_C}/{len(rows)} ({out['frac_B_eq_C']*100:.1f}%)")
    print(f"\n-- AR-token margin (A-B), PR instruction #2 [full 108] --")
    for k in ("frac_sub_0p5", "frac_0p5_to_1p0", "frac_ge_1p0", "min", "median", "p95", "max"):
        print(f"   {k}: {hist_AB[k]}")
    print(f"\n-- M=8 top1-top2 gap (A-C) [full 108] --")
    for k in ("frac_sub_0p5", "frac_0p5_to_1p0", "frac_ge_1p0", "min", "median", "p95", "max"):
        print(f"   {k}: {hist_AC[k]}")
    print(f"\n-- strict 107 (drop 1 offtraj estimate) AC gap --")
    for k in ("frac_sub_0p5", "frac_0p5_to_1p0", "frac_ge_1p0", "max"):
        print(f"   {k}: {hist_AC_strict[k]}")
    print(f"\nserved_min_tau_for_zero_break = {served_min_tau:.4f} nat "
          f"(stark teacher-forced min_tau=0.5; transfers={out['transfers_to_served']})")
    print(f"VERDICT: {verdict}")
    print(f"\nworst forks by top1-top2 gap:")
    for w in worst[:6]:
        ab = f"{w['margin_AB']:.4f}" if w['margin_AB'] is not None else "None"
        print(f"   {w['id']} p={w['root_pos']} [{w['cls']}] A={w['A_str']!r} B={w['B_str']!r} "
              f"C={w['C_str']!r} gapAC={w['margin_AC']:.4f} gapAB={ab} BeqC={w['B_eq_C']}")
    print(f"\n[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
