#!/usr/bin/env python3
"""Spec-break reconvergence/realignment analysis (PR #686, analysis_only).

Post-hoc string analysis of already-generated 512-tok greedy decodes from the
#678/#673 confirm set (decode_{ar,suffix6,ngram5}_r0.jsonl). NO GPU, NO HF Job,
NO re-decode. Token-id comparison only (tokenizer not required).

Question: after the spec arm first diverges from AR at position p, do the two
token streams RECONVERGE (re-synchronize / "couple", in the Markov sense) or
fork permanently? A break that heals in a few tokens (transient near-tie wobble,
kanna #680/#673 near_tie_break_rate=0.0201) is far more likely to preserve the
eventual answer than one that forks the trajectory.

Two coupling definitions (both reported):
  * SHIFT-TOLERANT (headline): spec[q:q+W] exactly matches a contiguous AR window
    of length W -- ALLOWING a small index shift (insertion/deletion). This is the
    PR's literal definition ("matches a contiguous AR window") and the physically
    correct one: a single near-tie argmax flip frequently makes one stream emit
    one fewer/more token than the other (e.g. spec deletes a token), so the
    streams re-sync on identical CONTENT but at a SHIFTED token index. Measured
    with difflib longest-matching-block on the post-divergence tails.
  * SAME-POSITION (strict shift=0 lower bound): spec[q:q+W]==AR[q:q+W] at the same
    index. Only catches heals with zero token-count change.

W=16 is coincidence-proof for a 256k-vocab model: P(16 tokens match by chance)
= V^-16 ~ 1e-83, so any 16-gram match (shifted or not) is a genuine re-sync.

EOS-filler: ignore_eos=True forces 512 tokens. If AR emits <end_of_turn>(106) or
<eos>(1) at position e, everything after e is forced filler; a "break" at p>=e is
score-immaterial. These are flagged and excluded from the score-relevant frac.
"""
import json
import re
import difflib
import statistics
from collections import defaultdict

CONFIRM = "research/validity/specdec_official_dist_breakrate/_runs/confirm"
L = 512  # all completions are exactly 512 tokens (ignore_eos, max_tokens=512)
WINDOWS = [16, 8]  # W=16 headline, W=8 sensitivity
HEADLINE_W = 16
EOS_TOKENS = {106, 1}  # <end_of_turn>, <eos>

ANS_MC = re.compile(r"ANSWER:\s*\(?([A-J])\b", re.IGNORECASE)
ANS_NUM = re.compile(r"ANSWER:\s*\(?(-?\d+)")


def load(path):
    return [json.loads(line) for line in open(path)]


def first_div(A, B):
    n = min(len(A), len(B))
    for i in range(n):
        if A[i] != B[i]:
            return i
    return n if len(A) != len(B) else None


def first_eos(toks):
    for i, t in enumerate(toks):
        if t in EOS_TOKENS:
            return i
    return None


def matched_suffix_len(A, B):
    """Maximal common suffix (same-position match to token 512)."""
    k = 0
    n = min(len(A), len(B))
    while k < n and A[-1 - k] == B[-1 - k]:
        k += 1
    return k


def first_sameposition_window(A, B, p, W):
    for q in range(p + 1, len(A) - W + 1):
        if A[q:q + W] == B[q:q + W]:
            return q
    return None


def shift_tolerant_resync(A, B, p, W):
    """difflib longest-matching-block on post-divergence tails A[p:], B[p:].

    Returns dict with: ever (bool, a >=W coupling block exists after p),
    onset (abs spec index where first >=W block begins), gap (tokens after p
    until re-sync), shift (AR_offset - SP_offset of that block, +=spec deleted),
    block_size, stayed_to_end (final matched block reaches both tail ends),
    coverage (frac of tail covered by >=4-tok matching blocks)."""
    At, Bt = A[p:], B[p:]
    sm = difflib.SequenceMatcher(a=At, b=Bt, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    big = [b for b in blocks if b.size >= W]
    out = dict(ever=False, onset=None, gap=None, shift=None, block_size=None,
               stayed_to_end=False, ends_coupled=False, coverage=0.0,
               largest=max((b.size for b in blocks), default=0))
    if big:
        # first >=W block = smallest onset (min offset across the two tails)
        first = min(big, key=lambda b: min(b.a, b.b))
        gap = min(first.a, first.b)  # tokens after p before re-sync onset
        out.update(ever=True, onset=p + first.b, gap=gap,
                   shift=first.a - first.b, block_size=first.size)
        # stayed: a >=W block ends at BOTH tail ends (streams end coupled, shift=0
        # at the boundary -- rare under a shift, kept for completeness)
        for b in big:
            if (b.a + b.size == len(At)) and (b.b + b.size == len(Bt)):
                out["stayed_to_end"] = True
                break
        # ends_coupled: a >=W block reaches AR's tail END (a+size==len(At)) -- AR's
        # final W tokens are matched (possibly shifted) in spec. Since #685 showed
        # the answer is emitted LAST, coupling at the 512 truncation boundary is the
        # cleanest predictor that the 6144-tok continuation stays coupled -> answer
        # preserved. (#682 is the ground truth for the 6144 continuation.)
        for b in big:
            if b.a + b.size == len(At):
                out["ends_coupled"] = True
                break
    # coverage: tail tokens inside >=4-token matching blocks / tail length
    covered = sum(b.size for b in blocks if b.size >= 4)
    out["coverage"] = covered / max(len(At), 1)
    return out


def extract_answer(text):
    m = ANS_MC.search(text)
    if m:
        return ("mc", m.group(1).upper())
    m = ANS_NUM.search(text)
    if m:
        return ("num", m.group(1))
    return None


def classify(rec):
    if not rec["assessable"]:
        return "NOT_ASSESSABLE"
    if rec["filler"]:
        return "FILLER"
    if rec["samepos_ever"]:
        return "SAMEPOS_HEAL"
    if rec["shift_ever"]:
        return "SHIFTED_HEAL"
    return "PERMANENT_FORK"


def analyze_cell(ar, spec, W):
    per_prompt = []
    for a, s in zip(ar, spec):
        A = a["completion_token_ids"]
        S = s["completion_token_ids"]
        p = first_div(A, S)
        if p is None:
            continue
        source = a["id"].split("-")[0]
        assessable = (L - p) >= W
        ar_eos = first_eos(A)
        filler = (ar_eos is not None) and (p >= ar_eos)

        # same-position (strict shift=0)
        q_sp = first_sameposition_window(A, S, p, W) if assessable else None
        samepos_ever = assessable and (q_sp is not None)
        msuf = matched_suffix_len(A, S)
        samepos_stayed = assessable and (msuf >= W) and (L - msuf > p)

        # shift-tolerant (headline)
        st = shift_tolerant_resync(A, S, p, W) if assessable else dict(
            ever=False, onset=None, gap=None, shift=None, block_size=None,
            stayed_to_end=False, ends_coupled=False, coverage=0.0, largest=0)

        ar_ans = extract_answer(a["generated_text"])
        sp_ans = extract_answer(s["generated_text"])
        both_commit = (ar_ans is not None) and (sp_ans is not None)

        rec = dict(
            id=a["id"], source=source, p=p, assessable=assessable, filler=filler,
            ar_eos=ar_eos,
            samepos_ever=samepos_ever, samepos_q=q_sp,
            samepos_gap=(q_sp - p) if samepos_ever else None,
            samepos_stayed=samepos_stayed, matched_suffix_len=msuf,
            shift_ever=st["ever"], shift_onset=st["onset"],
            shift_gap=st["gap"], shift=st["shift"], shift_block=st["block_size"],
            shift_stayed=st["stayed_to_end"], ends_coupled=st["ends_coupled"],
            coverage=round(st["coverage"], 3), largest_block=st["largest"],
            ar_commit=ar_ans is not None, sp_commit=sp_ans is not None,
            both_commit=both_commit,
            answer_flip=both_commit and (ar_ans != sp_ans),
        )
        rec["klass"] = classify(rec)
        per_prompt.append(rec)
    return per_prompt


def frac(x, d):
    return (x / d) if d else float("nan")


def aggregate(pp):
    n_break = len(pp)
    assess = [r for r in pp if r["assessable"]]
    n_assess = len(assess)
    filler = [r for r in assess if r["filler"]]
    # score-relevant denominator = assessable & not filler
    relevant = [r for r in assess if not r["filler"]]
    n_rel = len(relevant)

    shift_rec = [r for r in assess if r["shift_ever"]]
    sp_rec = [r for r in assess if r["samepos_ever"]]
    shift_rec_rel = [r for r in relevant if r["shift_ever"]]
    shift_gaps = [r["shift_gap"] for r in shift_rec]
    sp_gaps = [r["samepos_gap"] for r in sp_rec]
    shifts = [r["shift"] for r in shift_rec if r["shift"] is not None]
    perm = [r for r in assess if not r["shift_ever"]]
    redivs = [r for r in assess if r["shift_ever"] and not r["shift_stayed"]]
    ends_coupled = [r for r in assess if r["ends_coupled"]]
    ends_coupled_rel = [r for r in relevant if r["ends_coupled"]]
    klass = defaultdict(int)
    for r in pp:
        klass[r["klass"]] += 1

    return dict(
        n_break=n_break, assessable=n_assess,
        assessable_frac=frac(n_assess, n_break),
        not_assessable=n_break - n_assess,
        filler=len(filler), score_relevant=n_rel,
        # HEADLINE: shift-tolerant reconverge over assessable
        reconverge_frac=frac(len(shift_rec), n_assess),
        reconverged=len(shift_rec),
        # over score-relevant denominator (excl filler)
        reconverge_frac_relevant=frac(len(shift_rec_rel), n_rel),
        # strict same-position (shift=0) lower bound
        reconverge_frac_samepos=frac(len(sp_rec), n_assess),
        reconverged_samepos=len(sp_rec),
        permanent_divergence_frac=frac(len(perm), n_assess),
        permanent_divergence=len(perm),
        redivergence=len(redivs),
        # coupled at the 512 truncation boundary (best #682 retention predictor)
        ends_coupled_frac=frac(len(ends_coupled), n_assess),
        ends_coupled=len(ends_coupled),
        ends_coupled_frac_relevant=frac(len(ends_coupled_rel), n_rel),
        # realign gap = tokens after p until re-sync onset (shift-tolerant)
        median_realign_gap=(statistics.median(shift_gaps) if shift_gaps else None),
        mean_realign_gap=(round(statistics.mean(shift_gaps), 1) if shift_gaps else None),
        max_realign_gap=(max(shift_gaps) if shift_gaps else None),
        median_samepos_gap=(statistics.median(sp_gaps) if sp_gaps else None),
        median_abs_shift=(statistics.median([abs(x) for x in shifts]) if shifts else None),
        max_abs_shift=(max([abs(x) for x in shifts]) if shifts else None),
        mean_coverage=(round(statistics.mean([r["coverage"] for r in assess]), 3) if assess else None),
        both_commit=sum(1 for r in pp if r["both_commit"]),
        answer_flips=sum(1 for r in pp if r["answer_flip"]),
        klass=dict(klass),
    )


def per_source(pp):
    by = defaultdict(list)
    for r in pp:
        by[r["source"]].append(r)
    return {src: aggregate(rows) for src, rows in sorted(by.items())}


def main():
    ar = load(f"{CONFIRM}/decode_ar_r0.jsonl")
    cells = {
        "suffix6": load(f"{CONFIRM}/decode_suffix6_r0.jsonl"),
        "ngram5": load(f"{CONFIRM}/decode_ngram5_r0.jsonl"),
    }
    assert len(ar) == 128
    for name, spec in cells.items():
        assert len(spec) == 128
        for a, s in zip(ar, spec):
            assert a["prompt_sha256"] == s["prompt_sha256"], f"misalign {name}"

    report = {"L": L, "windows": WINDOWS, "headline_W": HEADLINE_W, "cells": {}}
    for name, spec in cells.items():
        report["cells"][name] = {"W": {}}
        for W in WINDOWS:
            pp = analyze_cell(ar, spec, W)
            agg = aggregate(pp)
            agg["per_source"] = per_source(pp)
            report["cells"][name]["W"][W] = agg
            report["cells"][name]["per_prompt_W%d" % W] = pp

    with open("research/validity/specbreak_reconvergence/reconvergence_report.json", "w") as f:
        json.dump(report, f, indent=2)

    def fm(v, nd=4):
        if v is None:
            return "n/a"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    for name in cells:
        print(f"\n{'='*74}\nCELL {name}\n{'='*74}")
        for W in WINDOWS:
            a = report["cells"][name]["W"][W]
            tag = "   <<< HEADLINE" if W == HEADLINE_W else ""
            print(f"\n  W={W}{tag}")
            print(f"    n_break={a['n_break']}  assessable={a['assessable']} "
                  f"(frac={fm(a['assessable_frac'])})  not_assessable={a['not_assessable']}  "
                  f"filler={a['filler']}  score_relevant={a['score_relevant']}")
            print(f"    RECONVERGE_FRAC (shift-tolerant / assessable) = "
                  f"{fm(a['reconverge_frac'])}  [{a['reconverged']}/{a['assessable']}]   <<< headline")
            print(f"      reconverge_frac (excl filler)               = "
                  f"{fm(a['reconverge_frac_relevant'])}")
            print(f"      reconverge_frac_samepos (strict shift=0)    = "
                  f"{fm(a['reconverge_frac_samepos'])}  [{a['reconverged_samepos']}/{a['assessable']}]")
            print(f"    ends_coupled_frac (coupled at 512 boundary)   = "
                  f"{fm(a['ends_coupled_frac'])}  [{a['ends_coupled']}/{a['assessable']}]  "
                  f"(excl filler={fm(a['ends_coupled_frac_relevant'])})  <<< best #682 predictor")
            print(f"    permanent_divergence_frac (never re-syncs)    = "
                  f"{fm(a['permanent_divergence_frac'])}  [{a['permanent_divergence']}/{a['assessable']}]")
            print(f"    redivergence (re-synced, boundary not exact)  = {a['redivergence']}")
            print(f"    realign gap (shift): median={fm(a['median_realign_gap'],1)} "
                  f"mean={fm(a['mean_realign_gap'],1)} max={fm(a['max_realign_gap'],1)} tok   "
                  f"samepos_gap_median={fm(a['median_samepos_gap'],1)}")
            print(f"    shift magnitude: median|shift|={fm(a['median_abs_shift'],1)} "
                  f"max|shift|={fm(a['max_abs_shift'],1)}  mean_coverage={fm(a['mean_coverage'],3)}")
            print(f"    classes: {a['klass']}")
            print(f"    answer cross-check: both_commit={a['both_commit']}  answer_flips={a['answer_flips']}")
            print(f"    per-source:")
            for src, sa in a["per_source"].items():
                print(f"      {src:13s} n={sa['n_break']:2d} assess={sa['assessable']:2d} "
                      f"reconv={fm(sa['reconverge_frac'])} "
                      f"samepos={fm(sa['reconverge_frac_samepos'])} "
                      f"perm_div={fm(sa['permanent_divergence_frac'])} "
                      f"med_gap={fm(sa['median_realign_gap'],1)} klass={sa['klass']}")
    print("\nwrote reconvergence_report.json")


if __name__ == "__main__":
    main()
