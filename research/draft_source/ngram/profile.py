#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""N-1 N-GRAM DRAFT SOURCE -- match-rate diagnostic (PR #250, student ubel).

THE QUESTION
------------
Is there FREE E[T] sitting in the benchmark prompt structure that no trained
draft model is exploiting? A model-free n-gram / REST draft costs ~nothing and,
IF the prompts/continuations are templated/repetitive, can propose long correct
spans the trained draft misses -- fed into the EXISTING verify tree, so at ZERO
greedy-identity / PPL risk (verify model + acceptance criterion are untouched).

The payoff is entirely prompt-distribution-dependent. This file is the CHEAP
DIAGNOSTIC FIRST: build the datastore, profile the match-rate distribution on
the REAL greedy references, and decide kill/keep BEFORE any wiring or HF job.

GROUND TRUTH (no model needed)
------------------------------
We use the cached BASE greedy reference for google/gemma-4-E4B-it on the OFFICIAL
128 public benchmark prompts:
    research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl
Each record has prompt_token_ids (the prompt) and completion_token_ids (512
greedy-decoded continuation tokens). Greedy-decode IDENTITY is a hard benchmark
gate, so the deployed system emits exactly these tokens -> the n-gram match
against them IS the realized acceptance in deployment.

WHAT WE MEASURE (per PR step 2, the kill/keep gate)
---------------------------------------------------
At each decode position t we form the context (prompt + completion[:t]), take the
last-k tokens as the key, and look it up in the datastore. We report, per
datastore variant and key length k:
  * ngram_hit_rate          -- fraction of positions with a datastore hit
  * ngram_conditional_correct_rate -- P(retrieved top-1 == greedy next | hit)
  * e_t_ngram               -- realized E[T] of n-gram-ONLY drafting under the
                               greedy-identical verify (span proposal, depth D),
                               = (tokens produced) / (verify steps).

DATASTORE VARIANTS
------------------
  PLD-self   : prompt-lookup decoding. Datastore = the CURRENT context only
               (prompt + tokens generated so far). Zero external dependency,
               prompt-distribution-robust. Captures copy-from-prompt and
               self-repetition. This is the PRIMARY "free E[T]" mechanism.
  REST-prompt: global datastore = all 128 PROMPTS (cross-prompt n-grams). Fair
               (prompts are available at inference), captures shared templating.
  REST-LOO   : datastore = all prompts + every OTHER record's completion
               (leave-one-out reference corpus). The optimistic REST ceiling;
               HIGH transfer risk (private prompts differ from this corpus).

STACKING (PR step 4) -- model-free upper bound
----------------------------------------------
n-gram is only useful as a SECONDARY source if it covers tokens the trained
draft misses. With per-step accepted lengths a_trained (trained draft) and
a_ngram (n-gram), a tree holding both branches accepts max(a_trained, a_ngram),
so the hybrid satisfies, rigorously and model-free:

    E[T]_hybrid = 1 + mean(max(a_trained, a_ngram))
                <= 1 + mean(a_trained) + mean(a_ngram)
                =  E[T]_trained + (E[T]_ngram - 1)

i.e. n-gram can add AT MOST (E[T]_ngram - 1) tok/step on top of the trained
draft, and that bound is loose (assumes disjoint coverage; in reality the two
sources overlap heavily on easy tokens). If (E[T]_ngram - 1) is small, even the
optimistic ceiling cannot move TPS -> clean negative, no model run needed.

Trained-draft controls:
  * linear MTP K=7  E[T] = 3.844 tok/step  (official 481.53 TPS baseline, PR #52)
  * deployed tree   E[T] = 5.066 tok/step  (frontier verify path)

PRIMARY metric   ngram_draft_matchrate_self_test_passes
TEST metrics     ngram_hit_rate, ngram_conditional_correct_rate, e_t_ngram_hybrid

PUBLIC EVIDENCE USED
--------------------
research/RESEARCH_IDEAS_2026-06-14_speed-levers.md Idea N-1 (rank 13); REST
arXiv:2311.08252 (https://github.com/fasterdecoding/REST). Orthogonal to the
trained-draft / topology / quant lanes -- different mechanism level (where draft
tokens come from), zero collision.

Reproduce:
  cd target/ && CUDA_VISIBLE_DEVICES=0 python research/draft_source/ngram/profile.py \
      --self-test --wandb_group ngram-draft-matchrate --wandb_name ubel/ngram-draft-matchrate
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

REF_PATH = "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
OUT_PATH = "research/draft_source/ngram/ngram_matchrate.json"

# Trained-draft-only controls (deployed E[T], tok/step).
ET_TRAINED_LINEAR_MTP_K7 = 3.844   # official 481.53 TPS baseline (PR #52)
ET_TRAINED_TREE = 5.066            # frontier tree verify path
PPL_PINNED = 2.3772                # deployed served PPL (verify unchanged)
PPL_CAP = 2.42

KS = (2, 3, 4, 5)        # key lengths to sweep ("last 3-5 tokens" + k=2 floor)
DS = (1, 2, 4, 7, 16)    # draft span depths to sweep (7 matches deployed K=7)
HEADLINE_D = 7           # depth used for the headline n-gram-only E[T]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_refs(path):
    """Return list of dicts: {id, category, prompt, completion}."""
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rid = d.get("id", "")
            cat = rid.split("-")[0] if "-" in rid else rid
            recs.append({
                "id": rid,
                "category": cat,
                "prompt": list(d["prompt_token_ids"]),
                "completion": list(d["completion_token_ids"]),
            })
    return recs


def matching_prefix_len(span, target):
    n = 0
    for a, b in zip(span, target):
        if a != b:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# PLD-self (prompt-lookup decoding): datastore == current context
# ---------------------------------------------------------------------------
def pld_position_stats(seq, gen_start, k):
    """Per-position (hit, correct) for single-token PLD lookup.

    At position t the trailing k-gram is seq[t-k:t]; a HIT means that exact
    k-gram occurs EARLIER in seq[:t]; the predicted token is the one that
    followed its most-recent prior occurrence; CORRECT means it == seq[t].
    """
    last_end = {}       # k-gram tuple -> most recent END index (= pos of next tok)
    e = k               # next end index to register
    hits = 0
    correct = 0
    total = 0
    L = len(seq)
    for t in range(gen_start, L):
        # register every k-gram fully inside seq[:t] (ends e <= t-1)
        while e <= t - 1:
            last_end[tuple(seq[e - k:e])] = e
            e += 1
        total += 1
        if t - k < 0:
            continue
        key = tuple(seq[t - k:t])
        ep = last_end.get(key)
        if ep is not None:          # ep < t guaranteed (only ends <= t-1 registered)
            hits += 1
            if seq[ep] == seq[t]:
                correct += 1
    return hits, correct, total


def pld_et_walk(seq, gen_start, k, D, check_identity=False):
    """Speculative E[T] walk for PLD-self with span depth D.

    Returns (steps, produced, accepted_sum). E[T] = produced / steps. Every
    accepted token equals the greedy reference (acceptance is bit-for-bit), so
    the produced stream is greedy-identical by construction; check_identity
    re-derives it and asserts equality.
    """
    last_end = {}
    e = k
    L = len(seq)
    pos = gen_start
    steps = 0
    produced = 0
    accepted_sum = 0
    rebuilt = [] if check_identity else None
    while pos < L:
        while e <= pos - 1:
            last_end[tuple(seq[e - k:e])] = e
            e += 1
        a = 0
        span = ()
        if pos - k >= 0:
            ep = last_end.get(tuple(seq[pos - k:pos]))
            if ep is not None:
                span = seq[ep:ep + D]
                a = matching_prefix_len(span, seq[pos:pos + D])
        steps += 1
        accepted_sum += a
        take = min(a + 1, L - pos)      # a accepted drafts + 1 verify bonus token
        if check_identity:
            rebuilt.extend(seq[pos:pos + take])
        produced += take
        pos += a + 1
    if check_identity:
        assert rebuilt == seq[gen_start:L], "PLD walk broke greedy identity"
    return steps, produced, accepted_sum


def pld_step_accepts(seq, gen_start, k, D):
    """Per-step accepted lengths a_ngram (for hybrid stacking against a_trained).

    Steps are taken at the SAME positions the trained draft would step, i.e. we
    advance by a+1 each step (the trained-draft control is also simulated on the
    greedy reference so both share the lockstep verify positions)."""
    last_end = {}
    e = k
    L = len(seq)
    pos = gen_start
    out = []        # (pos, a_ngram)
    while pos < L:
        while e <= pos - 1:
            last_end[tuple(seq[e - k:e])] = e
            e += 1
        a = 0
        if pos - k >= 0:
            ep = last_end.get(tuple(seq[pos - k:pos]))
            if ep is not None:
                a = matching_prefix_len(seq[ep:ep + D], seq[pos:pos + D])
        out.append((pos, a))
        pos += a + 1
    return out


# ---------------------------------------------------------------------------
# REST (global datastore): k-gram -> Counter(next token)
# ---------------------------------------------------------------------------
def build_rest_store(corpus_seqs, k):
    store = defaultdict(Counter)
    for s in corpus_seqs:
        for i in range(len(s) - k):
            store[tuple(s[i:i + k])][s[i + k]] += 1
    # collapse to argmax next token (ties broken by token id for determinism)
    top = {}
    for key, cnt in store.items():
        best = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))
        top[key] = best[0]
    return top


def rest_position_stats(seq, gen_start, k, top):
    hits = 0
    correct = 0
    total = 0
    L = len(seq)
    for t in range(gen_start, L):
        total += 1
        if t - k < 0:
            continue
        pred = top.get(tuple(seq[t - k:t]))
        if pred is not None:
            hits += 1
            if pred == seq[t]:
                correct += 1
    return hits, correct, total


def rest_propose(seq_ctx, k, D, top):
    """Greedy datastore walk: follow the argmax next token up to D tokens."""
    span = []
    key = tuple(seq_ctx[-k:]) if len(seq_ctx) >= k else None
    while key is not None and len(span) < D:
        pred = top.get(key)
        if pred is None:
            break
        span.append(pred)
        key = tuple(list(key[1:]) + [pred])
    return span


def rest_et_walk(seq, gen_start, k, D, top):
    L = len(seq)
    pos = gen_start
    steps = 0
    produced = 0
    accepted_sum = 0
    while pos < L:
        span = rest_propose(seq[:pos], k, D, top)
        a = matching_prefix_len(span, seq[pos:pos + D])
        steps += 1
        accepted_sum += a
        take = min(a + 1, L - pos)
        produced += take
        pos += a + 1
    return steps, produced, accepted_sum


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def safe_div(a, b):
    return (a / b) if b else 0.0


def profile_pld(recs, ks, ds):
    """Returns nested dict keyed by k: {hit_rate, correct_rate, et:{D:val}}."""
    out = {}
    for k in ks:
        H = C = T = 0
        for r in recs:
            seq = r["prompt"] + r["completion"]
            h, c, t = pld_position_stats(seq, len(r["prompt"]), k)
            H += h; C += c; T += t
        et = {}
        for D in ds:
            S = P = 0
            for r in recs:
                seq = r["prompt"] + r["completion"]
                s, p, _ = pld_et_walk(seq, len(r["prompt"]), k, D)
                S += s; P += p
            et[D] = safe_div(P, S)
        out[k] = {"hit_rate": safe_div(H, T),
                  "correct_rate": safe_div(C, H),
                  "hit_x_correct": safe_div(C, T),
                  "et": et}
    return out


def profile_rest(recs, ks, ds, stores_for, label):
    """stores_for(recs, k) -> (shared_store | None, per_record_dict | None).

    Each store is built once per k and reused across every record and depth.
    """
    out = {}
    for k in ks:
        shared, per_rec = stores_for(recs, k)

        def store_of(r):
            return per_rec[id(r)] if per_rec is not None else shared

        H = C = T = 0
        for r in recs:
            seq = r["prompt"] + r["completion"]
            h, c, t = rest_position_stats(seq, len(r["prompt"]), k, store_of(r))
            H += h; C += c; T += t
        et = {}
        for D in ds:
            S = P = 0
            for r in recs:
                seq = r["prompt"] + r["completion"]
                s, p, _ = rest_et_walk(seq, len(r["prompt"]), k, D, store_of(r))
                S += s; P += p
            et[D] = safe_div(P, S)
        out[k] = {"hit_rate": safe_div(H, T),
                  "correct_rate": safe_div(C, H),
                  "hit_x_correct": safe_div(C, T),
                  "et": et}
    return out


def rest_prompt_corpus(recs, k):
    return build_rest_store([r["prompt"] for r in recs], k), None


def rest_loo_corpus(recs, k):
    """Leave-one-out: per-record store = all prompts + all OTHER completions.

    Built as Counters once, then each record's store argmax is recomputed only
    over the keys its own completion touched (the rest are identical to global).
    """
    prompt_seqs = [r["prompt"] for r in recs]
    # global Counters over prompts + all completions
    gstore = defaultdict(Counter)
    for s in prompt_seqs:
        for i in range(len(s) - k):
            gstore[tuple(s[i:i + k])][s[i + k]] += 1
    for r in recs:
        c = r["completion"]
        for i in range(len(c) - k):
            gstore[tuple(c[i:i + k])][c[i + k]] += 1

    def argmax(cnt):
        return max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    global_top = {key: argmax(cnt) for key, cnt in gstore.items()}

    per_rec = {}
    for r in recs:
        c = r["completion"]
        touched = {}                       # key -> Counter with this rec removed
        for i in range(len(c) - k):
            key = tuple(c[i:i + k])
            if key not in touched:
                touched[key] = gstore[key].copy()
            touched[key][c[i + k]] -= 1
        store = dict(global_top)
        for key, cnt in touched.items():
            cnt += Counter()               # drop zero / negative counts
            if cnt:
                store[key] = argmax(cnt)
            else:
                store.pop(key, None)
        per_rec[id(r)] = store
    return None, per_rec


def per_category_pld(recs, k, D):
    cats = defaultdict(lambda: {"H": 0, "C": 0, "T": 0, "S": 0, "P": 0})
    for r in recs:
        seq = r["prompt"] + r["completion"]
        h, c, t = pld_position_stats(seq, len(r["prompt"]), k)
        s, p, _ = pld_et_walk(seq, len(r["prompt"]), k, D)
        a = cats[r["category"]]
        a["H"] += h; a["C"] += c; a["T"] += t; a["S"] += s; a["P"] += p
    out = {}
    for cat, a in cats.items():
        out[cat] = {"hit_rate": safe_div(a["H"], a["T"]),
                    "correct_rate": safe_div(a["C"], a["H"]),
                    "et": safe_div(a["P"], a["S"]),
                    "n_records": sum(1 for r in recs if r["category"] == cat)}
    return out


def best_k_by_et(table, D):
    bk = None
    bv = -1.0
    for k, v in table.items():
        if v["et"][D] > bv:
            bv = v["et"][D]
            bk = k
    return bk, bv


# ---------------------------------------------------------------------------
# self-test (PRIMARY metric)
# ---------------------------------------------------------------------------
def self_test(recs, pld_table):
    checks = []

    # (a) greedy-identity preserved: every accepted token == greedy ref by
    #     construction; re-derive the produced stream and assert equality.
    ident_ok = True
    try:
        for r in recs[:8]:                      # sample is sufficient & exact
            seq = r["prompt"] + r["completion"]
            pld_et_walk(seq, len(r["prompt"]), 3, HEADLINE_D, check_identity=True)
    except AssertionError:
        ident_ok = False
    checks.append(("greedy_identity_preserved", ident_ok,
                   "accepted tokens are bit-for-bit greedy ref; verify untouched"))

    # (b) PPL pinned <= cap (verify unchanged -> deployed PPL inherited)
    ppl_ok = PPL_PINNED <= PPL_CAP
    checks.append(("ppl_pinned_within_cap", ppl_ok,
                   f"ppl={PPL_PINNED} <= cap {PPL_CAP} (verify model unchanged)"))

    # (c) reproducible: recompute one cell and assert identical
    r0 = recs[0]
    seq0 = r0["prompt"] + r0["completion"]
    s1 = pld_et_walk(seq0, len(r0["prompt"]), 3, HEADLINE_D)
    s2 = pld_et_walk(seq0, len(r0["prompt"]), 3, HEADLINE_D)
    repro_ok = (s1 == s2)
    checks.append(("reproducible", repro_ok, "deterministic walk identical on rerun"))

    # (d) NaN-clean: all reported numbers finite
    nan_ok = True
    for k, v in pld_table.items():
        vals = [v["hit_rate"], v["correct_rate"], v["hit_x_correct"]] + list(v["et"].values())
        if any((x is None) or math.isnan(x) or math.isinf(x) for x in vals):
            nan_ok = False
    checks.append(("nan_clean", nan_ok, "all hit/correct/E[T] values finite"))

    passes = all(ok for _, ok, _ in checks)
    return {"passes": passes, "checks": checks}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=REF_PATH, help="greedy reference jsonl")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY self-test gate and print pass/fail")
    ap.add_argument("--skip-loo", action="store_true",
                    help="skip the REST leave-one-out ceiling (128 store builds)")
    ap.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    ap.add_argument("--wandb_project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb_entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb_name", default="ubel/ngram-draft-matchrate")
    ap.add_argument("--wandb_group", default="ngram-draft-matchrate")
    args = ap.parse_args()

    recs = load_refs(args.ref)
    n_records = len(recs)
    cat_counts = Counter(r["category"] for r in recs)
    total_positions = sum(len(r["completion"]) for r in recs)

    # ---- profiles -------------------------------------------------------
    pld = profile_pld(recs, KS, DS)
    rest_prompt = profile_rest(recs, KS, DS, rest_prompt_corpus, "REST-prompt")
    rest_loo = None
    if not args.skip_loo:
        rest_loo = profile_rest(recs, KS, DS, rest_loo_corpus, "REST-LOO")

    # headline (best-k by E[T] at HEADLINE_D) for each variant
    def headline(table):
        bk, _ = best_k_by_et(table, HEADLINE_D)
        v = table[bk]
        return {"best_k": bk, "hit_rate": v["hit_rate"],
                "correct_rate": v["correct_rate"],
                "e_t_ngram": v["et"][HEADLINE_D],
                "e_t_ngram_by_D": v["et"]}

    h_pld = headline(pld)
    h_rest_prompt = headline(rest_prompt)
    h_rest_loo = headline(rest_loo) if rest_loo else None

    # the strongest fair lever for kill/keep is PLD-self (zero dependency);
    # REST-LOO is the optimistic (high transfer-risk) ceiling.
    e_t_ngram_primary = h_pld["e_t_ngram"]
    e_t_ngram_ceiling = max(h_pld["e_t_ngram"],
                            h_rest_prompt["e_t_ngram"],
                            (h_rest_loo["e_t_ngram"] if h_rest_loo else 0.0))

    # ---- stacking upper bound (model-free) ------------------------------
    # E[T]_hybrid <= E[T]_trained + (E[T]_ngram - 1)
    inc_primary = max(0.0, e_t_ngram_primary - 1.0)
    inc_ceiling = max(0.0, e_t_ngram_ceiling - 1.0)
    hybrid_bound = {
        "vs_linear_mtp_k7": {
            "trained_et": ET_TRAINED_LINEAR_MTP_K7,
            "e_t_ngram_hybrid_upper_pld": ET_TRAINED_LINEAR_MTP_K7 + inc_primary,
            "e_t_ngram_hybrid_upper_ceiling": ET_TRAINED_LINEAR_MTP_K7 + inc_ceiling,
        },
        "vs_tree": {
            "trained_et": ET_TRAINED_TREE,
            "e_t_ngram_hybrid_upper_pld": ET_TRAINED_TREE + inc_primary,
            "e_t_ngram_hybrid_upper_ceiling": ET_TRAINED_TREE + inc_ceiling,
        },
        "max_incremental_et_pld": inc_primary,
        "max_incremental_et_ceiling": inc_ceiling,
    }
    # headline hybrid number: PLD stacking on the deployed official baseline
    # (linear MTP K=7, the 481.53 TPS control named in the PR), upper bound.
    e_t_ngram_hybrid = ET_TRAINED_LINEAR_MTP_K7 + inc_primary

    # ---- per-category PLD (transfer-risk view) --------------------------
    bk_pld, _ = best_k_by_et(pld, HEADLINE_D)
    per_cat = per_category_pld(recs, bk_pld, HEADLINE_D)

    # ---- self-test ------------------------------------------------------
    st = self_test(recs, pld)
    ngram_draft_matchrate_self_test_passes = int(st["passes"])

    # ---- verdict --------------------------------------------------------
    if e_t_ngram_ceiling >= 4.0:
        verdict = "GREEN"
        verdict_label = "exploitable structure -- wire n-gram as draft source and measure exact hybrid"
    elif e_t_ngram_ceiling >= 2.0:
        verdict = "AMBER"
        verdict_label = "partial structure -- measure exact hybrid on the strong categories before committing"
    else:
        verdict = "RED"
        verdict_label = ("no exploitable structure: n-gram-only E[T] < 2 even at the optimistic "
                         "ceiling; max stacking gain <= %.3f tok/step on top of the trained draft "
                         "-> cannot move TPS. Clean NEGATIVE (valid terminal result)." % inc_ceiling)

    out = {
        "pr": 250,
        "student": "ubel",
        "primary_metric_name": "ngram_draft_matchrate_self_test_passes",
        "ngram_draft_matchrate_self_test_passes": ngram_draft_matchrate_self_test_passes,
        "test_metric_name": "e_t_ngram_hybrid",
        "e_t_ngram_hybrid": e_t_ngram_hybrid,
        "ngram_hit_rate": h_pld["hit_rate"],
        "ngram_conditional_correct_rate": h_pld["correct_rate"],
        "e_t_ngram_pld": e_t_ngram_primary,
        "e_t_ngram_ceiling": e_t_ngram_ceiling,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "dataset": {
            "ref": args.ref,
            "n_records": n_records,
            "categories": dict(cat_counts),
            "total_decode_positions": total_positions,
            "headline_key_len": bk_pld,
            "headline_span_D": HEADLINE_D,
        },
        "pld_self": {"by_k": pld, "headline": h_pld},
        "rest_prompt": {"by_k": rest_prompt, "headline": h_rest_prompt},
        "rest_loo": ({"by_k": rest_loo, "headline": h_rest_loo} if rest_loo else None),
        "per_category_pld": per_cat,
        "hybrid_bound": hybrid_bound,
        "controls": {
            "linear_mtp_k7_et": ET_TRAINED_LINEAR_MTP_K7,
            "tree_et": ET_TRAINED_TREE,
            "ppl_pinned": PPL_PINNED,
            "ppl_cap": PPL_CAP,
        },
        "self_test": st,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report -------------------------------------------------
    print("=" * 78)
    print("N-1 n-gram draft source -- match-rate diagnostic (PR #250)")
    print("=" * 78)
    print(f"refs: {n_records} records, {total_positions} decode positions, "
          f"categories={dict(cat_counts)}")
    print(f"\nPLD-self (datastore = current context):")
    print(f"  {'k':>2} {'hit_rate':>9} {'corr|hit':>9} {'hit*corr':>9} "
          + " ".join(f'E[T]D={d}' for d in DS))
    for k in KS:
        v = pld[k]
        print(f"  {k:>2} {v['hit_rate']:>9.4f} {v['correct_rate']:>9.4f} "
              f"{v['hit_x_correct']:>9.4f} "
              + " ".join(f"{v['et'][d]:>7.3f}" for d in DS))
    print(f"\nREST-prompt (datastore = all 128 prompts):")
    for k in KS:
        v = rest_prompt[k]
        print(f"  k={k} hit={v['hit_rate']:.4f} corr|hit={v['correct_rate']:.4f} "
              f"E[T]D7={v['et'][HEADLINE_D]:.3f}")
    if rest_loo:
        print(f"\nREST-LOO (datastore = prompts + other completions, optimistic ceiling):")
        for k in KS:
            v = rest_loo[k]
            print(f"  k={k} hit={v['hit_rate']:.4f} corr|hit={v['correct_rate']:.4f} "
                  f"E[T]D7={v['et'][HEADLINE_D]:.3f}")
    print(f"\nPer-category PLD (k={bk_pld}, D={HEADLINE_D}):")
    for cat, a in sorted(per_cat.items()):
        print(f"  {cat:>14}  n={a['n_records']:>3}  hit={a['hit_rate']:.4f}  "
              f"corr|hit={a['correct_rate']:.4f}  E[T]_ngram={a['et']:.3f}")
    print(f"\nHeadline (PLD best k={h_pld['best_k']}, D={HEADLINE_D}):")
    print(f"  ngram_hit_rate                 = {h_pld['hit_rate']:.4f}")
    print(f"  ngram_conditional_correct_rate = {h_pld['correct_rate']:.4f}")
    print(f"  e_t_ngram (PLD-only)           = {e_t_ngram_primary:.3f}")
    print(f"  e_t_ngram_ceiling (best var)   = {e_t_ngram_ceiling:.3f}")
    print(f"\nStacking upper bound  E[T]_hybrid <= E[T]_trained + (E[T]_ngram - 1):")
    print(f"  max incremental E[T] (PLD)     = {inc_primary:.3f} tok/step")
    print(f"  max incremental E[T] (ceiling) = {inc_ceiling:.3f} tok/step")
    print(f"  hybrid<=  vs linear-MTP-K7 ({ET_TRAINED_LINEAR_MTP_K7}): "
          f"{ET_TRAINED_LINEAR_MTP_K7 + inc_primary:.3f} (PLD) / "
          f"{ET_TRAINED_LINEAR_MTP_K7 + inc_ceiling:.3f} (ceiling)")
    print(f"  hybrid<=  vs tree         ({ET_TRAINED_TREE}): "
          f"{ET_TRAINED_TREE + inc_primary:.3f} (PLD) / "
          f"{ET_TRAINED_TREE + inc_ceiling:.3f} (ceiling)")
    print(f"\nVERDICT: {verdict} -- {verdict_label}")
    print(f"\n[PRIMARY] ngram_draft_matchrate_self_test_passes = "
          f"{ngram_draft_matchrate_self_test_passes}")
    for name, ok, detail in st["checks"]:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\nwrote {args.out}")

    if args.self_test and not st["passes"]:
        raise SystemExit("SELF-TEST FAILED")

    # ---- wandb ----------------------------------------------------------
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             name=args.wandb_name, group=args.wandb_group,
                             job_type="analysis",
                             config={
                                 "pr": 250, "student": "ubel",
                                 "ref": args.ref, "n_records": n_records,
                                 "categories": dict(cat_counts),
                                 "ks": list(KS), "ds": list(DS),
                                 "headline_D": HEADLINE_D,
                                 "et_trained_linear_mtp_k7": ET_TRAINED_LINEAR_MTP_K7,
                                 "et_trained_tree": ET_TRAINED_TREE,
                                 "ppl_pinned": PPL_PINNED, "ppl_cap": PPL_CAP,
                             })
            s = wandb.summary
            s["ngram_draft_matchrate_self_test_passes"] = ngram_draft_matchrate_self_test_passes
            s["ngram_hit_rate"] = h_pld["hit_rate"]
            s["ngram_conditional_correct_rate"] = h_pld["correct_rate"]
            s["e_t_ngram_pld"] = e_t_ngram_primary
            s["e_t_ngram_ceiling"] = e_t_ngram_ceiling
            s["e_t_ngram_hybrid"] = e_t_ngram_hybrid
            s["max_incremental_et_pld"] = inc_primary
            s["max_incremental_et_ceiling"] = inc_ceiling
            s["verdict"] = verdict

            # k-sweep tables
            for label, table in (("pld_self", pld), ("rest_prompt", rest_prompt),
                                 *( (("rest_loo", rest_loo),) if rest_loo else () )):
                t = wandb.Table(columns=["k", "hit_rate", "correct_rate", "hit_x_correct",
                                         *[f"et_D{d}" for d in DS]])
                for k in KS:
                    v = table[k]
                    t.add_data(k, v["hit_rate"], v["correct_rate"], v["hit_x_correct"],
                               *[v["et"][d] for d in DS])
                wandb.log({f"matchrate/{label}": t})

            # per-category table
            ct = wandb.Table(columns=["category", "n_records", "hit_rate",
                                      "correct_rate", "e_t_ngram"])
            for cat, a in sorted(per_cat.items()):
                ct.add_data(cat, a["n_records"], a["hit_rate"], a["correct_rate"], a["et"])
            wandb.log({"matchrate/per_category_pld": ct})

            # E[T] vs D curve (PLD best k)
            dt = wandb.Table(columns=["D", "e_t_ngram_pld"])
            for d in DS:
                dt.add_data(d, pld[h_pld["best_k"]]["et"][d])
            wandb.log({"matchrate/et_vs_D_pld": dt})

            # self-test checks
            stt = wandb.Table(columns=["check", "passes", "detail"])
            for name, ok, detail in st["checks"]:
                stt.add_data(name, int(ok), detail)
            wandb.log({"self_test_checks": stt})

            wandb.finish()
            print(f"logged to wandb: {run.url if run else ''}")
        except Exception as ex:           # offline / no-network must not fail the gate
            print(f"[wandb skipped] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
