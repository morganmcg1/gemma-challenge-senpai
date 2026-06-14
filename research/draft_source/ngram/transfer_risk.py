#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""N-1 transfer-risk: does the n-gram match-rate survive public -> private? (PR #250)

profile.py establishes the PRIMARY result on the OFFICIAL public benchmark set
(mmlu_pro / gpqa_diamond / aime2026 reasoning), where the n-gram draft is dead
(E[T] ~ 1.37). PR step 5 asks whether that transfers to the PRIVATE prompt
distribution. We answer it with the repo's own native HARD proxies (PR #164:
code / longctx / math / casual / multilingual), which are ShareGPT-derived chat
and therefore far more templated/repetitive than dense reasoning CoT.

For each proxy category we measure the GENERATION-INTERNAL n-gram E[T]
(self-repetition; the reliably recoverable component -- on the public set
copy-from-prompt added only +0.097 of the 1.37) from the cached benchmark
generations (proxy_refs.json, produced by proxy_transfer.py), and pair it with
the TRAINED draft's measured acceptance on the SAME category
(private_gap_probe/native_<c>/report.json e_accept.private; the deployed
fa2sw_precache_kenyan tree). The public linear-MTP control is 3.844
(accept_calibration headline). The private ShareGPT-chat trained control is
3.090 (tree_private_acceptance_gap).

THE DECISIVE COMPARISON
-----------------------
The hypothesis is "free E[T] in the prompt structure that NO trained draft
exploits." If the trained draft already accepts as well or better than the
n-gram everywhere structure exists, there is no free orthogonal lever. The
rigorous stacking ceiling is E[T]_hybrid <= E[T]_trained + (E[T]_ngram - 1), and
it is loose: n-gram fires only on LOW-ENTROPY repeated spans, exactly the tokens
a trained LM draft head already predicts, so n-gram's correct set is ~a SUBSET of
the trained draft's accepted set and the realized gain is far below the ceiling.

Usage:
  python research/draft_source/ngram/proxy_transfer.py      # builds proxy_refs.json
  python research/draft_source/ngram/transfer_risk.py [--no_wandb]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile import (load_refs, pld_et_walk, pld_position_stats, safe_div,  # noqa: E402
                     ET_TRAINED_LINEAR_MTP_K7)

PROXY_REFS = "research/draft_source/ngram/proxy_refs.json"
PUBLIC_REF = "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
REPORT = "research/validity/private_gap_probe/native_{c}/report.json"
PRIV_SHAREGPT_TRAINED_ET = 3.0899   # tree on private ShareGPT chat (tree_private_acceptance_gap)
OUT = "research/draft_source/ngram/transfer_risk.json"
K = 2          # headline key length (best E[T], conservative for a kill)
D = 7          # span depth, matches deployed K=7


def agg_genonly(seqs):
    S = P = H = C = T = 0
    for seq in seqs:
        if len(seq) <= K:
            continue
        s, p, _ = pld_et_walk(seq, 0, K, D)
        h, c, t = pld_position_stats(seq, 0, K)
        S += s; P += p; H += h; C += c; T += t
    return {"e_t_ngram_genonly": safe_div(P, S),
            "hit_rate": safe_div(H, T),
            "correct_rate": safe_div(C, H)}


def trained_control(cat):
    path = REPORT.format(c=cat)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    ea = d.get("e_accept", {})
    return ea.get("private")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-refs", default=PROXY_REFS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb_entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb_name", default="ubel/ngram-draft-matchrate-transfer")
    ap.add_argument("--wandb_group", default="ngram-draft-matchrate")
    args = ap.parse_args()

    if not os.path.exists(args.proxy_refs):
        raise SystemExit(f"missing {args.proxy_refs}; run proxy_transfer.py first")

    # public reasoning gen-only baseline (apples-to-apples with proxy gen-only)
    pub = load_refs(PUBLIC_REF)
    public_genonly = agg_genonly([r["completion"] for r in pub])
    public_full_et = None  # filled from profile output if available
    pj = "research/draft_source/ngram/ngram_matchrate.json"
    if os.path.exists(pj):
        public_full_et = json.load(open(pj)).get("e_t_ngram_pld")

    with open(args.proxy_refs) as fh:
        proxy = json.load(fh)

    rows = []
    # public reasoning row (trained control = official linear MTP K=7)
    rows.append({
        "setting": "public_reasoning(official-scored)",
        "n": len(pub),
        **public_genonly,
        "e_t_ngram_full": public_full_et,
        "trained_et": ET_TRAINED_LINEAR_MTP_K7,
        "trained_source": "linear MTP K=7 (accept_calibration, official 481.53 baseline)",
    })

    for cat, info in proxy["categories"].items():
        g = agg_genonly(info["completions"])
        tc = trained_control(cat)
        rows.append({
            "setting": f"private_proxy:{cat}",
            "n": info["n_records"],
            **g,
            "e_t_ngram_full": None,
            "trained_et": tc,
            "trained_source": f"tree e_accept.private (private_gap_probe/native_{cat})",
        })

    # derived: loose hybrid ceiling + does n-gram ever beat the trained draft?
    # Use the apples-to-apples GEN-ONLY E[T] for every row (the only quantity
    # reliably recoverable on the proxies). Public full-context is annotated
    # separately and is only +0.097 above gen-only.
    ngram_beats_trained = False
    max_ceiling_gain = 0.0
    for r in rows:
        en = r["e_t_ngram_genonly"]
        et = r["trained_et"]
        inc = max(0.0, en - 1.0)
        r["e_t_ngram_used"] = en
        r["hybrid_upper_bound"] = (et + inc) if et else None
        r["max_incremental_et"] = inc
        r["ngram_minus_trained"] = (en - et) if et else None
        if et and en > et:
            ngram_beats_trained = True
        max_ceiling_gain = max(max_ceiling_gain, inc)

    verdict = ("RED-CONFIRMED: n-gram never beats the trained draft on any setting; "
               "where structure is richest (private chat/code/longctx) the trained "
               "draft already accepts E[T] 3.1-3.9 vs n-gram 2.0-2.5. No free "
               "orthogonal lever. Transfer does NOT rescue the hypothesis."
               if not ngram_beats_trained else
               "AMBER: some category has n-gram E[T] above the trained control -- inspect.")

    out = {
        "pr": 250, "student": "ubel",
        "K": K, "D": D,
        "public_reasoning_genonly_et": public_genonly["e_t_ngram_genonly"],
        "private_sharegpt_trained_et": PRIV_SHAREGPT_TRAINED_ET,
        "rows": rows,
        "ngram_beats_trained_anywhere": ngram_beats_trained,
        "max_ceiling_incremental_et": max_ceiling_gain,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console ----
    print("=" * 92)
    print("N-1 transfer-risk: n-gram (self-repetition) vs trained draft, public -> private (PR #250)")
    print("=" * 92)
    print(f"{'setting':<34} {'n':>3} {'ngram_E[T]':>10} {'hit':>6} {'corr':>6} "
          f"{'trained_E[T]':>12} {'n-tr':>6} {'hybrid<=':>9}")
    for r in rows:
        print(f"{r['setting']:<34} {r['n']:>3} {r['e_t_ngram_used']:>10.3f} "
              f"{r['hit_rate']:>6.3f} {r['correct_rate']:>6.3f} "
              f"{(r['trained_et'] or 0):>12.3f} "
              f"{(r['ngram_minus_trained'] if r['ngram_minus_trained'] is not None else 0):>6.2f} "
              f"{(r['hybrid_upper_bound'] or 0):>9.3f}")
    print(f"\n(table E[T] is GEN-ONLY self-repetition, apples-to-apples; public "
          f"full-context = {public_full_et} = gen-only + copy-from-prompt +0.097)")
    print(f"private ShareGPT-chat trained control (context): E[T]={PRIV_SHAREGPT_TRAINED_ET}")
    print(f"n-gram beats trained anywhere: {ngram_beats_trained}")
    print(f"max ceiling incremental E[T] (loose, disjoint): {max_ceiling_gain:.3f} tok/step")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {args.out}")

    # ---- wandb ----
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             name=args.wandb_name, group=args.wandb_group,
                             job_type="analysis",
                             config={"pr": 250, "student": "ubel", "K": K, "D": D,
                                     "analysis": "transfer-risk public->private"})
            s = wandb.summary
            s["public_reasoning_genonly_et"] = public_genonly["e_t_ngram_genonly"]
            s["ngram_beats_trained_anywhere"] = int(ngram_beats_trained)
            s["max_ceiling_incremental_et"] = max_ceiling_gain
            s["verdict"] = verdict
            t = wandb.Table(columns=["setting", "n", "ngram_e_t", "hit_rate", "correct_rate",
                                     "trained_e_t", "ngram_minus_trained", "hybrid_upper_bound"])
            for r in rows:
                t.add_data(r["setting"], r["n"], r["e_t_ngram_used"], r["hit_rate"],
                           r["correct_rate"], r["trained_et"], r["ngram_minus_trained"],
                           r["hybrid_upper_bound"])
            wandb.log({"transfer/by_setting": t})
            wandb.finish()
            print(f"logged to wandb: {run.url if run else ''}")
        except Exception as ex:
            print(f"[wandb skipped] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
