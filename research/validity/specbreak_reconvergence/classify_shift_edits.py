#!/usr/bin/env python3
"""Shift-edit content audit (PR #689, analysis_only).

Mechanism "why" behind #686's answer_flips=0: #686 found 82% of strict-#319 spec
breaks reconverge, but via SHIFTED ~6-token insert/delete edits (median |shift|=
6.5), not clean same-position tie-wobbles. The decisive question for the human's
strict-#319 policy: are those shift edits COSMETIC (whitespace / punctuation /
markdown-or-latex delimiters / number-formatting / equivalent-token) -> the break
is score-immaterial by construction; or CONTENTFUL (different words, numbers,
operators, reasoning steps) -> reconvergence over-states score-safety.

This is pure CPU tokenizer-level analysis on the already-generated #678/#673
confirm decodes (decode_{ar,suffix6,ngram5}_r0.jsonl) + #686's reconvergence_
report.json. NO GPU, NO HF Job, NO re-decode, NO model download (Gemma tokenizer
is loaded offline from the local HF cache).

Outputs (KB-scale):
  * shift_edit_classification.json  -- aggregate per-class counts + cosmetic_edit_
    frac (shifted heals) + permfork_content_divergence_frac (permanent forks),
    per cell / pooled / dedup / per-source.
  * fork_risk_manifest.jsonl        -- per break-prompt: its class (cosmetic-heal /
    content-heal / format-fork / content-fork) + decoded snippet + key numbers,
    for wirbel #682 per-prompt retention cross-check.

Edit-region definition (matches #686's shift_tolerant_resync): after the first
divergence p, recompute difflib's first contiguous matching block of size >=W on
the post-divergence tails A[p:], B[p:]. That block at tail offsets (first.a,
first.b) is the re-couple q. The edit region (the ~6-token excursion) is then
  AR side : A[p : p+first.a]
  SP side : B[p : p+first.b]
and the edit is "transform the AR excursion into the SP excursion". One side may
be empty (pure insertion / deletion). We decode both with the Gemma tokenizer and
classify the *content* of the difference.
"""
import argparse
import json
import os
import re
import difflib
import statistics
from collections import defaultdict, Counter

CONFIRM = "research/validity/specdec_official_dist_breakrate/_runs/confirm"
OUTDIR = "research/validity/specbreak_reconvergence"
DEFAULT_TOK = ("/senpai-run/home/student-kanna/.cache/huggingface/hub/"
               "models--google--gemma-4-E4B-it/snapshots/"
               "fee6332c1abaafb77f6f9624236c63aa2f1d0187")
HEADLINE_W = 16
EOS_TOKENS = {106, 1}  # <end_of_turn>, <eos>
ANS_MC = re.compile(r"ANSWER:\s*\(?([A-J])\b", re.IGNORECASE)
ANS_NUM = re.compile(r"ANSWER:\s*\(?(-?\d[\d,]*\.?\d*)")

# Perm-fork content-divergence threshold: a fork whose two tails share <=this
# fraction of content (difflib seq-ratio on lowercased alnum word streams, and
# Jaccard of the content-word sets both below the gate) is a genuine CONTENT_FORK;
# above it the streams discuss the same entities/numbers in different
# order/verbosity (FORMAT_LENGTH_FORK). Tuned by inspection of the ~dozen forks.
FORK_SEQ_GATE = 0.50
FORK_JACCARD_GATE = 0.60


# --------------------------------------------------------------------------- #
# string normalisation helpers
# --------------------------------------------------------------------------- #
def strip_latex(s):
    """Drop LaTeX/markdown control sequences so only literal content remains.

    \\text \\frac \\times ... -> space ; \\( \\) \\[ \\] \\\\ ... -> space.
    Leaves the alphanumeric *content* inside (e.g. \\text{CH}_3 -> 'CH 3')."""
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"\\.", " ", s)
    return s


def alnum_residual(s):
    """Alphanumeric content only: strips ALL whitespace, punctuation, markdown,
    and LaTeX delimiters. Two edits with equal residual differ ONLY in
    cosmetic (whitespace/punct/markdown/latex) characters."""
    return re.sub(r"[^A-Za-z0-9]", "", strip_latex(s))


def numval(s):
    """Single numeric value of s if (after latex strip) it is purely ONE number
    (digits, one dot, optional sign, thousands commas, currency/ws/punct around
    it); else None. Crucially returns None when ANY letter remains -- prose that
    merely contains digits (e.g. 'nucleon (1) ... (2)') is NOT a number-format
    edit. '0.5'->0.5  '.5'->0.5  '$5'->5  '1,000'->1000  ' mathematical'->None
    'nucleon (1)...(2)'->None."""
    s2 = strip_latex(s)
    if re.search(r"[A-Za-z]", s2):        # prose with scattered digits is not a number
        return None
    s2 = s2.replace(",", "")
    s2 = re.sub(r"[^0-9.+\-]", "", s2)
    if re.fullmatch(r"[-+]?\d*\.?\d+", s2):
        try:
            return float(s2)
        except ValueError:
            return None
    return None


def classify_edit(ar_str, sp_str):
    """Cosmetic-vs-content label for one shifted-heal edit (AR excursion ->
    SP excursion). Precedence: text-identical > whitespace/punct/markdown/latex
    > number-format > case-equivalent > content."""
    a, b = ar_str, sp_str
    if a == b:                                   # same text, diff token-ids
        return "EQUIVALENT_TOKEN"
    ra, rb = alnum_residual(a), alnum_residual(b)
    if ra == rb:                                 # only ws/punct/markdown/latex differ
        return "WHITESPACE_PUNCT"
    na, nb = numval(a), numval(b)
    if na is not None and nb is not None and na == nb:
        return "NUMBER_FORMAT"                   # same value, diff surface
    if ra.lower() == rb.lower():                 # case-only
        return "EQUIVALENT_TOKEN"
    return "CONTENT"


COSMETIC = {"WHITESPACE_PUNCT", "NUMBER_FORMAT", "EQUIVALENT_TOKEN"}


def content_words(text):
    return re.findall(r"[a-z0-9]+", strip_latex(text).lower())


def extract_answer(text):
    m = ANS_MC.search(text)
    if m:
        return ("mc", m.group(1).upper())
    m = ANS_NUM.search(text)
    if m:
        return ("num", m.group(1).replace(",", ""))
    return None


# --------------------------------------------------------------------------- #
# difflib re-sync (same logic as #686 analyze_reconvergence.shift_tolerant_resync)
# --------------------------------------------------------------------------- #
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


def first_resync_block(A, B, p, W):
    """First >=W matching block (min onset) on tails A[p:], B[p:]. Returns
    (first_a, first_b) tail offsets, or None if no >=W block (permanent fork)."""
    At, Bt = A[p:], B[p:]
    sm = difflib.SequenceMatcher(a=At, b=Bt, autojunk=False)
    big = [b for b in sm.get_matching_blocks() if b.size >= W]
    if not big:
        return None
    first = min(big, key=lambda b: min(b.a, b.b))
    return first.a, first.b


# --------------------------------------------------------------------------- #
# main classification
# --------------------------------------------------------------------------- #
def load_decodes():
    ar = {json.loads(l)["id"]: json.loads(l)
          for l in open(f"{CONFIRM}/decode_ar_r0.jsonl")}
    cells = {}
    for name, fn in [("suffix6", "decode_suffix6_r0.jsonl"),
                     ("ngram5", "decode_ngram5_r0.jsonl")]:
        cells[name] = {json.loads(l)["id"]: json.loads(l)
                       for l in open(f"{CONFIRM}/{fn}")}
    return ar, cells


def classify_shift_break(tok, ar_rec, sp_rec, rep_rec, W):
    """Classify one SHIFTED_HEAL break. Returns dict with edit class + decoded
    snippets + the (first_a, first_b) excursion lengths."""
    A = ar_rec["completion_token_ids"]
    S = sp_rec["completion_token_ids"]
    p = rep_rec["p"]
    blk = first_resync_block(A, S, p, W)
    if blk is None:           # should not happen for a SHIFTED_HEAL
        return None
    fa, fb = blk
    ar_ed_ids = A[p:p + fa]
    sp_ed_ids = S[p:p + fb]
    ar_ed = tok.decode(ar_ed_ids) if ar_ed_ids else ""
    sp_ed = tok.decode(sp_ed_ids) if sp_ed_ids else ""
    klass = classify_edit(ar_ed, sp_ed)
    # cross-check vs #686 stored shift/onset
    rep_fb = (rep_rec["shift_onset"] - p) if rep_rec["shift_onset"] is not None else None
    rep_shift = rep_rec["shift"]
    consistent = (rep_fb == fb) and (rep_shift == (fa - fb))
    return dict(
        edit_class=klass, cosmetic=klass in COSMETIC,
        ar_edit=ar_ed, sp_edit=sp_ed,
        first_a=fa, first_b=fb, shift=fa - fb,
        excursion_len=max(fa, fb),
        report_consistent=consistent,
    )


def classify_perm_fork(tok, ar_rec, sp_rec, rep_rec, W):
    """Classify one PERMANENT_FORK: CONTENT_FORK vs FORMAT_LENGTH_FORK on the
    post-divergence tails (capped at each stream's own EOS to drop forced
    filler)."""
    A = ar_rec["completion_token_ids"]
    S = sp_rec["completion_token_ids"]
    p = rep_rec["p"]
    a_eos = first_eos(A)
    s_eos = first_eos(S)
    a_end = a_eos if a_eos is not None else len(A)
    s_end = s_eos if s_eos is not None else len(S)
    ar_tail_ids = A[p:a_end]
    sp_tail_ids = S[p:s_end]
    ar_tail = tok.decode(ar_tail_ids) if ar_tail_ids else ""
    sp_tail = tok.decode(sp_tail_ids) if sp_tail_ids else ""
    wa, wb = content_words(ar_tail), content_words(sp_tail)
    seq_ratio = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).ratio()
    sa, sb = set(wa), set(wb)
    jaccard = (len(sa & sb) / len(sa | sb)) if (sa | sb) else 1.0
    ar_ans = extract_answer(ar_rec["generated_text"])
    sp_ans = extract_answer(sp_rec["generated_text"])
    both_commit = (ar_ans is not None) and (sp_ans is not None)
    answer_match = both_commit and (ar_ans == sp_ans)
    # decision: answer-preserving, or high content overlap => format/length fork
    if answer_match:
        klass = "FORMAT_LENGTH_FORK"
    elif seq_ratio >= FORK_SEQ_GATE or jaccard >= FORK_JACCARD_GATE:
        klass = "FORMAT_LENGTH_FORK"
    else:
        klass = "CONTENT_FORK"
    return dict(
        fork_class=klass, content_fork=klass == "CONTENT_FORK",
        seq_ratio=round(seq_ratio, 3), jaccard=round(jaccard, 3),
        both_commit=both_commit, answer_match=answer_match,
        ar_ans=ar_ans, sp_ans=sp_ans,
        tail_len_ar=len(ar_tail_ids), tail_len_sp=len(sp_tail_ids),
        ar_tail_head=tok.decode(ar_tail_ids[:36]) if ar_tail_ids else "",
        sp_tail_head=tok.decode(sp_tail_ids[:36]) if sp_tail_ids else "",
    )


def frac(x, d):
    return (x / d) if d else float("nan")


def summarize_shift(records):
    cnt = Counter(r["edit_class"] for r in records)
    n = len(records)
    cosmetic = sum(1 for r in records if r["cosmetic"])
    return dict(
        n=n, classes=dict(cnt),
        cosmetic=cosmetic, content=n - cosmetic,
        cosmetic_edit_frac=round(frac(cosmetic, n), 4),
        median_excursion=(statistics.median([r["excursion_len"] for r in records
                                             if "excursion_len" in r])
                          if any("excursion_len" in r for r in records) else None),
    )


def summarize_fork(records):
    cnt = Counter(r["fork_class"] for r in records)
    n = len(records)
    content = sum(1 for r in records if r["content_fork"])
    return dict(
        n=n, classes=dict(cnt),
        content_fork=content, format_length_fork=n - content,
        permfork_content_divergence_frac=round(frac(content, n), 4),
    )


def by_source(records, summ):
    out = {}
    bs = defaultdict(list)
    for r in records:
        bs[r["source"]].append(r)
    for src, rows in sorted(bs.items()):
        out[src] = summ(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=f"{OUTDIR}/reconvergence_report.json")
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--W", type=int, default=HEADLINE_W)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rep = json.load(open(args.report))
    ar, cells = load_decodes()

    shift_recs = defaultdict(list)   # cell -> [record]
    fork_recs = defaultdict(list)
    samepos_recs = defaultdict(list)
    manifest = []

    for cell in ("suffix6", "ngram5"):
        pp = rep["cells"][cell][f"per_prompt_W{args.W}"]
        for rr in pp:
            pid = rr["id"]
            ar_rec, sp_rec = ar[pid], cells[cell][pid]
            base = dict(id=pid, source=rr["source"], cell=cell, p=rr["p"],
                        klass686=rr["klass"])
            if rr["klass"] == "SHIFTED_HEAL":
                c = classify_shift_break(tok, ar_rec, sp_rec, rr, args.W)
                if c is None:
                    continue
                rec = {**base, **c, "source": rr["source"]}
                shift_recs[cell].append(rec)
                manifest.append(dict(
                    id=pid, source=rr["source"], cell=cell, p=rr["p"],
                    break_klass="SHIFTED_HEAL",
                    final_class=("cosmetic-heal" if c["cosmetic"] else "content-heal"),
                    edit_class=c["edit_class"], shift=c["shift"],
                    excursion_len=c["excursion_len"],
                    ar_edit=c["ar_edit"], sp_edit=c["sp_edit"]))
            elif rr["klass"] == "PERMANENT_FORK":
                c = classify_perm_fork(tok, ar_rec, sp_rec, rr, args.W)
                rec = {**base, **c}
                fork_recs[cell].append(rec)
                manifest.append(dict(
                    id=pid, source=rr["source"], cell=cell, p=rr["p"],
                    break_klass="PERMANENT_FORK",
                    final_class=("content-fork" if c["content_fork"] else "format-fork"),
                    fork_class=c["fork_class"], seq_ratio=c["seq_ratio"],
                    jaccard=c["jaccard"], answer_match=c["answer_match"],
                    ar_tail_head=c["ar_tail_head"], sp_tail_head=c["sp_tail_head"]))
            elif rr["klass"] == "SAMEPOS_HEAL":
                # bonus: classify the same-position single-token wobble too
                A = ar_rec["completion_token_ids"]
                S = sp_rec["completion_token_ids"]
                p = rr["p"]
                q = rr["samepos_q"]
                ar_ed = tok.decode(A[p:q]) if q and q > p else tok.decode([A[p]])
                sp_ed = tok.decode(S[p:q]) if q and q > p else tok.decode([S[p]])
                kl = classify_edit(ar_ed, sp_ed)
                samepos_recs[cell].append(dict(**base, edit_class=kl,
                                               cosmetic=kl in COSMETIC,
                                               ar_edit=ar_ed, sp_edit=sp_ed))

    # ----- aggregates ----- #
    out = dict(
        pr=689, analysis_only=True, official_tps=0, fires=False,
        model="google/gemma-4-E4B-it", headline_W=args.W,
        tokenizer_snapshot=args.tokenizer,
        edit_classes=["WHITESPACE_PUNCT", "NUMBER_FORMAT", "EQUIVALENT_TOKEN", "CONTENT"],
        cosmetic_classes=sorted(COSMETIC),
        fork_gate=dict(seq=FORK_SEQ_GATE, jaccard=FORK_JACCARD_GATE),
        shift_edit={}, perm_fork={}, samepos_heal={},
    )

    pooled_shift, pooled_fork = [], []
    for cell in ("suffix6", "ngram5"):
        out["shift_edit"][cell] = {**summarize_shift(shift_recs[cell]),
                                   "per_source": by_source(shift_recs[cell], summarize_shift)}
        out["perm_fork"][cell] = {**summarize_fork(fork_recs[cell]),
                                  "per_source": by_source(fork_recs[cell], summarize_fork)}
        out["samepos_heal"][cell] = summarize_shift(samepos_recs[cell])
        pooled_shift += shift_recs[cell]
        pooled_fork += fork_recs[cell]

    out["shift_edit"]["pooled"] = {**summarize_shift(pooled_shift),
                                   "per_source": by_source(pooled_shift, summarize_shift)}
    out["perm_fork"]["pooled"] = {**summarize_fork(pooled_fork),
                                  "per_source": by_source(pooled_fork, summarize_fork)}

    # dedup by prompt-id: worst-case (a prompt is content if ANY cell is content)
    def dedup_shift(recs):
        by = defaultdict(list)
        for r in recs:
            by[r["id"]].append(r)
        ded = []
        for pid, rows in by.items():
            worst = max(rows, key=lambda r: (0 if r["cosmetic"] else 1))
            ded.append(worst)
        return ded

    def dedup_fork(recs):
        by = defaultdict(list)
        for r in recs:
            by[r["id"]].append(r)
        ded = []
        for pid, rows in by.items():
            worst = max(rows, key=lambda r: (1 if r["content_fork"] else 0))
            ded.append(worst)
        return ded

    ds = dedup_shift(pooled_shift)
    df = dedup_fork(pooled_fork)
    out["shift_edit"]["dedup"] = {**summarize_shift(ds),
                                  "per_source": by_source(ds, summarize_shift)}
    out["perm_fork"]["dedup"] = {**summarize_fork(df),
                                 "per_source": by_source(df, summarize_fork)}

    # headline scalars
    cos_pooled = out["shift_edit"]["pooled"]["cosmetic_edit_frac"]
    cfork_pooled = out["perm_fork"]["pooled"]["permfork_content_divergence_frac"]
    if cos_pooled >= 0.7 and cfork_pooled <= 0.4:
        verdict = "SHIFT_EDITS_COSMETIC"
    elif cos_pooled <= 0.4:
        verdict = "SHIFT_EDITS_CONTENTFUL"
    else:
        verdict = "MIXED"
    out["headline"] = dict(
        cosmetic_edit_frac_pooled=cos_pooled,
        cosmetic_edit_frac_dedup=out["shift_edit"]["dedup"]["cosmetic_edit_frac"],
        permfork_content_divergence_frac_pooled=cfork_pooled,
        permfork_content_divergence_frac_dedup=out["perm_fork"]["dedup"]["permfork_content_divergence_frac"],
        n_shift_pooled=len(pooled_shift), n_fork_pooled=len(pooled_fork),
        n_shift_dedup=len(ds), n_fork_dedup=len(df),
        verdict=verdict,
    )

    with open(f"{OUTDIR}/shift_edit_classification.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(f"{OUTDIR}/fork_risk_manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    # ----- console report ----- #
    consistent = all(r.get("report_consistent", True) for r in pooled_shift)
    print(f"report_consistent (recomputed re-sync == #686 stored): {consistent}")
    print(f"\n{'='*72}\nSHIFTED-HEAL EDIT CLASSIFICATION  (W={args.W})\n{'='*72}")
    for cell in ("suffix6", "ngram5", "pooled", "dedup"):
        s = out["shift_edit"][cell]
        print(f"  {cell:8s} n={s['n']:2d}  cosmetic={s['cosmetic']:2d} "
              f"content={s['content']:2d}  cosmetic_edit_frac={s['cosmetic_edit_frac']:.4f}  "
              f"{s['classes']}")
    print("  per-source (pooled):")
    for src, ss in out["shift_edit"]["pooled"]["per_source"].items():
        print(f"    {src:13s} n={ss['n']:2d} cosmetic_frac={ss['cosmetic_edit_frac']:.4f} {ss['classes']}")
    print(f"\n{'='*72}\nPERMANENT-FORK TAIL CLASSIFICATION  (W={args.W})\n{'='*72}")
    for cell in ("suffix6", "ngram5", "pooled", "dedup"):
        s = out["perm_fork"][cell]
        print(f"  {cell:8s} n={s['n']:2d}  content_fork={s['content_fork']:2d} "
              f"format_fork={s['format_length_fork']:2d}  "
              f"content_div_frac={s['permfork_content_divergence_frac']:.4f}  {s['classes']}")
    print("  per-source (pooled):")
    for src, ss in out["perm_fork"]["pooled"]["per_source"].items():
        print(f"    {src:13s} n={ss['n']:2d} content_div_frac={ss['permfork_content_divergence_frac']:.4f} {ss['classes']}")
    print(f"\nHEADLINE: {json.dumps(out['headline'], indent=2)}")
    print(f"\nwrote {OUTDIR}/shift_edit_classification.json")
    print(f"wrote {OUTDIR}/fork_risk_manifest.jsonl  ({len(manifest)} rows)")


if __name__ == "__main__":
    main()
