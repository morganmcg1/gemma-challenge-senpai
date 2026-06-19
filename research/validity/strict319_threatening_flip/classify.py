"""Classify the end-to-end answer outcome of each #694 flagged spec break.

For each flagged prompt: compare the EXTRACTED answer of the AR M=1 reference vs
the spec stack (the method that #689 flagged the break under -- suffix6 primary,
ngram5 where the flag was ngram5-only), both decoded at the full 6144-tok
natural-EOS budget.

Outcome (the PR #694 taxonomy):
  PRESERVED   -- spec answer == AR answer (the break healed to the same answer)
  FLIP_BENIGN -- spec != AR, score-neutral (single-gold MC/AIME => both wrong)
  FLIP_R2W    -- AR right -> spec wrong   (the only score-DAMAGING direction)
  FLIP_W2R    -- AR wrong -> spec right    (score-helping)
  (NO_COMMIT  -- one/both arms produced no extractable answer even at 6144 tok)

threatening_answer_flip_frac = (# spec != AR) / (# flagged).  R2W count is the
load-bearing number reconciled against wirbel #682's aggregate symmetric reshuffle.

The extractor is #689's canonical inspect-style ANSWER: regex, but takes the LAST
match (the prompt mandates the answer on the final line; robust to restatements).
First-vs-last agreement is reported as a sanity check.

Gold is resolved ONLY for prompts that actually flip (AR != spec), via
gold_resolver.GOLD (a small audited map for the flipped subset); PRESERVED prompts
need no gold.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANS_MC = re.compile(r"ANSWER:\s*\(?([A-J])\b", re.IGNORECASE)
ANS_NUM = re.compile(r"ANSWER:\s*\(?(-?\d[\d,]*\.?\d*)")


def extract_answer(text, *, last=True):
    """(#689 canonical) ('mc', LETTER) | ('num', NUMSTR) | None.
    last=True takes the final ANSWER: occurrence (the mandated answer line)."""
    if not text:
        return None
    mc = list(ANS_MC.finditer(text))
    if mc:
        m = mc[-1] if last else mc[0]
        return ("mc", m.group(1).upper())
    num = list(ANS_NUM.finditer(text))
    if num:
        m = num[-1] if last else num[0]
        return ("num", m.group(1).replace(",", ""))
    return None


def load_arm(path):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()}


def spec_method_for(prompt):
    cells = prompt.get("n689_cells") or []
    return "suffix6" if "suffix6" in cells else "ngram5"


def classify():
    runs = HERE / "_runs" / "confirm"
    subset = json.loads((HERE / "flagged_subset.json").read_text())
    ar = load_arm(runs / "decode_ar.jsonl")
    suffix6 = load_arm(runs / "decode_suffix6.jsonl")
    ngram5 = load_arm(runs / "decode_ngram5.jsonl")
    arms = {"suffix6": suffix6, "ngram5": ngram5}

    try:
        from gold_resolver import GOLD
    except Exception:
        GOLD = {}

    rows = []
    for p in subset["prompts"]:
        pid = p["id"]
        method = spec_method_for(p)
        ar_rec, sp_rec = ar[pid], arms[method][pid]
        ar_ans = extract_answer(ar_rec["generated_text"])
        sp_ans = extract_answer(sp_rec["generated_text"])
        ar_first = extract_answer(ar_rec["generated_text"], last=False)
        sp_first = extract_answer(sp_rec["generated_text"], last=False)

        ar_committed = ar_ans is not None
        sp_committed = sp_ans is not None
        both_commit = ar_committed and sp_committed
        flipped = ar_ans != sp_ans  # tuple compare; None handled

        gold = GOLD.get(pid)
        outcome, direction = _outcome(ar_ans, sp_ans, both_commit, flipped, gold)

        rows.append({
            "id": pid,
            "source": p["source"],
            "flag_reason": p["flag_reason"],
            "spec_method": method,
            "label": p["label"],
            "ar_answer": _fmt(ar_ans),
            "spec_answer": _fmt(sp_ans),
            "ar_finish": ar_rec["finish_reason"],
            "spec_finish": sp_rec["finish_reason"],
            "ar_comp_tok": ar_rec["num_completion_tokens"],
            "spec_comp_tok": sp_rec["num_completion_tokens"],
            "ar_committed": ar_committed,
            "spec_committed": sp_committed,
            "both_commit": both_commit,
            "flipped": bool(flipped),
            "gold": _fmt(gold) if gold else None,
            "outcome": outcome,
            "direction": direction,
            "extractor_first_last_agree": (ar_ans == ar_first) and (sp_ans == sp_first),
        })

    return subset, rows


def _outcome(ar_ans, sp_ans, both_commit, flipped, gold):
    if not both_commit:
        return "NO_COMMIT", "none"
    if not flipped:
        return "PRESERVED", "none"
    # a real flip (both committed, answers differ)
    if gold is None:
        return "FLIP_UNGRADED", "needs_gold"
    ar_ok = ar_ans == gold
    sp_ok = sp_ans == gold
    if ar_ok and not sp_ok:
        return "FLIP_R2W", "right_to_wrong"
    if sp_ok and not ar_ok:
        return "FLIP_W2R", "wrong_to_right"
    return "FLIP_BENIGN", "both_wrong" if not ar_ok and not sp_ok else "both_right"


def _fmt(ans):
    if ans is None:
        return None
    return f"{ans[0]}:{ans[1]}"


def summarize(rows):
    from collections import Counter
    n = len(rows)
    n_both = sum(r["both_commit"] for r in rows)
    n_flip = sum(r["flipped"] and r["both_commit"] for r in rows)
    n_flip_any = sum(r["flipped"] for r in rows)
    oc = Counter(r["outcome"] for r in rows)
    return {
        "n_flagged": n,
        "n_both_commit": n_both,
        "n_ar_commit": sum(r["ar_committed"] for r in rows),
        "n_spec_commit": sum(r["spec_committed"] for r in rows),
        "threatening_answer_flip_frac": round(n_flip_any / n, 4) if n else 0.0,
        "flip_count_both_commit": n_flip,
        "preserved_count": oc.get("PRESERVED", 0),
        "flip_r2w_count": oc.get("FLIP_R2W", 0),
        "flip_w2r_count": oc.get("FLIP_W2R", 0),
        "flip_benign_count": oc.get("FLIP_BENIGN", 0),
        "flip_ungraded_count": oc.get("FLIP_UNGRADED", 0),
        "no_commit_count": oc.get("NO_COMMIT", 0),
        "outcome_breakdown": dict(oc),
        "extractor_first_last_disagreements": sum(not r["extractor_first_last_agree"] for r in rows),
    }


if __name__ == "__main__":
    subset, rows = classify()
    summ = summarize(rows)
    (HERE / "outcome_table.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (HERE / "outcome_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print("\n=== per-prompt ===")
    for r in rows:
        print(f"  {r['id']:26s} {r['source']:12s} {r['spec_method']:7s} "
              f"AR={str(r['ar_answer']):8s} SPEC={str(r['spec_answer']):8s} "
              f"{r['outcome']:14s} {r['direction']}")
