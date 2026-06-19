"""Resolve AUTHORITATIVE gold for the prompts that actually FLIP (AR != spec),
the only subset that needs gold (PRESERVED contribute 0 to R2W regardless).

Authoritative sources (NOT hand-adjudication -- the spheroid-radiation flip below
proves naive physics intuition is unreliable; gold must come from the canonical
datasets):
  gpqa_diamond -- inspect_evals.gpqa.get_gpqa_diamond_dataset(shuffle_choices=False),
                  the SAME canonical openaipublic gpqa_diamond.csv the #643/#598
                  quality harness uses (the gated Idavidrein repo is unreachable
                  from this token; the harness sources the ungated canonical CSV).
                  Unshuffled => choices[0] is the correct answer text.
  mmlu_pro     -- TIGER-Lab/MMLU-Pro test split via the HF datasets-server /search;
                  options[answer_index] is the correct answer text.

Each flip's gold LETTER is resolved by TEXT: take the source correct-answer text and
match it against THIS prompt's option texts (the #694 eval set shuffles option order
independently of the source), so the gold letter is in OUR prompt's lettering -- the
same lettering #689's extractor reads. Writes gold_audit.json (full provenance:
question-match ratio, source text, per-letter match scores) for advisor verification.

Run with the venv that has inspect_evals:  .venv/bin/python resolve_gold.py
"""
from __future__ import annotations

import difflib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"


def _human(rec):
    return next(c["value"] for c in rec["conversations"] if c.get("from") == "human")


def parse_prompt(text):
    opts = {m.group(1): m.group(2).strip()
            for m in re.finditer(r"(?m)^([A-J])\)\s*(.+?)\s*$", text)}
    first = min((text.index(f"\n{L})") for L in opts if f"\n{L})" in text), default=len(text))
    head = text[:first]
    parts = head.split("\n\n", 1)
    stem = (parts[1] if len(parts) > 1 else head).replace("Question:", "").replace("Options:", "").strip()
    return stem, opts


def _norm(s):
    s = s.lower().replace("\\", "").replace("$", "").replace("^", "").replace(" ", "").replace("~", "")
    return re.sub(r"[^a-z0-9/().,=*+-]", "", s)


def _letter_for(correct_text, opts):
    scores = {L: difflib.SequenceMatcher(None, _norm(t), _norm(correct_text)).ratio()
              for L, t in opts.items()}
    gold = max(scores, key=scores.get)
    return gold, {k: round(v, 4) for k, v in scores.items()}


def resolve_gpqa(flips, dataset):
    import inspect_evals.gpqa.gpqa as g
    ds = g.get_gpqa_diamond_dataset(shuffle_choices=False)
    rows = []
    for s in ds:
        qstem = s.input.split("\n\n")[0] if "\n\n" in s.input else s.input
        rows.append((qstem, s.choices[0]))  # unshuffled: choices[0] == correct
    out = {}
    for pid in flips:
        stem, opts = parse_prompt(_human(dataset[pid]))
        best = max(rows, key=lambda r: difflib.SequenceMatcher(None, _norm(r[0]), _norm(stem)).ratio())
        qm = difflib.SequenceMatcher(None, _norm(best[0]), _norm(stem)).ratio()
        gold, ls = _letter_for(best[1], opts)
        out[pid] = {"gold": ["mc", gold], "source": "gpqa_diamond",
                    "question_match_ratio": round(qm, 4), "source_correct_text": best[1],
                    "our_option_text": opts[gold], "letter_match_scores": ls,
                    "method": "inspect_evals get_gpqa_diamond_dataset(shuffle_choices=False), choices[0]=correct"}
    return out


def resolve_mmlu(flips, dataset):
    tok = os.environ.get("HF_TOKEN")

    def get(url):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
        with urllib.request.urlopen(req, timeout=40) as r:
            return json.loads(r.read())

    out = {}
    for pid in flips:
        stem, opts = parse_prompt(_human(dataset[pid]))
        q = " ".join(stem.split()[:24])
        url = ("https://datasets-server.huggingface.co/search?dataset="
               + urllib.parse.quote("TIGER-Lab/MMLU-Pro")
               + "&config=default&split=test&query=" + urllib.parse.quote(q))
        rows = get(url).get("rows", [])
        best = max(rows, key=lambda r: difflib.SequenceMatcher(
            None, _norm(r["row"].get("question", "")), _norm(stem)).ratio())
        R = best["row"]
        qm = difflib.SequenceMatcher(None, _norm(R["question"]), _norm(stem)).ratio()
        correct_text = R["options"][R["answer_index"]]
        gold, ls = _letter_for(correct_text, opts)
        out[pid] = {"gold": ["mc", gold], "source": "mmlu_pro",
                    "question_match_ratio": round(qm, 4), "source_correct_text": correct_text,
                    "source_answer_letter_in_source_order": R.get("answer"),
                    "our_option_text": opts[gold], "letter_match_scores": ls,
                    "method": "TIGER-Lab/MMLU-Pro test split, datasets-server /search, options[answer_index]"}
    return out


def main():
    dataset = {r["id"]: r for r in json.loads(DATASET.read_text())}
    table = HERE / "outcome_table.jsonl"
    flip_rows = [json.loads(l) for l in table.read_text().splitlines() if l.strip()]
    # the prompts needing gold: both committed AND answers differ
    flips = [r["id"] for r in flip_rows if r["flipped"] and r["both_commit"]]
    by_src = {"gpqa_diamond": [], "mmlu_pro": [], "aime2026": []}
    for pid in flips:
        by_src[pid.split("-")[0]].append(pid)

    audit = {}
    if by_src["gpqa_diamond"]:
        audit.update(resolve_gpqa(by_src["gpqa_diamond"], dataset))
    if by_src["mmlu_pro"]:
        audit.update(resolve_mmlu(by_src["mmlu_pro"], dataset))
    if by_src["aime2026"]:
        raise SystemExit(f"aime2026 flips need a gold source (none committed-flip expected): {by_src['aime2026']}")

    # integrity: every match must be high-confidence
    weak = {pid: a["question_match_ratio"] for pid, a in audit.items() if a["question_match_ratio"] < 0.95}
    audit["_meta"] = {"n_flips_needing_gold": len(flips), "flips": flips,
                      "weak_question_matches": weak,
                      "note": "gold letter is in THIS prompt's option lettering (source order differs)"}
    (HERE / "gold_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({pid: {"gold": a["gold"], "qmatch": a["question_match_ratio"]}
                      for pid, a in audit.items() if pid != "_meta"}, indent=2))
    if weak:
        print("WARNING weak question matches:", weak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
