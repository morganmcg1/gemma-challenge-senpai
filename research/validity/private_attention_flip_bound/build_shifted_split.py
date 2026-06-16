#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #497 (ubel) -- build the PRIVATE-LIKE / SHIFTED reasoning-STEM prompt split.

WHY A NEW SPLIT (the load-bearing realization)
----------------------------------------------
The 128 PUBLIC benchmark prompts (official/.../speed_benchmark/data/eval_prompts_sharegpt.json,
identical ids to ppl_ground_truth_tokens.jsonl) are ALREADY 100% reasoning/STEM: mmlu_pro 57 /
gpqa_diamond 57 / aime2026 14 (confirmed by #330 DATASET_ANALYSIS + the id breakdown). So #491's
0.524% "public" attention-flip anchor was itself measured on reasoning/STEM. The organizer's PRIVATE
audit set is HELD-OUT reasoning/STEM -- same DOMAIN, DIFFERENT problems. A faithful "private-like /
shifted" test therefore needs a DISJOINT set of reasoning/STEM problems, formatted IDENTICALLY to the
public eval so the ONLY shift is the held-out items (not the prompt format).

SOURCES (provably disjoint from the public 128 by DATASET, not just by item)
---------------------------------------------------------------------------
  * allenai/ai2_arc  ARC-Challenge  (split test) -- grade-school SCIENCE MCQ (4-5 options).
  * deepmind/aqua_rat  raw           (split test) -- algebra word-problem MCQ (5 options).
  * openai/gsm8k  main               (split test) -- grade-school MATH, free-form numeric answer.
None of {arc, aqua, gsm8k} overlaps {mmlu_pro, gpqa_diamond, aime2026}; the public set draws from
none of these three, so item-level overlap is impossible. Pulled deterministically (dataset order,
offset 0) through the HF dataset-viewer REST API -- no `datasets` lib needed.

FORMAT (mirror the public eval EXACTLY so the comparison is a clean held-out shift)
----------------------------------------------------------------------------------
  * MCQ (arc, aqua) -> the public mmlu_pro/gpqa template: "Answer the following multiple choice
    question. ... 'ANSWER: $LETTER' ... LETTER is one of {letters}. Think step by step ...\n\n
    Question:\n{q} \nOptions:\nA) ...".
  * free-form math (gsm8k) -> the public aime template: "Solve the following math problem step by
    step.\n ... 'ANSWER: $ANSWER' ...\n\n{q}".
Tokenized with the deployed gemma-4-E4B chat template (apply_chat_template -> render -> tok(...,
add_special_tokens=False)), so context_token_ids start with bos=2 and match the public tokenization
byte-for-byte in structure (head [2, 105, 2364, 107, ...]).

OUTPUT: shifted_reasoning_stem.jsonl (128 records {id, source, domain, format, context_token_ids,
n_ctx_tokens}) + shifted_reasoning_stem.meta.json (full provenance). The census reads the COMMITTED
jsonl, so reproducing the flip measurement needs NO internet. This is the documented split flagged for
denken #495 reuse (read the same committed jsonl; do NOT re-pull).

TWO BRACKET ENDPOINTS (--plan; advisor 8gpu coordination relay, PR #497)
-----------------------------------------------------------------------
denken #495 measured ITS reasoning/STEM shift as EASIER than the public benchmark (drafter accepted
MORE: dacc=-2.44%, shift_assumption_validated=False) -- the public eval is itself reasoning/STEM, so a
GSM8K/MATH/MMLU-STEM move is mild within-domain. An EASIER split UNDERSTATES the worst-case attention
argmax-flip rate, so a single easy point is only a LOWER BOUND on the offline-audit exposure. To give
an honest bound we BRACKET with two endpoints:
  * --plan reasoning_stem (default): ARC/AQuA/GSM8K -- grade-school-to-undergrad, the EASY/mild lower
    bound (this card's own independently-built, dataset-disjoint easy proxy).
  * --plan hard_ood: MATH-500 level-5 (hardest tier) + AIME-2024 -- competition-grade freeform math,
    the HARD/OOD upper endpoint. Freeform-numeric-heavy ON PURPOSE: long numeric derivations maximize
    near-tie digit tokens, the regime where attention-reduction argmax flips concentrate, so this is a
    deliberately flip-prone worst-case probe. Disjoint from public {mmlu_pro, gpqa_diamond, aime2026}
    by DATASET identity and by YEAR (AIME-2024 != AIME-2026; MATH competitions predate the public set).

SCOPE: data-prep only. No GPU, no model load (tokenizer only), no HF Job, no submission.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

PUBLIC_PROMPTS = (ROOT / "official" / "main_bucket" / "shared_resources" / "speed_benchmark"
                  / "data" / "ppl_ground_truth_tokens.jsonl")
# the public 128 source datasets -- the shifted split must avoid every one of these
PUBLIC_SOURCE_DATASETS = {"mmlu_pro", "gpqa", "gpqa_diamond", "aime", "aime2026"}

VIEWER = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCDEFGHIJKLMNOP"

# clean per-dataset record-id prefixes (also the disjointness guard keys)
SRC_KEYS = {
    "allenai/ai2_arc": "arc", "deepmind/aqua_rat": "aqua", "openai/gsm8k": "gsm8k",
    "HuggingFaceH4/MATH-500": "math500", "Maxwell-Jia/AIME_2024": "aime2024",
}

# deterministic pull plans (dataset order, paged from offset 0): (dataset, config, split, target,
# domain, kind). reasoning_stem = EASY/mild lower bound; hard_ood = HARD/OOD upper endpoint.
PLANS = {
    "reasoning_stem": [
        ("allenai/ai2_arc", "ARC-Challenge", "test", 64, "science", "mcq"),
        ("deepmind/aqua_rat", "raw", "test", 50, "math", "mcq"),
        ("openai/gsm8k", "main", "test", 14, "math", "freeform"),
    ],
    "hard_ood": [
        ("HuggingFaceH4/MATH-500", "default", "test", 98, "math", "freeform"),
        ("Maxwell-Jia/AIME_2024", "default", "train", 30, "math", "freeform"),
    ],
}
PLAN_DIFFICULTY = {"reasoning_stem": "easy_mild_lowerbound", "hard_ood": "hard_ood_upper"}


def out_paths(plan: str) -> tuple[Path, Path]:
    stem = {"reasoning_stem": "shifted_reasoning_stem", "hard_ood": "shifted_hard_ood"}[plan]
    return HERE / f"{stem}.jsonl", HERE / f"{stem}.meta.json"

MCQ_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step "
    "by step before answering.\n\nQuestion:\n{question} \nOptions:\n{options}"
)
FREEFORM_TEMPLATE = (
    "Solve the following math problem step by step.\nThe last line of your response should be of the "
    'form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n{question}'
)


def _fetch_rows(dataset: str, config: str, split: str, length: int, offset: int = 0,
                retries: int = 5) -> list[dict]:
    """One dataset-viewer page at `offset`. Returns the list of `row` dicts in dataset order. The
    viewer occasionally 502s under load -> retry with linear backoff (transient, not a data error)."""
    import time
    length = min(length, 100)
    url = (f"{VIEWER}?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
           f"&split={split}&offset={offset}&length={length}")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = json.load(r)
            return [item["row"] for item in payload.get("rows", [])]
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[build] fetch {dataset} attempt {attempt + 1}/{retries} failed: {repr(e)[:80]}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"dataset-viewer fetch failed for {dataset} after {retries} tries: {last}")


def _fmt_mcq(question: str, options: list[str]) -> str | None:
    options = [str(o).strip() for o in options if str(o).strip()]
    if len(options) < 2 or len(options) > len(LETTERS):
        return None
    letters = ",".join(LETTERS[: len(options)])
    # strip any pre-existing "A) " prefix the source already carries, re-letter uniformly
    cleaned = []
    for i, o in enumerate(options):
        o = re.sub(r"^\s*[A-Pa-p]\s*[\).:\-]\s*", "", o).strip()
        cleaned.append(f"{LETTERS[i]}) {o}")
    return MCQ_TEMPLATE.format(letters=letters, question=question.strip(),
                               options="\n".join(cleaned))


def _arc_to_text(row: dict) -> str | None:
    ch = row.get("choices") or {}
    texts = ch.get("text") or []
    return _fmt_mcq(row.get("question", ""), list(texts))


def _aqua_to_text(row: dict) -> str | None:
    return _fmt_mcq(row.get("question", ""), list(row.get("options") or []))


def _gsm8k_to_text(row: dict) -> str | None:
    q = (row.get("question") or "").strip()
    return FREEFORM_TEMPLATE.format(question=q) if q else None


def _math500_to_text(row: dict) -> str | None:
    # keep ONLY the hardest tier (level 5) -- the genuinely-hard endpoint
    try:
        if int(row.get("level", 0)) != 5:
            return None
    except (TypeError, ValueError):
        return None
    q = (row.get("problem") or "").strip()
    return FREEFORM_TEMPLATE.format(question=q) if q else None


def _aime_to_text(row: dict) -> str | None:
    q = (row.get("Problem") or "").strip()
    return FREEFORM_TEMPLATE.format(question=q) if q else None


CONVERTERS = {"allenai/ai2_arc": _arc_to_text, "deepmind/aqua_rat": _aqua_to_text,
              "openai/gsm8k": _gsm8k_to_text, "HuggingFaceH4/MATH-500": _math500_to_text,
              "Maxwell-Jia/AIME_2024": _aime_to_text}


def _load_tokenizer(model_dir: str | None):
    from transformers import AutoTokenizer
    if model_dir is None:
        import glob
        cands = glob.glob(str(Path("~/.cache/huggingface/hub").expanduser()
                              / "models--google--gemma-4-E4B-it-qat-w4a16-ct" / "snapshots" / "*"))
        if not cands:
            raise FileNotFoundError("no local gemma snapshot for the tokenizer")
        model_dir = sorted(cands)[0]
    return AutoTokenizer.from_pretrained(model_dir), model_dir


def _encode(tok, text: str) -> list[int]:
    """Render the deployed chat template then tokenize WITHOUT extra specials (the rendered string
    already carries <bos>), exactly matching the public eval tokenization."""
    conv = [{"role": "user", "content": text}]
    rendered = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    return [int(t) for t in ids]


def build(args) -> None:
    plan_name = args.plan
    plan = PLANS[plan_name]
    out_jsonl, out_meta = out_paths(plan_name)
    target_total = sum(p[3] for p in plan)
    tok, tok_dir = _load_tokenizer(args.model_dir)
    if PUBLIC_PROMPTS.exists():
        # ppl file is token-ids only; source-dataset disjointness is the guarantee. We still record
        # the public id-prefix set for the meta provenance.
        public_src = {json.loads(l)["id"].split("-")[0] for l in open(PUBLIC_PROMPTS)}
    else:
        public_src = set()

    records: list[dict] = []
    per_source: dict[str, int] = {}
    min_ctx = args.min_ctx_tokens
    src_keys = sorted({SRC_KEYS[p[0]] for p in plan})
    for dataset, config, split, target, domain, kind in plan:
        src_key = SRC_KEYS[dataset]
        assert src_key.split("_")[0] not in PUBLIC_SOURCE_DATASETS, src_key
        conv = CONVERTERS[dataset]
        taken, offset, gri = 0, 0, 0
        while taken < target:
            rows = _fetch_rows(dataset, config, split, length=100, offset=offset)
            if not rows:
                break  # dataset exhausted
            for row in rows:
                if taken >= target:
                    break
                idx, gri = gri, gri + 1
                text = conv(row)
                if not text:
                    continue
                ids = _encode(tok, text)
                if len(ids) < min_ctx:
                    continue
                records.append({
                    "id": f"{src_key}-{idx:04d}",
                    "source": src_key, "domain": domain, "format": kind,
                    "context_token_ids": ids, "n_ctx_tokens": len(ids),
                })
                taken += 1
            offset += len(rows)
            if len(rows) < 100:
                break  # last page
        per_source[src_key] = taken
        if taken < target:
            print(f"[build] WARNING {src_key}: wanted {target}, got {taken}")

    if len(records) < target_total:
        print(f"[build] WARNING total {len(records)} < target {target_total}")

    out_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))

    lens = sorted(r["n_ctx_tokens"] for r in records)
    meta = {
        "card": "private_attention_flip_bound", "pr": 497, "agent": "ubel",
        "kind": f"shifted-split:{plan_name}", "plan": plan_name,
        "difficulty": PLAN_DIFFICULTY[plan_name],
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "n_records": len(records), "target_total": target_total,
        "per_source": per_source,
        "domain_counts": {d: sum(1 for r in records if r["domain"] == d)
                          for d in sorted({r["domain"] for r in records})},
        "format_counts": {k: sum(1 for r in records if r["format"] == k)
                          for k in sorted({r["format"] for r in records})},
        "ctx_token_len": {"min": lens[0] if lens else None, "max": lens[-1] if lens else None,
                          "median": lens[len(lens) // 2] if lens else None},
        "sources": [{"dataset": d, "config": c, "split": s, "target": t, "domain": dm, "kind": k}
                    for (d, c, s, t, dm, k) in plan],
        "tokenizer_dir": tok_dir,
        "public_source_datasets": sorted(public_src),
        "disjoint_from_public_by_dataset": bool(
            not (set(src_keys) & {p.split("_")[0] for p in public_src})),
        "viewer_api": VIEWER,
        "note": ({
            "reasoning_stem": (
                "PRIVATE-LIKE held-out reasoning/STEM, formatted identically to the public eval; "
                "DISJOINT source datasets => zero item overlap with the public 128. The EASY/mild "
                "lower-bound endpoint (grade-school-to-undergrad). Committed for reproducibility and "
                "flagged for #495 reuse."),
            "hard_ood": (
                "HARD/OOD upper endpoint: MATH-500 level-5 (hardest tier) + AIME-2024, competition-"
                "grade freeform math. Freeform-numeric-heavy on purpose -- long numeric derivations "
                "maximize near-tie digit tokens, the attention-flip-prone regime. DISJOINT from public "
                "{mmlu_pro,gpqa_diamond,aime2026} by dataset identity and by year (AIME-2024!=AIME-2026; "
                "MATH competitions predate the public set). Committed for reproducible census."),
        }[plan_name]),
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[build] plan={plan_name} wrote {out_jsonl} ({len(records)} records) + {out_meta}")
    print(f"[build] per_source={per_source} ctx_len_median={meta['ctx_token_len']['median']} "
          f"disjoint_by_dataset={meta['disjoint_from_public_by_dataset']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=sorted(PLANS), default="reasoning_stem",
                    help="reasoning_stem = EASY/mild lower bound (default); hard_ood = HARD/OOD upper "
                    "endpoint (MATH-500 level-5 + AIME-2024). Bracket both for an honest worst-case bound.")
    ap.add_argument("--model-dir", default=None, help="tokenizer dir (default: local gemma snapshot)")
    ap.add_argument("--min-ctx-tokens", type=int, default=48,
                    help="drop too-short prompts so contexts carry real reasoning content")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
