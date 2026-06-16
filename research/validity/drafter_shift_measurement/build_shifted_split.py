#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #495 -- build a held-out, private-like SHIFTED reasoning/STEM prompt split.

WHY
---
denken #492 (`cqtlhsvz`) established ANALYTICALLY that the 3.661% drafter
acceptance bucket of the public->private TPS gap IS a distribution-shift term
(intrinsic a1 cliff cancels in the ratio r_accept=0.9634). That was a literature
estimate (SambaNova arXiv:2503.07807), NOT a measurement of THIS drafter on
shifted data. To ground it we need a held-out, private-like shifted split to run
through the deployed submission and read the spec-decode acceptance counters.

The public benchmark (`eval_prompts_sharegpt.json`, 128 prompts) is itself
reasoning/STEM: 57 mmlu_pro + 57 gpqa_diamond + 14 aime2026. The #492 "private
domain" is therefore a MILD WITHIN-reasoning/STEM shift, not gross OOD. So this
shifted split is a balanced reasoning/STEM mix drawn from DIFFERENT datasets than
the public set (zero family overlap with mmlu_pro/gpqa/aime), matched to the
public PROMPT FORMAT so the only moved variable is the content distribution:

  * GSM8K (grade-school math word problems)      -> aime-style "math" template
  * hendrycks MATH (competition math, all levels)-> aime-style "math" template
  * MMLU-STEM (college/HS STEM multiple choice)  -> mmlu_pro-style "MC" template

Output: a ShareGPT-format JSON (exactly 128 records) the official
`read_sharegpt_prompts` reader consumes verbatim, so the measurement harness
drives it identically to the public set (same chat template, ignore_eos=512,
greedy, conc=1, M=8 verify). LOCAL data prep only -- no HF Job/submission/train.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research" / "validity" / "drafter_shift_measurement"
PUBLIC_PROMPTS = (
    ROOT / "official" / "main_bucket" / "shared_resources"
    / "speed_benchmark" / "data" / "eval_prompts_sharegpt.json"
)

# Exact public instruction templates (extracted from eval_prompts_sharegpt.json).
MATH_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    'The last line of your response should be of the form "ANSWER: $ANSWER" '
    "(without quotes) where $ANSWER is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line at the end in the form '
    '"ANSWER: $ANSWER".'
)
MC_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}. Think step by step before "
    "answering.\n\nQuestion:\n{question}\nOptions:\n{options}"
)

# A spread of MMLU STEM subjects (college + HS + foundational STEM).
MMLU_STEM_SUBJECTS = [
    "college_physics", "college_chemistry", "college_biology",
    "college_computer_science", "college_mathematics", "abstract_algebra",
    "high_school_physics", "high_school_chemistry", "high_school_biology",
    "high_school_mathematics", "high_school_statistics", "electrical_engineering",
    "conceptual_physics", "astronomy", "machine_learning", "computer_security",
    "elementary_mathematics", "anatomy",
]


def _norm(text: str) -> str:
    """Normalise for overlap dedup: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_public_norms() -> set[str]:
    data = json.loads(PUBLIC_PROMPTS.read_text())
    norms = set()
    for item in data:
        conv = item.get("conversations") or []
        if conv and isinstance(conv[0], dict):
            v = conv[0].get("value")
            if isinstance(v, str):
                norms.add(_norm(v))
                # also index the raw question body (between 'Question:' and 'Options:')
                m = re.search(r"Question:\s*(.*?)\s*Options:", v, re.S)
                if m:
                    norms.add(_norm(m.group(1)))
    return norms


def fmt_math(problem: str) -> str:
    return MATH_TEMPLATE.format(problem=problem.strip())


def fmt_mc(question: str, choices: list[str]) -> str:
    letters = ",".join(chr(ord("A") + i) for i in range(len(choices)))
    opts = "\n".join(f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))
    return MC_TEMPLATE.format(letters=letters, question=question.strip(), options=opts)


def sample_gsm8k(n: int, rng: random.Random, public: set[str], seen: set[str]) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        q = ds[i]["question"]
        prompt = fmt_math(q)
        nq = _norm(q)
        if nq in public or nq in seen:
            continue
        seen.add(nq)
        out.append({"id": f"gsm8k-{i:05d}", "source": "gsm8k", "prompt": prompt})
        if len(out) >= n:
            break
    return out


def sample_math(n: int, rng: random.Random, public: set[str], seen: set[str]) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        p = ds[i]["problem"]
        prompt = fmt_math(p)
        npq = _norm(p)
        if npq in public or npq in seen:
            continue
        seen.add(npq)
        out.append({
            "id": f"math-{i:05d}", "source": "math",
            "level": ds[i].get("level"), "subject": ds[i].get("subject"),
            "prompt": prompt,
        })
        if len(out) >= n:
            break
    return out


def sample_mmlu_stem(n: int, rng: random.Random, public: set[str], seen: set[str]) -> list[dict]:
    from datasets import load_dataset
    # round-robin across subjects for variety; deterministic order via sorted subjects
    pools = []
    for subj in MMLU_STEM_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            continue
        rows = list(range(len(ds)))
        rng.shuffle(rows)
        pools.append((subj, ds, rows))
    out = []
    cursor = {s: 0 for s, _, _ in pools}
    # round-robin
    while len(out) < n and pools:
        progressed = False
        for subj, ds, rows in pools:
            c = cursor[subj]
            if c >= len(rows):
                continue
            cursor[subj] = c + 1
            progressed = True
            r = rows[c]
            q = ds[r]["question"]
            choices = list(ds[r]["choices"])
            prompt = fmt_mc(q, choices)
            nq = _norm(q)
            if nq in public or nq in seen:
                continue
            seen.add(nq)
            out.append({"id": f"mmlu_{subj}-{r:05d}", "source": "mmlu_stem",
                        "subject": subj, "prompt": prompt})
            if len(out) >= n:
                break
        if not progressed:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-gsm8k", type=int, default=43)
    ap.add_argument("--n-math", type=int, default=42)
    ap.add_argument("--n-mmlu", type=int, default=43)
    ap.add_argument("--seed", type=int, default=495)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "shifted_reasoning_stem_128.json")
    args = ap.parse_args(argv)

    public = load_public_norms()
    seen: set[str] = set()
    rng = random.Random(args.seed)

    parts = []
    parts += sample_gsm8k(args.n_gsm8k, rng, public, seen)
    parts += sample_math(args.n_math, rng, public, seen)
    parts += sample_mmlu_stem(args.n_mmlu, rng, public, seen)

    total = args.n_gsm8k + args.n_math + args.n_mmlu
    if len(parts) != total:
        print(f"[build] WARNING: got {len(parts)} != requested {total}; "
              f"topping up from GSM8K")
        extra = sample_gsm8k(total - len(parts) + 5, rng, public, seen)
        parts += extra[: total - len(parts)]

    parts = parts[:total]

    # Emit ShareGPT format (conversations[0].value is the user prompt).
    records = []
    src_counts: dict[str, int] = {}
    for p in parts:
        src_counts[p["source"]] = src_counts.get(p["source"], 0) + 1
        records.append({
            "id": p["id"],
            "source": p["source"],
            "conversations": [
                {"from": "human", "value": p["prompt"]},
                {"from": "gpt", "value": "ok"},
            ],
        })

    args.out.write_text(json.dumps(records, indent=1))

    # overlap audit against public
    overlap = 0
    for r in records:
        if _norm(r["conversations"][0]["value"]) in public:
            overlap += 1

    manifest = {
        "out": str(args.out),
        "total": len(records),
        "source_counts": src_counts,
        "seed": args.seed,
        "public_overlap_records": overlap,
        "public_prompts": str(PUBLIC_PROMPTS),
    }
    (OUT_DIR / "shifted_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
