#!/usr/bin/env python3
"""Build byte-faithful base-eval prompts for the keepset coverage-gap analysis (PR #528).

Reconstructs the SAME three eval sets this cycle used, so the base-argmax "needed
token" stream is directly comparable to ubel #511 / fern #514:

  * MMLU-Pro  : inspect_evals mmlu_pro, seeded subset n=500 seed=12345
                (reuses run_eval.build_mmlu_pro_task from the merged #511 harness).
  * GPQA-Diamond: inspect_evals gpqa, full 198, seed=12345 choice-shuffle
                (reuses run_eval.build_gpqa_diamond_task).
  * AIME-2024 : inspect_evals canonical Maxwell-Jia/AIME_2024 (30), the exact
                dataset + prompt template fern #514 used (aime_years=['2024'],
                n_problems=30, run efgljuxs).

Prompt rendering is byte-identical to inspect_ai's multiple_choice / aime solver:
the MC user content == inspect_ai.solver._multiple_choice.prompt() output, and the
AIME user content == inspect_evals.utils.aime_common.USER_PROMPT_TEMPLATE.

Output: one JSONL record per sample with the rendered user-content string. The
generation step (gen_base_greedy.py, vLLM env) wraps it in a chat turn and
greedily decodes, capturing per-step argmax token ids. This script does NOT touch
a GPU and only needs inspect_ai/inspect_evals + datasets (the .venv).

Run (in repo .venv):
  .venv/bin/python research/validity/keepset_coverage_gap/build_prompts.py \
      --out research/validity/keepset_coverage_gap/prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reuse the EXACT #511 task builders so prompts are byte-identical across arms.
_HARNESS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "downstream_quality_eval",
)
sys.path.insert(0, _HARNESS)

import inspect_ai.solver._multiple_choice as mc  # noqa: E402
from inspect_evals.mmlu_pro.mmlu_pro import (  # noqa: E402
    USER_PROMPT_TEMPLATE as MMLU_TEMPLATE,
)
from inspect_evals.utils.aime_common import (  # noqa: E402
    USER_PROMPT_TEMPLATE as AIME_TEMPLATE,
)
from run_eval import build_gpqa_diamond_task, build_mmlu_pro_task  # noqa: E402

GPQA_COT_TEMPLATE = mc.SINGLE_ANSWER_TEMPLATE_COT


def _choice_value(c) -> str:
    return c.value if hasattr(c, "value") else str(c)


def render_mc(question: str, choices, template: str) -> str:
    """Byte-identical to inspect_ai.solver._multiple_choice.prompt() for str/Choice
    choices: 'A) c0\\nB) c1\\n...' options, 'A,B,...' letters, template.format(...)."""
    opts = "\n".join(
        f"{chr(65 + i)}) {_choice_value(choices[i])}" for i in range(len(choices))
    )
    letters = ",".join(chr(65 + i) for i in range(len(choices)))
    return template.format(choices=opts, letters=letters, question=question)


def render_aime(problem: str) -> str:
    return AIME_TEMPLATE.format(prompt=problem)


def build_records(mmlu_n: int, seed: int, limit: int | None):
    records = []

    # ---- MMLU-Pro (n=500 seed 12345) ----
    mtask = build_mmlu_pro_task(mmlu_n, seed)
    msamps = sorted(mtask.dataset, key=lambda s: str(s.id))
    if limit:
        msamps = msamps[:limit]
    for s in msamps:
        records.append(
            {
                "task": "mmlu_pro",
                "id": str(s.id),
                "prompt_text": render_mc(str(s.input), s.choices, MMLU_TEMPLATE),
                "gold_answer": (s.target if isinstance(s.target, str) else json.dumps(s.target)),
                "gold_kind": "letter",
                "n_choices": len(s.choices),
            }
        )

    # ---- GPQA-Diamond (full 198, seed 12345 choice-shuffle) ----
    gtask = build_gpqa_diamond_task(seed)
    gsamps = sorted(gtask.dataset, key=lambda s: str(s.id))
    if limit:
        gsamps = gsamps[:limit]
    for s in gsamps:
        records.append(
            {
                "task": "gpqa_diamond",
                "id": str(s.id),
                "prompt_text": render_mc(str(s.input), s.choices, GPQA_COT_TEMPLATE),
                "gold_answer": (s.target if isinstance(s.target, str) else json.dumps(s.target)),
                "gold_kind": "letter",
                "n_choices": len(s.choices),
            }
        )

    # ---- AIME-2024 (full 30, fern #514 set) ----
    from datasets import load_dataset

    aime = load_dataset(
        "Maxwell-Jia/AIME_2024",
        split="train",
        revision="8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
    )
    arows = list(aime)
    if limit:
        arows = arows[:limit]
    for r in arows:
        records.append(
            {
                "task": "aime2024",
                "id": str(r["ID"]),
                "prompt_text": render_aime(str(r["Problem"])),
                "gold_answer": str(r["Answer"]),
                "gold_kind": "integer",
                "n_choices": None,
            }
        )

    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmlu-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=0, help="cap per task (smoke); 0=all")
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    records = build_records(args.mmlu_n, args.seed, limit)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    by_task = Counter(r["task"] for r in records)
    print(f"[build_prompts] wrote {len(records)} prompts -> {args.out}", flush=True)
    print(f"[build_prompts] by task: {dict(by_task)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
