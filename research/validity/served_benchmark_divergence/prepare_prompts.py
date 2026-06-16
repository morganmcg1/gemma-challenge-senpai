#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #529 (denken) -- benchmark-distribution prompt builder for the served head-prune
divergence audit.

The downstream evals the organizer scores on are MMLU-Pro / GPQA-Diamond / AIME -- long
chain-of-thought reasoning + competition math, NOT the n_new=24 PPL-window decode the #520
neutrality card lived on. We cannot pull those exact splits offline on the pod, but we CAN
build a faithful PROXY of the same prompt distribution from datasets cached in the HF hub:

  * MMLU per-subject STEM/reasoning  (cais/mmlu, 18 cached STEM subjects) -> MMLU-Pro / GPQA
  * Hendrycks MATH                   (nlile/hendrycks-MATH-benchmark)     -> AIME / competition math
  * GSM8K                            (openai/gsm8k)                       -> grade-school reasoning

Each item is rendered as a CoT-eliciting chat turn through the gemma-4-E4B-it chat template
(add_generation_prompt=True) so the served model decodes a long step-by-step answer -- the same
generation geometry the benchmarks score. We emit pre-tokenized `context_token_ids` so the GPU
harness conditions on BYTE-IDENTICAL prompts (the whole A/B rests on identical conditioning).

Decoupled from the GPU harness because it needs `datasets`/`transformers` (run via `uv run
--with datasets --with transformers`), which the vLLM server venv does not carry.

  uv run --with datasets --with transformers python prepare_prompts.py \
      --out prompts.jsonl --n-per-domain 40 --seed 12345 --ctx-cap 1024
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
# QAT-w4a16-ct snapshot carries the gemma-4-E4B-it tokenizer (and is the divergence substrate).
TOKENIZER_PARENT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
)

# 18 STEM/reasoning MMLU subjects cached on the pod (the 'all' config is NOT cached offline).
MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
)
MMLU_LETTERS = ("A", "B", "C", "D")


def _resolve_tokenizer_dir() -> str:
    p = Path(TOKENIZER_PARENT)
    if (p / "tokenizer.json").exists():
        return str(p)
    for sub in sorted(p.glob("*")):
        if (sub / "tokenizer.json").exists():
            return str(sub)
    raise FileNotFoundError(f"no tokenizer under {TOKENIZER_PARENT}")


def _mmlu_prompt(rec: dict) -> str:
    q = rec["question"].strip()
    opts = list(rec["choices"])
    lines = [f"{MMLU_LETTERS[i]}) {opts[i]}" for i in range(len(opts))]
    return (
        "Answer the following multiple-choice question. Reason step by step, then end "
        "with a single line 'Answer: X' where X is one of A, B, C, D.\n\n"
        f"Question: {q}\n" + "\n".join(lines)
    )


def _math_prompt(rec: dict) -> str:
    return (
        "Solve the following math problem. Show your reasoning step by step, then end with "
        "a single line 'Answer: <final answer>'.\n\n"
        f"Problem: {rec['problem'].strip()}"
    )


def _gsm8k_prompt(rec: dict) -> str:
    return (
        "Solve the following problem. Show your reasoning step by step, then end with a "
        "single line 'Answer: <final number>'.\n\n"
        f"Problem: {rec['question'].strip()}"
    )


def _load_mmlu(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    # round-robin one bucket per subject so the STEM mix is broad, not subject-skewed.
    per_subject: dict[str, list[dict]] = {}
    for subj in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception as exc:  # noqa: BLE001
            print(f"[prep] mmlu {subj} skipped: {exc!r}")
            continue
        rows = list(ds)
        rng.shuffle(rows)
        per_subject[subj] = rows
    out: list[dict] = []
    idx = 0
    subjects = [s for s in MMLU_SUBJECTS if per_subject.get(s)]
    while len(out) < n and subjects:
        subj = subjects[idx % len(subjects)]
        bucket = per_subject[subj]
        if bucket:
            rec = bucket.pop()
            out.append({
                "id": f"mmlu/{subj}/{len(out)}",
                "source": f"mmlu:{subj}", "domain": "mmlu",
                "prompt_text": _mmlu_prompt(rec),
            })
        else:
            subjects.remove(subj)
            continue
        idx += 1
    return out


def _load_math(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed + 1)
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for rec in rows[:n]:
        out.append({
            "id": f"math/{rec.get('unique_id', len(out))}",
            "source": f"math:{rec.get('subject', 'math')}", "domain": "math",
            "prompt_text": _math_prompt(rec),
        })
    return out


def _load_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed + 2)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for rec in rows[:n]:
        out.append({
            "id": f"gsm8k/{len(out)}",
            "source": "gsm8k", "domain": "gsm8k",
            "prompt_text": _gsm8k_prompt(rec),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--n-per-domain", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ctx-cap", type=int, default=1024,
                    help="drop prompts whose tokenized context exceeds this (keeps decode budget)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_resolve_tokenizer_dir())

    items = (
        _load_mmlu(args.n_per_domain, args.seed)
        + _load_math(args.n_per_domain, args.seed)
        + _load_gsm8k(args.n_per_domain, args.seed)
    )

    n_written = 0
    n_dropped = 0
    by_domain: dict[str, int] = {}
    with open(args.out, "w") as f:
        for it in items:
            msgs = [{"role": "user", "content": it["prompt_text"]}]
            enc = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True
            )
            # transformers 5.x returns a BatchEncoding (a UserDict, NOT a dict subclass);
            # older returns a bare list. hasattr(.,"keys") catches both dict-likes.
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            ids = [int(x) for x in ids]
            if len(ids) < 4 or len(ids) > args.ctx_cap:
                n_dropped += 1
                continue
            rec = {
                "id": it["id"], "source": it["source"], "domain": it["domain"],
                "context_token_ids": ids, "ctx_len": len(ids),
                "prompt_preview": it["prompt_text"][:240],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            by_domain[it["domain"]] = by_domain.get(it["domain"], 0) + 1

    print(f"[prep] wrote {n_written} prompts (dropped {n_dropped} over ctx-cap={args.ctx_cap}) "
          f"-> {args.out}")
    print(f"[prep] by domain: {by_domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
