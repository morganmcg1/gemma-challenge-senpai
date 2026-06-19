#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Build SAME-FAMILY held-out reasoning/knowledge proxies for the #739 delta_stock probe.

The official public benchmark set (eval_prompts_sharegpt.json) is NOT chat: it is
57 MMLU-Pro + 57 GPQA-Diamond + 14 AIME multiple-choice/math prompts under fixed
templates. So the most decision-relevant "distribution-shift" subsets for the
stock-drafter private re-run are:

  * knowledge_mmlupro : 128 HELD-OUT MMLU-Pro test MCQs (same official template,
                        deduped vs the public 57) -> the closest realization of a
                        "private = same MCQ family, fresh instances" set.
  * reasoning_math    : 128 free-response math problems (AIME_2024 + gsm8k) under
                        the official AIME template -> a genuine reasoning tail that
                        is NOT in public (public AIME is 2026; cached AIME is 2024).

Both are emitted in the byte-identical sglang `conversations` shape the public set
and the native ShareGPT proxies use, so the delta_stock driver loads them the same
way. The gpt turn is an unused placeholder (we regenerate with ignore_eos).

CPU/local only. Reads cached HF parquet via pyarrow; writes data/*.json. No GPU,
no HF Job, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
HUB = Path.home() / ".cache/huggingface/hub"
PUBLIC = REPO / "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"

MMLU_PRO_TEST = next((HUB / "datasets--TIGER-Lab--MMLU-Pro").rglob("test-*.parquet"))
AIME_2024 = next((HUB / "datasets--Maxwell-Jia--AIME_2024").rglob("*.parquet"))
GSM8K_TEST = next((HUB / "datasets--openai--gsm8k").rglob("main/test-*.parquet"))

LETTERS = "ABCDEFGHIJ"

MMLU_HEADER = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of A,B,C,D,E,F,G,H,I,J. Think step by step "
    "before answering."
)
AIME_HEADER = (
    "Solve the following math problem step by step.\nThe last line of your "
    'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
    "$ANSWER is the answer to the problem."
)
AIME_FOOTER = (
    'Remember to put your answer on its own line at the end in the form '
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the '
    "problem, and you do not need to use a \\boxed command."
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def public_question_keys() -> set[str]:
    """Normalized question-text fragments from the public set, to dedup against."""
    data = json.loads(PUBLIC.read_text())
    keys: set[str] = set()
    for x in data:
        v = x["conversations"][0]["value"]
        # public MCQ embeds the raw question after "Question:\n" (mmlu) or after
        # the header (gpqa). Key on a long normalized slice of the whole prompt.
        keys.add(_norm(v)[:200])
        # also key on the question body if present
        m = re.search(r"Question:\n(.+?)\nOptions:", v, re.S)
        if m:
            keys.add(_norm(m.group(1))[:160])
    return keys


def mmlu_record(qid, question, options, idx) -> dict:
    opt_lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    prompt = f"{MMLU_HEADER}\n\nQuestion:\n{question}\nOptions:\n{opt_lines}"
    return {
        "id": f"mmlu_pro_heldout-{idx:04d}",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "Sure -- here is a helpful response."},
        ],
    }


def math_record(problem, idx, tag) -> dict:
    prompt = f"{AIME_HEADER}\n\n{problem}\n\n{AIME_FOOTER}"
    return {
        "id": f"reasoning_{tag}-{idx:04d}",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "Sure -- here is a helpful response."},
        ],
    }


def build_knowledge(n: int, pub_keys: set[str], seed: int) -> list[dict]:
    t = pq.read_table(MMLU_PRO_TEST)
    rows = t.to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[dict] = []
    for r in rows:
        q, opts = r["question"], list(r["options"])
        if not q or not (2 <= len(opts) <= 10):
            continue
        if _norm(q)[:160] in pub_keys or _norm(q)[:200] in pub_keys:
            continue
        # length band: keep prompt comfortably under model len so prefill is sane
        if not (40 <= len(q) <= 1600):
            continue
        out.append(mmlu_record(r["question_id"], q, opts, len(out)))
        if len(out) >= n:
            break
    return out


def build_reasoning(n: int, pub_keys: set[str], seed: int) -> list[dict]:
    out: list[dict] = []
    # AIME_2024 (held-out vs public aime2026): all 30 problems.
    aime = pq.read_table(AIME_2024).to_pylist()
    for r in aime:
        prob = r["Problem"]
        if prob and _norm(prob)[:160] not in pub_keys:
            out.append(math_record(prob, len(out), "aime2024"))
    # gsm8k test: fill the remainder with grade-school word problems.
    gsm = pq.read_table(GSM8K_TEST).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(gsm)
    for r in gsm:
        if len(out) >= n:
            break
        q = r["question"]
        if q and 40 <= len(q) <= 1200 and _norm(q)[:160] not in pub_keys:
            out.append(math_record(q, len(out), "gsm8k"))
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--seed", type=int, default=739)
    args = ap.parse_args()

    pub_keys = public_question_keys()
    print(f"[build] public dedup keys: {len(pub_keys)}", flush=True)

    know = build_knowledge(args.num, pub_keys, args.seed)
    reas = build_reasoning(args.num, pub_keys, args.seed)
    print(f"[build] knowledge_mmlupro n={len(know)}  reasoning_math n={len(reas)}", flush=True)
    assert len(know) == args.num, f"knowledge short: {len(know)}"
    assert len(reas) == args.num, f"reasoning short: {len(reas)}"

    (REPO / "data/deltastock_knowledge_mmlupro.json").write_text(
        json.dumps(know, ensure_ascii=False, indent=2))
    (REPO / "data/deltastock_reasoning_math.json").write_text(
        json.dumps(reas, ensure_ascii=False, indent=2))
    # provenance
    n_aime = sum(1 for r in reas if "aime2024" in r["id"])
    meta = {
        "purpose": "PR #739 delta_stock same-family held-out reasoning/knowledge proxies",
        "public_composition": "57 mmlu_pro + 57 gpqa_diamond + 14 aime2026 (MCQ/math)",
        "knowledge_mmlupro": {"n": len(know), "source": "TIGER-Lab/MMLU-Pro test",
                              "template": "official public MMLU-Pro MCQ", "dedup": "vs public question text"},
        "reasoning_math": {"n": len(reas), "source": f"AIME_2024 ({n_aime}) + gsm8k ({len(reas)-n_aime})",
                           "template": "official public AIME free-response",
                           "held_out": "public AIME is 2026; cached AIME is 2024"},
        "seed": args.seed,
    }
    (REPO / "data/deltastock_proxies.meta.json").write_text(json.dumps(meta, indent=2))
    print("[build] wrote data/deltastock_knowledge_mmlupro.json, data/deltastock_reasoning_math.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
