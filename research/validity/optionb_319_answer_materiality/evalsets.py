#!/usr/bin/env python
"""PR #626 — eval-set construction + tokenization + scoring for the GREEDY
answer-materiality matched arm.

Builds the SAME GPQA-Diamond (n=198) and GSM8K (n=500) instruments the #620
sampled card used, but pre-tokenizes each prompt to INTEGER token ids (client
side, via the body's own tokenizer + chat template) so the generation driver can
hit /v1/completions with `return_token_ids: true` and capture per-arm completion
token ids. That is the canonical official greedy-identity harness path
(decode_outputs.py) — the substrate the strict-#319 contract is defined over —
so a token-level divergence read is faithful to #319 rather than to the
inspect_ai chat path.

Both arms receive the BYTE-IDENTICAL `prompt_token_ids` produced here (one
tokenization, shared), so the cross-arm prompt_sha gate is satisfied by
construction and asserted downstream.

Prompt construction is reused verbatim from the live evals:
  * GPQA: inspect_ai `multiple_choice(cot=True)` formatting — `mc.prompt(question,
    Choices(choices), SINGLE_ANSWER_TEMPLATE_COT)` over the seeded choice-shuffle
    dataset (`build_gpqa_diamond_task` semantics). Scoring replicates inspect's
    `parse_answers` (single-answer) regex + the `choice()` letter match.
  * GSM8K: `gsm8k_eval` 8-shot CoT prompt (`build_prompt`) over the seeded
    500-subset; scoring reuses `gsm8k_eval.extract_pred` + `is_correct`.

NO GPU. Importable. `--dump` runs a no-GPU self-check (construct + tokenize +
score a couple of synthetic completions) so the wiring is provable before any boot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Body tokenizer == the served tokenizer (same dir vLLM loads): byte-identical
# prompt tokens to what the server would build from chat messages.
BODY = "/workspace/gemma_build/int4_g128_lmhead"

# Defaults: greedy is deterministic, so ONE layout per eval (no seed averaging
# like #620's sampled CI). Seeds reuse #620's first dataset seeds for continuity.
GPQA_SEED = 12345
GSM8K_SEED = 1234
GSM8K_N = 500
GSM8K_NSHOT = 8

_DECODE_OUTPUTS = (
    ROOT / "official/main_bucket/shared_resources/speed_benchmark/scripts/decode_outputs.py"
)


def _load_decode_helpers():
    spec = importlib.util.spec_from_file_location("decode_outputs", _DECODE_OUTPUTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DO = _load_decode_helpers()
sha256_tokens = _DO.sha256_tokens  # authoritative comma-joined-decimal recipe


def load_tokenizer(path: str = BODY):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def encode_chat_prompt(tok, prompt_text: str) -> list[int]:
    """One user turn -> chat-templated prompt token ids (official encode_prompt)."""
    return _DO.encode_prompt(tok, prompt_text)


# --------------------------------------------------------------------------- #
# GPQA-Diamond
# --------------------------------------------------------------------------- #
def build_gpqa_items(tok, seed: int = GPQA_SEED, limit: int = 0) -> list[dict[str, Any]]:
    from inspect_evals.gpqa.gpqa import get_gpqa_diamond_dataset
    import inspect_ai.solver._multiple_choice as mc
    from inspect_ai.solver._task_state import Choices

    ds = get_gpqa_diamond_dataset(shuffle_choices=False)
    ds.shuffle_choices(seed=seed)
    samples = sorted(list(ds), key=lambda s: str(s.id))  # stable, arm-independent
    if limit and limit > 0:
        samples = samples[:limit]
    items: list[dict[str, Any]] = []
    for s in samples:
        choices = list(s.choices)
        prompt_text = mc.prompt(
            question=str(s.input),
            choices=Choices(choices),
            template=mc.SINGLE_ANSWER_TEMPLATE_COT,
        )
        ids = encode_chat_prompt(tok, prompt_text)
        items.append(
            {
                "id": str(s.id),
                "kind": "gpqa",
                "prompt_text": prompt_text,
                "prompt_token_ids": ids,
                "prompt_sha256": sha256_tokens(ids),
                "target": str(s.target),
                "n_choices": len(choices),
            }
        )
    return items


# inspect_ai parse_answers (single-answer) — verbatim regex from
# inspect_ai/solver/_multiple_choice.py, so scoring is byte-identical to choice().
_MC_STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
_MC_LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def _answer_char(i: int) -> str:
    return chr(ord("A") + i)


def score_gpqa(text: str, target: str, n_choices: int) -> tuple[str | None, bool]:
    """(extracted_letter|None, correct). Mirrors inspect single-answer parse_answers
    + choice() letter match."""
    matches = _MC_STRICT.findall(text or "")
    if not matches:
        matches = _MC_LOOSE.findall(text or "")
    if not matches:
        return None, False
    matched = matches[-1].strip().rstrip(".").upper()
    allowed = {_answer_char(i) for i in range(n_choices)}
    ans = matched if matched in allowed else None
    return ans, bool(ans is not None and ans == str(target).strip().upper())


# --------------------------------------------------------------------------- #
# GSM8K (8-shot CoT)
# --------------------------------------------------------------------------- #
def build_gsm8k_items(
    tok, n: int = GSM8K_N, seed: int = GSM8K_SEED, n_shot: int = GSM8K_NSHOT, limit: int = 0
) -> list[dict[str, Any]]:
    from research.downstream_quality_gsm8k.gsm8k_eval import (
        _load_split,
        build_fewshot,
        build_prompt,
        gold_answer,
    )

    full = _load_split("test", n=None)
    rng = random.Random(seed)
    order = list(range(len(full)))
    rng.shuffle(order)
    if n is not None and n >= 0:
        order = order[:n]
    if limit and limit > 0:
        order = order[:limit]
    exemplars, fewshot_sig = build_fewshot(n_shot, seed)
    items: list[dict[str, Any]] = []
    for i in order:
        question = full[i]["question"]
        prompt_text = build_prompt(exemplars, question)
        ids = encode_chat_prompt(tok, prompt_text)
        items.append(
            {
                "id": f"test-{i}",
                "kind": "gsm8k",
                "prompt_text": prompt_text,
                "prompt_token_ids": ids,
                "prompt_sha256": sha256_tokens(ids),
                "gold": gold_answer(full[i]["answer"]),
            }
        )
    return items, fewshot_sig


def score_gsm8k(text: str, gold: float | None) -> tuple[float | None, str, bool]:
    from research.downstream_quality_gsm8k.gsm8k_eval import extract_pred, is_correct

    pred, mode = extract_pred(text or "")
    return pred, mode, bool(is_correct(pred, gold))


def score_item(item: dict[str, Any], text: str) -> dict[str, Any]:
    if item["kind"] == "gpqa":
        ans, correct = score_gpqa(text, item["target"], item["n_choices"])
        return {"answer": ans, "correct": correct, "extract_mode": None}
    pred, mode, correct = score_gsm8k(text, item.get("gold"))
    # normalize the numeric answer to a canonical string so divergence compares
    # exact values, not float repr noise.
    ans = None if pred is None else (str(int(pred)) if float(pred).is_integer() else repr(pred))
    return {"answer": ans, "correct": correct, "extract_mode": mode}


# --------------------------------------------------------------------------- #
# No-GPU self-check
# --------------------------------------------------------------------------- #
def _dump(limit: int) -> int:
    tok = load_tokenizer()
    gp = build_gpqa_items(tok, limit=limit or 3)
    gs, sig = build_gsm8k_items(tok, limit=limit or 3)
    print(f"[evalsets] GPQA items={len(gp)} GSM8K items={len(gs)} fewshot_sig={','.join(sig)}")
    g0 = gp[0]
    print(
        f"[evalsets] GPQA[0] id={g0['id']} target={g0['target']} n_choices={g0['n_choices']} "
        f"ptoks={len(g0['prompt_token_ids'])} sha={g0['prompt_sha256'][:12]}"
    )
    # score synthetic completions
    a, c = score_gpqa("Reasoning...\nANSWER: %s" % g0["target"], g0["target"], g0["n_choices"])
    print(f"[evalsets] GPQA score (correct-letter completion) -> answer={a} correct={c}")
    a, c = score_gpqa("ANSWER: %s" % _answer_char((int_target := 0)), g0["target"], g0["n_choices"])
    print(f"[evalsets] GPQA score (letter A completion)        -> answer={a} correct={c}")
    s0 = gs[0]
    print(
        f"[evalsets] GSM8K[0] id={s0['id']} gold={s0['gold']} "
        f"ptoks={len(s0['prompt_token_ids'])} sha={s0['prompt_sha256'][:12]}"
    )
    goldv = s0["gold"]
    txt = "We compute ... The answer is %s." % (int(goldv) if goldv is not None else 0)
    r = score_item(s0, txt)
    print(f"[evalsets] GSM8K score (gold completion)           -> {r}")
    r = score_item(s0, "The answer is 999999.")
    print(f"[evalsets] GSM8K score (wrong completion)          -> {r}")
    # prompt-sha determinism: rebuild GPQA and confirm identical hashes
    gp2 = build_gpqa_items(tok, limit=limit or 3)
    same = all(a["prompt_sha256"] == b["prompt_sha256"] for a, b in zip(gp, gp2))
    print(f"[evalsets] GPQA prompt_sha deterministic across rebuilds: {same}")
    print("[evalsets] OK" if same else "[evalsets] FAIL: nondeterministic prompts")
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="no-GPU wiring self-check")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.dump:
        return _dump(args.limit)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
