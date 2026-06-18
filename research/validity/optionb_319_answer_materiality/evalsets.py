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
def build_gpqa_items(
    tok, seed: int = GPQA_SEED, limit: int = 0, id_suffix: str = ""
) -> list[dict[str, Any]]:
    """GPQA-Diamond items at one choice-shuffle ``seed``.

    The choice-shuffle (which option is labelled A/B/C/D) is seeded, so a *different*
    seed yields a genuinely different prompt for the same underlying question — a
    distinct, non-degenerate paired-greedy evaluation instance (greedy is invariant to
    the *sampling* seed, but NOT to the choice order, since the model has position bias
    and the rendered letters change). This is the GPQA multi-permutation lever PR #637
    Leg 2 uses to extend n past the 198 unique Diamond questions: each (question, seed)
    is its own paired unit. ``id_suffix`` keeps the primary seed's ids byte-identical to
    the #626-banked file (id_suffix="") while tagging extra seeds (e.g. "#s23456") so
    they coexist in one jsonl; ``base_qid`` carries the underlying question for the
    by-question cluster-bootstrap sensitivity (avoids treating two shuffles of one
    question as fully independent)."""
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
                "id": f"{s.id}{id_suffix}",
                "base_qid": str(s.id),
                "shuffle_seed": seed,
                "kind": "gpqa",
                "prompt_text": prompt_text,
                "prompt_token_ids": ids,
                "prompt_sha256": sha256_tokens(ids),
                "target": str(s.target),
                "n_choices": len(choices),
            }
        )
    return items


def build_gpqa_items_multi(
    tok, seeds: list[int], limit: int = 0
) -> list[dict[str, Any]]:
    """Pool GPQA-Diamond across multiple choice-shuffle seeds (PR #637 Leg 2).

    The FIRST seed gets id_suffix="" so it reuses the #626-banked ``gpqa_*.jsonl``
    (id == question id, no regeneration); subsequent seeds are suffixed "#s<seed>".
    The generation driver is idempotent by id, so the banked primary-seed items are
    skipped and only the new shuffle instances are generated.

    DEDUP BY PROMPT: two distinct seeds can land the SAME choice permutation for a
    question (≈1/24 per 4-choice item → ~8/198 collisions per added seed). A colliding
    shuffle yields a prompt byte-identical to an earlier seed's, so under greedy it would
    produce an EXACT-duplicate paired unit — pseudo-replication that anti-conservatively
    tightens the per-instance CI (the ±4pp gate) and wastes GPU regenerating a known
    completion. We therefore drop any item whose ``prompt_sha256`` already appeared in the
    pool, keeping the earliest seed (primary wins). Every pooled unit is thus a genuinely
    distinct prompt; ``base_qid`` still groups a question's surviving shuffles for the
    honest by-question cluster bootstrap."""
    out: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    n_collision = 0
    for i, sd in enumerate(seeds):
        suffix = "" if i == 0 else f"#s{sd}"
        for it in build_gpqa_items(tok, seed=sd, limit=limit, id_suffix=suffix):
            sha = it["prompt_sha256"]
            if sha in seen_sha:
                n_collision += 1
                continue
            seen_sha.add(sha)
            out.append(it)
    if n_collision:
        print(f"[gen] gpqa multi-shuffle: dropped {n_collision} duplicate-prompt "
              f"collision(s) across {len(seeds)} seeds -> {len(out)} distinct units",
              flush=True)
    return out


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


# --------------------------------------------------------------------------- #
# AIME (greedy maj@1, no-think) — PR #637 Leg 1
# --------------------------------------------------------------------------- #
# AIME answers are confident integers, so a tie-flip that reaches the boxed answer is
# a LARGE-margin answer flip by construction — exactly the decisive event #626's
# `n_large_margin_answer_flips=0` verdict says never happens. We mirror the existing
# downstream_quality_aime instrument (same dataset registry, same boxed-int extractor)
# but pre-tokenize the no-think prompt onto the canonical greedy-identity completions
# path (return_token_ids) so the token-divergence + gap probe machinery applies.
AIME_YEARS_DEFAULT = ("2024", "2025-I", "2025-II")  # 30 + 15 + 15 = 60


def build_aime_items(
    tok, years: tuple[str, ...] | list[str] = AIME_YEARS_DEFAULT, limit: int = 0
) -> list[dict[str, Any]]:
    from research.downstream_quality_aime.aime_eval import (
        AIME_INSTRUCTION,
        load_aime,
    )

    problems = load_aime(list(years), limit=(limit or None))
    items: list[dict[str, Any]] = []
    for p in problems:
        # build_messages(problem) == single user turn {problem}\n\n{INSTRUCTION}; the
        # greedy-identity encode_chat_prompt wraps that in the chat template with NO
        # enable_thinking kwarg -> Gemma's default no-think render (the gate config).
        prompt_text = f"{p['problem']}\n\n{AIME_INSTRUCTION}"
        ids = encode_chat_prompt(tok, prompt_text)
        items.append(
            {
                "id": str(p["id"]),
                "kind": "aime",
                "year": p.get("year"),
                "prompt_text": prompt_text,
                "prompt_token_ids": ids,
                "prompt_sha256": sha256_tokens(ids),
                "gold": int(p["answer"]),
            }
        )
    return items


def score_aime(text: str, gold: int | None) -> tuple[int | None, bool]:
    """(extracted_int|None, correct). Mirrors downstream_quality_aime: last boxed
    integer (else last 0-999 int), integer-match against gold."""
    from research.downstream_quality_aime.aime_eval import extract_answer

    pred = extract_answer(text or "")
    return pred, bool(pred is not None and gold is not None and pred == gold)


def score_item(item: dict[str, Any], text: str) -> dict[str, Any]:
    if item["kind"] == "gpqa":
        ans, correct = score_gpqa(text, item["target"], item["n_choices"])
        return {"answer": ans, "correct": correct, "extract_mode": None}
    if item["kind"] == "aime":
        pred, correct = score_aime(text, item.get("gold"))
        # canonical string so divergence compares integer values, not repr noise.
        ans = None if pred is None else str(int(pred))
        return {"answer": ans, "correct": correct, "extract_mode": "boxed_int"}
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
