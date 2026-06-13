"""Local, faithful replication of the official benchmark harness I/O.

The greedy-identity proof must decode the *same* 128 prompts, with the *same*
chat-template tokenization and the *same* sampling parameters the HF Jobs harness
uses, so that comparing full-vocab vs sparse-verify decode is apples-to-apples and
the resulting ``decode_outputs.jsonl`` is schema-identical to the harness output.

The selection/encoding logic here mirrors
``official/main_bucket/shared_resources/speed_benchmark/scripts/decode_outputs.py``
and the PPL record handling mirrors ``ppl_endpoint.py``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEED_BENCH = ROOT / "official/main_bucket/shared_resources/speed_benchmark"
EVAL_PROMPTS = SPEED_BENCH / "data/eval_prompts_sharegpt.json"
PPL_GROUND_TRUTH = SPEED_BENCH / "data/ppl_ground_truth_tokens.jsonl"

DEFAULT_NUM_PROMPTS = 128
DEFAULT_OUTPUT_LEN = 512
DEFAULT_SEED = 1


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_tokens(tokens: list[int]) -> str:
    body = ",".join(str(token) for token in tokens)
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def read_sharegpt_prompts(
    path: Path = EVAL_PROMPTS, *, num_prompts: int = DEFAULT_NUM_PROMPTS, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Replicate decode_outputs.read_sharegpt_prompts exactly (seed-1 shuffle)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected ShareGPT JSON list in {path}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conversations = item.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        first = conversations[0]
        if not isinstance(first, dict):
            continue
        prompt = first.get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        records.append(
            {
                "id": str(item.get("id", index)),
                "dataset_index": index,
                "prompt_text": prompt,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:num_prompts]


def normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        for key in ("input_ids", "token_ids", "prompt_token_ids"):
            if key in value:
                return normalize_token_ids(value[key])
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if all(isinstance(t, int) and t >= 0 for t in value):
            return value
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return normalize_token_ids(value[0])
    raise ValueError("expected a list of non-negative integer token IDs")


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Replicate decode_outputs.encode_prompt (chat template, add_generation_prompt)."""
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    except Exception:
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if not isinstance(formatted, str):
            raise ValueError("tokenizer returned a non-string chat template for tokenize=False")
        encoded = tokenizer.encode(formatted, add_special_tokens=False)
    return normalize_token_ids(encoded)


def read_ppl_records(path: Path = PPL_GROUND_TRUTH) -> list[dict[str, Any]]:
    """Read the PPL ground-truth records: {id, context_token_ids, target_token_ids}."""
    text = Path(path).read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for index, rec in enumerate(records):
        out.append(
            {
                "id": str(rec.get("id", index)),
                "context_token_ids": list(rec["context_token_ids"]),
                "target_token_ids": list(rec["target_token_ids"]),
            }
        )
    return out


def decode_row(
    *,
    record: dict[str, Any],
    index: int,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    generated_text: str = "",
) -> dict[str, Any]:
    """Build one decode_outputs.jsonl row in the harness schema."""
    return {
        "id": record["id"],
        "index": index,
        "dataset_index": record.get("dataset_index", index),
        "prompt_text": record.get("prompt_text", ""),
        "prompt_sha256": sha256_text(record.get("prompt_text", "")),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_sha256": sha256_tokens(prompt_token_ids),
        "generated_text": generated_text,
        "completion_token_ids": completion_token_ids,
        "completion_token_sha256": sha256_tokens(completion_token_ids),
        "num_prompt_tokens": len(prompt_token_ids),
        "num_completion_tokens": len(completion_token_ids),
    }
