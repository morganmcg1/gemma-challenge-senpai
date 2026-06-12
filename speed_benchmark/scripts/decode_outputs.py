#!/usr/bin/env python
"""Capture generated text and token IDs from an endpoint for later audits.

The scored speed run uses SGLang's fixed benchmark. This companion pass asks
the same participant endpoint to decode the public prompt set through the
vLLM-compatible /v1/completions API with integer-token prompts and
return_token_ids: true. The captured artifacts are not used for scoring in the
participant job; they make later organizer-side verification possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER = "google/gemma-4-E4B-it"
DEFAULT_NUM_PROMPTS = 128
DEFAULT_OUTPUT_LEN = 512
DEFAULT_SEED = 1
DEFAULT_TIMEOUT_S = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--request-timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_tokens(tokens: list[int]) -> str:
    body = ",".join(str(token) for token in tokens)
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def read_sharegpt_prompts(path: Path, *, num_prompts: int, seed: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
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


def normalize_token_ids(value: Any, field: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, Mapping):
        for key in ("input_ids", "token_ids", "prompt_token_ids"):
            if key in value:
                return normalize_token_ids(value[key], f"{field}.{key}")

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if all(isinstance(token, int) and token >= 0 for token in value):
            return value
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return normalize_token_ids(value[0], field)

    raise ValueError(f"{field} did not contain a list of integer token IDs")


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    except Exception:
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(formatted, str):
            raise ValueError("tokenizer returned a non-string chat template for tokenize=False")
        encoded = tokenizer.encode(formatted, add_special_tokens=False)
    return normalize_token_ids(encoded, "prompt tokenization")


def post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
    return json.loads(response_body)


def choice_from_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("decode response did not include choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("decode response choice is not an object")
    return choice


def integer_token_list(value: Any) -> list[int] | None:
    if isinstance(value, list) and all(isinstance(token, int) and token >= 0 for token in value):
        return value
    return None


def value_at_path(root: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_generated_token_ids(
    response: dict[str, Any],
    choice: dict[str, Any],
    prompt_token_ids: list[int],
) -> tuple[list[int], str, str]:
    candidates: list[tuple[str, Any]] = [
        ("choices[0].token_ids", choice.get("token_ids")),
        ("choices[0].output_token_ids", choice.get("output_token_ids")),
        ("choices[0].generated_token_ids", choice.get("generated_token_ids")),
        ("choices[0].completion_token_ids", choice.get("completion_token_ids")),
        ("choices[0].text_token_ids", choice.get("text_token_ids")),
        ("choices[0].logprobs.token_ids", value_at_path(choice, ("logprobs", "token_ids"))),
        ("token_ids", response.get("token_ids")),
        ("output_token_ids", response.get("output_token_ids")),
        ("generated_token_ids", response.get("generated_token_ids")),
        ("completion_token_ids", response.get("completion_token_ids")),
    ]

    for source, value in candidates:
        token_ids = integer_token_list(value)
        if token_ids is None:
            continue
        if len(token_ids) >= len(prompt_token_ids) and token_ids[: len(prompt_token_ids)] == prompt_token_ids:
            return token_ids[len(prompt_token_ids) :], source, "prompt_plus_completion"
        return token_ids, source, "completion"

    raise ValueError(
        "endpoint did not return generated token IDs. Submissions must support "
        "return_token_ids: true on /v1/completions and return choices[0].token_ids."
    )


def generated_text_from_choice(choice: dict[str, Any]) -> str:
    text = choice.get("text")
    if isinstance(text, str):
        return text
    message = choice.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def request_decode(
    *,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    output_len: int,
    timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": False,
        "add_special_tokens": False,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    return post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout_s)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    dataset_path = Path(args.dataset_path)
    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = read_sharegpt_prompts(dataset_path, num_prompts=args.num_prompts, seed=args.seed)
    if len(records) != args.num_prompts:
        raise ValueError(f"expected {args.num_prompts} prompts, found {len(records)} in {dataset_path}")

    started_at = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    token_id_sources: dict[str, int] = {}
    with output_file.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            prompt_text = record["prompt_text"]
            prompt_token_ids = encode_prompt(tokenizer, prompt_text)
            response = request_decode(
                base_url=args.base_url,
                model=args.model,
                prompt_token_ids=prompt_token_ids,
                output_len=args.output_len,
                timeout_s=args.request_timeout_s,
            )
            choice = choice_from_response(response)
            completion_token_ids, token_id_source, source_kind = extract_generated_token_ids(
                response,
                choice,
                prompt_token_ids,
            )
            generated_text = generated_text_from_choice(choice)

            total_prompt_tokens += len(prompt_token_ids)
            total_completion_tokens += len(completion_token_ids)
            token_id_sources[token_id_source] = token_id_sources.get(token_id_source, 0) + 1

            row = {
                "id": record["id"],
                "index": index,
                "dataset_index": record["dataset_index"],
                "prompt_text": prompt_text,
                "prompt_sha256": sha256_text(prompt_text),
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_sha256": sha256_tokens(prompt_token_ids),
                "generated_text": generated_text,
                "completion_token_ids": completion_token_ids,
                "completion_token_sha256": sha256_tokens(completion_token_ids),
                "num_prompt_tokens": len(prompt_token_ids),
                "num_completion_tokens": len(completion_token_ids),
                "token_id_source": token_id_source,
                "token_id_source_kind": source_kind,
                "request": {
                    "model": args.model,
                    "max_tokens": args.output_len,
                    "temperature": 0.0,
                    "stream": False,
                    "add_special_tokens": False,
                    "ignore_eos": True,
                    "return_token_ids": True,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    summary = {
        "num_records": len(records),
        "num_prompt_tokens": total_prompt_tokens,
        "num_completion_tokens": total_completion_tokens,
        "output_len": args.output_len,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "dataset_path": str(dataset_path),
        "output_file": str(output_file),
        "duration_s": time.time() - started_at,
        "request_timeout_s": args.request_timeout_s,
        "token_ids_required": True,
        "required_request_field": "return_token_ids: true",
        "required_response_field": "choices[0].token_ids",
        "token_id_sources": token_id_sources,
    }
    write_json(summary_file, summary)
    print(f"decode_records={summary['num_records']}", flush=True)
    print(f"decode_completion_tokens={summary['num_completion_tokens']}", flush=True)
    print(f"decode_summary_file={summary_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
