#!/usr/bin/env python
"""Compute perplexity from endpoint prompt-token logprobs.

This script is wired into the HF Jobs harness but is not enabled by default.
It expects future ground-truth token records in JSONL or JSON-list form.

Supported record shapes:

* {"id": "...", "context_token_ids": [...], "target_token_ids": [...]}
* {"id": "...", "prompt_token_ids": [...], "score_token_start": 123}

The endpoint must accept vLLM-compatible /v1/completions requests with an
integer-token prompt and return prompt_logprobs for those prompt tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_S = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--request-timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"expected JSON list in {path}")
        records = data
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"all PPL records must be JSON objects: {path}")
    return records


def token_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(token, int) and token >= 0 for token in value):
        raise ValueError(f"{field} must be a list of non-negative integers")
    return value


def normalized_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    record_id = str(record.get("id", index))
    if "context_token_ids" in record and "target_token_ids" in record:
        context = token_list(record["context_token_ids"], "context_token_ids")
        target = token_list(record["target_token_ids"], "target_token_ids")
        prompt_token_ids = context + target
        score_start = len(context)
        score_end = len(prompt_token_ids)
    elif "prompt_token_ids" in record:
        prompt_token_ids = token_list(record["prompt_token_ids"], "prompt_token_ids")
        score_start = int(
            record.get(
                "score_token_start",
                record.get("target_start", record.get("score_start", 1)),
            )
        )
        score_end = int(record.get("score_token_end", record.get("target_end", len(prompt_token_ids))))
    else:
        raise ValueError(
            f"PPL record {record_id} must include either context_token_ids + target_token_ids "
            "or prompt_token_ids"
        )

    score_start = max(score_start, 1)
    if not prompt_token_ids:
        raise ValueError(f"PPL record {record_id} has no prompt tokens")
    if score_end > len(prompt_token_ids):
        raise ValueError(f"PPL record {record_id} score end exceeds prompt length")
    if score_start >= score_end:
        raise ValueError(f"PPL record {record_id} has no scoreable tokens")

    return {
        "id": record_id,
        "prompt_token_ids": prompt_token_ids,
        "score_start": score_start,
        "score_end": score_end,
    }


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


def request_prompt_logprobs(
    *,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    timeout_s: int,
    prompt_logprobs: int,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "prompt_logprobs": prompt_logprobs,
        "add_special_tokens": False,
        "return_token_ids": True,
    }
    return post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout_s)


def choice_from_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("PPL endpoint response did not include choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("PPL endpoint response choice is not an object")
    return choice


def candidate_logprob(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        logprob = value.get("logprob")
        if isinstance(logprob, (int, float)):
            return float(logprob)
    return None


def extract_token_logprob(entry: Any, token_id: int) -> float:
    if entry is None:
        raise ValueError(f"missing prompt logprob entry for token {token_id}")
    if not isinstance(entry, dict):
        raise ValueError(f"prompt logprob entry for token {token_id} is not an object")

    keys = (token_id, str(token_id), f"token_id:{token_id}")
    for key in keys:
        if key in entry:
            logprob = candidate_logprob(entry[key])
            if logprob is not None:
                return logprob

    for key, value in entry.items():
        if str(key) in {str(token_id), f"token_id:{token_id}"}:
            logprob = candidate_logprob(value)
            if logprob is not None:
                return logprob

    sample_keys = list(entry.keys())[:5]
    raise ValueError(f"token {token_id} missing from prompt logprob entry; keys={sample_keys}")


def score_record(
    record: dict[str, Any],
    *,
    base_url: str,
    model: str,
    timeout_s: int,
    prompt_logprobs: int,
    max_tokens: int,
) -> dict[str, Any]:
    started_at = time.time()
    response = request_prompt_logprobs(
        base_url=base_url,
        model=model,
        prompt_token_ids=record["prompt_token_ids"],
        timeout_s=timeout_s,
        prompt_logprobs=prompt_logprobs,
        max_tokens=max_tokens,
    )
    choice = choice_from_response(response)
    logprobs = choice.get("prompt_logprobs") or response.get("prompt_logprobs")
    if not isinstance(logprobs, list):
        raise ValueError("PPL endpoint response did not include prompt_logprobs")

    prompt_token_ids = record["prompt_token_ids"]
    score_start = record["score_start"]
    score_end = record["score_end"]
    if len(logprobs) < score_end:
        raise ValueError(f"received {len(logprobs)} prompt logprobs for {score_end} prompt tokens")

    token_logprobs = [
        extract_token_logprob(logprobs[index], prompt_token_ids[index])
        for index in range(score_start, score_end)
    ]
    neg_log_likelihood = -sum(token_logprobs)
    num_tokens = len(token_logprobs)
    return {
        "id": record["id"],
        "num_tokens": num_tokens,
        "neg_log_likelihood": neg_log_likelihood,
        "ppl": math.exp(neg_log_likelihood / num_tokens),
        "duration_s": time.time() - started_at,
        "score_start": score_start,
        "score_end": score_end,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file)
    records = [normalized_record(record, index) for index, record in enumerate(read_records(dataset_path))]
    if not records:
        raise ValueError(f"no PPL records found in {dataset_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_nll = 0.0
    total_tokens = 0
    record_ppls: list[float] = []
    with output_file.open("w") as handle:
        for record in records:
            result = score_record(
                record,
                base_url=args.base_url,
                model=args.model,
                timeout_s=args.request_timeout_s,
                prompt_logprobs=args.prompt_logprobs,
                max_tokens=args.max_tokens,
            )
            total_nll += result["neg_log_likelihood"]
            total_tokens += result["num_tokens"]
            record_ppls.append(result["ppl"])
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()

    summary = {
        "ppl": math.exp(total_nll / total_tokens),
        "mean_record_ppl": sum(record_ppls) / len(record_ppls),
        "num_records": len(records),
        "num_tokens": total_tokens,
        "neg_log_likelihood": total_nll,
        "dataset_path": str(dataset_path),
        "output_file": str(output_file),
        "model": args.model,
        "base_url": args.base_url,
        "prompt_logprobs": args.prompt_logprobs,
    }
    write_json(summary_file, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
