#!/usr/bin/env python
"""PR #660 streaming instrumented decode client (runs under the server venv).

Mirrors the official decode_outputs.py request EXACTLY (same prompts, same
chat-template, same integer-token payload, seed=1) except ``stream=True`` so we
can timestamp every emitted token. For each of the 128 prompts it records, with
``time.perf_counter()`` resolution:

  * t_send            -- just before the POST
  * ttft_s            -- send -> first token chunk (prefill + 1st decode step)
  * decode_window_s   -- first token chunk -> last token chunk (tokens 2..N)
  * request_wall_s    -- send -> last token chunk (full per-request wall)
  * chunk_rel_ms[]    -- arrival of every token chunk, ms relative to t_send
                         (bursty spec-decode structure is preserved: tokens that
                         land in the same verify step share an arrival time)
  * completion_tokens -- authoritative count from the final usage chunk
  * gen_text_sha256   -- sha of the concatenated streamed text (identity x-check
                         vs the non-streaming canonical pass)

These per-request splits let the offline analyzer compute wall_tps under BOTH
PR-named definitions without re-running the server:
  * full_e2e   = sum(completion_tokens) / sum(request_wall_s)   [TTFT included]
  * steady     = sum(completion_tokens-1) / sum(decode_window_s) [prefill excluded]

Prompts/encoding are imported from the official decode_outputs.py so the workload
is byte-identical to the scored pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def _load_official(decode_script: Path):
    spec = importlib.util.spec_from_file_location("senpai_decode_outputs", decode_script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--decode-script", required=True,
                   help="path to official decode_outputs.py (for prompt/encode reuse)")
    p.add_argument("--output-file", required=True)
    p.add_argument("--summary-file", required=True)
    p.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--request-timeout-s", type=int, default=600)
    return p.parse_args()


def stream_one(
    *, base_url: str, model: str, prompt_token_ids: list[int], output_len: int, timeout_s: int
) -> dict[str, Any]:
    """One streaming /v1/completions request. Returns per-request timing record."""
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    chunk_rel_ms: list[float] = []
    text_parts: list[str] = []
    completion_tokens: int | None = None
    t_send = time.perf_counter()
    t_send_wall = time.time()
    t_first: float | None = None
    t_last: float | None = None

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            # usage-only chunk (final, choices empty) when include_usage=True
            usage = obj.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text == "":
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            chunk_rel_ms.append(round((now - t_send) * 1e3, 3))
            text_parts.append(text)

    gen_text = "".join(text_parts)
    n_chunks = len(chunk_rel_ms)
    if completion_tokens is None:
        completion_tokens = n_chunks
    ttft_s = (t_first - t_send) if t_first is not None else float("nan")
    decode_window_s = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
    request_wall_s = (t_last - t_send) if t_last is not None else float("nan")
    return {
        "t_send_wall": t_send_wall,
        "ttft_s": ttft_s,
        "decode_window_s": decode_window_s,
        "request_wall_s": request_wall_s,
        "n_token_chunks": n_chunks,
        "completion_tokens": completion_tokens,
        "gen_text_sha256": hashlib.sha256(gen_text.encode("utf-8")).hexdigest(),
        "chunk_rel_ms": chunk_rel_ms,
    }


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    official = _load_official(Path(args.decode_script))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = official.read_sharegpt_prompts(
        Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed
    )
    if len(records) != args.num_prompts:
        raise ValueError(f"expected {args.num_prompts} prompts, found {len(records)}")

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with out_file.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            prompt_token_ids = official.encode_prompt(tokenizer, record["prompt_text"])
            rec = stream_one(
                base_url=args.base_url,
                model=args.model,
                prompt_token_ids=prompt_token_ids,
                output_len=args.output_len,
                timeout_s=args.request_timeout_s,
            )
            rec["index"] = index
            rec["id"] = record["id"]
            rec["num_prompt_tokens"] = len(prompt_token_ids)
            rows.append(rec)
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            handle.flush()
    wall_loop_s = time.perf_counter() - t0

    tot_tok = sum(r["completion_tokens"] for r in rows)
    sum_req_wall = sum(r["request_wall_s"] for r in rows)
    sum_decode_window = sum(r["decode_window_s"] for r in rows)
    sum_ttft = sum(r["ttft_s"] for r in rows)
    # steady excludes the first emitted token of each request (it shares the TTFT step)
    tot_tok_minus_first = sum(max(0, r["completion_tokens"] - 1) for r in rows)

    summary = {
        "num_records": len(rows),
        "num_completion_tokens": tot_tok,
        "output_len": args.output_len,
        "seed": args.seed,
        "wall_loop_s": wall_loop_s,
        "sum_request_wall_s": sum_req_wall,
        "sum_decode_window_s": sum_decode_window,
        "sum_ttft_s": sum_ttft,
        "mean_ttft_s": sum_ttft / len(rows) if rows else float("nan"),
        # PR-named definitions, computed from per-request splits
        "stream_full_e2e_wall_tps": tot_tok / sum_req_wall if sum_req_wall else float("nan"),
        "stream_steady_wall_tps": (
            tot_tok_minus_first / sum_decode_window if sum_decode_window else float("nan")
        ),
        "stream_loop_wall_tps": tot_tok / wall_loop_s if wall_loop_s else float("nan"),
    }
    Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_file).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
