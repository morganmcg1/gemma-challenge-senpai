#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Build a REASONING-length prompt set for the full-serve identity census (PR #534).

The full-serve identity census (`fullserve_identity_census_harness.py`) normally runs on the
128 challenge prompts (`ppl_ground_truth_tokens.jsonl`: 57 mmlu_pro + 57 gpqa_diamond + 14
aime2026 -- a multiple-choice-heavy distribution that elicits SHORT answers). The advisor's
open question (PR #534, 2026-06-16 23:04Z): does the ship's operative-1.0-within-1-bf16-ULP
identity hold OFF that distribution, on a REASONING-length workload where more decode steps =
more split-KV reduction chances = more potential ULP flips?

This builder produces that off-distribution reasoning set: real AIME 2024/2025 competition-math
problems (NONE overlap the challenge's aime2026), chat-formatted with the model's own template
exactly as the deployed AIME greedy eval does (`research/downstream_quality_aime/aime_eval.py`:
problem + "reason step by step ... \\boxed{}" instruction, add_generation_prompt). Free-running
greedy from one of these prompts produces a multi-hundred-token reasoning trace (the deployed AIME
greedy gens are ~1.8k tokens), which is exactly the "hundreds of decode steps" the census needs to
stress. The census then teacher-forces along that trace and reads the M=8 verify vs M=1 AR argmax.

Output schema MATCHES `ppl_ground_truth_tokens.jsonl` so the census loads it unchanged:
  {"id": "...", "context_token_ids": [...], "target_token_ids": []}
target is EMPTY: the census uses the full chat prompt as the cached prefix (reasoning mode: per-prompt
C = len(prompt)) and free-runs the reasoning itself -- there is no ground-truth continuation to force.

Run under the census venv (transformers + the gemma-4 tokenizer):
  /tmp/senpai-venvs/<hash>/bin/python build_reasoning_prompts.py --out reasoning_prompts_aime.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Reuse the EXACT deployed AIME conventions (registry, /rows loader, instruction, message format) from
# the downstream-quality AIME eval so this reasoning set is the same distribution the ship is graded on.
_AIME_EVAL_DIR = Path(__file__).resolve().parents[2] / "downstream_quality_aime"
sys.path.insert(0, str(_AIME_EVAL_DIR))
try:
    from aime_eval import AIME_REGISTRY, AIME_INSTRUCTION, build_messages, load_aime  # type: ignore
except Exception as e:  # pragma: no cover - fall back to an inlined copy if the sibling moves
    AIME_REGISTRY = {
        "2024": {"dataset": "Maxwell-Jia/AIME_2024", "config": "default", "split": "train",
                 "id_col": "ID", "problem_col": "Problem", "answer_col": "Answer"},
        "2025-I": {"dataset": "opencompass/AIME2025", "config": "AIME2025-I", "split": "test",
                   "id_col": "", "problem_col": "question", "answer_col": "answer"},
        "2025-II": {"dataset": "opencompass/AIME2025", "config": "AIME2025-II", "split": "test",
                    "id_col": "", "problem_col": "question", "answer_col": "answer"},
    }
    AIME_INSTRUCTION = ("Please reason step by step to solve the problem, and put your final answer "
                        "(a single integer between 0 and 999) within \\boxed{}.")

    def _to_int(value):
        if value is None:
            return None
        m = re.search(r"-?\d+", str(value).strip().replace(",", ""))
        return int(m.group(0)) if m else None

    def _rows_api(dataset, config, split, length=100):
        url = ("https://datasets-server.huggingface.co/rows"
               f"?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
               f"&split={urllib.parse.quote(split)}&offset=0&length={length}")
        req = urllib.request.Request(url, headers={"User-Agent": "senpai-aime-eval"})
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=60) as r:
            return [row["row"] for row in json.load(r).get("rows", [])]

    def load_aime(years, limit=None):
        out = []
        for year in years:
            spec = AIME_REGISTRY[year]
            for i, row in enumerate(_rows_api(spec["dataset"], spec["config"], spec["split"])):
                ans = _to_int(row[spec["answer_col"]])
                if ans is None:
                    continue
                pid = str(row[spec["id_col"]]) if spec["id_col"] else f"{year}-{i+1:02d}"
                out.append({"id": pid, "year": year, "problem": str(row[spec["problem_col"]]), "answer": ans})
        return out[:limit] if limit is not None else out

    def build_messages(problem):
        return [{"role": "user", "content": f"{problem}\n\n{AIME_INSTRUCTION}"}]

# The census resolves the int4 model dir the same way; reuse its resolver so the tokenizer matches the served model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fullserve_identity_census_harness import resolve_model_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "reasoning_prompts_aime.jsonl"))
    ap.add_argument("--years", default="2024,2025-I,2025-II",
                    help="comma list of AIME years (registry keys). 2026 is EXCLUDED on purpose (it is in the "
                         "128-challenge set; this set must be OFF that distribution).")
    ap.add_argument("--limit", type=int, default=None, help="cap total problems (default: all)")
    ap.add_argument("--no-thinking", action="store_true", default=True,
                    help="pass enable_thinking=False to the chat template (matches the deployed AIME greedy eval).")
    ap.add_argument("--model-dir", default=None, help="override tokenizer source (default: census resolve_model_dir)")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    model_dir = a.model_dir or resolve_model_dir()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    years = [y for y in str(a.years).split(",") if y]
    problems: list = []
    for y in years:
        try:                                          # datasets-server returns transient 500s per-config; a
            problems += load_aime([y])                # bad year must not abort the whole build.
        except Exception as e:
            print(f"[build] WARN year {y} failed ({type(e).__name__}: {e}); skipping", flush=True)
    if a.limit is not None:
        problems = problems[:a.limit]
    if not problems:
        print("[build] no AIME problems loaded -- network/registry issue", flush=True)
        return 1

    def _flatten_ids(ids) -> list[int]:
        # transformers 5.x apply_chat_template(tokenize=True) returns a BatchEncoding (has .input_ids) and/or a
        # batched [[...]] list; normalise to a flat list[int].
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(t) for t in ids]

    def encode(messages) -> list[int]:
        # Mirror the served path: apply_chat_template with a generation prompt. enable_thinking is a Gemma-4
        # template kwarg; tolerate templates that do not accept it.
        for kwargs in (({"enable_thinking": False},) if a.no_thinking else ()) + ({},):
            try:
                ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
            except TypeError:
                continue
            return _flatten_ids(ids)
        raise RuntimeError("apply_chat_template failed for both enable_thinking and default")

    rows, lens = [], []
    for p in problems:
        ids = encode(build_messages(p["problem"]))
        lens.append(len(ids))
        rows.append({"id": f"aime{p['year']}-{p['id']}", "year": p["year"], "answer": p["answer"],
                     "n_ctx": len(ids), "context_token_ids": ids, "target_token_ids": []})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    lens.sort()
    n = len(lens)
    print(f"[build] model_dir={model_dir}", flush=True)
    print(f"[build] wrote {n} AIME reasoning prompts -> {a.out}", flush=True)
    print(f"[build] ctx-token len: min={lens[0]} med={lens[n//2]} max={lens[-1]} "
          f"(p25={lens[n//4]} p75={lens[(3*n)//4]})", flush=True)
    print(f"[build] years={years} ; ids e.g. {[r['id'] for r in rows[:3]]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
