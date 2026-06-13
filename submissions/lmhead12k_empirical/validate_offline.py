#!/usr/bin/env python
"""Offline PPL sanity for the pruned checkpoint via vLLM's offline engine.

Loads the pruned checkpoint through the SAME path the server uses -- the
``vllm_lmhead12k`` plugin registers ``Gemma4ForCausalLMLMHead12k``, whose
``compute_logits`` scatters the kept-row logits back to full-vocab width
(-inf elsewhere). vLLM's prompt-logprobs path calls that exact ``compute_logits``
(gpu_model_runner: ``logits = self.model.compute_logits(prompt_hidden_states)``)
and then gathers the logprob of the real next prompt token by its full-vocab id,
so this offline number is computed identically to the authoritative served
``ppl_endpoint.py`` scorer -- just without the HTTP server or the benchmark loop.

Use it as a fast pre-serve smoke: it confirms (a) the checkpoint loads through
the plugin/custom class, and (b) every scored GT target token gets a finite
logprob (finite PPL by construction, since all GT targets are hard-included in
kept_ids). The served run remains the authoritative gate.

Scoring mirrors official ppl_endpoint.normalized_record / score_record exactly:
  prompt_token_ids = context + target; score_start = max(len(context), 1);
  score_end = len(prompt); NLL = -sum logprob(prompt_token_ids[i]) for i in
  [score_start, score_end); PPL = exp(sum NLL / sum tokens) across all records.

GPU-only (loads the full model in vLLM). Run inside the A10G window with
``CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT_FILE = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
DEFAULT_MODEL = "/workspace/gemma_build/lmhead12k_empirical"
PPL_GATE = 2.42


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gt-file", default=str(GT_FILE))
    ap.add_argument("--max-records", type=int, default=0, help="0 = all 128")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    # The custom model class locates kept_ids.json via MODEL_ID; serve.py sets
    # this in the engine subprocess, so mirror it here for the offline engine.
    os.environ.setdefault("MODEL_ID", args.model)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams

    records = _read_jsonl(Path(args.gt_file))
    if args.max_records:
        records = records[: args.max_records]

    prompts = [
        {"prompt_token_ids": rec["context_token_ids"] + rec["target_token_ids"]}
        for rec in records
    ]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(prompts, sp)

    total_nll = 0.0
    total_tokens = 0
    finite_violations = 0
    missing_entries = 0
    record_ppls: list[float] = []
    violation_records: list[dict] = []
    for rec, out in zip(records, outputs):
        ctx = rec["context_token_ids"]
        tgt = rec["target_token_ids"]
        prompt_ids = ctx + tgt
        score_start = max(len(ctx), 1)
        score_end = len(prompt_ids)
        # vLLM prompt_logprobs: list aligned to prompt positions; entry[i] is a
        # dict {token_id: Logprob} for predicting prompt_ids[i] from prefix <i.
        pls = out.prompt_logprobs
        rec_nll = 0.0
        rec_tokens = 0
        for index in range(score_start, score_end):
            gold = prompt_ids[index]
            entry = pls[index] if index < len(pls) else None
            lp = None
            if isinstance(entry, dict) and gold in entry:
                obj = entry[gold]
                lp = getattr(obj, "logprob", obj)
            if lp is None:
                missing_entries += 1
                finite_violations += 1
                continue
            if not math.isfinite(lp):
                finite_violations += 1
                continue
            rec_nll += -lp
            rec_tokens += 1
        if rec_tokens:
            record_ppls.append(math.exp(rec_nll / rec_tokens))
        total_nll += rec_nll
        total_tokens += rec_tokens
        n_bad = (score_end - score_start) - rec_tokens
        if n_bad:
            violation_records.append({"id": str(rec.get("id")), "clipped": n_bad})

    ppl = math.exp(total_nll / max(1, total_tokens))
    report = {
        "model": args.model,
        "records": len(records),
        "scored_tokens": total_tokens,
        "finite_violations": finite_violations,
        "missing_logprob_entries": missing_entries,
        "violation_records": violation_records[:20],
        "ppl_offline": round(ppl, 4),
        "mean_record_ppl": round(sum(record_ppls) / max(1, len(record_ppls)), 4),
        "ppl_gate": PPL_GATE,
        "passes_gate": ppl <= PPL_GATE and finite_violations == 0,
        "note": "computed via the same compute_logits scatter the served scorer uses",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passes_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
