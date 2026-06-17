#!/usr/bin/env python3
"""Deterministic inspect_evals MMLU-Pro / GPQA-Diamond driver for the downstream
quality gate (PR #511).

Hits a local vLLM OpenAI server (start_server.sh) with GREEDY decoding and scores
with the same inspect_evals tasks dixie-flatline used, so base and ship numbers are
directly comparable. The whole point is a clean A/B: base and ship must see
BYTE-IDENTICAL prompts. We guarantee that by constructing each dataset as a pure
function of --seed (MMLU-Pro: seeded subset of the fixed test split; GPQA: seeded
choice shuffle), then recording a per-question prompt hash that the compare step
asserts is identical across arms.

One arm (one server) at a time -- the int4 model + KV cache fills the A10G.

Usage:
  run_eval.py --task {mmlu_pro,gpqa_diamond} --arm {base,ship} --out results.json \
      [--n 250] [--seed 12345] [--limit 5] [--max-tokens 2048]
"""
import argparse
import hashlib
import json
import os
import random
import sys

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
# Be quiet + deterministic; never let HF try to phone home for a cached dataset.
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.dataset import MemoryDataset  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_ai.scorer import CORRECT, choice  # noqa: E402
from inspect_ai.solver import multiple_choice  # noqa: E402

# inspect_evals task internals (we reuse their exact prompt templates + record maps)
from inspect_evals.mmlu_pro.mmlu_pro import (  # noqa: E402
    USER_PROMPT_TEMPLATE as MMLU_USER_PROMPT_TEMPLATE,
    mmlu_pro,
)
from inspect_evals.gpqa.gpqa import get_gpqa_diamond_dataset  # noqa: E402


def _sample_prompt_sha(sample) -> str:
    """Stable hash of the model-visible content of a sample (question + ordered
    choices + correct letter). Independent of the model, so identical seeds ->
    identical hashes by construction; the compare step asserts base==ship."""
    choices = list(sample.choices) if sample.choices is not None else []
    payload = json.dumps(
        {"input": str(sample.input), "choices": choices, "target": sample.target},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_mmlu_pro_task(n: int, seed: int) -> Task:
    # shuffle=False -> deterministic test-split order; solver already shuffle=False.
    base_task = mmlu_pro(shuffle=False)
    full = list(base_task.dataset)
    ids = [s.id for s in full]
    if n and n < len(ids):
        rng = random.Random(seed)
        keep = set(rng.sample(ids, n))
        subset = [s for s in full if s.id in keep]
    else:
        subset = full
    subset.sort(key=lambda s: str(s.id))  # stable, arm-independent order
    ds = MemoryDataset(samples=subset, name="mmlu_pro_subset")
    return Task(
        dataset=ds,
        solver=[multiple_choice(template=MMLU_USER_PROMPT_TEMPLATE, shuffle=False)],
        scorer=choice(),
    )


def build_gpqa_diamond_task(seed: int) -> Task:
    # Load with correct-answer-always-A, then deterministically seed-shuffle the
    # choice order so position bias is removed AND both arms get the same layout.
    ds = get_gpqa_diamond_dataset(shuffle_choices=False)
    ds.shuffle_choices(seed=seed)
    samples = list(ds)
    samples.sort(key=lambda s: str(s.id))  # stable, arm-independent order
    ds2 = MemoryDataset(samples=samples, name="gpqa_diamond")
    return Task(
        dataset=ds2,
        solver=multiple_choice(cot=True, shuffle=False),
        scorer=choice(),
        epochs=1,  # greedy is deterministic; repeating epochs is pointless
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mmlu_pro", "gpqa_diamond"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250, help="MMLU-Pro subset size")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=0, help="cap to first N (smoke); 0=all")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-connections", type=int, default=16)
    # Decode protocol (PR #563): default = GREEDY (backwards-identical to #547/#511).
    # The lewtun #31 mandated downstream protocol is the generation_config.json
    # sampling params (gemma-4-E4B-it: do_sample=true temp=1.0 top_p=0.95 top_k=64).
    # temperature/top_p go through GenerateConfig (standard OpenAI fields the
    # OpenAICompatibleAPI forwards explicitly). top_k is NOT a standard OpenAI field
    # -> forward via extra_body so vLLM's SamplingParams picks it up (same mechanism
    # the existing min_tokens passthrough uses). --sampling-seed is the per-request
    # RNG seed (distinct from --seed, which builds the byte-identical dataset): vary
    # it across runs to estimate sampling variance while prompts stay fixed.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0,
                    help="vLLM top_k (0=disabled). Forwarded via extra_body. Set 64 for "
                         "the gemma-4-E4B-it generation_config protocol.")
    ap.add_argument("--sampling-seed", type=int, default=0,
                    help="per-request sampling RNG seed (vLLM SamplingParams.seed). "
                         "Distinct from --seed (dataset construction). Vary for seed-CI.")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="request-level min_tokens EOS-guard (0=off, identical to prior). "
                         "Forwarded as extra_body so vLLM masks EOS until N tokens are emitted "
                         "-- mirrors wirbel #541's GSM8K min_tokens passthrough; no served-file change.")
    ap.add_argument("--ids-file", default=None,
                    help="optional JSON list of sample ids: build the dataset from --seed as "
                         "usual (so prompts stay byte-identical) then keep only these ids. Used "
                         "to re-run just the zeroed subset under min_tokens without re-scoring all.")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    if args.task == "mmlu_pro":
        task = build_mmlu_pro_task(args.n, args.seed)
    else:
        task = build_gpqa_diamond_task(args.seed)

    if args.ids_file:
        keep = {str(x) for x in json.load(open(args.ids_file))}
        task.dataset = task.dataset.filter(lambda s: str(s.id) in keep)
        print(f"[run_eval] ids-file: kept {len(task.dataset)}/{len(keep)} requested ids", flush=True)

    # Record prompt hashes for the full constructed dataset (pre-limit), keyed by id.
    prompt_sha = {str(s.id): _sample_prompt_sha(s) for s in task.dataset}

    limit = args.limit if args.limit and args.limit > 0 else None

    # Use inspect's generic OpenAI-compatible provider (`openai-api/<service>/<model>`)
    # rather than the `openai/` provider: the latter's frontier-model heuristic
    # (is_latest_model is a catch-all -> True for any non-OpenAI name) misclassifies
    # the local model as gpt-5/o-series and silently STRIPS `temperature` from the
    # request. OpenAICompatibleAPI sends temperature=0 explicitly. responses_api=False
    # forces the canonical /v1/chat/completions path (what dixie's harness used).
    # min_tokens is a vLLM SamplingParams extension (not a standard OpenAI field):
    # forward it via extra_body so the server masks EOS until N tokens are emitted.
    extra_body = {}
    if args.min_tokens and args.min_tokens > 0:
        extra_body["min_tokens"] = args.min_tokens
    if args.top_k and args.top_k > 0:
        extra_body["top_k"] = args.top_k
    extra_body = extra_body or None
    model = get_model(
        f"openai-api/local/{args.model}",
        base_url=args.base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        responses_api=False,
        config=GenerateConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_connections=args.max_connections,
            seed=args.sampling_seed,
            extra_body=extra_body,
        ),
    )

    log_dir = args.log_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "_inspect_logs"
    )

    logs = inspect_eval(
        task,
        model=model,
        limit=limit,
        log_dir=log_dir,
        display="plain",
        score=True,
        score_on_error=True,
        retry_on_error=2,
        fail_on_error=0.10,  # tolerate up to 10% sample errors, surface them
    )
    log = logs[0]

    per_sample = []
    n_correct = 0
    n_scored = 0
    n_error = 0
    for s in log.samples or []:
        sid = str(s.id)
        err = None
        if s.error is not None:
            err = getattr(s.error, "message", None) or str(s.error)
            n_error += 1
        score = (s.scores or {}).get("choice")
        val = getattr(score, "value", None) if score is not None else None
        answer = getattr(score, "answer", None) if score is not None else None
        correct = val == CORRECT
        if score is not None and val in (CORRECT, "I"):
            n_scored += 1
            if correct:
                n_correct += 1
        # PR #548: additive empty/EOS-rate instrumentation (no scoring change).
        # An immediate first-token-EOS yields an empty completion -> the choice
        # scorer extracts no answer -> scored incorrect. Recording the raw
        # completion length separates a recoverable EOS-artifact empty from a
        # genuine wrong answer; `empty` is gated on err is None so a request
        # error is not miscounted as an EOS empty. Reads the sample output only.
        comp = ""
        stop_reason = None
        completion_tokens = None
        try:
            out_obj = getattr(s, "output", None)
            if out_obj is not None:
                comp = out_obj.completion or ""
                choices = getattr(out_obj, "choices", None)
                if choices:
                    stop_reason = getattr(choices[0], "stop_reason", None)
                usage = getattr(out_obj, "usage", None)
                if usage is not None:
                    completion_tokens = getattr(usage, "output_tokens", None)
        except Exception:
            comp = comp or ""
        is_empty = bool(err is None and not comp.strip())
        # PR #612: length-truncation diagnostic (#605 logged only empty/chars, never
        # finish_reason). inspect_ai maps vLLM's max_tokens cap -> 'max_tokens' and a
        # context cap -> 'model_length'; either means the generation was cut off
        # before the model emitted its final answer. Additive only -- no scoring change.
        is_length_trunc = bool(
            err is None and stop_reason in ("max_tokens", "model_length")
        )
        tgt = s.target if isinstance(s.target, str) else json.dumps(s.target)
        per_sample.append(
            {
                "id": sid,
                "target": tgt,
                "answer": answer,
                "value": val,
                "correct": bool(correct),
                "error": err,
                "empty": is_empty,
                "completion_chars": len(comp),
                "stop_reason": stop_reason,
                "completion_tokens": completion_tokens,
                "length_truncated": is_length_trunc,
                "prompt_sha": prompt_sha.get(sid),
            }
        )

    accuracy = (n_correct / n_scored) if n_scored else float("nan")
    n_empty = sum(1 for r in per_sample if r["empty"])
    empty_rate = (n_empty / len(per_sample)) if per_sample else float("nan")

    # PR #612: aggregate length-truncation diagnostics (the decisive evidence #605
    # lacked). If the model is running out of generation room before answering, a
    # high length_stop_rate at a tight max_tokens that collapses at a generous one
    # is the truncation smoking gun.
    n_len_trunc = sum(1 for r in per_sample if r["length_truncated"])
    n_stop_max_tokens = sum(1 for r in per_sample if r["stop_reason"] == "max_tokens")
    n_stop_model_length = sum(1 for r in per_sample if r["stop_reason"] == "model_length")
    length_stop_rate = (n_len_trunc / len(per_sample)) if per_sample else float("nan")
    _ctoks = sorted(r["completion_tokens"] for r in per_sample
                    if isinstance(r["completion_tokens"], (int, float)))
    if _ctoks:
        _n = len(_ctoks)
        ctok_mean = sum(_ctoks) / _n
        ctok_p50 = _ctoks[_n // 2]
        ctok_p95 = _ctoks[min(int(0.95 * _n), _n - 1)]
        ctok_max = _ctoks[-1]
    else:
        ctok_mean = ctok_p50 = ctok_p95 = ctok_max = None
    from collections import Counter
    stop_reason_counts = dict(Counter(r["stop_reason"] for r in per_sample))

    out = {
        "task": args.task,
        "arm": args.arm,
        "model": args.model,
        "seed": args.seed,
        "n_requested": (args.n if args.task == "mmlu_pro" else None),
        "limit": limit,
        "n_dataset": len(prompt_sha),
        "n_samples": len(per_sample),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_error": n_error,
        "n_empty": n_empty,
        "empty_rate": empty_rate,
        "n_length_truncated": n_len_trunc,
        "n_stop_max_tokens": n_stop_max_tokens,
        "n_stop_model_length": n_stop_model_length,
        "length_stop_rate": length_stop_rate,
        "stop_reason_counts": stop_reason_counts,
        "completion_tokens_mean": ctok_mean,
        "completion_tokens_p50": ctok_p50,
        "completion_tokens_p95": ctok_p95,
        "completion_tokens_max": ctok_max,
        "accuracy": accuracy,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens or None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k or None,
        "sampling_seed": args.sampling_seed,
        "decode": ("greedy" if args.temperature == 0.0 else "sampling"),
        "base_url": args.base_url,
        "eval_log": getattr(log, "location", None),
        "per_sample": sorted(per_sample, key=lambda r: r["id"]),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    _ctok_mean_s = f"{ctok_mean:.0f}" if ctok_mean is not None else "na"
    _ctok_p95_s = f"{ctok_p95}" if ctok_p95 is not None else "na"
    print(
        f"[run_eval] task={args.task} arm={args.arm} acc={accuracy:.4f} "
        f"scored={n_scored} correct={n_correct} err={n_error} "
        f"empty={n_empty} empty_rate={empty_rate:.4f} "
        f"len_trunc={n_len_trunc} (max_tok={n_stop_max_tokens} model_len={n_stop_model_length}) "
        f"len_stop_rate={length_stop_rate:.4f} ctok_mean={_ctok_mean_s} ctok_p95={_ctok_p95_s} "
        f"-> {args.out}",
        flush=True,
    )
    # NaN guard: a NaN accuracy means nothing scored -> a hard failure to surface.
    if n_scored == 0:
        print("[run_eval] FATAL: 0 samples scored (NaN accuracy)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
