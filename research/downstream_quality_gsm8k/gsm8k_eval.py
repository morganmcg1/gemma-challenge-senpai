"""Local GSM8K 8-shot CoT eval for the named-gate set (PR #533).

GSM8K is the one missing cell in Morgan's #515 ship-quality gate (the named axes are
GSM8K, AIME, MMLU-Pro). AIME (fern #514) and MMLU-Pro/GPQA (ubel #511) were measured
base-vs-ship; GSM8K never was. This harness establishes it with three model arms under
one byte-identical protocol:

  * ``base``      -- stock int4 ``gemma-4-E4B-it`` (submissions/int4_base_aime): the gate
    denominator.
  * ``ship-12k``  -- the live surgical-357 served substrate (submissions/fa2sw_strict_
    surgical357): osoi5 int4 + the 12k-row lm_head prune.
  * ``full-head`` -- osoi5 int4 with the 12k head-prune DISABLED (full 262,144-row
    lm_head); quality-safe-by-construction (no token is -inf'd).

Protocol (the team's usual lm-eval-harness GSM8K): standard **8-shot chain-of-thought**,
exact-match on the final numeric answer. Two decode regimes per arm:

  * ``sampled``  -- PRIMARY gate number, at the model's native ``generation_config``
    (do_sample=true, T=1.0, top_p=0.95, top_k=64). Per lewtun Issue #31 the downstream
    number is the sampled one, not greedy.
  * ``greedy``   -- diagnostic (T=0). Greedy isolates the kernel from the sampler: the
    surgical-357 fused-accept kernel emits the target argmax, so spec-on/off are identical
    by construction; the only quality mover left is the head-prune's -inf mask.

Why a custom harness and not lm-eval-harness: the library is not installed in this
container (same constraint fern hit for AIME / ubel for MMLU-Pro). We mirror the
lm-eval-harness GSM8K conventions -- 8-shot CoT exemplars drawn deterministically from
the train split, ``The answer is N.`` strict-match with a flexible last-number fallback,
integer exact-match scoring -- so the numbers stay comparable to the team's stack.

Two ways to drive a model (mirrors aime_eval.py):
  * ``--submission <dir>``  -- stand the submission up via the local-validation
    ``LocalServer`` (manifest deps + env, with the local-serving overrides this script
    injects) and eval the live endpoint.
  * ``--base-url <url>``    -- eval an endpoint someone else already started.

The same seeded test-item subset and the same 8 few-shot exemplars are used across every
arm (persisted in the output JSON), so arms run at different times still pair apples-to-
apples through ``gsm8k_combine.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Dataset loading (no `datasets` dep -- HF datasets-server /rows API, paginated)
# --------------------------------------------------------------------------- #
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"


def _rows_api(dataset: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
        f"&split={urllib.parse.quote(split)}&offset={offset}&length={length}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "senpai-gsm8k-eval"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:  # transient datasets-server hiccup
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"datasets-server /rows failed after retries: {last_err!r}")


def _load_split(split: str, n: int | None) -> list[dict[str, str]]:
    """Load up to ``n`` rows of a GSM8K split (paginated; /rows caps length at 100)."""
    out: list[dict[str, str]] = []
    offset = 0
    page = 100
    total = None
    while True:
        data = _rows_api(GSM8K_DATASET, GSM8K_CONFIG, split, offset, page)
        if total is None:
            total = data.get("num_rows_total")
        rows = data.get("rows", [])
        if not rows:
            break
        for row in rows:
            r = row["row"]
            out.append({"question": str(r["question"]), "answer": str(r["answer"])})
            if n is not None and len(out) >= n:
                return out
        offset += len(rows)
        if total is not None and offset >= total:
            break
    return out


# --------------------------------------------------------------------------- #
# Gold + prediction answer extraction (mirrors lm-eval-harness GSM8K)
# --------------------------------------------------------------------------- #
_GOLD_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
# "The answer is N", "answer: N", "#### N" -- the strict instructed/canonical anchors.
_STRICT_RE = re.compile(
    r"(?:the\s+answer\s+is|answer\s*[:=]|####)\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
# Any signed number with optional thousands separators / decimal.
_NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
_CALC_RE = re.compile(r"<<[^>]*>>")  # GSM8K calculator annotations in train CoT


def normalize_num(s: str) -> float | None:
    """Parse a GSM8K numeric token to a float, or None. Strips $ and thousands commas."""
    if s is None:
        return None
    t = str(s).strip().strip("$").replace(",", "").rstrip(".")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", t)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None


def gold_answer(answer_field: str) -> float | None:
    """Gold numeric answer from a GSM8K ``answer`` field (the value after '#### ')."""
    m = _GOLD_RE.search(answer_field)
    if not m:
        return None
    return normalize_num(m.group(1))


def extract_pred(text: str) -> tuple[float | None, str]:
    """Final numeric prediction from a completion.

    Priority: the LAST strict anchor ('the answer is N' / 'answer: N' / '#### N'); else the
    LAST bare number anywhere (flexible fallback). Returns (value, mode) where mode is
    'strict' | 'flexible' | 'none'.
    """
    if not text:
        return None, "none"
    strict = list(_STRICT_RE.finditer(text))
    if strict:
        val = normalize_num(strict[-1].group(1))
        if val is not None:
            return val, "strict"
    nums = list(_NUM_RE.finditer(text))
    for m in reversed(nums):
        val = normalize_num(m.group(0))
        if val is not None:
            return val, "flexible"
    return None, "none"


def is_correct(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < 1e-4


# --------------------------------------------------------------------------- #
# 8-shot CoT prompt (exemplars drawn deterministically from the train split)
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = (
    "Solve the grade-school math word problems. For each question, reason step by step, "
    "then state the final answer on its own at the end as 'The answer is N.' where N is a "
    "single number."
)


def _clean_cot(answer_field: str) -> tuple[str, float | None]:
    """Turn a GSM8K train ``answer`` (CoT + '#### N') into a clean exemplar CoT.

    Strips ``<<calc>>`` annotations and the '#### N' tail, re-appends 'The answer is N.'."""
    gold = gold_answer(answer_field)
    body = _GOLD_RE.sub("", answer_field).strip()
    body = _CALC_RE.sub("", body).strip()
    if gold is not None:
        g = int(gold) if float(gold).is_integer() else gold
        body = f"{body}\nThe answer is {g}."
    return body, gold


def build_fewshot(n_shot: int, seed: int) -> tuple[list[dict[str, str]], list[str]]:
    """Deterministic n_shot exemplars from the train split. Returns (exemplars, signature)."""
    train = _load_split("train", n=max(n_shot * 4, 64))
    import random

    rng = random.Random(seed)
    idxs = list(range(len(train)))
    rng.shuffle(idxs)
    chosen = idxs[:n_shot]
    exemplars: list[dict[str, str]] = []
    sig: list[str] = []
    for i in chosen:
        cot, gold = _clean_cot(train[i]["answer"])
        exemplars.append({"question": train[i]["question"], "cot": cot})
        sig.append(f"{i}:{gold}")
    return exemplars, sig


def build_prompt(exemplars: list[dict[str, str]], question: str) -> str:
    parts = [SYSTEM_INSTRUCTION, "", "Here are some worked examples:", ""]
    for ex in exemplars:
        parts.append(f"Question: {ex['question']}")
        parts.append(f"Answer: {ex['cot']}")
        parts.append("")
    parts.append("Now solve this problem.")
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
    timeout_s: int,
    min_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": 1,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "top_k": top_k,  # vLLM extension
    }
    if min_tokens is not None:
        # vLLM extension: forbid EOS/stop until >= min_tokens generated. Used to
        # guard the base_fullhead first-token-EOS pathology (PR #541): the surgical
        # kernels tip a spurious end-of-turn token to win the position-0 argmax on
        # ~10-15% of GSM8K prompts; min_tokens lets us measure whether the reasoning
        # underneath is intact (it is) without any served-file change.
        payload["min_tokens"] = min_tokens
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def eval_endpoint(
    base_url: str,
    model: str,
    problems: list[dict[str, Any]],
    exemplars: list[dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
    concurrency: int,
    request_timeout_s: int,
    save_text: bool = False,
    min_tokens: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    results: dict[int, dict[str, Any]] = {}

    def _one(idx: int, prob: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(exemplars, prob["question"])
        resp = chat_completion(
            base_url,
            model,
            build_messages(prompt),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            enable_thinking=enable_thinking,
            timeout_s=request_timeout_s,
            min_tokens=min_tokens,
        )
        choice = (resp.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content") or ""
        finish = choice.get("finish_reason")
        pred, mode = extract_pred(text)
        gold = prob["gold"]
        correct = is_correct(pred, gold)
        rec = {
            "id": prob["id"],
            "gold": gold,
            "pred": pred,
            "extract_mode": mode,
            "correct": correct,
            "finish_reason": finish,
            "sample_chars": len(text),
        }
        if save_text:
            rec["text"] = text
        return rec

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = {pool.submit(_one, i, p): i for i, p in enumerate(problems)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                prob = problems[i]
                results[i] = {
                    "id": prob["id"],
                    "gold": prob["gold"],
                    "pred": None,
                    "extract_mode": "error",
                    "correct": False,
                    "finish_reason": "error",
                    "sample_chars": 0,
                    "error": repr(exc),
                }
            done += 1
            if done % 25 == 0 or done == len(problems):
                acc = sum(1 for r in results.values() if r["correct"]) / done
                print(f"[gsm8k] {done}/{len(problems)} running_acc={acc:.4f}", flush=True)

    per_problem = [results[i] for i in range(len(problems))]
    n = len(per_problem)
    n_correct = sum(1 for r in per_problem if r["correct"])
    n_strict = sum(1 for r in per_problem if r["extract_mode"] == "strict")
    n_extract_fail = sum(1 for r in per_problem if r["extract_mode"] in ("none", "error"))
    n_trunc = sum(1 for r in per_problem if r["finish_reason"] == "length")
    return {
        "n_problems": n,
        "accuracy": n_correct / n if n else 0.0,
        "n_correct": n_correct,
        "strict_rate": n_strict / n if n else 0.0,
        "extract_fail_rate": n_extract_fail / n if n else 0.0,
        "truncation_rate": n_trunc / n if n else 0.0,
        "wall_s": time.time() - t0,
        "per_problem": per_problem,
    }


# --------------------------------------------------------------------------- #
# Self-test (no GPU): prove gold + prediction extraction is sound.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(f"[self-test] {'ok' if cond else 'FAIL'}: {msg}")
        if not cond:
            ok = False

    check(gold_answer("blah blah\n#### 18") == 18.0, "gold #### 18")
    check(gold_answer("calc <<3*2=6>>6 ... #### 72,000") == 72000.0, "gold comma thousands")
    p, m = extract_pred("First 2+2=4. The answer is 4.")
    check(p == 4.0 and m == "strict", "strict 'the answer is 4'")
    p, m = extract_pred("...\nThe answer is 1,024.")
    check(p == 1024.0 and m == "strict", "strict comma")
    p, m = extract_pred("we get 7 then 42 with no anchor")
    check(p == 42.0 and m == "flexible", "flexible last number")
    p, m = extract_pred("answer: $250")
    check(p == 250.0 and m == "strict", "strict 'answer: $250'")
    p, m = extract_pred("no digits here")
    check(p is None and m == "none", "no number -> none")
    check(is_correct(18.0, 18.0) and not is_correct(18.0, 19.0), "is_correct")
    check(_clean_cot("She has <<3*4=12>>12 left.\n#### 12")[0].endswith("The answer is 12."),
          "_clean_cot strips calc + appends 'The answer is N.'")
    print("[self-test] PASS" if ok else "[self-test] FAILURES PRESENT")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Local-serving overrides (HF-Job-path -> local-friendly); identical recipe to
# aime_eval.py so base/ship/full-head are apples-to-apples.
# --------------------------------------------------------------------------- #
def local_serve_overrides(max_num_seqs: int) -> dict[str, str]:
    return {
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_gsm8k_no_precache.json",
        "MAX_NUM_SEQS": str(max_num_seqs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="run extractor self-test and exit (no GPU)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--submission", type=Path, help="serve this submission dir via LocalServer, then eval")
    src.add_argument("--base-url", help="eval an already-running endpoint")
    ap.add_argument("--server-python", type=Path, default=None, help="python with vLLM (default: build from manifest deps)")
    ap.add_argument("--model", default="gemma-4-e4b-it", help="served model name")
    ap.add_argument("--label", required=False, help="arm label (base / ship12k / fullhead)")
    ap.add_argument("--regimes", default="sampled,greedy", help="comma list from {sampled,greedy}")
    ap.add_argument("--n", type=int, default=500, help="number of test items (seeded subset; -1 = full 1319)")
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap items (smoke); overrides --n when smaller")
    ap.add_argument("--seed", type=int, default=1234, help="seed for subset + fewshot (+ sampler unless --sampling-seed given)")
    ap.add_argument("--sampling-seed", type=int, default=None,
                    help="per-request decode RNG seed (vLLM SamplingParams.seed). Distinct from --seed "
                         "(which fixes the test subset + few-shot). Vary across runs to estimate decode "
                         "variance with a BYTE-IDENTICAL benchmark (PR #590 multi-seed CI). Default=None "
                         "-> falls back to --seed (back-compat with prior single-seed runs).")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="vLLM min_tokens: forbid EOS until N tokens generated "
                         "(PR #541 first-token-EOS guard; default None = unset, byte-identical to #533)")
    ap.add_argument("--enable-thinking", action="store_true", help="enable Gemma thinking channel (default off)")
    ap.add_argument("--concurrency", type=int, default=32, help="in-flight requests (exploit server batching)")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="server decode concurrency override")
    ap.add_argument("--save-text", action="store_true", help="persist raw completion text per item")
    ap.add_argument("--serve-env", action="append", default=[], metavar="KEY=VAL",
                    help="extra env override for the served submission (repeatable); e.g. LM_HEAD_PRUNE=0")
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path, default=Path("research/downstream_quality_gsm8k"))
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.submission and not args.base_url:
        ap.error("one of --submission or --base-url is required (or --self-test)")
    if not args.label:
        ap.error("--label is required for a model run")

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in regimes:
        if r not in ("sampled", "greedy"):
            ap.error(f"unknown regime {r!r} (want sampled|greedy)")

    # --- deterministic items + few-shot (shared across arms via seed) ---
    n_items = args.n
    if args.limit is not None:
        n_items = args.limit
    full = _load_split("test", n=None)
    import random

    rng = random.Random(args.seed)
    order = list(range(len(full)))
    rng.shuffle(order)
    if n_items is not None and n_items >= 0:
        order = order[:n_items]
    problems = []
    for rank, i in enumerate(order):
        problems.append(
            {
                "id": f"test-{i}",
                "question": full[i]["question"],
                "gold": gold_answer(full[i]["answer"]),
            }
        )
    exemplars, fewshot_sig = build_fewshot(args.n_shot, args.seed)
    print(f"[gsm8k] {len(problems)} test items (seed={args.seed}); {len(exemplars)}-shot "
          f"fewshot_sig={','.join(fewshot_sig)}", flush=True)

    # Decode RNG seed: --sampling-seed if given, else --seed (back-compat). The
    # subset/few-shot stay tied to --seed so the benchmark is byte-identical across
    # sampling seeds; only the per-request sampler RNG moves (PR #590 multi-seed CI).
    req_seed = args.sampling_seed if args.sampling_seed is not None else args.seed

    def _sampling(regime: str) -> dict[str, Any]:
        if regime == "greedy":
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                    "max_tokens": args.max_tokens, "seed": req_seed,
                    "enable_thinking": args.enable_thinking, "min_tokens": args.min_tokens}
        return {"temperature": 1.0, "top_p": args.top_p, "top_k": args.top_k,
                "max_tokens": args.max_tokens, "seed": req_seed,
                "enable_thinking": args.enable_thinking, "min_tokens": args.min_tokens}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def _run(base_url: str, model: str, submission: str | None, overrides: dict | None) -> None:
        for regime in regimes:
            s = _sampling(regime)
            print(f"[gsm8k] === arm={args.label} regime={regime} sampling={s} ===", flush=True)
            res = eval_endpoint(
                base_url, model, problems, exemplars,
                temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                max_tokens=s["max_tokens"], seed=s["seed"],
                enable_thinking=s["enable_thinking"],
                concurrency=args.concurrency, request_timeout_s=args.request_timeout_s,
                save_text=args.save_text, min_tokens=s["min_tokens"],
            )
            out = {
                "label": args.label,
                "regime": regime,
                "model": model,
                "submission": submission,
                "serve_overrides": overrides,
                "n_shot": args.n_shot,
                "fewshot_sig": fewshot_sig,
                "seed": args.seed,
                "sampling_seed": req_seed,
                "n_requested": n_items,
                "item_ids": [p["id"] for p in problems],
                "sampling": s,
                "max_num_seqs": args.max_num_seqs,
                "concurrency": args.concurrency,
                "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                **res,
            }
            seed_tag = "" if args.sampling_seed is None else f"_s{req_seed}"
            out_path = args.out_dir / f"{args.label}_{regime}{seed_tag}.json"
            out_path.write_text(json.dumps(out, indent=2))
            print(
                f"[gsm8k] DONE arm={args.label} regime={regime} acc={res['accuracy']:.4f} "
                f"({res['n_correct']}/{res['n_problems']}) strict={res['strict_rate']:.3f} "
                f"extract_fail={res['extract_fail_rate']:.3f} trunc={res['truncation_rate']:.3f} "
                f"wall={res['wall_s']:.0f}s -> {out_path}",
                flush=True,
            )

    if args.base_url:
        _run(args.base_url, args.model, None, None)
        return 0

    # Serve the submission locally and eval the live endpoint.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.local_validation import harness, paths  # noqa: E402

    for note in paths.prepare_local_gpu_env():
        print(f"[gsm8k] {note}", flush=True)
    manifest = harness.load_manifest(args.submission)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    overrides = local_serve_overrides(args.max_num_seqs)
    for kv in args.serve_env:
        if "=" not in kv:
            ap.error(f"--serve-env expects KEY=VAL, got {kv!r}")
        key, _, val = kv.partition("=")
        overrides[key.strip()] = val
    log_path = args.out_dir / f"server_{args.label}.log"
    print(f"[gsm8k] serving {args.submission} (overrides={overrides}; log: {log_path})", flush=True)
    with harness.LocalServer(
        args.submission,
        server_python=server_python,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        log_path=log_path,
        extra_env=overrides,
    ) as srv:
        _run(srv.base_url, srv.served_model_name, str(args.submission), overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
