"""Local AIME maj@k eval for the downstream-quality triad (PR #514).

Completes the MMLU-Pro / GPQA-Diamond / **AIME** quality triad that the challenge
is paused on. The deliverable is the **base-vs-ship A/B delta** under *real
sampling*: stock int4 ``gemma-4-E4B-it`` vs the served surgical-357 ship, at the
model's native ``generation_config`` (``do_sample:true, T=1.0, top_k=64,
top_p=0.95``). Greedy identity is already byte-exact (lawine #488); the new thing
this measures is whether the ship preserves the *sampled* distribution end to end
(denken #505, TV <= noise) on the hardest, fully-generative benchmark.

Why a custom maj@k harness and not ``inspect_evals``: the mandated library is not
installed in this container, and bending ``inspect_ai``'s provider plumbing to a
dual-endpoint A/B with maj@k under one A10G + a 90-min run bound is more fragile
than a small, auditable harness. We mirror the ``inspect_evals`` AIME conventions
(``\\boxed{}`` integer extraction, integer-match scoring, majority vote over k
samples) so the numbers stay comparable to the taskforce's stack. The taskforce
runs AIME *greedy* for cross-submission regression; this is the complementary
*sampled* leg.

Two ways to drive a model:
  * ``--submission <dir>``  -- stand the submission up via the local-validation
    ``LocalServer`` (manifest deps + env, with the local-serving overrides this
    script injects) and eval the live endpoint.
  * ``--base-url <url>``    -- eval an endpoint someone else already started.

Output: one JSON with per-problem samples, extracted answers, maj@k correctness,
and the continuous per-problem pass-rate (a finer distribution-match signal than
the discrete maj@k bit). Pair two of these (base + ship) with ``aime_combine.py``.
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
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Dataset loading (no `datasets` dep — HF datasets-server /rows API)
# --------------------------------------------------------------------------- #
# AIME is 30 problems/year, so one /rows page (length<=100) holds a whole split.
AIME_REGISTRY: dict[str, dict[str, str]] = {
    "2024": {
        "dataset": "Maxwell-Jia/AIME_2024",
        "config": "default",
        "split": "train",
        "id_col": "ID",
        "problem_col": "Problem",
        "answer_col": "Answer",
    },
    # opencompass/AIME2025 ships two configs (AIME2025-I / -II), 15 problems each.
    "2025-I": {
        "dataset": "opencompass/AIME2025",
        "config": "AIME2025-I",
        "split": "test",
        "id_col": "",
        "problem_col": "question",
        "answer_col": "answer",
    },
    "2025-II": {
        "dataset": "opencompass/AIME2025",
        "config": "AIME2025-II",
        "split": "test",
        "id_col": "",
        "problem_col": "question",
        "answer_col": "answer",
    },
}


def _rows_api(dataset: str, config: str, split: str, length: int = 100) -> list[dict[str, Any]]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
        f"&split={urllib.parse.quote(split)}&offset=0&length={length}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "senpai-aime-eval"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return [row["row"] for row in data.get("rows", [])]


def load_aime(years: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for year in years:
        spec = AIME_REGISTRY[year]
        rows = _rows_api(spec["dataset"], spec["config"], spec["split"])
        for i, row in enumerate(rows):
            ans_raw = row[spec["answer_col"]]
            ans = _to_int(ans_raw)
            if ans is None:
                continue  # AIME answers are integers 0-999; skip anything malformed
            pid = str(row[spec["id_col"]]) if spec["id_col"] else f"{year}-{i+1:02d}"
            problems.append(
                {
                    "id": pid,
                    "year": year,
                    "problem": str(row[spec["problem_col"]]),
                    "answer": ans,
                }
            )
    if limit is not None:
        problems = problems[:limit]
    return problems


# --------------------------------------------------------------------------- #
# Answer extraction (mirrors inspect_evals AIME: last \boxed{} integer)
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_INT_RE = re.compile(r"-?\d[\d,]*")


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().replace(",", "")
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _extract_boxed_spans(text: str) -> list[str]:
    """Return the brace-balanced contents of every ``\\boxed{...}`` in order."""
    spans: list[str] = []
    for m in _BOXED_RE.finditer(text):
        i = m.end()  # position just after the '{'
        depth = 1
        out: list[str] = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(c)
            i += 1
        spans.append("".join(out))
    return spans


def extract_answer(text: str) -> int | None:
    """AIME integer answer from a completion.

    Priority: the LAST ``\\boxed{...}`` that contains an integer (the final answer
    after any thinking channel); else the last ``0-999``-range integer token in the
    text. Returns ``None`` when nothing parses (counts as wrong, never crashes).
    """
    if not text:
        return None
    for span in reversed(_extract_boxed_spans(text)):
        val = _to_int(span)
        if val is not None:
            return val
    # Fallback: last integer in [0, 999] (AIME answer range) anywhere in the text.
    last: int | None = None
    for m in _INT_RE.finditer(text):
        v = _to_int(m.group(0))
        if v is not None and 0 <= v <= 999:
            last = v
    return last


def majority_vote(answers: list[int | None]) -> tuple[int | None, dict[str, int]]:
    """Most common non-None answer; deterministic tie-break by smallest value."""
    valid = [a for a in answers if a is not None]
    counts = Counter(valid)
    if not counts:
        return None, {}
    top = max(counts.values())
    winner = min(a for a, c in counts.items() if c == top)
    return winner, {str(k): v for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# Prompting + endpoint
# --------------------------------------------------------------------------- #
AIME_INSTRUCTION = (
    "Please reason step by step to solve the problem, and put your final answer "
    "(a single integer between 0 and 999) within \\boxed{}."
)


def build_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{problem}\n\n{AIME_INSTRUCTION}"}]


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
    timeout_s: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        # vLLM extension: top_k lives outside the OpenAI schema.
        "top_k": top_k,
    }
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
    *,
    k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
    request_timeout_s: int,
    save_text: bool = False,
) -> dict[str, Any]:
    per_problem: list[dict[str, Any]] = []
    n_correct_maj = 0
    pass_rates: list[float] = []
    extract_fail = 0
    total_samples = 0
    t0 = time.time()
    for idx, prob in enumerate(problems):
        messages = build_messages(prob["problem"])
        resp = chat_completion(
            base_url,
            model,
            messages,
            n=k,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            enable_thinking=enable_thinking,
            timeout_s=request_timeout_s,
        )
        texts = [c.get("message", {}).get("content") or "" for c in resp.get("choices", [])]
        finish = [c.get("finish_reason") for c in resp.get("choices", [])]
        answers = [extract_answer(t) for t in texts]
        total_samples += len(answers)
        extract_fail += sum(1 for a in answers if a is None)
        maj, counts = majority_vote(answers)
        gold = prob["answer"]
        correct_samples = sum(1 for a in answers if a == gold)
        pass_rate = correct_samples / len(answers) if answers else 0.0
        pass_rates.append(pass_rate)
        maj_correct = maj is not None and maj == gold
        n_correct_maj += int(maj_correct)
        per_problem.append(
            {
                "id": prob["id"],
                "year": prob["year"],
                "gold": gold,
                "answers": answers,
                "answer_counts": counts,
                "maj_answer": maj,
                "maj_correct": maj_correct,
                "correct_samples": correct_samples,
                "k": len(answers),
                "pass_rate": pass_rate,
                "finish_reasons": finish,
                "sample_chars": [len(t) for t in texts],
                **({"texts": texts} if save_text else {}),
            }
        )
        print(
            f"[aime] {idx+1}/{len(problems)} id={prob['id']} gold={gold} "
            f"maj={maj} ({'OK' if maj_correct else 'x'}) pass={correct_samples}/{len(answers)} "
            f"counts={counts}",
            flush=True,
        )
    n = len(problems)
    return {
        "n_problems": n,
        "maj_k": k,
        "maj_k_accuracy": n_correct_maj / n if n else 0.0,
        "n_correct_maj": n_correct_maj,
        "mean_pass_rate": sum(pass_rates) / n if n else 0.0,
        "extract_fail_rate": extract_fail / total_samples if total_samples else 0.0,
        "total_samples": total_samples,
        "wall_s": time.time() - t0,
        "per_problem": per_problem,
    }


# --------------------------------------------------------------------------- #
# Self-test (no GPU): prove the extractor is sound before any model run.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    cases: list[tuple[str, int | None]] = [
        ("After working it out, the answer is \\boxed{42}.", 42),
        ("<|think|>messy 17 then 200<|/think|> Final: \\boxed{073}", 73),
        ("two boxes \\boxed{1} ... and later \\boxed{ 204 }", 204),
        ("comma form \\boxed{1,024}", 1024),  # extractor strips commas (then out of range, see note)
        ("nested \\boxed{\\frac{3}{1}=3 so \\boxed{3}}", 3),
        ("no box, the value is 250 at the end", 250),
        ("garbage with no integer at all", None),
    ]
    ok = True
    for text, want in cases:
        got = extract_answer(text)
        # the comma case parses to 1024 which is out of AIME range only for the
        # *fallback*; a boxed value is taken verbatim, so 1024 is expected here.
        flag = "ok" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[self-test] {flag}: extract({text!r}) = {got} (want {want})")
    maj, counts = majority_vote([5, 5, 7, None, 5])
    if maj != 5:
        ok = False
        print(f"[self-test] FAIL majority_vote -> {maj} (want 5)")
    else:
        print(f"[self-test] ok: majority_vote -> {maj} counts={counts}")
    print("[self-test] PASS" if ok else "[self-test] FAILURES PRESENT")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Local-serving overrides for a submission (HF-Job-path -> local-friendly)
# --------------------------------------------------------------------------- #
def local_serve_overrides(extra_seqs: int) -> dict[str, str]:
    """Env overrides so a speed submission stands up on this single A10G.

    None of these touch the output *distribution* — they fix HF-Job-only paths and
    raise decode concurrency so maj@k is tractable:
      * PRECACHE -> point at an absent path so the precache patch skips+ungates
        (serve_patch_precache.py:143); precache is a latency warmup, not numerics.
      * MAX_NUM_SEQS -> raise from the deployed single-stream value so n=k samples
        batch (the ship is built batch/M-invariant, so per-sequence sampling is
        unchanged by batch size; we use the SAME value for base and ship).
    """
    return {
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_aime_no_precache.json",
        "MAX_NUM_SEQS": str(extra_seqs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="run the extractor self-test and exit (no GPU)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--submission", type=Path, help="serve this submission dir via LocalServer, then eval")
    src.add_argument("--base-url", help="eval an already-running endpoint")
    ap.add_argument("--server-python", type=Path, default=None, help="python with vLLM (default: build from manifest deps)")
    ap.add_argument("--model", default="gemma-4-e4b-it", help="served model name")
    ap.add_argument("--years", default="2024", help="comma list from {2024,2025-I,2025-II}")
    ap.add_argument("--k", type=int, default=8, help="maj@k samples per problem")
    ap.add_argument("--limit", type=int, default=None, help="cap number of problems (smoke)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-thinking", action="store_true", help="disable enable_thinking chat-template kwarg")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="decode concurrency override for serving")
    ap.add_argument("--save-text", action="store_true", help="persist raw completion text per problem (diagnostics)")
    ap.add_argument("--serve-env", action="append", default=[], metavar="KEY=VAL", help="extra env override for the served submission (repeatable); e.g. SENPAI_REFERENCE_MODE=1 to ablate spec-dec")
    ap.add_argument("--request-timeout-s", type=int, default=1200)
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--label", default=None, help="label for the output (e.g. base / ship)")
    ap.add_argument("--out", type=Path, required=False, help="results JSON path")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.submission and not args.base_url:
        ap.error("one of --submission or --base-url is required (or --self-test)")

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    problems = load_aime(years, limit=args.limit)
    print(f"[aime] loaded {len(problems)} problems from years={years}", flush=True)

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "enable_thinking": not args.no_thinking,
    }
    meta = {
        "label": args.label,
        "model": args.model,
        "years": years,
        "k": args.k,
        "sampling": sampling,
        "max_num_seqs": args.max_num_seqs,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }

    def _run(base_url: str) -> dict[str, Any]:
        return eval_endpoint(
            base_url,
            args.model,
            problems,
            k=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
            enable_thinking=not args.no_thinking,
            request_timeout_s=args.request_timeout_s,
            save_text=args.save_text,
        )

    if args.base_url:
        meta["base_url"] = args.base_url
        result = _run(args.base_url)
    else:
        # Serve the submission locally and eval the live endpoint.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.local_validation import harness, paths  # noqa: E402

        for note in paths.prepare_local_gpu_env():
            print(f"[aime] {note}", flush=True)
        manifest = harness.load_manifest(args.submission)
        server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
        overrides = local_serve_overrides(args.max_num_seqs)
        for kv in args.serve_env:
            if "=" not in kv:
                ap.error(f"--serve-env expects KEY=VAL, got {kv!r}")
            key, _, val = kv.partition("=")
            overrides[key.strip()] = val
        meta["submission"] = str(args.submission)
        meta["serve_overrides"] = overrides
        log_path = (args.out.parent if args.out else Path(".")) / f"server_{args.label or 'model'}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[aime] serving {args.submission} (log: {log_path})", flush=True)
        with harness.LocalServer(
            args.submission,
            server_python=server_python,
            port=args.port,
            startup_timeout_s=args.startup_timeout_s,
            log_path=log_path,
            extra_env=overrides,
        ) as srv:
            meta["model"] = srv.served_model_name
            args.model = srv.served_model_name
            result = _run(srv.base_url)

    out = {**meta, **result}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"[aime] wrote {args.out}", flush=True)
    print(
        f"[aime] DONE label={args.label} maj@{args.k}={result['maj_k_accuracy']:.4f} "
        f"({result['n_correct_maj']}/{result['n_problems']}) "
        f"mean_pass_rate={result['mean_pass_rate']:.4f} "
        f"extract_fail_rate={result['extract_fail_rate']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
