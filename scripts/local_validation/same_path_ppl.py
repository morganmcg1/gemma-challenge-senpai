"""Same-path PPL — score the ground-truth continuations through the TIMED
generation path, not the ``prompt_logprobs`` scoring endpoint.

``ppl_runner.py`` (and the official ``ppl_endpoint.py``) measure PPL by sending
the integer-token prompt with ``prompt_logprobs`` set. That is the *scored*
quality path. The leaderboard TPS, however, is measured on the plain
``/v1/completions`` greedy-generation path, which never sends ``prompt_logprobs``.
A submission that branches on the presence of ``prompt_logprobs`` can serve a
clean model when scored for quality but a faster, lossier config when timed —
passing the PPL gate while shipping a different model to the leaderboard.

This probe closes that blind spot. It teacher-forces the *same* 61,797-token
ground-truth span used by ``ppl_endpoint.py``, but reads the per-token logprobs
through ``echo: true`` + ``logprobs`` on ``/v1/completions`` — a request that
carries **no** ``prompt_logprobs`` field and otherwise mirrors the timed
generation config (``add_special_tokens: false``, ``temperature: 0.0``,
``ignore_eos: true``). On an honest single-path model both paths run the same
prefill forward pass, so the two PPL numbers agree to floating-point noise
(``submissions/vllm_baseline`` ≈ 2.30 on both, gap ≈ 0.00). A material gap is the
signature of a timed-vs-scored path split.

Two modes mirror ``ppl_runner``:

  * ``--base-url URL``  : score an endpoint you already have running.
  * ``--submission DIR``: serve the submission locally (PPL headroom env
                          applied) and then score it.

The summary field names (``ppl``, ``num_tokens``, ``model`` …) match
``ppl_endpoint.py`` so ``same_path_ppl_summary.json`` and ``ppl_summary.json``
are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import harness, paths
from .ppl_runner import _headroom_overrides

DEFAULT_TIMEOUT_S = 120
# A 1-token generation request still exercises a decode step (it looks like the
# timed path) while echo returns logprobs for every prompt token; we only score
# the prompt span, so the single generated token is ignored.
DEFAULT_MAX_TOKENS = 1
DEFAULT_ECHO_LOGPROBS = 1


def post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
    return json.loads(response_body)


def request_echo_logprobs(
    *,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    timeout_s: int,
    echo_logprobs: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Force the GT tokens through the generation path and ask for their logprobs.

    Deliberately omits ``prompt_logprobs`` — this is the whole point of the gate.
    ``echo: true`` makes the endpoint return the prompt's own per-token logprobs
    via ``choices[0].logprobs.token_logprobs``; everything else matches the timed
    ``decode_outputs.py`` request shape so a path-splitting submission cannot
    distinguish this from real throughput traffic by request fields.
    """
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "echo": True,
        "logprobs": echo_logprobs,
        "add_special_tokens": False,
        "ignore_eos": True,
    }
    return post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout_s)


def echo_token_logprobs(response: dict[str, Any]) -> list[Any]:
    """Pull the echoed per-prompt-token logprobs out of a completions response.

    With ``echo: true`` the OpenAI/vLLM completions API returns one entry per
    prompt token (index 0 is ``null`` — the first token has no left context),
    aligned 1:1 with the integer prompt we sent, then the generated token(s).
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("same-path response did not include choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("same-path response choice is not an object")
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError(
            "same-path response missing choices[0].logprobs — the endpoint must "
            "honor echo:true + logprobs and return prompt-token logprobs"
        )
    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        raise ValueError("same-path response logprobs missing a token_logprobs list")
    return token_logprobs


def score_record(
    record: dict[str, Any],
    *,
    base_url: str,
    model: str,
    timeout_s: int,
    echo_logprobs: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Teacher-forced NLL over a record's scored span via the echo logprobs.

    Mirrors ``ppl_endpoint.score_record`` exactly (same span, same aggregation)
    but extracts the forced-token logprob from the by-position ``token_logprobs``
    list instead of the ``prompt_logprobs`` dict. With ``add_special_tokens:
    false`` and an integer prompt, ``token_logprobs[i]`` is the logprob the model
    assigned to our forced prompt token ``i``.
    """
    started_at = time.time()
    response = request_echo_logprobs(
        base_url=base_url,
        model=model,
        prompt_token_ids=record["prompt_token_ids"],
        timeout_s=timeout_s,
        echo_logprobs=echo_logprobs,
        max_tokens=max_tokens,
    )
    token_logprobs = echo_token_logprobs(response)
    prompt_token_ids = record["prompt_token_ids"]
    score_start = record["score_start"]
    score_end = record["score_end"]
    if len(token_logprobs) < score_end:
        raise ValueError(
            f"received {len(token_logprobs)} echo token logprobs for {score_end} scored prompt tokens"
        )

    scored: list[float] = []
    for index in range(score_start, score_end):
        value = token_logprobs[index]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"echo token logprob at prompt index {index} (token {prompt_token_ids[index]}) "
                f"is {value!r}; expected a float — the endpoint did not return the forced-token "
                "logprob on the echo path"
            )
        scored.append(float(value))

    neg_log_likelihood = -sum(scored)
    num_tokens = len(scored)
    return {
        "id": record["id"],
        "num_tokens": num_tokens,
        "neg_log_likelihood": neg_log_likelihood,
        "ppl": math.exp(neg_log_likelihood / num_tokens),
        "duration_s": time.time() - started_at,
        "score_start": score_start,
        "score_end": score_end,
    }


def score_endpoint(
    base_url: str,
    model: str,
    *,
    out_dir: Path,
    dataset: Path | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    echo_logprobs: int = DEFAULT_ECHO_LOGPROBS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score the full GT set on a running endpoint; write the comparable summary.

    Reused by ``validate_submission --check-same-path`` so the gate and the
    standalone CLI share one scoring implementation.
    """
    dataset = dataset or paths.ppl_dataset()
    ppl_endpoint = paths.import_ppl_endpoint()
    records = [
        ppl_endpoint.normalized_record(record, index)
        for index, record in enumerate(ppl_endpoint.read_records(dataset))
    ]
    if not records:
        raise ValueError(f"no PPL records found in {dataset}")
    if limit is not None:
        records = records[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "same_path_ppl_results.jsonl"
    total_nll = 0.0
    total_tokens = 0
    record_ppls: list[float] = []
    with results_file.open("w") as handle:
        for record in records:
            result = score_record(
                record,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                echo_logprobs=echo_logprobs,
                max_tokens=max_tokens,
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
        "dataset_path": str(dataset),
        "output_file": str(results_file),
        "model": model,
        "base_url": base_url,
        "path": "same_path_echo_logprobs",
        "echo": True,
        "logprobs": echo_logprobs,
        "max_tokens": max_tokens,
    }
    summary_file = out_dir / "same_path_ppl_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base-url", help="score an already-running endpoint")
    src.add_argument("--submission", type=Path, help="serve this submission dir, then score it")
    ap.add_argument("--model", default=paths.DEFAULT_SERVED_NAME, help="served model name (for --base-url)")
    ap.add_argument("--dataset", type=Path, default=None, help="PPL ground-truth jsonl (default: official mirror)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir for same_path_ppl_summary.json")
    ap.add_argument("--server-python", type=Path, default=None, help="python with vLLM (default: build from manifest deps)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="generation length of the echo request (>=1 keeps it on the decode path)")
    ap.add_argument("--echo-logprobs", type=int, default=DEFAULT_ECHO_LOGPROBS, help="logprobs count requested with echo:true")
    ap.add_argument("--request-timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N records (smoke)")
    ap.add_argument("--no-ensure-headroom", dest="ensure_headroom", action="store_false")
    ap.add_argument("--wandb-name", default=None, help="log the same-path PPL summary to W&B under this run name")
    ap.add_argument("--wandb-group", default=None, help="W&B group (e.g. same-path-ppl-gate)")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[same-path-ppl] {note}", flush=True)

    dataset = args.dataset or paths.ppl_dataset()
    out_dir = args.out_dir or (paths.LOCALRUN_ROOT / f"same-path-ppl-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")

    def _emit(summary: dict[str, Any]) -> None:
        print(
            f"SAME_PATH_PPL={summary['ppl']:.4f} num_tokens={summary['num_tokens']} "
            f"model={summary['model']} -> {out_dir / 'same_path_ppl_summary.json'}",
            flush=True,
        )
        _maybe_log_wandb(args, summary)

    if args.base_url:
        summary = score_endpoint(
            args.base_url,
            args.model,
            out_dir=out_dir,
            dataset=dataset,
            timeout_s=args.request_timeout_s,
            echo_logprobs=args.echo_logprobs,
            max_tokens=args.max_tokens,
            limit=args.limit,
        )
        _emit(summary)
        return 0

    # Serve-then-score mode (mirrors ppl_runner: same headroom env).
    manifest = harness.load_manifest(args.submission)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    overrides = _headroom_overrides(manifest.get("env", {})) if args.ensure_headroom else {}
    if overrides:
        print(f"[same-path-ppl] injecting headroom env: {overrides}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "server.log"
    with harness.LocalServer(
        args.submission, server_python=server_python, port=args.port, log_path=log, extra_env=overrides
    ) as srv:
        summary = score_endpoint(
            srv.base_url,
            srv.served_model_name,
            out_dir=out_dir,
            dataset=dataset,
            timeout_s=args.request_timeout_s,
            echo_logprobs=args.echo_logprobs,
            max_tokens=args.max_tokens,
            limit=args.limit,
        )
    _emit(summary)
    return 0


def _maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    """Best-effort W&B log; a no-op without creds (init returns None)."""
    if not args.wandb_name:
        return
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover - logging must never break scoring
        print(f"[same-path-ppl] wandb logging unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="same-path-ppl",
        agent="senpai",
        name=args.wandb_name,
        tags=["same-path-ppl-gate", *( [args.wandb_group] if args.wandb_group else [] )],
        config={
            "path": summary.get("path"),
            "max_tokens": summary.get("max_tokens"),
            "echo_logprobs": summary.get("logprobs"),
            "dataset_path": summary.get("dataset_path"),
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        return
    log_summary(run, summary, step=0)
    finish_wandb(run)


if __name__ == "__main__":
    raise SystemExit(main())
