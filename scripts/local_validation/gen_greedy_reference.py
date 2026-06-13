"""Generate the exact-greedy reference decode_outputs.jsonl for the 128-prompt set.

The challenge's hard validity rule is: a submission's served greedy decode must be
token-identical to *plain greedy autoregressive (M=1) decode of the same
checkpoint*. This builds that reference. There are two modes, and which one you
pick is the whole game:

  served   (default — the canonical gate anchor)
      Serve the checkpoint through the SAME plain vLLM ``api_server`` path a
      submission uses, with speculation / drafter / lossy optimizations OFF
      (M=1 AR), and capture decode through the official ``decode_outputs.py``.
      Because the reference and a candidate then share the serving engine and
      kernels, the ONLY difference for a drafter/spec submission is speculation,
      so a DIVERGENT verdict is attributable to the optimization-under-test — not
      to cross-engine noise. This is the reference the greedy gate and
      ``validate_submission`` resolve by default.

  offline  (independent AR cross-check — NOT the gate anchor)
      vLLM offline ``LLM.generate``, temperature 0, no speculation. A pure-engine
      AR anchor. IMPORTANT: the offline path is NOT bit-identical to the served
      ``api_server`` path. On bf16 E4B at output_len 512 the two diverge on ~20%
      of prompts purely from floating-point reduction non-determinism + argmax
      tie-breaking — a stochastic subset, divergence onset scattered across the
      sequence, independent of prompt length (it is not a chunked-prefill
      artifact). So comparing an offline reference against a *served* candidate
      yields false DIVERGENT verdicts; use it only as a diagnostic second source,
      never as the hard gate.

Tokenization and record IDs come from the official ``decode_outputs.py`` so the
reference lines up prompt-for-prompt with a live endpoint capture.

Spec stacks (drafter/MTP/speculative submissions): the canonical reference must
be the SAME checkpoint decoded M=1 AR (speculation OFF) — kanna (#5) measured
int4 M=1 AR vs M=K+1 batched-verify diverging ~0.33%/tok, so a spec submission's
reference cannot be borrowed from the bf16 base or a verify-path capture. Pass
``--submission <dir> --spec-off`` to serve the submission's own engine with
``SENPAI_REFERENCE_MODE=1`` injected; a drafter ``serve.py`` honoring that
contract (see ``paths.REFERENCE_MODE_ENV``) then serves plain M=1 AR, isolating
speculation as the only removed variable. Submissions with a non-standard knob
can add ``--ref-env KEY=VALUE`` (which wins over ``--spec-off``).

Run with a python that has vLLM (the server venv), from the repo root:
    # plain base checkpoint (baseline can't speculate -> already M=1 AR):
    /tmp/server-venv/bin/python -m scripts.local_validation.gen_greedy_reference \\
        --mode served --model-id google/gemma-4-E4B-it --num-prompts 128
    # a drafter/spec submission, forced to M=1 AR for its canonical reference:
    /tmp/server-venv/bin/python -m scripts.local_validation.gen_greedy_reference \\
        --mode served --submission submissions/<drafter> --spec-off
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from . import harness, paths


def _load_decode_helpers():
    """Import the official decode_outputs.py module by path (stdlib-only top level)."""
    spec = importlib.util.spec_from_file_location("official_decode_outputs", paths.DECODE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_env_kvs(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--ref-env KEY=VALUE`` flags into an env override dict."""
    env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--ref-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def served_reference_path(model_id: str) -> Path:
    """Canonical (served) reference path for a checkpoint — the gate anchor."""
    return paths.REFERENCE_ROOT / paths.model_tag(model_id) / "decode_outputs.jsonl"


def offline_reference_path(model_id: str) -> Path:
    """Offline AR cross-check path — kept beside the canonical served reference."""
    return paths.REFERENCE_ROOT / paths.model_tag(model_id) / "decode_outputs.offline.jsonl"


# --------------------------------------------------------------------------- #
# served mode (canonical)
# --------------------------------------------------------------------------- #
def generate_served(args: argparse.Namespace) -> int:
    """Serve the checkpoint plain (spec-off) and capture the reference decode."""
    for note in paths.prepare_local_gpu_env():
        print(f"[ref] {note}", flush=True)

    ref_env = _parse_env_kvs(args.ref_env)
    # --spec-off injects the documented reference-mode contract env so a drafter
    # submission's serve.py decodes plain M=1 AR. It is the base layer; an
    # explicit --ref-env always wins so a submission with a non-standard knob can
    # still be steered.
    spec_off_env = {paths.REFERENCE_MODE_ENV: "1"} if args.spec_off else {}
    if args.submission:
        # Serve the submission's OWN stack with speculation disabled so the
        # reference is M=1 AR on the same engine/kernels/quant — the only removed
        # variable is speculation.
        submission = args.submission
        manifest = harness.load_manifest(submission)
        model_id = harness.resolve_model_id(str(manifest.get("model_id", paths.BF16_MODEL)), submission)
        extra_env = {**spec_off_env, **ref_env}
        served_via = f"submission:{submission}" + (" [spec-off]" if args.spec_off else "")
    else:
        # Serve the canonical plain baseline pointed at the checkpoint. The
        # baseline serve.py is inherently spec-off, so this is the M=1 AR anchor
        # for any submission that shares this base checkpoint.
        submission = args.baseline_submission
        manifest = harness.load_manifest(submission)
        model_id = args.model_id
        extra_env = {"MODEL_ID": model_id, "SERVED_MODEL_NAME": paths.DEFAULT_SERVED_NAME, **spec_off_env, **ref_env}
        served_via = f"plain-baseline:{submission} (MODEL_ID={model_id})"

    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    out = args.out or served_reference_path(model_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_file = out.parent / "decode_summary.json"
    log_path = out.parent / "served_reference_server.log"

    # The reference is only a valid M=1 AR anchor if speculation is actually off:
    # always true for the plain baseline, and for a submission only when
    # --spec-off and/or --ref-env disabled its drafter.
    spec_disabled = (not args.submission) or args.spec_off or bool(ref_env)
    ref_kind = "served_spec_off" if spec_disabled else "served_spec_ON_INVALID"

    print(f"[ref] served mode: {served_via} -> capture "
          f"{args.num_prompts} prompts x {args.output_len} tok (greedy, seed {args.seed})", flush=True)
    if args.spec_off:
        print(f"[ref] spec-off: injecting {paths.REFERENCE_MODE_ENV}=1 (submission must honor it for M=1 AR)", flush=True)
    if ref_env:
        print(f"[ref] reference env overrides: {ref_env}", flush=True)
    if not spec_disabled:
        print("[ref] WARNING: serving a submission with speculation still ON — this capture is NOT a "
              "canonical greedy reference. Re-run with --spec-off (or --ref-env) to disable the drafter.",
              flush=True)

    t0 = time.time()
    with harness.LocalServer(
        submission, server_python=server_python, port=args.port,
        log_path=log_path, extra_env=extra_env,
    ) as srv:
        summary = harness.capture_decode(
            server_python,
            base_url=srv.base_url,
            model=srv.served_model_name,
            out_file=out,
            summary_file=summary_file,
            num_prompts=args.num_prompts,
            output_len=args.output_len,
            seed=args.seed,
            tokenizer=args.tokenizer,
            dataset=args.dataset,
        )
        served_model_name = srv.served_model_name
        resolved_model_id = srv.model_id
    gen_s = time.time() - t0

    meta = {
        "model_id": resolved_model_id,
        "reference_kind": ref_kind,
        "served_via": served_via,
        "served_model_name": served_model_name,
        "spec_off": args.spec_off,
        "reference_mode_env": spec_off_env,
        "ref_env": ref_env,
        "num_records": summary["num_records"],
        "num_completion_tokens": summary["num_completion_tokens"],
        "output_len": args.output_len,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "dataset": str(args.dataset),
        "capture_wall_s": gen_s,
        "output_file": str(out),
    }
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[ref] wrote served reference {out} "
          f"({summary['num_records']} records, {summary['num_completion_tokens']} completion tokens) "
          f"in {gen_s:.0f}s", flush=True)
    print(f"[ref] meta -> {out.parent / 'meta.json'}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# offline mode (diagnostic cross-check)
# --------------------------------------------------------------------------- #
def _auto_quantization(model_id: str, override: str | None) -> str | None:
    if override is not None:
        return override or None
    low = model_id.lower()
    if "w4a16" in low or "qat" in low or "int4" in low or "compressed" in low:
        return "compressed-tensors"
    return None


def _generate(llm, sp, token_id_lists: list[list[int]]):
    """Drive offline greedy generation across vLLM API shapes for token-id prompts.

    vLLM returns outputs in input order, so the caller maps results positionally.
    """
    # Preferred modern API: list of TokensPrompt objects.
    try:
        from vllm import TokensPrompt  # type: ignore

        return llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in token_id_lists], sp)
    except Exception:
        pass
    # Dict-prompt form (stable across many vLLM versions).
    try:
        return llm.generate([{"prompt_token_ids": ids} for ids in token_id_lists], sp)
    except Exception:
        pass
    # Legacy kwarg API.
    return llm.generate(prompt_token_ids=token_id_lists, sampling_params=sp)


def generate_offline(args: argparse.Namespace) -> int:
    """vLLM offline LLM.generate AR reference (diagnostic cross-check only)."""
    for note in paths.prepare_local_gpu_env():
        print(f"[ref] {note}", flush=True)

    out = args.out or offline_reference_path(args.model_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    dec = _load_decode_helpers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = dec.read_sharegpt_prompts(args.dataset, num_prompts=args.num_prompts, seed=args.seed)
    if len(records) != args.num_prompts:
        raise SystemExit(f"expected {args.num_prompts} prompts, found {len(records)}")
    prompt_token_ids = [dec.encode_prompt(tokenizer, r["prompt_text"]) for r in records]

    quant = _auto_quantization(args.model_id, args.quantization)
    print(f"[ref] OFFLINE cross-check loading {args.model_id} dtype={args.dtype} quant={quant} "
          f"graphs={'off' if args.enforce_eager else 'on'}", flush=True)
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=args.model_id,
        dtype=args.dtype,
        quantization=quant,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        disable_log_stats=True,
        seed=args.seed,
    )
    print(f"[ref] model ready in {time.time() - t0:.0f}s; decoding {len(records)} prompts x {args.output_len} tok (greedy)", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    t0 = time.time()
    outputs = _generate(llm, sp, prompt_token_ids)
    gen_s = time.time() - t0
    if len(outputs) != len(records):
        raise SystemExit(f"vLLM returned {len(outputs)} outputs for {len(records)} prompts")

    # vLLM preserves input order; map outputs back positionally, but sanity-check
    # the prompt token ids when the engine surfaces them.
    total_completion = 0
    with out.open("w", encoding="utf-8") as fh:
        for index, (rec, pids) in enumerate(zip(records, prompt_token_ids)):
            o = outputs[index]
            o_pids = getattr(o, "prompt_token_ids", None)
            if o_pids is not None and list(o_pids) != list(pids):
                raise SystemExit(f"output order mismatch at index {index} (id={rec['id']})")
            comp = list(o.outputs[0].token_ids)
            total_completion += len(comp)
            row = {
                "id": rec["id"],
                "index": index,
                "dataset_index": rec["dataset_index"],
                "prompt_text": rec["prompt_text"],
                "prompt_sha256": dec.sha256_text(rec["prompt_text"]),
                "prompt_token_ids": pids,
                "prompt_token_sha256": dec.sha256_tokens(pids),
                "generated_text": tokenizer.decode(comp),
                "completion_token_ids": comp,
                "completion_token_sha256": dec.sha256_tokens(comp),
                "num_prompt_tokens": len(pids),
                "num_completion_tokens": len(comp),
                "reference_kind": "plain_greedy_ar_offline_vllm",
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "model_id": args.model_id,
        "reference_kind": "plain_greedy_ar_offline_vllm",
        "num_records": len(records),
        "num_completion_tokens": total_completion,
        "output_len": args.output_len,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "quantization": quant,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "generate_wall_s": gen_s,
        "output_file": str(out),
        "note": "offline LLM.generate AR cross-check; NOT bit-identical to the served "
                "api_server path — diverges on a stochastic ~20% prompt subset at "
                "output_len 512 from FP non-determinism. Not the gate anchor.",
    }
    (out.parent / "meta.offline.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[ref] wrote OFFLINE cross-check {out} ({len(records)} records, {total_completion} completion tokens) in {gen_s:.0f}s", flush=True)
    print(f"[ref] meta -> {out.parent / 'meta.offline.json'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["served", "offline"], default="served",
                    help="served = canonical gate anchor (default); offline = diagnostic AR cross-check")
    ap.add_argument("--model-id", default=paths.BF16_MODEL)
    ap.add_argument("--out", type=Path, default=None, help="output decode_outputs.jsonl (default: research/greedy_reference/<tag>/)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--tokenizer", default=paths.TOKENIZER)
    ap.add_argument("--dataset", type=Path, default=paths.EVAL_PROMPTS)
    # served-only
    ap.add_argument("--submission", type=Path, default=None,
                    help="[served] serve this submission's own stack (spec-off via --ref-env) as the reference")
    ap.add_argument("--baseline-submission", type=Path, default=paths.ROOT / "submissions" / "vllm_baseline",
                    help="[served] plain vLLM submission used to serve a bare --model-id (default: submissions/vllm_baseline)")
    ap.add_argument("--server-python", type=Path, default=None, help="[served] python with vLLM (default: build from manifest deps)")
    ap.add_argument("--spec-off", action="store_true",
                    help=f"[served] inject {paths.REFERENCE_MODE_ENV}=1 so a drafter submission's serve.py "
                         "decodes plain M=1 AR (the one-flag canonical reference for spec stacks)")
    ap.add_argument("--ref-env", action="append", default=None,
                    help="[served] extra KEY=VALUE env override to disable a non-standard drafter (repeatable; wins over --spec-off)")
    ap.add_argument("--port", type=int, default=8001, help="[served] port for the reference server (default 8001)")
    # offline-only
    ap.add_argument("--dtype", default="bfloat16", help="[offline] vLLM dtype")
    ap.add_argument("--max-model-len", type=int, default=4096, help="[offline]")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="[offline]")
    ap.add_argument("--max-num-batched-tokens", type=int, default=512, help="[offline]")
    ap.add_argument("--quantization", default=None, help="[offline] vLLM quantization (default: auto from model id)")
    ap.add_argument("--enforce-eager", action="store_true", help="[offline] disable CUDA graphs (default: graphs on)")
    args = ap.parse_args(argv)

    if args.mode == "served":
        return generate_served(args)
    return generate_offline(args)


if __name__ == "__main__":
    raise SystemExit(main())
