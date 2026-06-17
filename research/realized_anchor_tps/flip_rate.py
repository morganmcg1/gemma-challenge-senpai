#!/usr/bin/env python
"""PR #553 Stage 3a -- greedy-argmax flip rate: int4 head vs bf16-head anchor.

LOCAL, analysis-only. Quantifies how far the realized int4-262k-head (Stage 2)
departs from the bf16-262k-head quality-safe anchor (Stage 1) under the HARD
#319 strict-byte-exact greedy gate. Body + attention are byte-identical between
the two arms (same 11.5GB blob, same surgical kernels) -- ONLY the lm_head
precision differs -- so any divergence is attributable purely to the int4 head.

Two complementary measurements on a held-out ShareGPT stream:

  * FREE-RUN identity (bulletproof): each arm free-runs greedy (temperature=0,
    ignore_eos) over the SAME prompts. Compare token-IDs. ``strict_identity`` =
    every stream byte-identical. First-divergence positions give a censored
    per-token hazard (flips/positions-before-first-divergence).

  * TEACHER-FORCED flip rate (precise, uncensored): the bf16 reference
    trajectory R is fed back as the prompt to the int4 endpoint with
    ``prompt_logprobs=1``; the per-position rank-1 token is the int4 head's
    argmax GIVEN the identical hidden state (body+attn shared). Disagreement
    vs R at a position is a genuine argmax flip. A bf16-on-R self-pass
    validates that the prompt_logprobs readout reproduces R (== argmax).

Run under the SERVER venv (needs transformers tokenizer + vllm serve):
    CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
        research/realized_anchor_tps/flip_rate.py --n-prompts 32 --output-len 256
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "research" / "realized_anchor_tps"
SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
QAT_SNAPSHOT = Path(
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
INT4_DIR = Path("/tmp/base-int4-lmhead")

# arm serve-env diffs (everything else identical -> precision-only delta)
ARM_ENV: dict[str, dict[str, str]] = {
    "bf16": {
        "LOCAL_MODEL_DIR": str(QAT_SNAPSHOT),
        "PLE_FOLD_TARGET_MODEL": str(QAT_SNAPSHOT),
        "LM_HEAD_PRUNE": "0", "LM_HEAD_PRUNE_REQUIRE": "0",
        "LM_HEAD_FULL_REQUIRE": "1", "PCK04_KEEPSET": "",
    },
    "int4": {
        "LOCAL_MODEL_DIR": str(INT4_DIR),
        "PLE_FOLD_TARGET_MODEL": str(INT4_DIR),
        "LM_HEAD_PRUNE": "0", "LM_HEAD_PRUNE_REQUIRE": "0",
        "LM_HEAD_FULL_REQUIRE": "1", "PCK04_KEEPSET": "",
        "LMHEAD_INT4_SKIP_STRAY": "1",
    },
}


# ========================================================================== #
# Worker (runs UNDER the server venv: needs transformers tokenizer)
# ========================================================================== #
def _load_decode_module():
    import importlib.util

    from scripts.local_validation import paths
    spec = importlib.util.spec_from_file_location("official_decode", str(paths.DECODE_SCRIPT))
    od = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(od)
    return od, paths


def _argmax_token_from_plogprob(entry: Any) -> int | None:
    """entry: {token_id: {logprob, rank, decoded_token}} for one prompt position.
    Return the rank-1 (argmax) token id, else the max-logprob token id."""
    if not isinstance(entry, dict) or not entry:
        return None
    best_tok = None
    best_lp = None
    for tok, info in entry.items():
        try:
            tok_id = int(tok)
        except (TypeError, ValueError):
            continue
        rank = info.get("rank") if isinstance(info, dict) else None
        if rank == 1:
            return tok_id
        lp = info.get("logprob") if isinstance(info, dict) else None
        if lp is not None and (best_lp is None or lp > best_lp):
            best_lp, best_tok = lp, tok_id
    return best_tok


def _teacher_force_argmax(od, base_url: str, model: str, prompt_ids: list[int],
                          ref_completion: list[int], timeout_s: int) -> list[int | None]:
    """Feed [prompt + ref_completion] with prompt_logprobs=1; read the rank-1
    token at each position that PREDICTS a ref-completion token. The prompt
    logprob at index i is the model's distribution for position i given tokens
    [0..i-1]; the token it predicts for completion position j (0-based) sits at
    prompt index len(prompt)+j."""
    full = list(prompt_ids) + list(ref_completion)
    payload = {
        "model": model, "prompt": full, "max_tokens": 1,
        "temperature": 0.0, "stream": False, "add_special_tokens": False,
        "ignore_eos": True, "prompt_logprobs": 1,
    }
    resp = od.post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout_s)
    choice = od.choice_from_response(resp)
    plps = choice.get("prompt_logprobs")
    if plps is None:
        plps = resp.get("prompt_logprobs")
    if not isinstance(plps, list):
        return []
    p = len(prompt_ids)
    preds: list[int | None] = []
    for j in range(len(ref_completion)):
        idx = p + j
        entry = plps[idx] if 0 <= idx < len(plps) else None
        preds.append(_argmax_token_from_plogprob(entry))
    return preds


def _worker(args: argparse.Namespace) -> int:
    od, paths = _load_decode_module()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = od.read_sharegpt_prompts(Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed)
    ref_rows = None
    if args.ref_file:
        ref_rows = [json.loads(l) for l in Path(args.ref_file).read_text().splitlines() if l.strip()]

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt_ids = od.encode_prompt(tok, record["prompt_text"])
        row: dict[str, Any] = {"index": index, "id": record["id"],
                               "num_prompt_tokens": len(prompt_ids)}
        if args.mode == "freerun":
            resp = od.request_decode(base_url=args.base_url, model=args.model,
                                     prompt_token_ids=prompt_ids, output_len=args.output_len,
                                     timeout_s=args.request_timeout_s)
            choice = od.choice_from_response(resp)
            comp, src, kind = od.extract_generated_token_ids(resp, choice, prompt_ids)
            row["completion_token_ids"] = comp
            row["num_completion_tokens"] = len(comp)
            print(f"[worker:{args.mode}] {index+1}/{len(records)} comp={len(comp)} src={src}", flush=True)
        else:  # teacherforce
            ref = ref_rows[index]
            assert ref["index"] == index
            ref_comp = ref["completion_token_ids"]
            preds = _teacher_force_argmax(od, args.base_url, args.model, prompt_ids,
                                          ref_comp, args.request_timeout_s)
            row["ref_completion_token_ids"] = ref_comp
            row["tf_argmax_token_ids"] = preds
            n_valid = sum(1 for x in preds if x is not None)
            n_match = sum(1 for x, r in zip(preds, ref_comp) if x is not None and x == r)
            row["tf_valid"] = n_valid
            row["tf_match"] = n_match
            print(f"[worker:{args.mode}] {index+1}/{len(records)} valid={n_valid} "
                  f"match={n_match}/{len(ref_comp)}", flush=True)
        rows.append(row)

    with Path(args.out_file).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return 0


# ========================================================================== #
# Comparison / metrics
# ========================================================================== #
def _compare_freerun(bf16_rows: list[dict], int4_rows: list[dict]) -> dict[str, Any]:
    n = len(bf16_rows)
    identical = 0
    first_div: list[int] = []          # position of first divergence (per diverged prompt)
    matched_prefix_positions = 0       # valid same-prefix positions before first flip
    n_diverged = 0
    lens: list[int] = []
    for a, b in zip(bf16_rows, int4_rows):
        ca, cb = a["completion_token_ids"], b["completion_token_ids"]
        L = min(len(ca), len(cb))
        lens.append(L)
        d = None
        for t in range(L):
            if ca[t] != cb[t]:
                d = t
                break
        if d is None and len(ca) == len(cb):
            identical += 1
            matched_prefix_positions += L  # all L positions matched, no flip
        else:
            n_diverged += 1
            dd = d if d is not None else L
            first_div.append(dd)
            matched_prefix_positions += dd + 1  # dd matched-prefix positions + the flip at dd
    hazard = (n_diverged / matched_prefix_positions) if matched_prefix_positions else float("nan")
    return {
        "n_prompts": n,
        "n_identical_prompts": identical,
        "frac_identical_prompts": identical / n if n else float("nan"),
        "n_diverged_prompts": n_diverged,
        "strict_identity": n_diverged == 0,
        "first_divergence_positions": sorted(first_div),
        "first_divergence_median": statistics.median(first_div) if first_div else None,
        "first_divergence_min": min(first_div) if first_div else None,
        "censored_hazard_flip_rate": hazard,
        "compared_len_median": statistics.median(lens) if lens else None,
    }


def _self_det(rows_a: list[dict], rows_b: list[dict]) -> dict[str, Any]:
    """self_det == fraction of prompts whose greedy completion is byte-identical
    across two independent free-runs of the SAME serve (census definition)."""
    n = len(rows_a)
    identical = 0
    tok_total = 0
    tok_match = 0
    for a, b in zip(rows_a, rows_b):
        ca, cb = a["completion_token_ids"], b["completion_token_ids"]
        if ca == cb:
            identical += 1
        L = min(len(ca), len(cb))
        tok_total += max(len(ca), len(cb))
        tok_match += sum(1 for t in range(L) if ca[t] == cb[t])
    return {
        "self_det": identical / n if n else float("nan"),
        "self_det_token_agreement": tok_match / tok_total if tok_total else float("nan"),
        "n_identical_runs": identical, "n_prompts": n,
    }


def _compare_teacherforce(tf_rows: list[dict], *, label: str) -> dict[str, Any]:
    total_valid = 0
    total_match = 0
    total_positions = 0
    per_prompt_flips: list[int] = []
    for r in tf_rows:
        preds = r["tf_argmax_token_ids"]
        ref = r["ref_completion_token_ids"]
        total_positions += len(ref)
        v = m = 0
        for x, y in zip(preds, ref):
            if x is None:
                continue
            v += 1
            if x == y:
                m += 1
        total_valid += v
        total_match += m
        per_prompt_flips.append(v - m)
    flips = total_valid - total_match
    return {
        f"{label}_total_ref_positions": total_positions,
        f"{label}_valid_positions": total_valid,
        f"{label}_matched_positions": total_match,
        f"{label}_flips": flips,
        f"{label}_argmax_flip_rate": (flips / total_valid) if total_valid else float("nan"),
        f"{label}_readout_coverage": (total_valid / total_positions) if total_positions else float("nan"),
        f"{label}_per_prompt_flips": per_prompt_flips,
    }


# ========================================================================== #
# Orchestration
# ========================================================================== #
def _run_worker(server_python: Path, worker_env: dict[str, str], *, mode: str,
                base_url: str, model: str, out_file: Path, num_prompts: int,
                output_len: int, dataset_path: Path, tokenizer: str,
                request_timeout_s: int, ref_file: Path | None) -> list[dict]:
    cmd = [str(server_python), str(Path(__file__).resolve()), "--worker",
           "--mode", mode, "--base-url", base_url, "--model", model,
           "--dataset-path", str(dataset_path), "--tokenizer", tokenizer,
           "--num-prompts", str(num_prompts), "--output-len", str(output_len),
           "--seed", str(args_seed), "--out-file", str(out_file),
           "--request-timeout-s", str(request_timeout_s)]
    if ref_file is not None:
        cmd += ["--ref-file", str(ref_file)]
    subprocess.run(cmd, check=True, timeout=7200, env=worker_env)
    return [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]


def _serve_and_collect(arm: str, args: argparse.Namespace, server_python: Path,
                       worker_env: dict[str, str], paths, ref_file: Path | None):
    from scripts.local_validation import harness
    serve_env = dict(ARM_ENV[arm])
    log_path = OUT_ROOT / "logs" / f"flip_server_{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    free_out = OUT_ROOT / f"tokens_{arm}.jsonl"        # pass A == ref for teacher-force
    free_out_b = OUT_ROOT / f"tokens_{arm}_b.jsonl"    # pass B for self_det
    tf_out = OUT_ROOT / f"tf_{arm}_on_bf16.jsonl"
    result: dict[str, Any] = {}
    with harness.LocalServer(SUBMISSION, server_python=server_python, port=args.port,
                             startup_timeout_s=1800, log_path=log_path,
                             extra_env=serve_env) as srv:
        model = srv.served_model_name
        # token identity is warmth-independent; the worker's own pass warms the serve
        result["freerun_rows"] = _run_worker(
            server_python, worker_env, mode="freerun", base_url=srv.base_url,
            model=model, out_file=free_out, num_prompts=args.num_prompts,
            output_len=args.output_len, dataset_path=paths.EVAL_PROMPTS,
            tokenizer=paths.TOKENIZER, request_timeout_s=args.request_timeout_s,
            ref_file=None)
        # self_det: a second independent free-run on the SAME serve
        result["freerun_rows_b"] = _run_worker(
            server_python, worker_env, mode="freerun", base_url=srv.base_url,
            model=model, out_file=free_out_b, num_prompts=args.num_prompts,
            output_len=args.output_len, dataset_path=paths.EVAL_PROMPTS,
            tokenizer=paths.TOKENIZER, request_timeout_s=args.request_timeout_s,
            ref_file=None)
        # PPL on the canonical ground-truth token stream (comparable to 2.0057/2.3767)
        try:
            result["ppl_summary"] = harness.run_ppl(
                server_python, base_url=srv.base_url, model=model,
                out_file=OUT_ROOT / f"ppl_{arm}.jsonl",
                summary_file=OUT_ROOT / f"ppl_{arm}.summary.json")
        except Exception as exc:  # PPL is secondary; never block the flip rate
            result["ppl_error"] = repr(exc)
            print(f"[flip] PPL on arm {arm} FAILED (non-fatal): {exc!r}", flush=True)
        if ref_file is not None:
            result["tf_rows"] = _run_worker(
                server_python, worker_env, mode="teacherforce", base_url=srv.base_url,
                model=model, out_file=tf_out, num_prompts=args.num_prompts,
                output_len=args.output_len, dataset_path=paths.EVAL_PROMPTS,
                tokenizer=paths.TOKENIZER, request_timeout_s=args.request_timeout_s,
                ref_file=ref_file)
    return result


args_seed = 1  # module-level for _run_worker


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.local_validation import harness, paths
    global args_seed
    args_seed = args.seed
    for note in paths.prepare_local_gpu_env():
        print(f"[flip] {note}", flush=True)
    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])

    import os
    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    # Arm 1: bf16 -> free-run reference R + bf16-on-R self teacher-force check.
    bf16 = _serve_and_collect("bf16", args, server_python, worker_env, paths,
                              ref_file=(OUT_ROOT / "tokens_bf16.jsonl"))
    # Arm 2: int4 -> free-run + int4-on-R teacher-force flip rate.
    int4 = _serve_and_collect("int4", args, server_python, worker_env, paths,
                              ref_file=(OUT_ROOT / "tokens_bf16.jsonl"))

    report: dict[str, Any] = {
        "analysis_only": True, "pr": 553, "stage": "3a",
        "n_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed,
        "arms_env_diff": {"bf16": ARM_ENV["bf16"], "int4": ARM_ENV["int4"]},
    }
    report["freerun"] = _compare_freerun(bf16["freerun_rows"], int4["freerun_rows"])
    if "tf_rows" in bf16:
        report.update(_compare_teacherforce(bf16["tf_rows"], label="tf_bf16_selfcheck"))
    if "tf_rows" in int4:
        report.update(_compare_teacherforce(int4["tf_rows"], label="tf_int4"))

    # self_det (per arm) + canonical PPL (per arm) + int4-vs-bf16 PPL delta
    report["self_det_bf16"] = _self_det(bf16["freerun_rows"], bf16["freerun_rows_b"])
    report["self_det_int4"] = _self_det(int4["freerun_rows"], int4["freerun_rows_b"])
    report["self_det"] = report["self_det_int4"]["self_det"]
    ppl_bf16 = (bf16.get("ppl_summary") or {}).get("ppl")
    ppl_int4 = (int4.get("ppl_summary") or {}).get("ppl")
    report["ppl_bf16_anchor"] = ppl_bf16
    report["ppl_int4_head"] = ppl_int4
    report["ppl"] = ppl_int4
    report["ppl_delta_int4_minus_bf16"] = (
        (ppl_int4 - ppl_bf16) if (isinstance(ppl_int4, (int, float)) and isinstance(ppl_bf16, (int, float))) else None
    )
    report["ppl_summaries"] = {"bf16": bf16.get("ppl_summary"), "int4": int4.get("ppl_summary"),
                               "bf16_error": bf16.get("ppl_error"), "int4_error": int4.get("ppl_error")}

    # headline outputs (PR #553 Stage 3a)
    fr = report["freerun"]
    report["precision_head_strict_identity"] = fr["strict_identity"]
    tf_rate = report.get("tf_int4_argmax_flip_rate")
    sc = report.get("tf_bf16_selfcheck_argmax_flip_rate")
    report["tf_selfcheck_clean"] = (sc is not None and sc <= 1e-9)
    # prefer the precise teacher-forced rate when the self-check validates the readout
    if tf_rate is not None and report["tf_selfcheck_clean"]:
        report["precision_head_argmax_flip_rate"] = tf_rate
        report["precision_head_argmax_flip_rate_source"] = "teacher_forced"
    else:
        report["precision_head_argmax_flip_rate"] = fr["censored_hazard_flip_rate"]
        report["precision_head_argmax_flip_rate_source"] = "freerun_censored_hazard"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-prompts", "--n-prompts", dest="num_prompts", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--out-json", type=Path, default=OUT_ROOT / "flip_rate.json")
    # worker (internal)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=["freerun", "teacherforce"], default="freerun")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out-file")
    ap.add_argument("--ref-file", default=None)
    args = ap.parse_args(argv)

    if args.worker:
        return _worker(args)

    report = run(args)
    args.out_json.write_text(json.dumps(report, indent=2))
    fr = report["freerun"]
    print("\n" + "=" * 16 + " INT4-HEAD vs BF16-HEAD GREEDY FLIP RATE (PR #553 St3a) " + "=" * 16, flush=True)
    print(f"  prompts={report['n_prompts']} output_len={report['output_len']}", flush=True)
    print(f"  strict_identity = {report['precision_head_strict_identity']}  "
          f"(identical {fr['n_identical_prompts']}/{fr['n_prompts']} prompts)", flush=True)
    print(f"  argmax_flip_rate = {report['precision_head_argmax_flip_rate']:.4e}  "
          f"(source: {report['precision_head_argmax_flip_rate_source']})", flush=True)
    if "tf_int4_argmax_flip_rate" in report:
        print(f"    teacher-forced int4 flip rate = {report['tf_int4_argmax_flip_rate']:.4e} "
              f"({report['tf_int4_flips']}/{report['tf_int4_valid_positions']} positions); "
              f"self-check bf16 rate = {report.get('tf_bf16_selfcheck_argmax_flip_rate'):.2e} "
              f"(clean={report['tf_selfcheck_clean']})", flush=True)
    print(f"    free-run first-divergence median = {fr['first_divergence_median']}  "
          f"censored hazard = {fr['censored_hazard_flip_rate']:.4e}", flush=True)
    sd_i = report["self_det_int4"]; sd_b = report["self_det_bf16"]
    print(f"  self_det int4 = {sd_i['self_det']:.4f} (tok {sd_i['self_det_token_agreement']:.5f})  "
          f"bf16 = {sd_b['self_det']:.4f} (tok {sd_b['self_det_token_agreement']:.5f})", flush=True)
    if report.get("ppl") is not None and report.get("ppl_bf16_anchor") is not None:
        print(f"  PPL int4 = {report['ppl_int4_head']:.4f}  bf16 anchor = {report['ppl_bf16_anchor']:.4f}  "
              f"delta = {report['ppl_delta_int4_minus_bf16']:+.4f}", flush=True)
    print("=" * 90 + "\n", flush=True)
    print(f"[flip] report -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
