#!/usr/bin/env python
"""PR #646 — int4->int8->bf16 reasoning-ladder greedy quality driver (AR M=1, no drafter).

Measures the MIDDLE rung the panel is missing: the uniform-int8 (W8A16) body on the
two reasoning bars (GPQA-Diamond n=198, AIME n=60), greedy maj@1, at the EXACT panel
config so it is apples-to-apples with the banked int4 and the cited bf16 endpoints:

  VLLM_BATCH_INVARIANT=1, MAX_NUM_SEQS=1 serial, gb6144 (--max-model-len 8192,
  max_tokens 6144), min_tokens=8, temperature 0 greedy, vLLM 0.22.0, M=1 AR (no spec).

Apples-to-apples is enforced two ways:
  * Prompts are the BYTE-IDENTICAL banked prompt_token_ids from denken #637's
    materiality AR arm (research/validity/optionb_319_answer_materiality/results/
    ar_{gpqa,aime}.jsonl). Same prompt_sha256 -> same instrument as the int4/bf16 cells.
  * Scoring reuses evalsets.score_item verbatim (the same code that scored the banked
    int4 cells), so int4-vs-int8 is a like-for-like accuracy comparison.

Body served via submissions/int4_base_aime (a body-agnostic plain vLLM AR server,
--dtype auto, NO speculation, NO surgical patch); MODEL_ID selects int8 vs int4.

Idempotent: per-item jsonl flush + skip-done resume, so a run killed by the 90-min
bound resumes cleanly on relaunch. A soft wall-clock cap exits gracefully (so
LocalServer cleans up the vLLM child instead of leaking it).

ANALYSIS-ONLY. Local A10G. NO HF Job, NO submission, NO served-file change.
analysis_only=True, official_tps=0. W&B group int8-bf16-reasoning-ladder-fern.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

MAT = ROOT / "research" / "validity" / "optionb_319_answer_materiality"
sys.path.insert(0, str(MAT))
import evalsets  # noqa: E402

_DO = evalsets._DO

RES = HERE / "results"
BANK = MAT / "results"
# Serve from a stable copy OUTSIDE the tracked git tree. The entrypoint poll loop
# git-checkouts the advisor branch and back every ~10 min, which unlinks any
# tracked submissions/** dir held as a live server CWD -> os.getcwd() crash mid-run.
SUBMISSION = Path("/workspace/gemma_build/sub_int4_base_aime")
if not SUBMISSION.exists():
    SUBMISSION = ROOT / "submissions" / "int4_base_aime"
SERVER_PY = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")

BODIES = {
    "int8": "/workspace/gemma_build/int8_g128_lmhead",
    "int4": "/workspace/gemma_build/int4_g128_lmhead",
}

PORT = 8000
MAX_MODEL_LEN = 8192   # gb6144 = (--max-model-len 8192, max_tokens 6144); matches bf16 endpoint
MAX_TOKENS = 6144
MIN_TOKENS = 8         # #541 first-token-EOS guard
CONTEXT_MARGIN = 8
REQUEST_TIMEOUT_S = 1200
SOFT_CAP_MIN_DEFAULT = 82.0   # exit gracefully before the 90-min hard bound (resumable)

# Cited / banked ladder endpoints (greedy maj@1). bf16 = ubel #628 at conc16; int4 =
# denken #637 banked AR at conc1 (== my int8/int4 cells' concurrency). 90% bars below.
ENDPOINTS = {
    "gpqa": {"bf16": 0.4899, "bf16_run": "g3cig1xo", "int4": 0.4798, "bar90": 0.4409, "n": 198},
    "aime": {"bf16": 0.4667, "bf16_run": "zoszxnb0", "int4": 0.4000, "bar90": 0.4200, "n": 60},
}


def _env(body_path: str) -> dict[str, str]:
    """int4_base_aime extra_env: plain AR (no drafter), BI=1, conc1, gb6144."""
    return {
        "MODEL_ID": body_path,
        "SERVED_MODEL_NAME": "gemma-4-e4b-it",
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "MAX_NUM_SEQS": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # PyTorch-native lowest-index argmax tie-break
        "VLLM_SEED": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
    }


# --------------------------------------------------------------------------- items
def load_bank_items(kind: str, limit: int = 0) -> list[dict[str, Any]]:
    """Reuse the byte-identical banked prompt_token_ids (same prompt_sha256 as the
    int4/bf16 cells). NO tokenizer / dataset rebuild -> no inspect_evals dep, and a
    guaranteed-matched instrument."""
    path = BANK / f"ar_{kind}.jsonl"
    items: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        it: dict[str, Any] = {
            "id": r["id"], "kind": kind,
            "prompt_token_ids": r["prompt_token_ids"],
            "prompt_sha256": r["prompt_sha256"],
        }
        if kind == "gpqa":
            it["target"] = r["target"]
            it["n_choices"] = r["n_choices"]
        else:
            it["gold"] = r.get("gold")
            it["year"] = r.get("year")
        items.append(it)
    if limit and limit > 0:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------- request
def request_greedy(base_url: str, model: str, prompt_ids: list[int], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
        "min_tokens": MIN_TOKENS, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "seed": 0, "stream": False, "add_special_tokens": False,
        "ignore_eos": False, "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:300]}") from exc


# --------------------------------------------------------------------------- VRAM
def _gpu_used_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        peak["mib"] = max(peak["mib"], _gpu_used_mib())
        stop.wait(2.0)


# --------------------------------------------------------------------------- gen
def _arm_path(body: str, kind: str) -> Path:
    return RES / f"{body}_{kind}.jsonl"


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                done.add(str(json.loads(line)["id"]))
            except (ValueError, KeyError):
                continue
    return done


def gen_cell(body: str, kind: str, items: list[dict], srv, soft_deadline: float) -> bool:
    """Generate+score a (body,kind) cell. Returns True if fully complete, False if the
    soft wall-clock cap stopped it early (resume next launch)."""
    out_path = _arm_path(body, kind)
    done = _load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    print(f"[gen] {body}/{kind}: {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return True
    t0 = time.time()
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for it in todo:
            if time.time() >= soft_deadline:
                print(f"[gen] {body}/{kind} SOFT-CAP hit after {n_done} items — "
                      f"exiting for resume ({len(todo)-n_done} left)", flush=True)
                return False
            eff_max = max(MIN_TOKENS,
                          min(MAX_TOKENS, MAX_MODEL_LEN - len(it["prompt_token_ids"]) - CONTEXT_MARGIN))
            rec: dict[str, Any] = {
                "id": it["id"], "kind": kind,
                "prompt_sha256": it["prompt_sha256"], "max_tokens_eff": eff_max,
            }
            try:
                resp = request_greedy(srv.base_url, srv.served_model_name,
                                      it["prompt_token_ids"], eff_max)
                choice = _DO.choice_from_response(resp)
                comp_ids, _src, src_kind = _DO.extract_generated_token_ids(
                    resp, choice, it["prompt_token_ids"])
                text = _DO.generated_text_from_choice(choice)
                finish = choice.get("finish_reason")
                scored = evalsets.score_item(it, text)
                rec.update({
                    "completion_token_ids": comp_ids,
                    "completion_token_sha256": evalsets.sha256_tokens(comp_ids),
                    "completion_text": text,
                    "num_completion_tokens": len(comp_ids),
                    "finish_reason": finish,
                    "token_id_source_kind": src_kind,
                    "error": None,
                    **scored,
                })
                if kind == "gpqa":
                    rec["target"] = it["target"]; rec["n_choices"] = it["n_choices"]
                else:
                    rec["gold"] = it.get("gold"); rec["year"] = it.get("year")
            except Exception as exc:  # noqa: BLE001
                rec.update({
                    "completion_token_ids": [], "completion_token_sha256": None,
                    "completion_text": "", "num_completion_tokens": 0,
                    "finish_reason": "error", "answer": None, "correct": False,
                    "extract_mode": "error", "error": repr(exc)[:300],
                })
                print(f"[gen] {body}/{kind} id={it['id']} ERROR: {repr(exc)[:160]}", flush=True)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 8 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"[gen] {body}/{kind} {n_done}/{len(todo)} "
                      f"({el:.0f}s, {el/max(n_done,1):.1f}s/item)", flush=True)
    return True


# --------------------------------------------------------------------------- stats
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_cell(body: str, kind: str) -> dict[str, Any]:
    path = _arm_path(body, kind)
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(recs)
    err = sum(1 for r in recs if r.get("error"))
    n_eff = n - err
    correct = sum(1 for r in recs if r.get("correct"))
    trunc = sum(1 for r in recs if r.get("finish_reason") == "length")
    extract_fail = sum(1 for r in recs if r.get("answer") is None and not r.get("error"))
    toks = [r.get("num_completion_tokens", 0) for r in recs if not r.get("error")]
    acc = correct / n_eff if n_eff else 0.0
    lo, hi = wilson_ci(correct, n_eff)
    ep = ENDPOINTS.get(kind, {})
    bf16 = ep.get("bf16")
    pct_bf16 = (acc / bf16) if bf16 else None
    pct_lo = (lo / bf16) if bf16 else None
    pct_hi = (hi / bf16) if bf16 else None
    summ = {
        "body": body, "kind": kind, "n": n, "n_eff": n_eff, "errors": err,
        "correct": correct, "acc": acc, "ci_lo": lo, "ci_hi": hi,
        "truncation_rate": trunc / n if n else 0.0, "n_truncated": trunc,
        "extract_fail": extract_fail,
        "mean_completion_tokens": (sum(toks) / len(toks)) if toks else 0.0,
        "max_completion_tokens": max(toks) if toks else 0,
        "bf16_endpoint": bf16, "bf16_run": ep.get("bf16_run"),
        "int4_endpoint": ep.get("int4"), "bar90": ep.get("bar90"),
        "pct_of_bf16": pct_bf16, "pct_of_bf16_ci_lo": pct_lo, "pct_of_bf16_ci_hi": pct_hi,
        "clears_90pct_bar": (acc >= ep["bar90"]) if ep.get("bar90") else None,
        "max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS, "min_tokens": MIN_TOKENS,
        "max_num_seqs": 1, "batch_invariant": 1, "decode": "greedy_t0",
        "analysis_only": True, "official_tps": 0,
    }
    return summ


def log_wandb(summ: dict[str, Any], peak_vram_gb: float, group: str) -> str | None:
    try:
        import wandb
    except ImportError:
        print("[wandb] not available — skipping", flush=True)
        return None
    body, kind = summ["body"], summ["kind"]
    run = wandb.init(
        project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
        group=group, name=f"fern/{body}-{kind}-greedy-ladder",
        config={
            "pr": 646, "body": body, "eval": kind, "decode": "greedy_t0",
            "quant": ("int8_w8a16_g128_lmhead" if body == "int8" else "int4_w4a16_g128_lmhead"),
            "source_base": ("google/gemma-4-E4B-it@fee6332c (plain bf16)" if body == "int8"
                            else "qat-unquantized"),
            "max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS, "min_tokens": MIN_TOKENS,
            "max_num_seqs": 1, "batch_invariant": 1, "vllm": "0.22.0", "spec": "off_AR_M1",
            "analysis_only": True, "official_tps": 0,
            "wandb_group": group,
        },
        reinit=True,
    )
    log = {f"ladder/{k}": v for k, v in summ.items() if isinstance(v, (int, float, bool)) or v is None}
    log["ladder/peak_vram_gb"] = peak_vram_gb
    wandb.log(log)
    for k, v in summ.items():
        if isinstance(v, (int, float, bool)):
            run.summary[k] = v
    run.summary["peak_vram_gb"] = peak_vram_gb
    rid = run.id
    wandb.finish()
    print(f"[wandb] logged {body}/{kind} -> run {rid}", flush=True)
    return rid


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=list(BODIES))
    ap.add_argument("--evals", default="gpqa,aime")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--limit", type=int, default=0, help="cap items/eval (smoke)")
    ap.add_argument("--soft-cap-min", type=float, default=SOFT_CAP_MIN_DEFAULT)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-group", default="int8-bf16-reasoning-ladder-fern")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    limit = args.limit or (4 if args.mode == "smoke" else 0)
    body_path = BODIES[args.body]
    if not Path(body_path).exists():
        print(f"[fatal] body path missing: {body_path}", flush=True)
        return 2

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    eval_items = {k: load_bank_items(k, limit=limit) for k in evals}
    for k, its in eval_items.items():
        print(f"[items] {k}: {len(its)} banked items "
              f"(prompt_tokens {min(len(i['prompt_token_ids']) for i in its)}-"
              f"{max(len(i['prompt_token_ids']) for i in its)})", flush=True)

    soft_deadline = time.time() + args.soft_cap_min * 60.0
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    log_path = RES / f"_serve_{args.body}.log"
    complete: dict[str, bool] = {}
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=SERVER_PY, port=PORT,
            log_path=log_path, extra_env=_env(body_path), startup_timeout_s=1800,
        ) as srv:
            print(f"[serve] {args.body} ready at {srv.base_url} model={srv.served_model_name}", flush=True)
            for kind in evals:
                complete[kind] = gen_cell(args.body, kind, eval_items[kind], srv, soft_deadline)
    finally:
        stop.set()
        sampler.join(timeout=5)
    peak_gb = (peak["mib"] or 0.0) / 1024.0
    print(f"[serve] {args.body} peak {peak_gb:.1f} GB", flush=True)

    summaries = {}
    for kind in evals:
        summ_path = RES / f"summary_{args.body}_{kind}.json"
        # Carry a wandb_run_id forward across resume windows so a cell that finished in
        # an earlier window is NOT re-logged as a duplicate wandb run this window.
        prior_rid = None
        if summ_path.exists():
            try:
                prior_rid = json.loads(summ_path.read_text()).get("wandb_run_id")
            except (ValueError, OSError):
                prior_rid = None
        summ = summarize_cell(args.body, kind)
        if prior_rid:
            summ["wandb_run_id"] = prior_rid
        summaries[kind] = summ
        print(f"[summary] {args.body}/{kind}: acc={summ['acc']:.4f} "
              f"CI[{summ['ci_lo']:.4f},{summ['ci_hi']:.4f}] n_eff={summ['n_eff']} "
              f"trunc={summ['truncation_rate']:.1%} extract_fail={summ['extract_fail']} "
              f"pct_bf16={summ['pct_of_bf16']:.3f} clears90={summ['clears_90pct_bar']} "
              f"complete={complete.get(kind)}", flush=True)
        summ_path.write_text(json.dumps(summ, indent=2))
        if not args.no_wandb and complete.get(kind) and not prior_rid:
            try:
                rid = log_wandb(summ, peak_gb, args.wandb_group)
                if rid:
                    summ["wandb_run_id"] = rid
                    summ_path.write_text(json.dumps(summ, indent=2))
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb] log failed: {repr(exc)[:200]}", flush=True)

    all_done = all(complete.get(k) for k in evals)
    print(f"[done] {args.body} evals={evals} all_complete={all_done} "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    return 0 if all_done else 3


if __name__ == "__main__":
    raise SystemExit(main())
