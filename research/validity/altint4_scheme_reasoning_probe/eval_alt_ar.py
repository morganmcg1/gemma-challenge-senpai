#!/usr/bin/env python
"""PR #647 — M=1 AR greedy eval of the ALT int4-g128 (GPTQ) body on the two failing
bars (GPQA-D n=198, AIME n=60).

Reuses the #637 materiality/eval harness verbatim for tokenization + scoring
(``optionb_319_answer_materiality/evalsets.py``) and the LocalServer machinery, so the
ALT cells are byte-for-byte comparable to the QAT-int4 AR cells already banked in #637
(``ar_gpqa.jsonl`` / ``ar_aime.jsonl``). The ONLY change vs the #637 AR arm is the body
checkpoint: this serves the GPTQ alt body instead of the QAT body, drafter OFF.

Config (matches #637 exactly, per eval):
  * GPQA-D: max_model_len 6144, max_tokens 6144 (clamped) -> pools with #637 banked QAT 198.
  * AIME:   max_model_len 8192, max_tokens 6144           -> matches #637 banked QAT 60.
  * greedy T=0, min_tokens 8, VLLM_BATCH_INVARIANT=1, MAX_NUM_SEQS=1 (clean M=1 AR),
    NUM_SPECULATIVE_TOKENS=0 (no drafter), vLLM 0.22.0.

ANALYSIS-ONLY. Local A10G. NO HF Job, NO submission. analysis_only=True, official_tps=0.
Idempotent: skips any (eval) item already on disk so a cut run resumes cleanly.

Usage:
  eval_alt_ar.py --body /workspace/gemma_build/altint4_gptq_g128 --evals gpqa --max-model-len 6144
  eval_alt_ar.py --body /workspace/gemma_build/altint4_gptq_g128 --evals aime --max-model-len 8192
"""
from __future__ import annotations

import argparse
import json
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

# reuse the #637 eval-set construction + scoring verbatim (same instruments, same scoring)
OPTIONB = ROOT / "research" / "validity" / "optionb_319_answer_materiality"
sys.path.insert(0, str(OPTIONB))
import evalsets  # noqa: E402

RES = HERE / "results"
# #637 banked QAT AR cells — the SAME prompts the QAT arm scored. We load eval items
# straight from these (not a fresh inspect_evals rebuild) so the ALT prompts are
# byte-identical to QAT (the strongest possible scheme-delta pairing) and the driver
# needs no inspect_evals/inspect_ai. Each record carries prompt_token_ids + scoring
# fields (target/n_choices for gpqa, gold/year for aime). The integrity of each loaded
# prompt is re-checked against its banked prompt_sha256.
QAT_BANKED = OPTIONB / "results"
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"
SERVER_PY = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")
PORT = 8000

MIN_TOKENS = 8
CONTEXT_MARGIN = 8
MAX_TOKENS = 6144
REQUEST_TIMEOUT_S = 900
_DO = evalsets._DO

MAX_MODEL_LEN = 6144  # overridden by --max-model-len


def _spec_env(body: str) -> dict[str, str]:
    """int4_mtp_batchinv with the ALT body, drafter OFF, clean deterministic M=1 AR."""
    return {
        "MODEL_ID": body,
        "NUM_SPECULATIVE_TOKENS": "0",        # no drafter — isolate the body
        "VLLM_BATCH_INVARIANT": "1",
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "MAX_NUM_SEQS": "1",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }


def _gpu_used_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        peak["mib"] = max(peak["mib"], _gpu_used_mib())
        stop.wait(2.0)


def request_greedy(base_url: str, model: str, prompt_ids: list[int], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
        "min_tokens": MIN_TOKENS, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "seed": 0, "stream": False, "add_special_tokens": False, "ignore_eos": False,
        "return_token_ids": True,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:300]}") from exc


def load_items_from_banked(kind: str) -> list[dict]:
    """Eval items loaded DIRECTLY from the #637 banked QAT ``ar_{kind}.jsonl``.

    This is the scheme-delta pairing anchor: by serving the EXACT prompt_token_ids the
    QAT arm scored, every ALT cell is paired to its QAT cell on a byte-identical prompt
    (verified against the banked prompt_sha256). It also removes the inspect_evals/
    inspect_ai rebuild dependency from the driver. Scoring still uses the #637
    ``evalsets.score_item`` verbatim, so ALT and QAT are scored identically."""
    path = QAT_BANKED / f"ar_{kind}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"banked QAT cell not found: {path}")
    items: list[dict] = []
    n_bad = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids = r["prompt_token_ids"]
        if evalsets.sha256_tokens(ids) != r.get("prompt_sha256"):
            n_bad += 1
            continue
        it = {"id": str(r["id"]), "kind": kind,
              "prompt_token_ids": ids, "prompt_sha256": r["prompt_sha256"]}
        if kind == "gpqa":
            it["target"] = r["target"]; it["n_choices"] = r["n_choices"]
            it["base_qid"] = r.get("base_qid", r["id"])
            it["shuffle_seed"] = r.get("shuffle_seed")
        elif kind == "aime":
            it["gold"] = r.get("gold"); it["year"] = r.get("year")
        items.append(it)
    if n_bad:
        raise RuntimeError(f"{kind}: {n_bad} banked prompts failed sha256 integrity check")
    return items


def _arm_path(kind: str) -> Path:
    return RES / f"alt_ar_{kind}.jsonl"


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


def gen(kind: str, items: list[dict], srv, max_tokens: int) -> None:
    out_path = _arm_path(kind)
    done = _load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    print(f"[eval] {kind}: {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    t0 = time.time()
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for it in todo:
            eff_max = max(MIN_TOKENS,
                          min(max_tokens, MAX_MODEL_LEN - len(it["prompt_token_ids"]) - CONTEXT_MARGIN))
            rec = {"id": it["id"], "kind": kind,
                   "prompt_token_ids": it["prompt_token_ids"],
                   "prompt_sha256": it["prompt_sha256"], "max_tokens_eff": eff_max}
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
                    "error": None, **scored,
                })
                if kind == "gpqa":
                    rec["target"] = it["target"]; rec["n_choices"] = it["n_choices"]
                    rec["base_qid"] = it.get("base_qid", it["id"])
                    rec["shuffle_seed"] = it.get("shuffle_seed")
                elif kind == "aime":
                    rec["gold"] = it.get("gold"); rec["year"] = it.get("year")
            except Exception as exc:  # noqa: BLE001
                rec.update({
                    "completion_token_ids": [], "completion_token_sha256": None,
                    "completion_text": "", "num_completion_tokens": 0,
                    "finish_reason": "error", "answer": None, "correct": False,
                    "extract_mode": "error", "error": repr(exc)[:300]})
                print(f"[eval] {kind} id={it['id']} ERROR: {repr(exc)[:160]}", flush=True)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 8 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"[eval] {kind} {n_done}/{len(todo)} ({el:.0f}s, {el/max(n_done,1):.1f}s/item)",
                      flush=True)


def main() -> int:
    global MAX_MODEL_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, help="path to the ALT int4 checkpoint")
    ap.add_argument("--evals", default="gpqa,aime")
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--aime-years", default=",".join(evalsets.AIME_YEARS_DEFAULT))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap items per eval")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    MAX_MODEL_LEN = int(args.max_model_len)
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    print(f"[eval] body={args.body} max_model_len={MAX_MODEL_LEN} evals={evals}", flush=True)
    eval_items: dict[str, list[dict]] = {}
    for k in evals:
        its = load_items_from_banked(k)
        if args.limit and args.limit > 0:
            its = its[: args.limit]
        eval_items[k] = its
    for k, its in eval_items.items():
        ptoks = [len(it["prompt_token_ids"]) for it in its]
        print(f"[eval] {k}: {len(its)} items (banked QAT prompts, sha256-verified), "
              f"prompt_tokens {min(ptoks)}-{max(ptoks)}", flush=True)

    extra_env = _spec_env(args.body)
    log_path = RES / f"_serve_alt_ar_mml{MAX_MODEL_LEN}.log"
    print(f"[eval] === ALT-AR === {time.strftime('%H:%M:%S')} env={extra_env}", flush=True)
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=SERVER_PY, port=PORT, log_path=log_path,
            extra_env=extra_env, startup_timeout_s=1800,
        ) as srv:
            print(f"[eval] ready at {srv.base_url} model={srv.served_model_name}", flush=True)
            for kind in evals:
                gen(kind, eval_items[kind], srv, MAX_TOKENS)
    finally:
        stop.set()
        sampler.join(timeout=5)
    gb = (peak["mib"] or 0.0) / 1024.0
    print(f"[eval] === ALT-AR DONE === peak {gb:.1f} GB {time.strftime('%H:%M:%S')}", flush=True)

    meta_path = RES / "eval_meta.json"
    prev = {}
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
        except (ValueError, OSError):
            prev = {}
    per_eval = dict(prev.get("per_eval_config", {}))
    for k in evals:
        per_eval[k] = {"max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS,
                       **({"aime_years": args.aime_years} if k == "aime" else {})}
    peaks = dict(prev.get("peaks", {})); peaks[f"mml{MAX_MODEL_LEN}"] = gb
    meta_path.write_text(json.dumps({
        "body": args.body, "arms": ["ar"], "drafter": "OFF",
        "evals": sorted(set(prev.get("evals", [])) | set(evals)),
        "per_eval_config": per_eval, "min_tokens": MIN_TOKENS, "max_num_seqs": 1,
        "batch_invariant": 1, "peaks": peaks,
        "analysis_only": True, "official_tps": 0,
    }, indent=2))
    print(f"[eval] meta written {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
