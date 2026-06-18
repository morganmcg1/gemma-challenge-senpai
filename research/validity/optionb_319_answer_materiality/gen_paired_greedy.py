#!/usr/bin/env python
"""PR #626 — GREEDY matched-arm generation driver (spec vs AR, same int4 body).

Generates the paired per-item evidence for the answer-materiality question: do the
~0.43% residual greedy token-flips (#616 int4-Marlin grid-ties) ever change a final
extracted EVAL ANSWER? Two arms on the SAME int4 body substrate
(/workspace/gemma_build/int4_g128_lmhead), served via the SAME submission
(int4_mtp_batchinv, VLLM_BATCH_INVARIANT=1, MAX_MODEL_LEN=6144), toggling ONLY spec:

  * spec arm: NUM_SPECULATIVE_TOKENS=7  (Option-B candidate, M=8 verify; drafter /tmp/qat-assistant)
  * ar   arm: NUM_SPECULATIVE_TOKENS=0  (plain int4 M=1 AR; drafter OFF)

GREEDY (T=0), so each arm is a SINGLE deterministic generation (no seed averaging).

  *** MAX_NUM_SEQS=1, serial requests (concurrency=1). ***
  Continuous batching at conns>1 runs the decode forward at M=(#co-batched seqs),
  and VLLM_BATCH_INVARIANT=1 does NOT make the int4-Marlin GEMM M-invariant (that
  IS the #616 residual break). conns=1 pins the AR arm to a clean M=1 decode and the
  spec arm to a clean M=8 verify (#616's exact arms, free-running) and makes both
  deterministic — the only way to get a faithful paired greedy #319 read.

Prompts are pre-tokenized to integer ids (evalsets.encode_chat_prompt) and the SAME
ids are sent to both arms via /v1/completions (return_token_ids:true), the canonical
official greedy-identity harness path (decode_outputs.py). Records are written in the
official decode_outputs schema (id + completion_token_ids + completion_token_sha256)
so analyze_materiality can feed them straight to the greedy_identity verifier, PLUS
the per-item extracted answer / correctness for the answer-level read.

After the AR arm finishes (while its M=1 server is still up), runs the logit-gap probe
at each item's FIRST cross-arm divergence position, to confirm the flips are <0.5-nat
int4 ties and flag any large-margin answer-flipping divergence. The probe is a SINGLE
greedy decode step (max_tokens=1, logprobs=K) from the matched prefix on the AR M=1
server, NOT prompt_logprobs over the whole prefix: prompt_logprobs materialises a
[len(prefix), vocab] float32 log_softmax for every prefix position and OOMs / kills the
EngineCore on long CoT prefixes near the VRAM ceiling (observed engine-death on GSM8K
~1.4k-token prefixes; GPQA at gb6144 would need ~6 GiB). The 1-position decode is O(1)
in prefix length, and its decode-branch M=1 distribution is the exact distribution that
produced the AR free-run argmax at that step — strictly more faithful than the prefill
re-score for "was this free-run flip a near-tie".

ANALYSIS-ONLY. Local A10G. NO HF Job, NO submission. analysis_only=True, official_tps=0.
Idempotent: skips any (arm,eval) item already on disk so a crashed run resumes cleanly.

Usage:
  gen_paired_greedy.py --arms spec,ar --evals gpqa,gsm8k --mode full
  gen_paired_greedy.py --arms spec,ar --mode smoke     # 4 GPQA + 16 GSM8K
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

sys.path.insert(0, str(HERE))
import evalsets  # noqa: E402

RES = HERE / "results"
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"

# vLLM 0.22.0 submission stack — the #319-faithful reference (lawine #606: dev307 is
# NOT a faithful proxy; PR #626 says use 0.22.0 if picking one).
SERVER_PY = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")

BODY = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER = "/tmp/qat-assistant"
PORT = 8000
MODEL = "gemma-4-e4b-it"

# MAX_MODEL_LEN is overridable via --max-model-len (main() sets the module global).
# Leg-2 GPQA stays at 6144 to pool byte-for-byte with the #626-banked 198; Leg-1 AIME
# uses the PR #637 literal gb6144 = (--max-model-len 8192, max_tokens 6144) so long math
# CoT gets the full 6144 generation budget regardless of prompt length.
MAX_MODEL_LEN = 6144
# gb6144 (#612 clean budget): fill the context with generation room. The OUTPUT budget is
# clamped per item to MAX_MODEL_LEN - prompt_len - CONTEXT_MARGIN so a long CoT uses the
# whole remaining context (fern #612: 0% GPQA truncation), without vLLM's
# prompt+max_tokens>max_model_len 400. A flat max_tokens=MAX_MODEL_LEN leaves no prompt
# room and 400s every item — the clamp is what makes "gb6144" actually mean gb6144.
GPQA_MAX_TOKENS = 6144   # intent: fill context (clamped below); long-CoT room
GSM8K_MAX_TOKENS = 512   # faithful GSM8K gate protocol (#533); answers are short, fits
AIME_MAX_TOKENS = 6144   # PR #637 Leg 1: full gen budget for long math CoT (clamped below)
MIN_TOKENS = 8           # #541 first-token-EOS guard
CONTEXT_MARGIN = 8       # headroom so prompt+max_tokens stays < MAX_MODEL_LEN
N_LOGPROBS = 20          # decode-step top-k for the gap probe (#616 near-tie character)
REQUEST_TIMEOUT_S = 900

# import the official decode helpers (response parsing + token-id extraction)
_DO = evalsets._DO


def _spec_env(arm: str) -> dict[str, str]:
    """extra_env for int4_mtp_batchinv. Toggles ONLY speculation. MAX_NUM_SEQS=1 ->
    clean M=1 AR / M=8 verify, deterministic."""
    num_spec = 7 if arm == "spec" else 0
    return {
        "MODEL_ID": BODY,
        "DRAFTER_MODEL": DRAFTER,
        "NUM_SPECULATIVE_TOKENS": str(num_spec),
        "VLLM_BATCH_INVARIANT": "1",
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "MAX_NUM_SEQS": "1",            # serial -> clean per-arm M, deterministic greedy
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "2048",  # prefill chunk; identical across arms -> cancels
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # PyTorch-native lowest-index argmax tie-break
    }


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


# --------------------------------------------------------------------------- requests
def request_greedy(base_url: str, model: str, prompt_ids: list[int], max_tokens: int,
                   timeout_s: int = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "min_tokens": MIN_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "stream": False,
        "add_special_tokens": False,  # chat template already added specials
        "ignore_eos": False,          # real eval: stop at EOS (NOT decode_outputs' ignore_eos)
        "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:300]}") from exc


def request_decode_logprobs(base_url: str, model: str, prefix_ids: list[int],
                            n_logprobs: int = N_LOGPROBS,
                            timeout_s: int = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """ONE greedy decode step (M=1 decode on the AR server) from ``prefix_ids``, returning
    the top-``n_logprobs`` completion logprobs at that single position. O(1) in the prefix
    length (one position's log_softmax) — unlike prompt_logprobs, which materialises a
    [len(prefix), vocab] log_softmax and OOMs on long CoT prefixes. Greedy (temperature=0,
    no EOS masking) so the chosen token == the AR argmax at this step, and top_logprobs
    carries the competing spec token's logprob for the near-tie/gap read."""
    payload = {
        "model": model, "prompt": prefix_ids, "max_tokens": 1, "min_tokens": 0,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0, "stream": False,
        "add_special_tokens": False, "logprobs": n_logprobs, "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:300]}") from exc


def _detok_variants(tok, tid: int) -> list[str]:
    """Display-string variants of a single token id, to match vLLM's string-keyed
    top_logprobs (decoded-token keys). Tries the full decode and the raw sentencepiece
    piece (with the ▁ word-boundary marker translated to a space)."""
    out: list[str] = []
    try:
        out.append(tok.decode([tid]))
    except Exception:  # noqa: BLE001
        pass
    try:
        piece = tok.convert_ids_to_tokens(tid)
        if isinstance(piece, str):
            out.append(piece)
            out.append(piece.replace("▁", " "))
    except Exception:  # noqa: BLE001
        pass
    seen: set[str] = set()
    res: list[str] = []
    for s in out:
        if isinstance(s, str) and s not in seen:
            seen.add(s)
            res.append(s)
    return res


def _parse_decode_logprobs(resp: dict[str, Any], ar_tok: int, spec_tok: int,
                           tok) -> dict[str, Any] | None:
    """From a 1-token decode-with-logprobs response, extract the gap fields for ar_tok vs
    spec_tok at this position. Integer id for the chosen token via logprobs.token_ids;
    competing tokens matched in the (string-keyed) top_logprobs by single-token detok.
    Stores the raw top-k {str: lp} so any live-match miss is recoverable post-hoc."""
    choice = (resp.get("choices") or [{}])[0]
    lp = choice.get("logprobs") or {}
    top = lp.get("top_logprobs")
    if not isinstance(top, list) or not top or not isinstance(top[0], dict):
        return None
    top0 = {str(k): float(v) for k, v in top[0].items() if isinstance(v, (int, float))}
    if not top0:
        return None
    token_logprobs = lp.get("token_logprobs") or []
    chosen_ids = lp.get("token_ids") or choice.get("token_ids") or []
    chosen_id = int(chosen_ids[0]) if chosen_ids else None
    lp_chosen = float(token_logprobs[0]) if token_logprobs else None

    def _lp_for(tid: int) -> float | None:
        # chosen-token fast path: its logprob is authoritative regardless of string match.
        if chosen_id is not None and tid == chosen_id and lp_chosen is not None:
            return lp_chosen
        for cand in _detok_variants(tok, tid):
            if cand in top0:
                return top0[cand]
        return None

    lp_ar = _lp_for(ar_tok)
    lp_spec = _lp_for(spec_tok)
    floor = min(top0.values())
    amax_str = max(top0, key=lambda k: top0[k])
    return {
        "probe_chosen_id": chosen_id,
        "probe_argmax_is_ar_tok": (chosen_id == ar_tok) if chosen_id is not None else None,
        "lp_ar_tok": lp_ar,
        "lp_spec_tok": lp_spec,
        "ar_tok_in_topk": lp_ar is not None,
        "spec_tok_in_topk": lp_spec is not None,
        # gap between the two competing tokens under the AR decode dist; spec outside top-k
        # => large-margin (lower-bounded by the top-k floor). ar_tok is the decode argmax,
        # so gap >= 0 in the common case (#616 semantics: relaxed acceptor preserves spec
        # iff gap <= tau).
        "gap_ar_minus_spec": (
            (lp_ar - lp_spec) if (lp_ar is not None and lp_spec is not None)
            else ((lp_ar - floor) if lp_ar is not None else None)
        ),
        "spec_outside_topk": lp_spec is None,
        "topk_floor_lp": floor,
        "n_topk": len(top0),
        "probe_argmax_str": amax_str,
        "top_logprobs_raw": top0,
    }


# --------------------------------------------------------------------------- IO
def _arm_path(arm: str, kind: str) -> Path:
    return RES / f"{arm}_{kind}.jsonl"


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


def _load_records(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                out[str(r["id"])] = r
            except (ValueError, KeyError):
                continue
    return out


# --------------------------------------------------------------------------- gen
def gen_arm(arm: str, kind: str, items: list[dict], srv, max_tokens: int) -> None:
    out_path = _arm_path(arm, kind)
    done = _load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    print(f"[gen] arm={arm} {kind}: {len(done)} done, {len(todo)} to generate "
          f"(max_tokens={max_tokens})", flush=True)
    if not todo:
        return
    t0 = time.time()
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for it in todo:
            rec = {
                "id": it["id"], "kind": kind,
                "prompt_token_ids": it["prompt_token_ids"],
                "prompt_sha256": it["prompt_sha256"],
            }
            # clamp the OUTPUT budget so prompt+max_tokens fits the model context
            # (gb6144: fill the remaining context; both arms get the SAME per-item budget
            # since they share the prompt -> paired). vLLM 400s if this is exceeded.
            eff_max = max(MIN_TOKENS,
                          min(max_tokens, MAX_MODEL_LEN - len(it["prompt_token_ids"]) - CONTEXT_MARGIN))
            rec["max_tokens_eff"] = eff_max
            try:
                resp = request_greedy(srv.base_url, srv.served_model_name,
                                      it["prompt_token_ids"], eff_max)
                choice = _DO.choice_from_response(resp)
                comp_ids, src, src_kind = _DO.extract_generated_token_ids(
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
                    # carry the underlying question + shuffle seed for the by-question
                    # cluster-bootstrap sensitivity (PR #637 multi-shuffle pool).
                    rec["base_qid"] = it.get("base_qid", it["id"])
                    rec["shuffle_seed"] = it.get("shuffle_seed")
                elif kind == "aime":
                    rec["gold"] = it.get("gold"); rec["year"] = it.get("year")
                else:
                    rec["gold"] = it.get("gold")
            except Exception as exc:  # noqa: BLE001
                rec.update({
                    "completion_token_ids": [], "completion_token_sha256": None,
                    "completion_text": "", "num_completion_tokens": 0,
                    "finish_reason": "error", "answer": None, "correct": False,
                    "extract_mode": "error", "error": repr(exc)[:300],
                })
                print(f"[gen] arm={arm} {kind} id={it['id']} ERROR: {repr(exc)[:160]}", flush=True)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 16 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"[gen] arm={arm} {kind} {n_done}/{len(todo)} "
                      f"({el:.0f}s, {el/max(n_done,1):.1f}s/item)", flush=True)


# --------------------------------------------------------------------------- gap probe
def _first_divergence(ar_ids: list[int], spec_ids: list[int]) -> int | None:
    n = min(len(ar_ids), len(spec_ids))
    for i in range(n):
        if ar_ids[i] != spec_ids[i]:
            return i
    if len(ar_ids) != len(spec_ids):
        return n  # strict prefix then length divergence
    return None


def probe_gaps(kind: str, srv, tok) -> None:
    """While the AR (M=1) server is up: for each item whose spec & AR completions diverge,
    re-decode ONE greedy token from the matched prefix and record the AR-decode logit gap
    between the AR token (the decode argmax) and the SPEC token. Single-position decode
    (max_tokens=1, logprobs=K) -> O(1) memory in prefix length (no prompt_logprobs OOM /
    engine-death), and the M=1 decode-branch distribution is the exact one that drove the
    AR free-run argmax at this step."""
    ar_recs = _load_records(_arm_path("ar", kind))
    spec_recs = _load_records(_arm_path("spec", kind))
    common = sorted(set(ar_recs) & set(spec_recs))
    if not common:
        print(f"[probe] {kind}: no common items (need both arms) — skip", flush=True)
        return

    out_path = RES / f"gaps_{kind}.jsonl"
    done = _load_done_ids(out_path)
    t0 = time.time()
    n_probed = n_div = n_skip = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for iid in common:
            if iid in done:
                continue
            ar, sp = ar_recs[iid], spec_recs[iid]
            if ar.get("error") or sp.get("error"):
                continue
            ar_ids = ar.get("completion_token_ids") or []
            sp_ids = sp.get("completion_token_ids") or []
            k = _first_divergence(ar_ids, sp_ids)
            rec: dict[str, Any] = {"id": iid, "kind": kind, "first_div_index": k}
            if k is None or k >= len(ar_ids) or k >= len(sp_ids):
                # identical (k None) or pure length divergence (no competing token to probe)
                rec["divergent"] = (k is not None)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                continue
            n_div += 1
            ar_tok, spec_tok = ar_ids[k], sp_ids[k]
            prefix = list(ar["prompt_token_ids"]) + ar_ids[:k]  # dist is conditioned on this
            rec.update({"divergent": True, "ar_tok": ar_tok, "spec_tok": spec_tok,
                        "prefix_len": len(prefix)})
            if len(prefix) >= MAX_MODEL_LEN:  # no room to decode a token on top of the prefix
                rec["skipped"] = "prefix_exceeds_model_len"
                n_skip += 1
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                continue
            try:
                resp = request_decode_logprobs(srv.base_url, srv.served_model_name, prefix)
                parsed = _parse_decode_logprobs(resp, ar_tok, spec_tok, tok)
                if parsed is None:
                    rec["skipped"] = "no_logprobs"
                    n_skip += 1
                else:
                    rec.update(parsed)
                    n_probed += 1
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)[:200]
                n_skip += 1
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (n_probed + n_skip) % 16 == 0:
                print(f"[probe] {kind}: probed={n_probed} div={n_div} skip={n_skip} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[probe] {kind} DONE: divergent={n_div} probed={n_probed} skipped={n_skip} "
          f"({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- serve
def serve_arm(arm: str, eval_items: dict[str, list[dict]], evals: list[str],
              max_tokens_by_kind: dict[str, int], do_probe: bool, tok) -> dict[str, float]:
    extra_env = _spec_env(arm)
    log_path = RES / f"_serve_{arm}.log"
    print(f"[gen] === ARM {arm} === {time.strftime('%H:%M:%S')}  env={extra_env}", flush=True)
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=SERVER_PY, port=PORT,
            log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
        ) as srv:
            print(f"[gen] {arm} ready at {srv.base_url} model={srv.served_model_name}", flush=True)
            for kind in evals:
                gen_arm(arm, kind, eval_items[kind], srv, max_tokens_by_kind[kind])
            if do_probe and arm == "ar":
                for kind in evals:
                    probe_gaps(kind, srv, tok)
    finally:
        stop.set()
        sampler.join(timeout=5)
    gb = (peak["mib"] or 0.0) / 1024.0
    print(f"[gen] === ARM {arm} DONE === peak {gb:.1f} GB {time.strftime('%H:%M:%S')}", flush=True)
    return {"peak_vram_gb": gb}


def main() -> int:
    global MAX_MODEL_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="spec,ar")
    ap.add_argument("--evals", default="gpqa,gsm8k")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--no-probe", action="store_true", help="skip the logit-gap probe")
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                    help="server context (per-arm boot). Keep 6144 for GPQA to pool with "
                         "the #626 banked 198; 8192 for the AIME leg (PR #637 gb6144).")
    ap.add_argument("--gpqa-shuffle-seeds", default="",
                    help="comma list of EXTRA GPQA choice-shuffle seeds beyond the primary "
                         f"{evalsets.GPQA_SEED} (PR #637 Leg 2 n-extension). Empty = #626 behavior.")
    ap.add_argument("--aime-years", default=",".join(evalsets.AIME_YEARS_DEFAULT),
                    help="comma list from {2024,2025-I,2025-II,2025,...} for the AIME leg.")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    MAX_MODEL_LEN = int(args.max_model_len)
    smoke = args.mode == "smoke"
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    print(f"[gen] building eval sets (tokenize once, shared across arms) "
          f"max_model_len={MAX_MODEL_LEN}...", flush=True)
    tok = evalsets.load_tokenizer()
    eval_items: dict[str, list[dict]] = {}
    gpqa_seeds: list[int] = [evalsets.GPQA_SEED] + [
        int(s) for s in args.gpqa_shuffle_seeds.split(",") if s.strip()
    ]
    if "gpqa" in evals:
        eval_items["gpqa"] = evalsets.build_gpqa_items_multi(
            tok, seeds=gpqa_seeds, limit=4 if smoke else 0)
        nseeds = len(gpqa_seeds)
        print(f"[gen] gpqa shuffle seeds={gpqa_seeds} "
              f"({nseeds} permutation{'s' if nseeds != 1 else ''} pooled)", flush=True)
    if "aime" in evals:
        years = [y.strip() for y in args.aime_years.split(",") if y.strip()]
        eval_items["aime"] = evalsets.build_aime_items(
            tok, years=years, limit=4 if smoke else 0)
        print(f"[gen] aime years={years}", flush=True)
    if "gsm8k" in evals:
        gs, sig = evalsets.build_gsm8k_items(tok, limit=16 if smoke else 0)
        eval_items["gsm8k"] = gs
        (RES / "gsm8k_fewshot_sig.json").write_text(json.dumps(sig))
    for k, its in eval_items.items():
        ptoks = [len(it["prompt_token_ids"]) for it in its]
        print(f"[gen] {k}: {len(its)} items, prompt_tokens {min(ptoks)}-{max(ptoks)}", flush=True)

    max_tokens_by_kind = {
        "gpqa": (256 if smoke else GPQA_MAX_TOKENS),
        "gsm8k": (256 if smoke else GSM8K_MAX_TOKENS),
        "aime": (256 if smoke else AIME_MAX_TOKENS),
    }

    meta: dict[str, Any] = {"peaks": {}}
    do_probe = not args.no_probe
    for arm in arms:
        m = serve_arm(arm, eval_items, evals, max_tokens_by_kind, do_probe, tok)
        meta["peaks"][arm] = m["peak_vram_gb"]
    # Provenance MERGES across invocations: the AIME leg (max_model_len=8192) and the
    # GPQA leg (6144) run separately, so accumulate a per-eval config map instead of
    # letting the second invocation clobber the first. Each eval records the budget it
    # was actually generated under.
    meta_path = RES / "gen_meta.json"
    prev: dict[str, Any] = {}
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
        except (ValueError, OSError):
            prev = {}
    per_eval_cfg: dict[str, Any] = dict(prev.get("per_eval_config", {}))
    for k in evals:
        per_eval_cfg[k] = {
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": max_tokens_by_kind.get(k),
            **({"gpqa_shuffle_seeds": gpqa_seeds} if k == "gpqa" else {}),
            **({"aime_years": args.aime_years} if k == "aime" else {}),
        }
    peaks = dict(prev.get("peaks", {})); peaks.update(meta["peaks"])
    meta_path.write_text(json.dumps({
        "arms": arms, "evals": sorted(set(prev.get("evals", [])) | set(evals)),
        "mode": args.mode, "last_invocation_evals": evals,
        "max_tokens_by_kind": max_tokens_by_kind, "max_model_len": MAX_MODEL_LEN,
        "per_eval_config": per_eval_cfg,
        "gpqa_shuffle_seeds": gpqa_seeds, "aime_years": args.aime_years,
        "min_tokens": MIN_TOKENS, "max_num_seqs": 1, "batch_invariant": 1,
        "peaks": peaks, "analysis_only": True, "official_tps": 0,
    }, indent=2))
    print(f"[gen] ALL ARMS COMPLETE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
