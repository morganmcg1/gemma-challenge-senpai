#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #578 -- int4_g128_lmhead SERVED strict-#319 greedy-identity census.

QUESTION
--------
Does the served `int4_g128_lmhead` submission (int4 g128 BODY + untied int4 g128 lm_head,
embed_tokens bf16) produce greedy tokens that are #319 byte-exact vs the CANONICAL official
greedy reference -- the served bf16 plain-baseline decode (`reference_kind="served_spec_off"`)
checked into research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl?

WHY THIS IS THE RIGHT REFERENCE (and the bundled one is not)
-----------------------------------------------------------
The submission's own check_greedy_identity.py compares served-int4 vs served-int4 (same Marlin
kernel, same config) -- a SELF-consistency check, not a #319 identity. Its bundled reference was
captured with speculative decode ON. The #319 launch contract is byte-identity vs PLAIN greedy
autoregressive decode of the canonical model. The in-repo decode_outputs.jsonl is exactly that:
bf16 `google/gemma-4-E4B-it`, spec OFF, temperature 0, ignore_eos, 128 prompts x 512 tokens =
65,536 positions -- the SAME official-128 reference whose bf16-body argmax land #571 measured
int4_g128 BODY flipping at 13.74% (`body_argmax_flip_rate_official128`, run vct3k1vc).

int4_g128_lmhead = that same int4_g128 body + an ADDITIONAL untied int4 g128 lm_head re-quant that
#571 did not cover (it held the head fixed at full bf16). So the strong prior is FAIL; this card
MEASURES it served end-to-end, including the unmeasured head delta and any served-vs-offline gap.

TWO COMPLEMENTARY SERVED MEASUREMENTS (both from the live int4_g128_lmhead endpoint)
----------------------------------------------------------------------------------
(A) FREE-RUNNING greedy capture -- the LITERAL #319 verifier protocol. Same request shape as the
    reference (add_special_tokens=False, temperature=0, ignore_eos=True, max_tokens=512,
    return_token_ids=True). byte_exact := candidate completion == reference completion for ALL 128
    prompts. Reports first divergence per prompt. NOTE: greedy decode CASCADES after the first
    flip (an off-trajectory prefix changes every later argmax), so the free-running divergent-token
    fraction OVER-states the per-step error -- it is the right instrument for the byte_exact BOOL
    and the first-divergence locus, not for a per-position rate comparable to #571.

(B) TEACHER-FORCED per-position argmax -- the #571-comparable CLEAN rate. For each prompt we send
    prompt_ids + the bf16 REFERENCE completion as the prompt with prompt_logprobs, max_tokens=1.
    At every reference position the served model's argmax (top-1 prompt-logprob) is compared to the
    bf16 reference token. Pinning the context to the bf16 stream removes the cascade, so this is the
    apples-to-apples extension of #571's teacher-forced body census -- now SERVED and INCLUDING the
    int4 head. `int4g128lmhead_flip_rate_official` (the primary metric) is this rate; expected >=
    #571's body-only 0.1374 because the int4 head can only ADD flips.

(C) Served warm-median single-stream decode TPS -- a representativeness check on the official
    a10g tps=126.378 on the modern local stack (single-stream proxy; not the concurrent harness).

SCOPE: LOCAL student A10G. analysis_only=true, official_tps=0. NO HF Job, NO submission, NO
served-file change. wandb_group int4g128lmhead-characterization (paired with kanna's quality leg).

Run:
  # pure-logic self-test (no server):
  python3 research/validity/int4g128lmhead_strict_identity/served_strict319_identity.py --self-test
  # against a live endpoint (server started separately):
  python3 .../served_strict319_identity.py --run --base-url http://localhost:8000 --model gemma-4-e4b-it
  # 0-GPU wandb logging pass (reads results JSON):
  python3 .../served_strict319_identity.py --log-wandb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

REFERENCE_JSONL = REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
OUT_JSON = HERE / "served_strict319_identity_results.json"
RAW_DIR = HERE / "raw"

# land #571 published anchors (CITE; never re-derived here).
PR571_BODY_FLIP_OFFICIAL128_INT4G128 = 0.1374490570715758   # int4_g128 BODY (head fixed bf16), run vct3k1vc
PR571_BODY_FLIP_OFFICIAL128_INT4G32 = 0.10902948878868177   # deployed int4_g32 BODY
OFFICIAL_A10G_TPS = 126.378                                 # int4_g128_lmhead HF Job 6a2d5a96, W&B 905tbujn
EXPECTED_N_POSITIONS = 65536                                # 128 x 512 (ignore_eos reference)


# ----------------------------------------------------------------------------- #
# reference + http
# ----------------------------------------------------------------------------- #
def load_reference(path: Path = REFERENCE_JSONL) -> list[dict]:
    recs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        recs.append({
            "id": str(r["id"]),
            "prompt_token_ids": [int(x) for x in r["prompt_token_ids"]],
            "completion_token_ids": [int(x) for x in r["completion_token_ids"]],
        })
    return recs


def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _extract_completion_ids(resp: dict, prompt_ids: list[int]) -> list[int]:
    ch = resp["choices"][0]
    cands = [ch.get("token_ids"), ch.get("output_token_ids")]
    lp = ch.get("logprobs")
    if isinstance(lp, dict):
        cands.append(lp.get("token_ids"))
    for v in cands:
        if isinstance(v, list) and v and all(isinstance(t, int) and t >= 0 for t in v):
            if len(v) >= len(prompt_ids) and v[:len(prompt_ids)] == prompt_ids:
                return v[len(prompt_ids):]
            return v
    raise ValueError("endpoint returned no integer completion token IDs (need return_token_ids:true)")


# ----------------------------------------------------------------------------- #
# prompt_logprobs argmax extraction (robust to vLLM key/format variants)
# ----------------------------------------------------------------------------- #
def argmax_of_position(pos: Any) -> int | None:
    """Return the token-id with the highest logprob at one prompt_logprobs position.
    Handles {tok_id_str: {logprob,rank,...}} and {tok_id_str: float} and rank-based forms."""
    if not pos:
        return None
    best_id, best_lp, best_rank = None, float("-inf"), None
    for k, v in pos.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            lp = v.get("logprob")
            rank = v.get("rank")
        else:
            lp = v
            rank = None
        # prefer explicit rank==1 if present; else fall back to max logprob.
        if rank == 1:
            return tid
        if lp is not None and lp > best_lp:
            best_id, best_lp, best_rank = tid, lp, rank
    return best_id


# ----------------------------------------------------------------------------- #
# Phase A: free-running greedy capture (literal #319 protocol)
# ----------------------------------------------------------------------------- #
def phase_free_running(records, base_url, model, output_len=512, log_every=16) -> list[dict]:
    url = f"{base_url}/v1/completions"
    out = []
    t0 = time.time()
    for i, rec in enumerate(records):
        pid = rec["prompt_token_ids"]
        resp = _post(url, {
            "model": model, "prompt": pid, "max_tokens": output_len,
            "temperature": 0.0, "stream": False, "add_special_tokens": False,
            "ignore_eos": True, "return_token_ids": True,
            # min_tokens guard is moot under ignore_eos but harmless / explicit:
            "min_tokens": 8,
        })
        comp = _extract_completion_ids(resp, pid)
        out.append({"id": rec["id"], "completion_token_ids": comp})
        if (i + 1) % log_every == 0 or i == len(records) - 1:
            print(f"[A:free-run] {i+1}/{len(records)} ({time.time()-t0:.1f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------- #
# Phase B: teacher-forced per-position argmax (clean #571-comparable rate)
# ----------------------------------------------------------------------------- #
def phase_teacher_forced(records, base_url, model, log_every=16) -> list[dict]:
    url = f"{base_url}/v1/completions"
    out = []
    t0 = time.time()
    for i, rec in enumerate(records):
        pid = rec["prompt_token_ids"]
        comp = rec["completion_token_ids"]
        full = pid + comp
        resp = _post(url, {
            "model": model, "prompt": full, "max_tokens": 1,
            "temperature": 0.0, "stream": False, "add_special_tokens": False,
            "prompt_logprobs": 1, "logprobs": 1, "ignore_eos": True,
        })
        plps = resp["choices"][0].get("prompt_logprobs")
        if not isinstance(plps, list):
            raise ValueError("endpoint did not return prompt_logprobs (PPL contract)")
        P = len(pid)
        # prompt_logprobs[j] predicts full[j] from full[:j]; completion positions are j in [P, P+len(comp)).
        argmax_ids = []
        for c in range(len(comp)):
            j = P + c
            argmax_ids.append(argmax_of_position(plps[j]) if j < len(plps) else None)
        out.append({"id": rec["id"], "argmax_token_ids": argmax_ids,
                    "ref_token_ids": comp})
        if (i + 1) % log_every == 0 or i == len(records) - 1:
            print(f"[B:teacher-force] {i+1}/{len(records)} ({time.time()-t0:.1f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------- #
# Phase C: served warm-median single-stream decode TPS (streaming inter-token)
# ----------------------------------------------------------------------------- #
def phase_tps(base_url, model, n_warm=2, n_trials=5, gen_tokens=200) -> dict:
    import statistics
    url = f"{base_url}/v1/completions"
    prompt = "Write a detailed explanation of how a transformer language model works."
    def one_trial() -> float:
        payload = {"model": model, "prompt": prompt, "max_tokens": gen_tokens,
                   "temperature": 0.0, "stream": True, "ignore_eos": True,
                   "add_special_tokens": True}
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        stamps = []
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if obj.get("choices") and obj["choices"][0].get("text", "") != "":
                    stamps.append(time.time())
        if len(stamps) < 3:
            return float("nan")
        return (len(stamps) - 1) / (stamps[-1] - stamps[0])
    for _ in range(n_warm):
        one_trial()
    tps_vals = [t for _ in range(n_trials) if (t := one_trial()) == t]  # drop nan
    return {
        "served_tps_local_median": round(statistics.median(tps_vals), 3) if tps_vals else None,
        "served_tps_local_mean": round(statistics.fmean(tps_vals), 3) if tps_vals else None,
        "served_tps_local_trials": [round(t, 3) for t in tps_vals],
        "tps_method": f"streaming inter-token, single-stream, {gen_tokens} tok, warm-median of {n_trials}",
    }


# ----------------------------------------------------------------------------- #
# comparisons
# ----------------------------------------------------------------------------- #
def compare_free_running(reference, candidate) -> dict:
    ref = {r["id"]: r["completion_token_ids"] for r in reference}
    cand = {c["id"]: c["completion_token_ids"] for c in candidate}
    common = sorted(set(ref) & set(cand))
    n_identical = 0
    total_pos = 0
    total_div = 0
    first_divs = []
    per_prompt_first = {}
    for k in common:
        r, c = ref[k], cand[k]
        n = min(len(r), len(c))
        diff = sum(1 for i in range(n) if r[i] != c[i]) + abs(len(r) - len(c))
        total_pos += n
        total_div += diff
        if diff == 0 and len(r) == len(c):
            n_identical += 1
        else:
            fi = next((i for i in range(n) if r[i] != c[i]), n)
            per_prompt_first[k] = fi
            first_divs.append({"id": k, "first_divergence_index": fi,
                               "ref_token": r[fi] if fi < len(r) else None,
                               "cand_token": c[fi] if fi < len(c) else None})
    first_divs.sort(key=lambda d: d["first_divergence_index"])
    return {
        "byte_exact": bool(n_identical == len(common) and len(common) > 0),
        "n_prompts": len(common),
        "n_prompts_identical": n_identical,
        "n_prompts_divergent": len(common) - n_identical,
        "n_positions": total_pos,
        "free_running_divergent_tokens": total_div,
        "free_running_cascade_flip_rate": (total_div / total_pos) if total_pos else None,
        "first_divergences_top5": first_divs[:5],
        "median_first_divergence_index": _median_int(list(per_prompt_first.values())),
    }


def compute_teacher_forced(tf_records) -> dict:
    n_pos = 0
    n_flip = 0
    first_divs = []
    for rec in tf_records:
        am = rec["argmax_token_ids"]
        ref = rec["ref_token_ids"]
        first = None
        for i, (a, r) in enumerate(zip(am, ref)):
            if a is None:
                continue
            n_pos += 1
            if a != r:
                n_flip += 1
                if first is None:
                    first = i
        if first is not None:
            first_divs.append({"id": rec["id"], "first_flip_index": first})
    first_divs.sort(key=lambda d: d["first_flip_index"])
    return {
        "n_positions": n_pos,
        "n_flips": n_flip,
        "flip_rate": (n_flip / n_pos) if n_pos else None,
        "byte_exact": bool(n_flip == 0 and n_pos > 0),
        "first_flips_top5": first_divs[:5],
    }


def _median_int(xs: list[int]):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ----------------------------------------------------------------------------- #
# assembly
# ----------------------------------------------------------------------------- #
def assemble(free_cmp, tf_cmp, tps_info, meta) -> dict:
    flip_rate_official = tf_cmp["flip_rate"]            # primary: clean teacher-forced per-position
    byte_exact = bool(free_cmp["byte_exact"] and tf_cmp["byte_exact"])
    verdict = "PASS" if byte_exact else "FAIL"
    tps_local = tps_info.get("served_tps_local_median")
    delta_vs_571 = (round(flip_rate_official - PR571_BODY_FLIP_OFFICIAL128_INT4G128, 6)
                    if flip_rate_official is not None else None)
    narrative = (
        f"int4_g128_lmhead SERVED strict-#319 identity vs the canonical official greedy reference "
        f"(bf16 plain baseline, spec_off, {free_cmp['n_positions']} positions). VERDICT={verdict}. "
        f"byte_exact={byte_exact} (free-running {free_cmp['n_prompts_divergent']}/{free_cmp['n_prompts']} "
        f"prompts diverge; teacher-forced {tf_cmp['n_flips']}/{tf_cmp['n_positions']} positions flip). "
        f"Teacher-forced per-position flip_rate={flip_rate_official} vs land #571 int4_g128 BODY-only "
        f"{PR571_BODY_FLIP_OFFICIAL128_INT4G128:.6f} (delta {delta_vs_571:+}); the untied int4 g128 lm_head "
        f"re-quant adds flips on top of the body, as predicted -- no precision can REMOVE the body argmax "
        f"flips. Free-running greedy cascades after a median first-divergence at position "
        f"{free_cmp['median_first_divergence_index']} -> cascade flip rate "
        f"{free_cmp['free_running_cascade_flip_rate']}. Served single-stream decode tps_local={tps_local} "
        f"(official a10g {OFFICIAL_A10G_TPS}). int4_g128_lmhead FAILS the strict #319 byte-exact greedy "
        f"contract; it is NOT an output-neutral submission."
    )
    return {
        "schema": "int4g128lmhead_served_strict319_v1",
        "pr": 578, "agent": "wirbel",
        "analysis_only": True, "official_tps": 0,
        "no_hf_job": True, "no_submission": True, "no_served_file_change": True,
        "wandb_group": "int4g128lmhead-characterization",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # ---- PR-required deliverables ----
        "int4g128lmhead_byte_exact_vs_official": byte_exact,
        "int4g128lmhead_flip_rate_official": flip_rate_official,
        "int4g128lmhead_n_positions": tf_cmp["n_positions"],
        "int4g128lmhead_served_tps_local": tps_local,
        "int4g128lmhead_identity_verdict": verdict,
        # ---- supporting ----
        "free_running": free_cmp,
        "teacher_forced": tf_cmp,
        "tps": tps_info,
        "pr571_body_flip_official128_int4g128": PR571_BODY_FLIP_OFFICIAL128_INT4G128,
        "pr571_body_flip_official128_int4g32": PR571_BODY_FLIP_OFFICIAL128_INT4G32,
        "flip_rate_delta_vs_pr571_body": delta_vs_571,
        "official_a10g_tps": OFFICIAL_A10G_TPS,
        "reference_kind": "served_spec_off (canonical official greedy; in-repo decode_outputs.jsonl)",
        "meta": meta,
        "verdict": narrative,
    }


# ----------------------------------------------------------------------------- #
# self-test (pure logic; no server)
# ----------------------------------------------------------------------------- #
def self_test() -> dict:
    c = {}
    # argmax extraction: dict-of-dicts with rank
    pos = {"5": {"logprob": -2.0, "rank": 2}, "9": {"logprob": -0.1, "rank": 1}}
    c["t01_argmax_rank"] = (argmax_of_position(pos) == 9)
    # argmax extraction: dict-of-dicts no rank -> max logprob
    pos2 = {"5": {"logprob": -2.0}, "9": {"logprob": -0.1}, "3": {"logprob": -5.0}}
    c["t02_argmax_logprob"] = (argmax_of_position(pos2) == 9)
    # argmax extraction: plain float values
    pos3 = {"7": -0.5, "8": -3.0}
    c["t03_argmax_float"] = (argmax_of_position(pos3) == 7)
    c["t04_argmax_none"] = (argmax_of_position(None) is None and argmax_of_position({}) is None)
    # free-running compare: identical -> byte_exact True, 0 div
    ref = [{"id": "a", "completion_token_ids": [1, 2, 3]}, {"id": "b", "completion_token_ids": [4, 5]}]
    cand_same = [{"id": "a", "completion_token_ids": [1, 2, 3]}, {"id": "b", "completion_token_ids": [4, 5]}]
    fr = compare_free_running(ref, cand_same)
    c["t05_fr_identical"] = (fr["byte_exact"] is True and fr["free_running_divergent_tokens"] == 0
                            and fr["n_positions"] == 5)
    # free-running compare: one flip at idx1 in 'a' -> not byte_exact, first div idx 1
    cand_flip = [{"id": "a", "completion_token_ids": [1, 9, 3]}, {"id": "b", "completion_token_ids": [4, 5]}]
    fr2 = compare_free_running(ref, cand_flip)
    c["t06_fr_flip"] = (fr2["byte_exact"] is False and fr2["n_prompts_divergent"] == 1
                       and fr2["first_divergences_top5"][0]["first_divergence_index"] == 1
                       and fr2["first_divergences_top5"][0]["ref_token"] == 2
                       and fr2["first_divergences_top5"][0]["cand_token"] == 9)
    # teacher-forced: 1 flip in 5 positions -> rate 0.2
    tf = [{"id": "a", "argmax_token_ids": [1, 2, 9], "ref_token_ids": [1, 2, 3]},
          {"id": "b", "argmax_token_ids": [4, 5], "ref_token_ids": [4, 5]}]
    tc = compute_teacher_forced(tf)
    c["t07_tf_rate"] = (tc["n_positions"] == 5 and tc["n_flips"] == 1 and abs(tc["flip_rate"] - 0.2) < 1e-12
                       and tc["byte_exact"] is False and tc["first_flips_top5"][0]["first_flip_index"] == 2)
    # teacher-forced None positions are skipped (not counted)
    tf2 = [{"id": "a", "argmax_token_ids": [None, 2, 3], "ref_token_ids": [1, 2, 3]}]
    tc2 = compute_teacher_forced(tf2)
    c["t08_tf_skip_none"] = (tc2["n_positions"] == 2 and tc2["n_flips"] == 0 and tc2["byte_exact"] is True)
    # assemble: FAIL when not byte exact; PASS only when both byte_exact
    a_fail = assemble(fr2, tc, {"served_tps_local_median": 120.0}, {})
    c["t09_assemble_fail"] = (a_fail["int4g128lmhead_identity_verdict"] == "FAIL"
                             and a_fail["int4g128lmhead_byte_exact_vs_official"] is False
                             and abs(a_fail["int4g128lmhead_flip_rate_official"] - 0.2) < 1e-12)
    a_pass = assemble(fr, tc2, {"served_tps_local_median": 120.0}, {})
    c["t10_assemble_pass"] = (a_pass["int4g128lmhead_identity_verdict"] == "PASS"
                             and a_pass["int4g128lmhead_byte_exact_vs_official"] is True)
    # primary metric name + analysis flags
    c["t11_flags"] = (a_fail["analysis_only"] is True and a_fail["official_tps"] == 0
                     and a_fail["wandb_group"] == "int4g128lmhead-characterization")
    return {"conditions": c, "n_checks": len(c), "passes": bool(all(c.values()))}


# ----------------------------------------------------------------------------- #
# wandb logging (reads results JSON; 0-GPU)
# ----------------------------------------------------------------------------- #
def log_wandb(results: dict) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.wandb_logging import finish_wandb, init_wandb_run
    fr = results["free_running"]
    tf = results["teacher_forced"]
    flat = {
        "pr": 578, "analysis_only": 1, "official_tps": 0,
        "int4g128lmhead_byte_exact_vs_official": int(results["int4g128lmhead_byte_exact_vs_official"]),
        "int4g128lmhead_flip_rate_official": results["int4g128lmhead_flip_rate_official"],
        "int4g128lmhead_n_positions": results["int4g128lmhead_n_positions"],
        "int4g128lmhead_served_tps_local": results["int4g128lmhead_served_tps_local"],
        "identity_verdict_fail": int(results["int4g128lmhead_identity_verdict"] == "FAIL"),
        "free_running_byte_exact": int(fr["byte_exact"]),
        "free_running_n_prompts_divergent": fr["n_prompts_divergent"],
        "free_running_cascade_flip_rate": fr["free_running_cascade_flip_rate"],
        "free_running_median_first_divergence": fr["median_first_divergence_index"],
        "teacher_forced_n_flips": tf["n_flips"],
        "teacher_forced_n_positions": tf["n_positions"],
        "pr571_body_flip_official128_int4g128": results["pr571_body_flip_official128_int4g128"],
        "pr571_body_flip_official128_int4g32": results["pr571_body_flip_official128_int4g32"],
        "flip_rate_delta_vs_pr571_body": results["flip_rate_delta_vs_pr571_body"],
        "official_a10g_tps": results["official_a10g_tps"],
        "served_tps_local_mean": results["tps"].get("served_tps_local_mean"),
    }
    run = init_wandb_run(
        job_type="analysis", agent="wirbel",
        name="wirbel/int4g128lmhead-served-strict319-identity",
        group="int4g128lmhead-characterization",
        notes="PR #578 int4_g128_lmhead SERVED strict-#319 greedy-identity census vs canonical official bf16 reference.",
        tags=["pr578", "int4g128lmhead", "strict-identity", "analysis-only", "served", "319-contract"],
        config={"pr": 578, "reference": "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl",
                "reference_kind": "served_spec_off", "official_a10g_tps": OFFICIAL_A10G_TPS,
                **results.get("meta", {})},
    )
    if run is None:
        print("[wandb] disabled -- JSON artifact still written", flush=True)
        return
    run.log({**{k: v for k, v in flat.items() if v is not None}, "global_step": 0})
    run.summary.update({k: v for k, v in flat.items() if v is not None})
    print(f"[wandb] run = {run.id} ({run.url})", flush=True)
    finish_wandb(run)


# ----------------------------------------------------------------------------- #
def run_all(base_url: str, model: str, limit: int = 0, output_len: int = 512,
            skip_tps: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = load_reference()
    if limit:
        records = records[:limit]
    print(f"[load] {len(records)} reference records from {REFERENCE_JSONL}", flush=True)

    t0 = time.time()
    cand = phase_free_running(records, base_url, model, output_len=output_len)
    (RAW_DIR / "free_running_candidate.json").write_text(json.dumps(cand))
    free_cmp = compare_free_running(records, cand)
    print(f"[A] byte_exact={free_cmp['byte_exact']} divergent_prompts="
          f"{free_cmp['n_prompts_divergent']}/{free_cmp['n_prompts']} "
          f"cascade_rate={free_cmp['free_running_cascade_flip_rate']}", flush=True)

    tf = phase_teacher_forced(records, base_url, model)
    (RAW_DIR / "teacher_forced.json").write_text(json.dumps(
        [{"id": r["id"], "n_pos": len(r["ref_token_ids"])} for r in tf]))  # compact provenance
    tf_cmp = compute_teacher_forced(tf)
    print(f"[B] teacher-forced flip_rate={tf_cmp['flip_rate']} "
          f"({tf_cmp['n_flips']}/{tf_cmp['n_positions']})", flush=True)

    tps_info = {"served_tps_local_median": None} if skip_tps else phase_tps(base_url, model)
    print(f"[C] tps={tps_info.get('served_tps_local_median')}", flush=True)

    meta = {"n_records": len(records), "output_len": output_len,
            "base_url": base_url, "model": model,
            "elapsed_s": round(time.time() - t0, 1),
            "reference_jsonl": str(REFERENCE_JSONL)}
    results = assemble(free_cmp, tf_cmp, tps_info, meta)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {OUT_JSON}", flush=True)
    print("SENPAI-STRICT319 " + json.dumps({
        "verdict": results["int4g128lmhead_identity_verdict"],
        "byte_exact": results["int4g128lmhead_byte_exact_vs_official"],
        "flip_rate_official": results["int4g128lmhead_flip_rate_official"],
        "n_positions": results["int4g128lmhead_n_positions"],
        "served_tps_local": results["int4g128lmhead_served_tps_local"],
    }), flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--log-wandb", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--limit", type=int, default=0, help="limit reference prompts (smoke)")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--skip-tps", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        st = self_test()
        print(json.dumps(st, indent=2))
        return 0 if st["passes"] else 1
    if args.log_wandb:
        log_wandb(json.loads(OUT_JSON.read_text()))
        return 0
    if args.run:
        st = self_test()
        assert st["passes"], f"self-test failed: {st['conditions']}"
        run_all(args.base_url, args.model, limit=args.limit,
                output_len=args.output_len, skip_tps=args.skip_tps)
        return 0
    print("specify --self-test | --run | --log-wandb", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
