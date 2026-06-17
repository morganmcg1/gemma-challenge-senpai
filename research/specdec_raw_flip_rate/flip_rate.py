#!/usr/bin/env python
"""PR #616 wirbel — RAW per-step structural flip rate: teacher-forced M=8 verify vs M=1 AR.

THE LOAD-BEARING QUESTION (follow-up #1 of my #607 census):
#607 proved int4+MTP-K7 spec breaks strict #319 by 31048/65536 tokens (47.4%) in
FREE-RUNNING greedy. But that 47.4% is CASCADE-AMPLIFIED: one early argmax flip diverges
the whole downstream sequence (onset median 136). The number that decides whether option B
can EVER be made #319-compatible is the UN-amplified, per-step STRUCTURAL flip rate: holding
the reference greedy prefix FIXED (teacher-forced, no cascade), what fraction of positions
have a different next-token argmax under the M=K+1=8 batched spec-verify forward vs the M=1
AR forward?

  * tiny (<0.1%) + flips are near-ties  -> IDENTITY_RESCUABLE_BY_SMALL_TOLERANCE (option B
    rescuable under a relaxed top-1-logprob contract)
  * large (>1%)  OR  flips are decisive -> STRUCTURALLY_HOPELESS (stop trying to rescue)

REALIZATION (no vLLM patching, official endpoint path, exact-M faithful):
  The int4-Marlin GEMM is M-dependent (#607 mechanism: M=8 verify tile/reduction != M=1 AR).
  vLLM 0.22.0 v1 routes a Linear's GEMM at M = number of query tokens in the forward, with
  ONE M-agnostic compressed-tensors Marlin dispatch shared by prefill chunks AND the
  spec-verify forward (Explore-validated in vllm source). So:

    arm m8 : reference-mode (spec OFF) server, MAX_NUM_BATCHED_TOKENS=8 -> a long prompt is
             chunked-prefilled in EXACT 8-token forwards == M=8 GEMM batch + 8 query rows into
             the (prefill-branch) TRITON_ATTN kernel. This is the M=K+1=8 "verify-shape" forward.
    arm m1 : MAX_NUM_BATCHED_TOKENS=1 -> M=1 forwards == the M=1 "AR-shape" forward (same
             prefill-branch TRITON_ATTN, single query row). SAME prompts as m8.

  PRIMARY (PR step 3): teacher-force the EXACT #607 plain-AR int4 greedy reference
  (decode_ref_b.jsonl) as the prompt into BOTH arms and read prompt_logprobs at every
  generated position. A flip = argmax(m8 forward) != argmax(m1 forward) at that position.
  Because BOTH arms use the prefill-branch TRITON_ATTN, the attention CODE PATH is held
  constant and the ONLY thing that varies is the batch dimension M (8 vs 1) -> this isolates
  the int4-Marlin/attn M-dependence (#607 mechanism) with a floor of 0 BY CONSTRUCTION
  (m1-vs-m1 is identical). Same boot, same tokens, no cross-boot confound, no cascade.

  CAVEAT (measured, see secondary diagnostics): the deployed cand's verify forward runs as a
  *(decode, FULL)* cudagraph at size 8 (server_cand.log: "Capturing CUDA graphs (decode, FULL)"
  largest=8, TRITON_ATTN forced for Gemma4 het-head-dims), i.e. the DECODE attention branch,
  whereas teacher-forcing via prompt_logprobs only reaches the PREFILL branch. So m8 reproduces
  the deployed verify's M=8 GEMM exactly but its attention via the prefill branch. The
  prefill-vs-decode branch divergence is itself ~1% of argmaxes (secondary m1-vs-decode-ref),
  so the endpoint cannot pin the deployed decode-branch break below that; the PRIMARY is the
  clean lower bound on the M-dependence and the cleanest endpoint-reachable answer to step 3.

Headline = (# positions where m8 argmax != m1 argmax) / (total positions), 95% CI via
per-prompt CLUSTER bootstrap (positions within a prompt are autocorrelated). Plus the
logit-gap distribution at flips (gap = logprob_M8(m8_argmax) - logprob_M8(m1_argmax) >= 0; a
relaxed top-1-logprob acceptor with tolerance tau preserves the m1 token iff gap <= tau) and
the residual-flip-rate vs tolerance sweep that decides rescuability.

LOCAL ONLY: analysis_only=True, official_tps=0, single A10G, NO HF Job / no --launch / no
submission change. Reuses the official ppl_endpoint request shape + the #607 harness primitives.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "int4_mtp_batchinv"

# #607 build + drafter (reused, NO rebuild — disk is tight after two ENOSPC events).
MODEL_DIR = "/workspace/gemma_build/int4_g128_lmhead"
K = 7  # num_speculative_tokens of the fire candidate; M = K+1 = 8 verify granularity.

# The exact #607 plain-AR int4 greedy reference (warm scored 'b' pass): the teacher-forced
# prefix. 128 prompts x 512 generated tokens, same build/seed/prompts as this measurement.
REFERENCE_JSONL = ROOT / "research" / "specdec_verify_identity_census" / "decode_ref_b.jsonl"

PROMPT_LOGPROBS = 20  # top-k returned per position; argmax (rank 1) + ref token always in it.

# Cited anchors (NOT re-derived) — for the internal-consistency cross-check.
ANCHOR_607_DIVERGENT_FRAC = 31048 / 65536  # 0.4738 free-running (cascade-amplified)
ANCHOR_607_ONSET_MEDIAN = 136              # free-running first-divergence onset (#607)
ANCHOR_607_SEQ_EXACT = 0.2109
ANCHOR_KANNA19_FLIP = 0.00376              # kanna #19 int4+MTP BI=1 flip/tok (separate-boot)

# Verdict thresholds (PR #616).
FLIP_TINY = 0.001   # < 0.1% -> potentially rescuable
FLIP_LARGE = 0.01   # > 1%   -> structurally hopeless
# A flip is a "near-tie" (a small tolerance could rescue it) if the M=8 gap is small.
NEARTIE_NATS = 0.5
TOLERANCE_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


# --------------------------------------------------------------------------- server env
def base_env(chunk: int) -> dict[str, str]:
    """Reference-mode (spec OFF = plain int4 M=1 AR target) serve recipe, with the prefill
    CHUNK forced to `chunk` tokens via MAX_NUM_BATCHED_TOKENS (the only knob that sets the
    per-forward query-token count = the Marlin GEMM M). BI=1, conc=1, FlashInfer sampler off,
    cudagraph as-served (chunk prefill runs eager regardless)."""
    return {
        "MODEL_ID": MODEL_DIR,
        "SENPAI_REFERENCE_MODE": "1",          # spec OFF -> pure int4 target forward
        "NUM_SPECULATIVE_TOKENS": "0",
        "VLLM_BATCH_INVARIANT": "1",           # ship config; the path #607 showed Marlin escapes
        "MAX_NUM_SEQS": "1",
        "MAX_MODEL_LEN": "4096",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": str(chunk),  # == prefill chunk == GEMM M
        "VLLM_USE_FLASHINFER_SAMPLER": "0",     # PyTorch-native lowest-index argmax tie-break
    }


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


def wait_gpu_free(threshold_mib: float = 3000.0, timeout_s: float = 180.0) -> float | None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = _gpu_used_mib()
        if last <= threshold_mib:
            return last
        time.sleep(3.0)
    return last


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        peak["mib"] = max(peak["mib"], _gpu_used_mib())
        stop.wait(2.0)


# --------------------------------------------------------------------------- reference
def load_reference(path: Path, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = d["prompt_token_ids"]
            comp = d["completion_token_ids"]
            full = list(prompt) + list(comp)
            rows.append({
                "id": d["id"],
                "index": d["index"],
                "prompt_len": len(prompt),
                "comp_len": len(comp),
                "full": full,
                "score_start": len(prompt),
                "score_end": len(full),
            })
    rows.sort(key=lambda r: r["index"])
    return rows[:n]


# --------------------------------------------------------------------------- scoring
def request_prompt_logprobs(base_url: str, model: str, token_ids: list[int], timeout_s: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
        "prompt_logprobs": PROMPT_LOGPROBS,
        "add_special_tokens": False,
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
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc


def _normalize_entry(entry: Any) -> dict[int, tuple[float, int | None]]:
    """prompt_logprobs[i] -> {token_id: (logprob, rank)}; handles int/str/token_id: keys."""
    out: dict[int, tuple[float, int | None]] = {}
    if not isinstance(entry, dict):
        return out
    for k, v in entry.items():
        try:
            tok = int(k)
        except (ValueError, TypeError):
            ks = str(k)
            if ks.startswith("token_id:"):
                try:
                    tok = int(ks.split(":", 1)[1])
                except ValueError:
                    continue
            else:
                continue
        if isinstance(v, dict):
            lp = v.get("logprob")
            rank = v.get("rank")
        else:
            lp, rank = (float(v) if isinstance(v, (int, float)) else None), None
        if lp is None:
            continue
        out[tok] = (float(lp), rank)
    return out


def _argmax_token(norm: dict[int, tuple[float, int | None]]) -> tuple[int, float]:
    """argmax = rank-1 token (vLLM rank is 1-indexed); fall back to max logprob, lowest-id tie."""
    rank1 = [(tok, lp) for tok, (lp, rank) in norm.items() if rank == 1]
    if len(rank1) == 1:
        return rank1[0]
    best_tok, best_lp = None, -math.inf
    for tok, (lp, _rank) in norm.items():
        if lp > best_lp or (lp == best_lp and (best_tok is None or tok < best_tok)):
            best_tok, best_lp = tok, lp
    return best_tok, best_lp


def score_prompt(base_url: str, model: str, rec: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    resp = request_prompt_logprobs(base_url, model, rec["full"], timeout_s)
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("response had no choices")
    logprobs = choices[0].get("prompt_logprobs") or resp.get("prompt_logprobs")
    if not isinstance(logprobs, list):
        raise RuntimeError("response had no prompt_logprobs list")
    full = rec["full"]
    s, e = rec["score_start"], rec["score_end"]
    if len(logprobs) < e:
        raise RuntimeError(f"got {len(logprobs)} prompt_logprobs for {e} positions")

    # vs-reference (SECONDARY brittleness): flip = this-arm argmax != decode-path ref token.
    flips: list[int] = []
    gaps: list[float] = []     # gap = logprob(argmax) - logprob(ref_token); >0 iff flip
    missing_ref = 0
    # per-position state for the PRIMARY cross-arm (m8-vs-m1) comparison:
    amax_toks: list[int] = []          # this arm's argmax token id (-1 if no logprobs)
    amax_lps: list[float] = []         # this arm's logprob of that argmax
    tk_maps: list[dict[str, float]] = []  # this arm's top-k {str(token_id): logprob}
    for idx in range(s, e):
        ref_tok = full[idx]
        norm = _normalize_entry(logprobs[idx])
        if not norm:
            flips.append(0)
            gaps.append(0.0)
            amax_toks.append(-1)
            amax_lps.append(0.0)
            tk_maps.append({})
            continue
        amax_tok, amax_lp = _argmax_token(norm)
        amax_toks.append(amax_tok)
        amax_lps.append(amax_lp)
        tk_maps.append({str(t): round(lp, 6) for t, (lp, _r) in norm.items()})
        ref_lp = norm.get(ref_tok, (None, None))[0]
        if ref_lp is None:
            # ref token outside top-k AND not echoed (rare); a flip with unknown gap lower-bounded
            # by the top-k floor. Treat as flip; gap = amax_lp - (min top-k lp) as a lower bound.
            missing_ref += 1
            flips.append(1 if amax_tok != ref_tok else 0)
            gaps.append(max(0.0, amax_lp - min(lp for lp, _ in norm.values())))
            continue
        flip = 1 if amax_tok != ref_tok else 0
        flips.append(flip)
        gaps.append(max(0.0, amax_lp - ref_lp))
    return {
        "id": rec["id"], "index": rec["index"],
        "prompt_len": rec["prompt_len"], "n_positions": e - s,
        "n_flips": sum(flips), "flips": flips, "gaps": gaps,
        "missing_ref": missing_ref,
        "first_flip_idx": next((i for i, f in enumerate(flips) if f), None),
        "amax": amax_toks, "amax_lp": amax_lps, "tk": tk_maps,
    }


def _arm_files(label: str) -> tuple[Path, Path]:
    return HERE / f"score_{label}.jsonl", HERE / f"score_{label}.summary.json"


def _arm_complete(out_file: Path, summary_file: Path, n: int) -> bool:
    if not out_file.exists() or not summary_file.exists():
        return False
    try:
        rows = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        return False
    return len(rows) == n


def run_arm(*, server_python: Path, label: str, chunk: int, recs: list[dict[str, Any]],
            port: int, timeout_s: int, resume: bool) -> dict[str, Any]:
    out_file, summary_file = _arm_files(label)
    n = len(recs)
    if resume and _arm_complete(out_file, summary_file, n):
        print(f"[flip] [{label}] reusing complete score ({n} prompts) — skip boot", flush=True)
        rows = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
        summ = json.loads(summary_file.read_text())
        return {"label": label, "chunk": chunk, "rows": rows, "reused": True,
                "peak_vram_gb": 0.0, "plumbing": summ.get("plumbing", {})}

    extra_env = base_env(chunk)
    log_path = HERE / f"server_{label}.log"
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            print(f"[flip] [{label}] chunk={chunk} (M={chunk}) ready in {time.time()-t0:.0f}s; "
                  f"scoring {n} prompts via prompt_logprobs", flush=True)
            with open(out_file, "w", encoding="utf-8") as fh:
                for i, rec in enumerate(recs):
                    r = score_prompt(srv.base_url, srv.served_model_name, rec, timeout_s)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    rows.append(r)
                    if (i + 1) % 16 == 0 or i + 1 == n:
                        done = sum(x["n_flips"] for x in rows)
                        tot = sum(x["n_positions"] for x in rows)
                        print(f"[flip] [{label}] {i+1}/{n} prompts; flips so far {done}/{tot} "
                              f"({done/tot:.4%})", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=5)
    plumbing = _grep_log(log_path, [
        "SENPAI_REFERENCE_MODE active", "MarlinLinearKernel", "max_num_batched_tokens",
        "Speculative", "Capturing CUDA graphs", "chunked", "num_speculative_tokens",
    ])
    summ = {"label": label, "chunk": chunk, "n_prompts": n, "plumbing": plumbing,
            "peak_vram_gb": (peak["mib"] or 0.0) / 1024.0}
    summary_file.write_text(json.dumps(summ, indent=2))
    print(f"[flip] [{label}] done in {time.time()-t0:.0f}s, peak {summ['peak_vram_gb']:.1f} GB", flush=True)
    return {"label": label, "chunk": chunk, "rows": rows, "reused": False,
            "peak_vram_gb": summ["peak_vram_gb"], "plumbing": plumbing}


def _grep_log(log_path: Path, needles: list[str]) -> dict[str, bool]:
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return {n: False for n in needles}
    return {n: (n in text) for n in needles}


# --------------------------------------------------------------------------- statistics
def _quantiles(xs: list[float], qs: list[float]) -> dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": None for q in qs}
    s = sorted(xs)
    out = {}
    for q in qs:
        if len(s) == 1:
            out[f"p{int(q*100)}"] = s[0]
            continue
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        out[f"p{int(q*100)}"] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def cluster_bootstrap_ci(per_prompt: list[tuple[int, int]], iters: int = 10000,
                         seed: int = 1) -> tuple[float, float]:
    """95% CI on the pooled flip rate by resampling PROMPTS (clusters) with replacement.
    per_prompt = [(n_flips, n_positions), ...]."""
    if not per_prompt:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(per_prompt)
    rates = []
    for _ in range(iters):
        f = t = 0
        for _ in range(m):
            nf, npos = per_prompt[rng.randrange(m)]
            f += nf
            t += npos
        rates.append(f / t if t else 0.0)
    rates.sort()
    lo = rates[int(0.025 * (iters - 1))]
    hi = rates[int(0.975 * (iters - 1))]
    return (lo, hi)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), center + half)


def cross_compare(rows_m1: list[dict[str, Any]],
                  rows_m8: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PRIMARY (PR step 3): per-position argmax(M=1 prefill forward) vs argmax(M=8 prefill
    forward), matched by prompt id + position. Both arms are teacher-forced prefill, so the
    TRITON_ATTN code path is held constant and ONLY the batch dimension M (1 vs 8) differs ->
    isolates the int4-Marlin/attn M-dependence. m1-vs-m1 would be 0 by construction.

    gap at a flip = logprob_M8(amax_M8) - logprob_M8(amax_M1) >= 0: how much MORE the M=8
    forward prefers its own argmax over the token the M=1 forward chose. A relaxed top-1-logprob
    acceptor with tolerance tau preserves the M=1 token iff gap <= tau. Returns rows in the
    analyze_arm() schema so all the CI / gap / tolerance machinery is reused unchanged."""
    by_id_m8 = {r["id"]: r for r in rows_m8}
    out: list[dict[str, Any]] = []
    for ra in rows_m1:
        rb = by_id_m8.get(ra["id"])
        if rb is None:
            continue
        a_tok = ra["amax"]
        b_tok, b_lp, b_tk = rb["amax"], rb["amax_lp"], rb["tk"]
        n = min(ra["n_positions"], rb["n_positions"], len(a_tok), len(b_tok))
        flips: list[int] = []
        gaps: list[float] = []
        missing = 0
        for i in range(n):
            ta, tb = a_tok[i], b_tok[i]
            if ta < 0 or tb < 0 or ta == tb:
                flips.append(0)
                gaps.append(0.0)
                continue
            flips.append(1)
            lp_b_b = b_lp[i]
            lp_b_a = b_tk[i].get(str(ta))  # M=8 logprob of the token M=1 chose
            if lp_b_a is None:
                # M=1's token is outside M=8's top-k -> decisive flip; gap lower-bounded by the
                # M=8 top-k floor (a relaxed acceptor would NOT rescue it at any small tau).
                missing += 1
                floor_lp = min(b_tk[i].values()) if b_tk[i] else lp_b_b
                gaps.append(max(0.0, lp_b_b - floor_lp))
            else:
                gaps.append(max(0.0, lp_b_b - lp_b_a))
        out.append({
            "id": ra["id"], "index": ra["index"],
            "prompt_len": ra["prompt_len"], "n_positions": n,
            "n_flips": sum(flips), "flips": flips, "gaps": gaps,
            "missing_ref": missing,
            "first_flip_idx": next((i for i, f in enumerate(flips) if f), None),
        })
    return out


def analyze_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_prompt = [(r["n_flips"], r["n_positions"]) for r in rows]
    total_flips = sum(r["n_flips"] for r in rows)
    total_pos = sum(r["n_positions"] for r in rows)
    rate = total_flips / total_pos if total_pos else float("nan")
    flip_gaps = [g for r in rows for f, g in zip(r["flips"], r["gaps"]) if f]
    boot_lo, boot_hi = cluster_bootstrap_ci(per_prompt)
    wil_lo, wil_hi = wilson_ci(total_flips, total_pos)
    onsets = [r["first_flip_idx"] for r in rows if r["first_flip_idx"] is not None]
    prompt_rates = [nf / npos for nf, npos in per_prompt if npos]
    # residual flip rate after a relaxed top-1-logprob tolerance tau (a flip is rescued if its
    # M=8 gap <= tau, i.e. a relaxed acceptor would still accept the reference/draft token).
    tol_sweep = {}
    for tau in TOLERANCE_GRID:
        residual = sum(1 for g in flip_gaps if g > tau)
        tol_sweep[f"tau_{tau}"] = {
            "residual_flips": residual,
            "residual_rate": residual / total_pos if total_pos else float("nan"),
            "rescued_frac_of_flips": (1 - residual / total_flips) if total_flips else float("nan"),
        }
    return {
        "total_flips": total_flips,
        "total_positions": total_pos,
        "flip_rate": rate,
        "flip_rate_ci95_cluster_bootstrap": [boot_lo, boot_hi],
        "flip_rate_ci95_wilson_naive": [wil_lo, wil_hi],
        "n_prompts": len(rows),
        "n_prompts_with_any_flip": sum(1 for nf, _ in per_prompt if nf),
        "prompt_flip_rate_min": min(prompt_rates) if prompt_rates else None,
        "prompt_flip_rate_median": statistics.median(prompt_rates) if prompt_rates else None,
        "prompt_flip_rate_max": max(prompt_rates) if prompt_rates else None,
        "first_flip_idx_median": int(statistics.median(onsets)) if onsets else None,
        "first_flip_idx_min": min(onsets) if onsets else None,
        "missing_ref_positions": sum(r.get("missing_ref", 0) for r in rows),
        "gap_at_flips": {
            "n": len(flip_gaps),
            "mean": statistics.fmean(flip_gaps) if flip_gaps else None,
            **_quantiles(flip_gaps, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
            "frac_lt_0.05": sum(g < 0.05 for g in flip_gaps) / len(flip_gaps) if flip_gaps else None,
            "frac_lt_0.1": sum(g < 0.1 for g in flip_gaps) / len(flip_gaps) if flip_gaps else None,
            "frac_lt_0.5": sum(g < 0.5 for g in flip_gaps) / len(flip_gaps) if flip_gaps else None,
            "frac_lt_1.0": sum(g < 1.0 for g in flip_gaps) / len(flip_gaps) if flip_gaps else None,
        },
        "tolerance_sweep": tol_sweep,
    }


def synthesize(primary: dict[str, Any], m8_ref: dict[str, Any],
               m1_ref: dict[str, Any] | None) -> dict[str, Any]:
    """primary = cross-arm m8-vs-m1 (clean pure-M, PR headline). m8_ref/m1_ref = vs-decode-ref
    brittleness diagnostics (m8/m1 prefill argmax vs the #607 decode-path AR token)."""
    rate = primary["flip_rate"]
    ci_lo, ci_hi = primary["flip_rate_ci95_cluster_bootstrap"]
    gap = primary["gap_at_flips"]
    median_gap = gap.get("p50")
    frac_neartie = gap.get("frac_lt_0.5")
    # near-tie verdict: a small (NEARTIE_NATS) top-1-logprob tolerance rescues most flips?
    rescued_at_neartie = None
    sweep = primary["tolerance_sweep"].get(f"tau_{NEARTIE_NATS}")
    if sweep:
        rescued_at_neartie = sweep["rescued_frac_of_flips"]
    flips_are_neartie = (frac_neartie is not None and frac_neartie >= 0.5)

    if ci_hi < FLIP_TINY and flips_are_neartie:
        verdict = "IDENTITY_RESCUABLE_BY_SMALL_TOLERANCE"
    elif ci_lo > FLIP_LARGE:
        verdict = "STRUCTURALLY_HOPELESS"
    elif rate > FLIP_LARGE:
        verdict = "STRUCTURALLY_HOPELESS"
    elif rate < FLIP_TINY and flips_are_neartie:
        verdict = "IDENTITY_RESCUABLE_BY_SMALL_TOLERANCE"
    else:
        # middle band (0.1%-1%): the tolerance sweep + cascade arithmetic decide.
        verdict = "MIDDLE_BAND_TOLERANCE_DEPENDENT"

    # internal-consistency cross-check: an independent per-step flip prob f predicts the
    # FREE-RUNNING first-divergence onset (#607) as a geometric median ln(2)/f.
    pred_onset_median = (math.log(2) / rate) if rate > 0 else None
    pred_onset_mean = (1.0 / rate) if rate > 0 else None
    return {
        "pr616_verdict": verdict,
        "flip_rate": rate,
        "flip_rate_ci95": [ci_lo, ci_hi],
        "flip_rate_pct": rate * 100,
        "threshold_tiny_pct": FLIP_TINY * 100,
        "threshold_large_pct": FLIP_LARGE * 100,
        "median_gap_at_flips_nats": median_gap,
        "frac_flips_neartie_lt_0.5nats": frac_neartie,
        "rescued_frac_at_tol_0.5nats": rescued_at_neartie,
        "flips_are_neartie": flips_are_neartie,
        # the clean pure-M floor is 0 by construction (m1-vs-m1); these are brittleness probes.
        "pure_M_floor_m1_vs_m1": 0.0,
        "brittleness_m8_vs_decode_ref": m8_ref["flip_rate"],
        "brittleness_m1_vs_decode_ref_prefill_vs_decode_branch": (m1_ref or {}).get("flip_rate"),
        "predicted_607_onset_median_from_rate": pred_onset_median,
        "predicted_607_onset_mean_from_rate": pred_onset_mean,
        "measured_607_onset_median": ANCHOR_607_ONSET_MEDIAN,
        "onset_consistency_ratio": (pred_onset_median / ANCHOR_607_ONSET_MEDIAN)
        if pred_onset_median else None,
        "kanna19_flip_anchor": ANCHOR_KANNA19_FLIP,
        "anchor_607_free_running_divergent_frac": ANCHOR_607_DIVERGENT_FRAC,
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--floor-prompts", type=int, default=0,
                    help="prompts for the M=1 arm; 0 = match --num-prompts (needed for the "
                         "cross-arm m8-vs-m1 primary). Set <num for a faster partial run.")
    ap.add_argument("--reference", default=str(REFERENCE_JSONL))
    ap.add_argument("--m8-port", type=int, default=8041)
    ap.add_argument("--m1-port", type=int, default=8042)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--extra-chunks", default="",
                    help="comma list of extra chunk sizes to sweep (e.g. 4,16) on --num-prompts")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 prompts wiring check (not a verdict)")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default="specdec-raw-flip-rate")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts, args.floor_prompts = 4, 4

    for note in paths.prepare_local_gpu_env():
        print(f"[flip] {note}", flush=True)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"[flip] FATAL: reference {ref_path} not found", flush=True)
        return 2
    recs = load_reference(ref_path, args.num_prompts)
    print(f"[flip] loaded {len(recs)} reference seqs from {ref_path.name} "
          f"(prompt_len {min(r['prompt_len'] for r in recs)}-{max(r['prompt_len'] for r in recs)}, "
          f"score {recs[0]['comp_len']} positions/prompt)", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    vllm_ver = harness._dist_version(server_python, "vllm")
    tf_ver = harness._dist_version(server_python, "transformers")
    print(f"[flip] server_python={server_python} vllm={vllm_ver} transformers={tf_ver}", flush=True)
    if vllm_ver != "0.22.0":
        print(f"[flip] WARNING: expected vllm 0.22.0 (the submission pin), got {vllm_ver}", flush=True)

    resume = not args.no_resume
    arms: list[dict[str, Any]] = []

    # arm m8 — the M=8 verify forward (headline), full prompt set.
    m8 = run_arm(server_python=server_python, label="m8", chunk=8, recs=recs,
                 port=args.m8_port, timeout_s=args.timeout_s, resume=resume)
    arms.append(m8)

    # extra-M sweep (optional robustness: flip rate vs M).
    extra = [int(c) for c in args.extra_chunks.split(",") if c.strip()]
    for c in extra:
        wait_gpu_free()
        arms.append(run_arm(server_python=server_python, label=f"m{c}", chunk=c, recs=recs,
                            port=args.m8_port, timeout_s=args.timeout_s, resume=resume))

    # arm m1 — the M=1 prefill forward; SAME prompts as m8 so the cross-arm primary can match
    # every position. chunk=1 prefills one token at a time, so it is the slow arm.
    floor_arm = None
    if not args.no_floor:
        wait_gpu_free()
        floor_n = args.floor_prompts if args.floor_prompts > 0 else args.num_prompts
        floor_recs = recs[:floor_n]
        floor_arm = run_arm(server_python=server_python, label="m1", chunk=1, recs=floor_recs,
                            port=args.m1_port, timeout_s=max(args.timeout_s, 1800), resume=resume)
        arms.append(floor_arm)

    # ---- analysis ----
    # PRIMARY (PR step 3, clean): argmax(M=1 prefill) vs argmax(M=8 prefill) per position.
    cross_rows = cross_compare(floor_arm["rows"], m8["rows"]) if floor_arm else []
    primary_stats = analyze_arm(cross_rows) if cross_rows else None
    # SECONDARY brittleness: each arm's prefill argmax vs the #607 decode-path AR reference token.
    m8_ref_stats = analyze_arm(m8["rows"])
    m1_ref_stats = analyze_arm(floor_arm["rows"]) if floor_arm else None
    extra_stats = {a["label"]: analyze_arm(a["rows"]) for a in arms if a["label"] not in ("m8", "m1")}
    headline = primary_stats if primary_stats else m8_ref_stats
    verdict = synthesize(headline, m8_ref_stats, m1_ref_stats)

    report = {
        "pr": 616,
        "analysis_only": True,
        "official_tps": 0,
        "smoke": args.smoke,
        "config": {
            "model_dir": MODEL_DIR, "k": K, "M_verify": K + 1,
            "vllm_version": vllm_ver, "transformers_version": tf_ver,
            "batch_invariant": 1, "max_num_seqs": 1, "seed": paths.SEED,
            "num_prompts": args.num_prompts, "floor_prompts": args.floor_prompts,
            "output_len": recs[0]["comp_len"], "prompt_logprobs": PROMPT_LOGPROBS,
            "reference": str(ref_path),
            "method": "teacher-forced prompt_logprobs into two reference-mode servers; "
                      "MAX_NUM_BATCHED_TOKENS=chunk sets the prefill chunk = forward batch M. "
                      "PRIMARY = argmax(m8 chunk=8) vs argmax(m1 chunk=1) per position (both "
                      "prefill-branch, only M differs; pure-M, floor 0 by construction). "
                      "SECONDARY = each arm's argmax vs the #607 decode-path AR reference token.",
            "caveat": "deployed cand verify forward = (decode, FULL) cudagraph @ size 8 "
                      "(TRITON_ATTN, Gemma4 het-head-dims); prompt_logprobs only reaches the "
                      "PREFILL branch, so m8 matches the verify M=8 GEMM but not its decode-branch "
                      "attention. prefill-vs-decode-branch divergence ~= secondary m1-vs-ref; the "
                      "deployed decode-branch break is bounded below by the PRIMARY pure-M rate.",
            "analysis_only": True, "official_tps": 0,
        },
        "headline_pure_M_m8_vs_m1": primary_stats,
        "secondary_m8_vs_decode_ref": m8_ref_stats,
        "secondary_m1_vs_decode_ref": m1_ref_stats,
        "primary_unavailable_used_m8_ref_fallback": primary_stats is None,
        "extra_M_sweep": extra_stats,
        "verdict": verdict,
        "plumbing": {a["label"]: a["plumbing"] for a in arms},
        "peak_vram_gb": max((a.get("peak_vram_gb") or 0.0) for a in arms),
    }
    out = HERE / ("flip_report.smoke.json" if args.smoke else "flip_report.json")
    out.write_text(json.dumps(report, indent=2, default=str))

    # ---- console verdict ----
    print("\n" + "=" * 76, flush=True)
    print(f"[PR616] raw per-step structural flip rate ({'SMOKE' if args.smoke else 'FULL'})", flush=True)
    print(f"  vLLM={vllm_ver} K={K} M_verify={K+1} BI=1 conc=1  {args.num_prompts} prompts x "
          f"{recs[0]['comp_len']} positions", flush=True)
    r = headline
    tag = "pure-M m8-vs-m1" if primary_stats else "m8-vs-decode-ref (FALLBACK, confounded)"
    print(f"  HEADLINE  {tag} flip rate: {r['flip_rate']:.4%} "
          f"({r['total_flips']}/{r['total_positions']})  "
          f"CI95[cluster]=[{r['flip_rate_ci95_cluster_bootstrap'][0]:.4%}, "
          f"{r['flip_rate_ci95_cluster_bootstrap'][1]:.4%}]", flush=True)
    g = r["gap_at_flips"]
    print(f"  gap@flips (nats): median={g.get('p50')} mean={g.get('mean')} "
          f"p90={g.get('p90')}  frac<0.5={g.get('frac_lt_0.5')}", flush=True)
    print(f"  pure-M floor (m1-vs-m1): 0 by construction", flush=True)
    print(f"  BRITTLENESS  m8-vs-decode-ref: {m8_ref_stats['flip_rate']:.4%}"
          + (f"   m1-vs-decode-ref (prefill-vs-decode branch): {m1_ref_stats['flip_rate']:.4%}"
             if m1_ref_stats else ""), flush=True)
    for label, st in extra_stats.items():
        print(f"  sweep {label}: {st['flip_rate']:.4%}", flush=True)
    print(f"  predicted #607 onset median from rate = {verdict['predicted_607_onset_median_from_rate']}"
          f" (measured 136; ratio {verdict['onset_consistency_ratio']})", flush=True)
    print(f"  >>> PR616 VERDICT: {verdict['pr616_verdict']}", flush=True)
    print(f"  report -> {out}", flush=True)
    print("=" * 76, flush=True)

    if not args.no_wandb:
        # The report is already on disk (line above); a wandb hiccup (missing pkg in the
        # launch interpreter, network, init error) must NOT discard a completed analysis.
        try:
            _log_wandb(report, name=args.wandb_name, group=args.wandb_group)
        except Exception as exc:  # noqa: BLE001
            print(f"[flip] WARNING: wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"report preserved at {out}. Re-log with: uv run python flip_rate.py "
                  f"(resume reuses score_*.jsonl, no reboot).", flush=True)
    return 0


def _log_wandb(report: dict[str, Any], *, name: str | None, group: str | None) -> None:
    try:
        from scripts import wandb_logging as wl
    except ImportError:
        return
    cfg = report["config"]
    run = wl.init_wandb_run(
        job_type="specdec-raw-flip-rate", agent="wirbel",
        name=name or "wirbel/specdec-raw-flip-rate",
        group=group or "specdec-raw-flip-rate",
        notes="PR616 teacher-forced per-step argmax flip rate: M=8 verify vs M=1 AR, no cascade",
        tags=["pr616", "specdec", "greedy-identity", "flip-rate", "int4-mtp", "teacher-forced"],
        config=cfg,
    )
    if run is None:
        print("[flip] wandb not configured (no API key/mode) — skipping", flush=True)
        return
    metrics: dict[str, Any] = {}
    if report.get("headline_pure_M_m8_vs_m1"):
        metrics.update(wl.flatten_numeric("headline_pureM", report["headline_pure_M_m8_vs_m1"]))
    metrics.update(wl.flatten_numeric("secondary_m8_vs_ref", report["secondary_m8_vs_decode_ref"]))
    if report.get("secondary_m1_vs_decode_ref"):
        metrics.update(wl.flatten_numeric("secondary_m1_vs_ref", report["secondary_m1_vs_decode_ref"]))
    for label, st in report.get("extra_M_sweep", {}).items():
        metrics.update(wl.flatten_numeric(f"sweep_{label}", st))
    metrics.update(wl.flatten_numeric("verdict", report["verdict"]))
    metrics["peak_vram_gb"] = report["peak_vram_gb"]
    metrics["analysis_only"] = 1
    metrics["official_tps"] = 0
    wl.log_event(run, "flip_rate_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["pr616_verdict"] = report["verdict"]["pr616_verdict"]
    run.summary["flip_rate"] = report["verdict"]["flip_rate"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr616_flip_report", artifact_type="flip-rate", data=report)
    wl.finish_wandb(run)
    print("[flip] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
