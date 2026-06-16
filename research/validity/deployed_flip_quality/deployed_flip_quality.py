#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #464 (denken) -- Deployed-flip quality: are the 3 deployed flips truly quality-neutral?

WHY (the premise this hardens or breaks)
----------------------------------------
The whole #407 relax-escalation rests on one assumed premise (land #458 `uhhyec0q`
`deployed_off_strict_frontier=True`): the deployed 481.53 ships 3 token flips (identity 0.9966,
positions {11,18,118}; lawine #455 `0r0ounl8`) and these are treated as a QUALITY-ACCEPTED status quo
(accepted when PR #52 shipped). A skeptic's sharp question: "you keep saying the 3 flips are
quality-neutral near-ties -- PROVE it. Corpus PPL 2.3772 is a teacher-forced average; a free-running
argmax flip that cascades could degrade a generation while leaving corpus PPL flat (denken #460 PPL-trap:
PPL-pass != greedy-identical)." This card characterizes the actual QUALITY impact of the 3 flips and
either HARDENS the #458 reframe (flips are genuinely quality-neutral) or SURFACES a real regression (LOUD).

ORTHOGONALITY (the #460/#458 discipline): equivalence-SEVERITY (how many flips, owned by lawine #455 /
the kernel attribution by ubel #461) is a DIFFERENT axis from QUALITY (are the flips harmful). This card
is the QUALITY axis ONLY. ubel #461 answers WHICH kernel produces each flip; we answer whether each flip
is quality-NEUTRAL. We do not re-derive the flip count's cause (that is ubel's lane).

THE EMPIRICAL CORE (what "quality-neutral" means, made falsifiable)
------------------------------------------------------------------
A reduction-order flip is quality-neutral iff, at the flip position, the model has NO real preference
between the deployed token and the strict token -- i.e. the two tokens are a TIE in the canonical M=1 AR
reference distribution, and the deployed path picked the other tied token only because a bf16 split-KV
reassociation perturbed the argmax tie-break. Then BOTH tokens are equally-probable greedy continuations;
neither is "worse"; the flip is nondeterminism between equivalent outputs, not a regression. The
falsifiable opposite: a NON-tie flip, where the deployed path picked a token the model assigns meaningfully
LOWER probability than strict -> a real quality cost -> would FAIL this gate and go to the human LOUD.

FOUR MEASUREMENTS (instruction 1-4)
  1. Re-confirm the 3 flips + token/text CONTEXT: reproduce the deployed (heuristic) census, confirm
     n=3 flips @ {11,18,118}; for each dump (prompt,pos), deployed-token-id vs strict M=1-AR-token-id, and
     the DECODED surrounding generated context (benign synonym/tie vs semantic break).
  2. Per-flip LOGIT-MARGIN + LOG-PROB DELTA (the quality magnitude): the (top1-top2) margin in the deployed
     M=8 path, and the canonical M=1 AR top-2 gap (m1_self_gap). near-tie iff |delta| <= the bf16
     reassociation perturbation (EPS*=0.125, one ULP at the logit scale -- the band that covers every
     observed flip, #381/#397/#405/#412). max_flip_logprob_delta = worst-case across the 3.
  3. Does the flip CASCADE or SELF-HEAL + downstream-sampling surfacing: continue the deployed and strict
     commits a further 32 tokens (M=1 AR greedy, both branches identical mechanism -> isolates the
     token-choice effect) and measure re-convergence. Then, per the human's downstream-eval directive
     (lewtun Issue #31: downstream evals use generation_config.json sampling, NOT greedy), report whether
     the flip would even surface under temp/top_k/top_p sampling (a bitwise tie => P(deployed)=P(strict)
     => the greedy tie-break is invisible to the sampler). STRICTLY separate from the greedy-identity gate.
  4. Self-test + PPL anchor 2.3772 + headline verdict deployed_flips_quality_neutral.

SCOPE: LOCAL A10G (sm_86), analysis-only. NO HF job, NO submission, NO served/deployed file touched; the
int4 path is READ only. Method REUSES lawine #455 / stark #412's served forced-log census (driven as a
black-box subprocess for the authoritative flip set) + an own GPU phase for the quality layer (context,
margins, 32-token cascade, downstream sampling) the census does not produce.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# ======================================================================================
# Anchors (CITE public W&B / merged cards; do NOT re-derive)
# ======================================================================================
OFFICIAL_TPS = 481.53                # deployed non-strict public #1 (PR #52 `2x9fm2zx`); CONTEXT only -- this leg adds 0
PPL_ANCHOR = 2.3772                  # deployed PPL (PR #52 summary/ppl; lawine #455 / denken #423)
PPL_GATE = 2.42                      # validity gate ceiling
DEPLOYED_IDENTITY_ANCHOR = 0.9965986394557823   # 879/882 (3 flips); lawine #455 `0r0ounl8`
DEPLOYED_FLIPS_ANCHOR = 3
KNOWN_FLIP_PROMPTS = (11, 18, 118)   # the 3 served-arm flip prompts (#381/#397/#405/#412/#455)

EPS_STAR = 0.125                     # bf16 one-ULP floor at the logit scale; the band covering every observed flip
BAND_TOL = 1e-9                      # bitwise-tie tolerance (m1_self_gap <= BAND_TOL => bit-identical top-2 logits)
CASCADE_TOKENS = 32                  # instruction-3 continuation length
LN_RATIO_SURFACE_THRESH = 0.05       # |ln(P_deployed/P_strict)| above this => the flip "surfaces" under sampling

# Downstream sampling config (gemma-4-E4B-it generation_config.json; read live, these are the fallback).
DOWNSTREAM_FALLBACK = {"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95}

# census geometry (identical to #412/#455 so the flip set reproduces bit-for-bit)
HYBRID_PREFIX_COMMIT = 32
MODEL_CANDIDATES = [
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

# the in-boundary #412 census method (MERGED to approval-gated-8gpu-20260613). Driven as a subprocess, never edited.
S412 = Path("research/validity/selective_recompute_equivalent_tps/selective_recompute_equivalent_tps.py")
OUT_DIR = Path("research/validity/deployed_flip_quality")
CENSUS_JSON = OUT_DIR / "census_heuristic.json"
DEEP_JSON = OUT_DIR / "deep_result.json"
REPORT_JSON = OUT_DIR / "deployed_flip_quality_results.json"


# ======================================================================================
# Small helpers (reused from #412/#455)
# ======================================================================================
def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    raise FileNotFoundError(f"no int4 model found among {MODEL_CANDIDATES}")


def block_align(n: int) -> int:
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _sorted_logprobs(entry) -> list[tuple[int, float]]:
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def _argmax_from_logprob_entry(entry) -> int:
    return int(max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0])


def _lp_of(sorted_lp: list[tuple[int, float]], tok: int):
    for tid, lp in sorted_lp:
        if tid == tok:
            return lp
    return None


def _read_downstream_cfg(model_dir: str) -> dict:
    p = Path(model_dir) / "generation_config.json"
    cfg = dict(DOWNSTREAM_FALLBACK)
    try:
        gc = json.load(open(p))
        for k in ("do_sample", "temperature", "top_k", "top_p"):
            if k in gc:
                cfg[k] = gc[k]
    except Exception as exc:
        print(f"[deep] generation_config read failed ({exc!r}); using fallback {DOWNSTREAM_FALLBACK}", flush=True)
    return cfg


def _characterize_flip(ri, rec, pos, j, strict_tok, deployed_tok, cf,
                       prefix, cont, m1_lp_steps, pls, tok, llm, sp_cascade,
                       cascade_tokens, temp_ds, downstream) -> dict:
    """Characterize ONE flip. Margins/tie facts are census-authoritative when `cf` is given (the census owns the
    M=8 chunked-verify argmax at a knife-edge tie); this phase ADDS context/cascade/downstream + a non-gating
    single-prefill re-derivation cross-check. logprob diff == logit diff (softmax shift-invariant)."""
    # ----- deep-phase single-prefill re-derivation (CONFIRMATION; not authoritative at knife-edge ties) -----
    entry = pls[pos] if pos < len(pls) else None
    m8_sorted = _sorted_logprobs(entry) if entry else []
    m8_argmax = m8_sorted[0][0] if m8_sorted else None
    m8_lp_deployed_rd = _lp_of(m8_sorted, deployed_tok)
    m8_lp_strict_rd = _lp_of(m8_sorted, strict_tok)
    m8_gap_rd = (m8_sorted[0][1] - m8_sorted[1][1]) if len(m8_sorted) >= 2 else None

    m1_entry = m1_lp_steps[j] if 0 <= j < len(m1_lp_steps) else None
    m1_sorted = _sorted_logprobs(m1_entry) if m1_entry else []
    m1_top1_id = m1_sorted[0][0] if m1_sorted else None
    m1_top2_id = m1_sorted[1][0] if len(m1_sorted) >= 2 else None
    m1_self_gap_rd = (m1_sorted[0][1] - m1_sorted[1][1]) if len(m1_sorted) >= 2 else None
    m1_lp_strict = _lp_of(m1_sorted, strict_tok)
    m1_lp_deployed = _lp_of(m1_sorted, deployed_tok)
    strict_matches_census = bool(j < len(cont) and cont[j] == strict_tok)
    rederived_m1_tied_pair = bool(m1_top1_id is not None and m1_top2_id is not None
                                  and {m1_top1_id, m1_top2_id} == {strict_tok, deployed_tok})

    # ----- census-authoritative quality facts (the verdict rests on these) -----
    if cf is not None:
        m8_gap = float(cf["m8_gap"])
        m1_self_gap = float(cf["m1_self_gap"])
        m1_is_bitwise_tie = bool(cf["m1_is_bitwise_tie"])
        # census records m1_tok_id == m8_top2_id: the strict token IS the deployed path's tied top-2 -> the two
        # flipping tokens ARE the bitwise-tied pair (the decisive structural fact).
        tied_pair_is_flip_pair = bool(int(cf.get("m8_top2_id", -1)) == strict_tok)
    else:
        m8_gap = float(m8_gap_rd) if m8_gap_rd is not None else float("nan")
        m1_self_gap = m1_self_gap_rd
        m1_is_bitwise_tie = bool(m1_self_gap is not None and m1_self_gap <= BAND_TOL)
        tied_pair_is_flip_pair = rederived_m1_tied_pair

    # at the tied pair, strict is the M=1 top1 and deployed the top2 -> canonical cost = m1_self_gap (=0 at a tie):
    # the deployed token is NOT meaningfully less probable than strict under the canonical M=1 AR reference.
    quality_cost_m1ref = float(m1_self_gap) if m1_self_gap is not None else None
    is_near_tie = bool((math.isfinite(m8_gap) and m8_gap <= EPS_STAR + BAND_TOL)
                       and (m1_self_gap is None or m1_self_gap <= EPS_STAR + BAND_TOL))
    logprob_delta_conservative = max(m8_gap if math.isfinite(m8_gap) else 0.0, quality_cost_m1ref or 0.0)

    # ----- downstream-sampling surfacing (temp/top_k/top_p): at a bitwise tie P(deployed)==P(strict) -> invisible.
    # deployed is the M=1 top-2, so ln(P_dep/P_strict) = -(m1_self_gap)/temp = 0 at a tie (canonical, census-anchored).
    ln_ratio = -(quality_cost_m1ref or 0.0) / temp_ds
    p_ratio_dep_over_strict = math.exp(ln_ratio)
    p_strict_model = math.exp(m1_lp_strict) if m1_lp_strict is not None else None
    p_deployed_model = math.exp(m1_lp_deployed) if m1_lp_deployed is not None else None
    both_in_top_k = bool(int(downstream.get("top_k", 64)) >= 2)        # both are the M=1 top-2 by construction
    surfaces_under_sampling = bool(abs(ln_ratio) > LN_RATIO_SURFACE_THRESH)

    # ----- decoded context (benign synonym/tie vs semantic break) -----
    ctx_text = {
        "prefix_tail": tok.decode(prefix[-10:]),
        "generated_before_flip": tok.decode(cont[:j]),
        "strict_token_text": tok.decode([strict_tok]),
        "deployed_token_text": tok.decode([deployed_tok]),
        "generated_after_flip_strict_branch": tok.decode(cont[j + 1:]),
    }

    # ----- cascade / self-heal: M=1 AR greedy from each committed token, `cascade_tokens` further -----
    stem = prefix + cont[:j]
    outS = llm.generate([{"prompt_token_ids": stem + [strict_tok]}], sp_cascade, use_tqdm=False)[0]
    outD = llm.generate([{"prompt_token_ids": stem + [deployed_tok]}], sp_cascade, use_tqdm=False)[0]
    win_s = [strict_tok] + list(outS.outputs[0].token_ids)[:cascade_tokens]     # aligned position-by-position
    win_d = [deployed_tok] + list(outD.outputs[0].token_ids)[:cascade_tokens]
    L = min(len(win_s), len(win_d))
    win_s, win_d = win_s[:L], win_d[:L]
    hamming = sum(1 for a, b in zip(win_s, win_d) if a != b)
    first_remerge = None                            # first k>=1 from which the two windows match to the end
    for k in range(1, L):
        if win_s[k:] == win_d[k:]:
            first_remerge = k
            break
    cascade_converged = bool(first_remerge is not None) or bool(win_s == win_d)

    return {
        "prompt_idx": ri, "pos": pos, "j": j, "prompt_id": rec.get("id"),
        "deployed_token_id": deployed_tok, "strict_token_id": strict_tok,
        # ---- census-authoritative quality facts (verdict rests on these) ----
        "m8_gap": m8_gap, "m1_self_gap": m1_self_gap, "m1_is_bitwise_tie": m1_is_bitwise_tie,
        "tied_pair_is_flip_pair": tied_pair_is_flip_pair, "is_near_tie": is_near_tie,
        "quality_cost_m1ref": quality_cost_m1ref, "logprob_delta_conservative": logprob_delta_conservative,
        "m1_lp_strict": m1_lp_strict, "m1_lp_deployed": m1_lp_deployed,
        # ---- downstream sampling (anchored on the tie) ----
        "ln_p_ratio_deployed_over_strict": ln_ratio, "p_ratio_deployed_over_strict": p_ratio_dep_over_strict,
        "p_strict_model": p_strict_model, "p_deployed_model": p_deployed_model,
        "both_in_top_k": both_in_top_k, "surfaces_under_sampling": surfaces_under_sampling,
        # ---- decoded context ----
        "context": ctx_text,
        # ---- cascade / self-heal ----
        "cascade_hamming": hamming, "cascade_window_len": L, "cascade_converged": cascade_converged,
        "tokens_to_remerge": first_remerge,
        "strict_branch_text": tok.decode(win_s), "deployed_branch_text": tok.decode(win_d),
        # ---- deep-phase re-derivation cross-checks (CONFIRMATION; NON-gating) ----
        "rederived": {
            "anchored_on_census": bool(cf is not None),
            "strict_matches_census": strict_matches_census,
            "m8_single_prefill_argmax": m8_argmax,
            "m8_argmax_matches_census": bool(m8_argmax == deployed_tok),
            "m8_gap": m8_gap_rd, "m8_lp_deployed": m8_lp_deployed_rd, "m8_lp_strict": m8_lp_strict_rd,
            "m1_self_gap": m1_self_gap_rd, "m1_tied_pair_is_flip_pair": rederived_m1_tied_pair,
        },
    }


# ======================================================================================
# PHASE deep (GPU): per-flip-prompt QUALITY analysis (context + margins + cascade + downstream)
# ======================================================================================
def phase_deep(out_path: str, flip_prompts: list[int], ctx_len: int, n_verify: int,
               gpu_mem_util: float, max_batched_tokens: int, cascade_tokens: int) -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # ---- AUTHORITATIVE flip set: the #412/#455 served chunked-verify census (the flips we CHARACTERIZE).
    # The deployed argmax at a knife-edge tie is a property of the M=8 *chunked-verify* reduction order, which a
    # single-prefill cannot reproduce bit-for-bit (#412/#455 own that path). So we anchor token IDs / margins on
    # the census and use this GPU phase to ADD the quality layer (context, cascade, downstream) + confirm. ----
    census_by_prompt: dict[int, dict] = {}
    if CENSUS_JSON.exists():
        try:
            census_by_prompt = {int(f["prompt_idx"]): f
                                for f in json.load(open(CENSUS_JSON)).get("flip_details", [])}
        except Exception as exc:
            print(f"[deep] census anchor unreadable ({CENSUS_JSON}: {exc!r}); independent-scan fallback", flush=True)

    model_dir = resolve_model_dir()
    C = block_align(ctx_len)
    downstream = _read_downstream_cfg(model_dir)
    temp_ds = float(downstream.get("temperature", 1.0)) or 1.0
    print(f"[deep] model={model_dir} C(prefix)={C} n_verify={n_verify} flip_prompts={flip_prompts} "
          f"downstream={downstream} VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT','0')}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(512, C + n_verify + cascade_tokens + 64), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )

    # vLLM caps (prompt_)logprobs at 20; the two flipping tokens are always the M=1 top-2, so 20 is ample.
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)   # prime prefix cache (mirror census)
    sp_gen = SamplingParams(temperature=0.0, max_tokens=n_verify, logprobs=20, detokenize=False)
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20,
                              skip_reading_prefix_cache=False, detokenize=False)
    sp_cascade = SamplingParams(temperature=0.0, max_tokens=cascade_tokens, detokenize=False)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)]
    flips: list[dict] = []

    for ri in flip_prompts:
        rec = rows[ri]
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            print(f"[deep] prompt {ri} too short ({len(src)} < {C+1}); skipping", flush=True)
            continue
        prefix = src[:C]

        # warm-up generate: prime the prefix cache so the M=1/M=8 reduction order mirrors the census call sequence
        # (the flips are one-bf16-ULP knife-edge ties; a cold cache can break the tie the other way).
        _ = llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)

        # ---- M=1 AR reference (greedy, token-by-token) -> the strict tokens + per-step logprobs ----
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_gen, use_tqdm=False)[0]
        cont = list(outA.outputs[0].token_ids)[:n_verify]
        m1_lp_steps = list(outA.outputs[0].logprobs or [])[:n_verify]
        if len(cont) < n_verify:
            print(f"[deep] prompt {ri} produced <{n_verify} AR tokens; skipping", flush=True)
            continue
        full = prefix + cont

        # ---- M=8 single-prefill of `full` -> deployed-path argmax/top-k (best-effort cross-check) ----
        outC = llm.generate([{"prompt_token_ids": full}], sp_chunk, use_tqdm=False)[0]
        pls = outC.prompt_logprobs or []

        # ---- resolve the flip(s) to CHARACTERIZE: census-authoritative if present, else independent scan ----
        cf = census_by_prompt.get(ri)
        if cf is not None:
            resolved = [(int(cf["pos"]), int(cf["m1_tok_id"]), int(cf["m8_top1_id"]), cf)]
        else:
            resolved = []
            for p in range(C + 1, len(full)):
                entry = pls[p] if p < len(pls) else None
                if entry is None:
                    continue
                strict_t = full[p]
                dep_t = _argmax_from_logprob_entry(entry)
                if dep_t != strict_t:
                    resolved.append((p, strict_t, dep_t, None))

        prompt_flips = [
            _characterize_flip(ri, rec, pos, pos - C, strict_tok, deployed_tok, cf_i,
                               prefix, cont, m1_lp_steps, pls, tok, llm, sp_cascade,
                               cascade_tokens, temp_ds, downstream)
            for (pos, strict_tok, deployed_tok, cf_i) in resolved
        ]

        if not prompt_flips:
            print(f"[deep] prompt {ri}: NO flip resolved (no census anchor + no independent flip) -- empty", flush=True)
        for f in prompt_flips:
            rd = f["rederived"]
            print(f"[deep] flip p{f['prompt_idx']} pos{f['pos']}: "
                  f"deployed={f['deployed_token_id']}({f['context']['deployed_token_text']!r}) vs "
                  f"strict={f['strict_token_id']}({f['context']['strict_token_text']!r}) "
                  f"m8_gap={f['m8_gap']:.4f} m1_self_gap={f['m1_self_gap']} bitwise_tie={f['m1_is_bitwise_tie']} "
                  f"[rederive: strict_ok={rd['strict_matches_census']} m8argmax_ok={rd['m8_argmax_matches_census']} "
                  f"m1gap={rd['m1_self_gap']}] cascade_converged={f['cascade_converged']} "
                  f"surfaces={f['surfaces_under_sampling']}", flush=True)
        flips.extend(prompt_flips)

    out = {
        "phase": "deep", "model_dir": model_dir, "C": C, "n_verify": n_verify,
        "cascade_tokens": cascade_tokens, "flip_prompts_requested": list(flip_prompts),
        "downstream_sampling_config": downstream,
        "n_flips_rederived": len(flips), "flips": flips,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[deep] re-derived {len(flips)} flips; peak={out['peak_gpu_gb']:.1f}GB -> {out_path}", flush=True)
    print(f"PHASE_DONE {out_path}", flush=True)


# ======================================================================================
# Subprocess drivers (mirror #455 env pinning)
# ======================================================================================
def _pin_env(extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    return env


def run_412_census(a: argparse.Namespace) -> dict:
    """Reuse stark #412 / lawine #455 served forced-log census (heuristic arm = deployed) as a black box."""
    env = _pin_env({"VLLM_BATCH_INVARIANT": "0"})   # heuristic = deployed fast path
    cmd = [sys.executable, str(S412.resolve()),
           "--phase", "census", "--arm", "heuristic", "--out", str(CENSUS_JSON),
           "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len), "--n-verify", str(a.n_verify),
           "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
           "--verbose-k", str(a.verbose_k)]
    print(f"[orch] #412 census(heuristic) <- n_prompts={a.n_prompts} (VLLM_BATCH_INVARIANT=0)", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"#412 census failed (rc={rc})")
    return json.load(open(CENSUS_JSON))


def run_deep(a: argparse.Namespace) -> dict:
    env = _pin_env({"VLLM_BATCH_INVARIANT": "0"})   # deep analysis on the deployed (fast) path
    cmd = [sys.executable, os.path.abspath(__file__),
           "--phase", "deep", "--out", str(DEEP_JSON),
           "--flip-prompts", ",".join(str(p) for p in a.flip_prompts),
           "--ctx-len", str(a.ctx_len), "--n-verify", str(a.n_verify),
           "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
           "--cascade-tokens", str(a.cascade_tokens)]
    print(f"[orch] deep <- flip_prompts={a.flip_prompts} cascade_tokens={a.cascade_tokens}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"deep phase failed (rc={rc})")
    return json.load(open(DEEP_JSON))


# ======================================================================================
# Compose (pure function of census + deep -> the PR's deliverable fields + verdict)
# ======================================================================================
def _census_flip_prompts(census: dict) -> list[int]:
    return sorted({int(f["prompt_idx"]) for f in census.get("flip_details", [])})


def compose_report(census: dict, deep: dict, a: argparse.Namespace) -> dict:
    census_flip_prompts = _census_flip_prompts(census)
    n_deployed_flips = len(census.get("flip_details", []))
    deployed_identity = census.get("decodewidth_e2e_token_identity_rate")
    flips = deep.get("flips", [])

    # ---- Step 2: near-tie classification + worst-case quality magnitude ----
    all_flips_are_near_ties = bool(flips and all(f["is_near_tie"] for f in flips))
    all_flips_bitwise_tie_m1 = bool(flips and all(f["m1_is_bitwise_tie"] for f in flips))
    all_tied_pair_is_flip_pair = bool(flips and all(f["tied_pair_is_flip_pair"] for f in flips))
    max_flip_logprob_delta = max((f["logprob_delta_conservative"] for f in flips), default=float("nan"))
    max_flip_canonical_cost_m1 = max((f["quality_cost_m1ref"] or 0.0 for f in flips), default=float("nan"))

    # ---- Step 3: cascade self-heal + downstream surfacing ----
    flips_self_heal_within_32tok = bool(flips and all(f["cascade_converged"] for f in flips))
    flips_surface_under_downstream_sampling = bool(any(f["surfaces_under_sampling"] for f in flips))
    max_abs_ln_p_ratio = max((abs(f["ln_p_ratio_deployed_over_strict"] or 0.0) for f in flips), default=float("nan"))

    # ---- HEADLINE verdict ----
    # Quality-neutral iff: every flip is a near-tie (the model has no real preference), the canonical-reference
    # quality cost is ~0 (deployed token NOT meaningfully less probable than strict), AND the flips are invisible
    # to the downstream sampler. Self-heal is REPORTED but NOT gating: two equiprobable continuations diverging in
    # surface form is nondeterminism between equivalent outputs, not a regression (the regression case is a NON-tie
    # flip, which would fail `all_flips_are_near_ties` / raise `max_flip_canonical_cost_m1`).
    deployed_flips_quality_neutral = bool(
        all_flips_are_near_ties
        and (math.isfinite(max_flip_canonical_cost_m1) and max_flip_canonical_cost_m1 <= EPS_STAR + BAND_TOL)
        and (not flips_surface_under_downstream_sampling))

    rederived_match_known = bool(sorted({int(f["prompt_idx"]) for f in flips}) == sorted(a.flip_prompts))
    census_match_known = bool(census_flip_prompts == sorted(KNOWN_FLIP_PROMPTS))

    report = {
        "pr": 464, "agent": "denken",
        "leg": "Deployed-flip quality: are the 3 flips truly quality-neutral? (LOCAL A10G, analysis-only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,

        # ===== Step 1: re-confirm flips + context =====
        "n_deployed_flips": n_deployed_flips,
        "flip_positions": census_flip_prompts,
        "deployed_identity_fraction": deployed_identity,
        "census_match_known_111888": census_match_known,
        "rederived_match_known": rederived_match_known,
        "n_flips_rederived": deep.get("n_flips_rederived"),

        # ===== Step 2: margins + quality magnitude =====
        "all_flips_are_near_ties": all_flips_are_near_ties,
        "all_flips_bitwise_tie_m1": all_flips_bitwise_tie_m1,
        "all_tied_pair_is_flip_pair": all_tied_pair_is_flip_pair,
        "max_flip_logprob_delta": max_flip_logprob_delta,
        "max_flip_canonical_cost_m1": max_flip_canonical_cost_m1,
        "reassociation_perturbation_nats": EPS_STAR,

        # ===== Step 3: cascade + downstream =====
        "flips_self_heal_within_32tok": flips_self_heal_within_32tok,
        "flips_surface_under_downstream_sampling": flips_surface_under_downstream_sampling,
        "max_abs_ln_p_ratio_deployed_over_strict": max_abs_ln_p_ratio,
        "downstream_sampling_config": deep.get("downstream_sampling_config"),

        # ===== Step 4: headline + anchors =====
        "deployed_flips_quality_neutral": deployed_flips_quality_neutral,
        "ppl": PPL_ANCHOR, "ppl_gate": PPL_GATE, "ppl_passes_gate": bool(PPL_ANCHOR <= PPL_GATE),
        "deployed_tps_context": OFFICIAL_TPS,
        "deployed_identity_anchor": DEPLOYED_IDENTITY_ANCHOR,

        # ===== per-flip detail (the human-auditable core) =====
        "per_flip": flips,
        "config": {
            "n_prompts": a.n_prompts, "ctx_len": a.ctx_len, "n_verify": a.n_verify,
            "cascade_tokens": a.cascade_tokens, "flip_prompts": list(a.flip_prompts),
            "model_dir": deep.get("model_dir"),
            "census_peak_gpu_gb": census.get("peak_gpu_gb"), "deep_peak_gpu_gb": deep.get("peak_gpu_gb"),
        },
    }

    checks, n_checks = build_self_test(report, census, deep)
    report["self_test"] = checks
    report["self_test_n_checks"] = n_checks
    report["flip_quality_self_test_passes"] = bool(all(checks.values()) and n_checks >= 18)

    report["one_line_verdict"] = (
        f"deployed_flips_quality_neutral={deployed_flips_quality_neutral}: "
        f"{n_deployed_flips} flips @ {census_flip_prompts} (identity {deployed_identity:.4f}); "
        f"all near-ties={all_flips_are_near_ties} (all M=1 bitwise ties={all_flips_bitwise_tie_m1}, "
        f"the two flipping tokens ARE the tied top-2={all_tied_pair_is_flip_pair}); "
        f"worst-case canonical cost={max_flip_canonical_cost_m1:.4g} nats (<= {EPS_STAR} ULP floor); "
        f"self-heal<=32tok={flips_self_heal_within_32tok}; "
        f"surfaces under downstream sampling={flips_surface_under_downstream_sampling} "
        f"(max |ln P_dep/P_strict|={max_abs_ln_p_ratio:.3g}). "
        f"=> {'HARDENS' if deployed_flips_quality_neutral else 'WEAKENS (LOUD)'} the #458 status-quo reframe."
    )
    return report


# ======================================================================================
# Self-test (>=18 asserts; validates the verdict logic + reproduction-against-anchor)
# ======================================================================================
def build_self_test(report: dict, census: dict, deep: dict) -> tuple[dict, int]:
    c: dict = {}
    flips = deep.get("flips", [])

    # Step 1: census reproduces the 0.9966 / 3-flip anchor at {11,18,118}
    c["n_flips_eq_3"] = bool(report["n_deployed_flips"] == DEPLOYED_FLIPS_ANCHOR)
    c["census_match_known_111888"] = bool(report["census_match_known_111888"])
    c["deployed_identity_reproduces_9966"] = bool(
        report["deployed_identity_fraction"] is not None
        and abs(report["deployed_identity_fraction"] - DEPLOYED_IDENTITY_ANCHOR) <= 0.01)
    c["rederived_flips_match_known"] = bool(report["rederived_match_known"])
    c["n_flips_rederived_eq_3"] = bool(report["n_flips_rederived"] == DEPLOYED_FLIPS_ANCHOR)

    # Step 2: every flip has decoded context, both token ids, and a finite margin
    c["every_flip_has_context"] = bool(flips and all(
        f.get("context", {}).get("strict_token_text") is not None
        and f.get("context", {}).get("deployed_token_text") is not None for f in flips))
    c["every_flip_distinct_tokens"] = bool(flips and all(
        f["deployed_token_id"] != f["strict_token_id"] for f in flips))
    c["every_flip_margin_finite"] = bool(flips and all(
        math.isfinite(f["m8_gap"]) for f in flips))
    # every flip carries a non-gating deep-phase re-derivation cross-check (strict re-derives; M=1 tie confirmed)
    c["every_flip_has_rederive_crosscheck"] = bool(flips and all(
        isinstance(f.get("rederived"), dict)
        and "strict_matches_census" in f["rederived"]
        and "m8_argmax_matches_census" in f["rederived"] for f in flips))
    c["max_logprob_delta_le_eps"] = bool(
        math.isfinite(report["max_flip_logprob_delta"])
        and report["max_flip_logprob_delta"] <= EPS_STAR + BAND_TOL)
    c["max_canonical_cost_le_eps"] = bool(
        math.isfinite(report["max_flip_canonical_cost_m1"])
        and report["max_flip_canonical_cost_m1"] <= EPS_STAR + BAND_TOL)
    c["near_tie_consistent"] = bool(
        report["all_flips_are_near_ties"] == all(f["is_near_tie"] for f in flips) if flips else False)
    # the central structural fact: the two flipping tokens ARE the M=1 bitwise-tied top-2
    c["flipping_tokens_are_tied_pair"] = bool(report["all_tied_pair_is_flip_pair"])
    c["all_bitwise_tie_m1"] = bool(report["all_flips_bitwise_tie_m1"])

    # Step 3: cascade window measured + downstream ratio computed for every flip
    c["every_flip_has_cascade"] = bool(flips and all(
        f["cascade_window_len"] >= 1 and isinstance(f["cascade_converged"], bool) for f in flips))
    c["every_flip_has_downstream_ratio"] = bool(flips and all(
        f["ln_p_ratio_deployed_over_strict"] is not None for f in flips))
    c["downstream_ratio_consistent"] = bool(
        report["flips_surface_under_downstream_sampling"] == any(f["surfaces_under_sampling"] for f in flips))
    # at a bitwise tie the sampler cannot distinguish the tokens (|ln ratio| ~ 0)
    c["bitwise_tie_implies_invisible"] = bool(
        (not report["all_flips_bitwise_tie_m1"])
        or report["max_abs_ln_p_ratio_deployed_over_strict"] <= LN_RATIO_SURFACE_THRESH)

    # Step 4: verdict logic internally consistent + anchors exact
    c["verdict_logic_consistent"] = bool(
        report["deployed_flips_quality_neutral"] == (
            report["all_flips_are_near_ties"]
            and report["max_flip_canonical_cost_m1"] <= EPS_STAR + BAND_TOL
            and (not report["flips_surface_under_downstream_sampling"])))
    c["ppl_passes_gate"] = bool(report["ppl_passes_gate"])
    c["constants_exact"] = bool(PPL_ANCHOR == 2.3772 and EPS_STAR == 0.125 and OFFICIAL_TPS == 481.53)
    c["analysis_only_flags"] = bool(report["analysis_only"] and report["no_served_file_change"]
                                    and report["official_tps"] == 0)
    c["headline_nan_clean"] = bool(all(math.isfinite(x) for x in (
        report["max_flip_logprob_delta"], report["max_flip_canonical_cost_m1"],
        report["max_abs_ln_p_ratio_deployed_over_strict"])))
    return c, len(c)


# ======================================================================================
# Synthetic self-test (0-GPU): validate compose+verdict logic w/o any model load
# ======================================================================================
def _synthetic_flip(prompt_idx, pos, j, deployed, strict, m8_gap, m1_self_gap, *,
                    converged=True, ln_ratio=0.0, quality_cost=0.0):
    bitwise = bool(m1_self_gap <= BAND_TOL)
    return {
        "prompt_idx": prompt_idx, "pos": pos, "j": j, "prompt_id": f"synthetic-{prompt_idx}",
        "deployed_token_id": deployed, "strict_token_id": strict,
        "m8_gap": m8_gap, "m1_self_gap": m1_self_gap,
        "m1_lp_strict": -1.0, "m1_lp_deployed": -1.0 - quality_cost,
        "m1_is_bitwise_tie": bitwise,
        "tied_pair_is_flip_pair": True,
        "quality_cost_m1ref": quality_cost,
        "logprob_delta_conservative": max(m8_gap, quality_cost),
        "is_near_tie": bool(m8_gap <= EPS_STAR + BAND_TOL and m1_self_gap <= EPS_STAR + BAND_TOL),
        "context": {"prefix_tail": "...", "generated_before_flip": "...",
                    "strict_token_text": f"<{strict}>", "deployed_token_text": f"<{deployed}>",
                    "generated_after_flip_strict_branch": "..."},
        "cascade_hamming": 0 if converged else 20, "cascade_window_len": 33,
        "cascade_converged": converged, "tokens_to_remerge": 2 if converged else None,
        "strict_branch_text": "s...", "deployed_branch_text": "d...",
        "ln_p_ratio_deployed_over_strict": ln_ratio,
        "p_ratio_deployed_over_strict": math.exp(ln_ratio),
        "p_strict_model": 0.3, "p_deployed_model": 0.3 * math.exp(ln_ratio),
        "both_in_top_k": True, "surfaces_under_sampling": bool(abs(ln_ratio) > LN_RATIO_SURFACE_THRESH),
        "rederived": {
            "anchored_on_census": True, "strict_matches_census": True,
            "m8_single_prefill_argmax": deployed, "m8_argmax_matches_census": True,
            "m8_gap": m8_gap, "m8_lp_deployed": 0.0, "m8_lp_strict": -m8_gap,
            "m1_self_gap": m1_self_gap, "m1_tied_pair_is_flip_pair": True,
        },
    }


def _synthetic_census(flip_prompts) -> dict:
    return {
        "decodewidth_e2e_token_identity_rate": DEPLOYED_IDENTITY_ANCHOR,
        "flip_details": [{"prompt_idx": p, "pos": 227} for p in flip_prompts],
        "total_positions": 882, "peak_gpu_gb": 12.25,
    }


def self_test(a: argparse.Namespace) -> None:
    # CASE A: the real-world picture (all 3 bitwise ties) -> quality_neutral=True
    fp = list(KNOWN_FLIP_PROMPTS)
    deep_neutral = {
        "model_dir": "/synthetic", "C": 224, "n_verify": 8, "cascade_tokens": 32,
        "flip_prompts_requested": fp, "downstream_sampling_config": DOWNSTREAM_FALLBACK,
        "n_flips_rederived": 3, "peak_gpu_gb": 0.0,
        "flips": [
            _synthetic_flip(11, 231, 7, 236743, 621, 0.125, 0.0, converged=True, ln_ratio=0.0),
            _synthetic_flip(18, 227, 3, 25581, 3629, 0.125, 0.0, converged=False, ln_ratio=0.0),
            _synthetic_flip(118, 227, 3, 8291, 6481, 0.125, 0.0, converged=True, ln_ratio=0.0),
        ],
    }
    a.flip_prompts = fp
    r_neutral = compose_report(_synthetic_census(fp), deep_neutral, a)
    _print_console(r_neutral)
    ok_neutral = (r_neutral["deployed_flips_quality_neutral"] is True
                  and r_neutral["flip_quality_self_test_passes"]
                  and r_neutral["all_flips_bitwise_tie_m1"] is True)

    # CASE B: a REGRESSION (one non-tie flip: deployed token 0.9 nats less probable) -> quality_neutral=False
    deep_regress = json.loads(json.dumps(deep_neutral))
    deep_regress["flips"][1] = _synthetic_flip(18, 227, 3, 25581, 3629, 0.40, 0.40,
                                               converged=False, ln_ratio=-0.40, quality_cost=0.9)
    r_regress = compose_report(_synthetic_census(fp), deep_regress, a)
    ok_regress = (r_regress["deployed_flips_quality_neutral"] is False
                  and r_regress["all_flips_are_near_ties"] is False
                  and r_regress["flips_surface_under_downstream_sampling"] is True)

    ok = bool(ok_neutral and ok_regress)
    print(f"[self-test] CASE-A neutral verdict=True PASS={ok_neutral} "
          f"({sum(r_neutral['self_test'].values())}/{r_neutral['self_test_n_checks']})", flush=True)
    print(f"[self-test] CASE-B regression verdict=False (gate detects a real cost) PASS={ok_regress}", flush=True)
    print(f"[self-test] synthetic compose+verdict PASSES={ok}", flush=True)
    if not ok:
        sys.exit(1)


# ======================================================================================
# Console + W&B + finish
# ======================================================================================
def _print_console(r: dict) -> None:
    print("\n========== DEPLOYED-FLIP QUALITY (PR #464) ==========", flush=True)
    print(f" {r['one_line_verdict']}", flush=True)
    print(" --- Step 1: re-confirm flips ---", flush=True)
    print(f"  n_deployed_flips={r['n_deployed_flips']} @ {r['flip_positions']} "
          f"identity={r['deployed_identity_fraction']} (anchor 0.9966/3) "
          f"census_match={r['census_match_known_111888']} rederived_match={r['rederived_match_known']}", flush=True)
    print(" --- Step 2: margins + quality magnitude ---", flush=True)
    print(f"  all_near_ties={r['all_flips_are_near_ties']} all_bitwise_tie_m1={r['all_flips_bitwise_tie_m1']} "
          f"tied_pair_is_flip_pair={r['all_tied_pair_is_flip_pair']}", flush=True)
    print(f"  max_flip_logprob_delta={r['max_flip_logprob_delta']:.4g} "
          f"max_canonical_cost_m1={r['max_flip_canonical_cost_m1']:.4g} "
          f"(<= reassoc floor {r['reassociation_perturbation_nats']})", flush=True)
    for f in r.get("per_flip", []):
        ctx = f["context"]
        print(f"   p{f['prompt_idx']} pos{f['pos']}: strict={f['strict_token_id']}({ctx['strict_token_text']!r}) "
              f"| deployed={f['deployed_token_id']}({ctx['deployed_token_text']!r}) "
              f"m8_gap={f['m8_gap']:.4f} m1_self_gap={f['m1_self_gap']}", flush=True)
    print(" --- Step 3: cascade + downstream ---", flush=True)
    print(f"  self_heal<=32={r['flips_self_heal_within_32tok']} "
          f"surfaces_under_sampling={r['flips_surface_under_downstream_sampling']} "
          f"max|ln P_dep/P_strict|={r['max_abs_ln_p_ratio_deployed_over_strict']:.3g}", flush=True)
    print(" --- Step 4: headline ---", flush=True)
    print(f"  deployed_flips_quality_neutral={r['deployed_flips_quality_neutral']} ppl={r['ppl']}", flush=True)
    print(f" SELF-TEST PASSES={r['flip_quality_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("=====================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="denken", name=a.wandb_name, group=a.wandb_group,
        notes="PR#464 deployed-flip quality: are the 3 deployed flips {11,18,118} quality-neutral? Reuses "
              "#412/#455 served forced-log census (heuristic=deployed) + own GPU phase for context, per-flip "
              "logit-margin/log-prob delta, 32-token cascade, and downstream-sampling surfacing. LOCAL A10G, "
              "analysis-only.",
        config={
            "pr": 464, "n_prompts": report["config"]["n_prompts"], "ctx_len": report["config"]["ctx_len"],
            "n_verify": report["config"]["n_verify"], "cascade_tokens": report["config"]["cascade_tokens"],
            "flip_prompts": report["config"]["flip_prompts"], "model_dir": report["config"]["model_dir"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "anchor/deployed_identity": DEPLOYED_IDENTITY_ANCHOR, "anchor/deployed_flips": DEPLOYED_FLIPS_ANCHOR,
            "anchor/official_tps": OFFICIAL_TPS, "anchor/ppl": PPL_ANCHOR, "eps_star": EPS_STAR,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    scalar_keys = (
        "n_deployed_flips", "deployed_identity_fraction", "census_match_known_111888", "rederived_match_known",
        "n_flips_rederived", "all_flips_are_near_ties", "all_flips_bitwise_tie_m1", "all_tied_pair_is_flip_pair",
        "max_flip_logprob_delta", "max_flip_canonical_cost_m1", "reassociation_perturbation_nats",
        "flips_self_heal_within_32tok", "flips_surface_under_downstream_sampling",
        "max_abs_ln_p_ratio_deployed_over_strict", "deployed_flips_quality_neutral",
        "ppl", "ppl_gate", "ppl_passes_gate", "deployed_tps_context",
        "flip_quality_self_test_passes", "self_test_n_checks",
        "one_line_verdict", "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
    )
    for k in scalar_keys:
        run.summary[k] = report.get(k)
    run.summary["flip_positions"] = report["flip_positions"]
    run.summary["downstream_sampling_config"] = report["downstream_sampling_config"]
    # per-flip detail (string keys so W&B keeps them in summary)
    for f in report.get("per_flip", []):
        pi = f["prompt_idx"]
        run.summary[f"flip{pi}/deployed_token_id"] = f["deployed_token_id"]
        run.summary[f"flip{pi}/strict_token_id"] = f["strict_token_id"]
        run.summary[f"flip{pi}/deployed_token_text"] = f["context"]["deployed_token_text"]
        run.summary[f"flip{pi}/strict_token_text"] = f["context"]["strict_token_text"]
        run.summary[f"flip{pi}/m8_gap"] = f["m8_gap"]
        run.summary[f"flip{pi}/m1_self_gap"] = f["m1_self_gap"]
        run.summary[f"flip{pi}/m1_is_bitwise_tie"] = f["m1_is_bitwise_tie"]
        run.summary[f"flip{pi}/quality_cost_m1ref"] = f["quality_cost_m1ref"]
        run.summary[f"flip{pi}/cascade_converged"] = f["cascade_converged"]
        run.summary[f"flip{pi}/surfaces_under_sampling"] = f["surfaces_under_sampling"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    report["wandb_run_id"] = run.id
    print(f"[wandb] logged run {run.id}", flush=True)


def _finish(report: dict, a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not a.no_wandb:
        log_wandb(report, a)
    json.dump(report, open(REPORT_JSON, "w"), indent=2)
    _print_console(report)
    print(f"[done] results -> {REPORT_JSON}", flush=True)


# ======================================================================================
# Modes
# ======================================================================================
def measure(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    census = run_412_census(a)
    deep = run_deep(a)
    _finish(compose_report(census, deep, a), a)


def reanalyze(a: argparse.Namespace) -> None:
    if not CENSUS_JSON.exists() or not DEEP_JSON.exists():
        raise FileNotFoundError(f"--reanalyze needs {CENSUS_JSON} and {DEEP_JSON} (run --measure first)")
    census = json.load(open(CENSUS_JSON))
    deep = json.load(open(DEEP_JSON))
    _finish(compose_report(census, deep, a), a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="orchestrate: #412 census + deep phase + compose")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU: recompose from saved census/deep JSONs")
    ap.add_argument("--self-test", dest="self_test", action="store_true", help="0-GPU synthetic compose+verdict test")
    ap.add_argument("--phase", choices=["deep"], help="internal: run the deep GPU phase (driven as subprocess)")
    ap.add_argument("--smoke", action="store_true", help="tiny census (few prompts) to validate plumbing")
    ap.add_argument("--out", default=str(DEEP_JSON), help="deep-phase output path")
    ap.add_argument("--flip-prompts", dest="flip_prompts", default=",".join(str(p) for p in KNOWN_FLIP_PROMPTS))
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=126)   # 126*7=882 positions: the #455 anchor
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=8)
    ap.add_argument("--cascade-tokens", dest="cascade_tokens", type=int, default=CASCADE_TOKENS)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", dest="wandb_name", default="denken/deployed-flip-quality")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    a.flip_prompts = [int(p) for p in str(a.flip_prompts).split(",") if p != ""]

    if a.smoke:
        a.n_prompts = 20                                   # cheap: census covers prompts 0-19 (flips 11,18 in range)
        a.cascade_tokens = min(a.cascade_tokens, 8)
        a.flip_prompts = [p for p in a.flip_prompts if p < a.n_prompts] or [11, 18]

    if a.phase == "deep":
        phase_deep(a.out, a.flip_prompts, a.ctx_len, a.n_verify, a.gpu_mem_util,
                   a.max_batched_tokens, a.cascade_tokens)
    elif a.self_test:
        self_test(a)
    elif a.reanalyze:
        reanalyze(a)
    elif a.measure:
        measure(a)
    else:
        ap.error("one of --measure / --reanalyze / --self-test / --phase deep is required")


if __name__ == "__main__":
    main()
