#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #642 (stark) -- served-path de-teacher-forced identity confirmation (make-or-break, item #3).

WHY THIS, NOT #636's SCAN
-------------------------
#636 measured rescued_break_rate=0 along an OFFLINE vllm.LLM M=1 AR trajectory R (16/14035 flips,
all gap<=0.25 nat, tau_flag=0.5 -> 0 leaks). But BASELINE.md line 10: the M=1 AR reference MUST be
served -- offline AR diverges ~20% of prompts from the served reference (FP-reduction
nondeterminism, wirbel #8). land #632 ran the strict greedy-identity gate on the SERVED Option-B
BI=1 spec lane and found it forks from served M=1 AR on ~84% of prompts (served-compounded, PR #576
per-step identity 0.99638 -> 0.362%/pos -> 81% seq). So #636's break_rate=0 is TEACHER-FORCED on the
wrong (offline) trajectory; the advisor escalated item #3 to make-or-break.

THE DE-TEACHER-FORCE
--------------------
Walk the VALIDATED #381/#622/#636 chunk-read size_m=M_VERIFY verify scan along the SERVED M=1 AR
trajectory R_served (read from a SENPAI_REFERENCE_MODE=1 served decode jsonl) on the SERVED speed
prompts -- the ONLY change vs #636 is the trajectory (served, not offline). The offline chunk-read
M=8 itself is bit-exact to the deployed verify width under BI=1 (#381/#622: aten_mm + attn both
batch-invariant), so it faithfully reconstructs the served M=8 verify numerics; the served part that
#636 got wrong was the TRAJECTORY, which we now seed from the served decode. Per tau_flag over ALL
positions:
  flag_trigger_rate(tau)  = P(gap_Mverify < tau)            -- recompute frequency / TPS cost driver
  rescued_break_rate(tau) = P(flip AND gap_Mverify >= tau)  -- the flag MISSES; target 0
  unrescued_break_rate    = P(flip = m8_arg != R_served tok) -- served per-position fork rate
SOUNDNESS (induction, #636): rescued_break_rate==0 along R_served => the acceptor's free-running
served stream is byte-identical to served M=1 AR at every position => served break_rate=0. (At
gap<tau the acceptor recomputes M=1 == served AR token; at gap>=tau the 0-leak fact gives
m8_arg==M1==served AR token. The served stream never leaves R_served, so the scan-along-R_served
covers exactly the positions the acceptor visits.)

CROSS-CHECK (pure-python, no GPU): the served cascade. Compare the served un-rescued spec stream
S_served (a K-spec served decode jsonl) to R_served per prompt -> served per-prompt divergence
(expect ~84%, land #632) + per-token unrescued served break rate (expect ~0.3-0.4%, ABOVE the 0.114%
teacher-forced -- the gap the de-teacher-force closes). Confirms the offline scan's fork rate lines
up with the real served cascade.

CONTROL (subset): regenerate R_offline via this same vllm.LLM and byte-compare to R_served ->
reproduces BASELINE.md's offline-vs-served ~20% divergence, i.e. the exact reason #636's offline
trajectory under-counted.

SCOPE: local A10G, analysis_only=true, official_tps=0, NO HF Job / NO submission / NO served-file
change. vLLM 0.22.0, BI=1 both sides, int4 W4A16 body google/gemma-4-E4B-it-qat-w4a16-ct. The served
files are READ, never modified. The MTP drafter is NOT loaded (greedy temp=0 => drafter changes
acceptance/speed only, never the verify argmax, #621).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---- imported fleet anchors (DO NOT re-derive) ----
STARK_636_TF_BREAK_RATE = 0.0011400071250445315   # #636 teacher-forced unrescued (16/14035)
STARK_636_TF_FLIP_GAP_MAX = 0.25                  # #636 teacher-forced max flip gap (nat)
LAND_632_SERVED_SEQ_DIVERGENCE = 0.84             # land #632 served per-prompt divergence
PR576_SERVED_PERSTEP_IDENTITY = 0.99638           # PR #576 served matched-state per-step identity
LOCKED_319_AR_TPS = 126.378

HYBRID_PREFIX_COMMIT = 32
PROMPT_LOGPROBS_K = 20
TAU_FLAG_SWEEP = (0.2, 0.25, 0.3, 0.5, 0.75, 1.0)
GAP_HIST_EDGES = (0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                  0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))

DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
]
OUT_DIR = Path("research/validity/optionb_rescue_deproject/served_identity")


def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    return DEFAULT_MODEL


def _lmhead_is_int4(cfg: dict):
    qc = cfg.get("quantization_config") or {}
    groups = qc.get("config_groups") or {}
    for g in groups.values():
        if any("lm_head" in str(t) for t in (g.get("targets") or [])):
            return True
    ign = qc.get("ignore") or []
    if qc and not any("lm_head" in str(t) for t in ign) and groups:
        return None
    return False


def read_dims(model_dir: str) -> dict:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return {"hidden": 2560, "num_layers": None, "lmhead_quant": False}
    cfg = json.load(open(cfg_path))
    tc = cfg.get("text_config", cfg)
    return {"hidden": tc["hidden_size"], "num_layers": tc.get("num_hidden_layers"),
            "lmhead_quant": _lmhead_is_int4(cfg)}


def block_floor(n: int) -> int:
    return (n // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def block_ceil(n: int) -> int:
    return ((n + HYBRID_PREFIX_COMMIT - 1) // HYBRID_PREFIX_COMMIT) * HYBRID_PREFIX_COMMIT


def _sorted_logprobs(entry) -> list:
    return sorted(((int(t), float(getattr(lp, "logprob", lp))) for t, lp in entry.items()),
                  key=lambda kv: kv[1], reverse=True)


def _wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, center - half), min(1.0, center + half))


def _rule_of_three_ub(n: int) -> float:
    return (3.0 / n) if n > 0 else float("nan")


def load_ref_records(ref_jsonl: str, n_prompts: int) -> list:
    rows = []
    for line in open(ref_jsonl):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        P = list(r.get("prompt_token_ids") or [])
        R = list(r.get("completion_token_ids") or [])
        if P and R:
            rows.append({"id": r.get("id"), "P": P, "R": R})
        if len(rows) >= n_prompts:
            break
    return rows


# ======================================================================================
# GPU PHASE: chunk-read M=Mverify verify scan ALONG the served R_served trajectory
# ======================================================================================
def phase_scan(ref_jsonl: str, out_path: str, n_prompts: int, m_verify: int,
               gpu_mem_util: float, max_batched_tokens: int, verbose_k: int,
               offline_control_k: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    dims = read_dims(model_dir)
    rows = load_ref_records(ref_jsonl, n_prompts)
    max_len = max((len(r["P"]) + len(r["R"]) for r in rows), default=2048) + 64
    print(f"[scan] model={model_dir} hidden={dims['hidden']} layers={dims['num_layers']} "
          f"lmhead_int4={dims.get('lmhead_quant')} M_verify={m_verify} n_rows={len(rows)} "
          f"max_len={max_len} VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(1024, max_len), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )
    print(f"[scan] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_is_batch_invariant = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_is_batch_invariant = False

    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=PROMPT_LOGPROBS_K,
                              skip_reading_prefix_cache=False, detokenize=False)

    n_match = n_total = 0
    n_chunk_isolated = n_chunk_total = 0
    n_computed_rows_total = 0
    flip_gaps = []
    flip_margins = []
    gap_hist = [0] * (len(GAP_HIST_EDGES) - 1)
    flag_trigger_counts = {t: 0 for t in TAU_FLAG_SWEEP}
    rescued_break_counts = {t: 0 for t in TAU_FLAG_SWEEP}
    per_prompt = []
    # offline-vs-served R control (subset)
    offl_checked = offl_match_seq = 0
    offl_first_div = []

    def hist_bin(g: float) -> int:
        for i in range(len(GAP_HIST_EDGES) - 1):
            if g < GAP_HIST_EDGES[i + 1]:
                return i
        return len(GAP_HIST_EDGES) - 2

    t_run = time.time()
    for ri, rec in enumerate(rows):
        P = rec["P"]
        R = rec["R"]
        L_p = len(P)
        # commit P and P+R into the prefix cache in 32-blocks (we do NOT generate R; it is the
        # served trajectory) so the chunk-reads below hit an isolated size_m=M_verify suffix.
        llm.generate([{"prompt_token_ids": P}], sp_warm, use_tqdm=False)
        llm.generate([{"prompt_token_ids": P + R}], sp_warm, use_tqdm=False)

        # optional offline-vs-served control: regenerate R_offline on the SAME engine and byte-compare.
        if ri < offline_control_k:
            sp_gen = SamplingParams(temperature=0.0, max_tokens=len(R), detokenize=False)
            ro = list(llm.generate([{"prompt_token_ids": P}], sp_gen, use_tqdm=False)[0].outputs[0].token_ids)
            offl_checked += 1
            fd = next((i for i, (x, y) in enumerate(zip(ro, R)) if x != y), -1)
            offl_first_div.append(fd if fd >= 0 else min(len(ro), len(R)))
            offl_match_seq += int(ro[:len(R)] == R)

        # slide the size_m=M_verify chunk along R_served at offsets making (L_p+o) 32-aligned.
        o_start = block_ceil(L_p) - L_p          # first o with L_p+o on a 32 boundary
        prompt_match = prompt_total = 0
        prompt_min_gap = float("inf")
        o = o_start
        while o + m_verify <= len(R):
            full = P + R[:o + m_verify]
            out = llm.generate([{"prompt_token_ids": full}], sp_chunk, use_tqdm=False)[0]
            nct = out.num_cached_tokens or 0
            n_computed = len(full) - nct
            n_chunk_total += 1
            n_computed_rows_total += n_computed
            isolated = (n_computed == m_verify)
            n_chunk_isolated += int(isolated)
            if not isolated:
                o += HYBRID_PREFIX_COMMIT
                continue
            pls = out.prompt_logprobs or []
            for i in range(L_p + o + 1, L_p + o + m_verify):
                entry = pls[i] if i < len(pls) else None
                if entry is None:
                    continue
                sl = _sorted_logprobs(entry)
                m8_arg = int(sl[0][0])
                m1_tok = full[i]              # == R_served[i-L_p], the served M=1 AR token
                gap = (sl[0][1] - sl[1][1]) if len(sl) >= 2 else float("inf")
                prompt_min_gap = min(prompt_min_gap, gap)
                prompt_total += 1
                gap_hist[hist_bin(gap)] += 1
                is_flip = (m8_arg != m1_tok)
                for t in TAU_FLAG_SWEEP:
                    if gap < t:
                        flag_trigger_counts[t] += 1
                    elif is_flip:
                        rescued_break_counts[t] += 1
                if not is_flip:
                    prompt_match += 1
                else:
                    flip_gaps.append(gap)
                    lp_map = dict(sl)
                    margin = (sl[0][1] - lp_map[m1_tok]) if m1_tok in lp_map else None
                    flip_margins.append(margin)
            o += HYBRID_PREFIX_COMMIT

        if prompt_total == 0:
            continue
        n_match += prompt_match
        n_total += prompt_total
        per_prompt.append({
            "id": rec["id"], "L_p": L_p, "positions": prompt_total,
            "match_M8_vs_M1": prompt_match,
            "min_top2_gap": (prompt_min_gap if math.isfinite(prompt_min_gap) else None),
        })
        if ri < verbose_k or ri == len(rows) - 1:
            br = 1.0 - prompt_match / prompt_total if prompt_total else float("nan")
            print(f"[scan] prompt {ri} id={rec['id']} L_p={L_p} pos={prompt_total} "
                  f"break={prompt_total-prompt_match}/{prompt_total} ({br:.5f}) "
                  f"cum_pos={n_total} elapsed={time.time()-t_run:.0f}s", flush=True)

    n_flips = n_total - n_match
    break_rate = (n_flips / n_total) if n_total else float("nan")
    chunk_isolated_frac = (n_chunk_isolated / n_chunk_total) if n_chunk_total else float("nan")
    margins_present = [m for m in flip_margins if m is not None]
    n_m1_outside_topk = sum(1 for m in flip_margins if m is None)

    tau_table = []
    for t in TAU_FLAG_SWEEP:
        ftc, rbc = flag_trigger_counts[t], rescued_break_counts[t]
        tau_table.append({
            "tau_flag": t,
            "flag_trigger_count": ftc,
            "flag_trigger_rate": (ftc / n_total) if n_total else float("nan"),
            "rescued_break_count": rbc,
            "rescued_break_rate": (rbc / n_total) if n_total else float("nan"),
        })

    aten_ctrl = aten_mm_invariance_control(torch, dims["hidden"], m_verify)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "scan", "model_dir": model_dir, "ref_jsonl": ref_jsonl,
        "vllm_batch_invariant_env": batch_invariant_env,
        "attn_is_batch_invariant": attn_is_batch_invariant,
        "lmhead_int4": dims.get("lmhead_quant"),
        "n_prompts": len(per_prompt), "M_verify": m_verify,
        "total_positions": n_total, "matching_positions": n_match, "n_flips": n_flips,
        "unrescued_break_rate": break_rate,
        "chunk_isolated_fraction": chunk_isolated_frac,
        "n_chunks_total": n_chunk_total, "n_chunks_isolated": n_chunk_isolated,
        "mean_computed_rows": (n_computed_rows_total / n_chunk_total) if n_chunk_total else None,
        "n_m1_token_outside_topk": n_m1_outside_topk,
        "flip_gap_median": (statistics.median(flip_gaps) if flip_gaps else None),
        "flip_gap_max": (max(flip_gaps) if flip_gaps else None),
        "flip_margin_median": (statistics.median(margins_present) if margins_present else None),
        "flip_margin_max": (max(margins_present) if margins_present else None),
        "flip_gaps": [round(g, 5) for g in flip_gaps if math.isfinite(g)],
        "flip_margins": [round(m, 5) if m is not None else None for m in flip_margins],
        "gap_hist_edges": list(GAP_HIST_EDGES), "gap_hist": gap_hist,
        "tau_flag_table": tau_table,
        "offline_control": {
            "n_checked": offl_checked,
            "offline_eq_served_seq": offl_match_seq,
            "offline_vs_served_seq_divergence_rate": (
                1.0 - offl_match_seq / offl_checked) if offl_checked else None,
            "first_divergence_positions": offl_first_div,
        },
        "aten_mm_control": aten_ctrl,
        "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[scan] unrescued_break_rate={break_rate:.6f} ({n_flips}/{n_total})  "
          f"chunk_isolated={chunk_isolated_frac:.4f} attn_bi={attn_is_batch_invariant} "
          f"flip_gap_max={out['flip_gap_max']} peak={peak_gb:.1f}GB", flush=True)
    for r in tau_table:
        print(f"[scan]   tau_flag={r['tau_flag']:<5} flag_trigger_rate={r['flag_trigger_rate']:.5f} "
              f"rescued_break_rate={r['rescued_break_rate']:.6f} "
              f"(rescued_breaks={r['rescued_break_count']})", flush=True)
    oc = out["offline_control"]
    if oc["n_checked"]:
        print(f"[scan]   offline-vs-served R divergence={oc['offline_vs_served_seq_divergence_rate']:.4f} "
              f"({oc['n_checked']-oc['offline_eq_served_seq']}/{oc['n_checked']} prompts)", flush=True)
    print(f"SCAN_DONE {out_path}", flush=True)


def aten_mm_invariance_control(torch, hidden: int, batch_m: int) -> dict:
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    w = torch.randn(hidden, hidden, dtype=torch.bfloat16, device=dev)
    x = torch.randn(max(batch_m, 16), hidden, dtype=torch.bfloat16, device=dev)
    y1 = torch.mm(x[:1].contiguous(), w)
    ym = torch.mm(x[:batch_m].contiguous(), w)
    torch.cuda.synchronize()
    return {"bitexact_M1_vs_Mverify": bool(torch.equal(ym[:1].float(), y1.float())),
            "max_abs_diff": float((ym[:1].float() - y1.float()).abs().max()), "batch_m": batch_m}


# ======================================================================================
# CPU: served cascade S_served vs R_served (no GPU)
# ======================================================================================
def analyze_cascade(ref_jsonl: str, spec_jsonl: str) -> dict:
    ref = {r["id"]: r for r in load_ref_records(ref_jsonl, 10 ** 9)}
    spec = {r["id"]: r for r in load_ref_records(spec_jsonl, 10 ** 9)}
    ids = [i for i in ref if i in spec]
    n_prompts = len(ids)
    n_div_prompts = 0
    tok_total = tok_break = 0
    first_divs = []
    per = []
    for i in ids:
        R = ref[i]["R"]
        S = spec[i]["R"]
        L = min(len(R), len(S))
        fd = next((k for k in range(L) if R[k] != S[k]), -1)
        nb = sum(1 for k in range(L) if R[k] != S[k])
        diverged = (fd >= 0) or (len(R) != len(S))
        n_div_prompts += int(diverged)
        tok_total += L
        tok_break += nb
        first_divs.append(fd if fd >= 0 else L)
        per.append({"id": i, "len_R": len(R), "len_S": len(S),
                    "first_divergence": fd, "n_break_tokens": nb, "diverged": diverged})
    return {
        "n_prompts": n_prompts,
        "served_seq_divergence_rate": (n_div_prompts / n_prompts) if n_prompts else None,
        "n_diverged_prompts": n_div_prompts,
        "served_pertoken_break_rate": (tok_break / tok_total) if tok_total else None,
        "tok_total": tok_total, "tok_break": tok_break,
        "first_divergence_median": (statistics.median(first_divs) if first_divs else None),
        "ref_jsonl": ref_jsonl, "spec_jsonl": spec_jsonl,
        "per_prompt": per,
    }


# ======================================================================================
# orchestrator
# ======================================================================================
def run_scan_subprocess(args_list, extra_env=None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["VLLM_BATCH_INVARIANT"] = "1"   # BI=1 both sides (the #622 decisive config)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)} (BI={env['VLLM_BATCH_INVARIANT']})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"scan subprocess failed (rc={rc}): {args_list}")


def compose_and_report(a) -> dict:
    scan = json.load(open(OUT_DIR / f"scan_M{a.m_verify}.json"))
    cascade = None
    if a.spec_jsonl and Path(a.spec_jsonl).exists():
        cascade = analyze_cascade(a.ref_jsonl, a.spec_jsonl)
        json.dump(cascade, open(OUT_DIR / "served_cascade.json", "w"), indent=2)

    n_tot = scan["total_positions"]
    by_tau = {r["tau_flag"]: r for r in scan["tau_flag_table"]}
    # min tau_flag achieving 0 served rescued breaks
    min_tau = None
    for t in TAU_FLAG_SWEEP:
        if by_tau.get(t, {}).get("rescued_break_count", 1) == 0:
            min_tau = t
            break
    chosen_tau = min_tau if min_tau is not None else TAU_FLAG_SWEEP[-1]
    chosen = by_tau[chosen_tau]
    served_rescued_break_rate = chosen["rescued_break_rate"]
    served_ftr = chosen["flag_trigger_rate"]

    served_break_holds = (min_tau is not None) and (served_rescued_break_rate == 0.0)
    lb, ub = _wilson_ci(int(round(served_rescued_break_rate * n_tot)), n_tot)
    report = {
        "pr": 642, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "leg": "served-path de-teacher-forced identity (#636 acceptor along the SERVED M=1 AR "
               "trajectory R_served, real int4 body, BI=1 both sides)",
        "imported_anchors": {
            "stark_636_teacher_forced_break_rate": STARK_636_TF_BREAK_RATE,
            "stark_636_teacher_forced_flip_gap_max": STARK_636_TF_FLIP_GAP_MAX,
            "land_632_served_seq_divergence": LAND_632_SERVED_SEQ_DIVERGENCE,
            "pr576_served_perstep_identity": PR576_SERVED_PERSTEP_IDENTITY,
            "M_verify": a.m_verify,
        },
        # ---- make-or-break deliverable ----
        "served_rescued_break_rate": served_rescued_break_rate,
        "served_break_rate_holds_zero": served_break_holds,
        "min_tau_flag_for_zero_served_breaks": min_tau,
        "served_flag_trigger_rate_at_min_tau": served_ftr,
        "served_unrescued_break_rate": scan["unrescued_break_rate"],
        "served_unrescued_break_rate_ci95": list(_wilson_ci(scan["n_flips"], n_tot)),
        "served_rescued_break_rate_ci95": [lb, ub],
        "served_rescued_break_rule_of_three_ub": _rule_of_three_ub(n_tot),
        # ---- de-teacher-force comparison ----
        "teacher_forced_unrescued_break_rate_636": STARK_636_TF_BREAK_RATE,
        "served_minus_teacher_forced": scan["unrescued_break_rate"] - STARK_636_TF_BREAK_RATE,
        "offline_vs_served_R_divergence": scan["offline_control"].get(
            "offline_vs_served_seq_divergence_rate"),
        "served_flip_gap_max": scan["flip_gap_max"],
        "served_flip_margin_max": scan["flip_margin_max"],
        "served_n_m1_token_outside_topk": scan["n_m1_token_outside_topk"],
        # ---- served cascade cross-check (un-rescued) ----
        "served_cascade": (None if cascade is None else {
            "served_seq_divergence_rate": cascade["served_seq_divergence_rate"],
            "n_diverged_prompts": cascade["n_diverged_prompts"],
            "n_prompts": cascade["n_prompts"],
            "served_pertoken_break_rate": cascade["served_pertoken_break_rate"],
            "first_divergence_median": cascade["first_divergence_median"],
        }),
        # ---- frontier / controls ----
        "tau_flag_table": scan["tau_flag_table"],
        "total_positions": n_tot, "n_flips": scan["n_flips"],
        "gap_hist_edges": scan["gap_hist_edges"], "gap_hist": scan["gap_hist"],
        "attn_is_batch_invariant": scan["attn_is_batch_invariant"],
        "aten_mm_bitexact": scan["aten_mm_control"].get("bitexact_M1_vs_Mverify"),
        "chunk_isolated_fraction": scan["chunk_isolated_fraction"],
        "lmhead_int4": scan["lmhead_int4"],
        "M_verify": a.m_verify, "n_prompts": scan["n_prompts"], "model_dir": scan["model_dir"],
    }
    return report


def _finish(report, a) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_DIR / f"served_identity_report_M{a.m_verify}.json", "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(r) -> None:
    print("\n========== SERVED-PATH DE-TEACHER-FORCED IDENTITY (PR #642, item #3) ==========", flush=True)
    print(f" M_verify                            : {r['M_verify']}", flush=True)
    print(f" served_break_rate_holds_zero        : {r['served_break_rate_holds_zero']}  "
          f"(make-or-break)", flush=True)
    print(f" served_rescued_break_rate           : {r['served_rescued_break_rate']} "
          f"@ min_tau={r['min_tau_flag_for_zero_served_breaks']}", flush=True)
    print(f" served_flag_trigger_rate @ min_tau  : {r['served_flag_trigger_rate_at_min_tau']:.5f}", flush=True)
    print(f" served_unrescued_break_rate         : {r['served_unrescued_break_rate']:.6f} "
          f"({r['n_flips']}/{r['total_positions']})", flush=True)
    print(f"   teacher-forced #636 (offline R)   : {r['teacher_forced_unrescued_break_rate_636']:.6f} "
          f"(served - tf = {r['served_minus_teacher_forced']:+.6f})", flush=True)
    print(f" offline-vs-served R divergence      : {r['offline_vs_served_R_divergence']}", flush=True)
    print(f" served_flip_gap_max                 : {r['served_flip_gap_max']} nat", flush=True)
    if r["served_cascade"]:
        c = r["served_cascade"]
        print(f" served cascade (un-rescued spec):", flush=True)
        print(f"   seq_divergence_rate              : {c['served_seq_divergence_rate']} "
              f"({c['n_diverged_prompts']}/{c['n_prompts']} prompts) "
              f"[land #632 ~0.84]", flush=True)
        print(f"   pertoken_break_rate              : {c['served_pertoken_break_rate']}", flush=True)
    print(" tau_flag frontier (served):", flush=True)
    for row in r["tau_flag_table"]:
        print(f"   tau={row['tau_flag']:<5} flag_trigger_rate={row['flag_trigger_rate']:.5f}  "
              f"rescued_breaks={row['rescued_break_count']} (rate {row['rescued_break_rate']:.6f})",
              flush=True)
    print(f" controls: attn_bi={r['attn_is_batch_invariant']} aten_mm_bitexact={r['aten_mm_bitexact']} "
          f"chunk_isolated={r['chunk_isolated_fraction']:.4f} lmhead_int4={r['lmhead_int4']}", flush=True)
    print("===============================================================================\n", flush=True)


def log_wandb(report, a) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark", name=a.wandb_name, group=a.wandb_group,
        notes="PR#642 served-path de-teacher-forced identity: does the #636 recompute acceptor hold "
              "break_rate=0 along the SERVED M=1 AR trajectory (not the offline one)?",
        config={"pr": 642, "M_verify": a.m_verify, "n_prompts": report["n_prompts"],
                "model_dir": report["model_dir"], "tau_flag_sweep": list(TAU_FLAG_SWEEP),
                "ref_jsonl": a.ref_jsonl, "spec_jsonl": a.spec_jsonl, "stack_vllm": "0.22.0"},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return
    summary = {
        "served_break_rate_holds_zero": report["served_break_rate_holds_zero"],
        "served_rescued_break_rate": report["served_rescued_break_rate"],
        "min_tau_flag_for_zero_served_breaks": report["min_tau_flag_for_zero_served_breaks"],
        "served_flag_trigger_rate_at_min_tau": report["served_flag_trigger_rate_at_min_tau"],
        "served_unrescued_break_rate": report["served_unrescued_break_rate"],
        "teacher_forced_unrescued_break_rate_636": report["teacher_forced_unrescued_break_rate_636"],
        "served_minus_teacher_forced": report["served_minus_teacher_forced"],
        "offline_vs_served_R_divergence": report["offline_vs_served_R_divergence"],
        "served_flip_gap_max": report["served_flip_gap_max"],
        "total_positions": report["total_positions"], "n_flips": report["n_flips"],
        "served_rescued_break_rule_of_three_ub": report["served_rescued_break_rule_of_three_ub"],
        "attn_is_batch_invariant": report["attn_is_batch_invariant"],
        "aten_mm_bitexact": report["aten_mm_bitexact"],
        "chunk_isolated_fraction": report["chunk_isolated_fraction"],
        "M_verify": report["M_verify"],
    }
    if report["served_cascade"]:
        c = report["served_cascade"]
        summary["served_cascade_seq_divergence_rate"] = c["served_seq_divergence_rate"]
        summary["served_cascade_pertoken_break_rate"] = c["served_pertoken_break_rate"]
    for row in report["tau_flag_table"]:
        tag = str(row["tau_flag"]).replace(".", "p")
        summary[f"ftr_tau_{tag}"] = row["flag_trigger_rate"]
        summary[f"served_rescued_breaks_tau_{tag}"] = row["rescued_break_count"]
        summary[f"served_rescued_break_rate_tau_{tag}"] = row["rescued_break_rate"]
    for k, v in summary.items():
        run.summary[k] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def orchestrate(a) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_scan_subprocess([
        "--phase", "scan", "--ref-jsonl", a.ref_jsonl, "--n-prompts", str(a.n_prompts),
        "--m-verify", str(a.m_verify), "--gpu-mem-util", str(a.gpu_mem_util),
        "--max-batched-tokens", str(a.max_batched_tokens), "--verbose-k", str(a.verbose_k),
        "--offline-control-k", str(a.offline_control_k),
        "--out", str(OUT_DIR / f"scan_M{a.m_verify}.json"),
    ])
    _finish(compose_and_report(a), a)


def reanalyze(a) -> None:
    _finish(compose_and_report(a), a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["scan"], default=None)
    ap.add_argument("--reanalyze", action="store_true")
    ap.add_argument("--ref-jsonl", dest="ref_jsonl", required=False,
                    help="SENPAI_REFERENCE_MODE=1 served decode jsonl (R_served = M=1 AR trajectory)")
    ap.add_argument("--spec-jsonl", dest="spec_jsonl", default=None,
                    help="un-rescued spec served decode jsonl (S_served) for the cascade cross-check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--m-verify", dest="m_verify", type=int, default=8,
                    help="verify width M = K_spec+1 (K=7->8 #636 anchor; K=6->7 served; K=5->6)")
    ap.add_argument("--offline-control-k", dest="offline_control_k", type=int, default=16,
                    help="regenerate R_offline on the first K prompts to measure offline-vs-served R")
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=8)
    ap.add_argument("--wandb_group", dest="wandb_group", default="optionb-rescue-deproject-stark")
    ap.add_argument("--wandb_name", dest="wandb_name", default="stark/served-identity-deteacherforce")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.phase == "scan":
        phase_scan(a.ref_jsonl, a.out, a.n_prompts, a.m_verify, a.gpu_mem_util,
                   a.max_batched_tokens, a.verbose_k, a.offline_control_k)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
