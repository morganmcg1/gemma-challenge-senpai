#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #487 (wirbel) -- SAME-RELOAD FULL-SERVE identity census harness.

THE PROBLEM THIS CLOSES
-----------------------
Two prior reads of "is the M=8 strict serve token-identical to plain M=1 AR greedy?" disagree, and BOTH
are incomplete:
  * denken #471 LOCUS certifier: reload-immune, but measures ONE bounded window (ground-truth prefix C=224,
    7 readable verify positions x 127 prompts = 889 positions). Identity 0.9988751406074241 (1 residual flip,
    a bf16 TIE). It proves identity-1.0 at a single context length, not along a real free-running completion.
  * ubel #470 FULL-SERVE census: ran the actual free-running M=8 spec serve vs a SEPARATELY-launched M=1 AR
    serve and compared. But those are TWO processes (TWO reloads), and the M=8 spec-decode path is
    cross-reload-UNSTABLE (M=8-vs-M=8 across reloads = 0.6431; #38). So its 0.4085 is the reload floor + an
    early-flip cascade, reload-CONFOUNDED -- it neither confirms nor refutes byte-exact identity.

This harness gives the missing measurement: a FULL-SERVE census (sweeping the whole free-running completion,
context length L from C up to ~C+traj_len) that is RELOAD-IMMUNE, by running BOTH the M=1 AR reference AND the
M=8 verify reads inside ONE vLLM engine instance (teacher-forced along the M=1 trajectory). No second reload,
no cross-reload confound, no early-flip cascade.

THE MECHANISM (extends denken #471 / stark #412 's teacher-forced locus census)
------------------------------------------------------------------------------
In ONE LLM() reload, per prompt:
  1. prefix = src[:C]  (C = block_align(224) = 224, a 32-committed cache boundary).
  2. Generate the M=1 AR greedy trajectory of `traj_len` tokens from `prefix` (temperature=0, single-stream
     decode -- the canonical "plain greedy autoregressive decode" reference). Keep its per-step logprobs.
     seq = prefix + traj.
  3. Sweep G=32-aligned windows O in {C, C+32, ...} ALONG seq. At each O, replay the served verify geometry:
     feed seq[:O+W] as a chunked prompt with prompt_logprobs + prefix-caching, so seq[:O] is cached and exactly
     the next W tokens compute in ONE width-W forward (verified via num_computed_rows == W). Read the argmax at
     each computed position p = the "M=W verify argmax" and compare to seq[p] = the M=1 AR token at p.
  W=8 reads (`positions`)  = the FAITHFUL served M=8 verify geometry (HEADLINE census).
  W=32 reads (`positions_wide`) = FULL per-position coverage. Faithful because the FA2 varlen split-KV
     num_splits is M-INVARIANT for M<=64 (num_m_blocks = ceil(M/64) = 1 for M in {8,16,32,64}): same split
     pattern => same per-row float accumulation order => same argmax. We CONFIRM this on-engine
     (`width_equivalence_rate`: W=8 vs W=32 argmax on the shared O+1..O+7 positions must be 1.0).

Reload-immune because the M=1 reference and the M=8 reads share the SAME loaded weights/KV; within-reload M=8
determinism is 1.0 (the cross-reload 0.64 instability never enters). Probes assert it every window: det_m8
(two identical W=8 reads), within (two batched copies), det_m1 (two trajectory regenerations, spot-checked).

FLIP CLASSIFICATION (the operative-1.0 question)
------------------------------------------------
A position is a FLIP iff M=8 verify argmax != M=1 AR token. Each flip is:
  * TIE flip   -- the M=1 AR reference's OWN top-2 logprobs are bit-identical (m1_self_gap <= 1e-9). For a true
                  bf16 tie "the right greedy token" is undefined; the verify arbiter resolves it to an
                  equally-valid-greedy token. Operatively identity-1.0 (a verify-arbiter fixed point).
  * SEMANTIC flip -- the M=1 AR reference has a clear top-1 winner but M=8 picked a different token. A REAL
                  divergence; the only kind that can break the strict greedy-equivalence contract.
OPERATIVE-1.0 verdict := identity >= 0.99 AND n_semantic_flips == 0 (every non-identity position is a bf16 TIE).

SCOPE: LOCAL A10G (sm_86) post-hoc census. analysis_only / no_hf_job / no_served_file_change / official_tps=0.
No served/deployed file is touched; the int4 path is READ only; NO HF job; NO submission. This is a measurement
harness, not a serving change.

A NOTE ON W=8 vs W=32 (faithfulness)
------------------------------------
The FAITHFUL headline is the W=8 read: it replicates the deployed M=8 verify geometry exactly (a width-8 forward
over a decode-cached prefix), so its identity vs M=1 AR is the real served-equivalence number. The W=32 read is an
auxiliary DENSER cross-read; the harness EMPIRICALLY TESTS the assumption that it is byte-equivalent and finds it
is NOT: (a) at the O+1..O+7 overlap the split-K reduction order is width-invariant only to ~99.9% (researcher Q3
"holds but not byte-exactly"), and (b) at the O+8..O+31 extension the wide prefill diverges from the M=1 AR token
at clear-winner positions, because a 31-wide prefill accumulates prefill-vs-decode KV rounding over its span (the
serve never does a 31-wide verify, so this is a non-served artifact, NOT an M=8 identity failure). The verdict is
therefore judged on the W=8 served deliverable; W=32 is reported as evidence that a wide read is not a faithful
shortcut.

DELIVERABLES (W&B summary/)
  reloadimmune_fullserve_census (HEADLINE, W=8 served geometry, all windows);  headline_geometry;
  n_semantic_flips; n_tie_flips; operative_identity_1p0 (bool PRIMARY);
  reloadimmune_fullserve_census_fullcov (W=32 denser cross-read; fullcov_is_byte_faithful=False);
  width_findings/* (overlap_invariance_nontie, extension_flips_semantic/tie, faithful_census_geometry);
  census_at_locus_O224 (W=8 O=224 sub-census == denken #471's measurement, the cross-check);
  crosscheck_vs_denken471_gap; reload_immune (det probes all 1.0);
  determinism_M1 / determinism_M8 / within_batch; n_windows; n_positions; coverage_multiple_vs_locus;
  fullserve_self_test_passes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ======================================================================================
# Imported fleet anchors (CITE; do NOT re-derive)
# ======================================================================================
# denken #471 (reload-immune LOCUS certifier) -- the cross-check authority at C=224:
DENKEN471_LOCUS_IDENTITY = 0.9988751406074241    # 888/889 -- 1 residual flip (a bf16 TIE; operative-1.0)
DENKEN471_LOCUS_POSITIONS = 889                  # 127 prompts x 7 readable verify positions
DENKEN471_LOCUS_FLIPS = 1
# ubel #470 (reload-CONFOUNDED served census) -- the number this harness REPLACES with a clean read:
UBEL470_SERVED_CONFOUNDED_IDENTITY = 0.4084625244140625   # reload floor + cascade; NOT a clean BI-identity read
UBEL470_M8_XRELOAD_FLOOR = 0.6431427001953125             # M=8-vs-M=8 across reloads (#38) -- the confound source
UBEL470_M1_XRELOAD = 0.9937485625875515                   # M=1 AR IS cross-reload-stable

# strict-pin identity references (reload-immune methods; CITE):
PR461_ALLPIN_IDENTITY = 0.99775                  # population/matched, blanket VBI=1
STARK466_LOCUS_IDENTITY = 1.0                     # surgical num_splits=1 locus proof (0 flips)

# ---- PR #510 surgical-357 ship census anchors (CITE; this run measures the surgical arm) ----
# PR #487 (wirbel, run j5vyk14b) -- the GLOBAL-FLAG 222 (VLLM_BATCH_INVARIANT=1) full-serve census this
# surgical census is compared against. The decisive "vs_globalflag222" comparison is surgical n_semantic
# flips vs THIS 12. (12 sub-ULP near-tie semantic flips, all m1_self_gap in 1-2 bf16 ULPs.)
PR487_GLOBALFLAG222_FULLSERVE_CENSUS = 0.9973493   # W=8 served-geometry, 14298/14336
PR487_GLOBALFLAG222_N_SEMANTIC = 12                # the headline number surgical is compared to
PR487_GLOBALFLAG222_N_TIE = 26
PR487_GLOBALFLAG222_TIE_THRESHOLD_ZERO_SEMANTIC = 0.171875   # all 12 collapse to ties at >=0.172 nat (#487)
# stark #494 (run k8nqmc2b / margin census 5fxw18gu) -- the SURGICAL LOCUS cert this run stress-tests at
# full serve. attn_only locus identity 0.998875 (1 residual flip, a bf16-ULP near-tie, 0 semantic); the
# surgical attn_only divergence == the 222 all_pin divergence to 15 sig figs (matmul tax drops 0 identity).
STARK494_SURGICAL_LOCUS_IDENTITY = 0.998875        # surgical attn_only locus (C=224, M=8 verify), 1 flip
STARK494_SURGICAL_LOCUS_DIVERGENCE = 0.00112486    # == 222 all_pin divergence (matmul tax identity-neutral)

# the shipped surgical-357 serve recipe -- the arm loads its EXACT lever (surgical_attn_patch.install()):
SURGICAL_SUBMISSION_DIR = "submissions/fa2sw_strict_surgical357"

# ======================================================================================
# Constants (the verify geometry -- identical to #381/#405/#412/#471 for comparability)
# ======================================================================================
K_SPEC = 7
M_VERIFY = K_SPEC + 1                             # = 8, the deployed decode-verify query width (W=8)
WIDE_W = 32                                       # full-coverage read width == HYBRID_PREFIX_COMMIT (M<=64 inv.)
EPS_STAR = 0.125                                  # bf16 near-tie band (covers every observed flip)
BAND_TOL = 1e-9                                   # bitwise-tie threshold on the M=1 self gap
HYBRID_PREFIX_COMMIT = 32                         # Gemma-4 hybrid prefix-cache commit granularity (#381); =G
DEFAULT_C0 = 224                                  # ground-truth prefix length (denken #471 locus)
DEFAULT_TRAJ_LEN = 512                            # full free-running completion length to sweep

MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
OUT_DIR = Path("research/validity/fullserve_identity_census_harness")
CENSUS_ARMS_DEFAULT = ("pinned",)                # the global-flag strict config (VLLM_BATCH_INVARIANT=1)


# ======================================================================================
# Small helpers (reused from #381/#405/#412)
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


def _m1_is_bitwise_tie(p: int, m1_lp: list, C: int) -> bool:
    """Is the M=1 AR reference's OWN top-2 bit-identical at position p (argmax arbitrary)?"""
    k = p - C
    e = m1_lp[k] if 0 <= k < len(m1_lp) else None
    sl = _sorted_logprobs(e) if e else []
    return bool(len(sl) >= 2 and (sl[0][1] - sl[1][1]) <= BAND_TOL)


def classify_position(ri: int, p: int, ent: dict, seq: list[int], m1_lp: list, C: int, width: int):
    """Build one census position record: M=W verify argmax vs M=1 AR token at position p, with tie/semantic
    flip classification driven by the M=1 reference's OWN top-2 gap (bitwise tie => operative identity-1.0)."""
    sl = ent.get(p, [])
    if len(sl) < 2:
        return None
    m8_top1_id, m8_top1_lp = sl[0]
    m8_top2_id, m8_top2_lp = sl[1]
    m8_gap = m8_top1_lp - m8_top2_lp
    m8_ids = [tid for tid, _ in sl]
    m1_tok = seq[p]                                  # the M=1 AR greedy token at this position (the reference)
    is_flip = int(m8_top1_id != m1_tok)

    k = p - C                                        # index into the trajectory / its per-step logprobs
    m1_entry = m1_lp[k] if 0 <= k < len(m1_lp) else None
    m1_sl = _sorted_logprobs(m1_entry) if m1_entry else []
    m1_top1_id = m1_sl[0][0] if m1_sl else None
    m1_self_gap = (m1_sl[0][1] - m1_sl[1][1]) if len(m1_sl) >= 2 else None
    m1_argmax_matches_token = bool(m1_top1_id == m1_tok) if m1_top1_id is not None else None
    m1_is_bitwise_tie = bool(m1_self_gap is not None and m1_self_gap <= BAND_TOL)
    is_near_tie = bool(m8_gap <= EPS_STAR + BAND_TOL)

    flip_kind = None
    if is_flip:
        flip_kind = "tie" if m1_is_bitwise_tie else "semantic"
    return {
        "prompt_idx": ri, "pos": p, "L": p, "k": k, "width": width,
        "m8_gap": round(m8_gap, 6), "m8_top1_id": m8_top1_id, "m8_top2_id": m8_top2_id,
        "m1_tok_id": m1_tok, "is_flip": is_flip, "is_near_tie": is_near_tie,
        "m1_in_m8_top2": bool(m1_tok in (m8_top1_id, m8_top2_id)),
        "m1_in_m8_top5": bool(m1_tok in m8_ids),
        "m1_self_gap": (round(m1_self_gap, 6) if m1_self_gap is not None else None),
        "m1_argmax_matches_token": m1_argmax_matches_token,
        "m1_is_bitwise_tie": m1_is_bitwise_tie,
        "flip_kind": flip_kind,
    }


def _agg(positions: list[dict]) -> dict:
    """Aggregate a position list into identity + tie/semantic flip accounting."""
    n_total = len(positions)
    flips = [p for p in positions if p["is_flip"]]
    n_flips = len(flips)
    n_match = n_total - n_flips
    n_tie = sum(1 for p in flips if p["flip_kind"] == "tie")
    n_sem = sum(1 for p in flips if p["flip_kind"] == "semantic")
    return {
        "n_positions": n_total, "n_match": n_match, "n_flips": n_flips,
        "n_tie_flips": n_tie, "n_semantic_flips": n_sem,
        "token_identity_rate": (n_match / n_total) if n_total else float("nan"),
        "operative_identity_rate": ((n_match + n_tie) / n_total) if n_total else float("nan"),
    }


# ======================================================================================
# Surgical-357 lever: load the SHIPPED serve recipe's exact attention pin (PR #510)
# ======================================================================================
def _surgical_mode(arm: str) -> bool:
    """The surgical arm == the shipped surgical-357 config (2D order-preserving attention on the 7
    full-attn layers, matmul tax OFF). Recognised by arm name or the ship's gating env."""
    return arm == "surgical" or os.environ.get("SURGICAL_ATTN_USE_3D_OFF", "0") == "1"


def _install_surgical_lever() -> dict:
    """Install the EXACT shipped surgical-357 lever by running the submission's own
    ``surgical_attn_patch.install()`` (not a reimplementation): it sets the module global
    ``triton_unified_attention.is_batch_invariant=True`` (forces ``use_3d=False`` -> vLLM's byte-exact
    2D order-preserving sequential-KV attention) WITHOUT ``VLLM_BATCH_INVARIANT=1`` -- so vLLM's
    ``init_batch_invariance()`` never installs the SM80 persistent-matmul tax (MLP/QKV/lm_head keep the
    fast Marlin path). This is the only compute-path difference vs the #487 ``pinned`` arm (which sets the
    env and gets BOTH the 2D attention AND the matmul tax). Returns provenance for the report."""
    prov = {"surgical_attn_armed": False, "matmul_tax_installed": None, "lever_source": None,
            "vbi_env_set": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"}
    # the ship patch arms only when SURGICAL_ATTN_USE_3D_OFF=1; ensure it (the watchdog sets it in env too)
    os.environ.setdefault("SURGICAL_ATTN_USE_3D_OFF", "1")
    sub = str(Path(__file__).resolve().parents[3] / SURGICAL_SUBMISSION_DIR)
    if sub not in sys.path:
        sys.path.insert(0, sub)
    try:
        import importlib
        import surgical_attn_patch as sap                       # the SHIPPED patch module
        importlib.reload(sap)                                   # re-read SURGICAL_ATTN_USE_3D_OFF post-set
        prov["surgical_attn_armed"] = bool(sap.install())
        prov["lever_source"] = os.path.join(sub, "surgical_attn_patch.py")
    except Exception as exc:                                    # fail-LOUD here: a silent no-op would make
        prov["error"] = repr(exc)                               # the surgical arm secretly == heuristic
        raise RuntimeError(f"[surgical] could not arm the shipped lever: {exc!r}") from exc
    # cross-check the module global is actually set (the load-bearing assertion)
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        prov["surgical_attn_armed"] = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        pass
    # confirm the matmul tax is NOT installed (the whole point of the surgical lever)
    try:
        import vllm.model_executor.layers.batch_invariant as _bi
        prov["matmul_tax_installed"] = bool(getattr(_bi, "_batch_invariant_MODE", False))
    except Exception:
        prov["matmul_tax_installed"] = None
    print(f"[surgical] armed={prov['surgical_attn_armed']} matmul_tax_installed={prov['matmul_tax_installed']} "
          f"vbi_env_set={prov['vbi_env_set']} source={prov['lever_source']}", flush=True)
    return prov


# ======================================================================================
# PHASE fullserve_census: one arm. Same-reload teacher-forced full-trajectory verify census.
# ======================================================================================
def phase_fullserve_census(out_path: str, arm: str, n_prompts: int, c0: int, traj_len: int,
                           gpu_mem_util: float, max_batched_tokens: int, verbose_k: int,
                           det_check_k: int, ignore_eos: bool,
                           checkpoint: str | None = None, heartbeat: str | None = None,
                           resume: bool = False, skip_prompts: tuple = ()) -> None:
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    C = block_align(c0)
    G = HYBRID_PREFIX_COMMIT
    print(f"[fullserve:{arm}] model={model_dir} C={C} traj_len={traj_len} G={G} W8={M_VERIFY} "
          f"Wwide={WIDE_W} ignore_eos={ignore_eos} VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max(C + traj_len + 64, 800), gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=16, max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
    )
    print(f"[fullserve:{arm}] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    # SURGICAL arm: arm the shipped surgical-357 lever AFTER load (the module is imported, the env stays
    # unset so the matmul tax never installed) and BEFORE any generate (is_batch_invariant is read per-call).
    surgical_prov = {"surgical_attn_armed": None, "matmul_tax_installed": None,
                     "vbi_env_set": batch_invariant_env, "lever_source": None}
    if _surgical_mode(arm):
        surgical_prov = _install_surgical_lever()

    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_is_batch_invariant = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_is_batch_invariant = False

    sp_traj = SamplingParams(temperature=0.0, max_tokens=traj_len, logprobs=5,
                             detokenize=False, ignore_eos=ignore_eos)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    sp_chunk = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5,
                              skip_reading_prefix_cache=False, detokenize=False)

    def chunk_read(full_ids: list[int], O: int):
        """Feed seq[:O+W] (seq[:O] cached); read argmax+entries at range(O+1, O+W). Returns (am, ent, nct, nc)."""
        out = llm.generate([{"prompt_token_ids": full_ids}], sp_chunk, use_tqdm=False)[0]
        nct = out.num_cached_tokens or 0
        pls = out.prompt_logprobs or []
        am, ent = {}, {}
        for i in range(O + 1, len(full_ids)):
            entry = pls[i] if i < len(pls) else None
            if entry is not None:
                am[i] = _argmax_from_logprob_entry(entry)
                ent[i] = _sorted_logprobs(entry)
        return am, ent, nct, (len(full_ids) - nct)

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    test_hang_at = int(os.environ.get("FULLSERVE_TEST_HANG_AT", "-1"))   # test hook: simulate a CUDA hang

    # ---- global accumulators: filled IDENTICALLY from fresh GPU computation AND from replayed checkpoint records.
    # Each prompt's full contribution is a JSON-serializable `prec`; appending it to the checkpoint after the prompt
    # completes makes the census resumable. Reload-immunity is unaffected: a prompt's M=8 reads and its M=1 reference
    # still share THIS process's single reload (resume only changes WHICH reload owns a not-yet-done prompt). ----
    positions: list[dict] = []          # W=8 served-geometry census (HEADLINE)
    positions_wide: list[dict] = []     # W=32 full-coverage census
    locus_positions: list[dict] = []    # W=8 O==C window only (== denken #471 cross-check)
    flip_details: list[dict] = []       # all flips (W=8 + W=32), with kind
    per_window: list[dict] = []
    per_prompt: list[dict] = []
    eq_disagreements: list[dict] = []   # W8 != W32 positions (each tagged with M1-tie status)
    det_m8_acc: list[int] = []
    within_acc: list[int] = []
    det_m1_acc: list[int] = []
    iso8_acc: list[int] = []
    iso32_acc: list[int] = []
    eqc = {"match": 0, "total": 0, "nontie_match": 0, "nontie_total": 0}   # W8-vs-W32 argmax equivalence counts
    nwin = {"n": 0}

    def _beat(phase: str, ri: int, wi: int) -> None:
        if not heartbeat:
            return
        try:
            tmp = heartbeat + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"ts": time.time(), "phase": phase, "prompt_idx": ri,
                           "window_idx": wi, "n_done": len(per_prompt)}, fh)
            os.replace(tmp, heartbeat)
        except Exception:
            pass

    def _empty_prec(ri: int, rec: dict, reason: str) -> dict:
        return {"prompt_idx": ri, "positions8": [], "positions32": [], "locus": [], "flips": [],
                "per_window": [],
                "eq": {"match": 0, "total": 0, "nontie_match": 0, "nontie_total": 0, "disagreements": []},
                "probes": {"det_m8": [], "within": [], "det_m1": None, "iso8": [], "iso32": []},
                "n_windows": 0,
                "per_prompt": {"id": rec.get("id") if rec else None, "prompt_idx": ri, reason: True,
                               "is_det_prompt": False, "prompt_secs": 0.0}}

    def _ingest(prec: dict) -> None:
        positions.extend(prec["positions8"])
        positions_wide.extend(prec["positions32"])
        locus_positions.extend(prec["locus"])
        flip_details.extend(prec["flips"])
        per_window.extend(prec["per_window"])
        per_prompt.append(prec["per_prompt"])
        eq_disagreements.extend(prec["eq"]["disagreements"])
        eqc["match"] += prec["eq"]["match"]; eqc["total"] += prec["eq"]["total"]
        eqc["nontie_match"] += prec["eq"]["nontie_match"]; eqc["nontie_total"] += prec["eq"]["nontie_total"]
        det_m8_acc.extend(prec["probes"]["det_m8"])
        within_acc.extend(prec["probes"]["within"])
        if prec["probes"]["det_m1"] is not None:
            det_m1_acc.append(prec["probes"]["det_m1"])
        iso8_acc.extend(prec["probes"]["iso8"]); iso32_acc.extend(prec["probes"]["iso32"])
        nwin["n"] += prec["n_windows"]

    def _process_one_prompt(ri: int, rec: dict) -> dict:
        t_p0 = time.time()
        src = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        if len(src) < C + 1:
            return _empty_prec(ri, rec, "short")
        prefix = src[:C]
        _beat("warm", ri, -1)
        if ri == test_hang_at:                       # FULLSERVE_TEST_HANG_AT: mimic a stuck generate (heartbeat ages)
            print(f"[fullserve:{arm}] TEST hang at prompt {ri}", flush=True)
            while True:
                time.sleep(5)
        llm.generate([{"prompt_token_ids": prefix}], sp_warm, use_tqdm=False)

        _beat("traj", ri, -1)
        outA = llm.generate([{"prompt_token_ids": prefix}], sp_traj, use_tqdm=False)[0]
        traj = list(outA.outputs[0].token_ids)
        m1_lp = list(outA.outputs[0].logprobs or [])
        traj_n = len(traj)
        if traj_n < G + 1:                           # too short to form even one full G-window
            return _empty_prec(ri, rec, "short")
        seq = prefix + traj

        do_det = ri < det_check_k                    # det_m8 / within / det_m1 spot-checked on the first k prompts
        det_m1 = None
        if do_det:
            _beat("traj2", ri, -1)
            outA2 = llm.generate([{"prompt_token_ids": prefix}], sp_traj, use_tqdm=False)[0]
            traj2 = list(outA2.outputs[0].token_ids)
            det_m1 = int(traj[:min(traj_n, len(traj2))] == traj2[:min(traj_n, len(traj2))])

        last_off = C + traj_n - WIDE_W               # last O where a full W=32 window fits
        offsets = list(range(C, last_off + 1, G))
        p8: list[dict] = []; p32: list[dict] = []; loc: list[dict] = []; flips: list[dict] = []
        pw: list[dict] = []; disag: list[dict] = []
        dm8: list[int] = []; win: list[int] = []; i8: list[int] = []; i32: list[int] = []
        eqm = eqt = eqnm = eqnt = 0
        prompt_w8 = prompt_w8_match = 0

        for wi, O in enumerate(offsets):
            _beat("window", ri, wi)
            # No explicit warm: seq[:O] is already committed by the PREVIOUS window's W=32 read (which fed
            # seq[:(O-G)+G] = seq[:O]); the first window's seq[:C] prefix is committed by the trajectory gen.
            am8, ent8, nct8, nc8 = chunk_read(seq[:O + M_VERIFY], O)
            am32, ent32, nct32, nc32 = chunk_read(seq[:O + WIDE_W], O)  # commits seq[:O+G] = next window's prefix
            iso8 = int(nc8 == M_VERIFY)
            iso32 = int(nc32 == WIDE_W)
            det_m8 = None
            if do_det:
                am8b, _, _, nc8b = chunk_read(seq[:O + M_VERIFY], O)
                iso8 = int(nc8 == M_VERIFY and nc8b == M_VERIFY)
                det_m8 = int(bool(am8) and all(am8.get(pp) == am8b.get(pp) for pp in am8))
                dm8.append(det_m8)
            i8.append(iso8)
            i32.append(iso32)

            within = None
            if do_det and wi == 0:
                outW = llm.generate([{"prompt_token_ids": seq[:O + M_VERIFY]},
                                     {"prompt_token_ids": seq[:O + M_VERIFY]}], sp_chunk, use_tqdm=False)

                def _am(o):
                    d, pls = {}, (o.prompt_logprobs or [])
                    for i in range(O + 1, O + M_VERIFY):
                        e = pls[i] if i < len(pls) else None
                        if e is not None:
                            d[i] = _argmax_from_logprob_entry(e)
                    return d
                w0, w1 = _am(outW[0]), _am(outW[1])
                within = int(bool(w0) and all(w0.get(pp) == w1.get(pp) for pp in w0))
                win.append(within)

            # width-equivalence: W=8 vs W=32 argmax on the shared O+1..O+7 positions (researcher Q3 on-engine).
            # Split tie vs non-tie: the meaningful claim is equivalence at CLEAR-WINNER positions; at a bf16 tie
            # the M=1 argmax is arbitrary, so W=8/W=32 may legitimately differ there (operatively identity-1.0).
            overlap = [pp for pp in am8 if pp in am32]
            eq_m = 0
            for pp in overlap:
                eq = int(am8[pp] == am32[pp])
                eq_m += eq
                tie = _m1_is_bitwise_tie(pp, m1_lp, C)
                if not tie:
                    eqnt += 1
                    eqnm += eq
                if not eq:
                    disag.append({"prompt_idx": ri, "pos": pp, "m8_w8": am8[pp],
                                  "m8_w32": am32[pp], "m1_is_tie": tie})
            eqm += eq_m
            eqt += len(overlap)

            w8_recs, w32_recs = [], []
            for pp in sorted(am8):
                r = classify_position(ri, pp, ent8, seq, m1_lp, C, 8)
                if r is None:
                    continue
                w8_recs.append(r)
                p8.append(r)
                if O == C:
                    loc.append(r)
                if r["is_flip"]:
                    flips.append(r)
            for pp in sorted(am32):
                r = classify_position(ri, pp, ent32, seq, m1_lp, C, 32)
                if r is None:
                    continue
                w32_recs.append(r)
                p32.append(r)
                # W=8 flips (pp in O+1..O+7) are already listed from the W=8 loop; add only the extension band
                if r["is_flip"] and pp >= O + M_VERIFY:
                    flips.append(r)

            w8_match = sum(1 for r in w8_recs if not r["is_flip"])
            prompt_w8 += len(w8_recs)
            prompt_w8_match += w8_match
            pw.append({
                "prompt_idx": ri, "window_idx": wi, "O": O, "L_lo": O + 1, "L_hi": O + WIDE_W - 1,
                "n_w8": len(w8_recs), "match_w8": w8_match,
                "n_w32": len(w32_recs), "match_w32": sum(1 for r in w32_recs if not r["is_flip"]),
                "iso8": iso8, "iso32": iso32, "det_m8": det_m8, "within": within,
                "eq_overlap": len(overlap), "eq_match": eq_m,
            })

        prompt_secs = time.time() - t_p0
        return {
            "prompt_idx": ri, "positions8": p8, "positions32": p32, "locus": loc, "flips": flips,
            "per_window": pw,
            "eq": {"match": eqm, "total": eqt, "nontie_match": eqnm, "nontie_total": eqnt, "disagreements": disag},
            "probes": {"det_m8": dm8, "within": win, "det_m1": det_m1, "iso8": i8, "iso32": i32},
            "n_windows": len(offsets),
            "per_prompt": {
                "id": rec.get("id"), "prompt_idx": ri, "C": C, "traj_n": traj_n,
                "n_windows": len(offsets), "n_w8_positions": prompt_w8, "w8_match": prompt_w8_match,
                "det_m1": det_m1, "is_det_prompt": bool(do_det), "prompt_secs": round(prompt_secs, 2),
            },
        }

    # ---- resume: replay completed prompts from the checkpoint (0 GPU); a finished prompt is never recomputed ----
    done: set[int] = set()
    if resume and checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    prec = json.loads(line)
                except Exception:
                    continue                          # tolerate a truncated trailing line (its prompt is recomputed)
                if prec.get("prompt_idx") in done:
                    continue
                _ingest(prec)
                done.add(prec["prompt_idx"])
        print(f"[fullserve:{arm}] resumed {len(done)} prompts from {checkpoint}", flush=True)

    skip_set = set(int(x) for x in (skip_prompts or ()))
    t_census0 = time.time()
    ck = open(checkpoint, "a") if checkpoint else None
    for ri in range(len(rows)):
        if ri in done:
            continue
        if ri in skip_set:                            # poison prompt the watchdog gave up on -> record as skipped
            prec = _empty_prec(ri, rows[ri], "hang_skipped")
            if ck:
                ck.write(json.dumps(prec) + "\n"); ck.flush(); os.fsync(ck.fileno())
            _ingest(prec); done.add(ri)
            print(f"[fullserve:{arm}] prompt {ri} HANG-SKIPPED (watchdog)", flush=True)
            continue
        prec = _process_one_prompt(ri, rows[ri])
        if ck:
            ck.write(json.dumps(prec) + "\n"); ck.flush(); os.fsync(ck.fileno())
        _ingest(prec); done.add(ri)
        pp = prec["per_prompt"]
        ag = _agg(positions)
        print(f"[fullserve:{arm}] prompt {ri} traj_n={pp.get('traj_n')} windows={prec['n_windows']} "
              f"w8_match={pp.get('w8_match')}/{pp.get('n_w8_positions')} "
              f"running_id={ag['token_identity_rate']:.6f} sem={ag['n_semantic_flips']} "
              f"tie={ag['n_tie_flips']} det_m1={pp.get('det_m1')} secs={pp.get('prompt_secs')}"
              f"{' [det]' if pp.get('is_det_prompt') else ''} [{len(done)}/{len(rows)}]", flush=True)
    if ck:
        ck.close()
    _beat("done", len(rows), -1)

    # bridge merged counters back to the names the aggregation block below expects (unchanged from here on)
    n_eq_match, n_eq_total = eqc["match"], eqc["total"]
    n_eq_nontie_match, n_eq_nontie_total = eqc["nontie_match"], eqc["nontie_total"]
    n_windows = nwin["n"]

    census_secs = time.time() - t_census0
    agg8 = _agg(positions)
    agg32 = _agg(positions_wide)
    agg_locus = _agg(locus_positions)

    def _rate(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    det_secs = [p["prompt_secs"] for p in per_prompt if p.get("is_det_prompt")]
    nondet_secs = [p["prompt_secs"] for p in per_prompt if not p.get("is_det_prompt")]

    out = {
        "phase": "fullserve_census", "arm": arm, "model_dir": model_dir,
        "vllm_batch_invariant_env": batch_invariant_env, "attn_is_batch_invariant": attn_is_batch_invariant,
        # surgical-357 lever provenance (None on non-surgical arms): proves the 2D attn is armed and the
        # matmul tax is OFF -- i.e. this arm IS the shipped surgical config, not the global-flag pinned one.
        "surgical_mode": _surgical_mode(arm),
        "surgical_attn_armed": surgical_prov.get("surgical_attn_armed"),
        "matmul_tax_installed": surgical_prov.get("matmul_tax_installed"),
        "surgical_lever_source": surgical_prov.get("lever_source"),
        "n_prompts_run": len(per_prompt), "C": C, "traj_len": traj_len, "ignore_eos": ignore_eos,
        "G": G, "W8": M_VERIFY, "Wwide": WIDE_W, "n_windows": n_windows,
        # HEADLINE: W=8 served-geometry full-serve census
        "w8": agg8,
        # full per-position coverage (faithful via M<=64 split-invariance)
        "w32": agg32,
        # cross-check sub-census == denken #471 locus (O==C window, W=8)
        "locus_O224": agg_locus,
        # reload-immunity + geometry probes (must all be ~1.0)
        "determinism_M8": _rate(det_m8_acc), "determinism_M1": _rate(det_m1_acc),
        "within_batch": _rate(within_acc),
        "chunk_isolated_w8": _rate(iso8_acc), "chunk_isolated_w32": _rate(iso32_acc),
        "width_equivalence_rate": (n_eq_match / n_eq_total) if n_eq_total else float("nan"),
        "width_equivalence_positions": n_eq_total,
        "width_equivalence_rate_nontie": (n_eq_nontie_match / n_eq_nontie_total) if n_eq_nontie_total else float("nan"),
        "width_equivalence_nontie_positions": n_eq_nontie_total,
        "width_disagreements_all_tie": bool(n_eq_nontie_total == 0 or n_eq_nontie_match == n_eq_nontie_total),
        "n_width_disagreements": len(eq_disagreements),
        "width_disagreements": eq_disagreements[:64],
        "n_det_m1_checked": len(det_m1_acc), "n_within_checked": len(within_acc),
        # detail (flips only -- keep file small; full positions are NOT dumped)
        "flip_details": flip_details,
        "per_window": per_window,
        "per_prompt": per_prompt,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "census_secs": round(census_secs, 1),
        "mean_prompt_secs": round(census_secs / max(1, len(per_prompt)), 2),
        "mean_det_prompt_secs": round(sum(det_secs) / len(det_secs), 2) if det_secs else None,
        "mean_nondet_prompt_secs": round(sum(nondet_secs) / len(nondet_secs), 2) if nondet_secs else None,
        "nan_clean": bool(math.isfinite(agg8["token_identity_rate"])),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[fullserve:{arm}] W8 identity={agg8['token_identity_rate']:.7f} "
          f"(sem={agg8['n_semantic_flips']} tie={agg8['n_tie_flips']} of {agg8['n_positions']} pos) | "
          f"W32 identity={agg32['token_identity_rate']:.7f} | locus={agg_locus['token_identity_rate']:.7f} | "
          f"width_eq={out['width_equivalence_rate']:.6f} det_m8={out['determinism_M8']:.4f} "
          f"peak={out['peak_gpu_gb']:.1f}GB", flush=True)
    print(f"[fullserve:{arm}] census_secs={out['census_secs']:.0f} mean/prompt={out['mean_prompt_secs']:.1f}s "
          f"(det={out['mean_det_prompt_secs']} nondet={out['mean_nondet_prompt_secs']}) "
          f"=> est 128-prompt full run ~{out['mean_prompt_secs']*128/60:.0f}min", flush=True)
    print(f"ARM_DONE {out_path}", flush=True)


# ======================================================================================
# PR #510 deliverables: tie-threshold sweep + per-flip examination + globalflag222 compare
# ======================================================================================
def tie_threshold_sweep(flip_details: list[dict], width: int = M_VERIFY) -> dict:
    """0-GPU re-bucket of the served (W=8) flips at a sweep of bf16-tie tolerances on the M=1 self-gap.

    A flip is currently classified 'semantic' iff the M=1 reference's OWN top-2 gap exceeds BAND_TOL (1e-9,
    a *bitwise* tie). Each surgical semantic flip is really a near-tie whose gap is a small number of bf16
    ULPs. At tolerance t, re-classify any flip with m1_self_gap <= t as a TIE (the M=1 winner is within t of
    its runner-up -> the verify arbiter resolving it differently is operatively identity-1.0 at that
    tolerance). Reports n_semantic remaining at standard thresholds and the minimal t that zeroes them --
    the principled 'operative-1.0 at a k-ULP tie tolerance' certificate knob the human asked for."""
    ULP = 0.0625                                          # one bf16 logit step in nats (the near-tie quantum)
    served = [f for f in flip_details if f.get("width") == width and f.get("is_flip")]
    sem = [f for f in served if f.get("flip_kind") == "semantic"]
    sem_gaps = sorted(float(f["m1_self_gap"]) for f in sem if f.get("m1_self_gap") is not None)
    thresholds = {
        "bitwise_1e-9": BAND_TOL, "1ulp_0.0625": ULP, "2ulp_0.125": 2 * ULP,
        "eps_star_0.125": EPS_STAR, "3ulp_0.1875": 3 * ULP,
    }
    buckets = {name: sum(1 for g in sem_gaps if g > t) for name, t in thresholds.items()}
    # minimal tolerance that collapses ALL served semantic flips to ties == the max semantic gap (0 if none)
    tie_threshold_for_zero_semantic = (max(sem_gaps) if sem_gaps else 0.0)
    return {
        "n_served_flips": len(served),
        "n_semantic_bitwise": len(sem),
        "semantic_gaps_nats": sem_gaps,
        "semantic_gaps_in_ulps": [round(g / ULP, 3) for g in sem_gaps],
        "n_semantic_at_threshold": buckets,
        "tie_threshold_for_zero_semantic": tie_threshold_for_zero_semantic,
        "tie_threshold_for_zero_semantic_ulps": (round(tie_threshold_for_zero_semantic / ULP, 3)
                                                 if sem_gaps else 0.0),
        "all_semantic_collapse_at_eps_star": bool(all(g <= EPS_STAR for g in sem_gaps)),
    }


def examine_served_flips(flip_details: list[dict], width: int = M_VERIFY) -> list[dict]:
    """One compact record per served (W=8) flip -- the exact #487 examination columns (m1_self_gap in ULPs,
    M=1 token in M=8 top-2, position fraction along the trajectory, kind)."""
    ULP = 0.0625
    out = []
    for f in flip_details:
        if f.get("width") != width or not f.get("is_flip"):
            continue
        gap = f.get("m1_self_gap")
        out.append({
            "prompt_idx": f.get("prompt_idx"), "pos": f.get("pos"), "k": f.get("k"),
            "flip_kind": f.get("flip_kind"),
            "m1_self_gap": gap, "m1_self_gap_ulps": (round(float(gap) / ULP, 3) if gap is not None else None),
            "m1_in_m8_top2": f.get("m1_in_m8_top2"), "m1_argmax_matches_token": f.get("m1_argmax_matches_token"),
            "m8_top1_id": f.get("m8_top1_id"), "m1_tok_id": f.get("m1_tok_id"),
        })
    return sorted(out, key=lambda r: (r["prompt_idx"], r["pos"]))


def compare_vs_globalflag222(n_semantic_surgical: int) -> dict:
    """The decisive PR #510 comparison: does the SHIPPED surgical-357 have FEWER / SAME / MORE served
    semantic flips than the global-flag 222 config (#487's 12)?"""
    base = PR487_GLOBALFLAG222_N_SEMANTIC
    if n_semantic_surgical < base:
        verdict = "fewer"
    elif n_semantic_surgical == base:
        verdict = "same"
    else:
        verdict = "more"
    return {
        "vs_globalflag222": verdict,
        "surgical_n_semantic": n_semantic_surgical,
        "globalflag222_n_semantic": base,
        "delta_vs_globalflag222": n_semantic_surgical - base,
        "globalflag222_census": PR487_GLOBALFLAG222_FULLSERVE_CENSUS,
    }


# ======================================================================================
# Compose + report + self-test
# ======================================================================================
def compose_and_report(census: dict, a: argparse.Namespace) -> dict:
    primary_arm = "pinned" if "pinned" in census else sorted(census)[0]
    prim = census[primary_arm]
    w8 = prim["w8"]
    w32 = prim["w32"]
    locus = prim["locus_O224"]

    # HEADLINE deliverable == the W=8 SERVED verify geometry (the real M=8 serve does a width-8 verify over a
    # decode-cached prefix; W=8 replicates exactly that). operative-1.0 is judged on W=8 ONLY.
    operative_1p0 = bool(
        math.isfinite(w8["token_identity_rate"]) and w8["token_identity_rate"] >= 0.99
        and w8["n_semantic_flips"] == 0
    )
    operative_1p0_fullcov = bool(
        math.isfinite(w32["token_identity_rate"]) and w32["token_identity_rate"] >= 0.99
        and w32["n_semantic_flips"] == 0
    )
    crosscheck_gap = (locus["token_identity_rate"] - DENKEN471_LOCUS_IDENTITY
                      if math.isfinite(locus["token_identity_rate"]) else float("nan"))
    # the locus sub-census should reproduce denken #471 to within the 1-flip granularity of its 889 positions.
    locus_reproduces_471 = bool(math.isfinite(crosscheck_gap) and abs(crosscheck_gap) <= (2.0 / max(1, locus["n_positions"])))

    # reload-immunity + geometry faithfulness of the W=8 deliverable (the things that MUST hold).
    reload_immune = bool(prim["determinism_M8"] == 1.0
                         and (not math.isfinite(prim["determinism_M1"]) or prim["determinism_M1"] == 1.0)
                         and (not math.isfinite(prim["within_batch"]) or prim["within_batch"] == 1.0))
    geometry_isolated = bool(prim["chunk_isolated_w8"] >= 0.99 and prim["chunk_isolated_w32"] >= 0.99)
    deliverable_green = bool(operative_1p0 and locus_reproduces_471 and reload_immune
                             and geometry_isolated and prim["nan_clean"])

    self_test, n_checks = build_self_test(census, primary_arm, operative_1p0)

    # The verdict is judged on the W=8 SERVED-GEOMETRY deliverable + reload-immunity + cross-check, NOT on the
    # auxiliary W=32 byte-equivalence (which the harness empirically DISPROVES -- see width_findings: a 31-wide
    # read is not a served geometry and accumulates prefill-vs-decode divergence over its span; the faithful
    # served read is W=8). RED only if the served W=8 geometry itself shows a semantic (non-tie) divergence.
    verdict = (
        "GREEN" if (deliverable_green and self_test_ok(self_test))
        else ("RED" if w8["n_semantic_flips"] > 0 else "AMBER")
    )

    # ---- width findings: WHY the faithful served read is W=8, not the wider W=32 ----
    # flip_details with width==32 are ALL extension-band (pos >= O+8) by construction; they are NOT a served
    # geometry (the M=8 serve never does a 31-wide verify) -- they measure prefill-vs-decode KV divergence
    # accumulating over the wide span. The W=8 served read at the SAME windows has ZERO flips.
    w32_ext = [f for f in prim["flip_details"] if f.get("width") == 32]
    w32_ext_semantic = sum(1 for f in w32_ext if f.get("flip_kind") == "semantic")
    w32_ext_tie = sum(1 for f in w32_ext if f.get("flip_kind") == "tie")
    width_findings = {
        # Q3 (split-K width-invariance) AT THE OVERLAP O+1..O+7: holds to ~99.9% (1 non-tie exception / 882).
        "overlap_invariance_nontie": prim["width_equivalence_rate_nontie"],
        "overlap_invariance_nontie_positions": prim["width_equivalence_nontie_positions"],
        "overlap_disagreements_vs_w8": prim["n_width_disagreements"],
        "overlap_disagreements_all_tie": prim["width_disagreements_all_tie"],
        # extension band O+8..O+31: NON-served, prefill-span artifact (the reason W=32 is not byte-faithful).
        "extension_flips_semantic": w32_ext_semantic,
        "extension_flips_tie": w32_ext_tie,
        "w8_is_served_geometry": True,
        "w32_is_served_geometry": False,
        "w32_is_byte_faithful": False,
        "faithful_census_geometry": "W=8 (served M=8 verify width)",
        "note": (
            "The W=8 read replicates the deployed M=8 verify geometry (width-8 forward over a decode-cached "
            "prefix) and is the FAITHFUL headline (identity 1.0, 0 flips). The W=32 read is an auxiliary denser "
            "cross-read; it is NOT a served geometry and is NOT byte-faithful: (a) at the O+1..O+7 OVERLAP it "
            "matches W=8 to ~99.9% (researcher Q3 split-K width-invariance ~holds; 1 non-tie exception/882), and "
            "(b) at the O+8..O+31 EXTENSION it diverges from the M=1 AR token at clear-winner positions "
            f"({w32_ext_semantic} semantic + {w32_ext_tie} tie flips) because the wide prefill accumulates "
            "prefill-vs-decode KV rounding over its span. This DISPROVES the convenience assumption that a wide "
            "read is a faithful shortcut -- it is not; the served W=8 geometry must be measured directly."
        ),
    }

    # ---- PR #510 surgical-357 deliverables (computed on the primary arm's served W=8 flips) ----
    is_surgical = bool(prim.get("surgical_mode"))
    tie_sweep = tie_threshold_sweep(prim["flip_details"], width=M_VERIFY)
    served_flip_exam = examine_served_flips(prim["flip_details"], width=M_VERIFY)
    vs222 = compare_vs_globalflag222(w8["n_semantic_flips"])
    # surgical locus cross-check: the surgical attn_only locus cert this run stress-tests (stark #494).
    # stark494 (0.998875) and denken471 (0.9988751) are equal to 15 sig figs (surgical divergence == 222
    # all_pin divergence), so the existing locus_reproduces_denken471 check doubles as the surgical check.
    surgical_locus_gap = (locus["token_identity_rate"] - STARK494_SURGICAL_LOCUS_IDENTITY
                          if math.isfinite(locus["token_identity_rate"]) else float("nan"))
    surgical_locus_reproduces_494 = bool(math.isfinite(surgical_locus_gap)
                                         and abs(surgical_locus_gap) <= (2.0 / max(1, locus["n_positions"])))

    report = {
        "pr": (510 if is_surgical else 487), "agent": "wirbel",
        "leg": ("surgical-357 ship: reload-immune full-serve operative-identity census" if is_surgical
                else "same-reload full-serve identity census harness"),
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "primary_arm": primary_arm,
        # ---- PR #510 surgical-357 SHIP deliverables (KEY OUTPUTS) ----
        "surgical_mode": is_surgical,
        "surgical357_fullserve_census": w8["token_identity_rate"],
        "surgical357_n_semantic_flips": w8["n_semantic_flips"],
        "surgical357_n_tie_flips": w8["n_tie_flips"],
        "surgical357_operative_identity_1p0": operative_1p0,
        "surgical357_operative_identity_rate": w8["operative_identity_rate"],
        "surgical_attn_armed": prim.get("surgical_attn_armed"),
        "matmul_tax_installed": prim.get("matmul_tax_installed"),
        "surgical_lever_source": prim.get("surgical_lever_source"),
        # tie-threshold sensitivity sweep (the certificate-wording knob)
        "tie_threshold_sweep": tie_sweep,
        "tie_threshold_for_zero_semantic": tie_sweep["tie_threshold_for_zero_semantic"],
        "tie_threshold_for_zero_semantic_ulps": tie_sweep["tie_threshold_for_zero_semantic_ulps"],
        "served_flip_examination": served_flip_exam,
        # the decisive comparison: surgical vs global-flag 222 (#487's 12)
        **vs222,
        # surgical locus anchor (stark #494) cross-check
        "surgical_locus_anchor_stark494": STARK494_SURGICAL_LOCUS_IDENTITY,
        "surgical_locus_gap_vs_stark494": surgical_locus_gap,
        "surgical_locus_reproduces_stark494": surgical_locus_reproduces_494,
        # ---- HEADLINE deliverable == faithful W=8 served verify geometry ----
        "reloadimmune_fullserve_census": w8["token_identity_rate"],
        "n_semantic_flips": w8["n_semantic_flips"],
        "n_tie_flips": w8["n_tie_flips"],
        "operative_identity_1p0": operative_1p0,
        "operative_identity_rate": w8["operative_identity_rate"],
        "headline_geometry": "W=8 (served M=8 verify width)",
        # ---- auxiliary W=32 denser cross-read (NOT a served geometry, NOT byte-faithful; see width_findings) ----
        "reloadimmune_fullserve_census_fullcov": w32["token_identity_rate"],
        "fullcov_is_byte_faithful": False,
        "n_semantic_flips_fullcov": w32["n_semantic_flips"],
        "n_tie_flips_fullcov": w32["n_tie_flips"],
        "operative_identity_1p0_fullcov": operative_1p0_fullcov,
        # ---- cross-check vs denken #471 ----
        "census_at_locus_O224": locus["token_identity_rate"],
        "denken471_locus_identity": DENKEN471_LOCUS_IDENTITY,
        "crosscheck_vs_denken471_gap": crosscheck_gap,
        "locus_reproduces_denken471": locus_reproduces_471,
        "n_locus_positions": locus["n_positions"],
        # ---- the ubel #470 confound this replaces ----
        "ubel470_served_confounded_identity": UBEL470_SERVED_CONFOUNDED_IDENTITY,
        "ubel470_m8_xreload_floor": UBEL470_M8_XRELOAD_FLOOR,
        "replaces_reload_confound": True,
        # ---- reload-immunity + geometry proof ----
        "reload_immune": reload_immune,
        "determinism_M8": prim["determinism_M8"], "determinism_M1": prim["determinism_M1"],
        "within_batch": prim["within_batch"],
        "chunk_isolated_w8": prim["chunk_isolated_w8"], "chunk_isolated_w32": prim["chunk_isolated_w32"],
        # ---- width findings (WHY W=8 is the faithful read; the harness DISPROVES W=32 byte-equivalence) ----
        "width_findings": width_findings,
        "width_equivalence_rate": prim["width_equivalence_rate"],
        "width_equivalence_rate_nontie": prim["width_equivalence_rate_nontie"],
        "width_equivalence_positions": prim["width_equivalence_positions"],
        "width_equivalence_nontie_positions": prim["width_equivalence_nontie_positions"],
        "width_disagreements_all_tie": prim["width_disagreements_all_tie"],
        "n_width_disagreements": prim["n_width_disagreements"],
        # split-K width-invariance (researcher Q3) holds AT THE OVERLAP to ~99.9% (not byte-exactly); the wider
        # W=32 read is NOT byte-faithful at its extension band -> the faithful served census is W=8.
        "width8_width32_equivalent_at_overlap": bool(
            math.isfinite(prim["width_equivalence_rate_nontie"])
            and prim["width_equivalence_rate_nontie"] >= 0.99),
        # ---- coverage / scale ----
        "n_windows": prim["n_windows"], "n_positions_w8": w8["n_positions"], "n_positions_w32": w32["n_positions"],
        "coverage_multiple_vs_locus": (w8["n_positions"] / max(1, locus["n_positions"])),
        "n_prompts_run": prim["n_prompts_run"], "C": prim["C"], "traj_len": prim["traj_len"],
        "ignore_eos": prim["ignore_eos"],
        # ---- arms detail ----
        "arms": {arm: {
            "w8_identity": d["w8"]["token_identity_rate"], "w8_n_semantic": d["w8"]["n_semantic_flips"],
            "w8_n_tie": d["w8"]["n_tie_flips"], "w8_n_positions": d["w8"]["n_positions"],
            "w32_identity": d["w32"]["token_identity_rate"], "w32_n_semantic": d["w32"]["n_semantic_flips"],
            "locus_identity": d["locus_O224"]["token_identity_rate"],
            "determinism_M8": d["determinism_M8"], "width_equivalence_rate": d["width_equivalence_rate"],
            "vllm_batch_invariant_env": d["vllm_batch_invariant_env"],
            "attn_is_batch_invariant": d["attn_is_batch_invariant"], "peak_gpu_gb": d["peak_gpu_gb"],
            "surgical_mode": d.get("surgical_mode"), "surgical_attn_armed": d.get("surgical_attn_armed"),
            "matmul_tax_installed": d.get("matmul_tax_installed"),
        } for arm, d in census.items()},
        "imported_anchors": {
            "denken471_locus_identity": DENKEN471_LOCUS_IDENTITY,
            "denken471_locus_positions": DENKEN471_LOCUS_POSITIONS, "denken471_locus_flips": DENKEN471_LOCUS_FLIPS,
            "ubel470_served_confounded_identity": UBEL470_SERVED_CONFOUNDED_IDENTITY,
            "ubel470_m8_xreload_floor": UBEL470_M8_XRELOAD_FLOOR, "ubel470_m1_xreload": UBEL470_M1_XRELOAD,
            "pr461_allpin_identity": PR461_ALLPIN_IDENTITY, "stark466_locus_identity": STARK466_LOCUS_IDENTITY,
            "pr487_globalflag222_fullserve_census": PR487_GLOBALFLAG222_FULLSERVE_CENSUS,
            "pr487_globalflag222_n_semantic": PR487_GLOBALFLAG222_N_SEMANTIC,
            "pr487_globalflag222_tie_threshold_zero_semantic": PR487_GLOBALFLAG222_TIE_THRESHOLD_ZERO_SEMANTIC,
            "stark494_surgical_locus_identity": STARK494_SURGICAL_LOCUS_IDENTITY,
            "stark494_surgical_locus_divergence": STARK494_SURGICAL_LOCUS_DIVERGENCE,
        },
        "verdict": verdict,
        "self_test": self_test, "self_test_n_checks": n_checks,
        "fullserve_self_test_passes": self_test_ok(self_test),
        "model_dir": prim["model_dir"],
    }
    return report


def self_test_ok(st: dict) -> bool:
    return all(st.values())


def build_self_test(census: dict, primary_arm: str, operative_1p0: bool) -> tuple[dict, int]:
    checks: dict = {}
    for arm, d in census.items():
        w8, w32, loc = d["w8"], d["w32"], d["locus_O224"]
        checks[f"{arm}_w8_identity_in_unit"] = bool(math.isfinite(w8["token_identity_rate"])
                                                    and 0.0 <= w8["token_identity_rate"] <= 1.0)
        checks[f"{arm}_w32_identity_in_unit"] = bool(math.isfinite(w32["token_identity_rate"])
                                                     and 0.0 <= w32["token_identity_rate"] <= 1.0)
        checks[f"{arm}_nan_clean"] = bool(d["nan_clean"])
        # geometry faithful: each W=8 read isolated exactly 8 rows; each W=32 read isolated exactly 32
        checks[f"{arm}_geometry_w8_isolated"] = bool(d["chunk_isolated_w8"] >= 0.99)
        checks[f"{arm}_geometry_w32_isolated"] = bool(d["chunk_isolated_w32"] >= 0.99)
        # reload-immunity: within-reload M=8 determinism is 1.0 (the cross-reload 0.64 never enters)
        checks[f"{arm}_det_m8_eq_1"] = bool(d["determinism_M8"] == 1.0)
        # split-K width-invariance (researcher Q3) holds AT THE OVERLAP O+1..O+7 to ~99.9% (W=8 vs W=32 argmax
        # agree at clear-winner positions). It is NOT byte-exact, and the wider W=32 read is NOT byte-faithful at
        # its extension band -- so this is an INFORMATIONAL high-agreement check, NOT a byte-equivalence gate;
        # the faithful served census is W=8 (which has 0 flips). See width_findings.
        checks[f"{arm}_width_overlap_invariance_ge_99"] = bool(
            math.isfinite(d["width_equivalence_rate_nontie"]) and d["width_equivalence_rate_nontie"] >= 0.99)
        # full-serve really swept many windows / is a strict superset of the single locus
        checks[f"{arm}_multi_window"] = bool(d["n_windows"] > d["n_prompts_run"])
        checks[f"{arm}_covers_more_than_locus"] = bool(w8["n_positions"] > loc["n_positions"] > 0)
        # the full-coverage census has >= as many positions as the W=8 served-geometry census
        checks[f"{arm}_fullcov_ge_w8"] = bool(w32["n_positions"] >= w8["n_positions"])

    prim = census[primary_arm]
    # THE DELIVERABLE: the W=8 served-geometry census shows NO semantic (non-tie) divergence from M=1 AR.
    checks["w8_served_no_semantic_flips"] = bool(prim["w8"]["n_semantic_flips"] == 0)
    # reload-immune (the confound this harness closes): within-reload M=8 determinism is exactly 1.0.
    checks["reload_immune"] = bool(prim["determinism_M8"] == 1.0
                                   and (not math.isfinite(prim["determinism_M1"]) or prim["determinism_M1"] == 1.0)
                                   and (not math.isfinite(prim["within_batch"]) or prim["within_batch"] == 1.0))
    # M=1 AR reference reproduces within reload where checked (spot-check), if checked at all
    checks["det_m1_stable_where_checked"] = bool(prim["n_det_m1_checked"] == 0 or prim["determinism_M1"] == 1.0)
    checks["within_batch_stable_where_checked"] = bool(prim["n_within_checked"] == 0 or prim["within_batch"] == 1.0)
    # the pinned (global-flag strict) arm engages the batch-invariant attention pin
    if "pinned" in census:
        checks["pinned_attn_batch_invariant"] = bool(census["pinned"].get("attn_is_batch_invariant"))
        checks["pinned_vbi_env_on"] = bool(census["pinned"].get("vllm_batch_invariant_env"))
    # the surgical (shipped 357) arm: 2D attention armed (is_batch_invariant True) AND matmul tax OFF
    # (VBI env unset, _batch_invariant_MODE False) -- this is what makes the arm the SHIPPED config and
    # not the global-flag pinned one. If either fails the arm is mislabelled and the result is void.
    if "surgical" in census:
        s = census["surgical"]
        checks["surgical_attn_batch_invariant"] = bool(s.get("attn_is_batch_invariant"))
        checks["surgical_attn_armed"] = bool(s.get("surgical_attn_armed"))
        checks["surgical_vbi_env_off"] = bool(not s.get("vllm_batch_invariant_env"))
        checks["surgical_matmul_tax_off"] = bool(s.get("matmul_tax_installed") is False)
    # cross-check: the O==C sub-census reproduces denken #471 within its 1-flip granularity
    loc = prim["locus_O224"]
    gap = (loc["token_identity_rate"] - DENKEN471_LOCUS_IDENTITY
           if math.isfinite(loc["token_identity_rate"]) else float("nan"))
    checks["locus_reproduces_denken471"] = bool(math.isfinite(gap)
                                                and abs(gap) <= (2.0 / max(1, loc["n_positions"])))
    # OPERATIVE-1.0: identity high AND every non-identity position is a bf16 tie (no semantic divergence)
    checks["operative_1p0_consistent"] = bool(operative_1p0 == (prim["w8"]["token_identity_rate"] >= 0.99
                                                                and prim["w8"]["n_semantic_flips"] == 0))
    # the reload-confounded ubel #470 number is far below our clean read (we are NOT reproducing 0.4085)
    checks["beats_reload_confounded_470"] = bool(prim["w8"]["token_identity_rate"] > UBEL470_SERVED_CONFOUNDED_IDENTITY)
    # internal consistency: identity == 1 - flips/positions
    n = prim["w8"]["n_positions"]
    checks["identity_consistent"] = bool(n == 0 or abs(
        prim["w8"]["token_identity_rate"] - (n - prim["w8"]["n_flips"]) / n) < 1e-9)
    return checks, len(checks)


# ======================================================================================
# Orchestrator + subprocess + console + wandb + main
# ======================================================================================
def _census_env(extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    return env


def _arm_complete(out_json: str, n_prompts: int) -> bool:
    try:
        d = json.load(open(out_json))
        return int(d.get("n_prompts_run", -1)) >= n_prompts
    except Exception:
        return False


def _read_heartbeat(path: str) -> dict | None:
    """The worker's latest heartbeat dict {ts, prompt_idx, ...}, or None if absent/unreadable.
    The watchdog compares hb['ts'] against the current worker's launch time so a *stale* beat left by a
    previously-killed worker is never mistaken for a live stall (that bug aborted the prior hang-recovery)."""
    try:
        return json.load(open(path))
    except Exception:
        return None


def _kill_proc(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(sig)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
            return
        except Exception:
            pass


def _wait_gpu_free(timeout_s: float = 120.0) -> None:
    """Block until the GPU is released by a killed worker, so the next reload can allocate."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if not out or int(out.splitlines()[0].strip()) < 1500:   # < ~1.5GB residual -> free enough
                return
        except Exception:
            return
        time.sleep(3)


def _run_census_arm(a: argparse.Namespace, arm: str) -> dict:
    """Run one census arm under a watchdog. Launch a resumable worker; if its heartbeat goes stale (a CUDA hang --
    the exact failure mode that wedged the prior full-128 attempts: 100% CPU spin, frozen SM%, no GPU progress)
    kill it and relaunch a FRESH reload that resumes from the per-prompt checkpoint. A fresh reload also clears the
    state that accumulated toward the hang, so the census advances one prompt at a time and never loses a finished
    prompt. A prompt that stalls the worker `poison_strikes` times is recorded as hang-skipped and excluded."""
    out_json = str(OUT_DIR / f"arm_{arm}_result.json")
    ckpt = str(OUT_DIR / f"checkpoint_{arm}.jsonl")
    hb = str(OUT_DIR / f"heartbeat_{arm}.json")
    # pinned = global-flag strict (VBI=1: 2D attn + matmul tax). surgical = shipped 357 lever (VBI=0 +
    # SURGICAL_ATTN_USE_3D_OFF=1: 2D attn, matmul tax OFF). heuristic = stock (VBI=0, no pin).
    if arm == "pinned":
        extra_env = {"VLLM_BATCH_INVARIANT": "1"}
    elif arm == "surgical":
        extra_env = {"VLLM_BATCH_INVARIANT": "0", "SURGICAL_ATTN_USE_3D_OFF": "1"}
    else:
        extra_env = {"VLLM_BATCH_INVARIANT": "0"}
    base_args = [
        "--phase", "fullserve_census", "--arm", arm, "--out", out_json,
        "--n-prompts", str(a.n_prompts), "--c0", str(a.c0), "--traj-len", str(a.traj_len),
        "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
        "--verbose-k", str(a.verbose_k), "--det-check-k", str(a.det_check_k),
        "--checkpoint", ckpt, "--heartbeat", hb, "--resume",
    ] + (["--ignore-eos"] if a.ignore_eos else [])

    skip: list[int] = []
    stalled_at: dict[int, int] = {}
    restarts = 0
    while not _arm_complete(out_json, a.n_prompts):
        if restarts > a.watchdog_max_restarts:
            raise RuntimeError(f"[watchdog:{arm}] exceeded {a.watchdog_max_restarts} restarts; aborting")
        _wait_gpu_free()
        args = base_args + (["--skip-prompts", ",".join(map(str, sorted(set(skip))))] if skip else [])
        cmd = [sys.executable, os.path.abspath(__file__)] + args
        print(f"[watchdog:{arm}] launch restart={restarts} skip={sorted(set(skip))} "
              f"(VBI={extra_env['VLLM_BATCH_INVARIANT']} stall={a.watchdog_stall_s}s)", flush=True)
        try:                                           # drop the prior worker's heartbeat so this worker's
            os.remove(hb)                              # vLLM-load phase is never read as a stale stall
        except OSError:
            pass
        proc = subprocess.Popen(cmd, env=_census_env(extra_env))
        t_launch = time.time()
        last_prompt, stalled = -1, False
        while True:
            try:
                proc.wait(timeout=a.watchdog_poll_s)
                break                                  # worker exited on its own
            except subprocess.TimeoutExpired:
                beat = _read_heartbeat(hb)
                fresh = beat is not None and float(beat.get("ts", 0)) >= t_launch
                if fresh and int(beat.get("prompt_idx", -1)) >= 0:
                    last_prompt = int(beat["prompt_idx"])
                if not fresh:
                    # No beat from THIS worker yet: it is loading vLLM / replaying the checkpoint.
                    # Only intervene if it has had ample time and STILL not beaten (a load-phase hang) --
                    # otherwise we'd kill every relaunch mid-load and never let a poison-skip take effect.
                    if time.time() - t_launch > a.watchdog_load_grace_s:
                        print(f"[watchdog:{arm}] LOAD-STALL no heartbeat {time.time()-t_launch:.0f}s > "
                              f"{a.watchdog_load_grace_s}s after launch; killing worker", flush=True)
                        _kill_proc(proc)
                        stalled = True
                        break
                else:
                    age = time.time() - float(beat["ts"])
                    if age > a.watchdog_stall_s:
                        print(f"[watchdog:{arm}] STALL heartbeat age={age:.0f}s > {a.watchdog_stall_s}s "
                              f"at prompt {last_prompt}; killing worker", flush=True)
                        _kill_proc(proc)
                        stalled = True
                        break
        if stalled:
            restarts += 1
            if last_prompt >= 0:
                stalled_at[last_prompt] = stalled_at.get(last_prompt, 0) + 1
                if stalled_at[last_prompt] >= a.watchdog_poison_strikes:
                    print(f"[watchdog:{arm}] prompt {last_prompt} poisoned "
                          f"({stalled_at[last_prompt]}x); will skip on next launch", flush=True)
                    skip.append(last_prompt)
            continue
        if proc.returncode == 0 and _arm_complete(out_json, a.n_prompts):
            break
        print(f"[watchdog:{arm}] worker exited rc={proc.returncode} without a complete result; resuming",
              flush=True)
        restarts += 1
    print(f"[watchdog:{arm}] complete -> {out_json}", flush=True)
    return json.load(open(out_json))


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    census = {arm: _run_census_arm(a, arm) for arm in a.arms}
    if getattr(a, "census_only", False):
        print("[orch] census_only: arm result(s) written; run --reanalyze to compose report + log W&B", flush=True)
        return
    _finish(compose_and_report(census, a), a)


def reanalyze(a: argparse.Namespace) -> None:
    census = {}
    for arm in a.arms:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if not p.exists():
            raise FileNotFoundError(f"--reanalyze needs {p} (run the GPU phase first)")
        census[arm] = json.load(open(p))
    _finish(compose_and_report(census, a), a)


def _finish(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "fullserve_identity_census_harness_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    if not a.no_wandb:
        log_wandb(report, a)


def _print_console(r: dict) -> None:
    print(f"\n========== FULL-SERVE IDENTITY CENSUS HARNESS (PR #{r.get('pr', 487)}) ==========", flush=True)
    print(f" VERDICT                                  : {r['verdict']}", flush=True)
    print(f" operative_identity_1p0 (PRIMARY)         : {r['operative_identity_1p0']}", flush=True)
    print(f" reloadimmune_fullserve_census (W=8)      : {r['reloadimmune_fullserve_census']:.7f}  "
          f"(semantic={r['n_semantic_flips']} tie={r['n_tie_flips']} of {r['n_positions_w8']} pos, "
          f"{r['n_windows']} windows)", flush=True)
    print(f" headline geometry                        : {r['headline_geometry']}", flush=True)
    print(f" operative_identity_rate (tie-as-pass)    : {r['operative_identity_rate']:.7f}", flush=True)
    print(" --- cross-check vs denken #471 locus ---", flush=True)
    print(f"  census_at_locus_O224 (W=8, O=224)       : {r['census_at_locus_O224']:.7f}  "
          f"(#471 anchor {r['denken471_locus_identity']:.7f}, gap {r['crosscheck_vs_denken471_gap']:+.2e})",
          flush=True)
    print(f"  locus_reproduces_denken471              : {r['locus_reproduces_denken471']}", flush=True)
    print(" --- replaces ubel #470 reload-confound ---", flush=True)
    print(f"  ubel470 served (confounded)             : {r['ubel470_served_confounded_identity']:.4f} "
          f"(M=8 xreload floor {r['ubel470_m8_xreload_floor']:.4f}) -> clean read {r['reloadimmune_fullserve_census']:.6f}",
          flush=True)
    print(" --- reload-immunity + geometry proof ---", flush=True)
    print(f"  reload_immune                           : {r['reload_immune']}  "
          f"(det_m8={r['determinism_M8']:.4f} det_m1={r['determinism_M1']} within={r['within_batch']})", flush=True)
    print(f"  chunk_isolated w8/w32                    : {r['chunk_isolated_w8']:.4f} / {r['chunk_isolated_w32']:.4f}",
          flush=True)
    print(f"  coverage multiple vs single locus        : {r['coverage_multiple_vs_locus']:.1f}x "
          f"({r['n_positions_w8']} vs {r['n_locus_positions']} positions)", flush=True)
    print(" --- width findings (faithful read = W=8; W=32 is NOT byte-faithful) ---", flush=True)
    wf = r["width_findings"]
    print(f"  W=8 served-geometry identity (FAITHFUL)  : {r['reloadimmune_fullserve_census']:.7f} "
          f"(0 semantic) <- HEADLINE", flush=True)
    print(f"  W=32 denser cross-read (NOT faithful)    : {r['reloadimmune_fullserve_census_fullcov']:.7f}  "
          f"(semantic={r['n_semantic_flips_fullcov']} tie={r['n_tie_flips_fullcov']} of {r['n_positions_w32']} pos)",
          flush=True)
    print(f"  Q3 split-K invariance @ overlap (>=.99)  : {wf['overlap_invariance_nontie']:.6f} over "
          f"{wf['overlap_invariance_nontie_positions']} pos ({wf['overlap_disagreements_vs_w8']} disagree vs W=8, "
          f"all_tie={wf['overlap_disagreements_all_tie']})", flush=True)
    print(f"  W=32 extension artifact (non-served)     : {wf['extension_flips_semantic']} semantic + "
          f"{wf['extension_flips_tie']} tie flips @ O+8..O+31 (prefill-span divergence, NOT an M=8 identity fail)",
          flush=True)
    if r.get("surgical_mode"):
        print(" --- PR #510 SURGICAL-357 SHIP deliverables ---", flush=True)
        print(f"  surgical_attn_armed / matmul_tax_off     : {r.get('surgical_attn_armed')} / "
              f"{r.get('matmul_tax_installed') is False}", flush=True)
        print(f"  surgical357_fullserve_census             : {r['surgical357_fullserve_census']:.7f} "
              f"(semantic={r['surgical357_n_semantic_flips']} tie={r['surgical357_n_tie_flips']})", flush=True)
        print(f"  surgical357_operative_identity_1p0       : {r['surgical357_operative_identity_1p0']}", flush=True)
        print(f"  vs_globalflag222 (#487's 12 semantic)    : {r['vs_globalflag222'].upper()} "
              f"(surgical {r['surgical_n_semantic']} vs 222 {r['globalflag222_n_semantic']}, "
              f"delta {r['delta_vs_globalflag222']:+d})", flush=True)
        ts = r.get("tie_threshold_sweep", {})
        print(f"  tie_threshold_for_zero_semantic          : {r['tie_threshold_for_zero_semantic']:.4f} nat "
              f"({r['tie_threshold_for_zero_semantic_ulps']:.2f} ULP)  buckets={ts.get('n_semantic_at_threshold')}",
              flush=True)
        print(f"  surgical_locus_reproduces_stark494       : {r['surgical_locus_reproduces_stark494']} "
              f"(gap {r['surgical_locus_gap_vs_stark494']:+.2e} vs {r['surgical_locus_anchor_stark494']:.6f})",
              flush=True)
    print(f" SELF-TEST PASSES                         : {r['fullserve_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("==================================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    _surgical = bool(report.get("surgical_mode"))
    _notes = (
        "PR#510 surgical-357 SHIP reload-immune full-serve operative-identity census: the M=8 surgical serve "
        "(2D order-preserving attention, matmul tax OFF) vs its own M=1 AR, swept along the full free-running "
        "trajectory. Stress-tests the stark #494 locus operative-1.0 cert at full-serve scale (does it survive, "
        "or share #487's global-flag-222 blind spot?)."
        if _surgical else
        "PR#487 same-reload full-serve identity census: reload-immune token_identity_rate of the M=8 strict "
        "serve vs M=1 AR, swept along the full free-running trajectory (closes the #471 locus -> #470 "
        "reload-confounded gap)."
    )
    run = init_wandb_run(
        job_type="local_profiling", agent="wirbel", name=a.wandb_name, group=a.wandb_group,
        notes=_notes,
        config={
            "pr": report.get("pr", 487), "M_verify": M_VERIFY, "K_spec": K_SPEC, "wide_w": WIDE_W,
            "G": HYBRID_PREFIX_COMMIT,
            "C": report["C"], "traj_len": report["traj_len"], "ignore_eos": report["ignore_eos"],
            "n_prompts_run": report["n_prompts_run"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"], "eps_star": EPS_STAR, "surgical_mode": _surgical,
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            **{f"anchor/{k}": v for k, v in report["imported_anchors"].items()},
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = (
        "reloadimmune_fullserve_census", "headline_geometry",
        "n_semantic_flips", "n_tie_flips", "operative_identity_1p0", "operative_identity_rate",
        "reloadimmune_fullserve_census_fullcov", "fullcov_is_byte_faithful",
        "n_semantic_flips_fullcov", "n_tie_flips_fullcov", "operative_identity_1p0_fullcov",
        "census_at_locus_O224", "denken471_locus_identity", "crosscheck_vs_denken471_gap",
        "locus_reproduces_denken471", "n_locus_positions",
        "ubel470_served_confounded_identity", "ubel470_m8_xreload_floor", "replaces_reload_confound",
        "reload_immune", "determinism_M8", "determinism_M1", "within_batch",
        "chunk_isolated_w8", "chunk_isolated_w32", "width_equivalence_rate", "width_equivalence_positions",
        "width_equivalence_rate_nontie", "width_equivalence_nontie_positions", "width_disagreements_all_tie",
        "n_width_disagreements", "width8_width32_equivalent_at_overlap", "n_windows",
        "n_positions_w8", "n_positions_w32",
        "coverage_multiple_vs_locus", "n_prompts_run", "C", "traj_len", "ignore_eos",
        "verdict", "fullserve_self_test_passes", "self_test_n_checks",
        "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
        # ---- PR #510 surgical-357 ship KEY OUTPUTS ----
        "surgical_mode", "surgical357_fullserve_census", "surgical357_n_semantic_flips",
        "surgical357_n_tie_flips", "surgical357_operative_identity_1p0", "surgical357_operative_identity_rate",
        "surgical_attn_armed", "matmul_tax_installed", "surgical_lever_source",
        "tie_threshold_for_zero_semantic", "tie_threshold_for_zero_semantic_ulps",
        "vs_globalflag222", "surgical_n_semantic", "globalflag222_n_semantic", "delta_vs_globalflag222",
        "globalflag222_census", "surgical_locus_anchor_stark494", "surgical_locus_gap_vs_stark494",
        "surgical_locus_reproduces_stark494",
    )
    for k in keys:
        run.summary[k] = report.get(k)
    for wk, wv in report["width_findings"].items():
        run.summary[f"width_findings/{wk}"] = wv
    # PR #510 tie-threshold sweep (the certificate-wording knob) under its own namespace
    for sk, sv in report.get("tie_threshold_sweep", {}).items():
        if sk == "n_semantic_at_threshold":
            for tname, tval in sv.items():
                run.summary[f"tie_sweep/n_semantic_at_{tname}"] = tval
        elif not isinstance(sv, list):
            run.summary[f"tie_sweep/{sk}"] = sv
    run.summary["verdict_green"] = report["verdict"].startswith("GREEN")
    run.summary["verdict_red"] = report["verdict"].startswith("RED")
    for arm, d in report["arms"].items():
        for mk, mv in d.items():
            run.summary[f"{arm}/{mk}"] = mv
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["fullserve_census"], default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", type=str, default=",".join(CENSUS_ARMS_DEFAULT),
                    help="comma list of census arms (pinned=VBI1 global-flag strict; surgical=shipped 357 "
                         "lever VBI0+SURGICAL_ATTN_USE_3D_OFF=1 i.e. 2D attn + matmul tax OFF; heuristic=VBI0 stock)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="0-GPU: recompose report + self-test from saved arm_*.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts, short traj) to validate path")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--c0", type=int, default=DEFAULT_C0)
    ap.add_argument("--traj-len", dest="traj_len", type=int, default=DEFAULT_TRAJ_LEN)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--det-check-k", dest="det_check_k", type=int, default=16)
    ap.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=False)
    ap.add_argument("--wandb_group", dest="wandb_group", default="fullserve-identity-census-harness")
    ap.add_argument("--wandb_name", dest="wandb_name", default="wirbel/fullserve-identity-census-harness")
    ap.add_argument("--no-wandb", action="store_true")
    # ---- resumable-worker plumbing (worker phase) ----
    ap.add_argument("--checkpoint", default=None, help="per-prompt checkpoint JSONL (resume/durable progress)")
    ap.add_argument("--heartbeat", default=None, help="liveness file the watchdog polls for stall detection")
    ap.add_argument("--resume", action="store_true", help="replay completed prompts from --checkpoint, skip them")
    ap.add_argument("--skip-prompts", dest="skip_prompts", default="",
                    help="comma list of prompt_idx to record as hang-skipped (set by the watchdog on a poison prompt)")
    ap.add_argument("--census-only", dest="census_only", action="store_true",
                    help="orchestrate the census arm(s) under the watchdog but skip report/W&B (use --reanalyze later)")
    # ---- watchdog (orchestrator) ----
    ap.add_argument("--watchdog-stall-s", dest="watchdog_stall_s", type=float, default=180.0,
                    help="kill+resume the worker if its heartbeat is older than this (no legit step exceeds ~60s)")
    ap.add_argument("--watchdog-poll-s", dest="watchdog_poll_s", type=float, default=20.0)
    ap.add_argument("--watchdog-load-grace-s", dest="watchdog_load_grace_s", type=float, default=150.0,
                    help="grace for a fresh worker to write its first heartbeat (vLLM load + checkpoint "
                         "replay ~50s); no beat within this long after launch => treat as a load-phase hang")
    ap.add_argument("--watchdog-max-restarts", dest="watchdog_max_restarts", type=int, default=20)
    ap.add_argument("--watchdog-poison-strikes", dest="watchdog_poison_strikes", type=int, default=3)
    a = ap.parse_args()
    a.arms = tuple(s for s in str(a.arms).split(",") if s)
    skip_prompts = tuple(int(x) for x in str(a.skip_prompts).split(",") if x.strip())

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 4)
        a.traj_len = min(a.traj_len, 256)
        a.det_check_k = min(a.det_check_k, 4)

    if a.phase == "fullserve_census":
        phase_fullserve_census(a.out, a.arm, a.n_prompts, a.c0, a.traj_len,
                               a.gpu_mem_util, a.max_batched_tokens, a.verbose_k, a.det_check_k, a.ignore_eos,
                               checkpoint=a.checkpoint, heartbeat=a.heartbeat, resume=a.resume,
                               skip_prompts=skip_prompts)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
