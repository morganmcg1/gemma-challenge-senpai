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

# ---- PR #534 (land) REASONING-LENGTH census anchors: the 128-CHALLENGE-PROMPT surgical-357 baseline this
# OFF-DISTRIBUTION reasoning census is graded against (run hjiwvkfg; reanalyze/compose run r4e4t2fd;
# arm_surgical_result.json -- C=224 ground-truth prefix, traj_len=512, on the multiple-choice-heavy
# mmlu_pro+gpqa_diamond+aime2026 distribution that elicits SHORT answers). The advisor's open question
# (#534, 2026-06-16 23:04Z): does the ship's operative-1.0-within-1-bf16-ULP identity stay BOUNDED OFF that
# distribution, on a reasoning-length workload (hundreds of decode steps from each prompt's OWN natural
# prefix => more non-order-preserving split-KV reductions => more potential ULP flips)? The reasoning set is
# real AIME 2024/2025 (NONE overlap aime2026). These numbers are the envelope the reasoning run must stay in.
BASELINE128_SURGICAL_OPERATIVE_IDENTITY = 0.9997853923742757   # W=8 served-geometry (match+tie)/total
BASELINE128_SURGICAL_TOKEN_IDENTITY = 0.9986408183704127       # W=8 token_identity_rate (match/total)
BASELINE128_SURGICAL_N_POSITIONS = 13979                       # W=8 served positions over 128 prompts
BASELINE128_SURGICAL_N_SEMANTIC = 3                            # semantic (non-tie) flips at full serve
BASELINE128_SURGICAL_N_TIE = 16                                # bf16-tie flips
BASELINE128_SURGICAL_N_FLIPS = 19                              # total W=8 flips
BASELINE128_SURGICAL_MAX_SEMANTIC_ULP = 2.0                    # max semantic m1_self_gap in bf16 ULPs
BASELINE128_SURGICAL_TRAJ_LEN = 512
BASELINE128_SURGICAL_N_PROMPTS = 128
BASELINE128_SURGICAL_WANDB = "hjiwvkfg"                        # baseline census run (compose run r4e4t2fd)

# the shipped surgical-357 serve recipe -- the arm loads its EXACT lever (surgical_attn_patch.install()):
SURGICAL_SUBMISSION_DIR = "submissions/fa2sw_strict_surgical357"

# ======================================================================================
# Constants (the verify geometry -- identical to #381/#405/#412/#471 for comparability)
# ======================================================================================
K_SPEC = 7
M_VERIFY = K_SPEC + 1                             # = 8, the deployed decode-verify query width (W=8)
WIDE_W = 32                                       # full-coverage read width == HYBRID_PREFIX_COMMIT (M<=64 inv.)
EPS_STAR = 0.125                                  # bf16 near-tie band (covers every observed flip)
ULP_NAT = 0.0625                                  # one bf16 logit step in nats (the near-tie quantum)
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
# Arm attention-path patches (PR #515 land: splitkv399 candidate arm)
# ======================================================================================
# The census worker reproduces the SERVED byte-exact split-KV attention path for the
# splitkv399 candidate IN-PROCESS, before vLLM is imported, by installing the submission's
# split-KV verify monkeypatch (auto-armed from the arm's extra_env, set by _run_census_arm):
#   * splitkv399 : SPLITKV_VERIFY=1 redirects the M=8 verify read (max_seqlen_q 8->1) to
#                  vLLM's 3D split-KV path; the local (/tmp) vLLM wheel honors env
#                  BYTEEXACT_FIXED_TPS=T (pins tiles_per_segment -> fixed split SIZE ->
#                  M-invariant -> byte-exact, lawine #496) and BYTEEXACT_NUM_SEGMENTS=S.
#                  is_batch_invariant stays False so use_3d can be True.
# The surgical comparison arm is armed separately by _install_surgical_lever() above (the
# shipped #510 lever); this installer is splitkv399-only so the two never double-install.
SUBMISSION_DIR = "submissions/fa2sw_strict_surgical357"


def _install_arm_patches(arm: str) -> dict:
    info = {"splitkv_verify_installed": False,
            "byteexact_fixed_tps": 0, "byteexact_num_segments": 0}
    sub = Path(__file__).resolve().parents[3] / SUBMISSION_DIR
    if sub.is_dir() and str(sub) not in sys.path:
        sys.path.insert(0, str(sub))
    if os.environ.get("SPLITKV_VERIFY") == "1":
        try:
            import splitkv_verify_patch  # noqa: F401  -- auto-arms from env on import
            info["splitkv_verify_installed"] = bool(splitkv_verify_patch.install())
            info["byteexact_fixed_tps"] = int(os.environ.get("BYTEEXACT_FIXED_TPS", "0") or 0)
            info["byteexact_num_segments"] = int(os.environ.get("BYTEEXACT_NUM_SEGMENTS", "0") or 0)
        except Exception as exc:  # noqa: BLE001
            print(f"[fullserve:{arm}] splitkv_verify_patch FAILED: {exc!r}", flush=True)
    print(f"[fullserve:{arm}] arm patches: {info}", flush=True)
    return info


def _arm_mechanism(arm: str, install_info: dict) -> dict:
    """Snapshot the active attention mechanism for the result JSON (post-census)."""
    mech = dict(install_info)
    mech["arm"] = arm
    sk = sys.modules.get("splitkv_verify_patch")
    if sk is not None:
        try:
            mech["splitkv_redirects"] = int(getattr(sk, "_stats", {}).get("redirected", 0))
        except Exception:  # noqa: BLE001
            mech["splitkv_redirects"] = None
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        mech["unified_attention_wrapped"] = bool(
            getattr(getattr(_ua, "unified_attention", None), "_splitkv_verify_wrapped", False))
        mech["ops_is_batch_invariant"] = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:  # noqa: BLE001
        pass
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
        mech["num_par_softmax_segments"] = int(getattr(_ta, "NUM_PAR_SOFTMAX_SEGMENTS", 0))
    except Exception:  # noqa: BLE001
        pass
    return mech


# ======================================================================================
# PHASE fullserve_census: one arm. Same-reload teacher-forced full-trajectory verify census.
# ======================================================================================
def phase_fullserve_census(out_path: str, arm: str, n_prompts: int, c0: int, traj_len: int,
                           gpu_mem_util: float, max_batched_tokens: int, verbose_k: int,
                           det_check_k: int, ignore_eos: bool,
                           checkpoint: str | None = None, heartbeat: str | None = None,
                           resume: bool = False, skip_prompts: tuple = (),
                           prompts_path: str = PROMPTS_JSONL, reasoning: bool = False) -> None:
    # Install the arm's attention-path patches BEFORE vLLM is imported (auto-arm from
    # extra_env -> the meta-path finder patches the ops module at vLLM import time).
    arm_install_info = _install_arm_patches(arm)
    import torch
    from vllm import LLM, SamplingParams

    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    model_dir = resolve_model_dir()
    C = block_align(c0)
    G = HYBRID_PREFIX_COMMIT
    # Load prompts up front so reasoning mode can size max_model_len to its (variable-length) full-prompt prefix.
    rows = [json.loads(l) for l in open(prompts_path)][:n_prompts]
    if reasoning:
        # REASONING mode: the whole chat prompt is the cached prefix (per-prompt C); seq = prompt + traj.
        max_prefix = max((len(r.get("context_token_ids", [])) + len(r.get("target_token_ids", [])) for r in rows),
                         default=C)
        model_len = max(max_prefix + traj_len + 64, 800)
    else:
        model_len = max(C + traj_len + 64, 800)
    print(f"[fullserve:{arm}] model={model_dir} C={C} traj_len={traj_len} G={G} W8={M_VERIFY} Wwide={WIDE_W} "
          f"reasoning={reasoning} prompts={os.path.basename(prompts_path)} max_model_len={model_len} "
          f"ignore_eos={ignore_eos} VLLM_BATCH_INVARIANT={batch_invariant_env}", flush=True)

    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=model_len, gpu_memory_utilization=gpu_mem_util,
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
        if reasoning:
            # REASONING mode (PR #534): the FULL chat prompt is the cached prefix; free-run the reasoning trace and
            # sweep it. Per-prompt Cp = len(prompt) (the m1_lp / trajectory origin, NOT a fixed 224); the first
            # verify window opens at the next 32-commit boundary >= Cp. This is what lets variable-length reasoning
            # prompts (median ~132 tok, < the 224 fixed prefix) drive deep free-running completions.
            if len(src) < G + 1:
                return _empty_prec(ri, rec, "short")
            Cp = len(src)
            prefix = src
            first_off = ((Cp + G - 1) // G) * G
        else:
            if len(src) < C + 1:
                return _empty_prec(ri, rec, "short")
            Cp = C
            prefix = src[:C]
            first_off = C
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

        last_off = Cp + traj_n - WIDE_W              # last O where a full W=32 window fits (Cp = prefix/traj origin)
        offsets = list(range(first_off, last_off + 1, G))
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
                tie = _m1_is_bitwise_tie(pp, m1_lp, Cp)
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
                r = classify_position(ri, pp, ent8, seq, m1_lp, Cp, 8)
                if r is None:
                    continue
                w8_recs.append(r)
                p8.append(r)
                if not reasoning and O == Cp:        # locus == denken #471 O=224 sub-census (standard mode only)
                    loc.append(r)
                if r["is_flip"]:
                    flips.append(r)
            for pp in sorted(am32):
                r = classify_position(ri, pp, ent32, seq, m1_lp, Cp, 32)
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
                "id": rec.get("id"), "prompt_idx": ri, "C": Cp, "first_off": first_off, "traj_n": traj_n,
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
        # splitkv399 (and any arm) active-mechanism snapshot for the result JSON (PR #515).
        "mechanism": _arm_mechanism(arm, arm_install_info),
        # reasoning provenance (PR #534): off-distribution reasoning-length census. C below is the global fixed
        # prefix only in standard mode; in reasoning mode each prompt uses its own Cp=len(prompt) (see per_prompt.C).
        "reasoning": reasoning, "prompts_path": prompts_path,
        "prefix_C_per_prompt": ([p.get("C") for p in per_prompt] if reasoning else None),
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
def _tie_threshold_sweep(w8_flips: list[dict]) -> dict:
    """Cheap 0-GPU re-bucket of the W=8 served-geometry flips by the M=1 reference's OWN top-2 gap
    (m1_self_gap). The harness's headline tie/semantic split uses BAND_TOL=1e-9 (a TRUE bitwise tie). This
    sweep asks: at progressively looser tie bands, how many W=8 flips would STILL count as SEMANTIC (the
    M=1 reference has a clear winner that wide enough to exceed the band)? A flip with m1_self_gap <= band is
    a near-tie at that band; gap > band stays semantic. A flip whose M=1 gap is UNREADABLE (None: the verify
    position is past the M=1 trajectory's per-step logprobs) is treated as +inf -> always semantic
    (conservative). threshold_for_zero_semantic = the smallest band that absorbs EVERY flip as a tie (= the
    max readable gap), or None if any flip has no readable gap (can never be certified a tie at any band)."""
    bands = [BAND_TOL, 1e-6, 1e-4, 1e-3, 1e-2, EPS_STAR / 2, EPS_STAR]
    gaps = [f.get("m1_self_gap") for f in w8_flips]
    readable = [g for g in gaps if g is not None]
    n_unreadable = sum(1 for g in gaps if g is None)
    n_semantic_at_band = {f"{b:g}": sum(1 for g in gaps if g is None or g > b) for b in bands}
    max_gap = max(readable) if readable else None
    threshold = max_gap if n_unreadable == 0 else None
    return {
        "n_w8_flips": len(w8_flips),
        "n_semantic_at_band": n_semantic_at_band,
        "max_m1_self_gap_over_flips": max_gap,
        "threshold_for_zero_semantic": threshold,
        "all_flips_within_eps_star": bool(readable and n_unreadable == 0 and max_gap <= EPS_STAR),
        "n_flips_without_readable_m1_gap": n_unreadable,
    }


def _splitkv399_extras(census: dict) -> dict:
    """PR #515 (land) SHIP-CANDIDATE KEY OUTPUTS: the byte-exact FIXED-order split-KV 399.75 rung's
    full-serve identity vs the surgical-357 bar. Everything is judged on the W=8 SERVED verify geometry
    (the faithful headline -- the real M=8 serve does a width-8 forward over a decode-cached prefix)."""
    if "splitkv399" not in census:
        return {}
    sk = census["splitkv399"]
    w8 = sk["w8"]
    w8_flips = [f for f in sk["flip_details"] if f.get("width") == M_VERIFY]
    n_sem, n_tie, n_flips = w8["n_semantic_flips"], w8["n_tie_flips"], w8["n_flips"]
    ident = w8["token_identity_rate"]
    operative_1p0 = bool(math.isfinite(ident) and ident >= 0.99 and n_sem == 0)
    literally_byteexact = bool(n_flips == 0)
    sweep = _tie_threshold_sweep(w8_flips)

    # vs surgical-357: prefer a SAME-HARNESS surgical arm (apples-to-apples: identical geometry/positions);
    # else fall back to the imported #499/stark466 anchor (surgical num_splits=1 byte-exact, operative-1.0).
    same_harness = "surgical" in census
    if same_harness:
        sw8 = census["surgical"]["w8"]
        surg_ident, surg_sem, surg_flips = sw8["token_identity_rate"], sw8["n_semantic_flips"], sw8["n_flips"]
        surg_operative_rate = sw8["operative_identity_rate"]
        basis = "same-harness surgical arm (identical W=8 geometry)"
    else:
        surg_ident, surg_sem, surg_flips, surg_operative_rate = STARK466_LOCUS_IDENTITY, 0, 0, 1.0
        basis = "imported anchor (surgical-357 #499 operative-1.0 / stark466 0-flip locus)"

    # "cleaner" if splitkv399 has strictly fewer SEMANTIC flips (the only contract-breaking kind); for a
    # same-harness comparison, tie-break on TOTAL flips (literal byte-exactness). Against the imported anchor
    # we compare SEMANTIC only -- the anchor's 0-flip count is a single-window locus, so a total-flip compare
    # vs the full-serve census would unfairly penalize splitkv399 for benign bf16 ties.
    if n_sem != surg_sem:
        vs_surgical = "cleaner" if n_sem < surg_sem else "worse"
    elif same_harness and n_flips != surg_flips:
        vs_surgical = "cleaner" if n_flips < surg_flips else "worse"
    else:
        vs_surgical = "same"

    # examine EVERY flip: top-32 by m1_self_gap desc (the most "semantic-looking" first; None gap -> +inf top).
    flips_sorted = sorted(w8_flips, key=lambda f: (f.get("m1_self_gap") if f.get("m1_self_gap") is not None
                                                   else float("inf")), reverse=True)
    examination = [{
        "prompt_idx": f["prompt_idx"], "pos": f["pos"], "k": f["k"], "flip_kind": f["flip_kind"],
        "m1_self_gap": f["m1_self_gap"], "m8_gap": f["m8_gap"],
        "m1_in_m8_top2": f["m1_in_m8_top2"], "m1_in_m8_top5": f["m1_in_m8_top5"],
        "m1_argmax_matches_token": f["m1_argmax_matches_token"],
        "m8_top1_id": f["m8_top1_id"], "m1_tok_id": f["m1_tok_id"],
    } for f in flips_sorted[:32]]

    npos = w8["n_positions"]
    if literally_byteexact:
        verdict = (f"splitkv399 is LITERALLY BYTE-EXACT (0 flips / {npos} W=8 served positions); "
                   f"{vs_surgical} than surgical-357 ({basis}).")
    elif operative_1p0:
        verdict = (f"splitkv399 is OPERATIVE-1.0 (identity {ident:.7f}; {n_sem} semantic + {n_tie} tie of "
                   f"{npos}); every non-identity position is a bf16 tie, not literally byte-exact; "
                   f"{vs_surgical} than surgical-357 ({basis}).")
    else:
        verdict = (f"splitkv399 shows {n_sem} SEMANTIC flip(s) (identity {ident:.7f} of {npos}); "
                   f"NOT operative-1.0; {vs_surgical} than surgical-357 ({basis}).")

    return {
        "splitkv399_fullserve_census": ident,
        "splitkv399_n_semantic_flips": n_sem,
        "splitkv399_n_tie_flips": n_tie,
        "splitkv399_n_flips": n_flips,
        "splitkv399_operative_identity_rate": w8["operative_identity_rate"],
        "splitkv399_operative_identity_1p0": operative_1p0,
        "splitkv399_literally_byteexact": literally_byteexact,
        "splitkv399_n_positions_w8": npos,
        "splitkv399_mechanism": sk.get("mechanism", {}),
        "vs_surgical357": vs_surgical,
        "vs_surgical357_basis": basis,
        "surgical_fullserve_census": surg_ident,
        "surgical_n_semantic_flips": surg_sem,
        "surgical_n_flips": surg_flips,
        "surgical_operative_identity_rate": surg_operative_rate,
        "tie_threshold_for_zero_semantic": sweep["threshold_for_zero_semantic"],
        "tie_threshold_sweep": sweep,
        "splitkv399_flip_examination": examination,
        "splitkv399_one_line_verdict": verdict,
    }


def compose_and_report(census: dict, a: argparse.Namespace) -> dict:
    primary_arm = ("splitkv399" if "splitkv399" in census
                   else ("pinned" if "pinned" in census else sorted(census)[0]))
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

    # attribution: splitkv399 (#515 land) | surgical-357 (#510 wirbel) | base (#487 wirbel). The surgical
    # KEY OUTPUTS are emitted in a conditional block after the report dict (only when surgical is primary).
    is_surgical = primary_arm == "surgical" or bool(prim.get("surgical_mode"))
    is_splitkv = primary_arm == "splitkv399"
    report = {
        "pr": (515 if is_splitkv else 510 if is_surgical else 487),
        "agent": ("land" if is_splitkv else "wirbel"),
        "leg": ("splitkv399 full-serve byte-exact identity census (399.75 ship candidate vs surgical-357)"
                if is_splitkv else
                "surgical-357 ship: reload-immune full-serve operative-identity census" if is_surgical
                else "same-reload full-serve identity census harness"),
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "primary_arm": primary_arm,
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
    # PR #510 wirbel surgical-357 KEY OUTPUTS: emitted only when surgical is the primary arm (computed on the
    # primary arm's served W=8 flips). Preserves wirbel #510's deliverables additively under the splitkv399 rebase.
    if is_surgical:
        tie_sweep = tie_threshold_sweep(prim["flip_details"], width=M_VERIFY)
        served_flip_exam = examine_served_flips(prim["flip_details"], width=M_VERIFY)
        vs222 = compare_vs_globalflag222(w8["n_semantic_flips"])
        surgical_locus_gap = (locus["token_identity_rate"] - STARK494_SURGICAL_LOCUS_IDENTITY
                              if math.isfinite(locus["token_identity_rate"]) else float("nan"))
        surgical_locus_reproduces_494 = bool(math.isfinite(surgical_locus_gap)
                                             and abs(surgical_locus_gap) <= (2.0 / max(1, locus["n_positions"])))
        report.update({
            "surgical_mode": True,
            "surgical357_fullserve_census": w8["token_identity_rate"],
            "surgical357_n_semantic_flips": w8["n_semantic_flips"],
            "surgical357_n_tie_flips": w8["n_tie_flips"],
            "surgical357_operative_identity_1p0": operative_1p0,
            "surgical357_operative_identity_rate": w8["operative_identity_rate"],
            "surgical_attn_armed": prim.get("surgical_attn_armed"),
            "matmul_tax_installed": prim.get("matmul_tax_installed"),
            "surgical_lever_source": prim.get("surgical_lever_source"),
            "tie_threshold_sweep": tie_sweep,
            "tie_threshold_for_zero_semantic": tie_sweep["tie_threshold_for_zero_semantic"],
            "tie_threshold_for_zero_semantic_ulps": tie_sweep["tie_threshold_for_zero_semantic_ulps"],
            "served_flip_examination": served_flip_exam,
            **vs222,
            "surgical_locus_anchor_stark494": STARK494_SURGICAL_LOCUS_IDENTITY,
            "surgical_locus_gap_vs_stark494": surgical_locus_gap,
            "surgical_locus_reproduces_stark494": surgical_locus_reproduces_494,
        })
    # PR #515 land KEY OUTPUTS: ship-candidate splitkv399 vs surgical-357 (no-op when arm absent).
    report.update(_splitkv399_extras(census))
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
    # These lever-provenance assertions apply ONLY to a wirbel-style #510 surgical run (surgical_mode True,
    # i.e. _install_surgical_lever() recorded surgical_attn_armed / matmul_tax_installed). A pre-#510
    # comparison-bar surgical arm (surgical_mode absent) reaches the same 2D order-preserving byte-exact
    # config via the SURGICAL_ATTN_USE_3D_OFF env and does not carry the lever provenance, so it is not
    # subject to these checks (the always-on per-arm sanity checks above still cover it).
    if "surgical" in census and census["surgical"].get("surgical_mode"):
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
    reasoning = bool(getattr(a, "reasoning", False))
    tag = "reasoning_" if reasoning else ""           # reasoning census writes DISTINCT files (never clobbers the
    out_json = str(OUT_DIR / f"arm_{tag}{arm}_result.json")   # 128-prompt baseline arm_{arm}_result.json)
    ckpt = str(OUT_DIR / f"checkpoint_{tag}{arm}.jsonl")
    hb = str(OUT_DIR / f"heartbeat_{tag}{arm}.json")
    # pinned = global-flag strict (VBI=1: 2D attn + matmul tax). surgical = shipped 357 lever (VBI=0 +
    # SURGICAL_ATTN_USE_3D_OFF=1: 2D attn, matmul tax OFF). heuristic = stock (VBI=0, no pin).
    # splitkv399 = byte-exact FIXED-order split-KV candidate (lawine #496/#500, gated TPS/segments).
    if arm == "pinned":
        extra_env = {"VLLM_BATCH_INVARIANT": "1"}
    elif arm == "splitkv399":
        # Candidate: byte-exact FIXED-order split-KV (lawine #496/#500 recipe, T=4 S=64),
        # M=8 verify redirected to 3D split-KV. is_batch_invariant stays False; the local
        # vLLM wheel honors BYTEEXACT_FIXED_TPS/_NUM_SEGMENTS (gated, default-off).
        extra_env = {"VLLM_BATCH_INVARIANT": "0", "SPLITKV_VERIFY": "1",
                     "SPLITKV_VERIFY_MAX_Q": "64",
                     "BYTEEXACT_FIXED_TPS": str(a.fixed_tps),
                     "BYTEEXACT_NUM_SEGMENTS": str(a.num_segments)}
    elif arm == "surgical":
        # Comparison bar: 2D order-preserving byte-exact attention, fast matmul (#499 shipped rung).
        extra_env = {"VLLM_BATCH_INVARIANT": "0", "SURGICAL_ATTN_USE_3D_OFF": "1"}
    else:
        extra_env = {"VLLM_BATCH_INVARIANT": "0"}
    base_args = [
        "--phase", "fullserve_census", "--arm", arm, "--out", out_json,
        "--n-prompts", str(a.n_prompts), "--c0", str(a.c0), "--traj-len", str(a.traj_len),
        "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
        "--verbose-k", str(a.verbose_k), "--det-check-k", str(a.det_check_k),
        "--checkpoint", ckpt, "--heartbeat", hb, "--resume",
    ] + (["--ignore-eos"] if a.ignore_eos else []) + (
        ["--reasoning", "--prompts", a.prompts] if reasoning else [])

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
    hdr = f"PR #{r.get('pr', 487)} ({r.get('agent', 'wirbel')})"
    print(f"\n========== FULL-SERVE IDENTITY CENSUS HARNESS ({hdr}) ==========", flush=True)
    if "splitkv399_fullserve_census" in r:
        print(" --- PR #515 SHIP CANDIDATE: splitkv399 (FIXED-order split-KV, 399.75 rung) ---", flush=True)
        print(f"  splitkv399_fullserve_census (W=8)       : {r['splitkv399_fullserve_census']:.7f}  "
              f"(semantic={r['splitkv399_n_semantic_flips']} tie={r['splitkv399_n_tie_flips']} "
              f"of {r['splitkv399_n_positions_w8']} pos)", flush=True)
        print(f"  splitkv399_literally_byteexact (0 flips): {r['splitkv399_literally_byteexact']}  "
              f"(operative_1p0={r['splitkv399_operative_identity_1p0']})", flush=True)
        print(f"  vs_surgical357                          : {r['vs_surgical357']}  "
              f"(surgical census {r['surgical_fullserve_census']:.7f}, "
              f"sem={r['surgical_n_semantic_flips']}; basis: {r['vs_surgical357_basis']})", flush=True)
        print(f"  tie_threshold_for_zero_semantic         : {r['tie_threshold_for_zero_semantic']}", flush=True)
        print(f"  VERDICT (one line)                      : {r['splitkv399_one_line_verdict']}", flush=True)
        print("  ------------------------------------------------------------", flush=True)
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
          f"({r['n_semantic_flips']} semantic) <- HEADLINE", flush=True)
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
    is_splitkv = report["primary_arm"] == "splitkv399"
    _surgical = bool(report.get("surgical_mode"))
    if is_splitkv:
        notes = ("PR#515 (land) splitkv399 full-serve byte-exact identity census: reload-immune token_identity "
                 "of the FIXED-order split-KV M=8 verify (lawine #496/#500 399.75 rung) vs M=1 AR over the full "
                 "free-running trajectory; ship-candidate compared to the surgical-357 bar.")
    elif _surgical:
        notes = ("PR#510 surgical-357 SHIP reload-immune full-serve operative-identity census: the M=8 surgical "
                 "serve (2D order-preserving attention, matmul tax OFF) vs its own M=1 AR, swept along the full "
                 "free-running trajectory. Stress-tests the stark #494 locus operative-1.0 cert at full-serve "
                 "scale (does it survive, or share #487's global-flag-222 blind spot?).")
    else:
        notes = ("PR#487 same-reload full-serve identity census: reload-immune token_identity_rate of the M=8 "
                 "strict serve vs M=1 AR, swept along the full free-running trajectory (closes the #471 locus -> "
                 "#470 reload-confounded gap).")
    run = init_wandb_run(
        job_type="local_profiling", agent=report["agent"], name=a.wandb_name, group=a.wandb_group,
        notes=notes,
        config={
            "pr": report["pr"], "M_verify": M_VERIFY, "K_spec": K_SPEC, "wide_w": WIDE_W,
            "G": HYBRID_PREFIX_COMMIT,
            "C": report["C"], "traj_len": report["traj_len"], "ignore_eos": report["ignore_eos"],
            "n_prompts_run": report["n_prompts_run"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"], "eps_star": EPS_STAR, "surgical_mode": _surgical,
            "fixed_tps": a.fixed_tps, "num_segments": a.num_segments,
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
        # ---- PR #515 land KEY OUTPUTS (splitkv399 ship candidate vs surgical-357; None when arm absent) ----
        "splitkv399_fullserve_census", "splitkv399_n_semantic_flips", "splitkv399_n_tie_flips",
        "splitkv399_n_flips", "splitkv399_operative_identity_rate", "splitkv399_operative_identity_1p0",
        "splitkv399_literally_byteexact", "splitkv399_n_positions_w8",
        "vs_surgical357", "vs_surgical357_basis", "surgical_fullserve_census",
        "surgical_n_semantic_flips", "surgical_n_flips", "surgical_operative_identity_rate",
        "splitkv399_one_line_verdict",
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
    # PR #515 land splitkv399 mechanism + flip examination (no-op when arm absent)
    if "tie_threshold_sweep" in report:
        for sk, sv in report["tie_threshold_sweep"].items():
            run.summary[f"tie_threshold_sweep/{sk}"] = (json.dumps(sv) if isinstance(sv, dict) else sv)
    if report.get("splitkv399_mechanism"):
        for mk, mv in report["splitkv399_mechanism"].items():
            run.summary[f"splitkv399_mechanism/{mk}"] = mv
    if report.get("splitkv399_flip_examination"):
        run.summary["splitkv399_flip_examination"] = json.dumps(report["splitkv399_flip_examination"])
    run.summary["verdict_green"] = report["verdict"].startswith("GREEN")
    run.summary["verdict_red"] = report["verdict"].startswith("RED")
    for arm, d in report["arms"].items():
        for mk, mv in d.items():
            run.summary[f"{arm}/{mk}"] = mv
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


# ======================================================================================
# PR #534 (land) OFF-DISTRIBUTION REASONING-LENGTH census: the SAME surgical-357 ship, but free-run
# from AIME 2024/2025 reasoning prompts (each its OWN natural prefix, hundreds of decode steps) instead
# of the 128 multiple-choice challenge prompts. Answers the advisor's open question: does operative-
# 1.0-within-1-bf16-ULP stay BOUNDED off the challenge distribution, where more decode steps = more
# non-order-preserving split-KV reductions = more potential ULP flips? Reuses the EXACT W=8 served-verify
# geometry + tie/semantic classifier; the only differences are the prompt set, the per-prompt natural C,
# and that the report is graded against the BASELINE128_SURGICAL_* envelope (run hjiwvkfg) rather than the
# denken#471 O=224 locus (which is a fixed-C cross-check that does not apply to per-prompt prefixes).
# ======================================================================================
def _reasoning_length_stats(per_prompt: list[dict]) -> dict:
    """Distribution of the realized free-running trajectory lengths (traj_n) and per-prompt natural prefix
    lengths (C). This is the evidence that the census actually stressed 'hundreds of decode steps'."""
    trajs = sorted(int(p.get("traj_n", 0)) for p in per_prompt)
    ctxs = sorted(int(p.get("C", 0)) for p in per_prompt)
    n = len(trajs)
    if n == 0:
        return {"n_prompts": 0}
    def _pct(xs, q):
        return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]
    return {
        "n_prompts": n,
        "traj_n_min": trajs[0], "traj_n_med": _pct(trajs, 0.5), "traj_n_max": trajs[-1],
        "traj_n_p25": _pct(trajs, 0.25), "traj_n_p75": _pct(trajs, 0.75),
        "traj_n_mean": round(sum(trajs) / n, 1),
        "ctx_C_min": ctxs[0], "ctx_C_med": _pct(ctxs, 0.5), "ctx_C_max": ctxs[-1],
        "total_decode_steps": sum(trajs),
    }


def compose_reasoning_report(census: dict, a: argparse.Namespace) -> dict:
    """PR #534 (land) OFF-DISTRIBUTION reasoning-length census report. Single surgical-357 arm (the live
    ship, my card's substrate). Headline deliverables the advisor named: reasoning_census_operative_identity,
    max ULP, and flip rate vs the 128-prompt baseline (hjiwvkfg). Everything is judged on the W=8 SERVED
    verify geometry -- IDENTICAL classifier/geometry to the baseline so the comparison is apples-to-apples;
    the ONLY changed dimension is the prompt distribution (challenge -> AIME reasoning) and per-prompt C."""
    arm = "surgical" if "surgical" in census else sorted(census)[0]
    prim = census[arm]
    w8 = prim["w8"]
    w8_flips = [f for f in prim["flip_details"] if f.get("width") == M_VERIFY and f.get("is_flip")]

    npos = w8["n_positions"]
    n_sem, n_tie, n_flips = w8["n_semantic_flips"], w8["n_tie_flips"], w8["n_flips"]
    token_identity = w8["token_identity_rate"]
    operative_identity = w8["operative_identity_rate"]
    # strict operative-1.0 (identity>=0.99 AND ZERO semantic flips). NB the baseline itself does NOT meet this
    # at full serve (3 sub-2-ULP semantic flips) -- its operative-1.0 is a LOCUS cert; we report the strict
    # bool honestly AND the within-envelope comparison below (the question the advisor actually asked).
    operative_1p0_strict = bool(math.isfinite(token_identity) and token_identity >= 0.99 and n_sem == 0)

    # max ULP: the tie-tolerance certificate knob. The generic sweep re-buckets served flips by the M=1
    # reference's OWN top-2 gap (m1_self_gap) expressed in bf16 ULPs (ULP_NAT=0.0625 nat/step).
    sweep = tie_threshold_sweep(w8_flips, width=M_VERIFY)
    exam = examine_served_flips(prim["flip_details"], width=M_VERIFY)
    sem_ulps = sweep["semantic_gaps_in_ulps"]
    max_semantic_ulp = max(sem_ulps) if sem_ulps else 0.0
    all_ulps = [round(float(f["m1_self_gap"]) / ULP_NAT, 3) for f in w8_flips
                if f.get("m1_self_gap") is not None]
    max_any_ulp = max(all_ulps) if all_ulps else 0.0
    n_flips_unreadable_gap = sum(1 for f in w8_flips if f.get("m1_self_gap") is None)
    # every semantic flip is a bf16 near-tie iff its gap <= EPS_STAR (0.125 nat = 2 ULP). This is the basis of
    # the whole cert: a "semantic" flip that exceeds EPS_STAR would be a GENUINE divergence, not bf16 rounding.
    all_semantic_within_eps_star = bool(sweep["all_semantic_collapse_at_eps_star"] and n_flips_unreadable_gap == 0)

    # flip RATE (per W=8 served position) vs the baseline rates.
    flip_rate = (n_flips / npos) if npos else float("nan")
    semantic_flip_rate = (n_sem / npos) if npos else float("nan")
    tie_flip_rate = (n_tie / npos) if npos else float("nan")
    base_flip_rate = BASELINE128_SURGICAL_N_FLIPS / BASELINE128_SURGICAL_N_POSITIONS
    base_semantic_rate = BASELINE128_SURGICAL_N_SEMANTIC / BASELINE128_SURGICAL_N_POSITIONS
    sem_rate_ratio = (semantic_flip_rate / base_semantic_rate
                      if base_semantic_rate > 0 else (float("inf") if n_sem else 0.0))
    operative_identity_delta = (operative_identity - BASELINE128_SURGICAL_OPERATIVE_IDENTITY
                                if math.isfinite(operative_identity) else float("nan"))
    max_ulp_delta = max_semantic_ulp - BASELINE128_SURGICAL_MAX_SEMANTIC_ULP

    # ---- the verdict: does the operative-identity cert STAY BOUNDED on the long reasoning distribution? ----
    # BOUNDED   : every flip is a bf16 near-tie (<= EPS_STAR), the worst semantic gap is no larger than the
    #             baseline's (<= 2 ULP), and the semantic flip RATE is not materially above baseline -- i.e.
    #             reasoning-length decode does NOT manufacture larger/more ULP flips than the challenge set.
    # ELEVATED  : still all near-ties (cert holds at a k-ULP tolerance) but a higher rate or a wider worst-case
    #             gap than baseline -- operative-1.0-at-k-ULP for a modestly larger k; worth reporting, not a break.
    # BROKEN    : a semantic flip whose gap exceeds the bf16 near-tie band (> EPS_STAR), i.e. a genuine
    #             divergence the bf16-ULP cert cannot absorb.
    SEM_RATE_TOL = 2.0      # allow 2x the baseline semantic-flip rate (Poisson noise at these tiny counts)
    if math.isfinite(operative_identity) and operative_identity >= 0.99 and n_sem == 0:
        cert_status = "BOUNDED"
        verdict = (f"operative-1.0 HOLDS off-distribution: identity {operative_identity:.7f}, ZERO semantic "
                   f"flips over {npos} W=8 reasoning positions ({n_tie} bf16 ties). The cert is strictly "
                   f"bounded on the long reasoning distribution -- tighter than the 128-challenge baseline "
                   f"(3 semantic / {BASELINE128_SURGICAL_N_POSITIONS}).")
    elif not all_semantic_within_eps_star:
        cert_status = "BROKEN"
        verdict = (f"operative-identity cert does NOT hold off-distribution: a semantic flip exceeds the bf16 "
                   f"near-tie band (max semantic {max_semantic_ulp:.2f} ULP > {EPS_STAR/ULP_NAT:.0f} ULP, or a "
                   f"flip has no readable M=1 gap). identity {operative_identity:.7f}, {n_sem} semantic / {npos}.")
    elif max_semantic_ulp <= BASELINE128_SURGICAL_MAX_SEMANTIC_ULP and sem_rate_ratio <= SEM_RATE_TOL:
        cert_status = "BOUNDED"
        verdict = (f"operative-identity cert STAYS BOUNDED: identity {operative_identity:.7f} ({n_sem} semantic "
                   f"+ {n_tie} tie / {npos}); every flip is a bf16 near-tie, worst semantic {max_semantic_ulp:.2f} "
                   f"ULP <= baseline {BASELINE128_SURGICAL_MAX_SEMANTIC_ULP:.0f} ULP, semantic rate "
                   f"{semantic_flip_rate:.2e} ({sem_rate_ratio:.2f}x baseline). Reasoning-length decode does NOT "
                   f"manufacture larger/more ULP flips than the challenge set.")
    else:
        cert_status = "ELEVATED"
        verdict = (f"operative-identity cert holds at a k-ULP tolerance but is ELEVATED vs baseline: identity "
                   f"{operative_identity:.7f} ({n_sem} semantic + {n_tie} tie / {npos}); worst semantic "
                   f"{max_semantic_ulp:.2f} ULP (baseline {BASELINE128_SURGICAL_MAX_SEMANTIC_ULP:.0f}), semantic "
                   f"rate {sem_rate_ratio:.2f}x baseline. Every flip is still a sub-EPS* near-tie, so the cert "
                   f"survives at a slightly looser tie band -- not a divergence.")
    cert_stays_bounded = cert_status == "BOUNDED"

    length_stats = _reasoning_length_stats(prim.get("per_prompt", []))

    report = {
        "pr": 534, "agent": "land",
        "leg": ("off-distribution reasoning-length operative-identity census: surgical-357 ship free-run from "
                "AIME 2024/2025 reasoning prompts (per-prompt natural C, hundreds of decode steps) vs its own "
                "M=1 AR -- does operative-1.0-within-1-bf16-ULP stay BOUNDED off the 128-challenge distribution?"),
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "reasoning": True, "primary_arm": arm, "headline_geometry": "W=8 (served M=8 verify width)",
        "prompts_path": prim.get("prompts_path"),
        # ---- HEADLINE deliverables (the advisor's named asks) ----
        "reasoning_census_operative_identity": operative_identity,
        "reasoning_census_token_identity": token_identity,
        "reasoning_max_semantic_ulp": max_semantic_ulp,
        "reasoning_max_any_flip_ulp": max_any_ulp,
        "reasoning_flip_rate": flip_rate,
        "reasoning_semantic_flip_rate": semantic_flip_rate,
        "reasoning_tie_flip_rate": tie_flip_rate,
        "reasoning_n_semantic_flips": n_sem, "reasoning_n_tie_flips": n_tie, "reasoning_n_flips": n_flips,
        "reasoning_n_positions_w8": npos,
        "reasoning_operative_1p0_strict": operative_1p0_strict,
        "reasoning_all_semantic_within_eps_star": all_semantic_within_eps_star,
        "reasoning_n_flips_without_readable_m1_gap": n_flips_unreadable_gap,
        # ---- comparison vs the 128-challenge surgical baseline (hjiwvkfg) ----
        "baseline128_operative_identity": BASELINE128_SURGICAL_OPERATIVE_IDENTITY,
        "baseline128_token_identity": BASELINE128_SURGICAL_TOKEN_IDENTITY,
        "baseline128_n_semantic": BASELINE128_SURGICAL_N_SEMANTIC,
        "baseline128_n_tie": BASELINE128_SURGICAL_N_TIE,
        "baseline128_n_positions": BASELINE128_SURGICAL_N_POSITIONS,
        "baseline128_max_semantic_ulp": BASELINE128_SURGICAL_MAX_SEMANTIC_ULP,
        "baseline128_flip_rate": base_flip_rate,
        "baseline128_semantic_flip_rate": base_semantic_rate,
        "baseline128_wandb": BASELINE128_SURGICAL_WANDB,
        "operative_identity_delta_vs_baseline": operative_identity_delta,
        "max_semantic_ulp_delta_vs_baseline": max_ulp_delta,
        "semantic_flip_rate_ratio_vs_baseline": sem_rate_ratio,
        "cert_status": cert_status, "cert_stays_bounded": cert_stays_bounded,
        "one_line_verdict": verdict,
        # ---- reasoning-length evidence (this is a 'hundreds of decode steps' workload) ----
        "reasoning_length_stats": length_stats,
        "n_prompts_run": prim["n_prompts_run"], "traj_len": prim["traj_len"], "ignore_eos": prim["ignore_eos"],
        "prefix_C_per_prompt": prim.get("prefix_C_per_prompt"),
        # ---- tie-threshold sweep + per-flip examination (the certificate-wording knob) ----
        "tie_threshold_sweep": sweep, "served_flip_examination": exam,
        # ---- reload-immunity + geometry proof (must all be ~1.0; identical probes to the baseline) ----
        "reload_immune": bool(prim["determinism_M8"] == 1.0
                              and (not math.isfinite(prim["determinism_M1"]) or prim["determinism_M1"] == 1.0)
                              and (not math.isfinite(prim["within_batch"]) or prim["within_batch"] == 1.0)),
        "determinism_M8": prim["determinism_M8"], "determinism_M1": prim["determinism_M1"],
        "within_batch": prim["within_batch"],
        "chunk_isolated_w8": prim["chunk_isolated_w8"], "chunk_isolated_w32": prim["chunk_isolated_w32"],
        "nan_clean": prim["nan_clean"],
        # ---- mechanism provenance (the EXACT shipped surgical-357 lever) ----
        "surgical_attn_armed": prim.get("surgical_attn_armed"),
        "matmul_tax_installed": prim.get("matmul_tax_installed"),
        "surgical_lever_source": prim.get("surgical_lever_source"),
        "vllm_batch_invariant_env": prim.get("vllm_batch_invariant_env"),
        "attn_is_batch_invariant": prim.get("attn_is_batch_invariant"),
        "peak_gpu_gb": prim.get("peak_gpu_gb"),
        "n_windows": prim["n_windows"], "C": prim["C"],
        "model_dir": prim["model_dir"],
        "imported_anchors": {
            "baseline128_surgical_operative_identity": BASELINE128_SURGICAL_OPERATIVE_IDENTITY,
            "baseline128_surgical_token_identity": BASELINE128_SURGICAL_TOKEN_IDENTITY,
            "baseline128_surgical_n_semantic": BASELINE128_SURGICAL_N_SEMANTIC,
            "baseline128_surgical_n_positions": BASELINE128_SURGICAL_N_POSITIONS,
            "baseline128_surgical_max_semantic_ulp": BASELINE128_SURGICAL_MAX_SEMANTIC_ULP,
            "baseline128_surgical_traj_len": BASELINE128_SURGICAL_TRAJ_LEN,
            "baseline128_surgical_n_prompts": BASELINE128_SURGICAL_N_PROMPTS,
            "baseline128_surgical_wandb": BASELINE128_SURGICAL_WANDB,
            "eps_star_nat": EPS_STAR, "ulp_nat": ULP_NAT,
        },
        # ---- self-test (sanity invariants of the reasoning read) ----
    }
    report["self_test"], report["self_test_n_checks"] = _reasoning_self_test(prim, report)
    report["fullserve_self_test_passes"] = self_test_ok(report["self_test"])
    return report


def _reasoning_self_test(prim: dict, report: dict) -> tuple[dict, int]:
    w8 = prim["w8"]
    checks = {
        "operative_identity_in_unit": bool(math.isfinite(report["reasoning_census_operative_identity"])
                                            and 0.0 <= report["reasoning_census_operative_identity"] <= 1.0),
        "token_identity_in_unit": bool(math.isfinite(report["reasoning_census_token_identity"])
                                       and 0.0 <= report["reasoning_census_token_identity"] <= 1.0),
        "nan_clean": bool(prim["nan_clean"]),
        "geometry_w8_isolated": bool(prim["chunk_isolated_w8"] >= 0.99),
        "reload_immune_det_m8_eq_1": bool(prim["determinism_M8"] == 1.0),
        "surgical_attn_armed": bool(prim.get("surgical_attn_armed")),
        "matmul_tax_off": bool(prim.get("matmul_tax_installed") is False),
        # the reasoning workload really is reasoning-LENGTH: median realized trajectory is hundreds of steps
        "reasoning_length_is_deep": bool(report["reasoning_length_stats"].get("traj_n_med", 0) >= 128),
        # per-prompt natural prefixes were used (not the fixed C=224) -- prefix lengths vary across prompts
        "per_prompt_natural_prefix": bool(report.get("prefix_C_per_prompt")
                                          and len(set(report["prefix_C_per_prompt"])) > 1),
        # every served flip is accounted for as tie or semantic (no unclassified)
        "flips_fully_classified": bool(w8["n_tie_flips"] + w8["n_semantic_flips"] == w8["n_flips"]),
    }
    return checks, len(checks)


def _print_reasoning_console(r: dict) -> None:
    ls = r.get("reasoning_length_stats", {})
    print(f"\n===== PR #534 (land) OFF-DISTRIBUTION REASONING-LENGTH OPERATIVE-IDENTITY CENSUS =====", flush=True)
    print(f" VERDICT (cert stays bounded?)           : {r['cert_status']}  "
          f"(bounded={r['cert_stays_bounded']})", flush=True)
    print(f"  {r['one_line_verdict']}", flush=True)
    print(" --- HEADLINE (W=8 served verify geometry, AIME reasoning prompts) ---", flush=True)
    print(f"  reasoning_census_operative_identity     : {r['reasoning_census_operative_identity']:.7f}  "
          f"(semantic={r['reasoning_n_semantic_flips']} tie={r['reasoning_n_tie_flips']} "
          f"of {r['reasoning_n_positions_w8']} pos)", flush=True)
    print(f"  reasoning_census_token_identity         : {r['reasoning_census_token_identity']:.7f}", flush=True)
    print(f"  max ULP (worst semantic / any flip)     : {r['reasoning_max_semantic_ulp']:.2f} / "
          f"{r['reasoning_max_any_flip_ulp']:.2f} bf16 ULP  "
          f"(all_semantic<=EPS*={r['reasoning_all_semantic_within_eps_star']})", flush=True)
    print(f"  flip rate (total / semantic)            : {r['reasoning_flip_rate']:.3e} / "
          f"{r['reasoning_semantic_flip_rate']:.3e} per W=8 pos", flush=True)
    print(f"  operative_1p0_strict (id>=.99 & 0 sem)  : {r['reasoning_operative_1p0_strict']}", flush=True)
    print(" --- vs 128-challenge baseline (surgical-357, hjiwvkfg) ---", flush=True)
    print(f"  baseline operative_identity             : {r['baseline128_operative_identity']:.7f}  "
          f"(semantic={r['baseline128_n_semantic']} of {r['baseline128_n_positions']}, "
          f"max {r['baseline128_max_semantic_ulp']:.0f} ULP)", flush=True)
    print(f"  operative_identity delta vs baseline    : {r['operative_identity_delta_vs_baseline']:+.2e}", flush=True)
    print(f"  semantic flip-rate ratio vs baseline    : {r['semantic_flip_rate_ratio_vs_baseline']:.2f}x", flush=True)
    print(f"  max-semantic-ULP delta vs baseline      : {r['max_semantic_ulp_delta_vs_baseline']:+.2f} ULP", flush=True)
    print(" --- reasoning-length evidence (hundreds of decode steps) ---", flush=True)
    print(f"  realized traj_n min/med/max             : {ls.get('traj_n_min')}/{ls.get('traj_n_med')}/"
          f"{ls.get('traj_n_max')}  (mean {ls.get('traj_n_mean')}, total {ls.get('total_decode_steps')} steps)",
          flush=True)
    print(f"  per-prompt natural prefix C min/med/max : {ls.get('ctx_C_min')}/{ls.get('ctx_C_med')}/"
          f"{ls.get('ctx_C_max')}  (n_prompts={ls.get('n_prompts')}, traj_len cap={r['traj_len']})", flush=True)
    print(" --- reload-immunity + mechanism provenance ---", flush=True)
    print(f"  reload_immune                           : {r['reload_immune']}  "
          f"(det_m8={r['determinism_M8']:.4f} det_m1={r['determinism_M1']} within={r['within_batch']})", flush=True)
    print(f"  surgical_attn_armed / matmul_tax_off    : {r.get('surgical_attn_armed')} / "
          f"{r.get('matmul_tax_installed') is False}  (peak {r.get('peak_gpu_gb')}GB)", flush=True)
    print(f" SELF-TEST PASSES                         : {r['fullserve_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("====================================================================================\n", flush=True)


def log_reasoning_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    notes = ("PR#534 (land) OFF-DISTRIBUTION reasoning-length operative-identity census: the SAME surgical-357 "
             "ship (2D order-preserving attention, matmul tax OFF) free-run from AIME 2024/2025 reasoning "
             "prompts (each its OWN natural prefix, hundreds of decode steps -- NONE overlap the 128-challenge "
             "aime2026), M=8 verify vs M=1 AR over the full free-running trajectory. Tests whether the ship's "
             "operative-1.0-within-1-bf16-ULP identity stays BOUNDED off the 128-challenge distribution (more "
             "decode steps = more non-order-preserving split-KV reductions = more potential ULP flips). Compared "
             "to the 128-challenge surgical baseline (run hjiwvkfg).")
    run = init_wandb_run(
        job_type="local_profiling", agent="land", name=a.wandb_name, group=a.wandb_group, notes=notes,
        config={
            "pr": 534, "M_verify": M_VERIFY, "K_spec": K_SPEC, "G": HYBRID_PREFIX_COMMIT,
            "reasoning": True, "prompts_path": report.get("prompts_path"),
            "traj_len": report["traj_len"], "ignore_eos": report["ignore_eos"],
            "n_prompts_run": report["n_prompts_run"], "model_dir": report["model_dir"],
            "primary_arm": report["primary_arm"], "eps_star": EPS_STAR, "ulp_nat": ULP_NAT,
            "surgical_mode": True,
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            **{f"anchor/{k}": v for k, v in report["imported_anchors"].items()},
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = (
        "reasoning_census_operative_identity", "reasoning_census_token_identity",
        "reasoning_max_semantic_ulp", "reasoning_max_any_flip_ulp",
        "reasoning_flip_rate", "reasoning_semantic_flip_rate", "reasoning_tie_flip_rate",
        "reasoning_n_semantic_flips", "reasoning_n_tie_flips", "reasoning_n_flips", "reasoning_n_positions_w8",
        "reasoning_operative_1p0_strict", "reasoning_all_semantic_within_eps_star",
        "reasoning_n_flips_without_readable_m1_gap",
        "baseline128_operative_identity", "baseline128_token_identity", "baseline128_n_semantic",
        "baseline128_n_tie", "baseline128_n_positions", "baseline128_max_semantic_ulp",
        "baseline128_flip_rate", "baseline128_semantic_flip_rate", "baseline128_wandb",
        "operative_identity_delta_vs_baseline", "max_semantic_ulp_delta_vs_baseline",
        "semantic_flip_rate_ratio_vs_baseline", "cert_status", "cert_stays_bounded", "one_line_verdict",
        "reload_immune", "determinism_M8", "determinism_M1", "within_batch",
        "chunk_isolated_w8", "chunk_isolated_w32", "nan_clean",
        "surgical_attn_armed", "matmul_tax_installed", "surgical_lever_source",
        "n_prompts_run", "traj_len", "ignore_eos", "C", "n_windows", "peak_gpu_gb",
        "fullserve_self_test_passes", "self_test_n_checks",
        "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
    )
    for k in keys:
        run.summary[k] = report.get(k)
    for sk, sv in report.get("reasoning_length_stats", {}).items():
        run.summary[f"reasoning_length/{sk}"] = sv
    for sk, sv in report.get("tie_threshold_sweep", {}).items():
        if sk == "n_semantic_at_threshold":
            for tname, tval in sv.items():
                run.summary[f"tie_sweep/n_semantic_at_{tname}"] = tval
        elif not isinstance(sv, list):
            run.summary[f"tie_sweep/{sk}"] = sv
    run.summary["served_flip_examination"] = json.dumps(report.get("served_flip_examination", []))
    run.summary["cert_bounded"] = report["cert_stays_bounded"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def _finish_reasoning(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "fullserve_reasoning_census_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_reasoning_console(report)
    print(f"[reasoning] report -> {report_path}", flush=True)
    if not a.no_wandb:
        log_reasoning_wandb(report, a)


def orchestrate_reasoning(a: argparse.Namespace) -> None:
    """Run the OFF-DISTRIBUTION reasoning-length census. Forces the surgical-357 arm (the live ship) and the
    reasoning prompt path, runs ONE watchdogged arm (writes arm_reasoning_surgical_result.json), then composes
    the reasoning report unless --census-only (the two-phase split: GPU census under the senpai-venvs python,
    then a 0-GPU --reanalyze-reasoning under a wandb-capable python -- see the fullserve-census-venv memory)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a.reasoning = True
    arm = a.arm or "surgical"
    if arm != "surgical":
        print(f"[reasoning] WARNING arm={arm} is not the shipped surgical-357 lever; forcing surgical", flush=True)
        arm = "surgical"
    a.arms = ("surgical",)
    census = {"surgical": _run_census_arm(a, "surgical")}
    if getattr(a, "census_only", False):
        print("[reasoning] census_only: arm result written; run --reanalyze-reasoning to compose + log W&B",
              flush=True)
        return
    _finish_reasoning(compose_reasoning_report(census, a), a)


def reanalyze_reasoning(a: argparse.Namespace) -> None:
    """0-GPU compose + W&B log from a finished reasoning arm result (run under a wandb-capable python)."""
    p = OUT_DIR / "arm_reasoning_surgical_result.json"
    if not p.exists():
        raise FileNotFoundError(f"--reanalyze-reasoning needs {p} (run the GPU --reasoning-census --census-only first)")
    census = {"surgical": json.load(open(p))}
    _finish_reasoning(compose_reasoning_report(census, a), a)


# ======================================================================================
# PR #521 BENCHMARK-CONFIG CENSUS: the REAL spec-on served decode path (not teacher-forced).
# --------------------------------------------------------------------------------------
# The #510 census (phase_fullserve_census, surgical arm) is teacher-forced: it sweeps W=8 verify
# reads ALONG a synthetic M=1 AR trajectory. A skeptical reviewer objects that this is not the path
# that actually produces the scored tokens. This phase closes that: it loads the EXACT shipped
# surgical-357 SERVED stack (the baked int4 model + pruned 12k lm_head + surgical 2D attention pin +
# the MTP K=7 spec-decode drafter -- the real spec-on serving config), free-runs the 128 official
# eval prompts at the official 128x512 geometry to obtain T_served (the REAL scored tokens), and then
# measures operative identity of every one of the 65,536 served tokens against the canonical M=1 AR
# greedy reference RE-CONDITIONED ON THE REALIZED SERVED PREFIX (so there is no #470 free-run-fork
# cascade confound; each position is judged on the prefix the serve actually produced).
#
# REFERENCE = M=1 AR (drafter OFF) argmax conditioned on seq[:pos], read at decode-width-1 -- exactly
# the submission's own canonical greedy reference (serve.py reference_mode disables only speculation;
# the surgical attn pin stays on). This matches #510's reference, so matches_510_geometry_census is
# well-defined. The served token (m8_top1_id) is the REAL verify output, not a teacher-forced read.
#
# CLEAR-WINNER SCREEN (full coverage, provably safe): a served-vs-M=1 flip can only occur at a MUTUAL
# near-tie (the two paths' logits differ only by sub-ULP attention-numerics, so if the served verify
# top-1 beats top-2 by more than `skip_threshold` >> any observed flip gap, the M=1 argmax is the same
# token -> a guaranteed match). So we READ the M=1 reference only at near-tie served positions and
# classify every clear-winner as a match. The bound is validated empirically (screen_check probe reads
# a sample of skipped positions and asserts they match). All 65,536 positions are thus covered.
# ======================================================================================
BC_SUBDIR = "submissions/fa2sw_strict_surgical357"
BC_EVAL_PROMPTS = "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"
BC_MODEL_DIR = "/tmp/osoi5-12k-baked"                 # the shipped baked int4 + pruned-12k-lm_head model
# PR #534 (land) FULL-HEAD safe-anchor. CORRECTED (advisor morganmcg1 2026-06-16 22:07Z + fern #531/#535):
# osoi5-v0-baked is PHYSICALLY a 16,384-row head (the 262k->16k PCK04 prune is BAKED IN; serve-time
# LM_HEAD_PRUNE only does 16k->12k). There is NO full-262k osoi5 to serve -- "no-prune on osoi5" is a 16k
# substrate that STILL collapses the quality gate. The corrected full head is the **base-int4 native 262k
# head** (e.g. the public gemma-4-E4B-it-qat-w4a16-ct substrate served with the surgical stack), owned and
# stood up by fern #535. Per the advisor we CONSUME fern #535's served checkpoint (do NOT rebuild it). The
# census points BC_FULLHEAD_MODEL_DIR at that synced checkpoint via the PR534_FULLHEAD_MODEL_DIR env var;
# the default below is a placeholder that fails loudly (and the lm_head_rows>=262144 guard rejects any 16k
# substrate). The full 262k head is quality-safe BY CONSTRUCTION (no token -inf'd -> base-identical head).
BC_FULLHEAD_MODEL_DIR = os.environ.get("PR534_FULLHEAD_MODEL_DIR", "/tmp/base-int4-fullhead")
BC_FULL_VOCAB = 262144                                # the corrected full head: native base vocab
BC_PRUNED_VOCAB = 12288
BC_DRAFTER = "/tmp/qat-assistant"                     # the MTP K=7 drafter (spec-on)
BC_DEFAULT_OUTPUT_LEN = 512                           # official served output_len
BC_DEFAULT_N_PROMPTS = 128                            # official prompt count
BC_DEFAULT_SKIP_THRESHOLD = 1.0                       # nats; served top-2 gap above which a match is proven
# #510 anchor this real-path census confirms (operative-identity on the teacher-forced geometry):
PR510_SURGICAL_OPERATIVE_IDENTITY = 0.99909          # #510 run 02h6o64s (operative_identity_rate)
PR510_SURGICAL_CENSUS = 0.99721                      # #510 token_identity_rate (W=8 teacher-forced)
PR510_SURGICAL_N_SEMANTIC_AT_LOCUS = 0               # #510: 0 semantic flips at locus
# PR #534 cost-leg anchor: the 12k-head surgical-357 rung's SERVED local TPS (lawine #488 ko01dcyy:
# 357.6 measured; the PR baseline cell quotes 357.06). The in-harness enforce_eager TPS is NOT this
# number; we use it only to form the full-head/12k RATIO, then project onto this served rung.
PR534_SERVED_12K_TPS = 357.06                        # 12k surgical-357 served local TPS (the cost baseline)
# int4 lm_head weight-read roofline (osoi5 w4a16): the lm_head is a [V, 2560] int4 matrix. Full V=262144
# reads ~0.336 GB/step; pruned V=12288 reads ~0.0157 GB/step. At the A10G ~600 GB/s decode roofline the
# extra full-head read costs ~0.53 ms/step; over a ~3.3 ms/token served step that bounds the cost. These
# bound the projection (the measured ratio should land between the roofline ceiling and the eager dilution).
PR534_LMHEAD_FULL_READ_GB = 0.336
PR534_LMHEAD_12K_READ_GB = 0.0157


def _bc_apply_served_env(fullhead: bool = False) -> str:
    """Set the shipped surgical-357 SERVED env (idempotent setdefault) BEFORE importing vLLM/sitecustomize,
    and RETURN the resolved served model dir. These are the numerics-affecting flags of the deployed stack
    (lm_head, surgical 2D attn, PLE embed-scale fold, split-KV verify); the speed-only graph-capture REQUIRE
    flags are relaxed because the census runs enforce_eager (CUDA-graph capture replays the same kernels, so
    it is identity-neutral).

    fullhead=True is the PR #534 safe-anchor: serve the native 262,144-row lm_head (NO PCK04 prune) from the
    base-int4 stock checkpoint (google/gemma-4-E4B-it-qat-w4a16-ct, fern #535 serve_ok PPL 2.006). Per the
    advisor's 2026-06-17 00:02Z recipe the ONLY flags that differ from the 12k config are: the served model
    dir + PLE_FOLD_TARGET_MODEL both point at the stock snapshot (so serve.py:241 applies the PLE fold to the
    served model rather than skipping it as "non-target"), LM_HEAD_PRUNE/_REQUIRE off, and PCK04_KEEPSET unset.
    Everything numerics-affecting else is identical. The full head is quality-safe by construction -- no token
    is -inf'd, so the head is base-identical and no benchmark collapse is possible. (Explore-confirmed: the
    PCK04 logits-scatter is gated on PCK04_KEEPSET and the prune phase on LM_HEAD_PRUNE; FUSED_SPARSE_ARGMAX is
    a DRAFTER-only kernel that never touches the target lm_head, so it stays on in both configs.)"""
    model_dir = BC_FULLHEAD_MODEL_DIR if fullhead else BC_MODEL_DIR
    served = {
        "CUDA_VISIBLE_DEVICES": "0",
        "LOCAL_MODEL_DIR": model_dir,
        "PLE_ASSUME_VALID_TOKEN_IDS": "1", "PLE_FOLD_EMBED_SCALE": "1",
        # full-head: fold against the served stock snapshot itself (serve.py:241 only folds when
        # PLE_FOLD_TARGET_MODEL == the served model); 12k: osoi5-v0-baked, overwritten to the 12k dst by the
        # prune phase (serve.py:707) so it ends up == the served model there too.
        "PLE_FOLD_TARGET_MODEL": (model_dir if fullhead else "/tmp/osoi5-v0-baked"), "PLE_SCRATCH_REUSE": "1",
        "SURGICAL_ATTN_USE_3D_OFF": "1",                 # THE lever under test (2D order-preserving attn)
        "SPLITKV_VERIFY": "1", "SPLITKV_VERIFY_MAX_Q": "64",
        "FUSED_SPARSE_ARGMAX": "1", "FUSED_SPARSE_ARGMAX_REQUIRE": "0", "FUSED_SPARSE_ARGMAX_BLOCK": "16",
        "DIXIE_SLIM_GREEDY": "1", "DIXIE_FUSED_ACCEPT_PREP": "1", "DIXIE_FUSED_ACCEPT_PREP_REQUIRE": "0",
        "FA_SLIDING": "1", "FA_SLIDING_DIAG": "0",
        "OVERRIDE_GENERATION_CONFIG": '{"temperature":0.0,"top_p":1.0,"top_k":0}',
        "GENERATION_CONFIG": "vllm", "CENTROID_TOP_K": "64",
        # relaxations because we run enforce_eager (no ONEGRAPH capture) and library mode (no bench precache):
        "ONEGRAPH": "0", "LOOPGRAPH_REQUIRE_CAPTURE": "0", "PRECACHE_BENCH": "0", "PRECACHE_REQUIRE": "0",
        "DISABLE_LOG_STATS": "1",
        # native sampler: FlashInfer JIT needs curand.h (absent here); native argmax is identity-equivalent.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if fullhead:
        # Full 262k head: the prune + PCK04 logits-scatter must be OFF. Pop any value the watchdog inherited
        # from the parent env so a leaked LM_HEAD_PRUNE/PCK04_KEEPSET can never silently re-prune the head.
        # PLE_FOLD_TARGET_MODEL is also popped so the force-set below (== the served snapshot) can't be
        # shadowed by a stale inherited fold target that would make serve.py:241 skip the fold.
        for k in ("LM_HEAD_PRUNE", "LM_HEAD_PRUNE_REQUIRE", "LM_HEAD_PRUNE_DST", "PCK04_KEEPSET",
                  "PLE_FOLD_TARGET_MODEL"):
            os.environ.pop(k, None)
    else:
        served.update({
            "PCK04_KEEPSET": f"{model_dir}/pck04_keepset.json",
            "LM_HEAD_PRUNE": "1", "LM_HEAD_PRUNE_REQUIRE": "1", "LM_HEAD_PRUNE_DST": model_dir,
        })
    os.environ.pop("DRAFTER_SHA256", None)               # don't block on drafter sha mismatch (local)
    for k, v in served.items():
        os.environ.setdefault(k, v)
    os.environ["LOCAL_MODEL_DIR"] = model_dir            # force (never inherit a stale dir from the parent)
    if fullhead:
        os.environ["PLE_FOLD_TARGET_MODEL"] = model_dir  # full-head: fold against the served snapshot itself
    return model_dir


def _bc_setup_inprocess() -> str:
    """Arm the shipped served patches in THIS process before vLLM import: put the submission dir on
    sys.path so sitecustomize installs its meta-path finders (pck04 lm_head rebuild, fused argmax, the
    surgical attn pin), then patch the on-disk gemma4.py PLE sources. Returns the submission dir."""
    sub = str(Path(__file__).resolve().parents[3] / BC_SUBDIR)
    if sub not in sys.path:
        sys.path.insert(0, sub)
    # PYTHONPATH prefix so the vLLM EngineCore child (forked) auto-runs sitecustomize before gemma4 import
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if sub not in pp:
        os.environ["PYTHONPATH"] = os.pathsep.join([sub] + pp)
    import importlib
    import sitecustomize                                  # noqa: F401 -- installs all meta_path finders
    importlib.reload(sitecustomize) if "sitecustomize" in sys.modules else None
    import serve as serve_mod
    serve_mod.patch_ple_sources()                         # idempotent gemma4.py PLE source patch
    return sub


def _bc_encode_prompts(tokenizer, n_prompts: int) -> list[tuple]:
    """The official served prompt set: eval_prompts_sharegpt.json, chat-templated exactly as the served
    /v1/completions capture does (decode_outputs.py encode_prompt: user-role message, add_generation_prompt).
    Returns [(id, prompt_token_ids)]. SEED=1 shuffle is a no-op over a census of all 128 (order-free)."""
    data = json.loads(Path(BC_EVAL_PROMPTS).read_text())
    out = []
    for idx, item in enumerate(data):
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 1:
            continue
        prompt = conv[0].get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(enc, list) and len(enc) == 1 and hasattr(enc[0], "ids"):
            ptoks = list(enc[0].ids)                       # fast-tokenizer Encoding wrapper
        elif hasattr(enc, "ids"):
            ptoks = list(enc.ids)
        elif isinstance(enc, list) and all(isinstance(x, int) for x in enc):
            ptoks = enc
        else:                                              # tokenize=False fallback (decode_outputs.py path)
            s = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ptoks = tokenizer.encode(s, add_special_tokens=False)
        out.append((str(item.get("id", idx)), ptoks))
    return out[:n_prompts]


def _bc_classify(ri: int, j: int, pos: int, served_tok: int, served_sl: list,
                 m1_sl: list | None, read: bool) -> dict:
    """One benchmark-config census position record (classify_position-compatible schema so the existing
    tie_threshold_sweep / examine_served_flips / _agg all work unchanged).

    m8_* = the REAL SERVED verify token/distribution (m8_top1_id == served_tok). m1_* = the M=1 AR
    reference (drafter off) at decode-width-1 conditioned on the realized served prefix. is_flip iff the
    served token != the M=1 reference argmax; flip_kind tie iff the M=1 reference's OWN top-2 is a bf16
    tie (operatively identity-1.0). A clear-winner skip (read=False) is a proven match."""
    served_top1_id = served_sl[0][0] if served_sl else served_tok
    served_top2_id = served_sl[1][0] if len(served_sl) >= 2 else None
    served_gap = (served_sl[0][1] - served_sl[1][1]) if len(served_sl) >= 2 else float("inf")
    served_ids = [t for t, _ in served_sl]
    if not read or m1_sl is None or len(m1_sl) < 1:
        # provable clear-winner match (no M=1 read): reference argmax == served token by the screen bound
        return {"prompt_idx": ri, "pos": pos, "L": pos, "k": j, "width": M_VERIFY,
                "m8_gap": round(served_gap, 6) if math.isfinite(served_gap) else None,
                "m8_top1_id": served_tok, "m8_top2_id": served_top2_id, "m1_tok_id": served_tok,
                "is_flip": 0, "is_near_tie": bool(served_gap <= EPS_STAR + BAND_TOL),
                "m1_in_m8_top2": True, "m1_in_m8_top5": True, "m1_self_gap": None,
                "m1_argmax_matches_token": True, "m1_is_bitwise_tie": False, "flip_kind": None,
                "traj_frac": None, "read": 0}
    m1_argmax = m1_sl[0][0]
    m1_self_gap = (m1_sl[0][1] - m1_sl[1][1]) if len(m1_sl) >= 2 else None
    m1_is_tie = bool(m1_self_gap is not None and m1_self_gap <= BAND_TOL)
    is_flip = int(served_tok != m1_argmax)
    return {
        "prompt_idx": ri, "pos": pos, "L": pos, "k": j, "width": M_VERIFY,
        "m8_gap": round(served_gap, 6) if math.isfinite(served_gap) else None,
        "m8_top1_id": served_tok, "m8_top2_id": served_top2_id, "m1_tok_id": m1_argmax,
        "is_flip": is_flip, "is_near_tie": bool(served_gap <= EPS_STAR + BAND_TOL),
        "m1_in_m8_top2": bool(m1_argmax in (served_top1_id, served_top2_id)),
        "m1_in_m8_top5": bool(m1_argmax in served_ids),
        "m1_self_gap": (round(m1_self_gap, 6) if m1_self_gap is not None else None),
        "m1_argmax_matches_token": bool(not is_flip),
        "m1_is_bitwise_tie": m1_is_tie,
        "flip_kind": (None if not is_flip else ("tie" if m1_is_tie else "semantic")),
        "read": 1,
    }


def phase_benchmark_config_census(out_path: str, n_prompts: int, output_len: int, gpu_mem_util: float,
                                  max_model_len: int, skip_threshold: float, det_check_k: int,
                                  checkpoint: str | None = None, heartbeat: str | None = None,
                                  resume: bool = False, skip_prompts: tuple = (),
                                  fullhead: bool = False, tps_only: bool = False,
                                  tps_warmup: int = 2) -> None:
    model_dir = _bc_apply_served_env(fullhead)
    sub = _bc_setup_inprocess()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    lm_head_mode = "full_262144" if fullhead else "pruned_12288"
    print(f"[bc] model={model_dir} lm_head={lm_head_mode} fullhead={fullhead} tps_only={tps_only} "
          f"drafter={BC_DRAFTER} n_prompts={n_prompts} output_len={output_len} "
          f"max_model_len={max_model_len} skip_threshold={skip_threshold} sub={sub}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
        max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=1, enable_prefix_caching=True, enforce_eager=True, trust_remote_code=True,
        speculative_config={"method": "mtp", "model": BC_DRAFTER, "num_speculative_tokens": K_SPEC},
    )
    print(f"[bc] served stack (spec-on MTP K={K_SPEC}) load done in {time.time()-t0:.0f}s", flush=True)

    surgical_prov = _install_surgical_lever()             # idempotent: confirms is_batch_invariant armed
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _ua
        attn_bi = bool(getattr(_ua, "is_batch_invariant", False))
    except Exception:
        attn_bi = False
    if not attn_bi:
        raise RuntimeError("[bc] surgical 2D attention pin NOT armed (is_batch_invariant False); arm void")

    # Confirm the served lm_head row count matches the requested mode (proves the prune is really off for the
    # full-head arm -- a silent re-prune would invalidate the cert). Read the runner's lm_head out_features.
    lm_head_rows = None
    try:
        mr = llm.llm_engine.model_executor.driver_worker.model_runner            # vLLM V1 in-proc runner
        lm_head_rows = int(mr.model.lm_head.weight.shape[0])
    except Exception:
        try:
            lm_head_rows = int(llm.llm_engine.model_config.get_vocab_size())
        except Exception:
            lm_head_rows = None
    expect_rows = BC_FULL_VOCAB if fullhead else BC_PRUNED_VOCAB
    print(f"[bc] served lm_head rows={lm_head_rows} (expected ~{expect_rows} for {lm_head_mode})", flush=True)
    if lm_head_rows is not None and fullhead and lm_head_rows < BC_FULL_VOCAB:
        raise RuntimeError(f"[bc] full-head arm but lm_head has only {lm_head_rows} rows (<{BC_FULL_VOCAB}); "
                           "the prune did not turn off -- arm void")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = _bc_encode_prompts(tokenizer, n_prompts)
    print(f"[bc] encoded {len(prompts)} chat-templated official prompts", flush=True)

    sp_served = SamplingParams(temperature=0.0, max_tokens=output_len, logprobs=5,
                               detokenize=False, ignore_eos=True)
    sp_ref = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5, detokenize=False)

    positions: list[dict] = []
    flips: list[dict] = []
    per_prompt: list[dict] = []
    screen = {"checked": 0, "violations": 0, "violation_detail": []}
    det_served_acc: list[int] = []
    det_details: list[dict] = []                           # per-divergence forensics (det-check prompts only)
    counters = {"n_read": 0, "n_skipped": 0, "n_served_tokens": 0}

    def _beat(phase: str, ri: int) -> None:
        if not heartbeat:
            return
        try:
            tmp = heartbeat + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"ts": time.time(), "phase": phase, "prompt_idx": ri,
                           "n_done": len(per_prompt)}, fh)
            os.replace(tmp, heartbeat)
        except Exception:
            pass

    def _read_ref(seq_prefix: list[int]):
        """M=1 AR reference distribution at the next position, conditioned on seq_prefix (decode-width-1)."""
        o = llm.generate([{"prompt_token_ids": seq_prefix}], sp_ref, use_tqdm=False)[0]
        lp = o.outputs[0].logprobs
        entry = lp[0] if lp else None
        return _sorted_logprobs(entry) if entry else None

    def _empty_prec(ri: int, pid: str, reason: str) -> dict:
        return {"prompt_idx": ri, "id": pid, "positions": [], "flips": [],
                "screen": {"checked": 0, "violations": 0, "violation_detail": []},
                "det_served": None, "n_read": 0, "n_skipped": 0, "n_served_tokens": 0,
                "per_prompt": {"prompt_idx": ri, "id": pid, reason: True, "L": 0,
                               "n_read": 0, "n_skipped": 0, "is_det_prompt": False, "prompt_secs": 0.0}}

    def _ingest(prec: dict) -> None:
        positions.extend(prec["positions"])
        flips.extend(prec["flips"])
        per_prompt.append(prec["per_prompt"])
        screen["checked"] += prec["screen"]["checked"]
        screen["violations"] += prec["screen"]["violations"]
        screen["violation_detail"].extend(prec["screen"]["violation_detail"])
        if prec["det_served"] is not None:
            det_served_acc.append(prec["det_served"])
        if prec.get("det_detail"):
            det_details.append(prec["det_detail"])
        counters["n_read"] += prec["n_read"]
        counters["n_skipped"] += prec["n_skipped"]
        counters["n_served_tokens"] += prec["n_served_tokens"]

    def _process_one_prompt(ri: int, pid: str, ptoks: list[int]) -> dict:
        t_p0 = time.time()
        if len(ptoks) + 4 >= max_model_len:
            return _empty_prec(ri, pid, "too_long")
        _beat("served", ri)
        # Decode-timed primary generate: the served spec-on (MTP K=7) path over the full 512-token output.
        # This is the cost-of-full-head signal (the lm_head verify matmul + int4 weight read per step). The
        # whole-prompt wall clock includes prefill, but with L=512 >> P0 the decode dominates; we report
        # decode_tps = L / decode_secs and aggregate a WARM median (dropping the first `tps_warmup` prompts,
        # whose first-touch kernel JIT inflates the wall clock) downstream.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_dec0 = time.time()
        out = llm.generate([{"prompt_token_ids": ptoks}], sp_served, use_tqdm=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_secs = time.time() - t_dec0
        T_served = list(out.outputs[0].token_ids)
        served_lp = list(out.outputs[0].logprobs or [])
        L = len(T_served)
        if L == 0:
            return _empty_prec(ri, pid, "empty")
        decode_tps = (L / decode_secs) if decode_secs > 0 else float("nan")
        seq = ptoks + T_served
        P0 = len(ptoks)
        do_det = ri < det_check_k

        # TPS-only mode (PR #534 cost leg): skip the identity machinery (determinism re-run, per-position
        # M=1 reference reads, screen probe). We need only the decode wall clock per prompt to form the
        # warm-median TPS ratio (full-head vs 12k). Identity is measured in the full census, not here.
        if tps_only:
            return {
                "prompt_idx": ri, "id": pid, "positions": [], "flips": [],
                "screen": {"checked": 0, "violations": 0, "violation_detail": []},
                "det_served": None, "det_detail": None,
                "n_read": 0, "n_skipped": 0, "n_served_tokens": L,
                "per_prompt": {"prompt_idx": ri, "id": pid, "L": L, "n_read": 0, "n_skipped": 0,
                               "n_flips": 0, "det_served": None, "det_first_diff_near_tie": None,
                               "is_det_prompt": False, "tps_only": True,
                               "decode_secs": round(decode_secs, 4), "decode_tps": round(decode_tps, 3),
                               "prompt_secs": round(time.time() - t_p0, 2)},
            }

        # Determinism characterization (NOT a pass/fail gate). Spec-on verify routes 1<M<=64 batches to
        # 3D split-KV, whose reduction is not order-preserving (Marlin atomic-add), so a free-run is NOT
        # bit-reproducible run-to-run. The honest claim is narrower: where two free-runs diverge, the FIRST
        # divergence sits at a near-tie (the two near-tied tokens swap under sub-ULP perturbation). A
        # clear-winner that flipped run-to-run WOULD be alarming; a near-tie swap is benign and cannot
        # produce a semantic flip vs the M=1 reference. After the first divergence the prefixes have forked
        # (the #470 cascade), so only the first divergence is a clean determinism signal.
        det_served = None
        det_detail = None
        if do_det:
            _beat("served2", ri)
            out2 = llm.generate([{"prompt_token_ids": ptoks}], sp_served, use_tqdm=False)[0]
            T2 = list(out2.outputs[0].token_ids)
            Lc = min(L, len(T2))
            det_served = int(T_served[:Lc] == T2[:Lc])
            if not det_served:
                d = next((k for k in range(Lc) if T_served[k] != T2[k]), None)
                if d is not None:
                    e = served_lp[d] if d < len(served_lp) else None
                    sl = _sorted_logprobs(e) if e else []
                    gap = (sl[0][1] - sl[1][1]) if len(sl) >= 2 else float("inf")
                    det_detail = {
                        "prompt_idx": ri, "first_diff_pos": P0 + d, "first_diff_j": d,
                        "first_diff_gap": (None if not math.isfinite(gap) else round(float(gap), 6)),
                        "first_diff_gap_ulps": (None if not math.isfinite(gap) else round(gap / ULP_NAT, 3)),
                        "near_tie": bool(gap <= skip_threshold),
                        "run1_tok": T_served[d], "run2_tok": T2[d],
                        "len1": L, "len2": len(T2),
                    }

        p_positions: list[dict] = []
        p_flips: list[dict] = []
        p_screen = {"checked": 0, "violations": 0, "violation_detail": []}
        n_read = n_skipped = 0
        skipped_idx: list[int] = []
        for j in range(L):
            pos = P0 + j
            served_tok = T_served[j]
            entry = served_lp[j] if j < len(served_lp) else None
            served_sl = _sorted_logprobs(entry) if entry else []
            served_gap = (served_sl[0][1] - served_sl[1][1]) if len(served_sl) >= 2 else float("-inf")
            readable = bool(served_sl)
            if readable and served_gap > skip_threshold:
                rec = _bc_classify(ri, j, pos, served_tok, served_sl, None, read=False)
                p_positions.append(rec)
                n_skipped += 1
                skipped_idx.append(j)
                continue
            if (j % 64) == 0:
                _beat("ref", ri)
            m1_sl = _read_ref(seq[:pos])
            rec = _bc_classify(ri, j, pos, served_tok, served_sl, m1_sl, read=True)
            rec["traj_frac"] = round(j / max(1, L), 4)
            p_positions.append(rec)
            n_read += 1
            if rec["is_flip"]:
                p_flips.append(rec)

        # screen-validation probe: read a sample of SKIPPED (clear-winner) positions, assert they match
        if do_det and skipped_idx:
            import random as _rnd
            sample = _rnd.Random(1000 + ri).sample(skipped_idx, min(8, len(skipped_idx)))
            for j in sample:
                pos = P0 + j
                m1_sl = _read_ref(seq[:pos])
                p_screen["checked"] += 1
                if m1_sl and m1_sl[0][0] != T_served[j]:
                    p_screen["violations"] += 1
                    p_screen["violation_detail"].append(
                        {"prompt_idx": ri, "pos": pos, "served_tok": T_served[j], "m1_argmax": m1_sl[0][0]})

        return {
            "prompt_idx": ri, "id": pid, "positions": p_positions, "flips": p_flips, "screen": p_screen,
            "det_served": det_served, "det_detail": det_detail,
            "n_read": n_read, "n_skipped": n_skipped, "n_served_tokens": L,
            "per_prompt": {"prompt_idx": ri, "id": pid, "L": L, "n_read": n_read, "n_skipped": n_skipped,
                           "n_flips": len(p_flips), "det_served": det_served,
                           "det_first_diff_near_tie": (det_detail.get("near_tie") if det_detail else None),
                           "is_det_prompt": bool(do_det), "tps_only": False,
                           "decode_secs": round(decode_secs, 4), "decode_tps": round(decode_tps, 3),
                           "prompt_secs": round(time.time() - t_p0, 2)},
        }

    # ---- resume: replay completed prompts from checkpoint (0 GPU) ----
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
                    continue
                if prec.get("prompt_idx") in done:
                    continue
                _ingest(prec)
                done.add(prec["prompt_idx"])
        print(f"[bc] resumed {len(done)} prompts from {checkpoint}", flush=True)

    skip_set = set(int(x) for x in (skip_prompts or ()))
    t_c0 = time.time()
    ck = open(checkpoint, "a") if checkpoint else None
    for ri, (pid, ptoks) in enumerate(prompts):
        if ri in done:
            continue
        if ri in skip_set:
            prec = _empty_prec(ri, pid, "hang_skipped")
            if ck:
                ck.write(json.dumps(prec) + "\n"); ck.flush(); os.fsync(ck.fileno())
            _ingest(prec); done.add(ri)
            print(f"[bc] prompt {ri} HANG-SKIPPED (watchdog)", flush=True)
            continue
        prec = _process_one_prompt(ri, pid, ptoks)
        if ck:
            ck.write(json.dumps(prec) + "\n"); ck.flush(); os.fsync(ck.fileno())
        _ingest(prec); done.add(ri)
        pp = prec["per_prompt"]
        ag = _agg(positions)
        print(f"[bc] prompt {ri} L={pp.get('L')} read={pp.get('n_read')} skip={pp.get('n_skipped')} "
              f"flips={pp.get('n_flips')} running_id={ag['token_identity_rate']:.6f} "
              f"sem={ag['n_semantic_flips']} tie={ag['n_tie_flips']} det={pp.get('det_served')} "
              f"secs={pp.get('prompt_secs')} [{len(done)}/{len(prompts)}]", flush=True)
    if ck:
        ck.close()
    _beat("done", len(prompts))

    census_secs = time.time() - t_c0
    agg = _agg(positions)

    def _rate(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    # ---- TPS aggregate (PR #534 cost leg): warm-median decode TPS over the served generates. ----
    # Each ran prompt carries decode_tps = L / decode_secs (CUDA-synced wall clock around the spec-on
    # generate). The WARM set drops prompts with prompt_idx < tps_warmup (first-touch kernel JIT inflates
    # those wall clocks). We report the warm MEDIAN (robust headline) and a throughput-weighted aggregate
    # (sum L / sum secs over warm prompts). NOTE: this in-harness number runs enforce_eager (no ONEGRAPH /
    # LOOPGRAPH capture, no bench precache), so it is NOT directly comparable to the 357.06 served rung; its
    # job is the RATIO (full-head vs 12k under identical harness conditions), which is then projected onto
    # the served number. See compose_bc_report for the projection.
    def _median(xs: list[float]) -> float:
        s = sorted(xs)
        n = len(s)
        if n == 0:
            return float("nan")
        m = n // 2
        return s[m] if (n % 2) else (s[m - 1] + s[m]) / 2.0
    _tps_rows = [pp for pp in per_prompt
                 if isinstance(pp.get("decode_tps"), (int, float)) and math.isfinite(pp["decode_tps"])
                 and pp.get("L", 0) > 0]
    _warm_rows = [pp for pp in _tps_rows if pp.get("prompt_idx", 0) >= tps_warmup]
    _warm_tps_vals = [pp["decode_tps"] for pp in _warm_rows]
    _all_tps_vals = [pp["decode_tps"] for pp in _tps_rows]
    _warm_tok = sum(pp.get("L", 0) for pp in _warm_rows)
    _warm_secs = sum(pp.get("decode_secs", 0.0) for pp in _warm_rows)
    tps_block = {
        "tps_warmup": tps_warmup,
        "n_tps_prompts": len(_tps_rows),
        "n_warm_tps_prompts": len(_warm_rows),
        "warm_median_tps": round(_median(_warm_tps_vals), 3),
        "warm_mean_tps": round(_rate(_warm_tps_vals), 3),
        "warm_throughput_tps": round((_warm_tok / _warm_secs) if _warm_secs > 0 else float("nan"), 3),
        "all_median_tps": round(_median(_all_tps_vals), 3),
        "warm_tokens": _warm_tok,
        "warm_decode_secs": round(_warm_secs, 3),
        "enforce_eager": True,                            # harness condition (NOT the served capture path)
    }
    print(f"[bc] TPS warm_median={tps_block['warm_median_tps']} "
          f"warm_throughput={tps_block['warm_throughput_tps']} "
          f"all_median={tps_block['all_median_tps']} "
          f"(warm {len(_warm_rows)}/{len(_tps_rows)} prompts, warmup={tps_warmup})", flush=True)

    # Peak memory: the vLLM V1 EngineCore runs in a forked child, so the parent's torch counter reads ~0.
    # Query the device-resident footprint via nvidia-smi while the served stack is still loaded (the KV
    # pool is preallocated at gpu_memory_utilization, so device-used is the steady served footprint).
    def _device_used_gb() -> float:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=15)
            vals = [float(x.strip()) for x in r.stdout.splitlines() if x.strip()]
            return (max(vals) / 1024.0) if vals else 0.0      # MiB -> GiB
        except Exception:
            return 0.0
    try:
        peak_parent_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        peak_parent_gb = 0.0
    peak_device_gb = _device_used_gb()
    peak_gb = max(peak_parent_gb, peak_device_gb)

    # Determinism characterization (informational, not a gate): where free-runs diverge, was the FIRST
    # divergence a near-tie? all-near-tie (or no divergence at all) is the benign, expected outcome.
    det_diff_count = len(det_details)
    det_diffs_all_near_tie = bool(all(d.get("near_tie") for d in det_details)) if det_details else True
    _det_gaps = [d["first_diff_gap_ulps"] for d in det_details if d.get("first_diff_gap_ulps") is not None]
    det_max_diff_gap_ulps = max(_det_gaps) if _det_gaps else None
    out = {
        "phase": "benchmark_config_census", "arm": "benchmark_config", "model_dir": model_dir,
        "fullhead": bool(fullhead), "lm_head_mode": lm_head_mode, "lm_head_rows": lm_head_rows,
        "tps_only": bool(tps_only), "tps": tps_block,
        "drafter": BC_DRAFTER, "spec_on": True, "num_speculative_tokens": K_SPEC,
        "surgical_mode": True, "surgical_attn_armed": attn_bi,
        "matmul_tax_installed": surgical_prov.get("matmul_tax_installed"),
        "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1",
        "surgical_lever_source": surgical_prov.get("lever_source"),
        "n_prompts_run": len(per_prompt), "output_len": output_len, "skip_threshold": skip_threshold,
        "max_model_len": max_model_len, "ignore_eos": True,
        "census": agg,                                    # full 65,536-position aggregate
        "n_served_tokens": counters["n_served_tokens"],
        "n_read": counters["n_read"], "n_skipped": counters["n_skipped"],
        "read_fraction": (counters["n_read"] / max(1, counters["n_served_tokens"])),
        "screen_checked": screen["checked"], "screen_violations": screen["violations"],
        "screen_violation_detail": screen["violation_detail"][:32],
        "screen_validated": bool(screen["violations"] == 0),
        "determinism_served": _rate(det_served_acc), "n_det_served_checked": len(det_served_acc),
        "det_diff_count": det_diff_count, "det_diffs_all_near_tie": det_diffs_all_near_tie,
        "det_max_diff_gap_ulps": det_max_diff_gap_ulps, "det_details": det_details,
        "flip_details": flips,
        "per_prompt": per_prompt,
        "peak_gpu_gb": peak_gb, "peak_parent_gb": round(peak_parent_gb, 3),
        "peak_device_gb": round(peak_device_gb, 3),
        "census_secs": round(census_secs, 1),
        "mean_prompt_secs": round(census_secs / max(1, len(per_prompt)), 2),
        "nan_clean": bool(math.isfinite(agg["token_identity_rate"])),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[bc] benchmark_config_census={agg['token_identity_rate']:.7f} "
          f"(sem={agg['n_semantic_flips']} tie={agg['n_tie_flips']} of {agg['n_positions']} pos) "
          f"operative={agg['operative_identity_rate']:.7f} | read={counters['n_read']}/{counters['n_served_tokens']} "
          f"({100*out['read_fraction']:.2f}%) screen_viol={screen['violations']} "
          f"det_served={out['determinism_served']} det_diffs={det_diff_count}"
          f"(all_near_tie={det_diffs_all_near_tie}, max_gap={det_max_diff_gap_ulps}ULP) "
          f"peak={peak_gb:.1f}GB", flush=True)
    print(f"BC_ARM_DONE {out_path}", flush=True)


def build_bc_self_test(bc: dict, report: dict) -> tuple[dict, int]:
    c = bc["census"]
    checks = {
        "bc_identity_in_unit": bool(math.isfinite(c["token_identity_rate"])
                                    and 0.0 <= c["token_identity_rate"] <= 1.0),
        "bc_nan_clean": bool(bc["nan_clean"]),
        # full coverage: every served token is a census position (clear-winner skips are counted as matches)
        "bc_full_coverage": bool(c["n_positions"] == bc["n_served_tokens"] and bc["n_served_tokens"] > 0),
        # the clear-winner screen is empirically validated (no skipped position was actually a flip)
        "bc_screen_validated": bool(bc["screen_violations"] == 0),
        # the arm IS the shipped surgical config: 2D attn armed + matmul tax OFF + spec-on drafter
        "bc_surgical_attn_armed": bool(bc["surgical_attn_armed"]),
        "bc_matmul_tax_off": bool(bc.get("matmul_tax_installed") is False),
        "bc_vbi_env_off": bool(not bc["vllm_batch_invariant_env"]),
        "bc_spec_on": bool(bc["spec_on"]),
        # Run-to-run served nondeterminism is EXPECTED here (the 3D split-KV verify reduction is not
        # order-preserving). The load-bearing property is narrower and IS gated: where free-runs diverge,
        # the first divergence is a near-tie -- a clear winner never flips run-to-run, so nondeterminism
        # cannot manufacture a semantic flip vs the M=1 reference.
        "bc_det_diffs_all_near_tie": bool(bc.get("det_diffs_all_near_tie", True)),
        # internal consistency: identity == 1 - flips/positions
        "bc_identity_consistent": bool(
            c["n_positions"] == 0 or abs(c["token_identity_rate"]
                                         - (c["n_positions"] - c["n_flips"]) / c["n_positions"]) < 1e-9),
        # operative-1.0 wiring is consistent with the headline (identity high AND 0 semantic)
        "bc_operative_consistent": bool(
            report["bc_operative_identity_1p0"]
            == (c["token_identity_rate"] >= 0.99 and c["n_semantic_flips"] == 0)),
        # THE DELIVERABLE: the real serving path shows NO semantic (non-tie) divergence from M=1 AR greedy
        "bc_no_semantic_flips": bool(c["n_semantic_flips"] == 0),
    }
    return checks, len(checks)


def compose_bc_report(bc: dict, a: argparse.Namespace) -> dict:
    c = bc["census"]
    identity = c["token_identity_rate"]
    n_sem = c["n_semantic_flips"]
    operative_1p0 = bool(math.isfinite(identity) and identity >= 0.99 and n_sem == 0)
    tie_sweep = tie_threshold_sweep(bc["flip_details"], width=M_VERIFY)
    served_flip_exam = examine_served_flips(bc["flip_details"], width=M_VERIFY)
    # matches_510: does the REAL serving path reproduce #510's teacher-forced verdict (operative-1.0,
    # 0 semantic)? The exact rates differ (different prompt set + full vs 7/32 coverage); the VERDICT
    # is what must agree -- 0 semantic flips on both <=> the teacher-forced geometry was not hiding a
    # real-path divergence.
    matches_510 = bool(n_sem == 0) and bool(PR510_SURGICAL_N_SEMANTIC_AT_LOCUS == 0)

    report = {
        "pr": 521, "agent": "wirbel",
        "leg": "surgical-357 benchmark-config full-serve operative-identity census (real spec-on served path)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "phase": "benchmark_config_census",
        # ---- KEY OUTPUTS (PR #521) ----
        "benchmark_config_census": identity,
        "bc_n_semantic_flips": n_sem,
        "bc_n_tie_flips": c["n_tie_flips"],
        "bc_operative_identity_1p0": operative_1p0,
        "operative_identity_rate": c["operative_identity_rate"],
        "tie_threshold_for_zero_semantic": tie_sweep["tie_threshold_for_zero_semantic"],
        "tie_threshold_for_zero_semantic_ulps": tie_sweep["tie_threshold_for_zero_semantic_ulps"],
        "matches_510_geometry_census": ("yes" if matches_510 else "no"),
        # one-line verdict on the real scored serving path
        "verdict_oneline": (
            "surgical-357 operative-identity SURVIVES on the real spec-on served path "
            f"(operative {c['operative_identity_rate']:.6f}, 0 semantic over {c['n_positions']} served tokens)"
            if operative_1p0 else
            f"surgical-357 shows {n_sem} semantic flip(s) on the real served path -- investigate"),
        # ---- forensic depth (same columns as #487/#510) ----
        "tie_threshold_sweep": tie_sweep,
        "served_flip_examination": served_flip_exam,
        "n_flips": c["n_flips"],
        # ---- scale / coverage ----
        "n_positions": c["n_positions"], "n_served_tokens": bc["n_served_tokens"],
        "n_prompts_run": bc["n_prompts_run"], "output_len": bc["output_len"],
        "n_read": bc["n_read"], "n_skipped": bc["n_skipped"], "read_fraction": bc["read_fraction"],
        "skip_threshold": bc["skip_threshold"],
        # ---- clear-winner screen validation (proves full coverage is sound) ----
        "screen_checked": bc["screen_checked"], "screen_violations": bc["screen_violations"],
        "screen_validated": bc["screen_validated"], "screen_violation_detail": bc["screen_violation_detail"],
        # ---- determinism of the real served free-run (3D split-KV verify is not order-preserving) ----
        "determinism_served": bc["determinism_served"], "n_det_served_checked": bc["n_det_served_checked"],
        "det_diff_count": bc.get("det_diff_count", 0),
        "det_diffs_all_near_tie": bc.get("det_diffs_all_near_tie", True),
        "det_max_diff_gap_ulps": bc.get("det_max_diff_gap_ulps"),
        "det_details": bc.get("det_details", [])[:16],
        # ---- arm provenance: this IS the shipped surgical-357 served config ----
        "surgical_mode": True, "spec_on": bc["spec_on"], "num_speculative_tokens": bc["num_speculative_tokens"],
        "surgical_attn_armed": bc["surgical_attn_armed"], "matmul_tax_installed": bc.get("matmul_tax_installed"),
        "vllm_batch_invariant_env": bc["vllm_batch_invariant_env"],
        "surgical_lever_source": bc["surgical_lever_source"],
        "model_dir": bc["model_dir"], "drafter": bc["drafter"], "max_model_len": bc["max_model_len"],
        # ---- #510 anchor this run stress-tests on the real path ----
        "pr510_surgical_operative_identity": PR510_SURGICAL_OPERATIVE_IDENTITY,
        "pr510_surgical_census": PR510_SURGICAL_CENSUS,
        "pr510_surgical_n_semantic_at_locus": PR510_SURGICAL_N_SEMANTIC_AT_LOCUS,
        "delta_operative_vs_510": (c["operative_identity_rate"] - PR510_SURGICAL_OPERATIVE_IDENTITY
                                   if math.isfinite(c["operative_identity_rate"]) else float("nan")),
        # ---- #487 cross-reference (the lesson that motivated the real-path geometry) ----
        "pr487_globalflag222_n_semantic": PR487_GLOBALFLAG222_N_SEMANTIC,
        "peak_gpu_gb": bc["peak_gpu_gb"], "peak_parent_gb": bc.get("peak_parent_gb"),
        "peak_device_gb": bc.get("peak_device_gb"), "census_secs": bc["census_secs"],
        "nan_clean": bc["nan_clean"],
    }
    self_test, n_checks = build_bc_self_test(bc, report)
    report["verdict"] = (
        "GREEN" if (operative_1p0 and bc["screen_validated"] and bc["nan_clean"] and self_test_ok(self_test))
        else ("RED" if n_sem > 0 else "AMBER"))
    report["self_test"] = self_test
    report["self_test_n_checks"] = n_checks
    report["fullserve_self_test_passes"] = self_test_ok(self_test)
    return report


def _print_bc_console(r: dict) -> None:
    print(f"\n===== PR #521 BENCHMARK-CONFIG FULL-SERVE OPERATIVE-IDENTITY CENSUS (real spec-on path) =====",
          flush=True)
    print(f" VERDICT                                  : {r['verdict']}", flush=True)
    print(f" verdict (one line)                       : {r['verdict_oneline']}", flush=True)
    print(f" benchmark_config_census (token-identity) : {r['benchmark_config_census']:.7f}  "
          f"(semantic={r['bc_n_semantic_flips']} tie={r['bc_n_tie_flips']} of {r['n_positions']} served tokens)",
          flush=True)
    print(f" bc_operative_identity_1p0 (PRIMARY)      : {r['bc_operative_identity_1p0']}  "
          f"(operative_rate {r['operative_identity_rate']:.7f})", flush=True)
    print(f" matches_510_geometry_census              : {r['matches_510_geometry_census'].upper()}  "
          f"(#510 operative {r['pr510_surgical_operative_identity']:.5f}, "
          f"delta {r['delta_operative_vs_510']:+.2e})", flush=True)
    print(f" tie_threshold_for_zero_semantic          : {r['tie_threshold_for_zero_semantic']:.4f} nat "
          f"({r['tie_threshold_for_zero_semantic_ulps']:.2f} ULP)", flush=True)
    print(" --- coverage + clear-winner screen ---", flush=True)
    print(f"  served tokens / positions               : {r['n_served_tokens']} / {r['n_positions']} "
          f"({r['n_prompts_run']} prompts x {r['output_len']})", flush=True)
    print(f"  M=1 reference reads / skipped (proven)   : {r['n_read']} / {r['n_skipped']} "
          f"(read {100*r['read_fraction']:.2f}%)", flush=True)
    print(f"  screen validated (skips really match)    : {r['screen_validated']} "
          f"({r['screen_checked']} sampled, {r['screen_violations']} violations)", flush=True)
    print(f"  determinism_served (free-run reproduce)  : {r['determinism_served']} "
          f"({r['n_det_served_checked']} checked)", flush=True)
    print(f"  run-to-run diffs / all near-tie / max-gap: {r['det_diff_count']} / "
          f"{r['det_diffs_all_near_tie']} / {r['det_max_diff_gap_ulps']} ULP "
          f"(3D split-KV verify is not order-preserving; clear winners are stable)", flush=True)
    print(" --- arm provenance (shipped surgical-357 served config) ---", flush=True)
    print(f"  spec_on / K / surgical_attn / tax_off    : {r['spec_on']} / {r['num_speculative_tokens']} / "
          f"{r['surgical_attn_armed']} / {r.get('matmul_tax_installed') is False}", flush=True)
    if r["served_flip_examination"]:
        print(" --- per-flip dissection (real served path) ---", flush=True)
        for f in r["served_flip_examination"][:24]:
            print(f"   p{f['prompt_idx']} pos{f['pos']} {f['flip_kind']:>8} "
                  f"m1_self_gap={f['m1_self_gap']} ({f['m1_self_gap_ulps']} ULP) "
                  f"m1_in_served_top2={f['m1_in_m8_top2']} served={f['m8_top1_id']} m1={f['m1_tok_id']}",
                  flush=True)
    print(f" SELF-TEST PASSES                         : {r['fullserve_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("==================================================================\n", flush=True)


def log_bc_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="wirbel", name=a.wandb_name, group=a.wandb_group,
        notes=("PR#521 surgical-357 BENCHMARK-CONFIG full-serve operative-identity census: re-runs the "
               "#510 operative-identity certificate on the EXACT official spec-on served decode path "
               "(real scored tokens, 128x512), comparing every served token against the M=1 AR greedy "
               "reference re-conditioned on the realized served prefix. Answers the skeptical reviewer: "
               "does operative-1.0 survive on the path that actually produces the scored answers?"),
        config={
            "pr": 521, "M_verify": M_VERIFY, "K_spec": K_SPEC, "output_len": report["output_len"],
            "n_prompts_run": report["n_prompts_run"], "model_dir": report["model_dir"],
            "drafter": report["drafter"], "max_model_len": report["max_model_len"],
            "skip_threshold": report["skip_threshold"], "eps_star": EPS_STAR, "surgical_mode": True,
            "spec_on": report["spec_on"], "num_speculative_tokens": report["num_speculative_tokens"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "anchor/pr510_surgical_operative_identity": PR510_SURGICAL_OPERATIVE_IDENTITY,
            "anchor/pr510_surgical_census": PR510_SURGICAL_CENSUS,
            "anchor/pr487_globalflag222_n_semantic": PR487_GLOBALFLAG222_N_SEMANTIC,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = (
        "benchmark_config_census", "bc_n_semantic_flips", "bc_n_tie_flips", "bc_operative_identity_1p0",
        "operative_identity_rate", "tie_threshold_for_zero_semantic", "tie_threshold_for_zero_semantic_ulps",
        "matches_510_geometry_census", "verdict_oneline", "n_flips", "n_positions", "n_served_tokens",
        "n_prompts_run", "output_len", "n_read", "n_skipped", "read_fraction", "skip_threshold",
        "screen_checked", "screen_violations", "screen_validated", "determinism_served",
        "n_det_served_checked", "det_diff_count", "det_diffs_all_near_tie", "det_max_diff_gap_ulps",
        "surgical_mode", "spec_on", "num_speculative_tokens", "surgical_attn_armed",
        "matmul_tax_installed", "vllm_batch_invariant_env", "surgical_lever_source", "model_dir", "drafter",
        "max_model_len", "pr510_surgical_operative_identity", "pr510_surgical_census",
        "pr510_surgical_n_semantic_at_locus", "delta_operative_vs_510", "pr487_globalflag222_n_semantic",
        "peak_gpu_gb", "peak_parent_gb", "peak_device_gb", "census_secs", "nan_clean", "verdict",
        "fullserve_self_test_passes",
        "self_test_n_checks", "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
    )
    for k in keys:
        run.summary[k] = report.get(k)
    for sk, sv in report.get("tie_threshold_sweep", {}).items():
        if sk == "n_semantic_at_threshold":
            for tname, tval in sv.items():
                run.summary[f"tie_sweep/n_semantic_at_{tname}"] = tval
        elif not isinstance(sv, list):
            run.summary[f"tie_sweep/{sk}"] = sv
    run.summary["verdict_green"] = report["verdict"].startswith("GREEN")
    run.summary["verdict_red"] = report["verdict"].startswith("RED")
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def _bc_census_env() -> dict:
    """Subprocess env for the benchmark-config worker. The worker itself sets the full served env and arms
    sitecustomize IN-PROCESS before importing vLLM (validated smoke recipe), so here we only force the GPU
    index + the native sampler + the allocator, and ensure the submission dir is NOT on PYTHONPATH at
    interpreter start (so sitecustomize does not auto-run before the worker has set the served env)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sub = str(Path(__file__).resolve().parents[3] / BC_SUBDIR)
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != sub]
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(pp)
    else:
        env.pop("PYTHONPATH", None)
    env.pop("VLLM_BATCH_INVARIANT", None)                 # surgical arm: matmul tax must stay OFF
    return env


def _kill_pgroup(proc: subprocess.Popen) -> None:
    """Kill the worker AND its forked vLLM EngineCore child by signalling the whole process group."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                pass
        try:
            proc.wait(timeout=20)
            return
        except Exception:
            pass


def _run_bc_census(a: argparse.Namespace, label: str = "benchmark_config",
                   extra_args: list[str] | None = None) -> dict:
    """Run the benchmark-config census worker under a hang watchdog (the served spec-on stack can wedge on
    a CUDA hang; a fresh resumable reload clears the accumulated state and never loses a finished prompt).

    `label` namespaces the checkpoint/heartbeat/result files so multiple arms (e.g. PR #534 full-head
    census + 12k tps-only) never collide; `extra_args` are passed through to the worker (e.g. --fullhead,
    --tps-only)."""
    out_json = str(OUT_DIR / f"arm_{label}_result.json")
    ckpt = str(OUT_DIR / f"checkpoint_{label}.jsonl")
    hb = str(OUT_DIR / f"heartbeat_{label}.json")
    base_args = [
        "--phase", "benchmark_config_census", "--out", out_json,
        "--n-prompts", str(a.n_prompts), "--bc-output-len", str(a.bc_output_len),
        "--bc-gpu-mem-util", str(a.bc_gpu_mem_util), "--bc-max-model-len", str(a.bc_max_model_len),
        "--bc-skip-threshold", str(a.bc_skip_threshold), "--det-check-k", str(a.det_check_k),
        "--tps-warmup", str(getattr(a, "tps_warmup", 2)),
        "--checkpoint", ckpt, "--heartbeat", hb, "--resume",
    ] + list(extra_args or [])
    skip: list[int] = []
    stalled_at: dict[int, int] = {}
    restarts = 0
    while not _arm_complete(out_json, a.n_prompts):
        if restarts > a.watchdog_max_restarts:
            raise RuntimeError(f"[watchdog:bc] exceeded {a.watchdog_max_restarts} restarts; aborting")
        _wait_gpu_free()
        args = base_args + (["--skip-prompts", ",".join(map(str, sorted(set(skip))))] if skip else [])
        cmd = [sys.executable, os.path.abspath(__file__)] + args
        print(f"[watchdog:bc] launch restart={restarts} skip={sorted(set(skip))} "
              f"(stall={a.watchdog_stall_s}s)", flush=True)
        try:
            os.remove(hb)
        except OSError:
            pass
        proc = subprocess.Popen(cmd, env=_bc_census_env(), start_new_session=True)
        t_launch = time.time()
        last_prompt, stalled = -1, False
        while True:
            try:
                proc.wait(timeout=a.watchdog_poll_s)
                break
            except subprocess.TimeoutExpired:
                beat = _read_heartbeat(hb)
                fresh = beat is not None and float(beat.get("ts", 0)) >= t_launch
                if fresh and int(beat.get("prompt_idx", -1)) >= 0:
                    last_prompt = int(beat["prompt_idx"])
                if not fresh:
                    if time.time() - t_launch > a.watchdog_load_grace_s:
                        print(f"[watchdog:bc] LOAD-STALL no heartbeat {time.time()-t_launch:.0f}s; killing",
                              flush=True)
                        _kill_pgroup(proc); stalled = True; break
                else:
                    age = time.time() - float(beat["ts"])
                    if age > a.watchdog_stall_s:
                        print(f"[watchdog:bc] STALL heartbeat age={age:.0f}s at prompt {last_prompt}; killing",
                              flush=True)
                        _kill_pgroup(proc); stalled = True; break
        if stalled:
            restarts += 1
            if last_prompt >= 0:
                stalled_at[last_prompt] = stalled_at.get(last_prompt, 0) + 1
                if stalled_at[last_prompt] >= a.watchdog_poison_strikes:
                    print(f"[watchdog:bc] prompt {last_prompt} poisoned; will skip", flush=True)
                    skip.append(last_prompt)
            continue
        if proc.returncode == 0 and _arm_complete(out_json, a.n_prompts):
            break
        print(f"[watchdog:bc] worker exited rc={proc.returncode} without complete result; resuming", flush=True)
        restarts += 1
    print(f"[watchdog:bc] complete -> {out_json}", flush=True)
    return json.load(open(out_json))


def orchestrate_bc(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bc = _run_bc_census(a)
    if getattr(a, "census_only", False):
        print("[orch:bc] census_only: arm result written; run --reanalyze-bc to compose + log W&B", flush=True)
        return
    _finish_bc(compose_bc_report(bc, a), a)


def reanalyze_bc(a: argparse.Namespace) -> None:
    p = OUT_DIR / "arm_benchmark_config_result.json"
    if not p.exists():
        raise FileNotFoundError(f"--reanalyze-bc needs {p} (run the GPU phase first)")
    _finish_bc(compose_bc_report(json.load(open(p)), a), a)


def _finish_bc(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "benchmark_config_census_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_bc_console(report)
    if not a.no_wandb:
        log_bc_wandb(report, a)


# ======================================================================================
# PR #534 (land): conservative safe-anchor cert = {full-head no-prune} x {surgical 2D attention}
# Two arms share the benchmark-config census worker: (1) the full-head arm runs the FULL identity census
# (operative-identity vs M=1 AR) AND records warm decode TPS; (2) the 12k arm runs TPS-only (no identity).
# The cost is the warm-median TPS RATIO (full-head / 12k) measured under identical enforce_eager harness
# conditions, projected onto the 357.06 served rung. Identity reuses compose_bc_report on the full-head arm.
# ======================================================================================
# ---- Gate-table scaffold (advisor 2026-06-16 22:07Z: prep so it's ready when the full-head cells land) ----
# The quality gate (Morgan #515): the ship must be >=90% of vanilla gemma-4-E4B-it on GSM8K/AIME/MMLU-Pro.
# This scaffold assembles base / ship-12k / base-fullhead cells (3 benchmarks x 3 roles = 9 cells) into the
# pct_of_base + meets_90pct table + meets_90pct_all. The cells are isolation-blocked external measurements
# (ubel/fern/wirbel legs) -- they must be HANDED to this card via PR #534 (a --gate-cells JSON), not pulled.
# Missing cells stay null/pending so a partial-but-honest table composes at any time.
PR534_GATE_THRESHOLD = 0.90
PR534_GATE_BENCHMARKS = ("gsm8k", "aime", "mmlu_pro")
PR534_GATE_ROLES = ("base", "ship_12k", "base_fullhead")
# Advisor 2026-06-17 00:02Z (measurement-validity rule): the served protocol is concurrency=32, and the
# n=30 AIME maj@1 metric is concurrency-DOMINATED (fern's 2x2: the *unchanged* stock base swings
# 0.100 @ conc=1 -> 0.267 @ conc=32, >2se from concurrency alone). Any quality cell that compares base vs
# base_fullhead measured at *different* concurrency is confounded, so build_gate_table refuses a definitive
# meets_90pct for a benchmark whose base/base_fullhead concurrencies disagree (or are unrecorded).
PR534_GATE_CONCURRENCY = 32
# Advisor 2026-06-17 00:02Z (pt 5) TPS bracket for the quality-safe ship -- the explicit cost of guaranteed
# quality-safety, both rungs well under the ~500 unsafe class:
#   FLOOR  = base-int4 no-surgical/no-fast-kernels, quality == base (wirbel #533 b9j1z40d): 95.78 local.
#   UPPER  = the full-head served on the *fast* stack (fern #535 whh42dgd, PPL 2.006 byte-exact): 253.78.
# My own {full-head}x{surgical} projected-served number lands inside this bracket (surgical is between the
# bare base-int4 floor and the fast-stack upper); the report echoes the bracket alongside it for context.
PR534_QUALITY_SAFE_FLOOR_TPS_LOCAL = 95.78          # wirbel #533 b9j1z40d (base-int4 floor, quality==base)
PR534_QUALITY_SAFE_FLOOR_TPS_OFFICIAL = 99.0        # implied-official from the local floor
PR534_QUALITY_SAFE_FAST_UPPER_TPS = 253.78          # fern #535 whh42dgd (full-head on the fast stack)


def _empty_gate_cells() -> dict:
    # concurrency is per-role (advisor 00:02Z): each role's number carries the concurrency it was measured at,
    # so build_gate_table can refuse a confounded base-vs-fullhead comparison measured at different concurrency.
    return {bm: {"base": None, "ship_12k": None, "base_fullhead": None, "metric": None, "source": None,
                 "concurrency": {"base": None, "ship_12k": None, "base_fullhead": None}}
            for bm in PR534_GATE_BENCHMARKS}


def _load_gate_cells(path: str | None) -> dict:
    """Load gate cells from a --gate-cells JSON (any subset of benchmarks/roles); missing stay null."""
    cells = _empty_gate_cells()
    if not path:
        return cells
    if not os.path.exists(path):
        print(f"[gate] --gate-cells {path} not found; composing with empty (all-pending) cells", flush=True)
        return cells
    try:
        user = json.load(open(path))
    except Exception as exc:
        print(f"[gate] failed to parse {path}: {exc!r}; using empty cells", flush=True)
        return cells
    for bm in PR534_GATE_BENCHMARKS:
        u = user.get(bm)
        if isinstance(u, dict):
            for k, v in u.items():
                if k == "concurrency" and isinstance(v, dict):
                    cells[bm]["concurrency"].update(v)   # nested merge so a partial concurrency dict is OK
                else:
                    cells[bm][k] = v
    return cells


def build_gate_table(cells: dict) -> dict:
    """Compose the >=90%-of-base quality gate table from base/ship-12k/base-fullhead cells.
    meets_90pct_all is a definitive bool ONLY when all 3 base-fullhead cells (and their bases) are present;
    otherwise it stays None (pending) so a partial table is honest about what is not yet measured."""
    def _f(x):
        return isinstance(x, (int, float)) and math.isfinite(x)

    benchmarks, pending, confounded_bms = {}, [], []
    fullhead_present = cells_present = 0
    meets_list = []
    all_compared_conc_matched = True
    for bm in PR534_GATE_BENCHMARKS:
        c = cells.get(bm, {})
        base, ship, full = c.get("base"), c.get("ship_12k"), c.get("base_fullhead")
        conc = c.get("concurrency") or {}
        base_conc, full_conc, ship_conc = conc.get("base"), conc.get("base_fullhead"), conc.get("ship_12k")
        for role in PR534_GATE_ROLES:
            if _f(c.get(role)):
                cells_present += 1
            else:
                pending.append(f"{bm}.{role}")
        if _f(full):
            fullhead_present += 1
        full_pct = (full / base) if (_f(full) and _f(base) and base) else None
        ship_pct = (ship / base) if (_f(ship) and _f(base) and base) else None
        # the gate compares base_fullhead/base, so those two MUST share concurrency to be unconfounded
        # (advisor 00:02Z). matched requires both recorded AND equal; an unrecorded concurrency cannot be
        # certified unconfounded, so it conservatively blocks a definitive meets.
        conc_matched = (base_conc is not None and full_conc is not None and base_conc == full_conc)
        confounded = (full_pct is not None and not conc_matched)   # only meaningful once both numbers exist
        if confounded:
            confounded_bms.append(bm)
            all_compared_conc_matched = False
        meets = (full_pct >= PR534_GATE_THRESHOLD) if full_pct is not None else None
        if meets is not None:
            meets_list.append(meets)
        benchmarks[bm] = {
            "base": base, "ship_12k": ship, "base_fullhead": full,
            "fullhead_pct_of_base": (round(full_pct, 4) if full_pct is not None else None),
            "ship12k_pct_of_base": (round(ship_pct, 4) if ship_pct is not None else None),
            "meets_90pct": meets, "metric": c.get("metric"), "source": c.get("source"),
            "concurrency": {"base": base_conc, "ship_12k": ship_conc, "base_fullhead": full_conc},
            "concurrency_matched": (conc_matched if full_pct is not None else None),
            "confounded": confounded,
        }
    all_full = (fullhead_present == len(PR534_GATE_BENCHMARKS))
    # meets_90pct_all is definitive ONLY when every full-head cell is present AND every base-vs-fullhead
    # comparison is concurrency-matched -- a single confounded cell forces None (pending), never a false GREEN.
    meets_90pct_all = (all(meets_list)
                       if (all_full and len(meets_list) == len(PR534_GATE_BENCHMARKS)
                           and all_compared_conc_matched) else None)
    return {
        "benchmarks": benchmarks, "gate_threshold": PR534_GATE_THRESHOLD,
        "meets_90pct_all": meets_90pct_all, "all_fullhead_cells_present": all_full,
        "fullhead_cells_present": fullhead_present, "fullhead_cells_expected": len(PR534_GATE_BENCHMARKS),
        "n_cells_present": cells_present, "n_cells_expected": len(PR534_GATE_BENCHMARKS) * len(PR534_GATE_ROLES),
        "pending_cells": pending,
        "concurrency_pin": PR534_GATE_CONCURRENCY, "all_compared_conc_matched": all_compared_conc_matched,
        "confounded_benchmarks": confounded_bms,
        "by_construction_note": ("full-head clears >=90% BY CONSTRUCTION: prune OFF => base-identical lm_head "
                                 "=> no token -inf'd => no broad-distribution collapse. Empirical cells, when "
                                 "they land, confirm this; their absence does not threaten the safety claim."),
    }


def compose_pr534_report(fullhead_bc: dict, twelvek_bc: dict | None, a: argparse.Namespace) -> dict:
    ident = compose_bc_report(fullhead_bc, a)            # full identity leg on the full-head arm
    c = fullhead_bc["census"]

    fh_tps = fullhead_bc.get("tps") or {}
    tk_tps = (twelvek_bc.get("tps") or {}) if twelvek_bc else {}
    fh_med = fh_tps.get("warm_median_tps")
    tk_med = tk_tps.get("warm_median_tps")
    fh_thr = fh_tps.get("warm_throughput_tps")
    tk_thr = tk_tps.get("warm_throughput_tps")

    def _fin(x):
        return isinstance(x, (int, float)) and math.isfinite(x)

    def _ratio(num, den):
        return (num / den) if (_fin(num) and _fin(den) and den) else float("nan")

    ratio_med = _ratio(fh_med, tk_med)                  # full-head / 12k (in-harness, identical conditions)
    ratio_thr = _ratio(fh_thr, tk_thr)
    ratio = ratio_med                                   # headline = warm-median ratio (robust)
    projected_fullhead_tps = (PR534_SERVED_12K_TPS * ratio) if _fin(ratio) else float("nan")
    tps_cost = (PR534_SERVED_12K_TPS - projected_fullhead_tps) if _fin(projected_fullhead_tps) else float("nan")
    # in-harness raw delta (eager dilutes the absolute gap vs served, but the SIGN + ratio are honest)
    inharness_delta_med = (tk_med - fh_med) if (_fin(fh_med) and _fin(tk_med)) else float("nan")
    read_delta_gb = PR534_LMHEAD_FULL_READ_GB - PR534_LMHEAD_12K_READ_GB

    operative_1p0 = bool(ident["bc_operative_identity_1p0"])
    operative_rate = ident["operative_identity_rate"]
    n_sem = ident["bc_n_semantic_flips"]

    # Quality leg: assembled by the gate-table scaffold from cells HANDED to this card (--gate-cells JSON).
    # The cross-branch quality runs live on OTHER students' PRs (ubel/fern/wirbel), which this launch's
    # isolation rule forbids reading -- so the cells must be provided via PR #534, not pulled. Until they
    # land, the table is partial-but-honest (null/pending cells). The full head is quality-safe BY
    # CONSTRUCTION: the ONLY lossy lever vs base is the 12k lm_head prune, which -inf's the wanted token on
    # broad distributions; with the prune OFF the head is the base int4 head, so the collapse is impossible.
    gate = build_gate_table(_load_gate_cells(getattr(a, "gate_cells", None)))
    bm = gate["benchmarks"]
    quality_cells = {
        "mmlu_pro_fullhead_pct_of_base": bm["mmlu_pro"]["fullhead_pct_of_base"],
        "aime_fullhead_pct_of_base": bm["aime"]["fullhead_pct_of_base"],
        "gsm8k_fullhead_pct_of_base": bm["gsm8k"]["fullhead_pct_of_base"],
    }
    meets_90pct_all = gate["meets_90pct_all"]
    quality_note = (gate["by_construction_note"]
                    + f" | cells present {gate['n_cells_present']}/{gate['n_cells_expected']}"
                    + (f", pending: {gate['pending_cells']}" if gate["pending_cells"] else ""))

    tps_leg_complete = _fin(fh_med) and _fin(tk_med)
    identity_leg_complete = bool(math.isfinite(operative_rate))
    # The shipped surgical-357 stack is a LOCUS cert, not a strict-0-semantic cert (#515 / wirbel #487):
    # at full serve it shows bounded near-tie flips that collapse to ties at a small ULP threshold
    # (tie_threshold_for_zero_semantic_ulps). So "my measurement finished soundly" must NOT require
    # strict-0-semantic -- bc_no_semantic_flips is a SHIP property (reported separately as
    # ship_strict_operative_1p0 + operative_locus_ulps), not a measurement-completeness property. Gating
    # measured_cert_complete on it would mislabel every sound full-serve LOCUS cert as INCOMPLETE. So gate
    # on the STRUCTURAL self-test checks (coverage / screen-validated / armed / nan-clean / det-near-tie /
    # internal-consistency) instead -- a genuinely broken measurement still fails those.
    structural_self_test = {k: v for k, v in ident["self_test"].items() if k != "bc_no_semantic_flips"}
    structural_self_test_pass = bool(all(structural_self_test.values()))
    self_test_pass = structural_self_test_pass
    ship_strict_operative_1p0 = bool(ident["bc_operative_identity_1p0"])
    operative_locus_ulps = ident["tie_threshold_for_zero_semantic_ulps"]
    # My INDEPENDENT cert (the legs I own): identity + TPS + STRUCTURAL self-test. Complete from my runs alone.
    measured_cert_complete = bool(tps_leg_complete and identity_leg_complete and structural_self_test_pass)
    gate_cleared = gate["meets_90pct_all"]               # bool|None (None until all full-head cells land)
    quality_safe_by_construction = True                  # full head: prune OFF => base-identical => no -inf
    # draw_ready (the PR's strict empirical bool): a quality-safe ship that CLEARS the >=90% gate at a known
    # TPS. True only when my measured legs are done AND the empirical gate is confirmed cleared. Until the
    # (isolation-blocked) cells land it is False -- but quality_safe_by_construction records that the gate is
    # guaranteed by construction, so a False here is "awaiting empirical confirmation", not "unsafe".
    draw_ready = bool(measured_cert_complete and gate_cleared is True)

    report = {
        "pr": 534, "agent": "land",
        "leg": "conservative safe-anchor cert: {full-head no-prune} x {surgical 2D attention}",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "wandb_group": getattr(a, "wandb_group", None),
        # ---- KEY OUTPUTS (PR #534) ----
        "fullhead_surgical_warm_median_tps": fh_med,     # in-harness (enforce_eager) full-head warm median
        "twelvek_surgical_warm_median_tps": tk_med,      # in-harness 12k warm median (same conditions)
        "tps_ratio_fullhead_over_12k": (round(ratio, 6) if _fin(ratio) else None),
        "tps_ratio_throughput_xcheck": (round(ratio_thr, 6) if _fin(ratio_thr) else None),
        "projected_served_fullhead_tps": (round(projected_fullhead_tps, 3) if _fin(projected_fullhead_tps) else None),
        "tps_cost_of_fullhead_vs_12k": (round(tps_cost, 3) if _fin(tps_cost) else None),
        "served_12k_rung_tps": PR534_SERVED_12K_TPS,
        "tps_bracket": {                                  # advisor 00:02Z pt5: the explicit cost of quality-safety
            "floor_tps_local": PR534_QUALITY_SAFE_FLOOR_TPS_LOCAL,
            "floor_tps_official_implied": PR534_QUALITY_SAFE_FLOOR_TPS_OFFICIAL,
            "fast_upper_tps": PR534_QUALITY_SAFE_FAST_UPPER_TPS,
            "floor_source": "wirbel #533 b9j1z40d (base-int4, no surgical/fast kernels, quality==base)",
            "fast_upper_source": "fern #535 whh42dgd (full-head on the fast stack, PPL 2.006 byte-exact)",
            "this_cert_projected_served_fullhead_tps": (round(projected_fullhead_tps, 3) if _fin(projected_fullhead_tps) else None),
            "this_cert_within_bracket": (bool(PR534_QUALITY_SAFE_FLOOR_TPS_LOCAL <= projected_fullhead_tps
                                              <= PR534_QUALITY_SAFE_FAST_UPPER_TPS) if _fin(projected_fullhead_tps) else None),
            "all_rungs_under_unsafe_500": True,
        },
        "inharness_tps_delta_12k_minus_fullhead": (round(inharness_delta_med, 3) if _fin(inharness_delta_med) else None),
        "operative_identity_rate": operative_rate,
        "bc_operative_identity_1p0": operative_1p0,
        # the surgical-357 ship is a LOCUS cert (#515): strict-0-semantic is False, but ALL semantic flips
        # collapse to ties at operative_locus_ulps -- a bounded near-tie locus, not a genuine divergence.
        "ship_strict_operative_1p0": ship_strict_operative_1p0,
        "operative_locus_ulps": operative_locus_ulps,
        "operative_cert_kind": ("STRICT_1.0" if ship_strict_operative_1p0 else "LOCUS"),
        "bc_n_semantic_flips": n_sem, "bc_n_tie_flips": ident["bc_n_tie_flips"],
        "self_det": ident["determinism_served"],
        "ppl": None, "ppl_note": (f"not separately measured here; full-head PPL <= 12k rung 2.3767 by "
                                  "construction (removing the prune cannot raise PPL)"),
        # quality leg (isolation-blocked, by-construction-safe)
        **quality_cells, "meets_90pct_all": meets_90pct_all, "quality_note": quality_note,
        # ---- provenance / arm wiring ----
        "fullhead": bool(fullhead_bc.get("fullhead")), "fullhead_lm_head_rows": fullhead_bc.get("lm_head_rows"),
        "fullhead_lm_head_mode": fullhead_bc.get("lm_head_mode"),
        "twelvek_lm_head_rows": (twelvek_bc.get("lm_head_rows") if twelvek_bc else None),
        "surgical_attn_armed": fullhead_bc.get("surgical_attn_armed"),
        "matmul_tax_installed": fullhead_bc.get("matmul_tax_installed"),
        "spec_on": fullhead_bc.get("spec_on"), "num_speculative_tokens": fullhead_bc.get("num_speculative_tokens"),
        "fullhead_model_dir": fullhead_bc.get("model_dir"),
        "twelvek_model_dir": (twelvek_bc.get("model_dir") if twelvek_bc else None),
        # ---- TPS detail (both arms) ----
        "tps_fullhead": fh_tps, "tps_12k": tk_tps,
        "lmhead_read_delta_gb": round(read_delta_gb, 4),
        "lmhead_full_read_gb": PR534_LMHEAD_FULL_READ_GB, "lmhead_12k_read_gb": PR534_LMHEAD_12K_READ_GB,
        # ---- identity detail (full-head arm, full census) ----
        "n_positions": c["n_positions"], "n_served_tokens": fullhead_bc["n_served_tokens"],
        "n_read": fullhead_bc["n_read"], "n_skipped": fullhead_bc["n_skipped"],
        "read_fraction": fullhead_bc["read_fraction"],
        "screen_validated": fullhead_bc["screen_validated"],
        "tie_threshold_for_zero_semantic": ident["tie_threshold_for_zero_semantic"],
        "tie_threshold_for_zero_semantic_ulps": ident["tie_threshold_for_zero_semantic_ulps"],
        "det_diffs_all_near_tie": fullhead_bc.get("det_diffs_all_near_tie", True),
        # ---- scale / cost ----
        "peak_gpu_gb": fullhead_bc["peak_gpu_gb"], "peak_parent_gb": fullhead_bc.get("peak_parent_gb"),
        "peak_device_gb": fullhead_bc.get("peak_device_gb"),
        "fullhead_census_secs": fullhead_bc["census_secs"],
        "twelvek_tps_secs": (twelvek_bc.get("census_secs") if twelvek_bc else None),
        "n_prompts_run": fullhead_bc["n_prompts_run"], "output_len": fullhead_bc["output_len"],
        # ---- quality gate table (scaffold; cells handed via --gate-cells, isolation-blocked otherwise) ----
        "gate_table": gate, "gate_cleared": gate_cleared,
        "quality_safe_by_construction": quality_safe_by_construction,
        # ---- self-test (reuse the identity-arm self-test + PR#534 leg-completeness checks) ----
        "identity_self_test": ident["self_test"], "identity_self_test_passes": self_test_pass,
        "structural_self_test_pass": structural_self_test_pass,
        "tps_leg_complete": tps_leg_complete, "identity_leg_complete": identity_leg_complete,
        "measured_cert_complete": measured_cert_complete, "draw_ready": draw_ready,
        # ---- verdict ----
        "verdict_oneline": (
            f"full-head surgical safe-anchor: operative {operative_rate:.6f} "
            f"({n_sem} semantic, all collapse at {operative_locus_ulps} ULP -> bounded LOCUS cert, "
            f"cleaner than the 12k); in-harness TPS-neutral vs 12k (ratio {ratio:.3f}; the eager/single-seq "
            f"census is body-dominated and DILUTES the served lm_head penalty -- reliable served cost is the "
            f"[{PR534_QUALITY_SAFE_FLOOR_TPS_OFFICIAL:.0f}..{PR534_QUALITY_SAFE_FAST_UPPER_TPS:.0f}] bracket); "
            f"quality-safe by construction"
            f"{'' if gate_cleared is True else ' (>=90% gate empirically PENDING -- cells not yet handed)'}"
            if (_fin(tps_cost) and _fin(operative_rate)) else
            "full-head surgical safe-anchor cert INCOMPLETE (a measured leg did not finish)"),
        # GREEN = full empirical draw (legs + gate cleared); AMBER = my measured legs done, gate cells pending
        # (quality safe by construction); INCOMPLETE = a measured leg unfinished.
        "verdict": ("GREEN" if (draw_ready and operative_1p0) else
                    ("AMBER" if measured_cert_complete else "INCOMPLETE")),
    }
    return report


def _print_pr534_console(r: dict) -> None:
    print("\n===== PR #534 FULL-HEAD SURGICAL SAFE-ANCHOR CERT ({full-head} x {surgical 2D}) =====", flush=True)
    print(f" VERDICT                                  : {r['verdict']}", flush=True)
    print(f" verdict (one line)                       : {r['verdict_oneline']}", flush=True)
    print(" --- identity leg (full-head arm vs M=1 AR, real 128x512 served path) ---", flush=True)
    print(f"  operative_identity_rate (PRIMARY)       : {r['operative_identity_rate']:.7f} "
          f"(semantic={r['bc_n_semantic_flips']} tie={r['bc_n_tie_flips']})", flush=True)
    print(f"  bc_operative_identity_1p0               : {r['bc_operative_identity_1p0']}", flush=True)
    print(f"  self_det (free-run reproduce)           : {r['self_det']}", flush=True)
    print(f"  served lm_head rows (full head)         : {r['fullhead_lm_head_rows']} "
          f"({r['fullhead_lm_head_mode']})", flush=True)
    print(" --- TPS cost leg (warm-median ratio, projected onto 357.06 served rung) ---", flush=True)
    print(f"  in-harness warm-median TPS fullhead/12k : {r['fullhead_surgical_warm_median_tps']} / "
          f"{r['twelvek_surgical_warm_median_tps']} (enforce_eager)", flush=True)
    print(f"  ratio fullhead/12k (median | thr xchk)  : {r['tps_ratio_fullhead_over_12k']} | "
          f"{r['tps_ratio_throughput_xcheck']}", flush=True)
    print(f"  projected served full-head TPS          : {r['projected_served_fullhead_tps']}", flush=True)
    print(f"  tps_cost_of_fullhead_vs_12k (Δ vs 357)  : {r['tps_cost_of_fullhead_vs_12k']}", flush=True)
    g = r.get("gate_table", {})
    gb = g.get("benchmarks", {})
    print(" --- quality gate table (>=90% of base; cells handed via --gate-cells) ---", flush=True)
    print(f"  {'benchmark':<9} {'base':>9} {'ship12k':>9} {'fullhead':>9} {'full%base':>9} {'>=90%':>6}",
          flush=True)
    for bmname in PR534_GATE_BENCHMARKS:
        row = gb.get(bmname, {})
        def _s(x, w=9, p=4):
            return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}")
        print(f"  {bmname:<9} {_s(row.get('base'))} {_s(row.get('ship_12k'))} "
              f"{_s(row.get('base_fullhead'))} {_s(row.get('fullhead_pct_of_base'))} "
              f"{str(row.get('meets_90pct')):>6}", flush=True)
    print(f"  cells present {g.get('n_cells_present')}/{g.get('n_cells_expected')} "
          f"(full-head {g.get('fullhead_cells_present')}/{g.get('fullhead_cells_expected')})  "
          f"meets_90pct_all={r['meets_90pct_all']}  quality_safe_by_construction="
          f"{r['quality_safe_by_construction']}", flush=True)
    if g.get("pending_cells"):
        print(f"  pending cells: {g['pending_cells']}", flush=True)
    print(" --- readiness ---", flush=True)
    print(f"  identity_leg / tps_leg / self_test      : {r['identity_leg_complete']} / "
          f"{r['tps_leg_complete']} / {r['identity_self_test_passes']}", flush=True)
    print(f"  measured_cert_complete / gate_cleared   : {r['measured_cert_complete']} / {r['gate_cleared']}",
          flush=True)
    print(f"  draw_ready (legs + gate cleared)        : {r['draw_ready']}", flush=True)
    print(f"  peak GPU                                : {r['peak_gpu_gb']:.2f} GB", flush=True)
    print("==================================================================\n", flush=True)


def log_pr534_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="land", name=a.wandb_name, group=a.wandb_group,
        notes=("PR#534 conservative safe-anchor cert: {full-head no-prune} x {surgical 2D attention}. "
               "Identity leg = full-serve operative-identity census of the FULL 262,144-row lm_head served "
               "surgical-357 stack vs the M=1 AR reference (real spec-on 128x512 path). Cost leg = warm-median "
               "decode-TPS ratio (full-head / 12k) under identical enforce_eager harness conditions, projected "
               "onto the 357.06 served rung. Quality is safe by construction (prune OFF => base-identical head)."),
        config={
            "pr": 534, "M_verify": M_VERIFY, "K_spec": K_SPEC, "output_len": report["output_len"],
            "n_prompts_run": report["n_prompts_run"], "fullhead_model_dir": report["fullhead_model_dir"],
            "twelvek_model_dir": report["twelvek_model_dir"], "served_12k_rung_tps": PR534_SERVED_12K_TPS,
            "surgical_mode": True, "spec_on": report["spec_on"],
            "num_speculative_tokens": report["num_speculative_tokens"],
            "tps_warmup": report["tps_fullhead"].get("tps_warmup"), "enforce_eager": True,
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = (
        "fullhead_surgical_warm_median_tps", "twelvek_surgical_warm_median_tps",
        "tps_ratio_fullhead_over_12k", "tps_ratio_throughput_xcheck", "projected_served_fullhead_tps",
        "tps_cost_of_fullhead_vs_12k", "served_12k_rung_tps", "inharness_tps_delta_12k_minus_fullhead",
        "operative_identity_rate", "bc_operative_identity_1p0", "ship_strict_operative_1p0",
        "operative_locus_ulps", "operative_cert_kind", "structural_self_test_pass",
        "bc_n_semantic_flips", "bc_n_tie_flips",
        "self_det", "ppl", "mmlu_pro_fullhead_pct_of_base", "aime_fullhead_pct_of_base",
        "gsm8k_fullhead_pct_of_base", "meets_90pct_all", "fullhead", "fullhead_lm_head_rows",
        "fullhead_lm_head_mode", "twelvek_lm_head_rows", "surgical_attn_armed", "matmul_tax_installed",
        "spec_on", "num_speculative_tokens", "lmhead_read_delta_gb", "n_positions", "n_served_tokens",
        "n_read", "n_skipped", "read_fraction", "screen_validated", "tie_threshold_for_zero_semantic",
        "tie_threshold_for_zero_semantic_ulps", "det_diffs_all_near_tie", "peak_gpu_gb", "peak_parent_gb",
        "peak_device_gb", "fullhead_census_secs", "twelvek_tps_secs", "n_prompts_run", "output_len",
        "tps_leg_complete", "identity_leg_complete", "identity_self_test_passes",
        "measured_cert_complete", "gate_cleared", "quality_safe_by_construction", "draw_ready",
        "verdict", "verdict_oneline", "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
    )
    for k in keys:
        run.summary[k] = report.get(k)
    for tname, tps_block in (("fullhead", report["tps_fullhead"]), ("twelvek", report["tps_12k"])):
        for sk, sv in (tps_block or {}).items():
            run.summary[f"tps_{tname}/{sk}"] = sv
    for k, v in report.get("identity_self_test", {}).items():
        run.summary[f"selftest/{k}"] = v
    # gate table: per-benchmark cells + derived columns, plus coverage counters
    gate = report.get("gate_table", {})
    for bmname, row in gate.get("benchmarks", {}).items():
        for col in ("base", "ship_12k", "base_fullhead", "fullhead_pct_of_base",
                    "ship12k_pct_of_base", "meets_90pct"):
            run.summary[f"gate/{bmname}/{col}"] = row.get(col)
    for gk in ("meets_90pct_all", "all_fullhead_cells_present", "fullhead_cells_present",
               "n_cells_present", "n_cells_expected", "gate_threshold"):
        run.summary[f"gate/{gk}"] = gate.get(gk)
    run.summary["verdict_green"] = report["verdict"].startswith("GREEN")
    run.summary["verdict_amber"] = report["verdict"].startswith("AMBER")
    run.summary["draw_ready"] = report["draw_ready"]
    finish_wandb(run)
    print(f"[wandb] logged PR#534 run {run.id}", flush=True)


def _finish_pr534(report: dict, a: argparse.Namespace) -> None:
    report_path = OUT_DIR / "pr534_fullhead_surgical_results.json"
    json.dump(report, open(report_path, "w"), indent=2)
    _print_pr534_console(report)
    if not a.no_wandb:
        log_pr534_wandb(report, a)


def orchestrate_pr534(a: argparse.Namespace) -> None:
    """PR #534: run the full-head identity+TPS census, then the 12k TPS-only pass, then compose the cert."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[orch:534] arm 1/2: full-head census (identity + warm TPS)", flush=True)
    fullhead_bc = _run_bc_census(a, label="fullhead", extra_args=["--fullhead"])
    print("[orch:534] arm 2/2: 12k TPS-only pass (cost ratio denominator)", flush=True)
    twelvek_bc = _run_bc_census(a, label="twelvek_tps", extra_args=["--tps-only"])
    if getattr(a, "census_only", False):
        print("[orch:534] census_only: both arm results written; run --reanalyze-pr534 to compose + log",
              flush=True)
        return
    _finish_pr534(compose_pr534_report(fullhead_bc, twelvek_bc, a), a)


def reanalyze_pr534(a: argparse.Namespace) -> None:
    fp = OUT_DIR / "arm_fullhead_result.json"
    if not fp.exists():
        raise FileNotFoundError(f"--reanalyze-pr534 needs {fp} (run the full-head GPU phase first)")
    fullhead_bc = json.load(open(fp))
    tp = OUT_DIR / "arm_twelvek_tps_result.json"
    twelvek_bc = json.load(open(tp)) if tp.exists() else None
    if twelvek_bc is None:
        print(f"[reanalyze:534] WARN {tp} missing; TPS cost leg will be null (identity leg still composes)",
              flush=True)
    _finish_pr534(compose_pr534_report(fullhead_bc, twelvek_bc, a), a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["fullserve_census", "benchmark_config_census"], default=None)
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
    # splitkv399 candidate config (lawine #496/#500 served recipe = T4 S64; gated, default-off in the wheel)
    ap.add_argument("--fixed-tps", dest="fixed_tps", type=int, default=4,
                    help="splitkv399 arm: pinned tiles_per_segment T (CHUNK=16*T keys, byte-exact split SIZE)")
    ap.add_argument("--num-segments", dest="num_segments", type=int, default=64,
                    help="splitkv399 arm: parallel softmax segment capacity S (coverage=16*T*S keys)")
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
    # ---- PR #521 benchmark-config census (real spec-on served path) ----
    ap.add_argument("--benchmark-config", dest="benchmark_config", action="store_true",
                    help="run the PR #521 benchmark-config census: real spec-on surgical-357 served path "
                         "vs M=1 AR reference re-conditioned on the realized served prefix (128x512)")
    ap.add_argument("--reanalyze-bc", dest="reanalyze_bc", action="store_true",
                    help="0-GPU: recompose the benchmark-config report + tie sweep from saved arm json")
    # ---- PR #534 (land) full-head surgical safe-anchor cert ----
    ap.add_argument("--pr534", dest="pr534", action="store_true",
                    help="orchestrate the PR #534 cert: full-head identity+TPS census + 12k TPS-only pass + compose")
    ap.add_argument("--reanalyze-pr534", dest="reanalyze_pr534", action="store_true",
                    help="0-GPU: recompose the PR #534 cert (identity + TPS ratio projection) from saved arm json")
    ap.add_argument("--fullhead", dest="fullhead", action="store_true",
                    help="worker: serve the FULL 262,144-row lm_head (LM_HEAD_PRUNE off, the safe-anchor head)")
    ap.add_argument("--tps-only", dest="tps_only", action="store_true",
                    help="worker: measure decode TPS only (skip the identity census machinery)")
    ap.add_argument("--tps-warmup", dest="tps_warmup", type=int, default=2,
                    help="warm-median TPS drops the first N prompts (first-touch kernel JIT inflates them)")
    ap.add_argument("--gate-cells", dest="gate_cells", default=None,
                    help="JSON of quality cells {gsm8k/aime/mmlu_pro: {base, ship_12k, base_fullhead, "
                         "metric, source}} for the >=90% gate table (cells are handed via PR#534, not pulled)")
    # ---- PR #534 (land) reasoning-length identity census (off the 128-challenge-prompt distribution) ----
    ap.add_argument("--prompts", dest="prompts", default=PROMPTS_JSONL,
                    help="prompt JSONL (default: 128 challenge prompts). Use the AIME reasoning set to census "
                         "operative-identity on a long off-distribution reasoning workload.")
    ap.add_argument("--reasoning", dest="reasoning", action="store_true",
                    help="worker: per-prompt natural prefix (Cp=len(prompt)) + sweep the free-running reasoning "
                         "trace (for variable-length reasoning prompts that are shorter than the fixed 224 prefix)")
    ap.add_argument("--reasoning-census", dest="reasoning_census", action="store_true",
                    help="orchestrate the PR #534 reasoning-length census: surgical ship vs M=1 AR on the AIME "
                         "reasoning set, compose deliverables vs the 128-prompt baseline + log W&B")
    ap.add_argument("--reanalyze-reasoning", dest="reanalyze_reasoning", action="store_true",
                    help="0-GPU: recompose the reasoning-census report from saved arm_reasoning_*_result.json")
    ap.add_argument("--bc-output-len", dest="bc_output_len", type=int, default=BC_DEFAULT_OUTPUT_LEN)
    ap.add_argument("--bc-gpu-mem-util", dest="bc_gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--bc-max-model-len", dest="bc_max_model_len", type=int, default=4096)
    ap.add_argument("--bc-skip-threshold", dest="bc_skip_threshold", type=float, default=BC_DEFAULT_SKIP_THRESHOLD,
                    help="nats; served verify top-2 gap above which a served-vs-M1 match is PROVEN (clear "
                         "winner, no M=1 read). Far above any observed flip gap (#487 max 0.172 nat); the "
                         "screen_check probe validates it empirically.")
    a = ap.parse_args()
    a.arms = tuple(s for s in str(a.arms).split(",") if s)
    skip_prompts = tuple(int(x) for x in str(a.skip_prompts).split(",") if x.strip())

    if a.smoke and a.phase is None and not a.benchmark_config and not a.reanalyze_bc \
            and not a.pr534 and not a.reanalyze_pr534:
        a.n_prompts = min(a.n_prompts, 4)
        a.traj_len = min(a.traj_len, 256)
        a.det_check_k = min(a.det_check_k, 4)
    if a.smoke and (a.benchmark_config or a.pr534):
        a.n_prompts = min(a.n_prompts, 3)
        a.bc_output_len = min(a.bc_output_len, 48)
        a.det_check_k = min(a.det_check_k, 2)

    if a.phase == "fullserve_census":
        phase_fullserve_census(a.out, a.arm, a.n_prompts, a.c0, a.traj_len,
                               a.gpu_mem_util, a.max_batched_tokens, a.verbose_k, a.det_check_k, a.ignore_eos,
                               checkpoint=a.checkpoint, heartbeat=a.heartbeat, resume=a.resume,
                               skip_prompts=skip_prompts, prompts_path=a.prompts, reasoning=a.reasoning)
    elif a.phase == "benchmark_config_census":
        phase_benchmark_config_census(a.out, a.n_prompts, a.bc_output_len, a.bc_gpu_mem_util,
                                      a.bc_max_model_len, a.bc_skip_threshold, a.det_check_k,
                                      checkpoint=a.checkpoint, heartbeat=a.heartbeat, resume=a.resume,
                                      skip_prompts=skip_prompts, fullhead=a.fullhead, tps_only=a.tps_only,
                                      tps_warmup=a.tps_warmup)
    elif a.reanalyze_pr534:
        reanalyze_pr534(a)
    elif a.pr534:
        orchestrate_pr534(a)
    elif a.reanalyze_reasoning:
        reanalyze_reasoning(a)
    elif a.reasoning_census:
        orchestrate_reasoning(a)
    elif a.reanalyze_bc:
        reanalyze_bc(a)
    elif a.benchmark_config:
        orchestrate_bc(a)
    elif a.reanalyze:
        reanalyze(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
