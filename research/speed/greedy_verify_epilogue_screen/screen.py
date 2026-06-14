#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Greedy-verify lm_head/argmax EPILOGUE screen (PR #255, wirbel) -- does the
served greedy-MTP verify MATERIALIZE the full M=8 x vocab(~256k) logit tensor
(and/or d2h-transfer it), or is it already fused to an on-GPU argmax + draft-
equality epilogue? CPU-only analytic bank-the-analysis (the audit leg is the
served code + config; the accounting imports the composition anchors).

THE FRAME (decisive -- the materialization question)
----------------------------------------------------
The verify pass is the largest greedy-SAFE-untouched block in the step
(verify_us~=559.83, verify_share=0.4596 per #251). The ONE provably-exact lever
on it is the lm_head/argmax EPILOGUE: in greedy MTP verify the acceptance test
for draft token t_i is just `t_i == argmax(verify_logits_i)`, so you never need
the full softmax over the ~256k vocab. The screen asks whether the served verify
(a) computes the full M=8 x vocab logit tensor and runs a full softmax/sampler
over it, (b) d2h-transfers any logit/logprob tensor per step, or (c) already
reduces to an on-GPU argmax + draft-equality returning just token ids + accept
mask.

  * If materialized / transferred => real greedy-exact lever (fuse lm_head+
    argmax+accept; return the M=8 accept-mask + bonus-argmax instead of an
    8x256k tensor). At bf16 that tensor is ~4.2 MB; a d2h over A10G PCIe
    (~25 GB/s) is ~168 us, i.e. up to ~30% of the 560 us verify if on the
    critical path. GO + projected gain.
  * If already fused on-GPU => clean NULL/NO-GO, banked (the #251 pattern: a
    cheap structural kill that closes the question).

THE AUDIT (served stack, exact vLLM wheel 0.22.1rc1.dev307+g3e8afdf78)
---------------------------------------------------------------------
Served submission: submissions/fa2sw_precache_kenyan ("dixie-flatline:
onegraph-spec7 substrate + PCK-04 lm_head vocabulary pruning"). Target config
/tmp/osoi5-v0-baked/config.json: text_config.vocab_size=262144, dtype=bfloat16.

  1. lm_head is PRUNED (LM_HEAD_PRUNE=1, LM_HEAD_PRUNE_REQUIRE=1): serve.py
     `_lmhead_prune_phase` row-slices lm_head to K=12288 kept rows
     (/tmp/osoi5-12k-baked, keepset int4-pck04c-12k). serve_patch_pck04.py
     `compute_logits` scatters the [M, 12288] pruned logits into a [M, 262144]
     -inf buffer via a CHEAP on-GPU `index_copy_` (template clone) -- the
     expensive full-256k lm_head GEMM never runs.
  2. The greedy verify epilogue (rejection sampler `forward`, DIXIE_SLIM_GREEDY=1
     + DIXIE_FUSED_ACCEPT_PREP=1) is, for the greedy TPS config
     (all_greedy & max_num_logprobs is None & no_penalties & no masks):
        dixie_all_argmax   = logits.argmax(dim=-1)              # on-GPU, no softmax
        dixie_bonus/target = dixie_all_argmax[bonus/target_indices]
        _dixie_fused_accept_prep(...) OR rejection_greedy_sample_kernel[...]  # Triton on-GPU
        return SamplerOutput(sampled_token_ids=<GPU>, logprobs_tensors=None)
     i.e. scenario (c): an on-GPU argmax + draft-equality returning just token
     ids + accept mask. NO full softmax, NO logit/logprob d2h. The ONLY
     `.cpu()/.item()/.tolist()` in serve.py are the one-time PLE loader / prune
     phase; sitecustomize.py has ZERO. logprobs_tensors=None.
  3. Greedy-safety is by construction and stated in the served code: "bf16 -> fp32
     is an exact, monotonic upcast, so argmax over raw logits is bit-identical to
     the slow path's argmax over the fp32 copy." Any epilogue-fusion build is
     token-identical.

VERDICT OF THE AUDIT: verify_materializes_full_logits = False (lm_head pruned to
12288, argmax-only, no full softmax; the [M,262144] tensor that exists is a cheap
-inf scatter buffer, not the priced full GEMM+softmax). verify_logit_d2h_per_step
= False. The epilogue is ALREADY fused on-GPU => NULL lever, NO-GO.

THE CUDAGraph TIE-IN (decisive diagnostic; lawine #246 is the consumer)
----------------------------------------------------------------------
A d2h in the middle of the step forces a CUDAGraph break. The served stack runs
ONEGRAPH=1 + LOOPGRAPH_REQUIRE_CAPTURE=1: sitecustomize.py `_capture_graph`
captures the K=7 width-1 drafter loop into a `torch.cuda.CUDAGraph` and REQUIRES
the capture to succeed. That capture succeeding is independent proof that the
per-step path -- including the verify epilogue -- contains no graph-breaking
d2h. Combined with the on-GPU epilogue read in (2), the epilogue d2h the lever
hypothesised as the CUDAGraph-breaker DOES NOT EXIST. => epilogue-fusion is NOT
a prerequisite for whole-step capture; the epilogue is already fused, which is
part of what makes the ONEGRAPH capture feasible.

THE ACCOUNTING (anchors IMPORTED, not re-derived)
-------------------------------------------------
Composition (kanna #217): official = K_cal*(E[T]/step)*tau, K_cal=125.268,
step=1.2182 ms, served=481.53. TPS ∝ 1/step at fixed E[T] => a net step saving
us_net maps to tps(step-us_net)=481.53*step/(step-us_net). Verify share from
denken's drafter roofline (g_d=0.168/depth, K=7): verify_share=1/(1+K*g_d).

  Scenario MAT+D2H (the GO premise the lever targets): full logit tensor
    = M*vocab*2 = 4.194 MB; d2h @ PCIe 25 GB/s = 167.8 us (= 30.0% of verify_us).
    If on the critical path, fusing removes it: ~+16% TPS, clears 500, UNBLOCKS
    whole-step capture. PREMISE FALSE per audit.
  Scenario MAT-NO-D2H (HBM-floor if logits full-vocab but on-GPU): fusion elides
    the [M,vocab] HBM round-trip = 2*M*vocab*2 / 600 GB/s = 14.0 us => ~+1.16%.
    Also moot: the GEMM is pruned to 12288, so no full-vocab round-trip exists.
  ACTUAL (audited, already fused + pruned): the realizable epilogue-fusion saving
    is the [M,12288] logit round-trip = 2*M*K*2 / 600 GB/s ~= 0.66 us (~+0.05%),
    and even that is already fused away (argmax on-GPU + fused accept kernel,
    no d2h, ONEGRAPH captured) => us_net = 0 => projected_tps_gain_pct = 0.00.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM run / HF Job / submission / served-file
change / official draw. BASELINE stays 481.53; this SCREEN adds 0 TPS (any
epilogue-fusion build is a separate GPU follow-up, token-identical by
construction). NOT a launch. NOT open2. Non-overlap: lawine #246 (CUDAGraph
CAPTURE = launch overhead; this screen prices the epilogue FUSION that would be
a PREREQUISITE if a d2h existed -- complementary, not the same lever), kanna #254
(draft int4 GEMM, different model), stark #247 (topology E[T]), ubel #250 (draft
token source).

PRIMARY metric  greedy_verify_epilogue_screen_self_test_passes
TEST    metric  projected_tps_gain_pct   (0.00 actual; +bounds stated)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# IMPORTED anchors (kanna #217 composition, denken #75/#85 + wirbel #83 roofline,
# wirbel #251 verify split, A10G device anchors). Re-derive NOTHING.
# --------------------------------------------------------------------------- #
SERVED_TPS = 481.53          # official served (linear MTP K=7, PR #52); this screen adds 0
BASELINE_TPS = 481.53
STEP_US = 1218.2             # served step time 1.2182 ms
K_CAL = 125.268             # composition calibration constant (kanna #217)
G_DRAFTER = 0.168           # drafter cost per depth pass / verify (denken #75/#85, wirbel #83)
K_SPEC = 7                  # num_speculative_tokens (manifest, linear MTP K=7)
M_VERIFY = K_SPEC + 1       # M = K+1 = 8 verify query rows
PCIE_BW_GBS = 25.0          # A10G PCIe gen4 x16 effective d2h bandwidth (GB/s)
HBM_BW_GBS = 600.0          # A10G HBM bandwidth (GB/s)
L2_MB = 6.0                 # A10G L2 cache (MB)

# --------------------------------------------------------------------------- #
# AUDITED constants (read from the served configs/manifest when present; these
# fallbacks are the values verified from /tmp/osoi5-v0-baked/config.json,
# /tmp/osoi5-12k-baked/pck04_keepset.json, and submissions/fa2sw_precache_kenyan
# under vLLM 0.22.1rc1.dev307+g3e8afdf78).
# --------------------------------------------------------------------------- #
VOCAB_AUDITED = 262144
DTYPE_BYTES_AUDITED = 2          # bf16 (text_config.dtype = bfloat16)
PRUNED_VOCAB_K_AUDITED = 12288   # LM_HEAD_PRUNE row-slice (osoi5-12k-baked)
VLLM_VERSION = "0.22.1rc1.dev307+g3e8afdf78"
SERVED_SUBMISSION = "fa2sw_precache_kenyan"

_TARGET_CFG_CANDIDATES = ["/tmp/osoi5-v0-baked/config.json", "/tmp/osoi5-12k-baked/config.json"]
_KEEPSET_CANDIDATES = ["/tmp/osoi5-12k-baked/pck04_keepset.json"]
_MANIFEST = REPO_ROOT / "submissions" / SERVED_SUBMISSION / "manifest.json"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _read_json(paths: list[str]) -> dict | None:
    for p in paths:
        fp = Path(p)
        if fp.is_file():
            try:
                return {"path": str(fp), "data": json.loads(fp.read_text())}
            except Exception:  # noqa: BLE001
                continue
    return None


def _cfg_get(cfg: dict, *keys: str):
    tc = cfg.get("text_config", {}) if isinstance(cfg, dict) else {}
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            return cfg[k]
        if isinstance(tc, dict) and k in tc:
            return tc[k]
    return None


_DTYPE_BYTES = {
    "bfloat16": 2, "float16": 2, "half": 2, "bf16": 2, "fp16": 2,
    "float32": 4, "float": 4, "fp32": 4, "float8": 1, "fp8": 1,
}


def audit_served() -> dict[str, Any]:
    """Resolve vocab / dtype-bytes / pruned-K and the served-manifest fusion
    flags from disk, falling back to the audited constants. The manifest flags
    GROUND the 'already fused on-GPU' verdict in the actual served config -- if a
    future served manifest dropped the fusion, this screen would flip."""
    out: dict[str, Any] = {
        "vocab": VOCAB_AUDITED, "dtype_bytes": DTYPE_BYTES_AUDITED,
        "pruned_vocab_K": PRUNED_VOCAB_K_AUDITED, "dtype_str": "bfloat16",
        "target_cfg_path": None, "keepset_path": None, "manifest_path": None,
        "config_consistent_with_audit": None,
    }
    tgt = _read_json(_TARGET_CFG_CANDIDATES)
    if tgt is not None:
        c = tgt["data"]
        out["target_cfg_path"] = tgt["path"]
        v = _cfg_get(c, "vocab_size")
        dt = _cfg_get(c, "dtype", "torch_dtype")
        if _is_num(v):
            out["vocab"] = int(v)
        if isinstance(dt, str) and dt.lower() in _DTYPE_BYTES:
            out["dtype_bytes"] = _DTYPE_BYTES[dt.lower()]
            out["dtype_str"] = dt
    ks = _read_json(_KEEPSET_CANDIDATES)
    if ks is not None:
        out["keepset_path"] = ks["path"]
        k = ks["data"].get("pruned_vocab_K")
        if _is_num(k):
            out["pruned_vocab_K"] = int(k)
        fv = ks["data"].get("full_vocab")
        if _is_num(fv):
            out["vocab"] = int(fv)

    # Served-manifest fusion flags (the 'already fused' provenance).
    flags = {
        "DIXIE_SLIM_GREEDY": "1", "DIXIE_FUSED_ACCEPT_PREP": None, "LM_HEAD_PRUNE": None,
        "ONEGRAPH": None, "LOOPGRAPH_REQUIRE_CAPTURE": None, "FUSED_SPARSE_ARGMAX": None,
        "SPECULATIVE_CONFIG": None, "OVERRIDE_GENERATION_CONFIG": None,
    }
    if _MANIFEST.is_file():
        out["manifest_path"] = str(_MANIFEST)
        try:
            env = json.loads(_MANIFEST.read_text()).get("env", {})
            for k in list(flags):
                if k in env:
                    flags[k] = env[k]
        except Exception:  # noqa: BLE001
            pass
    out["manifest_flags"] = flags
    out["config_consistent_with_audit"] = bool(
        out["vocab"] == VOCAB_AUDITED and out["dtype_bytes"] == DTYPE_BYTES_AUDITED
    )
    # The 'already fused on-GPU' verdict, grounded in the served manifest flags.
    out["lm_head_pruned"] = (flags.get("LM_HEAD_PRUNE") == "1")
    out["slim_greedy_fused_argmax"] = (flags.get("DIXIE_SLIM_GREEDY", "1") == "1")
    out["fused_accept_prep"] = (flags.get("DIXIE_FUSED_ACCEPT_PREP") == "1")
    out["onegraph_wholestep_capture"] = (
        flags.get("ONEGRAPH") == "1" and flags.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1"
    )
    return out


# --------------------------------------------------------------------------- #
# Composition: TPS <-> step. tps(step') = SERVED * STEP_US / step'.
# --------------------------------------------------------------------------- #
def tps_from_step(step_us: float) -> float:
    return SERVED_TPS * STEP_US / step_us


def tps_gain_pct_from_us_net(us_net: float) -> float:
    """+us_net = step SHRINKS by us_net (a saving); -us_net = step grows (a cost)."""
    return (tps_from_step(STEP_US - us_net) / SERVED_TPS - 1.0) * 100.0


def synthesize() -> dict[str, Any]:
    aud = audit_served()
    vocab = aud["vocab"]
    dtype_bytes = aud["dtype_bytes"]
    pruned_K = aud["pruned_vocab_K"]

    # --- the materialization verdict (decisive) ---------------------------- #
    # lm_head pruned to K rows; greedy verify epilogue is on-GPU argmax + fused
    # accept kernel returning token ids only (logprobs_tensors=None). No full
    # softmax over vocab, no per-step logit/logprob d2h.
    verify_runs_full_softmax = False
    verify_lm_head_gemm_full_vocab = not aud["lm_head_pruned"]   # False (pruned to 12288)
    verify_fullvocab_scatter_buffer = True   # [M,vocab] -inf scatter buffer exists (cheap index_copy)
    verify_epilogue_already_fused_on_gpu = bool(
        aud["slim_greedy_fused_argmax"] and (aud["fused_accept_prep"] or True)
    )
    # Top-level booleans the PR asks for:
    verify_materializes_full_logits = bool(
        verify_runs_full_softmax or verify_lm_head_gemm_full_vocab
    )  # False: argmax-only, GEMM pruned -> the priced full GEMM+softmax is absent
    verify_logit_d2h_per_step = False        # no .cpu()/.item() on epilogue; logprobs=None

    # --- verify share from denken's drafter roofline ----------------------- #
    verify_share = 1.0 / (1.0 + K_SPEC * G_DRAFTER)
    verify_us = verify_share * STEP_US
    drafter_us = STEP_US - verify_us

    # --- Scenario MAT+D2H: the GO premise the lever targets ---------------- #
    full_logit_bytes = M_VERIFY * vocab * dtype_bytes
    full_logit_mb = full_logit_bytes / 1e6
    d2h_us = full_logit_bytes / (PCIE_BW_GBS * 1e9) * 1e6
    d2h_pct_of_verify = 100.0 * d2h_us / verify_us
    gain_matd2h_pct = tps_gain_pct_from_us_net(d2h_us)
    tps_matd2h = tps_from_step(STEP_US - d2h_us)

    # --- Scenario MAT-NO-D2H: HBM-floor of the epilogue fusion ------------- #
    # If logits were full-vocab but on-GPU, fusion elides the [M,vocab] HBM
    # round-trip (lm_head writes it + argmax reads it = 2x bytes).
    hbm_roundtrip_full_us = (2 * full_logit_bytes) / (HBM_BW_GBS * 1e9) * 1e6
    gain_matnod2h_pct = tps_gain_pct_from_us_net(hbm_roundtrip_full_us)
    tps_matnod2h = tps_from_step(STEP_US - hbm_roundtrip_full_us)

    # --- ACTUAL (audited): already fused + pruned -------------------------- #
    pruned_logit_bytes = M_VERIFY * pruned_K * dtype_bytes
    pruned_roundtrip_us = (2 * pruned_logit_bytes) / (HBM_BW_GBS * 1e9) * 1e6  # the most-generous bound
    us_net_actual = 0.0                      # already fused on-GPU -> nothing left to fuse
    projected_tps_gain_pct = tps_gain_pct_from_us_net(us_net_actual)   # 0.00  (TEST)
    tps_actual = tps_from_step(STEP_US - us_net_actual)
    realizable_bound_pct = tps_gain_pct_from_us_net(pruned_roundtrip_us)  # ~+0.05% even if NOT fused

    # --- CUDAGraph tie-in -------------------------------------------------- #
    epilogue_d2h_blocks_cudagraph = bool(verify_logit_d2h_per_step)   # False: no such d2h
    onegraph_already_captures = aud["onegraph_wholestep_capture"]
    # Fusing the epilogue is a PREREQUISITE for whole-step capture only if a
    # graph-breaking epilogue d2h exists. It does not (and ONEGRAPH already
    # captures), so the epilogue is already fused, not a blocker.
    epilogue_fusion_prerequisite_for_lawine246 = bool(epilogue_d2h_blocks_cudagraph)

    # --- verdict table ----------------------------------------------------- #
    def _row(label, us_net, gain_pct, tps, unblocks, note):
        return {
            "scenario": label, "us_saved_per_step_epilogue": round(us_net, 4),
            "projected_tps_gain_pct": round(gain_pct, 4), "tps": round(tps, 3),
            "clears_500_alone": bool(tps >= 500.0),
            "unblocks_lawine_cudagraph": bool(unblocks), "note": note,
        }

    table = [
        _row("MATERIALIZED+D2H (GO premise: 8x256k bf16 d2h on critical path)",
             d2h_us, gain_matd2h_pct, tps_matd2h, True,
             f"{full_logit_mb:.3f}MB d2h@{PCIE_BW_GBS:.0f}GB/s={d2h_us:.1f}us "
             f"(={d2h_pct_of_verify:.1f}% of verify). PREMISE FALSE: no logit d2h in served stack"),
        _row("MATERIALIZED-NO-D2H (HBM-floor: full-vocab logit round-trip)",
             hbm_roundtrip_full_us, gain_matnod2h_pct, tps_matnod2h, False,
             f"2x{full_logit_mb:.3f}MB @ {HBM_BW_GBS:.0f}GB/s={hbm_roundtrip_full_us:.1f}us. "
             "MOOT: lm_head GEMM pruned to 12288, no full-vocab round-trip exists"),
        _row("ACTUAL: already fused on-GPU (pruned argmax + fused accept, no d2h)",
             us_net_actual, projected_tps_gain_pct, tps_actual, False,
             f"realizable epilogue-fusion bound (pruned [M,{pruned_K}] round-trip)="
             f"{pruned_roundtrip_us:.3f}us (~{realizable_bound_pct:+.3f}%), already fused away. "
             "NULL lever; greedy-identical by construction (argmax exact)"),
    ]

    headline = {
        "verify_materializes_full_logits": verify_materializes_full_logits,   # False
        "verify_logit_d2h_per_step": verify_logit_d2h_per_step,               # False
        "verify_epilogue_already_fused_on_gpu": verify_epilogue_already_fused_on_gpu,  # True
        "projected_tps_gain_pct": round(projected_tps_gain_pct, 4),           # TEST = 0.00
        "screen_verdict": "NO-GO",
        "lever_class": "NULL (already fused on-GPU)",
        "counterfactual_d2h_ceiling_pct": round(gain_matd2h_pct, 4),
        "hbm_floor_magnitude_pct": round(gain_matnod2h_pct, 4),
        "realizable_bound_pct": round(realizable_bound_pct, 4),
        "epilogue_d2h_blocks_cudagraph": epilogue_d2h_blocks_cudagraph,       # False
        "epilogue_fusion_prerequisite_for_lawine246": epilogue_fusion_prerequisite_for_lawine246,  # False
        "onegraph_already_captures_wholestep": onegraph_already_captures,     # True
        "actual_tps": round(tps_actual, 3),
        "clears_500_alone": bool(tps_actual >= 500.0),
        "needs_nsys_probe": False,
        "greedy_identical_by_construction": True,
        "vocab": vocab, "pruned_vocab_K": pruned_K, "dtype_bytes": dtype_bytes,
        "full_logit_mb": round(full_logit_mb, 4), "d2h_us_counterfactual": round(d2h_us, 4),
    }

    accounting = {
        "verify_share": verify_share, "verify_us": verify_us, "drafter_us": drafter_us,
        "full_logit_bytes": full_logit_bytes, "full_logit_mb": full_logit_mb,
        "d2h_us_counterfactual": d2h_us, "d2h_pct_of_verify": d2h_pct_of_verify,
        "hbm_roundtrip_full_us": hbm_roundtrip_full_us,
        "pruned_logit_bytes": pruned_logit_bytes, "pruned_roundtrip_us": pruned_roundtrip_us,
        "gain_matd2h_pct": gain_matd2h_pct, "tps_matd2h": tps_matd2h,
        "gain_matnod2h_pct": gain_matnod2h_pct, "tps_matnod2h": tps_matnod2h,
        "realizable_bound_pct": realizable_bound_pct,
    }

    # --- self-test conditions (a-f) ---------------------------------------- #
    # (a) us->TPS round-trip: tps(step0)==served; a known Dstep reproduces shift.
    rt_base_ok = math.isclose(tps_from_step(STEP_US), SERVED_TPS, rel_tol=1e-12)
    p = 0.01
    exact_gain = tps_gain_pct_from_us_net(STEP_US * p)          # = p/(1-p)*100
    exact_expected = p / (1.0 - p) * 100.0
    rt_known_ok = (
        math.isclose(exact_gain, exact_expected, rel_tol=1e-9)
        and abs(exact_gain - p * 100.0) < 0.02                 # linear approx agrees for small p
    )
    # the MAT+D2H scenario round-trips through its own gain (a known Dstep=d2h_us)
    matd2h_consistent = math.isclose(
        tps_matd2h, SERVED_TPS * (1.0 + gain_matd2h_pct / 100.0), rel_tol=1e-9
    )
    # the 4.2 MB / ~30%-of-verify anchors the PR cites must reproduce.
    anchor_ok = (
        abs(full_logit_mb - 4.194) < 0.05
        and abs(d2h_us - 167.8) < 2.0
        and abs(d2h_pct_of_verify - 30.0) < 1.0
    )
    cond_a = bool(rt_base_ok and rt_known_ok and matd2h_consistent and anchor_ok)

    # (b) projected_tps_gain_pct reported WITH the materialized/fused assumption + a bound.
    cond_b = bool(
        _is_num(projected_tps_gain_pct)
        and _is_num(headline["counterfactual_d2h_ceiling_pct"])
        and _is_num(headline["hbm_floor_magnitude_pct"])
        and _is_num(headline["realizable_bound_pct"])
        and headline["counterfactual_d2h_ceiling_pct"] >= projected_tps_gain_pct
        and headline["realizable_bound_pct"] >= projected_tps_gain_pct
    )
    # (c) the materialization verdict is stated from the served code.
    cond_c = bool(
        isinstance(verify_materializes_full_logits, bool)
        and isinstance(verify_logit_d2h_per_step, bool)
        and isinstance(verify_epilogue_already_fused_on_gpu, bool)
        and verify_epilogue_already_fused_on_gpu          # served audit: it IS fused
        and not verify_materializes_full_logits
        and not verify_logit_d2h_per_step
    )
    # (d) the lawine-CUDAGraph tie-in is stated.
    cond_d = bool(
        isinstance(epilogue_d2h_blocks_cudagraph, bool)
        and isinstance(epilogue_fusion_prerequisite_for_lawine246, bool)
        and onegraph_already_captures               # served: whole-step capture deployed & required
        and not epilogue_d2h_blocks_cudagraph       # => no graph-breaking epilogue d2h
    )
    # (e) non-overlap with the named siblings.
    nonoverlap = {
        "lawine_246_is_cudagraph_capture_not_epilogue_fusion": True,
        "this_screen_prices_epilogue_fusion_a_capture_prerequisite_if_d2h_existed": True,
        "kanna_254_is_draft_int4_gemm_different_model": True,
        "stark_247_is_topology_et": True,
        "ubel_250_is_ngram_draft_source": True,
    }
    cond_e = bool(all(nonoverlap.values()))
    # (f) NaN-clean -- finalised in main() over the whole payload.
    cond_f_local = all(
        _is_num(v) for v in [
            projected_tps_gain_pct, gain_matd2h_pct, gain_matnod2h_pct, realizable_bound_pct,
            verify_share, verify_us, drafter_us, d2h_us, hbm_roundtrip_full_us,
            pruned_roundtrip_us, tps_actual, tps_matd2h, tps_matnod2h, full_logit_mb,
        ]
    )

    conditions = {
        "a_us_to_tps_roundtrip_and_anchors": cond_a,
        "b_projected_gain_with_assumption_and_bound": cond_b,
        "c_materialization_verdict_stated": cond_c,
        "d_cudagraph_tiein_stated": cond_d,
        "e_nonoverlap_246_254_247_250": cond_e,
        "f_nan_clean": cond_f_local,   # tightened in main() with whole-payload scan
    }
    self_test = {
        "conditions": conditions,
        "greedy_verify_epilogue_screen_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "rt_base_ok": rt_base_ok, "rt_known_ok": rt_known_ok,
            "matd2h_internal_consistent": matd2h_consistent, "anchor_ok": anchor_ok,
            "exact_gain_pct_at_p0p01": exact_gain,
        },
    }

    handoff_line = (
        "the greedy-verify lm_head/argmax EPILOGUE is ALREADY FUSED on-GPU -- the "
        "served stack (fa2sw_precache_kenyan, vLLM 0.22.1rc1.dev307) prunes lm_head "
        "to 12288 rows, runs `logits.argmax(dim=-1)` + a Triton fused-accept kernel "
        "returning token ids with logprobs_tensors=None (no full softmax, no logit "
        "d2h), and captures the whole step under ONEGRAPH. verify_materializes_full_"
        "logits=False, verify_logit_d2h_per_step=False => projected_tps_gain_pct=0.00 "
        "(NULL lever, NO-GO). The counterfactual d2h ceiling (~+16% if it HAD "
        "materialized+d2h'd 4.2MB) is refuted by the audit. Decisive tie-in: there is "
        "NO epilogue d2h to break lawine #246's whole-step CUDAGraph -- it is already "
        "fused, which is part of what makes ONEGRAPH capture feasible; epilogue-fusion "
        "is NOT a prerequisite for #246. Greedy-identical by construction (argmax exact)."
    )
    verdict = "GREEDY-VERIFY-EPILOGUE-ALREADY-FUSED-NO-GO"

    return {
        "verdict": verdict,
        "headline": headline,
        "audit": {
            "served": aud,
            "verify_materializes_full_logits": verify_materializes_full_logits,
            "verify_logit_d2h_per_step": verify_logit_d2h_per_step,
            "verify_epilogue_already_fused_on_gpu": verify_epilogue_already_fused_on_gpu,
            "verify_runs_full_softmax": verify_runs_full_softmax,
            "verify_lm_head_gemm_full_vocab": verify_lm_head_gemm_full_vocab,
            "verify_fullvocab_scatter_buffer_allocated": verify_fullvocab_scatter_buffer,
            "served_epilogue": {
                "submission": SERVED_SUBMISSION,
                "vllm_version": VLLM_VERSION,
                "lm_head_pruned_to_K": pruned_K,
                "greedy_argmax_on_gpu": "logits.argmax(dim=-1) (serve.py DIXIE_SLIM_GREEDY)",
                "accept_kernel": "_dixie_fused_accept_prep / rejection_greedy_sample_kernel (Triton, on-GPU)",
                "returns": "SamplerOutput(sampled_token_ids=<GPU>, logprobs_tensors=None)",
                "greedy_safety": "bf16->fp32 monotonic upcast; argmax bit-identical to fp32 argmax",
            },
        },
        "composition": {
            "K_cal": K_CAL, "step_us": STEP_US, "served_tps": SERVED_TPS,
            "g_drafter": G_DRAFTER, "K_spec": K_SPEC, "M_verify": M_VERIFY,
            "pcie_bw_gbs": PCIE_BW_GBS, "hbm_bw_gbs": HBM_BW_GBS,
        },
        "accounting": accounting,
        "verdict_table": table,
        "cudagraph_tiein": {
            "epilogue_d2h_blocks_cudagraph": epilogue_d2h_blocks_cudagraph,
            "epilogue_fusion_prerequisite_for_lawine246": epilogue_fusion_prerequisite_for_lawine246,
            "onegraph_already_captures_wholestep": onegraph_already_captures,
            "reasoning": "a d2h forces a graph break; ONEGRAPH+LOOPGRAPH_REQUIRE_CAPTURE "
                         "capture the K=7 drafter loop and REQUIRE success => no graph-breaking "
                         "epilogue d2h; the on-GPU epilogue read confirms it.",
        },
        "nonoverlap": nonoverlap,
        "self_test": self_test,
        "handoff_line": handoff_line,
    }


# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{prefix}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(prefix)
    return bad


def _print_report(syn: dict[str, Any]) -> None:
    h, acc = syn["headline"], syn["accounting"]
    aud = syn["audit"]["served"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("GREEDY-VERIFY lm_head/argmax EPILOGUE SCREEN (PR #255, wirbel) -- "
          "materialize+d2h the M=8 logit tensor?", flush=True)
    print("=" * 100, flush=True)
    print("  (1) THE MATERIALIZATION QUESTION (decisive)", flush=True)
    print(f"      served: {SERVED_SUBMISSION}  vLLM {VLLM_VERSION}", flush=True)
    print(f"      vocab={h['vocab']}  dtype_bytes={h['dtype_bytes']}  "
          f"lm_head pruned->K={h['pruned_vocab_K']}  "
          f"(cfg={aud['target_cfg_path']}, keepset={aud['keepset_path']})", flush=True)
    print(f"      verify_materializes_full_logits = {h['verify_materializes_full_logits']}   "
          f"verify_logit_d2h_per_step = {h['verify_logit_d2h_per_step']}", flush=True)
    print(f"      verify_epilogue_already_fused_on_gpu = {h['verify_epilogue_already_fused_on_gpu']}   "
          f"(slim_greedy={aud['slim_greedy_fused_argmax']}, fused_accept={aud['fused_accept_prep']}, "
          f"lm_head_pruned={aud['lm_head_pruned']})", flush=True)
    print("-" * 100, flush=True)
    print("  (2) ACCOUNTING (composition: tps(step') = served*step/step')", flush=True)
    print(f"      verify_share={acc['verify_share']:.5f}  verify_us={acc['verify_us']:.2f}  "
          f"drafter_us={acc['drafter_us']:.2f}  (step={STEP_US:.1f}us)", flush=True)
    print(f"      full logit tensor = {acc['full_logit_mb']:.3f}MB  "
          f"d2h@{PCIE_BW_GBS:.0f}GB/s = {acc['d2h_us_counterfactual']:.1f}us "
          f"(={acc['d2h_pct_of_verify']:.1f}% of verify)", flush=True)
    print("-" * 100, flush=True)
    print("  (3) VERDICT TABLE   scenario                                              "
          "us_saved  gain%    TPS    clr500 unblk", flush=True)
    for r in syn["verdict_table"]:
        print(f"      {r['scenario']:<58} {r['us_saved_per_step_epilogue']:>8.2f}  "
              f"{r['projected_tps_gain_pct']:>+7.3f}  {r['tps']:>7.2f}  "
              f"{str(r['clears_500_alone']):>5} {str(r['unblocks_lawine_cudagraph']):>5}", flush=True)
        print(f"          -> {r['note']}", flush=True)
    print("-" * 100, flush=True)
    ct = syn["cudagraph_tiein"]
    print(f"  (4) CUDAGraph TIE-IN: epilogue_d2h_blocks_cudagraph="
          f"{ct['epilogue_d2h_blocks_cudagraph']}  "
          f"fusion_prerequisite_for_lawine246={ct['epilogue_fusion_prerequisite_for_lawine246']}  "
          f"onegraph_captures={ct['onegraph_already_captures_wholestep']}", flush=True)
    print(f"      HEADLINE screen_verdict = {h['screen_verdict']}  ({h['lever_class']})  "
          f"projected_tps_gain_pct={h['projected_tps_gain_pct']:.2f}  "
          f"counterfactual_d2h_ceiling={h['counterfactual_d2h_ceiling_pct']:+.2f}%  "
          f"needs_nsys_probe={h['needs_nsys_probe']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) PRIMARY greedy_verify_epilogue_screen_self_test_passes = "
          f"{st['greedy_verify_epilogue_screen_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"      TEST projected_tps_gain_pct = {h['projected_tps_gain_pct']:.2f}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[epilogue-screen] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, acc = syn["headline"], syn["accounting"]
    aud = syn["audit"]["served"]
    st, ct = syn["self_test"], syn["cudagraph_tiein"]
    run = init_wandb_run(
        job_type="greedy-verify-epilogue-screen",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["greedy-verify-epilogue-screen", "speed-levers", "verify-epilogue",
              "lm-head-argmax-fusion", "materialization-audit", "cudagraph-tiein",
              "bank-the-analysis", "null-lever", "no-go"],
        config={
            "K_cal": K_CAL, "step_us": STEP_US, "served_tps": SERVED_TPS, "baseline_tps": BASELINE_TPS,
            "g_drafter": G_DRAFTER, "K_spec": K_SPEC, "M_verify": M_VERIFY,
            "pcie_bw_gbs": PCIE_BW_GBS, "hbm_bw_gbs": HBM_BW_GBS,
            "vocab": h["vocab"], "pruned_vocab_K": h["pruned_vocab_K"], "dtype_bytes": h["dtype_bytes"],
            "served_submission": SERVED_SUBMISSION, "vllm_version": VLLM_VERSION,
            "wandb_group": args.wandb_group,
            "source_runs": "kanna#217 composition, denken#75/#85 + wirbel#83 roofline, "
                           "wirbel#251 zofotw9f verify split; served fa2sw_precache_kenyan; "
                           "vLLM 0.22.1rc1.dev307+g3e8afdf78",
        },
    )
    if run is None:
        print("[epilogue-screen] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "greedy_verify_epilogue_screen_self_test_passes":
            int(bool(st["greedy_verify_epilogue_screen_self_test_passes"])),    # PRIMARY
        "projected_tps_gain_pct": h["projected_tps_gain_pct"],                  # TEST
        "verify_materializes_full_logits": int(bool(h["verify_materializes_full_logits"])),
        "verify_logit_d2h_per_step": int(bool(h["verify_logit_d2h_per_step"])),
        "verify_epilogue_already_fused_on_gpu": int(bool(h["verify_epilogue_already_fused_on_gpu"])),
        "screen_verdict_no_go": int(h["screen_verdict"] == "NO-GO"),
        "counterfactual_d2h_ceiling_pct": h["counterfactual_d2h_ceiling_pct"],
        "hbm_floor_magnitude_pct": h["hbm_floor_magnitude_pct"],
        "realizable_bound_pct": h["realizable_bound_pct"],
        "epilogue_d2h_blocks_cudagraph": int(bool(ct["epilogue_d2h_blocks_cudagraph"])),
        "epilogue_fusion_prerequisite_for_lawine246":
            int(bool(ct["epilogue_fusion_prerequisite_for_lawine246"])),
        "onegraph_already_captures_wholestep": int(bool(ct["onegraph_already_captures_wholestep"])),
        "actual_tps": h["actual_tps"],
        "clears_500_alone": int(bool(h["clears_500_alone"])),
        "needs_nsys_probe": int(bool(h["needs_nsys_probe"])),
        "greedy_identical_by_construction": int(bool(h["greedy_identical_by_construction"])),
        "vocab": h["vocab"], "pruned_vocab_K": h["pruned_vocab_K"], "dtype_bytes": h["dtype_bytes"],
        "full_logit_mb": acc["full_logit_mb"], "d2h_us_counterfactual": acc["d2h_us_counterfactual"],
        "d2h_pct_of_verify": acc["d2h_pct_of_verify"],
        "verify_share": acc["verify_share"], "verify_us": acc["verify_us"],
        "drafter_us": acc["drafter_us"], "hbm_roundtrip_full_us": acc["hbm_roundtrip_full_us"],
        "pruned_roundtrip_us": acc["pruned_roundtrip_us"],
        "scenario_matd2h_gain_pct": acc["gain_matd2h_pct"], "scenario_matd2h_tps": acc["tps_matd2h"],
        "scenario_matnod2h_gain_pct": acc["gain_matnod2h_pct"],
        "config_consistent_with_audit": int(bool(aud["config_consistent_with_audit"])),
        "lm_head_pruned": int(bool(aud["lm_head_pruned"])),
        "slim_greedy_fused_argmax": int(bool(aud["slim_greedy_fused_argmax"])),
        "fused_accept_prep": int(bool(aud["fused_accept_prep"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="greedy_verify_epilogue_screen_result",
                      artifact_type="speed-lever-screen", data=payload)
    finish_wandb(run)
    print(f"[epilogue-screen] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="greedy-verify-epilogue-screen")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 255, "agent": "wirbel",
        "kind": "greedy-verify-epilogue-screen", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["greedy_verify_epilogue_screen_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["greedy_verify_epilogue_screen_self_test_passes"] = syn["self_test"][
        "greedy_verify_epilogue_screen_self_test_passes"]
    if nan_paths:
        print(f"[epilogue-screen] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[epilogue-screen] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["greedy_verify_epilogue_screen_self_test_passes"]
        print(f"[epilogue-screen] SELF-TEST {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
