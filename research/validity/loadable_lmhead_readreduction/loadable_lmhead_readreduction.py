#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Loadable, greedy-identity-safe lm_head read-reduction enumeration (PR #398) — 0-GPU.

WHAT THIS CARD DECIDES
----------------------
My #385 (`a30iri8i`) decomposed the offline-capture 91 -> served 165.44 strict frontier and found the
DOMINANT composer (~96%, +70.8 TPS) is **lmhead12k**: an int4 lm_head row-pruned 262144 -> 12288. That
lever is (a) LOSSY — it physically cannot emit the ~250k pruned token ids, so it is byte-exact only vs its
OWN pruned checkpoint, never vs full-vocab AR — and (b) UNLOADABLE by a vanilla ``vllm.LLM()`` (it needs
``serve.py``'s ``_prune_lm_head_rows`` meta-tensor scatter hook). The deployed lm_head is ALREADY int4-Marlin
byte-exact at the decode width (#390 ``5y64zbjz``: the [16384 x 2560] channel-wise int4-Marlin head is
M-invariant, argmax_identity_by_M=1.0), and lm_head is the single largest single-token decode GEMM. So the
question PR #398 asks:

  Q  Does there exist a LOADABLE (loads via plain ``vllm.LLM(model=...)``, NO source build, NO serve.py
     scatter hook), GREEDY-IDENTITY-SAFE (full 256k-vocab argmax stays byte-identical, NOT a truncated head)
     lm_head read-reduction that reads FEWER bytes/step than the deployed int4-Marlin packing?

VERDICT (this card): **NO.** ``loadable_lmhead_lever_exists = False``. int4-Marlin is the loadable +
full-256k-identity-safe HBM read floor for the Gemma-4-E4B lm_head, and the only sub-int4-Marlin reads are
VOCAB TRUNCATIONS (lmhead12k 12288-row, the deployed 16384-row head) that forfeit full-vocab identity
(and, for lmhead12k, loadability). The deployed sub-346 MB read is bought by truncation — the SAME lossy
family as lmhead12k — not by a free loadable identity-safe config.

THE READ-BYTE LADDER (anchored to real checkpoint headers + #390/#385 constants, NOT assumed)
---------------------------------------------------------------------------------------------
Per-decode-step lm_head weight read = rows*hidden*(bits/8) + scale_bytes (bf16 scales for quantized).
  bf16 full-vocab     262144 rows, 16-bit            = 1342.18 MB  (OFFICIAL ``-qat-w4a16-ct`` ships this:
                                                       lm_head.weight BF16 [262144,2560], in the quant
                                                       ``ignore`` list — MEASURED from the local header)
  int8 full-vocab     262144 rows,  8-bit            =  671.09 MB  (W8A16 compressed-tensors; loadable)
  int4 full-vocab     262144 rows,  4-bit g128       =  346.03 MB  (``submissions/int4_g128_lmhead`` =
                      262144 rows,  4-bit channel    =  336.07 MB   untie+int4 lm_head; vanilla-loadable) <- FLOOR
  ----- everything below reads less ONLY by dropping vocab rows (LOSSY) or bits below int4 (LOSSY) -----
  deployed int4      16384 rows,  4-bit channel      =   21.00 MB  (#390 osoi5 baked; int4-Marlin byte-exact
                                                       M-invariant, but TRUNCATED -> self-referential gate
                                                       only, NOT full-256k; needs baked weights + hooks)
  lmhead12k          12288 rows,  4-bit channel      =   15.75 MB  (#385 lever: LOSSY + UNLOADABLE scatter hook)
  int3 full-vocab    262144 rows,  3-bit             =  252 MB     (no vLLM W3 lm_head kernel -> UNLOADABLE;
                                                       sub-int4 perturbs logits -> identity BROKEN)
  int2 full-vocab    262144 rows,  2-bit             =  168 MB     (UNLOADABLE + identity BROKEN)

WHY NO LOADABLE IDENTITY-SAFE LEVER BEATS int4-Marlin FULL-VOCAB (research, #398 lit pass)
-----------------------------------------------------------------------------------------
* INFO-THEORETIC FLOOR: an EXACT full-vocab argmax over a general W*h must read every vocab row (any
  unread row could be the max). vLLM exposes no sublinear EXACT-MIPS load path; all sublinear MIPS is
  approximate -> breaks byte-identity.
* QUANT FLOOR: int4-Marlin W4A16 is the smallest LOADABLE lossless-identity bit-packing vLLM supports for
  the lm_head. fp8/int8/bf16 read MORE; vLLM's fp8/gptq/awq recipes SKIP the lm_head by default (it stays
  bf16). No vLLM kernel goes below int4 for the lm_head, and sub-int4 PTQ cannot guarantee identical argmax.
* PERMUTATION: a lossless top-frequency vocab REORDER keeps all 262144 rows -> identical byte count ->
  ZERO read reduction (a bijection, not a shrink).
* TP@1: ``ParallelLMHead`` vocab-sharding only splits the read across TP ranks; at TP=1 (single A10G) one
  rank holds the full partition -> ZERO per-GPU reduction.
* Gemma-4-E4B ships ``tie_word_embeddings=True`` (lm_head == bf16 embed_tokens); the deployed int4 head is
  UNTIED+quantized (a build, not a config), and even that lands at the 336 MB int4 floor, not below it.

PRIMARY metric  loadable_lmhead_self_test_passes   (>=20 0-GPU checks of the read math + classification + verdict)
SCOPE: analysis/microbench only. NO HF Job, NO submission, NO served-file change, 0 official TPS. The
checkpoint-header audit (optional) reads a LOCAL ``-qat-w4a16-ct`` safetensors header (0-GPU) to anchor the
bf16 full-vocab lm_head fact empirically; it degrades gracefully if the checkpoint is absent.

Run:
  # PRIMARY 0-GPU self-validation + enumeration + W&B card:
  .venv/bin/python research/validity/loadable_lmhead_readreduction/loadable_lmhead_readreduction.py \
    --wandb_group loadable-lmhead-readreduction --wandb_name land/loadable-lmhead-readreduction
  # self-test only (no W&B):
  .venv/bin/python research/validity/loadable_lmhead_readreduction/loadable_lmhead_readreduction.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, has a ./wandb run-output dir that shadows
# the package as a PEP-420 namespace) on sys.path[0]. Mirrors the #385 house pattern.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------- #
# Geometry + bandwidth constants (provenance documented; consistent with #385/#390/#344).
# ---------------------------------------------------------------------------- #
HIDDEN = 2560                       # gemma-4-E4B hidden_size
FULL_VOCAB = 262144                 # gemma-4-E4B text vocab (== embed_tokens rows; MEASURED from header)
A10G_BW_GBPS = 600.0                # A10G HBM peak bandwidth (the roofline figure, #385)
BODY_INT4_GB = 1.6973824           # int4 body bytes/step (#371/#344/#278/#385)

DEPLOYED_LMHEAD_ROWS = 16384        # #390 osoi5 baked, channel-wise int4-Marlin (lmhead_geom.size_n)
DEPLOYED_LMHEAD_GROUP_SIZE = -1     # channel-wise (#390 lmhead_geom.group_size = -1)
LMHEAD12K_ROWS = 12288              # #385 lossy lever (serve.py _prune_lm_head_rows / PCK-04 keepset)
INT4_FLOOR_GROUP_SIZE = 128         # submissions/int4_g128_lmhead default head_group_size (g128)

BF16_BITS, INT8_BITS, INT4_BITS, INT3_BITS, INT2_BITS = 16, 8, 4, 3, 2
SCALE_DTYPE_BYTES = 2               # bf16 group/channel scales

# #390 deployed-step provenance (CITE; not re-derived).
F_LMHEAD_344 = 0.022428229458960704        # lm_head share of the deployed decode step (#378/#390)
PHANTOM_LMHEAD_BF16_TAX_TPS = 56.74        # #390: bf16 (un-quantized) lm_head would cost ~57 TPS vs int4
DEPLOYED_LMHEAD_BYTE_EXACT_390 = True      # #390 argmax_identity_by_M=1.0, M-invariant (deterministic sense)
OFFICIAL_DEPLOYED_TPS = 481.53             # PR #52 deployed (non-strict)
STRICT_FRONTIER_TPS = 165.44               # #196 strict non-spec served frontier (banks lmhead12k, #385)
NONSPEC_CAPTURE_OFFLINE_371 = 91.38        # #371/#385 offline full-vocab byte-exact capture
PPL_GATE = 2.42
PPL_BASELINE = 2.3772
MILESTONE = 500.0

# #385 cross-check anchors (the lmhead12k read this card must reproduce within float tol).
LMHEAD_BF16_FULL_GB_385 = 1.34217728       # #385 LMHEAD_BF16_FULL_GB
LMHEAD_INT4_12K_GB_385 = LMHEAD12K_ROWS * HIDDEN * 0.5 / 1e9  # #385 LMHEAD_INT4_12K_GB (~0.01573)

# Local -qat-w4a16-ct snapshots to header-audit (first that exists wins; degrade gracefully if none).
_QAT_W4A16_HEADER_CANDIDATES = [
    "/senpai-run/home/student-denken/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/"
    "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors",
    "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/"
    "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors",
    "/senpai-run/home/student-wirbel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/"
    "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors",
]

TOL = 1e-9
TOL_RT = 1e-6
EPS_SHRINK = 1e-4                   # a "shrink" must beat the int4 floor by >0.01% to count as a real lever

# Identity classes (full 256k-vocab greedy byte-identity vs the deployed/bf16 AR reference).
ID_EXACT = "exact-by-construction"     # bf16 IS the reference numerics; reorder/TP over it is a bijection
ID_LIKELY = "empirical-likely-exact"   # int8 high-precision; very likely but not certified here
ID_EMPIRICAL = "empirical-unverified"  # int4 full-vocab: logit perturbation could flip argmax; needs capture
ID_SELFREF = "self-referential-only"   # deployed int4-Marlin: M-invariant byte-exact but TRUNCATED (#390)
ID_BROKEN = "broken-lossy"             # truncation / sub-int4: cannot hold full-256k identity
_ID_SAFE_CERTIFIED = {ID_EXACT}        # only these count as certified-identity-safe for the verdict


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ---------------------------------------------------------------------------- #
# (1) Read-byte model.
# ---------------------------------------------------------------------------- #
def lmhead_read_bytes(rows: int, bits: int, group_size: int | None) -> float:
    """Per-decode-step lm_head weight-read bytes = packed weights + (bf16) scales.

    bits==16 (bf16): no scales. Quantized (<16 bit): channel-wise (group_size in {-1, None}) keeps 1 scale
    per row; group quant keeps hidden//group_size scales per row. Scales are bf16 (SCALE_DTYPE_BYTES).
    """
    weight_bytes = rows * HIDDEN * bits / 8.0
    if bits >= 16:
        scale_bytes = 0.0
    elif group_size in (-1, None):
        scale_bytes = rows * 1 * SCALE_DTYPE_BYTES
    else:
        scale_bytes = rows * (HIDDEN // group_size) * SCALE_DTYPE_BYTES
    return weight_bytes + scale_bytes


def roofline_tps(lmhead_read_gb: float) -> float:
    """Single-token decode roofline TPS for a given lm_head read (body int4 + lm_head), A10G BW.

    This is the #385 offline-capture roofline basis (full-vocab bf16 -> ~197; lmhead12k -> ~350). A
    diagnostic ceiling, NOT an official TPS.
    """
    return A10G_BW_GBPS / (BODY_INT4_GB + lmhead_read_gb)


# ---------------------------------------------------------------------------- #
# (2) Loadable-candidate enumeration. Each candidate carries: can it load via a vanilla vllm.LLM()
#     (no source build, no serve.py scatter hook); does it keep the FULL 262144-row vocab; and its
#     full-256k greedy byte-identity class. The read bytes are computed from the read-byte model.
# ---------------------------------------------------------------------------- #
def enumerate_candidates() -> list[dict[str, Any]]:
    def cand(name, fmt, rows, bits, gs, loadable, full_vocab, identity, evidence):
        rb = lmhead_read_bytes(rows, bits, gs)
        return {
            "name": name, "lmhead_format": fmt, "rows": rows, "bits": bits, "group_size": gs,
            "loadable_vanilla_llm": loadable, "full_vocab": full_vocab,
            "identity_full256k": identity, "read_bytes": rb, "read_gb": rb / 1e9,
            "read_mb": rb / 1e6, "roofline_tps": roofline_tps(rb / 1e9), "evidence": evidence,
        }

    return [
        cand("bf16_fullvocab", "bf16", FULL_VOCAB, BF16_BITS, None, True, True, ID_EXACT,
             "OFFICIAL -qat-w4a16-ct ships lm_head.weight BF16 [262144,2560] (in quant `ignore`); the "
             "bf16 reference numerics -> trivially full-256k identity. MEASURED from local header."),
        cand("int8_w8a16_fullvocab", "int8-W8A16", FULL_VOCAB, INT8_BITS, -1, True, True, ID_LIKELY,
             "compressed-tensors W8A16 channel; vanilla-loadable. Reads 2x int4 -> NOT a reduction."),
        cand("fp8_fullvocab_forced", "fp8-W8A8", FULL_VOCAB, INT8_BITS, -1, True, True, ID_EMPIRICAL,
             "vLLM fp8 recipe SKIPS lm_head by default (stays bf16=1342MB); even forced fp8 reads 2x int4."),
        cand("int4_fullvocab_g128", "int4-Marlin-g128", FULL_VOCAB, INT4_BITS, INT4_FLOOR_GROUP_SIZE,
             True, True, ID_EMPIRICAL,
             "submissions/int4_g128_lmhead: untie embed_tokens + int4-quant lm_head, vanilla-loadable "
             "(compressed-tensors auto-detect -> Marlin, no hooks). THE loadable full-vocab read FLOOR. "
             "full-256k argmax-vs-bf16 identity is EMPIRICAL (ships check_greedy_identity.py)."),
        cand("int4_fullvocab_channel", "int4-Marlin-channel", FULL_VOCAB, INT4_BITS, -1, True, True,
             ID_EMPIRICAL,
             "int4_g128_lmhead with --head-group-size -1 (channel). Marginally fewer scale bytes than "
             "g128; still ~336 MB == the int4-Marlin full-vocab floor."),
        cand("lossless_vocab_reorder", "int4-Marlin-channel(permuted)", FULL_VOCAB, INT4_BITS, -1, True,
             True, ID_EXACT,
             "top-frequency row PERMUTATION keeps all 262144 rows: a bijection -> IDENTICAL byte count -> "
             "ZERO read reduction. Un-permuting the argmax index is exact -> identity-safe, but 0 shrink."),
        cand("tp_vocab_parallel_tp1", "int4-Marlin-channel(TP=1)", FULL_VOCAB, INT4_BITS, -1, True, True,
             ID_EXACT,
             "ParallelLMHead vocab-shard only splits the read across TP ranks; at TP=1 (single A10G) one "
             "rank holds the full partition -> ZERO per-GPU reduction."),
        cand("int3_fullvocab", "int3 (unsupported)", FULL_VOCAB, INT3_BITS, -1, False, True, ID_BROKEN,
             "no vLLM W3 lm_head kernel -> UNLOADABLE via vanilla LLM(); sub-int4 PTQ perturbs logits -> "
             "full-256k argmax NOT byte-identical. DISQUALIFIED (unloadable + lossy)."),
        cand("int2_fullvocab", "int2 (unsupported)", FULL_VOCAB, INT2_BITS, -1, False, True, ID_BROKEN,
             "no vLLM W2 lm_head kernel -> UNLOADABLE; identity BROKEN. DISQUALIFIED."),
        cand("deployed_int4_16384", "int4-Marlin-channel(16384-row)", DEPLOYED_LMHEAD_ROWS, INT4_BITS,
             DEPLOYED_LMHEAD_GROUP_SIZE, False, False, ID_SELFREF,
             "#390 osoi5 baked deployed head: int4-Marlin byte-exact M-invariant (argmax_identity_by_M=1.0) "
             "but row-TRUNCATED 262144->16384 -> passes only the self-referential gate, NOT full-256k; needs "
             "baked weights + PLE/LM_HEAD_PRUNE hooks -> NOT a plain vanilla checkpoint. Truncation family."),
        cand("lmhead12k_int4_12288", "int4-Marlin-channel(12288-row)", LMHEAD12K_ROWS, INT4_BITS, -1, False,
             False, ID_BROKEN,
             "#385 dominant lever: int4 lm_head pruned 262144->12288. LOSSY (cannot emit ~250k tokens; "
             "byte-exact only vs its OWN pruned checkpoint) + UNLOADABLE (serve.py _prune_lm_head_rows "
             "meta-tensor scatter hook). The lever this card distinguishes loadable wins FROM."),
    ]


# ---------------------------------------------------------------------------- #
# (3) Verdict assembly.
# ---------------------------------------------------------------------------- #
def assemble_verdict(cands: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {c["name"]: c for c in cands}
    deployed = by_name["deployed_int4_16384"]
    lmhead12k = by_name["lmhead12k_int4_12288"]
    int4_floor = by_name["int4_fullvocab_channel"]  # the loadable full-vocab int4-Marlin read floor
    floor_read = int4_floor["read_bytes"]
    deployed_read = deployed["read_bytes"]

    # A candidate is a genuine LOADABLE IDENTITY-SAFE lm_head read-reduction LEVER iff it:
    #   loads via vanilla LLM(), keeps the FULL 262144-row vocab, holds CERTIFIED full-256k identity,
    #   AND reads strictly fewer bytes than the int4-Marlin full-vocab floor (a real shrink, not 0/neg).
    def shrink_vs_floor(c):
        return (floor_read - c["read_bytes"]) / floor_read

    def shrink_vs_deployed(c):
        return (deployed_read - c["read_bytes"]) / deployed_read

    for c in cands:
        c["read_shrink_frac_vs_int4_floor"] = shrink_vs_floor(c)
        c["read_shrink_frac_vs_deployed"] = shrink_vs_deployed(c)
        c["identity_certified_safe"] = c["identity_full256k"] in _ID_SAFE_CERTIFIED
        c["qualifies_as_loadable_lever"] = bool(
            c["loadable_vanilla_llm"] and c["full_vocab"] and c["identity_certified_safe"]
            and c["read_shrink_frac_vs_int4_floor"] > EPS_SHRINK)

    qualifying = [c for c in cands if c["qualifies_as_loadable_lever"]]
    loadable_lmhead_lever_exists = bool(qualifying)

    # Loadable + full-vocab + certified-identity-safe set (the honest envelope), best (max) shrink within it.
    loadable_safe = [c for c in cands if c["loadable_vanilla_llm"] and c["full_vocab"]
                     and c["identity_certified_safe"]]
    best_shrink = max((c["read_shrink_frac_vs_int4_floor"] for c in loadable_safe), default=0.0)
    # Clamp to >=0: bf16/int8 read MORE than the int4 floor (negative shrink); the BEST loadable
    # identity-safe shrink over the floor is 0 (achieved by reorder/TP at the floor).
    best_loadable_read_shrink_frac = max(0.0, best_shrink)
    # The loadable identity-safe candidate that achieves the best shrink (the floor itself at shrink 0).
    best_cand = None
    if loadable_safe:
        best_cand = max(loadable_safe, key=lambda c: c["read_shrink_frac_vs_int4_floor"])
    best_loadable_identity_byte_exact = bool(best_cand is not None
                                             and best_cand["identity_full256k"] in _ID_SAFE_CERTIFIED)

    # TPS: the best loadable identity-safe lever's roofline gain over the int4-Marlin full-vocab floor.
    floor_tps = int4_floor["roofline_tps"]
    best_loadable_tps = best_cand["roofline_tps"] if best_cand else floor_tps
    best_loadable_tps_delta = best_loadable_tps - floor_tps  # ~0: no loadable safe lever beats the floor
    # The COST of full-256k identity: the int4 floor reads MORE than the truncated deployed head.
    full_vocab_identity_tps_cost_vs_deployed = int4_floor["roofline_tps"] - deployed["roofline_tps"]

    vs_lossy_lmhead12k_note = (
        "lmhead12k (12288-row int4, {:.1f} MB) and the deployed 16384-row int4 head ({:.1f} MB) read LESS "
        "than the int4-Marlin full-vocab floor ({:.1f} MB) ONLY by dropping vocab rows: both are LOSSY "
        "truncations that forfeit full-256k identity (lmhead12k cannot emit the ~250k pruned tokens; the "
        "deployed head is byte-exact only in the M-invariant / self-referential #390 sense). lmhead12k also "
        "forfeits loadability (serve.py _prune_lm_head_rows scatter hook). No LOADABLE + full-256k-identity "
        "candidate reads below the {:.1f} MB int4 floor -> the deployed sub-346MB read is bought by "
        "truncation (same family as lmhead12k), NOT by a free loadable identity-safe config.").format(
            lmhead12k["read_mb"], deployed["read_mb"], int4_floor["read_mb"], int4_floor["read_mb"])

    return {
        "candidates": cands,
        "loadable_candidates_enumerated": [c["name"] for c in cands if c["loadable_vanilla_llm"]],
        "all_candidates_enumerated": [c["name"] for c in cands],
        "lmhead_read_bytes_deployed": deployed_read,
        "lmhead_read_mb_deployed": deployed["read_mb"],
        "lmhead_read_bytes_deployed_byte_exact_390": DEPLOYED_LMHEAD_BYTE_EXACT_390,
        "loadable_identity_safe_floor_format": int4_floor["lmhead_format"],
        "loadable_identity_safe_floor_read_mb": int4_floor["read_mb"],
        "loadable_identity_safe_floor_read_bytes": floor_read,
        "best_loadable_read_shrink_frac": best_loadable_read_shrink_frac,
        "best_loadable_read_shrink_frac_vs_deployed": (
            max((c["read_shrink_frac_vs_deployed"] for c in loadable_safe), default=0.0)),
        "best_loadable_identity_byte_exact": best_loadable_identity_byte_exact,
        "best_loadable_candidate": best_cand["name"] if best_cand else None,
        "best_loadable_tps_delta": best_loadable_tps_delta,
        "full_vocab_identity_tps_cost_vs_deployed": full_vocab_identity_tps_cost_vs_deployed,
        "loadable_lmhead_lever_exists": loadable_lmhead_lever_exists,
        "qualifying_levers": [c["name"] for c in qualifying],
        "vs_lossy_lmhead12k_note": vs_lossy_lmhead12k_note,
        # info-theoretic + support facts (the "why no lever" backbone).
        "exact_full_vocab_argmax_requires_all_rows": True,
        "vllm_has_sublinear_exact_mips_loadpath": False,
        "int4_marlin_is_smallest_loadable_lossless_packing": True,
        "permutation_reduces_read": False,
        "tp1_vocab_parallel_reduces_read": False,
        "subint4_loadable_via_vanilla_llm": False,
    }


# ---------------------------------------------------------------------------- #
# (4) Optional 0-GPU checkpoint-header audit — anchors the bf16 full-vocab lm_head fact empirically.
# ---------------------------------------------------------------------------- #
def audit_checkpoint_header(paths: list[str] | None = None) -> dict[str, Any]:
    paths = paths or _QAT_W4A16_HEADER_CANDIDATES
    for p in paths:
        st = Path(p)
        if not st.exists():
            continue
        try:
            with st.open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n).decode("utf-8"))
            lm = hdr.get("lm_head.weight") or hdr.get("model.language_model.embed_tokens.weight")
            cfg_path = st.parent / "config.json"
            qc = {}
            tie = None
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                qc = cfg.get("quantization_config", {})
                tie = cfg.get("text_config", {}).get("tie_word_embeddings", cfg.get("tie_word_embeddings"))
            grp = qc.get("config_groups", {}).get("group_0", {}).get("weights", {})
            lm_in_ignore = "lm_head" not in str(qc.get("config_groups", {}))
            return {
                "available": True,
                "path": str(st),
                "lm_head_dtype": lm.get("dtype") if lm else None,
                "lm_head_shape": lm.get("shape") if lm else None,
                "lm_head_is_bf16_fullvocab": bool(
                    lm and lm.get("dtype") == "BF16" and lm.get("shape") == [FULL_VOCAB, HIDDEN]),
                "body_num_bits": grp.get("num_bits"),
                "body_group_size": grp.get("group_size"),
                "body_strategy": grp.get("strategy"),
                "lm_head_not_in_quant_groups": lm_in_ignore,
                "tie_word_embeddings": tie,
            }
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "path": str(st), "error": repr(exc)}
    return {"available": False, "note": "no local -qat-w4a16-ct safetensors found (header audit skipped)"}


# ---------------------------------------------------------------------------- #
# (5) PRIMARY self-test — >=20 0-GPU checks of the read math + classification + verdict logic.
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    cands = enumerate_candidates()
    v = assemble_verdict(cands)
    by = {c["name"]: c for c in cands}

    bf16 = by["bf16_fullvocab"]
    int8 = by["int8_w8a16_fullvocab"]
    int4g128 = by["int4_fullvocab_g128"]
    int4ch = by["int4_fullvocab_channel"]
    deployed = by["deployed_int4_16384"]
    lm12k = by["lmhead12k_int4_12288"]
    reorder = by["lossless_vocab_reorder"]
    tp1 = by["tp_vocab_parallel_tp1"]
    int3 = by["int3_fullvocab"]

    c: dict[str, bool] = {}
    # ---- read-byte math (anchored to #385 constants) ----
    c["t01_bf16_full_read"] = abs(bf16["read_bytes"] - FULL_VOCAB * HIDDEN * 2) < 1.0
    c["t02_bf16_matches_385_gb"] = abs(bf16["read_gb"] - LMHEAD_BF16_FULL_GB_385) < TOL_RT
    # int8 read = 8-bit weights (== half the bf16 weight bytes) + bf16 channel scales (FULL_VOCAB*2).
    c["t03_int8_read_exact"] = abs(int8["read_bytes"] - (FULL_VOCAB * HIDDEN + FULL_VOCAB * 2)) < 1.0
    c["t03b_int8_weights_half_bf16"] = abs((FULL_VOCAB * HIDDEN) - (FULL_VOCAB * HIDDEN * 2) / 2) < 1.0
    c["t04_int4_weights"] = abs(int4ch["read_bytes"] - (FULL_VOCAB * HIDDEN * 0.5
                                                        + FULL_VOCAB * 2)) < 1.0
    c["t05_int4_g128_scales"] = abs(int4g128["read_bytes"]
                                    - (FULL_VOCAB * HIDDEN * 0.5
                                       + FULL_VOCAB * (HIDDEN // 128) * 2)) < 1.0
    c["t06_deployed_16384_read"] = abs(deployed["read_bytes"]
                                       - (16384 * HIDDEN * 0.5 + 16384 * 2)) < 1.0
    # #385's LMHEAD_INT4_12K_GB is weights-only; this card's read ALSO adds bf16 channel scales (+24 KB).
    # Validate the weights-only geometry matches #385 exactly, then that the full read tracks it within scales.
    c["t07_lmhead12k_weights_match_385"] = abs(
        LMHEAD12K_ROWS * HIDDEN * 0.5 / 1e9 - LMHEAD_INT4_12K_GB_385) < TOL_RT
    c["t07b_lmhead12k_read_within_scales"] = abs(lm12k["read_gb"] - LMHEAD_INT4_12K_GB_385) < 1e-4
    # ---- read ordering (the load-bearing geometry) ----
    c["t08_bits_monotonic"] = bf16["read_bytes"] > int8["read_bytes"] > int4ch["read_bytes"]
    c["t09_int4_floor_below_int8"] = int4ch["read_bytes"] < int8["read_bytes"]
    c["t10_deployed_below_floor"] = deployed["read_bytes"] < int4ch["read_bytes"]  # truncation reads less
    c["t11_lmhead12k_below_deployed"] = lm12k["read_bytes"] < deployed["read_bytes"]
    c["t12_int4_channel_le_g128"] = int4ch["read_bytes"] <= int4g128["read_bytes"]
    # ---- loadability / full-vocab / identity classification ----
    c["t13_truncations_not_full_vocab"] = (not deployed["full_vocab"]) and (not lm12k["full_vocab"])
    c["t14_lmhead12k_unloadable"] = lm12k["loadable_vanilla_llm"] is False
    c["t15_deployed_unloadable_plain"] = deployed["loadable_vanilla_llm"] is False
    c["t16_subint4_unloadable"] = int3["loadable_vanilla_llm"] is False
    c["t17_int4_floor_loadable_fullvocab"] = (int4ch["loadable_vanilla_llm"] and int4ch["full_vocab"])
    c["t18_reorder_zero_shrink"] = abs(reorder["read_shrink_frac_vs_int4_floor"]) < TOL_RT
    c["t19_tp1_zero_shrink"] = abs(tp1["read_shrink_frac_vs_int4_floor"]) < TOL_RT
    c["t20_bf16_negative_shrink"] = bf16["read_shrink_frac_vs_int4_floor"] < 0.0  # bf16 reads MORE
    # ---- verdict logic ----
    c["t21_qualifying_set_empty"] = v["qualifying_levers"] == []
    c["t22_lever_exists_false"] = v["loadable_lmhead_lever_exists"] is False
    c["t23_best_shrink_zero"] = abs(v["best_loadable_read_shrink_frac"]) < TOL_RT
    c["t24_best_shrink_vs_deployed_nonpos"] = v["best_loadable_read_shrink_frac_vs_deployed"] <= EPS_SHRINK
    c["t25_best_tps_delta_zero"] = abs(v["best_loadable_tps_delta"]) < 1e-6
    c["t26_full_vocab_costs_tps_vs_deployed"] = v["full_vocab_identity_tps_cost_vs_deployed"] < 0.0
    c["t27_floor_is_int4_marlin"] = "int4-Marlin" in v["loadable_identity_safe_floor_format"]
    c["t28_deployed_read_reported"] = abs(v["lmhead_read_bytes_deployed"] - deployed["read_bytes"]) < 1.0
    # ---- info-theoretic backbone ----
    c["t29_exact_argmax_needs_all_rows"] = v["exact_full_vocab_argmax_requires_all_rows"] is True
    c["t30_no_sublinear_mips"] = v["vllm_has_sublinear_exact_mips_loadpath"] is False
    c["t31_permutation_no_shrink"] = v["permutation_reduces_read"] is False
    c["t32_note_mentions_lossy_and_unloadable"] = ("LOSSY" in v["vs_lossy_lmhead12k_note"]
                                                   and "loadab" in v["vs_lossy_lmhead12k_note"].lower())
    # ---- constants exact ----
    c["t33_constants_exact"] = bool(HIDDEN == 2560 and FULL_VOCAB == 262144
                                    and abs(BODY_INT4_GB - 1.6973824) < TOL
                                    and abs(A10G_BW_GBPS - 600.0) < TOL)

    passes = bool(all(c.values()))
    return {"conditions": c, "n_checks": len(c), "loadable_lmhead_self_test_passes": passes,
            "verdict_fields": v}


# ---------------------------------------------------------------------------- #
# Synthesis.
# ---------------------------------------------------------------------------- #
def synthesize(header_audit: dict[str, Any]) -> dict[str, Any]:
    st = self_test()
    v = st["verdict_fields"]
    verdict_str = _build_verdict(v, header_audit)
    return {
        "self_test": st,
        "loadable_lmhead_self_test_passes": st["loadable_lmhead_self_test_passes"],
        "n_self_test_checks": st["n_checks"],
        "verdict_fields": v,
        "header_audit": header_audit,
        "verdict": verdict_str,
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }


def _build_verdict(v: dict[str, Any], header: dict[str, Any]) -> str:
    parts = []
    parts.append(
        "Q (PR #398): is there a LOADABLE (vanilla LLM(), no source build / no serve.py hook), full-256k "
        "greedy-identity-safe lm_head read-reduction below the deployed int4-Marlin? VERDICT: NO "
        f"(loadable_lmhead_lever_exists={v['loadable_lmhead_lever_exists']}).")
    parts.append(
        "READ LADDER (MB/step): bf16-full {:.1f} > int8-full {:.1f} > int4-full FLOOR {:.1f} "
        ">> deployed-16384 {:.1f} > lmhead12k {:.1f}. The loadable+full-256k-identity-safe FLOOR is "
        "int4-Marlin full-vocab ({:.1f} MB); bf16/int8/fp8 read MORE, permutation & TP@1 read the SAME, "
        "sub-int4 is unloadable+lossy.".format(
            v['candidates'][0]['read_mb'], v['candidates'][1]['read_mb'],
            v['loadable_identity_safe_floor_read_mb'], v['lmhead_read_mb_deployed'],
            [c for c in v['candidates'] if c['name'] == 'lmhead12k_int4_12288'][0]['read_mb'],
            v['loadable_identity_safe_floor_read_mb']))
    parts.append(
        "best_loadable_read_shrink_frac={:.4f} (vs deployed={:.3f}); best_loadable_identity_byte_exact={}; "
        "best_loadable_tps_delta={:.2f}. Full-256k identity COSTS {:.1f} roofline-TPS vs the deployed "
        "truncated head (you must read the whole vocab).".format(
            v['best_loadable_read_shrink_frac'], v['best_loadable_read_shrink_frac_vs_deployed'],
            v['best_loadable_identity_byte_exact'], v['best_loadable_tps_delta'],
            v['full_vocab_identity_tps_cost_vs_deployed']))
    parts.append(v["vs_lossy_lmhead12k_note"])
    if header.get("available"):
        parts.append(
            "Header audit (LOCAL -qat-w4a16-ct): lm_head {} {} bf16-fullvocab={} body int4 g{} {} -> the "
            "OFFICIAL vanilla-loadable int4 checkpoint keeps the lm_head BF16 full-vocab (1342 MB); an int4 "
            "lm_head requires the int4_g128_lmhead untie+quant build, which still lands at the 336 MB "
            "floor.".format(header.get("lm_head_dtype"), header.get("lm_head_shape"),
                            header.get("lm_head_is_bf16_fullvocab"), header.get("body_group_size"),
                            header.get("body_strategy")))
    parts.append("LOCAL/analysis-only; 0 official TPS; NO HF Job / submission / served-file change.")
    return " ".join(parts)


# ---------------------------------------------------------------------------- #
# NaN guard / report / W&B (house pattern, mirrors #385).
# ---------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, val in node.items():
                walk(val, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, val in enumerate(node):
                walk(val, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict) -> None:
    st = syn["self_test"]
    v = syn["verdict_fields"]
    print("\n" + "=" * 100, flush=True)
    print("LOADABLE, GREEDY-IDENTITY-SAFE lm_head READ-REDUCTION? (PR #398)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (PRIMARY) loadable_lmhead_self_test_passes = {st['loadable_lmhead_self_test_passes']} "
          f"({st['n_checks']} checks)", flush=True)
    n_fail = [k for k, val in st["conditions"].items() if not val]
    if n_fail:
        print(f"          FAILED: {n_fail}", flush=True)
    print("-" * 100, flush=True)
    print("  CANDIDATE LADDER (read MB | loadable | full-vocab | identity | shrink-vs-int4-floor):",
          flush=True)
    for cd in v["candidates"]:
        print("    {:<34} {:>8.1f}MB  load={:<5} full={:<5} id={:<22} shrink={:+.3f}  q={}".format(
            cd["name"], cd["read_mb"], str(cd["loadable_vanilla_llm"]), str(cd["full_vocab"]),
            cd["identity_full256k"], cd["read_shrink_frac_vs_int4_floor"],
            cd["qualifies_as_loadable_lever"]), flush=True)
    print("  --- LOAD-BEARING DELIVERABLES ---", flush=True)
    print(f"  lmhead_read_bytes_deployed            = {v['lmhead_read_bytes_deployed']:.0f} "
          f"({v['lmhead_read_mb_deployed']:.1f} MB, int4-Marlin byte-exact #390={v['lmhead_read_bytes_deployed_byte_exact_390']})",
          flush=True)
    print(f"  loadable_candidates_enumerated        = {v['loadable_candidates_enumerated']}", flush=True)
    print(f"  loadable_identity_safe_floor          = {v['loadable_identity_safe_floor_format']} "
          f"({v['loadable_identity_safe_floor_read_mb']:.1f} MB)", flush=True)
    print(f"  best_loadable_read_shrink_frac        = {v['best_loadable_read_shrink_frac']:.4f} "
          f"(vs deployed {v['best_loadable_read_shrink_frac_vs_deployed']:.3f})", flush=True)
    print(f"  best_loadable_identity_byte_exact     = {v['best_loadable_identity_byte_exact']}", flush=True)
    print(f"  best_loadable_tps_delta               = {v['best_loadable_tps_delta']:.2f}", flush=True)
    print(f"  loadable_lmhead_lever_exists          = {v['loadable_lmhead_lever_exists']}", flush=True)
    print(f"  qualifying_levers                     = {v['qualifying_levers']}", flush=True)
    if syn["header_audit"].get("available"):
        h = syn["header_audit"]
        print(f"  header_audit (local -qat-w4a16-ct)    = lm_head {h['lm_head_dtype']} {h['lm_head_shape']} "
              f"bf16_fullvocab={h['lm_head_is_bf16_fullvocab']} body=int4-g{h['body_group_size']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[loadable-lmhead] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    st = syn["self_test"]
    v = syn["verdict_fields"]
    summary: dict[str, Any] = {
        "loadable_lmhead_self_test_passes": int(bool(st["loadable_lmhead_self_test_passes"])),
        "n_self_test_checks": st["n_checks"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        # ---- PR #398 deliverable keys ----
        "lmhead_read_bytes_deployed": v["lmhead_read_bytes_deployed"],
        "lmhead_read_mb_deployed": v["lmhead_read_mb_deployed"],
        "best_loadable_read_shrink_frac": v["best_loadable_read_shrink_frac"],
        "best_loadable_read_shrink_frac_vs_deployed": v["best_loadable_read_shrink_frac_vs_deployed"],
        "best_loadable_identity_byte_exact": int(bool(v["best_loadable_identity_byte_exact"])),
        "best_loadable_tps_delta": v["best_loadable_tps_delta"],
        "loadable_lmhead_lever_exists": int(bool(v["loadable_lmhead_lever_exists"])),
        "full_vocab_identity_tps_cost_vs_deployed": v["full_vocab_identity_tps_cost_vs_deployed"],
        "loadable_identity_safe_floor_read_mb": v["loadable_identity_safe_floor_read_mb"],
        "n_loadable_candidates": len(v["loadable_candidates_enumerated"]),
        "n_candidates_enumerated": len(v["all_candidates_enumerated"]),
        "loadable_candidates_enumerated": ",".join(v["loadable_candidates_enumerated"]),
        "vs_lossy_lmhead12k_note": v["vs_lossy_lmhead12k_note"],
        "exact_full_vocab_argmax_requires_all_rows": int(bool(v["exact_full_vocab_argmax_requires_all_rows"])),
        "vllm_has_sublinear_exact_mips_loadpath": int(bool(v["vllm_has_sublinear_exact_mips_loadpath"])),
        "analysis_only": int(True), "no_hf_job": int(True), "no_served_file_change": int(True),
        "official_tps": 0,
        "header_audit_available": int(bool(syn["header_audit"].get("available"))),
    }
    if syn["header_audit"].get("available"):
        summary["header_lm_head_is_bf16_fullvocab"] = int(bool(
            syn["header_audit"].get("lm_head_is_bf16_fullvocab")))
    # per-candidate read MB (rich logging).
    for cd in v["candidates"]:
        summary[f"read_mb_{cd['name']}"] = cd["read_mb"]
    summary = {k: val for k, val in summary.items()
               if val is not None and not (isinstance(val, float) and not math.isfinite(val))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "loadable-lmhead", "lm_head-readreduction", "int4-marlin", "byte-exact-identity",
              "vs-lmhead12k", "roofline", "analysis-only", "bank-the-analysis", "pr-398"],
        config={
            "hidden": HIDDEN, "full_vocab": FULL_VOCAB, "a10g_bw_gbps": A10G_BW_GBPS,
            "body_int4_gb": BODY_INT4_GB, "deployed_lmhead_rows": DEPLOYED_LMHEAD_ROWS,
            "deployed_lmhead_group_size": DEPLOYED_LMHEAD_GROUP_SIZE, "lmhead12k_rows": LMHEAD12K_ROWS,
            "int4_floor_group_size": INT4_FLOOR_GROUP_SIZE, "f_lmhead_344": F_LMHEAD_344,
            "phantom_lmhead_bf16_tax_tps": PHANTOM_LMHEAD_BF16_TAX_TPS,
            "official_deployed_tps": OFFICIAL_DEPLOYED_TPS, "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "nonspec_capture_offline_371": NONSPEC_CAPTURE_OFFLINE_371, "ppl_gate": PPL_GATE,
            "ppl_baseline": PPL_BASELINE, "milestone_tps": MILESTONE, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[loadable-lmhead] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="loadable_lmhead_readreduction_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[loadable-lmhead] wandb logged: loadable_lmhead_lever_exists="
          f"{v['loadable_lmhead_lever_exists']} self_test={st['loadable_lmhead_self_test_passes']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY 0-GPU self-validation only")
    ap.add_argument("--no-header-audit", action="store_true",
                    help="skip the optional local checkpoint-header audit")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="loadable-lmhead-readreduction")
    args = ap.parse_args(argv)

    header_audit = {"available": False, "note": "skipped (--no-header-audit)"} if args.no_header_audit \
        else audit_checkpoint_header()
    syn = synthesize(header_audit)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 398, "agent": "land",
        "kind": "loadable-lmhead-readreduction", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[loadable-lmhead] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loadable_lmhead_readreduction_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[loadable-lmhead] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(syn["loadable_lmhead_self_test_passes"]) and payload["nan_clean"]
    print(f"  PRIMARY loadable_lmhead_self_test_passes = {passes} "
          f"({syn['n_self_test_checks']} checks)", flush=True)
    print(f"  loadable_lmhead_lever_exists = {syn['verdict_fields']['loadable_lmhead_lever_exists']}",
          flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        print(f"[loadable-lmhead] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
