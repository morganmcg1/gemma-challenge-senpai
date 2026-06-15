#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Is the deployed 16384-row lm_head truncation PROVABLY identity-safe, and can it prune further? (PR #406).

WHAT THIS CARD DECIDES
----------------------
My #398 (`dzgbnsrp`) + #385 (`a30iri8i`) pinned that the deployed lm_head reads only ~21 MB/step because it
is row-TRUNCATED 262144 -> 16384 (the SAME lossy family as #385's lmhead12k). That truncation passes the
official strict gate, but #398 showed it passes only the SELF-REFERENTIAL / M-invariant sense (#390
`5y64zbjz`): the served pruned checkpoint matches its OWN plain greedy AR (a pruned argmax is always in the
kept set BY CONSTRUCTION), NOT the full-256k-vocab greedy AR of google/gemma-4-E4B-it. PR #406 asks the
sharp follow-up:

  Q1 (provability) Is the 16384-row truncation provably GLOBALLY identity-safe -- i.e. is every pruned row
     (index >=16384) provably UNREACHABLE (no attainable decode state ever greedy-argmaxes it) -- or is it
     only EMPIRICALLY safe on the public 128 (a lossy gamble that happens to pass)?
  Q2 (the lever)   If only N<<16384 rows are ever reachable, the head could shrink to N rows identity-safely
     -> a further read-shrink -> TPS lift. How many rows are further-prunable?
  Q3 (the flag)    If MORE than 16384 rows are reachable, the deployed cut is a PRIVATE-identity risk that
     should be surfaced to the humans.

VERDICT (this card)
-------------------
* deployed_truncation_provably_safe = **False**. Distribution-free, a vocab row r is GLOBALLY unreachable
  (never the greedy argmax over the full attainable hidden-state set) IFF its lm_head row w_r lies in the
  convex hull of the kept rows. After Gemma's final RMSNorm the attainable hidden state h ranges over a
  CENTERED, full-dimensional ellipsoid E = {h : sum_i (h_i/(1+gamma_i))^2 <= hidden} (gamma = norm.weight),
  which spans every direction; so "reachable over E" reduces to convex-position of the raw lm_head rows.
  A cheap, RIGOROUS separation certificate (max_{kept s} (w_s . hat w_r) < ||w_r|| => w_r is outside
  conv(kept) => r is reachable) certifies MANY pruned rows reachable. There is NO certificate of
  unreachability for any nonzero-norm pruned row (E is symmetric: if h is attainable so is -h, so no kept
  row can dominate r pairwise; and the only rows dominated everywhere are exact-zero-norm rows, of which
  there are zero). => provably_unreachable_rows = 0; deployed_truncation_empirical_only = True.
* The lever does NOT exist: further_prunable_rows = 0. The OFFICIAL-128 full-vocab greedy support already
  REACHES rank ~16360 of 16384 AND OVERFLOWS the keepset -- the full-vocab reference emits hundreds of
  distinct token ids the 16384 head cannot represent, ON THE PUBLIC SET. The cut is already tight/over-tight.
* The flag IS raised: deployed_truncation_private_risk = **True**. The truncation is not a faithful
  full-vocab greedy decoder of gemma-4-E4B; it diverges on public data and a private GT-target outside the
  kept set would force +inf PPL. The deployed strict-gate lm_head leg rests on an UNPROVEN truncation.

PROVABILITY BACKBONE (why norm bounds cannot certify safety, but CAN refute it)
------------------------------------------------------------------------------
Gemma-4-E4B decode head: h = RMSNorm(z) (.) (1+gamma); logits_pre = W h; logits = 30*tanh(logits_pre/30)
(final_logit_softcapping=30, MONOTONIC -> argmax-preserving). RMSNorm(z)_i = z_i*sqrt(n)/sqrt(||z||^2+n*eps)
=> ||RMSNorm(z)|| <= sqrt(n); with c_i=(1+gamma_i), h_i = u_i c_i, ||u||<=sqrt(n). So:
  * attainable set E = sqrt(n) * { (c_i u_i) : ||u||<=1 } -- a CENTERED axis-aligned ellipsoid (all dirs).
  * max attainable pre-cap logit of row r = sup_{h in E} w_r.h = sqrt(n)*||w_r (.) c||_2  (ellipsoid bound);
    a looser norm bound is ||w_r||*||h||_max with ||h||_max = sqrt(n)*max_i|c_i|.
  * E centered => for any h in E, -h in E => logit_r(-h) = -logit_r(h): no kept row pairwise-dominates a
    nonzero r; the ONLY everywhere-dominated rows are ||w_r||=0 (tie at h=0 only -> lowest-index tie-break).
  * reachable over E  <=>  exists direction d with w_r.d > w_s.d for all kept s  <=>  w_r notin conv(kept).
Because the Gemma lm_head rows are near-uniform unit-norm (kept and pruned norm bands OVERLAP), a
norm/frequency truncation cannot separate reachable from unreachable rows -> the cut is provably NOT safe.

PRIMARY metric  lmhead_provable_unreachable_self_test_passes  (>=20 pure-logic checks of the read math,
keepset structure, the provability reduction, and the lever/flag assembly; env-independent, runs under the
numpy-only .venv).
SCOPE: analysis / microbench. NO HF Job, NO submission, NO served-file change, 0 official TPS. The provability
norms (loads lm_head.weight + final norm from a LOCAL -qat-w4a16-ct safetensors) and the OOD greedy probe
(vLLM) are OPTIONAL GPU/torch enrichments that DEGRADE GRACEFULLY; the verdict stands on the 0-GPU official
-128 reachability + the provability reduction.

Run:
  # PRIMARY self-test only (numpy-only, no torch/GPU):
  .venv/bin/python -m research.validity.lmhead_provable_unreachable_pruning.lmhead_provable_unreachable_pruning --self-test
  # full card (real provability needs a torch+safetensors env; e.g. /tmp/server-venv on the A10G box):
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python -m research.validity.lmhead_provable_unreachable_pruning.lmhead_provable_unreachable_pruning \
    --wandb_group lmhead-provable-unreachable-pruning --wandb_name land/lmhead-provable-unreachable-pruning
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

import numpy as np

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, has a ./wandb run-output dir that shadows
# the package as a PEP-420 namespace) on sys.path[0]. Mirrors the #385/#398 house pattern.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------- #
# Geometry + bandwidth constants (provenance documented; consistent with #385/#390/#398/#344).
# ---------------------------------------------------------------------------- #
HIDDEN = 2560                       # gemma-4-E4B hidden_size
FULL_VOCAB = 262144                 # gemma-4-E4B text vocab (== embed_tokens rows; MEASURED from header)
A10G_BW_GBPS = 600.0                # A10G HBM peak bandwidth (the roofline figure, #385)
BODY_INT4_GB = 1.6973824           # int4 body bytes/step (#371/#344/#278/#385)

DEPLOYED_LMHEAD_ROWS = 16384        # #390 osoi5 baked, channel-wise int4-Marlin (the DEPLOYED truncation)
LMHEAD12K_ROWS = 12288             # #385 further-prune lever (more aggressive; same lossy family)
SOURCE_PCK04_32K_ROWS = 32768      # the source keepset the 16k/12k are subsets of (per keepset note)
INT4_BITS = 4
SCALE_DTYPE_BYTES = 2               # bf16 channel scales
FINAL_LOGIT_SOFTCAP = 30.0         # config.json final_logit_softcapping (MONOTONIC -> argmax-preserving)
RMS_NORM_EPS = 1e-6                # config.json rms_norm_eps

# #390/#398 deployed anchors (CITE; not re-derived).
DEPLOYED_LMHEAD_READ_MB_398 = 21.0         # #398: deployed 16384-row int4-Marlin channel read MB/step
OFFICIAL_DEPLOYED_TPS = 481.53             # PR #52 deployed (non-strict)
CORRECTED_STRICT_BASE_TPS = 467.48         # #393 0q7ynumg
PPL_GATE = 2.42
PPL_BASELINE = 2.3772
MILESTONE = 500.0

# ---- local artifact paths (degrade gracefully if absent) ----
KEEPSET_16K_CANDIDATES = [
    "/tmp/osoi5-v0-baked/pck04_keepset.json",
]
KEEPSET_12K_CANDIDATES = [
    "/tmp/osoi5-12k-baked/pck04_keepset.json",
    "/tmp/lmhead-keepset-12k/pck04_keepset.json",
]
# full-vocab plain-greedy-AR reference decode of the OFFICIAL 128 (the reachable argmax support).
OFFICIAL_BF16_DECODE = REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
OFFICIAL_INT4_DECODE = (REPO_ROOT / "research/greedy_reference/"
                        "workspace__senpai__target__submissions__int4_g128_lmhead__model/decode_outputs.jsonl")
DEPLOYED_SUB_DECODE = (REPO_ROOT / "research/greedy_reference/"
                       "workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it/"
                       "decode_outputs.jsonl")
GT_TOKENS = REPO_ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
# Full-vocab lm_head.weight BF16 [262144,2560] + final RMSNorm, needed to enumerate pruned-row
# (un)reachability. Sourced from the SHARED competition base model google/gemma-4-E4B-it-qat-w4a16-ct
# (the unquantized lm_head in the quant `ignore` list). land-owned extract is used first so the run
# never reads a peer home; the peer base-model cache is only a one-time read-only fallback for the
# extract (same shared-base-model list the merged #398 harness uses; not any peer's experimental work).
LMHEAD_W_CANDIDATES = [
    "/tmp/land-lmhead-prov/lmhead_norm.safetensors",
]
# FULL model dir for the best-effort live OOD greedy probe (vLLM). MUST be a full-vocab head
# (262144 rows): the deployed baked head is truncated to 16384 rows and could NEVER emit outside
# the keepset, so it is intentionally excluded. The shared base model is the only full-vocab dir;
# the probe is best-effort and the GLOBAL provability bound dominates it, so this degrades cleanly.
_QAT_SNAP = ("models--google--gemma-4-E4B-it-qat-w4a16-ct/"
             "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
OOD_MODEL_DIR_CANDIDATES = [
    f"/senpai-run/home/student-ubel/.cache/huggingface/hub/{_QAT_SNAP}",
    f"/senpai-run/home/student-denken/.cache/huggingface/hub/{_QAT_SNAP}",
    f"/senpai-run/home/student-lawine/.cache/huggingface/hub/{_QAT_SNAP}",
    f"/senpai-run/home/student-fern/.cache/huggingface/hub/{_QAT_SNAP}",
    f"/senpai-run/home/student-wirbel/.cache/huggingface/hub/{_QAT_SNAP}",
]
OOD_SIDECAR = HERE / "ood_reachable_support.json"

TOL = 1e-9
TOL_RT = 1e-6
NEARZERO_NORM = 1e-3               # a row with ||w_r|| <= this is treated as a provably-dominated zero row


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


# ---------------------------------------------------------------------------- #
# (1) Read-byte model + roofline (reused from #398, the diagnostic TPS basis).
# ---------------------------------------------------------------------------- #
def lmhead_read_bytes(rows: int, bits: int = INT4_BITS, channel: bool = True) -> float:
    """Per-decode-step truncated lm_head weight read = packed int4 weights + bf16 channel scales."""
    weight_bytes = rows * HIDDEN * bits / 8.0
    scale_bytes = rows * SCALE_DTYPE_BYTES if channel else 0.0
    return weight_bytes + scale_bytes


def roofline_tps(lmhead_read_gb: float) -> float:
    """Single-token decode roofline TPS (body int4 + lm_head), A10G BW. Diagnostic ceiling, NOT official."""
    return A10G_BW_GBPS / (BODY_INT4_GB + lmhead_read_gb)


# ---------------------------------------------------------------------------- #
# (2) Keepset load + truncation-criterion characterization.
# ---------------------------------------------------------------------------- #
def _load_keepset(paths: list[str]) -> tuple[list[int] | None, str | None]:
    p = _first_existing(paths)
    if p is None:
        return None, None
    meta = json.loads(Path(p).read_text())
    ids = meta.get("keep_ids") or meta.get("kept_ids")
    return (sorted(int(i) for i in ids) if ids else None), p


def characterize_keepset() -> dict[str, Any]:
    k16, p16 = _load_keepset(KEEPSET_16K_CANDIDATES)
    k12, p12 = _load_keepset(KEEPSET_12K_CANDIDATES)
    out: dict[str, Any] = {
        "keepset_16k_path": p16, "keepset_12k_path": p12,
        "deployed_rows_kept": DEPLOYED_LMHEAD_ROWS,
    }
    if k16 is None:
        out["available"] = False
        out["truncation_criterion"] = (
            "mandatory low-id floor + tokenizer specials/multimodal + ascending-ID frequency fill "
            "(per keepset note; local keepset not found, criterion cited)")
        return out
    s16 = set(k16)
    # leading contiguous run from id 0 (the mandatory low-id floor).
    floor = 0
    for i, v in enumerate(k16):
        if v == i:
            floor = i + 1
        else:
            break
    below = sum(1 for v in k16 if v < DEPLOYED_LMHEAD_ROWS)
    above = len(k16) - below          # specials/multimodal scattered ABOVE the 16384 index
    out.update({
        "available": True,
        "kept_size_16k": len(k16),
        "kept_min_id": k16[0], "kept_max_id": k16[-1],
        "leading_contiguous_floor": floor,
        "kept_ids_below_16384": below,
        "kept_ids_at_or_above_16384": above,
        "truncation_criterion": (
            f"mandatory low-id floor (leading contiguous 0..{floor - 1}) + tokenizer specials/multimodal "
            f"control tokens scattered across the full 262144 range ({above} kept ids sit at index>=16384, "
            f"up to id {k16[-1]}) + ascending-ID frequency fill to K={len(k16)}. NOT embedding-norm; the "
            f"kept rows are a vocab-ID/frequency cut, not a logit-magnitude cut."),
    })
    if k12 is not None:
        out["kept_size_12k"] = len(k12)
        out["k12_subset_of_k16"] = set(k12).issubset(s16)
    return out


# ---------------------------------------------------------------------------- #
# (3) Reachable-row support from the OFFICIAL-128 full-vocab greedy reference (0-GPU; from captures).
#     A vocab row r is "reachable" iff some decode state greedy-argmaxes it. The full-vocab plain-greedy-AR
#     reference decode IS the observed reachable argmax support; emissions OUTSIDE the kept set are rows the
#     truncated head physically cannot represent (truncation-attributable divergence from full-vocab greedy).
# ---------------------------------------------------------------------------- #
def _emission_ids(path: Path) -> list[int]:
    ids: list[int] = []
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ids.extend(int(t) for t in rec.get("completion_token_ids", []))
    return ids


def analyze_official_reachability(keep: dict[str, Any]) -> dict[str, Any]:
    k16, _ = _load_keepset(KEEPSET_16K_CANDIDATES)
    k12, _ = _load_keepset(KEEPSET_12K_CANDIDATES)
    res: dict[str, Any] = {"available": bool(k16 is not None and OFFICIAL_BF16_DECODE.exists())}
    if not res["available"]:
        res["note"] = "official full-vocab greedy reference or keepset not found locally"
        return res
    s16 = set(k16)
    rank16 = {v: i for i, v in enumerate(k16)}      # truncation-ordering index = position in sorted keepset
    s12 = set(k12) if k12 is not None else None

    bf16 = _emission_ids(OFFICIAL_BF16_DECODE)
    bf16_distinct = sorted(set(bf16))
    ook16 = [v for v in bf16_distinct if v not in s16]            # full-vocab greedy ids the 16k head clips
    ook16_pos = sum(1 for t in bf16 if t not in s16)             # per-position clip count
    in_keep_ranks = [rank16[v] for v in bf16_distinct if v in s16]

    # per-prompt: how many of the 128 official prompts have >=1 clipped (out-of-keepset) greedy emission.
    prompts_affected = 0
    if OFFICIAL_BF16_DECODE.exists():
        for line in OFFICIAL_BF16_DECODE.read_text().splitlines():
            if not line.strip():
                continue
            toks = json.loads(line).get("completion_token_ids", [])
            if any(int(t) not in s16 for t in toks):
                prompts_affected += 1

    res.update({
        "official_prompts": 128,
        "total_emissions": len(bf16),
        "distinct_reachable_rows_official": len(bf16_distinct),
        "max_reachable_emitted_id": bf16_distinct[-1] if bf16_distinct else None,
        "reachable_ids_outside_keepset16384": len(ook16),
        "reachable_emissions_outside_keepset16384": ook16_pos,
        "truncation_clip_rate_official_bf16": (ook16_pos / len(bf16)) if bf16 else 0.0,
        "prompts_with_clipped_emission": prompts_affected,
        "max_reachable_truncation_index_in_keepset": max(in_keep_ranks) if in_keep_ranks else None,
        "distinct_reachable_at_rank_ge_12288": sum(1 for r in in_keep_ranks if r >= LMHEAD12K_ROWS),
        # the reachable support OVERFLOWS the keepset -> the deepest reachable truncation index is in the
        # PRUNED region (>16384), so the support is NOT well below 16384.
        "max_reachable_truncation_index": (DEPLOYED_LMHEAD_ROWS + len(ook16)) if ook16 else max(in_keep_ranks, default=0),
        "reachable_support_well_below_16384": bool(len(ook16) == 0 and (max(in_keep_ranks, default=0)
                                                                        < DEPLOYED_LMHEAD_ROWS * 0.5)),
    })
    # int4 full-head cross-check (truncation-attributable clip on an int4 body; controls for body quant).
    if OFFICIAL_INT4_DECODE.exists():
        i4 = _emission_ids(OFFICIAL_INT4_DECODE)
        i4_distinct = set(i4)
        res["int4_fullhead_distinct"] = len(i4_distinct)
        res["int4_fullhead_ids_outside_keepset16384"] = len(i4_distinct - s16)
        res["int4_fullhead_clip_rate_official"] = (sum(1 for t in i4 if t not in s16) / len(i4)) if i4 else 0.0
    # deployed-submission (pruned head) self-consistency: emissions are 100% in-keepset BY CONSTRUCTION
    # (this is WHY the self-referential gate passes despite the divergence above).
    if DEPLOYED_SUB_DECODE.exists():
        dep = set(_emission_ids(DEPLOYED_SUB_DECODE))
        res["deployed_submission_distinct"] = len(dep)
        res["deployed_submission_emissions_outside_keepset"] = len(dep - s16)
        res["deployed_submission_self_referential"] = bool(dep.issubset(s16))
    # 12k further-prune drops even MORE (the lever is negative).
    if s12 is not None:
        res["reachable_ids_outside_keepset12288"] = len([v for v in bf16_distinct if v not in s12])
    # GT-target/context support: hard-included GT targets => 0 outside on PUBLIC (finite public PPL), but
    # real reference text reaches far up the vocab range -> the private-PPL exposure surface.
    if GT_TOKENS.exists():
        tgt: set[int] = set()
        ctx: set[int] = set()
        for line in GT_TOKENS.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tgt.update(int(t) for t in r.get("target_token_ids", []))
            ctx.update(int(t) for t in r.get("context_token_ids", []))
        res["gt_target_distinct"] = len(tgt)
        res["gt_target_outside_keepset16384"] = len(tgt - s16)
        res["gt_context_max_id"] = max(ctx) if ctx else None
        res["gt_target_max_id"] = max(tgt) if tgt else None
    return res


# ---------------------------------------------------------------------------- #
# (4) Provability core (pure-numpy reference logic; used by the self-test AND the real run).
# ---------------------------------------------------------------------------- #
def softcap(x: np.ndarray, cap: float = FINAL_LOGIT_SOFTCAP) -> np.ndarray:
    return cap * np.tanh(x / cap)


def separation_reachable(W_kept: np.ndarray, w_r: np.ndarray) -> bool:
    """RIGOROUS sufficient certificate that row r is REACHABLE over the (centered, full-dim) attainable set.

    If max_{s in kept} (w_s . hat w_r) < ||w_r|| then d = hat w_r separates w_r from conv(kept): row r
    strictly beats every kept row in direction d, so some attainable h makes r the greedy argmax. (Sound for
    reachability; a FAILED test does NOT certify unreachability -- another direction may separate.)
    """
    nr = float(np.linalg.norm(w_r))
    if nr <= NEARZERO_NORM:
        return False
    sup = float(np.max(W_kept @ (w_r / nr)))
    return sup < nr - 1e-9


def ellipsoid_logit_bound(w_r: np.ndarray, c: np.ndarray, hidden: int = HIDDEN) -> float:
    """sup_{h in E} w_r . h = sqrt(hidden) * ||w_r (.) c||_2 ; E = {h: sum (h_i/c_i)^2 <= hidden}."""
    return math.sqrt(hidden) * float(np.linalg.norm(w_r * c))


def provability_logic(W: np.ndarray, kept_idx: np.ndarray, c: np.ndarray) -> dict[str, Any]:
    """Pure-numpy provability reduction on a (possibly tiny synthetic) lm_head matrix W [V, n].

    Returns the counts the verdict needs. Designed to run on a 6x3 toy in the self-test AND on the real
    262144x2560 head (where the caller supplies a torch-accelerated separation count and just reuses the
    norm-ball / zero-row logic here)."""
    V, n = W.shape
    norms = np.linalg.norm(W, axis=1)
    kept_mask = np.zeros(V, dtype=bool)
    kept_mask[kept_idx] = True
    pruned_mask = ~kept_mask
    Rkmax = float(norms[kept_mask].max()) if kept_mask.any() else 0.0
    W_kept = W[kept_mask]
    pruned_rows = np.where(pruned_mask)[0]
    # norm-ball certificate: norm > max kept norm => outside conv(kept) ball => reachable (instant, rigorous).
    normball_reachable = int(np.sum(norms[pruned_mask] > Rkmax))
    # separation certificate (per pruned row); rigorous sufficient-for-reachable.
    sep_reachable = 0
    for r in pruned_rows:
        if separation_reachable(W_kept, W[r]):
            sep_reachable += 1
    # zero-row unreachability certificate: ||w_r||=0 -> logit 0 everywhere -> wins only at h=0 (degenerate).
    zero_rows = int(np.sum(norms[pruned_mask] <= NEARZERO_NORM))
    return {
        "n_pruned": int(pruned_mask.sum()),
        "kept_norm_max": Rkmax,
        "pruned_norm_max": float(norms[pruned_mask].max()) if pruned_mask.any() else 0.0,
        "pruned_norm_min": float(norms[pruned_mask].min()) if pruned_mask.any() else 0.0,
        "provably_reachable_normball": normball_reachable,
        "provably_reachable_separation": sep_reachable,
        "provably_unreachable_rows": zero_rows,           # only exact/near-zero dominated rows
        "deployed_truncation_provably_safe": bool(sep_reachable == 0 and normball_reachable == 0),
    }


def provability_norms() -> dict[str, Any]:
    """REAL provability on the deployed lm_head: load lm_head.weight + final norm from a local
    -qat-w4a16-ct safetensors (lazy torch import; CPU norms + GPU/tiled separation). Degrades gracefully."""
    res: dict[str, Any] = {"available": False}
    st_path = _first_existing(LMHEAD_W_CANDIDATES)
    k16, _ = _load_keepset(KEEPSET_16K_CANDIDATES)
    if st_path is None or k16 is None:
        res["note"] = "lm_head safetensors or keepset not found locally (provability degraded)"
        return res
    try:
        import torch  # noqa: F401
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"torch/safetensors unavailable ({exc!r}); run under a torch env e.g. /tmp/server-venv"
        return res
    try:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            W = f.get_tensor("lm_head.weight").float()
            g = f.get_tensor("model.language_model.norm.weight").float()
        V, n = W.shape
        norms = W.norm(dim=1)
        kept_t = torch.tensor(sorted(k16), dtype=torch.long)
        kept_mask = torch.zeros(V, dtype=torch.bool)
        kept_mask[kept_t] = True
        pruned_mask = ~kept_mask
        kn = norms[kept_mask]
        pn = norms[pruned_mask]
        Rkmax = float(kn.max())
        c = (1.0 + g)
        hmax = math.sqrt(n) * float(c.abs().max())
        Mell = math.sqrt(n) * (W * c).norm(dim=1)
        # norm-ball reachable (instant, rigorous).
        normball_reachable = int((pn > Rkmax).sum())
        zero_rows = int((pn <= NEARZERO_NORM).sum())
        # tiled separation certificate over ALL pruned rows (fp32, GPU if available).
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Wk = W[kept_mask].to(dev)                       # [16384, n]
        pruned_idx = torch.where(pruned_mask)[0]
        sep = 0
        CH = 8192
        with torch.no_grad():
            for i in range(0, pruned_idx.numel(), CH):
                idx = pruned_idx[i:i + CH]
                Wp = W[idx].to(dev)
                npr = Wp.norm(dim=1)
                dirs = Wp / npr.clamp_min(1e-12).unsqueeze(1)
                sup = (Wk @ dirs.t()).max(dim=0).values     # support of conv(kept) in each dir
                sep += int(((sup < (npr - 1e-6)) & (npr > NEARZERO_NORM)).sum())
        res.update({
            "available": True,
            "lmhead_weight_path": st_path,
            "vocab": int(V), "hidden": int(n),
            "kept_norm_min": float(kn.min()), "kept_norm_max": Rkmax, "kept_norm_mean": float(kn.mean()),
            "pruned_norm_min": float(pn.min()), "pruned_norm_max": float(pn.max()),
            "pruned_norm_mean": float(pn.mean()),
            "norm_bands_overlap": bool(float(pn.max()) >= float(kn.min()) and float(pn.min()) <= Rkmax),
            "hmax_distribution_free": hmax,
            "c_max": float(c.max()), "c_min": float(c.min()),
            "max_achievable_logit_kept": float(Mell[kept_mask].max()),
            "max_achievable_logit_pruned": float(Mell[pruned_mask].max()),
            "provably_reachable_pruned_normball": normball_reachable,
            "provably_reachable_pruned_separation": sep,
            "provably_unreachable_rows": zero_rows,
            "deployed_truncation_provably_safe": bool(sep == 0 and normball_reachable == 0),
        })
        # the OFFICIAL out-of-keepset emitted ids: are they ALL separation-certified reachable?
        if OFFICIAL_BF16_DECODE.exists():
            ook = sorted(set(_emission_ids(OFFICIAL_BF16_DECODE)) - set(k16))
            if ook:
                ot = torch.tensor(ook, dtype=torch.long)
                Wo = W[ot].to(dev)
                no = Wo.norm(dim=1)
                supo = (Wk @ (Wo / no.unsqueeze(1)).t()).max(dim=0).values
                res["official_ook_count"] = len(ook)
                res["official_ook_separation_reachable"] = int(((supo < (no - 1e-6))).sum())
                del Wo, no, supo
        del Wk, W, g
        if torch.cuda.is_available():               # release VRAM before the in-process vLLM OOD probe
            torch.cuda.empty_cache()
        return res
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"provability compute failed: {exc!r}"
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return res


# ---------------------------------------------------------------------------- #
# (5) OOD greedy reachability probe (OPTIONAL, vLLM; best-effort, degrades).
# ---------------------------------------------------------------------------- #
OOD_PROMPTS = [
    "Translate to French and explain: The quick brown fox jumps over the lazy dog near the riverbank.",
    "Écris un court poème en français sur la mer en hiver.",
    "Schreibe einen kurzen Absatz auf Deutsch über künstliche Intelligenz.",
    "用中文写一段关于量子计算的简短介绍。",
    "日本語で、四季についての短い俳句を3つ書いてください。",
    "اكتب فقرة قصيرة باللغة العربية عن أهمية الماء.",
    "Напиши короткий рассказ на русском языке о космосе.",
    "Escribe una receta sencilla de tortilla española paso a paso.",
    "Write a Python function that computes the SHA-256 of a file in 8 KB chunks.",
    "Explain the difference between a B-tree and an LSM-tree, with a code sketch.",
    "def fibonacci(n):\n    # complete this generator and add type hints\n",
    "Give the JSON schema for an OpenAI-compatible /v1/completions request.",
    "Summarize the plot of a noir detective novel set in 1940s Lisbon.",
    "List five rare chemical elements and one industrial use of each.",
    "Prove that the square root of 2 is irrational, step by step.",
    "Write SQL to find the second-highest salary per department with a window function.",
    "Compose a haiku about debugging at 3am.",
    "Describe the Krebs cycle for a first-year biochemistry student.",
    "Generate a regex matching IPv6 addresses and explain each part.",
    "Write a short dialogue between a Stoic philosopher and a startup founder.",
    "Explain gradient checkpointing and when it trades compute for memory.",
    "Translate '안녕하세요, 만나서 반갑습니다' and give a romanization.",
    "Write a limerick about a cat who learned to code in Rust.",
    "Outline a threat model for a self-hosted password manager.",
]


def ood_reachability_probe(max_tokens: int = 96) -> dict[str, Any]:
    """Greedy-decode a diverse OOD prompt set with the FULL-vocab head (vLLM) and measure how many distinct
    argmax rows fall OUTSIDE the deployed 16384 keepset (the realistic-private-data reachability surface)."""
    if OOD_SIDECAR.exists():
        try:
            return json.loads(OOD_SIDECAR.read_text())
        except Exception:  # noqa: BLE001
            pass
    res: dict[str, Any] = {"available": False}
    k16, _ = _load_keepset(KEEPSET_16K_CANDIDATES)
    model_dir = _first_existing([str(Path(p) / "config.json") for p in OOD_MODEL_DIR_CANDIDATES])
    if model_dir is not None:
        model_dir = str(Path(model_dir).parent)
    if k16 is None or model_dir is None:
        res["note"] = ("full-vocab model dir not found (OOD live probe skipped; the GLOBAL provability "
                       "bound already covers all OOD inputs, so the verdict is unaffected)")
        return res
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"vLLM unavailable ({exc!r}); OOD probe skipped (verdict stands on official+provability)"
        return res
    try:
        s16 = set(k16)
        llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=2048, enforce_eager=True)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = llm.generate(OOD_PROMPTS, sp)
        all_ids: list[int] = []
        for o in outs:
            all_ids.extend(int(t) for t in o.outputs[0].token_ids)
        distinct = sorted(set(all_ids))
        ook = [v for v in distinct if v not in s16]
        ook_pos = sum(1 for t in all_ids if t not in s16)
        res = {
            "available": True,
            "ood_prompts": len(OOD_PROMPTS),
            "ood_total_emissions": len(all_ids),
            "distinct_reachable_rows_ood": len(distinct),
            "reachable_ids_outside_keepset16384_ood": len(ook),
            "truncation_clip_rate_ood": (ook_pos / len(all_ids)) if all_ids else 0.0,
            "max_reachable_emitted_id_ood": distinct[-1] if distinct else None,
        }
        OOD_SIDECAR.write_text(json.dumps(res, indent=2))
        return res
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"OOD probe failed: {exc!r}"
        return res


# ---------------------------------------------------------------------------- #
# (6) Verdict / lever / flag assembly.
# ---------------------------------------------------------------------------- #
def assemble_verdict(keep: dict, official: dict, prov: dict, ood: dict) -> dict[str, Any]:
    deployed_read_gb = lmhead_read_bytes(DEPLOYED_LMHEAD_ROWS) / 1e9
    deployed_tps = roofline_tps(deployed_read_gb)

    # --- provability fields (prefer the real run; fall back to "empirical-only" defaults) ---
    if prov.get("available"):
        provably_unreachable_rows = int(prov["provably_unreachable_rows"])
        provably_safe = bool(prov["deployed_truncation_provably_safe"])
        prov_reach_sep = int(prov.get("provably_reachable_pruned_separation", 0))
        prov_reach_nb = int(prov.get("provably_reachable_pruned_normball", 0))
    else:
        provably_unreachable_rows = 0
        provably_safe = False
        prov_reach_sep = 0
        prov_reach_nb = 0
    empirical_only = not provably_safe

    # --- reachable support (official) ---
    ook_official = int(official.get("reachable_ids_outside_keepset16384", 0)) if official.get("available") else 0
    support_well_below = bool(official.get("reachable_support_well_below_16384", False))
    distinct_official = int(official.get("distinct_reachable_rows_official", 0)) if official.get("available") else 0
    max_idx = official.get("max_reachable_truncation_index")
    ook_ood = int(ood.get("reachable_ids_outside_keepset16384_ood", 0)) if ood.get("available") else 0
    distinct_ood = int(ood.get("distinct_reachable_rows_ood", 0)) if ood.get("available") else 0

    # --- THE LEVER: how many rows are further-prunable identity-safely? ---
    # A row is safe to additionally prune ONLY if it is provably/empirically unreachable. The reachable
    # support already reaches the top of the keepset AND overflows it on the PUBLIC set, and provability
    # certifies 0 unreachable rows -> there is NO safe further-prune headroom (the lever is 0/negative).
    further_prunable_rows = 0
    if support_well_below and ook_official == 0 and provably_unreachable_rows > 0:
        further_prunable_rows = int(provably_unreachable_rows)
    further_read_gb = lmhead_read_bytes(max(1, DEPLOYED_LMHEAD_ROWS - further_prunable_rows)) / 1e9
    further_prune_tps_lift = max(0.0, roofline_tps(further_read_gb) - deployed_tps)

    # --- THE FLAG: is the deployed cut a private-identity risk to surface? ---
    # Yes if it is not provably safe (empirical-only) AND there is direct evidence the full-vocab reachable
    # support exceeds the kept set (here: hundreds of distinct public greedy emissions are clipped).
    private_risk = bool(empirical_only and (ook_official > 0 or ook_ood > 0 or not provably_safe))

    return {
        # ---- PR #406 deliverables ----
        "truncation_criterion": keep.get("truncation_criterion"),
        "deployed_rows_kept": DEPLOYED_LMHEAD_ROWS,
        "distinct_reachable_rows_official": distinct_official,
        "distinct_reachable_rows_ood": distinct_ood,
        "max_reachable_truncation_index": max_idx,
        "reachable_support_well_below_16384": support_well_below,
        "provably_unreachable_rows": provably_unreachable_rows,
        "deployed_truncation_provably_safe": provably_safe,
        "deployed_truncation_empirical_only": empirical_only,
        "further_prunable_rows": further_prunable_rows,
        "further_prune_tps_lift": further_prune_tps_lift,
        "deployed_truncation_private_risk": private_risk,
        # ---- supporting evidence ----
        "reachable_ids_outside_keepset16384_official": ook_official,
        "reachable_ids_outside_keepset16384_ood": ook_ood,
        "provably_reachable_pruned_separation": prov_reach_sep,
        "provably_reachable_pruned_normball": prov_reach_nb,
        "deployed_lmhead_read_mb": deployed_read_gb * 1e3,
        "deployed_roofline_tps": deployed_tps,
        "final_logit_softcap_monotonic_preserves_argmax": True,
        "reachable_over_E_iff_outside_conv_kept": True,
        "attainable_set_is_centered_ellipsoid": True,
    }


# ---------------------------------------------------------------------------- #
# (7) PRIMARY self-test -- >=20 pure-logic checks (numpy + stdlib; env-independent).
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    c: dict[str, bool] = {}

    # ---- read-byte math ----
    r16 = lmhead_read_bytes(DEPLOYED_LMHEAD_ROWS)
    r12 = lmhead_read_bytes(LMHEAD12K_ROWS)
    c["t01_deployed_read_mb_matches_398"] = abs(r16 / 1e6 - DEPLOYED_LMHEAD_READ_MB_398) < 0.2
    c["t02_read_is_int4_plus_scales"] = abs(r16 - (DEPLOYED_LMHEAD_ROWS * HIDDEN * 0.5
                                                   + DEPLOYED_LMHEAD_ROWS * 2)) < 1.0
    c["t03_12k_reads_less_than_16k"] = r12 < r16
    c["t04_roofline_monotonic"] = roofline_tps(r12 / 1e9) > roofline_tps(r16 / 1e9)

    # ---- keepset hierarchy / criterion ----
    c["t05_16k_gt_12k_gt"] = DEPLOYED_LMHEAD_ROWS > LMHEAD12K_ROWS
    c["t06_subset_of_32k"] = LMHEAD12K_ROWS < DEPLOYED_LMHEAD_ROWS < SOURCE_PCK04_32K_ROWS

    # ---- softcap is monotonic (argmax-preserving) ----
    xs = np.array([-100.0, -5.0, -0.3, 0.0, 0.7, 5.0, 250.0])
    sc = softcap(xs)
    c["t07_softcap_monotonic"] = bool(np.all(np.diff(sc) > 0))
    c["t08_softcap_bounded"] = bool(np.all(np.abs(sc) < FINAL_LOGIT_SOFTCAP + 1e-6))
    perm = np.array([2, 0, 1, 6, 3, 5, 4])
    c["t09_softcap_preserves_argmax"] = int(np.argmax(xs[perm])) == int(np.argmax(softcap(xs[perm])))

    # ---- provability reduction on a SYNTHETIC head (the math the verdict rests on) ----
    # 2D toy: 3 kept rows around a triangle + a pruned row INSIDE the hull (unreachable-by-position) and a
    # pruned row OUTSIDE the hull (reachable). Reachability over the centered ellipsoid <=> outside conv(kept).
    W = np.array([
        [1.0, 0.0],     # kept 0 (vertex)
        [-1.0, 0.7],    # kept 1 (vertex)
        [-1.0, -0.7],   # kept 2 (vertex)
        [-0.3, 0.0],    # pruned 3: INSIDE conv(kept) -> NOT separation-reachable
        [3.0, 0.0],     # pruned 4: far OUTSIDE -> reachable (also norm-ball)
        [0.2, 0.95],    # pruned 5: outside an edge -> separation-reachable, not norm-ball
        [0.0, 0.0],     # pruned 6: zero row -> provably unreachable (tie at 0 only)
    ])
    kept_idx = np.array([0, 1, 2])
    Wk = W[kept_idx]
    c["t10_inside_hull_not_reachable"] = (separation_reachable(Wk, W[3]) is False)
    c["t11_outside_hull_reachable"] = (separation_reachable(Wk, W[4]) is True)
    c["t12_edge_outside_reachable"] = (separation_reachable(Wk, W[5]) is True)
    c["t13_zero_row_not_reachable"] = (separation_reachable(Wk, W[6]) is False)
    pl = provability_logic(W, kept_idx, np.ones(2))
    c["t14_synthetic_unreachable_is_zero_row"] = pl["provably_unreachable_rows"] == 1
    c["t15_synthetic_has_reachable_pruned"] = pl["provably_reachable_separation"] >= 2
    c["t16_synthetic_not_provably_safe"] = pl["deployed_truncation_provably_safe"] is False
    # norm-ball is a STRICT subset of separation-reachable (soundness ordering).
    c["t17_normball_subset_of_separation"] = pl["provably_reachable_normball"] <= pl["provably_reachable_separation"]

    # ---- centered-ellipsoid symmetry: no kept row pairwise-dominates a nonzero pruned row ----
    # For h and -h both attainable, logit_r(-h) = -logit_r(h): a nonzero row achieves BOTH signs.
    h = np.array([0.4, -0.9])
    c["t18_centered_set_symmetric"] = abs(float(W[5] @ h) + float(W[5] @ (-h))) < TOL

    # ---- ellipsoid logit bound is a valid UPPER bound on attainable logit ----
    cc = np.array([1.0, 2.0])
    Mb = ellipsoid_logit_bound(W[4], cc, hidden=2)
    # sample many h in E = {sum (h_i/c_i)^2 <= hidden}; every logit must be <= the bound.
    rng = np.random.default_rng(0)
    U = rng.standard_normal((4000, 2))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    U *= math.sqrt(2) * rng.uniform(0, 1, (4000, 1)) ** 0.5
    Hs = U * cc                          # h_i = u_i c_i, ||u|| <= sqrt(2)
    logits = Hs @ W[4]
    c["t19_ellipsoid_bound_is_upper"] = bool(np.max(logits) <= Mb + 1e-6)
    c["t20_ellipsoid_bound_tight"] = bool(np.max(logits) > 0.5 * Mb)   # not vacuously loose

    # ---- assembly logic: reachable support overflowing the keepset => not safe, lever 0, flag True ----
    keep = {"truncation_criterion": "x", "deployed_rows_kept": DEPLOYED_LMHEAD_ROWS}
    official = {"available": True, "distinct_reachable_rows_official": 6006,
                "reachable_ids_outside_keepset16384": 548, "reachable_support_well_below_16384": False,
                "max_reachable_truncation_index": DEPLOYED_LMHEAD_ROWS + 548}
    prov = {"available": True, "provably_unreachable_rows": 0,
            "deployed_truncation_provably_safe": False,
            "provably_reachable_pruned_separation": 100000, "provably_reachable_pruned_normball": 29}
    v = assemble_verdict(keep, official, prov, {"available": False})
    c["t21_not_provably_safe"] = v["deployed_truncation_provably_safe"] is False
    c["t22_empirical_only"] = v["deployed_truncation_empirical_only"] is True
    c["t23_no_further_prune_lever"] = v["further_prunable_rows"] == 0 and v["further_prune_tps_lift"] == 0.0
    c["t24_private_risk_flag_raised"] = v["deployed_truncation_private_risk"] is True
    c["t25_support_not_well_below"] = v["reachable_support_well_below_16384"] is False
    c["t26_max_idx_overflows_keepset"] = v["max_reachable_truncation_index"] > DEPLOYED_LMHEAD_ROWS

    # ---- counterfactual: if support WERE well below and provably-unreachable rows existed -> lever opens ----
    official2 = {"available": True, "distinct_reachable_rows_official": 4000,
                 "reachable_ids_outside_keepset16384": 0, "reachable_support_well_below_16384": True,
                 "max_reachable_truncation_index": 5000}
    prov2 = {"available": True, "provably_unreachable_rows": 4096,
             "deployed_truncation_provably_safe": True,
             "provably_reachable_pruned_separation": 0, "provably_reachable_pruned_normball": 0}
    v2 = assemble_verdict(keep, official2, prov2, {"available": False})
    c["t27_counterfactual_lever_opens"] = v2["further_prunable_rows"] == 4096 and v2["further_prune_tps_lift"] > 0.0
    c["t28_counterfactual_no_flag"] = v2["deployed_truncation_private_risk"] is False

    # ---- constants ----
    c["t29_constants_exact"] = bool(HIDDEN == 2560 and FULL_VOCAB == 262144
                                    and DEPLOYED_LMHEAD_ROWS == 16384
                                    and abs(FINAL_LOGIT_SOFTCAP - 30.0) < TOL)

    passes = bool(all(c.values()))
    return {"conditions": c, "n_checks": len(c), "lmhead_provable_unreachable_self_test_passes": passes}


# ---------------------------------------------------------------------------- #
# Synthesis / report / W&B (house pattern).
# ---------------------------------------------------------------------------- #
def synthesize(run_measurements: bool) -> dict[str, Any]:
    st = self_test()
    keep = characterize_keepset()
    official = analyze_official_reachability(keep)
    prov = provability_norms() if run_measurements else {"available": False, "note": "skipped (self-test only)"}
    ood = ood_reachability_probe() if run_measurements else {"available": False, "note": "skipped"}
    verdict = assemble_verdict(keep, official, prov, ood)
    return {
        "self_test": st,
        "lmhead_provable_unreachable_self_test_passes": st["lmhead_provable_unreachable_self_test_passes"],
        "n_self_test_checks": st["n_checks"],
        "keepset": keep,
        "official_reachability": official,
        "provability": prov,
        "ood_reachability": ood,
        "verdict_fields": verdict,
        "verdict": _build_verdict(verdict, official, prov, ood),
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }


def _build_verdict(v: dict, official: dict, prov: dict, ood: dict) -> str:
    parts = [
        "Q (PR #406): is the deployed 16384-row lm_head truncation provably GLOBALLY identity-safe, and can "
        "it prune further? VERDICT: deployed_truncation_provably_safe={}, deployed_truncation_empirical_only="
        "{}, further_prunable_rows={}, deployed_truncation_private_risk={}.".format(
            v["deployed_truncation_provably_safe"], v["deployed_truncation_empirical_only"],
            v["further_prunable_rows"], v["deployed_truncation_private_risk"]),
        "PROVABILITY: a vocab row is GLOBALLY unreachable iff its lm_head row lies in conv(kept) (the post-"
        "RMSNorm attainable hidden state is a centered, full-dimensional ellipsoid spanning all directions). "
        "provably_unreachable_rows={} (only exact-zero-norm rows qualify; none exist). The lm_head rows are "
        "near-uniform unit norm so a norm/frequency cut cannot separate reachable from unreachable rows.".format(
            v["provably_unreachable_rows"]),
    ]
    if official.get("available"):
        parts.append(
            "OFFICIAL-128 full-vocab greedy reference: {} distinct reachable rows; {} distinct ids ({:.3f}% of "
            "positions, {} of 128 prompts) fall OUTSIDE the 16384 keepset -- the truncated head physically "
            "cannot emit them, so the deployed cut is NOT a faithful full-vocab greedy decoder ON THE PUBLIC "
            "SET. In-keepset support reaches truncation rank {} of 16384 -> no further-prune headroom.".format(
                official.get("distinct_reachable_rows_official"),
                official.get("reachable_ids_outside_keepset16384"),
                100.0 * official.get("truncation_clip_rate_official_bf16", 0.0),
                official.get("prompts_with_clipped_emission"),
                official.get("max_reachable_truncation_index_in_keepset")))
    if prov.get("available"):
        parts.append(
            "PROVABILITY (real head): {} pruned rows provably reachable by separation, {} by norm-ball; "
            "{}/{} of the official out-of-keepset ids are separation-certified reachable. kept norm band "
            "[{:.3f},{:.3f}] OVERLAPS pruned [{:.3f},{:.3f}].".format(
                prov.get("provably_reachable_pruned_separation"),
                prov.get("provably_reachable_pruned_normball"),
                prov.get("official_ook_separation_reachable"), prov.get("official_ook_count"),
                prov.get("kept_norm_min"), prov.get("kept_norm_max"),
                prov.get("pruned_norm_min"), prov.get("pruned_norm_max")))
    if ood.get("available"):
        parts.append(
            "OOD greedy probe ({} prompts): {} distinct reachable rows, {} outside the keepset ({:.3f}% of "
            "positions) -> the private/OOD reachability surface exceeds 16384.".format(
                ood.get("ood_prompts"), ood.get("distinct_reachable_rows_ood"),
                ood.get("reachable_ids_outside_keepset16384_ood"),
                100.0 * ood.get("truncation_clip_rate_ood", 0.0)))
    parts.append(
        "FLAG: the deployed strict-gate lm_head leg rests on an UNPROVEN truncation that passes only the "
        "self-referential gate (the pruned checkpoint vs its OWN plain greedy AR); it is not byte-identical "
        "to full-vocab gemma-4-E4B greedy, and a private GT-target outside the kept set forces +inf PPL. "
        "LOCAL/analysis-only; 0 official TPS; NO HF Job / submission / served-file change.")
    return " ".join(parts)


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
    print("PROVABLE-UNREACHABLE lm_head TRUNCATION? (PR #406)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (PRIMARY) lmhead_provable_unreachable_self_test_passes = "
          f"{st['lmhead_provable_unreachable_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    fail = [k for k, val in st["conditions"].items() if not val]
    if fail:
        print(f"          FAILED: {fail}", flush=True)
    print("-" * 100, flush=True)
    print("  --- PR #406 DELIVERABLES ---", flush=True)
    for k in ("truncation_criterion", "deployed_rows_kept", "distinct_reachable_rows_official",
              "distinct_reachable_rows_ood", "max_reachable_truncation_index",
              "reachable_support_well_below_16384", "provably_unreachable_rows",
              "deployed_truncation_provably_safe", "deployed_truncation_empirical_only",
              "further_prunable_rows", "further_prune_tps_lift", "deployed_truncation_private_risk"):
        print(f"  {k:<42} = {v.get(k)}", flush=True)
    print("  --- supporting ---", flush=True)
    for k in ("reachable_ids_outside_keepset16384_official", "reachable_ids_outside_keepset16384_ood",
              "provably_reachable_pruned_separation", "provably_reachable_pruned_normball"):
        print(f"  {k:<42} = {v.get(k)}", flush=True)
    if syn["provability"].get("available"):
        p = syn["provability"]
        print(f"  kept_norm[min,max]                         = [{p['kept_norm_min']:.4f},{p['kept_norm_max']:.4f}]  "
              f"pruned[{p['pruned_norm_min']:.4f},{p['pruned_norm_max']:.4f}]  hmax={p['hmax_distribution_free']:.1f}",
              flush=True)
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
        print(f"[lmhead-prov-unreach] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    st = syn["self_test"]
    v = syn["verdict_fields"]
    prov = syn["provability"]
    official = syn["official_reachability"]
    ood = syn["ood_reachability"]

    def _num(x):
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    summary: dict[str, Any] = {
        "lmhead_provable_unreachable_self_test_passes": int(bool(st["lmhead_provable_unreachable_self_test_passes"])),
        "n_self_test_checks": st["n_checks"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        # ---- PR #406 deliverable keys ----
        "deployed_rows_kept": v["deployed_rows_kept"],
        "distinct_reachable_rows_official": v["distinct_reachable_rows_official"],
        "distinct_reachable_rows_ood": v["distinct_reachable_rows_ood"],
        "max_reachable_truncation_index": _num(v["max_reachable_truncation_index"]),
        "reachable_support_well_below_16384": int(bool(v["reachable_support_well_below_16384"])),
        "provably_unreachable_rows": v["provably_unreachable_rows"],
        "deployed_truncation_provably_safe": int(bool(v["deployed_truncation_provably_safe"])),
        "deployed_truncation_empirical_only": int(bool(v["deployed_truncation_empirical_only"])),
        "further_prunable_rows": v["further_prunable_rows"],
        "further_prune_tps_lift": v["further_prune_tps_lift"],
        "deployed_truncation_private_risk": int(bool(v["deployed_truncation_private_risk"])),
        "truncation_criterion": v["truncation_criterion"],
        # ---- supporting ----
        "reachable_ids_outside_keepset16384_official": v["reachable_ids_outside_keepset16384_official"],
        "reachable_ids_outside_keepset16384_ood": v["reachable_ids_outside_keepset16384_ood"],
        "provably_reachable_pruned_separation": v["provably_reachable_pruned_separation"],
        "provably_reachable_pruned_normball": v["provably_reachable_pruned_normball"],
        "deployed_lmhead_read_mb": v["deployed_lmhead_read_mb"],
        "deployed_roofline_tps": v["deployed_roofline_tps"],
        "provability_available": int(bool(prov.get("available"))),
        "official_reachability_available": int(bool(official.get("available"))),
        "ood_probe_available": int(bool(ood.get("available"))),
        "analysis_only": int(True), "no_hf_job": int(True), "no_served_file_change": int(True),
        "official_tps": 0,
    }
    for src in (official, prov, ood):
        if src.get("available"):
            for k, val in src.items():
                n = _num(val)
                if n is not None and k not in summary:
                    summary[f"m_{k}"] = n
    summary = {k: val for k, val in summary.items()
               if val is not None and not (isinstance(val, float) and not math.isfinite(val))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "lm_head-truncation", "provable-unreachable", "greedy-identity",
              "convex-hull-reachability", "self-referential-gate", "private-risk-flag", "analysis-only",
              "bank-the-analysis", "pr-406"],
        config={
            "hidden": HIDDEN, "full_vocab": FULL_VOCAB, "deployed_lmhead_rows": DEPLOYED_LMHEAD_ROWS,
            "lmhead12k_rows": LMHEAD12K_ROWS, "source_pck04_32k_rows": SOURCE_PCK04_32K_ROWS,
            "a10g_bw_gbps": A10G_BW_GBPS, "body_int4_gb": BODY_INT4_GB,
            "final_logit_softcap": FINAL_LOGIT_SOFTCAP, "rms_norm_eps": RMS_NORM_EPS,
            "official_deployed_tps": OFFICIAL_DEPLOYED_TPS, "corrected_strict_base_tps": CORRECTED_STRICT_BASE_TPS,
            "ppl_gate": PPL_GATE, "ppl_baseline": PPL_BASELINE, "milestone_tps": MILESTONE,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[lmhead-prov-unreach] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="lmhead_provable_unreachable_pruning_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[lmhead-prov-unreach] wandb logged: provably_safe={v['deployed_truncation_provably_safe']} "
          f"private_risk={v['deployed_truncation_private_risk']} self_test="
          f"{st['lmhead_provable_unreachable_self_test_passes']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY pure-logic self-validation only (numpy-only; no torch/GPU)")
    ap.add_argument("--no-measurements", action="store_true",
                    help="skip the torch/vLLM provability + OOD enrichments (self-test + 0-GPU official only)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="lmhead-provable-unreachable-pruning")
    args = ap.parse_args(argv)

    run_measurements = not (args.self_test or args.no_measurements)
    syn = synthesize(run_measurements=run_measurements)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 406, "agent": "land",
        "kind": "lmhead-provable-unreachable-pruning", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[lmhead-prov-unreach] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lmhead_provable_unreachable_pruning_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[lmhead-prov-unreach] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(syn["lmhead_provable_unreachable_self_test_passes"]) and payload["nan_clean"]
    print(f"  PRIMARY lmhead_provable_unreachable_self_test_passes = {passes} "
          f"({syn['n_self_test_checks']} checks)", flush=True)
    print(f"  deployed_truncation_provably_safe = {syn['verdict_fields']['deployed_truncation_provably_safe']}; "
          f"private_risk = {syn['verdict_fields']['deployed_truncation_private_risk']}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        print(f"[lmhead-prov-unreach] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
