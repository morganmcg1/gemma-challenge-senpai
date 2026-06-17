#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #552 -- Provably-lossless head-prune: a smaller EXACT head -- does it beat lawine's +38?

WHAT THIS CARD DECIDES
----------------------
lawine #544 (d44b61gj) proved the 262k-head verify-tax is 82.2% of the
base_fullhead(252.31) -> osoi5(350.76) single-stream gap, and that the only
identity-preserving head lever it priced -- a LOWER-PRECISION full head (int4) --
recovers +38.3 -> 292.1 (ceiling; magically-free head tops at 328.9). This card
prices the SECOND identity-preserving head lever: a SMALLER-but-EXACT head. Instead
of keeping all 262,144 rows at lower precision, PRUNE the rows that are provably
never the argmax -> a head with fewer rows but bit-identical fp16 argmax.

Two tiers, reported SEPARATELY (never conflated):

  STRICT / PROVABLE (Stage 1-2, PRIMARY).  A row is provably-never-argmax iff its
  lm_head row w_r lies in conv(kept) in the c-weighted metric (c = 1+gamma, gamma =
  final RMSNorm gain). After Gemma's final RMSNorm the attainable hidden state h
  ranges over a CENTERED, full-dimensional ellipsoid E = {h : sum_i (h_i/c_i)^2 <=
  hidden} that spans EVERY direction; E centered => for any attainable h, -h is also
  attainable => no kept row pairwise-dominates a nonzero-norm row => the ONLY rows
  provably never the argmax are exact-zero-norm rows. The Gemma lm_head has none
  (near-uniform unit-norm rows). => provably_unreachable_rows = 0. A provable prune
  keeps ALL 262,144 rows -> +0 TPS. This is the #406 result, re-measured on the
  served base head, and it preserves #319 strict identity by construction (trivially:
  it prunes nothing). DECISIVE NO-GO that HARDENS lawine #544: the full head really
  is irreducible at strict identity; precision (int4/fp8) is the only head lever.

  EMPIRICAL (Stage 3, SECONDARY, identity-risk flagged).  A larger
  *empirically*-never-argmax prune (rows never observed as the full-vocab greedy
  argmax on a held-out corpus, + frequency margin) buys real TPS but is NOT strict.
  The deployed 16,384-row frequency cut already CLIPS the full-vocab argmax on
  0.0102 of official-128 / 0.092 of a 274-prompt held-out corpus / 0.152 of OOD
  positions -> argmax-flip rate > 0 -> FAILS #319. To drive the held-out flip to ~0
  you must keep ~261,976 rows (#414) -> +~0 TPS. The empirical TPS/identity tradeoff
  is a CLIFF: the rows you can prune cheaply are reachable; the rows that are safe to
  prune are ~none.

METHOD (reconcile against lawine #544; do NOT re-derive the head/body split).
  TPS is priced through lawine's MEASURED single-stream operating point
  (b_et = E[T] = 3.8194 tok/step, b_tcyc = 15.138 ms, tps = et/(tcyc/1e3) = 252.306)
  by sweeping ONLY the verify-head matmul time. The head matmul is HBM-weight-read
  bound (lawine: argmax ~0.03 ms, the cost is the dense weight read), so its time is
  LINEAR in row count. We MICROBENCH the real bf16 head matmul at M=8 (K=7 spec
  verify) over a sweep of row counts on THIS A10G, and map the matmul delta onto
  lawine's t_cycle. Pruning to N rows: save_ms(N) = mm_bf16(262144) - mm_bf16(N);
  tps_pruned(N) = tps(b_et, b_tcyc - save_ms(N)). N_provable = 262144 -> save 0 ->
  252.31. The int4-full-head lever is reproduced as a pipeline cross-check (must land
  on lawine's +38.3). NB the eager single-seq census CANNOT price the head
  (body-dominated, [[fullserve-census-tps-dilution]]); this microbench+decomposition
  is the sanctioned pricing.

SCOPE: LOCAL analysis on the student A10G. analysis_only=true, official_tps=0.
NO HF Job, NO submission, NO served-file/leaderboard change. wandb_group
lossless-head-prune-tps. All weights/keepset/decode read from land-owned local
paths (no peer home reads).

Run:
  # Stage 1-3 GPU analysis (torch + safetensors + CUDA):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/lossless_head_prune_tps/lossless_head_prune_tps.py
  # 0-GPU wandb logging pass (wandb-capable python):
  /tmp/land-mb-venv/bin/python \
      research/validity/lossless_head_prune_tps/lossless_head_prune_tps.py --log-wandb
  # pure-logic self-test (numpy only):
  python3 research/validity/lossless_head_prune_tps/lossless_head_prune_tps.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------- #
# Geometry constants (gemma-4-E4B-it, MEASURED from the served qat head header).
# ---------------------------------------------------------------------------- #
HIDDEN = 2560
FULL_VOCAB = 262144
FINAL_LOGIT_SOFTCAP = 30.0          # config.json final_logit_softcapping (MONOTONIC -> argmax-preserving)
NEARZERO_NORM = 1e-3                # ||w_r|| <= this => a provably-dominated zero row (#406)
DEPLOYED_KEEPSET_ROWS = 16384       # osoi5-v0 baked frequency cut (the deployed empirical prune)

# ---- lawine #544 (d44b61gj) MEASURED single-stream operating point (CITE; never re-derived). ----
# serve_results.json arms.base_fullhead: median_wall_tps / e_t_steptime / t_cycle_ms.
LAWINE_B_ET = 3.8194082146962955    # E[T] = expected accepted tokens / spec step (K=7)
LAWINE_B_TCYC_MS = 15.137999999999998
LAWINE_BFH_TPS = 252.30599912117162  # tps(b_et, b_tcyc) == this (the full-262k base_fullhead FLOOR)
FERN_BFH_TPS = 253.78               # fern #535 whh42dgd anchor (the alternate floor)
LAWINE_FREE_HEAD_TPS = 328.9        # lawine: t_cycle minus the FULL in-serve head tax (3.524 ms)
LAWINE_INT4_CEILING = 292.1         # lawine: +38.3 int4-full-head lever
LAWINE_INT4_RECOVER = 38.3
LAWINE_FP8_RECOVER = 24.5
UNSAFE_FRONTIER_LOCAL = 442.0
TAU_LO = 1.03524                    # local->official transfer (#267)

# ---- land-owned local artifacts (no peer home reads). ----
_QAT_SNAP_DIR = (Path.home() / ".cache/huggingface/hub/"
                 "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots")
KEEPSET_16K = Path("/tmp/osoi5-v0-baked/pck04_keepset.json")
OFFICIAL_BF16_DECODE = REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
HELDOUT_SIDECAR = REPO_ROOT / "research/validity/truevocab_lmhead_equivalence_cost/heldout_reachable_support.json"
OOD_SIDECAR = REPO_ROOT / "research/validity/lmhead_provable_unreachable_pruning/ood_reachable_support.json"
OFFICIAL406_RESULTS = (REPO_ROOT / "research/validity/lmhead_provable_unreachable_pruning/"
                       "lmhead_provable_unreachable_pruning_results.json")

OUT_JSON = HERE / "lossless_head_prune_tps_results.json"

# Microbench row-count sweep: full/provable, deployed keepset, official-reachable (6006), deep prunes.
# (Powers-of-two + 6006 keep cuBLAS kernel selection stable; arbitrary non-aligned N like 245760 can
# mis-select a slower kernel than the full head -> a timing artifact, excluded.)
ROW_SWEEP = [262144, 131072, 65536, 32768, 16384, 12288, 8192, 6006, 4096, 2048, 1024, 512, 256]


# ---------------------------------------------------------------------------- #
# lawine's TPS model (EXACT; reused verbatim).
# ---------------------------------------------------------------------------- #
def tps(et: float, t_cycle_ms: float) -> float:
    return et / (t_cycle_ms / 1e3)


def tps_pruned(save_ms: float) -> float:
    """Served single-stream TPS if the verify-head matmul is made `save_ms` cheaper."""
    return tps(LAWINE_B_ET, LAWINE_B_TCYC_MS - save_ms)


def _peak_mem_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _resolve_qat_snapshot() -> Path:
    snaps = sorted(p for p in _QAT_SNAP_DIR.glob("*") if (p / "model.safetensors").exists())
    if not snaps:
        raise RuntimeError(f"no qat-w4a16-ct snapshot with model.safetensors under {_QAT_SNAP_DIR}")
    return snaps[0]


# ---------------------------------------------------------------------------- #
# Stage 1 -- PROVABLE prune count (strict-identity tier).
# ---------------------------------------------------------------------------- #
def stage1_provable_prune() -> dict[str, Any]:
    """Load the served bf16 lm_head + final RMSNorm gain; count provably-never-argmax rows.

    provably_unreachable_rows = #{ rows with ||w_r|| <= NEARZERO } (the only rows that are in
    conv(kept) for ANY kept set, because E is a centered full-dim ellipsoid). Also reproduces the
    #406 separation diagnostic: of the 245,760 rows the deployed 16k keepset drops, how many are
    POSITIVELY certified REACHABLE (=> NOT safe to prune)? A high count shows the reachable support
    is dense -- there is no separable never-argmax cluster to harvest."""
    import torch
    from safetensors import safe_open

    snap = _resolve_qat_snapshot()
    with safe_open(str(snap / "model.safetensors"), framework="pt", device="cpu") as f:
        W_bf16 = f.get_tensor("lm_head.weight")                      # [262144, 2560] bf16
        gamma = f.get_tensor("model.language_model.norm.weight").float()
    V, n = W_bf16.shape
    assert (V, n) == (FULL_VOCAB, HIDDEN), f"unexpected head shape {(V, n)}"

    Wf = W_bf16.float()
    norms = Wf.norm(dim=1)
    c = (1.0 + gamma)                                               # ellipsoid axis scales
    # max attainable pre-softcap logit of row r over E = sqrt(n) * || w_r (.) c ||_2 (ellipsoid bound)
    Mell = math.sqrt(n) * (Wf * c).norm(dim=1)

    zero_rows = int((norms <= NEARZERO_NORM).sum())                 # PROVABLY-never-argmax rows
    provable_keep_rows = V - zero_rows

    # ---- separation diagnostic vs the deployed 16k keepset (reproduce #406 245592) ----
    keep = json.loads(KEEPSET_16K.read_text())["keep_ids"]
    kept_t = torch.tensor(sorted(set(keep)), dtype=torch.long)
    kept_mask = torch.zeros(V, dtype=torch.bool)
    kept_mask[kept_t] = True
    pruned_mask = ~kept_mask
    kn = norms[kept_mask]
    pn = norms[pruned_mask]
    Rkmax = float(kn.max())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Wk = Wf[kept_mask].to(dev)                                      # [16384, 2560]
    pruned_idx = torch.where(pruned_mask)[0]
    normball = int((pn > Rkmax).sum())
    sep = 0
    CH = 8192
    with torch.no_grad():
        for i in range(0, pruned_idx.numel(), CH):
            idx = pruned_idx[i:i + CH]
            Wp = Wf[idx].to(dev)
            npr = Wp.norm(dim=1)
            dirs = Wp / npr.clamp_min(1e-12).unsqueeze(1)
            sup = (Wk @ dirs.t()).max(dim=0).values                # support of conv(kept) in dir hat(w_r)
            sep += int(((sup < (npr - 1e-6)) & (npr > NEARZERO_NORM)).sum())
            del Wp, npr, dirs, sup
    peak_gpu_mib = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    del Wk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "vocab": V, "hidden": n,
        "lmhead_weight_path": str(snap / "model.safetensors"),
        "lmhead_dtype": "bf16",
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "kept_norm_min": float(kn.min()), "kept_norm_max": Rkmax,
        "pruned_norm_min": float(pn.min()), "pruned_norm_max": float(pn.max()),
        "norm_bands_overlap": bool(float(pn.max()) >= float(kn.min()) and float(pn.min()) <= Rkmax),
        "c_min": float(c.min()), "c_max": float(c.max()),
        "max_achievable_logit_kept": float(Mell[kept_mask].max()),
        "max_achievable_logit_pruned": float(Mell[pruned_mask].max()),
        # --- the PRIMARY strict-tier numbers ---
        "provably_unreachable_rows": zero_rows,
        "provable_keep_rows": provable_keep_rows,
        "provable_prune_count": zero_rows,
        "provable_prune_frac": zero_rows / V,
        "strict_identity_preserved": True,   # provable prune keeps ALL rows -> argmax bit-identical by construction
        # --- density diagnostic: deployed-16k-pruned rows that are PROVABLY reachable (NOT safe) ---
        "deployed_pruned_rows": int(pruned_mask.sum()),
        "provably_reachable_pruned_separation": sep,
        "provably_reachable_pruned_normball": normball,
        "deployed_pruned_provably_reachable_frac": sep / int(pruned_mask.sum()),
        "peak_gpu_mib_stage1": peak_gpu_mib,
        "bound_basis": (
            "centered full-dimensional ellipsoid E={h: sum_i (h_i/(1+gamma_i))^2 <= hidden}; "
            "row r provably-never-argmax IFF w_r in conv(kept) in the (1+gamma)-weighted metric; "
            "E centered => no kept row pairwise-dominates a nonzero-norm row => only ||w_r||<=%g rows "
            "qualify; the gemma-4-E4B head has %d such rows." % (NEARZERO_NORM, zero_rows)
        ),
    }


# ---------------------------------------------------------------------------- #
# Stage 2 -- served single-stream TPS of the (provably-)pruned head via microbench.
# ---------------------------------------------------------------------------- #
def _time_ms(fn, *, warmup: int = 20, iters: int = 100) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def stage2_microbench_tps() -> dict[str, Any]:
    """Microbench the real bf16 head matmul at M=8 over ROW_SWEEP; price each via lawine's t_cycle.

    Only the row COUNT N affects matmul time (HBM weight-read bound) -> head[:N] is a valid timing
    proxy for any N-row keep-set. The provable point is N=262144 (save 0 -> 252.31, gain 0)."""
    import torch
    assert torch.cuda.is_available(), "CUDA required for the head microbench"
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    snap = _resolve_qat_snapshot()
    from safetensors import safe_open
    with safe_open(str(snap / "model.safetensors"), framework="pt", device="cpu") as f:
        head = f.get_tensor("lm_head.weight").to(torch.bfloat16).to(dev)   # [262144, 2560]
    V, H = head.shape
    M = 8  # K=7 spec verify -> M+? we bench the M=8 verify batch (lawine convention); also M=1 (pure AR)

    rows_meas: dict[str, dict[str, float]] = {}
    for N in ROW_SWEEP:
        Wt = head[:N].t().contiguous()                          # [H, N]
        rec = {}
        for m in (8, 1):
            x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
            logits = x @ Wt
            rec[f"matmul_bf16_ms_m{m}"] = _time_ms(lambda: x @ Wt)
            rec[f"head_bf16_ms_m{m}"] = _time_ms(lambda: (x @ Wt).argmax(dim=-1))
            rec[f"argmax_ms_m{m}"] = _time_ms(lambda: logits.argmax(dim=-1))
            del x, logits
        rows_meas[str(N)] = rec
        del Wt
    peak_gpu_mib = torch.cuda.max_memory_allocated() / 2**20

    # effective HBM bandwidth back-solved from the full-262k bf16 M=8 matmul (lawine's method).
    t_full8 = rows_meas[str(FULL_VOCAB)]["matmul_bf16_ms_m8"] / 1e3
    eff_hbm_gbps = (FULL_VOCAB * H * 2 / 1e9) / t_full8
    mm_full8 = rows_meas[str(FULL_VOCAB)]["matmul_bf16_ms_m8"]
    argmax_full8 = rows_meas[str(FULL_VOCAB)]["argmax_ms_m8"]

    curve = []
    for N in ROW_SWEEP:
        mmN = rows_meas[str(N)]["matmul_bf16_ms_m8"]
        save = mm_full8 - mmN                                   # bf16 weight-read saved by pruning to N
        t = tps_pruned(save)
        curve.append({
            "rows": N, "matmul_bf16_ms_m8": round(mmN, 4),
            "save_ms": round(save, 4),
            "served_tps_bf16_pruned": round(t, 2),
            "tps_gain_vs_floor": round(t - LAWINE_BFH_TPS, 2),
            "frac_of_252_328_bracket": round((t - LAWINE_BFH_TPS) / (LAWINE_FREE_HEAD_TPS - LAWINE_BFH_TPS), 3),
        })

    # PRIMARY provable point: N=262144 (prune 0).
    prov = next(r for r in curve if r["rows"] == FULL_VOCAB)
    # int4-full-head lever pipeline CROSS-CHECK (must reproduce lawine +38.3): replace bf16 read by int4.
    int4_mm_full8 = (FULL_VOCAB * H * 0.5 / 1e9) / eff_hbm_gbps * 1e3
    save_int4_full = mm_full8 - (int4_mm_full8 + argmax_full8)
    tps_int4_full = tps_pruned(save_int4_full)
    int4_recover_reproduced = tps_int4_full - LAWINE_BFH_TPS

    return {
        "M_verify": M, "eff_hbm_gbps": round(eff_hbm_gbps, 1),
        "matmul262k_bf16_ms_m8": round(mm_full8, 4),
        "argmax262k_ms_m8": round(argmax_full8, 4),
        "rows_meas": rows_meas,
        "tps_curve": curve,
        "provable_prune_served_tps": round(prov["served_tps_bf16_pruned"], 2),
        "provable_prune_tps_gain": round(prov["tps_gain_vs_floor"], 2),
        # cross-check: my pipeline reproduces lawine's int4-full lever
        "xcheck_int4_full_recover_tps": round(int4_recover_reproduced, 2),
        "xcheck_int4_full_ceiling_tps": round(tps_int4_full, 2),
        "xcheck_matches_lawine_38": bool(abs(int4_recover_reproduced - LAWINE_INT4_RECOVER) < 5.0),
        "peak_gpu_mib_stage2": peak_gpu_mib,
    }


# ---------------------------------------------------------------------------- #
# Stage 3 -- empirical-tier upside (held-out argmax-flip vs prune depth).
# ---------------------------------------------------------------------------- #
def _official_argmax_ids() -> list[list[int]]:
    """Per-prompt full-vocab bf16 greedy argmax id sequences from the official-128 decode reference."""
    seqs = []
    with open(OFFICIAL_BF16_DECODE) as f:
        for line in f:
            r = json.loads(line)
            ids = r.get("completion_token_ids")
            if ids:
                seqs.append(list(ids))
    return seqs


def stage3_empirical(stage2: dict[str, Any]) -> dict[str, Any]:
    """Empirical prune: keep-set = rows ever argmax on a corpus (+freq margin). Flip = argmax-flip
    vs full-vocab greedy on a HELD-OUT corpus. Two operating points + a measured flip-vs-K curve."""
    heldout = json.loads(HELDOUT_SIDECAR.read_text())
    ood = json.loads(OOD_SIDECAR.read_text())
    off406 = json.loads(OFFICIAL406_RESULTS.read_text())["synthesis"]["official_reachability"]
    curve = stage2["tps_curve"]
    eff_hbm = stage2["eff_hbm_gbps"]
    mm_full = stage2["matmul262k_bf16_ms_m8"]

    def tps_at(rows: int) -> float:
        # Within the measured sweep -> interpolate the measured bf16 curve (descending in rows).
        # Outside (e.g. flip-safe ~262k, between the coarse 131072..262144 segment) -> price the
        # weight-read delta analytically from the back-solved HBM bandwidth (exact near full-vocab).
        pts = sorted(((c["rows"], c["served_tps_bf16_pruned"]) for c in curve))
        rmin, rmax = pts[0][0], pts[-1][0]
        # analytic HBM-linear pricing for the near-full regime (the wide measured gap is inaccurate there)
        if rows > 131072:
            save = max(0.0, (FULL_VOCAB - rows) * HIDDEN * 2 / (eff_hbm * 1e9) * 1e3)
            return tps_pruned(min(save, mm_full))
        for i in range(len(pts) - 1):
            (r0, t0), (r1, t1) = pts[i], pts[i + 1]
            if r0 <= rows <= r1:
                w = (rows - r0) / (r1 - r0) if r1 > r0 else 0.0
                return t0 + w * (t1 - t0)
        return pts[-1][1] if rows >= rmax else pts[0][1]

    # ---- cross-corpus clip (= argmax-flip) of the FIXED deployed 16k keepset vs full-vocab greedy ----
    flip_official_bf16 = off406["truncation_clip_rate_official_bf16"]
    flip_heldout = heldout["held_out_clip_rate"]
    flip_ood = ood["truncation_clip_rate_ood"]

    # ---- measured flip-vs-prune-depth via an official-128 prompt-level train/test split ----
    seqs = _official_argmax_ids()
    n_prompts = len(seqs)
    half = n_prompts // 2
    from collections import Counter
    build_ctr: Counter = Counter()
    for s in seqs[:half]:
        build_ctr.update(s)
    test_positions = [tid for s in seqs[half:] for tid in s]
    n_test = len(test_positions)
    ranked = [tid for tid, _ in build_ctr.most_common()]            # build-corpus frequency order
    flip_vs_k = []
    for K in [256, 512, 1024, 2048, 4096, 6006, 8192, 12288, 16384, 32768, 65536]:
        keep = set(ranked[:K])
        flips = sum(1 for tid in test_positions if tid not in keep)
        flip_vs_k.append({"keep_rows_top_freq": K, "heldout_split_flip_rate": round(flips / n_test, 5),
                          "served_tps_bf16": round(tps_at(K), 2)})
    distinct_build = len(build_ctr)

    # ---- operating points ----
    # (A) deployed 16k frequency cut: material TPS, but flips on the heavy tail (NOT strict).
    kk = DEPLOYED_KEEPSET_ROWS
    tps_16k = tps_at(kk)
    op_material = {
        "empirical_keep_rows": kk,
        "empirical_prune_served_tps": round(tps_16k, 2),
        "empirical_prune_tps_gain": round(tps_16k - LAWINE_BFH_TPS, 2),
        "flip_official_bf16": round(flip_official_bf16, 5),
        "flip_heldout_274p": round(flip_heldout, 5),
        "flip_ood_24p": round(flip_ood, 5),
        "fails_319": bool(flip_heldout > 0.0),
        "note": "deployed osoi5-v0 16k frequency cut; the natural 'empirically-never-argmax+margin' head",
    }
    # (B) flip-safe: min keep-set for held-out flip ~0 (#414) -> +~0 TPS. The cliff.
    safe_rows = 261976  # #414 min_identity_safe_keepset_size (99.94% of vocab)
    tps_safe = tps_at(safe_rows)
    op_flipsafe = {
        "empirical_keep_rows_flipsafe": safe_rows,
        "empirical_flipsafe_served_tps": round(tps_safe, 2),
        "empirical_flipsafe_tps_gain": round(tps_safe - LAWINE_BFH_TPS, 2),
        "flip_heldout_274p": 0.0,
        "source": "truevocab #414 min_identity_safe_keepset_size (held-out flips -> 0 with margin)",
    }

    return {
        "heldout_corpus": {"prompts": heldout["heldout_prompts"], "positions": heldout["heldout_total_positions"],
                           "distinct_reachable_rows": heldout["distinct_reachable_rows_heldout"],
                           "distinct_ids_outside_16k_keepset": heldout["held_out_distinct_ids_clipped"],
                           "reference_model": "google/gemma-4-E4B-it (full bf16, full-vocab greedy)"},
        "official128_corpus": {"positions": off406["total_emissions"],
                               "distinct_reachable_rows": off406["distinct_reachable_rows_official"],
                               "ids_outside_16k_keepset": off406["reachable_ids_outside_keepset16384"]},
        "ood_corpus": {"prompts": ood["ood_prompts"], "positions": ood["ood_total_emissions"],
                       "distinct_reachable_rows": ood["distinct_reachable_rows_ood"]},
        "official_split_distinct_build_ids": distinct_build,
        "flip_vs_prune_depth_officialsplit": flip_vs_k,
        "operating_point_material_tps": op_material,
        "operating_point_flip_safe": op_flipsafe,
        # headline empirical fields (the deployed-16k material-TPS point)
        "empirical_keep_rows": op_material["empirical_keep_rows"],
        "empirical_prune_served_tps": op_material["empirical_prune_served_tps"],
        "empirical_prune_tps_gain": op_material["empirical_prune_tps_gain"],
        "empirical_argmax_flip_rate": op_material["flip_heldout_274p"],   # held-out (the PR-named corpus)
    }


# ---------------------------------------------------------------------------- #
# Verdict assembly.
# ---------------------------------------------------------------------------- #
def assemble(stage1: dict, stage2: dict, stage3: dict, self_det: bool) -> dict[str, Any]:
    prov_gain = stage2["provable_prune_tps_gain"]
    lossless_green = bool(prov_gain >= 5.0)        # "material TPS gain at strict identity"; sigma_hw~4.864
    beats_lawine_38 = bool(prov_gain > LAWINE_INT4_RECOVER)
    emp_gain = stage3["empirical_prune_tps_gain"]
    emp_flip = stage3["empirical_argmax_flip_rate"]
    verdict = (
        "STRICT/PROVABLE TIER (PRIMARY): provably_unreachable_rows={pur} -> provable_keep_rows={pkr} "
        "(prune_frac={pf:.4g}). The attainable post-RMSNorm hidden state is a CENTERED full-dimensional "
        "ellipsoid; only zero-norm rows are provably-never-argmax and the gemma-4-E4B head has none, so a "
        "strict-identity prune keeps ALL {V} rows and recovers +{pg} TPS (served {pt}, == the base_fullhead "
        "FLOOR, bottom of lawine's [{flo}..{fre}] bracket). #319 strict identity preserved BY CONSTRUCTION. "
        "lossless_head_lever_is_green={green}; beats_lawine_38={beat}. DECISIVE NO-GO that HARDENS lawine "
        "#544: the full head is irreducible at strict identity; precision (int4 +38.3 / fp8 +24.5) is the "
        "only head lever -- and even that is not byte-exact. "
        "EMPIRICAL TIER (SECONDARY): the deployed 16k frequency cut buys +{eg} TPS (served {et}) -- nominally "
        "> lawine's +38 -- but CLIPS the full-vocab greedy argmax on {fo:.4g} official / {eflip:.4g} held-out "
        "/ {food:.4g} OOD positions => argmax-flip > 0 => FAILS #319 (quality-safe-but-not-strict, and worse "
        "identity risk than int4's near-tie jitter: a prune systematically removes heavy-tail tokens). To "
        "drive held-out flip ~0 you keep ~261,976 rows -> +~0 TPS. The empirical TPS/identity tradeoff is a "
        "CLIFF: rows you can prune cheaply are reachable; rows safe to prune are ~none."
    ).format(
        pur=stage1["provably_unreachable_rows"], pkr=stage1["provable_keep_rows"],
        pf=stage1["provable_prune_frac"], V=stage1["vocab"], pg=prov_gain, pt=stage2["provable_prune_served_tps"],
        flo=LAWINE_BFH_TPS, fre=LAWINE_FREE_HEAD_TPS, green=lossless_green, beat=beats_lawine_38,
        eg=emp_gain, et=stage3["empirical_prune_served_tps"],
        fo=stage3["operating_point_material_tps"]["flip_official_bf16"], eflip=emp_flip,
        food=stage3["operating_point_material_tps"]["flip_ood_24p"],
    )
    return {
        "schema": "lossless_head_prune_tps_v1",
        "pr": 552, "agent": "land", "analysis_only": True, "official_tps": 0,
        "no_hf_job": True, "no_submission": True, "no_served_file_change": True,
        "wandb_group": "lossless-head-prune-tps",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # ---- KEY OUTPUTS ----
        "provable_keep_rows": stage1["provable_keep_rows"],
        "provable_prune_frac": stage1["provable_prune_frac"],
        "provable_prune_served_tps": stage2["provable_prune_served_tps"],
        "provable_prune_tps_gain": stage2["provable_prune_tps_gain"],
        "empirical_keep_rows": stage3["empirical_keep_rows"],
        "empirical_prune_served_tps": stage3["empirical_prune_served_tps"],
        "empirical_argmax_flip_rate": stage3["empirical_argmax_flip_rate"],
        "strict_identity_preserved": stage1["strict_identity_preserved"],
        "lossless_head_lever_is_green": lossless_green,
        "beats_lawine_38": beats_lawine_38,
        "self_det": self_det,
        "primary_metric": {"name": "provable_prune_served_tps", "value": stage2["provable_prune_served_tps"]},
        # ---- reconciliation anchors (lawine #544 d44b61gj) ----
        "lawine_floor_tps": LAWINE_BFH_TPS, "fern_floor_tps": FERN_BFH_TPS,
        "lawine_free_head_ceiling_tps": LAWINE_FREE_HEAD_TPS,
        "lawine_int4_ceiling_tps": LAWINE_INT4_CEILING, "lawine_int4_recover_tps": LAWINE_INT4_RECOVER,
        "xcheck_int4_full_recover_reproduced": stage2["xcheck_int4_full_recover_tps"],
        "xcheck_matches_lawine_38": stage2["xcheck_matches_lawine_38"],
        "peak_gpu_mib": round(max(stage1.get("peak_gpu_mib_stage1", 0.0),
                                  stage2.get("peak_gpu_mib_stage2", 0.0)), 1),
        "peak_mem_mib": round(_peak_mem_mib(), 1),
        "verdict": verdict,
        "stage1_provability": stage1,
        "stage2_microbench_tps": stage2,
        "stage3_empirical": stage3,
    }


# ---------------------------------------------------------------------------- #
# Self-test (pure-logic, numpy-free; validates the TPS model + verdict wiring).
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    c: dict[str, bool] = {}
    # lawine operating point reproduces the floor
    c["t01_tps_model_reproduces_floor"] = abs(tps(LAWINE_B_ET, LAWINE_B_TCYC_MS) - LAWINE_BFH_TPS) < 1e-6
    # pruning 0 rows -> 0 gain
    c["t02_zero_save_zero_gain"] = abs(tps_pruned(0.0) - LAWINE_BFH_TPS) < 1e-9
    # a positive save -> strictly more TPS (monotone)
    c["t03_save_monotone"] = tps_pruned(1.0) > tps_pruned(0.0)
    # the int4-full save (1.996 ms) reproduces lawine's +38.3 within tolerance
    g_int4 = tps_pruned(1.996) - LAWINE_BFH_TPS
    c["t04_int4_save_recovers_38"] = abs(g_int4 - LAWINE_INT4_RECOVER) < 1.5
    # removing the full microbench matmul (~2.698 ms) lands BELOW lawine's free-head 328.9 (overhead survives)
    c["t05_prune_ceiling_below_freehead"] = tps_pruned(2.698) < LAWINE_FREE_HEAD_TPS
    # verdict booleans for a 0-gain provable tier
    fake_s1 = {"provably_unreachable_rows": 0, "provable_keep_rows": 262144, "provable_prune_frac": 0.0,
               "vocab": 262144, "strict_identity_preserved": True}
    fake_s2 = {"provable_prune_served_tps": 252.31, "provable_prune_tps_gain": 0.0,
               "xcheck_int4_full_recover_tps": 38.3, "xcheck_matches_lawine_38": True}
    fake_s3 = {"empirical_keep_rows": 16384, "empirical_prune_served_tps": 301.6,
               "empirical_prune_tps_gain": 49.3, "empirical_argmax_flip_rate": 0.092,
               "operating_point_material_tps": {"flip_official_bf16": 0.0102, "flip_ood_24p": 0.152}}
    v = assemble(fake_s1, fake_s2, fake_s3, self_det=True)
    c["t06_provable_green_false_at_0_gain"] = (v["lossless_head_lever_is_green"] is False)
    c["t07_beats_lawine_false_at_0_gain"] = (v["beats_lawine_38"] is False)
    c["t08_strict_identity_true"] = (v["strict_identity_preserved"] is True)
    c["t09_primary_is_provable_tps"] = (v["primary_metric"]["name"] == "provable_prune_served_tps")
    # a hypothetical material provable gain WOULD flip green (guards against a stuck-False bug)
    fake_s2b = dict(fake_s2, provable_prune_tps_gain=40.0, provable_prune_served_tps=292.0)
    vb = assemble(fake_s1, fake_s2b, fake_s3, self_det=True)
    c["t10_green_true_if_material"] = (vb["lossless_head_lever_is_green"] is True)
    c["t11_beats_lawine_true_if_gt_38"] = (vb["beats_lawine_38"] is True)
    # empirical flip>0 marks fails_319
    c["t12_empirical_flip_positive"] = fake_s3["empirical_argmax_flip_rate"] > 0.0
    return {"lossless_head_prune_self_test_passes": all(c.values()), "n_checks": len(c), "conditions": c}


# ---------------------------------------------------------------------------- #
# wandb logging (run under a wandb-capable python; 0-GPU).
# ---------------------------------------------------------------------------- #
def log_wandb(results: dict[str, Any]) -> None:
    import wandb
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.wandb_logging import init_wandb_run, finish_wandb

    s2 = results["stage2_microbench_tps"]
    flat = {
        "analysis_only": 1, "official_tps": 0, "pr": 552,
        "provable_keep_rows": results["provable_keep_rows"],
        "provable_prune_frac": results["provable_prune_frac"],
        "provable_prune_served_tps": results["provable_prune_served_tps"],
        "provable_prune_tps_gain": results["provable_prune_tps_gain"],
        "empirical_keep_rows": results["empirical_keep_rows"],
        "empirical_prune_served_tps": results["empirical_prune_served_tps"],
        "empirical_prune_tps_gain": results["stage3_empirical"]["empirical_prune_tps_gain"],
        "empirical_argmax_flip_rate": results["empirical_argmax_flip_rate"],
        "empirical_flip_official_bf16": results["stage3_empirical"]["operating_point_material_tps"]["flip_official_bf16"],
        "empirical_flip_ood_24p": results["stage3_empirical"]["operating_point_material_tps"]["flip_ood_24p"],
        "strict_identity_preserved": int(results["strict_identity_preserved"]),
        "lossless_head_lever_is_green": int(results["lossless_head_lever_is_green"]),
        "beats_lawine_38": int(results["beats_lawine_38"]),
        "self_det": int(results["self_det"]),
        "provably_unreachable_rows": results["stage1_provability"]["provably_unreachable_rows"],
        "provably_reachable_pruned_separation": results["stage1_provability"]["provably_reachable_pruned_separation"],
        "deployed_pruned_provably_reachable_frac": results["stage1_provability"]["deployed_pruned_provably_reachable_frac"],
        "eff_hbm_gbps": s2["eff_hbm_gbps"],
        "matmul262k_bf16_ms_m8": s2["matmul262k_bf16_ms_m8"],
        "xcheck_int4_full_recover_reproduced": results["xcheck_int4_full_recover_reproduced"],
        "lawine_floor_tps": results["lawine_floor_tps"],
        "lawine_free_head_ceiling_tps": results["lawine_free_head_ceiling_tps"],
        "lawine_int4_recover_tps": results["lawine_int4_recover_tps"],
        "peak_gpu_mib": results["peak_gpu_mib"], "peak_mem_mib": results["peak_mem_mib"],
    }
    run = init_wandb_run(
        job_type="analysis", agent="land",
        name="land/lossless-head-prune-tps", group="lossless-head-prune-tps",
        notes="PR #552 provably-lossless head-prune vs lawine #544 int4 +38 lever (strict vs empirical tier)",
        tags=["pr552", "lossless-head-prune", "analysis-only", "tps-ceiling", "lmhead"],
        config={"pr": 552, "k_spec": 7, "num_prompts": 128, "output_len": 512,
                "lawine_b_et": LAWINE_B_ET, "lawine_b_tcyc_ms": LAWINE_B_TCYC_MS},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode) -- JSON artifact still written", flush=True)
        return
    # TPS-vs-prune-depth curve as a wandb Table + line plot
    try:
        cur = s2["tps_curve"]
        tbl = wandb.Table(columns=["rows", "served_tps_bf16_pruned", "tps_gain_vs_floor", "save_ms"])
        for r in cur:
            tbl.add_data(r["rows"], r["served_tps_bf16_pruned"], r["tps_gain_vs_floor"], r["save_ms"])
        run.log({"tps_vs_prune_depth": tbl})
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] table log skipped: {exc!r}", flush=True)
    run.log({**flat, "global_step": 0})
    run.summary.update(flat)
    print(f"[wandb] run = {run.id} ({run.url})", flush=True)
    finish_wandb(run)


# ---------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--log-wandb", action="store_true", help="0-GPU: read results JSON + log to wandb")
    args = ap.parse_args()

    if args.self_test:
        st = self_test()
        print(json.dumps(st, indent=2))
        return 0 if st["lossless_head_prune_self_test_passes"] else 1

    if args.log_wandb:
        results = json.loads(OUT_JSON.read_text())
        log_wandb(results)
        return 0

    # ---- GPU analysis: Stage 1 + Stage 2 + Stage 3 ----
    st = self_test()
    assert st["lossless_head_prune_self_test_passes"], f"self-test failed: {st['conditions']}"

    print("[stage1] provable prune count ...", flush=True)
    stage1 = stage1_provable_prune()
    print(f"[stage1] provably_unreachable_rows={stage1['provably_unreachable_rows']} "
          f"provable_keep_rows={stage1['provable_keep_rows']} "
          f"deployed-16k-pruned provably-reachable={stage1['provably_reachable_pruned_separation']}/"
          f"{stage1['deployed_pruned_rows']}", flush=True)

    print("[stage1-selfdet] re-running provability for determinism check ...", flush=True)
    stage1b = stage1_provable_prune()
    self_det = bool(stage1["provably_unreachable_rows"] == stage1b["provably_unreachable_rows"]
                    and stage1["provably_reachable_pruned_separation"] == stage1b["provably_reachable_pruned_separation"])
    print(f"[stage1-selfdet] self_det={self_det}", flush=True)

    print("[stage2] head matmul microbench + TPS pricing ...", flush=True)
    stage2 = stage2_microbench_tps()
    print(f"[stage2] provable_prune_served_tps={stage2['provable_prune_served_tps']} "
          f"gain={stage2['provable_prune_tps_gain']} | int4-xcheck +{stage2['xcheck_int4_full_recover_tps']} "
          f"(matches_lawine_38={stage2['xcheck_matches_lawine_38']})", flush=True)

    print("[stage3] empirical-tier flip vs prune depth ...", flush=True)
    stage3 = stage3_empirical(stage2)
    print(f"[stage3] empirical_keep_rows={stage3['empirical_keep_rows']} "
          f"tps={stage3['empirical_prune_served_tps']} (+{stage3['empirical_prune_tps_gain']}) "
          f"flip_heldout={stage3['empirical_argmax_flip_rate']} flip_ood="
          f"{stage3['operating_point_material_tps']['flip_ood_24p']}", flush=True)

    results = assemble(stage1, stage2, stage3, self_det)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {OUT_JSON}", flush=True)
    print("SENPAI-LOSSLESS-PRUNE " + json.dumps({
        "provable_keep_rows": results["provable_keep_rows"],
        "provable_prune_served_tps": results["provable_prune_served_tps"],
        "provable_prune_tps_gain": results["provable_prune_tps_gain"],
        "empirical_keep_rows": results["empirical_keep_rows"],
        "empirical_prune_served_tps": results["empirical_prune_served_tps"],
        "empirical_argmax_flip_rate": results["empirical_argmax_flip_rate"],
        "lossless_head_lever_is_green": results["lossless_head_lever_is_green"],
        "beats_lawine_38": results["beats_lawine_38"],
        "strict_identity_preserved": results["strict_identity_preserved"],
        "self_det": results["self_det"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
