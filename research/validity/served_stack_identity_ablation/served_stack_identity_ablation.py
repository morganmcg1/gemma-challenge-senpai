#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Served-stack identity ablation (PR #385) — synthesis + W&B card (0-GPU).

WHAT THIS LEG DECIDES
---------------------
#371 isolated two bundled non-spec levers on the byte-exact int4 M=1 decode: CUDA-graph
CAPTURE is byte-exact (identity 1.0, eager 21.62 -> 91.38) and inductor FUSION breaks it
(0.0703, +6%). It also corrected the anchor: 165.44 is the *served* strict frontier
(lawine #196, submission ``fa2sw_nonspec_int4``). This card answers the two open questions
#371 pointed at:

  Q1  What composes the offline-capture 91.38 -> served 165.44 gap, and is it byte-exact?
  Q2/Q3  Is the served 165.44 itself fully strict (does it pass a spec-off NON-self-referential
         greedy-identity gate), or does it bank identity-breaking / lossy levers that only pass
         the SELF-REFERENTIAL served gate (reference on its own engine+kernels, REFERENCE_MODE_ENV)?

GROUND TRUTH (read from the repo, not assumed)
----------------------------------------------
The 165.44 submission ``submissions/fa2sw_nonspec_int4/manifest.json`` is byte-identical to
the deployed ``fa2sw_precache_kenyan`` EXCEPT ``SPECULATIVE_CONFIG`` blanked. It banks, per the
manifest env: ``LM_HEAD_PRUNE=1`` (lmhead12k: int4 lm_head pruned 262144->12288 rows,
serve.py:569 ``_prune_lm_head_rows`` — LOSSY), native cudagraph + inductor **fusion** ON (no
``--enforce-eager``/``--compilation-config``), ``FA_SLIDING=1``, ``SPLITKV_VERIFY=1``,
``PRECACHE_BENCH=1``, ``ONEGRAPH=1``. So the PR's named ladder {precache, split-KV,
weight-residency} is NOT the real composition:
  * precache (``serve_patch_precache.py``)  = a PUBLIC-only prefix-cache warmup of the 128
    public bench prompts; SKIPS when the dataset is absent (private rerun); warms prefill, not
    the decode step -> ~0 decode-step TPS lift. MEASURED CAVEAT (S1): the engine flag it depends
    on (``enable_prefix_caching``; serve.py:1056 passes ``--prefix-caching-hash-algo``) is NOT
    identity-neutral vs the eager reference — turning it on diverges greedy output for ~124/128
    prompts (chunked/block-prefill numerics flip argmax, then cascade). The submission's own
    "cache hits return bit-equal KV" claim is cache-hit SELF-consistency, not equivalence to a
    non-cached reference; so precache passes only the SELF-REFERENTIAL gate, like fusion.
  * split-KV (``SPLITKV_VERIFY``, max_q=64) = a SPEC-VERIFY 3D split path; INERT on the M=1
    non-spec base (no verify steps with speculation blanked) -> 0.
  * weight-residency = not a standalone lever; a cudagraph side-effect already in capture.
The dominant 91->165 composer is **lmhead12k** (LOSSY) + **fusion** (identity-breaking), plus a
local->official bridge. ROOFLINE: offline full-vocab reads body+bf16-lmhead = 1.697+1.342 = 3.04
GB/step (ceiling 197); served int4-12k lm_head reads body+0.0157 = 1.71 GB/step (ceiling 350).
91.38 x (350/197) = 162 ~= served. lmhead12k read-reduction alone explains ~the whole gap, and it
is LOSSY (byte-exact only vs its OWN pruned checkpoint, not vs full-vocab plain AR).

THE MEASURED LADDER (offline, full-vocab int4, == #371 venv + eager-E reference)
--------------------------------------------------------------------------------
S0 capture / S1 +precache / S2 +split-KV / S3 +residency / S4 +fusion. Each Sk's
``token_identity_rate`` is computed vs the #371 eager (E) decode = the spec-off NON-fused
full-vocab plain greedy-AR reference (E = no capture, no fusion, no prune). The named
identity-safe levers (S1/S2/S3) are expected to add ~0 over S0 (capture) at identity 1.0; S4
(fusion) re-confirms the #371 break. This PROVES the 91->165 gap is not the named levers and is
not byte-exact; the real composers (lmhead12k, fusion) are attributed on the roofline + #371.

PRIMARY metric  served_stack_self_test_passes
TEST    metric  frac_of_91_to_165_identity_safe   (expected ~0: gap is lossy+identity-breaking)

LOCAL pod-A10G only; 0 official TPS; NO HF Job / submission / served-file change.

Run:
  # GPU legs first (one process per config; see measure_stack.py), under the SERVER venv:
  for cfg in S0 S1 S2 S3 S4; do CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python \
    research/validity/served_stack_identity_ablation/measure_stack.py --config $cfg \
    --out-dir research/validity/served_stack_identity_ablation/measured; done
  # then synthesize + log (0-GPU), under the .venv (has wandb):
  .venv/bin/python research/validity/served_stack_identity_ablation/served_stack_identity_ablation.py \
    --measured-dir research/validity/served_stack_identity_ablation/measured \
    --wandb_group nonspec-baselift-cudagraph --wandb_name land/served-stack-identity-ablation
  # 0-GPU PRIMARY self-validation (no measured data needed):
  .venv/bin/python research/validity/served_stack_identity_ablation/served_stack_identity_ablation.py --self-test
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

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, has a ./wandb run-output dir
# that shadows the package as a PEP-420 namespace) on sys.path[0]. Caches the installed package
# in sys.modules so the later `import wandb` returns the real module regardless of path order.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------- #
# Imported constants (provenance documented; NOT re-derived).
# ---------------------------------------------------------------------------- #
BODY_INT4_GB = 1.6973824            # int4 body bytes/step (#371/#344/#278)
A10G_BW_GBPS = 600.0                # A10G HBM peak bandwidth (the roofline figure)
VOCAB = 262144                      # gemma-4-E4B vocab (text_config.vocab_size)
HIDDEN = 2560                       # gemma-4-E4B hidden_size
LMHEAD_BF16_FULL_GB = VOCAB * HIDDEN * 2 / 1e9     # full bf16 tied lm_head read (1.342 GB), offline cfg
LMHEAD12K_ROWS = 12288              # served lmhead12k keep-count (serve.py _prune_lm_head_rows / PCK-04)
LMHEAD_INT4_12K_GB = LMHEAD12K_ROWS * HIDDEN * 0.5 / 1e9  # int4-packed 12k-row lm_head read (~0.0157 GB)

STRICT_FRONTIER_TPS = 165.44        # lawine #196 strict non-spec frontier (SERVED, official-comparable)
STRICT_FRONTIER_LOCAL_TPS = 156.05  # lawine #196 local wall_tps (-> 165.44 official via the bridge)
NONSPEC_CAPTURE_OFFLINE_371 = 91.38 # #371 C: offline full-vocab capture (byte-exact, identity 1.0)
NONSPEC_EAGER_OFFLINE_371 = 21.62   # #371 E: offline full-vocab eager floor
FUSION_OFFLINE_371 = 97.01          # #371 F: offline full-vocab capture+fusion (identity 0.0703)
FUSION_IDENTITY_371 = 0.0703        # #371 measured fusion token_identity_rate (identity-breaking)
OFFICIAL_DEPLOYED_TPS = 481.53      # PR #52 deployed (NON-strict)
PPL_GATE = 2.42
PPL_BASELINE = 2.3772
MILESTONE = 500.0

GAP_91_TO_165 = STRICT_FRONTIER_TPS - NONSPEC_CAPTURE_OFFLINE_371  # 74.06

# The #371 eager (E) decode = the spec-off NON-fused full-vocab plain greedy-AR reference.
REF_DECODE_DEFAULT = (REPO_ROOT / "research/validity/nonspec_baselift_cudagraph/"
                      "measured/decode_E.jsonl")

LADDER = ["S0", "S1", "S2", "S3", "S4"]
LADDER_LEVER = {
    "S0": "capture anchor (mode=0 + cudagraph=FULL)",
    "S1": "+ precache (prefix-cache warm; PUBLIC-only serve lever, decode-neutral)",
    "S2": "+ split-KV (FA2 backend; SPLITKV_VERIFY spec-lever INERT at M=1 non-spec)",
    "S3": "+ weight-residency / served launch config (gpu_mem 0.90, batched_tokens 512)",
    "S4": "+ fusion (inductor mode=3) — the #371 IDENTITY BREAKER",
}
# Which named levers are identity-safe-CANDIDATE (S1/S2/S3) vs the identity-breaker (S4).
NAMED_IDENTITY_SAFE_RUNGS = ["S1", "S2", "S3"]

TOL = 1e-9
TOL_RT = 1e-6


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _import_verifier():
    from scripts.local_validation import paths  # noqa: E402
    return paths.import_greedy_identity()


# ---------------------------------------------------------------------------- #
# (1) Load the GPU measurement (one measure_<cfg>.json + decode_<cfg>.jsonl per rung).
# ---------------------------------------------------------------------------- #
def load_measured(measured_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": str(measured_dir), "configs": {}}
    for cfg in LADDER:
        mjson = measured_dir / f"measure_{cfg}.json"
        djsonl = measured_dir / f"decode_{cfg}.jsonl"
        entry: dict[str, Any] = {"present": False}
        if mjson.exists():
            try:
                entry = json.loads(mjson.read_text())
                entry["present"] = True
                entry["decode_jsonl_exists"] = djsonl.exists()
                entry["decode_jsonl"] = str(djsonl)
            except Exception as exc:  # noqa: BLE001
                entry = {"present": False, "error": f"load failed: {exc}"}
        out["configs"][cfg] = entry
    return out


# ---------------------------------------------------------------------------- #
# (2) Identity vs the spec-off NON-fused full-vocab plain-AR reference (#371 eager E).
# ---------------------------------------------------------------------------- #
def compare_identity(verifier, ref_records: dict, cand_records: dict) -> dict[str, Any]:
    rep = verifier.compare(ref_records, cand_records)
    n = rep.num_prompts_compared
    rate = (rep.num_identical / n) if n else float("nan")
    return {
        "verdict": rep.verdict,
        "num_identical": rep.num_identical,
        "num_divergent": rep.num_divergent,
        "num_prompts_compared": n,
        "total_divergent_tokens": rep.total_divergent_tokens,
        "rate": rate,
    }


def compute_identity(measured: dict, verifier, ref_decode: Path) -> dict[str, Any]:
    res: dict[str, Any] = {"ref_decode": str(ref_decode)}
    if not ref_decode.exists():
        res["available"] = False
        res["note"] = f"reference decode (spec-off non-fused E) not found at {ref_decode}"
        return res
    ref_rec = verifier.load_decode_outputs(ref_decode)
    res["available"] = True
    res["n_reference_records"] = len(ref_rec)
    res["per_rung"] = {}
    for cfg in LADDER:
        c = measured["configs"].get(cfg, {})
        if c.get("present") and c.get("decode_jsonl_exists"):
            c_rec = verifier.load_decode_outputs(Path(c["decode_jsonl"]))
            common_ref = {k: v for k, v in ref_rec.items() if k in c_rec}
            res["per_rung"][cfg] = compare_identity(verifier, common_ref, c_rec)
    return res


# ---------------------------------------------------------------------------- #
# (3) Roofline attribution of the 91 -> 165 gap (lmhead12k LOSSY + fusion id-breaking).
# ---------------------------------------------------------------------------- #
def compute_attribution(capture_tps: float | None, fused_tps: float | None) -> dict[str, Any]:
    fullvocab_read = BODY_INT4_GB + LMHEAD_BF16_FULL_GB
    lmhead12k_read = BODY_INT4_GB + LMHEAD_INT4_12K_GB
    ceiling_fullvocab = A10G_BW_GBPS / fullvocab_read
    ceiling_lmhead12k = A10G_BW_GBPS / lmhead12k_read
    lmhead12k_ceiling_ratio = ceiling_lmhead12k / ceiling_fullvocab  # == fullvocab_read/lmhead12k_read
    base = capture_tps if _finite(capture_tps) else NONSPEC_CAPTURE_OFFLINE_371

    # lmhead12k: implied served TPS at the SAME BW-fraction as offline capture (LOSSY lever).
    lmhead12k_implied_tps = base * lmhead12k_ceiling_ratio
    lmhead12k_delta = lmhead12k_implied_tps - base
    # fusion: measured offline (== #371 F/C); identity-BREAKING.
    fusion_ratio = (fused_tps / capture_tps) if (_finite(fused_tps) and _finite(capture_tps)
                                                 and capture_tps) else (FUSION_OFFLINE_371 / NONSPEC_CAPTURE_OFFLINE_371)
    fusion_delta = base * (fusion_ratio - 1.0)
    # local -> official bridge (#196's own 156.05 -> 165.44); identity-neutral measurement path.
    bridge_ratio = STRICT_FRONTIER_TPS / STRICT_FRONTIER_LOCAL_TPS
    bridge_delta = base * (bridge_ratio - 1.0)

    return {
        "fullvocab_read_gb": fullvocab_read,
        "lmhead12k_read_gb": lmhead12k_read,
        "ceiling_fullvocab_tps": ceiling_fullvocab,
        "ceiling_lmhead12k_tps": ceiling_lmhead12k,
        "lmhead12k_ceiling_ratio": lmhead12k_ceiling_ratio,
        "lmhead12k_implied_served_tps": lmhead12k_implied_tps,
        "lmhead12k_delta_tps": lmhead12k_delta,
        "lmhead12k_is_lossy": True,
        "lmhead12k_in_named_levers": False,
        "fusion_offline_ratio": fusion_ratio,
        "fusion_delta_tps": fusion_delta,
        "fusion_is_identity_breaking": True,
        "bridge_ratio_local_to_official": bridge_ratio,
        "bridge_delta_tps": bridge_delta,
        "bridge_is_identity_neutral": True,
        "note": (
            "lmhead12k read-reduction (ceiling x{:.3f}) alone implies ~{:.0f} TPS from the {:.1f} "
            "capture base -> ~the entire 91->165 gap, and it is LOSSY (byte-exact only vs its own "
            "pruned checkpoint). fusion adds ~+{:.0%} (identity-breaking, #371). The PR's named "
            "levers (precache/split-KV/residency) are NOT in this composition.").format(
                lmhead12k_ceiling_ratio, lmhead12k_implied_tps, base, fusion_ratio - 1.0),
    }


# ---------------------------------------------------------------------------- #
# (4) Assemble the verdict fields.
# ---------------------------------------------------------------------------- #
def _tps_of(measured: dict, cfg: str) -> float | None:
    c = measured["configs"].get(cfg, {})
    v = c.get("decode_tps_single_stream") if c.get("present") else None
    return v if _finite(v) else None


def _rate_of(identity: dict, cfg: str) -> float | None:
    r = identity.get("per_rung", {}).get(cfg, {}).get("rate") if identity.get("available") else None
    return r if _finite(r) else None


def assemble_metrics(measured: dict, identity: dict, attribution: dict) -> dict[str, Any]:
    tps = {cfg: _tps_of(measured, cfg) for cfg in LADDER}
    rate = {cfg: _rate_of(identity, cfg) for cfg in LADDER}

    # ladder deltas (from prev and from S0 capture anchor).
    delta_from_prev = {}
    delta_from_s0 = {}
    for i, cfg in enumerate(LADDER):
        if _finite(tps[cfg]):
            if i > 0 and _finite(tps[LADDER[i - 1]]):
                delta_from_prev[cfg] = tps[cfg] - tps[LADDER[i - 1]]
            if _finite(tps["S0"]):
                delta_from_s0[cfg] = tps[cfg] - tps["S0"]

    s0 = tps["S0"]
    # identity-safe rungs = those whose rate is exactly 1.0 (byte-exact vs the non-fused E ref).
    identity_safe = {cfg: (rate[cfg] is not None and abs(rate[cfg] - 1.0) < 1e-12) for cfg in LADDER}
    identity_safe_tps = [tps[cfg] for cfg in LADDER if identity_safe[cfg] and _finite(tps[cfg])]
    max_identity_safe_nonspec_tps = max(identity_safe_tps) if identity_safe_tps else None

    # gap_attribution for the PR's NAMED levers, measured offline (precache/split-KV/residency).
    # Each named-lever delta is its rung minus the previous rung; sum = realized identity-safe lift.
    named_safe_delta = 0.0
    gap_attribution_named = {}
    if _finite(s0):
        prev = s0
        for cfg in NAMED_IDENTITY_SAFE_RUNGS:
            if _finite(tps[cfg]) and identity_safe[cfg]:
                d = tps[cfg] - prev
                gap_attribution_named[cfg] = d
                named_safe_delta += d
                prev = tps[cfg]
            else:
                gap_attribution_named[cfg] = None
    frac_of_91_to_165_identity_safe = (named_safe_delta / GAP_91_TO_165) if GAP_91_TO_165 else None

    # Which NAMED levers turned out byte-exact vs identity-breaking (DATA-DRIVEN). The original
    # ladder ASSUMED S1/S2/S3 identity-safe; S1 (precache == enable_prefix_caching) measured
    # otherwise, so this is read from the rates, not assumed.
    named_levers_identity_safe = [c for c in NAMED_IDENTITY_SAFE_RUNGS if identity_safe.get(c)]
    named_levers_identity_breaking = [c for c in NAMED_IDENTITY_SAFE_RUNGS
                                      if rate.get(c) is not None and rate[c] < 1.0]
    precache_breaks_identity = bool(rate.get("S1") is not None and rate["S1"] < 1.0)

    # fusion is in the served 165.44 (manifest: cudagraph+fusion ON, no --enforce-eager).
    fusion_in_served_165 = True
    s4_rate = rate["S4"]
    fusion_breaks_identity_measured = (s4_rate is not None and s4_rate < 1.0)

    # gate semantics. The spec-off SERVED gate (REFERENCE_MODE_ENV) generates the reference on
    # the submission's OWN engine (fusion+lmhead12k ON, only speculation removed) -> 165.44 passes
    # it SELF-REFERENTIALLY (#196 reported 1.0). It is NOT byte-exact vs a non-fused full-vocab ref:
    # S4 (fusion-on) vs E measures exactly that divergence.
    served_165_passes_self_referential_gate = True   # = #196's reported token_identity_rate 1.0
    served_165_passes_nonfused_fullvocab_gate = bool(
        not fusion_in_served_165) and False  # banks fusion (and lmhead12k) -> False
    # headline (audited, non-self-referential reading): does it pass a spec-off NON-fused ref?
    served_165_passes_spec_off_gate = served_165_passes_nonfused_fullvocab_gate

    is_strict_byte_exact = served_165_passes_nonfused_fullvocab_gate  # vs non-fused full-vocab AR

    # PR item 3: the two explicit served-165 readings.
    #   self-referential: 165.44 at full speed (passes its OWN fused+pruned reference, #196 1.0).
    #   strict-fusion-off: strip the fusion speedup (165.44 / fusion_ratio). NB this is STILL
    #   lmhead12k-lossy, so it is NOT byte-exact vs full-vocab either; the truly byte-exact
    #   full-vocab frontier is the offline S0 capture (~91 = max_identity_safe_nonspec_tps).
    fusion_ratio = attribution.get("fusion_offline_ratio")
    served_165_self_referential_tps = STRICT_FRONTIER_TPS
    served_165_strict_tps_fusion_off = (STRICT_FRONTIER_TPS / fusion_ratio
                                        if _finite(fusion_ratio) and fusion_ratio else None)

    # does the byte-exact identity-safe floor MOVE up from the named levers? They add ~0, so no.
    moved = bool(max_identity_safe_nonspec_tps is not None
                 and _finite(s0) and (max_identity_safe_nonspec_tps - s0) > 1.0
                 and max_identity_safe_nonspec_tps > STRICT_FRONTIER_TPS)
    nonspec_strict_floor_moves = {
        "moves": moved,
        "new_value": max_identity_safe_nonspec_tps if moved else STRICT_FRONTIER_TPS,
        "reason": ("named identity-safe levers (precache/split-KV/residency) add ~0 offline; "
                   "no byte-exact lever lifts the frontier above 165.44. The 165.44 itself banks "
                   "fusion (identity-breaking) + lmhead12k (lossy), so it is NOT a clean "
                   "byte-exact-vs-non-fused frontier — it stands only under the self-referential gate."),
    }

    return {
        "tps": tps,
        "token_identity_rate": rate,
        "identity_safe": identity_safe,
        "delta_tps_from_prev": delta_from_prev,
        "delta_tps_from_s0": delta_from_s0,
        "gap_attribution_named_levers": gap_attribution_named,
        "named_identity_safe_delta_tps": named_safe_delta,
        "named_levers_identity_safe": named_levers_identity_safe,
        "named_levers_identity_breaking": named_levers_identity_breaking,
        "precache_breaks_identity": precache_breaks_identity,
        # ---- load-bearing verdict fields ----
        "frac_of_91_to_165_identity_safe": frac_of_91_to_165_identity_safe,
        "served_165_passes_spec_off_gate": served_165_passes_spec_off_gate,
        "served_165_passes_self_referential_gate": served_165_passes_self_referential_gate,
        "served_165_passes_nonfused_fullvocab_gate": served_165_passes_nonfused_fullvocab_gate,
        "fusion_in_served_165": fusion_in_served_165,
        "fusion_breaks_identity_measured": fusion_breaks_identity_measured,
        "served_165_self_referential_tps": served_165_self_referential_tps,
        "served_165_strict_tps_fusion_off": served_165_strict_tps_fusion_off,
        "max_identity_safe_nonspec_tps": max_identity_safe_nonspec_tps,
        "nonspec_strict_floor_moves": nonspec_strict_floor_moves,
        "is_strict_byte_exact": is_strict_byte_exact,
        # ---- attribution of the REAL composition ----
        "lmhead12k_implied_served_tps": attribution["lmhead12k_implied_served_tps"],
        "lmhead12k_delta_tps": attribution["lmhead12k_delta_tps"],
        "fusion_delta_tps": attribution["fusion_delta_tps"],
        "bridge_delta_tps": attribution["bridge_delta_tps"],
        # ---- #371/#196 reconciliation ----
        "s0_reproduces_371_capture": (abs(s0 - NONSPEC_CAPTURE_OFFLINE_371) < 8.0
                                      if _finite(s0) else None),
        "s4_reproduces_371_fusion_break": (s4_rate is not None and s4_rate < 0.5),
    }


# ---------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY) — 0-GPU synthetic validation of the math + identity + gate logic.
# ---------------------------------------------------------------------------- #
def self_test(verifier) -> dict[str, Any]:
    # Synthetic ladder: S0..S3 byte-exact (identity 1.0) ~91-93; S4 fusion-broken ~97 (rate<1).
    tps_synth = {"S0": 91.4, "S1": 91.3, "S2": 91.5, "S3": 92.6, "S4": 97.0}
    ref = {f"p{i}": {"completion_token_ids": [i, i + 1, i + 2, i + 3]} for i in range(8)}
    safe_rec = {k: {"completion_token_ids": list(v["completion_token_ids"])} for k, v in ref.items()}
    broken_rec = {k: {"completion_token_ids": list(v["completion_token_ids"])} for k, v in ref.items()}
    broken_rec["p3"]["completion_token_ids"][2] = 999999  # one divergence -> rate < 1.0

    measured = {"configs": {}}
    identity = {"available": True, "per_rung": {}}
    for cfg in LADDER:
        measured["configs"][cfg] = {"present": True, "decode_tps_single_stream": tps_synth[cfg],
                                    "decode_jsonl_exists": True}
        cmp = compare_identity(verifier, ref, safe_rec if cfg != "S4" else broken_rec)
        identity["per_rung"][cfg] = cmp

    attribution = compute_attribution(tps_synth["S0"], tps_synth["S4"])
    m = assemble_metrics(measured, identity, attribution)

    # (a) roofline: lmhead12k ceiling ratio == fullvocab_read/lmhead12k_read, and the implied
    #     served TPS lands near the served frontier band.
    fullvocab_read = BODY_INT4_GB + LMHEAD_BF16_FULL_GB
    lmhead12k_read = BODY_INT4_GB + LMHEAD_INT4_12K_GB
    a = bool(abs(attribution["lmhead12k_ceiling_ratio"] - fullvocab_read / lmhead12k_read) < TOL_RT
             and 150.0 < attribution["lmhead12k_implied_served_tps"] < 175.0)
    # (b) identity logic: S0..S3 identity-safe (rate 1.0), S4 broken (rate<1); max identity-safe
    #     TPS picks the fastest byte-exact rung and does NOT exceed 165.44.
    b = bool(all(m["identity_safe"][c] for c in ["S0", "S1", "S2", "S3"])
             and m["identity_safe"]["S4"] is False
             and m["fusion_breaks_identity_measured"] is True
             and m["max_identity_safe_nonspec_tps"] == max(tps_synth[c] for c in ["S0", "S1", "S2", "S3"])
             and m["max_identity_safe_nonspec_tps"] < STRICT_FRONTIER_TPS)
    # (c) gate logic: passes self-referential, FAILS non-fused full-vocab + spec-off headline;
    #     fusion_in_served_165 True; is_strict_byte_exact False.
    c = bool(m["served_165_passes_self_referential_gate"] is True
             and m["served_165_passes_nonfused_fullvocab_gate"] is False
             and m["served_165_passes_spec_off_gate"] is False
             and m["fusion_in_served_165"] is True
             and m["is_strict_byte_exact"] is False
             and m["served_165_self_referential_tps"] == STRICT_FRONTIER_TPS
             and m["served_165_strict_tps_fusion_off"] is not None
             and m["served_165_strict_tps_fusion_off"] < STRICT_FRONTIER_TPS)
    # (d) frac_of_91_to_165_identity_safe is ~0 (named levers add ~0), floor does NOT move.
    d = bool(m["frac_of_91_to_165_identity_safe"] is not None
             and abs(m["frac_of_91_to_165_identity_safe"]) < 0.10
             and m["nonspec_strict_floor_moves"]["moves"] is False)
    # (e) official verifier verdicts are the expected GREEDY_IDENTICAL / DIVERGENT.
    e = bool(identity["per_rung"]["S0"]["verdict"] == "GREEDY_IDENTICAL"
             and identity["per_rung"]["S4"]["verdict"] == "DIVERGENT"
             and identity["per_rung"]["S4"]["num_divergent"] == 1)
    # (f) imported constants exact + lmhead12k read math.
    f = bool(abs(BODY_INT4_GB - 1.6973824) < TOL and abs(A10G_BW_GBPS - 600.0) < TOL
             and abs(STRICT_FRONTIER_TPS - 165.44) < TOL
             and abs(LMHEAD_INT4_12K_GB - LMHEAD12K_ROWS * HIDDEN * 0.5 / 1e9) < TOL_RT
             and abs(LMHEAD_BF16_FULL_GB - 1.34217728) < TOL_RT
             and abs(GAP_91_TO_165 - (STRICT_FRONTIER_TPS - NONSPEC_CAPTURE_OFFLINE_371)) < TOL_RT)

    conditions = {
        "a_roofline_attribution": a,
        "b_identity_ladder_logic": b,
        "c_gate_semantics": c,
        "d_frac_identity_safe_near_zero": d,
        "e_official_verifier_verdicts": e,
        "f_imports_exact": f,
    }
    return {
        "conditions": conditions,
        "served_stack_self_test_passes": bool(all(conditions.values())),
        "synthetic_metrics": m,
    }


# ---------------------------------------------------------------------------- #
# Synthesis.
# ---------------------------------------------------------------------------- #
def synthesize(measured: dict | None, ref_decode: Path) -> dict[str, Any]:
    verifier = _import_verifier()
    st = self_test(verifier)

    if measured is None:
        return {
            "measured_present": False,
            "self_test": st,
            "served_stack_self_test_passes": st["served_stack_self_test_passes"],
            "frac_of_91_to_165_identity_safe": None,
            "verdict": "SELF-TEST ONLY (no measured GPU data supplied).",
        }

    identity = compute_identity(measured, verifier, ref_decode)
    capture_tps = _tps_of(measured, "S0")
    fused_tps = _tps_of(measured, "S4")
    attribution = compute_attribution(capture_tps, fused_tps)
    metrics = assemble_metrics(measured, identity, attribution)
    verdict = _build_verdict(metrics, attribution, identity)
    return {
        "measured_present": True,
        "measured": measured,
        "identity": identity,
        "attribution": attribution,
        "metrics": metrics,
        "self_test": st,
        "served_stack_self_test_passes": st["served_stack_self_test_passes"],
        "frac_of_91_to_165_identity_safe": metrics["frac_of_91_to_165_identity_safe"],
        "verdict": verdict,
    }


def _build_verdict(m: dict, attr: dict, identity: dict) -> str:
    tps, rate = m["tps"], m["token_identity_rate"]
    parts = []
    ladder_str = ", ".join(
        f"{c} {tps[c]:.1f}(id={rate[c]:.3f})" if _finite(tps[c]) and rate[c] is not None
        else f"{c} -" for c in LADDER)
    parts.append(f"OFFLINE full-vocab ladder vs spec-off non-fused E reference: {ladder_str}.")
    safe = m.get("named_levers_identity_safe") or []
    broke = m.get("named_levers_identity_breaking") or []
    parts.append(
        f"Named levers (precache/split-KV/residency) byte-exact={safe or 'NONE'}, "
        f"identity-BREAKING={broke or 'none'} (precache/prefix-caching breaks identity="
        f"{m.get('precache_breaks_identity')}); realized identity-safe delta = "
        f"{m['named_identity_safe_delta_tps']:.2f} TPS -> frac_of_91_to_165_identity_safe = "
        f"{m['frac_of_91_to_165_identity_safe']:.4f} (the {GAP_91_TO_165:.1f} TPS gap is NOT "
        f"composed of byte-exact named levers).")
    parts.append(
        f"REAL composition: lmhead12k read-reduction (LOSSY, NOT in named set) implies "
        f"~{attr['lmhead12k_implied_served_tps']:.0f} TPS (ceiling x{attr['lmhead12k_ceiling_ratio']:.2f}) "
        f"= ~the whole gap; fusion (#371, identity-breaking) +{attr['fusion_offline_ratio']-1.0:.0%}; "
        f"local->official bridge x{attr['bridge_ratio_local_to_official']:.3f}.")
    parts.append(
        f"GATE: served 165.44 passes the SELF-REFERENTIAL served gate (ref on its own fused+pruned "
        f"engine, #196 reported 1.0) but fusion_in_served_165={m['fusion_in_served_165']} and "
        f"S4(fusion) vs E identity={rate.get('S4')} -> it is NOT byte-exact vs a non-fused full-vocab "
        f"reference: served_165_passes_spec_off_gate(non-self-ref)={m['served_165_passes_spec_off_gate']}, "
        f"is_strict_byte_exact={m['is_strict_byte_exact']}.")
    parts.append(
        f"max_identity_safe_nonspec_tps={m['max_identity_safe_nonspec_tps']} (offline full-vocab, "
        f"<165.44); nonspec_strict_floor_moves={m['nonspec_strict_floor_moves']['moves']}. "
        "LOCAL proxy; 0 official TPS; strict identity gate is HARD.")
    return " ".join(parts)


# ---------------------------------------------------------------------------- #
# NaN guard / report / W&B (house pattern).
# ---------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict) -> None:
    print("\n" + "=" * 100, flush=True)
    print("SERVED-STACK IDENTITY ABLATION: attribute 91->165 + audit the 165.44 strict floor (PR #385)",
          flush=True)
    print("=" * 100, flush=True)
    st = syn["self_test"]
    print(f"  (PRIMARY) served_stack_self_test_passes = {st['served_stack_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    if not syn.get("measured_present"):
        print(f"  {syn['verdict']}", flush=True)
        print("=" * 100, flush=True)
        return
    m = syn["metrics"]
    print("-" * 100, flush=True)
    print("  LADDER (tps | identity vs spec-off non-fused E):", flush=True)
    for c in LADDER:
        t = m["tps"][c]
        r = m["token_identity_rate"][c]
        dp = m["delta_tps_from_prev"].get(c)
        print(f"    {c} {LADDER_LEVER[c]:<62} tps={t if t is None else round(t,2)}  "
              f"id_rate={r}  d_prev={None if dp is None else round(dp,2)}", flush=True)
    print("  --- LOAD-BEARING VERDICT FIELDS ---", flush=True)
    print(f"  frac_of_91_to_165_identity_safe   = {m['frac_of_91_to_165_identity_safe']}", flush=True)
    print(f"  served_165_passes_spec_off_gate   = {m['served_165_passes_spec_off_gate']} "
          f"(self-referential={m['served_165_passes_self_referential_gate']})", flush=True)
    print(f"  fusion_in_served_165              = {m['fusion_in_served_165']} "
          f"(S4 breaks identity measured={m['fusion_breaks_identity_measured']})", flush=True)
    print(f"  served_165_self_referential_tps   = {m['served_165_self_referential_tps']} "
          f"| strict_tps_fusion_off = {m['served_165_strict_tps_fusion_off']}", flush=True)
    print(f"  max_identity_safe_nonspec_tps     = {m['max_identity_safe_nonspec_tps']}", flush=True)
    print(f"  nonspec_strict_floor_moves        = {m['nonspec_strict_floor_moves']['moves']}", flush=True)
    print(f"  is_strict_byte_exact             = {m['is_strict_byte_exact']}", flush=True)
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
        print(f"[served-stack] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    st = syn["self_test"]
    summary: dict[str, Any] = {
        "served_stack_self_test_passes": int(bool(st["served_stack_self_test_passes"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        "measured_present": int(bool(syn.get("measured_present"))),
    }
    if syn.get("measured_present"):
        m = syn["metrics"]
        for c in LADDER:
            if _finite(m["tps"][c]):
                summary[f"tps_{c}"] = m["tps"][c]
            if m["token_identity_rate"][c] is not None:
                summary[f"identity_rate_{c}"] = m["token_identity_rate"][c]
        for key in ("frac_of_91_to_165_identity_safe", "max_identity_safe_nonspec_tps",
                    "named_identity_safe_delta_tps", "lmhead12k_implied_served_tps",
                    "lmhead12k_delta_tps", "fusion_delta_tps", "bridge_delta_tps",
                    "served_165_self_referential_tps", "served_165_strict_tps_fusion_off"):
            if m.get(key) is not None:
                summary[key] = m[key]
        for key in ("served_165_passes_spec_off_gate", "served_165_passes_self_referential_gate",
                    "served_165_passes_nonfused_fullvocab_gate", "fusion_in_served_165",
                    "fusion_breaks_identity_measured", "is_strict_byte_exact",
                    "precache_breaks_identity"):
            summary[key] = int(bool(m.get(key)))
        summary["nonspec_strict_floor_moves"] = int(bool(m["nonspec_strict_floor_moves"]["moves"]))

    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "served-stack-ablation", "nonspec-baselift", "byte-exact-identity",
              "lmhead12k", "inductor-fusion", "roofline", "bank-the-analysis", "pr-385"],
        config={
            "body_int4_gb": BODY_INT4_GB, "a10g_bw_gbps": A10G_BW_GBPS,
            "lmhead_bf16_full_gb": LMHEAD_BF16_FULL_GB, "lmhead_int4_12k_gb": LMHEAD_INT4_12K_GB,
            "lmhead12k_rows": LMHEAD12K_ROWS, "vocab": VOCAB, "hidden": HIDDEN,
            "strict_frontier_tps": STRICT_FRONTIER_TPS,
            "strict_frontier_local_tps": STRICT_FRONTIER_LOCAL_TPS,
            "nonspec_capture_offline_371": NONSPEC_CAPTURE_OFFLINE_371,
            "fusion_identity_371": FUSION_IDENTITY_371,
            "official_deployed_tps": OFFICIAL_DEPLOYED_TPS, "ppl_gate": PPL_GATE,
            "ppl_baseline": PPL_BASELINE, "milestone_tps": MILESTONE, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[served-stack] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="served_stack_identity_ablation_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[served-stack] wandb logged: {summary}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY 0-GPU self-validation")
    ap.add_argument("--reanalyze", action="store_true",
                    help="re-derive metrics from saved measure_*.json only (0-GPU)")
    ap.add_argument("--measured-dir", type=Path, default=HERE / "measured")
    ap.add_argument("--ref-decode", type=Path, default=REF_DECODE_DEFAULT,
                    help="spec-off NON-fused full-vocab plain-AR reference (default: #371 eager E)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="nonspec-baselift-cudagraph")
    args = ap.parse_args(argv)

    measured = None
    md = args.measured_dir
    if not args.self_test or args.reanalyze:
        if md and (md / "measure_S0.json").exists():
            measured = load_measured(md)
        elif not args.self_test:
            print(f"[served-stack] no measure_S0.json under {md} — synthesizing self-test only.",
                  flush=True)

    syn = synthesize(measured, args.ref_decode)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 385, "agent": "land",
        "kind": "served-stack-identity-ablation", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[served-stack] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "served_stack_identity_ablation_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[served-stack] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(syn["served_stack_self_test_passes"]) and payload["nan_clean"]
    print(f"  PRIMARY served_stack_self_test_passes = {passes}", flush=True)
    print(f"  TEST frac_of_91_to_165_identity_safe = {syn.get('frac_of_91_to_165_identity_safe')}",
          flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        print(f"[served-stack] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
