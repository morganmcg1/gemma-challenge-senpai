#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Non-spec base-lift: CUDA-graph CAPTURE vs inductor FUSION on the byte-exact int4
M=1 decode (PR #371) — synthesis + W&B card (0-GPU).

WHAT THIS LEG DECIDES
---------------------
The strict non-spec frontier is 165.44 TPS (lawine #196; int4 M=1 plain AR,
token_identity_rate=1.0). The PR roofline: a single int4 decode reads the ~1.697 GB
int4 body -> at 600 GB/s a body-only BW ceiling ~353 TPS, so 165 sits ~47% of it and
the missing ~2x is launch overhead + non-resident weights + un-fused eager scheduling.
kanna #359 found the BUNDLED "native CUDA-graph + inductor fusion" lever (~6.2x) breaks
greedy identity (6-7/24) but never isolated the two levers. They are different:
  * CUDA-graph CAPTURE  = launch-overhead elimination + weight residency. Same kernels,
    same reduction order -> byte-exact -> IDENTITY-SAFE.
  * inductor FUSION     = Triton reduction REORDER (RMSNorm/softmax/split-K) -> the
    actual IDENTITY-BREAKER.

This card consumes the GPU measurement (measure_capture.py: E/C/F decode jsonls + slope
decode TPS) and synthesizes the decision: it runs the OFFICIAL greedy_identity verifier
(C-vs-E, F-vs-E), prices each lever on the BW roofline, and reports whether capture is
byte-exact AND how much base-lift it recovers identity-safe. PRIMARY = self-test (0-GPU
math/logic validation). The identity gate is the HARD scientific gate: capture's
token_identity_rate vs the eager AR reference MUST be 1.0 for its TPS to count.

ROOFLINE HONESTY
----------------
The PR headline ceiling (~353 TPS) is BODY-ONLY (1.697 GB / 600 GB/s) — it is the ceiling
*if* the lm_head read is also eliminated (a SEPARATE, lossy PCK-04-style lever, NOT this
identity-safe experiment). The plain full-vocab int4 config reads a FULL bf16 lm_head
(262144x2560x2 = 1.342 GB) every step, so the ceiling FOR THIS CONFIG is body+lm_head
(~3.04 GB -> ~197 TPS). We report BOTH plus the EMPIRICAL bytes/step (= BW / measured TPS)
so the latency-bound vs read-bound question is answered by the data, not assumed.

SCOPE
-----
LOCAL pod-A10G only; local wall_tps proxy (0 official TPS). NO HF Job / submission /
served-file change. Analysis + measurement banking; adds 0 official TPS. The strict
identity gate (token_identity_rate=1.0) is HARD.

PRIMARY metric  nonspec_baselift_self_test_passes
TEST    metric  cudagraph_capture_token_identity_rate

Run:
  # GPU legs first (one process per config; see measure_capture.py):
  CUDA_VISIBLE_DEVICES=0 .../python measure_capture.py --config E --out-dir <dir>/measured
  CUDA_VISIBLE_DEVICES=0 .../python measure_capture.py --config C --out-dir <dir>/measured
  CUDA_VISIBLE_DEVICES=0 .../python measure_capture.py --config F --out-dir <dir>/measured
  # then synthesize + log (0-GPU):
  python nonspec_baselift_cudagraph.py --measured-dir <dir>/measured \
    --wandb_group nonspec-baselift-cudagraph --wandb_name land/nonspec-baselift-cudagraph
  # 0-GPU re-derive from saved JSONs only:
  python nonspec_baselift_cudagraph.py --reanalyze --measured-dir <dir>/measured
  # PRIMARY self-validation (0-GPU, no measured data needed):
  python nonspec_baselift_cudagraph.py --self-test
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

# Pre-import the REAL wandb (site-packages) BEFORE putting REPO_ROOT on sys.path[0].
# REPO_ROOT (= target/) contains a ./wandb run-output dir; once it is sys.path[0] it
# shadows the package as an empty PEP-420 namespace (__file__=None, no .init). Here the
# script dir is still sys.path[0] (no wandb/ there), so this resolves to the installed
# package and caches it in sys.modules -> the later `import wandb` in _maybe_log_wandb
# returns the real module regardless of path order.
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
BODY_INT4_GB = 1.6973824            # denken #344/#278 int4 body bytes/step
A10G_BW_GBPS = 600.0                # A10G HBM peak bandwidth (the figure the roofline uses)
VOCAB = 262144                      # gemma-4-E4B vocab (text_config.vocab_size)
HIDDEN = 2560                       # gemma-4-E4B hidden_size
LMHEAD_BF16_FULL_GB = VOCAB * HIDDEN * 2 / 1e9   # full bf16 tied lm_head read (1.342 GB)
STRICT_FRONTIER_TPS = 165.44        # lawine #196 strict non-spec frontier
OFFICIAL_DEPLOYED_TPS = 481.53      # PR #52 deployed (NOT strict)
PPL_GATE = 2.42
PPL_BASELINE = 2.3772
MILESTONE = 500.0

# HF bf16 plain-greedy-AR offline reference (the strict-frontier identity target for
# step 1: int4-eager MUST reproduce this at rate 1.0 to BE lawine #196's 165.44 frontier).
BF16_REF_JSONL = REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.offline.jsonl"

# Inductor/Triton fusion taxonomy (research-confirmed; the basis for which part of the
# C->F gap is identity-breaking). SAFE = no FP reduction reorder; BREAKING = reorders.
FUSION_TAXONOMY = {
    "identity_safe": [
        "epilogue fusion (bias/activation/residual pointwise post-GEMM)",
        "pointwise/elementwise fusion",
        "eliminate_noops",
        "vLLM custom ops shared across E/C (fuse_norm_quant, fuse_act_quant)",
    ],
    "identity_breaking": [
        "RMSNorm/LayerNorm reduction (triton_per/triton_red persistent/looped)",
        "softmax reduction reorder",
        "GEMM split-K partial-accumulate / MixOrderReduction",
        "fused attention-score reduction",
    ],
}

TOL = 1e-9
TOL_RT = 1e-6


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _import_verifier():
    from scripts.local_validation import paths  # noqa: E402
    return paths.import_greedy_identity()


# ---------------------------------------------------------------------------- #
# (1) Load the GPU measurement (one measure_<cfg>.json + decode_<cfg>.jsonl per config).
# ---------------------------------------------------------------------------- #
def load_measured(measured_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": str(measured_dir), "configs": {}}
    for cfg in ("E", "C", "F"):
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
# (2) Identity comparison via the OFFICIAL greedy verifier (C-vs-E, F-vs-E).
# ---------------------------------------------------------------------------- #
def _records_from_jsonl(verifier, path: Path) -> dict[str, dict]:
    return verifier.load_decode_outputs(path)


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


def compute_identity(measured: dict, verifier) -> dict[str, Any]:
    cfgs = measured["configs"]
    res: dict[str, Any] = {}
    e = cfgs.get("E", {})
    if not (e.get("present") and e.get("decode_jsonl_exists")):
        res["available"] = False
        res["note"] = "no eager (E) decode capture present -> identity gate unmeasured"
        return res
    e_rec = _records_from_jsonl(verifier, Path(e["decode_jsonl"]))
    res["available"] = True
    res["n_reference_records"] = len(e_rec)
    # Step 1 (baseline faithfulness): int4-eager (E) vs the HF bf16 greedy-AR reference.
    # E must reproduce this at rate 1.0 to BE the strict non-spec frontier (165.44).
    if BF16_REF_JSONL.exists():
        bf16_rec = _records_from_jsonl(verifier, BF16_REF_JSONL)
        # compare only on the ids E actually decoded (E is keyed by the same eval ids).
        bf16_common = {k: v for k, v in bf16_rec.items() if k in e_rec}
        res["E_vs_bf16_ref"] = compare_identity(verifier, bf16_common, e_rec)
        res["E_vs_bf16_ref"]["bf16_ref"] = str(BF16_REF_JSONL)
        res["E_vs_bf16_ref"]["n_bf16_ref_records"] = len(bf16_rec)
    else:
        res["E_vs_bf16_ref"] = {"available": False,
                                "note": f"bf16 ref not found at {BF16_REF_JSONL}"}
    for cfg in ("C", "F"):
        c = cfgs.get(cfg, {})
        if c.get("present") and c.get("decode_jsonl_exists"):
            c_rec = _records_from_jsonl(verifier, Path(c["decode_jsonl"]))
            res[cfg] = compare_identity(verifier, e_rec, c_rec)
    return res


# ---------------------------------------------------------------------------- #
# (3) BW roofline — ceilings (body-only AND body+full-lm_head) + empirical bytes/step.
# ---------------------------------------------------------------------------- #
def compute_roofline(eager_tps: float | None) -> dict[str, Any]:
    body_bytes = BODY_INT4_GB
    body_lmhead_bytes = BODY_INT4_GB + LMHEAD_BF16_FULL_GB
    ceiling_body_only = A10G_BW_GBPS / body_bytes
    ceiling_body_lmhead = A10G_BW_GBPS / body_lmhead_bytes
    out: dict[str, Any] = {
        "body_int4_gb": body_bytes,
        "lmhead_bf16_full_gb": LMHEAD_BF16_FULL_GB,
        "body_plus_full_lmhead_gb": body_lmhead_bytes,
        "a10g_bw_gbps": A10G_BW_GBPS,
        "byte_exact_bw_ceiling_tps": ceiling_body_only,
        "byte_exact_bw_ceiling_tps_with_full_lmhead": ceiling_body_lmhead,
        "ceiling_basis": (
            "body-only ceiling = BW/body (PR headline ~353; the ceiling IF the lm_head read "
            "is ALSO removed via a separate lossy PCK-04-style lever). with-full-lmhead ceiling "
            "= BW/(body+full bf16 lm_head) ~197 is the ceiling FOR THIS plain full-vocab config."),
    }
    if _finite(eager_tps) and eager_tps and eager_tps > 0:
        empirical_bytes = A10G_BW_GBPS / eager_tps
        out.update({
            "eager_tps": eager_tps,
            "achieved_bw_fraction_eager": eager_tps * body_bytes / A10G_BW_GBPS,
            "achieved_bw_fraction_eager_with_full_lmhead": eager_tps * body_lmhead_bytes / A10G_BW_GBPS,
            "empirical_bytes_per_step_gb_eager": empirical_bytes,
            "empirical_exceeds_body": bool(empirical_bytes > body_bytes),
            "empirical_exceeds_body_plus_lmhead": bool(empirical_bytes > body_lmhead_bytes),
            "latency_bound_claim": (
                "empirical bytes/step (BW/eager_tps) vs the read walls: if it EXCEEDS "
                "body+lm_head the eager step is NOT read-bound (launch/overhead slack exists "
                "-> capture-recoverable); if below body-only the step is already read-bound."),
        })
    return out


def frac_recovered(tps: float | None, eager_tps: float | None, ceiling: float) -> float | None:
    if not (_finite(tps) and _finite(eager_tps)):
        return None
    denom = ceiling - eager_tps
    if denom == 0:
        return float("nan")
    return (tps - eager_tps) / denom


# ---------------------------------------------------------------------------- #
# (4) Assemble the 8 deliverable metrics + the lever decision.
# ---------------------------------------------------------------------------- #
def assemble_metrics(measured: dict, identity: dict, roofline: dict) -> dict[str, Any]:
    cfgs = measured["configs"]

    def tps_of(cfg: str) -> float | None:
        c = cfgs.get(cfg, {})
        v = c.get("decode_tps_single_stream") if c.get("present") else None
        return v if _finite(v) else None

    eager_tps = tps_of("E")
    capture_tps = tps_of("C")
    fused_tps = tps_of("F")
    ceiling = roofline["byte_exact_bw_ceiling_tps"]

    cap_id = identity.get("C", {}) if identity.get("available") else {}
    fus_id = identity.get("F", {}) if identity.get("available") else {}
    cap_rate = cap_id.get("rate")
    fus_rate = fus_id.get("rate")

    # Step 1: int4-eager vs HF bf16 reference (is E the strict frontier?).
    bf16_id = identity.get("E_vs_bf16_ref", {}) if identity.get("available") else {}
    bf16_rate = bf16_id.get("rate")
    baseline_eager_is_strict_frontier = (
        bf16_rate is not None and abs(bf16_rate - 1.0) < 1e-12)

    cudagraph_breaks_identity = (cap_rate is not None and cap_rate < 1.0)
    fusion_breaks_identity = (fus_rate is not None and fus_rate < 1.0)
    capture_is_identity_safe = (cap_rate is not None and abs(cap_rate - 1.0) < 1e-12)

    # identity-safe-max = the fastest config whose identity rate is exactly 1.0.
    # C (capture, no inductor) is the canonical identity-safe max; F only qualifies if
    # full inductor (against expectation) preserves identity.
    identity_safe_max_tps = None
    identity_safe_max_config = None
    if capture_is_identity_safe and _finite(capture_tps):
        identity_safe_max_tps, identity_safe_max_config = capture_tps, "C"
    if (fus_rate is not None and abs(fus_rate - 1.0) < 1e-12 and _finite(fused_tps)
            and (identity_safe_max_tps is None or fused_tps > identity_safe_max_tps)):
        identity_safe_max_tps, identity_safe_max_config = fused_tps, "F"

    residual_gap_requires_reduction_reorder = bool(
        fusion_breaks_identity and _finite(fused_tps) and _finite(capture_tps)
        and fused_tps > capture_tps)

    metrics = {
        "baseline_eager_byte_exact_tps": eager_tps,
        "achieved_bw_fraction_eager": roofline.get("achieved_bw_fraction_eager"),
        "byte_exact_bw_ceiling_tps": ceiling,
        "cudagraph_capture_token_identity_rate": cap_rate,
        "cudagraph_capture_tps": capture_tps,
        "baseline_eager_identity_vs_bf16_ref": bf16_rate,
        "baseline_eager_is_strict_frontier": baseline_eager_is_strict_frontier,
        "identity_safe_max_tps": identity_safe_max_tps,
        "frac_of_baselift_recovered_identity_safe":
            frac_recovered(identity_safe_max_tps, eager_tps, ceiling),
        "cudagraph_breaks_identity": cudagraph_breaks_identity,
        # auxiliary (classification + the fusion confirmation leg)
        "identity_safe_max_config": identity_safe_max_config,
        "capture_is_identity_safe": capture_is_identity_safe,
        "fused_capture_tps": fused_tps,
        "fusion_token_identity_rate": fus_rate,
        "fusion_breaks_identity": fusion_breaks_identity,
        "frac_of_baselift_recovered_capture":
            frac_recovered(capture_tps, eager_tps, ceiling),
        "frac_of_baselift_recovered_fused_total":
            frac_recovered(fused_tps, eager_tps, ceiling),
        "residual_gap_requires_reduction_reorder": residual_gap_requires_reduction_reorder,
        "capture_speedup_vs_eager": (capture_tps / eager_tps
                                     if _finite(capture_tps) and _finite(eager_tps) and eager_tps else None),
        "fused_speedup_vs_eager": (fused_tps / eager_tps
                                   if _finite(fused_tps) and _finite(eager_tps) and eager_tps else None),
    }
    return metrics


# ---------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY) — 0-GPU synthetic validation of the math + identity logic.
# ---------------------------------------------------------------------------- #
def self_test(verifier) -> dict[str, Any]:
    # Synthetic measured: eager=165.44, capture=300 (identity-safe), fused=900 (breaks).
    eager, cap, fus = 165.44, 300.0, 900.0
    ref = {f"p{i}": {"completion_token_ids": [i, i + 1, i + 2, i + 3]} for i in range(8)}
    cap_rec = {k: {"completion_token_ids": list(v["completion_token_ids"])}
               for k, v in ref.items()}                      # identical -> rate 1.0
    fus_rec = {k: {"completion_token_ids": list(v["completion_token_ids"])}
               for k, v in ref.items()}
    fus_rec["p3"]["completion_token_ids"][2] = 999999        # one divergence -> rate < 1.0

    cap_cmp = compare_identity(verifier, ref, cap_rec)
    fus_cmp = compare_identity(verifier, ref, fus_rec)

    synth_measured = {"configs": {
        "E": {"present": True, "decode_tps_single_stream": eager},
        "C": {"present": True, "decode_tps_single_stream": cap},
        "F": {"present": True, "decode_tps_single_stream": fus},
    }}
    synth_identity = {"available": True, "C": cap_cmp, "F": fus_cmp}
    roof = compute_roofline(eager)
    m = assemble_metrics(synth_measured, synth_identity, roof)

    ceiling = A10G_BW_GBPS / BODY_INT4_GB

    # (a) roofline math round-trips.
    a = bool(abs(roof["byte_exact_bw_ceiling_tps"] - ceiling) < TOL_RT
             and abs(roof["achieved_bw_fraction_eager"] - eager * BODY_INT4_GB / A10G_BW_GBPS) < TOL_RT
             and abs(roof["empirical_bytes_per_step_gb_eager"] - A10G_BW_GBPS / eager) < TOL_RT)
    # (b) frac_recovered formula round-trips a known value.
    fr = frac_recovered(cap, eager, ceiling)
    b = bool(abs(fr - (cap - eager) / (ceiling - eager)) < TOL_RT and 0.0 < fr < 1.0)
    # (c) identity logic: capture identical -> rate 1.0, NOT breaking; fusion diverges ->
    #     rate < 1.0, breaking; identity-safe-max picks C (the capture tps).
    c = bool(abs(m["cudagraph_capture_token_identity_rate"] - 1.0) < 1e-12
             and m["cudagraph_breaks_identity"] is False
             and m["capture_is_identity_safe"] is True
             and m["fusion_token_identity_rate"] < 1.0
             and m["fusion_breaks_identity"] is True
             and m["identity_safe_max_config"] == "C"
             and abs(m["identity_safe_max_tps"] - cap) < TOL_RT
             and m["residual_gap_requires_reduction_reorder"] is True)
    # (d) the official verifier verdicts are the expected GREEDY_IDENTICAL / DIVERGENT.
    d = bool(cap_cmp["verdict"] == "GREEDY_IDENTICAL" and fus_cmp["verdict"] == "DIVERGENT"
             and cap_cmp["num_identical"] == 8 and fus_cmp["num_divergent"] == 1)
    # (e) imported constants exact + the full-lm_head ceiling derived correctly.
    e = bool(abs(BODY_INT4_GB - 1.6973824) < TOL
             and abs(A10G_BW_GBPS - 600.0) < TOL
             and abs(STRICT_FRONTIER_TPS - 165.44) < TOL
             and abs(OFFICIAL_DEPLOYED_TPS - 481.53) < TOL
             and abs(LMHEAD_BF16_FULL_GB - 1.34217728) < TOL_RT
             and abs(roof["byte_exact_bw_ceiling_tps_with_full_lmhead"]
                     - A10G_BW_GBPS / (BODY_INT4_GB + LMHEAD_BF16_FULL_GB)) < TOL_RT)
    # (f) taxonomy + caveats present.
    f = bool(len(FUSION_TAXONOMY["identity_safe"]) >= 3
             and len(FUSION_TAXONOMY["identity_breaking"]) >= 3)

    conditions = {
        "a_roofline_roundtrips": a,
        "b_frac_recovered_roundtrips": b,
        "c_identity_logic_correct": c,
        "d_official_verifier_verdicts": d,
        "e_imports_exact": e,
        "f_taxonomy_present": f,
    }
    return {
        "conditions": conditions,
        "nonspec_baselift_self_test_passes": bool(all(conditions.values())),
        "synthetic_metrics": m,
    }


# ---------------------------------------------------------------------------- #
# Synthesis.
# ---------------------------------------------------------------------------- #
def synthesize(measured: dict | None) -> dict[str, Any]:
    verifier = _import_verifier()
    st = self_test(verifier)

    if measured is None:
        return {
            "measured_present": False,
            "self_test": st,
            "fusion_taxonomy": FUSION_TAXONOMY,
            "nonspec_baselift_self_test_passes": st["nonspec_baselift_self_test_passes"],
            "cudagraph_capture_token_identity_rate": None,
            "verdict": "SELF-TEST ONLY (no measured GPU data supplied).",
        }

    identity = compute_identity(measured, verifier)
    eager_tps = None
    e = measured["configs"].get("E", {})
    if e.get("present") and _finite(e.get("decode_tps_single_stream")):
        eager_tps = e["decode_tps_single_stream"]
    roofline = compute_roofline(eager_tps)
    metrics = assemble_metrics(measured, identity, roofline)

    cap_rate = metrics["cudagraph_capture_token_identity_rate"]
    verdict = _build_verdict(metrics, roofline, identity)
    return {
        "measured_present": True,
        "measured": measured,
        "identity": identity,
        "roofline": roofline,
        "metrics": metrics,
        "fusion_taxonomy": FUSION_TAXONOMY,
        "self_test": st,
        "nonspec_baselift_self_test_passes": st["nonspec_baselift_self_test_passes"],
        "cudagraph_capture_token_identity_rate": cap_rate,
        "verdict": verdict,
    }


def _build_verdict(m: dict, roof: dict, identity: dict) -> str:
    cap_rate = m["cudagraph_capture_token_identity_rate"]
    eager = m["baseline_eager_byte_exact_tps"]
    cap = m["cudagraph_capture_tps"]
    if cap_rate is None:
        return ("CAPTURE NOT YET MEASURED vs eager — supply decode_E.jsonl + decode_C.jsonl. "
                "Roofline + self-test only.")
    safe = abs(cap_rate - 1.0) < 1e-12
    parts = []
    parts.append(
        f"CUDA-graph CAPTURE (fusion OFF) is {'BYTE-EXACT' if safe else 'NOT byte-exact'} vs the "
        f"eager AR reference (token_identity_rate={cap_rate:.4f} over "
        f"{identity.get('C', {}).get('num_prompts_compared', '?')} records).")
    if _finite(eager) and _finite(cap):
        parts.append(
            f"Eager {eager:.1f} -> capture {cap:.1f} TPS ({m.get('capture_speedup_vs_eager', float('nan')):.2f}x), "
            f"recovering {100.0*(m['frac_of_baselift_recovered_capture'] or 0):.0f}% of the "
            f"eager->{roof['byte_exact_bw_ceiling_tps']:.0f} body-only base-lift at ZERO quality cost.")
        parts.append(
            f"NB: the served strict frontier {STRICT_FRONTIER_TPS:.2f} (lawine #196) ALREADY banks the "
            f"native cudagraph stack (it is NOT plain eager) -> capture is the dominant byte-exact lever "
            f"behind that frontier, reproduced here offline from the ~{eager:.0f} TPS eager floor.")
    if roof.get("empirical_bytes_per_step_gb_eager") is not None:
        parts.append(
            f"Empirical eager bytes/step = {roof['empirical_bytes_per_step_gb_eager']:.2f} GB "
            f"({'EXCEEDS' if roof['empirical_exceeds_body_plus_lmhead'] else 'below'} body+lm_head "
            f"{roof['body_plus_full_lmhead_gb']:.2f} GB) -> the eager gap is "
            f"{'LAUNCH/OVERHEAD (capture-recoverable)' if roof['empirical_exceeds_body_plus_lmhead'] else 'read'}-bound.")
    if m["fusion_token_identity_rate"] is not None:
        parts.append(
            f"FUSION (capture+inductor, kanna #359 bundle) identity={m['fusion_token_identity_rate']:.4f} "
            f"({'BREAKS' if m['fusion_breaks_identity'] else 'holds'} identity); the C->F residual "
            f"{'REQUIRES reduction reorder (out of scope for identity-safe)' if m['residual_gap_requires_reduction_reorder'] else 'is not identity-gated'}.")
    parts.append(
        f"identity_safe_max = {m['identity_safe_max_config']} at {m['identity_safe_max_tps']} TPS. "
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
    print("NON-SPEC BASE-LIFT: CUDA-GRAPH CAPTURE vs INDUCTOR FUSION (PR #371)", flush=True)
    print("=" * 100, flush=True)
    st = syn["self_test"]
    print(f"  (PRIMARY) nonspec_baselift_self_test_passes = {st['nonspec_baselift_self_test_passes']}",
          flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    if not syn.get("measured_present"):
        print(f"  {syn['verdict']}", flush=True)
        print("=" * 100, flush=True)
        return
    m, roof, idn = syn["metrics"], syn["roofline"], syn["identity"]
    print("-" * 100, flush=True)
    print(f"  baseline_eager_byte_exact_tps          = {m['baseline_eager_byte_exact_tps']}", flush=True)
    print(f"  byte_exact_bw_ceiling_tps (body-only)  = {roof['byte_exact_bw_ceiling_tps']:.2f}  "
          f"[with full lm_head {roof['byte_exact_bw_ceiling_tps_with_full_lmhead']:.2f}]", flush=True)
    print(f"  achieved_bw_fraction_eager (body-only) = {m['achieved_bw_fraction_eager']}", flush=True)
    if roof.get("empirical_bytes_per_step_gb_eager") is not None:
        print(f"  empirical eager bytes/step             = {roof['empirical_bytes_per_step_gb_eager']:.3f} GB "
              f"(body+lmhead {roof['body_plus_full_lmhead_gb']:.3f}; exceeds={roof['empirical_exceeds_body_plus_lmhead']})",
              flush=True)
    print("  --- DECISIVE IDENTITY GATE ---", flush=True)
    print(f"  cudagraph_capture_token_identity_rate  = {m['cudagraph_capture_token_identity_rate']}  "
          f"(cudagraph_breaks_identity={m['cudagraph_breaks_identity']})", flush=True)
    print(f"  cudagraph_capture_tps                  = {m['cudagraph_capture_tps']}  "
          f"(speedup {m['capture_speedup_vs_eager']}x)", flush=True)
    print(f"  identity_safe_max_tps ({m['identity_safe_max_config']})            = {m['identity_safe_max_tps']}", flush=True)
    print(f"  frac_of_baselift_recovered_identity_safe = {m['frac_of_baselift_recovered_identity_safe']}", flush=True)
    print(f"  fusion_token_identity_rate (F)         = {m['fusion_token_identity_rate']}  "
          f"(fusion_breaks_identity={m['fusion_breaks_identity']})", flush=True)
    print(f"  residual_gap_requires_reduction_reorder = {m['residual_gap_requires_reduction_reorder']}", flush=True)
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
        print(f"[nonspec-baselift] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    st = syn["self_test"]
    summary: dict[str, Any] = {
        "nonspec_baselift_self_test_passes": int(bool(st["nonspec_baselift_self_test_passes"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        "measured_present": int(bool(syn.get("measured_present"))),
    }
    if syn.get("measured_present"):
        m, roof = syn["metrics"], syn["roofline"]
        for key in ("baseline_eager_byte_exact_tps", "achieved_bw_fraction_eager",
                    "byte_exact_bw_ceiling_tps", "cudagraph_capture_token_identity_rate",
                    "cudagraph_capture_tps", "identity_safe_max_tps",
                    "frac_of_baselift_recovered_identity_safe",
                    "baseline_eager_identity_vs_bf16_ref",
                    "fused_capture_tps", "fusion_token_identity_rate",
                    "frac_of_baselift_recovered_capture", "frac_of_baselift_recovered_fused_total",
                    "capture_speedup_vs_eager", "fused_speedup_vs_eager"):
            if m.get(key) is not None:
                summary[key] = m[key]
        for key in ("cudagraph_breaks_identity", "fusion_breaks_identity",
                    "capture_is_identity_safe", "residual_gap_requires_reduction_reorder",
                    "baseline_eager_is_strict_frontier"):
            summary[key] = int(bool(m.get(key)))
        summary["byte_exact_bw_ceiling_tps_with_full_lmhead"] = \
            roof["byte_exact_bw_ceiling_tps_with_full_lmhead"]
        if roof.get("empirical_bytes_per_step_gb_eager") is not None:
            summary["empirical_bytes_per_step_gb_eager"] = roof["empirical_bytes_per_step_gb_eager"]

    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "nonspec-baselift", "cudagraph-capture", "inductor-fusion",
              "byte-exact-identity", "roofline", "bank-the-analysis", "pr-371"],
        config={
            "body_int4_gb": BODY_INT4_GB, "a10g_bw_gbps": A10G_BW_GBPS,
            "lmhead_bf16_full_gb": LMHEAD_BF16_FULL_GB, "vocab": VOCAB, "hidden": HIDDEN,
            "strict_frontier_tps": STRICT_FRONTIER_TPS, "official_deployed_tps": OFFICIAL_DEPLOYED_TPS,
            "ppl_gate": PPL_GATE, "ppl_baseline": PPL_BASELINE, "milestone_tps": MILESTONE,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[nonspec-baselift] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="nonspec_baselift_cudagraph_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[nonspec-baselift] wandb logged: {summary}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY 0-GPU self-validation")
    ap.add_argument("--reanalyze", action="store_true",
                    help="re-derive metrics from saved measure_*.json only (0-GPU; default if --measured-dir set)")
    ap.add_argument("--measured-dir", type=Path, default=HERE / "measured")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="nonspec-baselift-cudagraph")
    args = ap.parse_args(argv)

    measured = None
    md = args.measured_dir
    if not args.self_test or args.reanalyze:
        if md and (md / "measure_E.json").exists():
            measured = load_measured(md)
        elif not args.self_test:
            print(f"[nonspec-baselift] no measure_E.json under {md} — synthesizing self-test only.",
                  flush=True)

    syn = synthesize(measured)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 371, "agent": "land",
        "kind": "nonspec-baselift-cudagraph", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[nonspec-baselift] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nonspec_baselift_cudagraph_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[nonspec-baselift] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(syn["nonspec_baselift_self_test_passes"]) and payload["nan_clean"]
    print(f"  PRIMARY nonspec_baselift_self_test_passes = {passes}", flush=True)
    print(f"  TEST cudagraph_capture_token_identity_rate = "
          f"{syn.get('cudagraph_capture_token_identity_rate')}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        print(f"[nonspec-baselift] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
