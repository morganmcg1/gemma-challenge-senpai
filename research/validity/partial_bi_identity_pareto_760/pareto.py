#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #760 (land) -- Partial-BI identity-coverage Pareto.

Question: is FULL batch-invariance strictly NECESSARY for served-greedy 128/128
byte-exactness, or does some intermediate BI-coverage rung recover strict identity
at less than the full fire-config tax?

This card OWNS the identity-vs-coverage axis. It reuses land's own merged #748
served-greedy-identity harness (research/validity/strict_clean_served_byteexact_748/),
which already measured the realizable (VLLM_BATCH_INVARIANT x cudagraph-capture) grid on
the loadable int4 QAT proxy under TRITON_ATTN. We add (a) the realizable-coverage
taxonomy grounded in the actual vLLM 0.22.0 batch_invariant.py source, (b) the anchored
TPS per rung via fern #750's merged official-anchoring method, and (c) the verdict.

Anchoring (fern #750, research/validity/fire_bi_tax_750/compute_projection.py):
    R_int4          = ANCHOR_OFFICIAL / local_int4_qat_nospec
    anchored_literal = local_tps * R_int4
where ANCHOR_OFFICIAL = 95.463 (int4_qat a10g-small official TPS, BASELINE.md) and
local_int4_qat_nospec is the SAME-METER local TPS of the int4 QAT ckpt, BI=0, spec OFF.
Our bi0_spec0 arm IS that denominator (int4 QAT, BI=0, no spec, our api_server meter) =
95.2821, which matches fern's local anchor 95.19 to ~0.1%, so our meter is anchorable.

Because the bare api_server proxy lacks the fire optimizations (lmhead12k prune, fa2sw,
precache, onegraph), its ABSOLUTE local TPS sits at the int4_qat baseline (~95), NOT the
fire stack (~229). What transfers (land #748) is the RELATIVE BI/coverage cost. So for a
fire-stack-consistent Pareto axis we ALSO report the fire-anchored prediction:
    anchored_fire = local_tps * (FERN_BI0_ANCHORED / local_int4_qat_nospec)
which puts each rung on fern's realized 229.85 (BI=0) -> 156.95 (full BI=1) axis via the
transferable local relative cost. This is an official-anchored local PREDICTION (not an
HF-Job measurement), exactly as the card mandates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
H748 = HERE.parent / "strict_clean_served_byteexact_748"
sys.path.insert(0, str(H748))
from analyze import identity, load_arm  # noqa: E402  (reuse land's own merged #748 code)

# ---- fern #750 merged anchoring constants ----
ANCHOR_OFFICIAL = 95.463       # int4_qat a10g-small OFFICIAL tps (BASELINE.md), same ckpt
FERN_BI0_ANCHORED = 229.847    # fern #750 BI=0 fire-stack official-anchored tps (RESULTS.json)
FERN_BI1_ANCHORED = 156.949    # fern #750 FULL BI=1 official-anchored tps (the sweep target)
BAR = 126.378                  # locked int4_g128_lmhead official tps (untouched by this card)

R748 = H748 / "runs"


def arm(tag: str) -> dict:
    return load_arm(R748 / tag)


def main() -> int:
    # ---- load the realizable #748 grid (TRITON_ATTN, int4 QAT proxy) ----
    bi0_ar = arm("bi0_spec0")        # BI=0 AR  (== the int4_qat no-spec anchor denominator)
    bi0_sp = arm("bi0_spec1")        # BI=0 spec (deployed reduction order)
    bi1_ar = arm("bi1_spec0")        # BI=1 AR  (cudagraph)
    bi1_sp = arm("bi1_spec1")        # BI=1 spec (cudagraph) -- route-b "cheap path"
    bi1_ar_e = arm("bi1_spec0_eager")  # BI=1 AR  (enforce_eager)
    bi1_sp_e = arm("bi1_spec1_eager")  # BI=1 spec (enforce_eager)
    bi1_rep = arm("bi1_spec0_rep")   # determinism repeat

    local_int4_qat = bi0_ar["summary"]["output_tps"]   # 95.2821, our-meter anchor denom
    R_int4 = ANCHOR_OFFICIAL / local_int4_qat           # ~1.0019 pod->official (fern method)
    fire_scale = FERN_BI0_ANCHORED / local_int4_qat     # ~2.412 fire-stack relative-cost scale

    def anchored_literal(tps: float) -> float:
        return tps * R_int4

    def anchored_fire(tps: float) -> float:
        return tps * fire_scale

    # ---- identity per realizable rung: spec vs SAME-CONFIG AR (only spec on/off differs) ----
    id_R0 = identity(bi0_sp, bi0_ar)        # zero BI, cudagraph (deployed)
    id_R1 = identity(bi1_sp, bi1_ar)        # GEMM+reduction BI, cudagraph (route-b)
    id_R2 = identity(bi1_sp_e, bi1_ar_e)    # GEMM+reduction BI + capture-sym (eager PROBE)
    id_floor = identity(bi1_rep, bi1_ar)    # determinism floor (same config twice)
    id_xchk = identity(bi1_ar, bi0_ar)      # AR-only BI x-check (spec OFF both)

    def rung(name, op_families, attn_split, capture_sym, deployable, realizable,
             idd, spec_tps, source):
        n, tot = idd["n_match"], idd["n_total"]
        return {
            "rung": name,
            "bi_op_families": op_families,
            "attention_split_pinned": attn_split,
            "cudagraph_capture_symmetric": capture_sym,
            "deployable": deployable,
            "realizable_with_current_toggles": realizable,
            "source": source,
            "identity_k": n, "identity_n": tot, "identity_frac": round(n / tot, 4),
            "per_token_flip_hazard": idd["per_token_flip_hazard"],
            "local_tps_spec": spec_tps,
            "anchored_tps_fire": (round(anchored_fire(spec_tps), 2)
                                  if spec_tps is not None else None),
            "anchored_tps_literal_int4qat": (round(anchored_literal(spec_tps), 2)
                                             if spec_tps is not None else None),
        }

    FAMILIES_BI1 = ["aten::mm", "aten::addmm", "aten::matmul", "aten::linear",
                    "aten::bmm", "aten::_log_softmax", "aten::softmax", "aten::mean.dim"]

    rungs = [
        rung("R0  zero-BI (deployed)", [], False, False, True, True,
             id_R0, bi0_sp["summary"]["output_tps"], "land #748 (mine)"),
        rung("R1  GEMM+reduction BI [route-b cheap]", FAMILIES_BI1, False, False, True, True,
             id_R1, bi1_sp["summary"]["output_tps"], "land #748 (mine)"),
        rung("R2  GEMM+reduction BI + capture-sym [EAGER PROBE, non-deployable]",
             FAMILIES_BI1, False, True, False, True,
             id_R2, bi1_sp_e["summary"]["output_tps"], "land #748 (mine)"),
    ]

    # PR-body-provided curve anchors (sanctioned in the #760 body; NOT read from any branch).
    given = [
        {"rung": "G1  attention-split only (num_splits=1)",
         "bi_op_families": ["attention num_splits=1 (no aten GEMM/reduction overrides)"],
         "attention_split_pinned": True, "cudagraph_capture_symmetric": False,
         "deployable": True, "realizable_with_current_toggles": False,
         "source": "lawine #755 (PR-body curve point)",
         "identity_k": 24, "identity_n": 128, "identity_frac": 0.1875,
         "per_token_flip_hazard": None, "local_tps_spec": None,
         "anchored_tps_fire": 235.99, "anchored_tps_literal_int4qat": None},
        {"rung": "G2  FULL BI (fire stack)",
         "bi_op_families": FAMILIES_BI1 + ["attention num_splits=1", "capture-aligned verify"],
         "attention_split_pinned": True, "cudagraph_capture_symmetric": True,
         "deployable": True, "realizable_with_current_toggles": False,
         "source": "fern #750 (PR-body sweep target)",
         "identity_k": 128, "identity_n": 128, "identity_frac": 1.0,
         "per_token_flip_hazard": None, "local_tps_spec": None,
         "anchored_tps_fire": FERN_BI1_ANCHORED, "anchored_tps_literal_int4qat": None},
    ]

    # ---- verdict: min REALIZABLE coverage achieving literal 128/128 ----
    realized_128 = [r for r in rungs if r["identity_k"] == r["identity_n"]]
    deployable_realized_128 = [r for r in realized_128 if r["deployable"]]
    max_realizable = max(rungs, key=lambda r: r["identity_k"])
    max_deployable_realizable = max((r for r in rungs if r["deployable"]),
                                    key=lambda r: r["identity_k"])

    full_bi_necessary = 0 if realized_128 else 1
    if realized_128:
        cheapest = max(realized_128, key=lambda r: r["anchored_tps_fire"] or -1)
        min_strict_bi_tps = cheapest["anchored_tps_fire"]
        strict_rung = cheapest["rung"]
    else:
        min_strict_bi_tps = FERN_BI1_ANCHORED   # full-BI fire anchor (per #760 instruction)
        strict_rung = "G2  FULL BI (fire stack)"

    out = {
        "card": "partial_bi_identity_pareto_760",
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "anchoring": {
            "method": "fern #750 official-anchoring (merged compute_projection.py)",
            "ANCHOR_OFFICIAL_int4qat": ANCHOR_OFFICIAL,
            "local_int4_qat_nospec_ourmeter": round(local_int4_qat, 4),
            "R_int4_pod_to_official": round(R_int4, 6),
            "fern_local_int4_qat_sglang": 95.19,
            "meter_agreement_pct": round(abs(local_int4_qat - 95.19) / 95.19 * 100, 4),
            "fire_stack_scale": round(fire_scale, 5),
            "FERN_BI0_ANCHORED": FERN_BI0_ANCHORED,
            "FERN_BI1_ANCHORED": FERN_BI1_ANCHORED,
            "note": ("bare api_server proxy sits at the int4_qat baseline (~95 local), not the "
                     "fire stack (~229); ABSOLUTE tps does not transfer, the RELATIVE BI/coverage "
                     "cost does (land #748). anchored_tps_fire = official-anchored local PREDICTION "
                     "via the transferable relative cost on fern's realized BI=0->BI=1 axis."),
        },
        "determinism_floor": {"k": id_floor["n_match"], "n": id_floor["n_total"],
                              "ok": bool(id_floor["n_match"] == id_floor["n_total"])},
        "ar_only_bi_xcheck": {"k": id_xchk["n_match"], "n": id_xchk["n_total"]},
        "rungs_realizable": rungs,
        "rungs_given_anchors": given,
        "verdict": {
            "full_bi_necessary": full_bi_necessary,
            "min_strict_bi_tps": round(min_strict_bi_tps, 2),
            "strict_rung": strict_rung,
            "max_realizable_identity": f"{max_realizable['identity_k']}/128 "
                                       f"({max_realizable['rung']})",
            "max_deployable_realizable_identity":
                f"{max_deployable_realizable['identity_k']}/128 "
                f"({max_deployable_realizable['rung']})",
            "n_realizable_rungs_hitting_128": len(realized_128),
            "n_deployable_realizable_hitting_128": len(deployable_realized_128),
        },
        "not_realizable_out_of_scope": {
            "per_op_family_BI_subsets": (
                "vLLM 0.22.0 enable_batch_invariant_mode() registers all op families "
                "(mm/addmm/matmul/linear/bmm/log_softmax/softmax/mean) in ONE monolithic call "
                "gated by a single VLLM_BATCH_INVARIANT bool; there is NO per-family env toggle, "
                "so 'matmul-only' / 'matmul+attention' / 'matmul+reduction' as separate rungs "
                "require patching enable_batch_invariant_mode -> OUT OF SCOPE."),
            "attention_split_BI_under_available_backend": (
                "ARCHITECTURALLY FORCED to TRITON_ATTN. Boot-log proof (runs/boot_FLASHINFER.log, "
                "config.py:100): 'Gemma4 model has heterogeneous head dimensions (head_dim=256, "
                "global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend "
                "numerical divergence.' -- requesting VLLM_ATTENTION_BACKEND=FLASHINFER (or any FA "
                "backend) is SILENTLY overridden to TRITON_ATTN for Gemma-4. And under TRITON_ATTN "
                "the parallel softmax-segment reduction is fixed at NUM_PAR_SOFTMAX_SEGMENTS=16 "
                "UNCONDITIONALLY -- VLLM_BATCH_INVARIANT never appears in triton_attn.py, so BI "
                "does NOT pin the attention split. The FA/flashinfer backends whose BI path sets "
                "num_splits=1 / disable_split_kv (flash_attn.py, flashinfer.py:559) are unreachable "
                "for this model. So an attention-split-coverage rung needs a different model, or a "
                "patch to the forced-backend logic / the TRITON segment count -> OUT OF SCOPE. "
                "(flashinfer 0.6.11 IS installed and boots, but is overridden to TRITON_ATTN, so a "
                "flashinfer-request arm reproduces R1=21/128 -- redundant, not run.)"),
        },
    }
    (HERE / "runs" / "pareto.json").write_text(json.dumps(out, indent=2))

    # ---- human-readable Pareto table ----
    print("=" * 100)
    print("PARTIAL-BI IDENTITY-COVERAGE PARETO (#760)")
    print("=" * 100)
    print(f"anchor: ANCHOR_OFFICIAL={ANCHOR_OFFICIAL} / local_int4_qat={local_int4_qat:.4f} "
          f"-> R_int4={R_int4:.5f}; fire_scale={fire_scale:.4f} (vs fern sglang 95.19, "
          f"agree {out['anchoring']['meter_agreement_pct']}%)")
    print("-" * 100)
    hdr = (f"{'rung':<48}{'identity':>10}{'attn?':>7}{'capSym?':>8}"
           f"{'deploy?':>8}{'anch_fire':>11}")
    print(hdr)
    print("-" * 100)
    for r in rungs + given:
        ident = f"{r['identity_k']}/{r['identity_n']}"
        af = r["anchored_tps_fire"]
        print(f"{r['rung']:<48}{ident:>10}{str(r['attention_split_pinned']):>7}"
              f"{str(r['cudagraph_capture_symmetric']):>8}{str(r['deployable']):>8}"
              f"{(f'{af:.1f}' if af is not None else '-'):>11}")
    print("-" * 100)
    print(f"determinism floor: {id_floor['n_match']}/{id_floor['n_total']} "
          f"(stack bit-reproducible within-config: {id_floor['n_match']==id_floor['n_total']})")
    print(f"AR-only BI x-check: {id_xchk['n_match']}/{id_xchk['n_total']}")
    print("=" * 100)
    v = out["verdict"]
    print(f"VERDICT: full_bi_necessary={v['full_bi_necessary']}  "
          f"min_strict_bi_tps={v['min_strict_bi_tps']}  ({v['strict_rung']})")
    print(f"  max realizable identity:            {v['max_realizable_identity']}")
    print(f"  max DEPLOYABLE realizable identity: {v['max_deployable_realizable_identity']}")
    print(f"  realizable rungs hitting 128/128:   {v['n_realizable_rungs_hitting_128']}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
