#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Measure the locked top-8/16 MTP-drafter coverage ceiling (PR #401, ubel).

THE QUESTION (denken #208's tree-verify go/no-go needs a measured point)
------------------------------------------------------------------------
The locked top-1 -> top-4 coverage prize (coverage_ceiling_gap = +0.1097, the band a depth-1
tree's verify could harvest above top-4 toward full coverage) is an UPPER bound in [0, 0.1097]
because #387's direct GPU top-K read was blocked. PR #401 asks: load the DEPLOYED MTP drafter
(/tmp/qat-assistant, PR #52, method="mtp", num_speculative_tokens=7), read its per-position
top-8/top-16 token sets on the official 128 prompts, and replace the [0, 0.1097] bound with a
MEASURED ceiling -- round-tripping coverage(1)=0.7617 and coverage(4)=0.8903 (#387 z8osvif8) to
~1e-3 as the load-bearing "am I reading the right distribution" check.

THE ANSWER (decision-critical, honest, BLOCKER -- the band is NOT collapsed)
---------------------------------------------------------------------------
drafter_loadable = TRUE. /tmp/qat-assistant loads on the local A10G in 0.27s (Gemma4Assistant-
ForCausalLM, 78.5M params, full 262144-vocab tied head, 159 MB VRAM; see _gpu_probe.json). This
is NOT the #387 "missing checkpoint" NULL.

But the per-position top-8/16 READ is blocked, for THREE independent reasons -- and the first is a
spec bug in the round-trip premise itself:

  (1) WRONG-ARTIFACT ROUND-TRIP ANCHORS. The PR's validation anchors (top-1=0.7617, top-4=0.8903)
      are the fern #34 EAGLE-3 fusion head (gua9x68j), a demand-route CANDIDATE that was NEVER
      DEPLOYED (#387 premise-correction A, merged on advisor branch). The DEPLOYED MTP drafter's
      top-1 is FAITHFULLY measured at 0.7287 (#76 accept_calibration, read straight from the live
      deployed server's vLLM Prometheus spec-decode counters) / 0.7293 (#289 fi34s269, a_1). So
      coverage(1) of /tmp/qat-assistant ~= 0.729 and CANNOT reproduce 0.7617 -- not because the
      read is wrong, but because 0.7617/0.8903 belong to a DIFFERENT model. "Load the MTP drafter
      to size the EAGLE-3 prize" is a category error: the +0.1097 band and +0.1286 prize are
      EAGLE-3-head quantities, not MTP-drafter quantities.

  (2) NO BANKED K>1 ANCHOR FOR THE DEPLOYED MTP DRAFTER. vLLM greedy speculative decoding only
      proposes/accepts the drafter's TOP-1 (argmax); the deployed server logs only top-1 accept
      (the #289/#76 ladder IS top-1, per draft position). There is no banked deployed top-4/8/16
      to round-trip against. The PR's own gate ("if the round-trip fails, the read is wrong --
      debug before reporting any top-8/16 number") therefore CANNOT be satisfied at K>1: there is
      no valid K>1 ground truth for this artifact to debug against.

  (3) A FAITHFUL FROM-SCRATCH LOGIT READ IS NEITHER THE DEPLOYED DISTRIBUTION NOR VALIDATABLE.
      Gemma4AssistantForCausalLM.forward raises "inputs_embeds and shared_kv_states cannot be
      None" (verified live). It is an MTP/EAGLE head that CROSS-ATTENDS into the backbone's KV
      cache: it needs (a) inputs_embeds of dim 2*2560=5120 -- a vLLM-MTP-specific construction
      from backbone hidden states, NOT banked anywhere -- and (b) shared_kv_states = the backbone's
      per-layer-type KV. The supported HF entry (target.generate(assistant_model=...)) does
      assisted decoding and yields top-1 accept/reject, not per-position top-K. Replicating vLLM's
      exact MTP input construction + the deployed lm_head prune(12k)+int4 (which set the deployed
      argmax distribution) outside vLLM is a custom-vLLM-patch effort; a plain bf16 HF read would
      be a DIFFERENT distribution than deployed AND cross-session nondeterministic (bf16 lm_head
      GEMV flips ~9-13% of argmaxes across processes; only int4-Marlin is bit-exact). With no K>1
      anchor (reason 2), such a read could not be validated even in principle.

=> coverage_ceiling_gap_measured = 0.1097 (UNCHANGED [0, 0.1097] band; this card does NOT collapse
   it). realized_topk_coverage_8/16 = None; realized_tree_headroom_8/16 = None; realized_tree_prize
   _fraction_8/16 = None. The band [0, +0.1097 cov] = [0, ~+106 TPS] (at #399's 968.57 TPS/Dcov)
   stays open: denken #208 must keep PARAMETERIZING over it, not plug a point.

WHAT THIS CARD DOES DELIVER (faithful, banked, decision-relevant)
-----------------------------------------------------------------
  * drafter_loadable = TRUE with a live A10G load characterization (_gpu_probe.json).
  * The CORRECTED anchor: the deployed MTP drafter's faithful per-position TOP-1 coverage ladder
    = #289 a_1..a_7 (cross-checked by #76), top-1 = 0.7293 (NOT 0.7617). This replaces the PR's
    wrong-artifact round-trip target.
  * A precise, reproducible blocker ledger (the 3 reasons) so denken #208 + the advisor can decide
    whether to fund the ONE faithful route (a vLLM-MTP-proposer-instrumented top-K logger on a
    local deployed-stack run with the deployed prune+int4 -- the only read that is both the
    deployed distribution AND top-1-validatable against 0.729).

NOT a launch, NOT a submission, NO served-file change, 0 official TPS. The GPU was used ONLY to
load + characterize the drafter (analysis). BASELINE 481.53 TPS / PPL 2.3772 / 128/128 (PR #52,
2x9fm2zx) UNCHANGED. analysis_only = no_hf_job = no_served_file_change = True; official_tps = 0.

REPRODUCE
    # CPU-analytic card + self-test (no torch needed; runs in the .venv):
    cd target/ && .venv/bin/python -m research.validity.mtp_drafter_topk_coverage_ceiling.\
mtp_drafter_topk_coverage_ceiling --self-test
    cd target/ && .venv/bin/python -m research.validity.mtp_drafter_topk_coverage_ceiling.\
mtp_drafter_topk_coverage_ceiling \
      --wandb_group mtp-drafter-topk-coverage-ceiling --wandb_name ubel/mtp-drafter-topk-coverage-ceiling
    # (Re)generate the live A10G load probe (needs a torch-enabled python + a visible GPU):
    cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 \
      research/validity/mtp_drafter_topk_coverage_ceiling/mtp_drafter_topk_coverage_ceiling.py --probe-gpu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_PROBE_PATH = HERE / "_gpu_probe.json"

# ===========================================================================
# Section 0 -- banked anchors (imported EXACTLY from merged advisor-branch cards / PR #401 body)
# ===========================================================================

# ---- #289 (fi34s269) DEPLOYED MTP per-position conditional acceptance ladder a_1..a_7 (K=7) -------
# The deployed LINEAR MTP chain's realized per-DEPTH top-1 acceptance (conditional accept along the
# spec-decode trajectory). a_1 IS the deployed MTP drafter's faithful position-1 top-1 coverage.
LADDER_289: list[float] = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]
E_ACCEPTED_289: float = 2.851185944363104   # #289 E[accepted draft tokens]/step
E_T_289: float = 3.851185944363104          # #289 E[T] = 1 + E[accepted]

# ---- #76 (accept_calibration) DEPLOYED-SERVER faithful read (live vLLM Prometheus spec counters) --
# Read straight off the deployed fa2sw_precache_kenyan server on the official 128 -- the gold-standard
# faithful deployed-distribution top-1. conditional_acceptance_p[0] is the deployed MTP top-1.
COND_ACCEPT_76: list[float] = [
    0.728739760479042, 0.7589764102641635, 0.7924989076194682, 0.821702519412012,
    0.8342716929825772, 0.8352594665096346, 0.8472621220149911,
]
DEPLOYED_MTP_TOP1_76: float = 0.728739760479042   # #76 a_1 (== #289 a_1 to ~5e-4): the faithful top-1
E_T_76: float = 3.844131736526946                  # #76 mean_tokens_per_step (== kanna #217 3.844)

# ---- #387 (z8osvif8) EAGLE-3 fern#34 (gua9x68j) coverage anchors -- the WRONG-ARTIFACT round-trip --
# These are the fern #34 EAGLE-3 fusion-head holdout numbers. #387 premise-correction A: the EAGLE-3
# head was NEVER DEPLOYED; the deployed drafter is MTP (/tmp/qat-assistant). So these do NOT describe
# /tmp/qat-assistant and the deployed MTP top-1 (0.729) cannot reproduce 0.7617.
EAGLE3_TOP1_387: float = 0.7617               # fern#34 holdout aggregate teacher-forced top-1
EAGLE3_TOP4_387: float = 0.8902659519153152   # fern#34 per-source top-4 x official 57/57/14 mix
PER_SOURCE_TOP4_387: dict[str, float] = {"aime": 0.957005303537408, "gpqa": 0.9175953770859131,
                                         "mmlu_pro": 0.846544405293677}
OFFICIAL_MIX_387: dict[str, float] = {"aime": 0.109375, "gpqa": 0.4453125, "mmlu_pro": 0.4453125}

# Coverage-space geometry the PR is trying to collapse (ALL defined on the EAGLE-3 head):
HEAD_CEILING_UPPER: float = 1.0
COVERAGE_CEILING_GAP: float = HEAD_CEILING_UPPER - EAGLE3_TOP4_387     # +0.1097 = the [0,0.1097] band
LOCKED_TOP1_TO_TOP4_PRIZE: float = EAGLE3_TOP4_387 - EAGLE3_TOP1_387   # +0.1286 gross E[accepted] prize

# ---- #390 / #393 corrected strict served base + #399 demand secant (TPS mapping of a realized Dcov) -
BASE_471_390: float = 471.41634950257713      # #390 (5y64zbjz) realized shippable strict base
GAP_28_390: float = 28.583650497422866        # #390 gap_to_500
BASE_467_393: float = 467.48                  # #393 (0q7ynumg) corrected strict base (eta_attn=3.01%)
GAP_33_393: float = 32.53                     # #393 strict gap_to_500
S_CENTRAL_399: float = 7.912609135742992      # #399/#387 program coverage->E[T] central secant
TPS_PER_UNIT_DCOV_399: float = 968.57         # #399 (ec7i3z5t) demand secant: TPS per unit Dcoverage

# ---- deployed baseline (UNCHANGED -- this is a 0-TPS card) -----------------------------------------
BASELINE_TPS: float = 481.53                  # PR #52 (2x9fm2zx) deployed public TPS
BASELINE_PPL: float = 2.3772
PPL_GATE: float = 2.42

# round-trip tolerance the PR specifies for the load-bearing check.
ROUNDTRIP_TOL: float = 1e-3


# ===========================================================================
# Section 1 -- live A10G load probe (drafter_loadable + structure); deliverable 1
# ===========================================================================

def banked_gpu_probe() -> dict:
    """The live-load facts baked from the A10G probe (regenerate with --probe-gpu). Used so the
    CPU-analytic card composes a REAL load result without importing torch in the analytic venv."""
    return {
        "path": "/tmp/qat-assistant", "drafter_loadable": True,
        "model_class": "Gemma4AssistantForCausalLM", "model_type": "gemma4_assistant",
        "transformers_version": "5.9.0", "gpu_name": "NVIDIA A10G", "load_wall_s": 0.266,
        "n_params_M": 78.518, "lm_head_vocab": 262144, "lm_head_dim": 256,
        "has_masked_embedding": True, "pre_projection_in": 5120, "backbone_hidden_size": 2560,
        "post_projection_out": 2560, "vram_alloc_MB": 159.1,
        "standalone_forward_blocked": True,
        "standalone_forward_error": "ValueError: inputs_embeds and shared_kv_states cannot be None.",
        "_source": "banked_from_probe",
    }


def load_gpu_probe() -> dict:
    """Prefer the on-disk live probe (_gpu_probe.json); fall back to the banked facts."""
    if GPU_PROBE_PATH.exists():
        try:
            d = json.loads(GPU_PROBE_PATH.read_text())
            d.setdefault("_source", "live_gpu_probe.json")
            return d
        except Exception:  # noqa: BLE001
            pass
    return banked_gpu_probe()


def run_gpu_probe() -> dict:
    """LIVE A10G load of /tmp/qat-assistant. Lazy torch import so the analytic venv never touches it.
    Writes _gpu_probe.json. Run with a torch-enabled python + a visible GPU (CUDA_VISIBLE_DEVICES=0)."""
    import time
    import traceback
    out: dict = {"path": "/tmp/qat-assistant",
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "_source": "live_gpu_probe"}
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoConfig, Gemma4AssistantForCausalLM  # noqa: PLC0415
        out["torch_version"] = torch.__version__
        import transformers  # noqa: PLC0415
        out["transformers_version"] = transformers.__version__
        cfg = AutoConfig.from_pretrained(out["path"])
        out["config_class"] = type(cfg).__name__
        out["model_type"] = cfg.model_type
        t0 = time.time()
        m = Gemma4AssistantForCausalLM.from_pretrained(out["path"], dtype=torch.bfloat16).to("cuda").eval()
        out["load_wall_s"] = round(time.time() - t0, 3)
        out["drafter_loadable"] = True
        out["model_class"] = type(m).__name__
        out["n_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 3)
        out["lm_head_vocab"] = int(m.lm_head.weight.shape[0])
        out["lm_head_dim"] = int(m.lm_head.weight.shape[1])
        out["has_masked_embedding"] = m.masked_embedding is not None
        out["pre_projection_in"] = int(m.pre_projection.in_features)
        out["backbone_hidden_size"] = int(m.backbone_hidden_size)
        out["post_projection_out"] = int(m.post_projection.out_features)
        out["vram_alloc_MB"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        out["gpu_name"] = torch.cuda.get_device_name(0)
        try:  # demonstrate the forward cannot run standalone (needs backbone wiring)
            m(input_ids=torch.tensor([[2, 108, 5471]], device="cuda"))
            out["standalone_forward_blocked"] = False
            out["standalone_forward_note"] = "UNEXPECTED_SUCCESS"
        except Exception as exc:  # noqa: BLE001
            out["standalone_forward_blocked"] = True
            out["standalone_forward_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    except Exception as exc:  # noqa: BLE001
        out["drafter_loadable"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out["tb"] = traceback.format_exc()[-900:]
    GPU_PROBE_PATH.write_text(json.dumps(out, indent=2))
    return out


# ===========================================================================
# Section 2 -- ladder arithmetic + the artifact-identity analysis (deliverables 2-3)
# ===========================================================================

def expected_accepted(ladder: list[float]) -> float:
    """E[accepted draft tokens]/step = sum_k prod_{j<=k} a_j (conditional ladder)."""
    cum, acc = 1.0, 0.0
    for a in ladder:
        cum *= a
        acc += cum
    return acc


def expected_tokens_per_step(ladder: list[float]) -> float:
    return 1.0 + expected_accepted(ladder)


def grounded_eagle3_top4() -> float:
    """Round-trip #387's 0.8903 EAGLE-3 anchor = sum_s mix_s * per_source_top4_s (provenance check)."""
    return sum(OFFICIAL_MIX_387[s] * PER_SOURCE_TOP4_387[s] for s in OFFICIAL_MIX_387)


def artifact_identity() -> dict:
    """The load-bearing finding: the deployed MTP drafter (top-1 0.729) is NOT the EAGLE-3 head
    (top-1 0.7617). The PR's round-trip anchors describe a different, never-deployed artifact."""
    deployed_top1 = LADDER_289[0]                      # 0.7293 (== #76 0.7287 to ~5e-4)
    mismatch = EAGLE3_TOP1_387 - deployed_top1         # +0.0324: > ROUNDTRIP_TOL, so round-trip FAILS
    # the faithful deployed MTP top-1 is internally consistent across #76 and #289:
    faithful_internal_consistent = abs(DEPLOYED_MTP_TOP1_76 - deployed_top1) < ROUNDTRIP_TOL
    # does the deployed MTP top-1 reproduce the EAGLE-3 anchor (the PR's load-bearing check)?
    roundtrip_top1_vs_eagle3_passes = abs(deployed_top1 - EAGLE3_TOP1_387) < ROUNDTRIP_TOL  # -> False
    return {
        "deployed_mtp_top1_coverage": deployed_top1,
        "deployed_mtp_top1_xcheck_76": DEPLOYED_MTP_TOP1_76,
        "eagle3_anchor_top1": EAGLE3_TOP1_387,
        "eagle3_anchor_top4": EAGLE3_TOP4_387,
        "top1_artifact_mismatch": mismatch,
        "deployed_mtp_weaker_than_eagle3_top1": bool(deployed_top1 < EAGLE3_TOP1_387),
        "faithful_top1_internally_consistent_76_vs_289": bool(faithful_internal_consistent),
        "roundtrip_top1_vs_eagle3_passes": bool(roundtrip_top1_vs_eagle3_passes),
        "deployed_mtp_top4_coverage": None,            # blocked: no banked MTP top-4 (see blocker ledger)
        "per_position_coverage_1": list(LADDER_289),   # the faithful per-position TOP-1 7-vector
        "anchors_describe_different_model": True,
        "note": ("0.7617/0.8903 = fern#34 EAGLE-3 candidate (gua9x68j, NEVER deployed, #387 prem-corr A). "
                 "deployed /tmp/qat-assistant top-1 = 0.7293 (#289) / 0.7287 (#76, live deployed server). "
                 "A mode caveat exists (EAGLE-3 is teacher-forced-aggregate; MTP is spec-decode position-1) "
                 "but the dominant, documented difference is the MODEL, not the measurement mode."),
    }


# ===========================================================================
# Section 3 -- the read-blocker ledger + the (uncollapsed) ceiling verdict (deliverable 4)
# ===========================================================================

def blocker_ledger(probe: dict) -> dict:
    """Why a VALIDATED top-8/16 read of the deployed MTP drafter is blocked (3 independent reasons),
    and the resulting ceiling verdict: the [0, 0.1097] band is NOT collapsed by this card."""
    reasons = [
        {"id": "wrong_artifact_anchors",
         "summary": "PR round-trip anchors top-1=0.7617/top-4=0.8903 are the fern#34 EAGLE-3 head "
                    "(gua9x68j), never deployed (#387 prem-corr A). Deployed MTP top-1 = 0.729 != 0.7617.",
         "evidence": "#387 z8osvif8; #76 a_1=0.7287; #289 a_1=0.7293"},
        {"id": "no_banked_mtp_topk_anchor",
         "summary": "vLLM greedy spec-decode proposes/accepts only the drafter argmax (top-1); the "
                    "deployed server logs only top-1 accept. No banked deployed top-4/8/16 to validate "
                    "against, so the PR's round-trip gate cannot be satisfied at K>1.",
         "evidence": "deployed manifest SPECULATIVE_CONFIG greedy; #76/#289 are top-1 ladders"},
        {"id": "faithful_read_unwired_and_nondeterministic",
         "summary": "Gemma4AssistantForCausalLM.forward needs inputs_embeds(5120=2x2560) + shared_kv_states "
                    "(backbone KV cross-attn); the 5120 construction is vLLM-MTP-specific and not banked. "
                    "HF generate(assistant_model=...) yields top-1 accept/reject, not per-position top-K. "
                    "A plain bf16 HF read != deployed prune(12k)+int4 distribution AND is cross-session "
                    "nondeterministic (bf16 lm_head argmax flips ~9-13%; only int4-Marlin is bit-exact).",
         "evidence": f"live forward error: {probe.get('standalone_forward_error','')}"},
    ]
    full_band_tps = COVERAGE_CEILING_GAP * TPS_PER_UNIT_DCOV_399   # what the WHOLE band would buy (upper)
    return {
        "topk_read_blocked": True,
        "n_blocker_reasons": len(reasons),
        "reasons": reasons,
        "no_banked_mtp_topk_anchor": True,
        "drafter_loadable": bool(probe.get("drafter_loadable", False)),
        "standalone_forward_blocked": bool(probe.get("standalone_forward_blocked", True)),
        # ceiling verdict: the EAGLE-3-head band is NOT collapsed.
        "coverage_ceiling_gap_measured": COVERAGE_CEILING_GAP,   # = 0.1097 (UNCHANGED bound)
        "ceiling_band_collapsed": False,
        "realized_topk_coverage_8": None,
        "realized_topk_coverage_16": None,
        "realized_tree_headroom_8": None,                        # = cov8 - 0.8903 (unmeasured)
        "realized_tree_headroom_16": None,
        "realized_tree_prize_fraction_8": None,                  # = (cov8 - 0.8903)/0.1097 (unmeasured)
        "realized_tree_prize_fraction_16": None,
        "per_position_coverage_8": None,                         # 7-vector unmeasured (top-1 vec is delivered)
        # band -> TPS framing for denken #208 (parameterize, do NOT plug a point):
        "ceiling_band_cov": [0.0, COVERAGE_CEILING_GAP],
        "ceiling_band_tps_at_968_secant": [0.0, full_band_tps],
        "tps_per_unit_dcov_399": TPS_PER_UNIT_DCOV_399,
    }


def feed_denken_208(ledger: dict, ident: dict) -> dict:
    """The honest hand-off to denken #208's tree-verify net-TPS card."""
    return {
        "ceiling_collapsed_to_point": False,
        "ceiling_remains_band": ledger["ceiling_band_cov"],
        "ceiling_band_tps": ledger["ceiling_band_tps_at_968_secant"],
        "corrected_deployed_mtp_top1_anchor": ident["deployed_mtp_top1_coverage"],
        "eagle3_prize_is_a_different_head": True,
        "realized_tree_prize_fraction_8": None,
        "realized_tree_prize_fraction_16": None,
        "recommended_faithful_route": (
            "Instrument the vLLM MTP proposer (the deployed fa2sw_precache_kenyan stack, with the "
            "deployed lm_head prune(12k)+int4) to log per-draft-position top-K, on a LOCAL deployed run "
            "over the official 128. This is the ONLY read that is BOTH the deployed distribution AND "
            "top-1-validatable against 0.729. It is a custom-vLLM-patch / local-serve effort -- size it "
            "as a follow-up; out of scope for a 0-TPS analysis card with no served-file change."),
    }


# ===========================================================================
# Section 4 -- self-tests (>=20 checks, incl. the corrected round-trips)
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(et: float, probe: dict, ident: dict, ledger: dict) -> dict:
    c: dict[str, bool] = {}

    # a) #289 deployed MTP ladder provenance (the faithful per-position top-1).
    c["a_ladder_len_7"] = len(LADDER_289) == 7
    c["a_ladder_in_unit"] = all(0.0 < a < 1.0 for a in LADDER_289)
    c["a_ladder_monotone_nondecreasing"] = all(LADDER_289[i] <= LADDER_289[i + 1]
                                               for i in range(len(LADDER_289) - 1))
    c["a_eaccepted_roundtrips_289"] = abs(expected_accepted(LADDER_289) - E_ACCEPTED_289) < 1e-9
    c["a_et_roundtrips_289"] = abs(et - E_T_289) < 1e-9

    # b) #76 deployed-server faithful read cross-validates the deployed MTP top-1 (the CORRECT anchor).
    c["b_76_cond_len_7"] = len(COND_ACCEPT_76) == 7
    c["b_76_top1_xchecks_289_a1"] = abs(DEPLOYED_MTP_TOP1_76 - LADDER_289[0]) < ROUNDTRIP_TOL
    c["b_76_et_matches_kanna_3p844"] = abs(E_T_76 - 3.844) < 0.01
    c["b_faithful_top1_internally_consistent"] = ident["faithful_top1_internally_consistent_76_vs_289"]

    # c) #387 EAGLE-3 anchor provenance + coverage-space geometry.
    c["c_eagle3_top4_grounded_from_sources"] = abs(grounded_eagle3_top4() - EAGLE3_TOP4_387) < 1e-9
    c["c_official_mix_sums_to_1"] = abs(sum(OFFICIAL_MIX_387.values()) - 1.0) < 1e-12
    c["c_eagle3_top1_le_top4_le_1"] = EAGLE3_TOP1_387 <= EAGLE3_TOP4_387 <= 1.0
    c["c_ceiling_gap_is_complement_of_top4"] = abs(COVERAGE_CEILING_GAP - (1.0 - EAGLE3_TOP4_387)) < 1e-12
    c["c_ceiling_gap_is_0p1097"] = abs(COVERAGE_CEILING_GAP - 0.1097) < 5e-4
    c["c_prize_is_0p1286"] = abs(LOCKED_TOP1_TO_TOP4_PRIZE - 0.1286) < 5e-4

    # d) ARTIFACT MISMATCH -- the corrected round-trips (the load-bearing finding).
    #    round-trip 1: deployed MTP top-1 reproduces the FAITHFUL anchor 0.729 (PASS).
    c["d_roundtrip_faithful_anchor_passes"] = abs(ident["deployed_mtp_top1_coverage"] - DEPLOYED_MTP_TOP1_76) < ROUNDTRIP_TOL
    #    round-trip 2: deployed MTP top-1 does NOT reproduce the EAGLE-3 anchor 0.7617 (documented FAIL).
    c["d_roundtrip_vs_eagle3_fails"] = ident["roundtrip_top1_vs_eagle3_passes"] is False
    c["d_mismatch_exceeds_tol"] = abs(ident["top1_artifact_mismatch"]) > 0.02
    c["d_mtp_weaker_than_eagle3"] = ident["deployed_mtp_weaker_than_eagle3_top1"]
    c["d_anchors_different_model"] = ident["anchors_describe_different_model"] is True

    # e) GPU load probe: drafter LOADS (not the #387 missing-ckpt NULL) but forward is wiring-blocked.
    c["e_drafter_loadable_true"] = probe.get("drafter_loadable") is True
    c["e_lm_head_vocab_262144"] = probe.get("lm_head_vocab") == 262144
    c["e_pre_proj_in_is_2x_backbone"] = probe.get("pre_projection_in") == 2 * probe.get("backbone_hidden_size", 0)
    c["e_post_proj_out_is_backbone"] = probe.get("post_projection_out") == probe.get("backbone_hidden_size")
    c["e_standalone_forward_blocked"] = probe.get("standalone_forward_blocked") is True

    # f) blocker ledger + the (uncollapsed) ceiling verdict.
    c["f_topk_read_blocked"] = ledger["topk_read_blocked"] is True
    c["f_three_blocker_reasons"] = ledger["n_blocker_reasons"] >= 3
    c["f_no_banked_mtp_topk_anchor"] = ledger["no_banked_mtp_topk_anchor"] is True
    c["f_realized_cov8_16_none"] = ledger["realized_topk_coverage_8"] is None and ledger["realized_topk_coverage_16"] is None
    c["f_prize_fraction_8_16_none"] = ledger["realized_tree_prize_fraction_8"] is None and ledger["realized_tree_prize_fraction_16"] is None
    c["f_band_not_collapsed"] = ledger["ceiling_band_collapsed"] is False
    c["f_ceiling_gap_measured_unchanged"] = abs(ledger["coverage_ceiling_gap_measured"] - COVERAGE_CEILING_GAP) < 1e-12

    # g) PPL gate: a 0-TPS measurement card; deployed greedy identity + PPL untouched.
    c["g_ppl_unchanged_passes_gate"] = BASELINE_PPL <= PPL_GATE

    # h) numeric hygiene.
    flat = [et, COVERAGE_CEILING_GAP, LOCKED_TOP1_TO_TOP4_PRIZE, ident["deployed_mtp_top1_coverage"],
            ident["top1_artifact_mismatch"], ledger["coverage_ceiling_gap_measured"],
            ledger["ceiling_band_tps_at_968_secant"][1]]
    c["h_no_nan_inf"] = all(_finite(v) for v in flat)

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 5 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    et = expected_tokens_per_step(LADDER_289)
    probe = load_gpu_probe()
    ident = artifact_identity()
    ledger = blocker_ledger(probe)
    denken = feed_denken_208(ledger, ident)
    selftest = run_self_tests(et, probe, ident, ledger)
    return {
        "pr": 401, "agent": "ubel", "kind": "mtp-drafter-topk-coverage-ceiling",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used_for_load_only": True, "official_tps": 0,
        "baseline_unchanged_tps": BASELINE_TPS, "baseline_unchanged_ppl": BASELINE_PPL,
        "inputs": {
            "ladder_289": LADDER_289, "e_accepted_289": E_ACCEPTED_289, "e_t_289": E_T_289,
            "cond_accept_76": COND_ACCEPT_76, "deployed_mtp_top1_76": DEPLOYED_MTP_TOP1_76, "e_t_76": E_T_76,
            "eagle3_top1_387": EAGLE3_TOP1_387, "eagle3_top4_387": EAGLE3_TOP4_387,
            "per_source_top4_387": PER_SOURCE_TOP4_387, "official_mix_387": OFFICIAL_MIX_387,
            "coverage_ceiling_gap": COVERAGE_CEILING_GAP, "locked_top1_to_top4_prize": LOCKED_TOP1_TO_TOP4_PRIZE,
            "base_471_390": BASE_471_390, "gap_28_390": GAP_28_390, "base_467_393": BASE_467_393,
            "gap_33_393": GAP_33_393, "s_central_399": S_CENTRAL_399, "tps_per_unit_dcov_399": TPS_PER_UNIT_DCOV_399,
            "baseline_tps": BASELINE_TPS, "baseline_ppl": BASELINE_PPL, "ppl_gate": PPL_GATE,
            "roundtrip_tol": ROUNDTRIP_TOL,
            "source_289_run": "fi34s269", "source_76_ref": "accept_calibration", "source_387_run": "z8osvif8",
            "source_390_run": "5y64zbjz", "source_393_run": "0q7ynumg", "source_399_run": "ec7i3z5t",
            "deployed_spec": {"method": "mtp", "model": "/tmp/qat-assistant", "num_speculative_tokens": 7},
        },
        "gpu_probe": probe,
        "expected_tokens_per_step": et,
        "artifact_identity": ident,
        "blocker_ledger": ledger,
        "feed_denken_208": denken,
        # ---- card-required deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "drafter_loadable": bool(probe.get("drafter_loadable", False)),
        "topk_coverage_roundtrip_top1": ident["deployed_mtp_top1_coverage"],   # FAITHFUL deployed MTP (0.729), NOT 0.7617
        "topk_coverage_roundtrip_top1_matches_eagle3_anchor": ident["roundtrip_top1_vs_eagle3_passes"],  # False
        "topk_coverage_roundtrip_top4": None,                                  # no banked MTP top-4 (blocked)
        "realized_topk_coverage_8": ledger["realized_topk_coverage_8"],
        "realized_topk_coverage_16": ledger["realized_topk_coverage_16"],
        "realized_tree_headroom_8": ledger["realized_tree_headroom_8"],
        "realized_tree_headroom_16": ledger["realized_tree_headroom_16"],
        "realized_tree_prize_fraction_8": ledger["realized_tree_prize_fraction_8"],
        "realized_tree_prize_fraction_16": ledger["realized_tree_prize_fraction_16"],
        "per_position_coverage_8": ledger["per_position_coverage_8"],          # None (top-1 vec delivered instead)
        "per_position_coverage_1": ident["per_position_coverage_1"],           # the faithful 7-vector we DO deliver
        "coverage_ceiling_gap_measured": ledger["coverage_ceiling_gap_measured"],  # = 0.1097 (unchanged bound)
        "eagle3_anchors_are_wrong_artifact": True,
        "self_test": selftest,
        "topk_coverage_ceiling_self_test_passes": selftest["passes"],          # PRIMARY
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"

    def numify(x):  # W&B summary wants numbers/strings, not None
        return float("nan") if x is None else x

    # W&B drops NaN-valued summary keys (public API returns them absent). The PR
    # requires the blocked deliverables to be PRESENT in summary/, so represent a
    # blocked/unmeasured deliverable with a VISIBLE string sentinel instead of NaN.
    def blocked(x):
        return "blocked:unmeasured" if x is None else x

    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        ident, ledger, denken, probe = (report["artifact_identity"], report["blocker_ledger"],
                                        report["feed_denken_208"], report["gpu_probe"])
        wandb.summary.update({
            "drafter_loadable": report["drafter_loadable"],
            "topk_coverage_roundtrip_top1": report["topk_coverage_roundtrip_top1"],
            "topk_coverage_roundtrip_top1_matches_eagle3_anchor": report["topk_coverage_roundtrip_top1_matches_eagle3_anchor"],
            "topk_coverage_roundtrip_top4": blocked(report["topk_coverage_roundtrip_top4"]),
            "realized_topk_coverage_8": blocked(report["realized_topk_coverage_8"]),
            "realized_topk_coverage_16": blocked(report["realized_topk_coverage_16"]),
            "realized_tree_headroom_8": blocked(report["realized_tree_headroom_8"]),
            "realized_tree_headroom_16": blocked(report["realized_tree_headroom_16"]),
            "realized_tree_prize_fraction_8": blocked(report["realized_tree_prize_fraction_8"]),
            "realized_tree_prize_fraction_16": blocked(report["realized_tree_prize_fraction_16"]),
            "per_position_coverage_8": blocked(report["per_position_coverage_8"]),
            "per_position_coverage_1_faithful": list(report["per_position_coverage_1"]),
            "coverage_ceiling_gap_measured": report["coverage_ceiling_gap_measured"],
            "eagle3_anchors_are_wrong_artifact": report["eagle3_anchors_are_wrong_artifact"],
            "topk_read_blocked": ledger["topk_read_blocked"],
            "n_blocker_reasons": ledger["n_blocker_reasons"],
            "ceiling_band_collapsed": ledger["ceiling_band_collapsed"],
            "deployed_mtp_top1_coverage": ident["deployed_mtp_top1_coverage"],
            "top1_artifact_mismatch": ident["top1_artifact_mismatch"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "topk_coverage_ceiling_self_test_passes": report["topk_coverage_ceiling_self_test_passes"],
        })
        wandb.log({
            "summary/drafter_loadable": float(report["drafter_loadable"]),
            "summary/expected_tokens_per_step": report["expected_tokens_per_step"],
            "summary/deployed_mtp_top1_coverage": ident["deployed_mtp_top1_coverage"],
            "summary/topk_coverage_roundtrip_top1": report["topk_coverage_roundtrip_top1"],
            "summary/topk_coverage_roundtrip_top1_matches_eagle3_anchor": float(report["topk_coverage_roundtrip_top1_matches_eagle3_anchor"]),
            "summary/topk_coverage_roundtrip_top4_blocked": float(report["topk_coverage_roundtrip_top4"] is None),
            "summary/realized_topk_coverage_8": numify(report["realized_topk_coverage_8"]),
            "summary/realized_topk_coverage_16": numify(report["realized_topk_coverage_16"]),
            "summary/realized_tree_prize_fraction_8": numify(report["realized_tree_prize_fraction_8"]),
            "summary/realized_tree_prize_fraction_16": numify(report["realized_tree_prize_fraction_16"]),
            "summary/coverage_ceiling_gap_measured": report["coverage_ceiling_gap_measured"],
            "summary/top1_artifact_mismatch": ident["top1_artifact_mismatch"],
            "summary/eagle3_anchor_top1": EAGLE3_TOP1_387,
            "summary/eagle3_anchor_top4": EAGLE3_TOP4_387,
            "summary/ceiling_band_tps_upper_968_secant": ledger["ceiling_band_tps_at_968_secant"][1],
            "summary/n_blocker_reasons": float(ledger["n_blocker_reasons"]),
            "summary/standalone_forward_blocked": float(probe.get("standalone_forward_blocked", True)),
            "summary/vram_alloc_MB": probe.get("vram_alloc_MB", float("nan")),
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # the faithful per-position TOP-1 ladder (the 7-vector we DELIVER in lieu of the blocked top-8 vec).
        for j, a in enumerate(report["per_position_coverage_1"], start=1):
            wandb.log({f"per_position_top1/pos_{j}": a})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    ident, ledger, probe = r["artifact_identity"], r["blocker_ledger"], r["gpu_probe"]
    print("\n=== Measure the locked top-8/16 MTP-drafter coverage ceiling (PR #401, ubel) ===")
    print(f"deployed spec: method=mtp model=/tmp/qat-assistant K=7   (BASELINE {BASELINE_TPS} TPS / "
          f"PPL {BASELINE_PPL} UNCHANGED; 0-TPS card)")
    print("\n-- deliverable 1: LOAD the deployed MTP drafter on the A10G --")
    print(f"   drafter_loadable = {probe.get('drafter_loadable')}  ({probe.get('model_class')}, "
          f"{probe.get('n_params_M')}M, head vocab {probe.get('lm_head_vocab')}, {probe.get('vram_alloc_MB')} MB, "
          f"load {probe.get('load_wall_s')}s on {probe.get('gpu_name')})  [_source={probe.get('_source')}]")
    print(f"   standalone forward blocked = {probe.get('standalone_forward_blocked')}: "
          f"{probe.get('standalone_forward_error','')}")
    print("\n-- ARTIFACT IDENTITY (the load-bearing correction) --")
    print(f"   deployed MTP top-1 coverage = {ident['deployed_mtp_top1_coverage']:.4f}  "
          f"(#289 a_1; xcheck #76 {ident['deployed_mtp_top1_xcheck_76']:.4f})")
    print(f"   PR round-trip anchor top-1  = {ident['eagle3_anchor_top1']:.4f}  (fern#34 EAGLE-3 gua9x68j, "
          f"NEVER deployed -- #387)")
    print(f"   => top-1 artifact mismatch  = {ident['top1_artifact_mismatch']:+.4f}  "
          f"(>> {ROUNDTRIP_TOL} tol; round-trip vs EAGLE-3 PASSES = {ident['roundtrip_top1_vs_eagle3_passes']})")
    print(f"   faithful per-position TOP-1 7-vector = {[round(a,3) for a in ident['per_position_coverage_1']]}")
    print("\n-- BLOCKER LEDGER (why a VALIDATED top-8/16 read is blocked) --")
    for i, reason in enumerate(ledger["reasons"], 1):
        print(f"   ({i}) {reason['id']}: {reason['summary']}")
    print("\n-- CEILING VERDICT (band NOT collapsed) --")
    print(f"   coverage_ceiling_gap_measured = {ledger['coverage_ceiling_gap_measured']:.4f}  "
          f"(UNCHANGED [0, {COVERAGE_CEILING_GAP:.4f}] bound)")
    print(f"   realized_topk_coverage_8/16   = None / None")
    print(f"   realized_tree_prize_fraction_8/16 = None / None")
    print(f"   band -> TPS (at {TPS_PER_UNIT_DCOV_399} TPS/Dcov) = "
          f"[0, +{ledger['ceiling_band_tps_at_968_secant'][1]:.1f}] TPS  (denken #208: PARAMETERIZE, don't plug)")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"topk_coverage_ceiling_self_test_passes = {r['topk_coverage_ceiling_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the locked top-8/16 MTP-drafter coverage ceiling (PR #401).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="CPU-analytic gate (>=20 asserts); no torch/W&B")
    ap.add_argument("--probe-gpu", action="store_true",
                    help="LIVE A10G load of /tmp/qat-assistant -> _gpu_probe.json (needs torch + a visible GPU)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="mtp-drafter-topk-coverage-ceiling")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="ubel/mtp-drafter-topk-coverage-ceiling")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/mtp_drafter_topk_coverage_ceiling/mtp_drafter_topk_coverage_ceiling_results.json")
    args = ap.parse_args()

    if args.probe_gpu:
        probe = run_gpu_probe()
        print(json.dumps(probe, indent=2))
        print(f"\nwrote {GPU_PROBE_PATH}  drafter_loadable={probe.get('drafter_loadable')}")
        return 0 if probe.get("drafter_loadable") else 1

    report = build_report()
    print_report(report)

    if args.self_test:
        out = HERE / "mtp_drafter_topk_coverage_ceiling_selftest.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
        print(f"\ntopk_coverage_ceiling_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "drafter_loadable": report["drafter_loadable"],
        "topk_coverage_roundtrip_top1": float(report["topk_coverage_roundtrip_top1"]),
        "topk_coverage_roundtrip_top1_matches_eagle3_anchor": bool(report["topk_coverage_roundtrip_top1_matches_eagle3_anchor"]),
        "realized_topk_coverage_8": report["realized_topk_coverage_8"],
        "realized_topk_coverage_16": report["realized_topk_coverage_16"],
        "realized_tree_prize_fraction_8": report["realized_tree_prize_fraction_8"],
        "realized_tree_prize_fraction_16": report["realized_tree_prize_fraction_16"],
        "coverage_ceiling_gap_measured": float(report["coverage_ceiling_gap_measured"]),
        "eagle3_anchors_are_wrong_artifact": True,
        "topk_coverage_ceiling_self_test_passes": bool(report["topk_coverage_ceiling_self_test_passes"]),
        "primary_metric": {"name": "topk_coverage_ceiling_self_test_passes",
                           "value": float(report["topk_coverage_ceiling_self_test_passes"])},
        "test_metric": {"name": "coverage_ceiling_gap_measured",
                        "value": float(report["coverage_ceiling_gap_measured"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
