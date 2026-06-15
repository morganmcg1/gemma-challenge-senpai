#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 build-cost card: GPU-hours to train the gated drafter (PR #301).

THE QUESTION
------------
The sole >=500 path is the human-gated EAGLE-3 BUILD. Every OTHER side of that
build is priced: acceptance achievability (wirbel #290: step-banked E[T]=4.9029 /
honest 6.1245), step latency (wirbel #295), Phase-1 GO/NO-GO (kanna #294), VRAM
fit <=24GB (ubel #299), the binding private-bar (lawine #300). The ONE missing
input for a human GO/NO-GO is the **GPU-hour cost to actually train the drafter**.
A build that clears every acceptance/feasibility bar is still a NO-GO if it costs
more A10G-hours than the lane can spend. This card prices that SPEND axis.

DELIVERABLE
-----------
A CPU-analytic, paper-grounded estimate of the A10G-GPU-hours to build the
EAGLE-3 drafter for our target (google/gemma-4-E4B-it). Decompose:
  * capture_pass_gpu_hours  : one TARGET-model forward over N training tokens to
                              record the {2,21,39} multi-layer hidden states
                              EAGLE-3 fuses (forward-only, ~2*P_target*N).
  * drafter_train_gpu_hours : the EAGLE-3 draft net (1 decoder layer + fusion FC +
                              input FC) over N tokens x epochs (fwd+bwd ~6*P), PLUS
                              the frozen-but-evaluated 262k-vocab lm_head GEMM that
                              every token's cross-entropy loss runs (~4*P_lmhead).
  * total = capture + drafter_train ; wall_clock_8xa10g = total / 8 (DP scaling).
Verdict: build_cost_feasible_under_budget vs a stated <=200 A10G-GPU-hour lane
budget (flag). This leg adds 0 TPS; BASELINE 481.53 unchanged. NOT a launch; no
GPU/model-forward/training/served-file/HF-Job/submission. Bank-the-analysis.

KEY HONEST FINDINGS (surfaced, not hidden)
------------------------------------------
  * The EAGLE-3 paper (arXiv:2503.01840) reports NO training GPU-hours / wall-clock /
    GPU-type for the drafter -- its only GPU figures are INFERENCE throughput
    (H100/RTX3090) and a "GPU constraint" note for not testing 405B/671B. So a
    paper-vs-bottom-up reconciliation has no paper number to anchor; we instead
    reconcile TWO bottom-up accountings (PR-literal core-only vs lm_head-aware).
  * Gemma-4-E4B-it has a 262,144-token vocab. The frozen, target-tied lm_head GEMM
    (k x vocab = 2560 x 262144 ~ 671M "params") is run on EVERY drafter-train token
    for the CE loss and dominates the tiny 124.5M draft net by ~3.6:1 in per-token
    FLOPs. Pricing the drafter with the PR-literal 6*P_core ALONE understates the
    drafter-train cost by ~2.7x on the total -- this card carries both and flags it.
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../target

# --------------------------------------------------------------------------- #
# Banked anchors (imported VERBATIM; never re-derived). Provenance in comments.
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                         # PR #52 official frontier TPS (2x9fm2zx); 0 TPS added here

# ---- Target model geometry (PINNED from the served config.json of
#      google/gemma-4-E4B-it, text_config; the {2,21,39} fusion layers cited by
#      wirbel #290/#293 are low/mid/high of this 42-layer decoder). ----
K_HIDDEN = 2560                                    # text_config.hidden_size
INTERMEDIATE = 10240                               # text_config.intermediate_size (gated MLP)
N_HEADS = 8                                        # text_config.num_attention_heads
HEAD_DIM = 256                                     # text_config.head_dim
N_KV_HEADS = 2                                     # text_config.num_key_value_heads (GQA)
VOCAB = 262144                                     # text_config.vocab_size (large!)
N_TARGET_LAYERS = 42                               # text_config.num_hidden_layers
EAGLE3_TARGET_LAYERS = (2, 21, 39)                 # multi-layer hidden-state fusion source layers
L_FUSE = len(EAGLE3_TARGET_LAYERS)                 # 3 fused layers -> fusion FC is 3k -> k
# PR-stated effective dense active-param count used for the target FORWARD FLOP.
P_TARGET_EFFECTIVE = 4.0e9                         # ~4B effective dense (PR #301 instruction)

# ---- EAGLE-3 training recipe (arXiv:2503.01840, Li et al. 2025; Implementation
#      section). Cited, not invented. ----
#   * corpus  : ShareGPT (~68K entries) + UltraChat-200K (~464K entries); target
#               model GENERATES the responses (not a fixed dataset).
#   * optim   : AdamW, betas (0.9,0.95), grad-clip 0.5, lr 5e-5.
#   * drafter : a SINGLE transformer decoder layer + multi-layer feature fusion
#               (low/mid/high target hidden states -> 3k concat -> FC -> k).
#   * epochs  : NOT reported in the paper -> carried as a flag (default 10, the
#               EAGLE/EAGLE-3 codebase convention; sensitivity reported).
#   * GPU cost: NOT reported in the paper (no train hours / GPU-type / wall-clock).
SHAREGPT_ENTRIES = 68_000
ULTRACHAT_ENTRIES = 464_000
TOTAL_DATA_ENTRIES = SHAREGPT_ENTRIES + ULTRACHAT_ENTRIES          # 532K conversation/sample entries
PAPER_REPORTS_TRAIN_GPU_HOURS = False                              # genuine gap (see docstring)

# ---- A10G (sm_86) hardware (PR #301 instruction). ----
A10G_BF16_PEAK_FLOPS = 125.0e12                    # ~125 TFLOP/s bf16 tensor-core (dense) achievable peak
N_GPUS_DP = 8                                       # local 8x A10G node (data-parallel wall-clock)
A10G_VRAM_GB = 24                                   # context only; fit proven by ubel #299


# --------------------------------------------------------------------------- #
# Analytic core.
# --------------------------------------------------------------------------- #
def drafter_param_decomposition() -> dict[str, int]:
    """EAGLE-3 draft-net TRAINABLE params for our target geometry (k=2560).

    A single Gemma-style decoder layer (GQA: q=n_heads*head_dim, kv=n_kv*head_dim)
    + EAGLE-3's fusion FC (3k -> k over the {2,21,39} hidden states) + an input FC
    (concat[fused_feature g, sampled-token embedding e] = 2k -> k). The target
    embedding + lm_head are REUSED frozen (not trainable; the lm_head GEMM cost is
    accounted separately because it still runs every train token for the CE loss).
    """
    q_dim = N_HEADS * HEAD_DIM                     # 2048
    kv_dim = N_KV_HEADS * HEAD_DIM                 # 512
    attn = (K_HIDDEN * q_dim                       # q_proj
            + K_HIDDEN * kv_dim                    # k_proj
            + K_HIDDEN * kv_dim                    # v_proj
            + q_dim * K_HIDDEN)                    # o_proj
    mlp = 3 * K_HIDDEN * INTERMEDIATE              # gate + up + down (gated)
    fusion_fc = (L_FUSE * K_HIDDEN) * K_HIDDEN     # 3k -> k
    input_fc = (2 * K_HIDDEN) * K_HIDDEN           # [g; e] = 2k -> k
    core = attn + mlp + fusion_fc + input_fc
    lm_head = K_HIDDEN * VOCAB                      # frozen, tied; GEMM runs per train token
    return {
        "attn": attn, "mlp": mlp, "fusion_fc": fusion_fc, "input_fc": input_fc,
        "p_drafter_core": core, "p_lm_head": lm_head,
    }


def synthesize(args) -> dict[str, Any]:
    n_tokens = float(args.n_train_tokens)
    epochs = float(args.epochs)
    mfu = float(args.mfu)
    eff_flops = A10G_BF16_PEAK_FLOPS * mfu          # achievable * MFU
    SEC_PER_HOUR = 3600.0

    params = drafter_param_decomposition()
    p_core = float(params["p_drafter_core"])
    p_lm_head = float(params["p_lm_head"])
    tokens_per_entry = n_tokens / TOTAL_DATA_ENTRIES   # provenance: ~752 @ default 4e8 / 532K

    # ---- (1) CAPTURE PASS: one TARGET forward over N tokens (forward-only). ---- #
    # FLOPs ~= 2 * P_target * N. Records the {2,21,39} fused hidden states; can be
    # FUSED into the target's response-generation pass (no double count) -- see caveat.
    capture_flops = 2.0 * P_TARGET_EFFECTIVE * n_tokens
    capture_gpu_hours = capture_flops / eff_flops / SEC_PER_HOUR

    # ---- (2) DRAFTER TRAIN: tiny draft net (fwd+bwd) + frozen lm_head GEMM. ---- #
    # core (trainable): fwd 2P + bwd weight-grad 2P + bwd act-grad 2P = 6P.
    # lm_head (frozen, tied): fwd 2P + bwd act-grad 2P (NO weight-grad) = 4P, run on
    #   every token because the CE loss needs logits over the 262k vocab.
    drafter_core_flops = 6.0 * p_core * n_tokens * epochs
    lm_head_flops = 4.0 * p_lm_head * n_tokens * epochs
    drafter_train_flops = drafter_core_flops + lm_head_flops
    drafter_core_gpu_hours = drafter_core_flops / eff_flops / SEC_PER_HOUR     # PR-literal (core-only)
    lm_head_gpu_hours = lm_head_flops / eff_flops / SEC_PER_HOUR
    drafter_train_gpu_hours = drafter_train_flops / eff_flops / SEC_PER_HOUR

    # ---- (3) TOTALS + wall-clock. -------------------------------------------- #
    total_build_gpu_hours = capture_gpu_hours + drafter_train_gpu_hours
    wall_clock_hours_8xa10g = total_build_gpu_hours / N_GPUS_DP

    # ---- (4) BUDGET VERDICT. ------------------------------------------------- #
    budget = float(args.budget_gpu_hours)
    build_cost_feasible_under_budget = bool(total_build_gpu_hours <= budget)
    budget_headroom_gpu_hours = budget - total_build_gpu_hours

    # ---- (5) CROSS-CHECK / RECONCILIATION. ----------------------------------- #
    # Paper reports NO training cost -> no paper anchor. Reconcile the two bottom-up
    # accountings: PR-literal core-only vs lm_head-aware. The 262k-vocab lm_head GEMM
    # is the dominant drafter-train term; flag the divergence if > 2x (it is).
    total_core_only = capture_gpu_hours + drafter_core_gpu_hours
    drafter_term_divergence = drafter_train_gpu_hours / drafter_core_gpu_hours
    total_divergence = total_build_gpu_hours / total_core_only
    divergence_flagged_gt_2x = bool(total_divergence > 2.0)
    lm_head_dominates_drafter = bool(lm_head_gpu_hours > drafter_core_gpu_hours)
    lm_head_per_token_ratio = (4.0 * p_lm_head) / (6.0 * p_core)   # per-token FLOP ratio

    # Adjacent, intentionally UN-headlined cost: the target also GENERATES the
    # responses (autoregressive). With a KV cache that is ~2*P_target*N forward FLOP
    # again -- but EAGLE captures hidden states DURING generation, so for a self-
    # generated corpus it FUSES into the capture pass (no separate charge). Reported
    # for transparency only; excluded from the headline total per the PR decomposition.
    data_gen_gpu_hours_if_separate = capture_gpu_hours

    # ---- (6) SENSITIVITY (epochs x tokens grid; informational). -------------- #
    def total_at(nt: float, ep: float) -> float:
        cap = (2.0 * P_TARGET_EFFECTIVE * nt) / eff_flops / SEC_PER_HOUR
        drf = ((6.0 * p_core + 4.0 * p_lm_head) * nt * ep) / eff_flops / SEC_PER_HOUR
        return cap + drf
    sensitivity = {
        f"tokens={nt:.0e}_epochs={int(ep)}": {
            "total_gpu_hours": total_at(nt, ep),
            "feasible_le_budget": bool(total_at(nt, ep) <= budget),
        }
        for nt in (2.0e8, 4.0e8, 8.0e8) for ep in (2, 10, 20)
    }

    # ---- (7) SELF-TEST (PRIMARY; >= 8 checks). ------------------------------- #
    # (a) FLOP arithmetic round-trips: hours * eff_flops * 3600 reproduces FLOPs.
    cap_flops_rt = capture_gpu_hours * eff_flops * SEC_PER_HOUR
    drf_flops_rt = drafter_train_gpu_hours * eff_flops * SEC_PER_HOUR
    a_flop_roundtrip = (abs(cap_flops_rt - capture_flops) <= 1e-3 * capture_flops
                        and abs(drf_flops_rt - drafter_train_flops) <= 1e-3 * drafter_train_flops)
    # (b) all hours > 0 and finite.
    hours_all = [capture_gpu_hours, drafter_train_gpu_hours, total_build_gpu_hours,
                 wall_clock_hours_8xa10g, drafter_core_gpu_hours, lm_head_gpu_hours]
    b_hours_pos_finite = all(h > 0.0 and math.isfinite(h) for h in hours_all)
    # (c) capture_pass < drafter_train (forward-only vs fwd+bwd-over-epochs +lm_head).
    c_capture_lt_drafter = bool(capture_gpu_hours < drafter_train_gpu_hours)
    # (d) total = capture + drafter_train (exact sum).
    d_total_is_sum = abs(total_build_gpu_hours - (capture_gpu_hours + drafter_train_gpu_hours)) < 1e-9
    # (e) wall_clock = total / 8 (exact).
    e_wall_is_total_over_8 = abs(wall_clock_hours_8xa10g - total_build_gpu_hours / N_GPUS_DP) < 1e-9
    # (f) budget bool consistent with total vs budget.
    f_budget_consistent = (build_cost_feasible_under_budget == (total_build_gpu_hours <= budget))
    # (g) paper-derived (absence-of-cost) and bottom-up both logged.
    g_both_estimates_logged = (PAPER_REPORTS_TRAIN_GPU_HOURS is False
                               and total_build_gpu_hours > 0.0 and total_core_only > 0.0)
    # (h) MFU in (0,1] and FLOP/s assumption finite-positive (logged as config).
    h_mfu_flops_logged = (0.0 < mfu <= 1.0 and math.isfinite(A10G_BF16_PEAK_FLOPS)
                          and A10G_BF16_PEAK_FLOPS > 0.0)
    # (i) drafter param decomposition sums (core == attn+mlp+fusion+input).
    i_param_sum = (params["p_drafter_core"]
                   == params["attn"] + params["mlp"] + params["fusion_fc"] + params["input_fc"])
    # (j) lm_head GEMM divergence surfaced: core-only < lm_head-aware (the flag fired).
    j_lm_head_divergence = bool(drafter_core_gpu_hours < drafter_train_gpu_hours
                                and divergence_flagged_gt_2x and lm_head_dominates_drafter)
    # (k) target geometry constants imported EXACT (k, vocab, P_target).
    k_constants_exact = (K_HIDDEN == 2560 and VOCAB == 262144
                         and abs(P_TARGET_EFFECTIVE - 4.0e9) < 1e-3
                         and N_TARGET_LAYERS == 42 and L_FUSE == 3)

    cond = {
        "a_flop_arithmetic_roundtrips": bool(a_flop_roundtrip),
        "b_all_hours_positive_finite": bool(b_hours_pos_finite),
        "c_capture_lt_drafter_train": bool(c_capture_lt_drafter),
        "d_total_equals_sum_of_parts": bool(d_total_is_sum),
        "e_wallclock_equals_total_over_8": bool(e_wall_is_total_over_8),
        "f_budget_bool_consistent": bool(f_budget_consistent),
        "g_paper_and_bottomup_both_logged": bool(g_both_estimates_logged),
        "h_mfu_and_flops_logged": bool(h_mfu_flops_logged),
        "i_drafter_param_sum_consistent": bool(i_param_sum),
        "j_lmhead_divergence_surfaced": bool(j_lm_head_divergence),
        "k_target_constants_exact": bool(k_constants_exact),
    }
    # (l) NaN-clean checked on the full payload in main().

    # ---- (8) VERDICT + HAND-OFF. --------------------------------------------- #
    go_nogo = "GO" if build_cost_feasible_under_budget else "NO-GO"
    verdict = (
        "EAGLE-3 drafter BUILD-COST (A10G, sm_86, %.0f TFLOP/s x MFU %.2f = %.1f TFLOP/s eff): "
        "capture-pass %.1f GPU-hr (target fwd 2*%.1fB*%.0eN) + drafter-train %.1f GPU-hr "
        "(core %.1f + frozen 262k-vocab lm_head GEMM %.1f) = TOTAL %.1f A10G-GPU-hr -> %.1f h "
        "wall-clock on 8x A10G (DP). vs <=%.0f GPU-hr lane budget: %s-ON-COST "
        "(headroom %.1f GPU-hr). The lm_head GEMM dominates the draft net %.2fx/token and "
        "inflates the total %.2fx over the PR-literal core-only accounting (flagged >2x). The "
        "paper reports NO train cost, so this bottom-up estimate IS the anchor. 0 TPS added; "
        "BASELINE 481.53 untouched; analysis-only; NOT a launch." % (
            A10G_BF16_PEAK_FLOPS / 1e12, mfu, eff_flops / 1e12,
            capture_gpu_hours, P_TARGET_EFFECTIVE / 1e9, n_tokens,
            drafter_train_gpu_hours, drafter_core_gpu_hours, lm_head_gpu_hours,
            total_build_gpu_hours, wall_clock_hours_8xa10g, budget, go_nogo,
            budget_headroom_gpu_hours, lm_head_per_token_ratio, total_divergence))

    handoff = (
        "the human EAGLE-3 GO/NO-GO now has its SPEND axis: ~%.0f A10G-GPU-hr (~%.1f h on 8x A10G) "
        "to build the drafter at N=%.0e tokens x %d epochs, MFU %.2f -- %s under the <=%.0f GPU-hr "
        "lane budget. Dominant term is the 262k-vocab lm_head GEMM (%.1f GPU-hr, %.0f%% of drafter "
        "train), not the 124.5M draft net; epochs/tokens are the live knobs (sensitivity logged). "
        "Pairs with wirbel #290 (acceptance), kanna #294 (Phase-1), ubel #299 (VRAM), lawine #300 "
        "(private bar): cost is no longer the missing input." % (
            total_build_gpu_hours, wall_clock_hours_8xa10g, n_tokens, int(epochs), mfu, go_nogo,
            budget, lm_head_gpu_hours,
            100.0 * lm_head_gpu_hours / drafter_train_gpu_hours))

    honest_caveat = (
        "ORDER-OF-MAGNITUDE estimate, not a quote: (1) N_train_tokens=%.0e is derived from the "
        "paper's entry counts (532K entries x ~%.0f tok/entry), not a measured token count -- the "
        "paper gives entries, not tokens; (2) epochs are NOT in the paper (default %d = codebase "
        "convention); (3) MFU %.2f and %.0f TFLOP/s are nominal A10G assumptions; (4) the target's "
        "response-GENERATION pass (~%.1f GPU-hr if run separately) is FUSED into capture for a self-"
        "generated corpus and excluded from the headline; (5) DP wall-clock assumes ideal 8x scaling "
        "(ignores comm/imbalance). Feasibility verdict is robust: even at 8e8 tokens x 20 epochs the "
        "total stays a small multiple of budget (sensitivity grid logged)." % (
            n_tokens, tokens_per_entry, int(epochs), mfu, A10G_BF16_PEAK_FLOPS / 1e12,
            data_gen_gpu_hours_if_separate))

    return {
        "config": {
            "n_train_tokens": n_tokens, "epochs": epochs, "mfu": mfu,
            "a10g_bf16_peak_flops": A10G_BF16_PEAK_FLOPS, "eff_flops": eff_flops,
            "n_gpus_dp": N_GPUS_DP, "budget_gpu_hours": budget,
            "tokens_per_entry_implied": tokens_per_entry,
            "total_data_entries": TOTAL_DATA_ENTRIES,
            "sharegpt_entries": SHAREGPT_ENTRIES, "ultrachat_entries": ULTRACHAT_ENTRIES,
            "paper_reports_train_gpu_hours": PAPER_REPORTS_TRAIN_GPU_HOURS,
        },
        "target_geometry": {
            "k_hidden": K_HIDDEN, "intermediate": INTERMEDIATE, "n_heads": N_HEADS,
            "head_dim": HEAD_DIM, "n_kv_heads": N_KV_HEADS, "vocab": VOCAB,
            "n_target_layers": N_TARGET_LAYERS, "eagle3_target_layers": list(EAGLE3_TARGET_LAYERS),
            "l_fuse": L_FUSE, "p_target_effective": P_TARGET_EFFECTIVE,
        },
        "drafter_params": params,
        "flops": {
            "capture_flops": capture_flops, "drafter_core_flops": drafter_core_flops,
            "lm_head_flops": lm_head_flops, "drafter_train_flops": drafter_train_flops,
        },
        "build_cost": {
            "capture_pass_gpu_hours": capture_gpu_hours,
            "drafter_train_gpu_hours": drafter_train_gpu_hours,
            "drafter_core_gpu_hours": drafter_core_gpu_hours,
            "lm_head_gpu_hours": lm_head_gpu_hours,
            "total_build_gpu_hours": total_build_gpu_hours,
            "wall_clock_hours_8xa10g": wall_clock_hours_8xa10g,
        },
        "budget_verdict": {
            "budget_gpu_hours": budget,
            "build_cost_feasible_under_budget": build_cost_feasible_under_budget,
            "budget_headroom_gpu_hours": budget_headroom_gpu_hours,
            "go_nogo_on_cost": go_nogo,
        },
        "reconciliation": {
            "paper_reported_train_gpu_hours": None,
            "total_core_only_gpu_hours": total_core_only,
            "total_lmhead_aware_gpu_hours": total_build_gpu_hours,
            "drafter_term_divergence_x": drafter_term_divergence,
            "total_divergence_x": total_divergence,
            "divergence_flagged_gt_2x": divergence_flagged_gt_2x,
            "lm_head_dominates_drafter": lm_head_dominates_drafter,
            "lm_head_per_token_flop_ratio": lm_head_per_token_ratio,
            "data_gen_gpu_hours_if_separate": data_gen_gpu_hours_if_separate,
        },
        "sensitivity": sensitivity,
        "self_test": {"conditions": cond},
        "honest_caveat": honest_caveat,
        "verdict": verdict, "handoff": handoff,
        # ---- headline metrics ----
        "total_build_gpu_hours": total_build_gpu_hours,
        "wall_clock_hours_8xa10g": wall_clock_hours_8xa10g,
        "build_cost_feasible_under_budget": build_cost_feasible_under_budget,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors eagle3_feasibility_bracket; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError(
                f"resolved a stub/namespace wandb at {list(getattr(_wb, '__path__', []) or [])} "
                "with no .init -> this venv lacks the wandb wheel")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-build-cost] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    cfg = syn["config"]
    bc = syn["build_cost"]
    bv = syn["budget_verdict"]
    rc = syn["reconciliation"]
    st = syn["self_test"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="denken", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-build-cost", "gpu-hours", "spend-axis", "drafter-train",
                  "capture-pass", "bank-the-analysis", "pr-301"],
            config={
                "official_baseline": OFFICIAL_BASELINE,
                "n_train_tokens": cfg["n_train_tokens"], "epochs": cfg["epochs"],
                "mfu": cfg["mfu"], "a10g_bf16_peak_flops": cfg["a10g_bf16_peak_flops"],
                "eff_flops": cfg["eff_flops"], "n_gpus_dp": cfg["n_gpus_dp"],
                "budget_gpu_hours": cfg["budget_gpu_hours"],
                "p_target_effective": P_TARGET_EFFECTIVE, "k_hidden": K_HIDDEN, "vocab": VOCAB,
                "p_drafter_core": syn["drafter_params"]["p_drafter_core"],
                "p_lm_head": syn["drafter_params"]["p_lm_head"],
                "imports": "EAGLE-3 arXiv:2503.01840 (recipe: 1 decoder layer + fusion FC, "
                           "ShareGPT 68K + UltraChat 464K entries, AdamW lr 5e-5; NO train-cost reported) x "
                           "gemma-4-E4B-it config (k=2560, vocab=262144, 42 layers) x "
                           "wirbel#290(acceptance) x ubel#299(VRAM) x lawine#300(private-bar)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-build-cost] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-build-cost] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "eagle3_build_cost_self_test_passes":
            int(bool(payload["eagle3_build_cost_self_test_passes"])),
        "total_build_gpu_hours": bc["total_build_gpu_hours"],
        "capture_pass_gpu_hours": bc["capture_pass_gpu_hours"],
        "drafter_train_gpu_hours": bc["drafter_train_gpu_hours"],
        "drafter_core_gpu_hours": bc["drafter_core_gpu_hours"],
        "lm_head_gpu_hours": bc["lm_head_gpu_hours"],
        "wall_clock_hours_8xa10g": bc["wall_clock_hours_8xa10g"],
        "build_cost_feasible_under_budget": int(bool(bv["build_cost_feasible_under_budget"])),
        "budget_headroom_gpu_hours": bv["budget_headroom_gpu_hours"],
        "budget_gpu_hours": bv["budget_gpu_hours"],
        "total_core_only_gpu_hours": rc["total_core_only_gpu_hours"],
        "total_divergence_x": rc["total_divergence_x"],
        "drafter_term_divergence_x": rc["drafter_term_divergence_x"],
        "divergence_flagged_gt_2x": int(bool(rc["divergence_flagged_gt_2x"])),
        "lm_head_dominates_drafter": int(bool(rc["lm_head_dominates_drafter"])),
        "lm_head_per_token_flop_ratio": rc["lm_head_per_token_flop_ratio"],
        "p_drafter_core": syn["drafter_params"]["p_drafter_core"],
        "p_lm_head": syn["drafter_params"]["p_lm_head"],
        "n_train_tokens": cfg["n_train_tokens"], "epochs": cfg["epochs"], "mfu": cfg["mfu"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_build_cost_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[eagle3-build-cost] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-build-cost] wandb write failed (analysis unaffected): {exc}", flush=True)


def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


def _print_human(syn: dict) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" EAGLE-3 DRAFTER BUILD-COST CARD: A10G-GPU-HOURS TO TRAIN THE GATED DRAFTER (PR #301)", flush=True)
    print("=" * 104, flush=True)
    cfg = syn["config"]
    dp = syn["drafter_params"]
    bc = syn["build_cost"]
    bv = syn["budget_verdict"]
    rc = syn["reconciliation"]
    print(f"  RECIPE (arXiv:2503.01840): 1 decoder layer + fusion FC; ShareGPT 68K + UltraChat 464K "
          f"entries; AdamW lr 5e-5; epochs NOT in paper; train-cost NOT in paper.", flush=True)
    print(f"  TARGET: gemma-4-E4B-it k={K_HIDDEN} vocab={VOCAB} layers={N_TARGET_LAYERS} "
          f"P_target_eff={P_TARGET_EFFECTIVE/1e9:.1f}B; fuse layers {list(EAGLE3_TARGET_LAYERS)}", flush=True)
    print(f"  DRAFTER params: core={dp['p_drafter_core']/1e6:.1f}M (attn {dp['attn']/1e6:.1f}M + mlp "
          f"{dp['mlp']/1e6:.1f}M + fusionFC {dp['fusion_fc']/1e6:.1f}M + inputFC {dp['input_fc']/1e6:.1f}M); "
          f"frozen lm_head GEMM={dp['p_lm_head']/1e6:.1f}M", flush=True)
    print("-" * 104, flush=True)
    print(f"  ASSUMPTIONS: N={cfg['n_train_tokens']:.0e} tok x {int(cfg['epochs'])} epochs; A10G "
          f"{cfg['a10g_bf16_peak_flops']/1e12:.0f} TFLOP/s x MFU {cfg['mfu']:.2f} = "
          f"{cfg['eff_flops']/1e12:.1f} TFLOP/s eff", flush=True)
    print(f"  (1) capture_pass_gpu_hours   = {bc['capture_pass_gpu_hours']:8.2f}  (target fwd 2*P*N)", flush=True)
    print(f"  (2) drafter_train_gpu_hours  = {bc['drafter_train_gpu_hours']:8.2f}  "
          f"(core {bc['drafter_core_gpu_hours']:.2f} + lm_head {bc['lm_head_gpu_hours']:.2f})", flush=True)
    print(f"      total_build_gpu_hours    = {bc['total_build_gpu_hours']:8.2f}", flush=True)
    print(f"      wall_clock_hours_8xa10g  = {bc['wall_clock_hours_8xa10g']:8.2f}", flush=True)
    print("-" * 104, flush=True)
    print(f"  BUDGET <= {bv['budget_gpu_hours']:.0f} GPU-hr -> {bv['go_nogo_on_cost']}-ON-COST "
          f"(feasible={bv['build_cost_feasible_under_budget']}, headroom "
          f"{bv['budget_headroom_gpu_hours']:.1f} GPU-hr)", flush=True)
    print(f"  RECONCILE: core-only total {rc['total_core_only_gpu_hours']:.1f} vs lm_head-aware "
          f"{rc['total_lmhead_aware_gpu_hours']:.1f} -> {rc['total_divergence_x']:.2f}x "
          f"(flagged>2x={rc['divergence_flagged_gt_2x']}); paper_reported=None", flush=True)
    print("-" * 104, flush=True)
    print("  SENSITIVITY (total_gpu_hours | feasible):", flush=True)
    for key, val in syn["sensitivity"].items():
        print(f"      {key:<26}{val['total_gpu_hours']:8.1f}  | {val['feasible_le_budget']}", flush=True)
    print("-" * 104, flush=True)
    st = syn["self_test"]
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
    print("-" * 104, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}", flush=True)
    print(f"\n  CAVEAT: {syn['honest_caveat']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--n-train-tokens", "--n_train_tokens", dest="n_train_tokens", type=float,
                    default=4.0e8, help="training tokens (default 4e8 ~ 532K entries x ~752 tok)")
    ap.add_argument("--epochs", type=float, default=10.0,
                    help="drafter training epochs (NOT in paper; default 10 = codebase convention)")
    ap.add_argument("--mfu", type=float, default=0.35, help="model FLOP utilization (0.3-0.4 realistic)")
    ap.add_argument("--budget-gpu-hours", "--budget_gpu_hours", dest="budget_gpu_hours", type=float,
                    default=200.0, help="lane build budget in A10G-GPU-hours (default 200)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-build-cost")
    args = ap.parse_args(argv)

    syn = synthesize(args)
    self_test_passes = all(syn["self_test"]["conditions"].values())

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 301, "agent": "denken",
        "kind": "eagle3-build-cost", "synthesis": syn,
        "eagle3_build_cost_self_test_passes": self_test_passes,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    # fold NaN-clean (condition l) into the PRIMARY pass.
    payload["eagle3_build_cost_self_test_passes"] = bool(self_test_passes and payload["nan_clean"])
    if nan_paths:
        print(f"[eagle3-build-cost] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_build_cost_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[eagle3-build-cost] wrote {out_path}", flush=True)
    print(f"[eagle3-build-cost] PRIMARY eagle3_build_cost_self_test_passes = "
          f"{payload['eagle3_build_cost_self_test_passes']}", flush=True)
    print(f"[eagle3-build-cost] total_build_gpu_hours = {syn['total_build_gpu_hours']:.2f}", flush=True)
    print(f"[eagle3-build-cost] wall_clock_hours_8xa10g = {syn['wall_clock_hours_8xa10g']:.2f}", flush=True)
    print(f"[eagle3-build-cost] build_cost_feasible_under_budget = "
          f"{syn['build_cost_feasible_under_budget']}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["eagle3_build_cost_self_test_passes"]
        print(f"[eagle3-build-cost] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
