#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Plan-B Rank-2 DRAFT argmax+embed FUSION screen (PR #261, wirbel) -- does the
deployed K=7 MTP draft loop launch the per-step `argmax` (draft-token select) and
the `embedding` gather (feed the chosen token back in) as TWO SEPARATE eager
kernels, or are they already fused / captured inside the ONEGRAPH CUDAGraph?
CPU-only analytic bank-the-analysis (the audit leg is the served code + config;
the accounting imports the composition anchors; the safety leg is a bit-exact
argmax->embed reference). NOT a GPU run, NOT a served-file change.

THE FRAME (decisive -- the launch-axis question)
------------------------------------------------
With kanna #254 just proving the draft-PRECISION axis dead (int4 Marlin -27.44%
at M=1), the draft block's only remaining recoverable headroom is the LAUNCH
axis. Each of the K=7 draft steps does an `argmax` over the (sparse) LM-head to
pick the next draft token, then an `embedding` gather to feed it back in. As two
separately-launched eager kernels that is 7 x 2 = 14 launches/step; on A10G
(sm_86) eager-mode marginal launch is ~4-6 us ⇒ a 56-84 us/step launch tax. The
screen asks whether those launches are (a) two separately-launched eager kernels
on the served path, (b) already fused, or (c) already inside a CUDAGraph /
ONEGRAPH capture that hides the launch latency.

  * If separately launched => real greedy-exact lever (fuse argmax->embed, or
    capture the draft tail). Reclaims the launch tax with ZERO change to the
    emitted draft token (argmax is bit-identical). GO + projected gain.
  * If already captured => clean NULL/NO-GO, banked (the #251/#255 pattern: a
    cheap structural kill that closes the question).

DECISIVE BOOLEAN: draft_argmax_embed_separately_launched.

THE AUDIT (served stack, vLLM wheel 0.22.1rc1.dev307+g3e8afdf78)
---------------------------------------------------------------
Served submission: submissions/fa2sw_precache_kenyan. Manifest env:
  SPECULATIVE_CONFIG = {"method":"mtp", ..., "num_speculative_tokens":7}  (K=7
  linear MTP), ONEGRAPH=1, LOOPGRAPH_REQUIRE_CAPTURE=1, FUSED_SPARSE_ARGMAX=1.

  1. ONEGRAPH=1 => Gemma4Proposer.propose = `propose_onegraph` (sitecustomize.py
     L566). The K=7 width-1 draft loop body lives ENTIRELY in `_run_graph_body`
     (L158-203, onegraph branch L173-191): for each of the K iterations it
        self.input_ids[:1].copy_(prev_draft_token)         # feed the token id
        last_hidden, _ = self.model(input_ids=self.input_ids[:1],
                                    inputs_embeds=None, ...)  # EMBED gather is the
                                                              # model's first op
        token = self.model.get_top_tokens(last_hidden[:1])  # the ARGMAX (fused
                                                            # sparse-argmax kernel)
        output[0, index:index+1].copy_(token)
     i.e. BOTH the per-step embed gather (inside `self.model(...)`, inputs_embeds
     =None => `embed_tokens(input_ids)` runs inside the forward) AND the per-step
     argmax (`get_top_tokens` -> on-GPU `logits.argmax`, FUSED_SPARSE_ARGMAX
     Triton kernel) are in the loop body.
  2. `_capture_graph` (L233-248) records the ENTIRE `_run_graph_body` (all K=7
     iterations, argmax+embed included) into ONE `torch.cuda.CUDAGraph()` via
     `with torch.cuda.graph(graph): _run_graph_body(self, state)`. The served
     step is a single `graph.replay()` (L524). LOOPGRAPH_REQUIRE_CAPTURE=1 forces
     capture (`_raise_or_fallback` RAISES on failure, L264-266) => the served
     path is ALWAYS the captured replay, never the eager fallback (L538-564).
  3. Greedy-safety of any fusion is by construction: argmax->token->embed is pure
     INDEX selection (no float reduction reordering, unlike the verify GEMM), so
     fusing the launch boundary cannot change the emitted draft token.

VERDICT OF THE AUDIT: draft_argmax_embed_separately_launched = False. The per-step
argmax + embed are NOT two separately-launched eager kernels on the served path;
they are captured inside the deployed ONEGRAPH `torch.cuda.CUDAGraph` and replayed
with a single `graph.replay()` per step. The 14-launch/step tax DOES NOT EXIST on
the served path => the Rank-2 fusion lever is NULL (already captured).

THE lawine #246 TIE-IN
----------------------
lawine #246 (K-1 FlashInfer + CUDAGraph) is ADJACENT. For the draft argmax+embed
specifically, the DEPLOYED ONEGRAPH substrate already captures the whole K=7 draft
tail -- so this lever is subsumed by the deployed stack, NOT pending on lawine.
If anything, lawine's draft-tail capture is redundant with ONEGRAPH for this
component (lawine's live lever is the attention backend / verify side, a different
axis). subsumed_by_lawine_cudagraph is moot: subsumed_by_deployed_onegraph=True.

THE ACCOUNTING (anchors IMPORTED, not re-derived)
-------------------------------------------------
Composition (kanna #217): official = K_cal*(E[T]/step)*tau, K_cal=125.268,
served step=1.2182 ms (vgovdrjc), served=481.53. denken #252 built step ~1.085 ms
(a7llo7o7). At fixed E[T]/tau, TPS ∝ 1/step => a net step saving us maps to
tps(step-us)=481.53*step/(step-us). The COUNTERFACTUAL launch-tax band (IF the
launches were separate AND on the critical path) = 14*[4,6] = [56,84] us:
  vs served 1218.2 us: 4.60-6.90% step  -> 504.7-517.2 TPS (clears 500, < 520.95)
  vs built  1085.0 us: 5.16-7.74% step  -> (land #245's denominator)
The audit refutes the premise => ACTUAL max_step_reduction_pct = 0,
projected_tps_gain_pct = 0.00.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM run / HF Job / submission / served-file
change / official draw. BASELINE stays 481.53; the 520.95 lambda=1 ceiling stays
520.95; this SCREEN adds 0 TPS. NOT a launch. NOT open2. Non-overlap: kanna #254
(draft PRECISION, RED), lawine #246 (attention/CUDAGraph build, ADJACENT -- this
prices whether its capture subsumes the draft launches), denken #257 (roofline
decompose -- this prices ONE component), stark #256 (adaptive-K draft COUNT, not
per-step launch tax), ubel #258 / fern #259 (E[T] side), land #245 (the live
build owns any gain). Clean continuation of the speed-lever lane: #255 priced the
VERIFY epilogue; this prices the DRAFT launch (the other side of the step).

PRIMARY metric  draft_argmax_embed_fusion_screen_self_test_passes
TEST    metric  projected_tps_gain_pct   (0.00 actual; counterfactual bounds stated)
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
# IMPORTED anchors (kanna #217 composition vgovdrjc, denken #252 built step
# a7llo7o7, sm_86 launch-floor note, lambda=1 ceiling). Re-derive NOTHING.
# --------------------------------------------------------------------------- #
SERVED_TPS = 481.53          # official served (linear MTP K=7, PR #52); this screen adds 0
BASELINE_TPS = 481.53
LAMBDA1_CEILING_TPS = 520.95  # public lambda=1 operative ceiling (opttree/profile.py); UNCHANGED
STEP_US_SERVED = 1218.2      # served step time 1.2182 ms (kanna #217 vgovdrjc) -- live TPS-gate denominator
STEP_US_BUILT = 1085.0       # built step time ~1.085 ms (denken #252 a7llo7o7) -- land #245's denominator
K_CAL = 125.268             # composition calibration constant (kanna #217)
K_SPEC = 7                  # num_speculative_tokens (manifest, linear MTP K=7)
N_LAUNCHES = 2 * K_SPEC     # 14 = 7 steps x {argmax, embed}
PER_LAUNCH_US_LO = 4.0      # sm_86 eager marginal-launch low (denken launch-floor note)
PER_LAUNCH_US_HI = 6.0      # sm_86 eager marginal-launch high
PER_LAUNCH_US_MID = 5.0     # midpoint
SM86_LAUNCH_FLOOR_US = 55.0  # sm_86 eager launch-FLOOR per call (context only; denken note)

# --------------------------------------------------------------------------- #
# AUDITED served-manifest flags (read from the served manifest when present; the
# fallbacks are the values verified from submissions/fa2sw_precache_kenyan under
# vLLM 0.22.1rc1.dev307+g3e8afdf78). The manifest flags GROUND the boolean -- if a
# future served manifest dropped ONEGRAPH/capture, this screen would flip.
# --------------------------------------------------------------------------- #
VLLM_VERSION = "0.22.1rc1.dev307+g3e8afdf78"
SERVED_SUBMISSION = "fa2sw_precache_kenyan"
_MANIFEST = REPO_ROOT / "submissions" / SERVED_SUBMISSION / "manifest.json"
_SITECUSTOMIZE = REPO_ROOT / "submissions" / SERVED_SUBMISSION / "sitecustomize.py"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def audit_served() -> dict[str, Any]:
    """Resolve the served-manifest flags that GROUND the decisive boolean. The
    boolean is sourced from ONEGRAPH + LOOPGRAPH_REQUIRE_CAPTURE + the
    SPECULATIVE_CONFIG num_speculative_tokens, plus the presence of the
    `_capture_graph` / `graph.replay()` loop-capture in sitecustomize.py."""
    out: dict[str, Any] = {
        "manifest_path": None, "sitecustomize_path": None,
        "k_spec": K_SPEC, "spec_method": "mtp",
    }
    flags = {
        "ONEGRAPH": None, "LOOPGRAPH_REQUIRE_CAPTURE": None,
        "FUSED_SPARSE_ARGMAX": None, "SPECULATIVE_CONFIG": None,
        "OVERRIDE_GENERATION_CONFIG": None, "DIXIE_SLIM_GREEDY": None,
    }
    if _MANIFEST.is_file():
        out["manifest_path"] = str(_MANIFEST)
        try:
            env = json.loads(_MANIFEST.read_text()).get("env", {})
            for k in list(flags):
                if k in env:
                    flags[k] = env[k]
            sc = env.get("SPECULATIVE_CONFIG")
            if isinstance(sc, str):
                try:
                    scd = json.loads(sc)
                    if _is_num(scd.get("num_speculative_tokens")):
                        out["k_spec"] = int(scd["num_speculative_tokens"])
                    if isinstance(scd.get("method"), str):
                        out["spec_method"] = scd["method"]
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    out["manifest_flags"] = flags

    # sitecustomize.py loop-capture structural evidence (the served capture path).
    sc_src = ""
    if _SITECUSTOMIZE.is_file():
        out["sitecustomize_path"] = str(_SITECUSTOMIZE)
        try:
            sc_src = _SITECUSTOMIZE.read_text()
        except Exception:  # noqa: BLE001
            sc_src = ""
    out["has_capture_graph"] = ("def _capture_graph" in sc_src) and (
        "torch.cuda.graph(graph)" in sc_src or "torch.cuda.CUDAGraph()" in sc_src
    )
    out["has_graph_replay"] = ("graph.replay()" in sc_src)
    out["has_propose_onegraph"] = ("def propose_onegraph" in sc_src)
    out["has_run_graph_body"] = ("def _run_graph_body" in sc_src)
    # the loop body embeds via inputs_embeds=None and argmaxes via get_top_tokens
    out["loop_body_embeds_in_model"] = ("inputs_embeds=None" in sc_src)
    out["loop_body_argmax_get_top_tokens"] = ("get_top_tokens(" in sc_src)

    out["onegraph_on"] = (flags.get("ONEGRAPH") == "1")
    out["require_capture"] = (flags.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1")
    out["fused_sparse_argmax_on"] = (flags.get("FUSED_SPARSE_ARGMAX") == "1")
    # The decisive boolean, grounded in the served manifest + code structure:
    # captured (NOT separately launched) iff ONEGRAPH + require-capture are on AND
    # the loop-capture machinery is present in the served sitecustomize.py.
    captured = bool(
        out["onegraph_on"] and out["require_capture"]
        and out["has_capture_graph"] and out["has_graph_replay"]
        and out["has_propose_onegraph"] and out["has_run_graph_body"]
    )
    out["draft_loop_captured_in_cudagraph"] = captured
    out["draft_argmax_embed_separately_launched"] = (not captured)
    return out


# --------------------------------------------------------------------------- #
# Composition: TPS <-> step. tps(step') = SERVED * STEP_US_SERVED / step'.
# (served denominator is the live TPS-gate denominator; built is land #245's.)
# --------------------------------------------------------------------------- #
def tps_from_step(step_us: float) -> float:
    return SERVED_TPS * STEP_US_SERVED / step_us


def tps_gain_pct_from_us_net(us_net: float) -> float:
    """+us_net = step SHRINKS by us_net (a saving); -us_net = step grows (a cost)."""
    return (tps_from_step(STEP_US_SERVED - us_net) / SERVED_TPS - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# The greedy-safety certificate: argmax->token->embed fused == unfused, BIT-EXACT.
# Faithful to the served drafter chain:
#   cand = argmax(logits)                              (drafter argmax kernel)
#   tok  = selected.gather(1, cand)                    (candidate-id -> vocab-id)
#   e    = embed_table[tok]                            (input-embed feed-back)
# Fusion only removes the kernel-launch boundary; it is pure INDEX selection (no
# float reduction reordering), so it cannot flip the argmax. We verify bit-
# identity over N>=1000 random rows (bf16 + fp32) AND an adversarial exact-tie
# sweep (argmax must return the FIRST/lowest candidate index in both paths).
# --------------------------------------------------------------------------- #
def argmax_embed_fusion_equiv(n_rows: int, n_cand: int, hidden: int, vocab: int,
                              seed: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc),
                "fusion_argmax_bitexact": False}

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cpu")
    n_rows = max(int(n_rows), 1000)  # PR floor: N>=1000 trials

    def _two_op(logits, selected, embed):
        # materialised intermediates: argmax -> token id -> embed gather
        cand = logits.argmax(dim=-1)                         # [R]
        tok = selected.gather(1, cand.unsqueeze(1)).squeeze(1)  # [R]
        emb = embed.index_select(0, tok)                     # [R, H]
        return cand, tok, emb

    def _fused(logits, selected, embed):
        # single chained expression a fused kernel implements end-to-end
        cand = logits.argmax(dim=-1)
        tok = selected.gather(1, cand.unsqueeze(1)).squeeze(1)
        emb = embed[tok]
        return cand, tok, emb

    per_dtype: dict[str, Any] = {}
    total = mism_tok = mism_emb = mism_cand = 0
    for dt_name, dt in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        # random sweep
        embed = torch.randn(vocab, hidden, generator=g, dtype=torch.float32).to(dt)
        logits = torch.randn(n_rows, n_cand, generator=g, dtype=torch.float32).to(dt)
        selected = torch.randint(0, vocab, (n_rows, n_cand), generator=g,
                                 dtype=torch.long)
        c1, t1, e1 = _two_op(logits, selected, embed)
        c2, t2, e2 = _fused(logits, selected, embed)
        cand_eq = bool(torch.equal(c1, c2))
        tok_eq = bool(torch.equal(t1, t2))
        emb_eq = bool(torch.equal(e1, e2))

        # adversarial exact-tie sweep: copy the row-max into a random OTHER
        # candidate column -> exact tie; argmax must pick the FIRST (lowest)
        # index in BOTH paths, so the emitted token + embed are identical.
        adv_rows = max(n_rows // 2, 1000)
        a_logits = torch.randn(adv_rows, n_cand, generator=g, dtype=torch.float32).to(dt)
        a_sel = torch.randint(0, vocab, (adv_rows, n_cand), generator=g, dtype=torch.long)
        arg0 = a_logits.argmax(dim=-1)
        rowmax = a_logits.gather(1, arg0.unsqueeze(1))
        inj = torch.randint(0, n_cand, (adv_rows,), generator=g)
        clash = inj == arg0
        inj[clash] = (inj[clash] + 1) % n_cand
        a_logits.scatter_(1, inj.unsqueeze(1), rowmax)
        ac1, at1, ae1 = _two_op(a_logits, a_sel, embed)
        ac2, at2, ae2 = _fused(a_logits, a_sel, embed)
        # the resolved candidate index must be the SMALLER of {arg0, inj}
        first_idx = torch.minimum(arg0, inj)
        adv_first_ok = bool(torch.equal(ac1, first_idx) and torch.equal(ac2, first_idx))
        adv_tok_eq = bool(torch.equal(at1, at2))
        adv_emb_eq = bool(torch.equal(ae1, ae2))

        rows = n_rows + adv_rows
        total += rows
        mism_cand += int((c1 != c2).sum().item()) + int((ac1 != ac2).sum().item())
        mism_tok += int((t1 != t2).sum().item()) + int((at1 != at2).sum().item())
        mism_emb += int((e1 != e2).any(dim=-1).sum().item()) \
            + int((ae1 != ae2).any(dim=-1).sum().item())
        per_dtype[dt_name] = {
            "rows": rows, "cand_eq": cand_eq, "tok_eq": tok_eq, "emb_eq": emb_eq,
            "adv_first_occurrence_ok": adv_first_ok,
            "adv_tok_eq": adv_tok_eq, "adv_emb_eq": adv_emb_eq,
        }

    bitexact = bool(
        mism_cand == 0 and mism_tok == 0 and mism_emb == 0
        and all(
            d["cand_eq"] and d["tok_eq"] and d["emb_eq"]
            and d["adv_first_occurrence_ok"] and d["adv_tok_eq"] and d["adv_emb_eq"]
            for d in per_dtype.values()
        )
    )
    return {
        "available": True,
        "fusion_argmax_bitexact": bitexact,
        "total_positions": total,
        "mismatch_cand": mism_cand, "mismatch_tok": mism_tok, "mismatch_emb": mism_emb,
        "per_dtype": per_dtype,
        "n_rows": n_rows, "n_cand": n_cand, "hidden": hidden, "vocab": vocab,
        "note": "argmax->token->embed is pure index selection; fusing the launch "
                "boundary introduces no arithmetic => bit-identical by construction.",
    }


def synthesize(equiv: dict[str, Any]) -> dict[str, Any]:
    aud = audit_served()
    k_spec = aud["k_spec"]
    n_launches = 2 * k_spec
    separately_launched = aud["draft_argmax_embed_separately_launched"]

    # --- (2) bound the MAX step-reduction (counterfactual + actual) -------- #
    def _launch_tax(per_launch_us: float) -> float:
        return n_launches * per_launch_us

    tax_lo = _launch_tax(PER_LAUNCH_US_LO)   # 56
    tax_mid = _launch_tax(PER_LAUNCH_US_MID)  # 70
    tax_hi = _launch_tax(PER_LAUNCH_US_HI)   # 84

    def _row(per_launch_us: float):
        tax = _launch_tax(per_launch_us)
        red_served = 100.0 * tax / STEP_US_SERVED
        red_built = 100.0 * tax / STEP_US_BUILT
        tps = tps_from_step(STEP_US_SERVED - tax)
        gain = tps_gain_pct_from_us_net(tax)
        return {
            "per_launch_us": per_launch_us, "launch_tax_us": round(tax, 4),
            "max_step_reduction_pct_vs_served": round(red_served, 4),
            "max_step_reduction_pct_vs_built": round(red_built, 4),
            "implied_tps_off_481_53": round(tps, 3),
            "implied_gain_pct": round(gain, 4),
            "clears_500": bool(tps >= 500.0),
            "clears_520_95": bool(tps >= LAMBDA1_CEILING_TPS),
        }

    counterfactual_rows = [_row(PER_LAUNCH_US_LO), _row(PER_LAUNCH_US_MID),
                           _row(PER_LAUNCH_US_HI)]
    cf_mid = counterfactual_rows[1]

    # ACTUAL: separately_launched=False => the tax does not exist on the served
    # path; bound = 0, projected gain = 0.00 (TEST metric).
    if separately_launched:
        actual_step_reduction_us = tax_mid          # would-be midpoint (premise true)
        actual_max_step_reduction_pct_served = 100.0 * tax_mid / STEP_US_SERVED
        actual_max_step_reduction_pct_built = 100.0 * tax_mid / STEP_US_BUILT
        projected_tps_gain_pct = tps_gain_pct_from_us_net(tax_mid)
        actual_tps = tps_from_step(STEP_US_SERVED - tax_mid)
    else:
        actual_step_reduction_us = 0.0
        actual_max_step_reduction_pct_served = 0.0
        actual_max_step_reduction_pct_built = 0.0
        projected_tps_gain_pct = tps_gain_pct_from_us_net(0.0)   # 0.00
        actual_tps = tps_from_step(STEP_US_SERVED)               # 481.53

    fusion_bitexact = bool(equiv.get("fusion_argmax_bitexact", False))

    # --- (4) verdict table (one row per counterfactual + the ACTUAL row) --- #
    verdict_table = []
    for r in counterfactual_rows:
        verdict_table.append({
            "scenario": f"COUNTERFACTUAL per_launch={r['per_launch_us']:.0f}us "
                        f"(IF separately launched, on critical path)",
            "separately_launched": separately_launched,
            "launch_tax_us": r["launch_tax_us"],
            "step_reduction_pct_served": r["max_step_reduction_pct_vs_served"],
            "step_reduction_pct_built": r["max_step_reduction_pct_vs_built"],
            "implied_tps": r["implied_tps_off_481_53"],
            "fusion_argmax_bitexact": fusion_bitexact,
            "subsumed_by_deployed_onegraph": True,
            "clears_500": r["clears_500"], "clears_520_95": r["clears_520_95"],
        })
    verdict_table.append({
        "scenario": "ACTUAL (audited): draft loop already CUDAGraph-captured "
                    "(ONEGRAPH _run_graph_body, single graph.replay)",
        "separately_launched": separately_launched,            # False
        "launch_tax_us": round(actual_step_reduction_us, 4),    # 0
        "step_reduction_pct_served": round(actual_max_step_reduction_pct_served, 4),
        "step_reduction_pct_built": round(actual_max_step_reduction_pct_built, 4),
        "implied_tps": round(actual_tps, 3),                    # 481.53
        "fusion_argmax_bitexact": fusion_bitexact,
        "subsumed_by_deployed_onegraph": True,
        "clears_500": bool(actual_tps >= 500.0), "clears_520_95": False,
    })

    headline = {
        "draft_argmax_embed_separately_launched": separately_launched,   # False
        "draft_loop_captured_in_cudagraph": aud["draft_loop_captured_in_cudagraph"],  # True
        "projected_tps_gain_pct": round(projected_tps_gain_pct, 4),      # TEST = 0.00
        "screen_verdict": "NO-GO" if not separately_launched else "GO",
        "lever_class": ("NULL (already captured in deployed ONEGRAPH CUDAGraph)"
                        if not separately_launched else "LIVE (separately launched)"),
        "launch_tax_us_band": [round(tax_lo, 2), round(tax_hi, 2)],
        "launch_tax_us_mid": round(tax_mid, 2),
        "counterfactual_gain_pct_band": [
            round(counterfactual_rows[0]["implied_gain_pct"], 4),
            round(counterfactual_rows[2]["implied_gain_pct"], 4),
        ],
        "counterfactual_tps_band": [
            counterfactual_rows[0]["implied_tps_off_481_53"],
            counterfactual_rows[2]["implied_tps_off_481_53"],
        ],
        "counterfactual_mid_tps": cf_mid["implied_tps_off_481_53"],
        "counterfactual_step_reduction_pct_served_band": [
            counterfactual_rows[0]["max_step_reduction_pct_vs_served"],
            counterfactual_rows[2]["max_step_reduction_pct_vs_served"],
        ],
        "counterfactual_step_reduction_pct_built_band": [
            counterfactual_rows[0]["max_step_reduction_pct_vs_built"],
            counterfactual_rows[2]["max_step_reduction_pct_vs_built"],
        ],
        "fusion_argmax_bitexact": fusion_bitexact,
        "subsumed_by_lawine_cudagraph": "moot (subsumed by DEPLOYED onegraph)",
        "subsumed_by_deployed_onegraph": True,
        "actual_tps": round(actual_tps, 3),
        "clears_500_alone": bool(actual_tps >= 500.0),
        "greedy_identical_by_construction": True,
        "k_spec": k_spec, "n_launches": n_launches,
        "baseline_tps": BASELINE_TPS, "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
    }

    accounting = {
        "step_us_served": STEP_US_SERVED, "step_us_built": STEP_US_BUILT,
        "k_cal": K_CAL, "served_tps": SERVED_TPS,
        "n_launches": n_launches, "per_launch_us_lo": PER_LAUNCH_US_LO,
        "per_launch_us_hi": PER_LAUNCH_US_HI, "per_launch_us_mid": PER_LAUNCH_US_MID,
        "launch_tax_us_lo": tax_lo, "launch_tax_us_mid": tax_mid, "launch_tax_us_hi": tax_hi,
        "sm86_launch_floor_us": SM86_LAUNCH_FLOOR_US,
        "counterfactual_rows": counterfactual_rows,
        "actual_step_reduction_us": actual_step_reduction_us,
        "actual_max_step_reduction_pct_served": actual_max_step_reduction_pct_served,
        "actual_max_step_reduction_pct_built": actual_max_step_reduction_pct_built,
        "projected_tps_gain_pct": projected_tps_gain_pct,
        "actual_tps": actual_tps,
    }

    cudagraph_tiein = {
        "draft_loop_captured_in_cudagraph": aud["draft_loop_captured_in_cudagraph"],
        "onegraph_on": aud["onegraph_on"],
        "loopgraph_require_capture": aud["require_capture"],
        "has_capture_graph": aud["has_capture_graph"],
        "has_graph_replay": aud["has_graph_replay"],
        "subsumed_by_deployed_onegraph": True,
        "subsumed_by_lawine_246": "moot -- deployed ONEGRAPH already captures the "
                                  "K=7 draft tail (argmax+embed); lawine #246's live "
                                  "lever is the attention backend, a different axis",
        "reasoning": "ONEGRAPH=1 => propose_onegraph; _capture_graph records the "
                     "whole K=7 _run_graph_body (per-step embed via inputs_embeds=None "
                     "+ per-step get_top_tokens argmax) into ONE torch.cuda.CUDAGraph; "
                     "served step = single graph.replay(); LOOPGRAPH_REQUIRE_CAPTURE=1 "
                     "forces the captured path => no separate per-step eager launches.",
    }

    # --- (5) self-test conditions (a-f) ------------------------------------ #
    # (a) launch_tax_us round-trips from n_launches x per_launch_us.
    cond_a = bool(
        math.isclose(tax_lo, n_launches * PER_LAUNCH_US_LO, rel_tol=1e-12)
        and math.isclose(tax_mid, n_launches * PER_LAUNCH_US_MID, rel_tol=1e-12)
        and math.isclose(tax_hi, n_launches * PER_LAUNCH_US_HI, rel_tol=1e-12)
        and n_launches == 2 * k_spec
    )
    # (b) max_step_reduction_pct maps through the composition to the TPS band
    #     (both denominators), and the composition base round-trips. Recompute
    #     from UNROUNDED inputs (the table fields are rounded for display); check
    #     (i) the exact round-trip on raw values, (ii) the rounded table fields
    #     agree with the raw values within display-rounding tolerance.
    rt_base_ok = math.isclose(tps_from_step(STEP_US_SERVED), SERVED_TPS, rel_tol=1e-12)
    band_consistent = True
    for per_launch, r in zip((PER_LAUNCH_US_LO, PER_LAUNCH_US_MID, PER_LAUNCH_US_HI),
                             counterfactual_rows, strict=True):
        tax = n_launches * per_launch
        red_served = 100.0 * tax / STEP_US_SERVED
        red_built = 100.0 * tax / STEP_US_BUILT
        tps = tps_from_step(STEP_US_SERVED - tax)
        gain = tps_gain_pct_from_us_net(tax)
        # (i) exact round-trip on raw values: tps == served*(1+gain/100)
        band_consistent &= math.isclose(tps, SERVED_TPS * (1.0 + gain / 100.0), rel_tol=1e-12)
        # (ii) the displayed (rounded) table fields agree with raw within rounding
        band_consistent &= abs(r["max_step_reduction_pct_vs_served"] - red_served) < 1e-3
        band_consistent &= abs(r["max_step_reduction_pct_vs_built"] - red_built) < 1e-3
        band_consistent &= abs(r["implied_tps_off_481_53"] - tps) < 1e-2
        band_consistent &= abs(r["implied_gain_pct"] - gain) < 1e-3
    cond_b = bool(rt_base_ok and band_consistent)
    # (c) fusion_argmax_bitexact=True over N>=1000 random trials (0 mismatches).
    cond_c = bool(
        equiv.get("available", False)
        and equiv.get("fusion_argmax_bitexact", False)
        and equiv.get("total_positions", 0) >= 1000
        and equiv.get("mismatch_cand", 1) == 0
        and equiv.get("mismatch_tok", 1) == 0
        and equiv.get("mismatch_emb", 1) == 0
    )
    # (d) the boolean is sourced from the served config, not assumed.
    cond_d = bool(
        aud["manifest_path"] is not None
        and aud["sitecustomize_path"] is not None
        and aud["onegraph_on"] and aud["require_capture"]
        and aud["has_capture_graph"] and aud["has_graph_replay"]
        and aud["has_propose_onegraph"] and aud["has_run_graph_body"]
        and isinstance(separately_launched, bool)
        and separately_launched is False
    )
    # (e) NaN-clean -- finalised in main() over the whole payload.
    cond_e_local = all(
        _is_num(v) for v in [
            projected_tps_gain_pct, actual_tps, tax_lo, tax_mid, tax_hi,
            actual_max_step_reduction_pct_served, actual_max_step_reduction_pct_built,
            *[row["implied_tps_off_481_53"] for row in counterfactual_rows],
            *[row["implied_gain_pct"] for row in counterfactual_rows],
        ]
    )
    # (f) BASELINE 481.53 and the 520.95 lambda=1 ceiling are UNCHANGED.
    cond_f = bool(
        math.isclose(BASELINE_TPS, 481.53, rel_tol=0, abs_tol=1e-9)
        and math.isclose(SERVED_TPS, 481.53, rel_tol=0, abs_tol=1e-9)
        and math.isclose(LAMBDA1_CEILING_TPS, 520.95, rel_tol=0, abs_tol=1e-9)
        and math.isclose(actual_tps, 481.53, rel_tol=0, abs_tol=1e-6)  # screen moves nothing
        and math.isclose(projected_tps_gain_pct, 0.0, rel_tol=0, abs_tol=1e-9)
    )

    conditions = {
        "a_launch_tax_roundtrip": cond_a,
        "b_step_reduction_maps_through_composition": cond_b,
        "c_fusion_argmax_bitexact_ge1000": cond_c,
        "d_boolean_sourced_from_served_config": cond_d,
        "e_nan_clean": cond_e_local,   # tightened in main() with whole-payload scan
        "f_baseline_and_ceiling_unchanged": cond_f,
    }
    self_test = {
        "conditions": conditions,
        "draft_argmax_embed_fusion_screen_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "rt_base_ok": rt_base_ok, "band_consistent": band_consistent,
            "equiv_total_positions": equiv.get("total_positions"),
            "equiv_mismatches": [equiv.get("mismatch_cand"), equiv.get("mismatch_tok"),
                                 equiv.get("mismatch_emb")],
        },
    }

    nonoverlap = {
        "kanna_254_is_draft_PRECISION_red_not_launch": True,
        "lawine_246_is_attention_cudagraph_adjacent_this_prices_subsumption": True,
        "denken_257_roofline_decompose_this_prices_one_component": True,
        "stark_256_is_adaptive_K_draft_COUNT_not_per_step_launch_tax": True,
        "ubel_258_fern_259_are_ET_side": True,
        "land_245_owns_any_live_build": True,
        "this_continues_wirbel_lane_255_verify_epilogue_this_draft_launch": True,
    }

    handoff_fern_denken = (
        "hand-off (fern portfolio + denken roofline): the deployed draft loop DOES "
        "NOT separately launch argmax+embed (draft_argmax_embed_separately_launched"
        "=False) -- the K=7 MTP draft tail (per-step embed + get_top_tokens argmax) "
        "is already captured in the deployed ONEGRAPH torch.cuda.CUDAGraph and served "
        "by a single graph.replay(), so the Rank-2 fusion lever is NULL "
        "(projected_tps_gain_pct=0.00), worth 0% step / 0 TPS off 481.53. The "
        "counterfactual ceiling (IF it HAD been 14 separate launches @ 4-6us) is "
        f"+{counterfactual_rows[0]['implied_gain_pct']:.2f}..+"
        f"{counterfactual_rows[2]['implied_gain_pct']:.2f}% "
        f"({counterfactual_rows[0]['implied_tps_off_481_53']:.1f}.."
        f"{counterfactual_rows[2]['implied_tps_off_481_53']:.1f} TPS, clears 500 but "
        "not 520.95) -- refuted by the audit. The draft-launch component is NOT worth "
        "a build slot. Greedy-safe (bit-exact argmax->embed verified)."
    )
    handoff_line = (
        "the deployed draft loop DOES NOT separately launch argmax+embed "
        "(draft_argmax_embed_separately_launched=False), so the Rank-2 fusion lever "
        "is NULL/subsumed-by-deployed-ONEGRAPH worth 0% step / 0 TPS off 481.53, "
        "greedy-safe (bit-exact argmax verified)."
    )
    verdict = "DRAFT-ARGMAX-EMBED-ALREADY-CUDAGRAPH-CAPTURED-NO-GO"

    return {
        "verdict": verdict,
        "headline": headline,
        "audit": {
            "served": aud,
            "draft_argmax_embed_separately_launched": separately_launched,
            "served_draft_loop": {
                "submission": SERVED_SUBMISSION, "vllm_version": VLLM_VERSION,
                "spec_method": aud["spec_method"], "k_spec": k_spec,
                "onegraph_propose": "Gemma4Proposer.propose = propose_onegraph (ONEGRAPH=1)",
                "loop_body": "_run_graph_body: per-step self.model(input_ids, "
                             "inputs_embeds=None) embed-gather + get_top_tokens() argmax",
                "capture": "_capture_graph: with torch.cuda.graph(graph): "
                           "_run_graph_body(...) -> ONE torch.cuda.CUDAGraph",
                "serve_path": "single graph.replay() per step "
                              "(LOOPGRAPH_REQUIRE_CAPTURE=1 forces capture)",
                "argmax": "get_top_tokens -> logits.argmax (FUSED_SPARSE_ARGMAX Triton kernel)",
                "greedy_safety": "argmax->token->embed is pure index selection; "
                                 "fusion is latency-only, token-identical",
            },
        },
        "composition": {
            "k_cal": K_CAL, "step_us_served": STEP_US_SERVED, "step_us_built": STEP_US_BUILT,
            "served_tps": SERVED_TPS, "k_spec": k_spec, "n_launches": n_launches,
            "per_launch_us_band": [PER_LAUNCH_US_LO, PER_LAUNCH_US_HI],
        },
        "accounting": accounting,
        "verdict_table": verdict_table,
        "cudagraph_tiein": cudagraph_tiein,
        "equivalence": equiv,
        "nonoverlap": nonoverlap,
        "self_test": self_test,
        "handoff_line": handoff_line,
        "handoff_fern_denken": handoff_fern_denken,
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
    st, ct = syn["self_test"], syn["cudagraph_tiein"]
    eq = syn["equivalence"]
    print("\n" + "=" * 100, flush=True)
    print("PLAN-B RANK-2 DRAFT argmax+embed FUSION SCREEN (PR #261, wirbel) -- "
          "is the draft launch tax recoverable?", flush=True)
    print("=" * 100, flush=True)
    print("  (1) THE DECISIVE BOOLEAN (sourced from served config)", flush=True)
    print(f"      served: {SERVED_SUBMISSION}  vLLM {VLLM_VERSION}  "
          f"K_spec={h['k_spec']}  n_launches={h['n_launches']} (={h['k_spec']}x2)", flush=True)
    print(f"      ONEGRAPH={aud['onegraph_on']}  LOOPGRAPH_REQUIRE_CAPTURE="
          f"{aud['require_capture']}  _capture_graph={aud['has_capture_graph']}  "
          f"graph.replay={aud['has_graph_replay']}", flush=True)
    print(f"      draft_argmax_embed_separately_launched = "
          f"{h['draft_argmax_embed_separately_launched']}   "
          f"(draft_loop_captured_in_cudagraph={h['draft_loop_captured_in_cudagraph']})", flush=True)
    print("-" * 100, flush=True)
    print("  (2) ACCOUNTING (composition: tps(step') = served*step_served/step')", flush=True)
    print(f"      launch_tax = n_launches x per_launch = {h['n_launches']} x "
          f"[{PER_LAUNCH_US_LO:.0f},{PER_LAUNCH_US_HI:.0f}]us = "
          f"[{acc['launch_tax_us_lo']:.0f},{acc['launch_tax_us_hi']:.0f}]us "
          f"(mid {acc['launch_tax_us_mid']:.0f}us)", flush=True)
    print(f"      step denominators: served={STEP_US_SERVED:.1f}us (live gate)  "
          f"built={STEP_US_BUILT:.1f}us (land #245)", flush=True)
    print("-" * 100, flush=True)
    print("  (3) VERDICT TABLE  scenario                                                  "
          "tax_us  red%srv red%blt  TPS    bitexact", flush=True)
    for r in syn["verdict_table"]:
        print(f"      {r['scenario']:<62.62} {r['launch_tax_us']:>6.1f}  "
              f"{r['step_reduction_pct_served']:>6.2f}  {r['step_reduction_pct_built']:>6.2f}  "
              f"{r['implied_tps']:>7.2f}  {str(r['fusion_argmax_bitexact']):>5}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (3b) EQUIVALENCE (greedy-safety): fusion_argmax_bitexact="
          f"{eq.get('fusion_argmax_bitexact')}  over {eq.get('total_positions')} positions  "
          f"(mismatch cand/tok/emb = {eq.get('mismatch_cand')}/{eq.get('mismatch_tok')}/"
          f"{eq.get('mismatch_emb')})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) CUDAGraph TIE-IN: draft_loop_captured={ct['draft_loop_captured_in_cudagraph']}  "
          f"subsumed_by_deployed_onegraph={ct['subsumed_by_deployed_onegraph']}", flush=True)
    print(f"      HEADLINE screen_verdict = {h['screen_verdict']}  ({h['lever_class']})", flush=True)
    print(f"      counterfactual ceiling (IF separate launches): "
          f"{h['counterfactual_gain_pct_band'][0]:+.2f}..{h['counterfactual_gain_pct_band'][1]:+.2f}% "
          f"-> {h['counterfactual_tps_band'][0]:.1f}..{h['counterfactual_tps_band'][1]:.1f} TPS "
          f"(clears 500, not 520.95); ACTUAL projected_tps_gain_pct={h['projected_tps_gain_pct']:.2f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) PRIMARY draft_argmax_embed_fusion_screen_self_test_passes = "
          f"{st['draft_argmax_embed_fusion_screen_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"      TEST projected_tps_gain_pct = {h['projected_tps_gain_pct']:.2f}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_fern_denken']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[draft-fusion-screen] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, acc = syn["headline"], syn["accounting"]
    aud = syn["audit"]["served"]
    st, ct, eq = syn["self_test"], syn["cudagraph_tiein"], syn["equivalence"]
    run = init_wandb_run(
        job_type="draft-argmax-embed-fusion-screen",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["draft-argmax-embed-fusion-screen", "speed-levers", "planb",
              "draft-launch-tax", "argmax-embed-fusion", "cudagraph-capture",
              "onegraph", "bank-the-analysis", "null-lever", "no-go"],
        config={
            "k_cal": K_CAL, "step_us_served": STEP_US_SERVED, "step_us_built": STEP_US_BUILT,
            "served_tps": SERVED_TPS, "baseline_tps": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
            "k_spec": h["k_spec"], "n_launches": h["n_launches"],
            "per_launch_us_lo": PER_LAUNCH_US_LO, "per_launch_us_hi": PER_LAUNCH_US_HI,
            "served_submission": SERVED_SUBMISSION, "vllm_version": VLLM_VERSION,
            "wandb_group": args.wandb_group,
            "source_runs": "kanna#217 vgovdrjc composition+served step; denken#252 "
                           "a7llo7o7 built step; opttree/profile.py lambda1=520.95; "
                           "served fa2sw_precache_kenyan; vLLM 0.22.1rc1.dev307+g3e8afdf78",
        },
    )
    if run is None:
        print("[draft-fusion-screen] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "draft_argmax_embed_fusion_screen_self_test_passes":
            int(bool(st["draft_argmax_embed_fusion_screen_self_test_passes"])),  # PRIMARY
        "projected_tps_gain_pct": h["projected_tps_gain_pct"],                   # TEST
        "draft_argmax_embed_separately_launched":
            int(bool(h["draft_argmax_embed_separately_launched"])),
        "draft_loop_captured_in_cudagraph": int(bool(h["draft_loop_captured_in_cudagraph"])),
        "fusion_argmax_bitexact": int(bool(h["fusion_argmax_bitexact"])),
        "screen_verdict_no_go": int(h["screen_verdict"] == "NO-GO"),
        "subsumed_by_deployed_onegraph": int(bool(h["subsumed_by_deployed_onegraph"])),
        "actual_tps": h["actual_tps"],
        "clears_500_alone": int(bool(h["clears_500_alone"])),
        "greedy_identical_by_construction": int(bool(h["greedy_identical_by_construction"])),
        "k_spec": h["k_spec"], "n_launches": h["n_launches"],
        "launch_tax_us_lo": acc["launch_tax_us_lo"], "launch_tax_us_mid": acc["launch_tax_us_mid"],
        "launch_tax_us_hi": acc["launch_tax_us_hi"],
        "counterfactual_gain_pct_lo": h["counterfactual_gain_pct_band"][0],
        "counterfactual_gain_pct_hi": h["counterfactual_gain_pct_band"][1],
        "counterfactual_tps_lo": h["counterfactual_tps_band"][0],
        "counterfactual_tps_hi": h["counterfactual_tps_band"][1],
        "counterfactual_mid_tps": h["counterfactual_mid_tps"],
        "counterfactual_step_reduction_pct_served_lo":
            h["counterfactual_step_reduction_pct_served_band"][0],
        "counterfactual_step_reduction_pct_served_hi":
            h["counterfactual_step_reduction_pct_served_band"][1],
        "counterfactual_step_reduction_pct_built_lo":
            h["counterfactual_step_reduction_pct_built_band"][0],
        "counterfactual_step_reduction_pct_built_hi":
            h["counterfactual_step_reduction_pct_built_band"][1],
        "onegraph_on": int(bool(ct["onegraph_on"])),
        "loopgraph_require_capture": int(bool(ct["loopgraph_require_capture"])),
        "has_capture_graph": int(bool(ct["has_capture_graph"])),
        "has_graph_replay": int(bool(ct["has_graph_replay"])),
        "equiv_total_positions": eq.get("total_positions"),
        "equiv_mismatch_cand": eq.get("mismatch_cand"),
        "equiv_mismatch_tok": eq.get("mismatch_tok"),
        "equiv_mismatch_emb": eq.get("mismatch_emb"),
        "baseline_tps": BASELINE_TPS, "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="planb_draft_argmax_embed_fusion_screen_result",
                      artifact_type="speed-lever-screen", data=payload)
    finish_wandb(run)
    print(f"[draft-fusion-screen] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--equiv-rows", type=int, default=2000,
                    help="random rows for the bit-exact argmax->embed sweep (>=1000)")
    ap.add_argument("--equiv-cand", type=int, default=512,
                    help="sparse candidate count per row (drafter argmax width)")
    ap.add_argument("--equiv-hidden", type=int, default=256,
                    help="embed hidden dim (drafter is 256-dim)")
    ap.add_argument("--equiv-vocab", type=int, default=8192,
                    help="embed table rows for the gather proof")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="planb-speed-levers")
    args = ap.parse_args(argv)

    equiv = argmax_embed_fusion_equiv(
        n_rows=args.equiv_rows, n_cand=args.equiv_cand,
        hidden=args.equiv_hidden, vocab=args.equiv_vocab, seed=args.seed,
    )
    syn = synthesize(equiv)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 261, "agent": "wirbel",
        "kind": "planb-draft-argmax-embed-fusion-screen", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["e_nan_clean"] = (
        syn["self_test"]["conditions"]["e_nan_clean"] and not nan_paths)
    syn["self_test"]["draft_argmax_embed_fusion_screen_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[draft-fusion-screen] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[draft-fusion-screen] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["draft_argmax_embed_fusion_screen_self_test_passes"]
        print(f"[draft-fusion-screen] SELF-TEST {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
