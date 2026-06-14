#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-path PPL-margin bound (PR #166) — pure-analytic, CPU-only.

The launch's PPL validity rests on ``ppl <= 2.42`` (scorer hard gate). The measured
frontier PPL is **2.37667** (margin **0.0433**), measured on the LINEAR (M=1) stack.
The TREE path runs int4-Marlin verify GEMMs at batch width **M=32**, which carries
documented batch-variance — the same FP-reduction-order source as denken #158's
0.6169 spec-vs-AR completion divergence (Issue #124, ruled NOT a contract
violation). #158 proved per-token **argmax** fidelity; it did NOT bound how much the
M=32 batch-variance can inflate the **aggregate softmax mass** that PPL integrates
over. This harness closes that last unbounded validity dimension before launch.

It is a *synthesis*, not a new measurement: it imports committed outputs and
propagates them analytically. Two anchors, both committed on the advisor branch,
characterise ONE variance source (int4-Marlin batch-variance):

  * **MAGNITUDE** — ``research/validity/verify_argmax_margin/.../summary.json``
    (kanna #87): the REAL Marlin kernel at M=32 perturbs logits by
    ``max|Δlogit| = 0.25`` vs M=8, flipping **0** argmaxes over 65,536 positions.
  * **FREQUENCY** — ``research/descent_greedy_exact_harness/runs/.../result.json``
    (denken #158): M=1-AR-vs-batched completion token divergence
    ``40426/65536 = 0.6169`` (118/128 prompts), the compounded footprint of the
    same per-step batch-variance.

Central PPL anchor: #158's committed linear-stack cross-check ``2.376664808823738``
(cross-checked against the int4 split-KV verify prefill PPL 2.3766775 and the noise
-floor PPL 2.3766828 — all agree to 2.37667; ``2.42 - 2.37667 = 0.0433``).

------------------------------------------------------------------------------
STRUCTURAL FINDING (the model-free headline)
------------------------------------------------------------------------------
The scorer's PPL is **teacher-forced prefill** via ``prompt_logprobs`` (max_tokens:1)
— see ``research/validity/same_path_ppl.md`` §2(c)/§3. Speculative tree decode
(the M=32 verify batch) runs only in the **decode** phase; a ``prompt_logprobs``
request never enters it. The prefill that computes the scored logprobs is therefore
**M-invariant**: the scored tree-path PPL equals the scored linear-path PPL = 2.37667
and the M=32 batch-variance contributes **zero** to the gated quantity. The 0.0433
margin is not consumed by M=32.

(Scope: this covers the M=32 verify-batch dimension #158 characterised. If land #71's
tree submission ALSO changes the *prefill* chunk geometry, that is a distinct
batch-variance leg to audit separately — same caveat #158 raised for upstream logits.)

------------------------------------------------------------------------------
CONSERVATIVE TRANSPLANT BOUND (the corroborating leg the PR asked for)
------------------------------------------------------------------------------
Grant the worst-case counterfactual that the *decode* M=32 logit jitter DID land on
every *prefill*-scored token. Propagate per-token logit perturbation -> per-token NLL
perturbation -> aggregate PPL. PPL = exp(mean NLL), NLL_i = -log softmax(z_i)[t_i].

For a per-coordinate logit perturbation δ with ‖δ‖_∞ ≤ ε:
  * gradient   g_i = ∂NLL_i/∂z = -(1-p_t)·e_t + p_{k≠t},   ‖g_i‖_1 = 2(1-p_t) ≤ 2.
  * Hessian    H_i = diag(p_i) - p_i p_iᵀ  ⪰ 0  (softmax/logsumexp Jacobian, PSD),
               tr(H_i) = 1 - ‖p_i‖² ≤ 1.

(1) EXPECTED (mean-zero, argmax-preserving noise — the physical regime).
    Batch-variance is mean-zero FP-reduction noise (E[δ]=0). First order vanishes in
    expectation; the PSD Hessian gives a one-signed second-order bias:
        E[ΔNLL_i] = ½ tr(H_i Σ_δ) ≤ ½ σ² (1-‖p_i‖²) ≤ ½ σ²,   σ² = Var per logit.
    i.e. symmetric logit noise can ONLY increase PPL, by O(σ²). Over perturbed
    fraction f:  mean E[ΔNLL] ≤ f·½σ².

(2) WORST-CASE (mean-zero bias + a 6σ aggregate fluctuation).
    Var(ΔNLL_i) ≤ σ²‖g_i‖² ≤ 2σ². Aggregate-mean std ≤ √(f·2σ²/N); 6σ tail added.

(3) ADVERSARIAL Lipschitz (UNPHYSICAL upper bound — reported, not operative).
    Every perturbed token pushed worst-direction by the full ε:
        mean ΔNLL ≤ f·2ε·mean(1-p_t),  mean(1-p_t) ≤ 1 - 1/PPL (Jensen).
    This requires the rounding noise to conspire against the target token on every
    token — falsified by the 0 measured argmax flips and the symmetric ±1-bf16-ULP
    final-cast regime (verify_argmax_margin.md "Numerics regime").

σ is taken conservatively from the measured MAX: treating ε=0.25 as a per-logit std
(uniform[-ε,ε] ⇒ σ²=ε²/3) over-counts — the true per-logit jitter is ≪0.25 (M=16 is
bit-identical; median Δlogit ≈ 0). f is swept over {per-step flip ≈0.005, #158
divergence 0.6169, every-token 1.0}; the f=1 extreme is the binding worst case.

Run:
    python -m research.validity.tree_path_ppl_margin.ppl_margin_bound --self-test \
        --wandb-name denken/tree-path-ppl-margin-bound \
        --wandb-group tree-path-ppl-margin-bound
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

# --------------------------------------------------------------------------- #
# Paths to committed anchors (advisor-branch content; no external PR borrow).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

DEFAULT_DIVERGENCE_ANCHOR = (
    REPO_ROOT
    / "research/descent_greedy_exact_harness/runs/20260614T135333Z/greedy_exact_harness_result.json"
)
DEFAULT_MARGIN_ANCHOR = (
    REPO_ROOT / "research/validity/verify_argmax_margin/20260614T041541Z/summary.json"
)
# canonical scored-span PPL artifact (int4 split-KV verify prefill): central + N + NLL.
DEFAULT_PPL_ANCHOR = (
    REPO_ROOT / "research/profiling/splitkv_verify/ppl_patched/ppl_summary.json"
)
DEFAULT_NOISE_FLOOR_PPL = (
    REPO_ROOT / "research/tps_noise_floor/ppl_validity/ppl_check.summary.json"
)

PPL_CAP = 2.42
# #158's committed linear-stack PPL cross-check (the PR's "2.3777" central anchor).
PR158_LINEAR_PPL = 2.376664808823738

TOL_CENTRAL = 1e-9  # central (ε→0) must reproduce the measured PPL to this tolerance.
SIGMA_K = 6.0  # aggregate-fluctuation tail (6σ) for the worst-case.


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Anchor import (committed outputs only — no re-derivation).
# --------------------------------------------------------------------------- #
def load_anchors(
    divergence_path: Path,
    margin_path: Path,
    ppl_path: Path,
    noise_floor_path: Path,
) -> dict[str, Any]:
    """Import the committed batch-variance characterisation and the measured PPL."""
    div = _load_json(divergence_path)
    mrg = _load_json(margin_path)
    ppl = _load_json(ppl_path)

    ar = div["ar_aggregate"]
    divergence_rate = float(ar["divergence_rate"])
    num_divergent_prompts = int(ar["num_divergent"])
    num_identical_prompts = int(ar["num_prompts_compared"]) - num_divergent_prompts
    num_prompts = int(ar["num_prompts_compared"])
    total_divergent_tokens = int(ar["total_divergent_tokens"])
    total_completion_tokens = int(ar["total_tokens_compared"])
    pr158_ppl = float(div["self_test"]["ppl"]["value"])

    # MAGNITUDE: real Marlin kernel, M=32 vs M=8, 0 flips over 65,536 positions.
    eps_m32 = float(mrg["mwiden_max_abs_dlogit"]["32"])
    m32_flips = int(mrg["mwiden_flip_count_vs_refM8"]["32"])
    m16_dlogit = float(mrg["mwiden_max_abs_dlogit"]["16"])
    min_positive_margin = float(mrg.get("min_margin", 0.0))
    median_margin = float(mrg["median_margin"])

    # central anchor = #158's committed LINEAR-stack PPL (the PR's "2.3777" to import);
    # scored token count N from the canonical ppl_summary; int4 split-KV verify prefill
    # PPL + noise-floor PPL are cross-checks (all cluster at 2.37667, margin 0.0433).
    central_ppl = pr158_ppl
    num_scored_tokens = int(ppl["num_tokens"])
    central_nll_total = math.log(central_ppl) * num_scored_tokens

    cross_checks = {
        "pr158_linear_stack": pr158_ppl,
        "splitkv_verify_prefill": float(ppl["ppl"]),
    }
    try:
        nf = _load_json(noise_floor_path)
        cross_checks["tps_noise_floor"] = float(nf["ppl"])
    except Exception:  # cross-check is optional, never fatal
        pass
    cross_check_spread = max(cross_checks.values()) - min(cross_checks.values())

    return {
        "divergence_rate": divergence_rate,
        "num_divergent_prompts": num_divergent_prompts,
        "num_identical_prompts": num_identical_prompts,
        "num_prompts": num_prompts,
        "total_divergent_tokens": total_divergent_tokens,
        "total_completion_tokens": total_completion_tokens,
        "eps_m32_max_abs_dlogit": eps_m32,
        "m32_argmax_flips": m32_flips,
        "m16_max_abs_dlogit": m16_dlogit,
        "min_margin": min_positive_margin,
        "median_margin": median_margin,
        "central_ppl": central_ppl,
        "num_scored_tokens": num_scored_tokens,
        "central_nll_total": central_nll_total,
        "pr158_linear_ppl": pr158_ppl,
        "ppl_cross_checks": cross_checks,
        "ppl_cross_check_spread": cross_check_spread,
        "ppl_cap": PPL_CAP,
        "_paths": {
            "divergence": str(divergence_path),
            "margin": str(margin_path),
            "ppl": str(ppl_path),
        },
    }


# --------------------------------------------------------------------------- #
# Analytic propagation:  logit perturbation  ->  NLL perturbation  ->  PPL.
# --------------------------------------------------------------------------- #
def per_step_flip_rate(num_identical_prompts: int, num_prompts: int, comp_len: int) -> float:
    """Implied per-STEP argmax-flip probability q from P(completion identical).

    Completion divergence compounds: one per-step flip diverges the whole downstream
    context. If per-step flip prob is q, P(identical 512-token completion) = (1-q)^L.
    So q = 1 - (n_identical / n_prompts)^(1/L). This is the physically-correct
    perturbed fraction for *teacher-forced* PPL (no compounding), ~0.5%, far below the
    compounded 0.6169 completion-divergence rate.
    """
    if num_identical_prompts <= 0 or num_prompts <= 0 or comp_len <= 0:
        return float("nan")
    p_identical = num_identical_prompts / num_prompts
    return 1.0 - p_identical ** (1.0 / comp_len)


def ppl_bounds(
    *,
    central_ppl: float,
    eps: float,
    f: float,
    n_tokens: int,
    cap: float,
    sigma_k: float = SIGMA_K,
) -> dict[str, float]:
    """Closed-form PPL bound ladder for per-logit perturbation magnitude ε, perturbed
    fraction f, over n_tokens scored tokens. Mean-zero noise treated with std σ=ε/√3
    (uniform[-ε,ε]); see module docstring for the derivation."""
    L0 = math.log(central_ppl)  # central mean NLL/token
    sigma2 = (eps * eps) / 3.0  # conservative: measured MAX treated as per-logit std

    # (1) expected — second-order PSD-Hessian bias under mean-zero noise (≤ ½σ²/token).
    bias = f * 0.5 * sigma2
    expected_logppl = L0 + bias
    # (2) worst-case — bias + sigma_k·(aggregate-mean std). Var(ΔNLL_i) ≤ 2σ².
    fluct = sigma_k * math.sqrt(f * 2.0 * sigma2 / n_tokens) if n_tokens > 0 else 0.0
    worstcase_logppl = L0 + bias + fluct
    # (3) adversarial Lipschitz (unphysical). mean(1-p_t) ≤ 1 - 1/PPL (Jensen).
    mean_one_minus_pt_ub = 1.0 - math.exp(-L0)
    adversarial_logppl = L0 + f * 2.0 * eps * mean_one_minus_pt_ub
    return {
        "eps": eps,
        "f": f,
        "sigma2": sigma2,
        "central_logppl": L0,
        "expected_bias": bias,
        "worstcase_fluct": fluct,
        "central_ppl": central_ppl,  # ε→0 reproduces central exactly
        "expected_ppl": math.exp(expected_logppl),
        "worstcase_ppl": math.exp(worstcase_logppl),
        "adversarial_ppl": math.exp(adversarial_logppl),
        "expected_margin": cap - math.exp(expected_logppl),
        "worstcase_margin": cap - math.exp(worstcase_logppl),
        "adversarial_margin": cap - math.exp(adversarial_logppl),
    }


def breakeven_eps(
    *, central_ppl: float, f: float, n_tokens: int, cap: float, sigma_k: float = SIGMA_K
) -> float:
    """ε at which the WORST-CASE bound first touches the cap (bisection)."""
    lo, hi = 0.0, 4.0
    target = math.log(cap)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        wlog = ppl_bounds(
            central_ppl=central_ppl, eps=mid, f=f, n_tokens=n_tokens, cap=cap, sigma_k=sigma_k
        )
        wc = math.log(wlog["worstcase_ppl"])
        if wc < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Synthesis + self-test.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any]) -> dict[str, Any]:
    central = anchors["central_ppl"]
    cap = anchors["ppl_cap"]
    eps = anchors["eps_m32_max_abs_dlogit"]
    f_div = anchors["divergence_rate"]
    n_tokens = anchors["num_scored_tokens"]
    comp_len = (
        anchors["total_completion_tokens"] // anchors["num_prompts"]
        if anchors["num_prompts"]
        else 512
    )

    f_perstep = per_step_flip_rate(
        anchors["num_identical_prompts"], anchors["num_prompts"], comp_len
    )
    # frequency ladder: per-step flip (physical for PPL) < #158 divergence < every-token.
    freq_ladder = {
        "per_step_flip_rate": f_perstep,
        "pr158_completion_divergence": f_div,
        "every_token": 1.0,
    }
    bounds_by_f = {
        label: ppl_bounds(central_ppl=central, eps=eps, f=val, n_tokens=n_tokens, cap=cap)
        for label, val in freq_ladder.items()
        if _finite(val)
    }

    # primary anchored bound = #158 divergence frequency; binding = every-token extreme.
    anchored = bounds_by_f["pr158_completion_divergence"]
    binding = bounds_by_f["every_token"]

    # ε sensitivity at the binding (every-token) frequency: where would the bound break?
    eps_grid = [0.0, eps * 0.5, eps, eps * 1.5, eps * 2.0]
    eps_sensitivity = [
        {
            "eps": e,
            "eps_in_bf16_ulp_at_mag16": e / 0.125,
            **{
                k: ppl_bounds(central_ppl=central, eps=e, f=1.0, n_tokens=n_tokens, cap=cap)[k]
                for k in ("expected_ppl", "worstcase_ppl")
            },
        }
        for e in eps_grid
    ]
    breakeven_div = breakeven_eps(central_ppl=central, f=f_div, n_tokens=n_tokens, cap=cap)
    breakeven_all = breakeven_eps(central_ppl=central, f=1.0, n_tokens=n_tokens, cap=cap)

    # zero-perturbation reproduction check (central).
    central_repro = ppl_bounds(
        central_ppl=central, eps=0.0, f=1.0, n_tokens=n_tokens, cap=cap
    )["worstcase_ppl"]

    # ----- self-test conditions ----- #
    cond_central_reproduces = abs(central_repro - central) <= TOL_CENTRAL
    # conservative ordering, evaluated at the binding (every-token) frequency:
    cond_ordering = (
        central
        <= binding["expected_ppl"] + TOL_CENTRAL
        <= binding["worstcase_ppl"] + TOL_CENTRAL
        <= binding["adversarial_ppl"] + TOL_CENTRAL
    )
    # verdict gated on the MOST conservative (binding, every-token) worst case ≤ cap.
    ppl_margin_under_2p42 = binding["worstcase_ppl"] <= cap
    # documented (non-gating) diagnostic: the unphysical adversarial bound breaches.
    adversarial_breaches = binding["adversarial_ppl"] > cap

    self_test_passes = bool(
        cond_central_reproduces and cond_ordering and ppl_margin_under_2p42
    )

    tree_path_ppl_worst_case = binding["worstcase_ppl"]  # TEST metric (binding extreme)

    verdict = "SAFE" if ppl_margin_under_2p42 else "BREACH"
    handoff = _handoff_line(
        verdict=verdict,
        central=central,
        cap=cap,
        eps=eps,
        anchored_wc=anchored["worstcase_ppl"],
        binding_wc=binding["worstcase_ppl"],
        breakeven_div=breakeven_div,
        breakeven_all=breakeven_all,
        m32_flips=anchors["m32_argmax_flips"],
    )

    return {
        "self_test": {
            "ppl_margin_bound_self_test_passes": self_test_passes,
            "conditions": {
                "central_reproduces_2p377": cond_central_reproduces,
                "conservative_ordering": cond_ordering,
                "ppl_margin_under_2p42": ppl_margin_under_2p42,
            },
            "central_reproduction": {
                "computed_at_eps0": central_repro,
                "measured_central": central,
                "abs_err": abs(central_repro - central),
                "tol": TOL_CENTRAL,
            },
            "adversarial_breaches_but_unphysical": adversarial_breaches,
        },
        "structural_finding": {
            "scored_ppl_is_teacher_forced_prefill": True,
            "m32_is_decode_only": True,
            "scored_ppl_M_invariant": True,
            "structural_scored_tree_path_ppl": central,
            "margin_untouched_by_m32": cap - central,
            "ref": "research/validity/same_path_ppl.md §2(c)/§3",
        },
        "bounds_by_frequency": bounds_by_f,
        "frequency_ladder": freq_ladder,
        "anchored_bound_pr158_f": anchored,
        "binding_bound_every_token": binding,
        "eps_sensitivity_at_f1": eps_sensitivity,
        "breakeven_eps_at_pr158_f": breakeven_div,
        "breakeven_eps_at_f1": breakeven_all,
        "tree_path_ppl_worst_case": tree_path_ppl_worst_case,
        "tree_path_ppl_worst_case_anchored": anchored["worstcase_ppl"],
        "tree_path_ppl_expected": binding["expected_ppl"],
        "ppl_margin_under_2p42": ppl_margin_under_2p42,
        "adversarial_ppl_unphysical": binding["adversarial_ppl"],
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _handoff_line(
    *,
    verdict: str,
    central: float,
    cap: float,
    eps: float,
    anchored_wc: float,
    binding_wc: float,
    breakeven_div: float,
    breakeven_all: float,
    m32_flips: int,
) -> str:
    if verdict == "SAFE":
        return (
            f"SAFE: the {cap - central:.4f} PPL margin survives M=32 batch-variance. "
            f"(1) STRUCTURAL: the scored PPL is teacher-forced prefill (prompt_logprobs); "
            f"M=32 is decode-only, so the scored PPL is M-invariant = {central:.5f}, margin "
            f"untouched. (2) CONSERVATIVE TRANSPLANT (decode M=32 jitter ε={eps} applied to "
            f"prefill): worst-case PPL = {anchored_wc:.4f} at #158's f=0.6169, "
            f"{binding_wc:.4f} at the every-token extreme — both ≤ {cap}. The per-logit M=32 "
            f"perturbation would have to grow from the measured {eps} to ~{breakeven_div:.2f} "
            f"(#158 f) / ~{breakeven_all:.2f} (every-token f) to breach. Only an unphysical "
            f"adversarial-on-every-token model breaches — ruled out by {m32_flips} measured "
            f"argmax flips at M=32 and the symmetric ±1-bf16-ULP final-cast regime."
        )
    return (
        f"BREACH: conservative worst-case PPL = {binding_wc:.4f} > {cap}. The {cap - central:.4f} "
        f"margin does NOT survive M=32 batch-variance under the every-token model; a tighter "
        f"batch-variance bound (ε < ~{breakeven_all:.2f}) or a smaller verify width M would be "
        f"required."
    )


# --------------------------------------------------------------------------- #
# W&B logging (matches scripts/wandb_logging.py helper API; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict, anchors: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb,
            init_wandb_run,
            log_json_artifact,
            log_summary,
        )
    except Exception as exc:  # logging must never break the instrument
        print(f"[ppl-margin] wandb logging unavailable: {exc}", flush=True)
        return

    run = init_wandb_run(
        job_type="tree-path-ppl-margin-bound",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["tree-path-ppl-margin-bound", "validity-gate", "ppl-margin"],
        config={
            "ppl_cap": anchors["ppl_cap"],
            "central_ppl": anchors["central_ppl"],
            "eps_m32_max_abs_dlogit": anchors["eps_m32_max_abs_dlogit"],
            "m32_argmax_flips": anchors["m32_argmax_flips"],
            "divergence_rate": anchors["divergence_rate"],
            "num_scored_tokens": anchors["num_scored_tokens"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[ppl-margin] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = payload["self_test"]
    binding = payload["binding_bound_every_token"]
    anchored = payload["anchored_bound_pr158_f"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "ppl_margin_bound_self_test_passes": int(bool(st["ppl_margin_bound_self_test_passes"])),
        "tree_path_ppl_worst_case": payload["tree_path_ppl_worst_case"],
        # verdict + structural
        "ppl_margin_under_2p42": int(bool(payload["ppl_margin_under_2p42"])),
        "verdict_safe": int(payload["verdict"] == "SAFE"),
        "structural_scored_ppl_M_invariant": 1,
        "structural_scored_tree_path_ppl": payload["structural_finding"][
            "structural_scored_tree_path_ppl"
        ],
        "margin_untouched_by_m32": payload["structural_finding"]["margin_untouched_by_m32"],
        # bounds
        "tree_path_ppl_worst_case_anchored_f0p6169": anchored["worstcase_ppl"],
        "tree_path_ppl_expected": payload["tree_path_ppl_expected"],
        "worstcase_margin_under_cap": binding["worstcase_margin"],
        "anchored_worstcase_margin_under_cap": anchored["worstcase_margin"],
        "adversarial_ppl_unphysical": payload["adversarial_ppl_unphysical"],
        "adversarial_breaches": int(bool(st["adversarial_breaches_but_unphysical"])),
        "breakeven_eps_at_pr158_f": payload["breakeven_eps_at_pr158_f"],
        "breakeven_eps_at_f1": payload["breakeven_eps_at_f1"],
        # anchors echoed
        "anchor_central_ppl": anchors["central_ppl"],
        "anchor_eps_m32": anchors["eps_m32_max_abs_dlogit"],
        "anchor_m32_flips": anchors["m32_argmax_flips"],
        "anchor_divergence_rate": anchors["divergence_rate"],
        "ppl_cap": anchors["ppl_cap"],
        # self-test conditions
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        "central_reproduction_abs_err": st["central_reproduction"]["abs_err"],
    }
    # NaN-clean guard.
    summary = {k: v for k, v in summary.items() if not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="ppl_margin_bound_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[ppl-margin] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# Reporting helpers + CLI.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
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


def _print_report(anchors: dict, syn: dict) -> None:
    print("\n" + "=" * 78, flush=True)
    print("TREE-PATH PPL-MARGIN BOUND (PR #166) — pure-analytic synthesis", flush=True)
    print("=" * 78, flush=True)
    print(
        f"  anchors: central_ppl={anchors['central_ppl']:.9f} (cap={anchors['ppl_cap']}, "
        f"margin={anchors['ppl_cap'] - anchors['central_ppl']:.4f})  N={anchors['num_scored_tokens']}",
        flush=True,
    )
    print(
        f"           ε(M=32)={anchors['eps_m32_max_abs_dlogit']} max|Δlogit| @ "
        f"{anchors['m32_argmax_flips']} flips  |  M=16 Δ={anchors['m16_max_abs_dlogit']} (bit-identical)",
        flush=True,
    )
    print(
        f"           f(#158 divergence)={anchors['divergence_rate']:.4f} "
        f"({anchors['total_divergent_tokens']}/{anchors['total_completion_tokens']}, "
        f"{anchors['num_divergent_prompts']}/{anchors['num_prompts']} prompts)",
        flush=True,
    )
    print("-" * 78, flush=True)
    sf = syn["structural_finding"]
    print(
        f"  STRUCTURAL: scored PPL = teacher-forced prefill (M-invariant) = "
        f"{sf['structural_scored_tree_path_ppl']:.5f}  margin {sf['margin_untouched_by_m32']:.4f} "
        f"untouched by M=32",
        flush=True,
    )
    print("-" * 78, flush=True)
    print("  CONSERVATIVE TRANSPLANT (decode M=32 ε applied to prefill):", flush=True)
    for label, b in syn["bounds_by_frequency"].items():
        print(
            f"    f={b['f']:.4f} [{label:<26}]  expected={b['expected_ppl']:.5f}  "
            f"worst={b['worstcase_ppl']:.5f}  (margin {b['worstcase_margin']:+.4f})",
            flush=True,
        )
    print(
        f"  ADVERSARIAL (unphysical): {syn['adversarial_ppl_unphysical']:.4f} "
        f"{'BREACHES' if syn['self_test']['adversarial_breaches_but_unphysical'] else 'holds'} "
        f"— ruled out by {anchors['m32_argmax_flips']} flips + symmetric ±1-ULP regime",
        flush=True,
    )
    print(
        f"  break-even ε: {syn['breakeven_eps_at_pr158_f']:.3f} (#158 f) / "
        f"{syn['breakeven_eps_at_f1']:.3f} (every-token f)   [measured ε={anchors['eps_m32_max_abs_dlogit']}]",
        flush=True,
    )
    print("-" * 78, flush=True)
    st = syn["self_test"]
    print(
        f"  PRIMARY ppl_margin_bound_self_test_passes = {st['ppl_margin_bound_self_test_passes']}",
        flush=True,
    )
    print(f"  TEST    tree_path_ppl_worst_case        = {syn['tree_path_ppl_worst_case']:.5f}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 78, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--divergence-anchor", type=Path, default=DEFAULT_DIVERGENCE_ANCHOR)
    ap.add_argument("--margin-anchor", type=Path, default=DEFAULT_MARGIN_ANCHOR)
    ap.add_argument("--ppl-anchor", type=Path, default=DEFAULT_PPL_ANCHOR)
    ap.add_argument("--noise-floor-ppl", type=Path, default=DEFAULT_NOISE_FLOOR_PPL)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="tree-path-ppl-margin-bound")
    args = ap.parse_args(argv)

    anchors = load_anchors(
        args.divergence_anchor, args.margin_anchor, args.ppl_anchor, args.noise_floor_ppl
    )
    syn = synthesize(anchors)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 166,
        "agent": "denken",
        "kind": "tree-path-ppl-margin-bound",
        "anchors": anchors,
        **syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[ppl-margin] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(anchors, syn)

    out_dir = args.out_dir or (HERE / "runs" / created_at)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppl_margin_bound_result.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[ppl-margin] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload, anchors)

    if args.self_test:
        ok = syn["self_test"]["ppl_margin_bound_self_test_passes"] and payload["nan_clean"]
        print(f"[ppl-margin] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
