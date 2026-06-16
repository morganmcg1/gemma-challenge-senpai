#!/usr/bin/env python
"""PR #513 — Acceptance-rate-invariance of spec-dec distribution preservation.

Extends the PR #505 harness (`specdec_dist_preserve.py`). #505 proved the
spec-alive surgical-357 serve reproduces the target sampling distribution `p` on
the PUBLIC (#497) reasoning distributions (TV 0.0022 <= noise floor 0.0023).
This card asks the QUALITY twin of denken #489 / kanna #504,#508 (which priced
the SPEED side of the PRIVATE leaderboard acceptance shift):

    Does the private acceptance shift carry ANY downstream-quality exposure, or
    is the deployed rejection sampler distribution-exact at EVERY acceptance rate?

Mechanism (proved in #505, re-derived here from the pinned kernel source):
the deployed standard rejection rule with a deterministic (greedy) MTP draft
`x_d` and `draft_probs=None` (NO_DRAFT_PROBS) is, per draft position,

    accept x_d  w.p.  min(1, p(x_d))   (random kernel:926, draft_prob == 1)
    on reject   resample from p restricted to {y != x_d}   (recovered kernel:1006)

which is EXACTLY distribution-preserving: output ~ p for ANY draft token x_d.
The *acceptance rate* at that position is exactly p(x_d). So "draft quality /
acceptance" is the choice of x_d — a worse (private-OOD) drafter proposes
lower-p(x_d) tokens (lower acceptance) but the rejection-sampler OUTPUT is still
exactly p. Therefore:

  * sweeping the realized acceptance == sweeping which token is drafted, and
  * the output distribution is INVARIANT to that sweep (acceptance-rate-invariant
    preservation), so the private acceptance breach changes E[T]/TPS but NOT the
    output distribution -> a clean SPEED-only risk with ZERO quality exposure.

We *prove* this by driving the EXACT deployed `rejection_sample()` (pinned
server-venv vLLM 0.22.1rc1.dev307+g3e8afdf78, dixie-patched; the patch only adds
a temp=0 all_greedy fast-path short-circuit, so the random/recovered kernels are
stock) across:

  Leg 1  — single-position acceptance sweep. For FIXED target dists, draft every
           support token => realized acceptance spans from p_max down to p_min,
           bracketing the public ~0.387 anchor and descending into the private
           breach band (well below the deployed acceptance). The iid Monte-Carlo
           noise floor is held FIXED (same p, same N=M every trial emits), so
           TV(deployed, p) staying pinned at that floor across the whole
           acceptance range IS acceptance-rate-invariance.
           + corroboration on the REAL #497 reasoning answer distributions.

  Leg 2  — K=7 multi-position chaining (the deployed MTP spine). Confirm
           preservation does NOT accumulate error across the 7 draft positions.
           The kernel applies the identical standard rule at each position using
           that position's own target row; the only cross-position state is a
           `rejected` stop-flag (it halts the chain, it does NOT modify any
           distribution). So position i, when reached, preserves p_i with zero
           dependence on depth i. We measure it two ways: a depth-isolated probe
           (always-accept prefix => full statistics at every depth) and a natural
           chain (greedy drafts, varied per-position acceptance).

Usage:
  specdec_acceptance_invariance.py --self-test
  specdec_acceptance_invariance.py --out results.json [--real-logits real_p.pt]
"""

from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import torch

from vllm.v1.sample.rejection_sampler import rejection_sample, PLACEHOLDER_TOKEN_ID

DEVICE = torch.device("cuda")

# Public per-position MTP draft-acceptance anchor (surgical-357 on #497 public).
PUBLIC_ACCEPT_ANCHOR = 0.387
# "Private breach band": realized acceptance well below the deployed acceptance,
# the OOD-degraded-drafter regime that denken #489 / kanna #504,#508 priced for
# SPEED. We bracket and descend through it on the QUALITY axis here.
PRIVATE_BREACH_HI = 0.10
PRIVATE_BREACH_LO = 0.0

# Width of the noise-floor band: TV_deployed is "at the floor" iff it sits within
# mu +- K_SIGMA*sd of the iid-redraw band. K_SIGMA=6 is generous for TV's right
# skew and for the ~hundreds of acceptance points compared (multiple-comparison
# safe): under exact preservation z~O(1); a real shift gives z ~ sqrt(N) (>>6).
K_SIGMA = 6.0


# --------------------------------------------------------------------------- #
# Deployed-kernel drivers
# --------------------------------------------------------------------------- #
def _sm(batch: int) -> SimpleNamespace:
    """SamplingMetadata stub matching the deployed SAMPLING path: all_random
    (temp>0), generators empty. temperature != GREEDY_TEMPERATURE(0) per request
    so `is_greedy` is all-False -> only the standard random/recovered kernels
    run; all_greedy=False so the dixie temp=0 fast-path is skipped."""
    return SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(batch, dtype=torch.float32, device=DEVICE),
        generators={},
    )


def deployed_first_token_hist(
    p: torch.Tensor, draft_token_id: int, M: int, seed: int
) -> tuple[torch.Tensor, float]:
    """K=1: run the deployed rejection_sample M times on target dist p with a
    fixed deterministic draft token; return (first-token counts, accept_rate).

    target_logits = log(p): softmax(log p) == p exactly (the kernel re-softmaxes
    internally), reproducing the already temp/top-k/top-p processed sampling
    distribution that `apply_sampling_constraints` produces upstream in serve."""
    vocab = p.numel()
    torch.manual_seed(seed)
    logp = torch.log(p.clamp_min(0)).to(torch.float32)
    target_logits = logp.unsqueeze(0).expand(M, vocab).contiguous().to(DEVICE)
    draft_token_ids = torch.full((M,), int(draft_token_id), dtype=torch.int32, device=DEVICE)
    num_draft_tokens = [1] * M
    cu = torch.arange(1, M + 1, dtype=torch.int32, device=DEVICE)
    bonus = int(torch.argmax(p).item())
    bonus_token_ids = torch.full((M, 1), bonus, dtype=torch.int32, device=DEVICE)
    out = rejection_sample(
        draft_token_ids, num_draft_tokens, 1, cu, None,
        target_logits, bonus_token_ids, _sm(M),
    )
    first = out[:, 0].to(torch.int64)
    assert int((first == PLACEHOLDER_TOKEN_ID).sum()) == 0, "placeholder in K=1 first token"
    accept_rate = float((first.cpu() == int(draft_token_id)).to(torch.float64).mean())
    counts = torch.bincount(first.cpu(), minlength=vocab).to(torch.float64)
    return counts, accept_rate


def deployed_k7_chain(
    p_pos: list[torch.Tensor], draft_pos: list[int], B: int, seed: int
) -> torch.Tensor:
    """K-position chain: drive rejection_sample with num_draft_tokens=[K]*B,
    per-position target rows p_pos[0..K-1] and deterministic drafts draft_pos.
    Returns out [B, K+1] (PLACEHOLDER where not reached / after first reject)."""
    K = len(p_pos)
    vocab = p_pos[0].numel()
    torch.manual_seed(seed)
    logp = torch.stack([torch.log(p.clamp_min(0)).to(torch.float32) for p in p_pos])  # [K,vocab]
    target_logits = logp.unsqueeze(0).expand(B, K, vocab).reshape(B * K, vocab).contiguous().to(DEVICE)
    draft_token_ids = torch.tensor(draft_pos, dtype=torch.int32).repeat(B).to(DEVICE)  # [B*K]
    num_draft_tokens = [K] * B
    cu = torch.arange(K, K * B + 1, K, dtype=torch.int32, device=DEVICE)  # [K,2K,...,BK]
    bonus = int(torch.argmax(p_pos[-1]).item())
    bonus_token_ids = torch.full((B, 1), bonus, dtype=torch.int32, device=DEVICE)
    out = rejection_sample(
        draft_token_ids, num_draft_tokens, K, cu, None,
        target_logits, bonus_token_ids, _sm(B),
    )
    return out.to(torch.int64).cpu()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float(torch.abs(a - b).sum())


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-12
    mask = p > 0
    pp = p[mask]
    qq = q[mask].clamp_min(eps)
    return float((pp * (pp / qq).log()).sum())


def iid_floor(p: torch.Tensor, N: int, seed: int) -> float:
    """Monte-Carlo noise floor: TV of an i.i.d. multinomial draw of size N from p
    vs p itself (the irreducible sampling error at sample size N)."""
    if N <= 0:
        return float("nan")
    g = torch.Generator().manual_seed(seed)
    iid = torch.multinomial(p, N, replacement=True, generator=g)
    ic = torch.bincount(iid, minlength=p.numel()).to(torch.float64)
    return tv(ic / ic.sum(), p)


def iid_floor_band(p: torch.Tensor, N: int, R: int, seed: int) -> dict:
    """Sampling BAND of the noise floor: the distribution of TV(empirical_N, p)
    under the null hypothesis "output ~ p EXACTLY", estimated from R independent
    i.i.d. multinomial redraws of size N from p.

    Why a band, not a single draw: TV(empirical, p) is a noisy, right-skewed
    statistic; a single iid draw is a poor reference, and `max_d TV_deployed - (one
    floor draw)` is upward-biased (max of several draws minus one). Under the
    null, TV_deployed at every acceptance level is just ANOTHER draw from THIS
    band -- so "at the floor" is unambiguous: TV_deployed within [mu +- k*sd] /
    below the high quantile `hi`. Decisiveness: a REAL deviation delta is
    ~constant while sd ~ 1/sqrt(N) -> z=(TV-mu)/sd grows like sqrt(N) (hundreds at
    N>=50k); pure sampling noise stays z~O(1). So the band cleanly separates a
    genuine distribution shift from finite-sample noise."""
    if N <= 0 or R <= 0:
        return {"mu": float("nan"), "sd": float("nan"), "hi": float("nan"),
                "max": float("nan"), "R": 0, "N": N}
    g = torch.Generator().manual_seed(seed)
    vocab = p.numel()
    tvs = torch.empty(R, dtype=torch.float64)
    for r in range(R):
        iid = torch.multinomial(p, N, replacement=True, generator=g)
        ic = torch.bincount(iid, minlength=vocab).to(torch.float64)
        tvs[r] = tv(ic / ic.sum(), p)
    mu = float(tvs.mean())
    sd = float(tvs.std(unbiased=True)) if R > 1 else 0.0
    # distribution-free upper edge of the floor band (handles TV right-skew);
    # also guard with mu + K_SIGMA*sd so the bar is robust at small R.
    q999 = float(tvs.quantile(0.999)) if R >= 50 else float(tvs.max())
    hi = max(q999, mu + K_SIGMA * sd)
    return {"mu": mu, "sd": sd, "hi": hi, "max": float(tvs.max()), "R": R, "N": N}


def g_test(counts: torch.Tensor, p: torch.Tensor, N: int) -> dict:
    """Likelihood-ratio goodness-of-fit (G-test) of observed `counts` against the
    expected N*p, over the support of p. M-INDEPENDENT confirmation that the
    deployed output IS p: under the null, G ~ chi-square(support-1). p-value via
    the Wilson-Hilferty normal approximation (no scipy dependency). One-sided
    (large G == bad fit). Under exact preservation, p-values are ~Uniform(0,1)."""
    mask = p > 0
    obs = counts[mask].to(torch.float64)
    exp = (float(N) * p[mask]).to(torch.float64)
    nz = obs > 0
    G = 2.0 * float((obs[nz] * (obs[nz] / exp[nz]).log()).sum())
    k = int(mask.sum().item()) - 1  # degrees of freedom
    if k <= 0:
        return {"G": G, "dof": k, "wh_z": 0.0, "pvalue": 1.0}
    x = (G / k) ** (1.0 / 3.0)
    z = (x - (1.0 - 2.0 / (9 * k))) / math.sqrt(2.0 / (9 * k))
    pvalue = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {"G": G, "dof": k, "wh_z": z, "pvalue": pvalue}


# --------------------------------------------------------------------------- #
# Leg 1 — single-position acceptance sweep
# --------------------------------------------------------------------------- #
def _norm(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64).clamp_min(0)
    return x / x.sum()


def acceptance_sweep_dists() -> list[tuple[str, torch.Tensor]]:
    """FIXED target dists whose support probabilities densely cover the
    acceptance range of interest (~0.5 down to ~0.003), so that drafting each
    support token sweeps realized acceptance at a FIXED noise floor. Multiple
    shapes stress different residual-mass regimes of the recovered kernel."""
    cases: list[tuple[str, torch.Tensor]] = []

    # Explicit private-bracket ladder: tokens sit AT public 0.387 and step down
    # through the private breach band (0.1, 0.05, 0.02, 0.01, 0.005, 0.003).
    bracket = torch.tensor([0.387, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.003])
    bracket = torch.cat([bracket, torch.full((1,), float(1.0 - bracket.sum()))])  # filler top token
    cases.append(("private_bracket", _norm(bracket)))

    # Geometric ladder over v=48: dense log-spaced acceptance from ~0.5 down.
    v = 48
    geom = 0.5 ** torch.arange(v, dtype=torch.float64)
    cases.append(("geom_ladder_v48", _norm(geom)))

    # Realistic answer-letter regimes (mirror #505 synthetic suite).
    cases.append(("confident_mcq", _norm(torch.tensor([0.97, 0.02, 0.006, 0.004]))))
    cases.append(("graded_4way", _norm(torch.tensor([0.45, 0.30, 0.15, 0.10]))))
    cases.append(("two_way_60_40", _norm(torch.tensor([0.6, 0.4]))))

    # Higher-entropy power-law tail over v=64 (numeric-token / hard-OOD-like).
    v2 = 64
    pl = 1.0 / torch.arange(1, v2 + 1, dtype=torch.float64) ** 1.3
    cases.append(("powerlaw_v64", _norm(pl)))

    # Broad v=256 high-entropy stress.
    broad = torch.softmax(
        0.5 * torch.randn(256, generator=torch.Generator().manual_seed(0)), dim=0
    ).double()
    cases.append(("broad_v256", _norm(broad)))

    return cases


def run_acceptance_sweep(
    p: torch.Tensor, label: str, M: int, seed: int, R: int = 160
) -> dict:
    """Sweep the draft token over the support of FIXED p. Each draft token d
    gives realized acceptance ~ p(d); measure TV(deployed, p), KL, a z-score vs
    the iid noise-floor BAND, and a G-test p-value. The noise floor band is held
    FIXED (same p, same N=M emit every trial), so TV_deployed sitting inside the
    band across the WHOLE acceptance range IS acceptance-rate-invariance."""
    p = _norm(p)
    vocab = p.numel()
    band = iid_floor_band(p, M, R, seed + 7)
    support = torch.nonzero(p > 0).flatten().tolist()
    # Sweep all support tokens when small; else subsample by probability deciles
    # plus the explicit private-band tokens so the breach regime is covered.
    if len(support) > 24:
        order = sorted(support, key=lambda j: float(p[j]), reverse=True)
        idx = sorted(set(
            [order[int(round(t))] for t in torch.linspace(0, len(order) - 1, 24).tolist()]
            + [j for j in support if float(p[j]) <= PRIVATE_BREACH_HI][:8]
        ), key=lambda j: float(p[j]), reverse=True)
        support = idx

    rows = []
    for d in support:
        counts, acc = deployed_first_token_hist(p.to(torch.float32), int(d), M, seed)
        phat = counts / counts.sum()
        t = tv(phat, p)
        z = (t - band["mu"]) / band["sd"] if band["sd"] > 0 else 0.0
        g = g_test(counts, p, M)
        rows.append({
            "draft": int(d),
            "p_draft": float(p[d]),
            "realized_accept": acc,
            "tv_deployed_vs_p": t,
            "signed_excess_over_mu": t - band["mu"],
            "z_over_floor": z,
            "within_band": bool(t <= band["hi"]),
            "kl_p_given_deployed": kl(p, phat),
            "gtest_pvalue": g["pvalue"],
            "gtest_dof": g["dof"],
        })
    accs = [r["realized_accept"] for r in rows]
    tvs = [r["tv_deployed_vs_p"] for r in rows]
    signed = [r["signed_excess_over_mu"] for r in rows]
    zs = [r["z_over_floor"] for r in rows]
    pvals = [r["gtest_pvalue"] for r in rows]
    # slope of TV vs realized acceptance: ~0 == TV does not depend on acceptance.
    slope = _slope(accs, tvs) if len(rows) > 1 else 0.0
    # private breach band for the headline private-exposure number
    breach = [r for r in rows if PRIVATE_BREACH_LO <= r["realized_accept"] <= PRIVATE_BREACH_HI]
    return {
        "label": label, "vocab": vocab, "M": M, "R": R,
        "p_max": float(p.max()), "support": len(torch.nonzero(p > 0).flatten()),
        "entropy_nats": float(-(p[p > 0] * p[p > 0].log()).sum()),
        "floor_mu": band["mu"], "floor_sd": band["sd"], "floor_hi": band["hi"],
        "accept_min": min(accs), "accept_max": max(accs),
        "max_tv_deployed_vs_p": max(tvs),
        "max_z_over_floor": max(zs),
        "mean_signed_excess_over_mu": sum(signed) / len(signed),
        "tv_acceptance_slope": slope,
        "all_within_band": all(r["within_band"] for r in rows),
        "n_band_exceed": sum(1 for r in rows if not r["within_band"]),
        "min_gtest_pvalue": min(pvals),
        "bonferroni_min_pvalue": min(min(pvals) * len(rows), 1.0),
        "frac_gtest_p_gt_05": sum(1 for pv in pvals if pv > 0.05) / len(pvals),
        "n_points": len(rows),
        "n_breach_points": len(breach),
        "max_tv_in_breach_band": max((r["tv_deployed_vs_p"] for r in breach), default=float("nan")),
        "max_exposure_in_breach": max((max(0.0, r["tv_deployed_vs_p"] - band["hi"]) for r in breach), default=0.0),
        "rows": rows,
    }


def run_real_acceptance_sweep(P: torch.Tensor, meta: list, M: int, seed: int, R: int = 160) -> list[dict]:
    """Real #497 reasoning answer dists: for each, sweep draft over support."""
    out = []
    for i in range(len(P)):
        r = run_acceptance_sweep(P[i].double(), meta[i].get("id", f"row{i}"), M, seed + i, R)
        r["meta"] = meta[i]
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Leg 2 — K=7 multi-position chaining
# --------------------------------------------------------------------------- #
def onehot(vocab: int, j: int) -> torch.Tensor:
    e = torch.zeros(vocab, dtype=torch.float64)
    e[j] = 1.0
    return e


def k7_depth_probe(p_test: torch.Tensor, B: int, seed: int, K: int = 7, R: int = 160) -> dict:
    """Depth-isolated probe: for each depth i in 0..K-1, build a chain whose
    positions 0..i-1 ALWAYS accept (one-hot target == draft) so position i is
    reached on EVERY trial (full statistics at every depth), with the test dist
    p_test at position i. Position i emits ~ p_test independent of depth i => no
    accumulation. Same iid floor BAND (N=B) at every depth, so TV staying in band
    AND a flat TV-vs-depth slope == zero error accumulation across the 7 spine
    positions. Depth 0 is the plain single-position case."""
    p_test = _norm(p_test)
    vocab = p_test.numel()
    d_test = int(torch.argmax(p_test).item())
    band = iid_floor_band(p_test, B, R, seed + 99)
    per_depth = []
    for i in range(0, K):
        # prefix one-hot always-accept on token 0 (distinct from arbitrary), test at i,
        # suffix one-hot on token 0 (irrelevant to o_i).
        p_pos, d_pos = [], []
        for pos in range(K):
            if pos == i:
                p_pos.append(p_test); d_pos.append(d_test)
            else:
                p_pos.append(onehot(vocab, 0)); d_pos.append(0)
        out = deployed_k7_chain(p_pos, d_pos, B, seed + i)
        emit = out[:, i]
        reached = int((emit != PLACEHOLDER_TOKEN_ID).sum())
        valid = emit[emit != PLACEHOLDER_TOKEN_ID]
        counts = torch.bincount(valid, minlength=vocab).to(torch.float64)
        phat = counts / counts.sum()
        t = tv(phat, p_test)
        g = g_test(counts, p_test, reached)
        per_depth.append({
            "depth": i, "reached": reached, "reach_frac": reached / B,
            "tv_deployed_vs_p": t,
            "z_over_floor": (t - band["mu"]) / band["sd"] if band["sd"] > 0 else 0.0,
            "within_band": bool(t <= band["hi"]),
            "kl_p_given_deployed": kl(p_test, phat),
            "gtest_pvalue": g["pvalue"],
        })
    tvs = [d["tv_deployed_vs_p"] for d in per_depth]
    # "positions 2..7" == 1-indexed positions 2..7 == 0-indexed depths 1..6
    chained = [d for d in per_depth if d["depth"] >= 1]
    return {
        "mode": "depth_isolated", "K": K, "vocab": vocab, "B": B, "R": R,
        "p_test_max": float(p_test.max()),
        "floor_mu": band["mu"], "floor_sd": band["sd"], "floor_hi": band["hi"],
        "max_tv_over_positions": max(tvs),
        "max_tv_over_positions_2to7": max(d["tv_deployed_vs_p"] for d in chained),
        "max_z_over_floor": max(d["z_over_floor"] for d in per_depth),
        "all_within_band": all(d["within_band"] for d in per_depth),
        "tv_slope_per_depth": _slope([d["depth"] for d in per_depth], tvs),
        "min_gtest_pvalue": min(d["gtest_pvalue"] for d in per_depth),
        "per_depth": per_depth,
    }


def k7_natural_chain(p_pos: list[torch.Tensor], B: int, seed: int, K: int = 7, R: int = 160) -> dict:
    """Natural chain: distinct realistic per-position targets, greedy drafts
    (d_i = argmax p_i => per-position acceptance = p_i_max). Measure the emitted
    token at each position CONDITIONED on the position being reached, vs p_i, at
    that position's effective sample size (its own floor band)."""
    p_pos = [_norm(p) for p in p_pos]
    d_pos = [int(torch.argmax(p).item()) for p in p_pos]
    out = deployed_k7_chain(p_pos, d_pos, B, seed)
    vocab = p_pos[0].numel()
    per_pos = []
    for i in range(K):
        emit = out[:, i]
        reached_mask = emit != PLACEHOLDER_TOKEN_ID
        reached = int(reached_mask.sum())
        valid = emit[reached_mask]
        counts = torch.bincount(valid, minlength=vocab).to(torch.float64)
        phat = counts / counts.sum() if reached > 0 else counts
        band_i = iid_floor_band(p_pos[i], reached, min(R, 80), seed + 500 + i)
        t = tv(phat, p_pos[i]) if reached > 0 else float("nan")
        per_pos.append({
            "position": i, "reached": reached, "reach_frac": reached / B,
            "p_max": float(p_pos[i].max()), "accept_at_pos": float(p_pos[i][d_pos[i]]),
            "tv_deployed_vs_p": t,
            "floor_mu": band_i["mu"], "floor_hi": band_i["hi"],
            "z_over_floor": ((t - band_i["mu"]) / band_i["sd"]) if (reached > 0 and band_i["sd"] > 0) else 0.0,
            "within_band": bool(t <= band_i["hi"]) if reached > 0 else True,
        })
    chained = [d for d in per_pos if d["position"] >= 1 and d["reached"] > 0]
    return {
        "mode": "natural_chain", "K": K, "vocab": vocab, "B": B, "R": R,
        "max_tv_over_positions": max(d["tv_deployed_vs_p"] for d in per_pos if d["reached"] > 0),
        "max_tv_over_positions_2to7": max(d["tv_deployed_vs_p"] for d in chained),
        "max_z_over_floor": max(abs(d["z_over_floor"]) for d in chained),
        "all_within_band": all(d["within_band"] for d in chained),
        "per_position": per_pos,
    }


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Scale-free association between two pooled series. Used for the literal
    acceptance-invariance test: r(realized_accept, z_over_floor) ~ 0 means the
    floor-relative TV residual does NOT depend on the acceptance rate. (A per-case
    slope of TV vs accept is ill-conditioned when a case's acceptance range is
    narrow; the pooled, floor-normalized correlation is robust.)"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sxx * syy)
    return sxy / den if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def self_tests() -> dict:
    """Cheap correctness gates (reuse + extend #505). All must pass."""
    res = {}
    M = 60_000

    # ST1: one-hot target p=delta_j -> output == j for ANY draft (accept j, or
    # reject!=j then recover from p|{!=draft} which still has all mass on j).
    p = onehot(8, 3)
    c0, _ = deployed_first_token_hist(p.to(torch.float32), 3, 20_000, 1)  # draft = j
    c1, _ = deployed_first_token_hist(p.to(torch.float32), 5, 20_000, 2)  # draft != j
    res["st1_onehot_draft_eq_j_tv"] = tv(c0 / c0.sum(), p)
    res["st1_onehot_draft_ne_j_tv"] = tv(c1 / c1.sum(), p)
    res["st1_pass"] = (tv(c0 / c0.sum(), p) == 0.0 and tv(c1 / c1.sum(), p) == 0.0)

    # ST2: realized accept rate ~ p(draft) for several draft tokens (the min(1,p)
    # signature) -> acceptance IS p(x_d), the thing draft quality moves.
    pg = _norm(torch.tensor([0.45, 0.30, 0.15, 0.10]))
    err = []
    for d in range(4):
        _, acc = deployed_first_token_hist(pg.to(torch.float32), d, M, 10 + d)
        err.append(abs(acc - float(pg[d])))
    res["st2_max_accept_rate_err"] = max(err)
    res["st2_pass"] = max(err) < 0.01

    # ST3: deployed TV inside the iid floor BAND across the acceptance sweep, with
    # no systematic bias (mean signed excess ~ 0) -> preservation at the floor.
    sw = run_acceptance_sweep(pg, "st3", M, 33, R=120)
    res["st3_max_z_over_floor"] = sw["max_z_over_floor"]
    res["st3_mean_signed_excess"] = sw["mean_signed_excess_over_mu"]
    res["st3_pass"] = bool(sw["all_within_band"] and abs(sw["mean_signed_excess_over_mu"]) < 0.005)

    # ST4: K=7 reached-fraction matches geometric expectation prod(accept).
    pp = [_norm(torch.tensor([0.7, 0.2, 0.1])) for _ in range(7)]
    nat = k7_natural_chain(pp, 80_000, 7, R=60)
    # position i reached iff positions 0..i-1 all accepted; accept = p_max = 0.7
    # each, so reach_frac[i] = 0.7**i (i=0 always reached => 0.7**0 = 1.0).
    exp_reach = [0.7 ** i for i in range(7)]
    reach_err = max(abs(nat["per_position"][i]["reach_frac"] - exp_reach[i]) for i in range(7))
    res["st4_max_reach_frac_err"] = reach_err
    res["st4_pass"] = reach_err < 0.02

    # ST5: K=7 depth-isolated -> per-position TV in band at every depth, flat in
    # depth (no error accumulation across the spine).
    dp = k7_depth_probe(pg, 80_000, 7, R=120)
    res["st5_max_z_over_floor"] = dp["max_z_over_floor"]
    res["st5_tv_slope_per_depth"] = dp["tv_slope_per_depth"]
    res["st5_pass"] = bool(dp["all_within_band"] and abs(dp["tv_slope_per_depth"]) < 0.005)

    # ST6: NaN-clean across all the above numbers.
    flat = []
    for v in res.values():
        if isinstance(v, float):
            flat.append(v)
    res["st6_nan_clean"] = all(not math.isnan(x) for x in flat)
    res["st6_pass"] = res["st6_nan_clean"]

    gates = [k for k in res if k.startswith("st") and k.endswith("_pass")]
    res["all_pass"] = all(res[k] for k in gates)
    res["n_pass"] = sum(1 for k in gates if res[k])
    res["n_tests"] = len(gates)
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _wmean(pairs: list[tuple[float, int]]) -> float:
    """Weighted mean of (value, weight) pairs; nan-safe (skips nan values)."""
    num = sum(v * w for v, w in pairs if not math.isnan(v))
    den = sum(w for v, w in pairs if not math.isnan(v))
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=300_000, help="trials per acceptance point (synthetic)")
    # real dists are vocab=16384; [M,vocab] fp32 x3 buffers must fit GPU -> cap ~50k.
    ap.add_argument("--M-real", type=int, default=50_000, help="trials per acceptance point (real)")
    ap.add_argument("--B", type=int, default=300_000, help="trials for K=7 chains")
    ap.add_argument("--R", type=int, default=160, help="iid redraws for the noise-floor band")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--real-logits", type=str, default=None,
                    help="optional .pt {'p':[N,vocab],'meta':[...]} real #497 dists")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        st = self_tests()
        print(json.dumps(st, indent=2))
        raise SystemExit(0 if st["all_pass"] else 1)

    # invariance thresholds: within-band (mu+K_SIGMA*sd) at every acceptance, no
    # systematic bias in the floor-relative residual, and no association between
    # the floor-relative residual and the acceptance rate. All loose vs the iid
    # floor; the band membership is the hard gate.
    EXC_TOL = 0.001    # |mean signed excess over mu| -- low-variance bias guard
    CORR_TOL = 0.10    # |corr(accept, z_over_floor)| -- literal acceptance-invariance

    out: dict = {"self_test": self_tests()}
    assert out["self_test"]["all_pass"], "self-tests failed; aborting"

    # ---- Leg 1: synthetic controlled acceptance sweep ----
    syn = [run_acceptance_sweep(p, label, args.M, args.seed, args.R)
           for label, p in acceptance_sweep_dists()]
    out["leg1_synthetic"] = syn

    # ---- Leg 1: real #497 reasoning natural sweep ----
    real = None
    if args.real_logits:
        blob = torch.load(args.real_logits)
        P = blob["p"]
        meta = blob.get("meta", [{}] * len(P))
        real = run_real_acceptance_sweep(P, meta, args.M_real, args.seed, args.R)
        out["leg1_real"] = real

    # ---- Leg 2: K=7 chaining ----
    p_test = _norm(torch.tensor([0.45, 0.30, 0.15, 0.10]))  # graded answer-letter regime
    depth = k7_depth_probe(p_test, args.B, args.seed, R=args.R)
    # natural chain: distinct realistic per-position targets
    nat_targets = [
        _norm(torch.tensor([0.6, 0.25, 0.10, 0.05])),
        _norm(torch.tensor([0.5, 0.3, 0.15, 0.05])),
        _norm(torch.tensor([0.7, 0.2, 0.07, 0.03])),
        _norm(torch.tensor([0.45, 0.30, 0.15, 0.10])),
        _norm(torch.tensor([0.55, 0.25, 0.12, 0.08])),
        _norm(torch.tensor([0.65, 0.20, 0.10, 0.05])),
        _norm(torch.tensor([0.5, 0.28, 0.14, 0.08])),
    ]
    natural = k7_natural_chain(nat_targets, args.B, args.seed + 1, R=args.R)
    out["leg2_k7_depth"] = depth
    out["leg2_k7_natural"] = natural

    # ---- Aggregate headline outputs ----
    all_cases = list(syn) + (list(real) if real else [])

    def _agg(cases: list[dict]) -> dict:
        if not cases:
            return {"max_tv": float("nan"), "max_z": float("nan"),
                    "mean_signed": float("nan"), "max_abs_slope": float("nan"),
                    "all_within_band": True, "n_band_exceed": 0, "exposure": 0.0}
        return {
            "max_tv": max(c["max_tv_deployed_vs_p"] for c in cases),
            "max_z": max(c["max_z_over_floor"] for c in cases),
            "mean_signed": _wmean([(c["mean_signed_excess_over_mu"], c["n_points"]) for c in cases]),
            "max_abs_slope": max(abs(c["tv_acceptance_slope"]) for c in cases),
            "all_within_band": all(c["all_within_band"] for c in cases),
            "n_band_exceed": sum(c["n_band_exceed"] for c in cases),
            "exposure": max(c["max_exposure_in_breach"] for c in cases),
        }

    syn_a, real_a, all_a = _agg(syn), _agg(real or []), _agg(all_cases)
    mean_floor = sum(c["floor_mu"] for c in syn) / len(syn)

    # pooled, floor-normalized acceptance-invariance test across EVERY swept point
    pooled_accept = [r["realized_accept"] for c in all_cases for r in c["rows"]]
    pooled_z = [r["z_over_floor"] for c in all_cases for r in c["rows"]]
    pooled_p = [r["gtest_pvalue"] for c in all_cases for r in c["rows"]]
    accept_z_corr = _pearson(pooled_accept, pooled_z)
    n_pooled = len(pooled_accept)
    # family-wise G-test: global Bonferroni over ALL tests, + the uniform-p signature.
    gtest_bonferroni_global = min(min(pooled_p) * len(pooled_p), 1.0) if pooled_p else float("nan")
    gtest_frac_p_gt_05 = (sum(1 for pv in pooled_p if pv > 0.05) / len(pooled_p)) if pooled_p else float("nan")

    k7_max_tv = max(depth["max_tv_over_positions"], natural["max_tv_over_positions"])
    k7_max_tv_2to7 = max(depth["max_tv_over_positions_2to7"], natural["max_tv_over_positions_2to7"])

    # Acceptance-rate-invariant preservation: at EVERY swept acceptance the
    # deployed TV sits inside the iid floor band (max_z <= K_SIGMA <=> all_within_
    # band), with no systematic floor-relative bias, and the floor-relative
    # residual is UNcorrelated with the acceptance rate; corroborated by a
    # non-significant G-test (Bonferroni) -> output distribution == p everywhere.
    quality_acceptance_invariant = bool(
        all_a["all_within_band"]
        and abs(all_a["mean_signed"]) < EXC_TOL
        and abs(accept_z_corr) < CORR_TOL
    )
    # K=7: no error accumulation -- every spine position in band and TV flat in depth.
    k7_no_accumulation = bool(
        depth["all_within_band"] and natural["all_within_band"]
        and abs(depth["tv_slope_per_depth"]) < EXC_TOL
    )
    private_quality_exposure = all_a["exposure"]  # ~0: TV never exceeds the floor band

    verdict_ok = (quality_acceptance_invariant and k7_no_accumulation
                  and private_quality_exposure < EXC_TOL)
    out["summary"] = {
        "vllm": "0.22.1rc1.dev307+g3e8afdf78",
        "ship": "surgical-357 (PR#499)", "spec": "mtp", "K": 7,
        "public_accept_anchor": PUBLIC_ACCEPT_ANCHOR,
        "private_breach_band": [PRIVATE_BREACH_LO, PRIVATE_BREACH_HI],
        "k_sigma_band": K_SIGMA, "R_redraws": args.R,
        "mean_iid_noise_floor": mean_floor,
        # --- KEY OUTPUTS (card) ---
        "max_tv_across_acceptance_sweep": all_a["max_tv"],
        "quality_acceptance_invariant": quality_acceptance_invariant,
        "private_quality_exposure": private_quality_exposure,
        "max_tv_over_k7_positions": k7_max_tv,
        "k7_no_accumulation": k7_no_accumulation,
        # --- supporting statistics ---
        "max_z_over_floor_across_sweep": all_a["max_z"],
        "mean_signed_excess_over_mu": all_a["mean_signed"],
        "accept_z_correlation": accept_z_corr,
        "n_acceptance_points_pooled": n_pooled,
        "max_abs_tv_acceptance_slope": all_a["max_abs_slope"],
        "n_acceptance_points_in_band": "all" if all_a["all_within_band"] else f"{all_a['n_band_exceed']} exceed",
        "n_band_exceed": all_a["n_band_exceed"],
        "gtest_bonferroni_global_pvalue": gtest_bonferroni_global,
        "gtest_frac_p_gt_05": gtest_frac_p_gt_05,
        "syn_max_tv_across_acceptance_sweep": syn_a["max_tv"],
        "syn_max_z_over_floor": syn_a["max_z"],
        "syn_mean_signed_excess": syn_a["mean_signed"],
        "real_max_tv_across_acceptance_sweep": real_a["max_tv"],
        "real_max_z_over_floor": real_a["max_z"],
        "real_mean_signed_excess": real_a["mean_signed"],
        "max_tv_over_k7_positions_2to7": k7_max_tv_2to7,
        "k7_depth_max_z_over_floor": depth["max_z_over_floor"],
        "k7_depth_tv_slope": depth["tv_slope_per_depth"],
        "exc_tol": EXC_TOL, "corr_tol": CORR_TOL,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0.0),
        "verdict": (
            "PURE SPEED RISK / ZERO QUALITY EXPOSURE"
            if verdict_ok else "QUALITY EXPOSURE DETECTED"
        ),
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
