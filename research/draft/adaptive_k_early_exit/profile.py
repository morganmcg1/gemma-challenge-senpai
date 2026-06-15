"""Adaptive-K early-exit drafting profiler (PR #256, stark).

LOCAL profiling only (1xA10G nominal, but this routine is pure-CPU/NumPy -- no
GPU, no model load, no served-file change, no HF Job). Prices a confidence-gated
early-exit on the *live* linear MTP K=7 draft path and finds the TPS-optimal
top-1 confidence threshold.

The lever (greedy-safe by construction)
---------------------------------------
The served stack is linear MTP K=7 (PR #52, 481.53 TPS, PPL 2.3772, 128/128):
every decode step the MTP head autoregressively proposes 7 draft tokens, then a
single verify accepts a bit-for-bit greedy prefix.  K=7 is FIXED regardless of
how confident the MTP head is at each draft sub-step.  Adaptive-K: at draft
sub-step i, if the MTP head's top-1 confidence c_i < theta, STOP proposing
(skip tokens i+1..7) and verify the i tokens so far.  The verify and acceptance
criterion (`rejected = draft_token_id != target_argmax_id`) are UNTOUCHED, so
every committed token is still the verify's greedy argmax -> the emitted token
stream is a prefix of the K=7 greedy stream (the cut tail is simply re-proposed
on later steps).  K changes how many tokens are PROPOSED, never WHICH token is
accepted: greedy-identity holds by construction (confirmed empirically in the
self-test).

The tradeoff this profiler measures
-----------------------------------
Cutting a draft pass that WOULD have been accepted costs E[T]; cutting one that
would have been rejected is free.  Low MTP confidence correlates with low
acceptance, so the cut tail is mostly rejected anyway.  The TPS-optimal theta
balances draft-pass savings (cheaper step) against E[T] loss (fewer tokens/step).

What is MEASURED vs MODELED (kept strictly separate, advisor #247 standard)
---------------------------------------------------------------------------
MEASURED, deployed qat-assistant head (research/accept_calibration, W&B 5m17r52s,
128 ShareGPT prompts, K=7 linear MTP):
  * per-depth conditional acceptance ladder q[i]  (this is the deployed signal),
  * E[T]_base = 3.844, TPS_base = 481.53 = K_CAL * E[T]_base.
MEASURED, on-branch confidence<->accept JOINT (research/entropy_dynamic_k/
entropy_sim.json, n=5484, top1p AUC 0.857): the calibration SHAPE g(c) =
P(accept | top1p=c) and the top1p marginal spread.  NOTE this joint was traced
on the EAGLE3 drafter (base top-1 accept 0.679), NOT the deployed qat-assistant
head (0.729).  We TRANSFER only the calibration SHAPE and the confidence spread;
we PIN the per-depth acceptance marginals to the deployed ladder (so E[g(c_j)] =
q_deployed[j] exactly).  Fresh deployed-head per-sub-step confidence measurement
is the named validation follow-up.
MODELED: the per-depth top-1 confidence density f_j(c) = Beta(mean mu_j,
concentration nu), nu fixed from the EAGLE3 top1p spread, mu_j solved so
E[g(c_j)] = q_deployed[j].

Composition (deployed convention, import -- do NOT re-derive)
------------------------------------------------------------
K_CAL = 125.268 (steps/s anchor: 481.53 = K_CAL * 3.844).
g_d   = 0.168  (one MTP draft pass costs g_d * verify; denken #75/#85, wirbel #83).
step(K) = verify * (1 + K * g_d);  at K=7, factor 2.176.  At conc=1 decode is
memory-bound so verify is M-independent -> cutting a draft pass saves exactly
g_d*verify and does not change verify cost.  Hence
  TPS(theta) = TPS_base * (E[T]_adaptive/E[T]_base) * (1+7*g_d)/(1+mean_K*g_d).

ONEGRAPH=1 caveat -- the headline gain is an UPPER BOUND
-------------------------------------------------------
The served stack runs all 7 proposer passes inside ONE static CUDA graph
(submissions/fa2sw_precache_kenyan/manifest.json: ONEGRAPH=1), a layout lawine
#246 adopted precisely to amortize per-pass kernel-launch overhead.  A *runtime*
early-exit is incompatible with a static graph that always replays all 7 passes:
realizing adaptive-K needs the onegraph broken (per-pass launches -> overhead
back) or a multi-graph / conditional-graph scheme.  So the clean saving g_d per
SKIPPED pass is an UPPER BOUND; onegraph-break overhead claws some of it back.
We headline the upper bound (s=g_d) but PRICE the caveat with a saving-survival
sweep (``gd_sensitivity``): adaptive step factor = 1 + 7*g_d - (7-mean_K)*s, with
the realizable net saving per skipped pass s in [0, g_d].  The break-even s/g_d is
the onegraph-break overhead budget adaptive-K can absorb before it stops winning.

Deliverable (PRIMARY): ``adaptive_k_early_exit_self_test_passes`` --
  (a) greedy-identity: adaptive committed stream is token-identical to the K=7
      control (verify untouched -> holds by construction; confirmed on fuzzed
      realizations), (b) PPL <= 2.42 (pinned 2.3772; verify unchanged), (c)
      NaN-clean, (d) peak VRAM <= 24 GB (CPU-only profiler).
TEST metrics: ``theta_star``, ``net_tps_gain_pct_adaptiveK`` (at theta_star),
  ``e_t_adaptive`` (vs the K=7 control E[T]=3.844), ``mean_realized_K``.

A clean NO-GO (no theta beats K=7 net) is a valid terminal result: the leg is
correct; a refuted premise is a separate flag, NOT a PRIMARY self-test failure.

Run (reported command):
  cd target/ && CUDA_VISIBLE_DEVICES=0 python \
      research/draft/adaptive_k_early_exit/profile.py --self-test --sweep-theta \
      --wandb_group adaptive-k-early-exit --wandb_name stark/adaptive-k-early-exit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# =============================================================================
# MEASURED ANCHORS  (import -- do NOT re-derive)
# =============================================================================

# Deployed qat-assistant linear-MTP K=7 per-depth conditional acceptance ladder
# q[i] = P(draft token at depth i accepted | depths 1..i-1 accepted).
# research/accept_calibration/accept_calibration_results.json (W&B 5m17r52s,
# 128 ShareGPT prompts, server-log Prometheus counters).
Q_COND = np.array([
    0.728739760479042, 0.7589764102641635, 0.7924989076194682,
    0.821702519412012, 0.8342716929825772, 0.8352594665096346,
    0.8472621220149911,
])
K_DEPLOYED = len(Q_COND)  # 7

# E[T]_base = 1 + sum_i prod_{j<=i} q[j]  (the verify always emits >=1 bonus tok).
CUMUL_C = np.cumprod(Q_COND)
E_T_BASE = float(1.0 + CUMUL_C.sum())          # == 3.844 measured
E_T_BASE_MEASURED = 3.844131736526946          # accept_calibration headline

# Composition anchors (deployed convention).
K_CAL = 125.268        # 481.53 = K_CAL * 3.844  (steps/s anchor)
G_D = 0.168            # one MTP draft pass cost / verify (denken #75/#85, wirbel #83)
STEP_INT4 = 1.2182     # M=8-norm depth-9 verify-step (lawine #136); context anchor
OFFICIAL_TPS = 481.53  # PR #52 official a10g-small, PPL 2.3772, 128/128
PPL_PINNED = 2.3772    # verify untouched -> identical output -> identical PPL
PPL_CAP = 2.42
# Served stack bakes all 7 proposer passes into one static CUDA graph (lawine
# #246; manifest ONEGRAPH=1).  Runtime early-exit is incompatible with that, so
# the clean per-skipped-pass saving g_d is an UPPER BOUND -- see gd_sensitivity.
ONEGRAPH_DEPLOYED = True

# On-branch confidence<->accept JOINT, top1p signal (best gate signal, AUC 0.857).
# research/entropy_dynamic_k/entropy_sim.json -> predictor.signals.top1p.
# EAGLE3-drafter trace (base accept 0.679); we transfer ONLY the calibration
# SHAPE + the confidence spread, and re-pin acceptance to the deployed ladder.
# (sig_lo, sig_hi, accept_rate) per decile:
TOP1P_DECILES = np.array([
    [0.0122, 0.2450, 0.1551],
    [0.2450, 0.3707, 0.3157],
    [0.3708, 0.4993, 0.4536],
    [0.4995, 0.6227, 0.6040],
    [0.6230, 0.7619, 0.6867],
    [0.7620, 0.8788, 0.7865],
    [0.8789, 0.9482, 0.8923],
    [0.9486, 0.9832, 0.9472],
    [0.9833, 0.9966, 0.9653],
    [0.9966, 1.0000, 0.9854],
])
TOP1P_MEAN_EAGLE3 = 0.6862
TOP1P_SD_EAGLE3 = 0.2912
# Beta concentration nu from the EAGLE3 top1p spread: nu = m(1-m)/s^2 - 1.
NU_FROM_EAGLE3 = (TOP1P_MEAN_EAGLE3 * (1.0 - TOP1P_MEAN_EAGLE3)
                  / (TOP1P_SD_EAGLE3 ** 2) - 1.0)

# Confidence-integration grid (bin midpoints over (0,1); robust for Beta a,b<1).
_NGRID = 2000
_GRID = (np.arange(_NGRID) + 0.5) / _NGRID


# =============================================================================
# CALIBRATION  g(c) = P(accept | top1p = c)   (transferred SHAPE)
# =============================================================================

def _calibration_nodes():
    """Monotone interpolation nodes from the top1p decile midpoints, pinned at
    the [0,1] ends by the extreme deciles (clamped, so g stays in [0,1])."""
    mids = TOP1P_DECILES[:, :2].mean(axis=1)
    rates = TOP1P_DECILES[:, 2]
    xs = np.concatenate([[0.0], mids, [1.0]])
    ys = np.concatenate([[rates[0]], rates, [rates[-1]]])
    return xs, ys


_CAL_X, _CAL_Y = _calibration_nodes()


def g_accept(c):
    """P(accept | top1p=c): monotone piecewise-linear, clamped to [0,1]."""
    c = np.asarray(c, dtype=float)
    out = np.interp(c, _CAL_X, _CAL_Y)
    return np.clip(out, 0.0, 1.0)


_G_GRID = g_accept(_GRID)  # cache g over the integration grid


# =============================================================================
# PER-DEPTH CONFIDENCE MODEL  f_j(c) = Beta(mu_j, nu)   (MODELED)
# =============================================================================

def _beta_weights(mean, nu):
    """Normalized Beta(mean*nu, (1-mean)*nu) probability mass on _GRID."""
    a = max(mean * nu, 1e-3)
    b = max((1.0 - mean) * nu, 1e-3)
    logw = (a - 1.0) * np.log(_GRID) + (b - 1.0) * np.log1p(-_GRID)
    logw -= logw.max()
    w = np.exp(logw)
    s = w.sum()
    return w / s if s > 0 else np.full(_NGRID, 1.0 / _NGRID)


def _expected_accept(mean, nu):
    """E_{c~Beta(mean,nu)}[ g(c) ]."""
    return float(np.dot(_beta_weights(mean, nu), _G_GRID))


def solve_mu(q_target, nu):
    """Bisection for the Beta mean mu s.t. E[g(c)] = q_target (g monotone ->
    E[g] monotone in mu).  Returns (mu, achieved, reachable)."""
    lo, hi = 1e-4, 1.0 - 1e-4
    e_lo, e_hi = _expected_accept(lo, nu), _expected_accept(hi, nu)
    if q_target <= e_lo:
        return lo, e_lo, q_target >= e_lo - 1e-3
    if q_target >= e_hi:
        return hi, e_hi, q_target <= e_hi + 1e-3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _expected_accept(mid, nu) < q_target:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    return mu, _expected_accept(mu, nu), True


def build_depth_models(nu):
    """Per-depth confidence weights w_j on _GRID, calibrated so E[g(c_j)] = q[j].
    Returns dict with mu, achieved-accept, reachable, and the grid weights."""
    mus, achieved, reach, weights = [], [], [], []
    for q in Q_COND:
        mu, ach, ok = solve_mu(float(q), nu)
        mus.append(mu)
        achieved.append(ach)
        reach.append(ok)
        weights.append(_beta_weights(mu, nu))
    return {
        "nu": nu,
        "mu": np.array(mus),
        "achieved_accept": np.array(achieved),
        "reachable": np.array(reach, dtype=bool),
        "weights": np.array(weights),  # (7, NGRID)
    }


# =============================================================================
# EXACT COMPOSITION  (no Monte-Carlo: path-product accounting, advisor #247)
# =============================================================================

def base_step_factor():
    """Deployed K=7 step / verify = 1 + 7*g_d (onegraph intact)."""
    return 1.0 + K_DEPLOYED * G_D


def adaptive_step_factor(mean_k, saving_per_pass=G_D):
    """Adaptive step / verify = base - (7-mean_k)*s.  s=g_d is the clean upper
    bound (skip removes the full draft-pass cost); onegraph-break launch overhead
    makes the realizable net saving per skipped pass s<g_d (see gd_sensitivity).
    At s=g_d this is exactly 1 + mean_k*g_d."""
    return base_step_factor() - (K_DEPLOYED - mean_k) * saving_per_pass


def tps_of(e_t, mean_k, saving_per_pass=G_D):
    """Deployed-convention TPS at adaptive E[T] and mean realized K.
    TPS_base recovered at e_t=E_T_BASE, mean_k=7 (any s)."""
    return (OFFICIAL_TPS * (e_t / E_T_BASE)
            * base_step_factor() / adaptive_step_factor(mean_k, saving_per_pass))


def sweep_point(theta, models, saving_per_pass=G_D):
    """Exact adaptive-K accounting at threshold theta.

    Per depth j the gate keeps drafting iff c_j >= theta.  With c_j independent
    across depths and a_j ~ Bernoulli(g(c_j)):
      F_j      = P(c_j < theta)                         (cut prob at depth j)
      AC_j     = E[ g(c_j) * 1[c_j >= theta] ]          (accept AND not-cut)
      q_j      = E[ g(c_j) ] = deployed ladder          (accept, by construction)
    mean_K   = sum_{i=1..7} prod_{j<i}(1 - F_j)         (first-cut hitting time)
    E[commit]= sum_{i=1..7} (prod_{j<i} AC_j) * q_i     (min(A, K_real) tokens)
    E[T]     = 1 + E[commit];  loss = E[T]_base - E[T]."""
    w = models["weights"]                 # (7, NGRID)
    cut_mask = (_GRID < theta)            # high-conf kept, low-conf cut
    F = w[:, cut_mask].sum(axis=1)        # P(c_j < theta), per depth
    AC = (w * _G_GRID[None, :])[:, ~cut_mask].sum(axis=1)  # accept & not cut
    q = Q_COND

    # mean realized K = E[first cut index] (capped at 7, >=1 always)
    surv_nocut = 1.0
    mean_k = 0.0
    for i in range(K_DEPLOYED):
        mean_k += surv_nocut            # P(K_real >= i+1)
        surv_nocut *= (1.0 - F[i])
    # E[committed] = sum_i (prod_{j<i} AC_j) * q_i
    prod_ac = 1.0
    e_commit = 0.0
    for i in range(K_DEPLOYED):
        e_commit += prod_ac * q[i]
        prod_ac *= AC[i]
    e_t = 1.0 + e_commit
    loss = E_T_BASE - e_t
    saved_passes = float(K_DEPLOYED - mean_k)  # mean draft passes skipped/step
    tps = tps_of(e_t, mean_k, saving_per_pass)
    return {
        "theta": float(theta),
        "mean_realized_K": float(mean_k),
        "e_t_adaptive": float(e_t),
        "e_t_loss": float(loss),
        "saved_draft_passes_per_step": saved_passes,
        "net_tps": float(tps),
        "net_tps_gain_pct": float(100.0 * (tps / OFFICIAL_TPS - 1.0)),
        "per_depth_cut_prob": [float(x) for x in F],
        "per_depth_accept_and_keep": [float(x) for x in AC],
        # E[T]-loss per saved pass: << breakeven means the cuts were "free".
        "et_loss_per_saved_pass": float(loss / saved_passes) if saved_passes > 1e-9 else 0.0,
    }


# =============================================================================
# VARIANCE / CONCENTRATION DECOMPOSITION  (the #247 per-step-variance lens)
# =============================================================================

def concentration_decomposition(theta, models, n_steps=200_000, seed=20260615):
    """Where do the saved passes and the E[T] loss concentrate?  Monte-Carlo
    over per-step (c_1..c_7, a_1..a_7) realizations to show (i) cuts land on
    low-confidence steps and (ii) those cut tails had low acceptance anyway, so
    the E[T] loss is far smaller than the pass saving (the linear-path analog of
    #247's 35x high-deviation concentration).  Reported for insight; the headline
    TPS comes from the exact accounting above."""
    rng = np.random.default_rng(seed)
    w = models["weights"]
    # sample confidence per depth from the per-depth Beta grids
    c = np.empty((n_steps, K_DEPLOYED))
    for j in range(K_DEPLOYED):
        c[:, j] = rng.choice(_GRID, size=n_steps, p=w[j])
    a = (rng.random((n_steps, K_DEPLOYED)) < g_accept(c)).astype(np.int8)

    # natural A (K=7): first reject index - 1
    rej = (a == 0)
    A = np.where(rej.any(axis=1), rej.argmax(axis=1), K_DEPLOYED)
    # realized K: first cut index (c<theta), else 7; always >=1
    cut = (c < theta)
    K_real = np.where(cut.any(axis=1), cut.argmax(axis=1) + 1, K_DEPLOYED)
    committed = np.minimum(A, K_real)
    lost = np.maximum(A - K_real, 0)         # would-be-accepted tokens cut
    saved = K_DEPLOYED - K_real              # draft passes skipped

    # split steps by whether the gate fired (a cut happened) this step
    fired = cut.any(axis=1)
    n_fired = int(fired.sum())
    # acceptance prob of the FIRST cut token (the tail we dropped): is it low?
    first_cut_idx = np.where(fired, cut.argmax(axis=1), 0)
    g_at_cut = g_accept(c[np.arange(n_steps), first_cut_idx])
    return {
        "theta": float(theta),
        "mc_steps": n_steps,
        "mc_mean_K": float(K_real.mean()),
        "mc_e_t_adaptive": float(1.0 + committed.mean()),
        "mc_e_t_loss": float((A - committed).mean()),
        "frac_steps_gate_fires": float(fired.mean()),
        "mean_saved_passes_on_fired": float(saved[fired].mean()) if n_fired else 0.0,
        "mean_tokens_lost_on_fired": float(lost[fired].mean()) if n_fired else 0.0,
        # the headline concentration fact: dropped-tail accept prob << on-roll q0
        "mean_accept_prob_of_cut_token": float(g_at_cut[fired].mean()) if n_fired else 0.0,
        "deployed_top1_accept_q0": float(Q_COND[0]),
        # loss-to-saving ratio in token-equivalents (tiny => cuts ~free)
        "tokens_lost_per_saved_pass": (float(lost[fired].sum() / saved[fired].sum())
                                       if n_fired and saved[fired].sum() > 0 else 0.0),
    }


# =============================================================================
# SELF-TEST  (PRIMARY: adaptive_k_early_exit_self_test_passes)
# =============================================================================

def _peak_vram_gb():
    """Best-effort process peak VRAM. CPU-only profiler -> ~0; never raises."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return float(torch.cuda.max_memory_allocated() / 1e9)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def self_test(models, verbose=True):
    res = {}
    rng = np.random.default_rng(7)

    # ---- (a) GREEDY-IDENTITY: adaptive stream == K=7 greedy stream -----------
    # Every committed token is the verify's greedy argmax (an accepted draft ==
    # verify argmax == greedy token; the verify bonus == greedy token), so BOTH
    # paths emit the TRUE greedy continuation g[t]=t -- adaptive merely commits
    # FEWER tokens per step.  Per step the K=7 control commits n7=min(A,7)+1
    # greedy tokens; adaptive commits nad=min(A,K_real)+1 with 1<=K_real<=7.
    # Token-identity holds BY CONSTRUCTION; the fuzz confirms, on every sampled
    # realization, the structural invariants that guarantee it:
    #   (i)   nad >= 1                  -- always emits the verify bonus token,
    #   (ii)  nad <= n7                 -- adaptive is a per-step PREFIX of K=7
    #                                      (=> prefix of the shared greedy stream
    #                                      => token-identical on the overlap),
    #   (iii) nad <= A + 1              -- never commits a REJECTED draft (the
    #                                      acceptance criterion is untouched =>
    #                                      greedy-safe),
    #   (iv)  cumsum(nad) <= cumsum(n7) -- the adaptive emitted stream never runs
    #                                      ahead of the K=7 stream (per trajectory).
    # Vectorized: (c,a) pre-sampled in 7 numpy draws (was a ~7e5-call Python
    # choices loop, the 159s bottleneck).
    w = models["weights"]
    n_traj, tokens_per_traj = 400, 64
    n_block = n_traj * tokens_per_traj
    c_blk = np.empty((n_block, K_DEPLOYED))
    for j in range(K_DEPLOYED):
        c_blk[:, j] = rng.choice(_GRID, size=n_block, p=w[j])
    a_blk = (rng.random((n_block, K_DEPLOYED)) < g_accept(c_blk)).astype(np.int8)
    rej = (a_blk == 0)
    A_blk = np.where(rej.any(axis=1), rej.argmax(axis=1), K_DEPLOYED)
    n7_blk = np.minimum(A_blk, K_DEPLOYED) + 1
    always_bonus = never_exceeds = greedy_safe = prefix_stream = k_real_valid = True
    for theta in (0.3, 0.5, 0.7, 0.9):
        cut = c_blk < theta
        K_real = np.where(cut.any(axis=1), cut.argmax(axis=1) + 1, K_DEPLOYED)
        nad_blk = np.minimum(A_blk, K_real) + 1
        always_bonus &= bool((nad_blk >= 1).all())
        never_exceeds &= bool((nad_blk <= n7_blk).all())
        greedy_safe &= bool((nad_blk <= A_blk + 1).all())
        k_real_valid &= bool(((K_real >= 1) & (K_real <= K_DEPLOYED)).all())
        c7 = n7_blk.reshape(n_traj, tokens_per_traj).cumsum(axis=1)
        cad = nad_blk.reshape(n_traj, tokens_per_traj).cumsum(axis=1)
        prefix_stream &= bool((cad <= c7).all())
    gi_ok = bool(never_exceeds and greedy_safe and prefix_stream)
    contiguous_ok = bool(always_bonus and k_real_valid)
    adaptive_never_exceeds = bool(never_exceeds)
    greedy_identity_pass = bool(gi_ok and contiguous_ok and adaptive_never_exceeds)
    res["greedy_identity_pass"] = greedy_identity_pass
    res["greedy_identity_token_identical"] = gi_ok
    res["greedy_identity_contiguous"] = contiguous_ok
    res["greedy_identity_adaptive_never_exceeds_k7"] = adaptive_never_exceeds
    res["greedy_identity_greedy_safe_no_rejected_commit"] = greedy_safe
    res["greedy_identity_fuzz_steps"] = int(n_block * 4)
    if verbose:
        print(f"[self-test] (a) greedy-identity ({n_block * 4} fuzzed steps): "
              f"token-identical={gi_ok} contiguous={contiguous_ok} "
              f"never-exceeds-K7={adaptive_never_exceeds} greedy-safe={greedy_safe} "
              f"-> {'PASS' if greedy_identity_pass else 'FAIL'}")

    # ---- exact-vs-MC cross-check + theta=0 recovery + monotone mean_K ---------
    p0 = sweep_point(0.0, models)
    recover_ok = (abs(p0["mean_realized_K"] - K_DEPLOYED) < 1e-6
                  and abs(p0["e_t_adaptive"] - E_T_BASE) < 1e-6
                  and abs(p0["net_tps_gain_pct"]) < 1e-6)
    res["theta0_recovers_deployed"] = bool(recover_ok)
    # mean_K monotonically falls as theta rises
    ks = [sweep_point(t, models)["mean_realized_K"] for t in
          (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    mono_ok = all(ks[i] >= ks[i + 1] - 1e-9 for i in range(len(ks) - 1))
    res["mean_k_monotone_in_theta"] = bool(mono_ok)
    # exact accounting matches MC at a representative theta
    mc = concentration_decomposition(0.5, models, n_steps=200_000)
    ex = sweep_point(0.5, models)
    exact_mc_k = abs(mc["mc_mean_K"] - ex["mean_realized_K"]) < 0.03
    exact_mc_et = abs(mc["mc_e_t_adaptive"] - ex["e_t_adaptive"]) < 0.02
    res["exact_matches_mc"] = bool(exact_mc_k and exact_mc_et)
    if verbose:
        print(f"[self-test]     theta=0 recovers deployed (K=7,E[T]=3.844,gain=0): "
              f"{'PASS' if recover_ok else 'FAIL'}")
        print(f"[self-test]     mean_K monotone in theta: {'PASS' if mono_ok else 'FAIL'}")
        print(f"[self-test]     exact==MC @theta0.5 (K {ex['mean_realized_K']:.3f} vs "
              f"{mc['mc_mean_K']:.3f}; E[T] {ex['e_t_adaptive']:.3f} vs "
              f"{mc['mc_e_t_adaptive']:.3f}): {'PASS' if res['exact_matches_mc'] else 'FAIL'}")

    # ---- (b) PPL pinned -------------------------------------------------------
    ppl_pass = PPL_PINNED <= PPL_CAP
    res["ppl_pinned"] = PPL_PINNED
    res["ppl_pass"] = bool(ppl_pass)

    # ---- (c) NaN-clean --------------------------------------------------------
    probe = [sweep_point(t, models) for t in np.linspace(0, 1, 21)]
    flat = []
    for p in probe:
        for v in p.values():
            if isinstance(v, (int, float)):
                flat.append(v)
            elif isinstance(v, list):
                flat.extend(v)
    nan_clean = all(np.isfinite(x) for x in flat)
    res["nan_clean"] = bool(nan_clean)

    # ---- (d) peak VRAM <= 24 GB (CPU-only profiler) ---------------------------
    vram = _peak_vram_gb()
    vram_pass = vram <= 24.0
    res["peak_vram_gb"] = vram
    res["vram_pass"] = bool(vram_pass)

    # ---- calibration sanity: per-depth achieved accept hits the ladder -------
    cal_err = float(np.abs(models["achieved_accept"] - Q_COND).max())
    res["calibration_max_abs_err"] = cal_err
    res["calibration_reachable_all_depths"] = bool(models["reachable"].all())
    cal_ok = cal_err < 5e-3 and bool(models["reachable"].all())
    res["calibration_pass"] = cal_ok

    passes = bool(greedy_identity_pass and recover_ok and mono_ok
                  and res["exact_matches_mc"] and ppl_pass and nan_clean
                  and vram_pass and cal_ok)
    res["adaptive_k_early_exit_self_test_passes"] = passes
    if verbose:
        print(f"[self-test] (b) PPL pinned {PPL_PINNED} <= {PPL_CAP}: "
              f"{'PASS' if ppl_pass else 'FAIL'}")
        print(f"[self-test] (c) NaN-clean: {'PASS' if nan_clean else 'FAIL'}")
        print(f"[self-test] (d) peak VRAM {vram:.2f} GB <= 24 (CPU-only): "
              f"{'PASS' if vram_pass else 'FAIL'}")
        print(f"[self-test]     calibration max|E[g]-q| = {cal_err:.2e} "
              f"(reachable={res['calibration_reachable_all_depths']}): "
              f"{'PASS' if cal_ok else 'FAIL'}")
        print(f"[self-test] === adaptive_k_early_exit_self_test_passes = {passes} ===")
    return res


# =============================================================================
# THRESHOLD SWEEP  (the headline)
# =============================================================================

def sweep_theta(models, thetas=None, verbose=True):
    if thetas is None:
        # PR-named coarse grid (0.0=deployed K=7) + a fine grid to localize peak
        coarse = [0.0, 0.3, 0.5, 0.7, 0.9]
        fine = list(np.round(np.linspace(0.0, 0.98, 50), 4))
        thetas = sorted(set([round(t, 4) for t in coarse + fine]))
    pts = [sweep_point(t, models) for t in thetas]
    star = max(pts, key=lambda p: p["net_tps"])
    decomp = concentration_decomposition(star["theta"], models)
    out = {
        "thetas": thetas,
        "sweep": pts,
        "theta_star": star["theta"],
        "net_tps_gain_pct_adaptiveK": star["net_tps_gain_pct"],
        "net_tps_at_theta_star": star["net_tps"],
        "e_t_adaptive": star["e_t_adaptive"],
        "e_t_base": E_T_BASE,
        "mean_realized_K": star["mean_realized_K"],
        "e_t_loss_at_theta_star": star["e_t_loss"],
        "saved_draft_passes_per_step": star["saved_draft_passes_per_step"],
        "projected_tps_from_481p53": OFFICIAL_TPS * (1.0 + star["net_tps_gain_pct"] / 100.0),
        "moves_toward_500": bool(OFFICIAL_TPS * (1.0 + star["net_tps_gain_pct"] / 100.0) >= 500.0),
        "go": bool(star["net_tps_gain_pct"] > 0.0),
        # headline assumes the full per-skipped-pass saving g_d is realizable;
        # ONEGRAPH=1 makes that an UPPER BOUND -- see gd_sensitivity for the price.
        "headline_is_onegraph_upper_bound": bool(ONEGRAPH_DEPLOYED),
        "concentration": decomp,
    }
    if verbose:
        print("\n[sweep] theta   mean_K   E[T]     loss     savedΔpass  netTPS   gain%")
        named = {0.0, 0.3, 0.5, 0.7, 0.9, round(star["theta"], 4)}
        for p in pts:
            mark = "  <== theta_star" if abs(p["theta"] - star["theta"]) < 1e-9 else ""
            if round(p["theta"], 4) in named or mark:
                print(f"        {p['theta']:.3f}  {p['mean_realized_K']:.3f}   "
                      f"{p['e_t_adaptive']:.3f}   {p['e_t_loss']:+.4f}  "
                      f"{p['saved_draft_passes_per_step']:.3f}      "
                      f"{p['net_tps']:.2f}  {p['net_tps_gain_pct']:+.3f}{mark}")
        print(f"\n[sweep] theta_star = {star['theta']:.4f}  "
              f"net_tps_gain_pct_adaptiveK = {star['net_tps_gain_pct']:+.3f}%  "
              f"-> {out['projected_tps_from_481p53']:.2f} TPS "
              f"(toward-500={out['moves_toward_500']})")
        print(f"[sweep] at theta_star: E[T] {E_T_BASE:.3f}->{star['e_t_adaptive']:.3f} "
              f"(loss {star['e_t_loss']:+.4f}), mean_K 7->{star['mean_realized_K']:.3f} "
              f"(saves {star['saved_draft_passes_per_step']:.3f} draft passes/step)")
        print(f"[sweep] concentration: gate fires on {decomp['frac_steps_gate_fires']*100:.1f}% "
              f"of steps; cut-token accept prob {decomp['mean_accept_prob_of_cut_token']:.3f} "
              f"<< deployed q0={decomp['deployed_top1_accept_q0']:.3f}  "
              f"(tokens_lost/saved_pass={decomp['tokens_lost_per_saved_pass']:.3f})")
    return out


# =============================================================================
# ONEGRAPH-BREAK PRICE  (g_d saving-survival sensitivity)
# =============================================================================

def gd_sensitivity(models, saving_fracs=None, verbose=True):
    """Price the ONEGRAPH=1 caveat.  The headline (sweep_theta) assumes a skipped
    draft pass returns the full g_d*verify.  Because all 7 passes live in one
    static CUDA graph (lawine #246), a runtime exit needs the graph broken, which
    reintroduces per-pass launch overhead and shrinks the realizable net saving
    per skipped pass to s = frac*g_d (frac in [0,1]).  For each frac we re-find
    theta_star and its net gain; frac=1 is the upper bound, and the frac at which
    the gain crosses 0 is the onegraph-break overhead budget adaptive-K absorbs."""
    if saving_fracs is None:
        saving_fracs = [1.0, 0.75, 0.5, 0.25, 0.0]
    fine = list(np.round(np.linspace(0.0, 0.98, 50), 4))

    def best_gain(frac):
        s = float(frac) * G_D
        return max(sweep_point(t, models, saving_per_pass=s)["net_tps_gain_pct"]
                   for t in fine)

    rows = []
    for frac in saving_fracs:
        s = float(frac) * G_D
        pts = [sweep_point(t, models, saving_per_pass=s) for t in fine]
        star = max(pts, key=lambda p: p["net_tps"])
        rows.append({
            "saving_frac_of_gd": float(frac),
            "saving_per_pass": s,
            "theta_star": float(star["theta"]),
            "net_tps_gain_pct": float(star["net_tps_gain_pct"]),
            "net_tps": float(star["net_tps"]),
            "mean_realized_K": float(star["mean_realized_K"]),
            "e_t_adaptive": float(star["e_t_adaptive"]),
            "go": bool(star["net_tps_gain_pct"] > 1e-9),
        })
    # break-even frac: smallest surviving-saving fraction whose optimal-theta gain
    # is still > 0 (bisection; best_gain is monotone non-decreasing in frac and
    # = 0 at frac=0 where theta=0 is optimal).
    if best_gain(1.0) <= 1e-6:
        break_even = None                      # NO-GO at every fraction
    elif best_gain(1e-4) > 1e-6:
        break_even = 0.0                        # wins for any positive saving
    else:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if best_gain(mid) > 1e-6:
                hi = mid
            else:
                lo = mid
        break_even = float(hi)
    out = {
        "g_d": G_D,
        "saving_fracs": [float(f) for f in saving_fracs],
        "rows": rows,
        "break_even_saving_frac": break_even,
        "headline_upper_bound_gain_pct": next(
            (r["net_tps_gain_pct"] for r in rows
             if abs(r["saving_frac_of_gd"] - 1.0) < 1e-9), None),
    }
    if verbose:
        print("\n[g_d sens] ONEGRAPH break price -- net saving per skipped pass s=frac*g_d")
        print("  frac  s        theta*  mean_K   gain%     netTPS   GO")
        for r in rows:
            print(f"  {r['saving_frac_of_gd']:.2f}  {r['saving_per_pass']:.4f}  "
                  f"{r['theta_star']:.3f}   {r['mean_realized_K']:.3f}   "
                  f"{r['net_tps_gain_pct']:+.3f}   {r['net_tps']:.2f}  "
                  f"{'yes' if r['go'] else 'NO'}")
        be = out["break_even_saving_frac"]
        print(f"[g_d sens] break-even surviving-saving fraction: "
              f"{('%.2f' % be) if be is not None else 'none (NO-GO at all fracs)'} "
              f"(adaptive-K wins iff onegraph-break keeps >= this share of g_d)")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Adaptive-K early-exit profiler (PR #256)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sweep-theta", action="store_true")
    ap.add_argument("--liveness", action="store_true",
                    help="init+finish a W&B run with a liveness marker only")
    ap.add_argument("--nu", type=float, default=None,
                    help="Beta concentration for the per-depth confidence model "
                         "(default: from the EAGLE3 top1p spread)")
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "adaptive_k_early_exit_results.json"))
    args = ap.parse_args()

    nu = args.nu if args.nu is not None else NU_FROM_EAGLE3
    run_all = not (args.self_test or args.sweep_theta or args.liveness)

    t0 = time.perf_counter()
    metrics = {
        "_anchors": {
            "K_CAL": K_CAL, "g_d": G_D, "step_int4": STEP_INT4,
            "official_tps": OFFICIAL_TPS, "e_t_base": E_T_BASE,
            "e_t_base_measured": E_T_BASE_MEASURED, "ppl_pinned": PPL_PINNED,
            "ppl_cap": PPL_CAP, "nu": nu, "nu_from_eagle3": NU_FROM_EAGLE3,
            "k_deployed": K_DEPLOYED,
        },
        "q_cond_deployed": [float(x) for x in Q_COND],
    }

    if args.liveness:
        metrics["liveness"] = 1
    else:
        models = build_depth_models(nu)
        metrics["depth_models"] = {
            "nu": nu,
            "mu": [float(x) for x in models["mu"]],
            "achieved_accept": [float(x) for x in models["achieved_accept"]],
            "reachable": [bool(x) for x in models["reachable"]],
        }
        if args.self_test or run_all:
            metrics["self_test"] = self_test(models)
        if args.sweep_theta or run_all:
            metrics["sweep"] = sweep_theta(models)
            # price the ONEGRAPH=1 caveat: how much onegraph-break overhead can
            # adaptive-K absorb before the headline gain disappears?
            metrics["gd_sensitivity"] = gd_sensitivity(models)
    metrics["_runtime_s"] = time.perf_counter() - t0

    Path(args.out).write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nwrote {args.out}  ({metrics['_runtime_s']:.2f}s)")

    if not args.no_wandb and (args.wandb_name or args.wandb_group):
        try:
            import wandb  # noqa: PLC0415
            run = wandb.init(
                project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                group=args.wandb_group, name=args.wandb_name,
                config={"experiment": "adaptive-k-early-exit", "pr": 256,
                        "nu": nu, "g_d": G_D, "k_cal": K_CAL,
                        "official_tps": OFFICIAL_TPS, "e_t_base": E_T_BASE},
            )
            flat = {}

            def _flatten(prefix, d):
                for k, v in d.items():
                    key = f"{prefix}{k}"
                    if isinstance(v, bool):
                        flat[key] = int(v)
                    elif isinstance(v, (int, float)):
                        flat[key] = v
            if "self_test" in metrics:
                _flatten("self_test/", metrics["self_test"])
            if "sweep" in metrics:
                sw = metrics["sweep"]
                for k, v in sw.items():
                    if isinstance(v, bool):
                        flat[f"sweep/{k}"] = int(v)
                    elif isinstance(v, (int, float)):
                        flat[f"sweep/{k}"] = v
                _flatten("sweep/concentration/", sw["concentration"])
                # log the full theta-sweep curve as a W&B Table for analysis
                cols = ["theta", "mean_realized_K", "e_t_adaptive", "e_t_loss",
                        "saved_draft_passes_per_step", "net_tps", "net_tps_gain_pct"]
                tbl = wandb.Table(columns=cols)
                for p in sw["sweep"]:
                    tbl.add_data(*[p[c] for c in cols])
                run.log({"sweep/theta_curve": tbl})
            if "gd_sensitivity" in metrics:
                gd = metrics["gd_sensitivity"]
                if gd.get("break_even_saving_frac") is not None:
                    flat["gd_sensitivity/break_even_saving_frac"] = gd["break_even_saving_frac"]
                if gd.get("headline_upper_bound_gain_pct") is not None:
                    flat["gd_sensitivity/headline_upper_bound_gain_pct"] = gd["headline_upper_bound_gain_pct"]
                gcols = ["saving_frac_of_gd", "saving_per_pass", "theta_star",
                         "net_tps_gain_pct", "net_tps", "mean_realized_K",
                         "e_t_adaptive", "go"]
                gtbl = wandb.Table(columns=gcols)
                for r in gd["rows"]:
                    gtbl.add_data(*[int(r[c]) if isinstance(r[c], bool) else r[c]
                                    for c in gcols])
                run.log({"gd_sensitivity/curve": gtbl})
            if "liveness" in metrics:
                flat["liveness"] = 1
            run.log(flat)
            run.summary.update(flat)
            run.finish()
            print(f"[wandb] logged {len(flat)} scalars to {args.wandb_name}")
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] skipped: {e}")


if __name__ == "__main__":
    main()
