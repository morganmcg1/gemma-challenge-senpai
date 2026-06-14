"""Frozen-vs-fresh LOCAL regime pin (lawine #209).

Re-benchmark the deployed ``fa2sw_precache_kenyan`` served checkpoint across
N>=8 fresh ``LocalServer`` reloads, decompose the run-to-run ``wall_tps``
variance, and pin whether the LOCAL re-benchmark harness is **FROZEN**
(sigma_reload ~ hardware-timing only, the relative analog of #188's sigma_hw) or
**FRESH** (sigma_reload also carries a prompt/sampling resample component ~
kanna #202's sigma_draw).

This empirically resolves kanna #202's (`533jd6l1`) load-bearing regime
assumption — the one fern #185 carries as a *conservative default*
(``mu_bar_frozen_p95``=504.87) — using LOCAL serves only: no official draw, no
human gate, adds 0 TPS. It doubles as a determinism audit: under the token-
identity contract (fixed 128 prompts + deterministic greedy => identical tokens
every run) the ONLY run-to-run ``wall_tps`` variance source IS hardware timing
=> FROZEN; a measured sigma_reload > sigma_hw with token divergence would expose
a non-determinism source (a #192-relevant finding).

LOCAL-only, contract-neutral: serves the deployed stack UNCHANGED, decode-only
timed runs. No HF Job / submission / official draw / served-file change.
BASELINE stays 481.53. Greedy/PPL untouched.

Reuses the #72/#196 measurement VERBATIM (``run_noise_floor.run_fresh`` +
``timed_decode`` + ``build_serve_env`` + ``preflight_gpu``) — does NOT reinvent
the measurement. Imports (does NOT re-derive): the #180 wall->official bridge
(``local_official_projection``), kanna #202's sigma decomposition + budget bars,
#188's sigma_hw, and #196's reload-vs-reload ``self_identity`` pattern.

Run under the repo ``.venv`` (has wandb); serve/decode subprocs use the
submission's own serve venv. Example::

    .venv/bin/python research/validity/frozen_regime_local_pin/run_regime_pin.py \
        --reloads 8 --wandb_group frozen-regime-local-pin \
        --wandb_name lawine/frozen-regime-local-pin
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
# Reuse the #72 fresh-reload measurement harness verbatim -- do NOT reinvent it.
from research.tps_noise_floor.run_noise_floor import run_fresh  # noqa: E402

# PR #180/#99 wall->official bridge (imported, not re-derived). Optional: the raw
# wall_tps regime pin stands even if the projection import is unavailable.
try:
    from scripts.profiler import local_official_projection as projection  # noqa: E402
except Exception:  # pragma: no cover - defensive
    projection = None

OUT_ROOT = ROOT / "research" / "validity" / "frozen_regime_local_pin"

# ---------------------------------------------------------------------------
# Imported constants (PR #209: IMPORT, do NOT re-derive).
# ---------------------------------------------------------------------------
# kanna #202 (533jd6l1) sigma decomposition, in OFFICIAL-TPS units.
SIGMA_HW = 4.864        # #188 (pp1r5orx) one-shot hardware std (timing re-draw only)
SIGMA_SAMPLE = 5.564    # per-checkpoint / prompt-resample component
SIGMA_DRAW = 7.391      # total fresh-draw std = sigma_sample (+) sigma_hw
FROZEN_FRACTION_BREAKEVEN = 0.846   # kanna #202 partial-freeze breakeven coordinate

# kanna #202 / fern #185 budget bars (OFFICIAL TPS), and the official anchors.
MU_BUILD = 512.2            # kanna #202 build-to-mu (N=1 freeze-robust target)
MU_BAR_FROZEN_P95 = 504.87  # conservative FROZEN build bar (fern #185 default)
MU_BAR_FRESH_P95 = 499.08   # looser FRESH bar (admissible only if regime is FRESH)
OFFICIAL_BASELINE_TPS = 481.53
OFFICIAL_TARGET_TPS = 500.0

# PR #180-nominal bridge (cross-checked against the calibrated value at runtime).
BRIDGE_NOMINAL = 1.0602

# Frozen test tolerance: the harness is FROZEN if sigma_reload (official units) is
# within this fraction above sigma_hw (i.e. not inflated above the timing band).
FROZEN_TOL = 0.10

EXPECTED_OUTPUT_LEN = paths.OUTPUT_LEN     # 512
EXPECTED_PROMPTS = paths.NUM_PROMPTS       # 128

random.seed(20260614)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# Step 1 -- N fresh-reload re-benchmarks (reuse #72 run_fresh verbatim)
# ---------------------------------------------------------------------------
def run_reloads(args, out_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """N fresh-server decode-only timed runs of the deployed stack.

    Each iteration is a full LocalServer boot + 128-prompt x 512-token conc=1
    decode (the #196 way). Returns the per-reload records (with ``wall_tps``) and
    the sorted decode-capture paths (each carries per-prompt
    ``completion_token_ids`` for the identity leg)."""
    submission_dir = (ROOT / "submissions" / args.submission).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[regime] submission={submission_dir.name} server_python={server_python} "
          f"reloads={args.reloads}", flush=True)

    # run_fresh consumes a noise-floor-style args namespace.
    nf_args = SimpleNamespace(
        n_runs=args.reloads,
        num_prompts=args.num_prompts,
        output_len=args.output_len,
        seed=args.seed,
        clock_interval_ms=args.clock_interval_ms,
        settle_s=args.settle_s,
        steptime=args.steptime,
    )
    records_path = out_dir / "records.jsonl"
    with open(records_path, "w") as records_fh:
        records = run_fresh(nf_args, submission_dir, server_python, out_dir, records_fh)
    decode_paths = sorted((out_dir / "decode").glob("run*.jsonl"))
    return records, decode_paths


# ---------------------------------------------------------------------------
# Step 2 -- token identity across reloads (the determinism leg, #196 pattern)
# ---------------------------------------------------------------------------
def _load_decode(path: Path) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = r.get("id", r.get("index"))
        out[key] = {
            "tokens": r.get("completion_token_ids"),
            "sha": r.get("completion_token_sha256"),
        }
    return out


def token_identity_across_reloads(decode_paths: list[Path]) -> dict[str, Any]:
    """Reload-vs-reload byte-identity of completion_token_ids vs reload-0.

    Reports the token-level identity fraction across the 128 x 512 tokens (PR
    item 2), the per-prompt all-equal fraction, the number of divergent reloads,
    and a structural sanity check (128/128, output_len uniform 512)."""
    runs = [_load_decode(p) for p in decode_paths]
    n_runs = len(runs)
    ref = runs[0]
    keys = sorted(ref.keys(), key=lambda k: (str(type(k)), k))
    n_prompts = len(keys)

    total_positions = 0
    matched_positions = 0
    n_divergent_reloads = 0
    pairwise_token_rate_vs_run0: dict[str, float] = {}
    prompt_all_equal = {k: True for k in keys}
    first_divergences: list[dict[str, Any]] = []

    for i in range(1, n_runs):
        run_i = runs[i]
        div_reload = False
        run_total = 0
        run_matched = 0
        for k in keys:
            a = ref.get(k, {}).get("tokens")
            b = run_i.get(k, {}).get("tokens")
            if a is None or b is None:
                prompt_all_equal[k] = False
                div_reload = True
                if len(first_divergences) < 20:
                    first_divergences.append({"key": k, "reload": i, "reason": "missing_in_a_run"})
                continue
            if a == b:
                run_matched += len(a)
                run_total += len(a)
                continue
            # Slow path only when this prompt diverged (expected: never).
            prompt_all_equal[k] = False
            div_reload = True
            lo = min(len(a), len(b))
            hi = max(len(a), len(b))
            run_matched += sum(1 for p in range(lo) if a[p] == b[p])
            run_total += hi
            if len(first_divergences) < 20:
                pos = next((p for p in range(lo) if a[p] != b[p]), lo)
                first_divergences.append({
                    "key": k, "reload": i, "first_diff_pos": pos,
                    "len_run0": len(a), "len_run_i": len(b),
                })
        total_positions += run_total
        matched_positions += run_matched
        pairwise_token_rate_vs_run0[f"run0_vs_run{i}"] = (
            run_matched / run_total if run_total else float("nan"))
        if div_reload:
            n_divergent_reloads += 1

    token_identity_rate = (matched_positions / total_positions
                           if total_positions else float("nan"))
    n_all_equal = sum(1 for k in keys if prompt_all_equal[k])
    prompt_identity_rate = (n_all_equal / n_prompts if n_prompts else float("nan"))

    # Structural sanity (128/128 and uniform 512 every reload).
    per_run_counts = [len(r) for r in runs]
    per_run_total_tokens = [
        sum(len(v["tokens"] or []) for v in r.values()) for r in runs]
    per_run_uniform_512 = [
        all(len(v["tokens"] or []) == EXPECTED_OUTPUT_LEN for v in r.values())
        for r in runs]
    completes_128 = (all(c == EXPECTED_PROMPTS for c in per_run_counts)
                     and all(t == EXPECTED_PROMPTS * EXPECTED_OUTPUT_LEN
                             for t in per_run_total_tokens))

    return {
        "n_reloads": n_runs,
        "n_prompts": n_prompts,
        "token_identity_rate_across_reloads": token_identity_rate,
        "prompt_all_equal_rate": prompt_identity_rate,
        "n_prompts_all_equal": n_all_equal,
        "n_divergent_reloads": n_divergent_reloads,
        "pairwise_token_rate_vs_run0": pairwise_token_rate_vs_run0,
        "all_reloads_token_identical": bool(n_prompts) and token_identity_rate == 1.0,
        "per_run_record_count": per_run_counts,
        "per_run_total_tokens": per_run_total_tokens,
        "per_run_output_len_uniform_512": per_run_uniform_512,
        "completes_128": completes_128,
        "decode_files": [str(p) for p in decode_paths],
        "divergences": first_divergences,
    }


# ---------------------------------------------------------------------------
# Step 3 -- variance decomposition / regime pin
# ---------------------------------------------------------------------------
def bootstrap_std_ci(values: list[float], reps: int = 8000,
                     alpha: float = 0.05) -> dict[str, Any]:
    """Bootstrap CI for the run-to-run std (resample reloads w/ replacement).

    Dependency-free (no scipy); the UPPER bound is what the frozen test needs
    (sigma_reload_hi <= sigma_hw band). Lower bound can hit 0 at small N when a
    resample draws all-equal -- that is expected and harmless here."""
    vals = [float(v) for v in values if _is_num(v)]
    n = len(vals)
    point = statistics.stdev(vals) if n > 1 else 0.0
    if n < 2:
        return {"std_point": point, "std_lo": None, "std_hi": None, "n": n}
    stds = []
    for _ in range(reps):
        sample = [random.choice(vals) for _ in range(n)]
        stds.append(statistics.stdev(sample))
    stds.sort()
    lo = stds[int((alpha / 2) * reps)]
    hi = stds[min(reps - 1, int((1 - alpha / 2) * reps))]
    return {"std_point": point, "std_lo": lo, "std_hi": hi, "n": n, "reps": reps}


def variance_decomposition(reload_walltps: list[float], bridge: float) -> dict[str, Any]:
    """sigma_reload (wall + official units) vs the imported sigma_hw / sigma_draw,
    the f_resample_local partial-freeze coordinate, and the regime verdict."""
    vals = [float(v) for v in reload_walltps if _is_num(v)]
    n = len(vals)
    mean_wall = statistics.fmean(vals) if vals else float("nan")
    sigma_wall = statistics.stdev(vals) if n > 1 else 0.0
    cv_reload = 100.0 * sigma_wall / mean_wall if mean_wall else float("nan")

    boot = bootstrap_std_ci(vals)
    sigma_wall_hi = boot.get("std_hi")
    sigma_wall_lo = boot.get("std_lo")

    # wall -> official: the bridge is a pure multiplicative hardware/env transfer,
    # so the std scales by the same factor as the mean.
    sigma_off = sigma_wall * bridge
    sigma_off_hi = (sigma_wall_hi * bridge) if _is_num(sigma_wall_hi) else None
    sigma_off_lo = (sigma_wall_lo * bridge) if _is_num(sigma_wall_lo) else None
    mean_off = mean_wall * bridge

    # f_resample = (sigma_reload^2 - sigma_hw^2) / sigma_sample^2, clipped to [0,1]
    # (kanna #202 partial-freeze coordinate). 0 => FROZEN (no resample term),
    # 1 => FRESH (full sigma_draw).
    f_raw = (sigma_off ** 2 - SIGMA_HW ** 2) / (SIGMA_SAMPLE ** 2)
    f_resample = max(0.0, min(1.0, f_raw))

    frozen_threshold = SIGMA_HW * (1.0 + FROZEN_TOL)
    fresh_threshold = SIGMA_DRAW * (1.0 - FROZEN_TOL)
    # Point-estimate verdict; the CI upper bound corroborates "statistically <=".
    is_frozen_point = sigma_off <= frozen_threshold
    is_frozen_ci = (sigma_off_hi is not None and sigma_off_hi <= frozen_threshold)
    if sigma_off <= frozen_threshold:
        regime_local = "FROZEN"
    elif sigma_off >= fresh_threshold:
        regime_local = "FRESH"
    else:
        regime_local = "INTERMEDIATE"

    return {
        "n_reloads": n,
        "reload_walltps": vals,
        "mean_reload_walltps": mean_wall,
        "sigma_reload_walltps": sigma_wall,
        "sigma_reload_walltps_ci95": [sigma_wall_lo, sigma_wall_hi],
        "cv_reload_pct": cv_reload,
        "bridge_wall_to_official": bridge,
        "mean_reload_official": mean_off,
        "sigma_reload_official": sigma_off,
        "sigma_reload_official_ci95": [sigma_off_lo, sigma_off_hi],
        "sigma_hw": SIGMA_HW,
        "sigma_sample": SIGMA_SAMPLE,
        "sigma_draw": SIGMA_DRAW,
        "frozen_threshold_official": frozen_threshold,
        "fresh_threshold_official": fresh_threshold,
        "f_resample_local": f_resample,
        "f_resample_local_raw": f_raw,
        "frozen_fraction_breakeven_kanna202": FROZEN_FRACTION_BREAKEVEN,
        "local_harness_is_frozen": bool(is_frozen_point),
        "local_harness_is_frozen_ci": bool(is_frozen_ci),
        "regime_local": regime_local,
        # honest secondary diagnostic: does sigma_reload quantitatively REPRODUCE
        # sigma_hw (two-sided), or is the local stack simply quieter than the
        # official hardware band? (see report note).
        "sigma_reload_over_sigma_hw": (sigma_off / SIGMA_HW) if SIGMA_HW else None,
        "sigma_reload_reproduces_sigma_hw_twosided": bool(
            0.5 * SIGMA_HW <= sigma_off <= 1.5 * SIGMA_HW),
    }


# ---------------------------------------------------------------------------
# Step 4 -- budget implication
# ---------------------------------------------------------------------------
def budget_implication(vd: dict[str, Any]) -> dict[str, Any]:
    regime = vd["regime_local"]
    frozen = regime == "FROZEN"
    if frozen:
        text = (
            f"LOCAL re-benchmark harness is FROZEN (sigma_reload="
            f"{vd['sigma_reload_official']:.3f} official-TPS, f_resample="
            f"{vd['f_resample_local']:.3f}, well below sigma_hw={SIGMA_HW} and "
            f"sigma_draw={SIGMA_DRAW}). fern #185's conservative "
            f"mu_bar_frozen_p95={MU_BAR_FROZEN_P95} default is the empirically "
            f"correct one under the shared token-identity contract; the looser "
            f"FRESH bar {MU_BAR_FRESH_P95} is NOT admissible from the local "
            f"evidence. The OFFICIAL-scorer regime remains kanna #202's "
            f"human-gated pin, but the local result corroborates FROZEN.")
    elif regime == "FRESH":
        text = (
            f"LOCAL harness is FRESH (sigma_reload="
            f"{vd['sigma_reload_official']:.3f} ~ sigma_draw={SIGMA_DRAW}, "
            f"f_resample={vd['f_resample_local']:.3f}): a prompt/sampling "
            f"resample component exists, so the looser FRESH bar "
            f"{MU_BAR_FRESH_P95} would be admissible locally — but check the "
            f"determinism leg, because FRESH with token-identity 1.0 is "
            f"contradictory.")
    else:
        text = (
            f"LOCAL harness is INTERMEDIATE (f_resample="
            f"{vd['f_resample_local']:.3f}): partial-freeze; neither bar is "
            f"cleanly supported. Compare against kanna #202's "
            f"frozen_fraction_breakeven={FROZEN_FRACTION_BREAKEVEN}.")
    return {
        "regime_local": regime,
        "local_harness_is_frozen": vd["local_harness_is_frozen"],
        "f_resample_local": vd["f_resample_local"],
        "mu_build_n1": MU_BUILD,
        "mu_bar_frozen_p95": MU_BAR_FROZEN_P95,
        "mu_bar_fresh_p95": MU_BAR_FRESH_P95,
        "conservative_frozen_default_empirically_confirmed": frozen,
        "fresh_bar_admissible_locally": regime in ("FRESH", "INTERMEDIATE"),
        "official_baseline_tps": OFFICIAL_BASELINE_TPS,
        "official_target_tps": OFFICIAL_TARGET_TPS,
        "caveat": ("LOCAL LocalServer harness MAY differ from the OFFICIAL "
                   "scorer's re-benchmark behavior; the official pin remains "
                   "kanna #202's human-gated test. A frozen LOCAL result is "
                   "strong corroboration under the shared token-identity contract."),
        "verdict_text": text,
    }


# ---------------------------------------------------------------------------
# Step 5 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(ident: dict[str, Any], vd: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    rate = ident["token_identity_rate_across_reloads"]
    sigma_wall = vd["sigma_reload_walltps"]
    sigma_off = vd["sigma_reload_official"]
    f = vd["f_resample_local"]

    numeric_fields = [rate, sigma_wall, sigma_off, vd["mean_reload_walltps"],
                      vd["mean_reload_official"], f, ident["prompt_all_equal_rate"]]
    nan_clean = all(_is_num(x) for x in numeric_fields)

    checks = {
        # (a) deterministic harness => FROZEN consistent (the contract self-check)
        "a_token_identity_1p0": _is_num(rate) and rate == 1.0,
        # (b) a real measurement, not a degenerate single value
        "b_sigma_reload_finite_positive": _is_num(sigma_wall) and sigma_wall > 0.0,
        # (c) sigma_hw anchor (one-sided frozen-consistency): sigma_reload is in the
        #     timing-only band, i.e. NOT inflated above sigma_hw*(1+tol). [The strict
        #     two-sided "reproduces sigma_hw" is reported separately and is FALSE
        #     when the local stack is quieter than the official hardware band.]
        "c_sigma_reload_consistent_with_frozen": vd["local_harness_is_frozen"],
        # (d) well-posed partial-freeze coordinate
        "d_f_resample_in_unit_interval": _is_num(f) and 0.0 <= f <= 1.0,
        # (e) NaN-clean
        "e_nan_clean": nan_clean,
    }
    passes = all(checks.values())
    return checks, passes


# ---------------------------------------------------------------------------
# Bridge resolution
# ---------------------------------------------------------------------------
def resolve_bridge() -> dict[str, Any]:
    """The #180/#99 wall->official multiplier (imported, not re-derived)."""
    calibrated = None
    if projection is not None:
        try:
            calibrated = float(projection.calibrate().multiplier)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[regime] projection.calibrate failed ({exc}); using nominal bridge",
                  flush=True)
    bridge = calibrated if _is_num(calibrated) else BRIDGE_NOMINAL
    return {"bridge": bridge, "bridge_calibrated": calibrated,
            "bridge_nominal_pr180": BRIDGE_NOMINAL}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_report(args, records: list[dict[str, Any]], decode_paths: list[Path],
                 elapsed_s: float | None) -> dict[str, Any]:
    reload_walltps = [r.get("wall_tps") for r in records]
    bridge_info = resolve_bridge()
    vd = variance_decomposition(reload_walltps, bridge_info["bridge"])
    ident = token_identity_across_reloads(decode_paths)
    budget = budget_implication(vd)
    checks, primary_passes = self_test(ident, vd)

    server_ready = [r.get("server_ready_s") for r in records if _is_num(r.get("server_ready_s"))]
    e_accept = [r.get("e_accept_exact") for r in records if _is_num(r.get("e_accept_exact"))]

    report = {
        "pr": 209,
        "lane": "lawine runner / local-measurement (#72/#107/#168/#173/#180/#196)",
        "submission": f"submissions/{args.submission}",
        "kind": "LOCAL fresh-reload re-benchmark variance / frozen-vs-fresh regime pin",
        "no_official_draw": True,
        "no_launch": True,
        "served_files_unchanged": True,
        "adds_tps": 0.0,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "reloads": args.reloads},
        "elapsed_s": elapsed_s,
        "bridge": bridge_info,
        # --- Step 1/3 variance decomposition ---
        "variance": vd,
        # --- Step 2 token identity (determinism leg) ---
        "token_identity": ident,
        # headline fields (flat, for easy reading / wandb)
        "reload_walltps": vd["reload_walltps"],
        "sigma_reload_walltps": vd["sigma_reload_walltps"],
        "cv_reload_pct": vd["cv_reload_pct"],
        "sigma_reload_official": vd["sigma_reload_official"],
        "f_resample_local": vd["f_resample_local"],
        "regime_local": vd["regime_local"],
        "local_harness_is_frozen": vd["local_harness_is_frozen"],
        "token_identity_rate_across_reloads": ident["token_identity_rate_across_reloads"],
        "n_divergent_reloads": ident["n_divergent_reloads"],
        # --- Step 4 budget ---
        "budget_implication": budget,
        # --- Step 5 self-test (PRIMARY) ---
        "self_test": checks,
        "frozen_regime_pin_self_test_passes": primary_passes,
        # diagnostics
        "server_ready_s_mean": statistics.fmean(server_ready) if server_ready else None,
        "e_accept_exact_mean": statistics.fmean(e_accept) if e_accept else None,
        "imported_sources": {
            "kanna_202": "533jd6l1 (FROZEN beats sigma_hw, sigma_draw decomposition, budget bars)",
            "pr_188": "pp1r5orx (sigma_hw=4.864 one-shot hardware std)",
            "pr_196": "y4tavh9p/ekds1cy5 (reload harness, token-identity 1.0)",
            "pr_180": "kbn064b0 (wall->official bridge 1.0602)",
        },
        "git": _git_info(),
    }
    # one-sentence hand-off to kanna #206 + fern #185
    frozen = vd["regime_local"] == "FROZEN"
    report["handoff_sentence"] = (
        f"the LOCAL re-benchmark harness is {vd['regime_local']} "
        f"(sigma_reload={vd['sigma_reload_official']:.3f} official-TPS vs sigma_hw "
        f"{SIGMA_HW} / sigma_draw {SIGMA_DRAW}, token-identity "
        f"{ident['token_identity_rate_across_reloads']:.4f}), so fern's "
        f"conservative mu_bar_frozen_p95={MU_BAR_FROZEN_P95} default is "
        f"{'empirically-confirmed' if frozen else 'loosenable to ' + str(MU_BAR_FRESH_P95)}; "
        f"the OFFICIAL-scorer regime remains kanna #202's human-gated pin, but the "
        f"local result corroborates it under the shared token-identity contract.")
    return report


def _git_info() -> dict[str, Any]:
    try:
        from scripts import wandb_logging
        return wandb_logging.git_info()
    except Exception:
        return {}


def _log_wandb(args, report: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[regime] wandb_logging import failed ({exc}); skipping wandb", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="frozen-regime-local-pin", agent="lawine",
            name=args.wandb_name or "lawine/frozen-regime-local-pin",
            group=args.wandb_group,
            tags=["frozen-regime-local-pin", args.submission, "validity-202", "runner-lane"],
            config={
                "submission": args.submission, "reloads": args.reloads,
                "num_prompts": args.num_prompts, "output_len": args.output_len,
                "seed": args.seed, "bridge": report["bridge"]["bridge"],
                "sigma_hw": SIGMA_HW, "sigma_sample": SIGMA_SAMPLE,
                "sigma_draw": SIGMA_DRAW, "mu_bar_frozen_p95": MU_BAR_FROZEN_P95,
                "mu_bar_fresh_p95": MU_BAR_FRESH_P95,
            },
        )
    except Exception as exc:
        print(f"[regime] wandb init failed ({exc}); skipping wandb", flush=True)
        return
    if run is None:
        print("[regime] wandb disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return
    try:
        for i, rec in enumerate(records):
            wt = rec.get("wall_tps")
            metrics = {
                "reload/wall_tps": wt,
                "reload/wall_tps_official": (wt * report["bridge"]["bridge"]) if _is_num(wt) else None,
                "reload/server_ready_s": rec.get("server_ready_s"),
                "reload/e_accept_exact": rec.get("e_accept_exact"),
                "reload/decode_duration_s": rec.get("decode_duration_s"),
                "reload/sm_clock_mhz_load": (rec.get("clock") or {}).get("sm_clock_mhz_load", {}).get("mean"),
                "reload/temp_c_max": (rec.get("clock") or {}).get("temp_c", {}).get("max"),
            }
            metrics = {k: v for k, v in metrics.items() if _is_num(v)}
            wandb_logging.log_event(run, "reload", step=i, metrics=metrics)
        flat: dict[str, Any] = {}
        for k, v in report.items():
            if isinstance(v, (int, float, bool)):
                flat[f"pin/{k}"] = v
        for k, v in report["variance"].items():
            if isinstance(v, (int, float, bool)):
                flat[f"variance/{k}"] = v
        for k, v in report["token_identity"].items():
            if isinstance(v, (int, float, bool)):
                flat[f"identity/{k}"] = v
        for k, v in report["budget_implication"].items():
            if isinstance(v, (int, float, bool)):
                flat[f"budget/{k}"] = v
        for k, v in report["self_test"].items():
            flat[f"selftest/{k}"] = 1.0 if v else 0.0
        flat["pin/self_test_passes_int"] = 1.0 if report["frozen_regime_pin_self_test_passes"] else 0.0
        flat["pin/regime_local_code"] = {"FROZEN": 0.0, "INTERMEDIATE": 1.0, "FRESH": 2.0}.get(
            report["regime_local"], -1.0)
        wandb_logging.log_summary(run, flat, step=args.reloads)
        wandb_logging.log_json_artifact(
            run, name="frozen_regime_local_pin_report",
            artifact_type="validity-regime-pin", data=report)
    except Exception as exc:
        print(f"[regime] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


def _print_report(report: dict[str, Any]) -> None:
    vd = report["variance"]
    ident = report["token_identity"]
    print(f"\n[regime] ===== FROZEN-vs-FRESH LOCAL REGIME PIN (PR #209) =====", flush=True)
    print(f"  reloads={vd['n_reloads']}  workload={report['workload']['num_prompts']}"
          f"x{report['workload']['output_len']} seed={report['workload']['seed']}", flush=True)
    print(f"  reload_walltps median-ish mean={vd['mean_reload_walltps']:.4f} "
          f"sigma={vd['sigma_reload_walltps']:.5f} CV={vd['cv_reload_pct']:.4f}%", flush=True)
    print(f"  bridge={report['bridge']['bridge']:.5f} -> mean_official="
          f"{vd['mean_reload_official']:.3f}  sigma_reload_official={vd['sigma_reload_official']:.4f} "
          f"CI95={vd['sigma_reload_official_ci95']}", flush=True)
    print(f"  ANCHORS: sigma_hw={SIGMA_HW}  sigma_sample={SIGMA_SAMPLE}  sigma_draw={SIGMA_DRAW}", flush=True)
    print(f"  f_resample_local={vd['f_resample_local']:.4f}  regime_local={vd['regime_local']}  "
          f"local_harness_is_frozen={vd['local_harness_is_frozen']} (CI {vd['local_harness_is_frozen_ci']})", flush=True)
    print(f"  sigma_reload/sigma_hw={vd['sigma_reload_over_sigma_hw']:.4f}  "
          f"reproduces_sigma_hw_twosided={vd['sigma_reload_reproduces_sigma_hw_twosided']}", flush=True)
    print(f"  token_identity_rate_across_reloads={ident['token_identity_rate_across_reloads']:.6f}  "
          f"n_divergent_reloads={ident['n_divergent_reloads']}  completes_128={ident['completes_128']}", flush=True)
    print(f"  PRIMARY frozen_regime_pin_self_test_passes={report['frozen_regime_pin_self_test_passes']} "
          f"checks={report['self_test']}", flush=True)
    print(f"  TEST local_harness_is_frozen={report['local_harness_is_frozen']}", flush=True)
    print(f"\n  >>> {report['budget_implication']['verdict_text']}\n", flush=True)
    print(f"  HANDOFF: {report['handoff_sentence']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default="fa2sw_precache_kenyan")
    ap.add_argument("--reloads", type=int, default=8, help="N fresh LocalServer reloads (>=8)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--clock-interval-ms", type=int, default=250)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--steptime", dest="steptime", action="store_true", default=True)
    ap.add_argument("--no-steptime", dest="steptime", action="store_false")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--analyze-only", action="store_true",
                    help="re-run the analysis from existing records.jsonl + decode/ "
                         "(no re-serve); cheap insurance if wandb/analysis failed")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="frozen-regime-local-pin")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    out_dir = (args.out_dir or OUT_ROOT).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import time as _time
    t0 = _time.time()
    if args.analyze_only:
        records = [json.loads(l) for l in (out_dir / "records.jsonl").read_text().splitlines() if l.strip()]
        decode_paths = sorted((out_dir / "decode").glob("run*.jsonl"))
        if not records or not decode_paths:
            raise SystemExit(f"--analyze-only: no records/decode under {out_dir}")
        elapsed = None
    else:
        for note in paths.prepare_local_gpu_env():
            print(f"[regime] {note}", flush=True)
        records, decode_paths = run_reloads(args, out_dir)
        elapsed = _time.time() - t0
        if len(records) < 2:
            raise SystemExit(f"need >=2 reloads for a variance estimate, got {len(records)}")

    report = build_report(args, records, decode_paths, elapsed)
    report_path = out_dir / "regime_pin_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    _print_report(report)
    print(f"[regime] artifacts -> {report_path}", flush=True)
    _log_wandb(args, report, records)
    return 0 if report["frozen_regime_pin_self_test_passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
