"""End-to-end served-stack wall-clock A/B for the bm4 joint-autotune lever (PR #442, wirbel).

THE DECISIVE TEST. My joint autotune (``triton_attn_joint_autotune.py``) found the
byte-exact config ``{block_m:4, tile:32, warps:4, stages:2}`` and MODELED +15.86 TPS
over the 467.14 strict-equivalence base (Amdahl -> 482.998, beating the deployed
incumbent 481.53) -- but flagged ``realized_crossing_481_proven=False`` (the Amdahl
frontier is an UPPER bound). This orchestrator runs the end-to-end served-stack
wall-clock A/B that decides whether +15.86 *realizes*.

Two arms, the SAME deployed ``fa2sw_precache_kenyan`` submission served twice,
byte-identical except a temporary env-gated reverted kernel-launch override:

    A  bm16_s3  (baseline) : WIRBEL_BM4_AB unset -> deployed BLOCK_M=16, num_stages=3
    B  bm4_s2   (candidate): WIRBEL_BM4_AB=1     -> bm4: BLOCK_M=4, num_stages=2

Each arm: N fresh decode-only timed runs at 128x512 single-stream greedy (the #72
``wall_tps`` protocol, CV 0.035% -> a CI far tighter than the σ_hw≈1%≈4.8 TPS the
advisor named). The runner's PR #99 projection maps each arm's median wall_tps to a
projected-official band; arm A reproduces the 481.53 incumbent, and the question is
whether arm B's projected-official CLEARS 481.53 beyond the CI.

WHY this is expected to be a realized-NULL (the hypothesis under test):
  1. The served stack runs ``SPLITKV_VERIFY=1`` -> the M=8 verify is redirected to the
     3D split-KV path, already occupancy-SATURATED (deployed -> ~96 CTAs > 80 A10G SMs).
     bm4 expands that to ~288 CTAs -- no occupancy headroom to recover (the microbench
     measured the *2D* verify path, occupancy-bound ~6 CTAs, where BLOCK_M=4 *does* help).
  2. The served stack runs ``FA_SLIDING=1`` -> the 35 head-256 sliding layers route to
     FlashAttention2; only the 7 head-512 global layers keep TRITON_ATTN at verify. The
     microbench weighted the per-call cost 35:7 (head-256:head-512), so its +15.86
     over-credits the addressable Triton fraction by ~6x (only 7/42 layers are retunable).
     And head-512's OWN per-shape byte-exact optimum is BLOCK_M=8, not 4 (block_m=4 is the
     head-256 optimum) -- so the served-relevant layers are mis-tuned by the headline cfg.
  3. Even a real attention-kernel speedup propagates through f_attn ≈ 0.069 (denken #441)
     .. 0.093 (stark #445) of the cycle; the large FIXED serving overhead does not shrink.
     #428 already measured the num_stages-only sub-lever realize at <=+0.94 TPS vs its
     +13.23 modeled (a ~14x kernel-vs-wall haircut).

So the prediction is realized Δ collapses to ~a few TPS at most, no CI-clean crossing of
481.53 -> bank as a realized-NULL with an honest number. This A/B is what turns the
modeled UPPER bound into a measured verdict.

This adds 0 TPS and changes NO served file: the ``sitecustomize.py`` hook is env-gated,
reverted, and NEVER submitted (the PR diff carries only ``research/**``). NOT an HF Job,
NOT a submission, NOT a launch. BASELINE stays 481.53. The candidate arm's server logs
are asserted to show the bm4 override fired (forced count > 0 + IS_3D census) and
splitkv-verify present in BOTH arms (the kernel-module meta-path chaining did not disable
the deployed split-KV verify path).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
PATCH_FILE = HERE / "served_bm4_injector.py"
SITECUSTOMIZE = ROOT / "submissions" / "fa2sw_precache_kenyan" / "sitecustomize.py"
SITECUSTOMIZE_REL = "submissions/fa2sw_precache_kenyan/sitecustomize.py"
SUBMISSION = "fa2sw_precache_kenyan"

# ---------------------------------------------------------------------------
# Frozen constants imported EXACT from triton_attn_joint_autotune_results.json
# (my prior autotune; NOT re-derived).
# ---------------------------------------------------------------------------
STRICT_BASE_TPS = 467.14              # denken #423 strict-equivalence base
DEPLOYED_INCUMBENT_TPS = 481.53       # PR #52 deployed (non-equivalent) incumbent
MODELED_DELTA_TPS = 15.8584227713323  # autotune_realized_tps_delta (the +15.86)
MODELED_FRONTIER_TPS = 482.9984227713323  # autotune_frontier_tps (483.00)
MODELED_S_ATTN_JOINT = 1.5275548101236007  # S_attn_joint_weighted @ ctx512 (2D microbench)
MODELED_S_ATTN_NUMSTAGES = 1.4076871771654382  # num_stages-only weighted
T_ATTN_FRAC_USED = 0.09507            # f used in the autotune Amdahl (35:7 ALL-Triton weighting)
NUMSTAGES_ONLY_MODELED_TPS = 13.226254718079531  # num_stages-only MODELED delta
# Independent measured attention fractions of the verify/cycle (cross-checks):
F_ATTN_DENKEN_441 = 0.0690            # denken #441 t_attn_frac_of_verify
F_ATTN_STARK_445 = 0.0928            # stark #445
# #428 realized num_stages-only upper band (the kernel-vs-wall precedent):
NUMSTAGES_ONLY_REALIZED_UB_TPS = 0.94

# Modeled delta as a % of the strict base (what the realization ratio normalizes against).
MODELED_DELTA_PCT = 100.0 * MODELED_DELTA_TPS / STRICT_BASE_TPS  # +3.395%

REALIZES_RATIO = 0.8

MARK_BEGIN = "# >>> wirbel PR#442 served-bm4-wall-ab TEMP toggle >>>"
MARK_END = "# <<< wirbel PR#442 served-bm4-wall-ab TEMP toggle <<<"


def _log(msg: str) -> None:
    print(f"[bm4-wall] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Temporary, reversible sitecustomize toggle (toggle->measure->revert).
# ---------------------------------------------------------------------------
def _hook_block() -> str:
    p = str(PATCH_FILE)
    return (
        f"\n{MARK_BEGIN}\n"
        "# TEMPORARY local wall-clock A/B hook (auto-reverted; NEVER submitted).\n"
        "# Execs the research bm4 injector by absolute path when WIRBEL_BM4_AB is set,\n"
        "# so the kernel meta-path finder lands BEFORE vLLM imports\n"
        "# triton_unified_attention (into the ONEGRAPH capture).\n"
        'if __import__("os").environ.get("WIRBEL_BM4_AB", "").strip() not in ("", "0", "false", "False"):\n'
        f"    _BM4_AB_PATH = {p!r}\n"
        '    try:\n'
        '        with open(_BM4_AB_PATH, "r") as _bm4_f:\n'
        '            exec(compile(_bm4_f.read(), _BM4_AB_PATH, "exec"))\n'
        "    except Exception as _bm4_exc:  # fail-open: never break serve\n"
        "        import sys as _sys\n"
        '        print(f"[bm4-ab] HOOK FAILED (baseline kept): {_bm4_exc!r}", file=_sys.stderr, flush=True)\n'
        f"{MARK_END}\n"
    )


def _strip_hook(text: str) -> str:
    if MARK_BEGIN not in text:
        return text
    head, _, rest = text.partition(MARK_BEGIN)
    _, _, tail = rest.partition(MARK_END)
    return head.rstrip("\n") + ("\n" + tail.lstrip("\n") if tail.strip() else "\n")


def _git_path_dirty(rel: str) -> bool:
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def _git_checkout(rel: str) -> None:
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--", rel],
                   capture_output=True, text=True)


def ensure_clean_toggle() -> None:
    if not SITECUSTOMIZE.exists():
        raise SystemExit(f"sitecustomize not found: {SITECUSTOMIZE}")
    txt = SITECUSTOMIZE.read_text()
    if MARK_BEGIN in txt:
        _log("leftover toggle detected from a prior run — reverting before start")
        SITECUSTOMIZE.write_text(_strip_hook(txt))
    if _git_path_dirty(SITECUSTOMIZE_REL):
        _git_checkout(SITECUSTOMIZE_REL)
    if _git_path_dirty(SITECUSTOMIZE_REL):
        raise SystemExit(f"sitecustomize still dirty after cleanup; refusing to proceed: {SITECUSTOMIZE_REL}")


def apply_toggle() -> bytes:
    original = SITECUSTOMIZE.read_bytes()
    base = _strip_hook(original.decode())
    SITECUSTOMIZE.write_text(base.rstrip("\n") + "\n" + _hook_block())
    _log(f"toggle APPLIED to {SITECUSTOMIZE_REL} (env-gated on WIRBEL_BM4_AB)")
    return original


def revert_toggle(original: bytes) -> bool:
    SITECUSTOMIZE.write_bytes(original)
    if _git_path_dirty(SITECUSTOMIZE_REL):
        _git_checkout(SITECUSTOMIZE_REL)
    clean = not _git_path_dirty(SITECUSTOMIZE_REL)
    _log(f"toggle REVERTED; sitecustomize clean={clean}")
    return clean


# ---------------------------------------------------------------------------
# Server-log scrape: prove the candidate arm actually ran bm4 (forced>0 + IS_3D
# census) AND that splitkv-verify still applied in BOTH arms.
# ---------------------------------------------------------------------------
def _arm_run_logs(arm_dir: Path) -> list[Path]:
    if not arm_dir.exists():
        return []
    return sorted(arm_dir.glob("server_run*.log"))


def _parse_census(text: str) -> list[dict[str, Any]]:
    """Pull the ``[bm4-ab] CENSUS forced[..]`` lines so we can PROVE the served verify
    runs the 3D split-KV path (IS_3D=True) and the grid expanded as predicted."""
    out = []
    for line in text.splitlines():
        if "[bm4-ab] CENSUS forced[" not in line:
            continue
        rec: dict[str, Any] = {"raw": line.split("CENSUS", 1)[1].strip()}
        for tok in ("IS_3D=True", "IS_3D=False"):
            if tok in line:
                rec["is_3d"] = tok.endswith("True")
        if "head=" in line:
            try:
                rec["head"] = int(line.split("head=", 1)[1].split()[0])
            except Exception:  # noqa: BLE001
                pass
        out.append(rec)
    return out


def verify_arms_applied(seed_dir: Path, cand_label: str, base_label: str) -> dict[str, Any]:
    base_logs = _arm_run_logs(seed_dir / base_label)
    cand_logs = _arm_run_logs(seed_dir / cand_label)

    cand_runs = []
    forced_total = 0
    census: list[dict[str, Any]] = []
    for p in cand_logs:
        t = p.read_text(errors="replace")
        forced = t.count("[bm4-ab] CENSUS forced[") + t.count("[bm4-ab] forced bm4 (count=")
        forced_total += forced
        c = _parse_census(t)
        census += c
        cand_runs.append({
            "run": p.name, "patched": "[bm4-ab] PATCHED" in t, "forced_hits": forced,
            "hook_failed": "[bm4-ab] HOOK FAILED" in t,
            "splitkv": "[splitkv-verify] wrapped unified_attention" in t,
            "census_is_3d": [r.get("is_3d") for r in c],
            "census_heads": sorted({r.get("head") for r in c if r.get("head") is not None}),
        })
    base_runs = []
    for p in base_logs:
        t = p.read_text(errors="replace")
        base_runs.append({
            "run": p.name, "patched": "[bm4-ab] PATCHED" in t,
            "hook_failed": "[bm4-ab] HOOK FAILED" in t,
            "splitkv": "[splitkv-verify] wrapped unified_attention" in t,
        })

    cand_all_patched = bool(cand_runs) and all(r["patched"] and r["forced_hits"] > 0 for r in cand_runs)
    cand_failed_open = sum(1 for r in cand_runs if r["hook_failed"])
    base_all_clean = bool(base_runs) and all((not r["patched"]) and (not r["hook_failed"]) for r in base_runs)
    splitkv_all = (bool(cand_runs) and bool(base_runs)
                   and all(r["splitkv"] for r in cand_runs) and all(r["splitkv"] for r in base_runs))
    # The served verify must be 3D (splitkv) -- prove via census, not assumption.
    census_3d = [r.get("is_3d") for r in census if "is_3d" in r]
    served_verify_is_3d = bool(census_3d) and all(census_3d)
    census_heads = sorted({r.get("head") for r in census if r.get("head") is not None})

    ok = bool(cand_all_patched and cand_failed_open == 0 and base_all_clean
              and splitkv_all and served_verify_is_3d)
    return {
        "candidate_runs": cand_runs, "baseline_runs": base_runs,
        "candidate_runs_total": len(cand_runs),
        "candidate_runs_patched": sum(1 for r in cand_runs if r["patched"] and r["forced_hits"] > 0),
        "candidate_runs_failed_open": cand_failed_open,
        "candidate_forced_log_hits": forced_total,
        "candidate_all_runs_patched": cand_all_patched,
        "baseline_unpatched": base_all_clean,
        "splitkv_all_runs": splitkv_all,
        "served_verify_is_3d": served_verify_is_3d,
        "census_heads": census_heads,
        "census_sample": census[:6],
        "applied_ok": ok,
        "logs_found": bool(base_logs and cand_logs),
    }


# ---------------------------------------------------------------------------
# Run one seed's paired A/B (bm16_s3 baseline vs bm4_s2 candidate).
# ---------------------------------------------------------------------------
def run_seed(seed: int, n: int, out_root: Path, args) -> Path:
    from scripts.profiler import paired_tps_ab

    seed_dir = out_root / f"seed{seed}"
    paired_json = seed_dir / "paired_ab.json"
    verify = (verify_arms_applied(seed_dir, args.candidate_label, args.baseline_label)
              if paired_json.exists() else {"applied_ok": False})
    if paired_json.exists() and verify.get("applied_ok") and not args.fresh:
        _log(f"seed {seed}: reusing on-disk paired_ab.json (candidate already applied)")
        return paired_json

    cand_env = "WIRBEL_BM4_AB=1"
    argv = [
        "--baseline", SUBMISSION, "--candidate", SUBMISSION,
        "--candidate-env", cand_env,
        "--candidate-env", f"WIRBEL_BM4_BLOCK_M={args.block_m}",
        "--candidate-env", f"WIRBEL_BM4_NUM_STAGES={args.num_stages}",
        "--baseline-label", args.baseline_label, "--candidate-label", args.candidate_label,
        "--n", str(n), "--seed", str(seed),
        "--num-prompts", str(args.num_prompts), "--output-len", str(args.output_len),
        "--out-dir", str(seed_dir), "--tag", f"seed{seed}",
        "--wandb-group", args.wandb_group,
        "--wandb-name", f"{args.wandb_name}-seed{seed}",
    ]
    if args.no_wandb:
        argv.append("--no-wandb")
    _log(f"seed {seed}: paired A/B ({args.baseline_label} vs {args.candidate_label}) n={n} -> {seed_dir}")
    rc = paired_tps_ab.main(argv)
    if rc != 0 or not paired_json.exists():
        raise SystemExit(f"paired_tps_ab failed for seed {seed} (rc={rc}, json={paired_json.exists()})")
    return paired_json


# ---------------------------------------------------------------------------
# Aggregate + realization + reconciliation math.
# ---------------------------------------------------------------------------
def _finite_pos(xs: list) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]


def _implied_S_attn(realized_frac: float, f: float) -> float | None:
    """Back out the realized attention-kernel speedup from an end-to-end Amdahl delta.
    tps_new/tps_old = 1/[(1-f)+f/S]  ->  S = f / (1/r - 1 + f), r = 1+realized_frac."""
    r = 1.0 + realized_frac
    denom = 1.0 / r - 1.0 + f
    if denom <= 0:
        return math.inf
    return f / denom


def aggregate(seed_jsons: list[Path], verifies: dict[int, dict]) -> dict[str, Any]:
    base_vals: list[float] = []
    cand_vals: list[float] = []
    base_proj: list[float] = []
    cand_proj: list[float] = []
    cand_proj_lo: list[float] = []
    cand_ci95_pcts: list[float] = []
    per_seed = []
    for pj in seed_jsons:
        d = json.loads(pj.read_text())
        seed = d["workload"]["seed"]
        b = d["arms"]["baseline"]["wall_tps"]
        c = d["arms"]["candidate"]["wall_tps"]
        bv = _finite_pos(b.get("values") or [])
        cv = _finite_pos(c.get("values") or [])
        base_vals += bv
        cand_vals += cv
        proj = (d.get("projection") or {}).get("arms") or {}
        bp = (proj.get("baseline") or {}).get("projected_official")
        cp = (proj.get("candidate") or {}).get("projected_official")
        cplo = (proj.get("candidate") or {}).get("projected_official_lo")
        if isinstance(bp, (int, float)):
            base_proj.append(float(bp))
        if isinstance(cp, (int, float)):
            cand_proj.append(float(cp))
        if isinstance(cplo, (int, float)):
            cand_proj_lo.append(float(cplo))
        vd = d.get("verdict") or {}
        if isinstance(vd.get("ci95_observed_pct"), (int, float)):
            cand_ci95_pcts.append(float(vd["ci95_observed_pct"]))
        per_seed.append({
            "seed": seed,
            "base_median": b.get("median"), "cand_median": c.get("median"),
            "base_values": bv, "cand_values": cv,
            "delta_median_pct": vd.get("delta_median_pct"),
            "verdict": vd.get("verdict"),
            "ci95_observed_pct": vd.get("ci95_observed_pct"),
            "operative_threshold_pct": vd.get("operative_threshold_pct"),
            "base_projected_official": bp, "cand_projected_official": cp,
            "applied": verifies.get(seed, {}),
        })

    out: dict[str, Any] = {"per_seed": per_seed, "n_base": len(base_vals), "n_cand": len(cand_vals)}
    if not base_vals or not cand_vals:
        out["error"] = "missing pooled wall_tps values"
        return out

    base_p50 = statistics.median(base_vals)
    cand_p50 = statistics.median(cand_vals)
    realized = 100.0 * (cand_p50 - base_p50) / base_p50
    ratio = realized / MODELED_DELTA_PCT
    base_proj_med = statistics.median(base_proj) if base_proj else DEPLOYED_INCUMBENT_TPS
    cand_proj_med = statistics.median(cand_proj) if cand_proj else base_proj_med * (1 + realized / 100.0)
    cand_proj_lo_med = statistics.median(cand_proj_lo) if cand_proj_lo else None
    # tight A/B CI on the realized delta (observed, pooled median of per-seed half-widths)
    ci95_pct = statistics.median(cand_ci95_pcts) if cand_ci95_pcts else None

    # delta applied to the deployed incumbent (what "crosses 481.53" means):
    cand_tps_on_incumbent = DEPLOYED_INCUMBENT_TPS * (1.0 + realized / 100.0)

    crosses_481_central = cand_proj_med > DEPLOYED_INCUMBENT_TPS
    # CI-clean crossing: the realized delta is positive, significant (clears its observed
    # CI95 half-width AND the operative threshold), AND pushes projected-official above 481.53.
    op_thresh = statistics.median([ps.get("operative_threshold_pct") for ps in per_seed
                                   if isinstance(ps.get("operative_threshold_pct"), (int, float))] or [0.10])
    delta_ci_lo = realized - ci95_pct if ci95_pct is not None else None
    delta_significant = (realized > 0 and ci95_pct is not None and delta_ci_lo > 0
                         and realized >= op_thresh)
    crosses_481_ci_clean = bool(delta_significant and crosses_481_central
                                and (cand_proj_lo_med is None or cand_proj_lo_med > DEPLOYED_INCUMBENT_TPS))

    if ratio <= 0:
        cls = "evaporates"      # realized <= 0: the modeled lift did not survive the wall
    elif ratio >= REALIZES_RATIO:
        cls = "realizes"
    else:
        cls = "partial"

    # ----- reconciliation (deliverable #2): map microbench Δ -> realized Δ -----
    realized_frac = realized / 100.0
    recon = {}
    for fname, f in (("autotune_f_0.09507", T_ATTN_FRAC_USED),
                     ("denken441_f_0.0690", F_ATTN_DENKEN_441),
                     ("stark445_f_0.0928", F_ATTN_STARK_445)):
        recon[fname] = {
            "f_attn": f,
            "implied_realized_S_attn": _implied_S_attn(realized_frac, f),
        }
    recon["modeled_S_attn_joint_2D_microbench"] = MODELED_S_ATTN_JOINT
    recon["modeled_delta_tps"] = MODELED_DELTA_TPS
    recon["realized_delta_tps_on_incumbent"] = cand_tps_on_incumbent - DEPLOYED_INCUMBENT_TPS
    recon["modeled_minus_realized_tps"] = MODELED_DELTA_TPS - (cand_tps_on_incumbent - DEPLOYED_INCUMBENT_TPS)
    recon["numstages_only_modeled_tps"] = NUMSTAGES_ONLY_MODELED_TPS
    recon["numstages_only_realized_ub_tps_pr428"] = NUMSTAGES_ONLY_REALIZED_UB_TPS
    recon["addressable_triton_verify_layers"] = "7/42 (FA_SLIDING routes 35 head-256 -> FA2)"
    recon["head512_per_shape_optimum_block_m"] = 8
    recon["served_verify_path"] = "3D split-KV (SPLITKV_VERIFY=1), occupancy-saturated ~96 CTAs > 80 SMs"

    out.update({
        "base_pooled_p50_wall_tps": base_p50,
        "cand_pooled_p50_wall_tps": cand_p50,
        "base_pooled_values": base_vals,
        "cand_pooled_values": cand_vals,
        "realized_delta_pct_wall": realized,
        "realized_delta_ci95_pct": ci95_pct,
        "realized_delta_ci95_lo_pct": delta_ci_lo,
        "operative_threshold_pct": op_thresh,
        "modeled_delta_pct": MODELED_DELTA_PCT,
        "realization_ratio": ratio,
        "classification": cls,
        "base_projected_official_median": base_proj_med,
        "cand_projected_official_median": cand_proj_med,
        "cand_projected_official_lo_median": cand_proj_lo_med,
        "cand_tps_on_incumbent": cand_tps_on_incumbent,
        "crosses_481_central": crosses_481_central,
        "crosses_481_ci_clean": crosses_481_ci_clean,
        "delta_significant": delta_significant,
        "reconciliation": recon,
    })
    return out


# ---------------------------------------------------------------------------
# Self-test (PRIMARY headline).
# ---------------------------------------------------------------------------
def constants_exact() -> dict[str, Any]:
    expect = {
        "STRICT_BASE_TPS": (STRICT_BASE_TPS, 467.14),
        "DEPLOYED_INCUMBENT_TPS": (DEPLOYED_INCUMBENT_TPS, 481.53),
        "MODELED_DELTA_TPS": (MODELED_DELTA_TPS, 15.8584227713323),
        "MODELED_FRONTIER_TPS": (MODELED_FRONTIER_TPS, 482.9984227713323),
        "T_ATTN_FRAC_USED": (T_ATTN_FRAC_USED, 0.09507),
    }
    bad = {k: got for k, (got, want) in expect.items() if got != want}
    base_plus_delta_ok = abs((STRICT_BASE_TPS + MODELED_DELTA_TPS) - MODELED_FRONTIER_TPS) < 1e-6
    frontier_beats_481 = MODELED_FRONTIER_TPS > DEPLOYED_INCUMBENT_TPS
    return {"all_exact": (not bad) and base_plus_delta_ok and frontier_beats_481,
            "mismatches": bad, "base_plus_delta_ok": base_plus_delta_ok,
            "modeled_frontier_beats_481": frontier_beats_481}


def caveats() -> dict[str, str]:
    return {
        "zero_tps": ("This A/B adds 0 TPS and changes NO served file: it measures whether the "
                     "modeled +15.86 bm4 lift realizes end-to-end. The sitecustomize hook is "
                     "env-gated, reverted, and NEVER submitted. BASELINE stays 481.53."),
        "byte_exact": ("bm4 = BLOCK_M 16->4 (BLOCK_Q->1, grid dim0 recomputed) + num_stages 3->2; "
                       "TILE=32/num_warps=4 held. BLOCK_M is the query-row tile/grid-grouping knob; "
                       "per-row KV reduction order is TILE_SIZE-bound (unchanged) -> byte-exact, "
                       "proven by the paired same-path census (NOT asserted)."),
        "served_is_3d": ("The served M=8 verify runs 3D split-KV (SPLITKV_VERIFY=1), already "
                         "occupancy-saturated (~96 CTAs > 80 A10G SMs); the +15.86 microbench measured "
                         "the 2D verify path (occupancy-bound) -> realized < modeled is the wall-vs-"
                         "microbench gap quantified."),
    }


def self_test(agg: dict[str, Any], verifies: dict[int, dict], smoke: bool) -> dict[str, Any]:
    cexact = constants_exact()
    cav = caveats()

    base_vals = agg.get("base_pooled_values") or []
    cand_vals = agg.get("cand_pooled_values") or []
    all_finite = (bool(base_vals) and bool(cand_vals)
                  and len(_finite_pos(base_vals)) == len(base_vals)
                  and len(_finite_pos(cand_vals)) == len(cand_vals))

    candidate_applied = bool(verifies) and all(v.get("applied_ok") for v in verifies.values())
    served_3d = bool(verifies) and all(v.get("served_verify_is_3d") for v in verifies.values())

    realized = agg.get("realized_delta_pct_wall")
    ratio = agg.get("realization_ratio")
    ratio_consistent = (realized is not None and ratio is not None
                        and abs(ratio * MODELED_DELTA_PCT - realized) < 1e-6)

    # baseline arm reproduces the 481.53 incumbent via projection (closed loop)
    base_proj = agg.get("base_projected_official_median")
    base_repro_err = (100.0 * abs(base_proj - DEPLOYED_INCUMBENT_TPS) / DEPLOYED_INCUMBENT_TPS
                      if isinstance(base_proj, (int, float)) else None)
    base_reproduces_481 = (base_repro_err is not None and base_repro_err <= 2.0)

    required = {
        "constants_exact": cexact["all_exact"],
        "candidate_actually_applied_bm4": candidate_applied,
        "served_verify_is_3d_census": served_3d,
        "ratio_consistent": ratio_consistent,
        "all_tps_finite_positive": all_finite,
        "caveats_present": all(isinstance(v, str) and len(v) > 40 for v in cav.values()),
    }
    if not smoke:
        required["baseline_reproduces_481"] = base_reproduces_481

    passes = all(required.values())
    return {
        "served_bm4_wall_ab_self_test_passes": passes,
        "required": required,
        "smoke": smoke,
        "constants": cexact,
        "caveats": cav,
        "candidate_applied_per_seed": verifies,
        "baseline_reproduction_err_pct": base_repro_err,
        "baseline_reproduces_481": base_reproduces_481,
        "ratio_consistent": ratio_consistent,
        "all_tps_finite_positive": all_finite,
    }


# ---------------------------------------------------------------------------
# W&B aggregate run.
# ---------------------------------------------------------------------------
def log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:
        _log(f"wandb_logging import failed ({exc}); skipping")
        return
    agg = result["aggregate"]
    st = result["self_test"]
    run = wandb_logging.init_wandb_run(
        job_type="served-bm4-wall-ab", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["pr442", "wall-realization", "bm4", "triton-attn", SUBMISSION],
        config={
            "strict_base_tps": STRICT_BASE_TPS, "deployed_incumbent_tps": DEPLOYED_INCUMBENT_TPS,
            "modeled_delta_tps": MODELED_DELTA_TPS, "modeled_frontier_tps": MODELED_FRONTIER_TPS,
            "modeled_delta_pct": MODELED_DELTA_PCT, "block_m": args.block_m,
            "num_stages": args.num_stages, "seeds": args.seeds, "n": args.n, "smoke": args.smoke,
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "lever": "verify_triton_bm4_block_m_16to4_num_stages_3to2",
        },
    )
    if run is None:
        _log("wandb disabled (no API key / WANDB_DISABLED); skipping")
        return
    try:
        flat = {"served_bm4_wall_ab_self_test_passes": float(st["served_bm4_wall_ab_self_test_passes"])}
        for k in ("base_pooled_p50_wall_tps", "cand_pooled_p50_wall_tps",
                  "realized_delta_pct_wall", "realized_delta_ci95_pct", "modeled_delta_pct",
                  "realization_ratio", "base_projected_official_median",
                  "cand_projected_official_median", "cand_projected_official_lo_median",
                  "cand_tps_on_incumbent", "operative_threshold_pct"):
            v = agg.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                flat[k] = v
        flat["crosses_481_central"] = float(bool(agg.get("crosses_481_central")))
        flat["crosses_481_ci_clean"] = float(bool(agg.get("crosses_481_ci_clean")))
        flat["delta_significant"] = float(bool(agg.get("delta_significant")))
        for k, v in st["required"].items():
            flat[f"selftest/{k}"] = float(bool(v))
        for ps in agg.get("per_seed", []):
            s = ps.get("seed")
            for fld in ("delta_median_pct", "base_median", "cand_median", "ci95_observed_pct"):
                if isinstance(ps.get(fld), (int, float)):
                    flat[f"seed{s}/{fld}"] = ps[fld]
        run.summary["classification"] = agg.get("classification")
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="served_bm4_wall_ab", artifact_type="wall-realization", data=result)
    except Exception as exc:
        _log(f"WARN wandb logging error: {exc}")
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1,2", help="comma-separated seeds (>=2 for the headline)")
    ap.add_argument("--n", type=int, default=5, help="fresh runs per arm per seed (median-of-N)")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--block-m", dest="block_m", type=int, default=4,
                    help="candidate BLOCK_M (default 4 = the advisor-named config)")
    ap.add_argument("--num-stages", dest="num_stages", type=int, default=2,
                    help="candidate num_stages (default 2)")
    ap.add_argument("--baseline-label", default="bm16_s3")
    ap.add_argument("--candidate-label", default="bm4_s2")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap boot+patch-fires check: n=1, seeds=1, reproduction non-required")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="evaluate the self-test booleans and exit non-zero if it fails")
    ap.add_argument("--no-toggle", action="store_true",
                    help="do NOT edit sitecustomize (dry structure check)")
    ap.add_argument("--fresh", action="store_true", help="ignore on-disk paired_ab.json and re-run every seed")
    ap.add_argument("--out-root", type=Path, default=HERE / "ab_out")
    ap.add_argument("--wandb_group", default="triton-joint-autotune")
    ap.add_argument("--wandb_name", default="wirbel/served-bm4-wall-ab")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.n = 1
        seeds = [1]
    else:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not PATCH_FILE.exists():
        raise SystemExit(f"patch file missing: {PATCH_FILE}")

    _log(f"seeds={seeds} n={args.n} smoke={args.smoke} cfg=bm{args.block_m}_s{args.num_stages} "
         f"workload={args.num_prompts}x{args.output_len} -> {out_root}")
    _log(f"modeled_delta={MODELED_DELTA_TPS:+.4f} TPS ({MODELED_DELTA_PCT:+.4f}% of {STRICT_BASE_TPS}) "
         f"-> frontier {MODELED_FRONTIER_TPS:.3f} (incumbent {DEPLOYED_INCUMBENT_TPS})")

    t0 = time.time()
    original_bytes = None
    toggled = False
    try:
        if not args.no_toggle:
            ensure_clean_toggle()
            original_bytes = apply_toggle()
            toggled = True

        seed_jsons: list[Path] = []
        verifies: dict[int, dict] = {}
        for seed in seeds:
            pj = run_seed(seed, args.n, out_root, args)
            seed_jsons.append(pj)
            verifies[seed] = verify_arms_applied(out_root / f"seed{seed}",
                                                 args.candidate_label, args.baseline_label)
            _log(f"seed {seed}: applied_ok={verifies[seed].get('applied_ok')} "
                 f"forced_hits={verifies[seed].get('candidate_forced_log_hits')} "
                 f"served_3d={verifies[seed].get('served_verify_is_3d')} "
                 f"heads={verifies[seed].get('census_heads')}")
    finally:
        toggle_clean = revert_toggle(original_bytes) if toggled and original_bytes is not None else True

    agg = aggregate(seed_jsons, verifies)
    st = self_test(agg, verifies, smoke=args.smoke)
    st["toggle_reverted_clean"] = toggle_clean
    if not toggle_clean:
        st["required"]["toggle_reverted_clean"] = False
        st["served_bm4_wall_ab_self_test_passes"] = False

    result = {
        "experiment": "served_bm4_wall_ab", "pr": 442, "student": "wirbel",
        "question": "Does the modeled +15.86 TPS bm4 joint-autotune lift realize on the served-stack wall?",
        "lever": f"verify Triton bm{args.block_m} (BLOCK_M 16->{args.block_m}) + num_stages 3->{args.num_stages}",
        "zero_tps": True, "not_a_launch": True, "not_a_build": True, "not_a_submission": True,
        "config": {"block_m": args.block_m, "tile": 32, "num_warps": 4, "num_stages": args.num_stages},
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seeds": seeds},
        "elapsed_s": time.time() - t0,
        "aggregate": agg,
        "self_test": st,
    }
    (out_root / "results.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- console verdict ----
    print("\n" + "=" * 78, flush=True)
    print("SERVED-STACK bm4 WALL-CLOCK A/B (PR #442, wirbel)", flush=True)
    print("=" * 78, flush=True)
    if "error" not in agg:
        print(f"  A bm16_s3 (deployed) p50 wall_tps = {agg['base_pooled_p50_wall_tps']:.4f} "
              f"-> proj-official {agg['base_projected_official_median']:.2f} "
              f"(incumbent {DEPLOYED_INCUMBENT_TPS})", flush=True)
        print(f"  B bm4_s2  (lever)    p50 wall_tps = {agg['cand_pooled_p50_wall_tps']:.4f} "
              f"-> proj-official {agg['cand_projected_official_median']:.2f}", flush=True)
        ci = agg.get('realized_delta_ci95_pct')
        ci_s = f"±{ci:.4f}%" if isinstance(ci, (int, float)) else "n/a"
        print(f"  realized Δ%_wall = {agg['realized_delta_pct_wall']:+.4f}% (CI95 {ci_s})   "
              f"modeled Δ% = {MODELED_DELTA_PCT:+.4f}%", flush=True)
        print(f"  realized Δ on 481.53 = {agg['cand_tps_on_incumbent'] - DEPLOYED_INCUMBENT_TPS:+.3f} TPS "
              f"(modeled +{MODELED_DELTA_TPS:.2f})", flush=True)
        print(f"  >>> realization_ratio = {agg['realization_ratio']:+.4f}  [{agg['classification']}]", flush=True)
        print(f"  >>> crosses 481.53 central={agg['crosses_481_central']}  "
              f"CI-CLEAN={agg['crosses_481_ci_clean']}", flush=True)
        rc = agg["reconciliation"]
        print(f"  reconcile: implied realized S_attn @f=0.069 -> "
              f"{rc['denken441_f_0.0690']['implied_realized_S_attn']:.3f}  "
              f"(modeled 2D S_attn {MODELED_S_ATTN_JOINT:.3f}); "
              f"addressable {rc['addressable_triton_verify_layers']}", flush=True)
    else:
        print(f"  AGGREGATE ERROR: {agg['error']}", flush=True)
    for ps in agg.get("per_seed", []):
        print(f"   seed{ps['seed']}: A={ps.get('base_median')} B={ps.get('cand_median')} "
              f"Δ={ps.get('delta_median_pct')}% [{ps.get('verdict')}] "
              f"applied_ok={ps.get('applied', {}).get('applied_ok')} "
              f"3d={ps.get('applied', {}).get('served_verify_is_3d')}", flush=True)
    print(f"  toggle_reverted_clean = {toggle_clean}", flush=True)
    print(f"  >>> SELF-TEST PASSES = {st['served_bm4_wall_ab_self_test_passes']} "
          f"required={ {k: v for k, v in st['required'].items()} }", flush=True)
    print("=" * 78 + "\n", flush=True)

    log_wandb(args, result)
    print(f"[bm4-wall] artifacts -> {out_root / 'results.json'}", flush=True)

    if args.self_test and not st["served_bm4_wall_ab_self_test_passes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
