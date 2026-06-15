"""Free-ceiling wall-clock realization orchestrator (PR #298, stark).

Does the banked **487.7 free step ceiling** actually realize on a direct local
host-to-host **wall-clock** A/B? The whole programme treats 487.7 TPS as the
closed "free step ceiling" (wirbel #285 lossless envelope 487.729; denken #291
kernel-event floor 487.7289), but that number is **analytically-composed /
kernel-µs**. The launch gate is a MEASURED ≥500 wall TPS, and the composed
+1.287% (481.53→487.729) has never been measured on the wall — the exact thing my
#273 method exists to check, and the exact thing static-K *failed* (realization
ratio −2.02).

CRUX: apply the banked greedy-safe lossless lever — the verify SDPA ``num_stages``
3→2 tune (wirbel #285/#279 isolated, denken #291 priced) — on the deployed stack
locally via a **temporary, env-gated, reverted** toggle, re-run the #273 paired
wall A/B (``scripts/profiler/paired_tps_ab.py``, 128×512 single-stream greedy,
≥2 seeds, p50), and compute::

    realized_delta_pct_wall = 100·(s2_p50 − s3_p50)/s3_p50
    realization_ratio_487   = realized_delta_pct_wall / composed_delta_pct (+1.287%)

classify ``realizes`` (ratio≥0.8) / ``partial`` (0<ratio<0.8) / ``over_credits``
(ratio≤0, the static-K class).

**This leg adds 0 TPS.** It wall-audits the banked free ceiling. It does NOT
produce a ≥500 build, does NOT change the served checkpoint, is NOT an HF Job, is
NOT a submission, NOT a launch, NOT a build, NOT open2. BASELINE stays 481.53.
The launch gate stays land #245's MEASURED ≥500 at λ̂≥0.9780 AND PPL≤2.42,
human-approval-gated. The ``sitecustomize.py`` toggle is reverted; the PR diff
carries only ``research/**``.

The lever is NOT a serve-time env toggle (the deployed served ``unified_attention``
launches ``kernel_unified_attention`` as a bare ``@triton.jit`` with triton-default
``num_stages=3``). To wall-measure it we inject ``num_stages=2`` at the kernel
launch via a temporary env-gated patch (``sdpa_num_stages_ab.py``) loaded by an
appended hook in the submission's own ``sitecustomize.py`` (toggle→measure→revert).
With ``SDPA_NUM_STAGES_AB`` unset the deployed stack is byte-identical (the #273
reproduction). The candidate arm's server logs are asserted to show the forced
``num_stages=2`` count > 0 (guards against a silent no-op that would fake an
``over_credits`` verdict), and splitkv-verify is asserted present in BOTH arms
(proves the kernel-module meta-path chaining did not disable the deployed
split-KV verify path).
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
PATCH_FILE = HERE / "sdpa_num_stages_ab.py"
SITECUSTOMIZE = ROOT / "submissions" / "fa2sw_precache_kenyan" / "sitecustomize.py"
SITECUSTOMIZE_REL = "submissions/fa2sw_precache_kenyan/sitecustomize.py"

# ---------------------------------------------------------------------------
# Imported EXACT (NOT re-derived) — frozen tripwire constants.
#   wirbel #285 envelope / denken #291 floor / kanna #286 basis / step bookkeeping.
# ---------------------------------------------------------------------------
FRONTIER_TPS = 481.53                      # deployed official frontier
ENVELOPE_TPS = 487.72885498477575          # wirbel #285 lossless micro-lever envelope
DENKEN_FLOOR_TPS = 487.7289                 # denken #291 kernel-event floor (rounded form)
KANNA_BASIS_TPS = 493.64                    # kanna #286 basis-honest
STEP_US = 1218.2                            # deployed per-step µs
NEW_STEP_US = 1202.7171244939168           # num_stages=2 per-step µs (wirbel #285)
BRIDGE_0278 = 0.2147                        # denken #278 bridge
K_CAL = 125.268                            # calibration constant
E_T = 3.844                                # E[T] accept length (== composition_e_t_k7 3.8444…)
SDPA_FULL_SAVING_US = 15.482875            # wirbel #285 standalone SDPA saving

# composed delta the realization ratio normalizes against: +1.2873247741%.
COMPOSED_DELTA_PCT = 100.0 * (ENVELOPE_TPS - FRONTIER_TPS) / FRONTIER_TPS
COMPOSED_DELTA_PCT_JSON = 1.2873247741     # wirbel #285 lossless_micro_lever_envelope.json

# #273 deployed-K7 wall reference (static_k_wallclock_ab report.json k7_baseline).
DEPLOYED_K7_REF_WALL_TPS = 453.6177679392844
# #273 static-K precedent (51bdsbpw): K4-vs-K7 = −8.629%, ratio −2.018.
STATIC_K_PRECEDENT_RATIO = -2.018

REALIZES_RATIO = 0.8                        # ratio≥0.8 → realizes
SUBMISSION = "fa2sw_precache_kenyan"

# Toggle markers (so a leftover hook from a killed run is detectable + strippable).
MARK_BEGIN = "# >>> stark PR#298 free-ceiling-wallclock-realize TEMP toggle >>>"
MARK_END = "# <<< stark PR#298 free-ceiling-wallclock-realize TEMP toggle <<<"


def _log(msg: str) -> None:
    print(f"[free-ceiling] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Temporary, reversible sitecustomize toggle (toggle→measure→revert).
# ---------------------------------------------------------------------------
def _hook_block() -> str:
    p = str(PATCH_FILE)
    return (
        f"\n{MARK_BEGIN}\n"
        "# TEMPORARY local wall-clock A/B hook (auto-reverted; NEVER submitted).\n"
        "# Execs the research num_stages injector by absolute path when\n"
        "# SDPA_NUM_STAGES_AB is set, so the kernel meta-path finder lands BEFORE\n"
        "# vLLM imports triton_unified_attention (into the ONEGRAPH capture).\n"
        'if __import__("os").environ.get("SDPA_NUM_STAGES_AB", "").strip():\n'
        f"    _SDPA_AB_PATH = {p!r}\n"
        '    try:\n'
        '        with open(_SDPA_AB_PATH, "r") as _sdpa_f:\n'
        '            exec(compile(_sdpa_f.read(), _SDPA_AB_PATH, "exec"))\n'
        "    except Exception as _sdpa_exc:  # fail-open: never break serve\n"
        "        import sys as _sys\n"
        '        print(f"[sdpa-ab] HOOK FAILED (baseline kept): {_sdpa_exc!r}", file=_sys.stderr, flush=True)\n'
        f"{MARK_END}\n"
    )


def _strip_hook(text: str) -> str:
    if MARK_BEGIN not in text:
        return text
    head, _, rest = text.partition(MARK_BEGIN)
    _, _, tail = rest.partition(MARK_END)
    # also drop the single leading newline we added before MARK_BEGIN
    return head.rstrip("\n") + ("\n" + tail.lstrip("\n") if tail.strip() else "\n")


def _git_path_dirty(rel: str) -> bool:
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def _git_checkout(rel: str) -> None:
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--", rel],
                   capture_output=True, text=True)


def ensure_clean_toggle() -> None:
    """Recover from a prior hard-killed run: if sitecustomize carries our marker,
    strip it and git-checkout so we start byte-identical."""
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
    _log(f"toggle APPLIED to {SITECUSTOMIZE_REL} (env-gated on SDPA_NUM_STAGES_AB)")
    return original


def revert_toggle(original: bytes) -> bool:
    SITECUSTOMIZE.write_bytes(original)
    if _git_path_dirty(SITECUSTOMIZE_REL):
        _git_checkout(SITECUSTOMIZE_REL)
    clean = not _git_path_dirty(SITECUSTOMIZE_REL)
    _log(f"toggle REVERTED; sitecustomize clean={clean}")
    return clean


# ---------------------------------------------------------------------------
# Server-log scrape: prove the candidate arm actually ran num_stages=2 AND that
# splitkv-verify still applied in BOTH arms (kernel-module finder chaining OK).
# ---------------------------------------------------------------------------
def _read_logs(arm_dir: Path) -> str:
    if not arm_dir.exists():
        return ""
    return "\n".join(p.read_text(errors="replace") for p in sorted(arm_dir.glob("server_run*.log")))


def verify_arms_applied(seed_dir: Path) -> dict[str, Any]:
    s3 = _read_logs(seed_dir / "sdpa_s3")
    s2 = _read_logs(seed_dir / "sdpa_s2")
    s2_patched = "[sdpa-ab] PATCHED" in s2
    s2_forced = s2.count("forced num_stages=2")
    s3_clean = "[sdpa-ab] PATCHED" not in s3  # baseline must NOT carry the patch
    splitkv_s3 = "[splitkv-verify] wrapped unified_attention" in s3
    splitkv_s2 = "[splitkv-verify] wrapped unified_attention" in s2
    ok = bool(s2_patched and s2_forced > 0 and s3_clean and splitkv_s3 and splitkv_s2)
    return {
        "s2_patched": s2_patched,
        "s2_forced_log_hits": s2_forced,
        "s3_baseline_unpatched": s3_clean,
        "splitkv_in_s3": splitkv_s3,
        "splitkv_in_s2": splitkv_s2,
        "applied_ok": ok,
        "logs_found": bool(s3 and s2),
    }


# ---------------------------------------------------------------------------
# Run one seed's paired A/B (s3 baseline vs s2 candidate) via the #273 runner.
# ---------------------------------------------------------------------------
def run_seed(seed: int, n: int, out_root: Path, args) -> Path:
    from scripts.profiler import paired_tps_ab

    seed_dir = out_root / f"seed{seed}"
    paired_json = seed_dir / "paired_ab.json"
    verify = verify_arms_applied(seed_dir) if paired_json.exists() else {"applied_ok": False}
    if paired_json.exists() and verify.get("applied_ok") and not args.fresh:
        _log(f"seed {seed}: reusing on-disk paired_ab.json (candidate already applied)")
        return paired_json

    argv = [
        "--baseline", SUBMISSION, "--candidate", SUBMISSION,
        "--candidate-env", "SDPA_NUM_STAGES_AB=2",
        "--baseline-label", "sdpa_s3", "--candidate-label", "sdpa_s2",
        "--n", str(n), "--seed", str(seed),
        "--num-prompts", str(args.num_prompts), "--output-len", str(args.output_len),
        "--out-dir", str(seed_dir), "--tag", f"seed{seed}",
        "--reference-wall-tps", repr(DEPLOYED_K7_REF_WALL_TPS),
        "--wandb-group", args.wandb_group,
        "--wandb-name", f"{args.wandb_name}-seed{seed}",
    ]
    if args.no_wandb:
        argv.append("--no-wandb")
    _log(f"seed {seed}: paired A/B (s3 vs s2) n={n} -> {seed_dir}")
    rc = paired_tps_ab.main(argv)
    if rc != 0 or not paired_json.exists():
        raise SystemExit(f"paired_tps_ab failed for seed {seed} (rc={rc}, json={paired_json.exists()})")
    return paired_json


# ---------------------------------------------------------------------------
# Aggregate + realization math.
# ---------------------------------------------------------------------------
def _finite_pos(xs: list) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]


def aggregate(seed_jsons: list[Path], verifies: dict[int, dict]) -> dict[str, Any]:
    s3_vals: list[float] = []
    s2_vals: list[float] = []
    s3_proj: list[float] = []
    per_seed = []
    for pj in seed_jsons:
        d = json.loads(pj.read_text())
        seed = d["workload"]["seed"]
        b = d["arms"]["baseline"]["wall_tps"]
        c = d["arms"]["candidate"]["wall_tps"]
        bv = _finite_pos(b.get("values") or [])
        cv = _finite_pos(c.get("values") or [])
        s3_vals += bv
        s2_vals += cv
        proj = (((d.get("projection") or {}).get("arms") or {}).get("baseline") or {})
        po = proj.get("projected_official")
        if isinstance(po, (int, float)) and math.isfinite(po):
            s3_proj.append(float(po))
        per_seed.append({
            "seed": seed,
            "s3_median": b.get("median"), "s2_median": c.get("median"),
            "s3_values": bv, "s2_values": cv,
            "delta_median_pct": (d.get("verdict") or {}).get("delta_median_pct"),
            "verdict": (d.get("verdict") or {}).get("verdict"),
            "s3_projected_official": po,
            "s3_reproduces_k7": proj.get("reproduces_reference"),
            "applied": verifies.get(seed, {}),
        })

    out: dict[str, Any] = {"per_seed": per_seed, "n_s3": len(s3_vals), "n_s2": len(s2_vals)}
    if not s3_vals or not s2_vals:
        out["error"] = "missing pooled wall_tps values"
        return out

    s3_p50 = statistics.median(s3_vals)
    s2_p50 = statistics.median(s2_vals)
    realized = 100.0 * (s2_p50 - s3_p50) / s3_p50
    ratio = realized / COMPOSED_DELTA_PCT
    s3_proj_med = statistics.median(s3_proj) if s3_proj else FRONTIER_TPS
    measured_ceiling = s3_proj_med * (1.0 + realized / 100.0)
    measured_ceiling_frontier = FRONTIER_TPS * (1.0 + realized / 100.0)

    if ratio <= 0:
        cls = "over_credits"
    elif ratio >= REALIZES_RATIO:
        cls = "realizes"
    else:
        cls = "partial"

    out.update({
        "s3_pooled_p50_wall_tps": s3_p50,
        "s2_pooled_p50_wall_tps": s2_p50,
        "s3_pooled_values": s3_vals,
        "s2_pooled_values": s2_vals,
        "realized_delta_pct_wall": realized,
        "composed_delta_pct": COMPOSED_DELTA_PCT,
        "realization_ratio_487": ratio,
        "classification": cls,
        "free_ceiling_realizes_on_wall": ratio >= REALIZES_RATIO,
        "s3_projected_official_median": s3_proj_med,
        "measured_free_ceiling_tps": measured_ceiling,
        "measured_free_ceiling_tps_frontier_anchored": measured_ceiling_frontier,
        "free_ceiling_below_composed": measured_ceiling < ENVELOPE_TPS,
    })
    return out


# ---------------------------------------------------------------------------
# Self-test (PRIMARY headline).
# ---------------------------------------------------------------------------
def constants_exact() -> dict[str, Any]:
    expect = {
        "FRONTIER_TPS": (FRONTIER_TPS, 481.53),
        "ENVELOPE_TPS": (ENVELOPE_TPS, 487.72885498477575),
        "DENKEN_FLOOR_TPS": (DENKEN_FLOOR_TPS, 487.7289),
        "KANNA_BASIS_TPS": (KANNA_BASIS_TPS, 493.64),
        "STEP_US": (STEP_US, 1218.2),
        "NEW_STEP_US": (NEW_STEP_US, 1202.7171244939168),
        "BRIDGE_0278": (BRIDGE_0278, 0.2147),
        "K_CAL": (K_CAL, 125.268),
        "E_T": (E_T, 3.844),
    }
    bad = {k: got for k, (got, want) in expect.items() if got != want}
    composed_ok = abs(COMPOSED_DELTA_PCT - COMPOSED_DELTA_PCT_JSON) < 1e-6
    return {"all_exact": (not bad) and composed_ok, "mismatches": bad,
            "composed_delta_pct": COMPOSED_DELTA_PCT, "composed_matches_json": composed_ok}


def caveats() -> dict[str, str]:
    return {
        "zero_tps": ("This leg adds 0 TPS: it wall-audits the banked free step ceiling. "
                     "It does NOT produce a ≥500 build and does NOT change the served checkpoint. "
                     "BASELINE stays 481.53; launch gate stays MEASURED ≥500, human-approval-gated."),
        "measures_deployed_not_built": ("Measures the DEPLOYED stack + the banked greedy-safe num_stages "
                                        "3→2 lever via a temporary reverted toggle — NOT a new build, NOT a "
                                        "submission, NOT an HF Job, NOT a launch, NOT open2."),
        "wall_not_kernel": ("487.7 is an analytically-composed / kernel-µs ceiling (no in-graph overlap "
                            "in a standalone replay → an UPPER bound). This A/B measures the host-to-host "
                            "wall, where the large FIXED serving overhead does not shrink with the µs saving; "
                            "realization_ratio<1 quantifies the wall-vs-kernel gap (static-K precedent: −2.02)."),
    }


def self_test(agg: dict[str, Any], verifies: dict[int, dict], smoke: bool) -> dict[str, Any]:
    cexact = constants_exact()
    cav = caveats()

    s3_vals = agg.get("s3_pooled_values") or []
    s2_vals = agg.get("s2_pooled_values") or []
    all_finite = (bool(s3_vals) and bool(s2_vals)
                  and len(_finite_pos(s3_vals)) == len(s3_vals)
                  and len(_finite_pos(s2_vals)) == len(s2_vals))

    candidate_applied = bool(verifies) and all(v.get("applied_ok") for v in verifies.values())

    realized = agg.get("realized_delta_pct_wall")
    ratio = agg.get("realization_ratio_487")
    ratio_consistent = (realized is not None and ratio is not None
                        and abs(ratio * COMPOSED_DELTA_PCT - realized) < 1e-6)

    # s3 reproduces the deployed-K7 wall reference within the operative 1% MDE.
    s3_p50 = agg.get("s3_pooled_p50_wall_tps")
    repro_err = (100.0 * abs(s3_p50 - DEPLOYED_K7_REF_WALL_TPS) / DEPLOYED_K7_REF_WALL_TPS
                 if isinstance(s3_p50, (int, float)) else None)
    reproduces_k7 = (repro_err is not None and repro_err <= 1.0)

    required = {
        "constants_exact": cexact["all_exact"],
        "candidate_actually_applied_s2": candidate_applied,
        "ratio_consistent": ratio_consistent,
        "all_tps_finite_positive": all_finite,
        "caveats_present": all(isinstance(v, str) and len(v) > 40 for v in cav.values()),
    }
    if not smoke:
        required["s3_reproduces_deployed_k7"] = reproduces_k7

    passes = all(required.values())
    return {
        "free_ceiling_wallclock_realize_self_test_passes": passes,
        "required": required,
        "smoke": smoke,
        "constants": cexact,
        "caveats": cav,
        "candidate_applied_per_seed": verifies,
        "s3_reproduction_err_pct": repro_err,
        "s3_reproduces_deployed_k7": reproduces_k7,
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
        job_type="free-ceiling-wallclock-realize", agent="stark",
        name=args.wandb_name, group=args.wandb_group,
        tags=["pr298", "wall-realization", "num_stages", SUBMISSION],
        config={
            "frontier_tps": FRONTIER_TPS, "envelope_tps": ENVELOPE_TPS,
            "denken_floor_tps": DENKEN_FLOOR_TPS, "composed_delta_pct": COMPOSED_DELTA_PCT,
            "deployed_k7_ref_wall_tps": DEPLOYED_K7_REF_WALL_TPS,
            "seeds": args.seeds, "n": args.n, "smoke": args.smoke,
            "num_prompts": args.num_prompts, "output_len": args.output_len,
            "lever": "verify_sdpa_num_stages_3to2",
        },
    )
    if run is None:
        _log("wandb disabled (no API key / WANDB_DISABLED); skipping")
        return
    try:
        flat = {
            "free_ceiling_wallclock_realize_self_test_passes": float(
                st["free_ceiling_wallclock_realize_self_test_passes"]),
        }
        for k in ("s3_pooled_p50_wall_tps", "s2_pooled_p50_wall_tps",
                  "realized_delta_pct_wall", "composed_delta_pct", "realization_ratio_487",
                  "measured_free_ceiling_tps", "measured_free_ceiling_tps_frontier_anchored",
                  "s3_projected_official_median"):
            v = agg.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                flat[k] = v
        flat["free_ceiling_realizes_on_wall"] = float(bool(agg.get("free_ceiling_realizes_on_wall")))
        flat["free_ceiling_below_composed"] = float(bool(agg.get("free_ceiling_below_composed")))
        flat["s3_reproduces_deployed_k7"] = float(bool(st.get("s3_reproduces_deployed_k7")))
        for k, v in st["required"].items():
            flat[f"selftest/{k}"] = float(bool(v))
        for ps in agg.get("per_seed", []):
            s = ps.get("seed")
            if isinstance(ps.get("delta_median_pct"), (int, float)):
                flat[f"seed{s}/delta_median_pct"] = ps["delta_median_pct"]
            if isinstance(ps.get("s3_median"), (int, float)):
                flat[f"seed{s}/s3_median"] = ps["s3_median"]
            if isinstance(ps.get("s2_median"), (int, float)):
                flat[f"seed{s}/s2_median"] = ps["s2_median"]
        run.summary["classification"] = agg.get("classification")
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="free_ceiling_wallclock_realize",
            artifact_type="wall-realization", data=result)
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
    ap.add_argument("--n", type=int, default=3, help="fresh runs per arm per seed (median-of-N)")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--smoke", action="store_true",
                    help="cheap boot+patch-fires check: n=1, seeds=1, reproduction non-required")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="evaluate the self-test booleans and exit non-zero if it fails")
    ap.add_argument("--no-toggle", action="store_true",
                    help="do NOT edit sitecustomize (expects an external toggle / dry structure check)")
    ap.add_argument("--fresh", action="store_true", help="ignore on-disk paired_ab.json and re-run every seed")
    ap.add_argument("--out-root", type=Path, default=HERE / "ab_out")
    ap.add_argument("--wandb_group", default="free-ceiling-wallclock-realize")
    ap.add_argument("--wandb_name", default="stark/free-ceiling-wallclock-realize")
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

    _log(f"seeds={seeds} n={args.n} smoke={args.smoke} workload={args.num_prompts}x{args.output_len} "
         f"-> {out_root}")
    _log(f"composed_delta_pct={COMPOSED_DELTA_PCT:.10f}% (json {COMPOSED_DELTA_PCT_JSON})")

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
            verifies[seed] = verify_arms_applied(out_root / f"seed{seed}")
            _log(f"seed {seed}: applied_ok={verifies[seed].get('applied_ok')} "
                 f"forced_hits={verifies[seed].get('s2_forced_log_hits')}")
    finally:
        toggle_clean = revert_toggle(original_bytes) if toggled and original_bytes is not None else True

    agg = aggregate(seed_jsons, verifies)
    st = self_test(agg, verifies, smoke=args.smoke)
    st["toggle_reverted_clean"] = toggle_clean
    if not toggle_clean:
        st["required"]["toggle_reverted_clean"] = False
        st["free_ceiling_wallclock_realize_self_test_passes"] = False

    result = {
        "experiment": "free_ceiling_wallclock_realize", "pr": 298, "student": "stark",
        "question": "Does the banked 487.7 free step ceiling realize on a direct local wall-clock A/B?",
        "lever": "verify SDPA num_stages 3->2 (wirbel #285/#279, denken #291)",
        "zero_tps": True, "not_a_launch": True, "not_a_build": True,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seeds": seeds},
        "elapsed_s": time.time() - t0,
        "aggregate": agg,
        "self_test": st,
    }
    (out_root / "results.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- console verdict ----
    print("\n" + "=" * 78, flush=True)
    print("FREE-CEILING WALL-CLOCK REALIZATION (PR #298, stark)", flush=True)
    print("=" * 78, flush=True)
    if "error" not in agg:
        print(f"  s3 (deployed, num_stages=3) p50 wall_tps = {agg['s3_pooled_p50_wall_tps']:.4f} "
              f"(ref {DEPLOYED_K7_REF_WALL_TPS:.4f}, Δ{st.get('s3_reproduction_err_pct'):.3f}%)", flush=True)
        print(f"  s2 (lever,    num_stages=2) p50 wall_tps = {agg['s2_pooled_p50_wall_tps']:.4f}", flush=True)
        print(f"  realized Δ%_wall = {agg['realized_delta_pct_wall']:+.4f}%   "
              f"composed Δ% = {COMPOSED_DELTA_PCT:+.4f}%", flush=True)
        print(f"  >>> realization_ratio_487 = {agg['realization_ratio_487']:+.4f}  "
              f"[{agg['classification']}]", flush=True)
        print(f"  measured_free_ceiling_tps = {agg['measured_free_ceiling_tps']:.3f}  "
              f"(< composed {ENVELOPE_TPS:.3f}? {agg['free_ceiling_below_composed']})", flush=True)
    else:
        print(f"  AGGREGATE ERROR: {agg['error']}", flush=True)
    for ps in agg.get("per_seed", []):
        print(f"   seed{ps['seed']}: s3={ps.get('s3_median')} s2={ps.get('s2_median')} "
              f"Δ={ps.get('delta_median_pct')}% [{ps.get('verdict')}] "
              f"applied_ok={ps.get('applied', {}).get('applied_ok')}", flush=True)
    print(f"  toggle_reverted_clean = {toggle_clean}", flush=True)
    print(f"  >>> SELF-TEST PASSES = {st['free_ceiling_wallclock_realize_self_test_passes']} "
          f"required={ {k: v for k, v in st['required'].items()} }", flush=True)
    print("=" * 78 + "\n", flush=True)

    log_wandb(args, result)
    print(f"[free-ceiling] artifacts -> {out_root / 'results.json'}", flush=True)

    if args.self_test and not st["free_ceiling_wallclock_realize_self_test_passes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
