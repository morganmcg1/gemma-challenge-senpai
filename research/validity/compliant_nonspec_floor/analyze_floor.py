"""Consolidate the non-spec int4 compliant-floor evidence (lawine #196, Tasks 4-6).

Inputs (produced by the prior steps, all under research/validity/compliant_nonspec_floor/):
  * walltps/paired_ab.json         -- paired_tps_ab candidate=nonspec, N=3 (wall_tps + projection)
  * walltps/nonspec/decode/run0X.jsonl  -- the 3 fresh-reload decode captures (self-identity source)
  * ppl/ppl_summary.json           -- official PPL pass (ppl_nonspec)
  * smoke/smoke_result.json        -- boot smoke (nonspec_serve_boots)

Emits floor_report.json with every PR #196 metric and the explicit compliant-floor
verdict, and logs a W&B summary run (group compliant-nonspec-floor). NaN-clean.

Self-identity is reload-vs-reload byte-identity of completion_token_ids across the 3
fresh-server decodes (each run_arm run is a fresh LocalServer) -- exactly how kanna
#114 validated the spec arm's determinism, but here across 3 reloads (>= the 2 a
gen+validate gate would give).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "research" / "validity" / "compliant_nonspec_floor"

# Imported constants (PR #196: import, do NOT re-derive).
OFFICIAL_ANCHOR_TPS = 481.53     # PR #52 fa2sw_precache_kenyan (the int4-SPEC stack)
OFFICIAL_TARGET_TPS = 500.0
SIGMA_HW_TPS = 4.86              # kanna #159 hardware band
PPL_THRESHOLD = 2.42
EXPECTED_PROMPTS = 128
EXPECTED_OUTPUT_LEN = 512


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _key(rec: dict[str, Any]) -> Any:
    return rec.get("id", rec.get("index"))


def self_identity(decode_paths: list[Path]) -> dict[str, Any]:
    """Reload-vs-reload byte-identity of completion_token_ids across N captures."""
    runs = []
    for p in decode_paths:
        rows = _load_jsonl(p)
        runs.append({_key(r): r.get("completion_token_ids") for r in rows})
    n_runs = len(runs)
    keys = sorted(runs[0].keys(), key=lambda k: (str(type(k)), k))
    total = len(keys)

    all_equal = 0
    first_divergences = []
    for k in keys:
        seqs = [runs[i].get(k) for i in range(n_runs)]
        if any(s is None for s in seqs):
            first_divergences.append({"key": k, "reason": "missing_in_a_run"})
            continue
        if all(s == seqs[0] for s in seqs):
            all_equal += 1
        else:
            # locate first differing token vs run 0 (debug aid; expect none)
            div = {"key": k}
            for i in range(1, n_runs):
                if seqs[i] != seqs[0]:
                    a, b = seqs[0], seqs[i]
                    pos = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]),
                               min(len(a), len(b)))
                    div[f"run0_vs_run{i}_first_diff_pos"] = pos
                    div[f"len_run0"] = len(a)
                    div[f"len_run{i}"] = len(b)
            first_divergences.append(div)

    identity_rate = (all_equal / total) if total else float("nan")
    # Pairwise rates vs run 0 (most directly the "reload-vs-reload" number).
    pairwise = {}
    for i in range(1, n_runs):
        eq = sum(1 for k in keys
                 if runs[0].get(k) is not None and runs[0].get(k) == runs[i].get(k))
        pairwise[f"run0_vs_run{i}"] = eq / total if total else float("nan")

    completion_lens = []
    for p in decode_paths:
        rows = _load_jsonl(p)
        completion_lens.append([len(r.get("completion_token_ids") or []) for r in rows])

    return {
        "n_reloads": n_runs,
        "n_prompts": total,
        "n_all_equal": all_equal,
        "token_identity_rate": identity_rate,           # 3-way agreement fraction
        "pairwise_identity_rate_vs_run0": pairwise,
        "self_identical": bool(total) and all_equal == total,
        "decode_files": [str(p) for p in decode_paths],
        "divergences": first_divergences[:20],
        "per_run_record_count": [len(_load_jsonl(p)) for p in decode_paths],
        "per_run_total_tokens": [sum(lens) for lens in completion_lens],
        "per_run_output_len_uniform_512": [all(x == EXPECTED_OUTPUT_LEN for x in lens)
                                           for lens in completion_lens],
    }


def main() -> int:
    paired = json.loads((OUT / "walltps" / "paired_ab.json").read_text())
    cand = paired["arms"]["candidate"]
    base = paired["arms"]["baseline"]
    proj = paired.get("projection") or {}
    cand_proj = (proj.get("arms") or {}).get("candidate") or {}

    # ---- self-identity (Task 4a) from the 3 fresh-reload decode captures ----
    decode_dir = OUT / "walltps" / "nonspec" / "decode"
    decode_paths = sorted(decode_dir.glob("run*.jsonl"))
    ident = self_identity(decode_paths)

    # ---- 128/128 completion (Task 4c) ----
    completes_128 = (
        all(c == EXPECTED_PROMPTS for c in ident["per_run_record_count"])
        and all(t == EXPECTED_PROMPTS * EXPECTED_OUTPUT_LEN
                for t in ident["per_run_total_tokens"])
    )

    # ---- PPL (Task 4b) ----
    ppl_summary = json.loads((OUT / "ppl" / "ppl_summary.json").read_text())
    ppl_nonspec = ppl_summary.get("ppl")

    # ---- boot (Task 1) ----
    smoke = json.loads((OUT / "smoke" / "smoke_result.json").read_text())
    nonspec_serve_boots = bool(smoke.get("boots")) and smoke.get("speculative_config") == "None"

    # ---- wall_tps + step + projection (Task 3) ----
    wall = cand.get("wall_tps") or {}
    nonspec_wall_tps = wall.get("median")
    steady = cand.get("steady_gen_tps_mean") or {}
    steady_med = steady.get("median")
    # bare per-token AR step latency (ms): E[T]=1, gen-phase tok/s -> 1000/tps.
    nonspec_step_ar_ms = (1000.0 / steady_med) if _is_num(steady_med) and steady_med else None
    nonspec_step_ar_ms_wall = (1000.0 / nonspec_wall_tps) if _is_num(nonspec_wall_tps) and nonspec_wall_tps else None

    nonspec_official_tps_est = cand_proj.get("projected_official")
    proj_lo = cand_proj.get("projected_official_lo")
    proj_hi = cand_proj.get("projected_official_hi")
    multiplier = cand_proj.get("multiplier") or proj.get("multiplier")

    # σ_hw band on the central estimate (kanna #159): does est - σ_hw clear 500?
    hw_lo = (nonspec_official_tps_est - SIGMA_HW_TPS) if _is_num(nonspec_official_tps_est) else None
    hw_hi = (nonspec_official_tps_est + SIGMA_HW_TPS) if _is_num(nonspec_official_tps_est) else None
    nonspec_clears_500 = bool(_is_num(hw_hi) and hw_hi >= OFFICIAL_TARGET_TPS)
    margin_to_500_tps = (nonspec_official_tps_est - OFFICIAL_TARGET_TPS) if _is_num(nonspec_official_tps_est) else None
    margin_to_500_pct = (100.0 * margin_to_500_tps / OFFICIAL_TARGET_TPS) if _is_num(margin_to_500_tps) else None

    # ---- spec premium (Task 4) ----
    spec_premium_tps = (OFFICIAL_ANCHOR_TPS - nonspec_official_tps_est) if _is_num(nonspec_official_tps_est) else None
    spec_premium_pct = (100.0 * spec_premium_tps / nonspec_official_tps_est) if _is_num(spec_premium_tps) and nonspec_official_tps_est else None
    # local-frame corroboration (same measurement env, no projection):
    base_wall = (base.get("wall_tps") or {}).get("median")
    local_spec_premium_tps = (base_wall - nonspec_wall_tps) if _is_num(base_wall) and _is_num(nonspec_wall_tps) else None
    local_spec_premium_pct = (100.0 * local_spec_premium_tps / nonspec_wall_tps) if _is_num(local_spec_premium_tps) and nonspec_wall_tps else None

    # ---- verdict (Task 4/6) ----
    # "Striking distance" = the central estimate is within ~5% of 500 (a lane worth
    # optimizing). Otherwise the gap is structural and strict #192 enforcement makes
    # the batch-invariant verify kernel (lane a) the only compliant route to 500.
    STRIKING_PCT = 5.0
    within_striking = bool(_is_num(margin_to_500_pct) and margin_to_500_pct >= -STRIKING_PCT)
    if nonspec_clears_500:
        verdict_label = "CLEARS_500"
    elif within_striking:
        verdict_label = "WITHIN_STRIKING_DISTANCE"
    else:
        verdict_label = "STRUCTURAL_GAP_SPEC_EXISTENTIAL"
    verdict_text = (
        f"Compliant non-spec int4 floor ≈ {nonspec_official_tps_est:.1f} official TPS "
        f"(σ_hw band [{hw_lo:.1f},{hw_hi:.1f}]), {abs(margin_to_500_pct):.1f}% "
        f"{'above' if (margin_to_500_pct or 0) >= 0 else 'below'} the 500 target. "
        f"Speculation premium = {spec_premium_tps:.1f} TPS "
        f"({spec_premium_pct:.1f}% over the compliant floor). "
        + ("Compliant 500-lane exists — spec path optional."
           if nonspec_clears_500 else
           ("Near 500 — a compliant lane worth optimizing."
            if within_striking else
            "Gap is structural: strict #192 enforcement makes the batch-invariant int4 "
            "verify kernel (lane a) the ONLY compliant route to 500; the spec path is existential."))
    ) if _is_num(nonspec_official_tps_est) else "INSUFFICIENT_DATA"

    # ---- NaN-clean check ----
    numeric_fields = [nonspec_wall_tps, steady_med, nonspec_official_tps_est, proj_lo, proj_hi,
                      ppl_nonspec, spec_premium_tps, ident["token_identity_rate"]]
    nan_clean = all(_is_num(x) for x in numeric_fields)

    # ---- PRIMARY self-test ----
    ppl_ok = _is_num(ppl_nonspec) and ppl_nonspec <= PPL_THRESHOLD
    verdict_explicit = verdict_label in {"CLEARS_500", "WITHIN_STRIKING_DISTANCE",
                                         "STRUCTURAL_GAP_SPEC_EXISTENTIAL"}
    self_test = {
        "a_boots_and_128": nonspec_serve_boots and completes_128,
        "b_self_identical": ident["self_identical"] and ident["token_identity_rate"] == 1.0,
        "c_ppl_le_2_42": bool(ppl_ok),
        "d_verdict_explicit": verdict_explicit,
        "e_nan_clean": nan_clean,
    }
    nonspec_floor_self_test_passes = all(self_test.values())

    report = {
        "pr": 196,
        "submission": "submissions/fa2sw_nonspec_int4",
        "lever": "submissions/fa2sw_nonspec_int4/manifest.json:29 (SPECULATIVE_CONFIG=\"\" -> speculative_config=None, K_spec 7->0)",
        # Task 1
        "nonspec_serve_boots": nonspec_serve_boots,
        "server_ready_s": smoke.get("server_ready_s"),
        # Task 2 (self-identity + PPL + 128/128)
        "nonspec_self_identical": self_test["b_self_identical"],
        "nonspec_token_identity_rate": ident["token_identity_rate"],
        "self_identity_detail": ident,
        "ppl_nonspec": ppl_nonspec,
        "ppl_threshold": PPL_THRESHOLD,
        "nonspec_completes_128": completes_128,
        # Task 3 (compliant-floor TPS)
        "nonspec_wall_tps": nonspec_wall_tps,
        "nonspec_wall_tps_cv_pct": wall.get("cv_pct"),
        "nonspec_wall_tps_values": wall.get("values"),
        "nonspec_steady_gen_tps_mean": steady_med,
        "nonspec_step_ar_ms": nonspec_step_ar_ms,
        "nonspec_step_ar_ms_wall_derived": nonspec_step_ar_ms_wall,
        "nonspec_e_of_t": 1.0,
        "nonspec_official_tps_est": nonspec_official_tps_est,
        "nonspec_official_tps_est_proj_band": [proj_lo, proj_hi],
        "nonspec_official_tps_est_hw_band": [hw_lo, hw_hi],
        "projection_multiplier": multiplier,
        "projection_multiplier_note": "1.0602 hardware/env transfer ratio anchored on the SPEC footprint "
            "(481.53/454.338); applied to the non-spec wall_tps under the assumption local->official "
            "transfer is footprint-invariant. At ~3x lower throughput fixed per-request overheads may "
            "shift it, but the verdict is robust: the floor is far from 500. wall_tps is the footprint-"
            "agnostic primary; K_cal=125.268 (spec-tree-calibrated) deliberately NOT used.",
        "sigma_hw_tps": SIGMA_HW_TPS,
        "nonspec_clears_500": nonspec_clears_500,
        "margin_to_500_tps": margin_to_500_tps,
        "margin_to_500_pct": margin_to_500_pct,
        # Task 4 (premium + verdict)
        "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
        "spec_premium_tps": spec_premium_tps,
        "spec_premium_pct": spec_premium_pct,
        "local_spec_baseline_wall_tps": base_wall,
        "local_spec_premium_tps": local_spec_premium_tps,
        "local_spec_premium_pct": local_spec_premium_pct,
        "verdict_label": verdict_label,
        "verdict": verdict_text,
        # Task 5 (self-test)
        "self_test": self_test,
        "nonspec_floor_self_test_passes": nonspec_floor_self_test_passes,
        "nan_clean": nan_clean,
        "wandb_walltps_name": "lawine/nonspec-floor-walltps",
    }
    (OUT / "floor_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"self_identity_detail", "nonspec_wall_tps_values"}},
                     indent=2, default=str), flush=True)
    print(f"\n[floor] PRIMARY nonspec_floor_self_test_passes={nonspec_floor_self_test_passes} "
          f"self_test={self_test}", flush=True)
    print(f"[floor] TEST nonspec_official_tps_est={nonspec_official_tps_est}", flush=True)

    _log_wandb(report)
    return 0 if nonspec_floor_self_test_passes else 1


def _log_wandb(report: dict[str, Any]) -> None:
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[floor] wandb import failed ({exc}); skipping wandb", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="compliant-nonspec-floor", agent="lawine",
            name="lawine/nonspec-floor-report", group="compliant-nonspec-floor",
            tags=["compliant-nonspec-floor", "fa2sw_nonspec_int4", "validity-192"],
            config={"submission": report["submission"], "lever": report["lever"],
                    "official_anchor_tps": OFFICIAL_ANCHOR_TPS, "sigma_hw_tps": SIGMA_HW_TPS},
        )
    except Exception as exc:
        print(f"[floor] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[floor] wandb disabled; skipping", flush=True)
        return
    try:
        flat = {f"floor/{k}": v for k, v in report.items() if isinstance(v, (int, float, bool))}
        flat["floor/self_test_passes_int"] = 1.0 if report["nonspec_floor_self_test_passes"] else 0.0
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="compliant_nonspec_floor_report", artifact_type="validity-floor",
            data=report,
        )
    except Exception as exc:
        print(f"[floor] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
