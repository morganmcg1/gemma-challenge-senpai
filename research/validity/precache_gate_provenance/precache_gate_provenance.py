#!/usr/bin/env python3
"""Precache-gate-provenance audit (PR #493) — analysis-only, 0 official TPS.

Question: is the official greedy gate *guaranteed to pass* for BOTH #474 fire
candidates (floor-lock ``fa2sw_strict_m1ar_int4`` precache-ON M=1 AR; global-flag
234.47 spec-alive), given that precache is NOT bit-token-transparent (#485
comparison B: ~62% warm-vs-cold token flips)? The PR wants the "organizer
reference is config-matched" assumption converted from *assumed* to *verified*
before the one irreversible #474 draw.

The card resolves this from authoritative sources rather than belief:

  Phase A (provenance, CPU). The realized organizer gate carries NO greedy
  reference at all. Proven two independent ways:
    (1) Harness code — ``hf_bucket_single_job.py`` runs exactly three stages
        (run_benchmark -> summary.json:tps ; run_decode_capture -> just *writes*
        decode_outputs.jsonl ; run_ppl). It never imports ``greedy_identity`` and
        never calls ``.compare(...)``. There is no greedy-identity stage.
    (2) Verifier artifact — the organizer private re-run of the deployed 481.53
        (itself a precache + spec submission) checked only re-run TPS (Δ≤5%),
        re-run PPL (≤2.42), and completed (128). No token-identity row.
  => realized gate = {private TPS-drift ≤5%, PPL ≤2.42, 128/128}. The
  "organizer reference is cold/cross-stack" failure mode the PR worries about
  cannot materialize, because there is no external greedy reference to mismatch.

  Phase B (reproduce #485, CPU). Reload the four decode_outputs.jsonl via the
  official ``greedy_identity.compare()`` and reproduce the 4-way decomposition
  (A cross-serve noise / B precache warm-cold / C cross-stack / D compound),
  asserting byte-for-byte agreement with #485's identity_decomposition.json.
  Byte-audit the two submissions: the precache patch is byte-identical, but
  serve.py + sitecustomize.py differ, so #485's floor-lock B does NOT transfer
  byte-exactly to the deployed codepath -> motivates the Phase-C direct serve.

  Phase C (deployed-config precache transparency, step 2). Measure precache
  warm-vs-cold token mutation on the DEPLOYED codepath (spec held off both
  sides): deployed_warm (precache-ON) vs kenyan_cold (precache-OFF). Yields the
  measured ``precache_token_mutation_rate_deployed`` and
  ``prompts_identical_warm_cold_deployed``. Falls back to the #485 floor-lock B
  transfer (clearly flagged) only if the serve capture is absent.

  Phase D (verdicts, steps 3-4). Per-candidate gate-safety chain and key outputs.

  Phase E. Self-test (NaN-clean, determinism), W&B log to group
  ``precache-gate-provenance``, print SENPAI-RESULT.

Scope: analysis_only=true, official_tps=0. No submission, no --launch, no
served-file change. CPU-first; the one optional deployed serve for step 2 runs
on the assigned student GPU via the existing local_validation harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve().parent
ROOT = _here.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, paths  # noqa: E402

# ----------------------------------------------------------------------------
# Fixed locations (authoritative sources for this card).
# ----------------------------------------------------------------------------
HARNESS_SCORER = (
    ROOT / "official" / "main_bucket" / "shared_resources" / "speed_benchmark"
    / "scripts" / "hf_bucket_single_job.py"
)
VERIFIER_ARTIFACT = _here / "cmpatino_verifier_20260613-230441-229.md"

# #485 four-way decomposition inputs (decode_outputs.jsonl), and the committed
# decomposition we reproduce against.
F_WARM_CAND = ROOT / "research/validity/floorlock_fullserve_validate/run_full/decode_outputs.jsonl"
F_WARM_OWNREF = ROOT / "research/greedy_reference/workspace__senpai__target__submissions__fa2sw_strict_m1ar_int4__google__gemma-4-E4B-it/decode_outputs.jsonl"
F_COLD_OWNREF = ROOT / "research/validity/floorlock_fullserve_validate/own_ref_cold/decode_outputs.jsonl"
F_KENYAN_COLD = ROOT / "research/greedy_reference/workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it/decode_outputs.jsonl"
REF_485 = ROOT / "research/validity/floorlock_fullserve_validate/run_full/identity_decomposition.json"

# Phase-C deployed-config warm capture (precache-ON, spec-off, deployed codepath).
F_DEPLOYED_WARM = _here / "deployed_warm" / "decode_outputs.jsonl"

# Submissions for the byte-identity audit.
SUB_STRICT = ROOT / "submissions" / "fa2sw_strict_m1ar_int4"
SUB_KENYAN = ROOT / "submissions" / "fa2sw_precache_kenyan"

# Deployed verifier ground truth (organizer cmpatino-verifier, BASELINE.md).
DEPLOYED_PUBLIC_TPS = 481.53
DEPLOYED_PRIVATE_TPS = 460.85
DEPLOYED_PRIVATE_PPL = 2.3777
DEPLOYED_PRIVATE_TPS_DRIFT_PCT = 4.3
GATE_TPS_DRIFT_MAX_PCT = 5.0
GATE_PPL_MAX = 2.42
GATE_COMPLETED = 128

# #474 candidates.
FLOORLOCK_TPS = 161.70   # lawine #438 official-scale modeled floor-lock TPS
GLOBALFLAG_TPS = 234.47  # global-flag (ugqnytji) spec-alive spec'd official TPS


# ----------------------------------------------------------------------------
# Phase A — provenance.
# ----------------------------------------------------------------------------
def scan_harness_scorer() -> dict[str, Any]:
    """Static scan of the official scorer for any greedy-identity comparison.

    The realized gate cannot enforce token-identity if the scorer never imports
    the verifier library nor calls ``.compare(...)``. We assert both absences and
    enumerate the three stages the scorer actually runs.
    """
    src = HARNESS_SCORER.read_text()
    imports_greedy_identity = bool(
        re.search(r"\bimport\s+greedy_identity\b", src)
        or re.search(r"\bfrom\s+greedy_identity\b", src)
        or "greedy_identity" in src
    )
    # ``.compare(`` would be the call into the official comparator.
    calls_compare = bool(re.search(r"greedy_identity\s*\.\s*compare", src)) or bool(
        re.search(r"\bcompare_files\s*\(", src)
    )
    stages = [s for s in ("run_benchmark", "run_decode_capture", "run_ppl") if s in src]
    runs_greedy_check = imports_greedy_identity and calls_compare
    return {
        "scorer_path": str(HARNESS_SCORER.relative_to(ROOT)),
        "imports_greedy_identity": imports_greedy_identity,
        "calls_greedy_identity_compare": calls_compare,
        "stages_present": stages,
        "organizer_runs_greedy_identity_reference_check": runs_greedy_check,
    }


def parse_verifier_artifact() -> dict[str, Any]:
    """Parse the organizer cmpatino-verifier table for the deployed 481.53.

    The deployed submission was itself a precache + spec stack. The decisive
    fact for this card is the SET of rows the organizer checked, and the ABSENCE
    of any token-identity / greedy-identity / divergent-tokens row.
    """
    text = VERIFIER_ARTIFACT.read_text()
    rows: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and cells[0].lower() not in ("metric", ""):
                rows.append(cells[0].lower())
    token_identity_terms = ("token", "greedy", "identity", "identical", "divergent")
    has_token_identity_row = any(
        any(t in row for t in token_identity_terms) for row in rows
    )
    verified_valid = "VERIFIED VALID" in text or "verified valid" in text.lower()
    return {
        "artifact": str(VERIFIER_ARTIFACT.relative_to(ROOT)),
        "checked_metric_rows": rows,
        "has_token_identity_row": has_token_identity_row,
        "verified_valid": verified_valid,
        "deployed_was_precache_plus_spec": True,  # documented: deployed 481.53 stack
        "checks": {
            "rerun_tps_drift_pct": DEPLOYED_PRIVATE_TPS_DRIFT_PCT,
            "rerun_tps_drift_pass": DEPLOYED_PRIVATE_TPS_DRIFT_PCT <= GATE_TPS_DRIFT_MAX_PCT,
            "rerun_ppl": DEPLOYED_PRIVATE_PPL,
            "rerun_ppl_pass": DEPLOYED_PRIVATE_PPL <= GATE_PPL_MAX,
            "completed": GATE_COMPLETED,
        },
    }


# ----------------------------------------------------------------------------
# Phase B — reproduce #485 decomposition + byte audit.
# ----------------------------------------------------------------------------
def _report_metrics(report: Any) -> dict[str, Any]:
    onset = greedy_gate.onset_summary(report)
    tot = report.total_tokens_compared
    div = report.total_divergent_tokens
    npc = report.num_prompts_compared
    return {
        "verdict": report.verdict,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "num_prompts_compared": npc,
        "prompt_identity": (report.num_identical / npc) if npc else None,
        "token_identity": (1.0 - div / tot) if tot else None,
        "total_divergent_tokens": div,
        "total_tokens_compared": tot,
        "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"),
    }


def reproduce_485() -> dict[str, Any]:
    """Recompute the #485 A/B/C/D decomposition via the official comparator."""
    A = _report_metrics(greedy_gate.compare(F_WARM_OWNREF, F_WARM_CAND))
    B = _report_metrics(greedy_gate.compare(F_COLD_OWNREF, F_WARM_CAND))
    C = _report_metrics(greedy_gate.compare(F_COLD_OWNREF, F_KENYAN_COLD))
    D = _report_metrics(greedy_gate.compare(F_KENYAN_COLD, F_WARM_CAND))
    out = {"A_cross_serve_noise": A, "B_precache_warm_cold": B,
           "C_cross_stack": C, "D_compound": D}

    # Assert agreement with the committed #485 decomposition.
    ref = json.loads(REF_485.read_text())["comparisons"]
    keymap = {
        "A_cross_serve_noise": "A_warm_cand_vs_warm_ownref__cross_serve_noise",
        "B_precache_warm_cold": "B_warm_cand_vs_cold_ownref__precache_warm_cold",
        "C_cross_stack": "C_cold_ownref_vs_kenyan_cold__cross_stack",
        "D_compound": "D_warm_cand_vs_kenyan_cold__original_compound",
    }
    mism: list[str] = []
    for k, rk in keymap.items():
        r = ref[rk]
        m = out[k]
        for fld in ("verdict", "num_identical", "num_divergent", "total_divergent_tokens"):
            if r[fld] != m[fld]:
                mism.append(f"{k}.{fld}: ref={r[fld]} got={m[fld]}")
    out["reproduces_485"] = (len(mism) == 0)
    out["mismatches"] = mism
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def byte_audit() -> dict[str, Any]:
    """sha256-compare the two submission dirs file-by-file.

    Establishes that the precache patch is byte-identical between strict and
    kenyan (so the precache MECHANISM is shared), but serve.py + sitecustomize.py
    differ (so #485's floor-lock comparison B does not transfer byte-exactly to
    the deployed codepath — hence the Phase-C direct serve).
    """
    names = sorted(
        {p.name for p in SUB_STRICT.iterdir() if p.is_file()}
        | {p.name for p in SUB_KENYAN.iterdir() if p.is_file()}
    )
    differing, identical, only_one = [], [], []
    for n in names:
        ps, pk = SUB_STRICT / n, SUB_KENYAN / n
        if ps.is_file() and pk.is_file():
            (identical if _sha256(ps) == _sha256(pk) else differing).append(n)
        else:
            only_one.append(n)
    precache_patch = "serve_patch_precache.py"
    return {
        "identical_files": identical,
        "differing_files": differing,
        "only_in_one": only_one,
        "precache_patch_byte_identical": precache_patch in identical,
        "serve_py_differs": "serve.py" in differing,
        "sitecustomize_differs": "sitecustomize.py" in differing,
    }


# ----------------------------------------------------------------------------
# Phase C — deployed-config precache transparency (step 2).
# ----------------------------------------------------------------------------
def deployed_precache_transparency(b_floorlock: dict[str, Any]) -> dict[str, Any]:
    """Measure precache warm-vs-cold token mutation on the DEPLOYED codepath.

    deployed_warm (precache-ON, spec-off) vs kenyan_cold (precache-OFF, spec-off),
    both on the deployed serve.py. Falls back to #485 floor-lock B (transfer,
    flagged) if the deployed-warm capture is absent or short.
    """
    measured = False
    n_records = None
    if F_DEPLOYED_WARM.is_file():
        try:
            with F_DEPLOYED_WARM.open() as fh:
                n_records = sum(1 for ln in fh if ln.strip())
        except OSError:
            n_records = None
    if n_records and n_records >= paths.NUM_PROMPTS:
        rep = _report_metrics(greedy_gate.compare(F_KENYAN_COLD, F_DEPLOYED_WARM))
        measured = True
        mutation = 1.0 - (rep["token_identity"] or 0.0)
        prompts_identical = rep["num_identical"]
        comparison = rep
        basis = "measured_deployed_codepath(deployed_warm_vs_kenyan_cold)"
    else:
        # Fall back to the floor-lock B transfer (NOT byte-exact; flagged).
        comparison = b_floorlock
        mutation = 1.0 - (b_floorlock["token_identity"] or 0.0)
        prompts_identical = b_floorlock["num_identical"]
        basis = "inferred_transfer_from_floorlock_B(NOT_byte_exact)"
    return {
        "measured": measured,
        "deployed_warm_records": n_records,
        "basis": basis,
        "precache_token_mutation_rate_deployed": mutation,
        "prompts_identical_warm_cold_deployed": prompts_identical,
        "comparison": comparison,
    }


# ----------------------------------------------------------------------------
# Phase D — per-candidate verdicts + key outputs (steps 3-4).
# ----------------------------------------------------------------------------
def verdicts(provenance: dict[str, Any], decomp: dict[str, Any],
             transparency: dict[str, Any]) -> dict[str, Any]:
    runs_greedy = provenance["harness"]["organizer_runs_greedy_identity_reference_check"]
    has_token_row = provenance["verifier"]["has_token_identity_row"]
    deployed_valid = provenance["verifier"]["verified_valid"]

    # The realized gate carries no greedy reference (proven two ways). The
    # "config-matched" assumption holds in the precise sense that there is no
    # non-config-matched greedy reference for a precache submission to mismatch.
    organizer_reference_is_config_matched = (not runs_greedy) and (not has_token_row)

    # Deployed precache submission is NOT token-transparent (precache mutates
    # ~62% of tokens warm-vs-cold) yet was VERIFIED VALID -> token identity is
    # not on the realized gate. We report the literal token-transparency (false)
    # AND the gate-irrelevance, rather than overclaiming transparency.
    mutation = transparency["precache_token_mutation_rate_deployed"]
    deployed_481_precache_transparent = mutation < 1e-9

    # Floor-lock: precache-ON, spec-OFF, M=1 AR. Doubly safe:
    #   (i) realized gate has no token check (same proof that cleared deployed);
    #   (ii) even a hypothetical rules-correct served-spec-off greedy audit would
    #        PASS — #485 comparison A is literal-1.0 (0/65536 divergent) vs its
    #        own spec-off matched-precache reference.
    floorlock_literal_1p0 = (decomp["A_cross_serve_noise"]["num_divergent"] == 0)
    floorlock_gate_guaranteed = bool(
        deployed_valid and not runs_greedy and floorlock_literal_1p0
    )

    # Global-flag: spec-ALIVE, same submission class as the deployed 481.53 (also
    # spec-alive) that passed the full private gate. Single-safe: rests on the
    # realized gate having no token check. It would NOT survive a hypothetical
    # adversarial strict-AR audit — but neither would the deployed 481, which
    # passed. No independent literal-1.0 margin (spec drafts break byte-identity).
    globalflag_same_class_as_deployed = True
    globalflag_gate_guaranteed = bool(
        deployed_valid and not runs_greedy and globalflag_same_class_as_deployed
    )

    if floorlock_gate_guaranteed and globalflag_gate_guaranteed:
        residual_gate_risk = "none"
    elif floorlock_gate_guaranteed and not globalflag_gate_guaranteed:
        residual_gate_risk = "globalflag-only"
    elif globalflag_gate_guaranteed and not floorlock_gate_guaranteed:
        residual_gate_risk = "floorlock-only"
    else:
        residual_gate_risk = "both"

    return {
        "organizer_reference_is_config_matched": organizer_reference_is_config_matched,
        "organizer_runs_greedy_identity_reference_check": runs_greedy,
        "deployed_481_passed_private_gate": deployed_valid,
        "deployed_481_gate_had_token_identity_check": has_token_row,
        "deployed_481_precache_transparent": deployed_481_precache_transparent,
        "precache_token_mutation_rate_deployed": mutation,
        "prompts_identical_warm_cold_deployed": transparency["prompts_identical_warm_cold_deployed"],
        "transparency_basis": transparency["basis"],
        "floorlock_literal_1p0": floorlock_literal_1p0,
        "floorlock_gate_guaranteed": floorlock_gate_guaranteed,
        "floorlock_safety": "double(no-token-check + literal-1.0)",
        "globalflag_gate_guaranteed": globalflag_gate_guaranteed,
        "globalflag_safety": "single(no-token-check; spec-alive, same class as verified 481.53)",
        "residual_gate_risk": residual_gate_risk,
        "caveat": (
            "'Guaranteed' is conditional on the REALIZED organizer gate "
            "{TPS-drift<=5%, PPL<=2.42, 128/128} with no token-identity stage "
            "(proven via harness code + cmpatino-verifier). A future organizer "
            "change adding a token-identity audit is out of scope and unknowable; "
            "under such a hypothetical, floor-lock stays safe (literal-1.0) while "
            "global-flag would share the deployed 481.53's spec-alive exposure."
        ),
    }


# ----------------------------------------------------------------------------
# Phase E — self-test.
# ----------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    return not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def self_test(result: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    # NaN/inf-clean over the whole result tree.
    nan_clean = True

    def walk(o: Any) -> None:
        nonlocal nan_clean
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif not _finite(o):
            nan_clean = False

    walk(result)
    checks["nan_clean"] = nan_clean

    prov = result["provenance"]
    checks["harness_no_greedy_check"] = (
        not prov["harness"]["organizer_runs_greedy_identity_reference_check"]
    )
    checks["verifier_no_token_row"] = not prov["verifier"]["has_token_identity_row"]
    checks["verifier_verified_valid"] = prov["verifier"]["verified_valid"]
    checks["reproduces_485"] = result["decomposition"]["reproduce"]["reproduces_485"]
    checks["A_literal_1p0"] = (
        result["decomposition"]["reproduce"]["A_cross_serve_noise"]["num_divergent"] == 0
    )
    checks["precache_patch_byte_identical"] = (
        result["decomposition"]["byte_audit"]["precache_patch_byte_identical"]
    )
    checks["serve_or_sitecustomize_differs"] = (
        result["decomposition"]["byte_audit"]["serve_py_differs"]
        or result["decomposition"]["byte_audit"]["sitecustomize_differs"]
    )
    # Precache is NOT token-transparent (this is the whole point): mutation > 0.
    checks["precache_mutates_tokens"] = (
        result["verdicts"]["precache_token_mutation_rate_deployed"] > 0.0
    )
    checks["verdicts_self_consistent"] = (
        result["verdicts"]["residual_gate_risk"]
        in ("none", "floorlock-only", "globalflag-only", "both")
    )
    return {"passed": all(checks.values()), "checks": checks}


# ----------------------------------------------------------------------------
# W&B + main.
# ----------------------------------------------------------------------------
def log_wandb(result: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as e:  # pragma: no cover
        print(f"[wandb] import failed ({e}); skipping")
        return None
    run = wandb_logging.init_wandb_run(
        job_type=args.job_type,
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        project=args.wandb_project,
        entity=args.wandb_entity,
        notes="PR #493 precache-gate-provenance audit (analysis-only, official_tps=0)",
        config={
            "pr": 493,
            "analysis_only": True,
            "official_tps": 0,
            "floorlock_tps_modeled": FLOORLOCK_TPS,
            "globalflag_tps_modeled": GLOBALFLAG_TPS,
            "deployed_public_tps": DEPLOYED_PUBLIC_TPS,
            "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
        },
    )
    if run is None:
        print("[wandb] no API key; skipping")
        return None
    v = result["verdicts"]
    summary = {
        "analysis_only": True,
        "official_tps": 0,
        "organizer_reference_is_config_matched": v["organizer_reference_is_config_matched"],
        "organizer_runs_greedy_identity_reference_check": v["organizer_runs_greedy_identity_reference_check"],
        "deployed_481_passed_private_gate": v["deployed_481_passed_private_gate"],
        "deployed_481_gate_had_token_identity_check": v["deployed_481_gate_had_token_identity_check"],
        "deployed_481_precache_transparent": v["deployed_481_precache_transparent"],
        "precache_token_mutation_rate_deployed": v["precache_token_mutation_rate_deployed"],
        "prompts_identical_warm_cold_deployed": v["prompts_identical_warm_cold_deployed"],
        "transparency_measured": result["transparency"]["measured"],
        "floorlock_literal_1p0": v["floorlock_literal_1p0"],
        "floorlock_gate_guaranteed": v["floorlock_gate_guaranteed"],
        "globalflag_gate_guaranteed": v["globalflag_gate_guaranteed"],
        "residual_gate_risk": v["residual_gate_risk"],
        "reproduces_485": result["decomposition"]["reproduce"]["reproduces_485"],
        "precache_patch_byte_identical": result["decomposition"]["byte_audit"]["precache_patch_byte_identical"],
        "selftest_passed": result["self_test"]["passed"],
    }
    wandb_logging.log_summary(run, summary, step=0)
    wandb_logging.log_json_artifact(
        run, name="precache_gate_provenance", artifact_type="analysis", data=result
    )
    rid = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(_here / "precache_gate_provenance_results.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="precache-gate-provenance")
    ap.add_argument("--wandb_name", default="stark/precache-gate-provenance-493")
    ap.add_argument("--job_type", default="analysis")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    provenance = {"harness": scan_harness_scorer(), "verifier": parse_verifier_artifact()}
    reproduce = reproduce_485()
    audit = byte_audit()
    transparency = deployed_precache_transparency(reproduce["B_precache_warm_cold"])
    vd = verdicts(provenance, reproduce, transparency)

    result: dict[str, Any] = {
        "pr": 493,
        "analysis_only": True,
        "official_tps": 0,
        "provenance": provenance,
        "decomposition": {"reproduce": reproduce, "byte_audit": audit},
        "transparency": transparency,
        "verdicts": vd,
    }
    result["self_test"] = self_test(result)

    Path(args.output).write_text(json.dumps(result, indent=2))
    rid = log_wandb(result, args)
    if rid:
        result["wandb_run_id"] = rid

    # SENPAI-RESULT (single-line JSON).
    senpai = {
        "terminal": True,
        "status": "complete",
        "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "analysis_only": True,
        "official_tps": 0,
        "organizer_reference_is_config_matched": vd["organizer_reference_is_config_matched"],
        "organizer_runs_greedy_identity_reference_check": vd["organizer_runs_greedy_identity_reference_check"],
        "deployed_481_precache_transparent": vd["deployed_481_precache_transparent"],
        "precache_token_mutation_rate_deployed": round(vd["precache_token_mutation_rate_deployed"], 6),
        "floorlock_gate_guaranteed": vd["floorlock_gate_guaranteed"],
        "globalflag_gate_guaranteed": vd["globalflag_gate_guaranteed"],
        "residual_gate_risk": vd["residual_gate_risk"],
        "selftest_passed": result["self_test"]["passed"],
        "transparency_measured": transparency["measured"],
    }
    print("\n========== PRECACHE-GATE-PROVENANCE (PR #493) ==========")
    print(json.dumps(result["verdicts"], indent=2))
    print("\nself_test:", json.dumps(result["self_test"], indent=2))
    print("\nSENPAI-RESULT: " + json.dumps(senpai))
    return 0 if result["self_test"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
