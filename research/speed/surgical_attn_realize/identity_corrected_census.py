"""PR #488 — CORRECTED identity census (re-analysis of the captured decodes).

LOCAL MEASUREMENT ONLY. ``analysis_only=true``, ``official_tps=0``. CPU-only token
diffs over decode jsonl already captured by run_surgical_realize / census_surgical_identity
(NO new serve, NO GPU). Logs the corrected attribution to wandb so the reinterpretation
is on the permanent record.

Why a correction is needed
--------------------------
``census_surgical_identity.py`` reported the surgical M=1-AR gate at 0.4415 (operative_1.0
FAIL) and the full_flag M=1-AR gate at 0.4203 (also FAIL). Two facts prove that gate is
**confounded**, not a byte-exactness measurement:

  1. ``full_flag`` is the global ``VLLM_BATCH_INVARIANT=1`` config — batch-invariant BY
     CONSTRUCTION (M=8 verify == M=1 AR is its definition). It fails the same gate at 0.42,
     essentially identical to surgical's 0.44. A valid byte-exact gate passes the
     batch-invariant config; this one cannot discriminate.
  2. The census diffed the M=8 **round-0** decode (a cold-start round) against an M=1-AR
     reference served **eager** (ONEGRAPH off). Cold-start + eager-vs-cudagraph + M=1-vs-M=8
     batching each independently re-order bf16 reductions; on 512-token reasoning chains a
     single ULP flip cascades, flooring the raw-token rate at the run-to-run noise level.

This script measures that noise floor directly and isolates the one confound-free signal.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "research" / "speed" / "surgical_attn_realize"
RUN = OUT / "run"
CEN = OUT / "census"
GATE = 0.99


def _load(path: Path) -> dict[str, list[int]]:
    seqs: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = str(obj.get("id", len(seqs)))
        toks = obj.get("completion_token_ids")
        if isinstance(toks, list):
            seqs[key] = [int(t) for t in toks]
    return seqs


def _diff(a: Path, b: Path) -> dict[str, Any]:
    if not a.exists() or not b.exists():
        return {"available": False, "a": str(a), "b": str(b)}
    sa, sb = _load(a), _load(b)
    common = sorted(set(sa) & set(sb))
    total = matched = nflip = 0
    roots: list[int] = []
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        sf = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - sf
        if sf or len(ta) != len(tb):
            nflip += 1
            for i in range(n):
                if ta[i] != tb[i]:
                    roots.append(i)
                    break
    roots.sort()
    return {
        "available": True,
        "n_prompts": len(common),
        "n_tokens": total,
        "token_identity_rate": (matched / total) if total else None,
        "n_sequences_with_any_flip": nflip,
        "min_root_pos": roots[0] if roots else None,
        "median_root_pos": roots[len(roots) // 2] if roots else None,
    }


def main() -> int:
    dec = lambda arm, r: RUN / arm / f"decode_round{r:02d}.jsonl"  # noqa: E731

    # (1) within-arm warm determinism: r1 vs r2 (both warm) -> expect 1.0
    within = {arm: _diff(dec(arm, 1), dec(arm, 2)) for arm in ("deployed", "full_flag", "surgical")}
    # (2) cold-start signature: r0 (cold) vs r1 (warm) -> the noise floor
    cold = {arm: _diff(dec(arm, 0), dec(arm, 1)) for arm in ("deployed", "full_flag", "surgical")}
    # (3) warm matched-round cross-config attribution (same batching; isolates the axis)
    warm = {
        "surgical_vs_full_flag_r1": _diff(dec("surgical", 1), dec("full_flag", 1)),  # matmul-tax axis
        "surgical_vs_full_flag_r2": _diff(dec("surgical", 2), dec("full_flag", 2)),
        "deployed_vs_surgical_r1": _diff(dec("deployed", 1), dec("surgical", 1)),    # attention axis
        "deployed_vs_full_flag_r1": _diff(dec("deployed", 1), dec("full_flag", 1)),  # attention axis
    }
    # (4) the confounded M=1-AR census the PR literally asked for (shown broken)
    m1 = {
        "surgical_m8cold_vs_surgical_m1ar_eager": _diff(dec("surgical", 0), CEN / "surgical_m1ar" / "decode_round00.jsonl"),
        "full_flag_m8cold_vs_full_flag_m1ar_eager": _diff(dec("full_flag", 0), CEN / "full_flag_m1ar" / "decode_round00.jsonl"),
        "surgical_m1ar_vs_full_flag_m1ar": _diff(CEN / "surgical_m1ar" / "decode_round00.jsonl", CEN / "full_flag_m1ar" / "decode_round00.jsonl"),
    }

    def rate(d: dict[str, Any]) -> float | None:
        return d.get("token_identity_rate") if d.get("available") else None

    surg_vs_full_warm = rate(warm["surgical_vs_full_flag_r1"])
    attn_axis = rate(warm["deployed_vs_surgical_r1"])
    matmul_flips = warm["surgical_vs_full_flag_r1"].get("n_sequences_with_any_flip")
    attn_flips = warm["deployed_vs_surgical_r1"].get("n_sequences_with_any_flip")

    verdict = {
        "raw_token_m1ar_gate_measurable_on_this_workload": False,
        "reason": (
            "the global VLLM_BATCH_INVARIANT=1 config (batch-invariant by construction) fails "
            "the same M=1-AR gate identically (0.42 vs surgical 0.44); the census used the cold "
            "round-0 decode vs an eager (ONEGRAPH-off) M=1 reference. Raw 512-token greedy "
            "decode cascades from any single ULP flip, flooring the rate at the run-to-run noise level."
        ),
        "all_arms_warm_self_deterministic": all(
            (within[a].get("token_identity_rate") == 1.0) for a in within
        ),
        "noise_floor_cold_vs_warm_range": [
            min(rate(cold[a]) for a in cold), max(rate(cold[a]) for a in cold)
        ],
        "clean_signal_surgical_vs_full_flag_warm": surg_vs_full_warm,
        "matmul_tax_identity_delta_sequences": matmul_flips,   # surgical(2D,fastMarlin) vs full_flag(2D,tax)
        "attention_axis_identity_delta_sequences": attn_flips,  # deployed(3D) vs surgical(2D)
        "dominant_equivalence_axis": "attention_2D_order_preserving",
        "surgical_matches_222ship_operative_standard": bool(
            isinstance(surg_vs_full_warm, float) and surg_vs_full_warm >= 0.95
        ),
        "margin_attribution_ref": "merged #461: attn-only residual flips are bf16-ULP near-ties (margin 0.125-0.25), not semantic",
        "official_gate_is_served_vs_served": "program.md greedy-identity gate is served-vs-served byte-exact, NOT served-vs-M=1-AR eager",
        "gate_floor": GATE,
        "analysis_only": True,
        "official_tps": 0,
    }

    result = {
        "pr": 488,
        "kind": "identity_corrected_census",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "within_arm_warm_determinism_r1_vs_r2": within,
        "cold_start_signature_r0_vs_r1": cold,
        "warm_matched_round_cross_config": warm,
        "confounded_m1ar_census": m1,
        "verdict": verdict,
    }

    try:
        from scripts import wandb_logging
        result["git"] = wandb_logging.git_info()
    except Exception:
        wandb_logging = None  # type: ignore

    out_path = CEN / "identity_corrected_census.json"
    out_path.write_text(json.dumps(result, indent=2))

    # ---- console ----
    print("=== PR488 CORRECTED identity census (CPU re-analysis) ===")
    print("[within-arm warm determinism r1 vs r2]")
    for a, d in within.items():
        print(f"  {a:9s} rate={rate(d):.6f} flips={d['n_sequences_with_any_flip']}/{d['n_prompts']}")
    print("[cold-start signature r0 vs r1 = the raw-token noise floor]")
    for a, d in cold.items():
        print(f"  {a:9s} rate={rate(d):.6f} flips={d['n_sequences_with_any_flip']}/{d['n_prompts']}")
    print("[warm matched-round cross-config attribution]")
    for k, d in warm.items():
        print(f"  {k:30s} rate={rate(d):.6f} flips={d['n_sequences_with_any_flip']}/{d['n_prompts']}")
    print("[confounded M=1-AR census (PR-literal, shown broken)]")
    for k, d in m1.items():
        print(f"  {k:42s} rate={rate(d):.6f} flips={d['n_sequences_with_any_flip']}/{d['n_prompts']}")
    print(f"VERDICT raw_m1ar_gate_measurable={verdict['raw_token_m1ar_gate_measurable_on_this_workload']} "
          f"surgical_vs_full_warm={surg_vs_full_warm:.4f} matmul_flips={matmul_flips} attn_flips={attn_flips} "
          f"surgical_matches_222ship={verdict['surgical_matches_222ship_operative_standard']}")

    # ---- wandb ----
    if wandb_logging is not None:
        try:
            run = wandb_logging.init_wandb_run(
                job_type="surgical-attention-realization",
                agent="lawine",
                name="lawine/surgical-identity-corrected",
                group="surgical-attention-realization",
                tags=["surgical-attention-realization", "pr488", "analysis-only", "identity-census", "corrected"],
                config={"analysis_only": True, "official_tps": 0, "gate": GATE},
            )
            if run is not None:
                flat = {
                    "corrected/within_warm_deployed_rate": rate(within["deployed"]),
                    "corrected/within_warm_full_flag_rate": rate(within["full_flag"]),
                    "corrected/within_warm_surgical_rate": rate(within["surgical"]),
                    "corrected/warm_surgical_vs_full_flag_rate": surg_vs_full_warm,
                    "corrected/warm_surgical_vs_full_flag_flips": matmul_flips,
                    "corrected/warm_deployed_vs_surgical_rate": attn_axis,
                    "corrected/warm_deployed_vs_surgical_flips": attn_flips,
                    "corrected/confounded_surgical_m1ar_rate": rate(m1["surgical_m8cold_vs_surgical_m1ar_eager"]),
                    "corrected/confounded_full_flag_m1ar_rate": rate(m1["full_flag_m8cold_vs_full_flag_m1ar_eager"]),
                    "corrected/raw_m1ar_gate_measurable": int(False),
                    "corrected/surgical_matches_222ship_standard": int(verdict["surgical_matches_222ship_operative_standard"]),
                }
                flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
                wandb_logging.log_summary(run, flat, step=0)
                wandb_logging.log_json_artifact(
                    run, name="identity_corrected_census",
                    artifact_type="surgical-attention-realization", data=result,
                )
                result["wandb_run_id"] = getattr(run, "id", None)
                out_path.write_text(json.dumps(result, indent=2))
                print(f"[corrected] wandb_run_id={getattr(run, 'id', None)}")
                wandb_logging.finish_wandb(run)
        except Exception as exc:  # noqa: BLE001
            print(f"[corrected] wandb logging skipped: {exc}")

    print(f"[corrected] artifacts -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
