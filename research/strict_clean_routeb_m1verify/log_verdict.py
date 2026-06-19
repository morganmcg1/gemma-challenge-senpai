"""PR #746 route-b FINAL verdict run — logs the composed K-sweep + the decisive
structural verdict to the strict-clean-routeb-m1verify group.

Verdict: route-b (K-sequential M=1 verify) CANNOT be both byte-exact (G1-immune)
AND clear 126.378. Its M=1 single-query verify provides ZERO weight-read
amortization, so route_b_tps <= plain byte-exact AR. The fastest *genuinely*
byte-exact static config is int4_g128_lmhead AR ~= 126.4 local ~= the bar
(denken #740, MEASURED in research/ar_identity_safe_tps), so route-b < bar by the
no-amortization + drafter tax. Only the NON-byte-exact batched verify (#730 fire)
clears the bar (measured 28/128 greedy identity -> carries the G1 DQ route-b
exists to remove). G1-immunity and clearing the bar are mutually exclusive.

De-projection note (#642): the route-b ceiling is NOT the modeled 161.70 the
setup run cited. submissions/fa2sw_strict_m1ar_int4 manifest says "modeled 161.70
OFFICIAL TPS (lawine #438)" for the heavily-approximated dixie-flatline stack
(lm_head-prune 12k vocab, FA-sliding, fused argmax) — byte-exact WITHIN-stack, not
vs the canonical reference. The genuine byte-exact ceiling is MEASURED ~126.4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb  # noqa: F401  (win over any ./wandb shadow dir)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402

BAR = 126.378
TAU = 1.03524
# MEASURED genuine byte-exact static AR ceiling = int4_g128_lmhead (the bar's own
# config) from research/ar_identity_safe_tps (same pod, Jun 17). This is the
# route-b ceiling on the bar's base; ~= the bar itself (denken #740).
G128_LMHEAD_AR_LOCAL = 126.4  # ref/ref2/refS1/refS1b/fulldecode: 126.36-127.23


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> int:
    table = _load(HERE / "routeb_table_fastkern.json")
    ident = _load(HERE / "identity_partial.json")
    ar_bi = _load(HERE / "arref" / "arm_result.json")
    ar_fk = _load(HERE / "arref_fastkern" / "arm_result.json")

    run = wandb_logging.init_wandb_run(
        job_type="analysis",
        agent="stark",
        name="stark/routeb-m1verify-verdict",
        group="strict-clean-routeb-m1verify",
        notes="PR #746 FINAL: route-b cannot be both byte-exact AND clear 126.378. "
              "route_b_tps <= plain byte-exact AR (no M=1 amortization); byte-exact "
              "AR ceiling = int4_g128_lmhead ~=126.4 ~= bar (measured). Only "
              "non-byte-exact batched verify clears the bar (28/128 identity).",
        tags=["pr746", "route-b", "m1verify", "strict-byteexact", "analysis-only",
              "verdict", "negative-result"],
        config={
            "pr": 746,
            "base_submission": "int4_mtp_batchinv",
            "baseline_official_tps": BAR,
            "tau_local_to_official": TAU,
            "official_tps": 0,
            "analysis_only": True,
            "verdict": "route-b dominated: byte-exact => <= AR => < bar",
            "routeb_byte_exact_ceiling_local": ar_fk["wall_tps"],
            "byte_exact_static_ceiling_local_g128_lmhead": G128_LMHEAD_AR_LOCAL,
            "deprojection_note": "fa2sw 161.70 is MODELED OFFICIAL for an approximated "
                                 "stack, not a measured byte-exact local AR; real ceiling "
                                 "~=126.4 (int4_g128_lmhead, measured).",
        },
    )
    if run is None:
        print("[wandb] disabled / no API key — nothing logged", flush=True)
        return 1

    step = 0
    # --- anchors ---
    wandb.log({
        "global_step": step,
        "phase": 1,
        "anchor/ar_batchinv_wall_tps": ar_bi["wall_tps"],
        "anchor/ar_batchinv_official_proj": ar_bi["wall_tps"] * TAU,
        "anchor/ar_batchinv_ppl": ar_bi.get("ppl"),
        "anchor/ar_fastkern_wall_tps": ar_fk["wall_tps"],
        "anchor/ar_fastkern_official_proj": ar_fk["wall_tps"] * TAU,
        "anchor/ar_fastkern_ppl": ar_fk.get("ppl"),
        "anchor/byte_exact_ceiling_g128_lmhead_local": G128_LMHEAD_AR_LOCAL,
        "bar_official_tps": BAR,
    })

    # --- per-K rows: batched (measured) + route-b (projected, bounded by AR) ---
    for row in table["rows"]:
        step += 1
        wandb.log({
            "global_step": step,
            "k": row["k"],
            "accept_len": row["accept_len"],
            "emitted_per_step": (row["accept_len"] + 1.0) if row["accept_len"] else None,
            "batched/wall_tps": row["wall_tps_batched"],
            "batched/official_proj": row["wall_tps_batched_official_proj"],
            "batched/clears_bar": int(bool(row["batched_clears_bar"])),
            "routeb/tps_upper_local": row["routeb_tps_upper"],
            "routeb/tps_est_local": row["routeb_tps_est"],
            "routeb/upper_clears_bar": int(bool(row["routeb_upper_clears_bar"])),
            "byteexact_tax_batched_minus_ar": row["batched_minus_AR"],
        })

    # --- byte-exactness of batched verify (the crux: non-byte-exact) ---
    step += 1
    be = {f"identity/{k}": v for kk in ("batched_k2_vs_arref",
                                        "batched_k3_vs_arref",
                                        "batched_k4_vs_arref")
          for k, v in {
              f"{kk}_num_identical": ident[kk]["num_identical"],
              f"{kk}_rate": ident[kk]["identity_rate"],
              f"{kk}_onset_min": ident[kk]["onset_min"],
          }.items()}
    wandb.log({"global_step": step, **be})

    # --- summary scalars ---
    run.summary["status"] = "complete"
    run.summary["verdict"] = "route-b cannot clear 126.378 while byte-exact"
    run.summary["routeb_byte_exact_ceiling_local"] = ar_fk["wall_tps"]
    run.summary["routeb_best_upper_local_K2"] = table["rows"][0]["routeb_tps_upper"]
    run.summary["batched_best_local_K4"] = table["rows"][2]["wall_tps_batched"]
    run.summary["batched_k2_identity_rate"] = ident["batched_k2_vs_arref"]["identity_rate"]
    run.summary["bar_official_tps"] = BAR
    run.summary["primary_metric_routeb_upper_K2"] = table["rows"][0]["routeb_tps_upper"]

    wandb_logging.log_json_artifact(
        run, name="routeb_table_fastkern", artifact_type="analysis", data=table)
    wandb_logging.log_json_artifact(
        run, name="batched_byteexact_identity", artifact_type="analysis", data=ident)

    print(f"[wandb] route-b verdict run live: {run.id} ({run.url})", flush=True)
    wandb_logging.finish_wandb(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
