#!/usr/bin/env python
"""Offline re-synthesis for PR #584 — recompute the corrected Pareto verdict from
the SAVED served passes (no GPU / no re-serve), overwrite pareto_report.json, and
update the existing W&B run summary in place.

Fix: the headline `any_measured_drafter_clears_ship` / `best_ngram_projected_tps`
must use the CLEAN realized frame, not the anchor-inflated #573 frame (which
double-counts the spec benefit because 252.69 is the MTP-K7-SERVED number, not a
no-spec baseline). See build_pareto docstring.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:] = [p for p in sys.path if p not in ("", str(HERE))]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import the (now-corrected) build_pareto from the driver
spec = importlib.util.spec_from_file_location("pareto_driver", str(HERE / "pareto_driver.py"))
pd = importlib.util.module_from_spec(spec)
sys.modules["pareto_driver"] = pd
spec.loader.exec_module(pd)

WANDB_ENTITY = "wandb-applied-ai-team"
WANDB_PROJECT = "gemma-challenge-senpai"
WANDB_RUN_ID = "gd5s78ze"


def main():
    report = json.loads((HERE / "pareto_report.json").read_text())
    ref_rows = json.loads((HERE / "ref_pass0.json").read_text())["per_request"]

    ref = {"tps": {"warm_median_tps": report["ref_local_tps"]}}
    ngram_served = {int(k): v for k, v in report["ngram_served"].items()}
    mtp_served = {int(k): v for k, v in report["mtp_served"].items()}
    ngram_ks = report["ngram_ks"]
    mtp_ks = report["mtp_ks"]
    n_list = report["n_list"]
    warm = report["warm_discarded"]

    syn = pd.build_pareto(ref, ref_rows, ngram_served, mtp_served, ngram_ks, mtp_ks, n_list, warm)
    # carry forward fields main() added post-build
    self_det = report.get("ref_self_determinism", {})
    syn["ref_self_determinism_seq"] = pd._f(self_det.get("sequence_exact_rate"))
    syn["ref_self_determinism_tok"] = pd._f(self_det.get("token_identity_rate"))
    syn["peak_vram_gb"] = max(
        [pd._f((report.get("ngram_served", {}).get(str(k), {}) or {}).get("peak_vram_gb")) for k in ngram_ks]
        + [pd._f((report.get("mtp_served", {}).get(str(k), {}) or {}).get("peak_vram_gb")) for k in mtp_ks]
        + [18.8974609375]
    )
    report["synthesis"] = syn
    report["resynth_note"] = ("verdict recomputed on CLEAN realized frame; 573-frame demoted to "
                              "labeled INFLATED diagnostic (anchor 252.69 is MTP-K7-served, #575)")
    (HERE / "pareto_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    pd._print_pareto(syn)

    # ---- update the existing W&B run summary in place ----
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")
        scal = {f"summary/{k}": v for k, v in syn.items()
                if isinstance(v, (int, float, bool)) and (not isinstance(v, float) or math.isfinite(v))}
        scal["summary/primary_metric"] = syn["ngram_max_acceptance"]
        scal["resynth_corrected"] = True
        run.summary.update(scal)
        run.summary.update({})  # flush
        print(f"[resynth] updated W&B run {WANDB_RUN_ID} summary ({len(scal)} keys)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[resynth] W&B summary update failed: {exc!r} (report saved locally)", flush=True)

    # headline echo
    print("\n[resynth] HEADLINE (corrected):")
    for k in ("ngram_max_acceptance", "ngram_clears_268", "best_ngram_projected_tps",
              "best_ngram_projected_tps_573frame_INFLATED", "any_measured_drafter_clears_ship",
              "any_measured_drafter_clears_ship_573frame_INFLATED", "upper_left_corner_occupied",
              "upper_left_corner_literal_2_68_bar", "only_ngram_loadable", "a_ship_clean_at_ngram_cost"):
        print(f"    {k} = {syn.get(k)}")


if __name__ == "__main__":
    main()
