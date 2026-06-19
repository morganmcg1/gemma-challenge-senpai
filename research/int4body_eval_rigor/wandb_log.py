"""Log the PR #693 decode-basis re-measure to W&B (analysis-only).

Reads verdict.json + the six per-basis result JSONs and logs the headline
scalars the launch has been missing -- eval_decode_basis / eval_sampling /
eval_min_tokens -- plus the measured int4-body AIME / gpqa_diamond numbers,
Wilson CIs, gate calls, and the GAP_* verdict. Run under .venv/bin/python
(serve venv has no wandb). wandb is wrapped so a logging failure never loses
the on-disk analysis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

WD = Path("/workspace/senpai/target/research/int4body_eval_rigor")


def basis_meta(row: dict) -> tuple[str, str, int]:
    """(eval_decode_basis, eval_sampling, eval_min_tokens) for a result row."""
    basis = row["basis"]
    if basis == "greedy":
        return ("greedy", "T=0.0,top_p=1.0,top_k=0", 0)
    # #31 generation_config.json sampling
    mt = 8 if basis.endswith("mintok8") else 0
    return ("generation_config_sampling", "T=1.0,top_p=0.95,top_k=64", mt)


def main() -> None:
    verdict = json.loads((WD / "verdict.json").read_text())
    rows = verdict["rows"]

    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] import failed ({e}); analysis is on disk at verdict.json")
        return

    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name="lawine/int4body-eval-rigor",
            group="int4body-eval-rigor-lawine",
            job_type="analysis",
            config={
                "pr": 693,
                "analysis_only": True,
                "official_tps": 0,
                "no_hf_job": 1,
                "fires": False,
                "model": verdict["model"],
                "serve_stack": verdict["serve_stack"],
                "gate_aime_2024": verdict["gates"]["aime_2024"],
                "gate_gpqa_diamond": verdict["gates"]["gpqa_diamond"],
            },
        )

        # Rich per-basis table with the explicit decode-basis columns.
        cols = ["bench", "basis", "eval_decode_basis", "eval_sampling",
                "eval_min_tokens", "metric", "n", "k_correct", "point",
                "wilson_lo", "wilson_hi", "gate", "gate_call"]
        tbl = wandb.Table(columns=cols)
        for r in rows:
            edb, esamp, emt = basis_meta(r)
            tbl.add_data(r["bench"], r["basis"], edb, esamp, emt, r["metric"],
                         r["n"], r["k_correct"], r["point"], r["wilson_lo"],
                         r["wilson_hi"], r["gate"], r["verdict"])
            # also flat scalars per arm (queryable without unpacking the table)
            pfx = f'{r["bench"]}__{r["basis"]}'
            run.summary[f"{pfx}__point"] = r["point"]
            run.summary[f"{pfx}__wilson_lo"] = r["wilson_lo"]
            run.summary[f"{pfx}__wilson_hi"] = r["wilson_hi"]
            run.summary[f"{pfx}__n"] = r["n"]
            run.summary[f"{pfx}__gate_call"] = r["verdict"]
            run.summary[f"{pfx}__eval_decode_basis"] = edb
            run.summary[f"{pfx}__eval_sampling"] = esamp
            run.summary[f"{pfx}__eval_min_tokens"] = emt
        run.log({"basis_table": tbl})

        s = run.summary
        s["analysis_only"] = True
        s["official_tps"] = 0
        s["no_hf_job"] = 1
        s["fires"] = False
        s["verdict"] = verdict["verdict"]
        s["aime_compliant"] = verdict["aime_compliant"]
        s["aime_compliant_basis"] = verdict["aime_compliant_basis"]
        s["aime_compliant_verdict"] = verdict["aime_compliant_verdict"]
        s["gpqa_compliant"] = verdict["gpqa_compliant"]
        s["gpqa_compliant_basis"] = verdict["gpqa_compliant_basis"]
        s["gpqa_compliant_verdict"] = verdict["gpqa_compliant_verdict"]
        s["aime_greedy_to_compliant_delta"] = verdict["aime_greedy_to_compliant_delta"]
        s["gpqa_greedy_to_compliant_delta"] = verdict["gpqa_greedy_to_compliant_delta"]
        run.finish()
        print(f"[wandb] logged run id={run.id} name={run.name}")
        (WD / "wandb_run_id.txt").write_text(run.id + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] logging failed ({e}); analysis is on disk at verdict.json")


if __name__ == "__main__":
    main()
