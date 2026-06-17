#!/usr/bin/env python
"""Log the PR #606 deterministic-gate + submission-parity analysis to W&B.

Analysis-only: reads the committed arm_result / compare JSONs and logs a compact
scalar summary + a parity table so the fleet has a queryable record. Run under
the repo .venv python (the serve venv has no wandb). See env_wandb_serve_venv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

OUT = Path("research/deterministic_gate_parity")


def load(p: str) -> dict:
    return json.loads((OUT / p).read_text())


def verdict_json(p: str) -> dict:
    # the *_crossstart / d1 files are the check_greedy_identity stdout (json + a
    # trailing VERDICT line); parse the leading json object.
    txt = (OUT / p).read_text()
    depth = 0
    end = 0
    for i, ch in enumerate(txt):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(txt[:end])


def _load_ids(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r.get("id"))] = r["completion_token_ids"]
    return out


def compare_decodes(ref_path: Path, cand_path: Path) -> tuple[int, int]:
    """Return (num_identical_prompts, num_prompts_compared) byte-exact."""
    a, b = _load_ids(ref_path), _load_ids(cand_path)
    keys = sorted(set(a) & set(b))
    ident = sum(1 for k in keys if a[k] == b[k])
    return ident, len(keys)


def main() -> None:
    ref_v22 = load("ref_v22/arm_result.json")
    ref2_v22 = load("ref2_v22/arm_result.json")
    dev307 = load("dev307_ppl/arm_result.json")
    dev307b = load("dev307_ppl2/arm_result.json")
    d1 = verdict_json("d1_ref_vs_ref2.json")          # 0.22.0 cross-start (gate)
    devx = verdict_json("dev307_crossstart.json")     # dev307 cross-start (same-attractor draw)

    # extra cross-comparisons that expose dev307's multi-attractor behaviour
    AR = Path("research/ar_identity_safe_tps")
    dev_cross_attractor = compare_decodes(             # #601 ref vs ref2 (A vs B)
        AR / "ref/decode_outputs.jsonl", AR / "ref2/decode_outputs.jsonl")
    dev307_vs_v0220 = compare_decodes(                 # dev307(B) vs 0.22.0 decode
        OUT / "dev307_ppl/decode_outputs.jsonl", OUT / "ref_v22/decode_outputs.jsonl")

    # official anchor (advisor-provided in PR #606 body)
    off_tps, off_ppl = 126.378, 2.019

    run = wandb.init(
        project="gemma-challenge-senpai",
        entity="wandb-applied-ai-team",
        name="lawine/deterministic-gate-submission-parity",
        group="deterministic-gate-parity",
        job_type="analysis",
        config={
            "pr": 606,
            "analysis_only": True,
            "official_tps": 0,
            "checkpoint": "submissions/int4_g128_lmhead/model",
            "serve_cmd": "canonical serve.py (api_server, dtype bf16, mml 4096, "
                         "gmu 0.90, mnbt 512, trust-remote-code)",
            "vllm_deterministic_build": "0.22.0 (manifest pin)",
            "vllm_dev_build": "0.22.1rc1.dev307+g3e8afdf78",
            "num_prompts": 128,
            "output_len": 512,
            "official_anchor_tps": off_tps,
            "official_anchor_ppl": off_ppl,
            "official_anchor_job": "6a2d5a96234ca64b60121aa5",
            "official_anchor_wandb": "905tbujn",
        },
    )

    def ident_frac(v: dict) -> float:
        return v["num_identical"] / v["num_prompts_compared"]

    summary = {
        # D1: deterministic gate restored on 0.22.0 (target 1.0 = 0/128 divergent)
        "v0220_crossstart_identical_frac": ident_frac(d1),
        "v0220_crossstart_num_divergent": d1["num_divergent"],
        "v0220_crossstart_tokens_divergent": d1["total_divergent_tokens"],
        # dev307 cross-start — bimodal/multi-attractor (single verdict is unreliable)
        "dev307_crossstart_sameattractor_identical": devx["num_identical"],   # 128 (lucky same-attractor draw)
        "dev307_crossstart_crossattractor_identical": dev_cross_attractor[0],  # 16 (#601 A-vs-B)
        "dev307_vs_v0220_decode_identical": dev307_vs_v0220[0],                # 0 (numerically different stack)
        # TPS parity (M=1 single-stream wall_tps)
        "tps_v0220_ref": ref_v22["wall_tps"],
        "tps_v0220_ref2": ref2_v22["wall_tps"],
        "tps_dev307": dev307["wall_tps"],
        "tps_dev307_b": dev307b["wall_tps"],
        "tps_official_anchor": off_tps,
        # PPL parity
        "ppl_v0220": ref_v22["ppl"],
        "ppl_dev307": dev307["ppl"],
        "ppl_dev307_b": dev307b["ppl"],
        "ppl_official_anchor": off_ppl,
        # gaps
        "tps_v0220_vs_official_pct": 100 * (ref_v22["wall_tps"] / off_tps - 1),
        "tps_dev307_vs_official_pct": 100 * (dev307["wall_tps"] / off_tps - 1),
        "ppl_v0220_vs_official_pct": 100 * (ref_v22["ppl"] / off_ppl - 1),
        "ppl_dev307_vs_official_pct": 100 * (dev307["ppl"] / off_ppl - 1),
        "ppl_dev307_vs_v0220_pct": 100 * (dev307["ppl"] / ref_v22["ppl"] - 1),
        "peak_vram_gib_estimate": 20.2,
    }
    run.summary.update(summary)

    tbl = wandb.Table(columns=["build", "vllm", "crossstart_identical", "wall_tps", "ppl"])
    tbl.add_data("0.22.0 (a)", "0.22.0", f"{d1['num_identical']}/128 (reliable)", ref_v22["wall_tps"], ref_v22["ppl"])
    tbl.add_data("0.22.0 ref2", "0.22.0", "", ref2_v22["wall_tps"], None)
    tbl.add_data("dev307 (b)", "dev307",
                 f"{devx['num_identical']}/128 same-attr; {dev_cross_attractor[0]}/128 cross-attr",
                 dev307["wall_tps"], dev307["ppl"])
    tbl.add_data("dev307 b2", "dev307", "", dev307b["wall_tps"], dev307b["ppl"])
    tbl.add_data("official (c)", "0.22.0", "128/128 VALID", off_tps, off_ppl)
    run.log({"parity_table": tbl})

    print(json.dumps(summary, indent=2))
    print(f"WANDB_RUN_ID={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
