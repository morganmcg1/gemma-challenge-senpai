#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #577 — materialize the 3 workload strata for the spec-dec dispersion leg.

Spec-dec speedup is workload-dependent: the cheap drafter's argmax tracks the
target on predictable text and diverges where the target is "thinking". To
measure the DISPERSION of acceptance + net-TPS we partition the prompt space
into 3 strata spanning the predictability spectrum (the land #556 spine), each
materialized here as a self-contained ShareGPT file the official
``decode_outputs.py`` consumes verbatim:

  EASY  -> ``data/private_proxy_native_code.json``        (boilerplate / templated
           code: high self-repetition, the most drafter-predictable arm; land
           #556 NL/code ~0.05% flip end).
  MIX   -> official 128 ``eval_prompts_sharegpt.json``    (the REAL benchmark mix:
           mmlu_pro 57 + gpqa_diamond 57 + aime2026 14; land #556 ~1.38% flip).
  HARD  -> ``shifted_reasoning_stem_128.json``            (pure "solve step by
           step" math/STEM CoT: the lowest-acceptance, DECISION-RELEVANT stratum
           where the quality gate lives; land #556 math-mix ~2.1% flip end).

Selection is done by the official ``read_sharegpt_prompts`` (seed=1) so the exact
prompts are pinned and reproduce the harness's own shuffle+slice. We write
``N_PER_STRATUM`` records per stratum back out in ShareGPT shape so the captured
stratum file is the audit artifact, not a moving upstream path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official" / "main_bucket" / "shared_resources"
                      / "speed_benchmark" / "scripts"))
import decode_outputs  # noqa: E402  (official selector — same shuffle as the harness)

OUT = ROOT / "research" / "validity" / "specdec_worstcase_dispersion"
N_PER_STRATUM = 48
SEED = 1

SOURCES = {
    "easy": ROOT / "data" / "private_proxy_native_code.json",
    "mix": ROOT / "official" / "main_bucket" / "shared_resources"
           / "speed_benchmark" / "data" / "eval_prompts_sharegpt.json",
    "hard": ROOT / "research" / "validity" / "drafter_shift_measurement"
            / "shifted_reasoning_stem_128.json",
}


def main() -> int:
    manifest = {"n_per_stratum": N_PER_STRATUM, "seed": SEED, "strata": {}}
    for name, src in SOURCES.items():
        records = decode_outputs.read_sharegpt_prompts(
            src, num_prompts=N_PER_STRATUM, seed=SEED
        )
        if len(records) != N_PER_STRATUM:
            raise SystemExit(
                f"stratum {name}: source {src} yielded {len(records)} valid "
                f"prompts, need {N_PER_STRATUM}"
            )
        # Re-emit in ShareGPT shape so decode_outputs.py can re-read the captured
        # file directly; keep a gpt placeholder turn so the >=2-turn filter holds.
        sharegpt = [
            {
                "id": r["id"],
                "conversations": [
                    {"from": "human", "value": r["prompt_text"]},
                    {"from": "gpt", "value": ""},
                ],
            }
            for r in records
        ]
        dst = OUT / f"stratum_{name}.json"
        dst.write_text(json.dumps(sharegpt, ensure_ascii=False, indent=0))
        manifest["strata"][name] = {
            "source": str(src.relative_to(ROOT)),
            "n": len(sharegpt),
            "file": str(dst.relative_to(ROOT)),
            "first_prompt_head": sharegpt[0]["conversations"][0]["value"][:120],
        }
        print(f"[strata] {name:5s} n={len(sharegpt)} <- {src.relative_to(ROOT)}",
              flush=True)
    (OUT / "strata_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[strata] wrote strata_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
