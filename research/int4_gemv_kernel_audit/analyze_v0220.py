#!/usr/bin/env python
"""Analyze the SHIP-version (vLLM 0.22.0) deterministic CONTROL for PR #675.

Pure replay over arms_v0220/{v0220_a,v0220_b}/decode_outputs.jsonl. Answers ONE
load-bearing question the dev307 sweep could not: is the SHIP environment (0.22.0,
the version that produced the live 126.378 official number) run-to-run
deterministic under the shipped serve config (no VLLM_BATCH_INVARIANT)?

  break_rate(v0220_a, v0220_b) == 0  -> ship env deterministic, #319 gate HOLDS
                                        there; dev307's ~0.90 is a local artifact.
  break_rate(v0220_a, v0220_b)  > 0  -> even the ship env is non-deterministic and
                                        #319 is at structural risk (a bigger find).

Also reports the cross-version break_rate (0.22.0 vs the dev307 base1 reference) for
context, and the 0.22.0 wall_tps reps vs the dev307 anchor.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMDIR = HERE / "arms_v0220"
DEV307_BASE1 = HERE / "arms" / "base1" / "decode_outputs.jsonl"
CHECK = ROOT / "submissions/int4_g128_lmhead/check_greedy_identity.py"


def _load_check():
    spec = importlib.util.spec_from_file_location("cgi_mod", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wall_tps(arm: str):
    rf = ARMDIR / arm / "arm_result.json"
    if not rf.exists():
        return None
    return json.loads(rf.read_text()).get("wall_tps")


def _cmp(cgi, ref: Path, cand: Path) -> dict:
    rep = cgi.compare(cgi.load_decode_outputs(ref), cgi.load_decode_outputs(cand))
    n = rep["num_prompts_compared"] or 1
    return {
        "verdict": rep["verdict"],
        "num_prompts_compared": rep["num_prompts_compared"],
        "num_divergent": rep["num_divergent"],
        "break_rate": rep["num_divergent"] / n,
        "total_tokens_compared": rep["total_tokens_compared"],
        "total_divergent_tokens": rep["total_divergent_tokens"],
        "token_break_rate": rep["total_divergent_tokens"] / (rep["total_tokens_compared"] or 1),
    }


def main() -> int:
    cgi = _load_check()
    a = ARMDIR / "v0220_a" / "decode_outputs.jsonl"
    b = ARMDIR / "v0220_b" / "decode_outputs.jsonl"
    out: dict = {
        "vllm_version": "0.22.0 (ship; venv 20f658587e8a6643)",
        "config": "shipped serve.py (NO VLLM_BATCH_INVARIANT), bf16, mml=4096, "
                  "gpu-util=0.90, mnbt=512, M=1 AR greedy, official 128x512",
        "wall_tps": {"v0220_a": _wall_tps("v0220_a"), "v0220_b": _wall_tps("v0220_b")},
    }
    if a.exists() and b.exists():
        out["ship_self_determinism"] = _cmp(cgi, a, b)
        out["ship_env_deterministic"] = out["ship_self_determinism"]["break_rate"] == 0.0
    else:
        out["ship_self_determinism"] = None
        out["ship_env_deterministic"] = None
    if a.exists() and DEV307_BASE1.exists():
        out["crossversion_v0220a_vs_dev307base1"] = _cmp(cgi, DEV307_BASE1, a)

    (HERE / "results_v0220.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    sd = out.get("ship_self_determinism") or {}
    print(f"\nSHIP(0.22.0) self break_rate = {sd.get('break_rate')}  "
          f"-> ship_env_deterministic = {out.get('ship_env_deterministic')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
