#!/usr/bin/env python3
"""PR #545 terminal marker: base_fullhead GSM8K EOS-guard (min_tokens serve-floor)
as a SERVED ship default.

Consolidates the three legs of the validation:

  (1) GSM8K served-floor re-measure (chat endpoint, n=500, 8-shot, seed=1234,
      sampled PRIMARY + greedy), WITHOUT a request-level --min-tokens flag --
      the floor comes from the server (MIN_TOKENS_FLOOR=8). Source:
      truefullhead_servefloor_{sampled,greedy}.json (--save-text).

  (2) Single-stream TPS on the floor-active server, /v1/completions (ignore_eos,
      128x512 conc=1). Source: ../base_int4_floor_tps/report_base_fullhead_servefloor.json.
      Compared against the #541 no-guard floor (report_base_fullhead.json, run 56qyjxm1).

  (3) PPL + greedy-identity: UNCHANGED by construction. The floor patch is scoped
      to vllm/.../chat_completion/protocol.py:to_sampling_params (the /v1/chat path
      GSM8K uses). The /v1/completions path -- which carries TPS, PPL (prompt_logprobs)
      and the greedy-identity decode audit -- is byte-identical to the no-floor build
      (vllm/.../completion/protocol.py untouched). Verified at the source level below.

Local-only analysis. No HF job. analysis_only=true, official_tps=0.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # target/
sys.path.insert(0, str(ROOT))

# ---- anchors (public evidence used) -------------------------------------------------
BASE_ACC_SAMPLED = 0.878          # base gemma-4-E4B-it GSM8K sampled (gate denominator, #541)
BASE_ACC_GREEDY = 0.896           # base greedy (#541)
GATE_FRAC = 0.90                  # >=90% of base
GATE_SAMPLED = round(GATE_FRAC * BASE_ACC_SAMPLED, 4)   # 0.7902
GATE_GREEDY = round(GATE_FRAC * BASE_ACC_GREEDY, 4)     # 0.8064

# #541 no-floor base_fullhead (truefullhead arm) -- the FAIL we are rescuing
NOFLOOR_ACC_SAMPLED = 0.762
NOFLOOR_EMPTY_SAMPLED = 0.104     # extract_fail proxy == immediate-EOS / empty rate
NOFLOOR_ACC_GREEDY = 0.756
NOFLOOR_EMPTY_GREEDY = 0.148

# byte-exact safety anchors
NOGUARD_TPS_MEDIAN = 252.49052873786857   # report_base_fullhead.json, run 56qyjxm1
NOGUARD_TPS_AGG = 251.82733643849258
BASE_FULLHEAD_PPL = 2.006                  # fern #535 (base-int4 full head)

VENV_PURELIB = pathlib.Path(
    "/tmp/senpai-venvs/5f4c623f772358a2/lib/python3.12/site-packages"
)


def _load(p: pathlib.Path) -> dict:
    return json.loads(p.read_text())


def _verify_endpoint_scoping() -> dict:
    """Source-level proof that the floor is chat-only and the /v1/completions path
    is byte-identical to the no-floor build."""
    chat = VENV_PURELIB / "vllm/entrypoints/openai/chat_completion/protocol.py"
    comp = VENV_PURELIB / "vllm/entrypoints/openai/completion/protocol.py"
    chat_src = chat.read_text() if chat.exists() else ""
    comp_src = comp.read_text() if comp.exists() else ""
    chat_patched = "PR #545 serve-stack min_tokens floor" in chat_src and "min_tokens=(" in chat_src
    # the completion path must still carry the raw, unmodified assignment
    comp_raw = "min_tokens=self.min_tokens," in comp_src
    comp_untouched = "PR #545" not in comp_src
    return {
        "chat_protocol_patched": bool(chat_patched),
        "completion_protocol_raw": bool(comp_raw),
        "completion_protocol_untouched": bool(comp_untouched),
        "scoping_airtight": bool(chat_patched and comp_raw and comp_untouched),
        "chat_protocol_path": str(chat),
        "completion_protocol_path": str(comp),
    }


def main() -> int:
    # ---- (1) GSM8K served-floor --------------------------------------------------
    s = _load(HERE / "truefullhead_servefloor_sampled.json")
    g = _load(HERE / "truefullhead_servefloor_greedy.json")

    def empty_rate(d: dict) -> float:
        pp = d["per_problem"]
        return sum(1 for e in pp if not str(e.get("text", "")).strip()) / len(pp)

    served_acc_sampled = round(float(s["accuracy"]), 4)
    served_acc_greedy = round(float(g["accuracy"]), 4)
    served_empty_sampled = round(empty_rate(s), 4)
    served_empty_greedy = round(empty_rate(g), 4)
    # extract_fail_rate is the harness's own immediate-EOS/empty proxy; cross-check
    served_extract_fail_sampled = round(float(s["extract_fail_rate"]), 4)
    served_extract_fail_greedy = round(float(g["extract_fail_rate"]), 4)

    gate_pass_sampled = served_acc_sampled >= GATE_SAMPLED
    gate_pass_greedy = served_acc_greedy >= GATE_GREEDY

    # ---- (2) TPS -----------------------------------------------------------------
    tps = _load(HERE.parent / "base_int4_floor_tps" / "report_base_fullhead_servefloor.json")
    served_tps = round(float(tps["warm_median_tps"]), 4)
    served_tps_agg = round(float(tps["warm_aggregate_tps"]), 4)
    tps_delta = round(served_tps - NOGUARD_TPS_MEDIAN, 4)
    tps_delta_pct = round(100.0 * tps_delta / NOGUARD_TPS_MEDIAN, 4)
    tps_all_full_length = bool(tps["measured"]["tps"]["all_full_length"])
    # "free" = within run-to-run noise (|delta| < 1% of floor)
    tps_free = abs(tps_delta_pct) < 1.0 and tps_all_full_length

    # ---- (3) PPL + greedy-identity: by construction ------------------------------
    scope = _verify_endpoint_scoping()
    ppl_unchanged = bool(scope["scoping_airtight"])
    greedy_identity_unchanged = bool(scope["scoping_airtight"])

    # ---- top-line ----------------------------------------------------------------
    quality_safe_served = bool(
        gate_pass_sampled
        and served_empty_sampled == 0.0
        and tps_free
        and ppl_unchanged
        and greedy_identity_unchanged
    )

    marker = {
        "served_mintokens_gsm8k_acc": served_acc_sampled,
        "served_gate_pass": bool(gate_pass_sampled),
        "served_empty_rate": served_empty_sampled,
        "served_tps": served_tps,
        "tps_delta_vs_noguard": tps_delta,
        "ppl": BASE_FULLHEAD_PPL,
        "ppl_unchanged": ppl_unchanged,
        "greedy_identity_unchanged": greedy_identity_unchanged,
        "base_fullhead_gsm8k_quality_safe_served": quality_safe_served,
    }

    report = {
        "pr": 545,
        "analysis_only": True,
        "official_tps": 0,
        "marker": marker,
        "gsm8k": {
            "n_problems": int(s["n_problems"]),
            "n_shot": int(s["n_shot"]),
            "seed": int(s["seed"]),
            "client_min_tokens": None,          # floor came from the server, not the request
            "server_min_tokens_floor": 8,
            "sampled": {
                "accuracy": served_acc_sampled,
                "gate_threshold": GATE_SAMPLED,
                "gate_pass": bool(gate_pass_sampled),
                "pct_of_base": round(served_acc_sampled / BASE_ACC_SAMPLED, 4),
                "empty_rate": served_empty_sampled,
                "extract_fail_rate": served_extract_fail_sampled,
                "truncation_rate": round(float(s["truncation_rate"]), 4),
            },
            "greedy": {
                "accuracy": served_acc_greedy,
                "gate_threshold": GATE_GREEDY,
                "gate_pass": bool(gate_pass_greedy),
                "pct_of_base": round(served_acc_greedy / BASE_ACC_GREEDY, 4),
                "empty_rate": served_empty_greedy,
                "extract_fail_rate": served_extract_fail_greedy,
                "truncation_rate": round(float(g["truncation_rate"]), 4),
            },
            "noguard_541": {
                "sampled_acc": NOFLOOR_ACC_SAMPLED,
                "sampled_empty_rate": NOFLOOR_EMPTY_SAMPLED,
                "greedy_acc": NOFLOOR_ACC_GREEDY,
                "greedy_empty_rate": NOFLOOR_EMPTY_GREEDY,
            },
            "recovery": {
                "sampled_acc_delta": round(served_acc_sampled - NOFLOOR_ACC_SAMPLED, 4),
                "sampled_empty_delta": round(served_empty_sampled - NOFLOOR_EMPTY_SAMPLED, 4),
                "greedy_acc_delta": round(served_acc_greedy - NOFLOOR_ACC_GREEDY, 4),
            },
        },
        "tps": {
            "served_tps_warm_median": served_tps,
            "served_tps_warm_aggregate": served_tps_agg,
            "noguard_tps_warm_median": NOGUARD_TPS_MEDIAN,
            "noguard_tps_warm_aggregate": NOGUARD_TPS_AGG,
            "delta_vs_noguard": tps_delta,
            "delta_vs_noguard_pct": tps_delta_pct,
            "all_full_length": tps_all_full_length,
            "free": tps_free,
            "peak_vram_gb": round(float(tps["peak_vram_gb"]), 4),
            "wandb_run_id": tps.get("wandb_run_id"),
        },
        "ppl_greedy_by_construction": {
            "ppl": BASE_FULLHEAD_PPL,
            "ppl_source": "fern #535 (base-int4 full head, /v1/completions prompt_logprobs)",
            "ppl_unchanged": ppl_unchanged,
            "greedy_identity_unchanged": greedy_identity_unchanged,
            "basis": (
                "min_tokens floor patch is scoped to ChatCompletionRequest.to_sampling_params "
                "(/v1/chat/completions); the /v1/completions path that carries TPS, PPL "
                "(prompt_logprobs) and the greedy-identity decode audit is byte-identical to "
                "the no-floor build -- so PPL and greedy-identity cannot move."
            ),
            "endpoint_scoping": scope,
        },
        "public_evidence_used": {
            "base_acc_sampled": BASE_ACC_SAMPLED,
            "base_acc_greedy": BASE_ACC_GREEDY,
            "gate_frac": GATE_FRAC,
            "noguard_floor_run": "56qyjxm1",
            "gsm8k_gate_541": "bfvbueb1",
            "ppl_anchor": "fern #535",
        },
    }

    out = HERE / "servefloor_marker_545.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nSENPAI-MARKER " + json.dumps(marker), flush=True)
    print(f"\n[wrote] {out}", flush=True)

    # ---- wandb -------------------------------------------------------------------
    if "--no-wandb" not in sys.argv:
        try:
            from scripts.wandb_logging import (
                init_wandb_run,
                log_summary,
                log_json_artifact,
                finish_wandb,
            )

            run = init_wandb_run(
                job_type="downstream-quality",
                agent="wirbel",
                name="wirbel/base-fullhead-servefloor-marker",
                group="base-fullhead-gsm8k-layerdrop",
                tags=["gsm8k", "min-tokens-floor", "eos-guard", "base-fullhead", "pr545"],
                notes="PR #545 served min_tokens floor terminal marker (base_fullhead GSM8K quality-safe-served).",
                config={"pr": 545, "analysis_only": True, "official_tps": 0},
            )
            if run is not None:
                flat = {**marker,
                        "gsm8k_acc_greedy": served_acc_greedy,
                        "gate_pass_greedy": bool(gate_pass_greedy),
                        "served_tps_aggregate": served_tps_agg,
                        "tps_delta_vs_noguard_pct": tps_delta_pct,
                        "served_empty_rate_greedy": served_empty_greedy,
                        "noguard_acc_sampled": NOFLOOR_ACC_SAMPLED,
                        "noguard_empty_sampled": NOFLOOR_EMPTY_SAMPLED}
                log_summary(run, flat, step=0)
                log_json_artifact(run, name="servefloor_marker_545",
                                  artifact_type="quality-marker", data=report)
                finish_wandb(run)
                print(f"[wandb] logged run={run.id}", flush=True)
            else:
                print("[wandb] skipped (no API key / disabled)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] error (non-fatal): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
