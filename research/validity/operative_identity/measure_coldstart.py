#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #599 (wirbel) -- cold-start C1 disambiguation, Q2 diagnostic. LOCAL, NO FIRE.

analysis_only=true, official_tps=0. No HF Job, no train.py --launch, no /v1/jobs:run,
no submission, no served-file change. Local serve + inference only on the assigned GPU.
The enable_prefix_caching toggle here is a DIAGNOSTIC INSTRUMENT to localize the C1
cold-start transient -- NOT a proposed change to any served/submission file.

WHAT THIS SETTLES (Q2)
----------------------
PR #588 (run n32yblfs) pinned the canonical operative-#319 bar and proved base_fullhead
passes it byte-for-byte at WARM steady state (b_vs_c GREEDY_IDENTICAL, 128/128, 0/65536
divergent), with the ONLY run-to-run nondeterminism a one-time first-pass cold-start
transient C1 (pass a dissents from both warm passes at 67 prompts, all <=4 ULP near-ties).
#588 attributed C1 to two legs:
  (1) prefix-cache cold-vs-warm chunked-prefill numerics (enable_prefix_caching=True), and
  (2) lazy Triton-JIT / FlashInfer-autotune kernel settling on first inference.

This driver re-runs the EXACT #588 R=3 free-running census (128x512, official harness)
with enable_prefix_caching=False to ISOLATE leg (1). Combined with #588 (the stock
prefix-ON baseline) this is a clean 2-config factorial on the prefix-caching dimension:

  | config (M=1, spec-OFF, temp=0) | a (cold) vs {b,c} | b vs c (warm/warm) |
  | stock, prefix ON  [#588]       | DIVERGENT (67p, <=4ULP) | GREEDY_IDENTICAL |
  | prefix OFF       [this run]    | ?                       | ? (expect IDENT) |

Readout (literal official check_greedy_identity.py, zero tolerance):
  - prefix_caching_off_collapses_C1 := (a == b == c) under prefix OFF, i.e. does removing
    the prefix-cache leg make even the LITERAL first pass byte-identical to warm? If pass a
    still dissents but b==c, the residual cold pass is the JIT/autotune leg (1 eliminated).
  - warmup_pass_collapses_C1 := (b == c): the two post-warmup passes agree (a acts as the
    warmup). Confirmed under stock by #588's b_vs_c; re-confirmed here under prefix OFF.
  - combined (prefix OFF + warmup) := this run's b_vs_c (b,c are prefix-off AND warmed).

Serve config = the #588 base_fullhead arm (stock int4 native-262k head + FA_SLIDING +
SURGICAL_ATTN_USE_3D_OFF 2D attention + PLE fold), MAX_NUM_SEQS=1, spec-OFF, plus the single
diagnostic toggle --no-enable-prefix-caching. Reuses the #588 module's validated decode +
official-verifier + near-tie classification helpers verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import measure_operative_319 as m  # the #588 harness module (helpers reused verbatim)

HERE = m.HERE
gid = m.gid


def start_server(log_path: Path, *, prefix_caching: bool) -> subprocess.Popen:
    """Identical to measure_operative_319.start_server, plus the diagnostic prefix-caching
    toggle. enable_prefix_caching defaults True in this vLLM build; --no-enable-prefix-caching
    forces every pass to prefill fresh (no cross-pass KV-block reuse), removing C1 leg (1)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["PYTHONPATH"] = str(m.SERVE_INJECT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env["PR557_PATCH_DIR"] = str(m.SUBMISSION)
    for k, v in m.BASE_FULLHEAD_ENV.items():
        env[k] = v
    cmd = [
        str(m.SERVER_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", m.STOCK, "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(m.PORT),
        "--dtype", "bfloat16", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "1",                      # DEPLOYED served geometry (M=1), spec-OFF
        "--trust-remote-code", "--disable-log-stats",
        "--override-generation-config", '{"temperature":0.0,"top_p":1.0,"top_k":0}',
    ]
    if not prefix_caching:
        cmd.append("--no-enable-prefix-caching")    # DIAGNOSTIC ONLY -- isolates C1 leg (1)
    print(f"[serve] base_fullhead M=1 spec-OFF prefix_caching={prefix_caching} "
          f"flags={m.BASE_FULLHEAD_ENV}", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-caching", choices=["on", "off"], default="off",
                    help="diagnostic toggle: 'off' adds --no-enable-prefix-caching to isolate C1 leg (1)")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--r-passes", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: 3 prompts, 24 tokens, 2 passes")
    args = ap.parse_args()

    prefix_on = (args.prefix_caching == "on")
    tagbase = "pon" if prefix_on else "poff"
    if args.smoke:
        m.NUM_PROMPTS, m.OUTPUT_LEN, m.R_PASSES = 3, 24, 2
        tagbase += "_smoke"
    else:
        m.NUM_PROMPTS, m.OUTPUT_LEN, m.R_PASSES = args.num_prompts, args.output_len, args.r_passes

    HERE.mkdir(parents=True, exist_ok=True)
    m.wait_gpu_free()
    log = HERE / f"server_{tagbase}_m1.log"
    proc = start_server(log, prefix_caching=prefix_on)

    result: dict = {
        "pr": 599,
        "agent": "wirbel",
        "card": "cold-start C1 disambiguation (Q2): isolate the prefix-cache leg of C1 by "
                "re-running the #588 R=3 census with enable_prefix_caching=False; compare to "
                "#588 stock (prefix ON) as the 2-config factorial baseline.",
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "no_submission": True,
        "official_tps": 0,
        "arm": f"base_fullhead prefix_caching={'ON' if prefix_on else 'OFF'} (DIAGNOSTIC toggle)",
        "diagnostic_toggle": ("(none -- stock)" if prefix_on else "--no-enable-prefix-caching"),
        "serve_geometry": "MAX_NUM_SEQS=1 (deployed served geometry), spec-OFF, greedy temp=0",
        "serve_env": m.BASE_FULLHEAD_ENV,
        "enable_prefix_caching": prefix_on,
        "official_decode_harness": str(m.DECODE_PY),
        "official_verifier": str(m.VERIFIER_DIR / "check_greedy_identity.py"),
        "prompt_suite": str(m.PROMPTS),
        "num_prompts": m.NUM_PROMPTS,
        "output_len": m.OUTPUT_LEN,
        "seed": m.SEED,
        "r_passes": m.R_PASSES,
        "eps_star_nat": m.EPS_STAR,
        "ulp_nat": m.ULP_NAT,
        "model_dir": os.path.realpath(m.STOCK),
        "build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
    }
    try:
        startup_s = m.wait_ready(proc)
        result["server_startup_s"] = round(startup_s, 1)
        print(f"[driver] base_fullhead M=1 prefix={args.prefix_caching} READY in {startup_s:.0f}s", flush=True)

        # R independent free-running greedy decode passes (a, b, c). base_fullhead is spec-OFF
        # at M=1 so each pass IS an independent plain greedy AR decode of the same int4 ckpt.
        tags = [f"{tagbase}_{chr(ord('a') + i)}" for i in range(m.R_PASSES)]
        passes = [m.decode_pass(t) for t in tags]
        result["peak_gpu_gb"] = m.peak_gpu_gb()
        result["passes"] = [{"tag": t, "path": p["_path"], "num_records": p["num_records"],
                             "num_completion_tokens": p["num_completion_tokens"],
                             "wall_s": round(p["_wall_s"], 1)} for t, p in zip(tags, passes)]

        # Pairwise: LITERAL (official check_greedy_identity.py, zero tolerance) + OPERATIVE
        # (near-tie classification of each prompt's first divergence). Pairs: a_vs_b, a_vs_c, b_vs_c.
        def lit(i: int, j: int) -> dict:
            rep = gid.compare_files(passes[i]["_path"], passes[j]["_path"])
            return {"pair": f"{chr(97+i)}_vs_{chr(97+j)}", "verdict": rep.verdict,
                    "num_identical": rep.num_identical, "num_prompts_compared": rep.num_prompts_compared,
                    "num_divergent": rep.num_divergent,
                    "total_divergent_tokens": rep.total_divergent_tokens,
                    "total_tokens_compared": rep.total_tokens_compared,
                    "self_determinism_token_rate": (1.0 - rep.total_divergent_tokens / rep.total_tokens_compared)
                    if rep.total_tokens_compared else None}

        pairs_literal, pairs_operative = {}, {}
        idx = [(0, 1), (0, 2), (1, 2)] if m.R_PASSES >= 3 else [(0, 1)]
        for i, j in idx:
            name = f"{chr(97+i)}_vs_{chr(97+j)}"
            pairs_literal[name] = lit(i, j)
            cls = m.classify_pair(passes[i]["_path"], passes[j]["_path"], name)
            cls.pop("details", None)  # keep summary compact; full details re-derivable from jsonl
            pairs_operative[name] = cls
            L = pairs_literal[name]
            print(f"[verify] {name}: LITERAL={L['verdict']} ident={L['num_identical']}/{L['num_prompts_compared']} "
                  f"div_tok={L['total_divergent_tokens']}/{L['total_tokens_compared']} | "
                  f"OPERATIVE all_near_tie={cls['all_first_div_near_tie']} "
                  f"semantic={cls['n_first_div_semantic']} max_gap_ulps={cls['max_first_div_gap_ulps']}", flush=True)

        result["pairwise_literal"] = pairs_literal
        result["pairwise_operative"] = pairs_operative

        # --- Verdicts ---
        warm_warm = pairs_literal.get("b_vs_c")
        cold_warm = [pairs_literal[k] for k in ("a_vs_b", "a_vs_c") if k in pairs_literal]
        warm_warm_identical = bool(warm_warm and warm_warm["verdict"] == "GREEDY_IDENTICAL")
        cold_pass_identical = bool(cold_warm and all(c["verdict"] == "GREEDY_IDENTICAL" for c in cold_warm))
        all_identical = bool(warm_warm_identical and cold_pass_identical)

        result["verdicts"] = {
            # does removing the prefix-cache leg make even the LITERAL first pass byte-identical?
            "prefix_caching_off_collapses_C1": (cold_pass_identical if not prefix_on else None),
            # do the two post-warmup passes agree? (a acts as warmup)
            "warmup_pass_collapses_C1": warm_warm_identical,
            # combined = prefix OFF + warmed (b,c) -> b_vs_c here
            "combined_prefix_off_plus_warmup_identical": (warm_warm_identical if not prefix_on else None),
            # strongest: ALL passes (incl literal pass 1) byte-identical under this config
            "all_three_passes_byte_identical": all_identical,
            "warm_warm_pair": "b_vs_c",
            "cold_warm_pairs": ["a_vs_b", "a_vs_c"],
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            pass

    out_name = f"coldstart_{tagbase}.json"
    (HERE / out_name).write_text(json.dumps(result, indent=2))
    v = result.get("verdicts", {})
    print(f"[driver] prefix_caching={args.prefix_caching} | "
          f"prefix_off_collapses_C1={v.get('prefix_caching_off_collapses_C1')} "
          f"warmup_collapses_C1={v.get('warmup_pass_collapses_C1')} "
          f"all_three_identical={v.get('all_three_passes_byte_identical')} -> {out_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
