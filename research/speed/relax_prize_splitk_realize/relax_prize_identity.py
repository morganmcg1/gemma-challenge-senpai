#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Relax-prize IDENTITY COST arm (PR #452, stark). LOCAL A10G, analysis-only.
NO HF Job, NO submission, NO served-file change.

THE QUESTION (instruction #3)
-----------------------------
The relax lever (use_fp32_reduce=False, the FP-reassociating cross-split-K Marlin reduce)
is the only in-wheel served-numeric split-K knob. The speed arm (relax_prize_splitk_realize.py)
showed it nets ~0/-1 TPS end-to-end (the prize COLLAPSES) and BREAKS reduction-order
byte-exactness on 3/4 body GEMM shapes. But a GEMM-output bit-flip (~1e-3) does not
necessarily flip the ARGMAX token. This arm measures the deployed-faithful TOKEN-level cost:
  - byte-exact greedy identity fraction of relax vs strict (the served int4 argmax path)
  - token-flip count
  - PPL of relax vs strict (vs the <=2.42 gate; deployed anchor 2.3772)
If relax tokens == strict tokens within the run-to-run nondeterminism floor -> the relax
config is EFFECTIVELY STRICT at the token level (relax_prize_is_effectively_strict=True):
the (tiny, ~0) speed delta would then be greedy-SAFE -- flagged LOUD per the card.

DESIGN (3 isolated subprocess arms; the relax monkeypatch never leaks across arms)
---------------------------------------------------------------------------------
  strict   : served default (use_fp32_reduce=True). Reference greedy + PPL.
  strict2  : served default AGAIN, fresh process. The cross-process run-to-run
             nondeterminism FLOOR (int4+vLLM atomics are not bit-stable launch-to-launch;
             [[greedy-identity-validation]]). strict-vs-strict2 flips bound "noise".
  relax    : wrap apply_gptq_marlin_linear in the served caller namespace
             (kernels.linear.mixed_precision.marlin) to force use_fp32_reduce=False BEFORE
             the engine is built (the served apply_weights relies on the import-time-bound
             default arg, so the module global cannot be monkeypatched -- must wrap the
             caller's name). enforce_eager=True so the wrapped python fn is invoked each
             forward (no CUDA-graph baking of the original). A call counter proves it fired.

A relax flip count <= the strict-vs-strict2 floor => relax is indistinguishable from served
nondeterminism => EFFECTIVELY STRICT. Greedy + PPL on the 128-prompt official PPL set.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/relax_prize_splitk_realize/relax_prize_identity.py \
  --n-prompts 128 --gen-tokens 96
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Avoid the flashinfer sampling JIT (curand.h missing in tool-venv) -> use the native
# vLLM sampler; FLASH_ATTN backend + expandable segments match #381's working config.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_here = os.path.dirname(os.path.abspath(__file__))

# HF-cache snapshot FIRST: the bare-vLLM-loadable int4 checkpoint #381 used. /tmp/osoi5-v0-baked
# is PLE-folded -> bare vLLM raises a vocab AssertionError, so it is only a last-resort fallback.
MODEL_CANDIDATES = [
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
PPL_GATE = 2.42
PPL_ANCHOR = 2.3772     # deployed PPL (PR #52); strict arm should land near this


def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    raise FileNotFoundError(f"no int4 model among {MODEL_CANDIDATES}")


def apply_relax_patch():
    """Force use_fp32_reduce=False on the SERVED int4 GEMM path by wrapping the caller's
    bound name. apply_weights (kernels/linear/mixed_precision/marlin.py) calls the module
    global apply_gptq_marlin_linear with NO explicit use_fp32_reduce -> it takes the
    import-time-bound default (USE_FP32_REDUCE_DEFAULT=True); patching the marlin_utils
    module global is too late. Wrapping the name in the caller's namespace is honored."""
    import vllm.model_executor.kernels.linear.mixed_precision.marlin as kmarlin
    orig = kmarlin.apply_gptq_marlin_linear
    state = {"n": 0}

    def wrapped(*a, **kw):
        kw["use_fp32_reduce"] = False
        state["n"] += 1
        return orig(*a, **kw)

    kmarlin.apply_gptq_marlin_linear = wrapped
    return state


# --------------------------------------------------------------------------- #
def run_arm(arm, n_prompts, gen_tokens, ctx_cap, gpu_mem, out_path, max_num_seqs, max_batched_tokens):
    relax_state = apply_relax_patch() if arm == "relax" else None
    from vllm import LLM, SamplingParams

    model_dir = resolve_model_dir()
    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    contexts = [list(r["context_token_ids"])[:ctx_cap] for r in rows]
    fulls = [list(r["context_token_ids"]) + list(r["target_token_ids"]) for r in rows]
    n_ctx = [len(list(r["context_token_ids"])) for r in rows]
    maxlen = max(max(len(f) for f in fulls), max(len(c) for c in contexts) + gen_tokens) + 16

    # Memory bounds for the 24 GB A10G (served decode is single-stream M=8, not a 128-way batch):
    #  - max_num_seqs caps concurrent decode KV.
    #  - max_num_batched_tokens caps the chunked-prefill forward, which is what the teacher-forced
    #    PPL pass (prompt_logprobs=1) blows up: a full-vocab logprobs tensor over every scheduled
    #    prefill token. The default 8192 OOMs on the long (2943-token) PPL sequences. Chunked
    #    prefill is numerically equivalent, so PPL/greedy are unchanged; all arms share both knobs,
    #    so the strict/strict2/relax token-identity comparison stays apples-to-apples.
    t0 = time.time()
    llm = LLM(model=model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=maxlen, gpu_memory_utilization=gpu_mem, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=max_batched_tokens, enable_chunked_prefill=True,
              enable_prefix_caching=False, enforce_eager=True, trust_remote_code=True)
    print(f"[arm:{arm}] vLLM load {time.time()-t0:.0f}s model={model_dir} maxlen={maxlen} "
          f"max_num_seqs={max_num_seqs} max_batched_tokens={max_batched_tokens}", flush=True)

    # --- greedy continuation (M=1 AR argmax under THIS kernel config) ---
    sp_gen = SamplingParams(temperature=0.0, max_tokens=gen_tokens, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": c} for c in contexts], sp_gen, use_tqdm=False)
    greedy = [list(o.outputs[0].token_ids) for o in outs]

    # --- teacher-forced PPL over the target span (the official PPL set) ---
    sp_ppl = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    outs2 = llm.generate([{"prompt_token_ids": f} for f in fulls], sp_ppl, use_tqdm=False)
    nll_sum = 0.0
    ntok = 0
    per = []
    for i, o in enumerate(outs2):
        plp = o.prompt_logprobs or []
        full = fulls[i]
        s = 0.0
        n = 0
        for t in range(n_ctx[i], len(full)):
            if t < len(plp) and plp[t]:
                tok = full[t]
                entry = plp[t]
                if tok in entry:
                    s += -float(entry[tok].logprob)
                    n += 1
        nll_sum += s
        ntok += n
        per.append({"idx": i, "nll": s, "n": n})
    ppl = math.exp(nll_sum / ntok) if ntok else float("nan")

    dump = {"arm": arm, "n_prompts": len(rows), "gen_tokens": gen_tokens, "ctx_cap": ctx_cap,
            "greedy": greedy, "ppl": ppl, "nll_sum": nll_sum, "n_ppl_tokens": ntok,
            "ppl_per_prompt": per,
            "relax_marlin_calls": (relax_state["n"] if relax_state else 0),
            "patch_applied": bool(arm == "relax")}
    with open(out_path, "w") as fh:
        json.dump(dump, fh)
    print(f"[arm:{arm}] ppl={ppl:.4f} ntok={ntok} relax_marlin_calls={dump['relax_marlin_calls']} "
          f"-> {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
def compare(ref, cand):
    """Token-level greedy comparison cand vs ref. Reports sequence-identity fraction,
    token-identity fraction, total flips, first-divergence stats."""
    ga, gb = ref["greedy"], cand["greedy"]
    n = min(len(ga), len(gb))
    n_identical = n_diverged = 0
    total_pos = match_pos = 0
    first_flips = []
    per_prompt = []
    for i in range(n):
        sa, sb = ga[i], gb[i]
        L = min(len(sa), len(sb))
        diff = [j for j in range(L) if sa[j] != sb[j]]
        identical = (not diff) and (len(sa) == len(sb))
        if identical:
            n_identical += 1
        else:
            n_diverged += 1
        ff = diff[0] if diff else (L if len(sa) != len(sb) else None)
        if ff is not None:
            first_flips.append(ff)
        total_pos += L
        match_pos += sum(1 for j in range(L) if sa[j] == sb[j])
        per_prompt.append({"idx": i, "n_tokens": L, "n_flips": L - sum(1 for j in range(L) if sa[j] == sb[j]),
                           "identical": identical, "first_flip_pos": ff})
    return {"n_prompts": n, "n_identical": n_identical,
            "seq_identity_frac": n_identical / n if n else float("nan"),
            "n_diverged": n_diverged,
            "token_identity_frac": match_pos / total_pos if total_pos else float("nan"),
            "total_token_flips": total_pos - match_pos, "total_positions": total_pos,
            "median_first_flip": (statistics.median(first_flips) if first_flips else None),
            "per_prompt": per_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["strict", "strict2", "relax"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--ctx-cap", type=int, default=384)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-batched-tokens", type=int, default=1024)
    ap.add_argument("--reuse-arms", action="store_true",
                    help="skip re-spawning the GPU arms if _arm_*.json already exist; just "
                         "recompute the (CPU-only) token comparison + self-test from them")
    ap.add_argument("--output", default=os.path.join(_here, "relax_prize_identity.json"))
    ap.add_argument("--wandb_group", default="relax-equivalence-prize")
    ap.add_argument("--wandb_name", default="stark/relax-prize-identity")
    args = ap.parse_args()

    # ---- subprocess arm mode ----
    if args.arm is not None:
        assert args.out, "--out required in arm mode"
        return run_arm(args.arm, args.n_prompts, args.gen_tokens, args.ctx_cap, args.gpu_mem,
                       args.out, args.max_num_seqs, args.max_batched_tokens)

    # ---- parent: spawn the 3 arms, isolated processes ----
    tmp = {a: os.path.join(_here, f"_arm_{a}.json") for a in ("strict", "strict2", "relax")}
    arms_present = all(os.path.exists(tmp[a]) for a in ("strict", "strict2", "relax"))
    if args.reuse_arms and arms_present:
        print("[parent] --reuse-arms: skipping GPU arms, recomputing comparison from "
              f"existing {list(tmp.values())}", flush=True)
    else:
        for a in ("strict", "strict2", "relax"):
            cmd = [sys.executable, os.path.abspath(__file__), "--arm", a, "--out", tmp[a],
                   "--n-prompts", str(args.n_prompts), "--gen-tokens", str(args.gen_tokens),
                   "--ctx-cap", str(args.ctx_cap), "--gpu-mem", str(args.gpu_mem),
                   "--max-num-seqs", str(args.max_num_seqs),
                   "--max-batched-tokens", str(args.max_batched_tokens)]
            print(f"[parent] launching arm {a}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            print(f"[parent] arm {a} rc={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
            if r.returncode != 0:
                print(f"[parent] arm {a} FAILED", flush=True)
                return 1

    dumps = {a: json.load(open(tmp[a])) for a in ("strict", "strict2", "relax")}
    floor = compare(dumps["strict"], dumps["strict2"])      # cross-process strict noise floor
    relax_cmp = compare(dumps["strict"], dumps["relax"])    # the relax effect

    ppl_strict = dumps["strict"]["ppl"]
    ppl_strict2 = dumps["strict2"]["ppl"]
    ppl_relax = dumps["relax"]["ppl"]
    relax_calls = dumps["relax"]["relax_marlin_calls"]

    # relax is EFFECTIVELY STRICT if its flips vs strict do not exceed the strict-vs-strict
    # cross-process nondeterminism floor (indistinguishable from served run-to-run noise)
    effectively_strict = bool(relax_cmp["total_token_flips"] <= floor["total_token_flips"])
    relax_zero_flips = bool(relax_cmp["total_token_flips"] == 0)
    # PPL noise floor: strict vs a fresh strict process (cross-run int4-atomics nondeterminism)
    ppl_strict_noise_floor = abs(ppl_strict2 - ppl_strict)
    relax_ppl_delta = ppl_relax - ppl_strict
    PPL_DELTA_MATERIALITY = 0.01      # 0.4% of the 2.42 gate; below this relax is PPL-neutral

    st = {}
    st["patch_fired"] = bool(relax_calls > 0)               # the relax wrap was actually invoked
    st["ppl_strict_near_anchor"] = bool(abs(ppl_strict - PPL_ANCHOR) < 0.5)
    # NB: this is a deployed-FAITHFUL proxy (bare vLLM on the raw QAT checkpoint, not the
    # PLE-folded served fa2sw stack), so the absolute proxy PPL sits a bit above the served
    # 2.3772 / 2.42 gate. The served gate is validated in PR #52 and is untouched by this
    # analysis-only card. The deployed-faithful identity-COST question is whether the relax
    # reduce DEGRADES PPL vs strict under the SAME proxy -> gate on the relax-vs-strict delta,
    # not on the proxy absolute vs the served gate.
    st["relax_ppl_not_worse_than_strict"] = bool(relax_ppl_delta <= PPL_DELTA_MATERIALITY)
    st["floor_finite"] = bool(math.isfinite(floor["token_identity_frac"]))
    st["relax_finite"] = bool(math.isfinite(relax_cmp["token_identity_frac"]))
    st["n_prompts_ok"] = bool(relax_cmp["n_prompts"] == args.n_prompts)
    self_test_passes = all(st.values())

    verdict = {
        "relax_prize_identity_self_test_passes": self_test_passes,
        "relax_prize_identity_fraction": relax_cmp["token_identity_frac"],
        "relax_prize_seq_identity_fraction": relax_cmp["seq_identity_frac"],
        "relax_prize_token_flips": relax_cmp["total_token_flips"],
        "relax_prize_total_positions": relax_cmp["total_positions"],
        "relax_prize_n_diverged_prompts": relax_cmp["n_diverged"],
        "relax_prize_median_first_flip": relax_cmp["median_first_flip"],
        "strict_noise_floor_token_flips": floor["total_token_flips"],
        "strict_noise_floor_identity_fraction": floor["token_identity_frac"],
        "strict_noise_floor_n_diverged": floor["n_diverged"],
        "relax_prize_is_effectively_strict": effectively_strict,
        "relax_prize_zero_flips": relax_zero_flips,
        "relax_prize_ppl": ppl_relax, "ppl_strict": ppl_strict, "ppl_strict2": ppl_strict2,
        "relax_prize_ppl_delta": relax_ppl_delta,
        "ppl_strict_noise_floor": ppl_strict_noise_floor,
        "relax_ppl_within_noise_floor": bool(abs(relax_ppl_delta) <= max(ppl_strict_noise_floor, PPL_DELTA_MATERIALITY)),
        "ppl_strict_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
        # informational only: this proxy (raw QAT checkpoint, not served PLE-folded stack) runs
        # a touch above the served gate; the served stack's gate compliance (2.3772) is unaffected.
        "proxy_ppl_strict_under_served_gate": bool(ppl_strict <= PPL_GATE),
        "proxy_ppl_relax_under_served_gate": bool(ppl_relax <= PPL_GATE),
        "is_deployed_faithful_proxy": True,
        "relax_marlin_calls": relax_calls,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
        "official_tps": 0,
        "self_test_conditions": st,
    }
    payload = {
        "config": {"n_prompts": args.n_prompts, "gen_tokens": args.gen_tokens,
                   "ctx_cap": args.ctx_cap, "model_dir": resolve_model_dir(),
                   "note": "3 isolated subprocess arms (strict/strict2/relax); relax wraps "
                           "apply_gptq_marlin_linear to force use_fp32_reduce=False on the served "
                           "int4 path; enforce_eager; greedy + teacher-forced PPL on the 128-prompt "
                           "official PPL set. No serve change, no HF Job, no submission."},
        "verdict": verdict,
        "identity": relax_cmp, "noise_floor": floor,
        "arms": {a: {k: v for k, v in dumps[a].items() if k != "greedy" and k != "ppl_per_prompt"}
                 for a in dumps},
    }
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n[parent] wrote {args.output}", flush=True)
    print(f"[parent] relax vs strict: identity_frac={relax_cmp['token_identity_frac']:.6f} "
          f"flips={relax_cmp['total_token_flips']} (floor strict-vs-strict2 flips="
          f"{floor['total_token_flips']})  effectively_strict={effectively_strict}", flush=True)
    print(f"[parent] ppl_strict={ppl_strict:.4f} (anchor {PPL_ANCHOR}) ppl_relax={ppl_relax:.4f} "
          f"delta={ppl_relax-ppl_strict:+.4f}  relax_marlin_calls={relax_calls}", flush=True)
    print(f"[parent] self_test={self_test_passes} {st}", flush=True)
    return 0 if self_test_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
