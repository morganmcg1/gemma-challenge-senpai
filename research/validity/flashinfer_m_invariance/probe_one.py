#!/usr/bin/env python
"""FlashInfer M-invariance probe — ONE backend config per process (PR #582).

Analysis-only, LOCAL A10G. No HF Job, no served-file change.

The PR hypothesis: is FlashInfer's batch-1 (M=1) GEMV / sampler reduction
byte-exact M-invariant by construction (a free #319-safe identity lever)?

This process loads the base_fullhead no-spec stack (stock int4 QAT ckpt, full
262k native head, cudagraph, greedy temp=0) under ONE backend config and
measures:
  (A) M-invariance: a fixed target prompt is greedy-decoded at decode batch
      width M in {1,8,16} by co-scheduling M-1 distinct fillers (ignore_eos +
      fixed max_tokens => constant width M for every decode step). >=2 repeats
      per M give self-determinism. Per-position flip rate is computed on the
      TARGET token-id sequence across M.
  (B) wall_tps: single-stream (M=1) decode of output_len 512, warm median,
      prefill-corrected (decode-only, comparable to the 252.69 anchor).

Backend is selected by the CALLER via env set BEFORE this process starts:
  VLLM_ATTENTION_BACKEND      (unset => Gemma4 force-pin picks TRITON_ATTN)
  VLLM_USE_FLASHINFER_SAMPLER (0/1)
Local probe knobs:
  PROBE_OUT   output json path (required)
  PROBE_TAG   label for the config
  PROBE_N     M-invariance decode length (default 200)
  PROBE_TPS_N wall_tps decode length (default 512)
"""

import json
import os
import statistics
import sys
import time
import traceback

MODEL = os.environ.get("PROBE_MODEL", "/tmp/gemma4-e4b-qat-w4a16-ct")
OUT = os.environ["PROBE_OUT"]
TAG = os.environ.get("PROBE_TAG", "unknown")
N = int(os.environ.get("PROBE_N", "200"))
TPS_N = int(os.environ.get("PROBE_TPS_N", "512"))
WIDTHS = [1, 8, 16]
REPEATS = 2

# A fixed, deterministic target prompt that forces a multi-step greedy chain
# (a long reasoning chain maximises the chance of accumulating a near-tie flip).
TARGET = (
    "Solve this step by step. A train leaves city A at 60 km/h heading toward "
    "city B, 420 km away. Thirty minutes later a second train leaves city B "
    "heading toward city A at 90 km/h on a parallel track. At what distance "
    "from city A do they meet? Show every step of the arithmetic."
)

# Distinct fillers of comparable length (different content => realistic
# co-scheduled batch; NOT copies of the target).
FILLERS = [
    "Explain how a four-stroke internal combustion engine converts fuel into motion, naming each stroke.",
    "Write a short proof that the square root of two is irrational, stating each logical step.",
    "Describe the water cycle from evaporation through precipitation and runoff in full detail.",
    "Given a list of integers, explain an algorithm to find the two numbers that sum to a target value.",
    "Summarise the causes of the fall of the Western Roman Empire across political and economic factors.",
    "Walk through how photosynthesis turns sunlight, water and carbon dioxide into glucose and oxygen.",
    "Explain the difference between TCP and UDP and when an engineer would choose each protocol.",
    "Derive the quadratic formula by completing the square, showing every algebraic manipulation.",
    "Describe how a CPU executes a single machine instruction through fetch, decode and execute stages.",
    "Explain compound interest and compute the value of 1000 dollars at 5 percent over ten years.",
    "Outline the steps a compiler takes to turn source code into an executable binary program.",
    "Explain why the sky appears blue during the day and red near sunrise and sunset.",
    "Describe the process of cellular respiration and how ATP is produced in the mitochondria.",
    "Explain how public-key cryptography lets two strangers communicate securely over the internet.",
    "Walk through long division of 4096 by 17, showing each quotient digit and remainder.",
]

assert len(FILLERS) >= max(WIDTHS) - 1, "need enough distinct fillers for max width"

result = {
    "tag": TAG,
    "model": MODEL,
    "env": {
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "<unset>"),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER", "<unset>"
        ),
    },
    "widths": WIDTHS,
    "repeats": REPEATS,
    "decode_len": N,
    "tps_len": TPS_N,
    "load_ok": False,
}


def write(obj):
    with open(OUT, "w") as f:
        json.dump(obj, f, indent=2)


try:
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        max_num_seqs=max(WIDTHS),
        enable_prefix_caching=False,  # clean batch-width control (no cache confound)
        enforce_eager=False,  # cudagraph ON => matches the 252.69 anchor stack
        trust_remote_code=True,
        seed=0,
        disable_log_stats=True,
    )
    result["load_ok"] = True
    result["load_s"] = round(time.perf_counter() - t0, 2)

    # Record the attention backend vLLM actually resolved.
    try:
        vc = llm.llm_engine.vllm_config
        be = getattr(vc.attention_config, "backend", None)
        result["resolved_attention_backend"] = str(be)
    except Exception as e:  # noqa: BLE001
        result["resolved_attention_backend"] = f"<introspect-failed: {e}>"

    tok = llm.get_tokenizer()

    def templ(text):
        msgs = [{"role": "user", "content": text}]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    target_p = templ(TARGET)
    filler_p = [templ(f) for f in FILLERS]

    # ---- (A) M-invariance probe -------------------------------------------
    # target_ids[M][rep] = list[int] greedy token ids of the TARGET at width M
    target_ids = {M: [] for M in WIDTHS}
    for M in WIDTHS:
        for rep in range(REPEATS):
            prompts = [target_p] + filler_p[: M - 1]
            sp = SamplingParams(
                temperature=0.0,
                max_tokens=N,
                min_tokens=N,
                ignore_eos=True,  # force exactly N tokens => constant width M
                seed=0,
            )
            outs = llm.generate(prompts, sp, use_tqdm=False)
            # llm.generate preserves input order; index 0 is the target.
            ids = list(outs[0].outputs[0].token_ids)
            target_ids[M].append(ids)

    RAMP = 16  # ignore the first few decode steps (prefill ramp) for steady-state

    def flip_rate(a, b):
        L = min(len(a), len(b))
        if L == 0:
            return None, None
        diffs = [i for i in range(L) if a[i] != b[i]]
        first = diffs[0] if diffs else None
        return len(diffs) / L, first

    def steady_flip_rate(a, b):
        L = min(len(a), len(b))
        if L <= RAMP:
            return None
        diffs = sum(1 for i in range(RAMP, L) if a[i] != b[i])
        return diffs / (L - RAMP)

    # self-determinism per width (rep0 vs rep1)
    self_det = {}
    for M in WIDTHS:
        fr, first = flip_rate(target_ids[M][0], target_ids[M][1])
        self_det[M] = {
            "identical_frac": None if fr is None else round(1.0 - fr, 6),
            "first_divergence": first,
            "len": len(target_ids[M][0]),
        }

    # cross-width flips on rep0
    pairs = [(1, 8), (1, 16), (8, 16)]
    cross = {}
    for a, b in pairs:
        fr, first = flip_rate(target_ids[a][0], target_ids[b][0])
        sfr = steady_flip_rate(target_ids[a][0], target_ids[b][0])
        cross[f"M{a}_vs_M{b}"] = {
            "flip_rate": None if fr is None else round(fr, 6),
            "steady_flip_rate": None if sfr is None else round(sfr, 6),
            "first_divergence": first,
            "n_flips": None if fr is None else int(round(fr * len(target_ids[a][0]))),
        }

    byte_exact = all(
        cross[f"M{a}_vs_M{b}"]["flip_rate"] == 0.0 for a, b in pairs
    )
    self_det_min = min(
        sd["identical_frac"] for sd in self_det.values() if sd["identical_frac"] is not None
    )

    result["m_invariance"] = {
        "self_determinism": self_det,
        "cross_width": cross,
        "byte_exact_m_invariant": byte_exact,
        "self_det_min": self_det_min,
        "target_ids_head": {str(M): target_ids[M][0][:24] for M in WIDTHS},
    }

    # ---- (B) wall_tps (single-stream, decode-only, warm median) -----------
    def time_gen(n_tokens):
        sp = SamplingParams(
            temperature=0.0, max_tokens=n_tokens, min_tokens=n_tokens,
            ignore_eos=True, seed=0,
        )
        t = time.perf_counter()
        llm.generate([target_p], sp, use_tqdm=False)
        return time.perf_counter() - t

    time_gen(8)  # warmup (graph + caches hot)
    prefill_t = statistics.median(time_gen(1) for _ in range(3))  # TTFT proxy
    decode_tps_runs = []
    total_tps_runs = []
    for _ in range(3):
        full_t = time_gen(TPS_N)
        decode_t = max(full_t - prefill_t, 1e-9)
        decode_tps_runs.append((TPS_N - 1) / decode_t)
        total_tps_runs.append(TPS_N / full_t)

    result["tps"] = {
        "decode_tps_warm_median": round(statistics.median(decode_tps_runs), 4),
        "total_tps_warm_median": round(statistics.median(total_tps_runs), 4),
        "prefill_s": round(prefill_t, 5),
        "decode_tps_runs": [round(x, 4) for x in decode_tps_runs],
    }

    write(result)
    print(f"[probe_one:{TAG}] OK byte_exact={byte_exact} "
          f"self_det_min={self_det_min} "
          f"decode_tps={result['tps']['decode_tps_warm_median']}")
except Exception as e:  # noqa: BLE001
    result["error"] = f"{type(e).__name__}: {e}"
    result["traceback"] = traceback.format_exc()
    write(result)
    print(f"[probe_one:{TAG}] FAILED: {result['error']}", file=sys.stderr)
    sys.exit(3)
