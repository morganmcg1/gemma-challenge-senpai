#!/usr/bin/env python
"""PR #681 lawine -- int4-Marlin verify-WIDTH determinism, the VERSION axis.

DECISIVE QUESTION
-----------------
kanna #673 found that the compressed-tensors int4-Marlin GEMM is batch-WIDTH
sensitive on the dev307 venv: a width-(K+1)=6 spec-VERIFY forward produces a
different argmax than a width-1 AR forward at near-tie decode positions, flipping
the bonus token -> spec output diverges from greedy AR -> strict #319 fails. That
is the single blocker on the only live speed lever (spec-dec).

This card asks the version-axis complement: does that verify-width break reproduce
on the SHIP vLLM **0.22.0**, or only on dev307?  If 0.22.0's int4-Marlin GEMM is
width-invariant, the strict-#319 spec blocker is a dev307 numerics artifact, MOOT
on the version we actually ship -> spec lever unblocked (VERIFY_WIDTH_DEV307_ARTIFACT).
If both versions break above their controls, the width-sensitivity is intrinsic to
the int4-Marlin batched GEMM (VERIFY_WIDTH_VERSION_FUNDAMENTAL).

METHOD (pure forward-pass, NO spec scheduler)
---------------------------------------------
1. Capture REAL width-1 AR hidden states. Decode the official 128-prompt set
   (chat-templated, seed=1, greedy temp=0, ignore_eos -- byte-matching the official
   decode_outputs.py) ONE prompt at a time (max_num_seqs=1 -> every decode step is a
   genuine width-1 forward). A monkeypatch on model.compute_logits records the
   post-final-norm hidden each step feeds the int4-Marlin lm_head GEMM.
2. Replay each hidden through the SAME compute_logits at M in {1,2,4,6}. The M=6
   batch is the spec verify shape: row 0 = the bonus position's hidden h_t, rows 1-5
   = the next 5 AR hiddens h_{t+1..t+5} (a perfect drafter -- the conservative case:
   even a correct draft must verify, and if the width-6 forward flips the bonus
   argmax vs width-1, strict #319 breaks). Compare argmax of ROW 0 at M=6 vs M=1.

   For an exact GEMM, C[0,:] depends only on A[0,:] and B -- NEVER on rows 1-5. So a
   row-0 argmax flip is *proof* the Marlin kernel uses a width-dependent reduction
   /tiling (the #319 mechanism). We also record whether row-0 logits are BIT-identical
   at M=6 vs M=1 (the pure kernel width-invariance signal, independent of near-ties).

METRICS (per version)
---------------------
  verify_width_break_rate : frac(positions : argmax(M6 row0) != argmax(M1 row0))
  ar_vs_ar_break          : frac(positions : argmax(M1) != argmax(M1) on a 2nd call)
                            -- within-process determinism control, must be 0
  m1_matches_generated    : frac(positions : argmax(M1) == the token vLLM generated)
                            -- capture-alignment sanity, must be ~1.0
  bit_break_rate          : frac(positions : M6 row0 logits NOT torch.equal M1 row0)
  near-tie slice + onset  : break rate among small-top1-top2-gap positions; min/median
                            /max break position (cf. kanna onset 355-388/512)

LOCAL ONLY. analysis_only, official_tps=0, fires=0. NO HF Job / no --launch / served
file untouched. Runs identically on both venvs (selected by which python invokes it):
  0.22.0 ship : /tmp/senpai-venvs/20f658587e8a6643/bin/python
  dev307      : /tmp/senpai-venvs/5f4c623f772358a2/bin/python
Writes width_results_<tag>.json; log_wandb.py merges both versions to one W&B run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# --- container shims MUST precede torch/vllm import ------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")       # native argmax (lowest-index tie-break)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")    # in-process model -> monkeypatch applies
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "submissions" / "int4_g128_lmhead" / "model"

# import the OFFICIAL prompt loader/formatter by path so prompt selection + chat
# templating is byte-identical to the scored decode_outputs.py.
sys.path.insert(0, str(paths.DECODE_SCRIPT.parent))
import decode_outputs as dout  # noqa: E402

WIDTHS = [1, 2, 4, 6]          # 6 = K+1 (K=5) spec verify; sweep characterizes onset
K_VERIFY = 6


def get_model(llm):
    for p in (
        lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
    ):
        try:
            m = p()
            if m is not None:
                return m
        except Exception:
            continue
    raise RuntimeError("could not locate model_runner.model (need VLLM_ENABLE_V1_MULTIPROCESSING=0)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="version tag, e.g. v0220 or dev307")
    ap.add_argument("--num-prompts", type=int, default=24)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--near-tie-eps", type=float, default=1e-3,
                    help="post-softcap top1-top2 logit gap below which a position is a near-tie")
    ap.add_argument("--smoke", action="store_true", help="4 prompts x 64 tok wiring check")
    args = ap.parse_args()
    if args.smoke:
        args.num_prompts, args.output_len = 4, 64

    for note in paths.prepare_local_gpu_env():
        print(f"[width] {note}", flush=True)

    import torch
    from vllm import LLM, SamplingParams

    t0 = time.time()
    print(f"[width:{args.tag}] loading int4_g128_lmhead (auto-detect compressed-tensors) ...", flush=True)
    llm = LLM(
        model=str(MODEL_DIR),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
        trust_remote_code=True,
    )
    import vllm
    vllm_version = vllm.__version__
    print(f"[width:{args.tag}] load done in {time.time()-t0:.0f}s  vllm={vllm_version}", flush=True)

    model = get_model(llm)
    lm_head_kind = type(getattr(model, "lm_head", None)).__name__
    try:
        qmethod = type(getattr(getattr(model, "lm_head", None), "quant_method", None)).__name__
    except Exception:
        qmethod = "?"
    print(f"[width:{args.tag}] lm_head={lm_head_kind} quant_method={qmethod}", flush=True)

    orig_compute_logits = model.compute_logits
    captured: list[Any] = []
    rec = {"on": False}

    def hooked(hidden_states, *a, **k):
        if rec["on"]:
            captured.append(hidden_states.detach().clone())
        return orig_compute_logits(hidden_states, *a, **k)

    model.compute_logits = hooked

    def call_logits(h):
        try:
            return orig_compute_logits(h)
        except TypeError:
            return orig_compute_logits(h, None)

    # official prompt selection + chat templating (identical to scored decode) -------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.TOKENIZER)
    records = dout.read_sharegpt_prompts(paths.EVAL_PROMPTS, num_prompts=args.num_prompts, seed=paths.SEED)
    print(f"[width:{args.tag}] {len(records)} prompts; decoding {args.output_len} tok each (greedy, conc=1)", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True, detokenize=False)

    # running aggregates --------------------------------------------------------------
    n_pos = 0
    n_break = {m: 0 for m in WIDTHS if m != 1}
    n_bit_break = {m: 0 for m in WIDTHS if m != 1}
    n_ar_ar_break = 0
    n_m1_match_gen = 0
    near_tie_pos = 0
    near_tie_break6 = 0
    break6_positions: list[int] = []          # decode index of each M6 break (onset)
    per_prompt: list[dict[str, Any]] = []
    onset_first_break6: list[int] = []        # first M6 break index per prompt (kanna onset proxy)
    H = None
    peak_vram_gb = 0.0

    for pi, record in enumerate(records):
        prompt_ids = dout.encode_prompt(tok, record["prompt_text"])
        captured.clear()
        rec["on"] = True
        out = llm.generate({"prompt_token_ids": prompt_ids}, sampling_params=sp, use_tqdm=False)
        rec["on"] = False
        gen_ids = list(out[0].outputs[0].token_ids)

        if not captured:
            print(f"[width:{args.tag}] WARN prompt {pi}: no compute_logits captured -- skipping", flush=True)
            continue
        # trajectory of post-norm hiddens, one row per generated token, in order
        traj = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured], dim=0)  # [steps, H]
        if H is None:
            H = traj.shape[-1]
        steps = traj.shape[0]
        # index i predicts gen_ids[i]; align lengths defensively
        usable = min(steps, len(gen_ids))

        pp_break6 = 0
        pp_pos = 0
        pp_first_break6 = None
        last_t = usable - K_VERIFY
        for t in range(0, max(0, last_t)):
            h_t = traj[t:t + 1]                       # [1, H]
            batch6 = traj[t:t + K_VERIFY]             # [6, H], row0 == h_t
            lg1 = call_logits(h_t)                    # [1, V]
            a1 = int(lg1[0].argmax().item())
            # capture-alignment sanity: M1 replay argmax == token vLLM generated
            if t < len(gen_ids) and a1 == int(gen_ids[t]):
                n_m1_match_gen += 1
            # ar-vs-ar within-process determinism control
            lg1b = call_logits(h_t)
            if int(lg1b[0].argmax().item()) != a1:
                n_ar_ar_break += 1
            # near-tie characterization on the M1 logits (post-softcap)
            top2 = torch.topk(lg1[0], 2).values
            gap = float((top2[0] - top2[1]).item())
            is_near = gap < args.near_tie_eps
            if is_near:
                near_tie_pos += 1
            # width sweep argmax(row0)
            broke6 = False
            for m in (2, 4, 6):
                lgm = call_logits(traj[t:t + m])      # [m, V]
                am = int(lgm[0].argmax().item())
                if am != a1:
                    n_break[m] += 1
                    if m == 6:
                        broke6 = True
                if not torch.equal(lgm[0], lg1[0]):
                    n_bit_break[m] += 1
            if broke6:
                pp_break6 += 1
                break6_positions.append(t)
                if pp_first_break6 is None:
                    pp_first_break6 = t
                if is_near:
                    near_tie_break6 += 1
            pp_pos += 1
            n_pos += 1

        per_prompt.append({
            "index": pi, "id": record["id"], "steps": int(steps),
            "tested_positions": pp_pos, "m6_breaks": pp_break6,
            "first_m6_break_index": pp_first_break6,
        })
        if pp_first_break6 is not None:
            onset_first_break6.append(pp_first_break6)
        try:
            peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            pass
        print(f"[width:{args.tag}] prompt {pi+1}/{len(records)} id={record['id']} "
              f"pos={pp_pos} m6_breaks={pp_break6} "
              f"first_break={pp_first_break6} (cum rate={n_break[6]/max(1,n_pos):.5f})", flush=True)

    rate = {m: n_break[m] / max(1, n_pos) for m in n_break}
    bit_rate = {m: n_bit_break[m] / max(1, n_pos) for m in n_bit_break}
    result = {
        "pr": 681, "tag": args.tag, "analysis_only": True, "official_tps": 0, "fires": 0,
        "vllm_version": vllm_version, "model_dir": str(MODEL_DIR),
        "lm_head_kind": lm_head_kind, "quant_method": qmethod, "hidden_size": H,
        "config": {
            "num_prompts": len(records), "output_len": args.output_len, "seed": paths.SEED,
            "widths": WIDTHS, "k_verify": K_VERIFY, "near_tie_eps": args.near_tie_eps,
            "enforce_eager": True, "max_num_seqs": 1, "tie_break": "native-lowest-index",
        },
        "num_positions": n_pos,
        "verify_width_break_rate": rate.get(6),     # headline (M=6 = K+1 spec verify)
        "width_break_rate_by_M": rate,
        "bit_break_rate_by_M": bit_rate,
        "bit_break_rate": bit_rate.get(6),
        "ar_vs_ar_break": n_ar_ar_break / max(1, n_pos),
        "ar_vs_ar_break_count": n_ar_ar_break,
        "m1_matches_generated": n_m1_match_gen / max(1, n_pos),
        "m6_break_count": n_break[6],
        "near_tie_positions": near_tie_pos,
        "near_tie_fraction": near_tie_pos / max(1, n_pos),
        "near_tie_m6_break_count": near_tie_break6,
        "near_tie_m6_break_rate": near_tie_break6 / max(1, near_tie_pos),
        "onset_break6_min": min(break6_positions) if break6_positions else None,
        "onset_break6_median": int(statistics.median(break6_positions)) if break6_positions else None,
        "onset_break6_max": max(break6_positions) if break6_positions else None,
        "prompts_with_m6_break": sum(1 for p in per_prompt if p["m6_breaks"] > 0),
        "first_break_per_prompt_median": int(statistics.median(onset_first_break6)) if onset_first_break6 else None,
        "peak_vram_gb": peak_vram_gb,
        "per_prompt": per_prompt,
    }
    out_path = HERE / f"width_results_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 72, flush=True)
    print(f"[PR681 width:{args.tag}] vllm={vllm_version}  positions={n_pos}", flush=True)
    print(f"  verify_width_break_rate (M=6) = {result['verify_width_break_rate']:.6f} "
          f"({n_break[6]} flips)  bit_break_rate={result['bit_break_rate']:.6f}", flush=True)
    print(f"  width sweep break rate: " + "  ".join(f"M{m}={rate[m]:.6f}" for m in sorted(rate)), flush=True)
    print(f"  ar_vs_ar_break={result['ar_vs_ar_break']:.6f} (must be 0)  "
          f"m1_matches_generated={result['m1_matches_generated']:.4f} (must be ~1.0)", flush=True)
    print(f"  near-tie: {near_tie_pos} pos ({result['near_tie_fraction']:.4f}), "
          f"M6 break rate among near-ties={result['near_tie_m6_break_rate']:.4f}", flush=True)
    print(f"  onset M6 break idx: min={result['onset_break6_min']} "
          f"median={result['onset_break6_median']} max={result['onset_break6_max']} "
          f"(kanna dev307 onset 355-388/512)", flush=True)
    print(f"  prompts_with_m6_break={result['prompts_with_m6_break']}/{len(records)}  "
          f"peak_vram={peak_vram_gb:.1f} GB", flush=True)
    print(f"  -> {out_path}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
