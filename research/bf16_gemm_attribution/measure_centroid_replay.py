"""Decide whether the int4head 17% bf16 slice (40x ampere_bf16, ~1.342GB read)
is a STEADY-STATE decode cost or a one-time centroid-graph CAPTURE artifact.

Attribution (settled from code + dims): the slice is the Gemma4 MTP drafter's
centroid-masking get_top_tokens. _setup_centroids_cuda_graphs captures graphs of
masked_emb.get_top_tokens(static_input[size, 2560], lm_head_weight) for sizes
[1,2,4,8,16,32,64]; _select_and_score gathers num_tokens x num_selected(4096)
rows x backbone_dim(2560) bf16 from the shared embed_tokens. At size=64 the
gather reads 64*4096*2560*2 = 1.342GB (== the "N=262144" fingerprint: 64*4096).

This patches Gemma4Proposer._greedy_sample to log, PER CALL and tagged by phase
(capture/warmup/decode), the token count T and which centroid graph size is
replayed -> the per-decode-step gather bytes. If decode replays small sizes
(T~1) the 1.342GB lived only in the one-time capture (artifact); if decode
replays size=64 every step it is a real steady-state bandwidth wall.

Run under the server venv (same env as profile_shapes_stack.py)."""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

OUT = ROOT / "research" / "bf16_gemm_attribution" / "centroid_replay.json"
NUM_SELECTED = 4096  # top_k(32) * vocab_per_centroid(262144/2048=128)


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[mc] {note}", flush=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_id = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g32_lmhead")
    drafter = os.environ.get("DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6"))

    import torch  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer  # noqa: E402

    phase = {"name": "init"}
    log = collections.defaultdict(lambda: {"calls": 0, "T_counts": collections.Counter(),
                                           "size_counts": collections.Counter(), "bytes": 0})

    _orig_greedy = Gemma4Proposer._greedy_sample

    def patched_greedy(self, hidden_states):
        rec = log[phase["name"]]
        T = int(hidden_states.shape[0])
        rec["calls"] += 1
        rec["T_counts"][T] += 1
        chosen = None
        if getattr(self, "_centroids_sizes", None):
            for size in self._centroids_sizes:
                if size >= T:
                    chosen = size
                    break
        if chosen is not None:
            hsz = self.model.masked_embedding.hidden_size
            rec["size_counts"][chosen] += 1
            rec["bytes"] += chosen * NUM_SELECTED * hsz * 2  # gather of bf16 rows
        else:
            rec["size_counts"]["eager_fallback"] += 1
        return _orig_greedy(self, hidden_states)

    Gemma4Proposer._greedy_sample = patched_greedy

    # time the one-time centroid-graph capture (runs inside load_model)
    _orig_setup = Gemma4Proposer._setup_centroids_cuda_graphs

    def patched_setup(self):
        phase["name"] = "capture"
        t0 = time.perf_counter()
        out = _orig_setup(self)
        dt = time.perf_counter() - t0
        hsz = self.model.masked_embedding.hidden_size
        # 3 warmup + 1 capture pass per size
        cap_bytes = sum(s * NUM_SELECTED * hsz * 2 * 4 for s in self._centroids_sizes)
        print(f"[mc] centroid-graph CAPTURE took {dt*1e3:.1f} ms, "
              f"~{cap_bytes/1e9:.3f} GB of gather reads (one-time)", flush=True)
        phase["name"] = "post_capture"
        return out

    Gemma4Proposer._setup_centroids_cuda_graphs = patched_setup

    print(f"[mc] building LLM (graphs ON, uniproc, K={num_spec}) ...", flush=True)
    llm = LLM(
        model=model_id, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.90, max_num_batched_tokens=512, max_num_seqs=1,
        trust_remote_code=True, enforce_eager=False,
        speculative_config={"model": drafter, "num_speculative_tokens": num_spec},
    )

    sp = SamplingParams(temperature=0.0, max_tokens=40, seed=1)
    phase["name"] = "warmup"
    _ = llm.generate(["Hello there, tell me about gravity."], sp)

    phase["name"] = "decode"
    t0 = time.perf_counter()
    _ = llm.generate(["Explain why the sky is blue in one detailed paragraph."], sp)
    decode_dt = time.perf_counter() - t0

    Gemma4Proposer._greedy_sample = _orig_greedy
    Gemma4Proposer._setup_centroids_cuda_graphs = _orig_setup

    out = {"num_selected": NUM_SELECTED, "decode_wall_s": decode_dt, "phases": {}}
    print("\n[mc] === per-phase _greedy_sample stats ===", flush=True)
    for name, rec in log.items():
        gb = rec["bytes"] / 1e9
        print(f"\n  phase={name}: calls={rec['calls']} centroid_gather={gb:.3f} GB", flush=True)
        print(f"     T distribution: {dict(rec['T_counts'])}", flush=True)
        print(f"     replayed sizes: {dict(rec['size_counts'])}", flush=True)
        out["phases"][name] = {
            "calls": rec["calls"], "gather_bytes": rec["bytes"],
            "T_counts": dict(rec["T_counts"]),
            "size_counts": {str(k): v for k, v in rec["size_counts"].items()},
        }

    dec = log["decode"]
    dec_gb = dec["bytes"] / 1e9
    print(f"\n[mc] DECODE centroid gather = {dec_gb:.3f} GB over {decode_dt:.2f}s "
          f"({dec_gb/max(decode_dt,1e-9):.1f} GB/s effective)", flush=True)
    print(f"[mc] verdict input: 114.75ms@~480GB/s ~= 0.055 GB equivalent of bf16 GEMM; "
          f"compare to decode steady-state centroid bytes above.", flush=True)

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"[mc] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
