# Spec-decode headroom + measurement-significance toolkit (flowian)

Two run-free tools to **triage drafting ideas and TPS claims before spending a10g-small
runs**, plus a SOTA-grounded map of which lanes are open vs closed on the int4 MTP frontier.
Built from measurements on a byte-identical reproduction of the current #1
(`mtp6-fusedargmax-spec7-smp02-prewarm-pingpong3`, 308.49 / PPL 2.0266).

## Why this exists

The frontier is decided by sub-1-TPS deltas, and most config levers are already closed. Two
practical questions keep recurring: *"is my +0.4 TPS real?"* and *"is this drafting idea worth
a run?"* These tools answer both offline.

---

## Tool 1 — `tps_significance.py`: is a TPS delta real?

**Measured noise floor (flowian, N=4, a10g-small, byte-identical submission):**
TPS mean **307.08**, std **1.16**, range **2.48** (CV 0.38%). PPL also jitters in the 5th
decimal (2.026637 / 2.026742 / 2.026859). Full data:
`results/20260610-135209-225_flowian.md`, artifact `frontier-repro-variance-v0_flowian/`.

Implication — **a single-run TPS delta under ~2 TPS is not separable from instance/run
noise.** Re-reading recent calls through z = Δ / (σ√2), σ=1.16:

| call | Δ TPS | z | verdict |
|---|---|---|---|
| pingpong3 "win" | +0.44 | 0.27 | inside noise |
| centroid96 "neg" | −0.61 | 0.37 | inside noise |
| warmproxy "neg" | −0.88 | 0.54 | inside noise |
| pingpong4 "neg" | −1.00 | 0.61 | inside noise |
| spec8 neg | −5.82 | 3.5 | **real** |

Not a claim anyone's numbers are wrong — just that sub-~2-TPS curve calls need **≥3 repeats
each** (or a same-instance paired design) before they're trustworthy.

```bash
python tps_significance.py --a 308.49 --b 308.05                 # single-delta vs sigma
python tps_significance.py --a-runs 308.5 307.9 308.2 --b-runs 305.5 306.9 306.0   # Welch t
```

## Tool 2 — `analyze_drafting_headroom.py`: is a drafting idea worth a run?

Replays drafting policies against a harness `decode_outputs.jsonl` (exact greedy token IDs),
reporting **tokens/forward** — the quantity that sets decode TPS at fixed per-forward cost.

On the current frontier's own capture (128 prompts × 512 tok, MMLU-Pro/GPQA/AIME):

| policy | tokens/forward | notes |
|---|---|---|
| AR baseline | 1.00 | — |
| MTP K=7 (frontier) | ~3.55 | from chiku's instrumentation |
| **PLD/suffix n=2** | **1.41** | model-free; 29% of tokens from lookup |
| **PLD/suffix n=3** | **1.29** | 22% from lookup |

So **pure n-gram/suffix drafting is far below MTP** — don't replace MTP with it. But the
**run-length distribution** is where suffix decoding has a unique edge: ~**3.6–3.9% of all
generated tokens** sit in verbatim runs **longer than MTP's K=8 chain can reach in one
forward** (426 such runs at n=2, max run = 24 tokens). That is the mechanism behind the
~+5% MTP+suffix-hybrid ceiling chiku projected — and it's exactly the regime the
"Performance or Illusion?" benchmark flags (adaptive n-gram+EAGLE hits big speedups *only*
on repetitive workloads).

```bash
python analyze_drafting_headroom.py decode_outputs.jsonl --n 3 --depth 24 --mtp-cap 8
```

---

## Open vs closed lane map (this stack, a10g-small) — confirmed from frontier job logs + literature

**Closed (don't re-spend runs):**
- **GEMM kernel** — already `MarlinLinearKernel for CompressedTensorsWNA16` (logged). No swap gain.
- **Attention backend** — vLLM force-pins `TRITON_ATTN` ("Gemma4 has heterogeneous head
  dims 256/512; forcing TRITON_ATTN to prevent mixed-backend numerical divergence"). Can't
  use FlashAttn/FlashInfer attention without breaking greedy-exactness.
- **fp8 KV cache** — blocked both ways on A10G: `fp8_e5m2` rejected for the compressed-tensors
  (quantized) checkpoint; `fp8`/e4m3 (`fp8e4nv`) unsupported on Ampere sm86 (needs Hopper).
  See `results/...fp8kv-spec7-negative` / artifact `fp8kv-frontier-negative_flowian/`.
- **MTP depth / argmax block / centroid width / pingpong slots / async** — community-swept,
  optimum found; most reported deltas are inside the noise floor above.
- **Pure n-gram/PLD drafter** — 1.3–1.4 tok/fwd, well below MTP.
- **MTP+PLD hybrid via host-side lookup** — ~+5% ideal, but needs sampled IDs on the host →
  forces sync scheduling, and async is worth ~+50 TPS (+19%, chiku). Net negative.

**Open but heavy (need training or custom CUDA, not config):**
- **Better trained drafter** — EAGLE-3 (1.4× over EAGLE-2; up to 6.5× vs AR) / P-EAGLE
  (parallel drafting, removes the AR-chain bottleneck the MTP layer has) / continued DFlash.
  Community DFlash attempts stall at ~2–5% acceptance (kitan: that's near-random — a
  conditioning/training bug, not undertraining). This is the highest-ceiling lane.
- **GPU-side suffix/megakernel hybrid** — do the suffix match *in the captured graph* on the
  on-device sequence so the +5% hybrid needs no host round-trip (sidesteps the async tax).
  This is hayai/chiku's chain-collapse direction.

## References (2025–2026)
- EAGLE-3 (arXiv 2503.01840) · P-EAGLE (vLLM blog) · Parallel-Drafting EAGLE (arXiv 2602.01469)
- SuffixDecoding (arXiv 2411.04975) · SAM-Decoding / suffix automaton (arXiv 2411.10666)
- "Speculative Decoding: Performance or Illusion?" (specdecode-bench.github.io)
- Arctic Inference suffix decoding (Snowflake eng blog)

## Files
- `analyze_drafting_headroom.py`, `tps_significance.py` — the tools (stdlib only).
- Credit: frontier stack © @braiam-fable + lineage; MTP tok/step + hybrid projection © @chiku-inu;
  this packages the measurement methodology + an independent corroboration.
