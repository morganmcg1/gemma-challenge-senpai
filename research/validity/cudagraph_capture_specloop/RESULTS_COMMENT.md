STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["qlvakiyu"],"primary_metric":{"name":"host_overhead_frac","value":0.004555},"test_metric":{"name":"ppl","value":2.3778}}

## Results — MEASURED-ABORT: the K=7/M=8 spec-decode loop is ALREADY CUDA-graph-captured

**TL;DR.** The hypothesis premise is false on the served stack: the entire K=7 draft loop is already
replayed as **one** CUDA graph (`ONEGRAPH=1`). Per-step host/CPU overhead OUTSIDE the graph is **0.46%**
of the cycle — **below the 0.5% self-abort gate (instr. 2)** → I banked the negative and did **not**
prototype a new capture. **This PR adds 0 TPS** (the lever is inside the deployed numbers;
`crosses_deployed_481 = False`). As a bonus I measured the launch overhead the deployed loop-graph
**already** recovers (the number the advisor asked me to flag for land #444): **67 µs/step → +2.80 TPS
local, −3.93 TPS on the 467.14 frontier if graphs are forced OFF.**

Empirical capture proof from the served arm's engine log:
`[onegraph] captured K=7 width-1 propose graph at eligible call 21 with slots=3`. Code path:
`ONEGRAPH=1` → `_capture_graph` records `_run_graph_body` (7 width-1 drafter forwards + fused-sparse-argmax
+ hidden copies); hot path is a single `graph.replay()` (`propose_onegraph`); paged-KV block-table
refresh stays OUTSIDE the graph (`_refresh_static_buffers`) — exactly the boundary the PR requires.

### 1. Self-abort gate (instr. 2) — FIRES

| metric (graph-on, deployed) | value |
|---|---|
| decode wall / step (STEPTIME p50, host-to-host) | 7903 µs |
| GPU-busy (verify 6441 + drafter 1426) | 7867 µs |
| **host/CPU overhead (wall − GPU-busy)** | **36 µs = 0.46%** |
| GPU-busy share | 99.5% |
| self-abort gate | 0.5% → **self_abort = True** |

Reproduces ubel #284's host-overhead read (0.50% at out=512; this card confirms **0.46% at the PR's
128→128 point**, |Δ| < 1pp). The loop is GPU-bound; there is no host gap left for more static capture to
recover.

### 2. Launch overhead the deployed loop-graph already recovers (graph_on vs graph_off) — the #444 number

One-variable serve-time A/B (`ONEGRAPH` 1 vs 0), everything else held identical:

| µs/step | graph_on (deployed) | graph_off (eager K=7 loop) |
|---|---|---|
| drafter GPU span | **1426** | **1494** |
| decode wall | 7903 | 7970 |
| E_accept | 3.588 | 3.596 |

- graph_off − graph_on wall = **67 µs/step (0.84% of the eager cycle)**.
- **Mechanism:** the loop-graph fuses the 7 width-1 drafter micro-forwards, removing inter-kernel launch
  latency → the **drafter GPU span shrinks 1494 → 1426 µs**. The host-overhead residual is unchanged
  (35 ↔ 36 µs), so the graph recovers GPU-adjacent dispatch, not a host gap.
- realized = **+2.80 TPS** (local, conc=1); priced on the 467.14 frontier, disabling graphs → 463.21 TPS
  (**−3.93 TPS**).

**@advisor — coordination flag for land #444:** this **67 µs/step / −3.93 TPS** is the cost land #444
(async pipelined drafting) re-introduces **iff** a second async stream forces the loop-graph OFF. The
host-overhead self-abort metric (0.46%) is just under the 0.5% gate, but the graph-recovered launch
overhead (0.84% of the eager cycle) is the load-bearing #444 quantity — #444 must keep the drafter loop
graph-captured (or recover the equivalent fusion on the async stream) or it forfeits ~3.9 TPS.

### 3. Equivalence (instr. 5) — graph_on ≠ graph_off byte-exactly; cross-session control proves it's the graph, not noise

| comparison | verdict | prompts identical | divergent tokens | onset |
|---|---|---|---|---|
| graph_on vs graph_off | DIVERGENT | 127/128 | 23/16384 (0.14%) | idx 105 |
| **graph_on vs graph_on (2 procs, control)** | **GREEDY_IDENTICAL** | **128/128** | **0** | — |
| graph_off vs graph_on(B) (control) | DIVERGENT | 127/128 | 23/16384 | idx 105 |

I added a cross-session control because the lone divergence (late onset idx 105, single prompt) looked
like a floating-point near-tie. The control settles it: **same-config on-vs-on across two serve processes
is byte-exact (128/128, 0 divergent tokens)** — the stack is deterministic cross-session (int4 lm_head
Marlin). The single divergence reproduces against a *second* graph_on process, so it is **caused by the
`ONEGRAPH` 1↔0 flip itself**, not noise: the captured drafter proposes marginally different tokens
(E_accept 3.588 vs 3.596), flipping one near-tie verify argmax at token 105 that cascades 23 tokens.

**This is not a regression and graph_on is greedy-safe:**
- **graph_on is the DEPLOYED arm** — the non-byte-exactness vs the eager counterfactual is a property of
  the shipped stack, not introduced by this analysis-only card (no served file changed).
- graph_on **PPL = 2.3778 passes the ≤ 2.42 gate** (deployed ref 2.3772).
- 0.14% on/off token divergence is **inside the deployed incumbent's own non-equivalence** (identity
  0.9966 = 0.34%). graph_off is the slower (−2.80 TPS), non-deployed arm.

### 4. Metrics summary

| metric | value | gate / baseline |
|---|---|---|
| primary `host_overhead_frac` (graph-on) | **0.456%** | < 0.5% self-abort → **abort** |
| test `ppl` (graph-on) | **2.3778** | ≤ 2.42 → **pass** (deployed 2.3772) |
| launch overhead recovered by deployed graph | **67 µs/step = +2.80 TPS** | −3.93 TPS on 467.14 if OFF (#444) |
| greedy identity graph_on vs graph_off | 127/128, 0.14% tok | graph-attributable (control-proven) |
| cross-session determinism (on vs on) | byte-exact 128/128 | measurement-integrity check ✅ |
| E_accept (on / off) | 3.588 / 3.596 | — |
| realized TPS contributed by THIS PR | **0.00** | lever already deployed |
| completed requests (each arm) | 128/128 | — |

### Exact commands

```bash
# full 2-arm probe (graph_on: timing + capture + PPL; graph_off: timing + capture)
python research/validity/cudagraph_capture_specloop/cudagraph_capture_specloop.py \
  --num-prompts 128 --output-len 128 \
  --wandb-name ubel/cudagraph-capture-spec-loop --wandb-group cudagraph-capture-kopt
# cross-session control (second graph_on process; on-vs-on + off-vs-on(B))
python research/validity/cudagraph_capture_specloop/control_xsession.py
```

- **W&B run:** `qlvakiyu` (project `gemma-challenge-senpai`, group `cudagraph-capture-kopt`)
- **Peak GPU memory:** ≈ 20.7 GiB on the A10G (0.90 util cap of 23.0 GB); model load 8.85 GiB + KV cache
  9.46 GiB + CUDA-graph pool 0.04 GiB. Both arms fit identically; graph capture adds only 0.04 GiB.
- **Operating point:** `submissions/fa2sw_precache_kenyan`, precache-on, single-stream greedy, K=7 (M=8),
  pod A10G (sm_86). Instrument: shipped STEPTIME probe (CUDA-event at the Python boundary, outside the
  graph) — `torch.profiler`'s kernel pass 500s on this stack (#284), so STEPTIME is the correct primary
  instrument and directly yields the CPU-vs-GPU per-step split the gate needs.

### What happened

The PR hypothesized that the inter-step K=7 spec-decode orchestration is **not** captured and that
capturing it would recover per-step launch overhead as **new** realized TPS. Reading the served
`sitecustomize.py` (and confirming in the engine log) shows the loop is **already** captured in one CUDA
graph. So this became the clean measured-abort the PR's instr. 2 anticipates: host overhead 0.46% < 0.5%
→ no prototype. The genuinely-additive work was (a) re-confirming the gate precisely at 128→128, and
(b) the graph_on/off A/B that sizes what the deployed graph already banks (67 µs/step, the #444 cost).
The one surprise — graph_on/off not byte-exact — I chased with a cross-session control and proved it is
the graph flip (not FP noise), but it is within the deployed stack's known non-equivalence and the
deployed (graph_on) arm passes the PPL gate.

### Suggested follow-ups

1. **Land #444 must hold the drafter loop graph-captured.** Async pipelined drafting that forces graphs
   OFF forfeits the measured −3.93 TPS. If #444 needs a second stream, recover the drafter
   micro-kernel-launch fusion on that stream (e.g. capture the per-stream draft chain) before merging.
2. **PPL on graph_off** (not measured here — graph_off is the non-deployed arm) would let us state "both
   arms pass the PPL gate; the 1-prompt difference is a near-tie that doesn't move PPL." Cheap (~4 min)
   if the advisor wants the equivalence story fully closed on both arms.
3. The remaining ~0.46% host residual is the inter-graph host hop + accept/scheduler — irreducible
   without fusing verify+draft+accept into one mega-graph, which is forbidden (paged-KV block-table
   updates must stay outside the graph). Not worth pursuing.

_Certificate: `analysis_only = True`. STEPTIME timing-only + official decode/PPL capture. No served-file
change, no emitted-token change, no HF Job, no submission, NOT a launch. Frontier 467.14 / deployed
481.53 unchanged; this card adds 0 TPS._
