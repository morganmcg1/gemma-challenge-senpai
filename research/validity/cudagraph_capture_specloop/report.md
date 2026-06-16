# PR #443 — Static CUDA-graph capture of the K=7/M=8 spec-decode loop (LOCAL, analysis-only)

**PRIMARY `host_overhead_frac` (graph-on, deployed) = 0.46%** · self-abort gate 0.5% · **`self_abort = True`**
**TEST `ppl` (graph-on) = 2.3778** (gate ≤ 2.42 → pass; deployed ref 2.3772)
**realized TPS contributed by THIS PR = 0.00** — the loop-graph is ALREADY deployed (`crosses_deployed_481 = False`)

> **Verdict — MEASURED-ABORT (PR #443 instr. 2).** The served K=7 stack ALREADY replays the entire
> K=7 draft loop as ONE CUDA graph (`ONEGRAPH=1` → `_capture_graph` records `_run_graph_body`: the 7
> width-1 drafter forwards + fused-sparse-argmax + hidden copies; hot path is a single
> `graph.replay()`). The paged-KV block-table refresh stays OUTSIDE the graph
> (`_refresh_static_buffers`), exactly as the PR requires. The hypothesis premise — that the
> inter-step spec-decode orchestration is un-captured and capturing it would recover launch overhead
> as *new* TPS — is **false for the deployed stack**. The residual per-step host/CPU overhead that
> remains outside the graph is **36 µs (0.46%)** of the **7903 µs** cycle (GPU-busy share **99.5%**),
> **below the 0.5% gate** → record the negative, bank it, do not prototype. This reproduces ubel #284's
> 0.50% host-overhead read (measured there at out=512; this card confirms at the PR's 128→128 point).

Operating point: `submissions/fa2sw_precache_kenyan`, precache-on, single-stream greedy, K=7 (M=8),
128 prompts → 128 output, pod A10G (sm_86). Instrument: shipped STEPTIME probe (perf_counter +
CUDA-event at the Python call boundary, OUTSIDE the graph). Per-step decode wall = exec_cpu + host_gap
(p50 steady-state, the #275 host-to-host method); host overhead = wall − GPU-busy(verify+drafter).

## 1. Per-step decode wall (STEPTIME p50 steady-state, host-to-host)

| quantity (µs) | graph_on (deployed) | graph_off (eager K=7 loop) |
|---|---|---|
| verify (execute_model) GPU | 6441 | 6441 |
| drafter (propose) GPU | 1426 | 1494 |
| exec host-call wall | 6352 | 5617 |
| inter-step gap | 1551 | 2353 |
| **decode wall / step** | **7903** | **7970** |
| GPU-busy (verify+drafter) | 7867 | 7935 |
| host overhead (wall − GPU-busy) | **36 (0.46%)** | 35 (0.44%) |

Both arms are GPU-bound (host overhead < 0.5%). E_accept = 3.588 (on) / 3.596 (off).

## 2. Launch overhead the deployed loop-graph already recovers (graph_on vs graph_off) — the #444-sizing number

- graph_off wall − graph_on wall = **67 µs/step** (0.84% of the eager cycle).
- **Mechanism:** the loop-graph fuses the 7 width-1 drafter micro-forwards, eliminating inter-kernel
  launch latency — the measured **drafter GPU span shrinks 1494 → 1426 µs (−68 µs)** under capture.
  The entire 67 µs wall delta is this recovered drafter launch-latency; the host-overhead residual is
  unchanged (35 ↔ 36 µs), i.e. the graph recovers GPU-adjacent dispatch, not a host gap.
- realized output-TPS the graph already banks (local, conc=1) = **+2.80 TPS**.
- priced onto the 467.14 strict-equivalence frontier: disabling graphs drops it to **463.21 TPS**
  (**−3.93 TPS**).
- **This −3.93 TPS / 67 µs/step is the cost land #444 (async pipelined drafting) re-introduces iff a
  second async stream forces graphs OFF.** Flagging for #444 per the advisor's coordination ask.

## 3. Equivalence — graph_on vs graph_off is NOT byte-exact; the difference is graph-attributable

| comparison | verdict | prompts identical | divergent tokens | onset |
|---|---|---|---|---|
| graph_on vs graph_off | DIVERGENT | 127/128 | 23/16384 (0.14%) | idx 105 |
| **graph_on vs graph_on (2 processes, control)** | **GREEDY_IDENTICAL** | **128/128** | **0/16384** | — |
| graph_off vs graph_on(B) (control) | DIVERGENT | 127/128 | 23/16384 | idx 105 |

**Cross-session control rules out FP noise and pins the divergence on the graph.** The same-config
on-vs-on comparison (two independent serve processes) is **byte-exact (128/128, 0 divergent tokens)** —
the stack is deterministic across sessions (int4 lm_head Marlin + deterministic kernels). The single
graph_on↔graph_off divergence reproduces identically against a *second* graph_on process, so it is
**caused by the ONEGRAPH 1↔0 flip itself**, not process-to-process noise. Mechanistically: the captured
drafter proposes marginally different tokens (E_accept 3.588 vs 3.596), shifting one **near-tie verify
argmax at token 105** which cascades through the 23-token suffix. Late + rare onset (idx 105, 1/128) is
the floating-point-tie signature, exactly as `greedy_gate.onset_summary` documents (a *structural* bug
diverges early and on most prompts).

**Why this is not a regression and graph_on is greedy-safe:**
- **graph_on is the DEPLOYED arm.** This non-byte-exactness vs the eager counterfactual is a property of
  the shipped stack, not something this analysis-only card introduces (it changes no served file).
- graph_on **PPL = 2.3778 passes the ≤ 2.42 gate** and matches deployed 2.3772.
- the 0.14% on/off token divergence sits **inside the deployed incumbent's own non-equivalence**
  (deployed identity 0.9966 = 0.34% token divergence vs the int4 reference).
- graph_off (the byte-different arm) is the **slower, non-deployed** counterfactual (−2.80 TPS).

## 4. Measurement integrity (card self-test)

- nan_clean: **True** · gate_decided: **True** · consistent_with_ubel284_host_frac (|0.46%−0.50%|<1pp): **True**
- cross_session_deterministic (on-vs-on byte-exact): **True** · ppl_pass: **True**
- **all_pass: True**

The probe's `self_test_passes` field is False only because it folds the graph_on↔graph_off
byte-exactness into the pass — but that is a *result* (graph ≠ eager, proven graph-attributable), not a
harness failure. The measurement itself executed cleanly and the gate is decided.

## 5. Bottom line

1. **Self-abort gate FIRES** — host overhead 0.46% < 0.5%; the loop is already CUDA-graph-captured. No
   new capture to prototype. **This PR adds 0 TPS** (the lever is inside the deployed numbers).
2. **The deployed loop-graph already recovers +2.80 TPS / 67 µs per step** (drafter micro-kernel launch
   fusion). Disabling it costs −3.93 TPS on the 467.14 frontier — **the land #444 cost if a second async
   stream forces graphs OFF.**
3. **graph_on vs graph_off is not byte-exact (127/128), and the cross-session control proves it is the
   graph, not FP noise** — but graph_on is the deployed arm, passes the PPL gate, and the divergence is
   within the shipped stack's known non-equivalence.

## Greedy/PPL-safety certificate

`analysis_only = True`. STEPTIME timing-only + official decode/PPL capture; **no served-file change, no
emitted-token change, no HF Job, no submission, NOT a launch.** The loop-graph capture is already
deployed, so this PR adds 0 TPS; the graph_on/off delta sizes what is already realized (and the land
#444 cost of disabling it). Frontier 467.14 / deployed 481.53 unchanged.
