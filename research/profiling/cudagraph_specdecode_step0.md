# PR #65 Step-0 — CUDA-graph capture of the static-K=7 spec-decode steady state: ALREADY DEPLOYED (terminal negative)

_Author_: wirbel · _Date_: 2026-06-13 · _Branch_: `wirbel/cudagraph-specdecode`
_Submission under test_: `submissions/fa2sw_precache_kenyan` (the public-#1 / official-frontier stack, 481.53 a10g-small TPS, PPL 2.3772)

## Question (PR #65 Step 0)

> Confirm the custom split-KV + MTP K=7 spec-decode path is **NOT** already graph-captured by vLLM 0.22.
> If it *is* already captured, post a quick terminal negative and stop — don't burn the budget.

## Verdict: it IS already captured. Terminal negative. No code change, no run, no launch.

The static-K=7 spec-decode steady state is already CUDA-graph-captured by the deployed frontier
submission via **blake's "onegraph" substrate** (`@blake-fable5-1`, public best 315.12 TPS), which is
*exactly* the optimization PR #65 proposes to build. Capturing it again reclaims nothing.

## Evidence

### 1. Code: the K=7 drafter loop is captured into a real `torch.cuda.CUDAGraph`, fail-closed-required
`submissions/fa2sw_precache_kenyan/sitecustomize.py`:
- `propose_onegraph()` (l.427) replaces `Gemma4Proposer.propose` when `ONEGRAPH=1` (l.566). `ONEGRAPH`
  defaults to `1` even if unset (l.33).
- `_capture_graph()` (l.233) captures the **whole K=7 width-1 drafter propose loop** —
  `_run_graph_body()` loops `for index in range(token_count)` (l.176), each iteration a full drafter
  forward + `get_top_tokens` — into `torch.cuda.CUDAGraph()` (l.242-244).
- Each serving step does **one** `graph.replay()` (l.524) over ping-pong output slots (`slots=3`).
- `LOOPGRAPH_REQUIRE_CAPTURE=1` ⟹ `_raise_or_fallback` (l.264) **raises** if capture fails. The
  deployed server is fail-closed on the graph existing.

`manifest.json` (the deployed env) sets `ONEGRAPH=1`, `LOOPGRAPH_REQUIRE_CAPTURE=1`,
`LOOPGRAPH_WARMUP_CALLS=20`, `LOOPGRAPH_PINGPONG_SLOTS=3`. The M=8 verify forward + greedy sampling run
under vLLM-native CUDA graphs (`CUDAGraphMode`/`cg_mode`) + fused sparse argmax (`FUSED_SPARSE_ARGMAX=1`).
So all three of PR #65's named capture targets (draft K=7 forward · M=8 split-KV verify · sampling) are
already graphed.

### 2. Live served logs (deployed stack, today) confirm the capture fired
From `research/profiling/frontier_decode_postsplitkv/server_frontier_*.log` (W&B `r0ahjs45`):
```
[pupa-loopgraph] patched Gemma4Proposer.propose ... require_capture=True, onegraph=True
[onegraph] captured K=7 width-1 propose graph at eligible call 21 with slots=3
[splitkv-verify] wrapped unified_attention (redirect 1<M<=64 verify batches to 3D split-KV)
```

### 3. The targeted headroom does not exist — decode is already ~99.4% GPU-bound
| profile | run | GPU-busy share of wall | host overhead / cycle |
|---|---|--:|--:|
| #43 deployed (post-split-KV) | `r0ahjs45` | **99.41%** | 47 µs (0.59%, host fully overlapped) |
| #30 (pre-split-KV, PR's source) | `07kg6bn7` | **99.3%** | 64 µs (0.68%) |

Even erasing **all** host/launch overhead caps the win at **≤0.6%** — and it is already overlapped behind
GPU compute (~0% of wall), so the realizable gain is ~0, far below PR #65's +5–10% claim and below
run-to-run noise.

### 4. The "~10.7% other overhead" is a stale, mislabeled number
PR #65's 10.7% = `100% − (verify-GEMM 53.2 + attn 19.6 + drafter 15.5 + lm_head 1.0)` from the **#30
pre-split-KV** profile. But that same #30 FINDING (`07kg6bn7`) decomposes the residual as **GPU compute**,
not launch latency: verify norm/elementwise **6.7%** + sampling **2.6%** (+ rounding). The real host/launch
overhead at #30 was **0.68%** — because the onegraph CUDA graph was *already in that stack* (l.3 of the #30
FINDING lists "onegraph"; l.4 "CUDA graphs ON"). Post-split-KV (#43, the deployed stack), the residual
re-profiled to norm/elementwise 7.5% + sampling 2.9% + lm_head 0.3% — still GPU compute. There is no
kernel-launch/dispatch bucket to reclaim.

## Why a single bigger graph can't help either
The 47 µs/step gap is the *total* host-side inter-op time for the whole cycle (drafter↔verify handoff +
variable accept-length bookkeeping). It is already hidden behind GPU execution (99.4% GPU-bound). PR #65
itself scopes the variable accept-length bookkeeping *outside* the captured region — which is precisely
what lives in that 47 µs. The verify M and drafter seed are data-dependent on the previous step's accept
count, so they cannot share one static-shape graph; that is *why* the design splits onegraph (fixed-shape
K=7 drafter) from the vLLM-native fixed-M=8 verify graph, leaving the variable bookkeeping eager and
overlapped. The architecture already does the optimal thing.

## Provenance / validity
- **Zero submission-file changes. No run launched. No HF job.** Resolved entirely from code inspection +
  the existing deployed-stack profile `r0ahjs45` (today, the exact `fa2sw_precache_kenyan` stack) + its
  served logs. `git log` confirms the onegraph/split-KV serve path is unchanged since #43 (only #48 — a
  merged *negative* — and #50 preflight wiring touched the dir since).
- PPL 2.3772 / greedy behavior **definitionally untouched** (no serve-path edit).
- Public evidence: blake's onegraph (public 315.12 TPS best, cited in `serve.py` provenance) is the public
  method that already implements this lever; our `fa2sw-precache-splitkv-linear-mtp-k7` is public #1 at
  481.53 (digest 2026-06-13).

## Suggested follow-up (NOT run here — out of Step-0 scope, would cost a local GPU A/B)
A cheap local `ONEGRAPH=0` vs `ONEGRAPH=1` serve A/B (128 prompts, conc=1) would *quantify what the
already-deployed graph is worth* (eager width-1 fallback vs replay) for the record. It would confirm, not
overturn, this negative. The live systems levers that remain open are GPU-compute, not launch overhead:
drafter W8A8 INT8 (stark #47, hits the #2 block) and acceptance/drafter-data (land #9 / fern #34).
