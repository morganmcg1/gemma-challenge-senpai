# PR #311 — EAGLE-3 #101 risk: capture-SIZE dispatch, not VRAM — price it

**PRIMARY `capture_dispatch_risk_self_test_passes` = True** (all 7 conditions a–g)
**TEST `max_safe_tree_width_under_deployed_list` = 16** · **`deployed_m8_dispatch_safe` = True**
**W&B `os01ttw9`** (group `eagle3-dispatch-risk`) · LOCAL read-only static analysis, 0 GPU, 0 TPS

> **Verdict:** the deployed frontier launch (`fa2sw_precache_kenyan`) does **NOT pin**
> `cudagraph_capture_sizes` (no `--compilation-config` / `--cuda-graph-sizes` / `--enforce-eager`);
> it inherits vLLM's default, ceiling `max_cudagraph_capture_size=16` (banked #101/#306). The
> verify-side dispatchable widths are the **(1+K)=8-multiples within it: {8, 16}**. **Deployed M=8**
> (prewarm shape `(1,8)`, parsed from `serve.py:487`) is dispatch-**SAFE**; **M=16** sits **at the
> boundary** (captured); **M=32** (32 tokens > 16) **falls through to lawine's IndexError CRASH
> regime**. The M=32 blocker is the **DISPATCH LIST** — #306 priced its VRAM cost at a trivial
> **+12 MiB** — fixed by **ADDING `[24, 32]`** to the list and raising the ceiling to 32, **NOT by
> freeing memory**. The **draft side** (K=7) is a **manual ONEGRAPH capture** (`CUDAGraphMode.NONE`),
> not list-dispatched → **safe by construction**. The dispatch risk lives **entirely on the verify
> side**, which is dispatch-safe only up to **M=16** under the deployed list.

## 1. Deployed capture config — parsed read-only from the served source (instruction 1)

Parsed from `submissions/fa2sw_precache_kenyan/{manifest.json, serve.py, sitecustomize.py}`:

| knob | value | source |
|---|---|---|
| `num_speculative_tokens` (K) | **7** (mtp) | `manifest.json` `SPECULATIVE_CONFIG` |
| deployed verify width (M) | **8** (= 1+K spine) | `serve.py:487` prewarm `torch.full((1, 8))`, `cu_num_draft_tokens=[7]` |
| `max_num_seqs` | 1 | `manifest.json` |
| `max_num_batched_tokens` | 512 | `manifest.json` |
| `performance_mode` | interactivity | `manifest.json` |
| `cudagraph_capture_sizes` **pinned?** | **NO** (no `--compilation-config` / `--cuda-graph-sizes`) | `serve.py` launch args |
| `--enforce-eager`? | **NO** (graphs ARE captured) | `serve.py` launch args |
| drafter capture mode | **manual ONEGRAPH** `CUDAGraphMode.NONE` | `sitecustomize.py:170` |
| `max_cudagraph_capture_size` (effective) | **16** | banked #101/#306 (default for this engine) |
| verify captured (1+K)-multiples | **{8, 16}** | #306 `precedent_101.valid_captured_sizes` |

**Key finding:** the list is **inherited, not pinned** — which is precisely the lever for the
mitigation (§3). The effective ceiling 16 is the banked empirical value (lawine #101 size-29 crash,
re-confirmed by #306 `y1lji0c6`); this leg does not re-derive vLLM's internal default formula.

## 2. Per-width dispatch arithmetic (instruction 2)

For the verify graph to dispatch, the per-replay token-count must be a captured size ≤ the ceiling.
The verify forward processes **M** tokens (M tree nodes, one query row each). Draft side: the
ONEGRAPH manual graph (batch=1 inner, K-unrolled) is **not** routed through the vLLM list.

| side | per-replay tokens | in `{8,16}` & ≤16? | dispatch | regime |
|---|---|---|---|---|
| **draft (K=7)** | manual ONEGRAPH (batch=1) | n/a (not list-dispatched) | ✅ **SAFE** | by construction |
| **verify M=8** (deployed) | 8 | ✅ (8 ✓, 8\|8) | ✅ **SAFE** | clear |
| **verify M=16** | 16 | ✅ (16 = max) | ✅ **SAFE** | boundary |
| **verify M=32** | 32 | ❌ (32 > 16) | ❌ **CRASH** | IndexError dispatch fall-through |

→ `max_safe_tree_width_under_deployed_list = 16`, `deployed_m8_dispatch_safe = True`.

## 3. #306 cross-check + the exact mitigating edit (instruction 3)

Reloaded `eagle3_capture_peak/eagle3_capture_peak_results.json` (`y1lji0c6`); all imported
constants match to **`max_abs_err = 0.0` (≤ 1e-6)**: build peak 20.158 GiB, peak headroom 3.842
(24-hard) / 2.842 (23-usable), capture transient 0.0410, logit buffers `262144·M·2B`.

**M=32's VRAM cost is exactly +12 MiB** (`262144·32·2B − 262144·8·2B = 12,582,912 B`) — trivially
affordable. **So the M=32 blocker is the DISPATCH LIST, not memory.** The fix is a config-list edit,
not freeing bytes:

- **M=16:** already dispatch-safe (16 = boundary, captured) — no list edit needed.
- **M=32:** **ADD `[24, 32]`** to `cudagraph_capture_sizes` and raise the ceiling to **32**:
  ```
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32]}'
  ```
  (equivalently `--cuda-graph-sizes 32`), in `serve.py` launch args / manifest env.

**Also required (flagged, out of scope here):** widening M also needs the verify **prewarm shape**
`serve.py:487-492` (`torch.full((1, M))`) and the tree `SpecDecodeMetadata` widened to M — a
**served-file change** this read-only audit does not make.

## 4. Mechanism corroboration — dispatch IndexError, not OOM (instruction 4)

`crash_is_vram_oom = False`, `crash_is_capture_size_dispatch_indexerror = True`. At dispatch the
runtime-mode lookup indexes the captured-size set for the forward's padded token-count;
token-count > `max_cudagraph_capture_size` finds no captured graph → **IndexError**. size-29 > max-16
⇒ crash. Corroborated by **vLLM #29091 / PR#23679** and repo
`research/tree_verify_path/report_descend_walk.md:87` ("graph-capture is the size-29 crash →
enforce-eager").

**Public-board honesty (two size-boundary failure modes):** the public challenge board
(openevolve `20260615-012216`, vidraft-darwin `20260614-234841`) attributes the **in-serve** size-29
tree-verify crash to the custom **star_gqa attention KERNEL** (fixed by a dense masked attn at
**N≤32**) — a **different** axis from the cudagraph capture-size dispatch IndexError this leg prices.
Whether they are the same event or two distinct boundaries, **both must be cleared** by any verify
width past M=8; this leg scopes **only** the config-list dispatch axis.

## Self-test (PRIMARY, a–g, NaN-clean)
a deployed list parsed from source ✓ · b per-width arithmetic correct (multiples of 1+K within list)
✓ · c M=8 SAFE / M=32 CRASH ✓ · d imported #306 constants ≤1e-6 ✓ · e NaN-clean ✓ · f mitigating
config edit emitted explicitly ✓ · g honest caveats carried (7) ✓ → **PRIMARY PASS**.

## Greedy/PPL-safety certificate
`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT
a launch, NOT a build. This is a **config-list correctness property** — it depends only on integer
token-counts and the capture list, not on tensor values (random-init shapes / no tensors at all
transfer). BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off
EAGLE-3 capture-SIZE dispatch priced: deployed **M=8 is dispatch-SAFE**; the verify tree is
dispatch-safe only **up to M=16** under the (inherited, unpinned) deployed list. Widening to **M=32
re-enters lawine #101's IndexError crash** — a **DISPATCH-LIST blocker** (VRAM cost a trivial +12 MiB
per #306), fixed by **adding `[24, 32]`** to `cudagraph_capture_sizes` (ceiling → 32) on the vLLM
launch, **not by freeing memory** — AND widening the verify prewarm (`serve.py:487-492`) + tree
metadata. The human GO/NO-GO can treat the **dispatch-list correctness** sub-clause as: **deployed
M=8 clears; any width past M=16 requires an explicit capture-size pin (a served-file change)**.
Topology desirability is a separate lane.
