STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ns2x9f7e"],"primary_metric":{"name":"relax_exec_spec_pending_slots","value":10},"test_metric":{"name":"ppl","value":2.3772}}

## Results

CPU-only analytic — the **EXECUTION-readiness skeleton** (the GO-branch "how-to-go-safely" checklist, the EXECUTION complement to #462's verdict surface). **stark #452 has now LANDED — and it is NEGATIVE.** The realized numbers fill the slots, and the skeleton's own `execution_verdict()` returns a **bar-invariant ROLLBACK**: real-data proof the GO-branch correctly **refuses to ship a dominated, identity-breaking config**. Per the advisor (2026-06-16), I completed the skeleton with stark #452's actual numbers **and kept the parameterized `<PENDING #452>` form alongside** (in case the human wants the spec for a future lever).

`relax_exec_self_test_passes = True` (**12/12** conditions a–l, incl. the new `l_realized_stark452_bar_invariant_rollback`). W&B run [`ns2x9f7e`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ns2x9f7e), group `equivalence-escalation-anchors`. Imports #462's `recommend()` **verbatim**; round-trips committed `#462` + `#458` + `#457` (= ubel #450 roofline) + `#448` JSONs (every banked number round-trips at **0.0**). `relax_exec_spec_pending_slots = 10` (PRIMARY — the parameterized skeleton kept intact).

### ⮕ stark #452 LANDED — relax lane CLOSED, gate returns ROLLBACK

stark #452 (runs [`daqrzr99`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/daqrzr99) / [`00ovtdnt`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/00ovtdnt), MERGED, W&B-verified) **built and benchmarked** the relax split-K re-partition. It came back dominated and identity-breaking:

| metric | deployed (481.53 config) | stark #452 realized relax | verdict |
|:--|--:|--:|:--|
| TPS | 481.53 | **466.20** | **DOMINATED** (gain **−15.33** vs deployed; **−0.94** vs strict 467.14) |
| identity | 0.9966 | **0.730** | collapsed |
| flips | 3 | **3317** | exploded |
| flip-KIND | (near-ties) | **cascading free-running divergence** | new-KIND (same_kind=False) |
| PPL | 2.3772 | **2.3782** (Δ +0.001) | ≤ 2.42 — **PASSES** the gate |

Dropping these into the **shipped** `execution_verdict()` (the exact function this card produces):

```
execution_verdict(measured_tps=466.20, measured_ppl=2.3782, flip_kind_same=False, completed=128, human_bar_tps=B)
  →  ROLLBACK   for every bar B (including the most-lenient B=0)  ⟹  bar-INVARIANT ROLLBACK
```

**Per-clause harness eval on the realized config:**

| clause | check | realized | result |
|:--|:--|:--|:--:|
| (a) TPS CI-clean over bar | gain − σ_hw ≥ B | gain −15.33 (≤ 0 < any B) | **FAIL** |
| (b) PPL ≤ 2.42 | hard gate | 2.3782 | **PASS** |
| (c) flip-KIND same-family | not new mode | cascading divergence | **FAIL** |
| (d) 128/128 completed | validity | 128 | **PASS** |

- **The validation point:** PPL — #462's *most-sensitive* input — **PASSED** (2.3782 ≤ 2.42, margin 0.0418). **PPL alone would have waved this config through.** It is the **TPS-domination (clause-a) and the new-KIND (clause-c)** clauses that catch it. This vindicates the multi-clause design *and* the count-vs-KIND orthogonality discipline (3317 flips is a *count*; the categorical *KIND* is what gates).
- **HOLD vs ROLLBACK (honest note):** the advisor's comment called the outcome "HOLD/ROLLBACK — never SHIP-READY" and asked for the **HOLD** verdict. Mechanically the gate returns the **hard ROLLBACK** — strictly stronger than HOLD. HOLD (`recommend()` CI-AMBIGUOUS) is *impossible* here because CI-AMBIGUOUS requires `same_kind=True`, and the realized flip-KIND is cascading divergence (`same_kind=False`); the config *also* independently fails clause-a (dominated TPS). Two hard NO-GO clauses ⟹ ROLLBACK. Both labels sit inside the advisor's "never SHIP-READY" envelope; I report the precise machine output (ROLLBACK) rather than soften the gate. Happy to relabel to "HOLD" in the packet if you prefer the softer framing.

### (1) The exact served-file change spec — parameterized; realized slot now filled

The relax prize lives in **one place**: the **int4-Marlin W4A16 GEMM K-reduction**. Two layers (the trap that sank four isolated levers — conflating the cheap flag with the real re-partition):

| | **knob A — in-tree Python proxy** | **knob B — build-level split-K re-partition** |
|:--|:--|:--|
| role | SUB-PRIZE PROXY (proves the FP-reassociation hazard) | **THE PRIZE** (re-partitions the K-reduction) |
| file / symbol | `marlin_utils.py` → `apply_gptq_marlin_linear(..., use_fp32_reduce=<bool>)` | Marlin CUDA kernel `csrc/quantization/gptq_marlin/*` → split-K geometry, BLOCK_K, num_warps |
| current → proposed | `use_fp32_reduce = True` → `False` | split-K geometry = f(M) auto-select → re-partitioned |
| modeled ΔTPS | +0.64 UB (sub-bar) | +17.05 realistic / +29.34 ceiling **(modeled)** |
| **stark #452 realized** | — | **+(−15.33)** — the build came back **DOMINATED** |
| exposed Python knob? | yes | **NO — auto-selected in-kernel as f(M)** (#448 / kanna #122) |
| greedy-safety | REASSOCIATING → breaks byte-exactness 3/4 shapes | REASSOCIATING → greedy-**UNSAFE** (#450); realized identity 0.730 |

- **Key finding (still true):** the prize has **no Python knob** — it required a **patched-Marlin wheel BUILD**. stark #452 produced that wheel (`realized_kernel_artifact_ref` slot = the daqrzr99/00ovtdnt split-K wheel) and the result was negative. The submission anchor that *would* have changed is `manifest.json dependencies[0]` (pinned `vllm-0.22.1rc1.dev307+g3e8afdf78`).
- The 5 spec slots (`knob_A.exact_realized_setting`, `knob_B.proposed_{block_k,num_warps,split_k_partition}`, `realized_kernel_artifact_ref`) are **kept** in PENDING form for any future lever; the realized overlay records what stark #452 built.
- **Applying remains HUMAN-GATED — operator Directive #3** (`Approval request: HF job for …` + explicit approval). This card SPECIFIES the change; it does NOT make it (`analysis_only=true`, `no_served_file_change=true`). **Moot now** — the lane is closed.

### (2) Post-relax validation harness — 4 clauses, each mapped to a `recommend()` clause

| clause | check | stark #452 metric | `recommend()` clause | realized |
|:--|:--|:--|:--|:--:|
| **(a)** TPS CI-clean over bar | gain − 4.8153 ≥ B | `measured_relax_tps` | clause-1 | **FAIL** |
| **(b)** PPL hard gate | ≤ 2.42 | `measured_relax_ppl` | clause-2 | PASS |
| **(c)** flip-KIND same-family | not new mode | `measured_relax_flip_kind` (**KIND, never N**) | clause-3 | **FAIL** |
| **(d)** 128/128 completed | validity | `completed` | DOMAIN precondition | PASS |

`validation_checklist_clauses = 4`. All-pass ⟺ SHIP-READY; here 2/4 fail (hard) ⟹ ROLLBACK.

### (3) Rollback criterion + single-knob reversibility

`relax_change_is_single_knob_reversible = True`. ROLLBACK if **any** of {PPL > 2.42; new-KIND flips; TPS gain not CI-clean of B; < 128/128}. The realized config trips **two** (TPS not CI-clean + new-KIND). Revert is a **single-knob, byte-for-byte** return to the deployed **481.53 / 2.3772 / 128-128** config: re-point `manifest.json dependencies[0]` to the stock pinned wheel + restore `use_fp32_reduce=True`. (In practice nothing was ever applied — the spec stayed analysis-only — so the deployed config is already the live one.)

### (4) Pre-wired stark #452 → SHIP-READY / HOLD / ROLLBACK (the one-call wrapper)

`execution_verdict()` **wraps #462's imported `recommend()`** + the 128/128 precondition:

```python
execution_verdict(measured_tps, measured_ppl, flip_kind_same, completed, human_bar_tps, k=1):
    if completed < 128:               return "ROLLBACK"
    gain = measured_tps - 481.53
    return {GO:"SHIP-READY", CI-AMBIGUOUS:"HOLD", NO-GO:"ROLLBACK"}[recommend(gain, measured_ppl, flip_kind_same, human_bar_tps, k)]
```

`exec_skeleton_maps_to_recommend = True` (verified equal to `recommend()` over a 4×3×2×3 grid). Worked corners (modeled) + the realized row:

| gain | PPL | kind | completed | bar B | → action |
|:--|:--|:--|:--:|:--|:--:|
| +17.05 | 2.3772 | same | 128 | +0.26 | SHIP-READY |
| +17.05 | 2.43 | same | 128 | +0.26 | ROLLBACK |
| +17.05 | 2.3772 | **new** | 128 | +0.26 | ROLLBACK |
| +17.05 | 2.3772 | same | 128 | +15 | HOLD |
| +17.05 | 2.3772 | same | **120** | +0.26 | ROLLBACK |
| +29.34 | 2.3772 | same | 128 | +20 | SHIP-READY |
| **−15.33 (stark #452 realized)** | **2.3782** | **new** | **128** | **any (incl. 0)** | **ROLLBACK** |

- **Handed to fern #357:** the capstone reads this as **"the gate, validated against the config that tried and failed to clear it."** The card produces the execution wrapper only — no verdict duplication.

### (5) Self-test + metrics

12/12 conditions pass: the original 11 (banked round-trip 0.0; parents green; knob surface = #448; proxy/prize demotion; 10 slots enumerated; 4 clauses → `recommend()`; `execution_verdict` wraps + equals `recommend()` on grid; single-knob rollback; Directive #3 + PPL anchor; NaN-clean) **plus (l)** the realized stark #452 config returns a bar-invariant ROLLBACK with clauses a+c failing and b+d passing.

### Baseline comparison

| quantity | baseline (PR body / banked) | this card | match |
|:--|:--|:--|:--:|
| deployed / strict / realistic / ceiling | 481.53 / 467.14 / 498.58 / 510.87±4.82 | identical (round-trip resid **0.0**) | ✓ |
| σ_hw / PPL gate / margin | 4.8153 / ≤2.42 / 0.0428 | identical | ✓ |
| decision_flip_tps_threshold (#462) | 12.2346 | 12.2346 (imported) | ✓ |
| **stark #452 realized relax TPS** | (was pending) | **466.20** (gain −15.33) | new |
| **stark #452 realized identity / flips / PPL** | (was pending) | **0.730 / 3317 / 2.3782** | new |
| **gate verdict on realized config** | (was pending) | **ROLLBACK (bar-invariant)** | new |

Adds **0 TPS**; greedy/PPL untouched (served PPL anchor **2.3772**, `official_tps = 0`, `analysis_only = True`, `no_served_file_change = True`). The realized relax PPL 2.3782 is stark #452's measurement of the *relax* config, not a change to the served model.

### Command

```bash
cd target/
python3 research/validity/relax_execution_readiness/relax_execution_readiness.py \
  --wandb_name "land/relax-execution-readiness" --wandb_group "equivalence-escalation-anchors"
# self-test only (CPU): add --self-test  → "self-test PASS"
```

### Peak memory

**13.57 MiB** (CPU-only; no GPU, no vLLM, no model load).

### What happened

Worked cleanly, then sharpened on real data. The skeleton went from a parameterized GO-branch template to a **validated gate**: stark #452 built the relax split-K wheel, it came back **dominated** (466.20 < deployed 481.53) with a **cascading identity collapse** (0.9966 → 0.730, 3 → 3317 flips), and the skeleton's own `execution_verdict()` returned a **bar-invariant ROLLBACK**. Three things worth flagging:

1. **The gate refused exactly the right config — and PPL alone wouldn't have.** The realized PPL barely moved (+0.001, passes the 2.42 gate). If the gate were PPL-only it would have **waved a dominated, identity-broken config through**. It was the TPS-domination (clause-a) and new-KIND (clause-c) clauses that caught it. The multi-clause design and the count-vs-KIND orthogonality (3317 flips is a count; the KIND gates) are what made the refusal correct.
2. **The relax prize did not exist in practice.** The +17.05/+29.34 was a roofline-modeled ceiling; the realized split-K re-partition is **−15.33** — it trades away accuracy *and* loses speed. The relax lane is **CLOSED**. Deployed 481.53 stands.
3. **Honest HOLD-vs-ROLLBACK note.** The advisor asked the harness to "return HOLD"; mechanically it returns the **hard ROLLBACK** (same_kind=False forecloses CI-AMBIGUOUS/HOLD, and the TPS is dominated independently). I report the precise machine output rather than mis-build the gate — both sit in the "never SHIP-READY" envelope. Easy to relabel in the one-screen packet if HOLD reads better.

### Public evidence used

Leaderboard digest (`as=senpai`, fetched 2026-06-16): the **valid** frontier sits at ~489.66 TPS (firfir-cast `hayai-ctk48-mwfix-v1`, verification=valid); **pending** split-K relax attempts reach 508.63 (fabulous-frenzy `ff-splitkv-frantic-fawindow`). The `ff-splitkv-*` name confirms the prize *location* this spec targeted — and stark #452's realized −15.33 shows that location, built, does not pay off on this served stack. The rank-1 `f64-max` row (1.79e308 TPS) is a non-physical sentinel, ignored. (No board write; CPU-only reconciliation of committed banked JSONs — `#462`, `#458`, `#457`, `#448` — plus stark #452's realized numbers. No HF job, submission, or served-file change.)

### Suggested follow-ups

- **fern #357 capstone:** this card now composes as "the execution gate, validated against the config that failed it" — the realized ROLLBACK is a concrete worked example for the one-screen packet (alongside #462's verdict surface). Round-trip resid 0.0 with all parents.
- **Relax lane:** CLOSED on the served stack. The parameterized `<PENDING #452>` form is retained so a *different* future lever (if one is ever proposed) can reuse the same gate without re-derivation — but the int4-Marlin split-K re-partition specifically is a measured dead end (dominated + identity-breaking).
- If you'd prefer the packet to read **HOLD** rather than **ROLLBACK** for the realized config, say the word and I'll relabel the live verdict string (the underlying gate stays the same — it's a NO-GO either way).
