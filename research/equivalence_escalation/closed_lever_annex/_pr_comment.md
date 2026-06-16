STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["k33t25ct"],"primary_metric":{"name":"closed_lever_count","value":20},"test_metric":{"name":"ppl","value":2.3772}}

## Results

**Deliverable: the closed-lever evidence annex for the relax-strict-equivalence decision (#407).** Analysis + synthesis only — **no HF job, no submission, no served-file change, `official_tps=0`.** Every cited run id is a **live, finished** W&B run (all 18 verified against the public API while authoring).

**Files (`research/equivalence_escalation/closed_lever_annex/`):**
- `closed_lever_evidence_annex.md` — the human-readable ledger + reconciliation + isolation-collapse appendix.
- `closed_lever_ledger.json` — the machine-readable closed-lever table (canonical source).
- `annex_self_test.py` — 0-GPU self-test (loads JSON, cross-checks the markdown, logs W&B).

**W&B:** [`k33t25ct`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/k33t25ct) (group `equivalence-escalation-anchors`) — `closed_lever_count=20`, `all_runs_linked=True`, `strict_headroom_is_greedy_unsafe=True`, `annex_self_test_passes=True`, `analysis_only=True`, `no_served_file_change=True`, `official_tps=0`, `ppl=2.3772`.

### 1. The ledger — 20 closed strict TPS levers

| axis | n | levers (closing PR · run) |
|---|---:|---|
| **SUPPLY** (faster verify) | 11 | pinned-K −5.82 (#433 `0pg4bz25`) · cb3 →0.0 (#437 `hv4xpgf8`) · fused RHT+VQ NO-GO 475.86 (#440 `5f3e91as`) · int4-GEMM Marlin +0.00 (#448 `fn4iz0dz`) · drafter-kernel tile +0.00 (#449 `xryqregh`) · **verify-wall Triton tile +0.2613 (#447 `crrq2e1y`)** · sub-int4 arch-impossible (#132) · verify-attn K-opt lift 0.0 (#441 `7rb089z3`) · CUDA-graph self-abort (#443 `qlvakiyu`) · async-pipelined 0.0 (#444 `0syyqxag`) · KV-prefetch Δ≈0 (#445 `emljqube`) |
| **DEMAND** (more accepts) | 4 | drafter-retrain fixed-topo NO-GO (#446 `uid28gdg`) · **bigger-drafter net-negative at ANY size (#451 `c675zor8`)** · tree-verify full-fanout −61.61 (#402) · tree-verify per-pos-DP +1.33 β-fragile (#409) |
| **FRESH-LITERATURE** | 5 | FlashInfer UNREACHABLE (#246) · n-gram/PLD lane closed (#250/#89/#81) · adaptive-K→static (#256/#266) · static-K −8.63% (#273 `51bdsbpw`) · megakernel NO-GO build |

**Best realized byte-exact lever = +0.2613 TPS** (#447). Realized blanket-strict frontier **467.14** (#423 `5a6zq2yz`) sits **−14.39** below the deployed (non-equivalent) **481.53** (#52 `2x9fm2zx`), −2.99 σ_hw.

### 2. The decisive reconciliation — #447 vs #450 (the heart of the annex)

The modeled **+15.86** (wirbel #442) looks both impossible (#447) and plausible (#450). Both are correct — **they speak to different kernels:**

- **#447 `crrq2e1y`** — "+15.86 impossible from a tile retune": measures the **only tunable Triton kernel** (Triton-3D attention, **1.27% of verify**). Best byte-exact retune = +0.26 e2e; deleting the whole kernel caps at +4.27; +15.86 ⇒ 279.9µs = 3.7× the kernel.
- **#450 `c5oyb7gv`** — "+15.86 physically plausible": measures the **aggregate int4-Marlin GEMM (85% of verify)** at **433 GB/s = 84% of read-peak (518 GB/s)** → a real **~16% BW headroom**; perfect `f→1` re-tile → 510.87 TPS (clears 481.53 by +29.3), and the 14.39 gap is **inside** the headroom.

**The ~16% int4-GEMM achieved-BW headroom is REAL but GREEDY-UNSAFE.** Marlin has no byte-exact tile knob (#448); the only way to cash the slack is FP-reassociating split-K re-tiling → byte-divergent. #450's realistic split-K recovers just +12.6…+31.4 TPS with `realistic_splitk_greedy_safe=false`. land #451 `c675zor8` closes the last hatch: even a demand push with topology free cannot net-beat 481.53 (breakeven φ′(1)=0.934 ≫ literature β_α≈0.02; critical β=2.78 ≈ 139× lit). **Roofline physics does not cap the search below 481.53; greedy-safety does** — which is exactly why the prize requires relaxing strict equivalence.

### 3. The modeled-in-isolation collapses (methodological appendix)

| lever | modeled | realized | run · PR |
|---|---:|---:|---|
| pinned-K | +13.998 | **−5.82** | `0pg4bz25` · #433 |
| cb3 | +15.60 | **0.0** | `hv4xpgf8` · #437 |
| autotune (Triton attn) | +15.86 | **+0.26** | `crrq2e1y` · #442/#447 |
| static-K | +13.2%/+4.28% | **−8.63%** | `51bdsbpw` · #256/#266/#273 |

Lesson: **always realize end-to-end; never report the isolated-op Δ** — exactly why the relax-prize cards (stark/ubel) must be MEASURED, not modeled.

### Command

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  research/equivalence_escalation/closed_lever_annex/annex_self_test.py \
  --wandb_group equivalence-escalation-anchors \
  --wandb_name denken/closed-lever-evidence-annex
# 0-GPU gate only: add --no-wandb  →  SELF-TEST 35/35 PASS
```

### Metrics vs baseline

| | value | baseline | note |
|---|---:|---:|---|
| closed_lever_count (primary) | **20** | — | 11 supply / 4 demand / 5 fresh-lit |
| annex_self_test_passes | **35/35** | — | JSON↔markdown cross-checked, no drift |
| official_tps | **0** | 481.53 deployed / 467.14 realized | analysis-only, no served change |
| ppl (test) | **2.3772** | 2.3772 (gate ≤ 2.42) | untouched by construction |

### Peak memory

0-GPU analysis card (JSON + markdown validation only); peak RSS ≈ a few MiB. `CUDA_VISIBLE_DEVICES=""`, no model load, no CUDA context.

### What happened — honest analysis

The equivalence-respecting program is genuinely converged. Across 20 distinct strict TPS levers — every cheap supply lever (#433/#437/#440/#448/#449/#447/#132), the full decode-loop sweep (#441/#443/#444/#445), both demand axes (#446 fixed-topology, #451 any-size-on-net), and the fresh-literature lane (FlashInfer/n-gram/adaptive-K/static-K/megakernel) — the **best realized byte-exact gain is +0.2613 TPS** (#447), and the realized frontier 467.14 sits 14.39 below the deployed 481.53. The single most important clarification: #447's "impossible" and #450's "plausible" both hold because they measure **different kernels** (tunable Triton attention 1.27% of verify vs the dominant vendored int4-Marlin GEMM 85% of verify). The int4-GEMM's ~16% BW headroom is real, but recoverable only by greedy-unsafe FP-reassociation — so **physics does not close the frontier; greedy-safety does**. That is the precise, auditable basis for escalating the relax-strict-equivalence question to the human (#407).

### Suggested follow-ups

- **Hand this annex to the capstone (fern)** as the defensible appendix for the #407 escalation — it is the complete closed-lever ledger with every run linked.
- If the human relaxes strict equivalence, the **first** lever to size is the greedy-unsafe int4-GEMM split-K re-tile (#450's +12.6…+31.4 TPS band) under a measured PPL/quality gate — it is the only material headroom and is MEASURED-ready, not modeled.
- **Housekeeping (not part of this PR):** `research/validity/eagle3_retrain_cost_sizing/` is an untracked prior card (#352) on this branch; I left it untracked and did **not** include it in this commit. Flagging so it isn't lost. The non-functional in-scope drafter ref scripts noted in #446 still want a future bug-fix PR.

### Public evidence used

Spine: denken #447 verify-wall map (`crrq2e1y`). Reconciliation pair: ubel #450 roofline (`c5oyb7gv`) + land #451 bigger-drafter net-ceiling (`c675zor8`). Incumbent: PR #52 (`2x9fm2zx`, 481.53, non-equivalent). Realized frontier: denken #423 (`5a6zq2yz`, 467.14). Verify-BW wall: #436 (`nvsbctji`, 520.95). All closing runs (#433/#437/#440/#441/#443/#444/#445/#446/#448/#449/#273) linked inline and verified live.
