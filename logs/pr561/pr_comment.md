STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["v74ad5jb"],"self_det":true,"fire_decision":"NO-FIRE","two_gate_satisfiable":false,"primary_metric":{"name":"base_fullhead_quality_safe_ceiling_tps","value":311.25},"test_metric":{"name":"gap_to_current_ship_tps","value":64.61}}

## Results — the two-gate FIRE-decision capstone: **NO-FIRE**, stated as one hard number

**TL;DR (read once).** The quality half is **MEASURED-PASS** on base_fullhead across all four downstream axes (≥90% of vanilla base on every one). The speed half is **FAIL**: the quality-safe hard ceiling is **311.25 TPS magically-free (#554) / ~292 strict (#544)** — hardware-rooted (denken #550 HBM byte-rate wall + my #554 fixed-overhead floor) and **below the current shipped 375.857-official ship by 64.61 TPS** (83.76 at the strict ceiling). The mirror config (osoi5 / surgical-357) is fast enough (375.857) but its quality **COLLAPSES** (the moat, wirbel #548). **No config is (PASS, PASS) → `two_gate_satisfiable=FALSE` → NO-FIRE.** Every open speed lever's pass-bar is pre-registered below and **none is satisfiable**: candidate-verify (#549/#560) caps at +44 → 309 < ship (structural); depth-drop (#546) needs an (identity ∧ quality ∧ +65 TPS) conjunction that is near-empty on a dense 4B; head-width (#547) is CLOSED. The framework axis is closed across four legs. `self_det=true` (19/19). Pure synthesis — `analysis_only=true`, `official_tps=0`, **0 GPU**, no HF job, no served-file change, BASELINE (481.53) untouched.

### Stage 1 — the two-gate truth table (PRIMARY)

| config | **quality gate** (Morgan #515 ≥90% of base) | **speed gate** (#524 beat ship 375.857) | both? |
|---|---|---|---|
| **base_fullhead** (the quality-safe ship) | **PASS** — 4/4 axes ≥90% | **FAIL** — ceiling 311.25 / 292 ≪ 375.857 | ❌ |
| **osoi5 / surgical-357** (current ship) | **FAIL** — moat collapse (#548) | **PASS** — 375.857 shipped | ❌ |

**The single decisive cell: no config is (PASS, PASS).** → `two_gate_satisfiable = FALSE`, `fire_decision = "NO-FIRE"`.

The four quality legs (base_fullhead, all clear the ≥90%-of-vanilla-base gate):

| axis | base_fullhead | % of base anchor | source (W&B) |
|---|---|---|---|
| MMLU-Pro | 0.636 | **95.2%** | stark #542 (`92pcnx6a`), anchor 0.668 ubel #538 |
| GPQA-Diamond | 0.4697 | **99.9%** | stark #542 (`92pcnx6a`) |
| GSM8K | 0.973 | **97.3%** | wirbel #541/#545 (`uqnkzlf9`), min_tokens=8 EOS guard (as-served 86.8% recoverable) |
| AIME maj@1 | 0.1444 | **118.17%** | fern #514/#535 (`xtanouk7`), = 118.17% of base 0.1222 |

### Stage 2 — the precise TPS gap to a fire (PRIMARY)

| gap | from magically-free **311.25** (#554) | from strict **292.1** (#544) |
|---|---|---|
| `gap_to_current_ship_tps` (vs **375.857**) | **64.61** | 83.76 |
| `gap_to_official_1_tps` (vs **481.53**) | 170.28 | 189.43 |
| (vs **live** public #1 508.63, digest 2026-06-17) | 197.38 | 216.53 |

- `base_fullhead_quality_safe_ceiling_tps` = **311.25** magically-free (head bytes→0, body intact, KV→0, fixed floor kept; #554) / **292.1** strict-via-int4-precision (#544, an *upper* bound — land #556 may lower it). The best identity-safe head lever, fern #549's candidate-verify, realizes **305.4** central (band **[293.8, 309.0]**) — and it sits **under** the magically-free 311.25, so the entire quality-safe band **[252.31 floor … 311.25 ceiling]** is below the ship.
- This 311.25 is **the number that would flip the verdict if it ever exceeded 375.857 at quality-PASS** — and it provably cannot: it is pinned from both ends (denken #550 byte-rate wall 482.9 GB/s = 80.5% of A10G peak + my #554 fixed-overhead floor 0.573 ms / 42 sequential SDPA launches).

### Stage 3 — pre-registered pass/fail bar for every open speed lever (PRIMARY)

So that when each lands there is **zero ambiguity**:

- **fern #560 candidate-verify served-realize** *(IN FLIGHT — single slot left open)*: **pass bar** = a served gain that lifts base_fullhead **above 375.857** at `argmax_identity_rate=1.0`. Pre-registered: #549 caps this lever at **+44 best → 309.0**, which is *structurally below* 375.857 by 66.9 TPS. **`candidate_verify_can_fire = FALSE` (structural).** The verdict holds whether #560 measures +28 or +44 — both ceil below the ship.
- **ubel #546 body depth-drop** *(un-landed)*: **pass bar** = the conjunction **(a)** `argmax_identity_rate=1.0` (dropping a transformer block changes the logits → near-certain #319 flip) **AND (b)** downstream quality ≥90% of vanilla base on all four axes **AND (c)** adds **≥ +64.61 TPS** over the 311.25 ceiling (≥ +83.76 over strict). `depth_drop_pass_bar_tps = 64.61`. **`depth_drop_conjunction_plausible = FALSE`** — the moat (#548) shows depth costs quality, and the (identity ∧ quality ∧ +65 TPS) conjunction is almost certainly empty on a dense 4B.
- **kanna #547 head-width**: already **CLOSED** — `fast_quality_safe_ship_exists=FALSE` (12k fails MMLU-Pro 0.550; 32k is the safe-minimum width at only modest TPS). **`head_width_lever_open = FALSE`.**

### Stage 4 — the verdict + the flip condition (SECONDARY)

**The airtight NO-FIRE verdict, one paragraph:** base_fullhead is a confirmed **quality-PASS** ship (MMLU-Pro 95.2% / GPQA-D 99.9% / GSM8K 97.3% / AIME 118.17%, all ≥90% of base), but its **hard speed ceiling 311.25 sits 64.61 TPS below the current 375.857 ship** and 170.28 below the official #1; the ceiling is hardware-rooted (byte-rate wall #550 + fixed-overhead floor #554, kernel-robust, KV-robust, framework-robust). The mirror config osoi5/surgical-357 is fast (375.857) but quality genuinely collapses (#548). **The framework axis is closed** (SGLang denken #498 + lawine #558; TRT-LLM fern #502; FlashInfer-standalone fern #507 — no alternate stack serves byte-identically AND faster), **the EXACT-head lever is structurally zero** (land #552: lossless prune = 0 rows), and **the head-width lever is closed** (#547). The only un-landed lever is ubel #546 depth-drop, whose pass-bar conjunction is near-empty. **`verdict_flip_condition`:** a *served*, #319-identity-1.0, quality-≥90% config measured **above 375.857 TPS** — provably forbidden for base_fullhead by the hardware ceiling, and never once produced by any quality-passing config.

### KEY OUTPUTS (W&B `v74ad5jb`, group `two-gate-fire-decision`)

- **Stage 1:** `two_gate_satisfiable=FALSE`, `fire_decision="NO-FIRE"`, `quality_gate_base_fullhead=PASS`, `speed_gate_base_fullhead=FAIL`, `quality_gate_osoi5=FAIL`, `speed_gate_osoi5=PASS`
- **Stage 2:** `base_fullhead_quality_safe_ceiling_tps=311.25` (strict 292.1), `gap_to_current_ship_tps=64.61` (strict 83.76), `gap_to_official_1_tps=170.28` (strict 189.43)
- **Stage 3:** `candidate_verify_can_fire=FALSE`, `depth_drop_pass_bar_tps=64.61`, `depth_drop_conjunction_plausible=FALSE`, `head_width_lever_open=FALSE`
- **Stage 4:** `verdict_flip_condition` (logged in full)
- `self_det=TRUE` (**19/19** self-tests), **peak GPU 0** (pure synthesis). `primary_metric = base_fullhead_quality_safe_ceiling_tps = 311.25`.

### Comparison vs baseline (PR body)

| | PR baseline | this capstone |
|---|---|---|
| quality gate (base_fullhead) | MEASURED-MET 4 axes | **PASS** (truth-table cell, all ≥90%) |
| quality-safe hard ceiling | 311.25 / ~292 | **synthesized as the speed-gate FAIL number** |
| current ship to beat | 375.857 official | gap = **64.61** (magfree) / 83.76 (strict) |
| official #1 | 481.53 | gap = **170.28** (live 508.63 → 197.38, wider) |
| two-gate satisfiable | implied NO | **FALSE, stated** + every lever's bar pre-registered |

### Exact command (LOCAL, analysis_only, 0 GPU, no fire)

```
cd target/ && .venv/bin/python research/two_gate_fire_decision/two_gate_fire_decision.py \
  --wandb-group two-gate-fire-decision --wandb-name lawine/two-gate-fire-decision
```

Artifact `research/two_gate_fire_decision/two_gate_fire_decision.json`. **Peak GPU 0** (no torch import, no served job, no microbench). W&B run **`v74ad5jb`** · `analysis_only=true`, `official_tps=0`, no HF job. Every cited number traces to a banked run (provenance map in the artifact): #544 `d44b61gj` · #551 `5rnkxttp` · #554 `fi8vr1nb` · #550 `5aobahij` · #552 `e4s81mih` · #549 `p9ga96xo` · #535 `xtanouk7` · #542 `92pcnx6a` · #541/#545 `uqnkzlf9` · #534 `ivpk7g7z`.

### Public evidence used

Public digest (`GET /v1/digest?as=senpai`, pulled 2026-06-17): the live public **#1 is now `ff-splitkv-frantic-fawindow-clean-v0-w256` at 508.63 TPS** (and `fawindow-w256` 505.88), with the unsafe **osoi5 lineage** sitting at **#4–5 ~489.63** — all *above* the PR-registered 481.53, which only **widens** `gap_to_official_1` and **hardens** NO-FIRE (I report the registered 481.53 as the pre-registered headline and the live 508.63 as the honest widening). The challenge remains **PAUSED on downstream quality** — the precise reason this is a two-gate decision, not a leaderboard race. This card **synthesizes** banked evidence (it does not re-measure): ceiling #544/#551/#554/#550/#552, quality #538/#542/#541/#535, head-width #547, anchor #553, framework #558/#507/#502/#498, Morgan #515/#524.

### What happened

The card asked for a synthesis, not a measurement, and the synthesis is decisive: **the two-gate is unsatisfiable, and the gap is a single hard number — 64.61 TPS** (magically-free) or 83.76 (strict). The most useful thing the capstone does is make the verdict robust to *every* remaining uncertainty: (1) it holds across the whole quality-safe ceiling band [252.31 … 311.25], since even the magically-free upper bound (which no real lever achieves) is below the ship; (2) it holds whether fern #560 lands the candidate-verify lever at +28 or +44, since +44 → 309 still ceils below 375.857; (3) it holds against the live leaderboard moving to 508.63, which only widens the official gap. One honest axis-caveat: 311.25 is *local* and 375.857 is *official* — mapping to a consistent axis via my #267 transfer (official ≈ 1.0352·local) makes the same-axis gap ~52–54 TPS, **tighter but the same sign** (not load-bearing; the raw 64.61 gap already settles it). The NO-FIRE verdict is now airtight, forward-looking, and reads in one pass.

### Suggested follow-ups

- **Feed this packet straight into Morgan #524** as the closing two-gate artifact: quality PASS, speed ceiling 64.61 TPS below the ship, no open lever's pass-bar satisfiable. The only un-landed slot is ubel #546 depth-drop — and its pass-bar conjunction (identity ∧ quality ∧ +65 TPS on a dense 4B) is the single remaining thing to confirm-empty before declaring the speed frontier exhausted.
- **When fern #560 lands**, drop its measured served number into the pre-registered `candidate_verify_pass_bar_tps=375.857` slot; the structural bar (309 < ship) means it confirms NO-FIRE either way — a clean closed loop, not a re-open.
- **Do not re-open the head, head-width, kernel, KV, or framework lanes** — all six convergent lenses + four framework legs are banked NO-GO. The only physical lever that could, in principle, move the ceiling is a faster identity-safe attention kernel (denken #550's per-byte end), and that attacks per-byte rate, not the launch-count floor — out of scope and #319-gated.
