# PR #517 — Cross-axis reopen-action capstone (one recommended fire + a risk-keyed decision tree)

**Run:** `ui5a48ax` · `--wandb_group reopen-capstone` · analysis_only, official_tps=0, CPU-only.
NO serve, NO HF job, NO `--launch`, NO submission, NO served-file change, NO evals run.
Pure composition of three **banked** dossiers — re-derives nothing.

## The single reviewer-facing line (paste this)

> **Reopen action: publish/fire surgical-357 (SHIPPED, official `summary.json:tps` = 375.857, `j7qao5e9`) as
> primary** — the only rung already landed on official a10g-small, quality-**dominant** (#512), operative-1.0
> (1 bf16-ULP near-tie, 0 semantic), and the #508 **bracketed-SHIP** choice. **Hold floor-lock-166.23**
> (literal-1.0, private-safe) as invalidation insurance; **stage byteexact-399** (+42 local TPS, 0 semantic,
> 5 ULP e2e) as the forward speed-upgrade, gated on one official launch **and** the human's draw-standard
> ruling (#500 fu#3). **No reopen rule lets the quality pause hurt the ship.**

## Recommended primary action (table-driven, not a hard-coded lean)

| role | rung | number | why (derived from rung fields) |
|---|---|---|---|
| **PRIMARY (now)** | **surgical-357** | **375.857 official** (`j7qao5e9`) | only rung with `shipped=True ∧ official_tps>0`; quality-dominant (#512); operative-1.0; #508 bracketed-ship. Score locked — only the human `--publish` nod remains. |
| **FALLBACK** | floor-lock-166.23 | 166.23 (literal) | the only `literal-1.0 ∧ private_safe` rung → guaranteed-valid under a literal private-identity rule / maximin. P(surgical < floor-lock on raw TPS) ≈ 0 (51σ) → never a speed case, purely a validity hedge. |
| **FORWARD UPGRADE** | byteexact-399 | 399.97 local matched / 444.82 fire-time | fastest not-yet-shipped quality-safe rung. **Not yet actionable:** no official number (1 launch needed) + operative/quality-equivalent (5 ULP e2e), so it needs the human's draw-standard ruling (#500 fu#3). |

## Decision tree (keyed on the human's risk preference)

| risk preference | predicate | → fire | headline TPS |
|---|---|---|---|
| **zero private-speed-risk required** | invalidate on LITERAL private greedy identity, or maximin/guaranteed-floor | **floor-lock-166.23** (literal-1.0, private-safe) | 166.23 |
| **max quality-safe speed, byte-identical tokens preferred** | penalize-breach / speed-invalidate w/ E-value; integrity posture wants ≤1-flip census | **surgical-357** (SHIPPED) | 375.857 official |
| **max quality-safe speed, operative (0-semantic) acceptable** | byte-identical-token NOT required (realized scorer has no token-identity gate, #493) | **byteexact-399** (+42 local) | 399.97 local |

**Decisive fork = two axes.** (A) the private validity rule/objective: literal-identity or maximin → floor-lock;
penalize or speed-invalidate-E-value → the fastest valid rung. (B) token-identity strictness of the speed rung:
byte-identical-token → surgical-357 (shipped); 0-semantic operative acceptable → byteexact-399. There is **no axis
on which downstream quality hurts us** — every rung is quality = base.

## Per-rung risk/reward table (every number resolves to a banked run)

| rung | official TPS | best-local TPS | private band (mean [95%]) | worst-case | census flips (semantic) | quality (MMLU/GPQA) | label | draw-ready | dossier |
|---|---|---|---|---|---|---|---|---|---|
| **floor-lock-166.23** | 166.23 | 166.23 | 166.23 [162.97–169.49] (0 breach) | — | **0 (0)** | 0.668 / 0.470 = base | **literal-1.0** | pre-staged fallback | #508 `fn2v5wox` |
| **surgical-357** | **375.857** | 357.6 | **341.9 [335.2–348.6]** (4.3% breach) | 24%-WC 95-lo **266.2** | **1 (0)** | 0.668 / 0.470 = base | **operative-1.0** | **SHIPPED** | #508 `fn2v5wox` + #512 `3fxrmc8u` |
| **byteexact-399** | 0 (never launched) | **399.97 / 444.82** | *null — not separately propagated* | *null* | **5 (0)** | 0.668 / 0.470 = base prior | **quality-equivalent** | staged + locally certified (r1-r2=1.0) | #500 `m76qbs3l`/`feof8wtk`/`rvl5w50z` |

Honesty notes baked into the artifact: byteexact-399's official TPS is **unmeasured** (`official_tps=0`); its `+42`
is a **local matched-workload delta** (399.97 vs surgical 357.6 at 32×256); its private band is **`null`, never
faked**. Surgical's 375.857 official is **+5.2% above** its ~357.6 local pod (official-faster-than-pod, **not**
drift) — the private band ≈342 is a *separate* local→private 4.3% breach anchored on the 357.22 local public
anchor (#508).

## Quality context — the pause selects FOR us (all three rungs pass; pruned competitors collapse)

Morgan's gate (#483): MMLU-Pro ≥ 0.60, GPQA-Diamond ≥ 0.42. All three of our rungs = base (0.668 / 0.470) → **PASS**.
Pruned-substrate competitors **FAIL** (0.330 / 0.283, GPQA near the 0.25 chance floor). The token-identity
difference between our rungs is a *speed/strictness* axis, **not** a quality axis.

## Inputs (banked dossiers + advisor-provided anchors — reused, not re-derived)

| input | value | source run |
|---|---|---|
| surgical-357 official TPS / PPL | 375.857 / 2.37673 | ship `j7qao5e9` (stark #499) |
| surgical-357 private band | 341.9 [335.2–348.6]; 24%-WC 95-lo 266.2 | #508 `fn2v5wox` (← #504 `0urxqwob`) |
| surgical-357 census | 1 bf16-ULP near-tie, 0 semantic | stark #494 `k8nqmc2b`/`5fxw18gu` |
| quality verdict (dominant) | MMLU 0.668 / GPQA 0.470 = base | #512 `3fxrmc8u` |
| floor-lock TPS / label | 166.23 / literal-1.0 | #508 `fn2v5wox` (← stark #485 `pavotwci`) |
| byteexact-399 economics | 444.82 / 399.97 local, 0 semantic, 5 ULP, PPL 2.37666 | #500 `m76qbs3l`/`feof8wtk`/`rvl5w50z` |
| quality gate / anchors | MMLU≥0.60, GPQA≥0.42; base 0.668/0.470; pruned 0.330/0.283 | Morgan/dixie #483 |
| realized scorer (no token-identity gate) | drift≤5%, PPL≤2.42, 128/128 | stark #493 `xuvmnpav` |

## Self-test (`self_test.passes = True`, 45/45)

Reproduces each banked headline number exactly (375.857 official, 341.88 private mean, 266.17 24%-WC, 166.23
floor-lock, 399.97 / 444.82 byteexact local); 3 distinct rungs with the three distinct labels; all three pass the
quality gate with 0 semantic flips while the pruned competitor fails (GPQA near chance); surgical dominates
floor-lock on raw TPS (95-lo > floor 95-hi, even the 24% worst-case); the +42 is a local delta and byteexact's
private band is `null` not faked; the decision tree's three branches each resolve to the intended rung by a
data-driven rule and cover all three rungs; the recommendation derives primary=surgical(shipped) / fallback=floor-lock(literal) /
forward=byteexact; labels consistent with each dossier verdict (#508 bracketed, #512 dominant); PR discipline
(`analysis_only`, `official_tps=0`, all rungs PPL ≤ 2.42); every numeric leaf finite.

## Command

```bash
.venv/bin/python -m research.reopen_capstone.compose_reopen_action \
    --name kanna/reopen-action-capstone --group reopen-capstone
```

Peak memory: negligible (pure-Python CPU composition; no model load, no serve, no eval). W&B run `ui5a48ax`
(`reopen_action_capstone` artifact attached).
