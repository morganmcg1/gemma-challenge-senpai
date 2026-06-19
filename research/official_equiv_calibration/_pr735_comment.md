STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3hdyip2b"],"primary_metric":{"name":"official_equiv_public_authoritative","value":167.9},"test_metric":{"name":"p_clears_126378_after_4to9pct_haircut","value":1.0}}

## Results — ONE defensible official-equiv for the #730 candidate

### Headline (the number for the fire packet)

> **Official-equiv = 167.9 TPS public / ~157 private (mid 6.5% haircut). P(clears 126.378 after the 4–9% haircut) = 1.00.** Worst defensible corner **+17.7%** (148.7 TPS, T=0.960 × 9% haircut). Breakeven transfer = **0.816** — the fire fails only below it, *under* every measured int4 pair and under the advisor's own 0.880 (which still clears at +7.8%).
>
> **GO is robust — but treat 167.9 as an UPPER BOUND.** The measured local used the **fast `/tmp/qat-assistant`** proxy, not the stock-Hub drafter the #730 submission ships (see Provenance). Even the slower stock projection (147.55) clears post-9%-haircut at **+6.2%**.

This **supersedes the scattered 139.9 / 147.55 / ×1.192** projections with one int4-precision-anchored number on the correct config.

### Step 1 (the crux): same-substrate? → **NO**, so #728's ×1.192 is **INVALID** here

The packet's "≈172.7 stock-drafter K=6 local" traces to `research/walltps_ab/optionb_bi1_stock_int4/ksweep/k5/paired_ab.json` (land **PR#82**, branch `land/optionb-bi1-k-sweep`). That measurement's `override_env`:

| field | measured value | #730 ship config | #728 AR base (106.02) |
|---|---|---|---|
| `NUM_SPECULATIVE_TOKENS` | **5** (→ it's K=5) | 6 | — (AR) |
| `DRAFTER_MODEL` | **`/tmp/qat-assistant`** (fast) | stock-Hub `…qat-q4_0-unquantized-assistant` | drafter OFF |
| `MODEL_ID` | `…qat-w4a16-ct` | `…qat-w4a16-ct` ✅ | `/workspace/…/int4_g128_lmhead` |
| engine | PR#82 campaign | (HF runner) | fresh faithful 0.22.0 BI=1 |

#728's anchored **×1.192 = 126.378 / 106.02** cancels the local→official gap **for #728's own faithful-engine + `int4_g128_lmhead` checkpoint**. Land's 170.21 is a **different checkpoint (`w4a16-ct`) on a different engine**, so its local clock already embeds a different local→official relationship. Applying ×1.192 to it (170.21 → **202.9**, or the K=5 172.74 → 205.9) **double-counts** an engine/checkpoint gap the ratio was never meant to cross. ⇒ anchored form rejected.

**Speedup-ratio fallback is unavailable in-branch:** the entire optionb k-sweep has **no drafter-off AR arm** (every arm is spec, K∈{3..7}) → there is no same-substrate AR base for land's 170.21. Gap flagged; I **bound** with named int4 transfer points exactly as #725 did, carrying #725's 4–9% haircut so the number is comparable.

### Step 2: the authoritative transfer rule (written, explicit)

> **For an int4 candidate, anchor the local→official transfer on int4 same-precision pairs.** In-branch these cluster at **T ∈ [0.960, 1.000]**, central **T = 0.986**:
> - `T_int4_match = 0.9863` — `int4_g128_lmhead` 126.378/128.13 (same precision as the candidate) ← central
> - `T_int4_pess  = 0.9603` — `lmhead12k` 126.378/131.60 ← pessimistic floor
> - `T_definition = 1.000`  — wall_tps == official `output_throughput` ← ceiling
> - (#732's captured-graph meter-matched pair independently gave **0.9971 ≈ definitional** — corroborates.)
>
> **Reject** for the authoritative number: `T_flagship 1.060` (bf16 — wrong precision; an int4 meter doesn't inherit the bf16 low-read tail), `T_advisor 0.880` (below every int4 pair), `T_728 1.192` (cross-checkpoint, invalid above).
> *Aside:* the bf16 flagship's 1.060 is on the **same PR#82 engine** as the 170.21 → that engine's local wall_tps reads ~6% **low**, so 0.986 is, if anything, conservative for this engine.

### Step 3: reconciliation — why the three estimates disagree

| estimate | local used | transfer | official-equiv | authoritative? | what drives the gap |
|---|---|---|---:|:--:|---|
| **#725 packet 139.9** | 159 (advisor verbal) | 0.880 | 139.9 | ❌ | low *unverified* local **×** transfer below every measured int4 pair — doubly conservative |
| **#732-fn 147.55** | ~170 (fast proxy) | ~0.87 ("stark tax") | 147.55 | ❌ | fast-drafter proxy **×** pessimistic transfer; `CURRENT_RESEARCH_STATE.md` labels it "fast `/tmp/qat-assistant`, **provenance-suspect**" |
| **#728-anchored** | 170.21 (K6) | 1.192 | **202.9** | ❌ | **INVALID** — cross-checkpoint (`int4_g128_lmhead` ≠ `w4a16-ct`) + cross-engine ratio misapplied |
| **THIS (authoritative)** | **170.21 (true K=6)** | **0.986** (band .960–1.000) | **167.9 pub / 157 priv** | ✅ | int4 same-precision, correct K, correct base model; carries #725's 4–9% haircut |

The spread is entirely **(local input) × (transfer choice) × (substrate validity)**. 139.9 and 147.55 are the *same family* (low-ish local × a sub-int4 "tax" transfer); ×1.192 is a different-substrate artifact. None is the int4-anchored central.

### Step 4: honest caveats (provenance is thin — the human must see this before firing)

1. **K mismatch.** The packet's 172.7 is **K=5** (NUM_SPEC=5). The true **K=6** = **170.21** (NUM_SPEC=6, same file's `k6/`). I used 170.21 — correct K *and* ~1.5% more conservative.
2. **Drafter mismatch — the load-bearing caveat.** Every optionb measurement used **`/tmp/qat-assistant`** (a local path, can't run on the HF a10g runner). The #730 **manifest ships the stock-Hub drafter** `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`. `CURRENT_RESEARCH_STATE.md` (cycle-58DG) explicitly splits *"134.87 (shippable stock-Hub drafter) / 147.55 (fast /tmp/qat-assistant, provenance-suspect)"*. ⇒ **170.21 is an UPPER BOUND** on the literal stock-Hub un-rescued K=6 local, which is **unmeasured on-branch**. The fired config sits **at or below 167.9 public** (bounded below by the rescued-stock 134.87-equiv, since un-rescuing only adds speed); even the slowest named stock projection clears (147.55 → 134.2 after 9% haircut, **+6.2%**).
3. **Substrate gap.** No same-substrate AR base exists for the 170.21 (no AR arm in the sweep) → speedup-ratio form unavailable; bounded via transfer points.
4. **Separate 5% private-repro validity gate** (not the bar): the 4–9% haircut band straddles 5%; a drafter stack can beat 126.378 yet be DQ'd (kanna #44). This — not the bar margin — is the real residual risk, and it is **unmeasured** for the stock drafter.
5. **lawine's direct `lawine-publishable-spec-ceiling` measurement of the publishable drafter will supersede this projection** when it lands; this is the defensible number to use *right now*.

### Command
```bash
cd target/
python3 research/official_equiv_calibration/officialequiv_reconcile_730.py --wandb \
  --wandb_group kanna-730-officialequiv-reconcile --name kanna/730-officialequiv-reconcile
```

- **W&B run:** `3hdyip2b` — group `kanna-730-officialequiv-reconcile`, `analysis_only=1` / `official_tps=0` / `no_hf_job=1` / `fires=0`, state finished. Locked `int4_g128_lmhead` @ 126.378 (W&B `905tbujn`) untouched; **no HF Job, no submission, no fire.**
- **Peak memory:** N/A — analysis-only (NumPy 400k-draw Monte Carlo on CPU; no server, no GPU).
- **Inputs (all in-scope on this branch):** `optionb_bi1_stock_int4/ksweep/{k5,k6}/paired_ab.json` (land PR#82, consolidated here), `local_official_projection/projection_cal.json` (flagship 1.060), `spec_achievable_ceiling/runs/sweep/report.json` (#728 ×1.192, AR base 106.02), #725 `0gpahz4c` / #732 `2gvuanyx` machinery.

### What happened
The three estimates were never about the same thing: **139.9** = advisor 0.880 floor on a 159 verbal local; **147.55** = the *fast proxy* under a stark-tax transfer; **×1.192→203** = a cross-checkpoint over-count. Anchoring the **correct K=6 local (170.21)** on the **int4 same-precision transfer (0.986, band 0.960–1.000)** and carrying #725's haircut gives **167.9 public / ~157 private, P(clears 126.378) = 1.00, breakeven 0.816**. The decision is robust: even the advisor's own 0.880 and the slower stock projection both clear after the full 9% haircut. The one thing that is *not* settled is **which drafter the packet means** — the measured number is the fast proxy and bounds the stock ship config from above.

### Suggested follow-ups
1. **Resolve the drafter before firing.** Confirm whether the #730 fire serves the stock-Hub drafter (manifest) or the fast `/tmp/qat-assistant` (all local measurements). If stock-Hub, the honest central drops toward ~150–160 public (still clears); the cleanest fix is lawine's direct publishable-drafter measurement.
2. **Measure the stock-Hub un-rescued K=6 local directly** (one local serve, no fire) to retire the upper-bound caveat — it is the single biggest provenance gap.
3. **Probe the stock drafter's private E_accept drift** (kanna #44 machinery) to replace the borrowed 4–9% haircut band with data for the 5% validity gate — the real residual risk.
