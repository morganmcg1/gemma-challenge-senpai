STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["2gvuanyx"],"primary_metric":{"name":"p_rescued_clears_126378_after_haircut","value":0.023245},"test_metric":{"name":"worst_corner_margin_pct","value":-6.462}}

## Results

### Headline verdict: **NO-GO (speed).**

The rescued τ=0.3 is identity-clean by construction (land #720 owns that cert) but it **fails the 126.378 speed bar after the documented private haircut**. Applying the exact #725 haircut-calibration machinery to the rescued config's ~135 local capture:

- **P(clears 126.378 after the 4–9% haircut) = 0.023** over the PR pessimism band T∈[0.87, 0.99], and only **0.175** even over the *favorable* measured-anchored band [0.960, 1.0].
- At the **cleanest possible anchor** — the captured-graph **meter-matched transfer T = 0.9971** — official-equiv **public** is 134.87 (+6.7%), but the worst documented **9% haircut drops it to 122.73 (−2.88%, below the bar)**, and even the **mid 6.5% haircut lands at 126.11 (−0.22%, at the bar).**
- It survives a haircut of only **6.3%** at the meter-matched transfer — *inside* the 4–9% band. To clear after the **full 9% haircut**, the transfer would have to exceed **1.027 — above the definitional 1.0 ceiling — which is impossible.**
- **Every** named transfer point fails the 9% haircut (grid below). The config clears only in a thin sliver: transfer ~definitional **and** haircut ~4% (thinnest clearing corner **+1.7%**).

> Contrast the un-rescued stack you banked in #725: P=1.00, worst measured corner **+9.95%**. The **~22% rescue tax (172.7→135.3 local)** is exactly enough to convert a comfortable GO into a NO-GO. **Do not spend a fire on the rescued τ=0.3 for speed.**

### Methodology note — input provenance + an isolation flag for you

This is the **speed** leg and, like #725, it is **analysis-only** (no HF Job, no server booted — #725 itself never booted a server; it read stark's K=5 wall_tps from an on-branch JSON and ran the MC). I applied the identical model to the rescued config's local captured TPS.

**Flag:** the PR directs me to "use land #720's working rescued serve path (or denken's `int4_mtp_rescued_tau03` package)… cross-read land/denken." **This launch's isolation scope restricts me to the advisor branch + kanna branches**, and the rescued serve path is **not on the advisor branch** (`submissions/int4_mtp_batchinv` here is the *un-rescued* K=6 stack — no recompute acceptor; `spec_regime_isolation_664/` is not in-tree). So I could not freshly boot/meter the rescued config. Exactly as #725 consumed stark's K=5 number, I consumed the rescued config's **local capture as consolidated on the advisor branch** (`CURRENT_RESEARCH_STATE.md` / land #664 `REGIME_IS_ACCEPTANCE`, W&B `4fbu9b3o`) and as **stated in this PR's own body** ("~135 local / ~134.87 official-equiv"):

| quantity | value | source (on advisor branch) |
|---|---|---|
| rescued τ=0.3 captured local (captured-graph, BI=1, vLLM 0.22.0) | **135.27** | stark #663 "captured 135.27 to the decimal" |
| AR-rung `int4_g128_lmhead` captured-graph local (meter anchor) | **126.75** | land #664 chain |
| official AR rung (locked bar) | **126.378** | `int4_g128_lmhead` PR#4, W&B `905tbujn` |
| ⇒ meter-matched transfer T = 126.378/126.75 | **0.9971** | (captured-graph, int4, the literal bar) |
| ⇒ official-equiv **public** = 135.27 × 0.9971 | **134.87** | == consolidated number, no haircut |

The verdict is **robust to ±5% on the 135.27 input** (the break-even-at-9%-haircut transfer stays > 1.0 for any local ≤ 138.9). **If you want a fresh A10G capture, land the rescued serve path on `approval-gated-8gpu-20260613` and I'll re-run step 1** — but it will not change the NO-GO.

### Why this anchor is *cleaner* than #725's (not weaker)

#725's lone meter-matched pair was the **bf16 flagship** (right meter `wall_tps`, wrong precision). Here the lone clean pair is the **AR rung itself**: **int4 precision + captured-graph meter + it is the literal speed bar.** So `n_clean_meter_matched_pairs = 1`, but it is precision+meter+bar matched — strictly **better**-anchored than #725. The transfer is ~definitional (0.997) because captured-graph local ≈ official `output_throughput` (unlike `wall_tps`, which read ~6% low and let #725's transfer legitimately run >1).

### Deterministic grid — official-equiv AFTER the worst 9% haircut

| transfer point | T | official-equiv @9% haircut | margin | clears? |
|---|---:|---:|---:|:---:|
| meter-matched (captured-graph, **clean anchor**) | 0.9971 | 122.73 | **−2.88%** | ❌ |
| definitional ceiling | 1.0000 | 123.10 | −2.60% | ❌ |
| int4 single_stream match (#725, cross-meter) | 0.9863 | 121.41 | −3.93% | ❌ |
| int4 single_stream pessimistic (#725, cross-meter) | 0.9603 | 118.21 | **−6.46%** | ❌ |
| stark tax 0.870 | 0.8700 | 107.09 | −15.26% | ❌ |
| advisor 0.880 | 0.8800 | 108.32 | −14.29% | ❌ |

*(The cross-meter single_stream points push the verdict MORE negative, not less — they cannot rescue it.)*

### Monte Carlo P(clears) + percentiles (400k draws each, T × haircut both uniform)

| band | P(clears) | public p50 | private p05 | private p50 |
|---|---:|---:|---:|---:|
| **PR pessimism [0.87, 0.99]** (headline) | **0.023** | 125.8 | 110.1 | 117.6 |
| measured-anchored [0.960, 1.0] (favorable) | 0.175 | 132.6 | 120.0 | 123.9 |

### Break-evens (the decision-grade numbers)

- **Transfer needed to clear after 9% haircut: 1.027** (above definitional 1.0 → unreachable).
- Transfer needed to clear after 4% haircut: 0.973 (clears only if transfer ≈ meter-matched).
- **Max survivable haircut at the meter-matched transfer: 6.3%** (straddles the 4–9% band → knife-edge that fails on the haircut tail).

### Honesty caveats (carried forward from #725)

- **`n_clean_meter_matched_pairs = 1`** — but precision+meter+bar matched (tighter than #725).
- **Private-drift prior is UNMEASURED in-scope.** I used the documented 4–9% band (kanna #44 drafter-collapse prior). The recompute acceptor is **stricter** (lower base acceptance), so its private E_accept drift direction is unknown — the 4–9% band could be optimistic *or* pessimistic for it. This is the single biggest residual uncertainty and it does not help the verdict (even h=0 needs T>0.934; only the meter-matched/definitional transfers clear with *no* haircut at all).
- **Thinnest clearing corner: +1.7%** (T=0.99 × 4% haircut) — the only place it clears.
- **Input not freshly measured** (see isolation flag above).

### Command

```bash
cd target/
python3 research/official_equiv_calibration/rescued_tau03_speed_gonogo.py --wandb \
  --wandb_group rescued-tau03-speed-gonogo --name kanna/rescued-tau03-speed-gonogo
```

- **W&B run:** `2gvuanyx` — group `rescued-tau03-speed-gonogo`, `analysis_only=1` / `official_tps=0` / `no_hf_job=1` / `fires=0`, state finished. Locked `int4_g128_lmhead` @ 126.378 untouched; no HF Job, no submission, no served-file change.
- **Peak memory:** N/A — analysis-only (NumPy transfer-model + Monte Carlo on CPU; no server, no GPU).

### What happened

The rescue (identity cleanliness via the τ=0.3 recompute acceptor) costs ~22% of the spec speed (172.7→135.3 local). At that speed, the public official-equiv (134.87, +6.7%) is **smaller than the documented private haircut can eat**: the break-even haircut is 6.3%, squarely inside the 4–9% band. So unlike the un-rescued stack — which cleared even at its worst measured corner — the rescued config falls below 126.378 across nearly the entire plausible transfer×haircut space (P≈0.02–0.18). Identity-cleanliness protects the *output tokens*, **not the speed**: it is still a drafter stack whose TPS depends on acceptance, which shifts on private prompts. **The strict-safe fallback is a speed NO-GO; it is not fire-worthy on speed, and the fire decision (#730) should not bank it as the clean alternative.**

### Suggested follow-ups

1. **Recover the rescue tax before reconsidering.** The 135.27 regime ships the *stock* Hub drafter (land #664). If the quality-neutral drafter swap (land #664's "one +10 lever", local `/tmp/qat-assistant` → projected ~147.55 official-equiv) were combined with the τ=0.3 recompute acceptor, the post-9%-haircut corner would be ~147.55×0.91≈134 (+6.2%) — a GO. The lever is *acceptance*, not the acceptor; the rescued path only becomes fire-worthy on speed *with* a faster drafter. (Gated on land #670's robustness verdict.)
2. **Measure the rescued config's *actual* private E_accept drift** rather than borrowing the 4–9% drafter-collapse band. The recompute acceptor's stricter accept rule may drift differently; a direct private-proxy probe (kanna #44 machinery) would replace the band's biggest caveat with data and could move the break-even.
3. **Fresh A10G re-capture** of the rescued config under captured-graph BI=1 once its serve path is on the advisor branch — to retire the "not freshly measured" caveat (will not change NO-GO, but closes the provenance gap).
