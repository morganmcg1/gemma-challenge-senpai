## Speed Pareto (M=1 single-stream; local A10G proxy → official via τ=1.0245 anchored on AR)

_Local AR steady_gen_tps = 123.35 → τ = 126.378/123.35 = 1.0245; proj_official = τ·S_local._

| config | S_local | proj_official_TPS | vs AR 126.378 | E_accept | accept_rate |
|---|---|---|---|---|---|
| ar | 123.35 | 126.38 | +0.0% | — | — |
| ng_max2_k3 | 98.39 | 100.80 | -20.2% | 2.090 | 0.364 |
| ng_max2_k5 | 102.40 | 104.91 | -17.0% | 2.337 | 0.268 |
| ng_max2_k7 | 102.40 | 104.91 | -17.0% | 2.470 | 0.211 |
| ng_max3_k3 | 99.86 | 102.31 | -19.0% | 2.128 | 0.376 |
| ng_max3_k5 | 102.40 | 104.91 | -17.0% | 2.376 | 0.276 |
| ng_max3_k7 | 102.76 | 105.29 | -16.7% | 2.510 | 0.217 |
| ng_max4_k3 | 99.66 | 102.11 | -19.2% | 2.165 | 0.388 |
| ng_max4_k5 | 105.53 | 108.12 | -14.5% | 2.412 | 0.283 |
| ng_max4_k7 | 102.44 | 104.95 | -17.0% | 2.549 | 0.222 |

_MTP candidate reference: 427.7 official-proxy TPS._

**Best-speed ngram config: `ng_max4_k5` → proj 108.12 TPS.**


## Identity census (#319, WARM free-run greedy, full 128×512, seed 1)

- **Cross-start floor (AR vs AR', dev307 control):** GREEDY_IDENTICAL — 0/128 prompts differ, 0 tokens.
- **decode_ng_max4_k5_vs_ar_ref:** DIVERGENT — 99/128 prompts differ, 28553 tokens.

## GSM8K quality (sampled T=1.0/top_p=0.95/top_k=64, min_tokens=8, bar 0.807)

- **ng609_ar_int4g128_sampled:** acc=0.8840 (n=500) [PASS vs 0.807] strict=0.92 trunc=0.088
- **ng609_ngram_max4k5_sampled:** acc=0.8900 (n=500) [PASS vs 0.807] strict=0.932 trunc=0.078