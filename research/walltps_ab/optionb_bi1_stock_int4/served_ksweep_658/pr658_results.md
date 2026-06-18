STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["4pv32mbv"],"primary_metric":{"name":"served_walltps_best_K","value":148.766},"test_metric":{"name":"official_tps","value":0}}

## Results — Served wall-TPS K-sweep (rescued, acceptor-ON de-projection)

**Verdict: `FASTER_K_EXISTS`** (modest) — **K=5 is the rescued served-TPS peak; the shipped K=6 manifest is ~1.3% sub-optimal, and every K∈{5,6,7} holds #319 identity.** But the K-gain is small and the absolute level is harness-sensitive (see cross-val), so this is a *weak* faster-K, not a dramatic one. **LOCAL only — does NOT trigger a fire** (`analysis_only=true`, `official_tps=0`, no HF Job).

### Headline
- `served_walltps_best_K` = **148.77 LOCAL tps at K=5** (rescued de-projection), vs shipped K=6 **146.86** → **+1.91 tps (+1.30%)**; vs K=7 **133.32** → +15.4 (+11.6%).
- The drafter (`gemma4_assistant`, 4-layer Gemma4MTP draft proposer) empirically supports **K≥7** (#632 ran K=3..7, acceptance still rising at K=7) → the required {5,6,7} are all architecturally valid. K=8 not tested (not fabricated).

### Per-K table
Rescued served wall-TPS is a **de-projection** (stark #636/#642's method — the live in-engine acceptor is not yet a clean runnable number), self-consistent because all inputs are drawn from the *same* `int4_mtp_batchinv` BI=1, MAX_NUM_SEQS=1, greedy, 128-prompt/seed=1 served captures:

> `rescued_wall_tps(K) = 1 / ( 1/U(K) + f(K)/A )`  — each recompute fire costs one M=1 AR forward (A = local M=1 AR-rung = **77.96** tps, == stark arm-d 77.89).

| K | un-rescued U (local) | mean acc. len | fire-rate τ=0.5 | **rescued LOCAL** | rescued (stark-mix*) | on-AR head break | conf. miss >0.5nat | identity |
|---|---|---|---|---|---|---|---|---|
| 3 | 165.75 | 2.856 | 7.286% | 143.52 | 151.30 | 3.08% | **0** | HOLDS |
| 4 | 171.72 | 3.204 | 7.280%ⁱ | 147.99 | 156.26 | — | — | (not in sweep) |
| **5** | **172.74** | 3.474 | 7.274% | **148.77 ★** | 157.12 | 2.81% | **0** | **HOLDS** |
| **6** (ship) | 170.21 | 3.657 | 7.282%ⁱ | 146.86 | 155.01 | 2.53%ⁱ | **0** | **HOLDS** |
| **7** | 152.31 | 3.825 | 7.291% | 133.32 | 140.00 | 2.24% | **0** | **HOLDS** |

`ⁱ` = K-independent quantity interpolated for a K not directly measured by #648/#651 (immaterial — fire-rate spread across K is 0.017pp). `*`stark-mix = same formula but pricing the recompute at the **official** 126.378 rate (reproduced only so K=6 is directly comparable to stark's de-projection; not a local-measured number).

### The four required per-K measures
1. **Rescued served wall-TPS (headline):** K5 148.77 > K6 146.86 > K7 133.32 (LOCAL). K* = 5.
2. **Mean accepted length / verify step:** 3.474 → 3.657 → 3.825 (rises with K). *Note K=5 wins despite the SHORTER accepted runs* — see physics below.
3. **Recompute fire-rate (reused #648 τ=0.5 M=1 census):** 7.27% — **K-independent** (7.274/7.282/7.291% at K5/6/7), as #648 found. Confirmed it does **not** move with K.
4. **#319 identity (reused #654 `ar_ref_m1_canonical` oracle via #651 served census):** on-AR head break-rate 2.2–2.8% (falls with K), but **0 confident off-AR head misses (>0.5 nat) at EVERY K**. Every break is an int4 ULP/quantum tie (≤0.25 nat) with the AR token as the recompute runner-up → byte-exact at the int4 quantization floor. **No K breaks identity.** PPL = AR PPL = **2.0055** (unchanged by K, < 2.42 cap).

### Cross-validation leg — my K=6 vs stark #642's K=6
| quantity | land (this card) | stark #642 | gap |
|---|---|---|---|
| M=1 AR-rung (local) | 77.96 | 77.89 (arm-d) | **0.09%** ✅ |
| un-rescued K=6 (local) | **170.21** (#632) | **155.58** | **9.4%** ⚠️ |
| rescued K=6 (stark-mix) | 155.01 | *headline PENDING* (nearest: #636 K7 proj 139.20) | — |

**The two independent reads agree on the AR-rung anchor (0.09%) but the un-rescued K=6 throughput differs by 9.4%** — my #632 `wall_tps` runs ~9% hotter than stark's K=6 ceiling. Since the AR refs match, this is a spec-path *throughput-metric* difference, not a global hardware-speed difference. **This gap is the deliverable:** the K-*ordering* (K5 > K6 > K7) is rock-solid within my harness (un-rescued K5 172.74±0.01 vs K6 170.21±0.04, n=3, cv < 0.1% → ~60σ), but the **absolute** rescued level carries ≈9% harness uncertainty. So "does rescued clear 126.378 **OFFICIAL**?" is NOT answerable locally — it needs the HF benchmark. This tells us how much to discount the de-projection before spending quota.

### Why a lower K nets more (physics)
#632's cost fit: `cycle_time(K) = 1.86·K + 11.22 ms`. Each extra speculative token adds one M=1 BI-taxed draft forward to the verify cycle. Net wall-TPS = `e_accept(K)/cycle_time(K)` peaks at **K=5**: acceptance length keeps rising (3.47→3.83 across K5→K7) but the per-cycle cost rises faster. The recompute fire-rate is K-independent (7.27%), so it scales every K down by the same ~14% and does **not** change the K-ordering — the rescued optimum is the same K=5 as the un-rescued optimum.

### Baseline comparison (from PR body)
- Locked AR rung `int4_g128_lmhead` @ **126.378 OFFICIAL** — untouched (no fire).
- stark #642 un-rescued K=6 ceiling **155.58 local** → my independent un-rescued K=6 = 170.21 (9.4% hotter, above).
- #648 cost model cross-check (K=5): 141.05 — vs my rescued_local K5 148.77 / stark-mix 157.12. (#648 used stark's borrowed r=3.089 against a 172.74 base; this card prices fires at the directly-measured 77.96 AR-rung instead, hence the cleaner LOCAL 148.77.)
- #654 identity K3/5/7 on-AR head break ≤3.41%, 0 confident misses — reproduced exactly here.

### Exact commands (LOCAL, CPU-only analysis)
```bash
# rescued de-projection: reuses #632 un-rescued U(K), #648 fire-rate f(K), #651/#654 identity
python research/walltps_ab/optionb_bi1_stock_int4/served_ksweep_658/deproject_rescued_ksweep.py
# W&B closeout -> group served-ksweep-walltps-land
python research/walltps_ab/optionb_bi1_stock_int4/served_ksweep_658/log_658_wandb.py
```
Reused source captures (all `int4_mtp_batchinv`, VLLM_BATCH_INVARIANT=1, MAX_NUM_SEQS=1, greedy, 128×512, seed=1): #632 un-rescued sweep (K5 `uo6netrr` / K6 `obfvs9ma` / K7 `8sfauo3i`), #648 fire census `dyseni93`, #654 identity oracle `ah3fe0h1`.

### Logistics
- **W&B run:** `4pv32mbv` (group `served-ksweep-walltps-land`, `analysis_only=true`, `official_tps=0`).
- **Peak memory:** analysis is **CPU-only** (no GPU server booted this card). Reused source captures peaked at **~19,917 MiB** VRAM on the A10G (int4_mtp_batchinv BI=1 served stack).
- **Public/internal evidence used:** extends land #632 (un-rescued K-sweep), reuses land #648 (fire census) + #654 (`ar_ref_m1_canonical` oracle) via #651 (served-rescue census); cross-validates stark #642 / #636 published numbers (from PR body — stark's branch not inspected). Refutes the implicit assumption that the shipped K=6 is the served-TPS optimum.

### What happened — honest analysis
A K **does** beat the shipped K=6 (K=5, +1.30% rescued), and it holds identity, so technically `FASTER_K_EXISTS` — **but it's a weak win.** The un-rescued curve is flat-topped across K=4–6 (171.7 / 172.7 / 170.2), so K is a low-gain knob near the top; K=5 edges K=6 by <2 local tps. That +1.3% is **smaller than the 9.4% harness-sensitivity** I surfaced against stark, so I would not re-cut the manifest to K=5 on the strength of a local de-projection alone. The genuinely useful outputs of this card are (a) **K=6 is confirmed near-optimal** — the shipped manifest is not leaving meaningful served TPS on the table; (b) **identity holds at every K** (0 confident misses, K-independent) — the K-knob is identity-safe to tune; (c) the **de-projection is ~9% harness-sensitive at the un-rescued base** even though the AR-rung anchor matches stark to 0.1% — so any rescued number (mine or stark's) should be treated as ±~9% until an OFFICIAL HF benchmark lands. Nothing here is measured >126.378 official, so the #481 autonomous-fire trigger stays un-armed.

### Suggested follow-ups
1. **If/when an HF benchmark is approved for the rescued stack, request it at K=5 (not K=6)** — same identity, marginally faster, free to choose. But the +1.3% is within harness noise; K=6 is a safe default.
2. **Resolve the 9.4% un-rescued gap with stark** by aligning the exact `wall_tps` definition (full end-to-end vs steady-state window) and prompt ordering on one shared harness — that, not the K-knob, is the bigger lever on whether the de-projection clears 126.378 official.
3. **The live in-engine acceptor wall-TPS remains the one un-measured quantity** (stark #642's pending headline). Until it exists, every "rescued" number is a projection that under-prices real recompute serialization/KV-recompute overhead; a true live acceptor measurement would supersede this whole de-projection.
