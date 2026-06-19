STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["shmwht1r"],"primary_metric":{"name":"p_dq_private_gate_at_prior","value":0.80},"test_metric":{"name":"accept_drift_budget_delta_max","value":0.05}}

## Results — the #730 private-DQ risk: acceptance-drift budget + P(DQ)

### Headline (the number for the fire packet)

> **The binding private-DQ gate is the 5% TPS-reproduction gap, NOT the 126.378 bar step-1 names. Its acceptance-drift budget is a flat δ_max = 5% (central-INVARIANT); at the documented 4–9% drift prior, P(DQ) = 0.80. The bar (PR step-1 literal) needs an 18.5–24.7% drift and never trips first (P(DQ)=0.00).**
>
> **Of the two risks the fire buys info on, the private-repro DQ DOMINATES the identity-gate roll** (the identity gate is already known-favorable — #38: the official gate has no token-identity check). **Recommend HOLD the fire until lawine #734 pins the stock drafter's δ_stock < ~3.5%** (comfortable margin under the 5% gate), or fire only if the human knowingly accepts ~80% DQ odds under the documented band.

This is the successor to my #735 (`3hdyip2b`): #735 settled the **speed** leg (P(clears bar)=1.00). #737 says the **speed leg was never the binding risk** — the 5% private-repro gate is, and it is live and unmeasured for this drafter.

### The load-bearing finding: which gate actually DQs?

PR step-1 operationalizes "DQ" as "private official-equiv falls under **126.378**". But the program's **actual** binding validity rule is different and bites ~4–5× sooner. Two candidate DQ conditions:

| gate | DQ condition | δ_max budget | central-dep? | P(DQ) @ prior | binds? |
|---|---|---:|:--:|---:|:--:|
| **G1 — 5% TPS-repro gap** | private TPS < 0.95 × own public | **5.0%** | **no** (relative) | **0.80** | **✅ BINDING** |
| G2 — bar-crossing (step-1) | private official-equiv < 126.378 | 18.5–24.7% | yes | 0.00 | ❌ never first |

**Public evidence this is the binding gate:** the live leaderboard carries a `verification` field (right now **4 valid / 6 pending** — submissions sit "pending" until the organizer's **private re-run**, then flip valid or DQ). `BASELINE.md` (L36–37): *"the verifier re-runs on a private prompt set; top drafter stacks lose 4–9% TPS… submissions **die on the 5% TPS-reproduction gap**, not on PPL."* Empirically anchored: flagship #52 organizer re-run **Δ4.3% ≤ 5% → VALID** (verifier `20260613-230441-229_cmpatino-verifier.md`); my #44 chat-proxy **12.4% → would-FAIL**.

G1 trips at 5%, G2 at 18.5–24.7% → **G1 always binds first; the bar is a red herring here.** I compute both and lead with G1; happy to collapse to the bar reading if you intended it (then the answer is P(DQ)≈0 — but it contradicts the contract).

### The acceptance→TPS model (so "drift in mean acceptance" → a TPS haircut)

`e_accept_exact` is the **block efficiency** (tokens emitted per spec step, **incl. the bonus**) — verified on the #730 K=6 records: `e_accept_exact 3.6552 == accepted/step 2.6552 + 1 bonus` (accept-rate 44.3%). With step time ~constant in acceptance (always draft K=6, verify in one pass), **TPS ∝ e_accept_exact**, so a relative drop δ in E_accept (the PR's exact `E_accept_private=(1−δ)·E_accept_public`) gives an **equal** TPS haircut: **h = δ (linear)**. Matches #44 empirically (accept-ratio 0.872 ≈ TPS-ratio 0.887, no cushion). Because h≡δ, the #725 documented band U[4%,9%] **is** the δ_stock prior with no conversion, and P(δ_stock>5%)=0.80 transfers directly. *(A cushioned alt — measuring per-token accept-rate α instead — gives a* ***larger*** *budget; linear is both PR-faithful and the tighter/conservative choice.)*

### Step 1 — acceptance-drift budget δ_max

- **G1 (binding):** δ_max = **5.0%**, **central-invariant** (the gate is relative to the submission's own public number, so the transfer factor and the central both cancel).
- **G2 (bar, step-1 literal, non-binding):** δ_max = 1 − 126.378/central:
  - optimistic **167.9** → **24.7%**
  - honest-stock **155** → **18.5%** (band [150,160] → [15.7%, 21.0%])

### Step 2 — prior P(DQ), δ_stock ~ U[4%,9%]

- **G1 (binding): P(DQ) = 0.80.** Note the prior band's **center 6.5% already exceeds the 5% gate** → expected DQ. (This is exactly #725's `p_private_drift_gt_5pct_naive`.)
- **G2 (bar): P(DQ) = 0.00** (= 1 − #735's P(clears)=1.00; the worst corner 148.7 still clears 126.378).

### Step 3 — parametric P(DQ) vs δ_stock (plugs into lawine #734, zero rework)

Curve + plot logged (`p_dq_vs_delta_stock_curve`). The human reads lawine's measured δ_stock off it. **Coin-flips (P=0.5):** G1 repro **5.0%**; G2 bar **24.2%** (optimistic) / **18.5%** (honest-stock). Named anchors on the curve:

| δ_stock anchor | source | vs 5% gate |
|---|---|:--:|
| 4.3% | flagship #52 private re-run (wide-trained) | **PASS** |
| 6.5% | documented prior **center** | DQ |
| 12.4% | kanna #44 chat-proxy upper bound | **FAIL** |

The binding curve is a near-step at 5% (transfer cancels; sharpness only softened by finite-128-prompt realization noise). lawine #734's direct publishable-drafter measurement pins the real δ_stock — read P(DQ) straight off.

### Step 4 — sensitivity / dominant term

**δ_max for the binding gate is CENTRAL-INVARIANT (flat 5%).** So the entire object #735 reconciled — the 167.9-vs-155 central — **does not touch the binding private-DQ risk.** Moving the central 167.9→155 swings the *bar* budget by ~6.3pp but the *binding* budget by 0 and P(DQ) by 0 (stays 0.80). The load-bearing unknown is **the drift δ_stock itself** (swings P(DQ) over [0,1]). Dominant order: **gate choice (G1 vs G2) ≫ drift ≫ central.**

### Step 5 — rank the two residual risks the fire buys info on

- **(a) identity-gate information-roll → SMALL / known-favorable.** `BASELINE.md` L49 + kanna #38: the official HF-Jobs gate = **PPL + completion + modalities** and **never** compares served tokens to a greedy-AR reference → spec stacks are leaderboard-legal (the whole ~489 frontier ships MTP spec). The int4 benign-tie strict-identity worry is an **internal** pre-flight, not the official gate. The fire buys little new info here.
- **(b) private-DQ gate → DOMINANT / live / UNMEASURED.** Budget only 5%; prior-center 6.5% already over it; naive P(DQ)=0.80. The one favorable anchor (flagship 4.3%) is a **wide-trained** stack; the #730 stock-Hub drafter `…qat-q4_0-unquantized-assistant` is **not** wide-trained (`BASELINE.md` L38) → if anything **more** drift-prone. Its private E_accept drift has **never been measured on-branch**.

**Verdict: (b) dominates (a). Fire now? NO on the binding gate.** The speed leg (P(clears bar)=1.00, #735) and identity leg (#38) are both settled-favorable, but neither is the binding constraint. The dominant residual risk sits at naive 0.80 and is unmeasured. **Carry the same 4–9% haircut convention as #725/#735 so it's comparable.**

### Command
```bash
cd target/
python3 research/official_equiv_calibration/private_dq_risk_730.py --wandb \
  --wandb_group kanna-730-private-dq-risk --name kanna/730-private-dq-risk
```

- **W&B run:** `shmwht1r` — group `kanna-730-private-dq-risk`, state finished, `analysis_only=1`/`official_tps=0`/`no_hf_job=1`/`fires=0`. **No HF Job, no submission, no fire.** Locked `int4_g128_lmhead` @ 126.378 (`905tbujn`) untouched.
- **Peak memory:** N/A — analysis-only (NumPy 400k-draw MC on CPU; no server, no GPU).
- **Inputs (all in-scope on this branch):** `optionb_bi1_stock_int4/ksweep/k6/paired_ab.json` (K=6 local 170.21, e_accept 3.658), my #735 `results/officialequiv_reconcile_730.json` (central 167.9), #725 drift band, `BASELINE.md` gate docs.
- **summary.json fields:** N/A (no benchmark run — this is a CPU risk model; `tps`/`ppl`/`completed`/`run_prefix` not applicable to an analysis-only card).

### What happened
The PR framed the residual risk correctly (the private gate, not the bar margin) but step-1 operationalized DQ as the **bar** — which, per the contract, is the wrong/non-binding gate. The binding gate is the **5% TPS-reproduction gap**, with a flat **5% budget**, **P(DQ)=0.80** at the documented prior, and — critically — **central-invariant**, so #735's hard-won central reconciliation is orthogonal to it. The decision reduces to one unmeasured number: the stock drafter's true private drift. At the prior it's expected to **fail** (center 6.5% > 5%).

### Suggested follow-ups
1. **Hold the fire for lawine #734.** Their direct publishable-drafter private-drift measurement pins δ_stock; read P(DQ) off the Step-3 curve. Fire if δ_stock < ~3.5%.
2. **Measure the stock-Hub drafter's private E_accept drift directly** with my #44 machinery (`scripts/validity/private_gap_probe.py`) on a chat-heavy proxy — no HF Job, retires the single load-bearing unknown locally.
3. **If the human wants to fire anyway,** the cheapest de-risk is the *rescued* stock drafter or a wide-distribution-trained drafter (BASELINE.md L38: wide-trained ⇒ private-stable, like the flagship's 4.3%).
4. **Advisor ruling check:** if you intended the **bar** (not the 5% repro gap) as the operative DQ, P(DQ)≈0 and the fire is clear — but flagging because it contradicts `BASELINE.md`'s documented rule and the live `verification` gate.
