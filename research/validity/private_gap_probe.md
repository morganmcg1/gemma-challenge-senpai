# Local private-stability probe — predict the public→private TPS gap before an HF Job (PR #44)

**TL;DR.** The honest VALID frontier stack `fa2sw_precache_kenyan` loses **12.4% TPS**
going from the public reasoning prompts to a chat-heavy private proxy
(423.6 → 371.0 tok/s, single-stream). That is **far above the 5% reproduction
rule → it would be INVALIDATED on the private re-run.** The drop is almost
entirely a **drafter-acceptance collapse on chat** (E_accept 4.06 → 3.57,
accept-rate 43.7% → 36.7%); the public-overfit precache contributes only ~1.2%.
PPL is 2.377 on both sets (= the published value; gate passes). A reusable probe
(`scripts/validity/private_gap_probe.py`) reproduces this locally with **zero HF
Job quota**, so any student can screen a candidate stack before submitting.

This is the locally-feasible half of the #1 BASELINE risk ("honest stacks lose
4–9% public→private; >5% = INVALID"). It complements #38: the official gate is
PPL + completion + modalities (not token-identity), so spec stacks are
leaderboard-legal and **the private TPS re-run is the real gate**
(`research/validity/served_gate_reconciliation.md`).

---

## 1. Question

Above ~286 TPS the binding constraint is the **private-set re-run**, not the
public leaderboard. Two mechanisms move TPS when the prompt distribution shifts
public→private on a speculative-decode + precache stack:

1. **Drafter acceptance.** MTP (K=7) speed scales with E_accept = mean tokens
   emitted per target forward pass (1..1+K). The drafter is tuned on the
   public/reasoning distribution; chat is its weakest distribution, so E_accept
   drops and TPS with it.
2. **Public-overfit precache.** `fa2sw_precache_kenyan` replays the *public*
   bench prompts into the prefix KV cache during warmup. On a novel (private)
   set those prefill hits vanish.

Can we measure the resulting gap **locally**, before spending an HF Job and
getting invalidated? The absolute A10G TPS is exploratory; the **relative
public→private ratio** is the signal, and it is exactly what the verifier gates
on.

## 2. Method

### 2.1 Private-proxy prompt set (`scripts/validity/build_private_proxy.py`)

- Source: `anon8231489123/ShareGPT_Vicuna_unfiltered`
  (`ShareGPT_V3_unfiltered_cleaned_split.json`), first-human chat turns.
- Reasoning-template markers excluded (MMLU/GPQA-style "answer the following
  multiple choice…", "$letter", "the last line of your response should be…") so
  the proxy is genuinely conversational, not reasoning in disguise.
- **Hard-deduped** vs the public 128 (`data/eval_prompts_sharegpt.json`):
  text-hash + 80-char prefix guard; 0 overlap, 128 unique.
- **Length-matched** to the public chat-templated token-length distribution by
  greedy nearest-neighbour (longest-first). Residual: max 1 token, mean 0.04,
  p90 = 0 → the public and proxy length distributions are **identical**
  (min/p10/p25/p50/p75/p90/max/mean all equal; mean 272.2 tok). **So the measured
  gap reflects distribution shift, not length.**
- Output: `data/private_proxy_sharegpt.json` (+ `.meta.json` provenance). Seed 44;
  public always loaded with the official seed 1.

### 2.2 Three scenarios, each on a fresh server (one named changed variable)

The probe drives the **official** benchmark path — `sglang.bench_serving
--backend vllm-chat`, sharegpt dataset, `output_len 512`, `num_prompts 128`,
`max_concurrency 1`, `request_rate inf`, `warmup_requests 4`, `seed 1`,
`--extra-request-body '{"ignore_eos": true}'` — so the local ratio lines up with
the official public/private re-run. Precache is overridden per scenario.

| scenario | precache | bench set | meaning |
|---|---|---|---|
| `leaderboard`   | public | public  | the number you'd submit |
| `private_rerun` | off    | private | the number the verifier sees |
| `public_cold`   | off    | public  | isolates the precache benefit |

- `headline gap        = (leaderboard − private_rerun) / leaderboard`
- `precache benefit    = (leaderboard − public_cold)   / leaderboard`
- `distribution gap    = (public_cold − private_rerun) / public_cold`  (both
  precache-cold; the pure drafter/length effect)

E_accept is measured per-bench from vLLM's spec-decode counters
(`vllm:spec_decode_num_drafts / num_accepted_tokens / num_draft_tokens`, exposed
with `DISABLE_LOG_STATS=0` — host-side only, does not change GPU compute), and
**cross-checked** independently against vLLM's own `[spec]` log lines. PPL uses
the fixed `ppl_ground_truth_tokens.jsonl` (content-determined → identical across
prompt sets; a per-server health gate, not a per-set quantity).

## 3. Result (fa2sw_precache_kenyan, A10G, n=128, run `20260613T194357Z`)

| scenario | precache | bench | **TPS** | E_accept | accept-rate | dur (s) | completed |
|---|---|---|---|---|---|---|---|
| leaderboard   | public | public  | **423.63** | 4.061 | 0.437 | 154.7 | 128/128 |
| public_cold   | off    | public  | 418.37 | 4.089 | 0.441 | 156.6 | 128/128 |
| private_rerun | off    | private | **370.96** | 3.565 | 0.366 | 176.7 | 128/128 |

- **Headline public→private gap = 12.43%.**  Verdict: **WOULD-FAIL (>5% → INVALID).**
- **PPL = 2.377 on both** (= the published `fa2sw_precache` value; ≤ 2.42 gate passes).
- Decomposition: **precache benefit = 1.24%** (public), **distribution gap = 11.33%**.
- Peak GPU ≈ 19.4 GB / 23 GB; KV cache 9.46 GiB; server ready ~90–100 s.
- W&B: `jgxdnmwz` (group tag `private-gap-probe`), artifact `private_gap_report`.

### 3.1 Mechanism: the gap is the drafter, not the precache

E_accept drops **4.06 → 3.57** public→private (Δ 0.495; accept-rate 43.7% → 36.7%).
Speculative TPS scales ~linearly with E_accept (tokens per target forward pass),
and the numbers close the loop almost exactly:

```
E_accept ratio   private/public(cold) = 3.565 / 4.089 = 0.872
TPS ratio        private/public(cold) = 370.96 / 418.37 = 0.887
```

→ the 11.3% precache-neutral distribution gap is **fully accounted for by the
acceptance collapse on chat**. The two independent acceptance measurements agree
(counter E_accept 4.061/4.089/3.565 vs log-scraped `e_accept_exact`
4.056/4.089/3.565). The public-overfit precache is a minor 1.2% effect —
consistent with kenyan-duma's own "precache ~1% to private" claim.

This is exactly mechanism #1 the hypothesis predicted: chat is the drafter's
weakest distribution.

## 4. Cross-check vs public evidence

- Honest spec stacks lose **4–9%** public→private (BASELINE.md "Key risk").
- The verifier just invalidated **firfir-cast at 7.2%**
  (`20260613-185613-207_cmpatino-verifier.md`).
- kenyan-duma claimed its **precache** contributes only **~1%** to the private
  gap (`research/validity/fa2sw_precache_notes.md` §4 deferred that audit — this
  probe closes it: 1.24%).

Our probe puts honest `fa2sw_precache_kenyan` at **12.4%** — *above* the 4–9%
band and above the 7.2% firfir-cast cap. The **direction is robust** (this stack
is clearly a >5% private-re-run risk); the **magnitude is an upper-ish estimate**
(see §5).

## 5. Honesty caveats (carried from #38)

- **Proxy fidelity (the dominant caveat).** The proxy is pure ShareGPT chat —
  plausibly *harder* (further from reasoning) than the real private set, which is
  only "believed wide/chat-heavy" (likely a *mix*). So 12.4% probably
  **over-states** the true private gap. The valuable property for a screening
  gate is the **failure direction**: the proxy is calibrated such that a real
  invalidated stack (firfir-cast 7.2%) and this honest stack both read >5% — i.e.
  it does **not** have the dangerous false-negative ("proxy says <5% but real
  dies >5%"). A slightly pessimistic early-warning is the safe direction. If a
  future stack reads just over 5% here, treat it as "needs an HF Job to confirm",
  not "definitely invalid".
- **Served nondeterminism.** Served greedy decode on this A10G is run-to-run
  non-deterministic (#38: FA_SLIDING reduction noise; sub-1% TPS deltas are
  noise). The 12.4% headline is **12× the noise floor** and is corroborated by
  the deterministic-aggregate E_accept drop, so it is not a fluke — but a single
  replicate of the headline pair would tighten the bound cheaply.
- **PPL is a gate, not a gap.** It is content-determined and identical across
  sets (2.377); it confirms server health, it does not move with distribution.

## 6. Reusable probe (deliverable)

```bash
# full run (leaderboard + private_rerun + public_cold decomposition), n=128
python3 scripts/validity/private_gap_probe.py \
    --submission submissions/<candidate> \
    [--private data/private_proxy_sharegpt.json] \
    [--wandb_group private-gap-probe]

# fast plumbing check first (8+8, leaderboard + private_rerun only)
python3 scripts/validity/private_gap_probe.py --smoke
```

Writes `research/validity/private_gap_probe/<ts>/report.json` (+ per-scenario
bench/ppl/server logs). `--wandb_group` optionally logs the headline scalars and
report artifact to W&B (lazy + defensive — never breaks the probe; no GPU cost).
**LOCAL ONLY — launches no HF Job and makes no submission.** Rebuild the proxy
with `scripts/validity/build_private_proxy.py` if the public set changes.

## 7. Artifacts

- Probe: `scripts/validity/private_gap_probe.py`
- Proxy builder: `scripts/validity/build_private_proxy.py`
- Proxy set: `data/private_proxy_sharegpt.json` (+ `.meta.json`)
- Run: `research/validity/private_gap_probe/20260613T194357Z/` (report.json + logs)
- W&B: run `jgxdnmwz`, project `gemma-challenge-senpai`
- Related: `research/validity/fa2sw_precache_notes.md`,
  `research/validity/served_gate_reconciliation.md`
