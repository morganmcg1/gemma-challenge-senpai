STUDENT kanna: heartbeat — **n=7 landed; verdict crossed KNIFE_EDGE → `READING_A_GPQA_FAILS`.** Continuing to n=10 per your steer. NOT terminal. Plus a shared-volume reclaim (issue #603).

## Hard proof-of-life @ 14:26Z
Run healthy. Seed **89012** sampled in flight (W&B `cr3c4y3q`, resumed), and I verified the conc=1 contract at the **server** this time: `vllm:num_requests_running=1.0`, `num_requests_waiting=0.0` → exactly one in-flight request. Seed 78901 finished **clean** (`n_error=0`, `n_empty=0`) and the invocation closed the W&B run normally (`status=completed`), so the run is in a clean state, not `crashed`. The keepalive thread (flagged 14:10Z) is active from seed 89012 onward.

## Panel state — sampled n=7 → FAILS
- seeds [0.4899, 0.4747, 0.4495, 0.4747, 0.4747, 0.4293, **0.4545**] → mean **0.4639**, 95% t-CI **[0.4450, 0.4829]**
- **CI-upper 0.4829 < bar 0.4864** → CI now sits **entirely below** the bar → `READING_A_GPQA_FAILS` (was KNIFE_EDGE at n=6; seed 78901's 0.4545 pulled CI-upper under).
- **85.8% of base** (0.5404, ubel #628 `ilg4z6e9`), CI [82.3%, 89.4%] — all below the 90% bar.
- **92.97% of AR-body** (0.4990, ubel #638).
- **Greedy n=3 (health, capped):** mean **0.5034**, finish_length@4096 **2.5%** → implied-3072 **11.8%** (≈#631's 13%) → **cap-artifact confirmed, gate point genuinely healthy** (`cap_released_healthy=true`).

The verdict has resolved, but per your "accumulate toward n=10 for CI-comparability to ubel #638 (n=1980) / lawine #639 (n=1980)" steer — and because the margin is narrow (CI-upper is 0.0035 under the bar) — I'm **not terminalizing at n=7**. Running 89012 (→n=8) → 90123 (→n=9) → 13579 (→n=10), one resumable cell per detached invocation (<90 min each), then the terminal `SENPAI-RESULT`.

## Ops note — freed 41G on the shared volume (issue #603)
The shared `/` volume was at **99% (18G free)** with another student's build still growing — a disk-full risk to my multi-hour panel (and the fleet). I pruned **my own** verified-stale HF cache: the three large gemma **source** models (`gemma-4-E4B-it` 15G, `-qat-q4_0-unquantized` 15G, `-qat-w4a16-ct` 11G). Verification before deletion: (a) **not mapped/open by any running process** (`/proc/*/maps` + `lsof` clean), (b) **not referenced in any eval/serve code** (only the 183M `-assistant` drafter is), (c) the Option-B serve path uses the **local** `int4_g128_lmhead` body (self-contained: own tokenizer.json + configs) + the **183M** drafter — both kept and untouched. Result: **18G → 59G free** (99%→94%). Re-downloadable from the Hub if a future rebuild needs them. Running server + eval re-verified healthy post-prune. Flagging since you're coordinating the volume; happy to do nothing further or to re-pull on request.

Guardrails held: `analysis_only=true`, `official_tps=0`, dev307 conc=1, BI=1, LOCAL A10G, **no HF Job / no submission**. Holding `status:wip`; no blocking question.
