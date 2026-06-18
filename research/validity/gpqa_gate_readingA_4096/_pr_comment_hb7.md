STUDENT kanna: heartbeat — n=8 landed, n=9 in flight; verdict firming, run healthy with keepalive active. NOT terminal.

Two clean seeds since the 14:33Z update.

## Sampled panel n=8 → FAILS (firming)
**Seed 89012 (n=8)** landed clean (`n_error=0`, `n_empty=0`, finish_len 4.0%, acc 0.4495).
- seeds [0.4899, 0.4747, 0.4495, 0.4747, 0.4747, 0.4293, 0.4545, **0.4495**] → mean **0.4621**, 95% t-CI **[0.4457, 0.4786]**
- CI-upper **0.4786 < bar 0.4864** → `READING_A_GPQA_FAILS`; firming vs n=7 (CI-upper was 0.4829).
- **85.5% of base** (0.5404, ubel #628 `ilg4z6e9`); 92.6% of AR-body (0.4990, ubel #638).

## Hard proof-of-life @ 15:18Z — seed 90123 (n=9) running clean
- **Exactly one** eval client (PID 503034), and verified at the **server**: `vllm:num_requests_running=1.0` / `num_requests_waiting=0.0` → conc=1 contract holds. GPU 100% / 20.1 GB (peak this PR 19.62 — no OOM). HTTP retries 0.
- The 60s W&B keepalive thread (flagged 14:10Z) is **active from seed 89012 onward**, and 89012 closed the run cleanly (`status=completed`, no false `crashed`). So `cr3c4y3q` should stay `running` through the rest of the panel — the earlier flapping is fixed.

## Greedy (capped n=3, health — unchanged)
mean **0.5034**, finish_length@4096 **2.5%** → implied-3072 **11.8%** (≈ #631's 13%) → **cap-artifact confirmed, gate point genuinely healthy** (`cap_released_healthy=true`).

## Plan
90123 (→n=9) then 13579 (→**n=10**, CI-comparable to ubel #638 0.4990 / lawine #639 0.5056), chained one clean resumable cell per launch (each <90 min). **ETA n=10 ~16:50Z**, then the terminal `SENPAI-RESULT` with sampled mean+CI, pct-of-base (0.5404), pct-of-AR-body (0.4990), finish_length@4096 + implied-3072, n_seeds, and the verdict. Guardrails held: `analysis_only=true`, `official_tps=0`, LOCAL A10G, **no HF Job / no submission**. Holding `status:wip`; no blocking question.
