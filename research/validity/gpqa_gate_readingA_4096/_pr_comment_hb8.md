STUDENT kanna: heartbeat — n=9 (seed 90123) running clean; **re-armed the n=10 (seed 13579) launch after its chain-waiter died**; on track for terminal at n=10. NOT terminal.

## n=9 in flight, healthy
Seed **90123** sampled at step **76/198** (38%), `openai-api: 1/1`, HTTP retries 0, acc ~0.50. Server PID 320447 up, GPU 100% / 20.1 GB (peak this PR 19.62 — no OOM), **exactly one** eval client (conc=1 holds). Per-seed walltime ~46 min → ETA ~16:25Z.

## Caught + fixed: the n=10 chain-waiter had died (would have idled the A10G)
The detached waiter meant to launch the final seed **13579** after 90123 exits was **gone** — it died on a session boundary (last log line 15:19:40Z "waiting for driver 502671"; the seed-90123 driver itself survived). Left as-is, the panel would have **stalled at n=9** and the A10G sat idle after ~16:25Z. I re-armed it with a **setsid-detached chain** (`_chain_13579_v2.sh`, own session so it survives a session teardown) that waits for the 90123 driver to exit, then launches 13579 sampled. It's guarded against double-launch (`_launch_seed.sh` refuses if any `run_gate.py`/`run_eval.py` is alive → only one 13579 driver can ever start), and I'm backing it with a scheduled self-wakeup to launch manually if the chain dies again. The 13579 driver self-finalizes (writes the n=10 `gate_summary.json` + logs the artifact + finishes W&B).

## Panel state (unchanged from 15:20Z; n=10 ETA ~17:15Z)
- **sampled n=8:** mean **0.4621**, 95% t-CI **[0.4457, 0.4786]** → CI-upper 0.4786 < bar 0.4864 → `READING_A_GPQA_FAILS`. **85.5% of base** (0.5404, ubel #628 `ilg4z6e9`); 92.6% of AR-body (0.4990, ubel #638).
- **greedy n=3 (capped, health):** mean **0.5034**, finish_length@4096 **2.5%** → implied-3072 **11.8%** (≈ #631's 13%) → **cap-artifact confirmed, gate point genuinely healthy**.

Guardrails held: `analysis_only=true`, `official_tps=0`, dev307 conc=1, BI=1, LOCAL A10G, **no HF Job / no submission**. Holding `status:wip`; no blocking question. Terminal `SENPAI-RESULT` at n=10.
