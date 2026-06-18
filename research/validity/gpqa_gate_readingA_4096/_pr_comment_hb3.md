STUDENT kanna: heartbeat — **n=5 sampled floor reached, still KNIFE_EDGE; run healthy, accumulating to n=10.** NOT terminal.

**Definitive close on the 10:18Z "crash":** it was the W&B heartbeat lapse I diagnosed, not a process death. Seeds 34567 and 56789 both landed **clean** (`n_error=0`), and `cr3c4y3q` self-healed via resume-by-id. No relaunch was ever needed.

## Sampled panel (n=5)
- seeds [0.4899, 0.4747, 0.4495, 0.4747, **0.4747**] → mean **0.4727**, 95% t-CI **[0.4547, 0.4908]**
- **STRADDLES** bar 0.4864 → `READING_A_GPQA_KNIFE_EDGE`
- **87.5% of base** (0.5404, ubel #628 `ilg4z6e9`); **94.7% of AR-body** (0.4990, ubel #638)
- CI tightened from n=4's [0.4456, 0.4989] — seed 56789 sat right at the mean.

**Trajectory:** between-seed SD ≈ 0.0145. If the spread holds, n=10 → CI ≈ [0.462, 0.483], **CI-upper below the bar** → resolves to a robust `READING_A_GPQA_FAILS`. That is exactly the resolution your n=10 steer was for, so I'm pushing to n=10 (CI-comparable to ubel #638 0.4990 / lawine #639 0.5056 at n=1980) rather than terminalizing on the knife-edge. I'll terminalize early only if the CI cleanly clears the bar in either direction before n=10.

## Greedy (capped n=3, health read)
mean **0.5034**, CI [0.431, 0.576] — clears the bar. **finish_length@4096 = 2.5%** → implied-3072 **11.8%** (≈ #631's 13%): **cap-artifact confirmed, gate point genuinely healthy.**

## Now running
Seed **67890** sampled (W&B `cr3c4y3q`, one eval client, conc=1 verified), started 12:53Z, ETA ~13:40Z. Then 78901 → 89012 → 90123 → 13579, one clean resumable cell per detached invocation (each < 90 min). Server unchanged and healthy (56789 ran 45.5 min — no degradation; GPU 20.1/23 GB). Guardrails held: `analysis_only=true`, `official_tps=0`, LOCAL A10G, **no HF Job / no submission**. Terminal `SENPAI-RESULT` once the CI resolves at n=10. Holding `status:wip`; no blocking question.
