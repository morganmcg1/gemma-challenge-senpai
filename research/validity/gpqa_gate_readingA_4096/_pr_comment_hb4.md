STUDENT kanna: heartbeat — sampled **n=6 landed, still KNIFE_EDGE but firming toward FAILS**; accumulating to n=10. NOT terminal.

## n=6 sampled
Seed 67890 came in clean (`n_error=0`, `n_empty=0`, finish_len 2.0%) but **low: 0.4293** — the lowest seed yet.
- seeds [0.4899, 0.4747, 0.4495, 0.4747, 0.4747, **0.4293**] → mean **0.4655**, 95% t-CI **[0.4424, 0.4886]**
- point estimate now sits **clearly below** the bar 0.4864 (was 0.4727 at n=5); CI-hi **0.4886** still grazes it → **STRADDLES → `READING_A_GPQA_KNIFE_EDGE`**
- **86.1% of base** (0.5404, ubel #628 `ilg4z6e9`); **93.3% of AR-body** (0.4990, ubel #638)
- between-seed SD widened 0.0145→**0.022** (seed 67890 is a low outlier), so the CI is tightening slower than the n=5 projection — exactly why your n=10 floor matters; I'm not terminalizing on a grazing CI.

## Greedy (capped n=3, health read — unchanged)
mean **0.5034**, finish_length@4096 **2.5%** → implied-3072 **11.8%** (≈#631's 13%) → **cap-artifact confirmed, gate point genuinely healthy.**

## Now running
Seed **78901** sampled (W&B `cr3c4y3q`, **one** eval client verified, conc=1), started ~13:40Z, ETA ~14:26Z. Then 89012 → 90123 → 13579 to reach the canonical n=10 (CI-comparable to ubel #638 0.4990 / lawine #639 0.5056). Server healthy and continuous (GPU 20.1/23 GB, no creep, no OOM); `cr3c4y3q` may again flicker to "crashed" at a subprocess fork — that's the per-session W&B heartbeat lapse, not a process death (on-disk JSONs + `gate_summary.json` are source of truth). Guardrails held: `analysis_only=true`, `official_tps=0`, local A10G, **no HF Job / no submission**. Terminal `SENPAI-RESULT` at n=10 (or earlier clean CI resolution). Holding `status:wip`; no blocking question.
