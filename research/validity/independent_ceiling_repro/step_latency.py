import statistics

data = {
    "k4": dict(tps=[152.28, 153.95], drafts=40817, draft_tok=163268, acc=90407, K=4),
    "k5": dict(tps=[154.37, 157.26], drafts=37690, draft_tok=188450, acc=93588, K=5),
    "k6": dict(tps=[170.21, 174.12], drafts=35595, draft_tok=213570, acc=95678, K=6),
}
print("arm  K  med_tps  acc_len  steps/rep  step_ms  accept_rate")
for k, d in data.items():
    med = statistics.median(d["tps"])
    acc_len = (d["acc"] + d["drafts"]) / d["drafts"]   # emitted per step = accepted draft + 1 target
    steps_per_rep = d["drafts"] / 2.0
    rep_dur = 65536 / med
    step_ms = 1000 * rep_dur / steps_per_rep
    accept_rate = d["acc"] / d["draft_tok"]
    print("%-4s %2d %8.2f %8.3f %10.0f %8.3f %11.4f" % (k, d["K"], med, acc_len, steps_per_rep, step_ms, accept_rate))
