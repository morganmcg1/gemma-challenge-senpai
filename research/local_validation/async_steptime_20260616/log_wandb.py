import json, os
import wandb

probe = json.load(open("research/local_validation/async_steptime_20260616/twostream_probe.json"))

D = 1.433    # drafter (propose) gpu p50 ms, decode-only, served K=7 config
V = 6.445    # verify (execute_model) gpu p50 ms, decode-only
D_mean, V_mean = 1.4258, 6.2366
ANCHOR = 467.14   # realized equivalence frontier (denken #423), the sync baseline
DEPLOYED = 481.53 # deployed NON-equivalent incumbent (identity 0.9966)

ratio = D / V
sync_cycle = D + V
best_cycle = max(D, V)
prize_pct = (1 - best_cycle / sync_cycle) * 100  # == D/(D+V)

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="gemma-challenge-senpai",
    group="async-pipelined-drafting",
    name="land/async-pipelined-drafting-profile",
    job_type="analysis",
    config={
        "submission": "fa2sw_precache_kenyan",
        "method": "land/async-pipelined-drafting",
        "hardware": "A10G sm_86 (72 SM), LOCAL exploratory",
        "K_speculative": 7,
        "batch_size": 1,
        "decode_shape": "128->128",
        "analysis_only": True,
        "official_tps": 0,
        "self_abort_threshold_DoverV": 0.05,
        "sync_baseline_anchor_tps": ANCHOR,
        "deployed_incumbent_tps": DEPLOYED,
        "ppl_gate": 2.42,
        "served_path_modified": False,
    },
)

wandb.log({
    # --- step 1: measured D and V on the served stack ---
    "drafter_D_ms_p50": D,
    "verify_V_ms_p50": V,
    "drafter_D_ms_mean": D_mean,
    "verify_V_ms_mean": V_mean,
    # --- step 2: self-abort gate ---
    "D_over_V_p50": round(ratio, 4),
    "D_over_V_mean": round(D_mean / V_mean, 4),
    "self_abort_gate_trips": int(ratio < 0.05),
    # --- the prize vs realizable ---
    "sync_cycle_ms_DplusV": round(sync_cycle, 4),
    "best_overlap_cycle_ms_maxDV": round(best_cycle, 4),
    "theoretical_prize_pct": round(prize_pct, 4),
    "realized_async_delta_pct_byte_exact": 0.0,
    "realized_async_tps_byte_exact": ANCHOR,
    "crosses_deployed_481_53": 0,
    # --- step 4 (hardware ceiling probe, contaminated best case) ---
    "probe_overlap_efficiency_0to1": probe["overlap_efficiency_0to1"],
    "probe_contaminated_wall_speedup_pct": probe["concurrent_speedup_vs_serial_pct"],
    "probe_serial_ms": probe["serial_ms"],
    "probe_concurrent_ms": probe["concurrent_two_stream_ms"],
    "graph_disable_dispatch_cost_ms_per_step_lit": 1.25,  # ~1-1.5ms/step, SSD/vLLM lit
})

# rich summary table
tbl = wandb.Table(columns=["quantity", "value", "note"])
for r in [
    ["D drafter p50 (ms)", D, "propose / MTP K=7 onegraph replay"],
    ["V verify p50 (ms)", V, "execute_model int4 body forward"],
    ["D/V ratio", round(ratio, 4), "gate=0.05 -> does NOT trip"],
    ["theoretical prize %", round(prize_pct, 2), "D/(D+V) if drafter fully hidden"],
    ["byte-exact realized %", 0.0, "wait_event(verify_done) re-serializes -> 0 overlap"],
    ["contaminated probe overlap eff", probe["overlap_efficiency_0to1"], "fp16 best case, breaks identity"],
    ["graph-disable cost ms/step", 1.25, "erases the contaminated 8% -> net <=0"],
    ["verdict", 0, "realized-NULL byte-exact; does NOT cross 481.53"],
]:
    tbl.add_data(*r)
wandb.log({"async_pipelining_summary": tbl})

print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_URL", run.url)
run.finish()
