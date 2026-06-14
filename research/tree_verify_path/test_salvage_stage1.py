#!/usr/bin/env python3
"""CPU validation of the STAGE-1 salvage-probe JOIN math (no GPU / no server).

Replicates the exact stash (proposer) + observe (verify) logic added to
sitecustomize.py against the REAL PARENT_M16 tree, on deterministic scenarios:

  A. correct salvage at root: rank-1 spine diverges at chain row 0, the rank-2
     branch == verifier argmax at row 0 (the CORRECT/decoupled-A target row) ->
     branch_hit_correct fires, conflated does NOT.
  B. the tli+1 TRAP: rank-2 == argmax at row 1 (the conflated row) but != row 0 ->
     branch_hit_conflated fires, correct does NOT. (Component 3a on the real join.)
  C. full accept (no divergence) -> full_accept, no branch test.
  D. divergence at a width-1 spine pos (row 4) -> div_no_branch (salvage impossible).
  E. divergence at a width-2 pos with a MISS (rank-2 != either row) -> counted in
     div_at_branch denominator, no hit.
"""
import sys
from importlib import util as _u
from pathlib import Path

REPO = Path("/workspace/senpai/target")
TS = REPO / "scripts" / "profiler" / "tree_spec.py"
_spec = _u.spec_from_file_location("_pr71_ts_salv", TS)
ts = _u.module_from_spec(_spec)
sys.modules[_spec.name] = ts
_spec.loader.exec_module(ts)

tree = ts.TreeSpec(ts.PARENT_M16)
print(f"tree M={tree.num_nodes} spine={tree.spine} max_branch={tree.max_branch}")


def make_stash(draft_token):
    """Replicate _run_tree_emit_probe stash: rank-2 token per width>=2 spine depth."""
    branch_rank2 = {}
    for d, snode in enumerate(tree.spine):
        ch = tree.children[snode]
        if len(ch) >= 2:
            branch_rank2[d] = draft_token[ch[1]]
    spine_chain = [draft_token[n] for n in tree.spine[1:]]
    return {"branch_rank2": branch_rank2, "spine_chain": spine_chain, "ready": True}


def observe(state, stash, draft_token_ids, target_argmax):
    """Replicate _salvage_probe_observe accounting (pure)."""
    if not stash.get("ready"):
        state["skipped_no_stash"] += 1
        return
    stash["ready"] = False
    dti, tgt = list(draft_token_ids), list(target_argmax)
    K = len(dti)
    if K == 0 or len(tgt) < K:
        state["skipped_read_err"] += 1
        return
    cmp = min(K, len(stash["spine_chain"]))
    if dti[:cmp] != stash["spine_chain"][:cmp]:
        state["skipped_unaligned"] += 1
        return
    state["steps"] += 1
    first_div = next((p for p in range(K) if dti[p] != tgt[p]), None)
    if first_div is None:
        state["full_accept"] += 1
        return
    state["divergence"] += 1
    br = stash["branch_rank2"]
    if first_div not in br:
        state["div_no_branch"] += 1
        return
    state["div_at_branch"] += 1
    r2 = br[first_div]
    if r2 == tgt[first_div]:
        state["branch_hit_correct"] += 1
        state["per_pos_hit"][first_div] = state["per_pos_hit"].get(first_div, 0) + 1
    if first_div + 1 < K and r2 == tgt[first_div + 1]:
        state["branch_hit_conflated"] += 1
    state["per_pos_div"][first_div] = state["per_pos_div"].get(first_div, 0) + 1


def fresh_state():
    return {
        "steps": 0, "full_accept": 0, "divergence": 0, "div_at_branch": 0,
        "div_no_branch": 0, "branch_hit_correct": 0, "branch_hit_conflated": 0,
        "per_pos_div": {}, "per_pos_hit": {}, "skipped_no_stash": 0,
        "skipped_unaligned": 0, "skipped_read_err": 0,
    }


# A canonical draft_token assignment for the M16 tree. Spine tokens 100+depth;
# branch (rank-2) tokens 200+node. children[spine[d]][1] is the rank-2 branch.
def canonical_draft_token():
    dt = [None] * tree.num_nodes
    dt[0] = 100  # root
    for node in range(1, tree.num_nodes):
        if tree.rank_in_parent[node] == 1:
            dt[node] = 100 + tree.depth[node]   # rank-1 spine continuation token
        else:
            dt[node] = 200 + node               # distinct rank-2 branch token
    return dt


K = 7  # deployed linear chain length (num_draft); verifier checks rows 0..6
ok = True


def expect(name, cond, detail=""):
    global ok
    print(f"  [{'ok' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        ok = False


# spine_chain the verifier sees = rank-1 spine tokens [100+1, 100+2, ...][:K]
dt = canonical_draft_token()
chain = [dt[n] for n in tree.spine[1:]][:K]   # == draft_token_ids the verify gets
stash_tmpl = make_stash(dt)
print(f"branch_rank2(depth->token)={stash_tmpl['branch_rank2']}  chain={chain}")

# --- A. correct salvage at root (first_div=0, rank-2 depth0 == argmax[0]) -------
st = fresh_state()
stash = make_stash(dt)
r2_root = stash["branch_rank2"][0]
tgt = list(chain)
tgt[0] = r2_root            # verifier argmax at row0 == the rank-2 branch token
observe(st, stash, chain, tgt)
expect("A.correct_salvage_root",
       st["div_at_branch"] == 1 and st["branch_hit_correct"] == 1
       and st["branch_hit_conflated"] == 0,
       f"div_at_branch={st['div_at_branch']} correct={st['branch_hit_correct']} "
       f"conflated={st['branch_hit_conflated']}")

# --- B. the tli+1 TRAP (rank-2 == argmax[1], not argmax[0]) ---------------------
st = fresh_state()
stash = make_stash(dt)
r2_root = stash["branch_rank2"][0]
tgt = list(chain)
tgt[0] = 999999           # row0 argmax != rank-1 (divergence) AND != rank-2
tgt[1] = r2_root          # the CONFLATED row (first_div+1) == rank-2 token
observe(st, stash, chain, tgt)
expect("B.conflated_trap_fires_correct_does_not",
       st["div_at_branch"] == 1 and st["branch_hit_correct"] == 0
       and st["branch_hit_conflated"] == 1,
       f"correct={st['branch_hit_correct']} conflated={st['branch_hit_conflated']} "
       "(== Component 3a: wrong target row = the byteshark 3% bug)")

# --- C. full accept (chain == target everywhere) -------------------------------
st = fresh_state()
stash = make_stash(dt)
observe(st, stash, chain, list(chain))
expect("C.full_accept",
       st["full_accept"] == 1 and st["divergence"] == 0 and st["div_at_branch"] == 0,
       f"full_accept={st['full_accept']}")

# --- D. divergence at a width-1 spine pos (row 4: spine[4]=node9 has 1 child) ---
st = fresh_state()
stash = make_stash(dt)
tgt = list(chain)
tgt[4] = 888888           # first divergence at row 4 (no branch there)
observe(st, stash, chain, tgt)
expect("D.div_no_branch_at_width1_pos",
       st["divergence"] == 1 and st["div_at_branch"] == 0 and st["div_no_branch"] == 1,
       f"div_no_branch={st['div_no_branch']} (first_div=4 not in {sorted(stash['branch_rank2'])})")

# --- E. divergence at width-2 pos but rank-2 MISSES (denominator, no hit) -------
st = fresh_state()
stash = make_stash(dt)
tgt = list(chain)
tgt[1] = 777777           # row1 diverges; rank-2 depth1 != 777777 and != row2
observe(st, stash, chain, tgt)
expect("E.div_at_branch_miss_counts_denominator",
       st["div_at_branch"] == 1 and st["branch_hit_correct"] == 0,
       f"div_at_branch={st['div_at_branch']} correct={st['branch_hit_correct']}")

# --- F. alignment guard: stale stash (wrong spine) is skipped, not counted ------
st = fresh_state()
stash = make_stash(dt)
bad_chain = [c + 1 for c in chain]   # not the stashed spine
observe(st, stash, bad_chain, list(bad_chain))
expect("F.unaligned_skipped",
       st["skipped_unaligned"] == 1 and st["steps"] == 0,
       f"skipped_unaligned={st['skipped_unaligned']}")

# --- G. consume handshake: a 2nd observe on a consumed stash is skipped ---------
st = fresh_state()
stash = make_stash(dt)
observe(st, stash, chain, list(chain))   # consumes
observe(st, stash, chain, list(chain))   # ready=False now
expect("G.consume_handshake",
       st["steps"] == 1 and st["skipped_no_stash"] == 1,
       f"steps={st['steps']} skipped_no_stash={st['skipped_no_stash']}")

print("\nVERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
