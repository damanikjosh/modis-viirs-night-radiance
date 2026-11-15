#!/usr/bin/env python
"""Model pick rule (configs/model_pick_rule.json, fixed before any score of seeds 2-3 existed): read the SELECT score files of the eight best-validation checkpoints, print the per-seed line table, the counts of steps 1-2, the winner and the frozen candidate block.
Deterministic: no randomness, no time; the constants below are those of the rule file. Does NOT freeze anything: the block it prints is pasted by hand into configs/closed_test_candidates.json when the candidate is frozen.
  python scripts/apply_pick_rule.py <dir with <run>/score_select.json> [--candidates configs/closed_test_candidates.json] [--allow-unstamped] [--out result.json]"""
import argparse, json, os, sys

SETTINGS = {"100": "gammabw100-nf2-s", "1000": "gammabw1000-nf2-s"}; SEEDS = (0, 1, 2, 3)
E_LO, E_HI, F_LO, F_HI, HIT, TOP1 = 0.8, 1.25, 0.5, 2.0, 0.5, 0.33; N_TILES = 2645


def lines(j):
    s, am = j["summary"], j["amount"]; rule = (s.get("rule") or {}).get("model") or {}
    en = ((am.get("pixel") or {}).get("methods") or {}).get("model", {}).get("ratio_of_sums"); fl = rule.get("flag_count_ratio"); hit = rule.get("block_hit_rate")
    top = ((s.get("conc") or {}).get("model") or {}).get("top1_share"); dl = ((am.get("block3") or {}).get("decision") or {}).get("delivers")
    L = {"2_energy": en is not None and E_LO <= en <= E_HI, "3_flags": fl is not None and hit is not None and F_LO <= fl <= F_HI and hit >= HIT, "4_hottest": top is not None and top >= TOP1, "5_amount3km": bool(dl)}
    return L, {"energy_ratio_px": en, "flag_ratio": fl, "hit_rate": hit, "top1": top, "delivers_3km": dl}


def apply(files, cands=None, require_stamp=True):
    rows = {}
    for w, pre in SETTINGS.items():
        for sd in SEEDS:
            run = f"{pre}{sd}"
            if run not in files:
                raise SystemExit(f"missing score file for {run} (the rule needs all eight best-validation scores)")
            j = files[run]
            if j.get("run") != run or j.get("set") != "select" or j.get("n_tiles") != N_TILES:
                raise SystemExit(f"{run}: score file is not the full SELECT best-validation score of this run (run/set/n_tiles = {j.get('run')}/{j.get('set')}/{j.get('n_tiles')})")
            if require_stamp and not (j.get("guard") and j.get("code_sha256")):
                raise SystemExit(f"{run}: score file has no guard / code_sha256 stamp (re-score it with the stamped scorer, or pass --allow-unstamped and say so)")
            L, v = lines(j); rows[run] = {"setting": w, "seed": sd, "epoch": j.get("epoch"), "lines": L, "values": v, "all_2_5": all(L.values()), "passes": sum(L.values()), "stamped": bool(j.get("guard") and j.get("code_sha256")), "ckpt": j.get("ckpt")}
    tot = {w: {"seeds_meeting_all_2_5": sum(rows[f"{pre}{s}"]["all_2_5"] for s in SEEDS), "line_passes_of_16": sum(rows[f"{pre}{s}"]["passes"] for s in SEEDS)} for w, pre in SETTINGS.items()}
    a, b = tot["100"], tot["1000"]
    if a["seeds_meeting_all_2_5"] != b["seeds_meeting_all_2_5"]:
        win, step = ("100" if a["seeds_meeting_all_2_5"] > b["seeds_meeting_all_2_5"] else "1000"), 1
    elif a["line_passes_of_16"] != b["line_passes_of_16"]:
        win, step = ("100" if a["line_passes_of_16"] > b["line_passes_of_16"] else "1000"), 2
    else:
        win, step = "1000", 3
    cand = f"{SETTINGS[win]}0"; out = {"per_seed": rows, "totals": tot, "decided_at_step": step, "winner_weight": win, "candidate_run": cand, "candidate_epoch": rows[cand]["epoch"]}
    r0 = rows[cand]; out["why_this_candidate"] = {
        "pre_stated_lines_met_on_SELECT": f"seed 0 of weight {win} at its best-validation checkpoint (epoch {r0['epoch']}) meets lines " + (", ".join(k.split('_')[0] for k, v in r0["lines"].items() if v) or "none of 2-5") + " of 2-5 on full SELECT; line 1 (block ranking vs even spread) is not met by any gamma-family run",
        "seeds": f"weight {win}: {tot[win]['seeds_meeting_all_2_5']} of 4 seeds meet all of lines 2-5, {tot[win]['line_passes_of_16']} of 16 line passes; other weight: {tot['1000' if win == '100' else '100']['seeds_meeting_all_2_5']} of 4 seeds, {tot['1000' if win == '100' else '100']['line_passes_of_16']} of 16; decided at step {step} of the pick rule" + (" (tie at steps 1 and 2: weight 1,000 by rule)" if step == 3 else ""),
        "known_weaknesses": "block ranking does not beat the even spread (line 1 fails for every gamma-family run); selection of best-validation checkpoints and the setting was made on SELECT; 'delivers' at 3 km rests on a handful of scene pairs (e.g. 25 of 33 vs the 75 % line); the weight comparison is within seed noise"}
    if cands is not None:
        fr = [c for c in cands.get("frozen_candidates_available", []) if c["name"] == f"{cand}__best_val"]
        const = "train_const_gammabw1000-nf2-s0.json" if win == "1000" else "train_const_gammabw100-nf2-s0.json"
        out["candidate_block_for_closed_test_candidates_json"] = {"name": f"{cand}__best_val", "file": fr[0]["file"] if fr else None, "sha256": fr[0]["sha256"] if fr else None, "const_file": const, "chosen_by": "pick rule applied by scripts/apply_pick_rule.py; the author pastes this block", "chosen_at": "<time>", "why_this_candidate": out["why_this_candidate"]}
    return out


def table(out):
    t = ["| run | epoch | 2 energy (px) | 3 flags / hit | 4 hottest | 5 amount 3 km | all 2-5 | passes | stamped |", "|---|---|---|---|---|---|---|---|---|"]
    for run, r in out["per_seed"].items():
        v = r["values"]; L = r["lines"]; f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"; m = lambda b: "yes" if b else "no"
        t.append(f"| {run} | {r['epoch']} | {f(v['energy_ratio_px'])} {m(L['2_energy'])} | {f(v['flag_ratio'])} / {f(v['hit_rate'])} {m(L['3_flags'])} | {f(v['top1'])} {m(L['4_hottest'])} | {m(L['5_amount3km'])} | {m(r['all_2_5'])} | {r['passes']}/4 | {m(r['stamped'])} |")
    return "\n".join(t)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--candidates", default="configs/closed_test_candidates.json"); p.add_argument("--allow-unstamped", action="store_true"); p.add_argument("--out", default=None); a = p.parse_args()
    files = {}
    for w, pre in SETTINGS.items():
        for sd in SEEDS:
            f = os.path.join(a.dir, f"{pre}{sd}", "score_select.json")
            if os.path.exists(f):
                files[f"{pre}{sd}"] = json.load(open(f))
    out = apply(files, json.load(open(a.candidates)) if os.path.exists(a.candidates) else None, require_stamp=not a.allow_unstamped)
    print(table(out)); print(); print(json.dumps({k: out[k] for k in ("totals", "decided_at_step", "winner_weight", "candidate_run", "candidate_epoch")}, indent=1, sort_keys=True)); print(json.dumps(out.get("candidate_block_for_closed_test_candidates_json"), indent=1, sort_keys=True))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
