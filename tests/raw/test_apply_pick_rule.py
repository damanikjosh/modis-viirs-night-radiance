"""model pick rule: apply_pick_rule.py on synthetic score files, ties at each step, boundaries, refusals, determinism."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("pick", os.path.join(ROOT, "scripts", "apply_pick_rule.py")); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def score(run, energy=1.0, flag=1.0, hit=0.6, top1=0.35, delivers=True, n=2645, stamped=True, epoch=10):
    j = {"run": run, "set": "select", "n_tiles": n, "epoch": epoch, "ckpt": "h",
         "summary": {"rule": {"model": {"flag_count_ratio": flag, "block_hit_rate": hit}}, "conc": {"model": {"top1_share": top1}}},
         "amount": {"pixel": {"methods": {"model": {"ratio_of_sums": energy}}}, "block3": {"decision": {"delivers": delivers}}}}
    if stamped:
        j["guard"] = {"report_hits": 0}; j["code_sha256"] = {"score_run.py": "x"}
    return j


def make(overrides=None):
    """overrides: {run: dict of score() kwargs}; default = every seed meets all four lines."""
    files = {}
    for pre in ("gammabw100-nf2-s", "gammabw1000-nf2-s"):
        for s in range(4):
            run = f"{pre}{s}"; files[run] = score(run, **((overrides or {}).get(run, {})))
    return files


BAD = {"delivers": False}


def test_step1_the_setting_with_more_seeds_meeting_all_lines_wins():
    ov = {"gammabw1000-nf2-s1": BAD, "gammabw1000-nf2-s2": BAD}                      # weight 1000: 2 of 4 seeds; weight 100: 4 of 4
    r = P.apply(make(ov)); assert r["decided_at_step"] == 1 and r["winner_weight"] == "100" and r["totals"]["100"]["seeds_meeting_all_2_5"] == 4 and r["totals"]["1000"]["seeds_meeting_all_2_5"] == 2
    ov = {"gammabw100-nf2-s3": {"energy": 0.5}}; r = P.apply(make(ov)); assert r["decided_at_step"] == 1 and r["winner_weight"] == "1000"


def test_step2_total_line_passes_break_a_tie_in_seeds_meeting_all_lines():
    # both settings: 2 seeds meet all four lines; weight 100's other two seeds fail 1 line each, weight 1000's other two fail 2 lines each
    ov = {"gammabw100-nf2-s2": BAD, "gammabw100-nf2-s3": BAD, "gammabw1000-nf2-s2": {"delivers": False, "top1": 0.1}, "gammabw1000-nf2-s3": {"delivers": False, "top1": 0.1}}
    r = P.apply(make(ov)); assert r["totals"]["100"]["seeds_meeting_all_2_5"] == r["totals"]["1000"]["seeds_meeting_all_2_5"] == 2
    assert r["decided_at_step"] == 2 and r["winner_weight"] == "100" and r["totals"]["100"]["line_passes_of_16"] == 14 and r["totals"]["1000"]["line_passes_of_16"] == 12


def test_step3_weight_1000_wins_a_full_tie():
    r = P.apply(make()); assert r["decided_at_step"] == 3 and r["winner_weight"] == "1000"
    ov = {"gammabw100-nf2-s1": BAD, "gammabw1000-nf2-s1": BAD}                         # still equal in steps 1 and 2
    r = P.apply(make(ov)); assert r["decided_at_step"] == 3 and r["winner_weight"] == "1000"


def test_candidate_is_always_seed_0_of_the_winner_even_if_seed_0_is_the_weak_one():
    ov = {"gammabw100-nf2-s0": BAD, "gammabw1000-nf2-s1": BAD, "gammabw1000-nf2-s2": BAD, "gammabw1000-nf2-s3": BAD}
    r = P.apply(make(ov)); assert r["winner_weight"] == "100" and r["candidate_run"] == "gammabw100-nf2-s0"           # not the best seed of weight 100
    assert "seed 0" in r["why_this_candidate"]["pre_stated_lines_met_on_SELECT"] and r["why_this_candidate"]["known_weaknesses"]


def test_line_boundaries_are_inclusive():
    for kw in ({"energy": 0.8}, {"energy": 1.25}, {"flag": 0.5}, {"flag": 2.0}, {"hit": 0.5}, {"top1": 0.33}):
        L, _ = P.lines(score("x", **kw)); assert all(L.values()), kw
    for kw in ({"energy": 0.799}, {"energy": 1.251}, {"flag": 0.499}, {"flag": 2.001}, {"hit": 0.499}, {"top1": 0.329}, {"delivers": False}):
        L, _ = P.lines(score("x", **kw)); assert not all(L.values()), kw


def test_refusals_missing_file_wrong_size_wrong_run_and_unstamped():
    f = make(); del f["gammabw100-nf2-s2"]
    with pytest.raises(SystemExit): P.apply(f)
    with pytest.raises(SystemExit): P.apply(make({"gammabw100-nf2-s1": {"n": 2000}}))
    f = make(); f["gammabw100-nf2-s1"]["run"] = "gammabw100-nf2-s1-final"
    with pytest.raises(SystemExit): P.apply(f)
    with pytest.raises(SystemExit): P.apply(make({"gammabw1000-nf2-s0": {"stamped": False}}))
    assert P.apply(make({"gammabw1000-nf2-s0": {"stamped": False}}), require_stamp=False)["per_seed"]["gammabw1000-nf2-s0"]["stamped"] is False


def test_deterministic_output_and_candidate_block_from_the_frozen_list():
    cands = {"frozen_candidates_available": [{"name": "gammabw1000-nf2-s0__best_val", "file": "x.pt", "sha256": "ab" * 32}]}
    a = json.dumps(P.apply(make(), cands), sort_keys=True); b = json.dumps(P.apply(make(), cands), sort_keys=True); assert a == b
    blk = P.apply(make(), cands)["candidate_block_for_closed_test_candidates_json"]; assert blk["sha256"] == "ab" * 32 and blk["const_file"] == "train_const_gammabw1000-nf2-s0.json" and blk["chosen_at"] == "<time>"
    assert "| gammabw100-nf2-s0 |" in P.table(P.apply(make()))


def test_rule_file_is_committed_with_the_verbatim_text_and_the_three_steps():
    r = json.load(open(os.path.join(ROOT, "configs", "model_pick_rule.json")))
    assert "MODEL PICK RULE" in r["rule_text_verbatim"] and "weight 1,000" in r["rule_text_verbatim"] and len(r["ordered_steps"]) == 3 and r["candidates"]["seeds"] == [0, 1, 2, 3]
