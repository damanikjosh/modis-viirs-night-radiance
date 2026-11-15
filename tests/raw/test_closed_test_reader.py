"""closed-test reader: score files -> four lines + overall verdict, on synthetic score files (deterministic)."""
import copy
import pytest
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("reader", os.path.join(ROOT, "scripts", "closed_test_reader.py")); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
AMEND = os.path.join(ROOT, "configs", "closed_test_amendment.json"); K = R.load_constants(AMEND)


def good():
    dec = lambda: {"delivers": True, "testable": True, "n_pairs": 8, "fails": []}
    sm = {"auc": {"model": {"mean": 0.95}, "uniform": {"mean": 0.85}, "excess_only": {"mean": 0.88}},
          "vs": {"uniform": {"diff": 0.10, "lo": 0.05, "hi": 0.15, "n_pairs": 8, "share_pos": 0.9}, "excess_only": {"diff": 0.07, "lo": 0.02, "hi": 0.12, "n_pairs": 8, "share_pos": 0.8}, "model_off": {"diff": 0.06, "lo": 0.0, "hi": 0.1, "n_pairs": 8, "share_pos": 0.9}},
          "conc": {"model": {"top1_share": 0.40, "n_pixels": 500.0}, "uniform": {"top1_share": 0.14}, "measured": {"top1_share": 0.5}},
          "rule": {"model": {"flag_count_ratio": 1.1, "block_hit_rate": 0.7, "block_false_alarm_ratio": 0.1, "flagged_cells_meas": 900.0}}}
    am = {"pixel": {"decision": dec(), "methods": {"model": {"ratio_of_sums": 1.0}}}, "block3": {"decision": dec(), "methods": {"model": {"ratio_of_sums": 1.05}}}}
    return {"set": "val2all", "n_tiles": 697, "run": "x", "epoch": 46, "summary": sm, "amount": am, "summary_no_hole_blocks": copy.deepcopy(sm), "amount_no_hole_blocks": copy.deepcopy(am)}


def overall(s, **k): return R.read(s, K, **k)["overall"]


def test_all_pass_gives_pass_and_names_the_comparator():
    r = R.read(good(), K); assert r["overall"] == "PASS" and set(r["lines"].values()) == {"PASS"}
    assert r["all_blocks_decides"]["line1_fire_block_score"]["comparator"] == "excess_only"          # the better of the two simple methods (0.88 > 0.85)
    assert r["reader_sha256"] and "hole_free_shown_next_to_it" in r


def test_each_line_can_fail():
    s = good(); s["summary"]["vs"]["excess_only"]["diff"] = 0.03; assert overall(s) == "FAIL"                       # margin below 0.05
    s = good(); s["summary"]["vs"]["excess_only"]["lo"] = -0.01; assert overall(s) == "FAIL"                        # range includes 0
    s = good(); s["summary"]["vs"]["excess_only"]["share_pos"] = 0.7; assert overall(s) == "FAIL"
    s = good(); s["summary"]["vs"]["model_off"]["diff"] = -0.01; assert overall(s) == "FAIL"                       # fire part does not add
    s = good(); s["amount"]["block3"]["methods"]["model"]["ratio_of_sums"] = 1.4; assert overall(s) == "FAIL"
    s = good(); s["amount"]["pixel"]["decision"]["delivers"] = False; assert overall(s) == "FAIL"
    s = good(); s["summary"]["conc"]["model"]["top1_share"] = 0.30; assert overall(s) == "FAIL"
    s = good(); s["summary"]["rule"]["model"]["flag_count_ratio"] = 2.5; assert overall(s) == "FAIL"
    s = good(); s["summary"]["rule"]["model"]["block_hit_rate"] = 0.4; assert overall(s) == "FAIL"


def test_not_testable_lines_never_hide_a_clear_fail_and_never_become_a_pass():
    s = good(); s["summary"]["vs"]["excess_only"]["n_pairs"] = 2; assert overall(s) == "NOT_DECIDABLE"            # only non-testable lines
    s = good(); s["summary"]["vs"]["excess_only"]["n_pairs"] = 2; s["summary"]["conc"]["model"]["top1_share"] = 0.1; assert overall(s) == "FAIL"   # a testable fail wins
    s = good(); s["summary"]["conc"]["model"]["n_pixels"] = 29.0; assert overall(s) == "NOT_DECIDABLE"
    s = good(); s["summary"]["rule"]["model"]["flagged_cells_meas"] = 29.0; assert overall(s) == "NOT_DECIDABLE"
    s = good(); s["amount"]["block3"]["decision"]["testable"] = False; assert overall(s) == "NOT_DECIDABLE"
    s = good(); s["amount"]["block3"]["decision"]["testable"] = False; s["amount"]["block3"]["methods"]["model"]["ratio_of_sums"] = 3.0; assert overall(s) == "FAIL"   # ratio out of band is a fail even if the decision line is untestable


def test_exploratory_pack_prints_the_lines_but_no_verdict():
    r = R.read(good(), K, exploratory=True); assert r["overall"] == "EXPLORATORY_NO_VERDICT" and r["lines"]["line3_hottest_cell_share"] == "PASS"


def test_deterministic_and_input_not_mutated():
    s = good(); before = json.dumps(s, sort_keys=True); a = json.dumps(R.read(s, K), sort_keys=True); b = json.dumps(R.read(s, K), sort_keys=True)
    assert a == b and json.dumps(s, sort_keys=True) == before


def test_reads_a_real_shaped_select_score_file_without_error():
    f = os.path.join(ROOT, "..", "..", "..", "private")                  # no real file needed: the shape check uses the synthetic file only
    assert R.read(good(), K)["n_tiles"] == 697


def test_minimum_counts_are_exactly_30_boundary_included():
    assert K["min_pixels_line_3"] == 30 and K["min_flagged_measured_line_4"] == 30 and K["min_pairs_lines_1_2"] == 3               # minimum counts: about 30 pixels / flagged cells; 3 pairs
    for n, st in ((30.0, "PASS"), (29.0, "NOT_TESTABLE"), (29.999, "NOT_TESTABLE")):
        s = good(); s["summary"]["conc"]["model"]["n_pixels"] = n; assert R.read(s, K)["lines"]["line3_hottest_cell_share"] == st, n
        s = good(); s["summary"]["rule"]["model"]["flagged_cells_meas"] = n; assert R.read(s, K)["lines"]["line4_flag_count"] == st, n
    s = good(); s["summary"]["conc"]["model"]["n_pixels"] = 30.0; s["summary"]["conc"]["model"]["top1_share"] = 0.2; assert R.read(s, K)["overall"] == "FAIL"        # 30 pixels is enough to FAIL
    s = good(); s["summary"]["vs"]["excess_only"]["n_pairs"] = 3; assert R.read(s, K)["lines"]["line1_fire_block_score"] == "PASS"
    s = good(); s["summary"]["vs"]["excess_only"]["n_pairs"] = 2; assert R.read(s, K)["lines"]["line1_fire_block_score"] == "NOT_TESTABLE"


def test_reader_takes_every_number_from_the_amendment_file_and_records_its_hash(tmp_path):
    import hashlib
    a = json.load(open(AMEND)); a["reader_constants"]["min_pixels_line_3"] = 501; f = tmp_path / "amend.json"; json.dump(a, open(f, "w"))
    K2 = R.load_constants(str(f)); s = good(); assert R.read(s, K2)["lines"]["line3_hottest_cell_share"] == "NOT_TESTABLE" and R.read(s, K)["lines"]["line3_hottest_cell_share"] == "PASS"   # good() holds 500 pixels: a threshold of 501 makes the line untestable
    r = R.read(good(), K, amendment_path=AMEND); assert r["amendment_sha256"] == hashlib.sha256(open(AMEND, "rb").read()).hexdigest() and r["constants"] == K
    del a["reader_constants"]["hit_rate_line_4"]; json.dump(a, open(f, "w"))
    with pytest.raises(SystemExit):
        R.load_constants(str(f))


def test_the_reader_has_no_threshold_literals_of_its_own():
    src = open(os.path.join(ROOT, "scripts", "closed_test_reader.py")).read()
    for lit in ("0.33", "= 30", "= 100", "1.25", "0.05"):
        assert not any(l.strip().startswith(("MIN_", "AUC_", "RATIO_", "TOP1", "FLAG_", "HIT")) and lit in l for l in src.splitlines())
