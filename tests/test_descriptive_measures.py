"""descriptive_measures.py on synthetic score files: formulas by hand, division by zero, missing fields, refusals, determinism."""
import csv
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dm", os.path.join(ROOT, "scripts", "descriptive_measures.py")); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


def cal(preds, meas, n=10):
    return [{"bin": i, "n": n, "mean_pred": p, "mean_meas": m, "lo": m * 0.9, "hi": m * 1.1} for i, (p, m) in enumerate(zip(preds, meas))]


def score(run="gammabw1000-nf2-s0-final", st="select", hit=0.6, far=0.1, off_hit=0.05, off_far=0.05, err=0.4, err_even=0.8, err_ex=0.5, err_off=2.0, auc=0.86, auc_off=0.82, top=0.4, top_off=0.2, top_u=0.15, stamped=True):
    rule = lambda h, f: {"flag_count_ratio": 1.0, "block_hit_rate": h, "block_false_alarm_ratio": f, "flagged_cells_pred": 10, "flagged_cells_meas": 10}
    meth = lambda e: {"abs_log_err_pair_mean": e}
    am = lambda: {"n_items": 100, "methods": {"model": meth(err), "model_off": meth(err_off), "uniform": meth(err_even), "excess_only": meth(err_ex), "copy_planck": meth(0.7), "copy_raw": meth(1.0), "oracle_uniform": meth(err_ex)},
                  "decision": {"vs": {"uniform": {"diff": err_even - err, "lo": 0.2, "hi": 0.6, "share_pairs_better": 0.9}, "excess_only": {"diff": err_ex - err, "lo": 0.0, "hi": 0.2, "share_pairs_better": 0.7}}},
                  "calibration": {"model": cal([1, 2, 4], [1.1, 2, 3.5]), "uniform": cal([1, 2, 4], [0.5, 2, 9]), "copy_planck": cal([1, 2, 4], [1, 2, 4])}}
    j = {"run": run, "set": st, "n_tiles": 2645, "ckpt": "x",
         "summary": {"auc": {m: {"mean": a, "n_pairs": 5} for m, a in (("model", auc), ("model_off", auc_off), ("uniform", 0.85), ("excess_only", 0.88), ("oracle_uniform", 0.88), ("copy_planck", 0.84), ("copy_raw", 0.74))},
                     "rule": {"model": rule(hit, far), "model_off": rule(off_hit, off_far), "uniform": rule(0.4, 0.05), "oracle_uniform": rule(0.44, 0.05), "copy_planck": rule(0.3, 0.03), "copy_raw": rule(0.43, 0.05)},
                     "conc": {"model": {"top1_share": top, "n_pixels": 9}, "model_off": {"top1_share": top_off, "n_pixels": 9}, "uniform": {"top1_share": top_u, "n_pixels": 9}, "measured": {"top1_share": 0.46, "n_pixels": 9}}},
         "amount": {"block3": am(), "pixel": am()}}
    if stamped:
        j["guard"] = {"report_hits": 0}; j["code_sha256"] = {"a": "b"}
    return j


def base():
    j = score("baselines")
    for k in ("model", "model_off"):
        j["summary"]["auc"].pop(k); j["summary"]["rule"].pop(k); j["summary"]["conc"].pop(k, None)
        for lv in ("block3", "pixel"):
            j["amount"][lv]["methods"].pop(k); j["amount"][lv]["calibration"].pop(k, None)
    for lv in ("block3", "pixel"):
        j["amount"][lv].pop("decision")
    return j


def rowsof(j, tmp_path, extra=()):
    d = tmp_path / "select"; (d / j["run"]).mkdir(parents=True)
    (d / j["run"] / f"score_{j['set']}.json").write_text(json.dumps(j))
    for e in extra:
        (d / e["run"]).mkdir(parents=True); (d / e["run"] / f"score_{e['set']}.json").write_text(json.dumps(e))
    out = tmp_path / "out"; assert D.main(["--scores", str(d), "--out", str(out)]) == 0
    return list(csv.DictReader(open(out / "descriptive_measures.csv"))), out


def get(rows, measure, level="3km_block", subject="model", run=None):
    m = [r for r in rows if r["measure"] == measure and r["level"] == level and r["subject"] == subject and (run is None or r["run"] == run)]
    assert len(m) == 1, (measure, level, subject, len(m)); return m[0]


def test_csi_by_hand():
    # tp=60 fn=40 (hit .6), fp=6.667 -> far .1 : csi = tp/(tp+fp+fn)
    v, _ = D.csi(0.6, 0.1); assert v == pytest.approx(0.6 * 0.9 / (0.6 + 0.9 - 0.54)); assert v == pytest.approx(1 / (1 / 0.6 + 1 / 0.9 - 1))
    tp, fn, fp = 50.0, 30.0, 20.0; assert D.csi(tp / (tp + fn), fp / (tp + fp))[0] == pytest.approx(tp / (tp + fp + fn))


def test_csi_zero_and_missing():
    assert D.csi(0.0, None)[0] == 0.0 and D.csi(0.0, 0.5)[0] == 0.0        # nothing found
    assert D.csi(0.5, 1.0)[0] == 0.0                                          # everything invented
    assert D.csi(None, None)[0] is None and "undefined" in D.csi(None, None)[1]  # no real fire blocks
    assert D.csi(0.5, None)[0] is None
    assert D.csi(1.0, 0.0)[0] == 1.0
    assert D.csi(float("nan"), 0.1)[0] is None


def test_skill_and_fire_share_edges():
    assert D.skill(0.4, 0.8) == pytest.approx(0.5) and D.skill(1.6, 0.8) == pytest.approx(-1.0)
    assert D.skill(0.4, 0.0) is None and D.skill(None, 1.0) is None and D.skill(0.4, None) is None
    assert D.fire_share(0.9, 0.7, 0.5)[0] == pytest.approx(0.5)
    v, n = D.fire_share(0.5, 0.4, 0.5); assert v is None and "denominator" in n      # no skill above the reference
    assert D.fire_share(0.4, 0.4, 0.5)[0] is None
    assert D.fire_share(None, 0.4, 0.5)[0] is None
    assert D.fire_share(0.9, 0.9, 0.5)[0] == 0.0


def test_rows_values(tmp_path):
    rows, _ = rowsof(score(), tmp_path)
    r = get(rows, "csi"); assert float(r["value"]) == pytest.approx(1 / (1 / 0.6 + 1 / 0.9 - 1))
    assert float(r["ref_excess_only"]) == pytest.approx(1 / (1 / 0.44 + 1 / 0.95 - 1)) and r["checkpoint"] == "final" and r["seed"] == "0" and r["family"] == "gammabw1000-nf2" and r["stamped"] == "True"
    assert float(get(rows, "amount_skill_vs_even")["value"]) == pytest.approx(0.5) and float(get(rows, "amount_skill_vs_excess")["value"]) == pytest.approx(0.2)
    s = get(rows, "amount_skill_vs_even"); assert float(s["lo"]) == pytest.approx(0.25) and float(s["hi"]) == pytest.approx(0.75) and float(s["share_pairs_better"]) == 0.9
    assert float(s["ref_excess_only"]) == pytest.approx(1 - 0.5 / 0.8)
    assert float(get(rows, "fire_share_auc_block")["value"]) == pytest.approx((0.86 - 0.82) / (0.86 - 0.5))
    assert float(get(rows, "fire_share_hottest_cell_share", "modis_pixel")["value"]) == pytest.approx((0.4 - 0.2) / (0.4 - 0.15))
    assert float(get(rows, "fire_share_csi")["value"]) == pytest.approx(1 - D.csi(0.05, 0.05)[0] / D.csi(0.6, 0.1)[0])
    fa = float(get(rows, "fire_share_amount_skill_vs_even")["value"]); assert fa == pytest.approx((0.5 - (1 - 2.0 / 0.8)) / 0.5) and fa > 1
    assert "worse than" in get(rows, "fire_share_amount_skill_vs_even")["note"]
    assert float(get(rows, "calibration_abs_log_ratio")["value"]) > 0


def test_checkpoint_seed_parsing():
    assert D.parse_run("gammabw100-d05-nf2-s1") == ("gammabw100-d05-nf2", 1, "best_validation")
    assert D.parse_run("oldloss-nf2-s0") == ("oldloss-nf2", 0, "best_validation")
    assert D.parse_run("ref-legacy-s1")[:2] == ("ref-legacy", 1) and D.parse_run("baselines")[2] == "baseline" and D.parse_run("odd name")[2] == "unknown"


def test_model_without_skill_gives_undefined_fire_share(tmp_path):
    rows, _ = rowsof(score(auc=0.5, auc_off=0.45, err=0.8, err_even=0.8, top=0.1), tmp_path)
    for m in ("fire_share_auc_block", "fire_share_amount_skill_vs_even"):
        r = get(rows, m); assert r["value"] == "" and "denominator" in r["note"]
    assert get(rows, "fire_share_hottest_cell_share", "modis_pixel")["value"] == ""


def test_zero_reference_error_and_zero_flags(tmp_path):
    j = score(err_even=0.0, hit=0.0, far=0.0)
    j["summary"]["rule"]["model"]["block_false_alarm_ratio"] = None            # no flags predicted at all
    rows, _ = rowsof(j, tmp_path)
    assert get(rows, "amount_skill_vs_even")["value"] == "" and "undefined" in get(rows, "amount_skill_vs_even")["note"]
    assert float(get(rows, "csi")["value"]) == 0.0
    assert get(rows, "false_alarm_share")["value"] == "" and "missing or null" in get(rows, "false_alarm_share")["note"]


def test_missing_fields_do_not_crash_and_say_so(tmp_path):
    j = score(); del j["summary"]["rule"]; del j["summary"]["conc"]; j["amount"]["block3"].pop("calibration"); j["amount"]["pixel"].pop("decision")
    rows, _ = rowsof(j, tmp_path)
    assert get(rows, "csi")["value"] == "" and "not in summary.rule" in get(rows, "csi")["note"]
    assert get(rows, "hottest_cell_share", "modis_pixel")["value"] == "" and get(rows, "fire_share_hottest_cell_share", "modis_pixel")["value"] == ""
    assert get(rows, "calibration_abs_log_ratio")["value"] == "" and "not stored" in get(rows, "calibration_abs_log_ratio")["note"]
    p = get(rows, "amount_skill_vs_even", "modis_pixel"); assert float(p["value"]) == pytest.approx(0.5) and p["lo"] == ""          # no stored range -> none invented


def test_excess_only_flags_need_proof_of_equality(tmp_path):
    j = score(); j["summary"]["auc"]["oracle_uniform"]["mean"] = 0.5
    rows, _ = rowsof(j, tmp_path)
    assert get(rows, "csi")["ref_excess_only"] == ""                                     # not provably the same method -> omitted, not guessed


def test_baselines_file_and_qa(tmp_path):
    rows, out = rowsof(score(), tmp_path, extra=[base()])
    assert float(get(rows, "csi", subject="uniform", run="baselines")["value"]) == pytest.approx(1 / (1 / 0.4 + 1 / 0.95 - 1))
    assert float(get(rows, "amount_skill_vs_even", subject="excess_only", run="baselines")["value"]) == pytest.approx(1 - 0.5 / 0.8)
    q = json.load(open(out / "descriptive_measures.json"))["qa_max_abs_diff_simple_methods_vs_baselines_file"]; assert q["select|csi"] == pytest.approx(0.0)


def test_refuses_xdata_and_non_score(tmp_path):
    d = tmp_path / "xdata" / "r"; d.mkdir(parents=True); (d / "score_val2all.json").write_text(json.dumps(score(st="val2all")))
    with pytest.raises(SystemExit, match="INVALID"):
        D.main(["--scores", str(tmp_path / "xdata"), "--out", str(tmp_path / "o")])
    e = tmp_path / "bad" / "r"; e.mkdir(parents=True); (e / "score_select.json").write_text("{}")
    with pytest.raises(SystemExit, match="not a score file"):
        D.main(["--scores", str(tmp_path / "bad"), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="no score files"):
        D.main(["--scores", str(tmp_path / "nothing"), "--out", str(tmp_path / "o")])


def test_deterministic_and_unstamped_flag(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    for t in (a, b):
        t.mkdir(); rowsof(score(stamped=False), t, extra=[base()])
    for f in ("descriptive_measures.csv", "descriptive_measures.json", "calibration_bins.csv"):
        assert open(a / "out" / f, "rb").read() == open(b / "out" / f, "rb").read()
    rows = list(csv.DictReader(open(a / "out" / "descriptive_measures.csv"))); assert get(rows, "csi")["stamped"] == "False"
