"""The 88 listed tiles are retired from REPORT; REPORT = configs/nf2_split.json report MINUS configs/report_retired.json. Tests use ids and config only (no tile data)."""
import json
import os

import pytest

from nightrad.raw import nf2_split as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = os.path.join(ROOT, "configs", "nf2_split.json"); RET = os.path.join(ROOT, "configs", "report_retired.json"); XL = os.path.join(ROOT, "configs", "xdata_list.json")
import importlib.util
spec = importlib.util.spec_from_file_location("wiring", os.path.join(ROOT, "tests", "raw", "test_score_guard_wiring.py")); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def test_counts_match_and_the_retire_list_is_pinned():
    rep = json.load(open(SPLIT))["tiles"]["report"]; r = json.load(open(RET)); ids = r["retired_tile_ids"]
    assert len(rep) == 4033 and len(ids) == 88 and len(set(ids)) == 88 and set(ids) <= set(rep)
    assert len(S.report_effective(SPLIT, RET)) == 3945 == r["counts_verified_from_ids_and_manifest"]["report_after_retirement"]["tiles"]
    c = r["counts_verified_from_ids_and_manifest"]; assert (c["report_before"]["tiles"], c["report_before"]["modis_fire_px"], c["report_before"]["pairs"]) == (4033, 2309, 664)
    assert (c["report_after_retirement"]["modis_fire_px"], c["report_after_retirement"]["pairs"]) == (2071, 657) and c["retired_88"]["modis_fire_px"] == 238
    assert S.file_sha256(RET) == S.COMMITTED_RETIRED_SHA256 and S.file_sha256(SPLIT) == S.COMMITTED_SPLIT_SHA256        # the split pin is untouched
    assert set(ids) == set(json.load(open(XL))["tiles"]) & set(rep)                                                 # exactly the REPORT tiles of the invalid cross-data list


def test_retired_ids_never_appear_in_the_effective_report():
    eff = set(S.report_effective(SPLIT, RET)); assert not (eff & S.retired_ids(RET)) and len(eff) == 3945


def test_guard_refuses_retired_tiles_under_every_set_even_the_final_report_scoring(tmp_path):
    ret = sorted(S.retired_ids(RET))[:2]; eff = S.report_effective(SPLIT, RET)[:3]
    d = W.make_pack(tmp_path / "p", ret + eff, ["ER"] * 5)
    for set_name, final in (("select", False), ("val2all", False), ("report", True)):
        with pytest.raises(SystemExit) as e:
            S.check_scoring_set(d, SPLIT, set_name, final=final)
        assert "RETIRED" in str(e.value), (set_name, final)
    ok = W.make_pack(tmp_path / "q", eff, ["ER"] * 3)
    rec = S.check_scoring_set(ok, SPLIT, "report", final=True); assert rec["tiles_checked"] == 3 and rec["retired_hits"] == 0 and rec["retired_pinned"] and rec["retired_sha256"] == S.COMMITTED_RETIRED_SHA256
    with pytest.raises(SystemExit):
        S.check_scoring_set(ok, SPLIT, "report", final=False)                                                          # REPORT without --final still refused


def test_a_modified_retire_list_is_refused_unless_an_override_reason_is_logged(tmp_path):
    other = tmp_path / "r.json"; json.dump({"retired_tile_ids": []}, open(other, "w"))
    ok = W.make_pack(tmp_path / "q", S.report_effective(SPLIT, RET)[:2], ["ER"] * 2)
    with pytest.raises(SystemExit) as e:
        S.check_scoring_set(ok, SPLIT, "report", final=True, retired_json=str(other))
    assert "retire list" in str(e.value)
    rec = S.check_scoring_set(ok, SPLIT, "report", final=True, retired_json=str(other), split_override_reason="unit test"); assert rec["retired_pinned"] is False and rec["split_override_reason"] == "unit test"


def test_the_old_cross_data_list_is_still_refused_and_now_also_for_retirement(tmp_path):
    x = json.load(open(XL))["tiles"]; d = W.make_pack(tmp_path / "p", x, ["ES"] * len(x))
    for set_name, final in (("val2all", False), ("report", True)):
        with pytest.raises(SystemExit) as e:
            S.check_scoring_set(d, SPLIT, set_name, final=final)
        assert "RETIRED" in str(e.value) or "REPORT" in str(e.value)


def test_scorer_wiring_report_set_excludes_retired_tiles_and_stamps_the_retire_hash(tmp_path):
    ret = sorted(S.retired_ids(RET))[:2]; eff = S.report_effective(SPLIT, RET)[:3]
    d = W.make_pack(tmp_path / "p", ret + eff, ["ER"] * 5)
    f = tmp_path / "eps.json"; json.dump({"select": ["ES"], "report": ["ER"]}, open(f, "w"))
    r = W.run(d, "report", str(f), extra=["--final"], tmp=tmp_path)
    assert "'tiles_checked': 3" in r.stdout and "'retired_hits': 0" in r.stdout and S.COMMITTED_RETIRED_SHA256 in r.stdout          # the 2 retired tiles were dropped before the guard
    r2 = W.run(d, "val2all", str(f), tmp=tmp_path)                                                                                    # any other set: the retired tiles are refused, not dropped
    assert r2.returncode != 0 and "RETIRED" in r2.stderr
