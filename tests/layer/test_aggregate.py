import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import aggregate as agg
from nightrad.inference.version import vkey

T0 = 1154453000.0   # 2006-08-01 17:23 UTC


def rows(gran, cells, utc, vza=30.0):
    return pd.DataFrame({"lon": [c[0] for c in cells], "lat": [c[1] for c in cells], "utc": utc, "granule": gran, "region": "test", "pred4_rad": 1.0, "excess4_rad": 0.5, "bg4_rad": 0.3,
                         "pred_T4_K": 320.0, "pred_dT_K": 20.0, "modis4_rad": 0.5, "modis_T4_K": 300.0, "vza": vza, "modis_alone_flag": 0, "tile_row": 1, "tile_col": 1})


def setup(tmp, scenes, ids):
    """scenes: {id: (df or None, status, version)} -> (expected, done, failed); writes the parquets under tmp/pixels like the worker does."""
    done, failed = {}, {}
    for i, (df, st, v) in scenes.items():
        rec = {"id": i, "status": st, "pipeline_version": v, "t": f"2006-08-{int(i[5:8]) - 212:02d}T18:00:00.000Z", "tiles": 5, "valid_cells": 100}
        if st == "ok":
            rec["flagged"] = 0 if df is None else len(df)
            if df is not None:
                rel = f"pixels/year={i[1:5]}/month=08/{i}.parquet"
                os.makedirs(tmp / os.path.dirname(rel), exist_ok=True)
                t = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata({"pipeline_version": v, "granule": i})
                pq.write_table(t, tmp / rel)
                rec["parquet"] = {"key": rel}
            done[i] = rec
        else:
            failed[i] = rec
    q = [{"id": i, "t": f"2006-08-{int(i[5:8]) - 212:02d}T18:00:00.000Z"} for i in ids]
    return q, done, failed


def test_dedup_same_overpass_kept_once_other_overpass_kept(tmp_path):
    c = (100.1, 1.3)
    a = rows("A2006213.1720", [c, (100.2, 1.3)], [T0, T0 + 1], vza=50.0)
    b = rows("A2006213.1725", [c], [T0 + 200], vza=10.0)   # same overpass, smaller vza wins
    n = rows("A2006213.1900", [c], [T0 + 6500], vza=60.0)  # next orbit: kept
    ids = ["A2006213.1720", "A2006213.1725", "A2006213.1900"]
    q, done, failed = setup(tmp_path, {ids[0]: (a, "ok", "v1.2"), ids[1]: (b, "ok", "v1.2"), ids[2]: (n, "ok", "v1.2")}, ids)
    agg.build(str(tmp_path), q, done, failed, "v1.2")
    m = pd.read_parquet(tmp_path / "monthly" / "2006-08.parquet")
    assert len(m) == 3 and (m.lon == 100.1).sum() == 2
    assert set(m[m.lon == 100.1].granule) == {"A2006213.1725", "A2006213.1900"}
    assert m.acq_date.iloc[0] == "2006-08-01"
    cnt = json.load(open(tmp_path / "counts.json"))["2006-08"]
    assert cnt["raw_rows"] == 4 and cnt["duplicates_removed"] == 1 and cnt["flagged_cells"] == 3


def test_version_gate_and_status(tmp_path):
    a = rows("A2006213.1720", [(100.1, 1.3)], [T0]); b = rows("A2006214.1720", [(101.1, 1.3)], [T0 + 86400])
    ids = ["A2006213.1720", "A2006214.1720", "A2006215.1720", "A2006216.1720"]
    q, done, failed = setup(tmp_path, {ids[0]: (a, "ok", "v1.2"), ids[1]: (b, "ok", "v1.1"), ids[2]: (None, "failed", None)}, ids)
    man = agg.build(str(tmp_path), q, done, failed, "v1.2")
    c = json.load(open(tmp_path / "coverage.json"))["2006-08"]
    assert (c["expected"], c["ok"], c["failed"], c["missing"], c["old_version"], c["complete"]) == (4, 1, 1, 1, 1, False)
    assert len(pd.read_parquet(tmp_path / "monthly" / "2006-08.parquet")) == 1  # the v1.1 granule is excluded
    assert man["months_complete"] == []
    assert vkey(None) == (1, 0) and vkey("v1.10") > vkey("v1.2")


def test_complete_month_and_zero_flag_granule(tmp_path):
    ids = ["A2006213.1720", "A2006214.1720"]
    a = rows(ids[0], [(100.1, 1.3)], [T0])
    q, done, failed = setup(tmp_path, {ids[0]: (a, "ok", "v1.2"), ids[1]: (None, "ok", "v1.2")}, ids)
    man = agg.build(str(tmp_path), q, done, failed, "v1.2")
    assert json.load(open(tmp_path / "coverage.json"))["2006-08"]["complete"] and man["months_complete"] == ["2006-08"] and man["n_flagged_cells"] == 1


def test_row_count_mismatch_blocks_complete_and_excludes_granule(tmp_path):
    a = rows("A2006213.1720", [(100.1, 1.3), (100.2, 1.3)], [T0, T0])
    q, done, failed = setup(tmp_path, {"A2006213.1720": (a, "ok", "v1.2")}, ["A2006213.1720"])
    done["A2006213.1720"]["flagged"] = 5
    agg.build(str(tmp_path), q, done, failed, "v1.2")
    c = json.load(open(tmp_path / "coverage.json"))["2006-08"]
    assert c["row_count_mismatch"] == 1 and not c["complete"]
    assert not (tmp_path / "monthly" / "2006-08.parquet").exists()


def test_header_granule_mismatch_is_reported(tmp_path):
    a = rows("A2006213.1720", [(100.1, 1.3)], [T0])
    q, done, failed = setup(tmp_path, {"A2006213.1720": (a, "ok", "v1.2")}, ["A2006213.1720"])
    t = pa.Table.from_pandas(a, preserve_index=False).replace_schema_metadata({"pipeline_version": "v1.2", "granule": "A2006213.1725"})
    pq.write_table(t, tmp_path / done["A2006213.1720"]["parquet"]["key"])
    agg.build(str(tmp_path), q, done, failed, "v1.2")
    assert json.load(open(tmp_path / "counts.json"))["2006-08"]["row_count_mismatch"][0]["note"].startswith("file header")


def test_expected_from_catalogue_and_from_records(tmp_path):
    a = rows("A2006213.1720", [(100.1, 1.3)], [T0])
    q, done, failed = setup(tmp_path, {"A2006213.1720": (a, "ok", "v1.2")}, ["A2006213.1720"])
    cat = tmp_path / "cat.jsonl"
    cat.write_text("".join(json.dumps(r) + "\n" for r in q + [{"id": "A2006220.1720", "t": "2006-08-08T18:00:00Z"}]))
    assert len(agg.expected_scenes([str(cat)], done, failed)) == 2
    assert [r["id"] for r in agg.expected_scenes(None, done, failed)] == ["A2006213.1720"]
    agg.build(str(tmp_path), agg.expected_scenes([str(cat)], done, failed), done, failed, "v1.2")
    assert not json.load(open(tmp_path / "coverage.json"))["2006-08"]["complete"]   # one expected granule has no record
