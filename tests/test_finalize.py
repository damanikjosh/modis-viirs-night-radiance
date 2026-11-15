"""scripts/finalize.py on small synthetic stats: exclusions, split by block, buffer, leak check, weights, written tiles."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_build(root: Path, n_per=12):
    rng = np.random.default_rng(1)
    pairs = []
    for p in range(6):
        pid = f"N{p:04d}"
        (root / "tiles" / pid).mkdir(parents=True)
        rows = []
        lat0, lon0 = 20.0 + 3.0 * p, 95.0 + 4.0 * p  # one cluster per pair, far apart
        for i in range(n_per):
            name = f"{pid}_{i:03d}"
            np.savez(root / "tiles" / pid / f"{name}.npz", x=np.zeros(3), sample_weight=np.float32(1), split=np.array("train"), holdout=np.array(""))
            rows.append({"tile": name, "pair_id": pid, "region": "AS", "year": 2019, "month": 5, "day": 3, "lat": lat0 + rng.normal(0, .05), "lon": lon0 + rng.normal(0, .05),
                         "modis_fire_px": int(rng.integers(0, 4)), "viirs_fire_cells": int(rng.integers(0, 30)), "has_fire": int(i % 3 == 0), "random_tile": int(i % 4 == 0), "daynight": 1})
        (root / "stats").mkdir(exist_ok=True)
        json.dump({"pair_id": pid, "rows": rows}, open(root / "stats" / f"{pid}.json", "w"))
        pairs.append({"pair_id": pid, "region": "Mainland SE Asia"})
    # one pair inside the Sumatra block: every tile excluded
    pid = "N0099"
    (root / "tiles" / pid).mkdir(parents=True)
    rows = []
    for i in range(3):
        name = f"{pid}_{i:03d}"
        np.savez(root / "tiles" / pid / f"{name}.npz", x=np.zeros(3), sample_weight=np.float32(1), split=np.array("train"), holdout=np.array(""))
        rows.append({"tile": name, "pair_id": pid, "region": "AS", "year": 2019, "month": 5, "day": 3, "lat": -2.0, "lon": 103.0, "modis_fire_px": 1, "viirs_fire_cells": 1,
                     "has_fire": 1, "random_tile": 0, "daynight": 1})
    json.dump({"pair_id": pid, "rows": rows}, open(root / "stats" / f"{pid}.json", "w"))
    pairs.append({"pair_id": pid, "region": "Borneo"})
    pd.DataFrame(pairs).to_csv(root / "pairs.csv", index=False)


def test_finalize_end_to_end(tmp_path, monkeypatch):
    make_build(tmp_path / "b")
    out = tmp_path / "t"
    (tmp_path / "c.csv").write_text("lat,lon\n-80,0\n")
    monkeypatch.setattr(sys, "argv", ["finalize.py", "--pairs", str(tmp_path / "b" / "pairs.csv"), "--build", str(tmp_path / "b"), "--out", str(out), "--workers", "1", "--centres", str(tmp_path / "c.csv")])
    try:
        runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "finalize.py"), run_name="__main__")
    except SystemExit as e:
        assert e.code == 0
    leak = json.load(open(out / "leak_check.json"))
    assert leak["ok"] and leak["dropped_by_reason"]["sumatra_test_block_60km"] == 3
    m = pd.read_parquet(out / "manifest.parquet")
    assert len(m) == 72 and set(m.split) == {"train", "val2"} and not m.pair_id.eq("N0099").any()
    blocks = m.assign(b=m.pair_id).groupby("b").split.nunique()
    assert (blocks == 1).all()  # each cluster (block) lies wholly on one side
    t = np.load(out / m.path.iloc[0], allow_pickle=False)
    assert str(t["split"]) in ("train", "val") and float(t["sample_weight"]) == float(m.sample_weight.iloc[0])
    assert (m.groupby("split").sample_weight.mean().round(6) == 1).all()  # stratum weights average 1 inside a split
