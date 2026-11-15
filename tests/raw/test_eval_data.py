import json
import os

import numpy as np
import pytest
import torch

from nightrad.raw import prereg
from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards, make_loader, pack_dataset
from nightrad.raw.hide import HideConfig
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.schema import adapt_v3_tile, neighbour_table
from nightrad.raw.synth import SyntheticTiles
from raw_helpers import tiny_cfg


def test_test_split_guard(tmp_path):
    cfg = {"a": 1}
    assert prereg.open_split("/data/train", cfg=cfg) == "/data/train"
    with pytest.raises(PermissionError):
        prereg.open_split("/data/test_2023")
    log = str(tmp_path / "access.log")
    with pytest.raises(PermissionError):  # allowed but nothing frozen
        prereg.open_split("test_SA", allow_test=True, frozen_path=str(tmp_path / "none.json"), log_path=log)
    fz = str(tmp_path / "raw.json")
    h = prereg.freeze(cfg, fz)
    with pytest.raises(PermissionError):  # config changed after freezing
        prereg.open_split("test_SA", allow_test=True, frozen_path=fz, cfg={"a": 2}, log_path=log)
    assert not os.path.exists(log)
    prereg.open_split("test_SA", allow_test=True, frozen_path=fz, cfg=cfg, log_path=log)
    rec = json.loads(open(log).read().splitlines()[-1])
    assert rec["config"] == h and rec["split"] == "test_SA" and "git" in rec


def test_pack_and_read_shards(tmp_path):
    ds = SyntheticTiles(6, 0, 48, 48)
    pack_dataset(ds, str(tmp_path / "shards"))
    ps = PackedShards(str(tmp_path / "shards"))
    a, b = ds[2], ps[2]
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(torch.nan_to_num(a[k].float()), torch.nan_to_num(b[k].float())), k
    batch = next(iter(make_loader(ps, 3, 2)))
    assert batch["viirs_L"].shape == (3, 2, 48, 48)


def test_neighbour_table_and_adapter_roundtrip():
    raw = np.full((4, 6), -1, np.int64)
    raw[:, :3] = np.array([[10, 11, 12]] * 4); raw[:, 3:] = 10 + 1354 * 1
    raw[2:, :] = raw[2:, :] + 1354
    dense = np.unique(raw, return_inverse=True)[1].reshape(raw.shape).astype(np.int32)
    nb = neighbour_table(raw, dense)
    # cell (0,0) has id 10; its along-scan + neighbour is id 11 at (0,1)
    assert nb[1, 0, 0] == dense[0, 1] and nb[0, 0, 0] == -1 and nb[3, 0, 0] == dense[2, 0] if raw[2, 0] == 10 + 1354 else True
    keys = {"modis_pid": raw, "viirs_pid": raw}
    z = {}
    for s in ("modis", "viirs"):
        z.update({f"{s}_rad4": np.ones((4, 6)), f"{s}_rad11": np.ones((4, 6)), f"{s}_vclass": np.full((4, 6), 2), f"{s}_t4_band": np.zeros((4, 6)), f"{s}_sat": np.zeros((4, 6)),
                  f"{s}_psize": np.ones((2, 4, 6)), f"{s}_vza": np.ones((4, 6)), f"{s}_cossza": np.ones((4, 6))})
    z.update({"modis_pid": raw, "modis_footprint_complete": np.ones((4, 6)), "eval_fire_class": np.zeros((4, 6))})
    out = adapt_v3_tile(z)
    assert out["modis_pid"].max() >= 0 and "eval_fire_class" in out and out["modis_nbr"].shape == (4, 4, 6) and out["viirs_L"].shape == (2, 4, 6)
