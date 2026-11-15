"""Tile build rules: |dt| filter, random fire offset, provenance keys, centre filter, build-time split label, stratum weights."""
import numpy as np

from nightrad.data import splits as S
from nightrad.data.tiles import FIRE_OFFSET_KM, build_pair_tiles, shift_centres, validate_tile
from nightrad.data.tiles_gpu import build_pair_tiles_gpu
from test_tiles import T0, make_swath
from test_tiles_gpu import assert_tiles_equal


def pair_at(dt_s):
    m = make_swath("modis", T0, fires=[(110, 67, 5, 20.0), (115, 70, 4, 8.0)])
    v = make_swath("viirs", T0 + dt_s, fires=[(500, 320, 5, 3.0), (503, 322, 6, 9.0), (520, 330, 4, 1.0)])
    return m, v


def test_dt_filter_drops_tiles_above_30_min():
    m, v = pair_at(25 * 60)
    assert len(build_pair_tiles(m, v, max_fire_tiles=4, n_random=3, seed=1)) > 0  # 25 min + half granule < 30
    m, v = pair_at(45 * 60)
    assert build_pair_tiles(m, v, max_fire_tiles=4, n_random=3, seed=1) == []  # every tile > 30 min


def test_shift_bounded_deterministic_and_fire_not_centred():
    cand = np.array([[10.0, 100.0], [60.0, -170.0], [-33.0, 179.9]])
    a, oa = shift_centres(cand, np.random.default_rng(3))
    b, ob = shift_centres(cand, np.random.default_rng(3))
    assert np.array_equal(a, b) and np.array_equal(oa, ob) and np.abs(oa).max() <= FIRE_OFFSET_KM
    d_km = np.hypot((a[:, 0] - cand[:, 0]) * 111.195, ((a[:, 1] - cand[:, 1] + 180) % 360 - 180) * 111.195 * np.cos(np.deg2rad(cand[:, 0])))
    assert np.allclose(d_km, np.hypot(oa[:, 0], oa[:, 1]), atol=0.05) and (np.abs(a[:, 1]) <= 180).all()


def test_pair_tiles_have_offsets_and_provenance_and_cuda_path_equal():
    m, v = pair_at(600)
    tiles = build_pair_tiles(m, v, max_fire_tiles=4, n_random=3, seed=1)
    fire_t = [t for t in tiles if not int(t["random_tile"])]
    assert fire_t and any(np.abs(t["fire_offset_km"]).max() > 1 for t in fire_t)
    assert all(not np.any(t["fire_offset_km"]) for t in tiles if int(t["random_tile"]))
    t = fire_t[0]
    validate_tile(t)
    assert int(t["schema_version"]) == 5 and str(t["satellite"]) == "snpp" and int(t["year"]) == 2023 and int(t["month"]) == 11
    assert float(t["viirs_utc"]) - float(t["modis_utc"]) == float(t["dt_min"]) * 60 or abs(float(t["viirs_utc"]) - float(t["modis_utc"]) - 60 * float(t["dt_min"])) < 1
    g = build_pair_tiles_gpu(m, v, device="cpu", batch=3, max_fire_tiles=4, n_random=3, seed=1)
    assert len(g) == len(tiles)
    for a, b in zip(tiles, g):
        assert_tiles_equal(a, b)


def test_split_rule():
    assert S.split_of(-10, -60, "SA", 2013, 5) == ("test", "SA")                    # held-out region, any year
    assert S.split_of(-2, 103, "AS", 2013, 5) == ("test", "indonesia_block")        # Sumatra test block, any year
    assert S.split_of(-2, 112, "AS", 2013, 5) == ("train", "")                      # rest of Indonesia (Kalimantan) is train
    assert S.split_of(-2, 112, "AS", 2019, 8) == ("val", "")
    assert S.split_of(40, 10, "EU", 2023, 5) == ("test", "time2023")
    assert S.split_of(40, 10, "EU", 2019, 8) == ("val", "") and S.split_of(40, 10, "EU", 2022, 3) == ("val", "")
    assert S.split_of(40, 10, "EU", 2013, 8) == ("train", "") and S.split_of(40, 10, "EU", 2019, 1) == ("train", "")


def test_centre_filter_cpu_and_torch_paths_agree_and_none_is_unchanged():
    m, v = pair_at(600)
    base = build_pair_tiles(m, v, max_fire_tiles=4, n_random=3, seed=1)
    assert len(build_pair_tiles(m, v, max_fire_tiles=4, n_random=3, seed=1, centre_filter=None)) == len(base)
    keep_none = lambda lat, lon: np.zeros(len(lat), bool)          # drops every fire-centred candidate; random tiles are not filtered
    cpu = build_pair_tiles(m, v, max_fire_tiles=4, n_random=3, seed=1, centre_filter=keep_none)
    gpu = build_pair_tiles_gpu(m, v, device="cpu", batch=3, max_fire_tiles=4, n_random=3, seed=1, centre_filter=keep_none)
    assert len(cpu) == len(gpu) and all(int(t["random_tile"]) for t in cpu) and len(cpu) < len(base)
    for a, b in zip(cpu, gpu):
        assert_tiles_equal(a, b)


def test_stratum_weights_match_random_fire_share():
    split = np.array(["train"] * 200)
    rnd = np.zeros(200, bool); rnd[100:] = True          # 100 random tiles
    fire = np.zeros(200, bool); fire[:100] = True; fire[100:104] = True  # all 100 fire-centred + 4 random ones with fire -> q = 4%
    w = S.stratum_weights(split, fire, rnd)
    assert abs(w.mean() - 1) < 1e-6 and abs((w * fire).sum() / w.sum() - 0.04) < 1e-6
