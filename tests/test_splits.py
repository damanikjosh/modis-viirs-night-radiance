"""Exclusion zones, episodes, 100 km blocks, train / val2 split, weights (nightrad.data.splits)."""
import numpy as np
import pandas as pd

from nightrad.data import splits as S


def tiles(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cl = rng.integers(0, 8, n)  # 8 clusters per region, 3 degrees apart (blocks of 100 km)
    lat = np.where(np.arange(n) < n // 2, -8.0, 20.0) + rng.normal(0, 0.1, n)
    lon = np.where(np.arange(n) < n // 2, 112.0, 95.0) + cl * 3.0 + rng.normal(0, 0.1, n)
    k = pd.DataFrame({"lat": lat, "lon": lon, "year": rng.choice([2016, 2019], n), "month": rng.integers(1, 12, n), "day": rng.integers(1, 28, n),
                      "modis_fire_px": rng.integers(0, 5, n), "viirs_fire_cells": rng.integers(0, 40, n)})
    k["nf_region"] = np.where(lat < 0, "Java/Nusa Tenggara/Sulawesi S", "Mainland SE Asia")
    k["group_name"] = S.group_of(k.nf_region)
    k["episode"] = ["E%d" % e for e in S.episodes(k)]
    return k


def test_blocks_are_single_linkage_components():
    lat, lon = np.array([0, 0.5, 1.0, 10.0]), np.zeros(4)  # 55 km steps: 0-1-2 chain, 3 alone
    b = S.blocks(lat, lon, 100.0)
    assert b[0] == b[1] == b[2] != b[3]
    assert len(set(S.blocks(lat, lon, 20.0))) == 4


def test_episodes_need_distance_and_days():
    df = pd.DataFrame({"lat": [0, 0.1, 0.1, 5], "lon": [0, 0, 0, 0], "year": 2019, "month": 5, "day": [1, 3, 20, 1]})
    e = S.episodes(df)
    assert e[0] == e[1] and e[2] != e[0] and e[3] != e[0]


def test_exclusion_reason_first_match_and_centres():
    df = pd.DataFrame({"lat": [-2.0, 0.0, -1.0, 40.0, 10.0, 40.0], "lon": [103.0, 113.0, 108.3, 10.0, 20.0, 10.2], "year": [2019, 2015, 2019, 2023, 2019, 2019],
                       "month": [5, 9, 5, 5, 5, 5], "region_cont": ["AS", "AS", "AS", "EU", "AF", "EU"]})
    why, d = S.exclusion_reason(df, pd.DataFrame({"lat": [40.0], "lon": [10.0]}))
    assert list(why) == ["sumatra_test_block_60km", "kalimantan_sep_oct_2015_150km", "sumatra_test_block_60km", "year2023", "", "near_exclusion_centre"]
    assert d[4] > 100 and d[5] < 60


def test_centre_filter_matches_exclusion_zones():
    f = S.centre_filter(pd.DataFrame({"lat": [40.0], "lon": [10.0]}), 2015, 9)
    keep = f(np.array([-2.0, 0.0, 40.0, 10.0, -10.0]), np.array([103.0, 113.0, 10.0, 20.0, -60.0]))
    assert keep.tolist() == [False, False, False, True, False]
    assert S.centre_filter(None, 2019, 5)(np.array([0.0]), np.array([113.0])).tolist() == [True]


def test_choose_split_keeps_blocks_whole_and_respects_buffer():
    k = tiles()
    iv, info = S.choose_split(k, seeds=range(20))
    blk = S.blocks(k.lat.to_numpy(), k.lon.to_numpy())
    assert all(len(set(iv[blk == b])) == 1 for b in set(blk))
    assert 0 < iv.sum() < len(k) and info["val2_blocks"] >= 1
    d = S.min_dist_km(k.lat[~iv].to_numpy(), k.lon[~iv].to_numpy(), k.lat[iv].to_numpy(), k.lon[iv].to_numpy())
    assert d.min() >= S.BLOCK_KM
    iv2, _ = S.choose_split(k, seeds=range(20))
    assert (iv == iv2).all()  # deterministic


def test_region_weights_hit_targets():
    k = tiles()
    k["split"] = "train"
    k["group_name"] = np.where(np.arange(len(k)) % 2 == 0, "Southeast Asia", "Africa south")
    w = S.region_weights(k, sea_share=0.2)
    fw = (k.modis_fire_px + k.viirs_fire_cells / 5).to_numpy() * w
    sea = (k.group_name == "Southeast Asia").to_numpy()
    assert abs(fw[sea].sum() / fw.sum() - 0.2) < 0.02
