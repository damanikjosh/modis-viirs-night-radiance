"""Night-fire selection: FIRMS reading, eligibility, regions, candidate nights, quota (no network)."""
import numpy as np
import pandas as pd

from nightrad.data import nightfire as N


def det(n=50, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2019-03-01 18:00") + pd.to_timedelta(rng.integers(0, 60 * 24 * 40, n), unit="m")
    d = pd.DataFrame({"lat": rng.uniform(15, 20, n), "lon": rng.uniform(98, 102, n), "ts": ts, "sensor": rng.choice(["aqua", "viirs"], n), "country": "Thailand"})
    d["region"] = N.region_of_detections(d)
    return d


def test_read_firms_keeps_night_vegetation_fire_and_aqua(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("latitude,longitude,acq_date,acq_time,satellite,daynight,type\n1,2,2019-03-01,0130,Aqua,N,0\n1,2,2019-03-01,0130,Terra,N,0\n"
                 "1,2,2019-03-01,1330,Aqua,D,0\n1,2,2019-03-01,0130,Aqua,N,2\n")
    d = N.read_firms(f, "aqua", "Thailand")
    assert len(d) == 1 and d.ts.iloc[0] == pd.Timestamp("2019-03-01 01:30")
    assert len(N.read_firms(f, "viirs", "Thailand")) == 2  # satellite filter applies to MODIS only


def test_regions():
    d = pd.DataFrame({"lat": [-6.0, 12.0, 5.0, -3.0], "lon": [112.0, 100.0, 115.0, 25.0], "country": ["Indonesia", "Thailand", "Indonesia", "Democratic_Republic_of_the_Congo"]})
    assert list(N.region_of_detections(d)) == ["Java/Nusa Tenggara/Sulawesi S", "Mainland SE Asia", "Borneo", "Africa south"]


def test_eligibility_reasons():
    d = pd.DataFrame({"lat": [-2.0, 0.0, -10.0, 15.0, 15.0], "lon": [103.0, 113.0, -60.0, 100.0, 100.0],
                      "ts": pd.to_datetime(["2019-05-01", "2015-09-20", "2019-05-01", "2023-01-01", "2019-05-01"])})
    e = N.eligibility(d)
    assert list(e.why) == ["sumatra_test_block", "kalimantan_sep_oct_2015", "south_america", "year2023", ""]
    assert list(N.eligibility(d, pd.DataFrame({"lat": [15.2], "lon": [100.0]})).why)[-1] == "near_exclusion_centre"


def test_candidate_nights_split_strong_and_weak():
    rows = []
    for night, na, nv in [("2019-03-01", 5, 3), ("2019-03-02", 0, 40), ("2019-03-03", 1, 5)]:
        ts = pd.Timestamp(night + " 20:00")
        rows += [{"ts": ts, "sensor": "aqua", "region": "Mainland SE Asia", "eligible": True, "nid": ts.date()}] * na
        rows += [{"ts": ts, "sensor": "viirs", "region": "Mainland SE Asia", "eligible": True, "nid": ts.date()}] * nv
    keys = N.candidate_nights(pd.DataFrame(rows))
    nights = {str(k[1]) for k in keys}
    assert nights == {"2019-03-01", "2019-03-02", "2019-03-03"}  # strong: >=1 Aqua in SE Asia; weak: night 2 (Aqua 0, VIIRS 40)


def test_count_in_unwrapped_geometry():
    from shapely.geometry import box
    d = pd.DataFrame({"lat": [5.0, 5.0], "lon": [-175.0, 175.0], "ts": pd.to_datetime(["2019-01-01 01:00"] * 2)})
    assert N.count_in(d, box(170, 0, 190, 10), "2019-01-01 00:00", "2019-01-01 02:00") == 2
    assert N.count_in(d, box(0, 0, 10, 10), "2019-01-01 00:00", "2019-01-01 02:00") == 0


def test_select_quota_fills_year_round_robin_and_adds_weak():
    rows = []
    for y in (2014, 2016, 2018):
        for i in range(5):
            rows.append({"region": "Mediterranean", "modis_id": f"M{y}{i}", "viirs_id": f"V{y}{i}", "modis_time": f"{y}-06-0{i + 1}T20:00:00Z", "n_aqua_elig": 100 - i,
                         "n_viirs_elig": 50, "score": 110 - i})
    rows += [{"region": "Mediterranean", "modis_id": f"W{i}", "viirs_id": f"WV{i}", "modis_time": "2015-06-01T20:00:00Z", "n_aqua_elig": 0, "n_viirs_elig": 30 + i, "score": 6 + i} for i in range(4)]
    sel = N.select_quota(pd.DataFrame(rows), quota=300, rate=0.7)  # 100*.7=70 per pair -> 5 strong pairs
    strong = sel[sel.kind == "strong"]
    assert len(strong) == 5 and set(pd.to_datetime(strong.modis_time, utc=True).dt.year) == {2014, 2016, 2018}
    assert (sel.kind == "weak").sum() == 2  # half the strong count (rounded)
    ex = sel.head(2)
    again = N.select_quota(pd.DataFrame(rows), existing=ex.assign(pair_id=["N0000", "N0001"]), quota=300, rate=0.7)
    assert not set(zip(ex.modis_id, ex.viirs_id)) & set(zip(again.modis_id, again.viirs_id))
