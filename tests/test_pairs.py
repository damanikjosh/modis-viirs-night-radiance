"""Pair finder tests: saved CMR JSON fixtures (2019-09-14 03:00-03:20 Aqua; +/-30 min VIIRS), no network."""
import json
from pathlib import Path

import pandas as pd
import pytest

from nightrad.data import pairs as P
from nightrad.data.download import LoginError, data_root, fetch

FX = Path(__file__).parent / "fixtures"


def load(name):
    out = []
    for e in json.load(open(FX / f"cmr_{name}.json"))["feed"]["entry"]:
        r = P.granule_record(e)
        r["geom"] = P.granule_geom(r["polys"])
        out.append(r)
    return out


@pytest.fixture(scope="module")
def recs():
    return {n: load(n) for n in ["MYD14", "VNP14IMG", "VJ114IMG", "MYD03", "VNP03MOD", "VJ103MOD"]}


def test_granule_record(recs):
    m = recs["MYD14"][0]
    assert m["id"].startswith("MYD14.A2019257.") and not m["id"].endswith(".hdf")
    assert m["url"].startswith("https://") and m["url"].endswith(".hdf")
    assert recs["VNP14IMG"][0]["url"].endswith(".nc")
    assert m["geom"] is not None and m["geom"].is_valid


def test_pairs_within_30min(recs):
    prs = P.find_pairs(recs["MYD14"], {"VNP14IMG": recs["VNP14IMG"], "VJ114IMG": recs["VJ114IMG"]})
    assert prs
    assert all(abs(p["dt_min"]) <= 30 for p in prs)
    assert all(p["area"] > 0 and p["overlap"].is_valid for p in prs)
    assert {p["viirs_product"] for p in prs} <= {"VNP14IMG", "VJ114IMG"}
    tight = P.find_pairs(recs["MYD14"], {"VNP14IMG": recs["VNP14IMG"]}, max_dt_min=10)
    assert len(tight) < len(prs) and all(abs(p["dt_min"]) <= 10 for p in tight)


def test_pair_row_names_the_platform(recs):
    p = P.find_pairs(recs["MYD14"], {"VNP14IMG": recs["VNP14IMG"]})[0]
    r = P.pair_row(p)
    assert r["viirs_sensor"] == "snpp" and r["viirs_id"].startswith("VNP14IMG.") and r["modis_id"].startswith("MYD14.")
    assert P.pair_row({**p, "viirs_product": "VJ114IMG"})["viirs_sensor"] == "noaa20"


def test_overlap_inside_both(recs):
    p = P.find_pairs(recs["MYD14"], {"VNP14IMG": recs["VNP14IMG"]})[0]
    assert p["modis"]["geom"].buffer(1e-6).contains(p["overlap"])
    assert p["viirs"]["geom"].buffer(1e-6).contains(p["overlap"])


def test_antimeridian_overlap_and_area():
    a = P.parse_ring("0 170 0 -170 10 -170 10 170 0 170")    # lat lon pairs, crosses 180
    b = P.parse_ring("2 175 2 -175 8 -175 8 175 2 175")
    assert a.bounds[2] > 180
    ov = P.overlap_geom(a, b)
    assert ov is not None and ov.area == pytest.approx(60, rel=0.01)  # 10 deg lon x 6 deg lat
    assert 0.5e6 < P.area_km2(ov) < 0.8e6                          # ~1100 km x 660 km
    assert P.overlap_geom(a, P.parse_ring("0 0 0 10 10 10 10 0 0 0")) is None


def test_attach_granules_matches_by_stamp(monkeypatch):
    def fake(coll, a, b, bbox=None, session=None, page_size=2000):
        prod = next(k for k, v in P.COLLECTION.items() if v == coll)
        ext = ".hdf" if prod.startswith("MY") else ".nc"
        ent = []
        for st in ("A2019291.1625", "A2019291.1618"):
            ent.append({"producer_granule_id": f"{prod}.{st}.061.2020000000000{ext}", "time_start": "2019-10-18T16:25:00Z", "time_end": "2019-10-18T16:30:00Z",
                        "links": [{"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": f"https://x/{prod}.{st}.061.2020000000000{ext}"}]})
        return ent

    monkeypatch.setattr(P, "cmr_granules", fake)
    df = pd.DataFrame([{"pair_id": "X", "modis_id": "MYD14.A2019291.1625.061.1", "viirs_id": "VNP14IMG.A2019291.1618.002.1", "viirs_sensor": "snpp",
                        "modis_time": "2019-10-18T16:27:30Z", "viirs_time": "2019-10-18T16:21:00Z"},
                       {"pair_id": "Y", "modis_id": "MYD14.A2019291.0000.061.1", "viirs_id": "VNP14IMG.A2019291.0000.002.1", "viirs_sensor": "snpp",
                        "modis_time": "2019-10-18T16:27:30Z", "viirs_time": "2019-10-18T16:21:00Z"}])
    out = P.attach_granules(df, threads=1, log=lambda s: None)
    assert list(out.pair_id) == ["X"]                                     # Y has no granule with its stamp
    r = out.iloc[0]
    assert r.modis_l1b_id.startswith("MYD021KM.A2019291.1625") and r.viirs_geo_id.startswith("VNP03MOD.A2019291.1618") and r.viirs_l1b_url.endswith("VNP02IMG.A2019291.1618.061.2020000000000.nc")
    files = P.product_files(r.to_dict())
    assert [f[0] for f in files] == ["MYD14", "MYD03", "MYD021KM", "VNP14IMG", "VNP03MOD", "VNP02IMG"]


def test_pair_paths(tmp_path):
    for prod in ("MYD14", "MYD03", "MYD021KM", "VNP14IMG", "VNP03MOD", "VNP02IMG"):
        (tmp_path / prod).mkdir()
        (tmp_path / prod / f"{prod}.A2019291.1625.x.hdf").write_bytes(b"")
    paths = P.pair_paths(tmp_path)
    assert [p.parent.name for p in paths] == ["MYD14", "MYD03", "MYD021KM", "VNP14IMG", "VNP03MOD", "VNP02IMG"]


def test_fetch_skips_existing_and_checks_size(tmp_path):
    f = tmp_path / "a.hdf"
    f.write_bytes(b"x")
    assert fetch(None, "http://x/a.hdf", f) == "skipped"

    class R:
        status_code, headers = 200, {"Content-Length": "10"}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def raise_for_status(self): pass
        def iter_content(self, n): yield b"abc"

    class S:
        def get(self, *a, **k): return R()

    with pytest.raises(OSError, match="expected 10"):
        fetch(S(), "http://x/b.hdf", tmp_path / "b.hdf")


def test_login_error_message(monkeypatch):
    import earthaccess
    from nightrad.data import download as D
    monkeypatch.delenv("NR_EARTHDATA_TOKEN", raising=False)
    def boom(**k):
        raise RuntimeError("no creds")
    monkeypatch.setattr(earthaccess, "login", boom)
    with pytest.raises(LoginError, match="netrc"):
        D.login()


def test_token_login_and_data_root(monkeypatch, tmp_path):
    from nightrad.data import download as D
    monkeypatch.setenv("NR_EARTHDATA_TOKEN", "abc")
    assert D.login().headers["Authorization"] == "Bearer abc"
    monkeypatch.setenv("NR_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path
