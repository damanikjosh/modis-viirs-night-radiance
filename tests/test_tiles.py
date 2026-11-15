import json
from pathlib import Path

import numpy as np
import pytest

from nightrad.data import readers
from nightrad.data.readers import Swath, map_fire_mask, upsample2x, upsample2x_geo
from nightrad.data.tiles import H, W, build_pair_tiles, build_tile, validate_tile

T0 = 1_700_000_000.0


def make_swath(kind: str, t0: float, lat0=37.0, lon0=127.0, fires=(), nlines=None) -> Swath:
    """Flat-earth mimic of a reader output. MODIS: 1 km at nadir growing to ~4.8 km at the edge; VIIRS: ~0.4-0.8 km."""
    if kind == "modis":
        S, L = 135, 220
        k = np.abs(np.arange(S) - (S - 1) / 2) / ((S - 1) / 2)
        size_as = 1.0 + 3.8 * k ** 2
        size_at = 1.0 + 0.0 * k
        km_line = 1.0
    else:
        S, L = 640, 1000
        k = np.abs(np.arange(S) - (S - 1) / 2) / ((S - 1) / 2)
        size_as = 0.375 + 0.4 * k ** 2
        size_at = np.full(S, 0.375)
        km_line = 0.375
    sgn = np.sign(np.arange(S) - (S - 1) / 2)
    x_km = np.cumsum(size_as)
    x_km = x_km - x_km[S // 2]
    lat = lat0 + (np.arange(L) - L / 2)[:, None] * km_line / 111.2 + 0 * x_km[None]
    lon = lon0 + x_km[None] / (111.2 * np.cos(np.deg2rad(lat0))) + 0 * lat
    vza = np.rad2deg(np.arctan(np.abs(x_km) / 705.0))[None] + 0 * lat
    cls = np.full((L, S), 3, np.uint8)
    cls[:, :5] = 0
    cls[: L // 5] = 2
    frp = np.zeros((L, S), np.float32)
    for (li, si, c, f) in fires:
        cls[li, si], frp[li, si] = c, f
    return Swath(kind, "aqua" if kind == "modis" else "snpp", f"{kind}_test", lat.astype(np.float32),
                 lon.astype(np.float32), cls, frp, vza.astype(np.float32),
                 np.full((L, S), 40, np.float32), t0, t0 + 300)


@pytest.fixture(scope="module")
def pair():
    # same ground spot: modis pixel (110, 67) at the swath centre; viirs fires nearby
    m = make_swath("modis", T0, fires=[(110, 67, 5, 20.0), (115, 70, 4, 8.0)])
    v = make_swath("viirs", T0 + 600, fires=[(500, 320, 5, 3.0), (503, 322, 6, 9.0), (520, 330, 4, 1.0)])
    return m, v


def test_class_mapping_table():
    raw = np.arange(10, dtype=np.uint8)
    assert map_fire_mask(raw).tolist() == [0, 0, 0, 1, 2, 3, 0, 4, 5, 6]
    assert map_fire_mask(np.array([100, 255], np.uint8)).tolist() == [0, 0]


def test_upsample_linear_exact_and_antimeridian():
    a = np.arange(6.0)[:, None] * np.ones((1, 4))
    u = upsample2x(a, 0)
    assert u.shape == (12, 4)
    assert np.allclose(u[:, 0], (np.arange(12) - 0.5) / 2)
    lat = np.full((4, 4), 10.0)
    lon = np.array([[179.0, 179.5, -179.5, -179.0]] * 4)
    la2, lo2 = upsample2x_geo(lat, lon, scan_lines=None)
    assert la2.shape == (8, 8) and np.allclose(la2, 10, atol=1e-3)
    assert np.allclose(np.abs(lo2[0, 3:5]), 180, atol=0.3) and np.all(np.abs(lo2) > 178.5)


def test_build_tile_schema_and_content(pair):
    m, v = pair
    lat, lon = m.lat[110, 67], m.lon[110, 67]
    t = build_tile(m, v, float(lat), float(lon))
    validate_tile(t)
    assert t["modis_cls"].shape == (H, W)
    assert (t["modis_cls"] >= 4).any() and (t["viirs_cls"] >= 4).any()
    assert t["overlap"].mean() > 0.9
    assert t["dt_min"] == pytest.approx(10.0, abs=0.5)
    assert t["daynight"] == 0
    # fire cells carry log1p(FRP) of their pixel; MODIS pixel (110,67) -> pid
    pid = 110 * m.shape[1] + 67
    cells = t["modis_pid"] == pid
    assert cells.sum() >= 4  # a 1 km pixel covers several 375 m cells
    assert np.unique(t["modis_cls"][cells]).tolist() == [5]
    assert np.allclose(t["modis_frp"][cells], np.log1p(20.0))
    # pixel size at nadir ~1 km, viirs ~0.4 km
    assert t["modis_psize"][0][cells].mean() == pytest.approx(1.0, abs=0.15)
    assert t["viirs_psize"][0][t["viirs_pid"] >= 0].mean() < 0.6
    assert set(json.loads(str(t["meta"]))) >= {"modis_granule", "viirs_granule", "sensor", "lat", "lon", "date"}


def test_psize_grows_with_scan_angle():
    m = make_swath("modis", T0, fires=[(110, 125, 5, 5.0)])
    v = make_swath("viirs", T0 + 300)
    t = build_tile(m, v, float(m.lat[110, 125]), float(m.lon[110, 125]))
    validate_tile(t)
    obs = t["modis_pid"] >= 0
    assert obs.any()
    assert t["modis_psize"][0][obs].mean() > 3.0
    assert np.abs(t["modis_scan"][obs]).mean() > 5
    assert (t["modis_scan"][obs] > 0).all()  # east half of the swath -> positive


def test_outside_swath_is_class0():
    m = make_swath("modis", T0)
    v = make_swath("viirs", T0 + 300)
    t = build_tile(m, v, 37.0, 127.0 + 3.0)  # far east of both swaths
    validate_tile(t)
    assert t["overlap"].sum() == 0 and (t["modis_cls"] == 0).all() and (t["modis_pid"] == -1).all()


def test_build_pair_tiles(pair):
    m, v = pair
    tiles = build_pair_tiles(m, v, max_fire_tiles=5, n_random=6, seed=1)
    assert len(tiles) >= 1
    for t in tiles:
        validate_tile(t)
    has_fire = [bool((t["modis_cls"] >= 4).any() or (t["viirs_cls"] >= 4).any()) for t in tiles]
    assert any(has_fire) and not all(has_fire)
    rnd = [int(t["random_tile"]) for t in tiles]
    assert sum(rnd) == 6 and len(tiles) - sum(rnd) <= 5  # 6 random + up to 5 fire-centred
    assert all(float(t["sample_weight"]) > 0 for t in tiles)


def test_npz_roundtrip_without_pickle(pair, tmp_path):
    m, v = pair
    t = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]))
    np.savez_compressed(tmp_path / "t.npz", **t)
    z = np.load(tmp_path / "t.npz", allow_pickle=False)
    validate_tile({k: z[k] for k in z.files})


def test_viirs_reader_roundtrip(tmp_path):
    nc = pytest.importorskip("netCDF4")
    f14, f03 = tmp_path / "VNP14IMG.A2026318.0112.002.x.nc", tmp_path / "VNP03MOD.A2026318.0112.021.x.nc"
    raw = np.full((64, 80), 5, np.uint8)
    raw[10, 10], raw[11, 12] = 8, 9
    with nc.Dataset(f14, "w") as d:
        d.createDimension("y", 64), d.createDimension("x", 80), d.createDimension("fp", 2)
        d.createVariable("fire mask", "u1", ("y", "x"))[:] = raw
        d.createVariable("FP_line", "i4", ("fp",))[:] = [10, 11]
        d.createVariable("FP_sample", "i4", ("fp",))[:] = [10, 12]
        d.createVariable("FP_power", "f4", ("fp",))[:] = [4.5, 12.0]
    with nc.Dataset(f03, "w") as d:
        g = d.createGroup("geolocation_data")
        g.createDimension("y", 32), g.createDimension("x", 40)
        for n, val in [("latitude", 30.0), ("longitude", 100.0), ("sensor_zenith", 20.0), ("solar_zenith", 30.0)]:
            g.createVariable(n, "f4", ("y", "x"))[:] = np.full((32, 40), val, np.float32)
    s = readers.read_viirs(f14, f03)
    assert s.platform == "snpp" and s.shape == (64, 80)
    assert s.cls[10, 10] == 5 and s.cls[11, 12] == 6 and s.cls[0, 0] == 3
    assert s.frp[10, 10] == 4.5 and s.frp[11, 12] == 12.0 and s.frp.sum() == pytest.approx(16.5)
    assert s.t1 - s.t0 == 360 and s.vza[5, 5] == pytest.approx(20.0, abs=1e-4)
    assert s.conf[10, 10] == pytest.approx(0.6) and s.conf[11, 12] == pytest.approx(0.9) and s.conf[0, 0] == 0  # n / h


def test_modis_reader_roundtrip(tmp_path):
    hdf = pytest.importorskip("pyhdf.SD")
    SD, SDC = hdf.SD, hdf.SDC
    f14, f03 = tmp_path / "MYD14.A2026318.0105.061.x.hdf", tmp_path / "MYD03.A2026318.0105.061.x.hdf"
    raw = np.full((20, 30), 5, np.uint8)
    raw[3, 4] = 7
    sd = SD(str(f14), SDC.WRITE | SDC.CREATE)
    for name, typ, arr in [("fire mask", SDC.UINT8, raw), ("FP_line", SDC.INT32, np.array([3], np.int32)),
                           ("FP_sample", SDC.INT32, np.array([4], np.int32)),
                           ("FP_power", SDC.FLOAT32, np.array([7.5], np.float32)),
                           ("FP_confidence", SDC.UINT8, np.array([85], np.uint8))]:
        s = sd.create(name, typ, arr.shape)
        s[:] = arr
        s.endaccess()
    sd.end()
    sd = SD(str(f03), SDC.WRITE | SDC.CREATE)
    for name, typ, arr, sc in [("Latitude", SDC.FLOAT32, np.full((20, 30), 35.0, np.float32), None),
                               ("Longitude", SDC.FLOAT32, np.full((20, 30), 128.0, np.float32), None),
                               ("SensorZenith", SDC.INT16, np.full((20, 30), 2000, np.int16), 0.01),
                               ("SolarZenith", SDC.INT16, np.full((20, 30), 3000, np.int16), 0.01)]:
        s = sd.create(name, typ, arr.shape)
        s[:] = arr
        if sc:
            s.scale_factor = sc
        s.endaccess()
    sd.end()
    s = readers.read_modis(f14, f03)
    assert s.platform == "aqua" and s.cls[3, 4] == 4 and s.frp[3, 4] == 7.5 and s.cls[0, 0] == 3
    assert s.conf[3, 4] == pytest.approx(0.85) and s.conf[0, 0] == 0  # MODIS FP_confidence / 100
    assert s.vza[0, 0] == pytest.approx(20.0, abs=1e-3) and s.sza[0, 0] == pytest.approx(30.0, abs=1e-3)


# ---------------------------------------------------------------- review fixes
def test_upsample_scans_never_blends_across_scan_boundary():
    # coarse field: linear inside a scan of 16 lines, with a jump of 100 between scans
    a = (np.arange(48) + 100.0 * (np.arange(48) // 16))[:, None] * np.ones((1, 3))
    u = readers.upsample2x_scans(a, 16)
    assert u.shape == (96, 3)
    for s in range(3):  # exact inside each scan: fine line k sits at coarse coordinate (k-0.5)/2 within that scan
        k = np.arange(32)
        assert np.allclose(u[s * 32:(s + 1) * 32, 0], s * 16 + s * 100 + (k - 0.5) / 2)
    assert not np.isclose(readers.upsample2x(a, 0)[31, 0], u[31, 0])  # plain upsampling was off here (old bug)
    with pytest.raises(ValueError):
        readers.upsample2x_scans(np.zeros((17, 2)), 16)


def test_modis_reader_zero_fire_granule(tmp_path):
    hdf = pytest.importorskip("pyhdf.SD")
    SD, SDC = hdf.SD, hdf.SDC
    f14, f03 = tmp_path / "MYD14.A2026318.0105.061.x.hdf", tmp_path / "MYD03.A2026318.0105.061.x.hdf"
    sd = SD(str(f14), SDC.WRITE | SDC.CREATE)
    s = sd.create("fire mask", SDC.UINT8, (20, 30))
    s[:] = np.full((20, 30), 5, np.uint8)
    s.endaccess()
    for name, typ in [("FP_line", SDC.INT32), ("FP_sample", SDC.INT32), ("FP_power", SDC.FLOAT32)]:
        sd.create(name, typ, (0,)).endaccess()  # zero-length, as in a granule without fire pixels
    sd.end()
    sd = SD(str(f03), SDC.WRITE | SDC.CREATE)
    for name, typ, val in [("Latitude", SDC.FLOAT32, 35.0), ("Longitude", SDC.FLOAT32, 128.0),
                           ("SensorZenith", SDC.INT16, 2000), ("SolarZenith", SDC.INT16, 3000)]:
        s = sd.create(name, typ, (20, 30))
        s[:] = np.full((20, 30), val, np.float32 if typ == SDC.FLOAT32 else np.int16)
        s.endaccess()
    sd.end()
    s = readers.read_modis(f14, f03)  # used to raise 'SDreaddata failure'
    assert (s.cls == 3).all() and s.frp.sum() == 0


def test_viirs_reader_zero_fire_granule(tmp_path):
    nc = pytest.importorskip("netCDF4")
    f14, f03 = tmp_path / "VNP14IMG.A2026318.0112.002.x.nc", tmp_path / "VNP03MOD.A2026318.0112.021.x.nc"
    with nc.Dataset(f14, "w") as d:
        d.createDimension("y", 32), d.createDimension("x", 80), d.createDimension("fp", 0)
        d.createVariable("fire mask", "u1", ("y", "x"))[:] = np.full((32, 80), 5, np.uint8)
        for n, t in [("FP_line", "i4"), ("FP_sample", "i4"), ("FP_power", "f4")]:
            d.createVariable(n, t, ("fp",))
    with nc.Dataset(f03, "w") as d:
        g = d.createGroup("geolocation_data")
        g.createDimension("y", 16), g.createDimension("x", 40)
        for n, val in [("latitude", 30.0), ("longitude", 100.0), ("sensor_zenith", 20.0), ("solar_zenith", 30.0)]:
            g.createVariable(n, "f4", ("y", "x"))[:] = np.full((16, 40), val, np.float32)
    s = readers.read_viirs(f14, f03)
    assert s.shape == (32, 80) and (s.cls == 3).all() and s.frp.sum() == 0


# ---------------------------------------------------------------- thermal (L1B) inputs, schema v2
def _radiance(bt, nu):
    lam = 1e4 / nu
    return readers.PLANCK_C1 / (lam ** 5 * (np.exp(readers.PLANCK_C2 / (lam * bt)) - 1))


def test_planck_roundtrip():
    for nu in (2518.028, 908.0884):
        assert readers.planck_bt(_radiance(np.array([220.0, 300.0, 450.0]), nu), nu) == pytest.approx([220, 300, 450], abs=0.01)
    assert np.isnan(readers.planck_bt(np.array([0.0, -1.0]), 908.0)).all()


def test_modis_l1b_bt_fill_and_saturation_fallback():
    scales = np.full(16, 5e-3)
    offsets = np.full(16, 1000.0)
    ev = np.zeros((16, 2, 3), np.uint16)
    raw = lambda bt, nu, i: int(_radiance(bt, nu) / scales[i] + offsets[i])
    nu = readers.MODIS_NU
    ev[1], ev[2], ev[10] = raw(500, nu[21], 1), raw(300, nu[22], 2), raw(290, nu[31], 10)
    ev[2, 0, 0] = 65533  # band 22 flagged saturated -> use band 21
    ev[2, 0, 1] = raw(335, nu[22], 2)  # band 22 above the saturation guard -> band 21
    ev[10, 1, 2] = 65535  # fill at 11 um
    t4, t11, sat = readers.modis_l1b_to_bt(ev, scales, offsets)
    assert sat[0, 0] and sat[0, 1] and not sat[1, 2]
    assert t4[0, 0] == pytest.approx(500, abs=0.5) and t4[0, 1] == pytest.approx(500, abs=0.5)
    assert t4[1, 2] == pytest.approx(300, abs=0.5)
    assert np.isnan(t11[1, 2]) and t11[0, 0] == pytest.approx(290, abs=0.5)
    assert t4.dtype == np.float32


def test_viirs_bt_lut_flags():
    lut = np.full(65536, -999.9, np.float32)
    lut[:60000] = np.linspace(208, 360, 60000)
    raw = np.array([[0, 59999, 65532], [65533, 65535, 61000]], np.uint16)
    bt = readers.viirs_bt_from_lut(raw, lut)
    assert bt[0, 0] == pytest.approx(208) and bt[0, 1] == pytest.approx(360, abs=0.01)
    assert np.isnan(bt[0, 2]) and np.isnan(bt[1, 0]) and np.isnan(bt[1, 1]) and np.isnan(bt[1, 2])  # flags, fill LUT


def test_tile_thermal_layers(pair):
    m0, v0 = pair
    m = readers.attach_l1b(m0.__class__(**{**m0.__dict__}), np.full(m0.shape, 305.0, np.float32), np.full(m0.shape, 290.0, np.float32))
    v = readers.attach_l1b(v0.__class__(**{**v0.__dict__}), np.full(v0.shape, 310.0, np.float32), np.full(v0.shape, np.nan, np.float32))
    m.t4[110, 67] = 480.0  # a hot fire pixel
    t = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]))
    validate_tile(t)
    assert int(t["has_thermal"]) == 1 and int(t["schema_version"]) == 5
    obs = t["modis_pid"] >= 0
    pid = 110 * m.shape[1] + 67
    assert (t["modis_t4"][t["modis_pid"] == pid] == 480.0).all()
    assert (t["modis_t4"][obs & (t["modis_pid"] != pid)] == 305.0).all() and (t["modis_t4"][~obs] == 0).all()
    assert (t["viirs_t4"][t["viirs_pid"] >= 0] == 310.0).all()
    assert (t["viirs_t11"] == 0).all()  # NaN -> 0 (missing)
    # fire pixel is far hotter at 4 um than its neighbours; no L1B -> zeros and has_thermal 0
    t0 = build_tile(m0, v0, float(m0.lat[110, 67]), float(m0.lon[110, 67]))
    validate_tile(t0)
    assert int(t0["has_thermal"]) == 0 and (t0["modis_t4"] == 0).all()
    with pytest.raises(ValueError):
        readers.attach_l1b(m0, np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))


# ---------------------------------------------------------------- schema v3 keys
from nightrad.data.tiles import StaticSources, apply_sample_weights, region_of  # noqa: E402


def with_thermal(sw, t4=300.0, t11=285.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    sw = sw.__class__(**{**sw.__dict__})
    sw.t4 = (t4 + noise * rng.standard_normal(sw.shape)).astype(np.float32)
    sw.t11 = (t11 + noise * rng.standard_normal(sw.shape)).astype(np.float32)
    sw.sat = np.zeros(sw.shape, bool)
    return sw


def test_region_of():
    assert [region_of(*p) for p in [(37.5, 127.0), (48.8, 2.3), (-4.0, 23.0), (-3.0, -60.0), (31.0, -97.0),
                                    (-33.9, 151.2), (-2.0, 105.0), (60.0, 100.0), (24.0, 45.0)]] == \
        ["AS", "EU", "AF", "SA", "NA", "OC", "AS", "AS", "AS"]


def test_cloudadj_flags_pixels_next_to_cloud_only():
    m, v = make_swath("modis", T0), make_swath("viirs", T0 + 300)
    line = m.shape[0] // 5  # first non-cloud line (rows above are cloud)
    t = build_tile(m, v, float(m.lat[line, 67]), float(m.lon[line, 67]))
    validate_tile(t)
    adj = t["modis_cloudadj"] == 1
    assert adj.any() and (t["modis_cls"][adj] == 3).all()  # never cloud itself
    assert ((t["modis_pid"][adj] // m.shape[1]) == line).all()  # only the pixel row touching the cloud
    far = (t["modis_pid"] >= 0) & (t["modis_pid"] // m.shape[1] > line + 1)
    assert far.any() and not adj[far].any()


def test_static_flag_keeps_fire_class():
    m = make_swath("modis", T0, fires=[(110, 67, 5, 20.0)])
    v = make_swath("viirs", T0 + 300, fires=[(500, 320, 5, 3.0)])
    st = StaticSources([m.lat[110, 67]], [m.lon[110, 67]])
    t = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]), st)
    validate_tile(t)
    cells = t["modis_pid"] == 110 * m.shape[1] + 67
    assert t["modis_static"][cells].all() and (t["modis_cls"][cells] == 5).all()  # flagged, class kept
    assert 0 < t["modis_static"].sum() < 0.2 * t["modis_pid"].size
    assert t["viirs_static"].any()
    # VIIRS static cells lie within ~1 km (+ a pixel) of the point
    j, i = np.nonzero(t["viirs_static"])
    d_km = np.hypot(j - 47.5, i - 47.5) * 0.375
    assert d_km.max() < 2.0
    t0 = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]))
    assert not t0["modis_static"].any() and not t0["viirs_static"].any()


def test_missing_is_flagged_never_filled_and_sat_flags():
    m, v = with_thermal(make_swath("modis", T0)), with_thermal(make_swath("viirs", T0 + 300))
    m.t4[110, 67], m.t11[111, 67] = np.nan, np.nan  # two missing pixels, one in each channel
    m.sat[112, 67] = True
    v.t4[:] = np.nan  # everything missing in VIIRS T4 -> 0, flagged
    t = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]))
    validate_tile(t)
    for line in (110, 111):
        c = t["modis_pid"] == line * m.shape[1] + 67
        assert t["modis_t4_missing"][c].all()
        a4, a11 = (0.0, 285.0) if line == 110 else (300.0, 0.0)  # v5: the missing channel is 0 (never filled from neighbours), the other is copied
        assert (t["modis_t4"][c] == a4).all() and (t["modis_t11"][c] == a11).all()
        assert (t["modis_vclass"][c] == 5).all() and not t["modis_valid"][c].any()  # bad or missing
    assert not t["modis_t4_missing"][t["modis_pid"] == 112 * m.shape[1] + 67].any()
    assert t["modis_sat"][t["modis_pid"] == 112 * m.shape[1] + 67].all() and t["modis_sat"].sum() < 200
    assert (t["viirs_t4"] == 0).all() and t["viirs_t4_missing"][t["viirs_pid"] >= 0].all()


def test_background_zscores():
    m = with_thermal(make_swath("modis", T0, fires=[(110, 67, 5, 20.0)]), noise=2.0, seed=1)
    v = with_thermal(make_swath("viirs", T0 + 300, fires=[(500, 320, 5, 3.0)]), noise=2.0, seed=2)
    m.t4[110, 67] += 40.0  # fire pixel hot at 4 um, 11 um unchanged -> (T4 - T11) anomaly too
    t = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]))
    validate_tile(t)
    fire = t["modis_pid"] == 110 * m.shape[1] + 67
    clear = (t["modis_cls"] == 3) & (t["modis_bg_ok"] == 1) & ~fire
    assert t["modis_bg_ok"][fire].all()
    assert t["modis_t4_bgz"][fire].min() > 10 and t["modis_dt_bgz"][fire].min() > 5  # background excludes the fire itself
    assert abs(t["modis_t4_bgz"][clear].mean()) < 0.3 and 0.5 < t["modis_t4_bgz"][clear].std() < 1.5
    # a tile entirely inside the cloud: no valid background -> bg_ok 0 and z 0
    tc = build_tile(m, v, float(m.lat[10, 67]), float(m.lon[10, 67]))
    validate_tile(tc)
    assert not tc["modis_bg_ok"].any() and (tc["modis_t4_bgz"] == 0).all() and (tc["modis_dt_bgz"] == 0).all()


def test_conf_only_on_fire_cells():
    m = make_swath("modis", T0, fires=[(110, 67, 5, 20.0), (115, 70, 4, 8.0)])
    m.conf = np.zeros(m.shape, np.float32)
    m.conf[110, 67], m.conf[115, 70] = 0.85, 0.2
    v = make_swath("viirs", T0 + 300)
    t = build_tile(m, v, float(m.lat[110, 67]), float(m.lon[110, 67]))
    validate_tile(t)
    assert {round(float(x), 2) for x in np.unique(t["modis_conf"])} == {0.0, 0.2, 0.85}
    assert (t["modis_conf"][t["modis_cls"] < 4] == 0).all() and (t["viirs_conf"] == 0).all()


def _flat_swath(vza_a, vza_b):
    """30x30 flat 1 km grid; pixel (15,16) is moved onto pixel (15,15) (bow-tie style duplicate coverage)."""
    n = 30
    lat0, lon0 = 37.0, 127.0
    ii, jj = np.mgrid[0:n, 0:n]
    lat = lat0 + (ii - 15) / 111.2
    lon = lon0 + (jj - 15) / (111.2 * np.cos(np.deg2rad(lat0)))
    lat[15, 16], lon[15, 16] = lat[15, 15] + 1e-5, lon[15, 15] + 1e-5
    vza = np.full((n, n), 40.0, np.float32)
    vza[15, 15], vza[15, 16] = vza_a, vza_b
    cls = np.full((n, n), 3, np.uint8)
    cls[15, 15], cls[15, 16] = 5, 3  # A is fire, B is clear (or the reverse of who wins decides what we see)
    return Swath("modis", "aqua", "flat", lat.astype(np.float32), lon.astype(np.float32), cls, np.zeros((n, n), np.float32),
                 vza, np.full((n, n), 30, np.float32), T0, T0 + 300)


def test_modis_overlap_keeps_pixel_closest_to_nadir():
    v = make_swath("viirs", T0)
    for a, b in [(10.0, 20.0), (20.0, 10.0)]:
        s = _flat_swath(a, b)
        t = build_tile(s, v, 37.0, 127.0)
        A, B = 15 * 30 + 15, 15 * 30 + 16
        winner = A if a < b else B
        assert (t["modis_pid"][47:49, 47:49] == winner).all(), (a, b)  # the 4 cells at the doubly covered spot


def test_sample_weights_formula():
    mk = lambda fire, rnd: {"random_tile": np.uint8(rnd), "sample_weight": np.float32(1.0),
                            "modis_cls": np.full((2, 2), 5 if fire else 3, np.uint8),
                            "viirs_cls": np.full((2, 2), 3, np.uint8)}
    tiles = [mk(True, 0) for _ in range(10)] + [mk(True, 1)] + [mk(False, 1) for _ in range(4)]
    apply_sample_weights(tiles)
    q, n = (1 + 0.5) / (5 + 1), 15
    wf = [float(t["sample_weight"]) for t in tiles if t["modis_cls"][0, 0] == 5]
    wn = [float(t["sample_weight"]) for t in tiles if t["modis_cls"][0, 0] == 3]
    assert wf[0] == pytest.approx(q * n / 11) and wn[0] == pytest.approx((1 - q) * n / 4)
    assert sum(wf) / (sum(wf) + sum(wn)) == pytest.approx(q) and np.mean(wf + wn) == pytest.approx(1.0)
    only_fire = [mk(True, 0) for _ in range(3)]
    apply_sample_weights(only_fire)  # no random tiles -> weights stay 1
    assert all(float(t["sample_weight"]) == 1.0 for t in only_fire)
