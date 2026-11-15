"""Equivalence of the torch tile builder (device 'cpu' here; same code runs on CUDA) with the CPU reference."""
import numpy as np
import pytest
import torch

from nightrad.data import readers
from nightrad.data.readers_gpu import ViirsRaw, _up_geo, _up2d, viirs_gswath
from nightrad.data.tiles import StaticSources, build_pair_tiles, build_tile, validate_tile
from nightrad.data.tiles_gpu import GSwath, build_pair_tiles_gpu, build_tiles_gpu
from test_tiles import T0, make_swath, with_thermal

F_TOL = 1e-3  # float layers (scan angle, frp: float32 libm differences ~1e-5)


def assert_tiles_equal(a: dict, b: dict):
    assert set(a) == set(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape, k
        if x.dtype.kind in "US":
            assert x == y, k
        elif x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, atol=F_TOL, rtol=0, err_msg=k)
        else:
            np.testing.assert_array_equal(x, y, err_msg=k)


@pytest.fixture(scope="module")
def swaths():
    m = with_thermal(make_swath("modis", T0, fires=[(110, 67, 5, 20.0), (115, 70, 4, 8.0)]), noise=2.0, seed=1)
    v = with_thermal(make_swath("viirs", T0 + 600, fires=[(500, 320, 5, 3.0), (503, 322, 6, 9.0), (520, 330, 4, 1.0)]),
                     noise=2.0, seed=2)
    m.t4[110, 67], m.t11[111, 67], m.sat[112, 67] = np.nan, np.nan, True  # missing + saturated pixels
    v.t4[400:410, 300:310] = np.nan
    v.cls[505:510, 315:325] = 2  # cloud blob next to the fires -> cloudadj
    return m, v


def test_tiles_match_cpu_reference_incl_static_and_edges(swaths):
    m, v = swaths
    st = StaticSources([m.lat[110, 67], v.lat[520, 330]], [m.lon[110, 67], v.lon[520, 330]])
    # tile centres: on fires, on the swath edge columns (bow-tie / ragged window), and off the swath
    cents = [(float(m.lat[110, 67]), float(m.lon[110, 67])), (float(v.lat[503, 322]), float(v.lon[503, 322])),
             (float(m.lat[150, 8]), float(m.lon[150, 8])), (float(m.lat[120, 130]), float(m.lon[120, 130])),
             (float(m.lat[110, 67]) + 1.5, float(m.lon[110, 67]))]
    gm, gv = GSwath(m, "cpu"), GSwath(v, "cpu")
    got = build_tiles_gpu(gm, gv, cents, st, random_flags=[False, True, False, False, False], radius_km={"viirs": 80.0})
    for c, g, rnd in zip(cents, got, [False, True, False, False, False]):
        ref = build_tile(m, v, *c, st, random_tile=rnd)
        validate_tile(g)
        assert_tiles_equal(ref, g)
    assert got[0]["modis_static"].any() and got[0]["modis_t4_missing"].any() and got[1]["viirs_cloudadj"].any()
    assert got[0]["modis_sat"].any() and (got[4]["modis_pid"] < 0).all()  # tile off the swath: nothing assigned


def test_batch_size_does_not_change_tiles(swaths):
    m, v = swaths
    c = [(float(m.lat[110 + k, 67]), float(m.lon[110 + k, 67])) for k in (0, 3, 9)]
    gm, gv = GSwath(m, "cpu"), GSwath(v, "cpu")
    one = [build_tiles_gpu(gm, gv, [x])[0] for x in c]
    for a, b in zip(one, build_tiles_gpu(gm, gv, c)):
        assert_tiles_equal(a, b)


def test_pair_selection_matches_cpu(swaths):
    m, v = swaths
    ref = build_pair_tiles(m, v, max_fire_tiles=6, n_random=4, seed=3)
    got = build_pair_tiles_gpu(m, v, device="cpu", batch=3, max_fire_tiles=6, n_random=4, seed=3)
    assert len(ref) == len(got) > 0
    for a, b in zip(ref, got):
        assert_tiles_equal(a, b)  # same centres, same order, same weights


def test_no_thermal_swaths():
    m, v = make_swath("modis", T0, fires=[(110, 67, 5, 20.0)]), make_swath("viirs", T0 + 300, fires=[(500, 320, 5, 3.0)])
    c = (float(m.lat[110, 67]), float(m.lon[110, 67]))
    assert_tiles_equal(build_tile(m, v, *c), build_tiles_gpu(GSwath(m, "cpu"), GSwath(v, "cpu"), [c])[0])


def _viirs_raw(seed=0, L=64, S=96):
    rng = np.random.default_rng(seed)
    la = (30 + 0.01 * np.arange(L // 2)[:, None] + 0 * np.arange(S // 2)[None]).astype(np.float32)
    lo = (127 + 0.012 * np.arange(S // 2)[None] + 0 * np.arange(L // 2)[:, None]).astype(np.float32)
    la[3, 4] = np.nan
    lut = np.linspace(200, 360, 65536).astype(np.float32)
    lut[0] = -999.9
    mask = rng.choice([0, 3, 5, 7, 8, 9], size=(L, S)).astype(np.uint8)
    i4 = rng.integers(0, 65536, (L, S)).astype(np.uint16)
    i4[0, :5] = [65535, 65532, 65533, 65528, 0]
    return ViirsRaw("VNP14IMG.A2019246.1242.test.nc", "snpp", T0, T0 + 360, mask,
                    np.array([2, 5]), np.array([3, 7]), np.array([4.0, 9.0], np.float32), la, lo,
                    (la * 0 + 10).astype(np.float32), (la * 0 + 50).astype(np.float32), i4, i4[::-1].copy(), lut, lut)


def test_viirs_gpu_reader_matches_cpu_readers():
    r = _viirs_raw()
    g = viirs_gswath(r, "cpu")
    lat, lon = readers.upsample2x_geo(r.lat, r.lon)
    np.testing.assert_array_equal(g.lat.numpy(), lat)
    np.testing.assert_array_equal(g.lon.numpy(), lon)
    np.testing.assert_array_equal(g.vza.numpy(), readers.upsample2x_scalar(r.vza))
    np.testing.assert_array_equal(g.cls.numpy(), readers.map_fire_mask(r.mask))
    assert g.frp[2, 3] == 4.0 and g.frp[5, 7] == 9.0
    bt = readers.viirs_bt_from_lut(r.i4, r.lut4)
    np.testing.assert_array_equal(g.t4.numpy(), bt)
    top = np.nanmax(np.where(r.lut4 > 0, r.lut4, np.nan))
    np.testing.assert_array_equal(g.sat.numpy(), bt >= top - 1e-3)
    assert np.isnan(g.t4[0, :5].numpy()).all()  # flags + LUT fill -> NaN
