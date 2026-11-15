"""Truthful resampling (schema v5): a cell is a copy of ONE native pixel whose footprint covers the cell
centre; bow-tie-deleted pixels are skipped; masked ground stays masked; nothing is interpolated or filled."""
import numpy as np
import pytest
import torch

from nightrad.data import readers, tiles
from nightrad.data.readers import (VC_BAD, VC_CLEAR, VC_CLOUD, VC_GLINT, VC_NOT_COVERED, VC_OFF_SWATH, VC_WATER, Swath, modis_l1b_full,
                                   planck_bt, planck_rad, viirs_radiance)
from nightrad.data.tiles import build_tile, validate_tile
from nightrad.data.tiles_gpu import GSwath, build_tiles_gpu
from test_tiles import T0, make_swath, with_thermal

NL, S, NSCAN = 16, 120, 40  # lines per scan, samples, scans of the synthetic bow-tie swath
LAT0, LON0 = 37.0, 127.0


_K = np.abs(np.arange(S) - (S - 1) / 2) / ((S - 1) / 2)
SIZES = (0.4 + 0.4 * _K ** 2, 0.4 * (1 + _K ** 2))  # ground truth along-scan / along-track size per sample (km), not used by the builder


def bowtie_swath(seed=0, cloud_rows=None) -> Swath:
    """VIIRS-like swath on a flat earth: along-scan size 0.4 -> 0.8 km and along-track size 0.4 -> 0.8 km from nadir to edge, 16 lines
    per scan at a constant 6.4 km pitch, so scans overlap ~2x at the edge. Pixels whose footprint lies wholly in the neighbouring
    scan's zone are bow-tie deleted (raw 1, no measurement); every other pixel has a unique random T4/T11/radiance."""
    rng = np.random.default_rng(seed)
    k = np.abs(np.arange(S) - (S - 1) / 2) / ((S - 1) / 2)
    size_as, size_at = 0.4 + 0.4 * k ** 2, 0.4 * (1 + k ** 2)
    x_km = np.cumsum(size_as)
    x_km -= x_km[S // 2]
    P = NL * 0.4
    L = NL * NSCAN
    line = np.arange(L)
    scan, row = line // NL, line % NL
    y_km = scan[:, None] * P + (row[:, None] + 0.5 - NL / 2) * size_at[None]  # centre of each pixel along track
    dy = np.abs(y_km - scan[:, None] * P)
    deleted = dy - size_at[None] / 2 >= P / 2  # footprint entirely inside the neighbouring scan's zone
    lat = LAT0 + y_km / 111.2
    lon = LON0 + x_km[None] / (111.2 * np.cos(np.deg2rad(LAT0))) + 0 * lat
    raw = np.full((L, S), 5, np.uint8)
    raw[deleted] = readers.RAW_VIIRS_BOWTIE
    if cloud_rows is not None:
        raw[cloud_rows] = 4
    cls = readers.map_fire_mask(raw)
    sw = Swath("viirs", "snpp", "bowtie", lat.astype(np.float32), lon.astype(np.float32), cls, np.zeros((L, S), np.float32),
               np.rad2deg(np.arctan(np.abs(x_km) / 824.0))[None].repeat(L, 0).astype(np.float32), np.full((L, S), 40, np.float32),
               T0, T0 + 360, raw=raw, deleted=deleted)
    t4 = rng.uniform(280, 320, (L, S)).astype(np.float32)
    t11 = rng.uniform(270, 300, (L, S)).astype(np.float32)
    readers.attach_l1b(sw, np.where(deleted, np.nan, t4).astype(np.float32), np.where(deleted, np.nan, t11).astype(np.float32),
                       np.zeros((L, S), bool), rad4=np.where(deleted, np.nan, t4 / 100).astype(np.float32),
                       rad11=np.where(deleted, np.nan, t11 / 100).astype(np.float32), band=np.zeros((L, S), np.uint8))
    return sw


def _centre_tile(sw, scan=20, sample=S // 2):
    li, si = scan * NL + NL // 2, sample
    return build_tile(with_thermal(make_swath("modis", T0)), sw, float(sw.lat[li, si]), float(sw.lon[li, si])), (li, si)


def _covering_truth(t, sw, key="viirs"):
    """Independent check: the selected pixel's axis-aligned rectangle (synthetic grid is axis aligned) contains the cell centre."""
    tr = tiles._proj(float(t["tile_lat"]), float(t["tile_lon"]))
    pid = t[f"{key}_pid"]
    Hh, Ww = pid.shape
    jj, ii = np.meshgrid(np.arange(Ww), np.arange(Hh))
    cx, cy = (jj - (Ww - 1) / 2) * tiles.CELL_M, ((Hh - 1) / 2 - ii) * tiles.CELL_M
    m = pid >= 0
    li, si = pid[m] // sw.shape[1], pid[m] % sw.shape[1]
    px, py = tr.transform(sw.lon[li, si].astype(np.float64), sw.lat[li, si].astype(np.float64))
    size_as, size_at = SIZES
    return (np.abs(cx[m] - px) <= 0.5 * 1000 * size_as[si] * 1.06) & (np.abs(cy[m] - py) <= 0.5 * 1000 * size_at[si] * 1.06)


@pytest.mark.parametrize("sample", [S // 2, S // 2 + 22, 15])  # nadir, mid, near the swath edge where bow-tie overlap is largest
def test_bowtie_swath_no_stripes_values_are_native_and_footprints_cover(sample):
    sw = bowtie_swath()
    assert sw.deleted.mean() > 0.1  # the synthetic swath really has deleted duplicates
    t, _ = _centre_tile(sw, sample=sample)
    validate_tile(t)
    pid, vc = t["viirs_pid"], t["viirs_vclass"]
    inside = pid >= 0
    if sample > 20:  # tile entirely inside the swath: no stripe gaps at all
        assert inside.all() and (vc == VC_CLEAR).all() and t["viirs_valid"].all()
    assert (t["viirs_cls"][inside] != 0).all()
    assert not sw.deleted.ravel()[pid[inside]].any(), "a bow-tie deleted pixel was selected"
    # every filled cell is a bit-identical copy of its native pixel (no blending, no filling)
    for k_t, k_s in (("t4", "t4"), ("t11", "t11"), ("rad4", "rad4"), ("rad11", "rad11")):
        np.testing.assert_array_equal(t[f"viirs_{k_t}"][inside], getattr(sw, k_s).ravel()[pid[inside]])
    assert set(np.unique(t["viirs_t4"][inside])) <= set(np.unique(sw.t4[np.isfinite(sw.t4)]))
    # no cell is taken from a pixel that does not cover it
    assert _covering_truth(t, sw).all()
    # bow-tie overlap exists and is resolved: some cells are covered by pixels of two scans (duplicates), one measurement is kept
    if sample == 15:
        assert (t["viirs_ncover"] >= 2).any()


def test_old_nearest_pixel_rule_left_stripes_on_the_same_swath():
    """The v4 behaviour (nearest centre inside the half diagonal, deleted pixels selectable) made class-0 stripes; the new rule does not."""
    sw = bowtie_swath()
    t, (li, si) = _centre_tile(sw, sample=S // 2 + 22)
    # emulate the old rule on the tile: nearest pixel centre, deleted ones included
    from scipy.spatial import cKDTree

    tr = tiles._proj(float(t["tile_lat"]), float(t["tile_lon"]))
    x, y = tr.transform(sw.lon.astype(np.float64).ravel(), sw.lat.astype(np.float64).ravel())
    near = np.hypot(x - 0, y - 0) < 40000
    idx = np.flatnonzero(near)
    tree = cKDTree(np.stack([x[idx], y[idx]], 1))
    jj, ii = np.meshgrid(np.arange(96), np.arange(96))
    q = np.stack([((jj - 47.5) * 375.).ravel(), ((47.5 - ii) * 375.).ravel()], 1)
    _, nn = tree.query(q)
    old_deleted = sw.deleted.ravel()[idx[nn]]
    assert old_deleted.mean() > 0.01  # old rule would have taken deleted pixels (class 0 stripes) for these cells
    assert not sw.deleted.ravel()[t["viirs_pid"][t["viirs_pid"] >= 0]].any() and t["viirs_valid"][t["viirs_pid"] >= 0].all()


def test_cloudy_covering_pixel_masks_the_cell_even_with_clear_pixel_two_cells_away():
    v = with_thermal(make_swath("viirs", T0))
    v.cls[494:507, 314:327] = 2  # 13x13 cloudy pixels (~4.9 km); clear pixels start well outside
    v = Swath(**{**v.__dict__, "_vc": None})
    t = build_tile(with_thermal(make_swath("modis", T0)), v, float(v.lat[500, 320]), float(v.lon[500, 320]))
    validate_tile(t)
    c = slice(44, 52)  # the 8x8 centre cells (3 km) are covered by cloudy pixels only; clear pixels are >= 2 cells away
    assert (t["viirs_vclass"][c, c] == VC_CLOUD).all() and not t["viirs_valid"][c, c].any()
    pid = t["viirs_pid"][c, c]
    assert ((pid // v.shape[1] >= 494) & (pid // v.shape[1] <= 506)).all()  # every one of them is a copy of a cloudy pixel
    np.testing.assert_array_equal(t["viirs_t4"][c, c], v.t4.ravel()[pid])  # the cloud-top measurement itself is kept
    far = t["viirs_valid"] == 1
    assert far.any() and (t["viirs_vclass"][far] == VC_CLEAR).all()


def test_hole_stays_not_covered_never_filled_from_neighbours():
    v = with_thermal(make_swath("viirs", T0))
    blk = (slice(494, 506), slice(314, 326))
    v.cls[blk] = 0
    v = Swath(**{**v.__dict__, "_vc": None, "deleted": np.zeros(v.shape, bool), "raw": np.where(v.cls == 0, 1, 5).astype(np.uint8)})
    v.deleted[blk] = True
    t = build_tile(with_thermal(make_swath("modis", T0)), v, float(v.lat[500, 320]), float(v.lon[500, 320]))
    validate_tile(t)
    nc = t["viirs_vclass"] == VC_NOT_COVERED
    assert nc.sum() > 20 and (t["viirs_pid"][nc] == -1).all() and not t["viirs_valid"][nc].any()
    assert (t["viirs_cls"][nc] == 0).all() and (t["viirs_t4"][nc] == 0).all() and (t["viirs_raw"][nc] == 0).all()
    ok = t["viirs_pid"] >= 0
    pid = t["viirs_pid"][ok]
    assert not v.deleted.ravel()[pid].any()


def test_raw_codes_are_kept_and_reasons_are_separate():
    sw = bowtie_swath()
    raw = sw.raw.copy()
    li = 20 * NL + 8
    for off, code in enumerate((0, 2, 3, 4, 5, 6, 8)):
        raw[li, 40 + 3 * off:43 + 3 * off] = code
    sw2 = Swath(**{**sw.__dict__, "raw": raw, "cls": readers.map_fire_mask(raw), "_vc": None})
    vc = sw2.pixel_vclass()
    expect = {0: VC_BAD, 2: VC_GLINT, 3: VC_WATER, 4: VC_CLOUD, 5: VC_CLEAR, 6: VC_BAD, 8: VC_CLEAR}
    for off, code in enumerate((0, 2, 3, 4, 5, 6, 8)):
        assert (vc[li, 40 + 3 * off:43 + 3 * off] == expect[code]).all(), code
    m = Swath(**{**sw.__dict__, "sensor": "modis", "raw": raw, "_vc": None})
    assert (m.pixel_vclass()[li, 40:43] == VC_BAD).all() and (m.pixel_vclass()[li, 43:46] == VC_BAD).all()  # MODIS raw 0 / 2: not processed
    t, (_, _) = _centre_tile(sw2)
    for k, code in enumerate((0, 2, 3, 4, 5, 6, 8)):
        cells = (t["viirs_pid"] // sw.shape[1] == li) & (t["viirs_raw"] == code) & (t["viirs_pid"] >= 0)
        if cells.any():
            assert (t["viirs_vclass"][cells] == expect[code]).all()


def test_missing_thermal_is_bad_not_filled():
    sw = bowtie_swath()
    li, si = 20 * NL + 8, S // 2
    t4 = sw.t4.copy()
    t4[li, si] = np.nan
    sw2 = readers.attach_l1b(Swath(**{**sw.__dict__}), t4, sw.t11, sw.sat, rad4=sw.rad4, rad11=sw.rad11, band=sw.t4_band)
    t, _ = _centre_tile(sw2)
    c = t["viirs_pid"] == li * S + si
    if c.any():
        assert (t["viirs_vclass"][c] == VC_BAD).all() and (t["viirs_t4"][c] == 0).all() and (t["viirs_rad4"][c] == 0).all()
        assert t["viirs_t4_missing"][c].all()


def test_modis_nadir_preference_among_valid_and_valid_first():
    """MODIS keeps the closest-to-nadir pixel among valid covering pixels; a valid covering pixel beats a nearer cloudy one."""
    from test_tiles import _flat_swath

    v = make_swath("viirs", T0)
    s = _flat_swath(10.0, 20.0)  # pixel A (15,15) vza 10 vs B (15,16) vza 20 over the doubly covered spot
    t = build_tile(s, v, 37.0, 127.0)
    A, B = 15 * 30 + 15, 15 * 30 + 16
    assert (t["modis_pid"][47:49, 47:49] == A).all()
    s2 = Swath(**{**s.__dict__, "cls": s.cls.copy(), "_vc": None})
    s2.cls[15, 15] = 2  # nadir-most pixel is cloudy: the valid one must win
    t2 = build_tile(s2, v, 37.0, 127.0)
    assert (t2["modis_pid"][47:49, 47:49] == B).all() and (t2["modis_vclass"][47:49, 47:49] == VC_CLEAR).all()


def test_cpu_and_torch_builders_agree_on_bowtie_swath():
    sw = bowtie_swath(seed=3, cloud_rows=(slice(20 * NL + 1, 20 * NL + 3), slice(30, 60)))
    m = with_thermal(make_swath("modis", T0))
    centres = [(float(sw.lat[li, si]), float(sw.lon[li, si])) for li, si in ((20 * NL + 8, 60), (20 * NL + 8, 38), (20 * NL + 2, 10),
                                                                              (25 * NL + 12, 100))]
    g = build_tiles_gpu(GSwath(m, "cpu"), GSwath(sw, "cpu"), centres)
    for (la, lo), tg in zip(centres, g):
        tc = build_tile(m, sw, la, lo)
        assert set(tc) == set(tg)
        for k in tc:
            a, b = np.asarray(tc[k]), np.asarray(tg[k])
            assert a.dtype == b.dtype and a.shape == b.shape, k
            if a.dtype.kind == "f":
                np.testing.assert_allclose(a, b, atol=1e-3, rtol=0, err_msg=k)
            else:
                np.testing.assert_array_equal(a, b, err_msg=k)
        validate_tile(tg)


def test_radiance_and_flag_helpers():
    nu = readers.MODIS_NU[22]
    rad = np.array([0.5, 1.2, 3.0], np.float64)
    assert np.allclose(planck_rad(planck_bt(rad, nu), nu), rad, rtol=1e-5)
    ev = np.zeros((16, 1, 3), np.uint16)
    scales, offs = np.full(16, 1e-4), np.zeros(16)
    ev[2, 0] = (3000, 65535, 65533)  # band 22: valid, fill (not processed), flagged saturated
    ev[1, 0] = (3100, 2900, 2900)  # band 21
    ev[10, 0] = (2000, 2000, 2000)  # band 31
    d = modis_l1b_full(ev, scales, offs)
    assert d["band"][0, 0] == 22 and np.isclose(d["rad4"][0, 0], 0.3)
    assert d["band"][0, 1] == 21 and np.isclose(d["rad4"][0, 1], 0.29) and not d["sat"][0, 1]  # band 22 invalid -> band 21 used, NOT a saturation
    assert d["band"][0, 2] == 21 and d["sat"][0, 2]  # band 22 flagged saturated (65533) -> band 21 used and recorded as saturated
    assert np.isclose(d["rad11"][0, 0], 0.2) and np.isfinite(d["t4"]).all()
    r = viirs_radiance(np.array([100, 65533, 65535], np.uint16), 6.1e-5, 0.0017)
    assert np.isclose(r[0], 100 * 6.1e-5 + 0.0017, rtol=1e-5) and np.isnan(r[1:]).all()
