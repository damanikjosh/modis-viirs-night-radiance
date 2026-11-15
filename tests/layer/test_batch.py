"""Batch path: ownership grid / tile rules (no network, no GPU), catalogue pairing."""
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np

from nightrad.inference import batch as B
from nightrad.inference.tilegeo import laea_forward, tile_cell_lonlat


def _cell_ij(lat, lon, lat0, lon0):
    x, y = laea_forward(lat, lon, lat0, lon0)
    return np.rint(47.5 - y / 375.0).astype(int), np.rint(x / 375.0 + 47.5).astype(int)


def test_stride_and_overlap():
    for m in (0, 8, 12, 16):
        g = B.Grid(m)
        assert g.k == 96 - 2 * m and abs(g.core_m - (96 - 2 * m) * 375.0) < 1e-9


def test_every_cell_covered_exactly_once_and_inside_core():
    """Every lon/lat cell centre of a dense scene patch is owned by exactly one tile (r, c) of the lattice, and that tile holds it >= margin-1 cells inside its border."""
    for m in (8, 12, 16):
        for shift in ((0.0, 0.0), (0.5, 0.5)):
            g = B.Grid(m, shift)
            rng = np.random.default_rng(m)
            lat = rng.uniform(-10.5, 5.5, 4000); lon = rng.uniform(95, 141, 4000)
            c, r = g.col_of(lat, lon)
            worst = 99
            for la, lo, cc, rr in zip(lat, lon, c, r):
                lat0, lon0 = g.tile_centre(int(rr), int(cc))
                i, j = _cell_ij(la, lo, lat0, lon0)
                worst = min(worst, i, j, 95 - i, 95 - j)
            assert worst >= m - 1, (m, shift, worst)  # <= 1 cell of lattice/projection rounding


def test_owned_mask_keeps_a_border_hotspot_once():
    """A synthetic hot spot exactly on a former (no-margin) tile border is predicted by every tile whose 96x96 window holds it; owned_mask keeps it in exactly one."""
    old, new = B.Grid(0), B.Grid(12)
    lat = np.array([-19 * old.dlat]); lon = np.array([old.tile_centre(-20, 3000)[1]])  # on the row border of two old tiles
    c, r = new.col_of(lat, lon)
    built = {(int(r[0]) + dr, int(c[0]) + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    keepers = sum(int(B.owned_mask(new, int(r[0]) + dr, int(c[0]) + dc, lat, lon, built)[0].sum()) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    assert keepers == 1
    # and under no-margin the same point lies on the border of its owner tile (distance to the tile edge < 1 cell): this is what v1.2 removes
    lat0, lon0 = old.tile_centre(*(int(x) for x in (old.col_of(lat, lon)[1][0], old.col_of(lat, lon)[0][0])))
    i, j = _cell_ij(lat, lon, lat0, lon0)
    assert min(i[0], 95 - i[0]) <= 1
    lat0n, lon0n = new.tile_centre(int(r[0]), int(c[0]))
    i, j = _cell_ij(lat, lon, lat0n, lon0n)
    assert min(i[0], j[0], 95 - i[0], 95 - j[0]) >= 11


def test_orphan_flags_not_kept_with_margin_but_counted():
    g = B.Grid(12)
    lat = np.array([1.0]); lon = np.array([100.0])
    c, r = g.col_of(lat, lon)
    k, n = B.owned_mask(g, int(r[0]) + 1, int(c[0]), lat, lon, set())  # a neighbour predicted it, the owner core was never built
    assert not k.any() and n == 1


def test_select_cores_is_superset_of_cells_within_radius():
    g = B.Grid(12)
    la, lo = -3.0, 114.0
    fire = B._unit_xyz(np.array([la]), np.array([lo]))
    c0, r0 = g.col_of(np.array([la]), np.array([lo]))
    cells = [(int(r0[0]) + dr, int(c0[0]) + dc) for dr in range(-4, 5) for dc in range(-4, 5)]
    sel = set(B.select_cores(cells, g, fire, 12 * 0.375))
    rng = np.random.default_rng(1)
    for _ in range(300):  # any point within 4.5 km of the fire pixel must have its owner core selected
        ang, rr = rng.uniform(0, 6.283), rng.uniform(0, 4.5)
        lat = la + rr * np.sin(ang) / 111.195; lon = lo + rr * np.cos(ang) / (111.195 * np.cos(np.deg2rad(la)))
        c, r = g.col_of(np.array([lat]), np.array([lon]))
        assert (int(r[0]), int(c[0])) in sel
    assert B.select_cores(cells, g, None, 4.5) == []


def test_tile_rules_are_parameters():
    assert set(B.TILE_RULES) == {"all_land", "modis_firelevel", "modis_firelevel_nbr8"} and B.TILE_RULES["all_land"] is None


def test_select_cores_empty_cells_does_not_raise():
    g = B.Grid(12)
    assert B.select_cores([], g, B._unit_xyz(np.array([1.0]), np.array([100.0])), 4.5) == []  # scene with fire pixels but no land core inside the bbox (v1.2 first-hour failure)




def test_zero_margin_grid_keeps_orphans_in_neighbour():
    g = B.Grid(0)
    lat = np.array([1.0, 1.0]); lon = np.array([100.0, 100.4])
    c, r = g.col_of(lat, lon)
    own = {(int(r[0]), int(c[0]))}
    k, n = B.owned_mask(g, int(r[0]), int(c[0]), lat, lon, own)
    assert n == 0 and k[0]
    assert k[1]  # second point's owner core was not built -> margin 0 keeps it in the neighbouring tile


def test_grid_cell_roundtrip():
    g = B.Grid(0)
    for lat, lon in [(-3.3, 114.2), (0.4, 101.7), (5.9, 95.4), (-8.5, 116.0)]:
        r = g.row_of(lat)
        c, r2 = g.col_of(lat, lon)
        assert r == r2
        la, lo = g.tile_centre(int(r), int(c))
        assert abs(la - lat) <= g.dlat / 2 + 1e-9
        assert abs(lo - lon) <= g.dlon_of_row(r) / 2 + 1e-9
        assert tuple(int(x) for x in (g.row_of(la), g.col_of(la, lo)[0])) == (int(r), int(c))
    la, lo = tile_cell_lonlat(*g.tile_centre(-30, 3000))
    assert la.shape == (96, 96) and abs(g.core_m - 36000.0) < 1e-9


def test_land_tiles_uses_only_land_inside_bbox():
    lat = np.array([[1.0, 1.0], [1.0, 1.0]], np.float32); lon = np.array([[100.0, 100.01], [130.0, 100.0]], np.float32)
    sw = SimpleNamespace(lat=lat, lon=lon, landsea=np.array([[1, 0], [1, 7]], np.uint8))
    g = B.Grid(0)
    assert len(B.land_tiles(sw, (94.5, -11.5, 141.5, 6.5), g)) == 2  # (1.0, 100.0) and (1.0, 130.0); water pixels are skipped
    assert B.land_tiles(sw, (0, 0, 10, 10), g) == []


def test_each_location_has_one_owner_tile():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-10, 6, 500), rng.uniform(95, 141, 500)
    for m in (0, 12):
        g = B.Grid(m)
        c, r = g.col_of(lat, lon)
        for la, lo, cc, rr in zip(lat, lon, c, r):
            cl, cr = g.col_of(la, lo)
            assert (int(cr), int(cl)) == (int(rr), int(cc))


def test_catalogue_pairing(monkeypatch):
    spec = importlib.util.spec_from_file_location("cat", pathlib.Path(__file__).parents[2] / "scripts/layer/catalogue.py")
    cat = importlib.util.module_from_spec(spec); spec.loader.exec_module(cat)

    def entry(short, hhmm):
        n = f"{short}.A2006253.{hhmm}.061.x.hdf"
        return {"links": [{"href": f"https://data.laads/{n}"}], "granule_size": 10, "time_start": "2006-09-10T16:30:00Z"}
    monkeypatch.setattr(cat, "_query", lambda short, *a, **k: [entry(short, "1630"), entry(short, "1635")])
    r = cat.catalogue("0,0,1,1", "2006-09-10", "2006-09-12")
    assert [x["id"] for x in r] == ["A2006253.1630", "A2006253.1635"] and r[0]["geo"] and r[0]["fm"]
    assert [x["id"] for x in cat.catalogue("0,0,1,1", "2006-09-10", "2006-09-12", ids={"A2006253.1635"})] == ["A2006253.1635"]
