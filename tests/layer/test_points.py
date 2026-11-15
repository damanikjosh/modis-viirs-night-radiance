import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import aggregate as agg
import points as pts

T0 = 1154453000.0


def cells(gran, rad, lonlat, exc, vza=20.0, utc=1.1e9, firelevel=1):
    return pd.DataFrame({"lon": [c[0] for c in lonlat], "lat": [c[1] for c in lonlat], "utc": utc, "granule": gran, "region": "test", "vza": vza, "excess4_rad": exc, "modis4_rad": rad,
                         "modis_T4_K": 300.0, "pred_T4_K": 320.0, "pred_dT_K": 20.0, "own_pixel_firelevel": firelevel})


def test_k_rule_round_half_up_and_floor_one():
    assert list(pts.k_of([7.11, 18.0, 4.0, 1.0, 2.0, 30.0])) == [2, 5, 1, 1, 1, 8]     # 1.78->2, 4.5->5 (half up), 1.0->1, 0.25->1 (min 1), 0.5->1, 7.5->8


def test_geometric_N_is_footprint_area_over_cell_area():
    N0 = pts.n_cells_geometric([0.0])[0]; assert abs(N0 - 1 / 0.140625) < 1e-6          # nadir 1 km x 1 km
    Ne = pts.n_cells_geometric([60.0])[0]; assert Ne > 25                               # near the scan edge the pixel is several km across: tens of cells
    v = pts.n_cells_geometric(np.linspace(0, 65, 30)); assert np.all(np.diff(v) > 0)


def test_rotation_identity_distance_and_direction():
    lon, lat = pts.rotate_about(np.array([100.0]), np.array([0.0]), np.array([100.0]), np.array([0.0]), -12.0); assert lon[0] == 100.0 and lat[0] == 0.0   # no offset -> centroid
    kx = 111.32
    x, y = pts.rotate_about(np.array([100.0 + 1.0 / kx]), np.array([0.0]), np.array([100.0]), np.array([0.0]), 90.0)  # 1 km east turned +90 deg = 1 km north
    assert abs(x[0] - 100.0) < 1e-9 and abs(y[0] - 1.0 / 110.57) < 1e-9
    x, y = pts.rotate_about(np.array([100.0 + 1.0 / kx]), np.array([0.0]), np.array([100.0]), np.array([0.0]), -12.0)
    dx, dy = (x[0] - 100.0) * kx, y[0] * 110.57; assert abs(np.hypot(dx, dy) - 1.0) < 1e-9 and abs(np.degrees(np.arctan2(dy, dx)) + 12.0) < 1e-6   # same distance, turned -12 deg
    a, b = pts.rotate_about(x, y, np.array([100.0]), np.array([0.0]), 12.0); assert abs(a[0] - (100.0 + 1.0 / kx)) < 1e-9 and abs(b[0]) < 1e-9


def test_track_angle_about_12_deg_everywhere():
    a = pts.track_angle_deg(np.linspace(-11.5, 28.6, 50)); assert 11.5 < a.min() and a.max() < 12.8 and abs(pts.track_angle_deg(0.0) - 12.07) < 0.15


def test_footprint_used_is_viirs_not_modis():
    s, t, c = pts.footprint_km([0.0, 30.0, 60.0]); assert abs(s[0] - 0.375) < 1e-9 and s[2] < 1.0 and t[2] < 1.0       # never the 4.8 km MODIS pixel


def test_grouping_keeps_two_cell_pixel_and_separates_equal_radiance_pixels():
    far = 0.30                                                                              # 33 km apart, same radiance value
    df = pd.concat([cells("A1", 0.5, [(100.0, 0.0), (100.0027, 0.0)], [0.2, 0.1]),           # one real 2-cell pixel
                    cells("A1", 0.5, [(100.0 + far, 0.0)], [0.3]),                              # another pixel sharing the value
                    cells("A1", 0.7, [(100.0, 0.5)], [0.3])], ignore_index=True)
    g = pts.pixel_groups(df); assert g[0] == g[1] and g[2] != g[0] and g[3] != g[2] and len(set(g)) == 3


def test_select_points_k_centroid_every_pixel_keeps_one():
    # pixel P: 6 flagged cells at nadir (N=7.1 -> k=2): the 2 hottest are kept, placed around the centroid of ALL six
    P = [(100.0 + 0.0034 * i, 1.0) for i in range(6)]; exc = [0.1, 0.5, 0.2, 0.9, 0.3, 0.05]
    Q = [(101.0, 2.0)]                                                                    # one-cell pixel: still one point
    df = pd.concat([cells("A1", 0.5, P, exc, vza=0.0), cells("A1", 0.8, Q, [0.2], vza=0.0)], ignore_index=True)
    pk, st = pts.select_points(df)
    assert st["groups"] == 2 and st["groups_spanning_split"] == 0 and st["firelevel_pixels"] == 2 and st["firelevel_pixels_without_point"] == 0
    assert len(pk[pk.pixel_id == pk.pixel_id.iloc[0]]) in (1, 2) and len(pk) == 3
    a = pk[pk.granule == "A1"]; assert sorted(a[a.n_flagged == 6].excess4_rad.tolist()) == [0.5, 0.9] and (a[a.n_flagged == 6].k == 2).all()
    w = np.array(exc); clon = (np.array([c[0] for c in P]) * w).sum() / w.sum()
    assert np.allclose(a[a.n_flagged == 6].clon, clon)
    q = pk[pk.n_flagged == 1].iloc[0]; assert abs(q.longitude - 101.0) < 1e-12 and abs(q.latitude - 2.0) < 1e-12   # offset 0 -> stays at the cell
    assert (pk.scan > 0).all() and (pk.track > 0).all()


def test_pixel_with_fewer_cells_than_k_keeps_all_and_spanning_group_split():
    df = cells("A1", 0.5, [(100.0, 1.0)], [0.1], vza=60.0)
    pk, st = pts.select_points(df); assert len(pk) == 1 and st["groups_kept_all_cells"] == 1
    chain = cells("A2", 0.5, [(100.0 + 0.012 * i, 1.0) for i in range(8)], [0.1] * 8)      # a chain of linked cells over 0.084 deg: not one pixel
    pk, st = pts.select_points(chain)
    assert st["groups_spanning_split"] == 1 and st["cells_in_split_groups"] == 8 and len(pk) == 8 and st["firelevel_pixels_without_point"] == 0 and (pk.n_flagged == 1).all()   # split into one-cell pixels, counted, none lost


def test_every_night_kept_and_scan_changes_with_view_zenith():
    d = pd.concat([cells("A1", 0.5, [(100.0, 1.0)], [0.3], vza=5.0, utc=1.1e9), cells("A2", 0.5, [(100.0, 1.0)], [0.3], vza=55.0, utc=1.1e9 + 86400)], ignore_index=True)
    pk, st = pts.select_points(d); assert len(pk) == 2                                    # same lattice cell on two nights: two points
    assert pk.track.iloc[1] > pk.track.iloc[0] and pk.scan.iloc[1] != pk.scan.iloc[0]   # track grows with angle; scan has the aggregation-zone sawtooth



def test_modis_footprint_nadir_and_edge():
    s, t, c = pts.modis_footprint_km([0.0]); assert abs(s[0] - 1.0) < 1e-9 and abs(t[0] - 1.0) < 1e-9 and not c[0]
    z_edge = np.degrees(np.arcsin((pts.EARTH_R_KM + pts.AQUA_H) / pts.EARTH_R_KM * np.sin(np.radians(55.0))))     # local zenith at the 55 deg scan edge (65.5)
    s, t, c = pts.modis_footprint_km([z_edge - 1e-6]); assert 4.7 <= s[0] <= 4.9 and 1.95 <= t[0] <= 2.05 and not c[0]     # published ~4.8 x 2.0 km
    s2, t2, c2 = pts.modis_footprint_km([69.0]); assert c2[0] and abs(s2[0] - 4.85) < 0.1


def test_month_job_end_to_end_from_aggregate_output(tmp_path):
    """worker-style parquets -> aggregate.build -> points.month_job: rows carry the documented schema and no free text."""
    from test_aggregate import rows, setup
    step = 0.0034
    a = rows("A2006213.1720", [(100.0 + i * step, 1.3) for i in range(6)], [T0] * 6, vza=0.0); a["modis4_rad"] = 0.5; a["excess4_rad"] = [.1, .5, .2, .9, .3, .05]; a["own_pixel_firelevel"] = 1
    q, done, failed = setup(tmp_path, {"A2006213.1720": (a, "ok", "v1.2")}, ["A2006213.1720"])
    agg.build(str(tmp_path), q, done, failed, "v1.2")
    ym, st = pts.month_job(("2006-08", str(tmp_path / "monthly" / "2006-08.parquet"), str(tmp_path / "monthly-points-viirs" / "2006-08.parquet")))
    r = pd.read_parquet(tmp_path / "monthly-points-viirs" / "2006-08.parquet")
    assert st["points"] == len(r) == 2 and st["groups"] == 1 and st["firelevel_pixels_without_point"] == 0
    assert (r.confidence == "model").all() and r.province.isna().all() and set(r.region) == {"test"} and sorted(r.excess4_rad) == [np.float32(0.5), np.float32(0.9)]
    assert list(r.columns)[:6] == ["latitude", "longitude", "scan", "track", "acq_date", "confidence"] and (r.scan.between(0.30, 0.90)).all()


def test_only_complete_flag_skips_incomplete_month(tmp_path, monkeypatch):
    from test_aggregate import rows, setup
    a = rows("A2006213.1720", [(100.1, 1.3)], [T0]); a["own_pixel_firelevel"] = 1
    q, done, failed = setup(tmp_path, {"A2006213.1720": (a, "ok", "v1.2")}, ["A2006213.1720", "A2006214.1720"])
    agg.build(str(tmp_path), q, done, failed, "v1.2")        # the second expected granule is missing -> month incomplete
    monkeypatch.setattr("sys.argv", ["points.py", "--root", str(tmp_path), "--only-complete", "--workers", "1"])
    pts.main()
    man = json.load(open(tmp_path / "monthly-points-viirs" / "MANIFEST.json"))
    assert man["months_skipped_incomplete"] == ["2006-08"] and man["n_points"] == 0
