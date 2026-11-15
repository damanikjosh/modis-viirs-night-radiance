#!/usr/bin/env python
"""Layer step 4: the "Harmonized VIIRS" display points. Monthly de-duplicated cells (aggregate.py) -> monthly-points-viirs/YYYY-MM.parquet.

usage: points.py [--root DIR] [--only-complete] [--workers 4] [--months 2006-08 ...]
Input: <root>/monthly/YYYY-MM.parquet, coverage.json (from aggregate.py). Output: <root>/monthly-points-viirs/{YYYY-MM.parquet, counts.json, MANIFEST.json}.

Rule (tests in tests/layer/test_points.py):
 1. Cells of one MODIS source pixel = pixel_groups(): same granule + same MODIS 4 um radiance / T4, split into linked components (link = 1.5 x the MODIS scan footprint).
 2. Per pixel keep the k = max(1, round_half_up(0.25 * N)) cells with the highest predicted 4 um excess. N = number of 375 m cells in a MODIS pixel at the pixel's mean view
    zenith (MODIS footprint area / 0.375^2 km2; Aqua, R 6371 km, h 705 km). 0.25 = share of the 375 m cells of a MODIS fire-level pixel that real VIIRS flags (median of 83 validation pixels).
 3. Each kept cell is put around the excess-weighted centroid of ALL flagged cells of the pixel (weights = excess, floor 1e-6): its offset from the centroid is turned by -theta(lat), the
    angle between north-south and the ground track of a descending Aqua night pass (theta ~ 12 deg, derived from the orbit inclination 98.2 deg and Earth rotation, track_angle_deg()).
 4. scan / track = nominal VIIRS I-band footprint at the cell's own view zenith (viirs_footprint.py, Wolfe et al. 2013).
 5. Every night is kept (no per-month de-dup of nights). Rows carry no free text.
A pixel group wider than SPAN_MAX_DEG is a chain of several pixels, not one: each of its cells becomes its own one-cell pixel (counted), so no fire-level pixel is left without a point."""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viirs_footprint import footprint_km  # noqa: E402
from nightrad.inference.version import CKPT_SHA256, PIPELINE_VERSION  # noqa: E402

SHARE = 0.25
CELL_KM = 0.375
PIXEL_LINK = 1.5             # link distance, in MODIS scan footprints, between two cells of one pixel
SPAN_MAX_DEG = 0.06          # a group spanning more than this (6.7 km) is not one MODIS pixel
INCL_DEG, ORBIT_H_KM, EARTH_R_KM, OMEGA = 98.2, 705.0, 6371.0, 7.2921159e-5
AQUA_H, MODIS_MAX_SCAN = 705.0, 55.0


def data_root() -> str:
    return os.environ.get("NR_DATA_ROOT", "data")


def modis_footprint_km(vza_deg):
    """Real MODIS 1 km pixel footprint on the ground at the local zenith angle z (spherical Earth, Aqua h = 705 km): track = (1 km / h) * D, scan = track / cos z, D = slant range.
    Nadir 1.0 x 1.0 km, scan edge (55 deg) about 4.8 x 2.0 km. Zenith angles beyond the edge are clamped to the edge. -> (scan_km, track_km, clamped)."""
    z = np.radians(np.clip(np.asarray(vza_deg, dtype=float), 0, 90)); R, h = EARTH_R_KM, AQUA_H
    th = np.minimum(np.arcsin(np.clip(R / (R + h) * np.sin(z), 0, 1)), np.radians(MODIS_MAX_SCAN)); z2 = np.arcsin(np.clip((R + h) / R * np.sin(th), 0, 1))
    D = (R + h) * np.cos(th) - R * np.cos(z2)
    return (1.0 / h) * D / np.cos(z2), (1.0 / h) * D, np.arcsin(np.clip(R / (R + h) * np.sin(z), 0, 1)) > np.radians(MODIS_MAX_SCAN)


def pixel_groups(df: pd.DataFrame) -> np.ndarray:
    """-> int array: id of the MODIS source pixel of each row. Key = same granule + same rounded MODIS radiance (+ T4 when present); then connected components with link distance
    PIXEL_LINK x the scan footprint."""
    key = df["granule"].astype(str) + "_" + df["modis4_rad"].round(5).astype(str) + ("_" + df["modis_T4_K"].round(3).astype(str) if "modis_T4_K" in df else "")
    gid = np.full(len(df), -1, dtype=np.int64); nxt = 0
    lat = df["lat"].values; lon = df["lon"].values; scan_km = modis_footprint_km(df["vza"].values)[0] if "vza" in df else np.ones(len(df))
    for _, idx in pd.Series(np.arange(len(df))).groupby(key.values):
        idx = idx.values
        if len(idx) == 1:
            gid[idx] = nxt; nxt += 1; continue
        y = lat[idx] * 111.195; x = lon[idx] * np.cos(np.radians(lat[idx])) * 111.195; lk = PIXEL_LINK * np.maximum(scan_km[idx][:, None], scan_km[idx][None, :])
        adj = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2 <= lk ** 2
        lab = -np.ones(len(idx), dtype=np.int64)
        for i in range(len(idx)):
            if lab[i] >= 0:
                continue
            stack = [i]; lab[i] = nxt
            while stack:
                j = stack.pop()
                for m in np.nonzero(adj[j] & (lab < 0))[0]:
                    lab[m] = nxt; stack.append(m)
            nxt += 1
        gid[idx] = lab
    return gid


def k_of(n_cells, share=SHARE):
    """k = max(1, round half up (share * N))."""
    return np.maximum(1, np.floor(share * np.asarray(n_cells, dtype=float) + 0.5).astype(int))


def n_cells_geometric(vza_deg):
    """N = MODIS pixel footprint area on the ground at this view zenith / (0.375 km)^2 (float). Zenith clipped to 0..69.9."""
    sc, tr, _ = modis_footprint_km(np.clip(np.asarray(vza_deg, dtype=float), 0, 69.9))
    return sc * tr / (CELL_KM * CELL_KM)


def track_angle_deg(lat_deg):
    """Angle (deg) between the north-south line and the ground track of a DESCENDING Aqua pass at latitude lat (track points south and a little west).
    inertial heading: sin(az) = cos(i) / cos(lat); sub-satellite speed = v * R/(R+h); the Earth turns east under the satellite at omega * R * cos(lat). ~12.1 deg at the equator, ~12.4 at 25 N."""
    lat = np.radians(np.clip(np.asarray(lat_deg, dtype=float), -80, 80))
    v = np.sqrt(398600.4418 / (EARTH_R_KM + ORBIT_H_KM)); az = np.arcsin(np.clip(np.cos(np.radians(INCL_DEG)) / np.cos(lat), -1, 1))
    k = EARTH_R_KM / (EARTH_R_KM + ORBIT_H_KM)
    v_east = v * np.sin(az) * k - OMEGA * EARTH_R_KM * np.cos(lat)      # km/s, ground-relative (negative = west)
    v_north = -v * np.cos(az) * k                                      # descending: south
    return np.degrees(np.arctan2(np.abs(v_east), np.abs(v_north)))


def rotate_about(lon, lat, clon, clat, angle_deg):
    """Turn the offset (lon-clon, lat-clat) by angle_deg (counter-clockwise, x east / y north in km, kx = 111.32 cos(clat), ky = 110.57) and return the new lon, lat. Zero offset -> the centroid."""
    a = np.radians(angle_deg); kx = 111.32 * np.cos(np.radians(clat)); ky = 110.57
    dx = (lon - clon) * kx; dy = (lat - clat) * ky
    return clon + (dx * np.cos(a) - dy * np.sin(a)) / kx, clat + (dx * np.sin(a) + dy * np.cos(a)) / ky


def select_points(df: pd.DataFrame):
    """df = de-duplicated flagged cells of ONE month (lon, lat, utc, granule, vza, excess4_rad, modis4_rad, modis_T4_K, ...). -> (points DataFrame, stats dict).
    stats: counts of groups, split spanning groups, fire-level pixels without a point (must be 0)."""
    st = {"groups": 0, "groups_spanning_split": 0, "cells_in_split_groups": 0, "firelevel_pixels": 0, "firelevel_pixels_without_point": 0, "cells_flagged": int(len(df)),
          "groups_single_cell": 0, "groups_kept_all_cells": 0}
    if not len(df):
        return pd.DataFrame(), st
    d = df.reset_index(drop=True).copy()
    d["pixel_id"] = pixel_groups(d)
    span = np.maximum(d.groupby("pixel_id")["lon"].transform("max") - d.groupby("pixel_id")["lon"].transform("min"), d.groupby("pixel_id")["lat"].transform("max") - d.groupby("pixel_id")["lat"].transform("min"))
    wide = (span > SPAN_MAX_DEG).values
    st.update(groups_spanning_split=int(d.loc[wide, "pixel_id"].nunique()), cells_in_split_groups=int(wide.sum()))
    d.loc[wide, "pixel_id"] = d["pixel_id"].max() + 1 + np.arange(int(wide.sum()))
    d["w"] = d["excess4_rad"].clip(lower=1e-6); d["wl"] = d["lon"] * d["w"]; d["wa"] = d["lat"] * d["w"]
    g = d.groupby("pixel_id", sort=True)
    a = g.agg(wl=("wl", "sum"), wa=("wa", "sum"), w=("w", "sum"), n_flagged=("lon", "size"), vza_mean=("vza", "mean"))
    a["clon"] = a.wl / a.w; a["clat"] = a.wa / a.w
    a["N"] = n_cells_geometric(a.vza_mean.values); a["k"] = k_of(a["N"].values)
    hot = set(d.loc[d["own_pixel_firelevel"] == 1, "pixel_id"]) if "own_pixel_firelevel" in d else set()
    st.update(firelevel_pixels=len(hot), groups=int(len(a)), groups_single_cell=int((a.n_flagged == 1).sum()))
    d = d.join(a[["clon", "clat", "N", "k", "n_flagged", "vza_mean"]], on="pixel_id")
    d = d.sort_values(["pixel_id", "excess4_rad", "granule", "lat", "lon"], ascending=[True, False, True, True, True], kind="stable")
    d["rank_in_pixel"] = d.groupby("pixel_id").cumcount() + 1
    pk = d[d["rank_in_pixel"] <= d["k"]].copy()
    st["groups_kept_all_cells"] = int((a["n_flagged"] <= a["k"]).sum())
    st["firelevel_pixels_without_point"] = len(hot - set(pk["pixel_id"]))
    pk["longitude"], pk["latitude"] = rotate_about(pk.lon.values, pk.lat.values, pk.clon.values, pk.clat.values, -track_angle_deg(pk.clat.values))
    sc, tr, cl = footprint_km(pk["vza"].values)
    pk["scan"] = np.round(sc, 3); pk["track"] = np.round(tr, 3); st["clamped_to_scan_edge"] = int(cl.sum())
    return pk.reset_index(drop=True), st


def to_rows(pk: pd.DataFrame) -> pd.DataFrame:
    n = len(pk)
    t = pd.to_datetime(pk["utc"], unit="s", utc=True)
    return pd.DataFrame({"latitude": pk["latitude"].astype("float64").values, "longitude": pk["longitude"].astype("float64").values,
                         "scan": pk["scan"].astype("float32").values, "track": pk["track"].astype("float32").values, "acq_date": t.dt.strftime("%Y-%m-%d").values,
                         "confidence": "model", "province": pd.Series([None] * n, dtype=object).values, "clear": pd.array([pd.NA] * n, dtype="boolean"),
                         "region": pk["region"].values if "region" in pk else "region", "granule": pk["granule"].values, "pixel_id": pk["pixel_id"].astype("int64").values,
                         "rank_in_pixel": pk["rank_in_pixel"].astype("int16").values, "n_flagged_in_pixel": pk["n_flagged"].astype("int32").values,
                         "n_cells_pixel": pk["N"].astype("float32").values, "k": pk["k"].astype("int16").values, "vza_deg": pk["vza"].astype("float32").values,
                         "excess4_rad": pk["excess4_rad"].astype("float32").values, "pred_T4_K": pk["pred_T4_K"].astype("float32").values, "pred_dT_K": pk["pred_dT_K"].astype("float32").values,
                         "centroid_lon": pk["clon"].astype("float64").values, "centroid_lat": pk["clat"].astype("float64").values, "utc": pk["utc"].astype("float64").values})


def month_job(args):
    """worker: one month file -> (ym, n_points, stats)."""
    ym, src, dst = args
    cells = pd.read_parquet(src)
    pk, st = select_points(cells)
    rows = to_rows(pk) if len(pk) else pd.DataFrame()
    rows = rows.sort_values(["utc", "longitude", "latitude"], kind="stable").reset_index(drop=True) if len(rows) else rows
    st["points"] = int(len(rows))
    if len(rows):
        st["points_sizes_km"] = {f"{float(k):.1f}": int(v) for k, v in ((rows.scan + rows.track) / 2).astype("float64").round(1).value_counts().sort_index().items()}
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rows.to_parquet(dst, index=False, compression="zstd")
    return ym, st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="aggregate output directory (default $NR_DATA_ROOT/layer)")
    ap.add_argument("--only-complete", action="store_true", help="skip months that coverage.json does not mark complete")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--months", nargs="*", default=None)
    a = ap.parse_args()
    root = a.root or f"{data_root()}/layer"
    cover = json.load(open(f"{root}/coverage.json")) if os.path.exists(f"{root}/coverage.json") else {}
    jobs, skipped = [], []
    for f in sorted(glob.glob(f"{root}/monthly/*.parquet")):
        ym = os.path.basename(f)[:-8]
        if a.months and ym not in a.months:
            continue
        if a.only_complete and not cover.get(ym, {}).get("complete"):
            skipped.append(ym); continue
        jobs.append((ym, f, f"{root}/monthly-points-viirs/{ym}.parquet"))
    with Pool(max(1, a.workers)) as pool:
        res = pool.map(month_job, jobs, chunksize=1)
    counts = {ym: dict(st, complete=bool(cover.get(ym, {}).get("complete"))) for ym, st in res}
    os.makedirs(f"{root}/monthly-points-viirs", exist_ok=True)
    json.dump(counts, open(f"{root}/monthly-points-viirs/counts.json", "w"), indent=1, sort_keys=True)
    man = {"product": "harmonized-viirs-points", "status": "EXPERIMENTAL", "pipeline_version": PIPELINE_VERSION, "checkpoint_sha256": CKPT_SHA256,
           "generated_utc": datetime.now(timezone.utc).isoformat(), "months": sorted(counts), "months_skipped_incomplete": skipped, "n_points": int(sum(c["points"] for c in counts.values())),
           "rule": {"share": SHARE, "k": "max(1, round_half_up(0.25 * N)), N = MODIS footprint area at the pixel's mean view zenith / 0.375^2 km2", "centroid": "excess-weighted over all flagged cells of the pixel",
                    "rotation": "-theta(lat) deg, theta = descending Aqua ground-track angle from north-south (orbit-derived, ~12 deg)", "size": "nominal VIIRS I-band footprint at the cell's view zenith (S-NPP 829 km)",
                    "span_max_deg": SPAN_MAX_DEG}}
    json.dump(man, open(f"{root}/monthly-points-viirs/MANIFEST.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "rule"}))


if __name__ == "__main__":
    main()
