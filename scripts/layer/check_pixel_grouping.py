#!/usr/bin/env python
"""Check: the "same MODIS radiance + adjacency" pixel grouping (points.pixel_groups) against MYD03 geometry on a few granules.
usage: check_pixel_grouping.py [--root DIR] [--raw DIR] --granules A2006253.1815,...
Each flagged cell is assigned to the nearest MODIS pixel centre of the granule's MYD03 (KD-tree; an approximation of the 1 km footprint, so cells near a footprint edge can land in the
neighbour). Reports: groups whose cells all fall in ONE MODIS pixel, cells outside their group's majority pixel, MODIS pixels that hold cells of two groups (split).
Needs the granule's MYD03 file in the raw cache (<root of data>/raw/nasa/MYD03/, filled by worker.py)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from pyhdf.SD import SD, SDC
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import points as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--raw", default=None)
ap.add_argument("--granules", required=True)
a = ap.parse_args()
data = os.environ.get("NR_DATA_ROOT", "data")
root, raw = a.root or f"{data}/layer", a.raw or f"{data}/raw/nasa"
res = {}
for g in a.granules.split(","):
    rec = json.load(open(f"{root}/_done/{g}.json")); hdf = f"{raw}/MYD03/{rec['inputs']['geo']['name']}"
    sd = SD(hdf, SDC.READ); lat = sd.select("Latitude")[:].astype(np.float64); lon = sd.select("Longitude")[:].astype(np.float64)
    ok = (lat > -90) & (lon > -181); li, si = np.nonzero(ok)
    f = glob.glob(f"{root}/pixels/year=*/month=*/{g}.parquet")[0]; px = pq.read_table(f).to_pandas()
    px["gid"] = P.pixel_groups(px)
    sc = lambda la, lo: np.c_[lo * np.cos(np.radians(la)) * 111.195, la * 111.195]
    t = cKDTree(sc(lat[ok], lon[ok])); d, ix = t.query(sc(px.lat.values, px.lon.values))
    px["mpix"] = li[ix] * 100000 + si[ix]
    by = px.groupby("gid").mpix.agg(lambda s: s.value_counts().iloc[0] / len(s)); n = px.groupby("gid").size()
    multi = by[n[by.index] > 1]
    inv = px.groupby("mpix").gid.nunique()
    ext = px.assign(x=sc(px.lat.values, px.lon.values)[:, 0], y=sc(px.lat.values, px.lon.values)[:, 1]).groupby("gid").agg(dx=("x", lambda v: v.max() - v.min()), dy=("y", lambda v: v.max() - v.min()))
    extent = np.hypot(ext.dx, ext.dy); npx = px.groupby("gid").mpix.nunique()
    res[g] = {"cells": len(px), "groups": int(px.gid.nunique()), "groups_with_2plus_cells": int(len(multi)), "groups_all_in_one_nearest_pixel": round(float((multi == 1).mean()), 3),
              "cells_in_group_majority_pixel": round(float((px.groupby("gid").mpix.transform(lambda s: s.value_counts().index[0]) == px.mpix).mean()), 3),
              "modis_pixels_with_cells_of_2plus_groups": int((inv > 1).sum()), "group_extent_km_median_p90_max": [round(float(np.median(extent)), 2), round(float(extent.quantile(.9)), 2), round(float(extent.max()), 2)], "groups_spanning_<=2_nearest_pixels": round(float((npx <= 2).mean()), 3), "nearest_pixel_distance_km_median": round(float(np.median(d)), 3)}
    print(g, json.dumps(res[g]))

