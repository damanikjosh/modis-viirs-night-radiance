#!/usr/bin/env python
"""Check: geolocation of the flagged cells of one granule against the granule's own MYD03 (nearest MYD03 pixel centre distance, land / sea mask there, time inside the granule).
usage: check_geolocation.py PARQUET MYD03.hdf   (PARQUET = <root>/pixels/year=YYYY/month=MM/<id>.parquet)"""
import sys

import numpy as np
import pyarrow.parquet as pq
from pyhdf.SD import SD, SDC
from scipy.spatial import cKDTree

t = pq.read_table(sys.argv[1]).to_pandas()
f = SD(sys.argv[2], SDC.READ)
lat, lon, lsm = f.select("Latitude").get(), f.select("Longitude").get(), f.select("Land/SeaMask").get()
ok = np.isfinite(lat) & np.isfinite(lon) & (lat > -91)
xyz = lambda la, lo: np.stack([np.cos(np.deg2rad(la)) * np.cos(np.deg2rad(lo)), np.cos(np.deg2rad(la)) * np.sin(np.deg2rad(lo)), np.sin(np.deg2rad(la))], -1)
tree = cKDTree(xyz(lat[ok].astype(np.float64), lon[ok].astype(np.float64)))
d, i = tree.query(xyz(t.lat.values, t.lon.values))
km = d * 6371.0
li, si = np.nonzero(ok)
mask = lsm[li[i], si[i]]
d2, _ = tree.query(xyz(lat[ok].astype(np.float64)[i], lon[ok].astype(np.float64)[i]), k=2)   # local pixel spacing = distance to the nearest OTHER pixel centre
pix = d2[:, 1] * 6371.0
print(f"{sys.argv[1]}: {len(t)} flagged cells; nearest MYD03 pixel centre distance km: median {np.median(km):.2f} p95 {np.percentile(km, 95):.2f} max {km.max():.2f}")
print(f"  distance / local pixel spacing: max {np.max(km / pix):.2f} (a cell inside the footprint of its nearest pixel has <= ~0.71 for a square footprint)")
print("  LandSeaMask at nearest pixel counts:", {int(k): int((mask == k).sum()) for k in np.unique(mask)}, "(1 land, 2 coastline, 0 shallow ocean, 3 inland water, 6-7 ocean)")
print(f"  share land/coast: {np.isin(mask, (1, 2)).mean():.3f}")
print("  utc span s:", t.utc.max() - t.utc.min(), " cells lat", t.lat.min(), t.lat.max(), "lon", t.lon.min(), t.lon.max(), " MYD03 bbox lat", lat[ok].min(), lat[ok].max(), "lon", lon[ok].min(), lon[ok].max())
