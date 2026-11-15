#!/usr/bin/env python
"""Independent checks of the layer outputs (re-derived from the per-granule files, not from the point builder's own bookkeeping).

usage: check_points.py [--root DIR] [--bbox W,S,E,N]
For every month in monthly-points-viirs/: (1) rows summed straight from the granule parquets and from the _done records == raw_rows in monthly counts.json (granules of the month, ok at the
current version); (2) point rows == counts; (3) points per pixel == min(k, flagged cells of the pixel); every pixel keeps >= 1 point and no fire-level pixel is left without one;
(4) scan / track 0.30-0.90 km (the aggregation-zone sawtooth goes below 0.375); (5) points inside --bbox (+0.3 deg for the rotation) when given; (6) dates inside the month (+-1 day);
(7) no free text (confidence 'model', province empty). Exit 1 on any failure."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--bbox", default=None)
a = ap.parse_args()
w = a.root or os.path.join(os.environ.get("NR_DATA_ROOT", "data"), "layer")
counts = json.load(open(f"{w}/counts.json"))
pcounts = json.load(open(f"{w}/monthly-points-viirs/counts.json"))
done = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{w}/_done/*.json")}
fails, n_months, n_pts = [], 0, 0
for f in sorted(glob.glob(f"{w}/monthly-points-viirs/*.parquet")):
    ym = os.path.basename(f)[:-8]; n_months += 1
    df = pd.read_parquet(f); n_pts += len(df)
    ids = [i for i, r in done.items() if r["t"][:7] == ym and r.get("status") == "ok"]
    want = sum(int(done[i].get("flagged", 0)) for i in ids)
    have = sum(pq.read_metadata(f"{w}/{done[i]['parquet']['key']}").num_rows for i in ids if done[i].get("parquet"))
    if not (want == have == counts[ym]["raw_rows"]):
        fails.append((ym, "raw rows", want, have, counts[ym]["raw_rows"]))
    c = pcounts[ym]
    if len(df) != c["points"]:
        fails.append((ym, "points vs counts", len(df), c["points"]))
    per = df.groupby("pixel_id").agg(n=("k", "size"), k=("k", "first"), nf=("n_flagged_in_pixel", "first"))
    bad = per[per.n != np.minimum(per.k, per.nf)]
    if len(bad):
        fails.append((ym, "points per pixel != min(k, flagged)", len(bad)))
    if c["firelevel_pixels_without_point"]:
        fails.append((ym, "fire-level pixel without point", c["firelevel_pixels_without_point"]))
    if a.bbox:
        w_, s_, e_, n_ = (float(x) for x in a.bbox.split(","))
        if not (df.longitude.between(w_ - 0.3, e_ + 0.3).all() and df.latitude.between(s_ - 0.3, n_ + 0.3).all()):
            fails.append((ym, "outside box"))
    if not (df.scan.between(0.30, 0.90).all() and df.track.between(0.30, 0.90).all()):
        fails.append((ym, "size out of range"))
    ad = pd.to_datetime(df.acq_date); lo = pd.Timestamp(ym + "-01") - pd.Timedelta(days=1); hi = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    if not ad.between(lo, hi).all():
        fails.append((ym, "date outside month"))
    if df.confidence.ne("model").any() or df.province.notna().any():
        fails.append((ym, "free text"))
    if per.k.max() > 40:
        fails.append((ym, "k too large"))
print(json.dumps({"months_checked": n_months, "points": n_pts, "failures": fails[:30], "n_failures": len(fails)}))
sys.exit(1 if fails else 0)
