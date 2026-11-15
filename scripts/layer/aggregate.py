#!/usr/bin/env python
"""Layer step 3: per-granule cell files -> one de-duplicated cell file per month, with counts and coverage.

usage: aggregate.py [--root DIR] [--catalogue catalogue.jsonl ...] [--min-version v1.2] [--months 2006-08 ...]
Input (root = --root, default $NR_DATA_ROOT/layer): pixels/ and _done/, _failed/ written by worker.py.
Output in root: monthly/YYYY-MM.parquet (all per-cell columns + acq_date), counts.json, coverage.json, MANIFEST.json.

Rules (tested in tests/layer/test_aggregate.py):
 * A granule counts only when its _done record is status "ok" AND its pipeline version is >= --min-version (a record without a version is v1.0).
   State per granule: ok | old_version (ok at a lower version: re-run the worker) | failed | missing (expected by a --catalogue file but no record).
 * The month of a granule is the month of its start time t (UTC); acq_date of a cell is the UTC date of the cell's pixel time.
 * A granule whose parquet disagrees with its record (row count, granule id or version in the file header) is excluded and reported as a row-count mismatch.
 * De-duplication: a cell is identified by its coordinates on the global 375 m lattice (lon, lat rounded to 1e-6 deg). Rows of the same cell whose pixel times chain within
   OVERPASS_GAP_S (20 min) are ONE overpass (tile seams, consecutive 5-minute granules): one row is kept (smallest view zenith, tie: earlier granule id). The same cell on
   another overpass (next orbit, another night) is a new observation and is kept.
 * A month is complete when every expected granule is ok and there is no mismatch. Without --catalogue the expected set is every granule that has a record."""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nightrad.inference.version import CKPT_SHA256, PIPELINE_VERSION, vkey

OVERPASS_GAP_S = 1200.0
STATUS_TEXT = ("EXPERIMENTAL model output, not an official product. Night only, Aqua only. Inputs are MYD021KM + MYD03 radiance and geometry; the MYD14 file is read only for "
               "its cloud / water / quality classes, fire detections are not an input.")


def data_root() -> str:
    return os.environ.get("NR_DATA_ROOT", "data")


def dedup(df: pd.DataFrame):
    """One row per (cell, overpass). df has lon, lat, utc, vza, granule. Returns (kept, n_removed)."""
    if not len(df):
        return df, 0
    d = df.copy()
    d["_kx"] = np.round(d["lon"].values * 1e6).astype(np.int64)
    d["_ky"] = np.round(d["lat"].values * 1e6).astype(np.int64)
    d = d.sort_values(["_kx", "_ky", "utc"], kind="stable")
    same = (d["_kx"].values[1:] == d["_kx"].values[:-1]) & (d["_ky"].values[1:] == d["_ky"].values[:-1]) & (np.diff(d["utc"].values) < OVERPASS_GAP_S)
    d["_op"] = np.concatenate([[0], np.cumsum(~same)])
    d["_v"] = d["vza"].fillna(999.0).values
    d = d.sort_values(["_op", "_v", "granule"], kind="stable")
    kept = d.drop_duplicates("_op", keep="first").sort_values(["utc", "lon", "lat"], kind="stable").drop(columns=["_kx", "_ky", "_op", "_v"])
    return kept.reset_index(drop=True), len(df) - len(kept)


def scene_state(done_rec, failed_rec, min_version) -> str:
    if done_rec and done_rec.get("status") == "ok":
        return "ok" if vkey(done_rec.get("pipeline_version")) >= vkey(min_version) else "old_version"
    return "failed" if (done_rec or failed_rec) else "missing"


def load_records(root: str):
    load = lambda d: {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{root}/{d}/*.json")}
    return load("_done"), load("_failed")


def expected_scenes(catalogues, done, failed):
    """-> list of {"id", "t"}: the union of the catalogue files, else every granule with a record."""
    out, seen = [], set()
    for cf in catalogues or []:
        for line in open(cf):
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"]); out.append({"id": r["id"], "t": r["t"]})
    if not catalogues:
        for i, rec in {**failed, **done}.items():
            out.append({"id": i, "t": rec["t"]})
    return out


def read_scene(root: str, gid: str, rec: dict, min_version: str):
    """-> (DataFrame or None, mismatch dict or None). A scene without flagged cells has no parquet and zero rows."""
    want = int(rec.get("flagged", 0))
    key = (rec.get("parquet") or {}).get("key")
    f = f"{root}/{key}" if key else None
    if f is None or not os.path.exists(f):
        return (None, {"id": gid, "done_flagged": want, "rows": 0}) if want else (pd.DataFrame(), None)
    t = pq.read_table(f).to_pandas()
    if len(t) != want:
        return None, {"id": gid, "done_flagged": want, "rows": len(t)}
    md = pq.read_metadata(f).metadata or {}
    if vkey(md.get(b"pipeline_version", b"").decode()) < vkey(min_version) or md.get(b"granule", b"").decode() != gid:
        return None, {"id": gid, "note": "file header does not match the scene id / version"}
    return t, None


def build(root: str, expected, done, failed, min_version: str, months=None):
    """Writes root/monthly, counts.json, coverage.json, MANIFEST.json. Returns the manifest."""
    by_month = {}
    for r in expected:
        by_month.setdefault(r["t"][:7], []).append(r["id"])
    os.makedirs(f"{root}/monthly", exist_ok=True)
    counts, cover = {}, {}
    for ym in sorted(by_month):
        if months and ym not in months:
            continue
        ids = by_month[ym]
        st = {i: scene_state(done.get(i), failed.get(i), min_version) for i in ids}
        pieces, mism = [], []
        for i in sorted(i for i in ids if st[i] == "ok"):
            t, m = read_scene(root, i, done[i], min_version)
            if m:
                mism.append(m)
            elif len(t):
                pieces.append(t)
        raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        kept, n_dup = dedup(raw) if len(raw) else (raw, 0)
        path = f"{root}/monthly/{ym}.parquet"
        if len(kept):
            kept = kept.assign(acq_date=pd.to_datetime(kept["utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d"))
            kept.to_parquet(path, index=False, compression="zstd")
        elif os.path.exists(path):
            os.remove(path)
        n = {s: sum(1 for i in ids if st[i] == s) for s in ("ok", "failed", "missing", "old_version")}
        counts[ym] = {"flagged_cells": int(len(kept)), "raw_rows": int(len(raw)), "duplicates_removed": int(n_dup),
                      "granules_with_flags": int(kept["granule"].nunique()) if len(kept) else 0, "row_count_mismatch": mism}
        cover[ym] = {"expected": len(ids), **n, "row_count_mismatch": len(mism), "complete": bool(ids and n["ok"] == len(ids) and not mism),
                     "not_ok_ids": sorted(i for i in ids if st[i] != "ok")[:20]}
    json.dump(counts, open(f"{root}/counts.json", "w"), indent=1, sort_keys=True)
    json.dump(cover, open(f"{root}/coverage.json", "w"), indent=1, sort_keys=True)
    man = {"product": "harmonized-viirs-cells", "status": "EXPERIMENTAL", "status_text": STATUS_TEXT, "min_pipeline_version": min_version, "pipeline_version": PIPELINE_VERSION,
           "checkpoint_sha256": CKPT_SHA256, "overpass_gap_s": OVERPASS_GAP_S, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_flagged_cells": int(sum(c["flagged_cells"] for c in counts.values())), "months_with_data": sorted(m for m, c in counts.items() if c["flagged_cells"] > 0),
           "months_complete": sorted(m for m, c in cover.items() if c["complete"])}
    json.dump(man, open(f"{root}/MANIFEST.json", "w"), indent=1)
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="worker output directory (default $NR_DATA_ROOT/layer)")
    ap.add_argument("--catalogue", nargs="*", default=None, help="catalogue.jsonl files that define the expected granules (for coverage)")
    ap.add_argument("--min-version", default=PIPELINE_VERSION)
    ap.add_argument("--months", nargs="*", default=None, help="only these YYYY-MM")
    a = ap.parse_args()
    root = a.root or f"{data_root()}/layer"
    done, failed = load_records(root)
    man = build(root, expected_scenes(a.catalogue, done, failed), done, failed, a.min_version, set(a.months) if a.months else None)
    print(json.dumps({k: v for k, v in man.items() if k != "status_text"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
