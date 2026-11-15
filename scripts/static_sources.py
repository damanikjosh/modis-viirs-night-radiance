"""Static / volcano / offshore fire-detection locations from the FIRMS standard-processing archive -> a csv the tiler uses to flag cells.

FIRMS `type`: 0 presumed vegetation fire, 1 active volcano, 2 other static land source, 3 offshore. Types 1, 2, 3 of MODIS_SP, VIIRS_SNPP_SP and
VIIRS_NOAA20_SP (from 2019) are collected from the FIRMS area API for two 5-day windows (1-5 March, 1-5 September) in each of the years 2013, 2016,
2019, 2022, 2023 and deduplicated to 0.0001 degree. Class of a flagged cell is unchanged; the flag is stored as `{sensor}_static`.

  FIRMS_MAP_KEY=<key> python scripts/static_sources.py [--out STATIC.csv]      (free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/)
Output columns: lat, lon, type (smallest seen), n_det, years, sources. Default output $NR_DATA_ROOT/static/static_sources.csv."""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from nightrad.data.download import data_root

URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
KEEP_TYPES = {"1", "2", "3"}
YEARS = (2013, 2016, 2019, 2022, 2023)
MONTHS = (3, 9)


def fetch_window(key: str, source: str, start: date, days: int, retries=4) -> tuple[list[dict], int]:
    """Stream one world window of one source; returns (static rows, total rows)."""
    url = f"{URL}/{key}/{source}/-180,-90,180,90/{days}/{start.isoformat()}"
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                rd = csv.reader(l for l in r.iter_lines(decode_unicode=True) if l)
                head = next(rd)
                if "latitude" not in head or "type" not in head:
                    raise RuntimeError(f"unexpected FIRMS body: {' '.join(head)[:80]}")
                ilat, ilon, ity = head.index("latitude"), head.index("longitude"), head.index("type")
                kept, n = [], 0
                for row in rd:
                    n += 1
                    if row[ity] in KEEP_TYPES:
                        kept.append({"lat": float(row[ilat]), "lon": float(row[ilon]), "type": int(row[ity])})
                return kept, n
        except (requests.RequestException, RuntimeError):
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    a = ap.parse_args()
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("FIRMS_MAP_KEY is not set")
    jobs = [(src, date(y, m, 1)) for y in YEARS for m in MONTHS
            for src in ("MODIS_SP", "VIIRS_SNPP_SP") + (("VIIRS_NOAA20_SP",) if y >= 2019 else ())]

    def run(j):
        src, d = j
        kept, n = fetch_window(key, src, d, 5)
        print(f"{src} {d}: {n} rows, {len(kept)} static", flush=True)
        return [{**r, "source": src, "year": d.year} for r in kept]

    with ThreadPoolExecutor(4) as ex:
        rows = [r for rs in ex.map(run, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df["lat_r"], df["lon_r"] = df.lat.round(4), df.lon.round(4)
    g = df.groupby(["lat_r", "lon_r"]).agg(type=("type", "min"), n_det=("type", "size"), years=("year", lambda s: "+".join(map(str, sorted(set(s))))),
                                           sources=("source", lambda s: "+".join(sorted(set(s))))).reset_index()
    g = g.rename(columns={"lat_r": "lat", "lon_r": "lon"})
    out = Path(a.out) if a.out else data_root() / "static" / "static_sources.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out, index=False)
    print(f"{len(df)} detections -> {len(g)} unique locations; by type {g.type.value_counts().to_dict()} -> {out}")


if __name__ == "__main__":
    main()
