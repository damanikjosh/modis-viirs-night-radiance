"""Aqua-MODIS x VIIRS pairs within 30 minutes from NASA's public granule metadata (CMR, no login).

  python scripts/find_pairs.py 2019-10-18 2019-10-18 [--bbox W S E N] [--night] [--viirs snpp noaa20] [--out PAIRS.csv]

Writes one row per pair: the six granule ids and urls (MYD14, MYD03, MYD021KM; VNP14IMG, VNP03MOD, VNP02IMG), mid-times, dt_min (VIIRS minus MODIS),
overlap area. Default output $NR_DATA_ROOT/pairs/pairs.csv."""
import argparse
from pathlib import Path

from nightrad.data.download import data_root
from nightrad.data.pairs import build_manifest

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("start")
ap.add_argument("end")
ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
ap.add_argument("--viirs", nargs="+", default=["snpp"], choices=["snpp", "noaa20"])
ap.add_argument("--night", action="store_true", help="night granules only")
ap.add_argument("--max-dt", type=float, default=30.0, help="minutes between granule mid-times")
ap.add_argument("--out")
ap.add_argument("--versions", action="store_true", help="print the CMR version of every product collection and exit")
a = ap.parse_args()
if a.versions:
    from nightrad.data.pairs import collection_versions

    for k, v in collection_versions().items():
        print(f"{k:9s} {v}")
    raise SystemExit(0)
df = build_manifest(a.start, a.end, tuple(a.bbox) if a.bbox else None, tuple(a.viirs), a.night, a.max_dt)
out = Path(a.out) if a.out else data_root() / "pairs" / "pairs.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} pairs")
if len(df):
    print(df.groupby(["daynight", "viirs_sensor"]).size().rename("pairs").to_string())
