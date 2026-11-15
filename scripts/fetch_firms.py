"""Download the NASA FIRMS archive files (public, no key) used to choose which nights to tile.

  python scripts/fetch_firms.py [--countries Indonesia Thailand ...] [--years 2012 2022] [--out DIR]

Files: https://firms.modaps.eosdis.nasa.gov/data/country/<modis|viirs-snpp>/<year>/<sensor>_<year>_<Country>.csv -> $NR_DATA_ROOT/firms/.
Default countries: every country of nightrad.data.nightfire.COUNTRY_REGION (about 11 GB for 2012-2022)."""
import argparse
from pathlib import Path

from nightrad.data.download import data_root
from nightrad.data.nightfire import COUNTRY_REGION, download_firms

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--countries", nargs="+", default=sorted(COUNTRY_REGION))
ap.add_argument("--years", nargs=2, type=int, default=[2012, 2022], metavar=("FIRST", "LAST"))
ap.add_argument("--out")
a = ap.parse_args()
files = download_firms(Path(a.out) if a.out else data_root() / "firms", a.countries, range(a.years[0], a.years[1] + 1))
print(f"{len(files)} files")
