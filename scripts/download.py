"""Download the granules of a pair table: python scripts/download.py PAIRS.(csv|parquet) [--root DIR] [--threads 4] [--only P00001 ...]

PAIRS may be the word `paper` (the stored list of the paper's 1,017 pairs, granule ids only; urls are looked up in CMR).
Pair tables with granule ids only are completed with their download urls from CMR first.
Needs NR_EARTHDATA_TOKEN or ~/.netrc (see nightrad.data.download). Files go to $NR_DATA_ROOT/raw/<pair_id>/<PRODUCT>/."""
import argparse
import sys

from nightrad.data.download import LoginError, data_root, download
from nightrad.data.pairs import attach_granules
from nightrad.data.tables import read_pairs

ap = argparse.ArgumentParser()
ap.add_argument("pairs")
ap.add_argument("--root", help="default $NR_DATA_ROOT/raw")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--only", nargs="*", default=[], help="pair ids")
a = ap.parse_args()
df = read_pairs(a.pairs)
if a.only:
    df = df[df.pair_id.isin(a.only)]
if "viirs_l1b_url" not in df or df["viirs_l1b_url"].isna().any():
    df = attach_granules(df)
try:
    download(df, a.root or data_root() / "raw", a.threads)
except LoginError as e:
    sys.exit(f"ERROR: {e}")
