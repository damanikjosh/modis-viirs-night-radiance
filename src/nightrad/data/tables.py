"""Small readers for the tables shipped with the package and for pair tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESOURCES = Path(__file__).parent / "resources"
PAPER_PAIRS = RESOURCES / "pairs_paper.csv"  # the 1,017 pairs (granule ids) the paper's tiles were built from
EXCLUSION_CENTRES = RESOURCES / "exclusion_centres.csv"  # lat, lon of the centres of an earlier held-out tile set


def read_pairs(path: str | Path) -> pd.DataFrame:
    """Pair table (csv or parquet) with parsed times; the string 'paper' reads the packaged list of the paper's pairs."""
    p = PAPER_PAIRS if str(path) == "paper" else Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    for c in ("modis_time", "viirs_time"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def read_centres(path: str | Path | None = None) -> pd.DataFrame:
    """Exclusion centres (columns lat, lon); the packaged list when path is None."""
    return pd.read_csv(path or EXCLUSION_CENTRES, usecols=["lat", "lon"])
