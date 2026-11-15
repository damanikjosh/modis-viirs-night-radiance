"""Pair table of the held-out episode (Central Kalimantan, box 111-115 E, 3 S-1 N, 14-23 September 2015): every Aqua x S-NPP pair of the
box within 35 minutes (the tiler then enforces |dt| <= 30 min per tile), from the public granule metadata (CMR).

  python scripts/episode_index.py [--out PAIRS.csv]        default $NR_DATA_ROOT/pairs/episode_kalimantan_2015.csv
Pair ids are E15_A<YYYYDDD>_<HHMM>_V<HHMM> (Aqua and VIIRS granule start). Then: scripts/download.py PAIRS.csv, scripts/build_episode.py,
scripts/episode_finalize.py, scripts/episode_pack.py."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from nightrad.data.download import data_root
from nightrad.data.pairs import COLLECTION, MODIS, VIIRS, attach_granules, cmr_granules, find_pairs, granule_geom, granule_record, pair_row, stamp

BOX = (111.0, -3.0, 115.0, 1.0)  # W S E N
START, END = datetime(2015, 9, 14, tzinfo=timezone.utc), datetime(2015, 9, 24, tzinfo=timezone.utc)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out")
a = ap.parse_args()
ses = requests.Session()
pad = timedelta(minutes=40)


def fetch(prod, t0, t1):
    recs = [granule_record(e) for e in cmr_granules(COLLECTION[prod], t0, t1, BOX, ses)]
    recs = [r for r in recs if r["url"]]
    for r in recs:
        r["geom"] = granule_geom(r["polys"])
    return recs


modis = fetch(MODIS["fire"], START, END)
viirs = fetch(VIIRS["snpp"]["fire"], START - pad, END + pad)
pairs = find_pairs(modis, {VIIRS["snpp"]["fire"]: viirs}, max_dt_min=35.0)
df = attach_granules(pd.DataFrame([pair_row(p) for p in pairs]))
st = lambda s: stamp(s).split(".")
df.insert(0, "pair_id", [f"E15_A{st(m)[0][1:]}_{st(m)[1]}_V{st(v)[1]}" for m, v in zip(df.modis_id, df.viirs_id)])
df["daynight"] = df.daynight.str.lower()
out = Path(a.out) if a.out else data_root() / "pairs" / "episode_kalimantan_2015.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"{len(modis)} Aqua granules, {len(viirs)} S-NPP granules in the box -> {len(df)} pairs -> {out}")
print(df.daynight.value_counts().to_string())
