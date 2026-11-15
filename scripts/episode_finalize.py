"""Episode tiles, finalize: de-duplicate, manifest, coverage table.

  python scripts/episode_finalize.py [--out DIR]          default $NR_DATA_ROOT/episode (output of scripts/build_episode.py)
Granule edges give the same grid centre twice in one overpass: per grid centre and overpass (same centre within 30 min) the tile with the most
two-sensor overlap is kept (keep = 1), the others are marked keep = 0. Writes manifest.parquet (one row per tile), manifest.json (counts, granules and
tiles per night, night / day, valid-cell fraction, coverage table) and README.md."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nightrad.data.download import data_root

NC = 9216  # cells per tile
BOX_CELLS = 144 * NC

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out")
a = ap.parse_args()
out = Path(a.out) if a.out else data_root() / "episode"
rows, jobs, gran, dropped, secs = [], [], {}, {"low_overlap": 0, "dt": 0, "invalid": 0}, 0.0
for f in sorted(glob.glob(str(out / "stats" / "*.json"))):
    s = json.load(open(f))
    rows += s["rows"]
    jobs.append(s["job"])
    gran[s["job"]] = s["granule"]
    for k, v in s["dropped"].items():
        dropped[k] += v
    secs += s["secs"]["read"] + s["secs"]["build"]
df = pd.DataFrame(rows)
df["cover"] = df.overlap_frac
df["keep"] = 0
df = df.sort_values(["ci", "cj", "modis_utc"]).reset_index(drop=True)
df["overpass_group"] = (df.groupby(["ci", "cj"]).modis_utc.diff().fillna(1e9) > 1800).cumsum()
df.loc[df.groupby("overpass_group").cover.idxmax().values, "keep"] = 1
df["cloud_frac"] = df.modis_vc3 / NC
df["dn_name"] = np.where(df.dn == 1, "night", "day")
k = df[df.keep == 1].copy()
# local (UTC+7) night id: shifted so one night is one date (a night starts the evening before)
k["night_id"] = np.where(k.dn == 1, (pd.to_datetime(k.modis_utc, unit="s") + pd.Timedelta(hours=7 - 12)).dt.strftime("%Y-%m-%d"),
                         (pd.to_datetime(k.modis_utc, unit="s") + pd.Timedelta(hours=7)).dt.strftime("%Y-%m-%d"))
cov = [{"date_local": d, "dn": dn, "overpasses_jobs": int(g.job.nunique()), "tiles": len(g), "box_covered_frac": round(len(g) / 144, 3),
        "clear_land_frac_of_box": round(float(g.modis_vc2.sum() / BOX_CELLS), 4), "cloud_frac_of_covered": round(float(g.modis_vc3.sum() / (len(g) * NC)), 3),
        "common_valid_frac_of_box": round(float((g.common_valid_frac * NC).sum() / BOX_CELLS), 4),
        "tiles_with_fire_cells": int(((g.modis_fire_cells > 0) | (g.viirs_fire_cells > 0)).sum())} for (d, dn), g in k.groupby(["night_id", "dn_name"])]
man = {"episode": "kalimantan-2015", "tiler": "nightrad.data.tiles_gpu.build_tiles_gpu", "box": "111-115E, 3S-1N", "grid": "12 x 12 tiles of 96 x 96 cells @ 375 m, centre spacing 37 km",
       "jobs": len(jobs), "tiles_built": len(df), "tiles_kept_after_dedup": len(k), "duplicates_removed": len(df) - len(k), "dropped_while_building": dropped,
       "night_tiles": int((k.dn == 1).sum()), "day_tiles": int((k.dn == 0).sum()), "valid_cell_fraction_modis": round(float(k.modis_valid_frac.mean()), 4),
       "valid_cell_fraction_viirs": round(float(k.viirs_valid_frac.mean()), 4), "common_valid_cell_fraction": round(float(k.common_valid_frac.mean()), 4),
       "cloud_fraction_modis": round(float(k.cloud_frac.mean()), 4), "tiles_per_day": {str(d): len(g) for d, g in k.groupby("night_id")}, "coverage": cov,
       "build_seconds_total": round(secs, 1)}
df.to_parquet(out / "manifest.parquet")
json.dump(man, open(out / "manifest.json", "w"), indent=1)
(out / "README.md").write_text("""# kalimantan-2015 (held-out test tiles)
Central Kalimantan, 111-115 E 3 S-1 N, September 2015: Aqua MODIS x S-NPP VIIRS pair tiles, same rules as the training tiles (docs/tile_schema.md).
Do not train on these. Every tile carries split="none", holdout="episode-heldout". scripts/finalize.py never reads this directory and
nightrad.data.splits drops the area (+150 km) from every training and validation list. Fire-product arrays (`eval_*`) are evaluation side information only.
""")
print(json.dumps({k_: v for k_, v in man.items() if k_ not in ("coverage", "tiles_per_day")}, indent=1))
