"""Pack the episode tiles in the format of scripts/pack.py (nightrad.raw.data.pack_dataset).

  python scripts/episode_pack.py [--out DIR]          default $NR_DATA_ROOT/episode (after scripts/episode_finalize.py)
Writes <out>/packed/kalimantan-2015/{night,day}/ (+ tiles.json: row -> tile, job; geo.json: tile centres and dates; pack_report.json). The exclusion rules of
nightrad.raw.exclude are deliberately not applied: these are the held-out tiles. One scene id per satellite overpass (granules less than 40 min apart),
stored as pair_id E<2015000 + index>."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from nightrad.data.download import data_root
from nightrad.raw.data import NpzTiles, pack_dataset

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out")
a = ap.parse_args()
out = a.out or str(data_root() / "episode")
ep = "kalimantan-2015"
man = pd.read_parquet(os.path.join(out, "manifest.parquet"))
man = man[man.keep == 1]
jt = man.groupby("job").agg(t=("modis_utc", "mean")).reset_index().sort_values("t")
jt["ov"] = (jt.t.diff() > 2400).cumsum()
jid = dict(zip(jt.job, jt.ov))
base = 2015 * 1000


class EpTiles(NpzTiles):
    def __getitem__(self, i):
        from nightrad.raw.schema import adapt_v3_tile
        with np.load(self.files[i], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["pair_id"] = np.array(f"E{base + jid[str(d['pair_id'])]}")
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in adapt_v3_tile(d).items()}
        t["eval_has_viirs"] = torch.tensor(int(d["has_viirs"]), dtype=torch.uint8)
        t["eval_ci"] = torch.tensor(int(os.path.basename(self.files[i]).split("_")[-2]), dtype=torch.int16)
        t["eval_cj"] = torch.tensor(int(os.path.basename(self.files[i]).split("_")[-1][:-4]), dtype=torch.int16)
        for k_, dt_ in (("tile_lat", torch.float64), ("tile_lon", torch.float64), ("year", torch.int16), ("month", torch.int8), ("day", torch.int8)):
            t[f"eval_{k_}"] = torch.tensor(float(d[k_]) if "tile" in k_ else int(d[k_]), dtype=dt_)
        t["eval_day_ord"] = torch.tensor(int(d["year"]) * 1000 + int(pd.Timestamp(int(d["year"]), int(d["month"]), int(d["day"])).day_of_year), dtype=torch.int32)
        return t


for dn, name in ((1, "night"), (0, "day")):
    m = man[man.dn == dn].sort_values("tile")
    if not len(m):
        continue
    files = [os.path.join(out, "tiles", r.job, r.tile + ".npz") for r in m.itertuples()]
    dst = os.path.join(out, "packed", ep, name)
    pack_dataset(EpTiles(files), dst)
    geo = {"tile_lat": [], "tile_lon": [], "year": [], "month": [], "day": [], "raw_file": []}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            for k_ in ("tile_lat", "tile_lon", "year", "month", "day"):
                geo[k_].append(float(z[k_]) if "tile" in k_ else int(z[k_]))
        geo["raw_file"].append(os.path.basename(f))
    json.dump(geo, open(os.path.join(dst, "geo.json"), "w"))
    json.dump([{"tile": r.tile, "job": r.job} for r in m.itertuples()], open(os.path.join(dst, "tiles.json"), "w"))
    json.dump({"episode": ep, "split": name, "n": len(files), "excluded_rule_applied": False, "pair_id_map": {k_: int(v_) for k_, v_ in jid.items()}}, open(os.path.join(dst, "pack_report.json"), "w"))
    print(name, len(files), dst, flush=True)
