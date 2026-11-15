"""Pack the finalized tiles of one split into the array pack the model reads (nightrad.raw.data.pack_dataset).

  python scripts/pack.py --split train|val2 [--tiles DIR] --out DIR [--min-common 0.02] [--only-side select --split-json configs/nf2_split.json]

Every tile of the split in manifest order that is a night tile (daynight == 1) and whose common clear-land mask (both sensors' validity class 2)
covers at least --min-common of the tile is packed. Extra fields (evaluation only, never model inputs): eval_episode, eval_region (reporting
region = manifest group_name), eval_far_km (1e9: val2 is >= 100 km from every train tile by construction).
Also writes names.json (tile ids in pack order), weights.npy (sample_weight x region_weight, the sampler weight), regions.json, episodes.json, pack_report.json.
--only-side select packs only the tiles of that side of the SELECT/REPORT lists (the other tiles are never opened).
Default tiles = $NR_DATA_ROOT/tiles."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from nightrad.data.download import data_root
from nightrad.raw.data import NpzTiles, pack_dataset

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--split", required=True, choices=["train", "val2"])
p.add_argument("--tiles")
p.add_argument("--out", required=True)
p.add_argument("--min-common", type=float, default=0.02)
p.add_argument("--only-side", choices=["select"], default=None)
p.add_argument("--split-json", default="configs/nf2_split.json")
a = p.parse_args()
root = a.tiles or str(data_root() / "tiles")
m = pd.read_parquet(os.path.join(root, "manifest.parquet"))
epi = {e: i for i, e in enumerate(sorted(m.episode.unique()))}
reg = {r: i for i, r in enumerate(sorted(m.group_name.unique()))}
m = m[m.split == a.split].sort_values("tile")
if a.only_side:
    keep = set(json.load(open(a.split_json))["tiles"][a.only_side])  # by id only; the other tiles stay unopened
    m = m[m.tile.isin(keep)]
meta = {r.tile: (r.episode, r.group_name, float(r.sample_weight), float(r.region_weight)) for r in m.itertuples()}
rep = {"split": a.split, "manifest_tiles": len(m), "missing_file": 0, "dropped_filter": 0, "min_common": a.min_common, "night_only": True}
files, kept = [], []
for n in m.tile:
    f = os.path.join(root, a.split, n + ".npz")
    if not os.path.exists(f):
        rep["missing_file"] += 1
        continue
    with np.load(f, allow_pickle=False) as z:
        c = float(((z["modis_vclass"] == 2) & (z["viirs_vclass"] == 2)).mean())
        if c < a.min_common or int(z["daynight"]) != 1:
            rep["dropped_filter"] += 1
            continue
    files.append(f)
    kept.append(n)
rep["kept"] = len(files)


class Tiles(NpzTiles):
    def __getitem__(self, i):
        t = super().__getitem__(i)
        e, g, *_ = meta[kept[i]]
        t["eval_episode"] = torch.tensor(epi[e], dtype=torch.int32)
        t["eval_region"] = torch.tensor(reg[g], dtype=torch.int32)
        t["eval_far_km"] = torch.tensor(1e9, dtype=torch.float32)
        return t


pack_dataset(Tiles(files), a.out)
np.save(os.path.join(a.out, "weights.npy"), np.array([meta[n][2] * meta[n][3] for n in kept], np.float64))
json.dump(kept, open(os.path.join(a.out, "names.json"), "w"))
json.dump(reg, open(os.path.join(a.out, "regions.json"), "w"))
json.dump(epi, open(os.path.join(a.out, "episodes.json"), "w"))
json.dump(rep, open(os.path.join(a.out, "pack_report.json"), "w"), indent=1)
print(json.dumps(rep))
