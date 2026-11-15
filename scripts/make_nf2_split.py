#!/usr/bin/env python
"""Write the TRAIN / SELECT / REPORT tile lists (configs/nf2_split.json) from the manifest of scripts/finalize.py.
TRAIN = manifest split 'train'. SELECT / REPORT = manifest split 'val2' divided by 100 km spatial block (nightrad.raw.nf2_split). Verifies and records the leak and exclusion checks.
  python scripts/make_nf2_split.py $NR_DATA_ROOT/tiles/manifest.parquet configs/nf2_split.json [--seed 20261114]"""
import argparse, hashlib, json
import numpy as np, pandas as pd
from nightrad.raw import nf2_split as S
from nightrad.raw.exclude import dist_to_box_km

p = argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("out"); p.add_argument("--seed", type=int, default=20261114)
a = p.parse_args()
m = pd.read_parquet(a.manifest); tr = m[m.split == "train"].reset_index(drop=True); va = m[m.split == "val2"].reset_index(drop=True)
assert len(tr) and len(va) and set(m.split) == {"train", "val2"}
va["block"] = S.blocks(va.lat.values, va.lon.values, 100.0)
side = S.assign(va.group_name.values, va.block.values, va.modis_fire_px.values, va.tile.values, seed=a.seed)
va["side"] = va.block.map(side)
sel, rep = va[va.side == "select"], va[va.side == "report"]
chk = {}
chk["blocks_total"] = int(va.block.nunique()); chk["blocks_select"] = int(sel.block.nunique()); chk["blocks_report"] = int(rep.block.nunique())
chk["shared_blocks_select_report"] = int(len(set(sel.block) & set(rep.block)))
chk["shared_episodes_select_report"] = int(len(set(sel.episode) & set(rep.episode)))
chk["episodes_cut_across_sides"] = int((va.groupby("episode").side.nunique() > 1).sum())
chk["shared_episodes_train_select"] = int(len(set(tr.episode) & set(sel.episode))); chk["shared_episodes_train_report"] = int(len(set(tr.episode) & set(rep.episode)))
chk["min_km_train_select"] = S.min_dist_km(sel.lat, sel.lon, tr.lat, tr.lon); chk["min_km_train_report"] = S.min_dist_km(rep.lat, rep.lon, tr.lat, tr.lon)
chk["min_km_select_report"] = S.min_dist_km(sel.lat, sel.lon, rep.lat, rep.lon)
chk["min_dist_exclusion_centre_km_train"] = float(tr.dist_exclusion_centre_km.min()); chk["min_dist_exclusion_centre_km_select"] = float(sel.dist_exclusion_centre_km.min()); chk["min_dist_exclusion_centre_km_report"] = float(rep.dist_exclusion_centre_km.min())
sumatra = {"lon_min": 100.0, "lon_max": 108.0, "lat_min": -6.0, "lat_max": 2.0}; kali = {"lon_min": 111.0, "lon_max": 115.0, "lat_min": -3.0, "lat_max": 1.0}
SA = dict(lat_min=-56.0, lat_max=12.5, lon_min=-82.0, lon_max=-34.0)


def viol(d):
    sep_oct = (d.year == 2015) & d.month.isin([9, 10])
    return {"year_2023": int((d.year == 2023).sum()), "south_america_box": int(((d.lat >= SA["lat_min"]) & (d.lat <= SA["lat_max"]) & (d.lon >= SA["lon_min"]) & (d.lon <= SA["lon_max"])).sum()),
            "sumatra_block_plus60km": int((dist_to_box_km(d.lat, d.lon, sumatra) < 60).sum()), "kalimantan_sep_oct_2015_plus150km": int((sep_oct & (dist_to_box_km(d.lat, d.lon, kali) < 150)).sum())}


chk["exclusion_violations"] = {"train": viol(tr), "select": viol(sel), "report": viol(rep)}
chk["south_america_box_note"] = "approximate box lat -56..12.5, lon -82..-34 (the builder's own box is not in the manifest)"
assert chk["shared_blocks_select_report"] == 0 and chk["episodes_cut_across_sides"] == 0 and chk["min_km_train_select"] >= 100 and chk["min_km_train_report"] >= 100
rows = []
for r in sorted(m.group_name.unique()):
    row = {"region": r}
    for nm, d in (("train", tr), ("select", sel), ("report", rep)):
        d = d[d.group_name == r]
        row[nm] = {"tiles": int(len(d)), "modis_fire_px": int(d.modis_fire_px.sum()), "viirs_fire_cells": int(d.viirs_fire_cells.sum()), "tiles_with_modis_fire_px": int((d.modis_fire_px > 0).sum()), "blocks": int(d.block.nunique()) if "block" in d else None}
    rows.append(row)
tot = {nm: {"tiles": int(len(d)), "modis_fire_px": int(d.modis_fire_px.sum()), "viirs_fire_cells": int(d.viirs_fire_cells.sum()), "tiles_with_modis_fire_px": int((d.modis_fire_px > 0).sum())} for nm, d in (("train", tr), ("select", sel), ("report", rep))}
cfg = {"rule": ["Data properties only, no model output.", "TRAIN = manifest split train. SELECT/REPORT = manifest split val2 divided by 100 km spatial block (connected components, tiles closer than 100 km on any date), never splitting a block or a tile-level episode.",
       "Per region (group_name) blocks are visited by descending fire-level MODIS pixels and given to the side holding fewer of that region's fire pixels; fixed seed. See src/nightrad/raw/nf2_split.py. REPORT numbers are never looked at before stage 5."],
       "seed": a.seed, "manifest_sha256_16": hashlib.sha256(open(a.manifest, "rb").read()).hexdigest()[:16], "checks": chk, "table": rows, "totals": tot, 
       "tiles": {"train": tr.tile.tolist(), "select": sel.tile.tolist(), "report": rep.tile.tolist()},
       "blocks": {"select": sorted(int(b) for b in sel.block.unique()), "report": sorted(int(b) for b in rep.block.unique())}, "tile_block": {t: int(b) for t, b in zip(va.tile, va.block)}}
json.dump(cfg, open(a.out, "w"), indent=0)
print(json.dumps({"checks": chk, "totals": tot}, indent=1)); print(pd.DataFrame([{"region": r["region"], **{f"{n}_{k}": r[n][k] for n in ("train", "select", "report") for k in ("tiles", "modis_fire_px")}} for r in rows]).to_string())
