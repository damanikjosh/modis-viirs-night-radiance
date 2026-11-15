#!/usr/bin/env python
"""Experiment D step 1 (POST-HOC): per-tile pairing time gap |dt| (eval_dt_min of the pack = mean VIIRS line time - mean MODIS line time over the tile, from the tile meta) and the MODIS-only fire flag
(modis_fire_flags: >= 1 unsaturated complete MODIS pixel with 4 um excess >= 0.5 ln over its local median) for the packs given. Writes <out>.npz (per pack: dt, fire) and prints the counts per bin.
usage: d_dist.py <ckpt for cfg> <out prefix> <name=packdir> [<name=packdir> ...]   (TRAIN packs and SELECT / old-validation packs only)"""
import json, sys
import numpy as np, torch
from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards, modis_fire_flags
from nightrad.raw.sweep_eval import ref_cfg
ck, out = sys.argv[1:3]; cfg = ref_cfg(RunConfig.from_dict(torch.load(ck, map_location="cpu", weights_only=False)["cfg"])); res = {}; summ = {}
BINS = [("<=5", 0, 5), ("5-15", 5, 15), ("15-30", 15, 30), (">30", 30, 1e9)]
for nv in sys.argv[3:]:
    name, path = nv.split("="); ds = PackedShards(path); n = len(ds)
    dt = np.abs(np.load(path + "/eval_dt_min.npy").reshape(n, -1)[:, 0].astype(np.float64)); fire = modis_fire_flags(ds, cfg, torch.device("cpu"), workers=0)
    res[name + "_dt"] = dt; res[name + "_fire"] = fire
    summ[name] = {"n": int(n), "nan_dt": int(np.isnan(dt).sum()), "bins": {b: {"tiles": int(((dt > lo) & (dt <= hi) | ((lo == 0) & (dt <= hi))).sum()), "fire_tiles": int((((dt > lo) & (dt <= hi) | ((lo == 0) & (dt <= hi))) & fire).sum())} for b, lo, hi in BINS},
                  "dt_quantiles": {str(q): float(np.nanquantile(dt, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    print(name, json.dumps(summ[name]), flush=True)
np.savez(out + ".npz", **res); json.dump(summ, open(out + ".json", "w"), indent=1)
