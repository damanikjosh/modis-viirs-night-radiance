#!/usr/bin/env python
"""Experiment B8 (POST-HOC, no scores): inference cost of the network on the Mac CPU. Usage: inference_cost.py <pack> <ckpt> <out json> [threads ...]
Per tile = Prep (radiance views, masks) + the network forward of nightrad.raw.core.run_scenes, batch 8, eval mode, no_grad, SELECT tiles (guarded loader), one warm-up batch, 6 timed batches (48 tiles) x 3 repeats."""
import json, os, resource, statistics, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from nightrad.raw.core import Prep, run_scenes
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import ref_cfg
pack, ck, out = sys.argv[1:4]; threads = [int(t) for t in sys.argv[4:]] or [1, 2]
dev = torch.device("cpu"); tiles, idx, g = load_select(pack, n_max=64); m, cfg, _ = load_model(ck, dev); rc = ref_cfg(cfg); rc.hide = type(rc.hide)(**{**rc.hide.__dict__, "enabled": False}) if hasattr(rc.hide, "__dict__") else rc.hide
n_par = sum(p.numel() for p in m.parameters()); res = {"params": n_par, "param_bytes_fp32": n_par * 4, "n_tiles_timed": 48, "guard": g, "per_threads": {}}
batch = lambda i: {k: v[i:i + 8] for k, v in tiles.items()}
for th in threads:
    torch.set_num_threads(th); reps = []
    with torch.no_grad():
        inp, _ = split_batch(batch(0), dev); run_scenes(m, Prep(inp, rc), rc, None, None)   # warm-up
        for _ in range(3):
            t0 = time.perf_counter()
            for i in range(0, 48, 8):
                inp, _ = split_batch(batch(i), dev); P = Prep(inp, rc); run_scenes(m, P, rc, None, None)
            reps.append(48 / (time.perf_counter() - t0))
    res["per_threads"][th] = {"tiles_per_s": reps, "median": statistics.median(reps)}
    print("threads", th, "tiles/s", reps, flush=True)
res["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
res["note"] = "tile = 96x96 cells (code constant); a 5-minute MODIS granule is about 2030 x 1354 km; non-overlapping 36 km tiles (96 x 375 m, assumed) -> about 2,120 tiles: seconds per granule = 2120 / tiles_per_s (EXTRAPOLATION, not measured on a real granule; the full pipeline has no granule-level reader here)"
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)[:600])
