#!/usr/bin/env python
"""Experiment D step 2 (POST-HOC): training subsets by pairing window at EQUAL size. TRAIN = train_night (1,038) + nf_train_night (1,302) packs (the small list); dt = |eval_dt_min|; fire = MODIS-only fire flag (d_dist.py).
Arms: a = |dt| <= 5 min (all such tiles; this is the smallest subset and fixes N and F), b = |dt| <= 15 min, c = |dt| <= 30 min (all), each subsampled to N tiles with exactly F fire-bearing tiles (fixed seed 20261115; a is not subsampled).
The 'd' arm (<= 30 min, ALL tiles) is the existing Gamma-BW100 (short) run (all 2,340 tiles, same hyper-parameters). usage: d_subsets.py <dist.npz> <train_night dir> <nf_train_night dir> <out root>"""
import json, os, shutil, sys
import numpy as np
dist, p_tr, p_nf, out = sys.argv[1:5]; z = np.load(dist); SEED = 20261115
dt = np.concatenate([z["train_dt"], z["nf_dt"]]); fire = np.concatenate([z["train_fire"], z["nf_fire"]]).astype(bool); src = np.concatenate([np.zeros(len(z["train_dt"]), int), np.ones(len(z["nf_dt"]), int)])
n_tr = len(z["train_dt"]); allidx = np.arange(len(dt)); rng = np.random.default_rng(SEED)
a = allidx[dt <= 5]; N, F = len(a), int(fire[a].sum())
def draw(pool):
    f = pool[fire[pool]]; nf = pool[~fire[pool]]
    assert len(f) >= F and len(nf) >= N - F, (len(f), len(nf), F, N)
    return np.sort(np.concatenate([rng.choice(f, F, replace=False), rng.choice(nf, N - F, replace=False)]))
arms = {"a": a, "b": draw(allidx[dt <= 15]), "c": draw(allidx)}; info = {"N": N, "F": F, "seed": SEED}
for arm, idx in arms.items():
    info[arm] = {"tiles": int(len(idx)), "fire_tiles": int(fire[idx].sum()), "from_train_night": int((src[idx] == 0).sum()), "from_nf_train_night": int((src[idx] == 1).sum()), "mean_dt_min": float(dt[idx].mean()), "max_dt_min": float(dt[idx].max()),
                 "dt_bins": {"<=5": int((dt[idx] <= 5).sum()), "5-15": int(((dt[idx] > 5) & (dt[idx] <= 15)).sum()), "15-30": int((dt[idx] > 15).sum())}}
    for nm, p, off, cnt in (("train_night", p_tr, 0, n_tr), ("nf_train_night", p_nf, n_tr, len(allidx) - n_tr)):
        sel = idx[(idx >= off) & (idx < off + cnt)] - off; o = os.path.join(out, arm, nm); os.makedirs(o, exist_ok=True)
        for f in os.listdir(p):
            if f.endswith(".npy"):
                np.save(os.path.join(o, f), np.load(os.path.join(p, f), mmap_mode="r")[sel])
            elif f != "meta.json":
                shutil.copy(os.path.join(p, f), o)
        m = json.load(open(os.path.join(p, "meta.json"))); m["n"] = int(len(sel)); json.dump(m, open(os.path.join(o, "meta.json"), "w"))
    print(arm, json.dumps(info[arm]), flush=True)
json.dump(info, open(os.path.join(out, "subsets.json"), "w"), indent=1)
