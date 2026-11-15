#!/usr/bin/env python
"""Experiment D step 1 figure: distribution of |dt| (pairing time gap) of TRAIN (small list), SELECT and old validation tiles, with the 5 / 15 / 30 minute bin counts (tiles, fire-bearing tiles). usage: d_dist_fig.py <dist.npz> <dist.json> <out dir>"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
z = np.load(sys.argv[1]); summ = json.load(open(sys.argv[2])); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
sets = [("TRAIN, small list (2,340 tiles)", np.concatenate([z["train_dt"], z["nf_dt"]]), np.concatenate([z["train_fire"], z["nf_fire"]]), "#0072B2"), ("SELECT (2,645 tiles)", z["select_dt"], z["select_fire"], "#D55E00"), ("old 9-pair validation (322 tiles)", z["old_dt"], z["old_fire"], "#009E73")]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.0), sharey=False); edges = np.arange(0, 31, 1.5)
for ax, (nm, dt, fire, col) in zip(axs, sets):
    ax.hist(dt, bins=edges, color=col, alpha=.8, label="all tiles"); ax.hist(dt[fire.astype(bool)], bins=edges, color="k", alpha=.55, label="fire-bearing tiles")
    for x in (5, 15):
        ax.axvline(x, color="0.4", ls="--", lw=.8)
    n5, n15, n30 = [int(((dt > lo) & (dt <= hi) | ((lo == 0) & (dt <= hi))).sum()) for lo, hi in ((0, 5), (5, 15), (15, 30))]; f5, f15, f30 = [int((fire.astype(bool) & ((dt > lo) & (dt <= hi) | ((lo == 0) & (dt <= hi)))).sum()) for lo, hi in ((0, 5), (5, 15), (15, 30))]
    ax.set_title(nm, fontsize=8); ax.set_xlabel("|dt| between MODIS and VIIRS tile time (min)"); ax.text(0.98, 0.97, f"<=5: {n5} ({f5} fire)\n5-15: {n15} ({f15} fire)\n15-30: {n30} ({f30} fire)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
axs[0].legend(fontsize=6, loc="center right"); axs[0].set_ylabel("tiles")
fig.suptitle("D1. How far apart in time are the paired MODIS and VIIRS tiles? (tile meta; fire-bearing = MODIS-only fire flag)", fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(out, "d1_dt_distribution.pdf")); fig.savefig(os.path.join(out, "d1_dt_distribution.png"), dpi=200)
