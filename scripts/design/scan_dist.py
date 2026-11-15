#!/usr/bin/env python
"""Design rationale 3: which part of the swath do the paired TRAIN+SELECT tiles come from? Observed distribution of the computed scan angle at tile centres (pairing scan of the dataset) against the distribution expected if tiles were spread uniformly over the ground width of the swath (spherical Earth: sin(zenith)=(R+h)/R sin(scan), ground arc = zenith - scan).
Output: scan_dist.csv (bin shares) + scan_dist.json. Usage: scan_dist.py <tile_scan_angles.csv.gz> <out dir>"""
import json, sys
import numpy as np, pandas as pd
R = 6371.0088; SENS = {"modis": (705.0, 55.0), "viirs": (824.0, 56.28)}
t = pd.read_csv(sys.argv[1]); out = sys.argv[2]; bins = np.arange(0, 60, 5.0); rows = []; js = {}
for s, (h, half) in SENS.items():
    th = np.linspace(0, half, 20001); zen = np.arcsin(np.clip((R + h) / R * np.sin(np.deg2rad(th)), 0, 1)); g = (zen - np.deg2rad(th)) * R   # ground distance from nadir at each scan angle
    cdf = g / g[-1]; exp = np.diff(np.interp(bins.tolist() + [half], th, cdf)) if False else None
    edges = np.minimum(np.append(bins, half), half); e_cdf = np.interp(edges, th, cdf); share_exp = np.diff(e_cdf) / np.diff(e_cdf).sum() * (1 if edges[-1] == half else 1)
    a = t[f"{s}_scan_deg"].abs(); obs = np.histogram(a, np.append(bins, 90))[0]; obs = obs / obs.sum()
    for lo, hi, o, e in zip(bins, np.append(bins[1:], half), obs, np.append(share_exp, 0)[: len(bins)]):
        rows.append({"sensor": s, "scan_lo": lo, "scan_hi": hi, "observed_share": float(o), "uniform_ground_share": float(e)})
    gh = np.interp(40.0, th, cdf); js[s] = {"tiles": int(len(a)), "share_ge_40_obs": float((a >= 40).mean()), "share_ge_40_uniform": float(1 - gh), "mean_abs_scan_obs": float(a.mean()), "median_abs_scan_obs": float(a.median()),
                                           "max_obs": float(a.max()), "ground_halfwidth_km": float(g[-1])}
js["corr_modis_viirs_scan"] = float(np.corrcoef(t.modis_scan_deg.abs(), t.viirs_scan_deg.abs())[0, 1]); js["n_tiles"] = int(len(t)); js["sides"] = t.side.value_counts().to_dict()
pd.DataFrame(rows).to_csv(f"{out}/scan_dist.csv", index=False); json.dump(js, open(f"{out}/scan_dist.json", "w"), indent=1); print(json.dumps(js, indent=1)); print(pd.DataFrame(rows).to_string())
