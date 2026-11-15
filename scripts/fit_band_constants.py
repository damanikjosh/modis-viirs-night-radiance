#!/usr/bin/env python
"""Fit the effective wavelength of every band from the STORED radiance / brightness-temperature pairs of TRAIN tiles: lam = argmin mean (BT(rad, lam) - stored BT)^2
(scipy minimize_scalar), valid clear-land cells only; VIIRS I4 cells with BT >= 360 K (saturation placeholder) and MODIS band-21 cells are fitted separately.
  python scripts/fit_band_constants.py --tiles <dir of night TRAIN tiles> --out bands.json [--max-tiles 1038]"""
import argparse, glob, json
import numpy as np
from scipy.optimize import minimize_scalar
C1, C2 = 1.191042972e8, 14387.76877
bt = lambda L, lam: C2 / (lam * np.log1p(C1 / (lam ** 5 * L)))
p = argparse.ArgumentParser(); p.add_argument("--tiles", required=True); p.add_argument("--out", required=True); p.add_argument("--max-tiles", type=int, default=100000); p.add_argument("--per-tile", type=int, default=400)
a = p.parse_args(); rng = np.random.default_rng(0)
spec = {"viirs_I4": ("viirs", 4, None, 3.74), "viirs_I5": ("viirs", 11, None, 11.45), "modis_b22(4um)": ("modis", 4, 22, 3.96), "modis_b21(4um)": ("modis", 4, 21, 3.96), "modis_b31(11um)": ("modis", 11, None, 11.0)}
acc = {k: ([], []) for k in spec}
for f in sorted(glob.glob(a.tiles + "/*.npz"))[: a.max_tiles]:
    z = np.load(f, allow_pickle=False)
    for k, (s, ch, band, _) in spec.items():
        m = z[f"{s}_vclass"] == 2
        L, T = z[f"{s}_rad{ch}"].astype(np.float64), z[f"{s}_t{ch}"].astype(np.float64)
        m &= np.isfinite(L) & np.isfinite(T) & (L > 0) & (T > 150)
        if band is not None:
            m &= z[f"{s}_t4_band"] == band
        if s == "viirs" and ch == 4:
            m &= T < 360
        idx = np.flatnonzero(m.ravel())
        if len(idx) > a.per_tile:
            idx = rng.choice(idx, a.per_tile, replace=False)
        acc[k][0].append(L.ravel()[idx]); acc[k][1].append(T.ravel()[idx])
out = {}
for k, (s, ch, band, cur) in spec.items():
    L, T = np.concatenate(acc[k][0]), np.concatenate(acc[k][1])
    if not len(L):
        continue
    r = minimize_scalar(lambda lam: np.mean((bt(L, lam) - T) ** 2), bounds=(cur - 0.5, cur + 0.5), method="bounded", options={"xatol": 1e-6})
    out[k] = {"n_cells": int(len(L)), "current_lam": cur, "fitted_lam": float(r.x), "rms_K_current": float(np.sqrt(np.mean((bt(L, cur) - T) ** 2))), "rms_K_fitted": float(np.sqrt(r.fun)),
              "mean_K_current": float(np.mean(bt(L, cur) - T)), "median_T_K": float(np.median(T))}
    print(k, out[k])
json.dump(out, open(a.out, "w"), indent=1)
