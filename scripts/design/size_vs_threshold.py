#!/usr/bin/env python
"""Design test (i): how much data a tighter or wider time window would give. Deterministic, reads recorded tables only (no tile data).
 A. all Aqua x VIIRS granule overlaps (>=1 km2) of the 119 sampled days of dataset v2 (CMR scan of the dataset, +-12 h search window): counts with |gap| <= T, all / night, S-NPP and NOAA-20, area-weighted too.
 B. the night-fire candidate pool (waves 1+2; only gaps <= 30 min were ever enumerated) and the 1,017 selected pairs: counts with |gap| <= T.
 C. TRAIN + SELECT tiles (REPORT metadata is not used): tiles, pairs, fire-bearing tiles, MODIS fire-level pixels with |tile gap| <= T.
 D. extrapolation beyond 30 min (labelled ESTIMATE): C(30) x [A night overlaps(T) / A night overlaps(30)].
Usage: size_vs_threshold.py <overlaps_v2days.csv.gz> <cand1.parquet> <cand2.parquet> <pairs_nf2.csv> <nf2_manifest.parquet> <nf2_split.json> <out dir>"""
import json, sys
import numpy as np, pandas as pd
ov, c1, c2, pn, mf, sp, out = sys.argv[1:8]
T = [2, 5, 10, 15, 20, 30, 45, 60, 90, 120]
O = pd.read_csv(ov); O["a"] = O.dt_min.abs()
rows = []
for pt in ("aqua_snpp", "aqua_noaa20"):
    for dn, f in (("all", O.m_dn.notna()), ("night", O.m_dn == "NIGHT")):
        g = O[(O.ptype == pt) & f]
        for t in T:
            s = g[g.a <= t]; rows.append({"table": "A_overlaps", "ptype": pt, "dn": dn, "T": t, "n": len(s), "area_Mkm2": float(s.area_km2.sum() / 1e6), "days": int(O.day.nunique())})
A = pd.DataFrame(rows)
C1, C2 = pd.read_parquet(c1), pd.read_parquet(c2); C1["wave"] = 1; C2["wave"] = 2; C = pd.concat([C1, C2]); C["a"] = C.dt_min.abs()
PN = pd.read_csv(pn); PN["a"] = PN.dt_min.abs()
rows = []
for t in T:
    rows.append({"table": "B_pairs", "T": t, "candidates": int((C.a <= t).sum()), "candidates_with_aqua_fire": int(((C.a <= t) & (C.n_aqua_elig > 0)).sum()), "selected": int((PN.a <= t).sum()),
                 "enumerated": bool(t <= 30)})
B = pd.DataFrame(rows)
M = pd.read_parquet(mf); S = json.load(open(sp))["tiles"]; side = {x: k for k, v in S.items() if k != "report" for x in v}
M["side"] = M.tile.map(side); M = M[M.side.notna()].copy(); M["a"] = M.dt_min.abs(); M["fire"] = (M.modis_fire_px > 0) | (M.viirs_fire_cells > 0)
rows = []
for t in T:
    s = M[M.a <= t]
    rows.append({"table": "C_tiles", "T": t, "tiles": len(s), "pairs": int(s.pair_id.nunique()), "fire_tiles": int(s.fire.sum()), "tiles_with_modis_fire": int((s.modis_fire_px > 0).sum()),
                 "modis_fire_px": int(s.modis_fire_px.sum()), "train_tiles": int((s.side == "train").sum()), "select_tiles": int((s.side == "select").sum()), "enumerated": bool(t <= 30)})
Cn = pd.DataFrame(rows)
nn = A[(A.ptype == "aqua_snpp") & (A.dn == "night")].set_index("T").n
Cn["estimate_scale_vs_30"] = [float(nn[t] / nn[30]) for t in Cn["T"]]
for k in ("tiles", "pairs", "fire_tiles", "modis_fire_px"):
    Cn["est_" + k] = np.where(Cn["T"] <= 30, Cn[k], Cn[k + ""].iloc[Cn.index[Cn["T"] == 30][0]] * Cn.estimate_scale_vs_30)
for nm, d in (("A_overlaps", A), ("B_pairs", B), ("C_tiles", Cn)):
    d.to_csv(f"{out}/size_{nm}.csv", index=False)
print(A[(A.ptype == "aqua_snpp")].to_string()); print(B.to_string()); print(Cn.to_string())
json.dump({"tiles_train_select": len(M), "pairs": int(M.pair_id.nunique()), "overlap_days": int(O.day.nunique()), "overlap_rows": len(O)}, open(f"{out}/size_meta.json", "w"))
