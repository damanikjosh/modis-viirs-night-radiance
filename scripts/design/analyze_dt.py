#!/usr/bin/env python
"""Design tests (ii)+(iii): does the target (VIIRS energy in a MODIS fire pixel) agree less with the MODIS input as the time gap grows INSIDE the 30-minute window? Deterministic (seed 0).
Input: collect.py outputs (--base: baselines on TRAIN+SELECT; --model name=path ...: SELECT only with model predictions). Items = MODIS fire-level, unsaturated, complete pixels (same definition as the
amount measure, nightrad.raw.amount_eval). All ranges are pair-clustered bootstrap (resample scene pairs with replacement; n_boot draws), 95 % percentile.
Statistics per |gap| bin (tile mean gap VIIRS minus MODIS): n pixels / tiles / pairs; Spearman(MODIS excess, VIIRS energy); share of pixels whose footprint holds no VIIRS fire-level cell; Even-spread amount error
mean |ln((pred+eps)/(meas+eps))| (amendment 4, eps=1e-3) and ratio of sums; share of VIIRS fire footprints without MODIS fire level (tile table). Trend: OLS of the per-pixel error on |gap| (per 10 min) with
scan angle (per 10 deg) and region dummies, pair-clustered. Direction: VIIRS first (gap < 0) vs MODIS first (gap > 0)."""
import argparse, json, os
import numpy as np, pandas as pd
EPS = 1e-3
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--model", nargs="*", default=[]); ap.add_argument("--out", required=True); ap.add_argument("--nboot", type=int, default=1000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng0 = 0


def load(p):
    z = np.load(p, allow_pickle=False); px = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith("px_")}); tl = pd.DataFrame({k[5:]: z[k] for k in z.files if k.startswith("tile_")}); return px, tl


def rank(x):
    return pd.Series(x).rank().to_numpy()


def spearman(x, y):
    if len(x) < 5:
        return np.nan
    rx, ry = rank(x), rank(y)
    return np.nan if rx.std() == 0 or ry.std() == 0 else float(np.corrcoef(rx, ry)[0, 1])


def stats(d):
    """pixel statistics of one item set"""
    if len(d) == 0:
        return {}
    e = np.log((d.pred_uniform + EPS) / (d.meas + EPS)); e2 = np.log((d.pred_excess_only + EPS) / (d.meas + EPS))
    r = {"n": len(d), "spearman": spearman(d.mexc.to_numpy(), d.meas.to_numpy()), "no_viirs_fire_share": float((d.nvfire == 0).mean()), "abs_err_even": float(np.abs(e).mean()),
         "abs_err_excess_only": float(np.abs(e2).mean()), "signed_err_even": float(e.mean()), "ratio_of_sums_even": float(d.pred_uniform.sum() / max(d.meas.sum(), 1e-12))}
    if "pred_model" in d:
        em = np.log((d.pred_model + EPS) / (d.meas + EPS)); r["abs_err_model"] = float(np.abs(em).mean()); r["signed_err_model"] = float(em.mean()); r["ratio_of_sums_model"] = float(d.pred_model.sum() / max(d.meas.sum(), 1e-12))
        r["gain_vs_even"] = r["abs_err_even"] - r["abs_err_model"]
    return r


def boot(d, binfun, nb, keys):
    """pair-clustered bootstrap of stats(binfun(d)) -> {stat: (lo, hi)}"""
    pairs = d.pair.unique(); idx = {p: np.flatnonzero(d.pair.to_numpy() == p) for p in pairs}; rng = np.random.default_rng(rng0); out = {k: [] for k in keys}
    for _ in range(nb):
        sel = rng.choice(pairs, len(pairs)); ii = np.concatenate([idx[p] for p in sel]); s = stats(binfun(d.iloc[ii]))
        for k in keys:
            out[k].append(s.get(k, np.nan))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if np.isfinite(v).any() else (np.nan, np.nan) for k, v in out.items()}


def tile_reverse(tl, lo, hi):
    t = tl[(tl.dt.abs() >= lo) & (tl.dt.abs() < hi)]; v = t.n_vfire_fp.sum()
    return {"n_vfire_fp": int(v), "no_modis_fire_share": float(1 - t.n_vfire_fp_modis_fire.sum() / v) if v else np.nan}


def reverse_boot(tl, lo, hi, nb):
    t = tl[(tl.dt.abs() >= lo) & (tl.dt.abs() < hi)]; pairs = t.pair.unique(); g = {p: (t.n_vfire_fp[t.pair == p].sum(), t.n_vfire_fp_modis_fire[t.pair == p].sum()) for p in pairs}; rng = np.random.default_rng(rng0); v = []
    for _ in range(nb):
        s = [g[p] for p in rng.choice(pairs, len(pairs))]; A = sum(x[0] for x in s); B = sum(x[1] for x in s); v.append(1 - B / A if A else np.nan)
    return (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def trend(d, y, nb, extra=True):
    """OLS y ~ 1 + |gap|/10 + vza/10 (+ region dummies); coefficient on |gap| per 10 min with pair-clustered range"""
    reg = pd.get_dummies(d.region, drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), d.dt.abs().to_numpy() / 10, d.vza.to_numpy() / 10] + ([reg.to_numpy()] if extra else [])); yy = y.to_numpy()
    b = np.linalg.lstsq(X, yy, rcond=None)[0][1]
    pairs = d.pair.unique(); idx = {p: np.flatnonzero(d.pair.to_numpy() == p) for p in pairs}; rng = np.random.default_rng(rng0); bs = []
    for _ in range(nb):
        ii = np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))]); Xi = X[ii]
        try:
            bs.append(np.linalg.lstsq(Xi, yy[ii], rcond=None)[0][1])
        except Exception:
            pass
    return {"coef_per_10min": float(b), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "mean_y": float(yy.mean()), "n": int(len(d)), "n_pairs": int(len(pairs))}


def run(px, tl, tag, bins, nb):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        f = lambda d, lo=lo, hi=hi: d[(d.dt.abs() >= lo) & (d.dt.abs() < hi)]; d = f(px)
        s = stats(d); keys = [k for k in s if k not in ("n",)]
        ci = boot(px, f, nb, keys); r = {"set": tag, "gap_lo": lo, "gap_hi": hi, "tiles": int(d.tile_id.nunique()), "pairs": int(d.pair.nunique()), **s}
        for k, (l, h) in ci.items():
            r[k + "_lo"], r[k + "_hi"] = l, h
        if tl is not None:
            rv = tile_reverse(tl, lo, hi); r.update(rv); r["no_modis_fire_share_lo"], r["no_modis_fire_share_hi"] = reverse_boot(tl, lo, hi, nb)
        rows.append(r)
    return pd.DataFrame(rows)


px, tl = load(a.base); tabs = {}; res = {"n_boot": a.nboot, "eps": EPS}
for side in ("all", "train", "select"):
    d = px if side == "all" else px[px.side == side]; t = tl if side == "all" else tl[tl.side == side]
    tabs[f"bins6_{side}"] = run(d, t, side, [0, 5, 10, 15, 20, 25, 30.001], a.nboot)
    tabs[f"bins3_{side}"] = run(d, t, side, [0, 5, 15, 30.001], a.nboot)
d = px.copy(); d["e_even"] = np.abs(np.log((d.pred_uniform + EPS) / (d.meas + EPS))); d["no_v"] = (d.nvfire == 0).astype(float)
res["trend_even_abs_err"] = trend(d, d.e_even, a.nboot); res["trend_no_viirs_fire"] = trend(d, d.no_v, a.nboot); res["trend_even_abs_err_no_controls"] = trend(d, d.e_even, a.nboot, extra=False)
res["trend_even_abs_err_select_only"] = trend(d[d.side == "select"], d.e_even[d.side == "select"], a.nboot)
# stratified by scan angle
d["vbin"] = pd.cut(d.vza, [0, 20, 40, 90], labels=["<20", "20-40", ">=40"], right=False); d["gbin"] = pd.cut(d.dt.abs(), [0, 5, 15, 30.001], labels=["0-5", "5-15", "15-30"], right=False)
st = d.groupby(["vbin", "gbin"], observed=True).agg(n=("e_even", "size"), pairs=("pair", "nunique"), abs_err_even=("e_even", "mean"), no_viirs_fire=("no_v", "mean")).reset_index(); tabs["vza_x_gap"] = st
# direction
d["first"] = np.where(d.dt < 0, "VIIRS first", "MODIS first"); d["e_signed"] = np.log((d.pred_uniform + EPS) / (d.meas + EPS))
rows = []
for gb, g in d.groupby("gbin", observed=True):
    for fi, h in g.groupby("first"):
        rows.append({"gap": str(gb), "first": fi, "n": len(h), "pairs": h.pair.nunique(), "abs_err_even": h.e_even.mean(), "signed_err_even": h.e_signed.mean(), "no_viirs_fire": h.no_v.mean()})
tabs["direction"] = pd.DataFrame(rows)
# direction difference with pair-clustered range, pooled over bins (MODIS first minus VIIRS first), controlling nothing else (descriptive)
def dirdiff(dd):
    a_ = dd[dd.dt > 0]; b_ = dd[dd.dt < 0]; return a_.e_even.mean() - b_.e_even.mean(), a_.e_signed.mean() - b_.e_signed.mean()
pairs = d.pair.unique(); idx = {p: np.flatnonzero(d.pair.to_numpy() == p) for p in pairs}; rng = np.random.default_rng(rng0); bb = []
for _ in range(a.nboot):
    bb.append(dirdiff(d.iloc[np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))])]))
bb = np.array(bb); d0 = dirdiff(d)
res["direction_modis_first_minus_viirs_first"] = {"abs_err": [float(d0[0]), float(np.percentile(bb[:, 0], 2.5)), float(np.percentile(bb[:, 0], 97.5))], "signed_err": [float(d0[1]), float(np.percentile(bb[:, 1], 2.5)), float(np.percentile(bb[:, 1], 97.5))],
                                                  "n_modis_first": int((d.dt > 0).sum()), "n_viirs_first": int((d.dt < 0).sum())}

# strata trends and per-bin direction ranges (added after the first run showed a possible edge-angle / one-direction effect; both are descriptive, not pre-specified)
for nm, m in (("vza_lt40", d.vza < 40), ("vza_ge40", d.vza >= 40)):
    res[f"trend_even_abs_err_{nm}"] = trend(d[m], d.e_even[m], a.nboot)
def dir15(dd):
    s = dd[(dd.dt.abs() >= 15)]; return s[s.dt > 0].e_even.mean() - s[s.dt < 0].e_even.mean()
rng = np.random.default_rng(rng0); b15 = [dir15(d.iloc[np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))])]) for _ in range(a.nboot)]
res["direction_15_30_modis_first_minus_viirs_first_abs_err"] = [float(dir15(d)), float(np.percentile(b15, 2.5)), float(np.percentile(b15, 97.5))]
# model (SELECT)
for nm in a.model:
    k, p = nm.split("="); mp, mt = load(p)
    tabs[f"model_{k}_bins3"] = run(mp, None, k, [0, 5, 15, 30.001], a.nboot); tabs[f"model_{k}_bins6"] = run(mp, None, k, [0, 5, 10, 15, 20, 25, 30.001], a.nboot)
    mp["e_even"] = np.abs(np.log((mp.pred_uniform + EPS) / (mp.meas + EPS))); mp["e_model"] = np.abs(np.log((mp.pred_model + EPS) / (mp.meas + EPS))); mp["gain"] = mp.e_even - mp.e_model
    res[f"trend_model_gain_{k}"] = trend(mp, mp.gain, a.nboot); res[f"trend_model_abs_err_{k}"] = trend(mp, mp.e_model, a.nboot)
res["items"] = {"pixels": int(len(px)), "pixels_train": int((px.side == "train").sum()), "pixels_select": int((px.side == "select").sum()), "tiles_with_items": int(px.tile_id.nunique()), "pairs": int(px.pair.nunique()),
                "tiles_total": int(len(tl)), "dt_abs_max": float(tl.dt.abs().max()), "regions": sorted(px.region.unique().tolist())}
for k, v in tabs.items():
    v.to_csv(f"{a.out}/{k}.csv", index=False)
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
