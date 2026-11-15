#!/usr/bin/env python
"""Paper experiments B5-B7 (POST-HOC): analysis of the per-pixel collections (scripts/exp/run_collect.py) on SELECT. Items = MODIS fire-level, unsaturated, complete pixels (pixel-scale items of the amount measure).
Usage: b_analysis.py <px dir> <stored score dir (select)> <regions.json> <out dir> <run> [<run> ...]
Cross-check first: the pooled model numbers of every collection must equal the stored score file (amount.pixel.methods.model: n, ratio_of_sums, abs_log_err)."""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PX, SCD, REG, OUT = sys.argv[1:5]; RUNS = sys.argv[5:]; os.makedirs(OUT, exist_ok=True); EPS = 1e-3
reg = {v: k for k, v in json.load(open(REG)).items()}
COL = {"gammabw100-nf2": ("#009E73", "^", "Gamma-BW100 (best-val)"), "gammabw1000-nf2": ("#E69F00", "D", "Gamma-BW1000 (best-val)")}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
D = {r: dict(np.load(os.path.join(PX, f"px_{r}.npz"))) for r in RUNS}
le = lambda p, m: np.abs(np.log((p + EPS) / (m + EPS)))
fam = lambda r: re.sub(r"-s\d+$", "", r)


def boot(vals_by_pair, n=2000, seed=0):
    v = np.array(list(vals_by_pair.values())); rng = np.random.default_rng(seed)
    if len(v) < 3:
        return float(v.mean()) if len(v) else np.nan, np.nan, np.nan
    b = rng.choice(v, (n, len(v))).mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def pair_means(x, pair):
    return {p: float(x[pair == p].mean()) for p in np.unique(pair)}


# ---- cross-check against stored scores
chk = []
for r in RUNS:
    d = D[r]; st = json.load(open(os.path.join(SCD, r, "score_select.json")))["amount"]["pixel"]["methods"]["model"]
    mine = {"n": len(d["pair"]), "ratio_of_sums": float(d["pred_model"].sum() / d["meas"].sum()), "abs_log_err": float(le(d["pred_model"], d["meas"]).mean())}
    # stored ratio_of_sums uses (pred+eps)/(meas+eps)-free sums; abs_log_err uses EPS 1e-3 as here
    ok = mine["n"] == st["n"] and abs(mine["abs_log_err"] - st["abs_log_err"]) < 1e-4 and abs(mine["ratio_of_sums"] - st["ratio_of_sums"]) < 1e-4
    chk.append({"run": r, "mine": mine, "stored": {k: st[k] for k in ("n", "ratio_of_sums", "abs_log_err")}, "match": bool(ok)})
    print("CHECK", r, mine, {k: st[k] for k in ("n", "ratio_of_sums", "abs_log_err")}, "OK" if ok else "MISMATCH", flush=True)
json.dump(chk, open(os.path.join(OUT, "b_crosscheck_vs_stored.json"), "w"), indent=1)
assert all(c["match"] for c in chk), "collection does not reproduce the stored score file"
rows = []

# ---- B5: view angle
EDG = [0, 10, 20, 30, 40, 50, 90]; LAB = ["0-10", "10-20", "20-30", "30-40", "40-50", ">=50"]
fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0)); x = np.arange(len(LAB))
for r in RUNS:
    d = D[r]; c, mk, nm = COL[fam(r)]; sd = int(r[-1]); b = np.digitize(d["vza"], EDG[1:-1]); out = {k: [] for k in ("err", "err_u", "err_e", "ratio", "t1m", "t1p", "n", "np", "psz")}
    for i in range(len(LAB)):
        m_ = b == i; n_ = int(m_.sum()); out["n"].append(n_)
        if n_ < 10:
            for k in out:
                if k not in ("n",):
                    out[k].append(np.nan)
            continue
        out["np"].append(int(len(np.unique(d["pair"][m_])))); out["err"].append(float(le(d["pred_model"][m_], d["meas"][m_]).mean())); out["err_u"].append(float(le(d["pred_uniform"][m_], d["meas"][m_]).mean()))
        out["err_e"].append(float(le(d["pred_excess_only"][m_], d["meas"][m_]).mean())); out["ratio"].append(float(d["pred_model"][m_].sum() / d["meas"][m_].sum())); out["psz"].append(float(d["psize_scan"][m_].mean()))
        v = m_ & (d["ncell"] >= 3) & (d["meas"] > 1e-9) & (d["pred_model"] > 1e-9); out["t1m"].append(float(d["top1_meas"][v].mean())); out["t1p"].append(float(d["top1_model"][v].mean()))
        rows.append({"exp": "B5", "run": r, "bin": LAB[i], "n_items": n_, "n_pairs": out["np"][-1], "model_abs_log_err": out["err"][-1], "even_spread_abs_log_err": out["err_u"][-1], "excess_only_abs_log_err": out["err_e"][-1], "model_ratio_of_sums": out["ratio"][-1],
                     "top1_share_model": out["t1p"][-1], "top1_share_measured": out["t1m"][-1], "mean_scan_pixel_km": out["psz"][-1]})
    axs[0].plot(x, out["err"], color=c, marker=mk, ls="-" if sd == 0 else "--", label=f"{nm} s{sd}", ms=4)
    if sd == 0 and r == RUNS[0]:
        axs[0].plot(x, out["err_u"], color="0.2", marker="x", ls=":", label="Even spread"); axs[0].plot(x, out["err_e"], color="0.5", marker="+", ls=":", label="Oracle-background spread")
        axs[3].bar(x, out["n"], color="0.7"); [axs[3].text(i, n_, f"{p_:.1f}", ha="center", va="bottom", fontsize=6) for i, (n_, p_) in enumerate(zip(out["n"], out["psz"])) if not np.isnan(p_)]
    axs[1].plot(x, out["ratio"], color=c, marker=mk, ls="-" if sd == 0 else "--", ms=4)
    axs[2].plot(x, out["t1p"], color=c, marker=mk, ls="-" if sd == 0 else "--", ms=4)
    if r == RUNS[0]:
        axs[2].plot(x, out["t1m"], color="k", marker="o", ls=":", ms=3, label="measured VIIRS")
for ax, t in zip(axs, ["amount error per MODIS pixel\n(mean |ln ratio|, lower better)", "predicted / measured energy (1 = right)", "hottest-cell share of the pixel's excess", "items per bin\n(label: mean scan pixel size, km)"]):
    ax.set_title(t, fontsize=7); ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7); ax.set_xlabel("MODIS view zenith angle (deg)"); ax.grid(alpha=.3)
axs[1].axhline(1, color="0.4", ls=":", lw=.8); axs[0].legend(fontsize=5.5); axs[2].legend(fontsize=5.5)
fig.suptitle("B5. Does the model degrade toward the swath edge? (best-val checkpoints, solid s0, dashed s1, nf2 SELECT fire pixels)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "b5_view_angle.pdf")); fig.savefig(os.path.join(OUT, "b5_view_angle.png"), dpi=200); plt.close(fig)

# ---- B6: fire size (bins of measured VIIRS energy in the pixel, and of the MODIS-input excess); gain = Oracle-background-spread error minus model error, pair-clustered bootstrap
def bins_q(v, k=6):
    return np.unique(np.quantile(v, np.linspace(0, 1, k + 1)))


fig, axs = plt.subplots(2, 3, figsize=(11.5, 5.2)); base = D[RUNS[0]]
for ri, (key, title) in enumerate((("meas", "measured VIIRS energy in the pixel (the truth)"), ("mexc", "MODIS excess radiance of the pixel (the input)"))):
    edg = bins_q(base[key], 6); xs = np.arange(len(edg) - 1); labs = [f"{edg[i]:.2g}-{edg[i+1]:.2g}" for i in range(len(edg) - 1)]
    for r in RUNS:
        d = D[r]; c, mk, nm = COL[fam(r)]; sd = int(r[-1]); b = np.clip(np.digitize(d[key], edg[1:-1]), 0, len(edg) - 2); g_, lo_, hi_, rt, er = [], [], [], [], []
        for i in range(len(edg) - 1):
            m_ = b == i; gm = le(d["pred_excess_only"][m_], d["meas"][m_]) - le(d["pred_model"][m_], d["meas"][m_]); mu, lo, hi = boot(pair_means(gm, d["pair"][m_]))
            g_.append(mu); lo_.append(lo); hi_.append(hi); rt.append(float(d["pred_model"][m_].sum() / d["meas"][m_].sum())); er.append(float(le(d["pred_model"][m_], d["meas"][m_]).mean()))
            rows.append({"exp": "B6", "run": r, "binned_by": key, "bin": labs[i], "n_items": int(m_.sum()), "n_pairs": int(len(np.unique(d["pair"][m_]))), "gain_vs_excess_only": mu, "gain_lo": lo, "gain_hi": hi, "model_ratio_of_sums": rt[-1], "model_abs_log_err": er[-1],
                         "excess_only_abs_log_err": float(le(d["pred_excess_only"][m_], d["meas"][m_]).mean()), "even_spread_abs_log_err": float(le(d["pred_uniform"][m_], d["meas"][m_]).mean())})
        off = (0.08 if sd else -0.08) + (0.2 if "1000" in r else -0.2) * 0.5
        axs[ri, 0].errorbar(xs + off, g_, yerr=[np.array(g_) - np.array(lo_), np.array(hi_) - np.array(g_)], color=c, marker=mk, ls="none", ms=4, capsize=2, mfc=c if sd == 0 else "white", label=f"{nm} s{sd}")
        axs[ri, 1].plot(xs, rt, color=c, marker=mk, ls="-" if sd == 0 else "--", ms=4); axs[ri, 2].plot(xs, er, color=c, marker=mk, ls="-" if sd == 0 else "--", ms=4)
    d0 = base; b0 = np.clip(np.digitize(d0[key], edg[1:-1]), 0, len(edg) - 2)
    axs[ri, 2].plot(xs, [float(le(d0["pred_excess_only"][b0 == i], d0["meas"][b0 == i]).mean()) for i in xs], color="0.5", marker="+", ls=":", label="Oracle-background spread"); axs[ri, 2].plot(xs, [float(le(d0["pred_uniform"][b0 == i], d0["meas"][b0 == i]).mean()) for i in xs], color="0.2", marker="x", ls=":", label="Even spread")
    for ax, t in zip(axs[ri], ["gain over Oracle-background spread (error of Oracle-background spread minus model; 95% pair range)", "predicted / measured energy", "mean |ln ratio| (lower better)"]):
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=30, fontsize=6); ax.set_xlabel(f"bin of {title}", fontsize=6); ax.set_title(t, fontsize=7); ax.grid(alpha=.3)
    axs[ri, 0].axhline(0, color="0.4", ls="--", lw=.8); axs[ri, 1].axhline(1, color="0.4", ls=":", lw=.8)
axs[0, 0].legend(fontsize=5.5); axs[0, 2].legend(fontsize=5.5)
fig.suptitle("B6. Where does the gain come from? Fire-size bins (6 equal-count bins; top row binned by the measured energy, bottom row by the MODIS input), nf2 SELECT, (best-val) checkpoints", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "b6_fire_size.pdf")); fig.savefig(os.path.join(OUT, "b6_fire_size.png"), dpi=200); plt.close(fig)

# ---- B7: region (stored region-balanced block-ranking gain over even spread, all seeds present, + collected amount gain)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.6)); rg = sorted({int(k) for k in np.unique(D[RUNS[0]]["region"])})
names = [reg.get(k, str(k)) for k in rg]; ys = np.arange(len(rg))
for r in RUNS:
    d = D[r]; c, mk, nm = COL[fam(r)]; sd = int(r[-1]); mu = []; lo = []; hi = []
    for k in rg:
        m_ = d["region"] == k; gm = le(d["pred_excess_only"][m_], d["meas"][m_]) - le(d["pred_model"][m_], d["meas"][m_]); a, l, h = boot(pair_means(gm, d["pair"][m_])) if m_.sum() >= 10 else (np.nan,) * 3
        mu.append(a); lo.append(l); hi.append(h)
        rows.append({"exp": "B7", "run": r, "region": reg.get(k, str(k)), "n_items": int(m_.sum()), "n_pairs": int(len(np.unique(d["pair"][m_]))), "gain_vs_excess_only": a, "gain_lo": l, "gain_hi": h})
    off = (0.12 if sd else -0.12) + (0.0)
    axs[0].errorbar(mu, ys + off + (0.2 if "1000" in r else -0.0), xerr=[np.array(mu) - np.array(lo), np.array(hi) - np.array(mu)], color=c, marker=mk, ls="none", ms=4, capsize=2, mfc=c if sd == 0 else "white")
axs[0].axvline(0, color="0.4", ls="--", lw=.8); axs[0].set_yticks(ys); axs[0].set_yticklabels([f"{n_} (n={int((D[RUNS[0]]['region'] == k).sum())})" for n_, k in zip(names, rg)], fontsize=6); axs[0].invert_yaxis()
axs[0].set_title("pixel amount: gain over Oracle-background spread per region (95% pair range)\npositive = model better; filled s0, hollow s1; orange Gamma-BW1000, green Gamma-BW100", fontsize=7)
for r in RUNS:
    j = json.load(open(os.path.join(SCD, r, "score_select.json")))["summary"]["balanced_vs"]["uniform"]["per_region"]; c, mk, nm = COL[fam(r)]; sd = int(r[-1])
    axs[1].scatter([j.get(str(k), np.nan) for k in rg], ys + (0.12 if sd else -0.12), color=c, marker=mk, s=22, facecolors=c if sd == 0 else "white")
    for k in rg:
        rows.append({"exp": "B7", "run": r, "region": reg.get(k, str(k)), "stored_block_auc_gain_over_even_spread": j.get(str(k), np.nan)})
axs[1].axvline(0, color="0.4", ls="--", lw=.8); axs[1].set_yticks(ys); axs[1].set_yticklabels([]); axs[1].invert_yaxis(); axs[1].grid(alpha=.3); axs[0].grid(alpha=.3)
axs[1].set_title("block-ranking AUC gain over Even spread\n(stored; region mean over episodes)", fontsize=7)
fig.suptitle("B7. Performance by region of the SELECT tiles (best-val; Gamma-BW100 seeds 0-1, Gamma-BW1000 seeds 0-3)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "b7_region.pdf")); fig.savefig(os.path.join(OUT, "b7_region.png"), dpi=200); plt.close(fig)
keys = sorted({k for r in rows for k in r}); 
with open(os.path.join(OUT, "b5_b7_tables.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
print("B done")
