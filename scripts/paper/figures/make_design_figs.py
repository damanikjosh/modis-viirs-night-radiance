#!/usr/bin/env python
"""Design-decision figures (vector PDF + 300 dpi PNG, Okabe-Ito colours, markers differ so they read in grey). Every value comes from
results/design/*.csv (recorded outputs of scripts/design/ on TRAIN and SELECT tiles; the literature rates of the front figure are a
compiled table). Usage: python make_design_figs.py [--results results] [--out out/paper/figures]   (needs numpy, pandas, matplotlib)"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results")); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "paper", "figures"))
_a = ap.parse_args(); D = os.path.abspath(_a.results) + "/design"; OUT = os.path.abspath(_a.out); os.makedirs(OUT, exist_ok=True)
BL, VE, GR, PU, GY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close(fig)


def fig_size():
    A = pd.read_csv(f"{D}/size_A_overlaps.csv"); B = pd.read_csv(f"{D}/size_B_pairs.csv"); C = pd.read_csv(f"{D}/size_C_tiles.csv")
    a = A[(A.ptype == "aqua_snpp") & (A.dn == "night")].set_index("T").n; T = a.index.values
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(T, a / a[30], "o-", color=BL, ms=3, label="all night Aqua x S-NPP overlaps (CMR, 119 days)")
    ex = C[C["T"] <= 30]; ax.plot(ex["T"], ex.fire_tiles / ex.fire_tiles.iloc[-1], "s-", color=VE, ms=3, label="fire-bearing TRAIN+SELECT tiles (exact)")
    ax.plot(C["T"][C["T"] >= 30], C.est_fire_tiles[C["T"] >= 30] / ex.fire_tiles.iloc[-1], "s--", color=VE, ms=3, mfc="none", label="same, ESTIMATE beyond 30 min")
    b = B[B.enumerated]; ax.plot(b["T"], b.candidates / b.candidates.iloc[-1], "^-", color=GR, ms=3, label="night-fire candidate pairs (enumerated to 30 min)")
    ax.axvline(30, color=GY, lw=0.6, ls=":"); ax.axhline(1017 / 7462, color=GR, lw=0.6, ls=":"); ax.text(118, 1017 / 7462 + 0.03, "pairs actually used = 14 % of the pool", ha="right", color=GR, fontsize=6)
    ax.set_xlabel("window: |time gap| <= T (minutes)"); ax.set_ylabel("share of the 30-minute amount"); ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 15, 30, 60, 120]); ax.set_xticklabels(["2", "5", "10", "15", "30", "60", "120"]); ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    save(fig, "fig_design_size")


def fig_agree():
    t = pd.read_csv(f"{D}/an/bins6_all.csv"); x = (t.gap_lo + np.minimum(t.gap_hi, 30)) / 2
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.0))
    for a, col, lab, c in zip(ax, ("spearman", "no_viirs_fire_share", "abs_err_even"), ("Spearman: MODIS excess vs VIIRS energy", "MODIS fire pixels with no VIIRS fire cell", "Even-spread amount error (|ln ratio|)"), (BL, VE, GR)):
        a.errorbar(x, t[col], yerr=[t[col] - t[col + "_lo"], t[col + "_hi"] - t[col]], fmt="o", color=c, ms=3, lw=0.8, capsize=1.5); a.set_title(lab, fontsize=6.5, loc="left"); a.set_xlabel("|time gap| (min)"); a.set_xticks([2.5, 7.5, 12.5, 17.5, 22.5, 27.5]); a.set_xticklabels(["0-5", "5-10", "10-15", "15-20", "20-25", "25-30"], fontsize=5.5)
    fig.tight_layout(); save(fig, "fig_design_agree")


def fig_front():
    f = pd.read_csv(f"{D}/front_spread.csv"); f["lo"] = f.rate_lo_m_per_h * 0.5; f["hi"] = f.rate_hi_m_per_h * 0.5
    fig, ax = plt.subplots(figsize=(3.4, 2.4)); y = np.arange(len(f))[::-1]
    for yi, lo, hi in zip(y, f.lo, f.hi):
        ax.plot([max(lo, 0.003), hi], [yi, yi], color=BL, lw=5, solid_capstyle="butt")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(f.fire_type, fontsize=6); ax.axvline(375, color=VE, lw=1, ls="--"); ax.axvline(1000, color=GR, lw=1, ls="-.")
    ax.text(375, len(f) - 0.4, "VIIRS 375 m", color=VE, fontsize=6, ha="right"); ax.text(1000, len(f) - 0.4, "MODIS 1 km", color=GR, fontsize=6, ha="left")
    ax.set_xlabel("distance a front moves in 30 min (m, log scale)"); save(fig, "fig_design_front")


def fig_scan():
    s = pd.read_csv(f"{D}/scan_dist.csv"); fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.0), sharey=True)
    for a, sen in zip(ax, ("modis", "viirs")):
        d = s[(s.sensor == sen) & (s.scan_lo < 55)]; a.bar(d.scan_lo + 2.5, d.observed_share, 4, color=BL if sen == "modis" else VE, label="paired TRAIN+SELECT tiles"); a.plot(d.scan_lo + 2.5, d.uniform_ground_share, "k_", ms=7, mew=1.2, label="uniform over the ground swath")
        a.set_title("Aqua-MODIS" if sen == "modis" else "S-NPP-VIIRS", fontsize=6.5, loc="left"); a.set_xlabel("scan angle at tile centre (deg)")
    ax[0].set_ylabel("share of tiles"); ax[0].legend(fontsize=5.5, frameon=False); fig.tight_layout(); save(fig, "fig_design_scan")


def fig_modelgap():
    fs = sorted(f for f in os.listdir(f"{D}/an") if f.startswith("model_") and f.endswith("_bins3.csv"))
    if not fs: return
    t = pd.read_csv(f"{D}/an/{fs[0]}"); fig, ax = plt.subplots(figsize=(3.4, 2.4)); x = np.arange(len(t))
    ax.errorbar(x - 0.1, t.abs_err_even, yerr=[t.abs_err_even - t.abs_err_even_lo, t.abs_err_even_hi - t.abs_err_even], fmt="o", color=VE, ms=3, lw=0.8, capsize=1.5, label="Even spread")
    ax.errorbar(x + 0.1, t.abs_err_model, yerr=[t.abs_err_model - t.abs_err_model_lo, t.abs_err_model_hi - t.abs_err_model], fmt="s", color=BL, ms=3, lw=0.8, capsize=1.5, label="model")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(a)}-{int(round(b))}" for a, b in zip(t.gap_lo, t.gap_hi)]); ax.set_xlabel("|time gap| (min), SELECT"); ax.set_ylabel("amount error |ln ratio|"); ax.legend(frameon=False, fontsize=6); save(fig, "fig_design_modelgap")


if __name__ == "__main__":
    for f in (fig_size, fig_agree, fig_front, fig_scan, fig_modelgap):
        f()
