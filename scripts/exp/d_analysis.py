#!/usr/bin/env python
"""Experiment D (POST-HOC, short budget): figures + tables from the guarded scores by pairing-window bin. usage: d_analysis.py <scoresD dir (<run>/score_select_dt*.json ...)> <old scores for arm d: scores dir> <subsets.json> <out dir>
Arms: a <=5 min (N=449, 97 fire tiles), b <=15 min (same N, F), c <=30 min (same N, F), d <=30 min ALL 2,340 tiles (= Gamma-BW100 (short) of experiment C); candidate = frozen gammabw100-nf2-s0 (best-val)."""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SD, SC, SUB, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True); info = json.load(open(SUB))
ARMS = [("a", "(a) <=5 min, 449 tiles", "#0072B2", "s"), ("b", "(b) <=15 min, 449 tiles", "#009E73", "^"), ("c", "(c) <=30 min, 449 tiles", "#E69F00", "D"), ("d", "(d) <=30 min, all 2,340 tiles", "#CC79A7", "P")]
RUN = {"a": "dabl-a-s{}", "b": "dabl-b-s{}", "c": "dabl-c-s{}", "d": "abl-bw100-s{}"}
SETS = [("all", "SELECT, all |dt|", "score_select.json"), ("0-5", "SELECT tiles |dt| <= 5 min", "score_select_dt0-5.json"), ("5-15", "SELECT tiles 5 < |dt| <= 15", "score_select_dt5-15.json"), ("15-30", "SELECT tiles 15 < |dt| <= 30", "score_select_dt15-30.json")]
QS = ["block AUC gain over Even spread", "energy ratio, MODIS-selected", "flag count ratio", "hottest-cell share", "3 km amount gain over Even spread"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
def load(run, fn):
    for d in (SD, SC):
        f = os.path.join(d, run, fn)
        if os.path.exists(f):
            return json.load(open(f))
def q(j):
    s = j["summary"]; am = (j["amount"].get("block3") or {}); m = am.get("methods") or {}
    return {QS[0]: s["auc"]["model"]["mean"] - s["auc"]["uniform"]["mean"], QS[1]: s["energy_msel"]["model"], QS[2]: s["rule"]["model"]["flag_count_ratio"], QS[3]: s["conc"]["model"]["top1_share"],
            QS[4]: (m["uniform"]["abs_log_err"] - m["model"]["abs_log_err"]) if m else np.nan, "n_tiles": j["n_tiles"], "n_items_block3": am.get("n_items"), "n_pairs_block3": am.get("n_pairs_total"), "auc_model": s["auc"]["model"]["mean"], "auc_even": s["auc"]["uniform"]["mean"]}
rows = []
fig, axs = plt.subplots(len(SETS), len(QS), figsize=(11.5, 8.6), sharey=True)
for r_, (sk, st, fn) in enumerate(SETS):
    for ai, (arm, al, col, mk) in enumerate(ARMS):
        for sd in (0, 1):
            j = load(RUN[arm].format(sd), fn)
            if j is None:
                continue
            v = q(j); rows.append({"arm": arm, "seed": sd, "set": sk, **v})
            for c_, qn in enumerate(QS):
                axs[r_, c_].scatter([v[qn]], [ai], marker=mk, color=col, facecolor=col if sd == 0 else "white", s=30, zorder=3)
    for c_, qn in enumerate(QS):
        ax = axs[r_, c_]; ax.grid(axis="x", alpha=.3)
        if r_ == 0:
            ax.set_title(qn, fontsize=7)
    axs[r_, 0].set_ylabel(st.replace("SELECT tiles ", "").replace("SELECT, ", ""), fontsize=7)
    n_ = [r for r in rows if r["set"] == sk]
    if n_:
        axs[r_, -1].text(1.02, 0.5, f"n tiles {n_[0]['n_tiles']}\n3 km items {n_[0]['n_items_block3']}", transform=axs[r_, -1].transAxes, fontsize=6, va="center")
for r_ in range(len(SETS)):
    for c_, (lo, hi) in enumerate([(0, None), (0.8, 1.25), (0.5, 2.0), (0.33, None), (0, None)]):
        for x in (lo, hi):
            if x is not None and not (c_ in (0,) and False):
                axs[r_, c_].axvline(x, color="0.5", ls="--", lw=.7)
axs[0, 0].set_yticks(range(len(ARMS))); axs[0, 0].set_yticklabels([a[1] for a in ARMS], fontsize=7); axs[0, 0].invert_yaxis()
fig.suptitle("D. Does a tighter pairing window give a better model, at equal data and with all data? Gamma-BW100 (short, 1,750 steps, not converged); filled seed 0, hollow seed 1; dashed = pass line", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "d_pairing_window.pdf")); fig.savefig(os.path.join(OUT, "d_pairing_window.png"), dpi=200); plt.close(fig)
with open(os.path.join(OUT, "d_pairing_window.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# oldval overall
old = []
for arm, al, col, mk in ARMS:
    for sd in (0, 1):
        j = load(RUN[arm].format(sd), "score_oldval.json")
        if j: old.append({"arm": arm, "seed": sd, **q(j)})
if old:
    with open(os.path.join(OUT, "d_oldval.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(old[0].keys())); w.writeheader(); w.writerows(old)
# step 4: frozen candidate and Even spread by bin
cand = []
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.1)); xs = np.arange(3)
for bi, (sk, st, fn) in enumerate(SETS[1:]):
    j = load("gammabw100-nf2-s0", fn)
    if j is None:
        continue
    s = j["summary"]; m = (j["amount"].get("block3") or {}).get("methods") or {}
    rec = {"bin": sk, "n_tiles": j["n_tiles"], "n_items": (j["amount"]["block3"] or {}).get("n_items"), "auc_model": s["auc"]["model"]["mean"], "auc_even": s["auc"]["uniform"]["mean"], "auc_oracle_bg": s["auc"]["excess_only"]["mean"],
           "amount_err_model": m["model"]["abs_log_err"], "amount_err_even": m["uniform"]["abs_log_err"], "amount_err_oracle_bg": m["excess_only"]["abs_log_err"], "hottest_model": s["conc"]["model"]["top1_share"], "hottest_measured": s["conc"]["measured"]["top1_share"], "hottest_even": s["conc"]["uniform"]["top1_share"],
           "energy_ratio": s["energy_msel"]["model"], "flag_ratio": s["rule"]["model"]["flag_count_ratio"]}
    cand.append(rec)
if cand:
    for ax, (ks, t) in zip(axs, [(("amount_err_model", "amount_err_even", "amount_err_oracle_bg"), "3 km amount error (log, lower better)"), (("auc_model", "auc_even", "auc_oracle_bg"), "block ranking AUC (higher better)"), (("hottest_model", "hottest_even", "hottest_measured"), "hottest-cell share")]):
        for k, (lab, col, mk) in zip(ks, [("Gamma-BW100 (best-val), frozen s0", "#009E73", "^"), ("Even spread", "0.2", "x"), ("Oracle-background spread" if "oracle" in ks[2] else "measured VIIRS", "0.55", "+")]):
            ax.plot(xs, [c[k] for c in cand], marker=mk, color=col, label=lab, ls="-" if mk == "^" else ":")
        ax.set_xticks(xs); ax.set_xticklabels([f"{c['bin']} min\nn={c['n_tiles']} tiles" for c in cand], fontsize=7); ax.set_title(t, fontsize=8); ax.grid(alpha=.3)
    axs[0].legend(fontsize=6); fig.suptitle("D4. The frozen candidate and Even spread by pairing-time gap of the SELECT tile (no training)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "d4_candidate_by_dt.pdf")); fig.savefig(os.path.join(OUT, "d4_candidate_by_dt.png"), dpi=200); plt.close(fig)
    json.dump(cand, open(os.path.join(OUT, "d4_candidate_by_dt.json"), "w"), indent=1)
summ = {}
for r in rows:
    summ.setdefault(r["set"], {}).setdefault(r["arm"], []).append(r)
for sk in summ:
    for arm in summ[sk]:
        print(sk, arm, {qn[:14]: [round(float(x[qn]), 3) for x in summ[sk][arm]] for qn in QS}, summ[sk][arm][0]["n_tiles"])
print(json.dumps(cand, indent=0)[:1500])
