#!/usr/bin/env python
"""Experiment C (POST-HOC, short-budget): table + figure of the short-budget ablation runs (final step of 24 epochs = 1,752 steps, NOT converged) scored on nf2 SELECT with the guarded scorer.
usage: c_analysis.py <scores dir with <run>/score_select.json> <out dir>"""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SC, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
CFG = [("base", "Gamma (short)", "#0072B2", "s"), ("bw100", "Gamma-BW100 (short)", "#009E73", "^"), ("bw1000", "Gamma-BW1000 (short)", "#E69F00", "D"),
       ("noenergy", "Gamma, no energy term (short)", "#D55E00", "o"), ("novvmm", "Gamma, no vv/mm terms (short)", "#CC79A7", "P")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
def q(j):
    s, am = j["summary"], j["amount"]["block3"]["methods"]; auc = lambda m: s["auc"][m]["mean"]
    return {"block gain over best reference": auc("model") - max(auc("uniform"), auc("excess_only")), "energy ratio, MODIS-selected": s["energy_msel"]["model"], "flag count ratio": s["rule"]["model"]["flag_count_ratio"],
            "hottest-cell share": s["conc"]["model"]["top1_share"], "3 km amount gain over Oracle-background spread": am["excess_only"]["abs_log_err"] - am["model"]["abs_log_err"], "block AUC": auc("model"), "block AUC, Fire-off": auc("model_off")}
rows = []; vals = {}
for f in sorted(glob.glob(os.path.join(SC, "abl-*", "score_select.json"))):
    j = json.load(open(f)); m = re.fullmatch(r"abl-(\w+)-s(\d+)", j["run"]); cfg, sd = m.group(1), int(m.group(2))
    for k, v in q(j).items():
        vals.setdefault(k, {}).setdefault(cfg, {})[sd] = v; rows.append({"run": j["run"], "config": cfg, "seed": sd, "quantity": k, "value": v, "n_tiles": j["n_tiles"], "epoch": j.get("epoch")})
with open(os.path.join(OUT, "c_ablations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
qs = [k for k in vals if k not in ("block AUC", "block AUC, Fire-off")]
fig, axs = plt.subplots(1, len(qs), figsize=(11.5, 2.8), sharey=True)
present = [c for c in CFG if c[0] in vals[qs[0]]]
for ax, k in zip(axs, qs):
    for i, (cfg, lab, col, mk) in enumerate(present):
        v = vals[k].get(cfg, {}); ax.scatter(list(v.values()), [i] * len(v), s=28, marker=mk, color=col, zorder=3)
    ax.set_title(k.replace(" over Oracle-background spread", ""), fontsize=7); ax.grid(axis="x", alpha=.3)
for ax, (lo, hi) in zip(axs, [(0.05, None), (0.8, 1.25), (0.5, 2.0), (0.33, None), (0.0, None)]):
    for x in (lo, hi):
        if x is not None:
            ax.axvline(x, color="0.4", ls="--", lw=.8)
axs[0].set_yticks(range(len(present))); axs[0].set_yticklabels([c[1] for c in present]); axs[0].invert_yaxis()
fig.suptitle("C. Short-budget ablations (1,752 steps, not converged, small list, final step), nf2 SELECT; one dot = one seed; dashed = pass line", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c_ablations.pdf")); fig.savefig(os.path.join(OUT, "c_ablations.png"), dpi=200)
summ = {k: {cfg: {"mean": float(np.mean(list(v.values()))), "seeds": {str(s): x for s, x in v.items()}} for cfg, v in d.items()} for k, d in vals.items()}
json.dump(summ, open(os.path.join(OUT, "c_ablations.json"), "w"), indent=1); print(json.dumps({k: {c: round(v["mean"], 3) for c, v in d.items()} for k, d in summ.items()}, indent=1))
