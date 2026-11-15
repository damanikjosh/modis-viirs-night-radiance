#!/usr/bin/env python
"""Paper experiments A1-A4 (POST-HOC, nothing re-scored): from the stored score files (results/raw/{select,old}), the per-epoch in-training readings
(results/raw/<run>/log.jsonl, diag.proto; the reading before training, epoch 0, is not in the log) and the descriptive-measures table (results/raw/_descriptive).
SELECT (nf2, 2,645 tiles) and old 9-pair night validation only. Usage: a_stored.py <results dir> <out dir (figures+tables)>"""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
SC = os.path.join(RES, "raw"); DESC = os.path.join(SC, "_descriptive", "descriptive_measures.csv")
SET = [("oldloss-nf2", "Group-share", "#D55E00", "o"), ("gamma-nf2", "Gamma", "#0072B2", "s"), ("gammabw100-nf2", "Gamma-BW100", "#009E73", "^"),
       ("gammabw1000-nf2", "Gamma-BW1000", "#E69F00", "D"), ("gammabw100-d05-nf2", "Gamma-BW100-B0.5", "#56B4E9", "v"), ("gammabw1000-d05-nf2", "Gamma-BW1000-B0.5", "#CC79A7", "P")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})


def load(sub, pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(SC, sub, "*", "score_*.json"))):
        j = json.load(open(f))
        if re.fullmatch(pat, j["run"]):
            out[j["run"]] = j
    return out


def q(j):
    """The six numbers under the pre-stated lines (continuous values; the pass bands are in the text of the plot)."""
    s, am = j["summary"], j["amount"]["block3"]["methods"]; auc = lambda m: s["auc"][m]["mean"]
    best = max(auc("uniform"), auc("excess_only"))
    return {"block gain over best reference (AUC)": auc("model") - best, "energy ratio, MODIS-selected (pass 0.8-1.25)": s["energy_msel"]["model"],
            "flag count ratio (pass 0.5-2)": s["rule"]["model"]["flag_count_ratio"], "hottest-cell share (pass >= 0.33)": s["conc"]["model"]["top1_share"],
            "3 km amount error: Oracle-background spread minus model (log, higher = model better)": am["excess_only"]["abs_log_err"] - am["model"]["abs_log_err"]}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200); plt.close(fig)


def fam_seed(run):
    m = re.fullmatch(r"(.+)-s(\d+)(-final)?", run); return m.group(1), int(m.group(2)), bool(m.group(3))


# ------------------------------------------------------------------ A1
sel = load("select", r"(oldloss|gamma|gammabw100|gammabw1000|gammabw100-d05|gammabw1000-d05)-nf2-s\d(-final)?")
rows = []; vals = {}
for run, j in sel.items():
    fam, sd, fin = fam_seed(run)
    for k, v in q(j).items():
        vals.setdefault((k, fin), {}).setdefault(fam, {})[sd] = v; rows.append({"run": run, "family": fam, "seed": sd, "checkpoint": "final" if fin else "best_val", "quantity": k, "value": v, "epoch": j["epoch"]})
qs = list(q(next(iter(sel.values()))).keys())
fig, axs = plt.subplots(1, len(qs), figsize=(11.5, 3.0), sharey=True)
for ax, k in zip(axs, qs):
    for i, (fam, lab, c, mk) in enumerate(SET):
        for fin, dy, fc in ((False, -0.12, c), (True, 0.12, "white")):
            v = vals.get((k, fin), {}).get(fam, {})
            ax.scatter(list(v.values()), [i + dy] * len(v), s=26, marker=mk, facecolor=fc, edgecolor=c, linewidth=1.2, zorder=3)
    ax.set_title(k.split(" (")[0].replace("3 km amount error: Oracle-background spread minus model", "3 km amount gain"), fontsize=7)
    for lo_, hi_ in {"energy": (0.8, 1.25), "flag": (0.5, 2.0)}.items() if False else []:
        pass
    ax.grid(axis="x", alpha=.3)
for ax, (lo, hi) in zip(axs, [(0.05, None), (0.8, 1.25), (0.5, 2.0), (0.33, None), (0.0, None)]):
    for x in (lo, hi):
        if x is not None:
            ax.axvline(x, color="0.4", ls="--", lw=.8)
axs[0].set_yticks(range(len(SET))); axs[0].set_yticklabels([s[1] for s in SET]); axs[0].invert_yaxis()
fig.suptitle("A1. Seed-to-seed spread vs differences between settings (filled: (best-val), hollow: (final); dashed: pass line; one dot = one seed)", fontsize=8)
fig.tight_layout(); save(fig, "a1_seed_spread")
tab = []
for (k, fin), d in sorted(vals.items(), key=lambda t: (t[0][1], t[0][0])):
    fams = [f for f, *_ in SET if f in d]; rng = {f: (min(d[f].values()), max(d[f].values())) for f in fams}
    sp = {f: rng[f][1] - rng[f][0] for f in fams}; pairs = []; sep = 0; tot = 0
    for ia, a in enumerate(fams):
        for b in fams[ia + 1:]:
            tot += 1; s_ = rng[a][1] < rng[b][0] or rng[b][1] < rng[a][0]; sep += s_
            pairs.append((a, b, float(np.mean(list(d[a].values())) - np.mean(list(d[b].values()))), bool(s_)))
    tab.append({"quantity": k, "checkpoint": "final" if fin else "best_val", "mean_seed_range": float(np.mean(list(sp.values()))), "n_seeds": {f: len(d[f]) for f in fams}, "setting_pairs": tot, "pairs_separated_by_more_than_seed_range": int(sep), "gamma_family_pairs": sum(1 for p_ in pairs if "oldloss-nf2" not in (p_[0], p_[1])), "gamma_family_pairs_separated": sum(1 for p_ in pairs if "oldloss-nf2" not in (p_[0], p_[1]) and p_[3]),
                "ranges": {f: [float(rng[f][0]), float(rng[f][1])] for f in fams}, "pairs": [{"a": a, "b": b, "mean_diff": x, "separated": s_} for a, b, x, s_ in pairs]})
json.dump({"rows": rows, "summary": tab}, open(os.path.join(OUT, "a1_seed_spread.json"), "w"), indent=1)
with open(os.path.join(OUT, "a1_seed_spread.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ A2 (per-epoch in-training SELECT half-sample, diag.jsonl 'proto')
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.0)); curves = []
runs = {}
for f in sorted(glob.glob(os.path.join(SC, "*-nf2-s?", "log.jsonl"))):
    run = os.path.basename(os.path.dirname(f)); L = []
    for l in open(f):
        r = json.loads(l)
        if (r.get("diag") or {}).get("proto"):
            L.append({"epoch": r["epoch"] + 1, "step": r["step"], "proto": r["diag"]["proto"]})  # log epoch is 0-based; score files count completed epochs
    runs[run] = L
fin_ep = {}
for run in runs:
    fam, sd, _ = fam_seed(run); lab = next(s for s in SET if s[0] == fam)
    ep = [r["epoch"] for r in runs[run]]
    y = [(r["proto"]["auc_model"] - r["proto"]["auc_uniform"], r["proto"]["energy_msel"], r["proto"]["top1_model"]) for r in runs[run]]
    sj = sel.get(run); fj = sel.get(run + "-final"); be = sj["epoch"] if sj else None; fe = fj["epoch"] if fj else None
    for a, ax in enumerate(axs):
        ax.plot(ep, [t[a] for t in y], color=lab[2], lw=1, alpha=.8, ls="-" if sd == 0 else "--")
        for e_, mk, fc in ((be, lab[3], lab[2]), (fe, lab[3], "white")):
            if e_ is not None and e_ in ep:
                ax.scatter([e_], [y[ep.index(e_)][a]], s=28, marker=mk, facecolor=fc, edgecolor=lab[2], zorder=4)
    for r, t in zip(runs[run], y):
        curves.append({"run": run, "epoch": r["epoch"], "step": r["step"], "auc_gain_over_even_spread": t[0], "energy_ratio_msel": t[1], "hottest_cell_share": t[2], "n_pairs": r["proto"]["n_pairs"]})
for ax, t, ref in zip(axs, ["block ranking gain over Even spread (AUC)", "energy ratio, MODIS-selected (1 = right)", "hottest-cell share (VIIRS: ~0.46)"], [0.0, 1.0, None]):
    ax.set_title(t, fontsize=8); ax.set_xlabel("epoch (1 epoch = 647 steps)"); ax.grid(alpha=.3)
    if ref is not None:
        ax.axhline(ref, color="0.4", ls=":", lw=.8)
axs[2].legend(handles=[plt.Line2D([], [], color=c, marker=m, ls="", label=l) for _, l, c, m in SET], fontsize=6, loc="lower right")
fig.suptitle("A2. Learning curves on the in-training SELECT half-sample (seed 0 solid, seed 1 dashed; filled marker = (best-val), hollow = (final), as scored on full SELECT)", fontsize=8)
fig.tight_layout(); save(fig, "a2_learning_curves")
with open(os.path.join(OUT, "a2_learning_curves.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(curves[0].keys())); w.writeheader(); w.writerows(curves)
# best vs final on FULL select (the trade-off): per family mean over seeds
bf = {}
for run, j in sel.items():
    fam, sd, fin = fam_seed(run)
    bf.setdefault(fam, {}).setdefault("final" if fin else "best_val", []).append({k: v for k, v in q(j).items()} | {"epoch": j["epoch"]})
json.dump(bf, open(os.path.join(OUT, "a2_best_vs_final.json"), "w"), indent=1)

# ------------------------------------------------------------------ A3 small data (ref-*) vs multi-continent data (same loss), nf2 SELECT and old validation
pairs3 = [("Group-share", "ref-legacy", "oldloss-nf2", "#D55E00"), ("Gamma", "ref-gamma", "gamma-nf2", "#0072B2")]
fig, axs = plt.subplots(2, len(qs), figsize=(11.5, 4.6)); out3 = []
for r_, (setn, sub, pat) in enumerate([("nf2 SELECT (2,645 tiles)", "select", ""), ("old 9-pair night validation (322 tiles)", "old", "")]):
    S_ = load(sub, r"(ref-legacy|ref-gamma|oldloss-nf2|gamma-nf2)-s\d")
    for c_, k in enumerate(qs):
        ax = axs[r_, c_]
        for i, (lab, small, big, col) in enumerate(pairs3):
            for j_, (nm, dy, mk) in enumerate(((small, -0.1, "o"), (big, 0.1, "s"))):
                v = []
                for run, j in S_.items():
                    if run.startswith(nm + "-s"):
                        try:
                            if sub == "select":
                                v.append(q(j)[k])
                            else:
                                s_, am = j["summary"], j["amount"]["block3"]
                                v.append({qs[0]: s_["auc"]["model"]["mean"] - max(s_["auc"]["uniform"]["mean"], s_["auc"]["excess_only"]["mean"]), qs[1]: s_["energy_msel"]["model"], qs[2]: s_["rule"]["model"]["flag_count_ratio"], qs[3]: s_["conc"]["model"]["top1_share"],
                                          qs[4]: (am["methods"]["excess_only"]["abs_log_err"] - am["methods"]["model"]["abs_log_err"]) if am else np.nan}[k])
                        except Exception as e:
                            v.append(np.nan)
                ax.scatter(v, [i + dy] * len(v), s=26, marker=mk, facecolor=col if j_ else "white", edgecolor=col, linewidth=1.2, zorder=3)
                for x in v:
                    out3.append({"set": setn, "loss": lab, "data": "multi-continent (nf2)" if j_ else "small list", "quantity": k, "value": float(x)})
        ax.set_yticks(range(2)); ax.set_yticklabels([p[0] for p in pairs3] if c_ == 0 else []); ax.invert_yaxis(); ax.grid(axis="x", alpha=.3)
        if r_ == 0:
            ax.set_title(k.split(" (")[0].replace("3 km amount error: Oracle-background spread minus model", "3 km amount gain"), fontsize=7)
    axs[r_, 0].set_ylabel(setn.split(" (")[0], fontsize=7)
fig.suptitle("A3. Small data list (hollow circles) vs multi-continent data (filled squares), same loss, (best-val), 2 seeds each", fontsize=8)
fig.tight_layout(); save(fig, "a3_small_vs_multi")
with open(os.path.join(OUT, "a3_small_vs_multi.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(out3[0].keys())); w.writeheader(); w.writerows(out3)

# ------------------------------------------------------------------ A4 learned vs non-learning, all measures (descriptive-measures table + score files)
R = list(csv.DictReader(open(DESC)))
meas = [("3km_block", "auc_block", "block ranking (AUC)", True), ("3km_block", "csi", "fire blocks found without inventing (CSI)", True), ("3km_block", "hit_rate", "block hit rate", True),
        ("3km_block", "false_alarm_share", "false-alarm share (lower better)", False), ("3km_block", "flag_count_ratio", "flag count ratio (1 = right)", None), ("3km_block", "amount_error", "3 km amount error (log, lower better)", False),
        ("3km_block", "amount_skill_vs_even", "3 km amount: % better than Even spread", True), ("modis_pixel", "hottest_cell_share", "hottest-cell share (VIIRS: ~0.46)", True)]
FAMS = [("oldloss-nf2", "Group-share"), ("gamma-nf2", "Gamma"), ("gammabw100-nf2", "Gamma-BW100"), ("gammabw1000-nf2", "Gamma-BW1000")]
SIMPLE = [("uniform", "Even spread"), ("excess_only", "Oracle-background spread"), ("copy_planck", "copy (Planck)"), ("copy_raw", "copy raw")]
tab4 = []
fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.2)); axs = axs.ravel()
for ax, (lvl, m, title, hib) in zip(axs, meas):
    for si, st in enumerate(("select", "oldval")):
        for r in R:
            if r["set"] != st or r["measure"] != m or r["level"] != lvl or r["value"] == "":
                continue
            v = float(r["value"]); y = si * 0.0
            if r["subject"] in dict(SIMPLE):
                k = [s_[0] for s_ in SIMPLE].index(r["subject"]); tab4.append({"set": st, "kind": "non-learning", "name": r["subject"], "measure": m, "value": v, "n_runs": 1})
                if st == "select":
                    ax.scatter([v], [4 + (k - 1.5) * 0.15], marker="x", color="0.25", s=24, zorder=2); ax.annotate("UOPR"[k], (v, 4 + (k - 1.5) * 0.15), fontsize=6, xytext=(3, 2), textcoords="offset points")
            elif r["subject"] == "model" and r["checkpoint"] == "best_validation" and re.fullmatch(r"(oldloss|gamma|gammabw100|gammabw1000)-nf2-s\d", r["run"]):
                i = [f for f, _ in FAMS].index(r["family"]); col, mk = next((c, mk) for f, _, c, mk in SET if f == r["family"])
                if st == "select":
                    ax.scatter([v], [i], color=col, marker=mk, s=26, zorder=3)
                tab4.append({"set": st, "kind": "learned", "name": r["run"], "measure": m, "value": v, "n_runs": 1})
    ax.set_yticks(list(range(4)) + [4]); ax.set_yticklabels([n for _, n in FAMS] + ["non-learning (x)"] if ax in axs[[0, 4]] else []); ax.invert_yaxis(); ax.set_title(title, fontsize=7); ax.grid(axis="x", alpha=.3)
fig.suptitle("A4. Learned (best-val, one dot per seed) vs non-learning methods (x; U Even spread, O Oracle-background spread, P copy-Planck, R copy-raw), nf2 SELECT", fontsize=8)
fig.tight_layout(); save(fig, "a4_learned_vs_simple")
# table: families (mean of seeds, SELECT, best_val) vs simple, all measures incl. energy ratio from the score files
T = {}
for r in R:
    if r["set"] == "select" and r["value"] != "" and r["level"] in ("3km_block", "modis_pixel"):
        if r["subject"] in dict(SIMPLE):
            T.setdefault(r["subject"], {})[(r["level"], r["measure"])] = float(r["value"])
        elif r["subject"] == "model" and r["checkpoint"] == "best_validation" and re.fullmatch(r"(oldloss|gamma|gammabw100|gammabw1000)-nf2-s\d", r["run"]):
            T.setdefault(r["family"], {}).setdefault((r["level"], r["measure"]), []).append(float(r["value"]))
for f, fe in ((f, e) for f, _ in FAMS for e in [[q_ for run, j in sel.items() if fam_seed(run)[0] == f and not fam_seed(run)[2] for q_ in [j["summary"]["energy_msel"]["model"]]]]):
    T[f][("energy", "energy_ratio_msel")] = fe
bj = next(iter(load("select", r"baselines").values()), None)
if bj:
    for s_ in ("uniform", "excess_only", "copy_planck", "copy_raw"):
        T.setdefault(s_, {})[("energy", "energy_ratio_msel")] = bj["summary"]["energy_msel"][s_]
with open(os.path.join(OUT, "a4_learned_vs_simple.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(tab4[0].keys())); w.writeheader(); w.writerows(tab4)
json.dump({f"{k}": {f"{a}|{b}": v for (a, b), v in d.items()} for k, d in T.items()}, open(os.path.join(OUT, "a4_learned_vs_simple.json"), "w"), indent=1)
print("A done", OUT)
