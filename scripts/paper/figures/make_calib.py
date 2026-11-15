"""Fig. calib: 3 km calibration (SELECT set), predicted vs measured mean amount per bin, log-log, one thin line per seed,
best-validation runs of the four settings (bound 0.25); grey dashed = identity; dotted grey = even-spread curve.
Input: results/raw/_descriptive/calibration_bins.csv (columns run,family,seed,checkpoint,set,level,subject,bin,n,mean_pred,mean_meas,lo,hi)."""
from common import *

rows = [r for r in read_csv("_descriptive/calibration_bins.csv") if r["set"] == "select" and r["level"] == "3km_block"]
names = sorted({r["run"] for r in rows if r["subject"] == "model"})
sel = {n: classify(n) for n in names}
sel = {n: c for n, c in sel.items() if c and c["ckpt"] == "best" and c["bound"] == 0.25 and not c["fam"].startswith("ref")}
runs = runs_like(sel)
order = ["old", "gamma", "gammabw100", "gammabw1000"]

def curve(run, subject):
    pts = sorted([(int(r["bin"]), fnum(r["mean_pred"]), fnum(r["mean_meas"])) for r in rows if r["run"] == run and r["subject"] == subject])
    pts = [(p, m) for _, p, m in pts if p and m and p > 0 and m > 0]
    return [p for p, _ in pts], [m for _, m in pts]

allv = []
for n in sel:
    for sub in ("model", "uniform"):
        p, m = curve(n, sub); allv += p + m
lo, hi = min(allv) / 1.3, max(allv) * 1.3
fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 2.35), sharex=True, sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.26, wspace=0.08)
for ax, st in zip(axs, order):
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.7, ls="--", zorder=1)
    ns = [n for n in sel if setting_of(sel[n]) == st]
    drawn_even = False
    for n in sorted(ns, key=lambda n: sel[n]["seed"]):
        c = sel[n]
        # even-spread curve of the same scored set (stored per run; identical for all runs of the set)
        if not drawn_even:
            p, m = curve(n, "uniform")
            ax.plot(p, m, color=GREY, lw=0.8, ls=":", zorder=2)
            drawn_even = True
        p, m = curve(n, "model")
        ax.plot(p, m, color=SETTING_COLOR[st], lw=0.8, marker=seed_marker(c["seed"]), ms=2.8, mec="black", mew=0.3, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(SETTING_LABEL[st], fontsize=7.5, color=SETTING_COLOR[st])
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_off()
axs[0].set_ylabel("mean measured amount per bin")
fig.text(0.5, 0.075, "mean predicted amount per bin (3 km blocks)", ha="center", fontsize=8)
h = seed_legend_handles(runs, [Line2D([], [], color=GREY, ls="--", lw=0.7, label="identity"), Line2D([], [], color=GREY, ls=":", lw=0.8, label="even spread")])
fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.5, columnspacing=1.0, handletextpad=0.3)
fig.text(0.99, 0.0, pending_note(runs), ha="right", va="bottom", fontsize=6.3, style="italic", color="#333333")
save(fig, "fig_calib")
