"""Fig. curves: training dynamics of the big runs (best-validation setting family, bound 0.25) from results/raw/<run>/log.jsonl.
(a) validation (solid) and training (dashed) loss, each normalised to its own epoch-0 value (losses of the old and Gamma families are
not comparable in absolute terms); (b) in-training block AUC (diag.proto.auc_model) with the even-spread value (auc_uniform);
(c) energy_msel with the 0.8-1.25 band; (d) top-1 share (top1_model) with the 0.33 line.  Marker = the run's best-validation epoch
(last epoch with is_best_val true).  All proto readings are half-sample in-training readings of SELECT.  Runs shown are those that
have a log file; logs of runs still training end at the last epoch written."""
import re
from common import *

logs = {}
for f in sorted(glob.glob(os.path.join(RAW, "*", "log.jsonl"))):
    name = os.path.basename(os.path.dirname(f))
    c = classify(name)
    if not c or c["fam"].startswith("ref") or c["bound"] != 0.25 or c["ckpt"] != "best": continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    logs[name] = (c, rows)
runs = runs_like(logs)

def series(rows, getter):
    return [(e["epoch"] + 1, getter(e)) for e in rows if getter(e) is not None]

fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.9), sharex=True)
fig.subplots_adjust(left=0.09, right=0.985, top=0.87, bottom=0.095, wspace=0.27, hspace=0.18)
(a, b), (c_, d) = axs
a.set_yscale("log")
for name, (c, rows) in logs.items():
    col, mk = color_of(c), seed_marker(c["seed"])
    bv = [e["epoch"] + 1 for e in rows if e.get("is_best_val")]
    be = bv[-1] if bv else None
    v = series(rows, lambda e: g(e, "diag", "val_total")); t = series(rows, lambda e: e.get("total"))
    v0, t0 = v[0][1], t[0][1]
    a.plot(*zip(*[(x, y / v0) for x, y in v]), color=col, lw=0.8)
    a.plot(*zip(*[(x, y / t0) for x, y in t]), color=col, lw=0.8, ls="--", alpha=0.8)
    for ax, key in ((a, None), (b, "auc_model"), (c_, "energy_msel"), (d, "top1_model")):
        if key:
            s = series(rows, lambda e: g(e, "diag", "proto", key))
            ax.plot(*zip(*s), color=col, lw=0.8)
        if be is not None:
            if key:
                yv = dict(s)[be]
            else:
                yv = dict(v)[be] / v0
            ax.plot([be], [yv], marker=mk, color=col, mec="black", mew=0.5, ms=MS, ls="", zorder=5)
# reference lines read from the logs (auc_uniform: same value in every run of the same half-sample)
au = sorted({round(g(e, "diag", "proto", "auc_uniform"), 6) for _, rows in logs.values() for e in rows if g(e, "diag", "proto", "auc_uniform") is not None})
if au:
    b.axhline(sum(au) / len(au), color=GREY, lw=0.8, ls="--", zorder=1)
b.text(0.995, 0.02, "dashed: even spread (auc_uniform)", transform=b.transAxes, ha="right", va="bottom", fontsize=6, color="#444444")
d.text(0.995, 0.02, "dashed: pass line %.2f" % TOP1, transform=d.transAxes, ha="right", va="bottom", fontsize=6, color="#444444")
c_.text(0.995, 0.02, "shaded: pass band %g-%g" % (E_LO, E_HI), transform=c_.transAxes, ha="right", va="bottom", fontsize=6, color="#444444")
c_.axhspan(E_LO, E_HI, color=OI["sky"], alpha=0.25, lw=0, zorder=0)
d.axhline(TOP1, color=GREY, lw=0.8, ls="--", zorder=1)
a.set_ylabel("loss / loss at epoch 1  (log scale)")
b.set_ylabel("block AUC (in-training)")
c_.set_ylabel("energy ratio, MODIS-selected (energy_msel)")
d.set_ylabel("top-1 share (top1_model)")
for ax in (c_, d): ax.set_xlabel("epoch")
for ax, t in ((a, "(a) loss, normalised per run"), (b, "(b) block AUC"), (c_, "(c) energy ratio"), (d, "(d) top-1 share")):
    ax.set_title(t, fontsize=7, loc="left")
# raw-loss scales differ between loss families: print the epoch-0 value per family in the note
h = [Line2D([], [], color=SETTING_COLOR[k], lw=1.2, label=SETTING_LABEL[k]) for k in order] if (order := [k for k in SETTING_COLOR if any(setting_of(c) == k for c, _ in logs.values())]) else []
h += seed_legend_handles(runs, [Line2D([], [], color=GREY, lw=0.8, ls="-", label="solid: validation (a), reading (b-d)"), Line2D([], [], color=GREY, lw=0.8, ls="--", label="dashed: training loss (a)")])
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.5, columnspacing=0.9, handletextpad=0.3)
sel_runs, _ = load_select()
pend = pending_note(sel_runs).rstrip(".")
if any(n for n in logs if classify(n)["grp"] in CANDIDATE_GROUPS and classify(n)["seed"] in sum(missing_seeds(sel_runs).values(), [])):
    pend += " (scores; their training logs are drawn)"
pend += "."
fig.text(0.985, 0.0, "in-training half-sample of SELECT; markers: best-validation epoch.  " + pend, ha="right", va="bottom", fontsize=6, style="italic", color="#333333")
save(fig, "fig_curves")
