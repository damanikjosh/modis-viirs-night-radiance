"""Fig. desc (post hoc): 3 km critical-success index against flag-count ratio per run (SELECT set), pre-stated 0.5-2 band shaded,
simple methods as grey labelled points.  Input: results/raw/_descriptive/descriptive_measures.csv (+ select score files for the
list of scored runs).  Best-validation checkpoints only (final-epoch rows are in the CSV but not drawn)."""
from common import *

csi_ = descriptive("csi", "3km_block")
flg = descriptive("flag_count_ratio", "3km_block")
runs, _ = load_select()
fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 3.0))
fig.subplots_adjust(left=0.13, right=0.98, top=0.80, bottom=0.14)
ax.axvspan(F_LO, F_HI, color=OI["sky"], alpha=0.25, lw=0, zorder=0)
for n in csi_:
    c = classify(n)
    if not c or c["fam"].startswith("ref") or c["ckpt"] != "best" or n not in flg or csi_[n] is None or flg[n] is None: continue
    col = color_of(c)
    hollow = c["bound"] == 0.5
    ax.plot([flg[n]], [csi_[n]], marker=seed_marker(c["seed"]), color=col, mfc="white" if hollow else col, mec=col if hollow else "black",
            mew=0.9 if hollow else 0.4, ms=MS + 0.5, ls="", zorder=3)
# simple methods
simple = {}
for r in read_csv("_descriptive/descriptive_measures.csv"):
    if r["run"] == "baselines" and r["set"] == "select" and r["level"] == "3km_block" and r["measure"] in ("csi", "flag_count_ratio"):
        simple.setdefault(r["subject"], {})[r["measure"]] = fnum(r["value"])
LAB = {"uniform": "even spread", "excess_only": "oracle-bg. spread", "copy_planck": "copy Planck", "copy_raw": "copy raw"}
OFF = {"uniform": (6, -1), "excess_only": (-6, 3), "copy_planck": (6, 1), "copy_raw": (6, 2)}
for k, v in simple.items():
    if "csi" in v and "flag_count_ratio" in v and v["csi"] is not None and v["flag_count_ratio"] is not None:
        ax.plot([v["flag_count_ratio"]], [v["csi"]], marker="P", color=GREY, ms=4.5, ls="", zorder=2)
        dx, dy = OFF.get(k, (4, 3))
        ax.annotate(LAB.get(k, k), (v["flag_count_ratio"], v["csi"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, color="#444444",
                    ha="left" if dx > 0 else "right", va="center")
ax.set_xscale("log")
xt = [t for t in (0.25, 0.5, 1, 2, 4, 8) if ax.get_xlim()[0] * 0.9 <= t <= ax.get_xlim()[1] * 1.1]
ax.set_xticks(xt); ax.set_xticklabels([f"{t:g}" for t in xt]); ax.minorticks_off()
ax.set_xlabel("flag-count ratio (model flags / measured flags), 3 km blocks")
ax.set_ylabel("critical success index (CSI), 3 km blocks")
h = [Line2D([], [], marker="o", ls="", color=SETTING_COLOR[k], mec="black", mew=0.4, ms=MS, label=SETTING_LABEL[k]) for k in SETTING_COLOR]
h += [Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, mew=0.9, ms=MS, label="hollow: bound 0.5")]
h += [Line2D([], [], marker="P", ls="", color=GREY, ms=4.5, label="simple method")]
h += seed_legend_handles(runs)
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.3, columnspacing=0.8, handletextpad=0.2)
fig.text(0.98, 0.81, "post hoc.  Shaded: pre-stated band 0.5-2.  " + pending_note(runs), ha="right", va="bottom", fontsize=6, style="italic", color="#333333")
save(fig, "fig_desc")
