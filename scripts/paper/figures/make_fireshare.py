"""Fig. fireshare (post hoc): share of the model's skill lost when the fire output is switched off, per run, for the bounded
measures fire_share_auc_block, fire_share_csi (3 km blocks) and fire_share_hottest_cell_share (MODIS pixel).
fire_share_amount_skill_vs_even is left out: it is unbounded (>1 = fire-off model worse than the even spread).
Input: results/raw/_descriptive/descriptive_measures.csv, SELECT set, subject model, all scored runs (best-val. and final)."""
from common import *

MEAS = [("fire_share_auc_block", "3km_block", "block-ranking AUC (3 km)"), ("fire_share_csi", "3km_block", "CSI (3 km)"),
        ("fire_share_hottest_cell_share", "modis_pixel", "hottest-cell share (pixel)")]
runs, _ = load_select()
data = [descriptive(m, lv) for m, lv, _ in MEAS]
fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 3.3), sharey=True)
fig.subplots_adjust(left=0.185, right=0.985, top=0.86, bottom=0.2, wspace=0.08)
ordr = [k for k in GROUP_ORDER if by_group(runs)[k]]
lab = {"Old": "Group-share", "Gam": "Gamma", "H": "+BW100", "T": "+BW1000", "HF": "+BW100, final", "TF": "+BW1000, final",
       "HD": "+BW100-B0.5", "TD": "+BW1000-B0.5", "HDF": "+BW100, b. 0.5, final", "TDF": "+BW1000, b. 0.5, final"}
grp = by_group(runs)
ypos = {k: len(ordr) - 1 - i for i, k in enumerate(ordr)}
allx = [d[r["run"]] for d in data for k in ordr for r in grp[k] if d.get(r["run"]) is not None]  # drawn runs only (ref-* rows excluded)
for ax, (m, lv, ttl), d in zip(axs, MEAS, data):
    ax.axvline(0, color=GREY, lw=0.7, ls="--", zorder=1)
    ax.axvline(1, color="black", lw=0.7, ls=":", zorder=1)
    for k in ordr:
        for r in grp[k]:
            v = d.get(r["run"])
            if v is None: continue
            hollow = r["bound"] == 0.5
            c = color_of(r)
            ax.plot([v], [ypos[k] + (r["seed"] - 0.5) * 0.12], marker=seed_marker(r["seed"]), color=c, mfc="white" if hollow else c,
                    mec=c if hollow else "black", mew=0.9 if hollow else 0.4, ms=MS, ls="", zorder=3)
    ax.set_title(ttl, fontsize=7, loc="left")
    ax.set_xlim(min(0, min(allx)) - 0.06, max(1, max(allx)) + 0.06)
    ax.set_yticks([ypos[k] for k in ordr])
    ax.set_xlabel("share of skill lost when fire output is off", fontsize=7)
    ax.set_ylim(-0.6, len(ordr) - 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=6.3)
axs[0].set_yticklabels([lab[k] for k in ordr], fontsize=6.5)
h = seed_legend_handles(runs, [Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, mew=0.9, ms=MS, label="hollow: bound 0.5"),
                               Line2D([], [], ls="--", color=GREY, lw=0.7, label="0 %: fire output adds nothing"),
                               Line2D([], [], ls=":", color="black", lw=0.7, label="100 %: all skill from fire output")])
fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=6.3, columnspacing=0.9, handletextpad=0.3)
fig.text(0.985, 0.0, "post hoc.  " + pending_note(runs), ha="right", va="bottom", fontsize=6.3, style="italic", color="#333333")
save(fig, "fig_fireshare")
