"""Fig. auc: block-ranking AUC on the 2,645-tile SELECT set per run (grouped by setting), with even-spread and
excess-only reference lines and each run's fire-off AUC as a grey cross.  Input: results/raw/select/*/score_select.json."""
from common import *

runs, base = load_select()
xs, rx = x_positions(runs)
n_tiles = {r["j"].get("n_tiles") for r in runs.values()}
fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.27)
uni = g(base, "summary", "auc", "uniform", "mean")
exc = g(base, "summary", "auc", "excess_only", "mean")
ax.axhline(uni, color=GREY, lw=0.8, ls="--", zorder=1)
ax.axhline(exc, color="black", lw=0.8, ls=":", zorder=1)
for r in runs.values():
    x = rx[r["run"]]
    off = r["m"]["auc_off"]
    ax.plot([x], [off], marker="x", color=GREY, ms=3.3, mew=0.7, ls="", zorder=2)
    ax.plot([x], [r["m"]["auc"]], marker=seed_marker(r["seed"]), color=color_of(r), mfc=color_of(r), mec="black", mew=0.4,
            ms=MS, ls="", zorder=3)
group_axis(ax, xs, runs)
ax.set_ylabel("block-ranking AUC (3 km blocks)")
vals = [r["m"]["auc"] for r in runs.values()] + [r["m"]["auc_off"] for r in runs.values()] + [uni, exc]
pad = 0.02 * (max(vals) - min(vals))
ax.set_ylim(min(vals) - 3 * pad, max(vals) + 3 * pad)
h = seed_legend_handles(runs, [
    Line2D([], [], marker="x", ls="", color=GREY, ms=3.3, mew=0.7, label="same run, fire output off"),
    Line2D([], [], ls="--", color=GREY, lw=0.8, label="even spread"),
    Line2D([], [], ls=":", color="black", lw=0.8, label="oracle-background spread")])
fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handletextpad=0.3)
note = pending_note(runs)
fig.text(0.985, 0.835, f"SELECT set, {', '.join(f'{n:,}' for n in sorted(x for x in n_tiles if x))} tiles.  {note}", ha="right", va="bottom",
         fontsize=6.3, style="italic", color="#333333")
save(fig, "fig_auc")
