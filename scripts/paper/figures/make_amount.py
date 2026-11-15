"""Fig. amount: (a) pixel energy ratio per run with the 0.8-1.25 pass band; (b) 3 km mean absolute log error per run with
even-spread / excess-only lines; filled = amount.block3.decision.delivers.  Input: results/raw/select/*/score_select.json."""
from common import *

runs, base = load_select()
xs, rx = x_positions(runs)
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.30, wspace=0.22)

# (a) pixel-level energy ratio (ratio of sums), log y
a.axhspan(E_LO, E_HI, color=OI["sky"], alpha=0.25, lw=0, zorder=0)
a.axhline(1.0, color=GREY, lw=0.5, zorder=1)
for r in runs.values():
    a.plot([rx[r["run"]]], [r["m"]["energy_px"]], marker=seed_marker(r["seed"]), color=color_of(r), mec="black", mew=0.4,
           ms=MS, ls="", zorder=3)
group_axis(a, xs, runs, narrow=True)
a.set_yscale("log")
a.set_ylabel("pixel energy ratio (model / measured)")
v = [r["m"]["energy_px"] for r in runs.values()] + [E_LO, E_HI]
a.set_ylim(min(v) / 1.15, max(v) * 1.15)
a.set_yticks([t for t in (0.5, 0.8, 1.0, 1.25, 2, 3, 5) if a.get_ylim()[0] <= t <= a.get_ylim()[1]])
a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.minorticks_off()
a.set_title("(a) pixel level; shaded: pass band 0.8-1.25", fontsize=7, loc="left")

# (b) 3 km mean absolute log error per pair
uni = g(base, "amount", "block3", "methods", "uniform", "abs_log_err_pair_mean")
exc = g(base, "amount", "block3", "methods", "excess_only", "abs_log_err_pair_mean")
b.axhline(uni, color=GREY, lw=0.8, ls="--", zorder=1)
b.axhline(exc, color="black", lw=0.8, ls=":", zorder=1)
for r in runs.values():
    mm = g(r["j"], "amount", "block3", "methods", "model") or {}
    x, y = rx[r["run"]], mm.get("abs_log_err_pair_mean")
    lo, hi = mm.get("abs_log_err_lo"), mm.get("abs_log_err_hi")
    if lo is not None and hi is not None:
        b.plot([x, x], [lo, hi], color=color_of(r), lw=0.7, zorder=2)
    dl = bool(g(r["j"], "amount", "block3", "decision", "delivers"))
    b.plot([x], [y], marker=seed_marker(r["seed"]), color=color_of(r), mfc=color_of(r) if dl else "white", mec=color_of(r) if not dl else "black",
           mew=0.8 if not dl else 0.4, ms=MS, ls="", zorder=3)
group_axis(b, xs, runs, narrow=True)
b.set_ylabel("3 km mean abs. log error (pair mean)")
allv = [uni, exc] + [g(r["j"], "amount", "block3", "methods", "model", k) for r in runs.values() for k in ("abs_log_err_lo", "abs_log_err_hi", "abs_log_err_pair_mean")]
allv = [x for x in allv if x is not None]
p = 0.04 * (max(allv) - min(allv))
b.set_ylim(min(allv) - p, max(allv) + p)
b.set_title("(b) 3 km blocks; whiskers: lo-hi stored in score file", fontsize=7, loc="left")
h = seed_legend_handles(runs, [
    Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, mew=0.8, ms=MS, label="hollow: does not deliver (b)"),
    Line2D([], [], ls="--", color=GREY, lw=0.8, label="even spread"),
    Line2D([], [], ls=":", color="black", lw=0.8, label="oracle-background spread")])
fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.3, fontsize=6.5)
fig.text(0.99, 0.875, pending_note(runs), ha="right", va="bottom", fontsize=6.3, style="italic", color="#333333")
save(fig, "fig_amount")
