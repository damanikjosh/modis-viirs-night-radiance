"""Fig. lines: heatmap of the five pre-stated lines L1..L5 per run (lines_of() of the score files); rows of candidate-setting
seeds that are not scored yet are drawn hatched 'pending'.  Input: results/raw/select/*/score_select.json."""
from common import *
from matplotlib.patches import Rectangle

runs, base = load_select()
grp = by_group(runs)
miss = missing_seeds(runs)
LINE_TXT = {"L1": "gain over\nbest simple", "L2": "pixel energy\n0.8-1.25", "L3": "flags 0.5-2,\nhit >= 0.5", "L4": "top-1 share\n>= 0.33", "L5": "3 km amount\ndelivers"}
rows = []  # (group, seed, run or None)
for k in GROUP_ORDER:
    have = {r["seed"]: r for r in grp[k]}
    seeds = sorted(set(have) | set(miss.get(k, [])))
    for s in seeds:
        rows.append((k, s, have.get(s)))
n = len(rows)
H = 0.155 * n + 0.95   # inches: rows + header (0.62) + footnote (0.33)
fig = plt.figure(figsize=(WIDTH, H))
ax = fig.add_axes([0.30, 0.30 / H, 0.52, 0.155 * n / H])
ax.set_xlim(0, 5); ax.set_ylim(n, 0)
for i, (k, s, r) in enumerate(rows):
    if r is None:
        for c in range(5):
            ax.add_patch(Rectangle((c, i), 1, 1, fc="white", ec=LGREY, hatch="////", lw=0.4))
        ax.text(5.1, i + 0.55, "pending", va="center", fontsize=6, style="italic", color=GREY)
        continue
    L = lines_of(r["j"])
    for c, key in enumerate(("L1", "L2", "L3", "L4", "L5")):
        met = L[key]
        ax.add_patch(Rectangle((c, i), 1, 1, fc=color_of(r) if met else "#F2F2F2", ec="white", lw=0.8))
        if met: ax.text(c + 0.5, i + 0.58, "met", ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
    ax.text(5.1, i + 0.55, f"met {sum(L.values())}/5", va="center", fontsize=6.5)
# row labels and group separators
for i, (k, s, r) in enumerate(rows):
    ax.text(-0.08, i + 0.55, f"seed {s}", ha="right", va="center", fontsize=6.3)
start = 0
for k in GROUP_ORDER:
    cnt = sum(1 for x in rows if x[0] == k)
    if not cnt: continue
    lab = {"Old": "Group-share", "Gam": "Gamma", "H": "+BW100", "T": "+BW1000", "HF": "+BW100, final", "TF": "+BW1000, final",
           "HD": "+BW100-B0.5", "TD": "+BW1000-B0.5", "HDF": "+BW100-B0.5,\nfinal", "TDF": "+BW1000-B0.5,\nfinal"}[k]
    ax.text(-0.75, start + cnt / 2, lab, ha="right", va="center", fontsize=6.5, fontweight="bold", color=SETTING_COLOR[GROUP_SETTING[k]], transform=ax.transData)
    ax.axhline(start + cnt, color=GREY, lw=0.5, xmin=-0.0, xmax=1)
    start += cnt
ax.set_xticks([c + 0.5 for c in range(5)])
ax.set_xticklabels([f"{k}\n" + LINE_TXT[k] for k in ("L1", "L2", "L3", "L4", "L5")], fontsize=6.3)
ax.xaxis.tick_top()
ax.tick_params(length=0, labelleft=False)
ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
fig.text(0.985, 0.005, pending_note(runs) + "  Filled = line met; hatched = not scored yet.", ha="right", va="bottom", fontsize=6.3, style="italic", color="#333333")
save(fig, "fig_lines")
