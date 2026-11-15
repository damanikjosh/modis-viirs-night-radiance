"""Fig. width: overfit-gate training loss vs optimiser step for networks of different width/latent size, log y, coloured by parameter
count; left Gamma loss, right old ('legacy') loss.  Input: results/raw/wide/w<width>l<latent>_<gamma|legacy>_s<seed>_3k.json
(keys 'curve' = list of dicts with step, loss; 'n_params')."""
from common import *
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

files = []
for f in sorted(glob.glob(os.path.join(RAW, "wide", "*_3k.json"))):
    m = re.fullmatch(r"w(\d+)l(\d+)_(gamma|legacy)_s(\d+)_3k\.json", os.path.basename(f))
    if m:
        files.append((int(m[1]), int(m[2]), m[3], int(m[4]), json.load(open(f))))
npar = [j["n_params"] for *_, j in files]
norm = LogNorm(min(npar), max(npar))
cmap = plt.get_cmap("viridis")  # colour = log parameter count (value printed in the legend)
LAT_MK = dict(zip(sorted({x[1] for x in files}), "osD^v"))  # marker = latent size; dashed = seed 1
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
fig.subplots_adjust(left=0.095, right=0.99, top=0.93, bottom=0.36, wspace=0.28)
for ax, kind, ttl in zip(axs, ("gamma", "legacy"), ("(a) Gamma loss", "(b) Group-share")):
    sel = sorted([x for x in files if x[2] == kind], key=lambda x: (x[4]["n_params"], x[0], x[1], x[3]))
    handles = []
    for w, l, _, s, j in sel:
        cv = j["curve"]
        col = cmap(norm(j["n_params"]) * 0.9)
        ax.plot([p["step"] for p in cv], [p["loss"] for p in cv], color=col, lw=1.0, ls="-" if s == 0 else "--", marker=LAT_MK[l], ms=2.8, markevery=3, mec="black", mew=0.2)
        handles.append(Line2D([], [], color=col, lw=1.0, ls="-" if s == 0 else "--", marker=LAT_MK[l], ms=2.8, mec="black", mew=0.2,
                              label=f"width {w}, latent {l}, seed {s}: {j['n_params']/1e6:.2f} M params"))
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 5), numticks=20))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.3g")); ax.tick_params(axis="y", which="minor", labelsize=6, length=1.5)
    ax.set_title(ttl, fontsize=7, loc="left")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("training loss (overfit gate)")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=6, handlelength=2.6, labelspacing=0.25)
save(fig, "fig_width")
