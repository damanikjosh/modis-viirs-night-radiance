"""Fig. lossfamily: sweep 1 (small data set, old 9-pair validation), results/raw/sweep1/rows.jsonl.
Left: gain over even spread (vs_uniform.diff, range lo-hi) per run, grouped by loss, U-Net and 3-conv net separately; lines at 0 and
at the required +0.05.  Right: energy ratio at MODIS-bright blocks (energy_msel) with the 0.8-1.25 band.
Hollow marker = repeat run: a run name that appears more than once in the file (second and later occurrence), or a name containing 'repeat'."""
from common import *

rows = [json.loads(l) for l in open(os.path.join(RAW, "sweep1", "rows.jsonl")) if l.strip()]
seen, parsed = {}, []
for r in rows:
    m = re.fullmatch(r"(unet|conv3)__(.+)__s(\d+)", r["run"])
    arch, loss, seed = m[1], m[2], int(m[3])
    seen[r["run"]] = seen.get(r["run"], 0) + 1
    rep = seen[r["run"]] > 1 or "repeat" in r["run"]
    parsed.append(dict(r, arch=arch, loss=loss, seed=seed, rep=rep))
LOSS_ORDER = ["default-default", "fam-gamma", "fam-poisson", "fam-nmse"]
LOSS_LAB = {"default-default": "Group-share", "fam-gamma": "Gamma", "fam-poisson": "Poisson", "fam-nmse": "NMSE"}
LOSS_COL = {"default-default": SETTING_COLOR["old"], "fam-gamma": SETTING_COLOR["gamma"], "fam-poisson": OI["orange"], "fam-nmse": OI["sky"]}
ARCH_LAB = {"unet": "U-Net", "conv3": "3-conv"}
losses = [l for l in LOSS_ORDER if any(p["loss"] == l for p in parsed)]
slots = [(a, l) for a in ("unet", "conv3") for l in losses]
xpos = {s: i + (0.6 if s[0] == "conv3" else 0) for i, s in enumerate(slots)}
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.27, wspace=0.2)
cnt = {}
for p in parsed:
    key = (p["arch"], p["loss"]); k = cnt.get(key, 0); cnt[key] = k + 1
    n_in = sum(1 for q in parsed if (q["arch"], q["loss"]) == key)
    x = xpos[key] + (p["seed"] - 1) * 0.16 + (0.07 if p["rep"] else 0)
    col = LOSS_COL[p["loss"]]
    mk = seed_marker(p["seed"])
    v = p["vs_uniform"]
    a.plot([x, x], [v["lo"], v["hi"]], color=col, lw=0.7, zorder=2)
    a.plot([x], [v["diff"]], marker=mk, color=col, mfc="white" if p["rep"] else col, mec=col if p["rep"] else "black", mew=0.9 if p["rep"] else 0.4, ms=MS, ls="", zorder=3)
    b.plot([x], [p["energy_msel"]], marker=mk, color=col, mfc="white" if p["rep"] else col, mec=col if p["rep"] else "black", mew=0.9 if p["rep"] else 0.4, ms=MS, ls="", zorder=3)
a.axhline(0, color=GREY, lw=0.7, ls="--"); a.axhline(L1_MARGIN, color="black", lw=0.7, ls=":")
b.axhspan(E_LO, E_HI, color=OI["sky"], alpha=0.25, lw=0, zorder=0)
b.axhline(1.0, color=GREY, lw=0.5)
for ax in (a, b):
    ax.set_xticks([xpos[s] for s in slots])
    ax.set_xticklabels([LOSS_LAB[l] for _, l in slots], fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, max(xpos.values()) + 0.6)
    tr = ax.get_xaxis_transform()
    for arch in ("unet", "conv3"):
        xs_ = [xpos[s] for s in slots if s[0] == arch]
        ax.plot([min(xs_) - 0.4, max(xs_) + 0.4], [-0.3, -0.3], transform=tr, color=GREY, lw=0.6, clip_on=False)
        ax.text(sum([min(xs_), max(xs_)]) / 2, -0.33, ARCH_LAB[arch], transform=tr, ha="center", va="top", fontsize=6.5)
a.set_ylabel("AUC gain over even spread")
b.set_ylabel("energy ratio, MODIS-bright blocks")
b.set_yscale("log")
ev = [p["energy_msel"] for p in parsed] + [E_LO, E_HI]
b.set_ylim(min(ev) / 1.1, max(ev) * 1.1)
b.set_yticks([t for t in (0.8, 1, 1.25, 2, 2.5) if b.get_ylim()[0] <= t <= b.get_ylim()[1]])
b.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); b.minorticks_off()
a.set_title("(a) gain with 95 % range", fontsize=7, loc="left"); b.set_title("(b) energy ratio, band 0.8-1.25", fontsize=7, loc="left")
seeds = sorted({p["seed"] for p in parsed})
h = [Line2D([], [], marker=seed_marker(s), ls="", color=GREY, ms=MS, label=f"seed {s}") for s in seeds]
h += [Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, mew=0.9, ms=MS, label="hollow: repeat run"),
      Line2D([], [], ls="--", color=GREY, lw=0.7, label="0"), Line2D([], [], ls=":", color="black", lw=0.7, label="+%.2f required" % L1_MARGIN)]
fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.5, columnspacing=0.9, handletextpad=0.3)
fig.text(0.99, 0.0, "sweep 1: small data set, old 9-pair validation", ha="right", va="bottom", fontsize=6.3, style="italic", color="#333333")
save(fig, "fig_lossfamily")
