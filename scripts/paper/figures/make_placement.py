"""Fig. placement: right-hottest-cell rate (pooled hit rate of the predicted hottest cell) per training variant and seed, with range
(lo-hi), and reference lines for chance, centre-of-pixel, point-spread rules (psf, psf + neighbours).
Input: results/raw/placement/placement.json, runs placement-<variant>-s<seed>.
Block used: 'val2far' (33 scene pairs / 1,023 footprints; the 'val' block has only 6 pairs / 57 footprints).  n is read from the file."""
from common import *

d = json.load(open(os.path.join(RAW, "placement", "placement.json")))["runs"]
BLOCK = "val2far"
VARIANTS = [("old", "old"), ("oldnf", "oldnf"), ("oldnf-short", "oldnf-short"),
            ("oldnf-geometry_only", "control\ngeometry_only"), ("oldnf-radiance_only", "control\nradiance_only"), ("oldnf-modis_mask", "control\nmodis_mask")]  # variant names as in the file
VCOL = {"old": SETTING_COLOR["old"], "oldnf": OI["orange"], "oldnf-short": OI["yellow"], "oldnf-geometry_only": OI["sky"],
        "oldnf-radiance_only": OI["blue"], "oldnf-modis_mask": GREY}
runs = {}
for k, v in d.items():
    m = re.fullmatch(r"placement-(.+)-s(\d+)", k)
    if m and m[1] in dict(VARIANTS) and BLOCK in v:
        runs[k] = (m[1], int(m[2]), v[BLOCK])
nfp = {b["n_footprints"] for _, _, b in runs.values()}; npr = {b["n_pairs_all"] for _, _, b in runs.values()}
fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.17)
xs = {v: i for i, (v, _) in enumerate(VARIANTS)}
for k, (var, seed, b) in runs.items():
    h = b["pooled"]["hit"]["model"]
    x = xs[var] + (seed - 1) * 0.17
    ax.plot([x, x], [h["lo"], h["hi"]], color=VCOL[var], lw=0.8, zorder=2)
    ax.plot([x], [h["value"]], marker=seed_marker(seed), color=VCOL[var], mec="black", mew=0.4, ms=MS, ls="", zorder=3)
REF = [("chance", "chance (random cell)", ":", "black"), ("centre", "centre of pixel", "--", GREY), ("psf", "point-spread rule", "-.", "#444444"), ("psf_nbr", "point-spread + neighbours", (0, (5, 1, 1, 1, 1, 1)), "#999999")]
refs, refvals = [], []
for key, lab, ls, col in REF:
    vals = [b["pooled"]["hit"][key]["value"] for _, _, b in runs.values()]
    mu = sum(vals) / len(vals)
    ax.axhline(mu, color=col, ls=ls, lw=0.8, zorder=1)
    refs.append(Line2D([], [], ls=ls, color=col, lw=0.9, label=f"{lab} ({min(vals):.3f}-{max(vals):.3f} over runs)"))
    refvals += vals
ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels([l for _, l in VARIANTS], fontsize=6.3)
ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
ax.set_ylabel("hit rate of the predicted hottest cell")
ymax = max(b["pooled"]["hit"]["model"]["hi"] for _, _, b in runs.values()); ymin = min(b["pooled"]["hit"]["model"]["lo"] for _, _, b in runs.values())
ax.set_ylim(min(ymin, min(refvals)) - 0.02, ymax + 0.02)
seeds = sorted({s for _, s, _ in runs.values()})
hnd = [Line2D([], [], marker=seed_marker(s), ls="", color=GREY, ms=MS, label=f"seed {s}") for s in seeds]
fig.legend(handles=hnd + refs, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.3, handletextpad=0.3, columnspacing=1.0)
f = ", ".join(f"{n:,}" for n in sorted(nfp)); p = ", ".join(str(n) for n in sorted(npr))
fig.text(0.985, 0.80, f"{p} scene pairs, {f} footprints (block {BLOCK}). Lines: mean over runs of the rule's pooled hit rate; whiskers: lo-hi of the file.", ha="right", va="bottom", fontsize=6, style="italic", color="#333333")
save(fig, "fig_placement")
