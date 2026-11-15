"""Shared loaders, style and constants for the paper figures (matplotlib + numpy + stdlib only).

Every figure script reads ONLY the stored files in results/raw/ (provenance and sha256 in results/MANIFEST.json).
Options of every make_*.py:  --results DIR (default results)   --out DIR (default out/paper/figures).
The loaders below were copied from scripts/paper/gen_numbers.py (classify, measures, csi, lines_of) because that file runs its whole
pipeline on import and so cannot be imported; tests/paper/test_paper_numbers.py checks that both give the same line counts.
"""
import csv, glob, json, math, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

import argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--results", default=os.path.join(ROOT, "results"))
_ap.add_argument("--out", default=os.path.join(ROOT, "out", "paper", "figures"))
_a, _ = _ap.parse_known_args()
RES = os.path.abspath(_a.results)
RAW = os.path.join(RES, "raw")
OUT = os.path.abspath(_a.out)

# ---- pre-stated pass-line constants: configs/model_pick_rule.json (lines 2-5) and protocol_eval_v2_amendment4.json (line 1)
E_LO, E_HI = 0.8, 1.25      # line 2: pixel energy ratio band
F_LO, F_HI = 0.5, 2.0       # line 3: flag-count ratio band
HIT = 0.5                   # line 3: block hit rate at least
TOP1 = 0.33                 # line 4: top-1 share at least
L1_MARGIN, L1_SHARE = 0.05, 0.75   # line 1 (amendment 4): required gain over the better simple method, share of episodes
# candidate settings are scored with this many seeds in the protocol (seeds 0..3); missing ones are reported from the data
CANDIDATE_SEEDS = (0, 1, 2, 3)
CANDIDATE_GROUPS = ("H", "T")  # Gamma-BW100 / Gamma-BW1000, best-validation, bound 0.25

# ---- style
WIDTH = 6.5
def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Nimbus Roman", "Times New Roman", "DejaVu Serif"):
        if fam in names:
            break
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [fam, "DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "figure.dpi": 150,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    })
setup()

# Okabe-Ito, one colour per setting, used in every figure
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          verm="#D55E00", purple="#CC79A7", black="#000000")
GREY, LGREY = "#777777", "#BBBBBB"
SETTING_COLOR = {"old": OI["verm"], "gamma": OI["blue"], "gammabw100": OI["green"], "gammabw1000": OI["purple"]}
SETTING_LABEL = {"old": "Group-share", "gamma": "Gamma", "gammabw100": "Gamma-BW100", "gammabw1000": "Gamma-BW1000"}
SEED_MARKER = {0: "o", 1: "s", 2: "^", 3: "D"}
MS = 4.2

def setting_of(c):
    if c["fam"] == "old": return "old"
    if c["fam"] == "gamma": return "gamma"
    if c["fam"] == "gammabw": return "gammabw%d" % c["bw"]
    return None

def color_of(c):
    return SETTING_COLOR[setting_of(c)]

def seed_marker(seed):
    return SEED_MARKER.get(seed, "v")

# ---------------------------------------------------------------- loaders (copied from scripts/paper/gen_numbers.py)
def g(d, *path, default=None):
    for p in path:
        if d is None: return default
        try:
            d = d[p]
        except (KeyError, TypeError, IndexError):
            return default
    return d

def classify(run):
    m = re.fullmatch(r"oldloss-nf2-s(\d)", run)
    if m: return dict(fam="old", bw=0, bound=0.25, ckpt="best", seed=int(m[1]), grp="Old", label="Group-share")
    m = re.fullmatch(r"gamma-nf2-s(\d)", run)
    if m: return dict(fam="gamma", bw=0, bound=0.25, ckpt="best", seed=int(m[1]), grp="Gam", label="Gamma")
    m = re.fullmatch(r"gammabw(100|1000)(-d05)?-nf2-s(\d)(-final)?", run)
    if m:
        bw, d5, sd, fin = m[1], bool(m[2]), int(m[3]), bool(m[4])
        gg = ("H" if bw == "100" else "T") + ("D" if d5 else "") + ("F" if fin else "")
        lab = f"Gamma + w{bw}" + ("-B0.5" if d5 else "")
        return dict(fam="gammabw", bw=int(bw), bound=0.5 if d5 else 0.25, ckpt="final" if fin else "best", seed=sd, grp=gg, label=lab)
    m = re.fullmatch(r"ref-(gamma|legacy)-s(\d)", run)
    if m: return dict(fam="ref" + m[1], bw=0, bound=0.25, ckpt="best", seed=int(m[2]), grp="RefGam" if m[1] == "gamma" else "RefLeg",
                      label="small-list ref. " + ("Gamma" if m[1] == "gamma" else "Group-share"))
    return None

def csi(hit, fa):
    if hit is None or fa is None: return None
    found, missed, invented = hit, 1 - hit, fa * hit / (1 - fa) if fa < 1 else float("inf")
    den = found + missed + invented
    return found / den if den else None

def measures(j, m="model"):
    s, am = j["summary"], j["amount"]
    rule = g(s, "rule", m) or {}
    top = g(s, "conc", m) or {}
    b3 = g(am, "block3", "methods", m) or {}
    px = g(am, "pixel", "methods", m) or {}
    d = {
        "auc": g(s, "auc", m, "mean"), "auc_off": g(s, "auc", "model_off", "mean"),
        "energy_px": px.get("ratio_of_sums"), "energy_blk": b3.get("ratio_of_sums"),
        "flags": rule.get("flag_count_ratio"), "hit": rule.get("block_hit_rate"), "fa": rule.get("block_false_alarm_ratio"),
        "top1": top.get("top1_share"), "err3": b3.get("abs_log_err_pair_mean"), "err_px": px.get("abs_log_err_pair_mean"),
        "n_pairs3": g(am, "block3", "decision", "n_pairs"),
    }
    d["csi"] = csi(d["hit"], d["fa"])
    return d

def lines_of(j):
    """five pre-stated lines for a score JSON; line 1 per amendment 4, 2-5 per model_pick_rule.json"""
    s, am = j["summary"], j["amount"]
    bv = s.get("balanced_vs") or {}
    cands = {k: bv[k] for k in ("uniform", "excess_only") if bv.get(k) and bv[k].get("diff") is not None}
    l1 = False
    if cands:
        v = cands[min(cands, key=lambda k: cands[k]["diff"])]
        fire_adds = ((bv.get("model_off") or {}).get("diff") or 0) > 0
        l1 = v["diff"] >= L1_MARGIN and (v.get("lo") or -1) > 0 and (v.get("share_pos_episodes") or 0) >= L1_SHARE and fire_adds
    m = measures(j)
    deliv = g(am, "block3", "decision", "delivers")
    return {"L1": bool(l1),
            "L2": m["energy_px"] is not None and E_LO <= m["energy_px"] <= E_HI,
            "L3": m["flags"] is not None and m["hit"] is not None and F_LO <= m["flags"] <= F_HI and m["hit"] >= HIT,
            "L4": m["top1"] is not None and m["top1"] >= TOP1,
            "L5": bool(deliv)}

# group order for the grouped x axes: best-validation bound 0.25, then final epoch, then bound 0.5
GROUP_ORDER = ["Old", "Gam", "H", "T", "HF", "TF", "HD", "TD", "HDF", "TDF"]
GROUP_SETTING = {"Old": "old", "Gam": "gamma", "H": "gammabw100", "T": "gammabw1000", "HF": "gammabw100", "TF": "gammabw1000",
                 "HD": "gammabw100", "TD": "gammabw1000", "HDF": "gammabw100", "TDF": "gammabw1000"}
GROUP_TICK = {"Old": "Group-\nshare", "Gam": "Gamma", "H": "+BW100", "T": "+BW1000", "HF": "+BW100", "TF": "+BW1000",
              "HD": "+BW100", "TD": "+BW1000", "HDF": "+BW100", "TDF": "+BW1000"}
# header spans printed under the tick labels
GROUP_SPANS = [("best validation, bound 0.25", ["Old", "Gam", "H", "T"]), ("final epoch", ["HF", "TF"]),
               ("bound 0.5, best val.", ["HD", "TD"]), ("bound 0.5, final", ["HDF", "TDF"])]

GROUP_SPANS_NARROW = [("best val.,\nbound 0.25", ["Old", "Gam", "H", "T"]), ("final\nepoch", ["HF", "TF"]),
                      ("bound 0.5\nbest val.", ["HD", "TD"]), ("bound 0.5\nfinal", ["HDF", "TDF"])]

def load_select():
    """-> (runs dict run -> {c, j}, baselines json).  Only model runs that classify() knows (ref-* excluded here)."""
    runs, base = {}, None
    for f in sorted(glob.glob(os.path.join(RAW, "select", "*", "score_select.json"))):
        name = os.path.basename(os.path.dirname(f))
        j = json.load(open(f))
        if name == "baselines":
            base = j
            continue
        c = classify(name)
        if c and not c["fam"].startswith("ref"):
            runs[name] = dict(c, run=name, j=j, m=measures(j))
    return runs, base

def by_group(runs):
    out = {k: [] for k in GROUP_ORDER}
    for r in runs.values():
        if r["grp"] in out: out[r["grp"]].append(r)
    for k in out: out[k].sort(key=lambda r: r["seed"])
    return out

def missing_seeds(runs):
    """{group: [missing seeds]} for the two candidate settings (best-validation), computed from the files present"""
    grp = by_group(runs)
    return {k: [s for s in CANDIDATE_SEEDS if s not in {r["seed"] for r in grp[k]}] for k in CANDIDATE_GROUPS}

def pending_note(runs):
    """small italic note naming the missing seeds, built from the data; '' if nothing is missing"""
    miss = missing_seeds(runs)
    names = {"H": "BW100", "T": "BW1000"}
    bykey = {}
    for k, ms in miss.items():
        if ms: bykey.setdefault(tuple(ms), []).append(names[k])
    if not bykey: return "All candidate seeds (0-3) scored."
    parts = []
    for ms, ks in bykey.items():
        sd = f"{ms[0]}" if len(ms) == 1 else (f"{ms[0]}-{ms[-1]}" if list(ms) == list(range(ms[0], ms[-1] + 1)) else ",".join(map(str, ms)))
        parts.append(f"seed{'s' if len(ms) > 1 else ''} {sd} of {' / '.join(ks)}: pending")
    return "; ".join(parts) + "."

def x_positions(runs, width=0.62):
    """x slot per group; returns (xs {group: centre}, runs_x {run: x}) with seeds spread inside the slot"""
    grp = by_group(runs)
    xs, rx, x = {}, {}, 0.0
    gap_after = {"Old", "Gam", "T", "TF", "TD"}
    seeds_all = sorted({r["seed"] for r in runs.values()})
    n = max(len(seeds_all), 1)
    for k in GROUP_ORDER:
        xs[k] = x
        for r in grp[k]:
            i = seeds_all.index(r["seed"])
            rx[r["run"]] = x + (i - (n - 1) / 2) * width / max(n - 1, 1)
        x += 1.0 + (0.5 if k in {"T", "TF", "TD"} else 0.0)
    return xs, rx

def group_axis(ax, xs, runs, narrow=False):
    """tick labels per group and header spans below the axis (narrow: rotated ticks, short span labels for two-panel figures)"""
    ax.set_xticks([xs[k] for k in GROUP_ORDER])
    if narrow:
        ax.set_xticklabels([GROUP_TICK[k].replace("\n", " ") for k in GROUP_ORDER], fontsize=6.3, rotation=55, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([GROUP_TICK[k] for k in GROUP_ORDER], fontsize=6.5)
    ax.set_xlim(min(xs.values()) - 0.55, max(xs.values()) + 0.55)
    tr = ax.get_xaxis_transform()
    spans = GROUP_SPANS_NARROW if narrow else GROUP_SPANS
    y0 = -0.27 if narrow else -0.2
    for lab, ks in spans:
        x0, x1 = xs[ks[0]] - 0.4, xs[ks[-1]] + 0.4
        ax.plot([x0, x1], [y0, y0], transform=tr, color=GREY, lw=0.6, clip_on=False)
        ax.text((x0 + x1) / 2, y0 - 0.03, lab, transform=tr, ha="center", va="top", fontsize=6, color="#333333", linespacing=0.95)

def seed_legend_handles(runs, extra=()):
    seeds = sorted({r["seed"] for r in runs.values()})
    h = [Line2D([], [], marker=seed_marker(s), ls="", color=GREY, mfc=GREY, ms=MS, label=f"seed {s}") for s in seeds]
    return h + list(extra)

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p)
    print("wrote", p)
    return p

def read_csv(rel):
    with open(os.path.join(RAW, rel), newline="") as f:
        return list(csv.DictReader(f))

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def descriptive(measure, level, subject="model", set_="select"):
    """{run: value} from results/raw/_descriptive/descriptive_measures.csv (post hoc measures)"""
    out = {}
    for r in read_csv("_descriptive/descriptive_measures.csv"):
        if r["measure"] == measure and r["level"] == level and r["subject"] == subject and r["set"] == set_:
            out[r["run"]] = fnum(r["value"])
    return out

def runs_like(names):
    """pseudo run dict (for pending_note) from a collection of run names"""
    return {n: dict(classify(n), run=n) for n in names if classify(n) and not classify(n)["fam"].startswith("ref")}
