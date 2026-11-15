#!/usr/bin/env python3
"""Regenerate every number macro and table of the paper from the stored result files in results/.

Usage:  python scripts/paper/gen_numbers.py [--results results] [--out out/paper] [--no-verify]

Reads only <results>/ (stored score files, run records, training logs, placement rows, descriptive measures, the compacted post-hoc
measure summary, the recorded static numbers and tables; integrity checked against <results>/MANIFEST.json first) and writes into <out>/:
  numbers.tex      one \\newcommand per number (letters only in names), computed from result files or read from results/static_numbers.tsv
  NUMBERS.md       the number map: macro, value, source file, key path / note
  tab_*.tex        booktabs tables (decision table, per-run values, pick rule, baselines, descriptive measures, loss family, hyper-parameters, run list, ...)
No number is typed by hand here except the pre-stated pass-line constants (LINES below, copied from configs/model_pick_rule.json and
configs/protocol_eval_v2_amendment4.json). Values that cannot be recomputed from stored files are in results/static_numbers.tsv and
results/static_tables/*.tsv (recorded values, each with its source note); docs/paper.md lists them.
"""
import argparse, csv, glob, json, math, os, re, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--results", default=os.path.join(ROOT, "results"))
_ap.add_argument("--out", default=os.path.join(ROOT, "out", "paper"))
_ap.add_argument("--no-verify", action="store_true", help="skip the sha256 check against MANIFEST.json")
_args = _ap.parse_args()
RES = os.path.abspath(_args.results)
RAW = os.path.join(RES, "raw")
GEN = os.path.abspath(_args.out)
os.makedirs(GEN, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_manifest
MAN = json.load(open(os.path.join(RES, "MANIFEST.json")))
if not _args.no_verify:
    _bad = verify_manifest.verify(RES)
    if _bad:
        sys.exit("MANIFEST check failed:\n  " + "\n  ".join(_bad))
STEPS_PER_EPOCH = 647  # 29,762 steps / 46 epochs (run_meta + results.json of the big runs)

# pre-stated lines (configs/model_pick_rule.json lines_2_to_5 and protocol_eval_v2_amendment4.json), copied constants
E_LO, E_HI, F_LO, F_HI, HIT, TOP1 = 0.8, 1.25, 0.5, 2.0, 0.5, 0.33
L1_MARGIN, L1_SHARE = 0.05, 0.75

NUMS = []      # (macro, value-string, source, key)
def rel(p):
    return "results/" + os.path.relpath(p, RES)

DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
def mn(*parts):
    s = "".join(str(p) for p in parts)
    s = "".join(DIG.get(c, c) for c in s if c.isalnum())
    return "N" + s

def num(name, value, fmt, source, key):
    """register macro \\N<name>; value is formatted with fmt (a format spec or callable)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        txt = r"\textbf{n/a}"
    else:
        txt = fmt(value) if callable(fmt) else format(value, fmt)
    NUMS.append((mn(name), txt, source, key))
    return txt

def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

# ---------------------------------------------------------------- load scores
def load_scores(subdir, fname):
    out = {}
    for f in sorted(glob.glob(os.path.join(RAW, subdir, "*", fname))):
        out[os.path.basename(os.path.dirname(f))] = (json.load(open(f)), rel(f))
    return out

SEL = load_scores("select", "score_select.json")
OLD = load_scores("old", "score_oldval.json")
BASE_J, BASE_SRC = SEL.pop("baselines")
SEL_RUNS = {k: v for k, v in SEL.items()}

def classify(run):
    m = re.fullmatch(r"oldloss-nf2-s(\d)", run)
    if m: return dict(fam="old", bw=0, bound=0.25, ckpt="best", seed=int(m[1]), grp="Old", label="Group-share")
    m = re.fullmatch(r"gamma-nf2-s(\d)", run)
    if m: return dict(fam="gamma", bw=0, bound=0.25, ckpt="best", seed=int(m[1]), grp="Gam", label="Gamma")
    m = re.fullmatch(r"gammabw(100|1000)(-d05)?-nf2-s(\d)(-final)?", run)
    if m:
        bw, d5, sd, fin = m[1], bool(m[2]), int(m[3]), bool(m[4])
        g = ("H" if bw == "100" else "T") + ("D" if d5 else "") + ("F" if fin else "")
        lab = f"Gamma-BW{bw}" + ("-B0.5" if d5 else "")
        return dict(fam="gammabw", bw=int(bw), bound=0.5 if d5 else 0.25, ckpt="final" if fin else "best", seed=sd, grp=g, label=lab)
    m = re.fullmatch(r"ref-(gamma|legacy)-s(\d)", run)
    if m: return dict(fam="ref" + m[1], bw=0, bound=0.25, ckpt="best", seed=int(m[2]), grp="RefGam" if m[1] == "gamma" else "RefLeg",
                      label=("Gamma" if m[1] == "gamma" else "Group-share") + " (small data)")
    return None

GROUP_ORDER = ["Old", "Gam", "H", "T", "HF", "TF", "HD", "TD", "HDF", "TDF", "RefGam", "RefLeg"]
GROUP_LABEL = {"Old": "Group-share (best-val)", "Gam": "Gamma (best-val)", "H": "Gamma-BW100 (best-val)", "T": "Gamma-BW1000 (best-val)",
               "HF": "Gamma-BW100 (final)", "TF": "Gamma-BW1000 (final)", "HD": "Gamma-BW100-B0.5 (best-val)", "TD": "Gamma-BW1000-B0.5 (best-val)",
               "HDF": "Gamma-BW100-B0.5 (final)", "TDF": "Gamma-BW1000-B0.5 (final)",
               "RefGam": "Gamma (small data)", "RefLeg": "Group-share (small data)"}

def g(d, *path, default=None):
    for p in path:
        if d is None: return default
        try:
            d = d[p]
        except (KeyError, TypeError, IndexError):
            return default
    return d

def measures(j, m="model"):
    """All per-run measures from one score JSON (m = 'model' or a baseline method key)."""
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

def csi(hit, fa):
    if hit is None or fa is None: return None
    found, missed, invented = hit, 1 - hit, fa * hit / (1 - fa) if fa < 1 else float("inf")
    den = found + missed + invented
    return found / den if den else None

def lines_of(j):
    """five pre-stated lines for a score JSON; line 1 per amendment 4 (region-balanced, episode-clustered), 2-5 per model_pick_rule.json"""
    s, am = j["summary"], j["amount"]
    bv = s.get("balanced_vs") or {}
    cands = {k: bv[k] for k in ("uniform", "excess_only") if bv.get(k) and bv[k].get("diff") is not None}
    l1 = False
    if cands:  # the BETTER simple method = the one the model beats by the smaller margin (as scripts/verdict_table.py of the repo)
        v = cands[min(cands, key=lambda k: cands[k]["diff"])]
        fire_adds = ((bv.get("model_off") or {}).get("diff") or 0) > 0
        l1 = v["diff"] >= L1_MARGIN and (v.get("lo") or -1) > 0 and (v.get("share_pos_episodes") or 0) >= L1_SHARE and fire_adds
    m = measures(j)
    deliv = g(am, "block3", "decision", "delivers")
    L = {"L1": bool(l1),
         "L2": m["energy_px"] is not None and E_LO <= m["energy_px"] <= E_HI,
         "L3": m["flags"] is not None and m["hit"] is not None and F_LO <= m["flags"] <= F_HI and m["hit"] >= HIT,
         "L4": m["top1"] is not None and m["top1"] >= TOP1,
         "L5": bool(deliv)}
    return L

def stamped(j):
    return bool(j.get("guard") and j.get("code_sha256"))

RUNS = {}
for run, (j, src) in SEL_RUNS.items():
    c = classify(run)
    if not c: continue
    m = measures(j)
    L = lines_of(j)
    RUNS[run] = dict(c, run=run, src=src, j=j, m=m, L=L, met=sum(L.values()), met25=sum(L[k] for k in ("L2", "L3", "L4", "L5")),
                     all25=all(L[k] for k in ("L2", "L3", "L4", "L5")), stamped=stamped(j), epoch=j.get("epoch"), ckpt16=j.get("ckpt"),
                     n_tiles=j.get("n_tiles"), git=j.get("git"))

def grp_runs(grp):
    return sorted([r for r in RUNS.values() if r["grp"] == grp], key=lambda r: r["seed"])

# ---------------------------------------------------------------- macros from the data
BM = {k: measures(BASE_J, k) for k in ("uniform", "excess_only", "copy_planck", "copy_raw", "const")}
BM["excess_only"]["hit"] = g(BASE_J, "summary", "rule", "oracle_uniform", "block_hit_rate") if BM["excess_only"]["hit"] is None else BM["excess_only"]["hit"]
for k in ("excess_only",):
    rr = g(BASE_J, "summary", "rule", "oracle_uniform") or {}
    for fld, kk in (("hit", "block_hit_rate"), ("fa", "block_false_alarm_ratio"), ("flags", "flag_count_ratio")):
        if BM[k][fld] is None: BM[k][fld] = rr.get(kk)
    BM[k]["csi"] = csi(BM[k]["hit"], BM[k]["fa"])
BM["uniform"]["top1"] = BM["uniform"]["top1"] if BM["uniform"]["top1"] is not None else g(BASE_J, "summary", "conc", "uniform", "top1_share")
MEAS_TOP = g(BASE_J, "summary", "conc", "measured", "top1_share")

SRC_B = BASE_SRC
num("TilesSelect", next(iter(RUNS.values()))["n_tiles"], ",d", SRC_B, "n_tiles")
for k, nm in (("uniform", "Even"), ("excess_only", "Excess"), ("copy_planck", "Planck"), ("copy_raw", "Raw")):
    for f, lab, fm in (("auc", "Auc", ".3f"), ("err3", "ErrBlk", ".2f"), ("err_px", "ErrPx", ".2f"), ("hit", "Hit", ".2f"), ("fa", "Fa", ".2f"), ("csi", "Csi", ".2f"), ("top1", "Top", ".2f"), ("energy_blk", "EnBlk", ".2f"), ("flags", "Flags", ".2f")):
        num(f"Base{nm}{lab}", BM[k][f], fm, SRC_B, f"summary/amount[{k}] {f}")
num("BaseMeasuredTop", MEAS_TOP, ".2f", SRC_B, "summary.conc.measured.top1_share")
num("LnHalfSel", (BM["uniform"]["top1"] + MEAS_TOP) / 2, ".2f", SRC_B, "(even spread top-1 + measured top-1)/2 on SELECT")
num("BaseConstAuc", BM["const"]["auc"], ".3f", SRC_B, "summary.auc.const.mean")

def rng(vals, fm):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return (min(vals), max(vals), st.mean(vals), len(vals))

GROUP_STATS = {}
for gk in GROUP_ORDER:
    rs = grp_runs(gk)
    GROUP_STATS[gk] = rs
    num(f"{gk}N", len(rs), "d", "results/raw/select/*", f"count of score files in group {gk}")
    for f, lab, fm in (("auc", "Auc", ".3f"), ("auc_off", "AucOff", ".3f"), ("energy_px", "EnPx", ".2f"), ("energy_blk", "EnBlk", ".2f"), ("flags", "Flags", ".2f"),
                       ("hit", "Hit", ".2f"), ("fa", "Fa", ".2f"), ("csi", "Csi", ".2f"), ("top1", "Top", ".2f"), ("err3", "ErrBlk", ".2f"), ("err_px", "ErrPx", ".2f")):
        v = [r["m"][f] for r in rs if r["m"][f] is not None]
        if v:
            num(f"{gk}{lab}Min", min(v), fm, "results/raw/select/*/score_select.json", f"min over seeds of group {gk}: {f}")
            num(f"{gk}{lab}Max", max(v), fm, "results/raw/select/*/score_select.json", f"max over seeds of group {gk}: {f}")
            num(f"{gk}{lab}Mean", st.mean(v), fm, "results/raw/select/*/score_select.json", f"mean over seeds of group {gk}: {f}")
    for k in ("L1", "L2", "L3", "L4", "L5"):
        num(f"{gk}{k}", sum(r["L"][k] for r in rs), "d", "results/raw/select/*/score_select.json", f"seeds of group {gk} meeting {k}")
    num(f"{gk}AllTwoFive", sum(r["all25"] for r in rs), "d", "results/raw/select/*/score_select.json", f"seeds of group {gk} meeting lines 2-5")
    num(f"{gk}Passes", sum(r["met25"] for r in rs), "d", "results/raw/select/*/score_select.json", f"line passes (2-5) summed over seeds of group {gk}")
    num(f"{gk}Stamped", sum(r["stamped"] for r in rs), "d", "results/raw/select/*/score_select.json", f"files of group {gk} with guard+code_sha256 stamp")

# all Gamma-family runs (plain Gamma and brightness-weighted, every bound / checkpoint kind)
GF = [r for r in RUNS.values() if r["fam"] in ("gamma", "gammabw")]
num("GFN", len(GF), "d", "results/raw/select/*/score_select.json", "Gamma-family score files")
num("GFAucMax", max(r["m"]["auc"] for r in GF), ".3f", "results/raw/select/*/score_select.json", "max block AUC over Gamma-family runs")
num("GFAucMin", min(r["m"]["auc"] for r in GF), ".3f", "results/raw/select/*/score_select.json", "min block AUC over Gamma-family runs")
num("GFLOne", sum(r["L"]["L1"] for r in GF), "d", "results/raw/select/*/score_select.json", "Gamma-family runs meeting line 1")
num("GFAboveExcess", sum(1 for r in GF if r["m"]["auc"] > BM["excess_only"]["auc"]), "d", "results/raw/select/*/score_select.json", "Gamma-family runs with mean AUC above the excess-only spread (no margin or range)")
num("GFFlagsMin", min(r["m"]["flags"] for r in GF), ".2f", "results/raw/select/*/score_select.json", "min flag ratio Gamma family"); num("GFFlagsMax", max(r["m"]["flags"] for r in GF), ".2f", "results/raw/select/*/score_select.json", "max flag ratio")
num("GFErrBelowExcess", sum(1 for r in GF if r["m"]["err3"] < BM["excess_only"]["err3"]), "d", "results/raw/select/*/score_select.json", "Gamma-family runs with 3 km error below excess-only spread")
num("GFDeliver", sum(r["L"]["L5"] for r in GF), "d", "results/raw/select/*/score_select.json", "Gamma-family runs meeting line 5")
num("GFAllTwoFive", sum(r["all25"] for r in GF), "d", "results/raw/select/*/score_select.json", "Gamma-family runs meeting lines 2-5")
num("GFAllFive", sum(1 for r in GF if r["met"] == 5), "d", "results/raw/select/*/score_select.json", "Gamma-family runs meeting all five lines")
num("GFTopMin", min(r["m"]["top1"] for r in GF), ".2f", "results/raw/select/*/score_select.json", "min top-1 share Gamma family"); num("GFTopMax", max(r["m"]["top1"] for r in GF), ".2f", "results/raw/select/*/score_select.json", "max top-1 share Gamma family")

BW = [r for r in RUNS.values() if r["fam"] == "gammabw"]
for f, lab, fm in (("top1", "Top", ".2f"), ("err3", "ErrBlk", ".2f"), ("energy_px", "EnPx", ".2f"), ("flags", "Flags", ".2f"), ("auc", "Auc", ".3f")):
    v = [r["m"][f] for r in BW]
    num(f"BwAll{lab}Min", min(v), fm, "results/raw/select/*/score_select.json", f"min over all brightness-weighted runs: {f}"); num(f"BwAll{lab}Max", max(v), fm, "results/raw/select/*/score_select.json", f"max over all brightness-weighted runs: {f}")
num("BwAllN", len(BW), "d", "results/raw/select/*/score_select.json", "brightness-weighted runs scored")

def _ra(r, refs):
    v = r["j"]["amount"]["block3"]["decision"]["vs"]
    return all((v.get(k) or {}).get("range_above_zero") for k in refs)
num("GFRangeExcess", sum(_ra(r, ("excess_only",)) for r in GF), "d", "results/raw/select/*/score_select.json", "amount.block3.decision.vs.excess_only.range_above_zero (3 km, pair-clustered 95 % range of paired error difference excludes 0)")
num("GFRangeAllSimple", sum(_ra(r, ("uniform", "copy_planck", "copy_raw", "excess_only", "const")) for r in GF), "d", "results/raw/select/*/score_select.json", "same, against all five simple methods")

# pairs behind the 3 km amount decision (same in every file?)
npairs = sorted({r["m"]["n_pairs3"] for r in RUNS.values() if r["m"]["n_pairs3"]})
num("PairsBlk", npairs[0] if npairs else None, "d", "results/raw/select/*/score_select.json", "amount.block3.decision.n_pairs")
num("FilesStamped", sum(r["stamped"] for r in RUNS.values()), "d", "results/raw/select/*/score_select.json", "files with guard and code_sha256")
num("FilesTotal", len(RUNS), "d", "results/raw/select/*/score_select.json", "scored run files (model runs, SELECT)")

# pre-stated pass-line constants (copied from configs/model_pick_rule.json lines_2_to_5 and protocol_eval_v2 amendment 1/4)
LC_SRC = "configs/model_pick_rule.json lines_2_to_5; protocol_eval_v2_amendment4.json"
for nm, v, fm in (("LnELo", E_LO, "g"), ("LnEHi", E_HI, "g"), ("LnFLo", F_LO, "g"), ("LnFHi", F_HI, "g"), ("LnHit", HIT, "g"), ("LnTop", TOP1, "g"), ("LnMargin", L1_MARGIN, "g")):
    num(nm, v, fm, LC_SRC, nm)
num("LnShare", L1_SHARE * 100, ".0f", LC_SRC, "share of episodes / pairs positive (75 %)")

# ---------------------------------------------------------------- tables
def sym(b): return "Y" if b else r"--"
def f_(x, fm): return "--" if x is None else format(x, fm)
def booktabs(cols, header, rows, spec=None, extra_top="", small=True, midrules=()):
    spec = spec or ("l" + "r" * (cols - 1))
    out = [r"\begin{tabular}{%s}" % spec, r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules: out.append(r"\midrule")
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)

def write(name, s):
    with open(os.path.join(GEN, name), "w") as f:
        f.write(s + "\n")

# T1 decision table (per run)
rows = []
for gk in ["Old", "Gam", "H", "T", "HD", "TD", "HF", "TF", "HDF", "TDF"]:
    for sd in range(4):
        r = next((x for x in grp_runs(gk) if x["seed"] == sd), None)
        if r is None:
            assert gk not in ("H", "T"), f"missing candidate-setting seed {sd} of group {gk}"
            continue
        L = r["L"]
        rows.append([GROUP_LABEL[gk] if sd == 0 or True else "", str(sd), r["ckpt"], f"{r['epoch'] * STEPS_PER_EPOCH:,}".replace(",", r"\,") if r["epoch"] else "--",
                     sym(L["L1"]), sym(L["L2"]), sym(L["L3"]), sym(L["L4"]), sym(L["L5"]), f"{r['met']}/5", "yes" if r["stamped"] else "no"])
write("tab_decision.tex", booktabs(11, ["Setting", "Seed", "Ckpt", "Step", "L1", "L2", "L3", "L4", "L5", "Met", "Stamp"], rows, spec="llrrccccccc".replace("llrr", "llll")))

# T2 per-run measures (appendix) and group summary (main text)
rows = []
for gk in GROUP_ORDER:
    for r in grp_runs(gk):
        m = r["m"]
        rows.append([GROUP_LABEL[gk], str(r["seed"]), f_(m["auc"], ".3f"), f_(m["auc_off"], ".3f"), f_(m["energy_px"], ".2f"), f_(m["flags"], ".2f"), f_(m["hit"], ".2f"),
                     f_(m["top1"], ".2f"), f_(m["err3"], ".2f"), "Y" if r["L"]["L5"] else "--"])
write("tab_runs_all.tex", booktabs(10, ["Setting", "Seed", "AUC", "AUC off", "Energy", "Flags", "Hit", "Hottest share", "Err 3\\,km", "Del."], rows, spec="llrrrrrrrc"))

def rngs(vals, fm):
    v = [x for x in vals if x is not None]
    if not v: return "--"
    if len(v) == 1: return format(v[0], fm)
    return f"{st.mean(v):{fm[1:]}} [{min(v):{fm[1:]}}--{max(v):{fm[1:]}}]".replace("[", "(").replace("]", ")") if False else f"{format(st.mean(v), fm)} ({format(min(v), fm)}--{format(max(v), fm)})"

rows = []
for gk in ["Old", "Gam", "H", "T", "HD", "TD", "HF", "TF", "HDF", "TDF"]:
    rs = grp_runs(gk)
    if not rs: continue
    f = lambda k, fm: rngs([r["m"][k] for r in rs], fm)
    rows.append([GROUP_LABEL[gk], str(len(rs)), f("auc", ".3f"), f("auc_off", ".3f"), f("energy_px", ".2f"), f("flags", ".2f"), f("top1", ".2f"), f("err3", ".2f"), f"{sum(r['L']['L5'] for r in rs)}/{len(rs)}"])
rows.append(["\\emph{even spread}", "--", f_(BM["uniform"]["auc"], ".3f"), "--", "--", f_(BM["uniform"]["flags"], ".2f"), f_(BM["uniform"]["top1"], ".2f"), f_(BM["uniform"]["err3"], ".2f"), "--"])
rows.append(["\\emph{Oracle-background spread}", "--", f_(BM["excess_only"]["auc"], ".3f"), "--", "--", f_(BM["excess_only"]["flags"], ".2f"), "--", f_(BM["excess_only"]["err3"], ".2f"), "--"])
write("tab_groups.tex", booktabs(9, ["Setting", "$n$", "Block AUC", "AUC, fire off", "Energy (px)", "Flags", "Hottest share", "Err 3\\,km", "Del."], rows, spec="lrlllllll", midrules=(len(rows) - 2,)))

# T3 baselines table
rows = []
for k, lab in (("uniform", "Even spread"), ("excess_only", "Oracle-background spread"), ("copy_planck", "Planck copy of MODIS"), ("copy_raw", "Raw copy of MODIS"), ("const", "Constant")):
    m = BM[k]
    rows.append([lab, f_(m["auc"], ".3f"), f_(m["err3"], ".2f"), f_(m["err_px"], ".2f"), f_(m["energy_blk"], ".2f"), f_(m["hit"], ".2f"), f_(m["fa"], ".2f"), f_(m["csi"], ".2f"), f_(m["top1"], ".2f")])
write("tab_baselines.tex", booktabs(9, ["Method", "Block AUC", "Err 3\\,km", "Err px", "Energy 3\\,km", "Hit", "False al.", "CSI", "Hottest share"], rows, spec="lrrrrrrrr"))

# T4 pick rule (needs all eight best-validation files)
SETT = {"100": ("H", "gammabw100-nf2-s"), "1000": ("T", "gammabw1000-nf2-s")}
have8 = all(f"{pre}{sd}" in RUNS for _, pre in SETT.values() for sd in range(4))
have8_stamped = have8 and all(RUNS[f"{pre}{sd}"]["stamped"] for _, pre in SETT.values() for sd in range(4))
rows = []
tot = {}
for w, (gk, pre) in SETT.items():
    n_all, passes = 0, 0
    for sd in range(4):
        r = RUNS.get(f"{pre}{sd}")
        assert r is not None, f"missing candidate-setting seed {sd} of BW{w}"
        v = r["m"]; L = r["L"]
        rows.append([f"BW{w}", str(sd), f"{r['epoch']}", f"{v['energy_px']:.2f} {sym(L['L2'])}", f"{v['flags']:.2f} / {v['hit']:.2f} {sym(L['L3'])}", f"{v['top1']:.2f} {sym(L['L4'])}", sym(L["L5"]), sym(r["all25"]), "yes" if r["stamped"] else "no"])
        n_all += r["all25"]; passes += r["met25"]
    tot[w] = (n_all, passes)
write("tab_pick.tex", booktabs(9, ["Setting", "Seed", "Epoch", "L2 energy (px)", "L3 flags / hit", "L4 top-1", "L5 amount", "L2--5 all", "Stamp"], rows, spec="llrllccc c".replace(" ", ""), midrules=(4,)))
pick = None
if have8:
    a, b = tot["100"], tot["1000"]
    if a[0] != b[0]: win, step = ("100" if a[0] > b[0] else "1000"), 1
    elif a[1] != b[1]: win, step = ("100" if a[1] > b[1] else "1000"), 2
    else: win, step = "1000", 3
    pick = dict(winner=win, step=step, tot=tot, stamped=have8_stamped, candidate=f"{SETT[win][1]}0", epoch=RUNS[f"{SETT[win][1]}0"]["epoch"])
    num("PickWinnerWeight", int(win), "d", "results/raw/select/gammabw*-nf2-s[0-3]/score_select.json", "scripts/apply_pick_rule.py logic (model_pick_rule.json)")
    num("PickStep", step, "d", "same", "step of the rule that decided")
    num("PickSeedsHundred", a[0], "d", "same", "seeds meeting all of lines 2-5, weight 100")
    num("PickSeedsThousand", b[0], "d", "same", "seeds meeting all of lines 2-5, weight 1000")
    num("PickPassesHundred", a[1], "d", "same", "line passes of 16, weight 100")
    num("PickPassesThousand", b[1], "d", "same", "line passes of 16, weight 1000")

# T5 descriptive measures (post hoc), from the CSV
DESC = list(csv.DictReader(open(os.path.join(RAW, "_descriptive", "descriptive_measures.csv"))))
DESC_SRC = "results/raw/_descriptive/descriptive_measures.csv"
def dget(run, measure, level, setname="select", subject="model"):
    for r in DESC:
        if r["run"] == run and r["measure"] == measure and r["level"] == level and r["set"] == setname and r["subject"] == subject and r["value"] != "":
            return float(r["value"])
    return None
DESCM = {}
for run, r in RUNS.items():
    DESCM[run] = dict(csi=dget(run, "csi", "3km_block"), skill_even=dget(run, "amount_skill_vs_even", "3km_block"), skill_exc=dget(run, "amount_skill_vs_excess", "3km_block"),
                      calib=dget(run, "calibration_abs_log_ratio", "3km_block"), fs_csi=dget(run, "fire_share_csi", "3km_block"), fs_auc=dget(run, "fire_share_auc_block", "3km_block"))
for _run, _r in RUNS.items():  # CSI exactly from the stored hit rate and false-alarm share (as defined in the paper), not the builder's rounded copy
    if _r["m"].get("csi") is not None: DESCM[_run]["csi"] = _r["m"]["csi"]
rows = []
for gk in ["Old", "Gam", "H", "T", "HD", "TD", "HF", "TF", "HDF", "TDF"]:
    rs = grp_runs(gk)
    if not rs: continue
    f = lambda k, fm: rngs([DESCM[r["run"]][k] for r in rs], fm)
    rows.append([GROUP_LABEL[gk], str(len(rs)), f("csi", ".2f"), rngs([r["m"]["flags"] for r in rs], ".2f"), rngs([r["m"]["fa"] for r in rs], ".2f"), f("skill_even", ".2f"), f("skill_exc", ".2f"), f("calib", ".2f")])
def basedesc(sub, meas, lvl="3km_block"):
    return dget("baselines", meas, lvl, "select", sub)
rows.append(["\\emph{even spread}", "--", f_(BM["uniform"]["csi"], ".2f"), f_(BM["uniform"]["flags"], ".2f"), f_(BM["uniform"]["fa"], ".2f"), "0", f_(basedesc("uniform", "amount_skill_vs_excess"), ".2f"), "--"])
rows.append(["\\emph{Oracle-background spread}", "--", f_(BM["excess_only"]["csi"], ".2f"), f_(BM["excess_only"]["flags"], ".2f"), f_(BM["excess_only"]["fa"], ".2f"), f_(basedesc("excess_only", "amount_skill_vs_even"), ".2f"), "0", "--"])
write("tab_desc.tex", booktabs(8, ["Setting", "$n$", "CSI", "Flag ratio", "False-alarm share", "Skill vs even", "Skill vs oracle-bg.", "Calib. gap"], rows, spec="lrlllllll", midrules=(len(rows) - 2,)))
for gk in ["Old", "H", "T", "Gam"]:
    rs = grp_runs(gk)
    for k, lab in (("csi", "Csi"), ("skill_even", "SkillEven"), ("skill_exc", "SkillExc"), ("calib", "Calib"), ("fs_csi", "FsCsi"), ("fs_auc", "FsAuc")):
        v = [DESCM[r["run"]][k] for r in rs if DESCM[r["run"]][k] is not None]
        if v:
            fm = ".2f"
            num(f"Desc{gk}{lab}Min", min(v), fm, DESC_SRC, f"min over seeds group {gk}: {k}"); num(f"Desc{gk}{lab}Max", max(v), fm, DESC_SRC, f"max over seeds group {gk}: {k}")
# brightness-weighted all (bound 0.25 and 0.5, best-val and final) ranges, as quoted in the post-hoc paragraph
bw_all = [r for r in RUNS.values() if r["fam"] == "gammabw"]
for k, lab in (("skill_even", "SkillEven"), ("skill_exc", "SkillExc"), ("csi", "Csi"), ("fs_csi", "FsCsi"), ("fs_auc", "FsAuc")):
    v = [DESCM[r["run"]][k] for r in bw_all if DESCM[r["run"]][k] is not None]
    if v:
        num(f"DescBwAll{lab}Min", min(v), ".2f", DESC_SRC, f"min over all brightness-weighted runs: {k}"); num(f"DescBwAll{lab}Max", max(v), ".2f", DESC_SRC, f"max over all brightness-weighted runs: {k}")
num("DescBwAllN", len(bw_all), "d", DESC_SRC, "brightness-weighted runs scored")
num("DescBwAllFlagsMin", min(r["m"]["flags"] for r in bw_all), ".2f", "results/raw/select/*/score_select.json", "summary.rule.model.flag_count_ratio")
num("DescBwAllFlagsMax", max(r["m"]["flags"] for r in bw_all), ".2f", "results/raw/select/*/score_select.json", "summary.rule.model.flag_count_ratio")

# T6 hyper-parameters (from the run record of one candidate run)
ref_meta = json.load(open(os.path.join(RAW, "gammabw1000-nf2-s0", "run_meta.json")))
a, c = ref_meta["args"], ref_meta["cfg"]
HP_SRC = "results/raw/gammabw1000-nf2-s0/run_meta.json"
hp = [("Optimiser steps per epoch", f"{STEPS_PER_EPOCH}", "29,762 steps / 46 epochs (results.json)"),
      ("Epochs (cap)", str(a["epochs"]), "args.epochs"), ("Batch size (tiles)", str(a["bs"]), "args.bs"),
      ("Peak learning rate", f"{a['lr']:g}", "args.lr"), ("Weight decay", f"{a['wd']:g}", "args.wd"), ("Warm-up steps", str(a["warmup"]), "args.warmup"),
      ("Learning-rate schedule", "constant after warm-up" if a["lr_const"] else "scheduled", "args.lr_const"), ("Fire tiles per batch", f"{a['fire_batch_frac']:g}", "args.fire_batch_frac"), ("Early stopping", "none (fixed " + str(a["epochs"]) + " epochs)" if not a["stop_patience"] else str(a["stop_patience"]), "args.stop_patience"), ("Planck band constants", a["bands"], "args.bands"),
      ("Gradient-norm clip", f"{a['clip']:g}", "args.clip"), ("Trunk precision", a["amp"], "args.amp"), ("Tile size (cells)", f"{a['tile']}$\\times${a['tile']}", "args.tile"),
      ("Network width / latent channels", f"{c['net']['width']} / {c['net']['latent']}", "cfg.net"), ("Background bound $\\delta_{\\max}$", f"{c['net']['delta_max']:g}", "cfg.net.delta_max"),
      ("Max excess log-ratio $s_{\\max}$", f"{c['net']['s_max']:g}", "cfg.net.s_max"),
      ("Loss term weights mv / vm / mm / vv", " / ".join(f"{c['loss']['term_weights'][k]:g}" for k in ("mv", "vm", "mm", "vv")), "cfg.loss.term_weights"),
      ("Energy-consistency weight", f"{c['loss']['w_energy']:g}", "cfg.loss.w_energy"),
      ("Hiding window / tiles hidden", f"{c['hide']['window']}$\\times${c['hide']['window']} / p={c['hide']['p_tile']:g}", "cfg.hide"),
      ("Median background window / dilation", f"{c['median_k']}$\\times${c['median_k']} / {c['median_dil']}", "cfg.median_k, median_dil"),
      ("MODIS view kernel / $T_{\\mathrm{fire}}$", f"{c['view']['kernel']} / {c['view']['t_fire']:g} K", "cfg.view"),
      ("Minimum valid fraction per tile", f"{c['min_valid_frac']:g}", "cfg.min_valid_frac"),
      ("Day/night", c["daynight"], "cfg.daynight")]
_st = {l.split("\t")[0]: l.split("\t")[1] for l in open(os.path.join(RES, "static_numbers.tsv"), encoding="utf8") if l.strip() and not l.startswith("#")}
hp += [("Hardware per run", "one NVIDIA RTX 4090", "results/raw/gammabw1000-nf2-s0/run_meta.json")]
write("tab_hyper.tex", booktabs(2, ["Setting", "Value"], [[k, v] for k, v, _ in hp], spec="ll"))
for k, v, s_ in hp:
    pass
for nm, k in (("Wmv", "mv"), ("Wvm", "vm"), ("Wmm", "mm"), ("Wvv", "vv")):
    num("Hp" + nm, c["loss"]["term_weights"][k], "g", HP_SRC, f"cfg.loss.term_weights.{k}")
num("HpWen", c["loss"]["w_energy"], "g", HP_SRC, "cfg.loss.w_energy")
num("HpDelta", c["net"]["delta_max"], "g", HP_SRC, "cfg.net.delta_max (candidate runs 0.25)")
num("HpDeltaWide", json.load(open(os.path.join(RAW, "gammabw1000-d05-nf2-s0", "run_meta.json")))["cfg"]["net"]["delta_max"], "g", "results/raw/gammabw1000-d05-nf2-s0/run_meta.json", "cfg.net.delta_max of the bound-0.5 runs")
num("HpWidth", c["net"]["width"], "d", HP_SRC, "cfg.net.width"); num("HpLatent", c["net"]["latent"], "d", HP_SRC, "cfg.net.latent")
num("HpSmax", c["net"]["s_max"], "g", HP_SRC, "cfg.net.s_max"); num("HpSbias", c["net"]["s_bias"], "g", HP_SRC, "cfg.net.s_bias")
num("HpFireBatch", a["fire_batch_frac"], "g", HP_SRC, "args.fire_batch_frac")
num("HpHidePtile", c["hide"]["p_tile"], "g", HP_SRC, "cfg.hide.p_tile"); num("HpHideWin", c["hide"]["window"], "d", HP_SRC, "cfg.hide.window")
num("HpHuberDelta", c["loss"]["delta"], "g", HP_SRC, "cfg.loss.delta (old loss Huber, run record of a Gamma run)")
num("HpTfire", c["view"]["t_fire"], "g", HP_SRC, "cfg.view.t_fire")
num("HpWscan", c["view"]["w_scan"], "g", HP_SRC, "cfg.view.w_scan"); num("HpWtrack", c["view"]["w_track"], "g", HP_SRC, "cfg.view.w_track")
# speed from the epoch logs
import glob as _g
_ts, _tp, _ds = [], [], []
for f in sorted(_g.glob(os.path.join(RAW, "gammabw1000-nf2-s0", "log.jsonl"))):
    for l in open(f):
        r = json.loads(l); _ts.append(r["train_s"]); _tp.append(r["tiles_s"]); _ds.append((r.get("diag") or {}).get("diag_s", 0))
num("SpeedEpochSec", st.mean(_ts), ".0f", "results/raw/gammabw1000-nf2-s0/log.jsonl", "mean train_s per epoch")
num("SpeedTilesSec", st.mean(_tp), ".0f", "results/raw/gammabw1000-nf2-s0/log.jsonl", "mean tiles_s")
num("SpeedHoursTrain", sum(_ts) / 3600, ".1f", "results/raw/gammabw1000-nf2-s0/log.jsonl", "sum train_s / 3600, 46 epochs")
num("SpeedHoursDiag", sum(_ds) / 3600, ".1f", "results/raw/gammabw1000-nf2-s0/log.jsonl", "sum diag_s / 3600 (in-training evaluation)")
pk = ref_meta["data"]["data/train_night"]["pack_report"]; pv = ref_meta["data"]["data/val2_night"]["pack_report"]
num("TrainManifest", pk["manifest_tiles"], ",d", "results/raw/gammabw1000-nf2-s0/run_meta.json", "data.data/train_night.pack_report.manifest_tiles")
num("TrainKept", pk["kept"], ",d", "results/raw/gammabw1000-nf2-s0/run_meta.json", "pack_report.kept")
num("TrainDropped", pk["dropped_filter"], ",d", "results/raw/gammabw1000-nf2-s0/run_meta.json", "pack_report.dropped_filter")
num("ValManifest", pv["manifest_tiles"], ",d", "results/raw/gammabw1000-nf2-s0/run_meta.json", "data.data/val2_night.pack_report.manifest_tiles")
num("ValKept", pv["kept"], ",d", "results/raw/gammabw1000-nf2-s0/run_meta.json", "pack_report.kept")
num("StepsPerEpoch", STEPS_PER_EPOCH, "d", "results/raw/gammabw1000-nf2-s0/results.json", "29,762 / 46")
for nm, key in (("Lr", "lr"), ("Wd", "wd"), ("Bs", "bs"), ("Warmup", "warmup"), ("Epochs", "epochs")):
    num("Hp" + nm, a[key], "g" if key in ("lr", "wd") else "d", HP_SRC, f"args.{key}")

# T7 run list (appendix): commit, seed, epoch, checkpoint hash prefix recorded in the score file
rows = []
for run in sorted(RUNS, key=lambda r: (GROUP_ORDER.index(RUNS[r]["grp"]), RUNS[r]["seed"])):
    r = RUNS[run]
    meta_path = os.path.join(RAW, run.replace("-final", ""), "run_meta.json")
    commit = "--"
    if os.path.exists(meta_path):
        commit = json.load(open(meta_path)).get("git_commit", "--")[:10]
    rows.append([GROUP_LABEL[r["grp"]], r"\texttt{" + tex(run) + "}", commit, str(r["epoch"]), r"\texttt{" + str(r["ckpt16"]) + "}", (r["git"] or "--")[:7], "yes" if r["stamped"] else "no"])
write("tab_runlist.tex", booktabs(7, ["Name", "Run id", "Train commit", "Epoch", "Ckpt sha256 (16)", "Scorer", "Stamp"], rows, spec="llllllc"))

# T10 model variants (display name, loss, brightness weight, bound, run ids, seeds present)
rows = []
for gk in ["Old", "Gam", "H", "T", "HD", "TD"]:
    rs = grp_runs(gk) + grp_runs(gk + "F")
    ids = sorted({re.sub(r"-s\d(-final)?$", "", r["run"]) for r in rs})
    seeds = sorted({r["seed"] for r in rs})
    c0 = (rs[0] if rs else None)
    loss = {"Old": "group-share (Huber on ln radiance, fixed shares for background, near-fire and fire-level cells from the target)", "Gam": "Gamma deviance"}.get(gk, "Gamma deviance")
    rows.append([GROUP_LABEL[gk].replace(" (best-val)", ""), loss, "0" if gk in ("Old", "Gam") else ("100" if gk[0] == "H" else "1000"), "0.25" if "D" not in gk else "0.5", r"\texttt{" + tex(ids[0] if ids else "--") + "}", ",".join(map(str, seeds))])
write("tab_variants.tex", booktabs(6, ["Name", "Loss", "BW", "Bound", "Run id (prefix)", "Seeds"], rows, spec="lp{5.2cm}rrll"))
# T8 loss family sweep (small data, old validation, sweep 1) from rows.jsonl
SW = [json.loads(l) for l in open(os.path.join(RAW, "sweep1", "rows.jsonl"))]
def sw_class(r):
    n = r["run"]; net = "U-Net" if n.startswith("unet") else "3-conv"
    if "default-default" in n: fam = "Group-share"
    elif "fam-gamma" in n: fam = "Gamma"
    elif "fam-poisson" in n: fam = "Poisson"
    elif "fam-nmse" in n: fam = "NMSE"
    else: return None
    return net, fam
rows = []
SWG = {}
for r in SW:
    c = sw_class(r)
    if c: SWG.setdefault(c, []).append(r)
for (net, fam), rs in sorted(SWG.items(), key=lambda kv: (kv[0][0] != "U-Net", ["Group-share", "Gamma", "Poisson", "NMSE"].index(kv[0][1]))):
    rs = [x for x in rs if "repeat" not in x["run"]]
    rows.append([net, fam, str(len(rs)), rngs([x["auc_model"] for x in rs], ".3f"), rngs([x["vs_uniform"]["diff"] for x in rs], "+.3f"), rngs([x["energy_msel"] for x in rs if x.get("energy_msel") is not None], ".2f"), rngs([x["vs_model_off"]["diff"] for x in rs], "+.3f"), f"{sum(1 for x in rs if x['vs_uniform']['lo'] > 0)}/{len(rs)}"])
write("tab_lossfamily.tex", booktabs(8, ["Net", "Loss", "$n$", "Block AUC", "Gain vs even", "Energy (MODIS-bright)", "Gain vs fire-off", "Range $>0$"], rows, spec="llrlllll", midrules=(4,)))
uf = SWG.get(("U-Net", "Group-share"), [])
num("SweepOldN", len(uf), "d", "results/raw/sweep1/rows.jsonl", "U-Net old-loss rows")
nf = [x for k, v in SWG.items() if k[1] != "Group-share" for x in v if "repeat" not in x["run"]]
num("SweepNewN", len(nf), "d", "results/raw/sweep1/rows.jsonl", "new-loss rows (all, excluding repeat trainings)")
num("SweepNewAboveZero", sum(1 for x in nf if x["vs_uniform"]["lo"] > 0), "d", "results/raw/sweep1/rows.jsonl", "new-loss runs with 95% range above zero vs even spread")
ol = [x for k, v in SWG.items() if k[1] == "Group-share" for x in v]
num("SweepOldAboveZero", sum(1 for x in ol if x["vs_uniform"]["lo"] > 0), "d", "results/raw/sweep1/rows.jsonl", "old-loss runs above zero")
num("SweepOldRuns", len(ol), "d", "results/raw/sweep1/rows.jsonl", "old-loss runs")
num("SweepNewEnMin", min(x["energy_msel"] for x in nf if x.get("energy_msel")), ".2f", "results/raw/sweep1/rows.jsonl", "min energy_msel new losses")
num("SweepNewEnMax", max(x["energy_msel"] for x in nf if x.get("energy_msel")), ".2f", "results/raw/sweep1/rows.jsonl", "max energy_msel new losses")
num("SweepOldEnMin", min(x["energy_msel"] for x in ol), ".2f", "results/raw/sweep1/rows.jsonl", "min energy_msel old loss")
num("SweepOldEnMax", max(x["energy_msel"] for x in ol), ".2f", "results/raw/sweep1/rows.jsonl", "max energy_msel old loss")
num("SweepNewAucMin", min(x["auc_model"] for x in nf), ".3f", "results/raw/sweep1/rows.jsonl", "min auc new losses")
num("SweepNewAucMax", max(x["auc_model"] for x in nf), ".3f", "results/raw/sweep1/rows.jsonl", "max auc new losses")
num("SweepOldAucMin", min(x["auc_model"] for x in ol), ".3f", "results/raw/sweep1/rows.jsonl", "min auc old loss")
num("SweepOldAucMax", max(x["auc_model"] for x in ol), ".3f", "results/raw/sweep1/rows.jsonl", "max auc old loss")

# T9 width test from wide/*.json
rows = []
WIDE = []
for f in sorted(glob.glob(os.path.join(RAW, "wide", "*.json"))):
    w = json.load(open(f)); n = os.path.basename(f).replace("_3k.json", "")
    m = re.fullmatch(r"w(\d+)l(\d+)_(gamma|legacy)_s(\d)", n)
    WIDE.append(dict(width=int(m[1]), latent=int(m[2]), loss="Gamma" if m[3] == "gamma" else "Group-share", seed=int(m[4]), params=w["n_params"], start=w["loss_start"], end=w["loss_end"], fall=w["fall_factor"],
                     fire_err=w["fidelity_end"]["med_abs_ln_viirs_fire"], en=w["fidelity_end"]["energy_ratio_bright_cells"], src=rel(f)))
for w in sorted(WIDE, key=lambda w: (w["loss"] != "Gamma", w["params"], w["seed"])):
    rows.append([w["loss"], f"{w['width']}/{w['latent']}", f"{w['params']:,}".replace(",", r"\,"), str(w["seed"]), f"{w['fall']:.1f}$\\times$", f"{w['fire_err']:.3f}", f"{w['en']:.2f}"])
write("tab_width.tex", booktabs(7, ["Loss", "Width/latent", "Parameters", "Seed", "Loss fall", "Fire-cell error", "Energy ratio"], rows, spec="lllrrrr", midrules=(sum(1 for w in WIDE if w["loss"] == "Gamma"),)))
num("WideMaxParams", max(w["params"] for w in WIDE), ",d", "results/raw/wide/*_3k.json", "n_params of the widest run")
num("WideMinParams", min(w["params"] for w in WIDE), ",d", "results/raw/wide/*_3k.json", "n_params of the default run (32/16)")
num("WideRuns", len(WIDE), "d", "results/raw/wide/*_3k.json", "runs")
gm = [w for w in WIDE if w["loss"] == "Gamma"]; ol_ = [w for w in WIDE if w["loss"] == "Group-share"]
num("WideGamFallMin", min(w["fall"] for w in gm), ".1f", "results/raw/wide/*_3k.json", "fall_factor Gamma"); num("WideGamFallMax", max(w["fall"] for w in gm), ".1f", "results/raw/wide/*_3k.json", "fall_factor Gamma")
num("WideOldFallMin", min(w["fall"] for w in ol_), ".1f", "results/raw/wide/*_3k.json", "fall_factor old"); num("WideOldFallMax", max(w["fall"] for w in ol_), ".1f", "results/raw/wide/*_3k.json", "fall_factor old")

# placement probe (hottest-cell hit rate) from results/raw/placement/placement.json, block val2far (33 pairs, 1,023 footprints)
PL = json.load(open(os.path.join(RAW, "placement", "placement.json")))["runs"]
PL_SRC = "results/raw/placement/placement.json"
def pl_var(name):
    m = re.fullmatch(r"placement-(.+)-s(\d)", name); return m[1]
PLV = {}
for nm, r in PL.items():
    vf = r.get("val2far")
    if vf: PLV.setdefault(pl_var(nm), []).append((int(nm[-1]), vf))
for var, key in (("old", "Old"), ("oldnf", "OldNf"), ("oldnf-short", "Short"), ("oldnf-geometry_only", "Geom"), ("oldnf-radiance_only", "Rad"), ("oldnf-modis_mask", "Mask")):
    v = [vf["pooled"]["hit"]["model"]["value"] for _, vf in PLV.get(var, [])]
    num(f"Pl{key}Min", min(v), ".2f", PL_SRC, f"val2far.pooled.hit.model.value min over seeds, variant {var}"); num(f"Pl{key}Max", max(v), ".2f", PL_SRC, f"max over seeds, variant {var}"); num(f"Pl{key}N", len(v), "d", PL_SRC, f"seeds, variant {var}")
allv = [vf for l in PLV.values() for _, vf in l]
num("PlChanceMin", min(vf["pooled"]["hit"]["chance"]["value"] for vf in allv), ".2f", PL_SRC, "chance hit rate min"); num("PlChanceMax", max(vf["pooled"]["hit"]["chance"]["value"] for vf in allv), ".2f", PL_SRC, "chance max")
num("PlPsfMin", min(vf["pooled"]["hit"]["psf_nbr"]["value"] for vf in allv), ".2f", PL_SRC, "point-spread-with-neighbours rule min"); num("PlPsfMax", max(vf["pooled"]["hit"]["psf_nbr"]["value"] for vf in allv), ".2f", PL_SRC, "max")
num("PlCentreMax", max(vf["pooled"]["hit"]["centre"]["value"] for vf in allv), ".2f", PL_SRC, "nearest-to-pixel-centre rule max")
num("PlFootprints", allv[0]["n_footprints"], ",d", PL_SRC, "n_footprints"); num("PlPairs", allv[0]["n_pairs_all"], "d", PL_SRC, "n_pairs_all")
# line-5 margin: the share of scene pairs in which the model beats every simple method, for the runs that meet line 5
_pass = [r for r in GF if r["L"]["L5"]]
if _pass:
    sh = [min(v["share_pairs_better"] for k, v in r["j"]["amount"]["block3"]["decision"]["vs"].items() if k in ("uniform", "copy_planck", "copy_raw", "excess_only", "const")) for r in _pass]
    num("GFPassMinShare", min(sh) * 100, ".0f", "results/raw/select/*/score_select.json", "amount.block3.decision.vs[*].share_pairs_better, min over simple methods and over passing runs (%)")
    num("GFPassMinPairs", round(min(sh) * (npairs[0] if npairs else 0)), "d", "results/raw/select/*/score_select.json", "same, in pairs")
num("PairsNeeded", math.ceil(L1_SHARE * npairs[0] - 1e-9) if npairs else None, "d", LC_SRC, "ceil(0.75 x qualifying pairs)")

PK_SRC = "results/raw/_closed_test_prep/pick.json"
if os.path.exists(os.path.join(os.path.dirname(RES), PK_SRC)) or os.path.exists(os.path.join(RES, PK_SRC[len("results/"):])):
    PKJ = json.load(open(os.path.join(RES, PK_SRC[len("results/"):])))
    cb = PKJ["candidate_block_for_closed_test_candidates_json"]
    num("PkWinnerWeight", int(PKJ["winner_weight"]), "d", PK_SRC, "winner_weight"); num("PkStep", PKJ["decided_at_step"], "d", PK_SRC, "decided_at_step")
    for w in ("100", "1000"):
        num(f"PkSeeds{w}", PKJ["totals"][w]["seeds_meeting_all_2_5"], "d", PK_SRC, f"totals.{w}.seeds_meeting_all_2_5"); num(f"PkPass{w}", PKJ["totals"][w]["line_passes_of_16"], "d", PK_SRC, f"totals.{w}.line_passes_of_16")
    num("PkCandEpoch", PKJ["candidate_epoch"], "d", PK_SRC, "candidate_epoch")
    cr = RUNS[PKJ["candidate_run"]]["m"]
    num("PkCandTop", cr["top1"], ".4f", "results/raw/select/" + PKJ["candidate_run"] + "/score_select.json", "summary.conc.model.top1_share")
    num("PkCandSha", cb["sha256"][:16], "s", PK_SRC, "candidate sha256 (first 16 hex; full value in the file)")
    CAND_SHA_FULL = cb["sha256"]

# ---------------------------------------------------------------- post-hoc metrics2 block AP, FSS, hottest-cell hit rate, amount skill distribution
M2 = json.load(open(os.path.join(RES, "metrics2_summary.json")))["select"]
M2_SRC = "results/metrics2_summary.json (compacted from S3 big/_descriptive/metrics2/select/*/metrics2_select.json by scripts/extract_metrics2.py)"
def m2grp(pred):
    return [v for k, v in M2.items() if k != "baselines" and pred(k)]
M2G = {"GS": m2grp(lambda k: k.startswith("oldloss-nf2-s")), "Gam": m2grp(lambda k: k.startswith("gamma-nf2-s")),
       "BW": m2grp(lambda k: k.startswith("gammabw") and "-d05" not in k and not k.endswith("-final")),
       "BWA": m2grp(lambda k: k.startswith("gammabw100-nf2-s") and not k.endswith("-final")), "BWB": m2grp(lambda k: k.startswith("gammabw1000-nf2-s") and not k.endswith("-final"))}
M2K = (("ap_block", "Ap", ".2f"), ("fss1", "FssA", ".2f"), ("fss9", "FssB", ".2f"), ("fss27", "FssC", ".2f"), ("exact", "Exact", ".2f"), ("exact_flag", "ExactFlag", ".2f"), ("within1", "Within", ".2f"), ("skill_mean", "Skill", ".2f"), ("skill_share", "Share", ".2f"), ("skill_min", "Worst", ".2f"))
for gk, rs in M2G.items():
    num(f"M2{gk}N", len(rs), "d", M2_SRC, f"runs in group {gk}")
    for key, lab, fm in M2K:
        v = [r["methods"]["model"][key] for r in rs if r["methods"]["model"].get(key) is not None]
        if v:
            num(f"M2{gk}{lab}Min", min(v), fm, M2_SRC, f"min over seeds {gk}: {key}"); num(f"M2{gk}{lab}Max", max(v), fm, M2_SRC, f"max over seeds {gk}: {key}")
        vo = [r["methods"]["model_off"][key] for r in rs if r["methods"].get("model_off", {}).get(key) is not None]
        if vo:
            num(f"M2{gk}Off{lab}Min", min(vo), fm, M2_SRC, f"fire-off min {gk}: {key}"); num(f"M2{gk}Off{lab}Max", max(vo), fm, M2_SRC, f"fire-off max {gk}: {key}")
for meth, nm in (("uniform", "Even"), ("excess_only", "Orac"), ("copy_planck", "Planck")):
    for key, lab, fm in M2K[:7]:
        v = M2["baselines"]["methods"].get(meth, {}).get(key)
        if v is not None: num(f"M2{nm}{lab}", v, fm, M2_SRC, f"baseline {meth}: {key}")
num("M2Chance", M2["baselines"]["methods"]["uniform"]["chance_exact"], ".2f", M2_SRC, "chance exact hit rate")
_lf = [r["methods"]["model"]["exact_flag"] for g_ in ("GS", "Gam", "BW") for r in M2G[g_] if r["methods"]["model"].get("exact_flag") is not None]
num("M2LearnedExactFlagMin", min(_lf), ".2f", M2_SRC, "min over best-val Group-share, Gamma, Gamma-BW seeds: hottest cell exact, pixels flagged in VIIRS (pair mean)")
num("M2LearnedExactFlagMax", max(_lf), ".2f", M2_SRC, "max over the same")
_lx = [r["methods"]["model"]["exact"] for g_ in ("GS", "Gam", "BW") for r in M2G[g_] if r["methods"]["model"].get("exact") is not None]
num("M2LearnedExactMin", min(_lx), ".2f", M2_SRC, "min over best-val Group-share, Gamma, Gamma-BW seeds: hottest cell exact, MODIS fire-level pixels (pair mean)")
num("M2LearnedExactMax", max(_lx), ".2f", M2_SRC, "max over the same")
# final-epoch concentration of the bound-0.25 brightness-weighted settings
_ft = [r["m"]["top1"] for g_ in ("HF", "TF") for r in grp_runs(g_)]
num("FinalTopMin", min(_ft), ".2f", "results/raw/select/*/score_select.json", "summary.conc.model.top1_share, final-epoch Gamma-BW100 and Gamma-BW1000 (bound 0.25)")
num("FinalTopMax", max(_ft), ".2f", "results/raw/select/*/score_select.json", "same")
# line 1 as it is judged: region-balanced, episode-clustered AUC difference to the better simple method
def _bal(j):
    bv = j["summary"].get("balanced_vs") or {}
    c = {k: bv[k] for k in ("uniform", "excess_only") if bv.get(k) and bv[k].get("diff") is not None}
    return c[min(c, key=lambda k: c[k]["diff"])], bv.get("uniform")
for g_, nm in (("Old", "Old"), ("H", "H"), ("T", "T")):
    vs = [_bal(r["j"]) for r in grp_runs(g_)]
    num(f"Bal{nm}DiffMin", min(v[0]["diff"] for v in vs), ".2f", "results/raw/select/*/score_select.json", "summary.balanced_vs: difference to the better simple method (line 1 statistic), min over seeds")
    num(f"Bal{nm}DiffMax", max(v[0]["diff"] for v in vs), ".2f", "results/raw/select/*/score_select.json", "same, max over seeds")
    num(f"Bal{nm}LoMin", min(v[0]["lo"] for v in vs), ".2f", "results/raw/select/*/score_select.json", "lower end of the episode-clustered 95% range, min over seeds")
    num(f"Bal{nm}ShareMin", min(v[0]["share_pos_episodes"] for v in vs) * 100, ".0f", "results/raw/select/*/score_select.json", "share of episodes positive (%), min over seeds")
    num(f"Bal{nm}EvenMin", min(v[1]["diff"] for v in vs), ".2f", "results/raw/select/*/score_select.json", "balanced difference to the Even spread, min over seeds")
    num(f"Bal{nm}EvenMax", max(v[1]["diff"] for v in vs), ".2f", "results/raw/select/*/score_select.json", "same, max over seeds")
    num(f"Bal{nm}EvenLoMin", min(v[1]["lo"] for v in vs), ".3f", "results/raw/select/*/score_select.json", "lower end of the 95% range of the difference to the Even spread, min over seeds")
_om = [r["m"]["auc"] - BM["excess_only"]["auc"] for r in grp_runs("Old")]
num("OldPairMarginMin", min(_om), ".3f", "results/raw/select/*/score_select.json", "pair-mean AUC of Group-share minus Oracle-background spread, min over seeds")
num("OldPairMarginMax", max(_om), ".3f", "results/raw/select/*/score_select.json", "same, max")
_gf = [_bal(r["j"])[0]["diff"] for r in RUNS.values() if r["fam"] in ("gamma", "gammabw")]
num("BalGFDiffMax", max(_gf), ".2f", "results/raw/select/*/score_select.json", "best Gamma-family run: balanced difference to the better simple method")
num("PairEvenGainMin", min(RUNS[k]["m"]["auc"] for k in RUNS if RUNS[k]["grp"] in ("H", "T")) - BM["uniform"]["auc"], ".3f", "results/raw/select/*/score_select.json", "pair-mean AUC of Gamma-BW (best-val, bound 0.25) minus Even spread, min over seeds")
num("PairEvenGainMax", max(RUNS[k]["m"]["auc"] for k in RUNS if RUNS[k]["grp"] in ("H", "T")) - BM["uniform"]["auc"], ".3f", "results/raw/select/*/score_select.json", "same, max over seeds")
rows = []
for gk, lab in (("GS", "Group-share"), ("Gam", "Gamma"), ("BWA", "Gamma-BW100"), ("BWB", "Gamma-BW1000")):
    rs = M2G[gk]
    f = lambda k, fm: rngs([r["methods"]["model"][k] for r in rs], fm)
    f2 = lambda k, fm: rngs([r["methods"]["model_off"][k] for r in rs], fm)
    rows.append([lab, str(len(rs)), f("ap_block", ".2f"), f("fss1", ".2f"), f("fss9", ".2f"), f("exact", ".2f"), f("skill_mean", ".2f"), f("skill_share", ".2f"), f("skill_min", ".2f")])
for meth, lab in (("uniform", "Even spread"), ("excess_only", "Oracle-background spread")):
    b = M2["baselines"]["methods"][meth]
    rows.append(["\\emph{" + lab + "}", "--"] + [f_(b.get(k), ".2f") for k in ("ap_block", "fss1", "fss9", "exact")] + ["0" if meth == "uniform" else "--", "--", "--"])
rows.append(["\\emph{Chance}", "--", "--", "--", "--", f_(M2["baselines"]["methods"]["uniform"]["chance_exact"], ".2f"), "--", "--", "--"])
write("tab_metrics2.tex", booktabs(9, ["Setting", "$n$", "Block AP", "FSS 375\\,m", "FSS 3\\,km", "Hottest exact", "Amount skill", "Pairs $>0$", "Worst pair"], rows, spec="lrlllllll", midrules=(4,)))
rows = []
for gk, lab in (("GS", "Group-share"), ("Gam", "Gamma"), ("BWA", "Gamma-BW100"), ("BWB", "Gamma-BW1000")):
    rs = M2G[gk]
    rows.append([lab] + [rngs([r["methods"]["model_off"][k] for r in rs], ".2f") for k in ("ap_block", "fss1", "fss9", "exact", "skill_mean")])
write("tab_metrics2_off.tex", booktabs(6, ["Setting (fire output off)", "Block AP", "FSS 375\\,m", "FSS 3\\,km", "Hottest exact", "Amount skill"], rows, spec="lllll l".replace(" ", "")))

# ---------------------------------------------------------------- closed test (one run; night pack decides): results/raw/_closed_test/
CT_DIR = os.path.join(RAW, "_closed_test")
CTV = json.load(open(os.path.join(CT_DIR, "verdict_night.json")))
CTD = json.load(open(os.path.join(CT_DIR, "verdict_day.json")))
CTS = json.load(open(os.path.join(CT_DIR, "score_night.json")))
CTR = json.load(open(os.path.join(CT_DIR, "closed_test_record.json")))
CT_SRC = "results/raw/_closed_test/verdict_night.json"
CT_SC = "results/raw/_closed_test/score_night.json"
_d = CTV["all_blocks_decides"]
_l1, _l2, _l3, _l4 = _d["line1_fire_block_score"], _d["line2_amount"], _d["line3_hottest_cell_share"], _d["line4_flag_count"]
_au = CTS["summary"]["auc"]; _am = CTS["amount"]; _vs = CTS["summary"]["vs"]
assert CTV["overall"] == "FAIL" and CTR["verdict_night"] == "FAIL"
num("CtTiles", CTV["n_tiles"], "d", CT_SRC, "n_tiles")
num("CtDayTiles", CTD["n_tiles"], "d", "results/raw/_closed_test/verdict_day.json", "n_tiles")
num("CtPairs", _l1["n_pairs"], "d", CT_SRC, "line1.n_pairs")
num("CtPairsBlk", _l2["per_scale"]["block3"]["n_pairs"], "d", CT_SRC, "line2.block3.n_pairs")
num("CtModelAuc", _l1["model_auc"], ".3f", CT_SRC, "line1.model_auc")
num("CtOracleAuc", _l1["comparator_auc"], ".3f", CT_SRC, "line1.comparator_auc (Oracle-background spread)")
num("CtEvenAuc", _au["uniform"]["mean"], ".3f", CT_SC, "summary.auc.uniform.mean")
num("CtOffAuc", _au["model_off"]["mean"], ".3f", CT_SC, "summary.auc.model_off.mean")
num("CtDiffOracle", -_l1["diff"], ".3f", CT_SRC, "line1.diff (sign flipped: shortfall)")
num("CtDiffOracleLo", -_l1["range"][1], ".3f", CT_SRC, "line1.range")
num("CtDiffOracleHi", -_l1["range"][0], ".3f", CT_SRC, "line1.range")
num("CtDiffOff", _l1["diff_vs_fire_off"], ".3f", CT_SRC, "line1.diff_vs_fire_off")
num("CtDiffOffLo", _vs["model_off"]["lo"], ".3f", CT_SC, "summary.vs.model_off.lo")
num("CtDiffOffHi", _vs["model_off"]["hi"], ".3f", CT_SC, "summary.vs.model_off.hi")
num("CtPixRatio", _l2["per_scale"]["pixel"]["ratio_of_sums"], ".2f", CT_SRC, "line2.pixel.ratio_of_sums")
num("CtBlkRatio", _l2["per_scale"]["block3"]["ratio_of_sums"], ".2f", CT_SRC, "line2.block3.ratio_of_sums")
num("CtPixPairs", _l2["per_scale"]["pixel"]["n_pairs"], "d", CT_SRC, "line2.pixel.n_pairs")
for _k, _nm in (("model", "Model"), ("uniform", "Even"), ("copy_planck", "Planck"), ("excess_only", "Oracle")):
    num("CtPixErr" + _nm, _am["pixel"]["methods"][_k]["abs_log_err_pair_mean"], ".2f", CT_SC, f"amount.pixel.methods.{_k}.abs_log_err_pair_mean")
    num("CtBlkErr" + _nm, _am["block3"]["methods"][_k]["abs_log_err_pair_mean"], ".2f", CT_SC, f"amount.block3.methods.{_k}.abs_log_err_pair_mean")
num("CtTop", _l3["top1_share"], ".3f", CT_SRC, "line3.top1_share")
num("CtTopMeas", _l3["reference_measured"], ".3f", CT_SRC, "line3.reference_measured")
num("CtTopEven", _l3["reference_spread_evenly"], ".3f", CT_SRC, "line3.reference_spread_evenly")
num("CtTopOff", CTS["summary"]["conc"]["model_off"]["top1_share"], ".3f", CT_SC, "summary.conc.model_off.top1_share")
num("CtTopN", _l3["n_pixels"], ",.0f", CT_SRC, "line3.n_pixels")
num("CtFlagRatio", _l4["flag_count_ratio"], ".2f", CT_SRC, "line4.flag_count_ratio")
num("CtHit", _l4["block_hit_rate"], ".2f", CT_SRC, "line4.block_hit_rate")
num("CtFalse", _l4["false_alarm_share_reported_no_line"], ".2f", CT_SRC, "line4.false_alarm_share")
num("CtFlagsMeas", _l4["flagged_cells_measured"], ",.0f", CT_SRC, "line4.flagged_cells_measured")
num("CtFlagRatioOff", CTS["summary"]["rule"]["model_off"]["flag_count_ratio"], ".2f", CT_SC, "summary.rule.model_off.flag_count_ratio")
num("CtFlagRatioEven", CTS["summary"]["rule"]["uniform"]["flag_count_ratio"], ".2f", CT_SC, "summary.rule.uniform.flag_count_ratio")
num("CtHitEven", CTS["summary"]["rule"]["uniform"]["block_hit_rate"], ".2f", CT_SC, "summary.rule.uniform.block_hit_rate")
num("CtEpoch", CTV["ckpt_epoch"], "d", CT_SRC, "ckpt_epoch")
_hf = CTV["hole_free_shown_next_to_it"]
num("CtHfAuc", _hf["line1_fire_block_score"]["model_auc"], ".3f", CT_SRC, "hole_free_shown_next_to_it.line1.model_auc")
num("CtHfHit", _hf["line4_flag_count"]["block_hit_rate"], ".2f", CT_SRC, "hole_free_shown_next_to_it.line4.block_hit_rate")
num("CandSelAuc", RUNS["gammabw100-nf2-s0"]["m"]["auc"], ".3f", "results/raw/select/gammabw100-nf2-s0/score_select.json", "frozen candidate: pair-mean block AUC on SELECT")
num("CandSelHit", RUNS["gammabw100-nf2-s0"]["m"]["hit"], ".2f", "results/raw/select/gammabw100-nf2-s0/score_select.json", "frozen candidate: block hit rate on SELECT")
num("CtMinPairs", CTV["constants"]["min_pairs_lines_1_2"], "d", CT_SRC, "constants.min_pairs_lines_1_2")
CT_LINES = [("L1", _l1["status"]), ("L2", _l2["status"]), ("L3", _l3["status"]), ("L4", _l4["status"])]
NUMS.append((mn("CtFailed"), str(sum(1 for _, v in CT_LINES if v == "FAIL")), CT_SRC, "lines failed"))
NUMS.append((mn("CtPassed"), str(sum(1 for _, v in CT_LINES if v == "PASS")), CT_SRC, "lines passed"))

# ---------------------------------------------------------------- static numbers (read from wiki/code; each with its source)
STATIC = []
sp = os.path.join(RES, "static_numbers.tsv")
if os.path.exists(sp):
    for line in open(sp, encoding="utf8"):
        if not line.strip() or line.startswith("#"): continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            print("BAD static line:", line); continue
        name, val, source, note = p[0], p[1], p[2], p[3]
        NUMS.append((mn("S", name), val, source, note)); STATIC.append(name)


# ---------------------------------------------------------------- static tables: results/static_tables/<name>.tsv  (#source:, #spec:, #mid: lines, then header row, then rows)
STATIC_TABLES = []
for f in sorted(glob.glob(os.path.join(RES, "static_tables", "*.tsv"))):
    name = os.path.basename(f)[:-4]; src = spec = None; mid = (); rows_ = []
    for line in open(f, encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("#source:"): src = line[8:].strip(); continue
        if line.startswith("#spec:"): spec = line[6:].strip(); continue
        if line.startswith("#mid:"): mid = tuple(int(x) for x in line[5:].split(",") if x.strip()); continue
        if line.startswith("#") or not line.strip(): continue
        rows_.append(line.split("\t"))
    if not src:
        print("static table without #source:", f); continue
    hdr, body = rows_[0], rows_[1:]
    write(f"tab_{name}.tex", booktabs(len(hdr), hdr, body, spec=spec, midrules=mid))
    STATIC_TABLES.append((name, src))

NUMS.append((mn('S','StaticCount'), str(len(STATIC)), 'results/static_numbers.tsv', 'number of typed static macros'))
# ---------------------------------------------------------------- write numbers.tex and NUMBERS.md
seen = set()
tex_lines = ["% GENERATED by scripts/paper/gen_numbers.py -- do not edit. One macro per number; see NUMBERS.md for sources."]
md = ["# Number map", "", f"Generated from the stored result files in results/ (results/MANIFEST.json, {MAN['n_files']} files, sha256-checked).", "",
      "Every `\\N...` macro used in the paper is listed here with its value and source. Table cells are generated from the files named in the table's own entry below.", "",
      "| macro | value | source file | key / note |", "|---|---|---|---|"]
for name, val, src, key in NUMS:
    if name in seen:
        print("DUPLICATE macro", name); continue
    seen.add(name)
    tex_lines.append(r"\newcommand{\%s}{%s}" % (name, val))
    md.append(f"| `\\{name}` | {val.replace('|', '/')} | `{src}` | {key} |")
md += ["", "## Generated tables (cells computed from these files)", "",
       "| table file | source |", "|---|---|",
       "| tab_decision.tex, tab_runs_all.tex, tab_groups.tex, tab_pick.tex, tab_runlist.tex | results/raw/select/*/score_select.json (fields: summary.auc, summary.rule, summary.conc, summary.balanced_vs, amount.block3, amount.pixel, ckpt, epoch, git) |",
       "| tab_baselines.tex | results/raw/select/baselines/score_select.json |",
       "| tab_desc.tex | results/raw/_descriptive/descriptive_measures.csv (post hoc) and the score files above |",
       "| tab_hyper.tex | results/raw/gammabw1000-nf2-s0/run_meta.json (args, cfg) |",
       "| tab_lossfamily.tex | results/raw/sweep1/rows.jsonl |",
       "| tab_width.tex | results/raw/wide/*_3k.json |", "",
       "## Static tables (recorded values with the source named; every cell is a recorded value, not computed here)", "", "| table file | source |", "|---|---|"] + [f"| tab_{n}.tex | {sr} |" for n, sr in STATIC_TABLES] + ["",
       "## Files in the snapshot", "", "| local | size | sha256 (16) | note |", "|---|---|---|---|"]
for f in MAN["files"]:
    md.append(f"| `{f['local']}` | {f['size']} | `{f['sha256'][:16]}` | {f.get('note', '')} |")
write("numbers.tex", "\n".join(tex_lines))
open(os.path.join(GEN, "NUMBERS.md"), "w").write("\n".join(md) + "\n")
print(f"{len(seen)} macros ({len(STATIC)} recorded static), {len(RUNS)} runs, {len(STATIC_TABLES)} static tables -> {GEN}")
