#!/usr/bin/env python
"""Experiment C: read the four gate outputs of one configuration (gates/<cfg>/{path,seed,share,overfit}.json or .log RESULT line) and print PASS/FAIL per gate with the numbers. Criteria as in the workflow:
path: abs diff of train/eval path ~0 (< 1e-6) and the file's own pass flag; seed: pass flag (max abs loss diff < 1e-6); share: informational (numbers printed); overfit: the file's criterion_amendment5.PASS_all (bright-loss fall > 10x, median |ln error| at VIIRS fire cells < 0.1, energy ratio at MODIS-bright cells 0.9-1.1).
usage: c_gate_eval.py <gates dir of the configuration>; exit 0 = all PASS."""
import json, os, re, sys
d = sys.argv[1]; ok = True
def res(name):
    f = os.path.join(d, name + ".json")
    if os.path.exists(f):
        return json.load(open(f))
    L = [l for l in open(os.path.join(d, name + ".log")) if l.startswith("RESULT ")]
    return json.loads(L[-1][7:]) if L else None
p = res("path"); s = res("seed"); sh = res("share"); o = res("overfit")
for name, r in (("path", p), ("seed", s), ("share", sh), ("overfit", o)):
    if r is None:
        print(name, "MISSING"); ok = False
if p:
    diffs = [v.get("abs_diff") for k, v in p.items() if isinstance(v, dict) and "abs_diff" in v]; g = bool(p.get("pass", all(x is not None and x < 1e-6 for x in diffs))) and all(x < 1e-6 for x in diffs); ok &= g; print("path", "PASS" if g else "FAIL", "abs diffs", diffs)
if s:
    g = bool(s.get("pass")); ok &= g; print("seed", "PASS" if g else "FAIL", "max abs loss diff", s.get("max_abs_loss_diff"))
if sh:
    print("share (info)", {k: sh[k] for k in ("loss_share_bright", "bright_cell_fraction") if k in sh})
if o:
    c = o["criterion_amendment5"]; g = bool(c["PASS_all"]); ok &= g   # the workflow's own criterion: bright-loss fall > 10x, median |ln error| at VIIRS fire cells < 0.1, energy ratio at MODIS-bright cells 0.9-1.1
    print("overfit", "PASS" if g else "FAIL", {k: c[k] for k in ("bright_loss_fall_factor", "median_abs_ln_at_viirs_fire_cells_end", "energy_ratio_at_modis_bright_cells_end", "pass_i_fall_gt_10x", "pass_ii_median_err_lt_0.1", "pass_iii_energy_0.9_1.1")})
print("ALL PASS" if ok else "NOT ALL PASS"); sys.exit(0 if ok else 1)
