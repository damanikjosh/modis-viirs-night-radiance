#!/usr/bin/env python
"""Add the display names (as used in the paper) to the per-run JSON written by descriptive_score.py. Raw run ids stay as ids. The fields are added AFTER scoring
(flag 'labels_added_after_scoring'); no measure value is touched.
  Group-share (formerly old/legacy loss) · Gamma · Gamma-BW100 · Gamma-BW1000 · Gamma-BW100-B0.5 · Gamma-BW1000-B0.5, plus '(best-val)' / '(final)';
  references 'Even spread' and 'Oracle-background spread' (stored as excess_only / oracle_uniform: it uses the TRUE VIIRS background), 'Planck copy of MODIS', 'Raw MODIS copy', 'Input-blind constant'; control 'Fire-off'.
  python scripts/descriptive_label.py <dir with <run>/metrics2_*.json>"""
import glob, json, os, re, sys
METHODS = {"model": "the model", "model_off": "Fire-off", "uniform": "Even spread", "excess_only": "Oracle-background spread", "copy_planck": "Planck copy of MODIS", "copy_raw": "Raw MODIS copy", "const": "Input-blind constant"}
def display(run):
    if run == "baselines": return "Non-learning references"
    m = re.match(r"^(?P<fam>.+?)-s(?P<seed>\d+)(?P<fin>-final)?$", run); fam, fin = m["fam"], "(final)" if m["fin"] else "(best-val)"
    if fam == "oldloss-nf2": base = "Group-share"
    elif fam == "gamma-nf2": base = "Gamma"
    elif fam == "ref-gamma": base = "Gamma (small-list)"
    elif fam == "ref-legacy": base = "Group-share (small-list)"
    else:
        mm = re.match(r"gammabw(100|1000)(-d05)?-nf2", fam); base = f"Gamma-BW{mm.group(1)}{'-B0.5' if mm.group(2) else ''}"
    return f"{base} {fin}, seed {m['seed']}"
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "metrics2_*.json")) + glob.glob(os.path.join(sys.argv[1], "*", "metrics2_*.json"))):
    d = json.load(open(f)); d["display_name"] = display(d["run"]); d["display_names_methods"] = METHODS; d["labels_added_after_scoring"] = True
    json.dump(d, open(f, "w"), default=float, sort_keys=True)
print("labelled", sys.argv[1])
