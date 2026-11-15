#!/usr/bin/env python3
"""Compact the per-run post-hoc measure files (metrics2_select.json / metrics2_oldval.json, one per run; large, not stored in this repository)
into results/metrics2_summary.json (stored).
Measures per run and method: block average precision (pair mean), FSS (pair mean at 1,3,9,27 cells; 1 cell = 375 m), hottest-cell exact / within-one hit rate
(fire-level pixels, pair mean and pooled; chance values), amount skill against the even spread at 3 km (mean, quantiles, share of pairs above 0, worst pair).
Usage: python scripts/paper/extract_metrics2.py <dir with select/<run>/metrics2_select.json and old/<run>/metrics2_oldval.json> <out json>
The per-run inputs are not part of the stored results, so this step is not re-run by the tests; results/metrics2_summary.json is checked against the manifest."""
import glob, json, os
import sys
BASE, OUT = sys.argv[1:3]
out = {}
for setname, fn in (("select", "metrics2_select.json"), ("old", "metrics2_oldval.json")):
    for f in sorted(glob.glob(os.path.join(BASE, setname, "*", fn))):
        run = os.path.basename(os.path.dirname(f)); d = json.load(open(f)); r = {}
        for m, v in d["measures"]["blocks"].items():
            r.setdefault(m, {})["ap_block"] = v.get("ap_pair_mean"); r[m]["auc_block"] = v.get("auc_pair_mean")
        for m, v in d["measures"]["fss"].items():
            for sc in ("1", "3", "9", "27"):
                r.setdefault(m, {})["fss" + sc] = (v.get(sc) or {}).get("pair_mean")
        for m, v in d["measures"]["hottest"].items():
            fl = v.get("fire_level") or {}
            fg = v.get("flagged") or {}
            r.setdefault(m, {}).update(exact_flag=fg.get("exact_pair_mean"), exact=fl.get("exact_pair_mean"), exact_pooled=fl.get("exact_pooled"), within1=fl.get("within1_pair_mean"), chance_exact=fl.get("chance_exact"), chance_within1=fl.get("chance_within1"))
        sk = (d.get("amount_skill") or {}).get("block3", {}).get("skill", {}).get("uniform", {})
        for m, v in sk.items():
            r.setdefault(m, {}).update(skill_mean=v.get("mean"), skill_share=v.get("share_pairs_above_0"), skill_min=v.get("min"), skill_n=v.get("n_pairs"))
        out.setdefault(setname, {})[run] = dict(methods=r, n_tiles=d.get("n_tiles"), epoch=d.get("epoch"), stamped=bool(d.get("guard") and d.get("code_sha256")))
json.dump(out, open(OUT, "w"), indent=1)
print({k: len(v) for k, v in out.items()})
