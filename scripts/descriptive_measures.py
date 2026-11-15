#!/usr/bin/env python
"""DESCRIPTIVE, POST-HOC measures from stored score files. Not pass lines, not used to pick a model.

Reads score JSONs (one per run x set x checkpoint, read-only) and writes one tidy table. It never runs the scorer and never touches a map.
Measures (formulas as defined in docs/model.md; every row lists the JSON fields it used):
  csi / hit_rate / false_alarm_share / flag_count_ratio   block flags    summary.rule[m].{block_hit_rate, block_false_alarm_ratio, flag_count_ratio}
      CSI = found / (found + missed + invented) = 1 / (1/hit + 1/(1-FAR) - 1); exact, because hit = tp/(tp+fn), FAR = fp/(tp+fp), pooled over pairs.
  amount_error_* / amount_skill_*                  amount.{block3,pixel}.methods[m].abs_log_err_pair_mean (+ decision.vs[ref].{lo,hi,share_pairs_better})
      skill vs reference r = 1 - err_model / err_r. "even spread" = uniform, "excess-only" = the stronger simple method.
  auc_block, hottest_cell_share                    summary.auc[m].mean ; summary.conc[m].top1_share   (the raw scores the fire-share needs)
  fire_share_<measure>                             (score - score_fire_off) / (score - no_skill). no_skill: auc 0.5, csi 0, hottest-cell share = even spread, amount skill 0.
  calibration_*                                    amount.{block3,pixel}.calibration[m] (8 equal-count bins) -> calibration_bins.csv; one summary number per curve.
Cannot be computed from stored summaries (no per-block / per-cell values are kept): fractions skill score, precision-recall, right-hottest-cell rate,
size-stratified detection. They are not approximated here.
Deterministic: no randomness, no clock; same inputs -> byte-identical outputs.
  python scripts/descriptive_measures.py --scores <dir1> [<dir2> ...] --out <outdir>
where each <dir> holds <run>/score_<set>.json (e.g. .../big-scores/select and .../big-scores/old)."""
import argparse, csv, glob, hashlib, json, math, os, re, sys

SIMPLE = ("uniform", "excess_only", "copy_planck", "copy_raw")
REF_COL = {"uniform": "ref_even_spread", "excess_only": "ref_excess_only", "copy_planck": "ref_copy_planck", "copy_raw": "ref_copy_raw", "model_off": "ref_fire_off"}
COLS = ["run", "family", "seed", "checkpoint", "set", "subject", "level", "measure", "value", "ref_even_spread", "ref_excess_only", "ref_copy_planck", "ref_copy_raw", "ref_fire_off",
        "lo", "hi", "share_pairs_better", "n_items", "stamped", "fields_used", "note"]
CAL_COLS = ["run", "family", "seed", "checkpoint", "set", "level", "subject", "bin", "n", "mean_pred", "mean_meas", "lo", "hi"]
LEVELS = {"block3": "3km_block", "pixel": "modis_pixel"}
RUN_RE = re.compile(r"^(?P<family>.+?)-s(?P<seed>\d+)(?P<final>-final)?$")


def num(x):
    """float or None (missing, non-numeric, NaN, inf all become None)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def dig(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def parse_run(run):
    m = RUN_RE.match(run)
    if not m:
        return run, None, "baseline" if run == "baselines" else "unknown"
    return m["family"], int(m["seed"]), "final" if m["final"] else "best_validation"


def csi(hit, far):
    """Critical success index from the pooled hit rate and false-alarm share. None when undefined (no real fire blocks). 0 when nothing is found."""
    hit, far = num(hit), num(far)
    if hit is None:
        return None, "no real fire blocks (hit rate undefined)"
    if hit == 0.0:
        return 0.0, ""
    if far is None:
        return None, "false-alarm share missing although hits exist"
    if far >= 1.0:
        return 0.0, ""
    ok = 1.0 - far
    return hit * ok / (hit + ok - hit * ok), ""


def skill(err, err_ref):
    """1 - err/err_ref. None when err_ref is missing or 0 (a perfect reference has no room to be beaten)."""
    err, err_ref = num(err), num(err_ref)
    if err is None or err_ref is None or err_ref <= 0.0:
        return None
    return 1.0 - err / err_ref


def fire_share(score, off, no_skill):
    """(score - off) / (score - no_skill). None if any input is missing or the model has no skill over no_skill (denominator <= 0)."""
    score, off, no_skill = num(score), num(off), num(no_skill)
    if score is None or off is None or no_skill is None:
        return None, "missing input"
    den = score - no_skill
    if den <= 0.0:
        return None, "model not above the no-skill value (denominator <= 0)"
    return (score - off) / den, ""


def cal_summary(bins):
    """Weighted mean |ln(mean_pred / mean_meas)| over the bins (0 = on the perfect line). None if a bin has a non-positive mean."""
    tot = sw = 0.0
    for b in bins or []:
        p, m, n = num(b.get("mean_pred")), num(b.get("mean_meas")), num(b.get("n"))
        if p is None or m is None or n is None:
            return None, "bin without mean_pred / mean_meas / n"
        if p <= 0.0 or m <= 0.0:
            return None, "bin with non-positive mean (log ratio undefined)"
        tot += n * abs(math.log(p / m)); sw += n
    return (tot / sw, "") if sw > 0 else (None, "no bins")


class Run:
    def __init__(self, path, j):
        self.path, self.j = path, j
        self.run = j.get("run") or os.path.basename(os.path.dirname(path))
        self.family, self.seed, self.ckpt = parse_run(self.run)
        self.set = j.get("set")
        self.stamped = bool(j.get("guard") and j.get("code_sha256"))

    def rule_key(self, m):
        """The score files store the excess-only simple method's flag counts under 'oracle_uniform' (same method: identical AUC and amount numbers, checked)."""
        if m != "excess_only":
            return m, ""
        rule = dig(self.j, "summary", "rule") or {}
        if "excess_only" in rule:
            return "excess_only", ""
        a, b = num(dig(self.j, "summary", "auc", "excess_only", "mean")), num(dig(self.j, "summary", "auc", "oracle_uniform", "mean"))
        if a is not None and b is not None and a == b and "oracle_uniform" in rule:
            return "oracle_uniform", "excess_only flags read from the oracle_uniform entry (identical method: equal block AUC)"
        return None, "excess_only flags not stored and not provably equal to oracle_uniform"


def row(r, subject, level, measure, value, refs=None, lo=None, hi=None, spb=None, n=None, fields="", note=""):
    d = {c: "" for c in COLS}
    d.update(run=r.run, family=r.family, seed="" if r.seed is None else r.seed, checkpoint=r.ckpt, set=r.set, subject=subject, level=level, measure=measure,
           value=value, lo=lo, hi=hi, share_pairs_better=spb, n_items=n, stamped=r.stamped, fields_used=fields, note=note)
    for m, col in REF_COL.items():
        if refs and m in refs:
            d[col] = refs[m]
    for k in list(d):
        if d[k] is None:
            d[k] = ""
    return d


def flag_rows(r, subjects):
    out = []
    for meas, fld, fn in (("hit_rate", "block_hit_rate", None), ("false_alarm_share", "block_false_alarm_ratio", None), ("flag_count_ratio", "flag_count_ratio", None), ("csi", None, None)):
        vals, notes = {}, {}
        for m in subjects:
            k, kn = r.rule_key(m)
            rec = dig(r.j, "summary", "rule", k) if k else None
            if rec is None:
                vals[m], notes[m] = None, kn or "method not in summary.rule"
                continue
            if meas == "csi":
                vals[m], n = csi(rec.get("block_hit_rate"), rec.get("block_false_alarm_ratio")); notes[m] = "; ".join(x for x in (kn, n) if x)
            else:
                vals[m] = num(rec.get(fld)); notes[m] = kn or ("" if vals[m] is not None else f"{fld} missing or null (no flags / no fire blocks)")
        fields = "summary.rule[m].block_hit_rate + block_false_alarm_ratio (CSI = hit*(1-FAR)/(hit+(1-FAR)-hit*(1-FAR)))" if meas == "csi" else f"summary.rule[m].{fld}"
        out.append((meas, vals, notes, fields))
    return out


def rows_for_run(r):
    """All table rows of one score file. A baselines file has no 'model': its simple methods become the subjects."""
    j, out = r.j, []
    is_base = "model" not in (dig(j, "summary", "auc") or {}) and not dig(j, "amount", "block3", "methods", "model")
    subjects = list(SIMPLE) if is_base else ["model"]
    # --- 1 CSI family (block flags)
    for meas, vals, notes, fields in flag_rows(r, SIMPLE + ("model", "model_off")):
        for s in subjects:
            refs = {m: vals[m] for m in list(SIMPLE) + ["model_off"] if m in vals and vals[m] is not None and m != s}
            out.append(row(r, s, "3km_block", meas, vals.get(s), None if is_base else refs, fields=fields, note=notes.get(s, "")))
    # --- raw scores used by the fire share
    auc = {m: num(dig(j, "summary", "auc", m, "mean")) for m in SIMPLE + ("model", "model_off")}
    conc = {m: num(dig(j, "summary", "conc", m, "top1_share")) for m in ("model", "model_off", "uniform", "measured")}
    for s in subjects:
        out.append(row(r, s, "3km_block", "auc_block", auc.get(s), None if is_base else {m: auc[m] for m in list(SIMPLE) + ["model_off"] if auc.get(m) is not None},
                       n=dig(j, "summary", "auc", s, "n_pairs"), fields="summary.auc[m].mean", note="block ranking; descriptive floor only"))
        if s in ("model", "uniform"):
            refs = None if is_base else {m: conc[m] for m in ("uniform", "model_off") if conc.get(m) is not None}
            out.append(row(r, s, "modis_pixel", "hottest_cell_share", conc.get(s), refs, n=dig(j, "summary", "conc", s, "n_pixels"), fields="summary.conc[m].top1_share",
                           note="measured VIIRS = %s" % ("n/a" if conc["measured"] is None else round(conc["measured"], 4)) + ("; excess-only/copy methods not stored" if s == "model" else "")))
    # --- 2 amount error and skill
    skills = {}
    for lv, lvn in LEVELS.items():
        meth = dig(j, "amount", lv, "methods") or {}
        err = {m: num(dig(meth, m, "abs_log_err_pair_mean")) for m in SIMPLE + ("model", "model_off")}
        n_items = dig(j, "amount", lv, "n_items")
        vs = dig(j, "amount", lv, "decision", "vs") or {}
        for s in subjects:
            out.append(row(r, s, lvn, "amount_error", err.get(s), None if is_base else {m: err[m] for m in list(SIMPLE) + ["model_off"] if err.get(m) is not None}, n=n_items,
                           fields=f"amount.{lv}.methods[m].abs_log_err_pair_mean", note="mean over scene pairs of the mean |ln(pred/meas)|; lower is better"))
        for refm, tag in (("uniform", "even"), ("excess_only", "excess")):
            sk = {m: skill(err.get(m), err.get(refm)) for m in SIMPLE + ("model", "model_off")}
            skills[(lv, tag)] = sk
            for s in subjects:
                if s == refm:
                    continue
                lo = hi = spb = None; note = "1 - error / error of %s" % ("the even spread" if tag == "even" else "the excess-only even spread")
                rng = vs.get(refm)
                if s == "model" and rng and err.get(refm) and err[refm] > 0 and num(rng.get("lo")) is not None and num(rng.get("hi")) is not None:
                    lo, hi, spb = rng["lo"] / err[refm], rng["hi"] / err[refm], num(rng.get("share_pairs_better"))
                    note += "; lo/hi = stored 95% pair-clustered range of the error difference divided by the reference error (not a re-bootstrap of the ratio)"
                if sk.get(s) is None:
                    note += "; undefined (missing error or reference error 0)"
                refs = None if is_base else {m: sk[m] for m in list(SIMPLE) + ["model_off"] if sk.get(m) is not None and m != refm}
                out.append(row(r, s, lvn, f"amount_skill_vs_{tag}", sk.get(s), refs, lo, hi, spb, n_items,
                               f"amount.{lv}.methods[m].abs_log_err_pair_mean" + (f" ; amount.{lv}.decision.vs.{refm}.{{lo,hi,share_pairs_better}}" if lo is not None else ""), note))
    if is_base:
        return out
    # --- 3 fire share of skill (model vs its own fire-off), on every measure with a fire-off value
    csis = {m: csi(dig(j, "summary", "rule", m, "block_hit_rate"), dig(j, "summary", "rule", m, "block_false_alarm_ratio"))[0] for m in ("model", "model_off")}
    fs = [("auc_block", "3km_block", auc["model"], auc["model_off"], 0.5, "summary.auc.{model,model_off}.mean; no-skill 0.5", None),
          ("csi", "3km_block", csis["model"], csis["model_off"], 0.0, "summary.rule.{model,model_off}.{block_hit_rate,block_false_alarm_ratio}; no-skill 0", None),
          ("hottest_cell_share", "modis_pixel", conc["model"], conc["model_off"], conc["uniform"], "summary.conc.{model,model_off,uniform}.top1_share; no-skill = even spread", None)]
    for lv, lvn in LEVELS.items():
        sk = skills.get((lv, "even"), {})
        fs.append(("amount_skill_vs_even", lvn, sk.get("model"), sk.get("model_off"), 0.0, f"amount.{lv}.methods.{{model,model_off,uniform}}.abs_log_err_pair_mean; no-skill 0", None))
    for meas, lvn, sc, off, ns, fields, _ in fs:
        v, n = fire_share(sc, off, ns)
        out.append(row(r, "model", lvn, f"fire_share_{meas}", v, {"model_off": off}, fields=fields, note=n or ("share of the model's skill lost when the fire output is switched off; >1 means the fire-off model is worse than no skill at all (here: worse than the even spread)" if v is not None and v > 1 else "share of the model's skill lost when the fire output is switched off; <0 would mean fire-off scores better")))
    # --- 4 calibration summary
    for lv, lvn in LEVELS.items():
        cal = dig(j, "amount", lv, "calibration") or {}
        vals = {m: cal_summary(cal.get(m)) for m in ("model", "uniform", "copy_planck")}
        if "model" not in cal:
            out.append(row(r, "model", lvn, "calibration_abs_log_ratio", None, fields=f"amount.{lv}.calibration.model", note="calibration curve not stored in this file"))
            continue
        out.append(row(r, "model", lvn, "calibration_abs_log_ratio", vals["model"][0], {m: vals[m][0] for m in ("uniform", "copy_planck") if vals[m][0] is not None and m in REF_COL},
                       fields=f"amount.{lv}.calibration[m][].{{n,mean_pred,mean_meas}}", note=(vals["model"][1] or "n-weighted mean |ln(mean_pred/mean_meas)| over 8 bins; 0 = perfect. excess-only and fire-off curves are not stored")))
    return out


def cal_rows(r):
    out = []
    for lv, lvn in LEVELS.items():
        for s, bins in (dig(r.j, "amount", lv, "calibration") or {}).items():
            for b in bins or []:
                out.append({"run": r.run, "family": r.family, "seed": "" if r.seed is None else r.seed, "checkpoint": r.ckpt, "set": r.set, "level": lvn, "subject": s,
                            **{k: ("" if b.get(k) is None else b.get(k)) for k in ("bin", "n", "mean_pred", "mean_meas", "lo", "hi")}})
    return out


def load(dirs):
    runs, hashes = [], {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "score_*.json"))):
            raw = open(p, "rb").read()
            try:
                j = json.loads(raw)
            except ValueError as e:
                raise SystemExit(f"{p}: not valid JSON ({e})")
            if not isinstance(j, dict) or "summary" not in j or "amount" not in j:
                raise SystemExit(f"{p}: not a score file (no summary/amount)")
            if j.get("set") == "val2all" or "xdata" in os.path.normpath(p).split(os.sep):
                raise SystemExit(f"{p}: xdata/val2all score files are INVALID for this analysis")
            r = Run(p, j); runs.append(r); hashes[os.path.relpath(p, d) + "@" + os.path.basename(d.rstrip("/"))] = hashlib.sha256(raw).hexdigest()
    if not runs:
        raise SystemExit("no score files found")
    return runs, hashes


def qa_baselines(runs, rows):
    """Do the simple methods give the same numbers in every run's file as in the baselines file of the same set? (largest absolute difference per measure)"""
    base = {(x["set"], x["subject"], x["level"], x["measure"]): x["value"] for x in rows if x["run"] == "baselines" and x["value"] != ""}
    worst = {}
    for x in rows:
        if x["run"] == "baselines" or x["subject"] != "model":
            continue
        for m, col in REF_COL.items():
            if m == "model_off" or x[col] == "":
                continue
            b = base.get((x["set"], m, x["level"], x["measure"]))
            if b is not None and x["measure"] in ("csi", "hit_rate", "false_alarm_share", "flag_count_ratio", "amount_error", "auc_block"):
                k = f'{x["set"]}|{x["measure"]}'; worst[k] = max(worst.get(k, 0.0), abs(x[col] - b))
    return {k: worst[k] for k in sorted(worst)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs, hashes = load(a.scores)
    runs.sort(key=lambda r: (str(r.set), r.family, -1 if r.seed is None else r.seed, r.ckpt))
    rows, cal = [], []
    for r in runs:
        rows += rows_for_run(r); cal += cal_rows(r)
    os.makedirs(a.out, exist_ok=True)
    for name, cols, data in (("descriptive_measures.csv", COLS, rows), ("calibration_bins.csv", CAL_COLS, cal)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, cols, lineterminator="\n"); w.writeheader(); w.writerows(data)
    qa = qa_baselines(runs, rows)
    with open(os.path.join(a.out, "descriptive_measures.json"), "w") as f:
        json.dump({"status": "DESCRIPTIVE, POST-HOC: not pass lines, not used to pick the model", "columns": COLS, "rows": rows, "input_sha256": hashes, "qa_max_abs_diff_simple_methods_vs_baselines_file": qa,
                   "not_computable_from_stored_files": ["fractions_skill_score", "precision_recall", "right_hottest_cell_rate", "size_stratified_detection"]}, f, indent=1, sort_keys=True)
    print(f"{len(runs)} score files, {len(rows)} rows, {len(cal)} calibration bins -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
