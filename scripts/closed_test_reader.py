#!/usr/bin/env python
"""Closed test (held-out episode): turn the score files into the four pass/fail lines and the overall verdict, deterministically. Reads only score_val2all.json files written by scripts/score_run.py. ALL thresholds and minimum counts come from the
`reader_constants` block of the committed amendment (configs/closed_test_amendment.json; its sha256 is printed in the output and stored in the closed-test record): there are no numbers in this file to tune.
  python scripts/closed_test_reader.py <score_val2all.json> [--amendment configs/closed_test_amendment.json] [--exploratory] [--out verdict.json]
A line is PASS, FAIL or NOT_TESTABLE (below its minimum count). Overall: FAIL if ANY testable line fails (even when other lines are not testable); PASS only if all four lines are testable and pass;
NOT_DECIDABLE only when no testable line fails and at least one line is not testable. --exploratory (day pack): the same four lines are printed, no overall verdict."""
import argparse, hashlib, json, os, sys

SIMPLE_FOR_AUC = ("uniform", "excess_only")                       # comparator of line 1: the BETTER (higher mean AUC) of spread evenly and excess-only even spread
AMOUNT_SCALES = ("pixel", "block3")                              # line 2: the amount decision line must hold at both scales
NEEDED = ("min_pairs_lines_1_2", "min_pixels_line_3", "min_flagged_measured_line_4", "auc_margin_line_1", "auc_share_positive_pairs_line_1", "ratio_lo_line_2", "ratio_hi_line_2", "top1_share_line_3", "flag_ratio_lo_line_4", "flag_ratio_hi_line_4", "hit_rate_line_4")
DEFAULT_AMENDMENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "closed_test_amendment.json")


def load_constants(path):
    c = json.load(open(path)).get("reader_constants") or {}
    missing = [k for k in NEEDED if not isinstance(c.get(k), (int, float))]
    if missing:
        raise SystemExit(f"reader constants missing in {path}: {missing}")
    return {k: c[k] for k in NEEDED}


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def line1(summary, K):
    auc = summary.get("auc") or {}; vs = summary.get("vs") or {}
    means = {m: _g(auc, m, "mean") for m in SIMPLE_FOR_AUC if _g(auc, m, "mean") is not None}
    if not means:
        return {"status": "NOT_TESTABLE", "why": "no simple-method AUC in the score file"}
    comp = max(means, key=lambda m: means[m]); v = vs.get(comp) or {}; off = vs.get("model_off") or {}
    out = {"comparator": comp, "comparator_auc": means[comp], "model_auc": _g(auc, "model", "mean"), "diff": v.get("diff"), "range": [v.get("lo"), v.get("hi")], "share_pairs_positive": v.get("share_pos"), "n_pairs": v.get("n_pairs"),
           "fire_off_rule": "model minus model-with-fire-off point difference must be > 0", "diff_vs_fire_off": off.get("diff"),
           "pass_if": f"diff >= {K['auc_margin_line_1']} AND range above 0 AND positive in >= {K['auc_share_positive_pairs_line_1']:.0%} of pairs AND diff_vs_fire_off > 0"}
    if (v.get("n_pairs") or 0) < K["min_pairs_lines_1_2"] or v.get("diff") is None or off.get("diff") is None:
        out["status"] = "NOT_TESTABLE"; out["why"] = f"fewer than {K['min_pairs_lines_1_2']} qualifying pairs"; return out
    ok = v["diff"] >= K["auc_margin_line_1"] and v["lo"] > 0 and v["share_pos"] >= K["auc_share_positive_pairs_line_1"] and off["diff"] > 0
    out["status"] = "PASS" if ok else "FAIL"; return out


def line2(amount, K):
    out = {"comparator": "EVERY simple method (spread evenly, Planck copy, raw copy, excess-only/true-background + spread, constant)", "pass_if": f"3 km pooled ratio of sums in [{K['ratio_lo_line_2']}, {K['ratio_hi_line_2']}] AND the amount decision 'delivers' at pixel AND at 3 km"}
    per = {}; testable = True; ok = True
    for sc in AMOUNT_SCALES:
        b = (amount or {}).get(sc) or {}; d = b.get("decision") or {}; model = _g(b, "methods", "model") or {}
        per[sc] = {"delivers": d.get("delivers"), "testable": d.get("testable"), "n_pairs": d.get("n_pairs"), "ratio_of_sums": model.get("ratio_of_sums"), "fails": d.get("fails")}
        if not d.get("testable") or (d.get("n_pairs") or 0) < K["min_pairs_lines_1_2"]:
            testable = False
        elif not d.get("delivers"):
            ok = False
    r = per["block3"]["ratio_of_sums"]; out["per_scale"] = per
    if r is not None and not (K["ratio_lo_line_2"] <= r <= K["ratio_hi_line_2"]):
        ok = False                                               # a ratio outside the band is a failure whether or not the decision line is testable
        if not testable:
            out["status"] = "FAIL"; out["why"] = "3 km ratio outside the band"; return out
    if not testable:
        out["status"] = "NOT_TESTABLE"; out["why"] = f"fewer than {K['min_pairs_lines_1_2']} testable pairs at pixel or 3 km"; return out
    out["status"] = "PASS" if (ok and r is not None) else "FAIL"; return out


def line3(summary, K):
    c = _g(summary, "conc", "model") or {}; top1 = c.get("top1_share"); n = c.get("n_pixels") or 0
    out = {"top1_share": top1, "n_pixels": n, "pass_if": f"top-1 share >= {K['top1_share_line_3']} with >= {K['min_pixels_line_3']} MODIS-selected pixels", "reference_spread_evenly": _g(summary, "conc", "uniform", "top1_share"), "reference_measured": _g(summary, "conc", "measured", "top1_share")}
    if top1 is None or n < K["min_pixels_line_3"]:
        out["status"] = "NOT_TESTABLE"; out["why"] = f"fewer than {K['min_pixels_line_3']} pixels"; return out
    out["status"] = "PASS" if top1 >= K["top1_share_line_3"] else "FAIL"; return out


def line4(summary, K):
    r = _g(summary, "rule", "model") or {}; ratio = r.get("flag_count_ratio"); hit = r.get("block_hit_rate"); n = r.get("flagged_cells_meas") or 0
    out = {"flag_count_ratio": ratio, "block_hit_rate": hit, "false_alarm_share_reported_no_line": r.get("block_false_alarm_ratio"), "flagged_cells_measured": n,
           "pass_if": f"flag ratio in [{K['flag_ratio_lo_line_4']}, {K['flag_ratio_hi_line_4']}] AND hit rate >= {K['hit_rate_line_4']} with >= {K['min_flagged_measured_line_4']} measured flagged cells"}
    if ratio is None or hit is None or n < K["min_flagged_measured_line_4"]:
        out["status"] = "NOT_TESTABLE"; out["why"] = f"fewer than {K['min_flagged_measured_line_4']} measured flagged cells"; return out
    out["status"] = "PASS" if (K["flag_ratio_lo_line_4"] <= ratio <= K["flag_ratio_hi_line_4"] and hit >= K["hit_rate_line_4"]) else "FAIL"; return out


def read(score, K, exploratory=False, amendment_path=None):
    def block(sm, am):
        return {"line1_fire_block_score": line1(sm, K), "line2_amount": line2(am, K), "line3_hottest_cell_share": line3(sm, K), "line4_flag_count": line4(sm, K)}
    res = {"set": score.get("set"), "n_tiles": score.get("n_tiles"), "ckpt_epoch": score.get("epoch"), "run": score.get("run"), "exploratory": bool(exploratory), "constants": K,
           "all_blocks_decides": block(score.get("summary") or {}, score.get("amount") or {}), "hole_free_shown_next_to_it": block(score.get("summary_no_hole_blocks") or {}, score.get("amount_no_hole_blocks") or {})}
    st = [v["status"] for v in res["all_blocks_decides"].values()]
    res["lines"] = dict(zip(res["all_blocks_decides"], st))
    res["overall"] = "EXPLORATORY_NO_VERDICT" if exploratory else ("FAIL" if "FAIL" in st else ("PASS" if all(s == "PASS" for s in st) else "NOT_DECIDABLE"))
    res["overall_rule"] = "FAIL if any testable line fails; PASS only if all four are testable and pass; NOT_DECIDABLE only if no testable line fails and at least one is not testable"
    res["reader_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    res["amendment_sha256"] = hashlib.sha256(open(amendment_path, "rb").read()).hexdigest() if amendment_path else None
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("score"); p.add_argument("--amendment", default=DEFAULT_AMENDMENT); p.add_argument("--exploratory", action="store_true"); p.add_argument("--out", default=None); a = p.parse_args()
    r = read(json.load(open(a.score)), load_constants(a.amendment), a.exploratory, a.amendment); txt = json.dumps(r, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    print(txt)
