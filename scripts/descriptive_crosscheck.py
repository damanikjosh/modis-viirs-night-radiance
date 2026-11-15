#!/usr/bin/env python
"""where the new measures overlap with a stored score file (score_<set>.json written by scripts/score_run.py), they must reproduce its numbers. Checks per run:
 checkpoint (first 16 hex of sha256) · flagged cells (predicted, measured) · block hit rate · block false-alarm share · flag count ratio · mean block AUC over qualifying pairs · 3 km mean abs log error per pair (amount).
  python scripts/descriptive_crosscheck.py --mine <dir with <run>/metrics2_<set>.json> --stored <dir with <run>/score_<set>.json> --set select [--out report.json]
Exit code 1 if any check fails. Integer counts must be equal; floats must agree to 1e-9 (relative)."""
import argparse, glob, json, math, os, sys

p = argparse.ArgumentParser(); p.add_argument("--mine", required=True); p.add_argument("--stored", required=True); p.add_argument("--set", required=True); p.add_argument("--out"); p.add_argument("--rtol", type=float, default=0.0, help="0 = exact (same machine, same pack); >0 = allowed relative difference for counts, rates and AUC (stored files written on a GPU pod from the pod pack)")
a = p.parse_args()
MAP = {"model": "model", "model_off": "model_off", "uniform": "uniform", "copy_planck": "copy_planck", "copy_raw": "copy_raw", "excess_only": "oracle_uniform"}


def close(x, y, tol=1e-9):
    if x is None or y is None:
        return x is None and y is None
    t = max(tol, a.rtol)
    return math.isclose(x, y, rel_tol=t, abs_tol=t)


rows, bad = [], 0
for f in sorted(glob.glob(os.path.join(a.mine, "*", f"metrics2_{a.set}.json"))):
    run = os.path.basename(os.path.dirname(f)); sf = os.path.join(a.stored, run, f"score_{a.set}.json")
    if not os.path.exists(sf):
        rows.append((run, "no stored score file", None)); continue
    mine, st = json.load(open(f)), json.load(open(sf)); res = []
    chk = lambda name, ok, mv, sv: res.append({"check": name, "ok": bool(ok), "mine": mv, "stored": sv})
    chk("n_tiles", mine["n_tiles"] == st["n_tiles"], mine["n_tiles"], st["n_tiles"])
    if mine["ckpt_sha256"] is not None:
        chk("checkpoint sha16", mine["ckpt_sha256"][:16] == st["ckpt"], mine["ckpt_sha256"][:16], st["ckpt"])
    for m, sm in MAP.items():
        if m not in mine["measures"]["blocks"] or sm not in st["summary"]["rule"]:
            continue
        op, r = mine["measures"]["blocks"][m]["operating_point"], st["summary"]["rule"][sm]
        chk(f"{m} flagged cells predicted", close(op["flagged_cells"], r["flagged_cells_pred"], 0), op["flagged_cells"], r["flagged_cells_pred"])
        chk(f"{m} flagged cells measured", close(op["measured_cells"], r["flagged_cells_meas"], 0), op["measured_cells"], r["flagged_cells_meas"])
        chk(f"{m} block hit rate", close(op["recall"], r["block_hit_rate"]), op["recall"], r["block_hit_rate"])
        far = None if (op["tp"] + op["fp"]) == 0 else op["fp"] / (op["tp"] + op["fp"])
        chk(f"{m} block false-alarm share", close(far, r["block_false_alarm_ratio"]), far, r["block_false_alarm_ratio"])
        chk(f"{m} flag count ratio", close(op["flag_count_ratio"], r["flag_count_ratio"]), op["flag_count_ratio"], r["flag_count_ratio"])
    for m, sm in MAP.items():
        if m in mine["measures"]["blocks"] and sm in st["summary"]["auc"]:
            chk(f"{m} mean block AUC", close(mine["measures"]["blocks"][m]["auc_pair_mean"], st["summary"]["auc"][sm]["mean"], 1e-7), mine["measures"]["blocks"][m]["auc_pair_mean"], st["summary"]["auc"][sm]["mean"])
    for sc in ("block3", "pixel"):
        am, sa = mine["amount_skill"].get(sc), (st.get("amount") or {}).get(sc)
        if not am or not sa:
            continue
        for m in am["error_per_pair"]:
            if m in sa["methods"]:
                v = sum(am["error_per_pair"][m]) / len(am["error_per_pair"][m])
                chk(f"{sc} {m} amount error (mean of pairs)", close(v, sa["methods"][m]["abs_log_err_pair_mean"], 1e-7), v, sa["methods"][m]["abs_log_err_pair_mean"])
    nb = sum(1 for r_ in res if not r_["ok"]); bad += nb
    rows.append((run, f"{len(res)} checks, {nb} failed", res))
    print(run, f"{len(res)} checks, {nb} failed")
    for r_ in res:
        if not r_["ok"]:
            print("   FAIL", r_)
if a.out:
    json.dump([{"run": r[0], "summary": r[1], "checks": r[2]} for r in rows], open(a.out, "w"), indent=1, default=float)
sys.exit(1 if bad else 0)
