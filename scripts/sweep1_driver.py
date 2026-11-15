#!/usr/bin/env python
"""Loss-family sweep driver (paper, ablation table "Loss family"): runs the queue of configs/lossfamily_sweep.json, `--parallel` trainings at a time,
and appends ONE status line per finished run to --status. Resumable (a run with a results.json is skipped).
Every training evaluates each second epoch with nightrad.raw.sweep_eval (validation halves A/B by pair, best.pt selected on half A). Validation data only.
  python scripts/sweep1_driver.py --grid configs/lossfamily_sweep.json --train $NR_DATA_ROOT/train_night --val $NR_DATA_ROOT/val_night \\
      --out runs/lossfamily --status runs/lossfamily/status.txt --parallel 3 [--only unet__fam-gamma]"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
for k in ("grid", "train", "val", "out", "status"):
    p.add_argument("--" + k, required=True)
p.add_argument("--parallel", type=int, default=3)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--only", default=None, help="substring filter on run names")
a = p.parse_args()
g = json.load(open(a.grid))
runs = [r for r in g["runs"] if not a.only or a.only in r["name"]]
os.makedirs(a.out, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))


def line(msg):
    with open(a.status, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run(r):
    name, d = r["name"], os.path.join(a.out, r["name"])
    if os.path.exists(os.path.join(d, "results.json")):
        return
    t0 = time.time()
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "train_raw.py"), "--data", a.train, "--val-data", a.val, "--out", d, "--workers", str(a.workers)] + g["fixed"] + [str(x) for x in r["args"]]
    json.dump({"name": name, "factor": r["factor"], "level": r["level"], "seed": r["seed"], "cmd_args": cmd[2:], "grid": a.grid}, open(os.path.join(d, "config.json"), "w"), indent=1)
    with open(d + ".log", "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(os.path.join(d, "results.json")):
        line(f"FAILED {name} rc={rc}")
        return
    res = json.load(open(os.path.join(d, "results.json")))
    sw, fw = res.get("sweep"), res.get("final_sweep")
    m = lambda s, sp, k="mv3_fire|model": f"{s[sp][k]['median']:.3f}"
    line(f"DONE {name} t={time.time() - t0:.0f}s bestEp={res.get('epoch', -1) + 1} adm_A={sw['A']['admissible']['all'] if sw else '?'} "
         f"mv3 A model={m(sw, 'A')} unif={m(sw, 'A', 'mv3_fire|uniform')} B model={m(sw, 'B')} unif={m(sw, 'B', 'mv3_fire|uniform')} final A={m(fw, 'A')} B={m(fw, 'B')}")


line(f"START {len(runs)} runs parallel={a.parallel}")
with ThreadPoolExecutor(a.parallel) as ex:
    list(ex.map(run, runs))
line("SWEEP_DONE")
