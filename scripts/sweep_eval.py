#!/usr/bin/env python
"""Sweep-1 evaluation of a checkpoint on the VALIDATION night set (the single evaluation function: nightrad.raw.sweep_eval).
  python scripts/sweep_eval.py make-halves --data <packed val> --out configs/sweep1_halves.json
  python scripts/sweep_eval.py eval --ckpt best.pt --data <packed val> --halves configs/sweep1_halves.json --out ev.json [--items ev_items.pkl] [--shuffle]
Never a test split (prereg.open_split refuses it)."""
import argparse, json, pickle
import numpy as np, torch
from nightrad.raw import prereg, sweep_eval
from nightrad.raw.core import Prep, RunConfig
from nightrad.raw.data import PackedShards, make_loader
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.schema import split_batch


def main():
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["make-halves", "eval"]); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ckpt"); p.add_argument("--halves"); p.add_argument("--items"); p.add_argument("--shuffle", action="store_true"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prereg.open_split(a.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.cmd == "make-halves":
        cfg = sweep_eval.ref_cfg(RunConfig()); fire, tiles = {}, {}
        for b in make_loader(PackedShards(a.data), 32, 2):
            inp, ev = split_batch(b, dev); P = Prep(inp, cfg); pr = ev["eval_pair_id"].flatten().tolist()
            f = (P.common & (P.rho4["viirs"] >= sweep_eval.FIRE)).flatten(1).sum(1).tolist(); fm = (P.ok_foot & (P.rho4["modis"] >= sweep_eval.FIRE)).flatten(1).sum(1).tolist()
            for q, x, y in zip(pr, f, fm):
                fire[q] = fire.get(q, 0) + x; tiles[q] = tiles.get(q, 0) + 1
        h = sweep_eval.make_halves(fire, a.seed)
        h["_meta"] = {"seed": a.seed, "rule": "pairs with VIIRS fire-level cells and without are shuffled (seed) and dealt alternately to A and B", "fire_cells_per_pair": {str(k): v for k, v in fire.items()}, "tiles_per_pair": {str(k): v for k, v in tiles.items()}}
        json.dump(h, open(a.out, "w"), indent=1); print({k: v for k, v in h.items() if k in "AB"}); return
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); __import__("nightrad.raw.planck", fromlist=["x"]).use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    model = RawHarmonizer(cfg.net).to(dev); model.load_state_dict(ck["model"])
    halves = json.load(open(a.halves))
    res = sweep_eval.evaluate(model, make_loader(PackedShards(a.data), 32, 2), cfg, dev, shuffle=a.shuffle)
    out = {"summary": sweep_eval.summarize(res, halves) if not a.shuffle else None, "ckpt": prereg.file_hash(a.ckpt), "git": prereg.git_sha(), "epoch": ck.get("epoch", -1) + 1, "shuffle": a.shuffle}
    if a.items:
        pickle.dump(res, open(a.items, "wb"))
    json.dump(out, open(a.out, "w"), indent=1)
    if out["summary"]:
        for sn, d in out["summary"].items():
            print(sn, {k: (round(v["median"], 3) if v.get("median") is not None else None) for k, v in d.items() if k.startswith("mv3_fire")}, d["admissible"])


if __name__ == "__main__":
    main()
