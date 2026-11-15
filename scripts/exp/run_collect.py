#!/usr/bin/env python
"""Experiments B5-B7: per-pixel collection on SELECT for checkpoints. Usage: run_collect.py <pack> <ckpt root> <const json> <out dir> <run> [<run> ...]  (best_val.pt of each run; 1 thread)"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
pack, root, cj, out = sys.argv[1:5]; runs = sys.argv[5:]; setup_threads(1); dev = torch.device("cpu")
tiles, idx, g = load_select(pack); print("GUARD", g, "tiles", len(idx), flush=True); const = json.load(open(cj))["const"]
for r in runs:
    t0 = time.time(); m, cfg, ck = load_model(os.path.join(root, r, "best_val.pt"), dev)
    res = collect_px(m, tiles, cfg, dev, const); np.savez_compressed(os.path.join(out, f"px_{r}.npz"), **res)
    json.dump({"run": r, "n_tiles": len(idx), "n_items": int(len(res["pair"])), "seconds": time.time() - t0, "epoch": ck.get("epoch")}, open(os.path.join(out, f"px_{r}.json"), "w")); print(r, "done", time.time() - t0, flush=True)
