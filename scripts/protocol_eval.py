#!/usr/bin/env python
"""Re-score sweep checkpoints with protocol_eval_v2 (configs/protocol_eval_v2.json), VALIDATION only (data-based test guard).
  python scripts/protocol_eval.py --data <packed val> --train <packed train> --runs-dir R [--names a b ...] [--ckpt last.pt] [--shuffle-seed 0] [--force]
Writes R/<run>/proto_<ckpt-stem>.json = {summary, shuffled_summary, ckpt hash, epoch, git, protocol hash}."""
import argparse, glob, hashlib, json, os
import torch
from nightrad.raw import planck, prereg, protocol_eval as PE
from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards
from nightrad.raw.net import RawHarmonizer

p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--train", required=True); p.add_argument("--runs-dir", required=True)
p.add_argument("--names", nargs="*"); p.add_argument("--ckpt", default="last.pt"); p.add_argument("--shuffle-seed", type=int, default=0); p.add_argument("--force", action="store_true")
p.add_argument("--far-only", action="store_true", help="val2: keep only tiles >= 60 km from every old v3 train tile"); p.add_argument("--excess-scale", type=float, default=1.0, help="diagnostic (a): multiply the model's predicted excess by this before scoring"); p.add_argument("--no-shuffle", action="store_true"); p.add_argument("--tag", default="", help="suffix of the output file, e.g. _val2"); p.add_argument("--hole-mode", default=None, choices=[None, "near2", "tiles"]); p.add_argument("--protocol", default="configs/protocol_eval_v2.json")
a = p.parse_args()
prereg.open_split(a.data); prereg.open_split(a.train)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proto_hash = hashlib.sha256(json.dumps(json.load(open(a.protocol)), sort_keys=True).encode()).hexdigest()[:16]
tiles = PE.load_all(PackedShards(a.data)); const = None
if a.far_only:
    keep = tiles["eval_far_km"].reshape(len(tiles["eval_far_km"]), -1)[:, 0] >= 60.0
    tiles = {k: v[keep] for k, v in tiles.items()}; print("far-only tiles", int(keep.sum()), flush=True)
names = a.names or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.runs_dir, "*__s*")) if os.path.isdir(d))
for n in names:
    ck_path = os.path.join(a.runs_dir, n, a.ckpt); out = os.path.join(a.runs_dir, n, "proto_" + os.path.splitext(a.ckpt)[0] + a.tag + (("_xs%g" % a.excess_scale) if a.excess_scale != 1.0 else "") + (("_hole_" + a.hole_mode) if a.hole_mode else "") + ".json")
    if not os.path.exists(ck_path) or (os.path.exists(out) and not a.force):
        continue
    ck = torch.load(ck_path, map_location=dev, weights_only=False); planck.use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    if const is None:
        const = PE.train_constant(PackedShards(a.train), cfg, dev)
    model = RawHarmonizer(cfg.net).to(dev); model.load_state_dict(ck["model"])
    res = PE.collect(model, tiles, cfg, dev, const, hole_mode=a.hole_mode, excess_scale=a.excess_scale); res_s = None if a.no_shuffle else PE.collect(model, tiles, cfg, dev, const, shuffle_seed=a.shuffle_seed, hole_mode=a.hole_mode)
    json.dump({"summary": PE.summarize(res), "shuffled_summary": PE.summarize(res_s) if res_s else None, "const_ln": const, "epoch": ck.get("epoch", -1) + 1, "ckpt": prereg.file_hash(ck_path), "git": prereg.git_sha(), "protocol": proto_hash},
              open(out, "w"), default=float)
    s = json.load(open(out))["summary"]
    print(n, "AUC", {m: round(v["mean"], 3) for m, v in s["auc"].items() if v["mean"] is not None}, "pairs", s["auc"]["model"]["n_pairs"], flush=True)
