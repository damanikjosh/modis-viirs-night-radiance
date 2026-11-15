#!/usr/bin/env python
"""Compute the input-blind constant of scripts/score_run.py ONCE from the TRAIN pack and store it with its provenance (checkpoint config, pack names hash), so scoring on a machine without the TRAIN pack gives identical numbers.
  python scripts/train_const.py --train D/train_night --ckpt runs/<run>/best_val.pt --out train_const_<run>.json
The constant depends on the run's config (view/mask settings), so one file per checkpoint family; score_run.py --const-json <file> uses it."""
import argparse, hashlib, json, os, time
import torch
from nightrad.raw import planck, protocol_eval as PE
from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards

def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True); a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); planck.use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    t0 = time.time(); const = PE.train_constant(PackedShards(a.train), cfg, dev)
    names = os.path.join(a.train, "names.json")
    json.dump({"const": const, "ckpt_sha256": hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest(), "train_pack": a.train, "train_names_sha256": hashlib.sha256(open(names, "rb").read()).hexdigest() if os.path.exists(names) else None, "seconds": round(time.time() - t0, 1), "made": time.strftime("%Y-%m-%d %H:%M KST")}, open(a.out, "w"), indent=1)
    print("const", const)


if __name__ == "__main__":  # guard: DataLoader workers re-import this file when started by spawn
    main()
