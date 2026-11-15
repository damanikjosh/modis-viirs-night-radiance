#!/usr/bin/env python
"""DESCRIPTIVE, POST-HOC scorer: FSS (375 m ... 10 km), precision-recall / AP / CSI (blocks and cells), right-hottest-cell rate, amount skill at 3 km vs even spread.
NOT a pass line, NOT used to pick a model, NOT the closed test. SELECT (nf2) or the old 9-pair validation pack only: the set choice below has no REPORT option; tiles go through the same guard as scripts/score_run.py.
  python scripts/descriptive_score.py --data <val2 pack> --set select --runs-dir R --names a b --ckpt best_val.pt --const-json J --out O --git-sha <sha>
  python scripts/descriptive_score.py --data <old val pack> --set oldval --allow-unnamed-reason '...' ...
  python scripts/descriptive_score.py ... --baselines-from <run>      # simple methods only, written as run 'baselines' (the checkpoint only supplies the config)
Tiles are streamed in chunks (--chunk, default 128; a whole SELECT stacked in RAM would be ~14 GB). Output: <out>/<run>/metrics2_<set>.json. Run from the code dir with PYTHONPATH=src (configs/*.json are relative)."""
import argparse, json, os, sys, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--runs-dir", required=True); p.add_argument("--out", required=True)
p.add_argument("--set", required=True, choices=["select", "oldval"])
p.add_argument("--names", nargs="*", default=[]); p.add_argument("--ckpt", default="best_val.pt"); p.add_argument("--baselines-from"); p.add_argument("--const-json", default=None)
p.add_argument("--nf2-split", default="configs/nf2_split.json"); p.add_argument("--split", default="configs/nf2_split_episodes.json"); p.add_argument("--retired", default="configs/report_retired.json")
p.add_argument("--allow-unnamed-reason", default=None); p.add_argument("--git-sha", default=os.environ.get("GIT_SHA", "")); p.add_argument("--chunk", type=int, default=128); p.add_argument("--bs", type=int, default=8)
p.add_argument("--threads", type=int, default=0); p.add_argument("--force", action="store_true"); p.add_argument("--max-tiles", type=int, default=0, help="smoke test only: first N tiles of the guarded selection (recorded in the output)")
a = p.parse_args()
if len(a.git_sha) < 7 or a.git_sha == "unknown":
    raise SystemExit("pass --git-sha <real commit of the code>")
if a.threads:
    torch.set_num_threads(a.threads)
from nightrad.raw import amount_eval as AE, descriptive as D, planck, prereg
from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.nf2_split import file_sha256
from nightrad.raw import protocol_eval as PE

prereg.open_split(a.data)
idx, guard = D.select_tile_indices(a.data, a.set, a.split, a.nf2_split, a.allow_unnamed_reason, a.retired)
print("GUARD", guard, flush=True)
if a.max_tiles:
    idx = idx[:a.max_tiles]
names_hash = D.names_sha256(a.data, idx)
dev = torch.device("cpu")
D.install_run_cache()
import nightrad.raw.core as core
jobs = [("baselines", a.baselines_from)] if a.baselines_from else [(n, n) for n in a.names]
const = json.load(open(a.const_json))["const"] if a.const_json else None
if const is None:
    raise SystemExit("pass --const-json (input-blind constant from scripts/train_const.py)")
here = os.path.dirname(os.path.abspath(__file__)); src = os.path.join(os.path.dirname(here), "src", "nightrad", "raw")
for out_name, ck_name in jobs:
    ck_path = os.path.join(a.runs_dir, ck_name, a.ckpt); out = os.path.join(a.out, out_name, f"metrics2_{a.set}.json")
    if not os.path.exists(ck_path) or (os.path.exists(out) and not a.force):
        print("skip", out_name); continue
    t0 = time.time(); ck = torch.load(ck_path, map_location=dev, weights_only=False); planck.use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    model = None
    if not a.baselines_from:
        model = RawHarmonizer(cfg.net).to(dev); model.load_state_dict(ck["model"])
    acc = D.Acc(); items = {"pixel": [], "block3": []}
    for c0 in range(0, len(idx), a.chunk):
        tiles = PE.load_all(PackedShards(a.data, idx=idx[c0:c0 + a.chunk]))
        core.run_scenes._cache.clear()
        D.collect(model, tiles, cfg, dev, const, acc, bs=a.bs)
        it = AE.collect_items(model, tiles, cfg, dev, const, bs=a.bs, hole_mode=None)
        for sc, d in it.items():
            if d:
                items[sc].append(d)
        del tiles, it
        print(out_name, "tiles", min(c0 + a.chunk, len(idx)), "/", len(idx), round(time.time() - t0), "s", flush=True)
    summ = D.summarize(acc)
    methods = [m for m in D.METHODS if model is not None or m not in ("model", "model_off")]
    amount = {}
    for sc, lst in items.items():
        if lst:
            cat = {k: np.concatenate([d[k] for d in lst]) for k in lst[0]}
            amount[sc] = D.amount_skill(cat, methods=methods)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {"run": out_name, "set": a.set, "n_tiles": len(idx), "max_tiles_smoke": a.max_tiles or None, "measures": summ, "amount_skill": amount, "ckpt_sha256": None if a.baselines_from else file_sha256(ck_path),
           "epoch": ck.get("epoch", -1) + 1, "git": a.git_sha, "guard": guard, "tile_names_sha256": names_hash, "const": const, "const_json_sha256": file_sha256(a.const_json), "torch_threads": torch.get_num_threads(), "seconds": round(time.time() - t0, 1),
           "definitions": {"fss_scales_cells": list(D.SCALES), "headline_scales_cells": list(D.HEADLINE_SCALES), "hist_bins": D.NB, "min_pos_blocks": D.MIN_POS_BLOCKS, "min_neg_blocks": D.MIN_NEG_BLOCKS, "min_pixels": D.MIN_PIXELS, "min_items": D.MIN_ITEMS},
           "code_sha256": {"descriptive.py": file_sha256(os.path.join(src, "descriptive.py")), "descriptive_score.py": file_sha256(__file__), "score_run.py": file_sha256(os.path.join(here, "score_run.py")),
                           "nf2_split.py": file_sha256(os.path.join(src, "nf2_split.py")), "protocol_eval.py": file_sha256(os.path.join(src, "protocol_eval.py")), "amount_eval.py": file_sha256(os.path.join(src, "amount_eval.py"))}}
    json.dump(rec, open(out, "w"), default=float, sort_keys=True)
    mm = summ["blocks"].get("model") or {}
    print(out_name, "done", round(time.time() - t0), "s", "AP block", mm.get("ap_pooled"), flush=True)
