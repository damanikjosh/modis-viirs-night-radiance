#!/usr/bin/env python
"""Score checkpoints (or the simple methods alone) on a SET with the frozen protocol: block AUC, MODIS-selected energy, hottest-cell share, flag count
(configs/protocol_eval_v2.json, amendments 1-3) and the amount measure (amendment 4), each also without blocks next to masked VIIRS holes. Validation data only
(the data-based test guard refuses anything else); writes <out>/<run>/score_<set>[_far].json.
  python scripts/score_run.py --data <packed val2 pack with meta> --train <packed train> --set select|report|oldval|val2all [--far] --runs-dir R --names a b \
      --ckpt best_val.pt --const-json train_const_<run>.json --out O --git-sha $(git rev-parse HEAD)
  python scripts/score_run.py ... --baselines-from <one run name>    # simple methods only (model columns dropped), written as run 'baselines'
REPORT is refused unless --final (scored once per final run). --git-sha must be the commit of the code that scores. Run with PYTHONPATH=src.
--set select uses the episode lists of --split (default configs/nf2_split_episodes.json); the tile-id guard uses --nf2-split and --retired. --set val2all is the
held-out episode (closed_test.py)."""
import argparse, hashlib, json, os, time
import numpy as np
import torch
from nightrad.raw import amount_eval as AE, planck, prereg, protocol_eval as PE
from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards
from nightrad.raw.net import RawHarmonizer

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--train", required=True); p.add_argument("--runs-dir", required=True); p.add_argument("--out", required=True)
p.add_argument("--set", required=True, choices=["select", "report", "oldval", "val2all"]); p.add_argument("--far", action="store_true")
p.add_argument("--names", nargs="*", default=[]); p.add_argument("--ckpt", default="last.pt"); p.add_argument("--baselines-from"); p.add_argument("--final", action="store_true")
p.add_argument("--nf2-split", default="configs/nf2_split.json"); p.add_argument("--allow-unnamed-reason", default=None); p.add_argument("--split-override-reason", default=None); p.add_argument("--retired", default="configs/report_retired.json"); p.add_argument("--const-json", default=None, help="precomputed input-blind constant (scripts/train_const.py output, computed from the TRAIN pack): skips reading the TRAIN pack, so a machine without the training pack can score"); p.add_argument("--git-sha", default=os.environ.get("GIT_SHA", "")); p.add_argument("--split", default="configs/nf2_split_episodes.json"); p.add_argument("--force", action="store_true"); p.add_argument("--bs", type=int, default=8); p.add_argument("--dt-range", nargs=2, type=float, default=None, help="exp D: score only tiles with LO < |eval_dt_min| <= HI minutes (LO = 0 includes dt = 0); output suffix _dt<LO>-<HI>")
a = p.parse_args()
if len(a.git_sha) < 7 or a.git_sha == "unknown":
    raise SystemExit("pass --git-sha <real commit of the code> (amendment 5)")
if a.set == "report" and not a.final:
    raise SystemExit("REPORT set is scored once per final run: pass --final")
from nightrad.raw.nf2_split import check_scoring_set
prereg.open_split(a.data); prereg.open_split(a.train)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ds = PackedShards(a.data); nt = len(_ds); _f = lambda k: np.load(os.path.join(a.data, k + ".npy"), mmap_mode="r").reshape(nt, -1)[:, 0]
keep = np.ones(nt, bool)   # choose the tiles from the small eval_* arrays BEFORE loading anything (memory: a 3,700-tile val2 stacked in RAM is ~20 GB per process)
if a.set in ("select", "report"):
    ep_names = {v: k for k, v in json.load(open(os.path.join(a.data, "episodes.json"))).items()}
    keep &= np.isin(np.array([ep_names.get(int(e), "") for e in _f("eval_episode")]), json.load(open(a.split))[a.set])
if a.far:
    keep &= np.asarray(_f("eval_far_km")) >= 60.0
if a.dt_range:   # exp D: pairing-window bin of the SCORED tiles (selection by the tile time gap only; the guard below still runs on exactly these tiles)
    _dt = np.abs(np.asarray(_f("eval_dt_min"), float)); keep &= (_dt > a.dt_range[0]) & (_dt <= a.dt_range[1]) if a.dt_range[0] > 0 else (_dt <= a.dt_range[1])
if a.set == "report" and os.path.exists(os.path.join(a.data, "names.json")):   # REPORT = nf2 REPORT minus the retired ids (decision 1): an explicit, hashed exclusion on top of the unchanged split file
    from nightrad.raw.nf2_split import retired_ids
    _ret = retired_ids(a.retired); _names = json.load(open(os.path.join(a.data, "names.json")))
    keep &= np.array([n not in _ret for n in _names])
guard = check_scoring_set(a.data, a.nf2_split, a.set, a.allow_unnamed_reason, a.final, indices=[int(i) for i in np.flatnonzero(keep)], split_override_reason=a.split_override_reason, retired_json=a.retired); print("GUARD", guard, flush=True)   # on the tiles that will be scored, by id
tiles = PE.load_all(PackedShards(a.data, idx=[int(i) for i in np.flatnonzero(keep)]))
print("set", a.set, "far" if a.far else "all", "tiles", tiles["viirs_L"].shape[0], flush=True)
suffix = f"score_{a.set}{'_far' if a.far else ''}{('_dt%g-%g' % tuple(a.dt_range)) if a.dt_range else ''}.json"; const = None
# the two hole passes and the two collectors see the same model output for the same tiles: run the network once per batch (identical numbers, 4x fewer forward passes)
import nightrad.raw.core as _core
_orig_run, _cache = _core.run_scenes, {}
def _cached_run(model, P, cfg, gen=None, amp_dtype=None, hidden=None, mark=None):
    key = (id(model), P.inp["viirs_L"].shape[0], float(P.inp["viirs_L"].double().sum()))
    if key not in _cache:
        if len(_cache) > 400: _cache.clear()
        _cache[key] = _orig_run(model, P, cfg, gen, amp_dtype, hidden) if mark is None else _orig_run(model, P, cfg, gen, amp_dtype, hidden, mark)
    return _cache[key]
_core.run_scenes = _cached_run; PE.run_scenes = _cached_run


def alias(sm):   # amendment 5: 'excess-only even spread' is numerically the old 'oracle_uniform' row (MODIS input only)
    for k in ("auc", "pairs", "vs", "energy", "energy_msel", "energy_blk", "balanced", "balanced_vs"):
        if k in sm and "oracle_uniform" in sm[k]:
            sm[k]["excess_only"] = sm[k]["oracle_uniform"]
    for k in [k for k in sm.get("cells", {}) if k.endswith("|oracle_uniform")]:
        sm["cells"][k[:-len("oracle_uniform")] + "excess_only"] = sm["cells"][k]


jobs = [("baselines", a.baselines_from)] if a.baselines_from else [(n, n) for n in a.names]
for out_name, ck_name in jobs:
    ck_path = os.path.join(a.runs_dir, ck_name, a.ckpt); out = os.path.join(a.out, out_name, suffix)
    if not os.path.exists(ck_path) or (os.path.exists(out) and not a.force):
        print("skip", out_name); continue
    t0 = time.time(); ck = torch.load(ck_path, map_location=dev, weights_only=False); planck.use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    if const is None:
        const = json.load(open(a.const_json))["const"] if a.const_json else PE.train_constant(PackedShards(a.train), cfg, dev)
    model = RawHarmonizer(cfg.net).to(dev); model.load_state_dict(ck["model"]); _cache.clear()
    full = {}
    for hm in (None, "near2"):   # amendment 5: all blocks, and with blocks next to VIIRS mask holes removed
        res = PE.collect(model, tiles, cfg, dev, const, bs=a.bs, hole_mode=hm); summ_ = PE.summarize(res)
        items = AE.collect_items(model, tiles, cfg, dev, const, bs=a.bs, hole_mode=hm)
        methods = AE.SIMPLE if a.baselines_from else AE.METHODS
        am_ = AE.summarize(items, summ_.get("meta"), methods=methods)
        alias(summ_)
        full[hm or "all"] = (summ_, am_)
    summ, am = full["all"]; summ_nh, am_nh = full["near2"]
    if a.baselines_from:   # drop every model column: the checkpoint only carries the config
        for sm in (summ, summ_nh):
            for d in (sm["auc"], sm["vs"], sm["energy"], sm["energy_msel"], sm["energy_blk"], sm["rule"], sm["pairs"], sm.get("balanced", {}), sm.get("balanced_vs", {})):
                for k in ("model", "model_off", "model_c11"):
                    d.pop(k, None)
            sm["conc"].pop("model", None); sm["conc"].pop("model_off", None)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"run": out_name, "set": a.set, "far": a.far, "n_tiles": int(tiles["viirs_L"].shape[0]), "summary": summ, "amount": am, "summary_no_hole_blocks": summ_nh, "amount_no_hole_blocks": am_nh, "ckpt": None if a.baselines_from else prereg.file_hash(ck_path),
               "epoch": ck.get("epoch", -1) + 1, "git": a.git_sha, "protocol": hashlib.sha256(json.dumps(json.load(open("configs/protocol_eval_v2.json")), sort_keys=True).encode()).hexdigest()[:16],
               "amendment4": hashlib.sha256(open("configs/protocol_eval_v2_amendment4.json", "rb").read()).hexdigest()[:16], "seconds": round(time.time() - t0, 1), "guard": guard, "code_sha256": {"score_run.py": __import__("nightrad.raw.nf2_split", fromlist=["x"]).file_sha256(__file__), "nf2_split.py": __import__("nightrad.raw.nf2_split", fromlist=["x"]).file_sha256(__import__("nightrad.raw.nf2_split", fromlist=["x"]).__file__)}}, open(out, "w"), default=float)
    d = am["block3"]["decision"] if am.get("block3") and am["block3"]["decision"] else None
    print(out_name, "AUC", {m: round(v["mean"], 3) for m, v in summ["auc"].items() if v["mean"] is not None}, "| block3 ratio", {m: round(v["ratio_of_sums"], 2) for m, v in (am["block3"] or {"methods": {}})["methods"].items()}, "| delivers", d and (d["delivers"], d["n_pairs"]), round(time.time() - t0), "s", flush=True)
