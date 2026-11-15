#!/usr/bin/env python
"""Design-decision tests, data collector (CPU). One row per MODIS fire-level, unsaturated, complete pixel (the pixel-scale items of the amount measure, as scripts/exp/common.py collect_px,
which this extends) plus one row per tile. Added fields: signed tile time gap eval_dt_min (VIIRS minus MODIS, minutes), side (train|select), region, number of VIIRS fire-level cells inside the pixel,
and per-tile counts for the reverse direction (VIIRS fire footprints without MODIS fire level). Sources: TRAIN npz tiles (extract_train.py) and/or the SELECT-only pack (the SELECT-only pack).
Only ids of TRAIN/SELECT (scripts/design/guard.py) are loaded. Optional --ckpt adds the model prediction per pixel (SELECT scoring, not REPORT).
  collect.py --out OUT.npz --manifest manifest.parquet [--train-dir DIR] [--select-pack DIR] [--ckpt PATH --const JSON]"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import replace
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard
from nightrad.raw import amount_eval as AE, planck
from nightrad.raw.blocks import once_per_footprint
from nightrad.raw.core import Prep, RunConfig, run_scenes
from nightrad.raw.data import NpzTiles, PackedShards
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import FIRE, ref_cfg
from nightrad.raw.views import ModisView


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); planck.use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    m = RawHarmonizer(cfg.net).to(dev); m.load_state_dict(ck["model"]); m.eval()
    return m, cfg


def chunk_items(ds, i0, i1):
    items = [ds[i] for i in range(i0, i1)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@torch.no_grad()
def collect_chunk(tiles, names, side_of, region_of, cfg_run, model, dev, const, bs=8):
    cfg = ref_cfg(cfg_run); cfg.hide = replace(cfg.hide, enabled=False)
    mv = ModisView(cfg.view); f4 = torch.tensor([mv.f4, mv.f11], device=dev)[None, :, None, None]
    n = tiles["viirs_L"].shape[0]; f1 = lambda k: tiles[k].reshape(n, -1)[:, 0].numpy()
    tp, tdt = f1("eval_pair_id").astype(np.int64), f1("eval_dt_min").astype(np.float64)
    px, tl = {}, []
    for i0 in range(0, n, bs):
        idx = np.arange(i0, min(n, i0 + bs)); batch = {k: v[idx] for k, v in tiles.items()}
        inp, ev = split_batch(batch, dev); P = Prep(inp, cfg); scm = None
        if model is not None:
            sc_, _ = run_scenes(model, P, cfg, None, None); scm = sc_[("modis", "clean")]
        Lm = torch.exp(P.logL["modis"]); Bmed = torch.exp(P.med["modis"]); Bv = torch.exp(P.med["viirs"][:, 0]); tv = P.logL["viirs"][:, 0]
        pred = AE._predictions(P, scm, f4, const, Lm, Bmed, Bv, model is not None)
        ok = P.ok_foot; fm = ok & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0); s = once_per_footprint(fm, P.fp)
        tot = lambda e: P.fp.total(e[:, None])[:, 0]
        vfire = (ok & (P.rho4["viirs"] >= FIRE)).float(); nvf = tot(vfire)           # VIIRS fire-level cells inside each footprint
        sv = once_per_footprint(ok & (nvf > 0), P.fp)                                # footprints holding >= 1 VIIRS fire-level cell
        for b in range(len(idx)):
            tl.append((int(idx[b]), int(s[b].sum()), int((s[b] & (nvf[b] > 0)).sum()), int(sv[b].sum()), int((sv[b] & fm[b]).sum())))
        if not s.any():
            continue
        meas_e = (torch.exp(tv) - Bv).clamp(min=0) * ok
        vza = P.fp.mean(inp["modis_vza"].float()[:, None])[:, 0].abs(); psz = P.fp.mean(inp["modis_psize"].float())
        bi, yi, xi = s.nonzero(as_tuple=True); pid = P.fp.pid[bi, yi * s.shape[2] + xi]; bn = idx[bi.cpu().numpy()]
        rec = {"pair": tp[bn], "tile": bn, "dt": tdt[bn], "rho4": P.rho4["modis"][s].cpu().numpy(), "vza": vza[s].cpu().numpy(), "psize_scan": psz[:, 0][s].cpu().numpy(),
               "mexc": ((Lm[:, 0] - Bmed[:, 0]).clamp(min=0))[s].cpu().numpy(), "meas": tot(meas_e)[s].cpu().numpy(), "ncell": P.fp.count[:, 0][bi, pid].cpu().numpy(),
               "nvfire": nvf[s].cpu().numpy()}
        for m, L in pred.items():
            rec[f"pred_{m}"] = tot((L.clamp(min=1e-6) - Bv).clamp(min=0) * ok)[s].cpu().numpy()
        for k, v in rec.items():
            px.setdefault(k, []).append(v)
    px = {k: np.concatenate(v) for k, v in px.items()} if px else {}
    if px:
        nm = np.array([names[t] for t in px["tile"]]); px["tile_id"] = nm; px["side"] = np.array([side_of[x] for x in nm]); px["region"] = np.array([region_of[x] for x in nm])
    tl = np.array(tl); tn = np.array([names[t] for t in tl[:, 0]]) if len(tl) else np.array([])
    tile = {"tile_id": tn, "pair": tp[tl[:, 0]] if len(tl) else tp[:0], "dt": tdt[tl[:, 0]] if len(tl) else tdt[:0], "n_modis_fire": tl[:, 1], "n_modis_fire_with_v": tl[:, 2], "n_vfire_fp": tl[:, 3], "n_vfire_fp_modis_fire": tl[:, 4]}
    tile["side"] = np.array([side_of[x] for x in tn]); tile["region"] = np.array([region_of[x] for x in tn])
    return px, tile


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--train-dir"); ap.add_argument("--select-pack")
    ap.add_argument("--ckpt"); ap.add_argument("--const"); ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--split-json", default="configs/nf2_split.json")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(); torch.set_num_threads(a.threads); dev = torch.device("cpu")
    allowed = guard.allowed_ids(a.split_json); M = pd.read_parquet(a.manifest); region_of = dict(zip(M.tile, M.group_name)); side_of = allowed
    sources = []
    if a.train_dir:
        fs = sorted(f for f in os.listdir(a.train_dir) if f.endswith(".npz")); names = [f[:-4] for f in fs]; guard.check_ids(names, allowed)
        sources.append((NpzTiles([os.path.join(a.train_dir, f) for f in fs]), names))
    if a.select_pack:
        names = json.load(open(os.path.join(a.select_pack, "names.json"))); guard.check_ids(names, allowed)
        sel_ok = [i for i, nme in enumerate(names) if allowed[nme] == "select"]
        sources.append((PackedShards(a.select_pack, idx=sel_ok), [names[i] for i in sel_ok]))
    cfg_run, model, const = RunConfig(), None, 0.0
    if a.ckpt:
        model, cfg_run = load_model(a.ckpt, dev); const = json.load(open(a.const))["const"] if a.const else 0.0
    PX, TL = [], []; t0 = time.time(); done = 0
    for ds, names in sources:
        N = min(len(ds), a.limit) if a.limit else len(ds)
        for i0 in range(0, N, a.chunk):
            i1 = min(N, i0 + a.chunk); tiles = chunk_items(ds, i0, i1)
            px, tl = collect_chunk(tiles, names[i0:i1], side_of, region_of, cfg_run, model, dev, const)
            if px: PX.append(px)
            TL.append(tl); done += i1 - i0
            print(f"{time.strftime('%H:%M:%S')} {done} tiles {time.time()-t0:.0f}s", flush=True)
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]}
    P_, T_ = cat(PX), cat(TL)
    np.savez_compressed(a.out, **{"px_" + k: v for k, v in P_.items()}, **{"tile_" + k: v for k, v in T_.items()})
    print(json.dumps({"pixels": int(len(P_["pair"])), "tiles": int(len(T_["pair"])), "pairs": int(len(np.unique(T_["pair"]))), "sec": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
