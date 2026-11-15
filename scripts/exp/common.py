"""Paper experiments (POST-HOC, selection set only): shared loading and the per-pixel collector.
Tiles come ONLY from a pack of SELECT tiles built by scripts/pack_nf2.py --only-side select (REPORT/TRAIN members of the shard tars are never extracted) and pass the same guard as
scripts/score_run.py (nightrad.raw.nf2_split.check_scoring_set, set 'select'). The collector is collect_items (amount_eval) plus per-pixel descriptors; the numbers of the shared quantities are
checked against the stored score files by scripts/exp/b_analysis.py (cross-check before any analysis)."""
from __future__ import annotations
import json, os, sys, time
from dataclasses import replace
import numpy as np, torch
from nightrad.raw import amount_eval as AE, planck, protocol_eval as PE
from nightrad.raw.blocks import once_per_footprint
from nightrad.raw.core import Prep, RunConfig, run_scenes
from nightrad.raw.data import PackedShards
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.nf2_split import check_scoring_set
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import FIRE, ref_cfg
from nightrad.raw.views import ModisView


def load_select(pack, split_episodes="configs/nf2_split_episodes.json", n_max=None):
    """SELECT tiles of the pack, selected exactly as score_run.py does (episode list), guard run on the tiles that are loaded. Returns (tiles dict, indices, guard record)."""
    ds = PackedShards(pack); nt = len(ds)
    f = lambda k: np.load(os.path.join(pack, k + ".npy"), mmap_mode="r").reshape(nt, -1)[:, 0]
    ep_names = {v: k for k, v in json.load(open(os.path.join(pack, "episodes.json"))).items()}
    keep = np.isin(np.array([ep_names.get(int(e), "") for e in f("eval_episode")]), json.load(open(split_episodes))["select"])
    idx = [int(i) for i in np.flatnonzero(keep)]
    if n_max:
        idx = idx[:n_max]
    guard = check_scoring_set(pack, "configs/nf2_split.json", "select", None, False, indices=idx)
    return PE.load_all(PackedShards(pack, idx=idx)), idx, guard


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); planck.use_bands(ck.get("bands", "centre")); cfg = RunConfig.from_dict(ck["cfg"])
    m = RawHarmonizer(cfg.net).to(dev); m.load_state_dict(ck["model"]); m.eval()
    return m, cfg, ck


def _top1(e, fmz, fp):
    """share of a pixel's excess on its hottest cell (as protocol_eval.collect 'conc'); returns [B,n] share and total."""
    e = (e * fmz).clamp(min=0).flatten(1); b_ = e.shape[0]
    tot = torch.zeros(b_, fp.n, device=e.device).scatter_add_(1, fp.pid, e)
    mx = torch.full((b_, fp.n), -1.0, device=e.device).scatter_reduce(1, fp.pid, e, "amax", include_self=True)
    return mx.clamp(min=0) / tot.clamp(min=1e-9), tot


@torch.no_grad()
def collect_px(model, tiles, cfg_run, dev, const, bs=8):
    """One row per MODIS fire-level, unsaturated, complete pixel (the pixel-scale items of the amount measure): pair, tile, rho4, vza, scan pixel size, MODIS excess, measured VIIRS excess energy,
    predicted energy per method, hottest-cell share of the measured and the model excess, and the number of VIIRS cells in the pixel."""
    if model is not None:
        model.eval()
    cfg = ref_cfg(cfg_run); cfg.hide = replace(cfg.hide, enabled=False)
    mv = ModisView(cfg.view); f4 = torch.tensor([mv.f4, mv.f11], device=dev)[None, :, None, None]
    n = tiles["viirs_L"].shape[0]; tp = tiles["eval_pair_id"].reshape(n, -1)[:, 0].numpy().astype(np.int64)
    treg = tiles["eval_region"].reshape(n, -1)[:, 0].numpy(); acc = {}
    for i0 in range(0, n, bs):
        idx = np.arange(i0, min(n, i0 + bs)); batch = {k: v[idx] for k, v in tiles.items()}
        inp, ev = split_batch(batch, dev); P = Prep(inp, cfg); scm = None
        if model is not None:
            sc_, _ = run_scenes(model, P, cfg, None, None); scm = sc_[("modis", "clean")]
        Lm = torch.exp(P.logL["modis"]); Bmed = torch.exp(P.med["modis"]); Bv = torch.exp(P.med["viirs"][:, 0]); tv = P.logL["viirs"][:, 0]
        pred = AE._predictions(P, scm, f4, const, Lm, Bmed, Bv, model is not None)
        ok = P.ok_foot; fm = ok & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0)
        s = once_per_footprint(fm, P.fp)
        if not s.any():
            continue
        meas_e = (torch.exp(tv) - Bv).clamp(min=0) * ok; tot = lambda e: P.fp.total(e[:, None])[:, 0]
        vza = P.fp.mean(inp["modis_vza"].float()[:, None])[:, 0].abs()
        psz = P.fp.mean(inp["modis_psize"].float())   # [B,2,H,W]: along-scan, along-track pixel size (km) of the MODIS pixel
        fmz = fm
        t1m, _ = _top1(meas_e, fmz, P.fp); t1p = _top1(scm["E"][:, 0], fmz, P.fp)[0] if scm is not None else None
        bi, yi, xi = s.nonzero(as_tuple=True); pid = P.fp.pid[bi, yi * s.shape[2] + xi]
        rec = {"pair": tp[idx[bi.cpu().numpy()]], "tile": idx[bi.cpu().numpy()], "region": treg[idx[bi.cpu().numpy()]], "rho4": P.rho4["modis"][s].cpu().numpy(), "vza": vza[s].cpu().numpy(),
               "psize_scan": psz[:, 0][s].cpu().numpy(), "psize_track": psz[:, 1][s].cpu().numpy(), "mexc": ((Lm[:, 0] - Bmed[:, 0]).clamp(min=0))[s].cpu().numpy(), "meas": tot(meas_e)[s].cpu().numpy(),
               "ncell": P.fp.count[:, 0][bi, pid].cpu().numpy(), "top1_meas": t1m[bi, pid].cpu().numpy()}
        if t1p is not None:
            rec["top1_model"] = t1p[bi, pid].cpu().numpy()
        for m, L in pred.items():
            rec[f"pred_{m}"] = tot((L.clamp(min=1e-6) - Bv).clamp(min=0) * ok)[s].cpu().numpy()
        for k, v in rec.items():
            acc.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in acc.items()}


def setup_threads(k=2):
    torch.set_num_threads(k)
