"""THE evaluation of sweep 1: every number in the sweep (per-epoch log, final tables, report) comes from `evaluate` + the stat helpers below.

Fixed REFERENCE settings, independent of how a run was trained (so runs trained with another fire level / t_fire / min_valid_frac are scored on the same
cells and against the same baselines): fire-level = measured rho4 >= 0.5 on the target sensor, background = rho4 < 0.3, 4 um band, night, common mask with
>= 10 % valid cells, near radius 2, 800 K excess conversion, local-median background. Hiding at evaluation: the frozen default (3 top + 1 random 9x9, p 0.5),
seeded, so epochs are comparable; only the vv_hidden_* metrics use it (all M<->V scenes are the clean ones).

Metric (per item, |d ln|, lower is better):
  mv_fire   M->V, every VIIRS cell with measured rho4 >= 0.5             |ln L4_pred - ln L4_meas|
  mv_bg     M->V, every VIIRS cell with rho4 < 0.3 (per-pair sum, n)     same
  mv3_fire  M->V, 3x3 native-pixel blocks of footprints holding a fire-level VIIRS cell, one item per footprint: |ln((sum pred excess+eps)/(sum meas excess+eps))|  (PRIMARY)
  vm_fire / vm_bg / vm3_fire   the same for V->M on MODIS footprints with measured rho4 >= 0.5 (unsaturated) / < 0.3
  vv_hidden_fire / vv_hidden_bg   V->V on hidden VIIRS cells (model only)
  scene_gap  |ln(sum excess decoded from MODIS / from VIIRS)| over 3x3 blocks, fire footprints (diagnostic)
  ratio3x3   sum predicted / sum measured MODIS excess over 3x3 blocks (V->M), pooled;  mass_ratio = predicted / measured VIIRS excess on fire cells, pooled
M->V methods: model | model_off (excess switched off) | copy_raw (MODIS radiance copied) | copy_planck (Planck-converted copy) | uniform (excess spread evenly).
V->M methods: model | model_off | physical (the V scene seen through the MODIS view, no learning).
Headline statistic = MEDIAN of the items pooled over the pairs of a split (a few huge blocks would otherwise dominate); the mean is always reported next to it.
Pair-clustered bootstrap resamples PAIRS.  Items are one per footprint / cell; stats are computed on CPU numpy."""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from nightrad.raw.blocks import EPS, mv_blocks, once_per_footprint, vm_blocks
from nightrad.raw.core import Prep, RunConfig, run_scenes
from nightrad.raw.hide import HideConfig
from nightrad.raw.metrics import bg_only_pred_modis, physical_pred_modis
from nightrad.raw.schema import split_batch
from nightrad.raw.views import ModisView, ViewConfig, modis_to_viirs_background

FIRE, BG = 0.5, 0.3


def ref_cfg(cfg: RunConfig) -> RunConfig:
    c = copy.deepcopy(cfg)
    c.loss.fire_level, c.loss.near_radius = FIRE, 2
    c.view = ViewConfig(); c.min_valid_frac = 0.10; c.hide = HideConfig(); c.daynight = "night"; c.widen = 0
    return c


def perm_same_day(day: torch.Tensor, shift=1):
    perm = torch.arange(len(day))
    for d in (False, True):
        idx = (day.cpu() == d).nonzero()[:, 0]
        if len(idx) > 1:
            perm[idx] = idx.roll(shift)
    return perm


class Result:
    def __init__(self):
        self.items = defaultdict(lambda: ([], []))  # key -> (pairs, values)
        self.sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # key -> pair -> [sum, n]
        self.pairsums = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # ratio keys -> pair -> [pred, meas]

    def add_items(self, key, pairs_t, mask, val):
        idx = mask.nonzero()
        if len(idx):
            self.items[key][0].append(pairs_t[idx[:, 0].cpu().numpy()]); self.items[key][1].append(val[mask].float().cpu().numpy())

    def add_sums(self, key, pairs_t, mask, val):
        s = (val * mask).flatten(1).sum(1).double().cpu().numpy(); n = mask.flatten(1).sum(1).cpu().numpy()
        for b in range(len(s)):
            if n[b]:
                a = self.sums[key][int(pairs_t[b])]; a[0] += s[b]; a[1] += n[b]

    def add_ratio(self, key, pairs_t, pred, meas):
        p, m = pred.flatten(1).sum(1).double().cpu().numpy(), meas.flatten(1).sum(1).double().cpu().numpy()
        for b in range(len(p)):
            a = self.pairsums[key][int(pairs_t[b])]; a[0] += p[b]; a[1] += m[b]

    def freeze(self):
        """-> plain dict of numpy arrays for saving / bootstrapping."""
        d = {"items": {k: (np.concatenate(v[0]), np.concatenate(v[1])) if v[0] else (np.zeros(0, int), np.zeros(0)) for k, v in self.items.items()},
             "sums": {k: {p: tuple(x) for p, x in v.items()} for k, v in self.sums.items()},
             "ratios": {k: {p: tuple(x) for p, x in v.items()} for k, v in self.pairsums.items()}}
        return d


@torch.no_grad()
def evaluate(model, loader, cfg_run: RunConfig, dev, amp_dtype=None, shuffle=False, seed=1234) -> dict:
    """One pass over `loader` (validation only; never a test split). -> frozen Result dict. shuffle=True: the MODIS fields of every tile come from another
    tile of the same day/night class (shuffled-pair control; only model, model_off, uniform are scored)."""
    was = model.training; model.eval()
    cfg = ref_cfg(cfg_run)
    mv = ModisView(cfg.view)
    f4 = torch.tensor([mv.f4, mv.f11], device=dev)[None, :, None, None]
    R = Result()
    for bi, batch in enumerate(loader):
        inp, ev = split_batch(batch, dev)
        pairs_t = ev["eval_pair_id"].flatten().cpu().numpy().astype(np.int64)
        if shuffle:
            perm = perm_same_day(Prep(inp, cfg).day).to(dev)
            inp = dict(inp)
            for k in list(inp):
                if k.startswith("modis_"):
                    inp[k] = inp[k][perm]
        P = Prep(inp, cfg)
        gen = torch.Generator(device=dev).manual_seed(seed + bi)
        sc, hidden = run_scenes(model, P, cfg, gen, amp_dtype)
        scm, scv = sc[("modis", "clean")], sc[("viirs", "clean")]
        tv, tm = P.logL["viirs"][:, 0], P.logL["modis"][:, 0]
        Bv, Bmed = torch.exp(P.med["viirs"][:, 0]), torch.exp(P.med["modis"][:, 0])
        Lm = torch.exp(P.logL["modis"])
        # ---- M->V
        mvp = {"model": (scm["B"] + scm["E"])[:, 0], "model_off": scm["B"][:, 0],
               "uniform": (modis_to_viirs_background(torch.exp(P.med["modis"])) + (Lm - torch.exp(P.med["modis"])).clamp(min=0) / f4)[:, 0]}
        if not shuffle:
            mvp["copy_raw"] = Lm[:, 0]; mvp["copy_planck"] = modis_to_viirs_background(Lm)[:, 0]
        fire_v = P.common & (P.rho4["viirs"] >= FIRE); bg_v = P.common & (P.rho4["viirs"] < BG)
        _, sel3 = mv_blocks(P, mvp["model"], inp, FIRE); sel3 = once_per_footprint(sel3, P.fp)
        # selection-free views: cells / blocks chosen by the MODIS side (footprint fire-level by MODIS radiance), and ALL common cells of tiles with any fire
        fm_sel = P.ok_foot & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0)
        sel3m = once_per_footprint(fm_sel, P.fp)
        tile_any = (fire_v | fm_sel).flatten(1).any(1)[:, None, None] & P.common
        for m, L in mvp.items():
            l4 = L.clamp(min=1e-6); sg = torch.log(l4) - tv; e = sg.abs()
            R.add_items(f"mv_fire|{m}", pairs_t, fire_v, e); R.add_sums(f"mv_bg|{m}", pairs_t, bg_v, e)
            lr, _ = mv_blocks(P, l4, inp, FIRE)
            R.add_items(f"mv3_fire|{m}", pairs_t, sel3, lr.abs())
            R.add_items(f"mv_modsel|{m}", pairs_t, fm_sel, e); R.add_items(f"mv3_modsel|{m}", pairs_t, sel3m, lr.abs()); R.add_sums(f"mv_tile|{m}", pairs_t, tile_any, e)
            # signed ln(pred/meas): >0 = too bright. (decomposition of the raw copy: band-ratio part ln(raw/planck copy) + dilution/other part ln(planck copy/meas))
            R.add_items(f"sg_fire|{m}", pairs_t, fire_v, sg); R.add_items(f"sg_modsel|{m}", pairs_t, fm_sel, sg); R.add_sums(f"sg_tile|{m}", pairs_t, tile_any, sg)
            R.add_sums(f"sg_bg|{m}", pairs_t, bg_v, sg)
        if "copy_raw" in mvp:
            band = torch.log(mvp["copy_raw"].clamp(min=1e-6)) - torch.log(mvp["copy_planck"].clamp(min=1e-6))
            R.add_items("band_fire", pairs_t, fire_v, band); R.add_items("band_modsel", pairs_t, fm_sel, band); R.add_sums("band_bg", pairs_t, bg_v, band)
        if shuffle:
            continue
        # ---- V->M
        mview = lambda s: mv(s["B"], s["E"], P.fp)
        vmp = {"model": mview(scv)[:, 0], "model_off": mv(scv["B"], torch.zeros_like(scv["E"]), P.fp)[:, 0], "physical": physical_pred_modis(P, cfg)[:, 0]}
        fire_m = P.ok_foot & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0); bg_m = P.ok_foot & (P.rho4["modis"] < BG)
        once_m = once_per_footprint(fire_m, P.fp)
        for m, L in vmp.items():
            l4 = L.clamp(min=1e-12); e = (torch.log(l4) - tm).abs()
            R.add_items(f"vm_fire|{m}", pairs_t, once_m, e); R.add_sums(f"vm_bg|{m}", pairs_t, bg_m, e)
            lr, _ = vm_blocks(P, l4, inp, FIRE)
            R.add_items(f"vm3_fire|{m}", pairs_t, once_m, lr.abs())
        # ratio3x3 (V->M excess on 3x3 blocks, fire footprints) and the excess mass ratio on fire-level VIIRS cells
        pe3 = P.fp.sum3x3((vmp["model"].clamp(min=1e-12) - Bmed).clamp(min=0) * P.ok_foot); me3 = P.fp.sum3x3((torch.exp(tm) - Bmed).clamp(min=0) * P.ok_foot)
        R.add_ratio("ratio3x3", pairs_t, pe3 * once_m, me3 * once_m)
        meas_e = (torch.exp(tv) - Bv).clamp(min=0)
        R.add_ratio("mass_ratio", pairs_t, scv["E"][:, 0] * fire_v, meas_e * fire_v)
        sM, sV = P.fp.sum3x3(scm["E"][:, 0] * P.ok_foot), P.fp.sum3x3(scv["E"][:, 0] * P.ok_foot)
        R.add_items("scene_gap|model", pairs_t, once_m, torch.log((sM + 1e-4) / (sV + 1e-4)).abs())
        # ---- V->V hidden (model only)
        vv = torch.log((sc[("viirs", "hid")]["B"] + sc[("viirs", "hid")]["E"])[:, 0].clamp(min=1e-6)); e = (vv - tv).abs()
        hv = hidden["viirs"]
        R.add_items("vv_hidden_fire|model", pairs_t, hv & fire_v, e); R.add_sums("vv_hidden_bg|model", pairs_t, hv & bg_v, e)
    if was:
        model.train()
    return R.freeze()


# ------------------------------------------------------------------ stats (numpy, CPU)
def pair_ids(res: dict) -> list:
    s = set()
    for k, (p, v) in res["items"].items():
        s |= set(np.unique(p).tolist())
    for d in res["sums"].values():
        s |= set(d)
    return sorted(s)


def make_halves(fire_count: dict, seed: int = 0) -> dict:
    """Split pairs into halves A/B: pairs with fire-level cells and without are each shuffled (fixed seed) and dealt alternately, so both halves get fire."""
    rng = np.random.default_rng(seed)
    out = {"A": [], "B": []}
    for grp in ([p for p in sorted(fire_count) if fire_count[p] > 0], [p for p in sorted(fire_count) if fire_count[p] == 0]):
        grp = list(rng.permutation(grp))
        for i, p in enumerate(grp):
            out["AB"[i % 2]].append(int(p))
    return {k: sorted(v) for k, v in out.items()}


def sel_items(res, key, pairs=None):
    p, v = res["items"].get(key, (np.zeros(0, int), np.zeros(0)))
    if pairs is not None:
        m = np.isin(p, list(pairs)); p, v = p[m], v[m]
    return p, v


def stat_sums(res, key, pairs=None):
    """mean over pairs of per-pair mean."""
    d = res["sums"].get(key, {})
    vals = [s / n for p, (s, n) in d.items() if (pairs is None or p in pairs) and n > 0]
    return float(np.mean(vals)) if vals else float("nan")


def stat_ratio(res, key, pairs=None):
    d = res["ratios"].get(key, {})
    a = sum(x[0] for p, x in d.items() if pairs is None or p in pairs); b = sum(x[1] for p, x in d.items() if pairs is None or p in pairs)
    return float(a / b) if b > 0 else float("nan")


def summarize(res: dict, halves: dict) -> dict:
    """Compact per-split summary {split: {metric|method: {median, mean, n, pairs}} + scalars + admissibility}; splits: A, B, ALL."""
    out = {}
    splits = {"A": set(halves["A"]), "B": set(halves["B"]), "ALL": set(halves["A"]) | set(halves["B"])}
    for sn, ps in splits.items():
        d = {}
        for k in res["items"]:
            p, v = sel_items(res, k, ps)
            d[k] = {"median": float(np.median(v)) if len(v) else None, "mean": float(np.mean(v)) if len(v) else None, "n": int(len(v)), "pairs": int(len(np.unique(p)))}
        for k in res["sums"]:
            d[k] = {"mean": stat_sums(res, k, ps), "pairs": int(sum(1 for p in res["sums"][k] if p in ps))}
        for need in ("mv3_fire|model", "mv_fire|model", "vm3_fire|model"):
            if not d[need]["n"]:
                raise ValueError(f"sweep_eval: no items for {need} in split {sn} (empty measure; refusing to write NaN)")
        d["ratio3x3"] = stat_ratio(res, "ratio3x3", ps); d["mass_ratio"] = stat_ratio(res, "mass_ratio", ps)
        d["admissible"] = admissible(d)
        out[sn] = d
    return out


def admissible(d: dict) -> dict:
    """Admissibility: background errors not > 10 % worse than the baseline (best of Planck copy / uniform spread for M->V; physical for V->M),
    excess mass ratio in [0.8, 1.25], ratio3x3 >= 0.8."""
    g = lambda k: d.get(k, {}).get("mean")
    try:
        mvb = min(g("mv_bg|copy_planck"), g("mv_bg|uniform")); mv_ok = g("mv_bg|model") <= 1.10 * mvb
        vm_ok = g("vm_bg|model") <= 1.10 * g("vm_bg|physical")
        mass_ok = 0.8 <= d["mass_ratio"] <= 1.25; r3_ok = d["ratio3x3"] >= 0.8
    except TypeError:
        return {"all": False}
    return {"mv_bg": bool(mv_ok), "vm_bg": bool(vm_ok), "mass": bool(mass_ok), "ratio3x3": bool(r3_ok), "all": bool(mv_ok and vm_ok and mass_ok and r3_ok)}


def select_key(summ: dict, epoch1: int, min_epoch: int = 5):
    """Checkpoint selection on half A only: (0 if admissible else 1, A median of the model's M->V 3x3 error); epochs before `min_epoch` never selected
    over later ones (the network is ~the identity then)."""
    a = summ["A"]; v = a["mv3_fire|model"]["median"]
    v = float("inf") if v is None or v != v else v
    return [(0 if a["admissible"]["all"] else 1) + (2 if epoch1 < min_epoch else 0), v]


