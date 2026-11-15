"""Evaluation protocol v2 (configs/protocol_eval_v2.json): block detection score on COMPLETE 3x3 blocks + selection-free per-cell checks + energy ratio + a
shuffled-pair control whose partner is always from a DIFFERENT scene pair. Validation data only. Everything numeric of the re-scoring comes from here."""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from nightrad.raw import planck
from nightrad.raw.blocks import once_per_footprint
from nightrad.raw.c11 import modis11_on_viirs_grid
from nightrad.raw.core import Prep, run_scenes
from nightrad.raw.hide import HideConfig
from nightrad.raw.rules import detect
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import FIRE, ref_cfg
from nightrad.raw.views import ModisView, modis_to_viirs_background

METHODS = ("model", "model_off", "uniform", "copy_planck", "copy_raw", "oracle_uniform", "const")
CELL_SETS = ("tile", "modsel", "viirs_bright")


def load_all(ds) -> dict:
    """Whole packed split in memory as stacked tensors (322 night tiles ~ 0.3 GB)."""
    items = [ds[i] for i in range(len(ds))]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def partner_index(pairs: np.ndarray, seed: int = 0) -> np.ndarray:
    """For every tile a partner tile from a DIFFERENT scene pair (random, seeded)."""
    rng = np.random.default_rng(seed); out = np.zeros(len(pairs), int)
    for i, p in enumerate(pairs):
        cand = np.flatnonzero(pairs != p); out[i] = rng.choice(cand)
    return out


def train_constant(ds, cfg_run, dev) -> float:
    """Input-blind constant: median ln L4 of VIIRS fire-level cells of the TRAIN set."""
    cfg = ref_cfg(cfg_run); vals = []
    from nightrad.raw.data import make_loader
    for b in make_loader(ds, 32, 0):
        inp, _ = split_batch(b, dev); P = Prep(inp, cfg)
        sel = P.common & (P.rho4["viirs"] >= FIRE)
        if sel.any():
            vals.append(P.logL["viirs"][:, 0][sel].cpu())
    return float(torch.cat(vals).median())


@torch.no_grad()
def collect(model, tiles: dict, cfg_run, dev, const: float, amp_dtype=None, shuffle_seed=None, bs=8, hole_mode=None, excess_scale=1.0) -> dict:
    """hole_mode (review H2): None | 'near2' = drop blocks/cells within 2 cells of a VIIRS placeholder (367 K) hole | 'tiles' = drop every tile that contains such a hole."""
    was = model.training; model.eval()
    cfg = ref_cfg(cfg_run); cfg.hide = replace(cfg.hide, enabled=False)
    mv = ModisView(cfg.view); f4 = torch.tensor([mv.f4, mv.f11], device=dev)[None, :, None, None]
    n = tiles["viirs_L"].shape[0]; pairs_all = tiles["eval_pair_id"].reshape(n, -1)[:, 0].numpy().astype(np.int64)
    part = partner_index(pairs_all, shuffle_seed) if shuffle_seed is not None else None
    meta = {}
    if "eval_episode" in tiles:
        ep_all, rg_all = tiles["eval_episode"].reshape(n, -1)[:, 0].numpy(), tiles["eval_region"].reshape(n, -1)[:, 0].numpy()
        for pp, e_, r_ in zip(pairs_all, ep_all, rg_all):
            meta[int(pp)] = (int(e_), int(r_))
    meths = METHODS if shuffle_seed is None else ("model", "model_off", "uniform", "const")
    blocks = {m: defaultdict(lambda: ([], [])) for m in meths}   # pair -> (scores, labels)
    cells = {(s, m): defaultdict(lambda: [0.0, 0.0, 0.0]) for s in CELL_SETS for m in meths}   # pair -> [sum |ln|, sum signed, n]
    energy = {m: defaultdict(lambda: [0.0, 0.0]) for m in meths}
    energy_msel = {m: defaultdict(lambda: [0.0, 0.0]) for m in meths}  # amendment 2 (diagnostic): same but blocks SELECTED BY THE INPUT (MODIS fire-level pixels): unbiased for a mean-consistent predictor
    two = cfg_run.net.n_out == 2  # a 4 um-only model has no predicted 11 um: its rule uses the MODIS 11 um converted to the VIIRS band (c11); a two-band model gets the rule both ways ('model' = its own 11 um, 'model_c11' = converted)
    rule = {m: defaultdict(lambda: [0.0] * 6) for m in list(meths) + (["model_c11"] if (two and shuffle_seed is None) else [])}  # amendment 3 (success metric, after the model only): rule on the PREDICTED VIIRS-like view vs rule on MEASURED VIIRS: [flagged cells pred, flagged cells meas, block TP, FP, FN, TN]
    conc = {m: [0.0, 0.0, 0.0] for m in list(meths) + ["measured"]}  # within-footprint concentration (diagnostic): share of the pixel's excess on its top-1 / top-3 cells, MODIS fire-level pixels
    energy_blk = {m: defaultdict(lambda: [0.0, 0.0]) for m in meths}  # amendment 2 (diagnostic only): pred / measured excess over the COMPLETE fire blocks (the conserved quantity), one item per MODIS pixel
    for i0 in range(0, n, bs):
        idx = np.arange(i0, min(n, i0 + bs))
        batch = {k: v[idx] for k, v in tiles.items()}
        if part is not None:
            for k in batch:
                if k.startswith("modis_"):
                    batch[k] = tiles[k][part[idx]]
        inp, ev = split_batch(batch, dev); pt = pairs_all[idx]
        P = Prep(inp, cfg)
        sc, _ = run_scenes(model, P, cfg, None, amp_dtype); scm = sc[("modis", "clean")]
        if excess_scale != 1.0:
            scm = {**scm, "E": scm["E"] * excess_scale}
        Lm = torch.exp(P.logL["modis"]); Bmed = torch.exp(P.med["modis"]); Bv = torch.exp(P.med["viirs"][:, 0]); tv = P.logL["viirs"][:, 0]
        pred = {"model": (scm["B"] + scm["E"])[:, 0], "model_off": scm["B"][:, 0],
                "uniform": (modis_to_viirs_background(Bmed) + (Lm - Bmed).clamp(min=0) / f4)[:, 0],
                "oracle_uniform": Bv + (Lm - Bmed).clamp(min=0)[:, 0] / f4[:, 0],
                "copy_planck": modis_to_viirs_background(Lm)[:, 0], "copy_raw": Lm[:, 0], "const": torch.full_like(Bv, float(np.exp(const)))}
        f11 = f4[:, 1]
        c11 = modis11_on_viirs_grid(inp)[0]
        pred11 = {"model": (scm["B"] + scm["E"])[:, 1] if two else c11, "model_off": scm["B"][:, 1] if two else c11, "uniform": (modis_to_viirs_background(Bmed)[:, 1] + (Lm - Bmed)[:, 1].clamp(min=0) / f11)[:, 0] if False else (modis_to_viirs_background(Bmed)[:, 1] + (Lm - Bmed).clamp(min=0)[:, 1] / f4[:, 1, 0, 0]),
                  "oracle_uniform": torch.exp(P.med["viirs"][:, 1]) + (Lm - Bmed).clamp(min=0)[:, 1] / f4[:, 1, 0, 0], "copy_planck": modis_to_viirs_background(Lm)[:, 1], "copy_raw": Lm[:, 1], "const": torch.full_like(Bv, float(np.exp(const)))}
        ok = P.ok_foot
        hole = (inp["viirs_code"] == 5)  # product-saturated VIIRS cells removed through the common mask (night: no glint)
        if hole_mode == "near2":
            import torch.nn.functional as F
            nh = F.max_pool2d(hole.float()[:, None], 5, 1, 2)[:, 0] > 0
            ok = ok & ~(P.fp.total(nh.float()[:, None])[:, 0] > 0)  # whole MODIS pixel dropped when any of its cells is near a hole
        elif hole_mode == "tiles":
            ok = ok & ~hole.flatten(1).any(1)[:, None, None]
        complete = ok & (P.fp.sum3x3(ok.float()) > 8.5)
        det = detect(torch.exp(P.logL["viirs"][:, 0]), torch.exp(P.logL["viirs"][:, 1]), P.common, torch.zeros(len(idx), dtype=torch.bool, device=dev), planck.LAMBDA["viirs_4"], planck.LAMBDA["viirs_11"])
        blk = lambda e: P.fp.sum3x3(P.fp.total(e[:, None])[:, 0])
        label = blk(det.float() * ok) > 0
        once = once_per_footprint(complete, P.fp)
        keep = ok if hole_mode else P.common
        if True:
            fmz = ok & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0)
            def shares(e):
                e = (e * fmz).clamp(min=0).flatten(1); b_, hw_ = e.shape; pid = P.fp.pid
                tot = torch.zeros(b_, P.fp.n, device=e.device).scatter_add_(1, pid, e)
                work = e.clone(); acc = []
                for r_ in range(3):
                    mx = torch.full((b_, P.fp.n), -1.0, device=e.device).scatter_reduce(1, pid, work, "amax", include_self=True)
                    hit = work >= torch.gather(mx, 1, pid) - 1e-12
                    first = hit & (torch.cumsum(hit.int(), 1) == 1) if False else hit
                    acc.append(mx.clamp(min=0)); work = torch.where(hit, torch.full_like(work, -1.0), work)
                valid = (tot > 1e-9) & (P.fp.count[:, 0] >= 3) & torch.cat([fmz.flatten(1).new_zeros(b_, 1), torch.zeros(b_, 0, dtype=torch.bool, device=e.device)], 1).bool().logical_not()
                return tot, acc[0], acc[0] + acc[1] + acc[2], valid
            for m_, e_ in (("measured", (torch.exp(tv) - Bv).clamp(min=0)), ("model", (scm["E"][:, 0])), ("model_off", scm["B"][:, 0] * 0.0 + (scm["B"][:, 0] - Bv).clamp(min=0)), ("uniform", (pred["uniform"] - Bv).clamp(min=0))):
                if m_ not in conc:
                    continue
                tot, t1, t3, valid = shares(e_)
                has = torch.zeros(len(idx), P.fp.n, dtype=torch.bool, device=dev).scatter_(1, P.fp.pid, fmz.flatten(1)) & valid
                if has.any():
                    conc[m_][0] += float((t1 / tot.clamp(min=1e-9))[has].sum()); conc[m_][1] += float((t3 / tot.clamp(min=1e-9)).clamp(max=1)[has].sum()); conc[m_][2] += float(has.sum())
        fire_v = P.common & keep & (P.rho4["viirs"] >= FIRE)
        fm_sel = ok & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0)
        tile_any = (fire_v | fm_sel).flatten(1).any(1)[:, None, None] & P.common & keep
        sets = {"tile": tile_any, "modsel": fm_sel, "viirs_bright": fire_v}
        meas_e = (torch.exp(tv) - Bv).clamp(min=0)
        det_meas_cells = det
        day0 = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        for m in meths:
            L = pred[m].clamp(min=1e-6)
            if m != "const":
                dp = detect(L, pred11[m].clamp(min=1e-6), P.common, day0, planck.LAMBDA["viirs_4"], planck.LAMBDA["viirs_11"])
                lab_p = blk(dp.float() * ok) > 0
                npred, nmeas = (dp & P.common).flatten(1).sum(1).cpu().numpy(), (det_meas_cells & P.common).flatten(1).sum(1).cpu().numpy()
                tp, fp_, fn, tn = ((lab_p & label & once).flatten(1).sum(1), (lab_p & ~label & once).flatten(1).sum(1), (~lab_p & label & once).flatten(1).sum(1), (~lab_p & ~label & once).flatten(1).sum(1))
                tp, fp_, fn, tn = [x.cpu().numpy() for x in (tp, fp_, fn, tn)]
                for b in range(len(idx)):
                    r = rule[m][int(pt[b])]; r[0] += npred[b]; r[1] += nmeas[b]; r[2] += tp[b]; r[3] += fp_[b]; r[4] += fn[b]; r[5] += tn[b]
                if m == "model" and "model_c11" in rule:  # same predicted 4 um, 11 um from the MODIS measurement by physics
                    dpc = detect(L, c11.clamp(min=1e-6), P.common, day0, planck.LAMBDA["viirs_4"], planck.LAMBDA["viirs_11"])
                    lab_c = blk(dpc.float() * ok) > 0
                    npc = (dpc & P.common).flatten(1).sum(1).cpu().numpy()
                    tpc, fpc, fnc, tnc = [x.cpu().numpy() for x in ((lab_c & label & once).flatten(1).sum(1), (lab_c & ~label & once).flatten(1).sum(1), (~lab_c & label & once).flatten(1).sum(1), (~lab_c & ~label & once).flatten(1).sum(1))]
                    for b in range(len(idx)):
                        r = rule["model_c11"][int(pt[b])]; r[0] += npc[b]; r[1] += nmeas[b]; r[2] += tpc[b]; r[3] += fpc[b]; r[4] += fnc[b]; r[5] += tnc[b]
            sc_blk = blk((L - Bv).clamp(min=0) * ok)
            me_blk = blk(meas_e * ok); fb = once & label
            fm_b = once & fm_sel
            if fm_b.any():
                pa2, ma2 = (sc_blk * fm_b).flatten(1).sum(1).double().cpu().numpy(), (me_blk * fm_b).flatten(1).sum(1).double().cpu().numpy()
                for b in range(len(idx)):
                    r = energy_msel[m][int(pt[b])]; r[0] += pa2[b]; r[1] += ma2[b]
            if fb.any():
                pa_, ma_ = (sc_blk * fb).flatten(1).sum(1).double().cpu().numpy(), (me_blk * fb).flatten(1).sum(1).double().cpu().numpy()
                for b in range(len(idx)):
                    r = energy_blk[m][int(pt[b])]; r[0] += pa_[b]; r[1] += ma_[b]
            ii = once.nonzero()
            if len(ii):
                s_, l_ = sc_blk[once].cpu().numpy(), label[once].cpu().numpy()
                bi = pt[ii[:, 0].cpu().numpy()]
                for p in np.unique(bi):
                    mk = bi == p; blocks[m][int(p)][0].append(s_[mk]); blocks[m][int(p)][1].append(l_[mk])
            sg = torch.log(L) - tv
            for s, mask in sets.items():
                a_ = (sg.abs() * mask).flatten(1).sum(1).double().cpu().numpy(); b_ = (sg * mask).flatten(1).sum(1).double().cpu().numpy(); n_ = mask.flatten(1).sum(1).cpu().numpy()
                for b in range(len(idx)):
                    if n_[b]:
                        r = cells[(s, m)][int(pt[b])]; r[0] += a_[b]; r[1] += b_[b]; r[2] += n_[b]
            pe = ((L - Bv).clamp(min=0) * fire_v).flatten(1).sum(1).double().cpu().numpy(); me = (meas_e * fire_v).flatten(1).sum(1).double().cpu().numpy()
            for b in range(len(idx)):
                r = energy[m][int(pt[b])]; r[0] += pe[b]; r[1] += me[b]
    if was:
        model.train()
    return {"blocks": {m: {p: (np.concatenate(v[0]), np.concatenate(v[1])) for p, v in d.items()} for m, d in blocks.items()},
            "cells": {k: {p: tuple(v) for p, v in d.items()} for k, d in cells.items()}, "energy": {m: {p: tuple(v) for p, v in d.items()} for m, d in energy.items()}, "energy_blk": {m: {p: tuple(v) for p, v in d.items()} for m, d in energy_blk.items()}, "conc": conc, "rule": {m: {p: tuple(v) for p, v in d.items()} for m, d in rule.items()}, "energy_msel": {m: {p: tuple(v) for p, v in d.items()} for m, d in energy_msel.items()}, "meta": meta}


# ---------------------------------------------------------------- statistics
def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.astype(bool); n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    u, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); avg_rank = cum - (cnt - 1) / 2.0
    r = avg_rank[inv]
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_fa(score: np.ndarray, label: np.ndarray, fa: float = 0.01) -> float:
    pos = label.astype(bool); neg = score[~pos]
    if pos.sum() == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fa)
    return float((score[pos] > thr).mean())


def per_pair(res: dict, method: str, min_pos=10, min_neg=100) -> dict:
    out = {}
    for p, (s, l) in res["blocks"][method].items():
        npos, nneg = int(l.sum()), int((~l.astype(bool)).sum())
        out[p] = {"auc": auc(s, l), "rec1": recall_at_fa(s, l), "n_pos": npos, "n_neg": nneg, "qualifies": npos >= min_pos and nneg >= min_neg}
    return out


def boot_diff(a: dict, b: dict, n_boot=4000, seed=0) -> dict:
    """a, b: pair -> auc (qualifying pairs only, common keys). Mean AUC difference a-b with pair-clustered 95 % range, share of pairs positive."""
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    if len(d) < 2:
        return {"diff": None, "lo": None, "hi": None, "n_pairs": len(d), "share_pos": None}
    rng = np.random.default_rng(seed); bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_pairs": int(len(d)), "share_pos": float((d > 0).mean())}


def summarize(res: dict) -> dict:
    out = {"auc": {}, "pairs": {}, "vs": {}, "cells": {}, "energy": {}, "meta": {str(k): v for k, v in res.get("meta", {}).items()}}
    pp = {m: per_pair(res, m) for m in res["blocks"]}
    q = {m: {p: v["auc"] for p, v in pp[m].items() if v["qualifies"]} for m in pp}
    for m in pp:
        out["pairs"][m] = {str(p): v for p, v in pp[m].items()}
        out["auc"][m] = {"mean": float(np.mean(list(q[m].values()))) if q[m] else None, "n_pairs": len(q[m]),
                         "rec1_mean": float(np.mean([v["rec1"] for v in pp[m].values() if v["qualifies"]])) if q[m] else None}
    for m in pp:
        if m != "model":
            out["vs"][m] = boot_diff(q["model"], q[m])
    nblk = {p: len(res["blocks"]["model"][p][1]) for p in res["blocks"]["model"]}
    tot = sum(nblk.values()) or 1; out["largest_pair_share"] = max(nblk.values()) / tot if nblk else None
    for (s, m), d in res["cells"].items():
        v = [(a / n, b / n) for a, b, n in d.values() if n > 0]
        out["cells"][f"{s}|{m}"] = {"abs_mean_of_pairs": float(np.mean([x[0] for x in v])) if v else None, "signed_mean_of_pairs": float(np.mean([x[1] for x in v])) if v else None, "n_pairs": len(v),
                                     "abs_pooled": float(sum(a for a, b, n in d.values()) / max(1, sum(n for a, b, n in d.values())))}
    for m, d in res["energy"].items():
        a, b = sum(x[0] for x in d.values()), sum(x[1] for x in d.values())
        out["energy"][m] = float(a / b) if b > 0 else None
    out["conc"] = {m: {"top1_share": v[0] / v[2], "top3_share": v[1] / v[2], "n_pixels": v[2]} for m, v in res.get("conc", {}).items() if v[2] > 0}
    out["rule"] = {}
    for m, d in res.get("rule", {}).items():
        t = [sum(x[i] for x in d.values()) for i in range(6)]
        out["rule"][m] = {"flag_count_ratio": float(t[0] / t[1]) if t[1] > 0 else None, "block_hit_rate": float(t[2] / (t[2] + t[4])) if (t[2] + t[4]) > 0 else None,
                          "block_false_alarm_ratio": float(t[3] / (t[2] + t[3])) if (t[2] + t[3]) > 0 else None, "flagged_cells_pred": t[0], "flagged_cells_meas": t[1]}
    out["energy_msel"] = {}
    for m, d in res.get("energy_msel", {}).items():
        a, b = sum(x[0] for x in d.values()), sum(x[1] for x in d.values())
        out["energy_msel"][m] = float(a / b) if b > 0 else None
    out["energy_blk"] = {}
    for m, d in res.get("energy_blk", {}).items():
        a, b = sum(x[0] for x in d.values()), sum(x[1] for x in d.values())
        out["energy_blk"][m] = float(a / b) if b > 0 else None
    meta = {int(k): tuple(v) for k, v in out["meta"].items()}
    if meta:
        out["balanced"] = {m: region_balanced(q[m], meta) for m in pp}
        out["balanced_vs"] = {m: region_balanced_diff(q["model"], q[m], meta) for m in pp if m != "model"}
    return out


# ---------------------------------------------------------------- amendment 1: region-balanced mean over episodes (val2 primary)
def region_balanced(auc_pairs: dict, meta: dict, min_eps: int = 2) -> dict:
    """auc_pairs: pair -> AUC (qualifying pairs only); meta: pair -> (episode, region). Per region: pairs of one episode averaged first, then the mean over episodes;
    the balanced value = plain mean over regions with >= min_eps qualifying episodes. -> {"value", "regions": {region: {"mean", "n_episodes", "episodes": {ep: auc}}}}"""
    reg = {}
    for p, a in auc_pairs.items():
        if a != a or p not in meta:
            continue
        e, r = meta[p]; reg.setdefault(r, {}).setdefault(e, []).append(a)
    out = {}
    for r, d in reg.items():
        eps = {e: float(np.mean(v)) for e, v in d.items()}
        out[r] = {"mean": float(np.mean(list(eps.values()))), "n_episodes": len(eps), "episodes": eps}
    used = [v["mean"] for v in out.values() if v["n_episodes"] >= min_eps]
    return {"value": float(np.mean(used)) if used else None, "regions": out, "n_regions_used": len(used), "n_episodes_used": int(sum(v["n_episodes"] for v in out.values() if v["n_episodes"] >= min_eps))}


def region_balanced_diff(a_pairs: dict, b_pairs: dict, meta: dict, n_boot=4000, seed=0, min_eps=2) -> dict:
    """Region-balanced mean AUC difference (a - b) with an EPISODE-clustered bootstrap within each region (regions fixed); share of qualifying episodes with a positive difference."""
    ra, rb = region_balanced(a_pairs, meta, min_eps), region_balanced(b_pairs, meta, min_eps)
    regs = [r for r in ra["regions"] if ra["regions"][r]["n_episodes"] >= min_eps and r in rb["regions"]]
    if not regs:
        return {"diff": None}
    dd = {r: np.array([ra["regions"][r]["episodes"][e] - rb["regions"][r]["episodes"][e] for e in ra["regions"][r]["episodes"] if e in rb["regions"][r]["episodes"]]) for r in regs}
    rng = np.random.default_rng(seed)
    bs = np.mean([rng.choice(dd[r], (n_boot, len(dd[r]))).mean(1) for r in regs], axis=0)
    allep = np.concatenate(list(dd.values()))
    return {"diff": float(np.mean([dd[r].mean() for r in regs])), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_regions": len(regs), "n_episodes": int(len(allep)),
            "share_pos_episodes": float((allep > 0).mean()), "per_region": {r: float(dd[r].mean()) for r in regs}}
