"""Amendment 4 (configs/protocol_eval_v2_amendment4.json): the AMOUNT of fire energy per MODIS pixel and per 3x3 (3 km) block.
Items are chosen ONLY by the MODIS input (complete pixel, fire-level measured MODIS 4 um, unsaturated). Per item: sum of (VIIRS 4 um - measured
VIIRS local median)^+ over the pixel's / block's cells, measured vs predicted by each method. Diagnostics only; protocol v2 is untouched."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

EPS = 1e-3
FIRE = 0.5
SIMPLE = ("uniform", "copy_planck", "copy_raw", "excess_only", "oracle_uniform", "const")   # decision-line comparators (uniform = the no-learning physics conversion)
METHODS = ("model", "model_off") + SIMPLE
SCALES = ("pixel", "block3")


# ------------------------------------------------------------------ pure statistics (numpy)
def log_error(pred, meas, eps=EPS):
    return np.log((np.asarray(pred, float) + eps) / (np.asarray(meas, float) + eps))


def ratio_of_sums(pred, meas):
    m = float(np.sum(meas))
    return float(np.sum(pred) / m) if m > 0 else float("nan")


def _rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(u)); np.add.at(s, inv, r); return (s / cnt)[inv]


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _groups(pair):
    u, inv = np.unique(pair, return_inverse=True)
    return u, [np.flatnonzero(inv == i) for i in range(len(u))]


def calibration(pred, meas, pair, nbins=8, n_boot=4000, seed=0):
    """Equal-count bins of predicted energy (pooled; ties share a bin); mean predicted, mean measured, n, pair-clustered 95 % range of the mean measured."""
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, nbins + 1)[1:-1]))
    b = np.searchsorted(edges, pred, side="right"); nb = len(edges) + 1
    u, gi = _groups(pair); rng = np.random.default_rng(seed)
    sm = np.zeros((len(u), nb)); cn = np.zeros((len(u), nb))
    for k, ix in enumerate(gi):
        np.add.at(sm[k], b[ix], meas[ix]); np.add.at(cn[k], b[ix], 1)
    boots = rng.integers(0, len(u), (n_boot, len(u)))
    out = []
    for j in range(nb):
        n = int(cn[:, j].sum())
        if n == 0:
            continue
        bs_s, bs_n = sm[boots, j].sum(1), cn[boots, j].sum(1); ok = bs_n > 0
        mm = bs_s[ok] / bs_n[ok]
        out.append({"bin": j, "n": n, "mean_pred": float(pred[b == j].mean()), "mean_meas": float(meas[b == j].mean()),
                    "lo": float(np.percentile(mm, 2.5)), "hi": float(np.percentile(mm, 97.5))})
    return out


def _pair_means(vals, pair, min_items):
    u, gi = _groups(pair)
    return {int(p): float(np.mean(vals[ix])) for p, ix in zip(u, gi) if len(ix) >= min_items}


def paired_diff(loss_a, loss_b, pair, min_items=10, n_boot=4000, seed=0):
    """Mean over qualifying pairs of (per-pair mean loss_a - mean loss_b) [positive = b better] with a pair-clustered 95 % range and the share of pairs where b is better."""
    pa, pb = _pair_means(loss_a, pair, min_items), _pair_means(loss_b, pair, min_items)
    ks = sorted(pa); d = np.array([pa[k] - pb[k] for k in ks])
    if len(d) < 2:
        return {"diff": None, "lo": None, "hi": None, "n_pairs": len(d), "share_pos": None, "per_pair": {k: v for k, v in zip(ks, d)}}
    rng = np.random.default_rng(seed); bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_pairs": int(len(d)),
            "share_pos": float((d > 0).mean()), "per_pair": {int(k): float(v) for k, v in zip(ks, d)}}


def decide(preds: dict, meas, pair, simple=SIMPLE, min_items=10, n_boot=4000, model="model"):
    """Amendment 4 decision line at one scale on one set: the model delivers the amount if for EVERY simple method its per-pair mean |log-error| is smaller,
    the pair-clustered range of (method - model) is above zero and the model is smaller in >= 75 % of the qualifying pairs."""
    em = np.abs(log_error(preds[model], meas)); out = {"vs": {}}
    for s in simple:
        if s not in preds:
            continue
        r = paired_diff(np.abs(log_error(preds[s], meas)), em, pair, min_items, n_boot)
        out["vs"][s] = {"diff": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_pairs": r["n_pairs"], "share_pairs_better": r["share_pos"],
                        "range_above_zero": bool(r["lo"] is not None and r["lo"] > 0)}
    out["n_pairs"] = max([v["n_pairs"] for v in out["vs"].values()] + [0])
    out["testable"] = bool(out["n_pairs"] >= 3)   # fewer than 3 qualifying pairs: no range is meaningful; reported as 'not testable', never as pass or fail
    out["delivers"] = None if not out["testable"] else bool(out["vs"]) and all(v["range_above_zero"] and (v["share_pairs_better"] or 0) >= 0.75 for v in out["vs"].values())
    out["fails"] = [s for s, v in out["vs"].items() if not (v["range_above_zero"] and (v["share_pairs_better"] or 0) >= 0.75)]
    return out


def strata(rho4, vza):
    rho4, vza = np.asarray(rho4), np.asarray(vza)
    return {"all": np.ones(len(rho4), bool), "weak": rho4 < 1.0, "strong": rho4 >= 1.0, "vza<20": vza < 20, "vza20-40": (vza >= 20) & (vza < 40), "vza>=40": vza >= 40}


def method_stats(pred, meas, pair, min_items=10, n_boot=2000):
    e = log_error(pred, meas); r = (np.asarray(pred) + EPS) / (np.asarray(meas) + EPS)
    u, gi = _groups(pair); pr = [ratio_of_sums(pred[ix], meas[ix]) for ix in gi if len(ix) >= min_items]
    sp = [spearman(pred[ix], meas[ix]) for ix in gi if len(ix) >= min_items]
    mae = _pair_means(np.abs(e), pair, min_items)
    rng = np.random.default_rng(0); v = np.array(list(mae.values()))
    lo, hi = (np.percentile(rng.choice(v, (n_boot, len(v))).mean(1), [2.5, 97.5]) if len(v) >= 2 else (np.nan, np.nan))
    return {"n": int(len(pred)), "n_pairs": int(len(pr)), "ratio_of_sums": ratio_of_sums(pred, meas), "median_pair_ratio": float(np.median(pr)) if pr else None,
            "median_item_ratio": float(np.median(r)), "abs_log_err": float(np.abs(e).mean()), "abs_log_err_pair_mean": float(v.mean()) if len(v) else None,
            "abs_log_err_lo": float(lo), "abs_log_err_hi": float(hi), "log_err_sd": float(e.std()), "log_err_signed": float(e.mean()),
            "spearman_pooled": spearman(pred, meas), "spearman_pair_mean": float(np.nanmean(sp)) if sp else None}


def summarize(items: dict, meta=None, min_items=10, n_boot=4000, methods=METHODS):
    """items: {scale: {"pair","rho4","vza","meas", "pred_<m>"}} -> full amount summary (per scale: methods, calibration, decision, strata)."""
    out = {}
    for sc, it in items.items():
        if it is None or len(it["pair"]) == 0:
            out[sc] = None; continue
        ms = [m for m in methods if f"pred_{m}" in it]; P = {m: np.asarray(it[f"pred_{m}"], float) for m in ms}
        meas, pair = np.asarray(it["meas"], float), np.asarray(it["pair"])
        o = {"n_items": int(len(meas)), "n_pairs_total": int(len(np.unique(pair))), "methods": {m: method_stats(P[m], meas, pair, min_items) for m in ms},
             "calibration": {m: calibration(P[m], meas, pair, 8, n_boot) for m in ("model", "uniform", "copy_planck") if m in P},
             "decision": decide(P, meas, pair, [s for s in SIMPLE if s in P], min_items, n_boot) if "model" in P else None, "strata": {}}
        for sn, mk in strata(it["rho4"], it["vza"]).items():
            if mk.sum() == 0:
                continue
            st = {"n": int(mk.sum()), "n_pairs": int(len(np.unique(pair[mk]))), "too_few": bool(mk.sum() < 10 or len(np.unique(pair[mk])) < 3), "methods": {}}
            for m in ms:
                st["methods"][m] = {"ratio_of_sums": ratio_of_sums(P[m][mk], meas[mk]), "abs_log_err": float(np.abs(log_error(P[m][mk], meas[mk])).mean())}
            if "model" in P and not st["too_few"]:
                st["model_vs_spread"] = {k: v for k, v in paired_diff(np.abs(log_error(P["uniform"][mk], meas[mk])), np.abs(log_error(P["model"][mk], meas[mk])), pair[mk], 5, 2000).items() if k != "per_pair"} if "uniform" in P else None
            o["strata"][sn] = st
        if meta and "model" in P:   # region-balanced over episodes (amendment 1): per-pair mean |e| differences
            from nightrad.raw import protocol_eval as PE
            em = _pair_means(np.abs(log_error(P["model"], meas)), pair, min_items); bal = {}
            for s in SIMPLE:
                if s in P:
                    es = _pair_means(np.abs(log_error(P[s], meas)), pair, min_items)
                    bal[s] = PE.region_balanced_diff(es, em, {int(k): tuple(v) for k, v in meta.items()})
            o["balanced_vs"] = bal
            o["balanced_delivers"] = bool(bal) and all(v.get("diff") is not None and v["lo"] > 0 and v["share_pos_episodes"] >= 0.75 for v in bal.values())
        out[sc] = o
    return out


# ------------------------------------------------------------------ collection (torch)
def _predictions(P, scm, f4, const, Lm, Bmed, Bv, with_model=True):
    from nightrad.raw.views import modis_to_viirs_background
    d = {"uniform": (modis_to_viirs_background(Bmed) + (Lm - Bmed).clamp(min=0) / f4)[:, 0],
         "oracle_uniform": Bv + (Lm - Bmed).clamp(min=0)[:, 0] / f4[:, 0],
         "copy_planck": modis_to_viirs_background(Lm)[:, 0], "copy_raw": Lm[:, 0], "const": torch.full_like(Bv, float(np.exp(const)))}
    d["excess_only"] = d["oracle_uniform"]   # amendment 5: identical numbers (the score subtracts the VIIRS median that this adds): MODIS input only
    if with_model and scm is not None:
        d["model"] = (scm["B"] + scm["E"])[:, 0]; d["model_off"] = scm["B"][:, 0]
    return d


def collect_items(model, tiles: dict, cfg_run, dev, const: float, bs=8, amp_dtype=None, hole_mode=None):
    """Per-item arrays at pixel and 3 km scale, MODIS-selected items only (never VIIRS-selected). model=None -> baselines only."""
    from nightrad.raw.blocks import once_per_footprint
    from nightrad.raw.core import Prep, run_scenes
    from nightrad.raw.schema import split_batch
    from nightrad.raw.sweep_eval import ref_cfg
    from nightrad.raw.views import ModisView
    if model is not None:
        model.eval()
    cfg = ref_cfg(cfg_run); cfg.hide = replace(cfg.hide, enabled=False)
    mv = ModisView(cfg.view); f4 = torch.tensor([mv.f4, mv.f11], device=dev)[None, :, None, None]
    n = tiles["viirs_L"].shape[0]; tile_pair = tiles["eval_pair_id"].reshape(n, -1)[:, 0].numpy().astype(np.int64)
    acc = {sc: {} for sc in SCALES}
    with torch.no_grad():
        for i0 in range(0, n, bs):
            idx = np.arange(i0, min(n, i0 + bs)); batch = {k: v[idx] for k, v in tiles.items()}
            inp, ev = split_batch(batch, dev); P = Prep(inp, cfg)
            scm = None
            if model is not None:
                sc_, _ = run_scenes(model, P, cfg, None, amp_dtype); scm = sc_[("modis", "clean")]
            Lm = torch.exp(P.logL["modis"]); Bmed = torch.exp(P.med["modis"]); Bv = torch.exp(P.med["viirs"][:, 0]); tv = P.logL["viirs"][:, 0]
            pred = _predictions(P, scm, f4, const, Lm, Bmed, Bv, model is not None)
            ok = P.ok_foot
            if hole_mode == "near2":   # amendment 5: drop pixels within 2 cells of a VIIRS placeholder hole (same rule as protocol_eval.collect)
                import torch.nn.functional as F
                nh = F.max_pool2d((inp["viirs_code"] == 5).float()[:, None], 5, 1, 2)[:, 0] > 0
                ok = ok & ~(P.fp.total(nh.float()[:, None])[:, 0] > 0)
            complete = ok & (P.fp.sum3x3(ok.float()) > 8.5)
            fm = ok & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0)
            sel = {"pixel": once_per_footprint(fm, P.fp), "block3": once_per_footprint(fm & complete, P.fp)}
            meas_e = (torch.exp(tv) - Bv).clamp(min=0) * ok
            tot = lambda e: P.fp.total(e[:, None])[:, 0]
            blk = lambda e: P.fp.sum3x3(tot(e))
            vza = P.fp.mean(inp["modis_vza"].float()[:, None])[:, 0].abs()
            for scale, s in sel.items():
                if not s.any():
                    continue
                agg = tot if scale == "pixel" else blk
                bi = s.nonzero()[:, 0].cpu().numpy()
                rec = {"pair": tile_pair[idx[bi]], "tile": idx[bi], "rho4": P.rho4["modis"][s].cpu().numpy(), "vza": vza[s].cpu().numpy(),
                       "mexc": ((Lm[:, 0] - Bmed[:, 0]).clamp(min=0))[s].cpu().numpy(), "meas": agg(meas_e)[s].cpu().numpy()}
                for m, L in pred.items():
                    rec[f"pred_{m}"] = agg((L.clamp(min=1e-6) - Bv).clamp(min=0) * ok)[s].cpu().numpy()
                for k, v in rec.items():
                    acc[scale].setdefault(k, []).append(v)
    return {sc: ({k: np.concatenate(v) for k, v in d.items()} if d else None) for sc, d in acc.items()}

