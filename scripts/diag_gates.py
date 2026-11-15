#!/usr/bin/env python
"""Stage-1 sanity gates of the training workflow. Same argument names as train_raw.py (+ --gate, --steps, --out-json).
  --gate overfit : memorise 8 fixed training tiles, hiding OFF, loss must fall > 10x and the predicted VIIRS view must match the measured one
  --gate path    : same batch through the training path (train mode, grad on) and the evaluation path (eval mode, no grad, diag.full_eval loader path): same loss; masks and scaling identical; no hiding at eval
  --gate share   : at initialisation, share of loss and of gradient from cells under MODIS-bright pixels (by MODIS input only), old loss and gamma
  --gate seed    : two 40-step trainings with the same seed give identical losses
  python scripts/diag_gates.py --gate overfit --data D/train_night --loss gamma --steps 3000 --out-json o.json"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from train_raw import make_cfg, make_parser  # noqa: E402

from nightrad.raw import diag, losses  # noqa: E402
from nightrad.raw.core import Prep, forward_loss  # noqa: E402
from nightrad.raw.data import PackedShards  # noqa: E402
from nightrad.raw.net import RawHarmonizer  # noqa: E402
from nightrad.raw.schema import split_batch  # noqa: E402
from nightrad.raw.train import lr_at, make_param_groups, seed_all  # noqa: E402


def pick_tiles(ds, cfg, dev, n=8, scan=None):
    """8 fixed tiles: the training tiles with the most MODIS-bright cells (MODIS input only), distinct, deterministic."""
    sc = []
    for i in range(min(len(ds), scan or len(ds))):
        b = {k: v[None] for k, v in ds[i].items()}
        inp, _ = split_batch(b, dev); P = Prep(inp, cfg)
        sc.append((int(diag.bright_region(P, inp, cfg).sum()), i))
    sc.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in sc[:n]], [s for s, _ in sc[:n]]


def batch_of(ds, idx, dev):
    items = [ds[i] for i in idx]
    return {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}


@torch.no_grad()
def fidelity(model, inp, cfg, dev, amp):
    """How well the model reproduces the measured VIIRS on these tiles (hiding off): median |ln| error (all common cells, fire-bearing cells), energy ratio under bright pixels."""
    was = model.training; model.eval()
    P = Prep(inp, cfg)
    from nightrad.raw.core import run_scenes
    from nightrad.raw.views import viirs_view
    sc, _ = run_scenes(model, P, cfg, None, amp)
    s = sc[("modis", "clean")]
    Lp = viirs_view(s["B"], s["E"])[:, 0]
    tv = torch.exp(P.logL["viirs"][:, 0]); Bv = torch.exp(P.med["viirs"][:, 0])
    err = (torch.log(Lp.clamp(min=1e-6)) - torch.log(tv.clamp(min=1e-6))).abs()
    bright = diag.bright_region(P, inp, cfg)
    vf = P.common & (P.rho4["viirs"] >= 0.5)
    Ev = (tv - Bv).clamp(min=0); Ep = (Lp - Bv).clamp(min=0)
    if was:
        model.train()
    return {"med_abs_ln_all": float(err[P.common].median()), "med_abs_ln_viirs_fire": float(err[vf].median()) if vf.any() else None, "mean_abs_ln_bright": float(err[bright].mean()) if bright.any() else None,
            "energy_ratio_bright_cells": float(Ep[bright].sum() / Ev[bright].sum().clamp(min=1e-12)), "energy_ratio_viirs_fire_cells": float(Ep[vf].sum() / Ev[vf].sum().clamp(min=1e-12)) if vf.any() else None,
            "n_bright": int(bright.sum()), "n_viirs_fire": int(vf.sum())}


def amp_of(a, dev):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[a.amp] if dev.type == "cuda" else None


def gate_overfit(a, cfg, ds, dev):
    cfg.hide.enabled = False
    idx, nb = pick_tiles(ds, cfg, dev)
    inp, _ = split_batch(batch_of(ds, idx, dev), dev)
    seed_all(a.seed)
    model = RawHarmonizer(cfg.net).to(dev); amp = amp_of(a, dev)
    opt = torch.optim.AdamW(make_param_groups(model, a.wd, a.head_lr_mult), lr=a.lr, weight_decay=a.wd)
    log = []; f0 = fidelity(model, inp, cfg, dev, amp)
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.warmup, a.lr, const=a.lr_const) * g.get("lr_mult", 1.0)
        loss, logs = forward_loss(model, inp, cfg, None, amp)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step()
        if step % max(1, a.steps // 12) == 0 or step == a.steps - 1:
            fe = diag.full_eval_weighted(model, [inp], cfg, dev, amp); model.train()
            row = {"step": step, "loss": float(loss), **{k: float(v) for k, v in logs.items() if k.endswith("/loss") or k == "total"}, "loss_bright_cells": fe["terms_weighted_sum"] * fe["loss_share_bright"], "loss_other_cells": fe["terms_weighted_sum"] * (1 - fe["loss_share_bright"]), "fid": fidelity(model, inp, cfg, dev, amp)}
            log.append(row); print(json.dumps(row), flush=True)
    f1 = fidelity(model, inp, cfg, dev, amp)
    l0, l1 = log[0]["loss"], log[-1]["loss"]
    c0, c1 = log[0], log[-1]
    crit = {"bright_loss_start": c0["loss_bright_cells"], "bright_loss_end": c1["loss_bright_cells"], "bright_loss_fall_factor": c0["loss_bright_cells"] / max(c1["loss_bright_cells"], 1e-30),
            "median_abs_ln_at_viirs_fire_cells_end": f1["med_abs_ln_viirs_fire"], "energy_ratio_at_modis_bright_cells_end": f1["energy_ratio_bright_cells"]}
    crit["pass_i_fall_gt_10x"] = crit["bright_loss_fall_factor"] > 10; crit["pass_ii_median_err_lt_0.1"] = crit["median_abs_ln_at_viirs_fire_cells_end"] < 0.1; crit["pass_iii_energy_0.9_1.1"] = 0.9 <= crit["energy_ratio_at_modis_bright_cells_end"] <= 1.1
    crit["PASS_all"] = bool(crit["pass_i_fall_gt_10x"] and crit["pass_ii_median_err_lt_0.1"] and crit["pass_iii_energy_0.9_1.1"])
    out = {"gate": "overfit", "criterion_amendment5": crit, "anatomy": floor_anatomy(model, inp, cfg, dev, amp), "loss": a.loss, "tiles": idx, "bright_cells_per_tile": nb, "loss_start": l0, "loss_end": l1, "fall_factor": l0 / max(l1, 1e-30), "fidelity_init": f0, "fidelity_end": f1,
           "pass_loss_10x": bool(l0 / max(l1, 1e-30) > 10), "curve": log, "seconds": time.time() - t0}
    return out


@torch.no_grad()
def floor_anatomy(model, inp, cfg, dev, amp):
    """What is the loss floor of the memorise test made of? Gamma deviance of the clean M->M (MODIS view of the MODIS-decoded scene vs measured MODIS) and V->V (VIIRS view of the VIIRS-decoded scene vs measured VIIRS), per band, bright vs other, interior vs mask border, saturated, concentration in the worst 1 % of cells, signed bias."""
    import torch.nn.functional as F
    from nightrad.raw.core import run_scenes
    from nightrad.raw.losses import gamma_dev
    from nightrad.raw.views import ModisView, viirs_view
    model.eval(); P = Prep(inp, cfg); sc, _ = run_scenes(model, P, cfg, None, amp); mview = ModisView(cfg.view)
    bright = diag.bright_region(P, inp, cfg)
    interior = ~(F.max_pool2d((~P.common).float()[:, None], 5, 1, 2)[:, 0] > 0)
    out = {}
    for name, tgt, mask, pred in (("mm", "modis", P.ok_foot, mview(sc[("modis", "clean")]["B"], sc[("modis", "clean")]["E"], P.fp)),
                                  ("vv", "viirs", P.common, viirs_view(sc[("viirs", "clean")]["B"], sc[("viirs", "clean")]["E"]))):
        y = torch.exp(P.logL[tgt])[:, :pred.shape[1]]; d = gamma_dev(pred, y); sat = inp[f"{tgt}_sat"] > 0  # --out-bands 4: only the 4 um band is predicted and scored
        o = {}
        for b, bn in ((0, "4um"), (1, "11um"))[:pred.shape[1]]:
            db = d[:, b]; m = mask
            o[bn] = {"mean_dev": float(db[m].mean()), "mean_dev_bright": float(db[m & bright].mean()) if (m & bright).any() else None, "mean_dev_interior": float(db[m & interior].mean()), "mean_dev_mask_border": float(db[m & ~interior].mean()) if (m & ~interior).any() else None,
                     "mean_dev_saturated": float(db[m & sat].mean()) if (m & sat).any() else None, "frac_saturated": float((m & sat).float().sum() / m.float().sum()),
                     "signed_ln_bias": float((torch.log(pred[:, b].clamp(min=1e-6)) - torch.log(y[:, b].clamp(min=1e-6)))[m].mean()), "median_abs_ln": float((torch.log(pred[:, b].clamp(min=1e-6)) - torch.log(y[:, b].clamp(min=1e-6))).abs()[m].median())}
            v = db[m].flatten(); k = max(1, int(0.01 * v.numel())); top = torch.topk(v, k).values.sum()
            o[bn]["share_of_loss_from_worst_1pct_cells"] = float(top / v.sum().clamp(min=1e-12))
            o[bn]["mean_dev_without_worst_1pct"] = float((v.sum() - top) / max(v.numel() - k, 1))
            # is the residual spatially white noise? lag-1 autocorrelation of the ln residual
            r = (torch.log(pred[:, b].clamp(min=1e-6)) - torch.log(y[:, b].clamp(min=1e-6)))
            mm_ = m[:, :, 1:] & m[:, :, :-1]; a_, b_ = r[:, :, 1:][mm_], r[:, :, :-1][mm_]
            o[bn]["residual_lag1_autocorr"] = float((a_ * b_).mean() / (a_.pow(2).mean().clamp(min=1e-12)))
        out[name] = o
    return out


def gate_path(a, cfg, ds, dev):
    """Same batch, same hiding draw: training path (train mode, grad) vs evaluation path (eval mode, no grad, the diag.full_eval loader path)."""
    amp = amp_of(a, dev)
    seed_all(a.seed); model = RawHarmonizer(cfg.net).to(dev)
    idx = list(range(0, len(ds), max(1, len(ds) // 16)))[:16]
    b = batch_of(ds, idx, dev)
    # perturb the model a little (50 steps) so it is not at the trivial init
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for _ in range(a.steps):
        inp, _ = split_batch(b, dev); gen = torch.Generator(device=dev).manual_seed(1)
        loss, _ = forward_loss(model, inp, cfg, gen, amp); opt.zero_grad(); loss.backward(); opt.step()
    res = {"gate": "path", "loss": a.loss}
    for hiding in (False, True):
        cfg.hide.enabled = hiding
        inp, _ = split_batch(b, dev)
        model.train(); gen = torch.Generator(device=dev).manual_seed(7)
        lt, logt = forward_loss(model, inp, cfg, gen, amp)
        model.eval(); gen = torch.Generator(device=dev).manual_seed(7)
        with torch.no_grad():
            le, loge = forward_loss(model, inp, cfg, gen, amp)
        # loader path
        loader = [{k: v[i:i + 8] for k, v in b.items()} for i in range(0, 16, 8)]
        fe = diag.full_eval(model, loader, cfg, dev, amp, seed=7)
        res[f"hiding_{hiding}"] = {"loss_train_path": float(lt), "loss_eval_path": float(le), "abs_diff": abs(float(lt) - float(le)), "rel_diff": abs(float(lt) - float(le)) / max(abs(float(lt)), 1e-30),
                                   "loader_path_val_total_mean_of_2_batches": fe["val_total"]}
    # masks / scaling: Prep from the training config vs the evaluation reference config
    from nightrad.raw.sweep_eval import ref_cfg
    cfg.hide.enabled = False
    inp, _ = split_batch(b, dev); P1 = Prep(inp, cfg); P2 = Prep(inp, ref_cfg(cfg))
    res["masks_identical"] = {"common": bool((P1.common == P2.common).all()), "ok_foot": bool((P1.ok_foot == P2.ok_foot).all()), "logL_viirs": float((P1.logL["viirs"] - P2.logL["viirs"]).abs().max()), "med_viirs": float((P1.med["viirs"] - P2.med["viirs"]).abs().max())}
    res["view_cfg_train"] = {"w_scan": cfg.view.w_scan, "w_track": cfg.view.w_track, "kernel": cfg.view.kernel, "t_fire": cfg.view.t_fire}
    res["view_cfg_eval"] = {"w_scan": ref_cfg(cfg).view.w_scan, "w_track": ref_cfg(cfg).view.w_track, "kernel": ref_cfg(cfg).view.kernel, "t_fire": ref_cfg(cfg).view.t_fire}
    res["has_dropout_or_batchnorm"] = any(isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm2d, torch.nn.modules.batchnorm._BatchNorm)) for m in model.modules())
    # protocol collect disables hiding: check by construction
    from nightrad.raw import protocol_eval  # noqa: F401
    res["protocol_collect_hide_disabled"] = "replace(cfg.hide, enabled=False)" in open(protocol_eval.__file__).read()
    res["pass"] = bool(res["hiding_False"]["rel_diff"] < 1e-3 and res["hiding_True"]["rel_diff"] < 1e-3 and res["masks_identical"]["common"] and res["masks_identical"]["ok_foot"] and not res["has_dropout_or_batchnorm"])
    return res


def gate_share(a, cfg, ds, dev):
    """Loss share and gradient share of cells under MODIS-bright pixels at initialisation on the first 64 tiles (spread over the set)."""
    amp = amp_of(a, dev)
    seed_all(a.seed); model = RawHarmonizer(cfg.net).to(dev)
    idx = list(range(0, len(ds), max(1, len(ds) // 64)))[:64]
    loader = [batch_of(ds, idx[i:i + 16], dev) for i in range(0, len(idx), 16)]
    fe = diag.full_eval_weighted(model, loader, cfg, dev, amp)
    gs = diag.grad_shares(model, loader, cfg, dev, amp, n_batches=4)
    return {"gate": "share", "loss": a.loss, "tiles": len(idx), "loss_share_bright": fe["loss_share_bright"], "bright_cells": fe["n_bright_cells"], "cells": fe["n_cells"], "bright_cell_fraction": fe["n_bright_cells"] / fe["n_cells"],
            "terms_weighted": fe["weighted"], "grad": gs, "pred_excess_bright": fe["pred_excess_bright"], "meas_excess_bright": fe["meas_excess_bright"]}


def gate_bf16(a, cfg, ds, dev):
    """Review item: bf16 rounding of the head outputs. Same batch (the 16 tiles with most MODIS-bright cells), model after --steps bf16 steps, loss and predicted excess in fp32 vs bf16, difference on cells under MODIS-bright pixels."""
    from nightrad.raw.core import run_scenes
    cfg.hide.enabled = False
    idx, _ = pick_tiles(ds, cfg, dev, n=16)
    inp, _ = split_batch(batch_of(ds, idx, dev), dev)
    seed_all(a.seed); model = RawHarmonizer(cfg.net).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for _ in range(a.steps):
        loss, _ = forward_loss(model, inp, cfg, None, torch.bfloat16); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); res = {"gate": "bf16", "loss": a.loss, "steps": a.steps}
    with torch.no_grad():
        P = Prep(inp, cfg); bright = diag.bright_region(P, inp, cfg)
        out = {}
        for name, amp in (("fp32", None), ("bf16", torch.bfloat16)):
            sc, _ = run_scenes(model, P, cfg, None, amp); s = sc[("modis", "clean")]
            l, _ = forward_loss(model, inp, cfg, None, amp)
            out[name] = (s["E"][:, 0].float(), s["B"][:, 0].float(), float(l))
        E32, E16 = out["fp32"][0], out["bf16"][0]; B32, B16 = out["fp32"][1], out["bf16"][1]
        res["loss_fp32"], res["loss_bf16"] = out["fp32"][2], out["bf16"][2]; res["loss_rel_diff"] = abs(res["loss_fp32"] - res["loss_bf16"]) / abs(res["loss_fp32"])
        res["excess_bright_mean_fp32"], res["excess_bright_mean_bf16"] = float(E32[bright].mean()), float(E16[bright].mean())
        res["excess_bright_sum_rel_diff"] = float((E16[bright].sum() - E32[bright].sum()).abs() / E32[bright].sum())
        res["excess_bright_cellwise_median_rel_diff"] = float(((E16 - E32).abs() / E32.clamp(min=1e-9))[bright].median())
        res["background_cellwise_median_rel_diff"] = float(((B16 - B32).abs() / B32)[P.common].median())
        res["n_bright"] = int(bright.sum())
    res["pass"] = bool(res["excess_bright_sum_rel_diff"] < 0.01 and res["loss_rel_diff"] < 0.01)
    return res


def gate_seed(a, cfg, ds, dev):
    amp = amp_of(a, dev); runs = []
    for _ in range(2):
        seed_all(a.seed, deterministic=True); model = RawHarmonizer(cfg.net).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr); idx = list(range(0, len(ds), max(1, len(ds) // 32)))[:32]
        ls = []
        for s in range(a.steps):
            b = batch_of(ds, [idx[(s * 8 + j) % len(idx)] for j in range(8)], dev); inp, _ = split_batch(b, dev)
            gen = torch.Generator(device=dev).manual_seed(s)
            loss, _ = forward_loss(model, inp, cfg, gen, amp); opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss))
        runs.append(ls)
    d = float(np.max(np.abs(np.array(runs[0]) - np.array(runs[1]))))
    return {"gate": "seed", "loss": a.loss, "steps": a.steps, "max_abs_loss_diff": d, "first": runs[0][:3], "last": [runs[0][-1], runs[1][-1]], "pass": d < 1e-6}


def main():
    p = make_parser(); p.add_argument("--gate", required=True); p.add_argument("--steps", type=int, default=3000); p.add_argument("--out-json", default=None)
    a = p.parse_args(); cfg = make_cfg(a)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from nightrad.raw import prereg
    prereg.open_split(a.data, allow_test=False, frozen_path=a.frozen, cfg=cfg.to_dict(), ckpt=None)
    ds = PackedShards(a.data)
    out = {"overfit": gate_overfit, "path": gate_path, "share": gate_share, "seed": gate_seed, "bf16": gate_bf16}[a.gate](a, cfg, ds, dev)
    out["args"] = vars(a)
    print("RESULT", json.dumps({k: v for k, v in out.items() if k not in ("curve", "args")}, default=float))
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
