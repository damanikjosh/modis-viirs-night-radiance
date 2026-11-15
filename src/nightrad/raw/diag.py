"""Training diagnostics for (DIAG): the SAME loss on the full validation set, split into its terms and into the
contribution of cells under MODIS-bright pixels (selected by the MODIS INPUT only) vs all other cells; share of the gradient from those
cells; mean predicted excess; and a cheap protocol subset (block AUC, hottest-cell share, MODIS-selected energy ratio).
Nothing here changes training: the loss tap (losses.TAP) is None unless a diagnostic switches it on. New log fields only."""
from __future__ import annotations

import numpy as np
import torch

from nightrad.raw import losses
from nightrad.raw.core import Prep, forward_loss
from nightrad.raw.schema import split_batch


def term_spec(cfg, has_blocks: bool):
    """[(name, weight)] in the order the loss functions append to losses.TAP."""
    lc = cfg.loss; tw = lc.term_weights; out = []
    if lc.mode == "gamma":
        sw = lc.scale_w; ssum = max(sum(sw), 1e-9)
        out += [("mv_cell", tw["mv"] * sw[0] / ssum), ("mv_pix", tw["mv"] * sw[1] / ssum)]
        if has_blocks:
            out.append(("mv_blk", tw["mv"] * sw[2] / ssum))
        out += [("vm", tw["vm"]), ("vv", tw["vv"]), ("mm", tw["mm"])]
    else:
        out += [("mm", tw["mm"]), ("vv", tw["vv"]), ("vm", tw["vm"]), ("mv", tw["mv"])]
    if lc.w_energy > 0:
        out.append(("energy", lc.w_energy))
    return out


def bright_region(P: Prep, inp: dict, cfg) -> torch.Tensor:
    """Cells of MODIS-bright footprints, chosen from the MODIS input only (rho4 >= fire level, unsaturated, complete)."""
    return P.ok_foot & (P.rho4["modis"] >= cfg.loss.fire_level) & (inp["modis_sat"] == 0)


def tapped_loss(model, inp, cfg, dev, amp_dtype, gen):
    """forward_loss with the contribution tap on -> (total, logs, aux, [(name, weight, contrib [B,H,W])], bright mask)."""
    losses.TAP = []
    try:
        total, logs, aux = forward_loss(model, inp, cfg, gen, amp_dtype, return_aux=True)
        tap = losses.TAP
    finally:
        losses.TAP = None
    P = aux["P"]
    spec = term_spec(cfg, P.fp.nbr is not None)
    if len(spec) != len(tap):
        spec = (spec + [(f"extra{i}", 0.0) for i in range(len(tap))])[: len(tap)]
    return total, logs, aux, [(n, w, c) for (n, w), c in zip(spec, tap)], bright_region(P, inp, cfg)


@torch.no_grad()
def full_eval(model, loader, cfg, dev, amp_dtype, seed=1234, max_batches=None):
    """Same loss as training on the whole validation loader (fixed hiding draw), by term and by region; mean predicted vs measured excess."""
    was = model.training; model.eval()
    gen = torch.Generator(device=dev).manual_seed(seed)
    T = {}; tot = 0.0; n = 0; ex = np.zeros(6)  # pred/meas excess sums over bright cells, over all common cells; counts
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        inp, _ = split_batch(b, dev); nb = inp["viirs_L"].shape[0]
        total, logs, aux, taps, bright = tapped_loss(model, inp, cfg, dev, amp_dtype, gen)
        tot += float(total) * nb; n += nb
        for name, w, c in taps:
            d = T.setdefault(name, np.zeros(3))
            cb = float((c * bright).sum()); ca = float(c.sum())
            d += np.array([ca, cb, 0.0]) * nb
        P = aux["P"]
        E = aux["scenes"][("modis", "clean")]["E"][:, 0].float()
        Bv = torch.exp(P.med["viirs"][:, 0]); Ev = (torch.exp(P.logL["viirs"][:, 0]) - Bv).clamp(min=0)
        cm = P.common
        ex += np.array([float((E * bright).sum()), float((Ev * bright).sum()), float(bright.sum()), float((E * cm).sum()), float((Ev * cm).sum()), float(cm.sum())])
    if was:
        model.train()
    n = max(n, 1)
    out = {"val_total": tot / n, "terms": {}, "weighted": {}}
    for name, v in T.items():
        out["terms"][name] = {"all": v[0] / n, "bright": v[1] / n, "other": (v[0] - v[1]) / n}
    # weighted contributions to the total (the number the optimiser sees)
    out["pred_excess_bright"] = float(ex[0] / max(ex[2], 1)); out["meas_excess_bright"] = float(ex[1] / max(ex[2], 1))
    out["pred_excess_all"] = float(ex[3] / max(ex[5], 1)); out["meas_excess_all"] = float(ex[4] / max(ex[5], 1))
    out["n_bright_cells"] = float(ex[2]); out["n_cells"] = float(ex[5])
    return out


def full_eval_weighted(model, loader, cfg, dev, amp_dtype, seed=1234, max_batches=None):
    """full_eval + weighted contributions (weight x term) summed to the total: loss share of bright cells."""
    base = full_eval(model, loader, cfg, dev, amp_dtype, seed, max_batches)
    # weights are config-only: recompute the spec (block term present when the loader has neighbours: detect from the term names)
    has_blk = "mv_blk" in base["terms"]
    spec = dict(term_spec(cfg, has_blk))
    wt = {nm: {k: spec.get(nm, 0.0) * v for k, v in d.items()} for nm, d in base["terms"].items()}
    base["weighted"] = wt
    wa = sum(d["all"] for d in wt.values()); wb = sum(d["bright"] for d in wt.values())
    base["loss_share_bright"] = float(wb / wa) if wa > 0 else None
    base["terms_weighted_sum"] = float(wa)
    return base


def grad_shares(model, loader, cfg, dev, amp_dtype, seed=1234, n_batches=2):
    """Gradient of the total loss restricted to bright cells vs other cells (same hiding draw), per batch averaged:
    norms, and projection share <g_bright, g_total>/|g_total|^2 (shares add to 1)."""
    was = model.training; model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    gen = torch.Generator(device=dev).manual_seed(seed)
    acc = []
    for bi, b in enumerate(loader):
        if bi >= n_batches:
            break
        inp, _ = split_batch(b, dev)
        total, logs, aux, taps, bright = tapped_loss(model, inp, cfg, dev, amp_dtype, gen)
        if not taps:
            continue
        lb = sum(w * (c * bright).sum() for _, w, c in taps)
        lo = sum(w * (c * ~bright).sum() for _, w, c in taps)
        if not (lb.requires_grad and lo.requires_grad):
            continue
        gb = torch.autograd.grad(lb, params, retain_graph=True, allow_unused=True)
        go = torch.autograd.grad(lo, params, retain_graph=False, allow_unused=True)
        fl = lambda g: torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().float() for x, p in zip(g, params)])
        gb, go = fl(gb), fl(go); gt = gb + go
        nt2 = float(gt @ gt) + 1e-30
        acc.append([float(gb.norm()), float(go.norm()), float(gt.norm()), float(gb @ gt) / nt2, float(go @ gt) / nt2, float(lb.detach()), float(lo.detach())])
    if was:
        model.train()
    if not acc:
        return {}
    a = np.mean(acc, 0)
    return {"gnorm_bright": a[0], "gnorm_other": a[1], "gnorm_total": a[2], "grad_share_bright": a[3], "grad_share_other": a[4], "loss_bright": a[5], "loss_other": a[6]}


class ProtoSubset:
    """Cheap protocol_eval_v2 subset on the validation tiles (final model, hiding off): block AUC (model / spread evenly / model fire-off), hottest-cell share, MODIS-selected energy ratio."""

    def __init__(self, val_ds, every=1, max_tiles=None):
        from nightrad.raw.protocol_eval import load_all
        idx = list(range(0, len(val_ds), every))[:max_tiles]
        items = [val_ds[i] for i in idx]
        self.tiles = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        self.has_pair = "eval_pair_id" in self.tiles

    def __call__(self, model, cfg, dev, amp_dtype):
        if not self.has_pair:
            return {}
        from nightrad.raw import protocol_eval as pe
        res = pe.collect(model, self.tiles, cfg, dev, 0.0, amp_dtype)
        s = pe.summarize(res)
        o = {"auc_model": s["auc"]["model"]["mean"], "auc_uniform": s["auc"]["uniform"]["mean"], "auc_model_off": s["auc"]["model_off"]["mean"], "n_pairs": s["auc"]["model"]["n_pairs"],
             "energy_msel": s["energy_msel"].get("model"), "energy_msel_uniform": s["energy_msel"].get("uniform"), "energy_vfire": s["energy"].get("model")}
        c = s.get("conc", {})
        for m in ("model", "measured", "uniform"):
            if m in c:
                o[f"top1_{m}"] = c[m]["top1_share"]; o[f"top3_{m}"] = c[m]["top3_share"]
        r = s.get("rule", {}).get("model", {})
        o["flag_ratio"] = r.get("flag_count_ratio")
        return o
