"""Placement probe: can WHERE inside a MODIS fire pixel the VIIRS-measured excess sits be learned from MODIS input alone?

Probe model = the same RawHarmonizer U-Net (MODIS radiance + geometry features only, deterministic) with its footprint-allocation logit as the only output used:
share p_c = softmax over the cells of a MODIS footprint. Training target = the MEASURED VIIRS 4 um excess share q_c = e_c / sum_footprint e (target only; the input
never sees VIIRS); loss = footprint cross-entropy -sum q log p over footprints SELECTED BY THE MODIS INPUT (fire-level MODIS pixel, unsaturated, complete, >= 3 cells)
that also hold a measured VIIRS excess (otherwise the share is undefined). No fire product, no sampling head, no hard-total layer.

No-learning heuristics (all use the MODIS input and the footprint geometry only):
  chance      : uniform random cell, expected hit rate 1/N (N = cells of that footprint)
  centre      : the cell closest to the centroid of the pixel's cells
  psf         : cell score = sum over the pixel itself and its 4 neighbours (along-scan -/+, along-track -/+) of (MODIS excess of that pixel) x triangular kernel
                max(0, 1 - |cell - centroid_of_that_pixel| / sqrt(N)); the kernel is one pixel spacing wide (the point-spread overlap of neighbouring pixels)
  psf_nbr     : the same with the pixel's own term left out (neighbour overlap only)
"""
from __future__ import annotations

import torch

from nightrad.raw.core import Prep, RunConfig, _anchor, _features

FIRE = 0.5


def meas_excess(P: Prep) -> torch.Tensor:
    """Measured VIIRS 4 um excess over its local-median background [B,H,W] (>= 0), inside complete footprints. TARGET ONLY."""
    return (torch.exp(P.logL["viirs"][:, 0]) - torch.exp(P.med["viirs"][:, 0])).clamp(min=0) * P.ok_foot


def select_cells(P: Prep, inp: dict, e: torch.Tensor, min_cells: int = 3) -> torch.Tensor:
    """Cells of footprints chosen by the MODIS input (fire-level, unsaturated, complete) that hold a measured VIIRS excess (share defined). [B,H,W] bool."""
    cnt = P.fp.total(torch.ones_like(e)[:, None])[:, 0]
    tot = P.fp.total(e[:, None])[:, 0]
    return P.ok_foot & (P.rho4["modis"] >= FIRE) & (inp["modis_sat"] == 0) & (cnt >= min_cells) & (tot > 1e-9)


# build_features channel layout: 0-1 ln radiance (4, 11), 2-3 excess over local median, 4 mask, 5 saturated, 6 band-21 flag, 7 view zenith, 8-9 pixel size, 10 cos solar zenith, 11 4-11 contrast
ABLATE_ZERO = {"geometry_only": (0, 1, 2, 3, 5, 6, 11), "radiance_only": (7, 8, 9, 10)}


def predict_logits(model, P: Prep, cfg: RunConfig, amp_dtype=None, ablate: str | None = None) -> torch.Tensor:
    """MODIS clean pass only -> allocation logit [B,H,W] fp32.
    ablate (controls): 'geometry_only' = radiance channels zeroed (mask + view zenith + pixel size + sun angle stay), 'radiance_only' = geometry channels zeroed,
    'modis_mask' = the input mask is the MODIS validity only (no VIIRS-derived cloud/bad cells enter)."""
    from nightrad.raw.background import masked_local_median
    from nightrad.raw.mask import roi_mask
    inp = P.inp; im = P.common; med = P.med["modis"]
    if ablate == "modis_mask":
        im = roi_mask(inp["modis_code"]); med = masked_local_median(P.logL["modis"], im[:, None], cfg.median_k, cfg.median_dil)
    x, g = _features(P, "modis", im, med)
    if ablate in ABLATE_ZERO:
        x = x.clone(); x[:, list(ABLATE_ZERO[ablate])] = 0.0
    with torch.autocast(device_type=P.common.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(x, im, g, _anchor("modis", med), P.common.shape[0], None)
    return out["alloc"].float()


def share_loss(logits: torch.Tensor, fp, e: torch.Tensor, selc: torch.Tensor) -> torch.Tensor:
    """Footprint cross-entropy between the measured share q and the softmax share p, every selected footprint counted once."""
    logp = logits - fp.logsumexp(logits)
    q = e / fp.total(e[:, None])[:, 0].clamp(min=1e-12)
    cnt = fp.total(torch.ones_like(e)[:, None])[:, 0].clamp(min=1.0)
    w = selc.float()
    return -(q * logp * w).sum() / (w / cnt).sum().clamp(min=1e-9)


def _coords(fp):
    yy, xx = torch.meshgrid(torch.arange(fp.h, dtype=torch.float32), torch.arange(fp.w, dtype=torch.float32), indexing="ij")
    return torch.stack([yy, xx])[None].expand(fp.b, -1, -1, -1)


def _gather(tab, ids):  # tab [B,C,n], ids [B,HW] -> [B,C,HW]
    return torch.gather(tab, 2, ids[:, None].expand(-1, tab.shape[1], -1))


def centre_score(fp) -> torch.Tensor:
    xy = _coords(fp).to(fp.pid.device)
    cen = _gather(fp.pixel_mean_table(xy), fp.pid)                      # centroid of own pixel at every cell [B,2,HW]
    d = ((xy.reshape(fp.b, 2, -1) - cen) ** 2).sum(1).sqrt()
    return (-d).view(fp.b, fp.h, fp.w)


def psf_score(fp, exc: torch.Tensor, own: bool = True) -> torch.Tensor:
    """exc [B,H,W]: MODIS excess of each cell's pixel (constant within a pixel)."""
    assert fp.nbr is not None
    b = fp.b; dev = fp.pid.device
    xy = _coords(fp).to(dev).reshape(b, 2, -1)
    cen_t = fp.pixel_mean_table(_coords(fp).to(dev)); exc_t = fp.pixel_mean_table(exc[:, None])
    width = fp.count.sqrt()                                              # [B,1,n] linear pixel size in cells
    D = _gather(width, fp.pid)[:, 0]
    score = torch.zeros(b, fp.h * fp.w, device=dev)
    kern = lambda c: (1 - ((xy - c) ** 2).sum(1).sqrt() / D).clamp(min=0)
    if own:
        score = score + _gather(exc_t, fp.pid)[:, 0] * kern(_gather(cen_t, fp.pid))
    for k in range(4):
        ids = fp.nbr[:, k].reshape(b, -1)
        valid = fp._pix_ok_at(ids)
        score = score + torch.where(valid, _gather(exc_t, ids)[:, 0] * kern(_gather(cen_t, ids)), torch.zeros_like(score))
    return score.view(b, fp.h, fp.w)


def _rounds(score, fp, gen, k=3, qcap=None):
    """k rounds of 'take the best remaining cell of every footprint' (random tie break). Returns list of cell masks [B,HW] bool."""
    b = score.shape[0]; pid = fp.pid; dev = score.device
    work = score.reshape(b, -1).clone(); out = []
    for _ in range(k):
        mx = torch.full((b, fp.n), -float("inf"), device=dev).scatter_reduce(1, pid, work, "amax", include_self=True)
        hit = (work == torch.gather(mx, 1, pid)) & (work > -float("inf"))
        key = torch.where(hit, torch.rand(work.shape, generator=gen, device=dev), torch.full_like(work, -1.0))
        kmx = torch.full((b, fp.n), -1.0, device=dev).scatter_reduce(1, pid, key, "amax", include_self=True)
        one = hit & (key == torch.gather(kmx, 1, pid))
        out.append(one); work = torch.where(one, torch.full_like(work, -float("inf")), work)
    return out


def footprint_metrics(score: torch.Tensor, fp, e: torch.Tensor, selc: torch.Tensor, pairs: torch.Tensor, seed: int = 0) -> dict:
    """One row per selected footprint: pair, n_cells, hit (predicted top cell == measured hottest cell), top1 / top3 (measured share captured by the predicted top-1 / top-3 cells),
    max_share / max3_share (measured share on its own hottest / top-3 cells = the ceiling). Ties in the score are broken at random (seeded)."""
    dev = score.device; gen = torch.Generator(device=dev).manual_seed(seed); b = score.shape[0]
    q = (e / fp.total(e[:, None])[:, 0].clamp(min=1e-12)).reshape(b, -1)
    sp = _rounds(score, fp, gen); sm = _rounds(q, fp, gen)
    tab = lambda m: torch.zeros(b, fp.n, device=dev).scatter_add_(1, fp.pid, m.float())
    cap = lambda ms: [tab(m * q) for m in ms]
    cp, cm = cap(sp), cap(sm)
    hit = tab(sp[0] & sm[0])
    gat = lambda t: torch.gather(t, 1, fp.pid)
    from nightrad.raw.blocks import once_per_footprint
    once = once_per_footprint(selc, fp).reshape(b, -1)
    rows = lambda t: gat(t)[once].cpu().numpy()
    pr = pairs.reshape(b, -1)[:, 0].to(dev)[:, None].expand(-1, fp.h * fp.w)[once].cpu().numpy().astype(int)
    return {"pair": pr, "n_cells": gat(fp.count[:, 0])[once].cpu().numpy(), "hit": rows(hit), "top1": rows(cp[0]), "top3": rows(cp[0] + cp[1] + cp[2]),
            "max_share": rows(cm[0]), "max3_share": rows(cm[0] + cm[1] + cm[2])}
