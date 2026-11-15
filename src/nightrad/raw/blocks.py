"""3x3 native-pixel block errors of the excess (one definition: the sweep's evaluation, the block-level training term and the log all use these).

vm_blocks: V->M. Blocks centred on MODIS footprints that are fire-level by the MEASURED MODIS radiance (rho4 >= fl, unsaturated).
mv_blocks: M->V. Blocks centred on footprints that contain a fire-level measured VIIRS cell (rho4 >= fl).
Both return (ln((sum predicted excess + eps) / (sum measured excess + eps)) [B,H,W], sel [B,H,W] bool). Differentiable in the prediction."""
from __future__ import annotations

import torch

EPS = 1e-3


def vm_blocks(P, pred_m4: torch.Tensor, inp: dict, fl: float):
    """pred_m4 [B,H,W] predicted MODIS 4 um radiance (linear)."""
    Bm = torch.exp(P.med["modis"][:, 0]); tm = P.logL["modis"][:, 0]
    sel = P.ok_foot & (P.rho4["modis"] >= fl) & (inp["modis_sat"] == 0)
    pe = (pred_m4 - Bm).clamp(min=0) * P.ok_foot
    me = (torch.exp(tm) - Bm).clamp(min=0) * P.ok_foot
    return torch.log((P.fp.sum3x3(pe) + EPS) / (P.fp.sum3x3(me) + EPS)), sel


def mv_blocks(P, pred_v4: torch.Tensor, inp: dict, fl: float):
    """pred_v4 [B,H,W] predicted VIIRS 4 um radiance (linear)."""
    Bv = torch.exp(P.med["viirs"][:, 0]); tv = P.logL["viirs"][:, 0]
    fire_v = P.common & (P.rho4["viirs"] >= fl)
    blk = lambda e: P.fp.sum3x3(P.fp.total(e[:, None])[:, 0])
    pe = (pred_v4 - Bv).clamp(min=0) * P.ok_foot
    me = (torch.exp(tv) - Bv).clamp(min=0) * P.ok_foot
    sel = P.ok_foot & P.fp.any(fire_v & P.ok_foot)
    return torch.log((blk(pe) + EPS) / (blk(me) + EPS)), sel


def once_per_footprint(sel: torch.Tensor, fp) -> torch.Tensor:
    """sel [B,H,W] bool -> same but only the first selected cell of every footprint (each footprint counted once)."""
    b, h, w = sel.shape
    flat = sel.reshape(b, -1)
    idx = torch.arange(h * w, device=sel.device)[None].expand(b, -1)
    big = h * w + 1
    first = torch.full((b, fp.n), big, dtype=torch.long, device=sel.device)
    first.scatter_reduce_(1, fp.pid, torch.where(flat, idx, torch.full_like(idx, big)), "amin", include_self=True)
    return (flat & (torch.gather(first, 1, fp.pid) == idx)).view(b, h, w)
