"""Local masked median (robust background anchor). Only cells with valid=True can influence the result, exactly."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_local_median(x: torch.Tensor, valid: torch.Tensor, k: int = 7, dilation: int = 2) -> torch.Tensor:
    """x [B,C,H,W] float32 (log radiance), valid [B,1|C,H,W] bool -> median over the k*k taps (spacing `dilation`) of valid cells.

    Window span = dilation*(k-1)+1 (default 13 cells). Cells with no valid tap fall back to the tile's masked mean.
    NaN/inf at invalid cells is safe (replaced by +inf before sorting).
    """
    b, c, h, w = x.shape
    p = dilation * (k // 2)
    valid = valid.expand(b, c, h, w)
    xv = torch.where(valid, x, torch.full_like(x, float("inf")))
    u = F.unfold(F.pad(xv.reshape(b * c, 1, h, w), (p, p, p, p), value=float("inf")), k, dilation=dilation)  # [BC,k*k,HW]
    cnt = torch.isfinite(u).sum(1)
    su = u.sort(1).values
    idx = ((cnt - 1) // 2).clamp(min=0)
    med = su.gather(1, idx[:, None])[:, 0]
    tile_mean = (torch.where(valid, x, torch.zeros_like(x)).sum((2, 3)) / valid.sum((2, 3)).clamp(min=1)).reshape(b * c, 1)
    med = torch.where(cnt > 0, med, tile_mean.expand_as(med))
    return med.view(b, c, h, w)


_compiled = None


def enable_compile(on: bool = True):
    """torch.compile the median (fuses the pad/where/gather/fallback elementwise work around the sort)."""
    global _compiled
    _compiled = torch.compile(_masked_local_median, dynamic=False) if on else None


def masked_local_median(x, valid, k: int = 7, dilation: int = 2):
    return (_compiled or _masked_local_median)(x, valid, k, dilation)
