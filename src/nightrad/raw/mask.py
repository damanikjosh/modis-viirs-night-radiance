"""Masks: per-sensor ROI from the validity class, common mask, cloud-edge widening, footprint rule.

One boolean common mask is used identically for (a) what the network may see and (b) where the loss is scored.
Codes: 0 valid, 1 cloud, 2 water/non-land, 3 bad/missing/bow-tie, 4 not covered, 5 glint/other.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

VALID, CLOUD, WATER, BAD, NOT_COVERED, OTHER = range(6)


def roi_mask(code: torch.Tensor, widen: int = 0) -> torch.Tensor:
    """[B,H,W] bool. widen = d cells: also drop valid cells within d cells (Chebyshev) of a cloud cell."""
    ok = code == VALID
    if widen > 0:
        cloud = (code == CLOUD).float()[:, None]
        near = F.max_pool2d(cloud, 2 * widen + 1, 1, widen)[:, 0] > 0
        ok = ok & ~near
    return ok


def common_mask(code_modis: torch.Tensor, code_viirs: torch.Tensor, widen: int = 0) -> torch.Tensor:
    """Both sensors valid (code 0 implies covered, so overlap is implied)."""
    return roi_mask(code_modis, widen) & roi_mask(code_viirs, widen)


class Footprints:
    """Vectorised scatter ops over MODIS source-pixel ids. pid [B,H,W] long in 0..H*W-1, H*W = dummy (no pixel)."""

    def __init__(self, pid: torch.Tensor, nbr: torch.Tensor | None = None):
        """nbr [B,4,H,W] long (optional): dense ids of the native pixels next to each cell's pixel: along-scan -, +, along-track -, +;
        H*W = none. Needed only by the weighted (triangular) MODIS view and the 3x3 scorer."""
        b, h, w = pid.shape
        self.nbr = nbr
        self.pix_ok = None
        self.b, self.h, self.w, self.n = b, h, w, h * w + 1
        self.pid = pid.reshape(b, -1)
        self.has = self.pid != h * w
        self.count = self._sum(torch.ones(b, 1, h * w, device=pid.device)).clamp(min=1.0)  # [B,1,n]

    def _sum(self, v: torch.Tensor) -> torch.Tensor:  # v [B,C,HW] -> [B,C,n]
        out = torch.zeros(v.shape[0], v.shape[1], self.n, dtype=v.dtype, device=v.device)
        return out.scatter_add_(2, self.pid[:, None].expand(-1, v.shape[1], -1), v)

    def _gather(self, p: torch.Tensor) -> torch.Tensor:  # [B,C,n] -> [B,C,HW]
        return torch.gather(p, 2, self.pid[:, None].expand(-1, p.shape[1], -1))

    def mean(self, v: torch.Tensor) -> torch.Tensor:
        """Per-footprint mean broadcast back to every cell: [B,C,H,W] -> [B,C,H,W]."""
        b, c = v.shape[:2]
        m = self._gather(self._sum(v.reshape(b, c, -1)) / self.count)
        return m.view(b, c, self.h, self.w)

    def total(self, v: torch.Tensor) -> torch.Tensor:
        b, c = v.shape[:2]
        return self._gather(self._sum(v.reshape(b, c, -1))).view(b, c, self.h, self.w)

    def pixel_mean_table(self, v: torch.Tensor) -> torch.Tensor:
        b, c = v.shape[:2]
        return self._sum(v.reshape(b, c, -1)) / self.count

    def _pix_ok_at(self, idx: torch.Tensor) -> torch.Tensor:  # idx [B,HW] -> bool
        ok = self.pix_ok if self.pix_ok is not None else torch.ones(idx.shape[0], self.n, dtype=torch.bool, device=idx.device)
        ok = ok.clone()
        ok[:, -1] = False
        return torch.gather(ok, 1, idx)

    def tri_mean(self, v: torch.Tensor, w_scan: float, w_track: float) -> torch.Tensor:
        """Weighted neighbourhood view: (1-2a-2b) own pixel mean + a each along-scan neighbour + b each along-track neighbour.
        A neighbour that is missing, incomplete or not entirely inside the common mask is replaced by the own pixel (no masked value enters)."""
        assert self.nbr is not None, "tri view needs the neighbour table (modis_nbr)"
        b, c = v.shape[:2]
        t = self.pixel_mean_table(v)
        own = torch.gather(t, 2, self.pid[:, None].expand(-1, c, -1))
        acc = (1 - 2 * w_scan - 2 * w_track) * own
        for k, wk in enumerate((w_scan, w_scan, w_track, w_track)):
            if wk == 0:
                continue
            idx = self.nbr[:, k].reshape(b, -1)
            nb = torch.gather(t, 2, idx[:, None].expand(-1, c, -1))
            acc = acc + wk * torch.where(self._pix_ok_at(idx)[:, None], nb, own)
        return acc.view(b, c, self.h, self.w)

    def sum3x3(self, v: torch.Tensor) -> torch.Tensor:
        """Sum of the pixel values (v [B,H,W], constant per pixel) over the 3x3 native-pixel neighbourhood of each cell's pixel,
        counting only valid (complete, in-mask) pixels; broadcast to cells. Diagonals via the neighbour table of the neighbours."""
        b = v.shape[0]
        if self.nbr is None:  # no neighbour table: own pixel only
            return self.mean(v[:, None])[:, 0]
        t = self.pixel_mean_table(v[:, None])[:, 0]  # [B,n]
        tab = torch.zeros(b, 4, self.n, dtype=torch.long, device=v.device) + (self.n - 1)
        tab.scatter_(2, self.pid[:, None].expand(-1, 4, -1), self.nbr.reshape(b, 4, -1))
        own = self.pid
        s_m, s_p, t_m, t_p = [self.nbr[:, k].reshape(b, -1) for k in range(4)]
        g = lambda ids, k: torch.gather(tab[:, k], 1, ids)
        ids = [own, s_m, s_p, t_m, t_p, g(s_m, 2), g(s_m, 3), g(s_p, 2), g(s_p, 3)]
        tot = torch.zeros_like(own, dtype=v.dtype)
        for i, idx in enumerate(ids):
            val = torch.gather(t, 1, idx)
            tot = tot + torch.where(self._pix_ok_at(idx) | (i == 0), val, torch.zeros_like(val))
        return tot.view(b, self.h, self.w)

    def any(self, flag: torch.Tensor) -> torch.Tensor:
        """[B,H,W] bool -> True for every cell whose footprint contains a True cell."""
        return (self.total(flag.float()[:, None])[:, 0] > 0)

    def logsumexp(self, x: torch.Tensor) -> torch.Tensor:
        """Per-footprint logsumexp of x [B,H,W], broadcast to cells (for the allocation softmax)."""
        b = x.shape[0]
        xf = x.reshape(b, -1)
        mx = torch.full((b, self.n), -1e30, dtype=x.dtype, device=x.device).scatter_reduce(1, self.pid, xf, "amax", include_self=True)
        mxc = torch.gather(mx, 1, self.pid)
        s = torch.zeros(b, self.n, dtype=x.dtype, device=x.device).scatter_add_(1, self.pid, torch.exp(xf - mxc))
        return (torch.gather(s, 1, self.pid).clamp(min=1e-30).log() + mxc).view_as(x)

    def ok(self, common: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
        """Cell-level bool: footprint has a pixel, is complete, and ALL its cells are in the common mask."""
        b = common.shape[0]
        n_in = self._sum(common.float().reshape(b, 1, -1))
        n_complete = self._sum(complete.float().reshape(b, 1, -1))
        full = (n_in == self.count) & (n_complete == self.count)
        self.pix_ok = full[:, 0]  # [B,n] bool: pixel complete and entirely inside the common mask
        return (self._gather(full.float())[:, 0] > 0).view_as(common) & self.has.view_as(common)


def valid_fraction_ok(common: torch.Tensor, min_frac: float = 0.10) -> torch.Tensor:
    """[B] bool: tile keeps >= min_frac of its cells in the common mask."""
    return common.flatten(1).float().mean(1) >= min_frac


def dense_pid(pid: torch.Tensor) -> torch.Tensor:
    """int32 pid with -1 for none -> long with dummy slot H*W."""
    h, w = pid.shape[-2:]
    p = pid.long()
    return torch.where(p < 0, torch.full_like(p, h * w), p)
