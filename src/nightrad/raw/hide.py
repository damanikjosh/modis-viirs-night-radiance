"""Fire-weighted hiding of the raw signal: hidden on the network input, scored on the output. Configurable region selector."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nightrad.raw.mask import Footprints


@dataclass
class HideConfig:
    enabled: bool = True
    window: int = 9  # square window side, cells
    n_top: int = 3  # windows centred at the top-n cells by raw 4 um excess rho4
    rho_min: float = 0.5  # only cells with rho4 = ln(L4 / local median) >= rho_min (~ +11 K over the median)
    n_random: int = 1  # extra windows at random centres
    p_tile: float = 0.5  # probability that a tile is hidden at all
    partial_p: float = 0.0  # probability (per tile) of PARTIAL fire hiding: small off-centre windows that cover part of the fire only
    partial_window: int = 3
    center_jitter: int = 0  # NEW (review, sweep 1 stage 2 default 3): shift each top-cell window centre by up to +-this many cells so the hole's centre does not give away the fire position (0 = old behaviour)
    jitter: int = 2  # partial windows: centre shifted by up to +-jitter cells (rows and cols)


def hide_regions(rho4: torch.Tensor, valid: torch.Tensor, cfg: HideConfig, gen: torch.Generator | None = None,
                 footprints: Footprints | None = None) -> torch.Tensor:
    """rho4 [B,H,W] (log L4 minus local median, measured), valid [B,H,W] bool (common mask) -> hidden [B,H,W] bool, subset of valid.

    Footprints (MODIS): a hidden cell hides its whole native pixel, since every cell of a pixel carries the same measurement.
    """
    b, h, w = rho4.shape
    dev = rho4.device
    z = torch.zeros(b, h * w, device=dev)
    if cfg.n_top > 0:
        score = torch.where(valid & (rho4 >= cfg.rho_min), rho4, torch.full_like(rho4, float("-inf"))).flatten(1)
        vals, idx = score.topk(min(cfg.n_top, h * w), dim=1)
        if cfg.center_jitter > 0:
            yy, xx = idx // w, idx % w
            j = cfg.center_jitter
            yy = (yy + torch.randint(-j, j + 1, yy.shape, device=dev, generator=gen)).clamp(0, h - 1)
            xx = (xx + torch.randint(-j, j + 1, xx.shape, device=dev, generator=gen)).clamp(0, w - 1)
            idx = yy * w + xx
        z.scatter_(1, idx, torch.isfinite(vals).float())
    for _ in range(cfg.n_random):
        r = torch.randint(0, h * w, (b, 1), device=dev, generator=gen)
        z.scatter_(1, r, torch.ones(b, 1, device=dev))
    win = F.max_pool2d(z.view(b, 1, h, w), cfg.window, 1, cfg.window // 2)[:, 0] > 0
    if cfg.partial_p > 0:
        zf = z.view(b, 1, h, w)
        jy = torch.randint(-cfg.jitter, cfg.jitter + 1, (b,), device=dev, generator=gen)
        jx = torch.randint(-cfg.jitter, cfg.jitter + 1, (b,), device=dev, generator=gen)
        yy = (torch.arange(h, device=dev)[None, :, None] - jy[:, None, None]) % h
        xx = (torch.arange(w, device=dev)[None, None, :] - jx[:, None, None]) % w
        zs = zf[:, 0][torch.arange(b, device=dev)[:, None, None], yy, xx]
        part = F.max_pool2d(zs[:, None], cfg.partial_window, 1, cfg.partial_window // 2)[:, 0] > 0
        use_part = torch.rand(b, device=dev, generator=gen) < cfg.partial_p
        win = torch.where(use_part[:, None, None], part, win)
    on = torch.rand(b, device=dev, generator=gen) < cfg.p_tile
    hidden = win & valid & on[:, None, None]
    if footprints is not None:
        hidden = footprints.any(hidden) & valid
    return hidden
