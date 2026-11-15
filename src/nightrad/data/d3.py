"""Audit D3: native fire pixels that appear in NO cell of a tile although their centre lies well inside it.

A cell takes the nearest native pixel, so a small pixel (VIIRS) between larger neighbours or a MODIS bow-tie loser can own
zero cells. `fire_pixels` collects a swath's fire pixels once per pair; `count_lost` evaluates one tile."""
from __future__ import annotations

import numpy as np

from .tiles import CELL_M, H, W, _proj


def fire_pixels(lat, lon, cls):
    """(lat, lon, pid) float64/int64 arrays of the fire pixels (cls >= 4) with finite position; pid = line * n_samples + sample."""
    li, si = np.nonzero(np.asarray(cls) >= 4)
    la, lo = np.asarray(lat)[li, si].astype(np.float64), np.asarray(lon)[li, si].astype(np.float64)
    ok = np.isfinite(la) & np.isfinite(lo)
    return la[ok], lo[ok], (li * np.asarray(cls).shape[1] + si)[ok].astype(np.int64)


def count_lost(fp, lat0, lon0, tile_pid, margin: int = 2):
    """-> (n_in, n_lost): fire pixels whose centre falls >= `margin` cells inside the tile on an observed cell (pid >= 0),
    and how many of those pids own no cell of the tile. Also returns n_present: distinct fire pids with a cell in the tile
    is computed by the caller from the tile's own cls/pid."""
    la, lo, pid = fp
    dlon = ((lo - lon0 + 180.0) % 360.0) - 180.0
    box = (np.abs(la - lat0) < 0.3) & (np.abs(dlon) < 0.3 / max(np.cos(np.deg2rad(lat0)), 0.2))
    if not box.any():
        return 0, 0
    x, y = _proj(lat0, lon0).transform(lo[box], la[box])
    col = np.round(np.asarray(x) / CELL_M + (W - 1) / 2).astype(int)
    row = np.round((H - 1) / 2 - np.asarray(y) / CELL_M).astype(int)
    inn = (col >= margin) & (col < W - margin) & (row >= margin) & (row < H - margin)
    if not inn.any():
        return 0, 0
    r, c, p = row[inn], col[inn], pid[box][inn]
    obs = tile_pid[r, c] >= 0
    p = p[obs]
    return int(len(p)), int((~np.isin(p, np.unique(tile_pid[tile_pid >= 0]))).sum())
