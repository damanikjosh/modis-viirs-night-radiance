"""Tile geolocation. A tile is a 96x96 grid of 375 m cells on a local spherical Lambert azimuthal equal-area projection centred on (tile_lat, tile_lon):
x = (j - 47.5) * 375 m, y = (47.5 - i) * 375 m, row 0 = north (src/nightrad/data/tiles.py)."""
from __future__ import annotations

import numpy as np

R_EARTH_M = 6371007.181
CELL_M = 375.0
H = W = 96


def laea_forward(lat, lon, lat0, lon0):
    phi, dl, p0 = np.deg2rad(lat), np.deg2rad(lon - lon0), np.deg2rad(lat0)
    cphi = np.cos(phi)
    k = np.sqrt(2.0 / (1.0 + np.sin(p0) * np.sin(phi) + np.cos(p0) * cphi * np.cos(dl)))
    return R_EARTH_M * k * cphi * np.sin(dl), R_EARTH_M * k * (np.cos(p0) * np.sin(phi) - np.sin(p0) * cphi * np.cos(dl))


def laea_inverse(x, y, lat0, lon0):
    p0 = np.deg2rad(lat0)
    rho = np.hypot(x, y)
    c = 2 * np.arcsin(np.clip(rho / (2 * R_EARTH_M), -1, 1))
    rs = np.where(rho == 0, 1.0, rho)
    lat = np.arcsin(np.clip(np.cos(c) * np.sin(p0) + np.where(rho == 0, 0.0, y * np.sin(c) * np.cos(p0) / rs), -1, 1))
    lon = np.deg2rad(lon0) + np.arctan2(x * np.sin(c), rho * np.cos(p0) * np.cos(c) - y * np.sin(p0) * np.sin(c))
    return np.rad2deg(lat), (np.rad2deg(lon) + 180.0) % 360.0 - 180.0


def cell_xy():
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    return (j - (W - 1) / 2) * CELL_M, ((H - 1) / 2 - i) * CELL_M


def tile_cell_lonlat(lat0: float, lon0: float):
    """[H,W] lat, lon of the cell centres of one tile."""
    x, y = cell_xy()
    return laea_inverse(x, y, lat0, lon0)
