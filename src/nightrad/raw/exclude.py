"""Held-out episodes: tiles near the validation episode are excluded from train AND val of every run (config-level, committed in configs/raw.json).
Rule = the split builder's: a tile is dropped when its CENTRE is closer than `buffer_km` to the episode box, on a date inside the window."""
from __future__ import annotations

import datetime as dt

import numpy as np

EPISODES = [{"name": "kalimantan-2015", "box": {"lon_min": 111.0, "lon_max": 115.0, "lat_min": -3.0, "lat_max": 1.0},
             "window": ["2015-09-10", "2015-09-27"], "buffer_km": 60.0,
             "why": "story validation episode (Central Kalimantan 14-23 Sep 2015 +-4 days); must stay a true held-out test"}]


def dist_to_box_km(lat, lon, box) -> np.ndarray:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dla = np.maximum(np.maximum(box["lat_min"] - lat, lat - box["lat_max"]), 0) * 111.19
    dlo = np.maximum(np.maximum(box["lon_min"] - lon, lon - box["lon_max"]), 0) * 111.19 * np.cos(np.deg2rad(lat))
    return np.hypot(dla, dlo)


def excluded(lat, lon, year, month, day, episodes=None) -> np.ndarray:
    """Vectorised bool: tile centre (lat, lon) acquired on year-month-day falls in any episode (window and buffer)."""
    lat = np.atleast_1d(np.asarray(lat, float)); out = np.zeros(lat.shape, bool)
    date = np.array([dt.date(int(y), int(m), int(d)).isoformat() for y, m, d in zip(*[np.broadcast_to(np.atleast_1d(a), lat.shape) for a in (year, month, day)])])
    for e in episodes or EPISODES:
        out |= (date >= e["window"][0]) & (date <= e["window"][1]) & (dist_to_box_km(lat, np.broadcast_to(np.atleast_1d(lon), lat.shape), e["box"]) < e["buffer_km"])
    return out
