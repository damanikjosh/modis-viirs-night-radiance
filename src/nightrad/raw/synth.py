"""Synthetic tiles with KNOWN truth, in the schema.py contract.

Fine scene (375 m cells, VIIRS band): smooth background + sparse sub-cell fires (random temperature and area). Rendered to
 VIIRS: the cell, multiplicative noise, I4 saturation at 367 K.
 MODIS: native pixels = irregular groups of cells growing toward the swath edge (alongscan x4.8 at the edge), radiance computed
        in the MODIS band from the same fire temperatures (so the 800 K excess factor is a real approximation), mean over the
        footprint, noise per pixel, b22 -> b21 switch when b22 would exceed its ceiling.
 Masks: water (shared), clouds (per sensor, with a contaminated 1-cell ring), partial swath overlap, bow-tie edge cells.
Everything truth-related is returned as `eval_*` (never an input)."""
from __future__ import annotations

import numpy as np
import torch

from nightrad.raw import planck
from nightrad.raw.schema import densify_pid, neighbour_table

LAM = planck.LAMBDA
CELL_KM = 0.375


def _B(T, lam):
    return float(planck.C1) / (lam**5 * np.expm1(planck.C2 / (lam * np.asarray(T, np.float64))))


def _smooth(rng, h, w, scale):
    n = rng.standard_normal((h, w))
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None]
    f = np.exp(-0.5 * (fy**2 + fx**2) * (2 * np.pi * scale) ** 2)
    out = np.fft.ifft2(np.fft.fft2(n) * f).real
    out = out - out.mean()  # a heavy low-pass leaves mostly DC: remove it before normalising
    return out / (out.std() + 1e-9)


def _footprints(rng, h, w, u0):
    """pid [H,W] int (dense), complete [H,W] bool, psize [2,H,W] (km), scan u [W]. Irregular groups, growth toward the swath edge."""
    u = np.clip(u0 + 0.0 * np.arange(w) + (np.arange(w) - w / 2) * 0.0008, -1, 1)  # scan position across the tile
    f_scan = 1 + 3.8 * np.abs(u) ** 2.5  # along-scan size factor, 1 (nadir) .. 4.8 (edge)
    f_track = 1 + 1.0 * np.abs(u) ** 3
    pid = np.full((h, w), -1, np.int64)
    raw = np.full((h, w), -1, np.int64)  # native ids line*1354 + sample (grid adjacency)
    line = 0
    complete = np.zeros((h, w), bool)
    psa = np.zeros((h, w), np.float32)
    psc = np.zeros((h, w), np.float32)
    r = -int(rng.integers(0, 3))
    g = 0
    while r < h:
        hh = max(2, int(round(2.67 * f_track[min(max(r, 0), w - 1)] + rng.normal(0, 0.4))))
        c = -int(rng.integers(0, 3))
        samp = 100
        while c < w:
            ww = max(2, int(round(2.67 * f_scan[min(max(c, 0), w - 1)] + rng.normal(0, 0.5))))
            r0, r1, c0, c1 = max(r, 0), min(r + hh, h), max(c, 0), min(c + ww, w)
            pid[r0:r1, c0:c1] = g
            raw[r0:r1, c0:c1] = line * 1354 + samp
            full = r >= 0 and r + hh <= h and c >= 0 and c + ww <= w
            complete[r0:r1, c0:c1] = full
            psa[r0:r1, c0:c1] = ww * CELL_KM
            psc[r0:r1, c0:c1] = hh * CELL_KM
            g += 1
            samp += 1
            c += ww
        line += 1
        r += hh
    return pid, complete, np.stack([psa, psc]), u, raw


def make_tile(seed: int, h: int = 96, w: int = 96, fire_rate: float = 6.0) -> dict:
    rng = np.random.default_rng(seed)
    day = rng.random() < 0.5
    # ---- fine truth ----
    T11 = 295 + 5 * _smooth(rng, h, w, 18) + 1.5 * _smooth(rng, h, w, 4)
    dT = (4.0 + 1.5 * _smooth(rng, h, w, 25)) if day else (0.8 + 0.3 * _smooth(rng, h, w, 25))
    T4 = T11 + dT
    B4v, B11v = _B(T4, LAM["viirs_4"]), _B(T11, LAM["viirs_11"])
    B4m, B11m = _B(T4, LAM["modis_4"]), _B(T11, LAM["modis_11"])
    E4v = np.zeros((h, w)); E11v = np.zeros((h, w)); E4m = np.zeros((h, w)); E11m = np.zeros((h, w))
    nf = rng.poisson(fire_rate) if rng.random() < 0.85 else 0
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(nf):
        cy, cx = rng.uniform(3, h - 3), rng.uniform(3, w - 3)
        Tf = rng.uniform(600, 1300)
        p_tot = float(np.exp(rng.uniform(np.log(2e-5), np.log(4e-3))))  # sub-cell area fraction (sum over the cells it touches)
        sig = rng.uniform(0.35, 0.9)
        wgt = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sig**2)
        wgt[(np.abs(yy - cy) > 2.5) | (np.abs(xx - cx) > 2.5)] = 0
        wgt = p_tot * wgt / wgt.sum()
        E4v += wgt * (_B(Tf, LAM["viirs_4"]) - B4v); E11v += wgt * (_B(Tf, LAM["viirs_11"]) - B11v)
        E4m += wgt * (_B(Tf, LAM["modis_4"]) - B4m); E11m += wgt * (_B(Tf, LAM["modis_11"]) - B11m)
    E4v, E11v, E4m, E11m = [np.maximum(a, 0) for a in (E4v, E11v, E4m, E11m)]
    # ---- footprints, swath ----
    u0 = rng.uniform(-0.95, 0.9)
    pid, complete, psize_m, u, raw = _footprints(rng, h, w, u0)
    ncell = np.maximum(np.bincount(pid.ravel(), minlength=pid.max() + 1), 1).astype(np.float64)

    def fpmean(a):
        return (np.bincount(pid.ravel(), a.ravel(), minlength=len(ncell)) / ncell)[pid]

    # ---- coverage (partial overlap) ----
    tilt = rng.uniform(-0.6, 0.6)
    cover_m = (xx + tilt * yy) > rng.uniform(-5, 0.15 * w)
    cover_v = (xx + rng.uniform(-0.6, 0.6) * yy) < rng.uniform(0.85 * w, w + 20)
    # ---- masks ----
    water = _smooth(rng, h, w, 10) > 1.55
    cl_m = _smooth(rng, h, w, 7) > 1.0
    cl_v = np.roll(cl_m, (int(rng.integers(-3, 4)), int(rng.integers(-3, 4))), (0, 1)) | (_smooth(rng, h, w, 6) > 1.8)

    def ring(c):
        from scipy.ndimage import binary_dilation
        return binary_dilation(c, iterations=1) & ~c

    ring_m, ring_v = ring(cl_m), ring(cl_v)
    # ---- render VIIRS ----
    sat_ceiling = float(planck.planck_radiance(367.0, LAM["viirs_4"]))
    L4v = (B4v + E4v) * np.exp(rng.normal(0, 0.02, (h, w)))
    L11v = (B11v + E11v) * np.exp(rng.normal(0, 0.006, (h, w)))
    # ---- render MODIS (band-correct), per native pixel noise ----
    L4m_true = fpmean(B4m + E4m)
    L11m_true = fpmean(B11m + E11m)
    npx = len(ncell)
    n4, n11, n4b = rng.normal(0, 0.012, npx)[pid], rng.normal(0, 0.004, npx)[pid], rng.normal(0, 0.03, npx)[pid]
    b22_ceiling = float(planck.planck_radiance(331.0, LAM["modis_4"]))
    band21 = L4m_true > b22_ceiling
    L4m = L4m_true * np.exp(np.where(band21, n4b, n4))
    L11m = L11m_true * np.exp(n11)
    sat_m = L4m_true > float(planck.planck_radiance(500.0, 3.96))
    L4m = np.minimum(L4m, float(planck.planck_radiance(500.0, 3.96)))
    # cloud: cold radiance and contaminated ring
    for (L4, L11, c, rg, lv4, lv11) in ((L4v, L11v, cl_v, ring_v, "viirs_4", "viirs_11"),):
        L4[c] = _B(rng.uniform(230, 262, c.sum()), LAM[lv4]); L11[c] = _B(rng.uniform(225, 258, c.sum()), LAM[lv11])
        L4[rg] *= 0.7; L11[rg] *= 0.93
    cm = cl_m.copy()
    # MODIS clouds act on native pixels: any cloudy cell clouds the footprint
    cm_f = np.bincount(pid.ravel(), cm.ravel().astype(float), minlength=npx)[pid] > 0
    rm_f = np.bincount(pid.ravel(), ring_m.ravel().astype(float), minlength=npx)[pid] > 0
    L4m = np.where(cm_f, _B(250, LAM["modis_4"]) * np.exp(rng.normal(0, 0.05, (h, w))), np.where(rm_f, L4m * 0.7, L4m))
    L11m = np.where(cm_f, _B(245, LAM["modis_11"]) * np.exp(rng.normal(0, 0.02, (h, w))), np.where(rm_f, L11m * 0.93, L11m))
    sat_v = L4v >= sat_ceiling
    L4v = np.minimum(L4v, sat_ceiling)
    # ---- codes ----
    code_v = np.zeros((h, w), np.uint8); code_m = np.zeros((h, w), np.uint8)
    code_v[water] = 2; code_m[water] = 2
    code_v[cl_v] = 1
    code_m[cm_f] = 1
    code_v[~cover_v] = 4; code_m[~cover_m] = 4
    bow = rng.random(w) < 0.01  # bow-tie deleted columns (VIIRS)
    code_v[:, bow] = 3
    code_m[pid < 0] = 4
    for L, c in ((L4v, code_v), (L11v, code_v), (L4m, code_m), (L11m, code_m)):
        L[c >= 3] = np.nan
    cossza = np.full((h, w), (np.cos(np.deg2rad(rng.uniform(15, 60))) if day else -0.2), np.float32)
    vza_m = np.abs(u)[None].repeat(h, 0) * 55
    vza_v = np.abs(u)[None].repeat(h, 0) * 56
    psize_v = np.stack([CELL_KM * (1 + 1.8 * (vza_v / 56) ** 2), np.full((h, w), CELL_KM) * (1 + 0.6 * (vza_v / 56) ** 2)]).astype(np.float32)
    f32 = lambda a: np.asarray(a, np.float32)
    return {
        "modis_L": f32(np.stack([L4m, L11m])), "viirs_L": f32(np.stack([L4v, L11v])),
        "modis_code": code_m, "viirs_code": code_v,
        "modis_sat": sat_m.astype(np.uint8), "viirs_sat": sat_v.astype(np.uint8),
        "modis_band21": band21.astype(np.uint8), "viirs_band21": np.zeros((h, w), np.uint8),
        "modis_psize": f32(psize_m), "viirs_psize": psize_v, "modis_vza": f32(vza_m), "viirs_vza": f32(vza_v),
        "modis_cossza": cossza, "viirs_cossza": cossza,
        "modis_pid": densify_pid(pid).astype(np.int32), "modis_nbr": neighbour_table(raw, densify_pid(pid)), "modis_complete": complete.astype(np.uint8),
        "eval_E4": f32(E4v), "eval_E11": f32(E11v), "eval_E4_modisband": f32(E4m), "eval_E11_modisband": f32(E11m),
        "eval_B4": f32(B4v), "eval_B11": f32(B11v), "eval_day": np.asarray(day, np.uint8),
    }


class SyntheticTiles(torch.utils.data.Dataset):
    def __init__(self, n: int, seed0: int = 0, h: int = 96, w: int = 96, cache: bool = False):
        self.n, self.seed0, self.h, self.w = n, seed0, h, w
        self._cache = [None] * n if cache else None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._cache is not None and self._cache[i] is not None:
            return self._cache[i]
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in make_tile(self.seed0 + i, self.h, self.w).items()}
        if self._cache is not None:
            self._cache[i] = t
        return t
