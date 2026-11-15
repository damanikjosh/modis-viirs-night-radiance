"""Planck radiance <-> brightness temperature, band conversion, excess conversion. Radiance unit: W m-2 sr-1 um-1.

All functions work on float32/float64 tensors; keep fp32 (radiance spans > 3 decades, autocast must not touch it).
Band centre wavelengths only (no relative spectral response): a stated approximation (see docs in the PR).
"""
from __future__ import annotations

import math

import torch

C1 = 1.191042972e8  # 2 h c^2, W um^4 m-2 sr-1 (NIST CODATA)
C2 = 14387.76877  # h c / k, um K (NIST CODATA)

# band centre wavelengths, um
LAMBDA = {"modis_4": 3.96, "modis_11": 11.0, "viirs_4": 3.74, "viirs_11": 11.45}
# saturation brightness temperatures (K) used for the ceilings; MODIS b22 331 K per fact check, tiles flag 328 K (unresolved, #7)
BANDS_NAME = "centre"  # which set of effective wavelengths is active ("centre" = the original band centres; "data" = fitted to the stored radiance/BT pairs)
# SWEEP-1 factor 'band constants': effective wavelengths implied by the stored data (scripts/fit_band_constants.py on 1,038 night TRAIN tiles, 415k cells per band;
# residual rms 0.05 K VIIRS I4, 0.10 K I5, 0.00 K MODIS b22/b31 instead of 0.93 / 0.42 / 0.50 / 0.02 K with the centres)
LAMBDA_DATA = {"modis_4": 3.9714, "modis_11": 11.0120, "viirs_4": 3.7594, "viirs_11": 11.5868}
LAMBDA_B21_DATA = 3.9916


def use_bands(name: str):
    """Switch the effective wavelengths IN PLACE (call before building a model / view). 'centre' (default) or 'data'."""
    global BANDS_NAME, LAMBDA_B21
    assert name in ("centre", "data")
    LAMBDA.update(LAMBDA_DATA if name == "data" else _CENTRE)
    LAMBDA_B21 = LAMBDA_B21_DATA if name == "data" else 3.96
    BANDS_NAME = name


_CENTRE = dict(LAMBDA)
SAT_BT = {"viirs_4": 367.0, "modis_4": 331.0, "modis_4_b21": 500.0}
LAMBDA_B21 = 3.96
T_FIRE_DEFAULT = 800.0


def planck_radiance(T, lam: float):
    """B(lam, T)."""
    T = torch.as_tensor(T)
    return C1 / (lam**5 * torch.expm1(C2 / (lam * T)))


def brightness_temperature(L, lam: float):
    """Inverse of planck_radiance."""
    L = torch.as_tensor(L)
    return C2 / (lam * torch.log1p(C1 / (lam**5 * L)))


def convert_background(L, lam_src: float, lam_dst: float):
    """Radiance in band src -> radiance in band dst of a blackbody with the same temperature (closed form, no explicit T)."""
    L = torch.as_tensor(L)
    r = lam_src / lam_dst
    x = C1 / (lam_src**5 * L.clamp(min=1e-12))
    return (C1 / lam_dst**5) / torch.expm1(r * torch.log1p(x))


def excess_factor(lam_src: float, lam_dst: float, t_fire: float = T_FIRE_DEFAULT) -> float:
    """Factor converting a fire EXCESS radiance (src band) to dst band: B(dst, Tf) / B(src, Tf). Python float."""
    f = lambda lam: C1 / (lam**5 * math.expm1(C2 / (lam * t_fire)))
    return f(lam_dst) / f(lam_src)


def saturation_radiance() -> dict:
    return {
        "viirs_4": float(planck_radiance(SAT_BT["viirs_4"], LAMBDA["viirs_4"])),
        "modis_4": float(planck_radiance(SAT_BT["modis_4"], LAMBDA["modis_4"])),
        "modis_4_b21": float(planck_radiance(SAT_BT["modis_4_b21"], LAMBDA_B21)),
    }


