"""VIIRS reading split for the torch tile builder: the CPU only reads the files (raw 750 m geolocation, raw uint16 I4/I5
counts, fire mask); the 2x upsampling, LUT lookups and class mapping run on the device (they are ~3 s/pair on one CPU
core and ~1 GB of float64 temporaries). Results match `readers.read_viirs` + `read_viirs_l1b_full` (see tests)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .readers import (CONF_OF_CLASS, RAW_VIIRS_BOWTIE, VIIRS_FLAG_MIN, VIIRS_I4_NU, VIIRS_SAT_K, planck_rad, VIIRS_SCAN_LINES, Swath, _LUT, _viirs_platform, granule_start,
                      viirs_saturation_from_product)
from .tiles_gpu import GSwath


@dataclass
class ViirsRaw:
    """Everything read from disk for one VIIRS granule (numpy, small: ~0.4 GB vs ~1.5 GB for the dense `Swath`)."""
    name: str
    platform: str
    t0: float
    t1: float
    mask: np.ndarray  # [L, S] uint8 raw fire mask
    fl: np.ndarray  # FP_line / FP_sample / FP_power (empty if no fire)
    fs: np.ndarray
    fp: np.ndarray
    lat: np.ndarray  # [L/2, S/2] float32, NaN where missing (750 m)
    lon: np.ndarray
    vza: np.ndarray
    sza: np.ndarray
    i4: np.ndarray | None = None  # [L, S] uint16 raw counts (L1B); None without L1B
    i5: np.ndarray | None = None
    lut4: np.ndarray | None = None  # [65536] float32 brightness-temperature LUTs
    lut5: np.ndarray | None = None
    sat_prod: np.ndarray | None = None  # [L, S] bool I4 saturation declared by the fire product (readers.viirs_saturation_from_product)
    scale: tuple | None = None  # (I04 scale_factor, add_offset, I05 scale_factor, add_offset): counts -> radiance W m-2 um-1 sr-1


def read_viirs_raw(v14, v03mod, vnp02img=None, duration_s: float = 360.0) -> ViirsRaw:
    import netCDF4

    name = Path(v14).name
    with netCDF4.Dataset(str(v14)) as d14:
        d14.set_auto_maskandscale(False)
        raw = np.asarray(d14.variables["fire mask"][:])
        fl = fs = np.zeros(0, int)
        fp = np.zeros(0, np.float32)
        fp_t4 = np.zeros(0, np.float32)
        if "FP_line" in d14.variables and d14.variables["FP_line"].size:
            fl = np.atleast_1d(d14.variables["FP_line"][:]).astype(int)
            fs = np.atleast_1d(d14.variables["FP_sample"][:]).astype(int)
            fp = np.maximum(np.atleast_1d(d14.variables["FP_power"][:]).astype(np.float32), 0)
            if "FP_T4" in d14.variables:
                fp_t4 = np.atleast_1d(d14.variables["FP_T4"][:]).astype(np.float32)
        sat_prod = (viirs_saturation_from_product(fl, fs, fp_t4, d14.variables["algorithm QA"][:], raw)
                    if "algorithm QA" in d14.variables else None)
    with netCDF4.Dataset(str(v03mod)) as d03:
        g = d03.groups["geolocation_data"]
        get = lambda n: np.ma.filled(g.variables[n][:].astype(np.float32), np.nan)
        geo = [get(n) for n in ("latitude", "longitude", "sensor_zenith", "solar_zenith")]
    t0 = granule_start(v14)
    out = ViirsRaw(name, _viirs_platform(name), t0, t0 + duration_s, raw, fl, fs, fp, *geo, sat_prod=sat_prod)
    if vnp02img is not None:
        with netCDF4.Dataset(str(vnp02img)) as d:
            g = d.groups["observation_data"]
            for band, ak, lk in (("I04", "i4", "lut4"), ("I05", "i5", "lut5")):
                for v in (g.variables[band], g.variables[f"{band}_brightness_temperature_lut"]):
                    v.set_auto_maskandscale(False)
                setattr(out, ak, np.asarray(g.variables[band][:]))
                setattr(out, lk, np.asarray(g.variables[f"{band}_brightness_temperature_lut"][:], np.float32))
            out.scale = tuple(float(g.variables[b].getncattr(a)) for b in ("I04", "I05") for a in ("scale_factor", "add_offset"))
    return out


# ---------------------------------------------------------------- torch ports of readers.upsample2x*
def _up2x(a: torch.Tensor, axis: int) -> torch.Tensor:
    a = a.movedim(axis, 0)
    prev = torch.cat([2 * a[:1] - a[1:2], a[:-1]])
    nxt = torch.cat([a[1:], 2 * a[-1:] - a[-2:-1]])
    out = torch.empty((2 * a.shape[0],) + tuple(a.shape[1:]), dtype=a.dtype, device=a.device)
    out[0::2] = 0.75 * a + 0.25 * prev
    out[1::2] = 0.75 * a + 0.25 * nxt
    return out.movedim(0, axis)


def _up2d(a: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    blocks = a.reshape(n // VIIRS_SCAN_LINES, VIIRS_SCAN_LINES, *a.shape[1:])
    return _up2x(_up2x(blocks, 1).reshape(2 * n, *a.shape[1:]), 1)


def _up_geo(lat: torch.Tensor, lon: torch.Tensor):
    la, lo = torch.deg2rad(lat.double()), torch.deg2rad(lon.double())
    v = torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)])
    v = torch.stack([_up2d(c) for c in v])
    v = v / torch.linalg.norm(v, dim=0)
    return torch.rad2deg(torch.arcsin(torch.clamp(v[2], -1, 1))).float(), torch.rad2deg(torch.atan2(v[1], v[0])).float()


def viirs_gswath(r: ViirsRaw, device) -> GSwath:
    """Dense, on-device `GSwath` of a VIIRS granule from the raw reads."""
    d = torch.device(device)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(d)
    lat, lon = _up_geo(T(r.lat), T(r.lon))
    vza, sza = _up2d(T(r.vza)).float(), _up2d(T(r.sza)).float()
    if tuple(lat.shape) != tuple(r.mask.shape):
        raise ValueError(f"VIIRS 375 m fire mask {r.mask.shape} vs upsampled geolocation {tuple(lat.shape)}")
    cls = T(_LUT)[T(r.mask).long()]
    frp = torch.zeros(cls.shape, dtype=torch.float32, device=d)
    if len(r.fl):
        frp[T(r.fl), T(r.fs)] = T(r.fp)
    raw = T(r.mask)
    t = dict(lat=lat, lon=lon, cls=cls, frp=frp, vza=vza, sza=sza, conf=T(CONF_OF_CLASS)[cls.long()],
             raw=raw, deleted=raw == RAW_VIIRS_BOWTIE)
    if r.i4 is not None:
        def bt(raw, lut):
            raw, lut = T(raw).long(), T(lut)
            b = lut[raw]
            return torch.where((raw >= VIIRS_FLAG_MIN) | ~torch.isfinite(b) | (b <= 0), torch.full_like(b, float("nan")), b)
        t["t4"], t["t11"] = bt(r.i4, r.lut4), bt(r.i5, r.lut5)

        def rad(raw_, sc, of):
            raw_ = T(raw_).long()
            return torch.where(raw_ >= VIIRS_FLAG_MIN, torch.full((), float("nan"), device=d), (raw_.double() * sc + of).float())
        if r.scale is not None:
            t["rad4"], t["rad11"] = rad(r.i4, *r.scale[:2]), rad(r.i5, *r.scale[2:])
        lut4 = T(r.lut4)
        top = torch.where(lut4 > 0, lut4, torch.full_like(lut4, float("nan"))).nan_to_num(nan=-1e9).max()
        t["sat"] = t["t4"] >= top - 1e-3
        if r.sat_prod is not None:  # product-declared saturation (see readers.apply_viirs_saturation)
            hit = T(r.sat_prod) & torch.isfinite(t["t4"])
            t["t4"] = torch.where(hit, torch.full_like(t["t4"], VIIRS_SAT_K), t["t4"])
            t["sat"] = t["sat"] | hit
            if "rad4" in t:
                t["rad4"] = torch.where(hit & torch.isfinite(t["rad4"]), torch.full_like(t["rad4"], float(planck_rad(VIIRS_SAT_K, VIIRS_I4_NU))), t["rad4"])
    meta = Swath("viirs", r.platform, r.name, None, None, None, None, None, None, r.t0, r.t1)
    meta.t4 = np.zeros(0) if r.i4 is not None else None  # thermal marker for `assemble_tile`
    g = GSwath(meta, d, tensors=t)
    meta.lat = np.zeros(g.shape, np.bool_)  # only .shape is read from the metadata swath (zero-copy would be nicer)
    return g
