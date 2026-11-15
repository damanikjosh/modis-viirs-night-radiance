"""Readers: product files -> `Swath` (dense, schema-class arrays on the native pixel grid).

Specs followed (written from the official docs, not yet checked against real files; see docs/class_mapping.md):
- MODIS C6.1 Active Fire User Guide (MYD14): SDS `fire mask` [lines, 1354] uint8, `FP_line`, `FP_sample`, `FP_power` (MW).
- MYD03: SDS `Latitude`, `Longitude` (float32), `SensorZenith`, `SolarZenith` (int16, scale 0.01), same shape.
- VNP14IMG / VJ114IMG v002 (NetCDF4): variable `fire mask` [lines, 6400], `FP_line`, `FP_sample`, `FP_power`.
- VNP03MOD / VJ103MOD: group `geolocation_data`: latitude, longitude, sensor_zenith, solar_zenith at 750 m
  ([lines/2, 3200]); upsampled 2x here (see `upsample2x`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Raw fire-mask value -> schema class (0 none, 1 water, 2 cloud, 3 clear land, 4/5/6 fire low/nominal/high).
# Identical for MODIS C6.1 and VIIRS 375 m v002 (docs/class_mapping.md).
_LUT = np.zeros(256, np.uint8)
_LUT[3] = 1
_LUT[4] = 2
_LUT[5] = 3
_LUT[7] = 4
_LUT[8] = 5
_LUT[9] = 6


# fire class (4 low / 5 nominal / 6 high) -> confidence in 0..1 (VIIRS l/n/h = 0.3/0.6/0.9; MODIS fallback)
CONF_OF_CLASS = np.zeros(7, np.float32)
CONF_OF_CLASS[4:] = (0.3, 0.6, 0.9)


def map_fire_mask(raw: np.ndarray) -> np.ndarray:
    return _LUT[np.asarray(raw, np.uint8)]


# ---- reason-coded validity (schema v5): what a cell's measurement means, per cell and sensor (docs/tile_schema.md)
VC_OFF_SWATH, VC_NOT_COVERED, VC_CLEAR, VC_CLOUD, VC_WATER, VC_BAD, VC_GLINT = 0, 1, 2, 3, 4, 5, 6
RAW_VIIRS_BOWTIE = 1  # VNP14IMG / VJ114IMG fire mask value 1 = bow-tie deleted (= I04 flag 65533 Bowtie_Deleted, verified on real granules)
# raw fire-mask value -> pixel validity class. Sensor-specific: VIIRS raw 2 is sun glint (an observation, but not usable for the
# fire test); MODIS raw 1/2 are 'not processed' codes. VIIRS raw 1 (bow-tie deleted) is not an observation at all: see Swath.deleted.
_VC = {}
for _s, _glint in (("viirs", VC_GLINT), ("modis", VC_BAD)):
    _t = np.full(256, VC_BAD, np.uint8)
    _t[2], _t[3], _t[4], _t[5] = _glint, VC_WATER, VC_CLOUD, VC_CLEAR
    _t[7:10] = VC_CLEAR
    _VC[_s] = _t
# collapsed schema class -> validity class, used when a Swath carries no raw fire mask (synthetic swaths, old tests)
_VC_OF_CLS = np.array([VC_BAD, VC_WATER, VC_CLOUD, VC_CLEAR, VC_CLEAR, VC_CLEAR, VC_CLEAR], np.uint8)


@dataclass
class Swath:
    sensor: str  # "modis" | "viirs"
    platform: str  # aqua | snpp | noaa20 | noaa21
    granule: str
    lat: np.ndarray  # [L, S] float32, NaN where geolocation missing
    lon: np.ndarray
    cls: np.ndarray  # [L, S] uint8, schema classes
    frp: np.ndarray  # [L, S] float32 MW, 0 except fire pixels
    vza: np.ndarray  # [L, S] float32 degrees
    sza: np.ndarray
    t0: float  # granule start, unix seconds UTC
    t1: float  # granule end
    t4: np.ndarray | None = None  # [L, S] float32 K, 4 um brightness temperature (NaN = missing); from L1B, optional
    t11: np.ndarray | None = None  # [L, S] float32 K, 11 um brightness temperature
    sat: np.ndarray | None = None  # [L, S] bool, 4 um channel saturated (MODIS: band 22 saturated -> band 21 used; VIIRS: I4 at LUT max)
    conf: np.ndarray | None = None  # [L, S] float32 0..1 fire confidence (0 where no fire)
    sat_prod: np.ndarray | None = None  # [L, S] bool, VIIRS only: I4 saturation declared by the fire product (FP_T4 == 367 K, or QA bit 3)
    raw: np.ndarray | None = None  # [L, S] uint8, un-collapsed product fire-mask value (None: derive validity from cls)
    deleted: np.ndarray | None = None  # [L, S] bool, VIIRS bow-tie deleted pixel (raw 1): not an observation, never selected
    rad4: np.ndarray | None = None  # [L, S] float32 W m-2 um-1 sr-1, 4 um radiance of the band behind t4 (NaN = missing)
    rad11: np.ndarray | None = None  # [L, S] float32 W m-2 um-1 sr-1, 11 um radiance
    t4_band: np.ndarray | None = None  # [L, S] uint8: MODIS band used for t4/rad4 (21 or 22); 0 for VIIRS (I4)
    _vc: tuple | None = field(default=None, repr=False, compare=False)  # cache of pixel_vclass(): (input ids, array)

    def pixel_vclass(self) -> np.ndarray:
        """[L, S] uint8 validity class of every native pixel (VC_*; deleted pixels are reported as VC_BAD and excluded
        separately). Clear-land/fire pixels with a missing 4 or 11 um value are VC_BAD: no measurement, no valid cell."""
        key = (id(self.raw), id(self.cls), id(self.t4), id(self.t11))
        if self._vc is None or self._vc[0] != key:
            vc = _VC[self.sensor][self.raw] if self.raw is not None else _VC_OF_CLS[np.minimum(self.cls, 6)]
            if self.t4 is not None:
                vc = np.where((vc == VC_CLEAR) & ~(np.isfinite(self.t4) & np.isfinite(self.t11)), VC_BAD, vc).astype(np.uint8)
            self._vc = (key, vc)
        return self._vc[1]

    @property
    def shape(self):
        return self.lat.shape

    def line_time(self, line) -> np.ndarray:
        """Unix time of a scan line, linear between granule start and end (assumption)."""
        return self.t0 + (self.t1 - self.t0) * (np.asarray(line) + 0.5) / self.shape[0]


_NAME_RE = re.compile(r"\.A(\d{4})(\d{3})\.(\d{4})\.")


def granule_start(path: str | Path) -> float:
    m = _NAME_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot parse acquisition time from {path}")
    d = datetime.strptime(f"{m[1]}{m[2]}{m[3]}", "%Y%j%H%M").replace(tzinfo=timezone.utc)
    return d.timestamp()


def _viirs_platform(name: str) -> str:
    return {"VNP": "snpp", "VJ1": "noaa20", "VJ2": "noaa21"}[name[:3]]


# ---------------------------------------------------------------- MODIS (HDF4)
def _sds(sd, name: str) -> np.ndarray:
    s = sd.select(name)
    a = s.get()
    at = s.attributes()
    out = a.astype(np.float32)
    fill = at.get("_FillValue")
    if fill is not None:
        out[a == fill] = np.nan
    return out * np.float32(at.get("scale_factor", 1.0)) + np.float32(at.get("add_offset", 0.0))


def read_modis(myd14: str | Path, myd03: str | Path, duration_s: float = 300.0) -> Swath:
    from pyhdf.SD import SD, SDC

    f14, f03 = SD(str(myd14), SDC.READ), SD(str(myd03), SDC.READ)
    raw = f14.select("fire mask").get()
    lat, lon = _sds(f03, "Latitude"), _sds(f03, "Longitude")
    vza, sza = _sds(f03, "SensorZenith"), _sds(f03, "SolarZenith")
    if raw.shape != lat.shape:
        raise ValueError(f"MYD14 {raw.shape} and MYD03 {lat.shape} shapes differ: not the same granule")
    frp = np.zeros(raw.shape, np.float32)
    cls = map_fire_mask(raw)
    conf = CONF_OF_CLASS[cls]  # fallback: class -> confidence
    try:
        # granules with no fire pixels have zero-length FP_* arrays; pyhdf raises on .get() of those
        if "FP_line" in f14.datasets() and f14.select("FP_line").info()[2] > 0:
            fl = np.atleast_1d(f14.select("FP_line").get())
            fs = np.atleast_1d(f14.select("FP_sample").get())
            frp[fl, fs] = np.maximum(np.atleast_1d(f14.select("FP_power").get()), 0)
            if "FP_confidence" in f14.datasets():  # MODIS detection confidence 0-100 %
                conf = conf.copy()
                conf[fl, fs] = np.clip(np.atleast_1d(f14.select("FP_confidence").get()) / 100.0, 0, 1)
    finally:
        f14.end()
        f03.end()
    t0 = granule_start(myd14)
    return Swath("modis", "aqua", Path(myd14).name, lat, lon, cls, frp, vza, sza, t0, t0 + duration_s, conf=conf,
                 raw=np.asarray(raw, np.uint8))


# ---------------------------------------------------------------- VIIRS (NetCDF4)
VIIRS_SAT_K = 367.0  # VNP14IMG writes FP_T4 = 367 K for fire pixels whose I4 saturated
VIIRS_QA_SAT_BIT = 3  # 'algorithm QA' bit 3 flags I4 saturation on fire pixels (also set on bow-tie-deleted pixels, whose T4 is NaN)


def viirs_saturation_from_product(fl, fs, fp_t4, qa, raw=None) -> np.ndarray:
    """[L, S] bool: I4 saturated according to the 375 m fire product (FP_T4 >= 366.9 K at the fire pixels, or QA bit 3).
    Saturation fix: QA bit 3 is also set on raw-mask-0 'not processed' pixels, so the bit counts only where the
    fire mask is a fire class (7-9); pass `raw` (the fire mask) to apply it."""
    sat = ((np.asarray(qa, np.uint32) >> VIIRS_QA_SAT_BIT) & 1).astype(bool)
    if raw is not None:
        raw = np.asarray(raw)
        sat &= (raw >= 7) & (raw <= 9)
    if len(fl) and len(fp_t4):
        sat[fl, fs] |= np.asarray(fp_t4, np.float32) >= VIIRS_SAT_K - 0.1
    return sat


def upsample2x(a: np.ndarray, axis: int) -> np.ndarray:
    """Linear 2x upsampling of 750 m values to 375 m cells along one axis.

    A 750 m pixel i covers fine cells 2i and 2i+1; its centre sits between them, so fine cell k lies at coarse
    coordinate (k-0.5)/2: out[2i] = 0.75 a[i] + 0.25 a[i-1], out[2i+1] = 0.75 a[i] + 0.25 a[i+1]. Ends are
    linearly extrapolated (exact for linear fields).
    """
    a = np.moveaxis(a, axis, 0)
    prev = np.concatenate([2 * a[:1] - a[1:2], a[:-1]])
    nxt = np.concatenate([a[1:], 2 * a[-1:] - a[-2:-1]])
    out = np.empty((2 * a.shape[0],) + a.shape[1:], a.dtype)
    out[0::2] = 0.75 * a + 0.25 * prev
    out[1::2] = 0.75 * a + 0.25 * nxt
    return np.moveaxis(out, 0, axis)


VIIRS_SCAN_LINES = 16  # 750 m lines per scan (16 detectors); consecutive scans overlap (bow-tie), so never blend across


def upsample2x_scans(a: np.ndarray, scan_lines: int = VIIRS_SCAN_LINES) -> np.ndarray:
    """Along-track 2x upsampling done independently inside each scan of `scan_lines` coarse lines."""
    n = a.shape[0]
    if n % scan_lines:
        raise ValueError(f"{n} lines is not a multiple of the scan length {scan_lines}")
    blocks = a.reshape(n // scan_lines, scan_lines, *a.shape[1:])
    up = upsample2x(blocks, axis=1)
    return up.reshape(2 * n, *a.shape[1:])


def _up2d(a: np.ndarray, scan_lines: int | None) -> np.ndarray:
    a = upsample2x_scans(a, scan_lines) if scan_lines else upsample2x(a, 0)
    return upsample2x(a, 1)


def upsample2x_geo(lat: np.ndarray, lon: np.ndarray, scan_lines: int | None = VIIRS_SCAN_LINES):
    """Upsample lat/lon via unit vectors (safe across the antimeridian / poles), then renormalise."""
    la, lo = np.deg2rad(lat.astype(np.float64)), np.deg2rad(lon.astype(np.float64))
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    v = np.stack([_up2d(c, scan_lines) for c in v])
    v /= np.linalg.norm(v, axis=0)
    return (np.rad2deg(np.arcsin(np.clip(v[2], -1, 1))).astype(np.float32),
            np.rad2deg(np.arctan2(v[1], v[0])).astype(np.float32))


def upsample2x_scalar(a: np.ndarray, scan_lines: int | None = VIIRS_SCAN_LINES) -> np.ndarray:
    return _up2d(a, scan_lines).astype(np.float32)


def read_viirs(v14: str | Path, v03mod: str | Path, duration_s: float = 360.0) -> Swath:
    import netCDF4

    name = Path(v14).name
    with netCDF4.Dataset(str(v14)) as d14:
        d14.set_auto_maskandscale(False)
        raw = np.asarray(d14.variables["fire mask"][:])
        frp = np.zeros(raw.shape, np.float32)
        fl = fs = np.zeros(0, int)
        fp_t4 = np.zeros(0, np.float32)
        if "FP_line" in d14.variables and d14.variables["FP_line"].size:
            fl = np.atleast_1d(d14.variables["FP_line"][:]).astype(int)
            fs = np.atleast_1d(d14.variables["FP_sample"][:]).astype(int)
            frp[fl, fs] = np.maximum(np.atleast_1d(d14.variables["FP_power"][:]).astype(np.float32), 0)
            if "FP_T4" in d14.variables:
                fp_t4 = np.atleast_1d(d14.variables["FP_T4"][:]).astype(np.float32)
        sat_prod = (viirs_saturation_from_product(fl, fs, fp_t4, d14.variables["algorithm QA"][:], raw)
                    if "algorithm QA" in d14.variables else None)  # None: product without saturation info
    with netCDF4.Dataset(str(v03mod)) as d03:
        g = d03.groups["geolocation_data"]
        get = lambda n: np.ma.filled(g.variables[n][:].astype(np.float32), np.nan)
        lat, lon = upsample2x_geo(get("latitude"), get("longitude"))
        vza, sza = upsample2x_scalar(get("sensor_zenith")), upsample2x_scalar(get("solar_zenith"))
    if raw.shape != lat.shape:
        raise ValueError(f"VIIRS 375 m fire mask {raw.shape} vs upsampled geolocation {lat.shape}")
    t0 = granule_start(v14)
    cls = map_fire_mask(raw)
    # VIIRS 375 m confidence classes low / nominal / high -> 0.3 / 0.6 / 0.9
    return Swath("viirs", _viirs_platform(name), name, lat, lon, cls, frp, vza, sza, t0, t0 + duration_s,
                 conf=CONF_OF_CLASS[cls], sat_prod=sat_prod, raw=raw.astype(np.uint8), deleted=(raw == RAW_VIIRS_BOWTIE))


# ---------------------------------------------------------------- L1B thermal bands -> brightness temperature
# Planck in wavelength form, L in W m-2 um-1 sr-1, lambda in um:  BT = c2 / (lambda ln(1 + c1 / (lambda^5 L)))
# c1 = 2 h c^2 = 1.191042e8 W um^4 m-2 sr-1,  c2 = h c / k = 1.4387752e4 um K,  lambda = 1e4 / wavenumber[cm-1].
PLANCK_C1 = 1.191042e8
PLANCK_C2 = 1.4387752e4
# MODIS effective central wavenumbers (cm-1) from the MODIS L1B documentation (RSR-weighted), bands 21, 22, 31
MODIS_NU = {21: 2505.277, 22: 2518.028, 31: 908.0884}
MODIS_EMISSIVE_INDEX = {21: 1, 22: 2, 31: 10}  # position in EV_1KM_Emissive (bands 20,21,22,23,24,25,27,...,31,...)
MODIS_SAT_K = 328.0  # band 22 saturates at ~331 K; above this fall back to band 21
VIIRS_FLAG_MIN = 65528  # raw counts >= this are flags (Missing_EV 65532, Bowtie_Deleted 65533, Cal_Fail 65534, fill 65535)


def planck_bt(radiance: np.ndarray, wavenumber_cm: float) -> np.ndarray:
    lam = 1e4 / wavenumber_cm
    L = np.asarray(radiance, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = PLANCK_C2 / (lam * np.log1p(PLANCK_C1 / (lam ** 5 * L)))
    return np.where(L > 0, bt, np.nan).astype(np.float32)


def planck_rad(bt, wavenumber_cm: float) -> np.ndarray:
    """Inverse of `planck_bt`: radiance W m-2 um-1 sr-1 of a blackbody at bt (K)."""
    lam = 1e4 / wavenumber_cm
    return (PLANCK_C1 / (lam ** 5 * np.expm1(PLANCK_C2 / (lam * np.asarray(bt, np.float64))))).astype(np.float32)


def modis_l1b_full(ev: np.ndarray, scales, offsets) -> dict:
    """EV_1KM_Emissive [16, L, S] uint16 -> dict(t4, t11, sat, rad4, rad11, band).

    radiance = (raw - radiance_offset) * radiance_scale in W m-2 um-1 sr-1 (the L1B SDS units); raw > 32767 is fill/flag
    (65535 fill, 65533 saturated, ...) -> NaN. T4/rad4 = band 22, falling back to band 21 where band 22 is flagged or its BT
    exceeds MODIS_SAT_K (saturation); `band` records which one (21 or 22; 0 where neither is finite). sat = band 21 used.
    No MODIS temperature-correction (TCS/TCI) is applied (<~0.5 K; effective central wavenumbers only)."""
    def one(band):
        i = MODIS_EMISSIVE_INDEX[band]
        raw = ev[i]
        rad = (raw.astype(np.float64) - offsets[i]) * scales[i]
        bt = planck_bt(rad, MODIS_NU[band])
        bt[raw > 32767] = np.nan
        rad[raw > 32767] = np.nan
        return rad.astype(np.float32), bt

    r22, b22 = one(22)
    r21, b21 = one(21)
    r11, b11 = one(31)
    use21 = ~np.isfinite(b22) | (b22 > MODIS_SAT_K)
    sat22 = (ev[MODIS_EMISSIVE_INDEX[22]] == 65533) | (b22 > MODIS_SAT_K)  # F3 fix: saturated flag or BT above ceiling; not merely invalid
    band = np.where(use21, np.where(np.isfinite(b21), 21, 0), 22).astype(np.uint8)
    return dict(t4=np.where(use21, b21, b22).astype(np.float32), t11=b11, sat=use21 & np.isfinite(b21) & sat22,
                rad4=np.where(use21, r21, r22).astype(np.float32), rad11=r11, band=band)


def modis_l1b_to_bt(ev: np.ndarray, scales, offsets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EV_1KM_Emissive -> (T4, T11, sat), BT in K (see `modis_l1b_full`; sat = band 22 saturated, band 21 used)."""
    d = modis_l1b_full(ev, scales, offsets)
    return d["t4"], d["t11"], d["sat"]


def _modis_ev(myd021km):
    from pyhdf.SD import SD, SDC

    sd = SD(str(myd021km), SDC.READ)
    try:
        sds = sd.select("EV_1KM_Emissive")
        at = sds.attributes()
        return sds.get(), np.asarray(at["radiance_scales"], np.float64), np.asarray(at["radiance_offsets"], np.float64)
    finally:
        sd.end()


def read_modis_l1b_full(myd021km: str | Path) -> dict:
    """dict(t4, t11, sat, rad4, rad11, band) on the 1 km grid; feed to `attach_l1b(sw, **d)`."""
    return modis_l1b_full(*_modis_ev(myd021km))


def viirs_bt_from_lut(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """raw uint16 counts -> K via the product's brightness-temperature LUT; flags/fill/invalid LUT entries -> NaN."""
    bt = np.asarray(lut, np.float32)[raw]
    bt[(raw >= VIIRS_FLAG_MIN) | ~np.isfinite(bt) | (bt <= 0)] = np.nan
    return bt


def read_viirs_l1b(vnp02img: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T4 from I04, T11 from I05, sat) [lines, 6400] on the VNP14IMG 375 m grid, NaN where missing;
    sat = I04 at the top of its LUT (saturated).

    Uses the files' own `I04/I05_brightness_temperature_lut` (65536 entries; I4 tops out at 361.8 K for
    VNP02IMG v2, i.e. saturated fire pixels read as that value). Fill value of the LUT (-999.9) -> NaN.
    """
    import netCDF4

    with netCDF4.Dataset(str(vnp02img)) as d:
        g = d.groups["observation_data"]
        out = []
        for band in ("I04", "I05"):
            v = g.variables[band]
            v.set_auto_maskandscale(False)
            lut = g.variables[f"{band}_brightness_temperature_lut"]
            lut.set_auto_maskandscale(False)
            bt = viirs_bt_from_lut(np.asarray(v[:]), np.asarray(lut[:]))
            out.append(bt)
            if band == "I04":
                top = np.nanmax(np.where(np.asarray(lut[:]) > 0, np.asarray(lut[:]), np.nan))
                sat = bt >= top - 1e-3
    return out[0], out[1], sat


VIIRS_I4_NU = 1e4 / 3.74  # nominal I4 band centre 3.74 um (VIIRS SDR user guide; not re-fetched): used only for the radiance of saturated pixels


def viirs_radiance(raw: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """raw uint16 counts -> radiance W m-2 um-1 sr-1 (file attrs scale_factor/add_offset); flags/fill -> NaN."""
    rad = (np.asarray(raw, np.float64) * scale + offset).astype(np.float32)
    rad[np.asarray(raw) >= VIIRS_FLAG_MIN] = np.nan
    return rad


def read_viirs_l1b_full(vnp02img: str | Path) -> dict:
    """dict(t4, t11, sat, rad4, rad11) on the VNP14IMG 375 m grid. t4/t11/sat as `read_viirs_l1b`; rad4/rad11 are the L1B
    radiances (counts * scale_factor + add_offset, units W m-2 um-1 sr-1 from the file)."""
    import netCDF4

    t4, t11, sat = read_viirs_l1b(vnp02img)
    with netCDF4.Dataset(str(vnp02img)) as d:
        g = d.groups["observation_data"]
        rad = []
        for band in ("I04", "I05"):
            v = g.variables[band]
            v.set_auto_maskandscale(False)
            rad.append(viirs_radiance(np.asarray(v[:]), float(v.getncattr("scale_factor")), float(v.getncattr("add_offset"))))
    return dict(t4=t4, t11=t11, sat=sat, rad4=rad[0], rad11=rad[1])


def attach_l1b(sw: Swath, t4: np.ndarray, t11: np.ndarray, sat: np.ndarray | None = None, rad4: np.ndarray | None = None,
               rad11: np.ndarray | None = None, band: np.ndarray | None = None) -> Swath:
    for a in (t4, t11, rad4, rad11, band):
        if a is not None and a.shape != sw.shape:
            raise ValueError(f"L1B grid {a.shape} != fire-product grid {sw.shape}")
    sat = np.zeros(sw.shape, bool) if sat is None else sat
    if sw.sat_prod is not None:  # VIIRS: L1B I4 counts of saturated pixels read 356-358 K, or fold over to < 340 K
        t4, sat = apply_viirs_saturation(t4, sat, sw.sat_prod)
        if rad4 is not None:  # keep rad4 consistent with t4 = 367 K: a lower bound, flagged by sat
            hit = sw.sat_prod & np.isfinite(rad4)
            rad4 = np.where(hit, planck_rad(VIIRS_SAT_K, VIIRS_I4_NU), rad4).astype(np.float32)
    sw.t4, sw.t11, sw.sat = t4, t11, sat
    sw.rad4, sw.rad11, sw.t4_band = rad4, rad11, band
    sw._vc = None
    return sw


def apply_viirs_saturation(t4: np.ndarray, sat: np.ndarray, sat_prod: np.ndarray):
    """Saturated I4 pixels (product-declared, with a finite L1B reading): sat = True, T4 = 367 K (the product's own value,
    a lower bound). The L1B LUT never reaches its top for them, so `sat` from the LUT alone stays all False."""
    hit = sat_prod & np.isfinite(t4)
    return np.where(hit, np.float32(VIIRS_SAT_K), t4).astype(np.float32), sat | hit
