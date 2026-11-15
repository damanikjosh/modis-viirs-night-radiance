"""Tile builder: two `Swath`s -> schema-valid 96x96 375 m LAEA pair tiles (docs/tile_schema.md)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from pyproj import Transformer
from scipy.ndimage import binary_dilation, median_filter, uniform_filter
from scipy.spatial import cKDTree

from .readers import VC_CLEAR, VC_NOT_COVERED, VC_OFF_SWATH, Swath

BG_WIN = 21  # background window, native pixels
BG_MIN_FRAC = 0.25
STATIC_KM = 1.0
Z_CLIP = 50.0
N_RANDOM = 20

H = W = 96
CELL_M = 375.0
R_EARTH_M = 6371007.181
ALT_KM = {"modis": 705.0, "viirs": 824.0}
SENSORS = ("modis", "viirs")
SCHEMA_VERSION = 5  # v5: truthful resampling (footprint-covering selection, reason-coded validity, radiance, no fill); v2: thermal t4/t11; v3: quality flags, background z-scores, confidence, sampling weights, region; v4: provenance keys (satellite, times, split, offsets)
MAX_DT_MIN = 30.0  # tiles whose own |dt| (VIIRS - MODIS mean line time over the tile) exceeds this are dropped (audit D5)
FIRE_OFFSET_KM = 12.0  # fire-centred tiles are shifted by U(-12, 12) km in x and y (fire stays >= 6 km inside the 36 km tile; audit D4)
_COARSE = 4  # stride of the coarse coverage lattice
FOOTPRINT_HALF = 0.52  # footprint half-extent in pixel units: 0.5 + 2 % seam tolerance (neighbouring footprints from geolocation differ by ~1 %, which left 0.1-0.4 % of interior cells uncovered with 0.5; 0.52 leaves none on the test pair; far below the geolocation uncertainty)
K_COVER = 12  # candidates examined per cell (nearest pixel centres, deleted ones included); a covering pixel is never farther than half its diagonal


def _xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def shift_centres(cand: np.ndarray, rng: np.random.Generator, max_km: float = FIRE_OFFSET_KM):
    """Fire-centred tile centres -> centres shifted by a random offset in the local tangent plane. cand [N,2] (lat, lon).
    Returns (shifted [N,2] float64, offsets_km [N,2] float32 as (east, north)). Shared by the CPU and CUDA builders (same RNG stream)."""
    off = rng.uniform(-max_km, max_km, size=(len(cand), 2))
    c = np.asarray(cand, np.float64).reshape(-1, 2)
    lat = np.clip(c[:, 0] + off[:, 1] / 111.195, -89.9, 89.9)
    lon = c[:, 1] + off[:, 0] / (111.195 * np.cos(np.deg2rad(c[:, 0])))
    lon = (lon + 180.0) % 360.0 - 180.0
    return np.stack([lat, lon], 1), off.astype(np.float32)


def tile_ok(t: dict, min_overlap: float) -> bool:
    """Acceptance test of a built tile: enough two-swath overlap and |dt| <= MAX_DT_MIN (audit D5)."""
    return bool(t["overlap"].mean() >= min_overlap and abs(float(t["dt_min"])) <= MAX_DT_MIN)


def _proj(lat0, lon0) -> Transformer:
    crs = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +R={R_EARTH_M} +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def scan_angle_deg(vza: np.ndarray, sample: np.ndarray, nsamples: int, alt_km: float) -> np.ndarray:
    """Signed scan angle from view zenith: sin(scan) = R/(R+h) sin(vza); sign = side of the swath centre."""
    r = R_EARTH_M / 1000
    s = np.rad2deg(np.arcsin(np.clip(r / (r + alt_km) * np.sin(np.deg2rad(vza)), -1, 1)))
    return np.where(sample < (nsamples - 1) / 2, -s, s).astype(np.float32)


class _Window:
    """Native pixels near a tile centre, with projected coordinates and local geometry."""

    def __init__(self, sw: Swath, tr: Transformer, lat0, lon0, radius_km=80.0):
        c = _COARSE * 2
        la, lo = sw.lat[::c, ::c], sw.lon[::c, ::c]
        ok = np.isfinite(la)
        d = np.full(la.shape, np.inf)
        d[ok] = np.arccos(np.clip(_xyz(la[ok], lo[ok]) @ _xyz(lat0, lon0), -1, 1)) * R_EARTH_M / 1000
        hit = np.argwhere(d <= radius_km)
        self.empty = len(hit) == 0
        if self.empty:
            return
        pad = c + 2
        L, S = sw.shape
        self.l0 = max(int(hit[:, 0].min()) * c - pad, 0)
        self.l1 = min(int(hit[:, 0].max()) * c + pad + 1, L)
        self.s0 = max(int(hit[:, 1].min()) * c - pad, 0)
        self.s1 = min(int(hit[:, 1].max()) * c + pad + 1, S)
        lat, lon = sw.lat[self.l0:self.l1, self.s0:self.s1], sw.lon[self.l0:self.l1, self.s0:self.s1]
        self.ok = np.isfinite(lat) & np.isfinite(lon)
        x = np.full(lat.shape, np.nan)
        y = np.full(lat.shape, np.nan)
        x[self.ok], y[self.ok] = tr.transform(lon[self.ok].astype(np.float64), lat[self.ok].astype(np.float64))
        self.x, self.y = x, y
        # pixel size from geolocation spacing (central differences, km): axis 1 = along scan, axis 0 = along track
        fx, fy = np.nan_to_num(x, nan=1e9), np.nan_to_num(y, nan=1e9)
        gs = np.hypot(np.gradient(fx, axis=1), np.gradient(fy, axis=1)) / 1000
        # along-track: median over 9 neighbouring line-to-line spacings (the 2x-upsampled VIIRS scan boundary smears over ~3), so spacing across scan boundaries
        # (bow-tie overlap / gap between consecutive scans) does not leak in
        dl = np.hypot(np.diff(fx, axis=0), np.diff(fy, axis=0)) / 1000
        dl = np.concatenate([dl, dl[-1:]])  # one value per line
        gt = median_filter(dl, size=(9, 1), mode="nearest")
        self.psize = np.stack([gs, gt]).astype(np.float32)
        self.psize[:, ~self.ok] = np.nan
        self.psize[self.psize > 50] = np.nan  # gradient next to invalid neighbours
        # footprint = parallelogram spanned by the along-scan vector a (central difference of neighbouring centres, m) and the
        # along-track vector b (direction: 9-line median of the line-to-line displacement; length: gt), so neighbouring
        # footprints tile the ground even where the scan grid is skewed
        gx, gy = np.gradient(fx, axis=1), np.gradient(fy, axis=1)
        self.ax, self.ay = gx, gy
        dvx, dvy = np.diff(fx, axis=0), np.diff(fy, axis=0)
        dvx, dvy = np.concatenate([dvx, dvx[-1:]]), np.concatenate([dvy, dvy[-1:]])
        mx, my = median_filter(dvx, size=(9, 1), mode="nearest"), median_filter(dvy, size=(9, 1), mode="nearest")
        nb = np.maximum(np.hypot(mx, my), 1e-9)
        self.bx, self.by = mx / nb * gt * 1000, my / nb * gt * 1000


class StaticSources:
    """FIRMS static / volcano / offshore detection locations (scripts/static_sources.py CSV: lat, lon, ...)."""

    def __init__(self, lat, lon):
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        self.n = len(lat)
        self.tree = cKDTree(_xyz(lat, lon)) if self.n else None
        self.lat, self.lon = lat, lon

    @classmethod
    def from_csv(cls, path):
        import pandas as pd

        d = pd.read_csv(path)
        return cls(d.lat.values, d.lon.values)

    def projected_near(self, tr: Transformer, lat0, lon0, radius_km=120.0) -> np.ndarray:
        """Projected (x, y) metres of static points within radius_km of (lat0, lon0), shape [n, 2]."""
        if not self.n:
            return np.zeros((0, 2))
        chord = 2 * np.sin(radius_km / R_EARTH_M * 1000 / 2)
        ii = self.tree.query_ball_point(_xyz(lat0, lon0), chord)
        if not ii:
            return np.zeros((0, 2))
        x, y = tr.transform(self.lon[ii], self.lat[ii])
        return np.stack([x, y], 1)


def region_of(lat: float, lon: float) -> str:
    """Coarse continent from tile-centre lon/lat: AF, AS, EU, NA, SA, OC (bounding-box rules, ordered)."""
    if lon < -25:
        return "SA" if (lat < 12.5 and lon > -92) else "NA"
    if lon >= 110 and lat < -10 or (lon >= 141 and lat < -0.5 and lat > -13) or (lon > 160 and lat < -30):
        return "OC"
    if lat < 36.5 and -20 <= lon <= 52 and not (lon > 34 and lat > 12.5):
        return "AF"
    if -25 <= lon <= 45 and lat >= 36:
        return "EU"
    return "AS"


def native_layers(sw: Swath, win: _Window, static_xy: np.ndarray) -> dict:
    """Per-native-pixel layers over the window: cloud-adjacent, static, missing-thermal and background z-scores.

    Background (MODIS/VIIRS contextual-test style): 21x21 native pixels around each pixel, using only pixels that are
    clear land (cls 3), not static, with finite T4 and T11. Needs >= 25% of the window valid, else bg_ok=0 and z=0.
    z(T4) = (T4 - mean) / max(std, 1 K); z(dT) the same for T4 - T11. z clipped to +-50.
    """
    sl = (slice(win.l0, win.l1), slice(win.s0, win.s1))
    cls = sw.cls[sl]
    cloud = cls == 2
    out = {"cloudadj": binary_dilation(cloud, structure=np.ones((3, 3), bool)) & ~cloud}
    static = np.zeros(cls.shape, bool)
    if len(static_xy):
        tree = cKDTree(static_xy)
        thr = np.maximum(STATIC_KM, 0.5 * np.nanmax(win.psize, axis=0)) * 1000  # within 1 km (or inside the pixel)
        d, _ = tree.query(np.stack([np.nan_to_num(win.x, nan=1e12).ravel(), np.nan_to_num(win.y, nan=1e12).ravel()], 1))
        static = (d.reshape(cls.shape) <= np.nan_to_num(thr, nan=0)) & win.ok
    out["static"] = static
    out["conf"] = None
    if sw.t4 is None:
        return out
    t4, t11 = sw.t4[sl].astype(np.float64), sw.t11[sl].astype(np.float64)
    fin = np.isfinite(t4) & np.isfinite(t11)
    out["missing"] = ~fin
    out["sat"] = sw.sat[sl] if sw.sat is not None else np.zeros(cls.shape, bool)
    V = (cls == 3) & ~static & fin
    frac = uniform_filter(V.astype(np.float64), size=BG_WIN, mode="constant")
    ok = frac >= BG_MIN_FRAC
    out["bg_ok"] = ok
    for name, x in (("t4_bgz", t4 - 290.0), ("dt_bgz", (t4 - t11))):
        xv = np.where(V, x, 0.0)
        m = uniform_filter(xv, size=BG_WIN, mode="constant") / np.maximum(frac, 1e-9)
        v = uniform_filter(xv * xv, size=BG_WIN, mode="constant") / np.maximum(frac, 1e-9) - m * m
        sd = np.sqrt(np.maximum(v, 0))
        z = (x - m) / np.maximum(sd, 1.0)
        out[name] = np.where(ok & np.isfinite(x), np.clip(z, -Z_CLIP, Z_CLIP), 0.0).astype(np.float32)
    return out


def covers(dx, dy, ax, ay, bx, by):
    """Footprint test: does the pixel's footprint, the parallelogram spanned by the along-scan vector (ax, ay) and the
    along-track vector (bx, by) (metres, centred on the pixel), contain the offset (dx, dy) from its centre? i.e. solve
    d = alpha a + beta b and require |alpha|, |beta| <= 1/2. Shared formula of the CPU and CUDA builders."""
    det = ax * by - ay * bx
    with np.errstate(divide="ignore", invalid="ignore"):
        al, be = (dx * by - dy * bx) / det, (ax * dy - ay * dx) / det
    return (np.abs(al) <= FOOTPRINT_HALF) & (np.abs(be) <= FOOTPRINT_HALF)


def choose_covering(dist, cover, valid, vza, prefer_nadir, pid):
    """Pick one native pixel per cell among the K candidates. dist/cover/valid/vza/pid: [N, K] (cover: the candidate is a real,
    non-deleted observation whose footprint contains the cell centre). Rule 3: valid ones first; among them the smallest
    view zenith (prefer_nadir, MODIS) else the nearest (distance rounded to 1 mm); exact ties -> the smaller pixel id, so the
    CPU and CUDA builders agree. Returns (pick [N], has [N])."""
    score = (~valid) * 1e12 + (np.nan_to_num(vza, nan=1e3).astype(np.float64) * 1e6 if prefer_nadir else 0.0) + np.round(dist, 3)
    score = np.where(cover, score, np.inf)
    best = score.min(1, keepdims=True)
    tie = cover & (score == best)
    return np.argmin(np.where(tie, pid, np.iinfo(np.int64).max), axis=1), cover.any(1)


def _sensor_layers(sw: Swath, win: _Window, cx, cy, alt_km, static_xy=None, prefer_nadir=False):
    """Truthful resampling of one sensor onto the tile grid; returns (dict of schema arrays, mean pass time or None).

    Rule (docs/tile_schema.md, v5): a cell takes ONE native pixel's measurement, copied, never blended or filled. Only pixels whose
    rectangular footprint (centre from geolocation, along-scan size = central difference of neighbouring centres, along-track
    size = 9-line median of the line spacing, oriented along the local scan direction) contains the cell centre are candidates;
    bow-tie-deleted pixels are not observations and are skipped (the other scan line's pixel over the same ground is used, if
    any); valid (clear land/fire with finite T4 and T11) candidates come first, then the nearest (MODIS: closest to nadir).
    If every covering pixel is masked (cloud/water/bad/glint) the cell keeps that pixel's reason; it is never filled from a neighbour."""
    f32, u8 = np.float32, np.uint8
    out = {
        "cls": np.zeros((H, W), u8), "frp": np.zeros((H, W), f32),
        "scan": np.zeros((H, W), f32), "psize": np.zeros((2, H, W), f32),
        "vza": np.zeros((H, W), f32), "sza": np.zeros((H, W), f32),
        "pid": np.full((H, W), -1, np.int32),
        "t4": np.zeros((H, W), f32), "t11": np.zeros((H, W), f32),
        "cloudadj": np.zeros((H, W), u8), "static": np.zeros((H, W), u8), "t4_missing": np.zeros((H, W), u8),
        "sat": np.zeros((H, W), u8), "t4_bgz": np.zeros((H, W), f32), "dt_bgz": np.zeros((H, W), f32),
        "bg_ok": np.zeros((H, W), u8), "conf": np.zeros((H, W), f32),
        "raw": np.zeros((H, W), u8), "vclass": np.full((H, W), VC_OFF_SWATH, u8), "ncover": np.zeros((H, W), u8),
        "rad4": np.zeros((H, W), f32), "rad11": np.zeros((H, W), f32), "t4_band": np.zeros((H, W), u8),
    }
    if win.empty:
        return out, None
    good = win.ok & np.isfinite(win.psize[0]) & np.isfinite(win.psize[1])
    li, si = np.nonzero(good)
    if len(li) == 0:
        return out, None
    sl = (slice(win.l0, win.l1), slice(win.s0, win.s1))
    tree = cKDTree(np.stack([win.x[li, si], win.y[li, si]], 1))
    K = min(K_COVER, len(li))
    dist, idx = tree.query(np.stack([cx.ravel(), cy.ravel()], 1), k=K)
    dist, idx = dist.reshape(-1, K), idx.reshape(-1, K)
    fin = np.isfinite(dist)
    idx = np.where(fin, idx, 0)
    wl, ws_ = li[idx], si[idx]  # window coordinates [N, K]
    dx, dy = cx.ravel()[:, None] - win.x[wl, ws_], cy.ravel()[:, None] - win.y[wl, ws_]
    inside = fin & covers(dx, dy, win.ax[wl, ws_], win.ay[wl, ws_], win.bx[wl, ws_], win.by[wl, ws_])
    deleted = sw.deleted[sl][wl, ws_] if sw.deleted is not None else np.zeros(idx.shape, bool)
    real = inside & ~deleted
    vc_pix = sw.pixel_vclass()[sl][wl, ws_]
    pick, has = choose_covering(dist, real, vc_pix == VC_CLEAR, sw.vza[sl][wl, ws_], prefer_nadir,
                                (wl + win.l0).astype(np.int64) * sw.shape[1] + ws_ + win.s0)
    rows = np.arange(len(idx))
    wl, ws_ = wl[rows, pick], ws_[rows, pick]
    ins = has.reshape(H, W)
    gl, gs = wl + win.l0, ws_ + win.s0
    sel = lambda a: a[gl, gs].reshape(H, W)
    lsel = lambda a: a[wl, ws_].reshape(H, W)
    vc = np.where(has, vc_pix[rows, pick], np.where(inside.any(1), VC_NOT_COVERED, VC_OFF_SWATH)).reshape(H, W)
    out["vclass"] = vc.astype(u8)
    out["ncover"] = real.sum(1).reshape(H, W).astype(u8)
    out["raw"] = np.where(ins, sel(sw.raw), 0).astype(u8) if sw.raw is not None else out["raw"]
    out["cls"] = np.where(ins, sel(sw.cls), 0).astype(u8)
    out["frp"] = np.where(out["cls"] >= 4, np.log1p(sel(sw.frp)), 0).astype(f32)
    vza = np.nan_to_num(sel(sw.vza))
    out["vza"] = np.where(ins, vza, 0).astype(f32)
    out["sza"] = np.where(ins, np.nan_to_num(sel(sw.sza)), 0).astype(f32)
    out["scan"] = np.where(ins, scan_angle_deg(vza, gs.reshape(H, W), sw.shape[1], alt_km), 0).astype(f32)
    ps = win.psize[:, wl, ws_]
    out["psize"] = np.where(ins[None], ps.reshape(2, H, W), 0).astype(f32)
    out["pid"] = np.where(ins, (gl * sw.shape[1] + gs).reshape(H, W), -1).astype(np.int32)
    if sw.conf is not None:
        out["conf"] = np.where(ins & (out["cls"] >= 4), sel(sw.conf), 0).astype(f32)
    nl = native_layers(sw, win, static_xy if static_xy is not None else np.zeros((0, 2)))
    out["cloudadj"] = (ins & lsel(nl["cloudadj"])).astype(u8)
    out["static"] = (ins & lsel(nl["static"])).astype(u8)
    if sw.t4 is not None:
        t4 = np.where(ins, sel(sw.t4), np.nan)
        t11 = np.where(ins, sel(sw.t11), np.nan)
        ok4, ok11 = np.isfinite(t4), np.isfinite(t11)  # each channel is copied as is when measured; a missing one stays 0 + flagged, never filled
        out["t4"], out["t11"] = np.where(ok4, t4, 0).astype(f32), np.where(ok11, t11, 0).astype(f32)
        out["t4_missing"] = (ins & ~(ok4 & ok11)).astype(u8)
        out["sat"] = (ins & lsel(nl["sat"])).astype(u8)
        out["bg_ok"] = (ins & lsel(nl["bg_ok"])).astype(u8)
        out["t4_bgz"] = np.where(ins, lsel(nl["t4_bgz"]), 0).astype(f32)
        out["dt_bgz"] = np.where(ins, lsel(nl["dt_bgz"]), 0).astype(f32)
        if sw.rad4 is not None:
            out["rad4"] = np.where(ok4, np.nan_to_num(sel(sw.rad4)), 0).astype(f32)
            out["rad11"] = np.where(ok11, np.nan_to_num(sel(sw.rad11)), 0).astype(f32)
        if sw.t4_band is not None:
            out["t4_band"] = np.where(ok4, sel(sw.t4_band), 0).astype(u8)
    t = float(sw.line_time(gl.reshape(H, W)[ins]).mean()) if ins.any() else None
    return out, t


def build_tile(modis: Swath, viirs: Swath, lat0: float, lon0: float, static: StaticSources | None = None,
               random_tile: bool = False) -> dict:
    """One pair tile centred at (lat0, lon0); returns the schema dict of numpy arrays/scalars."""
    tr = _proj(lat0, lon0)
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = (j - (W - 1) / 2) * CELL_M, ((H - 1) / 2 - i) * CELL_M
    sxy = static.projected_near(tr, lat0, lon0) if static is not None else np.zeros((0, 2))
    layers_by, times = {}, {}
    for name, sw in (("modis", modis), ("viirs", viirs)):
        layers_by[name], times[name] = _sensor_layers(sw, _Window(sw, tr, lat0, lon0), cx, cy, ALT_KM[name], sxy,
                                                      prefer_nadir=(name == "modis"))
    return assemble_tile(modis, viirs, layers_by, times, lat0, lon0, random_tile)


def footprint_complete(pid, psize, vclass) -> np.ndarray:
    """[H, W] uint8, constant per native pixel: 1 if the pixel's footprint is fully represented in the tile, i.e. (a) all its cells are
    clear-land valid cells (it owns at least one), (b) none touches the tile border, (c) it owns >= 80 % of the cells its footprint
    area (along-scan x along-track size) predicts (cells lost to an overlapping pixel or to the tile edge make it incomplete)."""
    out = np.zeros(pid.shape, np.uint8)
    m = pid >= 0
    if not m.any():
        return out
    u, inv, cnt = np.unique(pid[m], return_inverse=True, return_counts=True)
    area = np.zeros(len(u))
    area[inv] = psize[0][m].astype(np.float64) * psize[1][m] / (CELL_M / 1000) ** 2  # predicted cells (same for every cell of a pixel)
    border = np.zeros(pid.shape, bool)
    border[0] = border[-1] = border[:, 0] = border[:, -1] = True
    touch = np.zeros(len(u), bool)
    touch[inv[border[m]]] = True
    allvalid = np.bincount(inv, weights=(vclass[m] == VC_CLEAR).astype(float), minlength=len(u)) == cnt
    ok = allvalid & ~touch & (cnt >= 0.8 * area)
    out[m] = ok[inv]
    return out


EVAL_KEYS = ("cls", "frp", "conf", "raw")  # fire-product arrays: stored as eval_<sensor>_<key>, never as model inputs/targets


def export_names(tile: dict) -> dict:
    """Rename the fire-product arrays of a validated tile for disk: {s}_cls -> eval_{s}_cls (also frp, conf, raw)."""
    out = {}
    for k, v in tile.items():
        hit = [(s, e) for s in SENSORS for e in EVAL_KEYS if k == f"{s}_{e}"]
        out[f"eval_{hit[0][0]}_{hit[0][1]}" if hit else k] = v
    return out


def assemble_tile(modis: Swath, viirs: Swath, layers_by: dict, times: dict, lat0: float, lon0: float,
                  random_tile: bool = False) -> dict:
    """Per-sensor layer dicts + pass times -> the schema tile dict (shared by the CPU and the torch builders)."""
    tile = {}
    for name in ("modis", "viirs"):
        layers = layers_by[name]
        for k, v in layers.items():
            tile[f"{name}_{k}"] = v
        tile[f"{name}_valid"] = (layers["vclass"] == VC_CLEAR).astype(np.uint8)
        tile[f"{name}_cossza"] = np.where(layers["pid"] >= 0, np.cos(np.deg2rad(layers["sza"].astype(np.float64))), 0).astype(np.float32)
        tile[f"{name}_footprint_complete"] = footprint_complete(layers["pid"], layers["psize"], layers["vclass"])
    tile["has_thermal"] = np.uint8(modis.t4 is not None and viirs.t4 is not None)
    tile["schema_version"] = np.int32(SCHEMA_VERSION)
    tile["random_tile"] = np.uint8(random_tile)
    tile["sample_weight"] = np.float32(1.0)
    tile["region"] = np.array(region_of(lat0, lon0))
    tile["overlap"] = ((tile["modis_pid"] >= 0) & (tile["viirs_pid"] >= 0)).astype(np.uint8)
    tm = times["modis"] if times["modis"] is not None else float(modis.line_time(modis.shape[0] // 2))
    tv = times["viirs"] if times["viirs"] is not None else float(viirs.line_time(viirs.shape[0] // 2))
    day0 = datetime.fromtimestamp(tm, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    ep = day0.timestamp()
    tile["modis_time"] = np.float32((tm - ep) / 60)
    tile["viirs_time"] = np.float32((tv - ep) / 60)
    tile["dt_min"] = np.float32((tv - tm) / 60)
    sza = tile["modis_sza"][tile["modis_pid"] >= 0]
    tile["daynight"] = np.uint8(1 if (sza.size and np.median(sza) > 90) else 0)
    # v4 provenance (additive): who/when/where, plus the split rule's verdict (nightrad.data.splits)
    from .splits import split_of

    tile["satellite"] = np.array(viirs.platform)
    tile["pair_id"] = np.array("")  # set by the pipeline from the pair manifest
    tile["year"], tile["month"], tile["day"] = np.int16(day0.year), np.int8(day0.month), np.int8(day0.day)
    tile["tile_lat"], tile["tile_lon"] = np.float64(lat0), np.float64(lon0)
    tile["modis_utc"], tile["viirs_utc"] = np.float64(tm), np.float64(tv)  # mean pass time over the tile, epoch seconds UTC
    sp, ho = split_of(float(lat0), float(lon0), str(tile["region"]), day0.year, day0.month)
    tile["split"], tile["holdout"] = np.array(sp), np.array(ho)
    tile["fire_offset_km"] = np.zeros(2, np.float32)  # (east, north) shift of a fire-centred tile centre vs the fire pixel; 0 for random tiles
    tile["meta"] = np.array(json.dumps({
        "modis_granule": modis.granule, "viirs_granule": viirs.granule, "sensor": viirs.platform,
        "lat": float(lat0), "lon": float(lon0), "date": day0.date().isoformat(), "epoch_utc": day0.isoformat(),
    }))
    return tile


_U8 = ("cloudadj", "static", "t4_missing", "sat", "bg_ok")
_F32 = ("t4_bgz", "dt_bgz", "conf")


def validate_tile(t: dict) -> None:
    """Assert the tile obeys docs/tile_schema.md (raises AssertionError)."""
    for s in SENSORS:
        for k, shp, dt in [("cls", (H, W), np.uint8), ("frp", (H, W), np.float32), ("valid", (H, W), np.uint8),
                           ("scan", (H, W), np.float32), ("psize", (2, H, W), np.float32),
                           ("vza", (H, W), np.float32), ("sza", (H, W), np.float32), ("pid", (H, W), np.int32),
                           ("t4", (H, W), np.float32), ("t11", (H, W), np.float32)] + \
                          [(k, (H, W), np.uint8) for k in _U8] + [(k, (H, W), np.float32) for k in _F32]:
            a = t[f"{s}_{k}"]
            assert a.shape == shp and a.dtype == dt, f"{s}_{k}: {a.shape} {a.dtype}"
        assert np.asarray(t[f"{s}_time"]).shape == ()
        c, pid = t[f"{s}_cls"], t[f"{s}_pid"]
        assert c.max() <= 6
        if f"{s}_vclass" in t:  # v5: reason-coded validity
            vc = t[f"{s}_vclass"]
            assert vc.shape == (H, W) and vc.dtype == np.uint8 and vc.max() <= 6
            assert np.array_equal(t[f"{s}_valid"], (vc == VC_CLEAR).astype(np.uint8)), "valid must be vclass == clear land"
            assert ((pid < 0) == (vc <= VC_NOT_COVERED)).all(), "cell without pixel must be off-swath/not covered, and only those"
            assert ((vc != VC_CLEAR) | ((c >= 3) & (c <= 6))).all(), "clear-land cell must carry a product class 3..6"
            for k, dt in (("raw", np.uint8), ("ncover", np.uint8), ("cossza", np.float32), ("footprint_complete", np.uint8), ("rad4", np.float32), ("rad11", np.float32), ("t4_band", np.uint8)):
                a = t[f"{s}_{k}"]
                assert a.shape == (H, W) and a.dtype == dt, f"{s}_{k}: {a.shape} {a.dtype}"
                assert (a[pid < 0] == 0).all(), f"{s}_{k} must be 0 outside any footprint"
            for k in ("rad4", "rad11"):
                assert np.isfinite(t[f"{s}_{k}"]).all() and (t[f"{s}_{k}"] >= 0).all()
            assert ((t[f"{s}_ncover"] >= 1) == (pid >= 0)).all()
        else:
            assert np.array_equal(t[f"{s}_valid"], ((c >= 3) & (c <= 6)).astype(np.uint8))
        assert (t[f"{s}_frp"][c < 4] == 0).all() and (t[f"{s}_frp"] >= 0).all()
        assert ((pid < 0) <= (c == 0)).all(), "cell without pixel must be cls 0"
        assert (np.abs(t[f"{s}_scan"]) <= 57.0).all()  # VIIRS edge cells reach 56.55 deg (pair P0089); was 56.5
        for k in ("t4", "t11"):
            a = t[f"{s}_{k}"]
            assert np.isfinite(a).all() and (a[pid < 0] == 0).all()
            assert ((a == 0) | ((a >= 150) & (a <= 600))).all(), f"{s}_{k} outside 150..600 K"
            if not t["has_thermal"]:
                assert (a == 0).all()
        for k in _U8:
            a = t[f"{s}_{k}"]
            assert a.max() <= 1 and (a[pid < 0] == 0).all(), f"{s}_{k} must be 0/1 and 0 outside the swath"
        assert (t[f"{s}_cloudadj"][c == 2] == 0).all(), "cloudadj excludes cloud cells themselves"
        cf = t[f"{s}_conf"]
        assert (cf >= 0).all() and (cf <= 1).all() and (cf[c < 4] == 0).all()
        for k in ("t4_bgz", "dt_bgz"):
            a = t[f"{s}_{k}"]
            assert np.isfinite(a).all() and (np.abs(a) <= Z_CLIP).all() and (a[t[f"{s}_bg_ok"] == 0] == 0).all()
        if not t["has_thermal"]:
            assert not (t[f"{s}_t4_missing"].any() or t[f"{s}_sat"].any() or t[f"{s}_bg_ok"].any())
        m = pid >= 0  # footprint-pooling contract: one native pixel -> one value of every pixel-level layer
        order = np.argsort(pid[m], kind="stable")
        pm = pid[m][order]
        same = pm[1:] == pm[:-1]
        for k in ("cls", "cloudadj", "static", "sat", "conf"):
            v = t[f"{s}_{k}"][m][order]
            assert (v[1:][same] == v[:-1][same]).all(), f"{s}_{k} not constant per native pixel"
    assert t["overlap"].shape == (H, W) and t["overlap"].dtype == np.uint8
    assert np.array_equal(t["overlap"], ((t["modis_pid"] >= 0) & (t["viirs_pid"] >= 0)).astype(np.uint8))
    assert set(t) >= {"dt_min", "daynight", "overlap", "meta", "modis_time", "viirs_time", "has_thermal",
                      "schema_version", "random_tile", "sample_weight", "region", "satellite", "pair_id", "year", "month",
                      "day", "tile_lat", "tile_lon", "modis_utc", "viirs_utc", "split", "holdout", "fire_offset_km"}
    assert str(t["satellite"]) == "none" or abs(float(t["dt_min"])) <= MAX_DT_MIN + 1e-3 or int(t["random_tile"]) < 0, "tile |dt| above MAX_DT_MIN"  # "none": MODIS-only episode tile (no VIIRS exists), dt_min is NaN
    assert str(t["satellite"]) in {"snpp", "noaa20", "noaa21", "none"} and str(t["split"]) in {"train", "val", "test", "none"}
    assert np.asarray(t["fire_offset_km"]).shape == (2,) and np.abs(t["fire_offset_km"]).max() <= FIRE_OFFSET_KM + 1e-3
    assert int(t["schema_version"]) == SCHEMA_VERSION
    assert int(t["random_tile"]) in (0, 1) and float(t["sample_weight"]) >= 0
    assert str(t["region"]) in {"AF", "AS", "EU", "NA", "SA", "OC"}
    json.loads(str(t["meta"]))


def _coarse_tree(sw: Swath):
    la, lo = sw.lat[::_COARSE, ::_COARSE], sw.lon[::_COARSE, ::_COARSE]
    ok = np.isfinite(la)
    return cKDTree(_xyz(la[ok].astype(np.float64), lo[ok].astype(np.float64)))


def _covered(tree: cKDTree, lat, lon, km=20.0) -> bool:
    return bool(tree.query(_xyz(lat, lon))[0] * R_EARTH_M / 1000 <= km)


def _has_fire(t: dict) -> bool:
    return bool((t["modis_cls"] >= 4).any() or (t["viirs_cls"] >= 4).any())


def apply_sample_weights(tiles: list[dict]) -> None:
    """Set sample_weight in place (documented in docs/tile_schema.md).

    Random tiles are uniform over the overlap, so their fire-bearing share q estimates the overlap's fraction of
    fire-bearing tile areas (smoothed: q = (k + 0.5) / (n_random + 1)). Two strata: fire-bearing tiles (any fire cell
    in either sensor; fire-centred and random ones) represent a share q of the overlap, fire-free tiles 1 - q.
    Weight of a tile = (stratum share) * n_tiles / (tiles in stratum), i.e. the inverse selection probability up to a
    per-pair constant, normalised so the weights of a pair average 1. Then the weighted share of fire-bearing tiles
    equals q. Without random tiles or without one of the strata all weights stay 1."""
    rnd = [t for t in tiles if int(t["random_tile"])]
    fb = [t for t in tiles if _has_fire(t)]
    free = [t for t in tiles if not _has_fire(t)]
    if not rnd or not fb or not free:
        return
    q = (sum(_has_fire(t) for t in rnd) + 0.5) / (len(rnd) + 1)
    n = len(tiles)
    for t in fb:
        t["sample_weight"] = np.float32(q * n / len(fb))
    for t in free:
        t["sample_weight"] = np.float32((1 - q) * n / len(free))


def build_pair_tiles(modis: Swath, viirs: Swath, *, max_fire_tiles=40, n_random=N_RANDOM, min_overlap=0.5,
                     min_sep_km=24.0, seed=0, max_attempts=None, static: StaticSources | None = None, centre_filter=None) -> list[dict]:
    """Up to `max_fire_tiles` fire-centred tiles plus `n_random` uniformly random tiles in the overlap.

    centre_filter: optional f(lat, lon arrays) -> bool keep-mask applied to the fire-centred candidates AFTER the random shift,
    so the tile cap is spent only where tiles may be used (the RNG stream is unchanged; None = no filter). Same as in
    `tiles_gpu.build_pair_tiles_gpu`.

    Fire centres = fire pixels of either sensor, shuffled, thinned to >= min_sep_km apart; a tile is kept if the two
    swaths overlap on >= min_overlap of its cells. Random tiles: centres drawn uniformly over VIIRS pixels (approximately
    uniform in area), kept if covered by both swaths with overlap >= min_overlap; they may contain fire."""
    rng = np.random.default_rng(seed)
    trees = {"modis": _coarse_tree(modis), "viirs": _coarse_tree(viirs)}
    cand = []
    for sw in (modis, viirs):
        li, si = np.nonzero(sw.cls >= 4)
        ll = np.stack([sw.lat[li, si], sw.lon[li, si]], 1)
        cand.append(ll[np.isfinite(ll).all(1)])
    cand = np.concatenate(cand)
    cand = cand[rng.permutation(len(cand))]
    cand, offs = shift_centres(cand, rng)  # D4: the fire is not at the tile centre
    if centre_filter is not None and len(cand):
        keep = np.asarray(centre_filter(cand[:, 0], cand[:, 1]), bool)
        cand, offs = cand[keep], offs[keep]
    tiles, chosen, tries = [], [], 0
    max_attempts = max_attempts or 4 * max_fire_tiles
    for ci, (lat, lon) in enumerate(cand):
        if len(tiles) >= max_fire_tiles or tries >= max_attempts:
            break
        v = _xyz(lat, lon)
        if chosen and (np.arccos(np.clip(np.array(chosen) @ v, -1, 1)) * R_EARTH_M / 1000 < min_sep_km).any():
            continue
        if not (_covered(trees["modis"], lat, lon) and _covered(trees["viirs"], lat, lon)):
            continue
        tries += 1
        t = build_tile(modis, viirs, float(lat), float(lon), static)
        if not tile_ok(t, min_overlap):
            continue
        t["fire_offset_km"] = offs[ci]
        chosen.append(v)
        tiles.append(t)
    got, rtries = 0, 0
    while got < n_random and rtries < 30 * max(n_random, 1):
        rtries += 1
        li, si = rng.integers(viirs.shape[0]), rng.integers(viirs.shape[1])
        lat, lon = viirs.lat[li, si], viirs.lon[li, si]
        if not np.isfinite(lat) or not _covered(trees["modis"], lat, lon):
            continue
        t = build_tile(modis, viirs, float(lat), float(lon), static, random_tile=True)
        if not tile_ok(t, min_overlap):
            continue
        tiles.append(t)
        got += 1
    apply_sample_weights(tiles)
    return tiles
