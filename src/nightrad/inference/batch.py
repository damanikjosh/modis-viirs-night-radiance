"""Batch path: one MODIS night granule (MYD021KM + MYD03, plus the MYD14 cloud/water/quality classes) -> flagged VIIRS-like 375 m cells, deterministic.

granule -> Swath (data/modis_l1b_only.read_modis_production) -> 96x96 tiles on a fixed lon/lat lattice (Grid; only cores that contain land pixels of the granule and, with
tile_rule modis_firelevel, lie near a MODIS fire-level pixel) -> the episode tiler (tiles_gpu.build_tiles_gpu, MODIS only: VIIRS side empty) -> adapt_v3_tile
-> inference.infer.predict(modis_only=True) -> flags = the documented rule (nightrad.raw.rules.detect) on the predicted VIIRS-view radiance.
Each flagged cell is kept once: by the tile that owns it, at least `margin` cells away from that tile's border."""
from __future__ import annotations

import numpy as np
import torch

from nightrad.inference.tilegeo import tile_cell_lonlat
from nightrad.inference.infer import predict
from nightrad.raw import planck

M_PER_DEG = 111194.9  # R = 6371007.181 m
LINES_PER_GRANULE = 2030  # time of a cell = granule start + 300 s * line / 2030 (5-minute granule, 203 scans of 10 lines)
S_STRIDE = 1354
MARGIN = 12  # cells cropped from each side of every 96x96 tile (see Grid)


class Grid:
    """Ownership grid. Every tile is predicted as 96x96 cells, but a tile OWNS only its central core of K = 96 - 2*margin cells per side
    (core = K*375 m); tile centres lie on a fixed lon/lat lattice of core-sized cells, so tiles overlap by 2*margin cells and each output
    cell is owned (kept) by exactly one tile, at least `margin` cells away from that tile's border (tile-edge artefact fix).
    margin=0 gives non-overlapping tiles. shift=(sr, sc) in core units moves the lattice (diagnostic: grid-shift test)."""

    def __init__(self, margin=MARGIN, shift=(0.0, 0.0)):
        assert 0 <= margin < 48
        self.margin, self.k = margin, 96 - 2 * margin
        self.core_m = self.k * 375.0
        self.dlat = self.core_m / M_PER_DEG
        self.sr, self.sc = shift

    def row_of(self, lat):
        return np.floor(np.asarray(lat, np.float64) / self.dlat + self.sr).astype(np.int64)

    def dlon_of_row(self, r):
        lat_c = ((np.asarray(r, np.float64) - self.sr) + 0.5) * self.dlat
        return self.core_m / (M_PER_DEG * np.cos(np.deg2rad(lat_c)))

    def col_of(self, lat, lon):
        r = self.row_of(lat)
        return np.floor(np.asarray(lon, np.float64) / self.dlon_of_row(r) + self.sc).astype(np.int64), r

    def tile_centre(self, r, c):
        return ((r - self.sr) + 0.5) * self.dlat, ((c - self.sc) + 0.5) * self.dlon_of_row(r)


def land_tiles(sw, bbox, grid):
    """Grid cells (row, col) holding >= 1 land pixel of the granule inside bbox (W,S,E,N). Land = MYD03 Land/SeaMask 1 or 2."""
    m = np.isin(sw.landsea, (1, 2)) & np.isfinite(sw.lat) & np.isfinite(sw.lon)
    lat, lon = sw.lat[m].astype(np.float64), sw.lon[m].astype(np.float64)
    keep = (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])
    lat, lon = lat[keep], lon[keep]
    if not len(lat):
        return []
    c, r = grid.col_of(lat, lon)
    return sorted({(int(a), int(b)) for a, b in zip(r, c)})


R_EARTH_KM = 6371.007
TILE_RULES = {"all_land": None, "modis_firelevel": 12, "modis_firelevel_nbr8": 96}  # rule -> radius in 375 m cells (12 = the context margin, 4.5 km; 96 = one 36 km tile)


def _unit_xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * R_EARTH_KM


def modis_firelevel_pixels(sw, device):
    """[L,S] bool: native MODIS pixels flagged by the documented MODIS-alone RADIANCE rule (nightrad.raw.rules.detect on the pixel's own 4/11 um radiance, night, valid = clear
    land by the same validity classes the tiler uses; window spacing 3 pixels = the 8 tile cells of the tile version). NOT the fire product: MYD14 fire classes are not used."""
    import torch as _t

    from nightrad.data.readers import VC_CLEAR
    from nightrad.raw.rules import RuleConfig, detect

    valid = (sw.pixel_vclass() == VC_CLEAR) & np.isfinite(sw.rad4) & np.isfinite(sw.rad11) & np.isfinite(sw.lat) & np.isfinite(sw.lon)
    f = lambda a: _t.from_numpy(np.ascontiguousarray(np.nan_to_num(a, nan=1.0).astype(np.float32)))[None].to(device)
    with _t.no_grad():
        fl = detect(f(sw.rad4), f(sw.rad11), _t.from_numpy(valid)[None].to(device), _t.zeros(1, dtype=_t.bool, device=device), planck.LAMBDA["modis_4"], planck.LAMBDA["modis_11"],
                    RuleConfig(dil_modis=3))
    return fl[0].cpu().numpy() & valid


def select_cores(cells, grid, fire_xyz, radius_km):
    """Cores (row, col) whose 96x96 context window can hold a cell within radius_km of a MODIS fire-level pixel (conservative superset: core centre within radius + half core diagonal).
    The final per-cell distance filter makes the result independent of the grid."""
    if fire_xyz is None or not len(fire_xyz) or not len(cells):
        return []
    from scipy.spatial import cKDTree

    cen = np.array([grid.tile_centre(r, c) for r, c in cells])
    d, _ = cKDTree(fire_xyz).query(_unit_xyz(cen[:, 0], cen[:, 1]))
    return [rc for rc, dd in zip(cells, d) if dd <= radius_km + 0.7072 * grid.core_m / 1000.0]



def owned_mask(grid, r, c, lat, lon, built):
    """Which cells (lat, lon) predicted by tile (r, c) does that tile keep? v1.2 (margin > 0): only cells whose owner core is (r, c), so each cell is kept exactly once;
    cells whose owner core was not built (no land of the granule there / <10 % cover) are not kept (returned count). v1.1 (margin 0): owner OR owner not built."""
    cc, rr = grid.col_of(lat, lon)
    own = (rr == r) & (cc == c)
    orphan = np.array([(int(a), int(b)) not in built for a, b in zip(rr, cc)], bool)
    if grid.margin == 0:
        return own | orphan, 0
    return own, int((orphan & ~own).sum())


def build_granule_tiles(gm, gv, cells, grid, batch=16, min_cover=0.10):
    """-> list of (row, col, adapted tile dict of tensors, raw native pixel ids [96,96] = line*1354+sample, -1 none)."""
    from nightrad.data.tiles import export_names, validate_tile
    from nightrad.data.tiles_gpu import build_tiles_gpu
    from nightrad.raw.schema import adapt_v3_tile

    out, drop = [], {"low_cover": 0, "invalid": 0}
    for b in range(0, len(cells), batch):
        part = cells[b:b + batch]
        cent = [grid.tile_centre(r, c) for r, c in part]
        for (r, c), t in zip(part, build_tiles_gpu(gm, gv, cent, None, [False] * len(part))):
            if float((t["modis_pid"] >= 0).mean()) < min_cover:
                drop["low_cover"] += 1; continue
            t["satellite"] = np.array("none"); t["dt_min"] = np.float32("nan"); t["viirs_time"] = np.float32("nan"); t["viirs_utc"] = np.float64("nan")
            t["overlap"] = np.zeros_like(t["overlap"]); t["has_thermal"] = np.uint8(1); t["has_viirs"] = np.uint8(0)
            t["split"], t["holdout"] = np.array("none"), np.array("production")
            t["pair_id"] = np.array("G0"); t["episode"] = np.array("fh27")
            try:
                validate_tile(t)
            except AssertionError:
                drop["invalid"] += 1; continue
            z = adapt_v3_tile(export_names(t))
            out.append((r, c, {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in z.items()}, np.asarray(t["modis_pid"]).copy()))
    return out, drop


def process_granule(sw, bbox, model, cfg, device, gm=None, gv=None, batch=16, margin=MARGIN, shift=(0.0, 0.0), tile_rule="modis_firelevel", input_screen_T4_K=None):
    """Returns (rows dict of numpy columns, stats dict). sw: Swath from read_modis_production. Defaults = the layer settings (margin 12, tile rule modis_firelevel, batch 16:
    batch size changes the output bit-wise, keep 16)."""
    from nightrad.data.readers import Swath
    from nightrad.data.tiles_gpu import GSwath

    grid = Grid(margin, shift)
    cells = land_tiles(sw, bbox, grid)
    stats = {"land_tiles": len(cells), "margin": margin, "tile_rule": tile_rule, "input_screen_T4_K": input_screen_T4_K}
    # MODIS fire-level pixels of the scene (documented MODIS-alone radiance rule, native pixels): used for the distance column and, optionally, the tile selection
    fl = modis_firelevel_pixels(sw, device)
    fire_xyz = _unit_xyz(sw.lat[fl], sw.lon[fl]) if fl.any() else None
    stats["firelevel_pixels"] = int(fl.sum())
    if tile_rule != "all_land":
        radius_km = TILE_RULES[tile_rule] * 0.375
        cells = select_cores(cells, grid, fire_xyz, radius_km)
        stats["selected_tiles"] = len(cells)
    if not cells:
        return None, {**stats, "tiles": 0, "flagged": 0, "valid_cells": 0}
    gm = gm or GSwath(sw, device)
    nan = np.full((8, 8), np.nan, np.float32)
    z8, zf = np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.float32)
    gv = gv or GSwath(Swath(sensor="viirs", platform="none", granule="", lat=nan, lon=nan.copy(), cls=z8, frp=zf, vza=zf.copy(), sza=zf.copy(), t0=sw.t0, t1=sw.t1), device)
    tl, drop = build_granule_tiles(gm, gv, cells, grid, batch)
    stats.update(drop)
    stats["tiles"] = len(tl)
    if not tl:
        return None, {**stats, "flagged": 0, "valid_cells": 0}
    tiles = {k: torch.stack([t[k] for _, _, t, _ in tl]) for k in tl[0][2]}
    R = predict(model, cfg, tiles, device, modis_only=True, bs=batch)
    stats["valid_cells"] = int(R["valid"].sum())
    built = {(r, c) for r, c, _, _ in tl}
    L = planck.LAMBDA
    names = ("lon", "lat", "utc", "pred4_rad", "excess4_rad", "bg4_rad", "pred_T4_K", "pred_dT_K", "modis4_rad", "modis_T4_K", "vza", "modis_alone_flag", "tile_row", "tile_col",
             "modis_T11_K", "dist_modis_firelevel_km", "own_pixel_firelevel", "myd14_raw_class")
    from scipy.spatial import cKDTree

    ftree = cKDTree(fire_xyz) if fire_xyz is not None else None
    rad_km = None if tile_rule == "all_land" else TILE_RULES[tile_rule] * 0.375
    cols = {k: [] for k in names}
    for n, (r, c, t, rpid) in enumerate(tl):
        fm = R["flag_model"][n]
        if not fm.any():
            continue
        lat0, lon0 = grid.tile_centre(r, c)
        la, lo = tile_cell_lonlat(lat0, lon0)
        ii, jj = np.nonzero(fm)
        lat, lon = la[ii, jj], lo[ii, jj]
        keep, n_orphan = owned_mask(grid, r, c, lat, lon, built)
        stats["flagged_orphan_dropped"] = stats.get("flagged_orphan_dropped", 0) + n_orphan
        ii, jj, lat, lon = ii[keep], jj[keep], lat[keep], lon[keep]
        if not len(ii):
            continue
        p4, p11, m4, m11 = R["pred4"][n][ii, jj], R["pred11"][n][ii, jj], R["modis4"][n][ii, jj], R["modis11"][n][ii, jj]
        bt = lambda x, lam: planck.brightness_temperature(torch.from_numpy(np.ascontiguousarray(x)), lam).numpy()
        T4, T11 = bt(p4, L["viirs_4"]), bt(p11, L["viirs_11"])
        pid = rpid[ii, jj]; line = pid // S_STRIDE
        ok = pid >= 0; sl, ss = np.where(ok, line, 0), np.where(ok, pid % S_STRIDE, 0)
        dist = np.full(len(ii), np.nan) if ftree is None else ftree.query(_unit_xyz(lat, lon))[0]
        extra = dict(modis_T11_K=bt(m11, L["modis_11"]), dist_modis_firelevel_km=dist.astype(np.float32), own_pixel_firelevel=(fl[sl, ss] & ok).astype(np.uint8),
                     myd14_raw_class=np.where(ok, sw.raw[sl, ss], 255).astype(np.uint8) if sw.raw is not None else np.full(len(ii), 255, np.uint8))
        keep2 = np.ones(len(ii), bool)
        if rad_km is not None:  # tile rule: a cell is kept only when a MODIS fire-level pixel of the scene lies within rad_km (grid-independent; no pixel in the scene -> none kept)
            keep2 &= np.isfinite(dist) & (dist <= rad_km)
        keep2 &= (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])  # output clipped to the requested box 
        if input_screen_T4_K is not None:  # per-cell screen on the own MODIS pixel's 4 um brightness temperature
            keep2 &= bt(m4, L["modis_4"]) >= input_screen_T4_K
        vals = dict(lon=lon, lat=lat, utc=sw.t0 + 300.0 * line / LINES_PER_GRANULE, pred4_rad=p4, excess4_rad=R["exc4"][n][ii, jj], bg4_rad=R["bg4"][n][ii, jj], pred_T4_K=T4,
                    pred_dT_K=T4 - T11, modis4_rad=m4, modis_T4_K=bt(m4, L["modis_4"]), vza=t["modis_vza"].numpy()[ii, jj], modis_alone_flag=R["flag_modis"][n][ii, jj].astype(np.uint8),
                    tile_row=np.full(len(ii), r), tile_col=np.full(len(ii), c))
        vals.update(extra)
        for k in names:
            cols[k].append(vals[k][keep2])
    cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    stats["flagged"] = int(len(cols["lon"]))
    return cols, stats
