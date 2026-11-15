"""Torch (CUDA or CPU) tile builder: same algorithm and schema as `tiles.build_pair_tiles`, batched over tiles.

Design: a pair's swaths live on the device for the whole pair;
tiles are built in batches of B (ragged crops padded with NaN). Everything the CPU path does per tile -- coarse window
search, LAEA projection, pixel size, nearest-pixel assignment (with MODIS nadir preference), cloud-adjacency, static
flags, 21x21 background z-scores, local-median fill -- is done with torch ops on [B, ...] tensors. Candidate selection
(`build_pair_tiles_gpu`) replays the CPU loop exactly (same RNG stream, same acceptance rules); only the tile builds are
speculative batches. Device float64 is used wherever the CPU path uses float64.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .readers import _VC, _VC_OF_CLS, VC_BAD, VC_CLEAR, VC_NOT_COVERED, VC_OFF_SWATH, Swath
from .tiles import (FOOTPRINT_HALF, K_COVER, ALT_KM, BG_MIN_FRAC, BG_WIN, CELL_M, H, N_RANDOM, R_EARTH_M, STATIC_KM, W, Z_CLIP, _COARSE, _xyz,
                    apply_sample_weights, assemble_tile, StaticSources, _proj, shift_centres, tile_ok)

# nearest-pixel candidate box around a tile: tile half-extent + margin (must exceed any pixel's half-diagonal)
_MARGIN_M = {"modis": 8000.0, "viirs": 4000.0}
# crop radius (km) of the coarse-lattice window; MODIS keeps the CPU value, VIIRS pixels are ~5x smaller so a smaller
# window already contains the tile plus the 10-pixel background margin
_RADIUS_KM = {"modis": 80.0, "viirs": 40.0}
_ELEMS = 6e7  # max elements of one [B, cells, candidates] distance block (fp32)
# bin width (m) of the nearest-pixel search: a cell only looks at its 3x3 bins, so the width must exceed any pixel's
# half-diagonal (MODIS <= ~3 km at the scan edge, VIIRS <= ~0.7 km); candidates farther than that cannot contain the cell
_BIN_M = {"modis": 6000.0, "viirs": 1500.0}
_BIN_ELEMS = 1.2e7


def _t(a, dev):
    return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def _unit(lat, lon):
    la, lo = torch.deg2rad(lat.double()), torch.deg2rad(lon.double())
    return torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)], -1)


class GSwath:
    """A `Swath` on the device (plus the coarse lattices used for coverage tests and window search)."""

    def __init__(self, sw: Swath, dev, tensors: dict | None = None):
        """`sw` supplies metadata (and the arrays, unless `tensors` -- already-on-device arrays by field name -- is given)."""
        self.sw, self.dev = sw, torch.device(dev)
        d = self.dev
        f = (lambda k: tensors.get(k)) if tensors is not None else (lambda k: _t(getattr(sw, k), d))
        self.lat, self.lon, self.cls = f("lat"), f("lon"), f("cls")
        self.frp, self.vza, self.sza = f("frp"), f("vza"), f("sza")
        self.t4, self.t11, self.sat, self.conf = f("t4"), f("t11"), f("sat"), f("conf")
        self.raw, self.deleted, self.rad4, self.rad11, self.t4_band = f("raw"), f("deleted"), f("rad4"), f("rad11"), f("t4_band")
        lut = torch.from_numpy(_VC[sw.sensor] if self.raw is not None else _VC_OF_CLS).to(d)  # pixel validity class (readers.Swath.pixel_vclass)
        vc = lut[(self.raw if self.raw is not None else self.cls.clamp(max=6)).long()]
        if self.t4 is not None:
            vc = torch.where((vc == VC_CLEAR) & ~(torch.isfinite(self.t4) & torch.isfinite(self.t11)), torch.full_like(vc, VC_BAD), vc)
        self.vclass = vc
        self.shape = tuple(self.lat.shape)
        la, lo = self.lat[::_COARSE, ::_COARSE], self.lon[::_COARSE, ::_COARSE]
        self.c4_ok = torch.isfinite(la)
        self.c4_xyz = _unit(torch.where(self.c4_ok, la, torch.zeros_like(la)),
                            torch.where(self.c4_ok, lo, torch.zeros_like(lo)))
        self.cov_pts = self.c4_xyz[self.c4_ok]  # [P, 3] float64: the CPU `_coarse_tree` points

    def covered(self, lat, lon, km=20.0):
        """bool [n]: nearest coarse-lattice point (chord * R, like the CPU KD-tree test) within km."""
        v = _unit(torch.as_tensor(np.asarray(lat, np.float64), device=self.dev),
                  torch.as_tensor(np.asarray(lon, np.float64), device=self.dev))
        best = torch.full((len(v),), -2.0, dtype=torch.float64, device=self.dev)
        step = max(1, int(3e7 // max(len(v), 1)))
        for i in range(0, len(self.cov_pts), step):
            best = torch.maximum(best, (v @ self.cov_pts[i:i + step].T).amax(1))
        chord = torch.sqrt(torch.clamp(2 - 2 * best, min=0))
        return (chord * R_EARTH_M / 1000 <= km).cpu().numpy()


def _binned_topk(px, py, cx, cy, KC, bs, lim):
    """For each cell (cx, cy) [Q] the KC nearest candidates among px/py [B, n] (fp32, filler >= 1e8) found in the cell's 3x3
    neighbourhood of a bs-wide grid. Returns candidate column indices [B, Q, KC]; n marks 'none'."""
    B, n = px.shape
    dev = px.device
    nb = int(np.ceil(2 * lim / bs)) + 1
    NB = nb * nb
    valid = px.abs() < 1e8
    bin_ = lambda v: ((v + lim) / bs).floor().long().clamp(0, nb - 1)
    key = torch.arange(B, device=dev)[:, None] * (NB + 1) + torch.where(valid, bin_(px) * nb + bin_(py), NB)
    order = key.reshape(-1).sort().indices
    counts = torch.bincount(key.reshape(-1), minlength=B * (NB + 1)).view(B, NB + 1)
    starts = (counts.reshape(-1).cumsum(0) - counts.reshape(-1)).view(B, NB + 1)  # positions in the sorted array
    counts[:, NB] = 0  # the filler bin is never searched
    maxc = max(int(counts.max()), -(-KC // 9), 1)
    off = torch.tensor([-1, 0, 1], device=dev)
    nix, niy = bin_(cx)[:, None, None] + off[None, :, None], bin_(cy)[:, None, None] + off[None, None, :]
    nbin = torch.where((nix >= 0) & (nix < nb) & (niy >= 0) & (niy < nb), nix * nb + niy, NB).reshape(-1, 9)  # [Q, 9]
    ar = torch.arange(maxc, device=dev)
    Q = len(cx)
    out = torch.empty((B, Q, KC), dtype=torch.int64, device=dev)
    qc = max(1, int(_BIN_ELEMS // (B * 9 * maxc)))
    for i in range(0, Q, qc):
        nbq = nbin[i:i + qc]  # [q, 9]
        st, ct = starts[:, nbq], counts[:, nbq]  # [B, q, 9]
        slot = st[..., None] + ar  # [B, q, 9, maxc]
        ok = ar < ct[..., None]
        col = order[slot.clamp(max=B * n - 1)] % n
        q = nbq.shape[0]
        col, ok = col.reshape(B, q, 9 * maxc), ok.reshape(B, q, 9 * maxc)
        x = torch.gather(px, 1, col.reshape(B, -1)).reshape(B, q, -1)
        y = torch.gather(py, 1, col.reshape(B, -1)).reshape(B, q, -1)
        d2 = (cx[None, i:i + q, None] - x) ** 2 + (cy[None, i:i + q, None] - y) ** 2
        d2 = torch.where(ok, d2, torch.full_like(d2, float("inf")))
        top = d2.topk(KC, dim=-1, largest=False)
        out[:, i:i + q] = torch.where(torch.isfinite(top.values), torch.gather(col, 2, top.indices),
                                      torch.full_like(top.indices, n))
    return out


def _laea(lat, lon, lat0, lon0):
    """Spherical Lambert azimuthal equal-area (pyproj +proj=laea +R=R_EARTH_M), float64. lat0/lon0: [B,1,1] tensors."""
    phi, dl = torch.deg2rad(lat), torch.deg2rad(lon - lon0)
    p0 = torch.deg2rad(lat0)
    cphi = torch.cos(phi)
    k = torch.sqrt(2.0 / (1.0 + torch.sin(p0) * torch.sin(phi) + torch.cos(p0) * cphi * torch.cos(dl)))
    return R_EARTH_M * k * cphi * torch.sin(dl), R_EARTH_M * k * (torch.cos(p0) * torch.sin(phi) - torch.sin(p0) * cphi * torch.cos(dl))


def _windows(gs: GSwath, v0: torch.Tensor, radius_km: float):
    """Per tile (l0, l1, s0, s1) as in tiles._Window, or None if no coarse-lattice point is within radius."""
    c = _COARSE * 2
    ok, xyz = gs.c4_ok[::2, ::2], gs.c4_xyz[::2, ::2]  # == sw.lat[::8, ::8]
    d = torch.acos(torch.clamp(xyz @ v0.T, -1, 1)) * R_EARTH_M / 1000  # [Ls, Ss, B]
    hit = (d <= radius_km) & ok[..., None]
    Ls, Ss = ok.shape
    ar_r = torch.arange(Ls, device=hit.device)[:, None]
    ar_c = torch.arange(Ss, device=hit.device)[:, None]
    ar, ac = hit.any(1), hit.any(0)  # [Ls,B], [Ss,B]
    big = 1 << 30
    r = torch.stack([torch.where(ar, ar_r, big).amin(0), torch.where(ar, ar_r, -1).amax(0),
                     torch.where(ac, ar_c, big).amin(0), torch.where(ac, ar_c, -1).amax(0)]).T.tolist()
    pad, (L, S), out = c + 2, gs.shape, []
    for r0, r1, c0, c1 in r:
        if r1 < 0:
            out.append(None)
        else:
            out.append((max(r0 * c - pad, 0), min(r1 * c + pad + 1, L), max(c0 * c - pad, 0), min(c1 * c + pad + 1, S)))
    return out


def _sensor_batch(gs: GSwath, name: str, lat0, lon0, static_xy: list, radius_km=None):
    """Layers of B tiles for one sensor. lat0/lon0: float64 [B] tensors; static_xy: list of [n,2] float64 tensors/None.
    Returns (dict of [B,...] tensors, mean line index per tile [B] float64 or NaN where no cell is covered)."""
    dev, B = gs.dev, len(lat0)
    nadir = name == "modis"
    KC = K_COVER
    radius_km = radius_km or _RADIUS_KM[name]
    v0 = _unit(lat0, lon0)
    wins = _windows(gs, v0, radius_km)
    L, S = gs.shape
    hs = [(w[1] - w[0]) if w else 0 for w in wins]
    ws = [(w[3] - w[2]) if w else 0 for w in wins]
    Hm, Wm = max(max(hs), 3), max(max(ws), 3)
    l0 = torch.tensor([w[0] if w else 0 for w in wins], device=dev)
    s0 = torch.tensor([w[2] if w else 0 for w in wins], device=dev)
    nh, nw = torch.tensor(hs, device=dev), torch.tensor(ws, device=dev)
    li = l0[:, None] + torch.arange(Hm, device=dev)
    si = s0[:, None] + torch.arange(Wm, device=dev)
    inb = (torch.arange(Hm, device=dev)[None, :, None] < nh[:, None, None]) & \
          (torch.arange(Wm, device=dev)[None, None, :] < nw[:, None, None])
    gl_, gs_ = li.clamp(max=L - 1)[:, :, None].expand(B, Hm, Wm), si.clamp(max=S - 1)[:, None, :].expand(B, Hm, Wm)
    nan = float("nan")
    lat = torch.where(inb, gs.lat[gl_, gs_].double(), torch.full((), nan, dtype=torch.float64, device=dev))
    lon = torch.where(inb, gs.lon[gl_, gs_].double(), torch.full((), nan, dtype=torch.float64, device=dev))
    ok = torch.isfinite(lat) & torch.isfinite(lon)
    x, y = _laea(lat, lon, lat0.view(B, 1, 1), lon0.view(B, 1, 1))
    x, y = torch.where(ok, x, torch.full_like(x, nan)), torch.where(ok, y, torch.full_like(y, nan))
    # pixel size: along-scan = |gradient| along samples; along-track = 9-line median of line-to-line spacing
    fx, fy = torch.nan_to_num(x, nan=1e9), torch.nan_to_num(y, nan=1e9)
    gxx, gyy = torch.gradient(fx, dim=2)[0], torch.gradient(fy, dim=2)[0]
    # ragged batch: the true last column of each crop needs the one-sided difference (np.gradient edge), not the pad
    arB, arH = torch.arange(B, device=dev)[:, None], torch.arange(Hm, device=dev)[None, :]
    lc = (nw - 1).clamp(min=1)[:, None]
    gxx[arB, arH, lc] = fx[arB, arH, lc] - fx[arB, arH, lc - 1]
    gyy[arB, arH, lc] = fy[arB, arH, lc] - fy[arB, arH, lc - 1]
    gsz = torch.hypot(gxx, gyy) / 1000
    dvx, dvy = torch.diff(fx, dim=1), torch.diff(fy, dim=1)
    dl = torch.hypot(dvx, dvy) / 1000
    dl = torch.cat([dl, dl[:, -1:]], 1)
    dvx, dvy = torch.cat([dvx, dvx[:, -1:]], 1), torch.cat([dvy, dvy[:, -1:]], 1)
    # one value per line; last real line repeats the previous spacing and the rows below it replicate it (edge 'nearest')
    src = torch.where(arH >= (nh - 1)[:, None], (nh - 2).clamp(min=0)[:, None], arH)
    sx = src[:, :, None].expand(B, Hm, Wm)
    dl, dvx, dvy = torch.gather(dl, 1, sx), torch.gather(dvx, 1, sx), torch.gather(dvy, 1, sx)
    med9 = lambda a: F.pad(a[:, None], (0, 0, 4, 4), mode="replicate").unfold(2, 9, 1)[:, 0].median(-1).values
    gt = med9(dl)
    mx, my = med9(dvx), med9(dvy)  # along-track direction (9-line median of the line displacement), see tiles._Window
    nb_ = torch.clamp(torch.hypot(mx, my), min=1e-9)
    bxx, byy = mx / nb_ * gt * 1000, my / nb_ * gt * 1000
    ps = torch.stack([gsz, gt], 1).float()  # [B,2,Hm,Wm]
    ps = torch.where(ps > 50, torch.full_like(ps, nan), ps)
    ps = torch.where(ok[:, None], ps, torch.full_like(ps, nan))
    good = ok & torch.isfinite(ps[:, 0]) & torch.isfinite(ps[:, 1])

    # ---- native layers over the crop
    cls_c = torch.where(inb, gs.cls[gl_, gs_], torch.zeros((), dtype=torch.uint8, device=dev))
    deleted_c = (gs.deleted[gl_, gs_] & inb) if gs.deleted is not None else None
    cloud = cls_c == 2
    cloudadj = (F.max_pool2d(cloud[:, None].float(), 3, 1, 1)[:, 0] > 0) & ~cloud
    static = torch.zeros_like(cloud)
    if any(s is not None and len(s) for s in static_xy):
        ns = max(len(s) for s in static_xy if s is not None)
        sp = torch.full((B, ns, 2), 1e12, dtype=torch.float64, device=dev)
        for b, s in enumerate(static_xy):
            if s is not None and len(s):
                sp[b, :len(s)] = s
        thr = torch.fmax(ps[:, 0], ps[:, 1]).double()
        thr = torch.nan_to_num(torch.clamp(thr * 500.0, min=STATIC_KM * 1000), nan=0.0)  # max(1 km, half the pixel size)
        thr = torch.where(torch.isfinite(torch.fmax(ps[:, 0], ps[:, 1])), thr, torch.zeros_like(thr))
        xf, yf = x.reshape(B, -1).float(), y.reshape(B, -1).float()
        dmin = torch.empty_like(xf)
        step = max(1, int(_ELEMS // (B * ns)))
        spf = sp.float()
        for i in range(0, xf.shape[1], step):
            dx = xf[:, i:i + step, None] - spf[:, None, :, 0]
            dy = yf[:, i:i + step, None] - spf[:, None, :, 1]
            dmin[:, i:i + step] = torch.sqrt((dx * dx + dy * dy).amin(-1))
        static = (dmin.reshape(B, Hm, Wm) <= thr.float()) & ok
    layers_c = {"cloudadj": cloudadj, "static": static}
    if gs.t4 is not None:
        t4 = torch.where(inb, gs.t4[gl_, gs_].double(), torch.full((), nan, dtype=torch.float64, device=dev))
        t11 = torch.where(inb, gs.t11[gl_, gs_].double(), torch.full((), nan, dtype=torch.float64, device=dev))
        fin = torch.isfinite(t4) & torch.isfinite(t11)
        V = (cls_c == 3) & ~static & fin
        box = lambda a: F.avg_pool2d(a[:, None], BG_WIN, 1, BG_WIN // 2, count_include_pad=True)[:, 0]
        frp_ = box(V.double())
        bg_ok = frp_ >= BG_MIN_FRAC
        layers_c["bg_ok"] = bg_ok
        for nm, xx in (("t4_bgz", t4 - 290.0), ("dt_bgz", t4 - t11)):
            xv = torch.where(V, xx, torch.zeros_like(xx))
            den = torch.clamp(frp_, min=1e-9)
            m = box(xv) / den
            v = box(xv * xv) / den - m * m
            z = (xx - m) / torch.clamp(torch.sqrt(torch.clamp(v, min=0)), min=1.0)
            layers_c[nm] = torch.where(bg_ok & torch.isfinite(xx), torch.clamp(z, -Z_CLIP, Z_CLIP),
                                       torch.zeros_like(z)).float()

    # ---- truthful resampling: candidate pixels per cell (K_COVER nearest centres, deleted ones included), footprint test,
    # selection (tiles.choose_covering): valid first, then nearest / closest to nadir (MODIS)
    jj, ii = torch.meshgrid(torch.arange(W, device=dev), torch.arange(H, device=dev), indexing="xy")
    cx = ((jj - (W - 1) / 2) * CELL_M).reshape(-1)  # [Q] float64
    cy = (((H - 1) / 2 - ii) * CELL_M).reshape(-1)
    Q = H * W
    lim = (W - 1) / 2 * CELL_M + _MARGIN_M[name]
    cand = good & (x.abs() <= lim) & (y.abs() <= lim)
    cand = cand.reshape(B, -1)
    cnt = cand.sum(1)
    KS = KC + 3  # shortlist from fp32 distances, re-ranked in fp64 (fp32 ties are ~1 mm wide)
    nmax = max(int(cnt.max()), KS)
    pos = cand.cumsum(1) - 1
    bi, ni = cand.nonzero(as_tuple=True)
    idxm = torch.full((B, nmax), -1, dtype=torch.int64, device=dev)
    idxm[bi, pos[bi, ni]] = ni
    xr, yr = x.reshape(B, -1), y.reshape(B, -1)
    safe = idxm.clamp(min=0)
    px = torch.where(idxm >= 0, torch.gather(xr, 1, safe), torch.full((), 1e9, dtype=torch.float64, device=dev)).float()
    py = torch.where(idxm >= 0, torch.gather(yr, 1, safe), torch.full((), 1e9, dtype=torch.float64, device=dev)).float()
    sel = _binned_topk(px, py, cx.float(), cy.float(), KS, _BIN_M[name], lim)  # candidate column or nmax (= none)
    idxm = torch.cat([idxm, torch.full((B, 1), -1, dtype=torch.int64, device=dev)], 1)
    flat = torch.gather(idxm, 1, sel.reshape(B, -1)).reshape(B, Q, KS)  # crop flat index, -1 = none
    fl0 = flat.clamp(min=0).reshape(B, -1)
    g = lambda a, k=KS: torch.gather(a, 1, fl0).reshape(B, Q, k)
    d64 = torch.hypot(cx[None, :, None] - g(xr), cy[None, :, None] - g(yr))
    d64 = torch.where(flat >= 0, d64, torch.full_like(d64, float("inf")))
    order = d64.argsort(dim=-1, stable=True)[..., :KC]
    flat = torch.gather(flat, 2, order)
    has = flat >= 0
    fl = flat.clamp(min=0)
    gk = lambda a: torch.gather(a.reshape(B, -1), 1, fl.reshape(B, -1)).reshape(B, Q, KC)  # crop-level map -> [B,Q,KC]
    dist = torch.hypot(cx[None, :, None] - gk(x), cy[None, :, None] - gk(y))
    dx, dy = cx[None, :, None] - gk(x), cy[None, :, None] - gk(y)
    ax_, ay_, bx_, by_ = gk(gxx), gk(gyy), gk(bxx), gk(byy)
    det = ax_ * by_ - ay_ * bx_
    al, be = (dx * by_ - dy * bx_) / det, (ax_ * dy - ay_ * dx) / det
    inside_k = has & (al.abs() <= FOOTPRINT_HALF) & (be.abs() <= FOOTPRINT_HALF)
    dele = gk(deleted_c) if deleted_c is not None else torch.zeros_like(inside_k)
    real = inside_k & ~dele
    gl_k = (l0[:, None, None] + fl // Wm).clamp(max=L - 1)
    gs_k = (s0[:, None, None] + fl % Wm).clamp(max=S - 1)
    valid_k = gs.vclass[gl_k, gs_k] == VC_CLEAR
    vz = torch.nan_to_num(gs.vza[gl_k, gs_k].double(), nan=1e3)
    score = (~valid_k).double() * 1e12 + (vz * 1e6 if nadir else 0.0) + torch.round(dist * 1000) / 1000
    score = torch.where(real, score, torch.full_like(score, float("inf")))
    tie = real & (score == score.amin(-1, keepdim=True))  # exact ties -> smaller pixel id (tiles.choose_covering)
    pidk = gl_k.long() * S + gs_k
    pick = torch.where(tie, pidk, torch.full_like(pidk, torch.iinfo(torch.int64).max)).argmin(-1)
    pk = pick[..., None]
    ins = real.any(-1)  # [B,Q]
    flp = torch.gather(fl, 2, pk)[..., 0]
    gl = (l0[:, None] + flp // Wm).clamp(max=L - 1)
    gsm = (s0[:, None] + flp % Wm).clamp(max=S - 1)
    zero = lambda dt: torch.zeros((), dtype=dt, device=dev)
    psf = ps.reshape(B, 2, -1)
    ps_sel = torch.stack([torch.gather(psf[:, a], 1, flp) for a in range(2)], 1)  # [B,2,Q]

    def cell(a):  # native-swath array -> [B,Q]
        return a[gl, gsm]

    def crop(a):  # crop-level map -> [B,Q]
        return torch.gather(a.reshape(B, -1), 1, flp)

    o = {}
    vcl = torch.where(ins, cell(gs.vclass), torch.where(inside_k.any(-1), torch.full((), VC_NOT_COVERED, dtype=torch.uint8, device=dev),
                                                          torch.full((), VC_OFF_SWATH, dtype=torch.uint8, device=dev)))
    o["vclass"] = vcl
    o["ncover"] = real.sum(-1).to(torch.uint8)
    o["raw"] = torch.where(ins, cell(gs.raw), zero(torch.uint8)) if gs.raw is not None else torch.zeros((B, Q), dtype=torch.uint8, device=dev)
    cls = torch.where(ins, cell(gs.cls), zero(torch.uint8))
    o["cls"] = cls
    o["frp"] = torch.where(cls >= 4, torch.log1p(cell(gs.frp)), zero(torch.float32))
    vza = torch.nan_to_num(cell(gs.vza))
    o["vza"] = torch.where(ins, vza, zero(torch.float32))
    o["sza"] = torch.where(ins, torch.nan_to_num(cell(gs.sza)), zero(torch.float32))
    r = R_EARTH_M / 1000
    sa = torch.rad2deg(torch.asin(torch.clamp(r / (r + ALT_KM[name]) * torch.sin(torch.deg2rad(vza)), -1, 1)))
    sa = torch.where(gsm < (S - 1) / 2, -sa, sa)
    o["scan"] = torch.where(ins, sa, zero(torch.float32))
    o["psize"] = torch.where(ins[:, None], ps_sel, zero(torch.float32))
    o["pid"] = torch.where(ins, (gl * S + gsm).int(), torch.full((), -1, dtype=torch.int32, device=dev))
    o["conf"] = torch.where(ins & (cls >= 4), cell(gs.conf), zero(torch.float32)) if gs.conf is not None \
        else torch.zeros((B, Q), dtype=torch.float32, device=dev)
    o["cloudadj"] = (ins & crop(layers_c["cloudadj"])).to(torch.uint8)
    o["static"] = (ins & crop(layers_c["static"])).to(torch.uint8)
    z8 = lambda: torch.zeros((B, Q), dtype=torch.uint8, device=dev)
    zf = lambda: torch.zeros((B, Q), dtype=torch.float32, device=dev)
    o.update(t4=zf(), t11=zf(), t4_missing=z8(), sat=z8(), t4_bgz=zf(), dt_bgz=zf(), bg_ok=z8(), rad4=zf(), rad11=zf(), t4_band=z8())
    if gs.t4 is not None:
        t4, t11 = cell(gs.t4), cell(gs.t11)
        ok4, ok11 = ins & torch.isfinite(t4), ins & torch.isfinite(t11)  # each channel copied as is; missing stays 0 + flagged, never filled
        o["t4"], o["t11"] = torch.where(ok4, t4, zero(torch.float32)), torch.where(ok11, t11, zero(torch.float32))
        o["t4_missing"] = (ins & ~(ok4 & ok11)).to(torch.uint8)
        o["sat"] = (ins & cell(gs.sat)).to(torch.uint8)
        o["bg_ok"] = (ins & crop(layers_c["bg_ok"])).to(torch.uint8)
        o["t4_bgz"] = torch.where(ins, crop(layers_c["t4_bgz"]), zero(torch.float32))
        o["dt_bgz"] = torch.where(ins, crop(layers_c["dt_bgz"]), zero(torch.float32))
        if gs.rad4 is not None:
            o["rad4"] = torch.where(ok4, torch.nan_to_num(cell(gs.rad4)), zero(torch.float32))
            o["rad11"] = torch.where(ok11, torch.nan_to_num(cell(gs.rad11)), zero(torch.float32))
        if gs.t4_band is not None:
            o["t4_band"] = torch.where(ok4, cell(gs.t4_band), zero(torch.uint8))
    # empty-window tiles: no cell can be inside (cand empty) -> fillers had ins False already
    n_in = ins.sum(1)
    mean_line = torch.where(n_in > 0, (gl.double() * ins).sum(1) / n_in.clamp(min=1), torch.full((B,), nan, dtype=torch.float64, device=dev))
    shp = {"psize": (B, 2, H, W)}
    o = {k: v.reshape(shp.get(k, (B, H, W))) for k, v in o.items()}
    return o, mean_line


def build_tiles_gpu(gm: GSwath, gv: GSwath, centres, static: StaticSources | None = None, random_flags=None,
                    radius_km: dict | None = None) -> list[dict]:
    """Schema tiles for a list of (lat0, lon0) centres (one batch). Numpy dicts, same as `tiles.build_tile`."""
    B = len(centres)
    dev = gm.dev
    lat0 = torch.tensor([c[0] for c in centres], dtype=torch.float64, device=dev)
    lon0 = torch.tensor([c[1] for c in centres], dtype=torch.float64, device=dev)
    sxy = [None] * B
    if static is not None and static.n:
        for b, (la, lo) in enumerate(centres):
            s = static.projected_near(_proj(la, lo), la, lo)
            sxy[b] = torch.as_tensor(s, dtype=torch.float64, device=dev) if len(s) else None
    res = {}
    for name, g in (("modis", gm), ("viirs", gv)):
        o, ml = _sensor_batch(g, name, lat0, lon0, sxy, (radius_km or {}).get(name))
        res[name] = ({k: v.cpu().numpy() for k, v in o.items()}, ml.cpu().numpy())
    out = []
    for b, (la, lo) in enumerate(centres):
        layers_by, times = {}, {}
        for name, g in (("modis", gm), ("viirs", gv)):
            o, ml = res[name]
            layers_by[name] = {k: v[b] for k, v in o.items()}
            sw = g.sw
            times[name] = None if np.isnan(ml[b]) else float(sw.t0 + (sw.t1 - sw.t0) * (ml[b] + 0.5) / sw.shape[0])
        out.append(assemble_tile(gm.sw, gv.sw, layers_by, times, float(la), float(lo),
                                 bool(random_flags[b]) if random_flags is not None else False))
    return out


def build_pair_tiles_gpu(modis: Swath, viirs: Swath, *, device="cuda", batch=16, max_fire_tiles=40, n_random=N_RANDOM,
                         min_overlap=0.5, min_sep_km=24.0, seed=0, max_attempts=None,
                         static: StaticSources | None = None, gm: GSwath | None = None, gv: GSwath | None = None,
                         radius_km: dict | None = None, centre_filter=None) -> list[dict]:
    """GPU counterpart of `tiles.build_pair_tiles` with the same selection logic and RNG stream.
    centre_filter: optional f(lat, lon arrays) -> bool keep-mask applied to the fire-centred candidates AFTER the
    random shift, so the 40-tile cap is spent only where tiles may be used (the RNG stream is unchanged; None = old behaviour)."""
    gm = gm or GSwath(modis, device)
    gv = gv or GSwath(viirs, device)
    rng = np.random.default_rng(seed)
    cand = []
    for g in (gm, gv):
        idx = (g.cls >= 4).nonzero()
        if len(idx):
            cand.append(torch.stack([g.lat[idx[:, 0], idx[:, 1]], g.lon[idx[:, 0], idx[:, 1]]], 1).cpu().numpy())
    cand = np.concatenate(cand) if cand else np.zeros((0, 2), np.float32)
    cand = _finite_rows(cand)
    cand = cand[rng.permutation(len(cand))]
    cand, offs = shift_centres(cand, rng)  # D4 (same RNG stream as the CPU path)
    if centre_filter is not None and len(cand):
        keep = np.asarray(centre_filter(cand[:, 0], cand[:, 1]), bool)
        cand, offs = cand[keep], offs[keep]
    cov_cache: dict[int, tuple[bool, bool]] = {}
    CB = 256

    def covered(i):
        if i not in cov_cache:
            j0 = (i // CB) * CB
            blk = cand[j0:j0 + CB]
            cm, cv = gm.covered(blk[:, 0], blk[:, 1]), gv.covered(blk[:, 0], blk[:, 1])
            for k in range(len(blk)):
                cov_cache[j0 + k] = (bool(cm[k]), bool(cv[k]))
        return cov_cache[i]

    cache: dict[int, dict] = {}
    tiles, chosen, tries = [], [], 0
    max_attempts = max_attempts or 4 * max_fire_tiles

    def sep_ok(v):
        return not (chosen and (np.arccos(np.clip(np.array(chosen) @ v, -1, 1)) * R_EARTH_M / 1000 < min_sep_km).any())

    for i, (lat, lon) in enumerate(cand):
        if len(tiles) >= max_fire_tiles or tries >= max_attempts:
            break
        v = _xyz(lat, lon)
        if not sep_ok(v):
            continue
        if not all(covered(i)):
            continue
        tries += 1
        if i not in cache:  # speculative batch: the next candidates that would pass the cheap tests right now
            todo = [i]
            for j in range(i + 1, min(len(cand), i + 1 + 8 * batch)):
                if len(todo) >= batch:
                    break
                if sep_ok(_xyz(*cand[j])) and all(covered(j)) and j not in cache:
                    todo.append(j)
            for j, t in zip(todo, build_tiles_gpu(gm, gv, [(float(cand[j][0]), float(cand[j][1])) for j in todo],
                                                  static, radius_km=radius_km)):
                cache[j] = t
        t = cache.pop(i)
        if not tile_ok(t, min_overlap):
            continue
        t["fire_offset_km"] = offs[i]
        chosen.append(v)
        tiles.append(t)
    # random tiles: the CPU loop draws (line, sample) pairs sequentially from the same RNG; pre-draw the same stream
    n_try = 30 * max(n_random, 1)
    draws = np.array([(rng.integers(viirs.shape[0]), rng.integers(viirs.shape[1])) for _ in range(n_try)]) \
        if n_random > 0 else np.zeros((0, 2), int)
    got, k = 0, 0
    if len(draws) and n_random > 0:
        li, si = torch.as_tensor(draws[:, 0], device=gv.dev), torch.as_tensor(draws[:, 1], device=gv.dev)
        rl = gv.lat[li, si].cpu().numpy()
        ro = gv.lon[li, si].cpu().numpy()
        fin = np.isfinite(rl)
        cm = np.zeros(len(draws), bool)
        cm[fin] = gm.covered(rl[fin], ro[fin])
        rcache: dict[int, dict] = {}
        while got < n_random and k < n_try:
            i, k = k, k + 1
            if not (fin[i] and cm[i]):
                continue
            if i not in rcache:
                todo = [i] + [j for j in range(i + 1, min(n_try, i + 1 + 4 * batch)) if fin[j] and cm[j]][:batch - 1]
                for j, t in zip(todo, build_tiles_gpu(gm, gv, [(float(rl[j]), float(ro[j])) for j in todo], static,
                                                      random_flags=[True] * len(todo), radius_km=radius_km)):
                    rcache[j] = t
            t = rcache.pop(i)
            if not tile_ok(t, min_overlap):
                continue
            tiles.append(t)
            got += 1
    apply_sample_weights(tiles)
    return tiles


def _finite_rows(a):
    return a[np.isfinite(a).all(1)] if len(a) else a
